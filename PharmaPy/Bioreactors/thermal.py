"""JSON adapters for the native vessel energy balance; no separate heat solver."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
from typing import Optional

import numpy as np
from PharmaPy.Commons import temp_ufun
from PharmaPy.Containers import Mixer
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController, ContinuousVesselController

from PharmaPy.DataClasses import StateKey
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Utilities import CoolingWater
from .culture import _evaluate, _references
from .BioreactorUnitConverter import BioreactorUnitConverter


# Validate thermal state at the native absolute-temperature boundary.
def positive_temperature(value: float, name: str = "temperature_k") -> float:
    """Validate an absolute temperature.

    Parameters
    ----------
    value : float
        Absolute temperature [K].
    name : str, optional
        Input label for errors.

    Returns
    -------
    float
        Finite, positive temperature [K].

    Raises
    ------
    ValueError
        The temperature is nonfinite or not positive.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        temperature = float(value)  # [K]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return temperature


# Use the existing native mechanism heat-generation hook.
class ThermalBiology(Mechanism):
    """Use the existing mechanism heat hook with accepted species sources.

    Heat expressions return power [W], positive into the liquid. Their
    ``source_rate`` leaves read signed native species sources [kg/s], after
    biological optimization/reconciliation, excluding physical inlet/outlet
    flow. Coefficients and their temperature dependence are supplied by users;
    no organism-specific heat yield or Arrhenius constant is assumed.
    """

    def thermal_temperature(self, phase: LiquidPhase) -> float:
        """Validate the live thermal state before evaluating biological rates.

        Parameters
        ----------
        phase : LiquidPhase
            Live liquid temperature [K] and specific heat [J/(kg K)].

        Returns
        -------
        float
            Positive live temperature [K].

        Raises
        ------
        ValueError
            Temperature is invalid, or the non-isothermal liquid has
            nonfinite/nonpositive heat capacity at the evaluated state.
        """
        temperature = positive_temperature(phase.temp)  # [K]
        if hasattr(self, "thermal_expression"):
            capacity = float(phase.getCp())  # [J/(kg K)]
            if not np.isfinite(capacity) or capacity <= 0.0:
                raise ValueError("non-isothermal liquid heat capacity must be finite and positive")
        return temperature

    def get_heat_generation(self, aux: Mapping, completed_state: Mapping, time: float) -> float:
        """Evaluate the configured heat expression without reevaluating kinetics.

        Parameters
        ----------
        aux : Mapping
            This evaluation's ``source_rates_kg_s`` mapping [kg/s].
        completed_state : Mapping
            Native state containing ``global_temp`` [K].
        time : float
            Absolute time [s].

        Returns
        -------
        float
            Net biological heat generation [W]; zero for legacy inputs.

        Raises
        ------
        ValueError
            The evaluated temperature is invalid or heat is nonfinite.
        """
        if not hasattr(self, "thermal_expression"):
            return 0.0
        snapshot = SimpleNamespace(  # temperature [K], time [s], species sources [kg/s]
            temperature_k=positive_temperature(completed_state[StateKey("global_temp")]),
            time_s=float(time), source_rates_kg_s=aux["source_rates_kg_s"])
        heat = _evaluate(self.thermal_expression, {}, self.thermal_definition,  # [W]
                         snapshot, self.conditions)
        if not np.isfinite(heat):
            raise ValueError("thermal.heat_generation_w must evaluate to finite power [W]")
        return float(heat)


# Reuse native utility construction and native sensible-enthalpy mixing.
def attach_utility(unit: MultiPhaseVessel, declaration: Optional[Mapping]) -> None:
    """Attach the existing constant-inlet cooling-water utility when requested.

    Parameters
    ----------
    unit : MultiPhaseVessel
        Native reactor with configured vessel heat-transfer coefficient.
    declaration : Mapping or None
        Native ``CoolingWater`` arguments: ``temp_in`` [K], ``h_conv``
        [W/(m2 K)] and ``mass_flow`` [kg/s] or ``vol_flow`` [m3/s]. None
        leaves the vessel without a utility.
    """
    if declaration is not None:
        unit.Utility = CoolingWater(**declaration)


def mixed_temperature(phase: LiquidPhase, initial_temperature: float, added_mass: np.ndarray,
                      feed_temperature: float) -> float:
    """Conserve sensible enthalpy across an instantaneous liquid addition.

    Parameters
    ----------
    phase : LiquidPhase
        Liquid after adding mass, with unchanged temperature and native
        ideal-mixture sensible enthalpies. No heat of mixing is assumed.
    added_mass : ndarray
        Species masses added by the event [kg].
    initial_temperature, feed_temperature : float
        Initial and feed temperatures [K].

    Returns
    -------
    float
        Adiabatic mixed temperature [K], from the unchanged native Mixer
        and its integrated heat-capacity correlations.

    Raises
    ------
    ValueError
        The native mixed temperature is nonfinite or nonpositive.
    RuntimeError
        The native Mixer enthalpy inversion does not converge.
    """
    added_total = float(np.sum(added_mass))  # [kg]
    if added_total == 0.0 or initial_temperature == feed_temperature:
        return initial_temperature
    initial_mass = np.asarray(phase.mass_j) - added_mass  # [kg]
    initial_total = float(initial_mass.sum())  # [kg]
    charge = LiquidPhase(phase.path_data, mass=initial_total,
                         mass_frac=initial_mass / initial_total,
                         temp=initial_temperature, verbose=False)
    feed = LiquidPhase(phase.path_data, mass=added_total,
                       mass_frac=added_mass / added_total,
                       temp=feed_temperature, verbose=False)
    mixer = Mixer()
    mixer.Inlets = [charge, feed]
    _, _, temperature = mixer.solve_unit()  # [K]
    return positive_temperature(temperature)


# Translate the optional thermal declaration into native constructor settings.
def thermal_options(operation: Mapping, definition: Mapping, phase: LiquidPhase,
                    mechanism: ThermalBiology) -> tuple[dict, Optional[Mapping]]:
    """Translate optional thermal JSON into native reactor constructor options.

    Parameters
    ----------
    operation : Mapping
        Existing operation section, optionally containing ``thermal``:
        heat_generation_w [W], native heat-transfer coefficients [W/(m2 K)],
        diameter [m], base area [m2], utility inlet temperature [K] and flow
        [kg/s] or [m3/s]. A prescribed profile uses temp_zero [K], rate [K/s]
        and optional t_zero [s].
    definition : Mapping
        Normalized mechanism; supplies parameters and constants in their
        user-declared units.
    phase : LiquidPhase
        Initial liquid; thermodynamic heat capacities must be available and
        positive when integrating an energy balance [J/(kg K)]. Prescribed
        temperature follows native control semantics and does not evaluate Cp.
    mechanism : ThermalBiology
        Biological mechanism to receive the heat expression.

    Returns
    -------
    tuple of dict and dict or None
        Native constructor keyword arguments and optional utility declaration.

    Raises
    ------
    ValueError
        Thermal mode, fields, references, coefficients or heat capacity are
        invalid. Energy-balanced modes require an explicit heat expression,
        including an explicit zero for a deliberately nonreacting liquid.
    MissingPropertyError
        Required heat-capacity data are absent.
    """
    thermal = operation.get("thermal")
    if thermal is None:
        return {"isothermal": True}, None
    if not isinstance(thermal, Mapping):
        raise ValueError("operation.thermal must be a mapping")
    mode = thermal.get("mode")
    if not isinstance(mode, str):
        raise ValueError("thermal.mode must name a supported thermodynamic process")
    if mode == "isothermal":
        if set(thermal) != {"mode"}:
            raise ValueError("isothermal thermal configuration requires only mode")
        return {"isothermal": True}, None
    if mode == "prescribed-temperature":
        if set(thermal) != {"mode", "temperature_profile"}:
            raise ValueError("prescribed-temperature requires mode and temperature_profile")
        profile = thermal["temperature_profile"]
        if (not isinstance(profile, Mapping) or not {"temp_zero", "rate"} <= set(profile)
                or set(profile) - {"temp_zero", "rate", "t_zero"}):
            raise ValueError("temperature_profile requires native temp_zero [K], rate [K/s] and optional t_zero [s]")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not np.isfinite(value) for value in profile.values()):
            raise ValueError("temperature_profile values must be finite real numbers")
        temperature = partial(temp_ufun, **dict(profile))
        initial = positive_temperature(temperature(0.0))  # [K]
        runtime = operation["runtime"]
        final_time = BioreactorUnitConverter.convert(runtime["value"], runtime["unit"], "s")  # [s]
        positive_temperature(temperature(final_time))
        # Eight ulps, as for recipe-clock normalization, allow only roundoff
        # from the native ramp's multiply/add operations, not a physical offset.
        if not np.isclose(phase.temp, initial, rtol=0.0,
                          atol=8.0 * np.finfo(float).eps * max(1.0, abs(initial))):
            raise ValueError("operation.temperature_k must match temperature_profile at time zero")
        controller_type = (ContinuousVesselController if operation["mode"] == "continuous"
                           else SimpleTemperatureController)
        return {"isothermal": False, "controller": controller_type(temperature)}, None
    if mode not in {"adiabatic", "heat-transfer"}:
        raise ValueError("thermal.mode must be isothermal, adiabatic, heat-transfer or prescribed-temperature")
    required = {"mode", "heat_generation_w"}
    if mode == "heat-transfer":
        required.add("heat_transfer")
    if set(thermal) != required:
        raise ValueError(f"thermal {mode} requires exactly {sorted(required)}")
    capacity = float(phase.getCp())  # [J/(kg K)]
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("non-isothermal liquid heat capacity must be finite and positive")
    expression = thermal["heat_generation_w"]
    if (not isinstance(expression, (int, float, Mapping)) or isinstance(expression, bool)
            or (isinstance(expression, (int, float)) and not np.isfinite(expression))):
        raise ValueError("thermal.heat_generation_w requires a finite power or expression [W]")
    unknown = _references(expression, "source_rate") - set(phase.name_species)
    states = _references(expression, "state") - {"temperature_k", "time_s"}
    if (unknown or states or _references(expression) or _references(expression, "accepted_flux")
            or _references(expression, "concentration")):
        raise ValueError("thermal heat references must use known source_rate species, temperature_k or time_s")
    model = definition["model"]
    parameters = definition.get("parameters", model.get("parameters", {}))
    constants = model.get("constants") or {}
    if (_references(expression, "parameter") - set(parameters)
            or _references(expression, "constant") - set(constants)):
        raise ValueError("thermal heat references unknown parameters or constants")
    mechanism.thermal_expression = deepcopy(expression)
    mechanism.thermal_definition = SimpleNamespace(parameters=parameters, constants=constants)
    options = {"isothermal": False, "adiabatic": mode == "adiabatic"}
    utility = thermal.get("heat_transfer")
    if mode == "heat-transfer":
        legacy_fields = {"vessel_coefficient_w_m2_k", "utility_coefficient_w_m2_k",
                         "diameter_m", "base_area_m2", "utility_temperature_k",
                         "utility_mass_flow_kg_s"}
        # Accept the earlier local declaration without mixing spelling schemes.
        if isinstance(utility, Mapping) and set(utility) == legacy_fields:
            utility = {"h_conv": utility["vessel_coefficient_w_m2_k"],
                       "diam": utility["diameter_m"], "area_base": utility["base_area_m2"],
                       "utility": {"h_conv": utility["utility_coefficient_w_m2_k"],
                                   "temp_in": utility["utility_temperature_k"],
                                   "mass_flow": utility["utility_mass_flow_kg_s"]}}
        if (not isinstance(utility, Mapping) or not {"h_conv", "diam", "utility"} <= set(utility)
                or set(utility) - {"h_conv", "diam", "area_base", "ht_mode", "utility"}):
            raise ValueError("heat_transfer requires native h_conv, diam, utility; optional area_base and ht_mode")
        if utility.get("ht_mode", "jacket") not in {"jacket", "bath"}:
            raise ValueError("heat_transfer.ht_mode must be jacket or bath; native coil geometry is not implemented")
        coolant = utility["utility"]
        if (not isinstance(coolant, Mapping) or not {"temp_in", "h_conv"} <= set(coolant)
                or set(coolant) - {"temp_in", "h_conv", "mass_flow", "vol_flow"}
                or len(set(coolant) & {"mass_flow", "vol_flow"}) != 1):
            raise ValueError("utility requires native temp_in, h_conv and exactly one of mass_flow or vol_flow")
        numeric = {key: value for key, value in utility.items() if key not in {"utility", "ht_mode"}}
        numeric.update({f"utility.{key}": value for key, value in coolant.items()})
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"heat_transfer.{name} must be a finite real number")
            if not np.isfinite(value) or value < 0.0 or (name != "area_base" and value == 0.0):
                raise ValueError(f"heat_transfer.{name} must be finite and positive (base area may be zero)")
        options.update({name: value for name, value in utility.items() if name != "utility"})
        utility = dict(coolant)
    return options, utility
