"""CoolingWater flow-basis reconciliation for dynamic controls (#279).

A ``DynamicInput`` controlling either ``mass_flow`` [kg/s] or ``vol_flow``
[m**3/s] must update the other basis through the constant utility density,
because jacket balances read the volume flow. Coverage includes scalar and
profile (array) time evaluation, unchanged static behavior, the rejection of
controls on both bases, minimal hand-written controllers that implement only
the documented duck-typed ``evaluate_inputs(time)`` protocol, and the handoff
into a real ``BatchReactor`` jacket balance evaluated through ``unit_model``
with the shipped PFR property data.
No solver backend is required: the reactor right-hand side is evaluated
directly.

Refs:
https://github.com/PharmaPy-org/PharmaPy/issues/279
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Reactors import BatchReactor
from PharmaPy.Utilities import CoolingWater


THERMO_PATH = str(
    Path(__file__).parent / "integration/data/pfr_test_pure_comp.json")
# CoolingWater's constant-property assumption for liquid water near ambient
# conditions; restated here so expectations do not read the production value.
WATER_DENSITY = 1000.0  # [kg/m**3]
STATIC_MASS_FLOW = 0.1  # [kg/s], shipped PFR utility flow
STATIC_VOL_FLOW = 1.0e-4  # [m**3/s], STATIC_MASS_FLOW / WATER_DENSITY
CONTROLLED_MASS_FLOW = 0.2  # [kg/s], twice the static flow to make it visible
CONTROLLED_VOL_FLOW = 2.0e-4  # [m**3/s], CONTROLLED_MASS_FLOW / WATER_DENSITY
UTILITY_TEMPERATURE = 300.0  # [K], shipped PFR utility inlet
CONTROLLED_TEMPERATURE = 290.0  # [K], distinct from the static inlet
MASS_FLOW_RAMP = 0.01  # [kg/s**2], synthetic linear profile slope
PROFILE_TIMES = np.array([0.0, 10.0, 25.0])  # [s], unequally spaced samples
# Synthetic A + B -> C mixture with inert solvent (as in
# test_reactor_correctness.py); not a calibrated case.
CONCENTRATIONS = np.array([0.15, 0.10, 0.02, 5.0])  # [mol/L]
LIQUID_TEMPERATURE = 320.0  # [K], above the jacket so heat flows outward
JACKET_TEMPERATURE = 305.0  # [K], differs from the coolant inlet temperature
LIQUID_VOLUME = 0.002  # [m**3], shipped PFR case volume
RATE_CONSTANT = 1.0e-3  # [L/mol/s], slow elementary bimolecular reaction
REACTION_HEAT = -1.0e4  # [J/mol of reaction as written], synthetic exotherm
# BatchReactor's documented jacket-volume modeling assumption: the jacket
# holds 15 % of the liquid volume (BatchReactor.energy_balances).
JACKET_VOLUME_RATIO = 0.15  # [-]
# Flow conversions and balance arithmetic differ only by floating-point
# roundoff of a few operations on O(1) quantities.
ALGEBRA_RTOL = 1.0e-12  # [-]


def _controlled_water(constructor_flow, **controls):
    """Build cooling water with a DynamicInput holding the given controls.

    Parameters
    ----------
    constructor_flow : dict
        Static flow keyword for ``CoolingWater``: ``{'mass_flow': [kg/s]}``
        or ``{'vol_flow': [m**3/s]}``.
    **controls : callable
        Control callables keyed by controlled field, mapping time [s] to the
        field's units ([kg/s], [m**3/s] or [K]).

    Returns
    -------
    CoolingWater
        Utility with inlet temperature ``UTILITY_TEMPERATURE`` [K] and the
        controls attached, or no ``DynamicInlet`` when ``controls`` is empty.
    """
    water = CoolingWater(temp_in=UTILITY_TEMPERATURE, **constructor_flow)
    if controls:
        control = DynamicInput()
        for name, function in controls.items():
            control.add_variable(name, function)
        water.DynamicInlet = control
    return water


@pytest.mark.unit
@pytest.mark.parametrize("constructor_flow", [
    {"mass_flow": STATIC_MASS_FLOW}, {"vol_flow": STATIC_VOL_FLOW}])
def test_mass_flow_control_updates_volume_flow(constructor_flow):
    water = _controlled_water(
        constructor_flow, mass_flow=lambda time: CONTROLLED_MASS_FLOW)
    inputs = water.get_inputs(0.0)
    assert inputs["mass_flow"] == pytest.approx(
        CONTROLLED_MASS_FLOW, rel=ALGEBRA_RTOL)
    assert inputs["vol_flow"] == pytest.approx(
        CONTROLLED_VOL_FLOW, rel=ALGEBRA_RTOL)
    assert inputs["temp_in"] == pytest.approx(
        UTILITY_TEMPERATURE, rel=ALGEBRA_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("constructor_flow", [
    {"mass_flow": STATIC_MASS_FLOW}, {"vol_flow": STATIC_VOL_FLOW}])
def test_volume_flow_control_updates_mass_flow(constructor_flow):
    water = _controlled_water(
        constructor_flow, vol_flow=lambda time: CONTROLLED_VOL_FLOW)
    inputs = water.get_inputs(0.0)
    assert inputs["vol_flow"] == pytest.approx(
        CONTROLLED_VOL_FLOW, rel=ALGEBRA_RTOL)
    assert inputs["mass_flow"] == pytest.approx(
        CONTROLLED_MASS_FLOW, rel=ALGEBRA_RTOL)


@pytest.mark.unit
def test_mass_flow_profile_converts_each_time():
    water = _controlled_water(
        {"mass_flow": STATIC_MASS_FLOW},
        mass_flow=lambda time: STATIC_MASS_FLOW + MASS_FLOW_RAMP * time)
    inputs = water.get_inputs(PROFILE_TIMES)
    # 0.1 + 0.01 * t at t = 0, 10, 25 s.
    expected_mass = np.array([0.1, 0.2, 0.35])  # [kg/s]
    expected_vol = np.array([1.0e-4, 2.0e-4, 3.5e-4])  # [m**3/s]
    assert inputs["vol_flow"].shape == PROFILE_TIMES.shape
    np.testing.assert_allclose(inputs["mass_flow"], expected_mass,
                               rtol=ALGEBRA_RTOL, atol=0)
    np.testing.assert_allclose(inputs["vol_flow"], expected_vol,
                               rtol=ALGEBRA_RTOL, atol=0)
    np.testing.assert_allclose(
        inputs["temp_in"], np.full(PROFILE_TIMES.shape, UTILITY_TEMPERATURE),
        rtol=ALGEBRA_RTOL, atol=0)


@pytest.mark.unit
def test_volume_flow_profile_converts_each_time():
    def volume_flow(time):
        """Return a linear volume-flow profile [m**3/s] at time [s]."""
        return STATIC_VOL_FLOW + MASS_FLOW_RAMP / WATER_DENSITY * time

    water = _controlled_water({"vol_flow": STATIC_VOL_FLOW},
                              vol_flow=volume_flow)
    inputs = water.get_inputs(PROFILE_TIMES)
    expected_mass = np.array([0.1, 0.2, 0.35])  # [kg/s]
    assert inputs["mass_flow"].shape == PROFILE_TIMES.shape
    np.testing.assert_allclose(inputs["mass_flow"], expected_mass,
                               rtol=ALGEBRA_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("constructor_flow", [
    {"mass_flow": STATIC_MASS_FLOW}, {"vol_flow": STATIC_VOL_FLOW}])
def test_static_inputs_are_unchanged(constructor_flow):
    water = _controlled_water(constructor_flow)
    inputs = water.get_inputs(0.0)
    assert inputs == {"vol_flow": water.vol_flow, "temp_in": water.temp_in,
                      "mass_flow": water.mass_flow}
    assert inputs["vol_flow"] == pytest.approx(STATIC_VOL_FLOW,
                                               rel=ALGEBRA_RTOL)
    assert inputs["mass_flow"] == pytest.approx(STATIC_MASS_FLOW,
                                                rel=ALGEBRA_RTOL)

    profile = water.get_inputs(PROFILE_TIMES)
    for name in ("vol_flow", "temp_in", "mass_flow"):
        np.testing.assert_array_equal(
            profile[name], np.full(PROFILE_TIMES.shape, inputs[name]))


@pytest.mark.unit
def test_temperature_control_keeps_static_flows():
    water = _controlled_water({"mass_flow": STATIC_MASS_FLOW},
                              temp_in=lambda time: CONTROLLED_TEMPERATURE)
    inputs = water.get_inputs(0.0)
    assert inputs["temp_in"] == pytest.approx(CONTROLLED_TEMPERATURE,
                                              rel=ALGEBRA_RTOL)
    assert inputs["mass_flow"] == pytest.approx(STATIC_MASS_FLOW,
                                                rel=ALGEBRA_RTOL)
    assert inputs["vol_flow"] == pytest.approx(STATIC_VOL_FLOW,
                                               rel=ALGEBRA_RTOL)


@pytest.mark.unit
def test_controls_on_both_flow_bases_are_rejected():
    water = _controlled_water(
        {"mass_flow": STATIC_MASS_FLOW},
        mass_flow=lambda time: CONTROLLED_MASS_FLOW,
        vol_flow=lambda time: CONTROLLED_VOL_FLOW)
    with pytest.raises(ValueError, match="control only one flow basis"):
        water.get_inputs(0.0)


class _TemperatureOnlyController:
    """Minimal controller returning only a fixed utility inlet temperature.

    ``CoolingWater.DynamicInlet`` documents a duck-typed protocol: the
    controller only needs ``evaluate_inputs(time)``. ``DynamicInput`` also
    carries a ``controls`` mapping, so it cannot show that ``CoolingWater``
    relies on nothing beyond that protocol, and no other PharmaPy object
    exposes ``evaluate_inputs`` controlling ``temp_in`` alone. This fixed,
    non-configurable object verifies exactly the ``evaluate_inputs`` handoff
    into ``CoolingWater.get_inputs``; the real ``DynamicInput`` collaborator
    is exercised by the other tests in this module.
    """

    def evaluate_inputs(self, time):
        """Return the controlled fields at ``time``.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation time [s]; unused because the control is constant.

        Returns
        -------
        dict
            ``{'temp_in': CONTROLLED_TEMPERATURE}`` [K].
        """
        return {"temp_in": CONTROLLED_TEMPERATURE}


class _MassFlowOnlyController:
    """Minimal controller returning only a fixed utility mass flow.

    Like ``_TemperatureOnlyController``, it implements only the documented
    duck-typed ``evaluate_inputs(time)`` protocol, without the ``controls``
    mapping of ``DynamicInput``, to verify that the controlled flow basis is
    identified from the returned fields. The real ``DynamicInput`` path is
    exercised by the other tests in this module.
    """

    def evaluate_inputs(self, time):
        """Return the controlled fields at ``time``.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation time [s]; unused because the control is constant.

        Returns
        -------
        dict
            ``{'mass_flow': CONTROLLED_MASS_FLOW}`` [kg/s].
        """
        return {"mass_flow": CONTROLLED_MASS_FLOW}


@pytest.mark.unit
def test_duck_typed_temperature_controller_keeps_static_flows():
    """Compatibility guard for the ``evaluate_inputs(time)`` protocol."""
    water = _controlled_water({"mass_flow": STATIC_MASS_FLOW})
    water.DynamicInlet = _TemperatureOnlyController()
    assert water.get_inputs(0.0) == pytest.approx({
        "temp_in": CONTROLLED_TEMPERATURE, "vol_flow": STATIC_VOL_FLOW,
        "mass_flow": STATIC_MASS_FLOW}, rel=ALGEBRA_RTOL)

    # Static fields are repeated per time for a profile evaluation.
    profile = water.get_inputs(PROFILE_TIMES)
    np.testing.assert_allclose(
        profile["vol_flow"], np.full(PROFILE_TIMES.shape, STATIC_VOL_FLOW),
        rtol=ALGEBRA_RTOL, atol=0)
    np.testing.assert_allclose(
        profile["mass_flow"], np.full(PROFILE_TIMES.shape, STATIC_MASS_FLOW),
        rtol=ALGEBRA_RTOL, atol=0)
    assert profile["temp_in"] == pytest.approx(CONTROLLED_TEMPERATURE,
                                               rel=ALGEBRA_RTOL)


@pytest.mark.unit
def test_duck_typed_mass_flow_controller_updates_volume_flow():
    water = _controlled_water({"mass_flow": STATIC_MASS_FLOW})
    water.DynamicInlet = _MassFlowOnlyController()
    inputs = water.get_inputs(0.0)
    assert inputs["mass_flow"] == pytest.approx(CONTROLLED_MASS_FLOW,
                                                rel=ALGEBRA_RTOL)
    assert inputs["vol_flow"] == pytest.approx(CONTROLLED_VOL_FLOW,
                                               rel=ALGEBRA_RTOL)
    assert inputs["temp_in"] == pytest.approx(UTILITY_TEMPERATURE,
                                              rel=ALGEBRA_RTOL)


def _jacketed_batch_derivatives(water):
    """Evaluate a jacketed BatchReactor right-hand side with a utility.

    Parameters
    ----------
    water : CoolingWater
        Utility supplying the jacket inlet temperature [K] and flow.

    Returns
    -------
    numpy.ndarray
        Derivatives of the participating concentrations [mol/L/s], the
        liquid temperature [K/s] and the jacket temperature [K/s].
    """
    reactor = BatchReactor(isothermal=False, ht_mode="jacket")
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=LIQUID_TEMPERATURE,
                                 vol=LIQUID_VOLUME, mole_conc=CONCENTRATIONS)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[RATE_CONSTANT], ea_params=[0.0],
        rxn_list=["A + B --> C"], delta_hrxn=REACTION_HEAT)
    reactor.Utility = water
    reactor.set_names()
    # Geometry and inert holdup that solve_unit sets before integrating.
    reactor.conc_inert = CONCENTRATIONS[~reactor.mask_species]  # [mol/L]
    vessel_volume = LIQUID_VOLUME / reactor.vol_offset  # [m**3]
    reactor.diam = (4 / np.pi * vessel_volume) ** (1 / 3)  # [m]
    reactor.area_base = np.pi / 4 * reactor.diam**2  # [m**2]
    states = np.array([*CONCENTRATIONS[reactor.mask_species],
                       LIQUID_TEMPERATURE, JACKET_TEMPERATURE])  # [mol/L], [K]
    return reactor.unit_model(0.0, states)


@pytest.mark.unit
def test_equivalent_flow_controls_reach_the_jacket_balance():
    """Both flow bases drive the same jacket derivative.

    The jacket balance is ``dTj/dt = F/Vj (Tin - Tj) + Q/(rho cp Vj)``.
    The states are identical in every case, so the heat term Q cancels and
    the controlled minus static jacket derivatives equal
    ``(F_controlled - F_static) / Vj * (Tin - Tj)``.
    """
    static = _jacketed_batch_derivatives(
        _controlled_water({"mass_flow": STATIC_MASS_FLOW}))  # [mixed /s]
    by_mass = _jacketed_batch_derivatives(_controlled_water(
        {"mass_flow": STATIC_MASS_FLOW},
        mass_flow=lambda time: CONTROLLED_MASS_FLOW))  # [mixed /s]
    by_volume = _jacketed_batch_derivatives(_controlled_water(
        {"mass_flow": STATIC_MASS_FLOW},
        vol_flow=lambda time: CONTROLLED_VOL_FLOW))  # [mixed /s]

    np.testing.assert_allclose(by_mass, by_volume, rtol=ALGEBRA_RTOL, atol=0)

    jacket_volume = JACKET_VOLUME_RATIO * LIQUID_VOLUME  # [m**3], 3e-4
    # (2e-4 - 1e-4) m**3/s / 3e-4 m**3 * (300 - 305) K = -5/3 K/s
    expected_shift = ((CONTROLLED_VOL_FLOW - STATIC_VOL_FLOW) / jacket_volume
                      * (UTILITY_TEMPERATURE - JACKET_TEMPERATURE))  # [K/s]
    assert expected_shift == pytest.approx(-5.0 / 3.0, rel=ALGEBRA_RTOL)
    assert by_mass[-1] - static[-1] == pytest.approx(expected_shift,
                                                     rel=ALGEBRA_RTOL)
    # Utility flow does not enter the species or liquid temperature balances.
    np.testing.assert_allclose(by_mass[:-1], static[:-1],
                               rtol=ALGEBRA_RTOL, atol=0)
