"""Crystallizer energy regressions through production model contracts."""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater


pytestmark = pytest.mark.unit

SLURRY_VOLUME = 1.0e-3  # [m**3]
TEMPERATURE = 300.0  # [K]
SPECIFIC_MOMENTS = np.array([1.0e8, 1.0e4, 1.0, 0.05])  # [m**n/m**3]


def _build_crystallizer(data_path, crystallizer_type=MSMPR, *, adiabatic=True):
    """Build a production crystallizer with real phases and kinetics.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    crystallizer_type : type, optional
        Production crystallizer class to configure.
    adiabatic : bool, optional
        Whether the vessel excludes utility heat transfer [-].

    Returns
    -------
    MSMPR or SemibatchCryst
        Configured crystallizer with a liquid feed and, when needed, a real
        cooling-water utility.
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    liquid = LiquidPhase(
        thermo_path,
        temp=TEMPERATURE,
        vol=SLURRY_VOLUME,  # [m**3]
        mass_frac=[0.55, 0.25, 0.10, 0.10],  # [-]
        verbose=False,
    )
    solid = SolidPhase(
        thermo_path,
        temp=TEMPERATURE,
        moments=SPECIFIC_MOMENTS * SLURRY_VOLUME,  # [m**n]
        mass_frac=[1.0, 0.0, 0.0, 0.0],  # [-]
        kv=0.5,  # [-]
    )
    slurry = Slurry(vol=SLURRY_VOLUME, moments=SPECIFIC_MOMENTS)
    slurry.Phases = [liquid, solid]

    crystallizer = crystallizer_type(
        "A",
        method="moments",
        vol_tank=SLURRY_VOLUME,  # [m**3]
        adiabatic=adiabatic,
    )
    crystallizer.Phases = slurry
    crystallizer.num_species = liquid.num_species  # [-]
    crystallizer.Kinetics = CrystKinetics(
        coeff_solub=[0.0],  # [kg/m**3]
        growth=[1.0e-3, 0.0, 1.0],  # [um/s], [J/mol], [-]
        sup_sat_type="absolute",
    )
    crystallizer.Kinetics.target_idx = crystallizer.target_ind

    inlet = LiquidStream(
        thermo_path,
        temp=305.0,  # [K]
        vol_flow=2.0e-6,  # [m**3/s]
        mass_frac=[0.60, 0.20, 0.10, 0.10],  # [-]
        verbose=False,
    )
    crystallizer.Inlet = inlet

    # These geometry values are the production initialization performed by
    # ``solve_unit`` before evaluating energy balances.
    crystallizer.diam_tank = (
        4.0 / np.pi * crystallizer.vol_tank
    ) ** (1.0 / 3.0)  # [m]
    crystallizer.area_base = (
        np.pi / 4.0 * crystallizer.diam_tank**2
    )  # [m**2]
    crystallizer.vol_tank /= crystallizer.vol_offset  # [m**3]

    if not adiabatic:
        crystallizer.Utility = CoolingWater(
            vol_flow=2.0e-5,  # [m**3/s]
            temp_in=285.0,  # [K]
        )

    return crystallizer


def _energy_arguments(crystallizer):
    """Assemble direct energy-balance inputs from production collaborators.

    Parameters
    ----------
    crystallizer : MSMPR or SemibatchCryst
        Configured production crystallizer.

    Returns
    -------
    dict
        Keyword arguments using [s], [kg/s] or [kg/m**3/s], [kg/m**3],
        [m**n], [kg/m**3], [K], [m**3], and [J/m**3] as appropriate to
        ``energy_balances``.
    """
    inlet_temperature = crystallizer.Inlet.temp  # [K]
    inlet_density = crystallizer.Inlet.getDensity(
        temp=inlet_temperature
    )  # [kg/m**3]
    inlet_enthalpy = crystallizer.Inlet.getEnthalpy(
        temp=inlet_temperature
    ) * inlet_density  # [J/m**3]
    suspension_density = crystallizer.Slurry.getDensity(
        temp=TEMPERATURE
    )  # [kg/m**3]
    densities = [
        suspension_density,
        np.array([inlet_density, None], dtype=object),
    ]
    moments = (
        SPECIFIC_MOMENTS
        if type(crystallizer) is MSMPR
        else crystallizer.Solid_1.moments
    )  # [m**n/m**3] or [m**n]
    return {
        "time": 0.0,  # [s]
        "params": None,
        "cryst_rate": 0.0,  # [kg/m**3/s] or [kg/s]
        "u_inputs": {"Inlet": {"vol_flow": crystallizer.Inlet.vol_flow}},
        "rhos": densities,
        "mu_n": moments,
        "distrib": None,
        "mass_conc": crystallizer.Liquid_1.mass_conc,
        "temp": TEMPERATURE,
        "vol": crystallizer.Liquid_1.vol,  # [m**3]
        "h_in": inlet_enthalpy,
    }


def test_msmpr_adiabatic_energy_balance_has_no_jacket_equation(data_path):
    """An adiabatic production MSMPR returns only the tank derivative."""
    crystallizer = _build_crystallizer(data_path, adiabatic=True)

    temperature_rate = crystallizer.energy_balances(
        temp_ht=None,
        **_energy_arguments(crystallizer),
    )  # [K/s]

    assert np.ndim(temperature_rate) == 0
    assert np.isfinite(temperature_rate)


def test_semibatch_jacket_uses_utility_inputs(data_path):
    """A production semibatch jacket uses a real cooling-water utility."""
    crystallizer = _build_crystallizer(
        data_path,
        crystallizer_type=SemibatchCryst,
        adiabatic=False,
    )

    tank_rate, jacket_rate = crystallizer.energy_balances(
        temp_ht=290.0,  # [K]
        **_energy_arguments(crystallizer),
    )  # [K/s], [K/s]

    assert np.isfinite(tank_rate)
    assert np.isfinite(jacket_rate)


def test_liquid_feed_enthalpy_is_volumetric_in_msmpr_flow_term(data_path):
    """The public MSMPR path passes liquid-feed enthalpy in [J/m**3]."""
    crystallizer = _build_crystallizer(data_path, adiabatic=True)
    # A real upstream transfer supplies the liquid-phase concentration field;
    # static stream attributes supply inlet temperature and volumetric flow.
    crystallizer.Inlet.y_upstream = {"mass_conc": True}
    crystallizer.Inlet.y_inlet = {
        "mass_conc": crystallizer.Inlet.mass_conc.copy()
    }  # [kg/m**3]
    raw_moments = SPECIFIC_MOMENTS * (1.0e6) ** np.arange(4)  # [um**n/m**3]
    states = np.concatenate((
        raw_moments,
        crystallizer.Liquid_1.mass_conc,  # [kg/m**3]
        [TEMPERATURE],  # [K]
    ))

    heat_components = crystallizer.unit_model(
        0.0,
        states,
        params=None,
        enrgy_bce=True,
    )  # [J/s]

    inlet_temperature = crystallizer.Inlet.temp  # [K]
    inlet_density = crystallizer.Inlet.getDensity(
        temp=inlet_temperature
    )  # [kg/m**3]
    inlet_enthalpy_mass = crystallizer.Inlet.getEnthalpy(
        temp=inlet_temperature
    )  # [J/kg]
    suspension_density = crystallizer.Slurry.getDensity(
        temp=TEMPERATURE
    )  # [kg/m**3]
    liquid_volume_fraction = (
        1.0 - crystallizer.Solid_1.kv * SPECIFIC_MOMENTS[3]
    )  # [-]
    suspension_enthalpy = crystallizer.Slurry.getEnthalpy(
        TEMPERATURE,
        [liquid_volume_fraction, 1.0 - liquid_volume_fraction],
        suspension_density,
    )  # [J/m**3]
    expected_flow_term = crystallizer.Inlet.vol_flow * (
        inlet_enthalpy_mass * inlet_density - suspension_enthalpy
    )  # [J/s]
    mass_only_flow_term = crystallizer.Inlet.vol_flow * (
        inlet_enthalpy_mass - suspension_enthalpy
    )  # [J/s], intentionally wrong basis

    assert heat_components[2] == pytest.approx(expected_flow_term)
    assert heat_components[2] != pytest.approx(mass_only_flow_term)


def test_slurry_getcp_times_vliq_does_not_mutate_volfracs(data_path):
    """The real slurry heat-capacity path leaves caller fractions unchanged."""
    crystallizer = _build_crystallizer(data_path, adiabatic=True)
    volume_fractions = [0.9, 0.1]  # [-]
    densities = crystallizer.Slurry.getDensity(
        temp=TEMPERATURE
    )  # [kg/m**3]

    crystallizer.Slurry.getCp(
        TEMPERATURE,
        volume_fractions,
        densities,
        times_vliq=True,
    )  # [J/m**3/K]

    assert volume_fractions == pytest.approx([0.9, 0.1])
