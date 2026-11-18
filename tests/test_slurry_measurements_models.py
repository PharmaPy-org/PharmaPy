"""Measurement predictions on the legacy and the refactored batch crystallizer.

Synthetic verification, not a calibration: species A of the test database
crystallizes from its solvent on seed, with a prescribed cooling and guessed
kinetics. Both models solve the same case with CVode. The tests check that each
adapter reads its model on the right basis (the solute stays in the vessel) and
that the two models then give the same predictions. Assimulo lane; two solves of
a few seconds.
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy import SlurryMeasurements as measurements
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance
from PharmaPy import Phases, Phases_Refactored
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController

DATABASE = str(Path(__file__).parent / 'Flowsheet' / 'data' / 'compound_database.json')
SOLUTE_INDEX, SOLVENT_INDEX = 0, 4  # A and solvent among A, B, C, D, solvent
SOLVENT_MASS = 0.35  # [kg]
SOLUTE_MASS = 0.0165  # [kg], dissolved at the start
SEED_MASS = 0.0005  # [kg]
SEED_SIZE = 50e-6  # [m], one size
SHAPE_FACTOR = np.pi / 6.0  # [-], length taken as the equal-volume sphere diameter
SOLID_DENSITY = 1230.0  # [kg/m**3], rho_solid of A in the test database
# Solubility as solute per solvent, ln(ratio) = intercept + slope / T: 0.040 kg/kg
# at 310 K and 0.019 kg/kg at 293 K, so the charge of 0.047 kg/kg starts supersaturated.
INTERCEPT, SLOPE = 9.976, -4094.0  # [-], [K]
START_TEMPERATURE, END_TEMPERATURE = 310.0, 293.0  # [K]
RAMP_TIME = 6000.0  # [s]
TIMES = np.linspace(0.0, 9000.0, 10)  # [s]
GROWTH = (0.02, 0.0, 1.5)  # k_g [um/s], E [J/mol], order [-]; guessed
SECONDARY_NUCLEATION = (1e9, 0.0, 2.0, 1.0)  # k_s [#/m**3/s], E [J/mol], orders [-]; guessed
# Measured on this case: the legacy model keeps the solute to 2.1e-7 of the charge
# and the refactored one to 7e-10; about five times the larger is allowed.
BALANCE_TOLERANCE = 1e-6  # [-]
# Measured on this case: the predictions of the two models differ by up to 1.0 %
# at the end (count signal), 0.7 % for the dissolved ratio. The legacy unit treats
# the liquid density as constant, a known limitation, so its liquid volume ends
# 0.6 % low; the refactored unit does not. Twice the measured difference is allowed.
PARITY_TOLERANCE = 0.02  # [-]


def temperature(time):
    """Return the prescribed temperature.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].

    Returns
    -------
    float or numpy.ndarray
        Temperature [K]: a linear ramp over `RAMP_TIME`, then constant.
    """
    return START_TEMPERATURE + (END_TEMPERATURE - START_TEMPERATURE) * np.minimum(np.asarray(time) / RAMP_TIME, 1.0)


def saturated_fraction(temp, conc=None):
    """Return the solubility as a mass fraction.

    Parameters
    ----------
    temp : float or numpy.ndarray
        Temperature [K].
    conc : numpy.ndarray, optional
        Liquid composition; unused.

    Returns
    -------
    float or numpy.ndarray
        Solute per solution at saturation [kg/kg].
    """
    ratio = np.exp(INTERCEPT + SLOPE / np.asarray(temp, dtype=float))  # [kg solute/kg solvent]
    return ratio / (1.0 + ratio)


def _charge():
    """Return the liquid mass [kg], its mass fractions [-] and the seed moments [m**k]."""
    total = SOLVENT_MASS + SOLUTE_MASS
    fractions = [SOLUTE_MASS / total, 0.0, 0.0, 0.0, SOLVENT_MASS / total]
    number = SEED_MASS / (SOLID_DENSITY * SHAPE_FACTOR * SEED_SIZE ** 3)  # [-]
    return total, fractions, number * SEED_SIZE ** np.arange(4)


def _kinetics(**options):
    """Return the kinetics both models share, with model-specific options."""
    return CrystKinetics(solub_fn=saturated_fraction, growth=GROWTH, nucl_sec=SECONDARY_NUCLEATION, **options)


@pytest.fixture(scope="module")
def legacy_state():
    total, fractions, moments = _charge()
    unit = BatchCryst('A', method='moments', controls={'temp': temperature}, basis='mass_frac')
    unit.Phases = (
        Phases.LiquidPhase(DATABASE, temp=START_TEMPERATURE, mass=total, mass_frac=fractions),
        Phases.SolidPhase(DATABASE, temp=START_TEMPERATURE, moments=moments, kv=SHAPE_FACTOR,
                          mass_frac=[1.0, 0.0, 0.0, 0.0, 0.0], num_mom=4))
    unit.Kinetics = _kinetics()
    time, states = unit.solve_unit(time_grid=TIMES, verbose=False)[:2]
    return measurements.state_from_total_moments(time, states, SOLUTE_INDEX, SOLVENT_INDEX, SHAPE_FACTOR)


@pytest.fixture(scope="module")
def refactored_state():
    total, fractions, moments = _charge()
    liquid = Phases_Refactored.LiquidPhase(DATABASE, temp=START_TEMPERATURE, mass=total, mass_frac=fractions,
                                           name_solv='solvent')
    solid = Phases_Refactored.SolidPhase(DATABASE, mass=SEED_MASS, mass_frac=[1.0, 0.0, 0.0, 0.0, 0.0])
    slurry_volume = liquid.vol + SEED_MASS / SOLID_DENSITY  # [m**3]
    solid.mechanisms = MomentsPopulationBalance(
        owning_phase=solid, target_components='A', solvent_name='solvent',
        moments_init=moments / measurements.METERS_PER_MICROMETER ** np.arange(4) / slurry_volume,  # [um**k/m**3]
        kv=SHAPE_FACTOR, density=SOLID_DENSITY)
    unit = BatchCrystallizer(integrator=AssimuloBackend(),
                             controller=SimpleTemperatureController(temp_func=lambda time: float(temperature(time))))
    unit.Phases = [liquid, solid]
    unit.CrystKinetics = _kinetics(solubility_basis='mass_per_mass_solution')
    time, _ = unit.solve_unit(time_grid=TIMES, verbose=False)
    result = unit.result
    return measurements.state_from_specific_moments(time, result.mu_n_solid0, result.mass_j_liquid0, result.vessel_vol,
                                                    SOLUTE_INDEX, SOLVENT_INDEX)


@pytest.mark.parametrize("fixture_name", ["legacy_state", "refactored_state"])
def test_each_model_is_read_on_the_right_basis(request, fixture_name):
    state = request.getfixturevalue(fixture_name)
    charged = SOLUTE_MASS + SEED_MASS  # [kg]

    np.testing.assert_allclose(state.time, TIMES)
    assert state.solute_mass[0] == pytest.approx(SOLUTE_MASS, rel=1e-9)
    assert measurements.solid_mass(state, SOLID_DENSITY, SHAPE_FACTOR)[0] == pytest.approx(SEED_MASS, rel=1e-9)
    np.testing.assert_allclose(measurements.total_solute(state, SOLID_DENSITY, SHAPE_FACTOR), charged, rtol=BALANCE_TOLERANCE)
    np.testing.assert_allclose(state.solvent_mass, SOLVENT_MASS, rtol=BALANCE_TOLERANCE)
    assert measurements.mean_size(state, upper=1, lower=0)[0] == pytest.approx(SEED_SIZE, rel=1e-9)
    # The case does crystallize: most of the solute above saturation leaves the liquid.
    assert measurements.dissolved_ratio(state)[-1] < 0.6 * SOLUTE_MASS / SOLVENT_MASS


def test_both_models_give_the_same_predictions(legacy_state, refactored_state):
    for name, prediction in (
            ("dissolved ratio", measurements.dissolved_ratio),
            ("solid mass", lambda state: measurements.solid_mass(state, SOLID_DENSITY, SHAPE_FACTOR)),
            ("count signal", lambda state: measurements.count_signal(state, 1, 1.0)),
            ("Sauter mean", measurements.mean_size)):
        legacy, refactored = prediction(legacy_state), prediction(refactored_state)
        assert refactored[0] == pytest.approx(legacy[0], rel=1e-9), name  # the same charge through both adapters
        np.testing.assert_allclose(refactored, legacy, rtol=PARITY_TOLERANCE, err_msg=name)
