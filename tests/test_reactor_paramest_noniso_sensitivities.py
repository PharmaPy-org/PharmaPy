"""Non-isothermal batch estimation sensitivities match the returned species.

A non-isothermal ``BatchReactor`` integrates the participating species
concentrations [mol/L] and the reactor and jacket temperatures [K]. Its
``paramest_wrapper`` returns concentrations for the participating species
only, so the CVodes sensitivities it returns with ``return_sens=True`` must
cover exactly those columns, in the layout that
``ParameterEstimation.select_sens`` splits by ``c_prof`` column. They used
to cover every integrated state, so the estimator paired the second species
measurement with the reactor-temperature sensitivity dT/dk [K*s] (#498).

Fixture: phase species A, B, C and solvent from the shipped PFR property
file, a jacketed reactor charged at 300 K and cooled with 290 K water, and
one exothermic elementary first-order reaction with its rate constant
defined at 300 K: A --> B (the issue case) or B --> C (a nonreacting
species before the participating ones). The reaction heats the reactor by
7 K (A --> B) or 20 K (B --> C) within 3 s, so the concentrations depend on
the energy balance. Expected sensitivities are central finite differences
of the wrapper's own concentrations; estimator Jacobians and covariances
with CVodes sensitivities are compared with those of the estimator's
forward differences. Observations are synthetic: the wrapper's
concentrations at the true rate constant plus deliberate errors at the
charge time, whose sensitivities are zero. A few dozen short reactor solves
and four small fits; requires the optional Assimulo stack.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater
from test_reactor_paramest_requested_rows import (
    FD_NOISE_ATOL, FD_RELATIVE_STEP, THERMO_PATH)

CHARGE_TEMPERATURE = 300.0  # [K], also the kinetics reference temperature
COOLANT_TEMPERATURE = 290.0  # [K], jacket water inlet
COOLANT_FLOW = 0.1  # [kg/s]
VOLUME = 0.002  # [m**3]
# Charge of A, B, C and solvent in the phase species order of THERMO_PATH.
CHARGE = [0.1, 0.2, 0.0, 5.0]  # [mol/L]
TRUE_RATE = 0.2  # [1/s], at CHARGE_TEMPERATURE
TRIAL_RATE = 0.15  # [1/s], evaluation point and fit seed
# Synthetic Arrhenius activation energy and exotherm: d ln k / dT =
# Ea / (R T**2) = 0.067 1/K at 300 K, so the 7-20 K rise changes the rate
# constant by up to 60-260 %.
ACTIVATION_ENERGY = 5.0e4  # [J/mol]
REACTION_HEAT = -1.0e5  # [J/mol]
# Per reaction: participating species (Kinetics.partic_species), written out.
REACTIONS = {'A --> B': ['A', 'B'], 'B --> C': ['B', 'C']}
NUM_PARAMS = 2  # [-], rate constant [1/s] and activation energy [J/mol]
NUM_SPECIES = 2  # [-], participating species of each reaction
TIMES = np.array([0.0, 1.0, 2.0, 3.0])  # [s]
CHARGE_TIMES = np.array([0.0, 0.0])  # [s], charge-time replicates
# Both integrators reach global errors within 100 times their local
# tolerances (budget of the committed reactor regressions), so each solve
# is within SOLVE_ERROR = 100 * (rtol * 0.3 mol/L + atol) = 3.1e-10 mol/L
# of the exact profile; all concentrations here stay below 0.3 mol/L.
SOLVER_OPTIONS = {'rtol': 1e-11, 'atol': 1e-13}  # [-], [mol/L]
SOLVE_ERROR = 3.1e-10  # [mol/L]
# Central differences with a 1e-4 relative step: their truncation error,
# about (h t)**2 / 6 relative for first-order decay (6e-10 at t = 3 s), is
# negligible, and two solve errors bound the rounding part by
# SOLVE_ERROR / h per parameter (1.6e-5 mol/L*s for the rate constant,
# 6.2e-11 mol**2/(L*J) for the activation energy). CVodes includes the
# sensitivities in its error test (suppress_sens=False) at the same
# tolerances, so their own error, 100 * rtol * |dc/dk| < 1e-9 mol/L*s, is
# smaller still. Observed differences are below 6e-9 mol/L*s and
# 7e-13 mol**2/(L*J).
CENTRAL_RELATIVE_STEP = 1e-4  # [-]
# The estimator's forward differences (step FD_RELATIVE_STEP times the rate
# constant) use the bounds of the requested-rows module: truncation
# FD_RELATIVE_STEP * k * t_max relative, twice the first-order-decay
# estimate (h t / 2) |d2c/dk2| / |dc/dk| (6e-4 at k = 0.2 1/s and
# t_max = 3 s; observed at most 1.5e-4 with the temperature coupling), and
# noise FD_NOISE_ATOL.
JACOBIAN_RTOL = FD_RELATIVE_STEP * TRUE_RATE * TIMES[-1]  # [-]
# A relative Jacobian error d changes inv(J J^T) by about 2 d; the residual
# mean squares agree because both fits leave only the charge-time errors.
# Observed covariance differences are at most 6e-4.
COVARIANCE_RTOL = 2 * JACOBIAN_RTOL  # [-]
# The charge-time errors have zero rate sensitivity, so the true rate
# constant is the optimum; both fits stop within 3e-8 1/s of it (observed).
# 1e-6 1/s, the bound of the requested-rows module, keeps a factor of 30.
FIT_ATOL = 1e-6  # [1/s]
# Deliberate observation errors at the charge-time replicates [mol/L],
# unequal and of opposite sign for the two measured species.
CHARGE_ERRORS = np.array([2e-3, -1e-3, 0.0, 0.0, 0.0])  # [mol/L]
FIT_TIMES = np.array([0.0, 0.0, 1.0, 2.0, 3.0])  # [s]


def _reactor(reaction, return_sens, rate=TRUE_RATE):
    """Build the jacketed non-isothermal batch reactor.

    Parameters
    ----------
    reaction : str
        Key of ``REACTIONS``.
    return_sens : bool
        Whether ``paramest_wrapper`` returns CVodes sensitivities.
    rate : float, optional
        Rate constant at ``CHARGE_TEMPERATURE`` [1/s].

    Returns
    -------
    BatchReactor
        Reactor charged with ``CHARGE`` [mol/L] at ``CHARGE_TEMPERATURE``
        [K], cooled by a jacket fed at ``COOLANT_TEMPERATURE`` [K].
    """
    reactor = BatchReactor(isothermal=False, return_sens=return_sens)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[rate], ea_params=[ACTIVATION_ENERGY],
        rxn_list=[reaction], temp_ref=CHARGE_TEMPERATURE,
        delta_hrxn=REACTION_HEAT, tref_hrxn=CHARGE_TEMPERATURE)
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=CHARGE_TEMPERATURE,
                                 vol=VOLUME, mole_conc=CHARGE)
    reactor.Utility = CoolingWater(temp_in=COOLANT_TEMPERATURE,
                                   mass_flow=COOLANT_FLOW)
    return reactor


def _evaluate(reactor, params, times, **kwargs):
    """Call ``paramest_wrapper`` with the module's solver options.

    Parameters
    ----------
    reactor : BatchReactor
        Configured reactor.
    params : numpy.ndarray
        Rate constant [1/s] and activation energy [J/mol].
    times : numpy.ndarray
        Requested times [s].
    **kwargs
        Further ``paramest_wrapper`` keywords.

    Returns
    -------
    numpy.ndarray or tuple
        Concentrations [mol/L] and, with ``return_sens``, sensitivities
        [mol/L per parameter unit].
    """
    return reactor.paramest_wrapper(
        params, times, run_args={'sundials_opts': SOLVER_OPTIONS}, **kwargs)


def _central_differences(reaction, params):
    """Differentiate the wrapper's concentrations by central differences.

    Parameters
    ----------
    reaction : str
        Key of ``REACTIONS``.
    params : numpy.ndarray
        Rate constant [1/s] and activation energy [J/mol].

    Returns
    -------
    numpy.ndarray
        Derivatives of the concentrations at ``TIMES``, shape
        ``(NUM_PARAMS, len(TIMES), NUM_SPECIES)``, in [mol/L*s] and
        [mol**2/(L*J)].
    """
    derivatives = []  # [mol/L*s] for k, [mol**2/(L*J)] for Ea
    for index, value in enumerate(params):
        step = CENTRAL_RELATIVE_STEP * value  # [1/s] or [J/mol]
        shifted = []
        for sign in (1.0, -1.0):
            trial = np.array(params, dtype=float)
            trial[index] = value + sign * step
            shifted.append(_evaluate(_reactor(reaction, False), trial,
                                     TIMES))  # [mol/L]
        derivatives.append((shifted[0] - shifted[1]) / (2 * step))
    return np.array(derivatives)


@pytest.mark.parametrize('reaction', list(REACTIONS))
def test_sensitivities_match_returned_concentrations(reaction):
    reactor = _reactor(reaction, return_sens=True)
    assert reactor.Kinetics.partic_species == REACTIONS[reaction]
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    # Concentrations [mol/L]; sensitivities [mol/L*s] for k and
    # [mol**2/(L*J)] for Ea.
    profile, stacked = _evaluate(reactor, params, TIMES, reord_sens=False)
    # The reactor integrates two temperatures besides the species.
    assert reactor.name_states == ['mole_conc', 'temp', 'temp_ht']
    assert profile.shape == (len(TIMES), NUM_SPECIES)
    assert stacked.shape == (NUM_PARAMS, len(TIMES), NUM_SPECIES)

    expected = _central_differences(
        reaction, params)  # [mol/L*s] for k, [mol**2/(L*J)] for Ea
    steps = CENTRAL_RELATIVE_STEP * params  # [1/s], [J/mol]
    for index in range(NUM_PARAMS):
        np.testing.assert_allclose(stacked[index], expected[index],
                                   rtol=0.0,
                                   atol=SOLVE_ERROR / steps[index])

    # Reordered layout: rows state-major (first species over the requested
    # times, then the second), one column per parameter.
    reordered = _evaluate(
        reactor, params, TIMES)[1]  # [mol/L*s] for k, [mol**2/(L*J)] for Ea
    assert reordered.shape == (NUM_SPECIES * len(TIMES), NUM_PARAMS)
    for species in range(NUM_SPECIES):
        rows = slice(species * len(TIMES), (species + 1) * len(TIMES))
        np.testing.assert_array_equal(reordered[rows],
                                      stacked[:, :, species].T)


@pytest.mark.parametrize('reaction', list(REACTIONS))
def test_charge_time_sensitivities_have_species_layout(reaction):
    reactor = _reactor(reaction, return_sens=True)
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    # Concentrations [mol/L]; sensitivities [mol/L*s] for k and
    # [mol**2/(L*J)] for Ea, in both layouts.
    profile, reordered = _evaluate(reactor, params, CHARGE_TIMES)
    stacked = _evaluate(reactor, params, CHARGE_TIMES,
                        reord_sens=False)[1]
    assert profile.shape == (len(CHARGE_TIMES), NUM_SPECIES)
    np.testing.assert_array_equal(
        reordered, np.zeros((NUM_SPECIES * len(CHARGE_TIMES), NUM_PARAMS)))
    np.testing.assert_array_equal(
        stacked, np.zeros((NUM_PARAMS, len(CHARGE_TIMES), NUM_SPECIES)))


def _estimation(reaction, return_sens):
    """Set up the estimation of the rate constant from two species.

    Parameters
    ----------
    reaction : str
        Key of ``REACTIONS``.
    return_sens : bool
        Whether the estimator uses CVodes sensitivities (True) or its
        forward differences (False).

    Returns
    -------
    SimulationExec
        Flowsheet with the reactor at ``TRIAL_RATE`` [1/s] and an
        estimator of the rate constant (activation energy fixed) from both
        participating species, observed at ``FIT_TIMES`` [s].
    """
    truth = _reactor(reaction, return_sens=False)
    model = _evaluate(truth, truth.Kinetics.concat_params(),
                      FIT_TIMES)  # [mol/L]
    first, second = REACTIONS[reaction]
    experiment = Experiment({
        'c_' + first: Measurement(0, FIT_TIMES,
                                  list(model[:, 0] + CHARGE_ERRORS)),
        'c_' + second: Measurement(1, FIT_TIMES,
                                   list(model[:, 1] - CHARGE_ERRORS))})
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = _reactor(reaction, return_sens, rate=TRIAL_RATE)
    simulation.SetParamEstimation(
        x_data=experiment, optimize_flags=[True, False],
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    return simulation


@pytest.mark.parametrize('reaction', list(REACTIONS))
def test_estimator_jacobian_matches_finite_differences(reaction):
    jacobians = {}
    for return_sens in (True, False):
        estimator = _estimation(reaction, return_sens).ParamInst
        estimator.get_objective([TRIAL_RATE], out_array=True)
        # Weighted-residual Jacobian row of the rate constant [mol/L*s],
        # packed by species, then sample.
        jacobians[return_sens] = estimator.get_gradient(
            [TRIAL_RATE], out_array=True)[0]

    assert jacobians[True].shape == (NUM_SPECIES * len(FIT_TIMES),)
    np.testing.assert_allclose(jacobians[True], jacobians[False],
                               rtol=JACOBIAN_RTOL, atol=FD_NOISE_ATOL)
    # The charge-time rows have zero sensitivity in both species.
    charge_rows = [0, 1, len(FIT_TIMES), len(FIT_TIMES) + 1]
    np.testing.assert_array_equal(jacobians[True][charge_rows], 0.0)


@pytest.mark.parametrize('reaction', list(REACTIONS))
def test_fit_with_sensitivities_matches_finite_difference_fit(reaction):
    fits, covariances = {}, {}
    for return_sens in (True, False):
        fits[return_sens], covariances[return_sens], _ = (
            _estimation(reaction, return_sens).EstimateParams(
                verbose=False))  # [1/s], [1/s**2]

    for return_sens in (True, False):
        np.testing.assert_allclose(fits[return_sens], [TRUE_RATE], rtol=0.0,
                                   atol=FIT_ATOL)
    np.testing.assert_allclose(covariances[True], covariances[False],
                               rtol=COVARIANCE_RTOL, atol=0.0)
