"""Tank-reactor estimation callbacks return the participating species (#497).

``_BaseReactor.paramest_wrapper`` returns the concentrations of the
participating species, in ``Kinetics.partic_species`` order, for every tank
reactor. ``BatchReactor`` integrates only those species, but ``CSTR`` and
``SemibatchReactor`` integrate every phase species, so the participating
species are not their leading state columns whenever a nonreacting species
precedes a reacting one in the phase. These callbacks used to return the
leading phase species instead.

Fixture: phase species A, B, C and solvent from the shipped PFR property
file, an isothermal charge, and for the CSTR and semibatch reactors a feed
of A and B. Three elementary reactions with zero activation energy select
the participating species: B --> C (issue #497, participating species after
a nonreacting one), A --> C (participating species separated by a
nonreacting one) and A + B --> C (leading species, unaffected by the
defect). Expected profiles come from an independent SciPy LSODA integration
of the batch, CSTR (constant volume, outflow equal to inflow) or semibatch
(accumulating volume, no outflow) balances; charge-time rows are the
written-out charge. A few dozen short reactor solves and three small fits;
requires the optional Assimulo stack.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import CSTR, BatchReactor, SemibatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater
from test_reactor_paramest_control_modifiers import (
    PROFILE_ATOL, PROFILE_RTOL, SOLVER_OPTIONS)
from test_reactor_paramest_requested_rows import THERMO_PATH

TEMPERATURE = 300.0  # [K], isothermal; zero activation energy below
VOLUME = 0.002  # [m**3], charged liquid volume
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
# Charge and feed of A, B, C and solvent, in the phase species order of
# THERMO_PATH. The feed carries B so that the inflow of a participating
# species enters the compared balances.
CHARGE = np.array([0.1, 0.2, 0.0, 5.0])  # [mol/L]
FEED = np.array([0.5, 0.3, 0.0, 5.0])  # [mol/L]
RESIDENCE_TIME = 5.0  # [s], charged volume over inflow, V0 / Q
SEMIBATCH_TANK_VOLUME = 2 * VOLUME  # [m**3], room for 2 s of feed
# Elementary rate constant: [1/s] for the first-order reactions and
# [L/mol/s] for A + B --> C, so k * t_max is 0.4 [-] or 0.4 L/mol. Between
# the charge and t_max every participating concentration changes by at
# least 7.5e-3 mol/L (batch A + B --> C; at least 0.027 mol/L for B --> C
# and 0.033 mol/L for A --> C), far above the PROFILE tolerances.
TRUE_RATE = 0.2  # [1/s] or [L/mol/s], see above
TRIAL_RATE = 0.15  # fit seed, in the units of TRUE_RATE
TIMES = np.array([0.0, 1.0, 2.0])  # [s]
CHARGE_TIMES = np.array([0.0, 0.0])  # [s], charge-time replicates
# Per reaction: participating species in phase order (the order of
# Kinetics.partic_species), written out; their phase columns; the
# stoichiometric coefficients over A, B, C and solvent [-]; and the
# elementary reaction orders [-].
CASES = {
    'B --> C': (['B', 'C'], [1, 2], [0.0, -1.0, 1.0, 0.0],
                [0, 1, 0, 0]),
    'A --> C': (['A', 'C'], [0, 2], [-1.0, 0.0, 1.0, 0.0],
                [1, 0, 0, 0]),
    'A + B --> C': (['A', 'B', 'C'], [0, 1, 2], [-1.0, -1.0, 1.0, 0.0],
                    [1, 1, 0, 0]),
}
# Reactions whose participating species are not the leading phase species.
SHIFTED_CASES = ['B --> C', 'A --> C']
REACTOR_CLASSES = [BatchReactor, CSTR, SemibatchReactor]
# Guard against a vacuous green: at every requested time, the leading
# phase columns that the defective CSTR and semibatch callbacks returned
# differ from the participating species by at least 0.104 mol/L in some
# column (max over columns; the minimum over rows, shifted cases and these
# two reactors is at 2 s in the semibatch reactor, B --> C; every charge row
# differs by 0.2 mol/L). Some single entries coincide (c_A in both
# selections for A --> C), so the guard uses the row maximum. 1e-2 mol/L
# is ten times below that separation and far above the PROFILE comparison
# tolerance (below 1e-7 mol/L here), so returning the leading columns at
# any requested time fails the comparison.
MIN_SEPARATION = 1e-2  # [mol/L]
# Residuals at the true rate are CVode minus LSODA errors of the observed
# c_B and c_C, whose magnitudes are at most the charge of B.
MAX_OBSERVED_CONC = 0.2  # [mol/L]
RESIDUAL_ATOL = PROFILE_RTOL * MAX_OBSERVED_CONC + PROFILE_ATOL  # [mol/L]
# Residual errors of at most RESIDUAL_ATOL (2.1e-8 mol/L) on the n = 4
# observations after the charge, whose rate-sensitivity norm |J| is at least
# 0.419 mol/L*s at k = 0.2 1/s (CSTR; batch 0.444, semibatch 0.427), shift
# the fit by at most sqrt(n) RESIDUAL_ATOL / |J| = 1.0e-7 1/s; the charge
# rows have zero sensitivity. 1e-6 1/s keeps a factor of ten.
FIT_ATOL = 1e-6  # [1/s]


def _reactor(reactor_cls, reaction, rate=TRUE_RATE):
    """Build an isothermal tank reactor for one fixture reaction.

    Parameters
    ----------
    reactor_cls : type
        ``BatchReactor``, ``CSTR`` or ``SemibatchReactor``.
    reaction : str
        Key of ``CASES``, the elementary reaction.
    rate : float, optional
        Rate constant, in [1/s] for first-order and [L/mol/s] for
        second-order reactions.

    Returns
    -------
    _BaseReactor
        Reactor charged with ``CHARGE`` [mol/L] at ``TEMPERATURE`` [K];
        the CSTR and semibatch reactors are fed ``FEED`` [mol/L] at
        ``VOLUME / RESIDENCE_TIME`` [m**3/s].
    """
    kwargs = {}
    if reactor_cls is SemibatchReactor:
        kwargs['vol_tank'] = SEMIBATCH_TANK_VOLUME  # [m**3]
    reactor = reactor_cls(isothermal=True, return_sens=False, **kwargs)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[rate], ea_params=[0.0],
        rxn_list=[reaction])  # rate units above, [J/mol]
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=list(CHARGE))
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    if reactor_cls is not BatchReactor:
        reactor.Inlet = LiquidStream(
            THERMO_PATH, temp=TEMPERATURE, mole_conc=list(FEED),
            vol_flow=VOLUME / RESIDENCE_TIME)  # [m**3/s]
    return reactor


def _independent_profile(reactor_cls, reaction, rate=TRUE_RATE,
                         times=TIMES):
    """Integrate the fixture balances with SciPy LSODA.

    Parameters
    ----------
    reactor_cls : type
        Reactor class whose balances are integrated: batch (no flow),
        CSTR (constant volume, outflow equal to inflow) or semibatch
        (accumulating volume, no outflow).
    reaction : str
        Key of ``CASES``.
    rate : float, optional
        Rate constant in the units of ``TRUE_RATE``.
    times : numpy.ndarray, optional
        Sample times [s], increasing from 0 s.

    Returns
    -------
    numpy.ndarray
        Concentrations of A, B, C and solvent [mol/L], shape
        ``(len(times), 4)``.
    """
    _, _, stoichiometry, orders = CASES[reaction]
    stoichiometry = np.array(stoichiometry)  # [-]
    orders = np.array(orders)  # [-]
    inflow = VOLUME / RESIDENCE_TIME  # [m**3/s]

    def rhs(time, states):
        """Return concentration [mol/L/s] and volume [m**3/s] rates.

        Parameters
        ----------
        time : float
            Time [s].
        states : numpy.ndarray
            Concentrations of A, B, C and solvent [mol/L], then the liquid
            volume [m**3], shape ``(5,)``.

        Returns
        -------
        numpy.ndarray
            Time derivatives of ``states``, shape ``(5,)``.
        """
        conc = states[:4]  # [mol/L]
        reaction_rate = rate * np.prod(conc**orders)  # [mol/L/s]
        dconc = stoichiometry * reaction_rate  # [mol/L/s]
        if reactor_cls is BatchReactor:
            return np.append(dconc, 0.0)
        volume = VOLUME if reactor_cls is CSTR else states[4]  # [m**3]
        dconc = dconc + inflow / volume * (FEED - conc)  # [mol/L/s]
        dvolume = 0.0 if reactor_cls is CSTR else inflow  # [m**3/s]
        return np.append(dconc, dvolume)

    initial = np.append(CHARGE, VOLUME)  # [mol/L], [m**3]
    solution = solve_ivp(rhs, (times[0], times[-1]), initial,
                         method='LSODA', t_eval=times,
                         rtol=SOLVER_OPTIONS['rtol'],
                         atol=SOLVER_OPTIONS['atol'])
    assert solution.success
    return solution.y[:4].T


def _evaluate(reactor, times):
    """Call ``paramest_wrapper`` at the reactor's current parameters.

    Parameters
    ----------
    reactor : _BaseReactor
        Configured reactor.
    times : numpy.ndarray
        Requested times [s].

    Returns
    -------
    numpy.ndarray
        The wrapper's concentrations [mol/L], one row per requested time.
    """
    return reactor.paramest_wrapper(
        reactor.Kinetics.concat_params(), times,
        run_args={'sundials_opts': SOLVER_OPTIONS})


@pytest.mark.parametrize('reaction', list(CASES))
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_wrapper_returns_participating_species(reactor_cls, reaction):
    names, phase_columns, _, _ = CASES[reaction]
    reactor = _reactor(reactor_cls, reaction)
    assert reactor.Kinetics.partic_species == names

    profile = _evaluate(reactor, TIMES)  # [mol/L]
    expected = _independent_profile(reactor_cls, reaction)  # [mol/L]
    assert profile.shape == (len(TIMES), len(names))
    np.testing.assert_allclose(profile, expected[:, phase_columns],
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)

    # The rows are the stored result's participating columns, selected by
    # species name, at the requested times.
    stored_names = list(reactor.states_di['mole_conc']['index'])
    stored_columns = [stored_names.index(name) for name in names]
    result_rows = [np.flatnonzero(reactor.result.time == time)[0]
                   for time in TIMES]
    np.testing.assert_array_equal(
        profile,
        reactor.result.mole_conc[np.ix_(result_rows, stored_columns)])

    if reactor_cls is not BatchReactor and reaction in SHIFTED_CASES:
        # Guard: at every requested time the leading phase columns, which
        # the defective callback returned, differ in some column.
        leading = expected[:, :len(names)]  # [mol/L]
        separation = np.max(np.abs(leading - expected[:, phase_columns]),
                            axis=1)  # [mol/L], per requested time
        assert np.all(separation > MIN_SEPARATION)


@pytest.mark.parametrize('reaction', list(CASES))
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_charge_time_rows_hold_participating_charge(reactor_cls, reaction):
    _, phase_columns, _, _ = CASES[reaction]
    reactor = _reactor(reactor_cls, reaction)
    profile = _evaluate(reactor, CHARGE_TIMES)  # [mol/L]
    np.testing.assert_array_equal(
        profile, np.tile(CHARGE[phase_columns], (len(CHARGE_TIMES), 1)))


@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_simulation_exec_fits_participating_species(reactor_cls):
    reaction = 'B --> C'
    _, phase_columns, _, _ = CASES[reaction]
    # Error-free observations of c_B and c_C at the true rate [mol/L].
    data = _independent_profile(reactor_cls, reaction)[:, phase_columns]
    experiment = Experiment({'c_B': Measurement(0, TIMES, list(data[:, 0])),
                             'c_C': Measurement(1, TIMES, list(data[:, 1]))})
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = _reactor(reactor_cls, reaction, rate=TRIAL_RATE)
    simulation.SetParamEstimation(
        x_data=experiment, optimize_flags=[True, False],
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    estimator = simulation.ParamInst

    residual = estimator.get_objective([TRUE_RATE],
                                       out_array=True)  # [mol/L]
    assert residual.shape == (2 * len(TIMES),)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=RESIDUAL_ATOL)

    fitted, _, _ = simulation.EstimateParams(verbose=False)  # [1/s]
    np.testing.assert_allclose(fitted, [TRUE_RATE], rtol=0.0, atol=FIT_ATOL)
