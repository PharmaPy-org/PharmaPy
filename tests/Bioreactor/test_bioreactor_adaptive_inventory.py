"""Native adaptive solvers on analytical sequential reactions and empty pools.

Real phases, mechanisms and vessels isolate generic depletion handling from
experimental parameter assumptions. CVode cases require the real Assimulo lane.
"""

from pathlib import Path
import json

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef, StateKey, TransferResult
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase
from tests.Bioreactor.test_bioreactor_thermal import thermal_case


# Synthetic first-order networks have closed-form, independently derived states.
THERMO = Path(__file__).parents[1] / 'fixtures/native_four_species.json'
EXAMPLE = Path(__file__).parent / 'fixtures/analytical_conversion'
RATE = 4.  # [1/s]; fast enough to exhaust a pool within the unrelated default 1 s limiter horizon


class SequentialReaction(Mechanism):
    """Equal-mass A -> B -> C with equal first-order rates, or an invalid sink."""

    def __init__(self, invalid=False):
        """Configure the analytical fixture.

        Parameters
        ----------
        invalid : bool
            Consume empty A at a constant 1 kg/s to test infeasibility rejection.
        """
        super().__init__()
        self.invalid = invalid

    def get_overrides(self, name=None):
        """Use the real phase's thermodynamics without overrides.

        Parameters
        ----------
        name : str, optional
            Requested property name.

        Returns
        -------
        dict or None
            Empty mapping or no property override.
        """
        return {} if name is None else None

    def get_solver_state_rates(self, process, phase, **kwargs):
        """Return conservative sequential reaction rates.

        Parameters
        ----------
        process : IntraPhaseProcess
            Native owning phase reference.
        phase : LiquidPhase
            Current component masses [kg].
        **kwargs : dict
            Additional native arguments, unused.

        Returns
        -------
        TransferResult
            Component derivatives [kg/s], with zero external mass supply.
        """
        first = 1. if self.invalid else RATE*phase.mass_j[0]  # [kg/s]; invalid fixture intentionally consumes an empty pool
        second = RATE*phase.mass_j[1]  # [kg/s]
        return TransferResult({StateKey('mass_j', process.phaseref):
                               np.array([-first, first-second, second, 0.])}, {}, 0.)


def vessel(backend, invalid=False):
    """Build a real native vessel with equal-mass conversion and inert carrier.

    Parameters
    ----------
    backend : IntegratorBackend
        Native adaptive solver.
    invalid : bool
        Start with empty A and a deliberately infeasible constant sink.

    Returns
    -------
    MultiPhaseVessel
        Isothermal vessel containing 1 kg carrier and optionally 1 kg A.
    """
    masses = np.array([0. if invalid else 1., 0., 0., 1.])  # [kg]; initially absent intermediates/products
    phase = LiquidPhase(THERMO, mass=masses.sum(), mass_frac=masses/masses.sum(), verbose=False)
    mechanism = SequentialReaction(invalid)
    phase.mechanisms = mechanism
    unit = MultiPhaseVessel(isothermal=True, integrator=backend)
    unit.Phases = phase
    unit.intraphase_processes = [IntraPhaseProcess(PhaseRef('liquid', 0), mechanism)]
    unit.compile_structure()
    return unit


def configured_case(backend, method=None):
    """Declare a fast first-order equal-mass conversion through ordinary JSON.

    Parameters
    ----------
    backend : str
        Native adaptive backend name.
    method : str, optional
        SciPy method; omitted retains BDF.

    Returns
    -------
    dict
        Batch case with 1 g biomass converted to substrate over 2 s at 4/s.
        Formal 1000 g/mol species make mmol/L equal to kg/m3 in this fixture.
    """
    case = json.loads((EXAMPLE/'mechanism.json').read_text())
    case['operation'].update(mode='batch', runtime={'unit':'s','value':2.},
                             recipe={'name':'batch','require_complete_inputs':True})
    case['recipes'] = {'batch':[]}
    case['state']['species']['amounts'] = {'biomass':1., 'substrate':0., 'product':0.}  # [g]
    case['parameters']['rates']['decay'] = RATE  # [1/s]; same analytical coefficient
    case['model']['kinetics'] = {
        'rule_graph':[
            {'identifier':'loss','expression':{'op':'multiply','args':[
                {'parameter':'rates.decay'}, {'concentration':'biomass'}]}},
            {'identifier':'negative','expression':{'op':'multiply','args':[-1.,{'ref':'loss'}]}}],
        'species_mass_rates':{'biomass':'negative','substrate':'loss'},
        'rate_unit':'kg/(m3 s)', 'constants':{}}
    case['rate_provider'] = None
    case['numerical'].update({'backend':backend,'step':{'unit':'s','value':.1},
                         'relative_tolerance':1e-8,'absolute_tolerance':1e-13,
                         'maximum_step':{'unit':'s','value':.1}})  # [s], [-], [kg,m3]; 20 reports and 1e-10 relative initial-mass budget
    if method is not None:
        case['numerical']['method'] = method
    return case


# Public solver paths are checked against physical solutions, not event layout.
@pytest.mark.integration
def test_scipy_sequential_reaction_conserves_and_resolves_transient():
    """B starts empty, grows, and decays; compare with t*exp(-t) exactly."""
    backend = ScipyBackend({'rtol':1e-7, 'atol':1e-12}, inventory_boundaries=True)
    unit = vessel(backend)
    times, states = unit.solve_unit(time_grid=np.linspace(0., 20., 101), verbose=False)  # [s], [kg]; resolves rise and near-depletion
    times = np.asarray(times)  # [s]; native backends may return a list
    remaining = np.exp(-RATE*times)  # [kg]; initial A=1 kg
    intermediate = RATE*times*remaining  # [kg]; equal sequential rate constants
    np.testing.assert_allclose(states[:,0], remaining, atol=1e-7, rtol=1e-6)  # [kg]; tighter than experiment-derived reporting budget
    np.testing.assert_allclose(states[:,1], intermediate, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(states[:,:3].sum(axis=1), 1., atol=1e-9, rtol=0.)  # [kg]; independent total reactive inventory
    assert np.min(states[:,:3]) >= 0.


@pytest.mark.integration
def test_scipy_rejects_consumption_of_empty_inventory():
    """Preserve genuine infeasibility; do not fix failures by zeroing a sink."""
    unit = vessel(ScipyBackend({'rtol':1e-7,'atol':1e-12}, inventory_boundaries=True), invalid=True)
    with pytest.raises(RuntimeError, match='negative rate at empty inventory'):
        unit.solve_unit(time_grid=np.array([0., 1.]), verbose=False)  # [s]


@pytest.mark.assimulo
@pytest.mark.integration
def test_cvode_adaptive_contract_resolves_fast_sequential_reaction():
    """A one-second explicit feasibility horizon must not limit adaptive kinetics."""
    pytest.importorskip('assimulo')
    from PharmaPy.IntegratorBackends import AssimuloBackend
    backend = AssimuloBackend({'rtol':1e-8, 'atol':1e-12}, inventory_boundaries=True)
    unit = vessel(backend)
    times, states = unit.solve_unit(time_grid=np.linspace(0., 5., 26), verbose=False)  # [s], [kg]
    times = np.asarray(times)  # [s]; native backends may return a list
    remaining = np.exp(-RATE*times)  # [kg]; exact unit initial mass
    np.testing.assert_allclose(states[:,0], remaining, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(states[:,1], RATE*times*remaining, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(states[:,:3].sum(axis=1), 1., atol=1e-9, rtol=0.)
    assert backend._inventory_rates is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_cvode_empty_inventory_remains_infeasible():
    """Reject a real invalid sink before the optional solver can wrap its error."""
    pytest.importorskip('assimulo')
    from PharmaPy.IntegratorBackends import AssimuloBackend
    backend = AssimuloBackend({'rtol':1e-8,'atol':1e-12}, inventory_boundaries=True)
    unit = vessel(backend, invalid=True)
    with pytest.raises(RuntimeError, match='negative rate at empty inventory'):
        unit.solve_unit(time_grid=np.array([0.,1.]), verbose=False)  # [s]
    assert backend._inventory_rates is None


@pytest.mark.assimulo
@pytest.mark.unit
def test_cvode_derivative_does_not_alias_reused_native_buffer():
    """Later real vessel evaluations must not mutate CVode's saved derivative."""
    pytest.importorskip('assimulo')
    from PharmaPy.IntegratorBackends import AssimuloBackend
    backend = AssimuloBackend()
    unit = vessel(backend)
    unit.positivity_horizon = .01  # [s]; explicit fixture horizon avoids default-mode feasibility rejection
    state = backend.compile_integrator(unit, verbose=False)  # [kg]
    first = backend._problem.rhs(0., state)  # [kg/s]
    saved = first.copy()  # [kg/s]
    changed = state.copy()  # [kg]
    changed[0] *= .5  # [-]; asymmetric physical state change
    second = backend._problem.rhs(0., changed)  # [kg/s]
    np.testing.assert_array_equal(first, saved)
    assert not np.shares_memory(first, second)
    assert second[0] == pytest.approx(.5*saved[0])


@pytest.mark.integration
@pytest.mark.parametrize('method', [None, 'BDF', 'Radau', 'LSODA', 'RK45', 'RK23', 'DOP853'])
def test_json_method_preserves_analytical_conversion(method):
    """Pass solver choice through normalization, construction and native solve.

    Parameters
    ----------
    method : str or None
        Existing native solver method, or unchanged BDF default.
    """
    assembly = build_bioreactor(configured_case('scipy', method), thermo_path=EXAMPLE/'thermo.json')
    history, = assembly.solve()
    biomass = assembly.phase.name_species.index('biomass')
    substrate = assembly.phase.name_species.index('substrate')
    expected = .001*np.exp(-RATE*history.time)  # [kg]; 1 g initial mass
    np.testing.assert_allclose(history.mass_j_liquid0[:,biomass], expected, atol=1e-10, rtol=1e-6)
    np.testing.assert_allclose(history.mass_j_liquid0[:,[biomass,substrate]].sum(axis=1), .001, atol=1e-12, rtol=0.)
    assert assembly.unit.integrator.method == (method or 'BDF')


@pytest.mark.assimulo
@pytest.mark.integration
def test_json_cvode_preserves_analytical_conversion():
    """The optional solver must work through the user JSON, not manual swapping."""
    pytest.importorskip('assimulo')
    assembly = build_bioreactor(configured_case('assimulo'), thermo_path=EXAMPLE/'thermo.json')
    history, = assembly.solve()
    biomass = assembly.phase.name_species.index('biomass')
    expected = .001*np.exp(-RATE*history.time)  # [kg]; same independent analytical oracle
    np.testing.assert_allclose(history.mass_j_liquid0[:,biomass], expected, atol=1e-10, rtol=1e-6)
    assert assembly.unit.integrator.inventory_boundaries


@pytest.mark.unit
def test_unknown_method_is_rejected_before_solving():
    """A misspelling must not silently select a different numerical algorithm."""
    with pytest.raises(ValueError, match='scipy method must be'):
        build_bioreactor(configured_case('scipy','radau'), thermo_path=EXAMPLE/'thermo.json')


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('with_feed', [False, True])
def test_cvode_preserves_thermal_balance_and_feed_restart(thermal_case, with_feed):
    """Verify reaction heat and a discontinuous cold feed with native CVode.

    Parameters
    ----------
    thermal_case : tuple
        Existing real constant-Cp fixture: 1 kg, 1000 J/(kg K), 300 K,
        0.01 kg substrate, 0.01/s conversion, and 2000 J full-conversion heat.
    with_feed : bool
        Add 0.5 kg of nutrient-free liquid at 270 K after 50 s.
    """
    pytest.importorskip('assimulo')
    document, path = thermal_case
    document['numerical']['backend'] = 'assimulo'
    if with_feed:
        document['operation']['mode'] = 'fed-batch'
        document['recipes']['continuous'] = [{
            'time': 50., 'event_type': 'feed', 'volume_l': .5,
            'concentrations_mmol_l': {}, 'composition': 'nutrient-free',
            'temperature_k': 270.}]  # [s], [L], [mmol/L], [K]; fixture density 1 kg/L
    assembly = build_bioreactor(document, thermo_path=path)
    histories = assembly.solve()
    assert len(histories) == (2 if with_feed else 1)
    substrate = assembly.phase.name_species.index('substrate')
    for segment, history in enumerate(histories):
        remaining = np.exp(-.01*history.time)  # [-]; exact first-order retained fraction
        expected_temperature = 300. + 2.*(1.-remaining)  # [K]; heat/Cp before any feed
        if segment:
            expected_temperature = 290. + (2./1.5)*(1.-remaining)  # [K]; (1*300+0.5*270)/1.5, with reaction heat over 1.5 kg
        np.testing.assert_allclose(np.asarray(history.global_temp).reshape(-1),
                                   expected_temperature, atol=1e-6, rtol=0.)  # [K]; existing independent thermal budget
        np.testing.assert_allclose(history.mass_j_liquid0[:, substrate],
                                   .01*remaining, atol=1e-9, rtol=1e-6)  # [kg]; feed contributes no substrate


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['scipy', pytest.param('assimulo', marks=pytest.mark.assimulo)])
def test_acceptance_never_authorizes_empty_pool_consumption(backend):
    """Even a deliberately permissive 1 kg output budget cannot authorize a sink."""
    limits = {StateKey('mass_j', PhaseRef('liquid', 0)): 1.}  # [kg]; tests separation, not recommended accuracy
    if backend == 'assimulo':
        pytest.importorskip('assimulo')
        from PharmaPy.IntegratorBackends import AssimuloBackend
        integrator = AssimuloBackend({'rtol': 1e-8, 'atol': 1e-12}, inventory_boundaries=True,
                                     inventory_limits=limits)  # [-], [kg]
    else:
        integrator = ScipyBackend({'rtol': 1e-8, 'atol': 1e-12}, inventory_boundaries=True,
                                  inventory_limits=limits)  # [-], [kg]
    unit = vessel(integrator, invalid=True)
    with pytest.raises(RuntimeError, match='negative rate at empty inventory'):
        unit.solve_unit(time_grid=np.array([0., 1.]), verbose=False)  # [s]



@pytest.mark.unit
def test_limits_cannot_be_silently_disabled():
    with pytest.raises(ValueError, match='inventory_boundaries=True'):
        ScipyBackend(inventory_limits={})
