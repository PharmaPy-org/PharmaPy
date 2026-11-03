"""Native adaptive solvers on analytical sequential reactions and empty pools.

Real phases, mechanisms and vessels isolate generic depletion handling from
experimental parameter assumptions. CVode cases require the real Assimulo lane.
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef, StateKey, TransferResult
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase


# Synthetic first-order networks have closed-form, independently derived states.
THERMO = Path(__file__).parents[1] / 'fixtures/native_four_species.json'
RATE = 4.  # [1/s]; exhausts a pool within the unrelated default 1 s limiter horizon


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



# Native adaptive solver and acceptance contracts.
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
