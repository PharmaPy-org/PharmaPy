"""A genuine source must permit equilibria below the depletion tolerance."""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import ScipyBackend, _inventory_rates
from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef, StateKey, TransferResult
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase


# Real fixture construction and independent qualification helpers.
class PoolBalance(Mechanism):
    """Synthetic first-order conversion with a constant external mass source."""

    def __init__(self, source, coefficient):
        """Configure the analytical source/conversion mechanism.

        Parameters
        ----------
        source : float
            External mass supply [kg/s].
        coefficient : float
            First-order conversion coefficient [1/s].
        """
        super().__init__()
        self.source = source  # [kg/s]
        self.coefficient = coefficient  # [1/s]

    def get_overrides(self, name=None):
        """Return the native no-override contract.

        Parameters
        ----------
        name : str, optional
            Requested property name.

        Returns
        -------
        dict or None
            Empty collection, or no override for the requested property.
        """
        return {} if name is None else None

    def get_solver_state_rates(self, process, phase, **kwargs):
        """Evaluate the analytical pool/conversion mass balance.

        Parameters
        ----------
        process : IntraPhaseProcess
            Native process identifying the phase state.
        phase : LiquidPhase
            Current component inventories [kg].
        **kwargs : dict
            Additional native mechanism arguments, unused here.

        Returns
        -------
        TransferResult
            Four component mass rates [kg/s]; no auxiliary states or heat.
        """
        uptake = self.coefficient * max(phase.mass_j[0], 0.)  # [kg/s]
        return TransferResult({StateKey('mass_j', process.phaseref):
                               np.array([self.source - uptake, uptake, 0., 0.])}, {}, 0.)


def pool_vessel(initial, source, coefficient):
    """Construct a real native vessel for a linear source/conversion balance.

    Parameters
    ----------
    initial : float
        Initial pool mass [kg].
    source : float
        External pool source [kg/s].
    coefficient : float
        First-order conversion coefficient [1/s].

    Returns
    -------
    MultiPhaseVessel
        Compiled isothermal vessel, with 1 kg inert carrier.
    """
    thermo = Path(__file__).parents[2] / 'tests/fixtures/native_four_species.json'
    masses = np.array([initial, 0., 1., 0.])  # [kg]
    phase = LiquidPhase(thermo, mass=masses.sum(), mass_frac=masses / masses.sum(), verbose=False)
    mechanism = PoolBalance(source, coefficient)
    phase.mechanisms = mechanism
    unit = MultiPhaseVessel(isothermal=True)
    unit.Phases = phase
    unit.intraphase_processes = [IntraPhaseProcess(PhaseRef('liquid', 0), mechanism)]
    unit.compile_structure()
    return unit


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('rate', [-2e-12, -0.5e-12, 2e-12])
def test_empty_inventory_rate_error_budget(rate):
    """Reject infeasibility while retaining bounded roundoff and real supply.

    Parameters
    ----------
    rate : float
        Net inventory rate [kg/s].
    """
    unit = pool_vessel(0., rate, 0.)
    states = unit.create_solver_init_states()  # [kg]
    original = unit.unit_model(0., states, limiter_dt=0.)  # [kg/s], native reusable buffer
    saved = original.copy()  # [kg/s]
    kwargs = dict(constrained=np.array([0]), unit=unit,
                  boundary_tolerance=np.array([1e-12]),  # [kg]
                  rate_tolerance=np.array([1e-12]))  # [kg/s], one-second solve
    if rate < -1e-12:
        with pytest.raises(RuntimeError, match='negative rate at empty inventory'):
            _inventory_rates(0., states, **kwargs)
    else:
        result = _inventory_rates(0., states, **kwargs)
        assert result[0] == (0. if abs(rate) <= 1e-12 else rate)
        assert not np.shares_memory(result, original)
    np.testing.assert_array_equal(original, saved)


@pytest.mark.parametrize('initial',[0.,1.])
@pytest.mark.parametrize('equilibrium',[0.,2.5e-7,2e-6])
def test_small_pool_source_sink_equilibrium(initial,equilibrium):
    """Resolve a physical source/sink equilibrium below boundary tolerance.

    Parameters
    ----------
    initial : float
        Initial pool inventory [kg].
    equilibrium : float
        Analytical steady-state pool inventory [kg].
    """
    coefficient = 400.  # [1/s], stiff compared with the one-second solve
    unit = pool_vessel(initial, coefficient * equilibrium, coefficient)
    unit.integrator = ScipyBackend({'rtol':1e-6,'atol':1e-6,'max_step':.05},
                                  inventory_boundaries=True, stall_calls=2000)
    times, states = unit.solve_unit(time_grid=np.linspace(0.,1.,11), verbose=False)
    assert np.min(states[:,0])>=0.
    assert states[-1,0]==pytest.approx(equilibrium,abs=1e-6)
    np.testing.assert_allclose(states[:, :2].sum(axis=1),initial+400.*equilibrium*times,atol=3e-6,rtol=1e-6)
