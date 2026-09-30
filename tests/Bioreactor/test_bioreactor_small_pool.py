"""A genuine source must permit equilibria below the depletion tolerance."""

from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.IntegratorBackends import SciPyBackend


@pytest.mark.parametrize('initial',[0.,1.])
@pytest.mark.parametrize('equilibrium',[0.,2.5e-7,2e-6])
def test_small_pool_source_sink_equilibrium(initial,equilibrium):
    calls=0
    def rhs(time,state,**kwargs):
        nonlocal calls
        calls+=1
        assert calls<2000, 'Artificial boundary switching prevents integration progress'
        uptake=400.*max(state[0],0.)
        return np.array([400.*equilibrium-uptake,uptake])
    y0=np.array([initial,0.])
    collection=SimpleNamespace(slices={'pool':slice(0,1),'consumed':slice(1,2)},
        states={'pool':SimpleNamespace(limit_negative_inventory=True),
                'consumed':SimpleNamespace(limit_negative_inventory=False)},pack=lambda state:y0.copy())
    unit=SimpleNamespace(elapsed_time=0.,solver_state_collection=collection,
        _initial_solver_state={},create_solver_init_states=lambda:y0.copy(),unit_model=rhs)
    backend=SciPyBackend({'rtol':1e-6,'atol':1e-6,'max_step':.05})
    backend._compiled=True
    times,states=backend.fast_solve(unit,time_grid=np.linspace(0.,1.,11))
    assert np.min(states[:,0])>=0.
    assert states[-1,0]==pytest.approx(equilibrium,abs=1e-6)
    np.testing.assert_allclose(states.sum(axis=1),initial+400.*equilibrium*times,atol=3e-6,rtol=1e-6)
