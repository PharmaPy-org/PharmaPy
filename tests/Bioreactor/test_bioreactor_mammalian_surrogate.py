"""Synthetic biologics trajectories and surrogate substitutions through native units."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Bioreactors import build_bioreactor

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT/'examples/bioreactors/mammalian_surrogate_fed_batch'


@pytest.fixture(scope='module')
def demonstration(tmp_path_factory):
    namespace = {}
    for index, cell in enumerate(c for c in json.loads((EXAMPLE/'workflow.ipynb').read_text())['cells']
                                 if c['cell_type'] == 'code'):
        exec(compile(''.join(cell['source']), f'mammalian:{index}', 'exec'), namespace)
        if index == 0:
            namespace['EXAMPLE'] = EXAMPLE
            namespace['EXPORT_DIR'] = tmp_path_factory.mktemp('mammalian_surrogate')
    return namespace


def test_notebook_exports_cells_product_and_surrogate_errors(demonstration):
    ns = demonstration
    for label in ['mechanistic', 'affine', 'polynomial']:
        values = ns['tables'][label]
        assert np.isfinite(values).all() and values.min() >= 0.
        assert values[-1,2] > values[0,2] and values[-1,4] > 0.
        assert (ns['EXPORT_DIR']/f'{label}.csv').exists()
    for label, record in ns['report']['runs'].items():
        assert record['maximum_normalized_trajectory_error'] < ns['settings']['maximum_normalized_trajectory_error']
        assert record['inside_validity_domain']
        assert (ns['EXPORT_DIR']/f'{label}_mechanism.json').exists()
    for extension in ('png','svg'):
        assert (ns['EXPORT_DIR']/f'trajectories.{extension}').stat().st_size > 1000
    assert any(sum(p)>1 for p in ns['providers']['polynomial']['powers'])


@pytest.mark.parametrize('label', ['affine','polynomial'])
def test_surrogates_reject_extrapolation_during_native_run(demonstration, label):
    ns = demonstration
    model = deepcopy(ns['mechanism'])
    model['rate_provider'] = deepcopy(ns['providers'][label])
    model['rate_provider']['validity_domain']['s'] = [50.,60.]
    with pytest.raises(ValueError, match='outside validity domain'):
        build_bioreactor(ns['case'], model, ns['thermo']).solve(verbose=False)


def test_mammalian_balances_against_independent_radau(demonstration):
    ns = demonstration
    # Independent amount balances in grams, litres and days. This checks the
    # declared reduced model, not elemental closure or real culture physiology.
    initial = [1.80156,.3,0.,0.,1.]
    predictions = []
    for segment in ns['histories']['mechanistic']:
        times = segment.time/86400.
        flow = 0. if times[0] < 1. else .03
        def rhs(time, y):
            glucose, viable, dead, product, volume = y
            concentration = glucose/volume*1000./180.156
            uptake = .5*concentration/(1.+concentration)*viable
            death = .01*viable
            return [-uptake+flow*5., .4*uptake-death, death, .01*viable, flow]
        solution = solve_ivp(rhs,(times[0],times[-1]),initial,t_eval=times,
                             method='Radau',rtol=1e-11,atol=1e-13)
        assert solution.success
        initial = solution.y[:,-1]
        amounts = solution.y.T
        volume = amounts[:,4]
        predictions.append(np.column_stack((times,amounts[:,0]/volume,
            amounts[:,1]/volume/.3,amounts[:,2]/volume/.3,amounts[:,3]/volume,volume)))
    np.testing.assert_allclose(ns['tables']['mechanistic'],np.vstack(predictions),rtol=2e-6,atol=1e-9)


def test_mammalian_refinement_preserves_predictions(demonstration):
    ns = demonstration
    case = deepcopy(ns['case'])
    case['numerics']['relative_tolerance'] *= .1
    case['numerics']['absolute_tolerance'] *= .1
    case['numerics']['maximum_step']['value'] *= .5
    assembly = build_bioreactor(case, ns['mechanism'], ns['thermo'])
    refined = ns['table'](assembly,assembly.solve(verbose=False))
    np.testing.assert_allclose(refined, ns['tables']['mechanistic'],rtol=2e-6,atol=1e-9)
