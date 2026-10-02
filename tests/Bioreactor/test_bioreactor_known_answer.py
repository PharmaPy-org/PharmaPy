"""Analytic projection and piecewise balances, independent of the production solver."""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor

EXAMPLE = Path(__file__).parents[2] / 'tests/Bioreactor/fixtures/reconciliation/analytic_balances'


def load(variant):
    folder = EXAMPLE / 'inputs' / variant
    return (json.loads((folder/'case.json').read_text()),
            json.loads((folder/'mechanism.json').read_text()), folder/'thermo.json')


def reference(histories, initial, rate, feed_concentration=2.):
    """Exact continuous balances for the documented feeds and material events.

    Columns: nutrient mmol, product g, million viable cells, volume L, IVCD.
    No production rule graph, optimizer, converter or event handler is used.
    """
    state = np.array([initial, 0., 1., 1., 0.])
    predictions = []
    flow = .1
    for history in histories:
        start = history.time[0]/86400.
        if np.isclose(start, .5):
            flow = .2
        elif np.isclose(start, .75):
            state[:3] *= 1. - .1/state[3]
            state[3] -= .1
        elif np.isclose(start, 1.):
            flow = 0.
            state[0] += .3
            state[3] += .1
        rows = []
        for time in history.time/86400.:
            dt = time-start
            supplied = flow*feed_concentration*dt
            consumed = min(state[0]+supplied, rate*state[2]*dt)
            volume = state[3]+flow*dt
            exposure = (np.log(volume/state[3])/flow if flow else dt/state[3])
            rows.append([state[0]+supplied-consumed, state[1]+.1*consumed, state[2], volume,
                         state[4]+state[2]/1000.*exposure])
        state = np.array(rows[-1])
        predictions.append(np.array(rows))
    return predictions


def observed(assembly, history):
    index = list(assembly.phase.name_species).index('nutrient')
    return np.column_stack((history.mass_j_liquid0[:, index]*10000.,
                            history.product_g_liquid0, history.viable_cells_million_liquid0,
                            np.asarray(history.vessel_vol).reshape(-1)*1000.,
                            history.ivcd_million_cell_day_per_ml_liquid0))


@pytest.mark.parametrize('variant, targets, expected', [
    ('consistent', (2., 2.), 2.), ('conflicting', (4., 2.), 3.),
])
def test_unique_reconciliation_projection(variant, targets, expected):
    assembly = build_bioreactor(*load(variant))
    parameters = load(variant)[1]['parameters']['rates']
    assert (parameters['uptake_target'], parameters['product_target']) == targets
    solution = assembly.mechanism._solve(dict(zip(('uptake', 'secretion'), targets)))
    np.testing.assert_allclose(solution.fluxes, expected, atol=1e-9, rtol=0.)
    matrix = np.array([[1., -1., 0.], [0., 1., -1.]])
    np.testing.assert_allclose(matrix @ solution.fluxes, 0., atol=1e-10)
    # Analytic KKT stationarity along the sole feasible direction.
    assert abs(2.*expected-sum(targets)) < 1e-12


@pytest.mark.parametrize('variant, initial, rate', [
    ('consistent', 10., 2.), ('conflicting', 10., 3.), ('depletion', .12, 3.),
])
def test_fed_batch_matches_independent_piecewise_solution(variant, initial, rate):
    assembly = build_bioreactor(*load(variant))
    histories = assembly.solve()
    expected = reference(histories, initial, rate, 0. if variant == 'depletion' else 2.)
    for history, exact in zip(histories, expected):
        actual = observed(assembly, history)
        assert np.isfinite(actual).all() and actual.min() >= -1e-9
        np.testing.assert_allclose(actual[:, :4], exact[:, :4], atol=1e-10, rtol=1e-10)
        # The explicit fixed-step exposure integral has first-order quadrature error.
        np.testing.assert_allclose(actual[:, 4], exact[:, 4], atol=2e-6, rtol=0.)
        # Nutrient/product interconversion conserves their formal molecular amount.
        np.testing.assert_allclose(actual[:, 0]+actual[:, 1]*10.,
                                   exact[:, 0]+exact[:, 1]*10., atol=1e-10, rtol=1e-10)
    if variant == 'depletion':
        first = observed(assembly, histories[0])
        assert first[-1, 0] < 1e-8
        resumed = observed(assembly, histories[-1])
        assert resumed[-1, 1] > resumed[0, 1]
        assert resumed[-1, 0] < 1e-8
        assert histories[0].time[np.flatnonzero(first[:, 0] < 1e-10)[0]]/86400. == pytest.approx(.04)
        assert histories[-1].time[np.flatnonzero(resumed[:, 0] < 1e-10)[0]]/86400. == pytest.approx(1.11)


def test_known_answer_exposure_converges_with_step_refinement():
    errors = []
    for step in (.01, .005):
        case, config, thermo = load('conflicting')
        case['numerics']['step']['value'] = step
        assembly = build_bioreactor(case, config, thermo)
        histories = assembly.solve()
        exact = reference(histories, 10., 3.)[-1][-1, 4]
        errors.append(abs(observed(assembly, histories[-1])[-1, 4]-exact))
    assert errors[1] < .55*errors[0]
