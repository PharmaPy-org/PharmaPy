"""Measured-field predictions through the public build/solve/predict path.

Scope: #361 criterion 5. Analytical checks use the shipped ``generic_batch`` and
``generic_fed_batch`` pathway cases (closed form: ``X = X0 exp(mu t)``,
``N = N0 + fed - (X - X0) / Y``, liquid volume = liquid mass / 1 kg/L). The
F10/F11 checks use the unchanged processed source observations. All runs use
the native SciPy backend; one fixed-step run checks the grid restriction.
Cost: about ten short solves plus the two F10/F11 solves (about 10 s).
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import (build_bioreactor, predict_measurements,
                                  read_observations, sampling_times)


EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'
ECOLI = EXAMPLES / 'ecoli_ye_fed_batch'
# The cases solve with rtol 1e-9; observed closed-form error is below 2e-7.
RTOL = 1e-6  # [-]
NUTRIENT_MW = 100.  # [g/mol], generic thermo.json
# Off-grid times between the 0.1 h reporting steps, plus the 3 h feed time.
OFF_GRID_H = [0.05, 1.234, 2.71]  # [h]


def _generic(name):
    """Load one generic example.

    Parameters
    ----------
    name : str
        Example folder name.

    Returns
    -------
    tuple
        ``(case, mechanism, thermo_path)``.
    """
    folder = EXAMPLES / name / 'inputs'
    return (json.loads((folder / 'case.json').read_text()),
            json.loads((folder / 'mechanism.json').read_text()), folder / 'thermo.json')


def _declaration(unit='mmol/L', side='pre', fields=('nutrient',)):
    """Declare nutrient concentration fields for the generic cases.

    Parameters
    ----------
    unit : str, optional
        Reported concentration unit. Default ``'mmol/L'``.
    side : str, optional
        Event side. Default ``'pre'``.
    fields : Sequence[str], optional
        Field names; each measures the nutrient.

    Returns
    -------
    dict
        Measurement declaration with sampling times in [h].
    """
    return {'schema_version': 1, 'time_unit': 'h', 'fields': {
        name: {'species': 'nutrient', 'quantity': 'concentration', 'unit': unit,
               'event_side': side, 'time_key': 't', 'value_key': 'y'} for name in fields}}


def _data(times_h, fields=('nutrient',)):
    """Build placeholder source observations at the given times.

    Parameters
    ----------
    times_h : Sequence[float]
        Sampling times [h].
    fields : Sequence[str], optional
        Field names.

    Returns
    -------
    dict
        Source mapping; values are zero because only the times are used.
    """
    return {name: {'t': list(times_h), 'y': [0.] * len(times_h)} for name in fields}


def _closed_form_mmol_l(mechanism, times_h, fed_after):
    """Return the analytical nutrient concentration.

    Parameters
    ----------
    mechanism : dict
        Shipped generic pathway mechanism.
    times_h : ndarray
        Sampling times [h].
    fed_after : ndarray of bool
        True where the 3 h feed has been applied.

    Returns
    -------
    ndarray
        Nutrient concentration [mmol/L].
    """
    pathways = mechanism['model']['pathways']
    uptake = mechanism['parameters']['rates']['uptake']['nutrient']  # [mol/(kgDW h)]
    yield_ = pathways['growth_coefficients'][0] / -pathways['exchange_matrix'][0][0]  # [kgDW/mol]
    biomass0 = mechanism['state']['biomass_kg']  # [kgDW]
    nutrient0 = mechanism['state']['species']['amounts']['nutrient']  # [mol]
    biomass = biomass0 * np.exp(uptake * yield_ * times_h)  # [kgDW]
    nutrient = nutrient0 - (biomass - biomass0) / yield_  # [mol]
    liquid_kg = 1. + NUTRIENT_MW / 1000. * (nutrient - nutrient0)  # [kg]; 1 kg initial liquid
    for event in next(iter(mechanism['recipes'].values())):
        fed = event['volume_l'] * event['concentrations_mmol_l']['nutrient'] / 1000.  # [mol]
        nutrient = nutrient + fed * fed_after
        # Carrier-basis feed: volume x 1 kg/L water plus the dissolved solute mass.
        liquid_kg = liquid_kg + fed_after * (event['volume_l'] * 1. + fed * NUTRIENT_MW / 1000.)
    return nutrient * 1000. / liquid_kg  # [mmol/L]; 1 kg/L liquid density


def _predict(name, declaration, data, report=True, case_edit=None):
    """Build, solve and predict one generic case.

    Parameters
    ----------
    name : str
        Example folder name.
    declaration : dict
        Measurement declaration.
    data : dict
        Source observations.
    report : bool, optional
        Pass the sampling times as report times. Default True.
    case_edit : callable, optional
        Function applied to the case before building.

    Returns
    -------
    tuple
        ``(prediction, observations, mechanism, histories)``.
    """
    case, mechanism, thermo = _generic(name)
    if case_edit is not None:
        case_edit(case)
    observations = read_observations(declaration, data)
    assembly = build_bioreactor(case, mechanism, thermo)
    histories = assembly.solve(report_times=sampling_times(observations) if report else None)
    return predict_measurements(assembly, histories, declaration, observations), observations, mechanism, histories


def test_off_grid_samples_match_closed_form_and_are_not_interpolated():
    prediction, observations, mechanism, histories = _predict(
        'generic_batch', _declaration(), _data(OFF_GRID_H))
    times_h = np.asarray(OFF_GRID_H)
    expected = _closed_form_mmol_l(mechanism, times_h, np.zeros(len(times_h)))
    np.testing.assert_allclose(prediction['values']['nutrient'], expected, rtol=RTOL)
    np.testing.assert_array_equal(prediction['time_s']['nutrient'], times_h * 3600.)
    reported = np.concatenate([h.time for h in histories])
    assert all(np.any(reported == t) for t in times_h * 3600.)
    assert prediction['units'] == {'nutrient': 'mmol/L'}


def test_missing_report_times_fail_instead_of_interpolating():
    with pytest.raises(RuntimeError, match='absent from the solution'):
        _predict('generic_batch', _declaration(), _data(OFF_GRID_H), report=False)


def test_report_times_do_not_change_existing_grid_states():
    case, mechanism, thermo = _generic('generic_batch')
    plain = build_bioreactor(case, mechanism, thermo).solve()[0]
    extra = build_bioreactor(case, mechanism, thermo).solve(
        report_times=np.asarray(OFF_GRID_H) * 3600.)[0]
    common = np.isin(extra.time, plain.time)
    np.testing.assert_array_equal(extra.time[common], plain.time)
    np.testing.assert_allclose(extra.mass_j_liquid0[common], plain.mass_j_liquid0,
                               rtol=1e-12, atol=0.)


@pytest.mark.parametrize('side', ['pre', 'post'])
def test_event_side_selects_state_before_or_after_feed(side):
    feed_h = 3.  # [h], generic_fed_batch recipe
    times_h = [2.5, feed_h, 4.25]
    prediction, _, mechanism, _ = _predict(
        'generic_fed_batch', _declaration(side=side), _data(times_h))
    fed_after = np.array([False, side == 'post', True])
    expected = _closed_form_mmol_l(mechanism, np.asarray(times_h), fed_after)
    np.testing.assert_allclose(prediction['values']['nutrient'], expected, rtol=RTOL)


def test_declared_unit_sets_basis_and_field_order_is_preserved():
    times_h = [0.5, 1.5]
    fields = ('nutrient_g', 'nutrient_mmol')
    declaration = _declaration(fields=fields)
    declaration['fields']['nutrient_g']['unit'] = 'g/L'
    prediction, _, _, _ = _predict('generic_batch', declaration, _data(times_h, fields))
    assert list(prediction['values']) == list(fields)
    np.testing.assert_allclose(prediction['values']['nutrient_g'],
                               prediction['values']['nutrient_mmol'] * NUTRIENT_MW / 1000., rtol=1e-14)
    reversed_declaration = copy.deepcopy(declaration)
    reversed_declaration['fields'] = dict(reversed(list(declaration['fields'].items())))
    reordered, _, _, _ = _predict('generic_batch', reversed_declaration,
                                  _data(times_h, fields[::-1]))
    assert list(reordered['values']) == list(fields[::-1])
    for name in fields:
        np.testing.assert_array_equal(reordered['values'][name], prediction['values'][name])


def test_unequal_grids_per_field_keep_source_order():
    data = {'early': {'t': [1.0, 0.25], 'y': [0., 0.]},
            'late': {'t': [5.5, 2.0, 5.5], 'y': [0., 0., 0.]}}
    prediction, observations, mechanism, _ = _predict(
        'generic_batch', _declaration(fields=('early', 'late')), data)
    for name in data:
        times_h = np.asarray(data[name]['t'])
        np.testing.assert_array_equal(observations['time_s'][name], times_h * 3600.)
        expected = _closed_form_mmol_l(mechanism, times_h, np.zeros(len(times_h)))
        np.testing.assert_allclose(prediction['values'][name], expected, rtol=RTOL)


def test_fixed_step_rejects_off_grid_sampling():
    def fixed(case):
        case['numerics'] = {'backend': 'fixed-step', 'step': case['numerics']['step'],
                            **{key: None for key in case['numerics']
                               if key not in {'backend', 'step'}}}
    with pytest.raises(ValueError, match='fixed-step report times'):
        _predict('generic_batch', _declaration(), _data([0.05]), case_edit=fixed)
    prediction, _, mechanism, _ = _predict(
        'generic_batch', _declaration(), _data([0.5, 1.0]), case_edit=fixed)
    assert prediction['values']['nutrient'].shape == (2,)


@pytest.mark.parametrize('edit, message', [
    (lambda d: d['fields']['nutrient'].update(species='oxygen'), 'unknown species'),
    (lambda d: d['fields']['nutrient'].update(unit='furlong'), 'unsupported'),
    (lambda d: d['fields']['nutrient'].update(event_side='during'), 'event_side'),
    (lambda d: d['fields']['nutrient'].update(quantity='mass'), 'quantity'),
    (lambda d: d['fields']['nutrient'].update(extra=1), 'requires exactly'),
    (lambda d: d.update(time_unit='fortnight'), 'time_unit'),
    (lambda d: d.update(schema_version=2), 'schema_version'),
])
def test_invalid_declarations_are_rejected(edit, message):
    declaration = _declaration()
    edit(declaration)
    with pytest.raises(ValueError, match=message):
        _predict('generic_batch', declaration, _data([0.5]))


@pytest.mark.parametrize('data, message', [
    ({'nutrient': {'t': [0.5, float('nan')], 'y': [0., 0.]}}, 'finite'),
    ({'nutrient': {'t': [0.5], 'y': [0., 0.]}}, 'equal nonempty'),
    ({'other': {'t': [0.5], 'y': [0.]}}, 'lack field'),
    ({'nutrient': {'t': [7.0], 'y': [0.]}}, 'within the simulated time grid'),
])
def test_invalid_observations_are_rejected(data, message):
    with pytest.raises(ValueError, match=message):
        _predict('generic_batch', _declaration(), data)


@pytest.mark.parametrize('run, folder', [('F10', 'inputs'), ('F11', 'inputs/F11')])
def test_baseline_predictions_cover_every_source_observation(run, folder):
    declaration = json.loads((ECOLI / 'inputs/observations.json').read_text())
    source = json.loads((ECOLI / f'reference/{run}.json').read_text())['measurements']
    observations = read_observations(declaration, source)
    for name, entry in source.items():
        np.testing.assert_array_equal(observations['values'][name], entry['g_l'])
        np.testing.assert_allclose(observations['time_s'][name],
                                   np.asarray(entry['time_h']) * 3600., rtol=1e-15)
    case = json.loads((ECOLI / folder / 'case.json').read_text())
    mechanism = json.loads((ECOLI / folder / 'mechanism.json').read_text())
    assembly = build_bioreactor(case, mechanism, str(ECOLI / 'inputs/thermo.json'))
    histories = assembly.solve(report_times=sampling_times(observations))
    prediction = predict_measurements(assembly, histories, declaration, observations)
    post_declaration = copy.deepcopy(declaration)
    for field in post_declaration['fields'].values():
        field['event_side'] = 'post'
    post = predict_measurements(assembly, histories, post_declaration, observations)
    names = list(assembly.phase.name_species)
    rows = np.concatenate([h.time for h in histories])
    masses = np.concatenate([h.mass_j_liquid0 for h in histories])
    volumes = np.concatenate([np.asarray(h.vessel_vol).reshape(-1) for h in histories])
    for name, values in prediction['values'].items():
        assert values.shape == observations['values'][name].shape
        assert np.isfinite(values).all() and values.min() >= 0.
        # Independent read of the first recorded state at each sampling time [g/L].
        first = [np.flatnonzero(rows == t)[0] for t in observations['time_s'][name]]
        expected = masses[first, names.index(name)] / volumes[first]
        np.testing.assert_allclose(values, expected, rtol=1e-14)
        # Well-mixed withdrawal leaves concentrations unchanged, so the
        # declared pre-sample side is not a modeling choice here.
        np.testing.assert_allclose(post['values'][name], values, rtol=1e-12)
    # Amount basis for a species that is not first in the phase: the molecular
    # weight must be the measured species' own, read here from thermo.json.
    assert names.index('glucose') > 0
    molar = copy.deepcopy(declaration)
    molar['fields']['glucose']['unit'] = 'mmol/L'
    molar_prediction = predict_measurements(assembly, histories, molar, observations)
    glucose_mw = json.loads((ECOLI / 'inputs/thermo.json').read_text())['glucose']['mw']  # [g/mol]
    np.testing.assert_allclose(molar_prediction['values']['glucose'],
                               prediction['values']['glucose'] * 1000. / glucose_mw, rtol=1e-14)
