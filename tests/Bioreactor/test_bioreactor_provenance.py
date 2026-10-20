"""WP1 #351 forward-case adoption without a duplicate measurement hierarchy.

Use original F10/F11 observations for ordering/round-trip checks and the small
analytical generic case for reload-and-rerun. No calibration, weighting or
missing-data Jacobian behavior is claimed. Generated records live in tmp_path.
"""

from copy import deepcopy
import json
from .input_helpers import read_input_json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import (build_bioreactor, make_record, read_record,
                                  record_observations, write_record, native_output_fields,
                                  read_observations, sampling_times, predict_measurements)


ROOT = Path(__file__).parents[2]
ECOLI = ROOT / 'examples/bioreactors/ecoli_ye_fed_batch'


def _record(run='F10'):
    """Assemble a record from the original case and processed observations.

    Parameters
    ----------
    run : str, optional
        F10 or F11 experiment identity.

    Returns
    -------
    dict
        Forward record, with unchanged source units and explicit time [s] adapter.
    """
    folder = ECOLI / 'inputs'
    manifest = read_input_json(folder / 'provenance.json')
    direct = Path(__file__).parent / 'fixtures/ecoli_ye_direct/inputs'
    inputs = direct if run == 'F10' else direct / run
    return make_record(run, read_input_json(inputs / 'case.json'),
                       read_input_json(inputs / 'mechanism.json'),
                       read_input_json(folder / 'thermo.json'),
                       read_input_json(folder / 'observations.json'),
                       read_input_json(ECOLI / 'reference' / f'{run}.json')['measurements'],
                       provenance=manifest['provenance'], parameter_metadata=manifest['parameter_metadata'],
                       model_revision='test-working-tree')


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_original_observations_round_trip_without_reordering(run, tmp_path):
    """JSON key sorting cannot change experiment, field or parameter order.

    Parameters
    ----------
    run : str
        Original F10 or F11 experiment identity.
    tmp_path : Path
        Isolated generated-record destination.
    """
    record = _record(run)
    original = deepcopy(record)
    path = tmp_path / 'record.json'
    write_record(record, path)
    # A content-preserving external tool may sort every mapping's keys.
    path.write_text(json.dumps(read_input_json(path), sort_keys=True))
    loaded = read_record(path)
    restored = record_observations(loaded)
    expected = read_observations(record['inputs']['observations'], record['source_observations'])
    assert loaded['experiment_id'] == run
    assert list(restored['values']) == ['biomass', 'glucose', 'acetate']
    assert loaded['parameter_order'] == record['parameter_order']
    for name in restored['values']:
        np.testing.assert_array_equal(restored['values'][name], expected['values'][name])
        np.testing.assert_array_equal(restored['time_s'][name], expected['time_s'][name])
    assert loaded['provenance']['uncertainty'] == record['provenance']['uncertainty']
    assert loaded['provenance']['calibration']['fitted_values'] is None
    assert record == original


def test_records_reject_changed_inputs_and_incomplete_parameter_provenance(tmp_path):
    """Hashes and complete parameter coverage fail independently.

    Parameters
    ----------
    tmp_path : Path
        Destination that must remain empty after validation failures.
    """
    record = _record()
    record['inputs']['case']['operation']['runtime']['value'] *= 2.  # [h]; changed model input
    with pytest.raises(ValueError, match='input hashes'):
        write_record(record, tmp_path / 'changed.json')
    record = _record()
    name = record['parameter_order'].pop()
    del record['parameters'][name]
    with pytest.raises(ValueError, match='every input parameter'):
        write_record(record, tmp_path / 'incomplete.json')
    assert not list(tmp_path.iterdir())


def test_missing_samples_and_singletons_keep_their_own_grids(tmp_path):
    """Omission never becomes a zero observation or a fabricated weight.

    Parameters
    ----------
    tmp_path : Path
        Isolated sparse-record destination.
    """
    record = _record()
    record['source_observations']['acetate'] = {'time_h': [1.5], 'g_l': [0.16]}  # [h], [g/L]; synthetic singleton
    record = make_record(record['experiment_id'], **{
        'case': record['inputs']['case'], 'mechanism': record['inputs']['mechanism'],
        'thermo': record['inputs']['thermo'], 'declaration': record['inputs']['observations'],
        'data': record['source_observations'], 'provenance': record['provenance'],
        'parameter_metadata': record['parameters'], 'model_revision': record['model']['revision']})
    write_record(record, tmp_path / 'sparse.json')
    restored = record_observations(read_record(tmp_path / 'sparse.json'))
    np.testing.assert_array_equal(restored['time_s']['acetate'], [5400.])  # [s]
    np.testing.assert_array_equal(restored['values']['acetate'], [0.16])  # [g/L]
    assert len(restored['values']['biomass']) > 1
    record['source_observations']['acetate']['g_l'] = [None]
    with pytest.raises(ValueError, match='source observation hash'):
        write_record(record, tmp_path / 'missing.json')


def test_reload_inputs_reruns_the_public_model_and_preserves_predictions(tmp_path):
    """A self-contained generic record reproduces accepted solver predictions.

    Parameters
    ----------
    tmp_path : Path
        Isolated destinations for the record and replay thermo inputs.
    """
    folder = ROOT / 'examples/bioreactors/generic_batch/inputs'
    case, mechanism, thermo = (read_input_json(folder / f'{part}.json')
                               for part in ('case', 'mechanism', 'thermo'))
    declaration = {'schema_version': 1, 'time_unit': 'h', 'fields': {
        'nutrient': {'species': 'nutrient', 'quantity': 'concentration', 'unit': 'mmol/L',
                     'event_side': 'pre', 'time_key': 't', 'value_key': 'y'}}}
    data = {'nutrient': {'t': [0.05, 1.234, 2.71], 'y': [1., 0.9, 0.8]}}  # [h], [mmol/L]; synthetic observations
    provenance = deepcopy(_record()['provenance'])
    provenance.update(sources=[{'role': 'synthetic test', 'uri': 'generic_batch/inputs'}],
                      transformations=[], sharing_conditions='Synthetic PharmaPy qualification fixture')
    metadata = {'rates.uptake.nutrient': {'unit': 'mol/(kgDW h)', 'basis': 'dry biomass',
                                         'bounds': None, 'bounds_status': 'not fitted'}}
    observed = read_observations(declaration, data)
    assembly = build_bioreactor(case, mechanism, folder / 'thermo.json')
    histories = assembly.solve(report_times=sampling_times(observed))
    prediction = predict_measurements(assembly, histories, declaration, observed)
    record = make_record('synthetic-batch', case, mechanism, thermo, declaration, data,
                         provenance=provenance, parameter_metadata=metadata,
                         model_revision='test-working-tree', predictions=prediction,
                         native_outputs=native_output_fields(assembly.unit, histories))
    write_record(record, tmp_path / 'run.json')
    loaded = read_record(tmp_path / 'run.json')
    fields = loaded['native_outputs'][0]['fields']
    np.testing.assert_array_equal(fields['mass_j_liquid0']['values'], histories[0].mass_j_liquid0)
    assert fields['mass_j_liquid0']['unit'] == 'kg'
    assert fields['mass_j_liquid0']['components'] == list(assembly.phase.name_species)
    (tmp_path / 'thermo.json').write_text(json.dumps(loaded['inputs']['thermo']))
    replay = build_bioreactor(loaded['inputs']['case'], loaded['inputs']['mechanism'], tmp_path / 'thermo.json')
    reobserved = record_observations(loaded)
    histories = replay.solve(report_times=sampling_times(reobserved))
    repeated = predict_measurements(replay, histories, declaration, reobserved)
    np.testing.assert_array_equal(repeated['values']['nutrient'], prediction['values']['nutrient'])
    np.testing.assert_array_equal(loaded['predictions']['values']['nutrient'], prediction['values']['nutrient'])
    loaded['predictions']['time_s']['nutrient'][0] += 1.  # [s]; deliberately misaligned sentinel
    with pytest.raises(ValueError, match='grid, unit and shape'):
        write_record(loaded, tmp_path / 'bad-prediction.json')


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_source_cases_reload_and_rerun_native_fields(run, tmp_path):
    """Replay a short pre-feed window of both source cases from JSON alone.

    Parameters
    ----------
    run : str
        Source case identity. All original observations remain in the record;
        this short numerical replay does not claim full-run experimental validation.
    tmp_path : Path
        Isolated destinations for generated record and thermo inputs.
    """
    original = _record(run)
    inputs = original['inputs']
    inputs['case']['operation']['runtime']['value'] = 0.2  # [h]; pre-feed replay window
    inputs['mechanism']['recipes'][run] = []
    thermo_path = tmp_path / 'thermo.json'
    thermo_path.write_text(json.dumps(inputs['thermo']))
    assembly = build_bioreactor(inputs['case'], inputs['mechanism'], thermo_path)
    histories = assembly.solve()
    record = make_record(run, inputs['case'], inputs['mechanism'], inputs['thermo'],
                         inputs['observations'], original['source_observations'],
                         provenance=original['provenance'], parameter_metadata=original['parameters'],
                         model_revision='test-working-tree-short-window',
                         native_outputs=native_output_fields(assembly.unit, histories))
    write_record(record, tmp_path / 'forward.json')
    loaded = read_record(tmp_path / 'forward.json')
    thermo_path.write_text(json.dumps(loaded['inputs']['thermo']))
    replay = build_bioreactor(loaded['inputs']['case'], loaded['inputs']['mechanism'], thermo_path)
    repeated = native_output_fields(replay.unit, replay.solve())
    assert repeated == loaded['native_outputs']
    fields = repeated[0]['fields']
    assert fields['oxygen_uptake_liquid0']['kind'] == 'algebraic'
    assert fields['oxygen_uptake_liquid0']['unit'] == 'g/(L h)'
    assert fields['mass_j_liquid0']['kind'] == 'state'
