"""Source observations and single-counted F10/F11 feed accounting.

The retained records are original processed operating/oxygen observations from
source revision c8d18b53, not generated evidence. These fast input audits preserve
the nominal profile scenario and distinguish it from logged glucose-feed totals.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit


EXAMPLE = Path(__file__).parents[2] / 'examples/bioreactors/ecoli_ye_fed_batch'


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_nominal_glucose_profile_is_one_feed_not_a_second_liquid(run):
    """The glucose log and profile share identity; recipes count only one.

    Parameters
    ----------
    run : str
        F10 or F11 operating record and nominal feed configuration.
    """
    source = json.loads((EXAMPLE / 'reference/operating_records.json').read_text())
    channels = source['experiments'][run]
    assert channels['Feed']['ilabname'] == 'Cumulated_feed_volume_glucose'
    assert source['raw_workbook']['columns']['Y:Z'] == 'Glu (Feed [V])'
    assert source['raw_workbook']['columns']['AA:AB'] == 'Glu (Profile)'
    assert channels['Acid']['vals'] == channels['Medium']['vals'] == []
    folder = EXAMPLE / 'inputs' if run == 'F10' else EXAMPLE / 'inputs' / run
    case = json.loads((folder / 'case.json').read_text())
    model = json.loads((folder / 'mechanism.json').read_text())
    events = model['recipes'][run]
    flows = [event for event in events if event['event_type'] == 'flow']
    np.testing.assert_allclose([e['time'] for e in flows], channels['Profile']['times'], rtol=0., atol=1e-12)
    # 600 g/L glucose stock, paper Table 2 and S3 Constants/LabScale.
    np.testing.assert_allclose([e['volume_l_h'] * 600. for e in flows], channels['Profile']['vals'], rtol=1e-12)
    end_h = case['operation']['runtime']['value']  # [h]
    boundaries_h = np.r_[channels['Profile']['times'], end_h]  # [h]
    nominal_glucose = np.dot(np.diff(boundaries_h), channels['Profile']['vals'])  # [g]
    assert nominal_glucose == pytest.approx(103.5, abs=1e-12)  # [g]; independent profile sum
    assert nominal_glucose / 600. == pytest.approx(0.1725, abs=1e-12)  # [L]
    # Only separately recorded base additions appear as nutrient-free feeds.
    additions = [e for e in events if e['event_type'] == 'feed']
    assert sum(e['volume_l'] for e in additions) == pytest.approx(sum(channels['Base']['vals']))
    assert sum(e['volume_l'] for e in events if e['event_type'] == 'sample') == pytest.approx(
        sum(channels['Sample']['vals']))
    # Actual logged feed volumes differ from the nominal scenario: do not call
    # this discrepancy a second liquid and do not silently replace either record.
    assert not np.isclose(sum(channels['Feed']['vals']), nominal_glucose / 600., rtol=0.01)


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_original_measurements_have_traceable_sources_and_oxygen_is_not_assumed_nonlimiting(run):
    """Preserve source references and the measured low-DOT observations.

    Parameters
    ----------
    run : str
        F10 or F11 experiment identity.
    """
    manifest = json.loads((EXAMPLE / 'inputs/provenance.json').read_text())
    reference = EXAMPLE / 'reference' / f'{run}.json'
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest['experiments'][run]['reference_sha256']
    oxygen = json.loads((EXAMPLE / 'reference/oxygen_observations.json').read_text())
    observed = oxygen['experiments'][run]
    assert len(observed['time_h']) == len(observed['percent_air_saturation'])
    assert min(observed['percent_air_saturation']) < 30.  # [% air saturation]; stated control target
    assert oxygen['source']['revision'] == 'c8d18b53105f695af0d99ee0126544a623b6ff65'
