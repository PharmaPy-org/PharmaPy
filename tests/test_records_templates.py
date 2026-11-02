"""Calibration-record templates shipped with PharmaPy (#351).

Scope: ``PharmaPy.Records.template_text`` and ``load_template`` read the
package-data template
``PharmaPy/data/records/calibration_record_template.json``. It must load
as a valid unfitted record (``solver``, ``covariance`` and
``weight_matrix`` null), save back byte for byte, hold the sections that
``doc/online_docs/examples/calibration_records.rst`` describes (one
experiment with two measurements, a missing observation, per-sample
uncertainties, callback arguments, a source, a transformation, a
bounded parameter, a model revision and a stored state) and be registered
as package data. Its numbers are the synthetic decay example of the user
guide (``test_records_calibration.py``); expected values are written out by
hand or come from the closed-form model. Core lane; milliseconds.
"""

from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.Records import (TEMPLATE_NAMES, CalibrationRecord,
                              load_template, template_text)
import test_records_calibration as case

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'PharmaPy' / 'data' / 'records' / \
    'calibration_record_template.json'
# The guide's hot run, written out by hand: sampling times [s], c_A
# [mol/L] with its standard deviation [mol/L], and T [K] at 2 s with its
# standard deviation [K]; the template adds a missing T sample at 4 s.
CA_TIME = [0.0, 1.0, 2.0, 4.0]  # [s]
CA_VALUES = [2.010, 1.200, 0.750, 0.265]  # [mol/L]
CA_STD = [0.02, 0.02, 0.02, 0.02]  # [mol/L]
T_TIME = [2.0, 4.0]  # [s]
T_STD = [0.5, 0.5]  # [K]
INITIAL = 2.0  # [mol/L], initial c_A of the hot run
RISE = 20.0  # [K*L/mol]
TEMP0 = 300.0  # [K]
SEED_RATE = 0.3  # [1/s]
# The stored state and the closed form below evaluate the same IEEE
# expression c_A0 * exp(-k t); one ulp of O(1) values bounds any
# difference in evaluation order.
ROUNDOFF_RTOL = 2 * np.finfo(float).eps  # [-]


def test_template_text_is_the_shipped_file():
    assert TEMPLATE_NAMES == ('calibration_record',)
    assert template_text() == TEMPLATE.read_text(encoding='utf-8')
    assert template_text('calibration_record') == template_text()


def test_template_round_trips_byte_for_byte(tmp_path):
    template = load_template()
    assert template.to_json() == template_text()
    path = tmp_path / 'case.json'
    template.save(path)
    # save writes '\n' line ends on every platform; template_text reads
    # with universal newlines, so a checkout with CRLF line ends compares
    # equal too.
    assert path.read_bytes() == template_text().encode('utf-8')
    assert CalibrationRecord.load(path) == template


def test_template_holds_every_documented_section():
    template = load_template()
    assert template.case == 'TEMPLATE-case'
    assert template.solver is None
    assert template.fitted_values is None and template.covariance is None
    assert template.weight_matrix is None
    assert template.output_names == ('c_A', 'T')

    assert list(template.experiments) == ['TEMPLATE-run']
    experiment = template.experiments['TEMPLATE-run']
    assert experiment.args == (INITIAL,)
    assert dict(experiment.kwargs) == {'rise_k_l_mol': RISE,
                                       'temp0_k': TEMP0}
    assert list(experiment.measurements) == ['c_A', 'T']
    concentration = experiment.measurements['c_A']
    assert (concentration.field, concentration.units, concentration.basis,
            concentration.x_units) == (0, 'mol/L', 'molar concentration',
                                       's')
    assert concentration.x.tolist() == CA_TIME
    assert concentration.values.tolist() == CA_VALUES
    assert concentration.uncertainty.tolist() == CA_STD
    temperature = experiment.measurements['T']
    assert (temperature.field, temperature.units, temperature.basis,
            temperature.x_units) == (1, 'K', 'temperature', 's')
    assert temperature.x.tolist() == T_TIME
    assert temperature.values[0] == case.HOT_T[0]
    assert np.isnan(temperature.values[1])  # the missing 4 s sample
    assert temperature.uncertainty.tolist() == T_STD
    assert '"values": [\n            325.6,\n            null\n' in \
        template_text()

    (source,) = template.sources
    assert source.origin == 'synthetic'
    assert 'illustrative template, not data' in source.sharing
    (transformation,) = template.transformations
    assert transformation.source == source.name
    assert transformation.fields == ('c_A', 'T')
    assert transformation.parameters == {'factor': {'value': 1.0,
                                                    'units': '-'}}
    (parameter,) = template.parameters
    assert (parameter.name, parameter.units, parameter.seed,
            parameter.lower, parameter.upper, parameter.optimized,
            parameter.fitted) == ('k', '1/s', SEED_RATE, 0.0, None, True,
                                  None)
    assert template.model.pharmapy_version is None

    (state,) = template.fields
    assert (state.experiment, state.identity.kind, state.name,
            state.identity.units, state.x_units) == (
        'TEMPLATE-run', 'state', 'c_A', 'mol/L', 's')
    assert state.x.tolist() == CA_TIME
    expected = [INITIAL * np.exp(-SEED_RATE * time)
                for time in CA_TIME]  # [mol/L]
    np.testing.assert_allclose(state.values, expected, rtol=ROUNDOFF_RTOL,
                               atol=0)


def test_template_placeholders_are_marked():
    template = load_template()
    texts = [template.case, template.description, template.model.name,
             template.model.revision, template.model.description,
             template.transformations[0].description,
             template.parameters[0].description,
             template.fields[0].identity.description]
    source = template.sources[0]
    texts += [source.sharing, source.description, source.citation,
              source.uri, source.revision]
    texts += list(template.experiments)
    assert all(text.startswith('TEMPLATE') for text in texts)


def test_template_experiments_fit_the_guide_model():
    # The template's callback association matches the guide's decay
    # callback, so the filled-in record can be fitted unchanged.
    template = load_template()
    estimator = ParameterEstimation(case.decay,
                                    **template.estimation_inputs())
    assert estimator.num_data_total == len(CA_VALUES) + 1
    # (model - data) / sigma of the first c_A sample at the seed [-]
    residuals = estimator.get_objective([SEED_RATE], out_array=True)
    assert residuals[0] == pytest.approx((INITIAL - CA_VALUES[0])
                                         / CA_STD[0], rel=ROUNDOFF_RTOL)


def test_unknown_template_names_are_rejected():
    with pytest.raises(ValueError, match=r"Template name must be one of "
                                         r"\('calibration_record',\); got "
                                         r"'calibration'"):
        load_template('calibration')


def test_templates_are_package_data():
    tomllib = pytest.importorskip('tomllib')  # Python >= 3.11
    with open(ROOT / 'pyproject.toml', 'rb') as file:
        patterns = tomllib.load(file)['tool']['setuptools'][
            'package-data']['PharmaPy']
    relative = TEMPLATE.relative_to(ROOT / 'PharmaPy').as_posix()
    assert any(fnmatch(relative, pattern) for pattern in patterns)
