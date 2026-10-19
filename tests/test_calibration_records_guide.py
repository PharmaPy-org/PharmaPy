"""Execute the published calibration-record guide and verify its claims.

Scope: ``doc/online_docs/examples/calibration_records.rst``. Its
``testcode`` directives run in document order in one namespace; each
directive's printed output must equal the ``testoutput`` directive that
follows it, so the documented template, capture, reload and rerun cannot
drift from the code. The documented values are then checked
independently: the fitted rate against a ``scipy.optimize.least_squares``
fit of the closed-form residuals (``test_experiment_measurement_guide``),
the missing observation against its position in the saved JSON, and the
schema section against every entry the record format writes and every
admissible finite-choice value. The template-fitting code block is executed
with the template and the guide's decay callback. The ``SimulationExec``
sketch must make the same receiver-qualified calls (found by parsing, not by
substring) as ``tests/test_records_ziegler_case.py``, which must define
every helper and constant the sketch uses. Core lane; under a second.
"""

import ast
import builtins
from collections.abc import Iterator
from pathlib import Path
import re
import textwrap

import numpy as np
import pytest
from scipy.optimize import least_squares

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.Records import (FIELD_KINDS, JACOBIAN_SOURCES,
                              OPTIMIZATION_METHODS, SOURCE_ORIGINS,
                              STEP_DESCRIPTORS, CalibrationRecord,
                              load_template)
import test_experiment_measurement_guide as measurement_guide
import test_records_calibration as decay_case

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'doc/online_docs/examples/calibration_records.rst'
ZIEGLER_TEST = ROOT / 'tests/test_records_ziegler_case.py'
# Keys of the JSON objects that hold user-chosen keys (callback keywords,
# optimizer options and outcome, model-solver settings and transformation
# constants); their contents are not schema entries.
USER_MAPPINGS = ('kwargs', 'options', 'outcome', 'model_solver')
# Receiver-qualified calls (``<object>.<name>(...)``) of the SimulationExec
# sketch that the Ziegler test must make too.
SKETCH_CALLS = {'SetParamEstimation', 'EstimateParams', 'from_estimation',
                'save', 'load', 'estimation_inputs', 'optimizer_inputs',
                'check_rerun', 'rerun'}
# The user-supplied model revision of the executed example.
REVISION = 'calibration_records.rst decay model, v1'


def _blocks(directive: str) -> Iterator[str]:
    """Extract the page's indented directives of one kind.

    Parameters
    ----------
    directive : str
        ``'testcode'``, ``'testoutput'`` or ``'code-block'``.

    Yields
    ------
    str
        Dedented directive content, in document order, with trailing
        whitespace removed from every line and option lines (``:name:``)
        dropped.
    """
    text = PAGE.read_text(encoding='utf-8').expandtabs()
    pattern = rf"^( *)\.\. {directive}::.*\n"
    for match in re.finditer(pattern, text, re.MULTILINE):
        indentation = len(match.group(1))
        lines = []
        for line in text[match.end():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                break
            lines.append(line.rstrip())
        block = textwrap.dedent("\n".join(lines)).strip("\n")
        yield "\n".join(line for line in block.splitlines()
                        if not re.match(r"^:\w[\w-]*:", line))


CODE = tuple(_blocks('testcode'))
OUTPUT = tuple(_blocks('testoutput'))


def _section(title: str) -> str:
    """Return the text of one page section.

    Parameters
    ----------
    title : str
        Section title.

    Returns
    -------
    str
        Text from the title to the next title of the same level.
    """
    text = PAGE.read_text(encoding='utf-8')
    start = text.index(f"\n{title}\n{'-' * len(title)}\n")
    following = re.search(r"\n[^\n]+\n-{3,}\n", text[start + 1:])
    return text[start:start + 1 + following.start()] if following \
        else text[start:]


def _qualified_calls(tree: ast.AST) -> set:
    """Collect the names of receiver-qualified calls.

    Parameters
    ----------
    tree : ast.AST
        Parsed source.

    Returns
    -------
    set of str
        ``name`` of every call written as ``<expression>.name(...)``.
    """
    return {node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)}


def _rerun_callbacks(tree: ast.AST) -> set:
    """Collect the attribute names passed as first argument of ``.rerun``.

    Parameters
    ----------
    tree : ast.AST
        Parsed source.

    Returns
    -------
    set of str
        ``attr`` of each ``<object>.rerun(<expression>.attr, ...)`` call.
    """
    return {node.args[0].attr for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'rerun' and node.args
            and isinstance(node.args[0], ast.Attribute)}


def _keys(data, parent=None) -> set:
    """Collect the schema keys of a record's JSON data.

    Parameters
    ----------
    data : object
        ``CalibrationRecord.to_dict()`` data or a nested entry.
    parent : str, optional
        Key holding ``data``.

    Returns
    -------
    set of str
        Keys of every nested JSON object, except the contents of
        ``USER_MAPPINGS`` and of transformation constants.
    """
    keys = set()
    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key)
            if key in USER_MAPPINGS or (parent == 'transformations'
                                        and key == 'parameters'):
                continue
            keys |= _keys(value, key)
    elif isinstance(data, list):
        for item in data:
            keys |= _keys(item, parent)
    return keys


def test_page_has_matching_code_and_output_blocks():
    """Keep an empty extraction from turning the checks vacuous."""
    assert len(CODE) == len(OUTPUT) == 3


def test_documented_examples_output_and_values(capsys):
    namespace = {'__name__': '__calibration_records_page__'}
    for source, expected in zip(CODE, OUTPUT):
        exec(compile(source, f"{PAGE.name}:testcode", 'exec'), namespace)
        printed = "\n".join(
            line.rstrip()
            for line in capsys.readouterr().out.strip("\n").splitlines())
        assert printed == expected

    # The documented rate is the independently fitted one, rounded.
    reference = least_squares(
        measurement_guide._standardized, [measurement_guide.SEED_RATE],
        xtol=measurement_guide.REFERENCE_TOL,
        ftol=measurement_guide.REFERENCE_TOL,
        gtol=measurement_guide.REFERENCE_TOL)
    documented = f"k = {reference.x[0]:.4f} 1/s"
    assert documented in OUTPUT[1] and documented in OUTPUT[2]

    # The cold run's 1.5 s sample is the missing one: null in the file,
    # NaN after loading, and nowhere else.
    saved = namespace['saved']
    cold = saved['experiments'][1]
    assert cold['name'] == 'cold'
    assert cold['measurements'][0]['values'] == [0.785, None, 0.220]
    values = namespace['loaded'].experiments['cold'].measurements[
        'c_A'].values
    assert np.isnan(values).tolist() == [False, True, False]
    estimator, loaded = namespace['estimator'], namespace['loaded']
    np.testing.assert_array_equal(loaded.fitted_values,
                                  estimator.params_convg)
    assert namespace['rebuilt'].params_convg[0] == pytest.approx(
        loaded.fitted_values[0], rel=namespace['replay_rtol'], abs=0)
    assert namespace['template'] == load_template()
    # The user-supplied model revision survives the file round trip.
    assert saved['model']['revision'] == REVISION
    assert loaded.model.revision == REVISION
    assert loaded.model == namespace['record'].model


def test_schema_section_documents_every_entry():
    namespace = {'__name__': '__calibration_records_page__'}
    for source in CODE[:2]:
        exec(compile(source, f"{PAGE.name}:testcode", 'exec'), namespace)
    captured = namespace['record'].to_dict()  # solver and covariance
    keys = _keys(load_template().to_dict()) | _keys(captured)
    # A weight matrix is not allowed with declared uncertainties, so the
    # documented examples cannot hold one; its keys are written out.
    keys |= {'index', 'columns', 'values'}
    assert {'solver', 'derivatives', 'jacobian', 'dx_finitediff',
            'include_mse', 'measurements', 'identity'} <= keys
    schema = _section('Record schema')
    assert sorted(key for key in keys if f"``{key}``" not in schema) == []
    choices = (SOURCE_ORIGINS + FIELD_KINDS + OPTIMIZATION_METHODS
               + JACOBIAN_SOURCES + STEP_DESCRIPTORS)
    assert [value for value in choices
            if f"``'{value}'``" not in schema] == []


def test_template_include_is_the_shipped_template():
    (include,) = re.findall(r"\.\. literalinclude:: (\S+)",
                            PAGE.read_text(encoding='utf-8'))
    shipped = ROOT / 'PharmaPy/data/records/calibration_record_template.json'
    assert (PAGE.parent / include).resolve() == shipped.resolve()


def test_template_fitting_block_runs_with_the_template():
    fitting, _ = _blocks('code-block')
    filled = load_template()
    namespace = {'filled': filled, 'func': decay_case.decay,
                 'ParameterEstimation': ParameterEstimation,
                 'CalibrationRecord': CalibrationRecord}
    exec(compile(fitting, f"{PAGE.name}:code-block", 'exec'), namespace)
    fitted = namespace['fitted']
    assert fitted.fitted_values is not None
    assert fitted.model == filled.model
    assert fitted.sources == filled.sources
    assert fitted.transformations == filled.transformations
    assert fitted.description == filled.description
    (parameter,) = fitted.parameters
    (template_parameter,) = filled.parameters
    assert (parameter.name, parameter.units, parameter.seed,
            parameter.lower, parameter.upper) == (
        template_parameter.name, template_parameter.units,
        template_parameter.seed, template_parameter.lower,
        template_parameter.upper)
    # As documented, parameter descriptions are not carried.
    assert template_parameter.description is not None
    assert parameter.description is None
    # The stored state precedes the measurement predictions.
    assert fitted.fields[0] == filled.fields[0]
    assert [field.identity.kind for field in fitted.fields[1:]] == [
        'measurement_prediction'] * 2


def test_simulation_exec_sketch_mirrors_the_ziegler_test():
    _, sketch = _blocks('code-block')
    sketch_tree = ast.parse(sketch, f"{PAGE.name}:code-block")
    test_tree = ast.parse(ZIEGLER_TEST.read_text(encoding='utf-8'))
    assert SKETCH_CALLS <= _qualified_calls(sketch_tree)
    assert SKETCH_CALLS <= _qualified_calls(test_tree)
    assert _rerun_callbacks(sketch_tree) == {'paramest_wrapper'}
    assert _rerun_callbacks(test_tree) == {'paramest_wrapper'}

    # Every free name of the sketch is a helper, constant or import of the
    # Ziegler test module.
    stored = {node.id for node in ast.walk(sketch_tree)
              if isinstance(node, ast.Name)
              and isinstance(node.ctx, ast.Store)}
    free = {node.id for node in ast.walk(sketch_tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            } - stored - set(dir(builtins))
    defined = set()
    for node in test_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined |= {target.id for target in node.targets
                        if isinstance(target, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(alias.asname or alias.name).split('.')[0]
                        for alias in node.names}
    assert {'stage1_seeds', 'build_reactor', 'provenance'} <= free
    assert sorted(free - defined) == []
