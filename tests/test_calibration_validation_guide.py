"""Execute the calibration-validation guide and verify its claims (#352).

Scope: ``doc/online_docs/examples/calibration_validation.rst``. Its
``testcode`` directives run in document order in one namespace; each
directive's printed output must equal the ``testoutput`` directive that
follows it, so the documented workflow (fit, convergence, derivative check,
identifiability, record and rerun) cannot drift from the code. The printed
values are then checked independently of the production code: the fitted
rate against a ``scipy.optimize.least_squares`` fit of the closed-form
residuals (``test_experiment_measurement_guide``), the derivative check
against the forward-difference truncation bracket and the standard error
against the closed-form sensitivities derived in
``test_calibration_validation_decay``. The page must state that its fits are
synthetic or stand-in fits, point to the three validation modules (which
must exist), list what the December demonstrators #358 and #361 must do and
name the #355 hand-off, and be listed in the examples index. Core lane;
under a second.
"""

from pathlib import Path
import re
import textwrap

import numpy as np
import pytest
from scipy.optimize import least_squares

import test_calibration_validation_decay as validation
import test_experiment_measurement_guide as measurement_guide
import test_records_calibration as case

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'doc/online_docs/examples/calibration_validation.rst'
INDEX = ROOT / 'doc/online_docs/examples/index.rst'
# Directives of the page: fit, convergence, derivatives, identifiability,
# #355 hand-off, record.
NUM_BLOCKS = 6  # [-]
VALIDATION_MODULES = ('tests/test_calibration_validation_decay.py',
                      'tests/test_calibration_validation_ziegler.py',
                      'tests/test_calibration_validation_crystallizer.py')
# Steps the December demonstrators must take, one bullet each.
DEMONSTRATOR_STEPS = ('run this workflow on the real case',
                      'check convergence and the recorded stopping '
                      'tolerances',
                      'reordering the experiments and the measurements',
                      'aligned with the layout',
                      'repeated fits are identical',
                      'observations are unchanged after the uncertainty',
                      'check derivatives with steps matched to the solver',
                      'passing ``bounds`` for parameters with a restricted '
                      'valid domain',
                      'report identifiability with declared measurement '
                      'uncertainties',
                      'record the fit with ``CalibrationRecord``',
                      'link the open defects',
                      'add the case to this validation suite')


def _blocks(directive):
    """Extract the page's indented directives of one kind.

    Parameters
    ----------
    directive : str
        ``'testcode'`` or ``'testoutput'``.

    Returns
    -------
    tuple of str
        Dedented directive contents in document order, with trailing
        whitespace removed from every line and option lines (``:name:``)
        dropped, as in ``test_calibration_records_guide``.
    """
    text = PAGE.read_text(encoding='utf-8').expandtabs()
    blocks = []
    for match in re.finditer(rf"^( *)\.\. {directive}::.*\n", text,
                             re.MULTILINE):
        indentation = len(match.group(1))
        lines = []
        for line in text[match.end():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                break
            lines.append(line.rstrip())
        block = textwrap.dedent("\n".join(lines)).strip("\n")
        blocks.append("\n".join(line for line in block.splitlines()
                                 if not re.match(r"^:\w[\w-]*:", line)))
    return tuple(blocks)


def _text():
    """Return the page text with line breaks folded to spaces.

    Returns
    -------
    str
        Page text, whitespace runs collapsed to one space.
    """
    return re.sub(r"\s+", " ", PAGE.read_text(encoding='utf-8'))


CODE = _blocks('testcode')
OUTPUT = _blocks('testoutput')


def test_page_has_matching_code_and_output_blocks():
    """Keep an empty extraction from turning the checks vacuous."""
    assert len(CODE) == len(OUTPUT) == NUM_BLOCKS


def test_documented_workflow_output_and_values(capsys):
    namespace = {'__name__': '__calibration_validation_page__'}
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
    assert documented in OUTPUT[0] and documented in OUTPUT[-1]

    # The documented relative error lies in the derived forward-difference
    # bracket (test_default_forward_differences_show_their_truncation_at_
    # the_fit), and is O(1e-3), not O(1).
    rate = namespace['params'][0]  # [1/s]
    check = namespace['check']
    leading, bound = validation.forward_difference_error(rate)  # [s], [s]
    reference_error = validation.reference_error_bound(check)  # [s]
    spread = np.linalg.norm(bound + reference_error)  # [s]
    exact_norm = np.linalg.norm(validation.exact_rate_row(rate))  # [s]
    low = (np.linalg.norm(leading) - spread) / (
        exact_norm + np.linalg.norm(reference_error))  # [-]
    high = (np.linalg.norm(leading) + spread) / (
        exact_norm - np.linalg.norm(reference_error))  # [-]
    assert low <= check.relative_error[0] <= high

    # Standard error: sqrt(s2 / ||J||**2) with the forward-difference row
    # norm bracketed around the closed form.
    report = namespace['report']
    residuals, _ = validation.deriv.closed_form(
        validation.rise_and_rate(rate))  # [-]
    mse = residuals @ residuals / (validation.NUM_OBSERVED - 1)  # [-]
    norm_low, norm_high = validation.forward_row_norm_bracket(rate)  # [s]
    assert (np.sqrt(mse) / norm_high <= report.standard_errors[0]
            <= np.sqrt(mse) / norm_low)
    assert f"s2 = {mse:.3f} [-]" in OUTPUT[3]
    np.testing.assert_array_equal(namespace['loaded'].fitted_values,
                                  namespace['params'])
    assert namespace['rerun'].params_convg[0] == pytest.approx(
        rate, rel=case.REPLAY_RTOL, abs=0)


def test_page_states_stand_in_status_suite_and_demonstrator_steps():
    text = _text()
    assert 'synthetic or stand-in fit' in text
    assert 'not evidence of an empirical calibration' in text
    for module in VALIDATION_MODULES:
        assert f"``{module}``" in text
        assert (ROOT / module).is_file()
    demonstrators = text[text.index('Adopting the December demonstrators'):]
    assert '#358' in demonstrators and '#361' in demonstrators
    for step in DEMONSTRATOR_STEPS:
        assert step in demonstrators, step
    for output in ('``information_matrix``', '``fisher_information``',
                   '``unresolved_directions``'):
        assert output in text
    assert '#355' in text
    # Reading guidance: unavailable uncertainty, solver-backed Jacobian
    # accuracy, the Ziegler noise assumption and which methods record.
    for statement in ('Read ``report.covariance_diagnostic``',
                      'Pass a bound established by ``check_derivatives`` '
                      'as ``jacobian_rtol``',
                      "(``noise_model`` ``'identity'``)",
                      'Calibration records can capture LM and IPOPT fits',
                      '``derivative_bound``'):
        assert statement in text, statement
    for issue in ('#222', '#314', '#328', '#534', '#535'):
        assert issue in text


def test_page_is_listed_in_the_examples_index():
    entries = INDEX.read_text(encoding='utf-8').split()
    assert 'calibration_validation' in entries
