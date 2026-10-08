"""Execute the published Experiment/Measurement example and verify its output.

Scope: ``doc/online_docs/examples/experiment_measurement.rst``. Its
``testcode`` directives run in document order in one namespace; each
directive's printed output must equal the ``testoutput`` directive that
follows it, so the published tables and numbers cannot drift from the code.
The documented values are then checked against independent references: a
hand-written residual layout, closed-form standardized residuals of the
first-order decay model, a ``scipy.optimize.least_squares`` fit of those
residuals, and the closed-form linearized covariance. Core lane; runs in
well under a second.
"""

from collections.abc import Iterator
from pathlib import Path
import re
import textwrap

import numpy as np
import pytest
from scipy.optimize import least_squares

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "doc/online_docs/examples/experiment_measurement.rst"

# The documented physical case, written out independently of the page:
# experiment -> (initial c_A [mol/L], rise [K*L/mol], initial T [K]).
CALLBACK = {"hot": (2.0, 20.0, 300.0), "cold": (1.0, 10.0, 290.0)}
# Observed entries: (experiment, measurement, time [s], observation
# [mol/L or K], standard deviation [mol/L or K]). The missing cold c_A
# observation at 1.5 s is absent.
OBSERVED = [
    ("hot", "c_A", 0.0, 2.010, 0.02), ("hot", "c_A", 1.0, 1.200, 0.02),
    ("hot", "c_A", 2.0, 0.750, 0.02), ("hot", "c_A", 4.0, 0.265, 0.02),
    ("hot", "T", 2.0, 325.6, 0.5),
    ("cold", "c_A", 0.5, 0.785, 0.01), ("cold", "c_A", 3.0, 0.220, 0.02),
]
# Hand-written packing: experiment, then measurement column (c_A, T), then
# model-grid row; the hot grid is [0, 1, 2, 4] s and the cold grid keeps the
# missing sample: [0.5, 1.5, 3] s.
LAYOUT = (
    [("hot", "c_A", 0, t, True) for t in (0.0, 1.0, 2.0, 4.0)]
    + [("hot", "T", 1, t, t == 2.0) for t in (0.0, 1.0, 2.0, 4.0)]
    + [("cold", "c_A", 0, t, t != 1.5) for t in (0.5, 1.5, 3.0)]
    + [("cold", "T", 1, t, False) for t in (0.5, 1.5, 3.0)])
SEED_RATE = 0.3  # [1/s], the documented seed
# The example uses PharmaPy's default forward finite differences with step
# h = sqrt(1e-6) * k (``dx_jac_p``). Their truncation error h/2 * d2r/dk2
# moves LM's stationary point and the Jacobian; the test computes both
# effects in closed form and allows twice their size.
FD_RELATIVE_STEP = 1e-3  # [-], sqrt of the 1e-6 relative tolerance
TRUNCATION_MARGIN = 2.0  # [-], allowance over the leading-order error
# LM stops on a 1e-8 relative step (``eps_2``) or 1e-8 gradient.
LM_STEP_RTOL = 1e-8  # [-]
# Tolerances of the SciPy reference fit, four decades below LM_STEP_RTOL so
# the reference error is negligible against the allowed rate difference.
REFERENCE_TOL = 1e-12  # [-]

def _blocks(directive: str) -> Iterator[str]:
    """Extract the page's indented, option-free directives of one kind.

    Parameters
    ----------
    directive : str
        ``'testcode'`` or ``'testoutput'``.

    Yields
    ------
    str
        Dedented directive content, in document order, with trailing
        whitespace removed from every line.
    """
    text = PAGE.read_text().expandtabs()
    pattern = rf"^( *)\.\. {directive}::\s*\n"
    for match in re.finditer(pattern, text, re.MULTILINE):
        indentation = len(match.group(1))
        lines = []
        for line in text[match.end():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                break
            lines.append(line.rstrip())
        yield textwrap.dedent("\n".join(lines)).strip("\n")


CODE = tuple(_blocks("testcode"))
OUTPUT = tuple(_blocks("testoutput"))


def _model(rate, experiment, measurement, time, order=1):
    """Evaluate the documented model at one entry in closed form.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    experiment, measurement : str
        Keys of ``CALLBACK`` and ``'c_A'`` or ``'T'``.
    time : float
        Time [s].
    order : int, optional
        Order of the returned derivative with respect to k (1 or 2).

    Returns
    -------
    value : float
        c_A [mol/L] or T [K].
    derivative : float
        ``order``-th derivative with respect to k [mol/L*s**order or
        K*s**order].
    """
    initial, rise, temp0 = CALLBACK[experiment]
    remaining = np.exp(-rate * time)  # [-]
    # d^n exp(-k t) / dk^n = (-t)^n exp(-k t) [s**n]
    term = (-time) ** order * remaining
    if measurement == "c_A":
        return initial * remaining, initial * term
    return (temp0 + rise * initial * (1 - remaining), -rise * initial * term)


def _standardized(rate):
    """Return closed-form ``(model - data) / sigma`` of the observed entries.

    Parameters
    ----------
    rate : numpy.ndarray or float
        Rate constant k [1/s]; a shape ``(1,)`` array is accepted.

    Returns
    -------
    numpy.ndarray
        Residuals [-], in ``OBSERVED`` order.
    """
    rate = float(np.ravel(rate)[0])  # [1/s]
    return np.array([(_model(rate, e, m, t)[0] - y) / sigma
                     for e, m, t, y, sigma in OBSERVED])


def test_page_has_matching_code_and_output_blocks():
    """Keep an empty extraction from turning the checks vacuous."""
    assert len(CODE) == len(OUTPUT) == 2


def test_documented_example_output_and_values(capsys):
    namespace = {"__name__": "__experiment_measurement_page__"}
    for source, expected in zip(CODE, OUTPUT):
        exec(compile(source, f"{PAGE.name}:testcode", "exec"), namespace)
        printed = "\n".join(
            line.rstrip()
            for line in capsys.readouterr().out.strip("\n").splitlines())
        assert printed == expected

    estimator = namespace["estimator"]
    layout = estimator.get_residual_layout()
    assert list(layout.itertuples(index=False, name=None)) == LAYOUT
    assert estimator.num_data_total == len(OBSERVED)

    reference = least_squares(_standardized, [SEED_RATE], xtol=REFERENCE_TOL,
                              ftol=REFERENCE_TOL, gtol=REFERENCE_TOL)
    rate = reference.x[0]  # [1/s]
    residuals = _standardized(rate)  # [-]
    # d r / dk and d2 r / dk2 of the standardized residuals [s], [s**2].
    jacobian, curvature = (
        np.array([_model(rate, e, m, t, order)[1] / sigma
                  for e, m, t, _, sigma in OBSERVED]) for order in (1, 2))
    information = jacobian @ jacobian  # [s**2], i.e. 1/(1/s)**2

    # Leading-order shift of the stationary point of J_fd^T r = 0 caused
    # by the forward-difference Jacobian error h/2 * r'' [1/s].
    step = FD_RELATIVE_STEP * rate  # [1/s]
    rate_atol = (TRUNCATION_MARGIN * step / 2
                 * abs(residuals @ curvature) / information
                 + LM_STEP_RTOL * rate)  # [1/s]
    params = namespace["params"]  # [1/s]
    np.testing.assert_allclose(params, [rate], rtol=0, atol=rate_atol)

    observed = layout["observed"].to_numpy()
    info = namespace["info"]
    np.testing.assert_allclose(info["fun"][observed], residuals, rtol=0,
                               atol=np.abs(jacobian).max() * rate_atol)
    assert np.all(info["fun"][~observed] == 0.0)
    jacobian_atol = (TRUNCATION_MARGIN * step / 2 * np.abs(curvature)
                     + np.abs(curvature) * rate_atol)  # [s]
    assert np.all(np.abs(info["jac"][0][observed] - jacobian)
                  <= jacobian_atol)
    assert np.all(info["jac"][0][~observed] == 0.0)

    reduced_chi_square = residuals @ residuals / (len(OBSERVED) - 1)  # [-]
    std_absolute = np.sqrt(1.0 / information)  # [1/s]
    std_relative = np.sqrt(reduced_chi_square / information)  # [1/s]
    # Relative error of J @ J.T from the Jacobian bound; a standard
    # deviation carries half of it. The residual mean square changes only
    # at second order in the rate shift.
    std_rtol = (np.abs(jacobian) @ jacobian_atol) / information  # [-]
    np.testing.assert_allclose(namespace["std_absolute"], std_absolute,
                               rtol=std_rtol)
    np.testing.assert_allclose(namespace["std_relative"], std_relative,
                               rtol=std_rtol)
    np.testing.assert_allclose(namespace["covariance"],
                               [[std_relative ** 2]], rtol=2 * std_rtol)
    # The last documented call leaves the default form in covar_params.
    assert np.sqrt(estimator.covar_params[0, 0]) == namespace["std_relative"]
    # The documented numbers are the rounded independent values, and the
    # page's claim that the reduced chi-square is below one holds.
    assert f"k = {rate:.4f} 1/s" in OUTPUT[1]
    assert f"{std_absolute:.2e} 1/s, {std_relative:.2e} 1/s" in OUTPUT[1]
    assert reduced_chi_square < 1.0
