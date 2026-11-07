"""Execute the QMOM guide and check its numbers against closed forms.

Scope: ``doc/online_docs/examples/qmom_crystallization.rst``. Its ``testcode``
directives run in document order in one namespace, and each one's printed output
must equal the ``testoutput`` that follows it. SciPy LSODA, about 15 s.
"""

from pathlib import Path
import re
import textwrap

import numpy as np
import pytest

from PharmaPy import CrystallizationKernels

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "doc/online_docs/examples/qmom_crystallization.rst"
INDEX = ROOT / "doc/online_docs/examples/index.rst"
# Directives of the page: setup, minimum size, batch, custom kernel, flowsheet.
NUM_BLOCKS = 5  # [-]
# The page's solves use rtol and atol 1e-10. The worst miss was 1.5e-8, on the
# flowsheet's crystal balance (6 Nov 2026); a margin of about seven.
SOLVE_RTOL = 1e-7  # [-]
# Rebuilding the moments from their own rule is algebra on the solved state,
# so only roundoff separates them; the worst miss was 2.2e-15 (6 Nov 2026).
ALGEBRAIC_RTOL = 1e-12  # [-]
METRE_PER_MICRON = 1e-6  # [m/um], exact
SHAPE_FACTOR = 0.5  # [-], the kv of the page's helper


def _blocks(directive):
    """Extract the page's directives of one kind.

    Parameters
    ----------
    directive : str
        ``'testcode'`` or ``'testoutput'``.

    Returns
    -------
    tuple of str
        Dedented contents in document order, without trailing whitespace.
    """
    text = PAGE.read_text(encoding="utf-8").expandtabs()
    blocks = []
    for match in re.finditer(rf"^( *)\.\. {directive}::.*\n", text, re.MULTILINE):
        indentation = len(match.group(1))
        lines = []
        for line in text[match.end():].splitlines():
            if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                break
            lines.append(line.rstrip())
        blocks.append(textwrap.dedent("\n".join(lines)).strip("\n"))
    return tuple(blocks)


@pytest.fixture(scope="module")
def page_run():
    """Run the page once, keeping what each block printed.

    Returns
    -------
    tuple
        The namespace after the last block, and the printed text per block.
    """
    namespace = {"__name__": "__qmom_guide_page__"}
    printed = []
    for source in _blocks("testcode"):
        lines = []
        namespace["print"] = lambda *values: lines.append(" ".join(str(v) for v in values))
        exec(compile(source, f"{PAGE.name}:testcode", "exec"), namespace)
        printed.append("\n".join(lines))
    return namespace, printed


def test_page_has_matching_code_and_output_blocks():
    assert len(_blocks("testcode")) == len(_blocks("testoutput")) == NUM_BLOCKS


def test_every_block_prints_what_the_page_shows(page_run):
    _, printed = page_run
    assert list(printed) == list(_blocks("testoutput"))


def test_batch_count_follows_the_logistic_balance(page_run):
    # Growth leaves mu_0 alone, and with constant rates dN/dt = S_0 N - beta_0 N**2 / 2,
    # so N approaches K = 2 S_0 / beta_0 along a logistic, whatever the sizes do.
    namespace, _ = page_run
    selection, kernel = namespace["SELECTION"], namespace["KERNEL"]
    start = np.sum(namespace["seed_counts"])  # [#/m**3]
    capacity = 2 * selection / kernel  # [#/m**3]
    expected = capacity / (1 + (capacity / start - 1)
                           * np.exp(-selection * namespace["RUNTIME"]))
    assert namespace["moments"][-1, 0] == pytest.approx(expected, rel=SOLVE_RTOL)


def test_reported_rule_reproduces_the_moments(page_run):
    namespace, _ = page_run
    nodes, weights = namespace["nodes"], namespace["weights"]
    rebuilt = np.sum(weights[:, :, None] * nodes[:, :, None] ** np.arange(6), axis=1)
    np.testing.assert_allclose(rebuilt, namespace["moments"], rtol=ALGEBRAIC_RTOL)


def test_custom_kernel_reads_the_supersaturation(page_run):
    # Nothing grows, so the supersaturation s is constant and the count follows
    # 1/N(t) = 1/N(0) + beta_0 s t / 2, with s from the charge's own composition.
    namespace, _ = page_run
    result = namespace["custom"].result
    count = namespace["custom_count"]  # [#/m**3]
    time = np.ravel(result.time)  # [s]
    moments = np.asarray(result.mu_n_solid0)  # [um**k/m**3]
    slurry_volume = np.ravel(result.vessel_vol)[0]  # [m**3]
    solid_fraction = SHAPE_FACTOR * moments[0, 3] * METRE_PER_MICRON ** 3  # [-]
    solute = np.asarray(result.mass_j_liquid0)[0, 2]  # [kg]
    concentration = solute / (slurry_volume * (1 - solid_fraction))  # [kg/m**3]
    supersaturation = concentration / namespace["SOLUBILITY"] - 1  # [-]

    expected = 1 / (1 / count[0] + namespace["KERNEL"] * supersaturation * time[-1] / 2)
    assert count[-1] == pytest.approx(expected, rel=SOLVE_RTOL)


def test_crystals_leaving_one_unit_arrive_in_the_next(page_run):
    namespace, _ = page_run
    sheet, held = namespace["sheet"], namespace["held"]
    upstream = held(sheet.U01)  # [#]
    # Guard against a vacuous pass: a good part of the seeds did leave.
    assert upstream[0] - upstream[-1] > 0.25 * upstream[0]
    assert held(sheet.U02)[-1] == pytest.approx(upstream[0] - upstream[-1], rel=SOLVE_RTOL)


def test_page_lists_exactly_the_built_in_rate_laws():
    text = PAGE.read_text(encoding="utf-8")
    pattern = r"``(\w+(?:Kernel|Selection|Fragmentation))``"
    named = set(re.findall(pattern, text))
    built_in = {name for name in vars(CrystallizationKernels)
                if re.fullmatch(r"\w+(?:Kernel|Selection|Fragmentation)", name)}
    assert named == built_in


def test_page_states_that_it_is_not_experimental_evidence():
    text = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))
    assert "**illustrative**" in text and "**not evidence**" in text


def test_page_is_listed_in_the_examples_index():
    assert "qmom_crystallization" in INDEX.read_text(encoding="utf-8").split()
