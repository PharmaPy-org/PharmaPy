"""All ten cases of Marchisio, Vigil and Fox (2003, Table 4) with two to four nodes.

Scope: which asymptotes of the paper's Tables 5 and 6 the QMOM moment rates reproduce,
as the comparison table of ``doc/online_docs/examples/qmom_crystallization.rst``
states. A fixed-step Runge-Kutta scheme integrates the rate functions themselves,
because adaptive solvers stall or go wrong where a node rests on the minimum breakage
size (#564); this is not ``solve_unit``. Sizes of 1 um at 1 #/m**3 stand in for the
paper's scaled L_0 = 1, and time is in its scaled units. The suite runs the two-node
column, about 10 s. ``python tests/test_qmom_marchisio_table.py`` runs all 30, prints
every number and fails if a verdict differs from the table, in about a minute.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pytest

from PharmaPy.CrystallizationKernels import (STANDARD_GRAVITY, BinaryFragmentation,
                                             ConstantKernel, ConstantSelection,
                                             DifferentialSedimentationKernel,
                                             ErosionFragmentation, ExponentialSelection,
                                             KernelConditions, ParabolicFragmentation,
                                             PowerLawSelection, ShearKernel, SumKernel,
                                             UniformFragmentation)
from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.DataClasses import StateKey
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import (QMOMPopulationBalance, _agglomeration_moment_rates,
                                 _breakage_moment_rates)
from PharmaPy.MomentInversion import invert_moments
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "doc/online_docs/examples/qmom_crystallization.rst"
PATH_DATA = str(ROOT / "tests" / "Flowsheet" / "data" / "compound_database.json")
TEMP = 298.15  # [K]; no term of these cases depends on it
SHAPE_FACTOR = 0.5  # [-]; cancelled by SCALED_SUM_RATE
VISCOSITY = 1e-3  # [Pa s]; cancelled by SCALED_DENSITY_DIFFERENCE
CUBIC_METRE_PER_CUBIC_MICRON = 1e-18  # [m**3/um**3], exact
# beta_0 kv 1e-18 = 1, so beta = L_i**3 + L_j**3, the paper's scaled form.
SCALED_SUM_RATE = 1 / (SHAPE_FACTOR * CUBIC_METRE_PER_CUBIC_MICRON)  # [1/s]
# gamma / 6 * 1e-18 = 1, so beta = (L_i + L_j)**3.
SCALED_SHEAR_RATE = 6 / CUBIC_METRE_PER_CUBIC_MICRON  # [1/s]
# pi g drho / (72 mu) * 1e-24 = 1, so beta = (L_i + L_j)**2 |L_i**2 - L_j**2|.
SCALED_DENSITY_DIFFERENCE = 72 * VISCOSITY / (np.pi * STANDARD_GRAVITY * 1e-24)  # [kg/m**3]
# Nothing grows; the built-in rate laws read only the temperature and the shape factor.
CONDITIONS = KernelConditions(temp=TEMP, supersaturation=0.0, growth_rate=0.0,
                              shape_factor=SHAPE_FACTOR)
NODE_COUNTS = (2, 3, 4)  # [-], QMOM2, QMOM3 and QMOM4 of the paper

# Starts of Table 4. m_k = 1 is one crystal of unit size.
CASE_7_MOMENTS = np.array([0.301, 0.308, 0.372, 0.930, 5.912, 51.0])  # [um**k/m**3]
# Table 4 prints case 10 to three decimals. Equal counts of L = 1 and L = 2**(1/3)
# give those digits (checked below) and the two further moments four nodes need.
CASE_10_SIZES = np.array([1.0, 2 ** (1 / 3)])  # [um]
CASE_10_COUNTS = np.array([0.5, 0.5])  # [#/m**3]
CASE_10_PRINTED = np.array([1.000, 1.130, 1.294, 1.500, 1.760, 2.087])  # [um**k/m**3]

# Rate laws and start of each case (Table 4). The paper's cases 4, 5 and 8 let a crystal
# exactly at the threshold break; ``minimum_size`` does not, the only difference.
CASES = {
    1: (ConstantKernel(1.0), ConstantSelection(0.02, minimum_size=1.0),
        BinaryFragmentation(0.5), "unit"),
    2: (SumKernel(SCALED_SUM_RATE), PowerLawSelection(0.02, 3.0, minimum_size=1.0),
        BinaryFragmentation(0.5), "unit"),
    3: (ShearKernel(SCALED_SHEAR_RATE), ExponentialSelection(0.1, 0.01, minimum_size=1.0),
        BinaryFragmentation(0.5), "unit"),
    4: (ShearKernel(SCALED_SHEAR_RATE),
        ExponentialSelection(0.1, 0.01, minimum_size=5 ** (1 / 3)),
        BinaryFragmentation(0.2), "unit"),
    5: (ShearKernel(SCALED_SHEAR_RATE),
        ExponentialSelection(0.1, 0.01, minimum_size=3 ** (1 / 3)),
        ErosionFragmentation(1.0), "unit"),
    6: (ShearKernel(SCALED_SHEAR_RATE), ExponentialSelection(0.1, 0.01, minimum_size=1.0),
        UniformFragmentation(), "unit"),
    7: (ShearKernel(SCALED_SHEAR_RATE), ExponentialSelection(0.1, 0.01, minimum_size=1.0),
        UniformFragmentation(), "case_7"),
    8: (ShearKernel(SCALED_SHEAR_RATE), PowerLawSelection(0.01, 6.0, minimum_size=3 ** (1 / 3)),
        ErosionFragmentation(1.0), "unit"),
    9: (ShearKernel(SCALED_SHEAR_RATE), PowerLawSelection(2.0, 1.5, minimum_size=1.0),
        ParabolicFragmentation(0.5), "unit"),
    10: (DifferentialSedimentationKernel(SCALED_DENSITY_DIFFERENCE, VISCOSITY),
         PowerLawSelection(0.01, 6.0, minimum_size=1.0), UniformFragmentation(), "case_10"),
}

# Asymptotes with two, three and four nodes: m_0 [#/m**3] from Table 5 and d_43 [um]
# from Table 6. Table 5's count for case 7 with three nodes is left out: its 0.0178 is
# out of line with the 0.0058 and 0.0053 beside it, and the run gives 0.0054.
PUBLISHED = {
    1: ((0.0400, 3.711), (0.0376, 4.198), (0.0362, 4.192)),
    2: ((0.0200, 4.920), (0.0200, 4.432), (0.0200, 4.430)),
    3: ((0.0040, 6.519), (0.0038, 6.667), (0.0039, 6.672)),
    4: ((0.0056, 6.324), (0.0052, 6.379), (0.0052, 6.454)),
    5: ((0.2470, 5.667), (0.2658, 4.157), (0.2675, 4.252)),
    6: ((0.0061, 6.573), (0.0056, 6.459), (0.0056, 6.588)),
    7: ((0.0058, 6.543), (None, 6.426), (0.0053, 6.553)),
    8: ((0.2739, 3.133), (0.2963, 3.209), (0.2787, 3.159)),
    9: ((0.1409, 3.301), (0.1423, 3.494), (0.1433, 3.506)),
    10: ((0.0974, 2.791), (0.0962, 2.837), (0.0965, 2.951)),
}
COUNT_DIGITS = 4  # [-], decimals Table 5 prints
MEAN_SIZE_DIGITS = 3  # [-], decimals Table 6 prints

# Which runs reproduce the paper, per case for two, three and four nodes: what these
# rate functions gave on 24 Nov 2026, 17 of 30. The cause of the 13 mismatches is not
# known, so a False here is a record and not a contract. A run that changes sides, either
# way, needs the guide's table and its discussion updated with it. The suite checks the
# first column and the script all three.
MATCHES = {
    1: (True, True, True),
    2: (False, True, True),
    3: (True, True, False),
    4: (True, True, True),
    5: (False, False, False),
    6: (True, True, True),
    7: (True, True, True),
    8: (False, True, False),
    9: (False, False, False),
    10: (False, False, False),
}

# The classical Runge-Kutta scheme is stable while its step times the fastest rate stays
# below 2.78. More nodes reach larger sizes, where the exponential and sixth-power
# selection functions are fastest, so the step shrinks with the node count. Cases 2 to 8
# kept their digits with two nodes at 0.05 and with three at 0.01. With four they did at
# 0.005, except case 3, which was unstable there and stable at 0.002. All 30 runs gave
# the same verdict at a single step of 0.002 over a run to 40 (24 Nov 2026). An unstable
# step can end on a plausible number, as case 6 with four nodes did at 0.05, so do not
# enlarge these without repeating that comparison.
STEPS = {2: 0.02, 3: 0.005, 4: 0.002}  # [s], scaled time
# Cases 2 to 8 hold their first five digits from 15 on; 9 and 10 do not match either way.
FINAL_TIME = 15.0  # [s], scaled time
# Case 1 is slow, with rates of 0.02 to 0.04, and needs neither a short run nor a small
# step: steps of 0.05 to 1 gave the same printed digits.
CASE_1_FINAL_TIME = 1000.0  # [s], scaled time
CASE_1_STEP = 0.5  # [s], scaled time
# The rates here and the vessel's own differ only by rounding.
ALGEBRAIC_RTOL = 1e-12  # [-]


def _initial_moments(start, num_mom):
    """Moments a case starts from.

    Parameters
    ----------
    start : str
        ``'unit'`` for m_k = 1, ``'case_7'`` or ``'case_10'``.
    num_mom : int
        Number of moments, twice the number of nodes.

    Returns
    -------
    numpy.ndarray
        m_0..m_{num_mom - 1} [um**k/m**3], shape ``(num_mom,)``.

    Notes
    -----
    Table 4 gives six moments for case 7. Four nodes need eight, and take the two
    further ones from the three-node rule of those six, an assumption of this test:
    the asymptote does not depend on them.
    """
    orders = np.arange(num_mom)
    if start == "unit":
        return np.ones(num_mom)
    if start == "case_10":
        return np.sum(CASE_10_COUNTS[:, None] * CASE_10_SIZES[:, None] ** orders, axis=0)

    rule = invert_moments(CASE_7_MOMENTS, 3)
    moments = np.sum(rule.weights[:, None] * rule.nodes[:, None] ** orders, axis=0)
    given = min(num_mom, len(CASE_7_MOMENTS))
    moments[:given] = CASE_7_MOMENTS[:given]
    return moments


def _moment_rates(moments, kernel, selection, fragments):
    """Moment rates of agglomeration and breakage on the quadrature of the moments.

    Parameters
    ----------
    moments : numpy.ndarray
        m_0..m_{2N-1} [um**k/m**3], shape ``(2N,)``.
    kernel : callable
        Agglomeration kernel [m**3/s].
    selection : callable
        Breakage selection function [1/s].
    fragments : object
        Fragment distribution.

    Returns
    -------
    numpy.ndarray
        d(m_k)/dt [um**k/(m**3 s)], shape ``(2N,)``, from the functions
        ``QMOMKernelSources`` calls, with nuclei of zero size.
    """
    num_mom = len(moments)
    rule = invert_moments(moments, num_mom // 2)  # [um], [#/m**3]
    nodes, weights = rule.nodes, rule.weights
    kernel_values = np.asarray(kernel(nodes[:, None], nodes[None, :], CONDITIONS),
                               dtype=float)  # [m**3/s]
    fragment_moments = np.asarray(fragments.moments(nodes, np.arange(num_mom)),
                                  dtype=float)  # [um**k]

    return (_agglomeration_moment_rates(nodes, weights, kernel_values, num_mom)
            + _breakage_moment_rates(nodes, weights,
                                     np.asarray(selection(nodes, CONDITIONS), dtype=float),
                                     fragment_moments))


def _asymptote(case, num_nodes):
    """Integrate one case to its asymptote with the classical Runge-Kutta scheme.

    Parameters
    ----------
    case : int
        Case of Table 4, 1 to 10.
    num_nodes : int
        Number of quadrature nodes, 2 to 4.

    Returns
    -------
    tuple of float
        m_0 [#/m**3] and d_43 = m_4 / m_3 [um] at the end of the run. Two nodes do
        not carry m_4, which then comes from their quadrature.
    """
    kernel, selection, fragments, start = CASES[case]
    moments = _initial_moments(start, 2 * num_nodes)  # [um**k/m**3]
    final_time, step = ((CASE_1_FINAL_TIME, CASE_1_STEP) if case == 1
                        else (FINAL_TIME, STEPS[num_nodes]))  # [s], [s]

    for _ in range(int(round(final_time / step))):
        slope_1 = _moment_rates(moments, kernel, selection, fragments)
        slope_2 = _moment_rates(moments + 0.5 * step * slope_1, kernel, selection, fragments)
        slope_3 = _moment_rates(moments + 0.5 * step * slope_2, kernel, selection, fragments)
        slope_4 = _moment_rates(moments + step * slope_3, kernel, selection, fragments)
        moments = moments + step / 6 * (slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4)

    rule = invert_moments(moments, num_nodes)  # [um], [#/m**3]
    fourth_moment = np.sum(rule.weights * rule.nodes ** 4)  # [um**4/m**3]
    return float(moments[0]), float(fourth_moment / moments[3])


def _within_last_digit(value, published, digits):
    """Whether a value, rounded as the paper prints it, is within one unit of its last digit.

    Parameters
    ----------
    value : float
        Computed value.
    published : float or None
        Printed value; None, for a value left out, always agrees.
    digits : int
        Decimals the paper prints.

    Returns
    -------
    bool
        True when the rounded value differs from the printed one by at most one unit
        of the last digit.
    """
    if published is None:
        return True
    # Half a unit of slack separates one unit from two after rounding.
    return abs(round(value, digits) - published) < 1.5 * 10.0 ** -digits


def _reproduces(case, position, count, mean_size):
    """Whether one run matches the paper in both m_0 and d_43.

    Parameters
    ----------
    case : int
        Case of Table 4.
    position : int
        Index into ``NODE_COUNTS``.
    count : float
        Computed m_0 [#/m**3].
    mean_size : float
        Computed d_43 [um].

    Returns
    -------
    bool
        True when both agree with Tables 5 and 6 within their last printed digit.
    """
    published_count, published_mean_size = PUBLISHED[case][position]
    return (_within_last_digit(count, published_count, COUNT_DIGITS)
            and _within_last_digit(mean_size, published_mean_size, MEAN_SIZE_DIGITS))


def _guide_table():
    """Read the comparison table of the guide page.

    Returns
    -------
    dict
        Case number to its three verdicts, for two, three and four nodes, True for
        a cell reading ``Yes``.
    """
    text = PAGE.read_text(encoding="utf-8")
    section = text[text.index("Comparison with the published results"):]
    table = section[section.index(".. list-table::"):section.index("**The cause of")]
    rows = [[cell.strip() for cell in re.split(r"\n\s+- ", row)]
            for row in re.split(r"\n\s+\* - ", table)[1:]]
    return {int(row[0]): tuple(cell == "Yes" for cell in row[4:7]) for row in rows[1:]}


def test_case_10_starts_from_the_moments_the_paper_prints():
    np.testing.assert_array_equal(np.round(_initial_moments("case_10", 6), 3), CASE_10_PRINTED)


def test_the_rates_integrated_are_those_of_the_vessel():
    # Case 4 on a population of three sizes, through the kernel process of a batch
    # crystallizer: the same numbers as the bare rate functions this module integrates.
    kernel, selection, fragments, _ = CASES[4]
    sizes, counts = np.array([2.0, 5.0, 8.5]), np.array([0.03, 0.004, 0.001])  # [um], [#/m**3]
    moments = np.sum(counts[:, None] * sizes[:, None] ** np.arange(6), axis=0)  # [um**k/m**3]
    solid = SolidPhase(PATH_DATA, mass=0.0, mass_frac=[0.0, 0.0, 1.0, 0.0, 0.0], temp=TEMP)
    with pytest.warns(RuntimeWarning, match="#564"):
        mechanism = QMOMPopulationBalance(
            owning_phase=solid, target_components="C", solvent_name="solvent",
            moments_init=moments, density=1200.0, kv=SHAPE_FACTOR,
            agglomeration=kernel, breakage=selection, fragments=fragments)
    solid.mechanisms = mechanism
    vessel = BatchCrystallizer(integrator=None, isothermal=True)
    vessel.Phases = [LiquidPhase(PATH_DATA, mass=1.0, mass_frac=[0.0, 0.0, 0.2, 0.0, 0.8],
                                 temp=TEMP, name_solv="solvent"), solid]
    vessel.CrystKinetics = CrystKinetics(coeff_solub=[50.0], growth=(0.0, 0.0, 1.0))
    vessel.compile_structure()
    process = next(candidate for candidate in vessel.intraphase_processes
                   if candidate.mechanism is mechanism.kernel_sources)
    key = StateKey(mechanism.moments_state_name, process.phaseref)

    through_vessel = mechanism.kernel_sources.get_solver_state_rates(
        process, solid, 0.0, {key: moments}).state_rates[key]

    expected = _moment_rates(moments, kernel, selection, fragments)
    assert np.all(expected[[0, 1, 2, 4, 5]] != 0)
    np.testing.assert_allclose(through_vessel[[0, 1, 2, 4, 5]], expected[[0, 1, 2, 4, 5]],
                               rtol=ALGEBRAIC_RTOL)
    # Both conserve crystal volume, so m_3 moves only by rounding.
    assert abs(through_vessel[3]) <= ALGEBRAIC_RTOL * abs(through_vessel[4]) / sizes[-1]


@pytest.mark.parametrize("case", CASES)
def test_two_node_asymptotes_match_the_published_tables_where_the_guide_says_so(case):
    # Three and four nodes take most of a minute, so they run in script mode only.
    assert _reproduces(case, 0, *_asymptote(case, NODE_COUNTS[0])) == MATCHES[case][0]


def test_the_guide_table_is_the_one_this_module_rebuilds():
    assert _guide_table() == MATCHES
    assert sum(sum(row) for row in MATCHES.values()) == 17


if __name__ == "__main__":
    changed = []
    print("case nodes   m_0 (Table 5)      d_43 (Table 6)   matches")
    for number in CASES:
        for index, nodes_asked in enumerate(NODE_COUNTS):
            m_0, d_43 = _asymptote(number, nodes_asked)
            printed_count, printed_mean_size = PUBLISHED[number][index]
            verdict = _reproduces(number, index, m_0, d_43)
            if verdict != MATCHES[number][index]:
                changed.append((number, nodes_asked))
            print("%4d %5d   %.5f (%s)   %.4f (%.3f)   %s"
                  % (number, nodes_asked, m_0,
                     "left out" if printed_count is None else "%.4f" % printed_count,
                     d_43, printed_mean_size, "Yes" if verdict else "No"))
    if changed:
        sys.exit("Verdicts that differ from MATCHES, as (case, nodes): %s" % changed)
    print("All 30 verdicts are those of MATCHES and of the guide's table.")
