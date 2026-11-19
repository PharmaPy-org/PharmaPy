"""Cases of Marchisio, Vigil and Fox (2003, Table 4) through ``BatchCrystallizer``.

Scope: the three-node asymptotes of cases 2, 4, 7 and 8 against the paper's Tables
5 and 6. Crystals of 1 um at 1 #/m**3 stand in for its scaled L_0 = 1 and m_k = 1,
and the kernels' constants are set to give its bare forms. Nothing grows. SciPy BDF,
about 10 s. These cases match the paper and solve quickly; the guide records the
other runs (``doc/online_docs/examples/qmom_crystallization.rst``).
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.CrystallizationKernels import (BinaryFragmentation, ErosionFragmentation,
                                             ExponentialSelection, PowerLawSelection, ShearKernel,
                                             SumKernel, UniformFragmentation)
from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Mechanisms import QMOMPopulationBalance
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase

pytestmark = pytest.mark.integration

PATH_DATA = str(Path(__file__).resolve().parent / "Flowsheet" / "data" / "compound_database.json")
LIQUID_MASS_FRAC = [0.0, 0.0, 0.2, 0.0, 0.8]  # [-]
CRYSTAL_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-], pure C
TEMP = 298.15  # [K]; no term of these cases depends on it
SOLUBILITY = 50.0  # [kg/m**3]; nothing grows, so it only needs to be positive
CRYSTAL_DENSITY = 1200.0  # [kg/m**3], the solute's liquid density in this database
SHAPE_FACTOR = 0.5  # [-]; cancelled by SCALED_SUM_RATE
CUBIC_METRE_PER_CUBIC_MICRON = 1e-18  # [m**3/um**3], exact
# beta_0 kv 1e-18 = 1, so beta = L_i**3 + L_j**3, the paper's scaled form.
SCALED_SUM_RATE = 1 / (SHAPE_FACTOR * CUBIC_METRE_PER_CUBIC_MICRON)  # [1/s]
# gamma / 6 * 1e-18 = 1, so beta = (L_i + L_j)**3, the paper's scaled form.
SCALED_SHEAR_RATE = 6 / CUBIC_METRE_PER_CUBIC_MICRON  # [1/s]
# Past the asymptote: runs to 20, 30 and 100 gave the same seven digits (6 Nov 2026).
RUNTIME = 40.0  # [s], scaled time
# LSODA at rtol 1e-8 gave the same seven digits, but did not finish case 2 in a minute.
SOLVER_OPTIONS = {"rtol": 1e-6, "atol": 1e-8}  # [-], [-]
UNIT_MOMENTS = np.ones(6)  # [um**k/m**3], m_k = 1 in Table 4
CASE_7_MOMENTS = np.array([0.301, 0.308, 0.372, 0.930, 5.912, 51.0])  # [um**k/m**3], Table 4
# Table 5 prints m_0 to four decimals, and one unit of that digit covers rounding or
# cutting off. It also tells case 8 from a run without its minimum size (0.2981).
COUNT_ATOL = 1e-4  # [#/m**3]
# Table 6 prints d_43 to three decimals: one unit of that digit, plus 2.5e-4, how far
# case 7 moves when its mu_3 moves by half of the last digit Table 4 prints for it.
MEAN_SIZE_ATOL = 1.25e-3  # [um]
# Agglomeration and breakage conserve crystal volume, so mu_3 moves only by rounding.
MU3_CONSERVATION_RTOL = 1e-12  # [-]

# Rate laws and starts from Table 4; the last two entries from Tables 5 and 6. Table 5's
# count for case 7 is left out: its 0.0178 is out of line with the 0.0058 and 0.0053
# the table gives for two and four nodes, and this run gives 0.0054.
CASES = {
    "case_2_sum_symmetric": (
        SumKernel(SCALED_SUM_RATE), PowerLawSelection(0.02, 3.0, minimum_size=1.0),
        BinaryFragmentation(0.5), UNIT_MOMENTS, 0.0200, 4.432),
    "case_4_hydrodynamic_mass_ratio_1_4": (
        ShearKernel(SCALED_SHEAR_RATE), ExponentialSelection(0.1, 0.01, minimum_size=5 ** (1 / 3)),
        BinaryFragmentation(0.2), UNIT_MOMENTS, 0.0052, 6.379),
    "case_7_hydrodynamic_uniform": (
        ShearKernel(SCALED_SHEAR_RATE), ExponentialSelection(0.1, 0.01, minimum_size=1.0),
        UniformFragmentation(), CASE_7_MOMENTS, None, 6.426),
    "case_8_hydrodynamic_erosion": (
        ShearKernel(SCALED_SHEAR_RATE), PowerLawSelection(0.01, 6.0, minimum_size=3 ** (1 / 3)),
        ErosionFragmentation(1.0), UNIT_MOMENTS, 0.2963, 3.209),
}


@pytest.mark.parametrize("name", CASES)
def test_three_node_asymptote_matches_the_published_tables(name):
    kernel, selection, fragments, moments, published_count, published_mean_size = CASES[name]
    solid = SolidPhase(PATH_DATA, mass=0.0, mass_frac=CRYSTAL_MASS_FRAC, temp=TEMP)
    solid.mechanisms = QMOMPopulationBalance(
        owning_phase=solid, target_components="C", solvent_name="solvent",
        moments_init=moments, density=CRYSTAL_DENSITY, kv=SHAPE_FACTOR,
        agglomeration=kernel, breakage=selection, fragments=fragments)
    vessel = BatchCrystallizer(integrator=ScipyBackend(method="BDF", options=SOLVER_OPTIONS),
                               isothermal=True)
    vessel.Phases = [LiquidPhase(PATH_DATA, mass=1.0, mass_frac=LIQUID_MASS_FRAC, temp=TEMP,
                                 name_solv="solvent"), solid]
    vessel.CrystKinetics = CrystKinetics(coeff_solub=[SOLUBILITY], growth=(0.0, 0.0, 1.0))

    vessel.solve_unit(runtime=RUNTIME, verbose=False)

    trajectory = np.asarray(vessel.result.mu_n_solid0)  # [um**k/m**3]
    if published_count is not None:
        assert trajectory[-1, 0] == pytest.approx(published_count, abs=COUNT_ATOL)
    assert np.ravel(vessel.result.L43)[-1] == pytest.approx(published_mean_size, abs=MEAN_SIZE_ATOL)
    np.testing.assert_allclose(trajectory[:, 3], moments[3], rtol=MU3_CONSERVATION_RTOL)
