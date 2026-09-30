"""Distillation solve-unit regressions."""

import pytest

from PharmaPy.Distillation import DistillationColumn
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


def test_steady_state_solve_uses_calc_plates_results(data_path):
    """``solve_unit`` carries real shortcut results into real outlets."""
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    feed = LiquidStream(
        thermo_path,
        temp=350.0,  # [K]
        mole_flow=25.0,  # [mol/s]
        mole_frac=[0.4, 0.6, 0.0, 0.0],  # [-]
        verbose=False,
    )
    column = DistillationColumn(
        pres=101325.0,  # [Pa]
        q_feed=1.0,  # [-]
        LK="A",
        HK="B",
        perc_LK=95.0,  # [%]
        perc_HK=5.0,  # [%]
        reflux=1.0,  # [-]
        num_plates=8,  # [-]
        num_feed=4,  # [-]
    )
    column.Inlet = feed

    design = column.solve_unit(solve_ss=True)

    material = design["material_balances"]
    assert column.result.num_plates == design["num_plates"]
    assert column.result.reflux == pytest.approx(design["reflux"])
    assert column.OutletDistillate.mole_flow == pytest.approx(
        material["dist_flow"]
    )
    assert column.OutletBottom.mole_flow == pytest.approx(
        material["bottom_flow"]
    )
    assert column.Outlet is column.OutletBottom
