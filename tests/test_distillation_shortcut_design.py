"""Shortcut distillation regressions through production column contracts."""

import numpy as np
import pytest
from scipy.optimize import brentq

import PharmaPy.Distillation as distillation
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


def _real_column(data_path, column_type=distillation.DistillationColumn,
                 **kwargs):
    """Build a production column with a representative liquid feed.

    Parameters
    ----------
    data_path : dict
        Paths to the repository test-data directories.
    column_type : type, optional
        Production distillation-column class to instantiate.
    **kwargs
        Column constructor overrides. Pressure is in [Pa], feed quality is
        [-], recoveries are [%], reflux is [-], and plate numbers are [-].

    Returns
    -------
    PharmaPy.Distillation._BaseDistillation
        Configured production column with a feed of 25 [mol/s].
    """
    thermo_path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    feed = LiquidStream(
        thermo_path,
        temp=350.0,  # [K]
        mole_flow=25.0,  # [mol/s]
        mole_frac=[0.4, 0.6, 0.0, 0.0],  # [-]
        verbose=False,
    )
    settings = {
        "pres": 101325.0,  # [Pa]
        "q_feed": 1.0,  # [-], saturated liquid feed
        "LK": "A",
        "HK": "B",
        "perc_LK": 95.0,  # [%]
        "perc_HK": 5.0,  # [%]
        "num_plates": 8,  # [-], equilibrium-stage count
        "num_feed": 4,  # [-], tray count from the top
    }
    settings.update(kwargs)
    column = column_type(**settings)
    column.Inlet = feed
    return column


def test_positive_reflux_below_minimum_uses_configured_shortcut_ratio(
        data_path):
    """A subminimum specified reflux uses the configured multiplier."""
    configured_ratio = 1.8  # [-], operating reflux/Rmin
    column = _real_column(
        data_path,
        reflux=0.1,  # [-], below the production-calculated minimum
        reflux_to_minimum_ratio=configured_ratio,
    )

    result = column.calculate_shortcut_design()

    assert result["min_reflux"] > 0.1
    assert result["reflux"] == pytest.approx(
        configured_ratio * result["min_reflux"]
    )


def test_default_shortcut_reflux_ratio_is_pinned(data_path):
    """The documented default multiplier remains 1.5 [-]."""
    column = _real_column(data_path)

    result = column.calculate_shortcut_design()

    assert distillation.DEFAULT_REFLUX_TO_MINIMUM_RATIO == pytest.approx(1.5)
    assert result["reflux"] == pytest.approx(
        distillation.DEFAULT_REFLUX_TO_MINIMUM_RATIO * result["min_reflux"]
    )


def test_backward_compatible_shortcut_aliases(data_path):
    """Deprecated aliases return the production methods' real results."""
    column = _real_column(data_path)
    canonical_design = column.calculate_shortcut_design()

    with pytest.warns(
            DeprecationWarning,
            match="calculate_heuristics is deprecated"):
        alias_design = column.calculate_heuristics()

    assert alias_design.keys() == canonical_design.keys()
    np.testing.assert_allclose(
        alias_design["material_balances"]["x_dist"],
        canonical_design["material_balances"]["x_dist"],
    )
    np.testing.assert_allclose(
        alias_design["material_balances"]["x_bottom"],
        canonical_design["material_balances"]["x_bottom"],
    )
    for name in ("min_reflux", "num_min", "reflux", "num_plates",
                 "num_feed"):
        assert alias_design[name] == pytest.approx(canonical_design[name])

    material = canonical_design["material_balances"]
    canonical_minimum = column.calc_underwood_min_reflux(
        material["x_dist"],
        material["x_bottom"],
        material["dist_flow"],
        material["bottom_flow"],
    )
    with pytest.warns(DeprecationWarning, match="calc_min_reflux is deprecated"):
        alias_minimum = column.calc_min_reflux(
            material["x_dist"],
            material["x_bottom"],
            material["dist_flow"],
            material["bottom_flow"],
        )
    assert alias_minimum == pytest.approx(canonical_minimum)


def test_underwood_min_reflux_includes_feed_quality_in_target(data_path):
    """Underwood minimum reflux uses the real feed-quality root target."""
    column = _real_column(data_path, q_feed=0.8)  # [-], liquid feed fraction
    x_dist, x_bottom, dist_flow, bottom_flow = column.global_material_bce()
    alpha = column.get_alpha(column.pres, column.z_feed)  # [-]
    underwood_target = 1.0 - column.q_feed  # [-]

    def underwood_residual(phi):
        """Evaluate the independently stated first Underwood equation.

        Parameters
        ----------
        phi : float
            Candidate Underwood root [-].

        Returns
        -------
        float
            Residual of the first Underwood equation [-].
        """
        return (
            np.sum(alpha * column.z_feed / (alpha - phi))
            - underwood_target
        )

    root_margin = 1.0e-10  # [-], excludes volatility singularities
    phi = brentq(
        underwood_residual,
        alpha[column.HK_index] + root_margin,
        alpha[column.LK_index] - root_margin,
    )  # [-]
    minimum_vapor_flow = np.sum(
        alpha * dist_flow * x_dist / (alpha - phi)
    )  # [mol/s]
    expected_minimum = (minimum_vapor_flow - dist_flow) / dist_flow  # [-]

    actual_minimum = column.calc_underwood_min_reflux(
        x_dist,
        x_bottom,
        dist_flow,
        bottom_flow,
    )

    assert underwood_residual(phi) == pytest.approx(0.0, abs=1.0e-9)
    assert actual_minimum == pytest.approx(expected_minimum, rel=1.0e-6)


def test_column_startup_accepts_deprecated_time_heuristics_keyword(data_path):
    """The old startup keyword initializes a production dynamic column."""
    column = _real_column(
        data_path,
        column_type=distillation.DynamicDistillation,
    )
    dynamic_inlet = DynamicInput()

    def mole_fraction_at_time(time_s):
        """Return a time-varying binary feed composition [-].

        Parameters
        ----------
        time_s : float
            Shortcut-design evaluation time [s].

        Returns
        -------
        numpy.ndarray
            Four-species feed mole fractions [-].
        """
        light_key_fraction = 0.40 - 0.01 * time_s  # [-]
        heavy_key_fraction = 1.0 - light_key_fraction  # [-]
        return np.array(
            [light_key_fraction, heavy_key_fraction, 0.0, 0.0]
        )  # [-]

    dynamic_inlet.add_variable("mole_frac", mole_fraction_at_time)
    column.Inlet.DynamicInlet = dynamic_inlet
    expected_at_12_s = column.calculate_shortcut_design(12.0)
    static_design = column.calculate_shortcut_design()

    with pytest.warns(DeprecationWarning, match="time_heuristics is deprecated"):
        column.column_startup(time_heuristics=12.0)  # [s]

    shortcut = column.shortcut_design
    assert column.heuristics is shortcut
    assert column.num_plates == int(shortcut["num_plates"])
    assert column.num_feed == int(shortcut["num_feed"])
    assert column.reflux == pytest.approx(shortcut["reflux"])
    np.testing.assert_allclose(
        shortcut["material_balances"]["x_dist"],
        expected_at_12_s["material_balances"]["x_dist"],
    )
    assert not np.allclose(
        shortcut["material_balances"]["x_dist"],
        static_design["material_balances"]["x_dist"],
    )
