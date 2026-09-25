"""Particle-size unit regressions for real deliquoring and drying setup.

``SolidPhase.x_distrib`` stores crystal sizes in micrometers, while capillary
and threshold-pressure correlations consume particle diameters in meters.
Solver-backed cases use real phases, cakes, solver problems, and CVode.
"""

import numpy as np
import pytest

from PharmaPy import SolidLiquidSep as solid_liquid_sep
from PharmaPy.SolidLiquidSep import DeliquoringStep


pytestmark = pytest.mark.unit

SIZE_GRID_UM = np.array([50.0, 100.0, 150.0])  # [um]
CSD_NUMBER = np.array([1.0, 2.0, 1.0])  # [#/um], total basis
MICRONIZED_SIZE_GRID_UM = np.linspace(0.5, 1.5, 5)  # [um]
MICRONIZED_CSD_NUMBER = np.ones_like(MICRONIZED_SIZE_GRID_UM)  # [#/um]
POROSITY = 0.45  # [-]
DEFAULT_UNIT_DIAMETER = 0.01  # [m]
SINGULAR_CASE_CAKE_HEIGHT = 0.02  # [m]


def _zeroth_moment(
        size_grid_um=SIZE_GRID_UM,
        csd_number=CSD_NUMBER) -> float:
    """Return the number-based CSD zeroth moment.

    Parameters
    ----------
    size_grid_um : ndarray, optional
        Particle-size grid stored by ``SolidPhase`` [um].
    csd_number : ndarray, optional
        Total-population number distribution [#/um].

    Returns
    -------
    float
        Total crystal count [-].
    """
    return solid_liquid_sep.trapezoidal_rule(size_grid_um, csd_number)


def _make_deliquoring_unit(
        drying_cake_factory,
        size_grid_um=SIZE_GRID_UM,
        csd_number=CSD_NUMBER,
        saturation=0.8):
    """Build a deliquoring unit with a real packed cake.

    Parameters
    ----------
    drying_cake_factory : callable
        Factory for real liquid, solid, and cake collaborators.
    size_grid_um : ndarray, optional
        Particle-size grid [um].
    csd_number : ndarray, optional
        Total-population number distribution [#/um].
    saturation : float, optional
        Initial cake saturation [-].

    Returns
    -------
    DeliquoringStep
        Configured production unit.
    """
    cake = drying_cake_factory(
        size_grid_um=size_grid_um,
        csd_number=csd_number,
        saturation=saturation,
    )
    unit = DeliquoringStep(num_nodes=2)
    unit.Phases = cake
    unit.CakePhase.z_external = np.array(
        [0.0, unit.cake_height]
    )  # [m]
    return unit


def _distribution_for_cake_height(
        drying_cake_factory,
        size_grid_um,
        csd_number,
        target_height=SINGULAR_CASE_CAKE_HEIGHT):
    """Scale a real cake's number distribution to a target height.

    Parameters
    ----------
    drying_cake_factory : callable
        Factory for real liquid, solid, and cake collaborators.
    size_grid_um : ndarray
        Particle-size grid [um].
    csd_number : ndarray
        Relative total-population number distribution [#/um].
    target_height : float, optional
        Required packed-cake height [m].

    Returns
    -------
    ndarray
        Total-population number distribution producing ``target_height``
        [#/um].

    Notes
    -----
    ``Cake.cake_vol`` is linear in the total number distribution. Scaling its
    magnitude changes cake inventory and height without changing normalized
    particle-size weights or calculated porosity.
    """
    reference_cake = drying_cake_factory(
        size_grid_um=size_grid_um,
        csd_number=csd_number,
    )
    cross_section = np.pi * DEFAULT_UNIT_DIAMETER**2 / 4.0  # [m**2]
    reference_height = reference_cake.cake_vol / cross_section  # [m]
    return np.asarray(csd_number) * target_height / reference_height


@pytest.mark.assimulo
def test_deliquoring_setup_converts_micrometer_grid_to_meter_diameters(
        drying_cake_factory):
    """Real deliquoring setup uses meter diameters in capillary quantities."""
    pytest.importorskip("assimulo")
    unit = _make_deliquoring_unit(drying_cake_factory)

    unit.solve_unit(deltaP=5.0e4, runtime=1.0e-3, verbose=False)  # [Pa], [s]

    delta_p = 5.0e4  # [Pa]
    correlation_inputs = (
        unit.Solid_1.distrib,
        delta_p,
        unit.Solid_1.getPorosity(),
        unit.cake_height,
        unit.Solid_1.moments[0],
        (
            np.mean(unit.Liquid_1.getSurfTension()),
            np.mean(unit.Liquid_1.getDensity()),
        ),
    )
    expected_saturation = solid_liquid_sep.get_sat_inf(
        SIZE_GRID_UM * 1e-6, *correlation_inputs)  # [-]
    micrometer_basis = solid_liquid_sep.get_sat_inf(
        SIZE_GRID_UM, *correlation_inputs)  # [-]

    np.testing.assert_allclose(unit.Solid_1.x_distrib, SIZE_GRID_UM)
    assert unit.sat_inf == pytest.approx(expected_saturation, rel=1e-12)
    assert unit.sat_inf != pytest.approx(micrometer_basis, rel=1e-10)


@pytest.mark.assimulo
def test_deliquoring_threshold_pressure_uses_expected_size_basis(
        drying_cake_factory):
    """Threshold pressure uses meter diameters and micrometer CSD weights."""
    pytest.importorskip("assimulo")
    unit = _make_deliquoring_unit(drying_cake_factory)

    unit.solve_unit(deltaP=5.0e4, runtime=1.0e-3, verbose=False)  # [Pa], [s]

    porosity = unit.Solid_1.getPorosity()  # [-]
    surface_tension = np.mean(unit.Liquid_1.getSurfTension())  # [N/m]
    diameter_m = SIZE_GRID_UM * 1.0e-6  # [m]
    pressure_by_size = (
        4.6 * (1.0 - porosity) * surface_tension
        / (porosity * diameter_m)
    )  # [Pa], Destro et al. (2021), Eq. 16
    expected_threshold = (
        solid_liquid_sep.trapezoidal_rule(
            SIZE_GRID_UM, pressure_by_size * CSD_NUMBER
        )
        / _zeroth_moment()
    )  # [Pa]

    assert unit.p_thresh == pytest.approx(expected_threshold, rel=1e-12)


def test_get_sat_inf_returns_per_node_values_for_array_properties():
    """Array-valued liquid properties return one saturation per property node."""
    size_grid_um = np.array([50.0, 100.0, 150.0, 200.0])  # [um]
    size_grid_m = size_grid_um * 1e-6  # [m]
    csd_number = np.array([1.0, 2.0, 1.0, 0.5])  # [#/um]
    delta_p = 5.0e4  # [Pa]
    cake_height = 0.02  # [m]
    surf_tens = np.array([0.065, 0.067, 0.066])  # [N/m]
    rho_liq = np.array([950.0, 960.0, 970.0])  # [kg/m**3]
    mu_zero = _zeroth_moment(size_grid_um, csd_number)  # [-]

    sat_inf = solid_liquid_sep.get_sat_inf(
        size_grid_m,
        csd_number,
        delta_p,
        POROSITY,
        cake_height,
        mu_zero,
        (surf_tens, rho_liq),
    )

    expected_sat_inf = np.array(
        [0.1650323056996773, 0.1651821984011924, 0.1651072525989054]
    )  # [-], frozen asymmetric array-property case

    assert sat_inf.shape == expected_sat_inf.shape
    np.testing.assert_allclose(sat_inf, expected_sat_inf, rtol=1e-12)


def test_get_sat_inf_clips_micronized_aggregate_roundoff():
    """Searched micronized values clip upper-bound roundoff to one."""
    size_grid_um = np.array(
        [0.29443721003613355, 1.3640842476616897, 1.3987695850176836]
    )  # [um]
    size_grid_m = size_grid_um * 1e-6  # [m]
    csd_number = np.array(
        [1.6568408995106283e-7, 4.5172148784017079, 2.1064206650858339e-10]
    )  # [#/um]
    delta_p = 1.0e3  # [Pa]
    cake_height = 0.02  # [m]
    surf_tens = 0.066  # [N/m]
    rho_liq = 950.0  # [kg/m**3]
    mu_zero = _zeroth_moment(size_grid_um, csd_number)  # [-]

    sat_inf = solid_liquid_sep.get_sat_inf(
        size_grid_m,
        csd_number,
        delta_p,
        POROSITY,
        cake_height,
        mu_zero,
        (surf_tens, rho_liq),
    )

    assert sat_inf <= 1.0


def test_deliquoring_rejects_singular_micronized_irreducible_saturation(
        drying_cake_factory):
    """Real deliquoring setup rejects an undefined reduced saturation."""
    scaled_csd = _distribution_for_cake_height(
        drying_cake_factory,
        MICRONIZED_SIZE_GRID_UM,
        MICRONIZED_CSD_NUMBER,
    )  # [#/um]
    unit = _make_deliquoring_unit(
        drying_cake_factory,
        size_grid_um=MICRONIZED_SIZE_GRID_UM,
        csd_number=scaled_csd,
    )

    with pytest.raises(ValueError, match="s_inf=.*deltaP=.*particle size grid"):
        unit.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]


@pytest.mark.assimulo
def test_drying_setup_converts_micrometer_grid_before_saturation(
        drying_unit_factory):
    """Real drying setup passes meter diameters to the shared correlation.

    Parameters
    ----------
    drying_unit_factory : callable
        Construct a drying unit with real phases and the synthetic size grid.

    Notes
    -----
    Recompute the correlation on an explicit metre grid using the initial
    volatile-liquid composition. This preserves the conversion assertion
    across the phase-composition normalization introduced by PR #263.
    """
    pytest.importorskip("assimulo")
    dryer = drying_unit_factory(number_nodes=2)
    liquid_fraction = dryer.CakePhase.Liquid_1.mass_frac.copy()  # [-]
    liquid_fraction[dryer.idx_supercrit] = 0.0  # [-], liquid volatile subset
    initial_temperature = dryer.Solid_1.temp  # [K]
    liquid_density = dryer.Liquid_1.getDensity(
        temp=initial_temperature, mass_frac=liquid_fraction,
        basis='mass')  # [kg/m**3]
    surface_tension = dryer.Liquid_1.getSurfTension(
        temp=initial_temperature, mass_frac=liquid_fraction)  # [N/m]

    time, states = dryer.solve_unit(
        deltaP=5.0e4,
        runtime=1.0e-8,  # [s], one short real-backend integration
        verbose=False,
    )

    correlation_inputs = (
        dryer.Solid_1.distrib, dryer.deltaP, dryer.CakePhase.porosity,
        dryer.cake_height, dryer.Solid_1.moments[0],
        (np.mean(surface_tension), liquid_density),
    )  # [#/um], [Pa], [-], [m], [#], ([N/m], [kg/m**3])
    expected_irreducible_saturation = solid_liquid_sep.get_sat_inf(
        SIZE_GRID_UM * 1e-6, *correlation_inputs)  # [-], diameters [m]
    micrometer_basis = solid_liquid_sep.get_sat_inf(
        SIZE_GRID_UM, *correlation_inputs)  # [-], deliberately wrong size basis
    assert len(time) == np.shape(states)[0]
    np.testing.assert_allclose(dryer.Solid_1.x_distrib, SIZE_GRID_UM)
    assert dryer.s_inf == pytest.approx(
        expected_irreducible_saturation, rel=1e-10
    )
    assert dryer.s_inf != pytest.approx(micrometer_basis, rel=1e-10)


def test_drying_rejects_singular_micronized_irreducible_saturation(
        drying_unit_factory, drying_cake_factory):
    """Real drying setup rejects an undefined reduced saturation."""
    scaled_csd = _distribution_for_cake_height(
        drying_cake_factory,
        MICRONIZED_SIZE_GRID_UM,
        MICRONIZED_CSD_NUMBER,
    )  # [#/um]
    dryer = drying_unit_factory(
        number_nodes=2,
        size_grid_um=MICRONIZED_SIZE_GRID_UM,
        csd_number=scaled_csd,
    )

    with pytest.raises(ValueError, match="s_inf=.*deltaP=.*particle size grid"):
        dryer.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]
