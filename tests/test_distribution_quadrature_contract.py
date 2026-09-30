"""Independently derived population inventories and conversion round trips.

Nodal volume fractions use the same trapezoidal support as population moments.
The asymmetric public fixture exercises occupied endpoints and unequal widths;
no private teaching inputs are used.
"""
import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.MixedPhases import Slurry
from PharmaPy.SolidLiquidSep import Filter

pytestmark = pytest.mark.unit

RELATIVE_ROUNDOFF = 1e-12  # [-], budget for fewer than 100 float64 operations
GRID = np.array([100., 200., 400., 800.])  # [um], unequal interval lengths
SUPPORT = np.array([50., 150., 300., 200.])  # [um], trapezoid nodal weights
FRACTIONS = np.array([0.1, 0.2, 0.3, 0.4])  # [-], asymmetric volume allocation
SOLID_DENSITY = 1230.  # [kg/m**3], pure A in the public flowsheet database
MASS = 2.46  # [kg], two litres of pure A


@pytest.mark.parametrize('shape_factor', [0.2, 0.5, 1.0])
@pytest.mark.parametrize('basis', ['vol_perc', 'mass_frac'])
def test_constructor_population_preserves_inventory_and_filter_handoff(
        data_path, shape_factor, basis):
    """Occupied endpoints must not lose mass before or after filter attachment.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    shape_factor : float
        Particle volume divided by diameter cubed [-].
    basis : str
        Volume or mass weights; identical for this uniform-density population.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    solid = SolidPhase(path, mass=MASS, mass_frac=[1, 0, 0, 0, 0],
                       kv=shape_factor, x_distrib=GRID, distrib=FRACTIONS,
                       distrib_type=basis)
    particle_mass = SOLID_DENSITY * shape_factor * (GRID * 1e-6)**3  # [kg/particle]
    expected_density = MASS * FRACTIONS / particle_mass / SUPPORT  # [#/um]
    np.testing.assert_allclose(solid.distrib, expected_density,
                               rtol=RELATIVE_ROUNDOFF, atol=0)
    represented_mass = np.dot(SUPPORT, solid.distrib * particle_mass)  # [kg]
    assert represented_mass == pytest.approx(MASS, rel=RELATIVE_ROUNDOFF)
    assert solid.moments[3] * shape_factor * SOLID_DENSITY == pytest.approx(
        MASS, rel=RELATIVE_ROUNDOFF)
    np.testing.assert_allclose(solid.convert_distribution(num_distr=solid.distrib),
                               FRACTIONS, rtol=RELATIVE_ROUNDOFF, atol=0)
    solid.updatePhase(distrib=solid.distrib.copy())
    assert solid.mass == pytest.approx(MASS, rel=RELATIVE_ROUNDOFF)
    liquid = LiquidPhase(path, vol=0.01, mass_frac=[0, 0, 0, 0, 1])  # [m**3]
    slurry = Slurry()
    slurry.Phases = (liquid, solid)
    unit = Filter(station_diam=0.2, alpha=1e10, resist_medium=1e9)  # [m, m/kg, 1/m]
    unit.Phases = slurry
    assert unit.Solid_1.mass == pytest.approx(MASS, rel=RELATIVE_ROUNDOFF)


def test_number_conversion_handles_population_histories_and_empty_rows(data_path):
    """Each history row normalizes independently, including an unseeded start.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    solid = SolidPhase(path, mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=GRID, distrib=np.zeros(4), kv=0.5)  # [-]
    number = np.array([2e5, 4e4, 3e3, 7e2])  # [#/um], asymmetric population
    history = np.vstack([np.zeros(4), number, 3 * number])  # [#/um]
    contributions = SUPPORT * number * GRID**3  # [um**3], shape factor cancels
    fractions = contributions / contributions.sum()  # [-]
    actual = solid.convert_distribution(num_distr=history)  # [-]
    np.testing.assert_allclose(actual, np.vstack([np.zeros(4), fractions, fractions]),
                               rtol=RELATIVE_ROUNDOFF, atol=0)


def test_zero_size_unoccupied_node_is_finite(data_path):
    """Zero size carries no volume; an empty origin must not produce NaN.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    grid = np.array([0., 100., 200.])  # [um]
    weights = np.array([0., 0.4, 0.6])  # [-]
    solid = SolidPhase(path, mass=MASS, mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=grid, distrib=weights)
    assert np.isfinite(solid.distrib).all()
    np.testing.assert_allclose(solid.convert_distribution(num_distr=solid.distrib),
                               weights, rtol=RELATIVE_ROUNDOFF, atol=0)
    with pytest.raises(ValueError, match='zero.*size'):
        solid.convert_distribution(vol_distr=[0.1, 0.3, 0.6], mass=MASS)
