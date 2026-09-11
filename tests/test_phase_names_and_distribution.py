"""Regressions for #66, #64, and #162 using real phases and slurry streams.

The shared thermodynamic database supplies pure solid A (1230 kg/m**3).
No optional solver backend is needed. Analytic bin masses and trapezoidal
moments distinguish number density, bin fractions, and slurry volume bases.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase, classify_phases
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit

SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A
LIQUID_COMPOSITION = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], valid solvent mixture
SHAPE_FACTOR = 0.5  # [-], chosen non-unit to expose missing shape factors
SOLID_DENSITY = 1230.0  # [kg/m**3], pure A in compound_database.json
# Roundoff allowance for short sums/products and composition epsilon handling.
REL_TOL = 1e-12  # [-]
OLD_GRID = np.array([0.0, 100.0, 200.0, 300.0])  # [um], #162 probe
UNIFORM_GRID = np.array([0.0, 50.0, 100.0, 150.0])  # [um], #162 probe
GEOMETRIC_GRID = np.array([25.0, 100.0, 400.0, 1600.0])  # [um], ratio four
# Boundaries are [12.5, 50, 200, 800, 3200] um for the ratio-four grid.
GEOMETRIC_WIDTHS = np.array([37.5, 150.0, 600.0, 2400.0])  # [um]
UNIFORM_WIDTH = 50.0  # [um]
NUMBER_DENSITY = np.array([0.0, 8e6, 1e6, 0.0])  # [#/um]


@pytest.fixture
def thermo_path(data_path):
    """Locate the shared pure-component property database.

    Parameters
    ----------
    data_path : dict
        Paths supplied by the repository test fixtures.

    Returns
    -------
    str
        Path to the flowsheet thermodynamic database.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def test_classify_phases_explicit_names(thermo_path):
    """Explicit names must identify the same real phase on both objects.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    """
    solid = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION)
    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION)
    container = SimpleNamespace(Phases=[solid, liquid])

    classify_phases(container, names=['A', 'B'])

    assert solid.name == 'A'
    assert liquid.name == 'B'
    assert container.A is solid
    assert container.B is liquid


def test_classify_phases_generated_names(thermo_path):
    """Generated names retain independent per-type counters and ordering.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    """
    phases = [
        LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION),
        SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION),
        LiquidPhase(thermo_path, mass_frac=LIQUID_COMPOSITION),
    ]
    container = SimpleNamespace(Phases=phases)

    classify_phases(container)

    assert [phase.name for phase in phases] == ['Liquid_1', 'Solid_1', 'Liquid_2']
    for name, phase in zip(['Liquid_1', 'Solid_1', 'Liquid_2'], phases):
        assert getattr(container, name) is phase


def test_mass_fraction_distribution_conserves_mass(thermo_path):
    """Mass fractions equal volume fractions for a uniform-density pure solid.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.

    Notes
    -----
    Pure A at 1230 kg/m**3 and kv=0.5 gives single-particle masses of
    [6.15e-10, 4.92e-9, 1.6605e-8, 3.936e-8, 7.6875e-8] kg at sizes
    [100, 200, 300, 400, 500] um, from density times kv times size in metres
    cubed. A 1.23 kg inventory split in the ratio 0:2:3:4:0 therefore puts
    [0, 2.46/9, 3.69/9, 4.92/9, 0] kg in the bins. Expectations use these
    independent particle masses rather than the conversion routine.

    The profile is zero-ended because ``convert_distribution`` normalizes on
    a rectangle (bin-width) basis while ``getMoments`` uses the trapezoidal
    rule, which halves end-node weights. With an end-loaded profile the bases
    disagree, and re-applying the phase's own distribution through
    ``updatePhase`` changes its mass. This pre-existing quadrature mismatch is
    tracked in https://github.com/PharmaPy-org/PharmaPy/issues/269. This
    zero-ended fixture is a provisional compatibility case; add end-loaded
    profiles and update the moment-based mass expectation when #269 is fixed.
    """
    grid = np.array([100.0, 200.0, 300.0, 400.0, 500.0])  # [um]
    weights = np.array([0.0, 2.0, 3.0, 4.0, 0.0])  # [-], unnormalized
    mass = 1.23  # [kg], one litre of pure A
    particle_masses = np.array([
        6.15e-10, 4.92e-9, 1.6605e-8, 3.936e-8, 7.6875e-8,
    ])  # [kg/particle], derived in Notes
    expected_bin_masses = np.array([0, 2.46/9, 3.69/9, 4.92/9, 0])  # [kg]
    phase = SolidPhase(
        thermo_path, mass=mass, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=grid, distrib=weights, distrib_type='mass_frac',
    )
    default_phase = SolidPhase(
        thermo_path, mass=mass, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=grid, distrib=weights,
    )

    reconstructed_bins = phase.distrib * phase.dx * particle_masses  # [kg]
    np.testing.assert_allclose(reconstructed_bins, expected_bin_masses,
                               rtol=REL_TOL, atol=0)
    assert reconstructed_bins.sum() == pytest.approx(mass, rel=REL_TOL)
    assert phase.kv * phase.moments[3] * SOLID_DENSITY == pytest.approx(
        mass, rel=REL_TOL)
    np.testing.assert_allclose(phase.distrib, default_phase.distrib,
                               rtol=REL_TOL, atol=0)


@pytest.mark.parametrize('distrib_type', ['mass_perc', 'unknown', 'VOL_PERC'])
def test_unknown_distribution_type_rejected_at_construction(
        thermo_path, distrib_type):
    """Invalid options fail even without a distribution to trigger conversion.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    distrib_type : str
        Undocumented distribution option.
    """
    with pytest.raises(ValueError, match="distrib_type must be 'vol_perc' or 'mass_frac'"):
        SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION,
                   distrib_type=distrib_type)


@pytest.mark.parametrize('distrib_type', ['vol_perc', 'mass_frac'])
def test_zero_mass_distribution_and_moments_are_preserved(thermo_path, distrib_type):
    """Zero mass keeps raw number density and moment-derived inventory.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    distrib_type : str
        Documented bin-weight basis, inactive when mass is zero.

    Notes
    -----
    On the 50 um grid, the two occupied bins each contribute 1e12 um**3/um
    to the third-moment integrand. Trapezoidal integration gives 1e14 um**3,
    or 1e-4 m**3. With kv=0.5 this is 5e-5 m**3 of pure A.
    """
    expected_moments = np.array([4.5e8, 2.5e4, 1.5, 1e-4])  # [m**n], n=0..3
    expected_volume = 5e-5  # [m**3], derived in Notes
    phase = SolidPhase(
        thermo_path, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=UNIFORM_GRID, distrib=NUMBER_DENSITY, distrib_type=distrib_type,
    )
    moment_phase = SolidPhase(
        thermo_path, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        moments=expected_moments, distrib_type=distrib_type,
    )

    np.testing.assert_array_equal(phase.distrib, NUMBER_DENSITY)
    np.testing.assert_allclose(phase.moments, expected_moments,
                               rtol=REL_TOL, atol=0)
    assert moment_phase.x_distrib is None
    assert not hasattr(moment_phase, 'dx')
    for solid in (phase, moment_phase):
        assert solid.vol == pytest.approx(expected_volume, rel=REL_TOL)
        assert solid.mass == pytest.approx(expected_volume * SOLID_DENSITY,
                                           rel=REL_TOL)


@pytest.mark.parametrize('grid, widths, expected_conversion', [
    (UNIFORM_GRID, UNIFORM_WIDTH, [0.0, 0.25, 0.25, 0.0]),
    (GEOMETRIC_GRID, GEOMETRIC_WIDTHS, [0.0, 2 / 165, 64 / 165, 0.0]),
])
def test_grid_update_refreshes_conversion_widths(
        thermo_path, grid, widths, expected_conversion):
    """Conversion consumes updated widths on uniform and geometric grids.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    grid : numpy.ndarray
        Updated crystal sizes [um].
    widths : float or numpy.ndarray
        Independently calculated bin widths [um].
    expected_conversion : list of float
        Expected output [-] of the existing number-to-volume conversion.

    Notes
    -----
    The uniform grid's third moment is 1e14 um**3; each occupied bin contributes
    0.5 * 50 * 1e12 um**3 after applying kv. On the geometric grid the third
    moment is 187.5 * 8e12 + 750 * 64e12 = 49.5e15 um**3. Bin volumes are
    0.5 * 150 * 8e12 and 0.5 * 600 * 64e12 um**3. These ratios preserve the
    existing conversion's normalization defect: a volume -> number -> volume
    round trip returns ``kv * v_i`` when the quadratures agree. Bin-sum
    quadrature in the conversion also differs from the trapezoidal moments,
    causing further disagreement for geometric grids or nonzero end weights.
    No tracking issue number exists yet; this is recorded as follow-up work
    on this branch. These provisional expectations pin current behavior only
    to isolate #162 and must change when the conversion is corrected.
    """
    phase = SolidPhase(
        thermo_path, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=OLD_GRID, distrib=NUMBER_DENSITY,
    )
    constructed = SolidPhase(
        thermo_path, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=grid, distrib=NUMBER_DENSITY,
    )
    np.testing.assert_allclose(constructed.dx, widths, rtol=REL_TOL, atol=0)

    phase.updatePhase(x_distrib=grid, distrib=NUMBER_DENSITY)

    np.testing.assert_allclose(phase.dx, widths, rtol=REL_TOL, atol=0)
    converted = phase.convert_distribution(num_distr=phase.distrib)  # [-]
    np.testing.assert_allclose(converted, expected_conversion,
                               rtol=REL_TOL, atol=0)

    # A grid-only update must also refresh spacing, without a new distribution.
    phase.updatePhase(x_distrib=OLD_GRID)
    original_width = 100.0  # [um], original #162 probe spacing
    assert phase.dx == pytest.approx(original_width, rel=REL_TOL)


@pytest.mark.parametrize('grid, widths', [
    (UNIFORM_GRID, UNIFORM_WIDTH), (GEOMETRIC_GRID, GEOMETRIC_WIDTHS),
])
def test_slurry_stream_handoff_refreshes_solid_grid(thermo_path, grid, widths):
    """The real slurry setter forwards the grid and flow-scaled distribution.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    grid : numpy.ndarray
        Slurry crystal sizes [um].
    widths : float or numpy.ndarray
        Expected bin widths [um] after the handoff.
    """
    volume_flow = 1e-3  # [m**3/s], one litre per second test stream
    slurry_distribution = NUMBER_DENSITY.copy()  # [#/m**3/um], volume basis
    solid = SolidStream(
        thermo_path, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        x_distrib=OLD_GRID, distrib=NUMBER_DENSITY,
    )
    liquid = LiquidStream(thermo_path, mass_frac=LIQUID_COMPOSITION)
    slurry = SlurryStream(vol_flow=volume_flow, x_distrib=grid,
                          distrib=slurry_distribution)

    slurry.Phases = (liquid, solid)

    assert slurry.Solid_1 is solid
    np.testing.assert_array_equal(solid.x_distrib, grid)
    np.testing.assert_allclose(solid.dx, widths, rtol=REL_TOL, atol=0)
    np.testing.assert_allclose(slurry.dx, solid.dx, rtol=REL_TOL, atol=0)
    expected_distribution = volume_flow * slurry_distribution  # [#/s/um]
    np.testing.assert_allclose(solid.distrib, expected_distribution,
                               rtol=REL_TOL, atol=0)


@pytest.mark.parametrize('grid, widths', [
    (UNIFORM_GRID, UNIFORM_WIDTH), (GEOMETRIC_GRID, GEOMETRIC_WIDTHS),
])
def test_moments_constructor_initializes_grid_widths(thermo_path, grid, widths):
    """Supplied moments retain raw distribution data and initialize bin widths.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    grid : numpy.ndarray
        Crystal-size grid [um], passed as a list to exercise array storage.
    widths : float or numpy.ndarray
        Independently computed bin widths [um].

    Notes
    -----
    Supplied moments describe NUMBER_DENSITY on UNIFORM_GRID, as derived in
    the zero-mass test. On the geometric grid they deliberately differ from
    the raw distribution's moments: construction must preserve their priority.
    A positive explicit mass would trigger normalization without that priority.
    """
    moments = np.array([4.5e8, 2.5e4, 1.5, 1e-4])  # [m**n], n=0..3
    mass = 1.23  # [kg], one litre of pure A
    phase = SolidPhase(
        thermo_path, mass=mass, mass_frac=SOLID_COMPOSITION, kv=SHAPE_FACTOR,
        moments=moments, x_distrib=grid.tolist(), distrib=NUMBER_DENSITY,
        distrib_type='mass_frac',
    )

    assert isinstance(phase.x_distrib, np.ndarray)
    np.testing.assert_array_equal(phase.x_distrib, grid)
    np.testing.assert_array_equal(phase.distrib, NUMBER_DENSITY)
    np.testing.assert_allclose(phase.dx, widths, rtol=REL_TOL, atol=0)
    converted = phase.convert_distribution(num_distr=phase.distrib)  # [-]
    assert converted.shape == NUMBER_DENSITY.shape
    assert np.isfinite(converted).all()
    assert np.all(converted[1:-1] > 0)


def test_grid_update_rejects_single_point(thermo_path):
    """A single size cannot determine a bin width.

    Parameters
    ----------
    thermo_path : str
        Path to the thermodynamic database.
    """
    phase = SolidPhase(thermo_path, mass_frac=SOLID_COMPOSITION)
    with pytest.raises(ValueError, match='at least two grid points'):
        phase.updatePhase(x_distrib=OLD_GRID[:1])
