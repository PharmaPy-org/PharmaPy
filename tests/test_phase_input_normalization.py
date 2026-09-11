"""Type normalization at public phase construction and update boundaries.
Refs:
https://github.com/PharmaPy-org/PharmaPy/issues/173
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase, _as_float_array
from PharmaPy.Streams import LiquidStream, SolidStream, VaporStream


pytestmark = pytest.mark.unit


# Synthetic asymmetric compositions and four-bin populations distinguish species
# ordering from size ordering; updated values exercise actual state changes.
MASS_FRAC_INITIAL = [0.8, 0.2]  # [-]
MASS_FRAC_UPDATED = [0.25, 0.75]  # [-]
X_DISTRIB_INITIAL = [10, 20, 30, 40]  # [um]
X_DISTRIB_UPDATED = [12.0, 24.0, 36.0, 48.0]  # [um]
DISTRIB_INITIAL = [0, 20000000, 10000000, 0]  # [#/um]
DISTRIB_UPDATED = [0.0, 1.0e7, 2.0e7, 0.0]  # [#/um]
MOMENTS_UPDATED = [3.0e8, 9.0e3, 0.3, 1.0e-5]  # heterogeneous [m**n]


@pytest.fixture
def thermo_path(tmp_path):
    """Write a minimal two-component liquid/solid property database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    str
        Path to the temporary thermophysical database.
    """
    # Synthetic constant-density species; coefficients serve only to compare
    # equivalent representations, not to validate a physical correlation.
    database = {
        "A": {
            "mw": 100.0,  # [g/mol]
            "rho_liq": 1000.0,  # [kg/m**3]
            "rho_solid": 1500.0,  # [kg/m**3]
            "cp_liq": [100.0],  # [J/mol/K]
            "cp_solid": [1500.0],  # [J/kg/K]
            "p_vap": [8.0, 1500.0, -40.0],  # Antoine [A [-], B [K], C [K]]
        },
        "B": {
            "mw": 50.0,  # [g/mol]
            "rho_liq": 900.0,  # [kg/m**3]
            "rho_solid": 1500.0,  # [kg/m**3]
            "cp_liq": [120.0],  # [J/mol/K]
            "cp_solid": [1600.0],  # [J/kg/K]
            "p_vap": [8.0, 1800.0, -40.0],  # Antoine [A [-], B [K], C [K]]
        },
    }
    path = tmp_path / "phase_input_database.json"
    path.write_text(json.dumps(database), encoding="utf-8")
    return str(path)


# Representation-only conversion should agree to roundoff in float64.
RTOL = 1e-13  # [-], comfortably above machine epsilon for these short operations


def assert_state_matches(actual, expected, names):
    """Compare nonempty float arrays and scalar state without changing units.

    Parameters
    ----------
    actual, expected : object
        Phase or stream with comparable state in the same physical basis.
    names : sequence of str
        Attributes to compare; compositions [-], concentrations [mol/L] or
        [kg/m**3], sizes [um], distributions [#/um], moments [m**n], density
        [kg/m**3], or amounts [kg, m**3, mol] (per second for streams).
    """
    for name in names:
        value = getattr(actual, name)
        reference = getattr(expected, name)
        if isinstance(reference, np.ndarray):
            assert isinstance(value, np.ndarray), name
            assert value.dtype == np.dtype(float), name
            assert value.shape == reference.shape and value.size > 0, name
        np.testing.assert_allclose(value, reference, rtol=RTOL, atol=0)


@pytest.mark.parametrize('container', [list, tuple, np.array])
@pytest.mark.parametrize('phase_type,boundary,composition', [
    (phase_type, boundary, composition)
    for phase_type in (LiquidPhase, LiquidStream, VaporPhase, VaporStream)
    for boundary in ('constructor', 'update')
    for composition in ('mass_frac', 'mole_frac', 'mass_conc', 'mole_conc')
    if not (boundary == 'constructor' and composition == 'mass_conc'
            and issubclass(phase_type, VaporPhase))
])
def test_fluid_composition_boundaries(thermo_path, container, phase_type,
                                      boundary, composition):
    """Normalize supported compositions before conversion and density calls."""
    values = ([2, 3] if composition.endswith('conc')
              else MASS_FRAC_UPDATED)  # [mol/L], [kg/m**3], or [-]
    amount = ('mass_flow' if issubclass(phase_type, (LiquidStream, VaporStream))
              else 'mass')
    initial = {amount: 1.0, 'verbose': False}  # [kg] or [kg/s], unit inventory
    name = ('concentr' if phase_type is LiquidStream and boundary == 'update'
            and composition == 'mole_conc' else composition)
    if boundary == 'constructor':
        actual = phase_type(thermo_path, **initial,
                            **{name: container(values)})
        expected = phase_type(thermo_path, **initial,
                              **{name: np.array(values, dtype=float)})
    else:
        actual = phase_type(thermo_path, mass_frac=MASS_FRAC_INITIAL, **initial)
        expected = phase_type(thermo_path, mass_frac=MASS_FRAC_INITIAL, **initial)
        actual.updatePhase(**{name: container(values)})
        expected.updatePhase(**{name: np.array(values, dtype=float)})
    names = ['mass_frac', 'mole_frac', 'mass_conc', 'mole_conc', 'mw_av']
    names += (['mass_flow', 'vol_flow', 'mole_flow']
              if phase_type is LiquidStream else ['mass', 'vol', 'moles'])
    if phase_type is VaporStream:
        names += ['mass_flow', 'vol_flow', 'mole_flow']
    assert_state_matches(actual, expected, names)
    for basis in ('mass', 'mole'):
        np.testing.assert_allclose(actual.getDensity(basis=basis),
                                   expected.getDensity(basis=basis),
                                   rtol=RTOL, atol=0)


@pytest.mark.parametrize('container', [list, tuple, np.array])
@pytest.mark.parametrize('phase_type', [SolidPhase, SolidStream])
@pytest.mark.parametrize('boundary', ['constructor', 'grid', 'distrib', 'moments'])
def test_solid_distribution_boundaries(thermo_path, container, phase_type,
                                        boundary):
    """Exercise each solid update independently with an array-based baseline."""
    initial = dict(mass_frac=MASS_FRAC_INITIAL,
                   x_distrib=X_DISTRIB_INITIAL, distrib=DISTRIB_INITIAL)
    if boundary == 'constructor':
        actual = phase_type(thermo_path, **{
            name: container(values) for name, values in initial.items()})
    else:
        actual = phase_type(thermo_path, **{
            name: np.array(values, dtype=float) for name, values in initial.items()})
    expected = phase_type(thermo_path, **{
        name: np.array(values, dtype=float) for name, values in initial.items()})
    updates = {'grid': ('x_distrib', X_DISTRIB_UPDATED),
               'distrib': ('distrib', DISTRIB_UPDATED),
               'moments': ('moments', MOMENTS_UPDATED)}
    if boundary in updates:
        name, values = updates[boundary]  # [um], [#/um], or [m**n]
        actual.updatePhase(**{name: container(values)})
        expected.updatePhase(**{name: np.array(values, dtype=float)})
    names = ['mass_frac', 'mole_frac', 'x_distrib', 'distrib', 'moments',
             'dx', 'mass', 'vol', 'moles']
    if phase_type is SolidStream:
        names += ['mass_flow', 'mole_flow']
    assert_state_matches(actual, expected, names)
    for property_name in ('getDensity', 'getMoments', 'getPorosity'):
        np.testing.assert_allclose(getattr(actual, property_name)(),
                                   getattr(expected, property_name)(),
                                   rtol=RTOL, atol=0)


@pytest.mark.parametrize('container', [list, tuple, np.array])
def test_solid_moments_constructor(thermo_path, container):
    """The moments branch must normalize all supplied population state (#173)."""
    inputs = dict(mass_frac=MASS_FRAC_INITIAL, moments=MOMENTS_UPDATED,
                  x_distrib=X_DISTRIB_UPDATED, distrib=DISTRIB_UPDATED)
    actual = SolidPhase(thermo_path, **{
        name: container(values) for name, values in inputs.items()})
    expected = SolidPhase(thermo_path, **{
        name: np.array(values, dtype=float) for name, values in inputs.items()})
    # Call first to reproduce the issue's delayed failure, independent of dtype.
    np.testing.assert_allclose(actual.getPorosity(), expected.getPorosity(),
                               rtol=RTOL, atol=0)
    assert_state_matches(actual, expected,
                         ['mass_frac', 'mole_frac', 'moments', 'x_distrib',
                          'distrib', 'mass', 'vol', 'moles'])
    moments_only = SolidPhase(thermo_path, mass_frac=MASS_FRAC_INITIAL,
                             moments=container(MOMENTS_UPDATED))
    assert moments_only.x_distrib is None and moments_only.distrib is None
    assert_state_matches(moments_only, expected, ['moments', 'mass', 'vol', 'moles'])


@pytest.mark.parametrize('phase_type', [LiquidPhase, LiquidStream, VaporPhase,
                                       VaporStream, SolidPhase, SolidStream])
@pytest.mark.parametrize('scalar', [int, np.int64, np.float32, np.array])
def test_scalar_temperature_storage(thermo_path, phase_type, scalar):
    """Scalar temperature state is Python float at every supported boundary."""
    options = dict(temp=scalar(300), mass_frac=MASS_FRAC_INITIAL)  # [K], [-]
    if issubclass(phase_type, SolidPhase):
        options['moments' if phase_type is SolidPhase else 'distrib'] = (
            MOMENTS_UPDATED if phase_type is SolidPhase else DISTRIB_INITIAL)
        options['x_distrib'] = X_DISTRIB_INITIAL  # [um]
        if phase_type is SolidPhase:
            options['temp_ref'] = scalar(290)  # [K], distinct reference
    else:
        options['check_input'] = False
    phase = phase_type(thermo_path, **options)
    assert type(phase.temp) is float
    if isinstance(phase, SolidPhase):
        assert type(phase.temp_ref) is float
    # Solid and liquid stream updates do not expose temperature arguments.
    if phase_type in (LiquidPhase, VaporPhase, VaporStream):
        phase.updatePhase(temp=scalar(310))
        assert type(phase.temp) is float
        assert phase.temp == pytest.approx(310, rel=RTOL, abs=0)


@pytest.mark.parametrize('phase_type', [SolidPhase, SolidStream])
@pytest.mark.parametrize('container', [list, tuple, np.array])
def test_solid_temperature_profile(thermo_path, phase_type, container):
    """Keep a three-node temperature field distinct from the two species."""
    temperatures = [290, 300, 310]  # [K], illustrative spatial profile
    phase = phase_type(thermo_path, temp=container(temperatures),
                       mass_frac=MASS_FRAC_INITIAL,
                       x_distrib=X_DISTRIB_INITIAL, distrib=DISTRIB_INITIAL)
    assert isinstance(phase.temp, np.ndarray)
    assert phase.temp.dtype == np.dtype(float)
    np.testing.assert_array_equal(phase.temp, temperatures)
    # Density is temperature independent for this database. Vector Cp for a
    # fixed composition is separately blocked by #248, outside #173's scope.
    assert phase.getDensity() == pytest.approx(1500, rel=RTOL, abs=0)
    phase.updatePhase(moments=MOMENTS_UPDATED)
    np.testing.assert_array_equal(phase.temp, temperatures)


@pytest.mark.parametrize('phase_type', [LiquidPhase, LiquidStream, VaporPhase,
                                       VaporStream, SolidPhase, SolidStream])
def test_array_pressure_is_retained(thermo_path, phase_type):
    """Pressure arrays remain available to AntoineEquation without scalar casts."""
    pressure = np.array([90000, 100000, 110000])  # [Pa], three nearby pressures
    options = dict(pres=pressure, mass_frac=MASS_FRAC_INITIAL)
    if issubclass(phase_type, SolidPhase):
        options.update(x_distrib=X_DISTRIB_INITIAL, distrib=DISTRIB_INITIAL)
    else:
        options['check_input'] = False
    phase = phase_type(thermo_path, **options)
    assert phase.pres is pressure
    actual = phase.AntoineEquation(pres=phase.pres)  # [K]
    expected = np.stack([phase.AntoineEquation(pres=value)
                         for value in pressure])  # [K], independent scalar calls
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)
    if phase_type in (LiquidPhase, VaporPhase, VaporStream):
        phase.updatePhase(pres=pressure)
        assert phase.pres is pressure


@pytest.mark.parametrize('phase_type,composition', [
    (phase_type, composition)
    for phase_type in (LiquidPhase, LiquidStream, VaporPhase, VaporStream,
                       SolidPhase, SolidStream)
    for composition in ('mass_frac', 'mole_frac')
    if not (issubclass(phase_type, SolidPhase) and composition == 'mole_frac')
])
@pytest.mark.parametrize('container', [list, tuple, np.array])
def test_integer_fraction_constructor(thermo_path, phase_type, composition,
                                      container):
    """Pure-species integer inputs must store float fractions, including solids."""
    options = {composition: container([1, 0])}  # [-], pure first species
    if issubclass(phase_type, SolidPhase):
        options.update(x_distrib=X_DISTRIB_INITIAL, distrib=DISTRIB_INITIAL)
    else:
        options['check_input'] = False
    actual = phase_type(thermo_path, **options)
    options[composition] = np.array([1, 0], dtype=float)  # [-]
    expected = phase_type(thermo_path, **options)
    assert_state_matches(actual, expected, ['mass_frac', 'mole_frac'])


@pytest.mark.parametrize('phase_type', [LiquidPhase, LiquidStream,
                                       VaporPhase, VaporStream])
@pytest.mark.parametrize('container', [list, tuple, np.array])
@pytest.mark.parametrize('composition', ['mass_frac', 'mole_frac', 'mole_conc'])
def test_composition_profile_shape_and_order(thermo_path, phase_type, container,
                                             composition):
    """Three distinct rows retain their species axis through both boundaries.

    Zero inventory isolates composition storage and density from amount
    reconciliation; the scalar inventory tests exercise amount preservation.
    """
    fractions = [[0.25, 0.75], [0.6, 0.4], [0.8, 0.2]]  # [-]
    values = ([[1, 3], [3, 2], [4, 1]] if composition == 'mole_conc'
              else fractions)  # [mol/L] or [-], known normalized rows
    options = dict(check_input=False, verbose=False)
    actual = phase_type(thermo_path, **{composition: container(values)}, **options)
    expected = phase_type(thermo_path,
                          **{composition: np.array(values, dtype=float)}, **options)
    names = ['mass_frac', 'mole_frac', 'mass_conc', 'mole_conc']
    assert_state_matches(actual, expected, names)
    fraction_name = 'mole_frac' if composition == 'mole_conc' else composition
    np.testing.assert_allclose(getattr(actual, fraction_name), fractions,
                               rtol=RTOL, atol=0)
    name = ('concentr' if phase_type is LiquidStream and composition == 'mole_conc'
            else composition)
    # Reorder rows to catch normalization that sorts or transposes the state.
    actual.updatePhase(**{name: container(values[::-1])})
    expected.updatePhase(**{name: np.array(values[::-1], dtype=float)})
    assert_state_matches(actual, expected, names)
    np.testing.assert_allclose(getattr(actual, fraction_name), fractions[::-1],
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(actual.getDensity(), expected.getDensity(),
                               rtol=RTOL, atol=0)


@pytest.mark.parametrize('container', [list, np.array])
def test_liquid_update_retains_temperature_profile(thermo_path, container):
    """Preserve the existing vector-temperature update path while coercing scalars."""
    phase = LiquidPhase(thermo_path, mass_frac=MASS_FRAC_INITIAL, check_input=False)
    temperatures = np.array([290., 300., 310.])  # [K], three spatial nodes
    phase.updatePhase(temp=container(temperatures))
    assert isinstance(phase.temp, np.ndarray)
    assert phase.temp.dtype == np.dtype(float)
    np.testing.assert_array_equal(phase.temp, temperatures)


@pytest.mark.parametrize('phase_type', [SolidPhase, SolidStream])
@pytest.mark.parametrize('num_species', [1, 2])
def test_solid_constructor_requires_mass_frac(thermo_path, tmp_path, phase_type,
                                               num_species):
    """Reject omitted composition before constructing invalid solid state."""
    database = json.loads(Path(thermo_path).read_text())
    selected_species = dict(list(database.items())[:num_species])
    path = tmp_path / 'solid_species.json'
    path.write_text(json.dumps(selected_species))
    with pytest.raises(ValueError, match='SolidPhase requires mass_frac'):
        phase_type(str(path))


def test_float_array_normalizer_rejects_none():
    """Missing input must not silently become a scalar NaN."""
    with pytest.raises(TypeError, match='numeric array-like input.*None'):
        _as_float_array(None)


@pytest.mark.parametrize('phase_type', [SolidPhase, SolidStream])
def test_solid_constructor_copies_mass_frac(thermo_path, phase_type):
    """The internal epsilon substitution must not modify caller composition."""
    mass_frac = np.array([1.0, 0.0])  # [-], pure first species exercises eps
    phase = phase_type(thermo_path, mass_frac=mass_frac)
    np.testing.assert_array_equal(mass_frac, [1.0, 0.0])
    assert not np.shares_memory(phase.mass_frac, mass_frac)
