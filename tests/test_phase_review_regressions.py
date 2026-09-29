"""Public density compatibility, vapor profiles, and rejected solid updates.

The shipped database fixes species masses and a pure solid density. Unequal
profile and species dimensions expose accidental broadcasting; no solver runs.
"""

import numpy as np
import pytest

from PharmaPy.Phases import SolidPhase, VaporPhase
from PharmaPy.Streams import VaporStream


pytestmark = pytest.mark.unit

COMPOSITION = np.array([.25, .75, 0., 0., 0.])  # [-], shipped A/B mixture
PROFILE = np.array([
    [.25, .75, 0., 0., 0.],
    [.6, .4, 0., 0., 0.],
    [.8, .2, 0., 0., 0.],
])  # [-], three rows and five species
PROFILE_MW = np.array([62.5, 80., 90.])  # [g/mol], A=100 and B=50 g/mol
GAS_CONSTANT = 8.314  # [J/mol/K], existing package ideal-gas convention
TEMPERATURE = 350.  # [K], stored state distinct from density overrides
PRESSURE = 200000.  # [Pa], stored state distinct from density overrides
REL_TOL = 1e-12  # [-], roundoff allowance for direct algebra


@pytest.fixture
def thermo_path(data_path):
    """Locate the shipped thermophysical database.

    Parameters
    ----------
    data_path : dict
        Repository data directories supplied by conftest.

    Returns
    -------
    str
        Path to the five-species thermophysical database.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


@pytest.mark.parametrize('phase_class', [VaporPhase, VaporStream])
@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('call_style', ['positional', 'full-positional', 'legacy-keywords', 'modern-keywords'])
def test_vapor_density_preserves_legacy_overrides(thermo_path, phase_class, basis, call_style):
    amount = 'mole_flow' if phase_class is VaporStream else 'moles'
    vapor = phase_class(thermo_path, mole_frac=COMPOSITION,
                        temp=TEMPERATURE, pres=PRESSURE, **{amount: 1.})
    override_pressure = 101325.  # [Pa], standard atmosphere
    override_temperature = 300.  # [K], synthetic override distinct from storage
    if call_style == 'positional':
        actual = vapor.getDensity(override_pressure, override_temperature, basis=basis)
    elif call_style == 'full-positional':
        actual = vapor.getDensity(override_pressure, override_temperature, 'gas', basis)
    elif call_style == 'legacy-keywords':
        actual = vapor.getDensity(pres_gas=override_pressure,
                                  temp_gas=override_temperature, basis=basis)
    else:
        actual = vapor.getDensity(pres=override_pressure,
                                  temp=override_temperature, basis=basis)
    expected = override_pressure / (GAS_CONSTANT * override_temperature) / 1000  # [mol/L]
    if basis == 'mass':
        expected = expected * 62.5  # [kg/m**3], mixture MW [g/mol] and g/L=kg/m**3
    assert np.ndim(actual) == 0
    assert actual == pytest.approx(expected, rel=REL_TOL, abs=0)
    assert vapor.temp == TEMPERATURE
    assert vapor.pres == PRESSURE


@pytest.mark.parametrize('legacy, modern, value', [
    ('pres_gas', 'pres', 101325.),  # [Pa], duplicate pressure override
    ('temp_gas', 'temp', 300.),  # [K], duplicate temperature override
])
def test_vapor_density_rejects_duplicate_override(thermo_path, legacy, modern, value):
    vapor = VaporPhase(thermo_path, mole_frac=COMPOSITION, moles=1.)
    with pytest.raises(ValueError, match=f"{legacy}.*{modern}"):
        vapor.getDensity(**{legacy: value, modern: value})


@pytest.mark.parametrize('phase_class', [VaporPhase, VaporStream])
@pytest.mark.parametrize('amount', ['mass', 'vol'])
@pytest.mark.parametrize('entry_point', ['constructor', 'update'])
def test_vapor_positive_inventory_preserves_composition_profiles(
        thermo_path, phase_class, amount, entry_point):
    argument = ({'mass': 'mass_flow', 'vol': 'vol_flow'}[amount]
                if phase_class is VaporStream else amount)
    options = dict(temp=TEMPERATURE, pres=PRESSURE, mole_frac=PROFILE)
    if entry_point == 'constructor':
        vapor = phase_class(thermo_path, **options, **{argument: 1.})
    else:
        vapor = phase_class(thermo_path, **options, check_input=False)
        vapor.updatePhase(**{argument: 1.})
    expected_moles = (1000 / PROFILE_MW if amount == 'mass' else
                      np.full(3, PRESSURE / (GAS_CONSTANT * TEMPERATURE)))  # [mol] or [mol/s]
    # A temperature-only update must conserve each row's inventory even when
    # the mass basis has produced an array of differing molar inventories.
    vapor.updatePhase(temp=2 * TEMPERATURE)
    np.testing.assert_allclose(vapor.moles, expected_moles, rtol=REL_TOL, atol=0)
    np.testing.assert_allclose(vapor.mass, expected_moles * PROFILE_MW / 1000,
                               rtol=REL_TOL, atol=0)  # [kg] or [kg/s]
    np.testing.assert_allclose(vapor.vol, expected_moles * GAS_CONSTANT * 2 * TEMPERATURE / PRESSURE,
                               rtol=REL_TOL, atol=0)  # [m**3] or [m**3/s]
    np.testing.assert_array_equal(vapor.mole_frac, PROFILE)
    if phase_class is VaporStream:
        np.testing.assert_allclose(vapor.mole_flow, expected_moles, rtol=REL_TOL, atol=0)
        np.testing.assert_array_equal(vapor.mass_flow, vapor.mass)
        np.testing.assert_array_equal(vapor.vol_flow, vapor.vol)


@pytest.mark.parametrize('invalid_grid', [[], [100.]])  # [um], insufficient nodes for widths
def test_rejected_solid_grid_keeps_existing_distribution_state(thermo_path, invalid_grid):
    grid = np.array([0., 100., 200., 300.])  # [um], uniform reference grid
    population = np.array([0., 8e6, 1e6, 0.])  # [#/um], asymmetric occupied bins
    solid = SolidPhase(thermo_path, mass_frac=[1., 0., 0., 0., 0.],
                       x_distrib=grid, distrib=population)
    before = {name: np.copy(getattr(solid, name)) for name in
              ('x_distrib', 'dx', 'distrib', 'moments', 'mass', 'vol', 'moles')}
    with pytest.raises(ValueError, match='at least two grid points'):
        solid.updatePhase(x_distrib=invalid_grid, distrib=2 * population)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(solid, name), original)
    # The retained grid must remain usable by a later public distribution update.
    solid.updatePhase(distrib=2 * population)
    np.testing.assert_allclose(solid.moments, 2 * before['moments'], rtol=REL_TOL, atol=0)
    assert solid.mass == pytest.approx(2 * before['mass'], rel=REL_TOL, abs=0)


@pytest.mark.parametrize('phase_class', [VaporPhase, VaporStream])
@pytest.mark.parametrize('amount', ['mass', 'vol'])
def test_rejected_vapor_composition_keeps_state_and_flow_aliases(
        thermo_path, phase_class, amount):
    """Keep the previous gas state when positive mass or volume is invalid.

    Parameters
    ----------
    thermo_path : str
        Shipped thermophysical database path.
    phase_class : type
        Vapor inventory or stream class under test.
    amount : str
        Positive mass [kg] or volume [m**3] argument, or its flow alias.
    """
    is_stream = phase_class is VaporStream
    initial_amount = 'mole_flow' if is_stream else 'moles'
    argument = f'{amount}_flow' if is_stream else amount
    vapor = phase_class(thermo_path, mole_frac=COMPOSITION,
                        temp=TEMPERATURE, pres=PRESSURE, **{initial_amount: 1.})
    attributes = ['temp', 'pres', 'moles', 'mass', 'vol', 'mw_av',
                  'mole_frac', 'mass_frac', 'mole_conc', 'mass_conc']
    if is_stream:
        attributes += ['mole_flow', 'mass_flow', 'vol_flow']
    before = {name: np.copy(getattr(vapor, name)) for name in attributes}
    new_temperature = 2 * TEMPERATURE  # [K], doubled temperature probe
    new_pressure = 3 * PRESSURE  # [Pa], tripled pressure probe
    with pytest.raises(ValueError, match='zero mixture molar mass'):
        vapor.updatePhase(mole_frac=np.zeros_like(COMPOSITION),
                          temp=new_temperature, pres=new_pressure,
                          **{argument: 1.})  # [kg] or [m**3], per second for streams
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(vapor, name), original)

    # A subsequent valid update must use its new state and preserve inventory.
    vapor.updatePhase(temp=new_temperature, pres=new_pressure)
    assert vapor.temp == new_temperature
    assert vapor.pres == new_pressure
    assert vapor.moles == pytest.approx(before['moles'], rel=REL_TOL, abs=0)
    assert vapor.mass == pytest.approx(before['mass'], rel=REL_TOL, abs=0)
    assert vapor.vol == pytest.approx(before['vol'] * 2 / 3, rel=REL_TOL, abs=0)
    if is_stream:
        assert vapor.mole_flow == vapor.moles
        assert vapor.mass_flow == vapor.mass
        assert vapor.vol_flow == vapor.vol
