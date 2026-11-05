"""Thermodynamic input and molar-volume contracts (#508, #509, #511).

Three synthetic species have MW = 20, 40, 100 g/mol, liquid densities
800, 1000, 1250 kg/m**3, constant Cp = 40, 120, 250 J/mol/K, and additive
molar volumes 25, 40, 80 cm**3/mol. These are designed algebraic fixtures,
not measured data. Two ten-mole charges (2, 3, 5) and (6, 3, 1) occupy
570 and 350 cm**3 and weigh 0.66 and 0.34 kg. Expectations below use those
amounts and volumes, independently of PharmaPy's conversion helpers.
Real managers, liquid phases and NameAnalyzer handoffs run without a solver.
"""

import copy
import json

import numpy as np
import pytest

from PharmaPy.NameAnalysis import NameAnalyzer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ThermoModule import MissingPropertyError, ThermoPhysicalManager


pytestmark = pytest.mark.unit

# Units per record: mw [g/mol], rho_liq [kg/m**3], constant cp_liq [J/mol/K],
# mol_vol [m**3/mol]. Synthetic values chosen for independent charge balances.
DATABASE = {
    'light': {'mw': 20.0, 'rho_liq': 800.0, 'cp_liq': [40.0],
              'mol_vol': 25e-6},
    'middle': {'mw': 40.0, 'rho_liq': 1000.0, 'cp_liq': [120.0],
               'mol_vol': 40e-6},
    'heavy': {'mw': 100.0, 'rho_liq': 1250.0, 'cp_liq': [250.0],
              'mol_vol': 80e-6},
}
MOLES = np.array([[2.0, 3.0, 5.0], [6.0, 3.0, 1.0]])  # [mol]
MASSES = np.array([[0.04, 0.12, 0.50], [0.12, 0.12, 0.10]])  # [kg]
VOLUMES = np.array([570e-6, 350e-6])  # [m**3], additive charge volumes
MOLE_FRAC = MOLES / MOLES.sum(axis=1, keepdims=True)  # [-]
MASS_FRAC = MASSES / MASSES.sum(axis=1, keepdims=True)  # [-]
TEMPERATURE = 300.0  # [K], arbitrary: fixture properties are constant
LITERS_PER_CUBIC_METER = 1000.0  # [L/m**3], exact conversion
RTOL = 1e-12  # [-], allowance for roundoff in algebraic sums and conversions


def _manager(tmp_path, database=None):
    """Load a real manager from a synthetic property database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Task-owned directory for the JSON database.
    database : dict, optional
        Species properties in the units documented beside ``DATABASE``.

    Returns
    -------
    ThermoPhysicalManager
        Manager preserving the database's species order.
    """
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(DATABASE if database is None else database))
    return ThermoPhysicalManager(str(path))


@pytest.mark.parametrize('basis', ['Mass', 'molar', None])
def test_concentration_rejects_unknown_basis(tmp_path, basis):
    manager = _manager(tmp_path)
    with pytest.raises(ValueError, match="^basis must be 'mass' or 'mole'$"):
        manager.frac_to_conc(mass_frac=MASS_FRAC[0], basis=basis)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('method', ['getCpMix', 'getDensityMix', 'frac_to_conc'])
def test_mixture_requires_fraction_inputs(tmp_path, method, basis):
    manager = _manager(tmp_path)
    kwargs = {'temp': TEMPERATURE} if method == 'getCpMix' else {}
    with pytest.raises(ValueError, match='mass_frac or mole_frac'):
        getattr(manager, method)(basis=basis, **kwargs)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('fraction_name', ['mass_frac', 'mole_frac'])
@pytest.mark.parametrize('rows', [None, 1, 2], ids=['vector', 'one-row', 'profile'])
def test_concentration_lists_and_arrays_preserve_values_and_shapes(
        tmp_path, basis, fraction_name, rows):
    manager = _manager(tmp_path)
    fractions = MASS_FRAC if fraction_name == 'mass_frac' else MOLE_FRAC  # [-]
    expected = (MASSES / VOLUMES[:, None] if basis == 'mass' else
                MOLES / (VOLUMES[:, None] * LITERS_PER_CUBIC_METER))
    # [kg/m**3] or [mol/L], derived from the two explicit charges above.
    fractions = fractions[0] if rows is None else fractions[:rows]  # [-]
    expected = expected[0] if rows is None else expected[:rows]
    for values in (fractions, fractions.tolist()):
        actual = manager.frac_to_conc(**{fraction_name: values}, basis=basis)
        assert actual.shape == fractions.shape
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_concentration_mass_input_precedes_unused_mole_input(tmp_path, basis):
    manager = _manager(tmp_path)
    # The unused input must not affect values, shape, or input coercion.
    actual = manager.frac_to_conc(mass_frac=MASS_FRAC.tolist(),
                                  mole_frac=['unused'], basis=basis)
    expected = (MASSES / VOLUMES[:, None] if basis == 'mass' else
                MOLES / (VOLUMES[:, None] * LITERS_PER_CUBIC_METER))
    # [kg/m**3] or [mol/L]
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_public_liquid_and_name_conversion_keep_concentration_basis(tmp_path, basis):
    manager = _manager(tmp_path)
    liquid = LiquidPhase(manager.path_data, temp=TEMPERATURE,
                         mass=MASSES[0].sum(), mass_frac=MASS_FRAC[0])
    analyzer = NameAnalyzer(['mass_frac'], ['mass_conc' if basis == 'mass'
                                          else 'mole_conc'], 3)
    liquid.y_upstream = {'mass_frac': MASS_FRAC}  # [-], two upstream samples
    actual = analyzer.convertUnits(liquid)
    expected = (MASSES / VOLUMES[:, None] if basis == 'mass' else
                MOLES / (VOLUMES[:, None] * LITERS_PER_CUBIC_METER))
    # [kg/m**3] or [mol/L]
    # NameAnalyzer's public return is keyed by downstream state name.
    np.testing.assert_allclose(next(iter(actual.values())), expected,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(liquid.mole_conc,
                               MOLES[0] / (VOLUMES[0] * LITERS_PER_CUBIC_METER),
                               rtol=RTOL, atol=0)


@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('rows', [None, 1, 2], ids=['vector', 'one-row', 'profile'])
@pytest.mark.parametrize('as_list', [False, True], ids=['array', 'list'])
@pytest.mark.parametrize('method', ['getCpMix', 'getDensityMix'])
def test_cp_and_density_fraction_precedence_and_conversion(
        tmp_path, basis, rows, as_list, method):
    """Check selected and converted fractions against independent charge totals.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the synthetic database.
    basis : {'mass', 'mole'}
        Cp and density basis: [J/kg/K], [kg/m**3] or [J/mol/K], [mol/L].
    rows : int or None
        Number of charge rows, or None for one fraction vector [-].
    as_list : bool
        Whether to supply fractions as lists instead of arrays.
    method : {'getCpMix', 'getDensityMix'}
        Mixture property evaluated on the selected basis.
    """
    manager = _manager(tmp_path)
    selected = MASS_FRAC if basis == 'mass' else MOLE_FRAC  # [-]
    converted = MOLE_FRAC if basis == 'mass' else MASS_FRAC  # [-]
    other_name = 'mole_frac' if basis == 'mass' else 'mass_frac'
    selected_name = 'mass_frac' if basis == 'mass' else 'mole_frac'
    expected_cp = np.array([1690.0, 850.0])  # [J/K], sum of charge n_i Cp_i
    amounts = MASSES.sum(axis=1) if basis == 'mass' else MOLES.sum(axis=1)
    # [kg] or [mol], charge amount for each requested property basis
    expected_cp = expected_cp / amounts  # [J/kg/K] or [J/mol/K]
    expected_density = amounts / VOLUMES  # [kg/m**3] or [mol/m**3]
    if basis == 'mole':
        expected_density = expected_density / LITERS_PER_CUBIC_METER  # [mol/L]
    selected = selected[0] if rows is None else selected[:rows]  # [-]
    converted = converted[0] if rows is None else converted[:rows]  # [-]
    expected_cp = expected_cp[0] if rows is None else expected_cp[:rows]
    # [J/kg/K] or [J/mol/K]
    expected_density = (expected_density[0] if rows is None else
                        expected_density[:rows])  # [kg/m**3] or [mol/L]
    if as_list:
        selected = selected.tolist()  # [-]
        converted = converted.tolist()  # [-]
    for kwargs in ({selected_name: selected, other_name: ['unused']},
                   {other_name: converted}):
        if method == 'getCpMix':
            actual = manager.getCpMix(TEMPERATURE, basis=basis, **kwargs)
            # [J/kg/K] or [J/mol/K]
            expected = expected_cp  # [J/kg/K] or [J/mol/K]
        else:
            actual = manager.getDensityMix(basis=basis, **kwargs)
            # [kg/m**3] or [mol/L]
            expected = expected_density  # [kg/m**3] or [mol/L]
        assert np.shape(actual) == np.shape(expected)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize('order', [[0, 1, 2], [2, 0, 1]])
def test_molar_volume_preserves_species_first_profiles_and_property_array(tmp_path, order):
    names = list(DATABASE)
    manager = _manager(tmp_path, {names[i]: DATABASE[names[i]] for i in order})
    stored = manager.mol_vol.copy()  # [m**3/mol]
    fractions = MOLE_FRAC[:, order].T  # [-], species first, legacy dot layout
    expected = VOLUMES / MOLES.sum(axis=1)  # [m**3/mol]
    for values in (fractions, fractions.tolist()):
        actual = manager.getMolVolMix(values)  # [m**3/mol]
        assert actual.shape == (2,)
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)
    vector = manager.getMolVolMix(fractions[:, 0])  # [m**3/mol]
    assert np.shape(vector) == ()
    assert vector == pytest.approx(expected[0], rel=RTOL, abs=0)
    one_column = manager.getMolVolMix(fractions[:, :1])  # [m**3/mol]
    assert one_column.shape == (1,)
    np.testing.assert_allclose(one_column, expected[:1], rtol=RTOL, atol=0)
    np.testing.assert_array_equal(manager.mol_vol, stored)


@pytest.mark.parametrize('missing', ['partial', 'absent', 'null'])
def test_molar_volume_names_only_contributing_missing_species(tmp_path, missing):
    database = copy.deepcopy(DATABASE)
    if missing == 'partial':
        database['middle'].pop('mol_vol')
    else:
        for record in database.values():
            if missing == 'absent':
                record.pop('mol_vol')
            else:
                record['mol_vol'] = None
    manager = _manager(tmp_path, database)
    stored = manager.mol_vol.copy() if hasattr(manager, 'mol_vol') else None
    # Species middle appears only in the second profile column.
    fractions = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # [-]
    selected = fractions if missing == 'partial' else fractions[:, 1]
    with pytest.raises(MissingPropertyError,
                       match=r"^Species \['middle'\].*'mol_vol'.*m\*\*3/mol"):
        manager.getMolVolMix(selected)
    if stored is not None:
        np.testing.assert_array_equal(manager.mol_vol, stored)
    else:
        assert not hasattr(manager, 'mol_vol')


def test_zero_fraction_missing_molar_volume_contributes_zero_without_mutation(tmp_path):
    database = copy.deepcopy(DATABASE)
    database['middle'].pop('mol_vol')
    manager = _manager(tmp_path, database)
    stored = manager.mol_vol.copy()  # [m**3/mol], includes missing-data NaN
    fractions = np.array([[0.2, 1.0], [0.0, 0.0], [0.8, 0.0]])  # [-]
    # 0.2*25 + 0.8*80 = 69 cm**3/mol; second column is pure light.
    expected = np.array([69e-6, 25e-6])  # [m**3/mol]
    np.testing.assert_allclose(manager.getMolVolMix(fractions), expected,
                               rtol=RTOL, atol=0)
    assert manager.getMolVolMix(fractions[:, 1]) == pytest.approx(
        expected[1], rel=RTOL, abs=0)
    np.testing.assert_array_equal(manager.mol_vol, stored)


@pytest.mark.parametrize('fractions', [
    [0.5, 0.5],
    MOLE_FRAC.tolist(),
    1.0,
    MOLE_FRAC[None, ...].tolist(),
], ids=['short-vector', 'row-first-profile', 'scalar', 'three-dimensional'])
def test_molar_volume_rejects_invalid_fraction_shapes(tmp_path, fractions):
    """Require a species-first vector or profile before property evaluation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the synthetic three-species database.
    fractions : array-like
        Invalid mole-fraction shape [-]; the row-first case has two charges.
    """
    manager = _manager(tmp_path)
    with pytest.raises(
            ValueError,
            match=r'^frac must have shape \(num_species,\) or '
                  r'\(num_species, num_points\) in database species order$'):
        manager.getMolVolMix(fractions)
