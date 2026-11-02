"""Field records and their legacy array/dictionary adapters (#351).

Scope: ``PharmaPy.Records.FieldIdentity``/``FieldRecord`` validation and
JSON round trips, and the documented adapters ``fields_from_array``,
``fields_to_array``, ``fields_from_mapping`` and ``fields_to_mapping`` that
convert legacy state arrays (rows = samples, columns = fields) and
``{name: array}`` dictionaries without changing values, ordering or dtype.

Fixtures are small asymmetric arrays (five samples, three fields, a
replicate time and NaN entries at distinct positions) whose expected
columns are written out by hand. A multi-basis phase composition (mass
fractions, molar concentrations, component and mixture densities) uses
the worked example posted on #336; its internal consistency is checked
independently. Core lane; no solver is used.
"""

import json

import numpy as np
import pytest

from PharmaPy.Records import (FieldIdentity, FieldRecord, fields_from_array,
                              fields_from_mapping, fields_to_array,
                              fields_to_mapping)

pytestmark = pytest.mark.unit

TIME = [0.0, 0.5, 0.5, 2.0, 7.25]  # [s], with a replicate at 0.5 s
C_A = [2.0, 1.5, 1.49, np.nan, 0.1]  # [mol/L], NaN = missing
TEMP = [300.0, 301.5, 301.6, 305.0, 310.25]  # [K]
MOMENT = [0.0, 1e10, 1.1e10, 3.3e11, np.nan]  # [#/m**3]
STATES = np.column_stack((C_A, TEMP, MOMENT))  # c_A, T, mu_0
IDENTITIES = (
    FieldIdentity('c_A', 'state', 'mol/L', basis='molar concentration'),
    FieldIdentity('T', 'state', 'K'),
    FieldIdentity('mu_0', 'state', '#/m**3',
                  basis='number per unit volume'),
)

# Worked example by Daniel Casas-Orozco on #336 (issue comment
# 6200391817): LiquidPhase(path, mass=1, mole_conc={'A': 1.0, 'B': 1.0},
# mass_frac={'ACN': 0.3, 'THF': 0.7}) and its results as posted, rounded
# to three decimals. A and B are solutes of his component file; ACN is
# acetonitrile and THF tetrahydrofuran. Species keep his (non-alphabetical)
# order. LiquidPhase cannot take both measures yet; its own regression test
# belongs to #336.
SPECIES = ('A', 'B', 'ACN', 'THF')
# A phase specification has no time axis; one sample at 0 s stands in for
# the record's independent variable (default x_units 's').
PHASE_TIME = [0.0]  # [s]
MASS_FRAC = [[0.108, 0.216, 0.203, 0.473]]  # [-], mass basis
MOLE_CONC = [[1.000, 1.000, 4.580, 6.083]]  # [mol/L]
RHO_SPECIES = [[1096.226, 1096.226, 777.719, 905.234]]  # [kg/m**3]
RHO_MIX = [926.677]  # [kg/m**3]
# Largest rounding error of a value posted with three decimals, in the
# units of that value.
POSTED_HALF_UNIT = 5e-4
PHASE_IDENTITIES = (
    FieldIdentity('mass_frac', 'state', '-', basis='mass fraction'),
    FieldIdentity('mole_conc', 'state', 'mol/L',
                  basis='molar concentration'),
    FieldIdentity('rho_species', 'algebraic', 'kg/m**3',
                  basis='pure-component liquid mass density'),
    FieldIdentity('rho_mix', 'algebraic', 'kg/m**3',
                  basis='mixture mass density'),
)


def assert_bits_equal(actual, expected):
    """Assert two float arrays are equal bit for bit, NaN included.

    Parameters
    ----------
    actual, expected : array_like
        Float arrays of equal shape.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.int64),
                                  expected.view(np.int64))


def test_array_adapter_splits_columns_in_identity_order():
    fields = fields_from_array(TIME, STATES, IDENTITIES, experiment='run')

    assert [field.name for field in fields] == ['c_A', 'T', 'mu_0']
    for field, column in zip(fields, (C_A, TEMP, MOMENT)):
        assert_bits_equal(field.values, column)
        assert_bits_equal(field.x, TIME)
        assert (field.x_units, field.experiment, field.columns) == (
            's', 'run', None)
    assert fields[2].identity.basis == 'number per unit volume'


def test_array_round_trip_is_exact_and_reorders_by_name():
    fields = fields_from_array(TIME, STATES, IDENTITIES)

    x, array = fields_to_array(fields)
    assert_bits_equal(x, TIME)
    assert_bits_equal(array, STATES)

    x, reordered = fields_to_array(fields, names=['mu_0', 'c_A'])
    assert_bits_equal(reordered, np.column_stack((MOMENT, C_A)))
    reordered[0, 0] = -1.0  # returned arrays are independent copies
    assert fields[2].values[0] == 0.0


def test_one_dimensional_array_is_one_field():
    field, = fields_from_array(TIME, TEMP, IDENTITIES[1:2], x_units='min')
    assert_bits_equal(field.values, TEMP)
    x, array = fields_to_array([field])
    assert array.shape == (5, 1)
    assert_bits_equal(array[:, 0], TEMP)


def test_integer_arrays_are_stored_as_float64():
    counts = np.array([[1, 2], [3, 4]], dtype=np.int32)  # [-]
    identities = (FieldIdentity('n_1', 'algebraic', '-'),
                  FieldIdentity('n_2', 'algebraic', '-'))
    _, array = fields_to_array(fields_from_array([0.0, 1.0], counts,
                                                 identities))
    assert_bits_equal(array, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize('x, array, identities, message', [
    (TIME, STATES[:, :2], IDENTITIES, "array has 2 columns but 3 "
                                      "identities"),
    (TIME[:4], STATES, IDENTITIES, r"one row per x sample \(n = 4\)"),
    (TIME, STATES[:, :, np.newaxis], IDENTITIES,
     "must be one- or two-dimensional"),
    (TIME, STATES[:, :2], IDENTITIES[:1] * 2,
     r"identities names must be unique; repeated \['c_A'\]"),
    (TIME[::-1], STATES, IDENTITIES,
     r"x must be non-decreasing; x\[1\] = 2.0 is smaller than x\[0\]"),
    (TIME, np.where(np.isnan(STATES), np.inf, STATES), IDENTITIES,
     "values must be finite or NaN"),
])
def test_array_adapter_rejects_mismatches(x, array, identities, message):
    with pytest.raises(ValueError, match=message):
        fields_from_array(x, array, identities)


def test_array_assembly_needs_one_shared_grid_and_known_names():
    fields = fields_from_array(TIME, STATES, IDENTITIES)
    shifted = FieldRecord(FieldIdentity('c_B', 'state', 'mol/L'),
                          np.add(TIME, 1.0), C_A)  # [s], [mol/L]
    other_units = FieldRecord(FieldIdentity('c_B', 'state', 'mol/L'),
                              TIME, C_A, x_units='h')
    with pytest.raises(ValueError, match="different x samples or x_units"):
        fields_to_array(fields + (shifted,))
    with pytest.raises(ValueError, match="different x samples or x_units"):
        fields_to_array(fields + (other_units,))
    elsewhere = fields_from_array(TIME, STATES[:, :1],
                                  (FieldIdentity('c_B', 'state', 'mol/L'),),
                                  experiment='cold')
    with pytest.raises(ValueError, match="belong to different experiments "
                                         r"\(None and 'cold'\)"):
        fields_to_array(fields + elsewhere)
    with pytest.raises(ValueError, match="names must contain at least one "
                                         "field name"):
        fields_to_array(fields, names=[])
    with pytest.raises(ValueError, match=r"names \['c_B'\] are not among"):
        fields_to_array(fields, names=['T', 'c_B'])
    with pytest.raises(ValueError, match=r"repeated \['c_A'\]"):
        fields_to_array(fields + fields[:1])
    with pytest.raises(ValueError, match="at least one FieldRecord"):
        fields_to_array([])
    psd = FieldRecord(FieldIdentity('psd', 'state', '#/m**4'), [0.0, 1.0],
                      [[1.0, 2.0], [3.0, 4.0]], columns=('L1', 'L2'))
    with pytest.raises(ValueError, match="use fields_to_mapping"):
        fields_to_array([psd])


def test_mapping_round_trip_preserves_declared_order_and_values():
    # Non-alphabetical declared order; a two-dimensional distribution.
    distribution = [[1.0, 0.5], [0.9, np.nan], [0.8, 0.4], [0.2, 0.1],
                    [0.0, 0.0]]  # [#/m**4]
    mapping = {'T': TEMP, 'psd': distribution, 'c_A': C_A}
    identities = {'c_A': IDENTITIES[0], 'T': IDENTITIES[1],
                  'psd': FieldIdentity('psd', 'state', '#/m**4')}

    fields = fields_from_mapping(TIME, mapping, identities,
                                 columns={'psd': ('L_10um', 'L_20um')})
    assert [field.name for field in fields] == ['T', 'psd', 'c_A']
    assert fields[1].columns == ('L_10um', 'L_20um')

    x, restored = fields_to_mapping(fields)
    assert_bits_equal(x, TIME)
    assert list(restored) == ['T', 'psd', 'c_A']
    assert_bits_equal(restored['T'], TEMP)
    assert_bits_equal(restored['psd'], distribution)
    assert_bits_equal(restored['c_A'], C_A)
    # fields_to_mapping inverts fields_from_mapping, x included.
    again = fields_from_mapping(x, restored, identities,
                                columns={'psd': fields[1].columns})
    assert again == fields
    restored['T'][0] = 0.0  # returned arrays are writable copies
    assert fields[0].values[0] == 300.0

    by_sequence = fields_from_mapping(TIME, {'c_A': C_A}, IDENTITIES[:1])
    assert by_sequence[0].identity == IDENTITIES[0]



def test_mapping_assembly_needs_one_shared_grid_and_experiment():
    hot = fields_from_mapping(TIME, {'c_A': C_A}, IDENTITIES[:1],
                              experiment='hot')
    cold = fields_from_mapping(TIME, {'T': TEMP}, IDENTITIES[1:2],
                               experiment='cold')
    later = fields_from_mapping(np.add(TIME, 1.0), {'T': TEMP},
                                IDENTITIES[1:2], experiment='hot')
    with pytest.raises(ValueError, match="belong to different experiments "
                                         r"\('hot' and 'cold'\)"):
        fields_to_mapping(hot + cold)
    with pytest.raises(ValueError, match="different x samples or x_units"):
        fields_to_mapping(hot + later)
    with pytest.raises(ValueError, match="at least one FieldRecord"):
        fields_to_mapping([])


@pytest.mark.parametrize('mapping, identities, columns, message', [
    ({'c_A': C_A}, IDENTITIES[:2], None,
     r"lacks identified fields \['T'\]"),
    ({'c_A': C_A, 'c_B': C_A}, IDENTITIES[:1], None,
     r"holds fields without identity \['c_B'\]"),
    ({'c_A': C_A}, {'conc': IDENTITIES[0]}, None,
     "identities key 'conc' differs from its identity name 'c_A'"),
    ({'c_A': C_A}, IDENTITIES[:1], {'c_B': ('a',)},
     r"columns names unknown fields \['c_B'\]"),
    ({'c_A': STATES[:, :2]}, IDENTITIES[:1], None,
     "pass columns with one label per component"),
    ({'c_A': STATES[:, :2]}, IDENTITIES[:1], {'c_A': ('a', 'a')},
     r"column names must be unique; repeated \['a'\]"),
    ({'c_A': STATES[:, :2]}, IDENTITIES[:1], {'c_A': ('a',)},
     "one column label per component; got 1 labels"),
    ({'c_A': C_A}, IDENTITIES[:1], {'c_A': ('a',)},
     "one-dimensional values; columns must be None"),
])
def test_mapping_adapter_rejects_mismatches(mapping, identities, columns,
                                            message):
    with pytest.raises(ValueError, match=message):
        fields_from_mapping(TIME, mapping, identities, columns=columns)


def test_field_record_json_round_trip_is_exact():
    field = FieldRecord(FieldIdentity('x_liq', 'state', '-',
                                      basis='mole fraction',
                                      description='liquid composition'),
                        [0.0, 1.0], [[0.1 + 0.2, np.nan], [1 / 3, -0.0]],
                        columns=('A', 'B'), experiment='run')  # [s], [-]
    text = json.dumps(field.to_dict(), allow_nan=False)
    assert 'NaN' not in text
    restored = FieldRecord.from_dict(json.loads(text))

    assert restored == field
    assert restored.identity.description == 'liquid composition'
    assert_bits_equal(restored.values, [[0.1 + 0.2, np.nan], [1 / 3, -0.0]])
    assert (restored.columns, restored.experiment) == (('A', 'B'), 'run')


def test_field_records_are_read_only_copies():
    values = np.array(TEMP)  # [K]
    field = FieldRecord(IDENTITIES[1], TIME, values)
    values[0] = 0.0
    assert field.values[0] == 300.0
    with pytest.raises(ValueError, match='read-only'):
        field.values[0] = 0.0
    with pytest.raises(AttributeError, match='immutable'):
        field.experiment = 'run'


@pytest.mark.parametrize('build, error, message', [
    (lambda: FieldRecord('c_A', TIME, C_A), TypeError,
     "identity must be a FieldIdentity"),
    (lambda: FieldRecord(IDENTITIES[0], [], []), ValueError,
     "x must contain at least one sample"),
    (lambda: FieldRecord(IDENTITIES[0], [0.0, np.nan], [1.0, 2.0]),
     ValueError, "x must be finite"),
    (lambda: FieldRecord(IDENTITIES[0], TIME, np.array(C_A) * 1j),
     TypeError, "values must be real"),
    (lambda: FieldRecord(IDENTITIES[0], TIME, C_A, x_units=''), ValueError,
     "x_units must be a non-empty str"),
    (lambda: FieldIdentity('', 'state', 'K'), ValueError,
     "FieldIdentity name must be a non-empty str"),
    (lambda: FieldIdentity('T', 'state', None), TypeError,
     "FieldIdentity units must be a str"),
])
def test_field_record_validation(build, error, message):
    with pytest.raises(error, match=message):
        build()


def test_multi_basis_phase_composition_round_trips_units_and_bases():
    mapping = {'mass_frac': MASS_FRAC, 'mole_conc': MOLE_CONC,
               'rho_species': RHO_SPECIES, 'rho_mix': RHO_MIX}
    columns = {'mass_frac': SPECIES, 'mole_conc': SPECIES,
               'rho_species': SPECIES}
    fields = fields_from_mapping(PHASE_TIME, mapping, PHASE_IDENTITIES,
                                 columns=columns, experiment='issue_336')
    text = json.dumps([field.to_dict() for field in fields], allow_nan=False)
    restored = tuple(FieldRecord.from_dict(data)
                     for data in json.loads(text))

    assert restored == fields
    # The two densities share units; only the basis tells them apart.
    assert [(field.name, field.identity.kind, field.identity.units,
             field.identity.basis) for field in restored] == [
        ('mass_frac', 'state', '-', 'mass fraction'),
        ('mole_conc', 'state', 'mol/L', 'molar concentration'),
        ('rho_species', 'algebraic', 'kg/m**3',
         'pure-component liquid mass density'),
        ('rho_mix', 'algebraic', 'kg/m**3', 'mixture mass density')]
    assert [field.columns for field in restored] == [SPECIES] * 3 + [None]
    assert {field.experiment for field in restored} == {'issue_336'}
    assert dict(zip(restored[1].columns, restored[1].values[0])) == {
        'A': 1.0, 'B': 1.0, 'ACN': 4.58, 'THF': 6.083}  # [mol/L]

    x, values = fields_to_mapping(restored)
    assert_bits_equal(x, PHASE_TIME)
    assert list(values) == list(mapping)
    for name, expected in mapping.items():
        assert_bits_equal(values[name], expected)

    # Reordered species keep their own order and carry their values along.
    # A and B share mole_conc and rho_species, so only mass_frac would show
    # an A/B swap.
    order = [3, 0, 2, 1]  # THF, A, ACN, B
    shuffled = fields_from_mapping(
        PHASE_TIME,
        {name: np.asarray(mapping[name])[:, order] for name in columns},
        PHASE_IDENTITIES[:3],
        columns={name: tuple(SPECIES[i] for i in order)
                 for name in columns})
    shuffled = tuple(FieldRecord.from_dict(json.loads(json.dumps(
        field.to_dict(), allow_nan=False))) for field in shuffled)
    assert len(shuffled) == len(columns)
    for original, moved in zip(restored, shuffled):
        assert moved.columns == ('THF', 'A', 'ACN', 'B')
        assert (dict(zip(moved.columns, moved.values[0]))
                == dict(zip(original.columns, original.values[0])))


def test_issue_336_worked_example_is_self_consistent():
    # Each check allows the first-order propagation of the rounding of
    # every posted value, POSTED_HALF_UNIT in its own units.
    mass_frac = np.asarray(MASS_FRAC[0])  # [-]
    mole_conc = np.asarray(MOLE_CONC[0])  # [mol/L]
    rho_species = np.asarray(RHO_SPECIES[0])  # [kg/m**3]
    rho_mix = RHO_MIX[0]  # [kg/m**3]
    half = POSTED_HALF_UNIT

    assert abs(mass_frac.sum() - 1.0) <= mass_frac.size * half

    # The solvent keeps the specified blend, mass_frac={'ACN': 0.3, ...};
    # r = w_ACN / s with s = w_ACN + w_THF has
    # |dr/dw_ACN| + |dr/dw_THF| = 1/s.
    specified_acn = 0.3  # [-], ACN share of the solvent
    acn = mass_frac[SPECIES.index('ACN')]  # [-]
    thf = mass_frac[SPECIES.index('THF')]  # [-]
    solvent = acn + thf  # [-]
    assert abs(acn / solvent - specified_acn) <= half / solvent

    # Additive volumes, the mixing rule of the #336 balances:
    # 1 / rho_mix = sum_i w_i / rho_i.
    specific_volume = np.sum(mass_frac / rho_species)  # [m**3/kg]
    volume_error = half * np.sum(1 / rho_species
                                 + mass_frac / rho_species**2)  # [m**3/kg]
    assert (abs(1 / specific_volume - rho_mix)
            <= rho_mix**2 * volume_error + half)

    # Mass and molar bases agree through M_i = w_i rho_mix / c_i, since
    # [kg/m**3] equals [g/L]. Solvent molar masses use the conventional
    # atomic weights of IUPAC CIAAW, "Standard atomic weights of the
    # elements 2021 (IUPAC Technical Report)", Pure Appl. Chem. 94 (2022)
    # 573-600, doi:10.1515/pac-2019-0603.
    carbon, hydrogen = 12.011, 1.008  # [g/mol]
    nitrogen, oxygen = 14.007, 15.999  # [g/mol]
    molar_mass = {'ACN': 2 * carbon + 3 * hydrogen + nitrogen,  # C2H3N
                  'THF': 4 * carbon + 8 * hydrogen + oxygen}  # C4H8O
    for name, expected in molar_mass.items():  # [g/mol]
        index = SPECIES.index(name)
        implied = mass_frac[index] * rho_mix / mole_conc[index]  # [g/mol]
        relative_error = half * (1 / mass_frac[index] + 1 / mole_conc[index]
                                 + 1 / rho_mix)  # [-]
        assert implied == pytest.approx(expected, rel=relative_error)
