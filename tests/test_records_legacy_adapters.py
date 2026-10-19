"""Field records and their legacy array/dictionary adapters (#351).

Scope: ``PharmaPy.Records.FieldIdentity``/``FieldRecord`` validation and
JSON round trips, and the documented adapters ``fields_from_array``,
``fields_to_array``, ``fields_from_mapping`` and ``fields_to_mapping`` that
convert legacy state arrays (rows = samples, columns = fields) and
``{name: array}`` dictionaries without changing values, ordering or dtype.

Fixtures are small asymmetric arrays (five samples, three fields, a
replicate time and NaN entries at distinct positions) whose expected
columns are written out by hand. Core lane; no solver is used.
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
