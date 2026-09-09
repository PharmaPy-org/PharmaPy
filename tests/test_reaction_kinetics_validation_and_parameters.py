"""B021 regressions using synthetic reaction networks without a solver backend.

The distinct reactions A <-> B and 2 B <-> 2 C expose species, reaction,
parameter, and time ordering, including the normalized extent of reaction 2.
All fixture values are contract examples, not calibrated physical data.
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics, disect_rxns, gas_ct, get_stoich
from PharmaPy.Errors import PharmaPyValueError

pytestmark = pytest.mark.unit

DATABASE = Path(__file__).parent / "integration/data/pfr_test_pure_comp.json"
SPECIES = ["A", "B", "C", "solv"]
STOICH = np.array([[-1, 1, 0, 0], [0, -2, 2, 0]])  # [-]
K_PARAMS = np.array([2.0, 3.0])  # [1/s], [L/mol/s], distinct synthetic rates
EA_PARAMS = np.array([4000.0, 7000.0])  # [J/mol], distinct thermal responses
KEQ = np.array([2.0, 4.0])  # [-], gives equilibrium B=2A and C=2B
HEATS = np.array([-5000.0, 7000.0])  # [J/mol_rxn], opposite thermal trends
TEMPERATURES = np.array([290.0, 310.0, 340.0])  # [K], three times != two reactions
REFERENCE_TEMP = 298.15  # [K], the public default reference temperature
SCALAR_TEMP = 320.0  # [K], off reference to exercise activation derivatives
EQUILIBRIUM_CONC = np.array([1.0, 2.0, 4.0, 5.0])  # [mol/L], Keq as above
AWAY_CONC = np.array([[1.3, 0.8, 1.7, 5.0],
                      [0.7, 1.6, 0.9, 5.0],
                      [2.1, 0.4, 1.2, 5.0]])  # [mol/L], asymmetric states
FD_STEP = np.cbrt(np.finfo(float).eps)  # [-], central-difference roundoff balance
FD_RTOL = 2e-8  # [-], allowance for cancellation in central differences
FD_ATOL = 1e-10  # [rate/parameter units], roundoff allowance near zero
RATE_RTOL = 1e-14  # [-], roundoff allowance for the direct sqrt(4) rate check


def _kinetics(**overrides):
    """Build the synthetic two-reaction model.

    Parameters
    ----------
    **overrides : dict
        Overrides in RxnKinetics units: concentrations [mol/L], energies
        [J/mol], reaction heat [J/mol_rxn], orders [-], and seconds as time.

    Returns
    -------
    RxnKinetics
        Model with the database species order and raw stoichiometric basis.
    """
    options = dict(path=str(DATABASE), stoich_matrix=STOICH,
                   partic_species=SPECIES, k_params=K_PARAMS,
                   ea_params=EA_PARAMS, keq_params=KEQ,
                   delta_hrxn=HEATS, temp_ref=REFERENCE_TEMP)
    options.update(overrides)
    return RxnKinetics(**options)


@pytest.mark.parametrize("oxygen", ["0.5 O2", "1/2 O2"])
def test_decimal_and_fractional_stoichiometric_coefficients(oxygen):
    reactions, species = disect_rxns([f"2 H2 + {oxygen} --> H2O"])
    assert species == ["H2", "O2", "H2O"]
    np.testing.assert_array_equal(get_stoich(reactions, species),
                                  [[-2.0, -0.5, 1.0]])
    # Reordering columns must retain each named species' coefficient.
    np.testing.assert_array_equal(get_stoich(reactions, species[::-1]),
                                  [[1.0, -0.5, -2.0]])


def test_integer_stoichiometry_preserves_fractional_custom_order():
    kinetics = _kinetics(stoich_matrix=STOICH[:1], k_params=[1.0],
                         ea_params=[0.0], keq_params=None, params_f=[[0.5]])
    np.testing.assert_array_equal(kinetics.params_f, [[0.5, 0.0, 0.0, 0.0]])
    assert kinetics.stoich_matrix.dtype == np.float64
    assert kinetics.params_f.dtype == np.float64
    # Unit rate constant and C_A=4 give sqrt(4)=2 mol/L/s.
    conc = np.array([4.0, 1.0, 1.0, 1.0])  # [mol/L]
    np.testing.assert_allclose(kinetics.get_rxn_rates(conc, REFERENCE_TEMP),
                               [-2.0, 2.0, 0.0, 0.0], rtol=RATE_RTOL, atol=0)


@pytest.mark.parametrize("invalid_row", [[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 0, 0]])
def test_reactantless_row_is_rejected_before_normalization(invalid_row):
    """Reject an invalid dimensionless stoichiometric row by project exception.

    Parameters
    ----------
    invalid_row : list of int
        Product-only or empty stoichiometric coefficients [-].
    """
    with pytest.raises(PharmaPyValueError, match=r"negative.*reactant.*\[1\]"):
        _kinetics(stoich_matrix=[STOICH[0], invalid_row])


def test_all_reactantless_row_indexes_are_reported():
    """Report every invalid dimensionless row in the project exception."""
    with pytest.raises(PharmaPyValueError, match=r"negative.*reactant.*\[1, 2\]"):
        _kinetics(stoich_matrix=[STOICH[0], [0, 1, 0, 0], [0, 0, 0, 0]])


@pytest.mark.parametrize("update", ["constructor", "dictionary"])
def test_reversible_nonstoichiometric_orders_are_rejected(update):
    if update == "constructor":
        with pytest.raises(ValueError, match="thermodynamic consistency"):
            _kinetics(params_f=[[0.5], [2.0]])
    else:
        kinetics = _kinetics(params_f=[[1.0], [2.0]])
        before = kinetics.concat_params().copy()  # [k units], [J/mol], orders [-]
        params = dict(k_params=K_PARAMS * 2, ea_params=EA_PARAMS,
                      params_f=[[0.5], [2.0]])  # [k units], [J/mol], [-]
        with pytest.raises(ValueError, match="thermodynamic consistency"):
            kinetics.set_params(params)
        np.testing.assert_array_equal(kinetics.concat_params(), before)


def test_reversible_explicit_stoichiometric_orders_remain_valid():
    kinetics = _kinetics(params_f=[[1.0], [2.0]], delta_hrxn=[0.0, 0.0])
    np.testing.assert_array_equal(kinetics.params_f,
                                  [[1.0, 0.0, 0.0, 0.0],
                                   [0.0, 2.0, 0.0, 0.0]])
    np.testing.assert_allclose(kinetics.get_rxn_rates(
        EQUILIBRIUM_CONC, REFERENCE_TEMP, delta_hrxn=[0.0, 0.0]),
        np.zeros(len(SPECIES)), atol=FD_ATOL)


@pytest.mark.parametrize("direct", [False, True])
def test_stored_reaction_heat_is_used_when_override_is_none(direct):
    kinetics = _kinetics()
    if direct:
        stored = kinetics.equilibrium_model(
            AWAY_CONC, TEMPERATURES, None)  # [mol/L], [(mol/L)**2]
        explicit = kinetics.equilibrium_model(
            AWAY_CONC, TEMPERATURES, HEATS)  # [mol/L], [(mol/L)**2]
    else:
        # Omitted temperature and heat exercise the originally failing call.
        stored = kinetics.get_rxn_rates(AWAY_CONC[0])  # [mol/L/s]
        explicit = kinetics.get_rxn_rates(
            AWAY_CONC[0], delta_hrxn=HEATS)  # [mol/L/s]
    np.testing.assert_array_equal(stored, explicit)


def test_runtime_reaction_heat_overrides_stored_heat():
    kinetics = _kinetics()
    zero_heat = np.zeros(2)  # [J/mol_rxn], temperature-independent Keq
    actual = kinetics.get_rxn_rates(AWAY_CONC, TEMPERATURES,
                                    overall_rates=False, delta_hrxn=zero_heat)  # [mol/L/s]
    # Independent concentration polynomials for A -> B and 2 B -> 2 C.
    expected_terms = np.column_stack((AWAY_CONC[:, 0] - AWAY_CONC[:, 1] / 2,
                                     AWAY_CONC[:, 1]**2 - AWAY_CONC[:, 2]**2 / 4))  # [mol/L], [(mol/L)**2]
    np.testing.assert_allclose(actual, kinetics.temp_term(TEMPERATURES)
                               * expected_terms, rtol=FD_RTOL, atol=FD_ATOL)
    stored = kinetics.get_rxn_rates(
        AWAY_CONC, TEMPERATURES, overall_rates=False)  # [mol/L/s]
    assert not np.allclose(actual, stored)


def _parameter_central_difference(kinetics, conc, temp, delta_hrxn):
    """Differentiate public rates in concat_params order.

    Parameters
    ----------
    kinetics : RxnKinetics
        Model; its parameters are restored after evaluation.
    conc : ndarray
        Species concentrations [mol/L], shape (species,) or (time, species).
    temp : float or ndarray
        Temperature [K], scalar or matching time vector.
    delta_hrxn : ndarray
        Explicit heat override [J/mol_rxn], on the raw reaction basis.

    Returns
    -------
    ndarray
        Rate/parameter derivatives, shape (species, parameters) or
        (time, species, parameters), with the parameter units of concat_params.
    """
    params = kinetics.concat_params().copy()  # [k units, J/mol] or phi [-]
    columns = []
    try:
        for index, value in enumerate(params):
            step = FD_STEP * max(1.0, abs(value))  # [parameter units]
            plus = params.copy()  # [parameter units]
            minus = params.copy()  # [parameter units]
            plus[index] += step
            minus[index] -= step
            kinetics.set_params(plus)
            rate_plus = kinetics.get_rxn_rates(
                conc, temp, delta_hrxn=delta_hrxn)  # [mol/L/s]
            kinetics.set_params(minus)
            rate_minus = kinetics.get_rxn_rates(
                conc, temp, delta_hrxn=delta_hrxn)  # [mol/L/s]
            columns.append((rate_plus - rate_minus) / (2 * step))
    finally:
        kinetics.set_params(params)
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("reformulate", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("at_equilibrium", [False, True])
def test_reversible_parameter_jacobian_matches_public_rates(
        reformulate, batch, at_equilibrium):
    kinetics = _kinetics(reformulate_kin=reformulate)
    temp = TEMPERATURES if batch else SCALAR_TEMP  # [K]
    if at_equilibrium:
        conc = np.tile(EQUILIBRIUM_CONC, (len(TEMPERATURES), 1))  # [mol/L]
        heat = np.zeros(2)  # [J/mol_rxn], keeps the chosen equilibrium at all T
    else:
        conc = AWAY_CONC.copy()  # [mol/L]
        heat = -HEATS  # [J/mol_rxn], runtime override distinct from stored heat
    if not batch:
        conc = conc[0]  # [mol/L]
    # Explicit heat isolates the sensitivity defect from the missing fallback.
    analytical = kinetics.derivatives(
        conc, temp, dstates=False, delta_hrxn=heat)  # [rate/parameter units]
    expected = _parameter_central_difference(
        kinetics, conc, temp, heat)  # [rate/parameter units]
    assert analytical.shape == ((3, 4, 4) if batch else (4, 4))
    np.testing.assert_allclose(analytical, expected, rtol=FD_RTOL, atol=FD_ATOL)
    if at_equilibrium:
        np.testing.assert_allclose(analytical, 0.0, atol=FD_ATOL)


@pytest.mark.parametrize("reformulate", [False, True])
@pytest.mark.parametrize("custom_orders", [False, True])
def test_scalar_parameter_path_matches_pre_fix_expression_bit_for_bit(
        reformulate, custom_orders):
    kinetics = _kinetics(stoich_matrix=STOICH.astype(float), keq_params=None,
                         reformulate_kin=reformulate,
                         params_f=[[0.5], [1.5]] if custom_orders else None)
    temp_term = kinetics.temp_term(SCALAR_TEMP)  # [k units]
    inv_temp = 1 / kinetics.temp_ref - 1 / SCALAR_TEMP  # [1/K]
    # Literal pre-fix scalar expression, including its operation order.
    if reformulate:
        first = np.diag(temp_term)  # [k units]
        second = temp_term * inv_temp * np.exp(kinetics.phi_2)  # [k units]
    else:
        first = np.diag(np.exp(kinetics.phi_2 / gas_ct * inv_temp))  # [-]
        second = temp_term / gas_ct * inv_temp  # [k units mol/J]
    expected_dk = np.atleast_2d(
        np.hstack((first, np.diag(second))))  # [k units/parameter units]
    np.testing.assert_array_equal(kinetics.dk_dkparams(SCALAR_TEMP), expected_dk)
    forward = kinetics.kinetic_model(
        AWAY_CONC[0], kinetics.params_f)  # [(mol/L)**forward order]
    reaction_derivative = (expected_dk.T * forward).T  # [rate/parameter units]
    if custom_orders:
        order_derivative = (
            kinetics.df_dthetaf(AWAY_CONC[0]).T * temp_term).T  # [mol/L/s]
        reaction_derivative = np.hstack((reaction_derivative, order_derivative))
    expected = np.dot(
        kinetics.normalized_stoich, reaction_derivative)  # [rate/parameter units]
    actual = kinetics.derivatives(
        AWAY_CONC[0], SCALAR_TEMP, dstates=False)  # [rate/parameter units]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reformulate", [False, True])
def test_batched_irreversible_custom_order_jacobian(reformulate):
    kinetics = _kinetics(stoich_matrix=STOICH.astype(float), keq_params=None,
                         params_f=[[0.5], [1.5]], reformulate_kin=reformulate)
    actual = kinetics.derivatives(
        AWAY_CONC, TEMPERATURES, dstates=False,
        delta_hrxn=HEATS)  # [rate/parameter units]
    expected = _parameter_central_difference(kinetics, AWAY_CONC,
                                             TEMPERATURES, HEATS)  # [rate/parameter units]
    assert actual.shape == (3, 4, 6)
    np.testing.assert_allclose(actual, expected, rtol=FD_RTOL, atol=FD_ATOL)
    dk_batch = kinetics.dk_dkparams(TEMPERATURES)  # [k units/parameter units]
    assert dk_batch.shape == (3, 2, 4)
    for index, temp in enumerate(TEMPERATURES):
        np.testing.assert_array_equal(dk_batch[index], kinetics.dk_dkparams(temp))


@pytest.mark.parametrize("coefficient", ["0.5", "1/2"])
def test_constructor_parses_decimal_and_fractional_reaction(coefficient):
    # Parser contract example; coefficients deliberately exercise both formats.
    kinetics = _kinetics(rxn_list=[f"2 A + {coefficient} B --> C"],
                         k_params=[1.0], ea_params=[0.0], keq_params=None)
    assert kinetics.partic_species == ["A", "B", "C"]
    np.testing.assert_array_equal(kinetics.stoich_matrix, [[-2.0, -0.5, 1.0]])
    np.testing.assert_array_equal(kinetics.normalized_stoich,
                                  [[-1.0], [-0.25], [0.5]])


def test_species_database_order_is_preserved_with_integer_stoichiometry():
    permuted = _kinetics(stoich_matrix=STOICH[:, ::-1],
                         partic_species=SPECIES[::-1])
    assert permuted.partic_species == SPECIES
    assert permuted.stoich_matrix.dtype == np.float64
    np.testing.assert_array_equal(permuted.stoich_matrix, STOICH)
    np.testing.assert_array_equal(permuted.normalized_stoich,
                                  [[-1.0, 0.0], [1.0, -1.0],
                                   [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(
        permuted.get_rxn_rates(AWAY_CONC, TEMPERATURES, delta_hrxn=HEATS),
        _kinetics().get_rxn_rates(AWAY_CONC, TEMPERATURES, delta_hrxn=HEATS))


@pytest.mark.parametrize("reformulate", [False, True])
def test_parameter_jacobian_uses_stored_heat_for_a_common_scalar_temperature(
        reformulate):
    kinetics = _kinetics(reformulate_kin=reformulate)
    actual = kinetics.derivatives(
        AWAY_CONC, SCALAR_TEMP, dstates=False)  # [rate/parameter units]
    expected = _parameter_central_difference(kinetics, AWAY_CONC,
                                             SCALAR_TEMP, HEATS)  # [rate/parameter units]
    assert actual.shape == (3, 4, 4)
    np.testing.assert_allclose(actual, expected, rtol=FD_RTOL, atol=FD_ATOL)


@pytest.mark.parametrize("orders", [
    [[0.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
    [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
])
def test_reversible_kinetics_rejects_custom_model_kind(orders):
    # Rejection must depend on model kind even when its orders match exactly.
    forward_model = _kinetics(keq_params=None).elem_f_model
    with pytest.raises(ValueError, match=(
            "Reversible.*built-in elementary.*not supported.*custom kinetic model")):
        _kinetics(kinetic_model=forward_model, params_f=orders)


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("order", [0.1 + 0.2, 0.5])
def test_reversible_fractional_orders_allow_only_roundoff(order, update):
    # 0.3 A <-> 0.3 B: 0.1 + 0.2 represents the same order up to roundoff.
    options = dict(stoich_matrix=[[-0.3, 0.3, 0.0, 0.0]],
                   k_params=K_PARAMS[:1], ea_params=EA_PARAMS[:1],
                   keq_params=KEQ[:1])  # coefficients [-], k [(mol/L)**0.7/s], Ea [J/mol], Keq [-]
    if update:
        kinetics = _kinetics(**options)
        params = dict(k_params=K_PARAMS[:1], ea_params=EA_PARAMS[:1],
                      params_f=[[order]])  # k [(mol/L)**0.7/s], Ea [J/mol], order [-]
        if order == 0.5:
            with pytest.raises(ValueError, match="thermodynamic consistency"):
                kinetics.set_params(params)
        else:
            kinetics.set_params(params)
    else:
        if order == 0.5:
            with pytest.raises(ValueError, match="thermodynamic consistency"):
                _kinetics(**options, params_f=[[order]])
            return
        kinetics = _kinetics(**options, params_f=[[order]])
    np.testing.assert_array_equal(kinetics.params_f, [[0.3, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize("reformulate", [False, True])
def test_reversible_explicit_orders_are_excluded_from_fitted_parameters(
        reformulate):
    kinetics = _kinetics(params_f=[[1.0], [2.0]], reformulate_kin=reformulate)
    implicit = _kinetics(reformulate_kin=reformulate)
    assert kinetics.fit_paramsf is False
    assert kinetics.num_params == 4  # two k parameters and two Ea parameters
    assert kinetics.name_params == implicit.name_params
    params = kinetics.concat_params()  # [k units, J/mol] or phi [-]
    assert params.shape == (4,)
    np.testing.assert_array_equal(params, implicit.concat_params())
    params[0] += 1.0  # [1/s] or [-], synthetic first-rate parameter change
    kinetics.set_params(params)
    implicit.set_params(params)
    np.testing.assert_array_equal(kinetics.concat_params(), params)
    np.testing.assert_array_equal(kinetics.params_f, implicit.params_f)
    # Dictionary updates with explicit orders must also keep them fixed.
    kinetics.set_params(dict(k_params=K_PARAMS, ea_params=EA_PARAMS,
                             params_f=[[1.0], [2.0]]))
    assert kinetics.fit_paramsf is False
    actual = kinetics.derivatives(
        AWAY_CONC, TEMPERATURES, dstates=False, delta_hrxn=HEATS)  # [rate/parameter units]
    expected = _parameter_central_difference(
        kinetics, AWAY_CONC, TEMPERATURES, HEATS)  # [rate/parameter units]
    assert actual.shape == (3, 4, 4)
    np.testing.assert_allclose(actual, expected, rtol=FD_RTOL, atol=FD_ATOL)


@pytest.mark.parametrize("batch", [False, True])
def test_none_reaction_heat_uses_documented_zero_default(batch):
    kinetics = _kinetics(delta_hrxn=None)
    zero_heat = _kinetics(delta_hrxn=0)
    conc = AWAY_CONC if batch else AWAY_CONC[0]  # [mol/L]
    temp = TEMPERATURES if batch else SCALAR_TEMP  # [K]
    actual = kinetics.get_rxn_rates(conc, temp)  # [mol/L/s]
    np.testing.assert_array_equal(kinetics.delta_hrxn, [0])
    np.testing.assert_array_equal(actual, zero_heat.get_rxn_rates(conc, temp))


@pytest.mark.parametrize("reformulate", [False, True])
@pytest.mark.parametrize("reversible", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_batched_sensitivities_preserve_shared_arrhenius_parameters(
        reformulate, reversible, shared):
    """Preserve the parameter basis for two first-order reactions.

    Parameters
    ----------
    reformulate : bool
        Use logarithmic Arrhenius parameters [-] instead of k [1/s] and
        activation energy [J/mol].
    reversible : bool
        Include reverse rates using the dimensionless fixture constants.
    shared : bool
        Use one Arrhenius pair shared by both first-order reactions, rather
        than independent parameters for each reaction.
    """
    # A -> B -> C gives both reactions the same first-order rate-constant units.
    stoich = np.array([[-1, 1, 0, 0], [0, -1, 1, 0]])  # [-]
    rate_constants = K_PARAMS[:1] if shared else K_PARAMS  # [1/s]
    activation_energies = EA_PARAMS[:1] if shared else EA_PARAMS  # [J/mol]
    kinetics = _kinetics(
        stoich_matrix=stoich, k_params=rate_constants,
        ea_params=activation_energies, keq_params=KEQ if reversible else None,
        reformulate_kin=reformulate)
    actual = kinetics.derivatives(
        AWAY_CONC, TEMPERATURES, dstates=False)  # [rate/parameter units]
    expected = _parameter_central_difference(
        kinetics, AWAY_CONC, TEMPERATURES, HEATS)  # [rate/parameter units]
    assert actual.shape == (len(TEMPERATURES), len(SPECIES),
                            len(kinetics.concat_params()))
    np.testing.assert_allclose(actual, expected, rtol=FD_RTOL, atol=FD_ATOL)
    batched_constants = kinetics.dk_dkparams(
        TEMPERATURES)  # [rate-constant/parameter units]
    scalar_constants = np.stack([
        kinetics.dk_dkparams(temp) for temp in TEMPERATURES
    ])  # [rate-constant/parameter units]
    np.testing.assert_array_equal(batched_constants, scalar_constants)
