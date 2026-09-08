"""B006 signed driving forces and parameter partials, without solver backends.

Synthetic power laws exercise both sides of a constant 2 kg/m**3 solubility.
A monodisperse population gives independently specified non-unit moment factors.
The physical prefactors below are numerical test cases, not calibrated kinetics.
"""

import warnings

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics, cryst_mechanism
from PharmaPy.Phases import LiquidPhase, SolidPhase

pytestmark = pytest.mark.unit

SATURATION = 2.0  # [kg/m**3], distinguishes absolute and relative forces
TEMPERATURE = 320.0  # [K], differs from Tref to exercise the phi_2 partial
REFERENCE_TEMPERATURE = 298.15  # [K], conventional 25 degrees Celsius
GAS_CONSTANT = 8.314  # [J/mol/K], established numerical value in Kinetics.py
KV = 0.5  # [-], synthetic crystal volume shape factor
NUMBER_DENSITY = 10.0  # [#/m**3], synthetic monodisperse population
CRYSTAL_LENGTH = 0.2  # [m], chosen to make both area and volume factors non-unit
MOMENTS = NUMBER_DENSITY * CRYSTAL_LENGTH ** np.arange(4)  # [m**j/m**3]
# Each tuple is k [rate/force**n/(kv*moment)**s_2], E [J/mol], n [-],
# and optional s_2 [-]. Unequal values distinguish mechanism/column ordering.
# E/(R*T) = 1, 2, 3, 4 are synthetic, moderate Arrhenius exponents that make
# cross-mechanism activation-energy substitutions visible without underflow.
PARAMETERS = {
    'nucl_prim': [2.3, GAS_CONSTANT * TEMPERATURE, 1.2],
    'nucl_sec': [3.7, 2 * GAS_CONSTANT * TEMPERATURE, 1.4, 0.7],
    'growth': [1.7, 3 * GAS_CONSTANT * TEMPERATURE, 1.6],
    'dissolution': [4.1, 4 * GAS_CONSTANT * TEMPERATURE, 1.8],
}
# Central differences balance O(h**2) truncation and O(eps/h) cancellation at
# h ~ eps**(1/3). Scale by each parameter magnitude, with one parameter unit
# as the zero/small-parameter scale. The tolerance allows accumulated roundoff
# in exp/power evaluation; zero absolute tolerance cannot hide a missing column.
FD_REL_STEP = np.cbrt(np.finfo(float).eps)  # [-]
FD_RTOL = 2e-8  # [-], comfortably above eps**(2/3) for these smooth test states


def make_kinetics(sup_sat_type, reformulate=False, moment_basis='volume'):
    """Build synthetic kinetics with distinct mechanism parameters.

    Parameters
    ----------
    sup_sat_type : str
        Driving-force option.
    reformulate : bool, optional
        Use transformed parameters, default False.
    moment_basis : str, optional
        Secondary moment selection, default 'volume'.

    Returns
    -------
    CrystKinetics
        Kinetics on the mass-concentration basis [kg/m**3], with rates
        [#/m**3/s] for nucleation and [um/s] for growth/dissolution.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=(r"^sup_sat_type='ratio' is deprecated: it now means S - 1 "
                     r"\(identical to 'relative'\)\. Prefactors fitted to the old S "
                     r"law must be refitted\.$"),
            category=DeprecationWarning)
        kinetics = CrystKinetics(
            coeff_solub=[SATURATION], temp_ref=REFERENCE_TEMPERATURE,
            sup_sat_type=sup_sat_type, reformulate_kin=reformulate,
            mu_sec_nucl=moment_basis, **PARAMETERS)
    kinetics.target_idx = 0
    return kinetics


def central_partial(kinetics, name, column, force):
    """Difference the standalone mechanism on its native parameter basis.

    Parameters
    ----------
    kinetics : CrystKinetics
        Mechanism parameter provider.
    name : str
        Mechanism name.
    column : int
        Native parameter index, including s_2 when requested.
    force : float
        Independently specified signed force [-] or [kg/m**3].

    Returns
    -------
    float
        Rate partial [rate/parameter] on the native parameter basis.
    """
    params = np.array(kinetics.params[name])  # [parameter units], see PARAMETERS
    step = FD_REL_STEP * max(abs(params[column]), 1.0)  # [parameter units]
    plus = params.copy()  # [parameter units]
    minus = params.copy()  # [parameter units]
    plus[column] += step
    minus[column] -= step
    upper = cryst_mechanism(  # [rate]
        force, MOMENTS, TEMPERATURE, kinetics.temp_ref, plus,
        kinetics.reformulate_kin, KV, kinetics.mu_sec_nucl)
    lower = cryst_mechanism(  # [rate]
        force, MOMENTS, TEMPERATURE, kinetics.temp_ref, minus,
        kinetics.reformulate_kin, KV, kinetics.mu_sec_nucl)
    return (upper - lower) / (2 * step)


@pytest.mark.parametrize('option', ['relative', 'ratio', 'absolute'])
@pytest.mark.parametrize('vector', [False, True])
def test_signed_rates_and_saturation(option, vector):
    """Check signed rates, exact saturation zeros, and equivalent force modes.

    Parameters
    ----------
    option : str
        Supersaturation driving-force convention.
    vector : bool
        Whether to evaluate all concentrations together or one at a time.
    """
    kinetics = make_kinetics(option)
    concentrations = np.array([3.5, 0.5, SATURATION])  # [kg/m**3], above/below/at
    if vector:
        rates = np.array(kinetics.get_kinetics(  # [nucleation, growth, dissolution]
            concentrations[:, None], np.full(3, TEMPERATURE), KV,
            np.tile(MOMENTS, (3, 1)), nucl_sec_out=True))
    else:
        rates = np.array([kinetics.get_kinetics(  # [nucleation, growth, dissolution]
            conc, TEMPERATURE, KV, MOMENTS, nucl_sec_out=True)
            for conc in concentrations]).T
    assert np.all(rates[:3, 0] > 0)
    assert rates[3, 1] < 0
    np.testing.assert_array_equal(rates[3, [0, 2]], 0)
    np.testing.assert_array_equal(rates[:3, 1:], 0)
    if option == 'ratio':
        relative = make_kinetics('relative')
        expected = np.array([relative.get_kinetics(  # same rate units
            conc, TEMPERATURE, KV, MOMENTS, nucl_sec_out=True)
            for conc in concentrations]).T
        # Allow roundoff from scalar/vector evaluation of these short products.
        roundoff_rtol = 1e-12  # [-], relative allowance above machine epsilon
        roundoff_atol = 1e-15  # [#/m**3/s] nucleation; [um/s] growth/dissolution
        np.testing.assert_allclose(
            rates, expected, rtol=roundoff_rtol, atol=roundoff_atol)


@pytest.mark.parametrize('option', ['relative', 'absolute'])
@pytest.mark.parametrize('reformulate', [False, True])
@pytest.mark.parametrize('concentration', [3.5, 0.5])  # [kg/m**3], signed cases
def test_existing_rate_expressions_are_bit_identical(option, reformulate, concentration):
    kinetics = make_kinetics(option, reformulate)
    actual = kinetics.get_kinetics(  # [#/m**3/s], [#/m**3/s], [um/s], [um/s]
        concentration, TEMPERATURE, KV, MOMENTS, nucl_sec_out=True)
    old_force = concentration - SATURATION  # [kg/m**3]
    if option == 'relative':
        old_force = old_force / SATURATION  # [-], pre-fix relative expression
    expected = []
    for name, params in kinetics.params.items():
        if (old_force >= 0) == (name == 'dissolution'):
            expected.append(0.0)
            continue
        first, second, exponent = params[:3]  # [native k, E, n] or [phi_1, phi_2, n]
        if reformulate:
            # The reference takes phi_1/phi_2 from the untouched production
            # transform_params; only the rate expression is independent here.
            pre_exp = np.exp(first + np.exp(second) * (  # [rate/force**n/moment**s_2]
                1 / REFERENCE_TEMPERATURE - 1 / TEMPERATURE))
        else:
            pre_exp = first * np.exp(-second / GAS_CONSTANT / TEMPERATURE)
        magnitude = np.maximum(np.finfo(float).eps, np.abs(old_force))  # [force]
        rate = pre_exp * old_force * magnitude ** (exponent - 1)  # [rate]
        if name == 'nucl_sec':
            rate *= (np.maximum(0, MOMENTS[3]) * KV) ** params[3]
        expected.append(rate)
    # Exact byte comparison is intentional: this is a bit-identity contract,
    # not a tolerance-based scientific comparison.
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_unknown_supersaturation_option_is_rejected():
    with pytest.raises(
            ValueError, match="sup_sat_type.*relative.*ratio.*absolute.*got 'Relative'"):
        make_kinetics('Relative')


def test_ratio_warns_about_changed_driving_force():
    with pytest.warns(
            DeprecationWarning,
            match="S - 1.*identical to 'relative'.*old S law must be refitted") as caught:
        kinetics = CrystKinetics(coeff_solub=[SATURATION], sup_sat_type='ratio')
    assert kinetics.sup_sat_type == 'ratio'
    assert caught[0].filename == __file__


@pytest.mark.parametrize('option', ['relative', 'ratio', 'absolute'])
@pytest.mark.parametrize('reformulate', [False, True])
@pytest.mark.parametrize('name', PARAMETERS)
@pytest.mark.parametrize('column', range(3), ids=['prefactor', 'activation', 'exponent'])
@pytest.mark.parametrize('moment_basis', ['area', 'volume'])
def test_parameter_partials_match_finite_differences(
        option, reformulate, name, column, moment_basis):
    kinetics = make_kinetics(option, reformulate, moment_basis)
    concentration = 0.5 if name == 'dissolution' else 3.5  # [kg/m**3]
    # Independently: c_sat=2, c=0.5 or 3.5 gives excess +/-1.5 and relative +/-0.75.
    force = -1.5 if name == 'dissolution' else 1.5  # [kg/m**3]
    if option != 'absolute':
        force = force / 2  # [-], half the absolute excess for c_sat=2 kg/m**3
    kinetics.get_kinetics(concentration, TEMPERATURE, KV, MOMENTS)
    partials = kinetics.deriv_cryst(  # [rate/parameter] arrays; [kg/m**3] solubility
        concentration, np.array([concentration]), TEMPERATURE)
    mechanism_index = kinetics.names_mechanisms.index(name)
    assert partials[mechanism_index].shape == (3,)
    expected = central_partial(kinetics, name, column, force)  # [rate/parameter]
    assert expected != 0  # All selected columns must exercise a nonzero partial.
    assert partials[mechanism_index][column] == pytest.approx(
        expected, rel=FD_RTOL, abs=0)
    inactive = range(3) if name == 'dissolution' else [3]
    for index in inactive:
        np.testing.assert_array_equal(partials[index], np.zeros(3))


@pytest.mark.parametrize('option', ['relative', 'ratio', 'absolute'])
@pytest.mark.parametrize('name', PARAMETERS)
@pytest.mark.parametrize('moment_basis', ['area', 'volume'])
def test_zero_prefactor_partial_retains_moment_factor(option, name, moment_basis):
    kinetics = make_kinetics(option, moment_basis=moment_basis)
    kinetics.params[name][0] = 0.0  # [prefactor units], exact zero-rate boundary
    concentration = 0.5 if name == 'dissolution' else 3.5  # [kg/m**3]
    force = -1.5 if name == 'dissolution' else 1.5  # [kg/m**3]
    if option != 'absolute':
        force = force / 2  # [-], c_sat=2 kg/m**3
    kinetics.get_kinetics(concentration, TEMPERATURE, KV, MOMENTS)
    partials = kinetics.deriv_cryst(  # [rate/parameter]; [kg/m**3]
        concentration, np.array([concentration]), TEMPERATURE)
    actual = partials[kinetics.names_mechanisms.index(name)]  # [rate/parameter]
    expected = central_partial(kinetics, name, 0, force)  # [rate/prefactor]
    assert expected != 0
    assert actual[0] == pytest.approx(expected, rel=FD_RTOL, abs=0)
    np.testing.assert_array_equal(actual[1:], 0)

    # Put the zero-prefactor mechanism on the inactive side. Without the active
    # gate its reconstructed prefactor partial would incorrectly remain nonzero.
    inactive_concentration = 3.5 if name == 'dissolution' else 0.5  # [kg/m**3]
    kinetics.get_kinetics(inactive_concentration, TEMPERATURE, KV, MOMENTS)
    inactive_partials = kinetics.deriv_cryst(  # [rate/parameter]; [kg/m**3]
        inactive_concentration, np.array([inactive_concentration]), TEMPERATURE)
    inactive_indices = [3] if name == 'dissolution' else range(3)
    for index in inactive_indices:
        np.testing.assert_array_equal(inactive_partials[index], np.zeros(3))


@pytest.mark.parametrize('option', ['relative', 'ratio', 'absolute'])
@pytest.mark.parametrize('reformulate', [False, True])
def test_parameter_partials_vanish_at_saturation(option, reformulate):
    kinetics = make_kinetics(option, reformulate)
    kinetics.get_kinetics(SATURATION, TEMPERATURE, KV, MOMENTS)
    partials = kinetics.deriv_cryst(  # [rate/parameter]; [kg/m**3]
        SATURATION, np.array([SATURATION]), TEMPERATURE)
    np.testing.assert_array_equal(partials[:4], np.zeros((4, 3)))


def test_omitted_secondary_without_moments_has_zero_partials():
    kinetics = CrystKinetics(coeff_solub=[SATURATION], growth=PARAMETERS['growth'])
    kinetics.get_kinetics(3.0, TEMPERATURE, KV)  # [kg/m**3], [K], [-]
    partials = kinetics.deriv_cryst(3.0, np.array([3.0]), TEMPERATURE)
    np.testing.assert_array_equal(partials[1], np.zeros(3))


@pytest.mark.parametrize('option', ['relative', 'ratio', 'absolute'])
@pytest.mark.parametrize('reformulate', [False, True])
def test_crystallizer_parameter_column_handoff(data_path, option, reformulate):
    """Exercise the real jac_params consumer, including its appended s_2.

    The fixture has one cubic metre of slurry, with the solid volume deducted
    from the liquid inventory, so cached normalized moments match the state.
    Rates are cached through unit_model with solver-supplied parameters;
    the growth-prefactor column is also checked through that public handoff.

    Parameters
    ----------
    data_path : dict
        Shipped test database paths.
    option : str
        Driving-force option.
    reformulate : bool
        Native parameterization choice.
    """
    kinetics = make_kinetics(option, reformulate)
    unit = BatchCryst(target_comp='A', method='moments')
    path = str(data_path['flowsheet'] / 'compound_database.json')
    composition = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure synthetic A
    volume = 1.0  # [m**3], slurry volume makes cached moments equal total moments
    liquid_volume = volume * (1 - KV * MOMENTS[3])  # [m**3], subtract crystal volume
    unit.Liquid_1 = LiquidPhase(path, mass_frac=composition, temp=TEMPERATURE,
                                vol=liquid_volume)
    unit.Solid_1 = SolidPhase(path, mass_frac=composition, moments=MOMENTS, kv=KV)
    unit.Phases = (unit.Liquid_1, unit.Solid_1)
    # Assign the actual kinetics; explicit state layout avoids solver setup.
    unit._Kinetics = kinetics
    unit.num_distr = len(MOMENTS)
    unit.num_species = len(composition)
    unit.target_ind = 0
    unit.kron_jtg = np.array(composition)  # [-], target selector
    unit.dim_states = [unit.num_distr, unit.num_species, 1]
    unit.name_states = ['mu_n', 'mass_conc', 'vol']
    unit.controls = {'temp': {'fun': lambda time: TEMPERATURE, 'args': (), 'kwargs': {}}}
    unit.mask_params = np.ones(kinetics.num_params, dtype=bool)
    length_conversion = 1e6  # [um/m], exact length conversion
    total_moments = MOMENTS * volume * length_conversion ** np.arange(4)  # [um**j]
    concentrations = np.array([3.5, 0.0, 0.0, 0.0, 0.0])  # [kg/m**3]
    states = np.concatenate((total_moments, concentrations, [liquid_volume]))  # mixed units above
    parameters = kinetics.concat_params()  # [native kinetic parameter units]
    unit.unit_model(0.0, states, params=parameters)
    jacobian = unit.jac_params(0.0, states, parameters)  # [state/s/parameter]
    force = 1.5 if option == 'absolute' else 0.75  # [kg/m**3] or [-]
    expected_nucleation = np.array([  # [#/m**3/s/parameter]
        central_partial(kinetics, name, column, force)
        for name, columns in [('nucl_prim', range(3)), ('nucl_sec', range(4))]
        for column in columns])
    np.testing.assert_allclose(jacobian[0, :7], volume * expected_nucleation,
                               rtol=FD_RTOL, atol=0)
    expected_growth = np.array([  # [um/s/parameter]
        central_partial(kinetics, 'growth', column, force) for column in range(3)])
    np.testing.assert_allclose(jacobian[1, 7:10], total_moments[0] * expected_growth,
                               rtol=FD_RTOL, atol=0)
    assert jacobian.shape == (len(states), 13)

    growth_column = 7  # growth prefactor follows three primary/four secondary columns
    step = FD_REL_STEP * max(abs(parameters[growth_column]), 1.0)  # [parameter unit]
    plus, minus = parameters.copy(), parameters.copy()  # [native parameter units]
    plus[growth_column] += step
    minus[growth_column] -= step
    upper = unit.unit_model(0.0, states, params=plus)  # [state unit/s]
    lower = unit.unit_model(0.0, states, params=minus)  # [state unit/s]
    rhs_partial = (upper[1] - lower[1]) / (2 * step)  # [um/s/parameter unit]
    assert jacobian[1, growth_column] == pytest.approx(rhs_partial, rel=FD_RTOL, abs=0)
