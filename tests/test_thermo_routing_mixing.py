"""shared-thermo regressions using real phases and synthetic JSON data.

No Assimulo is imported: evaporator residuals use real constructors and phases,
with phase attributes assigned directly to isolate initialization issue #258.
These tests specify thermodynamic residuals, not a provisional solver contract.
The synthetic constants are contract fixtures, not measured property data.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/27
https://github.com/PharmaPy-org/PharmaPy/issues/62
https://github.com/PharmaPy-org/PharmaPy/issues/89
https://github.com/PharmaPy-org/PharmaPy/issues/122
"""

from copy import deepcopy
import json
import warnings
from PharmaPy.Distillation import DistillationColumn, DynamicDistillation
import math
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Evaporators import (
    AdiabaticFlash, ContinuousEvaporator, Evaporator, IsothermalFlash,
)
from PharmaPy.Extractors import VALID_GAMMA_METHODS, validate_gamma_method
from PharmaPy.Phases import LiquidPhase, VaporPhase
from PharmaPy.ThermoModule import VALID_ACTIVITY_MODELS, validate_activity_model
from conftest import THERMO_TWO_SPECIES

pytestmark = pytest.mark.unit

# Roundoff tolerance for short algebraic expressions; no empirical fit error.
RTOL = 1e-12  # [-]
# Small pressure closure roundoff compared with the fixture's kPa pressures.
PRESSURE_ATOL = 1e-10  # [Pa]
TEMPERATURE = 350.0  # [K], the synthetic UNIQUAC reference temperature
PRESSURE = 1e5  # [Pa], subcritical fixture pressure
COMPOSITION = np.array([0.5, 0.5])  # [-], equimolar binary for analytic gamma
# Match UNIQUAC's R; set unlike tau = exp(-a/RT) = 1/2 at TEMPERATURE.
INTERACTION_ENERGY = 8.314 * TEMPERATURE * math.log(2)  # [J/mol]


@pytest.fixture
def thermo_path(tmp_path):
    """Write the binary contract database with an analytic UNIQUAC limit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated test directory.

    Returns
    -------
    str
        JSON path. Identical unit UNIQUAC sizes [-] remove the combinatorial
        contribution; symmetric unlike tau=1/2 gives gamma=4/3 for an equimolar
        liquid at 350 K. Antoine and density data retain the conftest units.
    """
    database = deepcopy(THERMO_TWO_SPECIES)
    for species in database.values():
        species.update(ri=1.0, qi=1.0, qip=1.0)  # [-], identical model sizes
        species['henry_constant'] = 2e6  # [Pa], distinct from Antoine pressures
    database['heavy']['t_crit'] = 400.0  # [K], crosses only the hot test rows
    database['interaction'] = {
        'amk': [[0, INTERACTION_ENERGY], [INTERACTION_ENERGY, 0]],  # [J/mol]
    }
    path = tmp_path / 'binary.json'
    path.write_text(json.dumps(database))
    return str(path)


@pytest.fixture
def liquid(thermo_path):
    """Create a real liquid phase for the binary thermodynamic contract.

    Parameters
    ----------
    thermo_path : str
        Synthetic JSON path.

    Returns
    -------
    LiquidPhase
        One mole [mol] of equimolar liquid [-] at 350 K and 1e5 Pa.
    """
    return LiquidPhase(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                       mole_frac=COMPOSITION, moles=1.0)


@pytest.mark.parametrize('temperatures', [
    [350.0, 450.0, 400.0], [450.0, 350.0], [350.0], [450.0],
])
def test_henry_mask_matches_scalar_rows(liquid, temperatures):
    """Check that Henry substitutions follow each temperature/species pair.

    Parameters
    ----------
    liquid : LiquidPhase
        Binary fixture with critical temperatures 650 K and 400 K.
    temperatures : list of float
        Temperatures [K], including reordered profiles and one-row arrays.
    """
    # [K]: A remains subcritical; B crosses 400 K, with equality subcritical.
    temperatures = np.array(temperatures)  # [K]
    expected = np.stack([
        liquid.getKeqVLE(temp=float(temp)) for temp in temperatures
    ])  # [-]
    actual = liquid.getKeqVLE(temp=temperatures)  # [-]
    assert actual.shape == (len(temperatures), 2)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)
    for row, temp in enumerate(temperatures):
        if temp > liquid.t_crit[1]:
            assert actual[row, 1] == pytest.approx(
                liquid.henry_constant[1] / PRESSURE, rel=RTOL)
        else:
            assert actual[row, 1] == pytest.approx(
                10 ** (8.0 - 1800.0 / (temp - 40.0)) / PRESSURE, rel=RTOL)


def _wilke_reference(mole_frac, molar_masses, viscosities):
    """Evaluate Wilke with scalar species-pair sums, independent of broadcasting.

    Parameters
    ----------
    mole_frac : ndarray
        One composition [-], shape (num_species,).
    molar_masses : ndarray
        Pure molar masses [g/mol], shape (num_species,).
    viscosities : ndarray
        Pure dynamic viscosities [Pa*s], shape (num_species,).

    Returns
    -------
    float
        Mixture viscosity [Pa*s]. Poling et al., 5th ed., eqs. 9-5.13/9-5.14
        fix the square, square root, fourth root, and factor eight below.
    """
    mixture = 0.0  # [Pa*s]
    for i in range(len(mole_frac)):
        denominator = 0.0  # [-]
        for j in range(len(mole_frac)):
            factor = (
                (1 + math.sqrt(viscosities[i] / viscosities[j])
                 * math.sqrt(math.sqrt(molar_masses[j] / molar_masses[i]))) ** 2
                / math.sqrt(8 * (1 + molar_masses[i] / molar_masses[j]))
            )  # [-]
            denominator += mole_frac[j] * factor
        mixture += mole_frac[i] * viscosities[i] / denominator
    return mixture


@pytest.mark.parametrize('equal_properties', [False, True])
def test_real_vapor_wilke_scalar_and_profile(tmp_path, equal_properties):
    """Check viscosity values, composition basis, shapes, and pure limits.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated JSON directory.
    equal_properties : bool
        Whether all molar masses [g/mol] and viscosities [Pa*s] are identical.
    """
    # Synthetic distinct masses and viscosities expose cross-species inversions.
    masses = np.array([18.0, 44.0, 100.0])  # [g/mol]
    viscosities = np.array([1e-5, 2.5e-5, 4e-5])  # [Pa*s], fixed at 350 K
    if equal_properties:
        masses[:] = masses[0]
        viscosities[:] = viscosities[0]
    database = {
        name: {'mw': mass, 'visc_gas': viscosity,  # [g/mol], [Pa*s]
               'rho_liq': 1000.0}  # [kg/m**3], required by phase construction
        for name, mass, viscosity in zip('ABC', masses, viscosities)
    }
    path = tmp_path / 'vapor.json'
    path.write_text(json.dumps(database))
    compositions = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])  # [-]
    vapor = VaporPhase(str(path), mole_frac=compositions[0],
                       temp=TEMPERATURE, pres=PRESSURE, moles=1.0)
    expected = np.array([
        _wilke_reference(row, masses, viscosities) for row in compositions
    ])  # [Pa*s]
    scalar = vapor.getViscosity()  # [Pa*s], real phase-to-manager handoff
    profile = vapor.getViscosity(mole_frac=compositions)  # [Pa*s]
    assert np.isscalar(scalar)
    assert profile.shape == (2,)
    np.testing.assert_allclose(scalar, expected[0], rtol=RTOL, atol=0)
    np.testing.assert_allclose(profile, expected, rtol=RTOL, atol=0)
    # Same molar composition supplied on a mass basis must reach the same rule.
    mass_weights = compositions * masses  # [g/mol], unnormalized mass weights
    mass_fractions = mass_weights / mass_weights.sum(axis=1)[:, None]  # [-]
    np.testing.assert_allclose(vapor.getViscosity(mass_frac=mass_fractions),
                               expected, rtol=RTOL, atol=0)
    if equal_properties:
        np.testing.assert_allclose(profile, viscosities[0], rtol=RTOL, atol=0)
    # The pure-component boundary must survive zero fractions of other species.
    np.testing.assert_allclose(vapor.getViscosity(mole_frac=np.eye(3)),
                               viscosities, rtol=RTOL, atol=0)


def test_vapor_missing_viscosity_is_actionable():
    """Require a property-specific error for the shipped database.

    Notes
    -----
    No vapor viscosity data [Pa*s] are fabricated for shipped species.
    """
    path = Path(__file__).parent / 'Flowsheet/data/compound_database.json'
    # The real shipped five-species database intentionally has no visc_gas.
    vapor = VaporPhase(str(path), mole_frac=[1, 0, 0, 0, 0], moles=1.0)
    with pytest.raises(ValueError, match=r"missing 'visc_gas'.*\[Pa\*s\].*JSON"):
        vapor.getViscosity()


@pytest.mark.parametrize('method', VALID_ACTIVITY_MODELS)
def test_activity_models_route_to_real_model_values(liquid, method):
    """Compare real activity and VLE routing with analytic fixture values.

    Parameters
    ----------
    liquid : LiquidPhase
        Equimolar fixture [-] at the 350 K interaction reference.
    method : str
        One of the three documented model selectors.

    Notes
    -----
    UNIFAC routing is verified by its missing ``a_unifac`` data error. A
    numerical UNIFAC value check still requires sourced group data.
    """
    if method == 'UNIFAC':
        with pytest.raises(AttributeError, match='a_unifac'):
            liquid.getActivityCoeff(method=method)
        with pytest.raises(AttributeError, match='a_unifac'):
            liquid.getKeqVLE(gamma_model=method)
        return
    # At equimolar x with unit r=q=qprime and unlike tau=1/2, each UNIQUAC
    # residual sum is 1 and log(gamma)=-log(3/4), hence gamma=4/3 exactly.
    expected_gamma = np.ones(2) * (1 if method == 'ideal' else 4/3)  # [-]
    actual_gamma = liquid.getActivityCoeff(method=method)  # [-]
    np.testing.assert_allclose(actual_gamma, expected_gamma, rtol=RTOL, atol=0)
    expected_k = expected_gamma * liquid.AntoineEquation(TEMPERATURE) / PRESSURE  # [-]
    np.testing.assert_allclose(liquid.getKeqVLE(gamma_model=method),
                               expected_k, rtol=RTOL, atol=0)


@pytest.mark.parametrize('boundary', ['manager', 'liquid', 'shared', 'extractor'])
def test_invalid_selector_has_shared_error(liquid, boundary):
    """Require the same actionable error contract at each shared boundary.

    Parameters
    ----------
    liquid : LiquidPhase
        Real phase providing thermodynamic entry points.
    boundary : str
        Selector-consuming API exercised by this case.
    """
    assert VALID_GAMMA_METHODS is VALID_ACTIVITY_MODELS
    invalid = 'uniquac'
    if boundary == 'manager':
        call = liquid.getKeqVLE
        kwargs = {'gamma_model': invalid}
    elif boundary == 'liquid':
        call = liquid.getActivityCoeff
        kwargs = {'method': invalid}
    elif boundary == 'shared':
        call = validate_activity_model
        kwargs = {'model': invalid, 'param_name': 'model'}
    else:
        call = validate_gamma_method
        kwargs = {'gamma_method': invalid}
    with pytest.raises(ValueError) as error:
        call(**kwargs)
    parameter = next(iter(kwargs))
    assert str(error.value) == (
        f'{parameter} must be one of {VALID_ACTIVITY_MODELS}, got {invalid!r}')


@pytest.mark.parametrize('unit_class, selector, kwargs', [
    (Evaporator, 'activity_model', {'vol_drum': 1.0}),  # [m**3]
    (ContinuousEvaporator, 'activity_model', {'vol_drum': 1.0}),  # [m**3]
    (IsothermalFlash, 'gamma_method', {}),
    *[(cls, 'gamma_model', dict(pres=PRESSURE, q_feed=1., LK='light', HK='heavy',
                              perc_LK=95., perc_HK=95.))
      for cls in (DistillationColumn, DynamicDistillation)],
    (AdiabaticFlash, 'gamma_method', {'pres_drum': PRESSURE}),  # [Pa]
])
@pytest.mark.parametrize('model', [*VALID_ACTIVITY_MODELS, 'uniquac'])
def test_unit_constructor_selector_validation(unit_class, selector, kwargs, model):
    """Check accepted selectors and immediate rejection during construction.

    Parameters
    ----------
    unit_class : type
        Flash or evaporator class.
    selector : str
        Established public parameter name.
    kwargs : dict
        Required constructor arguments: drum volume [m**3] or pressure [Pa].
    model : str
        Valid selector or deliberately invalid lowercase spelling.
    """
    if model in VALID_ACTIVITY_MODELS:
        unit = unit_class(**kwargs, **{selector: model})
        assert getattr(unit, selector) == model
    else:
        with pytest.raises(ValueError) as error:
            unit_class(**kwargs, **{selector: model})
        assert str(error.value) == (
            f'{selector} must be one of {VALID_ACTIVITY_MODELS}, got {model!r}')


def _evaporator_case(unit_class, liquid, model, pressure, supercritical=False):
    """Build a real residual caller independently of the initialization solver.

    Parameters
    ----------
    unit_class : type
        Batch or continuous evaporator class.
    liquid : LiquidPhase
        Synthetic binary phase.
    model : str
        Activity model selector.
    pressure : float
        Residual evaluation pressure [Pa].
    supercritical : bool, optional
        Whether the second species is supercritical at the evaluation state.

    Returns
    -------
    unit : Evaporator or ContinuousEvaporator
        Unit with real liquid and vapor collaborators.
    arguments : dict
        Material-balance arguments with amounts [mol], fractions [-],
        temperature [K], pressure [Pa], energy [J], time [s], and inlet flow
        [mol/s]. These arbitrary non-equilibrium holdups expose residual order.
    """
    unit = unit_class(vol_drum=1.0, pressure=pressure, activity_model=model)
    unit.Liquid_1 = liquid
    unit.Vapor_1 = VaporPhase(liquid.path_data, mole_frac=COMPOSITION,
                              temp=TEMPERATURE, pres=pressure, moles=1.0)
    unit.is_supercritic = np.array([False, supercritical])
    arguments = dict(
        time=0.0, mol_i=np.array([3.0, 7.0]),  # [s], [mol]
        x_liq=COMPOSITION, y_vap=np.array([0.7, 0.3]),  # [-]
        mol_liq=6.0, mol_vap=2.0,  # [mol]
        pres=pressure, u_int=0.0, temp=TEMPERATURE,  # [Pa], [J], [K]
        u_inputs={'mole_flow': 0.0, 'mole_frac': COMPOSITION},  # [mol/s], [-]
    )
    if unit_class is Evaporator:
        arguments['dmoli_dt'] = np.zeros(2)  # [mol/s]
    return unit, arguments


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
@pytest.mark.parametrize('model', ['ideal', 'UNIQUAC'])
def test_evaporator_equilibrium_and_pressure_use_same_model(liquid, unit_class, model):
    """Require coincident VLE and pressure closure for both model choices.

    Parameters
    ----------
    liquid : LiquidPhase
        Equimolar binary fixture with analytic gamma=4/3 [-] at 350 K.
    unit_class : type
        Batch or continuous evaporator.
    model : str
        Ideal or UNIQUAC activity model.
    """
    pressure_sat = liquid.AntoineEquation(TEMPERATURE)  # [Pa]
    gamma = 1.0 if model == 'ideal' else 4/3  # [-], analytic fixture gamma
    bubble_pressure = gamma * np.dot(COMPOSITION, pressure_sat)  # [Pa]
    unit, arguments = _evaporator_case(unit_class, liquid, model, bubble_pressure)
    vapor_fractions = COMPOSITION * gamma * pressure_sat / bubble_pressure  # [-]
    arguments['y_vap'] = vapor_fractions
    residual = unit.material_balances(**arguments)[1]  # [mol], [-], [mol], [m**3], [Pa]
    np.testing.assert_allclose(vapor_fractions.sum(), 1, rtol=RTOL, atol=0)
    np.testing.assert_allclose(residual[2:4], 0, rtol=0, atol=RTOL)
    assert residual[-1] == pytest.approx(0, abs=PRESSURE_ATOL)
    # Away from closure, pressure residual must change with exactly P*(sum Kx-1).
    arguments['pres'] = 2 * bubble_pressure  # [Pa], deliberate off-equilibrium state
    off_residual = unit.material_balances(**arguments)[1]
    assert off_residual[-1] == pytest.approx(-bubble_pressure, rel=RTOL)
    # Changing only the selector must change both equilibrium and pressure.
    unit.activity_model = 'UNIQUAC' if model == 'ideal' else 'ideal'
    other = unit.material_balances(**arguments)[1]
    assert not np.allclose(other[2:4], off_residual[2:4], rtol=RTOL, atol=RTOL)
    # At P=2*P_bub, the switched residual is
    # P_bub*(gamma_other/gamma_initial - 2): 4/3-2=-2/3 when starting
    # ideal, and 3/4-2=-5/4 when starting UNIQUAC.
    expected_other = (-2 * bubble_pressure / 3 if model == 'ideal'
                      else -5 * bubble_pressure / 4)  # [Pa]
    assert other[-1] == pytest.approx(expected_other, rel=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
def test_ideal_evaporator_full_residual_matches_old_expression(liquid, unit_class):
    """Compare the entire ideal residual vector with the pre-fix equations.

    Parameters
    ----------
    liquid : LiquidPhase
        Subcritical binary phase at 350 K and 1e5 Pa.
    unit_class : type
        Batch or continuous evaporator.
    """
    unit, args = _evaporator_case(unit_class, liquid, 'ideal', PRESSURE)
    result = unit.material_balances(**args)
    # Independent old algebraic ordering and ideal Raoult expression.
    old_residual = np.concatenate((
        args['mol_liq'] * args['x_liq'] + args['mol_vap'] * args['y_vap'] - args['mol_i'],
        args['y_vap'] - liquid.AntoineEquation(TEMPERATURE) / PRESSURE * args['x_liq'],
        [args['mol_liq'] + args['mol_vap'] - sum(args['mol_i'])],
        [args['mol_liq'] * np.dot(COMPOSITION, liquid.mw / liquid.rho_liq) / 1000
         + args['mol_vap'] * 8.314 * TEMPERATURE / PRESSURE - unit.vol_tot],
        [np.dot(COMPOSITION, liquid.AntoineEquation(TEMPERATURE)) - PRESSURE],
    ))  # [mol], [-], [mol], [m**3], [Pa]; R matches the existing gas law
    np.testing.assert_allclose(result[1], old_residual, rtol=RTOL, atol=RTOL)
    # Both fixtures have zero feed and closed liquid outlet; differential
    # residuals are therefore solely the computed vapor removal [mol/s].
    vapor_flow = result[2] if unit_class is Evaporator else result[3]  # [mol/s]
    np.testing.assert_allclose(result[0], -vapor_flow * args['y_vap'],
                               rtol=RTOL, atol=0)


def test_batch_pressure_retains_supercritical_partial_pressure(liquid):
    """Check explicit noncondensable pressure at and away from closure.

    Parameters
    ----------
    liquid : LiquidPhase
        Binary fixture whose second critical temperature is lowered below
        350 K to exercise the existing Henry [Pa] treatment.
    """
    # Give masked B a nonzero trial liquid fraction to expose accidental
    # inclusion of its Henry term. Only its explicit y_B*P enters closure.
    liquid.t_crit[1] = TEMPERATURE - 1  # [K], puts B strictly above its critical T
    vapor_super_fraction = 0.2  # [-], arbitrary noncondensable vapor inventory
    saturation = liquid.AntoineEquation(TEMPERATURE)[0]  # [Pa]
    liquid_condensable_fraction = 0.75  # [-], nonzero masked B trial fraction
    pressure = (liquid_condensable_fraction * saturation
                / (1 - vapor_super_fraction))  # [Pa], vapor closure
    unit, args = _evaporator_case(Evaporator, liquid, 'ideal', pressure, True)
    args['x_liq'] = np.array([liquid_condensable_fraction,
                              1 - liquid_condensable_fraction])  # [-]
    args['y_vap'] = np.array([1 - vapor_super_fraction, vapor_super_fraction])  # [-]
    k_values = liquid.getKeqVLE(temp=TEMPERATURE, pres=pressure,
                                x_liq=args['x_liq'])  # [-]
    assert k_values[1] == pytest.approx(liquid.henry_constant[1] / pressure, rel=RTOL)
    result = unit.material_balances(**args)[1]
    assert result[2] == pytest.approx(0, abs=RTOL)
    assert result[-1] == pytest.approx(0, abs=PRESSURE_ATOL)
    args['y_vap'] = np.array([0.7, 0.3])  # [-], increases explicit partial pressure
    old_pressure = (liquid_condensable_fraction * saturation
                    + args['y_vap'][1] * pressure - pressure)  # [Pa]
    assert unit.material_balances(**args)[1][-1] == pytest.approx(old_pressure, rel=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
def test_nitrogen_database_missing_henry_data(unit_class):
    """Check masked batch closure and actionable continuous missing-data errors.

    Parameters
    ----------
    unit_class : type
        Batch or continuous evaporator, with real nitrogen-augmented phases.

    Notes
    -----
    The supplied 550 K, 2.586e6 Pa review reproducer makes A, C and solvent
    supercritical without Henry data. Nitrogen has Henry data. The rounded
    pressure need not give zero; the batch contract is the old ideal expression.
    """
    root = Path(__file__).resolve().parents[1]
    paths = [str(root / 'tests/Flowsheet/data/compound_database.json'),
             str(root / 'data/evaporator/props_nitrogen.json')]
    temperature = 550.0  # [K], supplied missing-Henry-data reproducer
    pressure = 2.586e6  # [Pa], supplied rounded pressure
    x_liq = np.array([0., 1., 0., 0., 0., 0.])  # [-], pure subcritical B
    y_vap = np.array([0.1, 0.7, 0., 0., 0., 0.2])  # [-], includes nitrogen
    liquid = LiquidPhase(paths, temp=temperature, pres=pressure,
                         mole_frac=x_liq, moles=1.)
    vapor = VaporPhase(paths, temp=temperature, pres=pressure,
                       mole_frac=y_vap, moles=1.)
    unit = unit_class(vol_drum=1., pressure=pressure)
    unit.Liquid_1 = liquid
    unit.Vapor_1 = vapor
    supercritical = temperature > liquid.t_crit
    unit.is_supercritic = supercritical
    k_values = liquid.getKeqVLE()  # [-]
    assert np.isnan(k_values[supercritical]).any()
    assert np.isfinite(k_values[~supercritical]).all()
    arguments = dict(
        time=0., mol_i=x_liq + y_vap,  # [s], [mol], one mole of each phase
        x_liq=x_liq, y_vap=y_vap, mol_liq=1., mol_vap=1.,  # [-], [mol]
        pres=pressure, temp=temperature, u_int=0.,  # [Pa], [K], [J]
        u_inputs={'mole_flow': 0., 'mole_frac': x_liq},  # [mol/s], [-]
    )
    if unit_class is ContinuousEvaporator:
        with pytest.raises(ValueError) as error:
            unit.material_balances(**arguments)
        assert str(error.value) == (
            "Non-finite VLE K-values for species ['A', 'C', 'solvent']. Check "
            "missing or non-finite 'henry_constant' [Pa] data for "
            "supercritical species before evaluating material balances.")
    else:
        arguments['dmoli_dt'] = np.zeros(6)  # [mol/s]
        residual = unit.material_balances(**arguments)[1]
        old_pressure = (np.dot(liquid.AntoineEquation(temperature)
                               * (~supercritical), x_liq)
                        + sum(y_vap[supercritical] * pressure) - pressure)  # [Pa]
        assert np.isfinite(residual).all()
        assert residual[-1] == pytest.approx(old_pressure, rel=RTOL,
                                             abs=PRESSURE_ATOL)


def test_continuous_pressure_uses_finite_supercritical_henry(liquid):
    """Require Henry-based closure above the binary fixture's critical threshold.

    Parameters
    ----------
    liquid : LiquidPhase
        Binary fixture with finite Henry constants [Pa].
    """
    unit, args = _evaporator_case(ContinuousEvaporator, liquid, 'ideal', PRESSURE)
    args['temp'] = 450.0  # [K], A subcritical and B above its 400 K threshold
    k_values = liquid.getKeqVLE(temp=args['temp'], pres=PRESSURE,
                                x_liq=COMPOSITION)  # [-]
    assert k_values[1] == pytest.approx(liquid.henry_constant[1] / PRESSURE,
                                       rel=RTOL)
    expected_pressure = PRESSURE * (np.dot(COMPOSITION, k_values) - 1)  # [Pa]
    residual = unit.material_balances(**args)[1]
    assert residual[-1] == pytest.approx(expected_pressure, rel=RTOL)


@pytest.mark.parametrize('boundary', ['pure', 'phase'])
def test_partial_vapor_viscosity_is_actionable(tmp_path, boundary):
    """Reject viscosity data that omit one species instead of returning NaN.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated JSON directory.
    boundary : str
        Pure-property entry point or real vapor-phase mixing caller.
    """
    database = deepcopy(THERMO_TWO_SPECIES)
    database['light']['visc_gas'] = 1e-5  # [Pa*s], synthetic partial-data case
    path = tmp_path / 'partial_viscosity.json'
    path.write_text(json.dumps(database))
    vapor = VaporPhase(str(path), mole_frac=COMPOSITION, moles=1.)
    assert np.isnan(vapor.visc_gas[1])
    with pytest.raises(ValueError, match=r"missing 'visc_gas'.*\[Pa\*s\].*JSON"):
        if boundary == 'pure':
            vapor.getViscosityPure(phase='vapor')
        else:
            vapor.getViscosity()


@pytest.mark.parametrize('boundary', ['getViscosityPure', 'getViscosityMix'])
def test_viscosity_rejects_unknown_phase(liquid, boundary):
    """Reject unknown phase names at both viscosity entry points.

    Parameters
    ----------
    liquid : LiquidPhase
        Real phase supplying property entry points.
    boundary : str
        Pure or mixture viscosity method name.
    """
    with pytest.raises(ValueError) as error:
        getattr(liquid, boundary)(phase='gas')
    assert str(error.value) == "phase must be one of ('liquid', 'vapor'), got 'gas'"


@pytest.mark.parametrize('boundary', ['activity', 'vle'])
def test_uniquac_qip_fallback_at_both_boundaries(thermo_path, boundary):
    """Require qi fallback before either public UNIQUAC caller runs first.

    Parameters
    ----------
    thermo_path : str
        Synthetic equimolar binary JSON with analytic gamma=4/3 [-].
    boundary : str
        Activity-coefficient or VLE entry point exercised on a fresh phase.
    """
    path = Path(thermo_path)
    database = json.loads(path.read_text())
    for species in ('light', 'heavy'):
        del database[species]['qip']
    path.write_text(json.dumps(database))
    phase = LiquidPhase(thermo_path, mole_frac=COMPOSITION, temp=TEMPERATURE,
                        pres=PRESSURE, moles=1.)
    assert not hasattr(phase, 'qip')
    expected_gamma = np.full(2, 4/3)  # [-], analytic symmetric UNIQUAC fixture
    with pytest.warns(UserWarning, match="qip.*qi.*water.*alcohol"):
        if boundary == 'activity':
            np.testing.assert_allclose(phase.getActivityCoeff(method='UNIQUAC'),
                                       expected_gamma, rtol=RTOL, atol=0)
        else:
            expected_k = (expected_gamma * phase.AntoineEquation(TEMPERATURE)
                          / PRESSURE)  # [-]
            np.testing.assert_allclose(phase.getKeqVLE(gamma_model='UNIQUAC'),
                                       expected_k, rtol=RTOL, atol=0)
    assert not hasattr(phase, 'qip')
    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter('always')
        phase.getActivityCoeff(method='UNIQUAC')
    assert not repeated
