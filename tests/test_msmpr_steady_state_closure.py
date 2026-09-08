"""#223 steady states of the model's own solute bookkeeping.

Tests check the dynamic MSMPR balance under the helper's documented constant
property and growth assumptions, not stream mass conservation. Exponential
moments satisfy mu_j = j! B G**j tau**(j+1). Independent Simpson integration
of the returned distribution checks the dynamic moment and solute equations.
"""

from types import SimpleNamespace
from decimal import Decimal, localcontext
import copy

import numpy as np
import pytest
from scipy.integrate import simpson

from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase

pytestmark = pytest.mark.unit
TEMPERATURE = 310.0  # [K], synthetic isothermal vessel
DENSITY = 1000.0  # [kg/m**3], constant liquid density
# Arithmetic roundoff floor for the resolved grid and balance evaluation.
# Low-conversion tolerances additionally use concentration-root resolution
# divided by depletion (closure_rtol); cancellation is the dominant error.
ROUND_OFF = 128 * np.finfo(float).eps  # [-]
POPULATION_CLOSURE_RTOL = 1e-5  # [-], prescribed relative population gate
# Resolved cases reach 2.6e-7 error; false roots exceed 1.9e-2. This
# tolerance separates them by more than a decade on either side.


def make_unit(data_path, basis='mass_frac', secondary=False, num_mom=4):
    """Construct a constant-property MSMPR with a solid-free inlet.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    basis : str, optional
        Kinetic composition basis, mass_frac [kg/kg] or mass_conc [kg/m**3].
    secondary : bool, optional
        Enable concentration-dependent secondary nucleation.
    num_mom : int, optional
        Number of population moments, including order zero.

    Returns
    -------
    MSMPR
        Moment-mode unit with equal inlet/tank liquid densities.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    unit = MSMPR('A', method='moments', basis=basis)
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=2.0,
                         mass_frac=[0.2, 0.2, 0.2, 0.2, 0.2])  # [-]
    liquid.rho_liq[:] = DENSITY  # [kg/m**3], composition-independent density
    liquid.updatePhase(mass_frac=liquid.mass_frac, vol=2.0)  # [m**3], retain charge
    # 80 decay lengths suppress the tail; Simpson's error at 0.0005 decay
    # lengths per interval is within the floating-point accumulation bound.
    grid = np.linspace(0.0, 8000.0, 160001)  # [um], G*tau = 100 um
    solid = SolidPhase(path, temp=TEMPERATURE, kv=0.5,
                       x_distrib=grid, distrib=np.zeros_like(grid), num_mom=num_mom,
                       mass_frac=[1, 0, 0, 0, 0])  # [-], pure A crystals
    solid.rho_solid[:] = 2000.0  # [kg/m**3], twice the liquid density
    unit.Phases = (liquid, solid)
    unit._Inlet = SimpleNamespace(vol_flow=2.0, Liquid_1=copy.deepcopy(liquid))
    # [m**3/s], two cubic metres of slurry give D=1/s.
    basis_scale = DENSITY if basis == 'mass_conc' else 1.0  # [kg/m**3] or [-]
    unit.Kinetics = CrystKinetics(
        coeff_solub=[0.01 * basis_scale],  # [basis unit], same physical solubility
        nucl_prim=(1e8 if secondary else 1e10, 0.0, 0.0),
        # [#/m**3/s], [J/mol], [-], constant primary rate
        growth=(100.0, 0.0, 0.0),  # [um/s], [J/mol], [-], constant G
        nucl_sec=(((1 / 18) / 3e-12, 0.0, 1.0, 1.0)
                  if secondary else (0, 0, 0, 0)))
    # Secondary prefactor: [#/m**3/s per solid volume fraction]. Since
    # kv*mu_3=3e-12*B, the feedback gain is relative supersaturation / 18.
    # Other parameters: [J/mol], [-], [-].
    unit.Kinetics.target_idx = unit.target_ind
    return unit


@pytest.mark.parametrize('basis', ['mass_frac', 'mass_conc'])
@pytest.mark.parametrize('secondary,seed', [(False, 0.15), (True, 0.195),
                                           (True, 0.18)])
def test_steady_state_matches_dynamic_balance(data_path, basis, secondary, seed):
    """Check independent exponential moments and the real dynamic RHS.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    basis : str
        Composition basis used by both the helper and kinetics.
    secondary : bool
        Enable concentration-dependent secondary nucleation.
    seed : float
        Initial target mass fraction [kg/kg], converted to the configured basis.
    """
    unit = make_unit(data_path, basis, secondary)
    scale = DENSITY if basis == 'mass_conc' else 1.0  # [kg/m**3] or [-]
    grid, distribution, composition, info, residual = unit.solve_steady_state(
        seed * scale, TEMPERATURE)  # [unit's composition basis]
    if secondary:
        # q=kv*mu_3=0.0054/(19-100*w). Substituting into the dynamic solute
        # balance .2-w+q*(3*w-2)=0 gives 100*w**2-38.9838*w+3.7892=0.
        # The smaller root is below the feed and has a positive population.
        # Evaluate the exact fixture polynomial with decimal guard digits;
        # double rounding of w would be amplified by secondary feedback.
        with localcontext() as context:
            context.prec = 50  # decimal digits, well beyond float64 precision
            coefficient = Decimal('38.9838')  # [-], fixture polynomial coefficient
            fraction_decimal = (coefficient - (
                coefficient**2 - 400 * Decimal('3.7892')).sqrt()) / 200  # [kg/kg]
            rate_decimal = Decimal('1e8') / (1 - (100 * fraction_decimal - 1) / 18)
            # [#/m**3/s], independently solved primary/secondary fixed point
        expected_fraction = float(fraction_decimal)  # [kg/kg]
        expected_rate = float(rate_decimal)  # [#/m**3/s]
    else:
        expected_fraction = 2 / 13  # [kg/kg], .2-w+.03*(3*w-2)=0
        expected_rate = 1e10  # [#/m**3/s], constant primary nucleation
    assert info.converged
    assert composition / scale == pytest.approx(expected_fraction, rel=ROUND_OFF, abs=0)
    np.testing.assert_allclose(distribution,
                               expected_rate / 100 * np.exp(-grid / 100),
                               rtol=ROUND_OFF, atol=0)
    moments_um = np.array([simpson(distribution * grid**order, x=grid)
                           for order in range(unit.num_distr)])  # [um**n/m**3]
    moments_si = moments_um * np.array([1, 1e-6, 1e-12, 1e-18])  # [m**n/m**3]
    expected_moments = expected_rate * np.array([1, 100, 20000, 6000000])
    # [um**n/m**3], j!*G**j*tau**(j+1) with G=100 um/s and tau=1 s
    np.testing.assert_allclose(moments_um, expected_moments, rtol=ROUND_OFF, atol=0)
    concentrations = unit.Liquid_1.mass_conc.copy()  # [kg/m**3]
    concentrations[0] = composition / scale * DENSITY  # [kg/m**3]
    inputs = {'Inlet': {'vol_flow': unit.Inlet.vol_flow,
                        'mu_n': np.zeros(unit.num_distr)},
              'Liquid_1': {'mass_conc': unit.Inlet.Liquid_1.mass_conc.copy()}}
    # [m**3/s], [m**n/m**3] zero inlet solids, [kg/m**3]
    rhs, _ = unit.material_balances(
        0.0, None, inputs, [[DENSITY, 2000.0], [DENSITY, 2000.0]],
        moments_si, moments_um, concentrations, TEMPERATURE, None,
        unit.vol_slurry, [1.0, 0.0])
    # RHS: [um**n/m**3/s], [composition basis/s]; inlet is entirely liquid.
    dilution_rate = unit.Inlet.vol_flow / unit.vol_slurry  # [1/s]
    np.testing.assert_allclose(rhs[:unit.num_distr] / expected_moments,
                               0, rtol=0, atol=ROUND_OFF * dilution_rate)
    composition_rate_scale = dilution_rate * 0.2 * scale  # [basis unit/s]
    assert abs(rhs[unit.num_distr]) <= ROUND_OFF * composition_rate_scale


def test_invalid_composition_basis(data_path):
    """Reject an unknown basis before computing kinetics.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.basis = 'mole_frac'
    with pytest.raises(ValueError, match='basis.*mass_conc.*mass_frac'):
        unit.solve_steady_state(0.15, TEMPERATURE)


@pytest.mark.parametrize('seed', [0.0, 0.01])  # [kg/kg], below/at solubility
def test_nonpositive_growth_has_actionable_error(data_path, seed):
    """Name the invalid trial composition instead of dividing by zero.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    seed : float
        Initial target mass fraction [kg/kg].
    """
    unit = make_unit(data_path)
    unit.Kinetics.params['growth'] = [100.0, 0.0, 1.0]
    # [um/s], [J/mol], [-], linear growth vanishes at saturation.
    with pytest.raises(ValueError, match='Trial composition.*solubility'):
        unit.solve_steady_state(seed, TEMPERATURE)


@pytest.mark.parametrize('case', ['zero_kv', 'cancelled_coefficient'])
def test_zero_transfer_keeps_inlet_composition(data_path, case):
    """Keep the feed for kv=0; reject cancellation with nonzero crystal volume.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    case : str
        Remove crystal volume or cancel its net target-composition effect.
    """
    unit = make_unit(data_path)
    if case == 'zero_kv':
        unit.Solid_1.kv = 0.0  # [-], compatibility with the scalar-seed fixture
        expected_fraction = 0.2  # [kg/kg], unchanged inlet
    else:
        # rho_s*(1-w)-rho_l*w=0 at w=rho_s/(rho_s+rho_l)=2/3.
        expected_fraction = 2 / 3  # [kg/kg], exact cancellation in this model
        unit.Inlet.Liquid_1.updatePhase(mass_frac=[2 / 3, 1 / 3, 0, 0, 0], vol=2.0)
        # [-], [m**3], synthetic binary inlet with the cancellation composition
        with pytest.raises(ValueError, match=r'Unsupported feed concentration.*c\*.*different feed'):
            unit.solve_steady_state(expected_fraction, TEMPERATURE)
        return
    grid, distribution, composition, info, residual = unit.solve_steady_state(
        expected_fraction, TEMPERATURE)
    assert info.converged
    assert composition == pytest.approx(expected_fraction, rel=ROUND_OFF, abs=0)
    np.testing.assert_allclose(distribution, 1e8 * np.exp(-grid / 100),
                               rtol=ROUND_OFF, atol=0)
    dilution_rate = unit.Inlet.vol_flow / unit.vol_slurry  # [1/s]
    assert abs(residual) <= ROUND_OFF * dilution_rate * expected_fraction


def test_configured_moment_count_reaches_kinetics(data_path):
    """Pass all five configured moments, including the independent fourth.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path, num_mom=5)
    unit.solve_steady_state(0.15, TEMPERATURE)  # [kg/kg], supersaturated seed
    cached_moments, _ = unit.Kinetics._moment_inputs  # [m**n/m**3], [-]
    assert cached_moments.shape == (5,)
    # mu_4=4!*B*G**4*tau**5, with B=1e10, G=1e-4 m/s, tau=1 s.
    assert cached_moments[4] == pytest.approx(2.4e-5, rel=ROUND_OFF, abs=0)


def closure_rtol(unit, composition):
    """Bound boundary-density error from concentration-root resolution.

    Parameters
    ----------
    unit : MSMPR
        Constant-density fixture with dilution rate 1/s.
    composition : float
        Root composition in the configured basis [kg/kg] or [kg/m**3].

    Returns
    -------
    float
        Relative tolerance [-] from root resolution divided by depletion,
        plus the arithmetic roundoff bound. Boundary recovery subtracts
        c from c_in; this cancellation dominates at low conversion. The
        dynamic closure assertion additionally caps relative population
        error at the helper's prescribed 1e-5 acceptance tolerance.
    """
    scale = DENSITY if unit.basis == 'mass_frac' else 1.0  # [kg/m**3] or [-]
    concentration = composition * scale  # [kg/m**3]
    inlet = unit.Inlet.Liquid_1.mass_conc[0]  # [kg/m**3]
    resolution = 4 * np.finfo(float).eps * (inlet + abs(concentration))
    # [kg/m**3], specified xtol plus brentq's default rtol contribution
    return ROUND_OFF + resolution / (inlet - concentration)


def assert_dynamic_closure(unit, composition, distribution, growth, kinetic_rtol=0):
    """Check moments of the returned exponential through the actual RHS.

    Parameters
    ----------
    unit : MSMPR
        Constant-density moment-mode unit.
    composition : float
        Returned target composition [kg/kg] or [kg/m**3].
    distribution : ndarray
        Returned CSD [#/m**3/um], whose first point is at zero size.
    growth : float
        Independently evaluated size-independent growth [um/s].
    kinetic_rtol : float, optional
        Relative kinetic-rate error [-] from concentration-root resolution
        near a sensitive kinetic threshold; zero unless needed by the case.

    Returns
    -------
    float
        Dynamic target derivative [composition unit/s].
    """
    dilution = unit.Inlet.vol_flow / unit.vol_slurry  # [1/s]
    length = growth / dilution  # [um], exponential decay length
    moments_um = distribution[0] * np.array([
        length, length**2, 2 * length**3, 6 * length**4])  # [um**n/m**3]
    moments_si = moments_um * np.array([1, 1e-6, 1e-12, 1e-18])  # [m**n/m**3]
    scale = DENSITY if unit.basis == 'mass_frac' else 1.0  # [kg/m**3] or [-]
    concentrations = unit.Liquid_1.mass_conc.copy()  # [kg/m**3]
    concentrations[0] = composition * scale  # [kg/m**3]
    inlet_concentrations = unit.Inlet.Liquid_1.mass_conc.copy()  # [kg/m**3]
    inputs = {'Inlet': {'vol_flow': unit.Inlet.vol_flow, 'mu_n': np.zeros(4)},
              'Liquid_1': {'mass_conc': inlet_concentrations}}
    # [m**3/s], [m**n/m**3] zero inlet solids, [kg/m**3]
    rhs, _ = unit.material_balances(
        0.0, None, inputs, [[DENSITY, 2000.0],
                            [unit.Inlet.Liquid_1.getDensity(), 2000.0]],
        moments_si, moments_um, concentrations, TEMPERATURE, None,
        unit.vol_slurry, [1.0, 0.0])
    population_rtol = min(closure_rtol(unit, composition) + kinetic_rtol,
                          POPULATION_CLOSURE_RTOL)  # [-], numerical bound and gate
    np.testing.assert_allclose(rhs[:4] / moments_um, 0, rtol=0,
                               atol=population_rtol * dilution)
    target_scale = dilution * inlet_concentrations[0] / scale  # [basis unit/s]
    assert abs(rhs[4]) <= ROUND_OFF * target_scale
    return rhs[4]


@pytest.mark.parametrize('exponent,expected', [
    (1, 0.072064174127790499), (2, 0.035717325051584954)])
@pytest.mark.parametrize('seed', [0.03, 0.15, 2 / 3])  # [kg/kg], valid seed hints
def test_variable_growth_bracket(data_path, exponent, expected, seed):
    """Return the physical primary-nucleation root independently of the seed.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    exponent : int
        Growth supersaturation exponent [-].
    expected : float
        Target mass fraction [kg/kg]. Roots of
        .2-w+3e-10*(100*(100*w-1)**exponent)**3*(3*w-2)=0,
        computed by 60-digit decimal bisection on (.01, .2).
    seed : float
        Compatible supersaturated seed [kg/kg].
    """
    unit = make_unit(data_path)
    unit.Kinetics.params['nucl_prim'] = [1e8, 0.0, 0.0]  # [#/m**3/s], [J/mol], [-]
    unit.Kinetics.params['growth'] = [100.0, 0.0, exponent]  # [um/s], [J/mol], [-]
    _, distribution, composition, info, residual = unit.solve_steady_state(seed, TEMPERATURE)
    assert info.converged
    assert composition == pytest.approx(expected, rel=ROUND_OFF, abs=0)
    growth = 100 * (composition / 0.01 - 1)**exponent  # [um/s], fixture law
    assert distribution[0] > 0
    assert distribution[0] * growth == pytest.approx(1e8, rel=ROUND_OFF, abs=0)
    actual = assert_dynamic_closure(unit, composition, distribution, growth)
    assert residual == pytest.approx(actual, rel=0, abs=ROUND_OFF * 0.2)
    # [kg/kg/s], D=1/s and inlet mass fraction .2 set the residual scale.


def test_singular_seed_does_not_choose_negative_population(data_path):
    """Treat the old denominator pole as a hint, never as a solved state.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    _, distribution, composition, _, _ = unit.solve_steady_state(2 / 3, TEMPERATURE)
    assert composition == pytest.approx(2 / 13, rel=ROUND_OFF, abs=0)
    assert distribution[0] == pytest.approx(1e8, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


def test_variable_growth_with_secondary_nucleation(data_path):
    """Close population feedback with a concentration-dependent growth rate.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Kinetics.params['nucl_prim'] = [1e8, 0.0, 0.0]  # [#/m**3/s], [J/mol], [-]
    unit.Kinetics.params['nucl_sec'] = [1e9, 0.0, 0.0, 1.0]
    # [#/m**3/s per solid volume fraction], [J/mol], [-], [-], finite feedback
    unit.Kinetics.params['growth'] = [100.0, 0.0, 1.0]  # [um/s], [J/mol], [-]
    _, distribution, composition, _, _ = unit.solve_steady_state(0.15, TEMPERATURE)
    growth = 100 * (composition / 0.01 - 1)  # [um/s]
    assert_dynamic_closure(unit, composition, distribution, growth)
    assert unit.Kinetics.sec_nucl > 0


@pytest.mark.parametrize('basis', ['mass_frac', 'mass_conc'])
@pytest.mark.parametrize('inlet_density', [800.0, 900.0])  # [kg/m**3], unequal feeds
def test_inlet_density_uses_stream_concentration(data_path, basis, inlet_density):
    """Use inlet mass concentration independently of tank density.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    basis : str
        Target composition basis.
    inlet_density : float
        Feed liquid density [kg/m**3].
    """
    unit = make_unit(data_path, basis)
    inlet = unit.Inlet.Liquid_1
    assert inlet is not unit.Liquid_1
    inlet.rho_liq[:] = inlet_density  # [kg/m**3]
    inlet.updatePhase(mass_frac=inlet.mass_frac, vol=2.0)  # [m**3]
    scale = DENSITY if basis == 'mass_conc' else 1.0  # [kg/m**3] or [-]
    _, distribution, composition, _, _ = unit.solve_steady_state(0.15 * scale, TEMPERATURE)
    # Constant B/G gives q=.03; balance c_in/rho_l-w+.03*(3*w-2)=0.
    expected = (0.2 * inlet_density / DENSITY - 0.06) / 0.91  # [kg/kg]
    assert composition / scale == pytest.approx(expected, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


def test_final_residual_exposes_dynamic_impurity_factor(data_path):
    """Return the real dynamic residual when an omitted growth factor matters.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Kinetics.alpha_fn = lambda conc: 2.0  # [-], synthetic doubled dynamic growth
    _, _, composition, _, residual = unit.solve_steady_state(0.15, TEMPERATURE)
    # Nominal transfer is 60 kg/m**3/s. Doubling it adds this negative term
    # to the dynamic target equation; phi=.97 for the nominal exponential.
    expected = -60 * (1 - composition) / (0.97 * DENSITY)  # [kg/kg/s]
    assert residual == pytest.approx(expected, rel=ROUND_OFF, abs=0)


def test_finite_radius_is_rejected(data_path):
    """Reject nuclei that invalidate the zero-size exponential boundary.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.rad = 20.0  # [um], finite-radius case is outside the helper's model
    with pytest.raises(ValueError, match='finite.*radius|rad.*zero'):
        unit.solve_steady_state(0.15, TEMPERATURE)


@pytest.mark.parametrize('bins', [3, 200])
def test_fvm_bin_count_does_not_set_moment_count(data_path, bins):
    """Keep four analytical moments even when the sampled FVM grid differs.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    bins : int
        Number of FVM bins, deliberately unequal to the four phase moments.
    """
    moment_unit = make_unit(data_path)
    path = moment_unit.Liquid_1.path_data
    grid = np.linspace(0.0, 8000.0, bins)  # [um], samples only, not accurate quadrature
    solid = SolidPhase(path, temp=TEMPERATURE, kv=0.5, x_distrib=grid,
                       distrib=np.zeros(bins), mass_frac=[1, 0, 0, 0, 0])
    solid.rho_solid[:] = 2000.0  # [kg/m**3]
    unit = MSMPR('A', method='1D-FVM', basis='mass_frac')
    unit.Phases = (moment_unit.Liquid_1, solid)
    unit.Inlet = moment_unit.Inlet
    unit.Kinetics = moment_unit.Kinetics
    _, distribution, composition, _, _ = unit.solve_steady_state(0.15, TEMPERATURE)
    assert distribution.shape == (bins,)
    assert unit.Kinetics._moment_inputs[0].shape == (4,)
    assert composition == pytest.approx(2 / 13, rel=ROUND_OFF, abs=0)


@pytest.mark.parametrize('feed_fraction', [0.0, 0.01])  # [kg/kg], below/at solubility
def test_inlet_error_identifies_inlet(data_path, feed_fraction):
    """Distinguish an invalid feed from a valid supersaturated trial hint.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    feed_fraction : float
        Inlet target mass fraction [kg/kg].
    """
    unit = make_unit(data_path)
    unit.Inlet.Liquid_1.updatePhase(
        mass_frac=[feed_fraction, 1 - feed_fraction, 0, 0, 0], vol=2.0)
    # [-], [m**3], solid-free binary feed at the requested composition
    message = ('Inlet target mass concentration' if feed_fraction == 0
               else 'scanned tank concentration domain.*Washout')
    with pytest.raises(ValueError, match=message):
        unit.solve_steady_state(0.15, TEMPERATURE)


def test_no_positive_root_has_actionable_bracket_error(data_path):
    """Report the scanned domain and excluded washout states when B=0.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Kinetics.params['nucl_prim'] = [0.0, 0.0, 0.0]  # [#/m**3/s], [J/mol], [-]
    with pytest.raises(ValueError, match='scanned tank concentration domain.*Washout'):
        unit.solve_steady_state(0.15, TEMPERATURE)


def test_zero_transfer_rejects_zero_population(data_path):
    """Require a positive boundary from the kv=0 closed-form evaluation.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Solid_1.kv = 0.0  # [-], zero-transfer branch
    unit.Kinetics.params['nucl_prim'] = [0.0, 0.0, 0.0]  # [#/m**3/s], [J/mol], [-]
    with pytest.raises(RuntimeError, match='boundary must be positive'):
        unit.solve_steady_state(0.15, TEMPERATURE)


def test_population_closure_is_checked_after_solver(data_path, monkeypatch):
    """Reject a premature solver return using independently evaluated kinetics.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only the external scalar root solver with a premature return.
    """
    unit = make_unit(data_path)
    brackets = []

    def premature_root(function, lower, upper, full_output, xtol):
        """Emulate a solver claiming convergence at the bracket midpoint.

        Parameters
        ----------
        function : callable
            Scalar rescaled residual [kg/m**3/s].
        lower, upper : float
            Admissible concentration bounds [kg/m**3].
        full_output : bool
            Request the solver convergence record.

        xtol : float
            Requested absolute concentration resolution [kg/m**3].

        Returns
        -------
        tuple
            Incorrect root [kg/m**3] and a successful convergence record.
        """
        assert full_output
        assert xtol == 4 * np.finfo(float).eps * 200  # [kg/m**3], feed-scaled resolution
        assert function(lower) > 0
        assert function(upper) < 0
        brackets.append((lower, upper))
        return (lower + upper) / 2, SimpleNamespace(converged=True)

    monkeypatch.setattr('PharmaPy.Crystallizers.brentq', premature_root)
    with pytest.raises(ValueError, match='No accepted steady root.*scanned'):
        unit.solve_steady_state(0.15, TEMPERATURE)
    assert len(brackets) == 1
    assert 10 < brackets[0][0] < 2000 / 13 < brackets[0][1] < 200
    # [kg/m**3], bracket surrounds the independently known 2000/13 root


@pytest.mark.parametrize('exponent,seed,expected,boundary', [
    (0.5, 150.0, 10.0179295131, 3.21467e8),
    (1.5, 10.1, 10.000055772098442, 321488934.62775434),
    (1.5, 199.0, 10.000055772098442, 321488934.62775434)])
def test_secondary_only_root_selection(data_path, exponent, seed, expected, boundary):
    """Select interior area-feedback roots by seed proximity.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    exponent : float
        Secondary area exponent [-].
    seed, expected : float
        Selection hint and independently calculated concentration [kg/m**3].
    boundary : float
        Expected zero-size number density [#/m**3/um].
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics = CrystKinetics(
        coeff_solub=[10.0], nucl_prim=(0, 0, 0), growth=(100, 0, 0),
        nucl_sec=(1e12, 0, 1, exponent), mu_sec_nucl='area')
    unit.Kinetics.target_idx = unit.target_ind
    # [kg/m**3], [#/m**3/s], [um/s]; secondary prefactor multiplies
    # relative supersaturation and (kv*mu_2)**exponent [m**(-exponent)].
    # At exponent 1.5 elimination gives
    # (200-c)*(c-10)**2 = 6e-7-9e-10*c. Expected roots were evaluated
    # by 65-digit decimal bisection, not the production residual. The second
    # root at 199.99999999998838 has float64 population error 4.2e-4,
    # above the 1e-5 closure gate, so both hints select the lower root.
    _, distribution, composition, info, _ = unit.solve_steady_state(seed, TEMPERATURE)
    assert info.converged
    # The published exponent-.5 reference carries only 12/6 significant digits.
    reference_rtol = 2e-6 if exponent == 0.5 else closure_rtol(unit, composition)
    # [-], reference precision or concentration-resolution/depletion bound
    composition_rtol = 2e-11 if exponent == 0.5 else ROUND_OFF
    # [-], 12-significant-digit reference or float64 concentration resolution
    assert composition == pytest.approx(expected, rel=composition_rtol, abs=0)
    assert distribution[0] == pytest.approx(boundary, rel=reference_rtol, abs=0)
    resolution = 4 * np.finfo(float).eps * (200 + composition)  # [kg/m**3]
    kinetic_rtol = resolution / (composition - 10)  # [-], d(log B)/dc=1/(c-10)
    # Near onset the secondary supersaturation factor amplifies root error.
    assert_dynamic_closure(unit, composition, distribution, 100.0, kinetic_rtol)


def test_feed_above_amplitude_pole(data_path):
    """Find w=.5 below c* for the w_in=.8, B/G=2e9 synthetic feed.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Inlet.Liquid_1.updatePhase(mass_frac=[0.8, 0.2, 0, 0, 0], vol=2.0)
    # [kg/kg], [m**3], feed above c*/rho_l=2/3
    unit.Kinetics.params['nucl_prim'] = [2e11, 0, 0]  # [#/m**3/s], [J/mol], [-]
    _, distribution, composition, _, _ = unit.solve_steady_state(0.7, TEMPERATURE)
    # q=.6 gives .8-w+.6*(3*w-2)=0, hence w=.5.
    assert composition == pytest.approx(0.5, rel=ROUND_OFF, abs=0)
    assert distribution[0] == pytest.approx(2e9, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


@pytest.mark.parametrize('seed', [0.15, 0.5])  # [kg/kg], root-selection hints
def test_composition_dependent_solubility_domain(data_path, seed):
    """Determine growth admissibility at each composition, not at the hint.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    seed : float
        Root-selection hint [kg/kg].
    """
    unit = make_unit(data_path)
    unit.Kinetics.get_solubility = lambda temp, conc: 0.05 + 0.5 * np.asarray(conc).flat[0]
    # [kg/kg], synthetic solubility; positive growth starts at w=.1.
    _, distribution, composition, _, _ = unit.solve_steady_state(seed, TEMPERATURE)
    assert composition == pytest.approx(2 / 13, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


@pytest.mark.parametrize('primary', [1e4, 1e3, 3e2, 1e2])  # [#/m**3/s]
def test_low_conversion_population_closure(data_path, primary):
    """Allow boundary cancellation error fixed by concentration resolution.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    primary : float
        Constant primary nucleation rate [#/m**3/s].
    """
    unit = make_unit(data_path)
    unit.Kinetics.params['nucl_prim'] = [primary, 0, 0]  # [#/m**3/s], [J/mol], [-]
    _, distribution, composition, _, _ = unit.solve_steady_state(0.15, TEMPERATURE)
    assert distribution[0] * 100 == pytest.approx(
        primary, rel=closure_rtol(unit, composition), abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


def test_absolute_fraction_low_conversion(data_path):
    """Close the absolute-supersaturation fractional primary-growth case.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Kinetics = CrystKinetics(
        coeff_solub=[0.01], sup_sat_type='absolute',
        nucl_prim=(1e8, 0, 1.5), growth=(1, 0, 0))
    unit.Kinetics.target_idx = unit.target_ind
    # [kg/kg], [#/m**3/s/(kg/kg)**1.5], [um/s], synthetic constant growth
    _, distribution, composition, _, _ = unit.solve_steady_state(0.15, TEMPERATURE)
    assert_dynamic_closure(unit, composition, distribution, 1.0)  # [um/s]


@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
@pytest.mark.parametrize('raises', [False, True])
def test_steady_helper_preserves_phase_state(data_path, monkeypatch, method, raises):
    """Preserve phase temperatures and moments even if residual evaluation fails.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Inject a failure after the real material balance has run.
    method : str
        Population discretization.
    raises : bool
        Exercise cleanup on the exceptional path.
    """
    unit = make_unit(data_path)
    unit.method = method
    temperatures = (unit.Liquid_1.temp, unit.Solid_1.temp)  # [K]
    moments = unit.Solid_1.moments.copy()  # [m**n], phase inventory
    material_balances = unit.material_balances

    def evaluate_then_fail(*args, **kwargs):
        """Run the real balance before injecting a residual-evaluation failure.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed unchanged to MSMPR.material_balances, with its
            documented population, concentration, time, and temperature units.

        Raises
        ------
        RuntimeError
            Always, after the real balance updates its caches.
        """
        material_balances(*args, **kwargs)
        raise RuntimeError('residual sentinel')

    if raises:
        monkeypatch.setattr(unit, 'material_balances', evaluate_then_fail)
        with pytest.raises(RuntimeError, match='residual sentinel'):
            unit.solve_steady_state(0.15, 300.0)  # [kg/kg], [K], different temperature
    else:
        unit.solve_steady_state(0.15, 300.0)  # [kg/kg], [K]
    assert (unit.Liquid_1.temp, unit.Solid_1.temp) == temperatures
    np.testing.assert_array_equal(unit.Solid_1.moments, moments)


@pytest.mark.parametrize('num_scan', [1, 1.5, True])
def test_scan_resolution_requires_integer_samples(data_path, num_scan):
    """Reject resolutions that cannot define a concentration interval.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    num_scan : object
        Invalid numerical sample count [-].
    """
    unit = make_unit(data_path)
    with pytest.raises(ValueError, match='num_scan.*integer.*two'):
        unit.solve_steady_state(0.15, TEMPERATURE, num_scan=num_scan)


@pytest.mark.parametrize('seed', [0.6, 0.7])  # [kg/kg], either side of c*
@pytest.mark.parametrize('feed_fraction', [0.8, 2 / 3])  # [kg/kg], includes zero transfer
def test_negative_liquid_holdup_is_rejected(data_path, seed, feed_fraction):
    """Reject w=.65 with q=3, even though the divided dynamic RHS is zero.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    seed : float
        Root-selection hint [kg/kg].
    feed_fraction : float
        Feed target mass fraction [kg/kg], above or at the amplitude pole.
    """
    unit = make_unit(data_path)
    unit.Inlet.Liquid_1.updatePhase(mass_frac=[feed_fraction, 1 - feed_fraction, 0, 0, 0], vol=2.0)
    # [kg/kg], [m**3], feed at or above the amplitude pole
    unit.Kinetics.params['nucl_prim'] = [1e12, 0, 0]  # [#/m**3/s], [J/mol], [-]
    # B/G=1e10 gives kv*mu_3=.5*6e-10*1e10=3, hence phi=-2.
    message = (r'Unsupported feed concentration.*c\*' if feed_fraction == 2 / 3
               else 'No accepted steady root.*liquid volume fraction')
    with pytest.raises(ValueError, match=message):
        unit.solve_steady_state(seed, TEMPERATURE)


@pytest.mark.parametrize('failed_result', ['raises', 'nan', 'inf', 'growth_gap'])
def test_gaussian_solubility_keeps_earlier_root(data_path, monkeypatch, failed_result):
    """Keep a resolved root when a later bracket crosses an unsampled gap.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Exercise scalar-solver backends that return an unusable result.
    failed_result : str
        Raise on the actual growth gap, or return a nonfinite root or residual.
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics.get_solubility = lambda temp, conc: (
        10 + 60 * np.exp(-((np.asarray(conc).flat[0] - 50) / 0.8)**2))
    # [kg/m**3], Gaussian solubility peak at 50 with width .8 kg/m**3
    unit.Kinetics.params['nucl_prim'] = [1.5e10, 0, 1]
    # [#/m**3/s], [J/mol], [-], rate linear in relative supersaturation
    if failed_result != 'raises':
        from scipy.optimize import brentq

        def return_failed_result(*args, **kwargs):
            """Preserve real roots but emulate a backend returning on failure.

            Parameters
            ----------
            *args, **kwargs
                Passed unchanged to brentq; root and xtol are [kg/m**3].

            Returns
            -------
            tuple
                Root [kg/m**3] and convergence information. For the actual
                growth-gap failure only, return an unusable result.
            """
            try:
                return brentq(*args, **kwargs)
            except ValueError:
                result = {'nan': np.nan, 'inf': np.inf, 'growth_gap': 50.0}
                # [kg/m**3], 50 lies inside the known inadmissible growth gap
                return result[failed_result], SimpleNamespace(converged=True)

        monkeypatch.setattr('PharmaPy.Crystallizers.brentq', return_failed_result)
    with pytest.warns(UserWarning, match='skipped brackets: 1.*num_scan') as records:
        _, distribution, composition, info, _ = unit.solve_steady_state(30.0, TEMPERATURE)
    user_warnings = [record for record in records
                     if issubclass(record.category, UserWarning)]
    assert len(user_warnings) == 1
    assert user_warnings[0].filename == __file__  # warning points to the caller
    # Away from the peak, q=.045*(c/10-1) and the solute balance gives
    # .0135*c**2-10.135*c+290=0. Its smaller root is physical.
    expected = (10.135 - np.sqrt(10.135**2 - 4 * 0.0135 * 290)) / (2 * 0.0135)
    # [kg/m**3], independent polynomial solution; Gaussian tail is negligible
    assert info.converged
    assert composition == pytest.approx(expected, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


@pytest.mark.parametrize('width', [0.1, 0.8])  # [kg/m**3], interior/midpoint gaps
def test_solubility_island_reports_skipped_bracket(data_path, width):
    """Report the helper's domain error when growth vanishes around the root.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    width : float
        Width of the solubility peak [kg/m**3].
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics.get_solubility = lambda temp, conc: (
        10 + 200 * np.exp(-((np.asarray(conc).flat[0] - 2000 / 13) / width)**2))
    # [kg/m**3], peak excludes the otherwise unique c=2000/13 root.
    # The narrow case leaves the bracket midpoint admissible but makes
    # brentq evaluate a nonpositive-growth point in its interior.
    with pytest.raises(ValueError, match='No accepted steady root.*skipped brackets: 1'):
        unit.solve_steady_state(150.0, TEMPERATURE)


def test_steep_secondary_root_closes_at_concentration_resolution(data_path):
    """Accept a root resolved in concentration despite a steep rate slope.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics = CrystKinetics(
        coeff_solub=[10], nucl_prim=(0, 0, 0), growth=(100, 0, 0),
        nucl_sec=(3e15, 0, 1, 1.5), mu_sec_nucl='area')
    # [kg/m**3], [#/m**3/s], [um/s]; secondary prefactor multiplies
    # relative supersaturation and (kv*mu_2)**1.5 [m**(-1.5)].
    unit.Kinetics.target_idx = unit.target_ind
    _, distribution, composition, info, _ = unit.solve_steady_state(10.1, TEMPERATURE)
    assert info.converged
    # The nonzero root satisfies (200-c)*(c-10)**2 =
    # (6e-7-9e-10*c)/3000**2. Decimal bisection of this polynomial
    # gives c=10.000000018590697, close to the positive-growth threshold.
    assert composition == pytest.approx(10.000000018590697, rel=ROUND_OFF, abs=0)
    resolution = 4 * np.finfo(float).eps * (200 + composition)  # [kg/m**3]
    kinetic_rtol = resolution / (composition - 10)  # [-], d(log B)/dc=1/(c-10)
    assert_dynamic_closure(unit, composition, distribution, 100.0, kinetic_rtol)


@pytest.mark.parametrize('seed', [99.0, 100.0, 101.0])  # [kg/m**3], around the jump
def test_step_solubility_rejects_jump_and_selects_closed_root(data_path, seed):
    """Reject a sign change at a discontinuity in favor of the actual root.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    seed : float
        Root-selection hint [kg/m**3], close to the discontinuity.
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics.get_solubility = lambda temp, conc: (
        10.0 if np.asarray(conc).flat[0] < 100.0 else 90.0)
    # [kg/m**3], synthetic solubility step at c=100
    unit.Kinetics.params['nucl_prim'] = [3e9, 0, 1]  # [#/m**3/s], [J/mol], [-]
    _, distribution, composition, _, _ = unit.solve_steady_state(seed, TEMPERATURE)
    # Below the step, q=.009*(c/10-1). The balance gives
    # .0027*c**2-2.827*c+218=0; the smaller root lies on this branch.
    expected = (2.827 - np.sqrt(2.827**2 - 4 * .0027 * 218)) / (2 * .0027)
    # [kg/m**3], independent polynomial root: 83.82442202519093
    assert composition == pytest.approx(expected, rel=ROUND_OFF, abs=0)
    assert_dynamic_closure(unit, composition, distribution, 100.0)  # [um/s]


def test_unresolved_secondary_population_is_rejected(data_path):
    """Reject the smooth near-onset root with percent-scale population error.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Kinetics = CrystKinetics(
        coeff_solub=[10], nucl_prim=(0, 0, 0), growth=(100, 0, 0),
        nucl_sec=(1e20, 0, 1, 1.5), mu_sec_nucl='area')
    # [kg/m**3], [#/m**3/s], [um/s]; secondary prefactor multiplies
    # relative supersaturation and (kv*mu_2)**1.5 [m**(-1.5)].
    unit.Kinetics.target_idx = unit.target_ind
    with pytest.raises(ValueError, match=(
            r'No accepted steady root.*rejected by population closure gate.*'
            r'relative closure error=[0-9.e+-]+.*closure_rtol=1e-05')):
        unit.solve_steady_state(10.1, TEMPERATURE)


@pytest.mark.parametrize('primary,secondary', [
    (0.0, (15228426.39593909, 0, 1, 0.5)),
    (1e-10, (10, 0, 0, 0.5))])
def test_cancellation_feed_rejects_secondary_boundary_solve(data_path, primary, secondary):
    """Reject the unsupported c* feed instead of solving a boundary fixed point.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    primary : float
        Primary rate [#/m**3/s], zero or tiny compared to secondary nucleation.
    secondary : tuple
        Area-secondary parameters: prefactor, activation energy [J/mol],
        relative supersaturation exponent [-], and area exponent [-]. The
        prefactor multiplies (kv*mu_2)**0.5 [m**(-0.5)] to give [#/m**3/s].
    """
    unit = make_unit(data_path, basis='mass_conc')
    unit.Inlet.Liquid_1.updatePhase(mass_frac=[2 / 3, 1 / 3, 0, 0, 0], vol=2.0)
    # [kg/kg], [m**3], c_in=c*=2000/3 kg/m**3
    unit.Kinetics = CrystKinetics(
        coeff_solub=[10], nucl_prim=(primary, 0, 0), growth=(100, 0, 0),
        nucl_sec=secondary, mu_sec_nucl='area')
    # [kg/m**3], [#/m**3/s], [um/s]. The first case has a valid b=1e8,
    # phi=.97 fixed point: prefactor=1e10/((200/3-1)*sqrt(100)).
    unit.Kinetics.target_idx = unit.target_ind
    with pytest.raises(ValueError, match=r'Unsupported feed concentration.*c\*.*different feed'):
        unit.solve_steady_state(600.0, TEMPERATURE)  # [kg/m**3]


def test_zero_kv_boundary_is_exact_without_iteration(data_path):
    """Use B/G exactly when secondary area vanishes with kv=0.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit = make_unit(data_path)
    unit.Solid_1.kv = 0.0  # [-], zero crystal volume and secondary area
    unit.Kinetics.params['nucl_prim'] = [100, 0, 0]  # [#/m**3/s], [J/mol], [-]
    unit.Kinetics.params['nucl_sec'] = [10, 0, 0, 0.5]
    # [#/m**3/s per square root solid fraction], [J/mol], [-], [-]
    _, distribution, composition, info, _ = unit.solve_steady_state(0.15, TEMPERATURE)
    assert distribution[0] == 1.0  # exact B/G=100/100; no numerical iteration
    assert composition == pytest.approx(0.2, rel=ROUND_OFF, abs=0)
    assert info.converged
    assert info.iterations == 0
