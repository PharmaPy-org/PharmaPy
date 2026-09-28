"""Composition-basis regressions with real crystallizers, phases, and kinetics.

ODE states and residuals retain mass-concentration units on both kinetic input
bases. Public initialization and ``unit_model`` exercise Batch, Semibatch, and
MSMPR, including real slurry feeds with unequal inlet/tank/crystal densities.
The synthetic property database describes an ideal additive-volume mixture,
not measured material data. No optional solver is needed for these RHS probes.

The analytical-Jacobian probes deliberately use equal pure-liquid densities
and constant growth: the analytic model holds density fixed and its general
kinetic derivatives support mass_conc only. Constant growth makes the two
kinetic bases equivalent; these tests do not claim general mass_frac analytical
Jacobian support.
"""

import json

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit


# Synthetic additive-volume mixture: 1/rho = w_solute/1500 + w_solvent/900.
# Tank w=(1/4, 3/4) gives rho=1000; feed w=(1/2, 1/2) gives rho=1125.
PURE_LIQUID_DENSITIES = (1500.0, 900.0)  # [kg/m**3]
LIQUID_DENSITY = 1000.0  # [kg/m**3], also fixed-density Jacobian case
INLET_LIQUID_DENSITY = 1125.0  # [kg/m**3]
CRYSTAL_DENSITY = 1500.0  # [kg/m**3], synthetic pure-solute crystal
TANK_FRACTIONS = np.array([0.25, 0.75])  # [kg/kg]
INLET_FRACTIONS = np.array([0.5, 0.5])  # [kg/kg]
PURE_SOLUTE = [1.0, 0.0]  # [kg/kg], crystal contains only the target
TANK_CONC = np.array([250.0, 750.0])  # [kg/m**3], rho * tank fractions
INLET_CONC = np.array([562.5, 562.5])  # [kg/m**3], rho * inlet fractions
SHAPE_FACTOR = 0.5  # [-], crystal volume is half the cube of its size
GROWTH_RATE = 2.0  # [um/s], synthetic growth prefactor
SATURATED_MASS_FRAC = 0.125  # [kg/kg], half the initial target fraction
SATURATED_MASS_CONC = 125.0  # [kg/m**3], half the initial concentration

# Monodisperse charge: 2e8 crystals of size 100 um, mu_n=N*L**n.
TANK_MOMENTS_RAW = np.array([2e8, 2e10, 2e12, 2e14])  # [um**n]
TANK_MOMENTS = TANK_MOMENTS_RAW * 1e-6**np.arange(4)  # [m**n]
# Feed: 1e11 crystals/m**3 of size 100 um; kv*mu_3=0.05 solid fraction.
INLET_MOMENTS = np.array([1e11, 1e7, 1e3, 0.1])  # [m**n/m**3]
INLET_LIQUID_FRACTION = 0.95  # [-], complement of kv*inlet mu_3
LIQUID_VOL = 1e-3  # [m**3], initial liquid charge
SLURRY_VOL = 1.1e-3  # [m**3], liquid plus kv*mu_3=1e-4 m**3 crystal
INLET_VOL_FLOW = 1e-5  # [m**3/s], small positive seeded feed
TEMP = 300.0  # [K], isothermal test assumption
PREPARATION_TIME = 1.0  # [s], initialization horizon; no integration occurs

# At relative supersaturation (C/Csat-1)=1, linear growth is 2 um/s.
# Crystal transfer: N*rho_s*kv*3*L**2*G = 0.009 kg/s for this charge.
# Batch loses 0.009/1000 m**3/s, so dC=(-0.009+250*9e-6,
# 750*9e-6)/0.001 = (-6.75, 6.75) kg/m**3/s.
BATCH_DCOMP_DT = np.array([-6.75, 6.75])  # [kg/m**3/s]
BATCH_DVOL_DT = -9e-6  # [m**3/s]
# Semibatch liquid inflow: 0.95*1e-5*1125=0.0106875 kg/s.
# Net liquid volume rate: (0.0106875-0.009)/1000=1.6875e-6 m**3/s.
# Solute inventory rate: 0.0106875/2-0.009=-0.00365625 kg/s;
# solvent: 0.0106875/2=0.00534375 kg/s. Subtract C*dV then divide by V.
SEMIBATCH_DCOMP_DT = np.array([-4.078125, 4.078125])  # [kg/m**3/s]
SEMIBATCH_DVOL_DT = 1.6875e-6  # [m**3/s]
# MSMPR's existing fixed-slurry-volume closure uses liquid outlet Q*(10/11).
# Inlet species rates: (0.00534375, 0.00534375) kg/s; outlet species rates:
# (0.00227272727272727, 0.00681818181818182) kg/s. Divide net flow by
# liquid volume 0.001 and add the Batch transfer/volume-correction rates.
# This preserves the dynamic RHS contract, not a new steady-state closure;
# the separate MSMPR closure work remains in PR #380.
MSMPR_DCOMP_DT = np.array([-3.678977272727273,
                           5.275568181818182])  # [kg/m**3/s]

RATE_RTOL = 1e-12  # [-], deterministic algebra roundoff allowance
RATE_ATOL = 0.0  # [state-rate units], expected material rates are nonzero
# A binary-exact step close to 1e-6 keeps these dyadic concentration offsets
# representable. The real zero-exponent kinetic law computes s*s**-1; decimal
# offsets otherwise introduce one-ulp rate noise amplified by 1e14 raw moments.
FD_REL_STEP = 2.0**-20  # [-], central-difference truncation/roundoff compromise
JAC_RTOL = 1e-7  # [-], central-difference comparison allowance
JAC_ATOL = 1e-9  # [mixed derivative units], near-zero roundoff allowance
# 3 primary, 4 secondary, 3 growth, and 3 dissolution parameters.
NUM_KINETIC_PARAMETERS = 13
GROWTH_PREFACTOR_INDEX = 7


@pytest.fixture
def thermo_paths(tmp_path):
    """Write real-phase input data for variable- and fixed-density mixtures.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned temporary directory.

    Returns
    -------
    dict of str to str
        Database paths keyed by the liquid-density assumption.
    """
    paths = {}
    for name, densities in (
        ('variable', PURE_LIQUID_DENSITIES),
        ('fixed', (LIQUID_DENSITY, LIQUID_DENSITY)),
    ):
        properties = {}
        for species, density in zip(('solute', 'solvent'), densities):
            properties[species] = {
                # Constant synthetic thermal properties only enable real
                # slurry mixing/enthalpy APIs; no energy balance is asserted.
                'mw': 100.0,  # [g/mol], equal molecular weights by design
                'rho_liq': density,  # [kg/m**3]
                'rho_solid': CRYSTAL_DENSITY,  # [kg/m**3]
                'cp_liq': [100.0],  # [J/mol/K], constant test assumption
                'cp_solid': [1000.0],  # [J/kg/K], same mass-basis heat capacity
                # Required constructor input, not used by these liquid/solid
                # balances: log10(P/Pa)=8-1500/(T/K-40).
                'p_vap': [8.0, 1500.0, -40.0],  # [-], [K], [K], synthetic
            }
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(properties), encoding='utf-8')
        paths[name] = str(path)
    return paths


def _constant_temperature(time):
    """Return the prescribed isothermal temperature.

    Parameters
    ----------
    time : float or numpy.ndarray
        Evaluation time [s].

    Returns
    -------
    float or numpy.ndarray
        Liquid temperature [K], matching the time shape.
    """
    return TEMP + np.zeros_like(time)


def _make_crystallizer(thermo_paths, unit_type, basis, *, constant_growth=False,
                       active_growth_only=False):
    """Construct and publicly initialize the real seeded unit and inlet.

    Parameters
    ----------
    thermo_paths : dict
        Paths to the synthetic thermodynamic input files.
    unit_type : type
        BatchCryst, SemibatchCryst, or MSMPR.
    basis : {'mass_conc', 'mass_frac'}
        Kinetics input basis; ODE concentrations remain [kg/m**3].
    constant_growth : bool, optional
        Select fixed-density, zero-exponent kinetics for the analytical
        Jacobian's supported constant-property special case.
    active_growth_only : bool, optional
        Make only the growth prefactor [um/s] an active kinetic parameter.

    Returns
    -------
    unit : PharmaPy.Crystallizers._BaseCryst
        Fully constructed production crystallizer.
    states : numpy.ndarray
        Raw moments [um**n] (MSMPR: [um**n/m**3]), concentrations
        [kg/m**3], then liquid volume [m**3] for Batch/Semibatch.
    """
    path = thermo_paths['fixed' if constant_growth else 'variable']
    mask = None
    if active_growth_only:
        mask = np.zeros(NUM_KINETIC_PARAMETERS, dtype=bool)
        mask[GROWTH_PREFACTOR_INDEX] = True
    unit = unit_type('solute', method='moments', basis=basis, mask_params=mask,
                     controls={'temp': _constant_temperature})
    liquid = LiquidPhase(path, temp=TEMP, vol=LIQUID_VOL,
                         mass_frac=TANK_FRACTIONS)
    solid = SolidPhase(path, temp=TEMP, mass_frac=PURE_SOLUTE,
                       moments=TANK_MOMENTS, kv=SHAPE_FACTOR)
    unit.Phases = (liquid, solid)
    solubility = (SATURATED_MASS_FRAC if basis == 'mass_frac'
                  else SATURATED_MASS_CONC)  # [kg/kg] or [kg/m**3]
    exponent = 0.0 if constant_growth else 1.0  # [-], constant or linear law
    unit.Kinetics = CrystKinetics(
        coeff_solub=[solubility],
        growth=[GROWTH_RATE, 0.0, exponent],  # [um/s], [J/mol], [-]
    )
    if unit_type is not BatchCryst:
        inlet = SlurryStream(vol_flow=INLET_VOL_FLOW, moments=INLET_MOMENTS)
        inlet.Phases = (
            LiquidStream(path, temp=TEMP, mass_frac=INLET_FRACTIONS,
                         vol_flow=INLET_VOL_FLOW * INLET_LIQUID_FRACTION),
            SolidStream(path, temp=TEMP, mass_frac=PURE_SOLUTE, kv=SHAPE_FACTOR),
        )
        unit.Inlet = inlet
    states, _ = unit.initialize_states(runtime=PREPARATION_TIME)
    return unit, states


def _central_difference_jacobian(function, point):
    """Evaluate a central finite-difference Jacobian.

    Parameters
    ----------
    function : callable
        Map coordinates to a residual; units may differ by coordinate.
    point : numpy.ndarray
        Nonzero evaluation coordinates in their declared model units.

    Returns
    -------
    numpy.ndarray
        Jacobian with output-unit/input-unit entries by row and column.

    Raises
    ------
    ValueError
        If a coordinate is zero: this fixture uses relative perturbations.
    """
    point = np.asarray(point, dtype=float)
    base_output = np.asarray(function(point))
    jacobian = np.empty((base_output.size, point.size))  # [output/input units]
    for column, coordinate in enumerate(point):
        if coordinate == 0:
            raise ValueError('central-difference fixture coordinates must be nonzero')
        step = FD_REL_STEP * abs(coordinate)  # [same unit as this coordinate]
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[column] += step
        point_minus[column] -= step
        jacobian[:, column] = (
            np.asarray(function(point_plus)) - np.asarray(function(point_minus))
        ) / (2 * step)
    return jacobian


@pytest.mark.parametrize('basis', ['mass_frac', 'mass_conc'])
@pytest.mark.parametrize(
    'unit_type, expected_conc_rate, expected_volume_rate',
    [(BatchCryst, BATCH_DCOMP_DT, BATCH_DVOL_DT),
     (SemibatchCryst, SEMIBATCH_DCOMP_DT, SEMIBATCH_DVOL_DT),
     (MSMPR, MSMPR_DCOMP_DT, None)],
    ids=['batch', 'semibatch', 'msmpr'],
)
def test_composition_basis_preserves_concentration_rates(
        thermo_paths, unit_type, basis, expected_conc_rate, expected_volume_rate):
    """Check real input handoffs and independently calculated balance rates.

    Parameters
    ----------
    thermo_paths : dict
        Real-phase property database paths.
    unit_type : type
        Production crystallizer class.
    basis : str
        Kinetics composition basis.
    expected_conc_rate : numpy.ndarray
        Independently calculated species rates [kg/m**3/s].
    expected_volume_rate : float or None
        Expected liquid-volume rate [m**3/s], absent for MSMPR.
    """
    unit, states = _make_crystallizer(thermo_paths, unit_type, basis)
    concentration_slice = slice(unit.num_distr, unit.num_distr + unit.num_species)
    np.testing.assert_allclose(states[concentration_slice], TANK_CONC,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    expected_moments = TANK_MOMENTS_RAW.copy()  # [um**n]
    if unit_type is MSMPR:
        expected_moments = expected_moments / SLURRY_VOL  # [um**n/m**3]
    np.testing.assert_allclose(states[:unit.num_distr], expected_moments,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    assert unit.Liquid_1.getDensity() == pytest.approx(LIQUID_DENSITY)
    assert unit.Solid_1.getDensity() == pytest.approx(CRYSTAL_DENSITY)
    if unit_type is not BatchCryst:
        assert unit.Inlet.Liquid_1.getDensity() == pytest.approx(INLET_LIQUID_DENSITY)
        np.testing.assert_allclose(unit.Inlet.Liquid_1.mass_conc, INLET_CONC,
                                   rtol=RATE_RTOL, atol=RATE_ATOL)

    derivatives = unit.unit_model(0.0, states)  # [state units/s]
    assert unit.states_di['mass_conc']['units'] == 'kg/m**3'
    np.testing.assert_allclose(unit.Liquid_1.mass_conc, TANK_CONC,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    # Linear relative supersaturation makes a missing density conversion
    # observable in actual production kinetics, without recording calls.
    assert unit.Kinetics.growth == pytest.approx(GROWTH_RATE, rel=RATE_RTOL)
    np.testing.assert_allclose(derivatives[concentration_slice], expected_conc_rate,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    if expected_volume_rate is not None:
        assert derivatives[-1] == pytest.approx(expected_volume_rate, rel=RATE_RTOL)


def test_batch_unit_model_updates_real_phase_before_mass_frac_kinetics(thermo_paths):
    """Use a new ODE composition, not the phase's previous density or state.

    Parameters
    ----------
    thermo_paths : dict
        Real-phase property database paths.
    """
    unit, states = _make_crystallizer(thermo_paths, BatchCryst, 'mass_frac')
    concentration_slice = slice(unit.num_distr, unit.num_distr + unit.num_species)
    states[concentration_slice] = INLET_CONC  # [kg/m**3], new 50/50 liquid state
    derivatives = unit.unit_model(0.0, states)  # [state units/s]
    np.testing.assert_allclose(unit.Liquid_1.mass_conc, INLET_CONC,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    assert unit.Liquid_1.getDensity() == pytest.approx(INLET_LIQUID_DENSITY)
    # At w=0.5, relative supersaturation is 3: growth=6 um/s and transfer
    # triples to 0.027 kg/s. dV=-0.027/1125=-2.4e-5 m**3/s;
    # dC=(-0.027+562.5*2.4e-5, 562.5*2.4e-5)/0.001.
    expected_growth = 6.0  # [um/s], linear supersaturation prediction
    expected_concentration_rate = np.array([-13.5, 13.5])  # [kg/m**3/s]
    expected_volume_rate = -2.4e-5  # [m**3/s]
    assert unit.Kinetics.growth == pytest.approx(expected_growth, rel=RATE_RTOL)
    np.testing.assert_allclose(derivatives[concentration_slice],
                               expected_concentration_rate,
                               rtol=RATE_RTOL, atol=RATE_ATOL)
    assert derivatives[-1] == pytest.approx(expected_volume_rate, rel=RATE_RTOL)


def test_batch_mass_frac_state_jacobian_matches_finite_difference(thermo_paths):
    """Match analytic state derivatives for the constant-property special case.

    Parameters
    ----------
    thermo_paths : dict
        Real-phase property database paths.
    """
    unit, states = _make_crystallizer(
        thermo_paths, BatchCryst, 'mass_frac', constant_growth=True)

    def residual(candidate_states):
        """Evaluate the production residual.

        Parameters
        ----------
        candidate_states : numpy.ndarray
            Moments [um**n], concentrations [kg/m**3], and volume [m**3].

        Returns
        -------
        numpy.ndarray
            State derivatives [state units/s].
        """
        return unit.unit_model(0.0, candidate_states)

    finite_difference = _central_difference_jacobian(residual, states)
    residual(states)
    analytical = unit.jac_states(0.0, states, params=None, return_only=False)
    np.testing.assert_allclose(analytical, finite_difference,
                               rtol=JAC_RTOL, atol=JAC_ATOL)
    # Raw d(dV/dt)/d(mu_2) is ~1e-18: a mixed-unit absolute tolerance can
    # conceal its deletion or a missing density divisor. Since transfer is
    # linear in mu_2, its inventory-scaled sensitivity equals dV/dt itself.
    volume_response = analytical[-1, 2] * states[2]  # [m**3/s]
    assert volume_response == pytest.approx(BATCH_DVOL_DT, rel=RATE_RTOL, abs=0)


def test_batch_mass_frac_parameter_jacobian_matches_finite_difference(thermo_paths):
    """Match the real kinetics' active growth-prefactor partial derivative.

    Parameters
    ----------
    thermo_paths : dict
        Real-phase property database paths.
    """
    unit, states = _make_crystallizer(
        thermo_paths, BatchCryst, 'mass_frac', constant_growth=True,
        active_growth_only=True)
    active_params = np.array([GROWTH_RATE])  # [um/s]

    def residual(candidate_params):
        """Evaluate the production residual for an active growth prefactor.

        Parameters
        ----------
        candidate_params : numpy.ndarray
            Growth prefactor [um/s], shape (1,).

        Returns
        -------
        numpy.ndarray
            State derivatives [state units/s].
        """
        return unit.unit_model(0.0, states, params=candidate_params)

    finite_difference = _central_difference_jacobian(residual, active_params)
    residual(active_params)
    analytical = unit.jac_params(0.0, states, params=active_params)
    np.testing.assert_allclose(analytical, finite_difference,
                               rtol=JAC_RTOL, atol=JAC_ATOL)
