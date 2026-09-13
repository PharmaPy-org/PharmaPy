"""Regression coverage for crystallizer composition-basis boundaries.

Crystallizer ODE states always store liquid mass concentrations [kg/m**3].
The ``basis='mass_frac'`` option selects mass fractions [kg/kg] only at the
kinetics-input boundary; it does not change the state or residual basis.
Batch, semibatch, and MSMPR fixtures use unequal component concentrations and
unequal liquid, crystal, and inlet densities so a misplaced density conversion
cannot pass accidentally.

Fixture values use the crystallizer solver's units: raw moments in [um**n],
metre-basis moments in [m**n], mass concentrations in [kg/m**3], densities in
[kg/m**3], growth rates in [um/s], volumetric flow in [m**3/s], and volumes in
[m**3]. The analytical-Jacobian fixtures use constant growth with nonzero,
asymmetric states so both state and active-parameter derivatives are
finite-difference testable without coupling the expected values to the
production Jacobian expressions.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst


pytestmark = pytest.mark.unit


LIQUID_DENSITY = 1000.0  # [kg/m**3], tank liquid density
CRYSTAL_DENSITY = 1500.0  # [kg/m**3]
INLET_LIQUID_DENSITY = 990.0  # [kg/m**3]
SHAPE_FACTOR = 0.5  # [-], volumetric crystal shape factor
GROWTH_RATE = 2.0  # [um/s]
SATURATED_MASS_CONC = 100.0  # [kg/m**3], analytical-Jacobian solubility

# Monodisperse population: 2e8 crystals of 100 [um], so raw moments
# mu_n = N * L**n are mutually consistent on the total basis.
TANK_MOMENTS_RAW = np.array([2.0e8, 2.0e10, 2.0e12, 2.0e14])  # [um**n]
TANK_MOMENTS = TANK_MOMENTS_RAW * (1e-6) ** np.arange(4)  # [m**n]

# Inlet moments are first stated in the raw MSMPR units and converted to the
# metre basis expected by the upstream ``mu_n`` input.
INLET_MOMENTS_RAW = np.array([1.0e11, 1.0e13, 1.0e15,
                              1.0e17])  # [um**n/m**3]
INLET_MOMENTS = INLET_MOMENTS_RAW * (1e-6) ** np.arange(4)  # [m**n/m**3]

TANK_CONC = np.array([200.0, 50.0])  # [kg/m**3]
INLET_CONC = np.array([250.0, 60.0])  # [kg/m**3]
INLET_PHI = np.array([0.95, 0.05])  # [-], inlet liquid/solid volume fractions

LIQUID_VOL = 1.0e-3  # [m**3]
INLET_VOL_FLOW = 1.0e-5  # [m**3/s]
TEMP = 300.0  # [K]

# Total-basis transfer from the fixture kinetics:
# 1500 * 0.5 * 3 * 2.0 * 2.0e12 * (1e-6)**3 = 9.0e-3 [kg/s].
TRANSF = 9.0e-3  # [kg/s], or [kg/m**3/s] for the MSMPR intensive state

RATE_RTOL = 1.0e-12  # [-], deterministic algebra roundoff allowance
RATE_ATOL = 0.0  # [state-rate units], all expected rates are nonzero
# A 1e-6 relative central-difference step balances truncation and roundoff for
# the smooth polynomial fixture over the six-orders-of-magnitude state vector.
FD_REL_STEP = 1.0e-6  # [-]
JAC_RTOL = 1.0e-7  # [-], central-difference comparison allowance
JAC_ATOL = 1.0e-9  # [mixed derivative units], near-zero roundoff allowance

# ``CrystKinetics`` exposes 3 primary-nucleation, 4 secondary-nucleation,
# 3 growth, and 3 dissolution parameters. The growth prefactor is first in
# the growth block.
NUM_KINETIC_PARAMETERS = 13
GROWTH_PREFACTOR_INDEX = 7


class _Liquid:
    """Constant-density liquid phase carrying mass-concentration state."""

    def __init__(self):
        """Initialize the liquid state used by ``solve_unit`` and tests."""
        self.mass_conc = TANK_CONC.copy()  # [kg/m**3]
        self.temp = TEMP  # [K]
        self.last_mass_conc = None  # [kg/m**3]

    def getDensity(self, temp=None):
        """Return the constant liquid density.

        Parameters
        ----------
        temp : float or None, optional
            Liquid temperature [K]. The constant-density fixture ignores it.

        Returns
        -------
        float
            Liquid density [kg/m**3].
        """
        return LIQUID_DENSITY

    def updatePhase(self, **kwargs):
        """Store phase state written by the crystallizer.

        Parameters
        ----------
        **kwargs
            Phase values including ``mass_conc`` [kg/m**3] and ``vol``
            [m**3].

        Returns
        -------
        None
        """
        if 'mass_conc' in kwargs:
            self.last_mass_conc = np.asarray(kwargs['mass_conc']).copy()

        for name, value in kwargs.items():
            setattr(self, name, value)


class _Solid:
    """Constant-density crystal phase with a fixed volume shape factor."""

    kv = SHAPE_FACTOR  # [-], volumetric crystal shape factor

    def __init__(self):
        """Initialize the crystal temperature."""
        self.temp = TEMP  # [K]

    def getDensity(self, temp=None):
        """Return the constant crystal density.

        Parameters
        ----------
        temp : float or None, optional
            Crystal temperature [K]. The constant-density fixture ignores it.

        Returns
        -------
        float
            Crystal density [kg/m**3].
        """
        return CRYSTAL_DENSITY


class _Slurry:
    """Minimal batch slurry providing temperature and phase densities."""

    def __init__(self):
        """Initialize uncontrolled fallback state values."""
        self.temp = TEMP  # [K]
        self.temp_ht = None  # [K]

    def getDensity(self, temp=None):
        """Return liquid and crystal densities.

        Parameters
        ----------
        temp : float or None, optional
            Slurry temperature [K]. The constant-density fixture ignores it.

        Returns
        -------
        numpy.ndarray
            Liquid and crystal densities [kg/m**3], in phase order.
        """
        return np.array([LIQUID_DENSITY, CRYSTAL_DENSITY])  # [kg/m**3]


class _Kinetics:
    """Constant-growth kinetics with an independently variable prefactor."""

    def __init__(self):
        """Initialize rates and analytical-Jacobian parameter metadata."""
        self.growth_parameter = GROWTH_RATE  # [um/s]
        self.last_conc = None  # [kg/m**3] or [kg/kg], selected by ``basis``
        self.prim_nucl = 0.0  # [#/m**3/s]
        self.sec_nucl = 0.0  # [#/m**3/s]
        self.growth = GROWTH_RATE  # [um/s]
        self.dissol = 0.0  # [um/s]
        self.params = {
            'nucl_prim': [0.0, 0.0, 0.0],
            'nucl_sec': [0.0, 0.0, 0.0, 0.0],
            'growth': [GROWTH_RATE, 0.0, 0.0],
            'dissolution': [0.0, 0.0, 0.0],
        }  # [rate prefactor, J/mol, exponent [-]]; secondary adds exponent [-]

    def set_params(self, params):
        """Set the active growth prefactor.

        Parameters
        ----------
        params : array-like
            One active growth-rate prefactor [um/s].

        Returns
        -------
        None
        """
        self.growth_parameter = np.asarray(params)[0]  # [um/s]

    def get_kinetics(self, conc, temp, kv, moms):
        """Return constant nucleation, growth, and dissolution rates.

        Parameters
        ----------
        conc : numpy.ndarray
            Kinetics composition input: [kg/m**3] for ``'mass_conc'`` or
            [kg/kg] for ``'mass_frac'``.
        temp : float
            Liquid temperature [K].
        kv : float
            Crystal volumetric shape factor [-].
        moms : numpy.ndarray
            Crystal moments per unit slurry volume [m**n/m**3].

        Returns
        -------
        tuple of float
            Nucleation rate [#/m**3/s], growth rate [um/s], and dissolution
            rate [um/s].
        """
        self.last_conc = np.asarray(conc).copy()
        self.prim_nucl = 0.0  # [#/m**3/s]
        self.sec_nucl = 0.0  # [#/m**3/s]
        self.growth = self.growth_parameter  # [um/s]
        self.dissol = 0.0  # [um/s]
        return self.prim_nucl + self.sec_nucl, self.growth, self.dissol

    def get_solubility(self, temp, conc):
        """Return the fixed kinetic-basis solubility.

        Parameters
        ----------
        temp : float
            Liquid temperature [K].
        conc : numpy.ndarray
            Composition supplied by the analytical Jacobian [kg/m**3].

        Returns
        -------
        float
            Saturated mass concentration used by the analytical Jacobian
            [kg/m**3].
        """
        return SATURATED_MASS_CONC

    def deriv_cryst(self, conc_tg, conc, temp):
        """Return mechanism derivatives with respect to kinetic parameters.

        Parameters
        ----------
        conc_tg : float
            Target-component mass concentration [kg/m**3].
        conc : numpy.ndarray
            Liquid mass concentrations [kg/m**3].
        temp : float
            Liquid temperature [K].

        Returns
        -------
        tuple of numpy.ndarray and float
            Primary-nucleation, secondary-nucleation, growth, and dissolution
            parameter derivatives followed by solubility [kg/m**3]. The
            active growth-prefactor derivative is dimensionless; inactive
            entries are zero on their respective parameter bases.
        """
        inactive_derivatives = np.zeros(3)  # [mechanism rate/parameter unit]
        growth_derivatives = np.array([1.0, 0.0, 0.0])  # [-] then inactive
        return (inactive_derivatives.copy(), inactive_derivatives.copy(),
                growth_derivatives, inactive_derivatives.copy(),
                SATURATED_MASS_CONC)

    def alpha_fn(self, conc):
        """Return the constant growth-inhibition factor.

        Parameters
        ----------
        conc : numpy.ndarray
            Liquid mass concentrations [kg/m**3].

        Returns
        -------
        float
            Growth-rate multiplier [-].
        """
        return 1.0  # [-]


def _constant_temperature(time):
    """Return the controlled liquid temperature.

    Parameters
    ----------
    time : float
        Evaluation time [s].

    Returns
    -------
    float
        Liquid temperature [K].
    """
    return TEMP


def _configure(crystallizer, basis):
    """Populate attributes read by the real material-balance methods.

    Parameters
    ----------
    crystallizer : PharmaPy.Crystallizers._BaseCryst
        Instance created with ``__new__`` to bypass solver construction while
        retaining the production balance methods.
    basis : {'mass_conc', 'mass_frac'}
        Kinetics-input composition basis [-].

    Returns
    -------
    PharmaPy.Crystallizers._BaseCryst
        Configured instance.
    """
    crystallizer.num_distr = 4
    crystallizer.num_species = 2
    crystallizer.target_ind = 0
    crystallizer.kron_jtg = np.array([1.0, 0.0])  # [-], target selector
    crystallizer.basis = basis
    crystallizer.method = 'moments'
    crystallizer.scale = 1.0  # [-], distribution scaling factor
    crystallizer.rad = 1.0  # [um], nuclei radius
    crystallizer.Liquid_1 = _Liquid()
    crystallizer.Solid_1 = _Solid()
    crystallizer._Kinetics = _Kinetics()

    return crystallizer


def _configure_unit_model(crystallizer, basis):
    """Configure the production ``unit_model`` and state nomenclature.

    Parameters
    ----------
    crystallizer : PharmaPy.Crystallizers.BatchCryst
        Batch crystallizer allocated with ``__new__``.
    basis : {'mass_conc', 'mass_frac'}
        Kinetics-input composition basis [-].

    Returns
    -------
    PharmaPy.Crystallizers.BatchCryst
        Instance ready for a real ``unit_model`` call.
    """
    crystallizer = _configure(crystallizer, basis)
    crystallizer.controls = {
        'temp': {'fun': _constant_temperature, 'args': (), 'kwargs': {}}
    }
    crystallizer.adiabatic = False
    crystallizer.name_species = ['solute', 'impurity']
    crystallizer.states_uo = ['mass_conc']
    crystallizer.names_states_in = ['mass_conc']
    crystallizer.Slurry = _Slurry()
    crystallizer.nomenclature()

    return crystallizer


def _batch_states():
    """Return the Batch solver state vector.

    Returns
    -------
    numpy.ndarray
        Raw moments [um**n], mass concentrations [kg/m**3], then liquid
        volume [m**3].
    """
    return np.concatenate((TANK_MOMENTS_RAW, TANK_CONC, [LIQUID_VOL]))


def _batch_balance(basis):
    """Evaluate the Batch material balance on the selected kinetics basis.

    Parameters
    ----------
    basis : {'mass_conc', 'mass_frac'}
        Kinetics-input composition basis [-].

    Returns
    -------
    tuple of numpy.ndarray and float
        Composition derivatives [kg/m**3/s] and liquid-volume derivative
        [m**3/s].
    """
    crystallizer = _configure(BatchCryst.__new__(BatchCryst), basis)

    dmaterial_dt, _ = crystallizer.material_balances(
        time=0.0,
        params=None,
        u_inputs={},
        rhos=[LIQUID_DENSITY, CRYSTAL_DENSITY],
        mu_n=TANK_MOMENTS,
        distrib=TANK_MOMENTS_RAW,
        mass_conc=TANK_CONC,
        temp=TEMP,
        temp_ht=None,
        vol=LIQUID_VOL,
    )

    num_material = crystallizer.num_distr + crystallizer.num_species
    return (dmaterial_dt[crystallizer.num_distr:num_material],
            dmaterial_dt[num_material])


def _semibatch_balance(basis):
    """Evaluate the Semibatch balance on the selected kinetics basis.

    Parameters
    ----------
    basis : {'mass_conc', 'mass_frac'}
        Kinetics-input composition basis [-].

    Returns
    -------
    tuple of numpy.ndarray and float
        Composition derivatives [kg/m**3/s] and liquid-volume derivative
        [m**3/s].
    """
    crystallizer = _configure(SemibatchCryst.__new__(SemibatchCryst), basis)

    # Mixed units by field: flow [m**3/s], moments [um**n/m**3], and inlet
    # concentrations [kg/m**3].
    u_inputs = {
        'Inlet': {'vol_flow': INLET_VOL_FLOW, 'distrib': INLET_MOMENTS_RAW},
        'Liquid_1': {'mass_conc': INLET_CONC},
    }

    dmaterial_dt, _ = crystallizer.material_balances(
        time=0.0,
        params=None,
        u_inputs=u_inputs,
        rhos=[
            [LIQUID_DENSITY, CRYSTAL_DENSITY],
            [INLET_LIQUID_DENSITY, CRYSTAL_DENSITY],
        ],
        mu_n=TANK_MOMENTS,
        distrib=TANK_MOMENTS_RAW,
        mass_conc=TANK_CONC,
        temp=TEMP,
        temp_ht=None,
        vol=LIQUID_VOL,
        phi_in=INLET_PHI,
    )

    num_material = crystallizer.num_distr + crystallizer.num_species
    return (dmaterial_dt[crystallizer.num_distr:num_material],
            dmaterial_dt[num_material])


def _msmpr_balance(basis):
    """Evaluate the MSMPR balance on the selected kinetics basis.

    Parameters
    ----------
    basis : {'mass_conc', 'mass_frac'}
        Kinetics-input composition basis [-].

    Returns
    -------
    numpy.ndarray
        Liquid-composition derivatives [kg/m**3/s].
    """
    crystallizer = _configure(MSMPR.__new__(MSMPR), basis)

    # Mixed units by field: flow [m**3/s], moments [m**n/m**3], and inlet
    # concentrations [kg/m**3].
    u_inputs = {
        'Inlet': {'vol_flow': INLET_VOL_FLOW, 'mu_n': INLET_MOMENTS},
        'Liquid_1': {'mass_conc': INLET_CONC},
    }

    dmaterial_dt, _ = crystallizer.material_balances(
        time=0.0,
        params=None,
        u_inputs=u_inputs,
        rhos=[
            [LIQUID_DENSITY, CRYSTAL_DENSITY],
            [INLET_LIQUID_DENSITY, CRYSTAL_DENSITY],
        ],
        mu_n=TANK_MOMENTS,
        distrib=TANK_MOMENTS_RAW,
        mass_conc=TANK_CONC,
        temp=TEMP,
        temp_ht=None,
        vol=LIQUID_VOL,
        phi_in=INLET_PHI,
    )

    num_material = crystallizer.num_distr + crystallizer.num_species
    return dmaterial_dt[crystallizer.num_distr:num_material]


def _central_difference_jacobian(function, point):
    """Evaluate a central finite-difference Jacobian.

    Parameters
    ----------
    function : callable
        Function mapping ``point`` to a one-dimensional residual. Input and
        output units may differ by coordinate.
    point : numpy.ndarray
        Nonzero evaluation coordinates in their declared model units.

    Returns
    -------
    numpy.ndarray
        Jacobian with output-unit/input-unit entries by row and column.

    Raises
    ------
    ValueError
        If a coordinate is zero, because this fixture deliberately uses a
        purely relative perturbation.
    """
    point = np.asarray(point, dtype=float)
    base_output = np.asarray(function(point))
    jacobian = np.empty((base_output.size, point.size))  # mixed derivative units

    for column, coordinate in enumerate(point):
        if coordinate == 0:
            raise ValueError('central-difference fixture coordinates must be nonzero')

        step = FD_REL_STEP * abs(coordinate)  # same unit as this coordinate
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[column] += step
        point_minus[column] -= step
        jacobian[:, column] = (
            np.asarray(function(point_plus))
            - np.asarray(function(point_minus))
        ) / (2 * step)

    return jacobian


# Hand-computed Batch liquid balance [kg/m**3/s]:
# -9.0e-3 / 1.0e-3 * ((1, 0) - (200, 50) / 1000) = (-7.2, 0.45).
BATCH_DCOMP_DT = np.array([-7.2, 0.45])  # [kg/m**3/s]
BATCH_DVOL_DT = -TRANSF / LIQUID_DENSITY  # [m**3/s]

# Hand-computed Semibatch liquid balance [kg/m**3/s]:
# flow = 0.95 * 1e-5 * ((250, 60) - (200, 50) * 0.99)
#      = (4.94e-4, 9.975e-5) [kg/s]
# transfer = 9e-3 * (0.8, -0.05) = (7.2e-3, -4.5e-4) [kg/s].
SEMIBATCH_DCOMP_DT = np.array([-6.706, 0.54975])  # [kg/m**3/s]
SEMIBATCH_DVOL_DT = 4.05e-7  # [m**3/s]

# Hand-computed MSMPR balance with phi = 1 - 0.5 * 2e-4 = 0.9999 [-]:
# flow = 0.01 * ((250, 60) * 0.95 - (200, 50) * 0.9999)
#      = (0.3752, 0.07005) [kg/m**3/s]
# transfer = (0.0072, -0.00045) [kg/m**3/s], then divide by phi.
MSMPR_DCOMP_DT = np.array([0.368036803680368,
                           0.0705070507050705])  # [kg/m**3/s]


def test_batch_mass_frac_basis_keeps_concentration_state_rates():
    """Keep Batch composition residuals in [kg/m**3/s]."""
    dcomp_dt, dvol_dt = _batch_balance('mass_frac')

    np.testing.assert_allclose(
        dcomp_dt, BATCH_DCOMP_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )
    np.testing.assert_allclose(
        dvol_dt, BATCH_DVOL_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


def test_semibatch_mass_frac_basis_keeps_concentration_state_rates():
    """Keep Semibatch composition residuals in [kg/m**3/s]."""
    dcomp_dt, dvol_dt = _semibatch_balance('mass_frac')

    np.testing.assert_allclose(
        dcomp_dt, SEMIBATCH_DCOMP_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )
    np.testing.assert_allclose(
        dvol_dt, SEMIBATCH_DVOL_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


def test_msmpr_mass_frac_basis_keeps_concentration_state_rates():
    """Keep MSMPR composition residuals in [kg/m**3/s]."""
    dcomp_dt = _msmpr_balance('mass_frac')

    np.testing.assert_allclose(
        dcomp_dt, MSMPR_DCOMP_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


@pytest.mark.parametrize(
    'evaluate, expected_dcomp_dt, expected_dvol_dt',
    [
        (_batch_balance, BATCH_DCOMP_DT, BATCH_DVOL_DT),
        (_semibatch_balance, SEMIBATCH_DCOMP_DT, SEMIBATCH_DVOL_DT),
    ],
    ids=['batch', 'semibatch'],
)
def test_mass_conc_basis_is_unchanged(evaluate, expected_dcomp_dt,
                                      expected_dvol_dt):
    """Keep default Batch and Semibatch residual units unchanged."""
    dcomp_dt, dvol_dt = evaluate('mass_conc')

    np.testing.assert_allclose(
        dcomp_dt, expected_dcomp_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )
    np.testing.assert_allclose(
        dvol_dt, expected_dvol_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )


def test_msmpr_mass_conc_basis_is_unchanged():
    """Keep the default MSMPR residual in [kg/m**3/s]."""
    dcomp_dt = _msmpr_balance('mass_conc')

    np.testing.assert_allclose(
        dcomp_dt, MSMPR_DCOMP_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


def test_batch_unit_model_keeps_mass_conc_state_at_mass_frac_boundary():
    """Tie solver metadata, phase handoff, kinetics input, and residual basis."""
    crystallizer = _configure_unit_model(
        BatchCryst.__new__(BatchCryst), 'mass_frac'
    )
    states = _batch_states()  # raw [um**n], [kg/m**3], [m**3]
    initial_mass_conc = crystallizer.Liquid_1.mass_conc.copy()  # [kg/m**3]

    derivatives = crystallizer.unit_model(0.0, states)  # mixed state-rate units
    concentration_slice = slice(
        crystallizer.num_distr,
        crystallizer.num_distr + crystallizer.num_species,
    )

    assert crystallizer.states_di['mass_conc']['units'] == 'kg/m**3'
    np.testing.assert_allclose(states[concentration_slice], initial_mass_conc)
    np.testing.assert_allclose(
        crystallizer.Liquid_1.last_mass_conc, initial_mass_conc
    )
    np.testing.assert_allclose(
        crystallizer.Kinetics.last_conc,
        initial_mass_conc / LIQUID_DENSITY,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )
    np.testing.assert_allclose(
        derivatives[concentration_slice],
        BATCH_DCOMP_DT,
        rtol=RATE_RTOL,
        atol=RATE_ATOL,
    )


def test_batch_mass_frac_state_jacobian_matches_finite_difference():
    """Match the analytical state Jacobian to the mass-fraction input path."""
    crystallizer = _configure_unit_model(
        BatchCryst.__new__(BatchCryst), 'mass_frac'
    )
    states = _batch_states()  # raw [um**n], [kg/m**3], [m**3]

    def residual(candidate_states):
        """Evaluate the production residual at mixed-unit solver states."""
        return crystallizer.unit_model(0.0, candidate_states)

    finite_difference = _central_difference_jacobian(residual, states)
    residual(states)  # refresh cached kinetic rates at the exact state
    analytical = crystallizer.jac_states(
        time=0.0, states=states, params=None, return_only=False
    )

    np.testing.assert_allclose(
        analytical, finite_difference, rtol=JAC_RTOL, atol=JAC_ATOL
    )


def test_batch_mass_frac_parameter_jacobian_matches_finite_difference():
    """Match the active analytical growth derivative to finite differences."""
    crystallizer = _configure_unit_model(
        BatchCryst.__new__(BatchCryst), 'mass_frac'
    )
    active_mask = np.zeros(NUM_KINETIC_PARAMETERS, dtype=bool)
    active_mask[GROWTH_PREFACTOR_INDEX] = True
    crystallizer.mask_params = active_mask

    states = _batch_states()  # raw [um**n], [kg/m**3], [m**3]
    active_params = np.array([GROWTH_RATE])  # [um/s]

    def residual(candidate_params):
        """Evaluate the residual after setting the growth prefactor [um/s]."""
        crystallizer.Kinetics.set_params(candidate_params)
        return crystallizer.unit_model(0.0, states, params=candidate_params)

    finite_difference = _central_difference_jacobian(residual, active_params)
    residual(active_params)  # restore the exact active growth prefactor
    analytical = crystallizer.jac_params(
        time=0.0, states=states, params=active_params
    )

    np.testing.assert_allclose(
        analytical, finite_difference, rtol=JAC_RTOL, atol=JAC_ATOL
    )
