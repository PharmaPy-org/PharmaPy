"""Regression coverage for the ``basis='mass_frac'`` composition rescaling.

``BatchCryst.material_balances`` and ``SemibatchCryst.material_balances``
declare a ``{'mass_conc', 'mass_frac'}`` composition basis. On the
``'mass_frac'`` basis the returned liquid-composition derivatives must be
mass-fraction rates [1/s], i.e. the concentration rates [kg/m**3/s] divided by
the liquid density, exactly as :meth:`MSMPR.material_balances` already returns.
The liquid-volume derivative is not a composition and must stay [m**3/s] on
both bases.

Fixture values use the crystallizer solver's units: raw moments on the *total*
basis in [um**n], moments on a metre basis in [m**n], mass concentrations in
[kg/m**3], densities in [kg/m**3], growth rate in [um/s], volumetric flow in
[m**3/s], and volumes in [m**3]. The liquid, crystal and inlet-liquid densities
are deliberately unequal (1000, 1500 and 990 [kg/m**3]) so that dividing by the
wrong one is visible in the asserted values, and the two species carry
different concentrations so a component mix-up cannot pass.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, SemibatchCryst


pytestmark = pytest.mark.unit


LIQUID_DENSITY = 1000.0  # [kg/m**3], tank liquid density
CRYSTAL_DENSITY = 1500.0  # [kg/m**3]
INLET_LIQUID_DENSITY = 990.0  # [kg/m**3]
SHAPE_FACTOR = 0.5  # volumetric crystal shape factor [-]
GROWTH_RATE = 2.0  # [um/s]


class _Liquid:
    """Constant-density liquid phase holding the crystallizing solute."""

    def getDensity(self, temp=None):
        """Return the liquid density.

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
        """Accept the model's in-place phase update.

        Parameters
        ----------
        **kwargs
            Phase state written by the balance, such as ``mass_conc``
            [kg/m**3] and ``vol`` [m**3]. The fixture stores nothing because
            the balance reads none of it back.

        Returns
        -------
        None
        """
        return None


class _Solid:
    """Crystal phase; ``kv`` is the volumetric shape factor."""

    kv = SHAPE_FACTOR  # volumetric shape factor [-]

    def getDensity(self, temp=None):
        """Return the crystal density.

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


class _Kinetics:
    """Constant growth kinetics, so the transfer rate is hand-computable."""

    def get_kinetics(self, conc, temp, kv, moms):
        """Return constant crystallization rates.

        Parameters
        ----------
        conc : numpy.ndarray
            Liquid-phase composition on the model's basis, [kg/m**3] for
            ``'mass_conc'`` or [kg/kg] for ``'mass_frac'``.
        temp : float
            Liquid temperature [K].
        kv : float
            Crystal volumetric shape factor [-].
        moms : numpy.ndarray
            Crystal moments per unit slurry volume [m**n/m**3].

        Returns
        -------
        tuple of float
            Nucleation rate [#/s], growth rate [um/s], and dissolution rate
            [um/s], respectively.
        """
        nucleation_rate = 0.0  # [#/s]
        dissolution_rate = 0.0  # [um/s]
        return nucleation_rate, GROWTH_RATE, dissolution_rate

    def alpha_fn(self, conc):
        """Return the growth-rate impurity factor.

        Parameters
        ----------
        conc : numpy.ndarray
            Liquid-phase composition on the model's basis.

        Returns
        -------
        float
            Growth-rate impurity factor [-].
        """
        return 1.0  # [-]


# Monodisperse fixture population: 2e8 crystals of 100 [um] in the vessel, so
# the raw total moments mu_n = N * L**n are exactly consistent with each other.
TANK_MOMENTS_RAW = np.array([2.0e8, 2.0e10, 2.0e12, 2.0e14])  # [um**n]
TANK_MOMENTS = TANK_MOMENTS_RAW * (1e-6) ** np.arange(4)  # [m**n]

# Inlet moments per unit inlet volume [um**n/m**3]; they enter only the
# distribution rows, never the liquid-composition rows under test.
INLET_MOMENTS = np.array([1.0e11, 1.0e13, 1.0e15, 1.0e17])  # [um**n/m**3]

TANK_CONC = np.array([200.0, 50.0])  # [kg/m**3], liquid-phase concentrations
INLET_CONC = np.array([250.0, 60.0])  # [kg/m**3]
INLET_PHI = np.array([0.95, 0.05])  # [-], inlet [liquid, solid] volume fractions

LIQUID_VOL = 1.0e-3  # [m**3], vessel liquid volume
INLET_VOL_FLOW = 1.0e-5  # [m**3/s]
TEMP = 300.0  # [K]

# Crystallization rate on the total basis, from the fixture kinetics:
# transf = rho_crystal * kv * 3 * growth * mu_2_raw * (1e-6)**3
#        = 1500 * 0.5 * 3 * 2.0 * 2.0e12 * 1e-18 = 9.0e-3 [kg/s].
TRANSF = 9.0e-3  # [kg/s]

RATE_RTOL = 1.0e-12  # [-], deterministic algebra roundoff allowance
RATE_ATOL = 0.0  # no absolute slack for the nonzero rates


def _configure(crystallizer, basis):
    """Populate the attributes ``material_balances`` reads.

    Parameters
    ----------
    crystallizer : PharmaPy.Crystallizers._BaseCryst
        Instance created with ``__new__`` so that solver setup is bypassed
        while the real public balance method and collaborators are retained.
    basis : {'mass_conc', 'mass_frac'}
        Composition basis to configure [-].

    Returns
    -------
    PharmaPy.Crystallizers._BaseCryst
        The same instance, configured for a direct material-balance call.
    """
    crystallizer.num_distr = 4
    crystallizer.num_species = 2
    crystallizer.target_ind = 0
    crystallizer.kron_jtg = np.array([1.0, 0.0])  # [-], target-component selector
    crystallizer.basis = basis
    crystallizer.method = 'moments'
    crystallizer.scale = 1.0  # distribution scaling factor [-]
    crystallizer.rad = 1.0  # nuclei radius [um]
    crystallizer.Liquid_1 = _Liquid()
    crystallizer.Solid_1 = _Solid()
    crystallizer._Kinetics = _Kinetics()

    return crystallizer


def _batch_balance(basis):
    """Evaluate ``BatchCryst.material_balances`` on the given basis.

    Parameters
    ----------
    basis : {'mass_conc', 'mass_frac'}
        Composition basis to configure [-].

    Returns
    -------
    tuple of numpy.ndarray
        The liquid-composition derivative slice ([kg/m**3/s] or [1/s]) and the
        liquid-volume derivative [m**3/s].
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

    num_distr = crystallizer.num_distr
    num_material = num_distr + crystallizer.num_species

    return dmaterial_dt[num_distr:num_material], dmaterial_dt[num_material]


def _semibatch_balance(basis):
    """Evaluate ``SemibatchCryst.material_balances`` on the given basis.

    Parameters
    ----------
    basis : {'mass_conc', 'mass_frac'}
        Composition basis to configure [-].

    Returns
    -------
    tuple of numpy.ndarray
        The liquid-composition derivative slice ([kg/m**3/s] or [1/s]) and the
        liquid-volume derivative [m**3/s].
    """
    crystallizer = _configure(SemibatchCryst.__new__(SemibatchCryst), basis)

    # Mixed units by field: volumetric flow [m**3/s], inlet moments
    # [um**n/m**3], and inlet liquid-phase mass concentrations [kg/m**3].
    u_inputs = {
        'Inlet': {'vol_flow': INLET_VOL_FLOW, 'distrib': INLET_MOMENTS},
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

    num_distr = crystallizer.num_distr
    num_material = num_distr + crystallizer.num_species

    return dmaterial_dt[num_distr:num_material], dmaterial_dt[num_material]


# Hand-computed batch liquid-phase balance, all terms in [kg/m**3/s]:
#   dcomp_dt = -transf / vol_liq * (kron_jtg - mass_conc / rho_liq)
#            = -9.0e-3 / 1.0e-3 * ((1, 0) - (200, 50) / 1000)
#            = -9.0 * (0.8, -0.05) = (-7.2, 0.45)
BATCH_DCOMP_DT_MASS_CONC = np.array([-7.2, 0.45])  # [kg/m**3/s]
BATCH_DVOL_DT = -TRANSF / LIQUID_DENSITY  # [m**3/s], -9.0e-6

# Hand-computed semibatch liquid-phase balance:
#   flow_term   = phi_in_liq * flow * (conc_in - conc_tank * rho_in / rho_liq)
#               = 0.95 * 1.0e-5 * ((250, 60) - (200, 50) * 0.99)
#               = 9.5e-6 * (52.0, 10.5) = (4.94e-4, 9.975e-5) [kg/s]
#   transf_term = transf * (kron_jtg - conc_tank / rho_liq)
#               = 9.0e-3 * (0.8, -0.05) = (7.2e-3, -4.5e-4) [kg/s]
#   dcomp_dt    = (flow_term - transf_term) / vol_liq
#               = (-6.706e-3, 5.4975e-4) / 1.0e-3
SEMIBATCH_DCOMP_DT_MASS_CONC = np.array([-6.706, 0.54975])  # [kg/m**3/s]
#   dvol_dt = (phi_in_liq * flow * rho_in - transf) / rho_liq
#           = (9.405e-3 - 9.0e-3) / 1000
SEMIBATCH_DVOL_DT = 4.05e-7  # [m**3/s]


def test_batch_mass_frac_basis_rescales_returned_composition_rates():
    """Return batch composition rates as [1/s] when basis is 'mass_frac'."""
    dcomp_dt, dvol_dt = _batch_balance('mass_frac')

    expected_dcomp_dt = BATCH_DCOMP_DT_MASS_CONC / LIQUID_DENSITY  # [1/s]
    np.testing.assert_allclose(
        dcomp_dt, expected_dcomp_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )

    # The liquid volume is not a composition, so its rate keeps [m**3/s].
    np.testing.assert_allclose(
        dvol_dt, BATCH_DVOL_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


def test_semibatch_mass_frac_basis_rescales_returned_composition_rates():
    """Return semibatch composition rates as [1/s] when basis is 'mass_frac'."""
    dcomp_dt, dvol_dt = _semibatch_balance('mass_frac')

    expected_dcomp_dt = SEMIBATCH_DCOMP_DT_MASS_CONC / LIQUID_DENSITY  # [1/s]
    np.testing.assert_allclose(
        dcomp_dt, expected_dcomp_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )

    # The liquid volume is not a composition, so its rate keeps [m**3/s].
    np.testing.assert_allclose(
        dvol_dt, SEMIBATCH_DVOL_DT, rtol=RATE_RTOL, atol=RATE_ATOL
    )


@pytest.mark.parametrize(
    'evaluate, expected_dcomp_dt, expected_dvol_dt',
    [
        (_batch_balance, BATCH_DCOMP_DT_MASS_CONC, BATCH_DVOL_DT),
        (_semibatch_balance, SEMIBATCH_DCOMP_DT_MASS_CONC, SEMIBATCH_DVOL_DT),
    ],
    ids=['batch', 'semibatch'],
)
def test_mass_conc_basis_is_unchanged(evaluate, expected_dcomp_dt,
                                      expected_dvol_dt):
    """Keep the default 'mass_conc' derivatives in [kg/m**3/s] and [m**3/s]."""
    dcomp_dt, dvol_dt = evaluate('mass_conc')

    np.testing.assert_allclose(
        dcomp_dt, expected_dcomp_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )
    np.testing.assert_allclose(
        dvol_dt, expected_dvol_dt, rtol=RATE_RTOL, atol=RATE_ATOL
    )
