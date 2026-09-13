"""Particle-size unit regressions for deliquoring and drying setup.

``SolidPhase.x_distrib`` stores crystal sizes in micrometers, while the
capillary and threshold-pressure correlations consume particle diameters in
meters. These tests stop both units at the solver boundary so the assertions
exercise the real setup calculations without running an Assimulo transient.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.SolidLiquidSep import DeliquoringStep
from PharmaPy import SolidLiquidSep as solid_liquid_sep
from PharmaPy import Drying_Model as drying_model


pytestmark = pytest.mark.unit


SIZE_GRID_UM = np.array([50.0, 100.0, 150.0])  # [um]
CSD_NUMBER = np.array([1.0, 2.0, 1.0])  # [#/m**3/um]
MICRONIZED_SIZE_GRID_UM = np.linspace(0.5, 1.5, 5)  # [um]
MICRONIZED_CSD_NUMBER = np.ones_like(MICRONIZED_SIZE_GRID_UM)  # [#/m**3/um]
POROSITY = 0.45  # [-]
SURFACE_TENSION = np.array([0.065, 0.067])  # [N/m]


class StopAfterSetup(Exception):
    """Stop a setup-only solve before time integration begins."""


class DummyExplicitProblem:
    """Minimal ``Explicit_Problem`` replacement for setup-only tests."""

    def __init__(self, rhs, y0, t0, **kwargs):
        """Store solver problem inputs without constructing Assimulo objects.

        Parameters
        ----------
        rhs : callable
            Right-hand-side function that would be integrated [-].
        y0 : ndarray
            Initial flattened state passed to the solver [-].
        t0 : float
            Initial integration time [-].
        **kwargs : dict
            Additional problem options, such as event switches [-].
        """
        self.rhs = rhs
        self.y0 = y0
        self.t0 = t0
        self.kwargs = kwargs
        self.name = None


class StopSolver:
    """Minimal ``CVode`` replacement that stops at ``simulate``."""

    def __init__(self, problem):
        """Store the explicit problem supplied by ``solve_unit``.

        Parameters
        ----------
        problem : DummyExplicitProblem
            Problem object constructed after setup calculations complete [-].
        """
        self.problem = problem
        self.linear_solver = None
        self.verbosity = None

    def simulate(self, *args, **kwargs):
        """Stop before numerical integration.

        Parameters
        ----------
        *args : tuple
            Positional solver arguments that would include final time [-].
        **kwargs : dict
            Keyword solver arguments [-].

        Raises
        ------
        StopAfterSetup
            Always raised to keep the test focused on setup calculations.
        """
        raise StopAfterSetup


def _zeroth_moment(
    size_grid_um=SIZE_GRID_UM,
    csd_number=CSD_NUMBER,
) -> float:
    """Return the number-based CSD zeroth moment.

    Parameters
    ----------
    size_grid_um : ndarray, optional
        Particle-size grid stored by ``SolidPhase`` [um].
    csd_number : ndarray, optional
        Crystal-size distribution on ``size_grid_um`` [#/m**3/um].

    Returns
    -------
    float
        Integral of ``csd_number`` [#/m**3/um] over ``size_grid_um`` [um],
        giving the total crystal count per suspension volume [#/m**3].
    """
    return solid_liquid_sep.trapezoidal_rule(size_grid_um, csd_number)


def _make_deliquoring_unit(
    size_grid_um=SIZE_GRID_UM,
    csd_number=CSD_NUMBER,
    saturation=0.8,
):
    """Build a minimal deliquoring unit for setup-path tests.

    Parameters
    ----------
    size_grid_um : ndarray, optional
        Particle-size grid stored by ``SolidPhase`` [um].
    csd_number : ndarray, optional
        Crystal-size distribution on ``size_grid_um`` [#/m**3/um].
    saturation : float, optional
        Initial cake saturation [-].

    Returns
    -------
    DeliquoringStep
        Unit with enough phase attributes to execute ``solve_unit`` setup.
    """
    size_grid_um = np.asarray(size_grid_um)  # [um]
    csd_number = np.asarray(csd_number)  # [#/m**3/um]

    unit = DeliquoringStep(num_nodes=2)
    unit.cake_height = 0.02  # [m]
    unit.z_centers = np.array([0.25, 0.75])  # [-]
    unit.delta_z = np.array([0.5, 0.5])  # [-]
    unit.Solid_1 = SimpleNamespace(
        distrib=csd_number,
        x_distrib=size_grid_um,
        moments=np.array([_zeroth_moment(size_grid_um, csd_number)]),  # [#/m**3]
        getPorosity=lambda: POROSITY,
        getDensity=lambda: 1500.0,  # [kg/m**3]
    )
    unit.Liquid_1 = SimpleNamespace(
        num_species=2,
        name_species=["solute_a", "solute_b"],
        getDensity=lambda: np.array([950.0, 970.0]),  # [kg/m**3]
        getViscosity=lambda: np.array([1.0e-3, 1.2e-3]),  # [Pa*s]
        getSurfTension=lambda: SURFACE_TENSION,
        getDensityPure=lambda: np.array([1000.0, 1100.0]),  # [kg/m**3]
    )
    unit.CakePhase = SimpleNamespace(
        alpha=2.0e10,  # [m/kg]
        saturation=np.array([saturation]),  # [-]
        Liquid_1=SimpleNamespace(mass_conc=np.array([1.0, 2.0])),  # [kg/m**3]
        z_external=np.array([0.0, unit.cake_height]),  # [m]
    )

    return unit


def test_deliquoring_setup_converts_micrometer_grid_to_meter_diameters(
    monkeypatch,
):
    """Deliquoring capillary quantities use meter diameters from ``x_distrib``."""
    captured = {}  # [-]
    size_grid_m = SIZE_GRID_UM * 1e-6  # [m]

    def capture_sat_inf(x_vec, csd, deltaP, porosity, height, mu_zero, props):
        """Capture the diameter grid supplied to ``get_sat_inf``.

        Parameters
        ----------
        x_vec : ndarray
            Particle diameter grid supplied to the capillary correlation [m].
        csd : ndarray
            Crystal-size distribution on the stored grid [#/m**3/um].
        deltaP : float
            Pressure drop supplied to the saturation correlation [Pa].
        porosity : float
            Cake porosity [-].
        height : float
            Cake height [m].
        mu_zero : float
            Zeroth distribution moment [#/m**3].
        props : tuple
            Surface tension [N/m] and liquid density [kg/m**3].

        Returns
        -------
        float
            Synthetic irreducible saturation [-].
        """
        captured["x_vec"] = x_vec.copy()  # [m]
        return 0.25  # [-]

    monkeypatch.setattr(solid_liquid_sep, "get_sat_inf", capture_sat_inf)
    monkeypatch.setattr(solid_liquid_sep, "Explicit_Problem", DummyExplicitProblem)
    monkeypatch.setattr(solid_liquid_sep, "CVode", StopSolver)

    unit = _make_deliquoring_unit()

    with pytest.raises(StopAfterSetup):
        unit.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]

    np.testing.assert_allclose(captured["x_vec"], size_grid_m)


def test_deliquoring_threshold_pressure_uses_expected_size_basis(monkeypatch):
    """Threshold pressure uses meter diameters and micrometer CSD weights."""

    def fixed_sat_inf(x_vec, csd, deltaP, porosity, height, mu_zero, props):
        """Return a safe saturation while the threshold pressure is tested.

        Parameters
        ----------
        x_vec : ndarray
            Particle diameter grid supplied to the capillary correlation [m].
        csd : ndarray
            Crystal-size distribution on the stored grid [#/m**3/um].
        deltaP : float
            Pressure drop supplied to the saturation correlation [Pa].
        porosity : float
            Cake porosity [-].
        height : float
            Cake height [m].
        mu_zero : float
            Zeroth distribution moment [#/m**3].
        props : tuple
            Surface tension [N/m] and liquid density [kg/m**3].

        Returns
        -------
        float
            Synthetic irreducible saturation [-].
        """
        return 0.25  # [-]

    monkeypatch.setattr(solid_liquid_sep, "get_sat_inf", fixed_sat_inf)
    monkeypatch.setattr(solid_liquid_sep, "Explicit_Problem", DummyExplicitProblem)
    monkeypatch.setattr(solid_liquid_sep, "CVode", StopSolver)

    unit = _make_deliquoring_unit()

    with pytest.raises(StopAfterSetup):
        unit.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]

    # Destro et al. (2021), Eq. 16 gives p_i = 4.6*(1-eps)*sigma/(eps*d_i).
    # For d_i = [50, 100, 150] um and mean sigma = 0.066 N/m, p_i is
    # [7421.3333, 3710.6667, 2473.7778] Pa. Weighting p_i*CSD over the
    # stored [50, 100, 150] um grid gives 618444.444 Pa*#/m**3, and dividing
    # by mom_zero = trapz([1, 2, 1]) = 150 #/m**3 gives 4122.96296 Pa.
    expected_threshold = 4122.96296296  # [Pa]

    assert unit.p_thresh == pytest.approx(expected_threshold, rel=1e-9)


def test_get_sat_inf_returns_per_node_values_for_array_properties():
    """Array-valued liquid properties return one saturation per property node."""
    size_grid_um = np.array([50.0, 100.0, 150.0, 200.0])  # [um]
    size_grid_m = size_grid_um * 1e-6  # [m]
    csd_number = np.array([1.0, 2.0, 1.0, 0.5])  # [#/m**3/um]
    delta_p = 5.0e4  # [Pa]
    cake_height = 0.02  # [m]
    surf_tens = np.array([0.065, 0.067, 0.066])  # [N/m]
    rho_liq = np.array([950.0, 960.0, 970.0])  # [kg/m**3]
    mu_zero = _zeroth_moment(size_grid_um, csd_number)  # [#/m**3]

    sat_inf = solid_liquid_sep.get_sat_inf(
        size_grid_m,
        csd_number,
        delta_p,
        POROSITY,
        cake_height,
        mu_zero,
        (surf_tens, rho_liq),
    )

    # Literals are independently frozen from the array-property case so a
    # scalar reduction, flat sum, or node-ordering mutation cannot drift with
    # the production expression.
    expected_sat_inf = np.array(
        [0.1650323056996773, 0.1651821984011924, 0.1651072525989054]
    )  # [-]

    assert sat_inf.shape == expected_sat_inf.shape
    np.testing.assert_allclose(sat_inf, expected_sat_inf, rtol=1e-12)


def test_get_sat_inf_clips_micronized_aggregate_roundoff():
    """Searched micronized values clip upper-bound roundoff to one."""
    size_grid_um = np.array(
        [0.29443721003613355, 1.3640842476616897, 1.3987695850176836]
    )  # [um]
    size_grid_m = size_grid_um * 1e-6  # [m]
    csd_number = np.array(
        [1.6568408995106283e-7, 4.5172148784017079, 2.1064206650858339e-10]
    )  # [#/m**3/um]
    delta_p = 1.0e3  # [Pa]
    cake_height = 0.02  # [m]
    surf_tens = 0.066  # [N/m]
    rho_liq = 950.0  # [kg/m**3]
    mu_zero = _zeroth_moment(
        size_grid_um,
        csd_number,
    )  # [#/m**3]

    sat_inf = solid_liquid_sep.get_sat_inf(
        size_grid_m,
        csd_number,
        delta_p,
        POROSITY,
        cake_height,
        mu_zero,
        (surf_tens, rho_liq),
    )

    # These searched values are a numerical guard, not a physical cake. They
    # make all bin saturations hit the upper bound; before aggregate clipping,
    # volume-fraction roundoff sums the weighted saturation to
    # 1.0000000000000002 [-].
    assert sat_inf <= 1.0


def test_deliquoring_rejects_singular_micronized_irreducible_saturation():
    """Deliquoring rejects micronized cakes with undefined reduced saturation."""
    unit = _make_deliquoring_unit(
        size_grid_um=MICRONIZED_SIZE_GRID_UM,
        csd_number=MICRONIZED_CSD_NUMBER,
    )

    with pytest.raises(ValueError, match="s_inf=.*deltaP=.*particle size grid"):
        unit.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]


def _make_drying_unit(
    size_grid_um=SIZE_GRID_UM,
    csd_number=CSD_NUMBER,
):
    """Build a minimal drying unit for setup-path tests.

    Parameters
    ----------
    size_grid_um : ndarray, optional
        Particle-size grid stored by ``SolidPhase`` [um].
    csd_number : ndarray, optional
        Crystal-size distribution on ``size_grid_um`` [#/m**3/um].

    Returns
    -------
    Drying
        Unit with enough phase attributes to execute ``solve_unit`` setup.
    """
    size_grid_um = np.asarray(size_grid_um)  # [um]
    csd_number = np.asarray(csd_number)  # [#/m**3/um]

    dryer = drying_model.Drying(number_nodes=2, supercrit_names=["nitrogen"])
    dryer.names_states_in = ["temp", "mass_frac"]
    dryer.idx_supercrit = np.array([1])  # component indices [-]
    dryer.cake_height = 0.02  # [m]
    dryer.elapsed_time = 0.0  # [s]
    dryer.Liquid_1 = SimpleNamespace(
        num_species=3,
        getDensity=lambda temp, mass_frac, basis: np.array([930.0, 940.0]),  # [kg/m**3]
        getSurfTension=lambda temp, mass_frac: np.array([0.068, 0.070]),  # [N/m]
    )
    solid = SimpleNamespace(
        temp=302.0,  # [K]
        x_distrib=size_grid_um,
        distrib=csd_number,
        moments=np.array([_zeroth_moment(size_grid_um, csd_number)]),  # [#/m**3]
        getDensity=lambda: 1500.0,  # [kg/m**3]
        getCp=lambda: 700.0,  # [J/kg/K]
        getMoments=lambda mom_num: np.array([1.0, 1.0e-4, 1.0e-8, 1.0e-12, 0.0]),
    )
    dryer.Solid_1 = solid
    dryer.Vapor_1 = SimpleNamespace(
        mass_frac=np.array([0.01, 0.98, 0.01]),  # [-]
        temp=300.0,  # [K]
    )
    dryer.CakePhase = SimpleNamespace(
        Liquid_1=SimpleNamespace(mass_frac=np.array([0.20, 0.10, 0.70])),  # [-]
        Solid_1=solid,
        saturation=np.array([0.55]),  # [-]
        z_external=np.array([0.0, dryer.cake_height]),  # [m]
        alpha=2.0e10,  # [m/kg]
        porosity=POROSITY,  # [-]
    )

    return dryer


def test_drying_setup_converts_micrometer_grid_before_irreducible_saturation(
    monkeypatch,
):
    """Drying passes meter diameters to the shared saturation correlation."""
    captured = {}  # [-]

    def capture_sat_inf(x_vec, csd, deltaP, porosity, height, mu_zero, props):
        """Capture the diameter grid supplied by ``Drying.solve_unit``.

        Parameters
        ----------
        x_vec : ndarray
            Particle diameter grid supplied to the capillary correlation [m].
        csd : ndarray
            Crystal-size distribution on the stored grid [#/m**3/um].
        deltaP : float
            Pressure drop after medium resistance is removed [Pa].
        porosity : float
            Cake porosity [-].
        height : float
            Cake height [m].
        mu_zero : float
            Zeroth distribution moment [#/m**3].
        props : tuple
            Mean surface tension [N/m] and liquid density [kg/m**3].

        Returns
        -------
        float
            Synthetic irreducible saturation [-].
        """
        captured["x_vec"] = x_vec.copy()  # [m]
        return 0.25  # [-]

    monkeypatch.setattr(drying_model, "get_sat_inf", capture_sat_inf)
    monkeypatch.setattr(drying_model, "Explicit_Problem", DummyExplicitProblem)
    monkeypatch.setattr(drying_model, "CVode", StopSolver)

    dryer = _make_drying_unit()

    def stop_at_rhs(time, states, sw=None):
        """Stop the drying setup at the RHS evaluation boundary.

        Parameters
        ----------
        time : float
            Initial solve time [s].
        states : ndarray
            Flattened initial drying state [-] and [K].
        sw : list, optional
            Assimulo event switches [-].

        Raises
        ------
        StopAfterSetup
            Always raised to keep the test focused on setup calculations.
        """
        raise StopAfterSetup

    monkeypatch.setattr(dryer, "unit_model", stop_at_rhs)

    with pytest.raises(StopAfterSetup):
        dryer.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]

    np.testing.assert_allclose(captured["x_vec"], SIZE_GRID_UM * 1e-6)


def test_drying_rejects_singular_micronized_irreducible_saturation(
    monkeypatch,
):
    """Drying rejects micronized cakes with undefined reduced saturation."""
    monkeypatch.setattr(drying_model, "Explicit_Problem", DummyExplicitProblem)
    monkeypatch.setattr(drying_model, "CVode", StopSolver)

    dryer = _make_drying_unit(
        size_grid_um=MICRONIZED_SIZE_GRID_UM,
        csd_number=MICRONIZED_CSD_NUMBER,
    )

    def stop_at_rhs(time, states, sw=None):
        """Stop if drying reaches the RHS without rejecting ``s_inf``.

        Parameters
        ----------
        time : float
            Initial solve time [s].
        states : ndarray
            Flattened initial drying state [-] and [K].
        sw : list, optional
            Assimulo event switches [-].

        Raises
        ------
        StopAfterSetup
            Always raised when the validation guard is bypassed.
        """
        raise StopAfterSetup

    monkeypatch.setattr(dryer, "unit_model", stop_at_rhs)

    with pytest.raises(ValueError, match="s_inf=.*deltaP=.*particle size grid"):
        dryer.solve_unit(deltaP=5.0e4, runtime=10.0)  # [Pa], [s]
