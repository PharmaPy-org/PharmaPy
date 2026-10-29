"""Structural species presence in solid-liquid separation and drying (#425, B4).

``Drying`` integrates the gas and liquid mass fractions of every cake node
with CVode, whose difference-quotient Jacobian perturbs those of a species
that is never charged, fed or evaporated, and evaluated the gas viscosity,
the heat capacities, the liquid density, the vapor pressures and the latent
heats on them, so such a species needed ``visc_gas``, ``cp_vapor``,
``cp_liq``, ``rho_liq``, ``p_vap`` and Watson data. It now keeps two
structural presence records (liquid: initial volatiles; gas: initial gas,
the liquid's volatiles and the evaluated inlet gas, latched at their
evaluation time), weights those kernels with them, publishes absent species
at exactly zero and rolls back a failed solve. ``DeliquoringStep``,
``Filter`` and ``DisplacementWashing`` evaluate their kernels only on the
stored phases and cakes, never on solver states (probe evidence in the
#425 B4 report), but deliquoring evaluated the pure density and washing the
diffusivity of every species; both now evaluate the species present in the
cake (and, for washing, the wash solvent) only. ``getDensityPure``,
``getDiffusivityPure``, ``VaporPhase.getHeatVaporization``,
``getViscosityMix``/``VaporPhase.getViscosity`` and ``VaporPhase.getCp``
gained ``weights``.

Fixtures: the synthetic drying database of ``tests/conftest.py``
(``DRYING_THERMO_DATA``: water, ethanol and the nitrogen carrier), and the
shipped five-species database ``tests/Flowsheet/data/compound_database.json``
(A to D and solvent, liquid mass fractions 0.1 each and 0.6 solvent, solid
A) with synthetic diffusivities, as in
``tests/test_separation_balance_fixes.py``. Each adds a species ``api``:
``*_FULL`` gives it arbitrary synthetic data, ``API_BARE`` only its molar
mass. Expected values come from the same unit built on the database without
``api`` (an absent species must not change the present species) and from
the same unit with the full ``api`` record (its data must not change
anything). Core tests evaluate right-hand sides, initializations, result
publication and the analytical washing solution; the ``assimulo`` tests
integrate the dryer, the deliquoring step and the filter with CVode (well
under a second each).
"""

import copy
import json

import numpy as np
import pytest

from conftest import (DRYING_CSD_NUMBER, DRYING_SIZE_GRID_UM,
                      DRYING_THERMO_DATA)

from PharmaPy.Drying_Model import Drying
from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.SolidLiquidSep import (DeliquoringStep, DisplacementWashing,
                                     Filter)
from PharmaPy.Streams import VaporStream
from PharmaPy.ThermoModule import MissingPropertyError

# Arbitrary synthetic data for the drying database's api: mw [g/mol],
# rho_liq [kg/m**3], t_crit [K] (subcritical at every temperature here),
# constant cp_liq and cp_vapor [J/mol/K], visc_liq coefficients
# (log10(mu/[mPa*s]) = A + B/T + C*T + D*T**2, about 3.2 mPa*s), visc_gas
# [Pa*s], Antoine A [-], B [K], C [K] (log10(p/[Pa]) = A - B/(T + C); about
# 2e6 Pa at 300 K, far more volatile than water or ethanol, if it were
# present), Watson delta_hvap [J/mol] at tref_hvap [K], and surf_tension
# [N/m]. None of it may change a result where api is absent. Solid data
# are left out: the solid phase's own machine-epsilon sentinel rule (user
# guide) decides whether they enter the solid density.
API_DRYING_FULL = {
    "mw": 300.0, "rho_liq": 1300.0, "t_crit": 600.0, "cp_liq": [500.0],
    "cp_vapor": [400.0], "visc_liq": [0.5, 0.0, 0.0, 0.0],
    "visc_gas": 3.0e-5, "p_vap": [10.0, 1000.0, -40.0],
    "delta_hvap": 70000.0, "tref_hvap": 400.0, "surf_tension": 0.05}
API_BARE = {"mw": 300.0}  # [g/mol], no other property data
# api with only a critical temperature [K] above every temperature here, so
# that it could condense and a latent heat would need its Watson data.
API_CRITICAL_ONLY = {"mw": 300.0, "t_crit": 600.0}
API_DRYING_VARIANTS = {"bare": API_BARE, "critical_only": API_CRITICAL_ONLY}
# api only in the inlet gas: the gas data it needs there (visc_gas [Pa*s],
# constant cp_vapor [J/mol/K]) and no liquid data (no p_vap, cp_liq,
# rho_liq or Watson data; no t_crit, so it cannot condense).
API_GAS_ONLY = {"mw": 300.0, "visc_gas": 3.0e-5, "cp_vapor": [400.0]}
# Arbitrary synthetic data for the separation database's api: mw [g/mol],
# rho_liq [kg/m**3], visc_liq coefficients as above, surf_tension [N/m]
# and its diffusivity [m**2/s] (one value per reference species).
API_SEPARATION_FULL = {"mw": 300.0, "rho_liq": 1300.0,
                       "visc_liq": [0.5, 0.0, 0.0, 0.0],
                       "surf_tension": 0.05}
API_SEPARATION_DIFFUSIVITY = 3.3e-4  # [m**2/s]

# api first keeps the volatile column that issue #42 overwrites on water,
# as in the conftest dryer (provisional, see Drying.unit_model).
API_DRYING_INDEX = 0  # [-]
DRYING_SPECIES = ("water", "ethanol", "nitrogen")
# Compositions of the conftest dryer [-] (DRYING_LIQUID_MASS_FRACTION and
# DRYING_GAS_MASS_FRACTION), by species name.
LIQUID = {"water": 0.45, "ethanol": 0.45, "nitrogen": 0.10}  # [-]
GAS = {"water": 1e-6, "ethanol": 1e-6, "nitrogen": 0.999998}  # [-]
# A liquid that holds api, and an inlet gas that carries it [-].
LIQUID_WITH_API = {"api": 0.05, "water": 0.425, "ethanol": 0.425,
                   "nitrogen": 0.10}  # [-]
GAS_WITH_API = {"api": 1e-3, "water": 1e-6, "ethanol": 1e-6,
                "nitrogen": 0.998998}  # [-]
DRYING_NODES = 3  # [-]
# Species orders of the carrier-order cases (issue #527): the conftest order,
# with the nitrogen carrier last, and the same species with it first.
CARRIER_LAST = ("water", "ethanol", "nitrogen")
CARRIER_FIRST = ("nitrogen", "water", "ethanol")
# Probe state of the condensed energy balance: gas and condensed
# temperatures [K] with heat transfer only (zero drying rate), and volatile
# liquid mass fractions [-] with both volatiles nonzero.
PROBE_GAS_TEMP = 310.0  # [K]
PROBE_COND_TEMP = 300.0  # [K]
PROBE_LIQUID = {"water": 0.5, "ethanol": 0.5}  # [-]
PROBE_SATURATION = 0.55  # [-], the conftest cake
DRYING_PRESSURE_DROP = 5.0e4  # [Pa], as in tests/test_drying_model.py
CONDENSED_TEMP = 302.0  # [K], conftest dryer
GAS_TEMP = 300.0  # [K], conftest dryer
# A short integration [s]: CVode builds its Jacobian at t = 0, where the
# round-off of an absent species first appears (probe: the run without
# cp_liq failed at t = 0); about 100 output rows and 0.1 s of run time.
DRYING_RUNTIME = 1.0e-3  # [s]
# Inlet controller of the time-resolved cases: the inlet carries api from
# SWITCH_TIME on; BEFORE_SWITCH and AFTER_SWITCH are evaluation times [s].
SWITCH_TIME = 5.0e-4  # [s]
BEFORE_SWITCH = 2.0e-4  # [s]
AFTER_SWITCH = 8.0e-4  # [s]
# CVode internal-step limit [-] of a solve that stalls where api, without
# data, enters (issue #435): it fails within about a second (probed: 500
# steps take about 5 s, all at the switch time).
STALL_STEPS = 100  # [-]

SEPARATION_SPECIES = ("A", "B", "C", "D", "solvent")
API_SEPARATION_INDEX = 5  # [-], appended last, so the solvent stays 4
SOLVENT_INDEX = 4  # [-]
SEPARATION_LIQUID = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], solvent-rich cake
SEPARATION_SOLID = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], crystals of A
LIQUID_VOLUME = 1e-3  # [m**3]
GRID = np.array([100., 200., 300.])  # [um], uniform coarse crystals
POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um], asymmetric size weighting
# Synthetic diffusivities [m**2/s] of A to D and solvent, in every
# reference column: the low-Peclet probe scale of
# tests/test_separation_balance_fixes.py, distinct per species.
SEPARATION_DIFFUSIVITY = np.arange(1, 6) * 1e-4  # [m**2/s]
STATION_DIAMETER = 0.1  # [m], as in the separation tests
WASH_NODES = 4  # [-]
WASH_PRESSURE = 1.0  # [Pa], low-flow analytical probe
WASH_RATIO = 0.5  # [-], half a pore displacement
# api concentration [kg/m**3] in one cell of a cake field, an arbitrary
# value of the order of the shipped liquid's minor species.
CAKE_FIELD_API = 1.0  # [kg/m**3]
DELIQUORING_NODES = 3  # [-]
DELIQUORING_PRESSURE = 5.0e4  # [Pa], as in the separation tests
DELIQUORING_RUNTIME = 0.01  # [s], as in the separation tests
# Prescribed irreducible saturation [-] and time scale [1/s] of the
# solver-free deliquoring cases, as in tests/test_separation_spatial_handoff.py.
SAT_INF = 0.2  # [-]
THETA_CONV = 2.0  # [1/s]
FILTER_PRESSURE = 5.0e4  # [Pa], as in the separation tests

# Round-off of an absent species, of the size the solvers leave (issue
# #425 shows 1e-25 to 1e-21; the B4 probe saw 7e-17 inside the drying
# Jacobian), injected into states.
ROUND_OFF = 1.0e-20  # [-]
# Two databases differ only in data that enters through an exact zero, or
# in an absent species that adds exact zeros; a computed quantity can then
# differ only by the rounding of the few operations that form it. Eight
# units of float64 round-off bound that (assumption: an allowance for a
# handful of additions and products, not a derived count).
ROUNDING = 8 * np.finfo(float).eps  # [-]
# CVode local error limits of the drying integrations, relative [-] and
# absolute ([-] or [K]), and the allowance for two trajectories of
# different state dimension, which take different steps: well above the
# local limits, accumulated over a few hundred steps.
CVODE_RTOL = 1e-10  # [-]
CVODE_ATOL = 1e-12  # [state units]
TRAJECTORY_RTOL = 1e-6  # [-]
TRAJECTORY_ATOL = 1e-9  # [state units]
# DeliquoringStep integrates with CVode's default relative tolerance 1e-6
# [-], which solve_unit does not expose; with and without api the step
# sequences differ, and the present species then differ by up to about
# 8e-6 relative (probed). This allowance is an order of magnitude above
# that; an absent species that took part in the drainage would change the
# concentrations at the percent level.
DELIQUORING_TRAJECTORY_RTOL = 1e-4  # [-]


def _write(tmp_path, database, name):
    """Write a property database to a temporary JSON file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    database : dict
        Species records.
    name : str
        File name stem.

    Returns
    -------
    str
        Path of the written file.
    """
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(database))
    return str(path)


def _drying_database(tmp_path, api=None, name="drying",
                     order=DRYING_SPECIES, drop=None):
    """Write the conftest drying database, optionally with ``api`` first.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    api : dict, optional
        Property record of ``api``, placed first; None leaves it out.
    name : str, optional
        File name stem.
    order : sequence of str, optional
        Order of the conftest species in the database.
    drop : tuple of str, optional
        ``(species, property)`` to leave out of a conftest species.

    Returns
    -------
    str
        Path of the database.
    """
    database = {} if api is None else {"api": api}
    database.update({name_species: copy.deepcopy(
        DRYING_THERMO_DATA[name_species]) for name_species in order})
    if drop is not None:
        del database[drop[0]][drop[1]]
    return _write(tmp_path, database, name)


def _vector(path, values):
    """Spread named fractions over the species of a database.

    Parameters
    ----------
    path : str
        Property database whose species order is used.
    values : dict
        Fractions [-] by species name; other species get zero.

    Returns
    -------
    list of float
        Fractions [-] in the database's species order.
    """
    with open(path) as handle:
        names = list(json.load(handle))
    return [values.get(name, 0.0) for name in names]


def _dryer(path, liquid=LIQUID, gas=GAS, inlet=GAS):
    """Build the conftest dryer on a database and compositions.

    Parameters
    ----------
    path : str
        Property database.
    liquid, gas, inlet : dict, optional
        Mass fractions [-] by species name of the cake liquid, the initial
        pore gas and the inlet gas.

    Returns
    -------
    Drying
        Three-node dryer with the conftest cake (saturation 0.55 [-],
        condensed phases at 302 K) and gas (300 K).
    """
    liquid_phase = LiquidPhase(path, temp=CONDENSED_TEMP, mass=1.0e-3,
                               mass_frac=_vector(path, liquid))  # [kg]
    solid = SolidPhase(path, temp=CONDENSED_TEMP,
                       x_distrib=DRYING_SIZE_GRID_UM,
                       distrib=DRYING_CSD_NUMBER,
                       mass_frac=_vector(path, {"ethanol": 1.0}))
    cake = Cake(z_external=np.array([0.0, 1.0]),
                saturation=np.atleast_1d(0.55))  # [m], [-]
    cake.Phases = [liquid_phase, solid]
    dryer = Drying(DRYING_NODES, supercrit_names=["nitrogen"])
    dryer.Phases = cake
    dryer.CakePhase.z_external = np.array([0.0, dryer.cake_height])  # [m]
    dryer.Phases = VaporPhase(path, temp=GAS_TEMP, mass=1.0e-4,
                              mass_frac=_vector(path, gas))  # [kg]
    dryer.Inlet = VaporStream(path, temp=GAS_TEMP, mass_flow=1.0e-4,
                              mass_frac=_vector(path, inlet))  # [kg/s]
    return dryer


def _initial_states(dryer, round_off=False):
    """Return a dryer's initial states, optionally with round-off api.

    Parameters
    ----------
    dryer : Drying
        Dryer on a database with ``api`` first.
    round_off : bool, optional
        Set api's gas and liquid fractions to ``ROUND_OFF`` [-] at every
        node, as solver iterates would.

    Returns
    -------
    numpy.ndarray
        Node-major states, shape ``(num_nodes, 3 + num_species +
        num_volatiles)``: saturation [-], gas fractions [-], volatile
        liquid fractions [-], gas and condensed temperatures [K].
    """
    states = dryer.initialize_states(DRYING_PRESSURE_DROP).copy()
    if round_off:
        num_species = dryer.Liquid_1.num_species  # [-]
        states[:, 1 + API_DRYING_INDEX] = ROUND_OFF  # [-], gas
        states[:, 1 + num_species + API_DRYING_INDEX] = ROUND_OFF  # [-], liquid
    return states


def _without_api(states, num_species):
    """Drop api's gas and liquid columns from drying states.

    Parameters
    ----------
    states : numpy.ndarray
        Node-major states with api first (``_initial_states``), shape
        ``(num_nodes, 3 + num_species + num_volatiles)``, or its derivatives.
    num_species : int
        Number of species with api [-].

    Returns
    -------
    numpy.ndarray
        The same rows without api's two columns.
    """
    keep = np.ones(states.shape[1], dtype=bool)
    keep[[1 + API_DRYING_INDEX, 1 + num_species + API_DRYING_INDEX]] = False
    return states[:, keep]


def _rhs(dryer, states, time=0.0):
    """Evaluate a dryer's right-hand side by node.

    Parameters
    ----------
    dryer : Drying
        Initialized dryer.
    states : numpy.ndarray
        Node-major states (``_initial_states``); not modified.
    time : float, optional
        Evaluation time [s].

    Returns
    -------
    numpy.ndarray
        Derivatives [1/s] and [K/s] with the shape of ``states``.
    """
    return dryer.unit_model(time, states.copy().ravel()).reshape(
        states.shape)


def _switching_inlet(dryer, path, after=GAS_WITH_API):
    """Give a dryer an inlet gas that carries api from SWITCH_TIME on.

    Parameters
    ----------
    dryer : Drying
        Dryer whose inlet stream gets the controller.
    path : str
        Its property database.
    after : dict, optional
        Inlet mass fractions [-] by species name from SWITCH_TIME on;
        before it, ``GAS``.
    """
    before_vector = np.array(_vector(path, GAS))  # [-]
    after_vector = np.array(_vector(path, after))  # [-]

    def inlet_fractions(time):
        """Return the inlet gas mass fractions at some times.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation time [s], scalar or shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray
            Mass fractions [-], shape ``(num_species,)`` or
            ``(num_times, num_species)``.
        """
        switched = np.asarray(time, dtype=float) >= SWITCH_TIME  # [-]
        return np.where(switched[..., np.newaxis], after_vector,
                        before_vector)

    control = DynamicInput()
    control.add_variable("mass_frac", inlet_fractions)
    dryer.Inlet.DynamicInlet = control


# ---------------------------------------------------------------- kernels


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["density", "diffusivity"])
def test_weighted_pure_kernels_skip_unweighted_species(data_path, tmp_path,
                                                       kernel):
    """Only weighted species are evaluated; the others get exactly 0.

    The present species get the database values; api, without data and
    with zero weight, needs none. Without weights every species still
    needs the data.
    """
    path = _separation_database(data_path, tmp_path, API_BARE)
    liquid = _separation_liquid(path)
    weights = np.array([1.0] * 5 + [0.0])  # [-]
    shipped = json.loads(
        (data_path["flowsheet"] / "compound_database.json").read_text())
    if kernel == "density":
        values, molar = liquid.getDensityPure(weights=weights)  # [kg/m**3], [kmol/m**3]
        expected = [shipped[name]["rho_liq"] for name in SEPARATION_SPECIES]  # [kg/m**3]
        np.testing.assert_allclose(
            molar[:5], np.array(expected) / [shipped[name]["mw"]
                                             for name in SEPARATION_SPECIES],
            rtol=ROUNDING, atol=0)
        assert molar[API_SEPARATION_INDEX] == 0.0
        missing = "rho_liq"
        unweighted = liquid.getDensityPure
    else:
        values = liquid.getDiffusivityPure(SOLVENT_INDEX, weights=weights)  # [m**2/s]
        expected = SEPARATION_DIFFUSIVITY  # [m**2/s]
        missing = "diffusivity"

        def unweighted():
            return liquid.getDiffusivityPure(SOLVENT_INDEX)

    np.testing.assert_allclose(values[:5], expected, rtol=ROUNDING, atol=0)
    assert values[API_SEPARATION_INDEX] == 0.0
    with pytest.raises(MissingPropertyError,
                       match=rf"\['api'\].*'{missing}'"):
        unweighted()


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["density", "diffusivity"])
def test_weighted_pure_kernels_name_a_weighted_species_without_data(
        data_path, tmp_path, kernel):
    """A weighted species without the data raises the species-named error."""
    path = _separation_database(data_path, tmp_path, API_BARE)
    liquid = _separation_liquid(path)
    weights = np.zeros(6)  # [-]
    weights[[SOLVENT_INDEX, API_SEPARATION_INDEX]] = 1.0
    with pytest.raises(MissingPropertyError,
                       match=r"\['api'\] carry a nonzero weight"):
        if kernel == "density":
            liquid.getDensityPure(weights=weights)
        else:
            liquid.getDiffusivityPure(SOLVENT_INDEX, weights=weights)


@pytest.mark.unit
@pytest.mark.parametrize("shape", ["scalar", "short", "two_dimensional"])
@pytest.mark.parametrize("kernel", ["density", "diffusivity", "latent_heat"])
def test_weighted_pure_kernels_reject_weights_of_another_shape(
        data_path, tmp_path, kernel, shape):
    """Weights must have exactly the shape (num_species,).

    A scalar, a short vector or a one-row matrix would broadcast and could
    switch the missing-data check off.
    """
    if kernel == "latent_heat":
        phase = VaporPhase(_drying_database(tmp_path, API_BARE),
                           temp=GAS_TEMP, mass=1.0,
                           mass_frac=[0.0, 0.2, 0.3, 0.5])  # [kg], [-]
    else:
        phase = _separation_liquid(
            _separation_database(data_path, tmp_path, API_BARE))
    num_species = phase.num_species  # [-]
    weights = {"scalar": 1.0, "short": np.ones(num_species - 1),
               "two_dimensional": np.ones((1, num_species))}[shape]  # [-]
    with pytest.raises(ValueError,
                       match=rf"one entry per species, shape \({num_species},\)"):
        if kernel == "density":
            phase.getDensityPure(weights=weights)
        elif kernel == "diffusivity":
            phase.getDiffusivityPure(SOLVENT_INDEX, weights=weights)
        else:
            phase.getHeatVaporization(CONDENSED_TEMP, weights=weights)


@pytest.mark.unit
@pytest.mark.parametrize("api", ["bare", "full"])
def test_vapor_viscosity_weights_evaluate_the_weighted_species_only(
        tmp_path, api):
    """A zero-weight api, round-off or not, changes no vapor viscosity.

    The Wilke sums run over the weighted species, so api needs no
    ``visc_gas`` and, with data, its round-off fraction adds nothing: the
    result equals the database without api fed the same present fractions.
    """
    record = API_BARE if api == "bare" else API_DRYING_FULL
    path = _drying_database(tmp_path, record, api)
    reference_path = _drying_database(tmp_path, name="reference")
    present = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])  # [-], two nodes
    fractions = np.column_stack((np.full(2, ROUND_OFF), present))  # [-]
    weights = np.column_stack((np.zeros(2), np.ones((2, 3))))  # [-]
    vapor = VaporPhase(path, temp=GAS_TEMP, mass=1.0,
                       mass_frac=fractions[0])  # [kg]
    reference = VaporPhase(reference_path, temp=GAS_TEMP, mass=1.0,
                           mass_frac=present[0])  # [kg]

    viscosity = vapor.getViscosity(mass_frac=fractions, weights=weights)  # [Pa*s]

    np.testing.assert_allclose(
        viscosity, reference.getViscosity(mass_frac=present),
        rtol=ROUNDING, atol=0)
    if api == "bare":
        with pytest.raises(MissingPropertyError, match=r"\['api'\]"):
            vapor.getViscosity(mass_frac=fractions)


@pytest.mark.unit
def test_vapor_viscosity_weights_name_a_weighted_species_without_data(
        tmp_path):
    """A weighted api without ``visc_gas`` raises the species-named error."""
    path = _drying_database(tmp_path, API_BARE)
    vapor = VaporPhase(path, temp=GAS_TEMP, mass=1.0,
                       mass_frac=[0.0, 0.2, 0.3, 0.5])  # [kg], [-]
    with pytest.raises(MissingPropertyError,
                       match=r"\['api'\] carry a nonzero weight.*'visc_gas'"):
        vapor.getViscosity(weights=np.ones(4))
    with pytest.raises(ValueError, match="weights must have the shape"):
        vapor.getViscosity(weights=np.ones(3))


@pytest.mark.unit
def test_liquid_viscosity_weights_drop_zero_weight_entries(tmp_path):
    """A zero-weight entry needs no ``visc_liq`` and adds nothing.

    api, without liquid-viscosity data, carries a round-off mole fraction:
    with zero weight the logarithmic mixing rule leaves it out and equals
    the database without api; without weights the fraction rule needs its
    ``visc_liq``.
    """
    path = _drying_database(tmp_path, API_BARE)
    reference_path = _drying_database(tmp_path, name="reference")
    present = np.array([0.2, 0.3, 0.5])  # [-]
    fractions = np.concatenate(([ROUND_OFF], present))  # [-]
    liquid = LiquidPhase(path, temp=CONDENSED_TEMP, mass=1.0,
                         mole_frac=np.concatenate(([0.0], present)))  # [kg]
    reference = LiquidPhase(reference_path, temp=CONDENSED_TEMP, mass=1.0,
                            mole_frac=present)  # [kg]

    viscosity = liquid.getViscosityMix(
        mole_frac=fractions, weights=np.array([0.0, 1.0, 1.0, 1.0]))  # [Pa*s]

    assert viscosity == pytest.approx(
        reference.getViscosityMix(mole_frac=present), rel=ROUNDING, abs=0)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'visc_liq'"):
        liquid.getViscosityMix(mole_frac=fractions)


@pytest.mark.unit
def test_vapor_heat_capacity_weights(tmp_path):
    """VaporPhase.getCp passes weights: api needs no ``cp_vapor``."""
    path = _drying_database(tmp_path, API_BARE)
    reference_path = _drying_database(tmp_path, name="reference")
    present = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])  # [-]
    fractions = np.column_stack((np.full(2, ROUND_OFF), present))  # [-]
    vapor = VaporPhase(path, temp=GAS_TEMP, mass=1.0,
                       mass_frac=fractions[0])  # [kg]
    reference = VaporPhase(reference_path, temp=GAS_TEMP, mass=1.0,
                           mass_frac=present[0])  # [kg]
    temps = np.array([300.0, 320.0])  # [K]
    weights = np.column_stack((np.zeros(2), np.ones((2, 3))))  # [-]

    heat_capacity = vapor.getCp(temps, mass_frac=fractions, weights=weights)  # [J/kg/K]

    expected = reference.getCp(temps, mass_frac=present)  # [J/kg/K]
    np.testing.assert_allclose(heat_capacity, expected, rtol=ROUNDING, atol=0)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_vapor'"):
        vapor.getCp(temps, mass_frac=fractions)


@pytest.mark.unit
@pytest.mark.parametrize("api", ["critical_only", "full"])
def test_latent_heat_weights_zero_the_unweighted_columns(tmp_path, api):
    """Zero-weight columns are exactly zero, whatever api's Watson data.

    The other columns equal those of the database without api. api with a
    critical temperature [K] but no Watson data needs them once weighted
    (or without weights), since it can condense below t_crit.
    """
    record = API_CRITICAL_ONLY if api == "critical_only" else API_DRYING_FULL
    path = _drying_database(tmp_path, record, api)
    reference_path = _drying_database(tmp_path, name="reference")
    vapor = VaporPhase(path, temp=GAS_TEMP, mass=1.0,
                       mass_frac=[0.0, 0.2, 0.3, 0.5])  # [kg], [-]
    reference = VaporPhase(reference_path, temp=GAS_TEMP, mass=1.0,
                           mass_frac=[0.2, 0.3, 0.5])  # [kg], [-]
    temps = np.array([300.0, 310.0, 320.0])  # [K]

    latent = vapor.getHeatVaporization(
        temps, weights=np.array([0.0, 1.0, 1.0, 1.0]))  # [J/kg]

    np.testing.assert_array_equal(latent[:, API_DRYING_INDEX], 0.0)
    np.testing.assert_allclose(latent[:, 1:],
                               reference.getHeatVaporization(temps),
                               rtol=ROUNDING, atol=0)
    if api == "critical_only":
        for weights in (np.ones(4), None):
            with pytest.raises(MissingPropertyError, match=r"\['api'\]"):
                vapor.getHeatVaporization(temps, weights=weights)


# ----------------------------------------------------------------- drying


@pytest.mark.unit
@pytest.mark.parametrize("variant", sorted(API_DRYING_VARIANTS))
def test_drying_balances_ignore_round_off_of_an_absent_species(tmp_path,
                                                               variant):
    """Round-off api needs no data; its data change no derivative.

    With round-off api states, the derivatives equal those with api's full
    data. With api at exactly zero, its derivatives are exactly zero (its
    balances are homogeneous in its fractions) and those of the present
    species equal the dryer without api at the same states. Earlier the
    vapor viscosity required api's ``visc_gas``, the latent heats its
    Watson data when it has a critical temperature, and, inside CVode, the
    liquid density its ``rho_liq`` and the heat capacities its ``cp_liq``
    and ``cp_vapor``.
    """
    round_off, exact = {}, {}
    for label, record in (("bare", API_DRYING_VARIANTS[variant]),
                          ("full", API_DRYING_FULL)):
        dryer = _dryer(_drying_database(tmp_path, record, label))
        round_off[label] = _rhs(dryer, _initial_states(dryer, True))
        exact[label] = _rhs(dryer, _initial_states(dryer))
    num_species = dryer.Liquid_1.num_species  # [-]
    reference = _dryer(_drying_database(tmp_path, name="reference"))
    expected = _rhs(reference, _initial_states(reference))

    np.testing.assert_allclose(round_off["bare"], round_off["full"],
                               rtol=ROUNDING, atol=0)
    np.testing.assert_array_equal(exact["bare"], exact["full"])
    api_columns = [1 + API_DRYING_INDEX, 1 + num_species + API_DRYING_INDEX]
    np.testing.assert_array_equal(exact["bare"][:, api_columns], 0.0)
    np.testing.assert_allclose(_without_api(exact["bare"], num_species),
                               expected, rtol=ROUNDING, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["cp_liq", "cp_vapor", "visc_gas",
                                     "p_vap", "delta_hvap"])
def test_charged_species_without_data_is_named_at_the_start(tmp_path,
                                                            missing):
    """api in the liquid needs its liquid and gas data from the start.

    It evaporates from the start, so it is present in the gas although its
    initial gas fraction is zero: the initial evaluation that solve_unit
    makes before CVode names its missing ``cp_vapor``, which the fraction
    rule left to CVode, as well as the data the fraction rule required.
    """
    record = {key: value for key, value in API_DRYING_FULL.items()
              if key != missing}
    dryer = _dryer(_drying_database(tmp_path, record), liquid=LIQUID_WITH_API)
    states = _initial_states(dryer)
    assert np.all(states[:, 1 + API_DRYING_INDEX] == 0.0)

    with pytest.raises(MissingPropertyError, match=rf"'?api'?.*{missing}"):
        _rhs(dryer, states)


@pytest.mark.unit
def test_inlet_species_without_gas_data_is_named_at_the_start(tmp_path):
    """api in the inlet gas is present from its first evaluation on.

    The initial pore gas holds none, so the fraction rule needed no
    ``cp_vapor`` at the start; the inlet latches api, so the initial
    evaluation names it.
    """
    record = {key: value for key, value in API_GAS_ONLY.items()
              if key != "cp_vapor"}
    dryer = _dryer(_drying_database(tmp_path, record), inlet=GAS_WITH_API)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_vapor'"):
        _rhs(dryer, _initial_states(dryer))


@pytest.mark.unit
def test_gas_only_species_needs_no_liquid_data(tmp_path):
    """api only in the inlet gas needs no ``p_vap``, ``cp_liq``, ``rho_liq``.

    It never enters the liquid (nothing condenses), so its liquid
    derivatives are exactly zero and the inlet convects it into the first
    node. The derivatives equal those with arbitrary liquid data for it.
    Earlier the vapor pressure of every volatile required its ``p_vap``.
    """
    derivatives = {}
    records = {"gas_only": API_GAS_ONLY,
               "full": dict(API_DRYING_FULL, **API_GAS_ONLY)}
    for label, record in records.items():
        dryer = _dryer(_drying_database(tmp_path, record, label),
                       inlet=GAS_WITH_API)
        derivatives[label] = _rhs(dryer, _initial_states(dryer))
    num_species = dryer.Liquid_1.num_species  # [-]

    np.testing.assert_allclose(derivatives["gas_only"], derivatives["full"],
                               rtol=ROUNDING, atol=0)
    assert np.all(
        derivatives["gas_only"][:, 1 + num_species + API_DRYING_INDEX] == 0.0)
    assert derivatives["gas_only"][0, 1 + API_DRYING_INDEX] > 0.0


@pytest.mark.unit
def test_species_fed_later_is_present_from_its_first_evaluation(tmp_path):
    """The inlet latches api at the time it first carries it.

    Before the switch, round-off api needs no data; an evaluation after
    the switch needs its gas data; an evaluation before the switch still
    needs none afterwards and gives the same derivatives.
    """
    path = _drying_database(tmp_path, API_BARE)
    dryer = _dryer(path)
    _switching_inlet(dryer, path)
    states = _initial_states(dryer, round_off=True)

    before = _rhs(dryer, states, BEFORE_SWITCH)
    with pytest.raises(MissingPropertyError, match=r"\['api'\]"):
        _rhs(dryer, states, AFTER_SWITCH)
    again = _rhs(dryer, states, BEFORE_SWITCH)

    np.testing.assert_array_equal(again, before)


@pytest.mark.unit
def test_drying_publishes_each_row_with_its_presence(tmp_path):
    """Published fractions hold api at zero where it was absent.

    api is latched in the gas at AFTER_SWITCH; the gas row before it is
    published at exactly zero and the later row keeps its value. api never
    enters the liquid, so its liquid fraction is zero in both rows.
    """
    path = _drying_database(tmp_path, API_DRYING_FULL)
    dryer = _dryer(path)
    _switching_inlet(dryer, path)
    states = _initial_states(dryer, round_off=True)
    _rhs(dryer, states, AFTER_SWITCH)
    later = states.copy()
    num_species = dryer.Liquid_1.num_species  # [-]
    later[:, 1 + API_DRYING_INDEX] = 1e-4  # [-], api fed in by then
    rows = np.vstack((states.ravel(), later.ravel()))
    times = np.array([BEFORE_SWITCH, 2 * AFTER_SWITCH])  # [s]

    dryer.retrieve_results(times, rows)

    gas = dryer.result.y_gas["api"]  # [-], time rows by node
    np.testing.assert_array_equal(gas[0], np.zeros(DRYING_NODES))
    np.testing.assert_array_equal(gas[1], later[:, 1 + API_DRYING_INDEX])
    np.testing.assert_array_equal(dryer.result.x_liq["api"],
                                  np.zeros((2, DRYING_NODES)))
    np.testing.assert_array_equal(dryer.result.y_gas["water"][1],
                                  later[:, 1 + 1])
    assert np.all(rows[:, 1 + API_DRYING_INDEX::later.shape[1]] != 0.0)
    assert rows[0, 1 + num_species + API_DRYING_INDEX] == ROUND_OFF


def _probe_condensed_rate(dryer, path):
    """Return dT_cond/dt of a dryer at the probe state.

    Parameters
    ----------
    dryer : Drying
        Dryer on ``path``, initialized (``initialize_states``) and with
        ``rho_liq`` [kg/m**3] set by one ``unit_model`` evaluation.
    path : str
        Its property database.

    Returns
    -------
    numpy.ndarray
        Condensed-phase temperature derivative [K/s] by node at
        ``PROBE_GAS_TEMP``, ``PROBE_COND_TEMP``, ``PROBE_LIQUID`` and zero
        drying rate, so that gas-to-condensed heat transfer is the only
        source.
    """
    with open(path) as handle:
        names = list(json.load(handle))
    volatiles = [names[index] for index in dryer.idx_volatiles]
    x_liq = np.tile([PROBE_LIQUID[name] for name in volatiles],
                    (DRYING_NODES, 1))  # [-], volatile order
    gas = np.array(_vector(path, GAS))  # [-]
    _, condensed_rate = dryer.energy_balance(
        0.0, np.full(DRYING_NODES, PROBE_GAS_TEMP),
        np.full(DRYING_NODES, PROBE_COND_TEMP),
        np.full(DRYING_NODES, PROBE_SATURATION),
        np.tile(gas, (DRYING_NODES, 1)), x_liq, np.zeros(DRYING_NODES),
        np.ones(DRYING_NODES), np.zeros((DRYING_NODES, len(names))),
        {"temp": PROBE_GAS_TEMP, "mass_frac": gas})  # [K/s]
    return condensed_rate


@pytest.mark.unit
@pytest.mark.parametrize("order", [CARRIER_LAST, CARRIER_FIRST],
                         ids=["carrier_last", "carrier_first"])
def test_liquid_heat_capacity_follows_the_species_of_each_fraction(
        tmp_path, order):
    """The condensed balance uses the liquid's own heat capacity (#527).

    With heat transfer only, dT_cond/dt = h a (T_gas - T_cond) / (rho_s
    (1 - eps) cp_s + eps S cp_liq rho_liq), with cp_liq the mass-basis
    mixture sum_i w_i cp_i / M_i of the probe liquid, 3278.985507 J/kg/K
    (water 75/18 and ethanol 110/46 kJ/kg/K at 0.5/0.5), in either species
    order. Earlier the carrier-first order read the water fraction as
    nitrogen and the ethanol fraction as water (2601.190476 J/kg/K).
    """
    path = _drying_database(tmp_path, order=order)
    dryer = _dryer(path)
    _rhs(dryer, _initial_states(dryer))

    condensed_rate = _probe_condensed_rate(dryer, path)  # [K/s]

    heat_capacity = sum(
        PROBE_LIQUID[name] * DRYING_THERMO_DATA[name]["cp_liq"][0]
        / DRYING_THERMO_DATA[name]["mw"] * 1000
        for name in PROBE_LIQUID)  # [J/kg/K], 1000 g/kg
    assert heat_capacity == pytest.approx(3278.985507, abs=1e-6)
    capacity = (dryer.rho_sol * (1 - dryer.porosity) * dryer.cp_sol
                + dryer.porosity * PROBE_SATURATION * heat_capacity
                * dryer.rho_liq)  # [J/m**3/K]
    expected = (dryer.h_T_j * dryer.a_V
                * (PROBE_GAS_TEMP - PROBE_COND_TEMP) / capacity)  # [K/s]
    np.testing.assert_allclose(condensed_rate, expected, rtol=ROUNDING,
                               atol=0)


@pytest.mark.unit
def test_species_order_does_not_change_the_drying_balances(tmp_path):
    """The carrier-first and carrier-last orders give the same derivatives.

    The same physical system in both orders gives the same condensed rate
    at the probe state and, at the initial state, the same derivatives
    once the gas columns are put in one order.
    """
    rates, derivatives = {}, {}
    for label, order in (("last", CARRIER_LAST), ("first", CARRIER_FIRST)):
        path = _drying_database(tmp_path, name=label, order=order)
        dryer = _dryer(path)
        states = _initial_states(dryer)
        rhs = _rhs(dryer, states)
        rates[label] = _probe_condensed_rate(dryer, path)  # [K/s]
        # Gas columns in CARRIER_LAST order; the volatiles keep theirs.
        gas = [1 + order.index(name) for name in CARRIER_LAST]
        columns = [0] + gas + list(range(1 + len(order), rhs.shape[1]))
        derivatives[label] = rhs[:, columns]

    np.testing.assert_allclose(rates["first"], rates["last"], rtol=ROUNDING,
                               atol=0)
    np.testing.assert_allclose(derivatives["first"], derivatives["last"],
                               rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_present_volatile_needs_its_liquid_heat_capacity_where_zeroed(
        tmp_path):
    """Water, present in the liquid, needs ``cp_liq`` although zeroed.

    The provisional overwrite of issue #42 zeroes water's column of the
    liquid fractions in ``unit_model`` (the conftest dryer's first volatile),
    so the fraction rule let water's liquid heat capacity go unused and the
    dryer ran without its ``cp_liq``. Water is present in the liquid (45 %
    of the charge), so the presence weights require it: the initial
    evaluation names it. Once #42 is fixed, the fraction rule would require
    it too and this test remains valid.
    """
    dryer = _dryer(_drying_database(tmp_path, drop=("water", "cp_liq")))
    with pytest.raises(MissingPropertyError,
                       match=r"\['water'\].*'cp_liq'"):
        _rhs(dryer, _initial_states(dryer))


@pytest.mark.unit
def test_carrier_first_needs_no_carrier_liquid_heat_capacity(tmp_path):
    """A carrier never in the liquid needs no ``cp_liq``, in any order.

    With the carrier first and without its ``cp_liq``, the derivatives are
    finite and equal those with its data. The round-0 #425 weights placed
    the volatile weights in the carrier's column and required it.
    """
    derivatives = {}
    for label, drop in (("without", ("nitrogen", "cp_liq")), ("with", None)):
        path = _drying_database(tmp_path, name=label, order=CARRIER_FIRST,
                                drop=drop)
        dryer = _dryer(path)
        derivatives[label] = _rhs(dryer, _initial_states(dryer))

    assert np.all(np.isfinite(derivatives["without"]))
    np.testing.assert_array_equal(derivatives["without"], derivatives["with"])


@pytest.mark.assimulo
@pytest.mark.integration
def test_drying_integrates_without_data_for_an_absent_species(tmp_path):
    """A dryer with an absent api integrates without its data.

    The run equals the run with api's full data to round-off and the run
    without api to integration tolerance; api is published at zero.
    Earlier the run raised for api's ``visc_gas``.
    """
    pytest.importorskip("assimulo")
    options = {"rtol": CVODE_RTOL, "atol": CVODE_ATOL}
    runs = {}
    for label, record in (("bare", API_BARE), ("full", API_DRYING_FULL),
                          ("none", None)):
        dryer = _dryer(_drying_database(tmp_path, record, label))
        time, states = dryer.solve_unit(DRYING_PRESSURE_DROP,
                                        runtime=DRYING_RUNTIME,
                                        verbose=False, sundials_opts=options)
        runs[label] = (dryer, time, states)
    bare, time, states = runs["bare"]
    num_species = bare.Liquid_1.num_species  # [-]

    np.testing.assert_allclose(states, runs["full"][2], rtol=ROUNDING,
                               atol=ROUNDING * ROUND_OFF)
    final = _without_api(states[-1].reshape(DRYING_NODES, -1), num_species)
    np.testing.assert_allclose(
        final, runs["none"][2][-1].reshape(DRYING_NODES, -1),
        rtol=TRAJECTORY_RTOL, atol=TRAJECTORY_ATOL)
    assert time[-1] == pytest.approx(runs["none"][1][-1], rel=ROUNDING)
    np.testing.assert_array_equal(bare.result.y_gas["api"], 0.0)
    np.testing.assert_array_equal(bare.result.x_liq["api"], 0.0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_drying_solve_names_a_charged_species_without_data_before_cvode(
        tmp_path):
    """solve_unit names a liquid api without ``cp_vapor`` before CVode.

    Earlier its zero initial gas fraction let the initial evaluation pass,
    and CVode then failed without naming it.
    """
    pytest.importorskip("assimulo")
    record = {key: value for key, value in API_DRYING_FULL.items()
              if key != "cp_vapor"}
    dryer = _dryer(_drying_database(tmp_path, record), liquid=LIQUID_WITH_API)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_vapor'"):
        dryer.solve_unit(DRYING_PRESSURE_DROP, runtime=DRYING_RUNTIME,
                         verbose=False)
    assert not hasattr(dryer, "result")


@pytest.mark.assimulo
@pytest.mark.integration
def test_failed_drying_solve_keeps_no_results(tmp_path):
    """A solve that fails inside CVode restores the unit and its records.

    The second solve's inlet carries api, without data, from SWITCH_TIME
    on. CVode cannot step past it and stalls there (issue #435), so
    ``STALL_STEPS`` bounds the attempt, which then fails. The first solve's
    result, elapsed time and presence records remain, and api, latched
    only by the failed attempt, is not present. Earlier the attempt left
    ``elapsed_time`` at the first entry of its time grid.

    The ``CVodeError`` expectation is provisional, pending issue #435: once
    a species first fed during an integration raises its named
    ``MissingPropertyError`` or no longer stalls the integration, update
    the expected exception (and drop ``STALL_STEPS``); the rollback
    assertions stay.
    """
    pytest.importorskip("assimulo")
    from assimulo.solvers.sundials import CVodeError

    path = _drying_database(tmp_path, API_BARE)
    dryer = _dryer(path)
    dryer.solve_unit(DRYING_PRESSURE_DROP, runtime=DRYING_RUNTIME,
                     verbose=False)
    result, elapsed = dryer.result, dryer.elapsed_time  # [s]
    gas_mask = dryer._presence.mask
    _switching_inlet(dryer, path)

    with pytest.raises(CVodeError):
        dryer.solve_unit(DRYING_PRESSURE_DROP,
                         time_grid=np.array([BEFORE_SWITCH, AFTER_SWITCH]),
                         verbose=False,
                         sundials_opts={"maxsteps": STALL_STEPS})

    assert dryer.result is result
    assert dryer.elapsed_time == elapsed
    np.testing.assert_array_equal(dryer._presence.mask, gas_mask)
    assert not dryer._presence.mask[API_DRYING_INDEX]


# ----------------------------------------------- solid-liquid separation


def _separation_database(data_path, tmp_path, api=None, name="separation",
                         solvent_density=True):
    """Write the shipped five-species database with diffusivities.

    Parameters
    ----------
    data_path : dict
        Repository data directories.
    tmp_path : pathlib.Path
        Directory for the file.
    api : dict, optional
        Property record of ``api``, appended last; ``API_SEPARATION_FULL``
        also gets ``API_SEPARATION_DIFFUSIVITY``. None leaves api out.
    name : str, optional
        File name stem.
    solvent_density : bool, optional
        False drops the solvent's ``rho_liq``.

    Returns
    -------
    str
        Path of the database.
    """
    database = json.loads(
        (data_path["flowsheet"] / "compound_database.json").read_text())
    num_species = len(database) + (api is not None)  # [-]
    for name_species, value in zip(SEPARATION_SPECIES,
                                   SEPARATION_DIFFUSIVITY):
        database[name_species]["diffusivity"] = [value] * num_species  # [m**2/s]
    if not solvent_density:
        del database["solvent"]["rho_liq"]
    if api is not None:
        database["api"] = dict(api)
        if api is API_SEPARATION_FULL:
            database["api"]["diffusivity"] = (
                [API_SEPARATION_DIFFUSIVITY] * num_species)  # [m**2/s]
    return _write(tmp_path, database, name)


def _separation_liquid(path, fractions=SEPARATION_LIQUID):
    """Build the separation tests' liquid on a database.

    Parameters
    ----------
    path : str
        Property database, with or without api last.
    fractions : sequence of float, optional
        Mass fractions [-] of A to D and solvent; api gets zero.

    Returns
    -------
    LiquidPhase
        One litre of liquid.
    """
    with open(path) as handle:
        num_species = len(json.load(handle))  # [-]
    mass_frac = list(fractions) + [0.0] * (num_species - 5)  # [-]
    return LiquidPhase(path, mass_frac=mass_frac, vol=LIQUID_VOLUME)  # [m**3]


def _separation_phases(path, fractions=SEPARATION_LIQUID):
    """Build the separation tests' liquid and solid on a database.

    Parameters
    ----------
    path : str
        Property database, with or without api last.
    fractions : sequence of float, optional
        Liquid mass fractions [-] of A to D and solvent.

    Returns
    -------
    list
        Liquid and solid phases.
    """
    liquid = _separation_liquid(path, fractions)
    solid_frac = SEPARATION_SOLID + [0.0] * (liquid.num_species - 5)  # [-]
    solid = SolidPhase(path, mass_frac=solid_frac, x_distrib=GRID,
                       distrib=POPULATION)
    return [liquid, solid]


def _washer(path, fractions=SEPARATION_LIQUID):
    """Build a displacement-washing unit on a freshly packed cake.

    Parameters
    ----------
    path : str
        Property database.
    fractions : sequence of float, optional
        Liquid mass fractions [-] of A to D and solvent.

    Returns
    -------
    DisplacementWashing
        Four-node washer of the solvent (index 4).
    """
    washer = DisplacementWashing(solvent_idx=SOLVENT_INDEX,
                                 num_nodes=WASH_NODES,
                                 diam_unit=STATION_DIAMETER)
    washer.Phases = _separation_phases(path, fractions)
    return washer


def _deliquoring(path):
    """Build a deliquoring step with a prescribed irreducible saturation.

    Parameters
    ----------
    path : str
        Property database.

    Returns
    -------
    DeliquoringStep
        Three-node step on a saturated cake of the separation phases, with
        ``sat_inf`` = ``SAT_INF`` [-], ready for ``initialize_states``.
    """
    unit = DeliquoringStep(num_nodes=DELIQUORING_NODES,
                           diam_unit=STATION_DIAMETER)
    unit.Phases = _separation_phases(path)
    unit.sat_inf = SAT_INF  # [-]
    return unit


@pytest.mark.unit
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_washing_needs_no_data_for_an_absent_species(data_path, tmp_path,
                                                     dynamic):
    """An absent api needs no diffusivity and is washed at exactly zero.

    The concentrations, the normalized profile, the retained and effluent
    concentrations and the outlet equal those with api's full data and,
    for the present species, those without api. Earlier api needed its
    diffusivity and, with it, got a data-dependent normalized profile.
    """
    runs = {}
    for label, record in (("bare", API_BARE), ("full", API_SEPARATION_FULL),
                          ("none", None)):
        washer = _washer(_separation_database(data_path, tmp_path, record,
                                              label))
        outputs = washer.solve_unit(WASH_PRESSURE, wash_ratio=WASH_RATIO,
                                    dynamic=dynamic)
        runs[label] = (washer, outputs)
    washer, outputs = runs["bare"]

    for actual, full, reference in zip(outputs, runs["full"][1],
                                       runs["none"][1]):
        np.testing.assert_allclose(actual, full, rtol=ROUNDING, atol=0)
        np.testing.assert_array_equal(actual[..., API_SEPARATION_INDEX], 0.0)
        np.testing.assert_allclose(actual[..., :API_SEPARATION_INDEX],
                                   reference, rtol=ROUNDING, atol=0)
    np.testing.assert_array_equal(washer.concProf[..., API_SEPARATION_INDEX],
                                  0.0)
    np.testing.assert_array_equal(washer.result.mass_conc["api"], 0.0)
    assert washer.Outlet.Liquid_1.mass_frac[API_SEPARATION_INDEX] == 0.0
    np.testing.assert_allclose(
        washer.Outlet.Liquid_1.mass_frac[:API_SEPARATION_INDEX],
        runs["none"][0].Outlet.Liquid_1.mass_frac, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_washing_names_a_wash_solvent_without_density(data_path, tmp_path):
    """The wash liquid supplies the solvent, which then needs ``rho_liq``.

    The cake liquid holds no solvent, so building it needs no solvent
    density; the wash does. Earlier the missing density made the solvent's
    concentrations NaN, reported only as a non-finite retained inventory.
    """
    path = _separation_database(data_path, tmp_path,
                                solvent_density=False)
    washer = _washer(path, fractions=[0.25, 0.25, 0.25, 0.25, 0.0])
    with pytest.raises(MissingPropertyError,
                       match=r"\['solvent'\].*'rho_liq'"):
        washer.solve_unit(WASH_PRESSURE, wash_ratio=WASH_RATIO,
                          dynamic=False)


@pytest.mark.unit
def test_washing_names_a_present_species_without_diffusivity(data_path,
                                                             tmp_path):
    """api in the cake liquid needs its diffusivity, which it lacks."""
    path = _separation_database(data_path, tmp_path, API_SEPARATION_FULL)
    database = json.loads(open(path).read())
    del database["api"]["diffusivity"]
    path = _write(tmp_path, database, "no_diffusivity")
    washer = DisplacementWashing(solvent_idx=SOLVENT_INDEX,
                                 num_nodes=WASH_NODES,
                                 diam_unit=STATION_DIAMETER)
    liquid = LiquidPhase(path, mass_frac=[0.1, 0.1, 0.1, 0.1, 0.55, 0.05],
                         vol=LIQUID_VOLUME)  # [-], [m**3]
    solid = SolidPhase(path, mass_frac=SEPARATION_SOLID + [0.0],
                       x_distrib=GRID, distrib=POPULATION)
    washer.Phases = [liquid, solid]
    with pytest.raises(MissingPropertyError,
                       match=r"\['api'\].*'diffusivity'"):
        washer.solve_unit(WASH_PRESSURE, wash_ratio=WASH_RATIO,
                          dynamic=False)


@pytest.mark.unit
@pytest.mark.parametrize("api", ["bare", "full"])
def test_washing_presence_follows_the_initial_cake_field(data_path, tmp_path,
                                                         api):
    """api only in one cell of the cake's field is present.

    The liquid holds no api, but the cake's concentration field
    (``mass_concentr``, which the washing remaps) holds 1 kg/m**3 of it in
    the second node: without data its diffusivity is named; with data it is
    washed node by node, nonzero only in that node, where the profile is
    its normalized concentration times its initial concentration.
    """
    record = API_BARE if api == "bare" else API_SEPARATION_FULL
    washer = _washer(_separation_database(data_path, tmp_path, record, api))
    field = np.tile(washer.Liquid_1.mass_conc, (WASH_NODES, 1))  # [kg/m**3]
    field[1, API_SEPARATION_INDEX] = CAKE_FIELD_API  # [kg/m**3]
    washer.CakePhase.mass_concentr = field
    assert washer.Liquid_1.mass_frac[API_SEPARATION_INDEX] == 0.0
    if api == "bare":
        with pytest.raises(MissingPropertyError,
                           match=r"\['api'\].*'diffusivity'"):
            washer.solve_unit(WASH_PRESSURE, wash_ratio=WASH_RATIO,
                              dynamic=False)
        return

    conc, normalized, retained, _ = washer.solve_unit(
        WASH_PRESSURE, wash_ratio=WASH_RATIO, dynamic=False)

    api_profile = conc[:, API_SEPARATION_INDEX]  # [kg/m**3]
    np.testing.assert_array_equal(api_profile[[0, 2, 3]], 0.0)
    assert 0.0 < normalized[1, API_SEPARATION_INDEX] <= 1.0
    assert api_profile[1] == pytest.approx(
        normalized[1, API_SEPARATION_INDEX] * CAKE_FIELD_API, rel=ROUNDING)
    assert retained[API_SEPARATION_INDEX] > 0.0


@pytest.mark.unit
def test_deliquoring_initial_state_needs_no_density_of_an_absent_species(
        data_path, tmp_path):
    """api absent from the cake needs no ``rho_liq`` and gets no contrast.

    Its pure-density entry is the 0.0 placeholder, whatever its data, and
    the reduced initial state equals that of the cake without api. Earlier
    the pure density of every species was required.
    """
    runs = {}
    for label, record in (("bare", API_BARE), ("full", API_SEPARATION_FULL),
                          ("none", None)):
        unit = _deliquoring(_separation_database(data_path, tmp_path, record,
                                                 label))
        runs[label] = (unit, unit.initialize_states().reshape(
            DELIQUORING_NODES, -1))
    unit, states = runs["bare"]
    reference, reference_states = runs["none"]

    np.testing.assert_array_equal(states, runs["full"][1])
    assert unit.rho_j[API_SEPARATION_INDEX] == 0.0
    assert runs["full"][0].rho_j[API_SEPARATION_INDEX] == 0.0
    np.testing.assert_allclose(unit.rho_j[:API_SEPARATION_INDEX],
                               reference.rho_j, rtol=ROUNDING, atol=0)
    np.testing.assert_array_equal(states[:, 1 + API_SEPARATION_INDEX], 0.0)
    np.testing.assert_allclose(states[:, :1 + API_SEPARATION_INDEX],
                               reference_states, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_deliquoring_names_a_cake_species_without_density(data_path,
                                                          tmp_path):
    """Presence follows the cake's concentration field.

    api is absent from the liquid but present in the cake's field, so its
    reduction needs its pure density, which it lacks.
    """
    unit = _deliquoring(_separation_database(data_path, tmp_path, API_BARE))
    field = np.tile(unit.Liquid_1.mass_conc, (DELIQUORING_NODES, 1))  # [kg/m**3]
    field[1, API_SEPARATION_INDEX] = 1.0  # [kg/m**3], api in the middle cell
    unit.CakePhase.mass_concentr = field
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'rho_liq'"):
        unit.initialize_states()


@pytest.mark.unit
def test_deliquoring_publishes_an_absent_species_at_zero(data_path,
                                                         tmp_path):
    """Round-off in api's reduced state publishes exactly zero.

    With api's full data, its zero contrast makes its reconstructed
    concentration exactly zero in every published field; the present
    species equal the step without api. Earlier the contrast was api's
    pure density, which turned round-off into a nonzero concentration.
    """
    runs = {}
    for label, record in (("full", API_SEPARATION_FULL), ("none", None)):
        unit = _deliquoring(_separation_database(data_path, tmp_path, record,
                                                 label))
        initial = unit.initialize_states().reshape(DELIQUORING_NODES, -1)
        final = initial.copy()
        final[:, 0] = [0.4, 0.6, 0.8]  # [-], drained reduced saturation
        if record is not None:
            final[:, 1 + API_SEPARATION_INDEX] = ROUND_OFF  # [-]
        unit.theta_conv = THETA_CONV  # [1/s]
        unit.rho_s = unit.Solid_1.getDensity()  # [kg/m**3]
        unit.retrieve_results(np.array([0.0, 1.0]),
                              np.vstack((initial.ravel(), final.ravel())))
        runs[label] = unit
    unit, reference = runs["full"], runs["none"]

    np.testing.assert_array_equal(
        unit.Outlet.mass_concentr[:, API_SEPARATION_INDEX], 0.0)
    np.testing.assert_array_equal(unit.result.mass_conc["api"], 0.0)
    np.testing.assert_array_equal(unit.massjPerMassCake["api"], 0.0)
    assert unit.Outlet.Liquid_1.mass_frac[API_SEPARATION_INDEX] == 0.0
    assert unit.liquid_removed_species[API_SEPARATION_INDEX] == 0.0
    np.testing.assert_allclose(
        unit.Outlet.mass_concentr[:, :API_SEPARATION_INDEX],
        reference.Outlet.mass_concentr, rtol=ROUNDING, atol=0)
    np.testing.assert_allclose(
        unit.Outlet.Liquid_1.mass_frac[:API_SEPARATION_INDEX],
        reference.Outlet.Liquid_1.mass_frac, rtol=ROUNDING, atol=0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_deliquoring_integrates_without_data_for_an_absent_species(
        data_path, tmp_path):
    """A deliquoring step with an absent api integrates without its data.

    The run equals the run with api's full data and, for the present
    species, the run without api to integration tolerance; api is
    published at zero. Earlier api needed its ``rho_liq``.
    """
    pytest.importorskip("assimulo")
    runs = {}
    for label, record in (("bare", API_BARE), ("full", API_SEPARATION_FULL),
                          ("none", None)):
        unit = DeliquoringStep(num_nodes=DELIQUORING_NODES,
                               diam_unit=STATION_DIAMETER)
        unit.Phases = _separation_phases(
            _separation_database(data_path, tmp_path, record, label))
        theta, states = unit.solve_unit(DELIQUORING_PRESSURE,
                                        DELIQUORING_RUNTIME, verbose=False)
        runs[label] = (unit, states)
    unit, states = runs["bare"]
    reference = runs["none"][0]

    np.testing.assert_array_equal(states, runs["full"][1])
    np.testing.assert_array_equal(
        unit.Outlet.mass_concentr[:, API_SEPARATION_INDEX], 0.0)
    assert unit.Outlet.Liquid_1.mass_frac[API_SEPARATION_INDEX] == 0.0
    np.testing.assert_allclose(
        unit.Outlet.mass_concentr[:, :API_SEPARATION_INDEX],
        reference.Outlet.mass_concentr, rtol=DELIQUORING_TRAJECTORY_RTOL)
    np.testing.assert_allclose(unit.satProf[-1], reference.satProf[-1],
                               rtol=DELIQUORING_TRAJECTORY_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_filter_needs_no_data_for_an_absent_species(data_path, tmp_path):
    """The filter integrates masses only and reads the stored liquid.

    Its density, viscosity and surface tension follow the liquid's
    fractions, so an absent api already needed no data; this guards that
    contract: the filtrate history equals the filter without api, and the
    cake liquid holds api at zero.
    """
    pytest.importorskip("assimulo")
    runs = {}
    for label, record in (("bare", API_BARE), ("none", None)):
        unit = Filter(station_diam=STATION_DIAMETER)
        unit.Phases = _separation_phases(
            _separation_database(data_path, tmp_path, record, label))
        runs[label] = unit.solve_unit(deltaP=FILTER_PRESSURE, verbose=False)
        runs[label + "_unit"] = unit

    np.testing.assert_allclose(runs["bare"][1], runs["none"][1],
                               rtol=ROUNDING, atol=0)
    assert runs["bare_unit"].Outlet.Liquid_1.mass_frac[
        API_SEPARATION_INDEX] == 0.0
