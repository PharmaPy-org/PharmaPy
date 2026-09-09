"""B011 balance and washing regressions with real phases and no ODE backend.

Synthetic transport probes isolate units; the five-species shipped database
supplies separation properties. Drying uses the existing synthetic vapor table.
"""

import json
import re

import numpy as np
import pytest
from scipy.special import erfc

from PharmaPy.Drying_Model import Drying
from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.SolidLiquidSep import DisplacementWashing, Filter
import PharmaPy.SolidLiquidSep as separation
from PharmaPy.Streams import VaporStream
from test_phases_vapor_latent_heat_shape import THERMO_THREE_SPECIES

pytestmark = pytest.mark.unit

RTOL = 1e-12  # [-], roundoff allowance for short float64 balances
COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], solvent-rich cake
GRID = np.array([100., 200., 300.])  # [um], uniform coarse crystals
POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um], asymmetric size weighting


@pytest.fixture
def separation_phases(data_path):
    """Construct a dilute, litre-scale slurry from the shipped database.

    Parameters
    ----------
    data_path : dict
        Repository data directories.

    Returns
    -------
    tuple
        Real liquid and solid phases; liquid volume is 1 L.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    liquid = LiquidPhase(path, mass_frac=COMPOSITION, vol=1e-3)  # [m**3]
    solid = SolidPhase(path, mass_frac=[1, 0, 0, 0, 0],
                       x_distrib=GRID, distrib=POPULATION)
    return liquid, solid


@pytest.fixture
def dryer(tmp_path):
    """Build a real dryer for two volatile species and a final carrier.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-local directory for the existing synthetic property table.

    Returns
    -------
    Drying
        Three-node balance probe with prescribed porosity and liquid density.
    """
    path = tmp_path / 'thermo.json'
    properties = {name: {**values, 'visc_gas': 2e-5}
                  for name, values in THERMO_THREE_SPECIES.items()}  # visc_gas [Pa*s], synthetic dilute gas
    path.write_text(json.dumps(properties))
    unit = Drying(number_nodes=3, supercrit_names=['medium'])
    unit.Liquid_1 = LiquidPhase(str(path), mass_frac=[0.4, 0.6, 0.0], mass=1.)
    unit.Vapor_1 = VaporPhase(str(path), mass_frac=[0.1, 0.2, 0.7], mass=1.)
    unit.Inlet = VaporStream(str(path), mass_frac=[0., 0., 1.], temp=300., mass_flow=1.)
    unit.idx_volatiles = [0, 1]
    unit.idx_supercrit = [2]
    unit.porosity = 0.4  # [-], prescribed porous bed
    unit.rho_liq = np.full(3, 900.)  # [kg/m**3], constant-density liquid
    unit.rho_sol = 1200.  # [kg/m**3], prescribed solid density
    unit.cp_sol = 1000.  # [J/kg/K], prescribed solid capacity
    unit.h_T_j = 0.  # [W/m**2/K], isolate convection
    unit.a_V = 1.  # [m**2/m**3], immaterial with zero heat transfer
    unit.dz = np.ones(3)  # [m], unit-length cells
    return unit


def test_drying_rhs_hands_mass_source_and_velocity_to_balances(dryer):
    """Exercise real property, inlet, drying-rate and balance collaborators.

    Notes
    -----
    Provisional #42 fixture: the overwritten first volatile is already zero,
    and the carrier is last. Extend to mixed liquids and permuted carriers
    after #42 lands; this does not validate the deferred volatile-state bug.
    """
    dryer.num_volatiles = 2
    dryer.states_in_dict = {'Inlet': {'temp': 1, 'mass_frac': 3}}
    dryer.s_inf = 0.1  # [-], prescribed irreducible saturation
    dryer.k_perm = 1e-12  # [m**2], porous-bed probe
    dryer.dPg_dz = 1000.  # [Pa/m], weak Darcy flow
    dryer.pres_gas = np.full(3, 1e5)  # [Pa], isobaric property probe
    states = np.tile([0.5, 0., 0., 1., 0., 1., 320., 320.], (3, 1))  # S,y,x [-]; T [K]
    derivative = dryer.unit_model(0., states.ravel()).reshape(3, -1)
    # Pure volatile liquid: its equilibrium mole fraction is Psat/P.
    vapor_pressure = 10**(8. - 1800. / (320. - 40.))  # [Pa], table's heavy-species Antoine equation
    mass_source = dryer.k_y * dryer.a_V * vapor_pressure / 1e5 * 0.1  # [kg/m**3/s], MW=0.1 kg/mol
    density = 1e5 * 0.046 / (8.314 * 320.)  # [kg/m**3], pure medium carrier
    gas_holdup = density * dryer.porosity / 2  # [kg/m**3], half-saturated bed
    np.testing.assert_allclose(dryer.dry_rate[:, 1], mass_source, rtol=RTOL)
    np.testing.assert_allclose(derivative[:, 1:4],
                               np.tile([0., mass_source / gas_holdup,
                                        -mass_source / gas_holdup], (3, 1)), rtol=RTOL, atol=RTOL)
    # Turning off the pressure gradient removes only gas convection at this
    # fixed state. Independent Darcy and thermal calculations predict its size.
    reduced_saturation = (0.5 - 0.1) / (1 - 0.1)  # [-]
    relative_permeability = (1 - reduced_saturation)**2 * (1 - reduced_saturation**1.4)  # [-], existing Darcy model
    velocity = 1e-12 * relative_permeability * 1000. / 2e-5  # [m/s]
    dryer.dPg_dz = 0.  # [Pa/m]
    no_flow = dryer.unit_model(0., states.ravel()).reshape(3, -1)
    expected_convection = -velocity / (dryer.porosity * 0.5) * 20. * 60. / (60. - 8.314)  # [K/s], molar cp/cv ratio
    np.testing.assert_allclose(derivative[:, -2] - no_flow[:, -2],
                               [expected_convection, 0., 0.], rtol=RTOL, atol=RTOL)


# Porosity and saturation probes distinguish superficial from pore velocity [-].
@pytest.mark.parametrize('porosity', [0.3, 0.6])
@pytest.mark.parametrize('saturation', [0.2, 0.7])
@pytest.mark.parametrize('density', [0.5, 2.0])
def test_gas_heat_convection_has_no_residual_density(dryer, density, porosity, saturation):
    # A constant interior temperature and different inlet isolate the first
    # face jump: the limiter is zero on both sides of that jump.
    dryer.porosity = porosity  # [-]
    temperature = np.full(3, 320.)  # [K], uniform bed
    inlet_temperature = 300.  # [K], colder inlet
    velocity = np.full(3, 0.02)  # [m/s], prescribed existing velocity basis
    gas_fraction = np.tile(dryer.Vapor_1.mass_frac, (3, 1))  # [-]
    gas_rate, _ = dryer.energy_balance(
        0., temperature, temperature, np.full(3, saturation), gas_fraction,
        np.tile([0.4, 0.6], (3, 1)), velocity, np.full(3, density),
        np.zeros((3, 3)), {'temp': inlet_temperature})
    # Cp and R are independently mass-weighted from the synthetic molar table.
    cp = sum(w * props['cp_vapor'][0] * 1000 / props['mw']
             for w, props in zip(gas_fraction[0], THERMO_THREE_SPECIES.values()))  # [J/kg/K]
    gas_constant = sum(w * 8.314 * 1000 / props['mw']
                       for w, props in zip(gas_fraction[0], THERMO_THREE_SPECIES.values()))  # [J/kg/K]
    epsilon_gas = porosity * (1 - saturation)  # [-], gas volume per bed volume
    expected = [-velocity[0] / epsilon_gas * (temperature[0] - inlet_temperature)
                * cp / (cp - gas_constant), 0., 0.]  # [K/s]
    np.testing.assert_allclose(gas_rate, expected, rtol=RTOL, atol=0)


@pytest.mark.parametrize('transfer', [False, True])
@pytest.mark.parametrize('convection', [False, True])
def test_gas_fractions_follow_component_and_total_accumulation(dryer, transfer, convection):
    saturation = np.array([0.3, 0.5, 0.7])  # [-], unequal gas holdups
    density = np.array([0.8, 1.2, 1.5])  # [kg/m**3]
    gas_fraction = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5],
                             [0.4, 0.1, 0.5]])  # [-], normalized by node
    inlet = np.array([0.05, 0.05, 0.90])  # [-], different from first node
    rate = np.array([[0.02, 0.06, 0.], [0.03, 0.07, 0.],
                     [0.05, 0.09, 0.]]) * transfer  # [kg/m**3/s]
    velocity = np.array([0.01, 0.02, 0.03]) * convection  # [m/s]
    _, derivative, _ = dryer.material_balance(
        0., saturation, np.full(3, 300.), np.full(3, 300.), gas_fraction,
        np.tile([0.4, 0.6], (3, 1)), velocity, density, rate,
        {'mass_frac': inlet})
    # Open, constant-pressure pore gas: component balance is
    # d(H*y_i)/dt=F_i+r_i-y_i*V, and total balance dH/dt=sum(F+r)-V.
    # The bulk-composition vent V cancels in the quotient rule. Here F_i
    # uses the code's non-conservative upwind convection, whose species sum
    # vanishes for a normalized field. Thus normalization is preserved, and
    # the source correction restores small positive normalization errors.
    for node in range(3):
        upstream = inlet if node == 0 else gas_fraction[node - 1]  # [-]
        component_accumulation = (density[node] * velocity[node]
                                  * (upstream - gas_fraction[node]) + rate[node])  # [kg/m**3/s]
        total_accumulation = component_accumulation.sum()  # [kg/m**3/s]
        holdup = density[node] * dryer.porosity * (1 - saturation[node])  # [kg/m**3]
        np.testing.assert_allclose(
            holdup * derivative[node] + gas_fraction[node] * total_accumulation,
            component_accumulation, rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(derivative.sum(axis=1), 0., atol=RTOL)


@pytest.mark.parametrize('population_scale', [1., 7.])
def test_washing_diffusivity_uses_volume_weighted_peclet(separation_phases, population_scale):
    """Apply the dispersion correlation once to the volume-weighted Peclet.

    Notes
    -----
    On [100, 200, 400] um, midpoint sizes are [150, 300] um and widths
    are [100, 200] um. Both midpoint populations are 1.5e4 #/um times the
    population scale. The midpoint approximation to integral(kv*L**3*n*dL)
    gives bin volumes in the ratio 100*150**3 : 200*300**3 = 1 : 16.
    Thus volume fractions are [1/17, 16/17], independent of kv and scale.
    Cake height is held at 0.05 m to isolate weighting from geometry changes.
    At u=1e-4 m/s, bin Peclet numbers are [3/8, 3/4] for D=4e-8 m**2/s,
    and [3/2, 3] for D=1e-8 m**2/s. The volume-weighted averages are
    99/136 and 99/34, respectively, spanning the Re*Sc=1 threshold.
    These use Destro's thesis, section 4.3.5, Eq. 4.27; Eq. 4.26 is then
    applied once per species, rather than averaged after its nonlinear power.
    """
    liquid, solid = separation_phases
    size_grid = np.array([100., 200., 400.])  # [um], unequal integration widths
    solid.updatePhase(x_distrib=size_grid, distrib=POPULATION * population_scale)
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = [liquid, solid]
    velocity = 1e-4  # [m/s], gives Peclet numbers below and above unity
    diffusivity = np.array([4e-8, 1e-8])  # [m**2/s], transport probes
    peclet = np.array([99/136, 99/34])  # [-], independent bin-volume derivation above
    assert peclet[0] < 1 < peclet[1]
    expected_ratio = np.array([1 / np.sqrt(2),
                               1 / np.sqrt(2) + 55.5 * peclet[1]**0.96])  # [-]
    height = 0.05  # [m], fixed thin cake for population-scale invariance
    np.testing.assert_allclose(
        washer.get_diffusivity(velocity, diffusivity, cake_height=height),
        diffusivity * expected_ratio, rtol=RTOL)


@pytest.mark.parametrize('dynamic', [False, True])
def test_washing_public_solve_preserves_spatial_species_values(separation_phases, dynamic):
    """Preserve the analytical spatial profile and its attached bulk inventory.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases with shipped thermophysical data.
    dynamic : bool
        Select time-dependent or final-only analytical washing results.
    """
    liquid, solid = separation_phases
    # Deliberately large synthetic diffusivities avoid the separate large-Pe
    # exponential overflow problem and expose all five distinct columns.
    diffusivity = np.arange(1, 6) * 1e-4  # [m**2/s], synthetic transport probe
    liquid.diffusivity = np.tile(diffusivity[:, None], (1, 5))  # [m**2/s]
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = [liquid, solid]
    initial = liquid.mass_conc.copy()  # [kg/m**3]
    pressure = 1.  # [Pa], low-flow analytical probe
    ratio = 0.5  # [-], half a pore displacement
    height = washer.CakePhase.cake_vol / washer.cross_area  # [m]
    porosity = washer.CakePhase.porosity  # [-], packing that owns volume and resistance
    velocity = pressure / np.mean(liquid.getViscosity()) / (
        washer.CakePhase.alpha * solid.getDensity() * height * (1 - porosity)
        + washer.resist_medium)  # [m/s]
    effective = diffusivity / np.sqrt(2)  # [m**2/s], Peclet < 1
    assert np.all(velocity * 250e-6 / diffusivity < 1)  # largest midpoint [m]
    positions = np.linspace(0, height, 4)  # [m]
    inlet = np.zeros(5)  # [kg/m**3]
    inlet[4] = liquid.rho_liq[4]
    expected = np.empty((4, 5))  # [kg/m**3]
    for node, position in enumerate(positions):
        for species in range(5):
            scale = np.sqrt(velocity * height / effective[species])  # [-]
            first = (position / height - ratio) / (2 * np.sqrt(ratio)) * scale  # [-]
            second = (position / height + ratio) / (2 * np.sqrt(ratio)) * scale  # [-]
            fraction = 1 - (erfc(first) + np.exp(velocity * position / effective[species])
                            * erfc(second)) / 2  # [-], Lapidus-Amundson solution
            expected[node, species] = fraction * (initial[species] - inlet[species]) + inlet[species]
    concentration, _, _, _ = washer.solve_unit(pressure, wash_ratio=ratio, dynamic=dynamic)
    np.testing.assert_allclose(concentration, expected, rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(washer.Outlet.mass_concentr, expected, rtol=RTOL, atol=RTOL)
    # Endpoint-centered control volumes have widths L/6, L/3, L/3, L/6.
    species_mass = washer.CakePhase.porosity * washer.CakePhase.cake_vol * (
        expected[0] + 2 * expected[1] + 2 * expected[2] + expected[3]) / 6  # [kg]
    assert washer.Outlet.Liquid_1.mass_frac.shape == (5,)
    np.testing.assert_allclose(washer.Outlet.Liquid_1.mass_frac,
                               species_mass / species_mass.sum(), rtol=RTOL, atol=RTOL)
    assert washer.Outlet.Liquid_1.mass == pytest.approx(species_mass.sum(), rel=RTOL)
    np.testing.assert_allclose(washer.concProf[:, -1, :], expected, rtol=RTOL, atol=RTOL)
    assert washer.timeProf[-1] == pytest.approx(ratio * height / velocity, rel=RTOL)
    if not dynamic:
        assert washer.concProf.shape == (4, 1, 5)
        assert len(washer.result.time) == 1


@pytest.mark.parametrize('fraction', [0.5, 1.0000031])  # [-], partial recovery and observed solver overshoot
def test_filter_retrieval_scales_total_population_to_recovered_mass(
        separation_phases, fraction):
    liquid, solid = separation_phases
    unit = Filter(station_diam=0.1)
    unit.Phases = [liquid, solid]
    density = liquid.getDensity()  # [kg/m**3]
    recovered_mass = solid.mass * fraction  # [kg], nominal or overshot solver recovery
    expected_mass = solid.mass * min(fraction, 1.)  # [kg], available solid inventory
    filtrate_mass = liquid.mass / 2  # [kg], independent output profile
    unit.c_solids = recovered_mass * density / filtrate_mass  # [kg/m**3 filtrate]
    unit.mass_crit = filtrate_mass / fraction  # [kg], full-recovery filtrate mass
    unit.retrieve_results(np.array([0., 1.]),
                          np.array([[0., liquid.mass], [filtrate_mass, filtrate_mass]]),
                          density, solid.getDensity(), solid.getPorosity(), mass_solids=solid.mass)
    np.testing.assert_allclose(unit.Outlet.Solid_1.distrib, POPULATION * min(fraction, 1.), rtol=RTOL)
    np.testing.assert_allclose(unit.outputs[0, -len(POPULATION):], POPULATION * min(fraction, 1.), rtol=RTOL)
    assert unit.Outlet.Solid_1.mass == pytest.approx(expected_mass, rel=RTOL)
    assert unit.Outlet.Solid_1.kv * unit.Outlet.Solid_1.moments[3] * solid.getDensity() == pytest.approx(expected_mass, rel=RTOL)
    np.testing.assert_array_equal(solid.distrib, POPULATION)


def test_filter_rejects_slurry_without_drainable_liquid(separation_phases):
    liquid, solid = separation_phases
    liquid.updatePhase(vol=solid.vol / 10)  # [m**3], below saturated pore volume
    unit = Filter(station_diam=0.1)
    unit.Phases = [liquid, solid]
    original_params = unit.params
    pressure = 2e5  # [Pa], distinct from the initial unset pressure
    slurry_liquid_volume = liquid.vol  # [m**3]
    cake_liquid_volume = solid.vol * solid.getPorosity(diam_filter=0.1) / (
        1 - solid.getPorosity(diam_filter=0.1))  # [m**3]
    with pytest.raises(ValueError, match='liquid.*saturate.*cake') as error:
        unit.solve_unit(runtime=1., deltaP=pressure, model_params=[2e11, 3e9], verbose=False)
    reported_volumes = [float(value) for value in re.findall(
        r"liquid volume=(\S+) m\*\*3", str(error.value))]  # [m**3]
    np.testing.assert_allclose(reported_volumes,
                               [slurry_liquid_volume, cake_liquid_volume], rtol=RTOL)
    assert unit.deltaP is None
    assert unit.params == original_params


def test_filter_rejects_zero_solid_feed_before_integration(separation_phases):
    liquid, solid = separation_phases
    solid.updatePhase(distrib=np.zeros_like(POPULATION))  # [#/um], pure-liquid feed
    unit = Filter(station_diam=0.1, alpha=1e11)  # [m/kg], avoids estimating an empty cake
    unit.Phases = [liquid, solid]
    original_params = unit.params
    with pytest.raises(ValueError, match='positive solid mass.*zero-solid'):
        unit.solve_unit(runtime=0.01, verbose=False)
    assert unit.deltaP is None
    assert unit.params == original_params
    np.testing.assert_array_equal(solid.distrib, np.zeros_like(POPULATION))


def test_static_washing_rejects_time_grid(separation_phases):
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = separation_phases
    with pytest.raises(ValueError, match='time_vals.*dynamic=False'):
        washer.solve_unit(deltaP=1., dynamic=False, time_vals=[0.01, 0.02])  # [Pa], [s]


@pytest.mark.parametrize('log_params', [False, True])
def test_filter_seed_matches_estimation_space(separation_phases, log_params):
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=log_params)  # [m/kg]
    unit.Phases = separation_phases
    physical = np.array([1e11, 1e9])  # [m/kg, 1/m], constructor resistances
    expected = np.log(physical) if log_params else physical  # log SI values or [m/kg, 1/m]
    np.testing.assert_allclose(unit.param_seed, expected, rtol=RTOL)
    np.testing.assert_allclose(unit.params, physical, rtol=RTOL)
    unit.params = (2e11, 3e9)  # [m/kg, 1/m], a prior optimizer evaluation
    np.testing.assert_allclose(unit.param_seed, expected, rtol=RTOL)


@pytest.mark.parametrize('height', [0.05, 0.10, 0.20])  # [m], below/at/above the source limit
@pytest.mark.parametrize('peclet', [0.5, 2.])  # [-], strictly inside each ReSc branch
def test_washing_dispersion_uses_cake_height(separation_phases, height, peclet):
    """Use mean size 8825/38 um and hand-derived ReSc=2 branch ratios.

    Notes
    -----
    The fixture grid [100,200,300] um has midpoint sizes [150,250] um,
    equal widths and equal midpoint populations. Volume weights are
    [27/152,125/152], so the weighted size is 35300/152 = 8825/38 um.
    At ReSc=2, Eq. 4.26 gives 1/sqrt(2)+55.5*2**0.96 for a thin cake
    and 1/sqrt(2)+3.5 for a thick cake. At the stated 0.10 m boundary,
    the thin branch is retained. ReSc=0.5 probes the diffusion-only branch.
    """
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = separation_phases
    diffusivity = np.array([1e-8])  # [m**2/s], transport probe
    mean_size = 8825 / 38 * 1e-6  # [m], volume-weighted midpoint calculation above
    velocity = peclet * diffusivity[0] / mean_size  # [m/s], imposed Peclet
    # Fraction arithmetic: (27*150 + 125*250)/152 = 35300/152.
    expected_ratio = 1 / np.sqrt(2)  # [-], source diffusion-only ratio
    if peclet > 1:
        expected_ratio += 3.5 if height > 0.1 else 55.5 * 2**0.96  # [-], ReSc=2
    actual = washer.get_diffusivity(velocity, diffusivity, cake_height=height)
    np.testing.assert_allclose(actual, diffusivity * expected_ratio, rtol=RTOL)


def test_washing_rejects_empty_population(separation_phases):
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=0.1)
    washer.Phases = separation_phases
    washer.Solid_1.updatePhase(distrib=np.zeros_like(POPULATION))  # [#/um]
    with pytest.raises(ValueError, match='nonzero particle population'):
        washer.get_diffusivity(1e-4, np.array([1e-8]))  # [m/s], [m**2/s]



@pytest.mark.parametrize('height', [0.05, 0.20])
def test_washing_public_solve_uses_height_dependent_dispersion(separation_phases, height):
    """Check the real washing profile at Re*Sc=2 across the height threshold.

    Notes
    -----
    The volume-weighted size is 8825/38 um on GRID (see the bin-volume
    derivation in test_washing_dispersion_uses_cake_height). Molecular
    diffusivity sets Re*Sc=2. Eq. 4.26 gives D_ax/D=1/sqrt(2)+3.5 for
    0.20 m, and 1/sqrt(2)+55.5*2**0.96 for 0.05 m. The independently
    evaluated Lapidus-Amundson solution checks the full solve handoff.
    """
    liquid, solid = separation_phases
    cake = Cake()
    cake.Phases = [liquid, solid]
    diameter = np.sqrt(4 * cake.cake_vol / (np.pi * height))  # [m], prescribed height
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=diameter)
    washer.Phases = cake
    pressure = 1.  # [Pa], low-flow analytical probe
    porosity = cake.porosity  # [-], packing that owns volume and resistance
    velocity = pressure / np.mean(liquid.getViscosity()) / (
        cake.alpha * solid.getDensity() * height * (1 - porosity)
        + washer.resist_medium)  # [m/s], Darcy flow
    mean_size = 8825 / 38 * 1e-6  # [m], independently derived volume average
    diffusivity = velocity * mean_size / 2  # [m**2/s], fixes Re*Sc=2
    liquid.diffusivity = np.full((5, 5), diffusivity)  # [m**2/s], common transport probe
    dispersion_ratio = 1 / np.sqrt(2) + (3.5 if height > 0.1 else 55.5 * 2**0.96)  # [-]
    effective = diffusivity * dispersion_ratio  # [m**2/s]
    wash_ratio = 0.5  # [-], half a pore displacement
    positions = np.linspace(0, height, washer.num_nodes)  # [m]
    scale = np.sqrt(velocity * height / effective)  # [-]
    first = (positions / height - wash_ratio) / (2 * np.sqrt(wash_ratio)) * scale  # [-]
    second = (positions / height + wash_ratio) / (2 * np.sqrt(wash_ratio)) * scale  # [-]
    fraction = 1 - (erfc(first) + np.exp(velocity * positions / effective)
                    * erfc(second)) / 2  # [-], Lapidus-Amundson solution
    inlet = np.zeros(5)  # [kg/m**3]
    inlet[4] = liquid.rho_liq[4]
    expected = fraction[:, None] * (liquid.mass_conc - inlet) + inlet  # [kg/m**3]
    actual, _, _, _ = washer.solve_unit(pressure, wash_ratio=wash_ratio, dynamic=False)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL)
    assert washer.cake_height == pytest.approx(height, rel=RTOL)


@pytest.mark.parametrize('resistance_name', ['alpha', 'r_medium'])
@pytest.mark.parametrize('invalid', [np.nan, np.inf])
@pytest.mark.parametrize('log_params', [False, True])
def test_filter_seed_rejects_invalid_configured_resistances(
        separation_phases, resistance_name, invalid, log_params):
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=log_params)  # [m/kg]
    unit.Phases = separation_phases
    setattr(unit, resistance_name, invalid)  # [m/kg] or [1/m], invalid seed input
    with np.errstate(all='raise'):
        with pytest.raises(ValueError, match='finite'):
            unit.param_seed


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('height', [0.15, 1.5])  # [m], Pe about 1228 and 12280
def test_washing_high_peclet_profile_is_finite(separation_phases, dynamic, height):
    """Exercise the overflow reproduction through both public washing modes.

    Notes
    -----
    The grid [25,50,75] um with [1e8,2e8,1e8] #/um has mean size
    8825/152 um, one quarter of the coarse fixture's volume-weighted size.
    At ReSc=2 and D=1e-9 m**2/s, D_ax/D=1/sqrt(2)+3.5. For a 0.15 m
    cake, u*height/D_ax is about 1228; the 1.5 m cake raises it to 12280.
    Both exceed the naive exponential limit and probe vanishing tails.
    A half displacement leaves the far end unchanged and washes the inlet.
    """
    liquid, solid = separation_phases
    solid.updatePhase(x_distrib=np.array([25., 50., 75.]),
                       distrib=np.array([1e8, 2e8, 1e8]))  # [um], [#/um]
    cake = Cake()
    cake.Phases = [liquid, solid]
    diameter = np.sqrt(4 * cake.cake_vol / (np.pi * height))  # [m]
    washer = DisplacementWashing(solvent_idx=4, num_nodes=4, diam_unit=diameter)
    washer.Phases = cake
    diffusivity = 1e-9  # [m**2/s], molecular diffusion
    liquid.diffusivity = np.full((5, 5), diffusivity)  # [m**2/s]
    mean_size = 8825 / 152 * 1e-6  # [m], independently derived average above
    velocity = 2 * diffusivity / mean_size  # [m/s], imposes ReSc=2
    porosity = cake.porosity  # [-], packing that owns volume and resistance
    pressure = velocity * np.mean(liquid.getViscosity()) * (
        cake.alpha * solid.getDensity() * height * (1 - porosity)
        + washer.resist_medium)  # [Pa], inversion of Darcy's law
    initial = liquid.mass_conc.copy()  # [kg/m**3]
    inlet = np.zeros(5)  # [kg/m**3]
    inlet[4] = liquid.rho_liq[4]
    with np.errstate(all='raise'):
        outputs = washer.solve_unit(pressure, wash_ratio=0.5, dynamic=dynamic)
    for output in outputs:
        assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(washer.concProf))
    np.testing.assert_allclose(outputs[0][0], inlet, rtol=RTOL, atol=RTOL)
    np.testing.assert_allclose(outputs[0][-1], initial, rtol=RTOL, atol=RTOL)
    assert np.all(outputs[1] >= -RTOL)
    assert np.all(outputs[1] <= 1 + RTOL)


def test_scaled_erfc_product_matches_finite_naive_expression():
    """Cover both argument signs and broadcasting where direct exp is finite."""
    exponent = np.array([-3., 0., 2.])[:, None]  # [-], moderate exponential range
    argument = np.array([-2., -0.5, 0., 0.5, 2.])[None, :]  # [-], both erfc identities
    expected = np.exp(exponent) * erfc(argument)  # [-], direct independent reference
    with np.errstate(all='raise'):
        actual = separation._exp_erfc(exponent, argument)
    roundoff = 8 * np.finfo(float).eps  # [-], allowance for the short identity evaluation
    np.testing.assert_allclose(actual, expected, rtol=roundoff)


@pytest.mark.parametrize('log_params', [False, True])
def test_filter_estimation_rejects_zero_medium_seed(separation_phases, log_params):
    """Reject the singular zero-filtrate startup through the public seed handoff.

    Parameters
    ----------
    separation_phases : tuple
        Real liquid and solid phases with positive inventories [kg].
    log_params : bool
        Select physical or logarithmic estimation parameters.
    """
    from PharmaPy.SimExec import SimulationExec

    unit = Filter(station_diam=0.1, alpha=1e11, resist_medium=0.,
                  log_params=log_params)  # diameter [m], alpha [m/kg], medium [1/m]
    unit.Phases = separation_phases
    simulation = SimulationExec(unit.Liquid_1.path_data, {'F01': []})
    simulation.F01 = unit
    with np.errstate(all='raise'):
        with pytest.raises(ValueError, match='strictly positive.*zero-filtrate startup'):
            simulation.SetParamEstimation(np.array([0., 0.01]), np.array([0., 0.001]))  # [s], [kg], positive filtration probe
    assert unit.r_medium == 0  # [1/m], constructor compatibility remains


@pytest.mark.parametrize('resistance_name, invalid', [('alpha', 0.), ('alpha', -1.), ('r_medium', -1.)])
@pytest.mark.parametrize('log_params', [False, True])
def test_filter_seed_rejects_unphysical_resistances(
        separation_phases, resistance_name, invalid, log_params):
    """Reject configured resistances that cannot seed a physical filtration run.

    Parameters
    ----------
    separation_phases : tuple
        Real phases with positive liquid and solid inventories [kg].
    resistance_name : str
        Configured cake or medium resistance attribute.
    invalid : float
        Invalid resistance [m/kg] for alpha or [1/m] for medium resistance.
    log_params : bool
        Select physical or logarithmic estimation parameters.
    """
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=log_params)  # [m/kg]
    unit.Phases = separation_phases
    setattr(unit, resistance_name, invalid)  # [m/kg] or [1/m], invalid configured resistance
    with pytest.raises(ValueError, match='alpha.*strictly positive.*medium.*strictly positive'):
        unit.param_seed


@pytest.mark.parametrize('parameters', [
    {'alpha': 0.}, {'alpha': -1.}, {'resist_medium': -1.},
    {'alpha': np.nan}, {'resist_medium': np.inf},
])
@pytest.mark.parametrize('log_params', [False, True])
def test_filter_constructor_rejects_unphysical_resistances(parameters, log_params):
    with pytest.raises(ValueError, match='resistance'):
        Filter(station_diam=0.1, log_params=log_params, **parameters)
