"""B009 mass, population, and energy contracts with real phase collaborators.

The five-species shipped database supplies thermodynamics. No ODE solver is
needed: collector derivatives and instantaneous mixer balances are evaluated
directly. Continuous liquid solve/profile routing remains owned by #220.
"""

import json
import warnings

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from PharmaPy.Containers import DynamicCollector, Mixer
from PharmaPy.Crystallizers import BatchCryst, MSMPR
from PharmaPy.SolidLiquidSep import Filter
from test_crystallizer_heat_duty import make_unit as make_fvm_unit
from test_crystallizer_moment_inventory import inventory_unit
from PharmaPy.MixedPhases import Cake, Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = pytest.mark.unit

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], solvent-rich liquid
HOT_COMPOSITION = np.array([0.2, 0.1, 0.2, 0.1, 0.4])  # [-], less solvent
SOLID_COMPOSITION = [1, 0, 0, 0, 0]  # [-], pure A crystals
GRID = np.array([0., 100., 200., 300.])  # [um], includes the #187 zero-size bin
# Trapezoids: 100 * (1e6 * 100**3 + 1.25e5 * 200**3) * 1e-18 = 2e-4.
POPULATION = np.array([0., 1e6, 1.25e5, 0.])  # [#/um], third moment 2e-4 m**3
KV = 0.5  # [-], non-unit shape distinguishes volume from third moment
COLD = 300.0  # [K], cold charge
HOT = 350.0  # [K], hot charge and non-default enthalpy reference
MASS = 1.0  # [kg], liquid batch charge
FLOW = 1.0  # [kg/s], liquid feed rate
RTOL = 1e-10  # [-], allows root-solver error beyond float64 property roundoff


@pytest.fixture
def path(data_path):
    """Return the shipped property database.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    str
        Five-species database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def test_batch_reference_temperature(path):
    mixer = Mixer(temp_refer=HOT)
    mixer.Inlets = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION,
                                temp=COLD) for _ in range(2)]
    mixer.solve_unit()
    assert mixer.Outlet.temp == pytest.approx(COLD, rel=RTOL)


def test_profile_reference_temperature(path):
    mixer = Mixer(temp_refer=HOT)
    mixer.Inlets = [LiquidStream(path, mass_flow=FLOW, mass_frac=COMPOSITION,
                                 temp=COLD) for _ in range(2)]
    mixer.Liquid_1 = mixer.Inlets[0]
    # Exercise the real dynamic balance; #220 owns full liquid profile routing.
    inputs = {
        'mass_frac': tuple(np.tile(inlet.mass_frac, (3, 1)) for inlet in mixer.Inlets),
        'mass_flow': tuple(np.full(3, inlet.mass_flow) for inlet in mixer.Inlets),
        'temp': tuple(np.full(3, inlet.temp) for inlet in mixer.Inlets),
    }  # composition [-], flow [kg/s], temperature [K]; three profile times
    _, fractions, temperatures = mixer.dynamic_balances(inputs)
    np.testing.assert_allclose(temperatures, COLD, rtol=RTOL)
    np.testing.assert_allclose(fractions, np.tile(COMPOSITION, (3, 1)), rtol=RTOL)


def test_collector_temperature_derivative_mass_cp(path):
    collector = DynamicCollector()
    collector.Inlet = LiquidStream(path, mass_flow=FLOW, mass_frac=COMPOSITION,
                                    temp=HOT)
    collector.Phases = LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION,
                                    temp=COLD)
    with open(path) as database:
        properties = json.load(database)
    # Independently integrate each database Cp polynomial, converting molar
    # coefficients [J/mol/K**(n+1)] to mass coefficients with exact 1000 g/kg.
    mixture_cp = sum(COMPOSITION[index] * Polynomial(species['cp_liq'])
                     * 1000 / species['mw']
                     for index, species in enumerate(properties.values()))
    # [J/kg/K], polynomial in temperature [K]
    enthalpy = mixture_cp.integ()  # [J/kg], arbitrary common integration constant
    expected = FLOW / MASS * (enthalpy(HOT) - enthalpy(COLD)) / mixture_cp(COLD)
    # [K/s], adiabatic tank balance with identical feed/tank compositions
    states = np.r_[COMPOSITION, MASS, COLD]  # [-], [kg], [K]
    derivative = collector.unit_model(0.0, states)  # [1/s], [kg/s], [K/s]
    assert derivative[-1] == pytest.approx(expected, rel=RTOL)


def test_liquid_cp_default_mass_basis(path):
    liquid = LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=COLD)
    assert liquid.getCp() == pytest.approx(liquid.getCp(basis='mass'), rel=RTOL)
    assert liquid.getCp(basis='mole') == pytest.approx(
        liquid.getCp() * liquid.mw_av / 1000, rel=RTOL)


@pytest.mark.parametrize('continuous', [False, True])
@pytest.mark.parametrize('solids_first', [False, True])
def test_public_slurry_mix_preserves_phase_amounts_and_population(
        path, continuous, solids_first):
    """Close asymmetric phase balances for either inlet order and amount basis.

    Parameters
    ----------
    path : str
        Shipped five-species thermodynamic database path.
    continuous : bool
        Use phase flow rates [kg/s] instead of batch inventories [kg].
    solids_first : bool
        Place the slurry before the pure-liquid inlet.
    """
    liquid_class = LiquidStream if continuous else LiquidPhase
    solid_class = SolidStream if continuous else SolidPhase
    quantity = 'mass_flow' if continuous else 'mass'
    liquid = liquid_class(path, mass_frac=HOT_COMPOSITION, temp=HOT,
                          **{quantity: FLOW if continuous else MASS})
    solid = solid_class(path, mass_frac=SOLID_COMPOSITION, temp=COLD,
                        x_distrib=GRID, distrib=POPULATION, kv=KV)
    slurry = SlurryStream() if continuous else Slurry()
    slurry.Phases = [liquid_class(path, mass_frac=COMPOSITION, temp=COLD,
                                  **{quantity: (FLOW if continuous else MASS) / 2}), solid]
    # Explicit flow updates exercise the deleted LiquidStream.mass attribute.
    if continuous:
        liquid.updatePhase(mass_flow=liquid.mass_flow)
        slurry.Liquid_1.updatePhase(mass_flow=slurry.Liquid_1.mass_flow)
        assert not hasattr(slurry.Liquid_1, 'mass')
    liquid_amount = getattr(liquid, quantity) + getattr(slurry.Liquid_1, quantity)
    # [kg/s] for streams, [kg] for batches
    solid_amount = getattr(solid, quantity)  # [kg/s] or [kg]
    inlet_energy = sum(getattr(phase, quantity) * phase.getEnthalpy(basis='mass')
                       for phase in [liquid, *slurry.Phases])  # [J/s] or [J]
    mixer = Mixer()
    mixer.Inlets = [slurry, liquid] if solids_first else [liquid, slurry]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        mixer.solve_unit()
    if not continuous:
        assert not any("'mass', 'moles' and 'vol' are all set to zero" in str(item.message)
                       for item in captured)
    assert getattr(mixer.Liquid_1, quantity) == pytest.approx(liquid_amount, rel=RTOL)
    assert np.isfinite(mixer.Outlet.Solid_1.moments).all()
    assert mixer.oper_mode == ('Continuous' if continuous else 'Batch')
    assert isinstance(mixer.Outlet, SlurryStream if continuous else Slurry)
    assert getattr(mixer.Outlet.Liquid_1, quantity) == pytest.approx(liquid_amount, rel=RTOL)
    assert getattr(mixer.Outlet.Solid_1, quantity) == pytest.approx(solid_amount, rel=RTOL)
    expected_composition = (HOT_COMPOSITION + COMPOSITION / 2) / 1.5  # [-], 2:1 liquid mix
    np.testing.assert_allclose(mixer.Liquid_1.mass_frac, expected_composition, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Liquid_1.mass_frac, expected_composition, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Solid_1.distrib, POPULATION, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.distrib * mixer.Outlet.vol,
                               POPULATION, rtol=RTOL)
    assert mixer.Outlet.Solid_1.kv == KV
    # Independent exact third-moment trapezoids for the four-bin fixture.
    expected_volume = KV * 2e-4  # [m**3] or [m**3/s], see POPULATION
    assert mixer.Outlet.Solid_1.vol == pytest.approx(expected_volume, rel=RTOL)
    outlet_energy = sum(getattr(phase, quantity) * phase.getEnthalpy(basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J/s] or [J]
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL)
    assert mixer.Outlet.temp == pytest.approx(mixer.Outlet.Liquid_1.temp, rel=RTOL)


@pytest.mark.parametrize('added_pore_fill', [0.1, 2.0])
@pytest.mark.parametrize('solids_first', [False, True])
def test_public_partial_cake_mix_uses_attached_inventory(
        path, added_pore_fill, solids_first):
    """Conserve cake inventories and energy independently of inlet order.

    Parameters
    ----------
    path : str
        Shipped five-species thermodynamic database path.
    added_pore_fill : float
        Added liquid volume divided by the cake pore volume [-]; the
        smaller case retains Cake and the larger case produces Slurry.
    solids_first : bool
        Place the cake before the pure-liquid inlet.
    """
    cake = Cake(saturation=np.full(3, 0.2), z_external=np.linspace(0, 1, 3))
    # Saturation metadata is 0.2; attached liquid fills 0.6 of pore volume.
    solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, temp=COLD,
                        x_distrib=GRID, distrib=POPULATION, kv=KV)
    cake.Phases = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION,
                               temp=COLD), solid]
    pore_volume = cake.cake_vol * cake.porosity  # [m**3]
    cake.Liquid_1.updatePhase(vol=0.6 * pore_volume)  # [m**3], charged inventory
    liquid = LiquidPhase(path, mass_frac=HOT_COMPOSITION, temp=HOT,
                         vol=added_pore_fill * pore_volume)  # [m**3]
    liquid_mass = liquid.mass + cake.Liquid_1.mass  # [kg]
    solid_mass = solid.mass  # [kg]
    expected_composition = (liquid.mass * HOT_COMPOSITION
                            + cake.Liquid_1.mass * COMPOSITION) / liquid_mass  # [-]
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in [liquid, *cake.Phases])  # [J]
    mixer = Mixer()
    mixer.Inlets = [cake, liquid] if solids_first else [liquid, cake]
    mixer.solve_unit()
    assert isinstance(mixer.Outlet, Cake if added_pore_fill < 1 else Slurry)
    assert mixer.Outlet.Liquid_1.mass == pytest.approx(liquid_mass, rel=RTOL)
    assert mixer.Outlet.Solid_1.mass == pytest.approx(solid_mass, rel=RTOL)
    assert mixer.Outlet.Solid_1.kv == KV
    np.testing.assert_allclose(mixer.Outlet.Liquid_1.mass_frac,
                               expected_composition, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Solid_1.distrib, POPULATION, rtol=RTOL)
    assert np.isfinite(mixer.Outlet.Solid_1.moments).all()
    assert COLD < mixer.Outlet.Liquid_1.temp < HOT
    outlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J]
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL)
    assert mixer.Outlet.temp == pytest.approx(mixer.Outlet.Liquid_1.temp, rel=RTOL)
    if isinstance(mixer.Outlet, Cake):
        np.testing.assert_allclose(mixer.Outlet.saturation, 0.6 + added_pore_fill,
                                   rtol=RTOL)


@pytest.mark.parametrize('cake_first', [False, True])
def test_cake_stream_mix_rejects_incompatible_bases(path, cake_first):
    cake = Cake()
    stream = LiquidStream(path, mass_frac=COMPOSITION, mass_flow=FLOW)
    mixer = Mixer()
    inlets = [cake, stream] if cake_first else [stream, cake]
    with pytest.raises(ValueError, match=r'inventories.*kg.*flow rates.*kg/s.*duration'):
        mixer.Inlets = inlets
    assert mixer.Inlets == []


def _multi_solid_mixer(path, difference=None):
    """Build two slurries with an optional incompatible solid property.

    Parameters
    ----------
    path : str
        Five-species property database path.
    difference : str, optional
        Property changed on inlet index 2; None keeps both solids compatible.

    Returns
    -------
    Mixer
        Batch mixer with liquid first, then hot and cold slurries. Liquid
        amounts are MASS [kg]; solid populations are POPULATION and twice
        POPULATION [#/um].
    """
    mixer = Mixer()
    inlets = [LiquidPhase(path, mass=MASS, mass_frac=HOT_COMPOSITION, temp=HOT)]
    for index, temperature in enumerate((HOT, COLD)):
        shape_factor = 0.2 if index and difference == 'kv' else KV  # [-], distinct crystal shape
        composition = ([0, 1, 0, 0, 0] if index and difference == 'mass_frac'
                       else SOLID_COMPOSITION)  # [-], pure B or pure A
        grid = GRID.copy()  # [um]
        population = (index + 1) * POPULATION  # [#/um], unequal inlet counts
        if index and difference == 'grid_length':
            grid = grid[:-1]  # [um], omit the last size node
            population = population[:-1]  # [#/um]
        elif index and difference == 'grid_values':
            grid = grid * 2  # [um], distinct sizes on the same number of nodes
        slurry = Slurry()
        slurry.Phases = [
            LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=temperature),
            SolidPhase(path, mass_frac=composition, temp=temperature,
                       kv=shape_factor, x_distrib=grid, distrib=population),
        ]
        inlets.append(slurry)
    mixer.Inlets = inlets
    return mixer


@pytest.mark.parametrize('difference, quantity', [
    ('kv', 'kv'), ('mass_frac', 'mass_frac'),
    ('grid_length', 'x_distrib'), ('grid_values', 'x_distrib'),
])
def test_multi_solid_mixer_rejects_incompatible_inlet(path, difference, quantity):
    mixer = _multi_solid_mixer(path, difference)
    with pytest.raises(ValueError, match=rf'inlet 2.*{quantity}.*inlet 1'):
        mixer.solve_unit()


def test_multi_solid_mixer_closes_phase_mass_population_and_energy(path):
    mixer = _multi_solid_mixer(path)
    phases = [mixer.Inlets[0], *mixer.Inlets[1].Phases, *mixer.Inlets[2].Phases]
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in phases)  # [J], independent phase sum
    solid_mass = sum(inlet.Solid_1.mass for inlet in mixer.Inlets[1:])  # [kg]
    mixer.solve_unit()
    assert mixer.Outlet.Liquid_1.mass == pytest.approx(3 * MASS, rel=RTOL)
    assert mixer.Outlet.Solid_1.mass == pytest.approx(solid_mass, rel=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Solid_1.distrib, 3 * POPULATION, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.Liquid_1.mass_frac,
                               (HOT_COMPOSITION + 2 * COMPOSITION) / 3, rtol=RTOL)
    outlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J]
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL)


@pytest.mark.parametrize('pore_fill, grid_dtype', [(0.75, int), (0.75, float), (1.1, int)])
def test_consistent_cake_uses_moment_geometry_and_inlet_grid(path, pore_fill, grid_dtype):
    """Use particle moments for geometry while retaining attached mass.

    Parameters
    ----------
    path : str
        Property database path.
    pore_fill : float
        Liquid volume divided by moment-derived pore volume [-]. Values
        below and above one test both sides of the Cake/Slurry decision.
    grid_dtype : type
        Integer or float source coordinates [mm], converted to float metres
        before attachment; both cases check saturation and grid independence.
    """
    grid = np.array([100., 200., 300.])  # [um], uniform positive-size grid
    population = np.array([1e6, 2e6, 1e6])  # [#/um], symmetric raw number density
    solid = SolidPhase(path, mass=0, mass_frac=SOLID_COMPOSITION, temp=COLD,
                        kv=KV, x_distrib=grid, distrib=population)
    # Independent trapezoids: 100 * (100**3*1e6/2 + 200**3*2e6
    # + 300**3*1e6/2) * 1e-18 = 3e-3 m**3, before applying kv.
    particle_volume = KV * 3e-3  # [m**3]
    solid_mass = particle_volume * solid.getDensity()  # [kg]
    solid.updatePhase(mass=solid_mass)
    grid_mm = np.arange(7, dtype=grid_dtype)  # [mm], seven nodes at 1 mm spacing
    z_external = grid_mm * 1e-3  # [m], exact millimetre-to-metre conversion
    expected_grid = z_external.copy()  # [m], independent grid snapshot
    cake = Cake(z_external=z_external)
    cake.Phases = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION,
                               temp=COLD), solid]
    pore_volume = particle_volume * cake.porosity / (1 - cake.porosity)  # [m**3]
    cake.Liquid_1.updatePhase(vol=pore_volume / 4)  # [m**3], initial quarter filling
    liquid = LiquidPhase(path, mass_frac=COMPOSITION, temp=HOT,
                         vol=(pore_fill - 0.25) * pore_volume)  # [m**3]
    liquid_mass = liquid.mass + cake.Liquid_1.mass  # [kg]
    mixer = Mixer()
    mixer.Inlets = [liquid, cake]
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in [liquid, *cake.Phases])  # [J]
    mixer.solve_unit()
    assert isinstance(mixer.Outlet, Cake if pore_fill < 1 else Slurry)
    assert mixer.Outlet.Liquid_1.mass == pytest.approx(liquid_mass, rel=RTOL)
    assert mixer.Outlet.Solid_1.mass == pytest.approx(solid_mass, rel=RTOL)
    outlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J]
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL)
    if pore_fill < 1:
        np.testing.assert_allclose(mixer.Outlet.saturation, pore_fill, rtol=RTOL)
        np.testing.assert_array_equal(mixer.Outlet.z_external, expected_grid)
        cake.z_external[0] = -1  # [m], mutate the producer's grid after mixing
        np.testing.assert_array_equal(mixer.Outlet.z_external, expected_grid)
        assert mixer.Outlet.saturation.dtype.kind == 'f'
        assert mixer.Outlet.cake_vol == pytest.approx(
            particle_volume / (1 - cake.porosity), rel=RTOL)


@pytest.mark.parametrize('continuous', [False, True])
def test_mixer_rejects_bin_weight_mass_moment_mismatch(path, continuous):
    """Reject the inconsistent inventory before the nonconservative balance.

    Parameters
    ----------
    path : str
        Property database path.
    continuous : bool
        Test batch Cake inventories [kg] or SlurryStream mass flows [kg/s].
    """
    liquid_class = LiquidStream if continuous else LiquidPhase
    solid_class = SolidStream if continuous else SolidPhase
    amount_name = 'mass_flow' if continuous else 'mass'
    amount = FLOW if continuous else MASS  # [kg/s] or [kg]
    grid = np.array([100., 200., 300.])  # [um], uniform positive-size grid
    weights = np.array([0.25, 0.5, 0.25])  # [-], nonzero endpoint bin weights
    solid = solid_class(path, mass_frac=SOLID_COMPOSITION, temp=COLD,
                        kv=KV, x_distrib=grid, distrib=weights,
                        **{amount_name: amount})
    # Constructor bin weights sum to one, but trapezoidal integration halves
    # endpoint weights, so the moment-derived mass is only 0.75 of the input.
    moment_amount = 0.75 * amount  # [kg/s] or [kg], independent endpoint weights
    particle_volume = moment_amount / solid.getDensity()  # [m**3/s] or [m**3]
    porosity = solid.getPorosity()  # [-], same packing model as Cake
    pore_volume = particle_volume * porosity / (1 - porosity)  # [m**3/s] or [m**3]
    volume_name = 'vol_flow' if continuous else 'vol'
    inlet = SlurryStream() if continuous else Cake()
    inlet.Phases = [
        liquid_class(path, mass_frac=COMPOSITION, temp=COLD,
                     **{volume_name: pore_volume / 4}), solid,
    ]
    # Total liquid fills 1.1 times the moment-derived pores, reproducing the
    # prior 22.87% batch energy imbalance when the outlet becomes Slurry.
    liquid = liquid_class(path, mass_frac=COMPOSITION, temp=HOT,
                          **{volume_name: (1.1 - 0.25) * pore_volume})
    mixer = Mixer()
    mixer.Inlets = [liquid, inlet]
    with pytest.raises(ValueError, match=r'Mixer inlet 1.*attached solid.*moment-derived') as error:
        mixer.solve_unit()
    message = str(error.value)
    units = 'kg/s' if continuous else 'kg'
    assert f'attached solid {amount_name}={amount:g} {units}' in message
    assert f'moment-derived {amount_name}={moment_amount:g} {units}' in message
    assert 'attached masses authoritative' in message
    assert 'slurry enthalpy weights phases by moment-derived fractions' in message
    assert 'cannot be mixed conservatively' in message


def test_mixer_checks_later_solid_inventory(path):
    mixer = _multi_solid_mixer(path)
    solid = mixer.Inlets[2].Solid_1
    solid.updatePhase(mass=2 * solid.mass)  # [kg], double inventory without changing counts
    with pytest.raises(ValueError, match=r'Mixer inlet 2.*attached solid.*moment-derived'):
        mixer.solve_unit()



def test_mixer_rejects_stale_moment_crystallizer_distribution(data_path, path):
    unit, initial = inventory_unit(data_path, BatchCryst)
    seed_population = unit.Solid_1.distrib.copy()  # [#/um]
    seed_mass = unit.Solid_1.mass  # [kg], initial distribution-consistent inventory
    states = np.tile(initial, (2, 1))  # [um**n], [kg/m**3], [m**3]
    states[-1, :4] *= 2  # [um**n], double all moments without changing population shape
    unit.retrieve_results(np.array([0., 1.]), states)  # [s], synthetic retrieval interval
    np.testing.assert_array_equal(unit.Outlet.Solid_1.distrib, seed_population)
    assert unit.Outlet.Solid_1.mass == pytest.approx(2 * seed_mass, rel=RTOL)
    assert unit.Outlet.Solid_1.mass == pytest.approx(
        unit.Outlet.Solid_1.getDensity() * unit.Outlet.Solid_1.kv
        * unit.Outlet.Solid_1.moments[3], rel=RTOL)
    mixer = Mixer()
    mixer.Inlets = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=HOT),
                    unit.Outlet]
    with pytest.raises(ValueError, match=r'inlet 1.*moment-mode crystallizer outlets do not refresh their distribution'):
        mixer.solve_unit()


def test_mixer_rejects_moment_only_msmpr_stream(data_path, path):
    unit, initial = inventory_unit(data_path, MSMPR, gridless=True)
    unit.retrieve_results(np.array([0., 1.]), np.tile(initial, (2, 1)))  # [s], static profile
    assert getattr(unit.Outlet.Solid_1, 'distrib', None) is None
    mixer = Mixer()
    mixer.Inlets = [LiquidStream(path, mass_flow=FLOW, mass_frac=COMPOSITION), unit.Outlet]
    with pytest.raises(ValueError, match=r'inlet 1.*size-distributed solid on a size grid.*moment-only or gridless populations are not supported'):
        mixer.solve_unit()


@pytest.mark.parametrize('producer', ['fvm', 'filter', 'zero_population'])
def test_mixer_accepts_distributed_producer_and_closes_energy(data_path, path, producer):
    """Check real retrieval producers and a crystal-free seed slurry.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    path : str
        Property database path.
    producer : str
        FVM BatchCryst retrieval, Filter retrieval after full solids recovery,
        or a constructed slurry with zero crystal population.
    """
    if producer == 'fvm':
        unit, initial = make_fvm_unit(data_path, BatchCryst, adiabatic=True)
        states = np.tile(initial, (2, 1))  # [#/um], [kg/m**3], [m**3], [K]
        states[-1, :len(unit.Solid_1.distrib)] *= 2  # [#/um], changed final population
        unit.retrieve_results(np.array([0., 1.]), states)  # [s]
        inlet = unit.Outlet
    else:
        solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, temp=COLD, kv=KV,
                            x_distrib=GRID, distrib=(np.zeros_like(POPULATION)
                                                    if producer == 'zero_population'
                                                    else POPULATION))
        inlet = Slurry()
        inlet.Phases = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=COLD), solid]
        if producer == 'filter':
            unit = Filter(station_diam=1.0)  # [m], existing packing-model diameter
            unit.Phases = inlet
            filtrate_mass = inlet.Liquid_1.mass / 2  # [kg], half the liquid filtered
            liquid_density = inlet.Liquid_1.getDensity()  # [kg/m**3]
            unit.mass_crit = filtrate_mass  # [kg], full solid recovery at this filtrate mass
            states = np.array([[0., MASS], [filtrate_mass, MASS - filtrate_mass]])  # [kg]
            unit.retrieve_results(np.array([0., 1.]), states, liquid_density,
                                  solid.getDensity(), solid.getPorosity(), mass_solids=solid.mass)
            inlet = unit.Outlet
    liquid = LiquidPhase(path, mass=MASS, mass_frac=HOT_COMPOSITION, temp=HOT)
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in [liquid, *inlet.Phases])  # [J]
    liquid_mass = liquid.mass + inlet.Liquid_1.mass  # [kg]
    solid_mass = inlet.Solid_1.mass  # [kg]
    mixer = Mixer()
    mixer.Inlets = [liquid, inlet]
    with np.errstate(invalid='raise'):
        mixer.solve_unit()
    assert mixer.Outlet.Liquid_1.mass == pytest.approx(liquid_mass, rel=RTOL)
    assert mixer.Outlet.Solid_1.mass == pytest.approx(solid_mass, rel=RTOL)
    outlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                        for phase in mixer.Outlet.Phases)  # [J]
    assert outlet_energy == pytest.approx(inlet_energy, rel=RTOL)



def test_mixer_rejects_gridless_distribution(path):
    # Independent trapezoids of GRID and POPULATION give these total SI moments.
    moments = np.array([1.125e8, 1.25e4, 1.5, 2e-4])  # [m**n], n=0..3
    solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, temp=COLD, kv=KV,
                        moments=moments, distrib=POPULATION)
    assert solid.distrib is not None
    assert solid.x_distrib is None
    slurry = Slurry()
    slurry.Phases = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=COLD), solid]
    mixer = Mixer()
    mixer.Inlets = [LiquidPhase(path, mass=MASS, mass_frac=COMPOSITION, temp=COLD), slurry]
    with pytest.raises(ValueError, match=(
            r'inlet 1.*requires a size-distributed solid on a size grid.*'
            r'moment-only or gridless populations are not supported')):
        mixer.solve_unit()
