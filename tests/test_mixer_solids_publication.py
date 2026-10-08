"""Result and output publication of instantaneous solids ``Mixer`` solves.

Issue #289: a successful solids mix must publish a one-sample ``result`` and
named ``outputs`` that ``SimulationExec`` and ``Connection`` consume. The
fixture follows the issue reproduction with the shipped five-species
database: a 0.001 m**3 slurry at 350 K on the [0, 100, 200, 300] um grid
with 1e9 and 1.25e8 #/m**3/um in the inner bins, mixed with 1 kg of liquid at
300 K. The feed composition differs from the slurry liquid so that a
species-order error changes the mixed composition. Further cases cover a
two-cake batch mix that remains a Cake and a continuous SlurryStream mix.

Downstream handoffs use real collaborators without a solve, so no ODE
backend is needed: a Filter receives the batch slurry and an MSMPR reads the
continuous stream through ``get_inputs``. Connected inputs to a solids Mixer,
including solids Mixer to solids Mixer chains, are covered in
tests/test_mixer_solids_connected_inputs.py.
"""

import json

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.Crystallizers import MSMPR
from PharmaPy.MixedPhases import Cake, Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = pytest.mark.unit

SLURRY_LIQUID_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #289
FEED_COMPOSITION = np.array([0.4, 0.05, 0.15, 0.1, 0.3])  # [-], asymmetric feed
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
GRID = np.array([0., 100., 200., 300.])  # [um], issue #289 size grid
SLURRY_DISTRIB = np.array([0., 1e9, 1.25e8, 0.])  # [#/m**3/um], issue #289
SLURRY_VOL = 1e-3  # [m**3], issue #289 slurry volume
KV = 1.0  # [-], issue #289 volumetric shape factor
COLD = 300.0  # [K], liquid feed temperature
HOT = 350.0  # [K], slurry temperature
FEED_MASS = 1.0  # [kg], issue #289 liquid feed
# Two cakes on GRID with different total populations, so their sum is not
# a rescaled copy of either inlet.
CAKE_POPULATIONS = (np.array([0., 1e6, 1.25e5, 0.]),
                    np.array([0., 2e5, 4e5, 0.]))  # [#/um]
# Attached liquid volume per cake pore volume [-]. Both cakes are partly
# saturated, so the combined liquid stays below the combined pore volume and
# the outlet remains a Cake (asserted as a precondition).
CAKE_PORE_FILLS = (0.3, 0.2)  # [-]
CAKE_COORDINATES = np.linspace(0, 1e-2, 5)  # [m], five nodes over 1 cm
CAKE_LIQUID_SEED = 1e-3  # [kg], replaced by the pore-fill volume update
SLURRY_LIQUID_FLOW = 0.5  # [kg/s], liquid in the continuous slurry
FEED_FLOW = 1.0  # [kg/s], continuous liquid feed
SOLID_NUMBER_RATE = np.array([0., 1e6, 1.25e5, 0.])  # [#/um/s], crystal feed
# Issue #289 Filter parameters. The Filter is not solved, so they only need
# to be valid inputs: station diameter [m], specific cake resistance [m/kg],
# and filter medium resistance [1/m].
FILTER_DIAM = 0.1  # [m]
CAKE_RESISTANCE = 1e11  # [m/kg]
MEDIUM_RESISTANCE = 1e10  # [1/m]
MSMPR_VOL = 1e-3  # [m**3], tank volume; only the inlet handoff is evaluated
QUERY_TIMES = np.array([0., 5.])  # [s], two reads of the static feed
# Relative tolerance for independently recomputed sums, mixture properties,
# and moment quotients; float64 roundoff only, no physical uncertainty.
RTOL = 1e-10  # [-]
PUBLISHED_NAMES = ['mass_liq', 'mass_solid', 'mass_frac', 'temp']


@pytest.fixture
def path(data_path):
    """Return the shipped five-species property database.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    str
        Database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


@pytest.fixture
def species(path):
    """Return the database species order read directly from the JSON file.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    list of str
        Species names in database key order, independent of PharmaPy.
    """
    with open(path) as database:
        return list(json.load(database))


def _batch_slurry(path, temp=HOT):
    """Build the issue #289 batch slurry.

    Parameters
    ----------
    path : str
        Database path.
    temp : float, optional
        Slurry temperature [K].

    Returns
    -------
    Slurry
        Slurry of volume SLURRY_VOL [m**3] with total population
        SLURRY_DISTRIB * SLURRY_VOL [#/um].
    """
    liquid = LiquidPhase(path, mass_frac=SLURRY_LIQUID_COMPOSITION, temp=temp)
    solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, kv=KV, temp=temp)
    slurry = Slurry(vol=SLURRY_VOL, x_distrib=GRID.copy(),
                    distrib=SLURRY_DISTRIB.copy())
    slurry.Phases = (liquid, solid)
    return slurry


def _slurry_mixer(path, solids_first=False):
    """Build the issue #289 batch mixer and its inlets.

    Parameters
    ----------
    path : str
        Database path.
    solids_first : bool, optional
        Place the slurry before the liquid feed.

    Returns
    -------
    tuple
        Unsolved Mixer, liquid feed of FEED_MASS [kg] at COLD [K], and the
        slurry at HOT [K].
    """
    feed = LiquidPhase(path, mass_frac=FEED_COMPOSITION, mass=FEED_MASS,
                       temp=COLD)
    slurry = _batch_slurry(path)
    mixer = Mixer()
    mixer.Inlets = [slurry, feed] if solids_first else [feed, slurry]
    return mixer, feed, slurry


def _cake(path, population, pore_fill, composition, temp):
    """Build a partly saturated Cake on GRID.

    Parameters
    ----------
    path : str
        Database path.
    population : numpy.ndarray
        Total number distribution [#/um], shape (len(GRID),).
    pore_fill : float
        Attached liquid volume divided by the cake pore volume [-].
    composition : numpy.ndarray
        Liquid mass fractions [-] in database order.
    temp : float
        Temperature of both phases [K].

    Returns
    -------
    Cake
        Cake whose solid mass [kg] follows the population moments.
    """
    solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, kv=KV, temp=temp,
                       x_distrib=GRID.copy(), distrib=population.copy())
    cake = Cake(z_external=CAKE_COORDINATES.copy())
    cake.Phases = [LiquidPhase(path, mass=CAKE_LIQUID_SEED,
                               mass_frac=composition, temp=temp), solid]
    pore_volume = cake.cake_vol * cake.porosity  # [m**3]
    cake.Liquid_1.updatePhase(vol=pore_fill * pore_volume)  # [m**3]
    return cake


def _cake_mixer(path):
    """Build a two-cake batch mixer whose outlet remains a Cake.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    tuple
        Unsolved Mixer and its two Cake inlets.
    """
    cakes = (_cake(path, CAKE_POPULATIONS[0], CAKE_PORE_FILLS[0],
                   SLURRY_LIQUID_COMPOSITION, COLD),
             _cake(path, CAKE_POPULATIONS[1], CAKE_PORE_FILLS[1],
                   FEED_COMPOSITION, HOT))
    mixer = Mixer()
    mixer.Inlets = list(cakes)
    return mixer, cakes


def _continuous_mixer(path):
    """Build a continuous mixer of a liquid feed and a SlurryStream.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    tuple
        Unsolved Mixer, liquid feed of FEED_FLOW [kg/s] at COLD [K], and a
        SlurryStream at HOT [K] with liquid flow SLURRY_LIQUID_FLOW [kg/s]
        and crystal number rate SOLID_NUMBER_RATE [#/um/s].
    """
    slurry = SlurryStream()
    slurry.Phases = [
        LiquidStream(path, mass_frac=SLURRY_LIQUID_COMPOSITION, temp=HOT,
                     mass_flow=SLURRY_LIQUID_FLOW),
        SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=HOT, kv=KV,
                    x_distrib=GRID.copy(), distrib=SOLID_NUMBER_RATE.copy())]
    feed = LiquidStream(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                        mass_flow=FEED_FLOW)
    mixer = Mixer()
    mixer.Inlets = [feed, slurry]
    return mixer, feed, slurry


def _expected_liquid(feed_amount, slurry_amount):
    """Mix the feed and slurry liquid compositions by their amounts.

    Parameters
    ----------
    feed_amount, slurry_amount : float
        Liquid amounts [kg] or flows [kg/s] on the same basis.

    Returns
    -------
    tuple
        Total liquid amount [kg] or [kg/s] and mass fractions [-].
    """
    total = feed_amount + slurry_amount  # [kg] or [kg/s]
    composition = (feed_amount * FEED_COMPOSITION
                   + slurry_amount * SLURRY_LIQUID_COMPOSITION) / total  # [-]
    return total, composition


def _assert_published_contract(mixer, species, distrib_name, amount_units,
                               distrib_units):
    """Check published names, shapes, units, and result/output identity.

    Parameters
    ----------
    mixer : Mixer
        Solved solids mixer.
    species : list of str
        Expected species order.
    distrib_name : str
        Expected distribution name.
    amount_units, distrib_units : str
        Expected amount and distribution units.
    """
    names = PUBLISHED_NAMES + [distrib_name]
    assert mixer.names_states_out == names
    assert mixer.name_states == names
    assert list(mixer.outputs) == names + ['x_cryst', 'time']
    np.testing.assert_array_equal(mixer.result.time, np.zeros(1))
    np.testing.assert_array_equal(mixer.outputs['x_cryst'], GRID)
    shapes = {name: np.shape(mixer.outputs[name]) for name in mixer.outputs}
    assert shapes == {'mass_liq': (1,), 'mass_solid': (1,),
                      'mass_frac': (1, len(species)), 'temp': (1,),
                      distrib_name: (1, len(GRID)), 'x_cryst': (len(GRID),),
                      'time': (1,)}
    units = {name: mixer.states_di[name]['units'] for name in names}
    assert units == {'mass_liq': amount_units, 'mass_solid': amount_units,
                     'mass_frac': '', 'temp': 'K',
                     distrib_name: distrib_units}
    assert mixer.states_di['mass_frac']['index'] == species
    assert mixer.states_di[distrib_name]['index'] == list(range(len(GRID)))
    assert mixer.dim_states == [1, 1, len(species), 1, len(GRID)]
    assert all(di['type'] == 'alg' for di in mixer.states_di.values())
    assert mixer.result.di_states is mixer.states_di
    for name in mixer.outputs:
        assert getattr(mixer.result, name) is mixer.outputs[name]


@pytest.mark.parametrize('solids_first', [False, True])
def test_batch_slurry_publishes_one_sample_result(path, species, solids_first):
    """Publish the balanced batch slurry outlet as a one-sample result."""
    mixer, feed, slurry = _slurry_mixer(path, solids_first)
    liquid_mass, composition = _expected_liquid(
        feed.mass, slurry.Liquid_1.mass)  # [kg], [-]
    solid_mass = slurry.Solid_1.mass  # [kg]
    solid_vol = slurry.Solid_1.vol  # [m**3], unchanged population
    total_population = SLURRY_DISTRIB * SLURRY_VOL  # [#/um]

    returned = mixer.solve_unit()

    _assert_published_contract(mixer, species, 'distrib', 'kg', '#/m**3/um')
    outputs = mixer.outputs
    assert isinstance(mixer.Outlet, Slurry)
    # The legacy return tuple is unchanged and matches the published values.
    assert len(returned) == 5
    for name, value in zip(['mass_liq', 'mass_solid', 'mass_frac', 'distrib',
                            'temp'], returned):
        np.testing.assert_array_equal(outputs[name][0], value)

    assert outputs['mass_liq'][0] == pytest.approx(liquid_mass, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(solid_mass, rel=RTOL)
    np.testing.assert_allclose(outputs['mass_frac'][0], composition, rtol=RTOL)
    temp = outputs['temp'][0]  # [K]
    assert COLD < temp < HOT
    outlet_liquid_vol = LiquidPhase(path, mass=liquid_mass,
                                    mass_frac=composition, temp=temp).vol  # [m**3]
    expected_distrib = total_population / (outlet_liquid_vol + solid_vol)  # [#/m**3/um]
    np.testing.assert_allclose(outputs['distrib'][0], expected_distrib,
                               rtol=RTOL)

    outlet = mixer.Outlet
    assert outputs['mass_liq'][0] == pytest.approx(outlet.Liquid_1.mass, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(outlet.Solid_1.mass, rel=RTOL)
    np.testing.assert_allclose(outputs['mass_frac'][0], outlet.Liquid_1.mass_frac,
                               rtol=RTOL)
    assert temp == outlet.temp == outlet.Liquid_1.temp == outlet.Solid_1.temp
    np.testing.assert_array_equal(outputs['distrib'][0], outlet.distrib)
    np.testing.assert_allclose(outlet.Solid_1.distrib, total_population, rtol=RTOL)


def test_batch_cake_publishes_total_population(path, species):
    """Publish a Cake outlet with the summed inlet total populations."""
    mixer, cakes = _cake_mixer(path)
    liquid_mass = sum(cake.Liquid_1.mass for cake in cakes)  # [kg]
    composition = sum(cake.Liquid_1.mass * np.asarray(cake.Liquid_1.mass_frac)
                      for cake in cakes) / liquid_mass  # [-]
    solid_mass = sum(cake.Solid_1.mass for cake in cakes)  # [kg]
    total_population = CAKE_POPULATIONS[0] + CAKE_POPULATIONS[1]  # [#/um]

    mixer.solve_unit()

    assert isinstance(mixer.Outlet, Cake)
    _assert_published_contract(mixer, species, 'total_distrib', 'kg', '#/um')
    outputs = mixer.outputs
    assert outputs['mass_liq'][0] == pytest.approx(liquid_mass, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(solid_mass, rel=RTOL)
    np.testing.assert_allclose(outputs['mass_frac'][0], composition, rtol=RTOL)
    np.testing.assert_allclose(outputs['total_distrib'][0], total_population,
                               rtol=RTOL)
    np.testing.assert_array_equal(outputs['total_distrib'][0],
                                  mixer.Outlet.Solid_1.distrib)
    assert COLD < outputs['temp'][0] < HOT
    assert outputs['temp'][0] == mixer.Outlet.temp == mixer.Outlet.Liquid_1.temp


def test_continuous_slurry_stream_publishes_flow_basis(path, species):
    """Publish a SlurryStream outlet on the [kg/s] and [#/m**3/um] bases."""
    mixer, feed, slurry = _continuous_mixer(path)
    liquid_flow, composition = _expected_liquid(
        feed.mass_flow, slurry.Liquid_1.mass_flow)  # [kg/s], [-]
    solid_flow = slurry.Solid_1.mass_flow  # [kg/s]
    solid_vol_flow = slurry.Solid_1.vol  # [m**3/s], unchanged number rate

    mixer.solve_unit()

    assert isinstance(mixer.Outlet, SlurryStream)
    _assert_published_contract(mixer, species, 'distrib', 'kg/s', '#/m**3/um')
    outputs = mixer.outputs
    assert outputs['mass_liq'][0] == pytest.approx(liquid_flow, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(solid_flow, rel=RTOL)
    np.testing.assert_allclose(outputs['mass_frac'][0], composition, rtol=RTOL)
    temp = outputs['temp'][0]  # [K]
    assert COLD < temp < HOT
    liquid_vol_flow = LiquidStream(path, mass_flow=liquid_flow,
                                   mass_frac=composition, temp=temp).vol_flow  # [m**3/s]
    # Number rate [#/um/s] per slurry volume flow [m**3/s] is [#/m**3/um].
    expected_distrib = SOLID_NUMBER_RATE / (liquid_vol_flow + solid_vol_flow)
    np.testing.assert_allclose(outputs['distrib'][0], expected_distrib,
                               rtol=RTOL)

    outlet = mixer.Outlet
    assert outputs['mass_liq'][0] == pytest.approx(outlet.Liquid_1.mass_flow, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(outlet.Solid_1.mass_flow, rel=RTOL)
    assert temp == outlet.temp
    np.testing.assert_array_equal(outputs['distrib'][0], outlet.distrib)
    np.testing.assert_allclose(outlet.Solid_1.distrib, SOLID_NUMBER_RATE, rtol=RTOL)


@pytest.mark.parametrize('build', [_slurry_mixer, _cake_mixer, _continuous_mixer],
                         ids=['batch-slurry', 'batch-cake', 'continuous'])
def test_single_unit_flowsheet_records_zero_processing_time(path, build):
    """Solve a one-unit flowsheet without inventing a mixing duration."""
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = build(path)[0]

    sim.SolveFlowsheet(verbose=False)

    assert sim.time_processing == {'M01': 0.0}
    np.testing.assert_array_equal(sim.M01.result.time, np.zeros(1))
    assert isinstance(sim.M01.outputs, dict)


def test_batch_slurry_flowsheet_matches_standalone_solve(path):
    """Publish the same values from a flowsheet solve as from a standalone one."""
    standalone = _slurry_mixer(path)[0]
    standalone.solve_unit()
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = _slurry_mixer(path)[0]

    sim.SolveFlowsheet(verbose=False)

    assert list(sim.M01.outputs) == list(standalone.outputs)
    for name, value in standalone.outputs.items():
        np.testing.assert_allclose(sim.M01.outputs[name], value, rtol=RTOL)


def test_connection_transfers_batch_slurry_to_filter(path):
    """Hand the published slurry to a real Filter without solving it."""
    mixer, feed, slurry = _slurry_mixer(path)
    liquid_mass, composition = _expected_liquid(
        feed.mass, slurry.Liquid_1.mass)  # [kg], [-]
    solid_mass = slurry.Solid_1.mass  # [kg]
    total_population = SLURRY_DISTRIB * SLURRY_VOL  # [#/um]
    mixer.solve_unit()
    filter_unit = Filter(FILTER_DIAM, CAKE_RESISTANCE, MEDIUM_RESISTANCE)

    Connection(mixer, filter_unit).transfer_data()

    transferred = filter_unit.SlurryPhase
    assert transferred is not mixer.Outlet
    # Batch sources hand over the final published time as a scalar [s].
    assert transferred.time_upstream == 0.0
    assert list(transferred.y_upstream) == list(mixer.outputs)
    for name, value in mixer.outputs.items():
        np.testing.assert_array_equal(transferred.y_upstream[name], value)
    np.testing.assert_allclose(filter_unit.Solid_1.distrib, total_population,
                               rtol=RTOL)
    np.testing.assert_array_equal(transferred.distrib, mixer.outputs['distrib'][0])
    assert filter_unit.Liquid_1.mass == pytest.approx(liquid_mass, rel=RTOL)
    assert filter_unit.Solid_1.mass == pytest.approx(solid_mass, rel=RTOL)
    np.testing.assert_allclose(filter_unit.Liquid_1.mass_frac, composition,
                               rtol=RTOL)
    assert transferred.temp == mixer.outputs['temp'][0]
    assert filter_unit.Liquid_1.temp == mixer.outputs['temp'][0]


def test_connection_feeds_continuous_slurry_to_msmpr(path):
    """Read the published SlurryStream through a real MSMPR inlet."""
    mixer, _, _ = _continuous_mixer(path)
    mixer.solve_unit()
    crystallizer = MSMPR('A', method='1D-FVM', vol_tank=MSMPR_VOL,
                         adiabatic=True)
    crystallizer.Phases = _batch_slurry(path)

    Connection(mixer, crystallizer).transfer_data()
    inputs = crystallizer.get_inputs(QUERY_TIMES)

    inlet = crystallizer.Inlet
    np.testing.assert_array_equal(inlet.time_upstream, np.zeros(1))
    outputs = mixer.outputs
    temp = outputs['temp'][0]  # [K]
    liquid = LiquidStream(path, mass_flow=outputs['mass_liq'][0],
                          mass_frac=outputs['mass_frac'][0], temp=temp)
    expected_mass_conc = outputs['mass_frac'][0] * liquid.getDensity()  # [kg/m**3]
    num_times = len(QUERY_TIMES)
    np.testing.assert_allclose(
        inputs['Inlet']['distrib'],
        np.tile(outputs['distrib'][0], (num_times, 1)), rtol=RTOL)
    np.testing.assert_allclose(
        inputs['Liquid_1']['mass_conc'],
        np.tile(expected_mass_conc, (num_times, 1)), rtol=RTOL)
    np.testing.assert_allclose(inputs['Inlet']['temp'], np.full(num_times, temp),
                               rtol=RTOL)
    np.testing.assert_allclose(inputs['Inlet']['vol_flow'],
                               np.full(num_times, mixer.Outlet.vol_flow),
                               rtol=RTOL)
