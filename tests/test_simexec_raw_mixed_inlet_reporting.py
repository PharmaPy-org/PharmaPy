"""Raw mixed-phase inlets in raw-material, stream-table and OPEX reporting.

Issue #405: a raw continuous ``SlurryStream`` built without a slurry
distribution, so that its ``SolidStream`` carries the crystal number rate
[#/um/s], must report the solid volume flow kv * mu_3 [m**3/s] instead of
crashing. ``SolidStream`` now keeps a ``vol_flow`` alias of its per-second
``vol``. Issue #406: a raw ``Cake`` inlet of a ``Mixer`` must be classified
as raw (``Cake.y_upstream`` starts as None), while a Cake transferred from an
upstream unit stays excluded.

Fixtures use the shipped five-species database
``tests/Flowsheet/data/compound_database.json``. Its pure-component data
(densities, molecular weights) are read from the JSON file, so expected
values do not reuse PharmaPy property code. Expected solid volumes are
kv times the trapezoidal third moment of the size distribution; liquid
densities follow the ideal mass-basis mixing rule
1 / rho = sum_i w_i / rho_i.

Unit tests solve instantaneous ``Mixer`` flowsheets and need no ODE
backend; a Mixer has zero processing time, so its continuous raw totals
are zero. Tests marked ``assimulo`` integrate the issue #405 ``MSMPR``
flowsheet for 10 s with CVode to check nonzero totals and ``GetOPEX``.
``GetOPEX`` of a duty-free (Mixer-only) flowsheet fails in ``GetDuties``
(issue #261), so the Cake OPEX check adds a disconnected MSMPR that
publishes a heat duty.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Cake, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.Utilities import CoolingWater

UM3_TO_M3 = 1e-18  # [m**3/um**3], exact: (1e-6 m/um)**3
KG_TO_G = 1000.0  # [g/kg], exact SI conversion
PRESSURE = 101325.0  # [Pa], PharmaPy default phase pressure
SOLID_COMPOSITION = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # [-], pure A crystals
# Liquid compositions [-] in database order; unequal so that a row swap or a
# species-order error changes the reported values.
SLURRY_LIQUID_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
FEED_COMPOSITION = np.array([0.4, 0.05, 0.15, 0.1, 0.3])  # [-]
THIRD_COMPOSITION = np.array([0.2, 0.3, 0.1, 0.1, 0.3])  # [-]
COLD = 300.0  # [K], liquid feed and first cake
HOT = 350.0  # [K], slurry stream and second cake
WARM = 320.0  # [K], third cake

# ---------- Issue #405 continuous Mixer fixture (no ODE backend)
GRID = np.array([0., 100., 200., 300.])  # [um], PR #403 Mixer size grid
# Non-unit shape factor, so a missing kv in a volume changes the result.
SLURRY_KV = 0.5  # [-]
# Slurry-volume number density of the PR #403 Mixer fixture: with
# SLURRY_KV it gives a solid volume fraction of 0.1 [-].
SLURRY_DISTRIB = np.array([0., 1e9, 1.25e8, 0.])  # [#/m**3/um]
SLURRY_VOL_FLOW = 1e-4  # [m**3/s], total slurry volume flow
FEED_FLOW = 0.05  # [kg/s], raw liquid feed of the Mixer

# ---------- Issue #406 Cake fixture (issue reproduction populations)
CAKE_KV = 1.0  # [-], issue #406 shape factor
CAKE_POPULATIONS = (np.array([0., 1e6, 1.25e5, 0.]),
                    np.array([0., 2e5, 4e5, 0.]),
                    np.array([0., 5e5, 1e5, 0.]))  # [#/um], total populations
# Attached liquid volumes [m**3], each below its own cake's pore volume
# (asserted), so the inlets are partly saturated cakes.
CAKE_LIQUID_VOLS = (3e-5, 4e-5, 2e-5)  # [m**3]
CAKE_TEMPS = (COLD, HOT, WARM)  # [K]
CAKE_COMPOSITIONS = (SLURRY_LIQUID_COMPOSITION, FEED_COMPOSITION,
                     THIRD_COMPOSITION)  # [-]
CAKE_COORDINATES = np.linspace(0, 1e-2, 5)  # [m], five nodes over 1 cm

# ---------- Issue #405 MSMPR fixture (Assimulo lane)
MSMPR_GRID = np.array([10., 20., 40.])  # [um], issue #405 grid
MSMPR_KV = 0.5  # [-], issue #405 shape factor
MSMPR_TEMP = 310.0  # [K], issue #405 feed and holdup temperature
MSMPR_TANK_VOL = 1e-3  # [m**3], issue #405 tank
MSMPR_HOLDUP_VOL = 9e-4  # [m**3], issue #405 liquid holdup
MSMPR_SEED = np.array([1e7, 2e7, 1e7])  # [#/um], issue #405 holdup population
# Feed number density near the issue's holdup slurry basis (about
# 1.1e10 #/m**3/um), giving a solid volume fraction of 4.4e-3 [-].
MSMPR_FEED_DISTRIB = np.array([1e10, 2e10, 1e10])  # [#/m**3/um]
MSMPR_FEED_VOL_FLOW = 1e-5  # [m**3/s], issue #405 slurry feed
# Solubility far above the liquid's A concentration (about 95 kg/m**3)
# keeps the holdup undersaturated, so no crystal growth or nucleation.
MSMPR_SOLUBILITY = [2000.]  # [kg/m**3], issue #405
COOLING_FLOW = 1e-5  # [m**3/s], issue #405 cooling water
COOLING_TEMP = 300.0  # [K], issue #405 cooling water inlet
RUNTIME = 10.0  # [s], issue #405 integration horizon
PRICE = 3.0  # [USD/kg], arbitrary raw-material price, not unity

# Relative tolerance [-] for independently recomputed float64 sums,
# products and trapezoidal moments; no physical uncertainty is involved.
RTOL = 1e-10
# Absolute tolerance for expected values that are exactly zero, applied
# only to them (see _assert_rows); nonzero values use RTOL alone. The only
# such values are the raw totals of an instantaneous Mixer: amounts [kg] or
# [mol] and volumes [m**3], each a flow times the 0 s fed duration, hence
# 0.0 in float64. The smallest nonzero quantities they could become are the
# solid flows of 1e-5 m**3/s, 0.0123 kg/s and 0.123 mol/s, so any fed
# duration above 1e-10 s gives totals above 1e-15 in each unit.
ZERO_ATOL = 1e-15


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
def database(path):
    """Read pure-component data directly from the JSON database.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    dict
        ``species`` (names in database order), ``mw`` [g/mol],
        ``rho_liq`` [kg/m**3] and ``rho_solid`` [kg/m**3] arrays of shape
        (num_species,).
    """
    with open(path) as handle:
        data = json.load(handle)
    species = list(data)
    return {'species': species,
            'mw': np.array([data[name]['mw'] for name in species]),  # [g/mol]
            'rho_liq': np.array([data[name]['rho_liq']
                                 for name in species]),  # [kg/m**3]
            'rho_solid': np.array([data[name]['rho_solid']
                                   for name in species])}  # [kg/m**3]


def _third_moment(grid, distrib):
    """Return the trapezoidal third moment converted to cubic meters.

    Parameters
    ----------
    grid : numpy.ndarray
        Crystal sizes [um], shape (num_sizes,).
    distrib : numpy.ndarray
        Number distribution [#/um], [#/um/s] or [#/m**3/um], shape
        (num_sizes,).

    Returns
    -------
    float
        Third moment [m**3], [m**3/s] or [m**3/m**3], on the basis of
        ``distrib``.
    """
    integrand = distrib * grid**3  # [#*um**2] on the distrib basis
    widths = np.diff(grid)  # [um]
    return float(np.sum(widths * (integrand[1:] + integrand[:-1]) / 2)
                 * UM3_TO_M3)


def _liquid_density(database, composition):
    """Return the ideal mass-basis mixture density of a liquid.

    Parameters
    ----------
    database : dict
        Pure-component data from the ``database`` fixture.
    composition : numpy.ndarray
        Mass fractions [-], shape (num_species,).

    Returns
    -------
    float
        Mixture density [kg/m**3].
    """
    return 1 / np.dot(composition, 1 / database['rho_liq'])


def _moles_per_kg(database, composition):
    """Return moles per kilogram of a mixture.

    Parameters
    ----------
    database : dict
        Pure-component data from the ``database`` fixture.
    composition : numpy.ndarray
        Mass fractions [-], shape (num_species,).

    Returns
    -------
    float
        Amount per mass [mol/kg].
    """
    return KG_TO_G * np.dot(composition, 1 / database['mw'])


def _inlet_rows(table, unit, inlet):
    """Return one inlet's rows and assert their phase classes and order.

    Parameters
    ----------
    table : pandas.DataFrame
        Stream table or raw-material table with a (unit, source, phase)
        index.
    unit, inlet : str
        Unit-operation name and inlet source label.

    Returns
    -------
    pandas.DataFrame
        The inlet's phase rows, liquid first and solid second.
    """
    # A boolean mask keeps the reported row order; the index is unsorted.
    mask = ((table.index.get_level_values(0) == unit)
            & (table.index.get_level_values(1) == inlet))
    rows = table[mask].droplevel([0, 1])
    classes = [name.split('.')[0] for name in rows.index]
    assert len(classes) == 2
    assert classes[0] in ('LiquidStream', 'LiquidPhase')
    assert classes[1] in ('SolidStream', 'SolidPhase')
    return rows


# ---------- Issue #405: SolidStream volume-flow alias and raw SlurryStream


def _slurry_stream(path, construction, distrib, vol_flow, grid, kv, temp,
                   liquid_composition):
    """Build one raw slurry feed by either documented SlurryStream route.

    Parameters
    ----------
    path : str
        Database path.
    construction : {'solid-number-rate', 'slurry-distribution'}
        ``'solid-number-rate'`` builds ``SlurryStream()`` with no slurry
        distribution: the SolidStream carries the number rate
        ``distrib * vol_flow`` [#/um/s] and the liquid the remaining volume
        flow. ``'slurry-distribution'`` passes ``vol_flow`` and ``distrib``
        to the SlurryStream. Both describe the same feed.
    distrib : numpy.ndarray
        Slurry-volume number density [#/m**3/um], shape (num_sizes,).
    vol_flow : float
        Total slurry volume flow [m**3/s].
    grid : numpy.ndarray
        Crystal sizes [um], shape (num_sizes,).
    kv : float
        Volumetric shape factor [-].
    temp : float
        Temperature of both phases [K].
    liquid_composition : numpy.ndarray
        Liquid mass fractions [-], shape (num_species,).

    Returns
    -------
    SlurryStream
        Raw slurry stream with LiquidStream and SolidStream phases.
    """
    if construction == 'solid-number-rate':
        solid_vol_flow = kv * _third_moment(grid, distrib) * vol_flow  # [m**3/s]
        liquid = LiquidStream(path, mass_frac=liquid_composition, temp=temp,
                              vol_flow=vol_flow - solid_vol_flow)  # [m**3/s]
        solid = SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=temp,
                            kv=kv, x_distrib=grid.copy(),
                            distrib=distrib * vol_flow)  # [#/um/s]
        slurry = SlurryStream()
    else:
        # The slurry sets the liquid flow, so the liquid starts at zero.
        with pytest.warns(RuntimeWarning, match='all set to zero'):
            liquid = LiquidStream(path, mass_frac=liquid_composition,
                                  temp=temp)
        solid = SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=temp,
                            kv=kv, x_distrib=grid.copy(), distrib=distrib)
        slurry = SlurryStream(vol_flow=vol_flow, x_distrib=grid.copy(),
                              distrib=distrib.copy())  # [m**3/s], [#/m**3/um]
    slurry.Phases = (liquid, solid)
    return slurry


def _expected_slurry_rows(database, distrib, vol_flow, grid, kv,
                          liquid_composition, duration, basis):
    """Derive the liquid and solid raw rows of a slurry feed.

    Parameters
    ----------
    database : dict
        Pure-component data from the ``database`` fixture.
    distrib : numpy.ndarray
        Slurry-volume number density [#/m**3/um], shape (num_sizes,).
    vol_flow : float
        Total slurry volume flow [m**3/s].
    grid : numpy.ndarray
        Crystal sizes [um], shape (num_sizes,).
    kv : float
        Volumetric shape factor [-].
    liquid_composition : numpy.ndarray
        Liquid mass fractions [-], shape (num_species,).
    duration : float
        Fed duration [s].
    basis : {'mass', 'mole'}
        Reporting basis.

    Returns
    -------
    tuple of dict
        Liquid and solid rows: total ``mass`` [kg] or ``moles`` [mol],
        ``vol`` [m**3], flow ``mass_flow`` [kg/s] or ``mole_flow``
        [mol/s], and ``vol_flow`` [m**3/s].
    """
    solid_vol_flow = kv * _third_moment(grid, distrib) * vol_flow  # [m**3/s]
    liquid_vol_flow = vol_flow - solid_vol_flow  # [m**3/s]
    liquid_mass_flow = liquid_vol_flow * _liquid_density(
        database, liquid_composition)  # [kg/s]
    rho_a = database['rho_solid'][0]  # [kg/m**3], pure A crystals
    solid_mass_flow = solid_vol_flow * rho_a  # [kg/s]
    if basis == 'mass':
        amount, flow = 'mass', 'mass_flow'
        liquid_flow = liquid_mass_flow  # [kg/s]
        solid_flow = solid_mass_flow  # [kg/s]
    else:
        amount, flow = 'moles', 'mole_flow'
        liquid_flow = liquid_mass_flow * _moles_per_kg(
            database, liquid_composition)  # [mol/s]
        solid_flow = solid_mass_flow * KG_TO_G / database['mw'][0]  # [mol/s]
    rows = []
    for phase_flow, phase_vol_flow in ((liquid_flow, liquid_vol_flow),
                                       (solid_flow, solid_vol_flow)):
        rows.append({amount: phase_flow * duration,  # [kg] or [mol]
                     'vol': phase_vol_flow * duration,  # [m**3]
                     flow: phase_flow,  # [kg/s] or [mol/s]
                     'vol_flow': phase_vol_flow})  # [m**3/s]
    return tuple(rows)


def _assert_rows(rows, expected):
    """Compare reported phase rows with expected column values.

    Parameters
    ----------
    rows : pandas.DataFrame
        Two phase rows, liquid then solid.
    expected : tuple of dict
        Expected column values per phase in the units of the table.

    Notes
    -----
    Nonzero values are compared with the relative tolerance RTOL [-];
    exactly zero expected values with the absolute tolerance ZERO_ATOL, in
    the unit of their column.
    """
    for (_, row), values in zip(rows.iterrows(), expected):
        for column, value in values.items():
            if value == 0:
                assert row[column] == pytest.approx(0.0, abs=ZERO_ATOL), column
            else:
                assert row[column] == pytest.approx(value, rel=RTOL), column


@pytest.mark.unit
def test_solid_stream_volume_alias_follows_volume(path, database):
    """Keep ``vol_flow`` equal to the solid volume flow after every update."""
    rho_a = database['rho_solid'][0]  # [kg/m**3]
    number_rate = SLURRY_DISTRIB * SLURRY_VOL_FLOW  # [#/um/s]
    expected_vol_flow = SLURRY_KV * _third_moment(GRID, number_rate)  # [m**3/s]

    stream = SolidStream(path, mass_frac=SOLID_COMPOSITION, kv=SLURRY_KV,
                         x_distrib=GRID.copy(), distrib=number_rate)

    assert stream.vol_flow == pytest.approx(expected_vol_flow, rel=RTOL)
    assert stream.mass_flow == pytest.approx(expected_vol_flow * rho_a, rel=RTOL)

    # Explicit mass flow [kg/s], unequal to the distribution-derived one.
    updated_mass_flow = 3 * stream.mass_flow  # [kg/s]
    stream.updatePhase(mass_flow=updated_mass_flow)
    assert stream.vol_flow == pytest.approx(updated_mass_flow / rho_a, rel=RTOL)

    # A new number rate with a different shape redefines the volume flow.
    reshaped_rate = number_rate[::-1].copy()  # [#/um/s]
    stream.updatePhase(distrib=reshaped_rate)
    assert stream.vol_flow == pytest.approx(
        SLURRY_KV * _third_moment(GRID, reshaped_rate), rel=RTOL)
    assert stream.vol_flow == stream.vol


@pytest.mark.unit
def test_solid_stream_without_amount_gains_alias_on_update(path, database):
    """Create ``vol_flow`` once an update first defines the solid volume."""
    rho_a = database['rho_solid'][0]  # [kg/m**3]
    stream = SolidStream(path, mass_frac=SOLID_COMPOSITION, kv=SLURRY_KV)
    # No distribution: no volume yet, and no invented volume alias.
    assert not hasattr(stream, 'vol')
    assert not hasattr(stream, 'vol_flow')

    mass_flow = FEED_FLOW  # [kg/s], any positive flow
    stream.updatePhase(mass_flow=mass_flow)

    assert stream.vol_flow == pytest.approx(mass_flow / rho_a, rel=RTOL)


def _continuous_mixer_flowsheet(path, construction):
    """Solve a continuous Mixer of a raw liquid feed and a raw slurry.

    Parameters
    ----------
    path : str
        Database path.
    construction : {'solid-number-rate', 'slurry-distribution'}
        SlurryStream route; see :func:`_slurry_stream`.

    Returns
    -------
    SimulationExec
        Solved one-unit flowsheet ``M01``.
    """
    feed = LiquidStream(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                        mass_flow=FEED_FLOW)
    slurry = _slurry_stream(path, construction, SLURRY_DISTRIB,
                            SLURRY_VOL_FLOW, GRID, SLURRY_KV, HOT,
                            SLURRY_LIQUID_COMPOSITION)
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [feed, slurry]
    sim.SolveFlowsheet(verbose=False)
    return sim


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_mixer_raw_slurry_stream_reports_solid_volume_flow(path, database,
                                                           basis):
    """Report both SlurryStream routes with the same independent rows."""
    tables = {}
    raws = {}
    for construction in ('solid-number-rate', 'slurry-distribution'):
        sim = _continuous_mixer_flowsheet(path, construction)
        assert sim.time_processing == {'M01': 0.0}
        raws[construction] = sim.GetRawMaterials(basis=basis, totals=False)
        tables[construction] = sim.result.GetStreamTable(basis=basis)
        assert [key[:2] for key in raws[construction].index] == [
            ('M01', 'Inlet_0'), ('M01', 'Inlet_1'), ('M01', 'Inlet_1')]

    expected = _expected_slurry_rows(
        database, SLURRY_DISTRIB, SLURRY_VOL_FLOW, GRID, SLURRY_KV,
        SLURRY_LIQUID_COMPOSITION, 0.0, basis)  # instantaneous: 0 s fed
    fraction = 'mass_frac' if basis == 'mass' else 'mole_frac'
    for construction, table in tables.items():
        rows = _inlet_rows(table, 'M01', 'Inlet_1')
        _assert_rows(rows, expected)
        _assert_rows(_inlet_rows(raws[construction], 'M01', 'Inlet_1'),
                     expected)
        np.testing.assert_allclose(rows['temp'], [HOT, HOT], rtol=RTOL)
        np.testing.assert_allclose(rows['pres'], [PRESSURE, PRESSURE],
                                   rtol=RTOL)
        assert rows.iloc[1][f'{fraction}_A'] == pytest.approx(1.0, rel=RTOL)
        # Mixing conserves the crystals, so the outlet SolidStream carries
        # the same solid volume flow.
        outlet = _inlet_rows(table, 'M01', 'Outlet')
        assert outlet.iloc[1]['vol_flow'] == pytest.approx(
            expected[1]['vol_flow'], rel=RTOL)

    # Same feed, so the rows agree column by column (identifiers differ).
    first, second = (_inlet_rows(table, 'M01', 'Inlet_1').reset_index(drop=True)
                     for table in tables.values())
    assert list(first.columns) == list(second.columns)
    # Zero totals and the machine-epsilon fractions that PharmaPy stores
    # for absent species are identical in both routes, so RTOL suffices.
    np.testing.assert_allclose(first.to_numpy(dtype=float),
                               second.to_numpy(dtype=float), rtol=RTOL)


# ---------- Issue #406: raw and connected Cake Mixer inlets


def _cake(path, index):
    """Build the partly saturated Cake number ``index`` of the fixture.

    Parameters
    ----------
    path : str
        Database path.
    index : int
        Position in ``CAKE_POPULATIONS`` and the matching fixture tuples.

    Returns
    -------
    Cake
        Raw Cake on GRID with total population ``CAKE_POPULATIONS[index]``
        [#/um] and attached liquid of volume ``CAKE_LIQUID_VOLS[index]``
        [m**3], both at ``CAKE_TEMPS[index]`` [K].
    """
    temp = CAKE_TEMPS[index]  # [K]
    solid = SolidPhase(path, mass_frac=SOLID_COMPOSITION, kv=CAKE_KV,
                       temp=temp, x_distrib=GRID.copy(),
                       distrib=CAKE_POPULATIONS[index].copy())
    liquid = LiquidPhase(path, vol=CAKE_LIQUID_VOLS[index],
                         mass_frac=CAKE_COMPOSITIONS[index], temp=temp)
    cake = Cake(z_external=CAKE_COORDINATES.copy())
    cake.Phases = [liquid, solid]
    # Precondition: the liquid fits in the pores, as in a real cake.
    assert CAKE_LIQUID_VOLS[index] < cake.cake_vol * cake.porosity
    return cake


def _expected_cake_rows(database, index, basis):
    """Derive the liquid and solid raw rows of fixture Cake ``index``.

    Parameters
    ----------
    database : dict
        Pure-component data from the ``database`` fixture.
    index : int
        Fixture Cake position.
    basis : {'mass', 'mole'}
        Reporting basis.

    Returns
    -------
    tuple of dict
        Liquid and solid rows: ``mass`` [kg] or ``moles`` [mol], ``vol``
        [m**3] and ``temp`` [K].
    """
    composition = CAKE_COMPOSITIONS[index]  # [-]
    liquid_vol = CAKE_LIQUID_VOLS[index]  # [m**3]
    liquid_mass = liquid_vol * _liquid_density(database, composition)  # [kg]
    solid_vol = CAKE_KV * _third_moment(GRID, CAKE_POPULATIONS[index])  # [m**3]
    solid_mass = solid_vol * database['rho_solid'][0]  # [kg], pure A
    if basis == 'mass':
        amount = 'mass'
        liquid_amount, solid_amount = liquid_mass, solid_mass  # [kg]
    else:
        amount = 'moles'
        liquid_amount = liquid_mass * _moles_per_kg(database, composition)  # [mol]
        solid_amount = solid_mass * KG_TO_G / database['mw'][0]  # [mol]
    temp = CAKE_TEMPS[index]  # [K]
    return ({amount: liquid_amount, 'vol': liquid_vol, 'temp': temp},
            {amount: solid_amount, 'vol': solid_vol, 'temp': temp})


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_raw_cake_mixer_inlets_report_phase_totals(path, database, basis):
    """Report each raw Cake's phases and close the Mixer balance."""
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [_cake(path, 0), _cake(path, 1)]

    sim.SolveFlowsheet(verbose=False)
    table = sim.result.GetStreamTable(basis=basis)
    raw = sim.GetRawMaterials(basis=basis, totals=False)

    assert isinstance(sim.M01.Outlet, Cake)
    amount = 'mass' if basis == 'mass' else 'moles'
    assert [key[:2] for key in raw.index] == [
        ('M01', 'Inlet_0'), ('M01', 'Inlet_0'),
        ('M01', 'Inlet_1'), ('M01', 'Inlet_1')]
    totals = np.zeros(2)  # [kg] or [mol], liquid and solid sums
    for index, inlet in enumerate(('Inlet_0', 'Inlet_1')):
        expected = _expected_cake_rows(database, index, basis)
        rows = _inlet_rows(table, 'M01', inlet)
        _assert_rows(rows, expected)
        _assert_rows(_inlet_rows(raw, 'M01', inlet), expected)
        totals += [values[amount] for values in expected]

    outlet = _inlet_rows(table, 'M01', 'Outlet')
    np.testing.assert_allclose(outlet[amount], totals, rtol=RTOL)


@pytest.mark.unit
def test_connected_cake_is_not_a_raw_material(path, database):
    """Count a transferred Cake once, as its upstream unit's raw cakes."""
    sim = SimulationExec(path, {'M01': ['M02'], 'M02': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [_cake(path, 0), _cake(path, 1)]
    sim.M02 = Mixer()
    sim.M02.Inlets = [_cake(path, 2)]

    sim.SolveFlowsheet(verbose=False)
    raw = sim.GetRawMaterials(totals=False)
    table = sim.result.GetStreamTable()

    connected = sim.M02.Inlets[1]
    assert isinstance(connected, Cake)
    assert connected.y_upstream is not None
    assert sim.M02.Inlets[0].y_upstream is None
    assert [key[:2] for key in raw.index] == [
        ('M01', 'Inlet_0'), ('M01', 'Inlet_0'),
        ('M01', 'Inlet_1'), ('M01', 'Inlet_1'),
        ('M02', 'Inlet_0'), ('M02', 'Inlet_0')]
    _assert_rows(_inlet_rows(raw, 'M02', 'Inlet_0'),
                 _expected_cake_rows(database, 2, 'mass'))
    # Raw materials of both units account for the final outlet exactly.
    raw_by_phase = [sum(_expected_cake_rows(database, index, 'mass')[phase]['mass']
                        for index in range(3))
                    for phase in range(2)]  # [kg], liquid and solid
    outlet = _inlet_rows(table, 'M02', 'Outlet')
    np.testing.assert_allclose(outlet['mass'], raw_by_phase, rtol=RTOL)


# ---------- Solver-backed reporting and OPEX (Assimulo lane)


def _msmpr(path, inlet):
    """Build the issue #405 MSMPR with the given raw inlet.

    Parameters
    ----------
    path : str
        Database path.
    inlet : LiquidStream or SlurryStream
        Raw continuous feed.

    Returns
    -------
    MSMPR
        Unsolved crystallizer with an undersaturated 310 K holdup.
    """
    cryst = MSMPR('A', method='1D-FVM', vol_tank=MSMPR_TANK_VOL)
    holdup_liquid = LiquidPhase(path, temp=MSMPR_TEMP, vol=MSMPR_HOLDUP_VOL,
                                mass_frac=SLURRY_LIQUID_COMPOSITION)
    holdup_solid = SolidPhase(path, temp=MSMPR_TEMP, x_distrib=MSMPR_GRID.copy(),
                              distrib=MSMPR_SEED.copy(), kv=MSMPR_KV,
                              mass_frac=SOLID_COMPOSITION)
    cryst.Phases = (holdup_liquid, holdup_solid)
    cryst.Kinetics = CrystKinetics(coeff_solub=MSMPR_SOLUBILITY)
    cryst.Utility = CoolingWater(vol_flow=COOLING_FLOW, temp_in=COOLING_TEMP)
    cryst.Inlet = inlet
    return cryst


def _msmpr_slurry_feed(path, construction):
    """Build the MSMPR raw slurry feed by one SlurryStream route.

    Parameters
    ----------
    path : str
        Database path.
    construction : {'solid-number-rate', 'slurry-distribution'}
        SlurryStream route; see :func:`_slurry_stream`.

    Returns
    -------
    SlurryStream
        Feed of MSMPR_FEED_VOL_FLOW [m**3/s] at MSMPR_TEMP [K].
    """
    return _slurry_stream(path, construction, MSMPR_FEED_DISTRIB,
                          MSMPR_FEED_VOL_FLOW, MSMPR_GRID, MSMPR_KV,
                          MSMPR_TEMP, SLURRY_LIQUID_COMPOSITION)


RUN = {'C01': {'runtime': RUNTIME, 'verbose': False}}


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_msmpr_raw_slurry_stream_rows_are_route_independent(path, database,
                                                            basis):
    """Report flow times the 10 s fed duration for both SlurryStream routes."""
    pytest.importorskip('assimulo')
    expected = _expected_slurry_rows(
        database, MSMPR_FEED_DISTRIB, MSMPR_FEED_VOL_FLOW, MSMPR_GRID,
        MSMPR_KV, SLURRY_LIQUID_COMPOSITION, RUNTIME, basis)
    amount = 'mass' if basis == 'mass' else 'moles'
    rows = {}
    for construction in ('solid-number-rate', 'slurry-distribution'):
        sim = SimulationExec(path, {'C01': []})
        sim.C01 = _msmpr(path, _msmpr_slurry_feed(path, construction))
        sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)
        time = sim.C01.result.time  # [s]
        assert time[-1] - time[0] == pytest.approx(RUNTIME, rel=RTOL)

        raw = sim.GetRawMaterials(basis=basis, totals=False)
        _assert_rows(_inlet_rows(raw, 'C01', 'Inlet_0'), expected)
        table = sim.result.GetStreamTable(basis=basis)
        rows[construction] = _inlet_rows(table, 'C01', 'Inlet_0')
        _assert_rows(rows[construction], expected)
        # Default totals table: phase totals, flow times the 10 s fed
        # duration, and the solid's species-A total for pure A crystals.
        totals = _inlet_rows(sim.GetRawMaterials(basis=basis), 'C01',
                             'Inlet_0')
        _assert_rows(totals, tuple({amount: values[amount]}
                                   for values in expected))
        assert totals.iloc[1][f'{amount}_A'] == pytest.approx(
            expected[1][amount], rel=RTOL)

    first, second = (value.reset_index(drop=True) for value in rows.values())
    np.testing.assert_allclose(first.to_numpy(dtype=float),
                               second.to_numpy(dtype=float), rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_msmpr_raw_slurry_stream_opex_prices_solid_feed(path, database):
    """Price the solid-number-rate feed's fed solid mass in ``GetOPEX``."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'C01': []})
    sim.C01 = _msmpr(path, _msmpr_slurry_feed(path, 'solid-number-rate'))
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    _, raw_cost, _ = sim.GetOPEX(cost_raw=PRICE)

    expected = _expected_slurry_rows(
        database, MSMPR_FEED_DISTRIB, MSMPR_FEED_VOL_FLOW, MSMPR_GRID,
        MSMPR_KV, SLURRY_LIQUID_COMPOSITION, RUNTIME, 'mass')
    costs = _inlet_rows(raw_cost, 'C01', 'Inlet_0')  # [USD]
    np.testing.assert_allclose(
        costs['mass'], [PRICE * values['mass'] for values in expected],
        rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_raw_cake_mixer_opex_prices_cake_phases(path, database):
    """Price raw Cake phases in ``GetOPEX`` beside a duty-reporting unit.

    A Mixer publishes no heat duty and ``GetDuties`` cannot yet build an
    empty duty table (issue #261), so a disconnected MSMPR fed with plain
    liquid supplies one duty row. Remove it once #261 is fixed.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': [], 'C01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [_cake(path, 0), _cake(path, 1)]
    sim.C01 = _msmpr(path, LiquidStream(path, mass_frac=FEED_COMPOSITION,
                                        temp=MSMPR_TEMP,
                                        vol_flow=MSMPR_FEED_VOL_FLOW))
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    _, raw_cost, _ = sim.GetOPEX(cost_raw=PRICE)

    for index, inlet in enumerate(('Inlet_0', 'Inlet_1')):
        expected = _expected_cake_rows(database, index, 'mass')
        costs = _inlet_rows(raw_cost, 'M01', inlet)  # [USD]
        np.testing.assert_allclose(
            costs['mass'], [PRICE * values['mass'] for values in expected],
            rtol=RTOL)
