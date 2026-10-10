"""Slurry feeds of ``DynamicCollector`` from crystallizer and other sources.

Issue #416: a slurry-fed collector delegates to a ``SemibatchCryst`` and
needs ``KinCryst`` and ``kwargs_cryst``, which only crystallizer sources
supply through ``Connection``. Missing or inconsistent settings must raise a
specific, actionable error from the settings check, before the inlet layout
is rebuilt or any phase is built. Issue #417: the
collector must read the feed liquid concentration [kg/m**3] for moment and
1D-FVM populations, raw or connected, and a non-MSMPR slurry source (the
PR #403 continuous solids ``Mixer``) must hand over the crystallizer input
names rather than the liquid-mixer ones.

The slurry follows the issue reproductions with the shipped five-species
database: 0.5 kg/s of liquid at 350 K and a crystal number rate of
0/1e6/1.25e5/0 #/um/s of pure A on the [0, 100, 200, 300] um grid (kv = 1),
optionally mixed with 1 kg/s of a second liquid at 300 K. Expected
concentrations w * rho_liq use the ideal mass-basis mixing rule with the
pure-component densities read from the JSON database; expected inventories
are the feed rate times the 10 s collection time plus the collector's seed.

Unit tests stop at validation or at the ``Connection`` handoff and need no
ODE backend; they also check that liquid sources, including the
``BatchToFlowConnector`` route, keep the liquid-mixer names. Tests marked
``assimulo`` integrate the collector with CVode, including MSMPR sources
whose ``target_comp`` is a name, list, tuple or NumPy array.

The caller-supplied kinetics, ``CrystKinetics(coeff_solub=[2000.])`` with
its default zero rates, is the undersaturated non-crystallizing fixture of
tests/test_collector_crystal_seed.py, so collected inventories equal the
feed. Collector temperatures are not asserted: they drift upward (about
4.5 K over 10 s) even for this isothermal, non-crystallizing feed, which
appears to be the semibatch slurry-feed energy defect of issue #265. Assert
that the collector temperature equals the feed temperature once #265 is
fixed.
"""

import json

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import DynamicCollector, Mixer
from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import BatchToFlowConnector, LiquidStream, SolidStream
from PharmaPy.Utilities import CoolingWater

UM_TO_M = 1e-6  # [m/um], exact SI conversion
GRID = np.array([0., 100., 200., 300.])  # [um], issue #416/#417 grid
NUMBER_RATE = np.array([0., 1e6, 1.25e5, 0.])  # [#/um/s], issue crystal feed
KV = 1.0  # [-], issue volumetric shape factor
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
SLURRY_LIQUID_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
FEED_COMPOSITION = np.array([0.4, 0.05, 0.15, 0.1, 0.3])  # [-], asymmetric
SLURRY_LIQUID_FLOW = 0.5  # [kg/s], issue slurry liquid
FEED_FLOW = 1.0  # [kg/s], issue Mixer liquid feed
HOT = 350.0  # [K], slurry temperature
COLD = 300.0  # [K], Mixer liquid feed temperature
RUNTIME = 10.0  # [s], issue collection time
NUM_MOMENTS = 4  # [-], orders 0-3, the default SolidPhase moment count
# Undersaturated caller kinetics: 2000 kg/m**3 exceeds every A
# concentration here (at most about 300 kg/m**3); default rates are zero.
SOLUBILITY = [2000.]  # [kg/m**3]
TARGET_SETTINGS = {'target_ind': 0, 'target_comp': 'A'}  # species A first
# CVode error limits: rtol [-] two orders below SOLVER_RTOL; atol [state
# units] four orders below the smallest nonzero state to resolve, the
# initial seed liquid volume of about 1.1e-8 m**3 (all other nonzero states,
# populations [#/um] or moments and concentrations [kg/m**3], are larger).
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-12}
SOLVER_RTOL = 1e-7  # [-], integrated inventories, as in other collector tests
RTOL = 1e-10  # [-], float64 roundoff of independently recomputed values
# Issue #417 printed the collected liquid volume to five significant figures
# (0.014971 m**3): half a unit in the last place is 5e-7 m**3, a relative
# 3.4e-5, bounded here by 5e-5. Fixture cross-check only.
ISSUE_LIQUID_VOLUME = 0.014971  # [m**3], issue #417 diagnostic
ISSUE_ROUNDING_RTOL = 5e-5  # [-]
# Issue #405 MSMPR fixture: undersaturated 310 K holdup on a 10/20/40 um grid.
MSMPR_GRID = np.array([10., 20., 40.])  # [um]
MSMPR_SEED = np.array([1e7, 2e7, 1e7])  # [#/um], holdup population
MSMPR_TEMP = 310.0  # [K]
MSMPR_TANK_VOL = 1e-3  # [m**3]
MSMPR_HOLDUP_VOL = 9e-4  # [m**3]
MSMPR_FEED_VOL_FLOW = 1e-5  # [m**3/s], liquid feed
MSMPR_RUNTIME = 1.0  # [s], short crystallizer and collector solves
COOLING_FLOW = 1e-5  # [m**3/s], issue #405 cooling water
COOLING_TEMP = 300.0  # [K]


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
    """Read species names and liquid densities from the JSON database.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    dict
        ``species`` in database order and ``rho_liq`` [kg/m**3], shape
        (num_species,).
    """
    with open(path) as handle:
        data = json.load(handle)
    species = list(data)
    return {'species': species,
            'rho_liq': np.array([data[name]['rho_liq']
                                 for name in species])}  # [kg/m**3]


def _trapezoid(values):
    """Integrate nodal values over GRID with the trapezoidal rule.

    Parameters
    ----------
    values : numpy.ndarray
        Integrand on GRID, shape (len(GRID),), in [unit/um].

    Returns
    -------
    float
        Integral in [unit].
    """
    return float(np.sum(np.diff(GRID) * (values[1:] + values[:-1]) / 2))


def _feed(database, mixed):
    """Derive the liquid and solid flows of the slurry reaching the collector.

    Parameters
    ----------
    database : dict
        Pure-component data from the ``database`` fixture.
    mixed : bool
        Whether the issue slurry was first mixed with the liquid feed.

    Returns
    -------
    dict
        ``composition`` [-], ``mass_conc`` [kg/m**3], liquid ``liquid_flow``
        [m**3/s], solid ``solid_flow`` [m**3/s] and total ``vol_flow``
        [m**3/s].
    """
    if mixed:
        liquid_mass = FEED_FLOW + SLURRY_LIQUID_FLOW  # [kg/s]
        composition = (FEED_FLOW * FEED_COMPOSITION
                       + SLURRY_LIQUID_FLOW * SLURRY_LIQUID_COMPOSITION
                       ) / liquid_mass  # [-]
    else:
        liquid_mass = SLURRY_LIQUID_FLOW  # [kg/s]
        composition = SLURRY_LIQUID_COMPOSITION  # [-]
    density = 1 / np.dot(composition, 1 / database['rho_liq'])  # [kg/m**3]
    liquid_flow = liquid_mass / density  # [m**3/s]
    solid_flow = KV * _trapezoid(NUMBER_RATE * GRID**3) * UM_TO_M**3  # [m**3/s]
    return {'composition': composition,
            'mass_conc': composition * density,  # [kg/m**3]
            'liquid_flow': liquid_flow,  # [m**3/s]
            'solid_flow': solid_flow,  # [m**3/s]
            'vol_flow': liquid_flow + solid_flow}  # [m**3/s]


def _slurry(path, population='fvm'):
    """Build the issue slurry stream with a 1D-FVM or moment population.

    Parameters
    ----------
    path : str
        Database path.
    population : {'fvm', 'moments'}, optional
        ``'fvm'`` lets the SolidStream carry NUMBER_RATE [#/um/s];
        ``'moments'`` gives the slurry the equivalent volume-specific
        moments [m**n/m**3] of that rate.

    Returns
    -------
    SlurryStream
        Raw slurry at HOT [K].
    """
    liquid = LiquidStream(path, mass_frac=SLURRY_LIQUID_COMPOSITION, temp=HOT,
                          mass_flow=SLURRY_LIQUID_FLOW)
    if population == 'fvm':
        solid = SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=HOT,
                            kv=KV, x_distrib=GRID.copy(),
                            distrib=NUMBER_RATE.copy())
        slurry = SlurryStream()
    else:
        vol_flow = liquid.vol_flow + KV * _trapezoid(
            NUMBER_RATE * GRID**3) * UM_TO_M**3  # [m**3/s]
        slurry = SlurryStream(vol_flow=vol_flow,
                              moments=_moments(NUMBER_RATE / vol_flow))
        solid = SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=HOT,
                            kv=KV)
    slurry.Phases = [liquid, solid]
    return slurry


def _moments(distrib):
    """Return trapezoidal moments of orders 0 to 3 in SI lengths.

    Parameters
    ----------
    distrib : numpy.ndarray
        Distribution on GRID [#/um] or [#/m**3/um], shape (len(GRID),).

    Returns
    -------
    numpy.ndarray
        Moments [m**n] or [m**n/m**3], shape (NUM_MOMENTS,).
    """
    return np.array([_trapezoid(distrib * GRID**order) * UM_TO_M**order
                     for order in range(NUM_MOMENTS)])


def _configure(collector):
    """Give a collector the caller-supplied crystallization settings.

    Parameters
    ----------
    collector : DynamicCollector
        Collector to configure.

    Returns
    -------
    DynamicCollector
        The same collector, for chaining.
    """
    collector.KinCryst = CrystKinetics(coeff_solub=SOLUBILITY)
    collector.kwargs_cryst = dict(TARGET_SETTINGS)
    return collector


def _solids_mixer(path):
    """Build the PR #403 continuous solids Mixer of liquid and slurry.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Unsolved Mixer of FEED_FLOW [kg/s] liquid at COLD [K] and the slurry.
    """
    mixer = Mixer()
    mixer.Inlets = [LiquidStream(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                                 mass_flow=FEED_FLOW), _slurry(path)]
    return mixer


# ---------- Issue #416: crystallization settings (core lane)

MISSING_PREFIX = ('A slurry-fed DynamicCollector delegates to a SemibatchCryst '
                  'and needs crystallization settings; missing: ')
SOURCE_NOTE = ('Only crystallizer sources (PharmaPy.Crystallizers units) '
               'supply KinCryst and kwargs_cryst automatically through a '
               'Connection; set them on the collector for any other slurry '
               'source.')
KINETICS_ITEM = ('KinCryst (crystallization kinetics, e.g. '
                 'PharmaPy.Kinetics.CrystKinetics)')
KWARGS_ITEM = ("kwargs_cryst (mapping, e.g. dict, with 'target_ind' and "
               "'target_comp')")
SETTINGS_CASES = {
    'none': (False, None, f'{KINETICS_ITEM}; {KWARGS_ITEM}'),
    'no-kinetics': (False, dict(TARGET_SETTINGS), KINETICS_ITEM),
    'no-kwargs': (True, None, KWARGS_ITEM),
    'no-keys-no-kinetics': (False, {}, f"{KINETICS_ITEM}; kwargs_cryst keys "
                            "['target_ind', 'target_comp']"),
    'no-target-comp': (True, {'target_ind': 0},
                       "kwargs_cryst keys ['target_comp']"),
    'no-target-ind': (True, {'target_comp': 'A'},
                      "kwargs_cryst keys ['target_ind']"),
    }  # kinetics supplied, kwargs_cryst, exact "missing: ..." list


def _assert_rejected_before_setup(collector, error):
    """Check that the settings check raised before any solve setup.

    Parameters
    ----------
    collector : DynamicCollector
        Collector whose ``solve_unit`` raised.
    error : pytest.ExceptionInfo
        The raised exception.
    """
    assert error.traceback[-1].name == '_check_crystallization_settings'
    # solve_unit rebuilds the Inlet layout without the unused population
    # name before reading inputs; the setter's layout still has both.
    assert {'distrib', 'mu_n'} <= set(collector.states_in_dict['Inlet'])
    assert collector.Phases is None
    assert collector.CrystInst is None


@pytest.mark.unit
@pytest.mark.parametrize('case', list(SETTINGS_CASES))
def test_missing_settings_fail_before_setup(path, case):
    """Name exactly the missing settings before the solve is set up."""
    kinetics, settings, expected_missing = SETTINGS_CASES[case]
    collector = DynamicCollector()
    collector.Inlet = _slurry(path)
    if kinetics:
        collector.KinCryst = CrystKinetics(coeff_solub=SOLUBILITY)
    collector.kwargs_cryst = settings

    with pytest.raises(ValueError) as error:
        collector.solve_unit(runtime=RUNTIME, verbose=False)

    message = str(error.value)
    assert message.startswith(MISSING_PREFIX)
    missing, note = message[len(MISSING_PREFIX):].split('. Only ', 1)
    assert missing == expected_missing
    assert 'Only ' + note == SOURCE_NOTE
    if case == 'no-kinetics':
        assert 'kwargs_cryst' not in missing
    _assert_rejected_before_setup(collector, error)


INCONSISTENT_CASES = {
    'wrong-index': ({'target_ind': 1, 'target_comp': 'A'}, ValueError,
                    r"\['target_ind'\] = 1 must be the index of 'A'.*: 0\.$"),
    'list-first-species': ({'target_ind': 2, 'target_comp': ['C', 'A']},
                           ValueError, r"must be the index of 'A'.*: 0\.$"),
    'float-index': ({'target_ind': 0.0, 'target_comp': 'A'}, TypeError,
                    r"\['target_ind'\] must be the integer index of 'A'.*"
                    r": 0; got 0\.0\.$"),
    # True equals 1, the index of B, but a flag is not an index.
    'bool-index': ({'target_ind': True, 'target_comp': 'B'}, TypeError,
                   r"must be the integer index of 'B'.*: 1; got True\.$"),
    'unknown-species': ({'target_ind': 0, 'target_comp': ['A', 'Z']},
                        ValueError, r"unknown: \['Z'\]\.$"),
    'empty-target': ({'target_ind': 0, 'target_comp': []}, ValueError,
                     r"must name at least one species from the inlet "
                     r"species \[.*\]\.$"),
    'non-name-target': ({'target_ind': 0, 'target_comp': 0}, TypeError,
                        r"\['target_comp'\] must be a species name"),
    'two-dimensional-target': ({'target_ind': 0,
                                'target_comp': np.array([['A']])},
                               TypeError, r"one-dimensional array of names"),
    'non-mapping': ([('target_ind', 0), ('target_comp', 'A')], TypeError,
                    r'kwargs_cryst must be a mapping \(e\.g\. dict\) .*; '
                    r'got list\.$'),
    'reserved-method': (dict(TARGET_SETTINGS, method='moments'), ValueError,
                        r"^kwargs_cryst keys \['method'\] are set by the "
                        r"DynamicCollector"),
    'reserved-both': (dict(TARGET_SETTINGS, method='1D-FVM', adiabatic=False),
                      ValueError, r"^kwargs_cryst keys \['method', "
                      r"'adiabatic'\] are set by the DynamicCollector"),
    }  # kwargs_cryst, exception type, message pattern


@pytest.mark.unit
@pytest.mark.parametrize('case', list(INCONSISTENT_CASES))
def test_inconsistent_settings_are_rejected(path, case):
    """Reject settings that would seed one species and crystallize another."""
    settings, error_type, pattern = INCONSISTENT_CASES[case]
    collector = DynamicCollector()
    collector.Inlet = _slurry(path)
    collector.KinCryst = CrystKinetics(coeff_solub=SOLUBILITY)
    collector.kwargs_cryst = settings

    with pytest.raises(error_type, match=pattern) as error:
        collector.solve_unit(runtime=RUNTIME, verbose=False)

    _assert_rejected_before_setup(collector, error)
    assert collector.kwargs_cryst is settings


VALID_SETTINGS = {
    'name': {'target_ind': 0, 'target_comp': 'A'},
    'list': {'target_ind': 0, 'target_comp': ['A']},
    'tuple': {'target_ind': 0, 'target_comp': ('A',)},
    'array': {'target_ind': 0, 'target_comp': np.array(['A'])},
    # A precedes C in the database, so A (index 0) is the target species.
    'list-out-of-order': {'target_ind': 0, 'target_comp': ['C', 'A']},
    'list-later-species': {'target_ind': 2, 'target_comp': ['D', 'C']},
    'numpy-index': {'target_ind': np.int64(0), 'target_comp': 'A'},
    'scale': {'target_ind': 0, 'target_comp': 'A', 'scale': 1e-9},  # [-]
    'interp-points': {'target_ind': 0, 'target_comp': 'A',
                      'num_interp_points': 7},  # [-], replaced by collector
    }


@pytest.mark.unit
@pytest.mark.parametrize('case', list(VALID_SETTINGS))
def test_valid_settings_pass_the_check_unchanged(path, case):
    """Accept every supported settings form without modifying it.

    Guards the core lane against a check that rejects valid settings. The
    solve path is integrated in the Assimulo lane for the out-of-order
    targets and, through MSMPR sources, for the name, list, tuple and array
    forms.
    """
    settings = VALID_SETTINGS[case]
    originals = dict(settings)  # same value objects
    snapshot = {key: np.array(value, copy=True)
                for key, value in settings.items()}  # contents
    collector = _configure(DynamicCollector())
    kinetics = collector.KinCryst
    collector.kwargs_cryst = settings
    collector.Inlet = _slurry(path)

    assert collector._check_crystallization_settings() is None

    assert collector.kwargs_cryst is settings
    assert collector.KinCryst is kinetics
    assert list(settings) == list(snapshot)
    for key, value in settings.items():
        assert value is originals[key]
        np.testing.assert_array_equal(np.asarray(value), snapshot[key])


@pytest.mark.unit
def test_mixer_connection_leaves_settings_to_the_caller(path):
    """Fail early when a non-crystallizer source feeds an unconfigured collector."""
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _solids_mixer(path)
    sim.C01 = DynamicCollector()

    with pytest.raises(ValueError) as error:
        sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': RUNTIME}},
                           verbose=False)

    assert str(error.value) == (f'{MISSING_PREFIX}{KINETICS_ITEM}; '
                                f'{KWARGS_ITEM}. {SOURCE_NOTE}')
    assert sim.C01.model_type == 'crystallizer'
    assert sim.C01.KinCryst is None and sim.C01.kwargs_cryst is None
    _assert_rejected_before_setup(sim.C01, error)


# ---------- Issue #417: connected input names (core lane)


@pytest.mark.unit
def test_mixer_connection_hands_collector_crystallizer_inputs(path, database):
    """Transfer the slurry liquid concentration from a solids Mixer."""
    feed = _feed(database, mixed=True)
    mixer = _solids_mixer(path)
    mixer.solve_unit()
    collector = DynamicCollector()

    Connection(mixer, collector).transfer_data()

    transferred = collector.Inlet.y_inlet
    # No vol_flow: the solids Mixer publishes no flow-named states, so the
    # collector reads vol_flow from the transferred stream below. Provisional
    # for single-sample sources; a profiled source would need it published.
    assert list(transferred) == ['mass_conc', 'temp', 'distrib']
    np.testing.assert_allclose(transferred['mass_conc'][0], feed['mass_conc'],
                               rtol=RTOL)
    np.testing.assert_allclose(transferred['distrib'][0],
                               NUMBER_RATE / feed['vol_flow'], rtol=RTOL)
    np.testing.assert_array_equal(transferred['temp'], mixer.outputs['temp'])

    inputs = collector.get_inputs_new(0.)['Inlet']
    np.testing.assert_allclose(inputs['mass_conc'], feed['mass_conc'],
                               rtol=RTOL)
    assert inputs['vol_flow'] == pytest.approx(feed['vol_flow'], rel=RTOL)
    np.testing.assert_allclose(inputs['distrib'],
                               NUMBER_RATE / feed['vol_flow'], rtol=RTOL)
    assert inputs['temp'] == mixer.outputs['temp'][0]


@pytest.mark.unit
def test_liquid_connection_keeps_liquid_mixer_inputs(path):
    """Keep the liquid-mixer names for a liquid source."""
    mixer = Mixer()
    mixer.Inlets = [LiquidStream(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                                 mass_flow=FEED_FLOW),
                    LiquidStream(path, mass_frac=SLURRY_LIQUID_COMPOSITION,
                                 temp=HOT, mass_flow=SLURRY_LIQUID_FLOW)]
    mixer.solve_unit()
    collector = DynamicCollector()

    Connection(mixer, collector).transfer_data()

    assert collector.model_type == 'liquid_mixer'
    assert set(collector.Inlet.y_inlet) == {'mass_frac', 'mass_flow', 'temp'}


@pytest.mark.unit
def test_batch_to_flow_liquid_keeps_liquid_mixer_inputs(path):
    """Keep the liquid-mixer names on the BatchToFlowConnector route.

    The connector discharges only liquid holdups and rejects solids-bearing
    ones (issue #423; tests/test_batch_to_flow_connector.py), so this route
    cannot carry a slurry to the collector.
    """
    batch_mass = 2.0  # [kg], liquid charge
    cycle_time = 10.0  # [s], emptying time of the charge
    connector = BatchToFlowConnector(cycle_time=cycle_time)
    connector.Phases = LiquidPhase(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                                   mass=batch_mass)
    connector.solve_unit()
    collector = DynamicCollector()

    Connection(connector, collector).transfer_data()

    transferred = collector.Inlet.y_inlet
    discharge = batch_mass / cycle_time  # [kg/s]
    assert collector.model_type == 'liquid_mixer'
    assert set(transferred) == {'mass_frac', 'mass_flow', 'temp'}
    # One instantaneous sample with a leading time axis (issue #426).
    np.testing.assert_allclose(transferred['mass_flow'], [discharge],
                               rtol=RTOL)
    np.testing.assert_allclose(transferred['mass_frac'], [FEED_COMPOSITION],
                               rtol=RTOL)
    inputs = collector.get_inputs_new(0.)['Inlet']
    assert inputs['mass_flow'] == pytest.approx(discharge, rel=RTOL)
    np.testing.assert_allclose(inputs['mass_frac'], FEED_COMPOSITION,
                               rtol=RTOL)


# ---------- Integration (Assimulo lane)


def _assert_collected(collector, feed, initial_population):
    """Check the collected liquid volume and crystals against feed x time.

    Parameters
    ----------
    collector : DynamicCollector
        Collector solved for RUNTIME [s] from its initial seed.
    feed : dict
        Feed flows from :func:`_feed`.
    initial_population : str
        ``'total_distrib'`` (1D-FVM, [#/um]) or ``'mu_n'`` (moments,
        [m**n]).
    """
    seed = np.sqrt(np.finfo(float).eps)  # [m**3], established collector seed
    solid_fraction = feed['solid_flow'] / feed['vol_flow']  # [-]
    liquid = seed * (1 - solid_fraction) + feed['liquid_flow'] * RUNTIME  # [m**3]
    assert collector.result.vol[-1] == pytest.approx(liquid, rel=SOLVER_RTOL)
    # Seed population on the slurry basis plus the fed number rate.
    collected = NUMBER_RATE * (seed / feed['vol_flow'] + RUNTIME)  # [#/um]
    if initial_population == 'total_distrib':
        np.testing.assert_allclose(collector.result.total_distrib[-1],
                                   collected, rtol=SOLVER_RTOL)
    else:
        np.testing.assert_allclose(collector.result.mu_n[-1],
                                   _moments(collected), rtol=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('population, inventory',
                         [('fvm', 'total_distrib'), ('moments', 'mu_n')])
def test_raw_slurry_collector_reads_feed_concentration(path, database,
                                                       population, inventory):
    """Seed and collect a raw slurry with its own liquid concentration."""
    pytest.importorskip('assimulo')
    feed = _feed(database, mixed=False)
    collector = _configure(DynamicCollector())
    collector.Inlet = _slurry(path, population)

    collector.solve_unit(runtime=RUNTIME, verbose=False,
                         sundials_opts=dict(SOLVER_OPTIONS))

    inputs = collector.get_inputs_new(0.)['Inlet']
    np.testing.assert_allclose(inputs['mass_conc'], feed['mass_conc'],
                               rtol=RTOL)
    # The seed liquid starts at the feed concentration and keeps it.
    np.testing.assert_allclose(collector.result.mass_conc[0],
                               feed['mass_conc'], rtol=RTOL)
    np.testing.assert_allclose(collector.result.mass_conc[-1],
                               feed['mass_conc'], rtol=SOLVER_RTOL)
    _assert_collected(collector, feed, inventory)


@pytest.mark.assimulo
@pytest.mark.integration
def test_mixer_to_collector_flowsheet_collects_feed(path, database):
    """Integrate a solids Mixer feed into a caller-configured collector."""
    pytest.importorskip('assimulo')
    feed = _feed(database, mixed=True)
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _solids_mixer(path)
    sim.C01 = _configure(DynamicCollector())
    caller_settings = sim.C01.kwargs_cryst

    sim.SolveFlowsheet(kwargs_run={'C01': {
        'runtime': RUNTIME, 'verbose': False,
        'sundials_opts': dict(SOLVER_OPTIONS)}}, verbose=False)

    collector = sim.C01
    assert collector.kwargs_cryst is caller_settings
    # No vol_flow in y_inlet; see the comment in
    # test_mixer_connection_hands_collector_crystallizer_inputs.
    assert set(collector.Inlet.y_inlet) == {'mass_conc', 'temp', 'distrib'}
    np.testing.assert_allclose(collector.result.mass_conc[0],
                               feed['mass_conc'], rtol=RTOL)
    _assert_collected(collector, feed, 'total_distrib')
    # Fixture cross-check against the issue #417 diagnostic,
    # (1.6971e-3 - 2.0e-4) m**3/s over 10 s, printed to 5 significant figures.
    assert feed['liquid_flow'] * RUNTIME == pytest.approx(
        ISSUE_LIQUID_VOLUME, rel=ISSUE_ROUNDING_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('case', ['list-out-of-order', 'list-later-species'])
def test_solve_seeds_and_crystallizes_the_same_target(path, database, case):
    """Seed and crystallize the first target_comp species in inlet order."""
    pytest.importorskip('assimulo')
    settings = VALID_SETTINGS[case]
    collector = _configure(DynamicCollector())
    collector.kwargs_cryst = settings
    collector.Inlet = _slurry(path)

    collector.solve_unit(runtime=RUNTIME, verbose=False)

    target = settings['target_ind']  # [-]
    species = database['species']
    first_target = min(species.index(name) for name in settings['target_comp'])
    assert target == first_target
    assert collector.CrystInst.target_ind == target
    # Seed crystals are pure target species; SolidPhase stores absent
    # species as machine epsilon rather than zero.
    expected = np.eye(len(species))[target]  # [-]
    np.testing.assert_allclose(collector.Solid_1.mass_frac, expected, rtol=0,
                               atol=np.finfo(float).eps)


# target_comp forms the crystallizers accept: a name (stored as a list),
# a list, a tuple and a one-dimensional NumPy array of names.
TARGET_FORMS = {'name': 'A', 'list': ['A'], 'tuple': ('A',),
                'array': np.array(['A'])}


@pytest.mark.assimulo
@pytest.mark.integration
def test_msmpr_source_still_supplies_settings_and_inputs(path, database):
    """Collect an MSMPR outlet for every target_comp form it accepts."""
    pytest.importorskip('assimulo')
    # The MSMPR holdup and feed share the slurry liquid composition and stay
    # undersaturated, so its outlet liquid keeps that composition.
    expected_conc = _feed(database, mixed=False)['mass_conc']  # [kg/m**3]
    final_volumes = {}
    for form, target in TARGET_FORMS.items():
        cryst = MSMPR(target, method='1D-FVM', vol_tank=MSMPR_TANK_VOL)
        cryst.Phases = (
            LiquidPhase(path, temp=MSMPR_TEMP, vol=MSMPR_HOLDUP_VOL,
                        mass_frac=SLURRY_LIQUID_COMPOSITION),
            SolidPhase(path, temp=MSMPR_TEMP, x_distrib=MSMPR_GRID.copy(),
                       distrib=MSMPR_SEED.copy(), kv=KV,
                       mass_frac=SOLID_COMPOSITION))
        cryst.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
        cryst.Utility = CoolingWater(vol_flow=COOLING_FLOW,
                                     temp_in=COOLING_TEMP)
        cryst.Inlet = LiquidStream(path, temp=MSMPR_TEMP,
                                   mass_frac=SLURRY_LIQUID_COMPOSITION,
                                   vol_flow=MSMPR_FEED_VOL_FLOW)
        sim = SimulationExec(path, {'C01': ['H01'], 'H01': []})
        sim.C01 = cryst
        sim.H01 = DynamicCollector()

        sim.SolveFlowsheet(kwargs_run={
            'C01': {'runtime': MSMPR_RUNTIME, 'verbose': False},
            'H01': {'runtime': MSMPR_RUNTIME, 'verbose': False,
                    'sundials_opts': dict(SOLVER_OPTIONS)}}, verbose=False)

        collector = sim.H01
        settings = collector.kwargs_cryst
        assert collector.KinCryst is cryst.Kinetics
        assert set(settings) == {'target_ind', 'target_comp', 'scale'}
        assert settings['target_ind'] == 0  # species A is first in the database
        assert list(settings['target_comp']) == ['A'], form
        assert settings['scale'] == 1  # [-], MSMPR default scale
        assert {'mass_conc', 'temp', 'distrib'} <= set(collector.Inlet.y_inlet)
        np.testing.assert_allclose(collector.result.mass_conc[0],
                                   expected_conc, rtol=SOLVER_RTOL)
        final_volumes[form] = collector.result.vol[-1]  # [m**3]

    # The target form must not change the collected liquid.
    np.testing.assert_allclose(list(final_volumes.values()),
                               final_volumes['name'], rtol=RTOL)
