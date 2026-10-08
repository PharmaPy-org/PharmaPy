"""Connected-input sample detection of the solids ``Mixer`` (issue #276).

A solids mixer decides whether a connected inlet is profiled from the
number of samples in its ``time_upstream`` [s], not from the number of state
names in ``y_upstream``. Single-sample inlets are constant feeds whose
attached phase state is mixed; multi-sample inlets raise ``ValueError``
before any balance runs.

The fixture is the issue reproduction with the shipped five-species
database: 1 kg of liquid at 350 K mixed with a 0.001 m**3 slurry at 300 K
on the [0, 100, 200, 300] um grid, with 1e9 and 1.25e8 #/m**3/um in the
inner bins. Connection cases use real upstream Mixers, a real
``Connection.transfer_data()``, and a two-unit ``SimulationExec``
flowsheet; their downstream balances must equal those of the same phases
mixed directly. The profiled source is a continuous liquid Mixer whose raw
feed carries a two- or three-sample profile. No ODE backend is needed.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = pytest.mark.unit

ISSUE_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #276
# Asymmetric liquid feeds for the connection chains, so a species-order or
# weighting error changes the mixed composition.
FEED_COMPOSITION = np.array([0.4, 0.05, 0.15, 0.1, 0.3])  # [-]
SECOND_COMPOSITION = np.array([0.2, 0.3, 0.1, 0.1, 0.3])  # [-]
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
GRID = np.array([0., 100., 200., 300.])  # [um], issue #276 size grid
SLURRY_DISTRIB = np.array([0., 1e9, 1.25e8, 0.])  # [#/m**3/um], issue #276
SLURRY_VOL = 1e-3  # [m**3], issue #276 slurry volume
KV = 1.0  # [-], issue #276 volumetric shape factor
COLD = 300.0  # [K], issue slurry and chain liquid feed
HOT = 350.0  # [K], issue liquid feed and chain slurry
WARM = 320.0  # [K], second chain liquid feed
FEED_MASS = 1.0  # [kg], issue #276 liquid feed
SECOND_MASS = 0.7  # [kg], unequal to FEED_MASS so weighting errors show
# Outlet temperature the issue reports for the fixture without y_upstream;
# a cross-check of the independent root below, not its source.
ISSUE_TEMP = 312.391280366  # [K], printed to 1e-9 K
ISSUE_TEMP_ABS = 1e-9  # [K], last printed digit of ISSUE_TEMP
# Bracket for the independent energy-balance root; it contains every
# adiabatic mixing temperature between the COLD and HOT inlets.
TEMP_BRACKET = (COLD, HOT)  # [K]
# Root-finder step tolerance, far below RTOL * COLD (3e-8 K), so the
# independent root does not limit the comparison.
ROOT_XTOL = 1e-12  # [K]
# Relative tolerance [-] for independently recomputed sums, roots, and the
# mixing chains. scipy.optimize.newton stops at an absolute step of
# 1.48e-8 K, and a chain has two sequential roots, so temperatures can differ
# by about 3e-8 K, i.e. 1e-10 of 300 K. Other quantities have float64
# roundoff only.
RTOL = 1e-10  # [-]
# Connected feed profiles of two and three samples: the smallest profile
# and one more, so a guard that tolerates two samples is caught. Each row
# holds times [s], a rising mass flow [kg/s], and a warming temperature [K];
# the window end differs between them so the reported window is checked.
PROFILES = {
    'two-samples': (np.array([0., 1.5]), np.array([1.0, 1.3]),
                    np.array([300., 308.])),
    'three-samples': (np.array([0., 1., 2.]), np.array([1.0, 1.2, 1.4]),
                      np.array([300., 305., 310.])),
    }  # [s], [kg/s], [K]
# Expected message fragment per profile: sample count and [t0, tN] window [s].
PROFILE_MESSAGES = {
    'two-samples': r'2 samples over \[0\.0, 1\.5\] s',
    'three-samples': r'3 samples over \[0\.0, 2\.0\] s',
    }
SLURRY_LIQUID_FLOW = 0.5  # [kg/s], liquid in the continuous slurry
SOLID_NUMBER_RATE = np.array([0., 1e6, 1.25e5, 0.])  # [#/um/s], crystal feed


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


def _slurry(path, temp):
    """Build the issue #276 batch slurry.

    Parameters
    ----------
    path : str
        Database path.
    temp : float
        Temperature of both phases [K].

    Returns
    -------
    Slurry
        Slurry of volume SLURRY_VOL [m**3] with liquid composition
        ISSUE_COMPOSITION [-] and total population
        SLURRY_DISTRIB * SLURRY_VOL [#/um].
    """
    slurry = Slurry(vol=SLURRY_VOL, x_distrib=GRID.copy(),
                    distrib=SLURRY_DISTRIB.copy())
    slurry.Phases = (
        LiquidPhase(path, mass_frac=ISSUE_COMPOSITION, temp=temp),
        SolidPhase(path, mass_frac=SOLID_COMPOSITION, kv=KV, temp=temp))
    return slurry


def _issue_mixer(path):
    """Build the unsolved issue #276 mixer.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Mixer of FEED_MASS [kg] of liquid at HOT [K] and the slurry at
        COLD [K].
    """
    mixer = Mixer()
    mixer.Inlets = [LiquidPhase(path, mass_frac=ISSUE_COMPOSITION,
                                mass=FEED_MASS, temp=HOT),
                    _slurry(path, COLD)]
    return mixer


def _liquid(path, composition, mass, temp):
    """Build a batch liquid feed.

    Parameters
    ----------
    path : str
        Database path.
    composition : numpy.ndarray
        Mass fractions [-] in database order.
    mass : float
        Liquid mass [kg].
    temp : float
        Temperature [K].

    Returns
    -------
    LiquidPhase
        Liquid feed.
    """
    return LiquidPhase(path, mass_frac=composition, mass=mass, temp=temp)


def _three_inlet_mixer(path):
    """Solve the chains' reference: all three feeds mixed at once.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Solved mixer of FEED_MASS [kg] at COLD [K], the slurry at HOT [K],
        and SECOND_MASS [kg] at WARM [K].
    """
    mixer = Mixer()
    mixer.Inlets = [_liquid(path, FEED_COMPOSITION, FEED_MASS, COLD),
                    _slurry(path, HOT),
                    _liquid(path, SECOND_COMPOSITION, SECOND_MASS, WARM)]
    mixer.solve_unit()
    return mixer


def _assert_outputs_close(actual, expected, rtol):
    """Compare published solids-mixer outputs name by name.

    Parameters
    ----------
    actual, expected : dict
        Published ``outputs`` of two solved solids mixers.
    rtol : float
        Relative tolerance [-]; zero requires exact equality.
    """
    assert list(actual) == list(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=rtol, atol=0,
                                   err_msg=name)


def _independent_issue_temp(path):
    """Solve the issue fixture's adiabatic phase-level energy balance.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    float
        Outlet temperature [K] at which the outlet liquid and solid phase
        enthalpies [J] equal the inlet phase enthalpies.

    Notes
    -----
    All liquids share ISSUE_COMPOSITION, so the outlet liquid has that
    composition and the summed liquid mass [kg]; the solid keeps its mass.
    """
    feed = _liquid(path, ISSUE_COMPOSITION, FEED_MASS, HOT)
    slurry = _slurry(path, COLD)
    inlet_phases = (feed, slurry.Liquid_1, slurry.Solid_1)
    inlet_energy = sum(phase.mass * phase.getEnthalpy(basis='mass')
                       for phase in inlet_phases)  # [J]
    liquid_mass = feed.mass + slurry.Liquid_1.mass  # [kg]
    solid = slurry.Solid_1

    def residual(temp):
        """Return outlet-minus-inlet enthalpy [J] at temperature ``temp`` [K]."""
        liquid_h = feed.getEnthalpy(temp=temp, mass_frac=ISSUE_COMPOSITION,
                                    basis='mass')  # [J/kg]
        solid_h = solid.getEnthalpy(temp=temp, basis='mass')  # [J/kg]
        return np.asarray(liquid_mass * liquid_h + solid.mass * solid_h
                          - inlet_energy).item()  # [J]

    return brentq(residual, *TEMP_BRACKET, xtol=ROOT_XTOL)


@pytest.mark.parametrize('time_upstream', [0.0, np.zeros(1)],
                         ids=['scalar-time', 'one-element-time'])
@pytest.mark.parametrize('form', ['state-dict', 'one-row-array'])
def test_single_sample_upstream_matches_static_fixture(path, form, time_upstream):
    """Mix single-sample connected inlets exactly as unconnected phases.

    The issue's two-key state dictionary and its one-row-array
    counterexample both carry one upstream sample. Their values are not
    consumed, so the outputs must be bit-identical to the static fixture.
    """
    static = _issue_mixer(path)
    static.solve_unit()
    connected = _issue_mixer(path)
    for inlet in connected.Inlets:
        inlet.time_upstream = time_upstream  # [s], one static sample
        if form == 'state-dict':
            inlet.y_upstream = {'temp': np.array([COLD]),
                                'mass': np.array([FEED_MASS])}  # [K], [kg]
        else:
            inlet.y_upstream = np.array([[COLD, FEED_MASS]])  # [K], [kg]

    connected.solve_unit()

    _assert_outputs_close(connected.outputs, static.outputs, rtol=0)
    temp = connected.outputs['temp'][0]  # [K]
    assert temp == pytest.approx(_independent_issue_temp(path), rel=RTOL)
    assert temp == pytest.approx(ISSUE_TEMP, abs=ISSUE_TEMP_ABS)


def test_liquid_mixer_connection_feeds_solids_mixer(path):
    """Mix a connected liquid Mixer outlet as its attached liquid phase."""
    upstream = Mixer()
    upstream.Inlets = [_liquid(path, FEED_COMPOSITION, FEED_MASS, COLD),
                       _liquid(path, SECOND_COMPOSITION, SECOND_MASS, WARM)]
    upstream.solve_unit()
    downstream = Mixer()

    Connection(upstream, downstream).transfer_data()
    connected = downstream.Inlets[0]
    assert isinstance(connected.y_upstream, dict) and len(connected.y_upstream) > 1
    assert np.size(connected.time_upstream) == 1
    downstream.Inlets = _slurry(path, HOT)
    downstream.solve_unit()

    static = Mixer()
    static.Inlets = [_liquid(path, upstream.Outlet.mass_frac,
                             upstream.Outlet.mass, upstream.Outlet.temp),
                     _slurry(path, HOT)]
    static.solve_unit()
    _assert_outputs_close(downstream.outputs, static.outputs, rtol=RTOL)

    slurry = _slurry(path, HOT)
    liquid_mass = FEED_MASS + SECOND_MASS + slurry.Liquid_1.mass  # [kg]
    composition = (FEED_MASS * FEED_COMPOSITION
                   + SECOND_MASS * SECOND_COMPOSITION
                   + slurry.Liquid_1.mass * ISSUE_COMPOSITION) / liquid_mass  # [-]
    outputs = downstream.outputs
    assert outputs['mass_liq'][0] == pytest.approx(liquid_mass, rel=RTOL)
    assert outputs['mass_solid'][0] == pytest.approx(slurry.Solid_1.mass, rel=RTOL)
    np.testing.assert_allclose(outputs['mass_frac'][0], composition, rtol=RTOL)
    np.testing.assert_allclose(downstream.Outlet.Solid_1.distrib,
                               SLURRY_DISTRIB * SLURRY_VOL, rtol=RTOL)
    _assert_outputs_close(outputs, _three_inlet_mixer(path).outputs, rtol=RTOL)


def test_solids_mixer_connection_feeds_solids_mixer(path):
    """Mixing in two connected stages equals mixing all feeds at once.

    Adiabatic mixing of ideal phases is associative: the connected chain
    and the single three-inlet mixer share masses, composition,
    population, and temperature.
    """
    upstream = Mixer()
    upstream.Inlets = [_liquid(path, FEED_COMPOSITION, FEED_MASS, COLD),
                       _slurry(path, HOT)]
    upstream.solve_unit()
    downstream = Mixer()

    Connection(upstream, downstream).transfer_data()
    downstream.Inlets = _liquid(path, SECOND_COMPOSITION, SECOND_MASS, WARM)
    downstream.solve_unit()

    assert isinstance(downstream.Inlets[0], Slurry)
    assert isinstance(downstream.Outlet, Slurry)
    _assert_outputs_close(downstream.outputs, _three_inlet_mixer(path).outputs,
                          rtol=RTOL)


def test_two_unit_flowsheet_matches_three_inlet_mixer(path):
    """Solve a solids Mixer feeding a solids Mixer in a flowsheet."""
    sim = SimulationExec(path, {'M01': ['M02'], 'M02': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [_liquid(path, FEED_COMPOSITION, FEED_MASS, COLD),
                      _slurry(path, HOT)]
    sim.M02 = Mixer()
    sim.M02.Inlets = _liquid(path, SECOND_COMPOSITION, SECOND_MASS, WARM)

    sim.SolveFlowsheet(verbose=False)

    assert sim.time_processing == {'M01': 0.0, 'M02': 0.0}
    _assert_outputs_close(sim.M02.outputs, _three_inlet_mixer(path).outputs,
                          rtol=RTOL)


@pytest.mark.parametrize('profile', list(PROFILES))
def test_profiled_connected_inlet_raises_before_balance(path, profile):
    """Reject a connected multi-sample profile with an actionable message."""
    times, flows, temps = PROFILES[profile]  # [s], [kg/s], [K]
    feed = LiquidStream(path, mass_frac=FEED_COMPOSITION, temp=COLD,
                        mass_flow=flows[0])
    feed.time_upstream = times  # [s]
    feed.y_upstream = feed.y_inlet = {
        'mass_flow': flows,  # [kg/s]
        'mass_frac': np.tile(FEED_COMPOSITION, (len(times), 1)),  # [-]
        'temp': temps}  # [K]
    source = Mixer()
    source.Inlets = [feed]
    source.solve_unit()
    np.testing.assert_array_equal(source.result.time, times)
    slurry = SlurryStream()
    slurry.Phases = [
        LiquidStream(path, mass_frac=ISSUE_COMPOSITION, temp=HOT,
                     mass_flow=SLURRY_LIQUID_FLOW),
        SolidStream(path, mass_frac=SOLID_COMPOSITION, temp=HOT, kv=KV,
                    x_distrib=GRID.copy(), distrib=SOLID_NUMBER_RATE.copy())]
    downstream = Mixer()
    downstream.Inlets = slurry
    Connection(source, downstream).transfer_data()

    with pytest.raises(ValueError, match=(
            r'Mixer inlet 1 carries a connected profile of '
            + PROFILE_MESSAGES[profile] + r', but solids mixing is static')):
        downstream.solve_unit()
    # No balance ran: balances_solids would have created the Outlet.
    assert not hasattr(downstream, 'Outlet')
    assert downstream.outputs is None
