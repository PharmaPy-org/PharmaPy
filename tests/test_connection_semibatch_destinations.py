"""Semibatch destinations of flowsheet connections (issue #438).

A semibatch unit other than a ``DynamicCollector`` holds an initial charge
(``Phases``) and integrates a feed (``Inlet``). A connected continuous
source becomes its feed, replacing a pre-set ``Inlet``, and the unit must
already be charged; the holdup of any other source charges an uncharged
unit, or the charge it received on the previous solve, but never adds a
second charge to one the user assigned. Rejections raise
``PharmaPyValueError`` before either unit is changed. Before the fix a
charged destination silently ignored the transfer and an uncharged one
took a stream as its holdup.

Reactor cases use the synthetic four-species database and the
``configured`` ``SemibatchReactor`` of
tests/test_reactor_controls_events_continuation.py (2 L charge, 4 L tank,
own feed of 2 L per 1800 s), fed by the issue reproduction Mixer: 1.0 and
0.5 kg/s at 300 K of mass fractions (0.5, 0.2, 0.3, 0) and
(0.2, 0.5, 0.3, 0). The reactor volume obeys dV/dt = vol_flow of its feed,
so the expected growth is the feed volume flow times the run time, with the
volume flow derived from the database liquid densities by ideal mixing,
1/rho = sum(w_i/rho_i). Evaporator cases use the synthetic UNIQUAC binary
of tests/test_evaporator_consistency.py. Unit cases solve only static
Mixer sources and need no ODE backend; ``assimulo`` cases integrate the
destinations with the real solvers.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import SemibatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater
from test_reactor_controls_events_continuation import configured
from test_reactor_correctness import (
    CONCENTRATIONS, HEAT_REFERENCE_TEMPERATURE, RATE_CONSTANT, REACTION_HEAT,
    REACTOR_VOLUME, RESIDENCE_TIME, TEMPERATURE, THERMO_PATH,
    UTILITY_MASS_FLOW, UTILITY_TEMPERATURE)

FEED_A = np.array([.5, .2, .3, 0.])  # [-], mass fractions, issue #438
FEED_B = np.array([.2, .5, .3, 0.])  # [-], mass fractions, issue #438
FLOW_A = 1.0  # [kg/s], issue #438
FLOW_B = 0.5  # [kg/s], issue #438
FEED_TEMP = 300.0  # [K], issue #438
RUNTIME = 1.0  # [s], issue #438 reactor run
# Batch charge of a semibatch reactor: about 1.8 L, below the 4 L tank of
# ``configured``; dilute reactants A and B and product C in the solvent.
CHARGE_MASS = 1.6  # [kg]
CHARGE_FRACTIONS = np.array([0.1, 0.1, 0.05, 0.75])  # [-], mass fractions
RTOL = 1e-12  # [-], float64 roundoff of algebraic conversions
# [-], CVode's default relative tolerance; dV/dt is the constant feed flow,
# so the integrated volume differs from flow times time by solver error only.
SOLVER_RTOL = 1e-6
RUN = {'R01': {'runtime': RUNTIME, 'verbose': False}}

STREAM_INTO_UNCHARGED = (
    r"^Destination 'R01' \(SemibatchReactor\) is a semibatch unit without "
    r"an initial charge: the stream \[kg/s\] of continuous source 'M01' "
    r"\(Mixer\) becomes its Inlet \(feed\), never its initial holdup "
    r"\[kg\]\. Assign the initial charge to its Phases before solving the "
    r"flowsheet\.$")
HOLDUP_INTO_CHARGED = (
    r"^Destination 'R01' \(SemibatchReactor\) already holds an initial "
    r"charge \(Phases\) assigned outside the flowsheet, and the holdup "
    r"\[kg\] of source 'M01' \(Mixer\), which is not continuous, would be a "
    r"second initial charge\. Remove the destination's own charge so that "
    r"the transferred holdup charges it, or mix both holdups in a batch "
    r"Mixer connected to it\.$")


def _densities():
    """Read the pure liquid densities of the four-species database.

    Returns
    -------
    numpy.ndarray
        Densities [kg/m**3], in database order.
    """
    with open(THERMO_PATH) as handle:
        data = json.load(handle)
    return np.array([value['rho_liq']
                     for value in data.values()])  # [kg/m**3]


def _stream_mixer():
    """Build the issue #438 continuous Mixer.

    Returns
    -------
    Mixer
        Unsolved Mixer of FLOW_A and FLOW_B [kg/s].
    """
    mixer = Mixer()
    mixer.Inlets = [
        LiquidStream(THERMO_PATH, mass_flow=FLOW_A, mass_frac=FEED_A,
                     temp=FEED_TEMP),
        LiquidStream(THERMO_PATH, mass_flow=FLOW_B, mass_frac=FEED_B,
                     temp=FEED_TEMP)]
    return mixer


def _holdup_mixer():
    """Build a batch Mixer holding a reactor charge.

    Returns
    -------
    Mixer
        Unsolved Mixer of CHARGE_MASS [kg] of CHARGE_FRACTIONS [-] at the
        fixture temperature [K].
    """
    mixer = Mixer()
    mixer.Inlets = LiquidPhase(THERMO_PATH, temp=TEMPERATURE,
                               mass=CHARGE_MASS, mass_frac=CHARGE_FRACTIONS)
    return mixer


def _feed_volume_flow():
    """Return the volume flow of the mixed issue #438 feed.

    Returns
    -------
    float
        Ideal-mixing volume flow [m**3/s] of FLOW_A + FLOW_B [kg/s].
    """
    densities = _densities()  # [kg/m**3]
    flow = FLOW_A + FLOW_B  # [kg/s]
    fractions = (FLOW_A * FEED_A + FLOW_B * FEED_B) / flow  # [-]
    return flow * np.sum(fractions / densities)  # [m**3/s]


def _charge_volume():
    """Return the volume of the batch Mixer's outlet holdup.

    Returns
    -------
    float
        Ideal-mixing volume [m**3] of CHARGE_MASS [kg].
    """
    densities = _densities()  # [kg/m**3]
    return CHARGE_MASS * np.sum(CHARGE_FRACTIONS / densities)  # [m**3]


def _uncharged_reactor():
    """Build the fixture semibatch reactor without an initial charge.

    Returns
    -------
    SemibatchReactor
        Reactor with the fixture kinetics, utility and own feed of
        REACTOR_VOLUME / RESIDENCE_TIME [m**3/s], as ``configured``, but
        no ``Phases``.
    """
    reactor = SemibatchReactor(vol_tank=2 * REACTOR_VOLUME)  # [m**3]
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[RATE_CONSTANT], ea_params=[0.0],
        rxn_list=['A + B --> C'], delta_hrxn=REACTION_HEAT,
        temp_ref=HEAT_REFERENCE_TEMPERATURE,
        tref_hrxn=HEAT_REFERENCE_TEMPERATURE)
    reactor.Utility = CoolingWater(temp_in=UTILITY_TEMPERATURE,
                                   mass_flow=UTILITY_MASS_FLOW)
    reactor.Inlet = LiquidStream(
        THERMO_PATH, temp=TEMPERATURE, mole_conc=CONCENTRATIONS,
        vol_flow=REACTOR_VOLUME / RESIDENCE_TIME)
    return reactor


@pytest.mark.unit
def test_stream_is_rejected_by_an_uncharged_semibatch_reactor():
    """Never install a stream as a semibatch unit's initial holdup."""
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _stream_mixer()
    sim.R01 = _uncharged_reactor()
    own_inlet = sim.R01.Inlet

    with pytest.raises(PharmaPyValueError, match=STREAM_INTO_UNCHARGED):
        sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.R01.Phases is None
    assert sim.R01.Inlet is own_inlet
    assert sim.M01.Outlet.y_upstream is None


@pytest.mark.unit
def test_holdup_is_rejected_by_a_user_charged_semibatch_reactor():
    """Refuse a second initial charge instead of ignoring the transfer."""
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _holdup_mixer()
    sim.R01 = configured(SemibatchReactor)
    own_charge = sim.R01.Phases
    own_liquid = sim.R01.Liquid_1

    with pytest.raises(PharmaPyValueError, match=HOLDUP_INTO_CHARGED):
        sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.R01.Phases is own_charge
    assert sim.R01.Phases == [own_liquid]
    assert not sim.R01.material_from_upstream
    assert sim.M01.Outlet.y_upstream is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_stream_feeds_a_charged_semibatch_reactor():
    """Feed the reactor with the Mixer outlet instead of its own inlet."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _stream_mixer()
    sim.R01 = configured(SemibatchReactor)
    own_inlet = sim.R01.Inlet
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    feed = sim.R01.Inlet
    assert feed is not own_inlet
    assert feed is sim.connections['CONN1'].transferred_matter
    vol_flow = _feed_volume_flow()  # [m**3/s]
    # The model evaluated the upstream flow, not the own 1.1e-6 m**3/s.
    np.testing.assert_allclose(sim.R01.result.inlet_vol_flow, vol_flow,
                               rtol=RTOL)
    growth = sim.R01.result.vol[-1] - sim.R01.result.vol[0]  # [m**3]
    assert sim.R01.result.vol[0] == pytest.approx(REACTOR_VOLUME, rel=RTOL)
    assert growth == pytest.approx(vol_flow * RUNTIME, rel=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_resolved_stream_replaces_the_reactor_feed():
    """Replace the transferred feed by identity and continue the reactor.

    A second solve continues the reactor from its final state, as for any
    re-solved ``SemibatchReactor``: its result spans both runs, each of
    which grows the volume by the same feed.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _stream_mixer()
    sim.R01 = configured(SemibatchReactor)
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)
    first = sim.R01.Inlet
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.R01.Inlet is sim.connections['CONN1'].transferred_matter
    assert sim.R01.Inlet is not first
    result = sim.R01.result
    assert (result.time[0], result.time[-1]) == pytest.approx(
        (0., 2 * RUNTIME), abs=RTOL)
    growth = [run['vol'][-1] - run['vol'][0]
              for run in sim.R01.profiles_runs]  # [m**3]
    np.testing.assert_allclose(growth, _feed_volume_flow() * RUNTIME,
                               rtol=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_holdup_charges_an_uncharged_semibatch_reactor():
    """Charge the reactor with a batch holdup; its own feed still flows."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _holdup_mixer()
    sim.R01 = _uncharged_reactor()
    own_inlet = sim.R01.Inlet
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.R01.Inlet is own_inlet
    assert sim.R01.Phases == [sim.connections['CONN1'].transferred_matter]
    assert sim.R01.material_from_upstream
    result = sim.R01.result
    assert result.vol[0] == pytest.approx(_charge_volume(), rel=RTOL)
    own_flow = REACTOR_VOLUME / RESIDENCE_TIME  # [m**3/s]
    assert result.vol[-1] - result.vol[0] == pytest.approx(
        own_flow * RUNTIME, rel=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_resolved_holdup_replaces_the_transferred_charge():
    """Replace the charge of the previous solve instead of rejecting it.

    Provisional: the re-charged reactor keeps its clock, earlier segments
    and heat duty, a known limitation (issue #454,
    https://github.com/PharmaPy-org/PharmaPy/issues/454), so the second
    run is checked from its own segment. The clock assertion pins the
    current behavior; once #454 makes a re-charge start a new run, the
    second run must start at 0 s with a single retained segment.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(THERMO_PATH, {'M01': ['R01'], 'R01': []})
    sim.M01 = _holdup_mixer()
    sim.R01 = _uncharged_reactor()
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.R01.Phases == [sim.connections['CONN1'].transferred_matter]
    # The second run starts from the new charge; the continued clock is
    # the issue #454 limitation.
    last = sim.R01.profiles_runs[-1]
    assert last['vol'][0] == pytest.approx(_charge_volume(), rel=RTOL)
    assert last['time'][0] == pytest.approx(RUNTIME, rel=RTOL)


EVAPORATOR_FEED = 2.0  # [mol/s], connected feed of the evaporator cases
EVAPORATOR_OWN_FEED = 0.5  # [mol/s], pre-set feed, replaced by the transfer
EVAPORATOR_CHARGE = 0.2  # [m**3], liquid charge of the batch Mixer
# [-], [native packed-state units]; resolves the small nitrogen inventory,
# as in tests/test_evaporator_consistency.py
NITROGEN_SOLVER_OPTIONS = {'rtol': 1e-8, 'atol': 1e-10}
# [kg/s], heating water of the ``make_unit`` drum of
# tests/test_evaporator_consistency.py, reused unchanged
EVAPORATOR_UTILITY_FLOW = 1.0


def _evaporator_case(tmp_path, include_nitrogen=False):
    """Build the synthetic binary, an evaporator and its own feed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory for the database.
    include_nitrogen : bool, optional
        Model the nitrogen blanket of the evaporator.

    Returns
    -------
    tuple
        Database path and an ``Evaporator`` with a utility and its own
        ``Inlet`` of EVAPORATOR_OWN_FEED [mol/s], which makes it a
        semibatch unit, but no charge.
    """
    from PharmaPy.Evaporators import Evaporator
    from test_evaporator_consistency import (
        FRACTIONS, PRESSURE, TEMPERATURE as EVAPORATOR_TEMP, UTILITY_TEMP,
        VOLUME, write_uniquac_binary)
    thermo = write_uniquac_binary(tmp_path)
    unit = Evaporator(VOLUME, pressure=PRESSURE,
                      include_nitrogen=include_nitrogen)
    unit.Utility = CoolingWater(mass_flow=EVAPORATOR_UTILITY_FLOW,
                                temp_in=UTILITY_TEMP)
    unit.Inlet = LiquidStream(thermo, temp=EVAPORATOR_TEMP, pres=PRESSURE,
                              mole_flow=EVAPORATOR_OWN_FEED,
                              mole_frac=FRACTIONS)
    return thermo, unit


@pytest.mark.assimulo
@pytest.mark.integration
def test_stream_feeds_a_charged_evaporator(tmp_path):
    """Feed a semibatch evaporator with a connected stream."""
    pytest.importorskip('assimulo')
    from test_evaporator_consistency import (
        FRACTIONS, LIQUID_VOLUME, PRESSURE, TEMPERATURE as EVAPORATOR_TEMP)
    thermo, unit = _evaporator_case(tmp_path)
    unit.Phases = LiquidPhase(thermo, temp=EVAPORATOR_TEMP, pres=PRESSURE,
                              vol=LIQUID_VOLUME, mole_frac=FRACTIONS)
    sim = SimulationExec(thermo, {'M01': ['EV'], 'EV': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = LiquidStream(thermo, temp=EVAPORATOR_TEMP, pres=PRESSURE,
                                  mole_flow=EVAPORATOR_FEED,
                                  mole_frac=FRACTIONS)
    sim.EV = unit
    sim.SolveFlowsheet(kwargs_run={'EV': {'runtime': RUNTIME,
                                          'verbose': False}}, verbose=False)

    assert unit.oper_mode == 'Semibatch'
    assert unit.Inlet is sim.connections['CONN1'].transferred_matter
    feed = unit.get_inputs(RUNTIME)['Inlet']['mole_flow']  # [mol/s]
    assert feed == pytest.approx(EVAPORATOR_FEED, rel=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_resolved_holdup_recharges_an_evaporator(tmp_path):
    """Charge a fed evaporator from a batch Mixer, also on a re-solve.

    With a nitrogen blanket the evaporator replaces its phase objects
    while it solves, so the re-transfer recognizes the transferred charge
    by its marked reference copy, not by identity.
    """
    pytest.importorskip('assimulo')
    from test_evaporator_consistency import (
        FRACTIONS, PRESSURE, TEMPERATURE as EVAPORATOR_TEMP)
    thermo, unit = _evaporator_case(tmp_path, include_nitrogen=True)
    sim = SimulationExec(thermo, {'M01': ['EV'], 'EV': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = LiquidPhase(thermo, temp=EVAPORATOR_TEMP, pres=PRESSURE,
                                 vol=EVAPORATOR_CHARGE, mole_frac=FRACTIONS)
    sim.EV = unit
    run = {'EV': {'runtime': RUNTIME, 'verbose': False,
                  'sundials_opts': dict(NITROGEN_SOLVER_OPTIONS)}}
    sim.SolveFlowsheet(kwargs_run=run, verbose=False)
    first_moles = unit.result.mol_liq[0]  # [mol]
    assert unit.Liquid_1 is not sim.connections['CONN1'].transferred_matter

    sim.SolveFlowsheet(kwargs_run=run, verbose=False)

    charge = unit.__original_phase__
    assert charge.transferred_from_uo
    assert charge.moles == pytest.approx(sim.M01.Outlet.moles, rel=RTOL)
    # A new charge restarts the evaporator, so both solves are equal.
    assert unit.result.mol_liq[0] == pytest.approx(first_moles, rel=RTOL)
    assert unit.result.time[-1] == pytest.approx(RUNTIME, rel=RTOL)
