"""Liquid DynamicCollector holdups close their inventory balances (#467).

A liquid ``DynamicCollector`` accumulates its feed without outflow,
reaction or heat exchange, so its holdup after any time t is exactly the
seed plus the integrated feed: species masses m0 w0 + int F w_in dt [kg]
and enthalpy m0 h0 + int F h_in dt [J]. The collector used to integrate
mass fractions, dw/dt = F/m (w_in - w), from its 0.1 s seed with the
default CVode tolerances, whose error was not controlled: the collected
composition drifted from the inventory by up to a third for trace species,
and could leave the range of the feed compositions. It now integrates the
inventories, scaled by the seed.

Fixtures:

* A raw feed with ``DynamicInput`` controls on the shipped five-species
  database (tests/Flowsheet/data/compound_database.json): a trace-species
  composition with an early Gaussian pulse of A and C, a mass flow rising
  from 2.5 to 7.5 g/s, and an oscillating temperature, collected for 2 h.
  The profiles are synthetic, chosen only to vary on the collector's early
  and late time scales.
* The route of the PFR/batch tutorial
  (doc/online_docs/examples/PFR_Batch_solved.ipynb), its plug-flow reactor
  R01 feeding the collector HOLD01 for 2 h, on its case data
  (workshop_data/Case_study_3).

Expected inventories are integrated independently with the trapezoidal
rule on a 0.1 s grid of the collector's own feed values (the control laws,
or ``get_inputs_new`` for the connected reactor outlet). Enthalpies use
the liquid property model on the same reference as the collector. The
collector is integrated with CVode, so every case is marked ``assimulo``.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from scipy.integrate import cumulative_trapezoid

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Reactors import PlugFlowReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater

RUNTIME = 7200.0  # [s], the tutorial's collection time
GRID_POINTS = 72001  # [-], 0.1 s trapezoidal grid over RUNTIME
SEED_TIME = 0.1  # [s], the collector's provisional seed of feed (#342)
# Synthetic controlled feed (see module docstring).
BASE_FRACTIONS = np.array([0.002, 0.005, 0.015, 0.018, 0.96])  # [-]
PULSE_FRACTIONS = np.array([0.02, 0., 0.03, 0., 0.])  # [-], peak addition
PULSE_TIME = 600.0  # [s], pulse centre
PULSE_WIDTH = 300.0  # [s], pulse width
BASE_FLOW = 5e-3  # [kg/s]
FLOW_SWING = 0.5  # [-], relative flow change across the ramp
RAMP_TIME = 1800.0  # [s], ramp centre
RAMP_WIDTH = 600.0  # [s], ramp width
BASE_TEMP = 313.15  # [K]
TEMP_SWING = 10.0  # [K]
TEMP_PERIOD = 1200.0  # [s], angular period 1/omega of the oscillation
# CVode's default absolute tolerance [-] bounds each scaled species mass,
# M_i / m0, so a composition error above it, relative to the seed, is a
# solver failure; mass and enthalpy accumulate the default relative
# tolerance 1e-6 over the run, allowed tenfold.
COMPOSITION_ATOL = 1e-6  # [-]
INVENTORY_RTOL = 1e-5  # [-]
# Tutorial route (doc/online_docs/examples/PFR_Batch_solved.ipynb).
CASE_DATA = (Path(__file__).resolve().parents[1] / 'workshop_data'
             / 'Case_study_3' / 'compound_database.json')
REACTOR_VOLUME = 0.010  # [m**3]
RESIDENCE_TIME = 1800.0  # [s]
REACTOR_TEMP = 313.15  # [K]
FEED_CONC = np.array([0.33, 0.33, 0, 0, 0])  # [mol/L]
PRE_EXPONENTIAL = np.array([2.654e4, 5.3e2])  # [L/mol/s], bimolecular
ACTIVATION = np.array([4.0e4, 3.0e4])  # [J/mol]
COOLING_FLOW = 0.1  # [kg/s]
TUBE_DIAMETER = 0.5 * 0.0254  # [m], half inch
REACTOR_NODES = 50  # [-]


def _fractions(time):
    """Evaluate the controlled feed composition.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    numpy.ndarray
        Mass fractions [-], shape (5,) or (num_times, 5).
    """
    pulse = np.exp(-((np.asarray(time, dtype=float) - PULSE_TIME)
                     / PULSE_WIDTH) ** 2)  # [-]
    weights = BASE_FRACTIONS + np.multiply.outer(pulse, PULSE_FRACTIONS)
    return weights / weights.sum(axis=-1, keepdims=True)


def _flow(time):
    """Evaluate the controlled feed mass flow [kg/s].

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].

    Returns
    -------
    float or numpy.ndarray
        Mass flow [kg/s] with the shape of ``time``.
    """
    return BASE_FLOW * (1 + FLOW_SWING * np.tanh(
        (np.asarray(time, dtype=float) - RAMP_TIME) / RAMP_WIDTH))


def _temperature(time):
    """Evaluate the controlled feed temperature [K].

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].

    Returns
    -------
    float or numpy.ndarray
        Temperature [K] with the shape of ``time``.
    """
    return BASE_TEMP + TEMP_SWING * np.sin(
        np.asarray(time, dtype=float) / TEMP_PERIOD)


def _exact_inventory(grid, flow, fractions):
    """Integrate the seed plus the feed on a grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Times [s], shape (num_times,), from the start of collection.
    flow : numpy.ndarray
        Feed mass flow [kg/s], shape (num_times,).
    fractions : numpy.ndarray
        Feed mass fractions [-], shape (num_times, num_species).

    Returns
    -------
    numpy.ndarray
        Species masses [kg], shape (num_times, num_species): the 0.1 s
        seed of the first feed plus the cumulative trapezoidal feed.
    """
    seed = SEED_TIME * flow[0] * fractions[0]  # [kg]
    return seed + cumulative_trapezoid(flow[:, np.newaxis] * fractions,
                                       grid, axis=0, initial=0)  # [kg]


def test_profiled_feed_closes_the_inventory(data_path):
    """Match the integrated species and enthalpy inventories at all times."""
    path = str(data_path['flowsheet'] / 'compound_database.json')
    inlet = LiquidStream(path, temp=_temperature(0.), mass_flow=_flow(0.),
                         mass_frac=_fractions(0.))
    controls = DynamicInput()
    controls.add_variable('mass_frac', _fractions)
    controls.add_variable('mass_flow', _flow)
    controls.add_variable('temp', _temperature)
    inlet.DynamicInlet = controls
    collector = DynamicCollector()
    collector.Inlet = inlet
    time, states = collector.solve_unit(runtime=RUNTIME, verbose=False)

    grid = np.linspace(0., RUNTIME, GRID_POINTS)  # [s]
    flow, fractions = _flow(grid), _fractions(grid)  # [kg/s], [-]
    species = _exact_inventory(grid, flow, fractions)  # [kg]
    mass = species.sum(axis=1)  # [kg]
    expected = np.column_stack([np.interp(time, grid, column) for column
                                in (species / mass[:, np.newaxis]).T])  # [-]
    result = collector.result
    np.testing.assert_allclose(result.mass_frac, expected, rtol=0,
                               atol=COMPOSITION_ATOL)
    np.testing.assert_allclose(result.mass, np.interp(time, grid, mass),
                               rtol=INVENTORY_RTOL)
    # A mixture of the feeds stays within their composition range.
    assert np.all(result.mass_frac >= fractions.min(axis=0)
                  - COMPOSITION_ATOL)
    assert np.all(result.mass_frac <= fractions.max(axis=0)
                  + COMPOSITION_ATOL)

    # Adiabatic accumulation: m h(T, w) = m0 h0 + int F h_in dt [J].
    liquid = LiquidPhase(path, temp=BASE_TEMP, mass=1.0,
                         mass_frac=BASE_FRACTIONS)
    h_in = liquid.getEnthalpy(temp=_temperature(grid),
                              mass_frac=fractions)  # [J/kg]
    enthalpy = (SEED_TIME * flow[0] * h_in[0]
                + cumulative_trapezoid(flow * h_in, grid, initial=0))  # [J]
    held = result.mass * liquid.getEnthalpy(temp=result.temp,
                                            mass_frac=result.mass_frac)  # [J]
    np.testing.assert_allclose(held, np.interp(time, grid, enthalpy),
                               rtol=INVENTORY_RTOL)
    # The returned states are the published layout: fractions, mass, temp.
    np.testing.assert_array_equal(states[:, :5], result.mass_frac)
    np.testing.assert_array_equal(states[:, 5], result.mass)
    np.testing.assert_array_equal(states[:, 6], result.temp)


def test_tutorial_route_holdup_matches_the_closed_form():
    """Collect the tutorial's reactor outlet with its inventory composition."""
    path = str(CASE_DATA)
    sim = SimulationExec(path, 'R01 --> HOLD01')
    kinetics = RxnKinetics(path=path, rxn_list=['A + B --> C', 'A + C --> D'],
                           k_params=PRE_EXPONENTIAL, ea_params=ACTIVATION)
    sim.R01 = PlugFlowReactor(diam_in=TUBE_DIAMETER, num_discr=REACTOR_NODES,
                              isothermal=False)
    sim.R01.Utility = CoolingWater(mass_flow=COOLING_FLOW,
                                   temp_in=REACTOR_TEMP)
    sim.R01.Kinetics = kinetics
    sim.R01.Inlet = LiquidStream(path, REACTOR_TEMP, mole_conc=FEED_CONC,
                                 vol_flow=REACTOR_VOLUME / RESIDENCE_TIME,
                                 name_solv='solvent')
    sim.R01.Phases = (LiquidPhase(path, REACTOR_TEMP,
                                  mass_frac=np.array([0, 0, 0, 0, 1]),
                                  vol=REACTOR_VOLUME),)
    sim.HOLD01 = DynamicCollector()
    run = {name: {'runtime': RUNTIME, 'verbose': False}
           for name in ('R01', 'HOLD01')}
    sim.SolveFlowsheet(kwargs_run=run, verbose=False)

    grid = np.linspace(0., RUNTIME, GRID_POINTS)  # [s]
    inputs = sim.HOLD01.get_inputs_new(grid)['Inlet']
    flow = np.broadcast_to(np.atleast_1d(inputs['mass_flow']),
                           grid.shape)  # [kg/s]
    fractions = np.atleast_2d(inputs['mass_frac'])  # [-]
    species = _exact_inventory(grid, flow, fractions)[-1]  # [kg]
    collected = sim.HOLD01.Liquid_1.mass_frac  # [-]
    np.testing.assert_allclose(collected, species / species.sum(), rtol=0,
                               atol=COMPOSITION_ATOL)
    assert np.all(collected >= fractions.min(axis=0) - COMPOSITION_ATOL)
    assert np.all(collected <= fractions.max(axis=0) + COMPOSITION_ATOL)
