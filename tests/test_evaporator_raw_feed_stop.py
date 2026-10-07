"""Raw-material accounting of an Evaporator inlet shut by its volume event.

With ``stop_at_maxvol=False`` the semibatch ``Evaporator`` shuts its inlet
when the liquid reaches 95% of the drum volume and keeps evaporating.
``GetRawMaterials`` must stop the static and the ``DynamicInlet`` feed at
that time, including over a continuation with the inlet shut. The real IDA
model uses the synthetic binary of ``tests/conftest.py`` with the UNIQUAC
interaction of ``tests/test_evaporator_consistency.py``; the event time is
derived independently from the solved liquid-volume trajectory.

Related issue scope: https://github.com/PharmaPy-org/PharmaPy/issues/260
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from conftest import THERMO_TWO_SPECIES
from PharmaPy.Evaporators import Evaporator
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater


pytestmark = [pytest.mark.assimulo, pytest.mark.integration]

PRESSURE = 1000.0  # [Pa], keeps both synthetic species subcritical at boiling
TEMPERATURE = 350.0  # [K], initial liquid, feed and UNIQUAC reference temperature
FRACTIONS = [0.4, 0.6]  # [-], asymmetric binary
GAS_CONSTANT = 8.314  # [J/mol/K], value used by PharmaPy.Evaporators
DRUM_VOLUME = 1.0  # [m**3]
FILL_FRACTION = 0.95  # [-], Evaporator's liquid-volume event threshold
FILL_MARGIN = 0.001  # [-], initial gap so the event occurs early in the first segment
FEED_FLOW = 20.0  # [mol/s], fills the gap before evaporation dominates
TEMP_RAMP = 0.5  # [K/s], temperature-only DynamicInput feed law
SEGMENT = 20.0  # [s], each solve; the feed stops well within the first
UTILITY_TEMP = 450.0  # [K], heating utility above both bubble points
SOLVER_OPTIONS = {'rtol': 1e-8, 'atol': 1e-10}  # [-], [native state units]
# [-], relative root-surface allowance, as Evaporator's sqrt(eps) volume check
VOLUME_RTOL = np.sqrt(np.finfo(float).eps)
SUM_RTOL = 1e-10  # [-], roundoff of trapezoidal sums of constant flows


@pytest.fixture
def thermo_path(tmp_path):
    """Write the synthetic binary with equal UNIQUAC sizes and tau=1/2.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test directory.

    Returns
    -------
    str
        Thermodynamic JSON path; unlike interaction [J/mol] is R*T*ln(2).
    """
    data = deepcopy(THERMO_TWO_SPECIES)
    for species in data.values():
        species.update(ri=1.0, qi=1.0, qip=1.0)  # [-], equal molecular sizes
    interaction = GAS_CONSTANT * TEMPERATURE * np.log(2)  # [J/mol], tau=1/2 at T
    data['interaction'] = {'amk': [[0, interaction], [interaction, 0]]}
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(data))
    return str(path)


def feed_temperature(time):
    """Return the controlled feed temperature.

    Parameters
    ----------
    time : float or numpy.ndarray
        Absolute time [s].

    Returns
    -------
    float or numpy.ndarray
        Feed temperature [K], rising linearly from TEMPERATURE.
    """
    return TEMPERATURE + TEMP_RAMP * np.asarray(time)


def first_event_time(unit):
    """Locate the first liquid-volume event from the solved trajectory.

    Parameters
    ----------
    unit : Evaporator
        Solved evaporator whose result holds liquid holdup [mol], mole
        fractions [-] and temperature [K].

    Returns
    -------
    float
        First result time [s] at which the liquid volume [m**3] reaches the
        fill threshold within VOLUME_RTOL.
    """
    result = unit.result
    density = np.array([
        unit.Liquid_1.getDensity(mole_frac=fractions, temp=temp, basis='mole')
        for fractions, temp in zip(result.x_liq, result.temp)])  # [mol/L]
    liters_per_cubic_meter = 1000.0  # [L/m**3], exact
    volume = (np.ravel(result.mol_liq) / density
              / liters_per_cubic_meter)  # [m**3]
    cap = FILL_FRACTION * DRUM_VOLUME  # [m**3]
    reached = np.flatnonzero(volume >= cap * (1 - VOLUME_RTOL))
    assert reached.size > 0
    return float(result.time[reached[0]])


def inlet_record(simulation):
    """Return the raw-feed record of the evaporator inlet.

    Parameters
    ----------
    simulation : SimulationExec
        Solved one-unit flowsheet named ``EV``.

    Returns
    -------
    pandas.Series
        Molar raw record: ``moles`` [mol], ``temp`` [K] and other fields.
    """
    table = simulation.GetRawMaterials(basis='mole', totals=False)
    assert table.index[0][:2] == ('EV', 'Inlet_0')
    return table.iloc[0]


@pytest.mark.parametrize('dynamic', [False, True], ids=['static', 'dynamic'])
def test_raw_feed_stops_when_volume_event_shuts_inlet(thermo_path, dynamic):
    pytest.importorskip('assimulo')
    unit = Evaporator(DRUM_VOLUME, pressure=PRESSURE, stop_at_maxvol=False)
    unit.Phases = LiquidPhase(
        thermo_path, temp=TEMPERATURE, pres=PRESSURE,
        vol=(FILL_FRACTION - FILL_MARGIN) * DRUM_VOLUME, mole_frac=FRACTIONS)
    unit.Utility = CoolingWater(mass_flow=1.0, temp_in=UTILITY_TEMP)  # [kg/s], [K]
    inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_flow=FEED_FLOW, mole_frac=FRACTIONS)
    if dynamic:
        control = DynamicInput()
        control.add_variable('temp', feed_temperature)
        inlet.DynamicInlet = control
    unit.Inlet = inlet
    simulation = SimulationExec(thermo_path, {'EV': []})
    simulation.EV = unit
    run = {'EV': {'runtime': SEGMENT, 'verbose': False,
                  'sundials_opts': dict(SOLVER_OPTIONS)}}

    simulation.SolveFlowsheet(kwargs_run=run, verbose=False)
    event_time = first_event_time(unit)  # [s]
    first = inlet_record(simulation)

    assert not unit.allow_flow
    assert 0 < event_time < SEGMENT
    assert unit.inlet_stop_time == pytest.approx(event_time, rel=SUM_RTOL)
    # Constant molar feed from t = 0 until the volume event.
    assert first['moles'] == pytest.approx(FEED_FLOW * event_time,
                                           rel=SUM_RTOL)
    if dynamic:
        # Flow-weighted mean of the linear ramp over the fed interval only.
        assert first['temp'] == pytest.approx(
            feed_temperature(event_time / 2), rel=SUM_RTOL)

    # Continue with the inlet shut: no further raw feed is consumed.
    simulation.SolveFlowsheet(kwargs_run=run, verbose=False)
    second = inlet_record(simulation)

    assert unit.result.time[-1] == pytest.approx(2 * SEGMENT, rel=SUM_RTOL)
    assert second['moles'] == pytest.approx(first['moles'], rel=SUM_RTOL)

    unit.reset()
    assert unit.inlet_stop_time is None
    assert unit.allow_flow
