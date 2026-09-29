"""Jacketed moment balances through real Batch, Semibatch and MSMPR solves.

Frozen populations isolate thermal evolution from crystallization heat. The
shared inventory fixture supplies non-unit crystal shape and real feed phases.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, SemibatchCryst, MSMPR
from PharmaPy.Utilities import CoolingWater
from test_crystallizer_moment_inventory import inventory_unit
from test_crystallizer_parameter_evaluations import TEMPERATURE, RTOL


@pytest.mark.assimulo
@pytest.mark.parametrize('unit_type', [BatchCryst, SemibatchCryst, MSMPR])
def test_jacketed_moment_solve_preserves_thermal_state_order(data_path, unit_type):
    """Cool a moment vessel while retaining separate tank and jacket states.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Batch, Semibatch or MSMPR crystallizer class.
    """
    pytest.importorskip('assimulo')
    unit, _ = inventory_unit(data_path, unit_type, gridless=True)
    # Reconstruct without the fixture's prescribed-temperature control so both
    # thermal balances are active and initialized by solve_unit itself.
    jacketed = unit_type('A', method='moments')
    jacketed.Phases = unit.Phases
    jacketed.Kinetics = unit.Kinetics
    if unit_type is not BatchCryst:
        jacketed.Inlet = unit.Inlet
    coolant_temperature = TEMPERATURE - 10.0  # [K], imposed 10 K cooling offset
    jacketed.Utility = CoolingWater(temp_in=coolant_temperature, mass_flow=1.0)
    # [kg/s], finite water circulation for the synthetic vessel
    duration = 0.1  # [s], short interval resolves initial cooling direction
    time, states = jacketed.solve_unit(time_grid=[0.0, duration], verbose=False)
    assert states.shape == (len(time), sum(jacketed.dim_states))
    assert np.isfinite(states).all()
    assert jacketed.name_states[-2:] == ['temp', 'temp_ht']
    assert jacketed.result.temp[0] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    # solve_unit initializes jacket holdup at the vessel temperature [K];
    # the colder flowing utility subsequently cools both thermal states.
    assert jacketed.result.temp_ht[0] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    assert coolant_temperature < jacketed.result.temp[-1] < TEMPERATURE
    assert coolant_temperature < jacketed.result.temp_ht[-1] < TEMPERATURE
    # Issue #225 covers Batch/MSMPR duty reporting. Semibatch has no duty
    # diagnostic; its assertions above verify the thermal state contract.
    if unit_type is not SemibatchCryst:
        assert jacketed.heat_duty[1] > 0  # [J], heat removed to the colder jacket
