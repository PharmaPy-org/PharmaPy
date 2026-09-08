"""#225 heat-duty retrieval through real RHS, phases, and optional CVode.

Synthetic fixed profiles isolate jacket heat, prescribed storage, and feed
enthalpy. Solver tests exercise uncontrolled Batch retrieval end to end.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Utilities import CoolingWater

TEMPERATURE = 310.0  # [K], above enthalpy reference
LIQUID_VOLUME = 9e-4  # [m**3], synthetic liquid inventory
JACKET_TEMPERATURE = 300.0  # [K], establishes nonzero cooling
DURATION = 5.0  # [s], two-point interval detects lost integration interval
RTOL = 1e-12  # [-], direct energy accounting roundoff allowance


def make_unit(data_path, unit_type, adiabatic=False, ramp=None, feed_flow=0.0):
    """Create a small FVM vessel with no crystallization heat source.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    unit_type : type
        BatchCryst or MSMPR.
    adiabatic : bool, optional
        Exclude utility transfer.
    ramp : float or None, optional
        Prescribed temperature slope [K/s], or None for a dynamic temperature.
    feed_flow : float, optional
        MSMPR slurry-feed rate [m**3/s].

    Returns
    -------
    unit : BatchCryst or MSMPR
        Real unit with inactive kinetics and a real water utility.
    states : ndarray
        Initial state: total or volume-specific CSD [#/um] or [#/m**3/um],
        concentrations [kg/m**3], optional liquid volume [m**3], temperatures [K].
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    controls = None if ramp is None else {
        'temp': lambda time: TEMPERATURE + ramp * np.asarray(time)}
    unit = unit_type('A', adiabatic=adiabatic, controls=controls)
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=LIQUID_VOLUME,
                         mass_frac=[0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
    grid = np.array([10.0, 20.0, 40.0])  # [um], unequal FVM bins
    distribution = np.array([1e7, 2e7, 1e7])  # [#/um], asymmetric seeded population
    solid = SolidPhase(path, temp=TEMPERATURE, x_distrib=grid,
                       distrib=distribution, kv=0.5,
                       mass_frac=[1, 0, 0, 0, 0])  # [-], pure A, non-unit kv
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[2000.0])  # [kg/m**3], inactive rates
    unit.Kinetics.target_idx = unit.target_ind
    unit.Utility = CoolingWater(vol_flow=1e-5, temp_in=290.0)  # [m**3/s], [K]
    unit.num_species = len(liquid.mass_conc)
    unit.x_grid = grid  # [um]
    unit.dx = unit.Slurry.dx  # [um]
    unit.vol_tank = unit.Slurry.vol  # [m**3]
    unit.diam_tank = 0.1  # [m], synthetic vessel diameter
    unit.area_base = np.pi * unit.diam_tank**2 / 4  # [m**2]
    if unit_type is MSMPR:
        inlet = SlurryStream(vol_flow=feed_flow, x_distrib=grid,
                             distrib=unit.Slurry.distrib)
        inlet.Phases = (
            LiquidStream(path, temp=TEMPERATURE, mass_frac=liquid.mass_frac),
            SolidStream(path, temp=TEMPERATURE, mass_frac=solid.mass_frac,
                        x_distrib=grid, distrib=unit.Slurry.distrib, kv=solid.kv))
        unit.Inlet = inlet
        solid_states = unit.Slurry.distrib  # [#/m**3/um]
    else:
        solid_states = solid.distrib  # [#/um]
    parts = [solid_states, liquid.mass_conc]
    if unit_type is BatchCryst:
        parts.append([liquid.vol])
    if ramp is None:
        parts.append([TEMPERATURE])
        if not adiabatic:
            parts.append([JACKET_TEMPERATURE])
    return unit, np.concatenate(parts)


@pytest.mark.unit
@pytest.mark.parametrize('adiabatic', [False, True])
def test_batch_uncontrolled_retrieval_integrates_only_utility_rate(data_path, adiabatic):
    unit, initial = make_unit(data_path, BatchCryst, adiabatic=adiabatic)
    time = np.array([0.0, DURATION])  # [s]
    states = np.tile(initial, (2, 1))  # [state units], constant test profile
    area = 4 * unit.Slurry.vol / unit.diam_tank + unit.area_base  # [m**2]
    expected = (0.0 if adiabatic else unit.u_ht * area
                * (TEMPERATURE - JACKET_TEMPERATURE) * DURATION)  # [J]
    unit.retrieve_results(time, states)
    np.testing.assert_allclose(unit.heat_duty, [0, expected], rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('unit_type,feed_flow', [(BatchCryst, 0.0),
                                               (MSMPR, 1e-5)])
# [m**3/s], batch, closed MSMPR, and finite liquid feed
@pytest.mark.parametrize('ramp', [0.0, 1.0])  # [K/s], isothermal and linear heating
def test_prescribed_temperature_duty_closes_energy_balance(
        data_path, unit_type, feed_flow, ramp):
    unit, initial = make_unit(data_path, unit_type, ramp=ramp, feed_flow=feed_flow)
    time = np.array([0.0, DURATION])  # [s]
    states = np.tile(initial, (2, 1))  # [state units], composition held fixed
    temperatures = TEMPERATURE + ramp * time  # [K]
    capacities = np.array([sum(phase.mass * phase.getCp(temp, basis='mass')
                               for phase in unit.Phases)
                           for temp in temperatures])  # [J/K], real phase sum
    flows = np.zeros(len(time))  # [W], batch has no advective heat term
    if unit_type is MSMPR:
        inlet = unit.Inlet
        inlet_enthalpy = sum(phase.mass * phase.getEnthalpy(TEMPERATURE, basis='mass')
                             for phase in unit.Phases) / unit.Slurry.vol  # [J/m**3]

        for index, temp in enumerate(temperatures):
            tank_enthalpy = sum(phase.mass * phase.getEnthalpy(temp, basis='mass')
                                for phase in unit.Phases) / unit.Slurry.vol  # [J/m**3]
            flows[index] = feed_flow * (inlet_enthalpy - tank_enthalpy)  # [W]
    # Positive duty removes heat. For C*dT/dt = flow - source - Q,
    # no crystal source gives Q*dt = mean(flow)*dt - mean(C)*dT.
    expected = (flows.mean() * DURATION
                - capacities.mean() * (temperatures[-1] - temperatures[0]))  # [J]
    unit.retrieve_results(time, states)
    np.testing.assert_allclose(unit.heat_duty, [0, expected], rtol=RTOL, atol=0)


@pytest.mark.assimulo
@pytest.mark.parametrize('adiabatic', [False, True])
def test_batch_solver_retrieves_finite_duty(data_path, adiabatic):
    pytest.importorskip('assimulo')
    unit, _ = make_unit(data_path, BatchCryst, adiabatic=adiabatic)
    time, states = unit.solve_unit(
        time_grid=np.linspace(0.0, DURATION, 5), verbose=False)  # [s], reporting grid
    assert len(time) == 5
    assert np.isfinite(states).all()
    assert unit.heat_duty.shape == (2,)
    assert np.isfinite(unit.heat_duty).all()
    if adiabatic:
        np.testing.assert_array_equal(unit.heat_duty, [0.0, 0.0])
    else:
        assert unit.heat_duty[1] > 0



@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR])
def test_isothermal_duty_removes_crystallization_heat(data_path, unit_type):
    feed_flow = 1e-5  # [m**3/s], matching slurry feed has zero net sensible heat
    unit, initial = make_unit(data_path, unit_type, ramp=0.0, feed_flow=feed_flow)
    saturation = 1.0  # [kg/m**3], synthetic solubility enabling crystal growth
    unit.Kinetics = CrystKinetics(coeff_solub=[saturation],
                                  growth=(1.0, 0.0, 1.0))  # [um/s], [J/mol], [-]
    unit.Kinetics.target_idx = unit.target_ind
    growth = (unit.Liquid_1.mass_conc[0] / saturation - 1)  # [um/s], unit prefactor
    grid = unit.Solid_1.x_distrib  # [um]
    weighted = unit.Solid_1.distrib * grid**2  # [#/um]*[um**2]
    total_area_moment = np.sum(np.diff(grid) * (weighted[1:] + weighted[:-1]) / 2)
    # [um**2], independent trapezoids of the total number distribution
    mass_rate = (3 * unit.Solid_1.kv * unit.Solid_1.getDensity()
                 * growth * total_area_moment * 1e-18)  # [kg/s], cubic um -> m
    latent_heat = 1.46e4  # [J/kg], magnitude of the existing crystallizer latent model
    expected = mass_rate * latent_heat * DURATION  # [J], heat removed at constant T
    time = np.array([0.0, DURATION])  # [s]
    unit.retrieve_results(time, np.tile(initial, (2, 1)))
    assert unit.heat_duty[1] == pytest.approx(expected, rel=RTOL, abs=0)
