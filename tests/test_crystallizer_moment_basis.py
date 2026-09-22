"""#224/#227 micrometre state seeding and finite-size nucleation balances.

Core probes prepare states without a solver; the marked test runs real CVode.
The seed grid fixture has non-unit kv and two cubic metres of liquid.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR
from test_crystallizer_heat_duty import make_unit as make_heat_unit

from test_crystallizer_parameter_evaluations import (
    RTOL, TEMPERATURE, make_unit,
)


@pytest.mark.unit
def test_seeded_moments_initial_rhs(data_path):
    """Compare initialized moments and their growth RHS to grid integrals.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_unit(data_path, 'moments')
    # Trapezoids of the fixture's x**n*f, with x in um, give these totals.
    expected = np.array([4.5e8, 1.05e10, 2.85e11, 8.85e12])  # [um**n]
    initial_state, _ = unit.initialize_states(runtime=1.0)
    parameters = unit.Kinetics.concat_params()[unit.mask_params]  # native kinetic units
    rhs = unit.unit_model(0.0, initial_state, parameters)  # [state units/s]
    np.testing.assert_allclose(initial_state[:4], expected, rtol=RTOL, atol=0)
    force = unit.Liquid_1.mass_conc[0] / 2.0 - 1  # [-], fixture solubility=2 kg/m**3
    growth_rhs = np.array([0, 4.5e8, 2.1e10, 8.55e11]) * force  # [um**n/s]
    np.testing.assert_allclose(rhs[:4], growth_rhs, rtol=RTOL, atol=0)
    assert unit.states_di['mu_n']['units'] == 'm**n'


@pytest.mark.unit
def test_msmpr_seeds_volume_specific_micrometre_moments(data_path):
    """Convert SI slurry moments without changing their volume denominator.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    batch, _ = make_unit(data_path, 'moments')
    unit = MSMPR('A', method='moments',
                 controls={'temp': lambda time: TEMPERATURE})
    unit.Phases = batch.Phases
    unit.Kinetics = batch.Kinetics
    expected = np.array([4.5e8, 1.05e10, 2.85e11, 8.85e12]) / unit.Slurry.vol
    # [um**n/m**3], independently integrated total seed divided by slurry volume

    initial_state, _ = unit.initialize_states(runtime=1.0)
    np.testing.assert_allclose(initial_state[:4], expected, rtol=RTOL, atol=0)
    assert unit.states_di['mu_n']['units'] == 'm**n/m**3'


@pytest.mark.unit
def test_finite_radius_nucleation_total_moments_and_mass(data_path):
    """Count nuclei over the whole slurry, including their physical mass.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    radius = 20.0  # [um], finite nuclei distinguish every moment order
    volume = 2.5  # [m**3], non-unit slurry volume exposes missing factors
    rate = 4e8  # [#/m**3/s], synthetic constant primary nucleation
    unit, states = make_unit(data_path, 'moments', radius=radius)
    unit.Kinetics.params['nucl_prim'] = [rate, 0.0, 0.0]
    # [#/m**3/s], [J/mol], [-], constant rate for supersaturated liquid
    unit.Kinetics.params['growth'] = [0.0, 0.0, 0.0]  # [um/s], [J/mol], [-]
    rhs, mass = unit.method_of_moments(
        states[:4], states[4:-1], TEMPERATURE, None,
        unit.Solid_1.getDensity(), vol=volume)
    # A billion new particles per second, each of length 20 um.
    expected = np.array([1e9, 2e10, 4e11, 8e12])  # [um**n/s]
    np.testing.assert_allclose(rhs, expected, rtol=RTOL, atol=0)
    particle_volume = unit.Solid_1.kv * (20e-6)**3  # [m**3/particle]
    expected_mass = 1e9 * particle_volume * unit.Solid_1.getDensity()  # [kg/s]
    assert mass[0] == pytest.approx(expected_mass, rel=RTOL, abs=0)
    states[:4] = 0  # [um**n], zero seed makes slurry volume equal liquid volume
    states[-1] = volume  # [m**3]
    public_rhs = unit.unit_model(0.0, states)  # [state unit/s]
    np.testing.assert_allclose(public_rhs[:4], expected, rtol=RTOL, atol=0)
    assert -public_rhs[-1] * unit.Liquid_1.getDensity() == pytest.approx(
        expected_mass, rel=RTOL, abs=0)


@pytest.mark.assimulo
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR])
def test_seeded_moment_solver_reports_initial_si_moments(data_path, unit_type):
    """Preserve SI phase/result moments while integrating micrometre states.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    unit_type : type
        Batch or continuous crystallizer class.
    """
    pytest.importorskip('assimulo')
    unit, _ = make_unit(data_path, 'moments')
    initial_si = unit.Solid_1.moments.copy()  # [m**n]
    if unit_type is MSMPR:
        batch = unit
        feed_flow = 1e-5  # [m**3/s], finite flow avoids the separate zero-flow defect
        feed, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=feed_flow)
        unit = MSMPR('A', method='moments', controls={
            'temp': lambda time: TEMPERATURE + np.zeros_like(time)})
        unit.Phases = batch.Phases
        unit.Kinetics = batch.Kinetics
        unit.Inlet = feed.Inlet
        initial_si = initial_si / unit.Slurry.vol  # [m**n/m**3], slurry basis
    duration = 1e-5  # [s], short interval resolves the initial growth tangent
    _, states = unit.solve_unit(time_grid=[0.0, duration], verbose=False)
    expected_um = np.array([initial_si[0], initial_si[1] * 1e6,
                            initial_si[2] * 1e12, initial_si[3] * 1e18])
    # [um**n] or [um**n/m**3], exact conversion of each SI moment order
    np.testing.assert_allclose(states[0, :4], expected_um, rtol=RTOL, atol=0)
    np.testing.assert_allclose(unit.result.mu_n[0], initial_si, rtol=RTOL, atol=0)
    denominator = '/m**3' if unit_type is MSMPR else ''
    assert unit.states_di['mu_n']['units'] == 'm**n' + denominator
    assert unit.result.di_states['mu_n']['units'] == 'm**n' + denominator
