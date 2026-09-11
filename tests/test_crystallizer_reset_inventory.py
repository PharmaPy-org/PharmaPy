"""#222 reset restores phase inventories and cached slurry population together.

Real result retrieval creates changed populations; optional CVode coverage checks
repeatability through complete solves. Core probes stop at solver construction.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from test_crystallizer_heat_duty import make_unit as make_heat_unit
from test_crystallizer_moment_inventory import inventory_unit, FLOW
from test_crystallizer_parameter_evaluations import InitializationCaptured, RTOL


@pytest.mark.unit
def test_reset_charge_arrays_do_not_alias_live_phases(data_path):
    """Restore the charge after in-place mutation of an already-reset phase.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = inventory_unit(data_path, BatchCryst, gridless=True)
    original = unit.Liquid_1.mass_conc.copy()  # [kg/m**3], immutable charge oracle
    unit.reset()
    unit.Liquid_1.mass_conc *= 2  # [-], expose aliasing with the stored charge
    unit.reset()
    np.testing.assert_allclose(unit.Liquid_1.mass_conc, original, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('case', ['msmpr_moments', 'msmpr_fvm', 'semibatch_moments'])
def test_reset_restores_cached_slurry_at_solver_boundary(data_path, monkeypatch, case):
    """Restore a changed population or volume before capturing the next state.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace the optional solver-construction boundary.
    case : str
        Population representation and crystallizer operating mode.
    """
    if case == 'msmpr_fvm':
        unit, initial = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FLOW)
    else:
        unit_type = MSMPR if case == 'msmpr_moments' else SemibatchCryst
        unit, initial = inventory_unit(data_path, unit_type, gridless=True)
    original_phases = tuple(unit.Phases)
    original_volume = unit.Slurry.vol  # [m**3], charged slurry inventory
    original_moments = unit.Slurry.moments.copy()  # [m**n/m**3]
    profiles = np.tile(initial, (2, 1))  # [raw state units], initial/final rows
    profiles[-1, :unit.num_distr] *= 2  # [-], double the final crystal population
    if case == 'semibatch_moments':
        profiles[-1, -1] *= 2  # [m**3], double final liquid volume with filling
    unit.retrieve_results(np.array([0.0, 0.01]), profiles)  # [s], reporting interval
    unit.reset_states = True

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Check reset inventories at the actual ODE initialization boundary.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver options.
        states_init : ndarray
            Population, composition, and optional volume in raw solver units.
        params_mergd : ndarray
            Active kinetic parameters in their native units.

        Raises
        ------
        InitializationCaptured
            After checking state and slurry consistency.
        """
        np.testing.assert_allclose(states_init, initial, rtol=RTOL, atol=0)
        assert unit.Slurry.vol == pytest.approx(original_volume, rel=RTOL, abs=0)
        np.testing.assert_allclose(unit.Slurry.moments, original_moments,
                                   rtol=RTOL, atol=0)
        assert all(current is original for current, original
                   in zip(unit.Phases, original_phases))
        assert unit.Slurry.Solid_1 is unit.Solid_1
        assert unit.Slurry.Liquid_1 is unit.Liquid_1
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    with pytest.raises(InitializationCaptured):
        unit.solve_unit(runtime=0.01, verbose=False)  # [s], initialization only


@pytest.mark.assimulo
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_msmpr_reset_repeats_changing_population_solve(data_path, method):
    """Repeat complete dilution runs from the same configured seed inventory.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    method : str
        Moment or finite-volume population representation.
    """
    pytest.importorskip('assimulo')
    if method == 'moments':
        unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
        unit.Inlet.y_inlet['mu_n'] = np.zeros(unit.num_distr)  # [m**n/m**3]
    else:
        unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FLOW)
        unit.Inlet.distrib = np.zeros(unit.num_distr)  # [#/m**3/um], clear feed
    unit.reset_states = True
    duration = unit.Slurry.vol / FLOW  # [s], one nominal residence time
    times = np.array([0.0, duration])  # [s], endpoint reporting
    _, first = unit.solve_unit(time_grid=times, verbose=False)
    assert first[-1, 0] < first[0, 0] / 2  # [-], substantial population dilution
    _, second = unit.solve_unit(time_grid=times, verbose=False)
    np.testing.assert_allclose(second, first, rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_estimation_modifiers_refresh_slurry_initial_state(data_path, monkeypatch, method):
    """Normalize modified MSMPR inventories before the estimation solver starts.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only the solver-construction boundary.
    method : str
        Population representation.
    """
    if method == 'moments':
        unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
        total_population = unit.Solid_1.moments * 1e6**np.arange(unit.num_distr)
        # [um**n], total phase moments in the raw solver length basis
    else:
        unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FLOW)
        total_population = unit.Solid_1.distrib.copy()  # [#/um], total seed
    modified_liquid_volume = 2 * unit.Liquid_1.vol  # [m**3], doubled charge
    expected_volume = modified_liquid_volume + unit.Solid_1.vol  # [m**3]
    expected_population = total_population / expected_volume
    # [um**n/m**3] or [#/m**3/um], normalized on the modified slurry volume

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Check the modified volume basis at the public estimation handoff.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver options.
        states_init : ndarray
            Raw solver states with population entries first.
        params_mergd : ndarray
            Active kinetic parameters in their native units.

        Raises
        ------
        InitializationCaptured
            After verifying the modified initial population and volume.
        """
        np.testing.assert_allclose(states_init[:unit.num_distr],
                                   expected_population, rtol=RTOL, atol=0)
        assert unit.Slurry.vol == pytest.approx(expected_volume, rel=RTOL, abs=0)
        assert unit.vol_slurry == pytest.approx(expected_volume, rel=RTOL, abs=0)
        assert unit.vol_phase == pytest.approx(expected_volume, rel=RTOL, abs=0)
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    with pytest.raises(InitializationCaptured):
        unit.paramest_wrapper(
            unit.Kinetics.concat_params(), np.array([0.0, 0.01]),
            modify_phase={'Liquid': {'vol': modified_liquid_volume}})


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR])
@pytest.mark.parametrize('gridless', [False, True])
def test_solid_moment_modifier_preserves_liquid_inventory(
        data_path, monkeypatch, unit_type, gridless):
    """Change seed inventory without transferring volume out of the liquid.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only the solver-construction boundary.
    unit_type : type
        Batch or continuous crystallizer; population is total or normalized.
    gridless : bool
        Omit the plotting distribution or retain the original seed population.
    """
    unit, _ = inventory_unit(data_path, unit_type, gridless=gridless)
    liquid_volume = unit.Liquid_1.vol  # [m**3], independently charged liquid
    # Monodisperse 1e11 crystals of size 100 um: mu_n = count * size**n.
    modified_moments = np.array([1e11, 1e7, 1e3, 0.1])  # [m**n], total population
    solid_volume = 0.05  # [m**3], 1e11 crystals * 0.5 shape factor * (1e-4 m)**3
    expected_volume = liquid_volume + solid_volume  # [m**3], additive phase volumes
    expected_population = np.array([1e11, 1e13, 1e15, 1e17])  # [um**n]
    if unit_type is MSMPR:
        expected_population = expected_population / expected_volume  # [um**n/m**3]

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Verify independent phase inventories at the real solver handoff.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver options.
        states_init : ndarray
            Raw population, concentration, and optional liquid-volume states.
        params_mergd : ndarray
            Active kinetic parameters in their native units.

        Raises
        ------
        InitializationCaptured
            After checking liquid inventory, solid inventory, and population.
        """
        assert unit.Liquid_1.vol == pytest.approx(liquid_volume, rel=RTOL, abs=0)
        assert unit.Solid_1.vol == pytest.approx(solid_volume, rel=RTOL, abs=0)
        assert unit.Slurry.vol == pytest.approx(expected_volume, rel=RTOL, abs=0)
        assert unit.vol_phase == pytest.approx(expected_volume, rel=RTOL, abs=0)
        np.testing.assert_allclose(states_init[:unit.num_distr],
                                   expected_population, rtol=RTOL, atol=0)
        if unit_type is BatchCryst:
            assert states_init[-1] == pytest.approx(liquid_volume, rel=RTOL, abs=0)
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    with pytest.raises(InitializationCaptured):
        unit.paramest_wrapper(
            unit.Kinetics.concat_params(), np.array([0.0, 0.01]),
            modify_phase={'Solid': {'moments': modified_moments}})
