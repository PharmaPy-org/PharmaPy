"""B011 real CVode filter regressions for parameters, time grids and division."""

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]

from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.SimExec import SimulationExec
from test_separation_balance_fixes import RTOL, separation_phases


def pressure_resistance(pressure):
    """Return a synthetic linear pressure-dependent cake resistance.

    Parameters
    ----------
    pressure : float
        Pressure drop [Pa].

    Returns
    -------
    float
        Specific resistance [m/kg]; 1e6 m/kg/Pa is the synthetic slope.
    """
    slope = 1e6  # [m/kg/Pa], makes cake and medium terms measurable
    return slope * pressure


def pressure_medium(pressure):
    """Return a synthetic pressure-dependent medium resistance.

    Parameters
    ----------
    pressure : float
        Pressure drop [Pa].

    Returns
    -------
    float
        Medium resistance [1/m]; 1e4 1/m/Pa is the synthetic slope.
    """
    slope = 1e4  # [1/m/Pa], 1e9 1/m at a one-bar pressure drop
    return slope * pressure


@pytest.mark.parametrize('log_params', [False, True])
@pytest.mark.parametrize('callable_alpha', [False, True])
@pytest.mark.parametrize('list_grid', [False, True])
def test_filter_resolves_parameters_and_preserves_grids_across_pressures(
        separation_phases, callable_alpha, list_grid, log_params):
    liquid, solid = separation_phases
    alpha = pressure_resistance if callable_alpha else 1e11  # [m/kg] or callable
    medium = pressure_medium if callable_alpha else 1e9  # [1/m] or callable
    unit = Filter(station_diam=0.1, alpha=alpha, resist_medium=medium, log_params=log_params)
    unit.Phases = [liquid, solid]
    grid = [0., 0.01, 0.02] if list_grid else np.array([0., 0.01, 0.02])  # [s], pre-completion
    initial_grid = np.array(grid)  # [s]
    elapsed = 0.  # [s]
    solver_tolerance = 2e-6  # [-], tighter than default CVode for integral check
    for pressure in [1e5, 2e5, 1e5]:  # [Pa], repeat original pressure to expose lost callable
        times, states = unit.solve_unit(
            time_grid=grid, deltaP=pressure, verbose=False,
            sundials_opts={'rtol': 1e-9, 'atol': 1e-11})  # [-], [kg], sub-test tolerance
        resolved_alpha = pressure_resistance(pressure) if callable_alpha else alpha  # [m/kg]
        resolved_medium = pressure_medium(pressure) if callable_alpha else medium  # [1/m]
        np.testing.assert_allclose(unit.params, [resolved_alpha, resolved_medium], rtol=RTOL)
        assert unit.r_medium == medium
        assert callable(unit.r_medium) == callable_alpha
        assert unit.alpha == alpha
        assert callable(unit.alpha) == callable_alpha
        if callable_alpha:
            with pytest.raises(ValueError, match='Callable resistances are not estimable'):
                unit.param_seed
        np.testing.assert_array_equal(grid, initial_grid)
        assert isinstance(times, np.ndarray)
        np.testing.assert_allclose(times, elapsed + initial_grid, rtol=RTOL, atol=RTOL)
        density, viscosity, _, _ = unit.physical_props  # [kg/m**3], [Pa*s], [N/m], [kg/m**3]
        # Integrate Darcy's ODE independently: A*m**2/2+B*m=deltaP*t/mu.
        cake_coefficient = resolved_alpha * unit.c_solids / (unit.area_filt * density)**2  # [1/kg/m]
        medium_coefficient = resolved_medium / (unit.area_filt * density)  # [1/m]
        filtered = states[:, 0]  # [kg]
        elapsed_from_mass = viscosity / pressure * (
            cake_coefficient * filtered**2 / 2 + medium_coefficient * filtered)  # [s]
        np.testing.assert_allclose(elapsed_from_mass, initial_grid,
                                   rtol=solver_tolerance, atol=1e-9)
        elapsed += initial_grid[-1]
        assert unit.elapsed_time == pytest.approx(elapsed, rel=RTOL)


def test_filter_divided_batch_cake_closes_population_and_mass(separation_phases):
    liquid, solid = separation_phases
    unit = Filter(station_diam=0.1)
    unit.Phases = [liquid, solid]
    _, states = unit.solve_unit(slurry_div=2, verbose=False,
                                sundials_opts={'rtol': 1e-9, 'atol': 1e-11})
    tolerance = 1e-7  # [-], event-location and integration allowance
    assert unit.Outlet.Solid_1.mass == pytest.approx(solid.mass / 2, rel=tolerance)
    np.testing.assert_allclose(unit.Outlet.Solid_1.distrib, solid.distrib / 2, rtol=tolerance)
    volume_mass = unit.Outlet.Solid_1.kv * unit.Outlet.Solid_1.moments[3] * solid.getDensity()  # [kg]
    assert volume_mass == pytest.approx(unit.Outlet.Solid_1.mass, rel=RTOL)
    np.testing.assert_allclose(states.sum(axis=1), liquid.mass / 2, rtol=tolerance)


@pytest.mark.parametrize('log_params', [False, True])
def test_filter_parameter_override_controls_balance_and_completion_time(separation_phases, log_params):
    liquid, solid = separation_phases
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=log_params)  # [m/kg]
    unit.Phases = [liquid, solid]
    parameters = np.array([2e11, 3e9])  # [m/kg, 1/m], distinct from constructor
    grid = np.array([0., 0.01, 0.02])  # [s], pre-completion batch
    time, states = unit.solve_unit(
        time_grid=grid, model_params=np.log(parameters) if log_params else parameters,
        verbose=False, sundials_opts={'rtol': 1e-9, 'atol': 1e-11})
    np.testing.assert_allclose(unit.params, parameters, rtol=RTOL)
    unit.reset()
    np.testing.assert_allclose(unit.params, parameters, rtol=RTOL)
    assert unit.elapsed_time == 0
    density, viscosity, _, _ = unit.physical_props  # [kg/m**3], [Pa*s], [N/m], [kg/m**3]
    filtrate_volumes = states[:, 0] / density  # [m**3]
    # Constant-pressure filtration integral, including both resolved terms.
    expected_times = viscosity / unit.deltaP * (
        parameters[0] * unit.c_solids * filtrate_volumes**2 / (2 * unit.area_filt**2)
        + parameters[1] * filtrate_volumes / unit.area_filt)  # [s]
    np.testing.assert_allclose(time, expected_times, rtol=2e-6, atol=1e-9)  # integration allowance
    final_volume = unit.mass_crit / density  # [m**3]
    expected_completion = viscosity / unit.deltaP * (
        parameters[0] * unit.c_solids * final_volume**2 / (2 * unit.area_filt**2)
        + parameters[1] * final_volume / unit.area_filt)  # [s]
    assert unit.time_filt == pytest.approx(expected_completion, rel=RTOL)


def test_filter_log_seed_wrapper_matches_physical_solve(separation_phases):
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=True)  # [m/kg]
    unit.Phases = separation_phases
    grid = np.array([0., 0.01, 0.02])  # [s], pre-completion batch
    _, physical_states = unit.solve_unit(time_grid=grid, verbose=False)
    with np.errstate(over='raise'):
        actual = unit.paramest_wrapper(unit.param_seed, grid)
    assert actual[-1] > 0  # [kg], ensure filtration was exercised
    np.testing.assert_allclose(actual, physical_states[:, 0], rtol=1e-7, atol=1e-10)  # CVode allowance [kg]


@pytest.mark.parametrize('pressure_setting', ['explicit', 'omitted', 'none'])
def test_simexec_filter_log_seed_gives_finite_objective(separation_phases, pressure_setting):
    reference = Filter(station_diam=0.1, alpha=1e11)  # [m/kg]
    reference.Phases = separation_phases
    grid = np.array([0., 0.01, 0.02])  # [s], pre-completion measurements
    run_args = {'deltaP': 2e5, 'slurry_div': 2, 'verbose': False,
                'sundials_opts': {'rtol': 1e-9, 'atol': 1e-11}}  # [Pa], [-], solver tolerances [-], [kg]
    if pressure_setting != 'explicit':
        run_args.pop('deltaP')
    _, states = reference.solve_unit(time_grid=grid, **run_args)
    if pressure_setting == 'none':
        run_args['deltaP'] = None
    offset = 1e-3  # [kg], explicit measurement offset gives a nonzero objective
    unit = Filter(station_diam=0.1, alpha=1e11, log_params=True)  # [m/kg]
    unit.Phases = separation_phases
    simulation = SimulationExec(unit.Liquid_1.path_data, {'F01': []})
    simulation.F01 = unit
    simulation.SetParamEstimation(grid, states[:, 0] + offset, wrapper_kwargs=run_args)
    with np.errstate(over='raise'):
        objective = simulation.ParamInst.get_objective(simulation.ParamInst.param_seed)
    assert np.isfinite(objective)
    expected = len(grid) * offset**2 / 2  # [-], unit measurement weights in objective
    assert objective == pytest.approx(expected, rel=1e-7)
    np.testing.assert_allclose(simulation.ParamInst.y_runs[0][:, 0], states[:, 0],
                               rtol=1e-7, atol=1e-10)  # CVode allowance [kg]
    assert unit.deltaP == (2e5 if pressure_setting == 'explicit' else 1e5)  # [Pa], supplied/default
    assert unit.elapsed_time == pytest.approx(grid[-1], rel=RTOL)


@pytest.mark.parametrize('modifier', ['modify_phase', 'modify_controls'])
def test_filter_wrapper_rejects_unsupported_modifiers(separation_phases, modifier):
    unit = Filter(station_diam=0.1, alpha=1e11)  # [m/kg]
    unit.Phases = separation_phases
    grid = np.array([0., 0.01, 0.02])  # [s], pre-completion batch
    unit.solve_unit(time_grid=grid, verbose=False)
    unsupported = {'temp': 300.} if modifier == 'modify_phase' else {'deltaP': 2e5}  # [K] or [Pa]
    with pytest.raises(ValueError, match=modifier + '.*not supported'):
        unit.paramest_wrapper(unit.param_seed, grid, **{modifier: unsupported})
    assert unit.elapsed_time == pytest.approx(grid[-1], rel=RTOL)


@pytest.fixture
def filtration_plateau_data(separation_phases):
    """Generate observations from the integrated Darcy law and physical plateau.

    Parameters
    ----------
    separation_phases : tuple
        Liquid and solid inventories [kg] from the common fixture.

    Returns
    -------
    tuple
        Reference Filter, times [s], filtrate masses [kg], generating
        parameters [m/kg, 1/m], and CVode tolerance options.
    """
    generating = np.array([1e10, 1e9])  # [m/kg, 1/m], identifiable cake and medium terms
    reference = Filter(station_diam=0.1, alpha=generating[0], resist_medium=generating[1])
    reference.Phases = separation_phases
    solver_options = {'rtol': 1e-11, 'atol': 1e-13}  # [-], [kg], below finite-difference noise budget
    reference.solve_unit(time_grid=[0., 0.01], verbose=False, sundials_opts=solver_options)  # [s]
    grid = np.linspace(0, 1.2 * reference.time_filt, 25)  # [s], includes the physical plateau
    density, viscosity, _, _ = reference.physical_props  # [kg/m**3], [Pa*s], [N/m], [kg/m**3]
    # Integrate dm/dt = c/(2*a*m+b): a*m**2+b*m = c*t.
    quadratic = generating[0] * reference.c_solids / (2 * (reference.area_filt * density)**2)  # [1/kg**2]
    linear = generating[1] / (reference.area_filt * density)  # [1/kg]
    forcing = reference.deltaP / viscosity  # [1/s]
    growing_mass = 2 * forcing * grid / (linear + np.sqrt(linear**2 + 4 * quadratic * forcing * grid))  # [kg]
    observed = np.minimum(growing_mass, reference.mass_crit)  # [kg], exhausted free liquid
    return reference, grid, observed, generating, solver_options


@pytest.mark.parametrize('trial_factor', [0.3, 1.])  # [-], fast-side and exact seeds
@pytest.mark.parametrize('end_factor', [1., 1.2])  # [-], exact completion and plateau endpoint
@pytest.mark.parametrize('elapsed', [0., 1e6])  # [s], fresh and accumulated batch clocks
def test_filter_estimation_holds_completion_plateau(
        separation_phases, filtration_plateau_data, trial_factor, end_factor, elapsed):
    reference, _, _, generating, solver_options = filtration_plateau_data
    unit = Filter(station_diam=0.1, alpha=generating[0] * trial_factor,
                  resist_medium=generating[1] * trial_factor, log_params=False)
    unit.Phases = separation_phases
    unit.elapsed_time = elapsed  # [s]
    # The exact-completion assertion depends on bit-identical time_filt values.
    grid = np.array([0., 0.4, end_factor]) * reference.time_filt  # [s], relative input times
    input_grid = grid.copy()  # [s]
    with np.errstate(over='raise'):
        time, states = unit.solve_unit(time_grid=grid, model_params=unit.param_seed,
                                      verbose=False, sundials_opts=solver_options)
    plateau = [unit.mass_crit, separation_phases[0].mass - unit.mass_crit]  # [kg, kg]
    np.testing.assert_array_equal(states[-1], plateau)
    np.testing.assert_allclose(states.sum(axis=1), separation_phases[0].mass, rtol=RTOL)
    np.testing.assert_array_equal(unit.Outlet.Solid_1.distrib, separation_phases[1].distrib)
    assert unit.Outlet.Solid_1.mass == separation_phases[1].mass  # [kg], exact inventory plateau
    np.testing.assert_array_equal(time, grid + elapsed)
    np.testing.assert_array_equal(grid, input_grid)
    assert np.all(np.isfinite(states))


def test_simexec_lm_fits_filter_plateau_from_slow_seed(separation_phases, filtration_plateau_data):
    reference, grid, observed, generating, solver_options = filtration_plateau_data
    unit = Filter(station_diam=0.1, alpha=3e10, resist_medium=3e9, log_params=True)  # [m/kg], [1/m], slow-side seed
    unit.Phases = separation_phases
    simulation = SimulationExec(unit.Liquid_1.path_data, {'F01': []})
    simulation.F01 = unit
    simulation.SetParamEstimation(
        grid, observed, dx_finitediff=1e-4,  # [-], log-space step larger than CVode error
        wrapper_kwargs={'verbose': False, 'sundials_opts': solver_options})
    fitted, _, _ = simulation.ParamInst.optimize_fn(
        method='LM', verbose=False,
        optim_options={'eps_1': 1e-10, 'eps_2': 1e-10, 'tol_fun': 1e-10})  # [-], convergence tolerances
    np.testing.assert_allclose(np.exp(fitted), generating, rtol=1e-4)  # finite-difference/LM allowance [-]
    predicted = unit.paramest_wrapper(fitted, grid, run_args={'sundials_opts': solver_options})  # [kg]
    np.testing.assert_allclose(predicted, observed, rtol=1e-5, atol=1e-8)  # fit allowance [-], [kg]
    assert predicted[-1] == reference.mass_crit


def test_filter_rejects_unbounded_estimation_before_mutation(separation_phases):
    unit = Filter(station_diam=0.1, alpha=1e11)  # [m/kg]
    unit.Phases = separation_phases
    original = unit.__dict__.copy()
    with pytest.raises(ValueError, match='runtime|time_grid'):
        unit.solve_unit(model_params=unit.param_seed, verbose=False)
    assert unit.__dict__.keys() == original.keys()
    assert unit.params == original['params']
    assert unit.deltaP == original['deltaP']
    assert unit.elapsed_time == original['elapsed_time']


def test_filter_estimation_runtime_holds_completion(separation_phases, filtration_plateau_data):
    reference, _, _, generating, solver_options = filtration_plateau_data
    unit = Filter(station_diam=0.1, alpha=generating[0], resist_medium=generating[1])
    unit.Phases = separation_phases
    runtime = 1.2 * reference.time_filt  # [s], bounded run past completion
    time, states = unit.solve_unit(runtime=runtime, model_params=unit.param_seed,
                                  verbose=False, sundials_opts=solver_options)
    np.testing.assert_array_equal(states[-2:, 0], np.full(2, unit.mass_crit))
    assert time[-2] == unit.time_filt
    assert time[-1] == runtime
    np.testing.assert_allclose(states.sum(axis=1), separation_phases[0].mass, rtol=RTOL)


@pytest.mark.parametrize('grid_kind', ['growing', 'mixed', 'plateau'])
def test_simexec_filter_returns_only_requested_samples(
        separation_phases, filtration_plateau_data, grid_kind):
    reference, grid, observed, generating, solver_options = filtration_plateau_data
    selected = {'growing': slice(1, 11), 'mixed': slice(1, None),
                'plateau': grid > reference.time_filt}[grid_kind]
    grid, observed = grid[selected], observed[selected]  # [s], [kg], no initial-condition sample
    unit = Filter(station_diam=0.1, alpha=generating[0], resist_medium=generating[1])
    unit.Phases = separation_phases
    simulation = SimulationExec(unit.Liquid_1.path_data, {'F01': []})
    simulation.F01 = unit
    simulation.SetParamEstimation(
        grid, observed, wrapper_kwargs={'verbose': False, 'sundials_opts': solver_options})
    objective = simulation.ParamInst.get_objective(simulation.ParamInst.param_seed)
    predictions = simulation.ParamInst.y_runs[0][:, 0]  # [kg]
    assert np.isfinite(objective)
    assert predictions.shape == observed.shape
    np.testing.assert_allclose(predictions, observed, rtol=1e-7, atol=1e-10)  # CVode allowance [-], [kg]
    start = unit.elapsed_time  # [s], also check a later batch clock
    time, states = unit.solve_unit(time_grid=grid, model_params=unit.param_seed,
                                  verbose=False, sundials_opts=solver_options)
    assert isinstance(time, np.ndarray)
    np.testing.assert_array_equal(time, grid + start)
    assert states.shape == (len(grid), 2)


def test_filter_dense_default_grid_is_covered_or_reports_solver_options(
        separation_phases, filtration_plateau_data):
    """Detect omitted CVode final-step samples over a range of trial resistances.

    Notes
    -----
    A 101-point grid and 41 resistance factors from 0.3 to 1.5 reproduce
    short profiles at default tolerances. Coverage varies with CVode builds:
    each trial must either retain every requested sample or explain how to
    tighten sundials_opts, before an estimator attempts residual subtraction.
    """
    reference, _, _, generating, _ = filtration_plateau_data
    unit = Filter(station_diam=0.1, alpha=generating[0], resist_medium=generating[1])
    unit.Phases = separation_phases
    grid = np.linspace(0, 1.2 * reference.time_filt, 101)  # [s], dense plateau measurements
    for factor in np.linspace(0.3, 1.5, 41):  # [-], fast- through slow-side optimizer trials
        unit.reset()
        try:
            time, states = unit.solve_unit(time_grid=grid, model_params=generating * factor,
                                          verbose=False)
        except RuntimeError as error:
            assert 'sundials_opts' in str(error)
            assert 'rtol' in str(error) and 'atol' in str(error)
        else:
            assert isinstance(time, np.ndarray)
            np.testing.assert_array_equal(time, grid)
            assert states.shape == (len(grid), 2)
            assert np.all(np.isfinite(states))
