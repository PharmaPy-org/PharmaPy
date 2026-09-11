"""Crystallizer solver support, option precedence, and SciPy floor contracts.

Native sensitivity comparisons use synthetic nucleation and growth with real
phases, and independently perturb complete solves of the material balances.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from test_crystallizer_moment_inventory import inventory_unit


@pytest.mark.unit
@pytest.mark.parametrize('unit_type, method, radius', [
    (BatchCryst, '1D-FVM', 0.), (BatchCryst, 'moments', 10.),
    (MSMPR, 'moments', 0.), (SemibatchCryst, 'moments', 0.),
])  # radius [um], deliberately unsupported analytical combinations
def test_analytical_sensitivity_support_is_checked_before_solver(data_path, unit_type, method, radius):
    unit, _ = inventory_unit(data_path, unit_type)
    unit.method = method
    unit.rad = radius  # [um]
    unit.jac_type = 'analytical'
    with pytest.raises(NotImplementedError, match='analytical.*BatchCryst.*moments.*rad_zero=0'):
        unit.solve_unit(runtime=0.01, eval_sens=True, verbose=False)


def nucleating_batch(data_path, jacobian):
    """Build a positive seeded population with independently measurable rates.

    Parameters
    ----------
    data_path : dict
        Repository thermophysical data paths.
    jacobian : str or None
        Analytical, numerical, or solver-generated sensitivity Jacobian.

    Returns
    -------
    BatchCryst
        Isothermal total-moment model at unit relative supersaturation [-],
        with primary nucleation and growth prefactors selected for sensitivity.
    """
    from test_crystallizer_nucleation_jacobians import make_unit
    unit, _ = make_unit(data_path, 'volume')
    from PharmaPy.Phases import LiquidPhase, SolidPhase
    density, _ = unit.Liquid_1.getDensityPure(phase='liquid')  # [kg/m**3]
    concentration = 4.0  # [kg/m**3], twice the synthetic solubility
    target_fraction = ((concentration / density[-1])
                       / (1 - concentration / density[0] + concentration / density[-1]))  # [-]
    # Solve c_A=w_A/(w_A/rho_A+(1-w_A)/rho_solvent) for a binary feed.
    liquid = LiquidPhase(unit.Liquid_1.path_data, temp=320., vol=2.,
                         mass_frac=[target_fraction, 0, 0, 0, 1-target_fraction])  # [K], [m**3], [-]
    solid = SolidPhase(unit.Solid_1.path_data, temp=320., kv=.5,
                       mass_frac=[1, 0, 0, 0, 0],
                       moments=4e7 * (1e-6)**np.arange(4))
    # [m**n], forty million one-micrometer seeds; constructor reconciles mass.
    # Newly nucleated moments stay measurable without subtracting O(1e17) states.
    unit.Phases = (liquid, solid)
    assert unit.Liquid_1.mass_conc[0] == pytest.approx(concentration, rel=1e-12)
    seed_fraction = unit.Solid_1.vol / unit.Slurry.vol  # [-]
    secondary_rate = 1e8  # [#/m**3/s], comparable to primary 2e8 at the initial state
    unit.Kinetics.params['nucl_sec'][0] = secondary_rate / seed_fraction**0.7
    # [#/m**3/s / volume_fraction**0.7], chosen to make growth's secondary
    # nucleation feedback measurable in the zeroth moment as well.
    unit.controls['temp']['fun'] = lambda time: 320.0 + np.zeros_like(time)  # [K], constant program
    unit.jac_type = jacobian
    unit.mask_params[:] = False
    unit.mask_params[[0, 7]] = True
    return unit


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('jacobian', ['analytical', 'finite_diff'])
def test_native_nucleation_and_growth_sensitivities_match_complete_solves(data_path, jacobian):
    pytest.importorskip('assimulo')
    unit = nucleating_batch(data_path, jacobian)
    times = np.linspace(0., 2., 5)  # [s], measurable population change, short thermal program
    options = {'maxh': .05, 'rtol': 1e-10, 'atol': 1e-12,
               'sensmethod': 'STAGGERED', 'suppress_sens': True,
               'report_continuously': False}  # [s], [-], state units, backend selectors
    # These explicit sensitivity settings are exercised for precedence, not
    # prescribed as new defaults. Continuous reporting must still collect p_sol.
    time, states, sensitivities = unit.solve_unit(time_grid=times, eval_sens=True,
                                                 sundials_opts=options, verbose=False)
    assert unit.sundials_opt['sensmethod'] == 'STAGGERED'
    assert unit.sundials_opt['suppress_sens'] is True
    assert unit.sundials_opt['report_continuously'] is True
    np.testing.assert_array_equal(time, times)
    assert len(sensitivities) == 2
    assert states[-1, 0] > states[0, 0]
    for index, parameter_column in enumerate([0, 7]):
        for relative_step in [1e-3, 5e-4]:  # [-], central differences with independent refinement
            terminal = []  # total raw moment states [um**n]
            reference = unit.Kinetics.concat_params()[parameter_column]  # [native prefactor unit]
            step = reference * relative_step  # [native prefactor unit]
            for sign in [-1, 1]:
                trial = nucleating_batch(data_path, jacobian)
                parameters = trial.Kinetics.concat_params()  # [native kinetic units]
                parameters[parameter_column] += sign * step
                trial.Kinetics.set_params(parameters)
                _, trial_states = trial.solve_unit(time_grid=times, verbose=False,
                                                   sundials_opts={**options, 'report_continuously': True})
                terminal.append(trial_states[-1, :4])
            expected = (terminal[1] - terminal[0]) / (2 * step)  # [um**n/parameter unit]
            # 2e-5 bounds finite-difference cancellation in the weak primary
            # contribution to a large seed; both perturbation sizes must agree.
            np.testing.assert_allclose(sensitivities[index][-1, :4], expected, rtol=2e-5, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('radius', [5.0, 20.0])  # [um], differs from first grid point 10 um
@pytest.mark.parametrize('boundary', ['solve', 'rhs'])
def test_finite_nuclei_require_matching_fvm_grid_edge(data_path, radius, boundary):
    from test_crystallizer_parameter_evaluations import make_unit
    unit, states = make_unit(data_path, radius=radius)
    with pytest.raises(ValueError, match='rad_zero.*first.*grid'):
        if boundary == 'solve':
            unit.solve_unit(runtime=.01, verbose=False)
        else:
            unit.unit_model(0., states)


@pytest.mark.unit
@pytest.mark.parametrize('radius', [0., 10.])  # [um], legacy point nuclei or matching finite nuclei
def test_fvm_supported_nucleus_grid_modes(data_path, radius):
    from test_crystallizer_parameter_evaluations import make_unit
    unit, states = make_unit(data_path, radius=radius)
    derivative = unit.unit_model(0., states)  # [state units/s]
    assert derivative.shape == states.shape
    assert np.isfinite(derivative).all()
