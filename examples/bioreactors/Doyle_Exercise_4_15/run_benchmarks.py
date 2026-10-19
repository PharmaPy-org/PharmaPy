"""Local textbook qualification using unchanged native PharmaPy construction.

Run with the target checkout on PYTHONPATH. Reference equations below are
independently written concentration balances, not calls to PharmaPy rates.
All generated evidence stays in outputs/. No parameter fitting is performed.
"""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import ss2tf

import PharmaPy
from PharmaPy.Bioreactors import build_bioreactor

HERE = Path(__file__).resolve().parent
SPECIES = ('biomass', 'substrate', 'product')


def reference_rates(time_h, state, dilution_h, parameters, substrate_feed):
    """Evaluate textbook concentration balances independently.

    Parameters
    ----------
    time_h : float
        Time [h], unused for constant post-step operation.
    state : array_like
        Biomass, substrate, product [g/L].
    dilution_h : float
        Dilution rate [1/h].
    parameters : dict
        Published rates [1/h], affinities [g/L], yields [g/g].
    substrate_feed : float
        Feed substrate concentration [g/L].

    Returns
    -------
    ndarray
        Concentration derivatives [g/(L h)].
    """
    biomass, substrate, product = state  # [g/L]
    affinities = parameters['affinities']
    denominator = affinities['Ks'] + substrate  # [g/L]
    inhibition = 1.0  # [-]; no inhibition in Exercise 4.15
    if 'Ki' in affinities:
        denominator += substrate**2 / affinities['Ki']
        inhibition -= product / affinities['Pm']
    growth = parameters['rates']['mu_max'] * inhibition * substrate / denominator  # [1/h]
    product_rate = (parameters['yields'].get('alpha', 0.) * growth
                    + parameters['rates'].get('beta', 0.)) * biomass  # [g/(L h)]
    return np.array([(growth - dilution_h) * biomass,
                     dilution_h * (substrate_feed - substrate)
                     - growth * biomass / parameters['yields']['biomass'],
                     product_rate - dilution_h * product])


def run_native(mechanism, refined=False):
    """Construct a fresh native vessel and return concentration histories.

    Parameters
    ----------
    mechanism : dict
        Standard JSON input declarations with units.
    refined : bool
        Tighten both tolerances tenfold and halve maximum step when True.

    Returns
    -------
    tuple
        Time [h], concentrations [g/L], volume [L], wall time [s].
    """
    mechanism = json.loads(json.dumps(mechanism))
    if refined:
        mechanism['numerical']['relative_tolerance'] /= 10.  # [-]; refinement factor
        mechanism['numerical']['absolute_tolerance'] /= 10.  # [native units]
        mechanism['numerical']['maximum_step']['value'] /= 2.  # [h]
    started = time.perf_counter()  # [s]
    assembly = build_bioreactor(mechanism, thermo_path=HERE/'inputs/thermo.json')
    histories = assembly.solve()
    positions = [list(assembly.phase.name_species).index(name) for name in SPECIES]
    times, states, volumes = [], [], []
    for history in histories:
        volume = np.asarray(history.vessel_vol).reshape(-1) * 1000.  # [L]
        times.append(np.asarray(history.time) / 3600.)  # [h]
        states.append(np.asarray(history.mass_j_liquid0)[:, positions] * 1000. / volume[:, None])  # [g/L]
        volumes.append(volume)
    return np.concatenate(times), np.vstack(states), np.concatenate(volumes), time.perf_counter()-started


def linearized_evidence(state, dilution, parameters, feed):
    """Calculate local dynamics at an equation-consistent equilibrium.

    Parameters
    ----------
    state : array_like
        Published initial concentrations [g/L], used only as root initial guess.
    dilution : float
        Nominal dilution [1/h].
    parameters : dict
        Source kinetic coefficients in declared units.
    feed : float
        Feed substrate [g/L].

    Returns
    -------
    dict
        Corrected equilibrium, poles/zeros [1/h], gain [(g/L)/(1/h)].

    Raises
    ------
    RuntimeError
        The independent equilibrium calculation fails.
    """
    solution = root(equilibrium_residual, state, args=(dilution, parameters, feed))
    if not solution.success or np.max(np.abs(equilibrium_residual(solution.x, dilution, parameters, feed))) > 1e-9:
        raise RuntimeError('Independent equilibrium calculation failed')
    biomass, substrate, product = solution.x  # [g/L]
    aff, rates, yields = parameters['affinities'], parameters['rates'], parameters['yields']
    denominator = aff['Ks'] + substrate  # [g/L]
    derivative_denominator = 1.  # [-]
    inhibition, derivative_product = 1., 0.  # [-], [L/(g h)]
    if 'Ki' in aff:
        denominator += substrate**2 / aff['Ki']
        derivative_denominator += 2.*substrate/aff['Ki']
        inhibition -= product/aff['Pm']
        derivative_product = -rates['mu_max']*substrate/(aff['Pm']*denominator)
    mu = rates['mu_max']*inhibition*substrate/denominator  # [1/h]
    derivative_substrate = rates['mu_max']*inhibition*(denominator-substrate*derivative_denominator)/denominator**2  # [L/(g h)]
    yield_x = yields['biomass']  # [gDW/g substrate]
    alpha, beta = yields.get('alpha', 0.), rates.get('beta', 0.)  # [gP/gDW], [gP/(gDW h)]
    matrix = np.array([[mu-dilution, biomass*derivative_substrate, biomass*derivative_product],
        [-mu/yield_x, -dilution-biomass*derivative_substrate/yield_x, -biomass*derivative_product/yield_x],
        [alpha*mu+beta, alpha*biomass*derivative_substrate, alpha*biomass*derivative_product-dilution]])  # [1/h]
    forcing = np.array([-biomass, feed-substrate, -product])  # [g/L]
    numerator, denominator_poly = ss2tf(matrix, forcing[:, None], [[1., 0., 0.]], [[0.]])
    # ss2tf's first coefficient is zero for this strictly proper system.
    zeros = np.roots(numerator[0, 1:])  # [1/h]; includes uncancelled numerical pairs
    return dict(equilibrium_g_l=solution.x.tolist(), poles_h=[str(x) for x in np.linalg.eigvals(matrix)],
                zeros_h=[str(x) for x in zeros], gain_g_l_per_h_inv=float(-np.linalg.solve(matrix, forcing)[0]))


def equilibrium_residual(state, dilution, parameters, feed):
    """Return the steady-state balance residual.

    Parameters
    ----------
    state : array_like
        Concentrations [g/L].
    dilution : float
        Dilution [1/h].
    parameters : dict
        Source kinetic parameters with declared units.
    feed : float
        Feed concentration [g/L].

    Returns
    -------
    ndarray
        Residual [g/(L h)].
    """
    return reference_rates(0., state, dilution, parameters, feed)


def run_scenario(scenario, specification, output):
    """Compare one input-only native run with two independent integrations.

    Parameters
    ----------
    scenario, specification : dict
        Explicit operating conditions and predeclared acceptance budgets.
    output : Path
        Destination of generated inputs, trajectories and plots.

    Returns
    -------
    dict
        Per-species errors [g/L], residuals, runtimes, and pass/fail evidence.
    """
    model = json.loads((HERE/'inputs/mechanism.json').read_text())
    initial = np.asarray(scenario['initial_g_l'])  # [g/L]
    dilution = scenario['D_h']  # [1/h]
    volume = model['operation']['initial_volume']['value']  # [L]
    model['state']['species']['amounts'] = dict(zip(SPECIES, (initial*volume).tolist()))  # [g]
    model['recipes']['continuous'][0]['volume_l_h'] = dilution*volume  # [L/h]
    feed = model['recipes']['continuous'][0]['concentrations_g_l']['substrate']  # [g/L]
    parameters = model['parameters']
    folder = output/scenario['name']
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'mechanism.json').write_text(json.dumps(model, indent=2)+'\n')
    times, native, volumes, elapsed = run_native(model)
    refined_times, refined, _, refined_elapsed = run_native(model, refined=True)
    np.testing.assert_array_equal(times, refined_times)
    settings = specification['reference_numerics']
    references = []
    for method in ('DOP853', 'Radau'):
        result = solve_ivp(reference_rates, (times[0], times[-1]), initial, t_eval=times,
            args=(dilution, parameters, feed), method=method, rtol=settings['rtol'],
            atol=settings['atol'], max_step=settings['max_step_h'])
        if not result.success:
            raise RuntimeError(f'{method} reference failed: {result.message}')
        references.append(result.y.T)
    reference = references[0]  # [g/L]
    errors = np.max(np.abs(native-reference), axis=0)  # [g/L]
    agreement = float(np.max(np.abs(reference-references[1])))  # [g/L]
    refinement = float(np.max(np.abs(native-refined)))  # [g/L]
    yield_x = parameters['yields']['biomass']  # [gDW/g substrate]
    equivalent = native[:, 1]+native[:, 0]/yield_x  # [g substrate equivalent/L]
    exact = feed+(initial[1]+initial[0]/yield_x-feed)*np.exp(-dilution*times)  # [g/L]
    balance_error = float(np.max(np.abs(equivalent-exact)))  # [g/L]
    limits = specification['acceptance']
    checks = dict(trajectory=bool(errors.max() <= limits['trajectory_max_abs_g_l']),
        independent_references=agreement <= limits['reference_agreement_g_l'],
        refinement=refinement <= limits['refinement_max_abs_g_l'],
        equivalent_substrate=balance_error <= limits['equivalent_substrate_g_l'],
        volume=bool(np.max(np.abs(volumes-volume)) <= limits['constant_volume_l']),
        nonnegative=bool(native.min() >= -limits['nonnegative_g_l']),
        finite=bool(np.isfinite(native).all()))
    columns = np.column_stack([times, native, reference, refined, volumes])
    np.savetxt(folder/'trajectories.csv', columns, delimiter=',', comments='',
        header='time_h,'+','.join(f'{kind}_{name}_g_l' for kind in ('native','reference','refined') for name in SPECIES)+',volume_l')
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for column, axis in enumerate(axes):
        axis.plot(times, native[:, column], label='PharmaPy BDF')
        axis.plot(times[::16], reference[::16, column], 'o', fillstyle='none', label='Independent DOP853')
        axis.set(xlabel='Time [h]', ylabel=f'{SPECIES[column]} [g/L]')
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(scenario['name'])
    figure.tight_layout()
    figure.savefig(folder/'comparison.png', dpi=160)
    plt.close(figure)
    report = dict(scenario=scenario, max_abs_error_g_l=dict(zip(SPECIES, errors.tolist())),
        independent_reference_difference_g_l=agreement, refinement_difference_g_l=refinement,
        equivalent_substrate_error_g_l=balance_error, final_native_g_l=native[-1].tolist(),
        initial_residual_g_l_h=reference_rates(0., initial, dilution, parameters, feed).tolist(),
        native_wall_s=elapsed, refined_wall_s=refined_elapsed, checks=checks, passed=all(checks.values()))
    if scenario['name'].endswith('nominal'):
        report['linearization'] = linearized_evidence(initial, dilution, parameters, feed)
    (folder/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    """Run all declared cases and retain reproducible qualification evidence.

    Raises
    ------
    SystemExit
        Any predeclared acceptance criterion fails.
    """
    specification = json.loads((HERE/'benchmark.json').read_text())
    output = HERE/'outputs'
    output.mkdir(exist_ok=True)
    repo = Path(PharmaPy.__file__).resolve().parent.parent
    report = dict(source=specification['source'], acceptance=specification['acceptance'],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         pharmapy=str(repo), commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()),
        input_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [HERE/'benchmark.json', HERE/'run_benchmarks.py', *sorted((HERE/'inputs').glob('*.json'))]},
        cases=[])
    for scenario in specification['scenarios']:
        print('Running', scenario['name'], flush=True)
        result = run_scenario(scenario, specification, output)
        report['cases'].append(result)
        print(result['passed'], result['max_abs_error_g_l'], flush=True)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    report['passed'] = all(case['passed'] for case in report['cases'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise SystemExit('One or more predeclared benchmark criteria failed; see outputs/report.json')


if __name__ == '__main__':
    main()
