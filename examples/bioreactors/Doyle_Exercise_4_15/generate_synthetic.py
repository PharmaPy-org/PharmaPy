"""Generate local Doyle 4.15 estimation data; never fit or change simulator code."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

if __package__:
    from .run_benchmarks import reference_rates
else:
    from run_benchmarks import reference_rates

HERE = Path(__file__).resolve().parent


def write_rows(path: Path, fields: list, rows: list) -> None:
    """Save named observation records without altering numerical precision.

    Parameters
    ----------
    path : Path
        Destination CSV.
    fields : list
        Column names, including unit suffixes for numerical quantities.
    rows : list
        Records with times [h], concentrations and uncertainties [g/L].
    """
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(output: Path, *, truth_theta: float = 0., design: dict = None) -> dict:
    """Generate existing CSV assets with independently checked synthetic truth.

    Parameters
    ----------
    output : Path
        Destination for generated CSVs and provenance manifest.
    truth_theta : float, optional
        Log correction [-], within the hybrid's declared domain. Zero retains
        the original mechanistic generator, including paired random draws.
    design : dict, optional
        Existing synthetic design: times [h], D [1/h], relative noise [-],
        absolute noise [g/L], seed and realization count. Defaults to its JSON.

    Returns
    -------
    dict
        Manifest containing source checksums, truth inputs and numerical checks.

    Raises
    ------
    ValueError
        Truth is nonfinite/out of domain, or the design is malformed.
    RuntimeError
        A reference integration fails or independently computed solutions disagree.
    """
    config = (json.loads((HERE/'inputs/synthetic_design.json').read_text())
              if design is None else json.loads(json.dumps(design, allow_nan=False)))
    hybrid = json.loads((HERE/'inputs/hybrid.json').read_text())
    bounds = hybrid['rate_provider']['validity_domain']['theta']  # [-]
    if not np.isfinite(truth_theta) or not bounds[0] <= truth_theta <= bounds[1]:
        raise ValueError('truth_theta must lie in the declared hybrid domain')
    model = json.loads((HERE/'inputs/mechanism.json').read_text())
    benchmark = json.loads((HERE/'benchmark.json').read_text())
    parameters = model['parameters']
    times = np.asarray(config['sample_times_h'], dtype=float)  # [h]
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or times[0] < 0. or times[-1] <= 0. or np.any(np.diff(times) <= 0.)
            or times[-1] > model['operation']['runtime']['value']):
        raise ValueError('sample_times_h must increase within the model runtime')
    if (not isinstance(config['realizations'], int) or config['realizations'] < 1
            or not np.isfinite(config['absolute_sd_g_l']) or config['absolute_sd_g_l'] < 0.
            or any(not np.isfinite(value) or value < 0.
                   for value in config['noise_fractions'].values())):
        raise ValueError('realizations and noise settings must be finite and valid')
    names = [entry['name'] for entry in config['experiments']]
    if any(not isinstance(name, str) or not name or name in {'.', '..'}
           or '/' in name or '\\' in name
           for name in [*names, *config['noise_fractions']]):
        raise ValueError('experiment and noise names must be plain file identifiers')
    if (len(set(names)) != len(names) or not names
            or any(entry['split'] not in {'fit', 'validation'}
                   or not np.isfinite(entry['dilution_h_inv']) or entry['dilution_h_inv'] < 0.
                   for entry in config['experiments'])):
        raise ValueError('experiments need unique identities, valid splits and dilution rates')
    feed = model['recipes']['continuous'][0]['concentrations_g_l']['substrate']  # [g/L]
    initial = np.array([4.5, 1., 0.])  # [g/L]; Exercise 4.15 nominal state
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth_rows, observations = [], {label: [] for label in config['noise_fractions']}
    maximum_disagreement = 0.  # [g/L]
    for experiment_index, experiment in enumerate(config['experiments']):
        solutions = []
        for method in ('DOP853', 'Radau'):
            solution = solve_ivp(reference_rates, (0., times[-1]), initial,
                args=(experiment['dilution_h_inv'], parameters, feed, truth_theta,
                      hybrid['model']['kinetics']['constants']['feature_scale_g_l']),
                method=method, t_eval=times, **{
                    'rtol': benchmark['reference_numerics']['rtol'],
                    'atol': benchmark['reference_numerics']['atol'],
                    'max_step': benchmark['reference_numerics']['max_step_h']})
            if not solution.success:
                raise RuntimeError(solution.message)
            solutions.append(solution.y[:2].T)
        truth = solutions[0]  # [g/L]; biomass then substrate
        disagreement = float(np.max(np.abs(truth-solutions[1])))  # [g/L]
        maximum_disagreement = max(maximum_disagreement, disagreement)
        if disagreement > benchmark['acceptance']['reference_agreement_g_l']:
            raise RuntimeError('Independent reference solvers disagree')
        np.testing.assert_allclose(truth[:, 1]+truth[:, 0]/parameters['yields']['biomass'],
                                   feed, rtol=0., atol=benchmark['acceptance']['equivalent_substrate_g_l'])
        for index, time_h in enumerate(times):
            for column, variable in enumerate(('biomass', 'substrate')):
                truth_rows.append(dict(experiment=experiment['name'], split=experiment['split'],
                    time_h=time_h, variable=variable, value_g_l=truth[index, column]))
        for label, fraction in config['noise_fractions'].items():
            count = 1 if fraction == 0 else config['realizations']
            for realization in range(count):
                # Pair standardized draws across noise levels for controlled comparisons.
                seed = np.random.SeedSequence([config['seed'], experiment_index, realization])
                draws = np.random.Generator(np.random.PCG64(seed)).standard_normal(truth.shape)
                absolute = config['absolute_sd_g_l'] if fraction else 0.  # [g/L]
                sigma = np.sqrt(absolute**2+(fraction*truth)**2)  # [g/L]
                measured = truth+sigma*draws  # [g/L]; no clipping/rescaling
                for index, time_h in enumerate(times):
                    for column, variable in enumerate(('biomass', 'substrate')):
                        observations[label].append(dict(experiment=experiment['name'],
                            split=experiment['split'], realization=realization,
                            time_h=time_h, variable=variable,
                            value_g_l=measured[index, column], sigma_g_l=sigma[index, column]))
    write_rows(output/'truth.csv', list(truth_rows[0]), truth_rows)
    for label, rows in observations.items():
        write_rows(output/f'{label}.csv', list(rows[0]), rows)
        assert all(np.isfinite(row['value_g_l']) for row in rows)
        assert len(rows) == len(truth_rows)*(1 if config['noise_fractions'][label] == 0 else config['realizations'])
    assert [row['value_g_l'] for row in observations['noise_0pct']] == [row['value_g_l'] for row in truth_rows]
    report = dict(parameters=parameters, truth_theta=truth_theta,
        feature_scale_g_l=hybrid['model']['kinetics']['constants']['feature_scale_g_l'],
        initial_g_l=initial.tolist(), feed_g_l=feed,
        design=config, numpy_version=np.__version__, maximum_reference_disagreement_g_l=maximum_disagreement,
        rows={label: len(rows) for label, rows in observations.items()},
        negative_observations={label: int(sum(row['value_g_l'] < 0 for row in rows)) for label, rows in observations.items()},
        source_sha256={str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), HERE/'inputs/synthetic_design.json',
                         HERE/'inputs/mechanism.json', HERE/'inputs/hybrid.json',
                         HERE/'run_benchmarks.py', HERE/'benchmark.json']},
        sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(output.glob('*.csv'))})
    (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main() -> None:
    """Generate CSV assets; CLI options select output and synthetic theta [-]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'outputs/synthetic_data')
    parser.add_argument('--theta', type=float, default=0.)
    arguments = parser.parse_args()
    report = generate(arguments.output, truth_theta=arguments.theta)
    print(json.dumps({key: report[key] for key in ('rows', 'maximum_reference_disagreement_g_l', 'negative_observations')}, indent=2))


if __name__ == '__main__':
    main()
