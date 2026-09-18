"""Regenerate presentation comparisons from current native simulations."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PharmaPy.Bioreactors import build_bioreactor

HERE = Path(__file__).resolve().parent


def forward_simulation(example, check_regression=True):
    """Execute the notebook's five forward steps without inference or file writes."""
    notebook = Path(example) / 'workflow.ipynb'
    cells = [cell for cell in json.loads(notebook.read_text())['cells']
             if cell['cell_type'] == 'code']
    namespace = {'__name__': '__notebook__'}
    for index, cell in enumerate(cells[:5]):
        source = ''.join(cell['source'])
        exec(compile(source, f'{notebook}:cell-{index}', 'exec'), namespace)
        if index == 0:
            namespace['COMPARE_CANONICAL'] = check_regression
    return namespace['simulation']


def _finish(figure, folder, stem):
    for extension in ('png', 'svg'):
        figure.savefig(folder / f'{stem}.{extension}', dpi=200)
    plt.close(figure)


def _ecoli(rows, folder):
    inputs = HERE / 'batch_ecoli_dfba/inputs'
    case = json.loads((inputs / 'case.json').read_text())
    definition = json.loads((inputs / 'mechanism.json').read_text())
    model = build_bioreactor(case, definition, inputs / 'thermo.json').mechanism.model
    # mmol/L equals mol/m3; gDW/L equals kgDW/m3 numerically.
    fluxes = np.array([model.solve_fluxes(
        {name: row[f'{name}_mmol_l'] for name in model.state_names},
        row['biomass_gdw_l']).fluxes for row in rows])
    time = [row['time_h'] for row in rows]
    figure = plt.figure(figsize=(12, 8), layout='constrained')
    grid = figure.add_gridspec(2, 3)
    axes = [figure.add_subplot(grid[0, j]) for j in range(3)]
    flux_axis = figure.add_subplot(grid[1, :])
    axes[0].plot(time, [r['glucose_mmol_l'] for r in rows], color='tab:blue')
    axes[0].set(title='Glucose', ylabel='mmol/L')
    axes[1].plot(time, [r['acetate_mmol_l'] for r in rows], color='tab:orange', label='acetate')
    axes[1].set(title='Acetate and biomass', ylabel='Acetate (mmol/L)')
    biomass_axis = axes[1].twinx()
    biomass_axis.plot(time, [r['biomass_gdw_l'] for r in rows], color='tab:red', label='biomass')
    biomass_axis.set_ylabel('Biomass (gDW/L)')
    axes[1].legend(loc='upper left')
    biomass_axis.legend(loc='lower right')
    axes[2].plot(time, [r['oxygen_mmol_l'] for r in rows], color='tab:green')
    axes[2].set(title='Oxygen', ylabel='mmol/L')
    for j, name in enumerate(model.pathway_ids):
        flux_axis.plot(time, fluxes[:, j], label=name)
    flux_axis.set(title='Pathway activities at reported states', ylabel='Pathway activity (1/h)')
    flux_axis.legend()
    for axis in [*axes, flux_axis]:
        axis.set_xlabel('Time (h)')
        axis.grid(alpha=0.2)
    figure.suptitle('Current native E. coli model')
    _finish(figure, folder, 'states_and_fluxes')
    return [dict(row, **{name: float(fluxes[i, j]) for j, name in enumerate(model.pathway_ids)})
            for i, row in enumerate(rows)], 'states_and_fluxes'


def _cho(rows, folder):
    time = [row['time_day'] for row in rows]
    series = [
        ('Viable cell density', 'million cells/mL',
         [r['viable_cells_million'] / (1000 * r['liquid_volume_l']) for r in rows]),
        ('Glutamine', 'mmol/L', [r['GLN'] for r in rows]),
        ('Asparagine', 'mmol/L', [r['ASN'] for r in rows]),
        ('Glucose', 'mmol/L', [r['GLC'] for r in rows]),
        ('Monoclonal antibody', 'g/L', [r['product_g'] / r['liquid_volume_l'] for r in rows]),
        ('Glutamate', 'mmol/L', [r['GLU'] for r in rows]),
        ('Aspartate', 'mmol/L', [r['ASP'] for r in rows]),
        ('Lactate', 'mmol/L', [r['LAC'] for r in rows]),
        ('Ammonia', 'mmol/L', [r['NH3'] for r in rows]),
        ('Methionine', 'mmol/L', [r['MET'] for r in rows]),
        ('Serine', 'mmol/L', [r['SER'] for r in rows]),
        ('Alanine', 'mmol/L', [r['ALA'] for r in rows]),
    ]
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, layout='constrained')
    for axis, (title, unit, values) in zip(axes.flat, series):
        axis.plot(time, values, color='tab:blue')
        axis.set(title=title, ylabel=unit, xlabel='Time (day)')
        axis.grid(alpha=0.2)
    figure.suptitle('Current native CHO model at pH 7.0\nCorrected inventory-constrained reconstruction')
    _finish(figure, folder, 'process_trajectories')
    return rows, 'process_trajectories'


def main():
    """Run existing simulations without replacing their baseline artifacts."""
    for name, plot in [('batch_ecoli_dfba', _ecoli), ('fed_batch_cho', _cho)]:
        result = forward_simulation(HERE / name)
        if result['summary']['status'] != 'PASS':
            raise RuntimeError(f'{name}: model regression failed')
        example = HERE / name
        folder = example / 'outputs/figures'
        folder.mkdir(parents=True, exist_ok=True)
        rows, stem = plot(result['rows'], folder)
        with (folder / 'model_data.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        reference = example / 'documentation/published_figures'
        provenance = json.loads((reference / 'provenance.json').read_text())
        figure, axes = plt.subplots(1, 2, figsize=(18, 9), layout='constrained')
        for axis, path, title in zip(axes, [folder / f'{stem}.png', reference / provenance['image']],
                ['Current-model simulation', 'Published panel from supplied presentation']):
            axis.imshow(plt.imread(path))
            axis.set_title(title)
            axis.axis('off')
        figure.suptitle(f"Reference DOI: {provenance['paper_doi']}\n"
                       'Different model/data conditions; qualitative comparison, not experimental validation')
        _finish(figure, folder, 'paper_comparison')
        sources = [Path(__file__), *sorted((example / 'inputs').glob('*.json')),
                   reference / provenance['image'], reference / 'provenance.json']
        (folder / 'generation.json').write_text(json.dumps({
            'command': 'PYTHONPATH=. python examples/bioreactors/reproduce_figures.py',
            'model_regression': result['summary'],
            'source_sha256': {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            'flux_sampling': 'E. coli LP re-evaluated at reported states; not internal BDF stage history',
            'comparison': 'Presentation reference panels; no digitized experimental data or curve fitting',
        }, indent=2) + '\n')
        print(f'Figures saved: {folder}')


if __name__ == '__main__':
    main()
