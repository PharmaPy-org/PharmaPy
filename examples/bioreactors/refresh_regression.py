"""Explicitly regenerate model baselines; never called by normal acceptance runs."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from examples.bioreactors.reproduce_figures import forward_simulation
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = hashlib.sha256()
    for path in sorted((ROOT/'PharmaPy').rglob('*.py')):
        source.update(path.relative_to(ROOT).as_posix().encode() + b'\0' + path.read_bytes())
    for name, metrics, previous in (
        ('batch_ecoli_dfba', ['final_biomass_gdw_l'],
         {'final_biomass_gdw_l': 0.8984192534784751}),
        ('fed_batch_cho', ['final_vcd_million_ml', 'final_product_g'],
         {'final_vcd_million_ml': 6.380513082505868, 'final_product_g': 1.65453652125779}),
    ):
        folder = ROOT/'examples/bioreactors'/name
        result = forward_simulation(folder, check_regression=False)
        baseline = {
            'kind': 'generated_model_regression', 'independent_experimental_validation': False,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'command': 'python examples/bioreactors/refresh_regression.py',
            'relative_tolerance': 0.02,
            'tolerance_scope': 'unchanged project regression threshold, not an industry accuracy standard',
            'endpoints': {key: result['summary'][key] for key in metrics},
            'inputs_sha256': {p: hashlib.sha256((folder/'inputs'/p).read_bytes()).hexdigest()
                              for p in ('case.json', 'mechanism.json', 'thermo.json')},
            'source_tree_python_sha256': source.hexdigest(),
            'source_digest_rule': 'sorted PharmaPy/**/*.py: relative POSIX path, NUL, then file bytes',
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'python': platform.python_version(),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'matplotlib', 'pandas', 'autograd')},
            'superseded_untraced_constants': previous,
            'historical_note': 'Old constants had no established generation lineage; preserved here, not represented as measurements.',
        }
        (folder/'inputs/regression_baseline.json').write_text(json.dumps(baseline, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
