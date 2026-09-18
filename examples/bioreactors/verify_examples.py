"""Validate the library and directly executable workflow notebooks."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# Run this checkout's native package; dependencies must come from the isolated environment.
sys.path.insert(0, str(ROOT))


def main():
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Run inside the isolated demonstrator environment.')
    os.chdir(ROOT)
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'isolated_environment': sys.prefix != sys.base_prefix,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'requirements_sha256': hashlib.sha256((ROOT/'examples/bioreactors/requirements-py311.txt').read_bytes()).hexdigest(),
        'scope': 'Library and direct workflow notebooks; optional Assimulo/IDA not qualified here',
    }
    tests = subprocess.run([sys.executable, '-I', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                            '-o', 'pythonpath=.', 'tests/Bioreactor'], cwd=ROOT)
    report['pytest_exit_code'] = tests.returncode
    report['status'] = 'PASS' if tests.returncode == 0 else 'FAIL'
    (ROOT/'examples/bioreactors/installation_report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    if report['status'] != 'PASS':
        raise SystemExit('Demonstrator acceptance failed; see installation_report.json')


if __name__ == '__main__':
    main()
