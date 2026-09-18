"""Create an isolated environment and execute the source-checkout demonstrator."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, help='new environment directory outside the checkout')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Use Python 3.11; the qualified interpreter is 3.11.15.')
    if args.environment is None:
        directory = Path(tempfile.mkdtemp(prefix='pharmapy-bioreactor-env-'))
    else:
        directory = args.environment.resolve()
        if directory.exists():
            raise SystemExit('Environment must be new; existing directories are not modified.')
    if directory == ROOT or ROOT in directory.parents:
        raise SystemExit('Keep the environment outside the source checkout.')
    clean = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
        clean.pop(name, None)
    clean.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                 MPLCONFIGDIR=str(directory/'matplotlib'))
    print('Creating isolated environment:', directory, flush=True)
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(directory)
    python = directory/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    subprocess.run([str(python), '-I', '-m', 'pip', 'install', '-r',
                    str(ROOT/'examples/bioreactors/requirements-py311.txt')],
                   cwd=ROOT, env=clean, check=True)
    subprocess.run([str(python), '-I', '-m', 'pip', 'check'], cwd=ROOT, env=clean, check=True)
    subprocess.run([str(python), '-I', str(ROOT/'examples/bioreactors/verify_examples.py')],
                   cwd=ROOT, env=clean, check=True)
    print('Verified environment retained at:', directory, flush=True)


if __name__ == '__main__':
    main()
