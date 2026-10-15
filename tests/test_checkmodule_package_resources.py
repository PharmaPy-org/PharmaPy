"""Modeling-object validation without a repository-root data directory.

The subprocess imports a copied package in isolation, reproducing the installed
layout without requiring a build backend in the minimal core test environment.
Wheel inclusion is additionally verified by the distribution build check.
"""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_modeling_object_checks_work_without_repository_data(tmp_path):
    package = Path(__file__).resolve().parents[1] / 'PharmaPy'
    shutil.copytree(package, tmp_path / 'PharmaPy',
                    ignore=shutil.ignore_patterns('__pycache__'))
    script = '''
import pathlib
import sys
import warnings
sys.path.insert(0, sys.argv[1])
from PharmaPy.CheckModule import check_modeling_objects
import PharmaPy.CheckModule as checks
assert pathlib.Path(checks.__file__).is_relative_to(pathlib.Path(sys.argv[1]))

class MinimalBatch:
    oper_mode = 'Batch'
    Phases = object()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    check_modeling_objects(MinimalBatch(), 'B01')
    assert not caught
    unit = MinimalBatch()
    unit.Phases = None
    check_modeling_objects(unit, 'B01')
    assert len(caught) == 1
    assert "'B01' MinimalBatch" in str(caught[0].message)
    assert 'Phases' in str(caught[0].message)
print('installed-layout checks passed')
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'installed-layout checks passed'
