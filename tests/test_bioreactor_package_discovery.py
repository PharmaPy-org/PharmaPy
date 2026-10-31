"""Bioreactor subpackages must be discoverable in source and wheel installs."""

import importlib
from pathlib import Path

import PharmaPy
import pytest


@pytest.mark.parametrize('module_name', [
    'PharmaPy.Bioreactors',
    'PharmaPy.Metabolic.pathways',
    'PharmaPy.Metabolic.closures.reconciled',
])
def test_bioreactor_subpackages_ship_with_the_native_package(module_name):
    """Resolve a biological module within the selected PharmaPy installation.

    Parameters
    ----------
    module_name : str
        Public builder or metabolic implementation module to import.
    """
    module = importlib.import_module(module_name)
    assert Path(module.__file__).is_relative_to(Path(PharmaPy.__file__).parent)
