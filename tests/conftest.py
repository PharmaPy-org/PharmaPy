from pathlib import Path
import sys

import pytest


# Minimal two-species thermo file. Synthetic values separate molecular weights
# and keep dew-point temperatures subcritical at the tested 1.0e5 Pa.
# Antoine form: log10(P/[Pa]) = A - B/(T + C), with T [K].
THERMO_TWO_SPECIES = {
    "light": {
        "mw": 18.0,  # [g/mol]
        "t_crit": 650.0,  # [K]
        "rho_liq": 1000.0,  # [kg/m**3]
        "cp_liq": [75.0],  # [J/mol/K]
        "p_vap": [8.0, 1500.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 40000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "heavy": {
        "mw": 100.0,  # [g/mol]
        "t_crit": 700.0,  # [K]
        "rho_liq": 900.0,  # [kg/m**3]
        "cp_liq": [150.0],  # [J/mol/K]
        "p_vap": [8.0, 1800.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 60000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
}


TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _has_assimulo():
    try:
        import assimulo  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="session")
def data_path():
    return {
        "integration": TESTS_ROOT / "integration" / "data",
        "flowsheet": TESTS_ROOT / "Flowsheet" / "data",
    }


def pytest_collection_modifyitems(config, items):
    if _has_assimulo():
        return

    skip_assimulo = pytest.mark.skip(
        reason="assimulo is not installed; solver-backed integration tests skipped"
    )
    for item in items:
        if "assimulo" in item.keywords:
            item.add_marker(skip_assimulo)
