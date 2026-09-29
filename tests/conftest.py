import json
from pathlib import Path
import sys

import numpy as np
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

# Synthetic construction data for real drying collaborators. The values are
# explicit test assumptions, not measured properties or validated correlations.
# Constant heat capacities and viscosities keep the expected balances
# independently calculable while the production thermophysical APIs remain in
# the path under test.
DRYING_THERMO_DATA = {
    "water": {
        "mw": 18.0,  # [g/mol], rounded test value
        "t_crit": 650.0,  # [K], rounded test value
        "rho_liq": 1000.0,  # [kg/m**3], rounded test value
        "rho_solid": 1000.0,  # [kg/m**3], construction-only fallback
        "cp_liq": [75.0],  # [J/mol/K], rounded constant test value
        "cp_vapor": [35.0],  # [J/mol/K], synthetic constant test value
        "cp_solid": [700.0],  # [J/kg/K], construction-only fallback
        "visc_liq": [0.0, 0.0, 0.0, 0.0],  # log-correlation coefficients [-]
        "visc_gas": 1.8e-5,  # [Pa*s], constant test value
        "p_vap": [8.0, 1500.0, -40.0],  # [-], [K], [K], synthetic coefficients
        "delta_hvap": 40000.0,  # [J/mol], rounded test value
        "tref_hvap": 350.0,  # [K], test correlation reference
        "surf_tension": 0.072,  # [N/m], rounded test value
    },
    "ethanol": {
        "mw": 46.0,  # [g/mol], rounded test value
        "t_crit": 700.0,  # [K], chosen to keep the test species subcritical
        "rho_liq": 800.0,  # [kg/m**3], rounded test value
        "rho_solid": 900.0,  # [kg/m**3], construction-only fallback
        "cp_liq": [110.0],  # [J/mol/K], rounded constant test value
        "cp_vapor": [60.0],  # [J/mol/K], synthetic constant test value
        "cp_solid": [700.0],  # [J/kg/K], construction-only fallback
        "visc_liq": [0.0, 0.0, 0.0, 0.0],  # log-correlation coefficients [-]
        "visc_gas": 1.2e-5,  # [Pa*s], constant test value
        "p_vap": [8.0, 1800.0, -40.0],  # [-], [K], [K], synthetic coefficients
        "delta_hvap": 50000.0,  # [J/mol], rounded test value
        "tref_hvap": 350.0,  # [K], test correlation reference
        "surf_tension": 0.022,  # [N/m], rounded test value
    },
    "nitrogen": {
        "mw": 28.0,  # [g/mol]
        "t_crit": 250.0,  # [K], always supercritical in these cases
        "rho_liq": 850.0,  # [kg/m**3], construction-only fallback
        "rho_solid": 1000.0,  # [kg/m**3], construction-only fallback
        "cp_liq": [29.0],  # [J/mol/K]
        "cp_vapor": [29.0],  # [J/mol/K]
        "cp_solid": [700.0],  # [J/kg/K]
        "visc_liq": [0.0, 0.0, 0.0, 0.0],  # log-correlation coefficients [-]
        "visc_gas": 1.7e-5,  # [Pa*s]
        "p_vap": [8.0, 1200.0, -40.0],  # [-], [K], [K]
        "delta_hvap": 0.0,  # [J/mol], non-condensable test carrier
        "tref_hvap": 200.0,  # [K]
        "surf_tension": 0.0,  # [N/m], non-condensable test carrier
    },
}

DRYING_SIZE_GRID_UM = np.array([50.0, 100.0, 150.0])  # [um]
DRYING_CSD_NUMBER = np.array([1.0, 2.0, 1.0])  # [#/um], total basis
DRYING_LIQUID_MASS_FRACTION = np.array([0.45, 0.45, 0.10])  # [-]
DRYING_GAS_MASS_FRACTION = np.array([1e-6, 1e-6, 0.999998])  # [-]

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


@pytest.fixture(scope="session")
def drying_thermo_path(tmp_path_factory):
    """Write the documented synthetic drying-property database.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-scoped temporary-directory provider.

    Returns
    -------
    pathlib.Path
        JSON database path consumed by real PharmaPy phase classes.
    """
    thermo_path = tmp_path_factory.mktemp("drying") / "thermo.json"
    thermo_path.write_text(json.dumps(DRYING_THERMO_DATA))
    return thermo_path


@pytest.fixture
def drying_cake_factory(drying_thermo_path):
    """Provide a factory for real liquid-solid cake collaborators.

    Parameters
    ----------
    drying_thermo_path : pathlib.Path
        Synthetic thermophysical-property database.

    Returns
    -------
    callable
        Factory accepting size grid [um], number distribution [#/um],
        saturation [-], and condensed temperature [K].
    """
    from PharmaPy.MixedPhases import Cake
    from PharmaPy.Phases import LiquidPhase, SolidPhase

    def make_cake(
            size_grid_um=DRYING_SIZE_GRID_UM,
            csd_number=DRYING_CSD_NUMBER,
            saturation=0.55,
            temperature=302.0):
        """Construct a real ``Cake`` containing real PharmaPy phases.

        Parameters
        ----------
        size_grid_um : array_like, optional
            Particle-size grid [um].
        csd_number : array_like, optional
            Total-population number distribution [#/um].
        saturation : float, optional
            Initial liquid saturation [-].
        temperature : float, optional
            Common condensed-phase temperature [K].

        Returns
        -------
        Cake
            Packed cake with real liquid and solid phases.
        """
        liquid = LiquidPhase(
            str(drying_thermo_path),
            temp=temperature,
            mass=1.0e-3,  # [kg]
            mass_frac=DRYING_LIQUID_MASS_FRACTION,
        )
        solid = SolidPhase(
            str(drying_thermo_path),
            temp=temperature,
            x_distrib=np.asarray(size_grid_um),  # [um]
            distrib=np.asarray(csd_number),  # [#/um], total basis
            mass_frac=np.array([0.0, 1.0, 0.0]),  # [-], solid ethanol
        )
        cake = Cake(
            z_external=np.array([0.0, 1.0]),  # [m], rescaled by owning unit
            saturation=np.atleast_1d(saturation),  # [-]
        )
        cake.Phases = [liquid, solid]
        return cake

    return make_cake


@pytest.fixture
def drying_unit_factory(drying_thermo_path, drying_cake_factory):
    """Provide fully configured real drying units.

    Parameters
    ----------
    drying_thermo_path : pathlib.Path
        Synthetic thermophysical-property database.
    drying_cake_factory : callable
        Real cake collaborator factory.

    Returns
    -------
    callable
        Factory accepting node count [-], size grid [um], number distribution
        [#/um], saturation [-], and phase temperatures [K].
    """
    from PharmaPy.Drying_Model import Drying
    from PharmaPy.Phases import VaporPhase
    from PharmaPy.Streams import VaporStream

    def make_dryer(
            number_nodes=3,
            size_grid_um=DRYING_SIZE_GRID_UM,
            csd_number=DRYING_CSD_NUMBER,
            saturation=0.55,
            condensed_temperature=302.0,
            gas_temperature=300.0):
        """Construct a ``Drying`` unit with real phases, cake, and inlet.

        Parameters
        ----------
        number_nodes : int, optional
            Axial discretization node count [-].
        size_grid_um : array_like, optional
            Particle-size grid [um].
        csd_number : array_like, optional
            Total-population number distribution [#/um].
        saturation : float, optional
            Initial cake saturation [-].
        condensed_temperature : float, optional
            Initial liquid-solid temperature [K].
        gas_temperature : float, optional
            Initial and inlet gas temperature [K].

        Returns
        -------
        Drying
            Configured production drying model.
        """
        cake = drying_cake_factory(
            size_grid_um=size_grid_um,
            csd_number=csd_number,
            saturation=saturation,
            temperature=condensed_temperature,
        )
        gas_phase = VaporPhase(
            str(drying_thermo_path),
            temp=gas_temperature,
            mass=1.0e-4,  # [kg]
            mass_frac=DRYING_GAS_MASS_FRACTION,
        )
        gas_inlet = VaporStream(
            str(drying_thermo_path),
            temp=gas_temperature,
            mass_flow=1.0e-4,  # [kg/s]
            mass_frac=DRYING_GAS_MASS_FRACTION,
        )

        dryer = Drying(number_nodes, supercrit_names=["nitrogen"])
        dryer.Phases = cake
        dryer.CakePhase.z_external = np.array(
            [0.0, dryer.cake_height]
        )  # [m]
        dryer.Phases = gas_phase
        dryer.Inlet = gas_inlet
        return dryer

    return make_dryer


def pytest_collection_modifyitems(config, items):
    if _has_assimulo():
        return

    skip_assimulo = pytest.mark.skip(
        reason="assimulo is not installed; solver-backed integration tests skipped"
    )
    for item in items:
        if "assimulo" in item.keywords:
            item.add_marker(skip_assimulo)
