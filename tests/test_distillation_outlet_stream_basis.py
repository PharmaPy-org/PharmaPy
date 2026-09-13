"""Steady-state distillation outlet-stream composition basis.

``DistillationColumn.retrieve_results`` builds the distillate and bottoms
``LiquidStream`` objects from the shortcut material balance. Those balance
values are mole fractions [-], so they must enter the stream on a fraction
basis; passing them on a molar-concentration basis leaves the outlets carrying
fractions labelled [mol/L]. The self-contained two-species fixture needs no
repository data: molar densities are round numbers so the expected outlet
concentrations follow from the ideal-mixing molar volume by hand.
"""

import json

import numpy as np
import pytest

from PharmaPy.Distillation import DistillationColumn
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


# Minimal two-species thermo file. Molecular weights and liquid densities are
# deliberately unequal, and chosen so that each pure molar density
# rho_liq/mw is exact: 800/32 = 25 mol/L and 1250/100 = 12.5 mol/L. Antoine
# form: log10(P/[Pa]) = A - B/(T + C), with T [K].
THERMO_TWO_SPECIES = {
    "light": {
        "mw": 32.0,  # [g/mol]
        "t_crit": 650.0,  # [K]
        "rho_liq": 800.0,  # [kg/m**3]
        "cp_liq": [75.0],  # [J/mol/K]
        "p_vap": [8.0, 1500.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 40000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "heavy": {
        "mw": 100.0,  # [g/mol]
        "t_crit": 700.0,  # [K]
        "rho_liq": 1250.0,  # [kg/m**3]
        "cp_liq": [150.0],  # [J/mol/K]
        "p_vap": [8.0, 1800.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 60000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
}

_SPECIES = ("light", "heavy")
MW = np.array(
    [THERMO_TWO_SPECIES[name]["mw"] for name in _SPECIES]
)  # [g/mol]
RHO_MASS = np.array(
    [THERMO_TWO_SPECIES[name]["rho_liq"] for name in _SPECIES]
)  # [kg/m**3]
RHO_MOLE = RHO_MASS / MW  # [mol/L], equals [25.0, 12.5]

COLUMN_PRESSURE = 101325.0  # [Pa]
FEED_TEMPERATURE = 350.0  # [K]
FEED_MOLE_FRAC = np.array([0.5, 0.5])  # [-]
FEED_MOLE_FLOW = 3.0  # [mol/s]

# Deterministic shortcut-design outcome for a sharp binary split. The values are
# a self-consistent global material balance for the feed above and are supplied
# directly so the test exercises result retrieval, not the shortcut numerics.
X_DIST = np.array([0.90, 0.10])  # [-]
X_BOT = np.array([0.05, 0.95])  # [-]
DIST_MOLE_FLOW = 1.0  # [mol/s]
BOT_MOLE_FLOW = 2.0  # [mol/s]
PLATE_TEMPERATURES = np.array([340.0, 355.0, 370.0])  # [K], top to bottom


def _ideal_mixing_mole_conc(mole_frac):
    """Molar concentrations of an ideal liquid mixture.

    Parameters
    ----------
    mole_frac : ndarray
        Component mole fractions [-].

    Returns
    -------
    ndarray
        Component molar concentrations [mol/L].

    Notes
    -----
    Ideal mixing gives the mixture molar volume as
    ``v_mix = sum(x_i / rho_mole_i)`` [L/mol], so each component concentration
    is ``x_i / v_mix``. This is the defining relation, evaluated here from the
    fixture densities rather than from the production conversion helper.
    """
    molar_volume = np.dot(mole_frac, 1 / RHO_MOLE)  # [L/mol]

    return mole_frac / molar_volume  # [mol/L]


def _steady_column(thermo_path):
    """Build a steady-state column with a real feed stream.

    Parameters
    ----------
    thermo_path : str
        Path to the two-species thermo database file.

    Returns
    -------
    DistillationColumn
        Column whose ``Inlet`` is a real ``LiquidStream`` feed at
        ``COLUMN_PRESSURE`` [Pa] and ``FEED_TEMPERATURE`` [K].
    """
    column = DistillationColumn(
        pres=COLUMN_PRESSURE,  # [Pa]
        q_feed=1.0,  # [-], saturated-liquid feed
        LK="light",
        HK="heavy",
        perc_LK=90.0,  # [%]
        perc_HK=10.0,  # [%]
    )
    column.Inlet = LiquidStream(
        thermo_path,
        temp=FEED_TEMPERATURE,  # [K]
        pres=COLUMN_PRESSURE,  # [Pa]
        mole_frac=FEED_MOLE_FRAC,  # [-]
        mole_flow=FEED_MOLE_FLOW,  # [mol/s]
    )

    return column


def _retrieve(column):
    """Run result retrieval with the deterministic shortcut-design outcome.

    Parameters
    ----------
    column : DistillationColumn
        Column with an assigned feed stream.

    Returns
    -------
    None
        The column's ``result`` and outlet streams are populated in place.
    """
    plate_liquid = np.vstack([X_DIST, FEED_MOLE_FRAC, X_BOT])  # [-]
    plate_vapor = np.vstack([X_DIST, FEED_MOLE_FRAC, X_BOT])  # [-]

    column.retrieve_results(
        num_plates=len(PLATE_TEMPERATURES),  # [-]
        x=plate_liquid,  # [-]
        y=plate_vapor,  # [-]
        T=PLATE_TEMPERATURES,  # [K]
        bot_flowrate=BOT_MOLE_FLOW,  # [mol/s]
        dist_flowrate=DIST_MOLE_FLOW,  # [mol/s]
        reflux=1.8,  # [-]
        num_feed=2,  # [-]
        x_dist=X_DIST,  # [-]
        x_bot=X_BOT,  # [-]
        min_reflux=1.2,  # [-]
        N_min=4.0,  # [-]
    )


def test_steady_outlets_use_fraction_basis_for_balance_composition(tmp_path):
    """Outlet concentrations follow from the balance fractions, not equal them.

    The shortcut balance reports mole fractions, so each outlet's ``mole_conc``
    must be the ideal-mixing concentration at that composition and its
    ``mass_conc`` the molecular-weight-scaled counterpart. Reading the fractions
    as concentrations instead leaves ``mole_conc`` summing to 1.0 [mol/L], which
    no liquid mixture of these species can have.
    """
    thermo_path = tmp_path / "thermo_two_species.json"
    thermo_path.write_text(json.dumps(THERMO_TWO_SPECIES))

    column = _steady_column(str(thermo_path))
    _retrieve(column)

    expected_dist_conc = _ideal_mixing_mole_conc(X_DIST)  # [mol/L]
    expected_bot_conc = _ideal_mixing_mole_conc(X_BOT)  # [mol/L]

    np.testing.assert_allclose(
        column.OutletDistillate.mole_conc, expected_dist_conc, rtol=1e-12)
    np.testing.assert_allclose(
        column.OutletBottom.mole_conc, expected_bot_conc, rtol=1e-12)

    np.testing.assert_allclose(
        column.OutletDistillate.mass_conc, expected_dist_conc * MW, rtol=1e-12)
    np.testing.assert_allclose(
        column.OutletBottom.mass_conc, expected_bot_conc * MW, rtol=1e-12)

    # Under ideal volume mixing, mixture molar density is bounded by the pure
    # species values. A unit sum therefore identifies a fraction vector in
    # concentration clothing.
    for outlet in (column.OutletDistillate, column.OutletBottom):
        total_conc = outlet.mole_conc.sum()  # [mol/L]
        assert np.min(RHO_MOLE) <= total_conc <= np.max(RHO_MOLE)


def test_steady_outlets_preserve_balance_fractions_and_flows(tmp_path):
    """Retrieval carries each balance stream to its own outlet unchanged.

    Distillate and bottoms differ in composition, molar flow, and terminal plate
    temperature, so this also detects a swapped assignment. The asymmetric
    fixture keeps the check meaningful: mole and mass fractions differ from each
    other because the molecular weights are unequal.
    """
    thermo_path = tmp_path / "thermo_two_species.json"
    thermo_path.write_text(json.dumps(THERMO_TWO_SPECIES))

    column = _steady_column(str(thermo_path))
    _retrieve(column)

    expected_dist_mass_frac = X_DIST * MW / np.dot(X_DIST, MW)  # [-]
    expected_bot_mass_frac = X_BOT * MW / np.dot(X_BOT, MW)  # [-]

    np.testing.assert_allclose(
        column.OutletDistillate.mole_frac, X_DIST, rtol=1e-12)
    np.testing.assert_allclose(
        column.OutletBottom.mole_frac, X_BOT, rtol=1e-12)

    np.testing.assert_allclose(
        column.OutletDistillate.mass_frac, expected_dist_mass_frac, rtol=1e-12)
    np.testing.assert_allclose(
        column.OutletBottom.mass_frac, expected_bot_mass_frac, rtol=1e-12)

    assert column.OutletDistillate.mole_flow == pytest.approx(DIST_MOLE_FLOW)
    assert column.OutletBottom.mole_flow == pytest.approx(BOT_MOLE_FLOW)

    assert column.OutletDistillate.temp == pytest.approx(
        PLATE_TEMPERATURES[0])
    assert column.OutletBottom.temp == pytest.approx(PLATE_TEMPERATURES[-1])

    # The molar mass balance must close across the two outlets.
    total_moles = (
        column.OutletDistillate.mole_flow * column.OutletDistillate.mole_frac
        + column.OutletBottom.mole_flow * column.OutletBottom.mole_frac
    )  # [mol/s]
    np.testing.assert_allclose(
        total_moles.sum(), DIST_MOLE_FLOW + BOT_MOLE_FLOW, rtol=1e-12)

    assert column.Outlet is column.OutletBottom
