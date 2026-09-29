"""Solvent-index handling when the solvent is the first species (index 0).

`ThermoPhysicalManager.conc_to_frac`, `ThermoPhysicalManager.mass_conc_to_frac`
and `LiquidPhase.updatePhase` select their return arity from the solvent index.
A truthiness test cannot distinguish the valid index ``0`` from an absent
solvent, so these tests pin the behavior for a solvent declared as the first
species and check it against the same mixture with the species order permuted.
`LiquidStream` is exercised as well, because it is the entry point the
repository's flowsheets use and it reaches the converters through a positional
delegation to `LiquidPhase.updatePhase`.

The self-contained three-species fixture needs no repository data. Molecular
masses and liquid densities are deliberately distinct so that an index or axis
mistake changes the numbers rather than cancelling out.
"""

import json

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


# Synthetic three-species liquid. Values are chosen only to keep the molar
# volumes (mw / rho_liq) well separated; they do not model a real system.
# Antoine form: log10(P/[Pa]) = A - B/(T + C), with T [K].
_SPECIES = {
    "water": {
        "mw": 18.0,  # [g/mol]
        "rho_liq": 1000.0,  # [kg/m**3]
        "t_crit": 647.0,  # [K]
        "cp_liq": [75.0],  # [J/mol/K]
        "p_vap": [8.0, 1500.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 40000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "api": {
        "mw": 150.0,  # [g/mol]
        "rho_liq": 1200.0,  # [kg/m**3]
        "t_crit": 800.0,  # [K]
        "cp_liq": [220.0],  # [J/mol/K]
        "p_vap": [8.0, 2200.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 70000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
    "impurity": {
        "mw": 60.0,  # [g/mol]
        "rho_liq": 800.0,  # [kg/m**3]
        "t_crit": 700.0,  # [K]
        "cp_liq": [130.0],  # [J/mol/K]
        "p_vap": [8.0, 1800.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": 55000.0,  # [J/mol]
        "tref_hvap": 350.0,  # [K]
    },
}

# Solvent first: ind_solv == 0, the index a truthiness test cannot see.
SOLVENT_FIRST_ORDER = ("water", "api", "impurity")

# Same mixture with the solvent moved to a non-zero index. Species counts and
# positions differ from the order above, so a positional mistake cannot pass
# both fixtures with the same expected values.
SOLVENT_SECOND_ORDER = ("api", "water", "impurity")


def _write_thermo(tmp_path, order, filename):
    """Write a thermophysical-property file with the given species order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory that receives the generated file.
    order : sequence of str
        Species names, in the positional order the phase should use.
    filename : str
        Name of the generated JSON file.

    Returns
    -------
    str
        Absolute path of the written file, as `LiquidPhase` expects it.
    """
    path = tmp_path / filename
    path.write_text(json.dumps({name: _SPECIES[name] for name in order}))

    return str(path)


def _molar_volumes(order):
    """Return pure-species molar volumes for a species order.

    Parameters
    ----------
    order : sequence of str
        Species names in positional order.

    Returns
    -------
    numpy.ndarray
        Pure-species molar volumes with shape ``(num_species,)`` [L/mol].

    Notes
    -----
    ``mw / rho`` converts [g/mol] over [kg/m**3] directly into [L/mol],
    because 1 kg/m**3 equals 1 g/L. This is derived from the fixture
    constants rather than from `getDensityPure`, so the expectation does not
    reuse the production conversion under test.
    """
    return np.array(
        [_SPECIES[name]["mw"] / _SPECIES[name]["rho_liq"] for name in order]
    )  # [L/mol]


def _mass_densities(order):
    """Return pure-species liquid mass densities for a species order.

    Parameters
    ----------
    order : sequence of str
        Species names in positional order.

    Returns
    -------
    numpy.ndarray
        Pure-species mass densities with shape ``(num_species,)`` [kg/m**3].
    """
    return np.array([_SPECIES[name]["rho_liq"] for name in order])  # [kg/m**3]


def _molar_masses(order):
    """Return molecular masses for a species order.

    Parameters
    ----------
    order : sequence of str
        Species names in positional order.

    Returns
    -------
    numpy.ndarray
        Molecular masses with shape ``(num_species,)`` [g/mol].
    """
    return np.array([_SPECIES[name]["mw"] for name in order])  # [g/mol]


def _expected_mole_conc(order, solvent_name, solute_conc):
    """Close the ideal-mixture volume balance on a molar basis.

    Parameters
    ----------
    order : sequence of str
        Species names in positional order.
    solvent_name : str
        Name of the solvent species.
    solute_conc : dict
        Molar concentration of each non-solvent species, keyed by name
        [mol/L].

    Returns
    -------
    numpy.ndarray
        Molar concentrations with the solvent entry filled in, shape
        ``(num_species,)`` [mol/L].

    Notes
    -----
    The solvent closes ``sum_i c_i * v_i = 1`` with ``v_i`` the pure-species
    molar volume [L/mol], so ``c_solv = (1 - sum_solutes c_i v_i) / v_solv``.
    """
    mol_vol = _molar_volumes(order)  # [L/mol]
    conc = np.array(
        [solute_conc.get(name, 0.0) for name in order]
    )  # [mol/L]

    solvent_ind = order.index(solvent_name)
    solute_volume_frac = sum(
        solute_conc[name] * mol_vol[order.index(name)] for name in solute_conc
    )  # [-]
    conc[solvent_ind] = (1.0 - solute_volume_frac) / mol_vol[solvent_ind]

    return conc


def _expected_mass_conc(order, solvent_name, solute_conc):
    """Close the ideal-mixture volume balance on a mass basis.

    Parameters
    ----------
    order : sequence of str
        Species names in positional order.
    solvent_name : str
        Name of the solvent species.
    solute_conc : dict
        Mass concentration of each non-solvent species, keyed by name
        [kg/m**3].

    Returns
    -------
    numpy.ndarray
        Mass concentrations with the solvent entry filled in, shape
        ``(num_species,)`` [kg/m**3].

    Notes
    -----
    The solvent closes ``sum_i c_i / rho_i = 1`` with ``rho_i`` the
    pure-species mass density [kg/m**3], so
    ``c_solv = (1 - sum_solutes c_i / rho_i) * rho_solv``.
    """
    rho = _mass_densities(order)  # [kg/m**3]
    conc = np.array(
        [solute_conc.get(name, 0.0) for name in order]
    )  # [kg/m**3]

    solvent_ind = order.index(solvent_name)
    solute_volume_frac = sum(
        solute_conc[name] / rho[order.index(name)] for name in solute_conc
    )  # [-]
    conc[solvent_ind] = (1.0 - solute_volume_frac) * rho[solvent_ind]

    return conc


# Solute loadings shared by the molar-basis tests, keyed by species name.
MOLE_SOLUTES = {"api": 0.4, "impurity": 0.6}  # [mol/L]

# Solute loadings shared by the mass-basis tests, keyed by species name.
MASS_SOLUTES = {"api": 60.0, "impurity": 40.0}  # [kg/m**3]

# Relative tolerance for every comparison in this module.
#
# Both sides of each comparison evaluate the same governing equation on the
# same fixture constants in IEEE-754 double precision: the expected value here
# from `mw / rho_liq` and the volume-balance identity, the actual value inside
# PharmaPy from `1 / (rho_liq / mw)` and the same identity. The only admissible
# difference is therefore accumulated rounding over a chain of roughly ten
# elementary operations, whose worst case is a few units in the last place.
#
# The budget is set at 100 machine epsilons (~2.2e-14), about an order of
# magnitude above that worst case. It is deliberately a stated roundoff budget
# rather than a value fitted to an observed error: at the current fixture the
# two derivations happen to agree bit-for-bit (`mw / rho` and `1 / (rho / mw)`
# round identically for these particular constants), so every comparison in
# this module currently consumes none of the budget and even `rtol=0` would
# pass. That coincidence is a property of these numbers, not a guarantee, so
# the tolerance exists as headroom for future fixture edits. It is still four
# to seven orders of magnitude tighter than the NumPy (1e-7) and pytest (1e-6)
# defaults it replaces, so a genuine numerical regression is caught.
ROUNDOFF_RTOL = 100 * np.finfo(float).eps  # [-]

# No expected value in this module is zero, so no absolute floor is needed and
# an exact zero keeps the relative bound above authoritative. Raising this
# would silently weaken every comparison.
ROUNDOFF_ATOL = 0.0  # [-]


def test_conc_to_frac_returns_filled_conc_for_first_species_solvent(tmp_path):
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")
    phase = LiquidPhase(
        path, mass=1.0, mole_frac=np.array([0.98, 0.01, 0.01]),
        name_solv="water",
    )

    assert phase.ind_solv == 0

    expected_conc = _expected_mole_conc(
        SOLVENT_FIRST_ORDER, "water", MOLE_SOLUTES
    )  # [mol/L]
    expected_mole_frac = expected_conc / expected_conc.sum()  # [-]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]
    expected_mass_frac = (
        expected_mole_frac * mw / np.dot(expected_mole_frac, mw)
    )  # [-]

    # Only the solute entries are read; the solvent slot is back-calculated.
    conc_in = np.array(
        [0.0, MOLE_SOLUTES["api"], MOLE_SOLUTES["impurity"]]
    )  # [mol/L]

    # mass_frac [-], mole_frac [-], conc_out [mol/L]
    mass_frac, mole_frac, conc_out = phase.conc_to_frac(
        conc_in.copy(), solvent_ind=0
    )
    np.testing.assert_allclose(
        conc_out, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        mole_frac, expected_mole_frac, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        mass_frac, expected_mass_frac, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )

    # mole_frac_only [-], conc_mole_basis [mol/L]
    mole_frac_only, conc_mole_basis = phase.conc_to_frac(
        conc_in.copy(), solvent_ind=0, basis="mole"
    )
    np.testing.assert_allclose(
        mole_frac_only, expected_mole_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        conc_mole_basis, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )

    # mass_frac_only [-], conc_mass_basis [mol/L]
    mass_frac_only, conc_mass_basis = phase.conc_to_frac(
        conc_in.copy(), solvent_ind=0, basis="mass"
    )
    np.testing.assert_allclose(
        mass_frac_only, expected_mass_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        conc_mass_basis, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )


def test_mass_conc_to_frac_returns_filled_conc_for_first_species_solvent(
    tmp_path,
):
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")
    phase = LiquidPhase(
        path, mass=1.0, mole_frac=np.array([0.98, 0.01, 0.01]),
        name_solv="water",
    )

    expected_conc = _expected_mass_conc(
        SOLVENT_FIRST_ORDER, "water", MASS_SOLUTES
    )  # [kg/m**3]
    expected_mass_frac = expected_conc / expected_conc.sum()  # [-]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]
    expected_mole_frac = (
        (expected_mass_frac / mw) / np.dot(expected_mass_frac, 1 / mw)
    )  # [-]

    conc_in = np.array(
        [0.0, MASS_SOLUTES["api"], MASS_SOLUTES["impurity"]]
    )  # [kg/m**3]

    # mass_frac [-], mole_frac [-], conc_out [kg/m**3]
    mass_frac, mole_frac, conc_out = phase.mass_conc_to_frac(
        conc_in.copy(), solvent_ind=0
    )
    np.testing.assert_allclose(
        conc_out, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        mass_frac, expected_mass_frac, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        mole_frac, expected_mole_frac, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )

    # mass_frac_only [-], conc_mass_basis [kg/m**3]
    mass_frac_only, conc_mass_basis = phase.mass_conc_to_frac(
        conc_in.copy(), solvent_ind=0, basis="mass"
    )
    np.testing.assert_allclose(
        mass_frac_only, expected_mass_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        conc_mass_basis, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )

    # mole_frac_only [-], conc_mole_basis [kg/m**3]
    mole_frac_only, conc_mole_basis = phase.mass_conc_to_frac(
        conc_in.copy(), solvent_ind=0, basis="mole"
    )
    np.testing.assert_allclose(
        mole_frac_only, expected_mole_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        conc_mole_basis, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )


def test_liquid_phase_mole_conc_constructor_with_first_species_solvent(
    tmp_path,
):
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")

    expected_conc = _expected_mole_conc(
        SOLVENT_FIRST_ORDER, "water", MOLE_SOLUTES
    )  # [mol/L]
    expected_mole_frac = expected_conc / expected_conc.sum()  # [-]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]
    expected_mass_frac = (
        expected_mole_frac * mw / np.dot(expected_mole_frac, mw)
    )  # [-]
    expected_mw_av = np.dot(expected_mole_frac, mw)  # [g/mol]

    phase = LiquidPhase(
        path,
        mass=1.0,  # [kg]
        mole_conc=np.array(
            [0.0, MOLE_SOLUTES["api"], MOLE_SOLUTES["impurity"]]
        ),
        name_solv="water",
    )

    np.testing.assert_allclose(
        phase.mole_conc, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        phase.mole_frac, expected_mole_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase.mass_frac, expected_mass_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase.mass_conc, expected_conc * mw,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    assert phase.mw_av == pytest.approx(
        expected_mw_av, rel=ROUNDOFF_RTOL, abs=ROUNDOFF_ATOL
    )


@pytest.mark.parametrize("as_list", [False, True], ids=["ndarray", "list"])
def test_liquid_phase_update_mole_conc_with_first_species_solvent(
    tmp_path, as_list
):
    """Both argument types must return the back-calculated solvent entry.

    The converter writes the solvent concentration into the array produced by
    ``numpy.asarray``. That array is the caller's own object for an
    ``ndarray`` argument but a private copy for a list, so only the list case
    depends on the filled concentration actually being returned. Without the
    fix the list case silently keeps the unfilled solvent entry, which is the
    stale-composition impact reported in issue #57.
    """
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")
    phase = LiquidPhase(
        path, mass=1.0, mole_frac=np.array([0.98, 0.01, 0.01]),
        name_solv="water",
    )

    expected_conc = _expected_mole_conc(
        SOLVENT_FIRST_ORDER, "water", MOLE_SOLUTES
    )  # [mol/L]
    expected_mole_frac = expected_conc / expected_conc.sum()  # [-]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]

    conc_in = [0.0, MOLE_SOLUTES["api"], MOLE_SOLUTES["impurity"]]  # [mol/L]

    phase.updatePhase(
        mole_conc=conc_in if as_list else np.array(conc_in),
        mass=1.0,  # [kg]
    )

    np.testing.assert_allclose(
        phase.mole_conc, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        phase.mole_frac, expected_mole_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase.mass_conc, expected_conc * mw,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )


@pytest.mark.parametrize("as_list", [False, True], ids=["ndarray", "list"])
def test_liquid_phase_update_mass_conc_with_first_species_solvent(
    tmp_path, as_list
):
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")
    phase = LiquidPhase(
        path, mass=1.0, mole_frac=np.array([0.98, 0.01, 0.01]),
        name_solv="water",
    )

    expected_conc = _expected_mass_conc(
        SOLVENT_FIRST_ORDER, "water", MASS_SOLUTES
    )  # [kg/m**3]
    expected_mass_frac = expected_conc / expected_conc.sum()  # [-]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]

    conc_in = [
        0.0, MASS_SOLUTES["api"], MASS_SOLUTES["impurity"]
    ]  # [kg/m**3]

    phase.updatePhase(
        mass_conc=conc_in if as_list else np.array(conc_in),
        mass=1.0,  # [kg]
    )

    np.testing.assert_allclose(
        phase.mass_conc, expected_conc, rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )
    np.testing.assert_allclose(
        phase.mass_frac, expected_mass_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase.mole_conc, expected_conc / mw,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )


def test_liquid_stream_first_species_solvent(tmp_path):
    """`LiquidStream` is the entry point the repository's flowsheets use.

    It inherits the corrected converters through `LiquidPhase` and forwards
    to `LiquidPhase.updatePhase` positionally, mapping ``concentr`` to
    ``mole_conc`` and ``vol_flow``/``mass_flow``/``mole_flow`` to
    ``vol``/``mass``/``moles``. Asserting concentrations here pins that
    handoff as well as the solvent-index behavior, since an argument-order
    mistake would move the flow rate into a composition slot.
    """
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")

    expected_initial = _expected_mole_conc(
        SOLVENT_FIRST_ORDER, "water", MOLE_SOLUTES
    )  # [mol/L]
    mw = _molar_masses(SOLVENT_FIRST_ORDER)  # [g/mol]
    expected_mole_frac = expected_initial / expected_initial.sum()  # [-]
    expected_mw_av = np.dot(expected_mole_frac, mw)  # [g/mol]
    mass_flow = 1.0  # [kg/s]

    stream = LiquidStream(
        path,
        mass_flow=mass_flow,
        mole_conc=np.array(
            [0.0, MOLE_SOLUTES["api"], MOLE_SOLUTES["impurity"]]
        ),
        name_solv="water",
    )

    assert stream.ind_solv == 0
    np.testing.assert_allclose(
        stream.mole_conc, expected_initial,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        stream.mole_frac, expected_mole_frac,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    assert stream.mw_av == pytest.approx(
        expected_mw_av, rel=ROUNDOFF_RTOL, abs=ROUNDOFF_ATOL
    )
    # The flow rate must survive the positional handoff unchanged.
    assert stream.mass_flow == pytest.approx(
        mass_flow, rel=ROUNDOFF_RTOL, abs=ROUNDOFF_ATOL
    )

    # Update with a different, asymmetric loading so a stale composition
    # cannot pass by matching the constructor result.
    updated_solutes = {"api": 0.8, "impurity": 0.2}  # [mol/L]
    expected_updated = _expected_mole_conc(
        SOLVENT_FIRST_ORDER, "water", updated_solutes
    )  # [mol/L]

    stream.updatePhase(
        concentr=np.array(
            [0.0, updated_solutes["api"], updated_solutes["impurity"]]
        ),
        mass_flow=mass_flow,
    )

    np.testing.assert_allclose(
        stream.mole_conc, expected_updated,
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        stream.mole_frac, expected_updated / expected_updated.sum(),
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    assert stream.mass_flow == pytest.approx(
        mass_flow, rel=ROUNDOFF_RTOL, abs=ROUNDOFF_ATOL
    )

    # The two loadings must differ, or the update assertion is vacuous.
    assert not np.allclose(expected_initial, expected_updated)


def test_first_species_solvent_matches_reordered_species(tmp_path):
    """The same mixture must give permuted results when species are reordered.

    Reordering moves the solvent off index 0, which is the only case the
    truthiness check handled. Comparing the two orders therefore separates a
    genuine composition from a positional coincidence.
    """
    path_first = _write_thermo(
        tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json"
    )
    path_second = _write_thermo(
        tmp_path, SOLVENT_SECOND_ORDER, "solvent_second.json"
    )

    def _mole_conc_vector(order):
        return np.array(
            [MOLE_SOLUTES.get(name, 0.0) for name in order]
        )  # [mol/L]

    phase_first = LiquidPhase(
        path_first, mass=1.0, mole_conc=_mole_conc_vector(SOLVENT_FIRST_ORDER),
        name_solv="water",
    )
    phase_second = LiquidPhase(
        path_second, mass=1.0,
        mole_conc=_mole_conc_vector(SOLVENT_SECOND_ORDER),
        name_solv="water",
    )

    assert phase_first.ind_solv == 0
    assert phase_second.ind_solv == 1

    # Map the second ordering back onto the first so the same species are
    # compared, rather than the same positions.
    permutation = [SOLVENT_SECOND_ORDER.index(name)
                   for name in SOLVENT_FIRST_ORDER]

    np.testing.assert_allclose(
        phase_first.mole_conc, phase_second.mole_conc[permutation],
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase_first.mole_frac, phase_second.mole_frac[permutation],
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    np.testing.assert_allclose(
        phase_first.mass_frac, phase_second.mass_frac[permutation],
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )
    assert phase_first.mw_av == pytest.approx(
        phase_second.mw_av, rel=ROUNDOFF_RTOL, abs=ROUNDOFF_ATOL
    )

    # The permutation must actually reorder; otherwise the comparison above
    # would hold for the wrong reason.
    assert permutation != [0, 1, 2]


def test_absent_solvent_still_returns_fractions_only(tmp_path):
    """A phase without a declared solvent keeps the two-value return."""
    path = _write_thermo(tmp_path, SOLVENT_FIRST_ORDER, "solvent_first.json")
    phase = LiquidPhase(
        path, mass=1.0, mole_frac=np.array([0.98, 0.01, 0.01])
    )

    assert phase.ind_solv is None

    conc = np.array([50.0, 0.4, 0.6])  # [mol/L]
    mass_frac, mole_frac = phase.conc_to_frac(conc.copy())  # [-], [-]
    np.testing.assert_allclose(
        mole_frac, conc / conc.sum(), rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL
    )

    mass_conc = np.array([900.0, 60.0, 40.0])  # [kg/m**3]
    mass_frac_only, mole_frac_only = phase.mass_conc_to_frac(
        mass_conc.copy()
    )  # [-], [-]
    np.testing.assert_allclose(
        mass_frac_only, mass_conc / mass_conc.sum(),
        rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL,
    )

    # Single-value returns for an explicit basis must stay unwrapped.
    assert isinstance(
        phase.conc_to_frac(conc.copy(), basis="mole"), np.ndarray
    )
    assert isinstance(
        phase.mass_conc_to_frac(mass_conc.copy(), basis="mass"), np.ndarray
    )
