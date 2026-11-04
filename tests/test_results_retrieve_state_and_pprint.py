"""Tests for DynamicResult.retrieve_state and bounded component-list display (#341).

Covers:
- `DynamicResult.retrieve_state(field_name, species=None, copy=True)` across
  2-D tank arrays, dictionary-valued spatial profiles (PFR), derived
  `di_fstates` trajectories, and scalar/1-D unindexed states (`temp`, `vol`,
  `time`).
- Reordered species subsets, copy isolation, and dual `(KeyError, ValueError)`
  exceptions for unknown fields or species.
- Bounded component-list display with ellipsis in `Results.pprint` and
  `DynamicResult.__repr__` / `DynamicResult.pprint` without mutating stored
  metadata dictionaries.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from PharmaPy.Results import DynamicResult, pprint


pytestmark = pytest.mark.unit


def _make_tank_result() -> DynamicResult:
    """Construct a representative 6-species tank DynamicResult."""
    species = ['A', 'N', 'M', 'AA', 'BP', 'solvent']
    times = np.array([0.0, 10.0, 20.0])  # [s]
    # Distinct trajectories [mol/L] per species so ordering is verifiable
    mole_conc = np.array([
        [1.0, 50.0, 50.0, 50.0, 0.0, 12.0],
        [0.5, 49.5, 49.8, 49.8, 0.1, 12.0],
        [0.1, 49.1, 49.6, 49.6, 0.3, 12.0],
    ])  # [mol/L]
    temp = np.array([298.15, 300.15, 302.15])  # [K]
    vol = np.array([0.1, 0.1, 0.1])  # [m**3]
    mass_frac = mole_conc / np.sum(mole_conc, axis=1, keepdims=True)  # [-]

    di_states = {
        'mole_conc': {'dim': len(species), 'units': 'mol/L', 'index': list(species), 'type': 'diff'},
        'temp': {'dim': 1, 'units': 'K', 'type': 'diff'},
        'vol': {'dim': 1, 'units': 'm**3', 'type': 'diff'},
    }
    di_fstates = {
        'mass_frac': {'dim': len(species), 'units': '', 'index': list(species)},
    }
    return DynamicResult(
        di_states,
        di_fstates=di_fstates,
        time=times,
        mole_conc=mole_conc,
        temp=temp,
        vol=vol,
        mass_frac=mass_frac,
    )


def _make_pfr_result() -> DynamicResult:
    """Construct a representative dictionary-valued PFR DynamicResult."""
    species = ['A', 'B', 'C']
    times = np.array([0.0, 5.0])  # [s]
    # Shape (num_times=2, num_axial=4) [mol/L]
    mole_conc = {
        'A': np.array([[1.0, 0.8, 0.6, 0.4], [0.9, 0.6, 0.3, 0.1]]),
        'B': np.array([[0.0, 0.2, 0.4, 0.6], [0.1, 0.4, 0.7, 0.9]]),
        'C': np.array([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]]),
    }  # [mol/L]
    temp = np.full((2, 4), 310.0)  # [K]
    di_states = {
        'mole_conc': {'dim': len(species), 'units': 'mol/L', 'index': list(species), 'type': 'diff'},
        'temp': {'dim': 1, 'units': 'K', 'index': [1, 2, 3, 4], 'type': 'diff'},
    }
    return DynamicResult(
        di_states,
        time=times,
        mole_conc=mole_conc,
        temp=temp,
    )


def test_retrieve_state_tank_and_pfr_species_dictionaries() -> None:
    """Tank 2-D arrays and PFR dict states share the same named dict interface."""
    tank_res = _make_tank_result()
    conc_sim = tank_res.retrieve_state('mole_conc')
    assert isinstance(conc_sim, dict)
    assert list(conc_sim.keys()) == ['A', 'N', 'M', 'AA', 'BP', 'solvent']
    assert conc_sim['A'] == pytest.approx([1.0, 0.5, 0.1])
    assert conc_sim['BP'] == pytest.approx([0.0, 0.1, 0.3])

    # Single-species retrieval and reordered subset retrieval
    assert tank_res.retrieve_state('mole_conc', species='M') == pytest.approx([50.0, 49.8, 49.6])
    subset = tank_res.retrieve_state('mole_conc', species=['BP', 'A'])
    assert list(subset.keys()) == ['BP', 'A']
    assert subset['BP'] == pytest.approx([0.0, 0.1, 0.3])
    assert subset['A'] == pytest.approx([1.0, 0.5, 0.1])

    # Derived fstates retrieval
    mf_sim = tank_res.retrieve_state('mass_frac')
    assert list(mf_sim.keys()) == ['A', 'N', 'M', 'AA', 'BP', 'solvent']

    # Copy isolation by default
    conc_sim['A'][0] = -999.0
    assert tank_res.mole_conc[0, 0] == pytest.approx(1.0)

    # PFR dict-valued state retrieval preserves (time, axial) shape
    pfr_res = _make_pfr_result()
    pfr_conc = pfr_res.retrieve_state('mole_conc')
    assert list(pfr_conc.keys()) == ['A', 'B', 'C']
    assert pfr_conc['A'].shape == (2, 4)
    assert pfr_conc['B'][1, :] == pytest.approx([0.1, 0.4, 0.7, 0.9])
    assert pfr_res.retrieve_state('mole_conc', species='A').shape == (2, 4)


def test_retrieve_state_scalar_and_error_handling() -> None:
    """Verify unindexed state retrieval and dual (KeyError, ValueError) errors."""
    tank_res = _make_tank_result()
    assert tank_res.retrieve_state('temp') == pytest.approx([298.15, 300.15, 302.15])
    assert tank_res.retrieve_state('time') == pytest.approx([0.0, 10.0, 20.0])

    # Unknown state raises both KeyError and ValueError
    with pytest.raises(KeyError, match="State 'unknown_state' does not exist"):
        tank_res.retrieve_state('unknown_state')
    with pytest.raises(ValueError, match="State 'unknown_state' does not exist"):
        tank_res.retrieve_state('unknown_state')

    # Unknown species raises both KeyError and ValueError
    with pytest.raises(KeyError, match="Unknown species 'Z'"):
        tank_res.retrieve_state('mole_conc', species='Z')
    with pytest.raises(ValueError, match="Unknown species 'Z'"):
        tank_res.retrieve_state('mole_conc', species='Z')

    # Unindexed state rejects species filter
    with pytest.raises(ValueError, match="not indexed by species"):
        tank_res.retrieve_state('temp', species='A')


def test_bounded_component_list_display_and_non_mutation() -> None:
    """Verify bounded component-list truncation with ellipsis and metadata preservation."""
    tank_res = _make_tank_result()
    orig_states = copy.deepcopy(tank_res.di_states)
    orig_fstates = copy.deepcopy(tank_res.di_fstates)

    # Default max_components=5 truncates the 6-species list with '...'
    rendered_default = repr(tank_res)
    assert 'A, N, M, AA, ..., solvent' in rendered_default
    assert 'BP' not in rendered_default

    # Custom max_components=3
    rendered_3 = tank_res.pprint(max_components=3)
    assert 'A, N, ..., solvent' in rendered_3

    # Unbounded display with max_components=None shows all 6 species
    rendered_all = tank_res.pprint(max_components=None)
    assert 'A, N, M, AA, BP, solvent' in rendered_all

    # Stored metadata dictionaries must remain completely unmodified
    assert tank_res.di_states == orig_states
    assert tank_res.di_fstates == orig_fstates

    # Invalid max_components raises ValueError
    with pytest.raises(ValueError, match="max_components must be a positive integer"):
        pprint(tank_res.di_states, 'states', {'dim': '', 'units': 's', 'index': 's'}, max_components=0)

    # Instance-configured max_components is honored by result.pprint() when called with no args
    tank_res_custom = DynamicResult(
        tank_res.di_states,
        tank_res.di_fstates,
        time=tank_res.time,
        mole_conc=tank_res.mole_conc,
        temp=tank_res.temp,
        mass_frac=tank_res.mass_frac,
        max_components=3,
    )
    assert 'A, N, ..., solvent' in tank_res_custom.pprint()
    assert 'A, N, M, AA, BP, solvent' in tank_res_custom.pprint(max_components=None)


def test_retrieve_state_preserves_time_axis_for_squeezed_singleton_species() -> None:
    """Single-species trajectories unpacked as 1-D (n_time,) retain their time axis."""
    from PharmaPy.Commons import unpack_states

    di_states = {
        'mass_frac': {'dim': 1, 'units': '', 'index': ['A'], 'type': 'diff'},
        'temp': {'dim': 1, 'units': 'K', 'type': 'diff'},
    }
    raw_traj = np.array([[1.0, 300.0], [0.9, 305.0], [0.8, 310.0]])  # [-], [K]
    unpacked = unpack_states(raw_traj, [1, 1], ['mass_frac', 'temp'])
    assert unpacked['mass_frac'].shape == (3,)

    res = DynamicResult(di_states, time=np.array([0.0, 5.0, 10.0]), **unpacked)
    all_species = res.retrieve_state('mass_frac')
    assert list(all_species.keys()) == ['A']
    assert all_species['A'].shape == (3,)
    assert all_species['A'] == pytest.approx([1.0, 0.9, 0.8])

    single_sp = res.retrieve_state('mass_frac', species='A')
    assert single_sp.shape == (3,)
    assert single_sp == pytest.approx([1.0, 0.9, 0.8])

    subset_sp = res.retrieve_state('mass_frac', species=['A'])
    assert subset_sp['A'].shape == (3,)
    assert subset_sp['A'] == pytest.approx([1.0, 0.9, 0.8])
