"""Regression tests for distribution-driven ``SolidPhase.updatePhase``.

The volumetric shape factor ``kv`` relates the third moment of the crystal
size distribution to the crystal volume fraction, so every path that
back-calculates volume and mass from ``moments[3]`` must apply it. The
constructor does; the ``distrib`` branch of ``updatePhase`` must agree with it
rather than assuming ``kv = 1``.

Distribution updates must also preserve ``num_mom``, which sizes the moment
state used by classical moment crystallizers, including when a phase receives
its first distribution after construction.

The fixture uses a two-component solid with a common density, so the density
mixing rule cannot perturb the moments-derived mass, and a four-node size grid
whose length differs from the component count so a transposed axis cannot pass
unnoticed.
"""

import json

import numpy as np
import pytest

from PharmaPy.Phases import SolidPhase


_MW_A = 100.0  # [g/mol]
_MW_B = 50.0  # [g/mol]
_RHO_SOLID = 1500.0  # [kg/m**3]

_MASS_FRAC = [0.8, 0.2]  # [-]

# Non-unit volumetric shape factor, e.g. a needle-like habit. Any value other
# than 1 separates the constructor's convention from an implicit kv = 1.
_KV = 0.5  # [-]

# Internal size coordinate of the crystals, uniform 100 um spacing.
_X_DISTRIB = np.array([0.0, 100.0, 200.0, 300.0])  # [um]

# Number density with distinct interior nodes, so the quadrature weights of
# the two interior points cannot cancel a swapped-node error.
_DISTRIB = np.array([0.0, 1.0e7, 1.25e6, 0.0])  # [#/um]

# Third moment computed independently of PharmaPy: the trapezoidal integral of
# distrib * x**3 over the grid, converted from um**3 to m**3.
#   h * [ (0 + a*1e6)/2 + (a*1e6 + b*8e6)/2 + (b*8e6 + 0)/2 ]
#     = 100 * (1e7 * 1e6 + 8 * 1.25e6 * 1e6) = 2e15 um**3
_MOM_THREE = 2.0e15 * 1e-18  # [m**3]

# Scaling applied to the distribution passed to updatePhase, so the update is
# a genuine state change rather than a repeat of the constructor's input.
_UPDATE_SCALE = 2.0  # [-]


@pytest.fixture
def path_thermo(tmp_path):
    """Write a minimal two-component solid thermophysical database.

    Returns
    -------
    str
        Path to a JSON database with molecular weights [g/mol], solid heat
        capacities [J/kg/K], and solid densities [kg/m**3].
    """
    database = {
        'A': {'mw': _MW_A, 'cp_solid': [1600.0], 'rho_solid': _RHO_SOLID},
        'B': {'mw': _MW_B, 'cp_solid': [1600.0], 'rho_solid': _RHO_SOLID},
    }

    path = tmp_path / 'solid_db.json'
    path.write_text(json.dumps(database))

    return str(path)


def test_update_phase_distrib_applies_shape_factor(path_thermo):
    """``updatePhase(distrib=...)`` must scale the third moment by ``kv``.

    The constructor sets ``vol = moments[3] * kv``; updating the distribution
    must keep that convention and must derive the mass from the shape-corrected
    volume, otherwise both are overstated by ``1 / kv``.
    """
    phase = SolidPhase(path_thermo, mass=0, mass_frac=_MASS_FRAC,
                       x_distrib=_X_DISTRIB, distrib=_DISTRIB, kv=_KV)

    # Constructor baseline: confirms the fixture's hand-computed third moment
    # and pins the convention the update path must match.
    assert phase.moments[3] == pytest.approx(_MOM_THREE)
    assert phase.vol == pytest.approx(_MOM_THREE * _KV)  # [m**3]

    updated_distrib = _DISTRIB * _UPDATE_SCALE  # [#/um]
    phase.updatePhase(distrib=updated_distrib)

    expected_mom_three = _MOM_THREE * _UPDATE_SCALE  # [m**3]
    expected_vol = expected_mom_three * _KV  # [m**3]
    expected_mass = expected_vol * _RHO_SOLID  # [kg]

    assert phase.moments[3] == pytest.approx(expected_mom_three)

    # Compared jointly so a failure reports both the volume [m**3] and the
    # mass [kg] that the update produced.
    assert (phase.vol, phase.mass) == pytest.approx(
        (expected_vol, expected_mass))


def test_update_phase_distrib_preserves_configured_moment_count(path_thermo):
    """Distribution updates must retain every configured moment order.

    Classical moment crystallizers use ``SolidPhase.num_mom`` to size their
    moment state vector, and ``method_of_moments`` evolves every supplied
    order. Recomputing only orders 0--3 would silently change that state shape.
    """
    num_mom = 6  # [-], exercises orders 0--5 beyond the four-moment default
    updated_distrib = _DISTRIB * _UPDATE_SCALE  # [#/um]
    initialized_phase = SolidPhase(
        path_thermo, mass=0, mass_frac=_MASS_FRAC,
        x_distrib=_X_DISTRIB, distrib=_DISTRIB, kv=_KV, num_mom=num_mom,
    )
    deferred_phase = SolidPhase(
        path_thermo, mass=0, mass_frac=_MASS_FRAC, kv=_KV,
        num_mom=num_mom,
    )

    # Hand-computed trapezoidal moments for orders 0--5. Entry n has units
    # [m**n] on the total-population basis (order zero is a crystal count [-]).
    expected_moments = np.array([
        2.25e9, 2.5e5, 30.0, 4.0e-3, 6.0e-7, 1.0e-10,
    ])  # heterogeneous [m**n], n = 0,...,5
    for phase in (initialized_phase, deferred_phase):
        phase.updatePhase(x_distrib=_X_DISTRIB, distrib=updated_distrib)
        assert phase.num_mom == num_mom
        assert phase.moments == pytest.approx(expected_moments)


def test_update_phase_distrib_matches_zero_mass_constructor(path_thermo):
    """Distribution-derived construction and updating must agree.

    This equivalence applies when ``mass == 0`` and both paths receive a
    number-based total-population distribution [#/um]. With explicit mass, the
    constructor instead normalizes and converts ``distrib`` by
    ``distrib_type`` while ``updatePhase`` assigns it directly.
    """
    target_distrib = _DISTRIB * _UPDATE_SCALE  # [#/um]

    constructed = SolidPhase(
        path_thermo, mass=0, mass_frac=_MASS_FRAC,
        x_distrib=_X_DISTRIB, distrib=target_distrib, kv=_KV,
    )
    updated = SolidPhase(
        path_thermo, mass=0, mass_frac=_MASS_FRAC,
        x_distrib=_X_DISTRIB, distrib=_DISTRIB, kv=_KV,
    )
    updated.updatePhase(distrib=target_distrib)

    # The constructor and update path represent the same physical population,
    # so their solid volume [m**3] and mass [kg] must be identical.
    assert (updated.vol, updated.mass) == pytest.approx(
        (constructed.vol, constructed.mass))


def test_update_phase_preserves_unit_shape_factor_behavior(path_thermo):
    """The historical ``kv = 1`` calculation must remain bit-for-bit stable."""
    unit_shape_factor = 1.0  # [-]
    updated_distrib = _DISTRIB * _UPDATE_SCALE  # [#/um]
    phase = SolidPhase(
        path_thermo, mass=0, mass_frac=_MASS_FRAC,
        x_distrib=_X_DISTRIB, distrib=_DISTRIB, kv=unit_shape_factor,
    )

    phase.updatePhase(distrib=updated_distrib)

    # These are the exact pre-fix expressions when kv = 1. Comparing their
    # hexadecimal forms proves that the compatibility case is bit-identical.
    expected_vol = phase.moments[3]  # [m**3]
    expected_mass = phase.moments[3] * phase.getDensity()  # [kg]
    assert float(phase.vol).hex() == float(expected_vol).hex()
    assert float(phase.mass).hex() == float(expected_mass).hex()
