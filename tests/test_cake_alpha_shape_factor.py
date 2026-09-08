"""Regression tests for the shape factor used by ``Cake.get_alpha``.

Issue #160 records that the scalar volumetric shape factor cancels from the
normalized crystal-volume weights. The hydraulic resistance ``alpha`` should
therefore be independent of any positive scalar ``kv`` at fixed porosity,
while the method must
still obtain that factor from its attached solid phase rather than a literal.
"""

from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase


pytestmark = pytest.mark.unit

# Historical dimensionless literal replaced by the phase-owned value.
LEGACY_SHAPE_FACTOR = 0.524  # [-]

# Additional positive probes: 0.5 reproduces the #158 cake case, while 0.8 is
# a distinct scalar selected to exercise cancellation rather than calibration.
SHAPE_FACTOR_PROBES = (LEGACY_SHAPE_FACTOR, 0.5, 0.8)  # [-]

# Alpha recorded for the #158 fixture on unchanged master with kv = 0.524.
EXPECTED_LEGACY_ALPHA = 3.858303591823321e7  # [m/kg]

# Four binary64 epsilon allow the weighted sum to reassociate across NumPy
# builds while remaining a few units in the last place around the reference.
ALPHA_RELATIVE_TOLERANCE = 4 * np.finfo(float).eps  # [-]

# Liquid composition for the shared A/B/C/D/solvent database [mass fraction].
LIQUID_MASS_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-]

# Pure-component-A solid composition [mass fraction].
SOLID_MASS_FRAC = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-]


@pytest.fixture(scope="module")
def thermo_path(data_path):
    """Return the shared pure-component thermodynamic database path.

    Parameters
    ----------
    data_path : dict[str, pathlib.Path]
        Repository test-data directories.

    Returns
    -------
    str
        Path to the flowsheet compound database.
    """
    return str(data_path["flowsheet"] / "compound_database.json")


def _make_cake(thermo_path: str, shape_factor: float) -> Cake:
    """Build the #158 cake fixture with a selected crystal shape factor.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    shape_factor : float
        Volumetric shape factor of the solid crystals [-].

    Returns
    -------
    Cake
        Cake with real liquid and solid phase collaborators attached.
    """
    x_distrib = np.linspace(10.0, 500.0, 40)  # [um]
    mean_size = 200.0  # [um]
    size_std = 60.0  # [um]
    # Peak number density chosen in #158 to produce a litre-scale cake [#/um].
    peak_density = 8.0e5  # [#/um]
    distrib = peak_density * np.exp(
        -0.5 * ((x_distrib - mean_size) / size_std)**2)  # [#/um]

    liquid = LiquidPhase(thermo_path, mass_frac=LIQUID_MASS_FRAC)
    solid = SolidPhase(
        thermo_path,
        mass_frac=SOLID_MASS_FRAC,
        x_distrib=x_distrib,
        distrib=distrib,
        kv=shape_factor,
    )

    cake = Cake()
    cake.Phases = (liquid, solid)
    return cake


def test_cake_alpha_is_invariant_to_real_phase_shape_factor(thermo_path):
    """Verify shape-factor cancellation at fixed porosity with real phases.

    #165 makes the packing model use phase-owned kv. Its epsilon-regularized
    weights have a small kv dependence, so exact alpha invariance applies
    only at fixed porosity. Hold the historical porosity constant to isolate
    alpha's cancellation; separate packing regressions cover that dependence.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    """
    reference = _make_cake(thermo_path, LEGACY_SHAPE_FACTOR)
    alpha_values = []  # [m/kg]
    for shape_factor in SHAPE_FACTOR_PROBES:
        cake = _make_cake(thermo_path, shape_factor)
        cake.porosity = reference.porosity  # [-], isolate resistance weights
        alpha_values.append(cake.get_alpha())
    alpha_values = np.array(alpha_values)  # [m/kg]
    assert alpha_values[0] == pytest.approx(
        EXPECTED_LEGACY_ALPHA,
        rel=ALPHA_RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        alpha_values,
        EXPECTED_LEGACY_ALPHA,
        rtol=ALPHA_RELATIVE_TOLERANCE,
        atol=0.0,
    )


def test_cake_alpha_reads_shape_factor_from_solid_phase(thermo_path):
    """Verify ``alpha`` reads ``kv`` from the attached solid phase.

    Parameters
    ----------
    thermo_path : str
        Path to the pure-component thermodynamic database.
    """
    cake = _make_cake(thermo_path, LEGACY_SHAPE_FACTOR)

    with patch.object(
        type(cake.Solid_1),
        "kv",
        new_callable=PropertyMock,
        create=True,
        return_value=LEGACY_SHAPE_FACTOR,
    ) as phase_shape_factor:
        cake.get_alpha()

    phase_shape_factor.assert_called_once_with()
