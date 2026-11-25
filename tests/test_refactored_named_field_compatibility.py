"""Check inherited named retrieval on real post-#373 phases and streams.

The asymmetric five-species mixture uses the flowsheet property database.
Species-mapped constructors and setters remain a separate #338 follow-up;
these checks exercise retrieval of the established array-based state.
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.Streams_Refactored import LiquidStream, SolidStream, VaporStream


pytestmark = pytest.mark.unit
DATABASE = Path(__file__).parent / 'Flowsheet' / 'data' / 'compound_database.json'
# Distinct fractions expose species-order errors; their sum is exactly one.
MASS_FRACTIONS = [0.15, 0.20, 0.05, 0.0, 0.60]  # [-]


@pytest.mark.parametrize('material_class,amount', [
    (LiquidPhase, {'mass': 1.0}),  # [kg], unit inventory
    (SolidPhase, {'mass': 1.0}),  # [kg], unit inventory
    (VaporPhase, {'mass': 1.0}),  # [kg], unit inventory
    (LiquidStream, {'mass_flow': 1.0}),  # [kg/s], unit flow
    (SolidStream, {'mass_flow': 1.0}),  # [kg/s], unit flow
    (VaporStream, {'mass_flow': 1.0}),  # [kg/s], unit flow
])
def test_refactored_material_inherits_named_retrieval(material_class, amount):
    """Resolve names in caller order and preserve stored-state ownership.

    Parameters
    ----------
    material_class : type
        Concrete refactored phase or stream constructor.
    amount : dict of str to float
        Unit mass [kg] or mass flow [kg/s] specification.
    """
    material = material_class(
        DATABASE, mass_frac=MASS_FRACTIONS, **amount)
    selected = material.retrieve_field(
        'mass_frac', species=['solvent', 'A'], as_dict=True)  # [-]
    assert list(selected) == ['solvent', 'A']
    np.testing.assert_array_equal(list(selected.values()), [0.60, 0.15])

    copied = material.retrieve_field('mass_frac')  # [-]
    copied[0] = 0.0  # [-], mutation sentinel
    np.testing.assert_array_equal(material.mass_frac, MASS_FRACTIONS)
    stored_view = material.retrieve_field('mass_frac', copy=False)  # [-]
    assert np.shares_memory(stored_view, material.mass_frac)
    # Refactored mole fractions are computed properties, not stored buffers.
    computed = material.retrieve_field('mole_frac', copy=False)  # [-]
    assert not np.shares_memory(computed, material.mass_frac)
    with pytest.raises(KeyError, match='Unknown species'):
        material.retrieve_field('mass_frac', species='absent-species')
