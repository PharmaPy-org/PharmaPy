"""Missing liquid correlations must fail on use, not isothermal construction."""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.ThermoModule import MissingPropertyError


@pytest.mark.parametrize('phase_class', [LiquidPhase, LiquidStream])
def test_empty_correlations_remain_missing_until_requested(phase_class):
    """Construct density-only objects without inventing thermal properties.

    Parameters
    ----------
    phase_class : type
        Native liquid phase or stream constructor.
    """
    database = Path(__file__).parent / 'fixtures/native_four_species.json'
    inventory = {'mass': 1.0} if phase_class is LiquidPhase else {'mass_flow': 1.0}  # [kg] or [kg/s]
    fractions = np.full(4, 0.25)  # [-], equal synthetic species mass fractions
    phase = phase_class(database, mass_frac=fractions, verbose=False, **inventory)
    assert phase.getDensity() == pytest.approx(1000.0)  # [kg/m**3], fixture density
    assert 'cp_liq' not in phase.__dict__
    assert 'p_vap' not in phase.__dict__
    with pytest.raises(MissingPropertyError, match='cp_liq'):
        phase.getCp()
    with pytest.raises(MissingPropertyError, match='p_vap'):
        phase.AntoineEquation(temp=phase.temp)
