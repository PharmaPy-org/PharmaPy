"""Validate flash phase boundaries using real shipped thermophysical data.

The equimolar five-species mixture is liquid at 340 K and splits into two
phases at 360 K and atmospheric pressure. No optional ODE backend is used.
"""

import numpy as np
import pytest

from PharmaPy import Evaporators
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit
PRESSURE = 101325.0  # [Pa], standard atmosphere
FRACTIONS = np.full(5, 0.2)  # [-], equimolar shipped mixture
BALANCE_TOLERANCE = 1e-7  # [-], nonlinear residual allowance


@pytest.mark.parametrize('flash_class', [Evaporators.IsothermalFlash, Evaporators.AdiabaticFlash])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('temperature', [340.0, 360.0])  # [K], liquid and two-phase cases
def test_flash_publishes_physical_phase_boundary(data_path, flash_class, stream, temperature):
    phase_class = LiquidStream if stream else LiquidPhase
    amount_name = 'mole_flow' if stream else 'moles'
    unit = flash_class(pres_drum=PRESSURE)
    unit.Inlet = phase_class(str(data_path['flowsheet'] / 'compound_database.json'),
                            mole_frac=FRACTIONS, temp=temperature, pres=PRESSURE,
                            **{amount_name: 1.0})  # [mol/s] or [mol], unit feed
    unit.solve_unit()
    liquid = getattr(unit.LiquidOut, amount_name)  # [mol/s] or [mol]
    vapor = getattr(unit.VaporOut, amount_name)  # [mol/s] or [mol]
    assert 0 <= liquid <= 1 and 0 <= vapor <= 1
    assert liquid + vapor == pytest.approx(1, abs=BALANCE_TOLERANCE)
    np.testing.assert_allclose(liquid * unit.LiquidOut.mole_frac + vapor * unit.VaporOut.mole_frac,
                               FRACTIONS, atol=BALANCE_TOLERANCE, rtol=0)
    if temperature == 340:
        assert vapor == pytest.approx(0, abs=BALANCE_TOLERANCE)
    else:
        assert 0 < vapor < 1


@pytest.mark.parametrize('failure', ['unconverged', 'negative_phase', 'nonfinite'])
def test_flash_rejects_invalid_solver_output(data_path, monkeypatch, failure):
    unit = Evaporators.IsothermalFlash(pres_drum=PRESSURE)
    unit.Inlet = LiquidPhase(str(data_path['flowsheet'] / 'compound_database.json'),
                            mole_frac=FRACTIONS, temp=340.0, moles=1.0)  # [K], [mol]
    original = Evaporators.fsolve

    def invalid_solution(function, initial, **kwargs):
        """Corrupt the native solver result at its optional numerical boundary.

        Parameters
        ----------
        function : callable
            Flash residual, dimensionless after energy scaling.
        initial : ndarray
            Phase fractions and compositions [-].
        **kwargs : dict
            Forwarded SciPy options.

        Returns
        -------
        tuple or ndarray
            Native result with one deliberately invalid convergence condition.
        """
        result = original(function, initial, **kwargs)
        solution = result[0] if isinstance(result, tuple) else result
        if failure == 'negative_phase':
            solution[-1] = -0.1  # [-], materially invalid, not roundoff
        elif failure == 'nonfinite':
            solution[-1] = np.nan  # [-]
        elif isinstance(result, tuple):
            result = (solution, result[1], 5, 'synthetic lack of convergence')
        return result

    monkeypatch.setattr(Evaporators, 'fsolve', invalid_solution)
    with pytest.raises(RuntimeError, match='Flash'):
        unit.solve_unit()
    assert not hasattr(unit, 'LiquidOut')
