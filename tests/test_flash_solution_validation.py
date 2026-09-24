"""Validate flash phase boundaries using real shipped thermophysical data.

The equimolar five-species mixture is liquid at 340 K and splits into two
phases at 360 K and atmospheric pressure. No optional ODE backend is used.
"""

import numpy as np
import pytest

from scipy.optimize import fsolve

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
def test_flash_rejects_invalid_solver_output(data_path, failure):
    """Validate rejected candidates directly using the production result gate.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    failure : str
        Candidate convergence, phase-bound, or finiteness violation.
    """
    unit = Evaporators.IsothermalFlash(pres_drum=PRESSURE)
    unit.Inlet = LiquidPhase(str(data_path['flowsheet'] / 'compound_database.json'),
                            mole_frac=FRACTIONS, temp=340.0, moles=1.0)  # [K], [mol]
    seed = np.concatenate(([0.5], FRACTIONS, FRACTIONS, [0.5]))  # [-], equal split
    solution, _, status, message = fsolve(unit.unit_model, seed, full_output=True)
    assert status == 1, message
    if failure == 'negative_phase':
        solution[-1] = -0.1  # [-], materially invalid, not roundoff
        expected_error = r'Flash returned phase fractions outside \[0, 1\]'
    elif failure == 'nonfinite':
        solution[-1] = np.nan  # [-], invalid phase amount
        expected_error = 'Flash failed to converge'
    else:
        status = 5  # MINPACK code: iteration made insufficient progress
        expected_error = 'Flash failed to converge'
    with pytest.raises(RuntimeError, match=expected_error):
        Evaporators._validate_flash_solution(
            unit.unit_model, solution, status, message, -1, BALANCE_TOLERANCE)


@pytest.mark.parametrize('flash_class', [Evaporators.IsothermalFlash, Evaporators.AdiabaticFlash])
def test_flash_public_solve_rejects_unclosed_result(data_path, flash_class):
    """The public solve must validate native residuals before publishing phases.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    flash_class : type
        Isothermal or adiabatic public flash API.
    """
    unit = flash_class(pres_drum=PRESSURE)
    unit.Inlet = LiquidPhase(str(data_path['flowsheet'] / 'compound_database.json'),
                            mole_frac=FRACTIONS, temp=360.0, moles=1.0)  # [K], [mol]
    vapor_placeholder = getattr(unit, 'VaporOut', None)
    # Deliberately demand exact floating-point closure; this real nonlinear
    # mixture has nonzero roundoff residuals after MINPACK converges.
    impossible_tolerance = np.finfo(float).tiny  # [-], much below roundoff
    with pytest.raises(RuntimeError, match='Flash scaled residual exceeds'):
        unit.solve_unit(residual_tolerance=impossible_tolerance)
    assert not hasattr(unit, 'LiquidOut')
    if vapor_placeholder is None:
        assert not hasattr(unit, 'VaporOut')
    else:
        # AdiabaticFlash attaches a zero-flow vapor property holder to evaluate
        # energy residuals. A rejected solve must not publish a phase amount.
        assert unit.VaporOut is vapor_placeholder
        assert unit.VaporOut.mole_flow == pytest.approx(0.0)
