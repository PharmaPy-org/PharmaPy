"""Solubility against temperature in van't Hoff form.

    ln(ratio) = intercept + slope / T

``ratio`` is the dissolved solute per solvent at saturation [kg/kg] and ``T`` the
temperature [K]. For an ideal solution ``slope`` is minus the dissolution enthalpy
over the gas constant; here it is an empirical two-constant curve, valid only over
the temperatures it was fitted on.
"""
from __future__ import annotations

import numpy as np

GAS_CONSTANT = 8.314462618  # [J/mol/K], exact in the 2019 SI


def saturated_ratio(temperature, intercept: float, slope: float):
    """Return the solute per solvent at saturation.

    Parameters
    ----------
    temperature : float or numpy.ndarray
        Temperature [K]; positive.
    intercept : float
        Value of ``ln(ratio)`` extrapolated to infinite temperature [-].
    slope : float
        Slope of ``ln(ratio)`` against ``1/T`` [K]; negative when the solubility
        rises with temperature.

    Returns
    -------
    numpy.ndarray
        Solute per solvent at saturation [kg/kg], shape of ``temperature``.
    """
    return np.exp(intercept + slope / np.asarray(temperature, dtype=float))


def saturation_temperature(ratio, intercept: float, slope: float):
    """Return the temperature at which a solution is saturated.

    Parameters
    ----------
    ratio : float or numpy.ndarray
        Dissolved solute per solvent [kg/kg]; positive.
    intercept, slope : float
        Curve constants, as in `saturated_ratio`.

    Returns
    -------
    numpy.ndarray
        Saturation temperature [K], shape of ``ratio``.

    Raises
    ------
    ValueError
        If ``ratio`` is not positive, or the curve never reaches it at a positive
        temperature.
    """
    ratio = np.asarray(ratio, dtype=float)
    if np.any(ratio <= 0.0):
        raise ValueError("The solute ratio must be positive.")
    inverse_temperature = (np.log(ratio) - intercept) / slope  # [1/K]
    if np.any(inverse_temperature <= 0.0):
        raise ValueError("The curve does not reach a ratio of %g at any temperature." % ratio.flat[np.argmax(inverse_temperature <= 0.0)])
    return 1.0 / inverse_temperature


def fit_vant_hoff(temperature, ratio) -> tuple[float, float, float]:
    """Fit the curve to saturated compositions by least squares on ``ln(ratio)``.

    Parameters
    ----------
    temperature : array-like
        Temperatures [K], shape ``(n_points,)``; at least two distinct values.
    ratio : array-like
        Solute per solvent at saturation [kg/kg], shape ``(n_points,)``; positive.

    Returns
    -------
    intercept : float
        Fitted intercept [-].
    slope : float
        Fitted slope [K].
    scatter : float
        Root mean square of the residuals of ``ln(ratio)`` [-]. For small values
        it is the relative scatter of the points about the curve.

    Raises
    ------
    ValueError
        If the shapes disagree, a ratio is not positive, or fewer than two
        distinct temperatures are given.
    """
    temperature = np.asarray(temperature, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if temperature.ndim != 1 or temperature.shape != ratio.shape:
        raise ValueError("Temperatures %s and ratios %s must be equal one-dimensional shapes." % (temperature.shape, ratio.shape))
    if np.any(ratio <= 0.0):
        raise ValueError("Every solute ratio must be positive.")
    if np.unique(temperature).size < 2:
        raise ValueError("At least two distinct temperatures are needed.")
    design = np.column_stack((np.ones_like(temperature), 1.0 / temperature))
    (intercept, slope), *_ = np.linalg.lstsq(design, np.log(ratio), rcond=None)
    residuals = np.log(ratio) - design @ np.array([intercept, slope])  # [-]
    return float(intercept), float(slope), float(np.sqrt(np.mean(residuals ** 2)))


def dissolution_enthalpy(slope: float) -> float:
    """Return the apparent dissolution enthalpy of a fitted curve.

    Parameters
    ----------
    slope : float
        Slope of ``ln(ratio)`` against ``1/T`` [K].

    Returns
    -------
    float
        ``-R * slope`` [J/mol]. It equals the dissolution enthalpy only for an
        ideal solution with the composition as a mole fraction; with a mass ratio
        of a dilute solute it differs from that by the curvature of the
        conversion, so use it to compare curves, not as a thermodynamic value.
    """
    return -GAS_CONSTANT * slope
