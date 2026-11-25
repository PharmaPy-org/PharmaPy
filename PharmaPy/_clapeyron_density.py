"""Provisional single-phase density queries for two Clapeyron models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from math import fsum, isfinite
from numbers import Real
from typing import Literal, NamedTuple

import numpy as np


class Density(NamedTuple):
    """Scalar densities for the same state and volume root.

    Attributes
    ----------
    mass_density : float
        Mass per unit volume [kg/m³].
    molar_density : float
        Amount of substance per unit volume [mol/m³].
    """

    mass_density: float  # [kg/m³]
    molar_density: float  # [mol/m³]


def _positive_scalar(value: float, name: str) -> float:
    """Validate one finite, positive real scalar and preserve its units.

    Parameters
    ----------
    value : float
        Temperature [K] or absolute pressure [Pa].
    name : str
        Quantity name for the error message.

    Returns
    -------
    float
        Validated value in the same units.

    Raises
    ------
    ValueError
        If the value is non-scalar, non-real, non-finite, or non-positive.
    """
    error_message = f"{name} must be a finite, positive real scalar."
    # NumPy duration scalars can satisfy Real but carry time units.
    if isinstance(value, (bool, np.timedelta64)) or not isinstance(value, Real):
        raise ValueError(error_message)
    try:
        scalar = float(value)  # [K] or [Pa], same as value
    except OverflowError as error:
        raise ValueError(error_message) from error
    if not isfinite(scalar) or scalar <= 0:
        raise ValueError(error_message)
    return scalar


def single_phase_density(
    model: Literal["IAPWS95", "PR"],
    components: Sequence[str],
    temperature: float,
    pressure: float,
    mole_fractions: Sequence[float] | np.ndarray,
    *,
    phase: Literal["liquid", "vapor"],
) -> Density:
    """Evaluate the mass and molar density of one specified fluid phase.

    Parameters
    ----------
    model : {"IAPWS95", "PR"}
        Clapeyron equation of state. The prototype supports the IAPWS-95
        water reference and the default Peng–Robinson methane/ethane model.
    components : sequence of str
        ("water",) for IAPWS95 or ("methane", "ethane") in either order
        for PR. The sequence defines the order of `mole_fractions`.
    temperature : float
        Scalar absolute temperature [K].
    pressure : float
        Scalar absolute pressure [Pa].
    mole_fractions : array_like of float
        One-dimensional mole fractions [-] of shape `(len(components),)`.
        Entries must be finite and nonnegative and sum to one within one
        machine epsilon of the input precision (at least float64 epsilon).
        Fractions are passed in the supplied order without normalization.
    phase : {"liquid", "vapor"}
        Root requested from Clapeyron, which uses the specified phase to
        initialize or select the volume root.

    Returns
    -------
    Density
        Two Python floats on the mass [kg/m³] and molar [mol/m³] bases.

    Raises
    ------
    ValueError
        If the model, components, phase, scalar state, or mole fractions
        are invalid.
    ImportError
        If the optional Python wrapper is absent.
    RuntimeError
        If Clapeyron returns a non-finite or non-positive density.

    Notes
    -----
    Use the model within its physical domain and supply a homogeneous phase
    composition. A successful volume calculation alone cannot establish
    phase stability or the composition of a phase in a multiphase system.
    Julia initialization and backend errors propagate to the caller.
    """
    if not isinstance(model, str) or model not in ("IAPWS95", "PR"):
        raise ValueError("model must be 'IAPWS95' or 'PR'.")
    component_error = "components must be an ordered sequence of species names."
    if isinstance(components, (str, bytes, Mapping, Set)):
        raise ValueError(component_error)
    try:
        component_order = tuple(components)
    except TypeError as error:
        raise ValueError(component_error) from error
    if not all(isinstance(component, str) for component in component_order):
        raise ValueError(component_error)
    if model == "IAPWS95" and component_order != ("water",):
        raise ValueError("IAPWS95 requires components ('water',).")
    if model == "PR" and component_order not in (
        ("methane", "ethane"), ("ethane", "methane"),
    ):
        raise ValueError("PR requires methane and ethane in either order.")
    if not isinstance(phase, str) or phase not in ("liquid", "vapor"):
        raise ValueError("phase must be 'liquid' or 'vapor'.")

    temperature_kelvin = _positive_scalar(temperature, "temperature [K]")  # [K]
    pressure_pascal = _positive_scalar(pressure, "absolute pressure [Pa]")  # [Pa]
    if np.ma.is_masked(mole_fractions):
        raise ValueError("mole_fractions cannot include masked values.")
    fractions = np.asarray(mole_fractions)  # [-], ordered on a molar basis
    if fractions.shape != (len(component_order),) or fractions.dtype.kind not in "fiu":
        raise ValueError("mole_fractions must contain one real value per component.")
    if not np.all(np.isfinite(fractions) & (fractions >= 0) & (fractions <= 1)):
        raise ValueError("mole_fractions must contain finite values in [0, 1].")
    # One epsilon covers rounding of a unit sum in the source precision.
    sum_tolerance = np.finfo(float).eps  # [-]
    if fractions.dtype.kind == "f":
        sum_tolerance = max(sum_tolerance, np.finfo(fractions.dtype).eps)  # [-]
    if abs(fsum(fractions) - 1) > sum_tolerance:
        raise ValueError("mole_fractions must sum to one.")
    fractions_float64 = np.ascontiguousarray(fractions, dtype=float)  # [-], molar basis

    try:
        import pyclapeyron
    except ModuleNotFoundError as error:
        if error.name != "pyclapeyron":
            raise
        raise ImportError(
            "Clapeyron density requires pyclapeyron and a configured Julia "
            "runtime. See doc/online_docs/examples/clapeyron_density/README.md."
        ) from error

    if model == "IAPWS95":
        equation = pyclapeyron.IAPWS95()
    else:
        equation = pyclapeyron.PR([str(component) for component in component_order])
    root = pyclapeyron.jl.Symbol(str(phase))
    state = (equation, pressure_pascal, temperature_kelvin, fractions_float64)
    mass_density = float(pyclapeyron.mass_density(*state, phase=root))  # [kg/m³]
    molar_density = float(pyclapeyron.molar_density(*state, phase=root))  # [mol/m³]
    if not all(isfinite(value) and value > 0 for value in (mass_density, molar_density)):
        raise RuntimeError(f"{model} returned an invalid density for the {phase} root.")
    return Density(mass_density, molar_density)
