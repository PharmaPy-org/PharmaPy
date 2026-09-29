"""Aggregation contracts for the refactored MixedPhase and MixedStream.

``MixedPhase.__getattr__`` sums a name listed in ``EXTENSIVE_PROPERTIES`` and
mass-weights everything else, so a misclassified name fails silently.

The fixture is deliberately asymmetric: a 3 kg liquid against a 1 kg solid, so
a sum (4 kg) and a mass-weighted average (2.5 kg) cannot be confused, and
phase temperatures of 310 K and 290 K give a weighted mean of 305 K that no
unweighted mean reproduces.
"""

import os

import numpy as np
import pytest

from PharmaPy.MixedPhases_Refactored import MixedPhase, MixedStream
from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
from PharmaPy.Streams_Refactored import LiquidStream, SolidStream

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

LIQUID_MASS = 3.0  # [kg]
SOLID_MASS = 1.0  # [kg]
LIQUID_MASS_FRAC = [0.4, 0.6, 0.0, 0.0, 0.0]  # [-]
SOLID_MASS_FRAC = [0.0, 0.0, 1.0, 0.0, 0.0]  # [-]
LIQUID_TEMP = 310.0  # [K]
SOLID_TEMP = 290.0  # [K]

LIQUID_MASS_FLOW = 3.0  # [kg/s]
SOLID_MASS_FLOW = 1.0  # [kg/s]

# Aggregates are compared against sums of the same floating point members, so
# only accumulated roundoff is at stake.
AGGREGATE_RTOL = 1e-12  # [-]


def _liquid():
    """Build the liquid member of the fixture.

    Returns
    -------
    LiquidPhase
        ``LIQUID_MASS`` [kg] of the A/B mixture at ``LIQUID_TEMP`` [K].
    """
    return LiquidPhase(
        DATA_PATH, mass=LIQUID_MASS, mass_frac=LIQUID_MASS_FRAC,
        temp=LIQUID_TEMP
    )


def _solid():
    """Build the solid member of the fixture.

    Returns
    -------
    SolidPhase
        ``SOLID_MASS`` [kg] of pure species C at ``SOLID_TEMP`` [K].
    """
    return SolidPhase(
        DATA_PATH, mass=SOLID_MASS, mass_frac=SOLID_MASS_FRAC,
        temp=SOLID_TEMP
    )


def _streams():
    """Build the two-phase feed members.

    Returns
    -------
    tuple of (LiquidStream, SolidStream)
        Liquid and solid streams at ``LIQUID_MASS_FLOW`` and
        ``SOLID_MASS_FLOW`` [kg/s].
    """
    liquid = LiquidStream(
        DATA_PATH, mass_flow=LIQUID_MASS_FLOW, mass_frac=LIQUID_MASS_FRAC
    )
    solid = SolidStream(
        DATA_PATH, mass_flow=SOLID_MASS_FLOW, mass_frac=SOLID_MASS_FRAC
    )
    return liquid, solid


def test_mixed_phase_sums_extensive_amounts():
    """Holdup quantities add across the member phases."""
    liquid, solid = _liquid(), _solid()
    mixed = MixedPhase([liquid, solid])

    np.testing.assert_allclose(
        mixed.mass, LIQUID_MASS + SOLID_MASS, rtol=AGGREGATE_RTOL
    )
    np.testing.assert_allclose(
        mixed.vol, liquid.vol + solid.vol, rtol=AGGREGATE_RTOL
    )
    np.testing.assert_allclose(
        mixed.moles, liquid.moles + solid.moles, rtol=AGGREGATE_RTOL
    )


def test_mixed_phase_mass_weights_intensive_properties():
    """Temperature is a mass-weighted mean, not a sum."""
    mixed = MixedPhase([_liquid(), _solid()])

    expected = (
        LIQUID_MASS * LIQUID_TEMP + SOLID_MASS * SOLID_TEMP
    ) / (LIQUID_MASS + SOLID_MASS)  # [K], equals 305.0
    np.testing.assert_allclose(mixed.temp, expected, rtol=AGGREGATE_RTOL)


def test_mixed_stream_sums_flow_rates():
    """Flow rates add across the phases of a multiphase stream.

    A mass-weighted average reports 2.5 kg/s for a 3 kg/s liquid and a
    1 kg/s solid, understating the true 4 kg/s throughput.
    """
    liquid, solid = _streams()
    stream = MixedStream([liquid, solid])

    np.testing.assert_allclose(
        stream.mass_flow,
        LIQUID_MASS_FLOW + SOLID_MASS_FLOW,
        rtol=AGGREGATE_RTOL,
    )
    np.testing.assert_allclose(
        stream.vol_flow,
        liquid.vol_flow + solid.vol_flow,
        rtol=AGGREGATE_RTOL,
    )
    np.testing.assert_allclose(
        stream.mole_flow,
        liquid.mole_flow + solid.mole_flow,
        rtol=AGGREGATE_RTOL,
    )


def test_mixed_stream_sums_amount_aliases():
    """A stream still adds the amount names that alias its flow rates.

    Stream ``mass``, ``vol`` and ``moles`` read the same values as the flow
    rates, so the stream set must extend the phase set rather than replace it.
    """
    liquid, solid = _streams()
    stream = MixedStream([liquid, solid])

    np.testing.assert_allclose(
        stream.mass, LIQUID_MASS_FLOW + SOLID_MASS_FLOW, rtol=AGGREGATE_RTOL
    )
    np.testing.assert_allclose(
        stream.vol, liquid.vol_flow + solid.vol_flow, rtol=AGGREGATE_RTOL
    )
    np.testing.assert_allclose(
        stream.moles, liquid.mole_flow + solid.mole_flow, rtol=AGGREGATE_RTOL
    )


def test_stream_built_by_conversion_sums_flow_rates():
    """``to_stream`` yields a stream that follows the stream rules.

    The conversion copies the instance and reassigns ``__class__`` without
    running ``MixedStream.__init__``.
    """
    liquid, solid = _liquid(), _solid()
    stream = MixedPhase([liquid, solid]).to_stream()

    assert isinstance(stream, MixedStream)
    np.testing.assert_allclose(
        stream.mass_flow, LIQUID_MASS + SOLID_MASS, rtol=AGGREGATE_RTOL
    )
    np.testing.assert_allclose(
        stream.vol_flow, liquid.vol + solid.vol, rtol=AGGREGATE_RTOL
    )
