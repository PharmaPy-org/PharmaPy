"""Regression tests for extractor operating-mode flow/amount routing.

Continuous extraction should consume and return mole-flow rates [mol/s].
Batch extraction should keep material amounts [mol].
"""

import numpy as np
import pytest

from PharmaPy.Extractors import BatchExtractor, ContinuousExtractor
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


def test_continuous_extractor_uses_mole_flow_and_stream_outlets(data_path):
    """Continuous extractor routes inlet and outlet quantities as [mol/s]."""
    database_path = data_path["flowsheet"] / "compound_database.json"
    inlet_flow = 8.0  # [mol/s]
    inlet_mole_fraction = np.array([0.25, 0.75, 0.0, 0.0, 0.0])  # [-]
    inlet = LiquidStream(
        str(database_path),
        mole_flow=inlet_flow,
        mole_frac=inlet_mole_fraction,
        temp=298.15,  # [K]
        pres=101325.0,  # [Pa]
    )

    extractor = ContinuousExtractor()
    extractor.Inlet = inlet

    assert extractor.oper_mode == "Continuous"
    assert extractor.in_flow == pytest.approx(inlet_flow)

    phase_fraction = 0.25  # [-], deliberately chosen rather than solved equilibrium
    # Components A and B form the binary split; the trailing C, D, and solvent
    # entries remain absent from both deliberately chosen outlet compositions.
    liquid_a_mole_fraction = np.array([0.4, 0.6, 0.0, 0.0, 0.0])  # [-]
    liquid_b_mole_fraction = np.array([0.1, 0.9, 0.0, 0.0, 0.0])  # [-]
    extractor.retrieve_results((
        phase_fraction,
        liquid_a_mole_fraction,
        liquid_b_mole_fraction,
        {"error": 0.0, "num_iter": 1},
    ))

    outlets = [extractor.Liquid_2, extractor.Liquid_3]
    assert all(type(outlet) is LiquidStream for outlet in outlets)
    expected_heavy_flow = phase_fraction * inlet_flow  # [mol/s]
    expected_light_flow = inlet_flow - expected_heavy_flow  # [mol/s]
    assert extractor.Liquid_2.mole_flow == pytest.approx(expected_heavy_flow)
    assert extractor.Liquid_3.mole_flow == pytest.approx(expected_light_flow)
    assert extractor.Liquid_2.getDensity(basis="mole") > (
        extractor.Liquid_3.getDensity(basis="mole")
    )
    assert sum(outlet.mole_flow for outlet in outlets) == pytest.approx(
        inlet_flow
    )


def test_batch_extractor_keeps_batch_amount_semantics(data_path):
    """Batch extractor keeps inlet amount semantics as [mol]."""
    database_path = data_path["flowsheet"] / "compound_database.json"
    inlet_moles = 3.0  # [mol]
    inlet = LiquidPhase(
        str(database_path),
        moles=inlet_moles,
        mole_frac=np.array([0.25, 0.75, 0.0, 0.0, 0.0]),  # [-]
        temp=298.15,  # [K]
        pres=101325.0,  # [Pa]
    )
    extractor = BatchExtractor()
    extractor.Phases = inlet

    assert extractor.oper_mode == "Batch"
    assert extractor.in_flow == pytest.approx(inlet_moles)


@pytest.mark.parametrize("extractor_cls", [ContinuousExtractor, BatchExtractor])
@pytest.mark.parametrize("gamma_method", ["ideal", "UNIFAC", "UNIQUAC"])
def test_extractors_accept_documented_activity_models(extractor_cls,
                                                     gamma_method):
    """Extractor constructors accept the documented selector spellings."""
    extractor = extractor_cls(gamma_method=gamma_method)

    assert extractor.gamma_method == gamma_method


@pytest.mark.parametrize("extractor_cls", [ContinuousExtractor, BatchExtractor])
def test_extractors_reject_unknown_activity_model(extractor_cls):
    """Extractor constructors reject unknown activity-model selectors."""
    # Lowercase ``uniquac`` is deliberate: the documented selector is the
    # exact-cased ``UNIQUAC`` branch used by Phases.getActivityCoeff.
    with pytest.raises(ValueError, match="gamma_method must be one of"):
        extractor_cls(gamma_method="uniquac")
