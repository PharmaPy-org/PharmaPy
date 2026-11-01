"""Public annotation resolution on supported Python versions, including 3.9.

Run in both the core environment and the Python 3.9/Assimulo 3.4.3 lane.
Imports are real and solver independent; no version or module substitution is
used to emulate older Python's annotation evaluation.
"""

from typing import get_args, get_type_hints

import pytest


pytestmark = pytest.mark.unit


def test_estimator_verbosity_annotations_import_and_resolve_integer_or_boolean():
    from PharmaPy.ParamEstim import ParameterEstimation

    for method in (ParameterEstimation.optimize_fn,
                   ParameterEstimation.estimate_parameters):
        annotation = get_type_hints(method)["verbose"]
        assert set(get_args(annotation)) == {int, bool}
