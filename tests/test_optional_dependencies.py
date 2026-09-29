import numpy as np
import pytest


def test_commons_integration_imports_without_assimulo():
    from PharmaPy import Commons

    time = np.array([0.0, 1.0, 2.0])
    states = np.array([0.0, 1.0, 2.0])

    integral = Commons.integration(states, time)

    assert integral.shape == (1,)
    assert integral[0] == pytest.approx(2.0)
