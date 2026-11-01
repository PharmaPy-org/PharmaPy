"""Solver-independent exact sample mapping and actionable omission diagnostics.

Reported grids are explicit inputs to the production selection helper. Real
CVode handoffs are covered in test_reactor_paramest_requested_rows.py; these
tests require no optional solver and guarantee omission coverage on all versions.
"""

import numpy as np
import pytest

from PharmaPy.Reactors import _requested_rows


pytestmark = pytest.mark.unit


@pytest.mark.parametrize('reported,span', [
    ([0., 1., 3.], '3 times from 0.0 s to 3.0 s'),
    ([], 'no times'),
])
def test_missing_output_has_version_independent_times_and_recovery_guidance(reported, span):
    # [s], a missing interior point or a genuinely empty solver history.
    with pytest.raises(ValueError) as error:
        _requested_rows(np.asarray(reported), [2.])
    message = str(error.value)
    assert message.startswith(
        'Requested time 2.0 s is absent from the reactor solution, '
        f'which reports {span}.')
    assert "run_args={'sundials_opts': {'report_continuously': True}}" in message
    assert 'wrapper_kwargs in SimulationExec' in message
    assert 'terminating state event' in message


def test_exact_multiscale_mapping_preserves_order_and_replicates():
    reported = np.array([0., 1e-12, 2e-12, 2e-12, 3600.])  # [s]
    requested = np.array([2e-12, 0., 2e-12, 2e-12, 1e-12, 0., 3600.])  # [s]
    np.testing.assert_array_equal(_requested_rows(reported, requested),
                                  [2, 0, 3, 3, 1, 0, 4])


def test_neighbouring_float_is_not_a_reported_sample():
    reported = np.array([0., 1., 3600.])  # [s]
    requested = np.nextafter(1., 2.)  # [s], one representable step after 1 s
    with pytest.raises(ValueError, match='Requested time 1.0000000000000002 s'):
        _requested_rows(reported, [requested])
