"""Real batch/MSMPR result publication for the #269 conversion contract.

The public three-node asymmetric seed has volume contributions in the ratio
1:48:128, independently calculated using trapezoid support [5, 15, 10] um.
"""
import numpy as np
import pytest

pytest.importorskip('assimulo')
from PharmaPy.Crystallizers import BatchCryst, MSMPR
from test_crystallizer_heat_duty import make_unit

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]


@pytest.mark.parametrize('unit_type,flow', [(BatchCryst, 0.0), (MSMPR, 1e-5)])
def test_native_crystallizer_publishes_normalized_volume_distribution(
        data_path, unit_type, flow):
    """Retrieve normalized fractions through each real solver-backed public path.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    unit_type : type
        Batch or continuous crystallizer.
    flow : float
        Feed volume flow [m**3/s]; zero for a batch.
    """
    unit, _ = make_unit(data_path, unit_type, ramp=0., feed_flow=flow)  # [K/s]
    duration = 0.1  # [s], short inert evolution preserves the known population shape
    unit.solve_unit(runtime=duration, verbose=False)
    expected = np.array([1., 48., 128.]) / 177.  # [-], independently derived volume shares
    fractions = unit.result.vol_distrib  # [-], time by particle-size node
    assert fractions.shape[0] > 1
    np.testing.assert_allclose(fractions, np.tile(expected, (len(fractions), 1)),
                               rtol=1e-10, atol=0)  # [-], inert evolution + conversion roundoff
    np.testing.assert_allclose(fractions.sum(axis=1), 1., rtol=1e-12, atol=0)
