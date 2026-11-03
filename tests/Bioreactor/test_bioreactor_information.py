"""Native equilibrium ridge: refinement must expose unresolved information.

At S=1 g/L, X=4.5 gDW/L and D=0.1/h, observations determine only
mu_max*exp(theta/2)=0.2/h. Forward differences create a spurious second
direction; comparing two increments must not qualify it as resolved.
"""

import numpy as np
import pytest

from PharmaPy.Bioreactors import simulate_measurements
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from .test_bioreactor_estimation_handoff import configuration, callback_inputs, SAMPLE_H, HOUR_S

pytestmark = pytest.mark.integration


def test_refined_information_exposes_native_steady_culture_ridge():
    model = configuration(True)
    model['numerical']['relative_tolerance'] = 1e-12  # [-], resolve derivative artifacts
    model['numerical']['absolute_tolerance'] = 1e-15  # [native state units], paired refinement budget
    kwargs = callback_inputs(model, ('rates.mu_max', 'corrections.theta'))
    kwargs['declaration']['fields']['S']['unit'] = 'g/L'
    times = SAMPLE_H * HOUR_S  # [s]
    truth = np.tile([1., 4.5], (len(times), 1))  # [gS/L, gDW/L], exact equilibrium
    fits = []
    for step in (1e-4, 1e-5):  # [1/h, -], a decade of finite-difference refinement
        experiment = Experiment({
            'S': Measurement('S', times, truth[:, 0], uncertainty=.02, units='g/L'),
            'X': Measurement('X', times, truth[:, 1], uncertainty=.02, units='g/L'),
        }, kwargs=kwargs)
        estimator = ParameterEstimation(simulate_measurements, [.2, 0.],
            {'steady': experiment}, output_names=('S', 'X'),
            name_params=['rates.mu_max', 'corrections.theta'], dx_finitediff=step)
        estimator.optimize_fn(method='least_squares', verbose=False)
        fits.append(estimator)
    # Scales are the nominal growth scale and one unit of log correction.
    # 1e-8 alone would admit the spurious direction; refinement supplies the
    # larger observed spectral perturbation floor without a fitted cutoff.
    report = fits[0].assess_information([.2, 1.], 1e-8, fits[1].info_opt['jac'])
    assert report['resolved_rank'] == 1
    assert not report['full_rank']
    assert report['derivative_refinement_checked']
    assert report['relative_derivative_discrepancy'] > 1e-8
