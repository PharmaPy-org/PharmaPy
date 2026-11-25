"""Derivative checks leave a stateful reactor at its accepted parameters.

Scope: ``ParameterEstimation.check_derivatives`` on a ``SimulationExec``
estimator, whose model callback is the ``paramest_wrapper`` of a
``BatchReactor``: every evaluation sets the reactor's kinetic parameters and
replaces its ``result``. After the check the reactor must hold the accepted
parameters (``params_convg``) and the result of a solve at them again, as
after ``optimize_fn``, which refreshes its accepted point the same way.

Fixture: the jacketed non-isothermal A --> B batch reactor of
``test_reactor_paramest_noniso_sensitivities`` (rate constant estimated,
activation energy fixed), with CVodes sensitivities and with the
estimator's forward differences. A few short reactor solves and one small
fit per case; requires the optional Assimulo stack.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

import test_reactor_paramest_noniso_sensitivities as noniso

REACTION = 'A --> B'  # the issue case of the fixture module


def result_arrays(reactor):
    """Copy the array fields of a reactor's last result.

    Parameters
    ----------
    reactor : BatchReactor
        Reactor after a solve.

    Returns
    -------
    dict
        Field name -> copy of the array (time [s], concentrations
        [mol/L], temperatures [K], ...).
    """
    return {name: np.array(value, copy=True)
            for name, value in vars(reactor.result).items()
            if isinstance(value, np.ndarray)}


@pytest.mark.parametrize('return_sens, source',
                         [(True, 'model'), (False, 'finite_difference')])
def test_reactor_ends_at_the_accepted_parameters(return_sens, source):
    simulation = noniso._estimation(REACTION, return_sens)
    estimator = simulation.ParamInst
    estimator.optimize_fn(verbose=False)
    reactor = simulation.R01
    # [1/s], [J/mol]: rate constant and the fixed activation energy
    params = np.array(reactor.Kinetics.concat_params(), copy=True)
    np.testing.assert_array_equal(params[:1], estimator.params_convg)
    before = result_arrays(reactor)
    assert before

    # The module's central-difference step for this reactor; any step
    # evaluates the reactor away from the accepted rate.
    check = estimator.check_derivatives(
        rel_step=noniso.CENTRAL_RELATIVE_STEP)

    assert check.jacobian_source == source
    np.testing.assert_array_equal(reactor.Kinetics.concat_params(), params)
    after = result_arrays(reactor)
    assert set(after) == set(before)
    for name, values in before.items():
        np.testing.assert_array_equal(after[name], values, err_msg=name)
