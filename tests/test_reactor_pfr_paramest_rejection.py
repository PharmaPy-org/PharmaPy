"""PlugFlowReactor parameter estimation is rejected explicitly (issue #401).

``PlugFlowReactor`` has no defined estimation output contract (which axial
positions and states its callback would return, in which units, and how
sensitivities would be supplied). Its ``paramest_wrapper`` used to inherit
the tank-reactor callback, which called ``solve_unit(eval_sens=...)`` that
the PFR does not accept, so every objective evaluation raised ``TypeError``
after ``SimulationExec.SetParamEstimation`` had accepted the unit. The PFR
now declares the missing contract through the generic
``validate_param_estimation`` hook: ``SetParamEstimation`` calls it before
any setup for every input form, and the PFR's ``paramest_wrapper`` raises
the same ``PharmaPyNonImplementedError`` before changing the reactor.

Fixture: the committed PFR case of ``test_reactor_modes.py``
(``pfr_test_constructor_kwargs.json`` and ``pfr_test_pure_comp.json`` with
the two-reaction kinetics of ``_load_pfr_config``), built from real
collaborators. The rejection never reaches the solver, so each check runs in
the core lane and again in the Assimulo lane, where the same fixture is also
integrated to show that the rejection concerns estimation, not a broken
reactor. A single short PFR solve per Assimulo case.
"""

import numpy as np
import pytest

from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import PlugFlowReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater
from test_reactor_modes import _load_pfr_config

# The three requested times of the issue #401 reproduction.
TIMES_S = np.array([0.0, 10.0, 20.0])  # [s]
# Placeholder observations [mol/L] of the three reacting species A, B, C;
# their values are never compared because setup must fail first.
OBSERVATIONS = np.zeros((len(TIMES_S), 3))  # [mol/L]
MEASURED = [0, 1, 2]  # A, B, C columns
# Placeholder absorbance spectra [-] for the MultipleCurveResolution path:
# one spectrum of four wavelengths per requested time.
SPECTRA = np.zeros((len(TIMES_S), 4))  # [-]
REJECTION = (r"^PlugFlowReactor does not support parameter estimation: .*"
             r"issue #401.*BatchReactor")
LANES = [
    pytest.param('core', marks=pytest.mark.unit, id='core'),
    pytest.param('assimulo',
                 marks=[pytest.mark.assimulo, pytest.mark.integration],
                 id='assimulo'),
]


def _pfr(data_path, return_sens):
    """Build the committed PFR fixture from real collaborators.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Session fixture with the ``'integration'`` data directory.
    return_sens : bool
        ``return_sens`` of the reactor.

    Returns
    -------
    PlugFlowReactor
        Non-isothermal 50-cell tube of 0.0254 m diameter charged with
        solvent at 298.15 K, fed A and B at 0.15 mol/L with a 1800 s
        residence time, running A + B -> C and B -> C.
    """
    config, datapath = _load_pfr_config(data_path)
    reactor = PlugFlowReactor(**config['reactor'], return_sens=return_sens)
    reactor.Inlet = LiquidStream(datapath, **config['inlet'])
    reactor.Phases = LiquidPhase(datapath, **config['phase'])
    reactor.Kinetics = RxnKinetics(**config['kinetics'])
    reactor.Utility = CoolingWater(**config['utility'])
    return reactor


def _enter_lane(lane):
    """Skip an Assimulo-lane case when the solver stack is absent.

    Parameters
    ----------
    lane : {'core', 'assimulo'}
        Test lane of the parametrized case.
    """
    if lane == 'assimulo':
        pytest.importorskip('assimulo')


def _check_fixture_integrates(lane, data_path):
    """Integrate the same PFR fixture in the Assimulo lane.

    Parameters
    ----------
    lane : {'core', 'assimulo'}
        Test lane of the parametrized case; the core lane has no solver.
    data_path : dict of pathlib.Path
        Session fixture with the ``'integration'`` data directory.
    """
    if lane != 'assimulo':
        return
    reactor = _pfr(data_path, return_sens=False)
    # Times [s]; per cell concentrations [mol/L] and temperature [K].
    time, states = reactor.solve_unit(time_grid=TIMES_S, verbose=False)
    np.testing.assert_array_equal(time, TIMES_S)
    assert np.all(np.isfinite(states))


def _estimation_inputs(form):
    """Return ``SetParamEstimation`` arguments of one input form.

    Parameters
    ----------
    form : {'legacy', 'experiment', 'spectra'}
        Legacy time and observation arrays, an ``Experiment`` object, or
        spectra fitted with ``MultipleCurveResolution``.

    Returns
    -------
    dict
        Keyword arguments of ``SetParamEstimation``: times [s],
        concentrations [mol/L] or absorbances [-].
    """
    if form == 'legacy':
        return {'x_data': TIMES_S, 'y_data': OBSERVATIONS,
                'measured_ind': MEASURED}
    if form == 'experiment':
        return {'x_data': Experiment(
            {'c_A': Measurement(0, TIMES_S, OBSERVATIONS[:, 0])})}
    return {'x_data': TIMES_S, 'y_spectra': SPECTRA, 'fit_spectra': True,
            'measured_ind': MEASURED}


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('form', ['legacy', 'experiment', 'spectra'])
def test_set_param_estimation_rejects_pfr_at_setup(lane, form, data_path):
    _enter_lane(lane)
    simulation = SimulationExec(_load_pfr_config(data_path)[1], {'PFR': []})
    simulation.PFR = _pfr(data_path, return_sens=False)
    # k of A + B -> C [L/mol/s] and of B -> C [1/s], then Ea [J/mol].
    seed = simulation.PFR.Kinetics.concat_params().copy()
    with pytest.raises(PharmaPyNonImplementedError, match=REJECTION):
        simulation.SetParamEstimation(**_estimation_inputs(form))
    # Setup stopped before building an estimator or seeding the kinetics.
    assert not hasattr(simulation, 'ParamInst')
    np.testing.assert_array_equal(simulation.PFR.Kinetics.concat_params(),
                                  seed)
    _check_fixture_integrates(lane, data_path)


@pytest.mark.parametrize('lane', LANES)
@pytest.mark.parametrize('return_sens', [False, True])
def test_pfr_paramest_wrapper_rejects_estimation(lane, return_sens,
                                                 data_path):
    _enter_lane(lane)
    reactor = _pfr(data_path, return_sens)
    # k of A + B -> C [L/mol/s] and of B -> C [1/s], then Ea [J/mol].
    seed = reactor.Kinetics.concat_params().copy()
    trial = 2 * seed  # same units; parameters a callback would have set
    with pytest.raises(PharmaPyNonImplementedError, match=REJECTION):
        reactor.paramest_wrapper(trial, TIMES_S)
    # The callback rejects estimation before setting any parameter.
    np.testing.assert_array_equal(reactor.Kinetics.concat_params(), seed)
    _check_fixture_integrates(lane, data_path)
