"""collector configuration and static moment-feed regressions.

Real phases and crystallizers reach ODE construction in the core lane. Marked
Assimulo tests integrate frozen populations and constant liquid feeds.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/157
https://github.com/PharmaPy-org/PharmaPy/issues/224
https://github.com/PharmaPy-org/PharmaPy/issues/259
"""

from copy import deepcopy

import numpy as np
import pytest

import PharmaPy.Containers as containers
from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream, SolidStream

TEMPERATURE = 310.0  # [K], synthetic equal-temperature feed and seed
FLOW = 0.02  # [m**3/s], finite synthetic slurry feed
MASS_FLOW = 2.0  # [kg/s], finite synthetic liquid feed
FRACTIONS = np.array([0.1, 0.2, 0.3, 0.15, 0.25])  # [-], asymmetric species
SOLID_FRACTIONS = [0, 0, 1, 0, 0]  # [-], pure C distinguishes target index
KV = 0.5  # [-], non-unit crystal shape
NUMBER_DENSITY = 2e8  # [#/m**3], synthetic monodisperse population
SIZE = 20e-6  # [m], monodisperse crystal length
MOMENTS = NUMBER_DENSITY * SIZE ** np.arange(4)  # [m**n/m**3], n = 0,...,3
DURATION = 0.01  # [s], short interval isolates accumulation from long dynamics
RTOL = 1e-12  # [-], float64 roundoff allowance for direct algebra
SOLVER_RTOL = 1e-7  # [-], reused from test_crystallizer_moment_inventory.test_msmpr_moment_output_collects_without_grid
INTEGRATION_OPTIONS = {'rtol': 1e-9, 'atol': 1e-10}
# Relative [-] and absolute [state units] tolerances from test_crystallizer_moment_inventory.py collector
# regression: resolve the sqrt(eps) seed volume below the default absolute floor.


class ProblemCaptured(Exception):
    """Stop immediately before optional ODE construction."""


def make_collector(data_path, mode, connected=True):
    """Construct a liquid or frozen crystal collector from real streams.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    mode : str
        Liquid, moments, or FVM population representation.
    connected : bool, optional
        Supply constant upstream metadata when True.

    Returns
    -------
    DynamicCollector
        Collector at TEMPERATURE [K] with the declared constant feed.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    liquid = LiquidStream(path, temp=TEMPERATURE, mass_flow=MASS_FLOW,
                          mass_frac=FRACTIONS)
    collector = DynamicCollector()
    if mode == 'liquid':
        collector.Inlet = liquid
        return collector
    solid = SolidStream(path, temp=TEMPERATURE, kv=KV,
                        mass_frac=SOLID_FRACTIONS)
    if mode == 'moments':
        inlet = SlurryStream(vol_flow=FLOW, moments=MOMENTS.copy())
    else:
        grid = np.array([10.0, 20.0, 40.0])  # [um], unequal bin intervals
        distribution = np.array([1e7, 2e7, 1e7])  # [#/m**3/um], asymmetric seed
        solid = SolidStream(path, temp=TEMPERATURE, kv=KV,
                            mass_frac=SOLID_FRACTIONS,
                            x_distrib=grid, distrib=distribution)
        inlet = SlurryStream(vol_flow=FLOW, x_distrib=grid,
                             distrib=distribution)
    inlet.Phases = (liquid, solid)
    if connected:
        population = {'mu_n': inlet.moments} if mode == 'moments' else {
            'distrib': inlet.distrib}  # [m**n/m**3] or [#/m**3/um]
        inlet.y_inlet = dict(mass_conc=liquid.mass_conc, vol_flow=FLOW,
                             temp=TEMPERATURE, **population)
        inlet.y_upstream = inlet.y_inlet
        inlet.time_upstream = None
    collector.Inlet = inlet
    collector.KinCryst = CrystKinetics(coeff_solub=[2000.0])
    # [kg/m**3], synthetic solubility above feed; default kinetic rates are zero
    collector.kwargs_cryst = dict(target_ind=2, target_comp='C',
                                  num_interp_points=7)
    # Seven interpolation points [-] deliberately differ from collector default.
    return collector


@pytest.fixture
def capture_problem(monkeypatch):
    """Capture real initial states and times at the optional solver boundary.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Test-local constructor interception.

    Returns
    -------
    list
        Captured (model, states, time) tuples with model-native state units and
        time [s]. Crystal states use [um**n] or [#/um], [kg/m**3], [m**3], [K];
        liquid states use mass fractions [-], mass [kg], and temperature [K].
    """
    captured = []

    def liquid_problem(rhs, y0, t0):
        """Capture the liquid ODE state before integration.

        Parameters
        ----------
        rhs : callable
            Bound collector balance method.
        y0 : ndarray
            Mass fractions [-], mass [kg], and temperature [K].
        t0 : float
            Initial time [s].

        Raises
        ------
        ProblemCaptured
            Always, after recording the actual solver handoff.
        """
        captured.append((rhs.__self__, y0.copy(), t0))
        raise ProblemCaptured

    def crystal_problem(self, eval_sens, states_init, params, jac_v_prod):
        """Capture the real crystallizer state before integration.

        Parameters
        ----------
        self : SemibatchCryst
            Real delegated model and seed phases.
        eval_sens, jac_v_prod : bool
            Sensitivity and Jacobian options.
        states_init : ndarray
            Population [um**n] or [#/um], concentrations [kg/m**3], liquid
            volume [m**3], and temperature [K].
        params : ndarray
            Native kinetic parameters (unused with frozen kinetics).

        Raises
        ------
        ProblemCaptured
            Always, after recording the actual solver handoff.
        """
        captured.append((self, states_init.copy(), self.elapsed_time))
        raise ProblemCaptured

    monkeypatch.setattr(containers, 'Explicit_Problem', liquid_problem)
    monkeypatch.setattr(SemibatchCryst, 'set_ode_problem', crystal_problem)
    return captured


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['liquid', 'moments', 'fvm'])
def test_two_consecutive_setups(data_path, capture_problem, mode):
    """Reach ODE construction twice without consuming either selector.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    capture_problem : list
        Real model, native initial states, and start time [s] at each handoff.
    mode : str
        Liquid, moment, or FVM population representation.
    """
    collector = make_collector(data_path, mode)
    selectors = deepcopy((collector.names_states_in, collector.names_states_out))
    for _ in range(2):
        with pytest.raises(ProblemCaptured):
            collector.solve_unit(runtime=DURATION, verbose=False)
    assert len(capture_problem) == 2
    first, second = capture_problem
    np.testing.assert_allclose(second[1], first[1], rtol=RTOL, atol=0)
    assert first[2] == second[2] == 0
    assert (collector.names_states_in, collector.names_states_out) == selectors
    # Retained selectors must still support the public inlet setter.
    collector.Inlet = collector.Inlet


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['moments', 'fvm'])
def test_crystallizer_setup_preserves_caller_kwargs(data_path, capture_problem, mode):
    """Keep target selection and caller-owned constructor options reusable.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    capture_problem : list
        Real delegated models and their solver handoffs.
    mode : str
        Moment or FVM population representation.
    """
    collector = make_collector(data_path, mode)
    caller_kwargs = collector.kwargs_cryst
    expected = deepcopy(caller_kwargs)
    with pytest.raises(ProblemCaptured):
        collector.solve_unit(runtime=DURATION, verbose=False)
    assert caller_kwargs == expected
    model = capture_problem[0][0]
    assert model.target_ind == expected['target_ind']
    assert model.num_interp_points == collector.num_interp_points
    # SolidPhase replaces absent species with its established epsilon floor.
    expected_fractions = np.maximum(SOLID_FRACTIONS, np.finfo(float).eps)  # [-]
    np.testing.assert_allclose(model.Solid_1.mass_frac, expected_fractions,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mu_n', 'mass_conc'])
@pytest.mark.parametrize('time', [0.0, np.array([0.0]), np.array([0.0, 1.0, 2.0])])
def test_static_moment_input_values(data_path, field, time):
    """Resolve static moment and concentration fields on their declared axes.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    field : str
        Moment [m**n/m**3] or concentration [kg/m**3] field to verify.
    time : float or ndarray
        Scalar, single-element, or three-element evaluation times [s].
    """
    collector = make_collector(data_path, 'moments', connected=False)
    collector.states_in_dict = {'Inlet': {
        'mass_conc': len(FRACTIONS), 'vol_flow': 1, 'temp': 1, 'mu_n': len(MOMENTS)}}
    inputs = collector.get_inputs_new(time)['Inlet']
    expected = MOMENTS if field == 'mu_n' else collector.Inlet.Liquid_1.mass_conc
    # [m**n/m**3] or [kg/m**3]; input values must retain their physical basis.
    if np.size(time) > 1:
        expected = np.tile(expected, (np.size(time), 1))
    np.testing.assert_allclose(inputs[field], expected, rtol=RTOL, atol=0)
    assert not hasattr(collector.Inlet, 'mu_n')
    assert not hasattr(collector.Inlet, 'mass_conc')


@pytest.mark.unit
def test_static_moment_seed_reaches_crystallizer(data_path, capture_problem):
    """Initialize real phases from static feed composition and SI moments.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    capture_problem : list
        Real crystallizer phases captured at ODE construction.
    """
    collector = make_collector(data_path, 'moments', connected=False)
    with pytest.raises(ProblemCaptured):
        collector.solve_unit(runtime=DURATION, verbose=False)
    model = capture_problem[0][0]
    seed_volume = np.sqrt(np.finfo(float).eps)  # [m**3], established collector seed
    np.testing.assert_allclose(model.Solid_1.moments, MOMENTS * seed_volume,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(model.Liquid_1.mass_frac, FRACTIONS,
                               rtol=RTOL, atol=0)
    assert model.Liquid_1.vol == pytest.approx(
        seed_volume * (1 - KV * NUMBER_DENSITY * SIZE**3), rel=RTOL, abs=0)


@pytest.mark.assimulo
@pytest.mark.parametrize('mode', ['liquid', 'moments', 'fvm'])
def test_two_complete_solves(data_path, mode):
    """Reuse configuration while retaining the existing per-call seed policy.

    Each call constructs a fresh inlet-derived seed at the previous end time.
    Inventory continuation is not part of #259's configuration repair; changing
    that policy would also touch PR #207's liquid initialization hunks.

    For the nearly solid-free slurry fixture, the temperature assertion is a
    provisional deviation bound, not an energy-balance invariant. Issue #265
    tracks the inherited SemibatchCryst.energy_balances accumulation defect:
    inlet liquid density is combined with slurry-basis enthalpy, producing a
    converged temperature residual of about 2e-6 K in this dilute fixture.
    The current relative bound allows 3.1e-5 K at 310 K; it includes this
    model error and is not explained solely by solver tolerance. After #265,
    replace it with an energy-balance invariant tested at a substantial solid
    volume fraction; do not retain the dilute fixture as evidence of energy
    conservation. The issue records a 10 vol-% case with about 0.26 K drift.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    mode : str
        Liquid, moments, or FVM population representation.
    """
    pytest.importorskip('assimulo')
    collector = make_collector(data_path, mode)
    original_kwargs = deepcopy(collector.kwargs_cryst)
    temperatures = []  # [K], retain both trajectories for the thermal check
    for run in range(2):
        start = run * DURATION  # [s], absolute start of this call
        time, states = collector.solve_unit(
            runtime=DURATION, verbose=False, sundials_opts=INTEGRATION_OPTIONS)
        assert time[0] == pytest.approx(start, rel=RTOL, abs=0)
        assert time[-1] == pytest.approx(start + DURATION, rel=RTOL, abs=0)
        assert collector.elapsed_time == time[-1]
        temperatures.extend(collector.result.temp)
        if mode == 'liquid':
            seed_mass = MASS_FLOW * 0.1  # [kg], existing 0.1 s feed seed policy
            assert collector.result.mass[0] == pytest.approx(seed_mass, rel=RTOL)
            assert collector.result.mass[-1] == pytest.approx(
                seed_mass + MASS_FLOW * DURATION, rel=SOLVER_RTOL)
            np.testing.assert_allclose(states[0, :len(FRACTIONS)], FRACTIONS,
                                       rtol=RTOL, atol=0)
            assert collector.outputs['mass'] is collector.result.mass
        else:
            seed_volume = np.sqrt(np.finfo(float).eps)  # [m**3], existing seed
            if mode == 'moments':
                np.testing.assert_allclose(collector.result.mu_n[0],
                                           MOMENTS * seed_volume, rtol=RTOL, atol=0)
                np.testing.assert_allclose(
                    collector.Outlet.Solid_1.moments,
                    MOMENTS * (seed_volume + FLOW * DURATION),
                    rtol=SOLVER_RTOL, atol=0)
            else:
                np.testing.assert_allclose(
                    collector.result.distrib[0],
                    collector.Inlet.distrib * seed_volume, rtol=RTOL, atol=0)
                np.testing.assert_allclose(
                    collector.Outlet.Solid_1.distrib,
                    collector.Inlet.distrib * (seed_volume + FLOW * DURATION),
                    rtol=SOLVER_RTOL, atol=0)
            assert collector.result is collector.CrystInst.result
            assert collector.outputs is collector.CrystInst.outputs
        assert collector.kwargs_cryst == original_kwargs
    np.testing.assert_allclose(temperatures, TEMPERATURE,
                               rtol=SOLVER_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('profile', [False, True])
def test_connected_values_override_static_aliases(data_path, profile):
    """Retain connected values when static endpoint fields disagree.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    profile : bool
        Use a three-row upstream profile instead of constant metadata.
    """
    collector = make_collector(data_path, 'moments')
    collector.states_in_dict = {'Inlet': {
        'mass_conc': len(FRACTIONS), 'vol_flow': 1, 'temp': 1, 'mu_n': len(MOMENTS)}}
    upstream_moments = MOMENTS / 2  # [m**n/m**3], upstream differs from endpoint
    upstream_conc = collector.Inlet.Liquid_1.mass_conc[::-1].copy()  # [kg/m**3]
    collector.Inlet.y_inlet['mu_n'] = upstream_moments
    collector.Inlet.y_inlet['mass_conc'] = upstream_conc
    if profile:
        times = np.array([0.0, 1.0, 2.0])  # [s], three identical upstream rows
        collector.Inlet.time_upstream = times
        collector.Inlet.y_inlet = {
            name: np.tile(value, (len(times), 1)) if np.ndim(value) else
            np.full(len(times), value)
            for name, value in collector.Inlet.y_inlet.items()}
        # Reordering fields must preserve both species and population values.
        collector.Inlet.y_inlet = dict(reversed(collector.Inlet.y_inlet.items()))
    inputs = collector.get_inputs_new(0.0)['Inlet']
    np.testing.assert_allclose(inputs['mu_n'], upstream_moments, rtol=RTOL, atol=0)
    np.testing.assert_allclose(inputs['mass_conc'], upstream_conc, rtol=RTOL, atol=0)
    np.testing.assert_allclose(collector.Inlet.moments, MOMENTS, rtol=RTOL, atol=0)


@pytest.mark.assimulo
def test_static_moment_feed_full_solve(data_path):
    """Verify static population accumulation through the real solver.

    Solver tolerances resolve the small seed volume, as in test_crystallizer_moment_inventory.py connected
    collector regression. Frozen kinetics give an exact population integral.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    pytest.importorskip('assimulo')
    collector = make_collector(data_path, 'moments', connected=False)
    collector.solve_unit(runtime=DURATION, verbose=False,
                         sundials_opts=INTEGRATION_OPTIONS)
    seed_volume = np.sqrt(np.finfo(float).eps)  # [m**3], established seed policy
    np.testing.assert_allclose(collector.result.mu_n[0], MOMENTS * seed_volume,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(collector.result.mass_conc[0],
                               collector.Inlet.Liquid_1.mass_conc,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(
        collector.Outlet.Solid_1.moments,
        MOMENTS * (seed_volume + FLOW * DURATION), rtol=SOLVER_RTOL, atol=0)
    assert collector.result.vol[-1] == pytest.approx(
        (seed_volume + FLOW * DURATION) * (1 - KV * NUMBER_DENSITY * SIZE**3),
        rel=SOLVER_RTOL, abs=0)


@pytest.mark.unit
@pytest.mark.parametrize('source', ['dynamic_moments', 'connected_moments',
                                    'connected_concentration'])
def test_partial_input_sources_preserve_alias_fallbacks(data_path, source):
    """Resolve each supplied field before falling back to static aliases.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    source : str
        Dynamic moments or a connected moment/concentration field; the other
        field must come from the real static stream on its original basis.
    """
    collector = make_collector(data_path, 'moments', connected=False)
    collector.states_in_dict = {'Inlet': {
        'mass_conc': len(FRACTIONS), 'vol_flow': 1, 'temp': 1, 'mu_n': len(MOMENTS)}}
    supplied_moments = MOMENTS / 2  # [m**n/m**3], distinguish source from stream
    supplied_conc = collector.Inlet.Liquid_1.mass_conc[::-1].copy()  # [kg/m**3]
    if source == 'dynamic_moments':
        dynamic = DynamicInput()
        dynamic.add_variable('mu_n', lambda time: supplied_moments)
        collector.Inlet.DynamicInlet = dynamic
    else:
        supplied = ({'mu_n': supplied_moments} if source == 'connected_moments'
                    else {'mass_conc': supplied_conc})
        # [m**n/m**3] or [kg/m**3], deliberately incomplete upstream data
        collector.Inlet.y_inlet = supplied
        collector.Inlet.y_upstream = supplied
        collector.Inlet.time_upstream = None
    inputs = collector.get_inputs_new(0.0)['Inlet']
    expected_moments = (MOMENTS if source == 'connected_concentration'
                        else supplied_moments)  # [m**n/m**3]
    expected_conc = (supplied_conc if source == 'connected_concentration'
                     else collector.Inlet.Liquid_1.mass_conc)  # [kg/m**3]
    np.testing.assert_allclose(inputs['mu_n'], expected_moments, rtol=RTOL, atol=0)
    np.testing.assert_allclose(inputs['mass_conc'], expected_conc, rtol=RTOL, atol=0)
    assert inputs['vol_flow'] == pytest.approx(FLOW, rel=RTOL, abs=0)
    assert inputs['temp'] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    assert not hasattr(collector.Inlet, 'mu_n')
    assert not hasattr(collector.Inlet, 'mass_conc')


@pytest.mark.unit
def test_alias_requires_active_moment_input(data_path):
    """Leave generic concentration resolution unchanged without a moment map.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    collector = make_collector(data_path, 'moments', connected=False)
    collector.states_in_dict = {'Inlet': {'mass_conc': len(FRACTIONS)}}
    inputs = collector.get_inputs_new(0.0)['Inlet']
    # No population is requested, so Connections retains its missing-field
    # default [kg/m**3] instead of applying the moment-feed concentration alias.
    np.testing.assert_array_equal(inputs['mass_conc'], np.zeros(len(FRACTIONS)))
    assert not hasattr(collector.Inlet, 'mu_n')
    assert not hasattr(collector.Inlet, 'mass_conc')


@pytest.mark.unit
def test_moment_alias_without_attached_phases():
    """Skip the liquid alias when only stream-owned moments are available.

    Notes
    -----
    Attach the bare stream at the input boundary directly: the public Inlet
    setter requires species from attached phases before it can select a model.
    This probe covers partial stream construction, not a solvable collector.
    """
    collector = DynamicCollector()
    collector._Inlet = SlurryStream(vol_flow=FLOW, moments=MOMENTS.copy())
    collector.states_in_dict = {'Inlet': {
        'mu_n': len(MOMENTS), 'mass_conc': len(FRACTIONS), 'vol_flow': 1}}
    inputs = collector.get_inputs_new(0.0)['Inlet']
    np.testing.assert_allclose(inputs['mu_n'], MOMENTS, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(inputs['mass_conc'], np.zeros(len(FRACTIONS)))
    assert inputs['vol_flow'] == pytest.approx(FLOW, rel=RTOL, abs=0)
    assert not hasattr(collector.Inlet, 'mu_n')
    assert not hasattr(collector.Inlet, 'mass_conc')


@pytest.mark.assimulo
@pytest.mark.parametrize('dynamic_field', ['mu_n', 'mass_conc'])
def test_dynamic_moment_feed_reaches_delegated_solve(data_path, dynamic_field):
    """Integrate partial dynamic slurry inputs through the real collector.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    dynamic_field : str
        Override SI moments [m**n/m**3] or liquid concentration [kg/m**3];
        remaining feed fields retain the static stream values.
    """
    pytest.importorskip('assimulo')
    collector = make_collector(data_path, 'moments', connected=False)
    inlet = collector.Inlet
    original_moments = inlet.moments.copy()  # [m**n/m**3]
    original_concentration = inlet.Liquid_1.mass_conc.copy()  # [kg/m**3]
    expected_moments = (original_moments / 2 if dynamic_field == 'mu_n'
                        else original_moments)  # [m**n/m**3], distinct override
    expected_concentration = (original_concentration[::-1].copy()
                              if dynamic_field == 'mass_conc'
                              else original_concentration)  # [kg/m**3]
    supplied = (expected_moments if dynamic_field == 'mu_n'
                else expected_concentration)  # [m**n/m**3] or [kg/m**3]
    dynamic = DynamicInput()
    dynamic.add_variable(dynamic_field, lambda time: supplied)
    inlet.DynamicInlet = dynamic
    collector.solve_unit(runtime=DURATION, verbose=False,
                         sundials_opts=INTEGRATION_OPTIONS)
    delegated = collector.CrystInst.get_inputs(0.0)
    np.testing.assert_allclose(delegated['Inlet']['mu_n'], expected_moments,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(delegated['Liquid_1']['mass_conc'],
                               expected_concentration, rtol=RTOL, atol=0)
    assert delegated['Inlet']['vol_flow'] == pytest.approx(FLOW, rel=RTOL)
    assert delegated['Inlet']['temp'] == pytest.approx(TEMPERATURE, rel=RTOL)
    seed_volume = np.sqrt(np.finfo(float).eps)  # [m**3], established seed policy
    expected_inventory = expected_moments * (seed_volume + FLOW * DURATION)
    # [m**n], exact constant-feed integral with zero kinetic rates
    np.testing.assert_allclose(collector.Outlet.Solid_1.moments,
                               expected_inventory, rtol=SOLVER_RTOL, atol=0)
    np.testing.assert_allclose(collector.result.mass_conc[0],
                               expected_concentration, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(inlet.moments, original_moments)
    np.testing.assert_array_equal(inlet.Liquid_1.mass_conc, original_concentration)
    assert inlet.DynamicInlet is dynamic
    assert not hasattr(inlet, 'mu_n')
    assert not hasattr(inlet, 'mass_conc')
