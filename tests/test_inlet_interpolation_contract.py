"""Inlet interpolation contract for connections and stream callers (#218).

Covers Newton divided differences with integer samples and the shared inlet
evaluation used by ``Connections.interpolate_inputs``,
``LiquidStream.InterpolateInputs``, and ``SlurryStream.InterpolateInputs``.
The profile fixture uses non-uniform upstream times, more query times than
upstream samples, and three asymmetric states built from exact linear and
quadratic laws. Not-a-knot cubic splines and three-node Newton polynomials
reproduce such laws exactly, so expectations come from the closed forms. The
liquid stream uses the real single-species nitrogen database; no solver runs.
A second fixture puts a negligible terminal flow next to unit-scale samples,
so any polynomial evaluation at the final node loses it to round-off.

The deprecated module-level ``Streams.Interpolation`` and
``MixedPhases.Interpolation`` aliases must warn and use the corrected window.

Most tests are regressions that fail on the base revision. Tests whose names
end in ``_guard`` already pass there and protect established behaviour:
the vector post-horizon hold and scalar closed-form matches of
``interpolate_inputs``, float-sequence Newton samples, and the forwarding
of ``num_interpolation_points``.

Related issue: https://github.com/PharmaPy-org/PharmaPy/issues/218
"""

from pathlib import Path

import copy

import numpy as np
import pytest

from PharmaPy import MixedPhases, Streams
from PharmaPy.Connections import interpolate_inputs
from PharmaPy.Interpolation import NewtonInterpolation, local_newton_interpolation
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Streams import LiquidStream

# [-], float64 round-off allowance for spline/Newton reproduction of exact
# low-order polynomials with values up to ~3e2.
RTOL = 1e-10

UPSTREAM_TIME = np.array([0., 1., 3., 4., 6.])  # [s], non-uniform support
FINAL_TIME = UPSTREAM_TIME[-1]  # [s]
# [s], unsorted: post-horizon first and last, one pre-support time, the final
# node, interior times, and 5.5 s, whose nearest node is the final one; seven
# queries against five upstream samples.
QUERY_TIME = np.array([7.5, -0.5, 2.5, 6.0, 5.5, 0.5, 9.0])

NITROGEN_PATH = str(Path(__file__).resolve().parents[1]
                    / 'data/evaporator/props_nitrogen.json')
NITROGEN_TEMP = 78.0  # [K], liquid nitrogen fixture used by holdup tests
STREAM_FLOW = 1.0  # [kg/s], nonzero to avoid the zero-amount warning


def upstream_law(time):
    """Evaluate the synthetic upstream states in closed form.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    numpy.ndarray
        States of shape (3,) or (num_times, 3), ordered as mass flow
        ``1 + 0.5 t`` [kg/s], mass fraction ``0.8 - 0.1 t`` [-], and
        temperature ``300 + t**2`` [K].
    """
    time = np.asarray(time, dtype=float)  # [s]
    return np.stack((1 + 0.5 * time, 0.8 - 0.1 * time, 300 + time ** 2),
                    axis=-1)


def held_law(time):
    """Evaluate the expected inlet states with the terminal hold applied.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    numpy.ndarray
        ``upstream_law`` at ``min(time, FINAL_TIME)``; same shape and state
        order as :func:`upstream_law`.
    """
    return upstream_law(np.minimum(time, FINAL_TIME))


UPSTREAM_STATES = upstream_law(UPSTREAM_TIME)  # [kg/s, -, K], shape (5, 3)
TERMINAL_STATES = np.array([4., 0.2, 336.])  # [kg/s, -, K], law at t = 6 s


def make_liquid_stream(**kwargs):
    """Build a real liquid nitrogen stream without upstream data.

    Parameters
    ----------
    **kwargs : dict
        Extra ``LiquidStream`` keyword arguments.

    Returns
    -------
    PharmaPy.Streams.LiquidStream
        Stream with mass flow ``STREAM_FLOW`` [kg/s].
    """
    return LiquidStream(NITROGEN_PATH, mass_frac=[1.], temp=NITROGEN_TEMP,
                        mass_flow=STREAM_FLOW, **kwargs)


STREAM_FACTORIES = {'liquid': make_liquid_stream, 'slurry': SlurryStream}

CANCELLING_TIME = np.array([0., 1., 2.])  # [s]
# [kg/s, K]: mass flow drops to a negligible 1e-20 kg/s at the final sample,
# below the round-off of a polynomial through the unit-scale neighbours;
# temperature is an asymmetric second field.
CANCELLING_STATES = np.array([[1., 300.], [1., 310.], [1e-20, 305.]])
CANCELLING_TERMINAL = np.array([1e-20, 305.])  # [kg/s, K], final sample


def stream_with_profile(kind, time_upstream, y_inlet, **stream_kwargs):
    """Attach an array upstream profile to a real stream.

    Parameters
    ----------
    kind : {'liquid', 'slurry'}
        Stream class to build.
    time_upstream : numpy.ndarray
        Upstream times [s], shape (num_upstream_times,).
    y_inlet : numpy.ndarray
        Upstream values with time on the first axis.
    **stream_kwargs : dict
        Extra stream constructor keyword arguments.

    Returns
    -------
    LiquidStream or SlurryStream
        Stream whose ``InterpolateInputs`` evaluates the given profile.
    """
    stream = STREAM_FACTORIES[kind](**stream_kwargs)
    stream.time_upstream = time_upstream
    stream.y_inlet = y_inlet
    return stream


@pytest.mark.unit
def test_newton_interpolation_keeps_fractional_value_for_integer_samples():
    nodes = np.array([0, 2])  # [s], integer storage
    samples = np.array([[0, 3], [1, 0]])  # [kg/s], integer storage, two fields
    interp = NewtonInterpolation(x_data=nodes, y_data=samples[:, 0])
    assert interp.evalPolynomial(1.0) == pytest.approx(0.5, rel=RTOL)
    vector = NewtonInterpolation(x_data=nodes, y_data=samples)
    np.testing.assert_allclose(vector.evalPolynomial(1.0), [0.5, 1.5], rtol=RTOL)
    np.testing.assert_array_equal(samples, [[0, 3], [1, 0]])
    assert samples.dtype.kind == 'i'


@pytest.mark.unit
def test_local_newton_interpolation_keeps_fractional_value_for_integer_samples():
    time = np.array([0, 2, 4, 6])  # [s], integer storage
    values = np.array([0, 1, 4, 9])  # [kg/s], integer samples of (t/2)**2
    # The query lies between nodes 1 and 2; the three-node window [0, 2, 4]
    # reproduces the quadratic exactly: (3/2)**2 = 2.25.
    assert local_newton_interpolation(3.0, time, values) == pytest.approx(2.25, rel=RTOL)


# Samples [kg/s] at nodes 0 and 2 s; the midpoint 1 s is their mean.
SEQUENCE_SAMPLES = [
    pytest.param([0, 1], 0.5, id='int-list'),
    pytest.param([[0, 3], [1, 0]], [0.5, 1.5], id='nested-int-list'),
    pytest.param((0., 1.), 0.5, id='float-tuple_guard'),
    pytest.param(((0., 3.), (1., 0.)), [0.5, 1.5], id='nested-float-tuple_guard'),
]


@pytest.mark.unit
@pytest.mark.parametrize('samples, expected', SEQUENCE_SAMPLES)
def test_newton_interpolation_accepts_sequence_samples(samples, expected):
    nodes = np.array([0., 2.])  # [s]
    snapshot = copy.deepcopy(samples)
    interp = NewtonInterpolation(x_data=nodes, y_data=samples)
    np.testing.assert_allclose(interp.evalPolynomial(1.0), expected, rtol=RTOL)
    assert samples == snapshot


# Upstream samples [kg/s] at 0, 1, 2 s of t**2 (and 2 - t for two fields).
SEQUENCE_PROFILES = [
    pytest.param([0., 1., 4.], [2.25, 4.], id='float-list'),
    pytest.param([0, 1, 4], [2.25, 4.], id='int-list'),
    pytest.param([[0., 2.], [1., 1.], [4., 0.]], [[2.25, 0.5], [4., 0.]],
                 id='nested-list'),
]


@pytest.mark.unit
@pytest.mark.parametrize('profile, expected', SEQUENCE_PROFILES)
def test_interpolate_inputs_accepts_sequence_profile(profile, expected):
    time = np.array([0., 1., 2.])  # [s]
    queries = np.array([1.5, 3.0])  # [s], interior then held after support
    for query, target in zip(queries, expected):
        np.testing.assert_allclose(interpolate_inputs(query, time, profile), target,
                                   rtol=RTOL)
    np.testing.assert_allclose(interpolate_inputs(queries[::-1], time, profile),
                               expected[::-1], rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('num_points', [1, 3])
@pytest.mark.parametrize('query', [2.0, 3.0])  # [s], at and after final time
def test_interpolate_inputs_scalar_query_returns_actual_terminal_sample(num_points, query):
    result = interpolate_inputs(query, CANCELLING_TIME, CANCELLING_STATES,
                                num_points=num_points)
    np.testing.assert_allclose(result, CANCELLING_TERMINAL, rtol=RTOL)
    assert not np.shares_memory(result, CANCELLING_STATES)
    flow = interpolate_inputs(query, CANCELLING_TIME, CANCELLING_STATES[:, 0],
                              num_points=num_points)  # [kg/s]
    assert np.ndim(flow) == 0
    assert flow == pytest.approx(CANCELLING_TERMINAL[0], rel=RTOL)


@pytest.mark.unit
def test_interpolate_inputs_vector_query_returns_actual_terminal_sample():
    queries = np.array([3.0, 2.0, 0.0])  # [s], after, at, and before the end
    expected = np.vstack((CANCELLING_TERMINAL, CANCELLING_TERMINAL,
                          CANCELLING_STATES[0]))  # [kg/s, K]
    np.testing.assert_allclose(
        interpolate_inputs(queries, CANCELLING_TIME, CANCELLING_STATES),
        expected, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind, stream_kwargs', [
    ('liquid', {}), ('liquid', {'num_interpolation_points': 1}), ('slurry', {})])
@pytest.mark.parametrize('query', [2.0, 3.0])  # [s], at and after final time
def test_stream_query_returns_actual_terminal_sample(kind, stream_kwargs, query):
    stream = stream_with_profile(kind, CANCELLING_TIME, CANCELLING_STATES,
                                 **stream_kwargs)
    np.testing.assert_allclose(stream.InterpolateInputs(query), CANCELLING_TERMINAL,
                               rtol=RTOL)
    np.testing.assert_allclose(stream.InterpolateInputs(np.array([query])),
                               [CANCELLING_TERMINAL], rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('num_states', [1, 3])
def test_interpolate_inputs_holds_terminal_sample_in_query_order(num_states):
    values = UPSTREAM_STATES[:, :num_states]  # [kg/s, -, K]
    if num_states == 1:
        values = values[:, 0]  # [kg/s], scalar-field layout (num_upstream_times,)
    expected = held_law(QUERY_TIME)[:, :num_states]  # [kg/s, -, K]
    if num_states == 1:
        expected = expected[:, 0]  # [kg/s]
    result = interpolate_inputs(QUERY_TIME, UPSTREAM_TIME, values)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=RTOL)


@pytest.mark.unit
def test_interpolate_inputs_all_post_horizon_query_holds_terminal_sample_guard():
    times = np.array([9., 7.5])  # [s], descending and entirely after support
    result = interpolate_inputs(times, UPSTREAM_TIME, UPSTREAM_STATES)
    np.testing.assert_allclose(result, np.tile(TERMINAL_STATES, (2, 1)), rtol=RTOL)


@pytest.mark.unit
def test_interpolate_inputs_scalar_queries_match_held_law_guard():
    # Vector rows match the same law in the query-order test, so scalar and
    # vector evaluations agree without comparing one against the other.
    for time in QUERY_TIME:
        scalar = interpolate_inputs(time, UPSTREAM_TIME, UPSTREAM_STATES)
        assert scalar.shape == (3,)
        np.testing.assert_allclose(scalar, held_law(time), rtol=RTOL)


@pytest.mark.unit
def test_interpolate_inputs_integer_profile_keeps_fractional_values():
    time = np.array([0., 2., 4.])  # [s]
    values = np.array([[0, 3], [1, 1], [2, -1]])  # [kg/s], integer t/2 and 3 - t
    expected = {1.0: [0.5, 2.], 3.0: [1.5, 0.], 5.0: [2., -1.]}  # held after 4 s
    for query, target in expected.items():
        np.testing.assert_allclose(interpolate_inputs(query, time, values), target,
                                   rtol=RTOL)
    queries = np.array(list(expected))  # [s]
    np.testing.assert_allclose(interpolate_inputs(queries, time, values),
                               list(expected.values()), rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['liquid', 'slurry'])
def test_stream_vector_interpolation_holds_terminal_sample_in_query_order(kind):
    stream = stream_with_profile(kind, UPSTREAM_TIME, UPSTREAM_STATES)
    result = stream.InterpolateInputs(QUERY_TIME)
    assert result.shape == (len(QUERY_TIME), 3)
    np.testing.assert_allclose(result, held_law(QUERY_TIME), rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['liquid', 'slurry'])
def test_stream_all_post_horizon_query_holds_terminal_sample(kind):
    stream = stream_with_profile(kind, UPSTREAM_TIME, UPSTREAM_STATES)
    post_horizon = np.array([9., 7.5])  # [s], descending and entirely after support
    np.testing.assert_allclose(stream.InterpolateInputs(post_horizon),
                               np.tile(TERMINAL_STATES, (2, 1)), rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['liquid', 'slurry'])
def test_stream_scalar_interpolation_matches_held_law(kind):
    # Matching the law used for vector rows makes scalar and vector agree,
    # including at the final node and in the last upstream interval.
    stream = stream_with_profile(kind, UPSTREAM_TIME, UPSTREAM_STATES)
    for time in QUERY_TIME:
        scalar = stream.InterpolateInputs(time)
        assert scalar.shape == (3,)
        np.testing.assert_allclose(scalar, held_law(time), rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['liquid', 'slurry'])
def test_stream_scalar_field_profile_keeps_one_dimensional_shape(kind):
    stream = stream_with_profile(kind, UPSTREAM_TIME, UPSTREAM_STATES[:, 2])
    result = stream.InterpolateInputs(QUERY_TIME)
    assert result.shape == (len(QUERY_TIME),)
    np.testing.assert_allclose(result, held_law(QUERY_TIME)[:, 2], rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['liquid', 'slurry'])
def test_stream_integer_profile_keeps_fractional_scalar_value(kind):
    time = np.array([0., 2., 4.])  # [s]
    values = np.array([[0, 3], [1, 1], [2, -1]])  # [kg/s], integer t/2 and 3 - t
    stream = stream_with_profile(kind, time, values)
    np.testing.assert_allclose(stream.InterpolateInputs(1.0), [0.5, 2.], rtol=RTOL)


@pytest.mark.unit
def test_liquid_stream_forwards_interpolation_node_count_guard():
    stream = make_liquid_stream(num_interpolation_points=2)
    stream.time_upstream = UPSTREAM_TIME
    stream.y_inlet = UPSTREAM_STATES
    query = 2.5  # [s], nearest node 3 s gives the two-node window [1, 3] s
    # Linear states are exact; temperature is the secant through
    # 301 K at 1 s and 309 K at 3 s: 301 + 4 * 1.5 = 307 K.
    expected = np.array([2.25, 0.55, 307.])  # [kg/s, -, K]
    np.testing.assert_allclose(stream.InterpolateInputs(query), expected, rtol=RTOL)


LATE_QUERY = 5.5  # [s], between the last two nodes and nearest the final one


@pytest.mark.unit
@pytest.mark.parametrize('module, kwargs', [
    (Streams, {'num_points': 3}), (Streams, {}), (MixedPhases, {})])
def test_deprecated_interpolation_alias_uses_full_final_window(module, kwargs):
    """Warn and evaluate the deprecated aliases with a three-node window.

    Parameters
    ----------
    module : module
        ``PharmaPy.Streams`` or ``PharmaPy.MixedPhases``.
    kwargs : dict
        Optional ``num_points`` of the Streams alias (default three).

    Notes
    -----
    A three-node polynomial reproduces the linear and quadratic states of
    ``upstream_law`` exactly; the window used in v1.0.0 near the end of the
    profile had fewer nodes and missed the quadratic temperature.
    """
    with pytest.warns(DeprecationWarning,
                      match=rf'{module.__name__}\.Interpolation is deprecated'):
        value = module.Interpolation(UPSTREAM_TIME, UPSTREAM_STATES, LATE_QUERY,
                                     **kwargs)
    np.testing.assert_allclose(value, upstream_law(LATE_QUERY), rtol=RTOL, atol=0)
