"""DynamicInput fields mapped into phase-keyed inlet layouts (#221).

``Connections.get_inputs_new`` assigns each stream-level ``DynamicInput``
field to the inlet group that declares it, keeps static stream or phase
values for uncontrolled fields, and rejects names that several groups
declare. The fixture is the real 1D-FVM MSMPR of
``test_crystallizer_heat_duty``: five liquid species, three unequal size
bins, inactive kinetics, prescribed temperature, and a real slurry feed whose
``LiquidStream``/``SolidStream`` phases match the tank. The crystallizer
declares ``{'Liquid_1': {'mass_conc': 5}, 'Inlet': {'vol_flow': 1, 'temp': 1,
'distrib': 3}}``, so concentration and flow controls land in different groups.

Because feed and tank share one population, the inlet and tank liquid volume
fractions are equal and the MSMPR balances reduce to
``d(distrib)/dt = Q/V (distrib_in - distrib)`` and
``d(mass_conc)/dt = Q/V (mass_conc_in - mass_conc)``; expectations use these
closed forms. The core tests evaluate the RHS directly; the Assimulo test
integrates the same feed with CVode. A second fixture builds a real 1D-FVM
``SemibatchCryst`` through ``initialize_states`` with a seeded feed population
unlike the tank seed; its expectations follow the semibatch balances as
implemented, including the provisional static inlet liquid density described
in that test.

Tests ending in ``_guard`` already pass on the base revision and protect the
single-group ``{'Inlet': ...}`` behaviour; the others fail there with
``KeyError: 'Liquid_1'``.

Related issue: https://github.com/PharmaPy-org/PharmaPy/issues/221
"""

import numpy as np
import pytest

from PharmaPy.Connections import get_inputs_new
from PharmaPy.Crystallizers import MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream, SolidStream
from test_crystallizer_heat_duty import (make_unit as make_heat_unit,
                                         TEMPERATURE, LIQUID_VOLUME)
from test_crystallizer_moment_inventory import SOLVER_RTOL

RTOL = 1e-12  # [-], direct algebra round-off allowance
FEED_FLOW = 1e-4  # [m**3/s], static slurry feed; Q/V is about 0.1 1/s
FLOW_RAMP_TIME = 5.0  # [s], time over which the controlled feed doubles
# [kg/m**3/s], asymmetric per-species slopes of the controlled feed
# concentration; signs and magnitudes differ so species order is visible.
CONC_SLOPE = np.array([1.0, -2.0, 3.0, 0.5, -4.0])
EVALUATION_TIMES = (1.0, 2.5)  # [s], two distinct points on the control laws
DURATION = 5.0  # [s], integrated interval; Q/V-weighted exposure is ~0.8
INTEGRATION_RTOL = 1e-9  # [-], two orders tighter than SOLVER_RTOL
INTEGRATION_ATOL = 1e-10  # [state units], below every concentration/population
# [-], arbitrary factor so controlled values differ from the static stream
CONTROL_SCALE = 3.0

# Semibatch fixture: same compounds, grid, shape factor and inactive kinetics
# as the heat-duty MSMPR fixture, with a feed population unlike the seed.
LIQUID_FRACTIONS = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], liquid mass fractions
CRYSTAL_FRACTIONS = [1, 0, 0, 0, 0]  # [-], pure A crystals
SIZE_GRID = np.array([10.0, 20.0, 40.0])  # [um], unequal FVM bins
SEED_DISTRIB = np.array([1e7, 2e7, 1e7])  # [#/um], total tank seed
FEED_DISTRIB = np.array([3e9, 1e9, 2e9])  # [#/m**3/um], asymmetric feed
SHAPE_FACTOR = 0.5  # [-], non-unit volumetric shape factor
# [kg/m**3], constant polynomial solubility of the heat-duty fixture; no
# nucleation, growth or dissolution mechanism is supplied, so every
# crystallization rate is zero.
INACTIVE_SOLUBILITY = 2000.0


def feed_flow_law(time):
    """Evaluate the controlled slurry feed flow.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    float or numpy.ndarray
        Volumetric slurry flow ``FEED_FLOW (1 + t / FLOW_RAMP_TIME)``
        [m**3/s], with the shape of ``time``.
    """
    return FEED_FLOW * (1 + np.asarray(time) / FLOW_RAMP_TIME)


def make_feed_unit(data_path):
    """Build the real FVM MSMPR with its static slurry feed.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.

    Returns
    -------
    unit : MSMPR
        Prescribed-temperature unit with inactive kinetics.
    states : numpy.ndarray
        Volume-specific CSD [#/m**3/um], shape (3,), followed by liquid
        mass concentrations [kg/m**3], shape (5,).
    """
    return make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FEED_FLOW)


def make_semibatch_unit(data_path):
    """Build a real FVM semibatch crystallizer with a seeded slurry feed.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.

    Returns
    -------
    unit : SemibatchCryst
        Prescribed-temperature unit with inactive kinetics. Its tank liquid
        and the feed liquid share one composition; the feed population
        differs from the tank seed.
    inlet : SlurryStream
        The attached feed, without a ``DynamicInlet``.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    unit = SemibatchCryst(
        'A', method='1D-FVM',
        controls={'temp': lambda time: TEMPERATURE + np.zeros_like(time)})
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=LIQUID_VOLUME,
                         mass_frac=LIQUID_FRACTIONS)
    solid = SolidPhase(path, temp=TEMPERATURE, x_distrib=SIZE_GRID,
                       distrib=SEED_DISTRIB, kv=SHAPE_FACTOR,
                       mass_frac=CRYSTAL_FRACTIONS)
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[INACTIVE_SOLUBILITY])
    inlet = SlurryStream(vol_flow=FEED_FLOW, x_distrib=SIZE_GRID,
                         distrib=FEED_DISTRIB)
    inlet.Phases = (
        LiquidStream(path, temp=TEMPERATURE, mass_frac=LIQUID_FRACTIONS),
        SolidStream(path, temp=TEMPERATURE, mass_frac=CRYSTAL_FRACTIONS,
                    x_distrib=SIZE_GRID, distrib=FEED_DISTRIB, kv=SHAPE_FACTOR))
    unit.Inlet = inlet
    return unit, inlet


@pytest.mark.unit
@pytest.mark.parametrize('time', EVALUATION_TIMES)
def test_phase_keyed_layout_routes_each_dynamic_field(data_path, time):
    """Assign flow to ``Inlet`` and concentration to ``Liquid_1``.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    time : float
        Evaluation time [s].
    """
    unit, _ = make_feed_unit(data_path)
    inlet = unit.Inlet
    static_conc = inlet.Liquid_1.mass_conc.copy()  # [kg/m**3]
    static_distrib = inlet.distrib.copy()  # [#/m**3/um]
    dynamic = DynamicInput()
    dynamic.add_variable('mass_conc', lambda t: static_conc + CONC_SLOPE * t)
    dynamic.add_variable('vol_flow', feed_flow_law)
    inlet.DynamicInlet = dynamic

    inputs = get_inputs_new(time, inlet, unit.states_in_dict)

    assert list(inputs) == ['Liquid_1', 'Inlet']  # declared layout order
    assert list(inputs['Liquid_1']) == ['mass_conc']
    # Dynamic fields first, then static fallbacks in declared order.
    assert list(inputs['Inlet']) == ['vol_flow', 'temp', 'distrib']
    np.testing.assert_allclose(inputs['Liquid_1']['mass_conc'],
                               static_conc + CONC_SLOPE * time,
                               rtol=RTOL, atol=0)
    assert inputs['Inlet']['vol_flow'] == pytest.approx(
        FEED_FLOW * (1 + time / FLOW_RAMP_TIME), rel=RTOL, abs=0)
    assert inputs['Inlet']['temp'] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    np.testing.assert_allclose(inputs['Inlet']['distrib'], static_distrib,
                               rtol=RTOL, atol=0)
    # The stream itself is not modified by the dynamic evaluation.
    assert inlet.vol_flow == pytest.approx(FEED_FLOW, rel=RTOL, abs=0)
    np.testing.assert_array_equal(inlet.Liquid_1.mass_conc, static_conc)


@pytest.mark.unit
def test_field_shared_by_two_groups_is_rejected(data_path):
    """Refuse to guess which declared group a shared field name controls.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_feed_unit(data_path)
    num_species = len(unit.Inlet.Liquid_1.mass_conc)
    # Slurry total and liquid-only mass flows are both called mass_flow.
    layout = {'Inlet': {'mass_flow': 1, 'vol_flow': 1},
              'Liquid_1': {'mass_flow': 1, 'mass_conc': num_species}}
    dynamic = DynamicInput()
    dynamic.add_variable('mass_flow', lambda t: CONTROL_SCALE * unit.Inlet.mass_flow)
    unit.Inlet.DynamicInlet = dynamic
    with pytest.raises(ValueError,
                       match=r"'mass_flow'.*\['Inlet', 'Liquid_1'\]"):
        get_inputs_new(1.0, unit.Inlet, layout)


@pytest.mark.unit
def test_uncontrolled_shared_name_keeps_each_group_source(data_path):
    """Read a shared but uncontrolled name from each group's own object.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_feed_unit(data_path)
    inlet = unit.Inlet
    num_species = len(inlet.Liquid_1.mass_conc)
    layout = {'Inlet': {'mass_flow': 1, 'vol_flow': 1},
              'Liquid_1': {'mass_flow': 1, 'mass_conc': num_species}}
    slurry_mass_flow = inlet.mass_flow  # [kg/s], liquid plus crystals
    liquid_mass_flow = inlet.Liquid_1.mass_flow  # [kg/s], liquid only
    assert slurry_mass_flow > liquid_mass_flow > 0  # distinguishable sources
    dynamic = DynamicInput()
    dynamic.add_variable('vol_flow', feed_flow_law)
    inlet.DynamicInlet = dynamic

    inputs = get_inputs_new(1.0, inlet, layout)

    assert inputs['Inlet']['mass_flow'] == pytest.approx(
        slurry_mass_flow, rel=RTOL, abs=0)
    assert inputs['Liquid_1']['mass_flow'] == pytest.approx(
        liquid_mass_flow, rel=RTOL, abs=0)
    assert inputs['Inlet']['vol_flow'] == pytest.approx(
        feed_flow_law(1.0), rel=RTOL, abs=0)
    assert 'vol_flow' not in inputs['Liquid_1']


@pytest.mark.unit
def test_undeclared_field_joins_inlet_group(data_path):
    """Pass an undeclared control through under ``Inlet`` as before.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_feed_unit(data_path)
    inlet = unit.Inlet
    controlled_mass_flow = CONTROL_SCALE * inlet.mass_flow  # [kg/s], not consumed by MSMPR
    dynamic = DynamicInput()
    dynamic.add_variable('mass_flow', lambda t: controlled_mass_flow)
    inlet.DynamicInlet = dynamic

    inputs = get_inputs_new(1.0, inlet, unit.states_in_dict)

    assert list(inputs['Inlet']) == ['mass_flow', 'vol_flow', 'temp', 'distrib']
    assert inputs['Inlet']['mass_flow'] == controlled_mass_flow
    assert inputs['Inlet']['vol_flow'] == pytest.approx(FEED_FLOW, rel=RTOL, abs=0)
    assert list(inputs['Liquid_1']) == ['mass_conc']
    np.testing.assert_allclose(inputs['Liquid_1']['mass_conc'],
                               inlet.Liquid_1.mass_conc, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_undeclared_field_without_inlet_group_is_rejected(data_path):
    """Reject a control that no declared group, including ``Inlet``, owns.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_feed_unit(data_path)
    layout = {'Liquid_1': {'mass_conc': len(unit.Inlet.Liquid_1.mass_conc)}}
    dynamic = DynamicInput()
    dynamic.add_variable('vol_flow', feed_flow_law)
    unit.Inlet.DynamicInlet = dynamic
    with pytest.raises(ValueError, match=r"'vol_flow' is not declared.*'Inlet'"):
        get_inputs_new(1.0, unit.Inlet, layout)


@pytest.mark.unit
def test_single_inlet_layout_receives_every_dynamic_field_guard(data_path):
    """Keep the established ``{'Inlet': ...}`` contents and field order.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = make_feed_unit(data_path)
    inlet = unit.Inlet
    controlled_mass_flow = CONTROL_SCALE * inlet.mass_flow  # [kg/s], undeclared here
    dynamic = DynamicInput()
    dynamic.add_variable('mass_flow', lambda t: controlled_mass_flow)
    dynamic.add_variable('vol_flow', feed_flow_law)
    inlet.DynamicInlet = dynamic

    inputs = get_inputs_new(2.5, inlet, {'Inlet': {'vol_flow': 1, 'temp': 1}})

    assert list(inputs) == ['Inlet']
    assert list(inputs['Inlet']) == ['mass_flow', 'vol_flow', 'temp']
    assert inputs['Inlet']['mass_flow'] == controlled_mass_flow
    assert inputs['Inlet']['vol_flow'] == pytest.approx(
        feed_flow_law(2.5), rel=RTOL, abs=0)
    assert inputs['Inlet']['temp'] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)


@pytest.mark.unit
@pytest.mark.parametrize('time', EVALUATION_TIMES)
def test_msmpr_rhs_follows_time_varying_slurry_feed(data_path, time):
    """Drive the real MSMPR RHS with controlled flow and liquid composition.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    time : float
        Evaluation time [s].
    """
    unit, states = make_feed_unit(data_path)
    num_bins = len(unit.Inlet.distrib)
    tank_distrib = states[:num_bins]  # [#/m**3/um], equal to the static feed
    tank_conc = states[num_bins:]  # [kg/m**3]
    dynamic = DynamicInput()
    dynamic.add_variable('vol_flow', feed_flow_law)
    dynamic.add_variable('mass_conc', lambda t: tank_conc + CONC_SLOPE * t)
    unit.Inlet.DynamicInlet = dynamic
    volume = unit.Slurry.vol  # [m**3], MSMPR slurry holdup
    dilution_rate = feed_flow_law(time) / volume  # [1/s]

    derivatives = unit.unit_model(time, states)

    # Uncontrolled feed population keeps its static value, equal to the tank
    # population; a zero-filled inlet would give -Q/V*distrib instead.
    np.testing.assert_allclose(
        derivatives[:num_bins], 0.0, rtol=0,
        atol=RTOL * dilution_rate * tank_distrib.max())
    np.testing.assert_allclose(derivatives[num_bins:],
                               dilution_rate * CONC_SLOPE * time,
                               rtol=RTOL, atol=0)
    feed = unit.get_inputs(time)
    assert feed['Inlet']['temp'] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    np.testing.assert_allclose(feed['Inlet']['distrib'], tank_distrib,
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('time', EVALUATION_TIMES)
def test_semibatch_rhs_follows_time_varying_slurry_feed(data_path, time):
    """Drive the real FVM semibatch RHS with controlled flow and composition.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    time : float
        Evaluation time [s].

    Notes
    -----
    As implemented, the semibatch balances on a total basis are
    ``d(distrib)/dt = Q n_in``, ``dV/dt = phi_in Q rho_in / rho``, and
    ``d(mass_conc)/dt = phi_in Q / V (c_in - c rho_in / rho)``, with the
    inlet liquid volume fraction ``phi_in = 1 - kv mu_3,in`` [-]. The inlet
    liquid density ``rho_in`` is evaluated from the feed stream's *static*
    ``Liquid_1`` composition, not from the controlled ``mass_conc``; this
    limitation is documented in ``advanced_usage.rst`` and awaits a
    follow-up. Tank and static feed liquids share one composition here, so
    ``rho_in / rho = 1`` exactly under that provisional model. If inlet
    density is later made to follow the controlled composition, the
    expected ``dV/dt`` and ``d(mass_conc)/dt`` must include
    ``rho(c_in(t)) / rho`` and this test must be updated.
    """
    unit, inlet = make_semibatch_unit(data_path)
    static_conc = inlet.Liquid_1.mass_conc.copy()  # [kg/m**3]
    dynamic = DynamicInput()
    dynamic.add_variable('vol_flow', feed_flow_law)
    dynamic.add_variable(
        'mass_conc',
        lambda t: static_conc + np.multiply.outer(np.asarray(t), CONC_SLOPE))
    # [kg/m**3], shape (5,) or (num_times, 5) for array-time initialization
    inlet.DynamicInlet = dynamic
    states, _ = unit.initialize_states(runtime=DURATION)
    num_bins = len(SIZE_GRID)
    num_species = len(static_conc)
    np.testing.assert_allclose(states[num_bins:num_bins + num_species],
                               static_conc, rtol=RTOL, atol=0)
    liquid_volume = states[-1]  # [m**3]
    flow = feed_flow_law(time)  # [m**3/s]
    third_moment_integrand = FEED_DISTRIB * SIZE_GRID**3  # [#/m**3*um**2]
    feed_third_moment = 1e-18 * np.sum(
        0.5 * (third_moment_integrand[1:] + third_moment_integrand[:-1])
        * np.diff(SIZE_GRID))
    # [m**3/m**3], trapezoidal mu_3; 1e-18 is the exact um**3-to-m**3 factor
    feed_liquid_fraction = 1 - SHAPE_FACTOR * feed_third_moment  # [-]

    feed = unit.get_inputs(time)
    derivatives = unit.unit_model(time, states)

    assert list(feed['Liquid_1']) == ['mass_conc']
    assert list(feed['Inlet']) == ['vol_flow', 'temp', 'distrib']
    np.testing.assert_allclose(feed['Liquid_1']['mass_conc'],
                               static_conc + CONC_SLOPE * time,
                               rtol=RTOL, atol=0)
    assert feed['Inlet']['temp'] == pytest.approx(TEMPERATURE, rel=RTOL, abs=0)
    np.testing.assert_allclose(derivatives[:num_bins], flow * FEED_DISTRIB,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(
        derivatives[num_bins:num_bins + num_species],
        feed_liquid_fraction * flow / liquid_volume * CONC_SLOPE * time,
        rtol=RTOL, atol=0)
    assert derivatives[-1] == pytest.approx(feed_liquid_fraction * flow,
                                            rel=RTOL, abs=0)


@pytest.mark.assimulo
def test_msmpr_solve_integrates_time_varying_slurry_feed(data_path):
    """Integrate a ramped feed flow with a controlled liquid composition.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.

    Notes
    -----
    With constant feed concentration ``c_f`` and ``Q(t)`` from
    :func:`feed_flow_law`, ``dc/dt = Q(t)/V (c_f - c)`` gives
    ``c(t) = c_f + (c_0 - c_f) exp(-FEED_FLOW/V (t + t**2/(2 FLOW_RAMP_TIME)))``.
    """
    pytest.importorskip('assimulo')
    unit, states = make_feed_unit(data_path)
    num_bins = len(unit.Inlet.distrib)
    initial_distrib = states[:num_bins].copy()  # [#/m**3/um]
    initial_conc = states[num_bins:].copy()  # [kg/m**3]
    feed_conc = initial_conc + CONC_SLOPE * DURATION  # [kg/m**3], held constant
    dynamic = DynamicInput()
    dynamic.add_variable('vol_flow', feed_flow_law)
    dynamic.add_variable('mass_conc', lambda t: feed_conc)
    unit.Inlet.DynamicInlet = dynamic
    volume = unit.Slurry.vol  # [m**3], MSMPR slurry holdup
    time_grid = np.linspace(0.0, DURATION, 6)  # [s], five reporting intervals

    time, _ = unit.solve_unit(
        time_grid=time_grid, verbose=False,
        sundials_opts={'rtol': INTEGRATION_RTOL, 'atol': INTEGRATION_ATOL})

    time = np.asarray(time)  # [s]
    exposure = FEED_FLOW / volume * (time + time**2 / (2 * FLOW_RAMP_TIME))  # [-]
    expected_conc = (feed_conc + (initial_conc - feed_conc)
                     * np.exp(-exposure)[:, None])  # [kg/m**3]
    np.testing.assert_allclose(unit.result.mass_conc, expected_conc,
                               rtol=SOLVER_RTOL, atol=0)
    np.testing.assert_allclose(unit.result.vol_flow, feed_flow_law(time),
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(unit.result.distrib,
                               np.tile(initial_distrib, (len(time), 1)),
                               rtol=SOLVER_RTOL, atol=0)
    assert unit.Inlet.DynamicInlet is dynamic
    assert unit.Inlet.vol_flow == pytest.approx(FEED_FLOW, rel=RTOL, abs=0)
