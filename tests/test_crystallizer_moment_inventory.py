"""moment inventory, continuation, and collector handoff regressions.

Synthetic frozen populations separate inventory accounting from kinetics.
Collector and continuation tests use real CVode in the Assimulo lane.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/224
"""

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Connections import Connection
from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream
from test_crystallizer_parameter_evaluations import make_unit, RTOL, TEMPERATURE

DURATION = 0.01  # [s], short constant-feed accounting interval
FLOW = 0.02  # [m**3/s], positive flow avoids the deferred zero-flow stream defect
SOLVER_RTOL = 1e-7  # [-], allowance for default CVode integration error


def inventory_unit(data_path, unit_type=BatchCryst, gridless=False, num_mom=4):
    """Build a frozen seeded vessel and a matching constant slurry inlet.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type, optional
        Batch, Semibatch, or MSMPR crystallizer class.
    gridless : bool, optional
        Omit the seed grid and distribution when True.
    num_mom : int, optional
        Number of seed moments for gridless fixtures; four by default.

    Returns
    -------
    unit : crystallizer
        Two cubic metres of liquid, non-unit kv, and unequal solid volume.
    states : ndarray
        Raw moments [um**n] or [um**n/m**3], concentrations [kg/m**3],
        and liquid volume [m**3] for Batch/Semibatch.
    """
    batch, _ = make_unit(data_path, 'moments')
    liquid, solid = batch.Liquid_1, batch.Solid_1
    if gridless:
        solid = SolidPhase(solid.path_data, temp=TEMPERATURE, kv=solid.kv,
                           moments=solid.getMoments(mom_num=range(num_mom)),
                           mass_frac=solid.mass_frac)
    unit = unit_type('A', method='moments', controls={
        'temp': lambda time: TEMPERATURE + np.zeros_like(time)})
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[2000.0])
    # [kg/m**3], synthetic solubility above feed concentration; all rates zero
    unit.Kinetics.target_idx = unit.target_ind
    unit.num_species = len(liquid.mass_conc)
    unit.x_grid = solid.x_distrib  # [um] or None
    unit.diam_tank = 1.0  # [m], synthetic cylindrical vessel for heat retrieval
    unit.area_base = np.pi / 4  # [m**2], circular base of the unit-diameter tank
    population = solid.moments.copy()  # [m**n], total seed
    if unit_type is not BatchCryst:
        inlet = SlurryStream(vol_flow=FLOW, moments=unit.Slurry.moments.copy())
        inlet.Phases = (
            LiquidStream(liquid.path_data, temp=TEMPERATURE,
                         mass_frac=liquid.mass_frac, vol_flow=FLOW),
            SolidStream(solid.path_data, temp=TEMPERATURE,
                        mass_frac=solid.mass_frac, kv=solid.kv))
        # Same SI mu_n metadata supplied by a connected upstream MSMPR.
        inlet.y_inlet = dict(mu_n=inlet.moments, mass_conc=liquid.mass_conc,
                             temp=TEMPERATURE, vol_flow=FLOW)
        inlet.y_upstream = inlet.y_inlet
        inlet.time_upstream = None
        unit.Inlet = inlet
    if unit_type is MSMPR:
        population /= unit.Slurry.vol  # [m**n/m**3], constant-volume state basis
    parts = [population * 1e6**np.arange(len(population)), liquid.mass_conc]
    # [um**n] or [um**n/m**3], exact micrometre conversion
    if unit_type is not MSMPR:
        parts.append([liquid.vol])  # [m**3]
    return unit, np.concatenate(parts)


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [BatchCryst, SemibatchCryst])
@pytest.mark.parametrize('gridless', [False, True])
def test_final_moment_inventory(data_path, unit_type, gridless):
    """Reconcile a changed population shape and final solid inventory.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Batch or Semibatch crystallizer.
    gridless : bool
        Whether the initial solid lacks a size grid [um].
    """
    unit, initial = inventory_unit(data_path, unit_type, gridless)
    # Final population: nine hundred million crystals of length 30 um.
    expected = np.array([9e8, 2.7e4, 0.81, 2.43e-5])  # [m**n], n=0..3
    states = np.tile(initial, (2, 1))  # [raw state units], initial/final rows
    states[-1, :4] = expected * 1e6**np.arange(4)  # [um**n], exact conversion
    unit.retrieve_results(np.array([0.0, DURATION]), states)
    expected_volume = unit.Solid_1.kv * expected[3]  # [m**3]
    assert unit.Solid_1.vol == pytest.approx(expected_volume, rel=RTOL, abs=0)
    assert unit.Solid_1.mass == pytest.approx(
        expected_volume * unit.Solid_1.getDensity(), rel=RTOL, abs=0)
    np.testing.assert_allclose(unit.Solid_1.moments, expected, rtol=RTOL, atol=0)
    np.testing.assert_allclose(unit.Outlet.moments,
                               expected / (initial[-1] + expected_volume),
                               rtol=RTOL, atol=0)


@pytest.mark.unit
def test_semibatch_moment_feed_rhs(data_path):
    """Integrate the declared SI inlet moments into total moment rates.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, states = inventory_unit(data_path, SemibatchCryst, gridless=True)
    expected = np.array([4.5e8, 1.05e10, 2.85e11, 8.85e12])
    # [um**n], independent trapezoids of the seed fixture
    expected *= FLOW / unit.Slurry.vol  # [um**n/s], inlet particle flow
    rhs = unit.unit_model(0.0, states)  # [state unit/s], full public input handoff
    np.testing.assert_allclose(rhs[:4], expected, rtol=RTOL, atol=0)
    assert rhs[-1] == pytest.approx(
        FLOW * (1 - unit.Solid_1.kv * unit.Inlet.moments[3]), rel=RTOL, abs=0)


@pytest.mark.unit
def test_msmpr_outlet_retains_shape_factor(data_path):
    """Preserve the phase-owned shape factor in outlet solid flow.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, states = inventory_unit(data_path, MSMPR, gridless=True)
    unit.retrieve_results(np.array([0.0, DURATION]), np.tile(states, (2, 1)))
    assert unit.Outlet.Solid_1.kv == unit.Solid_1.kv
    assert unit.Outlet.Solid_1.vol_flow == pytest.approx(
        FLOW * unit.Solid_1.kv * unit.result.mu_n[-1, 3], rel=RTOL, abs=0)


@pytest.mark.assimulo
def test_two_run_moments_remain_si(data_path):
    """Keep both stored runs and the concatenated result on the SI basis.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    pytest.importorskip('assimulo')
    unit, _ = inventory_unit(data_path, BatchCryst, gridless=True)
    slurry = Slurry(vol=unit.Slurry.vol, moments=unit.Slurry.moments.copy())
    slurry.Phases = unit.Phases
    unit.Phases = slurry  # Public attachment of a moment-owned slurry without dx.
    expected = np.array([4.5e8, 1.05e4, 0.285, 8.85e-6])  # [m**n]
    unit.solve_unit(time_grid=[0.0, DURATION], verbose=False)
    unit.solve_unit(time_grid=[DURATION, 2 * DURATION], verbose=False)
    for profile in unit.profiles_runs:
        np.testing.assert_allclose(profile['mu_n'], np.tile(expected, (2, 1)),
                                   rtol=SOLVER_RTOL, atol=0)
    np.testing.assert_allclose(unit.result.mu_n,
                               np.tile(expected, (len(unit.result.time), 1)),
                               rtol=SOLVER_RTOL, atol=0)


@pytest.mark.assimulo
@pytest.mark.parametrize('washout', [False, True])
def test_msmpr_moment_output_collects_without_grid(data_path, washout):
    """Collect constant or changing MSMPR moments through a real Connection.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    washout : bool
        Feed zero inlet moments to produce a changing upstream profile.
    """
    pytest.importorskip('assimulo')
    unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
    if washout:
        unit.Inlet.y_inlet['mu_n'] = np.zeros(4)  # [m**n/m**3], solid-free feed
    integration_rtol = 1e-9  # [-], two orders tighter than the inventory assertion
    integration_atol = 1e-10  # [state units], below the smallest initial volume
    options = {'rtol': integration_rtol, 'atol': integration_atol}
    unit.solve_unit(time_grid=[0.0, DURATION], verbose=False, sundials_opts=options)
    collector = DynamicCollector()
    Connection(unit, collector).transfer_data()
    collector.solve_unit(time_grid=[0.0, DURATION], verbose=False, sundials_opts=options)
    np.testing.assert_allclose(collector.CrystInst.sundials_opt['rtol'],
                               integration_rtol, rtol=RTOL, atol=0)
    np.testing.assert_allclose(collector.CrystInst.sundials_opt['atol'],
                               integration_atol, rtol=RTOL, atol=0)
    assert collector.CrystInst.method == 'moments'
    assert collector.Outlet.distrib is None
    assert collector.Outlet.Solid_1.kv == unit.Solid_1.kv
    seed_volume = np.sqrt(np.finfo(float).eps)  # [m**3], established collector seed
    collected_volume = seed_volume + FLOW * DURATION  # [m**3]
    inlet_initial = unit.result.mu_n[0]  # [m**n/m**3], first upstream time
    inlet_mean = unit.result.mu_n.mean(axis=0)  # [m**n/m**3], linear inlet profile
    expected = inlet_initial * seed_volume + inlet_mean * FLOW * DURATION
    # [m**n], exact integral of the two-point interpolated feed plus initial seed
    np.testing.assert_allclose(collector.result.mu_n[0], inlet_initial * seed_volume,
                               rtol=RTOL, atol=0)
    assert collector.result.vol[0] == pytest.approx(
        seed_volume * (1 - unit.Solid_1.kv * inlet_initial[3]), rel=RTOL, abs=0)
    np.testing.assert_allclose(collector.Outlet.Solid_1.moments, expected,
                               rtol=SOLVER_RTOL, atol=0)
    solid_volume = unit.Solid_1.kv * expected[3]  # [m**3]
    assert collector.Outlet.Solid_1.vol == pytest.approx(solid_volume, rel=SOLVER_RTOL, abs=0)
    assert collector.Outlet.Liquid_1.vol == pytest.approx(
        collected_volume - solid_volume, rel=SOLVER_RTOL, abs=0)


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [SemibatchCryst, MSMPR])
@pytest.mark.parametrize('quantity', ['moments', 'liquid_fraction'])
def test_static_moment_feed_material_balance(data_path, unit_type, quantity):
    """Carry a static slurry's crystals and liquid fraction into the RHS.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Semibatch or continuous crystallizer.
    quantity : str
        Verify the particle feed or the liquid fraction used by the balance.
    """
    unit, states = inventory_unit(data_path, unit_type, gridless=True)
    unit.Inlet.y_upstream = None
    states[:4] = 0  # [um**n] or [um**n/m**3], initially crystal-free tank
    # Independent integrals of the fixture's size distribution, on the feed basis.
    feed_volume = 2.0 + unit.Solid_1.kv * 8.85e-6  # [m**3], original seed slurry
    feed_moments = np.array([4.5e8, 1.05e10, 2.85e11, 8.85e12]) / feed_volume
    # [um**n/m**3], per slurry volume
    solid_fraction = unit.Solid_1.kv * 8.85e-6 / feed_volume  # [-]
    rhs = unit.unit_model(0.0, states)  # [state unit/s], real material_balances
    if quantity == 'moments':
        total_feed = rhs[:4]  # [um**n/s], Semibatch total state
        if unit_type is MSMPR:
            total_feed = rhs[:4] * unit.Slurry.vol  # [um**n/s], intensive to total
        np.testing.assert_allclose(total_feed, FLOW * feed_moments, rtol=RTOL, atol=0)
    elif unit_type is SemibatchCryst:
        assert rhs[-1] == pytest.approx(FLOW * (1 - solid_fraction), rel=RTOL, abs=0)
    else:
        # At zero tank crystals and equal feed/tank liquid composition, the
        # MSMPR liquid feed deficit is Q/V*c*(phi_in-1).
        expected = -FLOW / unit.Slurry.vol * states[4:] * solid_fraction  # [kg/m**3/s]
        cancellation_rtol = 1e-9  # [-], subtracting liquid fractions loses ~6 digits
        np.testing.assert_allclose(rhs[4:], expected, rtol=cancellation_rtol, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('time', [0.0, np.array([0.0, DURATION, 2 * DURATION])])
def test_static_moment_feed_input_time_shape(data_path, time):
    """Broadcast static SI moments over array times without changing the feed.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    time : float or ndarray
        Input evaluation time(s) [s]; three times distinguish the moment axis.
    """
    unit, _ = inventory_unit(data_path, SemibatchCryst, gridless=True)
    unit.Inlet.y_upstream = None
    expected = np.array([4.5e8, 1.05e4, 0.285, 8.85e-6])
    # [m**n], independently integrated seed moments
    expected /= 2.0 + unit.Solid_1.kv * 8.85e-6  # [m**n/m**3]
    if np.ndim(time):
        expected = np.tile(expected, (len(time), 1))  # [m**n/m**3], time then order
    actual = unit.get_inputs(time)['Inlet']['mu_n']  # [m**n/m**3]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0)
    assert not hasattr(unit.Inlet, 'mu_n')


@pytest.mark.assimulo
@pytest.mark.parametrize('unit_type', [SemibatchCryst, MSMPR])
def test_gridless_feed_continuation_reports_si(data_path, unit_type):
    """Retain SI moments through two gridless feed-driven solves.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Semibatch or continuous crystallizer.
    """
    pytest.importorskip('assimulo')
    unit, _ = inventory_unit(data_path, unit_type, gridless=True)
    initial = np.array([4.5e8, 1.05e4, 0.285, 8.85e-6])  # [m**n], seed integrals
    volume = 2.0 + unit.Solid_1.kv * initial[3]  # [m**3], original slurry
    for start in (0.0, DURATION):  # [s], successive solves share their boundary
        unit.solve_unit(time_grid=[start, start + DURATION], verbose=False)
        times = unit.result.time  # [s]
        if unit_type is SemibatchCryst:
            expected = initial * (1 + FLOW / volume * times[:, None])  # [m**n]
        else:
            expected = np.tile(initial / volume, (len(times), 1))  # [m**n/m**3]
        np.testing.assert_allclose(unit.result.mu_n, expected, rtol=SOLVER_RTOL, atol=0)
    for profile in unit.profiles_runs:
        if unit_type is SemibatchCryst:
            expected = initial * (1 + FLOW / volume * profile['time'][:, None])  # [m**n]
        else:
            expected = np.tile(initial / volume, (len(profile['time']), 1))  # [m**n/m**3]
        np.testing.assert_allclose(profile['mu_n'], expected, rtol=SOLVER_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [SemibatchCryst, MSMPR])
@pytest.mark.parametrize('single_sample', [False, True])
def test_fvm_connection_supplies_missing_moment_feed(data_path, unit_type, single_sample):
    """Preserve FVM population histories through a real moment-mode connection.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Semibatch or continuous moment-mode destination.
    single_sample : bool
        Use one upstream sample, which must remain constant at later times.
    """
    from test_crystallizer_heat_duty import make_unit as make_heat_unit

    upstream, initial = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FLOW)
    profiles = np.tile(initial, (2, 1))  # [raw state units], two upstream times
    # Swap the first two bin populations: preserve values while changing shape.
    profiles[-1, :upstream.num_distr] = profiles[-1, [1, 0, 2]]
    upstream.retrieve_results(np.array([0.0, DURATION]), profiles)
    unit, states = inventory_unit(data_path, unit_type, gridless=True)
    Connection(upstream, unit).transfer_data()
    if single_sample:
        # Retain one connected sample to exercise the inlet's constant extrapolation.
        unit.Inlet.time_upstream = unit.Inlet.time_upstream[:1]  # [s]
        unit.Inlet.y_inlet = {name: values[:1]
                              for name, values in unit.Inlet.y_inlet.items()}
        # [field units], preserve one row of each converted field
        unit.Inlet.y_upstream = {**unit.Inlet.y_upstream,
                                 'mu_n': unit.Inlet.y_upstream['mu_n'][:1]}
        # [m**n/m**3], one raw upstream population sample
    assert 'mu_n' not in unit.Inlet.y_inlet
    # Independent trapezoids on x=[10,20,40] um have weights [5,15,10] um.
    # Initial amplitudes are [1,2,1]*1e7; final amplitudes are [2,1,1]*1e7.
    volume = 9e-4 + upstream.Solid_1.kv * 8.85e-6  # [m**3], fixed upstream slurry
    endpoint_moments = np.array([[4.5e8, 1.05e4, 0.285, 8.85e-6],
                                [3.5e8, 8e3, 0.23, 7.7e-6]]) / volume
    # [m**n/m**3], n=0..3, independent initial and final integrals
    if single_sample:
        endpoint_moments[1] = endpoint_moments[0]  # [m**n/m**3], hold sole sample
    times = np.array([0.0, DURATION / 2, DURATION])  # [s], include interpolation
    expected_si = np.array([endpoint_moments[0], endpoint_moments.mean(axis=0),
                            endpoint_moments[1]])  # [m**n/m**3], linear history
    states[:4] = 0  # [um**n] or [um**n/m**3], remove tank outflow of crystals
    for time, moments in zip(times, expected_si):
        # [s], [m**n/m**3], each independently specified inlet state
        rhs = unit.unit_model(time, states)  # [state unit/s], real material balance
        total_feed = rhs[:4]  # [um**n/s], Semibatch population basis
        if unit_type is MSMPR:
            total_feed = rhs[:4] * unit.Slurry.vol  # [um**n/s], intensive to total
        expected_feed = FLOW * moments * 1e6**np.arange(4)  # [um**n/s], exact um/m
        np.testing.assert_allclose(total_feed, expected_feed, rtol=RTOL, atol=0)
    inputs = unit.get_inputs(times)
    np.testing.assert_allclose(inputs['Inlet']['mu_n'], expected_si,
                               rtol=RTOL, atol=0)
    # Input retrieval must not mutate the connection's converted field mapping.
    assert 'mu_n' not in unit.Inlet.y_inlet


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [SemibatchCryst, MSMPR])
def test_connected_mu_n_takes_precedence_over_stream_moments(data_path, unit_type):
    """Retain upstream moment profiles when the stream holds a different endpoint.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Semibatch or continuous moment-mode destination.
    """
    upstream, initial = inventory_unit(data_path, MSMPR, gridless=True)
    sentinel = np.array([1e8, 4e3, 0.16, 6.4e-6])  # [m**n/m**3], 40 um crystals
    profiles = np.tile(initial, (2, 1))  # [raw state units], synthetic input trajectory
    profiles[:, :4] = np.array([sentinel, 2 * sentinel]) * 1e6**np.arange(4)
    # [um**n/m**3], doubled particle count at the endpoint distinguishes fallback
    upstream.retrieve_results(np.array([0.0, DURATION]), profiles)
    unit, _ = inventory_unit(data_path, unit_type, gridless=True)
    Connection(upstream, unit).transfer_data()
    unit.Inlet.y_upstream = {**unit.Inlet.y_upstream,
                             'mu_n': np.zeros_like(upstream.result.mu_n)}
    # [m**n/m**3], conflicting raw profile must not replace converted mu_n.
    np.testing.assert_allclose(unit.Inlet.moments, 2 * sentinel, rtol=RTOL, atol=0)
    np.testing.assert_allclose(unit.get_inputs(0.0)['Inlet']['mu_n'], sentinel,
                               rtol=RTOL, atol=0)
    inputs = unit.get_inputs(np.array([0.0, DURATION]))
    np.testing.assert_allclose(inputs['Inlet']['mu_n'], [sentinel, 2 * sentinel],
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_bare_slurry_inlet_keeps_existing_error(data_path, method):
    """Reject a phase inventory lacking the stream interface in either method.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    method : str
        Moment or FVM population representation.
    """
    batch, _ = make_unit(data_path, method)
    unit = MSMPR('A', method=method)
    unit.Phases = batch.Phases
    unit.Inlet = batch.Slurry
    # Slurry lacks DynamicInlet and vol_flow; preserve the existing error type.
    with pytest.raises(AttributeError):
        unit.get_inputs(0.0)  # [s]


@pytest.mark.unit
@pytest.mark.parametrize('unit_type', [SemibatchCryst, MSMPR])
@pytest.mark.parametrize('connected', [False, True])
def test_moment_feed_selects_required_orders(data_path, unit_type, connected):
    """Use a six-order feed in a four-order balance without changing the feed.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    unit_type : type
        Continuous or semibatch four-moment destination.
    connected : bool
        Transfer a real six-moment MSMPR profile instead of a static inlet.
    """
    upstream, initial = inventory_unit(data_path, MSMPR, gridless=True, num_mom=6)
    unit, states = inventory_unit(data_path, unit_type, gridless=True)
    if connected:
        upstream.retrieve_results(np.array([0.0, DURATION]), np.tile(initial, (2, 1)))
        Connection(upstream, unit).transfer_data()
    else:
        unit.Inlet = upstream.Inlet
        unit.Inlet.y_upstream = None
    # Independent trapezoids on the original [10,20,40] um fixture.
    volume = 2.0 + upstream.Solid_1.kv * 8.85e-6  # [m**3], seed slurry volume
    expected_si = np.array([4.5e8, 1.05e4, 0.285, 8.85e-6]) / volume
    # [m**n/m**3], required orders zero through three
    states[:4] = 0  # [um**n] or [um**n/m**3], no tank crystal outflow
    rhs = unit.unit_model(0.0, states)  # [state unit/s], public inlet handoff
    total_feed = rhs[:4]  # [um**n/s], semibatch inventory basis
    if unit_type is MSMPR:
        total_feed = total_feed * unit.Slurry.vol  # [um**n/s]
    expected_feed = FLOW * expected_si * 1e6**np.arange(4)  # [um**n/s]
    np.testing.assert_allclose(total_feed, expected_feed, rtol=RTOL, atol=0)
    times = np.array([0.0, DURATION / 2, DURATION])  # [s], unequal time/order axes
    np.testing.assert_allclose(unit.get_inputs(times)['Inlet']['mu_n'],
                               np.tile(expected_si, (3, 1)), rtol=RTOL, atol=0)
    assert len(unit.Inlet.moments) == 6
    if connected:
        assert unit.Inlet.y_inlet['mu_n'].shape == (2, 6)


@pytest.mark.unit
@pytest.mark.parametrize('connected', [False, True])
def test_moment_feed_rejects_missing_orders(data_path, connected):
    """Explain why a valid four-moment feed cannot supply a six-moment model.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    connected : bool
        Transfer a real MSMPR profile instead of using a static slurry stream.
    """
    upstream, initial = inventory_unit(data_path, MSMPR, gridless=True)
    unit, states = inventory_unit(data_path, MSMPR, gridless=True, num_mom=6)
    if connected:
        upstream.retrieve_results(np.array([0.0, DURATION]), np.tile(initial, (2, 1)))
        Connection(upstream, unit).transfer_data()
    else:
        unit.Inlet = upstream.Inlet
        unit.Inlet.y_upstream = None
    with pytest.raises(ValueError, match=r'Inlet mu_n must provide at least 6 moments.*got 4'):
        unit.unit_model(0.0, states)  # [s], [raw state units]
