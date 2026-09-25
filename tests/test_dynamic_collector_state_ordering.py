"""Regression tests for DynamicCollector result bookkeeping (issue #75).

The liquid tests drive the real ``DynamicCollector.solve_unit`` public path
with real inlet objects and thermodynamic data. Solver-backed liquid and
crystallizer paths run against the installed Assimulo backend; pre-solver input
validation remains in the core lane.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit

INLET_MASS_FLOW = 20.0  # [kg/s]
# Asymmetric composition over the four species in pfr_test_pure_comp.json
# (A, B, C, solv) so a shifted or truncated split is visible element-wise.
INLET_MASS_FRAC = np.array([0.1, 0.2, 0.3, 0.4])  # [-]
INLET_TEMP = 320.0  # [K]
RUNTIME = 5.0  # [s]


def test_liquid_mixer_requires_an_integration_end(data_path):
    """Reject a liquid solve with neither runtime nor requested times.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    inlet = LiquidStream(path, temp=INLET_TEMP, mass_flow=INLET_MASS_FLOW,
                         mass_frac=INLET_MASS_FRAC)
    collector = DynamicCollector()
    collector.Inlet = inlet

    message = (r"DynamicCollector\.solve_unit requires 'runtime' \[s\] or "
               r"'time_grid' \[s\]; neither was supplied\.")
    with pytest.raises(ValueError, match=message):
        collector.solve_unit()


@pytest.mark.assimulo
def test_liquid_mixer_result_labels_match_state_vector(data_path):
    """``result`` must label holdup mass [kg] and composition [-] correctly.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.

    Notes
    -----
    The holdup is created from the inlet stream, so its initial composition is
    the inlet composition and its temperature is the inlet temperature. With a
    constant inlet the only balance that moves is the total mass, whose exact
    solution over the integration interval is ``mass_flow * elapsed_time``
    [kg]. The zero-amount warning is an error because the real liquid phase
    must receive the positive initial mass used by the ODE state.
    """
    pytest.importorskip("assimulo")
    path = str(data_path["integration"] / "pfr_test_pure_comp.json")

    inlet = LiquidStream(path, temp=INLET_TEMP, mass_flow=INLET_MASS_FLOW,
                         mass_frac=INLET_MASS_FRAC)

    collector = DynamicCollector()
    collector.Inlet = inlet

    # Record warnings instead of erroring on RuntimeWarning alone: the
    # assertion then rejects a warning of any category and lists its message.
    with warnings.catch_warnings(record=True) as initialization_warnings:
        time, _ = collector.solve_unit(runtime=RUNTIME, verbose=False)
    assert not initialization_warnings, [
        str(warning.message) for warning in initialization_warnings]
    step = time[-1] - time[0]  # [s]

    result = collector.result

    # Composition is a mass fraction vector over the four species.
    num_times = len(time)  # [-]
    assert result.mass_frac.shape == (num_times, len(INLET_MASS_FRAC))
    np.testing.assert_allclose(result.mass_frac[0], INLET_MASS_FRAC,
                               rtol=1e-10)
    np.testing.assert_allclose(
        result.mass_frac.sum(axis=1), np.ones(num_times), rtol=1e-8
    )

    # Holdup mass accumulates the inlet mass flow: d(mass)/dt = mass_flow.
    assert result.mass.shape == (num_times,)
    assert result.mass[-1] - result.mass[0] == pytest.approx(
        INLET_MASS_FLOW * step, rel=1e-8)

    # Temperature is unchanged because inlet and holdup enthalpies coincide.
    np.testing.assert_allclose(
        result.temp, np.full(num_times, INLET_TEMP), rtol=1e-8
    )

    # `outputs` is the same labeled mapping handed downstream.
    np.testing.assert_allclose(collector.outputs["mass"], result.mass,
                               rtol=1e-10)
    np.testing.assert_allclose(collector.outputs["mass_frac"],
                               result.mass_frac, rtol=1e-10)

    # Profile attributes use the same metadata-derived state mapping.
    np.testing.assert_allclose(collector.wConcProf, result.mass_frac,
                               rtol=1e-10)
    np.testing.assert_allclose(collector.massProf, result.mass, rtol=1e-10)
    np.testing.assert_allclose(collector.tempProf, result.temp, rtol=1e-10)


# --- Crystallizer-mode result and plotting dispatch ------------------------

SLURRY_VOL_FLOW = 1e-4  # [m**3/s]
SOLID_MASS_FRAC = np.array([0.0, 0.0, 1.0, 0.0])  # [-], solid is species C
CRYST_RUNTIME = 0.01  # [s], short real-backend integration interval
NUM_CRYST_BINS = 15  # [-]


def _configured_crystallizing_collector(slurry_inlet):
    """Build a collector with real crystallization kinetics and inlet.

    Parameters
    ----------
    slurry_inlet : SlurryStream
        Real crystallizing inlet with composition [-], flow [m**3/s],
        temperature [K], and distribution [#/m**3/um] states.

    Returns
    -------
    DynamicCollector
        Production collector configured for a short Assimulo solve.

    Notes
    -----
    The crystallization constants and ``scale`` reproduce the established
    ``test_PFR_HOLD_BC_FILT`` integration case in
    ``tests/Flowsheet/flowsheet_tests.py``. The shorter runtime is sufficient
    to exercise construction, retrieval, and plotting without changing that
    case's physical parameterization.
    """
    solubility_coefficients = np.array([2.269e2, -1.88, 3.89e-3])
    # [kg/m**3], [kg/m**3/K], [kg/m**3/K**2], empirical test correlation
    kinetics = CrystKinetics(
        solubility_coefficients,
        nucl_prim=(3e8, 0.0, 3.0),  # [#/m**3/s], [J/mol], [-]
        nucl_sec=(4.46e10, 0.0, 2.0, 1e-5),  # [#/m**3/s], [J/mol], [-], [-]
        growth=(5.0, 0.0, 1.32),  # [um/s], [J/mol], [-]
        dissolution=(1.0, 0.0, 1.0),  # [um/s], [J/mol], [-]
    )
    collector = DynamicCollector()
    collector.Inlet = slurry_inlet
    collector.KinCryst = kinetics
    collector.kwargs_cryst = {
        "target_ind": 2,  # species C index [-]
        "target_comp": ["C"],
        "scale": 1e-9,  # [-], stabilizes the FVM distribution state
    }
    return collector


@pytest.fixture
def slurry_inlet(data_path):
    """Real crystallizing inlet built from the repository thermo data.

    A ``DynamicCollector`` only ever receives a crystallizing inlet from an
    upstream unit, so ``y_inlet``/``y_upstream`` are populated here the way
    ``PharmaPy.Connections.Connection`` populates them; without them the inlet
    concentration would fall back to the zero-filled missing-state default.
    """
    path = str(data_path["integration"] / "pfr_test_pure_comp.json")

    x_distrib = np.geomspace(1.0, 1500.0, num=NUM_CRYST_BINS)  # [um]
    # Narrow seed distribution centred on the grid [#/m**3/um].
    distrib = np.exp(-((np.log(x_distrib) - np.log(100.0)) / 0.5)**2) * 1e10

    liquid = LiquidStream(path, temp=310.0, mass_frac=[0.1, 0.2, 0.3, 0.4],
                          vol_flow=SLURRY_VOL_FLOW)
    solid = SolidStream(path, temp=310.0, x_distrib=x_distrib, distrib=distrib,
                        mass_frac=SOLID_MASS_FRAC)

    inlet = SlurryStream(vol_flow=SLURRY_VOL_FLOW, x_distrib=x_distrib,
                         distrib=distrib)
    inlet.Phases = [liquid, solid]

    inlet.y_inlet = {'mass_conc': liquid.mass_conc,  # [kg/m**3]
                     'vol_flow': SLURRY_VOL_FLOW,  # [m**3/s]
                     'temp': inlet.temp,  # [K]
                     'distrib': distrib}  # [#/m**3/um]
    inlet.y_upstream = inlet.y_inlet
    inlet.time_upstream = None

    return inlet


def test_inlet_assignment_sets_collector_model_mode(data_path, slurry_inlet):
    """Derive and refresh collector mode whenever an inlet is assigned.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    slurry_inlet : SlurryStream
        Real crystallizing inlet with composition [-], flow [m**3/s],
        temperature [K], and distribution [#/m**3/um] states.
    """
    collector = DynamicCollector()

    assert collector.is_cryst is False

    collector.Inlet = slurry_inlet

    assert collector.is_cryst is True

    path = str(data_path["integration"] / "pfr_test_pure_comp.json")
    liquid_inlet = LiquidStream(path, temp=INLET_TEMP,
                                mass_flow=INLET_MASS_FLOW,
                                mass_frac=INLET_MASS_FRAC)
    collector.Inlet = liquid_inlet

    assert collector.is_cryst is False


@pytest.mark.assimulo
def test_crystallizer_collector_delegates_plotting(slurry_inlet):
    """A crystallizing collector must plot through its crystallizer sub-model.

    ``plot_local`` reads liquid-shaped slices of the state vector, which for a
    crystallizer solve are crystal-size-distribution bins. After a crystallizer
    solve, ``plot_profiles`` must therefore delegate to the sub-model's own
    plotter instead.
    """
    pytest.importorskip("assimulo")
    collector = _configured_crystallizing_collector(slurry_inlet)
    # Plausible stale liquid profiles ensure a wrong local-plot branch reaches
    # the delegation assertions instead of failing on a missing attribute.
    collector.timeProf = np.array([0.0, CRYST_RUNTIME])  # [s]
    collector.wConcProf = np.tile(SOLID_MASS_FRAC, (2, 1))  # [-]
    collector.massProf = np.array([1.0, 2.0])  # [kg]
    collector.tempProf = np.array([310.0, 311.0])  # [K]

    collector.solve_unit(
        runtime=CRYST_RUNTIME,
        time_grid=np.array([0.0, CRYST_RUNTIME]),  # [s]
        verbose=False,
    )

    assert collector.model_type == 'crystallizer'
    assert isinstance(collector.CrystInst, SemibatchCryst)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="No artists with labels found to put in legend",
            category=UserWarning,
        )
        fig, axes = collector.plot_profiles()

    assert fig is not None
    assert np.asarray(axes).shape == (3, 2)
    plt.close(fig)


@pytest.mark.assimulo
def test_crystallizer_plot_forwards_figure_size(slurry_inlet):
    """Delegated crystallizer plots honour the collector figure size [in].

    Parameters
    ----------
    slurry_inlet : SlurryStream
        Real crystallizing inlet with composition [-], flow [m**3/s],
        temperature [K], and distribution [#/m**3/um] states.
    """
    pytest.importorskip("assimulo")
    collector = _configured_crystallizing_collector(slurry_inlet)
    collector.solve_unit(
        runtime=CRYST_RUNTIME,
        time_grid=np.array([0.0, CRYST_RUNTIME]),  # [s]
        verbose=False,
    )

    requested_figure_size = (9.0, 7.0)  # [in]
    plot_kwargs = {}
    fig, _ = collector.plot_profiles(
        fig_size=requested_figure_size, kwargs=plot_kwargs
    )

    np.testing.assert_allclose(fig.get_size_inches(), requested_figure_size)
    assert plot_kwargs == {}
    plt.close(fig)

    explicit_figure_size = (8.0, 6.0)  # [in]
    explicit_kwargs = {"figsize": explicit_figure_size}
    fig, _ = collector.plot_profiles(
        fig_size=requested_figure_size, kwargs=explicit_kwargs
    )

    np.testing.assert_allclose(fig.get_size_inches(), explicit_figure_size)
    assert explicit_kwargs == {"figsize": explicit_figure_size}
    plt.close(fig)


@pytest.mark.assimulo
def test_crystallizer_results_skip_liquid_profile_slicing(slurry_inlet):
    """Crystallizer retrieval must not overwrite liquid-profile attributes.

    Parameters
    ----------
    slurry_inlet : SlurryStream
        Real crystallizing inlet with composition [-], flow [m**3/s],
        temperature [K], and distribution [#/m**3/um] states.
    """
    pytest.importorskip("assimulo")
    collector = _configured_crystallizing_collector(slurry_inlet)
    stale_mass_fractions = np.tile(SOLID_MASS_FRAC, (2, 1))  # [-]
    stale_masses = np.array([1.0, 2.0])  # [kg]
    stale_temperatures = np.array([310.0, 311.0])  # [K]
    collector.wConcProf = stale_mass_fractions.copy()
    collector.massProf = stale_masses.copy()
    collector.tempProf = stale_temperatures.copy()

    time, states = collector.solve_unit(
        runtime=CRYST_RUNTIME,
        time_grid=np.array([0.0, CRYST_RUNTIME]),  # [s]
        verbose=False,
    )

    assert np.shape(states)[0] == len(time)
    assert collector.result is collector.CrystInst.result
    np.testing.assert_array_equal(collector.wConcProf, stale_mass_fractions)
    np.testing.assert_array_equal(collector.massProf, stale_masses)
    np.testing.assert_array_equal(collector.tempProf, stale_temperatures)
