"""Mixed old/new flowsheet tests.

Since the phase stack was split, old unit operations import PharmaPy.Phases /
Streams / MixedPhases while the MultiPhaseVessel units import the *_Refactored
modules. This script asks whether the two can be wired together in one
SimulationExec flowsheet.

It runs in stages, cheapest first, so that a failure in a mixed flowsheet can be
told apart from a mistake in how the units are driven:

    Stage 0  old Filter on its own            -- is the Filter being used right?
    Stage 1  all-old  R01 -> CR01 -> F01      -- is the harness right?
    Stage 2  new R01 -> new CR01 -> old F01
    Stage 3  new R01 -> old HOLD01 -> new CR01 -> old F01

Stages 0 and 1 are controls. If they pass and 2/3 fail, the failure is a real
old/new interoperability gap rather than a usage error.
"""

import os
import traceback

import numpy as np

# ---------------------------------------------------------------- old stack
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Reactors import BatchReactor as OldBatchReactor
from PharmaPy.Crystallizers import BatchCryst as OldBatchCryst
from PharmaPy.Containers import DynamicCollector
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Interpolation import PiecewiseLagrange

# ---------------------------------------------------------------- new stack
from PharmaPy.Phases_Refactored import (LiquidPhase as NewLiquidPhase,
                                        SolidPhase as NewSolidPhase)
from PharmaPy.Streams_Refactored import LiquidStream as NewLiquidStream
from PharmaPy.Reactors_Refactored import (BatchReactor as NewBatchReactor,
                                          SemiBatchReactor as NewSemiReactor,
                                          ContinuousReactor as NewContReactor)
from PharmaPy.Crystallizers_Refactored import (
    BatchCrystallizer as NewBatchCryst,
    SemiBatchCrystallizer as NewSemiBatchCryst,
    ContinuousCrystallizer as NewContCryst)
from PharmaPy.Mechanisms import OneDFVMMechanism
from PharmaPy.IntegratorBackends import AssimuloBackend
from PharmaPy.ProcessControl_Refactored import (SimpleTemperatureController,
                                                ContinuousVesselController)

# ---------------------------------------------------------------- shared
from PharmaPy.Kinetics import RxnKinetics, CrystKinetics
from PharmaPy.Utilities import CoolingWater

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(os.path.dirname(HERE), 'tests', 'Flowsheet', 'data',
                    'compound_database.json')

# Same chemistry and kinetics as tests/Flowsheet/flowsheet_tests.py, so any
# difference is attributable to the unit implementations rather than the model.
RXNS = ['A + B --> C', 'C + A --> D']
K_VALS = np.array([2.654e4, 5.3e2])
EA_VALS = np.array([4.0e4, 3.0e4])

PRIM = (3e8, 0, 3)
SEC = (4.46e10, 0, 2, 1e-5)
GROWTH = (5, 0, 1.32)
DISSOL = (1, 0, 1)
SOLUB = np.array([2.269e2, -1.88e0, 3.89e-3])

X_GR = np.geomspace(1, 1500, num=35)
MASSFRAC_SOLID = [0, 0, 1, 0, 0]

TEMP_INIT = 313.15
CONC_INIT = np.array([0.33, 0.33, 0, 0, 0])
VOL_INIT = 0.06

FEED_VOLFLOW = 1e-5      # m3/s

TIME_R01 = 3600.0
TIME_CR01 = TIME_R01 * 2.0

DELTA_P = 101325.0
FILT_AREA = 200          # cm**2
FILT_DIAM = np.sqrt(4 / np.pi * FILT_AREA) / 100   # m
ALPHA = 1e11
RESIST_MEDIUM = 1e10

# The refactored vessels compute u_ht = 1/(1/h_conv + 1/utility.h_conv) the
# moment Utility is assigned, so h_conv must be non-zero; area_ht is
# 4*vol/diam, so diam must be > 0. Both default to 0 on MultiPhaseVessel.
H_CONV = 10000.0         # W/m**2/K
VESSEL_DIAM = 0.4        # m -> ~0.6 m**2 jacket area at 0.06 m**3

TEMP_PROGRAM = np.array([[313.15, 308],
                         [308, 295],
                         [295, 278.15]], dtype=np.float64)


def rxn_kinetics():
    return RxnKinetics(path=PATH, rxn_list=RXNS, k_params=K_VALS,
                       ea_params=EA_VALS)


def cryst_kinetics():
    return CrystKinetics(SOLUB, nucl_prim=PRIM, nucl_sec=SEC, growth=GROWTH,
                         dissolution=DISSOL)


def cooling_profile(runtime=TIME_CR01):
    return PiecewiseLagrange(runtime, TEMP_PROGRAM).evaluate_poly


def make_filter():
    return Filter(FILT_DIAM, ALPHA, RESIST_MEDIUM)


def quiet(run_kwargs):
    for key in run_kwargs:
        run_kwargs[key]['verbose'] = False
    return run_kwargs


# =====================================================================
# Stage 0 -- drive an old Filter directly. Control for Filter usage.
# =====================================================================
def stage0_filter_alone():
    vol_liq = 2750e-6
    liquid = LiquidPhase(path_thermo=PATH, vol=vol_liq,
                         mass_frac=[0, 0, 0, 0, 1])

    mass_solids = vol_liq * 2.4e-2 * 1e3
    x_distr = np.arange(1, 501)
    distrib = np.ones_like(x_distr)
    solid = SolidPhase(PATH, mass=mass_solids, x_distrib=x_distr,
                       distrib=distrib, mass_frac=MASSFRAC_SOLID)

    slurry = Slurry()
    slurry.Phases = (solid, liquid)

    filt = make_filter()
    filt.Phases = slurry
    filt.solve_unit(deltaP=DELTA_P, verbose=False)

    return 'filtration time = %.1f s' % filt.timeProf[-1]


# =====================================================================
# Stage 1 -- all-old flowsheet. Control for the SimulationExec harness.
# =====================================================================
def stage1_all_old():
    flst = SimulationExec(PATH, flowsheet='R01 --> CR01 --> F01')

    liquid_init = LiquidPhase(PATH, temp=TEMP_INIT, mole_conc=CONC_INIT.copy(),
                              vol=VOL_INIT, name_solv='solvent')

    flst.R01 = OldBatchReactor(isothermal=False)
    flst.R01.Utility = CoolingWater(mass_flow=0.01, temp_in=TEMP_INIT)
    flst.R01.Phases = liquid_init
    flst.R01.Kinetics = rxn_kinetics()

    solid_cry = SolidPhase(PATH, x_distrib=X_GR, distrib=np.zeros_like(X_GR),
                           mass_frac=MASSFRAC_SOLID)
    flst.CR01 = OldBatchCryst(target_comp='C', method='1D-FVM', scale=1e-9,
                              controls={'temp': cooling_profile()})
    flst.CR01.Kinetics = cryst_kinetics()
    flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)
    flst.CR01.Phases = solid_cry

    flst.F01 = make_filter()

    run_kwargs = quiet({
        'R01': {'runtime': TIME_R01},
        'CR01': {'runtime': TIME_CR01, 'sundials_opts': {'maxh': 60}},
        'F01': {'runtime': None, 'deltaP': DELTA_P},
    })
    flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)

    return 'filtration time = %.1f s' % flst.F01.timeProf[-1]


# =====================================================================
# Stage 2 -- new reactor -> new crystallizer -> old filter
# =====================================================================
def stage2_new_new_old():
    flst = SimulationExec(PATH, flowsheet='R01 --> CR01 --> F01')

    liquid_init = NewLiquidPhase(PATH, temp=TEMP_INIT,
                                 mole_conc=CONC_INIT.copy(), vol=VOL_INIT,
                                 name_solv='solvent')

    flst.R01 = NewBatchReactor(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=SimpleTemperatureController(
            temp_func=lambda t: TEMP_INIT))
    flst.R01.Phases = liquid_init
    flst.R01.RxnKinetics = rxn_kinetics()
    flst.R01.Utility = CoolingWater(mass_flow=0.01, temp_in=TEMP_INIT)

    cryst_liquid = NewLiquidPhase(PATH, temp=TEMP_INIT,
                                  mole_conc=CONC_INIT.copy(), vol=VOL_INIT,
                                  name_solv='solvent')
    cryst_solid = NewSolidPhase(PATH, mass=0.0, mass_frac=MASSFRAC_SOLID)
    cryst_solid.mechanisms = OneDFVMMechanism(
        cryst_solid, target_components='C', solvent_name='solvent',
        x_grid=X_GR, distrib_init=np.zeros_like(X_GR), scale=1e-9)

    flst.CR01 = NewBatchCryst(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=SimpleTemperatureController(temp_func=cooling_profile()))
    flst.CR01.Phases = [cryst_liquid, cryst_solid]
    flst.CR01.CrystKinetics = cryst_kinetics()
    flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)

    flst.F01 = make_filter()

    run_kwargs = quiet({
        'R01': {'runtime': TIME_R01},
        'CR01': {'runtime': TIME_CR01},
        'F01': {'runtime': None, 'deltaP': DELTA_P},
    })
    flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)

    return 'filtration time = %.1f s' % flst.F01.timeProf[-1]


# =====================================================================
# Stage 3 -- new continuous reactor -> old hold -> new semibatch cryst
#            -> old filter
# =====================================================================
def stage3_new_old_new_old():
    flst = SimulationExec(PATH, flowsheet='R01 --> HOLD01 --> CR01 --> F01')

    liquid_init = NewLiquidPhase(PATH, temp=TEMP_INIT,
                                 mole_conc=CONC_INIT.copy(), vol=VOL_INIT,
                                 name_solv='solvent')
    feed = NewLiquidStream(PATH, temp=TEMP_INIT,
                           mole_conc=np.array([0.5, 0.5, 0, 0, 0]),
                           vol_flow=1e-5, name_solv='solvent')

    flst.R01 = NewContReactor(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=ContinuousVesselController(
            temp_func=lambda t: TEMP_INIT))
    flst.R01.Phases = liquid_init
    flst.R01.Inlet = feed
    flst.R01.RxnKinetics = rxn_kinetics()
    flst.R01.Utility = CoolingWater(mass_flow=0.01, temp_in=TEMP_INIT)

    flst.HOLD01 = DynamicCollector()

    cryst_liquid = NewLiquidPhase(PATH, temp=TEMP_INIT,
                                  mole_conc=CONC_INIT.copy(), vol=VOL_INIT,
                                  name_solv='solvent')
    cryst_solid = NewSolidPhase(PATH, mass=0.0, mass_frac=MASSFRAC_SOLID)
    cryst_solid.mechanisms = OneDFVMMechanism(
        cryst_solid, target_components='C', solvent_name='solvent',
        x_grid=X_GR, distrib_init=np.zeros_like(X_GR), scale=1e-9)

    flst.CR01 = NewSemiBatchCryst(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=SimpleTemperatureController(temp_func=cooling_profile()))
    flst.CR01.Phases = [cryst_liquid, cryst_solid]
    flst.CR01.CrystKinetics = cryst_kinetics()
    flst.CR01.Utility = CoolingWater(mass_flow=1, temp_in=283.15)

    flst.F01 = make_filter()

    run_kwargs = quiet({
        'R01': {'runtime': TIME_R01},
        'HOLD01': {'runtime': TIME_R01},
        'CR01': {'runtime': TIME_CR01},
        'F01': {'runtime': None, 'deltaP': DELTA_P},
    })
    flst.SolveFlowsheet(kwargs_run=run_kwargs, verbose=False)

    # The crystallizer must actually process what the holding vessel
    # collected. This stage used to report only a filtration time, and the
    # whole HOLD01 -> CR01 link was a no-op: Connections handed CR01
    # 32.6 kg, the semibatch branch dropped it because CR01 already had
    # Phases, and CR01 ran on its own initial charge instead.
    # The stack conversion used to build a fresh legacy object and leave
    # y_upstream / time_upstream / y_inlet behind, so HOLD01 ran on R01's
    # constant final snapshot. Checked structurally because the effect on
    # the collected mass is only a fraction of a percent -- far too small
    # for an end-to-end number to catch reliably.
    hold_inlet = flst.HOLD01.Inlet

    if getattr(hold_inlet, 'y_upstream', None) is None:
        raise AssertionError(
            'HOLD01 lost R01 trajectory (y_upstream) crossing the stacks')

    if np.ndim(getattr(hold_inlet, 'time_upstream', None)) == 0:
        raise AssertionError(
            'HOLD01 got a scalar time from a continuous source, so the '
            'trajectory was dropped crossing the stacks')

    if not getattr(hold_inlet, 'y_inlet', None):
        raise AssertionError(
            'HOLD01 lost the converted inlet states (y_inlet) crossing '
            'the stacks')

    collected = float(flst.HOLD01.Outlet.mass)

    if collected <= 0:
        raise AssertionError('HOLD01 collected nothing from R01')

    charged = float(np.asarray(flst.CR01.result.mass_j_liquid0)[0].sum())
    drift = abs(charged - collected) / collected

    if drift > 1e-6:
        raise AssertionError(
            'CR01 did not start from the HOLD01 contents: HOLD01 collected '
            '%.6f kg but CR01 began with %.6f kg (rel %.3e)'
            % (collected, charged, drift))

    return ('filtration time = %.1f s, CR01 charged with HOLD01 %.3f kg'
            % (flst.F01.timeProf[-1], collected))



# =====================================================================
# new -> new flowsheets. These assert, rather than only checking that
# SolveFlowsheet returned without raising.
# =====================================================================
def _new_liquid():
    return NewLiquidPhase(PATH, temp=TEMP_INIT, mole_conc=CONC_INIT.copy(),
                          vol=VOL_INIT, name_solv='solvent')


def _new_feed(mole_conc=None, vol_flow=FEED_VOLFLOW):
    if mole_conc is None:
        mole_conc = np.array([0.5, 0.5, 0, 0, 0])

    return NewLiquidStream(PATH, temp=TEMP_INIT, mole_conc=mole_conc,
                           vol_flow=vol_flow, name_solv='solvent')


def _new_reactor(cls, inlet=None, controller=None):
    unit = cls(integrator=AssimuloBackend(options={'maxh': 60}),
               h_conv=H_CONV, diam=VESSEL_DIAM,
               controller=controller or SimpleTemperatureController(
                   temp_func=lambda t: TEMP_INIT))
    unit.Phases = _new_liquid()
    unit.RxnKinetics = rxn_kinetics()

    if inlet is not None:
        unit.Inlet = inlet

    return unit


def _continuous_reactor():
    return _new_reactor(
        NewContReactor, inlet=_new_feed(),
        controller=ContinuousVesselController(temp_func=lambda t: TEMP_INIT))


def stage4_new_batch_to_batch():
    """The handoff must be exact, and splitting a batch must reproduce it."""
    flst = SimulationExec(PATH, flowsheet='R01 --> R02')
    flst.R01 = _new_reactor(NewBatchReactor)
    flst.R02 = _new_reactor(NewBatchReactor)

    split_a, split_b = 300.0, 3300.0
    flst.SolveFlowsheet(kwargs_run=quiet({'R01': {'runtime': split_a},
                                          'R02': {'runtime': split_b}}),
                        verbose=False)

    if flst.result is None:
        raise AssertionError('SolveFlowsheet left flst.result unset')

    handed_over = np.asarray(flst.R01.Phases.Liquids[0].mass_j)
    received = np.asarray(flst.R02.result.mass_j_liquid0)[0]
    scale = max(np.abs(handed_over).max(), 1e-30)
    drift = np.abs(received - handed_over).max() / scale

    if drift > 1e-12:
        raise AssertionError('handoff lost material: rel %.3e' % drift)

    # The same chemistry run as one long batch must land in the same place.
    whole = _new_reactor(NewBatchReactor)
    whole.solve_unit(runtime=split_a + split_b)

    split_end = np.asarray(flst.R02.result.mole_conc_liquid0)[-1]
    whole_end = np.asarray(whole.result.mole_conc_liquid0)[-1]
    cscale = max(np.abs(whole_end).max(), 1e-30)
    gap = np.abs(split_end - whole_end).max() / cscale

    if gap > 1e-5:
        raise AssertionError('split batch != single batch: rel %.3e' % gap)

    return 'handoff exact (%.1e), split==single (%.1e)' % (drift, gap)


def stage5_new_continuous_to_continuous():
    """The downstream unit must follow the upstream trajectory.

    The control is the case that used to pass: a standalone unit fed the
    upstream's *final* outlet, held constant. While the trajectory was being
    ignored the two agreed to 1e-15, so this asserts they now disagree.
    """
    flst = SimulationExec(PATH, flowsheet='R01 --> R02')
    flst.R01 = _continuous_reactor()
    flst.R02 = _continuous_reactor()
    flst.SolveFlowsheet(kwargs_run=quiet({'R01': {'runtime': TIME_R01},
                                          'R02': {'runtime': TIME_R01}}),
                        verbose=False)

    downstream = np.asarray(flst.R02.result.mole_conc_liquid0)[-1]
    upstream_end = np.asarray(flst.R01.result.mole_conc_liquid0)[-1]

    control = _new_reactor(
        NewContReactor,
        inlet=_new_feed(mole_conc=upstream_end,
                        vol_flow=float(flst.R01.Outlet.vol_flow)),
        controller=ContinuousVesselController(temp_func=lambda t: TEMP_INIT))
    control.solve_unit(runtime=TIME_R01)
    snapshot = np.asarray(control.result.mole_conc_liquid0)[-1]

    scale = max(np.abs(snapshot).max(), 1e-30)
    divergence = np.abs(downstream - snapshot).max() / scale

    if divergence < 1e-6:
        raise AssertionError(
            'downstream still matches a constant snapshot (rel %.3e), so the '
            'upstream trajectory is being ignored' % divergence)

    return 'differs from constant-snapshot control by %.3e' % divergence


def stage6_new_continuous_to_semibatch():
    """No holding vessel needed: a semibatch vessel accepts a flow."""
    flst = SimulationExec(PATH, flowsheet='R01 --> R02')
    flst.R01 = _continuous_reactor()
    flst.R02 = _new_reactor(NewSemiReactor, inlet=_new_feed())
    flst.SolveFlowsheet(kwargs_run=quiet({'R01': {'runtime': TIME_R01},
                                          'R02': {'runtime': TIME_R01}}),
                        verbose=False)

    # Reporting only the final mass let a completely ignored connection
    # pass: R02 kept its own static feed because the semibatch branch in
    # Connections dropped the transferred matter. Assert that R02 follows
    # the upstream trajectory the way stage 5 does -- a standalone run fed
    # R01's constant final outlet must give a different answer.
    downstream = np.asarray(flst.R02.result.mole_conc_liquid0)[-1]
    upstream_end = np.asarray(flst.R01.result.mole_conc_liquid0)[-1]

    # Two failure modes need two controls, because 'differs from a
    # snapshot' alone is satisfied by a connection that was ignored
    # outright -- which is how this stage passed while R02 ran entirely on
    # its own feed.
    #
    # A fed vessel must be told how fast it is being fed. names_states_in
    # used to mirror names_states_out, which says 'vol' for a semibatch
    # vessel, so NameAnalyzer had nothing to map the upstream 'vol_flow'
    # onto and the feed rate vanished from the trajectory -- leaving R02
    # following the upstream composition at its own configured flow.
    stream = flst.R02.inlet_connections[0].stream
    arrived = getattr(stream, 'y_inlet', None)

    if not arrived:
        # Connections writes onto the wrapper or the inner phase
        # depending on what it was handed; check both.
        for phase in stream:
            arrived = getattr(phase, 'y_inlet', None)
            if arrived:
                break

    if not arrived or 'vol_flow' not in arrived:
        raise AssertionError(
            'R02 inlet trajectory has no feed rate: got %s. A semibatch '
            'vessel must declare vol_flow as an inlet state.'
            % (sorted(arrived) if arrived else None))

    # (a) connection dropped: R02 would match a standalone run on the feed
    #     it was configured with.
    ignored = _new_reactor(NewSemiReactor, inlet=_new_feed())
    ignored.solve_unit(runtime=TIME_R01)
    ignored_end = np.asarray(ignored.result.mole_conc_liquid0)[-1]

    iscale = max(np.abs(ignored_end).max(), 1e-30)
    gap = np.abs(downstream - ignored_end).max() / iscale

    if gap < 1e-6:
        raise AssertionError(
            'R02 matches a standalone run on its own configured feed '
            '(rel %.3e), so the connection from R01 is being ignored' % gap)

    # (b) connected but frozen: R02 would match a run fed R01's constant
    #     final outlet.
    control = _new_reactor(
        NewSemiReactor,
        inlet=_new_feed(mole_conc=upstream_end,
                        vol_flow=float(flst.R01.Outlet.vol_flow)))
    control.solve_unit(runtime=TIME_R01)
    snapshot = np.asarray(control.result.mole_conc_liquid0)[-1]

    scale = max(np.abs(snapshot).max(), 1e-30)
    divergence = np.abs(downstream - snapshot).max() / scale

    if divergence < 1e-6:
        raise AssertionError(
            'downstream still matches a constant snapshot (rel %.3e), so '
            'the upstream trajectory is being ignored' % divergence)

    return ('R02 final mass %.6f kg, differs from constant-snapshot '
            'control by %.3e' % (flst.R02.Phases.mass, divergence))


def stage7_new_continuous_to_batch_refused():
    """Continuous -> Batch needs a hold, and must say so."""
    flst = SimulationExec(PATH, flowsheet='R01 --> R02')
    flst.R01 = _continuous_reactor()
    flst.R02 = _new_reactor(NewBatchReactor)

    try:
        flst.SolveFlowsheet(kwargs_run=quiet({'R01': {'runtime': TIME_R01},
                                              'R02': {'runtime': TIME_R01}}),
                            verbose=False)
    except TypeError as exc:
        if 'holding vessel' not in str(exc):
            raise AssertionError('refused, but not for the right reason: %s'
                                 % exc) from None
        return 'refused, naming the holding vessel'

    raise AssertionError('accepted a continuous feed into a batch unit')


# =====================================================================
# Stage 8 -- every continuous -> semibatch pairing of the two vessel
# kinds. A crystallizer source is the hard case: its outlet carries a
# crystal size distribution, which is owned by a mechanism rather than
# by the phase.
# =====================================================================
TEMP_CRYST = 278.15      # below the 4.94 kg/m3 solubility of C
CONC_CRYST = 40.0        # kg/m3 of C, comfortably supersaturated


def _cryst_solid():
    solid = NewSolidPhase(PATH, mass=0.0, mass_frac=MASSFRAC_SOLID)
    solid.mechanisms = OneDFVMMechanism(
        solid, target_components='C', solvent_name='solvent',
        x_grid=X_GR, distrib_init=np.zeros_like(X_GR), scale=1e-9)
    return solid


def _continuous_cryst():
    conc = np.array([0., 0., CONC_CRYST, 0., 0.])
    unit = NewContCryst(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=ContinuousVesselController(
            temp_func=lambda t: TEMP_CRYST))
    unit.Phases = [NewLiquidPhase(PATH, temp=TEMP_CRYST, mass_conc=conc,
                                  vol=VOL_INIT, name_solv='solvent'),
                   _cryst_solid()]
    unit.CrystKinetics = cryst_kinetics()
    unit.Inlet = NewLiquidStream(PATH, temp=TEMP_CRYST, mass_conc=conc,
                                 vol_flow=FEED_VOLFLOW,
                                 name_solv='solvent')
    unit.Utility = CoolingWater(mass_flow=1, temp_in=TEMP_CRYST)
    return unit


def _semibatch_cryst():
    unit = NewSemiBatchCryst(
        integrator=AssimuloBackend(options={'maxh': 60}),
        h_conv=H_CONV, diam=VESSEL_DIAM,
        controller=SimpleTemperatureController(
            temp_func=cooling_profile(TIME_R01)))
    unit.Phases = [_new_liquid(), _cryst_solid()]
    unit.CrystKinetics = cryst_kinetics()
    unit.Inlet = _new_feed()
    unit.Utility = CoolingWater(mass_flow=1, temp_in=283.15)
    return unit


def stage8_continuous_to_semibatch_matrix():
    """Each continuous source feeding each semibatch destination."""
    cases = (
        ('reactor -> reactor', _continuous_reactor,
         lambda: _new_reactor(NewSemiReactor, inlet=_new_feed()), False),
        ('reactor -> crystallizer', _continuous_reactor,
         _semibatch_cryst, False),
        ('crystallizer -> crystallizer', _continuous_cryst,
         _semibatch_cryst, True),
    )

    results = []

    for label, make_up, make_down, expect_distrib in cases:
        flst = SimulationExec(PATH, flowsheet='U01 --> U02')
        flst.U01 = make_up()
        flst.U02 = make_down()
        flst.SolveFlowsheet(
            kwargs_run=quiet({'U01': {'runtime': TIME_R01},
                              'U02': {'runtime': TIME_R01}}),
            verbose=False)

        stream = flst.U02.inlet_connections[0].stream
        arrived = getattr(stream, 'y_inlet', None)

        if not arrived:
            for phase in stream:
                arrived = getattr(phase, 'y_inlet', None)
                if arrived:
                    break

        if not arrived:
            raise AssertionError(
                '%s: downstream received no trajectory at all' % label)

        if 'vol_flow' not in arrived:
            raise AssertionError(
                '%s: trajectory carries no feed rate (%s)'
                % (label, sorted(arrived)))

        # A crystallizer source must hand over its size distribution; that
        # used to be refused outright, and before that the connection
        # could not even be built.
        if expect_distrib and 'distrib' not in arrived:
            raise AssertionError(
                '%s: crystallizer source passed no distribution (%s)'
                % (label, sorted(arrived)))

        results.append(label)

    # A reactor has no solid phase, so a slurry feed must be refused with
    # an error that says so rather than a bare KeyError on a PhaseRef.
    flst = SimulationExec(PATH, flowsheet='U01 --> U02')
    flst.U01 = _continuous_cryst()
    flst.U02 = _new_reactor(NewSemiReactor, inlet=_new_feed())

    try:
        flst.SolveFlowsheet(
            kwargs_run=quiet({'U01': {'runtime': TIME_R01},
                              'U02': {'runtime': TIME_R01}}),
            verbose=False)
    except ValueError as exc:
        if 'none to receive it' not in str(exc):
            raise AssertionError(
                'crystallizer -> reactor raised the wrong error: %s' % exc
            ) from None
    else:
        raise AssertionError(
            'a reactor accepted a solid phase it cannot hold')

    return '%d pairings connected; cryst -> reactor refused' % len(results)


STAGES = (
    ('0  old Filter alone                                 ', stage0_filter_alone),
    ('1  all-old   R01 -> CR01 -> F01                     ', stage1_all_old),
    ('2  new R01 -> new CR01 -> old F01                   ', stage2_new_new_old),
    ('3  new R01 -> old HOLD01 -> new CR01 -> old F01     ', stage3_new_old_new_old),
    ('4  new -> new   Batch -> Batch                      ', stage4_new_batch_to_batch),
    ('5  new -> new   Continuous -> Continuous            ', stage5_new_continuous_to_continuous),
    ('6  new -> new   Continuous -> Semibatch             ', stage6_new_continuous_to_semibatch),
    ('7  new -> new   Continuous -> Batch (must refuse)   ', stage7_new_continuous_to_batch_refused),
    ('8  continuous -> semibatch, all pairings            ', stage8_continuous_to_semibatch_matrix),
)


def main(show_traceback=True):
    results = {}
    for label, fn in STAGES:
        try:
            detail = fn()
            results[label] = ('PASS', detail)
        except Exception as exc:
            detail = '%s: %s' % (type(exc).__name__, exc)
            results[label] = ('FAIL', detail)
            if show_traceback:
                print('\n===== stage %s traceback =====' % label.strip())
                traceback.print_exc()

    print('\n' + '=' * 78)
    print('MIXED FLOWSHEET RESULTS')
    print('=' * 78)
    for label, (status, detail) in results.items():
        print('  [%s] %s %s' % (status, label, detail))
    return results


if __name__ == '__main__':
    main()
