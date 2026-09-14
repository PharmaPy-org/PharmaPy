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
    SemiBatchCrystallizer as NewSemiBatchCryst)
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

    return 'filtration time = %.1f s' % flst.F01.timeProf[-1]



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

    return 'R02 final mass %.6f kg' % flst.R02.Phases.mass


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


STAGES = (
    ('0  old Filter alone                                 ', stage0_filter_alone),
    ('1  all-old   R01 -> CR01 -> F01                     ', stage1_all_old),
    ('2  new R01 -> new CR01 -> old F01                   ', stage2_new_new_old),
    ('3  new R01 -> old HOLD01 -> new CR01 -> old F01     ', stage3_new_old_new_old),
    ('4  new -> new   Batch -> Batch                      ', stage4_new_batch_to_batch),
    ('5  new -> new   Continuous -> Continuous            ', stage5_new_continuous_to_continuous),
    ('6  new -> new   Continuous -> Semibatch             ', stage6_new_continuous_to_semibatch),
    ('7  new -> new   Continuous -> Batch (must refuse)   ', stage7_new_continuous_to_batch_refused),
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
