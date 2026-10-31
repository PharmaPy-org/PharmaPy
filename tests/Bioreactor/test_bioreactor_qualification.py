"""Independent population, network, constraint, operation and coupled-model checks."""


from functools import partial
from operator import itemgetter
import cProfile
import json
from .input_helpers import read_input_json
import os
import sys
import time
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.culture import CultureSnapshot
from PharmaPy.Metabolic.closures.base import MetabolicInfeasibleError, MetabolicNumericalError
from .reconciliation_reference import (
    compare, CASES as COUPLED_CASES, Reference, load as coupled_load,
    network_matrix, quadratic_program,
)


# Population qualification

POPULATION_BASE = Path(__file__).parents[2] / 'tests/Bioreactor/fixtures/reconciliation/population_dynamics'
# Characteristic synthetic-fixture magnitudes normalize errors without changing
# any state: [mmol, 10^6 cells, 10^6 cells, g, L, 10^6 cell day/mL].
POPULATION_SCALES = np.array([10., 1., .2, .1, 1., .001])


# These are existing qualification limits, not model parameters or solver settings.
OPERATION_TIMEOUT_S = 60.  # [s], campaign watchdog for small synthetic systems
OPERATION_FEASIBILITY_TOLERANCE = 1e-8  # [-], strict returned-flux qualification bound




# Network qualification

NETWORK_BASE = Path(__file__).parents[2] / 'tests/Bioreactor/fixtures/reconciliation/network_structure'
NETWORK_FIELDS = ['nutrient_a_mmol', 'nutrient_b_mmol', 'byproduct_mmol', 'product_g',
          'viable_cells_million', 'dead_cells_million', 'volume_l', 'ivcd_million_cell_day_per_ml']
# Characteristic synthetic-fixture magnitudes, in NETWORK_FIELDS order:
# [mmol, mmol, mmol, g, 10^6 cells, 10^6 cells, L, 10^6 cell day/mL].
NETWORK_SCALES = np.array([10., 10., 2., .1, 1., .1, 1., .001])
NETWORK_VARIANTS = ['compatible', 'conflicting', 'switching', 'growth', 'nonunique']


# Constraint qualification

CONSTRAINT_BASE = Path(__file__).parents[2]/'tests/Bioreactor/fixtures/reconciliation/optimization_constraints'
CONSTRAINT_CASES = ['relative', 'relaxed', 'untargeted', 'normalization', 'near_zero',
         'redundant', 'equivalent_units']
CONSTRAINT_IDS = ['ua', 'ub', 'p', 'q', 'g']
# [-], exact coordinate map implied by the synthetic network balances.
CONSTRAINT_MAPPING = np.array([[1., 1.], [0., 1.], [1., 0.], [0., 2.], [0., 0.]])


# Operation qualification

OPERATION_BASE = Path(__file__).parents[2]/'tests/Bioreactor/fixtures/reconciliation/operating_transitions'
OPERATION_CASES = sorted(p.name for p in (OPERATION_BASE/'inputs').iterdir() if p.is_dir())
# Operating-transition fixture magnitudes, in NETWORK_FIELDS order:
# [mmol, mmol, mmol, g, 10^6 cells, 10^6 cells, L, 10^6 cell day/mL].
OPERATION_SCALES = np.array([1.,1.,1.,.1,1.,1.,1.,.0001])
# Real fixture construction and independent qualification helpers.
def _check_operation_budget(elapsed):
    """Reject a campaign that has exhausted the existing wall-time budget.

    Parameters
    ----------
    elapsed : float
        Monotonic elapsed time [s].

    Raises
    ------
    TimeoutError
        Elapsed time exceeds the campaign budget [s].
    """
    if elapsed > OPERATION_TIMEOUT_S:
        raise TimeoutError('Operating-transition small-system run exceeded the 60-second campaign guard')


class _ClosureObserver:
    """Observe real solver returns without altering the solver or its inputs.

    Python profiling is needed because the last accepted solution omits earlier
    optimizer returns, including trial integration stages. All are qualified.
    The full campaign tests exercise the real PharmaPy/HiGHS/SLSQP handoff.
    """

    def __init__(self, closure, start, report):
        """Configure the observer for one real closure and its output report.

        Parameters
        ----------
        closure : RateReconciledMFAClosure
            Actual closure instance used by the native campaign.
        start : float
            Monotonic start time [s].
        report : dict
            Mutable counts [-], scaled feasibility [-], and elapsed time [s].
        """
        self.closure = closure
        self.code = closure.solve.__func__.__code__
        self.start = start  # [s]
        self.report = report

    def __call__(self, frame, event, result):
        """Check elapsed time and qualify each actual closure return.

        Parameters
        ----------
        frame : frame
            Frame supplied by Python's standard profiling API.
        event : str
            Profiling event name.
        result : object
            Actual returned value for a return event; None on an exception.

        Raises
        ------
        TimeoutError
            The campaign exceeds its wall-time budget [s].
        AssertionError
            An actual flux result exceeds the scaled feasibility limit [-].
        """
        _check_operation_budget(time.perf_counter() - self.start)
        if event != 'return' or frame.f_code is not self.code or result is None:
            return
        if frame.f_locals.get('self') is not self.closure:
            return
        self.report['closure_calls'] += 1
        scale = max(1., max(abs(result.fluxes)))  # [model flux unit], existing normalization
        error = max(result.mass_balance_residual_inf, result.bound_violation_inf) / scale  # [-]
        availability = frame.f_locals.get('availability')
        if availability is not None:
            error = max(error, -np.min(availability.residual(result.fluxes)))  # [-]
        self.report['max_scaled_feasibility'] = max(self.report['max_scaled_feasibility'], float(error))
        assert error <= OPERATION_FEASIBILITY_TOLERANCE


def _nonlinear_exposure(s):
    """Evaluate exposure for the declared small nonlinear constraint case.

    Parameters
    ----------
    s : float
        Sum of reduced uptake coordinates [mmol/(10^6 cell day)] in the synthetic nonlinear case.

    Returns
    -------
    float
        Trapezoidal viable-population exposure [10^6 cell day].
    """
    return 0.05 * (1.0 + np.exp((0.1 * s - 0.05) * 0.1))


def _nonlinear_capacity_residual(s, *, exposure):
    """Locate the declared small-case inventory boundary.

    Parameters
    ----------
    s : float
        Sum of reduced uptake coordinates [mmol/(10^6 cell day)] in the synthetic nonlinear case.
    exposure : callable
        Independent growth-to-exposure law [10^6 cell day].

    Returns
    -------
    float
        Inventory use minus its explicit 0.12 mmol capacity [mmol].
    """
    return s * exposure(s) - 0.12


def _projection_error(z, *, mapping, target):
    """Evaluate an independent coordinate-projection residual.

    Parameters
    ----------
    z : ndarray
        Reduced flux coordinates in the synthetic fixture's declared model flux basis.
    mapping : ndarray
        Linear coordinate map [-], with one row per physical flux.
    target : ndarray
        Target vector in the corresponding model coordinate units.

    Returns
    -------
    float
        Squared mismatch in the supplied target's squared units.
    """
    return np.sum((mapping @ z - target) ** 2)


def _operation_objective(z, *, hessian, linear):
    """Evaluate the independent reduced operating-transition quadratic.

    Parameters
    ----------
    z : ndarray
        Reduced flux coordinates in the synthetic fixture's declared model flux basis.
    hessian : ndarray
        Quadratic objective coefficients [1/(coordinate_i coordinate_j)].
    linear : ndarray
        Linear objective coefficients [1/model coordinate unit].

    Returns
    -------
    float
        Dimensionless quadratic objective [-], without its constant term.
    """
    return z @ hessian @ z - 2 * linear @ z


def _candidate_objective(s, *, candidate):
    """Read the independently evaluated candidate objective.

    Parameters
    ----------
    s : float
        Sum of reduced uptake coordinates [mmol/(10^6 cell day)] in the synthetic nonlinear case.
    candidate : callable
        Independent candidate evaluator returning a flux and dimensionless objective.

    Returns
    -------
    float
        Dimensionless objective [-], including genuine infeasible infinity.
    """
    return candidate(s)[1]


def _pivot_magnitude(k, *, a, i):
    """Rank decimal-elimination rows by their current pivot magnitude.

    Parameters
    ----------
    k : int
        Candidate pivot row index.
    a : list of list of Decimal
        Augmented linear system; coefficients retain each equation's row and coordinate units.
    i : int
        Selected species or pivot-column index.

    Returns
    -------
    Decimal
        Absolute pivot coefficient in its original matrix-entry unit.
    """
    return abs(a[k][i])


def _population_rhs(t, state, *, variant, flow, death):
    """Evaluate the independent six-state population balances.

    Parameters
    ----------
    t : float
        Integration time [day].
    state : ndarray
        Ordered state vector; component units and order are specified in Notes.
    variant : str
        Named synthetic qualification case or coordinate conversion.
    flow : float
        Inlet volume flow [L/day].
    death : float
        Specific death coefficient [1/day].

    Returns
    -------
    list of float
        Derivatives in each state component unit per day.

    Notes
    -----
    Population states contain nutrient [mmol], viable/dead populations
    [10^6 cells], product [g], volume [L], and integrated viable concentration
    [10^6 cell day/mL]. Operating-transition states use NETWORK_FIELDS.
    The independent time basis is day.
    """
    amount, viable, dead, product, volume, exposure = state
    mu = (.44*(amount/volume)/(1.+amount/volume) if variant == 'monod'
          else .25 if variant == 'constrained_growth' else .4)
    return [2.*flow-2.4*mu*viable, (mu-death)*viable, death*viable,
            .04*mu*viable, flow, viable/(1000.*volume)]


def _network_rhs(t, y, *, ua, ub, q, p, g):
    """Evaluate the independent eight-state network balances.

    Parameters
    ----------
    t : float
        Integration time [day].
    y : ndarray
        Ordered state vector; component units and order are specified in Notes.
    ua : float
        First uptake rate [mmol/(10^6 cell day)].
    ub : float
        Second uptake rate [mmol/(10^6 cell day)].
    q : float
        Byproduct formation rate [mmol/(10^6 cell day)].
    p : float
        Product formation [mmol/(10^6 cell day)]; fixture MW=100 g/mol converts it to g.
    g : float
        Growth-coordinate value in the input-declared model basis; conversion to [1/day] follows the fixture.

    Returns
    -------
    list of float
        Derivatives in NETWORK_FIELDS component units per day.

    Notes
    -----
    States follow NETWORK_FIELDS: three amounts [mmol], product [g],
    viable/dead populations [10^6 cells], volume [L], and integrated viable
    concentration [10^6 cell day/mL]. The time basis is day.
    """
    return [-ua*y[4], -ub*y[4], q*y[4], .1*p*y[4],
            (g/2.-.05)*y[4], .05*y[4], .1, y[4]/(1000.*y[6])]


def _constraint_objective(z, *, rows, values, total):
    """Evaluate the independently normalized target and internal-flux residuals.

    Parameters
    ----------
    z : ndarray
        Reduced flux coordinates in the synthetic fixture's declared model flux basis.
    rows : ndarray
        Normalized residual coefficients [1/model coordinate unit], or saved observation rows with their declared axes.
    values : array-like
        Reference target values or refinement records; normalized errors are dimensionless.
    total : float
        Internal-flux normalization in the same model coordinate unit as the fluxes.

    Returns
    -------
    float
        Dimensionless sum of squared residuals [-].
    """
    return float(np.sum((rows@z-values)**2)+np.sum(z**2)/total**2)


def operation_optimum(config, limits=(np.inf,np.inf)):
    """Enumerate the unconstrained minimum, edges and vertices of a 2-D QP.

    Parameters
    ----------
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.
    limits : ndarray or None
        Optional uptake-rate bounds [mmol/(10^6 cell day)], in nutrient order.

    Returns
    -------
    ndarray
        Five fluxes in CONSTRAINT_IDS order and the declared model flux basis.
    """
    growth = config['parameters']['rates']['g'] > 0.
    b = .2 if growth else 0.
    mapping = np.array([[1.,1.],[0.,1.],[1.-b,0.],[0.,2.-b],[b,b]])
    target = np.array([config['parameters']['rates'][n] for n in ('ua','ub','p','q','g')])
    relative = config['model']['reconciliation']['reconciliation_policy'] == 'relative-regularized'
    weights = 1./np.maximum(abs(target),1e-6) if relative else np.ones(5)
    rows = weights[:,None]*mapping
    hessian = rows.T@rows + (np.eye(2)/max(abs(sum(target)),1e-6)**2 if relative else 0.)
    linear = rows.T@(weights*target)
    constraints = np.vstack((-np.eye(2),mapping,[1.,1.],[0.,1.]))
    bounds = np.r_[0.,0.,np.full(5,10.),limits]
    if relative:
        constraints = np.vstack((constraints,mapping,-mapping))
        bounds = np.r_[bounds,1.5*target,-.5*target]
    keep = np.isfinite(bounds) & (np.linalg.norm(constraints,axis=1)>0.)
    constraints,bounds = constraints[keep],bounds[keep]
    candidates = [np.linalg.solve(hessian,linear)]
    for row,bound in zip(constraints,bounds):
        system = np.block([[hessian,row[:,None]],[row[None,:],np.zeros((1,1))]])
        candidates.append(np.linalg.solve(system,np.r_[linear,bound])[:2])
    for i,j in combinations(range(len(bounds)),2):
        if abs(np.linalg.det(constraints[[i,j]])) > 1e-12:
            candidates.append(np.linalg.solve(constraints[[i,j]],bounds[[i,j]]))
    candidates = [z for z in candidates if np.max(constraints@z-bounds) <= 1e-11]
    if not candidates:
        raise ValueError('independent policy infeasible')
    z = min(candidates,key=partial(_operation_objective, hessian=hessian, linear=linear))
    return mapping@z


def _operation_rhs(t, y, *, supply, config, death, flow):
    """Evaluate independent balances with constrained uptake and real supplies.

    Parameters
    ----------
    t : float
        Integration time [day].
    y : ndarray
        Ordered state vector; component units and order are specified in Notes.
    supply : ndarray
        Ordered extracellular inlet supplies [mmol/day].
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.
    death : float
        Specific death coefficient [1/day].
    flow : float
        Inlet volume flow [L/day].

    Returns
    -------
    list of float
        Derivatives in NETWORK_FIELDS component units per day.

    Notes
    -----
    States follow NETWORK_FIELDS: three amounts [mmol], product [g],
    viable/dead populations [10^6 cells], volume [L], and integrated viable
    concentration [10^6 cell day/mL]. The time basis is day.
    """
    limits = np.where(y[:2] <= 1e-12,supply/y[4],np.inf)
    ua,ub,p,q,g = operation_optimum(config,limits)
    net = supply-y[4]*np.array([ua,ub])
    net[abs(net)<1e-13] = 0.
    return [*net,q*y[4],.1*p*y[4],
            (g/2.-death)*y[4],death*y[4],flow,y[4]/(1000.*y[6])]


def _operation_exposure_balance(average, *, config, capacity, state, death, remaining):
    """Match average viable population to independently constrained growth.

    Parameters
    ----------
    average : float
        Average viable population over the interval [10^6 cells].
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.
    capacity : ndarray
        Available extracellular inventory rates [mmol/day].
    state : ndarray
        Ordered state vector; component units and order are specified in Notes.
    death : float
        Specific death coefficient [1/day].
    remaining : ndarray or float
        Remaining inventories [mmol] in the coupled oracle; remaining duration [day] in the operation balance.

    Returns
    -------
    float
        Average-population balance residual [10^6 cells].

    Notes
    -----
    Population states contain nutrient [mmol], viable/dead populations
    [10^6 cells], product [g], volume [L], and integrated viable concentration
    [10^6 cell day/mL]. Operating-transition states use NETWORK_FIELDS.
    The independent time basis is day.
    """
    growth_flux = operation_optimum(config,capacity/average)[4]
    return average-.5*state[4]*(1.+np.exp((growth_flux/2.-death)*remaining))


def _rename_network_inputs(value, *, rename):
    """Rename identities and reverse field order without changing values.

    Parameters
    ----------
    value : object
        JSON-compatible content; numeric physical values are preserved.
    rename : dict of str
        Original-to-new identifier mapping; numeric physical values are unchanged.

    Returns
    -------
    object
        Renamed, reordered input content with all numeric values preserved.
    """
    if isinstance(value, dict):
        return {rename.get(k,k):_rename_network_inputs(v, rename=rename) for k,v in reversed(list(value.items()))}
    if isinstance(value, list):
        return [_rename_network_inputs(v, rename=rename) for v in value]
    return rename.get(value, value) if isinstance(value, str) else value


def _decimal_constraint_solve(matrix, vector, *, precision):
    """Solve the independent small linear system in decimal arithmetic.

    Parameters
    ----------
    matrix : ndarray
        Declared stoichiometric/constraint coefficients; row units divided by their corresponding coordinate units.
    vector : ndarray
        Linear-system right-hand side, in each equation's row unit.
    precision : int
        Decimal significant-digit count [-] for independent elimination.

    Returns
    -------
    ndarray
        Solution in the supplied matrix's coordinate units.
    """
    with localcontext() as ctx:
        ctx.prec = precision
        a = [[Decimal(str(v)) for v in row]+[Decimal(str(b))] for row,b in zip(matrix,vector)]
        n = len(a)
        for i in range(n):
            pivot = max(range(i,n),key=partial(_pivot_magnitude, a=a, i=i))
            a[i],a[pivot] = a[pivot],a[i]
            value = a[i][i]
            a[i] = [v/value for v in a[i]]
            for k in range(n):
                if k != i:
                    value = a[k][i]
                    a[k] = [v-value*w for v,w in zip(a[k],a[i])]
        return np.array([float(row[-1]) for row in a])


def _nonlinear_candidate(s, *, exposure, direction, linear, h, rows, target):
    """Construct and qualify an independent small-case nonlinear candidate.

    Parameters
    ----------
    s : float
        Sum of reduced uptake coordinates [mmol/(10^6 cell day)] in the synthetic nonlinear case.
    exposure : callable
        Independent growth-to-exposure law [10^6 cell day].
    direction : ndarray
        Declared reduced-coordinate direction [-].
    linear : ndarray
        Linear objective coefficients [1/model coordinate unit].
    h : ndarray
        Reduced Hessian [1/(coordinate_i coordinate_j)].
    rows : ndarray
        Normalized residual coefficients [1/model coordinate unit], or saved observation rows with their declared axes.
    target : ndarray
        Target vector in the corresponding model coordinate units.

    Returns
    -------
    tuple
        Reduced flux coordinates in model units and objective [-]; infeasibility retains NaN/infinity.
    """
    low = max(.25,s-1.5)
    high = min(.75,s-.5,.04/exposure(s))
    if low > high+1e-12:return np.array([np.nan,np.nan]),np.inf
    origin = np.array([s,0.])
    y = np.clip(direction@(linear-h@origin)/(direction@h@direction),low,high)
    z = np.array([s-y,y])
    objective = np.sum((rows@z-1.)**2)+np.sum(z*z)/target.sum()**2
    return z,float(objective)


def _rename_operation_inputs(value, *, rename):
    """Rename operation identities and reverse field order.

    Parameters
    ----------
    value : object
        JSON-compatible content; numeric physical values are preserved.
    rename : dict of str
        Original-to-new identifier mapping; numeric physical values are unchanged.

    Returns
    -------
    object
        Renamed, reordered input content with all numeric values preserved.
    """
    if isinstance(value,dict):
        return {rename.get(k,k):_rename_operation_inputs(v, rename=rename) for k,v in reversed(list(value.items()))}
    if isinstance(value,list):
        return [_rename_operation_inputs(v, rename=rename) for v in value]
    return rename.get(value,value) if isinstance(value,str) else value


def population_load(variant):
    """Read one population qualification input set.

    Parameters
    ----------
    variant : str
        Named synthetic qualification case or coordinate conversion.

    Returns
    -------
    tuple
        Case, biological mappings with explicit units, and thermophysical JSON path.
    """
    p = POPULATION_BASE/'inputs'/variant
    return read_input_json(p/'case.json'), read_input_json(p/'mechanism.json'), p/'thermo.json'


def population_observed(a, h):
    """Read native histories into the population reference order.

    Parameters
    ----------
    a : BioreactorAssembly
        Real assembly defining species identities and native phase order.
    h : DynamicResult
        Native saved histories with masses [kg], time [s], and declared biological auxiliary states.

    Returns
    -------
    ndarray
        Columns: mmol, 10^6 viable cells, 10^6 dead cells, g product, L, and 10^6 cell day/mL.
    """
    j = list(a.phase.name_species).index('nutrient')
    return np.column_stack((h.mass_j_liquid0[:, j]*10000., h.viable_cells_million_liquid0,
                            h.dead_cells_million_liquid0, h.product_g_liquid0,
                            np.asarray(h.vessel_vol).reshape(-1)*1000.,
                            h.ivcd_million_cell_day_per_ml_liquid0))


def population_reference(histories, variant):
    """Independent ODE; constant-rate cases additionally checked analytically below.

    Parameters
    ----------
    histories : sequence of DynamicResult
        Native segment histories; time [s], masses [kg], and declared biological auxiliary states.
    variant : str
        Named synthetic qualification case or coordinate conversion.

    Returns
    -------
    list of ndarray
        Six-state reference samples for each segment, in population_observed component order and units.
    """
    y = np.array([10., 1., 0., 0., 1., 0.])
    answer = []
    for h in histories:
        start, end = h.time[[0, -1]]/86400.
        flow = .1 if start < .5 else .2 if start < 1. else 0.
        if np.isclose(start, .75):
            y[:4] *= 1.-.1/y[4]
            y[4] -= .1
        if np.isclose(start, 1.):
            y[0] += .3
            y[4] += .1
        death = 0. if variant == 'growth' else .1

        rhs = partial(_population_rhs, variant=variant, flow=flow, death=death)

        solution = solve_ivp(rhs, (start, end), y, t_eval=h.time/86400.,
                             method='DOP853', rtol=2e-13, atol=2e-15, max_step=.0025)
        assert solution.success
        rows = solution.y.T
        if variant != 'monod':
            mu = .25 if variant == 'constrained_growth' else .4
            dt = h.time/86400.-start
            total = y[1]*np.expm1((mu-death)*dt)/(mu-death)
            analytic = np.column_stack((y[0]+2.*flow*dt-2.4*mu*total,
                                        y[1]*np.exp((mu-death)*dt), y[2]+death*total,
                                        y[3]+.04*mu*total, y[4]+flow*dt))
            np.testing.assert_allclose(rows[:, :5], analytic, atol=2e-12, rtol=2e-12)
        answer.append(rows)
        y = rows[-1].copy()
    return answer


def network_load(variant):
    """Read one network-structure qualification input set.

    Parameters
    ----------
    variant : str
        Named synthetic qualification case or coordinate conversion.

    Returns
    -------
    tuple
        Case, biological mappings with explicit units, and thermophysical JSON path.
    """
    folder = NETWORK_BASE/'inputs'/variant
    return (read_input_json(folder/'case.json'),
            read_input_json(folder/'mechanism.json'), folder/'thermo.json')


def network_projection(variant, limits=(np.inf, np.inf)):
    """Enumerate all possible active sets in an independently reduced 2-D QP.

    Parameters
    ----------
    variant : str
        Named synthetic qualification case or coordinate conversion.
    limits : ndarray or None
        Optional uptake-rate bounds [mmol/(10^6 cell day)], in nutrient order.

    Returns
    -------
    tuple
        Reduced coordinates, full model fluxes, and independently minimized squared projection error.
    """
    b = .2 if variant == 'growth' else 0.
    mapping = np.array([[1., 1.], [0., 1.], [1.-b, 0.], [0., 2.-b], [b, b]])
    target = np.array([4., 1., 1., 4., 0.] if variant == 'conflicting'
                      else [1.5, .5, 1.-b, (2.-b)*.5, 1.5*b])
    constraints = np.vstack((-np.eye(2), mapping, [1., 1.], [0., 1.]))
    bounds = np.array([0., 0., 10., 10., 10., 10., 10., *limits])
    finite = np.isfinite(bounds)
    constraints, bounds = constraints[finite], bounds[finite]
    hessian, linear = mapping.T@mapping, mapping.T@target
    candidates = []
    for size in range(3):
        for subset in combinations(range(len(bounds)), size):
            rows = constraints[list(subset)]
            system = np.block([[hessian, rows.T], [rows, np.zeros((size, size))]])
            if np.linalg.matrix_rank(system) < len(system):
                continue
            solution = np.linalg.solve(system, np.r_[linear, bounds[list(subset)]])
            z = solution[:2]
            if np.max(constraints@z-bounds) <= 1e-9 and np.all(solution[2:] >= -1e-9):
                candidates.append(z)
    assert candidates
    z = min(candidates, key=partial(_projection_error, mapping=mapping, target=target))
    return z, mapping@z, float(np.sum((mapping@z-target)**2))


def network_observed(assembly, history, names=('nutrient_a', 'nutrient_b', 'byproduct')):
    """Read native histories into the network reference order.

    Parameters
    ----------
    assembly : BioreactorAssembly
        Actual input-built biological model and native reactor.
    history : DynamicResult
        Native saved histories with masses [kg], time [s], and declared biological auxiliary states.
    names : dict of str or sequence of str
        Consistent renaming map, or ordered requested species identifiers for observation extraction.

    Returns
    -------
    ndarray
        Columns in NETWORK_FIELDS order: mmol, mmol, mmol, g, 10^6 cells, 10^6 cells, L, 10^6 cell day/mL.
    """
    indexes = [list(assembly.phase.name_species).index(n) for n in names]
    return np.column_stack((history.mass_j_liquid0[:, indexes]*10000.,
                            history.product_g_liquid0, history.viable_cells_million_liquid0,
                            history.dead_cells_million_liquid0,
                            np.asarray(history.vessel_vol).reshape(-1)*1000.,
                            history.ivcd_million_cell_day_per_ml_liquid0))


def network_reference(histories, variant, continuous=False):
    """Exact finite-step balances for zero growth; independent ODE for growth.

    Parameters
    ----------
    histories : sequence of DynamicResult
        Native segment histories; time [s], masses [kg], and declared biological auxiliary states.
    variant : str
        Named synthetic qualification case or coordinate conversion.
    continuous : bool
        Use the independent continuous ODE; False uses the original finite-step equations.

    Returns
    -------
    list of ndarray
        Reference samples per segment in NETWORK_FIELDS order and its documented physical component units.
    """
    state = np.array([.6, .1, 0., 0., 1., 0., 1., 0.] if variant == 'switching'
                     else [10., 10., 0., 0., 1., 0., 1., 0.])
    answer = []
    for history in histories:
        times = history.time/86400.
        if np.isclose(times[0], .5):
            state[:2] += [.6, .2]
            state[6] += .1
        if np.isclose(times[0], .75):
            state[:6] *= 1.-.1/state[6]
            state[6] -= .1
        if variant == 'growth':
            _, fluxes, _ = network_projection(variant)
            ua, ub, p, q, g = fluxes
            rhs = partial(_network_rhs, ua=ua, ub=ub, q=q, p=p, g=g)
            result = solve_ivp(rhs, (times[0], times[-1]), state, t_eval=times,
                               method='DOP853', rtol=2e-13, atol=2e-15, max_step=.0025)
            assert result.success
            rows = result.y.T
        else:
            rows = [state.copy()]
            for dt in np.diff(times):
                remaining = dt
                while remaining > 1e-13:
                    limits = (np.where(state[:2] > 1e-10, np.inf, 0.) if continuous
                              else np.maximum(state[:2], 0.)/(state[4]*remaining))
                    _, fluxes, _ = network_projection(variant, limits)
                    ua, ub, p, q, _ = fluxes
                    consumption = state[4]*np.array([ua, ub])
                    duration = remaining
                    if continuous:
                        positive = consumption > 1e-10
                        duration = min(duration, np.min(state[:2][positive]/consumption[positive], initial=np.inf))
                    state[:4] += duration*state[4]*np.array([-ua, -ub, q, .1*p])
                    state[7] += (state[4]/100.*np.log1p(.1*duration/state[6]) if continuous
                                 else duration*state[4]/(1000.*state[6]))
                    state[6] += .1*duration
                    remaining -= duration
                rows.append(state.copy())
            rows = np.asarray(rows)
        answer.append(rows)
        state = rows[-1].copy()
    return answer


def constraint_load(name):
    """Read one optimizer-constraint qualification input set.

    Parameters
    ----------
    name : str
        Fixture name or requested property identifier.

    Returns
    -------
    tuple
        Case, biological mappings with explicit units, and thermophysical JSON path.
    """
    folder = CONSTRAINT_BASE/'inputs'/name
    return (read_input_json(folder/'case.json'),
            read_input_json(folder/'mechanism.json'), folder/'thermo.json')


def constraint_reference(config, solve=np.linalg.solve):
    """Enumerate unconstrained, edge and vertex minima, independently of SLSQP.

    Parameters
    ----------
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.
    solve : callable or None
        Optional independent linear-system solver; None retains the NumPy solver.

    Returns
    -------
    tuple
        Reduced model fluxes, dimensionless objective [-], and accepted relative target-bound envelope [-].
    """
    policy = config['model']['reconciliation']
    targets = np.array([config['parameters']['rates'][n] for n in CONSTRAINT_IDS])
    selected = np.array([n not in policy['untargeted_exchanges'] for n in CONSTRAINT_IDS])
    # All campaign fixtures declare the same physical mmol floor, independently
    # converted here without calling the production unit converter.
    amount_unit = config['model']['kinetics']['flux_basis']['exchange']['amount_unit']
    floor = 1e-6 * {'mmol': 1., 'umol': 1000.}[amount_unit]
    scales = np.maximum(abs(targets[selected]), floor)
    total = max(abs(sum(t for n,t,keep in zip(CONSTRAINT_IDS,targets,selected)
                        if keep and n not in policy['normalization_exclusions'])), floor)
    rows, values = CONSTRAINT_MAPPING[selected]/scales[:,None], targets[selected]/scales
    hessian, linear = rows.T@rows+np.eye(2)/total**2, rows.T@values
    coordinate = 1./np.sqrt(np.diag(hessian))
    hessian = coordinate[:,None]*hessian*coordinate
    linear = coordinate*linear
    reaction_rows = dict(zip(CONSTRAINT_IDS, CONSTRAINT_MAPPING))
    reaction_rows.update(x=np.array([1.,0.]), y=np.array([0.,1.]))
    # These fixtures declare envelopes 0.5 or 1.0 with increment 0.5.
    envelopes = [.5] if policy['reconciliation_relative_bound'] == .5 else [.5,1.]
    for envelope in envelopes:
        constraints, bounds = [], []
        for reaction in config['network']['reactions']:
            name = reaction['identifier']
            low, high = reaction['lower'], reaction['upper']
            if name in CONSTRAINT_IDS and selected[CONSTRAINT_IDS.index(name)]:
                t = targets[CONSTRAINT_IDS.index(name)]
                low, high = max(low,(1.-envelope)*t), min(high,(1.+envelope)*t)
            constraints.extend([reaction_rows[name], -reaction_rows[name]])
            bounds.extend([high, -low])
        a, b = np.array(constraints)*coordinate, np.array(bounds)
        norm = np.linalg.norm(a,axis=1)
        if np.any(b[norm == 0.] < 0.):
            continue
        keep = norm > 0.
        a, b = a[keep]/norm[keep,None], b[keep]/norm[keep]
        candidates = [solve(hessian,linear)]
        for row,bound in zip(a,b):
            system = np.block([[hessian,row[:,None]],[row[None,:],np.zeros((1,1))]])
            candidates.append(solve(system,np.r_[linear,bound])[:2])
        for i,j in combinations(range(len(b)),2):
            matrix = a[[i,j]]
            if abs(np.linalg.det(matrix)) > 1e-12:
                candidates.append(solve(matrix,b[[i,j]]))
        feasible = [z*coordinate for z in candidates if np.max(a@z-b) <= 1e-9]
        if feasible:
            objective = partial(_constraint_objective, rows=rows, values=values, total=total)
            z = min(feasible,key=objective)
            return z, objective(z), envelope
    raise ValueError('Independent reference is infeasible')


def constraint_decimal_solver(precision):
    """Bind the precision of the independent decimal linear solver.

    Parameters
    ----------
    precision : int
        Decimal significant-digit count [-] for independent elimination.

    Returns
    -------
    callable
        Small linear-system solver, retaining matrix/rhs units.
    """
    solve = partial(_decimal_constraint_solve, precision=precision)
    return solve


def constraint_floor_case(factor):
    """Build the target-floor qualification at an explicit coordinate scale.

    Parameters
    ----------
    factor : float
        Coordinate or target-floor scaling factor [-].

    Returns
    -------
    tuple
        Case, model, and thermophysical path, retaining explicit scaled flux units.
    """
    case, config, thermo = constraint_load('relative')
    config['parameters']['rates'].update(ua=1.5e-7,ub=5e-8,p=1e-7,q=2e-7,g=0.)
    if factor != 1.:
        for name in CONSTRAINT_IDS:
            config['parameters']['rates'][name] *= factor
        for reaction in config['network']['reactions']:
            reaction['lower'] *= factor
            reaction['upper'] *= factor
        for name,basis in config['model']['kinetics']['flux_basis'].items():
            basis['amount_unit'] = 'umol'
            if name == 'growth':basis['amount_per_million_cells'] *= factor
        for name in CONSTRAINT_IDS:config['rate_provider']['output_units'][name]='umol/(10^6 cell day)'
    return case,config,thermo


def constraint_nonlinear_reference(points=257):
    """Reduce the nonlinear QP to total uptake s=x+y and solve each y exactly.

    Parameters
    ----------
    points : int
        Number of scalar search-grid points [-]; refinement is a numerical setting.

    Returns
    -------
    tuple
        Reduced model fluxes, objective [-], and viable-population exposure [10^6 cell day].
    """
    mapping = np.array([[1.,1.],[0.,1.],[.8,0.],[0.,1.8],[.2,.2]])
    target = np.array([1.5,.5,.8,.9,.3])
    rows = mapping/target[:,None]
    h = rows.T@rows+np.eye(2)/target.sum()**2
    linear = rows.T@np.ones(5)
    exposure = _nonlinear_exposure
    top = brentq(partial(_nonlinear_capacity_residual, exposure=exposure),.75,2.25,xtol=1e-14)
    direction = np.array([-1.,1.])
    candidate = partial(_nonlinear_candidate, exposure=exposure, direction=direction, linear=linear, h=h, rows=rows, target=target)
    grid = np.linspace(.75,top,points)
    choices = [candidate(s) for s in grid]
    for i in range(1,len(grid)-1):
        if choices[i][1] <= min(choices[i-1][1],choices[i+1][1]):
            opt = minimize_scalar(partial(_candidate_objective, candidate=candidate),bounds=(grid[i-1],grid[i+1]),
                                  method='bounded',options={'xatol':1e-14})
            choices.append(candidate(opt.x))
    z,obj = min(choices,key=itemgetter(1))
    return z,obj,exposure(z.sum())


def constraint_small_closure(matrix, lower, upper, internal=()):
    """Construct the real small closure from supplied balances and bounds.

    Parameters
    ----------
    matrix : ndarray
        Declared stoichiometric/constraint coefficients; row units divided by their corresponding coordinate units.
    lower : ndarray
        Lower reaction bounds in the declared model flux unit.
    upper : ndarray
        Upper reaction bounds in the declared model flux unit.
    internal : sequence of str
        Reaction identifiers whose fluxes contribute to the regularization term.

    Returns
    -------
    RateReconciledMFAClosure
        Actual production closure on the supplied model flux basis.
    """
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.reconciled import RateReconciledMFAClosure
    matrix = np.asarray(matrix, dtype=float)
    metabolites = tuple(f'm{i}' for i in range(len(matrix)))
    reactions = tuple(ReactionDefinition(
        f'r{j}', dict(zip(metabolites, matrix[:, j])), lo, hi)
        for j, (lo, hi) in enumerate(zip(lower, upper)))
    network = MetabolicNetworkDefinition('edge', '1', metabolites, reactions, {'r0': 1.})
    return RateReconciledMFAClosure(network, internal)


def constraint_problem(closure, value, availability=None, targets=None):
    """Bind real optimizer and qualification contracts for small synthetic QPs.

    Parameters
    ----------
    closure : RateReconciledMFAClosure
        Synthetic network with dimensionless flux coordinates.
    value : float
        Target for r0 [-], when explicit targets are absent.
    availability : InventoryAvailability or None
        Additional declared inventory constraints.
    targets : sequence or None
        Explicit residual targets with unit scales [-].

    Returns
    -------
    tuple
        Real optimization callback, candidate qualification callback, and
        quadratic terms. No solver or constraint is replaced by a test double.
    """
    from scipy.linalg import null_space
    from scipy.optimize import LinearConstraint, NonlinearConstraint
    from PharmaPy.Metabolic.closures import reconciled as r
    lower, upper = closure.network.lower_bounds, closure.network.upper_bounds  # [-]
    targets = targets if targets is not None else (r.ReconciliationTarget('r0', value, 1.),)
    terms = r._make_terms(closure.network, closure.internal_reaction_ids, targets, 1.)
    free = lower != upper
    offset = np.where(free, 0., lower)  # [-]
    scales = np.ones(len(lower))  # [-], fixtures use unit coordinate scales
    matrix = closure.network.stoichiometric_matrix  # [-]
    equality, rhs = matrix[:, free], -matrix @ offset  # [-]
    low, high = lower[free], upper[free]  # [-]
    tangent = null_space(equality)  # [-]
    expand = partial(r._reconciliation_expand, offset=offset, free=free, variable_scale=scales)
    feasible = partial(r._reconciliation_feasible_flux, free=free, variable_scale=scales,
        closure=closure, lower=lower, upper=upper, availability=availability,
        equality=equality, rhs=rhs, low=low, high=high)
    certified = partial(r._reconciliation_certified, availability=availability, lower=lower,
        upper=upper, matrix=matrix, terms=terms, closure=closure)
    qualified = partial(r._reconciliation_qualified, feasible_flux=feasible, free=free,
        variable_scale=scales, terms=terms, low=low, high=high, closure=closure,
        availability=availability, tangent=tangent, equality=equality,
        narrow=np.zeros(len(lower), dtype=bool), certified=certified)
    constraints = [LinearConstraint(equality, rhs, rhs)]
    if availability is not None:
        constraints.append(NonlinearConstraint(availability.residual, 0., np.inf,
                                               jac=availability.jacobian))
    optimize = partial(r._reconciliation_optimize, constraints=constraints, low=low,
        high=high, tangent=tangent, reduced=equality, reduced_rhs=rhs, expand=expand,
        terms=terms, qualified=qualified, equality=equality, availability=availability,
        closure=closure, free=free, variable_scale=scales)
    return optimize, qualified, terms


def constraint_solve_small(closure, value, scale=1.):
    """Run the actual small closure with an explicit scaled target.

    Parameters
    ----------
    closure : RateReconciledMFAClosure
        Actual closure under qualification, with its declared flux basis.
    value : float
        Target flux in the closure's declared model flux unit.
    scale : float or ndarray
        Positive normalization in the corresponding flux or observation component units.

    Returns
    -------
    FluxSolution
        Actual reaction fluxes in closure reaction order and declared units.
    """
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.closures.reconciled import ReconciliationTarget
    return closure.solve(MetabolicEnvironment({}, {}),
                         targets=(ReconciliationTarget('r0', value, scale),), internal_scale=1.)


def operation_load(name, backend='fixed-step', step=.001):
    """Configure one real operating-transition qualification campaign.

    Parameters
    ----------
    name : str
        Fixture name or requested property identifier.
    backend : str
        Native integrator selector; scipy or fixed-step.
    step : float
        Fixed integration horizon [day].

    Returns
    -------
    tuple
        Case, biological mappings with explicit units, and thermophysical JSON path.
    """
    folder = OPERATION_BASE/'inputs'/name
    case = read_input_json(folder/'case.json')
    config = read_input_json(folder/'mechanism.json')
    case['numerics']['step']['value'] = step
    if backend == 'scipy':
        case['numerics'].update(backend='scipy', relative_tolerance=1e-10,
            absolute_tolerance=1e-15, maximum_step={'value':.002,'unit':'day'},
            jacobian_relative_step=1e-5, jacobian_state_scale=1e-6)
    return case, config, folder/'thermo.json'


def operation_initial(config):
    """Read the eight-state operating-transition initial condition.

    Parameters
    ----------
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.

    Returns
    -------
    ndarray
        Initial state in NETWORK_FIELDS order and component units.
    """
    state = config['state']
    return np.array([state['species']['amounts'][n] for n in ('nutrient_a','nutrient_b','byproduct')]
        +[state['product_g'],state['viable_cells_million'],state['dead_cells_million'],1.,state['integral']])


def operation_reference(case, config, histories, continuous=False, precision=1.):
    """Piecewise exact inventories/jumps; independent tight ODE only for growth.

    Parameters
    ----------
    case : dict
        Operating input with explicit time, volume, temperature, and recipe units.
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.
    histories : sequence of DynamicResult
        Native segment histories; time [s], masses [kg], and declared biological auxiliary states.
    continuous : bool
        Use the independent continuous ODE; False uses the original finite-step equations.
    precision : int
        Decimal significant-digit count [-] for independent elimination.

    Returns
    -------
    list of ndarray
        Independent segment samples in NETWORK_FIELDS order and documented physical component units.
    """
    state, inlets, answer = operation_initial(config), {}, []
    native = case['operation']['volume_policy'] == 'native'
    growth = config['parameters']['rates']['g'] > 0.
    for history in histories:
        times = history.time/86400.
        for event in config['recipes']['benchmark']:
            if abs(event['time']['value']-times[0]) > 1e-12:
                continue
            kind = event['event_type']
            if kind == 'flow':
                flow = event['volume_flow']['value']
                conc = event['concentrations']['values']
                inlets[event.get('inlet','inlet')] = (flow,flow*np.array([conc.get(n,0.) for n in ('nutrient_a','nutrient_b')]))
            elif kind == 'sample':
                volume = event['volume']['value']
                state[:6] *= 1.-volume/state[6]
                state[6] -= volume
            else:
                volume = event['volume']['value']
                state[:2] += volume*np.array([event['concentrations']['values'].get(n,0.) for n in ('nutrient_a','nutrient_b')])
                state[6] += volume
        flow = sum(v[0] for v in inlets.values())
        supply = sum((v[1] for v in inlets.values()),start=np.zeros(2))
        if growth and continuous:
            death = config['parameters']['rates']['death']
            rhs = partial(_operation_rhs, supply=supply, config=config, death=death, flow=flow)
            solution = solve_ivp(rhs,(times[0],times[-1]),state,t_eval=times,
                                 method='DOP853',rtol=3e-13,atol=3e-15*precision,max_step=.001*precision)
            assert solution.success
            rows = solution.y.T
        else:
            rows = [state.copy()]
            for dt in np.diff(times):
                remaining = dt
                while remaining > 1e-14:
                    limits = (np.where(state[:2] <= 1e-12,supply/state[4],np.inf) if continuous
                              else (np.maximum(state[:2],0.)/remaining+supply)/state[4])
                    ua,ub,p,q,g = operation_optimum(config,limits)
                    death = config['parameters']['rates']['death']
                    if growth:
                        capacity = np.maximum(state[:2],0.)/remaining+supply
                        exposure_balance = partial(_operation_exposure_balance, config=config, capacity=capacity, state=state, death=death, remaining=remaining)
                        average = brentq(exposure_balance,.5*state[4],
                            .5*state[4]*(1.+np.exp(5.*remaining)),xtol=1e-14)
                        ua,ub,p,q,g = operation_optimum(config,capacity/average)
                    end_viable = state[4]*np.exp((g/2.-death)*remaining) if growth else state[4]
                    exposure = .5*(state[4]+end_viable)
                    derivative = np.r_[supply-exposure*np.array([ua,ub]),q*exposure,.1*p*exposure]
                    duration = remaining
                    if continuous:
                        consuming = (derivative[:2] < -1e-12) & (state[:2] > 1e-12)
                        duration = min(duration,np.min(state[:2][consuming]/-derivative[:2][consuming],initial=np.inf))
                    volume_rate = flow-(.0001*(p+g)*exposure if native else 0.)
                    if continuous and abs(volume_rate)>1e-15:
                        state[7] += exposure/(1000.*volume_rate)*np.log1p(volume_rate*duration/state[6])
                    else:
                        state[7] += duration*exposure/(1000.*state[6])
                    state[:4] += duration*derivative
                    state[6] += volume_rate*duration
                    if growth:
                        state[5] += state[4]*(1.-np.exp(-death*duration))
                        state[4] = end_viable
                    remaining -= duration
                rows.append(state.copy())
            rows = np.asarray(rows)
        answer.append(rows)
        state = rows[-1].copy()
    return answer


def operation_run(case, config, thermo):
    """Run the real campaign while observing every returned closure solution.

    Parameters
    ----------
    case, config : dict
        Operating and biological declarations; units are explicit in the inputs.
    thermo : Path
        Native thermophysical-property input.

    Returns
    -------
    tuple
        Assembly, native histories, and a report of closure counts [-], maximum
        scaled feasibility error [-], and elapsed wall time [s].

    Raises
    ------
    TimeoutError
        The real campaign exceeds the existing wall-time budget at a profiling
        event. Native calls must return to Python before this guard can fire.
    AssertionError
        Any returned closure solution violates the qualification bound.

    Notes
    -----
    The standard profiling hook observes actual return values and arguments;
    it never replaces a model callable. Its previous registration is restored
    even on failure. The portable clock guard retains the old cooperative
    deadline: neither it nor a Python signal handler interrupts a native call
    that has not returned control to Python.
    """
    assembly = build_bioreactor(case, config, thermo)
    report = dict(closure_calls=0, max_scaled_feasibility=0.)
    start = time.perf_counter()  # [s], monotonic wall clock
    observer = _ClosureObserver(assembly.mechanism.closure, start, report)
    previous = sys.getprofile()
    try:
        sys.setprofile(observer)
        histories = assembly.solve()
    finally:
        sys.setprofile(previous)
        report['runtime_s'] = time.perf_counter() - start  # [s]
    return assembly, histories, report


def operation_error(assembly,histories,expected):
    """Measure the existing component-scaled operating-transition error.

    Parameters
    ----------
    assembly : BioreactorAssembly
        Actual input-built biological model and native reactor.
    histories : sequence of DynamicResult
        Native segment histories; time [s], masses [kg], and declared biological auxiliary states.
    expected : sequence of ndarray
        Independent component-wise observations in the same units and order as the native observations.

    Returns
    -------
    float
        Largest absolute reference discrepancy divided by OPERATION_SCALES [-].
    """
    return max(float(np.max(abs(network_observed(assembly,h)-r)/OPERATION_SCALES)) for h,r in zip(histories,expected))


def operation_record_refinement(name, values):
    """Write optional refinement evidence to the explicitly selected output directory.

    Parameters
    ----------
    name : str
        Fixture name or requested property identifier.
    values : array-like
        Reference target values or refinement records; normalized errors are dimensionless.
    """
    if os.environ.get('BIOREACTOR_VERIFICATION_OUTPUT'):
        path = Path(os.environ['BIOREACTOR_VERIFICATION_OUTPUT'])/'refinement.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        report = read_input_json(path) if path.exists() else {}
        report[name] = values
        path.write_text(json.dumps(report,indent=2)+'\n')


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('variant', ['growth', 'growth_death', 'constrained_growth', 'monod'])
def test_population_adaptive_trajectories(variant):
    a = build_bioreactor(*population_load(variant))
    histories = a.solve()
    expected = population_reference(histories, variant)
    for h, exact in zip(histories, expected):
        actual = population_observed(a, h)
        assert np.isfinite(actual).all() and actual.min() >= -1e-10
        assert np.max(abs(actual-exact)/POPULATION_SCALES) < 1e-6
        total = actual[:, 0]+10.*actual[:, 3]+2.*(actual[:, 1]+actual[:, 2])
        exact_total = exact[:, 0]+10.*exact[:, 3]+2.*(exact[:, 1]+exact[:, 2])
        np.testing.assert_allclose(total, exact_total, atol=1e-7, rtol=0.)
    for previous, following in zip(histories[:-1], histories[1:]):
        if np.isclose(following.time[0]/86400., .75):
            before, after = population_observed(a, previous)[-1], population_observed(a, following)[0]
            np.testing.assert_allclose(after[:4]/after[4], before[:4]/before[4], rtol=1e-12, atol=1e-12)
            assert after[5] == pytest.approx(before[5], abs=1e-14)


@pytest.mark.parametrize('variant, expected', [('growth', [.96, .8, .16, .8]),
                                              ('constrained_growth', [.6, .5, .1, .5])])
def test_population_growth_flux_projection(variant, expected):
    a = build_bioreactor(*population_load(variant))
    solution = a.mechanism._solve({'uptake':.96, 'growth_sink':.8, 'secretion':.16})
    np.testing.assert_allclose(solution.fluxes, expected, atol=1e-9, rtol=0.)
    assert a.mechanism.unit_converter.growth_scale == .5


@pytest.mark.parametrize('variant', ['growth', 'growth_death', 'constrained_growth', 'monod'])
def test_population_fixed_step_convergence(variant):
    errors = []
    for step in (.02, .01, .005):
        case, config, thermo = population_load(variant)
        case['numerics'] = {k: (v if k == 'step' else 'fixed-step' if k == 'backend' else None)
                            for k, v in case['numerics'].items()}
        case['numerics']['step']['value'] = step
        a = build_bioreactor(case, config, thermo)
        histories = a.solve()
        exact = population_reference(histories, variant)
        errors.append(np.max(np.concatenate([abs(population_observed(a,h)-r)/POPULATION_SCALES for h,r in zip(histories,exact)])))
    assert errors[1] < .65*errors[0] and errors[2] < .65*errors[1]


@pytest.mark.parametrize('boundary', ['zero_cells', 'zero_nutrient'])
def test_population_zero_inventory_boundary(boundary):
    case, config, thermo = population_load('growth_death')
    case['operation']['runtime'] = dict(unit='day', value=.01)
    config['recipes']['benchmark'] = []
    if boundary == 'zero_cells':
        config['state']['viable_cells_million'] = 0.
    else:
        config['state']['species']['amounts']['nutrient'] = 0.
    a = build_bioreactor(case, config, thermo)
    result, = a.solve()
    actual = population_observed(a, result)
    np.testing.assert_allclose(actual[:, 0], actual[0, 0], atol=1e-9)
    np.testing.assert_allclose(actual[:, 3], 0., atol=1e-10)
    expected = 0. if boundary == 'zero_cells' else np.exp(-.1*.01)
    assert actual[-1, 1] == pytest.approx(expected, abs=1e-8)


def test_population_equivalent_molecular_rate_basis():
    case, config, thermo = population_load('constrained_growth')
    original = build_bioreactor(case, config, thermo).solve()
    config = deepcopy(config)
    config['parameters']['coefficients']['biomass_amount_per_million_cells'] *= 1000.
    for name, declaration in config['model']['kinetics']['flux_basis'].items():
        declaration['amount_unit'] = 'umol'
        if name == 'growth':
            declaration['amount_per_million_cells'] *= 1000.
    for name in ('uptake', 'growth_flux', 'product'):
        config['rate_provider']['output_units'][name] = 'umol/(10^6 cell day)'
    for reaction in config['network']['reactions']:
        reaction['lower'] *= 1000.
        reaction['upper'] *= 1000.
    a = build_bioreactor(case, config, thermo)
    converted = a.solve()
    for x, y in zip(original, converted):
        assert np.max(abs(population_observed(a, x)-population_observed(a, y))/POPULATION_SCALES) < 1e-6


def test_population_equivalent_specific_growth_basis():
    case, config, thermo = population_load('constrained_growth')
    original = build_bioreactor(case, config, thermo).solve()
    kinetics = config['model']['kinetics']
    kinetics['flux_basis']['growth'] = dict(amount_unit='1', normalization='none', time_unit='day')
    kinetics['kinetic_outputs']['exchange_fluxes']['growth_sink'] = 'growth'
    for reaction in config['network']['reactions']:
        if reaction['identifier'] == 'growth_sink':
            reaction['stoichiometry']['B'] *= 2.
            reaction['upper'] /= 2.
    a = build_bioreactor(case, config, thermo)
    converted = a.solve()
    for x, y in zip(original, converted):
        assert np.max(abs(population_observed(a, x)-population_observed(a, y))/POPULATION_SCALES) < 1e-6


@pytest.mark.parametrize('policy', ['unweighted', 'relative-regularized'])
def test_reconciliation_copy_preserves_policy(policy):
    case, config, thermo = network_load('compatible')
    config['model']['reconciliation']['reconciliation_policy'] = policy
    if policy == 'relative-regularized':
        config['model']['reconciliation'].update(reconciliation_relative_bound=1.,
                                                reconciliation_bound_increment=.5)
    original = build_bioreactor(case, config, thermo).mechanism
    original.closure.tolerance = 5e-8
    copied = deepcopy(original)
    assert copied.closure is not original.closure
    assert copied.closure.internal_reaction_ids == original.closure.internal_reaction_ids
    assert copied.closure.tolerance == original.closure.tolerance
    targets = {name:config['parameters']['rates'][name] for name in ('ua', 'ub', 'p', 'q', 'g')}
    np.testing.assert_allclose(copied._solve(targets).fluxes, original._solve(targets).fluxes,
                               atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('variant', NETWORK_VARIANTS)
def test_network_trajectories(variant):
    assembly = build_bioreactor(*network_load(variant))
    histories = assembly.solve()
    for history, expected in zip(histories, network_reference(histories, variant)):
        actual = network_observed(assembly, history)
        assert np.isfinite(actual).all() and actual.min() >= -1e-9
        assert np.max(abs(actual-expected)/NETWORK_SCALES) < 1e-6
        weights = np.array([1., 1., 1., 10., 2., 2., 0., 0.])
        np.testing.assert_allclose(actual@weights, expected@weights, atol=1e-7, rtol=0.)
    for before, after in zip(histories, histories[1:]):
        if np.isclose(after.time[0]/86400., .75):
            x, y = network_observed(assembly, before)[-1], network_observed(assembly, after)[0]
            np.testing.assert_allclose(x[:6]/x[6], y[:6]/y[6], atol=1e-12)
            assert x[7] == pytest.approx(y[7], abs=1e-14)


@pytest.mark.parametrize('variant', NETWORK_VARIANTS)
def test_network_optimum(variant):
    case, config, thermo = network_load(variant)
    a = build_bioreactor(case, config, thermo)
    targets = {n:config['parameters']['rates'][n] for n in ('ua', 'ub', 'p', 'q', 'g')}
    result = a.mechanism._solve(targets)
    flux = dict(zip(result.reaction_ids, result.fluxes))
    z, expected, objective = network_projection(variant)
    actual = np.array([flux[n] for n in targets])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.)
    assert np.sum((actual-np.array(list(targets.values())))**2) == pytest.approx(objective, abs=1e-9)
    assert result.mass_balance_residual_inf < 1e-9 and result.bound_violation_inf < 1e-9
    assert flux['x']+flux.get('x_copy', 0.) == pytest.approx(z[0], abs=2e-6)
    assert flux['y'] == pytest.approx(z[1], abs=2e-6)
    if variant == 'nonunique':
        # Every split of total x between identical columns preserves balances and objective.
        reactions = config['network']['reactions']
        matrix = np.array([[r['stoichiometry'].get(s, 0.) for r in reactions]
                           for s in config['network']['internal_metabolites']])
        ids = [r['identifier'] for r in reactions]
        for fraction in (0., .5, 1.):
            candidate = np.array([flux[n] for n in ids])
            candidate[ids.index('x')], candidate[ids.index('x_copy')] = fraction*z[0], (1.-fraction)*z[0]
            np.testing.assert_allclose(matrix@candidate, 0., atol=2e-6)
            assert candidate.min() >= -1e-9 and candidate.max() <= 10.


@pytest.mark.parametrize('variant', ['conflicting', 'switching', 'growth', 'nonunique'])
def test_network_rename_and_permute(variant, tmp_path):
    case, config, thermo = network_load(variant)
    a = build_bioreactor(case, config, thermo)
    original = a.solve()
    names = ['nutrient_a', 'nutrient_b', 'byproduct']
    identifiers = names+config['network']['internal_metabolites']+[r['identifier'] for r in config['network']['reactions']]
    rename = {name:'renamed_'+str(i) for i,name in enumerate(identifiers)}
    transform = partial(_rename_network_inputs, rename=rename)
    config, thermo = transform(config), transform(read_input_json(thermo))
    for key in ('reactions', 'internal_metabolites', 'exchange_reactions'):
        config['network'][key].reverse()
    config['model']['kinetics']['extracellular_species'].reverse()
    # Parameter paths are strings in the general rule language, not species identities.
    for rule in config['model']['kinetics']['rule_graph']:
        path = rule['expression']['parameter'].split('.')
        rule['expression']['parameter'] = '.'.join(rename.get(part, part) for part in path)
    thermo_path = tmp_path/'thermo.json'
    thermo_path.write_text(json.dumps(thermo))
    b = build_bioreactor(case, config, thermo_path)
    changed = b.solve()
    assert len(original) == len(changed)
    for x, y in zip(original, changed):
        assert np.max(abs(network_observed(a,x)-network_observed(b,y,tuple(rename[n] for n in names)))/NETWORK_SCALES) < 1e-6


def test_network_switching_and_replenishment():
    a = build_bioreactor(*network_load('switching'))
    h = a.solve()
    rows = np.concatenate([network_observed(a,x) for x in h])
    time = np.concatenate([x.time/86400. for x in h])
    # B depletes first; the A-only branch continues, then stops until the bolus.
    for day, empty in ((.25, [False, True]), (.49, [True, True]), (.6, [False, False])):
        row = rows[np.argmin(abs(time-day))]
        assert list(row[:2] < 1e-8) == empty
    assert rows[np.argmin(abs(time-.49)), 3] > rows[np.argmin(abs(time-.25)), 3]
    for column, start, expected in ((1, 0., .2), (0, 0., .44), (0, .5, .9), (1, .5, .9)):
        depleted = (time >= start) & (rows[:, column] < 1e-8)
        if start:
            depleted &= time > start
        assert time[depleted][0] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('variant', ['switching', 'growth'])
def test_network_continuous_reference_convergence(variant):
    errors = []
    for step in (.01, .005, .0025):
        case, config, thermo = network_load(variant)
        case['numerics'] = {k:(v if k == 'step' else 'fixed-step' if k == 'backend' else None)
                            for k,v in case['numerics'].items()}
        case['numerics']['step']['value'] = step
        a = build_bioreactor(case, config, thermo)
        histories = a.solve()
        errors.append(max(np.max(abs(network_observed(a,h)-r)/NETWORK_SCALES)
                          for h,r in zip(histories, network_reference(histories, variant, continuous=True))))
    assert errors[1] < .65*errors[0] and errors[2] < .65*errors[1]


@pytest.mark.parametrize('change', ['tolerance', 'maximum_step'])
def test_network_adaptive_refinement(change):
    case, config, thermo = network_load('growth')
    if change == 'tolerance':
        case['numerics']['relative_tolerance'] /= 10.
        case['numerics']['absolute_tolerance'] /= 10.
    else:
        case['numerics']['maximum_step']['value'] /= 2.
    a = build_bioreactor(case, config, thermo)
    histories = a.solve()
    assert max(np.max(abs(network_observed(a,h)-r)/NETWORK_SCALES)
               for h,r in zip(histories, network_reference(histories, 'growth'))) < 1e-6


@pytest.mark.parametrize('name', CONSTRAINT_CASES)
def test_constraint_policy_optimum(name):
    case, config, thermo = constraint_load(name)
    mechanism = build_bioreactor(case,config,thermo).mechanism
    targets = {n:config['parameters']['rates'][n] for n in CONSTRAINT_IDS}
    result = mechanism._solve(targets)
    z, objective, envelope = constraint_reference(config)
    flux = dict(zip(result.reaction_ids,result.fluxes))
    unit_scale = 1000. if name == 'equivalent_units' else 1.
    np.testing.assert_allclose(np.array([flux['x'],flux['y']])/unit_scale,
                               z/unit_scale,atol=2e-6,rtol=0.)
    np.testing.assert_allclose(np.array([flux[n] for n in CONSTRAINT_IDS])/unit_scale,
                               CONSTRAINT_MAPPING@z/unit_scale,atol=2e-6,rtol=0.)
    assert -result.primary_objective == pytest.approx(objective,abs=1e-8,rel=1e-8)
    assert mechanism.last_reconciliation_relative_bound == envelope
    assert result.mass_balance_residual_inf/unit_scale < 1e-8
    assert result.bound_violation_inf/unit_scale < 1e-8


@pytest.mark.parametrize('name', ['relative','relaxed','untargeted'])
def test_constraint_forward_balances(name):
    case, config, thermo = constraint_load(name)
    a = build_bioreactor(case,config,thermo)
    histories = a.solve()
    z, _, _ = constraint_reference(config)
    ua,ub,p,q,g = CONSTRAINT_MAPPING@z
    state = np.array([10.,10.,0.,0.,1.,0.,1.,0.])
    scales = np.array([10.,10.,2.,.1,1.,.1,1.,.001])
    species = [list(a.phase.name_species).index(n) for n in ('nutrient_a','nutrient_b','byproduct')]
    for h in histories:
        times = h.time/86400.
        if np.isclose(times[0],.5):
            state[:2] += [.6,.2]
            state[6] += .1
        if np.isclose(times[0],.75):
            state[:6] *= 1.-.1/state[6]
            state[6] -= .1
        expected = [state.copy()]
        for dt in np.diff(times):
            state[:4] += dt*state[4]*np.array([-ua,-ub,q,.1*p])
            state[7] += dt*state[4]/(1000.*state[6])
            state[6] += .1*dt
            expected.append(state.copy())
        actual = np.column_stack((h.mass_j_liquid0[:,species]*10000.,h.product_g_liquid0,
                                  h.viable_cells_million_liquid0,h.dead_cells_million_liquid0,
                                  np.asarray(h.vessel_vol).reshape(-1)*1000.,
                                  h.ivcd_million_cell_day_per_ml_liquid0))
        assert np.isfinite(actual).all() and actual.min() >= -1e-9
        assert np.max(abs(actual-expected)/scales) < 1e-6


def test_constraint_infeasible_rejected():
    case,config,thermo = constraint_load('infeasible')
    with pytest.raises(ValueError,match='reference is infeasible'):
        constraint_reference(config)
    with pytest.raises(MetabolicInfeasibleError):
        build_bioreactor(case,config,thermo).solve()


def test_constraint_recovery_retains_problem_or_fails():
    """Check recovery refusal without inventory geometry and the real optimum."""
    import PharmaPy.Metabolic.closures.reconciled as module
    from PharmaPy.Metabolic.closures._recovery import boundary_candidate
    case,config,thermo = constraint_load('relative')
    mechanism = build_bioreactor(case,config,thermo).mechanism
    z, objective, envelope = constraint_reference(config)
    selected = {n: config['parameters']['rates'][n] for n in CONSTRAINT_IDS
                if n not in mechanism.definition.untargeted_exchanges}
    targets = mechanism._targets(selected)
    network = mechanism.network
    scale = max(abs(sum(v for n, v in selected.items()
                        if n not in mechanism.definition.normalization_exclusions)), mechanism.target_floor)
    terms = module._make_terms(network, mechanism.closure.internal_reaction_ids, targets, scale)
    lower, upper = network.bounds({t.reaction_id: ((1. - envelope) * t.value,
                                                   (1. + envelope) * t.value) for t in targets})
    candidate = boundary_candidate(network.stoichiometric_matrix, lower, upper,
                                   terms, None, mechanism.closure.tolerance)
    # Boundary recovery deliberately refuses cases without inventory geometry.
    assert candidate is None
    solution = mechanism._solve(selected)
    values, value = solution.fluxes, -solution.primary_objective
    flux = dict(zip(network.reaction_ids, values))
    np.testing.assert_allclose([flux['x'],flux['y']],z,atol=2e-6,rtol=0.)
    assert value == pytest.approx(objective,abs=1e-8,rel=1e-8)
    assert mechanism.closure._acceptable(values, lower, upper, None)


@pytest.mark.parametrize('name', CONSTRAINT_CASES)
def test_constraint_reference_precision(name):
    _, config, _ = constraint_load(name)
    normal = constraint_reference(config)
    for precision in (40,80):
        refined = constraint_reference(config,solve=constraint_decimal_solver(precision))
        scale = 1000. if name == 'equivalent_units' else 1.
        np.testing.assert_allclose(normal[0]/scale,refined[0]/scale,atol=4e-7,rtol=0.)
        assert normal[1] == pytest.approx(refined[1],abs=2e-9,rel=2e-9)
        assert normal[2] == refined[2]


def test_constraint_floor_unit_equivalence():
    physical = []
    for factor in (1.,1000.):
        case,config,thermo = constraint_floor_case(factor)
        a = build_bioreactor(case,config,thermo)
        result = a.mechanism._solve({n:config['parameters']['rates'][n] for n in CONSTRAINT_IDS})
        flux = dict(zip(result.reaction_ids,result.fluxes))
        physical.append(np.array([flux['x'],flux['y']])/factor)
    # Scale to the small physical target, not to one mmol: expose floor dependence.
    assert np.max(abs(physical[0]-physical[1]))/1e-7 < 1e-6


@pytest.mark.parametrize('seed', [None,(.5,.25),(.6,.3),(1.,.5)])
def test_constraint_nonlinear_multiple_starts(seed):
    case,config,thermo = constraint_load('nonlinear')
    a = build_bioreactor(case,config,thermo)
    if seed is not None:
        x,y = seed
        flux = dict(ua=x+y,ub=y,x=x,y=y,p=.8*x,q=1.8*y,g=.2*(x+y))
        a.mechanism.previous_solution = SimpleNamespace(fluxes=np.array([flux[n] for n in a.mechanism.network.reaction_ids]))
    h, = a.solve()
    z,obj,exposure = constraint_nonlinear_reference()
    result = a.mechanism.last_solution
    actual = dict(zip(result.reaction_ids,result.fluxes))
    np.testing.assert_allclose([actual['x'],actual['y']],z,atol=2e-6,rtol=0.)
    assert -result.primary_objective == pytest.approx(obj,abs=1e-8,rel=1e-8)
    indexes = [list(a.phase.name_species).index(n) for n in ('nutrient_a','nutrient_b','byproduct')]
    expected = [.12-z.sum()*exposure,.04-z[1]*exposure,1.8*z[1]*exposure]
    np.testing.assert_allclose(h.mass_j_liquid0[-1,indexes]*10000.,expected,atol=1e-7,rtol=0.)
    assert h.viable_cells_million_liquid0[-1] == pytest.approx(np.exp((.1*z.sum()-.05)*.1),abs=1e-7)
    assert h.product_g_liquid0[-1] == pytest.approx(.08*z[0]*exposure,abs=1e-8)


def test_constraint_nonlinear_reference_refinement():
    coarse = constraint_nonlinear_reference(257)
    fine = constraint_nonlinear_reference(2049)
    np.testing.assert_allclose(coarse[0],fine[0],atol=4e-7,rtol=0.)
    assert coarse[1] == pytest.approx(fine[1],abs=2e-9,rel=2e-9)


def test_constraint_forward_floor_unit_equivalence():
    physical = []
    for factor in (1.,1000.):
        a = build_bioreactor(*constraint_floor_case(factor))
        h = a.solve()[0]
        dt = (h.time[1]-h.time[0])/86400.
        j = list(a.phase.name_species).index('byproduct')
        x = (h.product_g_liquid0[1]-h.product_g_liquid0[0])/(.1*dt)
        y = (h.mass_j_liquid0[1,j]-h.mass_j_liquid0[0,j])*10000./(2.*dt)
        physical.append(np.array([x,y]))
    assert np.max(abs(physical[0]-physical[1]))/1e-7 < 1e-6


@pytest.mark.parametrize('factor', [1e-10, 1., 1e10])
def test_fixed_nonzero_flux_and_dependent_rows(factor):
    closure = constraint_small_closure([[1., -1.], [2., -2.]], [0., 2.*factor], [4.*factor, 2.*factor])
    result = constraint_solve_small(closure, factor, factor)
    np.testing.assert_allclose(result.fluxes/factor, [2., 2.], atol=1e-8)


@pytest.mark.parametrize('consistent', [True, False])
def test_all_fixed_fluxes(consistent):
    values = [2., 2. if consistent else 3.]
    closure = constraint_small_closure([[1., -1.]], values, values)
    if consistent:
        np.testing.assert_array_equal(constraint_solve_small(closure, 1.).fluxes, values)
    else:
        with pytest.raises(MetabolicInfeasibleError):
            constraint_solve_small(closure, 1.)


def test_inconsistent_eliminated_row():
    closure = constraint_small_closure([[0., 1.]], [0., 2.], [4., 2.])
    with pytest.raises(MetabolicInfeasibleError):
        constraint_solve_small(closure, 1.)


@pytest.mark.parametrize('value', [-2., 0., 2.])
def test_signed_zero_targets_and_unpenalized_directions(value):
    closure = constraint_small_closure([[0., 0.]], [-3., -3.], [3., 3.])
    result = constraint_solve_small(closure, value)
    assert result.fluxes[0] == pytest.approx(value, abs=1e-7)
    assert -3. <= result.fluxes[1] <= 3.
    assert -result.primary_objective < 1e-12


def test_nearly_dependent_rows_preserve_original_balances():
    closure = constraint_small_closure([[1., -1., 0.], [1., -1., 1e-10]],
                            [0., 0., -1.], [3., 3., 1.])
    result = constraint_solve_small(closure, 2.)
    np.testing.assert_allclose(result.fluxes, [2., 2., 0.], atol=1e-7)


def test_false_optimizer_success_is_rejected():
    """Reject an infeasible candidate independently of optimizer status."""
    # A bounded target forces optimization instead of the exact-target fast path.
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [1., 1.])
    _, qualified, _ = constraint_problem(closure, 2.)
    assert not qualified(np.zeros(2))
    assert qualified(np.ones(2))
    np.testing.assert_allclose(constraint_solve_small(closure, 2.).fluxes, [1., 1.], atol=1e-8)


def test_declared_scaling_through_builder_and_copy():
    from copy import deepcopy
    case, config, thermo = constraint_floor_case(1000.)
    declaration = {'value': 1e-6, 'basis': {
        'amount_unit': 'mmol', 'normalization': '10^6 cell', 'time_unit': 'day'}}
    config['model']['reconciliation']['reconciliation_scaling'] = {
        'target_floor': declaration, 'internal_scale': declaration,
        'target_floors': {'ua': {'value': 2e-6, 'basis': declaration['basis']}}}
    mechanism = build_bioreactor(case, config, thermo).mechanism
    assert mechanism.target_floor == pytest.approx(.001)
    assert mechanism.target_floors['ua'] == pytest.approx(.002)
    assert mechanism.internal_scale == pytest.approx(.001)
    targets = {n: config['parameters']['rates'][n] for n in CONSTRAINT_IDS}
    np.testing.assert_allclose(deepcopy(mechanism)._solve(targets).fluxes,
                               mechanism._solve(targets).fluxes, atol=1e-10)


def test_tiny_all_fixed_inconsistency_is_not_roundoff():
    closure = constraint_small_closure([[1., -1.]], [2e-12, 3e-12], [2e-12, 3e-12])
    with pytest.raises(MetabolicInfeasibleError):
        constraint_solve_small(closure, 1e-12, 1e-12)


@pytest.mark.parametrize('bad', [
    {'unknown': 1}, {'target_floors': {'missing_reaction': {'value': 1., 'basis': None}}},
    {'target_floor': {'value': 0., 'basis': None}},
])
def test_invalid_reconciliation_policy_scales_fail_at_construction(bad):
    case, config, thermo = constraint_load('relative')
    config['model']['reconciliation']['reconciliation_scaling'] = bad
    with pytest.raises(ValueError):
        build_bioreactor(case, config, thermo)


def test_mixed_basis_scales_require_compatible_declarations():
    case, config, thermo = constraint_load('relative')
    config['model']['kinetics']['flux_basis']['growth'] = {
        'amount_unit': '1', 'normalization': 'none', 'time_unit': 'day'}
    config['rate_provider']['output_units']['g'] = '1/day'
    # The common molecular floor cannot be used for a specific growth target.
    with pytest.raises(ValueError, match='not interchangeable'):
        build_bioreactor(case, config, thermo)
    policy = config['model']['reconciliation']['reconciliation_scaling']
    policy['target_floors'] = {'g': {'value': 1e-6, 'basis': {
        'amount_unit': '1', 'normalization': 'none', 'time_unit': 'day'}}}
    with pytest.raises(ValueError, match='mixed flux coordinates'):
        build_bioreactor(case, config, thermo)
    policy['internal_scale'] = dict(policy['target_floor'], value=1.)
    mechanism = build_bioreactor(case, config, thermo).mechanism
    result = mechanism._solve({n: config['parameters']['rates'][n] for n in CONSTRAINT_IDS})
    assert np.isfinite(result.fluxes).all()


def test_null_space_retry_preserves_nonzero_fixed_balance():
    """Verify affine null-space reconstruction against a fixed-flux balance."""
    closure = constraint_small_closure([[1., -1., -1.]], [0., 0., 1.], [3., 3., 1.])
    optimize, qualified, _ = constraint_problem(closure, 2.)
    result = optimize(np.array([1., 0.]), eliminate=True)
    assert result.success and qualified(result.x)
    np.testing.assert_allclose(result.x, [2., 1., 1.], atol=1e-8)
    np.testing.assert_allclose(constraint_solve_small(closure, 2.).fluxes, result.x, atol=1e-8)


def test_null_space_roundoff_restores_declared_bounds():
    """Restore roundoff only, then verify a real constrained minimum."""
    from PharmaPy.Metabolic.closures import reconciled
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [3., 3.])
    low, high = np.zeros(2), np.full(2, 3.)  # [-], scaled bounds
    rounded = reconciled._restore_flux_bounds(np.full(2, -1e-14), low, high, closure.tolerance)
    np.testing.assert_array_equal(rounded, np.zeros(2))
    outside = np.full(2, -1e-4)  # [-], well outside the numerical tolerance
    np.testing.assert_array_equal(reconciled._restore_flux_bounds(outside, low, high, closure.tolerance), outside)
    optimize, qualified, terms = constraint_problem(closure, -1.)
    result = optimize(np.ones(2), eliminate=True)
    assert result.success and qualified(result.x)
    assert np.min(result.x) >= 0.
    assert np.max(abs(closure.network.stoichiometric_matrix @ result.x)) < 1e-12
    assert result.fun == pytest.approx(1.)


def test_flux_variability_retains_analytical_intervals():
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.closures.reconciled import ReconciliationTarget
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [3., 3.], internal=('r0',))
    result = closure.solve(MetabolicEnvironment({}, {}),
                           targets=(ReconciliationTarget('r0', 2., 1.),), internal_scale=1.,
                           fva_reactions=('r0',), fva_fraction=.1)
    # J=x^2+(x-2)^2=2+2(x-1)^2; the 10% ceiling is 2.2.
    np.testing.assert_allclose(result.fva_intervals['r0'],
                               [1.-np.sqrt(.1), 1.+np.sqrt(.1)], atol=1e-7)


@pytest.mark.parametrize('name',OPERATION_CASES)
@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_campaign(name,backend):
    case,config,thermo = operation_load(name,backend)
    assembly,histories,report = operation_run(case,config,thermo)
    expected = operation_reference(case,config,histories,continuous=backend=='scipy')
    report['max_scaled_trajectory_error'] = operation_error(assembly,histories,expected)
    assert report['max_scaled_trajectory_error'] < 1e-6
    for h in histories:
        assert np.isfinite(network_observed(assembly,h)).all()
        assert np.min(h.mass_j_liquid0/np.array(assembly.phase.mw)*1e6) >= -1e-9
    for before,after in zip(histories,histories[1:]):
        assert before.time[-1] == after.time[0]
    if name in ('ordered','reversed'):
        assert network_observed(assembly,histories[-1])[-1,0] == pytest.approx(1.65 if name=='ordered' else 1.70,abs=1e-10)
    if name in ('growth','growth_supply'):
        tighter = operation_reference(case,config,histories,continuous=True,precision=.25)
        nominal = operation_reference(case,config,histories,continuous=True)
        assert max(np.max(abs(a-b)/OPERATION_SCALES) for a,b in zip(tighter,nominal)) < 2e-7
    if os.environ.get('BIOREACTOR_VERIFICATION_OUTPUT'):
        folder = Path(os.environ['BIOREACTOR_VERIFICATION_OUTPUT'])/name/backend
        folder.mkdir(parents=True,exist_ok=True)
        times = np.concatenate([h.time/86400. for h in histories])
        header = ','.join(['time_day',*NETWORK_FIELDS])
        for file,values in [('trajectories',np.vstack([network_observed(assembly,h) for h in histories])),('reference',np.vstack(expected))]:
            np.savetxt(folder/(file+'.csv'),np.column_stack((times,values)),delimiter=',',header=header,comments='')
        report.update(case=name,backend=backend,passed=True,reference='continuous' if backend=='scipy' else 'discrete')
        (folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


@pytest.mark.parametrize('name',['competing','schedule','combined','growth','growth_supply'])
def test_operation_fixed_refinement(name):
    errors=[]
    for step in (.002,.001,.0005):
        case,config,thermo=operation_load(name,step=step)
        assembly,histories,_=operation_run(case,config,thermo)
        assert operation_error(assembly,histories,operation_reference(case,config,histories)) < 1e-6
        errors.append(operation_error(assembly,histories,operation_reference(case,config,histories,continuous=True)))
    assert errors[2] < 1e-3
    assert max(errors) < 1e-6 or errors[2] < errors[1] < errors[0]
    operation_record_refinement('fixed_'+name,dict(steps_day=[.002,.001,.0005],scaled_errors=errors))


@pytest.mark.parametrize('change',['tolerance','maximum_step'])
@pytest.mark.parametrize('name',['competing','combined','growth','growth_supply'])
def test_operation_adaptive_refinement(name,change):
    case,config,thermo=operation_load(name,'scipy')
    if change=='tolerance':
        case['numerics']['relative_tolerance']/=10.
        case['numerics']['absolute_tolerance']/=10.
    else:
        case['numerics']['maximum_step']['value']/=2.
    a,h,_=operation_run(case,config,thermo)
    maximum = operation_error(a,h,operation_reference(case,config,h,continuous=True))
    assert maximum < 1e-6
    operation_record_refinement('adaptive_'+name+'_'+change,dict(scaled_error=maximum,numerics=case['numerics']))


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_off_grid_transition_brackets(backend):
    case,config,thermo=operation_load('competing',backend,step=.0005)
    a,h,_=operation_run(case,config,thermo)
    rows=network_observed(a,h[0]);times=h[0].time/86400.
    # B is exhausted at .0113/.5. Thereafter x=1.25 with y=0.
    for column,exact in ((1,.0226),(0,.0226+(.037-1.5*.0226)/1.25)):
        first=np.flatnonzero(rows[:,column] <= 1e-9)[0]
        assert times[first]-times[first-1] <= .0005+1e-12
        assert times[first-1]-1e-10 <= exact <= times[first]+1e-10


@pytest.mark.parametrize('name',['endpoints','combined'])
@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_reset_rebuild(name,backend):
    case,config,thermo=operation_load(name,backend)
    a,h,_=operation_run(case,config,thermo)
    a.reset();repeated=a.solve()
    b,fresh,_=operation_run(case,config,thermo)
    for x,y,z in zip(h,repeated,fresh):
        np.testing.assert_allclose(network_observed(a,x),network_observed(a,y),atol=1e-12,rtol=1e-9)
        np.testing.assert_allclose(network_observed(a,x),network_observed(b,z),atol=1e-12,rtol=1e-9)


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_native_volume(backend):
    case,config,thermo=operation_load('combined',backend)
    case['operation']['volume_policy']='native'
    a,h,_=operation_run(case,config,thermo)
    assert operation_error(a,h,operation_reference(case,config,h,continuous=backend=='scipy')) < 1e-6


@pytest.mark.parametrize('bad',[('sample',-1.),('sample',1.),('sample',2.),('flow',-1.)])
def test_operation_invalid_operations_are_atomic(bad):
    case,config,thermo=operation_load('combined')
    kind,value=bad
    event=deepcopy(next(e for e in config['recipes']['benchmark'] if e['event_type']==kind))
    event['time']['value']=0.
    event['volume' if kind=='sample' else 'volume_flow']['value']=value
    config['recipes']['benchmark']=[event]
    a=build_bioreactor(case,config,thermo)
    initial_mass=a.phase.mass_j.copy()
    with pytest.raises(ValueError):
        a.solve()
    np.testing.assert_array_equal(a.phase.mass_j,initial_mass)
    assert not a._inlets


def test_operation_relative_starvation_is_infeasible():
    case,config,thermo=operation_load('relative')
    config['recipes']['benchmark']=[]
    with pytest.raises(MetabolicInfeasibleError):
        operation_run(case,config,thermo)


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
@pytest.mark.parametrize('name',['combined','relative'])
def test_operation_equivalent_units_and_names(name,backend,tmp_path):
    case,config,thermo=operation_load(name,backend)
    original,histories,_=operation_run(case,config,thermo)
    case['operation']['runtime']={'value':.06*24.,'unit':'h'}
    case['operation']['initial_volume']={'value':1000.,'unit':'mL'}
    case['numerics']['step']={'value':.001*24.,'unit':'h'}
    for event in config['recipes']['benchmark']:
        event['time']={'value':event['time']['value']*24.,'unit':'h'}
        if 'volume' in event:
            event['volume']={'value':event['volume']['value']*1000.,'unit':'mL'}
        if 'volume_flow' in event:
            event['volume_flow']={'value':event['volume_flow']['value']*1000./24.,'unit':'mL/h'}
        if 'concentrations' in event:
            event['concentrations']['unit']='umol/L'
            event['concentrations']['values']={n:v*1000. for n,v in event['concentrations']['values'].items()}
    config['state']['species']['unit']='umol'
    config['state']['species']['amounts']={n:v*1000. for n,v in config['state']['species']['amounts'].items()}
    for name in ('ua','ub','p','q','g'):
        config['parameters']['rates'][name]*=1000.
        config['rate_provider']['output_units'][name]='umol/(10^6 cell day)'
    for basis in config['model']['kinetics']['flux_basis'].values():
        basis['amount_unit']='umol'
        if 'amount_per_million_cells' in basis:
            basis['amount_per_million_cells']*=1000.
    for reaction in config['network']['reactions']:
        reaction['lower']*=1000.;reaction['upper']*=1000.
    names=['nutrient_a','nutrient_b','byproduct']
    identifiers=names+config['network']['internal_metabolites']+[r['identifier'] for r in config['network']['reactions']]+['first','second']
    rename={name:'renamed_'+str(i) for i,name in enumerate(identifiers)}
    transform = partial(_rename_operation_inputs, rename=rename)
    config=transform(config)
    for rule in config['model']['kinetics']['rule_graph']:
        rule['expression']['parameter']='.'.join(rename.get(x,x) for x in rule['expression']['parameter'].split('.'))
    for key in ('reactions','internal_metabolites','exchange_reactions'):
        config['network'][key].reverse()
    renamed_thermo=tmp_path/'thermo.json'
    renamed_thermo.write_text(json.dumps(transform(read_input_json(thermo))))
    changed,results,_=operation_run(case,config,renamed_thermo)
    assert len(histories)==len(results)
    for a,b in zip(histories,results):
        np.testing.assert_allclose(a.time,b.time,atol=1e-10,rtol=0.)
        assert np.max(abs(network_observed(original,a)-network_observed(changed,b,tuple(rename[n] for n in names)))/OPERATION_SCALES) < 1e-6


def test_operation_continuation_does_not_replay_old_events():
    case,config,thermo=operation_load('combined')
    whole,all_history,_=operation_run(case,config,thermo)
    part=build_bioreactor(case,config,thermo)
    grid=part.time_grid.copy();split=.03*86400.
    part.time_grid=grid[grid<=split]
    first=part.solve()
    part.time_grid=grid[grid>=split]
    second=part.solve()
    np.testing.assert_allclose(network_observed(part,second[-1])[-1],network_observed(whole,all_history[-1])[-1],atol=1e-10,rtol=1e-8)
    assert first[-1].time[-1]==second[0].time[0]


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_pathway_receives_native_continuous_supply(backend):
    folder=Path(__file__).parents[2]/'examples/bioreactors/generic_fed_batch/inputs'
    case=read_input_json(folder/'case.json')
    config=read_input_json(folder/'mechanism.json')
    case['operation']['runtime']={'value':.06,'unit':'h'}
    case['operation']['volume_policy']='working-volume'
    case['numerics']=operation_load('combined',backend)[0]['numerics']
    case['numerics']['step']={'value':.001,'unit':'h'}
    config['model']['pathways']['pathway_bound_rules']=[]
    config['state']['species']['amounts']['nutrient']=0.
    config['recipes']['fed_batch']=[dict(time={'value':0.,'unit':'h'},event_type='flow',
        volume_flow={'value':.1,'unit':'L/h'},concentrations={'unit':'mmol/L','values':{'nutrient':1.}},composition='supplied')]
    a=build_bioreactor(case,config,folder/'thermo.json')
    h=a.solve()[0]
    hours=h.time/3600.
    np.testing.assert_allclose(h.biomass_kg_liquid0,1e-4+1e-5*hours,atol=1e-12,rtol=0.)
    index=list(a.phase.name_species).index('nutrient')
    np.testing.assert_allclose(h.mass_j_liquid0[:,index],0.,atol=1e-12,rtol=0.)
    np.testing.assert_allclose(np.asarray(h.vessel_vol).reshape(-1)*1000.,1.+.1*hours,atol=1e-9,rtol=0.)


@pytest.mark.parametrize('reverse',[False,True])
def test_operation_coincident_mixed_units_keep_declared_order(reverse):
    case,config,thermo=operation_load('reversed' if reverse else 'ordered')
    events=config['recipes']['benchmark']
    events[0]['time']={'value':.003*24.,'unit':'h'}
    events[1]['time']={'value':.003,'unit':'day'}
    a,h,_=operation_run(case,config,thermo)
    assert len(h)==2
    assert network_observed(a,h[-1])[-1,0]==pytest.approx(1.70 if reverse else 1.65,abs=1e-10)


def test_operation_final_endpoint_in_another_unit_is_not_applied():
    case,config,thermo=operation_load('endpoints')
    case['operation']['runtime']={'value':.003,'unit':'day'}
    config['recipes']['benchmark'][-1]['time']={'value':.003*24.,'unit':'h'}
    a,h,_=operation_run(case,config,thermo)
    assert len(h)==1
    assert network_observed(a,h[0])[-1,6]==pytest.approx(1.02,abs=1e-12)


@pytest.mark.parametrize('bad',['negative_concentration','unknown_species','empty_identifier'])
def test_operation_invalid_inlet_preserves_existing_supply(bad):
    case,config,thermo=operation_load('multiple')
    a=build_bioreactor(case,config,thermo)
    a._set_inlet(a.events[0])
    existing=a._inlets.copy()
    event=deepcopy(a.events[1])
    if bad=='negative_concentration':
        event['concentrations_mmol_l']['nutrient_b']=-1.
    elif bad=='unknown_species':
        event['concentrations_mmol_l']['missing']=1.
    else:
        event['inlet']=''
    with pytest.raises(ValueError):
        a._set_inlet(event)
    assert a._inlets==existing
    assert len(a.unit.inlet_connections)==1


# Coupled qualification

@pytest.mark.parametrize('name,shape,nullity',[
    ('coupled04',(10,16),6),('coupled10',(22,36),14),
    ('coupled20',(42,68),26),('coupled20_scaled',(42,68),26)])
def test_coupled_configuration_and_connected_coordinates(name,shape,nullity):
    case,config,thermo=coupled_load(name)
    assembly=build_bioreactor(case,config,thermo)
    assert assembly.input_audit()['feed_composition']=='COMPLETE'
    assert assembly.unit.result is None
    matrix=network_matrix(config)
    assert matrix.shape==shape
    assert np.linalg.matrix_rank(matrix)==shape[0]
    reference=Reference(config)
    assert reference.mapping.shape[1]==nullity
    np.testing.assert_allclose(matrix@reference.mapping,0.,atol=1e-12,rtol=0.)
    # Every reaction belongs to the same metabolite/reaction graph component.
    reached={0}
    adjacency=abs(matrix).T@abs(matrix)>0.
    while True:
        grown=reached|set(np.flatnonzero(adjacency[list(reached)].any(axis=0)))
        if grown==reached:break
        reached=grown
    assert len(reached)==shape[1]
    # All exchanges and biological bases are explicit and finite.
    assert len(config['model']['kinetics']['exchange_mappings'])==reference.n+1
    assert config['model']['kinetics']['flux_basis']['growth']['amount_per_million_cells']==1.


def test_coupled_reference_qp_against_closed_form():
    h=np.diag([2.,4.]);c=np.array([4.,4.])
    a=np.array([[-1.,0.],[0.,-1.],[1.,1.]])
    x,proof=quadratic_program(h,c,a,np.array([0.,0.,1.]))
    np.testing.assert_allclose(x,[2./3.,1./3.],atol=1e-12,rtol=0.)
    assert proof['objective_gap_bound']<1e-10
    with pytest.raises(ValueError,match='infeasible'):
        quadratic_program(h,c,np.array([[1.,0.],[-1.,0.]]),np.array([0.,-1.]))


@pytest.mark.parametrize('name',COUPLED_CASES)
def test_coupled_independent_reference_static_certificate(name):
    _,config,_=coupled_load(name);r=Reference(config);state=r.initial()
    for caps in (None,np.zeros(r.n),np.full(r.n,.01)):
        flux,proof=r.solve(state,caps)
        np.testing.assert_allclose(network_matrix(config)@flux,0.,atol=1e-10,rtol=0.)
        assert proof['objective_gap_bound']<2e-9*max(1.,proof['objective'])
        assert proof['stationarity']<1e-9
        if caps is not None:
            assert max(flux[r.selected[:r.n]]-caps)<1e-10
        if caps is not None and not caps.any():
            np.testing.assert_allclose(flux[r.selected],0.,atol=1e-10)
            assert proof['envelope']==1.


def test_coupled_reference_row_scaling_equivalence():
    references=[Reference(coupled_load(n)[1]) for n in ('coupled20','coupled20_scaled')]
    answers=[r.solve(r.initial()) for r in references]
    np.testing.assert_allclose(answers[0][0],answers[1][0],atol=1e-10,rtol=1e-9)
    assert answers[0][1]['objective']==pytest.approx(answers[1][1]['objective'],abs=1e-10)


@pytest.mark.parametrize('name',COUPLED_CASES)
def test_coupled_declared_kinetics_match_independent_formulas(name):
    case,config,thermo=coupled_load(name)
    assembly=build_bioreactor(case,config,thermo)
    reference=Reference(config)
    for integral in (0.,.00022,.0005):
        state=reference.initial();state[-1]=integral
        state[:reference.n]*=.2
        snapshot=CultureSnapshot(dict(zip(reference.species,state[:reference.n+1]/state[-2])),
            state[-2],state[reference.n+1],state[reference.n+2],state[reference.n+3],state[-1])
        rates=assembly.mechanism.rate_provider.evaluate(snapshot,assembly.mechanism.conditions).rates
        np.testing.assert_allclose([rates[r] for r in reference.exchanges],reference.targets(state),
                                   atol=1e-14,rtol=1e-12)
    assert assembly.unit.result is None


def test_coupled_formal_inventory_identity_and_sampling():
    _,config,_=coupled_load('coupled10');r=Reference(config);y=r.initial()
    supply=np.linspace(.001,.01,r.n+1)
    rates=r.rhs(y,supply,.05)
    # Formal mmol: liquid pools + one mmol/million live/dead cells + P/0.1 g/mmol.
    weights=np.r_[np.ones(r.n+3),10.,0.,0.]
    assert weights@rates==pytest.approx(sum(supply),abs=1e-10)
    before=weights@y;exposure=y[-1];fraction=1.-.02/y[-2]
    after=y.copy();after[:r.n+4]*=fraction;after[-2]-=.02
    assert weights@after==pytest.approx(before*fraction,abs=1e-12)
    assert after[-1]==exposure


def test_coupled_reference_constant_growth_against_analytic_solution():
    case,config,_=coupled_load('coupled04');config=deepcopy(config)
    config['recipes']['benchmark']=[]
    config['state']['species']['amounts']={s:10. for s in config['model']['kinetics']['extracellular_species']}
    case['operation']['runtime']['value']=.004
    r=Reference(config);y=r.initial();flux,_=r.solve(y)
    growth=flux[r.growth_index];death=config['parameters']['rates']['death'];net=growth-death
    times,states=r.trajectory(case)[0]
    exposure=np.expm1(net*times)/net
    expected=np.repeat(y[None,:],len(times),axis=0)
    expected[:,:r.n]-=exposure[:,None]*flux[r.selected[:r.n]]
    expected[:,r.n]+=exposure*flux[r.ids.index('w')]
    expected[:,r.n+1]=np.exp(net*times)
    expected[:,r.n+2]=death*exposure
    expected[:,r.n+3]=.1*flux[r.ids.index('p')]*exposure
    expected[:,-1]=exposure/1000.
    np.testing.assert_allclose(states,expected,atol=1e-11,rtol=1e-10)
    refined=r.trajectory(case,refinement=.5)[0][1]
    np.testing.assert_allclose(states,refined,atol=1e-12,rtol=1e-11)


def test_coupled_reference_finite_exposure_bound():
    _,config,_=coupled_load('coupled04');r=Reference(config);state=r.initial()
    flux,proof=r.fixed_step(state,np.zeros(r.n+1),.002)
    unconstrained,_=r.solve(state)
    np.testing.assert_allclose(flux,unconstrained,atol=1e-9,rtol=1e-8)
    assert proof['objective_gap_bound']<1e-9*max(1.,proof['objective'])


def test_coupled_reference_constrained_exposure_refinement():
    _,config,_=coupled_load('coupled04');r=Reference(config);state=r.initial()
    state[:r.n]=np.array([config['parameters']['rates'][name] for name in r.exchanges[:r.n]])*.0002
    answers=[r.fixed_step(state,np.zeros(r.n+1),.002,gap_tolerance=t) for t in (1e-9,1e-10)]
    for flux,proof in answers:
        exposure=.001*state[r.n+1]*(1.+np.exp((flux[r.growth_index]-.03)*.002))
        assert np.max(exposure*flux[r.selected[:r.n]]-state[:r.n])<1e-12
        assert proof['objective_gap_bound']<1e-9*max(1.,proof['objective'])
    np.testing.assert_allclose(answers[0][0][r.selected],answers[1][0][r.selected],atol=2e-7,rtol=0.)
def test_reference_comparison_preserves_event_sides_and_requires_coverage():
    reference=np.array([[0.,0.,1.],[0.,1.,2.],[1.,1.,7.],[1.,2.,8.]])
    rows=reference.copy()
    rows[2,2]+=.25
    assert compare(rows,reference,np.array([1.]),require_all=True)==.25
    with pytest.raises(ValueError,match='every saved time'):
        compare(np.array([[0.,.5,1.5]]),reference,np.array([1.]),require_all=True)


def test_operation_budget_retains_the_deadline_boundary():
    _check_operation_budget(OPERATION_TIMEOUT_S)
    with pytest.raises(TimeoutError, match='60-second campaign guard'):
        _check_operation_budget(OPERATION_TIMEOUT_S + 1.)


def test_operation_observer_counts_every_real_closure_return_and_restores_profile():
    case, config, thermo = operation_load('single')
    previous = sys.getprofile()
    _, observed, report = operation_run(case, config, thermo)
    assert sys.getprofile() is previous
    assembly = build_bioreactor(case, config, thermo)
    profile = cProfile.Profile()
    reference = profile.runcall(assembly.solve)
    code = assembly.mechanism.closure.solve.__func__.__code__
    count = sum(entry.callcount for entry in profile.getstats() if entry.code is code)
    assert count > 0
    assert report['closure_calls'] == count
    assert len(observed) == len(reference)
    for actual, expected in zip(observed, reference):
        np.testing.assert_array_equal(actual.time, expected.time)
        np.testing.assert_array_equal(actual.mass_j_liquid0, expected.mass_j_liquid0)


def test_operation_observer_restores_profile_after_real_infeasibility():
    case, config, thermo = operation_load('single')
    for reaction in config['network']['reactions']:
        # [model flux unit], fix an unbalanced uptake and close every other reaction.
        flux = 1. if reaction['identifier'] == 'ua' else 0.
        reaction['lower'] = reaction['upper'] = flux
    previous = sys.getprofile()
    with pytest.raises(MetabolicInfeasibleError, match='rate-reconciled MFA failed: all fluxes fixed'):
        operation_run(case, config, thermo)
    assert sys.getprofile() is previous
