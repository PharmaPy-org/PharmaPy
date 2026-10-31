"""Independent verification oracle for the configured coupled-network synthetic family.

This test-only module does not import PharmaPy. It is not a simulation entry point.
The continuous reference uses reduced-coordinate convex QPs and DOP853; finite-step
nonlinear closure references use certified growth-interval relaxations.
"""

from functools import partial
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, minimize_scalar

BASE = Path(__file__).parents[2]/'tests/Bioreactor/fixtures/reconciliation/coupled_systems'
CASES = ('coupled04', 'coupled10', 'coupled20', 'coupled20_scaled')




# Real fixture construction and independent qualification helpers.
class _GrowthIntervals:
    """Hold the shared bounds of one independent growth-interval certificate.

    The state replaces closure cells only: the QPs, interval queue ordering,
    exposure formula, and acceptance bounds remain unchanged.
    """

    def __init__(self, reference, state, step, remaining, envelope):
        """Initialize one envelope's shared interval-search state.

        Parameters
        ----------
        reference : Reference
            Actual independent coupled-network oracle.
        state : ndarray, shape (n_uptake + 6,)
            Extracellular amounts [mmol], viable/dead [10^6 cells], product [g],
            volume [L], and integrated viable concentration [10^6 cell day/mL].
        step : float
            Fixed integration horizon [day].
        remaining : ndarray, shape (n_uptake,)
            Available uptake-species inventories [mmol], including declared supply.
        envelope : float
            Relative target-bound half-width [-].

        Notes
        -----
        The object replaces shared closure cells only. It preserves the original
        interval ordering, feasible upper bounds, and independent gap certificates.
        """
        self.reference = reference
        self.state = state  # [mmol; 10^6 cells; g; L; 10^6 cell day/mL], documented oracle order
        self.step = step  # [day]
        self.remaining = remaining  # [mmol], available extracellular inventories
        self.envelope = envelope  # [-], relative target-bound width
        self.queue = []
        self.best = None
        self.upper = np.inf  # [-], objective upper bound before any feasible candidate
        self.counter = 0
        self.evaluations = 0

    def exposure(self, g):
        """Evaluate the original trapezoidal viable-population exposure.

        Parameters
        ----------
        g : float
            Specific growth rate [1/day], already converted to the reference population basis.

        Returns
        -------
        float
            Integrated viable population [10^6 cell day].
        """
        return .5*self.state[self.reference.n+1]*self.step*(1.+np.exp((g-self.reference.parameters['rates']['death'])*self.step))

    def candidate(self, flux, info):

        """Update the feasible upper bound only after the original inventory audit.

        Parameters
        ----------
        flux : ndarray
            Full reaction flux vector in reaction-identifier order and the declared model flux basis.
        info : dict
            Independent objective and certified objective-gap bound [-].

        Returns
        -------
        bool
            Whether the candidate is inventory-feasible within the existing arithmetic allowance.
        """
        used = self.exposure(flux[self.reference.growth_index])*flux[self.reference.selected[:self.reference.n]]  # [mmol]
        # An infeasible objective is not an upper bound. Permit only
        # arithmetic uncertainty relative to the actual inventory terms.
        rounding = 8*np.finfo(float).eps*(abs(used)+abs(self.remaining))  # [mmol], original eight-ulp inventory allowance
        if np.any(used-self.remaining > rounding):
            return False
        if info['objective'] < self.upper:
            self.best,self.upper=flux,info['objective']  # [model flux basis, -]
        return True

    def interval(self, left, right):

        """Queue the original certified lower bound and midpoint candidate.

        Parameters
        ----------
        left : float
            Left growth-coordinate bound [1/day].
        right : float
            Right growth-coordinate bound [1/day].
        """
        try:
            flux,info = self.reference.solve(self.state,self.remaining/self.exposure(left),self.envelope,(left,right))
        except ValueError:
            return
        lower = info['objective']-info['objective_gap_bound']
        self.candidate(flux,info)
        middle=(left+right)/2.
        try:
            point,proof=self.reference.solve(self.state,self.remaining/self.exposure(middle),self.envelope,(middle,middle))
            self.candidate(point,proof)
        except ValueError:
            pass
        self.counter+=1
        heapq.heappush(self.queue,(lower,self.counter,left,right))

    def objective(self, growth):

        """Evaluate and count one independent scalar-polishing candidate.

        Parameters
        ----------
        growth : float
            Specific growth coordinate [1/day] in the coupled reference.

        Returns
        -------
        float
            Feasible dimensionless objective [-]; infinity retains genuine infeasibility.
        """
        self.evaluations+=1
        try:
            flux,info=self.reference.solve(self.state,self.remaining/self.exposure(growth),self.envelope,(growth,growth))
        except ValueError:
            return np.inf
        return info['objective'] if self.candidate(flux,info) else np.inf
def _reference_rhs(t, y, *, reference, supply, flow, boundary):
    """Adapt the independent balance equations to SciPy's time-first API.

    Parameters
    ----------
    reference : Reference
        Actual independent coupled-system oracle.
    t : float
        Integration time [day].
    y : ndarray
        Ordered state vector; component units and order are specified in Notes.
    supply : ndarray
        Ordered extracellular inlet supplies [mmol/day].
    flow : float
        Inlet volume flow [L/day].
    boundary : float
        Numerical depletion boundary [mmol], distinct from a physical model parameter.

    Returns
    -------
    ndarray
        State derivatives in the input state component units per day.

    Notes
    -----
    States contain the declared uptake species and one waste species [mmol],
    viable/dead populations [10^6 cells], product [g], volume [L], and
    integrated viable concentration [10^6 cell day/mL]. Time is in day.
    """
    return reference.rhs(y, supply, flow, boundary)


def _depletion_root(t, y, *, i, boundary):
    """Locate the independent oracle's depletion boundary.

    Parameters
    ----------
    t : float
        Integration time [day].
    y : ndarray
        Ordered state vector; component units and order are specified in Notes.
    i : int
        Selected species or pivot-column index.
    boundary : float
        Numerical depletion boundary [mmol], distinct from a physical model parameter.

    Returns
    -------
    float
        Selected species amount minus twice the numerical boundary [mmol].

    Notes
    -----
    States contain the declared uptake species and one waste species [mmol],
    viable/dead populations [10^6 cells], product [g], volume [L], and
    integrated viable concentration [10^6 cell day/mL]. Time is in day.
    """
    return y[i]-2.*boundary


def network_matrix(config):
    """Assemble the independently declared stoichiometric matrix.

    Parameters
    ----------
    config : dict
        Biological input; species, coefficients, bounds, and flux bases retain their declared units.

    Returns
    -------
    ndarray
        Stoichiometric coefficients [-], metabolite rows by reaction columns.
    """
    network = config['network']
    return np.array([[r['stoichiometry'].get(s, 0.) for r in network['reactions']]
                     for s in network['internal_metabolites']])


def quadratic_program(hessian, linear, matrix, bound, tolerance=1e-11):
    """Primal active-set QP, independent of production SLSQP/recovery code.

    Minimize .5*x'H*x-c'x, A*x <= b. Diagonal conditioning preserves the
    problem. A final primal/dual certificate is mandatory, not solver status alone.

    Parameters
    ----------
    hessian : ndarray
        Quadratic objective coefficients [1/(coordinate_i coordinate_j)].
    linear : ndarray
        Linear objective coefficients [1/model coordinate unit].
    matrix : ndarray
        Declared stoichiometric/constraint coefficients; row units divided by their corresponding coordinate units.
    bound : ndarray
        Inequality right-hand side in each constraint row's units.
    tolerance : float
        Dimensionless active-set convergence tolerance [-]; certification is checked separately.

    Returns
    -------
    tuple
        Independent optimum in the original coordinate units and primal/dual/objective-gap certificate [-].

    Raises
    ------
    ValueError
        The independent problem is infeasible.
    RuntimeError
        Feasibility, KKT certification, or the iteration budget fails.
    """
    coordinate = 1./np.sqrt(np.diag(hessian))
    h = coordinate[:,None]*hessian*coordinate
    c = linear*coordinate
    a = matrix*coordinate
    norm = np.linalg.norm(a,axis=1)
    if np.any(bound[norm == 0.] < 0.):
        raise ValueError('reference infeasible')
    keep = norm > 0.
    a,b = a[keep]/norm[keep,None],bound[keep]/norm[keep]
    feasible = linprog(np.zeros(len(c)), A_ub=a, b_ub=b,
                       bounds=[(None,None)]*len(c), method='highs',
                       options={'primal_feasibility_tolerance':1e-10,
                                'dual_feasibility_tolerance':1e-10})
    if feasible.status == 2:
        raise ValueError('reference infeasible')
    if not feasible.success:
        raise RuntimeError('reference feasibility LP: '+feasible.message)
    x = feasible.x
    active = []
    for i in np.flatnonzero(abs(a@x-b)<1e-9):
        if np.linalg.matrix_rank(a[active+[i]],tol=1e-10)>len(active):
            active.append(int(i))
    for iteration in range(2000):
        rows = a[active]
        system = np.block([[h,rows.T],[rows,np.zeros((len(active),len(active)))]] )
        answer = np.linalg.solve(system,np.r_[c-h@x,np.zeros(len(active))])
        direction,multipliers = answer[:len(c)],answer[len(c):]
        if active:
            q,_ = np.linalg.qr(rows.T,mode='complete')
            tangent = q[:,len(active):]
            direction = tangent@(tangent.T@direction)
        if np.max(abs(direction)) <= tolerance:
            if len(active) and multipliers.min() < -tolerance:
                active.pop(int(np.argmin(multipliers)))
                continue
            # Resolve the KKT system for the absolute point, avoiding accumulated steps.
            answer = np.linalg.solve(system,np.r_[c,b[active]])
            x,multipliers = answer[:len(c)],answer[len(c):]
            dual = np.zeros(len(b)); dual[active] = np.maximum(multipliers,0.)
            gradient = c-a.T@dual
            primal = .5*x@h@x-c@x
            dual_value = -.5*gradient@np.linalg.solve(h,gradient)-b@dual
            allowance = 64*np.finfo(float).eps*(1.+abs(primal)+abs(dual_value))
            gap = primal-dual_value
            residual = max(0.,float(np.max(a@x-b)))
            stationarity = float(np.max(abs(h@x-c+a.T@dual)))
            if residual>1e-9 or stationarity>1e-9 or gap < -allowance:
                raise RuntimeError('reference KKT certificate failed')
            return x*coordinate,dict(primal_violation=residual,stationarity=stationarity,
                objective_gap_bound=float(2.*(max(gap,0.)+allowance)),iterations=iteration+1)
        rates = a@direction
        eligible = [i for i in range(len(b)) if i not in active and rates[i]>tolerance]
        alpha,blocker = 1.,None
        for i in eligible:
            distance = max(0.,b[i]-a[i]@x)/rates[i]
            if distance < alpha:
                if np.linalg.matrix_rank(a[active+[i]],tol=1e-10)<=len(active):
                    continue
                alpha,blocker = distance,i
        x += alpha*direction
        if blocker is not None:
            active.append(blocker)
    raise RuntimeError('reference active-set iteration limit')


class Reference:
    """Explicit synthetic-family equations, separate from the production rule graph."""

    def __init__(self, config):
        """Initialize the independent coupled-network reference.

        Parameters
        ----------
        config : dict
            Biological input; species, coefficients, bounds, and flux bases retain their declared units.

        Notes
        -----
        The coupled state order is extracellular amounts [mmol], viable/dead
        [10^6 cells], product [g], volume [L], and exposure [10^6 cell day/mL].
        """
        self.config = config
        self.species = config['model']['kinetics']['extracellular_species']
        self.n = len(self.species)-1
        reactions = config['network']['reactions']
        self.ids = [r['identifier'] for r in reactions]
        self.exchanges = [f'u{i+1:02}' for i in range(self.n)]+['g','p','w']
        self.free = [i for i,r in enumerate(self.ids) if r.startswith(('u','t')) or r in ('g','p')]
        dependent = [i for i in range(len(self.ids)) if i not in self.free]
        matrix = network_matrix(config)
        # Solve a square block analytically implied by A/H/C/P balances.
        self.mapping = np.zeros((len(reactions),len(self.free)))
        self.mapping[self.free] = np.eye(len(self.free))
        self.mapping[dependent] = np.linalg.solve(matrix[:,dependent],-matrix[:,self.free])
        if np.max(abs(matrix@self.mapping)) > 1e-11:
            raise ValueError('reference coordinate map violates balances')
        self.selected = [self.ids.index(r) for r in self.exchanges]
        self.internal = [self.ids.index(r) for r in config['model']['reconciliation']['internal_reactions']]
        self.lower = np.array([r['lower'] for r in reactions])
        self.upper = np.array([r['upper'] for r in reactions])
        self.growth_index = self.ids.index('g')
        self.parameters = config['parameters']

    def initial(self):
        """Read the independent oracle's configured initial inventories.

        Returns
        -------
        ndarray
            Extracellular species [mmol], viable/dead [10^6 cells], product [g], volume [L], exposure [10^6 cell day/mL].
        """
        state = self.config['state']
        return np.array([state['species']['amounts'][s] for s in self.species]+
            [state['viable_cells_million'],state['dead_cells_million'],state['product_g'],1.,state['integral']])

    def targets(self, state):
        """Evaluate the independently declared uptake, growth, and production laws.

        Parameters
        ----------
        state : ndarray
            Ordered state vector; component units and order are specified in Notes.

        Returns
        -------
        ndarray
            Rates in exchange order: uptake/waste/product [mmol/(10^6 cell day)] and growth [1/day].

        Notes
        -----
        States contain the declared uptake species and one waste species [mmol],
        viable/dead populations [10^6 cells], product [g], volume [L], and
        integrated viable concentration [10^6 cell day/mL]. Time is in day.
        """
        p = self.parameters
        phase = p['responses']['phase']
        late = 1./(1.+np.exp(-phase['slope']*(state[-1]-phase['midpoint'])))
        uptake = 1.+(phase['uptake_late']-1.)*late
        values = []
        for i,s in enumerate(self.species[:-1]):
            concentration = max(0.,state[i])/state[-2]
            half = p['affinities'][s]
            factor = concentration/(half+concentration) if half else 1.
            values.append(p['rates'][self.exchanges[i]]*uptake*factor)
        return np.r_[values,p['rates']['g']*(1.+(phase['growth_late']-1.)*late),
                     p['rates']['p']*(1.+(phase['product_late']-1.)*late),p['rates']['w']]

    def problem(self, state, target_values=None):
        """Assemble the unchanged independently normalized quadratic residuals.

        Parameters
        ----------
        state : ndarray
            Ordered state vector; component units and order are specified in Notes.
        target_values : dict or None
            Explicit target rates keyed by exchange reaction, in each declared model flux basis; None evaluates the rate laws.

        Returns
        -------
        tuple
            Targets in declared model-flux units, normalized residual matrix, and normalized rhs [-].

        Notes
        -----
        States contain the declared uptake species and one waste species [mmol],
        viable/dead populations [10^6 cells], product [g], volume [L], and
        integrated viable concentration [10^6 cell day/mL]. Time is in day.
        """
        targets = (self.targets(state) if target_values is None else
                   np.array([target_values[name] for name in self.exchanges]))
        rows = self.mapping[self.selected]/np.maximum(abs(targets),1e-6)[:,None]
        rhs = targets/np.maximum(abs(targets),1e-6)
        rows = np.vstack((rows,self.mapping[self.internal]/max(abs(sum(targets)),1e-6)))
        rhs = np.r_[rhs,np.zeros(len(self.internal))]
        return targets,rows,rhs

    def solve(self, state, caps=None, envelope=None, growth_interval=None):
        """Solve and certify the independent reduced-coordinate quadratic program.

        Parameters
        ----------
        state : ndarray
            Ordered state vector; component units and order are specified in Notes.
        caps : ndarray or None
            Maximum uptake fluxes [mmol/(10^6 cell day)], in uptake-species order.
        envelope : float
            Relative target-bound half-width [-].
        growth_interval : tuple of float or None
            Lower and upper growth-coordinate limits in the declared growth flux unit.

        Returns
        -------
        tuple
            Full fluxes in reaction order/model units, and feasibility/objective-gap certificate [-].

        Notes
        -----
        States contain the declared uptake species and one waste species [mmol],
        viable/dead populations [10^6 cells], product [g], volume [L], and
        integrated viable concentration [10^6 cell day/mL]. Time is in day.
        """
        targets,rows,rhs = self.problem(state)
        for width in ((.5,1.) if envelope is None else (envelope,)):
            lower,upper = self.lower.copy(),self.upper.copy()
            lower[self.selected] = np.maximum(lower[self.selected],(1.-width)*targets)
            upper[self.selected] = np.minimum(upper[self.selected],(1.+width)*targets)
            if caps is not None:
                upper[self.selected[:self.n]] = np.minimum(upper[self.selected[:self.n]],caps)
            if growth_interval is not None:
                lower[self.growth_index] = max(lower[self.growth_index],growth_interval[0])
                upper[self.growth_index] = min(upper[self.growth_index],growth_interval[1])
            if np.any(lower>upper):
                continue
            try:
                z,certificate = quadratic_program(rows.T@rows,rows.T@rhs,
                    np.vstack((self.mapping,-self.mapping)),np.r_[upper,-lower])
            except ValueError as error:
                if str(error) != 'reference infeasible':
                    raise
                continue
            flux = self.mapping@z
            objective = float(np.sum((rows@z-rhs)**2))
            certificate.update(objective=objective,envelope=width)
            return flux,certificate
        raise ValueError('reference infeasible')

    def fixed_step(self, state, supply, step, gap_tolerance=1e-9, max_intervals=4096,
                   polish_tolerance=None):
        """Bound the nonlinear optimum using monotone growth-exposure relaxations.

        For g in [lo,hi], E(lo) <= E(g); u>=0 makes capacity/E(lo) a
        necessary linear constraint. Each interval QP supplies a global lower
        bound; its flux is feasible only if checked with its own actual E(g).

        Parameters
        ----------
        state : ndarray
            Ordered state vector; component units and order are specified in Notes.
        supply : ndarray
            Ordered extracellular inlet supplies [mmol/day].
        step : float
            Fixed integration horizon [day].
        gap_tolerance : float
            Relative certified objective-gap limit [-]; the default is tighter than production flux acceptance.
        max_intervals : int
            Numerical interval-search budget [-]; exhaustion remains a genuine failure.
        polish_tolerance : float or None
            Optional growth-coordinate refinement tolerance [1/day], retaining the original global certificate.

        Notes
        -----
        States contain the declared uptake species and one waste species [mmol],
        viable/dead populations [10^6 cells], product [g], volume [L], and
        integrated viable concentration [10^6 cell day/mL]. Time is in day.

        Returns
        -------
        tuple
            Full reaction fluxes in declared model units and original global objective-gap certificate [-].

        Raises
        ------
        ValueError
            Declared external drains exceed inventory or the reference is infeasible.
        RuntimeError
            Bounds contradict the certificate or the interval-search budget is exhausted.
        """
        n = self.n
        remaining = state[:n]+step*supply[:n]
        remaining[-1] -= step*self.parameters['rates']['degradation']*state[n-1]
        if np.any(remaining < 0.):
            raise ValueError('reference external drain exceeds inventory')

        for envelope in (.5,1.):
            target = self.targets(state)[n]
            lo,hi = max(0.,(1.-envelope)*target),min(2.,(1.+envelope)*target)
            search = _GrowthIntervals(self, state, step, remaining, envelope)


            search.interval(lo,hi)
            for _ in range(max_intervals):
                if not search.queue:break
                lower,_,left,right=heapq.heappop(search.queue)
                if search.best is not None and search.upper < lower:
                    raise RuntimeError('reference upper bound is below the certified lower bound')
                if search.best is not None and search.upper-lower <= gap_tolerance*max(1.,abs(search.upper)):
                    search.evaluations=0
                    if polish_tolerance is not None:
                        # Refine rates inside the still-competitive growth intervals.
                        # The existing global lower bound remains the certificate;
                        # scalar minimization can only improve a feasible upper bound.
                        competitive=[(left,right),*[(a,b) for bound,_,a,b in search.queue if bound<=search.upper]]
                        bounds=(min(search.best[self.growth_index],min(a for a,b in competitive)),
                                max(search.best[self.growth_index],max(b for a,b in competitive)))

                        if bounds[1]-bounds[0]>polish_tolerance:
                            minimize_scalar(search.objective,bounds=bounds,method='bounded',
                                            options={'xatol':polish_tolerance,'maxiter':200})
                    if search.upper<lower:
                        raise RuntimeError('polished reference violates the certified lower bound')
                    return search.best,dict(objective=search.upper,objective_gap_bound=search.upper-lower,
                                     envelope=envelope,intervals=search.counter,polish_evaluations=search.evaluations)
                if lower>search.upper:continue
                if search.counter+2>max_intervals:
                    raise RuntimeError('reference growth-interval budget exhausted; no certified answer')
                middle=(left+right)/2.
                search.interval(left,middle);search.interval(middle,right)
            if search.queue:
                raise RuntimeError('reference growth-interval budget exhausted; no certified answer')
        raise ValueError('reference infeasible')

    def rhs(self, state, supply, flow, boundary=1e-13):
        """Evaluate the independent coupled population and material balances.

        Parameters
        ----------
        state : ndarray
            Ordered state vector; component units and order are specified in Notes.
        supply : ndarray
            Ordered extracellular inlet supplies [mmol/day].
        flow : float
            Inlet volume flow [L/day].
        boundary : float
            Numerical depletion boundary [mmol], distinct from a physical model parameter.

        Returns
        -------
        ndarray
            State derivatives in the corresponding state component units per day.

        Notes
        -----
        States contain the declared uptake species and one waste species [mmol],
        viable/dead populations [10^6 cells], product [g], volume [L], and
        integrated viable concentration [10^6 cell day/mL]. Time is in day.
        """
        evaluated = state.copy()
        evaluated[:self.n+1] = np.where(evaluated[:self.n+1]<=boundary,0.,evaluated[:self.n+1])
        viable=evaluated[self.n+1]
        caps=np.where(evaluated[:self.n]==0.,supply[:self.n]/max(viable,1e-300),np.inf)
        flux,certificate=self.solve(evaluated,caps)
        if certificate['objective_gap_bound']>2e-9*max(1.,certificate['objective']):
            raise RuntimeError('insufficient reference optimization accuracy')
        u=flux[self.selected[:self.n]]
        g,p,w=flux[self.selected[-3:]]
        rates=np.zeros_like(state)
        rates[:self.n+1]=supply
        rates[:self.n]-=viable*u
        rates[self.n]+=viable*w
        decay=self.parameters['rates']['degradation']*evaluated[self.n-1]
        rates[self.n-1]-=decay;rates[self.n]+=decay
        death=self.parameters['rates']['death']
        rates[self.n+1:]=[(g-death)*viable,death*viable,.1*p*viable,flow,viable/(1000.*evaluated[-2])]
        return rates

    def trajectory(self, case, refinement=1.):
        """Independent continuous balances and explicitly scheduled jump maps.

        Return one (time, state) pair per recipe segment, keeping both event sides.
        These packaged inputs use day/L/mmol; no production normalization is used.

        Parameters
        ----------
        case : dict
            Operating input with explicit time, volume, temperature, and recipe units.
        refinement : float
            Numerical tolerance and maximum-step multiplier [-], distinct from physical parameters.

        Returns
        -------
        list of tuple
            One (time [day], state) pair per recipe segment, retaining both event sides and the documented state component units.

        Raises
        ------
        RuntimeError
            The independent integrator fails or makes no progress.
        """
        end=case['operation']['runtime']['value']
        step=case['numerics']['step']['value']
        grid=np.r_[np.arange(0.,end,step),end]
        events=self.config['recipes']['benchmark']
        boundaries=sorted({0.,end,*[e['time']['value'] for e in events if 0.<e['time']['value']<end]})
        state=self.initial();inlets={};answer=[];boundary=1e-13*refinement
        for start,stop in zip(boundaries[:-1],boundaries[1:]):
            for event in events:
                if event['time']['value'] != start:continue
                if event['event_type']=='flow':
                    flow=event['volume_flow']['value']
                    values=event['concentrations']['values']
                    inlets[event['inlet']]=(flow,flow*np.array([values.get(s,0.) for s in self.species]))
                elif event['event_type']=='feed':
                    volume=event['volume']['value'];values=event['concentrations']['values']
                    state[:self.n+1]+=volume*np.array([values.get(s,0.) for s in self.species]);state[-2]+=volume
                else:
                    volume=event['volume']['value'];state[:self.n+4]*=1.-volume/state[-2];state[-2]-=volume
            flow=sum(f for f,_ in inlets.values());supply=sum((v for _,v in inlets.values()),start=np.zeros(self.n+1))
            times=np.r_[start,grid[(grid>start+1e-12)&(grid<stop-1e-12)],stop]
            samples=[];now=start;cursor=0
            while now<stop:
                candidates=np.flatnonzero(state[:self.n]>2.*boundary)
                roots=[]
                for i in candidates:
                    root = partial(_depletion_root, i=i, boundary=boundary)
                    root.terminal=True;root.direction=-1.;roots.append(root)
                solution=solve_ivp(partial(_reference_rhs, reference=self, supply=supply, flow=flow, boundary=boundary),(now,stop),state,
                    method='DOP853',rtol=2e-12*refinement,atol=2e-14*refinement,
                    max_step=.001*refinement,events=roots or None,dense_output=True)
                if not solution.success:raise RuntimeError(solution.message)
                last=solution.t[-1]
                while cursor<len(times) and times[cursor]<=last:
                    samples.append(solution.sol(times[cursor]));cursor+=1
                state=solution.y[:,-1].copy();now=last
                for i,records in zip(candidates,solution.t_events or ()):
                    if len(records):state[i]=0.
                if last<stop and not any(len(r) for r in solution.t_events):
                    raise RuntimeError('reference integration made no progress')
            answer.append((times,np.asarray(samples)))
        return answer


def _diagnostic_exposure(g, *, attributes):
    """Evaluate the frozen inventory exposure without replacing rate methods.

    Parameters
    ----------
    g : float
        Specific growth rate [1/day], already converted to the reference population basis.
    attributes : dict
        Captured real inventory fields, including populations [10^6 cells], horizon [day], and explicit flux/cell conversions.

    Returns
    -------
    float
        Population-time exposure on the captured inventory's cell-flux basis.
    """
    return attributes['viable']*attributes['step_day']*attributes['cell_flux_scale']/2.*(1.+np.exp((g-attributes['death'])*attributes['step_day']))


def load(name):
    """Read one coupled-system qualification input set.

    Parameters
    ----------
    name : str
        Fixture name or requested property identifier.

    Returns
    -------
    tuple
        Case, mechanism mappings with explicit units, and thermophysical JSON path.
    """
    path = BASE/'inputs'/name
    return json.loads((path/'case.json').read_text()), json.loads((path/'mechanism.json').read_text()), path/'thermo.json'


def compare(rows, reference, scale, require_all=False):
    """Compare saved matching times and event sides without interpolation.

    Parameters
    ----------
    rows : ndarray
        Normalized residual coefficients [1/model coordinate unit], or saved observation rows with their declared axes.
    reference : Reference or ndarray
        Independent oracle or saved independent trajectory with component-wise observation units.
    scale : float or ndarray
        Positive normalization in the corresponding flux or observation component units.
    require_all : bool
        Require every saved time/event side to be covered by the reference.

    Returns
    -------
    float
        Largest component-scaled reference discrepancy [-].
    """
    errors=[]
    for segment in np.unique(rows[:,0]):
        part=rows[rows[:,0]==segment];ref=reference[reference[:,0]==segment]
        # Only common saved times: no interpolation across depletion or event jumps.
        right=np.clip(np.searchsorted(ref[:,1],part[:,1]),0,len(ref)-1)
        left=np.maximum(right-1,0)
        index=np.where(abs(ref[left,1]-part[:,1])<abs(ref[right,1]-part[:,1]),left,right)
        valid=abs(ref[index,1]-part[:,1])<1e-11
        if require_all and not valid.all():
            raise ValueError('reference does not cover every saved time and event side')
        errors.extend(abs(part[valid,2:]-ref[index[valid],2:])/scale)
    return float(np.max(errors))


def diagnose_failure(report):
    """Certify feasibility/optimality of the frozen failed closure independently.

    Parameters
    ----------
    report : dict
        Captured real failed-closure diagnostics with explicit bounds, inventories, and flux units.

    Returns
    -------
    dict
        Independent feasibility/optimality certificate [-], or NO_CAPTURE when no real failed closure was recorded.
    """
    data=report.get('failed_closure')
    if data is None:return {'status':'NO_CAPTURE'}
    r=Reference(load(report['case'])[1])
    values={t['reaction_id']:t['value'] for t in data['targets']}
    target,rows,rhs=r.problem(r.initial(), target_values=values)
    lower,upper=r.lower.copy(),r.upper.copy()
    for name,(low,high) in data['bounds'].items():
        i=r.ids.index(name);lower[i]=max(lower[i],low);upper[i]=min(upper[i],high)
    matrix=np.vstack((r.mapping,-r.mapping));capacity=np.r_[upper,-lower]
    availability=data['availability'];attributes=None
    if availability is not None:
        attributes=availability['attributes']
        amounts=np.array(attributes['amounts']);other=np.array(attributes['other']);source=np.array(attributes['matrix'])
        scale=np.array(attributes['scale'])
        if availability['type']=='InventoryAvailability':
            exposure = partial(_diagnostic_exposure, attributes=attributes)
            index=attributes['growth_index']
            minimum=exposure(attributes['growth'] if index is None else lower[index]*attributes['growth_conversion'])
            keep=amounts+other>=0.
            matrix=np.vstack((matrix,-source[keep]@r.mapping))
            capacity=np.r_[capacity,(amounts+other)[keep]/minimum]
        else:
            matrix=np.vstack((matrix,-(source/scale[:,None])@r.mapping))
            capacity=np.r_[capacity,(amounts+other)/scale]
    z,proof=quadratic_program(rows.T@rows,rows.T@rhs,matrix,capacity)
    flux=r.mapping@z
    inventory_error=0.
    if attributes is not None:
        factor=1.
        if availability['type']=='InventoryAvailability':
            factor=exposure(attributes['growth'] if index is None else flux[index]*attributes['growth_conversion'])
        inventory_error=max(0.,float(np.max(-(amounts+other+factor*(source@flux))/scale)))
    proof.update(objective=float(sum((rows@z-rhs)**2)),original_inventory_violation=inventory_error,
                 original_balance_residual=0.)
    proof['original_balance_residual']=float(max(abs(network_matrix(r.config)@flux)))
    proof['status']='CERTIFIED_FEASIBLE_OPTIMUM' if inventory_error<=1e-10 else 'RELAXATION_ONLY'
    proof['scope']='The relaxation optimum is also feasible for the original problem.' if proof['status']=='CERTIFIED_FEASIBLE_OPTIMUM' else 'No nonlinear optimum claim.'
    return proof
