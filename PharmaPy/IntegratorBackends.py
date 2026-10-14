import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from scipy.integrate import solve_ivp
from PharmaPy._assimulo import (CVode, Explicit_Problem, IDA,
                                Implicit_Problem)
from PharmaPy.Commons import TerminateSimulation
from PharmaPy.DataClasses import *


def attach_assimulo_events(problem, unit, implicit=False):
    """
    Give an Assimulo problem the vessel's switching surfaces.

    Assimulo locates each sign change by root-finding, stops there and
    restarts, which is what lets the integrator take long steps between
    regime changes instead of shortening them to stumble across one.

    The explicit and implicit problem classes disagree on the signature:
    Explicit_Problem asks for ``(t, y, sw)`` while Implicit_Problem asks for
    ``(t, y, yd, sw)``.
    """

    if implicit:
        def state_events(time, states, derivatives, sw):
            return unit.evaluate_events(time, states)
    else:
        def state_events(time, states, sw):
            return unit.evaluate_events(time, states)

    def handle_event(solver, event_info):
        # event_info is (state_event_flags, time_event_flag)
        flags = event_info[0]

        triggered = [
            index for index, flag in enumerate(flags) if flag
        ]

        if unit.handle_event(solver.t, triggered):
            raise TerminateSimulation

    problem.state_events = state_events
    problem.handle_event = handle_event


class IntegratorBackend(ABC):

    def __init__(self):
        self._compiled = False

        # Neutral linear-solver request, set by the unit operation.
        self.linear_solver = None

    # Backends that cannot integrate a differential-algebraic system set
    # this False, and refuse at compile time rather than silently treating a
    # residual row as a derivative.
    supports_algebraic = False

    def check_algebraic_support(self, unit):
        """
        Every backend accepts a residual-carrying state vector; only some can
        solve one.

        A unit with no algebraic states costs nothing here, which is the
        common case. A unit that has them and lands on an ODE-only backend is
        refused outright: CVode would integrate the residual rows as if they
        were derivatives and return numbers that look plausible and are
        wrong.
        """

        if not unit.has_algebraic_balance or self.supports_algebraic:
            return

        keys = [
            str(key)
            for key in unit.solver_state_collection.algebraic_keys
        ]

        raise NotImplementedError(
            f"{type(self).__name__} solves ODEs only, but this unit declares "
            f"algebraic state(s) {keys}. Use AssimuloDAEBackend, or "
            "DiffeqpyBackend, which handles them through a mass matrix."
        )

    def set_linear_solver(self, kind):
        """
        Ask for a class of linear solver by neutral name.

        Unit operations know the shape of their own Jacobian ("krylov" for a
        large sparse discretized system, "dense" for a handful of states) but
        should not know which integrator they were handed. Each backend
        translates the request into its own vocabulary, and a backend with no
        such choice keeps its default: this is a performance hint, not a
        correctness requirement.
        """

        self.linear_solver = kind

    @abstractmethod
    def compile_integrator(
        self,
        unit,
        eval_sens=False,
        jac_v_prod=False,
        options=None,
        verbose=True,
        any_event=True,
    ):
        pass

    @abstractmethod
    def solve(
        self,
        unit,
        runtime=None,
        time_grid=None,
        eval_sens=False,
        jac_v_prod=False,
        verbose=True,
        options=None,
        any_event=True,
    ):
        pass

    def fast_solve(
        self,
        unit,
        runtime=None,
        time_grid=None,
        verbose=True,
    ):
        """
        Optional optimization for already-compiled integrators.
        Default implementation simply calls solve().
        """
        return self.solve(
            unit,
            runtime=runtime,
            time_grid=time_grid,
            verbose=verbose,
        )
    

class AssimuloBackend(IntegratorBackend):

    # Neutral name -> Assimulo/Sundials linear solver
    LINEAR_SOLVERS = {
        "krylov": "SPGMR",
        "dense": "DENSE",
        "sparse": "SPARSE",
    }

    def __init__(self,options={'maxh':1} ):

        super().__init__()

        self._problem = None
        self._solver = None
        self.options = options
        self.state_event_list = []

        self.eval_sens = False
        self.jac_v_prod = False

    def set_linear_solver(self, kind):

        super().set_linear_solver(kind)

        name = self.LINEAR_SOLVERS.get(kind)

        if name is not None and self._solver is not None:
            self._solver.linear_solver = name

    def compile_integrator(
            self,
            unit,
            eval_sens=False,
            jac_v_prod=False,
            options=None,
            verbose=True,
            any_event=True,
    ):

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod

        unit.reset()

        states_init = unit.create_solver_init_states()
        unit.save_initial_solver_state(states_init,unit.elapsed_time)
        
        self.check_algebraic_support(unit)

        self.set_ode_problem(unit,states_init)

        solver = CVode(self._problem)

        solver.iter = "Newton"
        solver.discr = "BDF"

        if options: #flagged for deprecation

            for name,val in options.items():

                setattr(solver,name,val)

                if name == "time_limit":
                    solver.report_continuously = True
        if self.options:
            for name,val in self.options.items():
            
                setattr(solver,name,val)

                if name == "time_limit":
                    solver.report_continuously = True

        if eval_sens:

            solver.sensmethod = "SIMULTANEOUS"
            solver.suppress_sens = False
            solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50


        self._solver = solver
        unit.configure_solver()

        self._compiled = True

        return states_init

    def solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            eval_sens=False,
            jac_v_prod=False,
            verbose=True,
            options=None,
            any_event=True,
    ):

        if (
            not self._compiled
            or eval_sens != self.eval_sens
            or jac_v_prod != self.jac_v_prod
        ):

            states_init = self.compile_integrator(
                unit,
                eval_sens,
                jac_v_prod,
                options,
                verbose,
                any_event,
            )

            unit.derivatives = self._problem.rhs(
                unit.elapsed_time,
                states_init,
            )

        time, states = self.fast_solve(
            unit,
            runtime=runtime,
            time_grid=time_grid,
            verbose=verbose,
        )

        unit.retrieve_results(time,states)

        return time, states
    def fast_solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            verbose=True,
    ):

        if not self._compiled:
            raise RuntimeError("Integrator has not been compiled.")

        states_init = unit.create_solver_init_states()

        if runtime is not None:
            final_time = unit.elapsed_time + runtime
        elif time_grid is not None:
            final_time = time_grid[-1]
        else:
            raise ValueError(
                "Either runtime or time_grid must be supplied."
            )

        self._solver.t = unit.elapsed_time
        self._solver.y = states_init

        self._solver.initialize()

        return self._solver.simulate(
            final_time,
            ncp_list=time_grid,
        )
    def set_ode_problem(
            self,
            unit,
            states_init,
    ):

        def model(time, states, sw=None):
            rhs = unit.unit_model(time=time, states=states)

            if len(rhs) != len(states):
                raise RuntimeError(
                    f"Model returned {len(rhs)} values for {len(states)} "
                    "states."
                )

            return rhs

        events = unit.compiled_events

        if events:
            problem = Explicit_Problem(
                model,
                states_init,
                t0=unit.elapsed_time,
                sw0=[True] * len(events),
            )

            attach_assimulo_events(problem, unit)

        else:
            problem = Explicit_Problem(
                model,
                states_init,
                t0=unit.elapsed_time
            )

        self._problem = problem

    def unit_jacobian(self, t, y):
        return self.jac_states_fun(t, y)

    def jac_states_numerical(self, time, states, params, return_only=True):
        #TODO check if necessary
        if return_only:
            return self.jac_states_vals
        else:
            def wrap_states(st): return self.unit_model(time, st, params)

            abstol = self.sundials_opt['atol']
            reltol = self.sundials_opt['rtol']
            jac_states = numerical_jac_central(wrap_states, states,
                                               dx=dx_jac_x,
                                               abs_tol=abstol, rel_tol=reltol)

            return jac_states

    def jac_params_numerical(self, time, states, params):
        #TODO check if necessary
        def wrap_params(theta): return self.unit_model(time, states, theta)

        abstol = self.sundials_opt['atol']
        reltol = self.sundials_opt['rtol']
        p_bar = self.sundials_opt['pbar']

        dp = np.abs(p_bar) * np.sqrt(max(reltol, eps))

        jac_params = numerical_jac_central(wrap_params, params,
                                           dx=dp,
                                           abs_tol=abstol, rel_tol=reltol)

        return jac_params
    
    def rhs_sensitivity(self, time, states, sens, params):

        jac_params_vals = self.jac_params_fn(time, states, params)

        jac_states_vals = self.jac_states_fn(time, states, params,
                                             return_only=False)

        rhs_sens = np.dot(jac_states_vals, sens) + jac_params_vals

        self.jac_states_vals = jac_states_vals

        return rhs_sens
    

class AssimuloDAEBackend(IntegratorBackend):
    """
    Assimulo IDA backend for differential-algebraic units.

    CVode integrates explicit ODEs only, so a unit that declares algebraic
    states needs the implicit solver. The vessel's contract is unchanged:
    ``unit_model`` returns one vector holding derivatives in the
    differential slots and residuals in the algebraic ones. This backend
    turns that into IDA's residual form

        res = yd - f(y)   on differential rows
        res =      f(y)   on algebraic rows

    so no mechanism has to know which solver it ended up under.
    """

    supports_algebraic = True

    LINEAR_SOLVERS = {
        "krylov": "SPGMR",
        "dense": "DENSE",
        "sparse": "SPARSE",
    }

    def __init__(self, options=None):

        """Store IDA options.

        Parameters
        ----------
        options : dict, optional
            IDA solver attributes. Defaults to ``{'maxh': 1}``, the previous
            default maximum step [s]; a copy is stored so that callers and
            instances never share one mutable default.
        """
        super().__init__()

        self._problem = None
        self._solver = None
        self.options = {"maxh": 1} if options is None else dict(options)

        self.eval_sens = False
        self.jac_v_prod = False

    def set_linear_solver(self, kind):

        super().set_linear_solver(kind)

        name = self.LINEAR_SOLVERS.get(kind)

        if name is not None and self._solver is not None:
            self._solver.linear_solver = name

    def make_residual(self, unit):
        """
        Wrap unit_model as an IDA residual.

        ``differential`` is precomputed because this runs on every residual
        evaluation; ``np.where`` on a cached mask is cheaper than branching
        per state.
        """

        algebraic = unit.solver_state_collection.algebraic_mask
        differential = ~algebraic

        def residual(time, states, derivatives, sw=None):
            values = np.asarray(
                unit.unit_model(time=time, states=states),
                dtype=float,
            )

            return np.where(
                differential,
                derivatives - values,
                values,
            )

        return residual

    def initial_derivatives(self, unit, states_init):
        """Consistent yd0: f(y0) on differential rows, zero on algebraic ones.

        IDA only needs the differential entries to be right; the algebraic
        entries of yd never enter the residual. The vessel evaluates them with
        ``differential_only=True`` so no algebraic constraint is evaluated at a
        possibly inconsistent initial point.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Compiled unit.
        states_init : ndarray
            Packed initial states [native state units].

        Returns
        -------
        ndarray
            Initial derivatives [state units/s], zero on algebraic rows.
        """
        return unit.create_solver_init_derivatives(states_init)

    def compile_integrator(
            self,
            unit,
            eval_sens=False,
            jac_v_prod=False,
            options=None,
            verbose=True,
            any_event=True,
    ):

        """Build the IDA problem and solver for one unit.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Unit with at least one algebraic state.
        eval_sens, jac_v_prod : bool, optional
            Not supported; must be False.
        options : dict, optional
            Per-call IDA attributes. They are applied after the instance options,
            so a per-call value overrides the instance default.
        verbose : bool, optional
            Print solver output.
        any_event : bool, optional
            Register the unit's state events.

        Returns
        -------
        ndarray
            Packed initial solver states [native state units].

        Raises
        ------
        NotImplementedError
            Sensitivities or Jacobian-vector products are requested.
        """
        if eval_sens or jac_v_prod:
            raise NotImplementedError(
                "AssimuloDAEBackend does not expose sensitivities or Jacobian-vector products."
            )

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod

        unit.reset()

        states_init = unit.create_solver_init_states()
        unit.save_initial_solver_state(states_init, unit.elapsed_time)

        derivatives_init = self.initial_derivatives(unit, states_init)

        events = unit.compiled_events

        problem = Implicit_Problem(
            self.make_residual(unit),
            states_init,
            derivatives_init,
            t0=unit.elapsed_time,
            sw0=[True] * len(events) if events else None,
        )

        # 1 marks a differential state, 0 an algebraic one. suppress_alg
        # keeps the algebraic residuals out of the error test, which is the
        # usual choice for a semi-explicit index-1 system.
        problem.algvar = np.where(
            unit.solver_state_collection.algebraic_mask, 0.0, 1.0
        )

        if events:
            attach_assimulo_events(problem, unit, implicit=True)

        self._problem = problem

        solver = IDA(problem)
        solver.suppress_alg = True

        for source in (self.options, options):
            if not source:
                continue
            for name, value in source.items():
                setattr(solver, name, value)
                if name == "time_limit":
                    solver.report_continuously = True

        if not verbose:
            solver.verbosity = 50

        self._solver = solver
        unit.configure_solver()

        self._compiled = True

        return states_init

    def solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            eval_sens=False,
            jac_v_prod=False,
            verbose=True,
            options=None,
            any_event=True,
    ):

        if (
            not self._compiled
            or eval_sens != self.eval_sens
            or jac_v_prod != self.jac_v_prod
        ):
            self.compile_integrator(
                unit, eval_sens, jac_v_prod, options, verbose, any_event,
            )

        time, states = self.fast_solve(
            unit,
            runtime=runtime,
            time_grid=time_grid,
            verbose=verbose,
        )

        unit.retrieve_results(time, states)

        return time, states

    def fast_solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            verbose=True,
    ):

        if not self._compiled:
            raise RuntimeError("Integrator has not been compiled.")

        states_init = unit.create_solver_init_states()

        if runtime is not None:
            final_time = unit.elapsed_time + runtime
        elif time_grid is not None:
            final_time = time_grid[-1]
        else:
            raise ValueError(
                "Either runtime or time_grid must be supplied."
            )

        self._solver.t = unit.elapsed_time
        self._solver.y = states_init
        self._solver.yd = self.initial_derivatives(unit, states_init)

        self._solver.make_consistent('IDA_YA_YDP_INIT')

        # IDA returns derivatives as well; the vessel only consumes (t, y).
        time, states, _ = self._solver.simulate(
            final_time,
            ncp_list=time_grid,
        )

        return time, states

class DiffeqpyBackend(IntegratorBackend):
    """
    DifferentialEquations.jl backend, reached through diffeqpy.

    Interchangeable with AssimuloBackend: a unit operation can be handed
    either without changing anything else. Option names follow Assimulo's
    (``atol``, ``rtol``, ``maxh``) and are translated onto SciML's; SciML's
    own names pass through untouched, so anything the table does not cover is
    still reachable.

    The Julia runtime is started on first use rather than at import. Starting
    it costs seconds and loads a large dependency tree, and most sessions
    never touch this backend.

    Parameters
    ----------
    options : dict, optional
        Solver options, in either vocabulary.
    algorithm : str or Julia algorithm, optional
        A key of ``ALGORITHMS``, or an already-constructed Julia algorithm
        object for anything not listed there. Defaults to ``'bdf'``
        (Sundials CVODE_BDF), which is the integrator AssimuloBackend uses,
        so the two backends can be compared directly.
    """

    # Assimulo option -> SciML keyword
    OPTION_ALIASES = {
        "atol": "abstol",
        "rtol": "reltol",
        "maxh": "dtmax",
        "minh": "dtmin",
        "inith": "dt",
        "maxsteps": "maxiters",
    }

    # Assimulo controls with no SciML counterpart. Dropped rather than
    # forwarded, because SciML would reject them outright.
    IGNORED_OPTIONS = frozenset({
        "verbosity",
        "report_continuously",
        "time_limit",
        "iter",
        "discr",
        "num_threads",
        "clock_step",
    })

    # Julia constructor expressions, evaluated in the DifferentialEquations
    # namespace. Kept as source so a Symbol argument stays a Symbol.
    #
    # The pure-Julia stiff solvers are pinned to AutoFiniteDiff: their default
    # is forward-mode AD, which evaluates the right-hand side on Dual numbers,
    # and a Python right-hand side can only ever see Float64.
    ALGORITHMS = {
        "bdf": "CVODE_BDF()",
        "bdf_krylov": "CVODE_BDF(linear_solver=:GMRES)",
        "bdf_fgmres": "CVODE_BDF(linear_solver=:FGMRES)",
        "adams": "CVODE_Adams()",
        "fbdf": "FBDF(autodiff=AutoFiniteDiff())",
        "rodas5p": "Rodas5P(autodiff=AutoFiniteDiff())",
        "tsit5": "Tsit5()",
    }

    # Mass-matrix problems need an algorithm that accepts one; CVODE_BDF
    # does not, so a DAE falls back to a pure-Julia stiff solver.
    supports_algebraic = True

    ALGEBRAIC_ALGORITHM = "rodas5p"

    # Neutral linear-solver request -> algorithm key
    LINEAR_SOLVER_ALGORITHMS = {
        "krylov": "bdf_krylov",
        "dense": "bdf",
        "sparse": "bdf",
    }

    def __init__(self, options=None, algorithm=None):

        super().__init__()

        self.options = dict(options) if options else {}
        self.algorithm = algorithm

        self._de = None
        self._rhs = None
        self._solution = None
        self._vector_type = None
        self._mass_matrix = None
        self._callbacks = None
        self._diagonal_type = None

        self.eval_sens = False
        self.jac_v_prod = False

    # ------------------------------------------------------------------
    # Julia interop
    # ------------------------------------------------------------------

    @staticmethod
    def find_julia():
        """
        Put Julia on PATH if it is installed but not visible.

        diffeqpy locates Julia with shutil.which, and jill (which diffeqpy
        uses to install it) does not add its own bin directory to PATH. Left
        alone, diffeqpy concludes Julia is missing and tries to install a
        second copy, which then blocks on an interactive prompt.
        """

        import os
        import shutil

        if shutil.which("julia"):
            return True

        candidates = [
            os.path.join(os.path.expanduser("~"), ".local", "bin"),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "julias", "bin"
            ),
            os.path.join(os.path.expanduser("~"), "julias", "bin"),
        ]

        for directory in candidates:

            if not directory or not os.path.isdir(directory):
                continue

            if shutil.which("julia", path=directory):
                os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
                return True

        return False

    @property
    def de(self):
        """The DifferentialEquations namespace, started on first access."""

        if self._de is None:

            if not self.find_julia():
                raise RuntimeError(
                    "DiffeqpyBackend needs a Julia runtime and none was "
                    "found on PATH. Install one with "
                    "'python -c \"import diffeqpy; diffeqpy.install()\"', "
                    "or put an existing Julia's bin directory on PATH."
                )

            try:
                from diffeqpy import de
            except ImportError as error:
                raise ImportError(
                    "DiffeqpyBackend needs the diffeqpy package. Install it "
                    "with 'pip install diffeqpy', then run "
                    "'python -c \"import diffeqpy; diffeqpy.install()\"' "
                    "once to fetch DifferentialEquations.jl."
                ) from error

            self._de = de

        return self._de

    def julia_eval(self, expression):
        """Evaluate Julia source in the DifferentialEquations namespace."""

        return self.de.seval(expression)

    def make_algorithm(self):
        """
        Build the Julia algorithm for this solve.

        An algorithm supplied by the caller wins. Otherwise the unit's
        linear-solver request picks one, defaulting to the same Sundials BDF
        that AssimuloBackend uses.
        """

        algorithm = self.algorithm

        if algorithm is None:

            if self._mass_matrix is not None:
                # CVODE_BDF cannot take a mass matrix, so the linear-solver
                # hint does not apply here.
                algorithm = self.ALGEBRAIC_ALGORITHM
            else:
                algorithm = self.LINEAR_SOLVER_ALGORITHMS.get(
                    self.linear_solver,
                    "bdf",
                )

        # Anything that is not one of our keys is taken to be an
        # already-constructed Julia algorithm and passed straight through.
        if not isinstance(algorithm, str):
            return algorithm

        try:
            expression = self.ALGORITHMS[algorithm]
        except KeyError:
            raise ValueError(
                f"Unknown algorithm {algorithm!r}. Choose one of "
                f"{sorted(self.ALGORITHMS)}, or pass a constructed Julia "
                "algorithm object."
            ) from None

        return self.julia_eval(expression)

    def translate_options(self):
        """Map the configured options onto SciML keyword arguments."""

        translated = {}

        for name, value in self.options.items():

            if name in self.IGNORED_OPTIONS:
                continue

            if name == "linear_solver":
                self.set_linear_solver(value)
                continue

            translated[self.OPTION_ALIASES.get(name, name)] = value

        # Assimulo's CVode defaults, so switching backends does not silently
        # change the accuracy a model was tuned against.
        translated.setdefault("abstol", 1e-6)
        translated.setdefault("reltol", 1e-6)

        return translated

    def make_rhs(self, unit):
        """
        Wrap the unit's model as an in-place SciML right-hand side.

        In-place rather than out-of-place for two reasons. SciML rejects an
        out-of-place Python right-hand side outright on every solver except
        Sundials ("non-constant types in an out-of-place ODE solve"), because
        it cannot infer the return type of a ``Py`` object. And writing
        through ``du`` copies the derivative out of the buffer that
        ``unit_model`` reuses on its next call, which an out-of-place form
        would have handed to the solver to keep.
        """

        def rhs(du, u, p, t):
            du[:] = unit.unit_model(
                time=float(t),
                states=np.asarray(u, dtype=float),
            )

        return rhs

    def to_julia_vector(self, values):
        """
        Copy a state vector into a Julia ``Vector{Float64}``.

        Sundials needs Julia-owned contiguous storage: handed a numpy array
        the problem's state type stays ``PyArray`` and CVODE cannot convert
        its own work vectors back into it.
        """

        if self._vector_type is None:
            self._vector_type = self.julia_eval("Vector{Float64}")

        return self._vector_type(np.asarray(values, dtype=float))

    def make_mass_matrix(self, unit):
        """
        Diagonal mass matrix marking which rows are constraints.

        ``M y' = f(y)`` with a 1 on every differential row and a 0 on every
        algebraic one turns the vector unit_model already returns into an
        index-1 DAE: the zero rows say "this entry of f is a residual to
        drive to zero", which is exactly the packing convention the vessel
        uses. Returns None for an ordinary ODE so nothing is allocated.
        """

        if not unit.has_algebraic_balance:
            return None

        mask = unit.solver_state_collection.algebraic_mask
        diagonal = np.where(mask, 0.0, 1.0)

        if self._diagonal_type is None:
            # Diagonal lives in LinearAlgebra, which is not in scope inside
            # diffeqpy's module, so it has to be imported explicitly. A
            # Diagonal rather than a dense matrix matters here: a
            # discretized phase can make this a few hundred rows square.
            self._diagonal_type = self.julia_eval(
                "import LinearAlgebra; LinearAlgebra.Diagonal"
            )

        return self._diagonal_type(self.to_julia_vector(diagonal))

    def make_callbacks(self, unit):
        """
        One ContinuousCallback per event, combined into a CallbackSet.

        VectorContinuousCallback would be the natural fit, but its event
        index arrives through PythonCall as raw bytes that cannot be
        converted to an int, so each event gets its own scalar callback
        instead and closes over its own index.
        """

        events = unit.compiled_events

        if not events:
            return None

        de = self.de
        callbacks = []

        for index, event in enumerate(events):

            def condition(u, t, integrator, index=index):
                return float(unit.evaluate_events(float(t), np.asarray(u))[index])

            def affect(integrator, index=index):
                if unit.handle_event(float(integrator.t), [index]):
                    de.seval("terminate!")(integrator)

            callbacks.append(de.ContinuousCallback(condition, affect))

        if len(callbacks) == 1:
            return callbacks[0]

        return de.CallbackSet(*callbacks)

    # ------------------------------------------------------------------
    # IntegratorBackend interface
    # ------------------------------------------------------------------

    def compile_integrator(
            self,
            unit,
            eval_sens=False,
            jac_v_prod=False,
            options=None,
            verbose=True,
            any_event=True,
    ):

        if eval_sens:
            raise NotImplementedError(
                "DiffeqpyBackend does not compute sensitivities. Use "
                "AssimuloBackend for parameter estimation."
            )

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod

        unit.reset()

        states_init = unit.create_solver_init_states()
        unit.save_initial_solver_state(states_init, unit.elapsed_time)

        if options:
            self.options.update(options)

        self._rhs = self.make_rhs(unit)
        self._mass_matrix = self.make_mass_matrix(unit)
        self._callbacks = self.make_callbacks(unit)

        # Gives the unit its chance to request a linear solver before the
        # algorithm is built, which happens per solve.
        unit.configure_solver()

        self._compiled = True

        return states_init

    def solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            eval_sens=False,
            jac_v_prod=False,
            verbose=True,
            options=None,
            any_event=True,
    ):

        if (
            not self._compiled
            or eval_sens != self.eval_sens
            or jac_v_prod != self.jac_v_prod
        ):

            states_init = self.compile_integrator(
                unit,
                eval_sens,
                jac_v_prod,
                options,
                verbose,
                any_event,
            )

            unit.derivatives = unit.unit_model(
                time=unit.elapsed_time,
                states=states_init,
            )

        time, states = self.fast_solve(
            unit,
            runtime=runtime,
            time_grid=time_grid,
            verbose=verbose,
        )

        unit.retrieve_results(time, states)

        return time, states

    def fast_solve(
            self,
            unit,
            runtime=None,
            time_grid=None,
            verbose=True,
    ):

        if not self._compiled:
            raise RuntimeError("Integrator has not been compiled.")

        de = self.de

        states_init = self.to_julia_vector(
            unit.create_solver_init_states()
        )

        start_time = float(unit.elapsed_time)

        if runtime is not None:
            final_time = start_time + runtime
        elif time_grid is not None:
            final_time = float(time_grid[-1])
        else:
            raise ValueError(
                "Either runtime or time_grid must be supplied."
            )

        if self._mass_matrix is None:
            function = self._rhs
        else:
            function = de.seval("ODEFunction")(
                self._rhs,
                mass_matrix=self._mass_matrix,
            )

        problem = de.ODEProblem(
            function,
            states_init,
            (start_time, float(final_time)),
        )

        kwargs = self.translate_options()

        if time_grid is not None:
            kwargs["saveat"] = np.asarray(time_grid, dtype=float)

        if self._callbacks is not None:
            kwargs["callback"] = self._callbacks

        solution = de.solve(
            problem,
            self.make_algorithm(),
            **kwargs,
        )

        self._solution = solution

        retcode = str(solution.retcode)

        if retcode not in ("Success", "Terminated"):
            raise RuntimeError(
                f"DifferentialEquations.jl returned retcode {retcode!r} "
                f"after {len(solution.t)} saved points. The last time "
                f"reached was {float(solution.t[-1]):g} of {final_time:g}."
            )

        time = np.asarray(solution.t, dtype=float)

        states = np.array(
            [np.asarray(state, dtype=float) for state in solution.u],
            dtype=float,
        )

        return time, states

    @property
    def statistics(self):
        """
        Solver counters, under the names AssimuloBackend reports.

        Only the counters SciML actually tracks appear, so a caller
        comparing backends sees a missing key rather than a fabricated zero.
        """

        if self._solution is None:
            return {}

        stats = getattr(self._solution, "stats", None)

        if stats is None:
            stats = getattr(self._solution, "destats", None)

        if stats is None:
            return {}

        mapping = {
            "nsteps": "naccept",
            "nfcns": "nf",
            "njacs": "njacs",
            "nerrfails": "nreject",
            "nniters": "nnonliniter",
            "nnfails": "nnonlinconvfail",
        }

        out = {}

        for ours, theirs in mapping.items():

            value = getattr(stats, theirs, None)

            if value is not None:
                out[ours] = int(value)

        return out


def _compile_ode_backend(backend, unit, eval_sens, jac_v_prod, name):
    """Initialize a sequential ODE backend without changing validation or reset order.

    Parameters
    ----------
    backend : IntegratorBackend
        ``FixedStepBackend`` being compiled.
    unit : MultiPhaseVessel
        Unit to integrate.
    eval_sens, jac_v_prod : bool
        Must be False; neither backend provides sensitivities.
    name : str
        Backend name for error messages.

    Returns
    -------
    ndarray
        Packed initial solver states [native state units].

    Raises
    ------
    NotImplementedError
        Sensitivities or Jacobian-vector products are requested.
    """
    if eval_sens or jac_v_prod:
        raise NotImplementedError(f"{name} does not expose sensitivities")
    backend.check_algebraic_support(unit)
    unit.reset()
    states = unit.create_solver_init_states()
    unit.save_initial_solver_state(states, unit.elapsed_time)
    backend._unit = unit
    backend._compiled = True
    return states


def _inventory_rates(
        time, states, *, constrained, boundary_tolerance, unit, rate_tolerance):
    """Evaluate rates with exhausted pools held at their boundary.

    Parameters
    ----------
    time : float
        Absolute time [s].
    states : ndarray
        Packed states.

    constrained : ndarray of int
        Indices of states requiring nonnegative inventories.
    boundary_tolerance : ndarray
        Per-state depletion threshold [corresponding state units].
    unit : MultiPhaseVessel
        Native vessel supplying rates; its work buffer may be reused.
    rate_tolerance : ndarray
        Per-state stationary-rate threshold [state units/s].

    Returns
    -------
    ndarray
        Owned rates [state units/s], independent of subsequent vessel evaluations;
        only boundary rates within the stationary-rate error budget are zeroed.

    Raises
    ------
    RuntimeError
        The model consumes an empty inventory beyond its rate error budget.
    """
    # Evaluate the boundary within the requested absolute error budget.
    # Otherwise roundoff supply/uptake imbalance repeatedly releases an
    # empty pool and exposes the discontinuous unconstrained kinetics.
    boundary = constrained[states[constrained] <= .5 * boundary_tolerance[constrained]]
    evaluation_state = states.copy()
    evaluation_state[boundary] = 0.0
    # SciPy retains derivatives across calls; the native vessel reuses its buffer.
    values = unit.unit_model(time, evaluation_state, limiter_dt=0.0).copy()
    replenishing = boundary[(states[boundary] > 0.0)
                             & (values[boundary] > rate_tolerance[boundary])]
    if replenishing.size:
        # A real source may balance uptake below the boundary tolerance.
        # Snapping that positive pool to zero creates an artificial switch.
        evaluation_state[replenishing] = states[replenishing]
        values = unit.unit_model(time, evaluation_state, limiter_dt=0.0).copy()
    stationary = boundary[abs(values[boundary]) <= rate_tolerance[boundary]]
    values[stationary] = 0.0
    invalid = boundary[(evaluation_state[boundary] == 0.0)
                       & (values[boundary] < -rate_tolerance[boundary])]
    if invalid.size:
        raise RuntimeError(
            f"negative rate at empty inventory indices {invalid.tolist()} "
            f"at time {time:g} s; configure feasible coupled rates instead "
            "of suppressing consumption")
    return values


def _inventory_boundary(
        time, states, *, position, boundary_tolerance):
    """Measure one constrained state's distance above its boundary tolerance.

    Parameters
    ----------
    time : float
        Absolute time [s].
    states : ndarray
        Packed states.
    position : int
        Constrained state index, bound when the event is constructed.

    boundary_tolerance : ndarray
        Per-state depletion threshold [corresponding state units].

    Returns
    -------
    float
        ``states[position] - boundary_tolerance[position]`` [state units];
        a downward zero crossing ends the segment.
    """
    return states[position] - boundary_tolerance[position]


class FixedStepBackend(IntegratorBackend):
    """Deterministic sequential propagation on a user-supplied native time grid.

    Each step calls every mechanism's ``prepare_step(dt)``, evaluates the
    unit with ``limiter_dt = dt``, applies an explicit Euler update, accepts
    the completed state and updates the phases. Mechanisms such as
    ``RateReconciledCulture`` use ``prepare_step`` to propagate populations
    exactly over the step, so this is the numerical method those mechanisms
    declare, not a generic explicit integrator.
    """

    def __init__(self):
        """Create an uncompiled backend."""
        super().__init__()
        self._unit = None

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True):
        """Reset the unit and record its initial solver state.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Unit to propagate.
        eval_sens, jac_v_prod : bool, optional
            Not supported; must be False.
        options, verbose, any_event : object, optional
            Accepted for interface compatibility; unused.

        Returns
        -------
        ndarray
            Packed initial states.
        """
        return _compile_ode_backend(self, unit, eval_sens, jac_v_prod, "FixedStepBackend")

    def solve(self, unit, runtime=None, time_grid=None, eval_sens=False,
              jac_v_prod=False, verbose=True, options=None, any_event=True):
        """Propagate the unit and store native results.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Unit to propagate.
        runtime : float, optional
            Unused; a time grid is required.
        time_grid : array_like
            Strictly increasing absolute times [s].
        eval_sens, jac_v_prod : bool, optional
            Not supported.
        verbose, options, any_event : object, optional
            Unused.

        Returns
        -------
        tuple of ndarray
            ``(time, states)``.
        """
        if not self._compiled or self._unit is not unit:
            self.compile_integrator(unit, eval_sens, jac_v_prod, options,
                                    verbose, any_event)
        time, states = self.fast_solve(unit, runtime, time_grid, verbose)
        unit.retrieve_results(time, states)
        return time, states

    def fast_solve(self, unit, runtime=None, time_grid=None, verbose=True):
        """Propagate on the supplied grid.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Compiled unit.
        runtime : float, optional
            Unused.
        time_grid : array_like
            Strictly increasing absolute times [s].
        verbose : bool, optional
            Unused.

        Returns
        -------
        tuple of ndarray
            ``(times, states)`` on the grid.

        Raises
        ------
        ValueError
            The grid is missing, has fewer than two points or is not increasing.
        RuntimeError
            A constrained state falls below the legacy -1e-12 cutoff
            [corresponding state units].

        Notes
        -----
        Values in [-1e-12, 0) are set to zero [corresponding state units].
        This retained absolute cutoff is not a proven roundoff bound.
        """
        if time_grid is None:
            raise ValueError("FixedStepBackend requires an explicit time_grid")
        times = np.asarray(time_grid, dtype=float)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("time_grid must be a strictly increasing vector")
        states = np.empty((len(times), unit.solver_state_collection.dim))
        states[0] = unit.create_solver_init_states()
        constrained = []
        for key, state_slice in unit.solver_state_collection.slices.items():
            if unit.solver_state_collection.states[key].limit_negative_inventory:
                constrained.extend(range(state_slice.start, state_slice.stop))
        for index, step in enumerate(np.diff(times)):
            for process in unit.intraphase_processes:
                prepare = getattr(process.mechanism, "prepare_step", None)
                if prepare is not None:
                    prepare(step)
            rates = unit.unit_model(times[index], states[index], limiter_dt=step)
            states[index + 1] = states[index] + step * rates
            if constrained and np.any(states[index + 1, constrained] < -1e-12):
                raise RuntimeError(
                    "fixed-step propagation violated a nonnegative inventory constraint"
                )
            states[index + 1, constrained] = np.maximum(
                states[index + 1, constrained], 0.0
            )
            # Fixed-step propagation accepts this state before the next closure.
            completed = unit.complete_state(
                unit.solver_state_collection.unpack(states[index + 1]),
                times[index + 1],
            )
            unit.update_phases_from_state(completed)
        return times, states


class _ScipyStateEvent:
    """
    One of the vessel's switching surfaces, in the shape solve_ivp wants.

    Carries two surfaces and swaps between them once per segment:

        armed     ->  g(t, y)
        disarmed  ->  g(t, y)**2 - band**2

    The disarmed surface is negative while ``|g|`` stays inside ``band`` and
    roots exactly when ``|g|`` leaves it, which is what makes restarting at a
    root safe. scipy's ``prepare_events`` reads ``terminal`` and ``direction``
    off the callable itself, so those are plain attributes.
    """

    def __init__(self, unit, index, direction, terminal):

        self.unit = unit
        self.index = index

        self.native_direction = float(direction)

        self.disarmed = False
        self.band = 0.0

        self.terminal = terminal
        self.direction = float(direction)

    def __call__(self, time, states):

        # evaluate_events memoizes on (time, states.tobytes()), so asking N
        # events about one point costs one complete_state, not N.
        value = float(self.unit.evaluate_events(time, states)[self.index])

        if self.disarmed:
            return value * value - self.band * self.band

        return value


def _controlled_difference_jacobian(rhs, relative_step, state_scale, time, states):
    """Forward-difference Jacobian with fixed relative state perturbations.

    Fixed perturbations avoid the adaptive probes of scipy's ``num_jac``,
    which can run away when the right-hand side carries solver-level noise,
    for example from an embedded optimization.

    Parameters
    ----------
    rhs : callable
        ``rhs(time, states)`` returning an owned rate array [state units/s].
    relative_step : float
        Relative perturbation [-], finite and positive.
    state_scale : ndarray
        Absolute floor for each perturbation [state units], finite and
        positive, aligned with ``states``.
    time : float
        Solver time [s].
    states : ndarray
        Packed solver states [state units].

    Returns
    -------
    ndarray, shape (n_states, n_states)
        ``d rates / d states`` [1/s]; column ``j`` uses the increment
        ``relative_step * max(|states[j]|, state_scale[j])``.
    """
    baseline = np.array(rhs(time, states), dtype=float)  # [state units/s]
    increments = relative_step * np.maximum(np.abs(states), state_scale)  # [state units]
    matrix = np.empty((len(states), len(states)))  # [1/s]
    for index, increment in enumerate(increments):
        perturbed = np.array(states, dtype=float)
        perturbed[index] += increment
        matrix[:, index] = (np.array(rhs(time, perturbed), dtype=float) - baseline) / increment
    return matrix


class ScipyBackend(IntegratorBackend):
    """
    scipy.integrate.solve_ivp backend.

    The one backend with nothing to install: scipy is already required by
    PharmaPy, whereas AssimuloBackend needs compiled SUNDIALS libraries and
    DiffeqpyBackend needs a Julia runtime. It covers the ODE reactors and
    crystallizers and is a drop-in for AssimuloBackend at any call site.

    ``method`` defaults to ``'BDF'``, the integrator AssimuloBackend uses
    (``iter='Newton'``, ``discr='BDF'``), so the two backends can be compared
    directly on the same model. Option names follow Assimulo's (``maxh``,
    ``atol``, ``rtol``) and are translated onto scipy's; genuine scipy
    keywords such as ``jac_sparsity`` pass through untouched.

    What it cannot do, and refuses rather than faking:

    - **Differential-algebraic systems.** solve_ivp has no DAE mode, so a unit
      declaring algebraic states is refused at compile time and pointed at
      AssimuloDAEBackend or DiffeqpyBackend.
    - **Sensitivities.** Parameter estimation still needs AssimuloBackend.
    - **A Krylov linear solver.** scipy's implicit methods always do a direct
      linear solve, so a crystallizer's ``set_linear_solver("krylov")`` is
      recorded and not acted on. On a large discretized phase that dense
      finite-difference Jacobian is the main cost against CVode with SPGMR;
      passing ``jac_sparsity`` through ``options`` is how to recover it.

    Events
    ------
    Every StateEvent in the refactored vessel is non-terminal: they exist so
    the integrator stops at the switching surface and *restarts* there rather
    than shortening steps to stumble across a kink. solve_ivp only records a
    non-terminal event, so this backend drives the restart itself, treating
    every event as terminal for one segment and relaunching from the root.

    Restarting exactly at a root would re-trigger it: scipy's
    ``find_active_events`` counts ``g <= 0 and g_new >= 0`` as a crossing, and
    at a root ``g`` is zero to within brentq's tolerance and lands on either
    side of it. So at each segment start an event whose value sits inside a
    band around zero is *disarmed* - replaced by a surface that roots when the
    value leaves the band - and re-armed once it has. The band is hysteretic
    (disarm below ``band/2``, re-arm at ``band``) so a re-arm root cannot fall
    straight back through the disarm threshold.

    The same rule handles a surface that sits identically at zero over an
    interval, such as a level controller holding its target: it stays disarmed
    and never fires until the level genuinely moves. A crossing costs two
    restarts rather than one, the second being the re-arm.

    Parameters
    ----------
    options : dict, optional
        Solver options, in either vocabulary. The optional pair
        ``jacobian_relative_step`` [-] and ``jacobian_state_scale`` [state
        units, scalar or per state] replaces scipy's adaptive
        finite-difference Jacobian with fixed perturbations
        ``step * max(|y|, scale)``. Both must be finite and positive and given
        together, and they cannot be combined with ``jac``. Omitting them
        leaves scipy's own Jacobian handling unchanged.
    method : str, optional
        A scipy solve_ivp method. Default ``'BDF'``.
    segment_events : bool, optional
        Stop and restart at each switching surface. Default True. Set False to
        hand the events to scipy as they are, which integrates straight
        through a non-terminal one.
    event_atol, event_rtol : float, optional
        Absolute and relative width of the disarm band. The relative part is
        scaled by the largest magnitude that event has reached, because event
        functions live on very different scales.
    max_restarts, max_short_segments, min_segment :
        Budgets that turn a degenerate switching surface into a diagnostic
        naming it, rather than a hang.
    inventory_boundaries : bool, optional
        Hold states flagged ``limit_negative_inventory`` at a depletion
        boundary instead of using the vessel's rate limiter. A terminal event
        stops the segment when such a pool reaches its boundary tolerance
        ``min(atol, rtol |y0|)`` (``atol`` when ``y0 = 0``); the pool is set
        to zero and integration restarts with uptake from it limited to
        actual supply. Depletion roots are restarts only: they are not
        reported to ``handle_event`` or added to the output. Constrained
        outputs below -1e-12 raise; values in [-1e-12, 0) are set to zero.
        Default False, which leaves every unit's behavior unchanged.
    """

    supports_algebraic = False

    METHODS = ("BDF", "LSODA", "Radau", "RK45", "RK23", "DOP853")

    # Assimulo option -> solve_ivp keyword. atol and rtol need no entry, scipy
    # spells them the same way.
    OPTION_ALIASES = {
        "maxh": "max_step",
        "inith": "first_step",
    }

    # Assimulo/Sundials controls with no scipy counterpart. scipy only warns
    # about a keyword it does not recognise, but that warning would fire on
    # every segment of every solve, so they are dropped here instead.
    IGNORED_OPTIONS = frozenset({
        "verbosity", "report_continuously", "time_limit", "clock_step",
        "iter", "discr", "maxord", "minh", "maxsteps", "num_threads",
        "usejac", "pbar", "sensmethod", "suppress_sens",
        "suppress_alg", "algvar", "make_consistent",
    })

    # Owned by fast_solve. Letting one through would produce "solve_ivp() got
    # multiple values for keyword argument", which says much less than this.
    RESERVED_OPTIONS = frozenset({
        "fun", "t_span", "y0", "method", "t_eval", "events", "dense_output",
        "args",
    })

    # Opt-in controlled Jacobian; handled by controlled_jacobian, never passed
    # to solve_ivp directly.
    JACOBIAN_OPTIONS = ("jacobian_relative_step", "jacobian_state_scale")

    def __init__(self, options=None, method="BDF", segment_events=True,
                 event_atol=1e-10, event_rtol=1e-6,
                 max_restarts=200, max_short_segments=3, min_segment=None,
                 stall_calls=200000, inventory_boundaries=False):

        """Store options and budgets; nothing is compiled until a unit is given.

        Parameters
        ----------
        options : dict, optional
            Solver options (see the class docstring). Copied, not shared.
        method : str, optional
            solve_ivp method. Default ``'BDF'``.
        segment_events : bool, optional
            Restart at each state-event root. Default True.
        event_atol, event_rtol : float, optional
            Absolute [event units] and relative [-] disarm-band widths.
            Defaults 1e-10 and 1e-6.
        max_restarts, max_short_segments : int, optional
            Restart and consecutive short-segment budgets. Defaults 200 and 3.
        min_segment : float, optional
            Shortest segment [s] not counted as short; default from the run span.
        stall_calls : int, optional
            Right-hand-side calls without progress before raising. Default 200000.
        inventory_boundaries : bool, optional
            Enable depletion boundaries for ``limit_negative_inventory`` states.
            Default False.

        Raises
        ------
        ValueError
            ``method`` is not a supported solve_ivp method.
        """
        super().__init__()

        if method not in self.METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose one of "
                f"{sorted(self.METHODS)}."
            )

        # Copied, not stored: AssimuloBackend's options={'maxh': 1} default is
        # one dict shared by every instance ever constructed.
        self.options = dict(options) if options else {}

        self.method = method
        self.segment_events = bool(segment_events)

        self.event_atol = float(event_atol)
        self.event_rtol = float(event_rtol)

        self.max_restarts = int(max_restarts)
        self.max_short_segments = int(max_short_segments)
        self.min_segment = min_segment
        self.stall_calls = int(stall_calls)
        self.inventory_boundaries = bool(inventory_boundaries)
        # Boundary-aware rate evaluator for the current solve, or None.
        self._inventory_rates = None

        self._progress_time = -np.inf
        self._progress_tol = 0.0
        self._stalled_calls = 0
        self._stall_final = None

        self._rhs = None
        self._events = []
        self._scale = np.zeros(0)

        self._counters = {}
        self._nsegments = 0
        self._nrestarts = 0
        self._event_log = []

        self.eval_sens = False
        self.jac_v_prod = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_linear_solver(self, kind):
        """
        Record the request; scipy has no linear solver to select.

        BDF and Radau always form a Jacobian and do a direct LU, LSODA does
        dense or banded, and the explicit methods do no linear algebra at all,
        so there is nothing to translate a "krylov" request onto. The base
        class calls this a performance hint rather than a correctness
        requirement, and that is how it is treated: the hint is kept and
        reported by ``statistics``, so it is visibly received rather than
        silently lost.

        Synthesising a ``jac_sparsity`` from the hint was rejected. Only the
        model knows its own sparsity pattern, and a guessed one that misses a
        structurally nonzero entry gives a wrong Jacobian and a wrong answer -
        too much to risk for a performance hint.
        """

        super().set_linear_solver(kind)

    def translate_options(self):
        """Map the configured options onto solve_ivp keyword arguments.

        Returns
        -------
        dict
            solve_ivp keywords: Assimulo names translated (``maxh`` ->
            ``max_step`` [s], ``inith`` -> ``first_step`` [s]), unsupported
            Assimulo controls and the controlled-Jacobian options dropped, and
            ``atol`` [state units] and ``rtol`` [-] defaulting to 1e-6 (CVode's
            defaults, so switching backends keeps a model's tuned accuracy).

        Raises
        ------
        ValueError
            A reserved solve_ivp keyword is supplied as an option.
        """

        translated = {}

        for name, value in self.options.items():

            if name in self.IGNORED_OPTIONS or name in self.JACOBIAN_OPTIONS:
                continue

            if name in self.RESERVED_OPTIONS:
                raise ValueError(
                    f"{name!r} is set by ScipyBackend itself and cannot be "
                    "passed as an option. Use time_grid instead of t_eval, "
                    "and the unit's state events instead of events."
                )

            if name == "linear_solver":
                self.set_linear_solver(value)
                continue

            translated[self.OPTION_ALIASES.get(name, name)] = value

        # Assimulo's CVode defaults, so switching backends does not silently
        # change the accuracy a model was tuned against.
        translated.setdefault("atol", 1e-6)
        translated.setdefault("rtol", 1e-6)

        return translated

    def controlled_jacobian(self, num_states):
        """Build the optional fixed-perturbation Jacobian for solve_ivp.

        Parameters
        ----------
        num_states : int
            Number of packed solver states.

        Returns
        -------
        callable or None
            ``jac(time, states)`` when both ``JACOBIAN_OPTIONS`` are set, else
            None so scipy keeps its default Jacobian handling.

        Raises
        ------
        ValueError
            Only one option is given, a value is nonfinite or not positive, the
            scale does not broadcast to the states, or ``jac`` is also set.
        """
        step = self.options.get("jacobian_relative_step")
        scale = self.options.get("jacobian_state_scale")
        if step is None and scale is None:
            return None
        if step is None or scale is None:
            raise ValueError(
                "jacobian_relative_step and jacobian_state_scale must be "
                "given together."
            )
        if "jac" in self.options:
            raise ValueError(
                "Choose either a supplied jac or the controlled Jacobian "
                "options, not both."
            )
        step = float(step)  # [-]
        try:
            scale = np.broadcast_to(
                np.asarray(scale, dtype=float), (num_states,)
            ).copy()  # [state units]
        except ValueError as error:
            raise ValueError(
                "jacobian_state_scale must be a scalar or one value per state."
            ) from error
        if (not np.isfinite(step) or step <= 0.0
                or not np.isfinite(scale).all() or np.any(scale <= 0.0)):
            raise ValueError(
                "jacobian_relative_step and jacobian_state_scale must be "
                "finite and positive."
            )
        return partial(_controlled_difference_jacobian, self._rhs, step, scale)

    def make_rhs(self, unit):
        """Wrap the unit's model as a solve_ivp right-hand side.

        The copy is not optional. unit_model refills and returns the same
        ``_solver_rate_buffer`` on every call, and scipy wraps a right-hand
        side in ``np.asarray(fun(t, y), dtype=float)``, which does not copy an
        array that is already float64 - so the solver would be left holding a
        buffer the next evaluation overwrites.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Compiled unit.

        Returns
        -------
        callable
            ``rhs(time, states)`` returning an owned rate array [state units/s].
            During a solve with ``inventory_boundaries`` it delegates to the
            depletion-boundary evaluator.

        Notes
        -----
        ``rhs`` is a pre-existing closure over the stall-detection state; it is
        kept local rather than restructured.
        """

        def rhs(time, states):

            """Evaluate rates, raising when the integration stops advancing.

            Parameters
            ----------
            time : float
                Solver time [s].
            states : ndarray
                Packed states [state units].

            Returns
            -------
            ndarray
                Owned rates [state units/s].

            Raises
            ------
            RuntimeError
                More than ``stall_calls`` evaluations occur without time advancing.
            """
            # scipy's implicit solvers do not fail when they cannot get past
            # a point: BDF clamps its step back up to min_step (about 1e-12
            # at t ~ 1e3) and keeps going, so a right-hand side it cannot
            # cross becomes an integration that runs for hours rather than
            # one that raises. Watching the furthest time reached catches
            # that, and unlike a plain call budget it will not fire on a
            # problem that is merely large, because such a problem still
            # advances.
            if time > self._progress_time + self._progress_tol:
                self._progress_time = time
                self._stalled_calls = 0
            else:
                self._stalled_calls += 1

                if self._stalled_calls > self.stall_calls:
                    reached = (
                        f"{self._progress_time:g}"
                        if np.isfinite(self._progress_time) else "the start"
                    )
                    of_final = (
                        f" of {self._stall_final:g}"
                        if self._stall_final is not None else ""
                    )
                    raise RuntimeError(
                        f"scipy {self.method} stalled at t={reached}"
                        f"{of_final}: {self._stalled_calls} right-hand side "
                        "evaluations without advancing, so the step size has "
                        "collapsed. That usually means the model is not "
                        "smooth enough there for an implicit solver to step "
                        "across, a kinetic regime change being the common "
                        "cause. Try another method, or use AssimuloBackend, "
                        "whose CVode can root-find the switch and restart on "
                        "it. Raise stall_calls if the model really is this "
                        "expensive per unit of time."
                    )

            if self._inventory_rates is not None:
                # Already an owned array; depletion boundaries replace the
                # vessel limiter for this solve.
                return self._inventory_rates(time, states)

            return np.array(
                unit.unit_model(time=time, states=states),
                dtype=float,
            )

        return rhs

    def make_events(self, unit):
        """Wrap each of the unit's StateEvents for solve_ivp."""

        events = unit.compiled_events

        if not events:
            return []

        return [
            _ScipyStateEvent(
                unit,
                index,
                event.direction,
                # In segmenting mode every event ends its segment, because
                # this backend performs the restart itself. Otherwise scipy's
                # own meaning of terminal applies.
                terminal=1 if self.segment_events
                else int(bool(event.terminal)),
            )
            for index, event in enumerate(events)
        ]

    # ------------------------------------------------------------------
    # Event bookkeeping
    # ------------------------------------------------------------------

    def event_bands(self):
        """Half-width of the dead band around zero, per event."""

        return self.event_atol + self.event_rtol * self._scale

    def arm_events(self, unit, time, states):
        """
        Choose each event's surface for the segment starting at (time, states).

        An event sitting inside half a band of zero is disarmed, so the
        segment cannot terminate on the root it just restarted from. The
        disarmed surface is given ``direction=1`` because only ``|g|`` growing
        out of the band is a boundary; falling back in is not.
        """

        if not self._events:
            return

        values = np.abs(
            np.asarray(unit.evaluate_events(time, states), dtype=float)
        )

        self._scale = np.maximum(self._scale, values)
        bands = self.event_bands()

        for index, event in enumerate(self._events):

            if values[index] < 0.5 * bands[index]:
                event.disarmed = True
                event.band = bands[index]
                event.direction = 1.0
            else:
                event.disarmed = False
                event.band = 0.0
                event.direction = event.native_direction

    def find_boundary(self, solution):
        """
        The root that ended this segment, and the state there.

        scipy's handle_events sorts the roots it found and truncates at the
        first terminating one, so the largest reported root is the one the
        segment stopped at. y_events carries the state there, which is why
        dense output is not needed.
        """

        index = None
        time = None

        for candidate, roots in enumerate(solution.t_events):

            if len(roots) == 0:
                continue

            root = float(roots[-1])

            if time is None or root > time:
                index, time = candidate, root

        if index is None:
            raise RuntimeError(
                "scipy reported an event stop but recorded no root."
            )

        states = np.asarray(solution.y_events[index][-1], dtype=float)

        return index, time, states

    def find_triggered(self, unit, index, time, states):
        """
        Every armed surface at zero here, not just the one scipy named.

        scipy reports the event it terminated on; Assimulo hands handle_event
        a flag per event and can report several at once. This recovers the
        rest, so a unit sees the same set under either backend.
        """

        triggered = [index]

        if len(self._events) > 1:

            values = np.abs(
                np.asarray(unit.evaluate_events(time, states), dtype=float)
            )
            bands = self.event_bands()

            for candidate, event in enumerate(self._events):

                if candidate == index or event.disarmed:
                    continue

                if values[candidate] <= bands[candidate]:
                    triggered.append(candidate)

        return sorted(triggered)

    def describe_event(self, unit, index):
        """Name an event the way the model declared it."""

        events = unit.compiled_events
        position = self._events[index].index

        if position >= len(events):
            return f"event {position}"

        event = events[position]

        return f"{event.name!r} (source {type(event.source).__name__})"

    # ------------------------------------------------------------------
    # Inventory depletion boundaries (opt-in)
    # ------------------------------------------------------------------

    def prepare_inventory_boundaries(self, unit, states_init, start_time,
                                     final_time, kwargs):
        """Set up depletion boundaries for one solve.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Compiled unit; states flagged ``limit_negative_inventory`` are
            constrained.
        states_init : ndarray
            Packed initial states [state units].
        start_time, final_time : float
            Solve interval [s].
        kwargs : dict
            Translated solve_ivp keywords supplying ``atol`` [state units]
            and ``rtol`` [-].

        Returns
        -------
        dict
            ``constrained`` (int indices), ``boundary_tolerance`` [state
            units], ``rate_tolerance`` [state units/s] and the mutable
            ``active`` set of exhausted indices used by depletion events.

        Raises
        ------
        ValueError
            A constrained initial state is negative.

        Notes
        -----
        The boundary tolerance is ``min(atol, rtol |y_ref|)`` with the unit's
        saved initial state as reference, or ``atol`` where that is zero.
        Rates within ``tolerance / (final_time - start_time)`` of zero at a
        boundary count as stationary, so roundoff imbalance between supply
        and uptake cannot repeatedly release an empty pool.
        """
        collection = unit.solver_state_collection
        constrained = np.asarray(
            [index for key, state_slice in collection.slices.items()
             if collection.states[key].limit_negative_inventory
             for index in range(state_slice.start, state_slice.stop)],
            dtype=int,
        )
        if constrained.size and np.any(states_init[constrained] < 0.0):
            raise ValueError("initial constrained inventories must be nonnegative")

        absolute_tolerance = np.broadcast_to(
            kwargs["atol"], states_init.shape)  # [state units]
        reference_state = np.abs(
            collection.pack(unit._initial_solver_state))  # [state units]
        boundary_tolerance = np.where(
            reference_state > 0.0,
            np.minimum(absolute_tolerance, kwargs["rtol"] * reference_state),
            absolute_tolerance,
        )  # [state units]
        rate_tolerance = boundary_tolerance / (final_time - start_time)  # [state units/s]
        active = {position for position in constrained
                  if states_init[position] == 0.0}

        self._inventory_rates = partial(
            _inventory_rates, constrained=constrained,
            boundary_tolerance=boundary_tolerance, unit=unit,
            rate_tolerance=rate_tolerance,
        )

        return {"constrained": constrained,
                "boundary_tolerance": boundary_tolerance,
                "rate_tolerance": rate_tolerance,
                "active": active}

    def depletion_events(self, boundaries, time, states):
        """Build terminal depletion events for the pools not yet exhausted.

        Parameters
        ----------
        boundaries : dict
            Result of ``prepare_inventory_boundaries``; its ``active`` set is
            updated in place.
        time : float
            Segment start time [s].
        states : ndarray
            Segment start states [state units].

        Returns
        -------
        tuple
            ``(positions, events)``: constrained indices that can still
            deplete, and one terminal, downward-crossing event per index.

        Notes
        -----
        An exhausted pool whose net rate at the segment start is positive
        (resupplied) is released, so it can deplete again later.
        """
        start_rates = self._inventory_rates(time, states)
        active = boundaries["active"]
        # Update in place: the same set object is bound into the evaluator.
        active.intersection_update(
            position for position in tuple(active) if start_rates[position] <= 0.0)
        positions = [position for position in boundaries["constrained"]
                     if position not in active]
        events = []
        for position in positions:
            event = partial(_inventory_boundary, position=position,
                            boundary_tolerance=boundaries["boundary_tolerance"])
            event.terminal = True
            event.direction = -1.0
            events.append(event)
        return positions, events

    # ------------------------------------------------------------------
    # IntegratorBackend interface
    # ------------------------------------------------------------------

    def compile_integrator(self, unit, eval_sens=False, jac_v_prod=False,
                           options=None, verbose=True, any_event=True):

        if eval_sens:
            raise NotImplementedError(
                "ScipyBackend does not compute sensitivities. Use "
                "AssimuloBackend for parameter estimation."
            )

        # Checked before unit.reset(), unlike AssimuloBackend, so a unit this
        # backend cannot solve is left exactly as the caller had it.
        self.check_algebraic_support(unit)

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod

        unit.reset()

        states_init = unit.create_solver_init_states()
        unit.save_initial_solver_state(states_init, unit.elapsed_time)

        if options:
            self.options.update(options)

        self._rhs = self.make_rhs(unit)
        self._events = self.make_events(unit)
        self._scale = np.zeros(len(self._events))

        # The unit's chance to request a linear solver before the first solve.
        unit.configure_solver()

        self._compiled = True

        return states_init

    def solve(self, unit, runtime=None, time_grid=None, eval_sens=False,
              jac_v_prod=False, verbose=True, options=None, any_event=True):

        """Compile if needed, integrate, and store native results.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Unit to integrate.
        runtime : float, optional
            Duration [s]; used when ``time_grid`` is None.
        time_grid : array_like, optional
            Absolute reporting times [s].
        eval_sens, jac_v_prod : bool, optional
            Must match the compiled settings; sensitivities are not supported.
        verbose : bool, optional
            Unused.
        options : dict, optional
            Options merged at compile time.
        any_event : bool, optional
            Unused.

        Returns
        -------
        tuple of ndarray
            ``(time, states)``: reported times [s] and packed states.

        Notes
        -----
        After compiling, ``unit.derivatives`` is set from the initial state. With
        ``inventory_boundaries`` it is evaluated without the vessel limiter, which
        that option replaces.
        """
        if (
            not self._compiled
            or eval_sens != self.eval_sens
            or jac_v_prod != self.jac_v_prod
        ):

            states_init = self.compile_integrator(
                unit,
                eval_sens,
                jac_v_prod,
                options,
                verbose,
                any_event,
            )

            limiter = {"limiter_dt": 0.0} if self.inventory_boundaries else {}
            unit.derivatives = np.array(
                unit.unit_model(time=unit.elapsed_time, states=states_init,
                                **limiter),
                dtype=float,
            )

        time, states = self.fast_solve(
            unit,
            runtime=runtime,
            time_grid=time_grid,
            verbose=verbose,
        )

        unit.retrieve_results(time, states)

        return time, states

    def fast_solve(self, unit, runtime=None, time_grid=None, verbose=True):

        """Integrate the compiled unit, restarting at each state-event root.

        Parameters
        ----------
        unit : MultiPhaseVessel
            Compiled unit.
        runtime : float, optional
            Duration [s] from the unit's elapsed time; used when ``time_grid`` is
            None.
        time_grid : array_like, optional
            Non-decreasing absolute reporting times [s]; points outside the run are
            dropped.
        verbose : bool, optional
            Unused.

        Returns
        -------
        tuple of ndarray
            ``(times, states)``: reported times [s], including event roots, and the
            packed states at those times.

        Raises
        ------
        RuntimeError
            The backend is not compiled, scipy fails, an event makes no progress,
            or the restart or short-segment budgets are exhausted.
        ValueError
            Neither ``runtime`` nor ``time_grid`` is given, the grid decreases, or
            the controlled-Jacobian options are invalid.
        """
        if not self._compiled:
            raise RuntimeError("Integrator has not been compiled.")

        states_init = np.asarray(
            unit.create_solver_init_states(), dtype=float
        )

        start_time = float(unit.elapsed_time)

        if runtime is not None:
            final_time = start_time + float(runtime)
        elif time_grid is not None:
            final_time = float(time_grid[-1])
        else:
            raise ValueError(
                "Either runtime or time_grid must be supplied."
            )

        grid = None

        if time_grid is not None:

            grid = np.asarray(time_grid, dtype=float).reshape(-1)

            if np.any(np.diff(grid) < 0):
                raise ValueError("time_grid must be non-decreasing.")

            # solve_ivp rejects a t_eval that leaves t_span, where Assimulo's
            # ncp_list simply ignores the excess. Clip rather than hand the
            # caller a scipy error about an argument they did not pass.
            grid = grid[(grid >= start_time) & (grid <= final_time)]

        kwargs = self.translate_options()

        boundaries = (
            self.prepare_inventory_boundaries(
                unit, states_init, start_time, final_time, kwargs)
            if self.inventory_boundaries else None
        )

        jacobian = self.controlled_jacobian(len(states_init))
        if jacobian is not None:
            kwargs["jac"] = jacobian

        min_segment = self.min_segment

        if min_segment is None:
            # brentq resolves a root to about 4*eps*|t|. Anything an order
            # below that is not a new boundary, it is the same one again.
            min_segment = max(
                16 * np.finfo(float).eps * max(1.0, abs(final_time)),
                1e-12 * (final_time - start_time),
            )

        self._counters = {}
        self._nsegments = 0
        self._nrestarts = 0
        self._event_log = []

        # Progress is measured against the span, so the stall detector means
        # the same thing whether the run covers 10 seconds or 10 hours.
        self._progress_time = -np.inf
        self._progress_tol = 1e-9 * max(final_time - start_time, 1.0)
        self._stalled_calls = 0
        self._stall_final = final_time

        times = []
        states = []

        # Assimulo's simulate() always reports t0. With t_eval=None scipy
        # seeds its own output with it; with t_eval it does not, and the
        # segment grids below are sliced strictly after their start, so this
        # is the only place t0 is emitted.
        if grid is not None:
            times.append(np.array([start_time]))
            states.append(states_init[None, :].copy())

        segment_time = start_time
        segment_states = states_init
        short_segments = 0

        while True:

            if self.segment_events:
                self.arm_events(unit, segment_time, segment_states)

            segment_grid = None

            if grid is not None:
                segment_grid = grid[
                    np.searchsorted(grid, segment_time, side="right"):
                ]

            events = list(self._events)
            depletion_positions = []

            if boundaries is not None:
                depletion_positions, depletion_events = self.depletion_events(
                    boundaries, segment_time, segment_states)
                events.extend(depletion_events)

            solution = solve_ivp(
                self._rhs,
                (segment_time, final_time),
                segment_states,
                method=self.method,
                t_eval=segment_grid,
                events=events or None,
                **kwargs,
            )

            self._nsegments += 1

            for name in ("nfev", "njev", "nlu"):
                self._counters[name] = (
                    self._counters.get(name, 0)
                    + int(getattr(solution, name, 0) or 0)
                )

            segment_times = np.asarray(solution.t, dtype=float).reshape(-1)

            # With t_eval set and nothing collected in it, scipy leaves t and
            # y as the list [], so y is (0,) rather than (0, n_states).
            if segment_times.size:
                times.append(segment_times)
                states.append(np.asarray(solution.y, dtype=float).T)

            if solution.status == -1:

                reached = (
                    segment_times[-1] if segment_times.size else segment_time
                )

                raise RuntimeError(
                    f"scipy {self.method} failed: {solution.message} The last "
                    f"time reached was {reached:g} of {final_time:g}."
                )

            if solution.status == 0:
                break

            index, event_time, event_states = self.find_boundary(solution)

            if index >= len(self._events):
                # A depleted pool: hold it at zero and restart. Not a unit
                # event, so the unit is not asked and nothing is reported.
                if event_time <= segment_time:
                    raise RuntimeError(
                        f"ScipyBackend made no progress at t={event_time:g}: "
                        "an inventory depletion boundary roots at the point "
                        "the segment started from."
                    )
                position = depletion_positions[index - len(self._events)]
                event_states = np.array(event_states, dtype=float)
                event_states[position] = 0.0
                boundaries["active"].add(position)
                self._event_log.append((event_time, position, "depletion", ()))
                segment_time = event_time
                segment_states = event_states
                continue

            was_rearm = self._events[index].disarmed

            if event_time <= segment_time:
                raise RuntimeError(
                    f"ScipyBackend made no progress at t={event_time:g}: "
                    f"state event {self.describe_event(unit, index)} roots at "
                    "the point the segment started from."
                )

            triggered = self.find_triggered(
                unit, index, event_time, event_states
            )

            # Assimulo reports the event point; so does this. The guard is for
            # a root that coincided with a grid point already emitted.
            if not times or times[-1][-1] != event_time:
                times.append(np.array([event_time]))
                states.append(event_states[None, :].copy())

            # A re-arm is bookkeeping, not a regime change, so the unit is not
            # asked about it.
            if not was_rearm:
                if unit.handle_event(event_time, triggered):
                    break

            self._nrestarts += 1
            self._event_log.append(
                (
                    event_time,
                    self._events[index].index,
                    "rearm" if was_rearm else "event",
                    tuple(self._events[i].index for i in triggered),
                )
            )

            if self._nrestarts > self.max_restarts:
                raise RuntimeError(
                    f"ScipyBackend restarted {self._nrestarts} times, the "
                    f"last at t={event_time:g} of {final_time:g} on state "
                    f"event {self.describe_event(unit, index)}. That is event "
                    "chatter rather than progress. Raise max_restarts if the "
                    "model really switches this often, raise event_rtol "
                    f"(currently {self.event_rtol:g}) to widen the re-arm "
                    "band, or pass ScipyBackend(segment_events=False) to "
                    "integrate straight through the switching surfaces."
                )

            # A re-arm segment is legitimately tiny and cannot chain, since
            # re-arming leaves |g| at the band and the next segment is armed.
            if not was_rearm:

                if event_time - segment_time < min_segment:

                    short_segments += 1

                    if short_segments > self.max_short_segments:
                        raise RuntimeError(
                            f"ScipyBackend stalled at t={event_time:g}: "
                            f"{short_segments} consecutive segments shorter "
                            f"than {min_segment:g}, all on state event "
                            f"{self.describe_event(unit, index)}. The surface "
                            "is not crossing transversally. Raise event_rtol "
                            f"(currently {self.event_rtol:g}), or pass "
                            "ScipyBackend(segment_events=False)."
                        )
                else:
                    short_segments = 0

            segment_time = event_time
            segment_states = event_states

        self._inventory_rates = None

        if not times:
            return (
                np.array([start_time]),
                states_init[None, :].copy(),
            )

        times, states = np.concatenate(times), np.vstack(states)

        if boundaries is not None and boundaries["constrained"].size:
            constrained = boundaries["constrained"]
            # Retained output policy of the depletion-boundary method: values
            # below this absolute cutoff are reported as a violation.
            if np.any(states[:, constrained] < -1e-12):
                raise RuntimeError(
                    "integration violated a nonnegative inventory constraint"
                )
            states[:, constrained] = np.maximum(states[:, constrained], 0.0)

        return times, states

    @property
    def statistics(self):
        """
        Solver counters, under the names AssimuloBackend reports.

        scipy's OdeResult carries only nfev, njev and nlu. The counters CVode
        prints and scipy does not track - nsteps, nerrfails, nniters, nnfails
        - are absent rather than zero, so a caller comparing backends sees a
        missing key instead of a fabricated number.

        nsegments and nrestarts are this backend's own rather than the
        solver's: they count how often the run stopped at a switching surface
        and started again, which is the cost of the event handling rather than
        of the integration.
        """

        if not self._counters and not self._nsegments:
            return {}

        mapping = {
            "nfcns": "nfev",
            "njacs": "njev",
            "nlus": "nlu",
        }

        out = {
            ours: int(self._counters[theirs])
            for ours, theirs in mapping.items()
            if theirs in self._counters
        }

        out["nsegments"] = self._nsegments
        out["nrestarts"] = self._nrestarts

        if self.linear_solver is not None:
            out["linear_solver_hint"] = self.linear_solver

        return out
