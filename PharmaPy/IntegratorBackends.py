import numpy as np
from abc import ABC, abstractmethod
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

    def __init__(self, options={'maxh': 1}):

        super().__init__()

        self._problem = None
        self._solver = None
        self.options = options

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
        """
        Consistent yd0: f(y0) on differential rows, zero on algebraic ones.

        IDA only needs the differential entries to be right; the algebraic
        entries of yd never enter the residual.
        """

        algebraic = unit.solver_state_collection.algebraic_mask

        values = np.asarray(
            unit.unit_model(time=unit.elapsed_time, states=states_init),
            dtype=float,
        )

        return np.where(algebraic, 0.0, values)

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
                "AssimuloDAEBackend does not compute sensitivities."
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

        for source in (options, self.options):
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
