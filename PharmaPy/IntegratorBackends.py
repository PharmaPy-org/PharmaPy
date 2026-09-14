import numpy as np
from abc import ABC, abstractmethod
from PharmaPy._assimulo import CVode, Explicit_Problem
from PharmaPy.Commons import eval_state_events
from PharmaPy.DataClasses import *

class IntegratorBackend(ABC):

    def __init__(self):
        self._compiled = False

        # Neutral linear-solver request, set by the unit operation.
        self.linear_solver = None

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
        
        self.set_ode_problem(unit,states_init)

        if unit.state_event_list:

            def new_handle(solver, info):
                return handle_events(
                    solver,
                    info,
                    unit.state_event_list,
                    any_event=any_event
                )

            self._problem.state_events = unit._eval_state_events
            self._problem.handle_event = new_handle

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

        if unit.state_event_list:

            sw0 = [True] * len(unit.state_event_list)

            def model(time, states, sw=None):
                return unit.unit_model(
                    time=time,
                    states=states,
                    sw=sw
                )

            problem = Explicit_Problem(
                model,
                states_init,
                t0=unit.elapsed_time,
                sw0=sw0
            )

            def new_handle(solver, info):
                return handle_events(
                    solver,
                    info,
                    unit.state_event_list,
                    any_event=True
                )

            problem.state_events = unit._eval_state_events
            problem.handle_event = new_handle

        else:

            def model(time, states):
                rhs = unit.unit_model(time=time, states=states)

                if len(rhs) != len(states):
                    print("RHS mismatch!")
                    print(len(states), len(rhs))
                    raise RuntimeError

                return rhs

            problem = Explicit_Problem(
                model,
                states_init,
                t0=unit.elapsed_time
            )

        self._problem = problem

    def _eval_state_events(self, time, states, sw):
        # TODO reactor version changes discretized_model to True if PFR (cobc in our case)
        events = eval_state_events(
            time, states, sw, self.len_states,
            self._solver_states, self.state_event_list, sdot=self.derivatives,
            discretized_model=False)

        return events

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

        if unit.state_event_list:
            raise NotImplementedError(
                "DiffeqpyBackend does not handle state events yet. They "
                "would map onto SciML callbacks, but the switch-handling "
                "protocol has no equivalent here, so the events would be "
                "quietly ignored rather than honoured."
            )

        self.eval_sens = eval_sens
        self.jac_v_prod = jac_v_prod

        unit.reset()

        states_init = unit.create_solver_init_states()
        unit.save_initial_solver_state(states_init, unit.elapsed_time)

        if options:
            self.options.update(options)

        self._rhs = self.make_rhs(unit)

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

        problem = de.ODEProblem(
            self._rhs,
            states_init,
            (start_time, float(final_time)),
        )

        kwargs = self.translate_options()

        if time_grid is not None:
            kwargs["saveat"] = np.asarray(time_grid, dtype=float)

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
