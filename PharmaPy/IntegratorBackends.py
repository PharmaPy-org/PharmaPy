from abc import ABC, abstractmethod
from PharmaPy._assimulo import CVode, Explicit_Problem
from PharmaPy.Commons import eval_state_events
from PharmaPy.DataClasses import *

class IntegratorBackend(ABC):

    def __init__(self):
        self._compiled = False

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

    def __init__(self,options={'maxh':1} ):

        super().__init__()

        self._problem = None
        self._solver = None
        self.options = options
        self.state_event_list = []

        self.eval_sens = False
        self.jac_v_prod = False

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
    