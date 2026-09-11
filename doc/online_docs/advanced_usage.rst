====================
Advanced features
====================

Liquid heat capacity
====================

:code:`LiquidPhase.getCp` now defaults to the mass basis [J/kg/K], like its siblings; callers needing [J/mol/K] must pass :code:`basis='mole'`.

Vapor density
=============

:code:`VaporPhase.getDensity` uses the ideal-gas equation at the requested temperature and pressure. The default mass basis returns [kg/m**3]; :code:`basis='mole'` returns [mol/L], equivalently [kmol/m**3]. Earlier releases returned [mol/m**3] regardless of the selected basis.

The positional arguments remain pressure [Pa], temperature [K], phase, and basis. The :code:`pres_gas` and :code:`temp_gas` keywords remain supported. New code can use :code:`pres` and :code:`temp` keywords, with optional :code:`mass_frac` or :code:`mole_frac` composition overrides. Do not supply both spellings of the same pressure or temperature argument.

Vapor concentrations use the same ideal-gas basis: :code:`mole_conc = mole_frac * pres / (R * temp) / 1000` [mol/L] and :code:`mass_conc = mole_conc * mw` [kg/m**3], with molecular weights [g/mol]. The last array axis is species; pressure and temperature profiles broadcast over preceding axes. Concentration inputs specify composition, while the equation of state fixes total concentration. A zero-amount vapor retains these intensive properties.

UNIQUAC data without :code:`qip` use :code:`qi` locally and emit a warning once per property object. This fallback assumes the ordinary surface parameter also describes the modified residual term; systems requiring special parameters, including relevant water/alcohol models, should provide :code:`qip` explicitly. The fallback does not create a :code:`qip` attribute, so callers can still detect missing data.

State events
============

State events are passed as Python dictionaries. Simple state events based directly on model states (i.e. :math:`f(y) = y`) can be passed as shown for the following structure. For detecting a given temperature, one would do:

.. testcode::

   my_state_event = {'state_name': 'temperature', 'value': 400} 

If a given state is not scalar but has to be indexed, e.g. concentration for a given component, the dictionary also needs to have the :code:`state_idx` field. For example, if the simulation is to be stopped when the  concentration the first component for a reactor model reaches a reference value, the dictionary describing the state event would be:

.. testcode::

   my_state_event = {'state_name': 'mole_conc', 'state_idx': 0, 'value': 0.2} 

For distributed models, :code:`state_idx` selects a component across all nodes. Use :code:`node_idx` to select particular nodes as well; omitting either index retains that axis. For example, monitoring component 1 at three nodes requires three conditions:

.. testcode::

   my_state_event = {'state_name': 'mole_conc', 'state_idx': 1,
                     'value': 0.2, 'num_conditions': 3}  # threshold [mol/L]

To monitor only the last of those nodes, add :code:`'node_idx': 2` and set :code:`'num_conditions': 1`. Multiple component or node indices can be supplied as lists. Conditions are flattened in node-major order within each definition, followed by the next definition in the event list. :code:`num_conditions` defaults to one and must equal the selected condition count, including for vector-valued callable events. Direction filters and event messages apply to the definition owning each condition.

More advanced usage of state events is allowed by passing a callable directly to PharmaPy. This callable will be able to make full use of all the instantaneous state and derivative information, which is passed by PharmaPy at each integration step. In this case, the function is passed using a dictionary with the keyword :code:`callable`:

.. testcode::

   my_state_event = {'callable': my_function}

where the passed callable function must have the signature :code:`my_function(time, states, sdot, **kwargs)` and must return a scalar whose sign changes only when the event is detected. The passed :code:`states` and :code:`sdot` arguments will be dictionaries that have the names of the states as keys. Reactor callbacks receive derivatives evaluated at the supplied event time and state, including intermediate states used to locate a crossing. Any keyword arguments can be optionally specified in the state event dictionary, e.g.:

.. testcode::

   my_state_event = {'callable': my_function, 'kwargs': {...}}

An example of a callable passed as a state event is when solubility wants to be monitored. A callable could have the following form:

.. testcode::

   def my_callable(time, y, ydot, a, b, c):
       solubility = a + b * y['temp'] + c * y['temp']**2  # a, b, and c are solubility constants
       event = solubility - y['mass_conc'][1]  # Let's say it is a binary system where the first component is the solvent and the second one is the API
       
       return event

In this case, the returned :code:`event` variable will be positive until the solubility limit is reached. When that happens, its sign change will be detected by PharmaPy and the integration will be interruped.

Reactor controls
================

Tank reactors accept a :code:`controls` dictionary mapping state names to callables :code:`f(time)` or records containing :code:`fun` and optional :code:`args` and :code:`kwargs`. The latter default to an empty tuple and dictionary. For example, these controls both prescribe a temperature initially at 320 K, increasing at 0.5 K/s:

.. testcode::

   controls = {'temp': lambda time: 320.0 + 0.5 * time}
   controls_record = {
       'temp': {'fun': lambda time, initial, rate: initial + rate * time,
                'args': (320.0,), 'kwargs': {'rate': 0.5}}}

Time is in seconds and the returned temperature is in kelvin. Tank controls receive scalar times during integration and result retrieval, so scalar Python functions and piecewise schedules are supported. A :code:`temp` control removes :code:`temp` and :code:`temp_ht` from the integrated tank states. Each return must be a finite scalar; singleton arrays are rejected. PFR rejects nonempty controls with :code:`NotImplementedError`, since it does not implement prescribed distributed states. Empty controls and :code:`None` remain supported. In bath mode, the Utility inlet temperature takes precedence over a :code:`temp_ht` control.

Tank heat rates use positive :code:`q_rxn` for reaction heat generation and positive :code:`q_ht` for utility heat added to the liquid, both in watts. This corrects the former isothermal Batch/CSTR sign conventions; consumers using those old signs must update their balances. Prescribed-temperature duty reconstructs the control derivative from the completed reporting interval, including a shortened interval after a terminating event. The nominal step is :code:`control_step_fraction` times the smallest reporting interval, bounded below by :code:`control_minimum_step` [s] and above by the smallest reporting interval and half the completed duration. Both settings are public :code:`solve_unit` arguments; their defaults are 1/1024 and 1/1024 s. Choose a smaller minimum step and a reporting grid that resolves fast controls. Second-order central and one-sided differences keep evaluations inside the completed interval. At least two finite, increasing reported times are required. Discontinuous controls are differentiated across jumps, so the apparent rate at a jump depends on the step.

For segmented solves, pass any custom :code:`control_step_fraction` and
:code:`control_minimum_step` values on every :code:`solve_unit` call. Omitting
either argument restores its default for that segment; settings are not inherited
from the previous solve. These arguments remain on :code:`solve_unit` because the
effective differentiation step also depends on that segment's reporting grid and
completed duration. Reusing the same values preserves the chosen settings, but
different grids or durations can still produce different effective steps.

:code:`heat_duty` sums each tank segment's trapezoidal integral in joules since the last reset; energy accuracy also depends on the reporting grid. Integrating segments separately preserves both one-sided heat rates when a feed or utility changes at their boundary. Batch, CSTR and Semibatch continuation offset a supplied relative time grid by elapsed time and retain the previous jacket state. :code:`reset_states=True` starts each solve from the original charge and clears the accumulated profiles and duties. A fresh CSTR/Semibatch jacket starts at the nominal :code:`Utility.temp_in`; a dynamic inlet temperature is subsequently evaluated in the utility balance. This may differ from a controller's value at time zero.

Each CSTR/Semibatch segment retains its sampled inlet concentration, temperature and flow. At a shared segment endpoint the earlier sample is retained, so replacing an inlet does not rewrite its historical flow profile. PFR duty describes its latest solve only. :code:`SimExec.GetDuties` copies each unit's stored energies into its heating/cooling columns without reconciling sign conventions or accumulation periods. Apply the model-specific conventions here, and accumulate per-run duties where needed before computing utility costs.

Batch and MSMPR crystallizers store :code:`heat_duty = [0, Q]` [J], filling the cooling column of :code:`SimExec.GetDuties`. Their :code:`heat_prof` and :code:`heat_duty` cover only the latest solve segment, even when result profiles include earlier segments. Positive duty means heat removed to the utility, opposite to the reactor heating column. Without a temperature control, these units integrate the jacket heat rate; with prescribed temperature, they reconstruct the utility rate from the energy balance and the temperature slope over each reporting interval. The prescribed-temperature Batch duty has changed sign relative to earlier releases. :code:`SemibatchCryst` does not publish these duty diagnostics.

Crystallizer jacket volume
==========================

Supply :code:`vol_ht` [m**3] from equipment data for a jacket model. When omitted, the model preserves a legacy jacket-volume assumption of 14% of its reference volume. Batch uses the current total slurry volume; MSMPR and Semibatch use :code:`vol_tank / vol_offset`. This is an equipment design assumption, not a correlation validated across vessel sizes. The factor is traceable to :code:`source/Crystallizers.py` in the repository's initial commit `e1e8164 <https://github.com/PharmaPy-org/PharmaPy/blob/e1e8164976e9103a396b24a7f7cfe5d1247c845c/source/Crystallizers.py>`_. Its original physical provenance remains the subject of `issue 113 <https://github.com/PharmaPy-org/PharmaPy/issues/113>`_. An explicit :code:`vol_ht` takes precedence and avoids relying on that assumption.

Crystallizer feeds and steady state
===================================

:code:`Slurry.Phases` rejects phase-only initialization with zero combined liquid and solid volume before normalizing moments or distributions. Supply a positive phase inventory first; a positive liquid inventory with zero crystals remains supported. Empty-slurry temperature initialization is not defined.

Moment-mode crystallizers accept static slurry moments and connected upstream moment profiles on the slurry-volume basis [m**n/m**3]. Connections from FVM crystallizers retain the reported moment history, rather than applying the final population at every time. Explicitly converted inlet moments retain precedence. The inlet must supply all moment orders required by the destination; extra higher orders are ignored, and missing orders raise an explanatory error.

Finite-radius FVM nucleation requires :code:`rad_zero` [um] to match the first size-grid point within floating-point roundoff. Legacy :code:`rad_zero=0` remains supported with positive-start grids. Moment-mode analytical Jacobians are restricted to prescribed-temperature Batch crystallization with zero-radius nuclei and :code:`mass_conc` kinetics. The built-in kinetic model and unit impurity factor assumptions also apply; unsupported operating modes, FVM, finite radii, and other concentration bases raise before solver construction.

Sensitivity :code:`sundials_opts` take precedence over the defaults :code:`sensmethod='SIMULTANEOUS'` and :code:`suppress_sens=False`. Continuous reporting remains required to collect CVODES sensitivities. Supported short nucleating cases are checked against complete-solve finite differences for numerical and analytical Jacobians. The longer shipped cooling-case benchmark remains an acceptance requirement of `issue 222 <https://github.com/PharmaPy-org/PharmaPy/issues/222>`_; these tests do not establish convergence for every kinetic regime.

:code:`MSMPR.solve_steady_state` initializes the kinetic target species itself; a prior dynamic solve is not required. It passes the complete species composition to kinetics in the selected basis, retains non-target species, and applies the same impurity growth factor as the dynamic model. Its constant-property, solid-free-feed, and growth assumptions still apply. Convergence diagnostics work with the declared SciPy 1.9 floor as well as the locked version. Numerical root convergence does not establish the full slurry physical balance: `issue 223 <https://github.com/PharmaPy-org/PharmaPy/issues/223>`_ still depends on the population/balance work tracked in `issue 300 <https://github.com/PharmaPy-org/PharmaPy/issues/300>`_. Crystallizer reset restores independent copies of the original phase inventories and rebuilds the slurry population and volume. Parameter-estimation phase modifiers also refresh those slurry quantities before the next solve. Relative supersaturation is :math:`S-1`, where :math:`S=c/c_{sat}`; callers relying on the former extra normalization must update their kinetic interpretation.

Evaporator heat duties
======================

Evaporator :code:`heat_profile` columns are powers [J/s], and :code:`heat_duty` contains their cumulative trapezoidal integrals [J] across segments since the last reset or public :code:`Phases` assignment. For batch evaporators, column 0 is heat into the drum (positive for heating) and column 1 is condensation heat (negative for cooling); the cumulative energies retain their signs. For continuous evaporators, column 0 is jacket/utility duty (positive for heat removed) and column 1 is condenser duty (negative for cooling), evaluated on the vapor-composition basis. This replaces the former liquid-composition basis, so existing zero-reflux runs also report a different column-1 value. Continuous duties accumulate signed energy first and then report its magnitude; each physical duty is counted once. Partition independence holds for a fixed trajectory: continuous :code:`solve_unit` restarts at time zero, so splitting solves need not reproduce the same trajectory.

For nitrogen-enabled semibatch evaporation, the original inlet remains attached throughout each solve. Feed controllers and connected profiles use the original condensable species order; the evaporator appends a zero nitrogen fraction after evaluating the feed. Changes made through the inlet's :code:`updatePhase` method take effect on the next segment.

Evaporator solver boundaries
============================

Flash solves verify convergence, finite scaled residuals, and phase-fraction bounds before publishing outlet phases. Tiny bound violations within the residual tolerance are clipped and the residual is rechecked; invalid or unconverged states raise :code:`RuntimeError`. :code:`residual_tolerance` is configurable and defaults to the square root of machine epsilon.

A batch evaporator using :code:`stop_at_maxvol=True` rejects an initial or reused charge at the 95% liquid-volume cap, including the event's roundoff neighborhood. Drain or reset before restarting a cap-limited run. With :code:`stop_at_maxvol=False`, the existing mode stops the feed at the cap and continues evaporation.

Continuous steady evaporation accepts :code:`fsolve_opts` for state scaling and solver controls. There is no universal scaling or trust-region factor: the regression suite covers both zero reflux and active reflux with fixture-specific :code:`diag`, :code:`xtol`, and :code:`factor`. Property failures during a trial preserve their original exception as the cause and identify thermophysical data and :code:`fsolve_opts` as diagnostic inputs. Successful solver termination should still be checked against physically meaningful balance residuals.

Solid distributions and filtration
===================================

:code:`SolidPhase.distrib_type` accepts :code:`'vol_perc'` or :code:`'mass_frac'`. The formerly tolerated :code:`'mass_perc'` spelling now raises an error. The selected basis normalizes bin weights before conversion to a number distribution. Porosity normalizes positive represented solid volume without adding a dimensional epsilon; a distribution with no represented volume raises. The zero-ended-grid quadrature contract remains provisional under `issue 269 <https://github.com/PharmaPy-org/PharmaPy/issues/269>`_.

Filter requires a resolved size distribution consistent with the current solid moments and mass. A moment-grown crystallizer can retain an old size distribution on its :code:`SolidPhase`; that handoff is rejected. Use a validated reconstruction or a compatible FVM population. Moment data alone do not uniquely determine a size distribution.

Filter pressure must be finite and positive; physical specific cake resistance must be positive and medium resistance nonnegative. Log parameters must exponentiate to finite values in this domain. These conditions are checked before optional solver construction. A positive-pressure run that finishes filtration retains its physical plateau. Repeated fresh portions use an absolute elapsed clock. Direct :code:`Filter.retrieve_results` calls must supply :code:`mass_solids` [kg] for the solved portion; it is no longer inferred from mutable phase inventory.

Washing inventory
=================

Displacement washing uses the attached cake's packing porosity for flow, adsorption, retained liquid, and effluent balances. The washing-unit diameter sets cross-sectional area and cake height; it does not imply repacking the cake at a different porosity. The dispersion correlation now implements the thick-cake branch for heights above 0.10 m: for :math:`Re\,Sc>1`, its convective contribution is :math:`1.75 Re\,Sc`, compared with :math:`55.5 (Re\,Sc)^{0.96}` at or below 0.10 m. The stagnant contribution remains :math:`1/\sqrt{2}`. Particle sizes are converted from micrometers to meters before evaluation. See :code:`DisplacementWashing.get_diffusivity` for the source equations and applicability assumptions.

Deliquoring checks inferred removal against initial and retained species inventories. Only arithmetic-scale negative roundoff is clipped. A larger negative removal emits a warning, preserves signed :code:`liquid_removed_species` and removal history, sets :code:`removal_diagnostics_valid=False`, and returns NaN removal composition. The transport-conservation defect tracked in `issue 29 <https://github.com/PharmaPy-org/PharmaPy/issues/29>`_ is not repaired by those diagnostics; invalid removal data must not be used as a physical effluent stream.


Interpolators
===============

Input trajectories can be specified via interpolators. To this purpose, PharmaPy contains a Lagrange polynomial represention that describes a time-dependent input :math:`u(t)` as:

.. math::

   u(t) = \sum_{i = 1}^{ord} u_{i, k} \ell_i^{(ord)} (\tau^{(k)}), \quad t \in [t_{k - 1}, t_k], \ k \in \{1, \ldots, n_{interv}\},

for a set of user-provided points :math:`u_{i, k}, \ i \in \{1, \ldots, ord\}` within the interval :math:`k`.  :math:`\tau` is a normalized time variable given by:

.. math::
   \tau = \frac{t - t_{k - 1}}{t_{k} - t_{k - 1}}, \quad t \in [t_{k - 1}, t_k]

and :math:`\ell^{(ord)}` are Lagrange interpolation polynomials given by:

.. math::
   \ell_{i}^{(ord)} =
   \begin{cases}
       1, & ord = 1, \\
       \prod_{j = 1, j \neq i}^{ord} \frac{\tau - \tau_j}{\tau_i - \tau_j}  & ord \geq 2,
   \end{cases}

where :math:`ord` represents the order of the interpolation (1 for piecewise constant, 2 for piecewise linear, etc).

For example, if a given flowrate wants to be specified as a piecewise constant function, a PharmaPy interpolator can be specified as: 

.. testcode::

   from PharmaPy.Interpolation import PiecewiseLagrange

   time_hor = 3600  # total time [s]
   flowrates = [0.1, 0.2, 0.1, 0.6]  # kg/s
   interpolator = PiecewiseLagrange(time_hor, flowrates, order=1)

In this particular case, the horizon time will be split into equally sized, 15-min (900 s) bins with their corresponding four specified flows as specified in the :code:`flowrates` variable. User-defined time marks can also be passed as a list or NumPy array by using the :code:`time_k` argument of the :code:`PiecewiseLagrange` interpolator, which needs to be of size :code:`n_y + 1`, where :code:`n_y` is the vector of interpolated values (:code:`flowrates` in this example).

Piecewise linear interpolators can also be used. In this case, the passed known values must be arranged into a numpy 2-D array, and the interpolation order will be 2. For example, a linear piecewise temperature profile would be constructed as:

.. testcode::

   from PharmaPy.Interpolation import PiecewiseLagrange

   time_hor = 3600  # total time [s]
   temperatures = np.array([[360, 345],
                            [345, 330],
                            [330, 318],
                            [318, 295]])  # K
   interpolator = PiecewiseLagrange(time_hor, temperatures, order=2)

Note that the values on the second column always match the value of the first column in the next raw, for continuity purposes. Higher orders will follow the same structure, where each row will represent a subinterval and the number of columns will dictate the interpolation order, which must be passed using the :code:`order` argument.

