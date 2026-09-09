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

Time is in seconds and the returned temperature is in kelvin. Tank controls receive scalar times during integration and result retrieval, so scalar Python functions and piecewise schedules are supported. A :code:`temp` control removes :code:`temp` and :code:`temp_ht` from the integrated tank states. PFR accepts these control forms but does not yet apply them to its integrated states. In bath mode, the Utility inlet temperature takes precedence over a :code:`temp_ht` control.

Tank heat rates use positive :code:`q_rxn` for reaction heat generation and positive :code:`q_ht` for utility heat added to the liquid, both in watts. Prescribed-temperature duty uses the control callable's temperature derivative in the energy balance. The nominal differentiation step is 1/1024 of the requested run duration, with a minimum of 1/1024 s. For a positive-duration run, the step is capped at half the run duration. Central differences are used in the interior, with second-order forward or backward differences near the boundaries; all evaluation points remain inside the run. A zero-duration run uses a forward difference and requires evaluation beyond its single time. Discontinuous controls are differentiated across their jumps, so the apparent rate at a jump depends on the differentiation step.

A single reported time still uses the requested run duration for differentiation. Direct result retrieval without a preceding solve uses the supplied profile's start and span, with the same minimum step. :code:`heat_duty` sums the trapezoidal integral of each tank segment in joules since the last reset; its accuracy also depends on the reporting grid. Integrating segments separately preserves both one-sided heat rates when a feed or utility changes at their boundary. Each CSTR/Semibatch segment retains its sampled inlet concentration, temperature and flow. At a shared segment endpoint the earlier sample is retained, so replacing an inlet does not rewrite its historical flow profile.

Batch and MSMPR crystallizers store :code:`heat_duty = [0, Q]` [J], filling the cooling column of :code:`SimExec.GetDuties`. Their :code:`heat_prof` and :code:`heat_duty` cover only the latest solve segment, even when result profiles include earlier segments. Positive duty means heat removed to the utility, opposite to the reactor heating column. Without a temperature control, these units integrate the jacket heat rate; with prescribed temperature, they reconstruct the utility rate from the energy balance and the temperature slope over each reporting interval. The prescribed-temperature Batch duty has changed sign relative to earlier releases. :code:`SemibatchCryst` does not publish these duty diagnostics.

Crystallizer feeds and steady state
===================================

Moment-mode crystallizers accept static slurry moments and connected upstream moment profiles on the slurry-volume basis [m**n/m**3]. Connections from FVM crystallizers retain the reported moment history, rather than applying the final population at every time. Explicitly converted inlet moments retain precedence. The inlet must supply all moment orders required by the destination; extra higher orders are ignored, and missing orders raise an explanatory error.

:code:`MSMPR.solve_steady_state` initializes the kinetic target species itself; a prior dynamic solve is not required. Its documented constant-property, solid-free-feed, and growth assumptions still apply. Crystallizer reset restores the original phase inventories and rebuilds the slurry population and volume. Parameter-estimation phase modifiers also refresh those slurry quantities before the next solve.

Evaporator heat duties
======================

Evaporator :code:`heat_profile` columns are powers [J/s], and :code:`heat_duty` contains their cumulative trapezoidal integrals [J] across segments since the last reset or public :code:`Phases` assignment. For batch evaporators, column 0 is heat into the drum (positive for heating) and column 1 is condensation heat (negative for cooling); the cumulative energies retain their signs. For continuous evaporators, column 0 is jacket/utility duty (positive for heat removed) and column 1 is condenser duty (negative for cooling), evaluated on the vapor-composition basis. This replaces the former liquid-composition basis, so existing zero-reflux runs also report a different column-1 value. Continuous duties accumulate signed energy first and then report its magnitude; each physical duty is counted once. Partition independence holds for a fixed trajectory: continuous :code:`solve_unit` restarts at time zero, so splitting solves need not reproduce the same trajectory.

For nitrogen-enabled semibatch evaporation, the original inlet remains attached throughout each solve. Feed controllers and connected profiles use the original condensable species order; the evaporator appends a zero nitrogen fraction after evaluating the feed. Changes made through the inlet's :code:`updatePhase` method take effect on the next segment.

Washing inventory
=================

Displacement washing uses the attached cake's packing porosity for flow, adsorption, retained liquid, and effluent balances. The washing-unit diameter sets cross-sectional area and cake height; it does not imply repacking the cake at a different porosity.


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

