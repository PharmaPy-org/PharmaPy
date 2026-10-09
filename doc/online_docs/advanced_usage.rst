====================
Advanced features
====================

Parameter-estimation outputs
============================

After :code:`ParameterEstimation.optimize_fn(method='LM')` or
:code:`SimulationExec.EstimateParams(method='LM')`, reported predictions and
residuals describe the accepted parameters in :code:`params_convg`, including
when the solver stops after rejecting its last trial. The estimator evaluates
the model once more per experiment to refresh its stored outputs. This also
restores a stateful unit-operation callback to those parameters; the unit retains
the last experiment's trajectory when several experiments share one unit.
This reporting step does not establish convergence. Inspect the solver's
termination diagnostics before interpreting the fit.

:code:`y_model`, :code:`y_runs`, and :code:`resid_runs` retain experiment order,
measured-state order, and the model's state units (for example [mol/L]).
:code:`weighted_residuals` applies the observation weighting once and follows
experiment, state, then sample order. Repeated fits replace :code:`y_model`
instead of appending earlier fits. The accepted LM :code:`info['x']`,
:code:`info['fun']`, :code:`info['jac']`, and solver counters are unchanged by
the extra reporting evaluation. Objective-history recording follows the usual
callback behavior, including duplicate removal with :code:`store_iter=True`.

:code:`weight_matrix` is the measurement-error covariance of the measured
states. Residuals are weighted by the lower Cholesky factor of its inverse (the
precision) in measured-state order. Earlier versions permuted the precision
whenever its LDL factorization pivoted, which some correlated covariances
require; such fits now give different estimates and covariances. Diagonal
covariances, and correlated ones that needed no pivoting, are unchanged.
:code:`weight_matrix` must be a finite, square, positive-definite covariance.
Indefinite matrices, which earlier versions accepted with NaN weights, now
raise :code:`numpy.linalg.LinAlgError` at construction; non-square or
non-finite ones raise :code:`ValueError`.

With staggered observation grids (per-state sampling times in :code:`x_data`),
unobserved model-grid entries now contribute zero residual and zero
sensitivity, whether the Jacobian comes from :code:`jac_fun`, from
sensitivities returned by the model, or from finite differences. Earlier
versions kept sensitivities at unobserved entries, so their parameter
covariances and confidence intervals overstated the information in the data.
:code:`info['fun']` and :code:`info['jac']` keep their model-grid layout, with
those entries equal to zero. Under a correlated :code:`weight_matrix`, a
sample row that observes only some states is weighted by the marginal precision
of those states, the inverse of their covariance block, rather than the
corresponding block of the full precision. The residual bootstrap
(:code:`StatisticsClass.get_bootsamples` and :code:`bootstrap_params`) draws
each state's errors only from its observed residuals and leaves unobserved
entries of the generated datasets as NaN. Without staggered grids and without
declared measurement uncertainties the draws are unchanged. When measurements
declare standard deviations (``Measurement`` uncertainties), the standardized
residuals are resampled and rescaled by the standard deviation of each
generated entry, so the generated errors follow the empirical residual scatter
rather than the declared absolute standard deviations; see
:doc:`examples/experiment_measurement`, which also describes residual packing,
uncertainty scaling and the two covariance forms of
:code:`get_covariance`.

:code:`MultipleCurveResolution` applies the same rules to non-spectral states
measured at fewer times than the spectra: their unobserved entries have zero
weighted residual and zero sensitivity for model-returned and
finite-difference Jacobians, and partially observed rows use the marginal
precision. Its :code:`weight_matrix` has one row and column per spectral
channel followed by one per non-spectral state and is the covariance of the
measurement errors. Spectral residuals are data minus prediction while
non-spectral residuals are model minus data, so correlations between spectral
and non-spectral columns are applied with the sign this implies. Its
finite-difference Jacobian previously had the opposite sign of the residual
derivative (so Levenberg-Marquardt steps were rejected), and its model-returned
Jacobian had that sign for non-spectral states; both now equal the derivative
of the weighted residuals. Spectra must still be recorded at every model-grid
time of an experiment.

:code:`StatisticsClass.bootstrap_params` returns the bootstrap estimates, one row
per generated dataset, and stores them in :code:`boot_params`. Each dataset is
fitted on an isolated copy of the estimator. The estimator's observations,
accepted parameters, residuals, predictions, solver information, covariance and
solver options are therefore unchanged afterwards, including when a sample fit
fails. Earlier versions left the last bootstrap dataset in :code:`y_data` and
the last refit in the fitted outputs. Afterwards the model is evaluated once
more at :code:`params_convg`, on a copy of the estimator. This returns a
stateful model callback, such as a :code:`SimulationExec` unit operation, to
the accepted parameters, also when a sample's exception propagates. A model
that can no longer be evaluated at the accepted parameters is an error: if
that evaluation fails after all samples were processed, its error is raised.
If a sample's exception is already propagating, that exception is raised and
the restore failure is reported as a :code:`RuntimeWarning`, which is dropped
when warnings are turned into errors. A keyboard interrupt aborts without the
extra evaluation.

Liquid heat capacity
====================

:code:`LiquidPhase.getCp` now defaults to the mass basis [J/kg/K], like its siblings; callers needing [J/mol/K] must pass :code:`basis='mole'`.

Dynamic extraction inlet temperatures
=====================================

``DynamicExtractor`` pairs each inlet's composition and temperature [K] by
its light/heavy phase role. The light stream enters the first stage and the
heavy stream enters the last, whether that stream is named ``feed`` or
``solvent``. Earlier versions swapped inlet temperatures when the feed was
heavy; corrected stage energies and temperatures can therefore differ for
unequal-temperature inlets in that configuration.

IDA's differential-state flags follow the same stage-major order as the
solution vector. The independent light-phase mole fractions and stage
internal energy receive integration error control; the final light-phase
mole fraction, heavy-phase composition, and temperature remain algebraic.
This also corrects previously inaccurate multistage trajectories under the
default algebraic-state suppression, including when the feed is light.
No solver-option override is needed to obtain the corrected state flags.

Continuous holdup accuracy
==========================

:code:`ContinuousHoldup.solve_unit` accepts :code:`sundials_opts`, like the
dynamic collector. Use :code:`{'rtol': relative_error, 'atol': absolute_error}`
to set CVode accuracy when comparing segmented and uninterrupted runs.
Relative error is dimensionless; absolute error follows the state order:
species mass fractions [-], then liquid temperature [K]. Omitted options keep
CVode's defaults, and the supplied mapping is not modified.

Vapor density
=============

:code:`VaporPhase.getDensity` uses the ideal-gas equation at the requested temperature and pressure. The default mass basis returns [kg/m**3]; :code:`basis='mole'` returns [mol/L], equivalently [kmol/m**3]. Earlier releases returned [mol/m**3] regardless of the selected basis.

The positional arguments remain pressure [Pa], temperature [K], phase, and basis. The :code:`pres_gas` and :code:`temp_gas` keywords remain supported. New code can use :code:`pres` and :code:`temp` keywords, with optional :code:`mass_frac` or :code:`mole_frac` composition overrides. Do not supply both spellings of the same pressure or temperature argument.

Vapor concentrations use the same ideal-gas basis: :code:`mole_conc = mole_frac * pres / (R * temp) / 1000` [mol/L] and :code:`mass_conc = mole_conc * mw` [kg/m**3], with molecular weights [g/mol]. The last array axis is species; pressure and temperature profiles broadcast over preceding axes. Concentration inputs specify composition, while the equation of state fixes total concentration. A zero-amount vapor retains these intensive properties.

UNIQUAC data without :code:`qip` use :code:`qi` locally and emit a warning once per property object. This fallback assumes the ordinary surface parameter also describes the modified residual term; systems requiring special parameters, including relevant water/alcohol models, should provide :code:`qip` explicitly. The fallback does not create a :code:`qip` attribute, so callers can still detect missing data.

Connected inlet interpolation
=============================

Connected inlet profiles are interpolated by :code:`PharmaPy.Connections.interpolate_inputs`, which :code:`LiquidStream.InterpolateInputs` and :code:`SlurryStream.InterpolateInputs` (used, for example, by :code:`DynamicCollector`) now share. At and after the last upstream time, scalar and array queries return the final upstream sample itself, and array queries are evaluated in the caller's order, also when unsorted. Earlier releases differed near and after the end of the profile. For scalar queries, :code:`LiquidStream.InterpolateInputs` held the second-to-last sample for queries closest to or after the last upstream time and used a two-point window ending at the second-to-last sample for queries closest to the second-to-last time, while :code:`SlurryStream.InterpolateInputs` returned the final sample at and after the last time but used a two-point window for queries closest to it. For array queries, :code:`interpolate_inputs` kept in-support rows in the caller's order but moved queries after the last upstream time to the end; the stream methods also moved them, filled them with the value of the last in-support query instead of the final sample, and raised :code:`IndexError` when every query was after the last upstream time. Newton interpolation of integer-valued profiles now keeps fractional results. The module-level :code:`PharmaPy.Streams.Interpolation` and :code:`PharmaPy.MixedPhases.Interpolation` helpers of v1.0.0 remain as deprecated aliases that emit :code:`DeprecationWarning`; they now evaluate a full local Newton window (as :code:`PharmaPy.Interpolation.local_newton_interpolation`) near the end of the profile. Use :code:`interpolate_inputs` for connected inlet profiles.

Connected flow conversions
==========================

Connected dynamic flows are converted between volume [m**3/s], mass [kg/s] and molar [mol/s] bases with each sample's own composition and, for vapor streams, each sample's temperature at the stored pressure. This applies in both directions: volume to mass or molar flow, and mass or molar flow to volume. Liquid density is the ideal-mixing value of the database pure-component densities, which are temperature-independent constants, so liquid conversions depend on composition only; concentration profiles contribute only their fractions. Earlier releases applied the source stream's stored final-state density to every volume-to-mass or volume-to-molar sample, so converted flows differ whenever the upstream composition varies over time; dynamic vapor conversions to volume flow also differ when the upstream temperature varies. Static sources are unchanged. A mass-concentration profile [kg/m**3] requested as mass fractions now yields mass fractions [-] only; earlier releases returned both mass and mole fractions, which downstream units could not consume. Dynamic conversions require a single-phase liquid or vapor stream: slurry profiles, whose densities are per phase, and unsupported composition bases now raise an error instead of returning unphysical flows. A dynamic sample whose composition is exactly zero, such as a stopped feed, converts to zero flow; non-finite samples propagate as NaN.

These conversions, like static raw-material rows, relate volume and amount through the ideal-mixing density. Dynamic raw-material accounting (below) instead reports concentration times volume flow as consumed by concentration-based species balances. The two agree only when the concentrations close the ideal-mixing density, for example for a :code:`LiquidStream` built from concentrations with :code:`name_solv`.

Dynamic raw-material accounting
===============================

See also *Connected flow conversions* above for how concentration-based accounting relates to ideal-mixing density conversions.

:code:`GetRawMaterials` and the raw-feed rows of :code:`GetStreamTable` account a continuous raw inlet with a :code:`DynamicInlet` as the feed the receiving unit actually consumed. At each of the unit's result times, the accounted feed holds the controlled fields and the static stream values of the other inlet fields the unit declares, exactly as the unit reads them: for integrating units each result time is evaluated separately, as during integration, so control callables need not accept arrays, while a :code:`Mixer` feed is evaluated once on the mixer's whole time grid, as the mixer does. Totals are trapezoidal integrals over those times. Units hold different fields: :code:`DynamicCollector` liquid feeds consume :code:`mass_flow` [kg/s], :code:`mass_frac` and :code:`temp`; evaporators, dynamic distillation and dynamic extraction consume :code:`mole_flow` [mol/s], :code:`mole_frac` and :code:`temp`; CSTR, semibatch and plug-flow reactors consume :code:`vol_flow` [m**3/s], :code:`mole_conc` [mol/L] and :code:`temp`. A composition-only control therefore holds the mass flow for a collector, the molar flow for an evaporator, and the volume flow for a reactor. Species amounts are the consumed concentration times the volume flow, or the consumed fractions times the consumed flow (a volume flow is converted with the density of each sample's composition and temperature). Samples whose composition is exactly zero, such as a stopped feed, contribute no material; non-finite values are not dropped and make the totals NaN. Slurry feeds of crystallizers and collectors consume :code:`vol_flow`, liquid :code:`mass_conc` [kg/m**3] and a population (:code:`mu_n` [m**n/m**3] or :code:`distrib` [#/m**3/um]); the liquid receives :code:`vol_flow * (1 - kv * mu_3) * mass_conc` and the crystals :code:`vol_flow * kv * mu_3` times the solid density, split by the static solid composition. Both phases report the consumed slurry temperature, and their rows follow the order of the slurry's :code:`Phases`, as for static feeds. Reported fractions are averaged with the flow on the accounting basis, so fraction times total equals each species' integrated amount; the consumed temperature is reported as is when constant and otherwise as its mass-flow-weighted average on both bases; pressure is the static value because no unit consumes inlet pressure. Dynamic rows report amounts and volume but no flow rates.

A control on a field the receiving unit does not consume would change the reported totals without changing the simulation, so it raises :code:`ValueError` naming the field and the fields the unit consumes. So does a :code:`DynamicInlet` attached to a phase of a slurry inlet, which no unit reads (attach the DynamicInput to the slurry itself), a population control on a single-phase stream, and a dynamic feed to a continuous unit that declares no inlet layout (for example :code:`ContinuousExtractor`), and a dynamic slurry feed to a unit that does not consume its volume flow, liquid concentration and population (for example :code:`ContinuousHoldup`). Earlier releases raised :code:`KeyError` unless a mass or molar flow was controlled and always reported the static composition. Cases that previously returned totals and now raise include: flow-controlled feeds whose DynamicInput also, or only, controls a field the unit does not consume (for example a :code:`mass_flow` or :code:`mole_flow` control on a slurry or reactor feed, or a :code:`mole_frac` control on a reactor feed); slurry feeds controlling :code:`mass_frac` or other unconsumed fields; phase-level DynamicInlets; and feeds controlling two flows. Totals also change for composition-controlled feeds and for feeds to units that do not consume :code:`mass_flow`, and the reported temperature of feeds controlling both flow and temperature is now the consumed, mass-flow-weighted value instead of the static stream temperature. The undocumented :code:`SimulationExec.get_dynamic_raw_inputs` helper, which split a mixed inlet's dynamic flow by static phase fractions, has been removed because that split contradicts the consumed feed; it was never part of a tagged release. Use :code:`GetRawMaterials(totals=False)` for per-phase records.

An :code:`Evaporator` with :code:`stop_at_maxvol=False` shuts its inlet when the liquid reaches 95% of the drum volume, keeps it shut on continuation, and records the absolute event time in :code:`inlet_stop_time` [s]; :code:`reset()` or a new :code:`Phases` assignment clears it. Raw-material accounting stops both dynamic and static feeds of such a unit there: dynamic totals integrate over the result times before the stop and end at the stop time, and static totals multiply the flow by the fed duration. Earlier releases kept accounting the feed after the inlet was shut, including over continuations.

A :code:`Mixer` that mixes a slurry or cake reads static phase values only, so an inlet with non-empty :code:`DynamicInlet` controls raises :code:`ValueError` naming the inlet; profiled and dynamic multiphase :code:`Mixer` inputs are unsupported. Earlier releases silently ignored such controls.

Solids mixing is instantaneous. After :code:`solve_unit`, the mixer publishes :code:`outputs`, a dictionary, and :code:`result`, a :code:`DynamicResult` whose attributes reference the same arrays (for example :code:`mixer.result.temp is mixer.outputs['temp']`), at the single time :code:`time = [0.]` [s], so a flowsheet records zero processing time for it. The published states are :code:`mass_liq` and :code:`mass_solid` ([kg] for batch phases, [kg/s] for streams, as recorded in :code:`states_di`), the liquid :code:`mass_frac` [-] in database species order, :code:`temp` [K], and the crystal population: :code:`distrib` [#/m**3/um] for a Slurry or SlurryStream outlet (for a stream, the crystal number rate [#/um/s] divided by the slurry volume flow [m**3/s]) or :code:`total_distrib` [#/um] for a Cake outlet. Each state has a leading time axis of length one, and :code:`x_cryst` holds the size grid [um], as in crystallizer results. :code:`solve_unit` still returns its five-element tuple. Earlier releases published that tuple as :code:`outputs` with no :code:`result`, so a solids mixer could not run in a flowsheet or connect to a downstream unit. A connected inlet with a single upstream sample, such as the outlet of another mixer or of a batch unit, is a constant feed: the mixer mixes its attached phase state, exactly as if the same phases were supplied directly. Whether an inlet is profiled is decided by the number of samples in its :code:`time_upstream` [s]. Earlier code counted the state names in the upstream output dictionary instead, so a connected inlet whose upstream published more than one state could not be mixed. Solids mixing is static, so a connected inlet whose :code:`time_upstream` has more than one sample raises :code:`ValueError` naming the inlet, the sample count, and the time window before any balance is evaluated.

:code:`SolidStream` keeps a :code:`vol_flow` [m**3/s] alias of its per-second solid volume :code:`vol` whenever that volume is defined, refreshed by :code:`updatePhase` together with :code:`mass_flow` [kg/s] and :code:`mole_flow` [mol/s]. For a raw continuous :code:`SlurryStream`, :code:`GetStreamTable` and :code:`GetRawMaterials(totals=False)` therefore report the solid row's :code:`vol_flow` as kv times the third moment of the crystal number rate [m**3/s] for every construction route, including :code:`SlurryStream()` whose :code:`SolidStream` carries the number rate [#/um/s]; :code:`GetRawMaterials` with totals and :code:`GetOPEX`, which total and price on the amount basis and do not show :code:`vol_flow`, complete for that route. Earlier releases raised :code:`AttributeError` in all four for a raw :code:`SlurryStream()` feed. Raw :code:`Cake` inlets of a :code:`Mixer` are reported as raw inlets, one row per phase, while a Cake transferred from an upstream unit is not a raw material. The liquid :code:`Mixer` declares :code:`mass_frac` in :code:`states_di` as dimensionless with one entry per species, as the solids mixer does, so :code:`dim_states` sums to the packed state width; earlier releases declared units of kg and a dimension of one.

Flowsheet execution order
=========================

:code:`SimulationExec` runs the units of a flowsheet graph in a reproducible topological order that does not depend on :code:`PYTHONHASHSEED`. Units without predecessors start in the order the graph declares them; afterwards, a unit joins a first-in, first-out queue when its last predecessor has run, and the successors of a unit are considered in the order of its successor list. Successor lists must therefore be ordered sequences such as lists; sets raise :code:`TypeError`. Each unit transfers its outlet to its successors right after it runs, so a :code:`Mixer` receives its inlets in the execution order of its predecessors, and a continuous liquid :code:`Mixer` takes its evaluation grid from the first of them with more than one sample. This equals declaration order when all predecessors are sources, but not in general: for :code:`{'S1': ['P'], 'P': ['M'], 'S2': ['M'], 'M': []}` the order is :code:`S1, S2, P, M`, so :code:`M` receives the :code:`S2` inlet before the :code:`P` inlet. Earlier releases could execute independent units, and hence feed a mixer, in a different order in each Python process. Recycles are not supported: a graph with a cycle, including a unit listed as its own successor, raises :code:`PharmaPyNonImplementedError` naming the units on or downstream of the recycle. Units that appear only as successors count toward this check; earlier releases compared the scheduled units with the graph keys alone and could silently skip recycle units. Printing a :code:`SimulationResult` shows :code:`A --> B --> C` only when the units form one connected directed path; otherwise it lists one :code:`source --> destination` line per connection, in declaration order, and the bare name of each unconnected unit, instead of joining the execution order with arrows.

Crystallizer initialization
===========================

Call ``unit.initialize_states(runtime=duration)`` or pass an absolute
``time_grid`` to prepare a crystallizer without integrating it. The method
returns the initial solver vector and absolute endpoint [s]; the final grid
entry takes precedence over a duration. It applies ``reset_states``, refreshes
state dimensions, and prepares vessel geometry exactly as ``solve_unit`` does.
An unset working volume is inferred from the charge or semibatch feed.

The vector follows ``name_states`` order. Crystal moments use micrometer
lengths internally; FVM populations include the configured numerical scale.
MSMPR crystal populations are per slurry volume. Batch and semibatch volume
states are liquid volume [m**3]. Reported ``result.mu_n`` retains its SI basis.

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

Utility flow controls
=====================

A :code:`DynamicInput`, or any controller exposing :code:`evaluate_inputs(time)`, attached to :code:`CoolingWater.DynamicInlet` may control :code:`temp_in` [K] and one flow basis, either :code:`mass_flow` [kg/s] or :code:`vol_flow` [m**3/s]; the fields it returns are the controlled ones. :code:`CoolingWater.get_inputs` derives the other basis from the controlled one with the constant utility density :code:`rho` [kg/m**3], elementwise for profile evaluations, so reactor and crystallizer jacket balances, which read :code:`vol_flow`, receive the controlled flow. Earlier releases returned the stored static value of the uncontrolled basis, so a :code:`mass_flow` control did not reach the jacket balances. Controlling both bases raises :code:`ValueError`; control only one. Without a :code:`DynamicInlet`, or with a :code:`temp_in`-only control, the stored static flows are returned unchanged.

Crystallizer jacket volume
==========================

Supply :code:`vol_ht` [m**3] from equipment data for a jacket model. When omitted, the model preserves a legacy jacket-volume assumption of 14% of its reference volume. Batch uses the current total slurry volume; MSMPR and Semibatch use :code:`vol_tank / vol_offset`. This is an equipment design assumption, not a correlation validated across vessel sizes. The factor is traceable to :code:`source/Crystallizers.py` in the repository's initial commit `e1e8164 <https://github.com/PharmaPy-org/PharmaPy/blob/e1e8164976e9103a396b24a7f7cfe5d1247c845c/source/Crystallizers.py>`_. Its original physical provenance remains the subject of `issue 113 <https://github.com/PharmaPy-org/PharmaPy/issues/113>`_. An explicit :code:`vol_ht` takes precedence and avoids relying on that assumption.

Crystallizer feeds and steady state
===================================

:code:`Slurry.Phases` rejects phase-only initialization with zero combined liquid and solid volume before normalizing moments or distributions. Supply a positive phase inventory first; a positive liquid inventory with zero crystals remains supported. Empty-slurry temperature initialization is not defined.

Moment-mode crystallizers accept static slurry moments and connected upstream moment profiles on the slurry-volume basis [m**n/m**3]. Connections from FVM crystallizers retain the reported moment history, rather than applying the final population at every time. Explicitly converted inlet moments retain precedence. The inlet must supply all moment orders required by the destination; extra higher orders are ignored, and missing orders raise an explanatory error.

A :code:`DynamicInput` attached to a slurry feed (:code:`Inlet.DynamicInlet`) keys its controls by plain field name. Each control is routed to the inlet group that declares it: for MSMPR and Semibatch crystallizers, :code:`vol_flow` [m**3/s], :code:`temp` [K], :code:`mu_n` [m**n/m**3] or :code:`distrib` [#/m**3/um] belong to the slurry inlet, and :code:`mass_conc` [kg/m**3] belongs to its :code:`Liquid_1` phase. Values are used as supplied, without unit or basis conversion. Declared fields without a control keep their static stream or phase values, and the stream itself is not modified. Controls that the unit does not declare are passed through with the slurry-inlet fields and ignored, as for single-phase feeds. A field name declared by more than one group cannot be attributed to one of them and raises :code:`ValueError`. A dynamic :code:`Liquid_1` :code:`mass_conc` enters the species balances only: for moment and 1D-FVM feeds alike, the inlet liquid density and the feed enthalpy, and the Semibatch liquid-volume balance, are still evaluated from the feed stream's static :code:`Liquid_1` composition. Earlier releases raised :code:`KeyError: 'Liquid_1'` for dynamic 1D-FVM slurry feeds.

Finite-radius FVM nucleation requires :code:`rad_zero` [um] to match the first size-grid point within floating-point roundoff. Legacy :code:`rad_zero=0` remains supported with positive-start grids. Moment-mode analytical Jacobians are restricted to prescribed-temperature Batch crystallization with zero-radius nuclei and :code:`mass_conc` kinetics. The built-in kinetic model and unit impurity factor assumptions also apply; unsupported operating modes, FVM, finite radii, and other concentration bases raise before solver construction.

Sensitivity :code:`sundials_opts` take precedence over the defaults :code:`sensmethod='SIMULTANEOUS'` and :code:`suppress_sens=False`. Continuous reporting remains required to collect CVODES sensitivities. Supported short nucleating cases are checked against complete-solve finite differences for numerical and analytical Jacobians. The longer shipped cooling-case benchmark remains an acceptance requirement of `issue 222 <https://github.com/PharmaPy-org/PharmaPy/issues/222>`_; these tests do not establish convergence for every kinetic regime.

Crystallizer :code:`basis` selects only the liquid composition passed to the kinetics: :code:`'mass_conc'` [kg/m**3], or :code:`'mass_frac'` [kg/kg], the mass concentration divided by the liquid density. Batch, Semibatch and MSMPR crystallizers integrate liquid mass concentration [kg/m**3] with either option, so their composition derivatives are [kg/m**3/s], as is the :code:`final_fn` residual returned by :code:`MSMPR.solve_steady_state`; its returned :code:`composition` remains in the selected basis. Earlier releases also divided the MSMPR composition derivative by the liquid density under :code:`'mass_frac'`, integrating the [kg/m**3] state with a [1/s] rate; Batch and Semibatch were unaffected. MSMPR results computed with :code:`basis='mass_frac'` therefore differ from earlier releases, and code consuming the former [1/s] derivative or residual must update its scaling.

:code:`MSMPR.solve_steady_state` initializes the kinetic target species itself; a prior dynamic solve is not required. It passes the complete species composition to kinetics in the selected basis, retains non-target species, and applies the same impurity growth factor as the dynamic model. Its constant-property, solid-free-feed, and growth assumptions still apply. Convergence diagnostics work with the declared SciPy 1.9 floor as well as the locked version. Numerical root convergence does not establish the full slurry physical balance: `issue 223 <https://github.com/PharmaPy-org/PharmaPy/issues/223>`_ still depends on the population/balance work tracked in `issue 300 <https://github.com/PharmaPy-org/PharmaPy/issues/300>`_. Crystallizer reset restores independent copies of the original phase inventories and rebuilds the slurry population and volume. Parameter-estimation phase modifiers also refresh those slurry quantities before the next solve. Relative supersaturation is :math:`S-1`, where :math:`S=c/c_{sat}`; callers relying on the former extra normalization must update their kinetic interpretation.

Slurry collection and batch discharge
=====================================

A :code:`DynamicCollector` fed a slurry delegates to an adiabatic :code:`SemibatchCryst` and needs two settings on the collector: :code:`KinCryst`, a crystallization kinetics object such as :code:`CrystKinetics`, and :code:`kwargs_cryst`, a dictionary with :code:`'target_ind'`, the zero-based integer index [-] of the crystallizing species in the inlet liquid's species order, and :code:`'target_comp'`, its name or a list, tuple or one-dimensional array of names. :code:`target_ind` must index the first :code:`target_comp` species in that order, the species the delegate crystallizes. Other entries, such as :code:`'scale'` [-], go to the :code:`SemibatchCryst` constructor, except that the collector sets :code:`method` and :code:`adiabatic`, so those keys are rejected, and replaces any :code:`num_interp_points` entry with its own. Only crystallizer sources supply both settings through a :code:`Connection`; for any other slurry source, such as a raw :code:`SlurryStream` or a solids :code:`Mixer`, set them on the collector. There is no non-crystallizing slurry collector. :code:`solve_unit` checks the settings before it sets up the solve: :code:`ValueError` names every missing setting, a reserved key, an empty or unknown :code:`target_comp`, or a :code:`target_ind` that indexes another species; :code:`TypeError` reports a :code:`kwargs_cryst` that is not a dictionary, a :code:`target_comp` of another type, or a :code:`target_ind` that is not an integer. Earlier releases failed later with a :code:`TypeError`, :code:`KeyError` or :code:`AttributeError`, depending on the missing setting, and silently seeded the species at :code:`target_ind` when it disagreed with :code:`target_comp`.

The collector seeds and reads the feed liquid's mass concentration [kg/m**3] for 1D-FVM as well as moment populations. A connected slurry, that is mixed-phase (:code:`PharmaPy.MixedPhases`) matter such as a :code:`SlurryStream`, from any source hands the collector its crystallizer inputs; liquid sources keep the liquid-mixer inputs. An :code:`MSMPR` publishes the liquid :code:`mass_conc` [kg/m**3], passed through unchanged. A source that publishes liquid mass fractions [-], such as the continuous solids :code:`Mixer`, has them converted to mass concentration as the fractions times the ideal mass-basis mixing density of the database pure-component liquid densities, :code:`w_i / sum_j(w_j / rho_liq_j)` [kg/m**3], which are temperature independent. Inputs the source does not publish, such as the solids Mixer's :code:`vol_flow` [m**3/s], are read from the transferred stream. Earlier releases read a zero concentration for raw 1D-FVM slurries, and the seed temperature solve then failed to converge.

:code:`BatchToFlowConnector` discharges a single liquid batch holdup as a :code:`LiquidStream` whose mass flow [kg/s] is the liquid holdup mass [kg] divided by :code:`cycle_time` [s] and multiplied by :code:`flow_mult` [-]. Only one liquid holdup is supported: its :code:`Phases` accept a :code:`LiquidPhase`, bare or as the only element of a list or tuple. Any other input is rejected before the connector changes, naming the offending classes: :code:`NotImplementedError` for a :code:`Slurry`, :code:`Cake`, :code:`SolidPhase` or :code:`VaporPhase` (or a subclass) and for several phases, such as two liquids or a liquid with a solid or vapor; :code:`TypeError` for streams, including :code:`LiquidStream` and :code:`SlurryStream`, and other non-phase objects; and :code:`ValueError` for an empty list or tuple. This includes a :code:`BatchCryst` connected to it in a flowsheet, whose :code:`Slurry` outlet is rejected. Earlier releases, for example, failed later with :code:`UnboundLocalError` for slurries and with :code:`TypeError: 'NoneType' object is not iterable` when a :code:`Cake` or a non-phase object was assigned to a new connector, silently discharged the previous holdup when a :code:`Cake`, a bare solid or vapor phase, a liquid stream or a non-phase object was assigned to a loaded one, and silently truncated a phase list or tuple to its first liquid, dropping its solid, vapor or second liquid. The transfer is instantaneous, like solids mixing: :code:`result` and :code:`outputs` hold one sample at :code:`time = [0.]` [s] with :code:`temp` [K], :code:`pres` [Pa] and :code:`mass_flow` [kg/s] of shape (1,) and :code:`mass_frac` [-] of shape (1, number of species), so a flowsheet records zero processing time for the connector and a :code:`Connection` hands the downstream continuous unit a constant feed. That feed does not stop after :code:`cycle_time`: it lasts for the downstream unit's whole runtime, so to transfer exactly the holdup mass times :code:`flow_mult` [kg], run the downstream unit for :code:`cycle_time` [s]; a longer run delivers more than the holdup. Earlier releases published :code:`time = None`, and every flowsheet containing the connector failed in :code:`SolveFlowsheet` with :code:`TypeError: 'NoneType' object is not subscriptable`.

Evaporator heat duties
======================

Evaporator :code:`heat_profile` columns are powers [J/s], and :code:`heat_duty` contains their cumulative trapezoidal integrals [J] across segments since the last reset or public :code:`Phases` assignment. For batch evaporators, column 0 is heat into the drum (positive for heating) and column 1 is condensation heat (negative for cooling); the cumulative energies retain their signs. For continuous evaporators, column 0 is jacket/utility duty (positive for heat removed) and column 1 is condenser duty (negative for cooling), evaluated on the vapor-composition basis. This replaces the former liquid-composition basis, so existing zero-reflux runs also report a different column-1 value. Continuous duties accumulate signed energy first and then report its magnitude; each physical duty is counted once. Partition independence holds for a fixed trajectory: continuous :code:`solve_unit` restarts at time zero, so splitting solves need not reproduce the same trajectory.

For nitrogen-enabled semibatch evaporation, the original inlet remains attached throughout each solve. Feed controllers and connected profiles use the original condensable species order; the evaporator appends a zero nitrogen fraction after evaluating the feed. Changes made through the inlet's :code:`updatePhase` method take effect on the next segment.

Evaporator solver boundaries
============================

Flash solves verify convergence, finite scaled residuals, and phase-fraction bounds before publishing outlet phases. Tiny bound violations within the residual tolerance are clipped and the residual is rechecked; invalid or unconverged states raise :code:`RuntimeError`. :code:`residual_tolerance` is configurable and defaults to the square root of machine epsilon.

A batch evaporator using :code:`stop_at_maxvol=True` rejects an initial or reused charge at the 95% liquid-volume cap, including the event's roundoff neighborhood. Drain or reset before restarting a cap-limited run. With :code:`stop_at_maxvol=False`, the existing mode stops the feed at the cap and continues evaporation.

Continuous steady evaporation accepts :code:`fsolve_opts` for state scaling and solver controls. There is no universal scaling or trust-region factor: the regression suite covers both zero reflux and active reflux with fixture-specific :code:`diag`, :code:`xtol`, and :code:`factor`. Property failures during a trial preserve their original exception as the cause and identify thermophysical data and :code:`fsolve_opts` as diagnostic inputs. Successful solver termination should still be checked against physically meaningful balance residuals.

Evaporator reporting times
==========================

``Evaporator.solve_unit(runtime, time_grid=times)`` accepts absolute reporting
points [s] within the current segment. They must be finite and strictly
increasing; the solver also reports the segment start. ``runtime`` remains the
segment duration [s], including on continuation. Omitting ``time_grid`` retains
adaptive reporting. This supports comparisons at shared times without changing
the native IDA solver or interpolating the returned trajectory.

Solid distributions and filtration
===================================

:code:`SolidPhase.distrib_type` accepts :code:`'vol_perc'` or :code:`'mass_frac'`. The formerly tolerated :code:`'mass_perc'` spelling now raises an error. The selected basis normalizes bin weights before conversion to a number distribution. Porosity normalizes positive represented solid volume without adding a dimensional epsilon; a distribution with no represented volume raises. The zero-ended-grid quadrature contract remains provisional under `issue 269 <https://github.com/PharmaPy-org/PharmaPy/issues/269>`_.

Filter requires a resolved size distribution consistent with the current solid moments and mass. A moment-grown crystallizer can retain an old size distribution on its :code:`SolidPhase`; that handoff is rejected. Use a validated reconstruction or a compatible FVM population. Moment data alone do not uniquely determine a size distribution.

Filter pressure must be finite and positive; physical specific cake resistance must be positive and medium resistance nonnegative. Log parameters must exponentiate to finite values in this domain. These conditions are checked before optional solver construction. A positive-pressure run that finishes filtration retains its physical plateau. Repeated fresh portions use an absolute elapsed clock. Direct :code:`Filter.retrieve_results` calls must supply :code:`mass_solids` [kg] for the solved portion; it is no longer inferred from mutable phase inventory.

Washing inventory
=================

Displacement washing uses the attached cake's packing porosity for flow, adsorption, retained liquid, and effluent balances. The washing-unit diameter sets cross-sectional area and cake height; it does not imply repacking the cake at a different porosity. The dispersion correlation now implements the thick-cake branch for heights above 0.10 m: for :math:`Re\,Sc>1`, its convective contribution is :math:`1.75 Re\,Sc`, compared with :math:`55.5 (Re\,Sc)^{0.96}` at or below 0.10 m. The stagnant contribution remains :math:`1/\sqrt{2}`. Particle sizes are converted from micrometers to meters before evaluation. See :code:`DisplacementWashing.get_diffusivity` for the source equations and applicability assumptions.

Deliquoring checks inferred removal against initial and retained species inventories. Only arithmetic-scale negative roundoff is clipped. A larger negative removal emits a warning, preserves signed :code:`liquid_removed_species` and removal history, sets :code:`removal_diagnostics_valid=False`, and returns NaN removal composition. The transport-conservation defect tracked in `issue 29 <https://github.com/PharmaPy-org/PharmaPy/issues/29>`_ is not repaired by those diagnostics; invalid removal data must not be used as a physical effluent stream.


Dynamic distillation initialization
===================================

After configuring the column with ``column_startup()``, use ``init_unit()`` to
inspect the initial DAE state and material rates without Assimulo. Rows run from
the top stage to the reboiler; the first state column is temperature [K] and the
remaining columns are species mole fractions [-]. Every stage starts at the
attached liquid composition and its configured activity-model bubble point.
``solve_unit`` calls the same preparation after calculating the shortcut design.

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
