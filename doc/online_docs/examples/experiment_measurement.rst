Experiments and measurements in parameter estimation
====================================================

``ParameterEstimation`` accepts its data as ``Experiment`` objects. Each
experiment holds named ``Measurement`` objects, every one with its own
independent-variable samples and observations, together with the callback
arguments that distinguish the experiment. Legacy arrays, lists and
dictionaries remain supported and are compiled by the same alignment code
(see `Legacy compatibility adapters`_ and :doc:`experiment_alignment`).

No unit conversion is performed anywhere in this path. Unit and basis labels
document the data and are checked for consistency; the numbers must already
be in the units and physical basis of the model callback output.

Measurement schema
------------------

``Measurement(field, x, values, *, uncertainty=None, units=None, basis=None,
x_units='s')`` describes one observed model-output field in one experiment.

``field``
   Identity of the observed model output: a zero-based ``int`` column of the
   callback output, or a ``str`` name resolved through the ``output_names``
   given to ``ParameterEstimation`` (the callback must then return exactly
   ``len(output_names)`` columns).
``x``
   Independent-variable samples, shape ``(n,)``, in ``x_units``; a scalar is
   a single sampled point. Samples must be finite and non-decreasing.
``x_units``
   Label of the independent-variable units, ``'s'`` (time in seconds) by
   default; ``None`` leaves them undeclared.
``values``
   Observations, shape ``(n,)``, already in the units and physical basis of
   the model output column. ``NaN`` marks a missing observation; infinite
   values are rejected and at least one observation is required.
``uncertainty``
   Standard deviation of the measurement error (not a variance and not a
   weight), a positive scalar or one entry per sample, in the units of
   ``values``. It must be finite and positive at every observed sample;
   entries at missing samples are ignored.
``units`` and ``basis``
   Labels such as ``'mol/L'`` and ``'molar concentration'``. They are not
   used for conversion. Declared labels must agree for every measurement of
   the same name and of the same model-output field; ``None`` (undeclared)
   agrees with anything.

The arrays are stored as private, read-only float copies: changing the
caller's arrays afterwards does not affect the measurement, and fitting,
bootstrapping and refitting never modify the measurement or the caller's
data.

Experiment schema
-----------------

``Experiment(measurements, args=(), kwargs=None)`` describes one
experimental run.

``measurements``
   A mapping of measurement names to ``Measurement`` objects, whose insertion
   order is the declared order, or a sequence of ``Measurement`` objects named
   after their field (the ``str`` field itself or ``'field_<int>'``).
   Replicate measurements of one field need the mapping form with distinct
   names. Declared ``x_units`` must agree for all measurements of all
   experiments fitted together, because one model callback receives every
   grid; ``None`` (undeclared) is compatible with any declared label.
``args`` and ``kwargs``
   The experiment's callback association: the model is evaluated as
   ``func(params, x_model, *args, **kwargs)``, with values in the units the
   model defines (for example an initial concentration [mol/L] or an
   initial temperature [K]). ``args`` is stored as a tuple; ``kwargs`` as a
   shallow copy exposed through a read-only view.

With ``Experiment`` input, ``y_data``, ``measured_ind``, ``args_fun`` and
``kwargs_fun`` of ``ParameterEstimation`` must be ``None``: observations,
measured fields and callback arguments live in the experiments.

Sampling grids
--------------

Each experiment is evaluated on one model grid, the multiset union of its
measurement grids: every distinct x appears as many times as the largest
number of replicates any measurement of the experiment has at that x, in
sorted order. The model callback therefore receives each experiment's grid
once per evaluation, and every measurement reads its own rows.

* **Different grids.** Measurements of one experiment may be sampled at
  different times. An entry of the model grid that a measurement does not
  sample is *unobserved* for it; nothing is interpolated, extrapolated or
  otherwise invented, also when the grids only partially overlap.
* **Single sampled point.** A scalar ``x`` (or a one-element array) is a
  valid measurement, for example one temperature reading.
* **Replicates.** A repeated x value within a measurement denotes replicate
  observations at that x; the k-th replicate maps to the k-th model-grid row
  holding that x, so the callback receives the repeated value. Replicates
  can also be given as separate measurements of the same field.
* **Order.** Decreasing samples raise ``ValueError``; sort the samples
  together with their values.
* **Missing observations.** ``NaN`` values are unobserved entries. They keep
  their model-grid row (the model is still evaluated there) but contribute
  no residual, no Jacobian entry and no data count.
* **Measurements absent from an experiment.** A measurement name that one
  experiment does not declare is unobserved in that whole experiment.
* **Unavailable model fields.** A ``str`` field without ``output_names``, or
  one that is not among them, and an ``int`` field outside the declared
  ``output_names`` raise ``ValueError`` at construction. Without
  ``output_names``, an ``int`` field beyond the columns the callback returns
  raises ``ValueError`` at the first evaluation, naming the measurement and
  the experiment.

Alignment by keys and declared order
------------------------------------

* **Experiments.** A mapping of names to ``Experiment`` objects is aligned by
  its keys, and its insertion order is the experiment order. A list or tuple
  names its experiments ``'exp_1'``, ``'exp_2'``, ...; a single
  ``Experiment`` is ``'exp_1'``.
* **Measurement columns.** Residual columns are the union of measurement
  names in first-appearance order (experiment order, then each experiment's
  declared order). A name must denote the same model-output field in every
  experiment. Reordering experiments or measurements never changes the
  objective, the fitted parameters (up to solver tolerance) or the parameter
  covariance. With declared uncertainties, the default identity weighting or
  a diagonal ``weight_matrix`` it only permutes the entries of the weighted
  residual vector and the Jacobian columns. With a correlated
  ``weight_matrix`` each sample row is whitened by a Cholesky factor that
  depends on the column order, so individual whitened entries change by
  more than a permutation (their sum of squares per row does not); compare
  fits through the objective, the parameters or the covariance rather than
  entry by entry.
* **Model outputs.** Fields select callback output columns by index or, with
  ``output_names``, by name. ``name_states`` defaults to the measurement
  names.
* **Weights.** With ``Experiment`` input a ``weight_matrix`` must be a
  ``pandas.DataFrame`` whose index and columns are the measurement names, in
  any order. It is reordered by name, so reordering experiments cannot
  re-weight the fit; a plain array raises ``TypeError``.

Residual packing
----------------

The weighted residual vector returned by
``get_objective(params, out_array=True)`` and stored in ``info['fun']`` is
packed by experiment, then measurement column, then model-grid row.
Unobserved entries keep their positions and are exactly zero, so the vector
can be longer than the number of observations ``num_data_total``.
``get_residual_layout()`` returns a ``pandas.DataFrame`` with one row per
entry and the columns ``experiment``, ``measurement``, ``field``, ``x`` (in
the model's x units) and ``observed``. The columns of ``info['jac']`` (and
the rows of ``sens``) follow the same layout, and their unobserved entries
are zero as well.

Uncertainty scaling for different physical quantities
-----------------------------------------------------

When the measurements declare standard deviations ``sigma``, each observed
residual becomes ``(model - data) / sigma`` [-] and each sensitivity is
divided by the same ``sigma``. The objective
``1/2 * sum(((model - data) / sigma)**2)`` then adds dimensionless terms,
so a concentration [mol/L] and a temperature [K] contribute in proportion
to how precisely each was measured.

Without declared uncertainties every residual keeps its own unit and the
default identity weighting implicitly assumes a standard deviation of one in
whatever unit each quantity happens to be expressed. The sum of squared
[mol/L] and [K] residuals is not dimensionally meaningful: recording the
temperature in mK instead of K would multiply its weight by one million and
change the fitted parameters. Declare uncertainties whenever measurements of
different physical quantities are fitted together.

* Either every measurement declares an uncertainty or none does; mixing
  dimensionless and unit-bearing residuals raises ``ValueError``.
* Declared uncertainties and ``weight_matrix`` are alternative weightings
  and cannot be combined (``ValueError``).

A ``weight_matrix`` is the covariance of the measurement errors of the
measurement columns at one sample, entry ``(i, j)`` in the product of the
units of columns ``i`` and ``j``; correlated covariances are supported and
only its lower triangle is read. Residual rows ``r_k`` are weighted by the
lower Cholesky factor of its inverse, so the objective is
``1/2 * sum_k r_k @ inv(weight_matrix) @ r_k`` (generalized least squares).
A sample row that observes only some columns ``o`` (staggered grids, missing
observations or measurements absent from the experiment) is weighted by the
marginal precision ``inv(weight_matrix[o][:, o])`` of the observed columns,
not by the corresponding block of the full precision, and its unobserved
entries stay zero. Missing data and correlated weighting therefore keep the
semantics described in :doc:`../advanced_usage`.

Statistics, covariance and bootstrap
------------------------------------

* **Degrees of freedom.** ``num_data_total`` counts observed entries only, and
  ``StatisticsClass`` uses ``num_data_total - num_params`` degrees of
  freedom.
* **Covariance.** ``get_covariance()`` returns ``s2 * inv(J @ J.T)`` with the
  residual mean square ``s2 = r @ r / (num_data_total - num_params)``. This
  is a parameter covariance in squared parameter units only when the
  weighted residuals are dimensionless (declared uncertainties, or a
  ``weight_matrix`` holding error covariances in squared measurement units)
  or all residuals share one physical unit. With identity weighting of mixed
  units, such as [mol/L] and [K], the result depends on the units chosen and
  is not a physical covariance. With
  declared uncertainties ``s2`` is the reduced chi-square [-], so the default
  treats the standard deviations as relative weights whose common scale is
  re-estimated from the residual scatter, like
  ``scipy.optimize.curve_fit(..., absolute_sigma=False)``. When the declared
  standard deviations are known in absolute terms,
  ``get_covariance(include_mse=False)`` returns ``inv(J @ J.T)``, the
  covariance that treats them as absolute (``absolute_sigma=True``).
  ``optimize_fn`` returns the ``include_mse=True`` form. Every
  ``get_covariance`` call stores its result in ``covar_params``, which
  ``StatisticsClass.get_intervals`` uses by default.
* **Residual bootstrap.** ``StatisticsClass.get_bootsamples`` and
  ``bootstrap_params`` resample, for each experiment and measurement column,
  only the residuals of observed entries; unobserved entries of the
  generated datasets are ``NaN``. With declared uncertainties the
  standardized residuals ``z = r / sigma`` [-] are resampled and entry ``i``
  of a generated dataset is ``y_fit[i] - sigma[i] * z``, so errors of
  precise samples are not placed at imprecise ones. The generated errors
  follow the empirical residual scatter, like ``include_mse=True`` (and
  ``curve_fit(..., absolute_sigma=False)``): the declared standard
  deviations set only the relative error size of each entry, and errors
  are not drawn with the declared absolute ``sigma``. Drawing them so would
  require a parametric bootstrap, which is not provided. Without declared
  uncertainties the draws are unchanged. A measurement with a single
  observation in an experiment has a pool of one residual and is
  reproduced unchanged in every generated dataset.
* **Isolation.** Bootstrap refits run on copies of the estimator; the
  estimator, its ``Experiment`` and ``Measurement`` objects and the caller's
  arrays are unchanged afterwards (see :doc:`../advanced_usage`).

Example
-------

Two runs of a first-order reaction A -> B with rate constant ``k`` [1/s] are
fitted together. The model returns the concentration of A [mol/L] and the
adiabatic temperature [K]. The ``hot`` run measures the concentration at four
times and the temperature once; the ``cold`` run measures only the
concentration, at other times, and one of its observations is missing.
Uncertainties are declared in each measurement's own unit.

.. testcode::

   import numpy as np
   from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation


   def decay(params, time_s, initial_mol_l, rise_k_l_mol=0.0, temp0_k=300.0):
       """Return the states of a first-order A -> B batch reaction.

       Parameters
       ----------
       params : numpy.ndarray
           Rate constant k, shape (1,) [1/s].
       time_s : numpy.ndarray
           Model grid, shape (n_times,) [s].
       initial_mol_l : float
           Initial concentration of A [mol/L].
       rise_k_l_mol : float, optional
           Adiabatic temperature rise per converted concentration [K*L/mol].
       temp0_k : float, optional
           Initial temperature [K].

       Returns
       -------
       numpy.ndarray
           Shape (n_times, 2); column 0 is c_A [mol/L], column 1 is T [K].
       """
       converted = initial_mol_l * (1 - np.exp(-params[0] * time_s))  # [mol/L]
       return np.column_stack((initial_mol_l - converted,
                               temp0_k + rise_k_l_mol * converted))


   hot = Experiment(
       {'c_A': Measurement(0, [0.0, 1.0, 2.0, 4.0],  # [s]
                           [2.010, 1.200, 0.750, 0.265],  # [mol/L]
                           uncertainty=0.02, units='mol/L'),
        'T': Measurement(1, 2.0, 325.6,  # one sample at 2 s [K]
                         uncertainty=0.5, units='K')},
       args=(2.0,),  # initial c_A [mol/L]
       kwargs={'rise_k_l_mol': 20.0})  # [K*L/mol]
   cold = Experiment(
       {'c_A': Measurement(0, [0.5, 1.5, 3.0],  # [s]
                           [0.785, np.nan, 0.220],  # [mol/L], NaN = missing
                           uncertainty=[0.01, 0.01, 0.02], units='mol/L')},
       args=(1.0,),  # initial c_A [mol/L]
       kwargs={'rise_k_l_mol': 10.0, 'temp0_k': 290.0})  # [K*L/mol], [K]

   estimator = ParameterEstimation(decay, [0.3], {'hot': hot, 'cold': cold},
                                   name_params=['k'])  # seed k [1/s]
   layout = estimator.get_residual_layout()
   print(layout)
   print(estimator.num_data_total)

The ``hot`` grid is the union of its concentration and temperature grids;
the ``cold`` grid keeps the 1.5 s sample whose observation is missing. The
temperature column is unobserved except at 2 s in ``hot``, and seven entries
are observed:

.. testoutput::

      experiment measurement  field    x  observed
   0         hot         c_A      0  0.0      True
   1         hot         c_A      0  1.0      True
   2         hot         c_A      0  2.0      True
   3         hot         c_A      0  4.0      True
   4         hot           T      1  0.0     False
   5         hot           T      1  1.0     False
   6         hot           T      1  2.0      True
   7         hot           T      1  4.0     False
   8        cold         c_A      0  0.5      True
   9        cold         c_A      0  1.5     False
   10       cold         c_A      0  3.0      True
   11       cold           T      1  0.5     False
   12       cold           T      1  1.5     False
   13       cold           T      1  3.0     False
   7

Fitting returns dimensionless residuals in the same layout:

.. testcode::

   # k [1/s], its covariance [1/s**2] and solver information
   params, covariance, info = estimator.optimize_fn(verbose=False)
   layout['residual'] = info['fun']  # [-], (model - data) / sigma
   print(layout[layout['observed']].round(3))
   print(f"k = {params[0]:.4f} 1/s")

   std_absolute = np.sqrt(
       estimator.get_covariance(include_mse=False)[0, 0])  # [1/s]
   std_relative = np.sqrt(estimator.get_covariance()[0, 0])  # [1/s]
   print(f"{std_absolute:.2e} 1/s, {std_relative:.2e} 1/s")

.. testoutput::

      experiment measurement  field    x  observed  residual
   0         hot         c_A      0  0.0      True    -0.500
   1         hot         c_A      0  1.0      True     0.584
   2         hot         c_A      0  2.0      True    -0.796
   3         hot         c_A      0  4.0      True     0.222
   6         hot           T      1  2.0      True    -0.563
   8        cold         c_A      0  0.5      True    -0.664
   10       cold         c_A      0  3.0      True     0.118
   k = 0.5011 1/s
   7.45e-03 1/s, 4.35e-03 1/s

The first standard deviation of ``k`` treats the declared uncertainties as
absolute. The second, the default that ``optimize_fn`` also returned,
rescales them by the residual scatter; it is smaller here because the
reduced chi-square is below one. The last call leaves the default form in
``covar_params``.

Legacy compatibility adapters
-----------------------------

Legacy ``x_data``/``y_data`` arrays, lists and dictionaries are adapted to
one measurement column per ``y_data`` column, named ``'y_0'``, ``'y_1'``,
..., with field ``measured_ind[j]``, and compiled by the same alignment code
as ``Experiment`` input. A shared grid is aligned by identity (row ``i`` of
``y_data`` to entry ``i`` of ``x_data``) and passed to the callback
unchanged; per-state (staggered) grids use the multiset union. Valid legacy
input therefore keeps its model grid, masks, objective, residuals, Jacobian,
fitted parameters and parameter ordering, the observation dtype and the
callback argument containers. Experiment identity rules for legacy input are
described in :doc:`experiment_alignment`.

Intentional changes for legacy input:

* A ``NaN`` observation marks a missing observation, excluded from the
  residuals and the data count; it previously made the objective ``NaN``.
* Object-dtype observations whose elements are all real numbers (for
  example pandas nullable columns exported with ``na_value=numpy.nan``) are
  converted to float64, so ``NaN`` entries are missing observations. Other
  object or non-numeric observations raise ``TypeError``.
* A column that is empty (outside staggered grids), has no non-NaN
  observation or holds an infinite value raises ``ValueError``, and complex
  observations raise ``TypeError``; previously they gave a non-finite or
  complex objective.
* Raw residuals (``resid_runs``) of unobserved entries are exactly zero for
  every observation dtype; with float32 observations they were previously
  rounding residues of the model values.
* Decreasing staggered grids raise ``ValueError``; they were previously
  paired with the wrong observations. Staggered grids with repeated values
  are now accepted as replicates.
* Observation arrays whose row count differs from the shared grid, or whose
  column count differs from ``measured_ind``, raise ``ValueError``; they were
  previously broadcast.
* Callback outputs must be one- or two-dimensional with one row per
  model-grid sample, and selected columns must exist; violations raise
  ``ValueError`` naming the experiment instead of being broadcast or raising
  ``IndexError``.
* ``measured_ind`` is read at construction; later changes to the caller's
  object do not affect residuals or sensitivities.
* For a one-dimensional callback output with a ``measured_ind`` listing
  column 0 more than once (for example ``[0, 0]``), ``y_runs`` holds one
  column per entry; it was previously one column broadcast against the
  observations, with equal residuals.

Nested ``'spectra'``/``'non_spectra'`` observation dictionaries and
``MultipleCurveResolution`` keep their earlier layout and do not accept
``Experiment`` objects.

Using experiments through SimulationExec
----------------------------------------

``SimulationExec.SetParamEstimation`` fits a flowsheet unit through its
``paramest_wrapper`` (time [s] is the independent variable). Every
experiment receives its own wrapper keywords ``modify_phase``,
``modify_controls`` and ``run_args``, built from ``phase_modifiers``,
``control_modifiers`` and a per-experiment copy of ``wrapper_kwargs``:

* **Named experiments** (a dictionary ``x_data`` or a mapping of
  ``Experiment`` objects): modifiers are dictionaries keyed by exactly the
  experiment names, in any order. Missing or unexpected names raise
  ``ValueError`` naming the argument and the experiments.
* **Positional experiments** (a list or tuple ``x_data``): modifiers are lists
  or tuples with one entry per experiment in ``x_data`` order. Several
  unnamed experiments are supported; a dictionary that cannot be aligned with
  them raises ``ValueError``.
* **One unnamed experiment**: the modifier dictionary itself, or a
  one-element list or tuple holding it.
* **Experiment objects**: new ``Experiment`` objects are built whose
  ``kwargs`` add the wrapper keywords to the experiment's own; the caller's
  objects are not modified. An experiment's ``args`` are bound to the
  wrapper signature after the parameters and time grid, so
  ``Experiment(..., args=(phase_modifier,))`` binds ``modify_phase``.
  Binding a wrapper keyword through ``args`` or ``kwargs`` while the
  corresponding ``SetParamEstimation`` argument is also given raises
  ``ValueError``.

Only modifier keys and entry types are validated by ``SetParamEstimation``;
applying control modifiers inside reactor callbacks is tracked in
`issue #271 <https://github.com/PharmaPy-org/PharmaPy/issues/271>`__.
