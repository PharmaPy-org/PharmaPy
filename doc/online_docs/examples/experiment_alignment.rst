Experiment identity in parameter estimation
===========================================

``ParameterEstimation`` uses the insertion order of ``x_data`` as the order of
named experiments. Observations and experiment-specific callback arguments are
selected by those keys, so changing their dictionary insertion order does not
change the objective or fitted parameters. Missing or extra experiment keys
raise ``ValueError`` with the input name and differing keys.

For example, these dictionaries describe the same two constant-rate experiments
despite their different insertion orders:

.. code-block:: python

   times = {"dilute": np.array([0., 1., 3.]),
            "concentrated": np.array([0., 2., 3.])}  # [s]
   observations = {"concentrated": np.array([10., 18., 22.]),
                   "dilute": np.array([1., 3., 7.])}  # [mol/L]
   initial_values = {"concentrated": (10.,), "dilute": (1.,)}  # [mol/L]
   gains = {"dilute": {"gain": 1.}, "concentrated": {"gain": 2.}}  # [-]

   estimator = ParameterEstimation(
       model, [1.5], times, observations,  # seed rate [mol/L/s]
       args_fun=initial_values, kwargs_fun=gains)

Here ``model(parameters, time, initial, gain=1.)`` returns
``initial + gain * parameters[0] * time`` in mol/L. The data have an exact
shared rate of 2 mol/L/s. Supply ``numpy`` and ``ParameterEstimation`` imports
and your model callback when using the pattern.

Compatibility and ordering
--------------------------

* An array represents one experiment. Lists or tuples of arrays retain their
  positional order. Positional callback lists contain one argument tuple or
  keyword dictionary per experiment. Counts must match the data.
* For one experiment, ``args_fun=(initial,)`` and
  ``kwargs_fun={"gain": multiplier}`` remain direct callback arguments.
  A named experiment also accepts ``args_fun={name: (initial,)}`` and
  ``kwargs_fun={name: {"gain": multiplier}}``.
* A single experiment's keyword dictionary is interpreted as experiment-keyed
  only when its sole key is the experiment name and its value is a dictionary.
  Other dictionaries retain their direct callback meaning. To pass the reserved
  structure itself as callback keywords, use ``kwargs_fun=[{name: {...}}]``.
* Legacy callback dictionaries with unnamed multiple datasets retain insertion
  order. Prefer positional lists, or name the experiments in ``x_data`` so keys
  can be checked.
* Alignment changes experiment order only. State columns, nested spectral and
  non-spectral measurements, sampling masks, units and physical bases retain
  their existing meaning. Nested measurement fields must already follow the
  corresponding independent-data field order.

``SimulationExec.SetParamEstimation`` uses the same constructor. Its named
phase/control modifiers therefore stay associated with the matching observations,
including when only one named experiment is supplied.
