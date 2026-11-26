Choosing the next experiment
============================

Use ``PharmaPy.ExperimentDesign.rank_experiments`` to compare a finite
set of feasible control and sampling plans at a converged calibration.
The example uses the synthetic first-order decay stand-in from
:doc:`calibration_validation`, with only the concentration measurements.
It verifies the software handoff. The crystallization and bioreactor
recommendations belong to their respective case studies.

The finite-candidate backend uses NumPy. For each plan it evaluates the
local prediction sensitivities :math:`H_{ki} = \partial y_k/\partial p_i`
in the accepted parameter coordinates. For prospective measurement error
covariance :math:`R` it adds :math:`H^\mathsf{T} R^{-1}H` to the
calibrated Fisher information. This assumes independent errors between
the calibration data and the new run, with the new covariance fixed at
the accepted point. Covariance can include correlations within the run.
Direct enumeration gives the D-optimal choice within the supplied plans
for one additional run.

The score maximizes the pooled log determinant in the fitted parameter
units. With a full-rank baseline, the reported gain is the log of the
determinant ratio, and :math:`\exp(-\mathrm{gain}/2)` is the predicted
joint uncertainty volume relative to the baseline. A plan can also
restore numerical rank to a deficient baseline. In that case the pool
uses the numerically resolved directions from the baseline and the
candidate's whitened sensitivities. The candidate is evaluated on the
unresolved directions independently of the strength of the retained
directions. Numerical null modes of the candidate are excluded from the
gain calculation. A determinant ratio is undefined
when the baseline is singular. Unrankable and failed plans carry an
explicit status and diagnostic. A report with no ranked entry has no
recommendation.

Use ``get_identifiability(include_mse=False)`` when the calibration
noise is absolute. Use ``include_mse=True`` when the calibration
weights have a common unknown variance scale. The latter divides the
Gauss–Newton information by the residual mean square to obtain Fisher
information. Either report can be combined with absolute prospective
variances. Set ``relative_noise=True`` for a candidate only if its
covariance should be multiplied by the same estimated calibration scale.
The report must already have a finite, positive estimate of that scale.
Declare an uncertainty or covariance for each measurement basis when the
outputs have different units.

Executable decay example
------------------------

The feasible controls in this example are a fixed initial concentration
of 2 mol/L and two concentration observations per candidate, in the
0.5–4 s range spanned by the positive calibration times. These are design
assumptions for the fixture. Every candidate uses the absolute hot-run
standard deviation of 0.02 mol/L. The fit uses relative calibration
weights, so this example combines two distinct noise declarations.

.. testcode::

   import numpy as np

   from PharmaPy.ExperimentDesign import CandidateExperiment, rank_experiments
   from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation


   def concentration(params, time_s, initial_mol_l):
       """Return first-order decay at the requested times.

       Parameters
       ----------
       params : array_like
           Rate k [1/s], shape (1,).
       time_s : array_like
           Observation times [s], shape (n_times,).
       initial_mol_l : float
           Initial molar concentration [mol/L].

       Returns
       -------
       numpy.ndarray
           Remaining molar concentration [mol/L], shape (n_times,).
       """
       return initial_mol_l * np.exp(-params[0] * np.asarray(time_s))


   def prediction_derivatives(time_s, initial_mol_l):
       """Bind a prospective schedule.

       Parameters
       ----------
       time_s : array_like
           Planned observation times [s], shape (n_times,).
       initial_mol_l : float
           Initial molar concentration [mol/L].

       Returns
       -------
       callable
           Takes the named accepted rate [1/s] and returns the raw
           concentration derivative [mol/L*s] in schedule order.
       """
       def sensitivities(point):
           """Evaluate the prospective concentration sensitivity.

           Parameters
           ----------
           point : mapping
               Accepted rate [1/s] under the key 'k'.

           Returns
           -------
           dict
               Derivative [mol/L*s] in schedule order under the key 'k'.
           """
           remaining = concentration([point['k']], time_s, initial_mol_l)  # [mol/L]
           return {'k': -np.asarray(time_s) * remaining}

       return sensitivities


   hot_std = 0.02  # [mol/L], declared hot-run measurement standard deviation
   experiments = {
       'hot': Experiment(
           {'c_A': Measurement(0, [0.0, 1.0, 2.0, 4.0],  # [s]
                               [2.010, 1.200, 0.750, 0.265],  # [mol/L]
                               uncertainty=hot_std, units='mol/L')},
           args=(2.0,)),
       'cold': Experiment(
           {'c_A': Measurement(0, [0.5, 1.5, 3.0],  # [s]
                               [0.785, np.nan, 0.220],  # [mol/L]
                               uncertainty=[0.01, 0.01, 0.02], units='mol/L')},
           args=(1.0,)),
   }  # args supply the initial concentration of each run [mol/L]
   estimator = ParameterEstimation(concentration, [0.3], experiments,
                                   name_params=['k'])  # seed k [1/s]
   estimator.estimate_parameters(method='LM', verbose=False)
   fit = estimator.get_identifiability(include_mse=True)

   initial = 2.0  # [mol/L], hot-run control used for every new plan
   schedules = {'early': (0.5, 1.0), 'middle': (1.5, 2.5),
                'late': (3.0, 4.0)}  # [s], feasible times for two observations
   covariance = np.eye(2) * hot_std**2  # [(mol/L)**2]
   candidates = [
       CandidateExperiment(name, prediction_derivatives(times, initial),
                           covariance)
       for name, times in schedules.items()
   ]
   design = rank_experiments(fit, candidates, parameter_units={'k': '1/s'},
                             criterion='D-optimal')
   best = design.candidates[0]
   print([candidate.name for candidate in design.candidates])
   print(schedules[best.name], "s")
   print(f"Predicted local uncertainty volume ratio: "
         f"{np.exp(-best.log_information_gain / 2):.3f}")

.. testoutput::

   ['middle', 'late', 'early']
   (1.5, 2.5) s
   Predicted local uncertainty volume ratio: 0.902

Forwarding a calibration to PyDex
---------------------------------

``pydex_prior(fit)`` prepares the inputs for
``Designer.set_prior_fim`` in the
`García-Muñoz PyDex fork <https://github.com/salvadorgarciamunoz/pydex/tree/58ee7104c54d46d39f3ace68ff27d19729633a05>`__.
The fork uses sensitivities scaled by the nominal parameter values by
default. The adapter therefore multiplies the calibrated Fisher entry
:math:`F_{ij}` by :math:`p_i p_j`. The simulator, accepted parameter
values, and Fisher matrix all use the same parameter order. This
parameter-relative coordinate system requires nonzero nominal values.
Recompute the calibration handoff when the nominal fit changes.

The optional continuation below requires Python 3.12 or newer and the
fork at the revision linked above. It compares the information of one
whole run per plan through PyDex's public ``eval_fim`` method.
Use the union of sample times on a single PyDex control point and give
each sample in the chosen plan one unit of effort. This represents a
whole schedule because the two errors are independent. The example
checks matrix agreement with PharmaPy in PyDex's coordinates. PyDex can
also optimize allocation across
candidates. That involves specifying an effort budget and experimental
constraints for the application.

.. code-block:: python
   :name: pydex-handoff-example

   from pydex.core.designer import Designer
   from PharmaPy.ExperimentDesign import pydex_prior


   def simulate(ti_controls, sampling_times, model_parameters):
       """Return predicted concentration [mol/L] on the PyDex grid.

       Parameters
       ----------
       ti_controls : numpy.ndarray
           Initial concentration [mol/L], shape (1,).
       sampling_times : numpy.ndarray
           Observation times [s], shape (n_times,).
       model_parameters : numpy.ndarray
           Accepted or perturbed rate k [1/s], shape (1,).

       Returns
       -------
       numpy.ndarray
           Concentrations [mol/L] of shape (n_times, 1).
       """
       return concentration(model_parameters, sampling_times,
                            ti_controls[0])[:, None]


   prior, nominal = pydex_prior(fit)  # [-], [1/s]
   designer = Designer()
   designer.simulate = simulate
   designer.model_parameters = nominal
   designer.model_parameter_names = list(fit.parameter_names)
   grid = np.unique(list(schedules.values()))  # [s]
   designer.ti_controls_candidates = np.array([[initial]])  # [mol/L]
   designer.sampling_times_candidates = grid[None, :]  # [s]
   designer.error_cov = np.array([[hot_std**2]])  # [(mol/L)**2]
   designer.initialize()
   designer.set_prior_fim(prior, nominal)

   native = {candidate.name: candidate for candidate in design.candidates}
   pydex_scores = {}
   for name, times in schedules.items():
       efforts = np.isin(grid, times).astype(float)[None, :]  # [-], sample counts
       pooled = designer.eval_fim(efforts)  # [-], PyDex coordinates
       expected = prior + native[name].added_information * nominal[0]**2  # [-]
       # PyDex uses extrapolated differences at small relative steps.
       # Allow finite-difference and roundoff error against analytic decay.
       np.testing.assert_allclose(pooled, expected, rtol=1e-7, atol=0)
       pydex_scores[name] = np.linalg.slogdet(pooled)[1]  # [-]
   assert max(pydex_scores, key=pydex_scores.get) == best.name

Applying the interface to the selected cases
--------------------------------------------

Keep the candidate IDs and the associated input records together. Both
applications need the fitted model revision or rerunnable calibration
record, its confirmed convergence and local identifiability report, the
optimized parameter names, transforms, units and fixed parameter values,
and a finite list of feasible controls and observation schedules. For
each plan provide the order, time, units and physical basis of every
prospective observation, its noise covariance and scaling declaration,
and a callback that evaluates aligned, independently checked prediction
sensitivities at the accepted parameters. Keep the model validity
constraints and any simulation failure reason with the results.

* WP2 (`#360 <https://github.com/PharmaPy-org/PharmaPy/issues/360>`__)
  supplies its selected crystallization calibration
  (`#359 <https://github.com/PharmaPy-org/PharmaPy/issues/359>`__),
  admissible recipe and operating ranges, and the feasible PAT sampling
  and noise model. Retain the conservation checks used to establish that
  a candidate trajectory is feasible.
* WP3 (`#363 <https://github.com/PharmaPy-org/PharmaPy/issues/363>`__)
  supplies its selected bioreactor calibration
  (`#362 <https://github.com/PharmaPy-org/PharmaPy/issues/362>`__),
  admissible feed/control profiles and biological constraints, and the
  available PAT or offline sampling times, measurement bases and noise
  model for the chosen fitted parameters.

Report the forecast at the calibration point, the uncertainty of the
sensitivities, active bounds or unresolved directions, and failed plans.
A practical identifiability or precision improvement requires evidence
from the subsequent observations and refit.
