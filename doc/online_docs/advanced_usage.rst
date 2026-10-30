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

:code:`SimulationExec.EstimateParams` stores the wall-clock duration of the
optimization in seconds as :code:`optimization_time`. It prints the
:code:`Optimization time` line only when :code:`verbose` is true; earlier
versions printed it also with :code:`verbose=False`.

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

The pure-component absorptivities :code:`eps` are resolved by least squares at
every evaluation and are not fitted parameters. :code:`MultipleCurveResolution`
minimizes one objective, :code:`f = 1/2 r_data @ r_data + mult_penalty *
sum(max(-eps, 0)**2)`, with every optimizer: :code:`get_objective(params,
out_array=True)` returns the weighted data residuals :code:`r_data` followed
by one penalty row :code:`sqrt(2 * mult_penalty) * max(-eps, 0)` per
absorptivity, so that :code:`f = 1/2 r @ r` for this augmented vector
:code:`r`, and the analytic IPOPT gradient includes the penalty derivative. Previously
Levenberg-Marquardt ignored the penalty, the analytic scalar gradient returned
a residual sensitivity instead of the objective gradient, and IPOPT fits
failed after solving. The penalty rows are zero when every absorptivity is
nonnegative, leaving such fits unchanged to roundoff. :code:`optimize_fn`
reports the data residuals in :code:`info['fun']` and :code:`info['jac']`
(covariance, statistics, degrees of freedom and, for SciPy fits,
:code:`sens` use them only) and the penalty rows in
:code:`info['fun_penalty']` and :code:`info['jac_penalty']`. The analytic
derivatives assume concentration profiles of constant rank, including
rank-deficient ones such as proportional species, and are undefined where
the rank changes. With :code:`global_analysis=False` each experiment resolves
its own absorptivities (:code:`epsilon_mcr` then has shape
:code:`(n_experiments, n_species, n_wavelengths)`); this mode previously
raised :code:`AttributeError`. The state-update keyword of
:code:`get_objective` is now :code:`set_self`, as in
:code:`ParameterEstimation`; :code:`update_self` remains a deprecated alias
that emits :code:`DeprecationWarning`. The MCR-specific :code:`func_aux`
override, which always raised :code:`AttributeError`, was removed, so
:code:`MultipleCurveResolution.func_aux` is the inherited
:code:`ParameterEstimation.func_aux`. After a fit, :code:`y_model` holds the
curve-resolved spectra and the model values of non-spectral states (NaN where
unobserved), and :code:`StatisticsClass.get_bootsamples` adds drawn spectral
residuals to them; earlier releases reported :code:`2 * data - prediction` for
every spectral channel (issue #500).

Finite-difference Jacobians perturb a floating-point copy of the parameters,
so integer seeds such as :code:`[1]` are no longer truncated to a zero
perturbation (which gave :code:`MultipleCurveResolution` an all-zero Jacobian
and a singular fit). The default :code:`dx_finitediff` step has magnitude
:code:`1e-3 * max(|p|, 1e-6)` in parameter units. It was :code:`1e-3 * |p|`,
so it is unchanged for :code:`|p| >= 1e-6`. For :code:`0 < |p| < 1e-6` it
grows from :code:`1e-3 * |p|` to 1e-9 and points away from zero (negative for
a negative parameter, so models such as :code:`log(-p)` stay defined), so
derivatives, fits and covariances involving such parameters change; pass
:code:`dx_finitediff` to keep a smaller step. Only a zero parameter
previously had a zero step: its Jacobian column was NaN, and
Levenberg-Marquardt returned the seed with a NaN covariance; it now fits.
A user step that is zero or infinite for an estimated parameter raises
:code:`ValueError` instead of producing NaN or zero derivatives (issue #239).
Without a :code:`dx` argument, :code:`numerical_jac`,
:code:`numerical_jac_data` and :code:`numerical_jac_central` in
:code:`PharmaPy.jac_module` now use the step :code:`eps * max(|x|, 1)`
(:code:`eps` about 1.49e-8) instead of a fixed :code:`eps`: unchanged for
:code:`|x| <= 1`, and relative for larger coordinates, where the fixed step was
lost to rounding (:code:`x = 1e9` gave a zero derivative).
:code:`numerical_jacv` now perturbs the state by a relative step along
:code:`v` rather than by a unit-length displacement, so it returns the local
directional derivative (e.g. 2, not 3, for :code:`x**2` at 1 along 1).

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

:code:`StatsModule.ParallelProblem(estimator).run_fit(sample)` fits one given
dataset the same way: :code:`sample` holds one array per experiment in the
layout of one :code:`get_bootsamples` dataset, shape
:code:`(n_times, n_measured)` on the model grid (a 1-D array is accepted for a
single measured state), and the method returns the parameters fitted to it.
The fit runs on an isolated copy with the estimator's solver method and
options, the estimator is unchanged, and the model is evaluated at
:code:`params_convg` afterwards. Earlier versions set an unused :code:`y_fit`
attribute and refitted the original observations, so the sample was ignored.
NaN or infinite values are accepted only at unobserved entries, and
:code:`MultipleCurveResolution` estimators are rejected (tracked in #505).

:code:`StatisticsClass.bootstrap_confid` (arguments :code:`num_samples`,
:code:`plot` and :code:`covar_params`) propagates the linearized parameter
uncertainty into the model predictions. It draws
:code:`theta_s = params_convg + L z_s`, with :code:`L` the lower Cholesky
factor of the parameter covariance (:code:`covar_params` of the estimator by
default, as used by :code:`get_intervals`) and :code:`z_s` standard normal
draws from NumPy's global generator, so :code:`numpy.random.seed` reproduces
them. A :code:`covar_params` given by the caller must be symmetric to
:code:`sqrt(eps)` on the correlation scale; the estimator's own covariance is
symmetrized without that check. It returns
:code:`(samples, predictions, fig, axis)`: :code:`predictions` holds, for each
draw, one :code:`(n_times, n_measured)` array of model predictions per
experiment on its model grid, and :code:`fig, axis` are :code:`None` when
:code:`plot=False`. :code:`StatisticsClass.plot_boots(num_samples, ind)` plots
the :code:`get_bootsamples` datasets of experiment :code:`ind`, one subplot per
measured column against that experiment's model grid, and returns
:code:`(fig, axes, samples)` with the plotted datasets. Both methods leave the
estimator unchanged; earlier versions failed for every input. The attribute
:code:`StatisticsClass.sigma_t`, which was never assigned, has been removed.

An explicit :code:`param_seed` given to :code:`SimulationExec.SetParamEstimation`
takes precedence over the fitted unit's default seed. For a unit with kinetics
it is set on the kinetics and read back, as before. For a unit without kinetics,
such as a :code:`Filter`, it is passed to the estimator as a float copy in the
unit's estimation basis and parameter order, without conversion: alpha [m/kg]
and medium resistance [1/m], or their natural logarithms when the Filter has
:code:`log_params=True`. The unit's configured parameters are not changed. A
seed that is not a finite one-dimensional array of real numbers with one entry
per parameter raises :code:`ValueError`, as does a Filter seed that breaks the
rules of :code:`Filter.param_seed`: callable configured resistances are not
estimable, and alpha and the medium resistance (after exponentiation when
:code:`log_params=True`) must be finite and strictly positive. Earlier versions
raised :code:`AttributeError` for such units.

:code:`optimize_fn` validates :code:`method` against :code:`'LM'`,
:code:`'IPOPT'`, :code:`'scipy'` and :code:`'least_squares'` and checks that
cyipopt is installed for IPOPT before changing the estimator (issue #479). A
call that raises, during the solve or afterwards (for example when a warnings
filter turns the undefined-covariance warning below into an error), keeps the
previous fit: :code:`opt_method`, :code:`optim_options`,
:code:`optimize_flag`, :code:`params_convg`, :code:`info_opt`, :code:`y_model`,
:code:`covar_params`, :code:`correl_params`, :code:`paramest_df`, the
iteration histories and the evaluation buffers (:code:`y_runs`,
:code:`resid_runs`, :code:`residuals`, :code:`weighted_residuals`,
:code:`sens`, and :code:`epsilon_mcr` for :code:`MultipleCurveResolution`)
that :code:`StatisticsClass` resamples. Earlier releases relabelled the
previous fit with the requested method before solving and left the buffers
at the last trial evaluation. Every call now also starts new iteration
histories: after a refit, :code:`params_iter`, :code:`objfun_iter` and
:code:`paramest_df` describe that refit, whereas earlier releases kept the
first fit's history, and :code:`store_iter=False` stores no
:code:`paramest_df`.

:code:`optimize_fn` no longer modifies the :code:`optim_options` mapping it
receives (issue #476). It passes the solver a private copy, with deep copies of
the non-callable values, that also holds the entries it sets itself
(:code:`full_output` and :code:`verbose` for LM, :code:`print_level` for IPOPT,
:code:`verbose` and :code:`bounds` for SciPy). :code:`optim_options` of the
estimator stores that copy as the solver left it, the effective options of the
last fit; for IPOPT, cyipopt has converted its keys to bytes and added its own
defaults. For LM and IPOPT, earlier releases wrote these entries, and
cyipopt's rewrite, into the caller's dictionary and stored that same object, so
later edits of the caller's dictionary changed the stored options and
calibration records. SciPy fits already used a shallow copy, so only in-place
edits of nested values, such as arrays or the bounds, reached their stored
options.

When the parameters are not (locally) identifiable from the observed data,
:code:`optimize_fn` keeps the solved fit and returns an all-NaN covariance
instead of discarding the fit or reporting meaningless numbers (issue #532).
The linearized covariance is undefined when :code:`J @ J.T` of the weighted
residual Jacobian is singular or numerically singular: a parameter changes no
observed entry, or the Jacobi-scaled matrix :code:`D J J^T D`, with
:code:`D = diag((J J^T)_ii**-0.5)`, has a condition number of at least
:code:`1/eps`. Here :code:`eps` is the machine epsilon, twice the unit
roundoff :code:`u`; a computed inverse has a relative error of order
:code:`u * cond` (Golub and Van Loan, *Matrix Computations*, 4th ed., Johns
Hopkins University Press, 2013, Sec. 2.6), so such an inverse would have no
correct significant digit. The scaled matrix, and hence the test, does not
change with the units chosen for the parameters; Jacobi scaling is moreover
within a factor of the number of parameters of the best diagonal scaling (van
der Sluis, A., 1969, Condition numbers and equilibration of matrices,
Numerische Mathematik 14(1), 14-23, doi:10.1007/BF02165096). The covariance
is also undefined when the Jacobian has non-finite entries, or when
:code:`J @ J.T` passes the test but its inversion fails or gives non-finite or
negative variances. :code:`get_covariance` then stores all-NaN
:code:`covar_params` and :code:`correl_params` and emits a
:code:`RuntimeWarning` that states the cause and names the parameters
involved: the parameters without effect and the poorly determined directions,
for example :code:`0.707*a - 0.707*b` when only the product :code:`a * b` is
determined; the parameters with non-finite sensitivities; or the parameters
with invalid variances and the least-determined direction.
:code:`get_confidence_intervals`, :code:`plot_correlation`,
:code:`StatisticsClass.get_intervals`, :code:`confidence_regions`,
:code:`bootstrap_confid` and :code:`CalibrationRecord.from_estimation` raise a
:code:`ValueError` saying that the covariance is undefined and pointing to that
warning; :code:`bootstrap_params` keeps the refitted estimates of such samples.
Earlier releases raised :code:`numpy.linalg.LinAlgError` for an exactly
singular :code:`J @ J.T` (Levenberg-Marquardt inside the solver, other methods
after it, discarding the fit; bootstrap refits became NaN rows), otherwise
returned meaningless values, including negative variances, and
:code:`plot_correlation` drew an empty heatmap. The final inverse of
:code:`levenberg_marquardt` is NaN instead of raising for a singular matrix.
Well-conditioned covariances are unchanged.

IPOPT fits report Ipopt's termination outcome in :code:`info`:
:code:`success`, :code:`status`, :code:`message` and the iteration count
:code:`nit` (issue #475). An unsuccessful termination, for example at
:code:`max_iter`, emits a :code:`RuntimeWarning` naming the status and
message; the returned parameters, covariance and statistics then describe the
last iterate, not a converged estimate. Earlier releases discarded the outcome,
so such fits were indistinguishable from converged ones.

Levenberg-Marquardt and SciPy fits report their outcome the same way (issue
#352): :code:`info['success']` (bool) and :code:`info['message']` (str) for
every method. For LM, :code:`success` is True when :code:`stop_criterion` is
one of its convergence tests (:code:`'Small step'`, :code:`'Small gradient'`,
:code:`'Value of the objective function lower than tolerance'`) and False for
:code:`'Maximum iterations exceeded'`; for :code:`'scipy'` and
:code:`'least_squares'` they are SciPy's own :code:`success`, :code:`status`
and :code:`message`. A fit that stops without converging, for example at
:code:`max_fun_eval` (LM) or :code:`max_nfev` (SciPy), emits a
:code:`RuntimeWarning` naming the method, the stop reason and the evaluation
counts, and stating that the returned parameters, covariance and statistics
describe the last accepted iterate, not a converged estimate. When the
evaluation budget stopped the fit (LM's :code:`'Maximum iterations
exceeded'`, SciPy status 0), the warning also names the limiting option with
its effective value and suggests raising it; other unsuccessful SciPy
statuses, such as -2 after a callback stop, are reported with their status
and message only. The returned values themselves are unchanged. Earlier
releases did not warn: the outcome was only available as LM's
:code:`info['stop_criterion']` or SciPy's native :code:`info['success']`,
:code:`info['status']` and :code:`info['message']`. An LM call
with :code:`max_fun_eval=0` evaluates the seed only: it reports
:code:`success` False with a message saying that no iterations were
requested, and does not warn. These warnings, like IPOPT's, point at the
user's call, also through :code:`estimate_parameters`,
:code:`SimulationExec.EstimateParams` and :code:`CalibrationRecord.rerun`.
Like other warnings, they can be turned into errors by a warnings filter;
a bootstrap refit that stops without converging then aborts the run
(see :code:`StatisticsClass.bootstrap_params`).

:code:`info['tolerances']` records the stopping criteria that governed the
fit with their effective values, the given options merged over the optimizer
defaults: :code:`max_fun_eval`, :code:`eps_1`, :code:`eps_2` and
:code:`tol_fun` for LM (defaults 100, 1e-8, 1e-8 and 1e-12, read from
:code:`levenberg_marquardt`), and :code:`ftol`, :code:`xtol`, :code:`gtol`
and :code:`max_nfev` for SciPy (defaults 1e-8, 1e-8, 1e-8 and None, read from
:code:`scipy.optimize.least_squares`; :code:`max_nfev=None` selects SciPy's
automatic limit, which is recorded as None rather than as a number). For
IPOPT it holds only the Ipopt termination options present in the solver's
options, i.e. the caller's and cyipopt's :code:`tol` default; Ipopt's native
defaults, such as :code:`max_iter`, are not recorded, as in calibration
records.

Default parameter labels (:code:`name_params` omitted) are numbered over the
full parameter vector (issue #477): with :code:`optimize_flags=[False, True]`
the optimized parameter is :code:`theta_2` in :code:`name_params`,
:code:`name_params_plot`, the :code:`paramest_df` columns and
:code:`name_params_total` alike. Earlier releases numbered the optimized
parameters only, so that parameter was labelled :code:`theta_1`, the name of
the fixed parameter in :code:`name_params_total`. :code:`name_params`,
:code:`name_params_total` and the :code:`paramest_df` columns are unchanged
when every parameter is optimized or :code:`name_params` is given. Default
plot labels (:code:`name_params_plot`) now brace the index, e.g.
:code:`$\theta_{10}$`, so that multi-digit indices form one subscript
(earlier :code:`$\theta_10$` rendered as theta_1 followed by 0); single-digit
labels render as before.

Parameter uncertainty and identifiability
-----------------------------------------

:code:`ParameterEstimation.get_identifiability(alpha=0.95, include_mse=True,
jacobian_rtol=None, output_rtol=1e-6, param_scale=None)` returns an
immutable, picklable :code:`StatsModule.IdentifiabilityReport` of an
accepted fit (issue #352). It is built from :code:`info_opt['jac']`,
:code:`info_opt['fun']`, :code:`params_convg`, the observations and the
weights only: nothing is refitted and the
estimator, including :code:`covar_params`, is not modified.
:code:`MultipleCurveResolution` inherits it. The report contains:

- :code:`noise_model` (:code:`'declared_uncertainty'`,
  :code:`'weight_matrix'` or :code:`'identity'`) and
  :code:`noise_assumption`, a statement built from the estimator: independent
  (or, with a :code:`weight_matrix`, correlated) zero-mean Gaussian
  measurement errors; declared uncertainties treated as relative weights
  rescaled by the residual mean square (:code:`include_mse=True`, like
  :code:`scipy.optimize.curve_fit(..., absolute_sigma=False)`) or as
  absolute (:code:`include_mse=False`); a supplied identity
  :code:`weight_matrix` as errors of standard deviation 1 in each measured
  unit; for no weighting of measurements in different declared units, that
  the covariance depends on the unit choice and is not a physical
  covariance; and that the analysis is local and linearized (Gauss-Newton)
  at the accepted parameters, assuming a correct model, an interior optimum
  and parameter-independent weights.
- :code:`information_matrix` (:code:`J @ J.T`) and
  :code:`fisher_information` (divided by the residual mean square when the
  scale is estimated), the hand-off to experiment design (issue #355).
- The covariance, correlation, standard errors, relative standard errors and
  Student-t intervals of :code:`get_covariance` and
  :code:`get_confidence_intervals`, bit for bit; :code:`t_ratios` and
  :code:`significant`, a test of each estimate against zero, which is not an
  identifiability test. When the variance scale cannot be estimated (a zero
  or non-finite residual mean square with :code:`include_mse=True`, e.g.
  noise-free synthetic data) or the covariance overflows or underflows to
  a non-positive variance, these are NaN and
  :code:`covariance_diagnostic` says why; :code:`get_covariance` itself keeps
  its earlier behaviour.
- :code:`active_bounds`, the parameters at an active bound of
  :code:`scipy.optimize.least_squares` (:code:`info_opt['active_mask']`),
  whose intervals and significance are not reported because the linearized
  analysis assumes an interior optimum; None for Levenberg-Marquardt, which
  ignores bounds (#533), and IPOPT, which stores no active set.
- The singular values of the Jacobi-scaled Jacobian :code:`D J` (unit row
  norms, as in the #532 test), the scaled condition number and the
  collinearity index :code:`1 / s_min` of Brun, Reichert and Kuensch (2001,
  Water Resources Research 37(4), 1015-1030, doi:10.1029/2000WR900350),
  reported without a threshold; the :code:`weakest_direction`; the
  parameters with zero information; and the :code:`unresolved_directions`.
- :code:`convergence`, the optimizer's :code:`success`, :code:`message` and
  :code:`tolerances`.

Whether the data resolve a direction depends on how accurate the
sensitivities are. In the row-normalized coordinates of :code:`D J`, the
error bound of row :code:`i` (:code:`sensitivity_error_bounds`) is
:code:`jacobian_rtol_i`, plus, for finite-difference Jacobians only,
:code:`2 * output_error_norm / (h_i * ||J_i||)`. :code:`jacobian_rtol` (one
value, or one per parameter) is the accuracy of each sensitivity row
relative to itself: machine epsilon by default for model-returned
sensitivities and :code:`jac_fun` (pass the solver :code:`rtol` for
sensitivities integrated by an ODE solver, which is how CVODES controls
their error), and the relative forward-difference step :code:`h_i /
scale_i` by default for finite differences, the truncation term of Nocedal
and Wright (*Numerical Optimization*, 2nd ed., Springer, 2006, Sec. 8.1,
eq. 8.5) under a well-scaled curvature; it does not depend on additive
offsets of the outputs. :code:`scale_i` is :code:`|estimate|`, or
:code:`param_scale`, which that default needs for an estimate at exactly
zero. The second term is the rounding term of the same equation: the
forward difference subtracts two model outputs, each accurate to
:code:`output_rtol * |y|`, so their difference may be wrong by twice that,
divided by the step :code:`h_i`. :code:`output_rtol` defaults to
:code:`jac_module.DEFAULT_OUTPUT_RTOL` (1e-6, CVode's default rtol), the
accuracy the default step :code:`sqrt(1e-6) * |p|` is designed for;
closed-form models can pass machine epsilon and ODE models their solver
:code:`rtol`. :code:`output_error_norm` whitens those output errors:
exactly, :code:`output_rtol * ||W |y|||` with :code:`y` the accepted model
outputs, for diagonal weights (declared
uncertainties, a diagonal or identity :code:`weight_matrix`), and through
the bound :code:`output_rtol * ||W||_2 * ||y||` for a correlated
:code:`weight_matrix`, with :code:`||W||_2` the largest singular value of
the whitening factors used, because whitening can amplify errors in
directions the predictions do not occupy. Norms are computed scaled, so
magnitudes near the floating-point limits neither overflow nor underflow.

A row whose error bound is at least 1 (its error may be as large as the row
itself) is consistent with a zero row under the declared accuracy: that
parameter is listed in :code:`unresolved_parameters` (distinct from the
exactly zero rows of :code:`zero_information`), reported as its own
unresolved direction, and set aside from the analysis of the other rows. By
Weyl's inequality (Golub and Van Loan, *Matrix Computations*, 4th ed., 2013,
Cor. 8.6.2) the singular values of the remaining rows of :code:`D J` move
by at most the Frobenius norm of their bounds, the
:code:`resolution_floor`; a direction whose singular value is at or below
it, or which makes the covariance undefined (#532), is unresolved.
:code:`identifiable` is True only when the covariance is not undefined
(#532) and no direction or parameter is unresolved; otherwise the standard
errors and intervals are NaN (undefined covariance) or kept but flagged as
not precise. :code:`summary()` starts with any
non-convergence, then the noise assumption, active bounds, the verdict and
directions, and a parameter table; :code:`to_frame()` returns the
per-parameter table as a :code:`pandas.DataFrame`, with an
:code:`unresolved` column (unresolved and zero-information parameters and
those carried by an unresolved direction, whose :code:`significant` entry
is missing) and :code:`identifiable` and :code:`covariance_diagnostic` in
:code:`DataFrame.attrs`.

.. code-block:: python

    estimator.optimize_fn()
    report = estimator.get_identifiability()
    print(report.summary())
    table = report.to_frame()

Liquid heat capacity
====================

:code:`LiquidPhase.getCp` now defaults to the mass basis [J/kg/K], like its siblings; callers needing [J/mol/K] must pass :code:`basis='mole'`.

Property basis options
======================

The :code:`basis` option of :code:`LiquidPhase.getEnthalpy`, :code:`SolidPhase.getEnthalpy`, :code:`LiquidPhase.getDensity` and :code:`SolidPhase.getDensity`, and of the shared :code:`ThermoPhysicalManager.getEnthalpy` and :code:`ThermoPhysicalManager.getDensityMix` methods, accepts exactly :code:`'mass'` or :code:`'mole'`, matching the heat-capacity, vapor and :code:`getProps` methods. Any other value, such as :code:`'Mass'` or :code:`'molar'`, raises :code:`ValueError("basis must be 'mass' or 'mole'")` before any computation. Earlier releases raised :code:`UnboundLocalError` for a mixture enthalpy and silently returned molar species enthalpies [J/mol] and molar densities [kmol/m**3] for any value other than :code:`'mass'`. Results for :code:`'mass'` and :code:`'mole'` are unchanged. See `issue 415 <https://github.com/PharmaPy-org/PharmaPy/issues/415>`_.

The :code:`phase` option of the shared kernels is validated the same way. :code:`ThermoPhysicalManager.getCpPure`, :code:`getCpMix` and :code:`getEnthalpy` accept :code:`'liquid'`, :code:`'solid'` and :code:`'vapor'`; :code:`getDensityPure` and :code:`getDensityMix` accept only :code:`'liquid'` and :code:`'solid'`, because vapor densities come from the ideal-gas law of :code:`VaporPhase.getDensity`, not from a stored property. Any other value, such as :code:`'Liquid'`, :code:`'gas'` or :code:`None`, raises :code:`ValueError("phase must be one of ('liquid', 'solid'), got 'gas'")` (with the accepted set of the method) before any computation, as the viscosity methods do; the density methods raised :code:`UnboundLocalError` in earlier releases. The pure densities are temperature-independent constants, so :code:`getDensityPure` and :code:`getDensityMix` ignore their :code:`temp` argument and no longer need a :code:`temp` attribute. :code:`ThermoPhysicalManager.getEnthalpy(total_h=True)` without :code:`mass_frac` or :code:`mole_frac` raises a :code:`ValueError` naming the missing composition instead of an :code:`AttributeError`. Results for accepted values are unchanged. See `issue 421 <https://github.com/PharmaPy-org/PharmaPy/issues/421>`_.

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

Assigning ``Inlet`` merges the given streams into the held ones: a key that
is assigned again, such as ``extractor.Inlet = {'feed': new_feed}``,
replaces the stream held under it, and keys that are not named, such as an
earlier ``solvent``, are kept. Earlier releases kept the old stream when a
key was assigned again, so the extractor was solved with the replaced feed
(issue #449).

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

Vapor enthalpy and latent heat
==============================

:code:`VaporPhase.getEnthalpy` and :code:`VaporPhase.getHeatVaporization` accept a scalar temperature [K] or a one-dimensional array of independent temperatures, such as the nodes of a distributed model. Each (temperature, species) entry is classified separately: a species is supercritical only strictly above its critical temperature, where it contributes vapor sensible heat and no latent heat. Every other entry, including a temperature exactly equal to :code:`t_crit`, uses liquid sensible heat plus the Watson latent heat, which is zero at :code:`t_crit`. A species without critical data in the property database (its :code:`t_crit` is parsed as NaN) is treated as non-condensing, as before: liquid sensible heat and zero latent heat at every temperature, with no latent-heat data required. A scalar or one-element temperature keeps the existing shapes: a scalar mixture enthalpy, species enthalpies of shape :code:`(1, num_species)` and latent heats of shape :code:`(num_species,)`. Several temperatures give :code:`(num_temperatures,)` mixture enthalpies and :code:`(num_temperatures, num_species)` species enthalpies and latent heats. Mixture weighting accepts one composition for all temperatures or a :code:`(num_temperatures, num_species)` profile whose row :code:`i` pairs with temperature :code:`i`; every row equals the evaluation at that row's temperature and composition alone.

Earlier releases compared a temperature array elementwise with the per-species critical temperatures. Enthalpy calls therefore raised unless the number of temperatures equaled the number of species. When the counts were equal, mixture rows were silently wrong if a temperature equaled the critical temperature of the species at the same index: that species lost its sensible heat in every row. Paired composition profiles were combined by a matrix product that mixed rows. Temperature sets that straddle a critical temperature raised instead of returning values. When another species was supercritical at a scalar temperature, a species exactly at its critical temperature, or without critical data, lost its column or its sensible heat. Apart from those two cases, mixture values for scalar temperatures differ from earlier releases only by floating-point summation order. Non-finite or multidimensional temperatures, fractions without one entry per species, profiles whose row count differs from the number of temperatures, and, for a species with critical data evaluated at or below its critical temperature, non-finite :code:`t_crit`, :code:`tref_hvap` or :code:`delta_hvap` or :code:`tref_hvap >= t_crit` now raise :code:`ValueError`.

.. _missing-property-data:

Missing property data
=====================

A property that one species omits while others supply it is stored as NaN, for list-valued correlation coefficients (:code:`cp_liq`, :code:`cp_solid`, :code:`cp_vapor`, :code:`visc_liq`, :code:`p_vap`) and :code:`diffusivity` rows as for scalars such as :code:`t_crit` or :code:`henry_constant`. A species that supplies no coefficients (absent key, :code:`null` or an empty list) gets a row of NaN. When no species supplies a list-valued property (every value is :code:`null` or an empty list), the key is treated as absent. A shorter coefficient list is still padded with trailing zeros, which is exact because the absent terms of these correlations are zero; short :code:`diffusivity` rows keep the same legacy padding. A scalar for a property that another species gives as a list raises :code:`TypeError` naming the species and the property. Earlier releases stored a row of zeros for a species without data, which evaluated silently as a zero heat capacity [J/mol/K], a 1 mPa*s viscosity or a 1 Pa vapor pressure. For example, :code:`VaporPhase.getEnthalpy` returned zero vapor sensible heat for a supercritical species without :code:`cp_vapor`.

A property method that needs missing or unusable property data raises :code:`PharmaPy.ThermoModule.MissingPropertyError`, naming the species and the property, also when no species supplies the property. This is the one exception for missing or invalid property data: list-valued coefficients, scalar properties (:code:`rho_liq`, :code:`rho_solid`, :code:`t_crit`, :code:`henry_constant`, :code:`surf_tension`, :code:`visc_gas`) and the Watson latent-heat data (:code:`delta_hvap`, :code:`tref_hvap`, including a reference with :code:`tref_hvap >= t_crit`). It subclasses both :code:`AttributeError` and :code:`ValueError`, the types that earlier releases raised for these cases, so existing :code:`except` clauses keep working; earlier releases raised a plain :code:`ValueError` for missing :code:`visc_gas` and invalid Watson data, a bare :code:`AttributeError` when no species supplied one of the listed scalars (for example :code:`t_crit`, :code:`henry_constant`, :code:`rho_liq`, :code:`rho_solid` or :code:`surf_tension`), and no error at all when only some species lacked :code:`henry_constant`, :code:`rho_liq`, :code:`rho_solid` or :code:`surf_tension`, but a NaN ratio, a NaN density, volume or concentration, or a surface tension of 0 (below). Scalar properties not listed keep their earlier behavior; for example, :code:`getMolVolMix` still returns NaN for a missing :code:`mol_vol` (`issue 511 <https://github.com/PharmaPy-org/PharmaPy/issues/511>`_). A malformed file is reported by :code:`ParseDatabase` instead, as the :code:`TypeError` above. A species is needed when the method returns its value: pure-component methods such as :code:`getCpPure`, :code:`getViscosityPure` (including :code:`visc_gas` for the vapor), :code:`AntoineEquation`, :code:`getDiffusivityPure`, :code:`getDensityPure`, :code:`LiquidPhase.getSurfTensionPure` and :code:`VaporPhase.getHeatVaporization`, and species enthalpies (:code:`total_h=False`). Wherever per-species values are weighted by amounts, only species with a nonzero weight need data, and the others contribute exactly zero (never NaN times zero):

- mixture values weighted by fractions: :code:`getCp`, :code:`getEnthalpy` and liquid :code:`getViscosity`, where a species is needed if its fraction is nonzero in some composition row;
- the reactor energy balances, which decide presence structurally and never from the instantaneous states. In a transient solve a species is present if it is in the initial liquid holdup, has a nonzero stoichiometric coefficient, or is nonzero in any feed the reactor evaluates during the solve. Only the feed actually evaluated counts (from a :code:`DynamicInlet`, a connected upstream profile or the static stream, with the usual precedence), not an overridden static composition or an ignored profile. Presence is latched over the current solve, so a species whose feed stops stays present, and is reset at the start of every :code:`solve_unit` and :code:`solve_steady`. The steady PFR uses only its inlet composition (including inert species) and the stoichiometry. The balances pass this mask as the new :code:`weights` keyword of :code:`getCpPure` and :code:`getEnthalpy(total_h=False)`; inlet enthalpies are weighted by the species fed so far in the solve (latched like the holdup mask), so the inlet's own database needs data only for the species it feeds. Isothermal Batch and PFR runs skip only the heat-capacity (capacitance) term: the heat-of-reaction correction still needs :code:`cp_liq` for reacting species, and the CSTR and semibatch flow enthalpies need it for every present species. A missing-data error raised by a right-hand side during integration is re-raised as :code:`MissingPropertyError` (an :code:`AttributeError`) chained from the CVode error, which does not name the species. This relies on the failure persisting: once recorded, every later right-hand side of that solve re-raises it, so CVode cannot step around it and stops;
- :code:`getHeatOfRxn`, which needs :code:`cp_liq` only for species with a nonzero stoichiometric coefficient;
- :code:`getKeqVLE`, which needs :code:`p_vap` only for a species that, in the same row, has a nonzero :code:`x_liq` entry and is at or below its critical temperature, and :code:`henry_constant` only for a species that, in the same row, has a nonzero entry and is above it; rows pair temperature :code:`i` with composition row :code:`i`. A zero-weight species without Antoine data gets the finite ratio of zero coefficients (1 Pa) that earlier releases computed, so products such as :code:`K * x` and :code:`y / K` stay exactly zero; a zero-weight supercritical species without a Henry constant keeps the NaN ratio of earlier releases, which the batch :code:`Evaporator` and :code:`AdiabaticFlash` index out for the supercritical species they exclude, and which the units that decide presence structurally (the evaporators and flash units, and the distillation columns, below) replace with a unit ratio for an absent species. Earlier releases returned that NaN for a present species too. The new keyword-only :code:`weights` replaces :code:`x_liq` in this rule when the caller uses only some ratios: the batch :code:`Evaporator` and :code:`AdiabaticFlash` pass zero weights for the supercritical species they exclude, and the evaporators, the flash units and the dynamic distillation column (for its plates) zero weights for absent species, while the shortcut design and the steady :code:`DistillationColumn` weight by the presence of the exact compositions they evaluate (below). A database in which no species supplies :code:`henry_constant` still raises whenever some species is supercritical, now with this error; with :code:`weights`, only when some species with a nonzero weight is;
- :code:`VaporPhase.getEnthalpy`, which needs :code:`cp_vapor` only for used supercritical entries, and :code:`cp_liq` and valid Watson data (:code:`delta_hvap`, :code:`tref_hvap`) only for used entries at or below the critical temperature, so that a zero-fraction species without them gives exactly the enthalpy of the mixture without that species. Earlier releases required the Watson data of every species at or below its critical temperature, whatever its fraction; :code:`total_h=False` and :code:`getHeatVaporization` without :code:`weights` still require them for every such species (with :code:`weights`, :code:`getHeatVaporization` evaluates only the weighted species, as the weighted :code:`getDensityPure` and :code:`getDiffusivityPure` do; see the paragraph on drying and solid-liquid separation below);
- densities: :code:`getDensityMix`, :code:`LiquidPhase.getDensity`, :code:`SolidPhase.getDensity` and the liquid concentration conversions (:code:`frac_to_conc`, and the solvent closure of :code:`conc_to_frac` and :code:`mass_conc_to_frac`, which always needs the solvent's density) need :code:`rho_liq` or :code:`rho_solid` only for species with a nonzero fraction or concentration. Earlier releases returned a NaN density, phase volume and concentrations whenever any species lacked its density, even at zero fraction; now a present species without it raises, also when a :code:`LiquidPhase` is built or its composition is updated (:code:`updatePhase`), and an absent one occupies no volume. :code:`SolidPhase` and :code:`SolidStream` evaluate their density at construction only when moments or a distribution make it relate mass and volume, so a solid given by mass alone builds without :code:`rho_solid`, as in earlier releases, and only a density request reports it; :code:`SolidPhase.updatePhase` evaluates it for a distribution or mass update;
- :code:`LiquidPhase.getSurfTension`, which needs :code:`surf_tension` only for species with a nonzero mass fraction. Earlier releases weighted a missing surface tension as 0 N/m and, through :code:`getSurfTensionPure`, wrote those zeros into the stored :code:`surf_tension` array; no getter modifies stored property data now, and :code:`getSurfTensionPure` returns a copy.

A database without any :code:`t_crit` raises this error, naming every species, where a calculation classifies species by critical temperature (:code:`getKeqVLE`, :code:`VaporPhase.getEnthalpy` and :code:`getHeatVaporization`); a species whose own :code:`t_crit` is missing keeps the documented treatment (Antoine branch, non-condensing vapor). A coefficient-list property (:code:`cp_liq`, :code:`cp_solid`, :code:`cp_vapor`, :code:`visc_liq`, :code:`p_vap` or :code:`diffusivity`) that the species give as scalars (for example :code:`"cp_solid": 1600`, or :code:`{"value": 1600}`), possibly with some species omitting it, is rejected when used: :code:`ParseDatabase` loads it as one value per species, without error or warning, and every calculation that evaluates the property raises this error, whatever the fractions, naming the property, the species, their scalar values, the database and the coefficient list the correlation needs. Earlier releases read the scalars as one species with as many coefficients as there were species, which gave, for example, a solid heat capacity of 1.3e13 J/mol/K for :code:`workshop_data/Case_study_1/pure_components_ziegler.json` and silently wrong values whenever the species count matched a valid coefficient count (one species, three for Antoine, four for :code:`visc_liq`). A scalar is never read as a constant correlation, because its units and basis are not known; a constant correlation is written as a one-element list after confirming them. The workshop files :code:`pure_components_ziegler.json` (:code:`cp_solid`) and :code:`workshop_data/Case_study_2/compounds_mom.json` (:code:`visc_liq`) keep loading for their isothermal liquid uses; their placeholder values are tracked in `issue 172 <https://github.com/PharmaPy-org/PharmaPy/issues/172>`_. This follows the maintainer decision (option A, reject when used) in `issue 424 <https://github.com/PharmaPy-org/PharmaPy/issues/424>`_.

Property keys are case sensitive. :code:`PharmaPy.ThermoModule.RECOGNIZED_PROPERTY_KEYS` lists the species-record keys that PharmaPy reads, with their units in its source comment: :code:`mw`, :code:`rho_liq`, :code:`rho_solid`, :code:`cp_liq`, :code:`cp_solid`, :code:`cp_vapor`, :code:`visc_liq`, :code:`visc_gas`, :code:`p_vap`, :code:`diffusivity`, :code:`t_crit`, :code:`henry_constant`, :code:`surf_tension`, :code:`delta_hvap`, :code:`tref_hvap`, :code:`mol_vol`, :code:`ri`, :code:`qi` and :code:`qip`, plus :code:`p_crit`, the documented critical-pressure key that no current calculation reads. Every other key is stored as metadata and never used. :code:`ParseDatabase` emits :code:`PharmaPy.ThermoModule.PropertyKeyWarning`, a :code:`UserWarning`, for each key that is a near miss of a recognized key: a misspelling found in shipped databases (:code:`cp_vap`, :code:`Tc`, :code:`temp_crit`, :code:`Pc`, :code:`surface_tension`) or a key equal to a recognized one apart from letter case and the separators :code:`_`, :code:`-` and space (for example :code:`MW` or :code:`T_crit`). The warning names the key, its species and the recognized key; the near-miss value is not used, so a calculation that needs the property still reports it as missing. Metadata keys such as :code:`Formula`, :code:`CAS`, :code:`source` or :code:`units` do not warn. Among the shipped databases, only workshop files of case studies 1 and 2 use such keys; their data are left unchanged pending the data owner (`issue 172 <https://github.com/PharmaPy-org/PharmaPy/issues/172>`_). See `issue 424 <https://github.com/PharmaPy-org/PharmaPy/issues/424>`_.

Nonzero means bitwise nonzero: a round-off-level fraction or concentration counts as present. :code:`SolidPhase` and :code:`SolidStream` store a zero fraction as machine epsilon. In their default :code:`getCp` and :code:`getEnthalpy` calls, a species whose stored mass fraction is at or below that sentinel is absent and needs no :code:`cp_solid`, and in their default :code:`getDensity` call (also at construction) it needs no :code:`rho_solid`, also in solids built from another solid's stored composition, such as mixer outlets; fractions passed explicitly keep the bitwise rule. Liquid :code:`getViscosity`, :code:`getViscosityPure` and :code:`getViscosityMix` also accept a temperature array, giving one value per temperature as :code:`getEnthalpy` does; earlier releases raised a NumPy shape error. The evaporators and flash units (:code:`Evaporator`, :code:`ContinuousEvaporator`, :code:`IsothermalFlash` and :code:`AdiabaticFlash`) also decide presence structurally, never from their states, which solver iterates and round-off perturb (`issue 425 <https://github.com/PharmaPy-org/PharmaPy/issues/425>`_). A species is present if it has a nonzero fraction in the flash feed, or in the evaporator's liquid charge as it is at the first initialization (so a charge edited with :code:`updatePhase` after its assignment counts; for :code:`Evaporator`, nitrogen too when :code:`include_nitrogen=True`), or in a feed row that the evaporator evaluates with a nonzero effective flow. A feed without flow (also a zero flow that a controller returns as a 0-d NumPy array), or one shut off by the 95 % liquid-volume event, adds nothing. The evaporators do not support feed flows of shape :code:`(1,)`, such as :code:`np.array([2.0])`, yet: their initialization and energy balances fail on them (`issue 520 <https://github.com/PharmaPy-org/PharmaPy/issues/520>`_). The feed is the one evaluated (from a :code:`DynamicInlet`, a connected upstream profile or the static stream), also for the initial feed enthalpy. A fed species is present from the time of its first such evaluation until a new :code:`Phases` assignment or :code:`reset()`, so :code:`Evaporator` continuations keep it; balances evaluated at an earlier time, including the replay of earlier rows for the heat duties and the published profiles, use the presence of their own time, so a species fed later needs no data in rows before its feed. A solve that raises, in its initialization, its integration or the publication of its results (the heat-duty replay, with its bubble points and enthalpies, and the published phases, which can raise the species-named error for a species present in a row without the data they need), keeps no results: it restores the record it started with, the unit's attributes (elapsed time, terminal states, inlet mode, profiles, results, duties, outlet) and the complete state of the attached phases (temperature, fractions, amounts and derived fields), which the attempt overwrote; :code:`ContinuousEvaporator` rebuilds the record from the liquid in the drum at every initialization, which still holds the species it was fed. The units pass this mask, as :code:`PharmaPy.Presence.SpeciesPresence` weights, to the VLE ratios, liquid and vapor enthalpies, liquid densities and bubble points (whose Antoine seed also leaves absent species out) they evaluate on states, and give an absent species a unit VLE ratio, so that its balances keep it at exactly zero whatever its data, also above its critical temperature without :code:`henry_constant`. With such weights, the activity model of the VLE ratios (UNIQUAC or UNIFAC; also :code:`LiquidPhase.getActivityCoeff` with its new :code:`weights` keyword) is evaluated on the weighted species only, as if the others were absent, so an absent species needs no :code:`ri`, :code:`qi` or :code:`qip` and its interaction parameters and group counts are not used, while a weighted species without :code:`ri`, :code:`qi` or :code:`qip` raises the species-named error (UNIFAC group counts are taken as given); an :code:`AdiabaticFlash` never treats an absent species as supercritical. :code:`ThermoPhysicalManager.getEnthalpy` (also with :code:`total_h=True`), :code:`getDensityMix`, :code:`LiquidPhase.getDensity`, :code:`LiquidPhase.getBubblePoint` and :code:`VaporPhase.getEnthalpy` accept such :code:`weights`, which must have exactly the shape of the fractions (with :code:`total_h=True` for the enthalpies); any other shape, including a scalar, raises :code:`ValueError`. In the vapor enthalpy a zero-weight entry adds exactly zero, row by row. With weights, the :code:`getKeqVLE` rule that some species must supply :code:`henry_constant` counts only species with a nonzero weight. An absent species therefore needs no property data other than its molar mass :code:`mw`, which phase and stream constructors use for the mass fractions, and the result equals, to round-off, the result with any data supplied for it; the published phases and outlets (and the continuous evaporator's :code:`mole_frac` output) hold it at exactly zero instead of the solver's round-off, while the state profiles keep the solver values. A species charged, or fed with a nonzero flow at the start, without the data it needs still raises the species-named error, because the initialization checks it. A species that first needs data during the integration, for example one in a feed that switches on mid-run, is instead reported by the integrator: the batch :code:`Evaporator` and the dynamic :code:`ContinuousEvaporator` raise Assimulo's :code:`IDAError`, which neither chains nor names the missing-data error, and the steady :code:`ContinuousEvaporator` raises :code:`RuntimeError` with the missing-data error as its cause (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). A fed species that :code:`AdiabaticFlash` removes as supercritical needs no :code:`henry_constant`, as before, because its ratio is never used. For example, an evaporator charge containing a species without :code:`p_vap` at zero fraction, which the #414 change made fail, runs again; earlier releases integrated it with zero coefficients.

:code:`ContinuousHoldup` decides presence structurally as well. Its mass fractions are CVode states, whose difference-quotient Jacobian perturbs the fraction of a species that is neither charged nor fed, so its heat capacity and enthalpy are weighted by a :code:`PharmaPy.Presence.SpeciesPresence` record instead of by those fractions. A species is present if it has a nonzero mass fraction in the liquid as it is at the start of :code:`solve_unit` (which still holds every species fed in earlier segments, and also counts a charge edited with :code:`updatePhase`), or in a feed that the holdup evaluates with a nonzero mass flow, from the time of that evaluation on; a feed without flow adds nothing and needs no data for its enthalpy, and a feed flow with one entry (a 0-d or one-entry NumPy array, also from a controller) is a scalar flow. The balances and the published mass fractions use the presence that applied at their time, so a species first fed later needs no data at earlier times and is published at exactly zero in earlier rows. An absent species needs no property data other than :code:`mw`, and the published :code:`Liquid_1`, :code:`Outlet`, :code:`outputs` and :code:`result` hold it at exactly zero, while the states returned by :code:`solve_unit` keep the solver values. :code:`solve_unit` evaluates the derivatives once at the initial state before CVode starts, so a species charged, or fed with a nonzero flow at the start, without :code:`cp_liq` raises the species-named :code:`MissingPropertyError`; a species first fed later without it is reported by CVode, whose error does not name it (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). A solve that raises, in that evaluation, the integration or the publication of its results (for example because the published liquid needs the :code:`rho_liq` of a species fed during the run), keeps no results: it restores the unit's attributes (:code:`elapsed_time`, :code:`outputs`, :code:`result`, :code:`Outlet`), the presence record it started with and the complete state of :code:`Liquid_1`. :code:`ThermoPhysicalManager.getCpMix` and :code:`LiquidPhase.getCp` accept the same :code:`weights`, with exactly the shape of the fractions on the requested basis. :code:`Mixer` and the liquid :code:`DynamicCollector` need no such record: the mixer's compositions are mass-weighted sums of the inlet fractions and only its outlet temperature is iterated, and the collector integrates species inventories whose rates depend on the feed alone, so an absent species stays exactly zero and the fraction rule already needs no data for it. Their inlet enthalpies are weighted by the inlets that flow: an inlet without mass or flow, or a collector feed row without flow, adds nothing and needs no :code:`cp_liq` for the species it holds (constructing the inlet still needs their :code:`rho_liq`). A time-independent inlet composition next to an inlet that sets a time grid counts as fed if the inlet flows at any time of the grid. A slurry-fed :code:`DynamicCollector` delegates to :code:`SemibatchCryst` and follows the crystallizers' rule below.

The crystallizers (:code:`BatchCryst`, :code:`MSMPR` and :code:`SemibatchCryst`, with moment and 1D-FVM populations) also decide presence structurally. Their liquid mass concentrations are CVode states, which the balances copy into the liquid phase whose density, heat capacity and enthalpy they read, also through :code:`Slurry`. A species is present if it has a nonzero concentration in the liquid as it is at the start of :code:`solve_unit` (after any :code:`reset_states` reset), if it is the crystallizing target (dissolution can return it to a liquid that holds none), if it is the liquid's named solvent (:code:`name_solv`, which the liquid recomputes from its volume closure on every update, so it is part of the liquid by construction), or, from the time of that evaluation on, if a feed that the unit evaluates with a nonzero flow carries it. The liquid phase that the balances read, the published :code:`mass_conc` profiles, the liquid and the outlet hold a species that was absent at that time at exactly zero, while the species balances and the states returned by :code:`solve_unit` keep the solver values; an absent species therefore needs no property data other than :code:`mw`, and the result equals, to round-off, the result with any data supplied for it. The initialization (:code:`initialize_states`, also inside :code:`solve_unit`) records the feed evaluated at the start and then requires the liquid data the balances and the result publication use (:code:`rho_liq` and :code:`cp_liq`, also for isothermal units, whose heat-duty retrieval or published slurry evaluates the liquid enthalpy) for every present species, whatever its concentration, so a present species without them, also a target at zero concentration or a species fed from the start, is reported by name before CVode starts; a species first fed later without them is reported by CVode (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). A solve that raises, in its initialization, the integration or the publication of its results, keeps no results: it restores the unit's attributes, the presence record and the complete state of :code:`Liquid_1`, :code:`Solid_1` and :code:`Slurry`. :code:`MSMPR.solve_steady_state` iterates only the target concentration, and :code:`Slurry` and :code:`Cake` read the compositions stored in their phases, so they need no record of their own. A feed evaluated without flow adds no enthalpy, so the species it holds need no :code:`cp_liq` there (building the feed stream still needs their :code:`rho_liq`); :code:`SemibatchCryst` floors the feed flow at machine epsilon in its balances, and that floor carries no feed enthalpy.

The distillation and extraction units also decide presence structurally. :code:`DynamicDistillation` integrates the plate liquid mole fractions with IDA, and :code:`DynamicExtractor` the stage mole fractions of both liquids (after an equilibrium correction whose :code:`root` iterates perturb every fraction), so their kernels are weighted by a :code:`PharmaPy.Presence.SpeciesPresence` record. For :code:`DynamicDistillation`, a species is present if it has a nonzero mole fraction in the holdup (:code:`Liquid_1`) as it is at the start of :code:`solve_unit`, or, from the time of that evaluation on, in a feed that the column evaluates with a nonzero molar flow; the plate VLE ratios (also :code:`init_unit`'s bubble point and the vapor profiles of :code:`retrieve_results`) use the presence of their time, and an absent species gets the unit ratio, so it stays at zero and adds nothing to the bubble-point residual whatever its data. For :code:`DynamicExtractor`, a species is present if it has a nonzero mole fraction in the holdup charge, if it is the last-indexed species (its fractions close the normalization of both liquids under the constant-flow approximation of `issue 195 <https://github.com/PharmaPy-org/PharmaPy/issues/195>`_, so they are nonzero during transients whatever its feed), or, from the time of that evaluation on, if the feed or the solvent supplies it with a nonzero molar flow (with a stage efficiency :code:`eff` other than 1, also if the light-phase inlet holds it, flow or not, because the Murphree term of the first stage carries that inlet's composition into the stages); one record serves both liquid phases, since stage equilibrium distributes a present species into both. The default activity-based :code:`k_fun` evaluates UNIQUAC or UNIFAC on the present species only (an absent species gets the distribution ratio 1), and the stage enthalpies and the initial densities are weighted by the record; a user :code:`k_fun` receives the state fractions unchanged. Both units publish absent species at exactly zero (:code:`result`, :code:`outputs`, the outlets, and the extractor's :code:`profiles_runs`), while the states returned by :code:`solve_unit` keep the solver values; the dynamic column's vapor profiles :code:`y_vap` follow from the published liquid. A species present at the start (charged, the extractor's closure species, or fed with a nonzero flow at the start, which the extractor records before its initial density and enthalpy check) without the data it needs raises the species-named :code:`MissingPropertyError` before IDA starts; a species first fed later without it is reported by IDA (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). A solve that raises, in its initialization (for the column, also the shortcut design), the integration or the publication of its results, keeps no results: it restores the unit's attributes (operating point, :code:`elapsed_time`, :code:`fixed_vals`, :code:`target_states`, :code:`result`, :code:`outputs`, the outlets, :code:`profiles_runs`) and the presence record. The shortcut design shared by :code:`DistillationColumn` and :code:`DynamicDistillation` (:code:`global_material_bce`, :code:`get_alpha`, the Underwood and Fenske estimates) and the steady :code:`DistillationColumn` plate-to-plate calculation work on exact compositions instead (the feed, the shortcut products and the operating-line vapors), in which a species without feed has exactly zero fraction; each composition defines its own presence, with the light and heavy keys present by construction because the design uses their equilibrium ratios whatever their fractions, so a key without its VLE data now raises the species-named error instead of using a placeholder ratio. The bubble points, ratios and dew points of these calculations are weighted by that presence: a species without feed needs no :code:`p_vap`, no :code:`henry_constant` (also when it is above its critical temperature) and, with UNIQUAC or UNIFAC, no activity parameters, and it gets the unit ratio; both Underwood sums run over the present species only, since the placeholder relative volatility of an absent species could coincide with the Underwood root. When :code:`DynamicDistillation` designs its operating point (:code:`column_startup`, also inside :code:`solve_unit`) on an inlet that has no flow at the design time, the design uses the composition of the holdup instead, because a feed without flow supplies nothing; a species only in that feed then needs no data and cannot change the operating point. :code:`sorted_by_volatility` lists the species present in the feed from the most to the least volatile, followed by the absent species in database order (earlier releases ranked a species without feed by its data, or by the placeholder ratio of missing data), and only present species can make the keys non-adjacent; the flows and compositions of the design are unchanged. :code:`VaporPhase.getDewPoint` accepts the same :code:`weights` as :code:`LiquidPhase.getBubblePoint`, with exactly the shape of the mole fractions; any other shape raises :code:`ValueError`. :code:`ContinuousExtractor` and :code:`BatchExtractor` keep a species without feed at exactly zero in every successive-substitution iterate, and the default activity-based ratio of :code:`material_balance` evaluates UNIQUAC or UNIFAC on the species of the composition being split only, and a species without feed needs no :code:`ri`, :code:`qi` or :code:`qip`. :code:`ThreePhaseSettler` integrates only the settled solid concentration and evaluates its densities, viscosities and surface tensions on the compositions stored in its phases, never on solver states, so the fraction rule already needs no data for an absent species there.

The drying and solid-liquid separation units also decide presence structurally. :code:`Drying` integrates the gas mass fractions of every species and the liquid mass fractions of the volatile species (all but the carriers named in :code:`supercrit_names`) at every cake node with CVode, whose difference-quotient Jacobian perturbs the fractions of a species that is never charged, fed or evaporated, so its gas viscosity, gas and liquid heat capacities, liquid density, vapor pressures and latent heats are weighted by two :code:`PharmaPy.Presence.SpeciesPresence` records instead of by those fractions. A volatile is present in the liquid if its initial liquid fraction is nonzero at some node; nothing condenses (negative drying rates are clipped), so nothing else enters the liquid. A species is present in the gas if its initial gas fraction is nonzero at some node, if it is present in the liquid (it evaporates from the start), or, from the time of that evaluation on, if the inlet gas that the dryer evaluates carries it; the inlet gas enters with the Darcy flow, a state, so the inlet's own flow does not matter. The kernels see the state fractions with absent species at exactly zero and the records as weights: the vapor pressure (Antoine) is evaluated only for the volatiles present in the liquid (an absent volatile has zero equilibrium partial pressure), the latent heat only for them, and the gas viscosity, heat capacities and liquid density only for the present species, while the balances and a user :code:`eta_fun` receive the state fractions unchanged. An absent species therefore needs no property data other than :code:`mw` (a species only in the inlet gas needs no liquid data: no :code:`p_vap`, :code:`cp_liq`, :code:`rho_liq` or Watson data), and the result equals, to round-off, the result with any data supplied for it. The published :code:`y_gas` and :code:`x_liq` of :code:`result` hold a species that was absent at each row's time at exactly zero, while the states returned by :code:`solve_unit` keep the solver values. :code:`solve_unit` evaluates the derivatives once at the initial state before CVode starts, so a species present at the start without the data it needs raises the species-named :code:`MissingPropertyError`, also when its initial fraction is zero (a liquid volatile without :code:`cp_vapor`, or a species in the inlet gas at the start), where earlier releases left it to CVode; likewise a volatile present in the liquid now needs its :code:`cp_liq` also where the provisional overwrite of `issue 42 <https://github.com/PharmaPy-org/PharmaPy/issues/42>`_ zeroes its liquid fraction in the balances (water in a water, ethanol and nitrogen dryer), where earlier releases evaluated the liquid heat capacity on the zeroed fraction and ran without it. The liquid heat capacity of the condensed-phase energy balance now pairs the volatile fractions with their own :code:`cp_liq` whatever the species order; earlier releases paired them with other species' data unless every carrier came after every volatile in the database, which changed the condensed temperatures (`issue 527 <https://github.com/PharmaPy-org/PharmaPy/issues/527>`_); a species first fed later is reported by CVode, which can also stall at the time it enters (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). A solve that raises, in its initialization, the integration or the publication of its results, keeps no results: it restores the unit's attributes (:code:`elapsed_time`, :code:`result`, the presence records and the model diagnostics). :code:`DeliquoringStep`, :code:`Filter` and :code:`DisplacementWashing` evaluate their kernels only on the compositions stored in their phases and cakes, never on solver states (deliquoring integrates reduced saturations and concentrations, the filter only masses, and washing is analytical), so the fraction rule already governs their mixture properties; but deliquoring evaluated the pure density, and washing the diffusivity, of every species. Deliquoring now evaluates the pure density only for the species present in its remapped initial cake field: an absent species gets a zero reduction contrast, so its concentrations, the published fields, the removed liquid and the outlet hold it at exactly zero whatever its reduced state, and it needs no :code:`rho_liq`. Washing counts the species of the initial field and the wash solvent (the wash liquid is the pure solvent at its :code:`rho_liq`) as present, evaluates the analytical solution for them only and returns an absent species at exactly zero, also in the normalized profile, whose ratio :math:`(c - c_{in})/(c_0 - c_{in})` is undefined for it and which earlier releases computed from its diffusivity; an absent species needs no :code:`diffusivity`, and a wash solvent without :code:`rho_liq` raises the species-named error instead of giving NaN concentrations. :code:`getDensityPure`, :code:`getDiffusivityPure` and :code:`VaporPhase.getHeatVaporization` accept :code:`weights` of exactly the shape :code:`(num_species,)`: only the species with a nonzero weight are evaluated, and the others get exactly 0.0, a placeholder rather than a property value, whatever their data. :code:`getViscosityMix` (also :code:`VaporPhase.getViscosity`) accepts :code:`weights` with exactly the shape of the fractions it mixes and then evaluates the mixing rule on the weighted species only, as if the others were absent (the Wilke sums do not use their :code:`visc_gas`), and :code:`VaporPhase.getCp` passes :code:`weights` to :code:`getCpMix`; any other shape raises :code:`ValueError`.

With these units, every solver-based unit operation family decides presence structurally: the reactors, the evaporators and flash units, :code:`ContinuousHoldup` and the collector, the crystallizers, distillation and extraction, the solid-liquid separation units and drying (`issue 425 <https://github.com/PharmaPy-org/PharmaPy/issues/425>`_). Known limitation: a species that first needs data during an integration, for example one in a feed that switches on mid-run, can be reported by the solver instead of by the species-named error, and the integration can stall at the time the species enters (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). The paragraphs above state how each unit reports it: the reactors re-raise the :code:`MissingPropertyError` after the solver fails, the steady :code:`ContinuousEvaporator` raises :code:`RuntimeError` with it as its cause, and the other integrators raise their own error, which does not name it.

The bubble-point, bubble-pressure and dew-point solvers start from the same Antoine seed as before (for :code:`getBubblePoint` and :code:`getDewPoint` with :code:`weights`, summed over the species with a nonzero weight only), and their Newton iterates keep the earlier Antoine treatment, with zero coefficients for a species without :code:`p_vap`, so the convergence path and the root are unchanged for Antoine data. The converged root is then checked with :code:`getKeqVLE`. The Henry rule, however, applies to every iterate: a species with a nonzero fraction (or, with :code:`weights`, a nonzero weight) that is above its critical temperature at an iterate and has no :code:`henry_constant` raises :code:`MissingPropertyError` naming it, where earlier releases failed to converge (a NaN ratio). The drying model evaluates :code:`AntoineEquation` with its new :code:`idx` keyword for the volatile species only, so a carrier gas needs no Antoine data. An unknown :code:`phase` in :code:`getCpPure`, :code:`getCpMix` or :code:`ThermoPhysicalManager.getEnthalpy` raises :code:`ValueError` instead of :code:`UnboundLocalError`. :code:`LiquidPhase` and :code:`SolidPhase` no longer require :code:`cp_liq`, :code:`p_vap` or :code:`cp_solid` at construction; a method that needs them reports their absence.

Apart from the :code:`Drying` liquid heat-capacity correction of `issue 527 <https://github.com/PharmaPy-org/PharmaPy/issues/527>`_ (above), results that never used a missing row are unchanged, subject to the known limitation above for a species that first needs data during an integration (`issue 435 <https://github.com/PharmaPy-org/PharmaPy/issues/435>`_). This includes :code:`Evaporator(include_nitrogen=True)`, whose merged database leaves the user species without :code:`cp_vapor` and nitrogen without :code:`visc_liq`. Calculations that did use one now raise instead of returning a value based on zero coefficients. See `issue 414 <https://github.com/PharmaPy-org/PharmaPy/issues/414>`_.

Connected inlet interpolation
=============================

Connected inlet profiles are interpolated by :code:`PharmaPy.Connections.interpolate_inputs`, which :code:`LiquidStream.InterpolateInputs` and :code:`SlurryStream.InterpolateInputs` (used, for example, by :code:`DynamicCollector`) now share. At and after the last upstream time, scalar and array queries return the final upstream sample itself, and array queries are evaluated in the caller's order, also when unsorted. Earlier releases differed near and after the end of the profile. For scalar queries, :code:`LiquidStream.InterpolateInputs` held the second-to-last sample for queries closest to or after the last upstream time and used a two-point window ending at the second-to-last sample for queries closest to the second-to-last time, while :code:`SlurryStream.InterpolateInputs` returned the final sample at and after the last time but used a two-point window for queries closest to it. For array queries, :code:`interpolate_inputs` kept in-support rows in the caller's order but moved queries after the last upstream time to the end; the stream methods also moved them, filled them with the value of the last in-support query instead of the final sample, and raised :code:`IndexError` when every query was after the last upstream time. Newton interpolation of integer-valued profiles now keeps fractional results. The module-level :code:`PharmaPy.Streams.Interpolation` and :code:`PharmaPy.MixedPhases.Interpolation` helpers of v1.0.0 remain as deprecated aliases that emit :code:`DeprecationWarning`; they now evaluate a full local Newton window (as :code:`PharmaPy.Interpolation.local_newton_interpolation`) near the end of the profile. Use :code:`interpolate_inputs` for connected inlet profiles.

Liquid concentrations without a solvent
=======================================

A :code:`LiquidPhase` or :code:`LiquidStream` built or updated from :code:`mole_conc` [mol/L] or :code:`mass_conc` [kg/m**3] covers every species, solvent included. With :code:`name_solv`, the solvent concentration is completed so that the concentrations close ideal mixing, :code:`sum_i c_i * v_i = 1` with :code:`v_i = mw_i / rho_liq_i` the pure molar volume [L/mol], and amounts follow the ideal-mixing density as before; these results are unchanged bit for bit. Without :code:`name_solv`, the concentrations are authoritative: the moles are :code:`1000 * sum(mole_conc) * vol` [mol], the mass is :code:`sum(mass_conc) * vol` [kg] (per second for streams), whichever of mass, volume or moles is supplied, and the fractions are normalized from the concentrations. The stored state then satisfies :code:`mole_conc * vol * 1000 == mole_frac * moles` and :code:`sum(mass_conc) * vol == mass`. :code:`LiquidPhase.updatePhase` keeps these invariants for composition profiles shaped (number of points, number of species) too, point by point: a profile update without an amount retains each point's mass, and scalar or per-point amounts (shape (number of points,)) control otherwise; an amount array of another length raises :code:`PharmaPyValueError`. Earlier releases left profile amounts unchanged by a composition update, contradicting the new composition. :code:`LiquidPhase.concentration_basis` (read-only) reports which rule applies: a fraction update switches it off, a concentration update without a solvent switches it on, and an amount-only or temperature-only update keeps it. :code:`LiquidPhase.getDensity` remains the ideal-mixing property model, a function of composition only; the new :code:`LiquidPhase.getInventoryDensity` returns the density that relates the phase's own amounts, :code:`sum(mass_conc)` [kg/m**3] or :code:`sum(mole_conc)` [mol/L] on the concentration basis and :code:`getDensity` otherwise. Consumers that convert a liquid's own volume to mass or moles use :code:`getInventoryDensity`: slurries hand the known liquid volume to their liquid and split a known slurry mass with the liquid's inventory density (:code:`Slurry.getFractions(vol_basis=False)`, which also weights the mixing energy balance, and :code:`getSolidsConcentr(basis='mass')`), so a concentration-basis liquid keeps its inventory, its crystal population and its temperature in a :code:`Slurry` or :code:`SlurryStream`; a :code:`Filter` converts its filtrate, liquid holdup and cake liquid volumes with it, so it filters the charged liquid mass and leaves the cake pores filled; :code:`DeliquoringStep` and :code:`DisplacementWashing`, which integrate the pore liquid as concentrations, hand such a liquid back with the pores' liquid volume (porosity times cake volume times the width-weighted saturation) and their mean concentrations, where earlier releases rebuilt it from mass and fractions on the ideal-mixing density, off by 20 % in volume for concentrations at 0.8 of ideal mixing (issue 463; liquids given by fractions keep that hand-back); and raw-material rows report volume with it. Property evaluations inside model equations, the ideal-mixing density of a state-dependent composition in crystallizer, evaporator and drying balances and their Jacobians, keep :code:`getDensity`; only conversions of a phase's own stored amount use :code:`getInventoryDensity`. The evaporators integrate moles with ideal mixing, so a charge with non-closing authoritative concentrations keeps its moles but starts at, and occupies, its ideal-mixing volume in the evaporator model (:code:`Evaporator` and :code:`ContinuousEvaporator` seed their headspace from that volume), and the continuous evaporator's residence time is the model's ideal-mixing volume over the feed volume flow computed the same way. A :code:`ThreePhaseSettler` passes a concentration-basis bottom liquid on with its concentrations and mass flow, and its solid outlet is the settled solid concentration times that mass flow. Dynamic raw-material accounting gives fraction samples that repeat a stream's own fractions, such as an uncontrolled composition, the stream's inventory density, so a temperature-only control reports the same volume as the static row. Concentrations that sum to zero cannot hold a positive amount and raise :code:`PharmaPy.Errors.PharmaPyValueError`, a :code:`ValueError`; earlier releases stored NaN fractions and amounts.

Behavior change (issue 439): earlier releases normalized such concentrations to fractions and took mass, moles, mass and mole flows and raw-material totals from the ideal-mixing density, contradicting the stored concentrations whenever they did not close ideal mixing. For :code:`mole_conc = [0.15, 0.10, 0.02, 5.0]` mol/L in 2 L of the PFR test species, :code:`sum_i c_i * v_i = 0.435`, and the reported mass was 1.794 kg instead of the 0.781 kg the concentrations hold, a factor of 2.30; :code:`GetRawMaterials` and :code:`BatchToFlowConnector` reported the same inflated amounts, and a downstream CSTR received concentrations inflated by 2.30. Amounts, flows, raw-material rows and connected concentration conversions now follow the concentrations, so results built from non-closing concentrations without :code:`name_solv`, including CSTR and PFR outlet streams, change accordingly. Phases and streams constructed from concentrations that close ideal mixing agree with earlier releases to rounding; this covers construction only, not the results of units that update their liquid from integrated concentrations (below). Reacting CSTR outlets now conserve mass flow when the reactions do (:code:`sum_j nu_j * mw_j = 0`). Units that update their liquid from integrated concentration states without a solvent also hold concentration-basis liquids after a solve, even when they were charged with fractions: the tank reactors (:code:`BatchReactor`, :code:`SemibatchReactor` and the :code:`CSTR` holdup), which update their liquid from the integrated :code:`mole_conc` at the reactor volume, so their final liquid mass is :code:`sum(mass_conc) * vol` and is conserved when the reactions conserve mass (a fraction-charged :code:`BatchReactor` running A + B → C now ends with its charged mass, where earlier releases rebuilt the reacted liquid on the ideal-mixing density and lost several percent of it), and their :code:`Outlet`, a :code:`BatchToFlowConnector` discharge and the downstream feed concentrations change accordingly; the crystallizers (:code:`BatchCryst`, :code:`SemibatchCryst`, :code:`MSMPR`), which update their liquid from the integrated :code:`mass_conc`, and the :code:`DynamicCollector`, which seeds its slurry liquid from the inlet :code:`mass_conc`. Their liquid mass, outlet liquid flows and the downstream :code:`Filter`, washing and drying results therefore follow the sum of the mass concentrations rather than the ideal-mixing density, which makes a crystallizer's liquid-plus-solid mass balance close; results change by the amount the integrated concentrations depart from ideal mixing. In the PFR/batch tutorial (:code:`PFR_Batch_solved.ipynb`) this changes the reactor outlet mass flow by about 0.44 %, which raises the crystal and cake mass from 0.3884 kg to 0.3903 kg (+0.5 %) once the collector holds its exact inventory (issue #467); before that fix the collector's composition error amplified the change into 1.128 kg of crystals.

Connected flow conversions
==========================

Connected dynamic flows are converted between volume [m**3/s], mass [kg/s] and molar [mol/s] bases with each sample's own composition and, for vapor streams, each sample's temperature at the stored pressure. This applies in both directions: volume to mass or molar flow, and mass or molar flow to volume. For fraction profiles, liquid density is the ideal-mixing value of the database pure-component densities, which are temperature-independent constants, so liquid conversions depend on composition only. Concentration profiles are authoritative (see *Liquid concentrations without a solvent* above): a :code:`mole_conc` [mol/L] or :code:`mass_conc` [kg/m**3] sample converts with its own totals, :code:`mole_flow = 1000 * sum(mole_conc) * vol_flow` [mol/s] and :code:`mass_flow = sum(mass_conc) * vol_flow` [kg/s], and no density model is used; until issue 439 they contributed only their fractions to an ideal-mixing density. Earlier releases applied the source stream's stored final-state density to every volume-to-mass or volume-to-molar sample, so converted flows differ whenever the upstream composition varies over time; dynamic vapor conversions to volume flow also differ when the upstream temperature varies. This per-sample change does not affect static (scalar) sources; static concentration sources and static fractions that are a liquid's held state changed with issue 439 as described below, and static fractions of a composition other than the stream's stored one now use their own molar mass and ideal-mixing density instead of the stored composition's. A mass-concentration profile [kg/m**3] requested as mass fractions now yields mass fractions [-] only; earlier releases returned both mass and mole fractions, which downstream units could not consume. Dynamic conversions require a single-phase liquid or vapor stream: slurry profiles, whose densities are per phase, and unsupported composition bases now raise an error instead of returning unphysical flows. A dynamic sample whose composition is exactly zero, such as a stopped feed, converts to zero flow; non-finite samples propagate as NaN.

Static or single-sample fractions that equal the stored fractions of a transferred liquid whose concentrations are authoritative, as published by :code:`BatchToFlowConnector`, are that liquid's held state and are converted, flows and concentrations alike, with its inventory density, the sum of its mass concentrations, so a Batch → :code:`BatchToFlowConnector` → CSTR route hands the CSTR the batch concentrations. The fractions must agree within a few tens of machine epsilons; a sample of any other composition is not that liquid's state. Other static fractions equal to the transferred stream's stored fractions keep its stored ideal-mixing density and molar mass, static fractions of any other composition use their own ideal-mixing density and molar mass (as stated above), and any other dynamic fraction profile, which carries no information about absolute packing, keeps the ideal-mixing density of each sample. Static raw-material rows likewise report volume as the amount over the liquid's inventory density, and dynamic raw-material accounting (below) reports concentration times volume flow, or a concentration feed's mass flow over its total mass concentration, so concentration-based accounting, connected conversions and raw-material rows now agree whether or not the concentrations close the ideal-mixing density (issue 439).

Dynamic raw-material accounting
===============================

See also *Connected flow conversions* above for how concentration-based accounting relates to connected flow conversions.

:code:`GetRawMaterials` and the raw-feed rows of :code:`GetStreamTable` account a continuous raw inlet with a :code:`DynamicInlet` as the feed the receiving unit actually consumed. At each of the unit's result times, the accounted feed holds the controlled fields and the static stream values of the other inlet fields the unit declares, exactly as the unit reads them: for integrating units each result time is evaluated separately, as during integration, so control callables need not accept arrays, while a :code:`Mixer` feed is evaluated once on the mixer's whole time grid, as the mixer does. Totals are trapezoidal integrals over those times. Units hold different fields: :code:`DynamicCollector` liquid feeds consume :code:`mass_flow` [kg/s], :code:`mass_frac` and :code:`temp`; evaporators, dynamic distillation and dynamic extraction consume :code:`mole_flow` [mol/s], :code:`mole_frac` and :code:`temp`; CSTR, semibatch and plug-flow reactors consume :code:`vol_flow` [m**3/s], :code:`mole_conc` [mol/L] and :code:`temp`. A composition-only control therefore holds the mass flow for a collector, the molar flow for an evaporator, and the volume flow for a reactor. Species amounts are the consumed concentration times the volume flow, or the consumed fractions times the consumed flow (a volume flow is converted with the density of each sample's composition and temperature). Samples whose composition is exactly zero, such as a stopped feed, contribute no material; non-finite values are not dropped and make the totals NaN. Slurry feeds of crystallizers and collectors consume :code:`vol_flow`, liquid :code:`mass_conc` [kg/m**3] and a population (:code:`mu_n` [m**n/m**3] or :code:`distrib` [#/m**3/um]); the liquid receives :code:`vol_flow * (1 - kv * mu_3) * mass_conc` and the crystals :code:`vol_flow * kv * mu_3` times the solid density, split by the static solid composition. Both phases report the consumed slurry temperature, and their rows follow the order of the slurry's :code:`Phases`, as for static feeds. Reported fractions are averaged with the flow on the accounting basis, so fraction times total equals each species' integrated amount; the consumed temperature is reported as is when constant and otherwise as its mass-flow-weighted average on both bases; pressure is the static value because no unit consumes inlet pressure. Dynamic rows report amounts and volume but no flow rates.

A control on a field the receiving unit does not consume would change the reported totals without changing the simulation, so it raises :code:`ValueError` naming the field and the fields the unit consumes. So does a :code:`DynamicInlet` attached to a phase of a slurry inlet, which no unit reads (attach the DynamicInput to the slurry itself), a population control on a single-phase stream, and a dynamic feed to a continuous unit that declares no inlet layout (for example :code:`ContinuousExtractor`), and a dynamic slurry feed to a unit that does not consume its volume flow, liquid concentration and population (for example :code:`ContinuousHoldup`). Earlier releases raised :code:`KeyError` unless a mass or molar flow was controlled and always reported the static composition. Cases that previously returned totals and now raise include: flow-controlled feeds whose DynamicInput also, or only, controls a field the unit does not consume (for example a :code:`mass_flow` or :code:`mole_flow` control on a slurry or reactor feed, or a :code:`mole_frac` control on a reactor feed); slurry feeds controlling :code:`mass_frac` or other unconsumed fields; phase-level DynamicInlets; and feeds controlling two flows. Totals also change for composition-controlled feeds and for feeds to units that do not consume :code:`mass_flow`, and the reported temperature of feeds controlling both flow and temperature is now the consumed, mass-flow-weighted value instead of the static stream temperature. The undocumented :code:`SimulationExec.get_dynamic_raw_inputs` helper, which split a mixed inlet's dynamic flow by static phase fractions, has been removed because that split contradicts the consumed feed; it was never part of a tagged release. Use :code:`GetRawMaterials(totals=False)` for per-phase records.

An :code:`Evaporator` with :code:`stop_at_maxvol=False` shuts its inlet when the liquid reaches 95% of the drum volume, keeps it shut on continuation, and records the absolute event time in :code:`inlet_stop_time` [s]; :code:`reset()` or a new :code:`Phases` assignment clears it. Raw-material accounting stops both dynamic and static feeds of such a unit there: dynamic totals integrate over the result times before the stop and end at the stop time, and static totals multiply the flow by the fed duration. Earlier releases kept accounting the feed after the inlet was shut, including over continuations.

:code:`GetRawMaterials`, :code:`GetOPEX` and :code:`GetStreamTable` charge a static continuous raw inlet its flow times the receiving unit's fed duration. A static continuous :code:`Mixer` is an instantaneous unit: its class declares :code:`is_instantaneous = True` and its result spans 0 s, because it passes its feed on as it arrives. (:code:`BatchToFlowConnector` declares the same flag for consistency, but it has no raw inlets.) Its raw inlets are charged, deliberately conservatively, only when the unit has exactly one successor in the flowsheet graph and the latest transfer to it is verified: the source has not been re-solved since, the successor has been solved since and holds that transferred object once. Transfers are recorded per edge across :code:`SolveFlowsheet` calls, so re-solving an unrelated unit (:code:`pick_units`) keeps the totals. The fed duration is then the successor's, followed through chains of instantaneous units and capped by a consumer's :code:`inlet_stop_time`, so a consumer solved for 0 s gives a total of 0. The successor's result must also come from exactly one solve after the transfer: a unit that keeps no :code:`profiles_runs` and whose result changed, or one whose :code:`profiles_runs` holds exactly one segment that is new since the transfer (a first, fresh, :code:`reset()` or :code:`reset_states=True` run). A consumer that continues an earlier solve, such as a :code:`CSTR` or :code:`ContinuousEvaporator` keeping several segments, has no defined share of its result consumed from this transfer, because units differ in how continued results are timed (issues #278, #443). Otherwise the totals are NaN and a :code:`RuntimeWarning` names the unit and the reason: a terminal unit with no consumer; a successor that was not solved, or not solved after the transfer; a successor without a timed result, such as a steady-state flash (issue #445); a source re-solved after its outlet was transferred; a successor that no longer holds the transferred outlet as a feed, because its input was replaced by hand and the successor solved again; a successor holding the transferred outlet more than once, which only happens when it is assigned to the successor again by hand; a successor that keeps several solve segments in :code:`profiles_runs` (issues #278, #443); a successor that keeps :code:`profiles_runs` but no segment, for example after :code:`reset()` without a new solve; or a unit outside the solved flowsheet. :code:`GetOPEX` prices these NaN totals as NaN raw costs. All other units, including a dynamic unit solved for 0 s, are charged for their own time span, as before. A raw inlet of an instantaneous unit whose :code:`DynamicInput` controls no field feeds its static stream and is accounted the same way; controls on such an inlet cannot be consumed (a :code:`Mixer` without a multi-sample connected grid rejects them when solving), and controls added after the solve make raw-material accounting raise :code:`ValueError`. Batch inlets are unchanged. Earlier releases charged the raw inlets of instantaneous units for 0 s, silently reporting zero raw usage.

A :code:`Mixer` that mixes a slurry or cake reads static phase values only, so an inlet with non-empty :code:`DynamicInlet` controls raises :code:`ValueError` naming the inlet; profiled and dynamic multiphase :code:`Mixer` inputs are unsupported. Earlier releases silently ignored such controls. A :code:`Mixer` mixes crystals only with their liquid, inside a :code:`Slurry`, :code:`SlurryStream` or :code:`Cake` inlet: assigning a bare :code:`SolidPhase` or :code:`SolidStream` raises :code:`NotImplementedError` naming each such inlet by position and class, before the Mixer changes, also when a connection replaces a transferred inlet on a re-solve. Earlier releases mixed a bare solid as a liquid, so the crystals came out as dissolved solute (issue #458).

Solids mixing is instantaneous. After :code:`solve_unit`, the mixer publishes :code:`outputs`, a dictionary, and :code:`result`, a :code:`DynamicResult` whose attributes reference the same arrays (for example :code:`mixer.result.temp is mixer.outputs['temp']`), at the single time :code:`time = [0.]` [s], so a flowsheet records zero processing time for it. The published states are :code:`mass_liq` and :code:`mass_solid` ([kg] for batch phases, [kg/s] for streams, as recorded in :code:`states_di`), the liquid :code:`mass_frac` [-] in database species order, :code:`temp` [K], and the crystal population: :code:`distrib` [#/m**3/um] for a Slurry or SlurryStream outlet (for a stream, the crystal number rate [#/um/s] divided by the slurry volume flow [m**3/s]) or :code:`total_distrib` [#/um] for a Cake outlet. Each state has a leading time axis of length one, and :code:`x_cryst` holds the size grid [um], as in crystallizer results. :code:`solve_unit` still returns its five-element tuple. Earlier releases published that tuple as :code:`outputs` with no :code:`result`, so a solids mixer could not run in a flowsheet or connect to a downstream unit. A connected inlet with a single upstream sample, such as the outlet of another mixer or of a batch unit, is a constant feed: the mixer mixes its attached phase state, exactly as if the same phases were supplied directly. Whether an inlet is profiled is decided by the number of samples in its :code:`time_upstream` [s]. Earlier code counted the state names in the upstream output dictionary instead, so a connected inlet whose upstream published more than one state could not be mixed. Solids mixing is static, so a connected inlet whose :code:`time_upstream` has more than one sample raises :code:`ValueError` naming the inlet, the sample count, and the time window before any balance is evaluated.

:code:`SolidStream` keeps a :code:`vol_flow` [m**3/s] alias of its per-second solid volume :code:`vol` whenever that volume is defined, refreshed by :code:`updatePhase` together with :code:`mass_flow` [kg/s] and :code:`mole_flow` [mol/s]. For a raw continuous :code:`SlurryStream`, :code:`GetStreamTable` and :code:`GetRawMaterials(totals=False)` therefore report the solid row's :code:`vol_flow` as kv times the third moment of the crystal number rate [m**3/s] for every construction route, including :code:`SlurryStream()` whose :code:`SolidStream` carries the number rate [#/um/s]; :code:`GetRawMaterials` with totals and :code:`GetOPEX`, which total and price on the amount basis and do not show :code:`vol_flow`, complete for that route. Earlier releases raised :code:`AttributeError` in all four for a raw :code:`SlurryStream()` feed. Raw :code:`Cake` inlets of a :code:`Mixer` are reported as raw inlets, one row per phase, while a Cake transferred from an upstream unit is not a raw material. A :code:`BatchExtractor`, a :code:`BatchToFlowConnector`, a :code:`Filter` and a :code:`ContinuousHoldup` keep the charge assigned to their :code:`Phases` as their initial-holdup snapshot, a copy taken at assignment, so :code:`GetRawMaterials`, :code:`GetOPEX` and :code:`GetStreamTable` report a user charge once, with the charged amounts and composition, as the unit's :code:`Initial_holdup` (one row per phase of a filter slurry), while a charge transferred from an upstream unit is counted under that unit only; earlier releases omitted the user charge of these units (issue #456). :code:`ContinuousHoldup.Phases` takes a :code:`LiquidPhase` or a list or tuple of phases; any other object, such as a stream, raises :code:`PharmaPyTypeError`, a :code:`TypeError`, naming its type, and the held charge is kept. Earlier releases silently kept the previous charge for a stream and raised an unrelated :code:`AttributeError` for a dict or string (issue #461). The charges of :code:`DynamicDistillation`, :code:`DynamicExtractor`, :code:`ThreePhaseSettler` and :code:`Drying` are not reported yet, nor the raw :code:`Inlet_bot_phase` feed of a :code:`ThreePhaseSettler`, and the gas :code:`Inlet` of a :code:`Drying` unit is charged as a fixed amount [kg] rather than over the run (`issue 460 <https://github.com/PharmaPy-org/PharmaPy/issues/460>`_). The liquid :code:`Mixer` declares :code:`mass_frac` in :code:`states_di` as dimensionless with one entry per species, as the solids mixer does, so :code:`dim_states` sums to the packed state width; earlier releases declared units of kg and a dimension of one.

Flowsheet execution order
=========================

:code:`SimulationExec` runs the units of a flowsheet graph in a reproducible topological order that does not depend on :code:`PYTHONHASHSEED`. Units without predecessors start in the order the graph declares them; afterwards, a unit joins a first-in, first-out queue when its last predecessor has run, and the successors of a unit are considered in the order of its successor list. Successor lists must therefore be ordered sequences such as lists; sets raise :code:`TypeError`. Each unit transfers its outlet to its successors right after it runs, so a :code:`Mixer` receives its inlets in the execution order of its predecessors, and a continuous liquid :code:`Mixer` takes its evaluation grid from the first of them with more than one sample. This equals declaration order when all predecessors are sources, but not in general: for :code:`{'S1': ['P'], 'P': ['M'], 'S2': ['M'], 'M': []}` the order is :code:`S1, S2, P, M`, so :code:`M` receives the :code:`S2` inlet before the :code:`P` inlet. Earlier releases could execute independent units, and hence feed a mixer, in a different order in each Python process. Recycles are not supported: a graph with a cycle, including a unit listed as its own successor, raises :code:`PharmaPyNonImplementedError` naming the units on or downstream of the recycle. Units that appear only as successors count toward this check; earlier releases compared the scheduled units with the graph keys alone and could silently skip recycle units. Printing a :code:`SimulationResult` shows :code:`A --> B --> C` only when the units form one connected directed path; otherwise it lists one :code:`source --> destination` line per connection, in declaration order, and the bare name of each unconnected unit, instead of joining the execution order with arrows. The :code:`flowsheet` argument of :code:`SimulationExec` must be a dictionary mapping each unit name to its successors or a linear :code:`'A --> B --> C'` string; any other value, such as a list of unit names or None, raises :code:`PharmaPyTypeError`, a :code:`TypeError`, naming the received type and both accepted forms. Earlier releases failed with an :code:`UnboundLocalError` (issue #448).

One outlet cannot feed several units. :code:`SolveFlowsheet` raises :code:`PharmaPyNonImplementedError` before solving any unit when a unit lists two or more successors, including the same successor listed twice, and the message names each such unit with its successors. Every successor would otherwise receive a full copy of the outlet stream, and the flowsheet would create mass, because PharmaPy has no stream-split contract; splitting an outlet among several units is not supported yet (see per-edge routing, issue #340). Connect each unit to one successor. The check covers the whole graph, whatever :code:`pick_units` selects, because an already solved unit still transfers its outlet. Such graphs can still be built, so :code:`execution_names` and the flowsheet diagram remain available. Earlier releases silently copied the full outlet into every branch (issue #437).

Only a :code:`Mixer` combines several inlets into one mass- and energy-balanced mixture. :code:`SolveFlowsheet` raises :code:`PharmaPyValueError` before solving any unit when a unit other than a :code:`Mixer` has two or more predecessors, and the message names the unit, its class and its predecessors. A later transfer to such a unit would replace its :code:`Inlet` or :code:`Phases` or, for a :code:`BatchCryst` or :code:`Drying` unit, be stacked with the earlier phases into a holdup list that is not a mixture of the feeds (a :code:`Slurry` replaces a :code:`BatchCryst`'s phases instead), so material would be silently dropped or left unmixed. Connect the predecessors to a :code:`Mixer` and the :code:`Mixer` to the unit, when they share an amount basis. A unit fed by a holdup [kg] and a stream [kg/s] takes one connected input, since a :code:`Mixer` cannot mix an inventory with a flow rate: connect one predecessor and assign the other input directly on the unit, the charge through :code:`Phases` or the feed through :code:`Inlet` (an :code:`Evaporator` without an :code:`Inlet` is a batch unit until its feed is assigned). A :code:`DynamicCollector` collects streams only: discharge a holdup through a :code:`BatchToFlowConnector` first. A predecessor whose basis is not known yet, such as an unassigned unit or a :code:`Mixer` fed only through the flowsheet, leaves the :code:`Mixer` remedy. A :code:`DynamicExtractor` takes its :code:`'feed'` connected; assign its solvent directly with :code:`Inlet = {'solvent': stream}`, because mixing the feed with the solvent would remove the countercurrent contact. The message gives the remedy for each unit. The check covers the whole graph, whatever :code:`pick_units` selects, because an already solved predecessor still transfers its outlet. Earlier releases silently accepted such graphs: depending on the destination, all but one of the transferred streams were replaced or ignored, or the phases were stacked without mixing (issue #444).

A connection needs both units to declare what :code:`Connection` reads. A source must publish an :code:`Outlet`, :code:`outputs`, a result with a time vector (:code:`result.time` [s]), :code:`is_continuous` and :code:`oper_mode`. A destination must declare :code:`oper_mode`, one of :code:`'Batch'`, :code:`'Semibatch'` or :code:`'Continuous'` (a :code:`Mixer` derives its mode from its inlets). When stream states are converted, the source must also declare :code:`names_states_out` and the destination :code:`names_states_in`. If any of these is missing, or a destination other than a :code:`Mixer` declares another mode, such as the lowercase :code:`'batch'` of :code:`MetaModelingClass`, the transfer raises :code:`PharmaPySpecificationError`, an :code:`AttributeError`, before either unit is changed; the message names each unit, its class and the missing attributes or the unsupported mode. A destination is also expected to provide the :code:`Inlet` or :code:`Phases` through which its mode receives matter, and a source with a dictionary of outlets its :code:`default_output`; these are not checked, because every PharmaPy unit declares them. :code:`IsothermalFlash`, :code:`AdiabaticFlash` and :code:`ContinuousExtractor` declare the contract of a continuous destination, so a continuous unit such as a :code:`Mixer` can feed them, and they need only an :code:`Inlet`. They separate the stored state of the transferred stream, its final state for a dynamic source, because they are steady-state units. They publish two outlets and no time profile, so they cannot be sources, and the raw feeds of a static :code:`Mixer` that feeds one have no defined fed duration: their totals are NaN with a warning. :code:`DistillationColumn` does not declare the contract and is rejected as a destination. Earlier releases failed with an :code:`AttributeError` such as :code:`'IsothermalFlash' object has no attribute 'oper_mode'` (issue #445).

:code:`GetStreamTable` reports the two products of these units through their :code:`get_reported_outlets()` method, which returns a dictionary of named outlets and is read before :code:`Outlet`: the rows :code:`Outlet_liquid` and :code:`Outlet_vapor` for :code:`IsothermalFlash` and :code:`AdiabaticFlash` (:code:`LiquidOut` and :code:`VaporOut`), and :code:`Outlet_heavy` and :code:`Outlet_light` for :code:`ContinuousExtractor` and :code:`BatchExtractor` (:code:`Liquid_2` and :code:`Liquid_3`; when two liquids form, the heavy one is the liquid with the larger molar density [mol/m**3], as in the extractor's :code:`x_heavy` and :code:`x_light` results). This is the extractor's current classification, not necessarily the larger mass density [kg/m**3]; classifying by mass density is tracked by `issue 257 <https://github.com/PharmaPy-org/PharmaPy/issues/257>`_, which will correct these rows without renaming them. The units still publish no :code:`Outlet`, so they remain destinations only. A raw stream fed directly to one of them, without a :code:`Mixer`, has no fed duration either: :code:`GetRawMaterials` and :code:`GetStreamTable` report its flow with NaN totals and a :code:`RuntimeWarning` naming the unit. A flash fed a raw :code:`LiquidPhase`, an amount [mol] rather than a flow, reports that amount as its raw material, on both bases; earlier releases raised :code:`AttributeError` in :code:`GetRawMaterials` and :code:`GetStreamTable` for it. :code:`GetOPEX` prices these raw materials, but it still raises :code:`ValueError` from :code:`GetDuties` for a flowsheet in which no unit reports a heat duty, such as a flash alone (`issue 261 <https://github.com/PharmaPy-org/PharmaPy/issues/261>`_). The flashes declare :code:`states_di` once an :code:`Inlet` is assigned, so the unit-operation table printed with a :code:`SimulationResult` lists them as :code:`ALG` units: :code:`IsothermalFlash` with 2N + 2 algebraic unknowns, the liquid and vapor fractions of the feed :code:`frac_liq` and :code:`frac_vap` [-] and the liquid and vapor mole fractions :code:`x_liq` and :code:`y_vap` [-] of the N species, and :code:`AdiabaticFlash` with 2N + 3, adding the drum temperature :code:`temp` [K]. The adiabatic flash removes species that are supercritical at the feed temperature, such as nitrogen, from its composition unknowns, so its N counts and indexes only the other species; the metadata is refreshed at every solve. Earlier releases declared none and left the flashes out of that table (issue #457). Earlier releases failed in :code:`GetStreamTable` with :code:`AttributeError: 'IsothermalFlash' object has no attribute 'Outlet'`, and a flash-only flowsheet failed in :code:`SolveFlowsheet` with :code:`ValueError: max() arg is an empty sequence` (issue #451).

A connection also keeps the amount basis of the source outlet. A continuous source, one with :code:`is_continuous` True, hands over a stream with a flow rate [kg/s]; this includes a :code:`BatchToFlowConnector`, whose :code:`oper_mode` is :code:`'Batch'`. Any other source hands over its final holdup, an amount [kg] without a flow rate. Before the transfer changes either unit, :code:`Connection` raises :code:`PharmaPyValueError`, naming both units, the basis mismatch and the remedy, in two cases. When a batch or semibatch source feeds a continuous unit or a :code:`DynamicCollector` directly, insert a :code:`BatchToFlowConnector` between them; it discharges a single liquid holdup as a stream over a cycle time (solids are not supported yet, issue #423). When a continuous source feeds a batch unit directly, including a :code:`BatchToFlowConnector` and an :code:`Evaporator` without an :code:`Inlet`, collect the stream for a duration in a :code:`DynamicCollector` and connect the collector, whose holdup can charge the batch unit. A :code:`Mixer` destination keeps its own rules: it takes the basis of its inlets and rejects mixing inventories with flow rates. Continuous to continuous, continuous to :code:`DynamicCollector`, batch to batch (for example :code:`R01 --> CR01 --> F01`) and batch or collector to :code:`BatchToFlowConnector` routes are unchanged. Earlier releases handed the holdup or the stream over as it was, and the destination failed later with an unrelated :code:`AttributeError` or :code:`TypeError` or a solver error, or, for a :code:`BatchCryst` fed by an :code:`MSMPR`, took the slurry stream as its holdup (issue #436).

A semibatch destination other than a :code:`DynamicCollector`, such as a :code:`SemibatchReactor`, a :code:`SemibatchCryst` or an :code:`Evaporator` with an :code:`Inlet`, holds an initial charge (:code:`Phases`) and integrates a feed (:code:`Inlet`). The stream of a continuous source becomes its :code:`Inlet`; the connection takes precedence over an :code:`Inlet` assigned before, as for a continuous unit, and the unit evaluates the upstream flow profile. The unit must already be charged, because a stream is never installed as an initial holdup: an uncharged destination raises :code:`PharmaPyValueError`. The holdup of any other source charges the destination's :code:`Phases` when it holds no charge, and replaces the charge that the previous solve transferred on the same connection; the unit keeps its own :code:`Inlet`. A destination charged by the user raises :code:`PharmaPyValueError` instead of receiving a second initial charge; remove its charge, or mix both holdups in a batch :code:`Mixer` connected to it. A crystallizer charged only with its seed crystals is the exception: a transferred liquid holdup completes the charge, as for a seeded :code:`BatchCryst`, and a re-solve replaces the transferred liquid. The units keep a reference copy of a complete charge, which tells a transferred charge from the user's; a crystallizer holding only its seed has none, so a transfer is then rejected only if the unit already holds a phase of the same kind (liquid, solid or vapor) as the transferred matter, for example a liquid charged by the user. In that seeded configuration :code:`GetRawMaterials` and :code:`GetOPEX` report the seed crystals as the crystallizer's :code:`Initial_holdup` and the transferred liquid only under its source, because initial holdups are accounted per phase: a phase transferred from an upstream unit is charged to that unit, the user's phases to the crystallizer. A connection marks every phase of a transferred mixture, so a slurry holdup transferred whole, such as a seeded crystallizer's outlet charging a :code:`BatchCryst`, is charged to its sources only, and a holdup charged wholly by the user is reported in full. Earlier releases skipped the whole initial holdup when any of its phases was transferred, silently omitting the seed (issue #455). A solve grows the held seed in place, and re-solving the flowsheet keeps it while it replaces the transferred liquid. The crystallizer then keeps the copy of the seed taken when it was charged, so a re-solve still reports the seed as charged rather than the previous solve's crystals. It keeps that copy only for a phase the connection's re-transfer keeps and that is unchanged since the unit's last solve: a seed the user edits after a solve, in place or by assigning another object, and every phase of a :code:`Phases` assignment by the user, are recorded afresh as the new charge. The state that :code:`reset()` restores follows the same rule, so after a re-solve :code:`reset()` restores the user's seed and a :code:`reset_states=True` re-solve repeats the first solve; earlier releases restored the previous run's crystals (issue #466). Both errors are raised before either unit is changed. An :code:`Evaporator` is a batch unit until an :code:`Inlet` is assigned, so a stream can feed only an evaporator that has its own :code:`Inlet` and charge, and a holdup charges an evaporator with or without one. A re-solved :code:`SemibatchReactor` fed by a stream continues from its final state and clock with the new transferred feed, as a :code:`CSTR` does. A new transferred charge restarts an :code:`Evaporator`, whose :code:`Phases` setter starts a new charge. A reactor charged again by a transfer, however, keeps its clock, its earlier solve segments and its accumulated heat duty, so time-dependent controls act on the shifted clock and the duties add up over the solves; this is a known limitation, not the intended contract (`issue 454 <https://github.com/PharmaPy-org/PharmaPy/issues/454>`_). Earlier releases silently ignored the transfer to a charged semibatch unit, which kept its own :code:`Inlet`, and took a stream as the initial holdup of an uncharged one; the raw feeds of a static :code:`Mixer` feeding such a unit were therefore reported as NaN (issue #438).

:code:`SemibatchCryst` declares :code:`oper_mode = 'Semibatch'` and :code:`is_continuous = False`. It reuses the :code:`MSMPR` model but accumulates its feed, and its :code:`Outlet` is its slurry holdup. In a flowsheet its final holdup therefore charges a batch unit, such as a :code:`BatchCryst`, or is mixed by a batch :code:`Mixer`, while a continuous successor such as an :code:`MSMPR` or a :code:`DynamicCollector` is rejected as described above. A connected continuous source, such as an :code:`MSMPR`, becomes its :code:`Inlet`. :code:`GetLabor` counts it as a batch unit, so a :code:`SemibatchCryst` is now staffed with 3 operators per shift instead of 2, which raises its labor cost by one half. Earlier releases inherited :code:`oper_mode = 'Continuous'` and :code:`is_continuous = True` from :code:`MSMPR`, so its holdup was routed as a stream: a downstream :code:`Mixer` raised a misleading :code:`ValueError` about a connected profile, and an :code:`MSMPR` or :code:`DynamicCollector` failed with :code:`AttributeError` (issue #447).

Solving a flowsheet again, completely or with :code:`pick_units`, replaces each connected input instead of adding another one. Every transfer on a graph edge removes, by object identity, the copy that the previous transfer on that edge handed to the destination, and keeps the destination's own inputs, such as a :code:`Mixer`'s raw inlets, a crystallizer's seed or a dryer's gas. A connected :code:`Mixer` therefore holds its raw inlets plus one copy of each upstream outlet, in the position of the first copy, so the inlet order and the evaluation grid do not change; a :code:`BatchCryst` or :code:`Drying` holdup does not stack a second transferred phase or cake; and a :code:`DynamicExtractor` takes the latest :code:`'feed'`. A repeated solve with unchanged inputs hands every destination the same transferred flows and amounts. A :code:`Mixer` keeps the amount basis of its first inlets, so if an upstream unit changes between batch inventories [kg] and flow rates [kg/s] between solves, the transfer raises :code:`ValueError` and leaves the :code:`Mixer` unchanged; assign a new :code:`Mixer` before solving again. It is not always a fresh run, because units keep their own continuation rules: a connected :code:`DynamicCollector` refills from its seed but continues its clock from :code:`elapsed_time`, so the second result spans, for example, 10 s to 20 s with the same collected masses, and a :code:`BatchCryst` keeps its grown crystals and continues its clock. A reactor charged again by the transfer starts from the new charge but keeps its clock, its earlier solve segments and its accumulated heat duty, so its controls and duties differ from a fresh run; this is a known limitation (`issue 454 <https://github.com/PharmaPy-org/PharmaPy/issues/454>`_). Earlier releases added another copy of the upstream outlet to a connected :code:`Mixer` on every solve, mixing the upstream flow twice on the second solve, stacked transferred phases in a :code:`BatchCryst` or :code:`Drying` unit, and kept the first transferred :code:`'feed'` of a :code:`DynamicExtractor` (issue #441). A re-solved :code:`SemibatchCryst` also continues its clock, so its result spans all its runs and :code:`GetRawMaterials` charges its own raw feed for that whole span: one more run of feed per re-solve, for example twice the feed of one run after the second solve. This is a known limitation, not the intended contract (`issue 454 <https://github.com/PharmaPy-org/PharmaPy/issues/454>`_).

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

Batch, MSMPR and semibatch crystallizers accept the same callable or record form of :code:`controls`, validate records like the tank reactors (entries other than :code:`fun`, :code:`args` and :code:`kwargs` raise :code:`ValueError`, and non-sequence :code:`args` or non-dictionary :code:`kwargs` raise :code:`TypeError`, where earlier releases accepted such records; a missing :code:`fun` raises :code:`KeyError`, as before), and store copies of the records and their :code:`kwargs`, so later changes to the caller's dictionaries no longer reach the unit.

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

A :code:`DynamicCollector` fed a slurry delegates to an adiabatic :code:`SemibatchCryst` and needs two settings on the collector: :code:`KinCryst`, a crystallization kinetics object such as :code:`CrystKinetics`, and :code:`kwargs_cryst`, a mapping (e.g. dict) with :code:`'target_ind'`, the zero-based integer index [-] of the crystallizing species in the inlet liquid's species order, and :code:`'target_comp'`, its name or a list, tuple or one-dimensional array of names. :code:`target_ind` must index the first :code:`target_comp` species in that order, the species the delegate crystallizes. Other entries, such as :code:`'scale'` [-], go to the :code:`SemibatchCryst` constructor, except that the collector sets :code:`method` and :code:`adiabatic`, so those keys are rejected, and replaces any :code:`num_interp_points` entry with its own. Only crystallizer sources supply both settings through a :code:`Connection`; for any other slurry source, such as a raw :code:`SlurryStream` or a solids :code:`Mixer`, set them on the collector. There is no non-crystallizing slurry collector. :code:`solve_unit` checks the settings before it sets up the solve: :code:`ValueError` names every missing setting, a reserved key, an empty or unknown :code:`target_comp`, or a :code:`target_ind` that indexes another species; :code:`TypeError` reports a :code:`kwargs_cryst` that is not a mapping (e.g. dict), a :code:`target_comp` of another type, or a :code:`target_ind` that is not an integer. Earlier releases failed later with a :code:`TypeError`, :code:`KeyError` or :code:`AttributeError`, depending on the missing setting, and silently seeded the species at :code:`target_ind` when it disagreed with :code:`target_comp`.

The collector seeds and reads the feed liquid's mass concentration [kg/m**3] for 1D-FVM as well as moment populations. A connected slurry, that is mixed-phase (:code:`PharmaPy.MixedPhases`) matter such as a :code:`SlurryStream`, from any source hands the collector its crystallizer inputs; liquid sources keep the liquid-mixer inputs. An :code:`MSMPR` publishes the liquid :code:`mass_conc` [kg/m**3], passed through unchanged. A source that publishes liquid mass fractions [-], such as the continuous solids :code:`Mixer`, has them converted to mass concentration as the fractions times the ideal mass-basis mixing density of the database pure-component liquid densities, :code:`w_i / sum_j(w_j / rho_liq_j)` [kg/m**3], which are temperature independent. Inputs the source does not publish, such as the solids Mixer's :code:`vol_flow` [m**3/s], are read from the transferred stream. Earlier releases read a zero concentration for raw 1D-FVM slurries, and the seed temperature solve then failed to converge.

A liquid :code:`DynamicCollector` integrates the species masses [kg] and the enthalpy [J] of its holdup, scaled by its seed, and derives the published mass fractions, mass and temperature from them, the temperature by inverting the mass-basis enthalpy h(T, w) = H/m. Its holdup therefore equals the seed plus the integrated feed, species by species, and stays within the range of the feed compositions; with an adiabatic holdup the enthalpy balance closes as well. The returned states, :code:`result` and :code:`outputs` keep their names and shapes (mass fractions, mass and temperature), and :code:`sundials_opts` tolerances now apply to the scaled species masses and enthalpy. Earlier releases integrated the mass fractions, dw/dt = F/m (w_in - w), from the 0.1 s seed with CVode's default tolerances, which did not control their error: the tutorial's collected composition was [0.00295, 0.00509, 0.01553, 0.01807, 0.95835] on master instead of the inventory's [0.00223, 0.00488, 0.01555, 0.01882, 0.95852], and a 0.44 % change in its feed could move it outside the range of the feed compositions; their temperature equation also used the holdup's composition instead of the feed's in the mixing enthalpy (issue #467). The slurry path, which delegates to a :code:`SemibatchCryst`, is unchanged. A :code:`DynamicCollector` collects its :code:`Inlet` starting from a small provisional seed that :code:`solve_unit` builds from it: 0.1 s of feed for a liquid inlet, or a slurry volume of sqrt(float64 epsilon) [m**3] for a slurry inlet; :code:`Phases` reports the phases it starts from. Assigning an initial charge to :code:`Phases` therefore raises :code:`NotImplementedError` and leaves the collector unchanged; starting from a user initial holdup, and replacing the provisional seed, is not implemented yet (`issue 342 <https://github.com/PharmaPy-org/PharmaPy/issues/342>`_). Connected and raw-inlet collectors are unaffected. Earlier releases accepted the charge and silently ignored it, so it was missing from the holdup and the raw materials (issue #459).

:code:`BatchToFlowConnector` discharges a single liquid batch holdup as a :code:`LiquidStream` whose mass flow [kg/s] is the liquid holdup mass [kg] divided by :code:`cycle_time` [s] and multiplied by :code:`flow_mult` [-]. Both must be finite real numbers greater than zero: the constructor raises :code:`ValueError` naming the parameter, its value and its unit for a zero, negative, NaN or infinite value or a boolean; earlier releases accepted them and gave a negative or NaN outlet flow, or a :code:`ZeroDivisionError` when solved (issue #434). Only one liquid holdup is supported: its :code:`Phases` accept a :code:`LiquidPhase`, bare or as the only element of a list or tuple. Any other input is rejected before the connector changes, naming the offending classes: :code:`NotImplementedError` for a :code:`Slurry`, :code:`Cake`, :code:`SolidPhase` or :code:`VaporPhase` (or a subclass) and for several phases, such as two liquids or a liquid with a solid or vapor; :code:`TypeError` for streams, including :code:`LiquidStream` and :code:`SlurryStream`, and other non-phase objects; and :code:`ValueError` for an empty list or tuple. This includes a :code:`BatchCryst` connected to it in a flowsheet, whose :code:`Slurry` outlet is rejected. Earlier releases, for example, failed later with :code:`UnboundLocalError` for slurries and with :code:`TypeError: 'NoneType' object is not iterable` when a :code:`Cake` or a non-phase object was assigned to a new connector, silently discharged the previous holdup when a :code:`Cake`, a bare solid or vapor phase, a liquid stream or a non-phase object was assigned to a loaded one, and silently truncated a phase list or tuple to its first liquid, dropping its solid, vapor or second liquid. When the holdup's concentrations are authoritative (:code:`LiquidPhase.concentration_basis`, see *Liquid concentrations without a solvent*), the outlet stream is built from its :code:`mole_conc` [mol/L] and the volume flow :code:`vol / cycle_time * flow_mult` [m**3/s], so it keeps the holdup concentrations with the same mass flow; earlier releases rebuilt it from mass fractions and the ideal-mixing density (issue 439). The transfer is instantaneous, like solids mixing: :code:`result` and :code:`outputs` hold one sample at :code:`time = [0.]` [s] with :code:`temp` [K], :code:`pres` [Pa] and :code:`mass_flow` [kg/s] of shape (1,) and :code:`mass_frac` [-] of shape (1, number of species), so a flowsheet records zero processing time for the connector and a :code:`Connection` hands the downstream continuous unit a constant feed. That feed does not stop after :code:`cycle_time`: it lasts for the downstream unit's whole runtime, so to transfer exactly the holdup mass times :code:`flow_mult` [kg], run the downstream unit for :code:`cycle_time` [s]; a longer run delivers more than the holdup. Earlier releases published :code:`time = None`, and every flowsheet containing the connector failed in :code:`SolveFlowsheet` with :code:`TypeError: 'NoneType' object is not subscriptable`.

Evaporator heat duties
======================

Evaporator :code:`heat_profile` columns are powers [J/s], and :code:`heat_duty` contains their cumulative trapezoidal integrals [J] across segments since the last reset or public :code:`Phases` assignment. For batch evaporators, column 0 is heat into the drum (positive for heating) and column 1 is condensation heat (negative for cooling); the cumulative energies retain their signs. For continuous evaporators, column 0 is jacket/utility duty (positive for heat removed) and column 1 is condenser duty (negative for cooling), evaluated on the vapor-composition basis. This replaces the former liquid-composition basis, so existing zero-reflux runs also report a different column-1 value. Continuous duties accumulate signed energy first and then report its magnitude; each physical duty is counted once. Partition independence holds for a fixed trajectory: continuous :code:`solve_unit` restarts at time zero, so splitting solves need not reproduce the same trajectory.

For nitrogen-enabled semibatch evaporation, the original inlet remains attached throughout each solve. Feed controllers and connected profiles use the original condensable species order; the evaporator appends a zero nitrogen fraction after evaluating the feed. Changes made through the inlet's :code:`updatePhase` method take effect on the next segment.

Evaporator solver boundaries
============================

Flash solves verify convergence, finite scaled residuals, and phase-fraction bounds before publishing outlet phases. Tiny bound violations within the residual tolerance are clipped and the residual is rechecked; invalid or unconverged states raise :code:`RuntimeError`. :code:`residual_tolerance` is configurable and defaults to the square root of machine epsilon.

:code:`IsothermalFlash` keeps the drum settings :code:`temp_drum` [K] and :code:`pres_drum` [Pa] as given. A setting left at None follows the current :code:`Inlet`: the effective drum conditions :code:`temp` and :code:`pres` are derived again on every :code:`Inlet` assignment and at the start of every :code:`solve_unit`, so a re-assigned inlet, an inlet changed in place, or a flowsheet re-solved after a feed change is flashed, and its heat duty evaluated, at the current feed temperature and pressure, as a fresh flash would be. A value set by the user always takes precedence. Earlier releases stored the first inlet's temperature and pressure as the drum settings and silently flashed every later inlet at them (issue #452).

:code:`AdiabaticFlash` derives the fed species (nonzero feed fraction) that are supercritical at the feed temperature, which it removes from its composition unknowns, from the current :code:`Inlet` on every :code:`solve_unit`, and :code:`num_comp` stays the number of feed species. Solving the same feed again, re-attaching it, or re-solving a flowsheet that feeds the flash gives the same products, and a solve leaves the :code:`LiquidOut` and :code:`VaporOut` published by an earlier solve unchanged. Earlier releases reduced :code:`num_comp` by the number of supercritical species on every solve, so a second solve of a feed with nitrogen failed with :code:`IndexError`, and a flowsheet re-solve, which re-attaches the feed, left :code:`num_comp` reduced; the energy balance also changed the composition of the previous :code:`VaporOut` while solving (issue #319).

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

Solid enthalpy references: :code:`SolidPhase.getEnthalpy()` now honors the :code:`temp_ref` [K] stored by the :code:`SolidPhase` constructor when the method's :code:`temp_ref` is omitted; an explicit method argument still takes precedence. Previously, an omitted reference always meant 298.15 K, so a direct call on a solid constructed with a nondefault :code:`temp_ref` now returns a different sensible enthalpy. Pass :code:`temp_ref=298.15` to reproduce the former value. Mixtures do not mix references: :code:`Slurry.getEnthalpy` and :code:`Cake.getEnthalpy` evaluate the liquid and solid on one common reference, set by their own :code:`temp_ref` keyword (default 298.15 K). The :code:`Slurry` and :code:`SlurryStream` initialization energy balance, run when :code:`Phases` is assigned, exposes no reference option and always evaluates both phases at 298.15 K. Their default results are unchanged, as are the Mixer and crystallizer energy balances that call them. The :code:`SolidStream` constructor accepts the same :code:`temp_ref` keyword [K], default 298.15 K; it follows the existing constructor parameters, so positional calls keep their meaning. Earlier releases raised :code:`TypeError` for it, so every stream solid stored 298.15 K. Unit operations do not yet apply one outlet reference policy (`issue 422 <https://github.com/PharmaPy-org/PharmaPy/issues/422>`_). Outlets they construct, such as the solids :code:`Mixer`, :code:`DynamicCollector`, :code:`MSMPR` and :code:`ThreePhaseSettler` outlets, use the 298.15 K default. Outlets they copy from a holdup solid, such as the :code:`BatchCryst`, semibatch crystallizer, :code:`Filter`, :code:`DeliquoringStep` and :code:`DisplacementWashing` outlets, keep that solid's :code:`temp_ref`. The difference affects only direct :code:`getEnthalpy()` calls on an outlet solid: :code:`Slurry`, :code:`SlurryStream` and :code:`Cake` pass one common reference to the liquid and the solid, so it does not enter their energy balances. See `issue 332 <https://github.com/PharmaPy-org/PharmaPy/issues/332>`_ and `issue 412 <https://github.com/PharmaPy-org/PharmaPy/issues/412>`_.

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

Result retrieval and plotting
=============================

:code:`PharmaPy.Plotting.get_states_result`, :code:`plot_function` and :code:`plot_distrib` treat a result without function-of-state metadata (:code:`di_fstates` is None, or a unit without :code:`fstates_di`) as having none. For a result with a single time sample, :code:`get_states_result` adds the leading time axis to states published without it: a 0-d value becomes shape (1,) and a per-index vector, such as the batch :code:`Mixer`'s mass fractions, shape (1, number of index entries); states that already have a time axis, and all multi-sample results, are returned as published. Plain and indexed states of, for example, a continuous or batch liquid :code:`Mixer` or a :code:`BatchToFlowConnector` can therefore be retrieved and plotted. With caller-supplied :code:`axes` and :code:`times`, :code:`plot_distrib` writes its x label on the axes' own figure. With caller-supplied :code:`axes`, it now returns the caller's own axes object for both :code:`times` and :code:`x_vals`, as :code:`plot_function` does; with :code:`x_vals`, earlier releases returned a one-element list or a flattened array instead. Earlier releases raised :code:`TypeError: unsupported operand type(s) for |` for results without function-of-state metadata, or an :code:`AttributeError` for :code:`plot_distrib` on a unit without :code:`fstates_di`; failed on the batch Mixer's single-sample states; and raised :code:`UnboundLocalError` in :code:`plot_distrib` when :code:`axes` and :code:`times` were supplied.

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
