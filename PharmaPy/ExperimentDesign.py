"""Local information-based ranking of feasible experiments after calibration."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from PharmaPy.StatsModule import IdentifiabilityReport


@dataclass(frozen=True)
class CandidateExperiment:
    """One feasible control and sampling plan with a fixed noise model.

    Attributes
    ----------
    name : str
        Unique identifier for the declared controls and observation schedule.
    sensitivities : callable
        Takes a read-only mapping of accepted values keyed by parameter name,
        in the fitted units. Returns a mapping with exactly the same names.
        Each value is a one-dimensional vector of raw prediction derivatives
        in [observation unit / parameter unit]. Vector entries and both
        covariance axes follow the same observation order, including
        replicates. The callback evaluates the declared plan and raises an
        exception if it cannot evaluate it. It owns model state and should
        produce the same sensitivities on repeated evaluations.
    noise_covariance : array_like
        Real, finite, symmetric positive-definite matrix of shape
        (n_observations, n_observations). Entry (i, j) has units
        [observation_i unit * observation_j unit] for absolute noise. For
        independent observations use a diagonal matrix of variances.
    relative_noise : bool, optional
        If True, multiply the supplied covariance by the calibration
        residual mean square. This requires a fit with estimated variance
        scaling. The supplied covariance has the absolute covariance units
        divided by the units of that scale. Default False treats the
        supplied covariance as absolute even when the fit estimated a scale.
    """

    name: str
    sensitivities: Callable[[Mapping[str, float]], Mapping[str, ArrayLike]]
    noise_covariance: ArrayLike
    relative_noise: bool = False


@dataclass(frozen=True, eq=False)
class CandidateEvaluation:
    """Information forecast for one additional experiment.

    Attributes
    ----------
    name : str
        Candidate identifier.
    status : {'ranked', 'rank_deficient', 'failed'}
        Full numerical rank of the pooled information permits ranking.
        A singular pooled matrix is rank deficient. Evaluation or input
        errors in this candidate produce a failed result. For a rank-deficient
        baseline the pool uses its numerically resolved subspace.
    added_information : numpy.ndarray or None
        Candidate Fisher information in fitted parameter order, shape
        (n_parameters, n_parameters), entries [1 / (p_i * p_j)]. Read-only.
        None on failure.
    candidate_rank, combined_rank : int or None
        Numerical ranks [-] of the added and pooled information. None on
        failure. A rank-deficient increment may complete the baseline rank.
    score : float or None
        Log determinant of the pooled information, evaluated numerically
        in the fitted parameter units. Higher gives smaller predicted joint
        uncertainty volume in those coordinates. None unless ranked.
    log_information_gain : float or None
        Log of the pooled-to-baseline determinant ratio [-]. The predicted
        joint uncertainty volume ratio is exp(-gain / 2), since volume is
        proportional to the square root of the covariance determinant.
        None unless both baseline and pooled information have full rank.
    diagnostic : str or None
        Reason why the candidate could not be ranked, if applicable.
    """

    name: str
    status: Literal['ranked', 'rank_deficient', 'failed']
    added_information: Optional[np.ndarray] = None
    candidate_rank: Optional[int] = None
    combined_rank: Optional[int] = None
    score: Optional[float] = None
    log_information_gain: Optional[float] = None
    diagnostic: Optional[str] = None


@dataclass(frozen=True, eq=False)
class ExperimentDesignReport:
    """Ranking and the calibration evidence used to compute it.

    Attributes
    ----------
    fit : IdentifiabilityReport
        Accepted fit, including parameter names and estimates, baseline
        information, noise assumption and practical identifiability.
    parameter_units : tuple of str
        Units in the order of fit.parameter_names.
    criterion : {'D-optimal'}
        Criterion used for scoring.
    baseline_rank : int
        Numerical rank [-] of the fit's Fisher information.
    candidates : tuple of CandidateEvaluation
        Ranked candidates in descending D-optimal order, using the
        log-information gain when the baseline has full rank. Exact ties
        are resolved by name. Rank-deficient and then failed candidates
        follow, ordered by name within each status. The first entry is the
        recommendation if its status is 'ranked'.
    """

    fit: IdentifiabilityReport
    parameter_units: tuple[str, ...]
    criterion: Literal['D-optimal']
    baseline_rank: int
    candidates: tuple[CandidateEvaluation, ...]


def _rank_and_score(rows: np.ndarray,
                    rank_rtol: float) -> tuple[int, Optional[float]]:
    """Find numerical rank and, at full rank, the D-optimal score.

    Parameters
    ----------
    rows : numpy.ndarray
        Whitened sensitivity rows with parameter columns [1 / p_i].
    rank_rtol : float
        Relative cutoff [-] for singular values on unit-norm columns.

    Returns
    -------
    rank : int
        Numerical rank [-], with zero columns excluded from the SVD.
    score : float or None
        Log determinant in the original parameter units, or None if rank
        is deficient.
    """
    maxima = np.max(np.abs(rows), axis=0)  # [1 / p_i]
    observed = maxima > 0
    if not observed.any():
        return 0, None
    scaled = rows[:, observed] / maxima[observed]  # [-], avoid overflow in norms
    norms = np.linalg.norm(scaled, axis=0)  # [-]
    singular = np.linalg.svd(scaled / norms, compute_uv=False)  # [-]
    rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))  # [-]
    if rank < rows.shape[1]:
        return rank, None
    score = 2 * (np.log(singular).sum() + np.log(norms).sum()
                 + np.log(maxima).sum())  # log det in fitted units
    return rank, float(score)


def _log_gain(relative: np.ndarray, rank: int) -> float:
    """Compute the determinant-ratio gain from resolved sensitivity modes.

    Parameters
    ----------
    relative : numpy.ndarray
        Candidate sensitivity rows in baseline-whitened coordinates [-].
    rank : int
        Numerical rank bound [-] from the prospective sensitivities before
        whitening by the baseline. Whitening cannot create new modes.

    Returns
    -------
    float
        Log determinant gain [-] of the independently resolved modes.
    """
    if rank == 0:
        return 0.0
    singular = np.linalg.svd(relative, compute_uv=False)[:rank]  # [-]
    with np.errstate(divide='ignore'):
        # log1p(s**2), without squaring very large values.
        return float(np.logaddexp(0, 2 * np.log(singular)).sum())


def _rank_repair(
        normalized: np.ndarray, basis: np.ndarray, roots: np.ndarray,
        null_basis: np.ndarray, candidate_rank: int,
        rank_rtol: float) -> tuple[int, Optional[float]]:
    """Resolve directions in a singular baseline and score their information.

    Parameters
    ----------
    normalized : numpy.ndarray
        Prospective whitened sensitivity rows in Jacobi-scaled parameter
        coordinates [-].
    basis : numpy.ndarray
        Orthonormal eigenvectors [-] of retained baseline directions.
    roots : numpy.ndarray
        Square roots [-] of the corresponding eigenvalues.
    null_basis : numpy.ndarray
        Orthonormal eigenvectors [-] of discarded baseline directions.
    candidate_rank : int
        Numerical rank [-] of the prospective sensitivities.
    rank_rtol : float
        Relative singular-value cutoff [-] of the candidate.

    Returns
    -------
    int
        Pooled numerical rank [-].
    float or None
        Log determinant contribution [-] beyond the retained baseline
        eigenvalues in the scaled coordinates. None if pooled rank is
        deficient.
    """
    new_rows = normalized @ null_basis  # [-]
    # A length-n dot product has roundoff on the scale of n*eps times
    # the sum of the magnitudes of its individual products. Suppress
    # projection leakage before normalizing the projected columns.
    projection_error = rank_rtol * (np.abs(normalized) @ np.abs(null_basis))  # [-]
    new_rows[np.abs(new_rows) <= projection_error] = 0  # [-]
    new_rank, new_score = _rank_and_score(new_rows, rank_rtol)  # [-], log det
    new_rank = min(new_rank, candidate_rank)  # [-]
    combined_rank = len(roots) + new_rank  # [-]
    if new_score is None or new_rank < null_basis.shape[1]:
        return combined_rank, None

    if candidate_rank == new_rank:
        # All candidate information is already used on the null space.
        return combined_rank, new_score

    # Schur complement: the part of the retained-direction sensitivities
    # orthogonal to the new columns supplies their conditional information.
    # Column scaling preserves the projection and conditioning of the QR.
    projection_columns = new_rows / np.max(np.abs(new_rows), axis=0)  # [-]
    orthogonal, _ = np.linalg.qr(projection_columns, mode='reduced')  # [-], [-]
    known_rows = normalized @ basis  # [-]
    conditional = known_rows - orthogonal @ (orthogonal.T @ known_rows)  # [-]
    known_gain = _log_gain(conditional / roots, candidate_rank - new_rank)  # [-]
    return combined_rank, new_score + known_gain


def _whiten_sensitivities(
        candidate: CandidateExperiment, derivatives: Mapping[str, ArrayLike],
        fit: IdentifiabilityReport) -> np.ndarray:
    """Whiten aligned prospective derivatives for Fisher information.

    Parameters
    ----------
    candidate : CandidateExperiment
        Candidate with noise covariance in the declared observation order
        and noise units.
    derivatives : mapping
        Observation vectors keyed by fitted parameter name. Each entry is
        in [observation unit / parameter unit].
    fit : IdentifiabilityReport
        Accepted fit supplying the parameter order and, for relative noise,
        the residual mean square [weighted residual unit**2].

    Returns
    -------
    numpy.ndarray
        Whitened derivatives, shape (n_observations, n_parameters) with
        columns [1 / p_i] in fitted order.

    Raises
    ------
    ValueError
        If keys, observation vectors, covariance or relative noise
        declaration are invalid.
    FloatingPointError
        If whitening overflows.
    """
    if not isinstance(derivatives, Mapping) or set(derivatives) != set(
            fit.parameter_names):
        raise ValueError("Sensitivities must be keyed by exactly the fitted "
                         f"parameters {fit.parameter_names}")
    columns = [np.asarray(derivatives[name])  # [observation_i / p_j]
               for name in fit.parameter_names]
    if any(column.ndim != 1 or column.dtype.kind not in 'iuf' for column in columns):
        raise ValueError("Each sensitivity must be a real observation vector")
    jacobian = np.column_stack(columns).astype(float)  # [observation_i / p_j]
    if not jacobian.size or not np.isfinite(jacobian).all():
        raise ValueError("Sensitivities must be nonempty and finite")

    noise = np.asarray(candidate.noise_covariance)  # [observation_i * observation_j]
    shape = (jacobian.shape[0],) * 2
    if noise.dtype.kind not in 'iuf' or noise.shape != shape:
        raise ValueError(f"Noise covariance must be real with shape {shape}")
    noise = noise.astype(float)  # [observation_i * observation_j], divided by s2 if relative
    if not np.isfinite(noise).all() or not np.array_equal(noise, noise.T):
        raise ValueError("Noise covariance must be finite and symmetric")
    if not isinstance(candidate.relative_noise, bool):
        raise ValueError("relative_noise must be a bool")
    if candidate.relative_noise and not fit.variance_scale_estimated:
        raise ValueError("Relative candidate noise requires an estimated "
                         "calibration variance scale")

    with np.errstate(over='raise', divide='raise', invalid='raise',
                     under='ignore'):
        # Apply the common scale to the Cholesky factor via its square root
        # to avoid forming a second, physically scaled covariance matrix.
        try:
            factor = np.linalg.cholesky(noise)  # [observation_i], divided by sqrt(s2) if relative
        except np.linalg.LinAlgError as error:
            raise ValueError("Noise covariance must be positive definite") from error
        if candidate.relative_noise:
            factor *= np.sqrt(fit.residual_mean_square)  # [observation_i]
        return np.linalg.solve(factor, jacobian)


def _fitted_fisher_information(fit: IdentifiabilityReport) -> np.ndarray:
    """Read the absolute Fisher matrix from a converged local fit.

    Parameters
    ----------
    fit : IdentifiabilityReport
        Calibration report containing accepted parameter estimates [p_i]
        and the measurement noise assumption.

    Returns
    -------
    numpy.ndarray
        Fisher information with entries [1 / (p_i * p_j)].

    Raises
    ------
    ValueError
        If the fit has unconfirmed convergence, non-finite parameters or
        unavailable Fisher information.
    """
    if not isinstance(fit, IdentifiabilityReport):
        raise ValueError("fit must be an IdentifiabilityReport from the fitted model")
    success = None if fit.convergence is None else fit.convergence['success']
    if not isinstance(success, (bool, np.bool_)) or not success:
        raise ValueError("Experiment design requires a fit with reported convergence")
    fisher = fit.fisher_information  # [1 / (p_i * p_j)]
    if not np.isfinite(fisher).all():
        raise ValueError("Calibration Fisher information is unavailable. Check "
                         "the fit sensitivities and the residual variance scale")
    if not np.isfinite(fit.estimates).all():
        raise ValueError("Accepted calibration parameters must be finite")
    return fisher


def pydex_prior(fit: IdentifiabilityReport) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the fitted information for PyDex's set_prior_fim.

    Parameters
    ----------
    fit : IdentifiabilityReport
        Converged calibration with a physically meaningful noise model.
        The nominal parameter estimates must be finite and nonzero.

    Returns
    -------
    normalized_fisher : numpy.ndarray
        Fisher information in PyDex's parameter-relative coordinates [-],
        shape (n_parameters, n_parameters). Entry (i, j) is the calibrated
        Fisher matrix entry multiplied by the product of estimates i and j.
    model_parameters : numpy.ndarray
        A copy of the accepted nominal values, shape (n_parameters,), in
        fitted name order and fitted units [p_i].

    Raises
    ------
    ValueError
        If Fisher information or convergence is unavailable, estimates
        are invalid or zero, or the normalized Fisher matrix overflows.

    Notes
    -----
    Use the same parameter order, units and nominal values for the PyDex
    simulator and model_parameters. This follows the public normalization
    contract of Salvador García-Muñoz's PyDex fork (see the experiment
    design guide for the verified revision). The application imports PyDex.
    """
    fisher = _fitted_fisher_information(fit)  # [1 / (p_i * p_j)]
    estimates = fit.estimates.copy()  # [p_i]
    if np.any(estimates == 0):
        raise ValueError("PyDex's parameter-relative prior requires nonzero estimates")
    with np.errstate(over='ignore', invalid='ignore'):
        normalized = (fisher * estimates[:, None]) * estimates  # [-]
    if not np.isfinite(normalized).all():
        raise ValueError("PyDex's normalized Fisher information must be finite")
    # Use one triangle for bitwise symmetry after sequential multiplication.
    lower = np.tril(normalized)  # [-]
    return lower + np.tril(normalized, k=-1).T, estimates


def rank_experiments(
        fit: IdentifiabilityReport, candidates: Sequence[CandidateExperiment],
        *, parameter_units: Mapping[str, str],
        criterion: Literal['D-optimal'] = 'D-optimal') -> ExperimentDesignReport:
    """Rank a finite set of feasible next experiments at an accepted fit.

    Parameters
    ----------
    fit : IdentifiabilityReport
        Fitted model's local calibration report from
        ParameterEstimation.get_identifiability. Use a converged fit and
        specify whether its noise weights are absolute or have an estimated
        common variance scale when creating the report.
    candidates : sequence of CandidateExperiment
        Nonempty, finite candidate set with unique, nonempty names. Supply
        only controls and sampling schedules that satisfy case constraints.
    parameter_units : mapping of str to str
        Unit for each fitted name, using the same units and parameterization
        as the fit and the candidate derivatives. Use '[-]' for a
        dimensionless parameter.
    criterion : {'D-optimal'}, optional
        Maximize the log determinant of the pooled Fisher information.

    Returns
    -------
    ExperimentDesignReport
        Reproducible ordering, candidate information and ranks, predicted
        information gains where defined, and failure diagnostics.

    Raises
    ------
    ValueError
        If the criterion, fit convergence, Fisher information, unit keys,
        or candidate names or collection are invalid.

    Notes
    -----
    NumPy evaluates each supplied candidate once. Its whitened sensitivity
    Gram matrix is added to fit.fisher_information, which already converts
    estimated-scale calibration weights to Fisher information. The
    prospective errors are independent of the calibration observations.
    Covariance is treated as fixed at the accepted point. For relative
    noise the estimated calibration scale is held fixed. Numeric scores
    use a common coordinate system, so consistent parameter-unit changes
    shift every full-rank score by the same constant.

    The calibration noise model must give its Fisher information physical
    meaning. Use declared uncertainties or a measurement covariance for
    heterogeneous units. Identity weighting with an estimated scale assumes
    commensurate calibration outputs with one shared residual variance.
    Absolute identity weighting assumes unit variance in the recorded units.

    Baseline rank uses a relative singular-value cutoff of sqrt(machine
    epsilon times the calibration observation count) on unit-norm parameter
    columns. The squared cutoff accommodates roundoff in its Gram matrix.
    Prospective rank uses the SVD cutoff of machine epsilon times the
    larger dimension of the scaled sensitivity matrix. For a singular
    baseline, test the candidate projection onto the unresolved subspace
    independently of the strength of the retained calibration directions.
    For a full-rank baseline, use the determinant lemma on resolved
    candidate modes in baseline-whitened coordinates.

    The score is a local forecast of information. Numerical rank alone
    cannot establish practical identifiability. Consult the fit's
    unresolved directions and active bounds, and validate sensitivities
    over the candidate domain before interpreting a precision forecast.
    """
    if criterion != 'D-optimal':
        raise ValueError("criterion must be 'D-optimal'")
    baseline = _fitted_fisher_information(fit)  # [1 / (p_i * p_j)]
    names = fit.parameter_names
    if not isinstance(parameter_units, Mapping) or set(parameter_units) != set(names) or any(
            not isinstance(unit, str) or not unit.strip()
            for unit in parameter_units.values()):
        raise ValueError(f"Provide a unit for each fitted parameter {names}")
    if (not isinstance(candidates, Sequence) or not candidates
            or any(not isinstance(candidate, CandidateExperiment) for candidate in candidates)):
        raise ValueError("Supply a nonempty sequence of candidate experiments")
    candidate_names = [candidate.name for candidate in candidates]
    if any(not isinstance(name, str) or not name.strip()
           for name in candidate_names) or len(set(candidate_names)) != len(candidates):
        raise ValueError("Candidates must have unique, nonempty names")

    baseline_count = fit.degrees_of_freedom + len(names)  # [-]
    rank_rtol = np.sqrt(np.finfo(float).eps * baseline_count)  # [-]
    diagonal = np.diag(baseline)  # [1 / p_i**2]
    # For a zero-information parameter, use one reciprocal fitted unit.
    scale = np.sqrt(np.where(diagonal > 0, diagonal, 1.0))  # [1 / p_i]
    scaled = baseline / scale[:, None] / scale  # [-]
    eigenvalues, vectors = np.linalg.eigh((scaled + scaled.T) / 2)  # [-], [-]
    retained = eigenvalues > rank_rtol**2 * eigenvalues[-1]
    roots = np.sqrt(eigenvalues[retained])  # [-]
    basis = vectors[:, retained]  # [-]
    null_basis = vectors[:, ~retained]  # [-]
    baseline_rank = len(roots)  # [-]
    baseline_score = 2 * (np.log(roots).sum() + np.log(scale).sum())  # log det
    estimates = MappingProxyType(dict(zip(names, map(float, fit.estimates))))  # [p_i]

    results = []
    for candidate in candidates:
        try:
            derivatives = candidate.sensitivities(estimates)  # [observation_i / p_j]
        except Exception as error:
            # Candidate callbacks define domain and simulator failures.
            results.append(CandidateEvaluation(
                candidate.name, 'failed',
                diagnostic=f"Sensitivity evaluation: {type(error).__name__}: {error}"))
            continue

        try:
            whitened = _whiten_sensitivities(candidate, derivatives, fit)  # [1 / p_i]
            with np.errstate(over='raise', invalid='raise'):
                added = whitened.T @ whitened  # [1 / (p_i * p_j)]
                candidate_rtol = np.finfo(float).eps * max(whitened.shape)  # [-]
                candidate_rank, _ = _rank_and_score(whitened, candidate_rtol)  # [-]
                normalized = whitened / scale  # [-]
                gain = None  # [-], undefined if the baseline is singular
                if baseline_rank == len(names):
                    combined_rank = baseline_rank  # [-], adding PSD preserves rank
                    gain = _log_gain((normalized @ basis) / roots, candidate_rank)  # [-]
                    score = float(baseline_score + gain)  # log det
                else:
                    combined_rank, addition = _rank_repair(
                        normalized, basis, roots, null_basis,
                        candidate_rank, candidate_rtol)  # [-], log det
                    score = None if addition is None else float(baseline_score + addition)
        except (TypeError, ValueError, np.linalg.LinAlgError,
                FloatingPointError, OverflowError) as error:
            results.append(CandidateEvaluation(
                candidate.name, 'failed',
                diagnostic=f"Candidate information: {type(error).__name__}: {error}"))
            continue

        added.setflags(write=False)
        if score is None:
            results.append(CandidateEvaluation(
                candidate.name, 'rank_deficient', added, candidate_rank,
                combined_rank, diagnostic="Pooled information is rank deficient"))
        else:
            results.append(CandidateEvaluation(
                candidate.name, 'ranked', added, candidate_rank,
                combined_rank, score, gain))

    status_order = {'ranked': 0, 'rank_deficient': 1, 'failed': 2}
    results.sort(key=lambda result: (
        status_order[result.status],
        -result.log_information_gain if result.log_information_gain is not None
        else -(result.score or 0), result.name))
    return ExperimentDesignReport(fit, tuple(parameter_units[name] for name in names),
                                  criterion, baseline_rank, tuple(results))
