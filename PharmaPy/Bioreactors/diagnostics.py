"""Opt-in, read-only checks on accepted native bioreactor histories.

These functions neither run a different simulator nor modify rates, inputs or
solver tolerances. Missing physical evidence is reported, never synthesized.
"""

from collections.abc import Mapping

import numpy as np
from scipy.integrate import trapezoid

from .culture import CultureSnapshot, evaluate_rule_graph
from .rate_providers import RuleGraphRateProvider
from .observations import predict_measurements, _validate_declaration


# Normalize real native result segments for the reporting calculations.
def _segments(histories, field):
    """Validate native time/state arrays without merging event sides.

    Parameters
    ----------
    histories : sequence
        Native DynamicResult segments; times [s].
    field : str
        Native state attribute, with its declared physical units.

    Returns
    -------
    list[tuple]
        Time [s] and two-dimensional values [field unit] for each segment.

    Raises
    ------
    ValueError
        Data are empty, nonfinite, unordered, misaligned or discontinuous in time.
    """
    result = []
    for history in histories:
        times = np.asarray(history.time, dtype=float).reshape(-1)  # [s]
        values = np.asarray(getattr(history, field), dtype=float)  # [field unit]
        if values.ndim == 1:
            values = values[:, None]
        if (not times.size or values.ndim != 2 or len(values) != len(times)
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or np.any(np.diff(times) <= 0.)):
            raise ValueError(f'invalid accepted history for {field}')
        if result:
            tolerance = 8 * np.finfo(float).eps * max(1., abs(times[0]))  # [s]; native event roundoff convention
            if (abs(times[0] - result[-1][0][-1]) > tolerance
                    or values.shape[1] != result[-1][1].shape[1]):
                raise ValueError('history segments must meet in time with aligned states')
        result.append((times, values))
    if not result:
        raise ValueError('at least one accepted history is required')
    return result


# Report represented inventories and numerical balance defects explicitly.
def inventory_report(assembly, histories, *, ledger=None, jumps_kg=None,
                     budget_kg=None, quadrature_error_kg=None,
                     invariants=None, invariant_budget=None) -> dict:
    """Audit phase-mass inventories against explicitly supplied source records.

    Parameters
    ----------
    assembly : NativeBioreactorAssembly
        Supplies phase species order. Auxiliary population states are excluded;
        do not claim total elemental closure when they carry omitted material.
    histories : sequence
        Accepted native results with mass_j_liquid0 [kg] and time [s].
    ledger : Mapping, optional
        Exactly inlet, outlet, transfer and biological, each a sequence of
        rate arrays [kg/s], matching each history's time and species axes.
        Supply actual native evaluated sources, not endpoint-inferred rates.
        Explicit zero arrays declare absent terms; missing records are unauditable.
    jumps_kg : array_like, optional
        Independently calculated scheduled inventory jumps [kg], shape
        (n_segments - 1, n_species). Required across event boundaries.
    budget_kg : array_like, optional
        Positive per-species accounting acceptance budget [kg]. No default.
    quadrature_error_kg : array_like, optional
        Nonnegative per-species bound on integrated source error [kg], established
        by refinement or analytical integration, bounding both segment and total
        integration errors. No claim of a certified bound
        is made by this function. Missing bounds prevent a pass decision.
    invariants : Mapping, optional
        Invariant name -> {species: coefficient [invariant unit/kg]}. Coefficients
        must come from declared chemistry. Maps cover selected phase pools only.
    invariant_budget : Mapping, optional
        Positive acceptance budgets in each invariant's unit.

    Returns
    -------
    dict
        Accounting residuals [kg], observed/declared jumps, and optional invariant
        residuals excluding biological sources from external supply. Status is
        passed, failed, unresolved_quadrature or not_assessable. Equality of a
        supplied ledger to the trajectory is not independent biological validation.

    Raises
    ------
    ValueError
        Ledger shapes, signs, units-by-contract or budgets are invalid.
    """
    segments = _segments(histories, 'mass_j_liquid0')
    names = tuple(assembly.phase.name_species)
    shape = (len(names),)
    if segments[0][1].shape[1] != len(names):
        raise ValueError('phase species do not align with histories')
    if ledger is None:
        return {'status': 'not_assessable', 'reason': 'native source ledger not supplied'}
    if not isinstance(ledger, Mapping) or set(ledger) != {'inlet', 'outlet', 'transfer', 'biological'}:
        raise ValueError('ledger requires inlet, outlet, transfer and biological records')
    integrated = {}
    for name, records in ledger.items():
        if len(records) != len(segments):
            raise ValueError('ledger segment count differs from histories')
        totals = []  # Per-segment integrals [kg].
        for (times, masses), record in zip(segments, records):
            rates = np.asarray(record, dtype=float)  # [kg/s]
            if rates.shape != masses.shape or not np.isfinite(rates).all():
                raise ValueError('ledger must contain finite aligned rate arrays')
            if (name == 'inlet' and np.any(rates < 0.)) or (name == 'outlet' and np.any(rates > 0.)):
                raise ValueError('inlet rates must be positive; outlet rates signed negative')
            totals.append(trapezoid(rates, times, axis=0))
        integrated[name] = np.asarray(totals)  # [kg]
    if jumps_kg is None and len(segments) > 1:
        return {'status': 'not_assessable', 'reason': 'independent scheduled jumps not supplied'}
    jumps = np.zeros((0, len(names))) if jumps_kg is None else np.asarray(jumps_kg, dtype=float)  # [kg]
    if jumps.shape != (len(segments)-1, len(names)) or not np.isfinite(jumps).all():
        raise ValueError('scheduled jumps must align with segment boundaries and species')
    observed_jumps = np.array([right[1][0] - left[1][-1]
                               for left, right in zip(segments[:-1], segments[1:])]).reshape(jumps.shape)  # [kg]
    change = segments[-1][1][-1] - segments[0][1][0]  # [kg]
    segment_change = np.array([values[-1]-values[0] for _, values in segments])  # [kg]
    segment_residual = segment_change - sum(integrated.values())  # [kg]
    residual = change - sum(integrated.values()).sum(axis=0) - jumps.sum(axis=0)  # [kg]
    jump_error = observed_jumps - jumps  # [kg]
    report = {'status': 'not_assessable', 'species': names, 'scope': 'declared phase pools only',
              'residual_kg': residual.tolist(), 'segment_residual_kg': segment_residual.tolist(),
              'jump_residual_kg': jump_error.tolist(),
              'invariants': {'status': 'not_assessable', 'reason': 'composition/invariants not supplied'}}
    error = None  # [kg]; absent integration-error evidence is not zero error.
    if quadrature_error_kg is not None:
        error = np.asarray(quadrature_error_kg, dtype=float)  # [kg]
        if error.shape != shape or not np.isfinite(error).all() or np.any(error < 0.):
            raise ValueError('quadrature error must be finite nonnegative per-species values')
    if budget_kg is not None:
        budget = np.asarray(budget_kg, dtype=float)  # [kg]
        if budget.shape != shape or not np.isfinite(budget).all() or np.any(budget <= 0.):
            raise ValueError('budgets must be finite positive per-species values')
    if budget_kg is not None and error is not None:
        defect = np.max(abs(np.vstack((residual, segment_residual))), axis=0)  # [kg]
        report['status'] = ('failed' if np.any(defect-error > budget) or np.any(abs(jump_error) > budget)
                            else 'passed' if np.all(defect+error <= budget)
                            else 'unresolved_quadrature')
    elif budget_kg is not None or quadrature_error_kg is not None:
        report['reason'] = 'both accounting budget and quadrature error bound are required'
    if invariants is not None:
        if not isinstance(invariants, Mapping) or not invariants:
            raise ValueError('invariants must be a nonempty named mapping')
        if invariant_budget is not None and set(invariant_budget) != set(invariants):
            raise ValueError('invariant budgets must match the declared invariants')
        checks = {}
        for name, coefficients in invariants.items():
            if not isinstance(coefficients, Mapping) or not coefficients or set(coefficients) - set(names):
                raise ValueError('invariant coefficients must name known phase species')
            row = np.array([coefficients.get(species, 0.) for species in names], dtype=float)  # [invariant/kg]
            if not np.isfinite(row).all() or not np.any(row):
                raise ValueError('invariant coefficients must be finite and nonzero')
            external = integrated['inlet'] + integrated['outlet'] + integrated['transfer']  # [kg]
            defect = float(row @ (change - external.sum(axis=0) - jumps.sum(axis=0)))  # [invariant]
            segment_defects = (segment_change - external) @ row  # [invariant]
            item = {'residual': defect, 'segment_residual': segment_defects.tolist(), 'status': 'not_assessable'}
            if invariant_budget is not None:
                allowance = float(invariant_budget[name])  # [invariant]
                if not np.isfinite(allowance) or allowance <= 0.:
                    raise ValueError('invariant budget must be finite and positive')
                if error is not None:
                    uncertainty = float(abs(row) @ error)  # [invariant]
                    largest = max(abs(defect), float(np.max(abs(segment_defects))))  # [invariant]
                    item['status'] = ('failed' if largest-uncertainty > allowance
                                      or np.any(abs(jump_error @ row) > allowance) else
                                      'passed' if largest+uncertainty <= allowance else 'unresolved_quadrature')
            checks[name] = item
        report['invariants'] = checks
    return report


# Compare numerical refinements without changing physical assumptions.
def refinement_report(coarse, medium, fine, *, scales: Mapping, tolerance: float) -> dict:
    """Compare h, h/2 and h/4 histories at common accepted reporting times.

    Parameters
    ----------
    coarse, medium, fine : sequence
        Independently solved native histories with identical event boundaries.
        Caller refines integration steps, not merely the output grid.
    scales : Mapping
        Native field -> positive column scales in that field's units. Include
        masses, population, product and volume as relevant to the model.
    tolerance : float
        Positive dimensionless maximum normalized refinement difference [-].
        This is a user-declared numerical budget, not an industry standard.

    Returns
    -------
    dict
        Maximum normalized differences and observed reduction per field.
        Passing requires both differences within budget; agreement alone is
        not an independent reference or proof of asymptotic convergence.

    Raises
    ------
    ValueError
        Fields, scales, budget, time grids or event segmentation are incompatible.
    """
    if not scales or not np.isfinite(tolerance) or tolerance <= 0.:
        raise ValueError('positive refinement budget and field scales required')
    report = {}
    for field, declared_scale in scales.items():
        levels = [_segments(run, field) for run in (coarse, medium, fine)]
        if len({len(level) for level in levels}) != 1:
            raise ValueError('refinement must retain event segmentation')
        scale = np.atleast_1d(np.asarray(declared_scale, dtype=float))  # [field unit]
        if scale.shape != (levels[0][0][1].shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0.):
            raise ValueError('positive per-column field scales required')
        differences = []
        for left, right in zip(levels[:-1], levels[1:]):
            largest = 0.  # [-]
            for (times, values), (refined_times, refined_values) in zip(left, right):
                if values.shape[1] != refined_values.shape[1]:
                    raise ValueError('refinement must retain field column order and count')
                indices = []
                for time in times:
                    epsilon = 8 * np.finfo(float).eps * max(1., abs(time))  # [s]; native roundoff convention
                    hits = np.flatnonzero(abs(refined_times - time) <= epsilon)
                    if len(hits) != 1:
                        raise ValueError('refined history must contain every coarser time and event side')
                    indices.append(hits[0])
                if times[0] != refined_times[0] or times[-1] != refined_times[-1]:
                    raise ValueError('refinement must retain segment endpoints')
                largest = max(largest, float(np.max(abs(values-refined_values[indices])/scale)))
            differences.append(largest)
        report[field] = {'coarse_medium': differences[0], 'medium_fine': differences[1],
                         'decreasing': differences[1] <= differences[0],
                         'within_budget': max(differences) <= tolerance}
    return {'passed': all(item['within_budget'] for item in report.values()), 'fields': report}


# Expose model-domain and observation-coverage limitations.
def validity_report(assembly, histories) -> dict:
    """Evaluate declared rule domains at all accepted states without live mutation.

    Parameters
    ----------
    assembly : NativeBioreactorAssembly
        Supplies biological provider, units and constant operating conditions.
    histories : sequence
        Accepted native histories; both sides of scheduled events are checked.

    Returns
    -------
    dict
        Scope, status and every offending segment/time/rule. Absent domains are
        unspecified, not validated. Custom-provider domains are not inferred.
        Existing runtime error/allow policies are neither changed nor bypassed
        during simulation; this function only evaluates already accepted states.

    Raises
    ------
    ValueError
        Native states or evaluated rule values are invalid.
    """
    mechanism = assembly.mechanism
    provider = getattr(mechanism, 'rate_provider', getattr(mechanism, 'uptake_provider', None))
    if not isinstance(provider, RuleGraphRateProvider):
        return {'status': 'not_assessable', 'reason': 'no built-in rule-domain contract'}
    if not provider.validity_domain:
        return {'status': 'unspecified', 'reason': 'no validity domain supplied'}
    segments = _segments(histories, 'mass_j_liquid0')
    volumes = _segments(histories, 'vessel_vol')
    population_fields = ('viable_cells_million', 'dead_cells_million', 'product_g',
                         'ivcd_million_cell_day_per_ml', 'biomass_kg')
    violations = []
    for segment, ((times, masses), (_, volume)) in enumerate(zip(segments, volumes)):
        if np.any(volume <= 0.):
            raise ValueError('accepted volumes must be positive')
        for index, time in enumerate(times):
            concentrations = masses[index] / assembly.phase.mw * 1e6 / (volume[index, 0]*1000.)  # [mmol/L]
            populations = {name: float(np.asarray(getattr(histories[segment], name+'_liquid0'))[index].item())
                           if hasattr(histories[segment], name+'_liquid0') else 0. for name in population_fields}
            snapshot = CultureSnapshot(dict(zip(assembly.phase.name_species, concentrations)),
                                       volume[index, 0]*1000., **populations)
            graph = evaluate_rule_graph(provider.definition, snapshot, mechanism.conditions)
            outside = [name for name, bounds in provider.validity_domain.items()
                       if not bounds[0] <= graph[name] <= bounds[1]]
            if outside:
                violations.append({'segment': segment, 'time_s': float(time), 'rules': outside})
    return {'status': 'outside' if violations else 'inside', 'violations': violations,
            'scope': 'accepted reporting states only; not a continuous-time domain certificate',
            'extrapolation_policy': provider.extrapolation}


def observation_report(assembly, histories, declaration, observations, *, windows=None) -> dict:
    """Report descriptive errors at exact measured times, without fitting.

    Parameters
    ----------
    assembly, histories, declaration, observations : object
        Same contracts as predict_measurements; times [s], field-specific units.
    windows : Mapping, optional
        Named (start, end) intervals [s], inclusive, chosen before comparison.
        No default biological phases or case-specific thresholds are inferred.

    Returns
    -------
    dict
        Per-field RMSE, bias, maximum error, units, event side and measured versus
        simulated windows. Unobserved tails are identified, not scored. No
        statistical acceptance is asserted without an external uncertainty model.

    Raises
    ------
    ValueError
        Windows or observations are invalid.
    RuntimeError
        An exact sampling time is absent from accepted states.
    """
    predicted = predict_measurements(assembly, histories, declaration, observations)
    fields = _validate_declaration(declaration, assembly.phase.name_species)
    intervals = dict(windows or {})
    if 'all' in intervals:
        raise ValueError("comparison window name 'all' is reserved")
    for endpoints in intervals.values():
        if len(endpoints) != 2 or not np.isfinite(endpoints).all() or endpoints[0] > endpoints[1]:
            raise ValueError('comparison windows must be finite ordered time pairs')
    reports = {}
    for name, field in fields.items():
        times = predicted['time_s'][name]  # [s]
        measured = np.asarray(observations['values'][name], dtype=float)  # [field unit]
        if (measured.shape != times.shape or not np.isfinite(measured).all()
                or observations['units'][name] != predicted['units'][name]):
            raise ValueError('observation values and units must match declared predictions')
        errors = predicted['values'][name] - measured  # [field unit]
        subsets = {'all': np.ones(len(times), dtype=bool)}
        subsets.update({label: (times >= start) & (times <= end) for label, (start, end) in intervals.items()})
        metrics = {}
        for label, mask in subsets.items():
            values = errors[mask]  # [field unit]
            metrics[label] = (dict(count=int(mask.sum()), rmse=float(np.sqrt(np.mean(values**2))),
                                  bias=float(np.mean(values)), maximum_absolute=float(np.max(abs(values))))
                              if mask.any() else {'count': 0, 'status': 'no_observations'})
        reports[name] = dict(unit=field['unit'], event_side=field['event_side'], metrics=metrics,
            measured_time_range_s=[float(times.min()), float(times.max())],
            simulation_time_range_s=[float(histories[0].time[0]), float(histories[-1].time[-1])],
            unobserved_head=bool(histories[0].time[0] < times.min()),
            unobserved_tail=bool(histories[-1].time[-1] > times.max()))
    return {'fields': reports, 'statistical_acceptance': 'not_assessed'}
