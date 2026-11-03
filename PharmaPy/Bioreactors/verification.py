"""Explicit convergence trials outside production integration; never retry a solve."""

from copy import deepcopy
import numpy as np

from .construction import build_bioreactor
from .input_format import NUMERICAL_FIELDS


# Compare named material inventories on the same requested time grid.
def _samples(assembly, histories, times):
    """Extract post-event native outputs without interpolation.

    Parameters
    ----------
    assembly : NativeBioreactorAssembly
        Real solved assembly, providing species ordering.
    histories : tuple
        Native segment results.
    times : ndarray
        Common requested reporting times [s].

    Returns
    -------
    dict
        Species names, mass matrix [kg], volume vector [m3] and temperature [K].

    Raises
    ------
    RuntimeError
        A requested reporting time was not returned by the solver.
    """
    returned = np.concatenate([history.time for history in histories])  # [s]
    positions = np.searchsorted(returned, times, side='right') - 1
    if np.any(positions < 0) or not np.array_equal(returned[positions], times):
        raise RuntimeError('convergence reporting times were not preserved')
    return {
        'species': tuple(assembly.phase.name_species),
        'mass_kg': np.vstack([history.mass_j_liquid0 for history in histories])[positions],
        'volume_m3': np.concatenate([np.asarray(history.vessel_vol).reshape(-1) for history in histories])[positions],
        'temperature_k': np.concatenate([np.asarray(history.global_temp).reshape(-1) for history in histories])[positions],
    }


def run_convergence(case: dict, thermo_path, settings: list, *, report_times,
                    reference_index: int, conserved_mass_kg=None, definition=None) -> list:
    """Run independently declared numerical trials and retain every outcome.

    Parameters
    ----------
    case : dict
        Existing combined process JSON or separate case declaration.
    thermo_path : path-like
        Existing thermodynamic input.
    settings : list of dict
        Explicit numerical overrides for each fresh solve. Physical inputs and
        inventory acceptance requirements cannot be changed by these overrides.
    report_times : array_like
        Common strictly increasing reporting times [s]. Where native histories
        provide both event sides, select the post-event result. Native terminal
        event semantics remain unchanged. Internal steps are not comparison points.
    reference_index : int
        Explicit trial index used for differences; no best trial is selected.
        If that trial fails, comparisons remain unavailable for every trial.
    conserved_mass_kg : float, optional
        Independently specified conserved total liquid mass [kg], only appropriate
        when the modeled process conserves it. No conservation law is inferred
        for fed, withdrawn, or incompletely represented material inventories.
    definition : dict, optional
        Existing separate mechanism declaration for legacy layouts.

    Returns
    -------
    list of dict
        Each record retains settings, success/error, and on success raw sampled
        masses [kg], volumes [m3], temperatures [K], minimum mass [kg], total
        mass change [kg], and requested conservation residual [kg]. Reference
        differences have the corresponding physical units. Success means the
        solve returned; agreement with a reference is not empirical validation.

    Raises
    ------
    ValueError
        Settings alter unsupported fields, times are invalid, the explicit
        reference is absent, or a conserved mass is nonfinite/negative.
    """
    times = np.asarray(report_times, dtype=float)  # [s]
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.):
        raise ValueError('report_times must be finite and strictly increasing')
    allowed = set(NUMERICAL_FIELDS) | {'method'}
    if (not settings or not isinstance(reference_index, int)
            or not 0 <= reference_index < len(settings)
            or any(not isinstance(item, dict) or set(item) - allowed for item in settings)):
        raise ValueError('declare numerical settings and a valid reference_index; acceptance must remain fixed')
    if conserved_mass_kg is not None and (not np.isfinite(conserved_mass_kg) or conserved_mass_kg < 0.):
        raise ValueError('conserved_mass_kg must be finite and nonnegative')
    records = []
    for options in settings:
        document = deepcopy(case)
        section = 'numerical' if document.get('schema_version') == 3 else 'numerics'
        document[section].update(deepcopy(options))
        record = {'settings': deepcopy(options), 'times_s': times.copy(), 'comparison': None}
        try:
            assembly = build_bioreactor(document, definition, thermo_path=thermo_path)
            sampled = _samples(assembly, assembly.solve(report_times=times), times)
            total = sampled['mass_kg'].sum(axis=1)  # [kg]
            record.update(sampled, status='succeeded', minimum_mass_kg=float(sampled['mass_kg'].min()),
                          total_mass_change_kg=float(total[-1]-total[0]),
                          conservation_error_kg=None if conserved_mass_kg is None else float(np.max(abs(total-conserved_mass_kg))))
        except Exception as error:
            # Batch-reporting boundary: record the failure, never substitute a trajectory.
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
        records.append(record)
    reference = records[reference_index]
    if reference['status'] == 'succeeded':
        for record in records:
            if record['status'] == 'succeeded':
                record['comparison'] = {name: float(np.max(abs(record[name]-reference[name])))
                                        for name in ('mass_kg', 'volume_m3', 'temperature_k')}
    return records
