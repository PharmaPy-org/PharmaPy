"""Physical construction contracts using native phase properties and metadata."""

from collections.abc import Mapping
from typing import Optional
import numpy as np
from PharmaPy.Phases_Refactored import LiquidPhase

from .BioreactorUnitConverter import BioreactorUnitConverter
from .mechanisms import DeclaredStreamDensity


# Preserve the established initial consistency budget; it is roundoff, not data uncertainty.
MASS_ROUNDOFF_KG = 1e-12  # [kg]; existing construction consistency threshold


def density_contract(operation: Mapping, phase: LiquidPhase) -> tuple:
    """Validate redundant initial quantities and select the density assumption.

    Parameters
    ----------
    operation : Mapping
        Initial mass [kg], volume [declared unit], carrier density [kg/L], and
        optional density declaration. Mass and species amounts are authoritative;
        volume is an independent consistency check, never silently overwritten.
    phase : LiquidPhase
        Native phase constructed from mass, composition and temperature.

    Returns
    -------
    tuple
        Optional existing density mechanism and JSON-serializable report.

    Raises
    ------
    ValueError
        Density selection is invalid, conflicts with prescribed working volume,
        or the redundant initial mass/volume values disagree.
    """
    declaration = operation.get('density')
    volume_l = BioreactorUnitConverter.convert(operation['initial_volume']['value'],
                                              operation['initial_volume']['unit'], 'L')  # [L]
    mechanism = None
    if declaration is None:
        mode = 'legacy-carrier'
        density = float(operation['carrier_density_kg_l'])  # [kg/L]; historical initial mixture assumption
    else:
        if not isinstance(declaration, Mapping) or declaration.get('mode') not in {'constant-mixture', 'native-phase'}:
            raise ValueError('density.mode must be constant-mixture or native-phase')
        if operation.get('volume_policy') not in {None, 'native'}:
            raise ValueError('explicit density requires native volume_policy; working-volume is independently prescribed')
        mode = declaration['mode']
        fields = {'mode', 'value', 'unit'} if mode == 'constant-mixture' else {'mode'}
        if set(declaration) != fields:
            raise ValueError(f'density {mode} requires exactly {sorted(fields)}')
        if mode == 'constant-mixture':
            density = BioreactorUnitConverter.convert(declaration['value'], declaration['unit'], 'kg/L')  # [kg/L]
        else:
            density = float(phase.getDensity()) / 1000.  # [kg/L]; actual initial composition and temperature
    if np.ndim(density) != 0 or not np.isfinite(density) or density <= 0.:
        raise ValueError('initial mixture density must be finite, positive and scalar')
    if mode == 'constant-mixture':
        mechanism = DeclaredStreamDensity(density * 1000., owning_phase=phase)  # [kg/m3]; exact L/m3 conversion
    if not np.isclose(phase.mass, volume_l * density, rtol=0., atol=MASS_ROUNDOFF_KG):
        raise ValueError('initial_mass_kg must agree with initial_volume and selected density')
    return mechanism, {'mode': mode, 'initial_check_density_kg_l': float(density),
                       'carrier_density_kg_l': float(operation['carrier_density_kg_l']),
                       'scope': 'initial mixture consistency; feed density is declared separately'}


def reaction_contract(declaration: Optional[Mapping], phase: LiquidPhase) -> dict:
    """Check explicitly declared reaction invariants without inventing composition.

    Parameters
    ----------
    declaration : dict or None
        Optional ``basis`` (kg/extent or mol/extent), named ``reactions`` mapping
        species to signed coefficients, and ``invariants``. Each invariant has
        ``unit``, complete ``coefficients`` [unit/kg], and ``absolute_tolerance``
        [unit/extent]. Extent denotes the independently declared reaction basis.
    phase : LiquidPhase
        Native species order and molecular weights [g/mol].

    Returns
    -------
    dict
        Passed declared-matrix residuals, or not_assessable with missing evidence.
        This is phase-only structural verification, not a kinetic trajectory check.

    Raises
    ------
    ValueError
        Declaration is malformed or its complete invariants are violated.
    """
    if declaration is None:
        return {'status': 'not_assessable', 'reason': 'reaction invariants not declared',
                'scope': 'phase species only; auxiliary biological pools require separate accounting'}
    if not isinstance(declaration, Mapping) or set(declaration) != {'basis', 'reactions', 'invariants'}:
        raise ValueError('conservation requires basis, reactions and invariants')
    if declaration['basis'] not in {'kg/extent', 'mol/extent'}:
        raise ValueError('conservation basis must be kg/extent or mol/extent')
    names = tuple(phase.name_species)
    reactions, invariants = declaration['reactions'], declaration['invariants']
    if not isinstance(reactions, Mapping) or not reactions or not isinstance(invariants, Mapping) or not invariants:
        raise ValueError('conservation requires nonempty reaction and invariant mappings')
    columns = []
    for reaction, coefficients in reactions.items():
        if not isinstance(coefficients, Mapping) or not coefficients or set(coefficients) - set(names):
            raise ValueError(f'reaction {reaction} must name known phase species')
        values = np.asarray([coefficients.get(name, 0.) for name in names], dtype=float)  # [declared amount/extent]
        if values.shape != (len(names),) or not np.isfinite(values).all() or not np.any(values):
            raise ValueError('reaction coefficients must be finite scalars and the reaction nonzero')
        if declaration['basis'] == 'mol/extent':
            values = values * np.asarray(phase.mw) / 1000.  # [kg/extent]; g/mol to kg/mol
        columns.append(values)
    matrix = np.column_stack(columns)  # [kg/extent]
    checks = {}
    for invariant, item in invariants.items():
        if not isinstance(item, Mapping) or set(item) != {'unit', 'coefficients', 'absolute_tolerance'}:
            raise ValueError('invariants require unit, coefficients and absolute_tolerance')
        coefficients = item['coefficients']
        if not isinstance(item['unit'], str) or not item['unit'] or not isinstance(coefficients, Mapping) or set(coefficients)-set(names):
            raise ValueError('invariant units and species coefficients must be explicit')
        tolerance = float(item['absolute_tolerance'])  # [invariant unit/extent]
        if not np.isfinite(tolerance) or tolerance < 0.:
            raise ValueError('invariant tolerance must be finite and nonnegative')
        if not np.isfinite(np.asarray(list(coefficients.values()), dtype=float)).all():
            raise ValueError('invariant coefficients must be finite')
        if set(coefficients) != set(names):
            checks[invariant] = {'status': 'not_assessable', 'missing_species': sorted(set(names)-set(coefficients))}
            continue
        row = np.asarray([coefficients[name] for name in names], dtype=float)  # [invariant unit/kg]
        if row.shape != (len(names),) or not np.isfinite(row).all() or not np.any(row):
            raise ValueError('invariant coefficients must be finite scalars and the invariant nonzero')
        residual = row @ matrix  # [invariant unit/extent]
        if np.any(abs(residual) > tolerance):
            raise ValueError(f'declared conservation invariant {invariant} violated: {dict(zip(reactions, residual.tolist()))}')
        checks[invariant] = {'status': 'passed', 'unit': item['unit']+'/extent',
                             'residuals': dict(zip(reactions, residual.tolist()))}
    return {'status': 'passed' if all(item['status']=='passed' for item in checks.values()) else 'not_assessable',
            'scope': 'declared reaction matrix only; does not certify implemented kinetic sources', 'invariants': checks}
