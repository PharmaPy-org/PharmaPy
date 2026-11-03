"""Optional declared-unit contracts for named biological equation forms.

This checks the finite relationship catalog, not arbitrary rule-graph expressions.
Dry mass, wet mass, molecular amount and cell count remain distinct bases.
"""

from .BioreactorUnitConverter import BioreactorUnitConverter as Converter


# Existing conversion tables fix the reference units: g, mol, L, s and cells.
UNIT_GROUPS = {'mass': Converter.MASS, 'amount': Converter.MOLAR,
               'volume': Converter.VOLUME, 'time': Converter.TIME,
               'cells': {'cell': 1., 'million_cell': Converter.CELLS['10^6 cell']},
               'dry_mass': {unit+'DW': scale for unit, scale in Converter.MASS.items()},
               'wet_mass': {unit+'Wet': scale for unit, scale in Converter.MASS.items()}}


def unit_signature(unit: str) -> tuple:
    """Read a supported amount, ratio, concentration or rate unit.

    Parameters
    ----------
    unit : str
        Known unit token, ``1/time``, ``amount/volume`` or
        ``amount/(normalizer time)``. Products are allowed in the denominator;
        arbitrary powers, offsets and symbolic unit expressions are unsupported.

    Returns
    -------
    tuple
        Physical-basis exponents and multiplicative scale to g/mol/L/s/cell
        reference coordinates. Dry and wet biomass have independent dimensions.

    Raises
    ------
    ValueError
        Unit syntax or a physical basis is unsupported.
    """
    if not isinstance(unit, str) or not unit:
        raise ValueError('relationship units must be explicit strings')
    text = unit.replace('10^6 cell', 'million_cell')
    parts = text.split('/')
    if len(parts) > 2 or not all(parts) or text.count('(') != text.count(')'):
        raise ValueError(f'unsupported relationship unit {unit!r}')
    numerator = parts[0]
    denominator = [] if len(parts) == 1 else parts[1].removeprefix('(').removesuffix(')').split()
    if len(parts) == 2 and not denominator:
        raise ValueError(f'unsupported relationship unit {unit!r}')
    dimensions = {}
    factor = 1.  # [reference units/declared units]; dimensionless identity
    for token, power in [(numerator, 1), *((item, -1) for item in denominator)]:
        if token in {'1', '-'}:
            continue
        matches = [(name, scales[token]) for name, scales in UNIT_GROUPS.items() if token in scales]
        if len(matches) != 1:
            raise ValueError(f'unsupported relationship unit token {token!r}')
        name, scale = matches[0]  # [reference token/declared token]
        dimensions[name] = dimensions.get(name, 0) + power
        factor *= scale ** power
    return {name: power for name, power in dimensions.items() if power}, factor


def _product(left: dict, right: dict, sign: int = 1) -> dict:
    """Combine declared dimensions for multiplication or division.

    Parameters
    ----------
    left, right : dict
        Integer physical-basis exponents [-].
    sign : int
        1 for multiplication, -1 for division [-].

    Returns
    -------
    dict
        Resulting physical-basis exponents [-].
    """
    result = dict(left)
    for name, power in right.items():
        result[name] = result.get(name, 0) + sign*power
    return {name: power for name, power in result.items() if power}


def _scaled(expression, factor):
    """Retain parameter bindings while expressing a declared unit conversion.

    Parameters
    ----------
    expression : object
        Existing rule-graph expression in the source unit.
    factor : float
        Multiplicative destination/source conversion factor.

    Returns
    -------
    object
        Existing expression or an ordinary multiplication node.
    """
    return expression if factor == 1. else {'op': 'multiply', 'args': [factor, expression]}


def _slot(expression, unit):
    """Validate known leaf bindings and normalize one explicitly declared slot.

    Parameters
    ----------
    expression : object
        Literal, parameter or expression. Arbitrary expression dimensions remain
        the author's responsibility; direct species concentration is mmol/L.
    unit : str
        Explicit slot unit.

    Returns
    -------
    tuple
        Converted expression and its declared physical dimensions.

    Raises
    ------
    ValueError
        A known leaf or nested typed relationship contradicts its declared unit.
    """
    dimensions, factor = unit_signature(unit)
    if isinstance(expression, dict):
        actual = None
        if set(expression) == {'concentration'}:
            actual = 'mmol/L'
        elif 'relationship' in expression and 'units' in expression:
            actual = expression['units'].get('output')
        if actual is not None and unit_signature(actual) != (dimensions, factor):
            raise ValueError(f'relationship slot declares {unit}, but its binding returns {actual}; convert explicitly')
    return _scaled(expression, factor), dimensions


def normalize_units(name: str, values: dict, units: dict) -> tuple:
    """Validate the named equation's dimensional identity and convert its slots.

    Parameters
    ----------
    name : str
        Canonical relationship identifier.
    values : dict
        Original relationship inputs, retaining parameter/expression bindings.
    units : dict
        Exactly the input keys plus ``output``; list inputs have corresponding
        lists of unit mappings. No biological normalization is inferred.

    Returns
    -------
    tuple
        Inputs in reference coordinates and conversion from reference output to
        the declared output unit.

    Raises
    ------
    ValueError
        Unit declarations are incomplete or the named equation is inconsistent.
    """
    if not isinstance(units, dict) or set(units) != set(values) | {'output'}:
        raise ValueError('relationship units must cover every input and output')
    output, factor = unit_signature(units['output'])
    normalized, dimensions = {}, {}
    for key, value in values.items():
        if isinstance(value, list):
            declared = units[key]
            if not isinstance(declared, list) or len(declared) != len(value):
                raise ValueError(f'relationship units for {key} must align with entries')
            normalized[key], dimensions[key] = [], []
            for item, labels in zip(value, declared):
                if not isinstance(labels, dict) or set(labels) != set(item):
                    raise ValueError('relationship entry units must name every field')
                converted, basis = {}, {}
                for field in item:
                    converted[field], basis[field] = _slot(item[field], labels[field])
                normalized[key].append(converted)
                dimensions[key].append(basis)
        else:
            normalized[key], dimensions[key] = _slot(value, units[key])
    checks = []
    if name in {'saturating-uptake', 'substrate-inhibited-growth'}:
        checks = [dimensions['maximum'] == output, dimensions['substrate'] == dimensions['half_saturation']]
        if name == 'substrate-inhibited-growth':
            checks.append(dimensions['substrate'] == dimensions['inhibition_constant'])
    elif name == 'capacity-limited-rate':
        checks = [dimensions['rate'] == dimensions['capacity'] == output]
    elif name == 'first-order-death':
        checks = [dimensions['coefficient'] == {'time': -1}, _product(dimensions['coefficient'], dimensions['population']) == output]
    elif name == 'linear-transfer':
        checks = [dimensions['coefficient'] == {'time': -1}, dimensions['equilibrium'] == dimensions['concentration'],
                  _product(dimensions['coefficient'], dimensions['concentration']) == output]
    elif name == 'maintenance-uptake':
        checks = [_product(dimensions['growth'], dimensions['yield'], -1) == dimensions['maintenance'] == output]
    elif name == 'net-production':
        checks = [_product(dimensions['yield'], dimensions['production']) == dimensions['consumption'] == output]
    elif name == 'yield-sum':
        checks = [_product(item['rate'], item['yield']) == output for item in dimensions['terms']]
    elif name == 'additive-inhibition':
        checks = [dimensions['rate'] == output, *(item['concentration'] == item['constant'] for item in dimensions['inhibitors'])]
    if not checks or not all(checks) or output.get('time') != -1:
        raise ValueError(f'incompatible physical units or biological bases in {name}')
    return normalized, 1./factor


def unit_audit(*documents) -> dict:
    """Count declared contracts without claiming arbitrary-expression inference.

    Parameters
    ----------
    *documents : object
        Original JSON declarations before named relationships are expanded.

    Returns
    -------
    dict
        Typed and untyped relationship counts; used only after normalization
        has successfully validated the selected input documents.
    """
    pending = list(documents)
    declared = untyped = 0
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if 'relationship' in value:
                if 'units' in value:
                    declared += 1
                else:
                    untyped += 1
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return {'declared_relationship_contracts': declared, 'untyped_relationships': untyped,
            'scope': 'declared named forms only; arbitrary expression dimensions and coefficient provenance remain user-declared'}
