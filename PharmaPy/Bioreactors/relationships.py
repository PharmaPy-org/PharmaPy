"""Expand reusable biological relationships into the existing kinetic rule graph.

These are mathematical forms, not organism models or parameter defaults. Inputs
retain their declared rate/concentration basis; yields must convert that basis.
No relationship introduces a reconciliation target or a hard constraint implicitly.
"""

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

from .BioreactorUnitConverter import BioreactorUnitConverter
from .kinetic_units import normalize_units, unit_signature


# Familiar names resolve to the existing equations and input contracts.
_ALIASES = {
    'monod': 'saturating-uptake',
    'haldane': 'substrate-inhibited-growth',
    'andrews': 'substrate-inhibited-growth',
    'pirt': 'maintenance-uptake',
}


# Human-readable unit contracts; the graph is not a dimensional-analysis engine.
_DESCRIPTIONS = {
    'saturating-uptake': ('maximum*S/(K+S)',
        'maximum: output rate; substrate and half_saturation: same concentration basis'),
    'substrate-inhibited-growth': ('maximum*S/(Ks+S+S*S/Ki)',
        'maximum: 1/time; substrate, half_saturation, inhibition_constant: same concentration basis'),
    'first-order-death': ('coefficient*population',
        'coefficient: 1/time; population: declared amount or concentration; output: positive loss magnitude'),
    'additive-inhibition': ('rate/(1+sum(concentration/constant))',
        'rate: output rate; each inhibitor concentration and constant: same concentration basis'),
    'yield-sum': ('sum(rate*yield)',
        'each term rate*yield must have the same output units'),
    'capacity-limited-rate': ('rate*capacity/(capacity+rate)',
        'rate and capacity: same rate basis; smooth attenuation, not a hard cap'),
    'net-production': ('production*yield-consumption',
        'production*yield and consumption: same output rate basis'),
    'maintenance-uptake': ('growth/yield+maintenance',
        'growth/yield and maintenance: same uptake rate basis; growth means gross synthesis'),
    'linear-transfer': ('coefficient*(equilibrium-concentration)',
        'coefficient: 1/time; equilibrium and concentration: same concentration basis'),
}


# Required, positive and nonnegative slots; values retain their declared bases.
_FAMILIES = {
    'saturating-uptake': ({'maximum', 'substrate', 'half_saturation'},
        {'half_saturation'}, {'maximum'}),
    'substrate-inhibited-growth':
        ({'maximum', 'substrate', 'half_saturation', 'inhibition_constant'},
         {'half_saturation', 'inhibition_constant'}, {'maximum'}),
    'first-order-death': ({'coefficient', 'population'}, set(), {'coefficient'}),
    'additive-inhibition': ({'rate', 'inhibitors'}, set(), set()),
    'yield-sum': ({'terms'}, set(), set()),
    'capacity-limited-rate': ({'rate', 'capacity'}, {'capacity'}, set()),
    'net-production': ({'yield', 'production', 'consumption'}, set(), {'yield'}),
    'maintenance-uptake': ({'growth', 'yield', 'maintenance'}, {'yield'}, {'maintenance'}),
    'linear-transfer': ({'coefficient', 'equilibrium', 'concentration'}, set(), {'coefficient'}),
}


# Build reusable equation graphs without organism-specific coefficients.
def _op(name, *args):
    """Build one rule-graph operator node.

    Parameters
    ----------
    name : str
        Operator identifier understood by ``culture._evaluate``.
    *args : object
        Operand expressions, numeric literals, or nested operator nodes. Their
        units are those of the declared rule inputs.

    Returns
    -------
    dict
        Operator node ``{'op': name, 'args': [...]}``.
    """
    return {'op': name, 'args': list(args)}


def _uptake(x):
    """Expand a saturating uptake relationship, ``maximum * S / (K + S)``.

    Parameters
    ----------
    x : dict
        Expanded slots ``maximum`` (rate in the declared output basis),
        ``substrate`` (concentration expression) and ``half_saturation``
        (concentration in the same basis as ``substrate``).

    Returns
    -------
    dict
        Rule-graph expression in the units of ``maximum``.

    Notes
    -----
    The Monod factor is dimensionless only when ``substrate`` and
    ``half_saturation`` share one concentration basis; the input declares both.
    """
    return _op('multiply', x['maximum'],
               _op('monod', x['substrate'], x['half_saturation']))


def _substrate_inhibited_growth(inputs: dict) -> dict:
    """Expand Haldane–Andrews growth using the existing Monod operator.

    Parameters
    ----------
    inputs : dict
        Expanded slots: ``maximum`` [1/time], ``substrate`` [concentration],
        ``half_saturation`` and ``inhibition_constant`` [same concentration].
        Maximum is nonnegative; both concentration constants are positive.

    Returns
    -------
    dict
        Specific gross growth [1/time], ``maximum*S/(Ks+S+S**2/Ki)`` for
        nonnegative substrate. No additional biological state is introduced.

    Notes
    -----
    Andrews (1968), Biotechnology and Bioengineering 10, pp. 708–711,
    equations (1) and (6), DOI: 10.1002/bit.260100602. ``maximum`` is the
    uninhibited scale, not the attainable peak at finite Ki. The existing
    Monod operator gives zero for nonpositive substrate; physical validity
    bounds must still be declared by the provider. No analytic derivatives
    are supplied; numerical differentiation inherits this boundary kink.
    """
    effective_affinity = _op('add', inputs['half_saturation'],  # [substrate concentration]
        _op('divide', _op('multiply', inputs['substrate'], inputs['substrate']),
            inputs['inhibition_constant']))
    return _op('multiply', inputs['maximum'],
               _op('monod', inputs['substrate'], effective_affinity))


def _first_order_death(inputs: dict) -> dict:
    """Expand a first-order loss magnitude without choosing its destination.

    Parameters
    ----------
    inputs : dict
        Expanded ``coefficient`` [1/time], finite and nonnegative, and
        ``population`` [declared viable amount or concentration]. Population
        must be nonnegative within the provider's declared physical domain.

    Returns
    -------
    dict
        Positive loss magnitude ``coefficient*population`` [population/time].
        The caller explicitly subtracts this from viable growth and, if
        modeled, adds the appropriate converted amount to a dead pool.

    Notes
    -----
    Derived from a constant per-capita hazard, dN/dt = -k*N. This is a
    modeling assumption, not an organism-specific mortality correlation.
    Zero population or zero hazard gives zero loss. No state, lysis product,
    recycling, clipping or analytic derivative is introduced implicitly.
    """
    return _op('multiply', inputs['coefficient'], inputs['population'])


def _inhibited(x):
    """Expand additive inhibition, ``rate / (1 + sum(I_i / K_i))``.

    Parameters
    ----------
    x : dict
        Expanded slots ``rate`` (any rate basis) and ``inhibitors``, a list of
        mappings with ``concentration`` and ``constant`` in one shared
        concentration basis per entry.

    Returns
    -------
    dict
        Rule-graph expression in the units of ``rate``.
    """
    return _op('divide', x['rate'], _op('add', 1., *[
        _op('divide', item['concentration'], item['constant']) for item in x['inhibitors']]))


def _yield_sum(x):
    """Expand a yield-weighted sum, ``sum(rate_i * yield_i)``.

    Parameters
    ----------
    x : dict
        Expanded slot ``terms``: a list of mappings with ``rate`` and ``yield``.
        Each yield converts its rate into the common output basis.

    Returns
    -------
    dict
        Rule-graph expression in the declared output basis.
    """
    return _op('add', *[_op('multiply', item['rate'], item['yield']) for item in x['terms']])


# Dispatch only validated families to their existing operator expressions.
def _family_expression(name: str, inputs: dict) -> dict:
    """Build one canonical relationship without anonymous function registries.

    Parameters
    ----------
    name : str
        Canonical family identifier, after alias resolution.
    inputs : dict
        Expanded graph slots, in the units/bases documented by the catalog.

    Returns
    -------
    dict
        Operator graph in the family's declared output rate basis.

    Raises
    ------
    ValueError
        The canonical family is unsupported.
    """
    if name == 'saturating-uptake':
        return _uptake(inputs)
    if name == 'substrate-inhibited-growth':
        return _substrate_inhibited_growth(inputs)
    if name == 'first-order-death':
        return _first_order_death(inputs)
    if name == 'additive-inhibition':
        return _inhibited(inputs)
    if name == 'yield-sum':
        return _yield_sum(inputs)
    if name == 'capacity-limited-rate':
        return _op('multiply', inputs['rate'], _op('divide', inputs['capacity'],
            _op('add', inputs['capacity'], inputs['rate'])))
    if name == 'net-production':
        return _op('subtract', _op('multiply', inputs['yield'], inputs['production']),
            inputs['consumption'])
    if name == 'maintenance-uptake':
        return _op('add', _op('divide', inputs['growth'], inputs['yield']),
            inputs['maintenance'])
    if name == 'linear-transfer':
        return _op('multiply', inputs['coefficient'],
            _op('subtract', inputs['equilibrium'], inputs['concentration']))
    raise ValueError(f'unsupported biological relationship {name!r}')


# Validate explicitly declared correction values before graph expansion.
def _scalar(value, parameters, name, positive=False):
    """Validate one scalar relationship slot that must be a literal or parameter.

    Parameters
    ----------
    value : float or dict
        Numeric literal, or ``{'parameter': name}`` referencing ``parameters``.
    parameters : Mapping[str, float]
        Flattened named model parameters in their declared units.
    name : str
        Slot label used in error messages.
    positive : bool, optional
        Require a strictly positive value; otherwise nonnegative. Default False.

    Raises
    ------
    ValueError
        The parameter is unknown, or the value is not a finite real scalar with
        the required sign.
    """
    if isinstance(value, dict) and set(value) == {'parameter'}:
        try:
            value = parameters[value['parameter']]
        except KeyError as error:
            raise ValueError(f'unknown relationship parameter {value["parameter"]!r}') from error
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or (value <= 0 if positive else value < 0)):
        bound = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} requires a finite {bound} scalar or parameter')


def _correction_value(value: object, parameters: Mapping, constants: Mapping,
                      name: str, positive: bool = False) -> None:
    """Validate a correction scalar while retaining its expression binding.

    Parameters
    ----------
    value : float or dict
        Literal or parameter/constant reference. Theta is dimensionless [-];
        the reference concentration uses the declared concentration unit.
    parameters, constants : Mapping
        Named values in their configured units.
    name : str
        Parameter label for validation errors.
    positive : bool, optional
        Require a strictly positive concentration scale; theta may be signed.

    Raises
    ------
    ValueError
        The binding is unknown, nonfinite, nonscalar or has an invalid sign.
    """
    if isinstance(value, dict):
        if set(value) == {'parameter'}:
            source, key = parameters, value['parameter']
        elif set(value) == {'constant'}:
            source, key = constants, value['constant']
        else:
            raise ValueError(f'correction {name} requires a scalar, parameter or constant')
        if not isinstance(key, str) or key not in source:
            raise ValueError(f'unknown correction {name} binding {key!r}')
        value = source[key]  # [declared concentration unit] or [-] for theta
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or (positive and value <= 0.)):
        raise ValueError(f'correction {name} requires a finite '
                         + ('positive ' if positive else '') + 'real scalar')


# Mutually recursive correction/relationship expansion preserves graph semantics.
def _apply_correction(base: dict, correction: dict, parameters: Mapping,
                      constants: Mapping) -> dict:
    """Expand a declared dimensionless multiplier using existing graph operators.

    Parameters
    ----------
    base : dict
        Already expanded rate expression [its original output units].
    correction : dict
        Exactly type, parameters, concentration_unit and input. The supported
        type is exponential-saturation: exp(theta*x/(x+reference_concentration)).
        Theta [-] is finite and signed; reference concentration is positive.
        Both may be literals or parameter/constant references. Input is an
        expression or an earlier rule's identifier, with values in the declared
        mass/volume or molar/volume unit. Numeric inputs must be nonnegative.
    parameters, constants : Mapping
        Named values in their configured units.

    Returns
    -------
    dict
        Ordinary graph expression with the same output units as base.

    Raises
    ------
    ValueError
        Fields, type, unit, scalar parameters or input are invalid.

    Notes
    -----
    This is an explicitly selected mathematical hypothesis, not a fitted model.
    Input and reference concentration must already use the same declared unit;
    the unit label does not convert arbitrary expression values. Runtime inputs
    must remain in the configured physical domain (normally x >= 0). Existing
    provider validity checks and graph finite-value checks remain authoritative.
    A positive multiplier preserves the base sign and zeros; it adds no states
    or balance connections. Zero theta recovers the base for valid finite inputs.
    """
    fields = {'type', 'parameters', 'concentration_unit', 'input'}
    if not isinstance(correction, dict) or set(correction) != fields:
        raise ValueError('correction requires exactly type, parameters, concentration_unit and input')
    if correction['type'] != 'exponential-saturation':
        raise ValueError('unsupported correction type; choose exponential-saturation')
    unit = correction['concentration_unit']
    parts = unit.split('/') if isinstance(unit, str) else []
    if (len(parts) != 2 or parts[0] not in
            set(BioreactorUnitConverter.MASS) | set(BioreactorUnitConverter.MOLAR)
            or parts[1] not in BioreactorUnitConverter.VOLUME):
        raise ValueError('correction concentration_unit requires supported mass/volume or molar/volume units')
    values = correction['parameters']
    if not isinstance(values, dict) or set(values) != {'theta', 'reference_concentration'}:
        raise ValueError('correction parameters require exactly theta and reference_concentration')
    _correction_value(values['theta'], parameters, constants, 'theta')
    _correction_value(values['reference_concentration'], parameters, constants,
                      'reference_concentration', positive=True)
    concentration = correction['input']  # [concentration_unit]
    if isinstance(concentration, str):
        if not concentration:
            raise ValueError('correction input must name an earlier concentration rule')
        concentration = {'ref': concentration}
    elif isinstance(concentration, dict):
        if not concentration:
            raise ValueError('correction input must be a graph expression')
    elif (isinstance(concentration, bool) or not isinstance(concentration, (int, float))
          or not isfinite(concentration) or concentration < 0.):
        raise ValueError('correction input must be a nonnegative finite concentration or graph expression')
    concentration = _expand(concentration, parameters, constants)
    feature = _op('divide', concentration,  # [-], concentration ratio
                  _op('add', concentration, values['reference_concentration']))
    return _op('multiply', base, _op('exp', _op('multiply', values['theta'], feature)))


def _expand(expr: object, parameters: Mapping, constants: Mapping) -> object:
    """Recursively replace relationship declarations with ordinary operator nodes.

    Parameters
    ----------
    expr : object
        Rule-graph expression, list of expressions, or literal.
    parameters : Mapping[str, float]
        Flattened named model parameters used to validate scalar slots.

    constants : Mapping[str, float]
        Named model constants in their declared units.

    Returns
    -------
    object
        Expression of the same structure in which every ``relationship`` node
        has been expanded; literals and ordinary nodes are returned unchanged.

    Raises
    ------
    ValueError
        A relationship name (canonical or alias) is unsupported, has missing
        or extra inputs, or a scalar
        slot violates its sign requirement, or a declared unit contract is
        dimensionally inconsistent.
    """
    if isinstance(expr, list):
        return [_expand(item, parameters, constants) for item in expr]
    if not isinstance(expr, dict):
        return expr
    if 'correction' in expr:
        base = {key: value for key, value in expr.items() if key != 'correction'}
        if not set(base) & {'relationship', 'op', 'ref', 'parameter', 'constant',
                            'concentration', 'state', 'condition', 'accepted_flux'}:
            raise ValueError('correction requires a base rate expression')
        return _apply_correction(_expand(base, parameters, constants),
                                 expr['correction'], parameters, constants)
    if 'relationship' not in expr:
        return {key: _expand(value, parameters, constants) for key, value in expr.items()}
    if not {'relationship', 'inputs'} <= set(expr) or set(expr) - {'relationship', 'inputs', 'units'}:
        raise ValueError('relationship expressions require relationship, inputs and optional units')
    name = expr['relationship']
    if not isinstance(name, str) or name not in _FAMILIES and name not in _ALIASES:
        choices = ', '.join(sorted(set(_FAMILIES) | set(_ALIASES)))
        raise ValueError(f'unsupported biological relationship {name!r}; choose one of: {choices}')
    name = _ALIASES.get(name, name)
    slots, positive, nonnegative = _FAMILIES[name]
    values = expr['inputs']
    if not isinstance(values, dict) or set(values) != slots:
        missing = sorted(slots - set(values)) if isinstance(values, dict) else sorted(slots)
        extra = sorted(set(values) - slots, key=repr) if isinstance(values, dict) else []
        raise ValueError(f'{name} requires inputs {sorted(slots)}; '
                         f'missing {missing}, unexpected {extra}')
    for slot in positive | nonnegative:
        _scalar(values[slot], parameters, f'{name}.{slot}', slot in positive)
    if name in {'yield-sum', 'additive-inhibition'}:
        slot, fields = (('terms', {'rate', 'yield'}) if name == 'yield-sum'
                        else ('inhibitors', {'concentration', 'constant'}))
        items = values[slot]
        if not isinstance(items, list) or not items:
            raise ValueError(f'{name}.{slot} must be a nonempty list')
        for item in items:
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError(f'{name}.{slot} entries require {sorted(fields)}')
            key = 'yield' if name == 'yield-sum' else 'constant'
            _scalar(item[key], parameters, f'{name}.{key}', key == 'constant')
    if 'units' in expr:
        values, output_factor = normalize_units(name, values, expr['units'])  # [declared output/reference output]
        return _op('multiply', output_factor, _family_expression(name, _expand(values, parameters, constants)))
    return _family_expression(name, _expand(values, parameters, constants))


# Describe existing equation families without constructing a simulation.
def relationship_catalog() -> dict:
    """Describe every supported kinetic relationship without creating a model.

    Returns
    -------
    dict
        Independent JSON-serializable entries keyed by canonical name, each
        containing aliases, required inputs, scalar sign constraints, equation,
        unit contract, optional correction contracts and fill-in templates.
        Null inputs require user-supplied values or graph expressions; no
        parameter is inferred.

    Notes
    -----
    Names are case-sensitive. Aliases and canonical names have identical input
    contracts. Concentration expressions read mmol/L unless explicitly converted.
    The selected equation does not infer biomass normalization, reaction targets,
    yields or material destinations. Existing explicit operator graphs remain
    supported. Catalog mutation cannot change the registry or later calls.
    """
    catalog = {}
    for name, (slots, positive, nonnegative) in _FAMILIES.items():
        equation, units = _DESCRIPTIONS[name]
        catalog[name] = dict(
            aliases=sorted(alias for alias, target in _ALIASES.items() if target == name),
            required_inputs=sorted(slots), positive_scalar_inputs=sorted(positive),
            nonnegative_scalar_inputs=sorted(nonnegative), equation=equation,
            units=units,
            corrections={'exponential-saturation': {
                'equation': 'exp(theta*x/(x+reference_concentration))',
                'theta_unit': '-', 'neutral_theta': 0.0,  # [-], exp(0)=1
                'input_binding': 'earlier rule identifier or expression in concentration_unit',
                'template': {'type': 'exponential-saturation',
                             'parameters': {'theta': 0.0, 'reference_concentration': None},
                             'concentration_unit': None, 'input': None}}},
            expression_template={'relationship': name,
                                 'inputs': dict.fromkeys(sorted(slots))})
        if name == 'yield-sum':
            catalog[name]['expression_template']['inputs']['terms'] = [{'rate': None, 'yield': None}]
        elif name == 'additive-inhibition':
            catalog[name]['expression_template']['inputs']['inhibitors'] = [{'concentration': None, 'constant': None}]
    return catalog


# Expose the validated relationship-to-graph translation.
def expand_relationships(model: Mapping, parameters: Mapping) -> dict:
    """Return a private ordinary rule graph, preserving explicit equations and order.

    Parameters
    ----------
    model : Mapping
        Model declaration whose optional ``rule_graph`` is a list of rules with
        ``identifier`` and ``expression``.
    parameters : Mapping[str, float]
        Flattened named model parameters in their declared units.

    Returns
    -------
    dict
        Deep copy of ``model`` in which relationship families are expanded into
        ordinary operator nodes, including optional correction multipliers.
        Rule order and identifiers are unchanged. Correction input unit labels
        declare a shared basis; they do not convert arbitrary graph expressions.
        Optional named-relationship units convert inputs and outputs within the
        declared equation. Direct species sources must match ``rate_unit``;
        the existing native source converter handles solver units. Other graph
        consumers retain the relationship's declared output units.

    Raises
    ------
    ValueError
        Rule identifiers are empty or duplicated, or a relationship is invalid.
    """
    result = deepcopy(model)
    graph = result.get('rule_graph')
    if graph is None:
        return result
    identifiers = [rule.get('identifier') for rule in graph]
    if (any(not isinstance(name, str) or not name for name in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('rule identifiers must be nonempty and unique')
    mappings = result.get('species_mass_rates')
    sources = set(mappings.values()) if isinstance(mappings, Mapping) else set()
    for rule in graph:
        expression = rule['expression']
        expanded = _expand(expression, parameters, result.get('constants', {}))
        if rule['identifier'] in sources and isinstance(expression, dict) and 'units' in expression:
            if 'rate_unit' not in result:
                raise ValueError('typed species sources require a declared rate_unit')
            declared, source_scale = unit_signature(expression['units']['output'])  # [reference/declared output]
            expected, target_scale = unit_signature(result['rate_unit'])  # [reference/native source unit]
            if declared != expected:
                raise ValueError(f"source {rule['identifier']} has incompatible biological or physical basis for {result['rate_unit']}")
            if source_scale != target_scale:
                raise ValueError(f"source {rule['identifier']} output unit must match rate_unit {result['rate_unit']}; choose that relationship output unit explicitly")
        rule['expression'] = expanded
    return result
