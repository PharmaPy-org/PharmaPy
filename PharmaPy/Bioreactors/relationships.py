"""Expand reusable biological relationships into the existing kinetic rule graph.

These are mathematical forms, not organism models or parameter defaults. Inputs
retain their declared rate/concentration basis; yields must convert that basis.
No relationship introduces a reconciliation target or a hard constraint implicitly.
"""

from copy import deepcopy
from math import isfinite


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


# Required slots, positive scalar slots, and nonnegative scalar slots. Scalars
# may be literals or named parameters; state-dependent inputs are expressions.
_FAMILIES = {
    'saturating-uptake': (_uptake,
        {'maximum', 'substrate', 'half_saturation'}, {'half_saturation'}, {'maximum'}),
    'additive-inhibition': (_inhibited, {'rate', 'inhibitors'}, set(), set()),
    'yield-sum': (_yield_sum, {'terms'}, set(), set()),
    'capacity-limited-rate': (lambda x: _op('multiply', x['rate'],
        _op('divide', x['capacity'], _op('add', x['capacity'], x['rate']))),
        {'rate', 'capacity'}, {'capacity'}, set()),
    'net-production': (lambda x: _op('subtract',
        _op('multiply', x['yield'], x['production']), x['consumption']),
        {'yield', 'production', 'consumption'}, set(), {'yield'}),
    'maintenance-uptake': (lambda x: _op('add',
        _op('divide', x['growth'], x['yield']), x['maintenance']),
        {'growth', 'yield', 'maintenance'}, {'yield'}, {'maintenance'}),
    'linear-transfer': (lambda x: _op('multiply', x['coefficient'],
        _op('subtract', x['equilibrium'], x['concentration'])),
        {'coefficient', 'equilibrium', 'concentration'}, set(), {'coefficient'}),
}


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


def _expand(expr, parameters):
    """Recursively replace relationship declarations with ordinary operator nodes.

    Parameters
    ----------
    expr : object
        Rule-graph expression, list of expressions, or literal.
    parameters : Mapping[str, float]
        Flattened named model parameters used to validate scalar slots.

    Returns
    -------
    object
        Expression of the same structure in which every ``relationship`` node
        has been expanded; literals and ordinary nodes are returned unchanged.

    Raises
    ------
    ValueError
        A relationship is unsupported, has missing or extra inputs, or a scalar
        slot violates its sign requirement.
    """
    if isinstance(expr, list):
        return [_expand(item, parameters) for item in expr]
    if not isinstance(expr, dict):
        return expr
    if 'relationship' not in expr:
        return {key: _expand(value, parameters) for key, value in expr.items()}
    if set(expr) != {'relationship', 'inputs'}:
        raise ValueError('relationship expressions require only relationship and inputs')
    name = expr['relationship']
    if not isinstance(name, str) or name not in _FAMILIES:
        raise ValueError(f'unsupported biological relationship {name!r}')
    build, slots, positive, nonnegative = _FAMILIES[name]
    values = expr['inputs']
    if not isinstance(values, dict) or set(values) != slots:
        raise ValueError(f'{name} requires inputs {sorted(slots)}')
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
    return build(_expand(values, parameters))


# Expose the validated relationship-to-graph translation.
def expand_relationships(model, parameters):
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
        ordinary operator nodes. Rule order and identifiers are unchanged.

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
    for rule in graph:
        rule['expression'] = _expand(rule['expression'], parameters)
    return result
