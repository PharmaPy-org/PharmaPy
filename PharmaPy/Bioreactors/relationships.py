"""Expand reusable biological relationships into the existing kinetic rule graph.

These are mathematical forms, not organism models or parameter defaults. Inputs
retain their declared rate/concentration basis; yields must convert that basis.
No relationship introduces a reconciliation target or a hard constraint implicitly.
"""

from copy import deepcopy
from math import isfinite


def _op(name, *args):
    return {'op': name, 'args': list(args)}


def _uptake(x):
    return _op('multiply', x['maximum'],
               _op('monod', x['substrate'], x['half_saturation']))


def _inhibited(x):
    return _op('divide', x['rate'], _op('add', 1., *[
        _op('divide', item['concentration'], item['constant']) for item in x['inhibitors']]))


def _yield_sum(x):
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


def _scalar(value, parameters, name, positive=False):
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


def expand_relationships(model, parameters):
    """Return a private ordinary graph, preserving explicit equations and order."""
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
