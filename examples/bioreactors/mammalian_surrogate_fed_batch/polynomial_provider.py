"""Example-local, input-declared polynomial surrogate; no organism-specific code."""

import numpy as np

from PharmaPy.Bioreactors import RateProviderResult


class PolynomialProvider:
    """Evaluate a fitted multivariate polynomial inside its declared domain."""

    def __init__(self, declaration, definition, expected_outputs):
        self.declaration = declaration
        self.inputs = declaration['inputs']
        self.outputs = declaration['outputs']
        self.powers = np.asarray(declaration['powers'])
        if set(self.outputs) != set(expected_outputs):
            raise ValueError('polynomial outputs must match assigned outputs')
        if (self.powers.ndim != 2 or self.powers.shape[1] != len(self.inputs)
                or np.any(self.powers < 0) or np.any(self.powers != self.powers.astype(int))):
            raise ValueError('polynomial powers must be nonnegative integer multi-indices')
        self.domain = declaration['validity_domain']
        if set(self.domain) != set(self.inputs):
            raise ValueError('validity domain must cover all polynomial inputs')
        for name, spec in self.inputs.items():
            if spec['source'] != 'concentration' or spec['unit'] != 'mmol/L':
                raise ValueError('polynomial inputs use native concentrations in mmol/L')
            bounds = self.domain[name]
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('invalid polynomial domain')
        if declaration['extrapolation'] not in {'error', 'allow'}:
            raise ValueError('declare polynomial extrapolation policy')
        for output in self.outputs.values():
            if len(output['coefficients']) != len(self.powers) or not np.isfinite(output['coefficients']).all():
                raise ValueError('invalid polynomial coefficients')

    def evaluate(self, snapshot, conditions):
        values = np.array([snapshot.concentrations_mmol_l[s['name']] for s in self.inputs.values()])
        if not np.isfinite(values).all():
            raise ValueError('polynomial inputs must be finite')
        outside = [name for name, value in zip(self.inputs, values)
                   if not self.domain[name][0] <= value <= self.domain[name][1]]
        if outside and self.declaration['extrapolation'] == 'error':
            raise ValueError(f'polynomial outside validity domain: {outside}')
        terms = np.prod(values ** self.powers, axis=1)
        rates = {name: float(terms @ spec['coefficients']) for name, spec in self.outputs.items()}
        return RateProviderResult(rates, {name: s['unit'] for name, s in self.outputs.items()},
            self.declaration['identifier'], self.declaration['version'], not outside,
            {'outside_inputs': outside, 'provider_type': 'registered-polynomial'})
