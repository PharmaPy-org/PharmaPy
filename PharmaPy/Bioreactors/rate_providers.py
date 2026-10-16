"""Named biological-rate contracts and declarative kinetic providers."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .culture import evaluate_rule_graph, _validate_graph


@dataclass(frozen=True)
class RateProviderResult:
    """Return validated rates, units, identity, and validity diagnostics.

    This is the shared biological-rate contract: every provider, whether a
    rule graph, a pathway LP or a registered alternative, returns one of these
    so the reactor balances never depend on how the rates were computed.

    Attributes
    ----------
    rates : Mapping[str, float]
        Named rates, each in the unit given by ``units``.
    units : Mapping[str, str]
        Unit string for every rate name, such as ``'mmol/(L h)'`` or ``'1/day'``.
    provider_id : str
        Provider identity.
    version : str
        Provider version.
    inside_validity_domain : bool
        False when a declared validity bound was exceeded and extrapolation was
        allowed.
    diagnostics : Mapping[str, object]
        Provider-specific evidence, such as solver status.

    Raises
    ------
    ValueError
        Rates and units disagree in names, a rate is nonfinite, a unit is empty,
        or the identity or version is missing.
    """
    rates: Mapping[str, float]
    units: Mapping[str, str]
    provider_id: str
    version: str
    inside_validity_domain: bool
    diagnostics: Mapping[str, object]

    def __post_init__(self):
        """Validate rate names and units, then freeze the mappings.

        Raises
        ------
        ValueError
            Rate and unit names differ or are empty, a rate is nonfinite, a unit is
            empty, or the identity or version is missing.
        """
        rates = {str(name): float(value) for name, value in self.rates.items()}
        units = {str(name): str(value) for name, value in self.units.items()}
        if not rates or set(rates) != set(units):
            raise ValueError("provider rates and units must have identical nonempty names")
        if not np.isfinite(list(rates.values())).all() or any(not unit for unit in units.values()):
            raise ValueError("provider rates must be finite and units nonempty")
        if not self.provider_id or not self.version:
            raise ValueError("provider identity and version are required")
        object.__setattr__(self, "rates", MappingProxyType(rates))
        object.__setattr__(self, "units", MappingProxyType(units))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def __deepcopy__(self, memo):
        """Preserve read-only rate mappings during native result replay.

        Parameters
        ----------
        memo : dict
            ``copy.deepcopy`` memo dictionary.

        Returns
        -------
        RateProviderResult
            Independent result with equal rates, units, identity and diagnostics.
        """
        return type(self)(dict(self.rates), dict(self.units), self.provider_id, self.version,
                          self.inside_validity_domain, deepcopy(dict(self.diagnostics), memo))


class RuleGraphRateProvider:
    """Supply biological rates from the existing declarative kinetic graph.

    Parameters
    ----------
    declaration : Mapping
        Provider declaration: ``output_units`` (required, one unit per expected
        output), optional ``identifier``, ``version``, ``validity_domain``
        (``{rule: [low, high]}`` in that rule's units) and ``extrapolation``
        (``'error'`` or ``'allow'``, required when a domain is declared).
    definition : object
        Object exposing ``rule_graph``, ``parameters`` and ``constants``.
    expected_outputs : Sequence[str]
        Rule identifiers the caller consumes as rates.

    Raises
    ------
    ValueError
        The graph is invalid, units do not cover the outputs, or the validity
        domain or extrapolation policy is malformed.
    """

    def __init__(self, declaration, definition, expected_outputs):
        """Validate the graph, output units and optional validity domain.

        Parameters
        ----------
        declaration : Mapping
            Provider declaration (see class docstring).
        definition : object
            Object exposing ``rule_graph``, ``parameters`` and ``constants``.
        expected_outputs : Sequence[str]
            Rule identifiers returned as rates.

        Raises
        ------
        ValueError
            The graph, units, validity domain or extrapolation policy is invalid.
        """
        self.definition = definition
        self.expected_outputs = tuple(expected_outputs)
        _validate_graph(definition, self.expected_outputs)
        units = declaration.get("output_units")
        if not isinstance(units, Mapping) or not set(self.expected_outputs) <= set(units):
            raise ValueError("rule-graph output_units must cover all assigned outputs")
        self.units = {name: units[name] for name in self.expected_outputs}
        self.provider_id = str(declaration.get("identifier", "rule-graph"))
        self.version = str(declaration.get("version", "1"))
        self.validity_domain = {name: tuple(map(float, bounds))
                                for name, bounds in declaration.get('validity_domain', {}).items()}
        if set(self.validity_domain) - {r['identifier'] for r in definition.rule_graph}:
            raise ValueError('rule-graph validity_domain references an unknown rule')
        if any(len(b) != 2 or not np.isfinite(b).all() or b[0] > b[1]
               for b in self.validity_domain.values()):
            raise ValueError('rule-graph validity bounds must be finite ordered pairs')
        self.extrapolation = declaration.get('extrapolation')
        if self.validity_domain and self.extrapolation not in {'error', 'allow'}:
            raise ValueError('rule-graph extrapolation must be explicitly error or allow')

    def evaluate(self, snapshot, conditions):
        """Evaluate the rule graph and return the expected outputs as rates.

        Parameters
        ----------
        snapshot : object
            Reactor state view; rules read attributes such as
            ``concentrations_mmol_l`` [mmol/L] and ``volume_l`` [L].
        conditions : Mapping[str, float]
            Declared operating conditions in their input units.

        Returns
        -------
        RateProviderResult
            Expected outputs in their declared units, with validity diagnostics.

        Raises
        ------
        ValueError
            A rule is nonfinite, or a validity bound is exceeded while
            ``extrapolation`` is ``'error'``.
        """
        graph = evaluate_rule_graph(self.definition, snapshot, conditions)
        outside = tuple(name for name, bounds in self.validity_domain.items()
                        if not bounds[0] <= graph[name] <= bounds[1])
        if outside and self.extrapolation == 'error':
            raise ValueError(f'rule-graph outside validity domain: {list(outside)}')
        rates = {name: graph[name] for name in self.expected_outputs}
        return RateProviderResult(rates, self.units, self.provider_id, self.version,
                                  not outside, {"provider_type": "rule-graph", "outside_rules": outside})


_PROVIDER_FACTORIES: dict[str, Callable] = {
    "rule-graph": RuleGraphRateProvider,
}


def register_rate_provider(provider_type, factory):
    """Register one new provider constructor without replacing existing types.

    Parameters
    ----------
    provider_type : str
        Name used in the ``rate_provider.type`` input field.
    factory : Callable
        ``factory(declaration, definition, expected_outputs)`` returning an
        object with ``evaluate(snapshot, conditions) -> RateProviderResult``.

    Raises
    ------
    ValueError
        The name is empty, the factory is not callable, or the type is already
        registered.
    """
    provider_type = str(provider_type)
    if not provider_type or not isinstance(factory, Callable):
        raise ValueError("provider registration requires a name and callable factory")
    if provider_type in _PROVIDER_FACTORIES:
        raise ValueError(f"biological-rate provider {provider_type!r} is already registered")
    _PROVIDER_FACTORIES[provider_type] = factory


def build_rate_provider(declaration, definition, expected_outputs):
    """Resolve one explicitly declared biological-rate provider.

    Parameters
    ----------
    declaration : Mapping
        Provider declaration with a registered ``type``.
    definition : object
        Object exposing ``rule_graph``, ``parameters`` and ``constants``.
    expected_outputs : Sequence[str]
        Unique rate names the mechanism will consume.

    Returns
    -------
    object
        Provider instance with ``evaluate(snapshot, conditions)``.

    Raises
    ------
    TypeError
        ``definition`` lacks the required attributes.
    ValueError
        The type is missing or unknown, or the outputs are empty or repeated.
    """
    if not all(hasattr(definition, name) for name in ('rule_graph', 'parameters', 'constants')):
        raise TypeError("definition must expose rule_graph, parameters and constants")
    if not isinstance(declaration, Mapping) or "type" not in declaration:
        raise ValueError("rate provider requires an explicit type")
    provider_type = declaration["type"]
    if provider_type not in _PROVIDER_FACTORIES:
        raise ValueError(f"unknown biological-rate provider {provider_type!r}")
    expected = tuple(expected_outputs)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("provider output names must be nonempty and unique")
    return _PROVIDER_FACTORIES[provider_type](declaration, definition, expected)
