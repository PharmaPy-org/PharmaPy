"""Biological-rate providers for mechanistic, surrogate, and hybrid models."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .culture import CultureModelDefinition, evaluate_rule_graph


@dataclass(frozen=True)
class RateProviderResult:
    """Return validated rates, units, identity, and validity diagnostics."""
    rates: Mapping[str, float]
    units: Mapping[str, str]
    provider_id: str
    version: str
    inside_validity_domain: bool
    diagnostics: Mapping[str, object]

    def __post_init__(self):
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


class RuleGraphRateProvider:
    """Supply biological rates from the existing declarative kinetic graph."""

    def __init__(self, declaration, definition, expected_outputs):
        self.definition = definition
        self.expected_outputs = tuple(expected_outputs)
        units = declaration.get("output_units")
        if not isinstance(units, Mapping) or not set(self.expected_outputs) <= set(units):
            raise ValueError("rule-graph output_units must cover all assigned outputs")
        self.units = {name: units[name] for name in self.expected_outputs}
        self.provider_id = str(declaration.get("identifier", "rule-graph"))
        self.version = str(declaration.get("version", "1"))

    def evaluate(self, snapshot, conditions):
        graph = evaluate_rule_graph(self.definition, snapshot, conditions)
        rates = {name: graph[name] for name in self.expected_outputs}
        return RateProviderResult(rates, self.units, self.provider_id, self.version,
                                  True, {"provider_type": "rule-graph"})


class AffineSurrogateRateProvider:
    """Supply declared affine rate approximations over an explicit domain."""
    def __init__(self, declaration, unused_definition, expected_outputs):
        allowed = {"type", "identifier", "version", "inputs", "outputs",
                   "validity_domain", "extrapolation"}
        if set(declaration) - allowed:
            raise ValueError("affine-surrogate declaration contains unsupported fields")
        self.provider_id = str(declaration.get("identifier", "affine-surrogate"))
        self.version = str(declaration.get("version", "1"))
        self.inputs = {str(name): dict(spec) for name, spec in declaration["inputs"].items()}
        self.outputs = {str(name): dict(spec) for name, spec in declaration["outputs"].items()}
        if set(self.outputs) != set(expected_outputs) or not self.inputs:
            raise ValueError(
                "surrogate inputs must be nonempty and outputs must match "
                "assigned outputs"
            )
        for alias, spec in self.inputs.items():
            if set(spec) != {"source", "name", "unit"} or spec["source"] not in {
                    "concentration", "state", "condition"}:
                raise ValueError(f"surrogate input {alias!r} is invalid")
            if not str(spec["name"]) or not str(spec["unit"]):
                raise ValueError(f"surrogate input {alias!r} requires name and unit")
        for name, spec in self.outputs.items():
            if set(spec) != {"unit", "intercept", "coefficients"}:
                raise ValueError(f"surrogate output {name!r} is invalid")
            if set(spec["coefficients"]) - set(self.inputs):
                raise ValueError(f"surrogate output {name!r} references an unknown input")
            if not str(spec["unit"]) or not np.isfinite(
                    [spec["intercept"], *spec["coefficients"].values()]).all():
                raise ValueError(f"surrogate output {name!r} has invalid coefficients")
        domain = declaration.get("validity_domain", {})
        if set(domain) != set(self.inputs):
            raise ValueError("surrogate validity_domain must cover every input")
        self.validity_domain = {name: tuple(map(float, limits))
                                for name, limits in domain.items()}
        if any(len(limits) != 2 or limits[0] > limits[1]
               for limits in self.validity_domain.values()):
            raise ValueError("surrogate validity bounds must be ordered pairs")
        self.extrapolation = declaration.get("extrapolation")
        if self.extrapolation not in {"error", "allow"}:
            raise ValueError("surrogate extrapolation must be explicitly 'error' or 'allow'")

    def evaluate(self, snapshot, conditions):
        inputs = {}
        for alias, spec in self.inputs.items():
            if spec["source"] == "concentration":
                value = snapshot.concentrations_mmol_l[spec["name"]]
            elif spec["source"] == "condition":
                value = conditions[spec["name"]]
            else:
                value = getattr(snapshot, spec["name"])
            inputs[alias] = float(value)
        if not np.isfinite(list(inputs.values())).all():
            raise ValueError("surrogate inputs must be finite")
        outside = tuple(name for name, value in inputs.items()
                        if not self.validity_domain[name][0] <= value
                        <= self.validity_domain[name][1])
        if outside and self.extrapolation == "error":
            raise ValueError(f"surrogate inputs outside validity domain: {list(outside)}")
        rates = {name: float(spec["intercept"] + sum(
            float(coefficient) * inputs[input_name]
            for input_name, coefficient in spec["coefficients"].items()
        )) for name, spec in self.outputs.items()}
        units = {name: spec["unit"] for name, spec in self.outputs.items()}
        return RateProviderResult(
            rates, units, self.provider_id, self.version, not outside,
            {"provider_type": "affine-surrogate", "outside_inputs": outside},
        )


class HybridRateProvider:
    """Assign required outputs to explicit mechanistic or surrogate providers."""
    def __init__(self, declaration, definition, expected_outputs):
        allowed = {"type", "identifier", "version", "providers",
                   "default_provider", "output_sources"}
        if set(declaration) - allowed:
            raise ValueError("hybrid declaration contains unsupported fields")
        items = tuple(declaration["providers"])
        identities = tuple(str(item["identifier"]) for item in items)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("hybrid provider identifiers must be nonempty and unique")
        default = declaration["default_provider"]
        sources = {name: declaration.get("output_sources", {}).get(name, default)
                   for name in expected_outputs}
        if set(sources.values()) - set(identities):
            raise ValueError("hybrid output_sources reference unknown providers")
        assigned = {identity: tuple(name for name, source in sources.items()
                                    if source == identity) for identity in identities}
        self.providers = {item["identifier"]: build_rate_provider(
            item, definition, assigned[item["identifier"]]
        ) for item in items if assigned[item["identifier"]]}
        self.sources = sources
        self.provider_id = str(declaration.get("identifier", "hybrid"))
        self.version = str(declaration.get("version", "1"))

    def evaluate(self, snapshot, conditions):
        results = {name: provider.evaluate(snapshot, conditions)
                   for name, provider in self.providers.items()}
        rates = {output: results[source].rates[output]
                 for output, source in self.sources.items()}
        units = {output: results[source].units[output]
                 for output, source in self.sources.items()}
        return RateProviderResult(
            rates, units, self.provider_id, self.version,
            all(item.inside_validity_domain for item in results.values()),
            {"provider_type": "hybrid", "output_sources": self.sources,
             "providers": tuple(results)},
        )


_PROVIDER_FACTORIES: dict[str, Callable] = {
    "rule-graph": RuleGraphRateProvider,
    "affine-surrogate": AffineSurrogateRateProvider,
    "hybrid": HybridRateProvider,
}


def register_rate_provider(provider_type, factory):
    """Register one new provider constructor without replacing existing types."""
    provider_type = str(provider_type)
    if not provider_type or not isinstance(factory, Callable):
        raise ValueError("provider registration requires a name and callable factory")
    if provider_type in _PROVIDER_FACTORIES:
        raise ValueError(f"biological-rate provider {provider_type!r} is already registered")
    _PROVIDER_FACTORIES[provider_type] = factory


def build_rate_provider(declaration, definition, expected_outputs):
    """Resolve one explicitly declared biological-rate provider."""
    if not isinstance(definition, CultureModelDefinition):
        raise TypeError("definition must be a CultureModelDefinition")
    if not isinstance(declaration, Mapping) or "type" not in declaration:
        raise ValueError("rate provider requires an explicit type")
    provider_type = declaration["type"]
    if provider_type not in _PROVIDER_FACTORIES:
        raise ValueError(f"unknown biological-rate provider {provider_type!r}")
    expected = tuple(expected_outputs)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("provider output names must be nonempty and unique")
    return _PROVIDER_FACTORIES[provider_type](declaration, definition, expected)
