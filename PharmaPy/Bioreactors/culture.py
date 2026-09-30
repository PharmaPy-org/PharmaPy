"""Input-declared kinetic rule graphs for native cell-culture mechanisms."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _flatten(values, prefix=""):
    result = {}
    for key, value in dict(values).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            result.update(
                {f"{name}.{i}": float(item) for i, item in enumerate(value)}
            )
        else:
            result[name] = float(value)
    return result


@dataclass(frozen=True)
class CultureModelDefinition:
    """Declare species-independent kinetics, reconciliation, and rate mappings."""

    extracellular_species: tuple
    internal_reactions: tuple
    rule_graph: tuple
    kinetic_outputs: dict
    exchange_mappings: tuple
    degradation_mappings: tuple
    product_mapping: dict
    constants: dict
    parameters: dict
    normalization_exclusions: tuple = ()
    condition_bounds: dict = None
    reconciliation_relative_bound: float = None
    reconciliation_bound_increment: float = None
    problem_name: str = "configured rate-reconciled culture"
    parameter_corrections: dict = None
    untargeted_exchanges: tuple = ()
    flux_basis: dict = None
    reconciliation_policy: str = "relative-regularized"
    reconciliation_scaling: dict = None

    def __post_init__(self):
        object.__setattr__(self, "flux_basis", MappingProxyType({
            key: MappingProxyType(dict(value)) for key, value in (self.flux_basis or {}).items()
        }))
        species = tuple(map(str, self.extracellular_species))
        internal = tuple(map(str, self.internal_reactions))
        if not species or len(species) != len(set(species)):
            raise ValueError("extracellular species must be nonempty and unique")
        if not internal or len(internal) != len(set(internal)):
            raise ValueError("internal reactions must be nonempty and unique")
        rules = tuple(MappingProxyType(dict(item)) for item in self.rule_graph)
        identifiers = [item.get("identifier") for item in rules]
        if any(not item for item in identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError("rule identifiers must be nonempty and unique")
        object.__setattr__(self, "extracellular_species", species)
        object.__setattr__(self, "internal_reactions", internal)
        object.__setattr__(self, "rule_graph", rules)
        object.__setattr__(self, "kinetic_outputs", MappingProxyType(dict(self.kinetic_outputs)))
        object.__setattr__(
            self,
            "exchange_mappings",
            tuple(MappingProxyType(dict(item)) for item in self.exchange_mappings),
        )
        object.__setattr__(
            self,
            "degradation_mappings",
            tuple(MappingProxyType(dict(item)) for item in self.degradation_mappings),
        )
        object.__setattr__(self, "product_mapping", MappingProxyType(dict(self.product_mapping)))
        object.__setattr__(self, "constants", MappingProxyType(_flatten(self.constants)))
        parameters = _flatten(self.parameters)
        corrections = dict(self.parameter_corrections or {})
        for name, declaration in corrections.items():
            if name not in parameters:
                raise ValueError(f"correction references unknown parameter {name!r}")
            if set(declaration) - {"kind", "choice", "source"}:
                raise ValueError(f"unknown correction fields for {name!r}")
            kind, choice = declaration.get("kind"), declaration.get("choice")
            if kind not in {"multiplicative", "additive"}:
                raise ValueError("correction kind must be multiplicative or additive")
            if choice not in {"neutral", "supplied"}:
                raise ValueError("correction choice must be neutral or supplied")
            value = parameters[name]
            if not np.isfinite(value):
                raise ValueError(f"correction {name!r} must be finite")
            if choice == "neutral" and value != (1.0 if kind == "multiplicative" else 0.0):
                raise ValueError(f"neutral correction {name!r} requires its neutral value")
            source = declaration.get("source")
            if choice == "supplied" and (not isinstance(source, str) or not source.strip()):
                raise ValueError(f"supplied correction {name!r} requires a source")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "parameter_corrections", MappingProxyType({
            name: MappingProxyType(dict(item)) for name, item in corrections.items()
        }))
        untargeted = tuple(self.untargeted_exchanges)
        exchanges = set(self.kinetic_outputs["exchange_fluxes"])
        if len(untargeted) != len(set(untargeted)) or not set(untargeted) <= exchanges:
            raise ValueError("untargeted exchanges must be unique declared exchange reactions")
        if not exchanges - set(untargeted):
            raise ValueError("rate reconciliation requires at least one kinetic target")
        object.__setattr__(self, "untargeted_exchanges", untargeted)
        object.__setattr__(
            self,
            "condition_bounds",
            MappingProxyType(dict(self.condition_bounds or {})),
        )
        if self.reconciliation_policy not in {"relative-regularized", "unweighted", "unit-scaled"}:
            raise ValueError("unknown reconciliation_policy")
        if self.reconciliation_policy == 'unit-scaled' and not self.flux_basis:
            raise ValueError('unit-scaled reconciliation requires declared flux_basis')
        if self.reconciliation_policy != "relative-regularized":
            if (self.reconciliation_relative_bound is not None
                    or self.reconciliation_bound_increment is not None):
                raise ValueError(f"{self.reconciliation_policy} reconciliation requires null relative target bounds")
            _validate_graph(self)
            return
        bound = self.reconciliation_relative_bound
        if bound is None or not np.isfinite(bound) or bound < 0.5:
            raise ValueError(
                "reconciliation_relative_bound must be explicitly supplied "
                "as a finite value of at least 0.5"
            )
        object.__setattr__(self, "reconciliation_relative_bound", float(bound))
        increment = self.reconciliation_bound_increment
        if increment is None or not np.isfinite(increment) or increment <= 0.0:
            raise ValueError(
                "reconciliation_bound_increment must be explicitly supplied "
                "as a finite positive value"
            )
        object.__setattr__(self, "reconciliation_bound_increment", float(increment))
        _validate_graph(self)

    @classmethod
    def from_mapping(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class CultureSnapshot:
    """Expose the native reactor state to a configured kinetic rule graph."""

    concentrations_mmol_l: dict
    volume_l: float
    viable_cells_million: float
    dead_cells_million: float
    product_g: float
    ivcd_million_cell_day_per_ml: float
    biomass_kg: float = 0.0

    @property
    def viable_cell_density_million_ml(self):
        return self.viable_cells_million / (self.volume_l * 1000.0)


def _references(expression, key="ref"):
    if not isinstance(expression, dict):
        return set()
    found = {expression[key]} if set(expression) == {key} else set()
    for value in expression.values():
        if isinstance(value, dict):
            found.update(_references(value, key))
        elif isinstance(value, list):
            for item in value:
                found.update(_references(item, key))
    return found


def _validate_graph(definition, outputs=None):
    available = set()
    for rule in definition.rule_graph:
        if set(rule) != {"identifier", "expression"}:
            raise ValueError("rules require only identifier and expression")
        if _references(rule['expression'], 'accepted_flux'):
            raise ValueError('accepted fluxes are only available after reconciliation')
        unknown = _references(rule["expression"]) - available
        if unknown:
            raise ValueError(f"rule graph contains forward references {sorted(unknown)}")
        available.add(rule["identifier"])
    required = (set(outputs) if outputs is not None else
                {definition.kinetic_outputs["growth"], definition.kinetic_outputs["death"],
                 *dict(definition.kinetic_outputs["exchange_fluxes"]).values()})
    if not required <= available:
        raise ValueError("kinetic outputs reference undeclared rules")


def _evaluate(expr, values, definition, state, conditions, accepted_fluxes=None):
    if isinstance(expr, (int, float)):
        return float(expr)
    if set(expr) == {"ref"}:
        return values[expr["ref"]]
    if set(expr) == {"parameter"}:
        return definition.parameters[expr["parameter"]]
    if set(expr) == {"constant"}:
        return definition.constants[expr["constant"]]
    if set(expr) == {"concentration"}:
        return state.concentrations_mmol_l[expr["concentration"]]
    if set(expr) == {"state"}:
        return float(getattr(state, expr["state"]))
    if set(expr) == {"condition"}:
        return float(conditions[expr["condition"]])
    if set(expr) == {'accepted_flux'}:
        if accepted_fluxes is None:
            raise ValueError('accepted fluxes are only available after reconciliation')
        return accepted_fluxes[expr['accepted_flux']]
    op = expr.get("op")
    if op == "greater-select":
        condition, positive, negative = expr["args"]
        branch = (positive if _evaluate(condition, values, definition, state, conditions, accepted_fluxes)
                  > float(expr["threshold"]) else negative)
        return _evaluate(branch, values, definition, state, conditions, accepted_fluxes)
    args = [
        _evaluate(item, values, definition, state, conditions, accepted_fluxes)
        for item in expr.get("args", ())
    ]
    if op == "add":
        return sum(args)
    if op == "multiply":
        return float(np.prod(args))
    if op == "subtract":
        return args[0] - args[1]
    if op == "divide":
        return args[0] / args[1]
    if op == "negate":
        return -args[0]
    if op == "exp":
        return float(np.exp(args[0]))
    if op == "maximum":
        return max(args)
    if op == "minimum":
        return min(args)
    if op == "monod":
        concentration = max(args[0], 0.0)
        return concentration / (args[1] + concentration)
    if op == "quadratic":
        return args[0] * args[3] ** 2 + args[1] * args[3] + args[2]
    raise ValueError(f"unsupported kinetic operator {op!r}")


def evaluate_rule_graph(definition, state, conditions):
    """Evaluate all input-declared scalar kinetic equations in dependency order."""
    values = {}
    for rule in definition.rule_graph:
        value = _evaluate(rule["expression"], values, definition, state, conditions)
        if not np.isfinite(value):
            raise ValueError(f"kinetic rule {rule['identifier']!r} is nonfinite")
        values[rule["identifier"]] = value
    return MappingProxyType(values)
