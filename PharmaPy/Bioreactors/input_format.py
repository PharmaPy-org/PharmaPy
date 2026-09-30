"""Versioned forward-input layout; translate declarations without changing models."""

from copy import deepcopy
from math import isfinite
from .BioreactorUnitConverter import BioreactorUnitConverter
from .relationships import expand_relationships


PATHWAY_FIELDS = (
    "state_ids", "pathway_ids", "exchange_matrix", "growth_coefficients",
    "objective_coefficients", "uptake_constraints", "pathway_bound_rules",
    "depletion_tolerance", "problem_name", "flux_basis", "rule_graph", "constants",
    "secondary_optimization",
)
CULTURE_FIELDS = (
    "extracellular_species", "internal_reactions", "rule_graph", "kinetic_outputs",
    "exchange_mappings", "degradation_mappings", "product_mapping", "constants",
    "normalization_exclusions", "condition_bounds", "reconciliation_relative_bound",
    "reconciliation_bound_increment", "untargeted_exchanges", "parameter_corrections",
    "problem_name", "flux_basis", "reconciliation_policy", "reconciliation_scaling",
)
MODEL_FIELDS = tuple(dict.fromkeys((*PATHWAY_FIELDS, *CULTURE_FIELDS)))
STATE_FIELDS = (
    "biomass_kg", "liquid_volume_m3", "species_mol", "viable_cells_million",
    "dead_cells_million", "product_g", "integral", "volume_l", "species_mmol",
)
MECHANISM_FIELDS = (
    "schema_version", "mechanism", "parameters", "model", "network",
    "rate_provider", "transport", "flux_time_unit", "recipes", "state",
)
NUMERICAL_FIELDS = (
    "backend", "step", "relative_tolerance", "absolute_tolerance", "maximum_step",
    "jacobian_relative_step", "jacobian_state_scale",
)
EVENT_FIELDS = (
    "time", "event_type", "volume_l", "concentrations_mmol_l", "target_species",
    "target_concentration_mmol_l",
)
UPTAKE_FIELDS = (
    "identifier", "type", "species", "maximum_parameter",
    "half_saturation_parameter", "mass_transfer_parameter", "saturation_parameter",
)
PARAMETER_GROUPS = (
    "rates", "affinities", "yields", "maintenance", "responses",
    "coefficients", "corrections",
)
KINETIC_FIELDS = (
    "extracellular_species", "rule_graph", "kinetic_outputs", "exchange_mappings",
    "degradation_mappings", "product_mapping", "constants", "condition_bounds",
    "flux_basis",
)
DIRECT_FIELDS = ('rule_graph', 'species_mass_rates', 'rate_unit', 'constants')
RECONCILIATION_FIELDS = tuple(key for key in CULTURE_FIELDS if key not in KINETIC_FIELDS)
STATE_V2_FIELDS = (
    "species", "biomass_kg", "viable_cells_million", "dead_cells_million",
    "product_g", "integral",
)


def _parameters(values, prefix=""):
    """Read compact coefficient groups as the solvers' named scalar parameters."""
    result = {}
    items = enumerate(values) if isinstance(values, list) else values.items()
    for key, value in items:
        if not str(key) or "." in str(key):
            raise ValueError("parameter identifiers must be nonempty and cannot contain dots")
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (dict, list)):
            result.update(_parameters(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            result[name] = value
        else:
            raise ValueError(f"parameter {name!r} needs a finite numeric value")
    return result


def _blank(value):
    if isinstance(value, dict):
        return all(_blank(item) for item in value.values())
    return value is None or value == []


def _provider_input(value):
    if not isinstance(value, dict) or value.get("type") not in {
            "rule-graph", "affine-surrogate", "hybrid"}:
        return value
    result = {key: item for key, item in value.items() if not _blank(item)}
    if "providers" in result:
        result["providers"] = [_provider_input(item) for item in result["providers"]]
    return result


def _version_two(case, definition):
    _keys(definition["parameters"], PARAMETER_GROUPS, "parameter groups")
    if any(not isinstance(group, dict) for group in definition["parameters"].values()):
        raise ValueError("each parameter group must be a mapping")
    definition["parameters"] = _parameters(definition["parameters"])
    for key in ("network", "rate_provider", "transport"):
        if _blank(definition.get(key)):
            definition[key] = None
    definition["rate_provider"] = _provider_input(definition.get("rate_provider"))
    kind = definition["mechanism"]
    sections = {"pathways": PATHWAY_FIELDS,
                "kinetics": tuple(dict.fromkeys((*KINETIC_FIELDS, *DIRECT_FIELDS))),
                "reconciliation": RECONCILIATION_FIELDS}
    _keys(definition["model"], sections, "model sections")
    if kind not in {"pathway-lp", "rate-reconciled-culture", "configured-kinetics"}:
        raise ValueError("select mechanism: pathway-lp, rate-reconciled-culture or configured-kinetics")
    active = ({"pathways"} if kind == "pathway-lp" else
              {"kinetics"} if kind == 'configured-kinetics' else {"kinetics", "reconciliation"})
    model = {}
    for name, fields in sections.items():
        section = definition["model"][name]
        if not isinstance(section, dict) or set(section) - set(fields):
            raise ValueError(f"unsupported fields in model.{name}")
        if name not in active and not _blank(section):
            raise ValueError(f"inactive model.{name} must be empty")
        if name in active:
            if name == 'kinetics':
                selected = DIRECT_FIELDS if kind == 'configured-kinetics' else KINETIC_FIELDS
                if any(not _blank(value) for key, value in section.items() if key not in selected):
                    raise ValueError('inactive kinetic fields must be empty')
                section = {key: value for key, value in section.items() if key in selected}
            model.update(section)
    if kind == 'configured-kinetics':
        _keys(definition, MECHANISM_FIELDS, 'mechanism')
        _keys(definition['state'], STATE_V2_FIELDS, 'state')
        _keys(definition['state']['species'], ('unit', 'amounts'), 'initial species amounts')
        definition['state'] = _active(definition['state'], {'species'}, 'state')
        definition['model'] = model
        definition.pop('schema_version')
        for key in ('network', 'transport', 'flux_time_unit'):
            if definition.pop(key) is not None:
                raise ValueError(f'inactive {key} must be null for configured kinetics')
        return
    definition["model"] = {key: model.get(key) for key in MODEL_FIELDS}
    state = definition["state"]
    _keys(state, STATE_V2_FIELDS, "state")
    _keys(state["species"], ("unit", "amounts"), "initial species amounts")
    unit = state["species"]["unit"]
    if unit not in BioreactorUnitConverter.MOLAR:
        raise ValueError("initial species amount unit must be a supported molecular amount")
    amount_key = "species_mol" if kind == "pathway-lp" else "species_mmol"
    target = "mol" if kind == "pathway-lp" else "mmol"
    factor = BioreactorUnitConverter.convert(1., unit, target)
    native_state = {key: state.get(key) for key in STATE_FIELDS}
    if not isinstance(state["species"]["amounts"], dict):
        raise ValueError("initial species amounts must be a mapping")
    native_state[amount_key] = {name: value * factor
                                for name, value in state["species"]["amounts"].items()}
    volume = case["operation"]["initial_volume"]
    factors = BioreactorUnitConverter.VOLUME
    if volume["unit"] not in factors:
        raise ValueError("initial_volume unit must be a supported volume")
    liters = BioreactorUnitConverter.convert(volume["value"], volume["unit"], "L")
    native_state["liquid_volume_m3" if kind == "pathway-lp" else "volume_l"] = (
        liters / 1000 if kind == "pathway-lp" else liters)
    definition["state"] = native_state
    definition["schema_version"] = 1


def _keys(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        unknown = sorted(set(value) - set(fields)) if isinstance(value, dict) else []
        detail = f"; unsupported {unknown}" if unknown else ""
        raise ValueError(f"{name} requires exactly {list(fields)}{detail}")


def _active(value, fields, name):
    inactive = [key for key, item in value.items() if key not in fields and item is not None]
    if inactive:
        raise ValueError(f"inactive {name} fields must be null: {inactive}")
    return {key: item for key, item in value.items() if key in fields and item is not None}


def normalize_forward_inputs(case, definition):
    """Accept the standard layout or legacy inputs and return private model inputs."""
    case, definition = deepcopy(case), deepcopy(definition)
    for document in (case, definition):
        if document.get("schema_version") == 2:
            document.pop("$comment", None)
    if "schema_version" in case:
        version = case.pop("schema_version")
        if version not in (1, 2):
            raise ValueError("unsupported case schema_version")
        if version == 2:
            _keys(case["biology"], ("conditions",), "biology")
            case["biology"]["mechanism"] = definition["mechanism"]
            case["numerics"] = {key: None if _blank(value) else value
                                for key, value in case["numerics"].items()}
        _keys(case["numerics"], NUMERICAL_FIELDS, "numerics")
        case["numerics"] = {key: value for key, value in case["numerics"].items()
                            if value is not None}
    if definition.get("schema_version") == 2:
        _version_two(case, definition)
    if definition['model'].get('rule_graph') is not None:
        definition['model'] = expand_relationships(definition['model'], definition['parameters'])
    if "schema_version" not in definition:
        return case, definition
    _keys(definition, MECHANISM_FIELDS, "mechanism")
    if definition.pop("schema_version") != 1:
        raise ValueError("unsupported mechanism schema_version")
    definition["model"].setdefault("flux_basis", None)
    definition["model"].setdefault("secondary_optimization", None)
    _keys(definition["model"], MODEL_FIELDS, "model")
    _keys(definition["state"], STATE_FIELDS, "state")
    kind = definition["mechanism"]
    if kind == "pathway-lp":
        fields = PATHWAY_FIELDS
        state_fields = STATE_FIELDS[:3]
        inactive = ("network",)
    elif kind == "rate-reconciled-culture":
        fields = CULTURE_FIELDS
        state_fields = STATE_FIELDS[3:]
        if definition['state'].get('biomass_kg') is not None:
            state_fields = ('biomass_kg', 'product_g', 'volume_l', 'species_mmol')
        inactive = ("transport", "flux_time_unit")
    else:
        raise ValueError(f"unsupported biological mechanism {kind!r}")
    definition = _active(definition, set(definition) - set(inactive), "mechanism")
    definition["model"] = _active(definition["model"], fields, "model")
    definition["state"] = _active(definition["state"], state_fields, "state")
    if kind == "pathway-lp":
        definition.setdefault("transport", None)
        definition["model"]["parameters"] = definition.pop("parameters")
        for rule in definition["model"]["uptake_constraints"]:
            _keys(rule, ('identifier', 'type', 'species', 'output')
                  if rule.get('type') == 'provider' else UPTAKE_FIELDS, "uptake constraint")
        definition["model"]["uptake_constraints"] = [
            {key: value for key, value in rule.items() if value is not None}
            for rule in definition["model"]["uptake_constraints"]
        ]
    for recipe, events in definition["recipes"].items():
        for event in events:
            optional = {'composition', 'volume_basis', 'density_kg_l', 'volume_flow',
                        'volume_l_h', 'inlet', 'volume', 'concentrations', 'concentrations_g_l', 'density'}
            if (event.get('event_type') not in {'flow', 'sample'}
                    and not {'volume', 'concentrations', 'concentrations_g_l', 'density'} & set(event)):
                _keys(event, (*EVENT_FIELDS, *(optional & set(event))), "recipe event")
        definition["recipes"][recipe] = [
            {key: value for key, value in event.items() if value is not None}
            for event in events
        ]
    return case, definition
