import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest


from PharmaPy.Bioreactors import (RateProviderResult, build_bioreactor,
                                  build_rate_provider, register_rate_provider)
from PharmaPy.Bioreactors.culture import CultureModelDefinition, CultureSnapshot
from PharmaPy.Reactors_Refactored import SemiBatchReactor


def _definition():
    return CultureModelDefinition.from_mapping({
        "extracellular_species": ["substrate"],
        "internal_reactions": ["uptake"],
        "rule_graph": [
            {"identifier": "growth", "expression": {
                "op": "multiply", "args": [
                    {"parameter": "mu"},
                    {"op": "monod", "args": [
                        {"concentration": "substrate"}, {"parameter": "half"}
                    ]}
                ]}},
            {"identifier": "death", "expression": {"parameter": "death"}},
            {"identifier": "uptake", "expression": {
                "op": "negate", "args": [{"ref": "growth"}]
            }},
        ],
        "kinetic_outputs": {"growth": "growth", "death": "death",
                            "exchange_fluxes": {"uptake": "uptake"}},
        "exchange_mappings": [], "degradation_mappings": [],
        "product_mapping": {}, "constants": {},
        "parameters": {"mu": 0.8, "half": 1.0, "death": 0.02},
        "reconciliation_relative_bound": 0.5,
        "reconciliation_bound_increment": 0.1,
    })


def _snapshot(substrate=2.0):
    return CultureSnapshot({"substrate": substrate}, 1.0, 1.0, 0.0, 0.0, 0.0)


def _rule_provider(identifier="mechanistic"):
    return {"type": "rule-graph", "identifier": identifier, "version": "1",
            "output_units": {"growth": "1/day", "death": "1/day",
                             "uptake": "model flux/day"}}


def _surrogate(outputs=("growth",), extrapolation="error"):
    configured = {}
    for output in outputs:
        configured[output] = {
            "unit": "1/day" if output in {"growth", "death"} else "model flux/day",
            "intercept": 0.1, "coefficients": {"substrate": 0.2},
        }
    return {
        "type": "affine-surrogate", "identifier": "surrogate", "version": "2026-09",
        "inputs": {"substrate": {"source": "concentration",
                                  "name": "substrate", "unit": "mmol/L"}},
        "outputs": configured, "validity_domain": {"substrate": [0.0, 5.0]},
        "extrapolation": extrapolation,
    }


def test_rule_graph_provider_satisfies_named_rate_contract():
    provider = build_rate_provider(
        _rule_provider(), _definition(), ("growth", "death", "uptake")
    )
    result = provider.evaluate(_snapshot(), {})
    assert result.rates["growth"] == pytest.approx(0.8 * 2.0 / 3.0)
    assert result.rates["uptake"] == pytest.approx(-result.rates["growth"])
    assert result.units["growth"] == "1/day"
    assert result.provider_id == "mechanistic"
    assert result.inside_validity_domain


@pytest.mark.parametrize('condition, guarded_branch', [(0., 1), (-1., 1), (1., 2)])
def test_conditional_rate_evaluates_only_selected_branch(condition, guarded_branch):
    definition = _definition()
    rules = [dict(rule) for rule in definition.rule_graph]
    args = [condition, 0.25, 0.5]
    args[guarded_branch] = {'op': 'divide', 'args': [1., 0.]}
    rules[0]['expression'] = {'op': 'greater-select', 'threshold': 0., 'args': args}
    provider = build_rate_provider(_rule_provider(), replace(definition, rule_graph=rules),
                                   ('growth', 'death', 'uptake'))
    expected = 0.25 if condition > 0. else 0.5
    assert provider.evaluate(_snapshot(), {}).rates['growth'] == expected
    args[0] = 1. if guarded_branch == 1 else 0.
    with pytest.raises(ZeroDivisionError):
        provider.evaluate(_snapshot(), {})


@pytest.mark.parametrize("kind,value", [("multiplicative", 1.0), ("additive", 0.0)])
def test_declared_corrections_validate_neutral_values_and_supplied_provenance(kind, value):
    definition = _definition()
    parameters = dict(definition.parameters, correction=value)
    policy = {"correction": {"kind": kind, "choice": "neutral"}}
    configured = replace(definition, parameters=parameters, parameter_corrections=policy)
    assert configured.parameters["correction"] == value
    with pytest.raises(ValueError, match="neutral value"):
        replace(configured, parameters=dict(parameters, correction=3.0))
    policy = {"correction": {"kind": kind, "choice": "supplied"}}
    with pytest.raises(ValueError, match="requires a source"):
        replace(configured, parameter_corrections=policy)
    policy["correction"]["source"] = "Declared scenario assumption, not calibration"
    supplied = replace(configured, parameters=dict(parameters, correction=3.0),
                       parameter_corrections=policy)
    assert supplied.parameters["correction"] == 3.0
    with pytest.raises(ValueError, match="finite"):
        replace(supplied, parameters=dict(parameters, correction=float("nan")))


@pytest.mark.parametrize("reactions", [("missing",), ("uptake", "uptake"), ("uptake",)])
def test_untargeted_exchanges_reject_invalid_or_empty_target_sets(reactions):
    with pytest.raises(ValueError):
        replace(_definition(), untargeted_exchanges=reactions)


def test_untargeted_rates_do_not_change_penalties_bounds_or_normalization():
    folder = Path(__file__).parents[2] / "tests/Bioreactor/fixtures/fed_batch_cho"
    case = json.loads((folder / "inputs/case.json").read_text())
    declaration = json.loads((folder / "inputs/mechanism.json").read_text())
    declaration["model"]["reconciliation"]["untargeted_exchanges"] = ["R046", "R050"]
    mechanism = build_bioreactor(case, declaration, folder / "inputs/thermo.json").mechanism
    calls = []

    def capture(environment, previous, **kwargs):
        calls.append((environment, kwargs))
        return object()

    mechanism.closure.solve = capture
    mechanism._solve({"R046": -10., "R050": 20., "R067": 1.})
    mechanism._solve({"R046": 1000., "R050": -2000., "R067": 1.})
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]["internal_scale"] == 1.0
    assert [target.reaction_id for target in calls[0][1]["targets"]] == ["R067"]
    assert set(calls[0][0].bound_overrides) == {"R067"}


def test_native_untargeted_exchange_run_is_independent_of_omitted_target_values():
    folder = Path(__file__).parents[2] / "tests/Bioreactor/fixtures/fed_batch_cho"
    case = json.loads((folder / "inputs/case.json").read_text())
    declaration = json.loads((folder / "inputs/mechanism.json").read_text())
    case["operation"]["runtime"] = {"unit": "day", "value": 0.1}
    declaration["model"]["reconciliation"]["untargeted_exchanges"] = ["R046", "R050"]
    first = build_bioreactor(case, declaration, folder / "inputs/thermo.json")
    first.solve(verbose=False)
    for name in declaration["model"]["reconciliation"]["parameter_corrections"]:
        target = declaration['parameters']
        keys = name.split('.')
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = 100.0
        declaration["model"]["reconciliation"]["parameter_corrections"][name].update(
            choice="supplied", source="Target-independence test scenario")
    second = build_bioreactor(case, declaration, folder / "inputs/thermo.json")
    second.solve(verbose=False)
    np.testing.assert_allclose(first.unit.result.mass_j_liquid0,
                               second.unit.result.mass_j_liquid0, rtol=1e-10, atol=1e-12)
    solution = first.mechanism.last_solution
    lower, upper = first.mechanism.network.bounds({})
    for reaction in ("R046", "R050"):
        i = solution.reaction_ids.index(reaction)
        assert solution.lower_bounds[i] == lower[i]
        assert solution.upper_bounds[i] == upper[i]


def test_affine_surrogate_is_dimensioned_deterministic_and_domain_checked():
    provider = build_rate_provider(_surrogate(), _definition(), ("growth",))
    first = provider.evaluate(_snapshot(2.0), {})
    second = provider.evaluate(_snapshot(2.0), {})
    assert first.rates == second.rates
    assert first.rates["growth"] == pytest.approx(0.5)
    assert first.units["growth"] == "1/day"
    with pytest.raises(ValueError, match="outside validity domain"):
        provider.evaluate(_snapshot(6.0), {})
    allowed = build_rate_provider(
        _surrogate(extrapolation="allow"), _definition(), ("growth",)
    ).evaluate(_snapshot(6.0), {})
    assert not allowed.inside_validity_domain


def test_hybrid_provider_assigns_each_output_to_one_declared_owner():
    declaration = {
        "type": "hybrid", "identifier": "hybrid", "version": "1",
        "providers": [_rule_provider(), _surrogate()],
        "default_provider": "mechanistic",
        "output_sources": {"growth": "surrogate"},
    }
    result = build_rate_provider(
        declaration, _definition(), ("growth", "death", "uptake")
    ).evaluate(_snapshot(), {})
    assert result.rates["growth"] == pytest.approx(0.5)
    assert result.rates["death"] == pytest.approx(0.02)
    assert result.diagnostics["output_sources"]["growth"] == "surrogate"


def test_unknown_incompatible_and_nonfinite_providers_fail_before_use():
    with pytest.raises(ValueError, match="unknown biological-rate provider"):
        build_rate_provider({"type": "unknown"}, _definition(), ("growth",))
    missing_unit = _surrogate()
    del missing_unit["outputs"]["growth"]["unit"]
    with pytest.raises(ValueError, match="surrogate output"):
        build_rate_provider(missing_unit, _definition(), ("growth",))
    nonfinite = _surrogate()
    nonfinite["outputs"]["growth"]["intercept"] = np.nan
    with pytest.raises(ValueError, match="invalid coefficients"):
        build_rate_provider(nonfinite, _definition(), ("growth",))
    hybrid = {"type": "hybrid", "providers": [_rule_provider()],
              "default_provider": "missing", "output_sources": {}}
    with pytest.raises(ValueError, match="unknown providers"):
        build_rate_provider(hybrid, _definition(), ("growth",))


def test_external_provider_can_be_registered_without_changing_reactor_code():
    class ConstantProvider:
        def __init__(self, declaration, definition, expected_outputs):
            self.outputs = tuple(expected_outputs)
        def evaluate(self, snapshot, conditions):
            return RateProviderResult(
                {name: 0.25 for name in self.outputs},
                {name: "1/day" for name in self.outputs},
                "registered-fixture", "1", True, {},
            )
    register_rate_provider("registered-test-provider", ConstantProvider)
    provider = build_rate_provider(
        {"type": "registered-test-provider"}, _definition(), ("growth",)
    )
    assert provider.evaluate(_snapshot(), {}).rates["growth"] == 0.25
    with pytest.raises(ValueError, match="already registered"):
        register_rate_provider("registered-test-provider", ConstantProvider)


def test_provider_substitution_keeps_native_reactor_and_recipe_lifecycle():
    folder = Path(__file__).parents[2] / "tests/Bioreactor/fixtures/fed_batch_cho"
    case = json.loads((folder / "inputs/case.json").read_text())
    definition = json.loads((folder / "inputs/mechanism.json").read_text())
    case["operation"]["runtime"] = {"unit": "day", "value": 0.1}
    baseline = build_bioreactor(case, definition, folder / "inputs/thermo.json")
    baseline.solve(verbose=False)

    hybrid_definition = copy.deepcopy(definition)
    surrogate = {
        "type": "affine-surrogate", "identifier": "growth-surrogate", "version": "1",
        "inputs": {"ph": {"source": "condition", "name": "ph", "unit": "-"}},
        "outputs": {"growth": {"unit": "1/day", "intercept": 0.5,
                                "coefficients": {"ph": 0.0}}},
        "validity_domain": {"ph": [6.75, 7.25]}, "extrapolation": "error",
    }
    hybrid_definition["rate_provider"] = {
        "type": "hybrid", "identifier": "demonstrated-hybrid", "version": "1",
        "providers": [definition["rate_provider"], surrogate],
        "default_provider": definition["rate_provider"]["identifier"],
        "output_sources": {"growth": "growth-surrogate"},
    }
    hybrid = build_bioreactor(case, hybrid_definition, folder / "inputs/thermo.json")
    hybrid.solve(verbose=False)
    assert type(baseline.unit) is type(hybrid.unit) is SemiBatchReactor
    assert baseline.events == hybrid.events
    assert type(baseline.unit.result) is type(hybrid.unit.result)
    assert hybrid.mechanism.last_rates["growth"] == pytest.approx(0.5)
    assert hybrid.mechanism.rate_provider.provider_id == "demonstrated-hybrid"
