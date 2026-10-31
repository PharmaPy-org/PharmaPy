"""Rule-graph rate contracts, reconciliation policies and registration."""

from dataclasses import replace
import json
from .input_helpers import read_input_json
from pathlib import Path

import numpy as np
import pytest


from PharmaPy.Bioreactors import (RateProviderResult, build_bioreactor,
                                  build_rate_provider, register_rate_provider)
from PharmaPy.Bioreactors.culture import CultureModelDefinition, CultureSnapshot




# Real fixture construction and independent qualification helpers.
class ConstantProvider:
    """Real registered extension fixture returning a declared constant 0.25 /day."""

    def __init__(self, declaration, definition, expected_outputs):
        """Store output identities for the real registered synthetic provider.

        Parameters
        ----------
        declaration : dict
            Provider configuration and output-unit declarations.
        definition : CultureModelDefinition
            Validated species/rate declaration, including the model flux and population bases.
        expected_outputs : sequence of str
            Requested rate identifiers, in their declared order.
        """
        self.outputs = tuple(expected_outputs)
    def evaluate(self, snapshot, conditions):
        """Evaluate the real named-rate or expression contract.

        Parameters
        ----------
        snapshot : CultureSnapshot
            Current concentrations [mmol/L], volume [L], populations [10^6 cells], integrated viable concentration [10^6 cell day/mL], and temperature [K].
        conditions : dict
            Input-declared operating conditions, preserving each quantity's units.

        Returns
        -------
        RateProviderResult
            Evaluated result retaining the provider or relationship's declared output units.
        """
        return RateProviderResult(
            {name: 0.25 for name in self.outputs},
            {name: "1/day" for name in self.outputs},
            "registered-fixture", "1", True, {},
        )
def _definition():
    """Declare the synthetic model used by this module's real collaborators.

    Returns
    -------
    CultureModelDefinition
        Model declaration; scientific units and bases follow the explicit fixture fields.
    """
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
    """Construct the real synthetic culture snapshot.

    Parameters
    ----------
    substrate : float
        Substrate concentration [mmol/L].

    Returns
    -------
    CultureSnapshot
        Substrate [mmol/L], volume [L], populations [10^6 cells], product [g], integrated viable concentration [10^6 cell day/mL], and temperature [K].
    """
    return CultureSnapshot({"substrate": substrate}, 1.0, 1.0, 0.0, 0.0, 0.0)


def _rule_provider(identifier="mechanistic"):
    """Declare the real rule-graph provider and explicit output units.

    Parameters
    ----------
    identifier : str
        Provider identifier used in the returned provenance.

    Returns
    -------
    dict
        Provider declaration retaining the model flux and time bases.
    """
    return {"type": "rule-graph", "identifier": identifier, "version": "1",
            "output_units": {"growth": "1/day", "death": "1/day",
                             "uptake": "model flux/day"}}




# Public-path regressions, independent balances, and failure qualification.
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
    """Verify real reconciliation bounds, target identity and objective terms.

    Notes
    -----
    R046 and R050 are deliberately untargeted. Only R067 has a unit target in
    the fixture's network coordinates. The relative-regularized policy therefore
    uses scale one for both its residual and intracellular penalty, and restricts
    only R067 to [0.5, 1.5] intersected with native bounds. These independently
    derived quantities expose accidental contributions from omitted predictions.
    Network coordinates use reaction-specific nmol or nmol-C per million cells
    per day; a numeric scale of one denotes one native unit in each coordinate.
    The solver tolerance is the existing closure tolerance, not a new allowance.
    """
    folder = Path(__file__).parents[2] / "tests/Bioreactor/fixtures/fed_batch_cho"
    case = read_input_json(folder / "inputs/case.json")
    declaration = read_input_json(folder / "inputs/mechanism.json")
    declaration["model"]["reconciliation"]["untargeted_exchanges"] = ["R046", "R050"]
    mechanism = build_bioreactor(case, declaration, folder / "inputs/thermo.json").mechanism
    native_lower, native_upper = mechanism.network.bounds({})  # [network flux coordinates]
    positions = {name: index for index, name in enumerate(mechanism.network.reaction_ids)}
    target_position = positions["R067"]
    expected_lower = native_lower.copy()  # [network flux coordinates]
    expected_upper = native_upper.copy()  # [network flux coordinates]
    # The policy's initial relative half-width is exactly 0.5 [-].
    target_value = 1.0  # [nmol-C/(10^6 cell day)], isolated growth-target fixture
    internal_scale = 1.0  # [reaction-specific native flux unit], one per coordinate
    expected_lower[target_position] = max(native_lower[target_position], 0.5 * target_value)
    expected_upper[target_position] = min(native_upper[target_position], 1.5 * target_value)
    internal_positions = [positions[name] for name in mechanism.definition.internal_reactions]
    solutions = []
    # Unequal omitted predictions straddle sign and span two orders of magnitude.
    # They are arbitrary stress inputs in network coordinates, not biological data.
    for omitted_uptake, omitted_exchange in ((-10., 20.), (1000., -2000.)):
        solution = mechanism._solve({"R046": omitted_uptake,
                                     "R050": omitted_exchange, "R067": target_value})
        diagnostics = mechanism.closure.last_diagnostics
        assert set(diagnostics.target_residuals) == {"R067"}
        np.testing.assert_array_equal(solution.lower_bounds, expected_lower)
        np.testing.assert_array_equal(solution.upper_bounds, expected_upper)
        target_residual = (solution.fluxes[target_position] - target_value) / target_value  # [-]
        internal_penalty = np.sum((solution.fluxes[internal_positions] / internal_scale)**2)  # [-]
        expected_objective = target_residual**2 + internal_penalty  # [-], declared quadratic objective
        np.testing.assert_allclose(diagnostics.target_residuals["R067"], target_residual,
                                   rtol=0., atol=mechanism.closure.tolerance)
        np.testing.assert_allclose(diagnostics.objective_value, expected_objective,
                                   rtol=mechanism.closure.tolerance, atol=mechanism.closure.tolerance)
        assert solution.mass_balance_residual_inf <= mechanism.closure.tolerance
        solutions.append(solution)
    np.testing.assert_allclose(solutions[0].fluxes, solutions[1].fluxes,
                               rtol=0., atol=mechanism.closure.tolerance)
    np.testing.assert_allclose(solutions[0].primary_objective, solutions[1].primary_objective,
                               rtol=0., atol=mechanism.closure.tolerance)


def test_native_untargeted_exchange_run_is_independent_of_omitted_target_values():
    folder = Path(__file__).parents[2] / "tests/Bioreactor/fixtures/fed_batch_cho"
    case = read_input_json(folder / "inputs/case.json")
    declaration = read_input_json(folder / "inputs/mechanism.json")
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






def test_unknown_provider_fails_before_use():
    with pytest.raises(ValueError, match="unknown biological-rate provider"):
        build_rate_provider({"type": "unknown"}, _definition(), ("growth",))


def test_external_provider_can_be_registered_without_changing_reactor_code():
    register_rate_provider("registered-test-provider", ConstantProvider)
    provider = build_rate_provider(
        {"type": "registered-test-provider"}, _definition(), ("growth",)
    )
    assert provider.evaluate(_snapshot(), {}).rates["growth"] == 0.25
    with pytest.raises(ValueError, match="already registered"):
        register_rate_provider("registered-test-provider", ConstantProvider)
