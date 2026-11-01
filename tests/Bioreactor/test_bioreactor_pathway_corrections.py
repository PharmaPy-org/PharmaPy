"""Real provider-to-LP correction handoff, depletion, transfer limits and histories.

Neutral configurations are compared with the original built-in constraints.
Nonzero coefficients use independent scalar capacities and the real LP solver.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Metabolic.closures.base import MetabolicNumericalError

ROOT = Path(__file__).parents[2] / 'examples/bioreactors'
CASES = ('generic_batch', 'generic_fed_batch', 'generic_chemostat', 'batch_ecoli_dfba')


# Real fixture construction and independent qualification helpers.
def load_case(name: str) -> dict:
    """Read one complete example without changing its physical units.

    Parameters
    ----------
    name : str
        Example directory.

    Returns
    -------
    dict
        Combined forward declaration with concentrations in mmol/L.
    """
    return json.loads((ROOT / name / 'inputs/mechanism.json').read_text())


def original_constraints(document: dict, name: str) -> dict:
    """Restore the prior constant/Monod route for independent comparisons.

    Parameters
    ----------
    document : dict
        New example declaration; retained unchanged.
    name : str
        Example directory determining its original kinetic form.

    Returns
    -------
    dict
        Original built-in uptake declarations, with physical transfer caps intact.
    """
    result = deepcopy(document)
    result['rate_provider'] = None
    section = result['model']['pathways']
    section.pop('rule_graph')
    species = 'glucose' if name == 'batch_ecoli_dfba' else 'nutrient'
    monod = name in ('generic_chemostat', 'batch_ecoli_dfba')
    section['uptake_constraints'][0] = {
        'identifier': species + '_uptake', 'type': 'monod' if monod else 'constant',
        'species': species, 'maximum_parameter': 'rates.uptake.' + species,
        'half_saturation_parameter': 'affinities.' + species if monod else None,
        'mass_transfer_parameter': None, 'saturation_parameter': None}
    return result


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.integration
@pytest.mark.parametrize('name', CASES)
@pytest.mark.parametrize('theta', [-np.log(2.), 0., np.log(2.)])
def test_provider_capacities_and_lp_constraints_across_depletion(name, theta):
    """Sweep negative trials, the cutoff, affinity and saturation through real LPs.

    Parameters
    ----------
    name : str
        Example directory.
    theta : float
        Log multiplier [-]; +/-ln(2) bounds the factor between one-half and two.
    """
    document = load_case(name)
    assert document['parameters']['corrections']['theta'] == 0.
    original = deepcopy(document)
    document['parameters']['corrections']['theta'] = theta  # [-]
    thermo = ROOT / name / 'inputs/thermo.json'
    assembly = build_bioreactor(document, thermo_path=thermo)
    reference = build_bioreactor(original_constraints(document, name), thermo_path=thermo)
    mechanism = assembly.mechanism
    rule = mechanism.model.definition.uptake_constraints[0]
    species = rule['species']
    maximum = document['parameters']['rates']['uptake'][species]  # [mol/(kgDW h)]
    monod = name in ('generic_chemostat', 'batch_ecoli_dfba')
    scale = (document['parameters']['affinities'][species] if monod else
             document['parameters']['corrections']['reference_concentration'])  # [mmol/L]
    cutoff = document['model']['pathways']['depletion_tolerance']  # [mmol/L]
    # One hundred affinities probes the saturated regime; half/twice cutoff bracket depletion.
    for concentration in (-scale, 0., cutoff / 2., cutoff, 2. * cutoff, scale, 100. * scale):
        concentrations = dict.fromkeys(mechanism.model.state_names, scale)  # [mmol/L]
        concentrations[species] = concentration  # [mmol/L]
        snapshot = SimpleNamespace(concentrations_mmol_l=concentrations)
        supplied = mechanism.uptake_provider.evaluate(snapshot, {}).rates
        nonnegative = max(concentration, 0.)  # [mmol/L], existing native extension
        base = (0. if nonnegative <= cutoff else maximum * nonnegative / (scale + nonnegative)) if monod else maximum  # [mol/(kgDW h)]
        expected = base * np.exp(theta * nonnegative / (scale + nonnegative))  # [mol/(kgDW h)]
        assert supplied[rule['output']] == pytest.approx(expected, rel=1e-13, abs=0.)
        # Both dilute and concentrated biomass expose switching oxygen-transfer limits.
        for biomass in (.001, 1., 100.):  # [kgDW/m3], synthetic solver qualification states
            if name == 'generic_chemostat' and concentration == 2. * cutoff:
                # Existing LP optimal-face guard rejects this tiny positive optimum
                # below HiGHS absolute feasibility resolution. Preserve the error
                # on both routes; do not treat an uncertified zero flux as success.
                with pytest.raises(MetabolicNumericalError, match='secondary solve changed the primary optimum'):
                    reference.mechanism.model.solve_fluxes(concentrations, biomass)
                with pytest.raises(MetabolicNumericalError, match='secondary solve changed the primary optimum'):
                    mechanism.model.solve_fluxes(concentrations, biomass, uptake_limits=supplied)
                continue
            actual = mechanism.model.solve_fluxes(concentrations, biomass, uptake_limits=supplied)
            baseline = reference.mechanism.model.solve_fluxes(concentrations, biomass)
            assert actual.constraint_limits[rule['identifier']] == pytest.approx(expected, rel=1e-13, abs=0.)
            assert actual.bound_violation_inf <= 1e-8  # [flux unit], existing LP feasibility budget
            if theta == 0.:
                np.testing.assert_allclose(actual.fluxes, baseline.fluxes, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual.extracellular_rates, baseline.extracellular_rates, rtol=1e-12, atol=1e-12)
            if name == 'batch_ecoli_dfba':
                assert actual.constraint_limits['oxygen_uptake'] == baseline.constraint_limits['oxygen_uptake']
                oxygen = mechanism.model.state_names.index('oxygen')
                assert -actual.extracellular_rates[oxygen] <= actual.constraint_limits['oxygen_uptake'] + 1e-8
    original['parameters']['corrections']['theta'] = theta  # [-], only caller-authorized mutation
    assert document == original


@pytest.mark.integration
@pytest.mark.parametrize('name', CASES)
def test_zero_theta_preserves_original_full_native_trajectory(name):
    """Preserve original complete histories, event segmentation and state order.

    Parameters
    ----------
    name : str
        Batch, fed-batch, chemostat or multi-pathway example.
    """
    document = load_case(name)
    thermo = ROOT / name / 'inputs/thermo.json'
    actual = build_bioreactor(document, thermo_path=thermo)
    reference = build_bioreactor(original_constraints(document, name), thermo_path=thermo)
    observed = actual.solve(verbose=False)
    expected = reference.solve(verbose=False)
    assert actual.events == reference.events
    assert actual.phase.name_species == reference.phase.name_species
    assert len(observed) == len(expected) > 0
    for first, second in zip(observed, expected):
        np.testing.assert_array_equal(first.time, second.time)
        for field in ('mass_j_liquid0', 'biomass_kg_liquid0', 'vessel_vol'):
            # Same equations; budget below configured integration tolerances.
            np.testing.assert_allclose(getattr(first, field), getattr(second, field),
                                       rtol=1e-11, atol=1e-14)


@pytest.mark.integration
def test_nonzero_theta_reaches_native_balances_without_bypassing_inventory():
    """Higher capacity increases growth while preserving the original molar yield."""
    biomass = []  # [kgDW], terminal states in increasing theta order
    nutrient = []  # [mol], terminal states in increasing theta order
    for theta in (-np.log(2.), 0., np.log(2.)):  # [-], at most a twofold correction
        document = load_case('generic_batch')
        document['parameters']['corrections']['theta'] = theta  # [-]
        document['operation']['runtime']['value'] = 1.  # [h], transient before depletion
        assembly = build_bioreactor(document, thermo_path=ROOT/'generic_batch/inputs/thermo.json')
        history = assembly.solve(verbose=False)[-1]
        index = list(assembly.phase.name_species).index('nutrient')
        nutrient_mol = history.mass_j_liquid0[:, index] * 1000. / assembly.phase.mw[index]  # [mol], kg -> g -> mol
        biomass_kg = history.biomass_kg_liquid0  # [kgDW]
        # The original LP consumes 10 mol nutrient per kgDW formed, with no feed.
        conserved = nutrient_mol + 10. * biomass_kg  # [mol nutrient equivalent]
        np.testing.assert_allclose(conserved, .011, rtol=0., atol=1e-10)  # [mol], 0.01 + 10*0.0001; below solver budget
        biomass.append(biomass_kg[-1])
        nutrient.append(nutrient_mol[-1])
    assert biomass[0] < biomass[1] < biomass[2]
    assert nutrient[0] > nutrient[1] > nutrient[2] > 0.


@pytest.mark.unit
@pytest.mark.parametrize('field,value,message', [
    ('theta', float('nan'), 'finite'),
    ('theta', True, 'finite'),
    ('reference_concentration', 0., 'positive'),
    ('reference_concentration', -1., 'positive'),
])
def test_pathway_input_rejects_invalid_correction_parameters(field, value, message):
    """Validate the same public correction contract on pathway inputs.

    Parameters
    ----------
    field : str
        Correction parameter name.
    value : object
        Invalid theta [-] or reference concentration [mmol/L].
    message : str
        Required actionable error fragment.
    """
    document = load_case('generic_batch')
    document['parameters']['corrections'][field] = value
    with pytest.raises(ValueError, match=message):
        build_bioreactor(document, thermo_path=ROOT/'generic_batch/inputs/thermo.json')


@pytest.mark.integration
def test_pathway_provider_can_select_haldane_and_apply_a_nonzero_correction():
    """Select Haldane via JSON slots and double a hand-calculated capacity."""
    document = load_case('generic_chemostat')
    expression = document['model']['pathways']['rule_graph'][0]['expression']['args'][1]
    expression['relationship'] = 'haldane'
    expression['inputs']['inhibition_constant'] = 4.  # [mmol/L], synthetic peak at sqrt(Ks*Ki)=2 mmol/L
    document['parameters']['corrections']['theta'] = 1.5 * np.log(2.)  # [-], S/(S+Ks)=2/3 makes multiplier two
    assembly = build_bioreactor(document, thermo_path=ROOT/'generic_chemostat/inputs/thermo.json')
    concentrations = {'nutrient': 2.}  # [mmol/L], synthetic peak
    supplied = assembly.mechanism.uptake_provider.evaluate(SimpleNamespace(concentrations_mmol_l=concentrations), {})
    # Base Haldane: maximum=4, S=2, Ks=1, Ki=4 gives 2 mol/(kgDW h).
    assert supplied.rates['nutrient_uptake_capacity'] == pytest.approx(4.)
    result = assembly.mechanism.model.solve_fluxes(concentrations, 1., uptake_limits=supplied.rates)  # biomass [kgDW/m3]
    assert result.growth_rate == pytest.approx(.4)  # [1/h], 4 mol/(kgDW h) / 10 mol/kgDW
