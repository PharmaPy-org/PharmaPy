"""One case, two kinetic closures: pathway LP (dFBA) and configured kinetics.

Scope: #364 criterion 4. The shipped ``generic_batch`` and ``generic_fed_batch``
cases (same case.json, recipe and numerics) are solved once with their
``pathway-lp`` mechanism and once with a ``configured-kinetics`` mechanism
whose rules encode the LP's optimum. Both run through ``build_bioreactor`` and
the native SciPy backend. Each route is checked against the closed-form
solution, not against the other route alone.

Fixture case: one nutrient N and one pathway with exchange ``-a`` and growth
``g`` per unit activity; a constant uptake cap ``q`` [mol/(kgDW h)] binds, so
``mu = g q / a`` [1/h] and the yield is ``Y = g / a`` [kgDW/mol]. The closed
form is ``X = X0 exp(mu t)`` and ``N = N0 + fed - (X - X0) / Y``. Neither
6 h run reaches nutrient depletion, so the LP's depletion switch is not
exercised here; that remains covered by the pathway-specific tests.

The configured route stores biomass as a phase species, because configured
kinetics have no mechanism-owned population state. Its formula weight is a
bookkeeping value with no physical meaning; the test runs two values to show
the mapping is basis-consistent. Cost: eight short SciPy solves (a few
seconds).
"""

import json
from .input_helpers import read_input_json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import ConfiguredRates, PathwayMetabolism, build_bioreactor


EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'
BIOMASS = 'biomass'

# Absolute and relative agreement with the closed form. The cases solve with
# rtol 1e-9; 1e-6 matches the existing generic-example analytical test. The
# observed error is below 2e-7 against the closed form and 2e-10 between routes.
RTOL = 1e-6  # [-]
NUTRIENT_ATOL_MOL = 1e-9  # [mol], 1e-6 of the 1 mmol scale of the inventories
BIOMASS_ATOL_KG = 1e-12  # [kgDW], 1e-8 of the 0.1 gDW initial biomass


# Real fixture construction and independent qualification helpers.
def _load(name):
    """Read one shipped generic example.

    Parameters
    ----------
    name : str
        Example folder name.

    Returns
    -------
    tuple
        ``(case, mechanism, thermo)`` JSON mappings.
    """
    folder = EXAMPLES / name / 'inputs'
    return tuple(read_input_json(folder / f'{part}.json')
                 for part in ('case', 'mechanism', 'thermo'))


def _pathway_constants(mechanism):
    """Derive the binding uptake, growth rate and yield from the LP declaration.

    Parameters
    ----------
    mechanism : dict
        Shipped ``pathway-lp`` mechanism with one state and one pathway.

    Returns
    -------
    tuple of float
        ``(uptake, growth, yield_)``: uptake cap [mol/(kgDW h)], specific growth
        [1/h], and biomass yield on nutrient [kgDW/mol].
    """
    pathways = mechanism['model']['pathways']
    assert pathways['state_ids'] == ['nutrient'] and len(pathways['pathway_ids']) == 1
    assert mechanism['flux_time_unit'] == '1/h'
    exchange = -pathways['exchange_matrix'][0][0]  # [mol nutrient per unit activity]
    growth_per_activity = pathways['growth_coefficients'][0]  # [1/h per unit activity]
    uptake = mechanism['parameters']['rates']['uptake']['nutrient']  # [mol/(kgDW h)]
    yield_ = growth_per_activity / exchange  # [kgDW/mol]
    return uptake, uptake * yield_, yield_


def _configured(case, mechanism, thermo, biomass_mw, tmp_path):
    """Declare the same physics as configured kinetics with biomass as a species.

    Parameters
    ----------
    case : dict
        Shipped case declaration; unchanged.
    mechanism : dict
        Shipped ``pathway-lp`` mechanism; its recipes and initial amounts are
        reused.
    thermo : dict
        Shipped thermodynamic declaration.
    biomass_mw : float
        Bookkeeping formula weight of the biomass species [g/mol].
    tmp_path : Path
        Folder for the extended thermodynamic file.

    Returns
    -------
    tuple
        ``(mechanism, thermo_path)`` for the configured route.
    """
    uptake, growth, _ = _pathway_constants(mechanism)
    nutrient_mw = thermo['nutrient']['mw']  # [g/mol]
    thermo = dict(thermo, **{BIOMASS: dict(thermo['nutrient'], mw=biomass_mw)})
    thermo_path = tmp_path / 'thermo.json'
    thermo_path.write_text(json.dumps(thermo))
    state = mechanism['state']
    biomass_mol = state['biomass_kg'] * 1000. / biomass_mw  # [mol]
    # Biomass concentration [gDW/L] from the phase amount [mmol/L].
    biomass_g_l = {'op': 'multiply', 'args': [
        {'concentration': BIOMASS}, {'constant': 'biomass_g_per_mmol'}]}
    rules = [
        {'identifier': 'biomass_g_l', 'expression': biomass_g_l},
        # [g/(L h)] = [1/h] x [gDW/L]
        {'identifier': 'biomass_rate', 'expression': {'op': 'multiply', 'args': [
            {'parameter': 'rates.growth'}, {'ref': 'biomass_g_l'}]}},
        # [g/(L h)] = -[mol/(kgDW h)] x [gDW/L] x [kg/g] x [g/mol]
        {'identifier': 'nutrient_rate', 'expression': {'op': 'negate', 'args': [
            {'op': 'multiply', 'args': [
                {'parameter': 'rates.uptake'}, {'ref': 'biomass_g_l'},
                {'constant': 'kg_per_g'}, {'constant': 'nutrient_mw_g_mol'}]}]}},
    ]
    configured = {
        'schema_version': 2,
        'mechanism': 'configured-kinetics',
        'parameters': {'rates': {'growth': growth, 'uptake': uptake},
                       'affinities': {}, 'yields': {}, 'maintenance': {},
                       'responses': {}, 'coefficients': {}, 'corrections': {}},
        'model': {'pathways': {}, 'reconciliation': {}, 'kinetics': {
            'rule_graph': rules,
            'species_mass_rates': {BIOMASS: 'biomass_rate', 'nutrient': 'nutrient_rate'},
            'rate_unit': 'g/(L h)',
            'constants': {'biomass_g_per_mmol': biomass_mw / 1000.,  # [g/mmol]
                          'kg_per_g': 1e-3,  # [kg/g]
                          'nutrient_mw_g_mol': nutrient_mw}}},  # [g/mol]
        'network': None, 'rate_provider': None, 'transport': None,
        'flux_time_unit': None,
        'recipes': mechanism['recipes'],
        'state': {'species': {'unit': state['species']['unit'], 'amounts': dict(
                      state['species']['amounts'], **{BIOMASS: biomass_mol})},
                  'biomass_kg': None, 'viable_cells_million': None,
                  'dead_cells_million': None, 'product_g': None, 'integral': None},
    }
    return configured, thermo_path


def _trajectory(histories, phase, biomass_from_species):
    """Concatenate segment histories into time, nutrient and biomass arrays.

    Parameters
    ----------
    histories : tuple
        Native results from ``NativeBioreactorAssembly.solve``.
    phase : LiquidPhase
        Phase supplying species order and ``mw`` [g/mol].
    biomass_from_species : bool
        Read biomass from the phase species (configured route) instead of the
        mechanism state ``biomass_kg`` (pathway route).

    Returns
    -------
    tuple of ndarray
        ``(time_h, nutrient_mol, biomass_kg)`` in [h], [mol] and [kgDW].
    """
    names = list(phase.name_species)
    nutrient = names.index('nutrient')
    times, nutrients, biomass = [], [], []
    for history in histories:
        mass = np.asarray(history.mass_j_liquid0, dtype=float)  # [kg]
        times.append(np.asarray(history.time, dtype=float) / 3600.)  # [h]
        nutrients.append(mass[:, nutrient] * 1000. / phase.mw[nutrient])  # [mol]
        biomass.append(mass[:, names.index(BIOMASS)] if biomass_from_species else
                       np.asarray(history.biomass_kg_liquid0, dtype=float).reshape(-1))  # [kgDW]
    return np.concatenate(times), np.concatenate(nutrients), np.concatenate(biomass)


def _closed_form(mechanism, time_h):
    """Return the analytical nutrient and biomass trajectories.

    Parameters
    ----------
    mechanism : dict
        Shipped ``pathway-lp`` mechanism.
    time_h : ndarray
        Times [h], with repeated event times ordered pre- then post-addition.

    Returns
    -------
    tuple of ndarray
        ``(nutrient_mol, biomass_kg)`` in [mol] and [kgDW].
    """
    _, growth, yield_ = _pathway_constants(mechanism)
    state = mechanism['state']
    biomass0 = state['biomass_kg']  # [kgDW]
    nutrient0 = state['species']['amounts']['nutrient']  # [mol]
    assert state['species']['unit'] == 'mol'
    fed = np.zeros_like(time_h)  # [mol]
    for events in mechanism['recipes'].values():
        for event in events:
            added = event['volume_l'] * event['concentrations_mmol_l']['nutrient'] / 1000.  # [mol]
            after = time_h > event['time']
            # The first sample at the event time precedes the addition.
            at = np.flatnonzero(np.isclose(time_h, event['time'], rtol=0., atol=1e-12))
            after[at[1:]] = True
            fed += added * after
    biomass = biomass0 * np.exp(growth * time_h)  # [kgDW]
    return nutrient0 + fed - (biomass - biomass0) / yield_, biomass


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('nutrient', [0., 1e-6])
def test_configured_consumption_rejects_empty_inventory(tmp_path, nutrient):
    """Reject constant uptake at initial or subsequently depleted substrate.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated configuration directory.
    nutrient : float
        Initial nutrient inventory [mol].
    """
    case, mechanism, thermo = _load('generic_batch')
    mechanism['state']['species']['amounts']['nutrient'] = nutrient
    configured, thermo_path = _configured(case, mechanism, thermo, 1., tmp_path)
    assembly = build_bioreactor(case, configured, thermo_path)
    with pytest.raises(RuntimeError, match='negative rate at empty inventory'):
        assembly.solve()


@pytest.mark.parametrize('biomass_mw', [1000., 24.6])  # [g/mol], bookkeeping only
@pytest.mark.parametrize('name', ['generic_batch', 'generic_fed_batch'])
def test_pathway_and_configured_kinetics_solve_the_same_case(name, biomass_mw, tmp_path):
    case, pathway, thermo = _load(name)
    thermo_path = EXAMPLES / name / 'inputs/thermo.json'
    lp = build_bioreactor(case, pathway, thermo_path)
    assert isinstance(lp.mechanism, PathwayMetabolism)
    lp_time, lp_nutrient, lp_biomass = _trajectory(lp.solve(), lp.phase, False)

    configured_mechanism, configured_thermo = _configured(
        case, pathway, thermo, biomass_mw, tmp_path)
    kinetic = build_bioreactor(case, configured_mechanism, configured_thermo)
    assert isinstance(kinetic.mechanism, ConfiguredRates)
    assert type(kinetic.unit) is type(lp.unit)
    assert kinetic.events == lp.events
    k_time, k_nutrient, k_biomass = _trajectory(kinetic.solve(), kinetic.phase, True)

    np.testing.assert_array_equal(k_time, lp_time)
    nutrient, biomass = _closed_form(pathway, lp_time)
    assert biomass[-1] > 3. * biomass[0]  # growth is large enough to discriminate
    for observed_nutrient, observed_biomass in ((lp_nutrient, lp_biomass),
                                                (k_nutrient, k_biomass)):
        np.testing.assert_allclose(observed_biomass, biomass, rtol=RTOL, atol=BIOMASS_ATOL_KG)
        np.testing.assert_allclose(observed_nutrient, nutrient, rtol=RTOL,
                                   atol=NUTRIENT_ATOL_MOL)
    np.testing.assert_allclose(k_biomass, lp_biomass, rtol=RTOL, atol=BIOMASS_ATOL_KG)
    np.testing.assert_allclose(k_nutrient, lp_nutrient, rtol=RTOL, atol=NUTRIENT_ATOL_MOL)
    # Both closures report through the shared rate-provider contract.
    assert lp.mechanism.last_provider_result.units['growth_rate'] == '1/h'
    assert set(kinetic.mechanism.last_provider_result.units.values()) == {'g/(L h)'}
