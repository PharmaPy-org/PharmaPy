"""Native algebraic oxygen-demand outputs for the source F10/F11 kinetics.

Eq. 19 of doi:10.3390/bioengineering12101081 is checked independently at
asymmetric concentrations. Short public solves prove native result replay and
unchanged carbon trajectories. No dissolved-oxygen/control validation is claimed.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor


INPUTS = Path(__file__).parents[2] / 'examples/bioreactors/ecoli_ye_fed_batch/inputs'


def _inputs(run='F10'):
    """Load a short source-configured case without feed events.

    Parameters
    ----------
    run : str, optional
        Source experiment identity.

    Returns
    -------
    tuple
        Case, mechanism and thermo path, retaining all original physical units.
    """
    folder = INPUTS if run == 'F10' else INPUTS / run
    case = json.loads((folder / 'case.json').read_text())
    mechanism = json.loads((folder / 'mechanism.json').read_text())
    case['operation']['runtime']['value'] = 0.2  # [h]; short pre-feed qualification
    mechanism['recipes'][case['recipe']['name']] = []
    return case, mechanism, INPUTS / 'thermo.json'


def _specific_demand(concentrations, parameters):
    """Evaluate the paper's independent specific oxygen balance.

    Parameters
    ----------
    concentrations : Mapping
        Species concentrations [g/L], each scalar or aligned vector.
    parameters : Mapping
        Published kinetic coefficients in the example's declared bases.

    Returns
    -------
    ndarray or float
        Specific oxygen demand [g O2/(gDW h)], Eq. 19. Carbon fractions are
        from S3 Constants/LabScale, with substrate and acetate both 1/30 C-mol/g.
    """
    p = parameters
    substrate = np.maximum(concentrations['glucose'], 0.)  # [g/L]
    acetate = np.maximum(concentrations['acetate'], 0.)  # [g/L]
    fraction_a = np.maximum(concentrations['fraction_a'], 0.)  # [g/L]
    fraction_b = np.maximum(concentrations['fraction_b'], 0.)  # [g/L]
    uptake_s = p['qS_max'] * substrate / (p['K_S'] + substrate)  # [g/(gDW h)]
    uptake_a = p['qYEF_A_max'] * fraction_a / (p['K_YEF_A'] + fraction_a)  # [g/(gDW h)]
    uptake_b = p['qYEF_B_max'] * fraction_b / (p['K_YEF_B'] + fraction_b)  # [g/(gDW h)]
    oxidative = uptake_s * p['K_qSox'] / (p['K_qSox'] + uptake_s) / (
        1. + uptake_a / p['Ki_YEA_qSox'] + uptake_b / p['Ki_YEB_qSox'])  # [g/(gDW h)]
    acetate_uptake = p['qAc_max'] * acetate / (p['K_A'] + acetate) / (
        1. + substrate / p['Ki_AS'])  # [g/(gDW h)]
    growth_s = (oxidative - p['qm']) * p['Y_XSem']  # [1/h]
    growth_a = acetate_uptake * p['Y_XA']  # [1/h]
    growth_ye = uptake_a * p['Y_XYEF_A'] + uptake_b * p['Y_XYEF_B']  # [1/h]
    # S3 oxygen yields [g O2/g substrate] and carbon contents [C-mol/g].
    return (1.067 * (oxidative - growth_s * 0.041 * 30.)
            + 1.067 * (acetate_uptake - growth_a)
            + 1.163 * (uptake_a + uptake_b - growth_ye * 0.0364 * 30.))


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_native_oxygen_outputs_match_independent_equation_and_preserve_carbon(run):
    """Reported rates use accepted replay states and leave solver states unchanged.

    Parameters
    ----------
    run : str
        F10 or F11 source configuration for a short native solve.
    """
    case, mechanism, thermo = _inputs(run)
    assembly = build_bioreactor(case, mechanism, thermo)
    history = assembly.solve()[0]
    # Native kg/m3 is numerically g/L; every species is read by name.
    concentrations = {name: history.mass_j_liquid0[:, i] / np.asarray(history.vessel_vol).reshape(-1)
                      for i, name in enumerate(assembly.phase.name_species)}  # [g/L]
    expected = _specific_demand(concentrations, mechanism['parameters']['coefficients'])  # [g O2/(gDW h)]
    np.testing.assert_allclose(np.asarray(history.specific_oxygen_uptake_liquid0).reshape(-1),
                               expected, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(history.oxygen_uptake_liquid0).reshape(-1),
                               expected * concentrations['biomass'], rtol=1e-12)
    plain = deepcopy(mechanism)
    del plain['model']['kinetics']['reported_rates']
    baseline = build_bioreactor(case, plain, thermo).solve()[0]
    np.testing.assert_array_equal(history.time, baseline.time)
    np.testing.assert_array_equal(history.mass_j_liquid0, baseline.mass_j_liquid0)
    assert not hasattr(baseline, 'oxygen_uptake_liquid0')


def test_reported_rate_units_and_unknown_rule_fail_explicitly():
    """An output declaration cannot relabel a material source or invent a rule."""
    case, mechanism, thermo = _inputs()
    reports = mechanism['model']['kinetics']['reported_rates']
    reports['oxygen_uptake']['rule'] = 'absent_rule'
    with pytest.raises(ValueError, match='kinetic outputs reference undeclared rules'):
        build_bioreactor(case, mechanism, thermo)
    case, mechanism, thermo = _inputs()
    name = next(iter(mechanism['model']['kinetics']['species_mass_rates'].values()))
    mechanism['model']['kinetics']['reported_rates']['oxygen_uptake']['rule'] = name
    mechanism['model']['kinetics']['reported_rates']['oxygen_uptake']['unit'] = '1/h'
    with pytest.raises(ValueError, match='conflicting'):
        build_bioreactor(case, mechanism, thermo)


def test_oxygen_replay_uses_both_sides_of_a_dilution_event():
    """Well-mixed dilution changes volumetric demand without a stale live phase."""
    case, mechanism, thermo = _inputs()
    mechanism['recipes'][case['recipe']['name']] = [
        {'event_type': 'feed', 'time': 0.1, 'volume_l': 0.1, 'volume_basis': 'solution',
         'concentrations_g_l': {}, 'density_kg_l': 1.}]
    # Synthetic 0.1 L carrier addition at 0.1 h to the 1 L source fixture.
    assembly = build_bioreactor(case, mechanism, thermo)
    histories = assembly.solve()
    assert len(histories) == 2
    before, after = histories
    for history in histories:
        concentrations = {name: history.mass_j_liquid0[:, i] / np.asarray(history.vessel_vol).reshape(-1)
                          for i, name in enumerate(assembly.phase.name_species)}  # [g/L]
        expected = _specific_demand(concentrations, mechanism['parameters']['coefficients'])
        np.testing.assert_allclose(np.asarray(history.oxygen_uptake_liquid0).reshape(-1),
                                   expected * concentrations['biomass'], rtol=1e-12)
    assert before.time[-1] == after.time[0]
    assert before.oxygen_uptake_liquid0[-1] > after.oxygen_uptake_liquid0[0]
