"""#39 Jacobians on total moments, with 10% solids relative to liquid volume.

Real phases and kinetics drive unit_model without a solver. Parameter probes
use its params argument to exercise the #222 solver-vector handoff.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase

pytestmark = pytest.mark.unit

TEMPERATURE = 320.0  # [K], synthetic isothermal case
LIQUID_VOLUME = 2.0  # [m**3]
KV = 0.5  # [-], non-unit crystal shape factor
LENGTH = 1e-4  # [m], monodisperse 100 um crystals
NUMBER = 4e11  # [#], gives 0.2 m**3 of solids, 10% of liquid volume
MOMENTS = NUMBER * LENGTH ** np.arange(4)  # [m**n], total moments
# Central differences balance O(h**2) truncation and O(eps/h) cancellation.
FD_STEP = np.cbrt(np.finfo(float).eps)  # [-], relative perturbation
FD_RTOL = 2e-7  # [-], roundoff allowance for differenced multi-term RHS values


def make_unit(data_path, moment_basis, concentration=4.0):
    """Build an isothermal total-moment state with finite crystal inventory.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    moment_basis : str
        Secondary nucleation moment, area or volume.
    concentration : float, optional
        Target concentration [kg/m**3]; saturation is 2 kg/m**3.

    Returns
    -------
    unit : BatchCryst
        Real model using synthetic power laws, not calibrated kinetics.
    states : ndarray
        Total moments [um**n], concentrations [kg/m**3], liquid volume [m**3].
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    unit = BatchCryst('A', method='moments',
                      controls={'temp': lambda time: TEMPERATURE})
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=LIQUID_VOLUME,
                         mass_frac=[0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
    solid = SolidPhase(path, temp=TEMPERATURE, moments=MOMENTS, kv=KV,
                       mass_frac=[1, 0, 0, 0, 0])  # [-], pure A crystals
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(
        coeff_solub=[2.0],  # [kg/m**3], constant synthetic solubility
        nucl_prim=(2e8, 0.0, 1.2),  # [#/m**3/s], [J/mol], [-]
        nucl_sec=(3e8, 0.0, 1.4, 0.7),  # [rate/(kv*mu_j)**s_2], [J/mol], [-], [-]
        growth=(1.0, 0.0, 1.1),  # [um/s], [J/mol], [-]
        dissolution=(2.0, 0.0, 1.3),  # [um/s], [J/mol], [-]
        mu_sec_nucl=moment_basis)
    unit.Kinetics.target_idx = unit.target_ind
    unit.num_species = len(liquid.mass_conc)
    concentrations = liquid.mass_conc.copy()  # [kg/m**3]
    concentrations[0] = concentration  # [kg/m**3]
    states = np.concatenate((MOMENTS * 1e6 ** np.arange(4),
                             concentrations, [LIQUID_VOLUME]))  # units in Returns
    return unit, states


@pytest.mark.parametrize('moment_basis', ['volume', 'area'])
@pytest.mark.parametrize('column', [2, 3, 4, -1])
def test_nucleation_state_row_matches_rhs(data_path, moment_basis, column):
    unit, states = make_unit(data_path, moment_basis)
    unit.unit_model(0.0, states)
    actual = unit.jac_states(0.0, states, None, return_only=False)[0, column]
    step = FD_STEP * abs(states[column])  # [state unit]
    plus, minus = states.copy(), states.copy()  # [state units]
    plus[column] += step
    minus[column] -= step
    expected = (unit.unit_model(0.0, plus)[0]
                - unit.unit_model(0.0, minus)[0]) / (2 * step)  # [#/s/state unit]
    assert actual == pytest.approx(expected, rel=FD_RTOL, abs=0)


@pytest.mark.parametrize('moment_basis', ['volume', 'area'])
@pytest.mark.parametrize('concentration,column', [(4.0, 0), (4.0, 3), (4.0, 6),
                                                  (1.0, 10), (1.0, 11), (1.0, 12)])
def test_parameter_columns_match_rhs(data_path, moment_basis, concentration, column):
    """Compare analytical columns against solver-supplied parameter probes.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    moment_basis : str
        Area or volume basis for secondary nucleation.
    concentration : float
        Target concentration [kg/m**3].
    column : int
        Parameter position in CrystKinetics.concat_params().
    """
    unit, states = make_unit(data_path, moment_basis, concentration)
    kinetics = unit.Kinetics
    parameters = kinetics.concat_params()  # [native parameter units]
    unit.unit_model(0.0, states)
    actual = unit.jac_params(0.0, states, parameters)
    step = FD_STEP * max(abs(parameters[column]), 1.0)  # [parameter unit]
    plus, minus = parameters.copy(), parameters.copy()  # [parameter units]
    plus[column] += step
    minus[column] -= step
    upper = unit.unit_model(0.0, states, params=plus)  # [state unit/s]
    lower = unit.unit_model(0.0, states, params=minus)  # [state unit/s]
    expected = (upper - lower) / (2 * step)  # [state unit/s/parameter unit]
    np.testing.assert_allclose(actual[:, column], expected, rtol=FD_RTOL, atol=0)


@pytest.mark.parametrize('sup_sat_type', ['relative', 'absolute'])
@pytest.mark.parametrize('column', range(10))  # four moments, five species, liquid volume
def test_dissolution_state_columns_match_rhs(data_path, sup_sat_type, column):
    unit, states = make_unit(data_path, 'volume', concentration=1.0)
    unit.Kinetics.sup_sat_type = sup_sat_type
    # Equal species densities isolate the kinetic derivatives from composition
    # dependence of density, which the analytical Jacobian holds fixed. Use
    # the shipped density of A for this synthetic constant-density mixture.
    unit.Liquid_1.rho_liq = np.full_like(
        unit.Liquid_1.rho_liq, unit.Liquid_1.rho_liq[0])  # [kg/m**3]
    unit.unit_model(0.0, states)
    assert unit.Kinetics.dissol < 0
    assert unit.Kinetics.growth == 0
    actual = unit.jac_states(0.0, states, None, return_only=False)
    np.testing.assert_array_equal(actual[0], np.zeros(len(states)))
    step = FD_STEP * abs(states[column])  # [state unit], central relative perturbation
    plus, minus = states.copy(), states.copy()  # [state units]
    plus[column] += step
    minus[column] -= step
    expected = (unit.unit_model(0.0, plus)
                - unit.unit_model(0.0, minus)) / (2 * step)  # [state unit/s/state unit]
    if unit.num_distr < column < len(states) - 1:
        # At fixed density, an inert concentration appears only in its own
        # dilution equation. Assert these exact structural zeros directly;
        # differencing normalized mass fractions can produce roundoff-sized
        # spurious density changes in otherwise independent RHS entries.
        np.testing.assert_array_equal(np.delete(actual[:, column], column),
                                      np.zeros(len(states) - 1))
        assert actual[column, column] == pytest.approx(
            expected[column], rel=FD_RTOL, abs=0)
    else:
        np.testing.assert_allclose(actual[:, column], expected, rtol=FD_RTOL, atol=0)


@pytest.mark.parametrize('moment_basis', ['volume', 'area'])
def test_seedless_secondary_exponent_partial_is_finite(data_path, moment_basis):
    unit, states = make_unit(data_path, moment_basis)
    states[:unit.num_distr] = 0  # [um**n], seedless population
    unit.Kinetics.params['nucl_sec'] = [4e8, 0.0, 1.5, 0.0]
    # [#/m**3/s], [J/mol], [-], [-]; s_2=0 keeps secondary nucleation active
    # even with no seed because the built-in moment factor is 0**0 = 1.
    unit.unit_model(0.0, states)
    assert unit.Kinetics.sec_nucl > 0
    actual = unit.jac_params(0.0, states, unit.Kinetics.concat_params())
    assert np.isfinite(actual).all()
    secondary_exponent_column = 6  # fourth secondary column after three primary columns
    assert actual[0, secondary_exponent_column] == 0

