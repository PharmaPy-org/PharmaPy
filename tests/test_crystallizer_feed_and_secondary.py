"""bare-liquid feed and FVM secondary-nucleation basis regressions.

Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/224
https://github.com/PharmaPy-org/PharmaPy/issues/227
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Streams import LiquidStream
from PharmaPy.Kinetics import CrystKinetics
from test_crystallizer_heat_duty import make_unit as make_heat_unit
from test_crystallizer_moment_inventory import inventory_unit, DURATION, FLOW, SOLVER_RTOL
from test_crystallizer_parameter_evaluations import make_unit, RTOL, KV, GROWTH, SATURATION


@pytest.mark.assimulo
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_bare_liquid_msmpr_solve(data_path, method):
    """Retrieve a solid-free liquid feed and reproduce analytical washout.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    method : str
        Population representation: moments or 1D-FVM.
    """
    pytest.importorskip('assimulo')
    if method == 'moments':
        unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
    else:
        unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=FLOW)
    liquid = unit.Liquid_1
    unit.Inlet = LiquidStream(liquid.path_data, temp=liquid.temp,
                              mass_frac=liquid.mass_frac, vol_flow=FLOW)
    initial_concentrations = liquid.mass_conc.copy()  # [kg/m**3]
    initial_moments = unit.Slurry.moments.copy()  # [m**n/m**3]
    volume = unit.Slurry.vol  # [m**3], constant MSMPR volume
    integration_rtol = 1e-9  # [-], two orders tighter than the washout assertion
    integration_atol = 1e-10  # [state units], below the smallest population state
    unit.solve_unit(time_grid=[0.0, DURATION], verbose=False,
                    sundials_opts={'rtol': integration_rtol, 'atol': integration_atol})
    assert np.isfinite(unit.result.mass_conc).all()
    np.testing.assert_allclose(unit.result.mass_conc[0], initial_concentrations,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(unit.result.vol_flow, FLOW, rtol=RTOL, atol=0)
    assert unit.Outlet.vol_flow == pytest.approx(FLOW, rel=RTOL, abs=0)
    assert unit.Outlet.Solid_1.kv == unit.Solid_1.kv
    # With frozen kinetics and no inlet crystals, every moment washes out
    # according to dmu/dt = -Q/V*mu, independently of population discretization.
    expected = initial_moments * np.exp(-FLOW / volume * DURATION)  # [m**n/m**3]
    np.testing.assert_allclose(unit.result.mu_n[-1], expected,
                               rtol=SOLVER_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('moment_basis', ['area', 'volume'])
def test_fvm_secondary_nucleation_matches_moments(data_path, moment_basis):
    """Compare volume-normalized secondary rates at identical physical states.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    moment_basis : str
        Public secondary-nucleation basis: area or volume.
    """
    order = {'area': 2, 'volume': 3}[moment_basis]
    rates = []  # [#/m**3/s], cached secondary rates from the public RHS
    prefactor = 8e8  # [#/m**3/s/(m**order/m**3)**2], synthetic quadratic rate
    for method in ('1D-FVM', 'moments'):
        unit, states = make_unit(data_path, method)
        unit.Kinetics = CrystKinetics(
            coeff_solub=[SATURATION], growth=(GROWTH, 0.0, 1.0),
            nucl_sec=(prefactor, 0.0, 1.0, 2.0), mu_sec_nucl=moment_basis)
        # [kg/m**3] solubility; [um/s], [J/mol], [-] growth parameters.
        # Secondary: native prefactor units above, [J/mol], [-], [-];
        # zero activation, linear force, and quadratic inventory.
        unit.Kinetics.target_idx = unit.target_ind
        unit.unit_model(0.0, states)
        rates.append(unit.Kinetics.sec_nucl)
    # Independent trapezoids of the fixture x**n*n(x), with SI conversion.
    moments = {2: 0.285, 3: 8.85e-6}  # [m**2], [m**3], total moments
    slurry_volume = 2.0 + KV * moments[3]  # [m**3], liquid plus crystals
    expected = prefactor * (KV * moments[order] / slurry_volume)**2  # [#/m**3/s]
    np.testing.assert_allclose(rates, expected, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_steady_state_rejects_bare_liquid_with_dynamic_alternative(data_path):
    """Explain the steady-state inlet restriction before accessing subphases.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
    liquid = unit.Liquid_1
    unit.Inlet = LiquidStream(liquid.path_data, temp=liquid.temp,
                              mass_frac=liquid.mass_frac, vol_flow=FLOW)
    with pytest.raises(ValueError, match=r'solve_steady_state.*LiquidStream.*solve_unit'):
        unit.solve_steady_state(liquid.mass_conc[0], liquid.temp)
