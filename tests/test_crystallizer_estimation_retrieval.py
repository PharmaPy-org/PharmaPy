"""#222 estimation, sensitivity, and retrieval parameter ownership regressions.

Real short Assimulo solves exercise the wrapper and retrieval. Core probes stop
only at solver construction when checking the wrapper's initial charge.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst, MSMPR
from PharmaPy.Kinetics import CrystKinetics
from test_crystallizer_heat_duty import make_unit as make_heat_unit
from test_crystallizer_parameter_evaluations import (
    make_unit, DISTRIBUTION, GROWTH_COLUMN, InitializationCaptured,
    LIQUID_VOLUME, RTOL, SATURATION)

FEED_FLOW = 1e-5  # [m**3/s], finite MSMPR feed with valid slurry-stream inventory
DURATION = 0.01  # [s], short integration keeps crystals inside the finite grid
TIME_GRID = np.array([0.0, DURATION])  # [s], endpoint reporting
ITERATE_GROWTH = 1.5  # [um/s], synthetic noninteger growth prefactor
HEAT_OF_CRYSTALLIZATION = -1.46e4  # [J/kg], established Crystallizers energy model


def assert_nominal_heat_source(unit, states, growth_prefactor):
    """Check source heat against the growth law and crystal-volume derivative.

    Parameters
    ----------
    unit : BatchCryst or MSMPR
        Unit with relative supersaturation kinetics and zero nucleation.
    states : ndarray
        Retrieved solver rows: total/volume-specific CSD [#/um] or
        [#/m**3/um], or moment states [um**n], then concentration [kg/m**3].
    growth_prefactor : float
        Nominal linear growth prefactor [um/s].

    Notes
    -----
    The moment solve currently seeds SI moments into the micrometre RHS
    (#222 follow-up in the next increment). Use the returned raw solver state
    here; this test checks nominal parameter ownership, not that seeding basis.
    """
    concentration = states[:, unit.num_distr + unit.target_ind]  # [kg/m**3]
    growth = growth_prefactor * (concentration / SATURATION - 1)  # [um/s]
    if unit.method == 'moments':
        second_moment = states[:, 2] * 1e-12  # [m**2], exact um**2 conversion
    else:
        grid = unit.Solid_1.x_distrib  # [um]
        integrand = states[:, :unit.num_distr] * grid**2  # [#*um] or [#*um/m**3]
        second_moment = np.sum((integrand[:, 1:] + integrand[:, :-1])
                               * np.diff(grid) / 2, axis=1) * 1e-12
        # [m**2] or [m**2/m**3], independent trapezoids and exact length conversion
    source = (HEAT_OF_CRYSTALLIZATION * unit.Solid_1.getDensity()
              * 3 * unit.Solid_1.kv * growth * 1e-6 * second_moment)  # [W] or [W/m**3]
    if isinstance(unit, MSMPR):
        source *= unit.Slurry.vol  # [W], convert volumetric source to total heat
    assert np.any(source != 0)
    np.testing.assert_allclose(unit.heat_prof[:, 0], source, rtol=RTOL, atol=0)


@pytest.mark.assimulo
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR])
@pytest.mark.parametrize('as_dict', [False, True])
def test_masked_estimation_retrieves_iterate_heat(data_path, unit_type, as_dict):
    """Run full-vector/dictionary estimation with only growth active.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    unit_type : type
        BatchCryst or MSMPR.
    as_dict : bool
        Supply the iterate as a mechanism dictionary when True.
    """
    pytest.importorskip('assimulo')
    unit, _ = make_heat_unit(data_path, unit_type, ramp=0.0, feed_flow=FEED_FLOW)
    unit.Kinetics = CrystKinetics(coeff_solub=[SATURATION],
                                 growth=(ITERATE_GROWTH, 0.0, 1.0))
    # [um/s], [J/mol], [-]; linear growth with zero activation energy
    unit.mask_params[:] = False
    unit.mask_params[GROWTH_COLUMN] = True
    nominal = unit.Kinetics.concat_params().copy()  # [native parameter units]
    parameters = ({key: list(value) for key, value in unit.Kinetics.params.items()}
                  if as_dict else nominal)
    states = unit.paramest_wrapper(parameters, TIME_GRID)
    assert_nominal_heat_source(unit, states, ITERATE_GROWTH)
    np.testing.assert_array_equal(unit.Kinetics.concat_params(), nominal)


@pytest.mark.assimulo
@pytest.mark.parametrize('unit_type', [BatchCryst, MSMPR])
def test_post_estimation_solve_uses_updated_kinetics(data_path, unit_type):
    """Ensure an old estimation iterate cannot replace a later kinetic update.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    unit_type : type
        BatchCryst or MSMPR.
    """
    pytest.importorskip('assimulo')
    unit, _ = make_heat_unit(data_path, unit_type, ramp=0.0, feed_flow=FEED_FLOW)
    unit.Kinetics = CrystKinetics(coeff_solub=[SATURATION],
                                 growth=(ITERATE_GROWTH, 0.0, 1.0))
    # [um/s], [J/mol], [-]; no primary/secondary nucleation
    iterate = unit.Kinetics.concat_params().copy()  # [native parameter units]
    unit.paramest_wrapper(iterate, TIME_GRID)
    updated = iterate.copy()  # [native parameter units]
    updated[GROWTH_COLUMN] *= 2  # [-], distinguish new kinetics from the iterate
    unit.Kinetics.set_params(updated)
    _, states = unit.solve_unit(runtime=DURATION, verbose=False)
    # heat_prof includes only the latest solve, whereas result may contain both.
    assert_nominal_heat_source(unit, states, updated[GROWTH_COLUMN])
    np.testing.assert_array_equal(unit.Kinetics.concat_params(), updated)


@pytest.mark.unit
def test_estimation_modifiers_survive_solve_reset(data_path, monkeypatch):
    """Follow wrapper -> real solve_unit -> solver boundary with reset enabled.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only optional solver construction.
    """
    unit, _ = make_unit(data_path)
    unit.reset_states = True
    modified_volume = 2 * LIQUID_VOLUME  # [m**3], four-cubic-metre charge
    modified_distribution = 2 * DISTRIBUTION  # [#/um], doubled crystal seed
    captures = []

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Record the actual ODE initial state and stop before CVode.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver configuration.
        states_init : ndarray
            CSD [#/um], concentrations [kg/m**3], liquid volume [m**3].
        params_mergd : ndarray
            Active parameters in native kinetic units.

        Raises
        ------
        InitializationCaptured
            Always, after capturing the state.
        """
        captures.append(states_init.copy())
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    with pytest.raises(InitializationCaptured):
        unit.paramest_wrapper(
            unit.Kinetics.concat_params(), TIME_GRID,
            modify_phase={'Liquid': {'vol': modified_volume},
                          'Solid': {'distrib': modified_distribution}})
    assert len(captures) == 1
    assert captures[0][-1] == pytest.approx(modified_volume, rel=RTOL, abs=0)
    np.testing.assert_allclose(captures[0][:unit.num_distr], modified_distribution,
                               rtol=RTOL, atol=0)
    assert unit.reset_states is True  # Restored even on a solver-boundary exception.


@pytest.mark.assimulo
def test_sensitivity_solve_restores_nominal_parameters_before_heat(data_path):
    """Check CVODES difference quotients cannot leak into kinetics or heat.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    pytest.importorskip('assimulo')
    unit, _ = make_unit(data_path, method='moments')
    unit.mask_params[:] = False
    unit.mask_params[GROWTH_COLUMN] = True
    nominal = unit.Kinetics.concat_params().copy()  # [native parameter units]
    _, states, _ = unit.solve_unit(time_grid=TIME_GRID, eval_sens=True, verbose=False)
    # Exact identity checks parameter ownership, not floating-point model accuracy.
    np.testing.assert_array_equal(unit.Kinetics.concat_params(), nominal)
    assert_nominal_heat_source(unit, states, nominal[GROWTH_COLUMN])
