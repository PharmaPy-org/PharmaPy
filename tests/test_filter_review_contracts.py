"""Filter input-domain and crystallizer population handoff regressions.

Public boundary checks use real liquid/solid collaborators. Native growth
and filtration runs are confined to tests marked for the Assimulo lane.
"""

import numpy as np
import pytest

from PharmaPy.SolidLiquidSep import Filter
from test_separation_balance_fixes import separation_phases


@pytest.mark.unit
@pytest.mark.parametrize('pressure', [0., -1., np.nan, np.inf])  # [Pa], invalid driving forces
def test_filter_rejects_invalid_pressure_before_solver(separation_phases, pressure):
    unit = Filter(station_diam=0.1, alpha=1e10)  # [m], [m/kg], bench fixture
    unit.Phases = separation_phases
    with pytest.raises(ValueError, match='deltaP.*finite.*positive'):
        unit.solve_unit(time_grid=[0., 1.], deltaP=pressure,
                        model_params=unit.param_seed, verbose=False)
    assert not hasattr(unit, 'result')


@pytest.mark.unit
@pytest.mark.parametrize('parameters, callable_parameters', [
    (parameters, callable_parameters)
    for parameters in ([0., 1e9], [-1e10, 1e9], [1e10, -1e9], [np.nan, 1e9], [1e10, np.inf])
    for callable_parameters in (False, True)
] + [([1e10], False)])  # [m/kg, 1/m], invalid domain or vector length
def test_filter_rejects_invalid_resolved_resistances(separation_phases, parameters, callable_parameters):
    unit = Filter(station_diam=0.1, alpha=1e10)
    unit.Phases = separation_phases
    overrides = parameters
    if callable_parameters:
        unit.alpha = lambda pressure: parameters[0]
        unit.r_medium = lambda pressure: parameters[1]
        overrides = None
    with pytest.raises(ValueError, match='resistances.*alpha.*medium'):
        unit.solve_unit(time_grid=[0., 1.], model_params=overrides, verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize('logarithms', [[1000., 1.], [-1000., 1.], [1., np.nan]])
def test_filter_rejects_nonphysical_logarithmic_trials(separation_phases, logarithms):
    # [-], exp(+/-1000) overflows/underflows float64; neither defines alpha>0.
    unit = Filter(station_diam=0.1, alpha=1e10, log_params=True)
    unit.Phases = separation_phases
    with pytest.raises(ValueError, match='resistances.*alpha.*medium'):
        unit.solve_unit(time_grid=[0., 1.], model_params=logarithms, verbose=False)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_grown_crystallizer_population_filter_handoff(data_path, method):
    pytest.importorskip('assimulo')
    from test_crystallizer_parameter_evaluations import make_unit
    unit, _ = make_unit(data_path, method)
    initial_mass = unit.Solid_1.mass  # [kg], initial seed
    unit.solve_unit(runtime=0.01, verbose=False,
                    sundials_opts={'rtol': 1e-9, 'atol': 1e-11})  # [s], short resolved growth
    assert unit.Solid_1.mass > initial_mass
    assert unit.Solid_1.mass == pytest.approx(
        unit.Solid_1.kv * unit.Solid_1.moments[3] * unit.Solid_1.getDensity(), rel=1e-12)
    filter_unit = Filter(station_diam=0.1, alpha=1e10)  # [m], [m/kg]
    if method == 'moments':
        with pytest.raises(ValueError, match='Filter.*distribution.*moments.*mass'):
            filter_unit.Phases = unit.Outlet
    else:
        filter_unit.Phases = unit.Outlet
        filter_unit.solve_unit(runtime=0.01, verbose=False,
                               sundials_opts={'rtol': 1e-9, 'atol': 1e-11})  # [s]
        cake = filter_unit.Outlet
        solid_volume = cake.Solid_1.mass / cake.Solid_1.getDensity()  # [m**3]
        assert solid_volume > 0
        assert cake.cake_vol > solid_volume
        assert solid_volume == pytest.approx(cake.cake_vol * (1-cake.porosity), rel=1e-10)
