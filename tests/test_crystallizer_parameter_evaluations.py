"""#222 parameter handoff and repeatable initialization with real phases.

Core initialization probes stop at the optional solver boundary. The Assimulo
regression also exercises two complete FVM solves with reset enabled.
"""

import numpy as np
import pytest

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase

TEMPERATURE = 310.0  # [K], synthetic isothermal vessel
LIQUID_VOLUME = 2.0  # [m**3], non-unit charge exposes total/volume bases
KV = 0.5  # [-], non-unit crystal shape factor
GRID = np.array([10.0, 20.0, 40.0])  # [um], unequal size intervals
DISTRIBUTION = np.array([1e7, 2e7, 1e7])  # [#/um], asymmetric seed population
GROWTH = 1.0  # [um/s], synthetic growth at unit relative supersaturation
SATURATION = 2.0  # [kg/m**3], constant synthetic solubility
RTOL = 1e-12  # [-], direct algebra roundoff allowance
GROWTH_COLUMN = 7  # three primary and four secondary parameters precede growth


def make_unit(data_path, method='1D-FVM', scale=1.0, radius=0.0):
    """Build a real batch unit at unit relative supersaturation.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    method : str, optional
        Crystal population discretization.
    scale : float, optional
        Numerical CSD multiplier [-].
    radius : float, optional
        Nucleation size [um], matching the size grid.

    Returns
    -------
    unit : BatchCryst
        Isothermal seeded unit with zero nucleation and finite growth.
    states : ndarray
        Total moments [um**n] or scaled CSD [#/um], species concentrations
        [kg/m**3], and liquid volume [m**3], in model state order.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    unit = BatchCryst('A', method=method, scale=scale, rad_zero=radius,
                      controls={'temp': lambda time: TEMPERATURE + np.zeros_like(time)})
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=LIQUID_VOLUME,
                         mass_frac=[0.1, 0.1, 0.1, 0.1, 0.6])  # [-]
    solid = SolidPhase(path, temp=TEMPERATURE, x_distrib=GRID,
                       distrib=DISTRIBUTION, kv=KV,
                       mass_frac=[1, 0, 0, 0, 0])  # [-], pure A
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(
        coeff_solub=[SATURATION], growth=(GROWTH, 0.0, 1.0))
    # Growth parameters: [um/s], [J/mol], [-]; zero activation and linear force.
    unit.Kinetics.target_idx = unit.target_ind
    unit.num_species = len(liquid.mass_conc)
    unit.x_grid = GRID  # [um]
    unit.dx = unit.Slurry.dx  # [um]
    concentrations = liquid.mass_conc.copy()  # [kg/m**3]
    concentrations[unit.target_ind] = 2 * SATURATION  # [kg/m**3], relative force=1
    population = (solid.moments * 1e6**np.arange(unit.num_distr)
                  if method == 'moments' else solid.distrib * scale)
    # [um**n] total moments, or scaled [#/um]; 1e6 is exact um/m conversion.
    return unit, np.concatenate((population, concentrations, [LIQUID_VOLUME]))


@pytest.mark.unit
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
@pytest.mark.parametrize('masked', [False, True])
def test_solver_parameters_double_growth_mass_source(data_path, method, masked):
    """Check A/B/A RHS evaluations and active/fixed parameter reconstruction.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    method : str
        Population discretization.
    masked : bool
        Select only the growth prefactor when True.
    """
    unit, states = make_unit(data_path, method)
    if masked:
        unit.mask_params[:] = False
        unit.mask_params[GROWTH_COLUMN] = True
    original = unit.Kinetics.concat_params()  # [native kinetic parameter units]
    parameters = original[unit.mask_params]  # [active kinetic parameter units]
    doubled = parameters.copy()  # [active kinetic parameter units]
    doubled[0 if masked else GROWTH_COLUMN] *= 2  # [-], exact linear prefactor probe
    first = unit.unit_model(0.0, states, params=parameters)  # [state unit/s]
    second = unit.unit_model(0.0, states, params=doubled)  # [state unit/s]
    repeated = unit.unit_model(0.0, states, params=parameters)  # [state unit/s]
    # Independently integrate the second moment on this three-point grid:
    # trapezoids of x**2*f give 2.85e11 um**2 = 0.285 m**2.
    second_moment = 0.285  # [m**2], total crystal second moment
    expected = (3 * KV * unit.Solid_1.getDensity()
                * GROWTH * 1e-6 * second_moment)  # [kg/s], d(kv*L**3)/dt
    density = unit.Liquid_1.getDensity()  # [kg/m**3]
    assert -first[-1] * density == pytest.approx(expected, rel=RTOL, abs=0)
    assert -second[-1] * density == pytest.approx(2 * expected, rel=RTOL, abs=0)
    np.testing.assert_allclose(repeated, first, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(unit.Kinetics.concat_params()[~unit.mask_params],
                                  original[~unit.mask_params])


class InitializationCaptured(Exception):
    """Stop after real initialization and before optional solver construction."""


@pytest.mark.unit
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
def test_integer_kinetics_accept_fractional_active_parameters(data_path, method):
    """Preserve fractional probes when all stored kinetic entries are integers.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    method : str
        Population discretization.
    """
    unit, states = make_unit(data_path, method)
    unit.Kinetics = CrystKinetics(coeff_solub=[SATURATION], growth=(1, 0, 1))
    # [um/s], [J/mol], [-]; integer-valued linear growth at zero activation
    unit.Kinetics.target_idx = unit.target_ind
    assert unit.Kinetics.concat_params().dtype.kind == 'i'
    unit.mask_params[:] = False
    unit.mask_params[GROWTH_COLUMN] = True
    first = unit.unit_model(0.0, states, params=np.array([1.0]))  # [state unit/s]
    fractional = unit.unit_model(0.0, states, params=np.array([1.5]))  # [state unit/s]
    # Growth is linear in its prefactor, so a 50% increase scales every RHS term.
    assert first[-1] != 0
    np.testing.assert_allclose(fractional, 1.5 * first, rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('method', ['moments', '1D-FVM'])
@pytest.mark.parametrize('parameters', [[], [1.0, 2.0]])  # [active kinetic units]
def test_parameter_count_must_match_mask(data_path, method, parameters):
    """Reject missing or extra active parameters before NumPy assignment.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    method : str
        Population discretization.
    parameters : list
        Invalid active parameter vector in native kinetic units.
    """
    unit, states = make_unit(data_path, method)
    unit.mask_params[:] = False
    unit.mask_params[GROWTH_COLUMN] = True
    with pytest.raises(ValueError, match='mask_params.*1 active parameter'):
        unit.unit_model(0.0, states, params=parameters)


@pytest.mark.unit
def test_fvm_output_option_is_validated(data_path):
    """Reject an unknown finite-choice option at the FVM public boundary.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    unit, states = make_unit(data_path)
    with pytest.raises(ValueError, match='output.*flux.*dstates'):
        unit.fvm_method(states[:unit.num_distr], unit.Solid_1.moments,
                        states[unit.num_distr:-1], TEMPERATURE, None,
                        unit.Solid_1.getDensity(), output='invalid')


@pytest.mark.unit
@pytest.mark.parametrize('explicit_volume', [False, True])
def test_repeated_initialization_keeps_geometry(data_path, monkeypatch, explicit_volume):
    """Check inferred and explicitly assigned vessel geometry without CVode.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only the optional solver-construction boundary.
    explicit_volume : bool
        Assign the vessel volume before initialization when True.
    """
    unit, _ = make_unit(data_path)
    if explicit_volume:
        unit.vol_tank = LIQUID_VOLUME  # [m**3], explicit working vessel volume
    captures = []

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Record geometry and stop at solver construction.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver callback options.
        states_init : ndarray
            Initial states in model units.
        params_mergd : ndarray
            Active kinetic parameters in native units.

        Raises
        ------
        InitializationCaptured
            Always, after recording volume [m**3], diameter [m], area [m**2].
        """
        captures.append([unit.vol_tank, unit.diam_tank, unit.area_base])
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    for _ in range(2):
        with pytest.raises(InitializationCaptured):
            unit.solve_unit(runtime=1.0, verbose=False)  # [s], initialization only
    expected_volume = LIQUID_VOLUME if explicit_volume else unit.Slurry.vol  # [m**3]
    assert captures[0][0] == pytest.approx(expected_volume, rel=RTOL, abs=0)
    np.testing.assert_allclose(captures[1], captures[0], rtol=RTOL, atol=0)


@pytest.mark.unit
def test_reset_precedes_initial_state_capture(data_path, monkeypatch):
    """Restore changed liquid and solid inventories before building the ODE.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    monkeypatch : pytest.MonkeyPatch
        Replace only the optional solver-construction boundary.
    """
    unit, _ = make_unit(data_path)
    expected = np.concatenate((unit.Solid_1.distrib,
                               unit.Liquid_1.mass_conc, [unit.Liquid_1.vol]))
    # [#/um], [kg/m**3], [m**3], original charged state
    unit.reset_states = True
    unit.Liquid_1.updatePhase(vol=2 * LIQUID_VOLUME,
                              mass_frac=[0.2, 0.1, 0.1, 0.1, 0.5])  # [-]
    unit.Solid_1.updatePhase(distrib=2 * DISTRIBUTION)  # [#/um], doubled seed

    def capture(eval_sens, states_init, params_mergd, jacv_prod):
        """Assert the original charge at the solver boundary.

        Parameters
        ----------
        eval_sens, jacv_prod : bool
            Solver callback options.
        states_init : ndarray
            Initial CSD [#/um], concentrations [kg/m**3], liquid volume [m**3].
        params_mergd : ndarray
            Active kinetic parameters in native units.

        Raises
        ------
        InitializationCaptured
            After checking the initial state.
        """
        np.testing.assert_allclose(states_init, expected, rtol=RTOL, atol=0)
        raise InitializationCaptured

    monkeypatch.setattr(unit, 'set_ode_problem', capture)
    with pytest.raises(InitializationCaptured):
        unit.solve_unit(runtime=1.0, verbose=False)  # [s], initialization only


@pytest.mark.assimulo
def test_two_reset_solves_repeat_initial_charge_and_geometry(data_path):
    """Exercise reset and geometry through two real CVode integrations.

    Parameters
    ----------
    data_path : dict
        Repository database paths.
    """
    pytest.importorskip('assimulo')
    unit, _ = make_unit(data_path)
    unit.reset_states = True
    duration = 0.01  # [s], short growth interval retains the finite grid population
    grid = np.array([0.0, duration])  # [s], endpoint reporting
    _, first = unit.solve_unit(time_grid=grid, verbose=False)
    geometry = np.array([unit.vol_tank, unit.diam_tank, unit.area_base])
    # [m**3], [m], [m**2], first initialization
    _, second = unit.solve_unit(time_grid=grid, verbose=False)
    np.testing.assert_allclose(second[0], first[0], rtol=RTOL, atol=0)
    np.testing.assert_allclose([unit.vol_tank, unit.diam_tank, unit.area_base],
                               geometry, rtol=RTOL, atol=0)
