"""#391 native-solver inventory convergence for an independently specified seeded batch.

This tests numerical resolution, not fitted process predictions. The Gaussian
seed stays far from both size boundaries, isolating FVM growth discretization
from particle escape. Moment mode provides a separate conservative reference.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.assimulo
pytest.importorskip('assimulo')

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase

TEMPERATURE = 310.0  # [K], prescribed isothermal diagnostic
VOLUME = 2.0  # [m**3], non-unit initial liquid charge
SHAPE_FACTOR = 0.5  # [-], synthetic non-spherical crystal shape assumption
SEED_AMPLITUDE = 1e10  # [#/um], finite seed giving resolvable crystallization mass
SEED_CENTER = 80.0  # [um], away from either numerical size boundary
SEED_WIDTH = 12.0  # [um], Gaussian scale rather than its standard deviation
SATURATION = 40.0  # [kg/m**3], constant synthetic solubility below the charge
GROWTH_PREFACTOR = 0.2  # [um/s], linear relative-supersaturation growth
DURATION = 50.0  # [s], growth remains far below the 500 um upper boundary
POPULATION_SCALE = 1e-9  # [-], balances numerical CSD/state magnitudes only
SOLVER_RTOL = 1e-9  # [-], tighter than the conservation acceptance budgets
SOLVER_ATOL = 1e-10  # [scaled state units], native integrator absolute budget
FINE_GRID_BUDGET = 1e-3  # [-], one part per thousand of crystallized mass
MOMENT_BUDGET = 1e-6  # [-], integration allowance with no size discretization


def solve_inventory_case(data_path, nodes: int, method: str) -> tuple:
    """Return independent target and inert inventories from a real batch solve.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    nodes : int
        Number of logarithmic size nodes [-] spanning 1 to 500 um.
    method : str
        '1D-FVM' or 'moments'; kinetics and physical conditions are unchanged.

    Returns
    -------
    liquid_inventory : ndarray
        Species mass [kg], shape (21 reporting times, 5 species).
    crystal_inventory : ndarray
        Pure-A solid mass [kg], shape (21,).

    Notes
    -----
    The liquid has no named solvent, so every supplied concentration is an
    integrated species. Named-solvent phase/ODE closure is separately tracked
    by #312; continuous phase-volume closure is tracked by #300.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    grid = np.geomspace(1.0, 500.0, nodes)  # [um], broad interior-seed domain
    population = SEED_AMPLITUDE * np.exp(-((grid-SEED_CENTER)/SEED_WIDTH)**2)  # [#/um]
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=VOLUME,
                         mass_frac=[0.1,0.1,0.1,0.1,0.6])  # [-], unequal solute/solvent fractions
    solid = SolidPhase(path, temp=TEMPERATURE, x_distrib=grid,
                       distrib=population, kv=SHAPE_FACTOR,
                       mass_frac=[1,0,0,0,0])  # [-], pure-A crystal
    scale = POPULATION_SCALE if method == '1D-FVM' else 1.0  # [-], moment mode's supported scale
    unit = BatchCryst('A', method=method, scale=scale, controls={
        'temp': lambda time: TEMPERATURE + np.zeros_like(time)})
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[SATURATION],
                                  growth=(GROWTH_PREFACTOR,0.0,1.0))
    # [um/s], [J/mol], [-]; zero activation, linear relative driving force
    times = np.linspace(0.0,DURATION,21)  # [s], endpoint-inclusive reporting grid
    unit.solve_unit(time_grid=times, verbose=False,
                    sundials_opts={'rtol':SOLVER_RTOL,'atol':SOLVER_ATOL})
    liquid_inventory = unit.result.mass_conc * unit.result.vol[:,None]  # [kg]
    crystal_inventory = unit.result.mu_n[:,3] * SHAPE_FACTOR * solid.getDensity()  # [kg]
    return liquid_inventory, crystal_inventory


def test_batch_fvm_crystal_mass_error_decreases_under_grid_refinement(data_path):
    """Resolve finite-grid growth error with an independent component balance.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    """
    errors = []  # [-], relative to actual solid growth, not the large liquid charge
    for nodes in (35,70,140,280):  # [-], successive approximate size-grid doublings
        liquid, crystal = solve_inventory_case(data_path,nodes,'1D-FVM')
        growth = crystal[-1]-crystal[0]  # [kg]
        assert growth > 0
        residual = liquid[-1,0]-liquid[0,0]+growth  # [kg], closed-batch target balance
        errors.append(abs(residual/growth))
        np.testing.assert_allclose(liquid[:,1:],np.broadcast_to(liquid[0,1:],liquid[:,1:].shape),
                                   rtol=MOMENT_BUDGET,atol=0)
    assert np.all(np.diff(errors)<0), errors
    assert errors[-1] < FINE_GRID_BUDGET, errors


def test_batch_moment_population_closes_species_inventories(data_path):
    """Check the same model without finite-grid growth transport error.

    Parameters
    ----------
    data_path : dict
        Public thermodynamic database paths.
    """
    liquid, crystal = solve_inventory_case(data_path,280,'moments')
    transferred = crystal[-1]-crystal[0]  # [kg]
    assert transferred > 0
    residual = liquid[:,0]+crystal-liquid[0,0]-crystal[0]  # [kg]
    np.testing.assert_allclose(residual,0,rtol=0,atol=MOMENT_BUDGET*transferred)
    np.testing.assert_allclose(liquid[:,1:],np.broadcast_to(liquid[0,1:],liquid[:,1:].shape),
                               rtol=MOMENT_BUDGET,atol=0)
