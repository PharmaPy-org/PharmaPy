"""Raw materials of a user-charged Filter (issue #456).

``Filter.Phases`` keeps the slurry as the ``__original_phase__``
initial-holdup snapshot, so ``SimulationExec.GetRawMaterials`` reports a
slurry charged by the user once, one ``Initial_holdup`` row per phase,
whether it was assigned as a phase list or as a ``Slurry``. A slurry
transferred from an upstream ``BatchCryst`` is counted under the
crystallizer's own charge only. Earlier releases kept no snapshot and
silently omitted the user charge.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json); a 1 L liquid of mass
fractions (0.1, 0.1, 0.1, 0.1, 0.6) at 300 K and crystals of A on a 100,
200, 300 um grid with populations (1, 2, 1)e4 #/um and kv = 1; a
synthetic specific cake resistance of 1e11 m/kg filtered at 1 bar.
Expected amounts are derived from the charge specification: the liquid
mass is its volume times the ideal-mixing density 1/sum(w_i/rho_i) of the
database pure liquids, the crystal mass is rho_solid(A) kv mu_3 with mu_3
the trapezoidal third moment of the population (PharmaPy's quadrature,
tests/test_distribution_quadrature_contract.py) converted from um**3 to
m**3, and molar amounts divide species masses by the database molar
masses. The upstream crystallizer has a constant solubility of A
(``CrystKinetics(coeff_solub=[2000.])``) and cooling water, as in
tests/test_simexec_seeded_holdup_raw_materials.py, so no mechanism
changes its crystals. The units are integrated with CVode, so every case
is marked ``assimulo``.
"""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Utilities import CoolingWater

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], mass fractions
SOLID = np.array([1., 0., 0., 0., 0.])  # [-], crystals of species A
LIQUID_VOLUME = 1e-3  # [m**3]
TEMP = 300.0  # [K]
GRID = np.array([100., 200., 300.])  # [um], three size classes
POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um]
# [-], volumetric shape factor (crystal volume kv * x**3); a synthetic
# cube-like value, not a calibrated crystal shape.
SHAPE_FACTOR = 1.0
STATION_DIAMETER = 0.1  # [m], synthetic bench filter
# [m/kg], synthetic specific cake resistance, so no correlation applies
CAKE_RESISTANCE = 1e11
PRESSURE_DROP = 1e5  # [Pa], filtration driving force
UM3_TO_M3 = 1e-18  # [m**3/um**3], exact SI conversion
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], float64 roundoff of the reported conversions: the holdups are read
# from the stored charge, not from the integration. Absent species are
# compared in absolute terms against this bound times the phase amount.
RTOL = 1e-12
SPECIES = ('A', 'B', 'C', 'D', 'solvent')  # database order
RUN = {'deltaP': PRESSURE_DROP, 'verbose': False}
SOLUBILITY = [2000.]  # [kg/m**3], constant solubility of A; no mechanism
UTILITY_FLOW = 1e-5  # [m**3/s], cooling water
UTILITY_TEMP = 290.0  # [K], cooling water inlet
CRYST_RUNTIME = 1.0  # [s], short crystallizer run


@pytest.fixture
def path(data_path):
    """Return the shipped five-species property database.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    str
        Database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def _phases(path):
    """Build the liquid and crystals of the charge.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    list
        ``LiquidPhase`` of LIQUID_VOLUME [m**3] and ``SolidPhase`` of
        POPULATION [#/um] on GRID [um], both at TEMP [K].
    """
    return [LiquidPhase(path, temp=TEMP, vol=LIQUID_VOLUME,
                        mass_frac=COMPOSITION),
            SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                       distrib=POPULATION.copy(), kv=SHAPE_FACTOR,
                       mass_frac=SOLID)]


def _expected(path):
    """Return the species amounts of both phases on both bases.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    list of dict
        For the liquid, then the crystals: ``{'mass': kg, 'mole': mol}``
        species amounts in database order.
    """
    with open(path) as handle:
        data = json.load(handle)
    rho_liq, rho_solid, weights = (
        np.array([value[key] for value in data.values()])
        for key in ('rho_liq', 'rho_solid', 'mw'))  # [kg/m**3], [g/mol]
    liquid_mass = (LIQUID_VOLUME / np.sum(COMPOSITION / rho_liq)
                   * COMPOSITION)  # [kg]
    integrand = POPULATION * GRID**3  # [#/um * um**3]
    mu_3 = np.sum(np.diff(GRID) * (integrand[1:] + integrand[:-1]) / 2)  # [um**3]
    solid_mass = (rho_solid[0] * SHAPE_FACTOR * mu_3 * UM3_TO_M3
                  * SOLID)  # [kg]
    return [{'mass': mass, 'mole': mass / (weights * G_TO_KG)}
            for mass in (liquid_mass, solid_mass)]  # [kg], [mol]


def _assert_charge(sim, path, unit, sources):
    """Assert one raw row per charge phase, on both bases.

    Parameters
    ----------
    sim : SimulationExec
        Solved flowsheet.
    path : str
        Database path.
    unit : str
        Unit expected to report the charge.
    sources : list of str
        Raw sources of the liquid and crystal rows, in order.
    """
    expected = _expected(path)
    for basis, amount in (('mass', 'mass'), ('mole', 'moles')):
        raw = sim.GetRawMaterials(basis=basis)
        assert [key[:2] for key in raw.index] == [
            (unit, source) for source in sources]
        assert [key[2].split('.')[0] for key in raw.index] == [
            'LiquidPhase', 'SolidPhase']
        for row, species in zip(range(len(raw)), expected):
            total = species[basis].sum()  # [kg] or [mol]
            assert raw[amount].iloc[row] == pytest.approx(total, rel=RTOL)
            reported = raw[['%s_%s' % (amount, name)
                            for name in SPECIES]].iloc[row]
            np.testing.assert_allclose(reported.to_numpy(dtype=float),
                                       species[basis], rtol=RTOL,
                                       atol=RTOL * total)


def _filter():
    """Build the synthetic bench filter.

    Returns
    -------
    Filter
        Filter of STATION_DIAMETER [m] and CAKE_RESISTANCE [m/kg].
    """
    return Filter(station_diam=STATION_DIAMETER, alpha=CAKE_RESISTANCE)


@pytest.mark.parametrize('form', ['list', 'slurry'])
def test_user_slurry_is_a_raw_material(path, form):
    """Report each phase of the user's slurry once as initial holdup."""
    phases = _phases(path)
    if form == 'slurry':
        charge = Slurry()
        charge.Phases = phases
    else:
        charge = phases
    sim = SimulationExec(path, {'F01': []})
    sim.F01 = _filter()
    sim.F01.Phases = charge
    sim.SolveFlowsheet(kwargs_run={'F01': RUN}, verbose=False)

    _assert_charge(sim, path, 'F01', ['Initial_holdup', 'Initial_holdup'])


def test_transferred_slurry_is_counted_under_its_source(path):
    """Count a crystallizer's slurry under the crystallizer's charge only."""
    sim = SimulationExec(path, {'CR01': ['F01'], 'F01': []})
    sim.CR01 = BatchCryst('A', method='1D-FVM')
    sim.CR01.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
    sim.CR01.Utility = CoolingWater(vol_flow=UTILITY_FLOW,
                                    temp_in=UTILITY_TEMP)
    sim.CR01.Phases = _phases(path)
    sim.F01 = _filter()
    sim.SolveFlowsheet(kwargs_run={'CR01': {'runtime': CRYST_RUNTIME,
                                            'verbose': False},
                                   'F01': RUN}, verbose=False)

    snapshot = sim.F01.__original_phase__
    assert all(phase.transferred_from_uo for phase in snapshot.Phases)
    _assert_charge(sim, path, 'CR01', ['Initial_holdup', 'Initial_holdup'])
