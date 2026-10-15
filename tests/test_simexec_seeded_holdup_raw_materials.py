"""Raw materials of crystallizers seeded by the user (issue #455).

A crystallizer charged by the user with seed crystals and completed by a
liquid holdup transferred from an upstream batch ``Mixer`` keeps a
``Slurry`` snapshot of its initial holdup that is marked as transferred,
because one of its phases was. ``SimulationExec.GetRawMaterials`` must
account that holdup per phase: the user's seed is a raw material of the
crystallizer, and the liquid is charged once, to the Mixer that received
it. A holdup transferred as a whole, here the slurry of a seeded
crystallizer charging a second one, is charged to its sources only, and a
holdup charged wholly by the user is reported in full.

Fixture (the triage reproduction of the issue): the shipped five-species
database (tests/Flowsheet/data/compound_database.json), a 0.5 L liquid of
mass fractions (0.1, 0.1, 0.1, 0.1, 0.6) at 310 K, and seed crystals of A
on a 10, 20, 40 um grid with populations (1, 2, 1)e7 #/um and kv = 1.
``CrystKinetics(coeff_solub=[2000.])`` declares a solubility only, and
crystallizers run for 100 s with cooling water. Expected amounts are
derived from the specifications: the liquid mass is its volume times the
ideal-mixing density 1/sum(w_i/rho_i) of the database pure liquids; the
seed mass is rho_solid(A) kv mu_3, with mu_3 the trapezoidal third moment
of the population (PharmaPy's quadrature, checked by
tests/test_distribution_quadrature_contract.py) converted from um**3 to
m**3; molar amounts divide species masses by the database molar masses.
The crystallizers are integrated with CVode, so every case is marked
``assimulo``.
"""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Containers import Mixer
from PharmaPy.Crystallizers import BatchCryst, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], mass fractions
TEMP = 310.0  # [K], issue #455 charge temperature
LIQUID_VOLUME = 5e-4  # [m**3], issue #455 liquid charge
GRID = np.array([10., 20., 40.])  # [um], three size classes
SEED = np.array([1e7, 2e7, 1e7])  # [#/um], asymmetric crystal population
SOLID = np.array([1., 0., 0., 0., 0.])  # [-], crystals of species A
# [-], volumetric shape factor (crystal volume kv * x**3) of the triage
# seed; a synthetic cube-like value, not a calibrated crystal shape.
SHAPE_FACTOR = 1.0
SOLUBILITY = [2000.]  # [kg/m**3], constant solubility of A; no mechanism
UTILITY_FLOW = 1e-5  # [m**3/s], cooling water
UTILITY_TEMP = 300.0  # [K], cooling water inlet
TANK = 2e-3  # [m**3], semibatch vessel
FEED = 1e-5  # [m**3/s], own liquid feed of the semibatch crystallizer
RUNTIME = 100.0  # [s], issue #455 triage run
UM3_TO_M3 = 1e-18  # [m**3/um**3], exact SI conversion
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], float64 roundoff of the reported conversions: the initial holdups
# and the raw inlets are read from the stored charges and feeds, not from
# the integration, so no solver tolerance enters the expectations. Stored
# zero solid fractions are machine epsilon, so absent species are compared
# in absolute terms against this bound times the phase amount.
RTOL = 1e-12


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


def _database(path):
    """Read the pure-component data used by the expectations.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    rho_liq : numpy.ndarray
        Pure liquid densities [kg/m**3], database order.
    rho_solid : numpy.ndarray
        Pure solid densities [kg/m**3], database order.
    mw : numpy.ndarray
        Molar masses [g/mol], database order.
    """
    with open(path) as handle:
        data = json.load(handle)
    return tuple(np.array([value[key] for value in data.values()])
                 for key in ('rho_liq', 'rho_solid', 'mw'))


def _expected_species(path, which, volume=LIQUID_VOLUME):
    """Return the expected species amounts of a liquid or the seed.

    Parameters
    ----------
    path : str
        Database path.
    which : {'liquid', 'seed'}
        Liquid of ``volume`` [m**3] at COMPOSITION, or the seed crystals.
    volume : float, optional
        Liquid volume [m**3].

    Returns
    -------
    dict
        ``{'mass': kg, 'mole': mol}`` species amounts, database order.
    """
    rho_liq, rho_solid, mw = _database(path)
    if which == 'liquid':
        density = 1 / np.sum(COMPOSITION / rho_liq)  # [kg/m**3]
        species_mass = COMPOSITION * density * volume  # [kg]
    else:
        integrand = SEED * GRID**3  # [#/um * um**3]
        mu_3 = np.sum(np.diff(GRID) * (integrand[1:] + integrand[:-1]) / 2)  # [um**3]
        mass = rho_solid[0] * SHAPE_FACTOR * mu_3 * UM3_TO_M3  # [kg], pure A
        species_mass = SOLID * mass  # [kg]
    return {'mass': species_mass,
            'mole': species_mass / (mw * G_TO_KG)}  # [kg], [mol]


def _crystallizer(cls, **kwargs):
    """Attach the triage kinetics and cooling water to a crystallizer.

    Parameters
    ----------
    cls : type
        Crystallizer class.
    **kwargs : dict
        Constructor settings, such as ``vol_tank`` [m**3].

    Returns
    -------
    object
        Crystallizer of species A on the ``1D-FVM`` grid, without phases.
    """
    unit = cls('A', method='1D-FVM', **kwargs)
    unit.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
    unit.Utility = CoolingWater(vol_flow=UTILITY_FLOW, temp_in=UTILITY_TEMP)
    return unit


def _liquid(path):
    """Build the triage liquid charge.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    LiquidPhase
        LIQUID_VOLUME [m**3] at TEMP [K].
    """
    return LiquidPhase(path, temp=TEMP, vol=LIQUID_VOLUME,
                       mass_frac=COMPOSITION)


def _seed(path):
    """Build the triage seed crystals.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    SolidPhase
        Crystals of A on GRID [um] with population SEED [#/um].
    """
    return SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                      distrib=SEED.copy(), kv=SHAPE_FACTOR, mass_frac=SOLID)


def _seeded_flowsheet(path, graph, cls, **kwargs):
    """Build a batch Mixer whose liquid completes a seeded crystallizer.

    Parameters
    ----------
    path : str
        Database path.
    graph : dict
        Flowsheet graph with ``'M01'`` feeding ``'CR01'``.
    cls : type
        Crystallizer class of ``CR01``.
    **kwargs : dict
        Constructor settings of ``CR01``.

    Returns
    -------
    SimulationExec
        Unsolved flowsheet; ``CR01`` holds only the seed.
    """
    sim = SimulationExec(path, graph)
    sim.M01 = Mixer()
    sim.M01.Inlets = _liquid(path)
    sim.CR01 = _crystallizer(cls, **kwargs)
    sim.CR01.Phases = _seed(path)
    return sim


def _assert_rows(raw, path, expected):
    """Assert raw-material rows and species amounts on both bases.

    Parameters
    ----------
    raw : dict
        ``GetRawMaterials`` tables keyed by basis, ``'mass'`` or ``'mole'``.
    path : str
        Database path.
    expected : list of tuple
        ``(unit, source, phase class, species amounts)`` in table order,
        with the amounts as returned by ``_expected_species``.
    """
    for basis, table in raw.items():
        amount = 'mass' if basis == 'mass' else 'moles'
        assert [key[:2] for key in table.index] == [
            row[:2] for row in expected]
        assert [key[2].split('.')[0] for key in table.index] == [
            row[2] for row in expected]
        for key, row in zip(table.index, expected):
            species = row[3][basis]  # [kg] or [mol]
            reported = table.loc[key, ['%s_%s' % (amount, name) for name in
                                       ('A', 'B', 'C', 'D', 'solvent')]]
            assert table.loc[key, amount] == pytest.approx(species.sum(),
                                                           rel=RTOL)
            np.testing.assert_allclose(reported.to_numpy(dtype=float),
                                       species, rtol=RTOL,
                                       atol=RTOL * species.sum())


def _raw(sim):
    """Return the raw-material totals on both bases.

    Parameters
    ----------
    sim : SimulationExec
        Solved flowsheet.

    Returns
    -------
    dict
        ``GetRawMaterials`` tables keyed by basis.
    """
    return {basis: sim.GetRawMaterials(basis=basis)
            for basis in ('mass', 'mole')}


def test_seeded_batch_crystallizer_reports_its_seed(path):
    """Report the user's seed once and the transferred liquid at M01."""
    sim = _seeded_flowsheet(path, {'M01': ['CR01'], 'CR01': []}, BatchCryst)
    sim.SolveFlowsheet(kwargs_run={'CR01': {'runtime': RUNTIME,
                                            'verbose': False}},
                       verbose=False)

    # The snapshot is marked transferred, by its liquid only.
    snapshot = sim.CR01.__original_phase__
    assert snapshot.transferred_from_uo
    assert [phase.transferred_from_uo for phase in snapshot.Phases] == [
        False, True]

    liquid = _expected_species(path, 'liquid')
    seed = _expected_species(path, 'seed')
    _assert_rows(_raw(sim), path, [
        ('M01', 'Inlet_0', 'LiquidPhase', liquid),
        ('CR01', 'Initial_holdup', 'SolidPhase', seed)])


def test_seeded_semibatch_crystallizer_reports_its_seed(path):
    """Report the seed and the own feed of a seeded SemibatchCryst."""
    sim = _seeded_flowsheet(path, {'M01': ['CR01'], 'CR01': []},
                            SemibatchCryst, vol_tank=TANK)
    sim.CR01.Inlet = LiquidStream(path, temp=TEMP, mass_frac=COMPOSITION,
                                  vol_flow=FEED)
    sim.SolveFlowsheet(kwargs_run={'CR01': {'runtime': RUNTIME,
                                            'verbose': False}},
                       verbose=False)

    _assert_rows(_raw(sim), path, [
        ('M01', 'Inlet_0', 'LiquidPhase', _expected_species(path, 'liquid')),
        ('CR01', 'Inlet_0', 'LiquidStream',
         _expected_species(path, 'liquid', FEED * RUNTIME)),
        ('CR01', 'Initial_holdup', 'SolidPhase',
         _expected_species(path, 'seed'))])


def test_wholly_transferred_slurry_is_not_a_raw_material(path):
    """Charge a slurry holdup to its sources only.

    The seeded crystallizer's outlet slurry carries copies of its liquid,
    marked as transferred from M01, and of its grown seed, not marked. The
    connection to CR02 marks every phase, so CR02 reports nothing and the
    seed is not counted twice.
    """
    sim = _seeded_flowsheet(
        path, {'M01': ['CR01'], 'CR01': ['CR02'], 'CR02': []}, BatchCryst)
    sim.CR02 = _crystallizer(BatchCryst)
    run = {'runtime': RUNTIME, 'verbose': False}
    sim.SolveFlowsheet(kwargs_run={'CR01': run, 'CR02': run}, verbose=False)

    snapshot = sim.CR02.__original_phase__
    assert snapshot.transferred_from_uo
    assert all(phase.transferred_from_uo for phase in snapshot.Phases)
    _assert_rows(_raw(sim), path, [
        ('M01', 'Inlet_0', 'LiquidPhase', _expected_species(path, 'liquid')),
        ('CR01', 'Initial_holdup', 'SolidPhase',
         _expected_species(path, 'seed'))])


def test_user_charge_is_reported_in_full(path):
    """Report both phases of a crystallizer charged wholly by the user."""
    sim = SimulationExec(path, {'CR01': []})
    sim.CR01 = _crystallizer(BatchCryst)
    sim.CR01.Phases = (_liquid(path), _seed(path))
    sim.SolveFlowsheet(kwargs_run={'CR01': {'runtime': RUNTIME,
                                            'verbose': False}},
                       verbose=False)

    _assert_rows(_raw(sim), path, [
        ('CR01', 'Initial_holdup', 'LiquidPhase',
         _expected_species(path, 'liquid')),
        ('CR01', 'Initial_holdup', 'SolidPhase',
         _expected_species(path, 'seed'))])
