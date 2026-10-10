"""Published state metadata of liquid ``Mixer`` solves (issue #407).

The liquid Mixer's ``mass_frac`` state is a dimensionless vector with one
entry per species, so ``states_di`` must declare empty units and
``dim == num_species`` in both flow and batch modes, as the solids Mixer
branch (PR #403) does. ``dim_states`` then sums to the packed width
num_species + 2 (amount or flow, fractions, temperature), the count that
``SimulationResult`` already reports.

Each mode runs with two shipped databases of different sizes, the
five-species ``tests/Flowsheet/data/compound_database.json`` and the
four-species ``tests/integration/data/pfr_test_pure_comp.json``, so a
hard-coded dimension cannot pass. Species names and counts are read from the
JSON files. Mixers are instantaneous; no ODE backend is needed.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.MixedPhases import Slurry
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

# Database file and two unequal inlet compositions [-] in its species order.
DATABASES = {
    'five-species': ('flowsheet', 'compound_database.json',
                     np.array([0.4, 0.05, 0.15, 0.1, 0.3]),
                     np.array([0.0, 0.0, 0.0, 0.0, 1.0])),
    'four-species': ('integration', 'pfr_test_pure_comp.json',
                     np.array([0.5, 0.2, 0.1, 0.2]),
                     np.array([0.0, 0.0, 0.0, 1.0])),
    }  # [-]
HOT = 320.0  # [K], first inlet
COLD = 300.0  # [K], second inlet
FIRST_AMOUNT = 1.0  # [kg] or [kg/s], first inlet
SECOND_AMOUNT = 3.0  # [kg] or [kg/s], unequal to FIRST_AMOUNT
# Solids Mixer fixture of PR #403 (issue #289): slurry on a four-node grid.
GRID = np.array([0., 100., 200., 300.])  # [um]
SLURRY_DISTRIB = np.array([0., 1e9, 1.25e8, 0.])  # [#/m**3/um]
SLURRY_VOL = 1e-3  # [m**3]
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
# Amount or flow, and temperature: the two scalar liquid states.
NUM_SCALAR_STATES = 2  # [-]
# Relative tolerance [-] for a float64 sum of mixed fractions; roundoff only.
RTOL = 1e-12


def _database(data_path, name):
    """Return a database path, its species, and two inlet compositions.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    name : str
        Key of ``DATABASES``.

    Returns
    -------
    tuple
        Database path, species names in JSON order, and two mass-fraction
        arrays [-] of shape (num_species,).
    """
    folder, file_name, first, second = DATABASES[name]
    path = str(data_path[folder] / file_name)
    with open(path) as handle:
        species = list(json.load(handle))
    return path, species, first, second


def _solve_liquid_mixer(path, mode, first, second):
    """Solve a one-unit flowsheet mixing two liquid inlets.

    Parameters
    ----------
    path : str
        Database path.
    mode : {'continuous', 'batch'}
        Streams with mass flows [kg/s] or phases with masses [kg].
    first, second : numpy.ndarray
        Inlet mass fractions [-], shape (num_species,).

    Returns
    -------
    SimulationExec
        Solved flowsheet with the Mixer ``M01``.
    """
    if mode == 'continuous':
        inlets = [LiquidStream(path, temp=HOT, mass_frac=first,
                               mass_flow=FIRST_AMOUNT),
                  LiquidStream(path, temp=COLD, mass_frac=second,
                               mass_flow=SECOND_AMOUNT)]
    else:
        inlets = [LiquidPhase(path, temp=HOT, mass_frac=first,
                              mass=FIRST_AMOUNT),
                  LiquidPhase(path, temp=COLD, mass_frac=second,
                              mass=SECOND_AMOUNT)]
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = inlets
    sim.SolveFlowsheet(verbose=False)
    return sim


@pytest.mark.parametrize('database', list(DATABASES))
@pytest.mark.parametrize('mode, amount, units', [
    ('continuous', 'mass_flow', 'kg/s'), ('batch', 'mass', 'kg')])
def test_liquid_mixer_declares_dimensionless_species_fractions(
        data_path, database, mode, amount, units):
    """Declare and print ``mass_frac`` as num_species dimensionless entries."""
    path, species, first, second = _database(data_path, database)
    num_species = len(species)

    sim = _solve_liquid_mixer(path, mode, first, second)
    mixer = sim.M01

    assert mixer.name_states == [amount, 'mass_frac', 'temp']
    assert mixer.states_di == {
        amount: {'units': units, 'dim': 1, 'type': 'alg'},
        'mass_frac': {'units': '', 'dim': num_species, 'index': species,
                      'type': 'alg'},
        'temp': {'units': 'K', 'dim': 1, 'type': 'alg'},
        }
    assert mixer.dim_states == [1, num_species, 1]
    assert sum(mixer.dim_states) == num_species + NUM_SCALAR_STATES
    # The published fractions have the declared length and sum to one.
    published = np.asarray(mixer.result.mass_frac)  # [-]
    assert published.shape[-1] == num_species
    assert np.sum(published) == pytest.approx(1.0, rel=RTOL)

    # DynamicResult row: name, dim, then the species index (no unit text).
    rows = [line.split() for line in repr(mixer.result).splitlines()]
    row = next(row for row in rows if row and row[0] == 'mass_frac')
    assert row == ['mass_frac', str(num_species)] + ', '.join(species).split()

    # SimulationResult counts are unchanged: num_species + 2 algebraic.
    summary = next(line.split() for line in sim.result.out_uos
                   if line.startswith('M01'))
    assert summary == ['M01', '0', str(num_species + NUM_SCALAR_STATES),
                       'ALG', 'Mixer']


def test_liquid_and_solids_mixers_share_mass_frac_metadata(data_path):
    """Use the solids branch convention for the liquid ``mass_frac`` state."""
    path, _, first, second = _database(data_path, 'five-species')
    liquid_mixer = _solve_liquid_mixer(path, 'batch', first, second).M01

    slurry = Slurry(vol=SLURRY_VOL, x_distrib=GRID.copy(),
                    distrib=SLURRY_DISTRIB.copy())
    # The slurry sets its liquid amount, so the liquid starts at zero.
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        liquid = LiquidPhase(path, mass_frac=second, temp=HOT)
    slurry.Phases = (liquid, SolidPhase(path, mass_frac=SOLID_COMPOSITION,
                                        temp=HOT))
    solids_mixer = Mixer()
    solids_mixer.Inlets = [LiquidPhase(path, mass_frac=first,
                                       mass=FIRST_AMOUNT, temp=COLD), slurry]
    solids_mixer.solve_unit()

    assert (liquid_mixer.states_di['mass_frac']
            == solids_mixer.states_di['mass_frac'])
