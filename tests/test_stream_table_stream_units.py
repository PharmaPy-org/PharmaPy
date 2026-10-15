"""Stream tables and unit summaries of flash and extractor flowsheets (#451).

``IsothermalFlash``, ``AdiabaticFlash`` and ``ContinuousExtractor`` publish
two products (``LiquidOut``/``VaporOut``, ``Liquid_2``/``Liquid_3``) and no
``Outlet``, because they cannot be flowsheet sources (issue #445).
``SimulationResult.GetStreamTable`` reports them through their
``get_reported_outlets`` hook, as the rows ``Outlet_liquid`` and
``Outlet_vapor`` (flashes) or ``Outlet_heavy`` and ``Outlet_light``
(extractor). A flowsheet of flashes alone, which declare no ``states_di``,
prints an empty unit summary instead of failing in ``SolveFlowsheet``.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json) and the feeds of
tests/test_connection_unit_contract.py: two liquids at 360 K and 1 atm,
1.0 and 0.5 mol/s of asymmetric compositions, either mixed by a continuous
``Mixer`` or fed premixed as one 1.5 mol/s raw stream. Both flashes split
the mixture into a liquid and a vapor; the extractor uses constant
synthetic partition coefficients, so its split is deterministic. Expected
product totals are the feed sums from these specifications, with mass
flows from the database molar masses; the vapor row is checked against the
ideal-gas volume flow. A steady-state unit has no fed duration, so the raw
feed amounts are NaN with a warning. A ``BatchExtractor`` inherits the
hook and reports the amounts [mol] of a 1.5 mol charge of the same mixed
composition. No ODE backend is needed.
"""

import json

import numpy as np
import pytest
from scipy.constants import R as GAS_CONSTANT  # [J/mol/K], CODATA

from PharmaPy.Containers import Mixer
from PharmaPy.Evaporators import AdiabaticFlash, IsothermalFlash
from PharmaPy.Extractors import BatchExtractor, ContinuousExtractor
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

FEED_1 = np.array([0.3, 0.1, 0.3, 0.1, 0.2])  # [-], mole fractions
FEED_2 = np.array([0.1, 0.3, 0.1, 0.3, 0.2])  # [-], mole fractions
FLOW_1 = 1.0  # [mol/s]
FLOW_2 = 0.5  # [mol/s]
TEMP = 360.0  # [K], two-phase for the mixed feed at 1 atm
PRESSURE = 101325.0  # [Pa], standard atmosphere
# [-], constant liquid-liquid partition coefficients of species A to D and
# the solvent; synthetic values that split A and B and leave the others.
PARTITION = np.array([2., .5, 1., 1., 1.])
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], relative balance tolerance: the flash residual allowance
# sqrt(float64 epsilon) of PharmaPy.Evaporators, with margin, as in
# tests/test_connection_unit_contract.py.
BALANCE_RTOL = 1e-7
# [-], PharmaPy's vapor density uses R = 8.314 J/mol/K, truncated from the
# CODATA value used here (relative difference 5.6e-5).
IDEAL_GAS_RTOL = 1e-4
# Column headers of the unit summary of SimulationResult; with no rows,
# PharmaPy.Results.pprint pads each header with two spaces.
SUMMARY_HEADERS = ('Unit operation', 'Diff eqns', 'Alg eqns', 'Model type',
                   'PharmaPy type')

UNITS = {
    'isothermal-flash': (lambda: IsothermalFlash(pres_drum=PRESSURE),
                         ('Outlet_liquid', 'Outlet_vapor')),
    'adiabatic-flash': (lambda: AdiabaticFlash(pres_drum=PRESSURE),
                        ('Outlet_liquid', 'Outlet_vapor')),
    'continuous-extractor': (lambda: ContinuousExtractor(
        k_fun=lambda x_1, x_2, temp: PARTITION),
        ('Outlet_heavy', 'Outlet_light')),
    }


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


def _feed(path, fractions, flow):
    """Build a raw liquid feed at the fixture temperature and pressure.

    Parameters
    ----------
    path : str
        Database path.
    fractions : numpy.ndarray
        Mole fractions [-], shape (5,).
    flow : float
        Molar flow [mol/s].

    Returns
    -------
    LiquidStream
        Raw feed stream.
    """
    return LiquidStream(path, temp=TEMP, pres=PRESSURE, mole_flow=flow,
                        mole_frac=fractions)


def _solved_flowsheet(path, case, with_mixer):
    """Solve a flowsheet ending in the unit of ``case``.

    Parameters
    ----------
    path : str
        Database path.
    case : str
        Key of ``UNITS``.
    with_mixer : bool
        If True, a continuous Mixer of the two feeds feeds the unit
        (``M01 --> U01``); otherwise the unit alone receives the premixed
        feed as a raw ``Inlet``.

    Returns
    -------
    SimulationExec
        Solved flowsheet.
    """
    make_unit, _ = UNITS[case]
    if with_mixer:
        sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
        sim.M01 = Mixer()
        sim.M01.Inlets = [_feed(path, FEED_1, FLOW_1),
                          _feed(path, FEED_2, FLOW_2)]
    else:
        sim = SimulationExec(path, {'U01': []})
    sim.U01 = make_unit()
    if not with_mixer:
        total = FLOW_1 + FLOW_2  # [mol/s]
        mixed = (FLOW_1 * FEED_1 + FLOW_2 * FEED_2) / total  # [-]
        sim.U01.Inlet = _feed(path, mixed, total)
    sim.SolveFlowsheet(verbose=False)
    return sim


def _species_inflow(path):
    """Return the species feed of the fixture on both bases.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    moles : numpy.ndarray
        Species molar flows [mol/s], database order, shape (5,).
    mass : numpy.ndarray
        Species mass flows [kg/s], database order, shape (5,).
    """
    moles = FLOW_1 * FEED_1 + FLOW_2 * FEED_2  # [mol/s]
    with open(path) as handle:
        weights = np.array([value['mw'] for value in json.load(handle)
                            .values()])  # [g/mol], database order
    return moles, moles * weights * G_TO_KG  # [mol/s], [kg/s]


@pytest.mark.parametrize('with_mixer', [True, False],
                         ids=['mixer-fed', 'single-unit'])
@pytest.mark.parametrize('case', list(UNITS))
def test_stream_table_reports_both_products(path, case, with_mixer):
    """Report both products of the unit and conserve the feed on each basis."""
    sim = _solved_flowsheet(path, case, with_mixer)
    _, product_rows = UNITS[case]
    species = sim.NamesSpecies
    species_moles, species_mass = _species_inflow(path)  # [mol/s], [kg/s]

    if with_mixer:
        unit, reason = 'instantaneous unit', (
            "downstream unit 'U01' has no timed result: its solve failed, "
            'or it is a steady-state unit such as a flash')
        name = 'M01'
    else:
        unit, reason = 'unit', ('it has no timed result: it is a '
                                'steady-state unit such as a flash, or its '
                                'solve failed')
        name = 'U01'
    message = (rf"^Raw inlets of {unit} '{name}' have no defined fed "
               rf"duration: {reason}; their totals are reported as NaN\.$")

    tables = {}
    for basis in ('mass', 'mole'):
        with pytest.warns(RuntimeWarning, match=message):
            tables[basis] = sim.result.GetStreamTable(basis=basis)

    for basis, table in tables.items():
        rows = table.index.droplevel(2).tolist()
        raw_rows = ([('M01', 'Inlet_0'), ('M01', 'Inlet_1'),
                     ('M01', 'Outlet')] if with_mixer
                    else [('U01', 'Inlet_0')])
        assert rows == raw_rows + [('U01', row) for row in product_rows]

        flow, frac, expected = (
            ('mass_flow', 'mass_frac', species_mass) if basis == 'mass'
            else ('mole_flow', 'mole_frac', species_moles))
        products = table.loc['U01'].loc[list(product_rows)]
        flows = products[flow].to_numpy()  # [kg/s] or [mol/s]
        fractions = products[['%s_%s' % (frac, comp)
                              for comp in species]].to_numpy()  # [-]
        assert np.all(flows > 0) and np.all(flows < expected.sum())
        assert flows.sum() == pytest.approx(expected.sum(), rel=BALANCE_RTOL)
        np.testing.assert_allclose(flows @ fractions, expected, rtol=0,
                                   atol=BALANCE_RTOL * expected.sum())

        # A steady-state unit has no fed duration: the raw totals are NaN,
        # but the raw feed flow is reported.
        raw = table.loc[raw_rows[0][0]].loc[[row for _, row in raw_rows
                                             if row != 'Outlet']]
        amount = 'mass' if basis == 'mass' else 'moles'
        assert np.isnan(raw[amount].to_numpy()).all()
        assert raw[flow].sum() == pytest.approx(expected.sum(),
                                                rel=BALANCE_RTOL)

    products = tables['mole'].loc['U01']
    molar_density = (products['mole_flow']
                     / products['vol_flow'])  # [mol/m**3]
    if case == 'continuous-extractor':
        # The extractor's heavy liquid is the one with the larger molar
        # density (its own convention, also used by x_heavy/x_light).
        assert (molar_density['Outlet_heavy'].iloc[0]
                > molar_density['Outlet_light'].iloc[0])
    else:
        vapor = products.loc['Outlet_vapor'].iloc[0]
        ideal_gas = vapor['pres'] / (GAS_CONSTANT * vapor['temp'])  # [mol/m**3]
        assert molar_density['Outlet_vapor'].iloc[0] == pytest.approx(
            ideal_gas, rel=IDEAL_GAS_RTOL)
        assert molar_density['Outlet_liquid'].iloc[0] > ideal_gas


@pytest.mark.parametrize('case', ['isothermal-flash', 'adiabatic-flash'])
def test_flash_only_flowsheet_prints_an_empty_unit_summary(path, case):
    """Solve a flash-only flowsheet whose unit summary has no rows."""
    sim = _solved_flowsheet(path, case, with_mixer=False)

    width = sum(len(header) + 2 for header in SUMMARY_HEADERS)  # [-]
    separator = '-' * width
    assert sim.result.out_uos[0] == separator
    assert sim.result.out_uos[1].split() == ' '.join(SUMMARY_HEADERS).split()
    assert sim.result.out_uos[2:] == [separator, separator]
    assert repr(sim.result).endswith(
        '\n'.join(sim.result.out_uos))
    # The hook reports the products without making the flash a source.
    assert not hasattr(sim.U01, 'Outlet')


def test_batch_extractor_reports_both_liquid_amounts(path):
    """Report the inherited heavy and light rows of a BatchExtractor charge.

    The batch extractor separates an inventory [mol], so its rows carry
    amounts instead of flows; heavy plus light equals the charge, which is
    reported first as the unit's initial holdup (issue #456).
    """
    charge = FLOW_1 + FLOW_2  # [mol], same numbers as the 1.5 mol/s feed
    species_moles, species_mass = _species_inflow(path)  # [mol], [kg]
    sim = SimulationExec(path, {'X01': []})
    sim.X01 = BatchExtractor(k_fun=lambda x_1, x_2, temp: PARTITION)
    sim.X01.Phases = LiquidPhase(path, temp=TEMP, pres=PRESSURE,
                                 moles=charge,
                                 mole_frac=species_moles / charge)
    sim.SolveFlowsheet(verbose=False)

    for basis, amount, frac, expected in (
            ('mass', 'mass', 'mass_frac', species_mass),
            ('mole', 'moles', 'mole_frac', species_moles)):
        table = sim.result.GetStreamTable(basis=basis)
        assert table.index.droplevel(2).tolist() == [
            ('X01', 'Initial_holdup'), ('X01', 'Outlet_heavy'),
            ('X01', 'Outlet_light')]
        amounts = table[amount].to_numpy()  # [kg] or [mol]
        fractions = table[['%s_%s' % (frac, comp)
                           for comp in sim.NamesSpecies]].to_numpy()  # [-]
        assert np.all(amounts > 0)
        for rows in (slice(0, 1), slice(1, 3)):  # the charge, the products
            np.testing.assert_allclose(
                amounts[rows] @ fractions[rows], expected, rtol=0,
                atol=BALANCE_RTOL * expected.sum())
