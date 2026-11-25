"""Synthetic bulk-density charges and their public BatchReactor handoff.

The constant densities are fixture assumptions. The real CVode case uses
an isothermal, first-order split with unequal molar masses and no mass
change per reaction. The standalone charge tests need no optional solver.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase
from PharmaPy.ThermoModule import MissingPropertyError


# A splits into B and C with 100 = 40 + 60 g/mol. Carrier is inert and its
# distinct molar mass makes a mass/mole basis or order error observable.
MOLAR_MASS = {'A': 100.0, 'B': 40.0, 'C': 60.0, 'carrier': 80.0}  # [g/mol]
SPECIES_ORDERS = (('C', 'carrier', 'A', 'B'), ('B', 'C', 'carrier', 'A'))
# Asymmetric initial amounts make each ordered concentration distinguishable.
SOLUTES = {'A': 1.25, 'B': 0.375, 'C': 0.5}  # [mol/L]
SOLUTE_DENSITY = 170.0  # [kg/m**3], 125 + 15 + 30 from the charge above
NATIVE_DENSITY = 800.0  # [kg/m**3], all pure species share this synthetic value
BULK_DENSITY = 1050.0  # [kg/m**3], assumed constant for the synthetic liquid
NATIVE_CARRIER = 7.875  # [mol/L], (800 - 170) kg/m**3 / (80 g/mol)
BULK_CARRIER = 11.0  # [mol/L], (1050 - 170) kg/m**3 / (80 g/mol)
SOURCE = 'Synthetic constant bulk density for the mass-conserving split fixture'
# Cp_A = Cp_B + Cp_C, so the synthetic split has zero sensible reaction heat.
HEAT_CAPACITY = {'A': 100.0, 'B': 40.0, 'C': 60.0, 'carrier': 80.0}  # [J/mol/K]
TEMPERATURE = 300.0  # [K], fixed by the isothermal reactor
VOLUME = 0.002  # [m**3], 2 L distinguishes moles from molar concentrations
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
LITRES_PER_CUBIC_METRE = 1000.0  # [L/m**3], exact SI conversion
RATE = 0.4  # [1/s], gives a resolvable partial conversion in the reporting span
TIMES = np.array([0.0, 0.25, 0.75, 1.5, 2.5])  # [s], nonuniform reporting grid
# Use the CVode tolerances and the 100-fold profile allowance established in
# test_reactor_paramest_requested_rows.py for analytic first-order kinetics.
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [mol/L]
PROFILE_RTOL = 1e-7  # [-]
PROFILE_ATOL = 1e-9  # [mol/L]
ROUNDING_RTOL = 64 * np.finfo(float).eps  # [-], budget for short conversion chains


@pytest.fixture
def thermo_path(tmp_path: Path, request: pytest.FixtureRequest) -> str:
    """Write the synthetic species data in the requested database order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated temporary directory for the pure-species data.
    request : pytest.FixtureRequest
        Optional indirect parameter giving species names in file order.

    Returns
    -------
    str
        JSON path with molar masses [g/mol], pure densities [kg/m**3],
        and constant heat capacities [J/mol/K].
    """
    order = getattr(request, 'param', SPECIES_ORDERS[0])
    data = {
        name: {'mw': MOLAR_MASS[name], 'rho_liq': NATIVE_DENSITY,
               'cp_liq': [HEAT_CAPACITY[name]], 'source': 'Synthetic test data'}
        for name in order
    }
    path = tmp_path / 'bulk_density_species.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('thermo_path', SPECIES_ORDERS, indirect=True)
@pytest.mark.parametrize('measured', [False, True], ids=['native', 'measured'])
def test_batch_reactor_conserves_density_closed_charge(thermo_path, measured):
    """Follow the charge through real CVode to the batch outlet and holdup.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species JSON path in the parametrized species order.
    measured : bool
        Use the synthetic bulk density to complete the initial carrier charge.
    """
    pytest.importorskip('assimulo')
    from PharmaPy.Kinetics import RxnKinetics
    from PharmaPy.Reactors import BatchReactor
    from PharmaPy.Utilities import CoolingWater

    options = {}
    if measured:
        from PharmaPy._liquid_properties import BulkLiquidDensity

        options['bulk_density'] = BulkLiquidDensity(mass_density=BULK_DENSITY,
                                                    source=SOURCE)

    # The caller's mapping order differs from both JSON orders. A supplied
    # carrier concentration is a placeholder and must not control closure.
    charge = {'carrier': SOLUTES['B'], **SOLUTES}  # [mol/L]
    phase = LiquidPhase(thermo_path, temp=TEMPERATURE, vol=VOLUME,
                        name_solv='carrier', mole_conc=charge, **options)
    reactor = BatchReactor(isothermal=True)
    reactor.Phases = phase
    reactor.Kinetics = RxnKinetics(
        thermo_path, k_params=[RATE], ea_params=[0.0],
        rxn_list=['A --> B + C'])  # [1/s], [J/mol], constant coefficient
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE, mass_flow=UTILITY_FLOW)

    database_order = tuple(json.loads(
        Path(thermo_path).read_text(encoding='utf-8')))
    participating_order = tuple(name for name in database_order
                                if name != 'carrier')
    expected_carrier = BULK_CARRIER if measured else NATIVE_CARRIER  # [mol/L]
    expected_density = BULK_DENSITY if measured else NATIVE_DENSITY  # [kg/m**3]
    molar_masses = np.array([MOLAR_MASS[name] for name in database_order])  # [g/mol]
    # Integrated law for first-order disappearance, followed by the 1:1:1
    # stoichiometric increments and fixed (inert) carrier inventory.
    reactant = SOLUTES['A'] * np.exp(-RATE * TIMES)  # [mol/L]
    extent = SOLUTES['A'] - reactant  # [mol/L]
    expected = {'A': reactant, 'B': SOLUTES['B'] + extent,
                'C': SOLUTES['C'] + extent,
                'carrier': np.full(TIMES.shape, expected_carrier)}  # [mol/L]
    ordered = np.column_stack(
        [expected[name] for name in database_order])  # [mol/L]
    inventory = ordered * (VOLUME * LITRES_PER_CUBIC_METRE)  # [mol]
    species_mass = ordered * molar_masses * VOLUME  # [kg], 1 (mol/L)(g/mol) = 1 kg/m**3

    assert tuple(phase.name_species) == database_order
    assert charge == {'carrier': SOLUTES['B'], **SOLUTES}
    np.testing.assert_allclose(phase.mole_conc, ordered[0],
                               rtol=ROUNDING_RTOL, atol=0)
    np.testing.assert_allclose(phase.mole_frac * phase.moles, inventory[0],
                               rtol=ROUNDING_RTOL, atol=0)
    np.testing.assert_allclose(phase.mass_frac * phase.mass, species_mass[0],
                               rtol=ROUNDING_RTOL, atol=0)
    assert phase.mass == pytest.approx(expected_density * VOLUME,
                                       rel=ROUNDING_RTOL, abs=0)
    assert phase.getInventoryDensity() == pytest.approx(
        expected_density, rel=ROUNDING_RTOL, abs=0)
    assert phase.getDensity() == pytest.approx(NATIVE_DENSITY,
                                               rel=ROUNDING_RTOL, abs=0)

    reported_times, states = reactor.solve_unit(
        time_grid=TIMES, sundials_opts=SOLVER_OPTIONS, verbose=False)

    np.testing.assert_allclose(reported_times, TIMES, rtol=ROUNDING_RTOL, atol=0)
    assert reactor.result.di_states['mole_conc']['units'] == 'mol/L'
    named = reactor.result.retrieve_state('mole_conc')  # [mol/L]
    assert tuple(named) == participating_order
    np.testing.assert_allclose(
        states, np.column_stack([expected[name] for name in participating_order]),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    for name, profile in named.items():
        np.testing.assert_allclose(profile, expected[name], rtol=PROFILE_RTOL,
                                   atol=PROFILE_ATOL)

    assert reactor.Outlet is reactor.Liquid_1 is phase
    assert tuple(reactor.Outlet.name_species) == database_order
    assert reactor.Outlet.vol == pytest.approx(VOLUME, rel=ROUNDING_RTOL, abs=0)
    assert reactor.Outlet.mass == pytest.approx(expected_density * VOLUME,
                                                rel=PROFILE_RTOL, abs=0)
    assert reactor.Outlet.getInventoryDensity() == pytest.approx(
        expected_density, rel=PROFILE_RTOL, abs=0)
    np.testing.assert_allclose(reactor.Outlet.mole_conc, ordered[-1],
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    np.testing.assert_allclose(reactor.Outlet.mole_frac * reactor.Outlet.moles,
                               inventory[-1], rtol=PROFILE_RTOL, atol=0)
    np.testing.assert_allclose(reactor.Outlet.mass_frac * reactor.Outlet.mass,
                               species_mass[-1], rtol=PROFILE_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mole_conc', 'mass_conc'])
@pytest.mark.parametrize('carrier_conc', [0.0, BULK_CARRIER],  # [mol/L]
                         ids=['zero-carrier', 'positive-carrier'])
def test_bulk_density_closes_molar_and_mass_charges(thermo_path, basis, carrier_conc):
    """Complete either input basis, including the exactly exhausted residual.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species JSON path in the default fixture order.
    basis : {'mole_conc', 'mass_conc'}
        Input concentrations [mol/L] or [kg/m**3], respectively.
    carrier_conc : float
        Expected carrier concentration from the bulk density [mol/L].
    """
    from PharmaPy._liquid_properties import BulkLiquidDensity

    carrier_mass_conc = carrier_conc * MOLAR_MASS['carrier']  # [kg/m**3]
    bulk_mass_density = SOLUTE_DENSITY + carrier_mass_conc  # [kg/m**3]
    bulk = BulkLiquidDensity(mass_density=bulk_mass_density, source=SOURCE)
    order = SPECIES_ORDERS[0]
    concentrations = {**SOLUTES, 'carrier': carrier_conc}  # [mol/L]
    expected_molar = np.array([concentrations[name] for name in order])  # [mol/L]
    expected_mass = np.array([concentrations[name] * MOLAR_MASS[name]
                              for name in order])  # [kg/m**3]
    supplied = (expected_molar if basis == 'mole_conc'
                else expected_mass).copy()  # [mol/L] or [kg/m**3]
    # Use A's entry as a positive placeholder in the same units as the input.
    carrier_index = order.index('carrier')
    supplied[carrier_index] = supplied[order.index('A')]  # [mol/L] or [kg/m**3]
    original = supplied.copy()  # [mol/L] or [kg/m**3]

    phase = LiquidPhase(thermo_path, mass=bulk_mass_density * VOLUME,
                        name_solv='carrier', bulk_density=bulk, **{basis: supplied})

    assert phase.concentration_basis
    assert phase.bulk_density.source == SOURCE
    np.testing.assert_array_equal(supplied, original)
    np.testing.assert_allclose(phase.mole_conc, expected_molar,
                               rtol=ROUNDING_RTOL, atol=0)
    np.testing.assert_allclose(phase.mass_conc, expected_mass,
                               rtol=ROUNDING_RTOL, atol=0)
    assert phase.vol == pytest.approx(VOLUME, rel=ROUNDING_RTOL, abs=0)
    assert phase.getInventoryDensity('mole') == pytest.approx(
        expected_molar.sum(), rel=ROUNDING_RTOL, abs=0)


@pytest.mark.unit
def test_bulk_density_matching_native_charge_is_equivalent(thermo_path):
    """Recover the additive-volume charge when both closures prescribe 800 kg/m³.

    Parameters
    ----------
    thermo_path : str
        Synthetic species file with every pure density set to 800 kg/m³.
    """
    from PharmaPy._liquid_properties import BulkLiquidDensity

    options = {'vol': VOLUME, 'mole_conc': SOLUTES, 'name_solv': 'carrier'}
    native = LiquidPhase(thermo_path, **options)
    measured = LiquidPhase(thermo_path, **options, bulk_density=BulkLiquidDensity(
        mass_density=NATIVE_DENSITY, source=SOURCE))

    np.testing.assert_allclose(measured.mole_conc, native.mole_conc,
                               rtol=ROUNDING_RTOL, atol=0)
    np.testing.assert_allclose(measured.mass_conc, native.mass_conc,
                               rtol=ROUNDING_RTOL, atol=0)
    assert measured.mass == pytest.approx(native.mass, rel=ROUNDING_RTOL, abs=0)
    assert measured.moles == pytest.approx(native.moles, rel=ROUNDING_RTOL, abs=0)


@pytest.mark.unit
def test_bulk_density_is_only_an_initial_charge_property(thermo_path):
    """An explicit new composition determines the inventory at fixed volume.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species JSON path in the default fixture order.
    """
    from PharmaPy._liquid_properties import BulkLiquidDensity

    bulk = BulkLiquidDensity(mass_density=BULK_DENSITY, source=SOURCE)
    phase = LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                        mole_conc=SOLUTES, bulk_density=bulk)
    changed = {**SOLUTES, 'A': 0.0, 'carrier': BULK_CARRIER}  # [mol/L], remove A
    density_after = 925.0  # [kg/m**3], 1050 - 1.25 mol/L * 100 g/mol
    phase.updatePhase(vol=VOLUME, mole_conc=changed)

    assert phase.bulk_density == bulk
    assert phase.mass == pytest.approx(density_after * VOLUME,
                                       rel=ROUNDING_RTOL, abs=0)
    assert phase.getInventoryDensity() == pytest.approx(density_after,
                                                        rel=ROUNDING_RTOL, abs=0)
    np.testing.assert_allclose(phase.mole_conc,
                               [changed[name] for name in SPECIES_ORDERS[0]],
                               rtol=ROUNDING_RTOL, atol=0)


@pytest.mark.unit
def test_bulk_charge_without_pure_solvent_density(thermo_path):
    """Closure succeeds without a pure density still required by getDensity.

    Parameters
    ----------
    thermo_path : str
        Test-owned synthetic property file edited to omit the carrier density.
    """
    from PharmaPy._liquid_properties import BulkLiquidDensity

    path = Path(thermo_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    del data['carrier']['rho_liq']
    path.write_text(json.dumps(data), encoding='utf-8')
    phase = LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                        mole_conc=SOLUTES, bulk_density=BulkLiquidDensity(
                            mass_density=BULK_DENSITY, source=SOURCE))

    assert phase.mass == pytest.approx(BULK_DENSITY * VOLUME,
                                       rel=ROUNDING_RTOL, abs=0)
    with pytest.raises(MissingPropertyError, match=r'carrier.*rho_liq'):
        phase.getDensity()


@pytest.mark.unit
def test_bulk_charge_allows_exhaustion_at_binary_roundoff(thermo_path):
    """Allow a zero solvent residual when decimal inputs exactly close."""
    from PharmaPy.Phases import BulkLiquidDensity

    # The second solute weighs twice the first. Their decimal mass
    # concentrations add to the bulk density, subject to binary rounding.
    solutes = {'A': 300.1, 'B': 600.2}  # [kg/m**3]
    bulk_density = BulkLiquidDensity(900.3, SOURCE)  # [kg/m**3]
    phase = LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                        mass_conc=solutes, bulk_density=bulk_density)

    assert phase.mass_conc[SPECIES_ORDERS[0].index('carrier')] == 0.0
    assert phase.getInventoryDensity() == pytest.approx(
        bulk_density.mass_density, rel=ROUNDING_RTOL, abs=0,
    )


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass_conc', 'mole_conc'])
def test_bulk_charge_exhaustion_agrees_in_mass_and_mole_bases(thermo_path, basis):
    """Preserve the same exhausted charge in either concentration basis.

    Parameters
    ----------
    thermo_path : str
        Synthetic species database to update with decimal molar masses [g/mol].
    basis : {'mass_conc', 'mole_conc'}
        Whether the test supplies solutes in [kg/m**3] or [mol/L].
    """
    from PharmaPy.Phases import BulkLiquidDensity

    path = Path(thermo_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    # The decimal factors yield 79.94 and 713.36 kg/m**3 exactly. Their
    # binary64 products and sum round above the given density.
    data['A']['mw'] = 57.1  # [g/mol]
    data['B']['mw'] = 96.4  # [g/mol]
    path.write_text(json.dumps(data), encoding='utf-8')
    mass_concs = {'A': 79.94, 'B': 713.36}  # [kg/m**3]
    mole_concs = {'A': 1.4, 'B': 7.4}  # [mol/L]
    bulk_density = BulkLiquidDensity(793.3, SOURCE)  # [kg/m**3], sum of decimal masses
    supplied = mass_concs if basis == 'mass_conc' else mole_concs  # [kg/m**3] or [mol/L]

    phase = LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                        bulk_density=bulk_density, **{basis: supplied})

    np.testing.assert_allclose(phase.mass_conc, [0, 0, 79.94, 713.36],
                               rtol=ROUNDING_RTOL, atol=0)
    np.testing.assert_allclose(phase.mole_conc, [0, 0, 1.4, 7.4],
                               rtol=ROUNDING_RTOL, atol=0)
    assert phase.mass == pytest.approx(1.5866, rel=ROUNDING_RTOL, abs=0)


@pytest.mark.unit
@pytest.mark.parametrize(('basis', 'excess_steps'), [
    ('mass_conc', 4), ('mole_conc', 6),
])
def test_bulk_charge_rejects_excess_beyond_roundoff(thermo_path, basis, excess_steps):
    """Reject excess just above the rounding bound using an exact binary conversion.

    Parameters
    ----------
    thermo_path : str
        Synthetic species database with a selected molar mass [g/mol].
    basis : {'mass_conc', 'mole_conc'}
        Whether the test supplies solutes in [kg/m**3] or [mol/L].
    excess_steps : int
        Representable float64 spacing steps [-] past the supplied bulk density.
    """
    from PharmaPy.Phases import BulkLiquidDensity

    path = Path(thermo_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    molar_mass = 64.0  # [g/mol], power of two so conversion is exact in binary
    data['A']['mw'] = molar_mass  # [g/mol]
    path.write_text(json.dumps(data), encoding='utf-8')
    bulk_density = BulkLiquidDensity(BULK_DENSITY, SOURCE)  # [kg/m**3]
    # Four or six steps place the solute one step above the maximum
    # three-stage mass or five-stage molar rounding allowance.
    solute_mass_conc = BULK_DENSITY + excess_steps * np.spacing(BULK_DENSITY)  # [kg/m**3]
    supplied = (solute_mass_conc if basis == 'mass_conc'
                else solute_mass_conc / molar_mass)  # [kg/m**3] or [mol/L]

    with pytest.raises(ValueError, match='exceed the bulk liquid mass density'):
        LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                    bulk_density=bulk_density, **{basis: {'A': supplied}})


@pytest.mark.unit
def test_bulk_charge_rejects_masked_density():
    """A masked density contains no measured value."""
    from PharmaPy.Phases import BulkLiquidDensity

    density = np.ma.array(BULK_DENSITY, mask=True)  # [kg/m**3]
    with pytest.raises(ValueError, match='mass_density.*masked'):
        BulkLiquidDensity(density, SOURCE)


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['array', 'mapping', 'masked-constant'])
def test_bulk_charge_rejects_masked_solutes(thermo_path, kind):
    """Reject a missing solute before its backing value becomes inventory.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species data in the default fixture order.
    kind : {'array', 'mapping', 'masked-constant'}
        Container of masked mass concentration data [kg/m**3].
    """
    from PharmaPy.Phases import BulkLiquidDensity

    mass_conc = SOLUTES['A'] * MOLAR_MASS['A']  # [kg/m**3]
    if kind == 'array':
        supplied = np.ma.array([0, 0, mass_conc, 0],
                               mask=[False, False, True, False])  # [kg/m**3]
    elif kind == 'mapping':
        supplied = {'A': np.ma.array(mass_conc, mask=True)}  # [kg/m**3]
    else:
        supplied = {'A': np.ma.masked}  # [kg/m**3]

    with pytest.raises(ValueError, match='mass_conc.*masked solute'):
        LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                    mass_conc=supplied,
                    bulk_density=BulkLiquidDensity(BULK_DENSITY, SOURCE))


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['array', 'mapping'])
def test_bulk_charge_replaces_a_masked_solvent_placeholder(thermo_path, kind):
    """Complete a solvent entry whose original backing value is unavailable.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species data in the default fixture order.
    kind : {'array', 'mapping'}
        Container of measured molar concentrations [mol/L].
    """
    from PharmaPy.Phases import BulkLiquidDensity

    if kind == 'array':
        supplied = np.ma.array([SOLUTES['C'], 0, SOLUTES['A'], SOLUTES['B']],
                               mask=[False, True, False, False])  # [mol/L]
        solvent_key = SPECIES_ORDERS[0].index('carrier')
    else:
        supplied = {**SOLUTES, 'carrier': np.ma.masked}  # [mol/L]
        solvent_key = 'carrier'
    phase = LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                        mole_conc=supplied,
                        bulk_density=BulkLiquidDensity(BULK_DENSITY, SOURCE))

    np.testing.assert_allclose(
        phase.mole_conc, [SOLUTES['C'], BULK_CARRIER, SOLUTES['A'], SOLUTES['B']],
        rtol=ROUNDING_RTOL, atol=0,
    )
    assert np.ma.is_masked(supplied[solvent_key])


@pytest.mark.unit
@pytest.mark.parametrize('supplied', [
    {'A': [SOLUTES['A']], 'B': [SOLUTES['B']]},
    {'carrier': [BULK_CARRIER]},
], ids=['solute-profiles', 'solvent-profile'])  # [mol/L], one-point profiles
def test_bulk_charge_rejects_profile_mapping_entries(thermo_path, supplied):
    """Reject profiles even when the named solvent will be overwritten.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species data in the default fixture order.
    supplied : dict[str, list[float]]
        Species-keyed one-point concentration profiles [mol/L].
    """
    from PharmaPy.Phases import BulkLiquidDensity

    with pytest.raises(ValueError, match='mole_conc mapping entries must be scalar'):
        LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                    mole_conc=supplied,
                    bulk_density=BulkLiquidDensity(BULK_DENSITY, SOURCE))


@pytest.mark.unit
@pytest.mark.parametrize('as_mapping', [False, True])
def test_bulk_density_rejects_complex_solute_concentrations(thermo_path, as_mapping):
    """Reject loss of the imaginary part in either composition format."""
    from PharmaPy.Phases import BulkLiquidDensity

    # A nonzero imaginary component makes this value an invalid concentration.
    concentration = np.complex128(SOLUTES['A'] * (1 + 1j))  # [mol/L]
    input_concentrations = {'A': concentration} if as_mapping else np.array([
        concentration if species == 'A' else 0 for species in SPECIES_ORDERS[0]
    ])  # [mol/L]
    bulk_density = BulkLiquidDensity(BULK_DENSITY, SOURCE)  # [kg/m**3]

    with pytest.raises(ValueError, match="mole_conc.*real"):
        LiquidPhase(thermo_path, vol=VOLUME, name_solv='carrier',
                    mole_conc=input_concentrations, bulk_density=bulk_density)


@pytest.mark.unit
@pytest.mark.parametrize('density', [0.0, -BULK_DENSITY, np.nan, np.inf])  # [kg/m**3]
def test_bulk_density_rejects_nonpositive_or_nonfinite_density(density):
    """Reject a density [kg/m**3] that cannot specify an initial liquid."""
    from PharmaPy._liquid_properties import BulkLiquidDensity

    with pytest.raises(ValueError, match='mass_density.*finite positive scalar'):
        BulkLiquidDensity(mass_density=density, source=SOURCE)


@pytest.mark.unit
@pytest.mark.parametrize('source', [None, '  '])
def test_bulk_density_requires_a_source(source):
    """Reject an absent or blank bulk-density source description."""
    from PharmaPy._liquid_properties import BulkLiquidDensity

    with pytest.raises(ValueError, match='source.*nonempty string'):
        BulkLiquidDensity(mass_density=BULK_DENSITY, source=source)


@pytest.mark.unit
@pytest.mark.parametrize('overrides, message', [
    ({'name_solv': None}, 'requires one named solvent'),
    ({'name_solv': 'unknown'}, 'Unknown solvent species'),
    ({'mole_conc': None}, 'requires exactly one of mole_conc or mass_conc'),
    ({'mole_conc': {'unknown': SOLUTES['A']}}, 'Unknown species.*unknown'),
    ({'mole_conc': [[SOLUTES.get(name, 0.0) for name in SPECIES_ORDERS[0]]]},
     'must have shape'),  # [mol/L], correct species axis with an extra time axis
    ({'mole_conc': [0.0, 0.0, -SOLUTES['A'], 0.0]},
     'finite and non-negative'),  # [mol/L], negative A
    ({'mole_conc': [0.0, 0.0, np.inf, 0.0]},
     'finite and non-negative'),  # [mol/L], nonfinite A
    ({'mass_conc': {'A': SOLUTE_DENSITY, 'B': 0.0}, 'mole_conc': None},
     'exceed the bulk liquid mass density'),  # [kg/m**3], A alone exceeds total
], ids=['omitted-solvent', 'unknown-solvent', 'missing-concentration',
        'unknown-species', 'profile', 'negative-solute', 'nonfinite-solute',
        'infeasible-mass'])
def test_bulk_charge_rejects_invalid_state(thermo_path, overrides, message):
    """Reject malformed charges or a negative residual for the named solvent.

    Parameters
    ----------
    thermo_path : str
        Synthetic pure-species JSON path in the default fixture order.
    overrides : dict
        Invalid constructor arguments, including any molar concentrations
        [mol/L] or mass concentrations [kg/m**3] in the case under test.
    message : str
        Regex for the corresponding public diagnostic.
    """
    from PharmaPy._liquid_properties import BulkLiquidDensity

    # 125 kg/m**3 can hold a pure A charge at its initial 1.25 mol/L,
    # but cannot hold 170 kg/m**3 of A. Other cases fail input validation.
    density = SOLUTES['A'] * MOLAR_MASS['A']  # [kg/m**3]
    options = {'name_solv': 'carrier', 'mole_conc': SOLUTES, **overrides}
    with pytest.raises(ValueError, match=message):
        LiquidPhase(thermo_path, vol=VOLUME, bulk_density=BulkLiquidDensity(
            mass_density=density, source=SOURCE), **options)
