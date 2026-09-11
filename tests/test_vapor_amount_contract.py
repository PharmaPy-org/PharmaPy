"""Ideal-gas vapor amounts, density, update precedence, and caller handoffs.

Synthetic unequal species masses and asymmetric compositions expose basis and
ordering errors. Tests exercise real converters without an optional ODE solver.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/45
https://github.com/PharmaPy-org/PharmaPy/issues/183
https://github.com/PharmaPy-org/PharmaPy/issues/216
"""

import json

import numpy as np
import pytest

from PharmaPy.NameAnalysis import NameAnalyzer
from PharmaPy.Phases import LiquidPhase, VaporPhase
from PharmaPy.Streams import LiquidStream, VaporStream
from conftest import THERMO_TWO_SPECIES

pytestmark = pytest.mark.unit

# Independent ideal-gas reference, rounded as required by the evaporator model.
GAS_CONSTANT = 8.314  # [J/mol/K]
TEMPERATURE = 300.0  # [K], synthetic ambient gas case
PRESSURE = 101325.0  # [Pa], one standard atmosphere
INITIAL_FRACTIONS = np.array([0.25, 0.75])  # [-], asymmetric synthetic mixture
INITIAL_MW = 79.5  # [g/mol], 0.25 * 18 + 0.75 * 100
UPDATED_FRACTIONS = np.array([0.6, 0.4])  # [-], different synthetic composition
UPDATED_MW = 50.8  # [g/mol], 0.6 * 18 + 0.4 * 100
ONE_MOLE = 1.0  # [mol], reference inventory (or mol/s for streams)
TWO_MOLES = 2.0  # [mol], doubled inventory (or mol/s for streams)
MOLAR_VOLUME = GAS_CONSTANT * TEMPERATURE / PRESSURE  # [m**3/mol], ideal gas
# Roundoff-only tolerance: direct algebra on two species needs no solver error.
REL_TOL = 1e-12  # [-]


@pytest.fixture
def thermo_path(tmp_path):
    """Write the established two-species synthetic thermophysical data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the test's JSON file.

    Returns
    -------
    str
        Path to the two-species JSON file.
    """
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(THERMO_TWO_SPECIES))
    return str(path)


@pytest.fixture(params=[VaporPhase, VaporStream], ids=['phase', 'stream'])
def vapor(request, thermo_path):
    """Construct one mole, or one mole per second, of the reference mixture.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selects the phase or stream class.
    thermo_path : str
        Path to the synthetic thermophysical data.

    Returns
    -------
    VaporPhase or VaporStream
        Gas at 300 K and 101325 Pa with a 1 mol inventory or 1 mol/s flow.
    """
    amount_name = 'mole_flow' if request.param is VaporStream else 'moles'
    return request.param(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS,
                         **{amount_name: ONE_MOLE})


def assert_amounts(vapor, moles, mw=INITIAL_MW, temp=TEMPERATURE, pres=PRESSURE):
    """Check inventory or flow against independent mass and ideal-gas balances.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Object whose amounts and optional flow aliases are checked.
    moles : float
        Expected inventory [mol] or flow [mol/s].
    mw : float, optional
        Independently derived average molecular weight [g/mol].
    temp, pres : float, optional
        Expected temperature [K] and pressure [Pa].
    """
    assert vapor.moles == pytest.approx(moles, rel=REL_TOL)
    assert vapor.mass * 1000 == pytest.approx(moles * mw, rel=REL_TOL)  # 1000 g/kg
    assert vapor.vol * pres == pytest.approx(
        moles * GAS_CONSTANT * temp, rel=REL_TOL)  # [J] or [J/s]
    assert vapor.mw_av == pytest.approx(mw, rel=REL_TOL)
    if isinstance(vapor, VaporStream):
        assert vapor.mole_flow == pytest.approx(moles, rel=REL_TOL)
        assert vapor.mass_flow == pytest.approx(vapor.mass, rel=REL_TOL)
        assert vapor.vol_flow == pytest.approx(vapor.vol, rel=REL_TOL)


def test_density_default_and_molar_basis(vapor):
    """Density default and molar basis.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    """
    expected_molar = 1 / MOLAR_VOLUME / 1000  # [mol/L], 1000 L/m**3
    assert vapor.getDensity() == pytest.approx(
        expected_molar * INITIAL_MW, rel=REL_TOL)  # [kg/m**3], g/L = kg/m**3
    assert vapor.getDensity(basis='mass') == pytest.approx(
        expected_molar * INITIAL_MW, rel=REL_TOL)
    assert vapor.getDensity(basis='mole') == pytest.approx(
        expected_molar, rel=REL_TOL)


@pytest.mark.parametrize('overrides, mw, temp, pres', [
    ({'temp': 600.0}, INITIAL_MW, 600.0, PRESSURE),  # [K], doubled temperature
    ({'pres': 2 * PRESSURE}, INITIAL_MW, TEMPERATURE, 2 * PRESSURE),  # [Pa]
    ({'mole_frac': UPDATED_FRACTIONS}, UPDATED_MW, TEMPERATURE, PRESSURE),
    ({'mass_frac': np.array([10.8, 40.0]) / UPDATED_MW},  # [-], species g/mol
     UPDATED_MW, TEMPERATURE, PRESSURE),
])
def test_density_overrides_do_not_mutate_state(vapor, overrides, mw, temp, pres):
    """Density overrides do not mutate state.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    overrides : dict
        Density or state overrides: fractions [-], temperature [K], pressure [Pa].
    mw : float
        Expected mixture molecular weight [g/mol].
    temp : float
        Expected temperature [K].
    pres : float
        Expected pressure [Pa].
    """
    expected = pres / (GAS_CONSTANT * temp) / 1000  # [mol/L], 1000 L/m**3
    assert vapor.getDensity(**overrides) == pytest.approx(expected * mw, rel=REL_TOL)
    assert vapor.getDensity(basis='mole', **overrides) == pytest.approx(
        expected, rel=REL_TOL)
    assert_amounts(vapor, ONE_MOLE)
    np.testing.assert_array_equal(vapor.mole_frac, INITIAL_FRACTIONS)
    assert vapor.temp == TEMPERATURE
    assert vapor.pres == PRESSURE


def test_density_rejects_unknown_basis(vapor):
    """Density rejects unknown basis.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    """
    with pytest.raises(ValueError, match="basis must be 'mass' or 'mole'"):
        vapor.getDensity(basis='volume')


@pytest.mark.parametrize('amount', ['mass', 'vol', 'moles'])
@pytest.mark.parametrize('stream', [False, True], ids=['phase', 'stream'])
def test_construction_equivalent_amount_bases(thermo_path, amount, stream):
    """Construction equivalent amount bases.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    amount : str
        Selected mass, volume, or molar amount argument.
    stream : bool
        Select a flow [per second] instead of an inventory.
    """
    inputs = {'mass': INITIAL_MW / 1000,  # [kg], 1000 g/kg
              'vol': MOLAR_VOLUME * ONE_MOLE,  # [m**3]
              'moles': ONE_MOLE}  # [mol]
    names = {'mass': 'mass_flow', 'vol': 'vol_flow', 'moles': 'mole_flow'}
    cls = VaporStream if stream else VaporPhase
    name = names[amount] if stream else amount
    phase = cls(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                mole_frac=INITIAL_FRACTIONS, **{name: inputs[amount]})
    assert_amounts(phase, ONE_MOLE)


@pytest.mark.parametrize('composition', ['mole_conc', 'mass_conc', 'mass_frac', 'mole_frac'])
@pytest.mark.parametrize('explicit_amount', [False, True])
def test_update_composition_branches(vapor, composition, explicit_amount):
    """Update composition branches.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    composition : str
        Selected concentration or fraction input name.
    explicit_amount : bool
        Supply an explicit amount instead of conserving moles.
    """
    inputs = {
        'mole_conc': np.array([3.0, 2.0]),  # [mol/L], ratio 0.6:0.4
        'mass_conc': np.array([54.0, 200.0]),  # [kg/m**3], c_i * MW_i
        'mass_frac': np.array([10.8, 40.0]) / UPDATED_MW,  # [-]
        'mole_frac': UPDATED_FRACTIONS,  # [-]
    }  # Units of each composition measure are given above.
    kwargs = {composition: inputs[composition]}
    if explicit_amount:
        kwargs['moles'] = TWO_MOLES  # [mol] or [mol/s]
    vapor.updatePhase(**kwargs)
    assert_amounts(vapor, TWO_MOLES if explicit_amount else ONE_MOLE, UPDATED_MW)
    np.testing.assert_allclose(vapor.mole_frac, UPDATED_FRACTIONS, rtol=REL_TOL)
    np.testing.assert_allclose(vapor.mass_frac, inputs['mass_frac'], rtol=REL_TOL)
    expected_conc = UPDATED_FRACTIONS / MOLAR_VOLUME / 1000  # [mol/L], gas EOS
    np.testing.assert_allclose(vapor.mole_conc, expected_conc, rtol=REL_TOL)
    np.testing.assert_allclose(vapor.mass_conc,
                               expected_conc * np.array([18.0, 100.0]),
                               rtol=REL_TOL)  # [kg/m**3], g/L = kg/m**3


@pytest.mark.parametrize('amount', ['mass', 'vol', 'moles'])
def test_amount_only_update(vapor, amount):
    """Amount only update.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    amount : str
        Selected mass, volume, or molar amount argument.
    """
    inputs = {'mass': TWO_MOLES * INITIAL_MW / 1000,  # [kg], 1000 g/kg
              'vol': TWO_MOLES * MOLAR_VOLUME,  # [m**3]
              'moles': TWO_MOLES}  # [mol]; per-second units for stream
    vapor.updatePhase(**{amount: inputs[amount]})
    assert_amounts(vapor, TWO_MOLES)


def test_no_argument_update_preserves_all_amounts(vapor):
    """No argument update preserves all amounts.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    """
    before = (vapor.mass, vapor.vol, vapor.moles)  # [kg, m**3, mol] or rates
    vapor.updatePhase()
    assert (vapor.mass, vapor.vol, vapor.moles) == before
    assert_amounts(vapor, ONE_MOLE)


@pytest.mark.parametrize('overrides, temp, pres', [
    ({'temp': 600.0}, 600.0, PRESSURE),  # [K], doubled temperature
    ({'pres': 2 * PRESSURE}, TEMPERATURE, 2 * PRESSURE),  # [Pa], doubled pressure
    ({'temp': 450.0, 'pres': 3 * PRESSURE}, 450.0, 3 * PRESSURE),  # [K, Pa]
])
def test_state_update_conserves_moles(vapor, overrides, temp, pres):
    """State update conserves moles.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    overrides : dict
        Density or state overrides: fractions [-], temperature [K], pressure [Pa].
    temp : float
        Expected temperature [K].
    pres : float
        Expected pressure [Pa].
    """
    vapor.updatePhase(**overrides)
    assert_amounts(vapor, ONE_MOLE, temp=temp, pres=pres)
    assert vapor.temp == temp
    assert vapor.pres == pres


def test_explicit_volume_uses_updated_composition_temperature_and_pressure(vapor):
    """Explicit volume uses updated composition temperature and pressure.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    """
    new_temp = 600.0  # [K], twice the initial temperature
    new_pres = 3 * PRESSURE  # [Pa], triple initial pressure
    vapor.updatePhase(vol=2 * MOLAR_VOLUME, mole_frac=UPDATED_FRACTIONS,
                      temp=new_temp, pres=new_pres)  # [m**3] or [m**3/s]
    # n scales as V P / T: 2 * 3 / 2 = 3 times the reference amount.
    assert_amounts(vapor, 3 * ONE_MOLE, UPDATED_MW, new_temp, new_pres)


@pytest.mark.parametrize('include_mass', [True, False], ids=['mass-first', 'volume-first'])
def test_first_positive_amount_precedence(vapor, thermo_path, include_mass):
    """First positive amount precedence.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    thermo_path : str
        Path to the synthetic thermophysical data.
    include_mass : bool
        Include positive mass to test its precedence over volume.
    """
    amounts = {'vol': 3 * MOLAR_VOLUME, 'moles': ONE_MOLE}  # [m**3, mol] or rates
    if include_mass:
        amounts['mass'] = TWO_MOLES * INITIAL_MW / 1000  # [kg] or [kg/s], 1000 g/kg
    expected = TWO_MOLES if include_mass else 3 * ONE_MOLE  # [mol] or [mol/s]
    vapor.updatePhase(**amounts)
    assert_amounts(vapor, expected)
    if isinstance(vapor, VaporStream):
        names = {'mass': 'mass_flow', 'vol': 'vol_flow', 'moles': 'mole_flow'}
        amounts = {names[key]: value for key, value in amounts.items()}  # flow rates
    constructed = type(vapor)(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                              mole_frac=INITIAL_FRACTIONS, **amounts)
    assert_amounts(constructed, expected)


@pytest.mark.parametrize('alias', ['mass_flow', 'vol_flow', 'mole_flow', 'moles'])
def test_stream_explicit_amount_overrides_retained_aliases(thermo_path, alias):
    """Stream explicit amount overrides retained aliases.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    alias : str
        Selected flow alias or phase-style molar argument.
    """
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS, mole_flow=ONE_MOLE)
    assert stream.mass_flow > 0
    amounts = {'mass_flow': TWO_MOLES * INITIAL_MW / 1000,  # [kg/s], 1000 g/kg
               'vol_flow': TWO_MOLES * MOLAR_VOLUME,  # [m**3/s]
               'mole_flow': TWO_MOLES, 'moles': TWO_MOLES}  # [mol/s]
    stream.updatePhase(**{alias: amounts[alias]})
    assert_amounts(stream, TWO_MOLES)


@pytest.mark.parametrize('phase_name, alias', [
    ('mass', 'mass_flow'), ('vol', 'vol_flow'), ('moles', 'mole_flow'),
])
@pytest.mark.parametrize('alias_value', [0.0, 3.0])  # [kg/s, m**3/s, or mol/s]
def test_stream_conflicting_alias_is_rejected_before_state_change(
        thermo_path, phase_name, alias, alias_value):
    """Stream conflicting alias is rejected before state change.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    phase_name : str
        Phase-style argument paired with the flow alias.
    alias : str
        Selected flow alias or phase-style molar argument.
    alias_value : float
        Conflicting flow value [kg/s], [m**3/s], or [mol/s], per alias.
    """
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS, mole_flow=ONE_MOLE)
    with pytest.raises(ValueError, match=f"'{phase_name}'.*'{alias}'"):
        stream.updatePhase(**{phase_name: 2.0, alias: alias_value,
                              'temp': 600.0})  # positive amount/rate; [K]
    assert_amounts(stream, ONE_MOLE)
    assert stream.temp == TEMPERATURE


def test_stream_zero_alias_and_explicit_phase_amount(thermo_path):
    """Stream zero alias and explicit phase amount.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    """
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS, mole_flow=ONE_MOLE)
    stream.updatePhase(mass_flow=0, moles=TWO_MOLES)
    assert_amounts(stream, TWO_MOLES)
    stream.updatePhase(mole_flow=0)
    assert_amounts(stream, TWO_MOLES)


def test_zero_composition_placeholder_and_evaporator_style_update(thermo_path):
    """Zero composition placeholder and evaporator style update.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    """
    with np.errstate(divide='raise', invalid='raise'):
        stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                             mole_frac=np.zeros(2), check_input=False, verbose=False)
        assert_amounts(stream, 0, mw=0)
        assert stream.getDensity() == 0
        stream.updatePhase()
        assert_amounts(stream, 0, mw=0)
        stream.updatePhase(moles=TWO_MOLES, mole_frac=UPDATED_FRACTIONS)
    assert_amounts(stream, TWO_MOLES, UPDATED_MW)
    np.testing.assert_array_equal(stream.mole_frac, UPDATED_FRACTIONS)


def test_zero_amount_construction_keeps_warning(thermo_path):
    """Zero amount construction keeps warning.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    """
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        phase = VaporPhase(thermo_path, mole_frac=INITIAL_FRACTIONS)
    assert_amounts(phase, 0)


def test_name_analyzer_converts_vapor_molar_flow_to_gas_volume(thermo_path):
    """Name analyzer converts vapor molar flow to gas volume.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    """
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS, mole_flow=TWO_MOLES)
    stream.y_upstream = {'mole_frac': INITIAL_FRACTIONS, 'mole_flow': TWO_MOLES}
    analyzer = NameAnalyzer(['mole_frac', 'mole_flow'], ['mole_frac', 'vol_flow'], 2)
    converted = analyzer.convertUnits(stream)
    assert converted['vol_flow'] == pytest.approx(
        TWO_MOLES * MOLAR_VOLUME, rel=REL_TOL)  # [m**3/s]
    np.testing.assert_array_equal(converted['mole_frac'], INITIAL_FRACTIONS)


def test_evaporator_initialization_preserves_moles_and_fractions(thermo_path):
    """Exercise the real initialization handoff without an ODE/DAE solve.

    Vapor enthalpy (#177/#178, PR #179) and intensive-state synchronization in
    the evaporator are outside https://github.com/PharmaPy-org/PharmaPy/issues/216; only its molar inventory and fractions
    are asserted here, not its energy initialization or vapor temperature.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    """
    from PharmaPy.Evaporators import ContinuousEvaporator

    liquid = LiquidPhase(thermo_path, moles=ONE_MOLE, mole_frac=INITIAL_FRACTIONS)
    inlet = LiquidStream(thermo_path, mole_flow=ONE_MOLE, mole_frac=INITIAL_FRACTIONS)
    headspace = MOLAR_VOLUME * ONE_MOLE  # [m**3], one reference gas-mole volume
    evaporator = ContinuousEvaporator(liquid.vol + headspace, adiabatic=True)
    evaporator.Phases = liquid
    evaporator.Inlet = inlet
    bubble_temp, vapor_fractions = liquid.getBubblePoint(pres=PRESSURE, y_vap=True)  # [K], [-]
    expected_moles = PRESSURE * headspace / (GAS_CONSTANT * bubble_temp)  # [mol]

    states, _ = evaporator.init_unit()  # state units and order defined in states_di

    assert evaporator.Vapor_1.moles == pytest.approx(expected_moles, rel=REL_TOL)
    assert evaporator.Vapor_1.mole_flow == pytest.approx(expected_moles, rel=REL_TOL)
    np.testing.assert_allclose(evaporator.Vapor_1.mole_frac, vapor_fractions, rtol=REL_TOL)
    np.testing.assert_allclose(states[4:6], vapor_fractions, rtol=REL_TOL)  # [-], y_vap
    assert states[7] == pytest.approx(expected_moles, rel=REL_TOL)  # [mol], mol_vap


def test_density_preserves_composition_row_and_species_order(vapor):
    """Density preserves composition row and species order.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference gas inventory [mol] or stream [mol/s] fixture.
    """
    fractions = np.array([INITIAL_FRACTIONS, UPDATED_FRACTIONS, [1.0, 0.0]])  # [-]
    expected_mw = np.array([INITIAL_MW, UPDATED_MW, 18.0])  # [g/mol], last row pure light
    expected = expected_mw / (1000 * MOLAR_VOLUME)  # [kg/m**3], 1000 g/kg
    density = vapor.getDensity(mole_frac=fractions)  # [kg/m**3], one value per row
    assert density.shape == (3,)
    np.testing.assert_allclose(density, expected, rtol=REL_TOL)
    order = [2, 0, 1]
    np.testing.assert_allclose(vapor.getDensity(mole_frac=fractions[order]),
                               expected[order], rtol=REL_TOL)


@pytest.mark.parametrize('amount', ['mass', 'vol', 'mass_flow', 'vol_flow'])
def test_zero_composition_rejects_positive_mass_or_volume(thermo_path, amount):
    """Reject mass and volume requests on an empty evaporator placeholder.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    amount : str
        Positive mass [kg/s] or volume [m**3/s] argument or alias to reject.
    """
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=np.zeros(2), check_input=False, verbose=False)
    with np.errstate(divide='raise', invalid='raise'):
        with pytest.raises(ValueError, match='zero mixture molar mass'):
            stream.updatePhase(**{amount: 1.0})  # [kg/s] or [m**3/s], positive probe
    assert_amounts(stream, 0, mw=0)
    stream.updatePhase(moles=ONE_MOLE)
    assert_amounts(stream, ONE_MOLE, mw=0)


@pytest.mark.parametrize('amount', ['mass', 'vol', 'moles'])
def test_negative_phase_amount_rejected_at_construction_and_update(thermo_path, vapor, amount):
    """Reject negative inventory inputs before updating stored state.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    vapor : VaporPhase or VaporStream
        Reference inventory or flow fixture.
    amount : str
        Negative mass [kg], volume [m**3], or molar [mol] amount to reject;
        the stream fixture uses corresponding per-second units.
    """
    invalid = {amount: -1.0}  # [kg, m**3, or mol], deliberately negative probe
    with pytest.raises(ValueError, match=f'{amount}.*nonnegative'):
        VaporPhase(thermo_path, mole_frac=INITIAL_FRACTIONS, **invalid)
    with pytest.raises(ValueError, match=f'{amount}.*nonnegative'):
        vapor.updatePhase(temp=2 * TEMPERATURE, **invalid)
    assert_amounts(vapor, ONE_MOLE)
    assert vapor.temp == TEMPERATURE


@pytest.mark.parametrize('amount, alias', [
    ('mass', 'mass_flow'), ('vol', 'vol_flow'), ('moles', 'mole_flow'),
])
def test_negative_stream_alias_and_shadowed_amount_rejected(thermo_path, amount, alias):
    """Reject negative aliases and phase amounts even when an alias replaces them.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical data.
    amount, alias : str
        Phase-style amount and its flow alias [kg/s], [m**3/s], or [mol/s].
    """
    invalid = {alias: -1.0}  # [kg/s, m**3/s, or mol/s], negative probe
    with pytest.raises(ValueError, match='nonnegative'):
        VaporStream(thermo_path, mole_frac=INITIAL_FRACTIONS, **invalid)
    stream = VaporStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_frac=INITIAL_FRACTIONS, mole_flow=ONE_MOLE)
    with pytest.raises(ValueError, match=f'{amount}.*nonnegative'):
        stream.updatePhase(**{amount: -1.0, alias: 1.0})  # opposing flow probes
    with pytest.raises(ValueError, match=f'{alias}.*nonnegative'):
        stream.updatePhase(temp=2 * TEMPERATURE, **invalid)
    assert_amounts(stream, ONE_MOLE)
    assert stream.temp == TEMPERATURE


@pytest.mark.parametrize('composition', ['mole_frac', 'mass_frac'])
def test_density_both_bases_return_one_value_per_composition_row(vapor, composition):
    """Keep the same two-row shape on mass and molar density bases.

    Parameters
    ----------
    vapor : VaporPhase or VaporStream
        Reference inventory or flow fixture.
    composition : str
        Species mole or mass fraction input name [-].
    """
    fractions = np.array([INITIAL_FRACTIONS, UPDATED_FRACTIONS])  # [-], (2, 2)
    expected_mw = np.array([INITIAL_MW, UPDATED_MW])  # [g/mol]
    if composition == 'mass_frac':
        fractions = fractions * np.array([18.0, 100.0]) / expected_mw[:, None]  # [-]
    expected_molar = np.full(2, 1 / MOLAR_VOLUME / 1000)  # [mol/L], 1000 L/m**3
    for basis in ('mass', 'mole'):
        density = vapor.getDensity(**{composition: fractions}, basis=basis)  # [kg/m**3] or [mol/L]
        expected = expected_molar * expected_mw if basis == 'mass' else expected_molar  # [kg/m**3] or [mol/L]
        assert np.shape(density) == (2,)
        np.testing.assert_allclose(density, expected, rtol=REL_TOL)


@pytest.mark.parametrize('amount', [0.0, ONE_MOLE])
@pytest.mark.parametrize('composition', ['mole_frac', 'mass_frac', 'mole_conc', 'mass_conc'])
def test_vapor_concentrations_obey_gas_eos(thermo_path, amount, composition):
    # One mole of the asymmetric binary has species amounts [0.25, 0.75] mol.
    inputs = {'mole_frac': INITIAL_FRACTIONS,
              'mass_frac': INITIAL_FRACTIONS * np.array([18., 100.]) / INITIAL_MW,
              'mole_conc': INITIAL_FRACTIONS / MOLAR_VOLUME / 1000,
              'mass_conc': INITIAL_FRACTIONS * np.array([18., 100.]) / MOLAR_VOLUME / 1000}
    # Fractions [-], molar concentrations [mol/L], mass concentrations [kg/m**3].
    vapor = VaporPhase(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                       moles=amount, check_input=False,
                       **{composition: inputs[composition]} if composition != 'mass_conc'
                       else {'mole_frac': INITIAL_FRACTIONS})
    if composition == 'mass_conc':
        vapor.updatePhase(mass_conc=inputs[composition])
    for temperature, pressure in [(TEMPERATURE, PRESSURE),
                                  (2 * TEMPERATURE, PRESSURE),
                                  (2 * TEMPERATURE, 3 * PRESSURE)]:
        # [K], [Pa], independent temperature and pressure changes.
        vapor.updatePhase(temp=temperature, pres=pressure)
        expected = INITIAL_FRACTIONS * pressure / (GAS_CONSTANT * temperature) / 1000  # [mol/L]
        np.testing.assert_allclose(vapor.mole_conc, expected, rtol=REL_TOL)
        np.testing.assert_allclose(vapor.mass_conc, expected * [18., 100.], rtol=REL_TOL)
        if amount:
            assert vapor.mole_conc.sum() == pytest.approx(vapor.moles / vapor.vol / 1000, rel=REL_TOL)
            assert vapor.mass_conc.sum() == pytest.approx(vapor.mass / vapor.vol, rel=REL_TOL)
