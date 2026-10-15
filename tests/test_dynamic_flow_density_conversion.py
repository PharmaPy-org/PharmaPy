"""Per-sample density in dynamic flow-basis conversion (#219).

``NameAnalyzer.convertUnits`` turns upstream volume flows [m**3/s] into mass
[kg/s] or molar [mol/s] flows, and molar flows back into volume flows. Each
dynamic sample must use the density of its own composition and temperature,
while static (scalar) sources keep the stored-state result.

Liquid cases use the shipped PFR test database, whose species A (1230 kg/m**3,
105.52 g/mol) and B (864.7 kg/m**3, 99.1741 g/mol) have unequal densities.
Expected values come from the database JSON with an independently coded
ideal-mixing rule, 1/rho = sum(w_i/rho_i), and pure rows are also checked
against the literal database densities. The public handoff runs real CSTR
profile retrieval into ``Connection.transfer_data`` with Mixer and
ContinuousEvaporator destinations; no ODE solver runs. Vapor cases use the
established two-species synthetic database and ideal-gas law, the supported
temperature-dependent density. Dynamic profiles carried by a real
``SlurryStream`` (per-phase densities) and unknown composition bases must
raise explicit errors instead of returning unphysical flows.

On the milestone base (71f918a) the ``_guard`` tests, which protect the
static result, and two parametrized cases of
``test_dynamic_liquid_mass_mole_flow_uses_sample_molar_mass``
(``mass_frac-mass_flow-mole_flow`` and ``mass_frac-mole_flow-mass_flow``,
whose base conversion already used each sample's molar mass) pass; every
other test fails there. Stopped samples (exactly zero composition) convert
to zero flow; ``test_non_finite_sample_still_propagates`` pins that a NaN
sample still yields NaN.

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/219
(coordination: https://github.com/PharmaPy-org/PharmaPy/issues/260)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy import Reactors
from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.Evaporators import ContinuousEvaporator
from PharmaPy.NameAnalysis import NameAnalyzer
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Streams import LiquidStream, SolidStream, VaporStream
from conftest import THERMO_TWO_SPECIES
from test_reactor_correctness import (
    CONCENTRATIONS, REACTOR_VOLUME, RESIDENCE_TIME, TEMPERATURE, THERMO_PATH,
    _configured_reactor)

RTOL = 1e-12  # [-], float64 round-off of short ideal-mixing sums

with open(THERMO_PATH) as handle:
    _PROPERTIES = json.load(handle)
MW = np.array([value['mw'] for value in _PROPERTIES.values()])  # [g/mol], A, B, C, solv
RHO_LIQ = np.array([value['rho_liq'] for value in _PROPERTIES.values()])  # [kg/m**3]
RHO_A, RHO_B = 1230.0, 864.7  # [kg/m**3], literal database entries for A and B

# [-], asymmetric rows: pure A, pure B, then an A/C/solvent mixture.
MOLE_FRAC_ROWS = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0.2, 0., 0.1, 0.7]])
TOTAL_CONC = np.array([8., 9., 5.])  # [mol/L], distinct per-row totals
TOTAL_MASS_CONC = np.array([800., 900., 700.])  # [kg/m**3], distinct row totals
VOL_FLOWS = np.array([2e-3, 1e-3, 3e-3])  # [m**3/s], unequal samples
PROFILE_TEMPS = np.array([320., 330., 340.])  # [K]

GAS_CONSTANT = 8.314  # [J/mol/K], the model's documented rounded value
GAS_PRESSURE = 101325.0  # [Pa], stored vapor pressure, one atmosphere
GAS_TEMPS = np.array([300., 350., 400.])  # [K], distinct sample temperatures
GAS_FLOW = 1.0  # [mol/s], stored stream flow; conversions use profile samples
GAS_MOLE_FRAC = np.array([[0.25, 0.75], [0.6, 0.4], [0.9, 0.1]])  # [-]
GAS_MW = np.array([THERMO_TWO_SPECIES[name]['mw']
                   for name in THERMO_TWO_SPECIES])  # [g/mol], light, heavy


def ideal_mixing(mole_frac):
    """Evaluate mixture properties from the database by hand.

    Parameters
    ----------
    mole_frac : numpy.ndarray
        Mole fractions [-], shape (num_times, num_species), database order.

    Returns
    -------
    tuple of numpy.ndarray
        Mass fractions [-] of the same shape, average molar mass [g/mol] and
        ideal-mixing mass density [kg/m**3], each of shape (num_times,).
    """
    mw_av = mole_frac @ MW  # [g/mol]
    mass_frac = mole_frac * MW / mw_av[:, np.newaxis]  # [-]
    density = 1 / (mass_frac @ (1 / RHO_LIQ))  # [kg/m**3]
    return mass_frac, mw_av, density


def expected_flow(down, vol_flow, mw_av, density):
    """Convert volume flow with independently computed properties.

    Parameters
    ----------
    down : {'mass_flow', 'mole_flow'}
        Requested downstream basis.
    vol_flow : numpy.ndarray
        Volume flows [m**3/s], shape (num_times,).
    mw_av : numpy.ndarray
        Average molar masses [g/mol], shape (num_times,).
    density : numpy.ndarray
        Mass densities [kg/m**3], shape (num_times,).

    Returns
    -------
    numpy.ndarray
        Mass flow [kg/s] or molar flow [mol/s], with 1000 g/kg.
    """
    mass_flow = vol_flow * density  # [kg/s]
    if down == 'mass_flow':
        return mass_flow
    return mass_flow * 1000 / mw_av  # [mol/s]


def upstream_composition(name, mole_frac):
    """Express composition rows in the requested upstream state.

    Parameters
    ----------
    name : {'mole_conc', 'mass_conc', 'mole_frac', 'mass_frac'}
        Upstream composition state.
    mole_frac : numpy.ndarray
        Mole fractions [-], shape (num_times, num_species).

    Returns
    -------
    numpy.ndarray
        Molar concentrations [mol/L] scaled by ``TOTAL_CONC``, mass
        concentrations [kg/m**3] scaled by ``TOTAL_MASS_CONC``, or
        fractions [-].
    """
    if name == 'mole_conc':
        return mole_frac * TOTAL_CONC[:len(mole_frac), np.newaxis]  # [mol/L]
    if name == 'mass_conc':
        mass_frac = ideal_mixing(mole_frac)[0]  # [-]
        return mass_frac * TOTAL_MASS_CONC[:len(mole_frac), np.newaxis]  # [kg/m**3]
    if name == 'mass_frac':
        return ideal_mixing(mole_frac)[0]  # [-]
    return mole_frac


def liquid_stream():
    """Build a real stream with the reactor fixture's stored composition.

    Returns
    -------
    PharmaPy.Streams.LiquidStream
        Stream at ``TEMPERATURE`` [K] with ``CONCENTRATIONS`` [mol/L].
    """
    return LiquidStream(THERMO_PATH, temp=TEMPERATURE, mole_conc=CONCENTRATIONS,
                        vol_flow=VOL_FLOWS[0])


@pytest.fixture
def gas_path(tmp_path):
    """Write the established two-species synthetic database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory for the JSON file.

    Returns
    -------
    str
        Path to the database.
    """
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(THERMO_TWO_SPECIES))
    return str(path)


@pytest.mark.unit
def test_pure_rows_use_their_own_database_density():
    stream = liquid_stream()
    stream.y_upstream = {'mole_frac': MOLE_FRAC_ROWS[:2], 'vol_flow': VOL_FLOWS[:2],
                         'temp': PROFILE_TEMPS[:2]}
    analyzer = NameAnalyzer(['mole_frac', 'vol_flow', 'temp'],
                            ['mass_frac', 'mass_flow', 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    np.testing.assert_allclose(converted['mass_flow'],
                               VOL_FLOWS[:2] * [RHO_A, RHO_B], rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('down', ['mass_flow', 'mole_flow'])
@pytest.mark.parametrize('comp_name', ['mole_conc', 'mass_conc', 'mole_frac', 'mass_frac'])
def test_dynamic_volume_flow_uses_each_sample_density(comp_name, down, reverse):
    order = slice(None, None, -1) if reverse else slice(None)
    mole_frac = MOLE_FRAC_ROWS[order]  # [-]
    vol_flow = VOL_FLOWS[order]  # [m**3/s]
    stream = liquid_stream()
    stream.y_upstream = {comp_name: upstream_composition(comp_name, mole_frac),
                         'vol_flow': vol_flow, 'temp': PROFILE_TEMPS[order]}
    down_comp = 'mole_frac' if down == 'mole_flow' else 'mass_frac'
    analyzer = NameAnalyzer([comp_name, 'vol_flow', 'temp'],
                            [down_comp, down, 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    mass_frac, mw_av, density = ideal_mixing(mole_frac)
    assert converted[down].shape == (len(vol_flow),)
    np.testing.assert_allclose(converted[down],
                               expected_flow(down, vol_flow, mw_av, density), rtol=RTOL)
    expected_comp = mass_frac if down_comp == 'mass_frac' else mole_frac  # [-]
    assert converted[down_comp].shape == expected_comp.shape
    np.testing.assert_allclose(converted[down_comp], expected_comp, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(converted['temp'], PROFILE_TEMPS[order])


@pytest.mark.unit
@pytest.mark.parametrize('down', ['mass_flow', 'mole_flow'])
def test_static_volume_flow_keeps_stored_state_density_guard(down):
    stream = liquid_stream()
    stream.y_upstream = {'mole_conc': CONCENTRATIONS, 'vol_flow': VOL_FLOWS[0],
                         'temp': TEMPERATURE}
    down_comp = 'mole_frac' if down == 'mole_flow' else 'mass_frac'
    analyzer = NameAnalyzer(['mole_conc', 'vol_flow', 'temp'],
                            [down_comp, down, 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    stored = (CONCENTRATIONS / CONCENTRATIONS.sum())[np.newaxis]  # [-]
    _, mw_av, density = ideal_mixing(stored)
    expected = expected_flow(down, VOL_FLOWS[:1], mw_av, density)[0]  # [kg/s] or [mol/s], per down
    assert np.ndim(converted[down]) == 0
    assert converted[down] == pytest.approx(expected, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('destination', ['mixer', 'evaporator'])
def test_connection_converts_cstr_volume_flow_profile(destination):
    reactor = _configured_reactor(Reactors.CSTR(isothermal=True))
    reactor.set_names()
    times = np.array([0., 1., 2.])  # [s], synthetic profile, no solve
    mole_conc = upstream_composition('mole_conc', MOLE_FRAC_ROWS)  # [mol/L]
    reactor.retrieve_results(times, mole_conc)
    if destination == 'mixer':
        unit, down, comp = Mixer(), 'mass_flow', 'mass_frac'
    else:
        # Volume [m**3] is arbitrary and positive; conversion does not use it.
        unit = ContinuousEvaporator(REACTOR_VOLUME, adiabatic=True)
        down, comp = 'mole_flow', 'mole_frac'
    connection = Connection(reactor, unit)
    connection.transfer_data()
    converted = connection.Matter.y_inlet
    vol_flow = np.full(len(times), REACTOR_VOLUME / RESIDENCE_TIME)  # [m**3/s], held inlet
    mass_frac, mw_av, density = ideal_mixing(MOLE_FRAC_ROWS)
    np.testing.assert_allclose(converted[down],
                               expected_flow(down, vol_flow, mw_av, density), rtol=RTOL)
    expected_comp = mass_frac if comp == 'mass_frac' else MOLE_FRAC_ROWS  # [-]
    np.testing.assert_allclose(converted[comp], expected_comp, rtol=RTOL, atol=0)
    np.testing.assert_allclose(converted['temp'], TEMPERATURE, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('down', ['mass_flow', 'mole_flow'])
def test_dynamic_gas_volume_flow_uses_sample_temperature(gas_path, down):
    stream = VaporStream(gas_path, temp=GAS_TEMPS[0], pres=GAS_PRESSURE,
                         mole_frac=GAS_MOLE_FRAC[0], mole_flow=GAS_FLOW)
    stream.y_upstream = {'mole_frac': GAS_MOLE_FRAC, 'vol_flow': VOL_FLOWS,
                         'temp': GAS_TEMPS}
    analyzer = NameAnalyzer(['mole_frac', 'vol_flow', 'temp'],
                            ['mole_frac', down, 'temp'], len(GAS_MW))
    converted = analyzer.convertUnits(stream)
    mole_flow = VOL_FLOWS * GAS_PRESSURE / (GAS_CONSTANT * GAS_TEMPS)  # [mol/s], PV = nRT
    mw_av = GAS_MOLE_FRAC @ GAS_MW  # [g/mol]
    expected = mole_flow if down == 'mole_flow' else mole_flow * mw_av / 1000  # [mol/s] or [kg/s], per down
    np.testing.assert_allclose(converted[down], expected, rtol=RTOL)


@pytest.mark.unit
def test_dynamic_gas_molar_flow_to_volume_uses_sample_temperature(gas_path):
    mole_flow = np.array([1., 3., 2.])  # [mol/s], unequal samples
    stream = VaporStream(gas_path, temp=GAS_TEMPS[0], pres=GAS_PRESSURE,
                         mole_frac=GAS_MOLE_FRAC[0], mole_flow=mole_flow[0])
    stream.y_upstream = {'mole_frac': GAS_MOLE_FRAC, 'mole_flow': mole_flow,
                         'temp': GAS_TEMPS}
    analyzer = NameAnalyzer(['mole_frac', 'mole_flow', 'temp'],
                            ['mole_frac', 'vol_flow', 'temp'], len(GAS_MW))
    converted = analyzer.convertUnits(stream)
    expected = mole_flow * GAS_CONSTANT * GAS_TEMPS / GAS_PRESSURE  # [m**3/s], V = nRT/P
    np.testing.assert_allclose(converted['vol_flow'], expected, rtol=RTOL)


@pytest.mark.unit
def test_dynamic_gas_mass_flow_to_volume_uses_sample_temperature(gas_path):
    mass_flow = np.array([0.05, 0.2, 0.1])  # [kg/s], unequal samples
    stream = VaporStream(gas_path, temp=GAS_TEMPS[0], pres=GAS_PRESSURE,
                         mole_frac=GAS_MOLE_FRAC[0], mole_flow=GAS_FLOW)
    stream.y_upstream = {'mole_frac': GAS_MOLE_FRAC, 'mass_flow': mass_flow,
                         'temp': GAS_TEMPS}
    analyzer = NameAnalyzer(['mole_frac', 'mass_flow', 'temp'],
                            ['mole_frac', 'vol_flow', 'temp'], len(GAS_MW))
    converted = analyzer.convertUnits(stream)
    mole_flow = mass_flow * 1000 / (GAS_MOLE_FRAC @ GAS_MW)  # [mol/s], 1000 g/kg
    expected = mole_flow * GAS_CONSTANT * GAS_TEMPS / GAS_PRESSURE  # [m**3/s], V = nRT/P
    np.testing.assert_allclose(converted['vol_flow'], expected, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('up_flow, down_flow', [('mass_flow', 'mole_flow'),
                                                ('mole_flow', 'mass_flow')])
@pytest.mark.parametrize('comp_name', ['mass_frac', 'mass_conc'])
def test_dynamic_liquid_mass_mole_flow_uses_sample_molar_mass(comp_name, up_flow,
                                                              down_flow):
    flow = np.array([2., 1., 3.])  # [kg/s] or [mol/s], per up_flow
    stream = liquid_stream()
    stream.y_upstream = {comp_name: upstream_composition(comp_name, MOLE_FRAC_ROWS),
                         up_flow: flow, 'temp': PROFILE_TEMPS}
    down_comp = 'mole_frac' if down_flow == 'mole_flow' else 'mass_frac'
    analyzer = NameAnalyzer([comp_name, up_flow, 'temp'],
                            [down_comp, down_flow, 'temp'], len(MW))
    converted = analyzer.convertUnits(stream)
    mass_frac, mw_av, _ = ideal_mixing(MOLE_FRAC_ROWS)
    if down_flow == 'mole_flow':
        expected = flow * 1000 / mw_av  # [mol/s], 1000 g/kg
    else:
        expected = flow * mw_av / 1000  # [kg/s]
    np.testing.assert_allclose(converted[down_flow], expected, rtol=RTOL)
    expected_comp = mass_frac if down_comp == 'mass_frac' else MOLE_FRAC_ROWS  # [-]
    np.testing.assert_allclose(converted[down_comp], expected_comp, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_mixer_consumes_converted_mass_conc_profile():
    times = np.array([0., 1., 2.])  # [s]
    mass_conc = upstream_composition('mass_conc', MOLE_FRAC_ROWS)  # [kg/m**3]
    stream = liquid_stream()
    stream.y_upstream = {'mass_conc': mass_conc, 'vol_flow': VOL_FLOWS,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['mass_conc', 'vol_flow', 'temp'],
                            ['mass_frac', 'mass_flow', 'temp'], len(MW))
    # The same conversion and stream fields Connection sets for a downstream unit.
    stream.y_inlet = analyzer.convertUnits(stream)
    stream.time_upstream = times
    mixer = Mixer()
    mixer.Inlets = [stream]
    mass_flow, mass_frac, temp = mixer.solve_unit()
    row_mass_frac = mass_conc / mass_conc.sum(axis=1, keepdims=True)  # [-]
    density = 1 / (row_mass_frac @ (1 / RHO_LIQ))  # [kg/m**3], ideal mixing
    np.testing.assert_allclose(mixer.result.time, times, rtol=RTOL)
    np.testing.assert_allclose(mass_flow, VOL_FLOWS * density, rtol=RTOL)
    np.testing.assert_allclose(mass_frac, row_mass_frac, rtol=RTOL, atol=0)
    np.testing.assert_allclose(temp, PROFILE_TEMPS, rtol=RTOL)


@pytest.mark.unit
def test_unknown_dynamic_composition_basis_raises():
    stream = liquid_stream()
    stream.y_upstream = {'vol_frac': MOLE_FRAC_ROWS, 'vol_flow': VOL_FLOWS,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['vol_frac', 'vol_flow', 'temp'],
                            ['vol_frac', 'mass_flow', 'temp'], len(MW))
    with pytest.raises(ValueError, match=r"'vol_frac' composition basis of LiquidStream"):
        analyzer.convertUnits(stream)


@pytest.mark.unit
def test_dynamic_slurry_volume_flow_conversion_raises():
    """Reject a dynamic slurry volume-flow conversion explicitly.

    Provisional expectation: per-phase dynamic slurry flow conversion (a total
    slurry ``vol_flow`` [m**3/s] with liquid ``mass_conc`` [kg/m**3] needs a
    per-phase split) is unsupported and will be tracked as a follow-up from
    the #219 pull request. Once supported, this test must assert converted
    values instead of the error.
    """
    flowsheet_db = str(Path(__file__).resolve().parent
                       / 'Flowsheet/data/compound_database.json')
    liquid_frac = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], A, B, C, D, solvent
    solid_frac = [1., 0., 0., 0., 0.]  # [-], pure A crystals
    x_distrib = np.array([0., 100., 200., 300.])  # [um]
    distrib = np.array([0., 1e9, 1.25e8, 0.])  # [#/m**3/um], 0.2 m**3/m**3 solids
    slurry = SlurryStream(vol_flow=VOL_FLOWS[0], x_distrib=x_distrib, distrib=distrib)
    # Phase flows are zero until the slurry setter derives them from the
    # total volume flow [m**3/s]; the constructors warn about that.
    with pytest.warns(RuntimeWarning, match='all set to zero'):
        phases = (LiquidStream(flowsheet_db, mass_frac=liquid_frac, temp=TEMPERATURE),
                  SolidStream(flowsheet_db, mass_frac=solid_frac, kv=1.0,
                              temp=TEMPERATURE))  # kv [-], cube shape factor
    slurry.Phases = phases
    mass_conc = np.outer(TOTAL_MASS_CONC, liquid_frac)  # [kg/m**3], (3, 5)
    slurry.y_upstream = {'mass_conc': mass_conc, 'vol_flow': VOL_FLOWS,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['mass_conc', 'vol_flow', 'temp'],
                            ['mass_frac', 'mass_flow', 'temp'], len(liquid_frac))
    with pytest.raises(NotImplementedError,
                       match=r"'vol_flow' to 'mass_flow'.*'mass_conc'.*SlurryStream"):
        analyzer.convertUnits(slurry)


@pytest.mark.integration
@pytest.mark.assimulo
def test_connection_from_solved_msmpr_to_mixer_raises(data_path):
    """Reject the real MSMPR slurry profile handoff to a liquid mixer.

    MSMPR publishes ``vol_flow`` [m**3/s] and liquid ``mass_conc``
    [kg/m**3] carried by a SlurryStream, whose densities are per phase.
    Provisional expectation: per-phase dynamic slurry flow conversion (the
    total slurry flow needs a per-phase split) is unsupported and will be
    tracked as a follow-up from the #219 pull request. Once supported, this
    test must assert converted values instead of the error.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic database paths.
    """
    pytest.importorskip('assimulo')
    from test_crystallizer_moment_inventory import DURATION, inventory_unit
    from PharmaPy.Crystallizers import MSMPR
    unit, _ = inventory_unit(data_path, MSMPR, gridless=True)
    unit.solve_unit(time_grid=[0.0, DURATION], verbose=False)
    with pytest.raises(NotImplementedError, match=r"'mass_conc'.*SlurryStream"):
        Connection(unit, Mixer()).transfer_data()


STOPPED_ROW = 1  # [-], index of the sample whose feed is stopped


@pytest.mark.unit
@pytest.mark.parametrize('down', ['mass_flow', 'mole_flow'])
def test_stopped_sample_converts_to_zero_flow(down):
    mass_frac, mw_av, density = ideal_mixing(MOLE_FRAC_ROWS)
    mass_frac = mass_frac.copy()  # [-]
    mass_frac[STOPPED_ROW] = 0
    vol_flow = VOL_FLOWS.copy()  # [m**3/s]
    vol_flow[STOPPED_ROW] = 0
    stream = liquid_stream()
    stream.y_upstream = {'mass_frac': mass_frac, 'vol_flow': vol_flow,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['mass_frac', 'vol_flow', 'temp'],
                            ['mass_frac', down, 'temp'], len(MW))

    converted = analyzer.convertUnits(stream)

    expected = expected_flow(down, vol_flow, mw_av, density)  # [kg/s] or [mol/s]
    assert expected[STOPPED_ROW] == 0
    np.testing.assert_allclose(converted[down], expected, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_non_finite_sample_still_propagates():
    mass_frac = ideal_mixing(MOLE_FRAC_ROWS)[0].copy()  # [-]
    mass_frac[STOPPED_ROW] = np.nan
    stream = liquid_stream()
    stream.y_upstream = {'mass_frac': mass_frac, 'vol_flow': VOL_FLOWS,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['mass_frac', 'vol_flow', 'temp'],
                            ['mass_frac', 'mass_flow', 'temp'], len(MW))

    converted = analyzer.convertUnits(stream)

    assert np.isnan(converted['mass_flow'][STOPPED_ROW])
    assert np.isfinite(np.delete(converted['mass_flow'], STOPPED_ROW)).all()


@pytest.mark.unit
def test_mixer_consumes_profile_with_stopped_sample():
    times = np.array([0., 1., 2.])  # [s]
    mass_frac, _, density = ideal_mixing(MOLE_FRAC_ROWS)
    mass_frac = mass_frac.copy()  # [-]
    mass_frac[STOPPED_ROW] = 0
    vol_flow = VOL_FLOWS.copy()  # [m**3/s]
    vol_flow[STOPPED_ROW] = 0
    stream = liquid_stream()
    stream.y_upstream = {'mass_frac': mass_frac, 'vol_flow': vol_flow,
                         'temp': PROFILE_TEMPS}
    analyzer = NameAnalyzer(['mass_frac', 'vol_flow', 'temp'],
                            ['mass_frac', 'mass_flow', 'temp'], len(MW))
    # The same conversion and stream fields Connection sets for a downstream unit.
    stream.y_inlet = analyzer.convertUnits(stream)
    stream.time_upstream = times
    mixer = Mixer()
    mixer.Inlets = [stream]

    mass_flow, _, _ = mixer.solve_unit()

    np.testing.assert_allclose(mass_flow, vol_flow * density, rtol=RTOL, atol=0)
