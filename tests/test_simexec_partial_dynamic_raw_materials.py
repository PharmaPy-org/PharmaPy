"""Raw-material accounting of the feed a unit consumes from a DynamicInlet (#260).

``SimulationExec.GetRawMaterials`` integrates a continuous raw inlet whose
``DynamicInlet`` controls only some fields. At every result time the accounted
feed is the one the receiving unit consumes: the controlled fields plus the
static stream values of the other fields that the unit declares. Units hold
different uncontrolled fields, so the same composition control is accounted
differently by a ``DynamicCollector`` (mass flow), an ``Evaporator`` (molar
flow) and a ``PlugFlowReactor`` (volume flow with molar concentrations).

Fixtures are real streams from the four-species ``pfr_test_pure_comp.json``
database (A, B, C, solv) attached to real, unsolved units whose
``DynamicResult`` holds a non-uniform time grid; the core tests run no solver.
Expectations use molar masses and pure-component densities read from the JSON
database and hand-written conversions, not PharmaPy's conversion routines.
Linear control laws are integrated in closed form (trapezoids are exact for
them); products of time-varying fields use ``scipy.integrate.trapezoid`` on
the independently evaluated law. Slurry cases use the two-phase fixture (solid
volume fraction 0.25, also rebuilt here with reversed phases, a hotter solid,
or a database copy whose solid densities differ from the liquid ones) and the
FVM MSMPR fixture of ``test_crystallizer_heat_duty``, which also receives a
bare liquid feed. Further fixtures: a one-species database copy, a
``ContinuousHoldup`` and a ``ContinuousExtractor``. Some control callables
accept only scalar times, as the solver supplies. The Assimulo test solves a
real collector inside a flowsheet and checks the raw row of
``GetStreamTable``.

On the milestone base (71f918a), tests ending in ``_guard`` pass, as do the
``mass_flow``-only collector cases (established usage), the zero-flow cases
(which pin the zero-total fallback of the new averages) and both cases of
``test_mixer_feed_is_evaluated_on_its_whole_grid`` (the base also evaluated
controls on the Mixer's whole grid). Every other test fails there, with ``KeyError`` for uncontrolled flows, with totals that ignore
the controlled composition or the unit's held flow, or without the expected
``ValueError``.

Related issue: https://github.com/PharmaPy-org/PharmaPy/issues/260
"""

import json

import numpy as np
import pytest
from scipy.integrate import trapezoid

from PharmaPy.Commons import trapezoidal_rule
from PharmaPy.Containers import ContinuousHoldup, DynamicCollector, Mixer
from PharmaPy.Crystallizers import MSMPR
from PharmaPy.Evaporators import Evaporator
from PharmaPy.Extractors import ContinuousExtractor
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Reactors import PlugFlowReactor
from PharmaPy.Results import DynamicResult
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream, SolidStream
from test_crystallizer_heat_duty import make_unit as make_heat_unit
from test_simexec_raw_materials import (_liquid_stream, _sim_with_inlet,
                                        _sim_with_units, _slurry_stream,
                                        _thermo_path)

RTOL = 1e-12  # [-], round-off allowance for algebraic conversions
# [s], non-uniform result grid with unequal intervals; six samples, 10 s run
TIME_GRID = (0.0, 1.0, 3.0, 4.0, 6.0, 10.0)
DURATION = TIME_GRID[-1] - TIME_GRID[0]  # [s]
SPECIES = ('A', 'B', 'C', 'solv')  # database order
STATIC_MASS_FLOW = 2.0  # [kg/s], uncontrolled liquid feed
# [-], static liquid feed, species order A, B, C, solv; all nonzero, unequal
STATIC_FRACTIONS = np.array([0.4, 0.3, 0.2, 0.1])
# [-], end-of-run composition reached by linear composition controls; unequal
# and not a permutation of STATIC_FRACTIONS, so species order stays visible
FINAL_FRACTIONS = np.array([0.15, 0.05, 0.5, 0.3])
STATIC_TEMPERATURE = 300.0  # [K], _liquid_stream fixture temperature
STATIC_PRESSURE = 101325.0  # [Pa], _liquid_stream fixture pressure
TEMPERATURE_SLOPE = 5.0  # [K/s], controlled temperature ramp
# Linear flow controls: value at t = 0 and slope, in each field's units.
FLOW_LAWS = {'mass_flow': (1.5, 0.25),  # [kg/s], [kg/s**2]
             'vol_flow': (1.5e-3, 2.0e-4)}  # [m**3/s], [m**3/s**2]
# [mol/L], start and end of a linear molar-concentration control, unequal
CONC_START = np.array([3.0, 2.5, 1.0, 1.5])
CONC_END = np.array([1.0, 2.0, 2.5, 3.5])
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact
LITERS_PER_CUBIC_METER = 1000.0  # [L/m**3], exact
SLURRY_SOLID_FRACTION = 0.25  # [-], _slurry_stream target kv * mu_3
# Phases of test_simexec_raw_materials._slurry_stream, species A, B, C, solv
SLURRY_LIQUID_FRACTIONS = np.array([0.8, 0.2, 0.0, 0.0])  # [-]
SLURRY_CRYSTAL_FRACTIONS = np.array([0.1, 0.9, 0.0, 0.0])  # [-]
SLURRY_LIQUID_FLOW = 1.0  # [kg/s], liquid mass flow before slurry closure
SLURRY_CRYSTAL_FLOW = 3.0  # [kg/s], crystal mass flow before slurry closure
SLURRY_FLOW = 4.0e-3  # [m**3/s], controlled slurry volume flow
POPULATION_SCALE = 0.4  # [-], controlled population relative to the static one
SHUTDOWN_TIME = 5.0  # [s], a stopped feed supplies no flow or composition after
FEED_RAMP_DOWN = 0.4  # [kg/s**2], linear ramp of the stopping feed
SOLID_TEMPERATURE = 350.0  # [K], crystal phase stored hotter than the liquid
# [kg/m**3], crystal densities of A and B for a copy of the database whose
# solid densities differ from the liquid ones (1230 and 864.7 kg/m**3)
ASYMMETRIC_SOLID_DENSITY = {'A': 1500.0, 'B': 1100.0}
HOLDUP_MASS = 5.0  # [kg], ContinuousHoldup liquid inventory
# Unit geometry required by the constructors only; raw accounting ignores it.
EVAPORATOR_DRUM_VOLUME = 1.0  # [m**3]
PFR_DIAMETER = 0.01  # [m]
PFR_NODES = 5  # [-]


def database_properties(data_path):
    """Read molar masses and pure-liquid densities from the JSON database.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.

    Returns
    -------
    molar_mass : numpy.ndarray
        Molar masses [g/mol], shape (4,), database species order.
    liquid_density : numpy.ndarray
        Constant pure-liquid densities [kg/m**3], shape (4,).
    """
    with open(_thermo_path(data_path)) as file:
        database = json.load(file)
    molar_mass = np.array([item['mw'] for item in database.values()])
    liquid_density = np.array([item['rho_liq'] for item in database.values()])
    return molar_mass, liquid_density


def mixture_density(mass_frac, liquid_density):
    """Ideal-mixing liquid density.

    Parameters
    ----------
    mass_frac : numpy.ndarray
        Mass fractions [-], shape (..., num_species).
    liquid_density : numpy.ndarray
        Pure-liquid densities [kg/m**3], shape (num_species,).

    Returns
    -------
    float or numpy.ndarray
        Mixture density [kg/m**3], shape of ``mass_frac`` without its last
        axis.
    """
    return 1 / np.sum(mass_frac / liquid_density, axis=-1)


def linear_integral(start, slope):
    """Integrate ``start + slope * t`` over the run in closed form.

    Parameters
    ----------
    start, slope : float or numpy.ndarray
        Value at the first time and its rate of change per second.

    Returns
    -------
    float or numpy.ndarray
        Integral over ``TIME_GRID``, in the value's units times [s].
    """
    return start * DURATION + slope * DURATION**2 / 2


def fraction_law(time):
    """Evaluate the linear composition blend at the given times.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    numpy.ndarray
        Fractions [-], shape (4,) or (num_times, 4), moving from
        ``STATIC_FRACTIONS`` at t = 0 to ``FINAL_FRACTIONS`` at the end of
        the run.
    """
    share = np.asarray(time, dtype=float)[..., np.newaxis] / DURATION  # [-]
    return STATIC_FRACTIONS + share * (FINAL_FRACTIONS - STATIC_FRACTIONS)


def temperature_law(time):
    """Evaluate the controlled temperature ramp.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s], scalar or shape (num_times,).

    Returns
    -------
    float or numpy.ndarray
        Temperature [K] with the shape of ``time``.
    """
    return STATIC_TEMPERATURE + TEMPERATURE_SLOPE * np.asarray(time, dtype=float)


def attach(inlet, controls):
    """Attach a DynamicInput with the given control laws to ``inlet``.

    Parameters
    ----------
    inlet : stream
        Raw feed.
    controls : dict
        Field names mapped to callables of time [s].

    Returns
    -------
    stream
        ``inlet``, now carrying the DynamicInput.
    """
    dynamic = DynamicInput()
    for name, law in controls.items():
        dynamic.add_variable(name, law)
    inlet.DynamicInlet = dynamic
    return inlet


def liquid_feed(data_path):
    """Return the static liquid feed of this module.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.

    Returns
    -------
    LiquidStream
        ``STATIC_MASS_FLOW`` [kg/s] of ``STATIC_FRACTIONS`` [-].
    """
    return _liquid_stream(data_path, mass_flow=STATIC_MASS_FLOW,
                          mass_frac=STATIC_FRACTIONS)


def simulation_with_unit(data_path, unit, inlet):
    """Attach ``inlet`` and the result grid to ``unit`` in an executor.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    unit : object
        Real, unsolved unit operation.
    inlet : stream
        Raw feed.

    Returns
    -------
    SimulationExec
        Executor holding ``unit`` with result times ``TIME_GRID`` [s].
    """
    unit.Inlet = inlet
    unit.result = DynamicResult({}, time=np.array(TIME_GRID))  # [s]
    return _sim_with_units(data_path, {'U01': unit})


def species_totals(simulation, basis):
    """Return the single raw row of GetRawMaterials as total and species.

    Parameters
    ----------
    simulation : SimulationExec
        Executor with one raw feed.
    basis : {'mass', 'mole'}
        Accounting basis.

    Returns
    -------
    total : float
        Total amount [kg] or [mol].
    species : numpy.ndarray
        Species amounts [kg] or [mol], shape (4,), database order.
    """
    table = simulation.GetRawMaterials(basis=basis)
    amount = 'mass' if basis == 'mass' else 'moles'
    assert list(table.columns) == [amount] + [
        f'{amount}_{name}' for name in SPECIES]
    row = table.iloc[0].to_numpy()
    return row[0], row[1:]


def on_basis(component_mass, molar_mass, basis):
    """Express integrated component masses on the accounting basis.

    Parameters
    ----------
    component_mass : numpy.ndarray
        Integrated species masses [kg], shape (num_species,).
    molar_mass : numpy.ndarray
        Molar masses [g/mol], shape (num_species,).
    basis : {'mass', 'mole'}
        Accounting basis.

    Returns
    -------
    numpy.ndarray
        Species amounts [kg] or [mol], shape (num_species,).
    """
    if basis == 'mass':
        return component_mass
    return component_mass * GRAMS_PER_KILOGRAM / molar_mass  # [mol]


def assert_species(simulation, basis, expected):
    """Check species totals and their sum against expectations.

    Parameters
    ----------
    simulation : SimulationExec
        Executor with one raw feed.
    basis : {'mass', 'mole'}
        Accounting basis.
    expected : numpy.ndarray
        Expected species amounts [kg] or [mol], shape (4,).
    """
    total, species = species_totals(simulation, basis)
    np.testing.assert_allclose(species, expected, rtol=RTOL, atol=0)
    assert total == pytest.approx(expected.sum(), rel=RTOL, abs=0)


# ---------- DynamicCollector: consumes mass_frac, mass_flow and temp

@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_collector_temperature_only_control_keeps_static_feed(data_path, basis):
    molar_mass, liquid_density = database_properties(data_path)
    inlet = attach(liquid_feed(data_path), {'temp': temperature_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    component_mass = STATIC_MASS_FLOW * DURATION * STATIC_FRACTIONS  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))

    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    # Constant flow: the mass-weighted temperature is the ramp's time mean.
    assert record['temp'] == pytest.approx(
        temperature_law(DURATION / 2), rel=RTOL)  # [K]
    assert record['pres'] == pytest.approx(STATIC_PRESSURE, rel=RTOL)
    assert record['vol'] == pytest.approx(
        STATIC_MASS_FLOW * DURATION
        / mixture_density(STATIC_FRACTIONS, liquid_density), rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_collector_mass_fraction_control_holds_mass_flow(data_path, basis):
    molar_mass, liquid_density = database_properties(data_path)
    inlet = attach(liquid_feed(data_path), {'mass_frac': fraction_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    component_mass = STATIC_MASS_FLOW * linear_integral(
        STATIC_FRACTIONS, (FINAL_FRACTIONS - STATIC_FRACTIONS) / DURATION)
    # [kg], the collector holds the static mass flow
    expected = on_basis(component_mass, molar_mass, basis)
    assert_species(simulation, basis, expected)

    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    frac = 'mass_frac' if basis == 'mass' else 'mole_frac'
    reported = record[[f'{frac}_{name}' for name in SPECIES]].to_numpy(dtype=float)
    np.testing.assert_allclose(reported, expected / expected.sum(),
                               rtol=RTOL, atol=0)
    volume_flow = STATIC_MASS_FLOW / mixture_density(
        fraction_law(np.array(TIME_GRID)), liquid_density)  # [m**3/s]
    assert record['vol'] == pytest.approx(trapezoid(volume_flow, TIME_GRID),
                                          rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_collector_mass_flow_only_control_keeps_composition(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    start, slope = FLOW_LAWS['mass_flow']
    inlet = attach(liquid_feed(data_path),
                   {'mass_flow': lambda time: start + slope * time})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    component_mass = linear_integral(start, slope) * STATIC_FRACTIONS  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_collector_flow_and_composition_controls_integrate_product(
        data_path, basis):
    molar_mass, _ = database_properties(data_path)
    start, slope = FLOW_LAWS['mass_flow']
    inlet = attach(liquid_feed(data_path), {
        'mass_flow': lambda time: start + slope * time,
        'mass_frac': fraction_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    time = np.array(TIME_GRID)  # [s]
    component_flow = (start + slope * time)[:, np.newaxis] * fraction_law(time)
    # [kg/s], controlled flow times controlled fractions
    component_mass = trapezoid(component_flow, time, axis=0)  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_collector_temperature_is_mass_flow_weighted(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    start, slope = FLOW_LAWS['mass_flow']
    inlet = attach(liquid_feed(data_path), {
        'mass_flow': lambda time: start + slope * time,
        'mass_frac': fraction_law, 'temp': temperature_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]

    time = np.array(TIME_GRID)  # [s]
    mass_flow = start + slope * time  # [kg/s]
    mole_flow = mass_flow * GRAMS_PER_KILOGRAM * np.sum(
        fraction_law(time) / molar_mass, axis=1)  # [mol/s]
    temperature = temperature_law(time)  # [K]
    mass_weighted = (trapezoid(mass_flow * temperature, time)
                     / trapezoid(mass_flow, time))  # [K]
    time_weighted = trapezoid(temperature, time) / DURATION  # [K]
    mole_weighted = (trapezoid(mole_flow * temperature, time)
                     / trapezoid(mole_flow, time))  # [K]
    assert record['temp'] == pytest.approx(mass_weighted, rel=RTOL)
    # The laws separate the three candidate averages by far more than RTOL.
    assert abs(mass_weighted - time_weighted) > 1e3 * RTOL * mass_weighted
    assert abs(mass_weighted - mole_weighted) > 1e3 * RTOL * mass_weighted


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_zero_flow_control_reports_zero_and_static_intensive_values(
        data_path, basis):
    inlet = attach(liquid_feed(data_path), {
        'mass_flow': lambda time: np.zeros_like(time),
        'mass_frac': fraction_law, 'temp': temperature_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    total, species = species_totals(simulation, basis)
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]

    assert total == 0
    np.testing.assert_array_equal(species, np.zeros(len(SPECIES)))
    assert record['temp'] == STATIC_TEMPERATURE  # [K], nothing to weight
    assert record['vol'] == 0
    frac = 'mass_frac' if basis == 'mass' else 'mole_frac'
    reported = record[[f'{frac}_{name}' for name in SPECIES]].to_numpy(dtype=float)
    np.testing.assert_allclose(reported, getattr(inlet, frac), rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mole_frac', 'vol_flow', 'mole_flow', 'pres'])
def test_collector_rejects_unconsumed_control(data_path, field):
    inlet = attach(liquid_feed(data_path),
                   {field: lambda time: np.ones_like(time)})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)
    message = (rf"controls \['{field}'\], which DynamicCollector does not "
               r"consume; it consumes \['mass_frac', 'mass_flow', 'temp'\]")
    with pytest.raises(ValueError, match=message):
        simulation.GetRawMaterials(basis='mass')


@pytest.mark.unit
def test_dynamic_accounting_rejects_unknown_basis(data_path):
    inlet = attach(liquid_feed(data_path), {'temp': temperature_law})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)
    unit = simulation.uos_instances['U01']
    # GetRawMaterials validates basis first; this checks the helper's own guard.
    with pytest.raises(ValueError, match=r"^basis must be either 'mass' or 'mole'$"):
        simulation._account_dynamic_inlet(unit, inlet, 'molar')


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_static_continuous_record_is_unchanged_guard(data_path, basis):
    inlet = liquid_feed(data_path)
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]

    amount = 'mass' if basis == 'mass' else 'moles'
    flow = 'mass_flow' if basis == 'mass' else 'mole_flow'
    frac = 'mass_frac' if basis == 'mass' else 'mole_frac'
    assert list(record.index) == (
        [amount, 'vol', 'temp', 'pres', flow, 'vol_flow']
        + [f'{frac}_{name}' for name in SPECIES])
    assert record[amount] == pytest.approx(getattr(inlet, flow) * DURATION,
                                           rel=RTOL)


# ---------- Evaporator: consumes mole_frac, mole_flow and temp

@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_evaporator_mole_fraction_control_holds_mole_flow(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    static_mole_flow = STATIC_MASS_FLOW * GRAMS_PER_KILOGRAM * np.sum(
        STATIC_FRACTIONS / molar_mass)  # [mol/s], hand conversion of the feed
    inlet = attach(liquid_feed(data_path), {'mole_frac': fraction_law})
    unit = Evaporator(vol_drum=EVAPORATOR_DRUM_VOLUME)
    simulation = simulation_with_unit(data_path, unit, inlet)

    component_moles = static_mole_flow * linear_integral(
        STATIC_FRACTIONS, (FINAL_FRACTIONS - STATIC_FRACTIONS) / DURATION)
    # [mol], the evaporator holds the static molar flow
    expected = (component_moles if basis == 'mole'
                else component_moles * molar_mass / GRAMS_PER_KILOGRAM)
    assert_species(simulation, basis, expected)

    consumed = unit.get_inputs(np.array(TIME_GRID))['Inlet']
    np.testing.assert_allclose(consumed['mole_flow'], static_mole_flow,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(consumed['mole_frac'],
                               fraction_law(np.array(TIME_GRID)),
                               rtol=RTOL, atol=0)

    _, liquid_density = database_properties(data_path)
    mole_frac = fraction_law(np.array(TIME_GRID))  # [-]
    component_mass_flow = (static_mole_flow * mole_frac * molar_mass
                           / GRAMS_PER_KILOGRAM)  # [kg/s]
    mass_flow = component_mass_flow.sum(axis=1)  # [kg/s]
    mass_frac = component_mass_flow / mass_flow[:, np.newaxis]  # [-], by hand
    volume_flow = mass_flow / mixture_density(mass_frac, liquid_density)  # [m**3/s]
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    assert record['vol'] == pytest.approx(trapezoid(volume_flow, TIME_GRID),
                                          rel=RTOL)


@pytest.mark.unit
def test_evaporator_rejects_unconsumed_mass_fraction(data_path):
    inlet = attach(liquid_feed(data_path), {'mass_frac': fraction_law})
    simulation = simulation_with_unit(
        data_path, Evaporator(vol_drum=EVAPORATOR_DRUM_VOLUME), inlet)
    with pytest.raises(ValueError, match=(
            r"controls \['mass_frac'\], which Evaporator does not consume; it "
            r"consumes \['mole_frac', 'mole_flow', 'temp'\]")):
        simulation.GetRawMaterials(basis='mole')


# ---------- PlugFlowReactor: consumes mole_conc, temp and vol_flow

def make_pfr():
    """Construct a real plug-flow reactor that consumes volume flow.

    Returns
    -------
    PlugFlowReactor
        Unsolved reactor; its geometry is irrelevant to raw accounting.
    """
    return PlugFlowReactor(diam_in=PFR_DIAMETER, num_discr=PFR_NODES)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_pfr_concentration_control_holds_volume_flow(data_path, basis):
    molar_mass, liquid_density = database_properties(data_path)
    static_vol_flow = STATIC_MASS_FLOW / mixture_density(
        STATIC_FRACTIONS, liquid_density)  # [m**3/s], hand conversion
    inlet = attach(liquid_feed(data_path), {
        'mole_conc': lambda time: CONC_START + np.asarray(time)[..., np.newaxis]
        / DURATION * (CONC_END - CONC_START)})
    unit = make_pfr()
    simulation = simulation_with_unit(data_path, unit, inlet)

    component_moles = static_vol_flow * LITERS_PER_CUBIC_METER * linear_integral(
        CONC_START, (CONC_END - CONC_START) / DURATION)
    # [mol], held volume flow times the integrated concentrations
    expected = (component_moles if basis == 'mole'
                else component_moles * molar_mass / GRAMS_PER_KILOGRAM)
    assert_species(simulation, basis, expected)

    consumed = unit.get_inputs(np.array(TIME_GRID))['Inlet']
    np.testing.assert_allclose(consumed['vol_flow'], static_vol_flow,
                               rtol=RTOL, atol=0)
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    assert record['vol'] == pytest.approx(static_vol_flow * DURATION, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_pfr_volume_flow_control_keeps_concentration(data_path, basis):
    molar_mass, liquid_density = database_properties(data_path)
    static_conc = (mixture_density(STATIC_FRACTIONS, liquid_density)
                   * STATIC_FRACTIONS / molar_mass)
    # [mol/L], [kg/m**3] / [g/mol] equals [kmol/m**3] = [mol/L]
    start, slope = FLOW_LAWS['vol_flow']
    inlet = attach(liquid_feed(data_path),
                   {'vol_flow': lambda time: start + slope * time})
    unit = make_pfr()
    simulation = simulation_with_unit(data_path, unit, inlet)

    component_moles = (linear_integral(start, slope) * LITERS_PER_CUBIC_METER
                       * static_conc)  # [mol]
    expected = (component_moles if basis == 'mole'
                else component_moles * molar_mass / GRAMS_PER_KILOGRAM)
    assert_species(simulation, basis, expected)

    consumed = unit.get_inputs(np.array(TIME_GRID))['Inlet']
    np.testing.assert_allclose(consumed['mole_conc'],
                               np.tile(static_conc, (len(TIME_GRID), 1)),
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mole_frac', 'mass_flow'])
def test_pfr_rejects_unconsumed_control(data_path, field):
    inlet = attach(liquid_feed(data_path), {
        field: fraction_law if field == 'mole_frac'
        else (lambda time: np.ones_like(time))})
    simulation = simulation_with_unit(data_path, make_pfr(), inlet)
    with pytest.raises(ValueError, match=(
            rf"controls \['{field}'\], which PlugFlowReactor does not consume; "
            r"it consumes \['mole_conc', 'temp', 'vol_flow'\]")):
        simulation.GetRawMaterials(basis='mole')


# ---------- Slurry feeds: vol_flow, liquid mass_conc and population

def slurry_phase_masses(data_path, flow, liquid_conc):
    """Integrate liquid and solid masses of the two-phase slurry fixture.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    flow : numpy.ndarray
        Slurry volume flow [m**3/s] on ``TIME_GRID``.
    liquid_conc : numpy.ndarray
        Liquid mass concentrations [kg/m**3], shape (num_times, 4).

    Returns
    -------
    liquid : numpy.ndarray
        Integrated liquid species masses [kg], shape (4,).
    solid : numpy.ndarray
        Integrated solid species masses [kg], shape (4,).
    """
    with open(_thermo_path(data_path)) as file:
        database = json.load(file)
    solid_density_pure = np.array([item['rho_solid'] for item in database.values()])
    solid_fractions = SLURRY_CRYSTAL_FRACTIONS  # [-]
    solid_density = mixture_density(solid_fractions, solid_density_pure)  # [kg/m**3]
    time = np.array(TIME_GRID)  # [s]
    liquid = trapezoid((flow * (1 - SLURRY_SOLID_FRACTION))[:, np.newaxis]
                       * liquid_conc, time, axis=0)  # [kg]
    solid = (trapezoid(flow, time) * SLURRY_SOLID_FRACTION * solid_density
             * solid_fractions)  # [kg]
    return liquid, solid


@pytest.mark.unit
def test_slurry_liquid_concentration_control(data_path):
    slurry = _slurry_stream(data_path)
    static_conc = slurry.Liquid_1.mass_conc.copy()  # [kg/m**3]
    final_conc = static_conc[::-1].copy()  # [kg/m**3], reversed, distinct
    conc_law = (lambda time: static_conc + np.asarray(time)[..., np.newaxis]
                / DURATION * (final_conc - static_conc))  # [kg/m**3]
    attach(slurry, {'vol_flow': lambda time: np.full_like(time, SLURRY_FLOW),
                    'mass_conc': conc_law})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)

    table = simulation.GetRawMaterials(basis='mass')

    time = np.array(TIME_GRID)  # [s]
    liquid, solid = slurry_phase_masses(
        data_path, np.full_like(time, SLURRY_FLOW), conc_law(time))
    rows = {name[2]: row.to_numpy()[1:] for name, row in table.iterrows()}
    np.testing.assert_allclose(rows[[key for key in rows if 'Liquid' in key][0]],
                               liquid, rtol=RTOL, atol=0)
    # The stored solid fractions carry round-off in the absent species, so
    # compare those entries against a tolerance scaled by the solid total.
    np.testing.assert_allclose(rows[[key for key in rows if 'Solid' in key][0]],
                               solid, rtol=RTOL, atol=RTOL * solid.sum())


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mole_flow', 'mass_flow', 'mass_frac'])
def test_slurry_rejects_unconsumed_control(data_path, field):
    slurry = attach(_slurry_stream(data_path),
                    {field: lambda time: np.ones_like(time)})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)
    with pytest.raises(ValueError, match=(
            rf"controls \['{field}'\], which DynamicCollector does not consume; "
            r"it consumes \['mass_conc', 'vol_flow', 'temp', 'distrib', 'mu_n'\]")):
        simulation.GetRawMaterials(basis='mass')


@pytest.mark.unit
def test_phase_level_dynamic_inlet_is_rejected(data_path):
    slurry = _slurry_stream(data_path)
    attach(slurry.Liquid_1, {'mass_frac': fraction_law})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)
    with pytest.raises(ValueError, match=(
            r"have their own DynamicInlet, which no unit operation reads; "
            r"attach the DynamicInput to the mixed inlet itself")):
        simulation.GetRawMaterials(basis='mass')


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_msmpr_slurry_feed_follows_crystallizer_inputs(data_path, basis):
    """Account an FVM MSMPR feed through the crystallizer's phase mapping.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories; the heat-duty fixture uses its
        five-species flowsheet database.
    basis : {'mass', 'mole'}
        Accounting basis.
    """
    unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=SLURRY_FLOW)
    inlet = unit.Inlet
    static_conc = inlet.Liquid_1.mass_conc.copy()  # [kg/m**3]
    flow_slope = SLURRY_FLOW / DURATION  # [m**3/s**2], flow doubles over the run
    attach(inlet, {'vol_flow': lambda time: SLURRY_FLOW + flow_slope * time,
                   'mass_conc': lambda time: static_conc * (
                       1 + np.asarray(time)[..., np.newaxis] / DURATION)})
    unit.result = DynamicResult({}, time=np.array(TIME_GRID))  # [s]
    simulation = SimulationExec(
        str(data_path['flowsheet'] / 'compound_database.json'), {'CR01': []})
    simulation.uos_instances = {'CR01': unit}

    table = simulation.GetRawMaterials(basis=basis, totals=False)

    time = np.array(TIME_GRID)  # [s]
    grid = inlet.Solid_1.x_distrib  # [um]
    integrand = inlet.distrib * grid**3  # [#/m**3/um * um**3]
    third_moment = 1e-18 * np.sum(np.diff(grid) * (integrand[1:] + integrand[:-1]) / 2)
    # [m**3/m**3], trapezoidal mu_3; 1e-18 is the exact um**3-to-m**3 factor
    solid_fraction = inlet.Solid_1.kv * third_moment  # [-]
    flow = SLURRY_FLOW + flow_slope * time  # [m**3/s]
    liquid_mass = trapezoid(
        (flow * (1 - solid_fraction))[:, np.newaxis]
        * static_conc * (1 + time[:, np.newaxis] / DURATION), time, axis=0)  # [kg]
    molar_mass = np.asarray(inlet.Liquid_1.mw)  # [g/mol], database values
    amount = 'mass' if basis == 'mass' else 'moles'
    liquid_row = table.loc[[name for name in table.index if 'Liquid' in name[2]][0]]
    expected = (liquid_mass.sum() if basis == 'mass'
                else np.sum(liquid_mass * GRAMS_PER_KILOGRAM / molar_mass))
    assert liquid_row[amount] == pytest.approx(expected, rel=RTOL)
    solid_row = table.loc[[name for name in table.index if 'Solid' in name[2]][0]]
    assert solid_row['vol'] == pytest.approx(
        trapezoid(flow * solid_fraction, time), rel=RTOL)  # [m**3]


@pytest.mark.assimulo
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_stream_table_reports_consumed_dynamic_raw_feed(data_path, basis):
    """Solve a real collector and read the raw feed row of GetStreamTable.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    basis : {'mass', 'mole'}
        Accounting basis.
    """
    pytest.importorskip('assimulo')
    molar_mass, _ = database_properties(data_path)
    path = _thermo_path(data_path)
    feed = LiquidStream(path, temp=STATIC_TEMPERATURE, pres=STATIC_PRESSURE,
                        mass_flow=STATIC_MASS_FLOW, mass_frac=STATIC_FRACTIONS,
                        verbose=False)
    attach(feed, {'mass_frac': fraction_law, 'temp': temperature_law})
    simulation = SimulationExec(path, {'COL': []})
    simulation.COL = DynamicCollector()
    simulation.COL.Inlet = feed
    simulation.SolveFlowsheet(
        kwargs_run={'COL': {'time_grid': list(TIME_GRID), 'verbose': False}},
        verbose=False)

    table = simulation.result.GetStreamTable(basis=basis)

    raw = table.loc[('COL', 'Inlet_0')].iloc[0]
    component_mass = STATIC_MASS_FLOW * linear_integral(
        STATIC_FRACTIONS, (FINAL_FRACTIONS - STATIC_FRACTIONS) / DURATION)
    # [kg], held mass flow times the integrated linear fractions
    expected = on_basis(component_mass, molar_mass, basis)
    amount = 'mass' if basis == 'mass' else 'moles'
    frac = 'mass_frac' if basis == 'mass' else 'mole_frac'
    assert raw[amount] == pytest.approx(expected.sum(), rel=RTOL)
    np.testing.assert_allclose(
        raw[[f'{frac}_{name}' for name in SPECIES]].to_numpy(dtype=float),
        expected / expected.sum(), rtol=RTOL, atol=0)
    # Constant mass flow: the mass-weighted temperature is the ramp's mean.
    assert raw['temp'] == pytest.approx(temperature_law(DURATION / 2), rel=RTOL)


# ---------- Round-2 regressions: evaluation, edge samples, slurry state

def build_slurry(path, solid_temp=STATIC_TEMPERATURE, reverse=False):
    """Build the two-phase slurry fixture from an arbitrary database.

    Parameters
    ----------
    path : str
        Four-species thermodynamic database (A, B, C, solv).
    solid_temp : float, optional
        Stored crystal-phase temperature [K]; the liquid is at
        ``STATIC_TEMPERATURE``.
    reverse : bool, optional
        Attach the phases as [solid, liquid] instead of [liquid, solid].

    Returns
    -------
    SlurryStream
        Liquid of ``SLURRY_LIQUID_FRACTIONS`` and crystals of
        ``SLURRY_CRYSTAL_FRACTIONS`` [-], with solid volume fraction
        ``SLURRY_SOLID_FRACTION`` (unit shape factor), as in
        ``test_simexec_raw_materials._slurry_stream``.
    """
    liquid = LiquidStream(path, temp=STATIC_TEMPERATURE, pres=STATIC_PRESSURE,
                          mass_flow=SLURRY_LIQUID_FLOW,
                          mass_frac=SLURRY_LIQUID_FRACTIONS,
                          verbose=False)  # [kg/s], [-]
    particle_size = np.array([1.0, 2.0, 3.0, 4.0])  # [um]
    distribution_shape = np.array([1.0, 2.0, 2.0, 1.0])  # [-]
    unscaled_third_moment = trapezoidal_rule(
        particle_size, distribution_shape * particle_size**3) * (1.0e-6)**3
    # [m**3], exact um-to-m conversion
    distribution = (distribution_shape * SLURRY_SOLID_FRACTION
                    / unscaled_third_moment)  # [#/m**3/um]
    solid = SolidStream(path, temp=solid_temp, pres=STATIC_PRESSURE,
                        mass_flow=SLURRY_CRYSTAL_FLOW,
                        mass_frac=SLURRY_CRYSTAL_FRACTIONS,
                        x_distrib=particle_size, distrib=distribution)
    slurry = SlurryStream(vol_flow=liquid.vol + solid.vol,
                          x_distrib=particle_size, distrib=distribution)
    slurry.Phases = [solid, liquid] if reverse else [liquid, solid]
    return slurry


def slurry_expectations(path, flow, solid_fraction):
    """Integrate liquid and crystal masses and volumes of a constant feed.

    Parameters
    ----------
    path : str
        Database supplying ``rho_liq`` and ``rho_solid`` [kg/m**3].
    flow : float
        Constant slurry volume flow [m**3/s].
    solid_fraction : float
        Crystal volume fraction ``kv * mu_3`` [-].

    Returns
    -------
    dict
        ``liquid_mass``, ``solid_mass`` [kg], ``liquid_vol``, ``solid_vol``
        [m**3] over ``DURATION``.
    """
    with open(path) as file:
        database = json.load(file)
    rho_liq = np.array([item['rho_liq'] for item in database.values()])
    rho_solid = np.array([item['rho_solid'] for item in database.values()])
    liquid_density = mixture_density(SLURRY_LIQUID_FRACTIONS, rho_liq)
    solid_density = mixture_density(SLURRY_CRYSTAL_FRACTIONS, rho_solid)
    # [kg/m**3], ideal mixing at the fixture's phase compositions
    liquid_vol = flow * DURATION * (1 - solid_fraction)  # [m**3]
    solid_vol = flow * DURATION * solid_fraction  # [m**3]
    return {'liquid_mass': liquid_vol * liquid_density,
            'solid_mass': solid_vol * solid_density,
            'liquid_vol': liquid_vol, 'solid_vol': solid_vol}


def phase_kinds(table):
    """List the stream class of each raw row in table order.

    Parameters
    ----------
    table : pandas.DataFrame
        GetRawMaterials(totals=False) output.

    Returns
    -------
    list of str
        Class names, such as ``LiquidStream`` or ``SolidStream``.
    """
    return [name.split('.')[0] for name in table.index.get_level_values(2)]


def phase_rows(table):
    """Split a two-phase raw table into its liquid and solid rows.

    Parameters
    ----------
    table : pandas.DataFrame
        GetRawMaterials(totals=False) output for one slurry inlet.

    Returns
    -------
    liquid, solid : pandas.Series
        Rows whose phase name starts with LiquidStream or SolidStream.
    """
    names = table.index.get_level_values(2)
    liquid = table[[name.startswith('LiquidStream') for name in names]].iloc[0]
    solid = table[[name.startswith('SolidStream') for name in names]].iloc[0]
    return liquid, solid


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_scalar_only_control_callable_is_evaluated_per_sample(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    saturation_time = 5.0  # [s], the ramp holds its value afterwards
    ramp_slope = 1.0  # [K/s], heating rate before saturation
    inlet = attach(liquid_feed(data_path), {
        'temp': lambda time: STATIC_TEMPERATURE
        + ramp_slope * min(time, saturation_time)})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    component_mass = STATIC_MASS_FLOW * DURATION * STATIC_FRACTIONS  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    temperature = STATIC_TEMPERATURE + ramp_slope * np.minimum(
        TIME_GRID, saturation_time)  # [K]
    # Constant mass flow: the mass-weighted temperature is the time mean.
    assert record['temp'] == pytest.approx(
        trapezoid(temperature, TIME_GRID) / DURATION, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('control', ['temp', 'mass_frac'])
def test_single_species_feed_keeps_species_axis(data_path, tmp_path, basis,
                                                control):
    """Keep the species axis for static and scalar-valued compositions.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    tmp_path : pathlib.Path
        Location of the one-species database copy.
    basis : {'mass', 'mole'}
        Accounting basis.
    control : {'temp', 'mass_frac'}
        Temperature ramp (static composition) or a composition control
        returning the scalar 1.0 for the only species.
    """
    with open(_thermo_path(data_path)) as file:
        database = json.load(file)
    pure_path = tmp_path / 'pure_a.json'
    pure_path.write_text(json.dumps({'A': database['A']}))
    inlet = LiquidStream(str(pure_path), temp=STATIC_TEMPERATURE,
                         pres=STATIC_PRESSURE, mass_flow=STATIC_MASS_FLOW,
                         mass_frac=[1.0], verbose=False)
    attach(inlet, {'temp': temperature_law} if control == 'temp'
           else {'mass_frac': lambda time: 1.0})
    collector = DynamicCollector()
    simulation = SimulationExec(str(pure_path), {'U01': []})
    simulation.uos_instances = {'U01': collector}
    collector.Inlet = inlet
    collector.oper_mode = 'Continuous'
    collector.result = DynamicResult({}, time=np.array(TIME_GRID))  # [s]

    table = simulation.GetRawMaterials(basis=basis)

    total_mass = STATIC_MASS_FLOW * DURATION  # [kg], pure A
    expected = (total_mass if basis == 'mass'
                else total_mass * GRAMS_PER_KILOGRAM / database['A']['mw'])
    np.testing.assert_allclose(table.iloc[0].to_numpy(), [expected, expected],
                               rtol=RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_stopped_feed_samples_contribute_nothing(data_path, basis):
    molar_mass, liquid_density = database_properties(data_path)
    inlet = attach(liquid_feed(data_path), {
        'mass_flow': lambda time: max(0.0, STATIC_MASS_FLOW - FEED_RAMP_DOWN * time),
        'mass_frac': lambda time: (STATIC_FRACTIONS if time < SHUTDOWN_TIME
                                   else np.zeros(len(SPECIES)))})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    time = np.array(TIME_GRID)  # [s]
    mass_flow = np.maximum(0.0, STATIC_MASS_FLOW - FEED_RAMP_DOWN * time)  # [kg/s]
    fed = time < SHUTDOWN_TIME
    component_flow = (mass_flow * fed)[:, np.newaxis] * STATIC_FRACTIONS  # [kg/s]
    component_mass = trapezoid(component_flow, time, axis=0)  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    volume_flow = component_flow.sum(axis=1) / mixture_density(
        STATIC_FRACTIONS, liquid_density)  # [m**3/s]
    assert record['vol'] == pytest.approx(trapezoid(volume_flow, time), rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_uncontrolled_slurry_reports_consumed_temperature(data_path, basis):
    slurry = build_slurry(_thermo_path(data_path), solid_temp=SOLID_TEMPERATURE)
    stored = sorted(phase.temp for phase in slurry.Phases)  # [K]
    assert stored == [STATIC_TEMPERATURE, SOLID_TEMPERATURE]
    attach(slurry, {'vol_flow': lambda time: SLURRY_FLOW})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)
    collector = simulation.uos_instances['U01']
    consumed_temp = collector.get_inputs_new(0.0)['Inlet']['temp']  # [K]
    assert consumed_temp == slurry.temp
    assert STATIC_TEMPERATURE < consumed_temp < SOLID_TEMPERATURE

    table = simulation.GetRawMaterials(basis=basis, totals=False)

    np.testing.assert_allclose(table['temp'], consumed_temp, rtol=RTOL, atol=0)


@pytest.mark.unit
def test_dynamic_slurry_rows_follow_phase_order(data_path):
    path = _thermo_path(data_path)
    static = build_slurry(path, reverse=True)
    static_table = _sim_with_inlet(data_path, static, time=TIME_GRID
                                   ).GetRawMaterials(basis='mass', totals=False)
    dynamic = build_slurry(path, reverse=True)
    static_flow = dynamic.vol_flow  # [m**3/s]
    attach(dynamic, {'vol_flow': lambda time: static_flow})
    dynamic_table = _sim_with_inlet(data_path, dynamic, time=TIME_GRID
                                    ).GetRawMaterials(basis='mass', totals=False)

    assert phase_kinds(static_table) == ['SolidStream', 'LiquidStream']
    assert phase_kinds(dynamic_table) == phase_kinds(static_table)
    np.testing.assert_allclose(dynamic_table['mass'].to_numpy(),
                               static_table['mass'].to_numpy(), rtol=1e-10, atol=0)
    # [-], the static phase flows were reconciled by the same density closure,
    # so the two accounts agree to round-off accumulated over the run.


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mu_n', 'distrib'])
def test_controlled_population_takes_precedence(data_path, field):
    path = _thermo_path(data_path)
    slurry = build_slurry(path)
    controlled = POPULATION_SCALE * (slurry.moments if field == 'mu_n'
                                     else slurry.distrib)
    # [m**n/m**3] or [#/m**3/um], scaled population; mu_3 scales identically
    attach(slurry, {'vol_flow': lambda time: SLURRY_FLOW,
                    field: lambda time: controlled})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)

    liquid, solid = phase_rows(simulation.GetRawMaterials(basis='mass',
                                                          totals=False))

    expected = slurry_expectations(path, SLURRY_FLOW,
                                   POPULATION_SCALE * SLURRY_SOLID_FRACTION)
    assert liquid['mass'] == pytest.approx(expected['liquid_mass'], rel=RTOL)
    assert solid['mass'] == pytest.approx(expected['solid_mass'], rel=RTOL)
    assert liquid['vol'] == pytest.approx(expected['liquid_vol'], rel=RTOL)
    assert solid['vol'] == pytest.approx(expected['solid_vol'], rel=RTOL)


@pytest.mark.unit
def test_crystal_mass_uses_solid_density(data_path, tmp_path):
    with open(_thermo_path(data_path)) as file:
        database = json.load(file)
    for name, density in ASYMMETRIC_SOLID_DENSITY.items():
        database[name]['rho_solid'] = density  # [kg/m**3]
    path = tmp_path / 'asymmetric_solid.json'
    path.write_text(json.dumps(database))
    slurry = build_slurry(str(path))
    attach(slurry, {'vol_flow': lambda time: SLURRY_FLOW})
    simulation = _sim_with_inlet(data_path, slurry, time=TIME_GRID)

    liquid, solid = phase_rows(simulation.GetRawMaterials(basis='mass',
                                                          totals=False))

    expected = slurry_expectations(str(path), SLURRY_FLOW, SLURRY_SOLID_FRACTION)
    assert solid['mass'] == pytest.approx(expected['solid_mass'], rel=RTOL)
    assert liquid['mass'] == pytest.approx(expected['liquid_mass'], rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
@pytest.mark.parametrize('conc_controlled', [True, False],
                         ids=['flow_and_conc', 'flow_only'])
def test_msmpr_liquid_feed_integrates_flow_times_concentration(
        data_path, basis, conc_controlled):
    unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=SLURRY_FLOW)
    liquid = unit.Liquid_1
    path = str(data_path['flowsheet'] / 'compound_database.json')
    feed = LiquidStream(path, temp=liquid.temp, mass_frac=liquid.mass_frac,
                        vol_flow=SLURRY_FLOW, verbose=False)
    static_conc = feed.mass_conc.copy()  # [kg/m**3]
    flow_slope = SLURRY_FLOW / DURATION  # [m**3/s**2], flow doubles over the run
    conc_growth = 1.0 if conc_controlled else 0.0  # [-], relative rise over the run
    controls = {'vol_flow': lambda time: SLURRY_FLOW + flow_slope * time}
    if conc_controlled:
        controls['mass_conc'] = lambda time: static_conc * (1 + time / DURATION)
    attach(feed, controls)
    unit.Inlet = feed
    unit.result = DynamicResult({}, time=np.array(TIME_GRID))  # [s]
    simulation = SimulationExec(path, {'CR01': []})
    simulation.uos_instances = {'CR01': unit}

    total, species = species_totals_named(simulation, basis, liquid.name_species)

    time = np.array(TIME_GRID)  # [s]
    flow = SLURRY_FLOW + flow_slope * time  # [m**3/s]
    component_mass = trapezoid(
        flow[:, np.newaxis] * static_conc
        * (1 + conc_growth * time[:, np.newaxis] / DURATION), time, axis=0)
    # [kg], solid-free feed: Q(t) * mass_conc(t); the crystallizer's own
    # input method supplies the static Liquid_1 concentration when uncontrolled
    molar_mass = np.array(liquid.mw, dtype=float)  # [g/mol], database values
    expected = on_basis(component_mass, molar_mass, basis)
    np.testing.assert_allclose(species, expected, rtol=RTOL, atol=0)
    assert total == pytest.approx(expected.sum(), rel=RTOL)
    record = simulation.GetRawMaterials(basis=basis, totals=False).iloc[0]
    assert record['vol'] == pytest.approx(linear_integral(SLURRY_FLOW, flow_slope),
                                          rel=RTOL)


def species_totals_named(simulation, basis, names):
    """Return total and species amounts for a database with given species.

    Parameters
    ----------
    simulation : SimulationExec
        Executor with one raw feed.
    basis : {'mass', 'mole'}
        Accounting basis.
    names : sequence of str
        Species names in database order.

    Returns
    -------
    total : float
        Total amount [kg] or [mol].
    species : numpy.ndarray
        Species amounts [kg] or [mol], database order.
    """
    table = simulation.GetRawMaterials(basis=basis)
    amount = 'mass' if basis == 'mass' else 'moles'
    assert list(table.columns) == [amount] + [f'{amount}_{name}' for name in names]
    row = table.iloc[0].to_numpy()
    return row[0], row[1:]


@pytest.mark.unit
def test_unit_without_inlet_layout_is_rejected(data_path):
    inlet = attach(liquid_feed(data_path), {'temp': temperature_law})
    extractor = ContinuousExtractor()
    simulation = simulation_with_unit(data_path, extractor, inlet)
    with pytest.raises(ValueError, match=(
            r"Unit ContinuousExtractor declares no inlet layout .*remove the "
            r"DynamicInlet from its raw inlet")):
        simulation.GetRawMaterials(basis='mass')


@pytest.mark.unit
def test_population_control_on_liquid_feed_is_rejected(data_path):
    field = 'distrib'  # the FVM MSMPR consumes a distribution, not moments
    unit, _ = make_heat_unit(data_path, MSMPR, ramp=0.0, feed_flow=SLURRY_FLOW)
    liquid = unit.Liquid_1
    path = str(data_path['flowsheet'] / 'compound_database.json')
    feed = LiquidStream(path, temp=liquid.temp, mass_frac=liquid.mass_frac,
                        vol_flow=SLURRY_FLOW, verbose=False)
    num_bins = unit.states_in_dict['Inlet'][field]
    attach(feed, {field: lambda time: np.ones(num_bins)})
    unit.Inlet = feed
    unit.result = DynamicResult({}, time=np.array(TIME_GRID))  # [s]
    simulation = SimulationExec(path, {'CR01': []})
    simulation.uos_instances = {'CR01': unit}
    with pytest.raises(ValueError, match=(
            rf"controls \['{field}'\] on the single-phase raw stream .*carries "
            r"no crystal population; remove those controls")):
        simulation.GetRawMaterials(basis='mass')


def make_holdup(data_path, inlet):
    """Attach ``inlet`` to a real ContinuousHoldup with a liquid phase.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    inlet : stream
        Raw feed.

    Returns
    -------
    SimulationExec
        Executor with the holdup and result times ``TIME_GRID`` [s].
    """
    holdup = ContinuousHoldup()
    holdup.Phases = LiquidPhase(_thermo_path(data_path), temp=STATIC_TEMPERATURE,
                                mass=HOLDUP_MASS, mass_frac=STATIC_FRACTIONS,
                                verbose=False)
    return simulation_with_unit(data_path, holdup, inlet)


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_holdup_composition_control_holds_mass_flow(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    simulation = make_holdup(data_path, attach(liquid_feed(data_path),
                                               {'mass_frac': fraction_law}))
    component_mass = STATIC_MASS_FLOW * linear_integral(
        STATIC_FRACTIONS, (FINAL_FRACTIONS - STATIC_FRACTIONS) / DURATION)
    # [kg], the holdup consumes mass_flow, so the static mass flow is held
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))


@pytest.mark.unit
def test_slurry_into_liquid_holdup_is_rejected(data_path):
    slurry = attach(_slurry_stream(data_path), {'temp': temperature_law})
    simulation = make_holdup(data_path, slurry)
    with pytest.raises(ValueError, match=(
            r"The unit consumes \['mass_flow', 'mass_frac', 'temp'\] from "
            r"slurry feed .*needs its vol_flow, liquid mass_conc and a "
            r"population")):
        simulation.GetRawMaterials(basis='mass')


# ---------- Round-3 regressions: buffers, invalid samples, full-grid units

@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_reused_control_buffer_keeps_each_sample(data_path, basis):
    molar_mass, _ = database_properties(data_path)
    buffer = np.empty(len(SPECIES))  # [-], one array mutated by every call

    def reused_fractions(time):
        """Write the linear blend at ``time`` [s] into a shared buffer.

        Parameters
        ----------
        time : float
            Time [s].

        Returns
        -------
        numpy.ndarray
            The shared buffer holding mass fractions [-], shape (4,).
        """
        buffer[:] = fraction_law(time)
        return buffer

    inlet = attach(liquid_feed(data_path), {'mass_frac': reused_fractions})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    component_mass = STATIC_MASS_FLOW * linear_integral(
        STATIC_FRACTIONS, (FINAL_FRACTIONS - STATIC_FRACTIONS) / DURATION)
    # [kg], held mass flow times the integrated linear fractions
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))


@pytest.mark.unit
def test_non_finite_composition_sample_is_not_dropped(data_path):
    invalid_time = TIME_GRID[2]  # [s], one interior sample carries NaN
    inlet = attach(liquid_feed(data_path), {
        'mass_frac': lambda time: (np.full(len(SPECIES), np.nan)
                                   if time == invalid_time else STATIC_FRACTIONS)})
    simulation = _sim_with_inlet(data_path, inlet, time=TIME_GRID)

    total, species = species_totals(simulation, 'mass')

    assert np.isnan(total)
    assert np.isnan(species).all()


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_mixer_feed_is_evaluated_on_its_whole_grid(data_path, basis):
    """Account a Mixer feed whose control accepts only array time.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    basis : {'mass', 'mole'}
        Accounting basis.

    Notes
    -----
    The Mixer takes its grid from a connected inlet whose upstream profile
    is supplied directly; ``solve_unit`` then evaluates the raw inlet's
    control once on that array.
    """
    molar_mass, _ = database_properties(data_path)
    start, slope = FLOW_LAWS['mass_flow']
    simulation = solved_mixer(data_path, {
        'mass_flow': lambda time: start + slope * time[...] * np.ones(len(time))})

    component_mass = linear_integral(start, slope) * STATIC_FRACTIONS  # [kg]
    assert_species(simulation, basis, on_basis(component_mass, molar_mass, basis))


def solved_mixer(data_path, controls):
    """Solve a real Mixer fed by a connected profile and a dynamic raw feed.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    controls : dict
        Control laws of the raw feed, evaluated by the Mixer on its grid.

    Returns
    -------
    SimulationExec
        Executor holding the solved Mixer as ``MIX``.

    Notes
    -----
    The connected inlet's upstream profile is supplied directly on
    ``TIME_GRID`` [s]; it fixes the Mixer's evaluation grid.
    """
    grid = np.array(TIME_GRID)  # [s]
    connected = liquid_feed(data_path)
    profile = {'mass_flow': np.full(len(grid), STATIC_MASS_FLOW),  # [kg/s]
               'mass_frac': np.tile(STATIC_FRACTIONS, (len(grid), 1)),  # [-]
               'temp': np.full(len(grid), STATIC_TEMPERATURE)}  # [K]
    connected.y_upstream = profile
    connected.y_inlet = profile
    connected.time_upstream = grid
    raw = attach(liquid_feed(data_path), controls)
    mixer = Mixer()
    mixer.Inlets = [connected, raw]
    mixer.solve_unit()
    return _sim_with_units(data_path, {'MIX': mixer})


@pytest.mark.unit
@pytest.mark.parametrize('field', ['mole_frac', 'mass_conc'])
def test_mixer_rejects_unconsumed_control(data_path, field):
    simulation = solved_mixer(data_path, {
        field: lambda time: np.tile(STATIC_FRACTIONS, (len(time), 1))})
    with pytest.raises(ValueError, match=(
            rf"controls \['{field}'\], which Mixer does not consume; it "
            r"consumes \['mass_frac', 'mass_flow', 'temp'\]")):
        simulation.GetRawMaterials(basis='mass')


@pytest.mark.unit
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_concentration_feed_static_and_dynamic_rows_agree(data_path, basis):
    """Pin static/dynamic agreement for a solvent-closed concentration feed.

    Parameters
    ----------
    data_path : dict
        Repository test-data directories.
    basis : {'mass', 'mole'}
        Accounting basis.

    Notes
    -----
    Dynamic raw accounting reports ``Q c`` as the PFR consumes it, while the
    static row uses the stream flow from its ideal-mixing density. With
    ``name_solv`` the solvent concentration closes that density, so the two
    agree; a no-op temperature control switches to the dynamic path.
    """
    def make_feed():
        """Build the solvent-closed concentration feed.

        Returns
        -------
        LiquidStream
            Feed at ``STATIC_TEMPERATURE`` [K] with ``CONC_START`` [mol/L]
            solutes and the solvent closing the volume.
        """
        return LiquidStream(_thermo_path(data_path), temp=STATIC_TEMPERATURE,
                            pres=STATIC_PRESSURE, mole_conc=CONC_START,
                            vol_flow=FLOW_LAWS['vol_flow'][0], name_solv='solv',
                            verbose=False)

    static_totals = species_totals(
        simulation_with_unit(data_path, make_pfr(), make_feed()), basis)[1]
    dynamic_feed = attach(make_feed(),
                          {'temp': lambda time: STATIC_TEMPERATURE})
    dynamic_totals = species_totals(
        simulation_with_unit(data_path, make_pfr(), dynamic_feed), basis)[1]

    # [-], the two paths differ only by round-off of the density closure
    np.testing.assert_allclose(dynamic_totals, static_totals, rtol=1e-10, atol=0)
