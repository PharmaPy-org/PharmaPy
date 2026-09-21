"""Liquid inventory and slurry flow contracts for #217, without solver backends.

Synthetic species have unequal densities and molar masses. Expected inventories
come from adding species volumes and mole amounts, independently of phase APIs.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/217
"""

import json

import numpy as np
import pytest

from PharmaPy.Commons import complete_dict_states
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit

# Synthetic contrast fixture, not a calibrated mixture: initial 4 kg contains
# 3.2 kg A and 0.8 kg B; the updated mixture contains 1 kg A and 3 kg B.
INITIAL_FRACTIONS = [0.8, 0.2]  # [-]
UPDATED_FRACTIONS = [0.25, 0.75]  # [-]
INITIAL_MASS = 4.0  # [kg], or [kg/s] for streams
INITIAL_VOLUME = 0.0045  # [m**3] = 3.2/800 + 0.8/1600 (per second for streams)
UPDATED_VOLUME = 0.003125  # [m**3] = 1/800 + 3/1600 (per second for streams)
UPDATED_MOLES = 70.0  # [mol] = 1/0.1 + 3/0.05 (per second for streams)
RTOL = 1e-12  # [-], roundoff allowance for short float64 mixture calculations


@pytest.fixture
def thermo_path(tmp_path):
    """Create a constant-property two-species contract fixture.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Repository-local pytest temporary directory.

    Returns
    -------
    str
        Path to the synthetic thermophysical database.
    """
    database = {
        "A": {
            "mw": 100.0,  # [g/mol]
            "rho_liq": 800.0,  # [kg/m**3]
            "rho_solid": 2000.0,  # [kg/m**3]
            # Unused Antoine placeholder required by the phase constructor.
            "p_vap": [0.0, 0.0, 0.0],  # [A [-], B [K], C [K]]
            "cp_liq": [100.0],  # [J/mol/K], constant polynomial
            "cp_solid": [100.0],  # [J/mol/K], constant polynomial
        },
        "B": {
            "mw": 50.0,  # [g/mol]
            "rho_liq": 1600.0,  # [kg/m**3]
            "rho_solid": 2000.0,  # [kg/m**3]
            # Unused Antoine placeholder required by the phase constructor.
            "p_vap": [0.0, 0.0, 0.0],  # [A [-], B [K], C [K]]
            "cp_liq": [100.0],  # [J/mol/K], constant polynomial
            "cp_solid": [100.0],  # [J/mol/K], constant polynomial
        },
    }
    path = tmp_path / "liquid_inventory.json"
    path.write_text(json.dumps(database), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("initial_basis", ["mass", "vol", "moles"])
@pytest.mark.parametrize("composition", ["mass_frac", "mole_frac",
                                         "mass_conc", "mole_conc"])
def test_composition_only_update_conserves_mass(thermo_path, initial_basis,
                                               composition):
    """Conserve mass across all supported composition and initial amount bases.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    initial_basis : str
        Initial amount basis: mass [kg], volume [m**3], or moles [mol].
    composition : str
        Updated composition basis: fractions [-], mass concentration [kg/m**3],
        or molar concentration [mol/L].
    """
    amounts = {"mass": INITIAL_MASS, "vol": INITIAL_VOLUME,
               "moles": 48.0}  # [kg], [m**3], [mol] = 3.2/0.1 + 0.8/0.05
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        **{initial_basis: amounts[initial_basis]})
    compositions = {
        "mass_frac": UPDATED_FRACTIONS,  # [-]
        "mole_frac": [1 / 7, 6 / 7],  # [-], 10 mol A and 60 mol B
        "mass_conc": [320.0, 960.0],  # [kg/m**3], 1 and 3 kg in 0.003125 m**3
        "mole_conc": [3.2, 19.2],  # [mol/L], 10 and 60 mol in 3.125 L
    }
    phase.updatePhase(**{composition: compositions[composition]})
    np.testing.assert_allclose(phase.mass_frac, UPDATED_FRACTIONS,
                               rtol=RTOL, atol=0)
    assert phase.mass == pytest.approx(INITIAL_MASS, rel=RTOL)
    assert phase.vol == pytest.approx(UPDATED_VOLUME, rel=RTOL)
    assert phase.moles == pytest.approx(UPDATED_MOLES, rel=RTOL)


@pytest.mark.parametrize("amounts", [
    {"mass": INITIAL_MASS, "vol": 1.0, "moles": 1.0},  # [kg], [m**3], [mol]
    {"vol": UPDATED_VOLUME, "moles": 1.0},  # [m**3], [mol]; volume wins
    {"moles": UPDATED_MOLES},  # [mol]
])
def test_explicit_amount_precedence(thermo_path, amounts):
    """Prefer explicit mass, then volume, then moles over retained inventory.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    amounts : dict
        Explicit mass [kg], volume [m**3], and/or moles [mol]. Unit-valued
        subordinate inputs deliberately conflict with the authoritative amount.
    """
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        mass=INITIAL_MASS)
    phase.updatePhase(mass_frac=UPDATED_FRACTIONS, **amounts)
    assert phase.mass == pytest.approx(INITIAL_MASS, rel=RTOL)
    assert phase.vol == pytest.approx(UPDATED_VOLUME, rel=RTOL)
    assert phase.moles == pytest.approx(UPDATED_MOLES, rel=RTOL)


def test_phase_no_argument_update_is_noop(thermo_path):
    """Leave stored state untouched when no update is requested.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    """
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        mass=INITIAL_MASS)
    before = vars(phase).copy()
    phase.updatePhase()
    assert vars(phase).keys() == before.keys()
    assert all(vars(phase)[name] is value for name, value in before.items())


@pytest.mark.parametrize("initial_mass", [0.0, INITIAL_MASS])  # [kg]
def test_intensive_only_update_does_not_return_early(thermo_path, initial_mass):
    """Apply temp/pres-only updates despite the no-argument early return.

    Density is independent of temperature and pressure in this model, so this
    test guards the early-return exception, not inventory recomputation.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    initial_mass : float
        Liquid inventory [kg].
    """
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        mass=initial_mass, check_input=False)
    temperature = 310.0  # [K], changed state with temperature-independent density
    pressure = 200000.0  # [Pa], changed state with pressure-independent density
    phase.updatePhase(temp=temperature, pres=pressure)
    assert phase.mass == pytest.approx(initial_mass, rel=RTOL)
    assert phase.vol == pytest.approx(initial_mass * INITIAL_VOLUME / INITIAL_MASS,
                                      rel=RTOL)
    assert phase.temp == temperature
    assert phase.pres == pressure


@pytest.mark.parametrize("flow,expected_mass", [
    ("vol_flow", 1 / (0.8 / 800 + 0.2 / 1600)),  # [kg/s] at 1 m**3/s
    ("mole_flow", 1 / (0.8 / 0.1 + 0.2 / 0.05)),  # [kg/s] at 1 mol/s
    ("mass_flow", 1.0),  # [kg/s]
])
def test_explicit_stream_flow_and_repeated_updates(thermo_path, flow,
                                                  expected_mass):
    """Reconcile flows after explicit, composition-only, and empty updates.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    flow : str
        Explicit flow alias, on the volume [m**3/s], molar [mol/s], or mass
        [kg/s] basis.
    expected_mass : float
        Independently derived mass flow [kg/s] for a unit requested flow.
    """
    stream = LiquidStream(thermo_path, mass_frac=INITIAL_FRACTIONS,
                          mass_flow=INITIAL_MASS)
    requested_flow = 1.0  # [m**3/s], [mol/s], or [kg/s], selected by flow
    stream.updatePhase(**{flow: requested_flow})
    assert getattr(stream, flow) == pytest.approx(requested_flow, rel=RTOL)
    assert stream.mass_flow == pytest.approx(expected_mass, rel=RTOL)
    assert stream.vol_flow == pytest.approx(
        expected_mass * INITIAL_VOLUME / INITIAL_MASS, rel=RTOL)
    specific_moles = 12.0  # [mol/kg] = 0.8/0.1 + 0.2/0.05
    assert stream.mole_flow == pytest.approx(expected_mass * specific_moles,
                                              rel=RTOL)
    # A second update must use flow storage
    # because the first one deleted the inherited phase amount attributes.
    assert not any(hasattr(stream, name) for name in ("mass", "vol", "moles"))
    stream.updatePhase(mass_frac=UPDATED_FRACTIONS)
    assert stream.mass_flow == pytest.approx(expected_mass, rel=RTOL)
    assert stream.vol_flow == pytest.approx(
        expected_mass * UPDATED_VOLUME / INITIAL_MASS, rel=RTOL)
    assert stream.mole_flow == pytest.approx(
        expected_mass * UPDATED_MOLES / INITIAL_MASS, rel=RTOL)
    before = vars(stream).copy()
    stream.updatePhase()
    assert vars(stream).keys() == before.keys()
    assert all(vars(stream)[name] is value for name, value in before.items())


def volume_control(time):
    """Return a constant flow to exercise the constructor control handoff.

    Parameters
    ----------
    time : float
        Elapsed time [s].

    Returns
    -------
    float
        Unit volumetric flow [m**3/s], chosen to expose stale mass precedence.
    """
    return 1.0


def test_constructor_volume_control_overrides_initial_mass_flow(thermo_path):
    """Apply a volume control even when the constructor supplied mass flow.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    """
    stream = LiquidStream(thermo_path, mass_frac=INITIAL_FRACTIONS,
                          mass_flow=INITIAL_MASS,
                          controls={"vol_flow": volume_control})
    assert stream.vol_flow == pytest.approx(volume_control(0), rel=RTOL)
    assert stream.mass_flow == pytest.approx(
        INITIAL_MASS / INITIAL_VOLUME, rel=RTOL)


@pytest.mark.parametrize("updated_liquid", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_slurry_from_phases_publishes_phase_flow_sums(thermo_path, updated_liquid,
                                                     reverse):
    """Publish constituent sums regardless of ordering or prior liquid updates.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    updated_liquid : bool
        Update the liquid first, deleting its inherited amount attributes.
    reverse : bool
        Attach the solid before the liquid to exercise classification order.
    """
    liquid = LiquidStream(thermo_path, mass_frac=INITIAL_FRACTIONS,
                          mass_flow=INITIAL_MASS)
    if updated_liquid:
        liquid.updatePhase(mass_frac=UPDATED_FRACTIONS)
    solid_mass_flow = 2.0  # [kg/s], pure A occupies 0.001 m**3/s
    grid = [10.0, 20.0, 30.0, 40.0]  # [um], four-bin asymmetric population
    distribution = [0.0, 2.0, 1.0, 0.0]  # [#/um], normalized to explicit mass
    solid = SolidStream(thermo_path, mass_frac=[1.0, 0.0],
                        mass_flow=solid_mass_flow, x_distrib=grid,
                        distrib=distribution)
    slurry = SlurryStream()
    slurry.Phases = [solid, liquid] if reverse else [liquid, solid]
    expected_volume = (UPDATED_VOLUME if updated_liquid
                       else INITIAL_VOLUME)  # [m**3/s]
    expected_volume += 0.001  # [m**3/s], exact solid volume from 2/2000
    assert slurry.Liquid_1 is liquid and slurry.Solid_1 is solid
    assert slurry.vol_flow == pytest.approx(expected_volume, rel=RTOL)
    assert slurry.mass_flow == pytest.approx(INITIAL_MASS + solid_mass_flow,
                                              rel=RTOL)
    assert slurry.vol == slurry.vol_flow
    assert slurry.mass_slurry == slurry.mass_flow

    # Reattaching uses the existing moment-based specification. It must refresh
    # aliases even when the requested aggregate volume changes.
    slurry.vol *= 2  # [m**3/s], deliberate doubling of the prescribed flow
    slurry.Phases = [solid, liquid] if reverse else [liquid, solid]
    assert slurry.vol_flow == pytest.approx(2 * expected_volume, rel=RTOL)
    assert slurry.vol_flow == pytest.approx(liquid.vol_flow + solid.vol,
                                             rel=RTOL)
    assert slurry.mass_flow == pytest.approx(liquid.mass_flow + solid.mass_flow,
                                              rel=RTOL)


def test_slurry_distribution_mass_basis_refreshes_volume_flow(thermo_path):
    """Publish derived volume flow when distribution input specifies total mass.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.

    Notes
    -----
    No public constructor input reaches the ``mass_slurry > 0`` branch, so
    this test sets that attribute directly before assigning the phases.
    """
    liquid = LiquidStream(thermo_path, mass_frac=INITIAL_FRACTIONS,
                          check_input=False)
    shape_factor = 0.5  # [-], non-unit factor distinguishes moment from volume
    solid = SolidStream(thermo_path, mass_frac=[1.0, 0.0], kv=shape_factor)
    grid = [0.0, 100.0, 200.0, 300.0]  # [um]
    # Trapezoidal third moment: 100*(1e9*100**3 + 1.25e8*200**3)*1e-18
    # = 0.2 m**3/m**3, giving 0.1 solid volume fraction with kv=0.5.
    distribution = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
    slurry = SlurryStream(x_distrib=np.array(grid), distrib=distribution)
    slurry.mass_slurry = 6.0  # [kg/s], density 0.9*(4/0.0045)+0.1*2000=1000
    slurry.Phases = [liquid, solid]
    expected_volume_flow = 0.006  # [m**3/s] = 6/1000
    expected_liquid_flow = 4.8  # [kg/s] = 0.9*0.006*(4/0.0045)
    expected_solid_flow = 1.2  # [kg/s] = 0.1*0.006*2000
    assert slurry.vol_flow == pytest.approx(expected_volume_flow, rel=RTOL)
    assert slurry.vol_flow == pytest.approx(liquid.vol_flow + solid.vol,
                                             rel=RTOL)
    assert liquid.mass_flow == pytest.approx(expected_liquid_flow, rel=RTOL)
    assert solid.mass_flow == pytest.approx(expected_solid_flow, rel=RTOL)
    assert slurry.mass_flow == pytest.approx(
        expected_liquid_flow + expected_solid_flow, rel=RTOL)


@pytest.mark.parametrize("initial_basis", ["vol", "moles"])
@pytest.mark.parametrize("update", ["composition", "uniform_composition",
                                    "temperature"])
def test_profile_inventory_is_unchanged(thermo_path, initial_basis, update):
    """Preserve baseline profile amounts for composition and temperature updates.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    initial_basis : str
        Unit volume [m**3] or mole amount [mol] used to construct the profile.
    update : str
        Change species fractions [-] to a profile or uniform composition,
        or change the temperature profile [K].
    """
    fractions = [INITIAL_FRACTIONS, UPDATED_FRACTIONS,
                 [0.6, 0.4]]  # [-], three asymmetric spatial nodes, two species
    phase = LiquidPhase(thermo_path, mass_frac=fractions,
                        **{initial_basis: 1.0})
    before = {name: getattr(phase, name) for name in ("mass", "vol", "moles")}
    assert np.shape(phase.mass) == (3,)
    if update == "composition":
        phase.updatePhase(mass_frac=fractions[::-1])
        np.testing.assert_allclose(phase.mass_frac, fractions[::-1],
                                   rtol=RTOL, atol=0)
    elif update == "uniform_composition":
        phase.updatePhase(mass_frac=UPDATED_FRACTIONS)
        np.testing.assert_allclose(phase.mass_frac, UPDATED_FRACTIONS,
                                   rtol=RTOL, atol=0)
    else:
        temperatures = [290.0, 300.0, 310.0]  # [K], distinct spatial values
        phase.updatePhase(temp=temperatures)
        np.testing.assert_array_equal(phase.temp, temperatures)
    for name, amount in before.items():  # [kg], [m**3], or [mol]
        assert getattr(phase, name) is amount


def test_washing_profile_keeps_scalar_inventory(thermo_path):
    """Keep scalar amounts when washing supplies a spatial composition field.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    """
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        mass=INITIAL_MASS)
    before = {name: getattr(phase, name) for name in ("mass", "vol", "moles")}
    concentrations = [[320.0, 960.0], [640.0, 320.0],
                      [160.0, 1280.0]]  # [kg/m**3], three zones, two species
    phase.updatePhase(mass_conc=concentrations)
    np.testing.assert_array_equal(phase.mass_conc, concentrations)
    for name, amount in before.items():  # [kg], [m**3], or [mol]
        assert np.ndim(getattr(phase, name)) == 0
        assert getattr(phase, name) is amount


@pytest.mark.parametrize("unit_type", [BatchReactor, BatchCryst])
@pytest.mark.parametrize("amount", [None, "vol", "mass", "moles"])
def test_paramest_modifier_preserves_charged_volume(thermo_path, monkeypatch,
                                                   unit_type, amount):
    """Exercise real wrapper reset and phase modification up to the solver call.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    monkeypatch : pytest.MonkeyPatch
        Replace only the expensive solve boundary; no backend is imported.
    unit_type : type
        Batch reactor or crystallizer whose estimation wrapper is exercised.
    amount : str or None
        Explicit volume [m**3], mass [kg], or moles [mol]; None requests only
        composition and must preserve the original charged volume [m**3].
    """
    liquid = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                         mass=INITIAL_MASS)
    if unit_type is BatchReactor:
        unit = BatchReactor(return_sens=False)
        unit.Phases = liquid
        # Synthetic mass-balanced A -> 2 B; kinetics are set but never evaluated.
        parameters = {"k_params": [1.0], "ea_params": [0.0]}  # [1/s], [J/mol]
        unit.Kinetics = RxnKinetics(thermo_path, **parameters,
                                   stoich_matrix=[[-1, 2]],
                                   partic_species=["A", "B"])
    else:
        unit = BatchCryst(target_comp="A")
        solid = SolidPhase(thermo_path, mass_frac=[1.0, 0.0],
                           x_distrib=[10.0, 20.0, 30.0, 40.0],  # [um]
                           distrib=[0.0, 2.0, 1.0, 0.0])  # [#/um]
        unit.Phases = [liquid, solid]
        unit.Kinetics = CrystKinetics()
        parameters = {}  # Default inactive mechanisms; no kinetics evaluated.

    modifier = {"mass_frac": UPDATED_FRACTIONS}  # [-]
    expected_volume = INITIAL_VOLUME  # [m**3], retain the charged volume
    if amount is not None:
        amounts = {"vol": 2 * INITIAL_VOLUME, "mass": INITIAL_MASS,
                   "moles": UPDATED_MOLES}  # [m**3], [kg], [mol]
        modifier[amount] = amounts[amount]
        expected_volume = (2 * INITIAL_VOLUME if amount == "vol"
                           else UPDATED_VOLUME)  # [m**3]
    supplied_keys = set(modifier)
    modifications = modifier if unit_type is BatchReactor else {"Liquid": modifier}
    solve_calls = []

    def stop_before_solve(**kwargs):
        """Record the actual handoff and stop before optional solver execution.

        Parameters
        ----------
        **kwargs : dict
            Solver options, including the requested time grid [s].

        Raises
        ------
        RuntimeError
            Always, with a sentinel identifying the reached solver boundary.
        """
        solve_calls.append(kwargs)
        raise RuntimeError("liquid inventory probe stopped at solve boundary")

    monkeypatch.setattr(unit, "solve_unit", stop_before_solve)
    time_grid = np.array([0.0, 1.0])  # [s], dummy start and end; never integrated
    with pytest.raises(RuntimeError, match="liquid inventory probe stopped at solve boundary"):
        unit.paramest_wrapper(parameters, time_grid, modify_phase=modifications)
    assert len(solve_calls) == 1
    assert solve_calls[0]["time_grid"] is time_grid
    assert unit.Liquid_1.vol == pytest.approx(expected_volume, rel=RTOL)
    assert unit.Liquid_1.mass == pytest.approx(
        expected_volume * INITIAL_MASS / UPDATED_VOLUME, rel=RTOL)
    np.testing.assert_allclose(unit.Liquid_1.mass_frac, UPDATED_FRACTIONS,
                               rtol=RTOL, atol=0)
    assert set(modifier) == supplied_keys


def test_completed_states_reads_reconciled_liquid_volume(thermo_path):
    """Expose the conserved-mass volume change through the balance state reader.

    Parameters
    ----------
    thermo_path : str
        Path to the synthetic thermophysical database.
    """
    phase = LiquidPhase(thermo_path, mass_frac=INITIAL_FRACTIONS,
                        mass=INITIAL_MASS)
    before = complete_dict_states(0.0, {}, ("vol",), phase, {})  # vol [m**3]
    phase.updatePhase(mass_frac=UPDATED_FRACTIONS)
    after = complete_dict_states(0.0, {}, ("vol",), phase, {})  # vol [m**3]
    assert before["vol"] == pytest.approx(INITIAL_VOLUME, rel=RTOL)
    assert after["vol"] == pytest.approx(UPDATED_VOLUME, rel=RTOL)
    assert after["vol"] < before["vol"]
