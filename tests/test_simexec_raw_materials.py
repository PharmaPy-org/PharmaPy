"""Raw-material accounting through production flowsheet collaborators."""

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Commons import trapezoidal_rule
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Results import DynamicResult
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream, SolidStream


pytestmark = pytest.mark.unit


def _thermo_path(data_path):
    """Return the production thermodynamic database used by these tests.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.

    Returns
    -------
    str
        Path to a four-species thermodynamic JSON database.
    """
    return str(data_path["integration"] / "pfr_test_pure_comp.json")


def _liquid_stream(data_path, *, mass_flow=0.0, mole_flow=0.0,
                   mass_frac=None, mole_frac=None):
    """Create a production liquid stream for raw-material accounting.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    mass_flow : float, optional
        Liquid mass flow [kg/s].
    mole_flow : float, optional
        Liquid molar flow [mol/s].
    mass_frac : array_like, optional
        Four-species mass fractions [-].
    mole_frac : array_like, optional
        Four-species mole fractions [-].

    Returns
    -------
    LiquidStream
        Production liquid stream at 300 [K] and 101325 [Pa].
    """
    return LiquidStream(
        _thermo_path(data_path),
        temp=300.0,  # [K]
        pres=101325.0,  # [Pa]
        mass_flow=mass_flow,  # [kg/s]
        mole_flow=mole_flow,  # [mol/s]
        mass_frac=mass_frac,
        mole_frac=mole_frac,
        verbose=False,
    )


def _liquid_phase(data_path, *, mass, mass_frac):
    """Create a production liquid phase for batch or holdup accounting.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    mass : float
        Liquid mass [kg].
    mass_frac : array_like
        Four-species mass fractions [-].

    Returns
    -------
    LiquidPhase
        Production liquid phase at 300 [K] and 101325 [Pa].
    """
    return LiquidPhase(
        _thermo_path(data_path),
        temp=300.0,  # [K]
        pres=101325.0,  # [Pa]
        mass=mass,  # [kg]
        mass_frac=mass_frac,
        verbose=False,
    )


def _collector_with_inlet(inlet, time=(0.0, 10.0)):
    """Attach a real inlet and solved result to a production collector.

    Parameters
    ----------
    inlet : LiquidStream, LiquidPhase, or SlurryStream
        Production raw-material inlet.
    time : tuple of float, optional
        Simulated start and end times [s].

    Returns
    -------
    DynamicCollector
        Production collector with a real ``DynamicResult`` time vector.
    """
    collector = DynamicCollector()
    collector.Inlet = inlet
    collector.oper_mode = (
        "Continuous" if hasattr(inlet, "mass_flow") else "Batch"
    )
    collector.result = DynamicResult(
        {}, time=np.asarray(time, dtype=float)
    )  # [s]
    collector.heat_duty = np.zeros(2)  # [J]
    collector.duty_type = np.zeros(2, dtype=int)  # [-]
    collector.outputs = {}
    return collector


def _sim_with_units(data_path, units):
    """Build a production simulation executor around solved unit operations.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    units : dict
        Unit-operation instances keyed by flowsheet identifier.

    Returns
    -------
    SimulationExec
        Production executor configured with the real species database.
    """
    graph = {name: [] for name in units}
    simulation = SimulationExec(_thermo_path(data_path), graph)
    simulation.uos_instances = units
    return simulation


def _sim_with_inlet(data_path, inlet, time=(0.0, 10.0)):
    """Build a one-mixer production simulation for a raw inlet.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    inlet : LiquidStream, LiquidPhase, or SlurryStream
        Production raw-material inlet.
    time : tuple of float, optional
        Simulated start and end times [s].

    Returns
    -------
    SimulationExec
        Production executor containing one dynamic collector.
    """
    unit = _collector_with_inlet(inlet, time=time)
    return _sim_with_units(data_path, {"U01": unit})


def _slurry_stream(data_path):
    """Create a production slurry stream with unequal phase flow rates.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.

    Returns
    -------
    SlurryStream
        Liquid/solid stream with phase mass flows of 1 and 3 [kg/s].
    """
    liquid = _liquid_stream(
        data_path,
        mass_flow=1.0,  # [kg/s]
        mass_frac=[0.8, 0.2, 0.0, 0.0],  # [-]
    )
    particle_size = np.array([1.0, 2.0, 3.0, 4.0])  # [um]
    distribution_shape = np.array([1.0, 2.0, 2.0, 1.0])  # [-]
    unscaled_third_moment = trapezoidal_rule(
        particle_size,
        distribution_shape * particle_size**3,
    ) * (1.0e-6)**3  # [m**3]
    target_solid_volume_fraction = 0.25  # [-]
    distribution = (
        distribution_shape
        * target_solid_volume_fraction
        / unscaled_third_moment
    )  # [#/m**3/um]
    solid = SolidStream(
        _thermo_path(data_path),
        temp=300.0,  # [K]
        pres=101325.0,  # [Pa]
        mass_flow=3.0,  # [kg/s]
        mass_frac=[0.1, 0.9, 0.0, 0.0],  # [-]
        x_distrib=particle_size,
        distrib=distribution,
    )
    slurry = SlurryStream(
        vol_flow=liquid.vol + solid.vol,  # [m**3/s]
        x_distrib=particle_size,
        distrib=distribution,
    )
    slurry.Phases = [liquid, solid]
    return slurry


def test_get_opex_passes_raw_material_keywords_and_holdup_flag(data_path):
    """Top-level raw-material flags take precedence in OPEX accounting."""
    inlet = _liquid_stream(
        data_path,
        mass_flow=2.0,  # [kg/s]
        mass_frac=[0.25, 0.75, 0.0, 0.0],  # [-]
    )
    unit = _collector_with_inlet(inlet)
    unit.__original_phase__ = _liquid_phase(
        data_path,
        mass=5.0,  # [kg]
        mass_frac=[0.4, 0.6, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_units(data_path, {"U01": unit})

    _, raw_cost, _ = simulation.GetOPEX(
        1.0,  # [USD/kg]
        include_holdups=False,
        kwargs_items={"raw_materials": {"basis": "mass"}},
    )

    expected_cost = [20.0, 5.0, 15.0, 0.0, 0.0]  # [USD]
    assert "Initial_holdup" not in raw_cost.index.get_level_values(1)
    assert list(raw_cost.columns) == [
        "mass", "mass_A", "mass_B", "mass_C", "mass_solv"
    ]
    np.testing.assert_allclose(raw_cost.iloc[0], expected_cost)


def test_get_opex_includes_initial_holdup_by_default(data_path):
    """OPEX includes initial holdup raw material unless disabled."""
    inlet = _liquid_stream(
        data_path,
        mass_flow=2.0,  # [kg/s]
        mass_frac=[0.25, 0.75, 0.0, 0.0],  # [-]
    )
    unit = _collector_with_inlet(inlet)
    unit.__original_phase__ = _liquid_phase(
        data_path,
        mass=5.0,  # [kg]
        mass_frac=[0.4, 0.6, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_units(data_path, {"U01": unit})

    _, raw_cost, _ = simulation.GetOPEX(
        1.0,  # [USD/kg]
        kwargs_items={"raw_materials": {"basis": "mass"}},
    )

    holdup_cost = raw_cost.xs("Initial_holdup", level=1)
    expected_holdup_cost = [5.0, 2.0, 3.0, 0.0, 0.0]  # [USD]
    assert len(holdup_cost) == 1
    np.testing.assert_allclose(holdup_cost.iloc[0], expected_holdup_cost)


def test_get_opex_maps_duty_codes_to_heat_exchange_costs(data_path):
    """OPEX maps duty codes directly to their heat-exchange costs."""
    duty_types = np.array([[-3, -2], [-1, 0], [1, 2], [3, 0]], dtype=int)
    units = {}
    for number, duty_type in enumerate(duty_types):
        inlet = _liquid_stream(
            data_path,
            mass_flow=1.0,  # [kg/s]
            mass_frac=[1.0, 0.0, 0.0, 0.0],  # [-]
        )
        unit = _collector_with_inlet(inlet)
        unit.heat_duty = np.array([1.0e9, 1.0e9])  # [J]
        unit.duty_type = duty_type  # [-]
        units[f"U{number:02d}"] = unit
    simulation = _sim_with_units(data_path, units)

    duty_cost, _, _ = simulation.GetOPEX(0.0, include_holdups=False)

    expected_cost = np.array([
        [14.12, 8.49],
        [4.77, 0.378],
        [4.54, 4.77],
        [5.66, 0.378],
    ])  # [USD], one [GJ] per duty entry
    np.testing.assert_allclose(duty_cost.to_numpy(), expected_cost)


def test_mole_basis_totals_use_canonical_singular_basis(data_path):
    """Mole-basis totals use the singular ``basis='mole'`` path."""
    inlet = _liquid_stream(
        data_path,
        mole_flow=4.0,  # [mol/s]
        mole_frac=[0.25, 0.75, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_inlet(data_path, inlet)

    raw_materials = simulation.GetRawMaterials(basis="mole")

    assert list(raw_materials.columns) == [
        "moles", "moles_A", "moles_B", "moles_C", "moles_solv"
    ]
    np.testing.assert_allclose(
        raw_materials.iloc[0],
        [40.0, 10.0, 30.0, 0.0, 0.0],  # [mol]
    )


def test_batch_raw_inlet_records_total_before_aggregation(data_path):
    """Batch raw inlets record total material before species aggregation."""
    inlet = _liquid_phase(
        data_path,
        mass=6.0,  # [kg]
        mass_frac=[0.2, 0.8, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_inlet(data_path, inlet)

    raw_materials = simulation.GetRawMaterials(basis="mass")

    assert list(raw_materials.columns) == [
        "mass", "mass_A", "mass_B", "mass_C", "mass_solv"
    ]
    np.testing.assert_allclose(
        raw_materials.iloc[0],
        [6.0, 1.2, 4.8, 0.0, 0.0],  # [kg]
    )


def test_mixed_phase_raw_inlet_is_decomposed_by_phase(data_path):
    """Static mixed raw inlets account each production phase once."""
    slurry = _slurry_stream(data_path)
    expected_phase_mass = np.sort([
        phase.mass_flow * 10.0 for phase in slurry.Phases  # [kg]
    ])
    simulation = _sim_with_inlet(data_path, slurry)

    raw_materials = simulation.GetRawMaterials(basis="mass", totals=False)

    assert len(raw_materials) == 2
    np.testing.assert_allclose(
        np.sort(raw_materials["mass"].to_numpy()),
        expected_phase_mass,
    )


def test_dynamic_mixed_phase_raw_inlet_splits_total_flow_by_phase(data_path):
    """Dynamic mixed inlets split rather than duplicate their total flow."""
    slurry = _slurry_stream(data_path)
    phase_fractions = np.array([
        phase.mass_flow / slurry.mass_flow for phase in slurry.Phases
    ])  # [-]
    dynamic_inlet = DynamicInput()
    dynamic_inlet.add_variable(
        "mass_flow",
        lambda time: np.full_like(time, 4.0, dtype=float),  # [kg/s]
    )
    slurry.DynamicInlet = dynamic_inlet
    simulation = _sim_with_inlet(data_path, slurry)

    raw_materials = simulation.GetRawMaterials(basis="mass", totals=False)

    expected_phase_mass = np.sort(4.0 * 10.0 * phase_fractions)  # [kg]
    assert len(raw_materials) == 2
    np.testing.assert_allclose(
        np.sort(raw_materials["mass"].to_numpy()),
        expected_phase_mass,
    )


def test_get_opex_applies_nonunity_raw_cost_vector(data_path):
    """Vector raw costs price total and species columns explicitly."""
    inlet = _liquid_stream(
        data_path,
        mass_flow=2.0,  # [kg/s]
        mass_frac=[0.25, 0.75, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_inlet(data_path, inlet)

    _, raw_cost, _ = simulation.GetOPEX(
        np.array([0.0, 2.0, 3.0, 0.0, 0.0]),  # [USD/kg]
        include_holdups=False,
        kwargs_items={"raw_materials": {"basis": "mass"}},
    )

    np.testing.assert_allclose(
        raw_cost.iloc[0],
        [0.0, 10.0, 45.0, 0.0, 0.0],  # [USD]
    )


def test_get_opex_rejects_raw_cost_vector_with_wrong_width(data_path):
    """Raw cost vectors must match the raw-material table width."""
    inlet = _liquid_stream(
        data_path,
        mass_flow=2.0,  # [kg/s]
        mass_frac=[0.25, 0.75, 0.0, 0.0],  # [-]
    )
    simulation = _sim_with_inlet(data_path, inlet)

    with pytest.raises(ValueError, match="one entry per raw-material column"):
        simulation.GetOPEX(
            np.array([1.0, 2.0]),  # [USD/kg]
            include_holdups=False,
            kwargs_items={"raw_materials": {"basis": "mass"}},
        )
