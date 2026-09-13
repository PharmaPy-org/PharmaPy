"""Stream tables retain quantities, species, and flowsheet order with pandas 3.

Real batch and continuous Mixer chains cover the public result handoff without
Assimulo. Pure feeds of A, solvent, and B have unequal amounts; reversing the
thermo database order checks that composition columns retain their meaning.
Instantaneous continuous mixing has zero duration, so its raw usage is zero
while its reported flow rates are nonzero. Separation holdup issue #311 is
outside these liquid-only fixtures.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit
TEMPERATURE = 300.0  # [K], common feed temperature isolates material reporting
PRESSURE = 101325.0  # [Pa], atmospheric fixture pressure
REL_TOL = 1e-12  # [-], roundoff allowance for algebraic mixing and conversions
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact SI conversion


@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("continuous", [False, True], ids=["batch", "continuous"])
@pytest.mark.parametrize("reverse_species", [False, True], ids=["species", "reversed"])
def test_stream_table_preserves_order_and_material_basis(
    data_path, tmp_path, basis, continuous, reverse_species,
):
    source = data_path["integration"] / "pfr_test_pure_comp.json"
    database = json.loads(source.read_text())
    if reverse_species:
        database = dict(reversed(list(database.items())))
    thermo_path = tmp_path / "thermo.json"
    thermo_path.write_text(json.dumps(database))
    species = list(database)

    constructor = LiquidStream if continuous else LiquidPhase
    quantity_key = "mass_flow" if continuous else "mass"
    # Unequal pure feeds make both mixing stages and every active species
    # distinguishable. The same numbers represent kg or kg/s in the two modes.
    feed_quantities = {"A": 2.0, "solv": 3.0, "B": 5.0}  # [kg] or [kg/s]
    feeds = []
    for name, quantity in feed_quantities.items():  # quantity [kg] or [kg/s]
        fractions = [float(item == name) for item in species]  # [-], pure feed
        feeds.append(constructor(
            str(thermo_path), temp=TEMPERATURE, pres=PRESSURE,
            mass_frac=fractions, **{quantity_key: quantity},
        ))

    sim = SimulationExec(str(thermo_path), {"Z_FIRST": ["A_SECOND"], "A_SECOND": []})
    sim.Z_FIRST = Mixer()
    sim.A_SECOND = Mixer()
    sim.Z_FIRST.Inlets = feeds[:2]
    sim.A_SECOND.Inlets = feeds[2]
    sim.SolveFlowsheet(verbose=False)

    table = sim.result.GetStreamTable(basis=basis)
    assert table.index.nlevels == 3
    assert table.index.droplevel(2).tolist() == [
        ("Z_FIRST", "Inlet_0"), ("Z_FIRST", "Inlet_1"), ("Z_FIRST", "Outlet"),
        ("A_SECOND", "Inlet_0"), ("A_SECOND", "Outlet"),
    ]

    # Hand material ledger in exactly the expected row order. This does not
    # use Mixer outputs or PharmaPy's fraction/basis conversion routines.
    component_mass = [  # [kg] or [kg/s], named component amounts per row
        {"A": 2.0}, {"solv": 3.0}, {"A": 2.0, "solv": 3.0},
        {"B": 5.0}, {"A": 2.0, "B": 5.0, "solv": 3.0},
    ]
    component_mass_values = np.array([  # [kg] or [kg/s], rows above, database columns
        [row.get(name, 0.0) for name in species] for row in component_mass
    ])
    molecular_weights = np.array([database[name]["mw"] for name in species])  # [g/mol]
    component_moles_values = (  # [mol] or [mol/s], independent component conversion
        component_mass_values * GRAMS_PER_KILOGRAM / molecular_weights
    )
    basis_amounts = {  # mass: [kg] or [kg/s]; mole: [mol] or [mol/s]
        "mass": component_mass_values, "mole": component_moles_values,
    }[basis]
    expected_totals = basis_amounts.sum(axis=1)  # [kg, mol] or [kg/s, mol/s]
    expected_fractions = basis_amounts / expected_totals[:, None]  # [-], species fractions

    fraction_columns = [f"{basis}_frac_{name}" for name in species]
    assert [name for name in table.columns if "_frac_" in name] == fraction_columns
    np.testing.assert_allclose(table[fraction_columns], expected_fractions, rtol=REL_TOL, atol=0)
    reported_quantity = ("mass" if basis == "mass" else "moles")
    if continuous:
        reported_quantity = f"{basis}_flow"
        raw_rows = table.index.get_level_values(1) != "Outlet"
        # A single-time static Mixer has no integration interval [s].
        inventory_column = "mass" if basis == "mass" else "moles"
        np.testing.assert_array_equal(table.loc[raw_rows, inventory_column], 0)
    np.testing.assert_allclose(table[reported_quantity], expected_totals, rtol=REL_TOL, atol=0)
    np.testing.assert_allclose(table["temp"], TEMPERATURE, rtol=REL_TOL, atol=0)
    np.testing.assert_allclose(table["pres"], PRESSURE, rtol=REL_TOL, atol=0)
