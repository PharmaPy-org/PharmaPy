"""Export a solved two-reactor batch sequence as a fixed-ratio STN instance."""

import argparse
import json
from pathlib import Path

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.Scheduling import BatchTask, export_batch_stn
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater


def build_stn() -> dict:
    """Solve a reference batch and export its scheduling instance.

    Returns
    -------
    dict
        Fixed-ratio STN data using bulk liquid material states [kg], physical
        processing and grid width [s], and discrete scheduling periods [-].

    Notes
    -----
    The reaction rate, charge and operation times define a synthetic
    scheduling demonstration. The two reactors advance the same
    mass-conserving reaction at fixed temperature and liquid volume.
    Each STN start represents one replay of a complete reference batch.
    Feed, intermediate and product refer to whole mixtures, including
    solvent and any unreacted species. Transfer and turnaround times are
    assumed zero, and storage holds at most one reference batch in each state.
    """
    checkout = Path(__file__).resolve().parents[3]
    properties = str(checkout / "tests/integration/data/pfr_test_pure_comp.json")

    temperature = 298.15  # [K], isothermal operation at the kinetics reference
    liquid_volume = 0.002  # [m**3], nominal volume of the shipped reactor case
    # [mol/L], A, B, C and solvent in database order; excess B prevents symmetry.
    concentrations = [0.1, 0.2, 0.0, 5.0]
    # [L/mol/s], synthetic rate giving measurable conversion in both stages.
    rate_constant = 0.04
    activation_energy = 0.0  # [J/mol], no thermal variation in this example
    utility_flow = 0.1  # [kg/s], completes the unchanged isothermal unit setup
    # [s], unequal durations chosen to round up to two and three grid periods.
    first_duration, second_duration = 25.0, 43.0
    period_seconds = 20.0  # [s], scheduling grid chosen for the rounding example
    horizon_periods = 5  # [-], exactly two plus three grid periods
    first_task, second_task = "R01", "R02"

    simulation = SimulationExec(
        properties, flowsheet={first_task: [second_task], second_task: []})
    for task in simulation.execution_names:
        reactor = BatchReactor(isothermal=True)
        reactor.Kinetics = RxnKinetics(
            properties, k_params=[rate_constant], ea_params=[activation_energy],
            rxn_list=["A + B --> C"], temp_ref=temperature,
            reformulate_kin=False)
        reactor.Utility = CoolingWater(
            temp_in=temperature, mass_flow=utility_flow)
        setattr(simulation, task, reactor)

    feed = LiquidPhase(properties, temp=temperature, vol=liquid_volume,
                       mole_conc=concentrations)
    feed_kg = feed.mass  # [kg], save the charge before reactor updates its phase
    simulation.R01.Phases = feed
    simulation.SolveFlowsheet(
        kwargs_run={
            first_task: {"runtime": first_duration, "verbose": False},
            second_task: {"runtime": second_duration, "verbose": False},
        },
        verbose=False,
    )

    intermediate_kg = simulation.R01.Outlet.mass  # [kg], first liquid batch
    product_kg = simulation.R02.Outlet.mass  # [kg], second liquid batch
    tasks = {
        first_task: BatchTask(
            resource="reactor-1", inputs_kg={"feed": feed_kg},
            outputs_kg={"intermediate": intermediate_kg}),
        second_task: BatchTask(
            resource="reactor-2",
            inputs_kg={"intermediate": intermediate_kg},
            outputs_kg={"product": product_kg}),
    }  # task amounts [kg] on each complete liquid-mixture basis
    storage_capacities_kg = {
        "feed": feed_kg,
        "intermediate": intermediate_kg,
        "product": product_kg,
    }  # [kg], one reference batch of each material state
    return export_batch_stn(
        simulation, tasks, period_seconds=period_seconds,
        horizon_periods=horizon_periods,
        storage_capacities_kg=storage_capacities_kg,
        initial_inventory_kg={"feed": feed_kg},  # [kg], all other states empty
        demand_kg={("product", horizon_periods): product_kg},  # [kg], one batch
    )


def main() -> None:
    """Write the solved fixed-ratio STN data to the requested JSON file.

    Notes
    -----
    The command takes one positional argument, the output path. Run it from
    a PharmaPy checkout in the Assimulo environment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path for the STN JSON file")
    output = parser.parse_args().output
    output.write_text(
        json.dumps(build_stn(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
