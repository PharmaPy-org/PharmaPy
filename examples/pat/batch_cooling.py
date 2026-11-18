"""Seeded batch cooling crystallization of a described dataset, on either batch crystallizer.

Usage, from the repository root::

    python -m examples.pat.batch_cooling "<dataset folder>" <run id> <legacy|refactored>

The model starts at the seeding time of the run: the whole charge dissolved, the
seed as crystals of one size, and the measured temperature as a prescribed
control. It runs on the legacy ``BatchCryst`` or the refactored
``BatchCrystallizer``, both with the method of moments, and returns a
``PharmaPy.SlurryMeasurements.SlurryState`` with one row per requested time.

The dataset description supplies the charge, the seed, the solid, the property
database and starting values of the kinetics; ``solubility.json`` supplies the
solubility curve. The kinetic constants in a description are starting values for a
fit, not results.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from PharmaPy import SlurryMeasurements

from . import dataset, solubility

MODELS = ("legacy", "refactored")
MOMENT_COUNT = 4  # orders 0 to 3: the least that gives the crystal volume
GRAMS_PER_KILOGRAM = 1e3  # [g/kg], exact
MICROMETERS_PER_METER = 1e6  # [um/m], exact
# A requested time counts as returned when the solver reports a time this close to
# it. The solver reports requested output times themselves, so this only absorbs
# floating-point representation of times of order 1e4 s.
TIME_MATCH_TOLERANCE = 1e-6  # [s]


@dataclass(frozen=True)
class CoolingCase:
    """Inputs of one seeded batch cooling crystallization.

    Parameters
    ----------
    database : str
        Path of the property database.
    solute, solvent : str
        Names of the crystallizing species and of the solvent in the database.
    solute_mass, solvent_mass : float
        Dissolved solute and solvent at the start [kg].
    seed_mass : float
        Solute added as seed crystals [kg].
    seed_size : float
        Characteristic length of the seed crystals, all of one size [m].
    shape_factor : float
        Crystal volume over the cube of the characteristic length [-].
    solute_density : float
        Mass of solute per crystal volume [kg/m**3].
    temperature : callable
        Temperature of the slurry [K] at a model time [s], counted from seeding.
    saturated_ratio : callable
        Solute per solvent at saturation [kg/kg] at a temperature [K].
    """

    database: str
    solute: str
    solvent: str
    solute_mass: float
    solvent_mass: float
    seed_mass: float
    seed_size: float
    shape_factor: float
    solute_density: float
    temperature: Callable
    saturated_ratio: Callable


def load_case(case: Path, run_id: str, treatment: str) -> tuple[CoolingCase, dict]:
    """Read the inputs of one run from its dataset.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder, holding ``runs.json`` and ``solubility.json``.
    run_id : str
        Run identifier from the description.
    treatment : str
        Treatment of the band calibration whose pooled solubility curve is used.

    Returns
    -------
    inputs : CoolingCase
        The inputs. Model time zero is the seeding time of the run.
    kinetics : dict
        Starting values from the description: ``growth`` ``(k_g [um/s], E
        [J/mol], g [-])`` and ``secondary_nucleation`` ``(k_s [#/m**3/s], E
        [J/mol], s_1 [-], s_2 [-])``.

    Raises
    ------
    ValueError
        If the run or the treatment is not in the dataset.
    """
    case = Path(case)
    description = dataset.load_description(case)
    runs = {run["id"]: run for run in description["runs"]}
    if run_id not in runs:
        raise ValueError("Unknown run %r; the description lists %s." % (run_id, sorted(runs)))
    curves = json.loads((case / "solubility.json").read_text(encoding="utf-8"))["treatments"]
    if treatment not in curves:
        raise ValueError("Unknown treatment %r; solubility.json holds %s." % (treatment, sorted(curves)))
    curve = curves[treatment]["pooled"]
    settings = description["simulation"]
    seeding_time = runs[run_id]["seeding_time_s"]  # [s] on the run clock
    log = dataset.read_reduced(case, run_id, settings["temperature_table"])
    log_time = log["time"] - seeding_time  # [s] model time
    log_temperature = log[settings["temperature_column"]]  # [K]

    def temperature(time):
        """Return the logged temperature [K] at a model time [s]; it is held at the ends of the log."""
        return np.interp(time, log_time, log_temperature)

    def saturated_ratio(temp):
        """Return the solute per solvent at saturation [kg/kg] at a temperature [K]."""
        return solubility.saturated_ratio(temp, curve["intercept"], curve["slope_K"])

    inputs = CoolingCase(
        database=str(case / description["properties"]["file"]),
        solute=description["charge"]["solute"], solvent=description["charge"]["solvent"],
        solute_mass=description["charge"]["solute_mass_g"] / GRAMS_PER_KILOGRAM,
        solvent_mass=description["charge"]["solvent_mass_g"] / GRAMS_PER_KILOGRAM,
        seed_mass=runs[run_id]["seed_mass_g"] / GRAMS_PER_KILOGRAM,
        seed_size=settings["seed_size_um"] / MICROMETERS_PER_METER,
        shape_factor=description["solid"]["volume_shape_factor"],
        solute_density=description["solid"]["solute_density_kg_m3"],
        temperature=temperature, saturated_ratio=saturated_ratio)
    starting = settings["kinetics_starting_values"]
    return inputs, {"growth": tuple(starting["growth"]), "secondary_nucleation": tuple(starting["secondary_nucleation"])}


def measurement_times(case: Path, run_id: str, kind: str, stop: float | None = None) -> np.ndarray:
    """Return the sample times of one reduced table after seeding, as model times.

    Parameters
    ----------
    case : pathlib.Path
        Dataset folder.
    run_id : str
        Run identifier from the description.
    kind : str
        Kind of reduced table, for example ``"ir"`` or ``"fbrm"``.
    stop : float, optional
        Last model time to keep [s]. Default: the end of the temperature log.

    Returns
    -------
    numpy.ndarray
        Times [s] counted from seeding, shape ``(n_samples,)``, all positive and
        inside the temperature log.
    """
    description = dataset.load_description(case)
    seeding_time = {run["id"]: run for run in description["runs"]}[run_id]["seeding_time_s"]  # [s]
    log = dataset.read_reduced(case, run_id, description["simulation"]["temperature_table"])
    last = log["time"][-1] - seeding_time if stop is None else stop  # [s]
    time = dataset.read_reduced(case, run_id, kind)["time"] - seeding_time
    return time[(time > 0.0) & (time <= last)]


def _seed_moments(inputs: CoolingCase) -> np.ndarray:
    """Return the moments of the seed, crystals of one size.

    Parameters
    ----------
    inputs : CoolingCase
        Case inputs.

    Returns
    -------
    numpy.ndarray
        Total moments, order k in [m**k], shape ``(MOMENT_COUNT,)``: the number of
        crystals that holds the seed mass, times the seed size to the power k.
    """
    number = inputs.seed_mass / (inputs.solute_density * inputs.shape_factor * inputs.seed_size ** 3)  # [-]
    return number * inputs.seed_size ** np.arange(MOMENT_COUNT)


def _species(inputs: CoolingCase) -> tuple[int, int, list, list]:
    """Return the species positions and the charge compositions in database order.

    Parameters
    ----------
    inputs : CoolingCase
        Case inputs.

    Returns
    -------
    solute_index, solvent_index : int
        Positions of the solute and the solvent in the database.
    liquid_fractions : list of float
        Mass fractions of the liquid charge [-], one per database species.
    solid_fractions : list of float
        Mass fractions of the solid [-]: the solute alone.
    """
    names = list(json.loads(Path(inputs.database).read_text(encoding="utf-8")))
    solute_index, solvent_index = names.index(inputs.solute), names.index(inputs.solvent)
    total = inputs.solute_mass + inputs.solvent_mass  # [kg]
    liquid = [0.0] * len(names)
    liquid[solute_index], liquid[solvent_index] = inputs.solute_mass / total, inputs.solvent_mass / total
    solid = [0.0] * len(names)
    solid[solute_index] = 1.0
    return solute_index, solvent_index, liquid, solid


def _rows_of(reported: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Return the solver row of every requested time.

    Parameters
    ----------
    reported : numpy.ndarray
        Times the solver returned [s], increasing.
    requested : numpy.ndarray
        Requested output times [s], increasing.

    Returns
    -------
    numpy.ndarray
        Row index in ``reported`` of each requested time.

    Raises
    ------
    RuntimeError
        If the solver did not return a requested time.
    """
    rows = np.clip(np.searchsorted(reported, requested), 0, reported.size - 1)
    rows = np.where((rows > 0) & (np.abs(reported[rows - 1] - requested) < np.abs(reported[rows] - requested)), rows - 1, rows)
    missing = np.abs(reported[rows] - requested) > TIME_MATCH_TOLERANCE
    if missing.any():
        raise RuntimeError("The solver did not return %d of %d requested times, first %g s."
                           % (missing.sum(), requested.size, requested[missing][0]))
    return rows


def simulate(inputs: CoolingCase, kinetics: dict, times, model: str) -> SlurryMeasurements.SlurryState:
    """Simulate the crystallization and return the slurry state at the requested times.

    Parameters
    ----------
    inputs : CoolingCase
        Case inputs.
    kinetics : dict
        ``growth`` ``(k_g [um/s], E [J/mol], g [-])`` and
        ``secondary_nucleation`` ``(k_s [#/m**3/s], E [J/mol], s_1 [-], s_2 [-])``
        of ``PharmaPy.Kinetics.CrystKinetics``, with the relative supersaturation
        of mass fractions as driving force.
    times : array-like
        Output times [s] counted from seeding, shape ``(n_times,)``, positive and
        strictly increasing.
    model : str
        ``"legacy"`` for ``BatchCryst`` or ``"refactored"`` for
        ``BatchCrystallizer``; both with the method of moments and CVode.

    Returns
    -------
    PharmaPy.SlurryMeasurements.SlurryState
        One row per requested time, in the same order.

    Raises
    ------
    ValueError
        If the model is unknown or the times are not positive and increasing.
    RuntimeError
        If the solver does not return a requested time.

    Notes
    -----
    The legacy unit is solved with CVode's continuous reporting: in its default
    output mode it leaves out requested times that fall inside its last internal
    step.
    """
    from PharmaPy.Kinetics import CrystKinetics

    if model not in MODELS:
        raise ValueError("Unknown model %r; choose one of %s." % (model, list(MODELS)))
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or times[0] <= 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Output times must be positive and strictly increasing.")
    grid = np.concatenate(([0.0], times))  # [s] the solver starts at the seeding time
    solute_index, solvent_index, liquid_fractions, solid_fractions = _species(inputs)
    start_temperature = float(inputs.temperature(0.0))  # [K]
    liquid_mass = inputs.solute_mass + inputs.solvent_mass  # [kg]
    moments = _seed_moments(inputs)  # [m**k]

    def saturated_fraction(temp, conc=None):
        """Return the solute per solution at saturation [kg/kg] at a temperature [K]."""
        ratio = inputs.saturated_ratio(temp)
        return ratio / (1.0 + ratio)

    rates = {"growth": kinetics["growth"], "nucl_sec": kinetics["secondary_nucleation"]}
    if model == "legacy":
        from PharmaPy.Crystallizers import BatchCryst
        from PharmaPy.Phases import LiquidPhase, SolidPhase

        unit = BatchCryst(inputs.solute, method="moments", controls={"temp": inputs.temperature}, basis="mass_frac")
        unit.Phases = (
            LiquidPhase(inputs.database, temp=start_temperature, mass=liquid_mass, mass_frac=liquid_fractions),
            SolidPhase(inputs.database, temp=start_temperature, moments=moments, kv=inputs.shape_factor,
                       mass_frac=solid_fractions, num_mom=MOMENT_COUNT))
        unit.Kinetics = CrystKinetics(solub_fn=saturated_fraction, **rates)
        reported, states = unit.solve_unit(time_grid=grid, verbose=False,
                                           sundials_opts={"report_continuously": True})[:2]
        rows = _rows_of(np.asarray(reported, dtype=float), times)
        return SlurryMeasurements.state_from_total_moments(
            times, np.asarray(states)[rows], solute_index, solvent_index, inputs.shape_factor, MOMENT_COUNT)

    from PharmaPy.Crystallizers_Refactored import BatchCrystallizer
    from PharmaPy.IntegratorBackends import AssimuloBackend
    from PharmaPy.Mechanisms import MomentsPopulationBalance
    from PharmaPy.Phases_Refactored import LiquidPhase, SolidPhase
    from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController

    liquid = LiquidPhase(inputs.database, temp=start_temperature, mass=liquid_mass, mass_frac=liquid_fractions,
                         name_solv=inputs.solvent)
    solid = SolidPhase(inputs.database, mass=inputs.seed_mass, mass_frac=solid_fractions)
    slurry_volume = liquid.vol + inputs.seed_mass / inputs.solute_density  # [m**3]
    solid.mechanisms = MomentsPopulationBalance(
        owning_phase=solid, target_components=inputs.solute, solvent_name=inputs.solvent,
        moments_init=moments * MICROMETERS_PER_METER ** np.arange(MOMENT_COUNT) / slurry_volume,  # [um**k/m**3]
        kv=inputs.shape_factor, density=inputs.solute_density)
    unit = BatchCrystallizer(integrator=AssimuloBackend(), controller=SimpleTemperatureController(
        temp_func=lambda time: float(inputs.temperature(time))))
    unit.Phases = [liquid, solid]
    unit.CrystKinetics = CrystKinetics(solub_fn=saturated_fraction, solubility_basis="mass_per_mass_solution", **rates)
    reported, _ = unit.solve_unit(time_grid=grid, verbose=False)
    rows = _rows_of(np.asarray(reported, dtype=float), times)
    result = unit.result
    return SlurryMeasurements.state_from_specific_moments(
        times, np.asarray(result.mu_n_solid0)[rows], np.asarray(result.mass_j_liquid0)[rows],
        np.asarray(result.vessel_vol)[rows], solute_index, solvent_index)


def main(arguments: list[str]) -> None:
    """Simulate one run with the starting kinetics and print predictions at the spectrum times.

    Parameters
    ----------
    arguments : list of str
        The dataset folder, the run identifier and the model.
    """
    case, run_id, model = Path(arguments[0]), arguments[1], arguments[2]
    description = dataset.load_description(case)
    inputs, kinetics = load_case(case, run_id, "common_slope")
    times = measurement_times(case, run_id, description["band_calibration"]["table"])
    state = simulate(inputs, kinetics, times, model)
    ratio = SlurryMeasurements.dissolved_ratio(state)  # [kg/kg]
    solid = SlurryMeasurements.solid_mass(state, inputs.solute_density, inputs.shape_factor)  # [kg]
    total = SlurryMeasurements.total_solute(state, inputs.solute_density, inputs.shape_factor)  # [kg]
    print("%s, %s model, starting kinetics %s" % (run_id, model, kinetics))
    print("%10s %8s %12s %12s %10s" % ("time [s]", "T [C]", "ratio [kg/kg]", "saturation", "solid [g]"))
    for row in np.linspace(0, times.size - 1, 12).astype(int):
        temp = float(inputs.temperature(times[row]))  # [K]
        print("%10.0f %8.2f %12.5f %12.5f %10.3f" % (times[row], temp - 273.15, ratio[row],
                                                    float(inputs.saturated_ratio(temp)), solid[row] * GRAMS_PER_KILOGRAM))
    charged = inputs.solute_mass + inputs.seed_mass  # [kg]
    print("%d rows; largest relative drift of the solute in the vessel: %.1e" % (times.size, np.abs(total / charged - 1.0).max()))


if __name__ == "__main__":
    main(sys.argv[1:])
