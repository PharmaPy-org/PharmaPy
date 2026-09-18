import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"


def test_only_declared_examples_are_present():
    folders = sorted(path.name for path in EXAMPLES.iterdir() if path.is_dir())
    assert folders == ["batch_ecoli_dfba", "fed_batch_cho", "generic_batch", "generic_fed_batch"]


def test_flagship_examples_are_self_contained_and_accepted():
    for name in ("batch_ecoli_dfba", "fed_batch_cho"):
        folder = EXAMPLES / name
        assert (folder / "README.md").is_file()
        assert (folder / "workflow.ipynb").is_file()
        for required in ("case.json", "mechanism.json", "source_manifest.json", "thermo.json"):
            assert (folder / "inputs" / required).is_file()
        for required in ("trajectories.csv", "trajectories.png", "summary.json",
                         "validation.json", "balance_audit.json"):
            assert (folder / "outputs" / required).is_file()
        assert json.loads((folder / "outputs/summary.json").read_text())["status"] == "PASS"
        assert json.loads((folder / "outputs/validation.json").read_text())["status"] == "PASS"
    for name, unit in (("batch_ecoli_dfba", "BatchReactor"),
                       ("fed_batch_cho", "SemiBatchReactor")):
        folder = EXAMPLES / name
        assert (folder / "inputs/estimation.json").is_file()
        estimation = json.loads(
            (folder / "outputs/parameter_estimation.json").read_text()
        )
        assert estimation["status"] == "PASS"
        assert estimation["native_estimator"] == "ParameterEstimation"
        assert estimation["native_unit"] == unit
        assert estimation["identifiable"] is True
        assert estimation["covariance"] and estimation["observations"]
    design = json.loads(
        (EXAMPLES / "fed_batch_cho/outputs/experiment_design.json").read_text()
    )
    assert design["status"] == "PASS"
    assert design["replayed"] is True
    assert design["score_improvement"] > 0.0
    assert all(item["feasible"] for item in design["candidate_evaluations"])
    for name in ("batch_ecoli_dfba", "fed_batch_cho"):
        folder = EXAMPLES / name
        settings = json.loads((folder / "inputs/design.json").read_text())
        report = json.loads((folder / "outputs/experiment_design.json").read_text())
        assert report["acceptance_profile"] == settings["acceptance"]["profile"] == "capability"
        assert report["capability_status"] == report["status"] == "PASS"
        assert all(report["checks"][key] for key in report["required_checks"])
        assert report["empirical_benefit_status"] == (
            "OBSERVED" if report["checks"]["realized_prediction_benefit"] else "NOT_DEMONSTRATED")
    ecoli_audit = json.loads(
        (EXAMPLES / "batch_ecoli_dfba/outputs/balance_audit.json").read_text()
    )
    assert ecoli_audit["nonnegative_inventory_constraint"] is True
    assert ecoli_audit["raw_minimum_species_mass_kg"] >= 0.0


def test_operational_modules_contain_no_flagship_identity_or_reaction_ids():
    forbidden = ("mahadevan", "reddy", "escherichia", "chinese hamster", "R047", "GLC")
    paths = [
        *list((ROOT / "PharmaPy/Bioreactors").glob("*.py")),
        *list((ROOT / "PharmaPy/Metabolic").rglob("*.py")),
    ]
    text = "\n".join(path.read_text().lower() for path in paths)
    for token in forbidden:
        assert token.lower() not in text


def test_terminal_benchmark_errors_remain_below_declared_thresholds():
    ecoli = json.loads((EXAMPLES / "batch_ecoli_dfba/outputs/summary.json").read_text())
    cho = json.loads((EXAMPLES / "fed_batch_cho/outputs/summary.json").read_text())
    assert ecoli["relative_final_biomass_error"] < 0.02
    assert cho["relative_final_vcd_error"] < 0.02
    assert cho["relative_final_product_error"] < 0.02


def test_cho_standalone_estimation_matches_design_calibration_declarations():
    folder = EXAMPLES / "fed_batch_cho/inputs"
    estimation = json.loads((folder / "estimation.json").read_text())
    design = json.loads((folder / "design.json").read_text())
    parameter = estimation["parameters"][0]
    design_parameter = design["parameters"][0]
    for name in ("name", "mechanism_path", "initial_value",
                 "lower_bound", "upper_bound"):
        assert parameter[name] == design_parameter[name]
    assert [time * 86400.0 for time in estimation["training_times"]] == (
        design["sample_times_s"]
    )
    assert [time * 86400.0 for time in estimation["held_out_times"]] == (
        design["withheld_times_s"]
    )
    assert estimation["covariance"] == design["covariance"]
    assert estimation["synthetic_data"]["true_parameters"] == (
        design["synthetic_data"]["true_parameters"]
    )
    assert estimation["synthetic_data"]["seed"] == (
        design["synthetic_data"]["baseline_seed"]
    )


def test_packaged_provenance_and_regression_inputs():
    for name in ("batch_ecoli_dfba", "fed_batch_cho"):
        inputs = EXAMPLES / name / "inputs"
        manifest = json.loads((inputs / "source_manifest.json").read_text())
        assert manifest["independent_experimental_validation"] is False
        for key, digest in manifest.items():
            if key.endswith("_sha256"):
                evidence = inputs / manifest[key.removesuffix("_sha256")]
                assert hashlib.sha256(evidence.read_bytes()).hexdigest() == digest
        baseline = json.loads((inputs / manifest["regression_baseline"]).read_text())
        assert baseline["kind"] == "generated_model_regression"
        assert baseline["independent_experimental_validation"] is False
        for path, digest in baseline["inputs_sha256"].items():
            assert hashlib.sha256((inputs / path).read_bytes()).hexdigest() == digest
        assert len(baseline["source_tree_python_sha256"]) == 64
        assert baseline["command"] and baseline["packages"]
