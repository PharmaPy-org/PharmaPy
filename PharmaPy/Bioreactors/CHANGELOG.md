# Bioreactor implementation: final net changes

## Scope

This document records the final difference between pre-bioreactor PharmaPy commit
`bd200f5e474243d60237cde344c174a2e4468137` and the completed WP3 bioreactor
implementation, including parent `MultiPhaseVesselRefactor` commit `7f815b2`. It describes endpoint behavior rather than the sequence used to
develop it. Superseded implementations, intermediate failures, temporary review
artifacts, and full source diffs are intentionally omitted. Git remains the
authoritative line-level record.

The extension retains PharmaPy's native composition path:

`LiquidPhase -> Mechanism -> BatchReactor/SemiBatchReactor -> solve_unit -> DynamicResult`

Biological and metabolic models enter through native mechanism state and rate
interfaces rather than through a separate reactor simulator.

## Existing PharmaPy files modified

| File | Final net change |
| --- | --- |
| `.gitignore` | Ignores Python bytecode, pytest caches, notebook checkpoints, and generated local result directories. |
| `PharmaPy/DataClasses.py` | Uses the parent's cached algebraic state layout and retains the differential-mask convenience property. Makes resolved outlet transfers scale volume and species flow together while preserving composition, including zero-flow composition. |
| `PharmaPy/IntegratorBackends.py` | Adds SciPy `solve_ivp` and fixed-step backends alongside the parent's CVode, IDA, and optional Julia backends. The SciPy path supports BDF, state depletion events and restarts, controlled finite-difference Jacobians, and accepted-step handling. ODE backends share initialization and reset behavior. IDA receives the differential mask and consistent derivative initialization. IDA uses the parent's mixed rate/residual contract and native events; differential-only initialization and explicit rejection of unsupported sensitivity modes are retained. |
| `PharmaPy/Mechanisms.py` | Adopts the parent's registered solver-state keys, phase ownership, event hooks, and `get_solver_state_residuals` interface. Postponed annotations retain Python 3.9 import compatibility for the available IDA environment. |
| `PharmaPy/MultiPhaseVessel.py` | Uses the parent's single mixed differential-rate/algebraic-residual vector and backend dispatch through `solve_unit`. Retains differential-only initialization, DAE convenience methods, limiter bypass, and fresh replay buffers. Per-transfer limiting preserves material/heat coupling and outlet composition; proposed rates are checked for infeasible depletion without altering reactions. |
| `PharmaPy/Reactors_Refactored.py` | Adopts the parent's renamed module and forwarding constructor. Biological construction imports this native reactor implementation. |

### Parent changes incorporated

The merge also retains the parent's native work rather than recreating it in the
bioreactor package:

- `Phases_Refactored.py`, `Streams_Refactored.py`, and
  `MixedPhases_Refactored.py` hold the refactored phase/stream implementation;
  `Phases.py`, `Streams.py`, and `MixedPhases.py` serve legacy workflows.
- `ProcessControl_Refactored.py` and `Crystallizers_Refactored.py` replace the
  earlier `_Refactor` names. The latter also defers annotations for Python 3.9.
- `Connections.py`, `Interpolation.py`, `Kinetics.py`, and `ThermoModule.py`
  include parent fixes for connected trajectories, interpolation, kinetic
  bounds, and thermodynamic evaluation.
- `MultiPhaseVessel.py` includes parent stream workspaces, state/event handling,
  algebraic packing, and transfer-specific limiting. `IntegratorBackends.py`
  retains the parent's optional `DiffeqpyBackend` without claiming Julia validation.
- Parent demonstration scripts `FlowsheetTester.py`, `Tester.py`, and
  `crystTester.py`, and the native packing/reactor/crystallizer tests are retained.
  Superseded `Phases_old.py` and `Streams_old.py` are removed by the parent merge.

`ParamEstim.py`, `Results.py`, and `_assimulo.py` remain reused rather than
reimplemented. Legacy `Reactors.py` is not the bioreactor construction path.
The old `unit_model(..., dae=True)` branch is replaced by the parent's normal
mixed-vector contract; algebraic mechanisms use `get_solver_state_residuals`.


## New bioreactor package

| File | Final responsibility |
| --- | --- |
| `PharmaPy/Bioreactors/__init__.py` | Exposes the supported bioreactor construction, culture, provider, estimation, and design interfaces. |
| `construction.py` | Validates input mappings and constructs native phases, mechanisms, batch or fed-batch reactors, solver settings, and scheduled operating events. |
| `culture.py` | Defines culture state, kinetic quantities, snapshots, unit conversions, growth/death updates, and configured kinetic-rule evaluation. |
| `mechanisms.py` | Implements reduced-pathway and rate-reconciled biological mechanisms and maps their source terms into native PharmaPy state rates. |
| `rate_providers.py` | Provides rule-graph kinetics, affine surrogates, hybrid per-output assignment, and registered external providers with domain and unit metadata. |
| `inventory.py` | Constructs finite-step extracellular inventory constraints for metabolic reconciliation. |
| `estimation.py` | Adapts named bioreactor observations and parameters to native PharmaPy residual weighting and bounded dynamic fitting. |
| `design.py` | Computes finite-difference sensitivities, information matrices, D/A/E scores, feasibility and cost checks, and ranked candidate designs. |
| `experiments.py` | Orchestrates observation extraction, calibration, follow-up selection, refitting, withheld prediction comparisons, and reproducible synthetic confirmation. |

## New metabolic package

| File | Final responsibility |
| --- | --- |
| `PharmaPy/Metabolic/__init__.py` | Exposes metabolic network and reduced-pathway interfaces. |
| `network.py` | Defines reactions, stoichiometric matrices, bounds, exchange mappings, and validated metabolic-network configuration. |
| `pathways.py` | Solves the configured reduced-pathway linear program and applies deterministic secondary tie-breaking. |
| `closures/base.py` | Defines the metabolic environment, flux result, and separate infeasibility and numerical-error contracts. |
| `closures/reconciled.py` | Reconciles supplied kinetic flux targets with steady-state network balances, reaction bounds, inventory feasibility, and optional flux-variability analysis. |
| `closures/_recovery.py` | Attempts restricted-space numerical recovery after a failed reconciliation solve and accepts a candidate only after checking the original constraints and objective-gap certificate. |
| Package `__init__.py` files | Publish the supported closure interfaces. |

## Tests reorganized and added

The optional-Assimulo import test moved from
`tests/test_optional_assimulo_imports.py` to
`tests/Bioreactor/test_optional_assimulo_imports.py`. Its repository-root lookup
was updated, the parent's renamed modules are covered by the import boundary, and the former
module-wide unit marker was removed.

The following endpoint tests were added:

- `tests/Bioreactor/test_bioreactor_dae.py`: DAE residuals, masks, initialization,
  and optional IDA integration boundaries.
- `tests/Bioreactor/test_bioreactor_mechanisms.py`: biological state/rate mapping,
  pathway behavior, reconciliation, and physical constraints.
- `tests/Bioreactor/test_bioreactor_rate_providers.py`: rule, affine, hybrid, and
  external provider contracts.
- `tests/Bioreactor/test_bioreactor_estimation.py`: observation handling,
  weighting, fitting, rank, and covariance behavior.
- `tests/Bioreactor/test_bioreactor_design.py`: information criteria, candidate
  selection, stability, and follow-up-study behavior.
- `tests/Bioreactor/test_bioreactor_examples.py`: packaged inputs, outputs,
  provenance and regression acceptance.
- `tests/Bioreactor/test_bioreactor_workflow_notebooks.py`: direct execution and
  behavior of both editable workflow notebooks.
- `tests/test_material_rate_limiting.py`: native reaction preservation, coupled
  transfer/outlet limiting, energy consistency, zero-flow composition,
  infeasible depletion, bypass, and `solve_unit` integration.

## Examples and reproducibility assets added

`examples/bioreactors/` contains the common documentation and reproducibility
entry points:

- `README.md` describes installation, execution, configuration, evidence, and
  the supported scope.
- `requirements-py311.txt` pins the qualified Python environment.
- `verify_installation.py` and `verify_examples.py` run installation and example
  acceptance checks; `installation_report.json` records the packaged result.
- `refresh_regression.py` is the explicit maintenance operation for replacing
  generated regression references.
- `reproduce_figures.py` regenerates current-model paper-layout figures and their
  comparison assets.
- Redundant `run_*.py` and `design_runner.py` scripts are excluded from the
  committed tree. Verification executes the notebook tests; figure reproduction
  and explicit baseline maintenance reuse the notebooks' forward cells.

### E. coli batch reduced-pathway example

`examples/bioreactors/batch_ecoli_dfba/` adds:

- An editable, directly executable `workflow.ipynb` containing forward simulation,
  inference, and design through the library API.
- Separate case, mechanism, thermodynamic, estimation, design, source-manifest,
  and generated-regression input documents.
- Documentation and traceable reduced-pathway reference evidence.
- Reproducible trajectory, balance, validation, estimation, design, summary, and
  figure outputs, including plotted data and published-panel provenance.

The delivered case is a 1 L, 10-hour batch culture. It demonstrates reduced-
pathway dynamic optimization, forward simulation, bounded trajectory fitting,
and information-based initial-glucose experiment selection.

### CHO fed-batch rate-reconciliation example

`examples/bioreactors/fed_batch_cho/` adds:

- An editable, directly executable `workflow.ipynb` containing forward simulation,
  inference, and design through the library API.
- Separate case, mechanism, thermodynamic, consistency, estimation, design,
  source-manifest, and generated-regression input documents.
- Documentation and traceable public CHO network reconstruction evidence.
- Reproducible trajectory, balance, consistency, validation, estimation, design,
  summary, and figure outputs, including plotted data and published-panel provenance.

The delivered case is a 0.75 L, 12-day fed-batch culture. It demonstrates
scheduled feeds and concentration targets, kinetic-target reconciliation with
network and finite-step inventory constraints, product-trajectory calibration, packaged standalone-fit
and mesh-refinement evidence, and information-based pH follow-up selection.

### Generic forward-only homework example

`examples/bioreactors/generic_batch/` contains three input JSON files, a seven-step
notebook, and trajectory CSV/PNG plus a summary report. It configures one nutrient
and one pathway through the existing native construction API, verifies the
six-hour trajectory against exponential growth and nutrient depletion formulas,
and checks conservation of liquid mass plus dry biomass. The supplied run stays
before depletion. No runtime modeling code, estimation, or design workflow is added.

### Generic fed-batch homework example

`examples/bioreactors/generic_fed_batch/` provides the matching three JSON inputs,
seven-step notebook, and CSV/PNG/summary outputs for one scheduled nutrient pulse.
It uses the unchanged pathway model through `SemiBatchReactor`, retains both sides
of the feed event, and checks the piecewise solution, biomass continuity, dilution,
and feed-corrected mass conservation. Feed amounts follow the existing carrier-volume
plus separately added solute-mass convention. No runtime modeling code is added.

## Final behavior and boundaries

The endpoint supports single, perfectly mixed liquid-phase batch and fed-batch
bioreactors; reduced-pathway dynamic FBA; rate-reconciled metabolic networks;
modular kinetic and surrogate providers; bounded dynamic parameter estimation;
and enumerated model-based experiment design. The flagship E. coli case uses
adaptive SciPy BDF integration. The CHO case uses explicit fixed-step culture
propagation with inventory-constrained reconciliation. DAE residual assembly and
the optional IDA adapter are implemented but are not a flagship demonstrated
solve.

The endpoint does not provide a packaged continuous chemostat, perfusion or cell-
retention model, spatial reactor, resolved gas phase, dynamic acid-base or pH
controller, product-quality model, automatic surrogate training, exhaustive
organism-scale reconstruction, global parameter inference, continuous optimal-
control design, independent experimental validation, or complete elemental
closure. Full mathematical, numerical, and capability details are maintained in
[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md).
