# Bioreactor implementation: final net changes

## Qualification and documentation consolidation (2026-09-28)

Levels 2–6 now share `tests/Bioreactor/test_bioreactor_qualification.py`, with
individually collected checks and descriptive population/network/constraint/
operation/coupled prefixes. `reconciliation_reference.py` replaces the level-named
independent helper. Existing input directories, assertions and tolerances remain
unchanged; all 80 test/helper function bodies are structurally identical after
identifier renaming. Dependent imports and documented commands were updated.

The technical summary now includes biological relationships and qualification
limits, replacing the separate relationship and validation-ladder documents.
The historical review was archived verbatim in the Alterable workspace; replaced
files have external recovery copies. No runtime or native PharmaPy code changed.

Post-consolidation verification: all 613 bioreactor checks passed (352.73 s),
plus 31 native reactor/flowsheet checks (18.20 s); no failures or skips.
The 38 bioreactor and three native warnings match the existing warning categories.
JUnit results and recovery copies are retained in the Alterable workspace.

## Industry review completion (2026-09-28)

The existing provider interface now connects to configured volumetric kinetics
and optional reduced-pathway uptake rules as well as reconciliation. Default
rules and existing uptake forms retain their mathematics. Named capacities,
units, finite values and validity policies are checked before use; no biological
coefficients or process identities enter the runtime. Immutable provider results
support native replay copying without modifying replay itself.

A synthetic mammalian fed-batch example demonstrates viable cells, product and
configuration-selected affine/registered polynomial surrogates. It fits a declared
synthetic rate grid, not desired trajectories. Existing example inputs remain
unchanged. The new notebook has independent balance and refinement checks and
explicit extrapolation-failure tests. It is not experimental validation.

Bioreactor tests select headless plotting. The small-target assertion uses
sqrt(1e-15) times the common rate scale, consistent with SLSQP's squared-residual
objective tolerance; the solver and physical constraints were not loosened.
The configuration guide, technical summary and human-facing validation record
replace stale delivery promises. Source feed interpretation is explicitly pending
domain review. No commit/push or native PharmaPy architecture change was made in
this completion. Worktree changes from earlier tasks remain separate.

Final reviewer-environment verification: 613 bioreactor tests passed in 352.46 s
without a plotting-backend override, and 31 native reactor/flowsheet tests passed.
There were no skips in that environment. Existing default numerical histories
match pre-review rate/LP methods exactly in controlled same-environment runs.
Synthetic affine/quadratic trajectory errors are 0.71%/0.12%; these are not
experimental accuracy claims. Feed-accounting domain sign-off remains open.


## Optional thermophysical input cleanup

The four pre-review examples retain every thermophysical property key but leave
unused arrays empty and unused scalar/text values null. Molecular weights and
liquid densities remain populated. Case/model values required by construction,
validation or the selected dynamics remain unchanged. This is an isothermal
example-input cleanup, not a change to native thermodynamics or runtime code.
Original-versus-cleaned forward histories match exactly for the four examples
and F11/reconciled F10/F11 variants. Optional energy/vapor calculations require
the relevant property slots to be populated before use.


## Input-selected biological relationships

`Bioreactors/relationships.py` expands named, composable rate relationships into
the existing kinetic rule graph. The only connection is in input normalization;
native reactors, integrators, reconciliation and pathway solvers are unchanged.
Explicit expressions remain supported. Required coefficient slots, signs and
duplicate rule ownership are checked without adding biological defaults.
F10/F11 direct and reconciled mechanism inputs use the catalog with exactly the
same normalized mathematical definitions as before. See `TECHNICAL_SUMMARY.md#biological-relationships` for
the equations, configuration contract, units and scientific scope. The new
relationship tests supplement the existing independent forward/reference tests;
synthetic qualification does not imply accuracy for every organism or network.

## Scope

This document records the final difference between pre-bioreactor PharmaPy commit
`bd200f5e474243d60237cde344c174a2e4468137` and the forward-simulation bioreactor
implementation, including parent `MultiPhaseVesselRefactor` commit `7f815b2`. It describes endpoint behavior rather than the sequence used to
develop it. Superseded implementations, intermediate failures, temporary review
artifacts, and full source diffs are intentionally omitted. Git remains the
authoritative line-level record.

The extension retains PharmaPy's native composition path:

`LiquidPhase -> Mechanism -> BatchReactor/SemiBatchReactor -> solve_unit -> DynamicResult`

Biological and metabolic models enter through native mechanism state and rate
interfaces rather than through a separate reactor simulator.

The delivered bioreactor API is forward-only: construction, configured rates,
metabolic reconciliation, integration and trajectory reporting. Native PharmaPy
facilities outside this extension retain their existing behavior. Frozen example
coefficients retain source/provenance notes; these are inputs, not an inference API.

Rate reconciliation supports a backward-compatible `relative-regularized` policy
and an explicit `unweighted` policy (equal absolute target residuals, no internal
penalty or target bounds). Both retain native network and inventory constraints.
The unweighted objective uses a common target-magnitude scale with a positive
floor to resolve small rates without changing the mathematical minimizer.
The shared JSON templates expose this choice; the Khare example uses `unweighted`.
Remaining reconciliation ties use an automatic capacity-normalized squared-flux
selection with primary-penalized coordinates fixed. It reuses existing solvers and
constraints, reports selection diagnostics, and needs no additional JSON fields.
Nonlinear inventory cases retain a local-solution qualification.
The opt-in `unit-scaled` primary policy carries explicit physical residual
reference scales through the existing unit converter, preserving the metric
under consistent flux-coordinate conversions. Legacy policies are unchanged.

Optional post-reconciliation mortality expressions consume unit-converted accepted
fluxes through the existing mechanism; finite-step inventory feasibility is
rechecked with the actual mortality. Rule-graph providers support explicit validity
domains and extrapolation policies. Assembly diagnostics expose actual population
rates and provider-domain status for the last evaluation. These options preserve
existing behavior when absent and introduce no process-specific biological law.

## Existing PharmaPy files modified

| File | Final net change |
| --- | --- |
| `.gitignore` | Ignores Python bytecode, pytest caches, notebook checkpoints, and generated local result directories. |
| `PharmaPy/DataClasses.py` | Uses the parent's cached algebraic state layout and retains the differential-mask convenience property. Makes resolved outlet transfers scale volume and species flow together while preserving composition, including zero-flow composition. |
| `PharmaPy/IntegratorBackends.py` | Adds SciPy `solve_ivp` and fixed-step backends alongside the parent's CVode, IDA, and optional Julia backends. The SciPy path supports BDF, tolerance-bounded state depletion events and restarts without numerical boundary chatter, preservation of positive source/sink equilibria below the depletion tolerance, controlled finite-difference Jacobians, and accepted-step handling. ODE backends share initialization and reset behavior. IDA receives the differential mask and consistent derivative initialization. IDA uses the parent's mixed rate/residual contract and native events; differential-only initialization and explicit rejection of unsupported sensitivity modes are retained. |
| `PharmaPy/Mechanisms.py` | Adopts the parent's registered solver-state keys, phase ownership, event hooks, and `get_solver_state_residuals` interface. Postponed annotations retain Python 3.9 import compatibility for the available IDA environment. |
| `PharmaPy/MultiPhaseVessel.py` | Uses the parent's single mixed differential-rate/algebraic-residual vector and backend dispatch through `solve_unit`. Retains differential-only initialization, DAE convenience methods, limiter bypass, and fresh replay buffers. Opt-in metabolic mechanisms receive resolved native inlet rates for availability checks without duplicating feed balances. Per-transfer limiting preserves material/heat coupling and outlet composition; proposed rates are checked for infeasible depletion without altering reactions. |
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
| `PharmaPy/Bioreactors/__init__.py` | Exposes the supported bioreactor construction, culture and provider interfaces. |
| `construction.py` | Validates input mappings and constructs native phases, mechanisms, batch or fed-batch reactors, solver settings, and scheduled operating events. Accepts an isothermal temperature and stock-to-target additions with solvent and dilution under constant-density, additive-volume mixing. Adds optional feed-completeness declarations/strict checking and post-run depletion diagnostics while retaining legacy material increments. |
| `culture.py` | Defines culture state, kinetic quantities, snapshots, unit conversions, growth/death updates, and configured kinetic-rule evaluation. Conditional expressions evaluate only the selected branch. Validates optional neutral/supplied correction policies and declared untargeted exchanges for rate reconciliation. |
| `input_format.py` | Validates one versioned forward-input layout across all examples and translates it into the existing model interfaces. Preserves legacy unversioned inputs and rejects non-null inactive model settings. |
| `BioreactorUnitConverter.py` | Centralizes declared biological source-basis compilation and physical quantity conversions. Both model families accept explicit population/amount/time bases; crossing cell and dry-mass bases requires supplied cellular mass. Pathway source and transfer-cap conversions share the declared scale. Reads JSON paths/mappings and converts scalar/array physical quantities with supplied properties where needed. Compiles optional unit-bearing reconciliation floors and normalization scales; omitted settings preserve legacy source-coordinate weighting. |
| `mechanisms.py` | Implements reduced-pathway and rate-reconciled biological mechanisms and maps their source terms into native PharmaPy state rates. Untargeted exchanges retain network/inventory constraints but do not contribute kinetic penalties, target bounds, or normalization. Validates exchange references and coefficients; exposes qualified conservation and declared exchange-role diagnostics. Optional explicit flux bases compile shared growth/exchange conversions and molecular, carbon-mole or mass product conversions; provider units are checked. An optional configured first-order dead-cell removal rate represents lysis; omission retains the prior zero-removal balance. Undeclared legacy mappings remain supported and unverified. |
| `rate_providers.py` | Provides rule-graph kinetics, affine surrogates, hybrid per-output assignment, and registered external providers with domain and unit metadata. |
| `inventory.py` | Shared linear boundary/end-inventory constraints and growth-dependent culture exposure constraints; both closures constrain accepted fluxes using their propagation source maps and actual declared supply. |

## New metabolic package

| File | Final responsibility |
| --- | --- |
| `PharmaPy/Metabolic/__init__.py` | Exposes metabolic network and reduced-pathway interfaces. |
| `network.py` | Defines reactions, stoichiometric matrices, bounds, exchange mappings, and validated metabolic-network configuration; provides positive conserved-mass-vector and closed-uptake export diagnostics, with optional reduced/chemical scope and boundary roles. |
| `pathways.py` | Solves the configured reduced-pathway linear program with inventory feasibility, sorted-ID secondary tie-breaking and a qualified closed-uptake screen. Returns the common flux result with pathway-specific fields. |
| `closures/base.py` | Defines the shared metabolic environment, flux result/diagnostics, and separate infeasibility and numerical-error contracts for both optimizers. |
| `closures/reconciled.py` | Reconciles supplied kinetic flux targets with steady-state network balances, reaction bounds, inventory feasibility, and optional flux-variability analysis. Substitutes fixed fluxes before rank reduction, scales both feasibility and optimization, checks stationarity and original constraints, and retries stagnant solves in balance-preserving coordinates. The main SLSQP solve uses `ftol=1e-15` to resolve flux differences that otherwise accumulate in reactor trajectories; physical and comparison tolerances are unchanged. |
| `closures/_recovery.py` | Attempts restricted-space numerical recovery after full-bound and null-space floating-point retries fail. Accepts a candidate only after checking the original constraints and objective-gap certificate. Exact arithmetic omits zero terms without changing the certificate. |
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


- `tests/Bioreactor/test_bioreactor_examples.py`: packaged inputs, outputs,
  and forward packaging.
- `tests/Bioreactor/test_bioreactor_workflow_notebooks.py`: direct execution and
  behavior of the retained forward-only workflow notebooks.
- `tests/test_material_rate_limiting.py`: native reaction preservation, coupled
  transfer/outlet limiting, energy consistency, zero-flow composition,
  infeasible depletion, bypass, and `solve_unit` integration.

## Examples and reproducibility assets added

The retained examples are `batch_ecoli_dfba`, `generic_batch`,
`generic_fed_batch`, `ecoli_ye_fed_batch`, and `mammalian_surrogate_fed_batch`. Each contains the standard input
JSONs, direct forward notebook, and trajectory exports. The E. coli fed-batch
case also includes the frozen-model reconciliation assessment. Shared templates,
source documentation, core reference comparisons and dependency pins remain.

Retired biological demonstrations, duplicate demos, runner scripts, installation
verification wrappers, regression-refresh helpers and old campaign reports are
removed. Active numerical tests retain only necessary historical input fixtures
and captured solver states. The runtime architecture is unaffected by this cleanup.
Run pytest directly; forward notebooks need only the declared input files.

### Generic forward-only homework example

`examples/bioreactors/generic_batch/` contains three input JSON files, a six-step forward-only
notebook, and trajectory CSV/PNG/SVG outputs. It configures one nutrient
and one pathway through the existing native construction API. Separate tests verify
the six-hour trajectory against exponential growth and nutrient depletion formulas
and check conservation of liquid mass plus dry biomass. The supplied run stays
before depletion. No runtime modeling code is added.

### Generic fed-batch homework example

`examples/bioreactors/generic_fed_batch/` provides the matching three JSON inputs,
six-step forward-only notebook, and CSV/PNG/SVG outputs for one scheduled nutrient pulse.
It uses the unchanged pathway model through `SemiBatchReactor`, retains both sides
of the feed event. Separate tests check the piecewise solution, biomass continuity,
dilution, and feed-corrected mass conservation. Feed amounts follow the existing carrier-volume
plus separately added solute-mass convention. No runtime modeling code is added.

## Final behavior and boundaries

The endpoint supports single, perfectly mixed liquid-phase batch and fed-batch
bioreactors; reduced-pathway dynamic FBA; rate-reconciled metabolic networks;
and modular kinetic and surrogate providers. The flagship E. coli case uses
adaptive SciPy BDF integration. The culture mechanism also supports fixed-step propagation with
inventory-constrained reconciliation. DAE residual assembly and
the optional IDA adapter are implemented but are not a flagship demonstrated
solve.

The endpoint does not provide a packaged continuous chemostat, perfusion or cell-
retention model, spatial reactor, resolved gas phase, dynamic acid-base or pH
controller, product-quality model, automatic surrogate training, exhaustive
organism-scale reconstruction, independent experimental validation, or complete elemental
closure. Full mathematical, numerical, and capability details are maintained in
[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md).

- Optional `working-volume` policy uses existing phase hooks and mechanism states;
  native mass/density remains the default. Feed inputs distinguish carrier from
  solution volume, and notebooks consume recorded reactor volume.

- Core forward notebooks execute simulation and trajectory CSV/PNG/SVG export.
  The synthetic mammalian notebook additionally fits and compares rate surrogates;
  independent numerical checks remain in the test suite.

- Shared construction also supports configured direct kinetic rules, named
  continuous liquid inlets, and scheduled well-mixed samples. Working volume
  integrates declared inlet flow; samples remove extensive biological inventories
  proportionally. The yeast-extract E. coli notebook uses this shared builder,
  with all process mathematics and schedules supplied by JSON.

- Rate reconciliation accepts the existing `biomass_kg` population input with
  explicit dry-mass flux bases, without requiring an invented cell-mass conversion.
  Cell-count configurations retain their existing states and conversions.
  The closure directly recovers a unique feasible zero-error target solution
  when available; other problems retain constrained optimization and tie-breaking.
