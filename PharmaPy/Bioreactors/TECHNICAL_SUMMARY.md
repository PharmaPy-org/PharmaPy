# Native bioreactor forward modeling

## Architecture and model selection

`mechanism.json + thermo.json → build_bioreactor → native phase +
mechanism + BatchReactor/SemiBatchReactor/ContinuousReactor → solve_unit → DynamicResult`.

Biological identities, coefficients, networks and operating schedules belong in
user inputs. Mechanisms supply source terms and population states through native
interfaces; vessels, phases, streams, state packing and integration remain the
execution path. This reduced branch retains the WP3.1 forward-model scope, including rate reconciliation.

| Mechanism | Role | Rate substitution |
| --- | --- | --- |
| `ConfiguredRates` | Declared volumetric sources enter native species balances. | Rule graph or registered provider; outputs match `rate_unit`. |
| `PathwayMetabolism` | Maximize cᵀv subject to uptake, pathway and inventory constraints; exchange = Ev, growth = gᵀv. | Constant/Monod/transfer-cap bounds or provider-defined nonnegative uptake capacities. |
| `RateReconciledCulture` | Reconcile predicted rates with network and inventory constraints. | Rule graph or registered provider, with explicit rate bases. |

Pathways support input-defined dimensions; reversible pathways require supported
nonnegative coordinates. Pathway models need no rule graph unless chosen for an
optional provider and are not routed through kinetic reconciliation.

Reduced pathways optionally accept `secondary_optimization` with policy
`scaled-minimum-norm` and positive reference scales keyed by pathway identifier.
This minimizes the squared normalized flux on the primary optimal face, with
independent linearized optimality and original-constraint checks. It is an
explicit modeling preference, not a biological law. Omitting it preserves the
legacy alphabetical selection; existing examples do not configure this option.

Providers return finite named rates, units, identity and validity diagnostics.
Pathway capacities use `mol/(kgDW time)`, with time matching `flux_time_unit`,
then convert to the declared exchange basis. Invalid units, negative capacities
and forbidden extrapolation fail explicitly.

## Biological relationships

Named relationships expand into ordinary rule graphs during input normalization;
explicit expressions and nested relationships remain supported. Inputs are not
mutated and dependencies are not reordered. For example:

```json
{"identifier": "uptake", "expression": {
  "relationship": "saturating-uptake",
  "inputs": {
    "maximum": {"parameter": "rates.maximum"},
    "substrate": {"ref": "substrate_concentration"},
    "half_saturation": {"parameter": "affinities.substrate"}
  }
}}
```

| Relationship | Required inputs | Equation / conditions |
| --- | --- | --- |
| `saturating-uptake` | maximum, substrate, half_saturation | qmax S/(K+S), existing nonnegative-substrate Monod evaluation; K>0, qmax≥0 |
| `additive-inhibition` | rate, inhibitors: {concentration, constant} list | rate/(1+Σ I/K); K>0; additive, not multiplicative inhibition |
| `yield-sum` | terms: {rate, yield} list | Σ Yᵢqᵢ; Yᵢ≥0; rates may include signed maintenance offsets |
| `capacity-limited-rate` | rate, capacity | rate C/(C+rate); C>0, intended nonnegative rate; smooth attenuation, not clipping |
| `net-production` | production, yield, consumption | Y production−consumption; Y≥0 |
| `maintenance-uptake` | growth, yield, maintenance | growth/Y+maintenance; Y>0, maintenance≥0; yield must not already incorporate maintenance |
| `linear-transfer` | coefficient, equilibrium, concentration | coefficient×(equilibrium−concentration); coefficient≥0; absorption or stripping |

Constrained coefficients must be finite scalar literals or existing scalar
parameter references. State-dependent slots accept expressions or nested forms.
Forward references and duplicate output identifiers are rejected. Other hypotheses
use explicit graphs; no organism, parameter, objective or constraint is inferred.

Rule-graph species concentrations use mmol/L. Users must supply compatible units
within expressions and yields consistent with output bases; conversion occurs at
mechanism boundaries, not through a general dimensional-analysis engine. Record
parameter sources and validity domains. These forms are not complete biological
models: uptake alone does not determine growth yield or oxygen demand. Direct and
reconciled models use `model.kinetics.rule_graph` and explicit output mappings.

## Balances and numerics

Native extensive inventories obey inlet − outlet + transport + biological source
balances; concentrations follow from amounts and volume. Supported operations include batch/fed-batch feeds and sampling, declared
working volume, and continuous culture through the native volume controller. Reduced pathways couple biomass to optimized growth. Reconciliation
supports biomass/cell normalization, growth/product mappings, kinetic mortality,
network constraints and finite-step inventory feasibility without post-solve
consumption clipping. Unweighted, relative-regularized and unit-scaled objective
contracts remain distinct.

Adaptive integration, fixed-step propagation and optional IDA retain their existing
roles; IDA tests depend on the installed environment. Rate providers supply sources through the native balance interfaces. Isothermal examples do not qualify
thermodynamic or dynamic gas-headspace models.

Optional `operation.thermal` now connects all three biological formulations to
the existing native energy balance, with explicit biological heat [W], live
temperature [K], and adiabatic or constant-inlet utility operation. Omission
preserves isothermal behavior. Heat can read accepted native species sources
[kg/s]; thermal coefficients and temperature-dependent kinetics remain declared
in the existing JSON rather than inferred from organism identity. Physical feed
temperatures and enthalpy-conserving bolus mixing use the existing recipe.
The [configuration guide](../../doc/online_docs/bioreactors/USER_GUIDE.md#non-isothermal-operation-in-the-same-input-file)
states units, signs, property requirements and limitations. Synthetic thermal
qualification does not qualify experimental heat yields, gas enthalpy closure,
dynamic jackets or the physical capacity of temperature control. Prescribed
temperature reuses `Commons.temp_ufun` and the native simple/continuous
temperature controllers, with their existing ideal-control semantics. Bolus
mixing delegates to the unchanged native `Containers.Mixer`, rather than a
bioreactor-specific enthalpy inversion. Utility settings use native constructor
names; the earlier local flat spellings remain compatible.

## Qualification and limits

`tests/Bioreactor/test_bioreactor_qualification.py` retains separate parameterized
checks for population dynamics, networks, constraints, operations and coupled-model
references. `reconciliation_reference.py` supplies independent test mathematics,
not production code. Known-answer and captured-failure regression modules provide
additional coverage. Synthetic reconciliation inputs are grouped by verification purpose under
`tests/Bioreactor/fixtures/reconciliation`, separately from user examples.

Qualification covers analytic balances/projections, growth/death/limitation,
nonunique networks, nonlinear exposure/inventory feasibility, units, population
responses, larger networks, renamed species and numerical refinement. Tests force
headless plotting. Passing constraints, accounting checks and convergence does not
establish experimental predictivity or elemental closure: omitted elements, gases
and biomass composition must be accounted for explicitly.

| Example | Evidence and limitations |
| --- | --- |
| Generic chemostat | Analytical Monod equilibrium and washout; synthetic qualification, not experimental validation. |
| Generic batch/fed-batch | Synthetic pathways, feeds/sampling and reconciliation qualification. |
| Batch E. coli | Qualitative diauxic-growth reproduction, not independent quantitative validation. |
| Fed-batch E. coli F10/F11 | Direct/reconciled source-configured reproduction and measured trends; quantitative discrepancies and source feed-accounting interpretation remain. |

Historical culture fixtures are test data. Source-calibrated E. coli reproduction
is not prospective validation. Generality means configurable supported homogeneous
batch/fed-batch and synthetic chemostat families, not all organisms, network sizes or spatial models;
new regimes require biological specification and qualification. The relationship
library neither fits missing parameters nor introduces a new validity-domain engine.
This forward package does not deliver WP3.3 estimation/design workflows.

See the [user guide](../../doc/online_docs/bioreactors/USER_GUIDE.md) for construction
and substitution and [examples](../../examples/bioreactors/README.md) for notebook
and test commands. Packaging is checked by building a wheel and running the
package-data and public-builder tests against a non-editable installation outside
the checkout; this is a verification procedure, not a separate runtime API.

## Acceptance scope and remaining scientific work

The [delivery acceptance table](../../doc/online_docs/bioreactors/ACCEPTANCE.md)
maps every #361, #364 and #365 criterion to implementation, tests and remaining
scientific or review conditions.

The E. coli yeast-extract fed-batch process (source runs F10/F11) is the selected baseline. The source workbook identifies the nominal
profile and logged additions as the same glucose feed; the current one-counted
solution-volume inputs are retained as the published-model reproduction. The
logged feed is gravimetric, and the delivered glucose dose (about 83-102 g) cannot
be resolved from available records; it is recorded as input uncertainty. See the
[source audit and decisions](../../examples/bioreactors/ecoli_ye_fed_batch/README.md#feed-volume-source-audit).

Direct F10/F11 kinetics report source Eq. 19 oxygen demand through optional
native algebraic output fields. No dissolved-oxygen inventory, transfer/control
validation or oxygen-limitation feedback is claimed; this demand-plus-applicability
treatment is the WP3.1 decision because transfer and control records are
unavailable. The recorded DOT excursions prevent assuming universally
nonlimiting oxygen from the controller target alone.

`predict_measurements` supplies exact-time concentration predictions. The plain
JSON forward-record adapter adopts #351 identity, ordering, units, source and
parameter provenance requirements without replacing the existing arrays. It
preserves input snapshots, solver settings, source observations and optional
native state/algebraic fields for replay. The integrated native `PharmaPy.Records.CalibrationRecord` API is used by the
Doyle estimation demonstration for export, integrity checks and relocated replay.
The forward adapter and that synthetic demonstration do not establish two-case
empirical record adoption or close all of #351. Weighting and derivative
capabilities remain those of the inherited native estimator.

Synthetic chemostat dilution, Monod equilibrium and washout are qualified through
`ContinuousReactor`. This does not establish experimental continuous-culture
predictivity or implement selective cell retention/perfusion. WP3.2 supplies named kinetic selection, bounded positive corrections and a
native estimation handoff with a reproducible synthetic training/validation
demonstration. Broad empirical estimation and experiment design remain WP3.3
scope. A successful forward run does not establish those scientific outcomes. Original observations remain in
`reference/`; generated evidence, notebook outputs and historical published
comparison figures are not distributed.

## Shared-core integration

| Shared change | Why WP3.1 needs it | Verification |
| --- | --- | --- |
| Retained estimation safeguards | Preserve the existing local derivative-resolution check, reject failed bootstrap fits, and qualify covariance/interval interpretation while adopting the newer native estimator and record APIs. This is shared compatibility, not a separate bioreactor estimator or WP3.3 delivery. | `tests/test_estimation_uncertainty_safeguards.py`, native statistics tests and the bioreactor steady-culture information test. |
| Requested outlet rates and exact finite-step transport limit | Reserve withdrawal before metabolic feasibility; avoid smoothing down an exactly feasible outlet. Default assembly/limiting remains unchanged. | Chemostat depletion, volume and failure checks. |
| Owned depletion-rate arrays | Native buffers are reused; the boundary rate evaluator returns copies so retained derivatives are not overwritten (upstream's right-hand side already copies). | Aliasing regression, analytical CSTR balances and upstream backend tests. |
| Inlet rates passed only to requesting mechanisms | Feasibility includes incoming material rather than treating a supplied empty pool as unavailable. | Reconciliation supply and operating-transition cases. |
| Explicit depletion checks and limiter bypass | Fixed-step infeasibility is reported; adaptive depletion events (`ScipyBackend(inventory_boundaries=True)`) provide the configured boundary policy. | Material-rate limiting, small-pool tests and the 18-case backend equivalence check. |
| Paired transfer and outlet scaling | Species, working volume and transferred heat use the same accepted transfer amount. | `test_material_rate_limiting.py`. |
| Independent replay buffers | Reconstructing output histories cannot overwrite the live solve's numerical workspaces. | Public native solve and result-history tests. |
| Differential initialization and complete DAE residual packing | Initial derivatives do not evaluate algebraic constraints; every declared equation must be supplied. | `test_bioreactor_dae.py`, including optional IDA runs. |
| Copied IDA options and call-option precedence | Instances do not share mutable options, and explicit solve options take precedence. | Native DAE option and integration tests. |
| Differential-state mask | The DAE interface distinguishes differential derivatives from algebraic residuals. | DAE state-order and residual checks. |
| Deferred mechanism annotations | Preserves annotation import compatibility with the declared Python minimum. | Import tests; the current numerical environment is Python 3.11. |
| Recursive package discovery | Installed distributions include `Bioreactors` and `Metabolic`. | Non-editable wheel package-data/import checks outside the checkout. |
| Explicit callback binding | Module-level reconciliation and inventory callbacks replace nested definitions while retaining bounds, scales and mutable depletion state. | Reconciliation regressions, small-pool tests and the complete fallback suite. |
| Optional IDA test marker | New IDA integration checks can be selected separately from the core lane. | Explicit `assimulo` tests including native continuous dilution; existing upstream tests are unchanged. |

The unrelated crystallizer annotation edit is excluded.

**Backend consolidation (D4).** The bioreactor builder now uses upstream
`ScipyBackend`; the duplicate `SciPyBackend` class is removed. Two opt-in
capabilities were added to `ScipyBackend`, both off by default so every other
unit keeps upstream behavior:

- `jacobian_relative_step` / `jacobian_state_scale` options: a forward-difference
  Jacobian with fixed perturbations `step * max(|y|, scale)`, replacing scipy's
  adaptive probes, which can run away when rates carry optimizer noise.
- `inventory_boundaries=True`: states flagged `limit_negative_inventory` stop at a
  depletion event, are set to zero and restart. The model must limit uptake to
  available supply: a materially negative net rate at an empty inventory raises
  rather than being masked independently of growth or product formation. Only
  stationary rates within `boundary_tolerance / solve_duration` are zeroed.
  The vessel limiter is bypassed for that solve. Depletion roots are restarts, not
  unit events, and are not added to the output. Upstream state events, arming and
  restart budgets are unchanged.

The existing output policy is retained under the boundary option: constrained
values below -1e-12 raise, and values in [-1e-12, 0) are reported as zero. That
absolute cutoff is a carried-over numerical policy, not a proven
scale-independent roundoff bound.

Evidence: all 18 bioreactor inputs that select the SciPy backend (examples and
reconciliation fixtures, including dFBA, F10/F11 and the coupled depletion cases)
were solved with the removed class and with the consolidated backend. Seventeen
are bitwise identical in every reported field; F10 reconciliation differs by
4e-28 kg. Without the boundary option, seven depletion cases fail, so the option
is required, not cosmetic. Upstream `ScipyBackend` tests pass unchanged, and
`tests/test_scipy_backend_controlled_jacobian.py` covers the Jacobian option
(exact values, handoff, trajectory agreement, rejected inputs). The SciPy default
tolerances differ between the classes (upstream 1e-6/1e-6, scipy's own
1e-3/1e-6); the builder always declares both, and the one direct test that did
not now states scipy's values explicitly.

## Numerical solver selection and fast reactions

The existing process JSON can select `numerical.backend: "assimulo"` for
native CVode when the optional Assimulo environment is installed. For
`"scipy"`, the optional `numerical.method` selects `BDF`, `Radau`, `LSODA`,
`RK45`, `RK23` or `DOP853`; omission retains BDF. Existing numerical layouts
remain valid. Both adaptive backends require explicit `relative_tolerance`
and `absolute_tolerance`, and accept `maximum_step` with its time unit.
These are numerical choices, not kinetic parameters or physical assumptions.
There is no automatic method switch or tolerance adjustment after a failure.

The bioreactor builder opts CVode into the same inventory-rate feasibility
evaluator used by SciPy. A fast but feasible reaction is therefore not rejected
merely because its instantaneous rate would exhaust the inventory over the
vessel's unrelated default one-second limiter horizon. Consumption of a truly
empty pool remains an error. This opt-in policy leaves other direct CVode
callers' limiter behavior unchanged. Both callbacks return owned derivative
arrays so a later vessel evaluation cannot overwrite an earlier derivative.

CVode does not acquire SciPy's depletion-root projection: it retains its
native adaptive integration and checks returned inventories against the
existing absolute negative-inventory budget. It returns smaller negative
numerical errors unchanged; it does not silently clip them. A failure requires
inspection of the model and numerical convergence, not removal of the guard.
The inherited fixed `1e-12` cutoff in native state units remains a limitation,
not a scale-independent guarantee of positivity.

`tests/Bioreactor/test_bioreactor_adaptive_inventory.py` verifies all exposed
methods against an analytical first-order conversion, sequential-reaction
transients and conservation, rejection of an infeasible empty-pool sink,
derivative ownership, and CVode reaction-heat and cold-feed restart balances.
These are numerical verification cases; they do not establish experimental
parameter accuracy or biological validity.

### Explicit inventory acceptance

Optionally add an absolute error requirement to the existing `numerical` mapping:

```json
"inventory_acceptance": {
  "mass_j": {"unit": "ng", "value": 1.0}
}
```

This example is exactly the inherited `1e-12 kg` budget expressed as one
nanogram; it is a compatibility example, not a recommended accuracy for every
process. Choose requirements from the physical quantities that need to be
resolved, then independently qualify the solver settings by convergence.
An integration `absolute_tolerance` is not a guaranteed global error bound.

State names, native units and component order come from native solver metadata.
`value` applies uniformly to that state's components. Alternatively, `values`
is a mapping naming **every** indexed component, including the carrier; its
JSON key order does not matter. Scalar mechanism inventories can be targeted
by their unique native state names. Ambiguous, unknown or unconstrained states,
unsupported unit conversions, incomplete component mappings, nonfinite values
and negative budgets are rejected. Zero explicitly requires nonnegative output.
The declaration can be omitted or null, but an empty mapping is an error.

`Bioreactors.numerical.inventory_limits` converts declared units and binds the
requirements after native compilation. One native `check_inventory` helper
checks both adaptive backends, reporting the state/component, time, value, unit
and limit on failure. Unspecified states retain the compatibility budget.
With explicit acceptance, reported negative values within budget remain visible:
no final output clipping is performed. Omission preserves legacy behavior,
including SciPy's existing final clipping. SciPy's depletion-root algorithm is
unchanged in both cases. The check covers reported states, not all internal
solver stages, and does not establish strict positivity or conservation.

Acceptance limits never change physical empty-pool feasibility, kinetic
parameters, solver tolerances or the initial-inventory rules. A permissive
output budget cannot authorize consumption from an empty pool.

### Independent convergence trials

`Bioreactors.verification.run_convergence` runs a fresh native assembly for each
explicit numerical setting and retains every success or failure. Supply a
common `report_times` grid in seconds and an explicit `reference_index`; there
is no automatic retry, best-run selection or replacement of a failed reference.
Overrides cannot alter the process or its inventory acceptance requirements.
The existing combined and separate input layouts are supported.

The returned records include raw liquid component masses, volumes, temperatures,
minimum masses, total-mass changes and differences from the selected reference
in their respective units. When both event sides are present in native
histories, the post-event value is selected without interpolation; native
terminal-event semantics are preserved. If the process genuinely conserves total liquid mass, supply
the independently known `conserved_mass_kg` to obtain a conservation residual.
Otherwise leave it unset: a fed or withdrawn inventory is not assumed constant.
No extra input file or production integration path is introduced. Agreement
between trials is numerical evidence, not experimental validation.

## Interchangeability evidence

`test_bioreactor_mechanism_interchange.py` runs the same generic batch and
fed-batch cases with configured kinetics and pathway LP mechanisms through the
public builder. Both routes are checked independently against exponential biomass
growth and the nutrient/feed balance, with two biomass formula weights testing
the configured species basis. These cases end before depletion. They qualify
this substitution boundary, not arbitrary biological equivalence.

## WP3.2 kinetics and estimation handoff

Named Monod, Haldane/Andrews and Pirt selections expand into the same existing
operator graph as the corresponding mathematical relationships. Optional
exponential-saturation corrections preserve the underlying rate's sign and
zeros; zero log correction recovers the mechanistic declaration. Parameters,
concentration bases and provider validity domains remain explicit inputs.

`simulate_measurements` reuses native construction, solution histories and
observation mapping. It binds existing parameter paths before graph expansion
and returns declared field columns at exact requested times [s], with fresh
input copies and native state for each trial. The shared estimator owns
weighting, residual alignment and numerical parameter derivatives. The
independent concentration-balance and constrained-fit tests qualify this
handoff without introducing a new reactor, optimizer or sensitivity engine.
