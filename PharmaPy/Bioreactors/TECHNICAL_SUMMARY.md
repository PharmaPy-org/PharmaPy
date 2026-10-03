# Native bioreactor forward modeling

## Architecture and model selection

`case.json + mechanism.json + thermo.json → build_bioreactor → native phase +
mechanism + BatchReactor/SemiBatchReactor → solve_unit → DynamicResult`.

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
balances; concentrations follow from amounts and volume. Supported batch/fed-batch
operations include scheduled feeds, well-mixed sampling and declared working
volume. Reduced pathways couple biomass to optimized growth. Reconciliation
supports biomass/cell normalization, growth/product mappings, kinetic mortality,
network constraints and finite-step inventory feasibility without post-solve
consumption clipping. Unweighted, relative-regularized and unit-scaled objective
contracts remain distinct.

Adaptive integration, fixed-step propagation and optional IDA retain their existing
roles; IDA tests depend on the installed environment. Rate providers supply sources through the native balance interfaces. Isothermal examples do not qualify
thermodynamic or dynamic gas-headspace models.

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
| Generic batch/fed-batch | Synthetic pathways, feeds/sampling and reconciliation qualification. |
| Batch E. coli | Qualitative diauxic-growth reproduction, not independent quantitative validation. |
| Fed-batch E. coli F10/F11 | Direct/reconciled source-configured reproduction and measured trends; quantitative discrepancies and source feed-accounting interpretation remain. |

Historical culture fixtures are test data. Source-calibrated E. coli reproduction
is not prospective validation. Generality means configurable supported homogeneous
batch/fed-batch families, not all organisms, network sizes or spatial models;
new regimes require biological specification and qualification. The relationship
library neither fits missing parameters nor introduces a new validity-domain engine.
This forward package does not deliver WP3.3 estimation/design workflows.

See the [user guide](../../doc/online_docs/bioreactors/USER_GUIDE.md) for construction
and substitution and [examples](../../examples/bioreactors/README.md) for notebook
and test commands. Packaging is checked by building a wheel and running the
package-data and public-builder tests against a non-editable installation outside
the checkout; this is a verification procedure, not a separate runtime API.

## Remaining acceptance work

F10/F11 is the selected baseline. Its feed-volume interpretation still needs
domain review.

Direct F10/F11 kinetics report source Eq. 19 oxygen demand through optional
native algebraic output fields. No dissolved-oxygen inventory, transfer/control
validation or oxygen-limitation feedback is claimed. The recorded DOT excursions
prevent assuming universally nonlimiting oxygen from the controller target alone.

`predict_measurements` supplies exact-time concentration predictions. The plain
JSON forward-record adapter adopts #351 identity, ordering, units, source and
parameter provenance requirements without replacing the existing arrays. It
preserves input snapshots, solver settings, source observations and optional
native state/algebraic fields for replay. The shared #346 object API remains
pending upstream; correlated weights and sparse-Jacobian work (#237/#236) are
not implemented here. This forward adapter does not close all of #351.
Continuous/chemostat qualification remains future work. WP3.2 affine/hybrid implementations and WP3.3
calibration/design are excluded. A successful forward run does not close those
acceptance gaps. Source observations in `reference/F10.json` and `F11.json` remain;
notebook outputs and historical published comparison figures are not distributed.

## Shared-core integration retained from the reference branch

| Shared change | Why WP3.1 needs it | Verification |
| --- | --- | --- |
| Inlet rates passed only to requesting mechanisms | Feasibility includes incoming material rather than treating a supplied empty pool as unavailable. | Reconciliation supply and operating-transition cases. |
| Explicit depletion checks and limiter bypass | Fixed-step infeasibility is reported; adaptive depletion events provide the configured boundary policy. | Material-rate limiting and small-pool tests. |
| Paired transfer and outlet scaling | Species, working volume and transferred heat use the same accepted transfer amount. | `test_material_rate_limiting.py`. |
| Independent replay buffers | Reconstructing output histories cannot overwrite the live solve's numerical workspaces. | Public native solve and result-history tests. |
| Differential initialization and complete DAE residual packing | Initial derivatives do not evaluate algebraic constraints; every declared equation must be supplied. | `test_bioreactor_dae.py`, including optional IDA runs. |
| Copied IDA options and call-option precedence | Instances do not share mutable options, and explicit solve options take precedence. | Native DAE option and integration tests. |
| Differential-state mask | The DAE interface distinguishes differential derivatives from algebraic residuals. | DAE state-order and residual checks. |
| Deferred mechanism annotations | Preserves annotation import compatibility with the declared Python minimum. | Import tests; the current numerical environment is Python 3.11. |
| Recursive package discovery | Installed distributions include `Bioreactors` and `Metabolic`. | Non-editable wheel package-data/import checks outside the checkout. |
| Explicit callback binding | Module-level reconciliation and inventory callbacks replace nested definitions while retaining bounds, scales and mutable depletion state. | Reconciliation regressions, small-pool tests and the complete fallback suite. |
| Optional IDA test marker | New IDA integration checks can be selected separately from the core lane. | Three explicit `assimulo` tests; existing upstream tests are unchanged. |

The unrelated crystallizer annotation edit is excluded. Backend consolidation
remains deferred under the compatibility-first decision. `SciPyBackend` retains
its reference numerical behavior; upstream `ScipyBackend` remains unchanged.

The D4 trial exposed a numerical-policy conflict: the reference adaptive backend
rounds negative output residues in its accepted numerical band to zero, while the
planned consolidation requires no output clipping. Removing that step produced
roundoff-scale negative inventories (one observed value was approximately
`-6.4e-27` in native units) that the next recipe segment correctly rejected as an
invalid initial inventory. Consolidation is therefore incomplete; the duplicate
class is retained and no equivalence claim is made. The compatibility-first policy retains the existing bioreactor backend until
native integration preserves all required capabilities and demonstrated examples.
This retention does not establish that its absolute clipping cutoff is a rigorous
roundoff bound. Any consolidation still requires independent Jacobian, depletion,
event and failure checks, plus unchanged upstream behavior with options off.

A direct native-backend check completed the generic batch and fed-batch examples
on identical output grids, with maximum species-inventory differences of about
`5.1e-12 kg`. This is a compatibility observation, not an independent correctness
proof. The batch dFBA and F10/F11 inputs explicitly configure controlled Jacobians,
which upstream `ScipyBackend` does not support. Their options must not be silently
dropped merely to replace the backend class.

## Interchangeability evidence

`test_bioreactor_mechanism_interchange.py` runs the same generic batch and
fed-batch cases with configured kinetics and pathway LP mechanisms through the
public builder. Both routes are checked independently against exponential biomass
growth and the nutrient/feed balance, with two biomass formula weights testing
the configured species basis. These cases end before depletion. They qualify
this substitution boundary, not arbitrary biological equivalence.
