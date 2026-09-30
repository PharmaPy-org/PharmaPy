# Native bioreactor forward modeling

## Architecture and model selection

`case.json + mechanism.json + thermo.json → build_bioreactor → native phase +
mechanism + BatchReactor/SemiBatchReactor → solve_unit → DynamicResult`.

Biological identities, coefficients, networks and operating schedules belong in
user inputs. Mechanisms supply source terms and population states through native
interfaces; vessels, phases, streams, state packing and integration remain the
execution path. This documentation/test consolidation changes no runtime code.

| Mechanism | Role | Rate substitution |
| --- | --- | --- |
| `ConfiguredRates` | Declared volumetric sources enter native species balances. | Rule graph, affine surrogate, hybrid or registered provider; outputs match `rate_unit`. |
| `PathwayMetabolism` | Maximize cᵀv subject to uptake, pathway and inventory constraints; exchange = Ev, growth = gᵀv. | Constant/Monod/transfer-cap bounds or provider-defined nonnegative uptake capacities. |
| `RateReconciledCulture` | Reconcile predicted rates with network and inventory constraints. | Rule graph, affine surrogate, hybrid or registered provider, with explicit rate bases. |

Pathways support input-defined dimensions; reversible pathways require supported
nonnegative coordinates. Pathway models need no rule graph unless chosen for an
optional provider and are not routed through kinetic reconciliation.

Providers return finite named rates, units, identity and validity diagnostics.
Pathway capacities use `mol/(kgDW time)`, with time matching `flux_time_unit`,
then convert to the declared exchange basis. Invalid units, negative capacities
and forbidden extrapolation fail explicitly. Hybrid outputs are selected
independently: users must replace coupled outputs consistently, not assume that
one replacement recomputes another provider's dependent rates.

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
roles; IDA tests depend on the installed environment. Surrogates do not replace
integrators, events or feasibility constraints. Isothermal examples do not qualify
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
| Mammalian surrogate fed-batch | Synthetic viable/dead cells and product; mechanistic, affine and registered non-affine substitution with explicit domains, not validated CHO physiology. |

Historical culture fixtures are test data. Source-calibrated E. coli reproduction
is not prospective validation. Generality means configurable supported homogeneous
batch/fed-batch families, not all organisms, network sizes or spatial models;
new regimes require biological specification and qualification. The relationship
library neither fits missing parameters nor introduces a new validity-domain engine.
This forward package does not deliver WP3.3 estimation/design workflows.

See the [user guide](../../doc/online_docs/bioreactors/USER_GUIDE.md) for construction
and substitution and [examples](../../examples/bioreactors/README.md) for notebook
and test commands. No separate automated installation-verification program is claimed.
