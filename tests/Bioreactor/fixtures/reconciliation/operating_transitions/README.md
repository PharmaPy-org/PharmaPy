# Operating transitions

## Acceptance criteria frozen before the campaign

Use the two-branch network from Network structure: exchanges `[x+y,y,x,2y,0]`, with targets
`[1.5,.5,1,1,0]`. Independently enumerate its two-dimensional quadratic-program
active sets. Integrate piecewise constant inventory rates to analytically located
depletions and scheduled jumps. Relative-policy cases add the declared scaled
internal penalty and target envelopes. Growth/death uses independently integrated
population/exposure balances. All values are manufactured, not fitted.

State scales, in order A mmol, B mmol, byproduct mmol, product g, viable/dead million
cells, volume L, IVCD million-cell-day/mL, are `[1,1,1,.1,1,1,1,.0001]`.
Maximum scaled discrete-equation and adaptive continuous-trajectory error: `1e-6`.
Scaled closure feasibility: `1e-8`; absolute saved material negativity: `1e-9 mmol`.
Independent reference refinement must differ by less than one fifth of the budget.
Analytic zero-growth references have no time-integration error.

For the first-order fixed-step method, use h=0.002, 0.001, 0.0005 day. Compare each
against its own discrete balance to `1e-6`, and separately require the finest
continuous trajectory error below `1e-3` (0.1% of the declared state scales), with
error decreasing across refinement unless all three already satisfy `1e-6` (an
exact integrated balance can reach the numerical error floor). This deliberately different operating budget
allows first-order quadrature across depletion; it does not relax closure or mass
accounting checks. Depletion observation brackets must be no wider than 0.0005 day
at the finest grid and locate analytic transitions within one such interval.
Adaptive refinements vary tolerance and maximum step separately and retain the
`1e-6` trajectory criterion. Endpoint and jump accounting retain `1e-6` on both
backends. The final endpoint excludes operations, as already documented.

All simulations must finish; per-run diagnostic timeout 60 s for this small system
is a campaign guard, not an industry runtime standard. Record actual runtimes and
closure calls. Check supply below/equal/above demand; stop/restart/composition;
independent named inlets; sampling/bolus near depletion; coincident and nearby
operations; initial/final events; invalid operations; replay; equivalent units and
renamed identifiers; native and working volume. Relative-policy starvation may
correctly be infeasible when positive lower bounds prohibit the required zero rate.

## Packaged cases and results

| Input folder | Purpose |
| --- | --- |
| `single`, `competing` | Off-grid depletion of one or two competing resources |
| `supply_below`, `supply_equal`, `supply_above` | Initially empty nutrients supplied below/at/above demand |
| `schedule`, `multiple` | Stop/restart/composition and independent named inlets |
| `bolus_sample`, `ordered`, `reversed`, `endpoints` | Near-depletion jumps, order-dependent jumps, initial/final operations |
| `combined` | Mixed schedule, coincident operations and a separately resolved nearby event |
| `relative` | Relative-regularized policy with declared physical residual floor |
| `growth`, `growth_supply` | Growth/death with abundant and feed-limited nutrients |

## Shared implementation corrections

Native inlet rates now reach metabolic availability without duplicating the native
feed balance. Adaptive depletion handling uses tolerance-bounded numerical event
locations and stationary boundary rates to avoid chatter; its correction is bounded
per transition, and small positive initial inventories retain their relative scale.
A pathway's constant capacity remains constant while inventory/supply constraints
limit uptake; explicit Monod and depletion rules remain intact. Assembly reset
clears scheduled inlet state, initial events occur after the native reset snapshot,
and event/grid times differing only by conversion roundoff retain one numerical
instant and the user's declared order. Distinct nearby events remain separate.

These are shared numerical/workflow corrections, not process-specific kinetics or
new JSON fields. The benchmark demonstrates numerical robustness over its declared
conditions. It does not establish Reddy-scale network coverage, biological
validation, or universal reliability for every possible configuration.


## Retained verification
