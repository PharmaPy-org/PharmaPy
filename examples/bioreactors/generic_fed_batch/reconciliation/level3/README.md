# Level 3: branching fed-batch reconciliation

This manufactured benchmark tests two independent conversion rates, competing
nutrients, two products, growth coupling and nonunique internal pathways. All
process information is in the same standard JSON structures used by the other
examples. No local adapters, new schema or alternate execution path are used.

Run all six cells of `workflow.ipynb`. Step 1 defaults to `inputs/switching` and
`outputs/switching`; change both paths to select another case. The notebook only
loads, builds, simulates, tabulates, plots and exports. Each output directory
contains CSV/PNG/SVG trajectories plus an independently generated reference CSV
and verification JSON. The latter records numerical settings, environment, input
and closure hashes, errors, timing and applicable convergence/nonuniqueness evidence.

## Declared equations and cases

The two branches are `x: A → (1−b) P + b biomass` and
`y: A+B → (2−b) Q + b biomass`. Thus the five exchange rates are:

```
ua = x+y; ub = y; p = (1−b)x; q = (2−b)y; g = b(x+y)
```

All rates are mmol/(million viable cells day), bounded between zero and 10.
The unweighted objective minimizes squared differences of these five exchange
rates from their kinetic targets. Common residual scaling does not change the
minimizer. Internal branches have no penalty. The network objective field is
metadata; this run does not maximize growth or product.

| Case | Targets [ua, ub, p, q, g] | Expected unconstrained [x, y] | Distinction |
| --- | --- | --- | --- |
| compatible | [1.5, 0.5, 1, 1, 0] | [1, 0.5] | Targets satisfy all balances |
| conflicting | [4, 1, 1, 4, 0] | [17/11, 21/11] | Unique least-squares compromise |
| switching | [1.5, 0.5, 1, 1, 0] | [1, 0.5] before depletion | Two small nutrient inventories and replenishment |
| growth | [1.5, 0.5, 0.8, 0.9, 0.3] | [1, 0.5] | b=0.2; accepted growth g/2=0.15/day; death=0.05/day |
| nonunique | [1.5, 0.5, 1, 1, 0] | [x+x_copy, y]=[1, 0.5] | Duplicate x branch; every split from 0 to 1 is optimal |

For all other cases b=0 and growth/death are zero. Nutrient and product pools
have formal MW=100 g/mol; biomass content is explicitly 2 mmol/million cells.
These are synthetic bookkeeping properties, not inferred biological composition.

Let A, B, Q be mmol, P product g, N and D viable/dead million cells, V volume L,
and I IVCD million cell day/mL. Between events, with time in days:

```
dA/dt = −ua N; dB/dt = −ub N
dQ/dt = q N; dP/dt = 0.1 p N
dN/dt = (g/2−kd)N; dD/dt = kd N
dV/dt = 0.1; dI/dt = N/(1000 V)
```

Start with V=1 L, N=1 million cells, zero other biological/product states and
A=B=10 mmol, except switching starts with A=0.6 and B=0.1 mmol. Nutrient-free
liquid enters continuously at 0.1 L/day. At day 0.5 a 0.1 L bolus adds A=0.6
and B=0.2 mmol. At day 0.75 a 0.1 L sample removes all extensive material and
population inventories proportionally, leaving IVCD unchanged. Run to day 1.
The formal conserved amount is `A+B+Q+10P+2(N+D)`, with feed/sample adjustments.

In switching, B depletes at day 0.2; y stops while x becomes 1.25. A then
depletes at day 0.44. Both branches resume after the day-0.5 bolus and both
nutrients are exhausted at day 0.9. These times are grid-aligned by construction.

## Independent verification

`tests/Bioreactor/test_bioreactor_qualification.py` enumerates the possible active sets
of the reduced two-variable convex quadratic problem using linear algebra. It
does not call the production optimizer to determine the answer. It checks
objective value, feasible fluxes and mapped reactor rates. For nonunique cases
only the total x flux and external rates are unique; the report explicitly
records the analytical internal intervals rather than imposing one arbitrary split.
This is an analytical benchmark result, not automatic runtime FVA.

Zero-growth forward runs use a 0.005-day fixed step. Their independent discrete
reference applies the finite-step nutrient constraints `ua ≤ A/(N Δt)` and
`ub ≤ B/(N Δt)` and exact extensive increments. IVCD uses the declared numerical
quadrature. Separately, an independent continuous piecewise reference integrates
to nutrient exhaustion and uses the exact volume-dependent IVCD integral.
The growth case uses adaptive SciPy integration and an independent DOP853 ODE
reference with rtol=2e-13, atol=2e-15 and maximum step 0.0025 day.

The frozen acceptance limit is maximum scaled trajectory error <1e-6, using
scales [10, 10, 2, 0.1, 1, 0.1, 1, 0.001] for [A, B, Q, P, N, D, V, I].
Feasibility residuals are checked at 1e-9, unscaled objective differences at 1e-9,
and individual projected fluxes at 2e-6. Renaming species/reactions and permuting
network and species order must preserve trajectories within the same 1e-6 budget.

| Case | Maximum scaled error against its stated reference |
| --- | ---: |
| compatible | 7.65e-11 |
| conflicting | 5.03e-9 |
| switching | 3.02e-11 |
| growth | 1.63e-7 |
| nonunique | 1.27e-9 |

Maximum formal conservation error is below 1.5e-13 mmol-equivalent. Notebook runs
took approximately 1.1–1.4 seconds each, including plotting/export on the recorded
environment. Fixed steps of 0.01, 0.005 and 0.0025 day halve the continuous-reference
error for switching and growth. This is a convergence result, not a claim that
those fixed steps attain the 1e-6 continuous-trajectory criterion. Independently
tightened adaptive tolerances and maximum step also retain the growth accuracy.

## Diagnosed numerical change and limits

The original reconciliation SLSQP stopping tolerance (`ftol=1e-10`) allowed
exchange errors up to about 7e-6 and scaled trajectory/permutation differences
above 1e-6. Tightening the existing stopping tolerance to 1e-12 resolves these
failures without changing the mathematical problem, constraints or interface.
This is the only shared runtime change for Level 3. The inner tolerance is not
currently a JSON option; changing ODE tolerances cannot correct a static QP error.
Acceptance criteria were not loosened. Broader regression results and any open
failures are recorded in `PharmaPy/Bioreactors/TECHNICAL_SUMMARY.md#qualification-and-limits`.

This verifies the stated unweighted branching configurations. It does not prove
all networks, weighted policies, growing-population depletion, off-grid transition
timing, large-network performance or experimental CHO prediction. Those remain
later coverage, not conclusions of this synthetic benchmark.

```sh
python -m pytest tests/Bioreactor/test_bioreactor_qualification.py tests/Bioreactor/test_bioreactor_known_answer.py -q
```


## Retained verification

Historical campaign reports and intermediate output trees have been removed.
The input triples, forward notebook and independent numerical checks remain.
Run `python -m pytest tests/Bioreactor/test_bioreactor_qualification.py` from the
repository root. The notebook regenerates forward trajectory outputs; the tests
check the declared equations directly rather than relying on packaged reports.
