# Analytic balances

This is a manufactured verification problem, not a biological dataset or a CHO
reproduction. It checks the existing shared reconciliation, unit conversion,
feeding, sampling and reactor balances against an independently derived answer.
No runtime architecture changes or process-specific Python adapters are required.

## Exactly specified optimization

The three nonnegative fluxes are nutrient uptake `u`, internal conversion `c`,
and product secretion `p`, each bounded above by 10. The two intracellular
balances are:

```
u - c = 0
c - p = 0
```

Thus every feasible flux vector is `(v, v, v)`. With the configured `unweighted`
policy the minimized objective is proportional to

```
(u - target_u)^2 + (p - target_p)^2
```

The shared common residual scale changes neither the weights nor the minimizer.
There is no internal-flux penalty under this policy. Consequently the unique
unconstrained-by-inventory answer is

```
v = min(10, max(0, (target_u + target_p)/2))
```

| Configuration | Uptake target | Product target | Expected v | Initial nutrient |
| --- | ---: | ---: | ---: | ---: |
| consistent | 2 | 2 | 2 | 10 mmol |
| conflicting | 4 | 2 | 3 | 10 mmol |
| depletion | 4 | 2 | 3 until nutrient limits it | 0.12 mmol |

Flux units are mmol/(million viable cells day). Viable inventory starts at one
million cells. Growth and death are explicitly zero, allowing exact population
and material balances. Nutrient and product are formal pools with equal supplied
molecular weights of 100 g/mol. Product is the culture mechanism's separate
`product_g` inventory, not a second liquid species. These bookkeeping choices
verify a balanced conversion; they do not assert an organism's elemental or
biomass composition. The network objective field is required metadata; this
reconciliation policy does not maximize it.

## Fed-batch balances and operations

Time is in days; volume is in L. Continuous feed is 0.1 L/day from 0 to 0.5 day,
then 0.2 L/day until it stops at day 1. Feed contains 2 mmol/L nutrient in the
consistent/conflicting cases and is nutrient-free in the depletion case.
At day 0.75, a 0.1 L well-mixed sample removes nutrient, product and viable cells
in proportion to volume. At day 1, a 0.1 L complete-solution bolus containing
3 mmol/L nutrient adds 0.3 mmol. All stocks declare or inherit 1 kg/L density.
Final volume is 1.15 L and final viable inventory is 10/11 million cells.

For nutrient amount A (mmol), product mass P (g), viable inventory N (million
cells), volume V (L), feed F (L/day), and stock concentration Cf (mmol/L):

```
dA/dt = F*Cf - v*N
dP/dt = 0.1*v*N
dN/dt = 0
dV/dt = F
```

Between events and before depletion, these balances have linear exact solutions.
The depletion case has no continuous nutrient supply, so the exact consumed
amount over an interval dt is `min(A_initial, 3*N*dt)`. Uptake stops at day 0.04,
resumes after the day-1 bolus, and stops again at day 1.11. The native fixed-step
inventory constraint limits consumption to the available amount without clipping
negative concentrations afterward.

The fixed step is 0.01 day. This benchmark has exact grid-aligned event and
depletion times, so its computed extensive balances can be checked against the
continuous solution. The IVCD exposure integral uses the solver's first-order
quadrature: its exact increment is `N/1000 * log(V_end/V_start)/F`, or
`N*dt/(1000*V)` when F is zero. Tests separately check its error and reduction
under step refinement rather than treating that quadrature as exact.

## Evidence and limits

The known-answer test computes an independent reference for each configuration.
The reference
uses closed-form balances, without the production rule evaluator, optimizer,
unit converter or event handlers. Verification checks all event sides, finite
nonnegative inventories, molecular-amount conservation, the projected rates,
substrate exhaustion and resumed production. Run:

```
python -m pytest tests/Bioreactor/test_bioreactor_known_answer.py -q
```

This establishes a controlled check of the **unweighted** reconciliation route
with fixed-step inventory feasibility. It does not validate relative-regularized
reconciliation, growth-coupled nonlinear constraints, unequal measurement weights,
realistic mammalian kinetics, experimental prediction or off-grid depletion timing.
Those are subsequent benchmarks, not conclusions drawn from this constructed case.
