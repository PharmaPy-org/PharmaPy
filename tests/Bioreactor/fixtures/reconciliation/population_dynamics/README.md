# Population dynamics

This manufactured benchmark verifies growing and dying populations coupled to
reconciled nutrient uptake and product formation. It uses the existing standard
JSON → `build_bioreactor` → `assembly.solve` → native reactor workflow. No shared
runtime code or JSON schema changes were required for this benchmark.

## Verification

```sh
python -m pytest tests/Bioreactor/test_bioreactor_qualification.py tests/Bioreactor/test_bioreactor_known_answer.py -q
```

## Declared model

All constants are synthetic, explicitly configured benchmark properties, not
fitted biological coefficients. The intracellular balances constrain uptake `u`,
conversion `v`, product secretion `p` and biomass formation `g`:

```
u = 1.2 v;  p = 0.2 v;  g = v
```

Fluxes are mmol/(million viable cells day). The declared biomass content is
β = 2 mmol/million cells, so accepted specific growth is μ = g/β, in 1/day.
The unweighted reconciliation minimizes squared departures of `u`, `p` and `g`
from their supplied kinetic targets, subject to these balances and flux bounds.
It does not maximize the network objective metadata.

| Variant | Growth target (1/day) | Death (1/day) | Accepted growth |
| --- | ---: | ---: | --- |
| growth | 0.4 | 0 | 0.4/day |
| growth_death | 0.4 | 0.1 | 0.4/day |
| constrained_growth | 0.4 | 0.1 | 0.25/day because uptake ≤ 0.6 |
| monod | 0.44 C/(1+C) | 0.1 | Concentration-dependent target, with C in mmol/L |

Let A denote nutrient mmol, N and D viable and dead million cells, P product g,
V volume L, and I IVCD in million cell day/mL. Between events:

```
dA/dt = 2 F − 2.4 μ N
dN/dt = (μ − kd) N
dD/dt = kd N
dP/dt = 0.04 μ N
dV/dt = F
dI/dt = N/(1000 V)
```

Initial values are A=10, N=1, D=P=I=0 and V=1. Product molecular weight is
100 g/mol. All runs last 1.5 days. Nutrient feed (2 mmol/L) enters at 0.1 L/day,
increases to 0.2 L/day at day 0.5, and stops at day 1. A 0.1 L sample at day
0.75 removes A, N, D and P proportionally but leaves I unchanged. A 0.1 L bolus
at day 1 adds 0.3 mmol nutrient. Final volume is 1.15 L.

The formal conserved amount is `A + 10 P + 2(N+D)`, adjusted for feeds and
sampling. This is a declared bookkeeping conservation law, not a verification of
organism-specific elemental composition or lysis chemistry.

## Evidence and acceptance

All six trajectories are compared against independent DOP853 balances with
rtol=2e-13, atol=2e-15 and maximum step 0.0025 day. Constant-rate population and
material references are also checked against closed-form exponential solutions.
The maximum absolute errors are divided by fixed scales `[10, 1, 0.2, 0.1, 1,
0.001]` for `[A, N, D, P, V, I]`; every scaled error must be below 1e-6.

Times include plotting/export and depend on the machine. The tighter adaptive
settings used for this accuracy check make the changing-target Monod case more
expensive; these times are not a general throughput guarantee.

Additional tests cover the known accepted flux vector, finite nonnegative states,
formal conservation, sampling continuity, zero-cell and zero-nutrient boundaries,
and equivalent mmol/µmol and amount/specific-growth configurations. Unit-equivalent
network columns, targets and bounds are transformed consistently.

Fixed steps of 0.02, 0.01 and 0.005 day each reduce the maximum scaled error by
at least 35% on halving the step. This demonstrates convergence, not attainment
of the adaptive 1e-6 accuracy criterion at these step sizes. Fixed-step death and
exposure updates introduce discretization error even when growth is constant.

## What this establishes

Population dynamics verifies the configured population–flux feedback, accepted-growth
conversion, death, exposure, feeds and sampling on a known-answer network.
It preserves the separate Analytic balances depletion/replenishment checks. It does not
establish realistic CHO prediction, arbitrary network correctness, weighted
reconciliation, or accurate timing of depletion during growing-population runs.
Multiple independent fluxes and competing nutrients are covered by the
[network-structure fixtures](../network_structure/README.md).


## Retained verification
