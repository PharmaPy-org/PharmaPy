# Level 4: reconciliation policy validation setup

## Cases and execution

Every input folder contains the same standard `case.json`, `mechanism.json` and
`thermo.json` structure. The network, initial inventories and feed/sample schedule
reuse Level 3's compatible two-branch model. All properties are synthetic and
declared; no biological fitting is involved.

| Folder | Change from the relative-regularized baseline | Validation target |
| --- | --- | --- |
| relative | Relative residuals, internal penalty, ±50% target bounds | Exact configured optimum; feasible targets need not remain unchanged |
| relaxed | Conflicting targets and uptake cap 1.5; allow ±100% envelope | First envelope infeasible; second feasible; record accepted envelope |
| untargeted | Remove q from kinetic reconciliation | q retains network/inventory constraints but no target penalty or target-derived bounds |
| normalization | Exclude q only from the internal normalization sum | q remains targeted; normalization exclusion is distinct from untargeting |
| near_zero | ub=1e-8, q=2e-8 | Declared physical floor and small target-derived bounds |
| redundant | Add a balance row equal to twice the A row | Equivalent feasible space and optimum |
| equivalent_units | Uniform mmol→µmol transformation of flux targets, bounds and declared rate bases | Same physical optimum despite different numerical magnitudes |
| nonlinear | Growth-dependent exposure with small A/B inventories; one 0.1-day step | Independent scalar-profile reference, four starting conditions and endpoint balances |
| floor_mmol / floor_umol | Equivalent tiny physical targets expressed in different units | Verify small-flux conditioning and physical floor equivalence |
| infeasible | ua upper bound zero, p lower bound one | Explicit physical/network infeasibility; no fabricated trajectory |

Run the validation checks:

```sh
python -m pytest tests/Bioreactor/test_bioreactor_qualification.py -q --tb=short
```

For forward simulation, run the six cells of `workflow.ipynb`; its default is
`inputs/relative`. Change both input and output paths in Step 1 to select another
case. It exports only forward CSV/PNG/SVG trajectories. The intentionally
infeasible case raises an error. The notebook is executed for `relative`. All ten feasible input sets have forward
CSV/PNG/SVG exports and independent reference CSVs. The intentionally infeasible
case has an error report, not fabricated trajectories. Per-case `verification.json`
records input/runtime hashes, environment, errors and wall time.

## Independent problem and frozen criteria

For branch rates z=[x,y], the exchange vector is
`f=[x+y, y, x, 2y, 0]`, ordered [ua,ub,p,q,g]. For targeted exchanges T:

```
f = declared 1e-6 mmol/(million cells day), converted to source coordinates
s_i = max(abs(target_i), f)
L = max(abs(sum(target_i for targeted, nonexcluded i)), f)
J = (x²+y²)/L² + sum(((f_i−target_i)/s_i)² for i in T)
```

At envelope e, intersect each targeted reaction's declared bounds with
`[(1−e)target_i, (1+e)target_i]`. These fixtures use e=0.5 initially and, for
relaxed only, e=1.0 subsequently. Targets are nonnegative here. The independent
reference enumerates unconstrained minima, edge minima and vertices of this
two-variable convex QP using linear algebra. It does not call the production
optimizer, rule evaluator or recovery routine. Coordinate scaling conditions the
reference without changing its objective. The reference is intentionally scoped
to these declared fixtures, not an alternate general runtime solver.

Acceptance is fixed before the campaign: projected flux errors ≤2e-6 in the
common mmol-based coordinates; objective agreement at absolute/relative 1e-8;
network and bound residuals <1e-8 in those coordinates; the expected first
feasible envelope must be selected. Forward tests for relative, relaxed and
untargeted compare all eight states against independent fixed-step balances,
with maximum scaled error <1e-6 using Level 3's scales and state ordering.

The recovery test deliberately fails the primary optimizer. Recovery
must return a solution matching the original independently specified problem or
raise `MetabolicNumericalError`. A passing explicit-failure branch does not prove
that recovery can solve every difficult problem. Record which branch occurred.

## Qualification and historical failure diagnosis

Seven convex references were recomputed using 40- and 80-digit Gaussian
elimination; comparisons met the predeclared one-fifth-error-budget limits.
This refines the linear solves for the independently assembled floating-point
problem, not a claim of exact arithmetic for every input coefficient.

The nonlinear fixture uses b=0.2, biomass content 2 mmol/million cells,
A=0.12 mmol, B=0.04 mmol, one million viable cells and death=0.05/day. For
s=x+y the population exposure is E(s)=0.05[1+exp((0.1s−0.05)0.1)]. Constraints
include s E(s)≤0.12 and y E(s)≤0.04. The reference reduces the problem to s:
at each s it solves the quadratic in y exactly within its feasible interval.
A 257-point scan with scalar minimization is checked against 2049 points.
Four production starting conditions and endpoint material/population checks
passed. This independently qualifies this fixture; it is not a general global
optimality theorem for arbitrary nonlinear networks.

The original injected primary-optimizer failure produced **explicit numerical failure**,
which passed the required safe-failure criterion. It did not demonstrate
successful recovery. The intentionally infeasible configuration was separately
rejected as infeasible. The synthetic source audit is complete for the
relevant Reddy equation/relaxation section and Khare SOM/KOM table.

| Original failed check (now repaired) | Pre-fix result | Diagnosis |
| --- | --- | --- |
| Redundant-row direct closure optimum | Returned x=1.032258, y=0.25, J=0.592616 instead of x=0.958977, y=0.498470, J=0.075513 | A feasible but suboptimal result is reported as successful; correct warm start reaches the reference optimum |
| Tiny-target direct unit equivalence | mmol-based direct closure raises a numerical error; µmol representation solves | Conditioning/absolute-tolerance sensitivity in the direct closure path |
| Tiny-target forward unit equivalence | Both forward runs solve their configured equations, but x differs by 36.2% of the 1e-7 reference scale | The numerical 1e-6 floor is fixed in source flux units; changing units changes relative weights and hence the optimization problem |

These are historical pre-fix results, retained to explain the repair. Current
unit-bearing floors yield matching tiny physical fluxes in both representations.
All original assertions remain active and now pass. The subsequent realistic
CHO investigation also exposed optimizer stagnation and motivated the general
balance-preserving retry. See `GENERAL_FIX_ANALYSIS.md` for the distinction
between original experiments and the implemented safeguards.

## Current general implementation

Fixed variables are substituted before rank reduction. Both LP and SLSQP use
scaled coordinates; accepted fluxes are checked for feasibility and stationarity.
A balance-preserving retry handles stagnation without changing biological
objectives or constraints. The shared optional `reconciliation_scaling` section
converts declared floors and normalization scales through the unit converter;
omission preserves legacy source-coordinate weighting. The common physical floor
is explicit in these fixtures. Mixed-dimensional policy scales require compatible
user declarations; no species-specific solver paths or coefficients were added.

## Retained verification

Historical campaign reports and intermediate output trees have been removed.
The input triples, forward notebook and independent numerical checks remain.
Run `python -m pytest tests/Bioreactor/test_bioreactor_qualification.py` from the
repository root. The notebook regenerates forward trajectory outputs; the tests
check the declared equations directly rather than relying on packaged reports.
