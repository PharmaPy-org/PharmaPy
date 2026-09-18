# Bioreactor implementation: changes, mathematics, numerics, and scope

**Scope:** current WP3 implementation relative to pre-bioreactor PharmaPy commit `bd200f5e474243d60237cde344c174a2e4468137`. The endpoint incorporates parent `MultiPhaseVesselRefactor` commit `7f815b2`. “Existing” means present in the baseline; additions inherited from the parent are identified separately from bioreactor-specific mathematics. The endpoint file inventory and net code changes are maintained only in the [CHANGELOG](CHANGELOG.md).

## 1. Architectural relationship to PharmaPy

The implementation uses the native PharmaPy execution path:

`LiquidPhase → Mechanism → BatchReactor/SemiBatchReactor → solve_unit → DynamicResult`

The bioreactor layer constructs native phases and reactor objects, implements
biological source terms through the mechanism interface, and uses the vessel's
state compilation, balances, solver dispatch, and result reconstruction. The
metabolic layer supplies either reduced-pathway optimization or network rate
reconciliation to those biological mechanisms. Estimation reuses PharmaPy's
residual and covariance-weighting contract; experiment design repeatedly builds
and simulates the same native reactor assembly.

The native state model is extended only where the workflow requires a general
facility: differential/algebraic classification, ODE/DAE backend dispatch,
algebraic residual assembly, phase-aware mechanism registration, and physically
coupled finite-step inventory limiting. Existing phase properties, material and
energy balances, streams, results, controller interfaces, and residual weighting
remain the foundation. The legacy `Reactors.py` is outside the construction path
used by the examples.

## 2. Mathematical theory: existing versus added

### 2.1 Native balances and biological source terms

**Existing PharmaPy mathematics:** phase properties and state completion, species-inventory balances, inlet/outlet and interphase contributions, energy-balance machinery, state packing, and result reconstruction. Schematically,

$$
\dot m_i=\dot m_{i,\mathrm{in}}-\dot m_{i,\mathrm{out}}+\dot m_{i,\mathrm{transfer}}+r_{i,\mathrm{biology}}.
$$

**Added:** biological mechanisms supply the last term and additional cell/product states through the existing `StateKey`/`TransferResult` interface. Inventories, rather than concentrations, remain solver states; concentrations are reconstructed from amount and phase volume. Native mass is kg and solver time is seconds; biological inputs explicitly convert hours/days, mol/mmol, cell counts, and product mass. Both supplied cases are isothermal, so the inherited energy equations are not qualified here.

### 2.2 Reduced-pathway dynamic FBA

[PathwayMetabolism](mechanisms.py) repeatedly solves a configured optimization problem as the reactor state changes:

$$
\max_{v\ge0}\;c^Tv,\qquad A(C,X)v\le b(C,X),\qquad l\le v\le u.
$$

Here `v` denotes pathway activities; the exchange matrix `E` gives extracellular rates `Ev`, and a separate growth vector gives `mu = g^T v`. Uptake bounds include constant capacity, Monod capacity `q_max C/(K+C)`, and a transfer-cap form `min(q_max, k C_sat/X)`. Depletion can disable declared pathways. Biomass obeys `dB/dt = mu B`; exchange rates are multiplied by biomass and converted to kg/s. Optional declared transfer contributes `kLa (C_sat − C) V` in mol/s.

**New mathematics in PharmaPy:** this nested LP/dynamic coupling, its constraints, and biological conversions. **Reused:** the reactor ODE balance/phase/result path. The reduced-pathway LP does not impose a separate general intracellular `Sv=0` matrix; it uses the configured pathway/exchange representation. That is distinct from the general network closure below.

### 2.3 Reconciled metabolic network and inventory constraints

The CHO mechanism predicts kinetic targets `q`, then reconciles fluxes against an intracellular network:

$$
\min_v\;J(v)=\sum_{j\in I}(v_j/s_I)^2+
\sum_{j\in T}[(v_j-q_j)/s_j]^2,
\quad Sv=0,\quad l\le v\le u.
$$

`I` indexes regularized internal reactions; `T` indexes targets. Target scales are `max(abs(q_j), target_floor)`; the internal scale is a declared-exclusion-adjusted sum with the same floor. The default culture floor is `1e-6`. The network-only problem is convex quadratic, but may have nonunique solutions in unpenalized directions.

For fixed-step culture propagation, [inventory.py](inventory.py) additionally requires

$$
n^{k+1}=n^k+\Delta n_{\mathrm{other}}+\alpha(v)Mv\ge0,
\qquad
\alpha(v)=\frac{N_v^k h\,k_{\mathrm{flux}}}{2}
[1+e^{(\mu(v)-k_d)h}].
$$

`h` is days, `M` maps fluxes to extracellular sources, and `alpha` represents the same average viable-cell exposure used by propagation. When growth is linked to a network reaction, `mu(v)` makes these constraints nonlinear; successful SLSQP convergence is **not** a general global-optimum guarantee.

The CHO configuration now derives growth and product formation from accepted reconciled fluxes, rather than using unreconciled targets for those same processes. Death remains kinetic. Its finite-step population formulas are

$$
N_v^{k+1}=N_v^k e^{(\mu-k_d)h},\quad
\Delta N_d=N_v^k(1-e^{-k_dh}),\quad
\bar N_v=(N_v^k+N_v^{k+1})/2.
$$

Product and extracellular increments use `bar N_v`; integrated viable-cell density uses trapezoidal exposure divided by liquid volume. The viable-cell update is exact for frozen rates; the dead-cell update and coupled exposure are finite-step approximations, not an exact integration of the full changing culture.

**All closure and biological equations above are added.** Native phase inventories and balance accumulation remain existing PharmaPy mathematics. The audit establishes declared-source accounting and network feasibility, not complete elemental or total-reactor mass closure.

### 2.4 Hybrid kinetics and algebraic states

**New rate-provider mathematics:** rule graphs evaluate declared kinetic expressions; affine surrogates compute `q = a + Wz`; hybrids assign each output to one provider. Hybrids do not automatically blend models or train a surrogate. Providers carry identity, units, domain status, and diagnostics.

**Shared native DAE assembly (inherited from the parent):** mechanisms provide `get_solver_state_residuals`; `unit_model` returns differential rates `f(t,y)` and algebraic residuals `g(t,y)` in the compiled state order. `alg_bce=True` exposes the residual dictionary. The backend converts differential rows to `ydot − f(t,y)` and retains algebraic rows as `g(t,y)` for IDA. A cached algebraic mask also supports the parent's optional mass-matrix backend. Bioreactor integration retains differential-only initialization via `create_solver_init_derivatives`, avoiding premature constraint evaluation, and the `dae_residual` convenience interface. Explicit ODE backends reject algebraic states before solving. Both flagship biological examples remain ODE/fixed-step models; DAE capability is a general vessel facility, not a second biological simulation engine.

### 2.5 Parameter estimation and experiment design

**Reused estimation mathematics:** unchanged [ParamEstim.py](../ParamEstim.py) supplies residual construction, observation alignment/masking, and covariance weighting through its inverse-covariance factorization. The new adapter fits

$$
\min_{\theta_L\le\theta\le\theta_U}\tfrac12\|r(\theta)\|^2,
\qquad r=W(\hat y(\theta)-y),\quad W^TW=R^{-1}.
$$

**Added orchestration/numerics:** named parameters and observations, bounds, native reactor callbacks, SciPy bounded least squares, and local diagnostics. With whitened residual Jacobian `J_r`, the reported covariance is

$$
\widehat{\mathrm{Cov}}(\theta)=
\frac{r^Tr}{N-p}(J_r^TJ_r)^{-1}.
$$

It is reported only with full local rank and positive residual degrees of freedom; otherwise covariance is undefined. This residual-scaled local approximation is not a Bayesian posterior or an uncertainty guarantee at active parameter bounds.

**New experiment-design mathematics:** finite-difference output sensitivities `H` give `I_new = H^T R^{-1} H`. Candidate scoring uses `I_total = I_baseline + I_new` where baseline information is supplied. Supported criteria maximize `log det(I)` (D), `−trace(I^-1)` (A), or the minimum eigenvalue (E). Predicted parameter SD is `sqrt(diag(I_total^-1))`. The study recomputes baseline information at fitted parameters using declared measurement noise; it does not invert the residual-scaled fit covariance and treat that as baseline information.

Follow-up studies compare the selected experiment with an equal-cost repeat of the baseline, refit both, and evaluate standardized withheld prediction MSE. Independent experiments use block-diagonal covariance; simultaneous assays can correlate, while distinct sampling times are independent in this workflow.

## 3. Architectural and numerical decisions, with rationale

### Architecture retained from the baseline

- **Native composition:** `LiquidPhase → Mechanism → BatchReactor/SemiBatchReactor → solve_unit → results`; avoids a second simulator with divergent balances.
- **Compiled state keys/slices and reusable buffers:** retain the baseline performance refactor. Mechanisms receive correct phase ownership; Jacobian calculations copy transient data before another evaluation can overwrite it.
- **Existing estimation residual contract and optional solver boundary:** reuse PharmaPy weighting and lazy Assimulo loading; biology does not duplicate them.
- **Input declarations separate from runtime:** species identities, coefficients, recipes, and case-specific assumptions live in JSON. Notebooks present canonical runner results; they do not implement alternative scientific models.
- **Fresh simulation state:** independent fits/design trials build fresh assemblies; replay uses separate work buffers. Within one parameter trial, identical deterministic control evaluations can be reused, without caching across changed parameters.
- **Shared setup:** the two portable ODE backends share initialization. Both notebooks call the library study coordinator directly; figure reproduction and baseline maintenance reuse their forward cells. Redundant example runner scripts are excluded from distribution. Distinct trajectory adapters and floating-point product-rate arithmetic remain because combining or cancelling them could change behavior.

### Numerical optimization and integration choices

| Choice | Implementation and reason |
| --- | --- |
| Pathway LP | SciPy HiGHS; primal/dual feasibility tolerances `1e-9`, tighter than the final `1e-8` uptake-constraint audit. Tightness prevents accepted LP noise from violating the outer audit. |
| LP degeneracy | Preserve the primary optimum, then minimize pathway activities sequentially in declared order, fixing each result. Values below `1e-10` are snapped to zero. Deterministic tie-breaking prevents arbitrary optimizer choices from changing trajectories. |
| Reconciliation initialization | HiGHS feasibility LP; inventory supplies a necessary linear relaxation, or an exact linear constraint when exposure is constant. Reuse the previous flux solution only after checking its suitability; otherwise use the feasible seed. |
| Reconciliation solve | SLSQP with analytic objective gradient and analytic inventory Jacobian, `ftol=1e-10`, `maxiter=1000`. QR removes dependent balance rows during optimization; all original rows are checked afterward. Curvature-based variable scaling and equality-row scaling improve conditioning without changing physical constraints. |
| Target relaxation | Start with ±50% target intervals, widen in configured increments only for reported infeasibility, and intersect with original reaction bounds. CHO uses increment `0.1`, maximum relative bound `2.5`. Numerical failure is a distinct error and does not authorize further biological relaxation. |
| Acceptance of fluxes | Default closure tolerance `1e-8`; culture sets `2e-7`. Check finite objective/fluxes, original balances and bounds scaled by `max(1,max(abs(v)))`, and inventory residuals. Inventory row scaling is `max(abs(n),1)`, not added nutrient or physical slack. |
| Numerical recovery | On SLSQP failure, `_recovery.py` proposes a restricted boundary candidate using LP-selected constraints and, when linked, growth at its lower bound. Rational elimination preserves exact relations for the supplied floating-point coefficients; reduced coordinates, SLSQP, and optional linear KKT polishing generate candidates. Acceptance still requires original constraints and an original-objective gap bound, not just success on the restricted problem. |
| Recovery certificate/settings | Rational Lagrangian bound on the necessary linear relaxation, including reconstruction defect; require `gap >= −defect` and `gap + defect <= tolerance*max(1,abs(objective))`. NNLS proposes nonnegative multipliers (`maxiter=10000`); activity thresholds `1e-7`, null-space row cutoff `1e-12`. Finite declared bounds are required. If unqualified, retry the full-bound SLSQP problem once from the LP seed, then fail explicitly. |
| Flux variability | Optional min/max selected flux at objective ceiling `J* + max(tolerance, abs(J*)*fraction)`; default fraction `0.05`. SLSQP uses `ftol=1e-10`, `maxiter=2000`, retaining inventory constraints when present. Nonlinear intervals are not global certificates. |
| E. coli integration | Adaptive SciPy BDF accommodates stiff/optimization-coupled rates without Assimulo. Case settings: `rtol=1e-8`, `atol=1e-11`, maximum internal step 30 s; reporting grid 0.05 h. Output spacing is not the adaptive integration step. |
| Controlled ODE Jacobian | Forward differences with `delta_j=1e-6*max(abs(y_j),1e-6)` for this case's mass states; scale/step are configurable. Controlled perturbations avoid noisy adaptive probes around LP solutions. Preserve a copied rate baseline. This is distinct from parameter sensitivities used in fitting/design. |
| Depletion and limiter | BDF uses downward zero-inventory events and restarts; active depleted states cannot continue consuming below zero. Disable the vessel's finite-step limiter for this path. Reject significant negative accepted inventory; clamp only the final tiny negative residue (threshold `1e-12`). Fixed-step propagation retains the limiter, but CHO must satisfy joint inventory feasibility before it; the audit rejects unaccounted limiter changes. The inherited limiter computes phase factors and applies a common factor to both ends of each transfer and its heat term. Outlets scale species and volume together, preserving composition. Proposed limited rates are checked before committing them; infeasible depletion raises instead of clipping reaction components. This is not a substitute for timestep refinement. |
| Fixed-step CHO | 0.1-day steps with mechanism preparation, frozen-rate population updates, and native phase updates after acceptance. Split execution at recipe boundaries. Sequential propagation makes inventory exposure and scheduled feeds explicit. |
| Observation/event alignment | Design observations use explicit pre/post-event selection; fixed-step sample times must lie on the mesh (integer-step check `atol=1e-8`). Extend an event-ending run by one step when needed to expose the post-event state. This avoids interpolating across a feed discontinuity or changing the mesh merely to sample. |
| DAE initialization | IDA receives differential mask, balance-derived derivative guesses, `suppress_alg=True`, and `make_consistent("IDA_YA_YDP_INIT")`. Caller options override defaults. Analytic IDA regression tests cover initialization, continuation, and terminal event location; neither flagship biological example uses IDA. |
| Fitting | Bounded SciPy `least_squares`; method/loss remain library defaults (trust-region reflective, linear least-squares loss), with finite differences rather than differentiation through the LP/NLP. The adapter sets `diff_step` when requested; pinned SciPy defaults are two-point differences and `ftol=xtol=gtol=1e-8`, with no explicit evaluation-budget override (`max_nfev=None`). No global or multistart fit is performed. |
| Design search | Enumerate declared candidates rather than optimize arbitrary continuous controls. Bound-aware differences use `delta=relative_step*max(abs(theta),1)` and actual high–low separation. Reject infeasible/over-budget/rank-deficient candidates; order ties by score, cost, identifier. Replay selected score at `rtol=1e-10`, `atol=1e-12`. |
| Design stability/noise | Re-evaluate at half/double sensitivity perturbations and refined integration. Use declared Gaussian noise/covariance and five paired confirmation seeds distinct from calibration seed; paired noise controls comparison variability. E. coli fit/design step is `1e-4`, CHO fit step `1e-6`; refinement factors are `0.1` and `0.5`, respectively. |

### Qualification thresholds are separate from solver tolerances

- **Estimation:** synthetic parameter recovery within 5%; training/withheld standardized RMSE ≤3; tenfold integration-tolerance refinement changes predictions by ≤0.1 measurement SD. These test the demonstration, not experimental accuracy.
- **CHO consistency:** meshes 0.1/0.05/0.025 day; accounting tolerance `1e-10 kg`, scaled network tolerance `2e-7`, mapped-increment tolerance `1e-9`, maximum fine-mesh peak-scaled difference 5%. The worst normalized refinement difference must decrease; not every species must decrease monotonically.
- **Regression:** terminal model-reference error <2%. CHO notebook trajectories/design score use `rtol=1e-4`, `atol=1e-8`; error fractions use absolute `1e-7`; time points match exactly. E. coli trajectory/summary comparisons retain NumPy defaults; its design-score comparison uses `1e-4`/`1e-8`. None is a universal industry standard.
- **Design acceptance:** both inputs explicitly choose `capability`: replay, matched cost, ≥1% predicted SD reduction, stable ranking/benefit, and finite confirmations. `predictive-benefit` additionally requires lower mean withheld MSE and is the default if no profile is supplied. CHO does not meet that additional empirical criterion; the report preserves this fact.
- **Reproducibility:** pinned Python 3.11 dependency file and isolated source-checkout verification avoid inherited dependency overlays. Baseline regeneration is a separate explicit maintenance action; normal verification cannot redefine its reference to obtain PASS.

## 4. Exact capabilities, deliverables, and limitations

| Area | Delivered | Not delivered or not qualified |
| --- | --- | --- |
| Reactor modes | Builder accepts `batch`, `fed-batch`, and `semibatch` (alias). Single liquid phase, isothermal operation; scheduled feed and target-concentration jumps between solves. | No packaged continuous chemostat, perfusion/cell-retention, harvest/bleed, plug-flow, or spatial reactor workflow. Existing PharmaPy classes elsewhere do not establish bioreactor support for those modes. Arbitrary continuous feed/controller behavior is not demonstrated by these event recipes. |
| Mechanistic models | Configurable reduced-pathway dynamic FBA; reconciled network MFA with cell-level/inventory constraints; growth/death, extracellular nutrients, product, and integrated viable-cell density. Declared lumped mass transfer in the pathway mechanism. | No exhaustive organism-specific network, resolved gas phase/mixing/CFD, qualified thermal biology, or product-quality/glycosylation model. CHO pH is a declared kinetic condition, not a dynamically solved acid–base/control system. |
| Hybrid/data-driven models | Rule graphs, configured affine surrogates, hybrid output assignment, external provider registration, validity-domain/extrapolation policy. | No surrogate training pipeline, automatic architecture selection, universal dimensional conversion, neural-network implementation, or learned-model uncertainty calibration. |
| Estimation | Bounded dynamic trajectory fitting through native residual mathematics; named units/times, valid-data masks, asynchronous series support, covariance weighting, local rank/covariance, synthetic withheld-time checks. | Availability times are validated metadata, not an online assimilation scheduler. No Bayesian inference, global identifiability proof, robust-loss configuration, or general time-correlated noise model in the delivered studies. |
| Experiment design | Calibration → information calculation → equal-cost follow-up selection → refit → withheld prediction comparison. D/A/E criteria, existing information, feasibility/cost gates, stability checks, and explicit empirical status. | No global optimal-control or continuous design search, adaptive laboratory execution, guaranteed predictive benefit, or statistical proof from five seeds. |
| Solver scope | SciPy ODE and fixed-step flagship runs. Implemented IDA adapter and DAE residual support. | New backends do not expose solver sensitivities/Jacobian-vector modes. General DAE models, the optional Julia backend, and native event combinations beyond the tested IDA terminal event require model-specific qualification. |
| Scientific validation | Network feasibility, declared-source accounting, numerical refinement, seeded synthetic recovery, and reproducible model-regression evidence. | No independent experimental predictive validation; no complete elemental/total-reactor mass closure. Missing material pools/compositions remain explicit limitations. |

**Packaged studies:** E. coli: 1 L, 10-hour batch simulation; glucose/biomass trajectory estimation; initial-glucose-loading design with early biomass sampling and withheld later predictions. CHO: 0.75 L, 12-day fed-batch simulation; scheduled feeds/GLC targets; standalone local productivity-coefficient estimation; three-mesh consistency audit; pH follow-up design with product measurements at 2, 3.5, and 5.5 days and withheld predictions at 6.5 and 8.5 days. Scientific inputs and sample schedules are configurable, not universal organism parameters.

**Deliverables:** open source-checkout code, two executable workflow notebooks, JSON scientific/operating/method inputs, CSV/PNG trajectories, fit/design/consistency/validation reports, packaged historical evidence and checksums, documented generated regression baselines, tests, pinned dependencies, and one-command installation verification. See [example instructions](../../examples/bioreactors/README.md).

The two workflow notebooks expose configuration, native construction, trajectory
reporting, plotting, and inference directly while calling the reusable PharmaPy
estimation and experiment-study APIs. Default execution is read-only; regression
comparisons apply only to unchanged canonical inputs. Custom cases retain their
physical and study checks, and optional export writes effective inputs to a
separate destination.
