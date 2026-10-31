# Doyle — Exercise 4.15

Open `workbook.ipynb` with the PharmaPy Python environment and run all cells.
The notebook shows the six explicit forward-simulation steps: choose folders,
load/edit inputs, construct and solve, extract trajectories, plot, and export.
It does not call a runner or apply benchmark scenario overrides. The nominal
input is an equilibrium; Step 2 shows how to request a dilution-step transient.

For the separate automated numerical comparisons, from this folder run:

```sh
PYTHONPATH=../../.. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_benchmarks.py
```

Only this example's scenarios are declared in `benchmark.json`. Inputs and the
unexecuted notebook are committable; generated `outputs/` remain local and ignored.
The runner is self-contained, requiring no Alterable files.

## Input field guide: direct-kinetics template

This example supplies a textbook numerical reference, not measured cultivation
data. Use its structure for direct kinetics; do not treat its coefficients or
lumped species properties as universal biological defaults.

| Section | Category | What to configure |
| --- | --- | --- |
| `schema_version`, `mechanism` | Required for this model | Keep version 3 and `configured-kinetics` for this formulation. |
| `$comment` | Descriptive | Explain the case; it does not affect the equations. |
| `operation` | Required for this model | Continuous mode, working volume, carrier, duration and selected recipe. The 1 L volume and density are benchmark conventions. |
| `state.species` | Required for this model | Initial amounts and their unit (`g` here). Product is intentionally zero and inactive, not a missing input. |
| `parameters` | Required for this model | Monod maximum growth rate, affinity and biomass yield; keep unused groups empty. |
| `model.kinetics` | Required for this model | Explicit rate graph, species mass-rate mapping, constants and rate units. |
| `recipes` | Required for this model | Sterile feed flow and composition; continuous operation supplies withdrawal through the native reactor. |
| Other model sections, `network`, `rate_provider`, `transport`, blank state fields | Optional | Leave the inactive slots empty/null; they are not needed for this direct-kinetics formulation. |
| `numerical` | Numerical | Solver, requested output grid, tolerances and maximum step. |
| `thermo.json` | Required for this model / optional properties | Preserve formal molecular weights and density conventions described below. Unused thermal properties remain blank. |
| `benchmark.json` | Numerical verification / scenario inputs | The verification runner uses declared scenario changes and acceptance limits. This is not a third runtime input required by `build_bioreactor`. |

For a different process, change equations and parameters together with their
units. Preserve the distinction between initial amounts and concentrations.
The separate verification runner applies declared benchmark scenarios; editing only
the nominal mechanism does not replace those overrides.

### Source and methodology

Source: Seborg, Edgar, Mellichamp and Doyle, *Process Dynamics and Control*,
fourth edition (Wiley, 2017). Exercise 4.15: printed p. 67 (PDF p. 79).
No copyrighted page images are redistributed here.

## Purpose and execution

These are numerical reproductions of a textbook model, not experimental data,
parameter estimation, or complete elemental-balance validation. They run through
unchanged `build_bioreactor -> NativeBioreactorAssembly.solve ->
ContinuousReactor.solve_unit -> ScipyBackend` using `configured-kinetics`.
They do not qualify the pathway LP or rate-reconciliation optimizer.

From an environment containing the existing PharmaPy dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/absolute/path/to/Bioreactor WP3.1" \
python run_benchmarks.py
```

The mechanism JSON file specifies every biological equation and coefficient.
`inputs/mechanism.json` includes physical `operation` and solver `numerical` settings.
`benchmark.json` specifies operating scenarios and acceptance limits. The runner
changes only the declared initial amounts and inlet flow for each scenario and
saves the complete resolved case/mechanism inputs beside the generated outputs.
Reference equations are independently written concentration balances; they do
not call the production rate evaluator, vessel balances, or rule interpreter.

## Inputs, units, and assumptions

- This example uses a perfectly mixed, isothermal continuous culture with
  sterile substrate feed, fixed working volume, and no cell retention.
- Working volume 1 L and density 1 kg/L are explicit benchmark conventions,
  not textbook measurements. Volume cancels from the concentration equations
  because feed flow is set to dilution rate times volume.
- Biomass, substrate and product are unspecified lumped masses. Formal molecular
  weights of 1000 g/mol make each numerical concentration in mmol/L equal to
  its value in g/L (1 g per formal mmol). This is a bookkeeping conversion,
  not a chemical identity. Water has MW 18.015 g/mol. No thermal properties or
  organism-specific constants are inferred.
- Kinetic rates returned by the graph are g/(L h). Native integration uses
  kg, m3 and seconds. No empirical correction, clipping of the textbook growth
  equation, or fitted coefficient is introduced.
- The substrate-plus-biomass invariant is the one implied by the printed
  model. Matching it does not establish complete elemental conservation.

## Exercise 4.15

The Monod model uses mu_max=0.20/h, Ks=1 g/L, Y=0.5 gDW/g substrate,
feed substrate=10 g/L. Initial X=4.5 gDW/L and S=1 g/L are the stated
equilibrium for D=0.1/h. Product is an inactive zero-valued bookkeeping species.
The +/-10% dilution steps and 100 h horizon are declared extensions for
transient verification; the exercise itself asks for a linearized model.

Balances are X'=(mu-D)X, S'=D(Sin-S)-mu X/Y, with mu=mu_max S/(Ks+S).
The exact positive equilibrium is S*=Ks D/(mu_max-D), X*=Y(Sin-S*), where
positive biomass is feasible. Its transfer function can independently be
derived; full nonlinear histories are compared with separate integrations.

## Predeclared acceptance

`benchmark.json` was fixed before the first simulation. Maximum full-history
absolute concentration discrepancy must be <=1e-5 g/L (well below the book's
two-decimal initial-state precision). Independent DOP853 and Radau reference
solutions must agree within 1e-7 g/L. Tightening native relative/absolute
tolerances tenfold and halving maximum step must change predictions by
<=1e-5 g/L. The analytical equivalent-substrate balance must hold to
1e-6 g/L, constant volume to 1e-9 L, and nonnegativity to 1e-9 g/L.

The reference identity is:

`S + X/Y = Sin + (S0 + X0/Y - Sin) exp(-D t)`.

Reference integrations use rtol=1e-11, atol=1e-12 g/L, max step=0.1 h.
Native baseline uses rtol=1e-9, atol=1e-12 in native units, max step=0.25 h.
All histories are compared every 0.25 h; these are reporting points, not an
assumption that the adaptive integrator uses fixed steps.

## Output and scope

`outputs/report.json` records source revision, runtime environment, input and runner
checksums, measured errors and all acceptance decisions. Each scenario has
resolved inputs, `trajectories.csv`, `comparison.png` and `metrics.json`.
Generated outputs are ignored by Git. Keep the source inputs, runner and this
documentation if later committing this standalone benchmark package.

This benchmark is a focused numerical qualification, not a clean-installation
sign-off or independent experimental validation.
