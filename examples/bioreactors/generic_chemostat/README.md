# Generic chemostat

Synthetic, isothermal, sterile-feed Monod culture using PharmaPy's native
`ContinuousReactor`, liquid streams and volume controller. These inputs are a
mathematical qualification case, not experimental data or fitted biology.

The 1 L culture receives 0.1 L/h containing 10 mmol/L nutrient. The native
well-mixed outlet maintains the target volume. Declared uptake is
`q = 4 S/(1 + S)` mol/(kgDW h), with S in mmol/L; yield is 0.1 kgDW/mol.
Thus maximum growth is 0.4/h and dilution is 0.1/h. No maintenance, death,
cell retention, gas transfer or oxygen limitation is represented.

At the positive equilibrium, `S* = Ks D/(mu_max-D) = 1/3 mmol/L` and
`X* = Y(Sin-S*) = 0.9667 gDW/L`. The washout threshold is
`Dcrit = mu(Sin) = 0.3636/h`. With growth disabled, biomass follows
`X(t) = X(0) exp(-D t)`. These are independent test oracles.

Run the six notebook steps. The default `inputs/` selects pathway optimization.
To run the same Monod model as configured kinetics, set `input_dir` to
`example_dir / "inputs/kinetics"` and update `thermo_path` accordingly in step 1.
That representation includes biomass as a phase species; its formula weight is
bookkeeping. Both representations use the same yield and operating conditions.
Exports are local generated files and are excluded from version control.

`operation.mode: "continuous"` is the only new selection. Existing flow events
set, change or stop inlet streams. `working-volume` assumes additive solution
volumes with explicit feed densities. The native controller targets the initial
volume; after a sample or bolus it relaxes back toward that target when the available inlet flow permits. This is a
well-mixed chemostat, not perfusion or selective cell retention. Exposure integrals
remain accumulated history rather than being withdrawn with cells.
