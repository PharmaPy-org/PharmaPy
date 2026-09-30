# Synthetic mammalian-cell fed-batch and surrogate substitution

This is an illustrative biologics-development capability demonstration, **not
experimental validation or a calibrated CHO model**. All numerical biological
coefficients and operating choices are explicit synthetic assumptions in inputs.

Open `workflow.ipynb` from this folder or the repository root and run its cells.
It builds the native SemiBatchReactor, solves the mechanistic case, samples its
rate equations over a declared synthetic concentration grid, fits affine and
quadratic rate surrogates, and reruns by changing only `rate_provider`.
The polynomial provider is registered in Python once, outside the runtime package;
its features, coefficients, units and domain are configuration, not species-specific
logic. No extra machine-learning dependency is used.

## Inputs and model

- `case.json`: three-day, isothermal fed-batch operation and solver settings.
- `mechanism.json`: glucose-dependent uptake, yield-based growth, first-order
  death and viable-biomass-dependent product secretion; feed begins at day one.
- `thermo.json`: molecular/formal weights and liquid density; unused properties blank.
- `surrogate.json`: synthetic training domain, cell dry mass for reporting and a
  predeclared 5% maximum peak-normalized trajectory error threshold.

Viable/dead pools are dry biomass inventories; cell counts are derived using the
explicit 0.3 ng dry mass/cell assumption. Glucose uses its molecular weight;
biomass/product use formal mass bookkeeping. Growth and product are reduced
phenomenological rates, not an intracellular or elemental model. Missing gases,
cell composition and byproducts prevent a claim of complete elemental closure.
There are no unexplained correction multipliers or experimentally sourced values
masquerading as measurements. Temperature is prescribed, not a thermal model.

## Outputs and qualification

The notebook exports mechanistic/affine/polynomial CSV trajectories, PNG/SVG plots
including viable cells and product titer, a numerical comparison with validity
status, and fitted mechanism JSONs reusable without refitting. Training targets
are instantaneous mechanistic rates, not desired output trajectories. Errors are
evaluated on independently integrated trajectories. Domain errors are not clipped
or silently extrapolated; tests exercise forbidden extrapolation explicitly.

This demonstrates modularity within WP3.1/WP3.2, not WP3.3 estimation from noisy
experiments. A real culture requires identified coefficients, valid training data,
appropriate conserved pools and independent validation for its intended use.
