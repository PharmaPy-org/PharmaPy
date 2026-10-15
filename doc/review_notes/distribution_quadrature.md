# Population conversion and inventory consistency (#269)

`SolidPhase.convert_distribution` uses trapezoidal nodal support for both
directions of conversion, matching `getMoments`. For sizes `x_i` in µm,
the endpoint support is half the adjacent interval; an interior support is
`(x_{i+1} - x_{i-1}) / 2`. These supports are distinct from the finite-volume
grid widths stored as `dx`, which remain unchanged.

A node contributes `n_i q_i kv (x_i × 10⁻⁶)³` m³ to the population volume.
Dividing each contribution by their sum gives dimensionless volume fractions.
The shape factor therefore cancels. Conversely, allocating a supplied solid
volume among those same contributions gives a number distribution whose third
moment reproduces the charged mass. Uniform, zero-ended input retains its
previous number density; occupied endpoints and nonuniform grids change to
conserve the supplied inventory.

This deliberately corrects both previously documented parts of
[issue #269](https://github.com/PharmaPy-org/PharmaPy/issues/269): shape-factor
dependence in reported fractions and disagreement between conversion and moment
quadrature. Merely dividing the old output by `kv` leaves the nonuniform-grid
normalization wrong and does not repair positive-mass construction.

Batch and MSMPR `result.vol_distrib` use the corrected normalized reporting.
Their underlying raw-number population ODEs are unchanged. Empty populations
report zero fractions rather than NaN, without claiming a defined normalized
distribution. Signed solver undershoots remain visible; the conversion does not
clip numerical trajectories. Positive input volume at zero size is rejected,
while an unoccupied zero-size node converts without division by zero.

Independently written regressions cover asymmetric occupied endpoints, unequal
intervals, mass and volume input bases, `kv` values 0.2, 0.5 and 1, history arrays,
round trips, repeated phase update, real filter attachment, and native batch/MSMPR
publication. Provisional expectations in the grid-refresh tests now use the
correct volume shares. Mixer rejection still uses a deliberately inconsistent
mass/population fixture; a corrected constructor is no longer its source of
inconsistency. No private exercise content or input data is included.
