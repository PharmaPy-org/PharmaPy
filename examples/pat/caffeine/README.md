# Caffeine in water: dataset description and reduced tables

First dataset described for the PAT readers in `examples/pat`. The raw files are not
in the repository; the reduced tables are.

| File | Content |
|---|---|
| `runs.json` | The four runs: charge, seeding, temperature program, clock offsets, reduction settings and notes. Each `basis` entry states where the numbers come from. |
| `huber_log.layout.json` | Layout of the thermostat log `expttsd1.csv`. |
| `reduced/<run>_temperature.csv` | Jacket, external and setpoint temperature [K], 30 s block means. |
| `reduced/<run>_fbrm.csv` | Counts in the bands 1 to 10, 10 to 100 and 100 to 1000 um [counts] and the square-weighted mean chord [um], every 10 s sample. The mean chord is empty for a sample without counts. |
| `reduced/<run>_ir.csv` | Height of the 1700 cm-1 band above the line through 1760 and 1720 cm-1 [-], and the absorbance at 1640 cm-1 [-], every spectrum. |
| `reduced_*.layout.json` | Layouts of the three reduced tables, for `table_layout.read_table`. |
| `band_calibration.json` | Plateau band heights and IR calibration constants of the three usable runs, under two treatments of the difference between runs. Written by the general workflow: `python -m examples.pat.calibrate_band examples/pat/caffeine`. |
| `solubility.json` | Solubility curve of caffeine in water between 19.6 and 41 C from the heating ramps of the three usable runs, per run and pooled, under both calibration treatments. Written by `python -m examples.pat.fit_solubility examples/pat/caffeine`. |
| `compounds.json` | Property database of caffeine and water for the crystallizer models. Each entry states the source of every value. |
| `plot_runs.py` | Draws one quality figure per run from the reduced tables: `python -m examples.pat.caffeine.plot_runs "<output folder>"`. The figures are not kept in the repository. |

Time in every reduced table is the run clock [s]: zero when the temperature program
starts. Each run folder holds one `.iCIR`, one `.iCRaman` and one `.icFbrm` file,
the thermostat log `expttsd1.csv` and the setpoint program `cmp1.1.csv`. To rebuild
the tables, run from the repository root:

    python -m examples.pat.reduce_runs examples/pat/caffeine "<folder that holds the four run folders>"

To simulate a run from its seeding time on either batch crystallizer, with the
starting kinetics of `runs.json` (they are not fitted values):

    python -m examples.pat.batch_cooling examples/pat/caffeine high_seed_slow_cool legacy

## What the reduced tables show

- Seeding is visible in every run as a short FBRM spike 2.7 to 5.0 minutes after the
  8460 s node of the setpoint program, during the hold at 37 C.
- The IR band height follows the dissolved caffeine in three runs: about 0.09 for
  the slurry at 20 C and about 0.13 once everything is dissolved. Water alone reads
  0.065 to 0.068, drifting upward over the ten minutes before the caffeine is
  added. Only `high_seed_slow_cool` has such spectra.
- `high_seed_slow_cool` records the caffeine addition. For at least five minutes
  the band reads 0.147 to 0.152, above the fully dissolved value, with coarse FBRM
  counts. Then fine counts rise eightfold and the band falls to 0.097. This is the
  as-received caffeine turning into the hydrate.
- The slurry at 20 C reads 0.0019 to 0.0023 higher at the end of every run than at
  its start, and the dissolved plateaus differ by 0.004 between runs. Both limit
  the IR calibration; `band_calibration.json` carries two treatments.
- The heating ramps give a solubility curve: 1.85 g caffeine per 100 g water at
  20 C and 4.51 at 40 C when the runs share one calibration slope. The three runs
  agree within 4.5 % under that treatment and within 7.8 % under the other.
- Only two runs give an IR concentration trace through the crystallization:
  `high_seed_slow_cool` and `low_seed_fast_cool`. `low_seed_slow_cool` has about 20
  spectra after the 37 C hold. In `high_seed_fast_cool` the probe was not in proper
  contact with the liquid: the absorbance at the water band stays below 0.8, against
  0.86 to 0.99 in the other runs, until 23,900 s.

## Assumptions to carry into any use of these runs

- The charge and seed masses are recipe targets. Weighed masses were not reported.
- The lab reuses the batch from one run to the next and adjusts it in between. The
  IR calibration and the simulation take the recipe charge for every run; part of
  the difference between the dissolved plateaus of the runs may be a real
  difference in charge.
- "External T" is taken as the temperature of the liquid in the vessel. The lab has
  not yet confirmed where that sensor sits.
- The seed slurry is made with supersaturated liquid that cools outside the vessel,
  so the seed returned to the vessel may exceed the weighed mass.
- The clock offsets come from file modification times and are good to about one
  minute. The timing of the seeding spike is consistent with them.
- The assignment of the 1700 cm-1 band to caffeine rests on its growth when the
  caffeine dissolves. It was not checked against a reference spectrum.

## December calibration and experiment design

Run the recorded case through the shared WP1 estimation and design interfaces
from a checkout with the caffeine dataset and the experiment-design API.
Use the locked Assimulo environment and a clean source revision. The first
command writes the fit record outside the checkout. The second checks the
recorded Git revision and inputs and reproduces the accepted parameters.
December refers to the checkpoint. The acquisition dates are in `runs.json`.

```bash
pixi run -e assimulo python -m examples.pat.caffeine_m6 \
    --revision "$(git rev-parse HEAD)" --record /tmp/caffeine-fit.json
pixi run -e assimulo python -m examples.pat.caffeine_m6 \
    --revision "$(git rev-parse HEAD)" --replay /tmp/caffeine-fit.json
```

The fit uses the measured 1700 cm⁻¹ IR band heights and temperature logs for
the high-seed slow-cooling and low-seed fast-cooling runs. It takes spectra
after the seeding time in `runs.json` through the first logged temperature
at or below the final 20 °C cooling target. This can include the transition
into the hold because the window follows the recorded temperature rather
than a setpoint clock. The approximate alignment of the instruments limits
the interpretation of the earliest samples. The nine post-seeding spectra
of the low-seed slow-cooling run are held out of growth fitting. Its earlier
spectra still contribute to band calibration and pooled solubility, so this
is a limited check on the growth fit. The high-seed fast-cooling IR trace
is excluded because of its documented probe issue.

The legacy moment model predicts liquid caffeine mass per solvent mass
[kg/kg]. The band equation in the
[case-selection note](../../../doc/review_notes/crystallization_case_selection.md#7-measurement-equations)
maps that ratio to absorbance. The fit estimates the growth prefactor
[µm/s] and supersaturation order [dimensionless], holding the secondary
nucleation rates and activation energies at the case starting values.
Both predicted mass inventories are checked at every requested time.
`--treatment common_baseline` repeats the study with the other documented
band and solubility calibration.

The two pre-seeding dissolved-plateau scatters define relative weights for
each run. The fit estimates a common residual variance scale and records
per-run RMSE relative to plateau scatter, bias and adjacent-observation
residual correlation. These diagnostics matter here. The fitted residuals
are much larger than the plateau scatter and strongly correlated across
adjacent observations. The local linearized covariance assumes independent
Gaussian errors and fixes the band calibration, solubility curve, seed
geometry, seeding times, other kinetics and prescribed temperature histories.
Use it as a conditional diagnostic. The growth prefactor remains weakly
determined.

The workflow compares the optimizer's derivatives with a second-order
finite-difference stencil at the fitted point. It reruns the two observation
grids at that stencil and the fitted point with both CVode tolerances divided
by 100. The observed change in the band is used as an empirical error scale
for the local sensitivity-rank calculation. The comparison and numerical
environment are stored in the record.

The sampling example compares two retained IR spectra at the first and last
recorded times of each third of the fitted spectrum sequence, on a hypothetical
repeat of the high-seed slow-cooling recipe. The log determinant gains describe
a model calculation under independent errors. They illustrate how the reported
information changes with the subset of spectra, with the same recipe and
number of spectra throughout. Selecting a physical experiment requires
independent repeats or a residual model that accounts for the observed
serial structure.

This IR-only workflow contributes to #351, #352, #355, #359 and #360.
The full crystallization case still needs secondary-nucleation and FBRM
analysis, seed-size and independent solubility sensitivity, and a domain
review. Adoption by the bioreactor demonstrator is tracked separately.
