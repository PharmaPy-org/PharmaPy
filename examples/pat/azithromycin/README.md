# Azithromycin dihydrate in ethyl acetate: dataset description and reduced tables

Second dataset described for the PAT readers in `examples/pat`. The raw files are
not in the repository; the reduced tables are.

| File | Content |
|---|---|
| `runs.json` | The four runs: recipe, seeding, temperature program, clock, reduction settings and notes. Each `basis` entry states where the numbers come from. |
| `huber_log.layout.json` | Layout of the thermostat log `expttsd1.csv`. |
| `reduced/<run>_temperature.csv` | Jacket, external and setpoint temperature [K], 30 s block means. |
| `reduced/<run>_fbrm.csv` | Counts in the bands 1 to 10, 10 to 100 and 100 to 1000 um [counts] and the square-weighted mean chord [um], every 10 s sample. `high_seed_slow_cool` has none: its iC FBRM file holds no sample. |
| `reduced/<run>_ir.csv` | Whole-spectrum IR score [-] with the probe-contact terms and the unexplained remainder, every spectrum taken while the temperature was logged. |
| `reduced/ir_score.json` | Sizes of the score fit, and each run's dissolved spectrum against that of the first run. |
| `reduced_*.layout.json` | Layouts of the three reduced tables, for `table_layout.read_table`. |
| `read_native_files.ipynb` | Reads the native IR, Raman and FBRM files of one run with `PharmaPy.PATReaders` and plots what comes back. Needs the raw files. |

Time in every reduced table is the run clock [s]: zero when the temperature program
starts. Each run folder holds one `.iCIR` and one `.icFbrm` file and the thermostat
log `expttsd1.csv`. To rebuild the tables, run from the repository root:

    python -m examples.pat.reduce_runs examples/pat/azithromycin "<folder that holds the four run folders>"

## The IR score

Ethyl acetate absorbs wherever the solute does, so the dissolved solute is read from
the whole spectrum between 900 and 1800 cm-1 (`examples/pat/spectral_score.py`), not
from one band. In every run the score is zero for the run's own charge, fully
dissolved in the hold at 37 C before seeding. A score of minus one is a drop of
dissolved solute equal to the one between the dissolved charge and the slurry at
20 C of `low_seed_slow_cool`. The score is not yet a concentration: that needs the
charge of each run, and the batch is reused and adjusted from run to run.

## What the reduced tables show

- The seed survives in three runs: the FBRM counts step up at seeding and keep
  rising through the cooling.
- In `low_seed_slow_cool` the seed dissolves. The counts fall back to within 20
  counts of their clear-solution level within five minutes and stay there down to
  20 C. Crystals appear in the hold at 20 C.
- `low_seed_fast_cool` starts as a clear solution at 20 C that forms crystals during
  the first hold.
- The FBRM reads counts in clear solution, before seeding: about 10 in
  `high_seed_fast_cool`, 100 in `low_seed_fast_cool` and 140 in `low_seed_slow_cool`.
- The dissolved spectrum of `low_seed_slow_cool` cannot be compared with that of the
  first run: the remainder of the comparison is 0.22 absorbance, against 0.001 for
  the other two runs. Its spectra sit on a different baseline.

## Assumptions to carry into any use of these runs

- The charge and seed masses are recipe values. Weighed masses were not reported,
  and the seed size distribution was not measured.
- The lab reuses the batch from one run to the next and adjusts it in between. The
  description gives the recipe of a fresh batch, not the charge of each run.
- "External T" is taken as the temperature of the liquid in the vessel.
- The clock of each run comes from the modification time of its thermostat log and
  is good to about one minute.
