# Caffeine in water: dataset description and reduced tables

First dataset described for the PAT readers in `examples/pat`. The raw files are not
in the repository; the reduced tables are.

| File | Content |
|---|---|
| `runs.json` | The four runs: charge, seeding, temperature program, clock offsets, reduction settings and notes. Each `basis` entry states where the numbers come from. |
| `huber_log.layout.json` | Layout of the thermostat log `expttsd1.csv`. |
| `reduce_runs.py` | Turns the raw run folders into the reduced tables. |
| `reduced/<run>_temperature.csv` | Jacket, external and setpoint temperature [K], 30 s block means. |
| `reduced/<run>_fbrm.csv` | Counts in the bands 1 to 10, 10 to 100 and 100 to 1000 um [counts] and the square-weighted mean chord [um], every 10 s sample. The mean chord is empty for a sample without counts. |
| `reduced/<run>_ir.csv` | Height of the 1700 cm-1 band above the line through 1760 and 1720 cm-1 [-], and the absorbance at 1640 cm-1 [-], every spectrum. |
| `reduced_*.layout.json` | Layouts of the three reduced tables, for `table_layout.read_table`. |

Time in every reduced table is the run clock [s]: zero when the temperature program
starts. Each run folder holds one `.iCIR`, one `.iCRaman` and one `.icFbrm` file,
the thermostat log `expttsd1.csv` and the setpoint program `cmp1.1.csv`. To rebuild
the tables, run from the repository root:

    python -m examples.pat.caffeine.reduce_runs "<folder that holds the four run folders>"

## What the reduced tables show

- Seeding is visible in every run as a short FBRM spike 2.7 to 5.0 minutes after the
  8460 s node of the setpoint program, during the hold at 37 C.
- The IR band height follows the dissolved caffeine in three runs: about 0.09 for
  the slurry at 20 C and about 0.13 once everything is dissolved. Water alone reads
  0.065.
- Only two runs give an IR concentration trace through the crystallization:
  `high_seed_slow_cool` and `low_seed_fast_cool`. `low_seed_slow_cool` has about 20
  spectra after the 37 C hold. In `high_seed_fast_cool` the probe was not in proper
  contact with the liquid: the absorbance at the water band stays below 0.8, against
  0.86 to 0.99 in the other runs, until 23,900 s.

## Assumptions to carry into any use of these runs

- The charge and seed masses are recipe targets. Weighed masses were not reported.
- "External T" is taken as the temperature of the liquid in the vessel. The lab has
  not yet confirmed where that sensor sits.
- The seed slurry is made with supersaturated liquid that cools outside the vessel,
  so the seed returned to the vessel may exceed the weighed mass.
- The clock offsets come from file modification times and are good to about one
  minute. The timing of the seeding spike is consistent with them.
- The assignment of the 1700 cm-1 band to caffeine rests on its growth when the
  caffeine dissolves. It was not checked against a reference spectrum.
