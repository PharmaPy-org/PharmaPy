# Caffeine in water: dataset description

First dataset described for the PAT readers in `examples/pat`. The raw files are not
in the repository.

| File | Content |
|---|---|
| `runs.json` | The four runs: charge, seeding, temperature program, clock offsets and notes. Each `basis` entry states where the numbers come from. |
| `huber_log.layout.json` | Layout of the thermostat log `expttsd1.csv`, read with `table_layout.read_table`. |

Each run folder holds one `.iCIR`, one `.iCRaman` and one `.icFbrm` file, the
thermostat log `expttsd1.csv` and the setpoint program `cmp1.1.csv`.

Assumptions to carry into any use of these runs:

- The charge and seed masses are recipe targets. Weighed masses were not reported.
- "External T" is taken as the temperature of the liquid in the vessel. The lab has
  not yet confirmed where that sensor sits.
- The seed slurry is made with supersaturated liquid that cools outside the vessel,
  so the seed returned to the vessel may exceed the weighed mass.
- The clock offsets come from file modification times and are good to about one
  minute. With them, the FBRM seeding spike falls 2.7 to 5.0 minutes after the
  8460 s node of the setpoint program in all four runs, which is consistent.
