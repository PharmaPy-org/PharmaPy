# PAT data readers

Workflow for process-analytical-technology data, started for the work package 2
demonstrator (#153, #358). The readers of the instruments' files, the IR signals of
the dissolved solute and their calibration are in the package (`PharmaPy.PATReaders`,
`PharmaPy.PATSignals`, `PharmaPy.PATCalibration`); the modules here are not part of
the PharmaPy package API. Nothing in them is specific to a compound; case inputs live in a
subdirectory per case.

## Text tables and the run table

| Module | Purpose |
|---|---|
| `table_layout.py` | Reads a delimited text file through a layout description (JSON): the column titles a file must have, their units, and a plausible range. Columns are found by title, so their order does not matter. Times are returned in seconds and temperatures in kelvin. A file that does not match is refused. |
| `run_table.py` | The common table of one run: named series, each with its own sample times, on a single run clock. Nothing is resampled. |

`table_layout.write_table` writes a table with a title row and a unit row, so reduced
tables are read back through a layout like any other file.

## Signal statistics

| Module | Purpose |
|---|---|
| `fbrm_statistics.py` | Counts per chord-length band and the weighted mean chord, defined as iC FBRM reports them. Against the statistics stored by the vendor software for the four caffeine runs: 10 to 100 um band within 0.6 %, 100 to 1000 um band within 3 %, square-weighted mean within 0.02 %. The band below 10 um was compared on three samples only (within 0.05 %). |
| `calibrate_band.py` | Applies the band-height calibration of `PharmaPy.PATCalibration` to a described dataset: `python -m examples.pat.calibrate_band "<dataset folder>"`. The `band_calibration` block of the description names the table and columns, the runs, the reference run with solvent-only spectra, and the time windows of known composition. Its `solute_ratio` returns the dissolved solute per solvent [kg/kg] of a run over time. |
| `solubility.py` | Solubility against temperature in van't Hoff form, `ln(ratio) = intercept + slope / T`, with its fit and its inverse. |
| `fit_solubility.py` | Fits that curve to a described dataset: `python -m examples.pat.fit_solubility "<dataset folder>"`. Each spectrum taken on a heating ramp while solid is still present gives one point. The `solubility` block of the description gives the time window and the solid signal with its threshold. Run it after `calibrate_band.py`. The points are a lower bound of the solubility, because dissolution lags the heating. |
| `reduce_runs.py` | Turns the raw run folders of a described dataset into its reduced tables: `python -m examples.pat.reduce_runs "<dataset folder>" "<folder that holds the run folders>"`. Each run folder holds one `.iCIR` and one `.icFbrm` file, or their vendor CSV exports (see below), and the thermostat log; the description names the log, its layout, the FBRM bands and the IR signal. That signal is the height of one band (`ir_band`) or the whole-spectrum score (`ir_score`), which needs all runs and is written last. |
| `batch_cooling.py` | Simulates the seeded batch cooling crystallization of a described dataset on the legacy `BatchCryst` or the refactored `BatchCrystallizer`, and returns the slurry state at the requested measurement times: `python -m examples.pat.batch_cooling "<dataset folder>" <run id> <legacy or refactored>`. The description supplies the charge, the seed, the solid, the property database and starting values of the kinetics. Needs the Assimulo solver stack. |
| `dataset.py` | Reads a dataset description and its reduced tables. A dataset folder holds `runs.json`, one `reduced_<kind>.layout.json` per kind of table, and `reduced/<run>_<kind>.csv`. The description names the solute and the solvent and gives their charged masses. |

The two IR signals of the dissolved solute are in the package, `PharmaPy.PATSignals`:

- `band_height`, the height of a spectral band above a straight line through two
  anchor points. Band and anchors are inputs; they belong to the compound and solvent.
- `SpectralScore`, the dissolved-solute score of whole spectra, for a solute without a
  band clear of the solvent. It is fitted on a reference run: spectra are moved to a
  reference temperature with a linear temperature effect per spectral point, and their
  difference from the dissolved charge is split into the score along the
  dissolved-minus-slurry direction and three terms for a change of probe contact:
  scale, offset and tilt. The fitted score is then applied to any run.

`spectrum_to_signal.ipynb` shows, figure by figure, how a spectrum becomes the band
height (caffeine) and the whole-spectrum score (azithromycin).
`oaba/ir_signals.ipynb` goes from the files of one OABA run to both signals with the
package alone and checks them against each other.
Their calibration to dissolved solute per solvent [kg/kg] is in
`PharmaPy.PATCalibration`: a band height is linear in the solute with a
temperature-dependent slope, and a score is counted from the charge of its run with
one sensitivity. The constants are solved from states of known composition.
`signal_to_concentration.ipynb` does both from the reduced tables, the band height of
caffeine and the score of azithromycin, and tests how far each can be trusted. It
needs no raw files.

The model side of these signals is in the package: `PharmaPy.SlurryMeasurements`
predicts the dissolved solute per solvent, the solute in the solid and a
particle-count signal from a crystallizer result.

A new dataset needs a layout description per text-file kind and a dataset
description; `caffeine/` and `azithromycin/` are the examples. Someone, or an assistant, writes the
layout once by looking at the file; from then on reading is deterministic.

## Native-file readers

The readers of the instruments' native files are in the package,
`PharmaPy.PATReaders`: `read_ic_spectra` for iC IR and iC Raman files (`.iCIR`,
`.iCRaman`) and `read_ic_fbrm` for iC FBRM files (`.icFbrm`). The formats are not
documented by the vendor; they were inferred by inspection of four
caffeine-in-water runs, of which one also has vendor CSV exports.

| Reads | Check against the vendor export of run 2026-10-23 |
|---|---|
| IR spectra and their metadata | 663 of 663 spectra, largest difference 5e-15 absorbance |
| Raman spectra and their metadata | 712 of 712 spectra, within the export's rounding (5e-10 relative) |
| raw chord-length counts, 1550 channels | 4313 samples rebinned to the 100 export channels: median relative difference 1e-4, largest 3.3e-3, totals within 1e-5 |

Limits:

- Checked on files written by iC IR 7.1.91.0, iC Raman 8.0.114.0 and iC FBRM
  4.4.33.0. A file of another build is read with a warning; compare one vendor
  export of it before relying on the values.
- The raw FBRM channel grid (0.2 um to 100 um, 1 um to 500 um, 2 um to 1800 um) is
  inferred from that one run. It is not a vendor specification.
- A file the acquisition software never closed has no name table.
  `PATReaders.read_package` then recovers the streams by scanning the blocks.
- FBRM times are local time of the acquisition computer. IR and Raman sample times
  are UTC.
- A spectrum stored without a metadata entry has no sample time. It is left out and
  its identifier is listed in `without_time` of the result.

`azithromycin/read_native_files.ipynb` shows the two calls on one run.

## Vendor CSV exports

`PATReaders.read_ic_ir_export` and `PATReaders.read_ic_fbrm_export` read the CSV
exports of iC IR and iC FBRM into the same objects as the native readers, so the
reduction takes either. A run is reduced from exports when its entry in
`runs.json` names them, as files in the run folder:

| Key of a run | Meaning |
|---|---|
| `fbrm_export` | iC FBRM export of unweighted counts, read instead of the `.icFbrm` file. |
| `ir_export` | iC IR export, read instead of the `.iCIR` file. |
| `ir_export_start_local` | Local time at which the iC IR experiment started (ISO 8601). The export titles each spectrum with the time since that start and holds no date. |

Native files against exports on the caffeine run of 2026-10-23, the only run with
exports (663 spectra, 4313 FBRM samples):

| Quantity | Difference between the two routes |
|---|---|
| IR absorbance, band height | none (5e-15 absorbance) |
| Times on the run clock | within 1 s; the export keeps whole seconds |
| Square-weighted mean chord | within 0.03 % |
| Counts 1 to 10 um and 10 to 100 um, with crystals present | within 0.5 % |
| Counts 100 to 1000 um, with crystals present | within 2 % |
| Band counts in the clear solution (below 1000 counts in total) | within 3 counts |

Limits:

- The band edges at 10 and 100 um fall inside channels of the 100-channel export,
  whose counts are then split in proportion to the channel length. The raw grid of
  a native file has edges there. This is the difference in the table.
- The iC IR export does not name its ordinate; it is read as absorbance.
- An export of weighted counts is refused, and so is a date and time not written
  like `10/24/2026 1:53:23 AM`.
- An iC Raman export is not read. The one in the data has titles to the minute,
  some of them twice, so its spectra cannot be placed in time.

The raw files are not in the repository. Small excerpts used by the tests are in
`tests/fixtures/pat`.
