# PAT reader fixtures

Excerpts of the caffeine-in-water run of 2026-10-23 (Nagy group, Purdue), used by
`tests/test_pat_native_readers.py`.

| File | Content |
|---|---|
| `ir_head_blocks.iCIR` | First four 64 KiB blocks of the iC IR file, followed by its name table. The name-table offset at byte 16 was rewritten to point at the moved table. Holds the first six spectra. |
| `ir_export_columns.csv` | Vendor CSV export of the same run: wavenumber and the first three spectra. |
| `spectrum_info_tags.xml` | Opening tags of five metadata entries from the iC IR and iC Raman experiment documents, reduced to the attributes the readers use. |
| `fbrm_three_samples.icFbrm` | Zip with a `ReactionSamples.vector` that holds samples 401, 2001 and 4001 of the iC FBRM file. The two record offsets at bytes 160 and 164 were rewritten for three records. |
| `huber_log_excerpt.csv` | First seven lines of the thermostat log `expttsd1.csv` of the same run, unchanged. Used by `tests/test_pat_run_tables.py`. |
| `fbrm_export_rows.csv` | Vendor CSV export of the same run: the four header rows and the same three samples. |

Identifying values were overwritten and carry no test meaning: the acquisition
computer name, instrument and probe serial numbers and the source folder path in the
iC IR excerpt (replaced by `x` of equal length, so every offset is kept), and the
instrument configuration between bytes 168 and 5478 of the FBRM stream (zeroed).
The decoded spectra are bit-identical before and after.
