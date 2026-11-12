# PAT data readers

Readers for process-analytical-technology data, started for the work package 2
demonstrator (#153, #358). They are not part of the PharmaPy package API. Nothing
in them is specific to a compound; case inputs live in a subdirectory per case.

## Native-file readers

Mettler Toledo iC IR, iC Raman and iC FBRM store each experiment in a native file.
The formats are not documented by the vendor; they were inferred by inspection of
four caffeine-in-water runs, of which one also has vendor CSV exports.

| Module | Reads | Check against the vendor export of run 2026-10-23 |
|---|---|---|
| `block_package.py` | container of `.iCIR` and `.iCRaman` | through the spectra below |
| `ic_spectra.py` | spectra and their metadata | IR: 663 of 663 spectra, largest difference 5e-15 absorbance. Raman: 712 of 712, within the export's rounding (5e-10 relative). |
| `ic_fbrm.py` | raw chord-length counts, 1550 channels | 4313 samples rebinned to the 100 export channels: median relative difference 1e-4, largest 3.3e-3, totals within 1e-5. |

Limits:

- Verified on iC IR 7.1 and iC FBRM 4.4 files only. Check a file from another
  software version or instrument against a vendor export before relying on it.
- The raw FBRM channel grid (0.2 um to 100 um, 1 um to 500 um, 2 um to 1800 um) is
  inferred from that one run. It is not a vendor specification.
- A file the acquisition software never closed has no name table.
  `block_package.read_package` then recovers the streams by scanning the blocks.
- FBRM times are local time of the acquisition computer. IR and Raman `SampleTime`
  values are UTC.

The raw files are not in the repository. Small excerpts used by the tests are in
`tests/fixtures/pat`.
