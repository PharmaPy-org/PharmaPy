====================
PAT file readers
====================

Readers for the native files of process-analytical-technology instruments.
``read_ic_spectra`` reads Mettler Toledo iC IR and iC Raman files (``.iCIR``,
``.iCRaman``) and ``read_ic_fbrm`` reads iC FBRM files (``.icFbrm``), without
the vendor software::

    from PharmaPy import PATReaders

    ir = PATReaders.read_ic_spectra("run.iCIR")
    ir.time, ir.wavenumber, ir.absorbance      # UTC, [cm-1], [-] (n_spectra, n_points)

    fbrm = PATReaders.read_ic_fbrm("run.icFbrm")
    fbrm.time, fbrm.channel_edges, fbrm.counts # local time, [um], [counts] (n_samples, 1550)

The vendor does not document these formats. The layouts were inferred from
files and checked against the vendor's CSV exports of one run, written by
iC IR 7.1.91.0, iC Raman 8.0.114.0 and iC FBRM 4.4.33.0. A file of another
build is read with a warning: compare one export of it before relying on the
values. The raw chord-length grid of iC FBRM (0.2 um channels to 100 um, 1 um
to 500 um, 2 um to 1800 um) is inferred from that run; it is not a vendor
specification. IR and Raman sample times are UTC; FBRM sample times are the
local time of the acquisition computer.

``read_ic_ir_export`` and ``read_ic_fbrm_export`` read the vendor's CSV exports
into the same objects::

    start = datetime.datetime(2026, 10, 24, 4, 28, 19)       # UTC start of the iC IR experiment
    ir = PATReaders.read_ic_ir_export("ir.csv", start)
    fbrm = PATReaders.read_ic_fbrm_export("fbrm.csv")        # (n_samples, 100) in the export checked

The iC IR export titles each spectrum with the time since the experiment
started and holds no date, so the caller gives the start. An export keeps time
stamps to whole seconds, and the iC FBRM export keeps the channels chosen in the
vendor software instead of the 1550 raw ones. An iC Raman export is not read.

.. automodule:: PharmaPy.PATReaders
    :members:
