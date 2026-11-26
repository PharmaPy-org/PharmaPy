====================
PAT signals
====================

One number per spectrum that follows the dissolved solute, from the spectra
read by ``PharmaPy.PATReaders``. ``band_height`` is for a solute with a band the
solvent leaves free; peak and anchors belong to the compound and solvent (the
values below are those used for caffeine in water)::

    from PharmaPy import PATReaders, PATSignals

    ir = PATReaders.read_ic_spectra("run.iCIR")
    height = PATSignals.band_height(ir.wavenumber, ir.absorbance, peak=1700.0, baseline=(1760.0, 1720.0))

``SpectralScore`` uses the whole spectrum, for a solute without such a band. It
is fitted on a reference run, given as three selections of its spectra (boolean
masks, one entry per spectrum), and then applied to any run of the same solute,
solvent and probe::

    fitted = PATSignals.SpectralScore.fit(
        ir.wavenumber, ir.absorbance, temperature,           # [cm-1], [-], [K] of each spectrum
        clear_solution=clear, dissolved=dissolved, slurry=slurry,
        reference_temperature=310.15, window=(900.0, 1800.0), maximum_absorbance=1.3)
    terms = fitted.score(ir.wavenumber, ir.absorbance, temperature, dissolved=dissolved)
    terms.score                                              # 0: dissolved charge, -1: reference slurry

The score needs the liquid temperature of every spectrum, which the spectrum
file does not hold. The temperature effect is taken as linear at every spectral
point and is fitted on the clear-solution spectra: ``temperature_range`` gives
the temperatures they cover, and ``score_per_kelvin`` sizes the part of a score
that rests on extrapolating beyond them. Neither signal is a concentration; a
calibration against known compositions gives one.

``examples/pat/spectrum_to_signal.ipynb`` explains both signals figure by
figure, and ``examples/pat/oaba/ir_signals.ipynb`` goes from the files of one
run to both signals.

.. automodule:: PharmaPy.PATSignals
    :members:
