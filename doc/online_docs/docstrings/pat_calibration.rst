====================
PAT calibration
====================

Calibration of a signal of ``PharmaPy.PATSignals`` to dissolved solute per
solvent [kg/kg], with a few constants solved from moments of a run where the
composition is known.

A band height is linear in the solute, with a slope that depends on
temperature. The baseline is the height of the solvent alone; the slope and its
temperature coefficient come from the whole charge dissolved at two
temperatures::

    from PharmaPy import PATCalibration

    baseline, _, _ = PATCalibration.window_statistics(time, height, solvent_start, solvent_stop)
    slope, slope_per_kelvin = PATCalibration.fit_band_slope(
        [height_hot, height_warm], [temperature_hot, temperature_warm],   # [-], [K]
        charge_ratio, baseline, reference_temperature)                     # [kg/kg], [-], [K]
    ratio = PATCalibration.ratio_from_band_height(
        height, temperature, baseline, slope, slope_per_kelvin, reference_temperature)

A whole-spectrum score is already at one temperature and is counted from the
dissolved charge of its run, so ``ratio = charge + sensitivity * score``. The
sensitivity comes from two states of known composition on one zero, and the
charge of a run from its own dissolved spectrum or from one known state::

    sensitivity = PATCalibration.fit_score_sensitivity(
        ratios=[charge_ratio, saturated_ratio], scores=[0.0, slurry_score])  # [kg/kg], [-]
    ratio = PATCalibration.ratio_from_score(score, charge_ratio, sensitivity)
    other_charge = PATCalibration.charge_from_score(saturated_ratio, other_slurry_score, sensitivity)

Which states are known, and which run carries them, is the user's reading of
the runs. ``examples/pat/signal_to_concentration.ipynb`` does both calibrations
on the caffeine and azithromycin datasets and tests how far each can be trusted.

.. automodule:: PharmaPy.PATCalibration
    :members:
