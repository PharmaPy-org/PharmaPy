====================
Slurry measurements
====================

Measurement predictions from a crystallizer result: dissolved solute per
solvent, solute in the solid, a particle-count signal and mean sizes. A
result of the legacy ``BatchCryst`` or of the refactored
``BatchCrystallizer`` (method of moments) is first converted to a
``SlurryState`` with the adapter of its layout; the predictions are functions
of that state.

.. automodule:: PharmaPy.SlurryMeasurements
    :members:
