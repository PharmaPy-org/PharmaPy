# Analytical conversion fixture

Declared synthetic inputs for construction and adaptive-solver verification.
The initial liquid is 1 L and 1 kg, with 1 g of the formal `biomass` pool.
Equal-mass conversion to `substrate` proceeds at 4/s for 2 s, so remaining
biomass is independently known as `0.001 exp(-4 t)` kg, with `t` in seconds.
These names do not assert a biological reaction or an organism-specific rate.

The constant-property thermodynamic inputs are unchanged from the existing
synthetic Doyle fixture. Formal solute molecular weights of 1000 g/mol make
mmol/L numerically equal to kg/m3. The water carrier uses 18.015 g/mol.
This independent copy keeps foundational tests runnable before the example PR.
No simulation outputs or experimental measurements are included.
