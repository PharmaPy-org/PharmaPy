# Crystal-grid inventory convergence (#391)

The finite-volume population's third moment is not exactly the continuum
growth integral used as the liquid crystallization sink. Coarse size grids can
therefore finish integration while their crystal mass disagrees with the solute
removed. This regression checks that independent component inventories converge
under refinement; it does not replace model outputs with archived reference data.

The committed synthetic Gaussian seed stays far from the size-domain boundaries
during the prescribed 50 s isothermal batch. A non-unit shape factor and 2 m³
liquid charge expose population and volume basis mistakes. The tests use actual
CVode, a public property database and no private case material. Noncrystallizing
species must remain conserved. A separate moment-mode solve closes the same
component inventory without finite-grid growth transport.

On master f20b3a2, target-mass residuals relative to newly formed crystal mass
decrease from approximately 7.93%, 3.96%, 0.700% to 0.0700% on 35/70/140/280
nodes. The fine-grid acceptance budget is a proposed 0.1% numerical target for
this diagnostic; it is not a universal resolution recommendation. Integrator
tolerances are materially tighter. The test does not assert that coarse grids
must remain inaccurate, and improvements preserving monotone convergence and
the final budget are allowed.

This does not validate a truncated continuous population, named-solvent phase
completion, or MSMPR volume closure. Those concerns remain in
[#300](https://github.com/PharmaPy-org/PharmaPy/issues/300) and
[#312](https://github.com/PharmaPy-org/PharmaPy/issues/312).
