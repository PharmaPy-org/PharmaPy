# Crystallizer concentration ownership (#312)

Crystallizer residuals and result retrieval now pass independent concentration
buffers to mutable phase converters. Integrated concentration columns remain
unchanged, including the final solvent entry; intentional FVM distribution
rescaling remains unchanged. Real phase tests cover Batch, MSMPR and Semibatch,
and native CVode tests cover unequal-density seeded growth.

The existing physical representations are explicitly distinct: the ODE profile
contains integrated species concentrations; a named-solvent LiquidPhase closes
the ideal mixture volume by completing the solvent concentration. Property
phases and outlets retain that convention. MSMPR now explicitly constructs its
outlet from the completed phase composition, rather than obtaining it through
an accidental mutation of the published profile. The unnamed-solvent control
retains every concentration value.

This is an ownership repair, not a resolution of the physical closure question
in [#312](https://github.com/PharmaPy-org/PharmaPy/issues/312). Conservation across
the ODE/phase volume representation and finite-volume crystal mass generation
requires separate numerical investigation. Do not infer inventory equivalence
from the buffer tests. The explicit phase-closure expectation in the regression
is provisional with respect to that open modeling question.
