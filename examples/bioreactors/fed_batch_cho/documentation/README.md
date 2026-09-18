# CHO source documentation and evidence

This package records the public sources, reconstruction assumptions, and
historical evidence available for the native CHO fed-batch example. It is not
a raw-data validation package.

## Scientific sources

Jayanth Venkatarama Reddy, Nikola Malinov, Jason Souvaliotis, Eleftherios Terry
Papoutsakis, and Marianthi Ierapetritou. *A dynamic metabolic flux analysis
(DMFA) model for performance predictions of diverse CHO cell culture process
modes and conditions*. bioRxiv, version dated January 12, 2026.
[Model paper](https://doi.org/10.64898/2026.01.11.698917).

Related experimental source: *Pseudo perfusion of Chinese Hamster Ovary (CHO)
cells as a reliable platform for data generation to model and guide continuous
perfusion biomanufacturing*. bioRxiv, version dated December 1, 2025.
[Related paper](https://doi.org/10.1101/2025.11.27.691016).
This second source is background to the historical reconstruction campaign;
it is not a separate validated perfusion example delivered in this folder.

## Included evidence

| File | Role | Limitation |
| --- | --- | --- |
| [reddy_cho_public_reconstruction.yaml](evidence/reddy_cho_public_reconstruction.yaml) | Historical source identities, units, ambiguities, and missing-data inventory | Describes a broader earlier campaign, not the current native implementation contract |
| [reddy_public_metrics.yaml](evidence/reddy_public_metrics.yaml) | Frozen model-generated qualification metrics | Not raw observations or a fresh test report for this checkout |
| [source_manifest.json](../inputs/source_manifest.json) | Local evidence paths and SHA-256 checksums | Integrity is not scientific validation |
| [Current outputs](../outputs/) | Native example results | Agreement with saved model outputs is regression evidence |

The YAML files are unchanged copies from the reference implementation's
`data/bioreactors/models/` and `data/bioreactors/validation/` respectively.
Their historical test counts, digests, mode coverage, and statements about
figure-digitization must not be presented as newly verified native-branch
results. No digitized or raw observation table is included in this package.

## Current inputs and benchmark interpretation

- [case.json](../inputs/case.json): reactor operation, event schedule, and solver settings.
- [mechanism.json](../inputs/mechanism.json): kinetic rules, network, and reconciliation configuration.
- [thermo.json](../inputs/thermo.json): native phase-property inputs.
- [workflow.ipynb](../workflow.ipynb): native construction, propagation, and reporting.

The following untraced internal constants were retired from the runner:

| Quantity | Retired constant | Former regression tolerance |
| --- | --- | --- |
| Viable cell density | 6.380513082505868 million cells/mL | Relative error < 2% |
| Total product amount | 1.65453652125779 g | Relative error < 2% |

These constants are not identified as measured endpoints in the supplied
sources. Their exact generation history is not established. They remain in
[regression_baseline.json](../inputs/regression_baseline.json) as historical
information, not active acceptance targets or experimental truth. In particular,
the historical YAML reports product **concentrations in g/L** for other runs;
those are not interchangeable with the runner's total product **amount in g**.
The 2% threshold is project-specific, not a universal industry standard.

The active baseline is explicitly generated from the corrected native model.
It records input hashes, source-tree digest, Python/package versions, generation
time and command. Run `PYTHONPATH=. python examples/bioreactors/refresh_regression.py` only
when intentionally establishing new software regression baselines; normal
simulation and acceptance runs only read them. Independently rerun both examples
after generation. A PASS means replay agreement, not agreement with Reddy data.

## Coupling review and current acceptance status

The native review found that the same modeled growth/product processes had two
different rates: the accepted network flux and the unreconciled reactor rate.
In the pre-correction 12-day run, the maximum relative target adjustments were
5.42% for growth and 100% for product. Forcing both kinetic targets as hard
equalities made the declared network/bounds infeasible. Those targets therefore
cannot simply be treated as simultaneously achievable process rates.

The correction uses the accepted product exchange flux and the growth exchange
explicitly named by `kinetic_outputs.growth_reaction`. Its unit conversion is
declared in `constants.growth_flux_to_per_day`. Unlinked kinetic growth remains
available to other configurations. No reaction identifier is encoded in the
runtime implementation. Kinetics still supplies the reconciliation targets;
death remains kinetic because no linked death reaction is declared.

This is a consistency correction to the supplied reconstruction, not a claim
that the inaccessible original author implementation used the same coupling.
The source flux basis and pseudo-biomass/product coefficients remain declared
reconstruction assumptions, not independently verified elemental chemistry.

The packaged consistency report was generated using the input `consistency.json` declares 0.1, 0.05 and 0.025-day
meshes and acceptance thresholds before evaluation. The generated
`outputs/consistency.json` independently checks S v, accepted flux bounds,
growth/product increments, extracellular exchange plus degradation increments,
scheduled material additions, cumulative inventories, and mesh agreement.
It also reports the existing finite-step death update's deviation from the
exact frozen-rate death integral; it does not call that update exact.

**Inventory feasibility is now enforced inside the metabolic closure.** The
general `InventoryAvailability` object constrains
`n_end = n_start + other_increment + exposure(v) * M @ v >= 0`.
The mechanism assembles M from all input mappings (summing shared-species
contributions), and includes declared degradation. For this fixed-step route,
exposure is the same trapezoidal viable-cell exposure used by propagation,
including its dependence on the optimized growth flux. Its analytical Jacobian
is checked against independent central differences. No uptake rate is clipped
after the solve. Scheduled feeds are already in the native inventory at the
start of the corresponding step.

An initial linear feasibility screen uses the minimum possible exposure to
obtain necessary uptake bounds; it never substitutes for the nonlinear
acceptance check. Reconciliation intervals are intersected with the network's
declared bounds, rather than allowing their expansion to reverse irreversible
reactions. Infeasibility remains an error. The optional constraints also apply
to FVA when requested. Nonlinear constrained solutions and flux intervals are
local numerical results, not global-optimality certificates. Constrained
optimization variables are scaled by the quadratic penalty curvature and
equality rows normalized. This equivalent coordinate transformation avoids
the severe conditioning sensitivity observed with unscaled optimization;
no physical bound, penalty weight or acceptance threshold is changed.

The shared vessel, limiter and integrators are unchanged. On the original
0.1/0.05/0.025-day meshes, limiter adjustments are below 1.4e-20 kg per step;
step accounting residuals are below 6e-16 kg and scaled network residuals below
1.1e-10. Feed accounting and mapped growth/product increments also pass.
These replace the prior persistent nutrient-accounting failures.

**The complete declared consistency gate now PASSES:** maximum coarse/middle
and middle/fine trajectory differences decrease from 4.02% to 1.63% of the
fine-mesh inventory peak, below the unchanged 5% criterion. Earlier unscaled
solver trials failed convergence (39.3% on these meshes, worsening on finer
meshes); those failures motivated the equivalent numerical normalization,
not a relaxation of criteria or modification of biological parameters.
This establishes numerical and declared-source accounting qualification,
not independent experimental accuracy or full elemental closure.

Scope: the current native fixed-step culture mechanism supplies its own
metabolic and degradation increments. Discrete feeds are applied before its
solve. Other independently consuming mechanisms or continuous withdrawals
would require their increments to be coordinated with this constraint; this
extension does not claim that arbitrary multi-mechanism flowsheets or adaptive
integration are automatically inventory-coupled.

`balance_audit.json` remains a legacy finiteness/nonnegativity diagnostic;
its `passed` flag is not conservation acceptance. The newer consistency report
is authoritative for the quantitative accounting gate. Even a passing declared
source-accounting test would not prove closed total reactor mass or elemental
closure: biological pools, gas exchanges and complete elemental compositions
are not all represented in the liquid inventory. The missing information must
not be filled with invented compositions or undocumented conversion factors.

The inventory-constrained product endpoint at 0.1 day is approximately 0.115 g
(VCD 8.841 million/mL), instead of the original 1.656 g product endpoint.
The corrected model does not reproduce the old constants; their failures are
preserved in the investigation record. The provenance task replaces them with
traceable current-model regression baselines, not revised experimental targets.
The previously generated
experiment-design report and notebook outputs predate this correction and
must not be cited as qualification of the corrected model. Rerun them after
the inventory-consistency gate is resolved. This task does not close provenance,
final WP3 acceptance.

The public reconstruction record also flags objective indexing, zero
normalizers, ambiguous feed labeling, rate-to-inventory units, and biomass/product
exchange sign conventions. These are recorded assumptions requiring traceability,
not proof that every ambiguity has been independently resolved.

## Missing information and permitted conclusions

The historical inventory identifies missing raw calibration and held-out
measurements, complete proprietary medium compositions, exact operating-event
logs, parameter covariance and fitting details, and reference intracellular
flux trajectories. Therefore this example supports a public-information
reconstruction and software demonstration, not exact reproduction of unpublished
data, independent experimental validation, or a universal CHO parameterization.

[workflow.ipynb](../workflow.ipynb) is a synthetic calibration-to-design study:
it compares information and withheld predictions after selected versus matched
baseline follow-up cultures. See the example README and design input for timing,
noise, and acceptance criteria. It must not be represented as a completed wet-lab
campaign or independently validated improvement in process performance.

## Paper access and redistribution

The historical source record lists CC-BY-NC-ND-4.0 for the two preprints. Complete papers remain linked. The published comparison panel from the supplied
presentation is now included in [published_figures/](published_figures/README.md),
with extraction provenance; it is not relicensed with PharmaPy. The
original publishers' license notices govern their reuse. The scientific-source
documentation is locally readable, but obtaining the complete papers still
requires following the DOI links or using separately obtained copies.
