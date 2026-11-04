# WP3 delivery: requirements, evidence and limits

This table assesses the local integrated WP3.1/WP3.2 implementation against the
saved issue criteria for [#361](https://github.com/PharmaPy-org/PharmaPy/issues/361),
[#364](https://github.com/PharmaPy-org/PharmaPy/issues/364) and
[#365](https://github.com/PharmaPy-org/PharmaPy/issues/365). The criteria were
captured on November 1–2, 2026; this is not a live GitHub status report.
“Covered” identifies implementation and executable verification, not experimental
validation, maintainer approval or permission to close an issue.

## Selected case and responsibility

| Item | Recorded selection |
|---|---|
| Baseline case | E. Coli Yeast-Extract Fed-Batch Process; source runs F10/F11 |
| Organism and mode | *Escherichia coli*, fed-batch; biomass, glucose and acetate observations |
| Data and assumptions | [Source audit, recipes and limitations](../../../examples/bioreactors/ecoli_ye_fed_batch/README.md); [source identity, transformations and sharing conditions](../../../examples/bioreactors/ecoli_ye_fed_batch/inputs/provenance.json) |
| Implementation/reconciliation owner | QHuckaba |
| Designated domain reviewer | andres9403, as recorded in the workspace-owned PR descriptions; this does not assert review acceptance |
| Optional network case | [Reduced E. coli batch dFBA demonstration](../../../examples/bioreactors/batch_ecoli_dfba/documentation/README.md), attributed to Mahadevan et al. (2002) |
| Hybrid/estimation demonstration | [Doyle synthetic training and held-out workflow](../../../examples/bioreactors/Doyle_Exercise_4_15/README.md) |

Earlier tracker material names a different prospective reviewer. The local record
above makes the intended assignment explicit; synchronizing public coordination
records and obtaining actual review remain separate actions.

## Requirement-by-requirement evidence

Test paths below are relative to `tests/Bioreactor/`, except where stated.
Each test exercises real PharmaPy objects or an explicitly independent numerical
reference. No row treats a successful solver termination as sufficient proof of
physical validity.

| Requirement | Implementation and verification | Assessment |
|---|---|---|
| **361.1:** Record organism/system, mode, shareable data, reviewer and owner | Selection above; provenance retains source revisions, hashes, transformations and sharing restrictions. [test_bioreactor_provenance.py](../../../tests/Bioreactor/test_bioreactor_provenance.py), [test_bioreactor_source_records.py](../../../tests/Bioreactor/test_bioreactor_source_records.py). | **Recorded locally.** Public coordination and reviewer acceptance are not inferred. |
| **361.2:** Biomass, substrate, product and relevant oxygen states; kinetics, yields, feeds, units and defensible assumptions | Maintained reconciliation inputs and frozen direct reference; explicit unit conversion, feed-density/composition audit, optional direct-model oxygen-demand outputs. [test_bioreactor_ecoli_reconciliation.py](../../../tests/Bioreactor/test_bioreactor_ecoli_reconciliation.py), [test_bioreactor_oxygen.py](../../../tests/Bioreactor/test_bioreactor_oxygen.py), [test_construction_contracts.py](../../../tests/Bioreactor/test_construction_contracts.py). | **Conditional selected-case acceptance.** Carbon/biomass/byproduct equations and nominal recipes are implemented. Actual delivered glucose and oxygen-transfer/control histories remain unresolved. Oxygen demand is not a dissolved-oxygen balance or feedback model. |
| **361.3:** Conservation and nominal/boundary behavior through public solve and measurement outputs | Native material/energy balances, declared-invariant and independent-ledger audits, infeasibility propagation, explicit inventory error budgets. [test_bioreactor_known_answer.py](../../../tests/Bioreactor/test_bioreactor_known_answer.py), [test_bioreactor_shared_physics.py](../../../tests/Bioreactor/test_bioreactor_shared_physics.py), [test_bioreactor_volume.py](../../../tests/Bioreactor/test_bioreactor_volume.py), [test_forward_audit.py](../../../tests/Bioreactor/test_forward_audit.py), [test_inventory_acceptance.py](../../../tests/Bioreactor/test_inventory_acceptance.py), [test_bioreactor_observations.py](../../../tests/Bioreactor/test_bioreactor_observations.py); native [tests/test_material_rate_limiting.py](../../../tests/test_material_rate_limiting.py). | **Covered for represented pools and declared balances.** Missing elemental or auxiliary-pool information is reported as not assessable. |
| **361.4:** Documented kinetics substitution boundary | [rate_providers.py](../../../PharmaPy/Bioreactors/rate_providers.py), [mechanisms.py](../../../PharmaPy/Bioreactors/mechanisms.py) and the [user guide](USER_GUIDE.md). [test_bioreactor_rate_providers.py](../../../tests/Bioreactor/test_bioreactor_rate_providers.py), [test_bioreactor_mechanism_interchange.py](../../../tests/Bioreactor/test_bioreactor_mechanism_interchange.py). | **Covered.** Replacement rates share the native reactor contract. |
| **361.5:** Repeatable simulation and measured-field predictions suitable for calibration | [construction.py](../../../PharmaPy/Bioreactors/construction.py), [observations.py](../../../PharmaPy/Bioreactors/observations.py); exact-time/event-side sampling, named fields, explicit units and independent replay. [test_bioreactor_observations.py](../../../tests/Bioreactor/test_bioreactor_observations.py), [test_bioreactor_workflow_notebooks.py](../../../tests/Bioreactor/test_bioreactor_workflow_notebooks.py), [test_bioreactor_estimation_handoff.py](../../../tests/Bioreactor/test_bioreactor_estimation_handoff.py). | **Covered as a forward/handoff capability.** No empirical calibration claim. |
| **364.1:** Cited network/case, uptake bounds, objective, coupling, units and validity assumptions | Reduced-network case above; [Metabolic/pathways.py](../../../PharmaPy/Metabolic/pathways.py), network declarations and source documentation. [test_bioreactor_dae.py](../../../tests/Bioreactor/test_bioreactor_dae.py), [test_bioreactor_known_answer.py](../../../tests/Bioreactor/test_bioreactor_known_answer.py), [test_pathway_objective_scaling.py](../../../tests/Bioreactor/test_pathway_objective_scaling.py). | **Covered for the declared reduced network.** Not a validated genome-scale reconstruction or a full-horizon dynamic optimum. |
| **364.2:** Flux-solve/ODE handoff, infeasibility and reproducible backend configuration | `PathwayMetabolism`, `ConfiguredPathwayModel`, shared inventory availability and native integrators. [test_bioreactor_provider_routes.py](../../../tests/Bioreactor/test_bioreactor_provider_routes.py), [test_bioreactor_shared_physics.py](../../../tests/Bioreactor/test_bioreactor_shared_physics.py), [test_bioreactor_adaptive_inventory.py](../../../tests/Bioreactor/test_bioreactor_adaptive_inventory.py). | **Covered.** No success-manufacturing retry or clipped stoichiometry. |
| **364.3:** Conservation and independently checkable flux/dynamic case before empirical use | Analytical constrained fluxes, independent piecewise balances, objective-coordinate invariance and optional optimal-face source ranges. [test_bioreactor_known_answer.py](../../../tests/Bioreactor/test_bioreactor_known_answer.py), [test_bioreactor_objective_units.py](../../../tests/Bioreactor/test_bioreactor_objective_units.py), [test_forward_audit.py](../../../tests/Bioreactor/test_forward_audit.py). | **Covered numerically.** An optimizer preference does not establish unique biological flux recovery. |
| **364.4:** Interchangeable mechanistic and dFBA kinetics | Same builder, phases, reactor balances and observation interface. [test_bioreactor_mechanism_interchange.py](../../../tests/Bioreactor/test_bioreactor_mechanism_interchange.py), [test_bioreactor_pathway_corrections.py](../../../tests/Bioreactor/test_bioreactor_pathway_corrections.py). | **Covered within the documented model contracts.** |
| **364.5:** Executable documentation, optional-backend tests and usable baseline without dFBA dependencies | Maintained batch workbook; SciPy/HiGHS route; marked Assimulo tests; isolated missing-Assimulo import tests. [test_bioreactor_examples.py](../../../tests/Bioreactor/test_bioreactor_examples.py), [test_bioreactor_dae.py](../../../tests/Bioreactor/test_bioreactor_dae.py), [test_bioreactor_qualification.py](../../../tests/Bioreactor/test_bioreactor_qualification.py); [tests/test_optional_assimulo_imports.py](../../../tests/test_optional_assimulo_imports.py). | **Covered.** No separate COBRA installation is required by this reduced-pathway implementation. |
| **365.1:** Rate inputs/outputs, units, ordering, domain and derivative availability | Named-provider contracts, validated named kinetic unit declarations, explicit parameter bindings and native numerical differentiation. [test_bioreactor_named_kinetics.py](../../../tests/Bioreactor/test_bioreactor_named_kinetics.py), [test_kinetic_unit_contracts.py](../../../tests/Bioreactor/test_kinetic_unit_contracts.py), [test_bioreactor_estimation_handoff.py](../../../tests/Bioreactor/test_bioreactor_estimation_handoff.py). | **Covered.** Arbitrary user expression graphs are not automatically dimensionally proved; analytic sensitivities are not claimed. |
| **365.2:** Mechanistic and representative hybrid/surrogate on the same reactor contract | Named mechanistic forms, affine/hybrid providers and optional bounded positive multiplicative corrections. [test_bioreactor_example_selection.py](../../../tests/Bioreactor/test_bioreactor_example_selection.py), [test_bioreactor_example_corrections.py](../../../tests/Bioreactor/test_bioreactor_example_corrections.py), [test_bioreactor_hybrid_correction.py](../../../tests/Bioreactor/test_bioreactor_hybrid_correction.py). | **Covered.** Neutral corrections preserve the mechanistic trajectory; no claim of accuracy for arbitrary learned models. |
| **365.3:** Positivity/conservation and actionable unsupported-domain behavior | Provider validity policy, unchanged native balances, coupled inventory feasibility and explicit numerical acceptance. [test_bioreactor_rate_providers.py](../../../tests/Bioreactor/test_bioreactor_rate_providers.py), [test_bioreactor_pathway_corrections.py](../../../tests/Bioreactor/test_bioreactor_pathway_corrections.py), [test_inventory_acceptance.py](../../../tests/Bioreactor/test_inventory_acceptance.py). | **Covered for declared constraints.** Positive multipliers preserve rate signs/zeros; they do not independently prove elemental closure. |
| **365.4:** Reproducible simulation/estimation, provenance and training/validation limits | Doyle training-only native fit, untouched held-out condition, native `CalibrationRecord` export/load, tamper checks and relocated replay/refitting. [test_bioreactor_calibration_workflow.py](../../../tests/Bioreactor/test_bioreactor_calibration_workflow.py), [test_bioreactor_estimation_stress.py](../../../tests/Bioreactor/test_bioreactor_estimation_stress.py). | **Covered synthetically.** Empirical predictive validity, population-level uncertainty coverage and broader WP3.3 remain separate. |

## Additional delivered capability

Batch, fed-batch and continuous/chemostat operation use the native reactor classes;
[test_bioreactor_continuous.py](../../../tests/Bioreactor/test_bioreactor_continuous.py) checks dilution, washout and known equilibria.
Non-isothermal operation reuses native energy accounting, utilities, mixing and
temperature controllers. [test_bioreactor_thermal.py](../../../tests/Bioreactor/test_bioreactor_thermal.py) checks analytic heating and
cooling, temperature-dependent kinetics and heat capacity, feed/event enthalpy,
state ordering and agreement with direct native construction. These tests do not
validate a particular organism's temperature response or thermal parameters.

The numerical qualification separates physical assumptions from solver settings.
Independent conservation budgets, explicit tolerance sweeps, different estimation
seeds and derivative refinement are required evidence; failed coarse trials remain
reported. Source-range diagnostics describe mathematical nonuniqueness on an LP
optimal face, not a probability distribution for biological fluxes.

## Conditions for declaring delivery complete

The local engineering evidence supports the implemented WP3.1/WP3.2 interfaces and
represented numerical cases. Full scientific acceptance still requires a domain
reviewer's decision on the baseline case's feed and oxygen applicability, adequate
independent experimental data and agreed prediction-error criteria. A local table
cannot supply missing operating records or another person's approval.

Integration with the saved prerequisite history does not prove that today's live
PRs are synchronized, reviewed or merged. No GitHub state is changed by this local
assessment. Shared-record adoption by other selected processes (#351), broad
estimation/experiment design (WP3.3), perfusion and universal organism coverage are
not silently included in these completion claims.
