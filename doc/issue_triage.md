# Issue triage

This note records open issues reviewed on the `fix/issues` branch and
deliberately skipped or excluded, with the reason, so each decision is traceable.
Each row is a snapshot of the review date; maintainers may reopen the decision.

Each section records the branch commit and date at which its rows were reviewed;
"open pull request" bases are valid as of that date and must be rechecked when
the PR merges or closes.

## Excluded: already addressed by an open pull request

Reviewed on 2026-09-07 against branch `fix/issues` at commit `c11abed`.

| Issue | Open PR | Basis |
| --- | --- | --- |
| #29 | PR #189 | PR closes #29. |
| #47 | PR #190 | PR closes #47. |
| #55 | PR #180 | PR closes #55. |
| #59 | PR #199 | PR closes #59. |
| #177 | PR #179 | PR closes #177. |
| #200 | PR #201 | PR closes #200. |
| #202 | PR #203 through PR #211 | The test-migration series (PR #203 through PR #211) carries the umbrella's remaining work; each PR intentionally leaves #202 open until the series lands. |
| #130 | PR #131 | PR #131 implements every repository-side acceptance criterion (Refs, not Closes); the remaining Read the Docs administration items are external and labeled blocked:external. |
| #17 | PR #189 refs it | Tracking epic for #18–#66; addressed through its sub-issues. |
| #7 | PR #201 refs it | Tracking epic for test coverage; addressed through its sub-issues. |

Reviewed on 2026-09-08 against branch `fix/issues` at commit `c019f07`.

| Issue | Open PR | Basis |
| --- | --- | --- |
| #75 | PR #207 | The issue's own comment states the production fix and regression coverage are in PR #207 (DynamicCollector result routing and plotting). |

Reviewed on 2026-09-08 against branch `fix/issues` at commit `c5a2c1d`.

| Issue | Open PR | Basis |
| --- | --- | --- |
| #214 | PR #210 | The issue states the one-cell FVM contract is implemented in PR #210 (consolidated `high_resolution_fvm` with zero-gradient outlet, empty-grid and unknown-limiter errors). |

## Skipped: planning, scoping, epic, blocked-external, and process issues

Reviewed on 2026-09-07 against branch `fix/issues` at commit `c11abed`.

| Issue | Title (short) | Reason |
| --- | --- | --- |
| #3 | Epic: restore CI, raise SE baseline, extend modeling | Cross-milestone tracking epic; its remaining work lives in the sub-issues and #154. No code change is defined by the epic itself. |
| #8 | Modernize packaging and project metadata | Packaging work is done on master (pyproject.toml, MANIFEST.in, untracked egg-info, `test` and `assimulo` extras); the only unchecked criterion, the `.[docs]` extra, lands with open PR #131, and declarative versioning is resolved by decision under #146 (see the 2026-08-15 status audit comment). |
| #10 | Decouple the ODE/DAE solver behind an interface | Architecture refactor with an explicit "prototype and validate first" plan, not a defect. Lazy Assimulo imports already landed (PR #135) and PR #203 adds a public `PharmaPy.solvers` adapter; a full backend abstraction exceeds a coherent incremental fix. |
| #12 | Shared UnitOperation base class and constants module | Behavior-preserving architecture refactor flagged as the riskiest change in the set; not a defect and not a bounded fix. |
| #13 | Type hints, docstring standardization, API docs | Open-ended incremental documentation effort; AGENTS.md already mandates NumPy docstrings for touched code, which every fix on this branch follows. No bounded fix is defined. |
| #14 | Scoping: bioreactor module | Scoping issue labeled blocked:external; no fix is defined. |
| #15 | Scoping: crystallizer model extensions | Scoping issue for new modeling capability; no defect. |
| #16 | Scoping: scheduling layer | Scoping issue for a new module; no defect. |
| #67 | Epic: June 2026 audit defects (#68–#89) | Tracking epic; the open sub-issues are triaged individually. |
| #118 | Epic: StateLayout single source of truth | Architecture epic; not a defect. |
| #133 | Units-and-basis correctness strategy | Strategy/infrastructure proposal spanning #12 and #118; not a bounded fix. |
| #139 | Scoping: evaluate POUNCE for parameter estimation | Evaluation/scoping, not a fix. |
| #142, #143, #144, #145 | StateLayout infrastructure and migrations | Planned refactor sub-tasks of #118; behavior-preserving migrations, not defects. |
| #146 | Publish 3.0.0 to PyPI | Release-gate task labeled blocked:external; requires maintainer credentials and a release decision. |
| #148 | Epic: integrate the MultiPhaseVessel architecture | Workstream epic; not a defect. |
| #149 | Balance-conservation regression tests for the MPV refactor | Multi-week test-writing gate for the #148 workstream, not a defect; deferred to that workstream. |
| #150 | Integration contract for parallel workflows | The integration contract itself is satisfied: `CONTRIBUTING.md` landed in PR #184. The remaining acceptance item, workstream acknowledgements from #152 and #153, is a maintainer process step, not a code change. |
| #151 | Epic: techno-economic reporting scaffold | Epic; no bounded fix. |
| #152 | Epic: bioreactor digital-twin demonstrator | Epic labeled blocked:external. |
| #153 | Epic: crystallization digital-twin demonstrator | Epic; no bounded fix. |
| #154 | Roadmap: 12-month delivery plan | Planning document, not a code change. |
| #168 | Scoping: coil heat-transfer mode for tank reactors | Scoping issue that first needs a maintainer decision on the geometry model; the related validation defect (#169) is handled separately. |
| #172 | Workshop materials as an executable regression suite | Depends on canonical case-study notebooks that must be obtained from the Nagy group (external) and on the Assimulo lane; not actionable here. |
| #212 | Sourced non-ideal activity data for DynamicExtractor tests | Requires primary-source UNIQUAC/UNIFAC parameters that are not in the repository; AGENTS.md forbids inventing provenance, and the work is a follow-up to open PR #206. |
| #215 | Cache reactor state Jacobians across sensitivity callbacks | Performance follow-up that builds on open PR #180; cannot be implemented until that PR lands. |

## Skipped: deferred behind an open pull request

Reviewed on 2026-09-07 against branch `fix/issues` at commit `dbdea97`.

| Issue | Title (short) | Reason |
| --- | --- | --- |
| #178 | VaporPhase.getEnthalpy compares the temperature axis against the species axis | Real defect, deferred: open PR #179 rewrites the same per-species branch of `VaporPhase.getEnthalpy` for #177, so a concurrent fix on this branch would conflict with that PR. Revisit once PR #179 lands. |

Reviewed on 2026-09-08 against branch `fix/issues` at commit `8e6e4d7`.

| Issue | Title (short) | Reason |
| --- | --- | --- |
| #213 | Restore required CI coverage for crystallizer ODE problem callbacks | Follow-up to open PR #204; the choice between making the Assimulo CI job required and extracting a solver-independent callback helper is a CI policy decision for maintainers, and the affected tests live in that PR. |

Reviewed on 2026-09-08 against branch `fix/issues` at commit `c5a2c1d`.

| Issue | Title (short) | Reason |
| --- | --- | --- |
| #140 | Drying initialization state layout | All initialization branches already include the condensed temperature on HEAD. State-vector shape validation at the solver boundary remains unmet; defer it behind PR #210's rewrite into `initialize_states`. |
| #42 | Drying volatile liquid state and species mapping | PR #210 explicitly says "Preserve the provisional behavior owned by #42". Removing the reset also requires correcting the `unit_model` docstring inside PR #210's hunk; defer the state and heat-capacity mapping fix until that PR lands. |

## Skipped: already satisfied on master

Reviewed on 2026-09-08 against branch `fix/issues` at commit `50cfd6e`.

| Issue | Title (short) | Reason |
| --- | --- | --- |
| #33 | PFR steady-state heat-transfer area uses diameter/4 | Already fixed on master by commit 6a0fb16 (both PFR energy balances use `a_prime = 4 / self.diam` [m**2/m**3]); no further change needed. |
