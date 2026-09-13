# PharmaPy Contributor Guidelines

This file is the canonical project guidance for every contributor, human or
coding agent. Read it before making changes. The development workflow and
integration contract live in [CONTRIBUTING.md](CONTRIBUTING.md). Codex reads
`AGENTS.md` natively; `CLAUDE.md` and `.github/copilot-instructions.md` are thin
adapters so that Codex, Claude Code, and GitHub Copilot share one policy. A
separate `CODEX.md` is not needed.

## Project vision

PharmaPy is scientific software for modeling and simulating pharmaceutical
manufacturing processes. Changes must preserve the physical meaning of the
models and make simulations trustworthy, reproducible, and understandable to
chemical-engineering users.

Prioritize, in order:

1. Physical and numerical correctness.
2. Clear units, bases, assumptions, and APIs.
3. Backward compatibility and reproducibility.
4. Maintainability.
5. Performance, when supported by measurements.

## Working principles

- Read the relevant implementation, tests, and documentation before editing.
- Review related open issues and coordination notes before editing. If work
  reveals a defect, dependency, or connection outside the requested scope,
  record it in an existing issue or, when authorized, open a focused follow-up
  instead of widening the current change. Cross-link the source, current,
  follow-up, and owning issues or PRs so the work can be traced before handoff.
- Make the smallest coherent change that solves the stated problem. Do not
  combine unrelated refactors with a fix or feature.
- Preserve public APIs and established model behavior unless the task explicitly
  requires a breaking change. Document any intentional compatibility impact.
- Never hide a physical-basis conversion or silently change a model assumption.
- Treat test data and expected numerical results as scientific evidence; do not
  update them merely to make a failing test pass.
- Remove dead commented code and stale TODOs from touched areas; use linked
  issues for actionable deferred work.
- Preserve existing line endings and keep formatting-only changes separate.
  After editing a tracked file, inspect `git ls-files --eol <path>` and
  `git diff --ignore-cr-at-eol -- <path>`. If line-ending churn appears, reapply
  the edit with the original terminators. Put deliberate normalization in a
  separate change with an explicit `.gitattributes` policy.
- When reviewing CRLF files with `git diff --check`, configure
  `core.whitespace` with `cr-at-eol` so intended carriage returns are not
  reported as trailing whitespace.
- Keep PR scope, compatibility notes, and issue references accurate. Use
  `Closes` only when all relevant acceptance criteria are satisfied.
- Do not commit generated files, local environments, credentials, or machine-
  specific configuration.

## Python documentation

- Every new or modified function and method, including private helpers, must
  have a NumPy-style docstring. When touching an undocumented function, bring
  its docstring into compliance as part of the change.
- Include `Parameters`, `Returns` or `Yields`, and `Raises` sections when they
  apply. Include `Notes`, `Warnings`, or `Examples` when assumptions, equations,
  side effects, conditional returns, defaults, option precedence, or usage would
  otherwise be unclear.
- Document the units, physical basis, expected shape, and meaning of every
  scientific input and output. For example: `temperature : float` followed by
  `Temperature of the liquid phase [K].`
- Class docstrings must describe the model's purpose and important assumptions;
  method docstrings must document method-specific behavior rather than relying
  on the class description.
- Comments should explain intent, assumptions, or non-obvious mathematics. Do
  not narrate code that is already clear.

## Units and physical quantities

- Every new or modified variable that stores a numeric physical or model
  quantity must have an adjacent unit comment in square brackets at its first
  definition. Use established project spellings and SI-derived notation where
  practical:

  ```python
  particle_diameter = 2.5e-4  # [m]
  mass_flow = inlet.mass_flow  # [kg/s]
  conversion = reacted_moles / initial_moles  # [-]
  ```

- Use `[-]` for dimensionless quantities. Pure bookkeeping values such as
  indices, booleans, names, and object references do not require a unit comment.
- For arrays, mappings, or tuples with heterogeneous units, state the unit of
  each component in the adjacent comment or NumPy-style docstring.
- Function parameters and return values must state units in their docstrings.
  Attributes, constants, intermediate calculations, and test fixtures must use
  adjacent unit comments.
- Repeat or update the unit comment whenever a conversion changes the unit or
  basis. Use explicit variable names for conversions instead of reusing a name
  with a different meaning.
- State the basis as well as the unit when ambiguity is possible, such as
  `[J/mol]`, `[J/kg]`, `[mol/s]`, or `[kg/s]`. Never mix mass, molar, volume, or
  particle-number bases implicitly.
- Check dimensional consistency in every equation. Centralize conversions and
  never assign a unit merely from convention without checking the equation.
- Correlations, heuristics, constants, conversions, and cost coefficients must
  state their source or derivation, units, basis, and valid range. Make tunable
  values documented parameters.
- Unit comments are source annotations. Inspect consumers before changing
  user-facing runtime metadata to bracketed notation.

## Constants and modeling assumptions

- Do not introduce unexplained numeric literals ("magic numbers") in model
  equations, algorithms, defaults, conditionals, tolerances, or tests. A
  literal may remain inline only when its meaning is unambiguous from the
  operation or a documented governing equation, such as zero initialization,
  an identity, an index, or an evident mathematical coefficient; otherwise,
  use a descriptive named constant or parameter.
- Justify every such constant where it is defined or used, in an adjacent
  comment or docstring. State what fixes its value: a cited equation, primary
  literature or data source, a shown derivation, an exact unit conversion, a
  physical or numerical constraint, an established algorithm, or an explicit
  modeling or design assumption. Cite the establishing document and its
  page, equation, figure, or table when available. Distinguish a
  dataset-specific observed range from a method's calibrated or validated
  range. Include units or `[-]`, basis, and valid range when they apply.
- Dimensionless ratios, thresholds, empirical factors, initial guesses,
  tolerances, and defaults are not exempt. Label heuristics and design choices
  as assumptions, explain why the selected value is appropriate, and make them
  documented parameters when users or models may reasonably need to change
  them.
- Do not invent provenance. If no defensible value or rationale exists, stop
  and request maintainer or domain guidance rather than silently choosing a
  plausible number. Do not commit a numeric artifact labeled "not verified";
  confirm its source and meaning or remove it. In tests, identify the physical
  or modeling case that fixture constants construct and derive expected values
  independently.

## Numerical and API conventions

- Use descriptive, domain-relevant names. Avoid one-letter names except for
  conventional short-lived mathematical indices or coordinates.
- Add type hints to new or modified public APIs and to internal code when they
  clarify array shapes, optional values, or return structure.
- Preserve scalar-versus-array behavior. Document and test exact NumPy shapes,
  axis meanings, positional state order, and physical basis; prefer named
  structures at public boundaries when practical.
- Validate inputs at public boundaries and raise specific exceptions with
  actionable messages for invalid units, shapes, ranges, or model state.
- Validate finite-choice options explicitly; never let an unknown value silently
  select a plausible but unintended model.
- Before changing the spelling or casing of a sentinel, enum-like value, or
  finite-choice option, search repository-wide for its producers, assignments,
  consumers, and comparisons. Treat mixed conventions as a latent bug and
  update distant consumers deliberately.
- Before adding hard validation, enumerate package, example, notebook, and test
  call sites for the affected API. A green rejection test alone does not prove
  that established valid usage remains compatible.
- When replacing a hard-coded value with a parameter, show that the generalized
  formulation reduces exactly to the previous formulation at the old assumed
  value and test that equivalence. Sweep a physically meaningful range and
  inspect convergence, residuals, and trend direction; repository consumers
  define the authoritative compatibility surface.
- Use explicit, justified tolerances for floating-point comparisons. Tests
  should use `pytest.approx` or NumPy testing helpers rather than exact equality
  for calculated floating-point values.
- Handle zero, negative, empty, NaN, and infinite values deliberately when they
  are plausible at a model boundary.
- Prefer existing project abstractions and NumPy/SciPy operations over duplicate
  helpers. Add a dependency only when the benefit justifies the maintenance cost.

## Testing and verification

- Add or update tests for every behavior change and bug fix. Demonstrate a
  distinct red/green result for each independent behavior by reverting its
  implementation hunk when practical. A test merely failing on the base branch
  is insufficient when an unrelated change could cause that failure. Derive
  expected values independently rather than duplicating the production
  expression.
- Inspect the actual red failure and confirm that it exercises the intended
  defect. Avoid broad `raises` assertions and stubs that fail for the wrong
  reason; check a sentinel value or specific exception and guard against
  vacuous greens such as recomputed expectations, bypassed paths, unentered stub
  branches, or two empty collections.
- For aggregate or structural changes, include content-preserving mutations
  such as reordered fields or columns and assert values and ordering, not only
  counts or shapes.
- Exercise the affected public path and caller-to-callee handoff, not only a
  repaired helper. Assert values, ordering, and basis rather than only shape or
  absence of exceptions; use unequal dimensions and asymmetric fixtures where
  they can expose axis or ordering mistakes.
- Include nominal behavior plus relevant boundary, dimensional, and failure
  cases. Assert units or physical basis indirectly through known balances or
  conversions when practical.
- Give each test module a concise docstring covering its scope and noteworthy
  fixtures, backends, or cost. Document individual tests only when their names
  and bodies are not self-explanatory.
- Prefer representative real collaborators. Use monkeypatches and stubs only at
  narrow optional, expensive, or external boundaries, and assert the actual
  handoff. Link the blocker when stable end-to-end coverage must be deferred.
- When expected behavior depends on an unfixed defect elsewhere, identify the
  blocking issue in the test docstring or adjacent comment, state the
  provisional expectation and what must change after the fix, and repeat that
  limitation in the PR body. A stopgap assertion is not the intended contract;
  update it when the blocker is fixed.
- Do not make a check green by deleting, skipping, or xfail-marking a relevant
  test; loosening a tolerance; adding `noqa`; or broadening warning ignores or
  filters. Resolve the underlying artifact or obtain maintainer direction when
  the check itself is wrong.
- Use the markers defined in `pytest.ini`: `unit`, `integration`, `slow`, and
  `assimulo`. Do not make core tests depend on the optional Assimulo stack.
- Exercise missing-optional-dependency fallbacks explicitly, either by blocking
  the import in a focused test or in an environment where the dependency is
  genuinely absent; a green rich environment proves only the installed path.
  Tests that import an optional-backend module must apply the matching marker
  and `pytest.importorskip` before that import. Inventory every optional import
  performed during module loading so the minimal CI lane remains dependency
  independent.
- Run the narrowest relevant tests while developing, then run the locked core
  task before handoff:

  ```bash
  pixi run test
  ```

- When pixi is unavailable but the dependencies are already provisioned in an
  active environment, run the underlying core command:

  ```bash
  python -m pytest tests/ -m "not assimulo"
  ```

- For solver-backed changes, also run the locked Assimulo task:

  ```bash
  pixi run -e assimulo test-assimulo
  ```

  `TESTING.md` documents the manual conda fallback. If neither environment is
  available, say exactly which verification was not run and why.
- Keep `[project]` and `[tool.pixi]` in `pyproject.toml`, `pixi.lock`,
  dependency mirrors, and `DEPENDENCIES.md` synchronized when changing
  dependencies or environment tasks.
- In workflows, use least-privilege permissions and supported actions, and avoid
  duplicate runs. Avoid `eval`, `exec`, or shell discovery when normal APIs work.

## Documentation and handoff

- Update user documentation and examples when public behavior, model equations,
  inputs, outputs, units, or installation steps change.
- In the final change summary, explain the scientific or behavioral effect,
  identify tests run, and call out assumptions, compatibility concerns, and
  unverified optional-solver paths.
- If a request conflicts with these guidelines, surface the conflict and obtain
  maintainer direction instead of silently ignoring the policy.
