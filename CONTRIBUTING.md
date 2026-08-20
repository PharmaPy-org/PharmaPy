# Contributing to PharmaPy

Thank you for contributing to PharmaPy. This document describes the
development workflow: how changes are proposed, kept in sync, and integrated.
The engineering standards themselves — physical correctness, units, docstrings,
testing, and verification — live in [AGENTS.md](AGENTS.md) and are binding for
every contribution, whether written by a person or a coding agent.

## Where the rules live

- [AGENTS.md](AGENTS.md) — canonical coding, units, documentation, and
  testing standards for all changes.
- [TESTING.md](TESTING.md) — test environments and how to run each lane.
- [DEPENDENCIES.md](DEPENDENCIES.md) — dependency policy and environments.
- [INSTALLATION.md](INSTALLATION.md) — installation instructions.

## Development setup

PharmaPy uses [pixi](https://pixi.sh) for reproducible development
environments backed by conda-forge. Follow [INSTALLATION.md](INSTALLATION.md)
to install pixi and create the development environments, then use
[TESTING.md](TESTING.md) for the canonical locked setup and test-lane commands.

## Pull request basics

- Contributors without repository write access first fork PharmaPy, create a
  focused branch from the current upstream `master`, and open a pull request
  against `PharmaPy-org/PharmaPy:master`. Contributors with write access may
  create the branch in this repository directly.
- Follow [AGENTS.md](AGENTS.md) for scope, testing, documentation, and issue-link
  requirements.
- Request maintainer review after the applicable required CI checks pass. The
  `Assimulo integration tests` job is informational; see
  [TESTING.md](TESTING.md). A maintainer merges accepted changes.

## Integration contract for parallel workstreams

A *workstream* is development coordinated by project maintainers or
collaborators and expected to run for multiple weeks in parallel with `master`
— a new capability module, an architecture refactor, or a large migration.
Outside contributors follow **Pull request basics** unless a maintainer agrees
to coordinate their contribution as a workstream; that maintainer owns its
project-board entry. The clauses below exist because long-lived branches that
drift silently are far more expensive to merge than the weekly discipline they
replace. They apply from the stream's first commit.

1. **Branch from `master` and open a draft pull request within the first
   week** of work, so the stream is visible from day one.
2. **The coordinating maintainer tracks every stream on the team project
   board**
   ([PharmaPy Development](https://github.com/orgs/PharmaPy-org/projects/1))
   with an owner and a current status.
3. **Update from `master` at least weekly**, by rebase or by an append-only
   merge from `master`. A stream must never sit more than two weeks behind
   `master`, and no pull request in the stream stays open longer than two
   weeks — slice the work so each PR merges within that window.
4. **Tests accompany every PR** in the stream, per [AGENTS.md](AGENTS.md).
   Parallel streams get no exemption: a stream that merges without tests
   transfers its verification debt to whoever integrates after it.
5. **Unit comments on every new physical quantity** at first definition, for
   example `mass_flow = inlet.mass_flow  # [kg/s]`, and NumPy-style
   docstrings stating units, bases, and assumptions, per
   [AGENTS.md](AGENTS.md).
6. **No `_old`, `_refactor`, scratch, or editor-configuration files** on any
   branch pushed to this repository. Superseded code is deleted, not renamed;
   history is preserved by git.
7. **Declare the merge order when two streams touch the same modules**, in
   both pull request descriptions and on both tracking issues. The newer,
   smaller, better-tested stream normally reconciles onto the other; when
   those factors disagree, maintainers record which stream reconciles. Until
   an order is recorded, an architecture refactor merges before capability
   streams that build on the affected modules.
8. **Acknowledge this contract before the first commit.** Each incoming
   workstream posts a short acknowledgement on its tracking issue confirming
   the clauses above.

Maintainers may pause a stream that repeatedly falls out of contract (for
example, two consecutive missed weekly syncs or a pull request open longer
than two weeks) and land what exists before the stream continues.
