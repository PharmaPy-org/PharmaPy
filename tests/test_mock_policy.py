"""Structural checks for prohibited test substitutes, with source-only fixtures.

The guard recognizes explicit framework APIs and imported PharmaPy attribute /
process-environment mutation. It does not execute fixture source strings or
perform general alias/data-flow analysis; review still checks other substitutes.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Mapping

import pytest


pytestmark = pytest.mark.unit

TESTS_ROOT = Path(__file__).resolve().parent
FORBIDDEN_MODULES = frozenset(
    {"_pytest.monkeypatch", "mock", "pytest_mock", "unittest.mock"}
)
FORBIDDEN_CONSTRUCTORS = frozenset(
    {
        "AsyncMock",
        "MagicMock",
        "Mock",
        "MonkeyPatch",
        "NonCallableMagicMock",
        "NonCallableMock",
        "PropertyMock",
    }
)
FORBIDDEN_FIXTURES = frozenset({"mocker", "monkeypatch"})
# os._Environ exposes these mutating mapping methods; reads/copies are allowed.
FORBIDDEN_ENVIRON_METHODS = frozenset(
    {
        "__setitem__", "__delitem__", "__ior__", "clear", "pop", "popitem",
        "setdefault", "update",
    }
)
FORBIDDEN_SYS_MODULE_METHODS = frozenset(
    {"__setitem__", "clear", "pop", "popitem", "setdefault", "update"}
)

# Issue #202 removes these exact legacy files in focused PRs. The content
# digests form a ratchet: any edit to a file that still uses monkeypatch makes
# the repository check fail. A migration must remove both every violation
# and its now-obsolete exemption row.
LEGACY_MONKEYPATCH_FILE_DIGESTS = {
    "tests/test_batch_cryst_concentration_jacobian.py": (
        "df111caf9baee2bd10a3f983ef8f90d902f2b678505328125fa30c62331db0f2"
    ),
    "tests/test_crystallizer_energy_balances.py": (
        "b48a7902a1046f06472e9a39a041826267fdd3f302589dbf64ae894e2eb2551b"
    ),
    "tests/test_deliquoring_particle_size_units.py": (
        "e9dc7ce86a1361c3f4b486586678d7f0f801aff3ad7bceab01ee9eed913a12d6"
    ),
    "tests/test_drying_energy_rate_basis.py": (
        "92c65c66296a5a57ee6e06fc695bc6eccf215dc37bc597aab49726c1c4b2fb88"
    ),
    "tests/test_drying_gas_balance.py": (
        "1efc9c1940200e60b39ecbf8cd020f049c6ccc57d3f9af070964abda8cd22c38"
    ),
    "tests/test_drying_latent_heat_factor.py": (
        "0c50a7437f94113114901fc630f5b167d0ec00546bdda7027b64805d22d00fd7"
    ),
    "tests/test_drying_model.py": (
        "d4592e91a578bcfec961530a659d3490b84d6b376231775e96d91f2ad773c9c5"
    ),
    "tests/test_dynamic_collector_state_ordering.py": (
        "10af90d6a1a1c682d6f3ba6de5e034b4b55d4f0139d6b5b930ed4e173c37d58c"
    ),
    "tests/test_dynamic_extractor_default_k_fun.py": (
        "cf9a39a039806799756b6d3ea2b9353ad623afbd8b58cec02414feacbd8216d3"
    ),
    "tests/test_dynamic_extractor_stage_efficiency.py": (
        "8a666ab0e76c7423de36c1019f5449a1bd882240932f5790e755974575c2e71e"
    ),
    "tests/test_exception_handling.py": (
        "22617523a4774cd3dfc40400778915494b3c0fc932fe07f0c626868887287d9a"
    ),
    "tests/test_extractor_modes.py": (
        "657659bd3ed2c53eed18935e15e45586ce4e190a75d203c9e353b4ac7ad3e49b"
    ),
    "tests/test_metamodeler_codegen.py": (
        "958a52435715c302ea81b3d5c17ce4dc14fa9ae3ea9100f30774c4860a33f02d"
    ),
    "tests/test_optional_assimulo_imports.py": (
        "e65a16b56b0620f02a7e831f995c2e07b8f6b70b8caf8eb480257dc56d5f1480"
    ),
    "tests/test_paramestim_calibration_fit_predict.py": (
        "39da4b9c6ff5e81bf625cb6c7cf44b05f231d53ad364a1ef457c6e8826bca304"
    ),
    "tests/test_paramestim_gradient_assembly.py": (
        "b73418cd050a3abaa428caced65cb78c654f25801f970c808b30d92bfbce16c8"
    ),
    "tests/test_phases_vapor_latent_heat_shape.py": (
        "e0c96b1d4fea3726d5dde9ed95cd14d9b0f39fd42adc83da28eee18e0d9c72d2"
    ),
    "tests/test_reactors_energy.py": (
        "d636392346ebbef16de3bd6abf13f4f9aa6c6cbec1e5891466a4162a27bd0298"
    ),
    "tests/test_reversible_jacobian_reverse_term.py": (
        "e568cb5ac7d3ede1bf4429974303872a00b8276b1083f1de72513bc37fb619a8"
    ),
    "tests/test_simexec_routing.py": (
        "7abab90e232de7f27614c51ea4d424346254dd4a03682dbaacefcf9c3b1f5afa"
    ),
}


def _is_forbidden_module(module_name: str) -> bool:
    """Return whether an import belongs to a prohibited mock framework.

    Parameters
    ----------
    module_name : str
        Fully qualified imported module name.

    Returns
    -------
    bool
        ``True`` when the import is a prohibited module or submodule.
    """
    return any(
        module_name == forbidden
        or module_name.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_MODULES
    )


def _attribute_path(node: ast.AST) -> str | None:
    """Return the dotted path represented by a name or attribute node.

    Parameters
    ----------
    node : ast.AST
        Syntax node that may represent a dotted attribute lookup.

    Returns
    -------
    str or None
        Dotted attribute path, or ``None`` for another expression type.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_path(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _imported_path(node: ast.AST, imports: Mapping[str, str]) -> str | None:
    """Resolve a dotted lookup rooted in an explicitly imported name.

    Parameters
    ----------
    node : ast.AST
        Name or attribute expression to inspect.
    imports : mapping of str to str
        Local import bindings and their fully qualified module/object paths.

    Returns
    -------
    str or None
        Qualified imported path, or ``None`` for local objects or expressions.

    Notes
    -----
    This structural check follows import aliases, not arbitrary assignments,
    dynamically obtained objects, or interprocedural data flow.
    """
    path = _attribute_path(node)
    if path is None:
        return None
    root, separator, suffix = path.partition(".")
    imported = imports.get(root)
    if imported is None:
        return None
    return imported + separator + suffix


def _is_pharmapy_import(node: ast.AST, imports: Mapping[str, str]) -> bool:
    """Identify a lookup rooted in a PharmaPy import rather than an instance.

    Parameters
    ----------
    node : ast.AST
        Candidate module or imported class expression.
    imports : mapping of str to str
        Local import bindings and their qualified paths.

    Returns
    -------
    bool
        Whether the expression resolves inside the PharmaPy namespace.
    """
    path = _imported_path(node, imports)
    return path is not None and (path == "PharmaPy" or path.startswith("PharmaPy."))


def _target_mutates_sys_modules(node: ast.AST) -> bool:
    """Return whether an assignment target replaces ``sys.modules`` state.

    Parameters
    ----------
    node : ast.AST
        Assignment or deletion target to inspect.

    Returns
    -------
    bool
        ``True`` when the target contains a ``sys.modules[...]`` lookup.
    """
    return any(
        isinstance(candidate, ast.Subscript)
        and _attribute_path(candidate.value) == "sys.modules"
        for candidate in ast.walk(node)
    )


def _mock_policy_violations(
    test_file: Path, display_path: Path | None = None
) -> list[str]:
    """Find prohibited substitute APIs in one test file.

    Parameters
    ----------
    test_file : pathlib.Path
        Python test source to inspect.
    display_path : pathlib.Path, optional
        Path to show in diagnostics. Defaults to the repository-relative path.

    Returns
    -------
    list[str]
        Human-readable violations with repository-relative paths and lines.
    """
    source = test_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(test_file))
    relative_path = display_path or test_file.relative_to(TESTS_ROOT.parent)
    violations = []
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                binding = imported.asname or imported.name.split(".")[0]
                imports[binding] = (
                    imported.name if imported.asname else binding
                )
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for imported in node.names:
                imports[imported.asname or imported.name] = (
                    f"{node.module}.{imported.name}"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if _is_forbidden_module(imported.name):
                    violations.append(
                        f"{relative_path}:{node.lineno}: import {imported.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imports_unittest_mock = (
                module_name == "unittest"
                and any(imported.name == "mock" for imported in node.names)
            )
            imports_pytest_monkeypatch = module_name == "pytest" and any(
                imported.name == "MonkeyPatch" for imported in node.names
            )
            if (
                _is_forbidden_module(module_name)
                or imports_unittest_mock
                or imports_pytest_monkeypatch
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: from {module_name} import"
                )
        elif isinstance(node, ast.arg) and node.arg in FORBIDDEN_FIXTURES:
            violations.append(
                f"{relative_path}:{node.lineno}: prohibited fixture '{node.arg}'"
            )
        elif (
            isinstance(node, ast.arg)
            and node.annotation is not None
            and ast.unparse(node.annotation).endswith("MonkeyPatch")
        ):
            violations.append(
                f"{relative_path}:{node.lineno}: pytest.MonkeyPatch annotation"
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = (
                node.targets if isinstance(node, (ast.Assign, ast.Delete))
                else [node.target]
            )
            if any(_target_mutates_sys_modules(target) for target in targets):
                violations.append(
                    f"{relative_path}:{node.lineno}: sys.modules replacement"
                )
            replacements = set()
            for target in targets:
                for candidate in ast.walk(target):
                    if not isinstance(
                        getattr(candidate, "ctx", None), (ast.Store, ast.Del)
                    ):
                        continue
                    if isinstance(candidate, ast.Attribute):
                        if _is_pharmapy_import(candidate.value, imports):
                            replacements.add("PharmaPy attribute")
                        if _imported_path(candidate, imports) == "os.environ":
                            replacements.add("environment")
                    elif isinstance(candidate, ast.Subscript):
                        if _imported_path(candidate.value, imports) == "os.environ":
                            replacements.add("environment")
                    elif isinstance(node, ast.AugAssign):
                        if _imported_path(candidate, imports) == "os.environ":
                            replacements.add("environment")
            for replacement in sorted(replacements):
                violations.append(
                    f"{relative_path}:{node.lineno}: {replacement} replacement"
                )
        elif isinstance(node, ast.Call):
            call_path = _attribute_path(node.func)
            imported_call = _imported_path(node.func, imports)
            environment_mutation = imported_call in {"os.putenv", "os.unsetenv"}
            if isinstance(node.func, ast.Attribute):
                environment_mutation |= (
                    _imported_path(node.func.value, imports) == "os.environ"
                    and node.func.attr in FORBIDDEN_ENVIRON_METHODS
                )
            if call_path in {"setattr", "delattr"} and node.args:
                if _is_pharmapy_import(node.args[0], imports):
                    violations.append(
                        f"{relative_path}:{node.lineno}: PharmaPy attribute replacement"
                    )
                environment_mutation |= (
                    _imported_path(node.args[0], imports) == "os"
                    and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "environ"
                )
            if environment_mutation:
                violations.append(
                    f"{relative_path}:{node.lineno}: environment replacement"
                )
            if call_path == "pytest.MonkeyPatch.context":
                violations.append(
                    f"{relative_path}:{node.lineno}: pytest.MonkeyPatch.context(...)"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "getfixturevalue"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in FORBIDDEN_FIXTURES
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: prohibited dynamic fixture "
                    f"'{node.args[0].value}'"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and _attribute_path(node.func.value) == "sys.modules"
                and node.func.attr in FORBIDDEN_SYS_MODULE_METHODS
            ):
                violations.append(
                    f"{relative_path}:{node.lineno}: sys.modules replacement"
                )

            if isinstance(node.func, ast.Name):
                constructor_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                constructor_name = node.func.attr
            else:
                continue

            if constructor_name in FORBIDDEN_CONSTRUCTORS:
                violations.append(
                    f"{relative_path}:{node.lineno}: {constructor_name}(...)"
                )

    return violations


def _legacy_exemption_failures(
    file_violations: Mapping[str, list[str]],
    legacy_digests: Mapping[str, str],
) -> list[str]:
    """Find legacy exemption rows that no longer describe live debt.

    Parameters
    ----------
    file_violations : mapping of str to list of str
        Prohibited substitute diagnostics keyed by repository-relative path.
    legacy_digests : mapping of str to str
        Exact source digests temporarily exempted under issue #202.

    Returns
    -------
    list[str]
        Diagnostics for missing or already-migrated legacy files.
    """
    failures = []
    for relative_path in legacy_digests:
        if relative_path not in file_violations:
            failures.append(
                f"{relative_path}: remove the exemption for the missing file"
            )
        elif not file_violations[relative_path]:
            failures.append(
                f"{relative_path}: remove the obsolete exemption after migration"
            )
    return failures


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from unittest.mock import Mock\nvalue = Mock()\n", "unittest.mock"),
        (
            "def test_example(monkeypatch):\n"
            "    monkeypatch.setattr(object, 'name', 1)\n",
            "prohibited fixture 'monkeypatch'",
        ),
        ("from pytest import MonkeyPatch\n", "from pytest import"),
        (
            "def test_example(patcher: pytest.MonkeyPatch):\n    pass\n",
            "pytest.MonkeyPatch annotation",
        ),
        (
            "with pytest.MonkeyPatch.context() as patcher:\n    pass\n",
            "pytest.MonkeyPatch.context",
        ),
        (
            "def test_example(request):\n"
            "    request.getfixturevalue('monkeypatch')\n",
            "prohibited dynamic fixture 'monkeypatch'",
        ),
        (
            "import sys\nsys.modules['assimulo'] = object()\n",
            "sys.modules replacement",
        ),
    ],
)
def test_policy_detects_prohibited_substitute_apis(
    tmp_path: Path, source: str, expected: str
) -> None:
    """Reject representative mock and monkeypatch spellings structurally.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest-provided directory for the source sample.
    source : str
        Python source containing a prohibited test-substitute API.
    expected : str
        Diagnostic fragment expected from the structural scan.
    """
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")

    violations = _mock_policy_violations(
        test_file, display_path=Path("tests/test_example.py")
    )

    assert any(expected in violation for violation in violations)


@pytest.mark.parametrize(
    "source",
    [
        "import PharmaPy.Reactors\nPharmaPy.Reactors.solve = replacement",
        "import PharmaPy.Reactors as reactors\nreactors.solve = replacement",
        "from PharmaPy import Reactors\ndel Reactors.solve",
        "from PharmaPy.Reactors import BatchReactor\n"
        "BatchReactor.unit_model = replacement",
        "from PharmaPy.Reactors import BatchReactor as Reactor\n"
        "setattr(Reactor, 'unit_model', replacement)",
        "import PharmaPy.Reactors\n"
        "setattr(PharmaPy.Reactors, 'solve', replacement)",
        "from PharmaPy import Reactors\ndelattr(Reactors, 'solve')",
        "from PharmaPy import Reactors\nReactors.solve: object = replacement",
        "from PharmaPy import Reactors\nReactors.solve += replacement",
        "from PharmaPy import Reactors\n(Reactors.solve, local) = values",
    ],
)
def test_policy_detects_imported_pharmapy_attribute_replacement(
    tmp_path: Path, source: str
) -> None:
    """Reject module and class replacement without executing the samples.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest-provided directory for the source sample.
    source : str
        Python source assigning or deleting an imported PharmaPy attribute.
    """
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")
    violations = _mock_policy_violations(
        test_file, display_path=Path("tests/test_example.py")
    )
    assert violations == [
        "tests/test_example.py:2: PharmaPy attribute replacement"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "import os\nos.environ['POLICY_TEST'] = 'value'",
        "import os as operating_system\n"
        "del operating_system.environ['POLICY_TEST']",
        "from os import environ as environment\n"
        "environment['POLICY_TEST']: str = 'value'",
        "import os\nos.environ |= {'POLICY_TEST': 'value'}",
        "import os\nos.environ.update({'POLICY_TEST': 'value'})",
        "from os import environ\nenviron.clear()",
        "from os import environ\nenviron.pop('POLICY_TEST')",
        "import os\nos.environ.popitem()",
        "import os\nos.environ.setdefault('POLICY_TEST', 'value')",
        "import os\nos.environ.__setitem__('POLICY_TEST', 'value')",
        "import os\nos.environ.__delitem__('POLICY_TEST')",
        "import os\nos.environ = replacement",
        "import os\nsetattr(os, 'environ', replacement)",
        "import os\ndelattr(os, 'environ')",
        "import os\nos.putenv('POLICY_TEST', 'value')",
        "from os import unsetenv\nunsetenv('POLICY_TEST')",
    ],
)
def test_policy_detects_environment_mutation(tmp_path: Path, source: str) -> None:
    """Reject mutation of the live environment, including imported aliases.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest-provided directory for the source sample.
    source : str
        Python source mutating the process environment.
    """
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")
    violations = _mock_policy_violations(
        test_file, display_path=Path("tests/test_example.py")
    )
    assert violations == ["tests/test_example.py:2: environment replacement"]


@pytest.mark.parametrize(
    "source",
    [
        "class Local: pass\nsetattr(Local, 'name', object())",
        "class Local: pass\nLocal.name = object()",
        "from PharmaPy.Reactors import BatchReactor\n"
        "reactor = BatchReactor()\nreactor.Phases = phases",
        "from PharmaPy.Reactors import BatchReactor\n"
        "reactor = BatchReactor()\nsetattr(reactor, 'Phases', phases)",
        "import os\nvalue = os.environ['POLICY_TEST']",
        "from os import environ\nvalue = environ.get('POLICY_TEST')",
        "import os\nchild_environment = os.environ.copy()\n"
        "child_environment['POLICY_TEST'] = 'value'",
        "source = \"import os; os.environ['POLICY_TEST'] = 'value'\"",
        "# setattr(PharmaPy.Reactors, 'solve', replacement)",
    ],
)
def test_policy_allows_real_object_setup_and_environment_copies(
    tmp_path: Path, source: str
) -> None:
    """Keep ordinary setup, read access, and subprocess source outside the ban.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest-provided directory for the source sample.
    source : str
        Python source containing permitted object or environment operations.
    """
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")
    assert _mock_policy_violations(
        test_file, display_path=Path("tests/test_example.py")
    ) == []


def test_policy_rejects_retired_legacy_exemptions() -> None:
    """Require missing and migrated files to remove obsolete digest rows."""
    relative_path = "tests/test_migrated.py"

    migrated_failures = _legacy_exemption_failures(
        {relative_path: []}, {relative_path: "retired-digest"}
    )
    missing_failures = _legacy_exemption_failures(
        {}, {relative_path: "missing-digest"}
    )

    assert migrated_failures == [
        f"{relative_path}: remove the obsolete exemption after migration"
    ]
    assert missing_failures == [
        f"{relative_path}: remove the exemption for the missing file"
    ]


def test_tests_do_not_use_prohibited_substitutes() -> None:
    """Keep new mock and monkeypatch APIs out of the PharmaPy test suite."""
    test_files = sorted(TESTS_ROOT.rglob("*.py"))
    assert test_files, "No Python tests were discovered for the policy check"

    file_violations = {}
    for test_file in test_files:
        relative_path = test_file.relative_to(TESTS_ROOT.parent).as_posix()
        file_violations[relative_path] = _mock_policy_violations(test_file)

    violations = _legacy_exemption_failures(
        file_violations, LEGACY_MONKEYPATCH_FILE_DIGESTS
    )
    for relative_path, path_violations in file_violations.items():
        if not path_violations:
            continue

        test_file = TESTS_ROOT.parent / relative_path
        legacy_digest = LEGACY_MONKEYPATCH_FILE_DIGESTS.get(relative_path)
        normalized_source = test_file.read_text(encoding="utf-8")
        current_digest = hashlib.sha256(
            normalized_source.encode("utf-8")
        ).hexdigest()
        if current_digest == legacy_digest:
            continue

        violations.extend(path_violations)

    message = (
        "Prohibited test substitutes found. Legacy files are grandfathered "
        "only at the exact digests in LEGACY_MONKEYPATCH_FILE_DIGESTS; remove "
        "all prohibited substitutes whenever one of those files changes:\n"
        + "\n".join(violations)
    )
    assert not violations, message
