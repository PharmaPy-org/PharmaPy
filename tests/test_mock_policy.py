"""Structural checks for prohibited test substitutes, with source-only fixtures.

The guard recognizes explicit framework APIs and imported PharmaPy attribute /
process-environment and import-state mutation. It does not execute fixture
source strings or perform general alias/data-flow analysis; review still checks
other substitutes.
"""

from __future__ import annotations

import ast
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
# Mapping mutations shared by os.environ and sys.modules; reads/copies are allowed.
FORBIDDEN_MAPPING_METHODS = frozenset(
    {
        "__setitem__", "__delitem__", "__ior__", "clear", "pop", "popitem",
        "setdefault", "update",
    }
)
FORBIDDEN_LIST_METHODS = frozenset(
    {
        "append", "extend", "insert", "remove", "pop", "clear", "reverse", "sort",
        "__setitem__", "__delitem__", "__iadd__", "__imul__",
    }
)
IMPORT_STATE_PATHS = frozenset(
    {"sys.modules", "sys.meta_path", "builtins.__import__"}
)


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
            replacements = set()
            for target in targets:
                for candidate in ast.walk(target):
                    if not isinstance(
                        getattr(candidate, "ctx", None), (ast.Store, ast.Del)
                    ):
                        continue
                    # Attribute/subscript writes mutate shared state. A plain
                    # imported-name assignment only rebinds a local name; an
                    # augmented mapping/list assignment can mutate the object.
                    import_state_path = None
                    if isinstance(candidate, ast.Attribute):
                        import_state_path = _imported_path(candidate, imports)
                    elif isinstance(candidate, ast.Subscript):
                        import_state_path = _imported_path(candidate.value, imports)
                    elif isinstance(node, ast.AugAssign):
                        path = _imported_path(candidate, imports)
                        if path in {"sys.modules", "sys.meta_path"}:
                            import_state_path = path
                    if import_state_path in IMPORT_STATE_PATHS:
                        replacements.add(import_state_path)
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
            import_state_mutations = set()
            if isinstance(node.func, ast.Attribute):
                environment_mutation |= (
                    _imported_path(node.func.value, imports) == "os.environ"
                    and node.func.attr in FORBIDDEN_MAPPING_METHODS
                )
                receiver = _imported_path(node.func.value, imports)
                if (
                    receiver == "sys.modules"
                    and node.func.attr in FORBIDDEN_MAPPING_METHODS
                ) or (
                    receiver == "sys.meta_path"
                    and node.func.attr in FORBIDDEN_LIST_METHODS
                ):
                    import_state_mutations.add(receiver)
            if (
                call_path in {"setattr", "delattr"}
                or imported_call in {"builtins.setattr", "builtins.delattr"}
            ) and node.args:
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
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    receiver = _imported_path(node.args[0], imports)
                    path = f"{receiver}.{node.args[1].value}"
                    if path in IMPORT_STATE_PATHS:
                        import_state_mutations.add(path)
            for path in sorted(import_state_mutations):
                violations.append(
                    f"{relative_path}:{node.lineno}: {path} replacement"
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
    ("source", "target"),
    [
        (
            "import sys as runtime\n"
            "runtime.modules['assimulo'] = replacement",
            "sys.modules",
        ),
        ("from sys import modules\nmodules['assimulo'] = replacement", "sys.modules"),
        ("from sys import modules as loaded\ndel loaded['assimulo']", "sys.modules"),
        ("import sys\nsys.modules |= {'assimulo': replacement}", "sys.modules"),
        (
            "from sys import modules\n"
            "modules |= {'assimulo': replacement}",
            "sys.modules",
        ),
        ("import sys\nsys.modules = replacement", "sys.modules"),
        ("import sys\nsys.modules: dict = replacement", "sys.modules"),
        ("import sys\ndel sys.modules", "sys.modules"),
        ("import sys\n(sys.modules, local) = values", "sys.modules"),
        ("import sys\nsetattr(sys, 'modules', replacement)", "sys.modules"),
        ("import sys\ndelattr(sys, 'modules')", "sys.modules"),
        (
            "from sys import modules as loaded\n"
            "loaded.__setitem__('assimulo', replacement)",
            "sys.modules",
        ),
        (
            "from sys import modules as loaded\n"
            "loaded.__delitem__('assimulo')",
            "sys.modules",
        ),
        (
            "from sys import modules as loaded\n"
            "loaded.__ior__({'assimulo': replacement})",
            "sys.modules",
        ),
        ("from sys import modules as loaded\nloaded.clear()", "sys.modules"),
        ("from sys import modules as loaded\nloaded.pop('assimulo')", "sys.modules"),
        ("from sys import modules as loaded\nloaded.popitem()", "sys.modules"),
        (
            "from sys import modules as loaded\n"
            "loaded.setdefault('assimulo', replacement)",
            "sys.modules",
        ),
        (
            "from sys import modules as loaded\n"
            "loaded.update(replacement)",
            "sys.modules",
        ),
        ("import sys\nsys.meta_path.insert(0, Blocker())", "sys.meta_path"),
        ("from sys import meta_path\nmeta_path[0] = replacement", "sys.meta_path"),
        ("import sys as runtime\ndel runtime.meta_path[:]", "sys.meta_path"),
        ("from sys import meta_path\nmeta_path += [replacement]", "sys.meta_path"),
        ("import sys\nsys.meta_path *= 0", "sys.meta_path"),
        ("import sys\nsys.meta_path = []", "sys.meta_path"),
        ("import sys\nsetattr(sys, 'meta_path', [])", "sys.meta_path"),
        (
            "from sys import meta_path as finders\n"
            "finders.append(replacement)",
            "sys.meta_path",
        ),
        (
            "from sys import meta_path as finders\n"
            "finders.extend([replacement])",
            "sys.meta_path",
        ),
        (
            "from sys import meta_path as finders\n"
            "finders.remove(replacement)",
            "sys.meta_path",
        ),
        ("from sys import meta_path as finders\nfinders.pop()", "sys.meta_path"),
        ("from sys import meta_path as finders\nfinders.clear()", "sys.meta_path"),
        (
            "from sys import meta_path as finders\n"
            "finders.__setitem__(0, replacement)",
            "sys.meta_path",
        ),
        (
            "from sys import meta_path as finders\n"
            "finders.__delitem__(0)",
            "sys.meta_path",
        ),
        (
            "from sys import meta_path as finders\n"
            "finders.__iadd__([replacement])",
            "sys.meta_path",
        ),
        ("from sys import meta_path as finders\nfinders.__imul__(0)", "sys.meta_path"),
        ("from sys import meta_path as finders\nfinders.reverse()", "sys.meta_path"),
        ("from sys import meta_path as finders\nfinders.sort()", "sys.meta_path"),
        ("import builtins\nbuiltins.__import__ = replacement", "builtins.__import__"),
        (
            "import builtins as builtin_module\n"
            "del builtin_module.__import__",
            "builtins.__import__",
        ),
        (
            "import builtins\n"
            "setattr(builtins, '__import__', replacement)",
            "builtins.__import__",
        ),
        ("import builtins\ndelattr(builtins, '__import__')", "builtins.__import__"),
        (
            "import builtins\n"
            "builtins.setattr(builtins, '__import__', replacement)",
            "builtins.__import__",
        ),
        (
            "import sys\n"
            "from builtins import delattr as delete\n"
            "delete(sys, 'meta_path')",
            "sys.meta_path",
        ),
    ],
)
def test_policy_detects_import_state_mutation(
    tmp_path: Path, source: str, target: str
) -> None:
    """Reject import-state mutation without executing the source fixtures.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated pytest-provided directory for the source sample.
    source : str
        Python source mutating the module cache, finders, or import hook.
    target : str
        Qualified import-state object expected in the diagnostic.
    """
    test_file = tmp_path / "test_example.py"
    test_file.write_text(source, encoding="utf-8")
    violations = _mock_policy_violations(
        test_file, display_path=Path("tests/test_example.py")
    )
    mutation_line = len(source.splitlines())
    assert violations == [
        f"tests/test_example.py:{mutation_line}: {target} replacement"
    ]


@pytest.mark.parametrize(
    "source",
    [
        "import sys\nvalue = sys.modules.get('assimulo')",
        "from sys import modules\ncopy = modules.copy()\ncopy['assimulo'] = replacement",
        "from sys import modules\nmodules = {}",
        "import sys\nfinders = sys.meta_path.copy()\nfinders.insert(0, local)",
        "from sys import meta_path\nfirst = meta_path[0]",
        "from sys import meta_path\nmeta_path = []",
        "import builtins\nmodule = builtins.__import__('sys')",
        "from builtins import __import__\n__import__ = local_function",
        "child_source = 'import sys; sys.meta_path.insert(0, Blocker())'",
        "class Local: pass\nsys = Local()\nsys.modules = {}",
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


def test_tests_do_not_use_prohibited_substitutes() -> None:
    """Reject prohibited substitutes in every Python file in the test suite."""
    test_files = sorted(TESTS_ROOT.rglob("*.py"))
    assert test_files, "No Python tests were discovered for the policy check"

    violations = [
        violation
        for test_file in test_files
        for violation in _mock_policy_violations(test_file)
    ]
    message = (
        "Prohibited test substitutes found. Follow the real-collaborator and "
        "isolated-subprocess alternatives in AGENTS.md; migrate the test "
        "instead of adding an exemption:\n"
        + "\n".join(violations)
    )
    assert not violations, message
