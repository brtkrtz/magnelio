"""Import sweep over the script directories that have no test coverage.

Walks every ``*.py`` file under examples/, validation/ and benchmarks/,
plus any extra directories passed as arguments (e.g. private script
folders kept outside the repository), extracts all ``magnelio`` imports
via the AST (the scripts are never executed), and verifies each imported
module and attribute against the installed package.  API breaks in the
library break these scripts silently otherwise — run this after every
rename.

Usage::

    ~/.local/share/mamba/envs/mio/bin/python \\
        validation/tools/check_imports.py [extra_dir ...]

Exit status 0 when every import resolves, 1 otherwise.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

SCRIPT_DIRS = ("examples", "validation", "benchmarks")


#: Entering one of these defers execution to call time; everything else
#: (module body, ``if``/``try`` guards, class bodies) runs on import.
_DEFERRING_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_type_checking_guard(node: ast.AST) -> bool:
    """True for ``if TYPE_CHECKING:`` — a block that never runs.

    Its imports exist for the type checker only, so counting them as a
    load-time dependency invents cycles that do not exist at run time.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _walk_scoped(node: ast.AST, at_module_level: bool):
    """Yield (node, at_module_level) for the whole tree below *node*."""
    for child in ast.iter_child_nodes(node):
        child_level = at_module_level and not (
            isinstance(child, _DEFERRING_SCOPES) or _is_type_checking_guard(child)
        )
        yield child, child_level
        yield from _walk_scoped(child, child_level)


def _iter_magnelio_imports(tree: ast.AST, *, module_level: bool | None = None):
    """Yield (lineno, module, name, at_module_level) tuples.

    *name* is None for plain ``import x`` statements.  *at_module_level*
    is True when the statement runs at import time, False when it sits in
    a function body: Magnelio defers many imports that way (``# noqa:
    PLC0415``) precisely to avoid import cycles, so a dependency graph
    that conflates the two reports cycles the package does not have.

    Pass *module_level* to yield only one of the two kinds.
    """
    for node, at_level in _walk_scoped(tree, True):
        if module_level is not None and at_level is not module_level:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "magnelio" or alias.name.startswith("magnelio."):
                    yield node.lineno, alias.name, None, at_level
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, not an magnelio reference
                continue
            if node.module == "magnelio" or (node.module or "").startswith("magnelio."):
                for alias in node.names:
                    yield node.lineno, node.module, alias.name, at_level


def _resolve(module: str, name: str | None) -> str | None:
    """Return an error message when the import target does not resolve."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 — report any import failure
        return f"cannot import module {module!r}: {type(exc).__name__}: {exc}"
    if name is None or name == "*":
        return None
    if hasattr(mod, name):
        return None
    try:  # ``from magnelio.x import y`` may name a submodule
        importlib.import_module(f"{module}.{name}")
    except Exception:  # noqa: BLE001
        return f"{module!r} has no attribute {name!r}"
    return None


def _display(path: Path, root: Path) -> str:
    """Path relative to the repo root when inside it, absolute otherwise."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    bases = [root / dirname for dirname in SCRIPT_DIRS]
    bases += [Path(arg).resolve() for arg in sys.argv[1:]]
    failures: list[str] = []
    n_files = 0
    n_imports = 0
    for base in bases:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            n_files += 1
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                failures.append(f"{_display(path, root)}:{exc.lineno}: syntax error: {exc.msg}")
                continue
            for lineno, module, name, _at_module_level in _iter_magnelio_imports(tree):
                n_imports += 1
                error = _resolve(module, name)
                if error is not None:
                    failures.append(f"{_display(path, root)}:{lineno}: {error}")

    print(f"checked {n_imports} magnelio imports in {n_files} files")
    if failures:
        print(f"\n{len(failures)} broken import(s):")
        for line in failures:
            print(f"  {line}")
        return 1
    print("all imports resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
