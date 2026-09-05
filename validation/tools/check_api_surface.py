"""Public-API surface audit: every name has exactly one documented home.

The documented home of a public name is the ``__all__`` it appears in —
either the core ``magnelio`` namespace or exactly one domain namespace
(``magnelio.geo``, ``magnelio.ports``, ...) — see DD-117.  This
script flags:

* names listed in more than one public ``__all__`` (duplicate homes),
* ``__all__`` entries that do not resolve,
* public attributes imported into a namespace but missing from its
  ``__all__`` (accidental exports, e.g. the historical ``GridLines`` leak),
* underscore-prefixed modules reachable through any public ``__all__``,
* drift of the core surface away from the DD-117 ``EXPECTED_CORE`` pin.

Usage::

    ~/.local/share/mamba/envs/mio/bin/python validation/tools/check_api_surface.py

Exit status 0 when the surface is clean, 1 otherwise.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import types

import magnelio

# The DD-117 thin core: the model container and run vocabulary, the
# problem classes, and the project-store entry points.  Growing this
# set is an API decision — record it in a DD before editing
# (``Excitation``, ``AnalysisTD``: DD-224; the verbosity switch:
# DD-246).
EXPECTED_CORE = frozenset(
    {
        "__version__",
        "set_verbosity",
        "get_verbosity",
        "GeometryModel",
        "Material",
        "Mesh",
        "MeshControl",
        "BoundaryConditions",
        "Excitation",
        "AnalysisTD",
        "AnalysisScatteringTD",
        "AnalysisEigenmode",
        "resume",
        "open_project",
    }
)


def _public_namespaces() -> list[types.ModuleType]:
    """Top-level package plus every non-underscore direct subpackage/module."""
    namespaces = [magnelio]
    for info in pkgutil.iter_modules(magnelio.__path__, prefix="magnelio."):
        short = info.name.rsplit(".", 1)[-1]
        if short.startswith("_"):
            continue
        namespaces.append(importlib.import_module(info.name))
    return namespaces


def main() -> int:
    problems: list[str] = []
    homes: dict[str, list[str]] = {}

    if set(magnelio.__all__) != EXPECTED_CORE:
        gained = set(magnelio.__all__) - EXPECTED_CORE
        lost = EXPECTED_CORE - set(magnelio.__all__)
        detail = "; ".join(
            part
            for part in (
                f"unexpected: {sorted(gained)}" if gained else "",
                f"missing: {sorted(lost)}" if lost else "",
            )
            if part
        )
        problems.append(f"core __all__ drifted from the DD-117 pin ({detail})")

    namespaces = _public_namespaces()
    for mod in namespaces:
        exported = getattr(mod, "__all__", None)
        if exported is None:
            problems.append(f"{mod.__name__}: public namespace without __all__")
            continue
        for name in exported:
            obj = getattr(mod, name, None)
            if obj is None and name not in vars(mod):
                problems.append(f"{mod.__name__}.__all__ lists unresolvable name {name!r}")
                continue
            if isinstance(obj, types.ModuleType) and obj.__name__.rsplit(".", 1)[-1].startswith(
                "_"
            ):
                problems.append(f"{mod.__name__}.__all__ exposes underscore module {obj.__name__}")
            homes.setdefault(name, []).append(mod.__name__)
        # Accidental exports: public, non-module attributes defined in magnelio
        # but absent from __all__ (imported-and-forgotten names).  A name
        # whose documented home is the core namespace is exempt — its
        # physical definition module deliberately does not re-list it.
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or attr in exported or isinstance(obj, types.ModuleType):
                continue
            if mod is not magnelio and attr in magnelio.__all__:
                continue
            origin = getattr(obj, "__module__", "") or ""
            if origin == mod.__name__ or (mod is magnelio and origin.startswith("magnelio")):
                problems.append(
                    f"{mod.__name__}: leaked public attribute {attr!r} (not in __all__)"
                )

    for name, owners in sorted(homes.items()):
        if len(owners) > 1:
            problems.append(f"{name!r} has {len(owners)} documented homes: {', '.join(owners)}")

    print(f"core __all__: {len(magnelio.__all__)} names; {len(namespaces) - 1} domain namespaces")
    if problems:
        print(f"\n{len(problems)} surface violation(s):")
        for line in problems:
            print(f"  {line}")
        return 1
    print("API surface is clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
