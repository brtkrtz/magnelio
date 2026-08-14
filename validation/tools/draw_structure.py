"""Structure diagrams for the Magnelio package.

Four views, because no single diagram carries them all:

``packages``
    Which subpackage imports which, as a layered graph.  The most
    informative of the four — it shows the pipeline and any cycle.
``inheritance``
    The class hierarchy below a root class.  Flat by design, so what it
    really shows is which classes are public and which are not.
``composition``
    How shapes nest into one another at run time: leaves, unary,
    binary and n-ary nodes of the geometry expression tree.
``classes``
    One cluster per subpackage, optionally with members.

Nothing here is part of the library or its published documentation; it
is a developer's map.  No third-party dependency is used — the DOT text
is written directly and rendered by the system ``dot`` binary, which
only ``--render`` needs.

Why four views and not one, and why the import graph needs two
corrections before it tells the truth: DD-145.

Usage::

    ~/.local/share/mamba/envs/mio/bin/python \\
        validation/tools/draw_structure.py packages [--render svg] [--stats]

Without ``--render`` the DOT source goes to stdout, so it can be piped
into ``dot`` or diffed between revisions.  ``--stats`` prints the
underlying numbers as text instead of a graph.

Exit status 0 on success, 1 when rendering fails.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from check_imports import _iter_magnelio_imports  # noqa: E402

REPO_ROOT = _TOOLS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src" / "magnelio"
DEFAULT_OUT = REPO_ROOT / "validation" / "results" / "structure"

# Palette: public things are drawn in colour, internals in grey, and
# anything the reader should treat as a warning in red.
COLOUR_PUBLIC = "#2b6cb0"
COLOUR_INTERNAL = "#718096"
COLOUR_WARN = "#c53030"
COLOUR_LEAF = "#2f855a"


# ─── shared helpers ─────────────────────────────────────────────────────────


def _subpackage(path: Path) -> str:
    """The subpackage a source file belongs to ('(top)' for magnelio/*.py)."""
    rel = path.relative_to(SRC_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else "(top)"


def _quote(text: str) -> str:
    """Escape a string for use as a DOT identifier or plain label."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _quote_label(text: str) -> str:
    """Quote a label while leaving DOT's own escapes intact.

    ``\\l`` (left-justified line break) and ``\\n`` have to reach DOT
    unescaped; running them through :func:`_quote` would double the
    backslash and print them literally.
    """
    return '"' + text.replace('"', '\\"') + '"'


def _is_internal(name: str) -> bool:
    return name.rsplit(".", 1)[-1].startswith("_")


def _parse_sources(root: Path):
    """Yield (path, tree) for every parseable source file under *root*."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            yield path, ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a broken file should not kill the map
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)


def _render(dot_text: str, name: str, out_dir: Path, fmt: str) -> int:
    """Write *dot_text* to *out_dir* and render it with the dot binary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{name}.dot"
    dot_path.write_text(dot_text, encoding="utf-8")
    image_path = out_dir / f"{name}.{fmt}"
    try:
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(image_path)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print(
            "error: the 'dot' binary was not found.  Install graphviz, or "
            "drop --render to get the DOT source on stdout.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: dot failed: {exc.stderr.decode(errors='replace')}", file=sys.stderr)
        return 1
    print(f"wrote {image_path.relative_to(REPO_ROOT)}")
    return 0


def _emit(dot_text: str, name: str, args) -> int:
    """Render to a file or print the DOT source, per the parsed args."""
    if args.render:
        return _render(dot_text, name, Path(args.out), args.render)
    print(dot_text)
    return 0


# ─── strongly connected components ──────────────────────────────────────────


def _strongly_connected(nodes, successors) -> list[list[str]]:
    """Tarjan's SCC, iterative so a deep graph cannot blow the stack.

    Returns the components in reverse topological order.  Any component
    with more than one member is an import cycle.
    """
    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    for root in nodes:
        if root in index_of:
            continue
        # Each frame is [node, iterator over its successors].
        work = [(root, iter(sorted(successors(root))))]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index_of:
                    index_of[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(successors(child)))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index_of[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                result.append(sorted(component))
    return result


def _layer_of(components, edges) -> dict[int, int]:
    """Longest-path depth of each component on the condensed DAG.

    The condensation is acyclic by construction, so this terminates and
    is independent of visit order — unlike a recursive depth over the
    raw graph, which a cycle makes path-dependent.
    """
    owner = {node: i for i, comp in enumerate(components) for node in comp}
    dag = defaultdict(set)
    for src, dst in edges:
        if owner[src] != owner[dst]:
            dag[owner[src]].add(owner[dst])
    depth: dict[int, int] = {}

    def compute(index: int) -> int:
        if index in depth:
            return depth[index]
        depth[index] = 0  # provisional; the DAG cannot revisit it
        children = dag.get(index, ())
        depth[index] = 1 + max((compute(c) for c in children), default=-1)
        return depth[index]

    for i in range(len(components)):
        compute(i)
    return depth


# ─── view: packages ─────────────────────────────────────────────────────────


def _package_edges():
    """Import counts between subpackages, split by when they execute."""
    module_level: Counter = Counter()
    deferred: Counter = Counter()
    for path, tree in _parse_sources(SRC_ROOT):
        source = _subpackage(path)
        for _lineno, module, _name, at_module_level in _iter_magnelio_imports(tree):
            parts = module.split(".")
            target = parts[1] if len(parts) > 1 else "(top)"
            if target == source:
                continue
            (module_level if at_module_level else deferred)[(source, target)] += 1
    return module_level, deferred


def _packages_dot(show_deferred: bool) -> str:
    module_level, deferred = _package_edges()
    nodes = sorted({n for edge in (*module_level, *deferred) for n in edge})

    # Layering follows the import-time graph only: a deferred import does
    # not constrain load order, so letting it set the rank would be wrong.
    successors = defaultdict(set)
    for src, dst in module_level:
        successors[src].add(dst)
    components = _strongly_connected(nodes, lambda n: successors.get(n, set()))
    depth = _layer_of(components, list(module_level))
    owner = {node: i for i, comp in enumerate(components) for node in comp}
    cyclic = {node for comp in components if len(comp) > 1 for node in comp}

    by_layer = defaultdict(list)
    for node in nodes:
        by_layer[depth[owner[node]]].append(node)

    legend = "edge label = number of names imported   red = import cycle"
    if show_deferred:
        legend = "solid = at import time   dashed = inside a function   " + legend
    lines = [
        "digraph magnelio_packages {",
        "  rankdir=BT;",
        "  splines=true;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#f7fafc"];',
        '  edge [fontname="Helvetica", fontsize=9];',
        f'  label="Magnelio subpackage dependencies\\n{legend}";',
        "  labelloc=t;",
        "",
    ]
    for node in nodes:
        colour = (
            COLOUR_WARN
            if node in cyclic
            else (COLOUR_INTERNAL if _is_internal(node) else COLOUR_PUBLIC)
        )
        penwidth = 2.5 if node in cyclic else 1.0
        lines.append(
            f"  {_quote(node)} [color={_quote(colour)}, penwidth={penwidth}, "
            f"fontcolor={_quote(colour)}];"
        )
    lines.append("")
    # Same-layer packages share a rank, which is what makes the pipeline
    # readable top to bottom.
    for layer in sorted(by_layer):
        members = " ".join(_quote(n) for n in sorted(by_layer[layer]))
        lines.append(f"  {{ rank=same; {members} }}")
    lines.append("")
    for (src, dst), count in sorted(module_level.items()):
        width = min(1.0 + count / 5.0, 4.0)
        both_ways = (dst, src) in module_level
        colour = COLOUR_WARN if both_ways else "#4a5568"
        lines.append(
            f"  {_quote(src)} -> {_quote(dst)} [label={_quote(str(count))}, "
            f"penwidth={width:.1f}, color={_quote(colour)}];"
        )
    if show_deferred:
        for (src, dst), count in sorted(deferred.items()):
            if (src, dst) in module_level:
                continue
            lines.append(
                f"  {_quote(src)} -> {_quote(dst)} [label={_quote(str(count))}, "
                f'style=dashed, constraint=false, color="#a0aec0", '
                f'fontcolor="#a0aec0"];'
            )
    lines.append("}")
    return "\n".join(lines)


def _packages_stats() -> None:
    module_level, deferred = _package_edges()
    nodes = sorted({n for edge in (*module_level, *deferred) for n in edge})
    only_deferred = {e for e in deferred if e not in module_level}
    successors = defaultdict(set)
    for src, dst in module_level:
        successors[src].add(dst)
    components = _strongly_connected(nodes, lambda n: successors.get(n, set()))
    cycles = [c for c in components if len(c) > 1]

    print(f"{len(nodes)} subpackages")
    print(f"{len(module_level)} module-level edges, {len(only_deferred)} deferred-only edges")
    print(
        f"{len(cycles)} module-level cycle(s): "
        + (", ".join(" <-> ".join(c) for c in cycles) if cycles else "none")
    )
    out = Counter()
    inn = Counter()
    for src, dst in module_level:
        out[src] += 1
        inn[dst] += 1
    print(f"\n{'package':14s} {'out':>5s} {'in':>5s}   (module-level)")
    for node in nodes:
        print(f"{node:14s} {out[node]:5d} {inn[node]:5d}")
    print("\nheaviest module-level edges:")
    for (src, dst), count in module_level.most_common(10):
        print(f"  {src:12s} -> {dst:14s}{count:4d}")


# ─── view: inheritance ──────────────────────────────────────────────────────


def _import_package():
    """Import magnelio and every submodule, so subclasses register.

    Internal modules are imported too: this is a developer's map, and
    leaving out ``_operators`` or ``_backend`` would hide exactly the
    parts that are hardest to learn from the public API.
    """
    import importlib
    import pkgutil

    import magnelio

    for info in pkgutil.walk_packages(magnelio.__path__, "magnelio."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:  # noqa: BLE001 — an optional backend may be absent
            print(f"warning: cannot import {info.name}: {exc}", file=sys.stderr)
    return magnelio


def _resolve_class(dotted: str):
    import importlib

    module_name, _, class_name = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _subclass_edges(root):
    """(edges, classes) below *root*, following __subclasses__ recursively."""
    edges: list[tuple[type, type]] = []
    seen = {root}
    queue = [root]
    while queue:
        parent = queue.pop()
        for child in sorted(parent.__subclasses__(), key=lambda c: c.__name__):
            edges.append((parent, child))
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return edges, seen


def _inheritance_dot(root_path: str) -> str:
    _import_package()
    root = _resolve_class(root_path)
    edges, classes = _subclass_edges(root)

    by_module = defaultdict(list)
    for cls in sorted(classes, key=lambda c: c.__name__):
        by_module[cls.__module__].append(cls)

    lines = [
        "digraph magnelio_inheritance {",
        "  rankdir=BT;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#f7fafc"];',
        f'  label="Inheritance below {root.__name__}\\nblue = public API   grey = internal";',
        "  labelloc=t;",
        "",
    ]
    for index, (module, members) in enumerate(sorted(by_module.items())):
        lines.append(f"  subgraph cluster_{index} {{")
        lines.append(f"    label={_quote(module.replace('magnelio.', ''))};")
        lines.append('    style=dotted; color="#cbd5e0"; fontname="Helvetica"; fontsize=10;')
        for cls in members:
            colour = COLOUR_INTERNAL if _is_internal(cls.__name__) else COLOUR_PUBLIC
            lines.append(
                f"    {_quote(cls.__name__)} [color={_quote(colour)}, fontcolor={_quote(colour)}];"
            )
        lines.append("  }")
    lines.append("")
    for parent, child in edges:
        lines.append(f"  {_quote(child.__name__)} -> {_quote(parent.__name__)};")
    lines.append("}")
    return "\n".join(lines)


def _inheritance_stats(root_path: str) -> None:
    _import_package()
    root = _resolve_class(root_path)
    edges, classes = _subclass_edges(root)
    subclasses = classes - {root}

    depth = {root: 0}
    for parent, child in edges:
        depth[child] = max(depth.get(child, 0), depth.get(parent, 0) + 1)
    internal = [c for c in subclasses if _is_internal(c.__name__)]
    print(f"root: {root.__module__}.{root.__name__}")
    print(f"{len(subclasses)} subclasses, max depth {max(depth.values())}")
    print(f"{len(internal)} internal, {len(subclasses) - len(internal)} public")
    print("\npublic:")
    for cls in sorted(subclasses - set(internal), key=lambda c: c.__name__):
        print(f"  {cls.__name__:22s} [{cls.__module__.replace('magnelio.', '')}]")


# ─── view: composition ──────────────────────────────────────────────────────

#: How many children a class holds, by the field names it recurses into.
ARITY_LABEL = {0: "leaf", 1: "unary", 2: "binary"}


def _child_shape_fields():
    """Map class name -> the ``self`` fields it recurses into.

    A shape delegates to its children by calling ``._occ_shape(scale)``
    on them, so that call site *is* the composition edge.  Two forms
    occur: a direct ``self.<field>._occ_shape(...)``, and a comprehension
    over ``self.<field>`` whose element is called instead.  The field
    types themselves say nothing — they are all annotated ``object``.
    """
    found: dict[str, set[str]] = defaultdict(set)
    classes: set[str] = set()
    for _path, tree in _parse_sources(SRC_ROOT / "geo"):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            classes.add(node.name)
            # Comprehension variables that stand for an element of self.<field>.
            element_source: dict[str, str] = {}
            for inner in ast.walk(node):
                if isinstance(inner, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
                    for generator in inner.generators:
                        iterated = generator.iter
                        if (
                            isinstance(iterated, ast.Attribute)
                            and isinstance(iterated.value, ast.Name)
                            and iterated.value.id == "self"
                            and isinstance(generator.target, ast.Name)
                        ):
                            element_source[generator.target.id] = iterated.attr
            for inner in ast.walk(node):
                if not (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "_occ_shape"
                ):
                    continue
                target = inner.func.value
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    found[node.name].add(target.attr)
                elif isinstance(target, ast.Name) and target.id in element_source:
                    found[node.name].add(element_source[target.id] + "[*]")
    return found, classes


def _arity_of(field_names) -> str:
    """'leaf' / 'unary' / 'binary' / 'n-ary' from the recursed-into fields."""
    if any(field.endswith("[*]") for field in field_names):
        return "n-ary"
    return ARITY_LABEL.get(len(field_names), "n-ary")


def _composition_dot() -> str:
    children, _classes = _child_shape_fields()
    # A class that never recurses is a leaf: it builds OCC geometry itself.
    import contextlib

    leaves: list[str] = []
    with contextlib.suppress(Exception):
        _import_package()
        from magnelio.geo.shape import Shape

        _edges, shape_classes = _subclass_edges(Shape)
        leaves = sorted(
            c.__name__ for c in shape_classes if c.__name__ not in children and c is not Shape
        )

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name in sorted(children):
        groups[_arity_of(children[name])].append((name, ", ".join(sorted(children[name]))))
    for name in leaves:
        groups["leaf"].append((name, ""))

    # One box per arity, one edge per box: fanning every class onto a
    # single Shape node produced a 19-edge comb two thousand pixels tall
    # and said no more than the grouping does.
    order = [
        ("leaf", "leaf — builds its own OCC solid", COLOUR_LEAF),
        ("unary", "unary — wraps one shape", COLOUR_PUBLIC),
        ("binary", "binary — combines two", COLOUR_PUBLIC),
        ("n-ary", "n-ary — takes a sequence", COLOUR_PUBLIC),
    ]
    lines = [
        "digraph magnelio_composition {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fillcolor="#f7fafc", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=10];',
        '  label="How shapes nest: the geometry expression tree\\n'
        "membership follows the ._occ_shape() calls a class makes on its own fields\\n"
        "those fields are annotated 'object', so the child class itself cannot be named\";",
        "  labelloc=t;",
        "",
        f'  "Shape" [shape=ellipse, fillcolor="#edf2f7", color={_quote(COLOUR_PUBLIC)}, '
        f"penwidth=2, fontsize=13];",
        "",
    ]
    # One node per arity, listing its members: as a node each, the 13
    # unary classes stacked into a column two thousand pixels tall and
    # said nothing the grouping does not already say.
    for arity, caption, colour in order:
        members = groups.get(arity)
        if not members:
            continue
        width = max(len(name) for name, _ in members)
        rows = [f"{caption}  ({len(members)})", ""]
        rows += [
            f"{name.ljust(width)}  {field_names}" if field_names else name
            for name, field_names in members
        ]
        label = "\\l".join(rows) + "\\l"
        lines.append(
            f"  {_quote(arity)} [label={_quote_label(label)}, color={_quote(colour)}, "
            f'fontname="Courier", fontsize=10, penwidth=1.5];'
        )
    lines.append("")
    for arity, _caption, colour in order:
        if arity == "leaf" or arity not in groups:
            continue
        count = {"unary": "1 child", "binary": "2 children"}.get(arity, "n children")
        lines.append(
            f'  {_quote(arity)} -> "Shape" [label={_quote(count)}, color={_quote(colour)}, '
            f"fontcolor={_quote(colour)}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _composition_stats() -> None:
    children, _classes = _child_shape_fields()
    print(f"{len(children)} classes recurse into child shapes\n")
    grouped = defaultdict(list)
    for name, field_names in children.items():
        arity = (
            "n-ary"
            if any(f.endswith("[*]") for f in field_names)
            else ARITY_LABEL.get(len(field_names), "n-ary")
        )
        grouped[arity].append((name, sorted(field_names)))
    for arity in ("unary", "binary", "n-ary"):
        if arity not in grouped:
            continue
        print(f"{arity}:")
        for name, field_names in sorted(grouped[arity]):
            print(f"  {name:22s} {', '.join(field_names)}")


# ─── view: classes ──────────────────────────────────────────────────────────


def _package_classes(package: str | None):
    """Map subpackage -> classes defined in it (via the imported package)."""
    import inspect

    _import_package()
    import magnelio

    by_package: dict[str, set] = defaultdict(set)
    for module_name, module in sorted(sys.modules.items()):
        if not module_name.startswith("magnelio.") or module is None:
            continue
        parts = module_name.split(".")
        owner = parts[1] if len(parts) > 2 else "(top)"
        if package is not None and owner != package:
            continue
        for _name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module_name:
                by_package[owner].add(cls)
    for _name, cls in inspect.getmembers(magnelio, inspect.isclass):
        if package in (None, "(top)") and cls.__module__.startswith("magnelio"):
            by_package.setdefault("(top)", set())
    return by_package


def _member_label(cls) -> str:
    """A record-shaped label listing fields and public methods."""
    rows = [cls.__name__]
    if is_dataclass(cls):
        names = [f.name for f in fields(cls) if not f.name.startswith("__")]
        if names:
            rows.append("\\l".join(names) + "\\l")
    methods = sorted(
        name for name, value in vars(cls).items() if not name.startswith("_") and callable(value)
    )
    if methods:
        rows.append("\\l".join(f"{m}()" for m in methods) + "\\l")
    return "{" + "|".join(rows) + "}"


def _classes_dot(package: str | None, with_members: bool) -> str:
    by_package = _package_classes(package)
    known = {cls for members in by_package.values() for cls in members}

    title = f"Classes in magnelio.{package}" if package else "Classes by subpackage"
    lines = [
        "digraph magnelio_classes {",
        "  rankdir=BT;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fillcolor="#f7fafc", fontsize=10];',
        f"  label={_quote(title)};",
        "  labelloc=t;",
        "",
    ]
    for index, (owner, members) in enumerate(sorted(by_package.items())):
        if not members:
            continue
        lines.append(f"  subgraph cluster_{index} {{")
        lines.append(f"    label={_quote(owner)};")
        lines.append('    style=dotted; color="#cbd5e0"; fontname="Helvetica";')
        for cls in sorted(members, key=lambda c: c.__name__):
            colour = COLOUR_INTERNAL if _is_internal(cls.__name__) else COLOUR_PUBLIC
            if with_members:
                lines.append(
                    f"    {_quote(cls.__name__)} [shape=record, style=filled, "
                    f"label={_quote_label(_member_label(cls))}, color={_quote(colour)}];"
                )
            else:
                lines.append(
                    f"    {_quote(cls.__name__)} [color={_quote(colour)}, "
                    f"fontcolor={_quote(colour)}];"
                )
        lines.append("  }")
    lines.append("")
    for cls in sorted(known, key=lambda c: c.__name__):
        for base in cls.__bases__:
            if base in known:
                lines.append(f"  {_quote(cls.__name__)} -> {_quote(base.__name__)};")
    lines.append("}")
    return "\n".join(lines)


def _classes_stats(package: str | None) -> None:
    by_package = _package_classes(package)
    total = sum(len(v) for v in by_package.values())
    print(f"{total} classes in {len(by_package)} subpackage(s)\n")
    for owner, members in sorted(by_package.items()):
        public = [c for c in members if not _is_internal(c.__name__)]
        print(f"{owner:14s} {len(members):3d} classes ({len(public)} public)")


# ─── entry point ────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "view",
        choices=("packages", "inheritance", "composition", "classes"),
        help="which structure view to draw",
    )
    parser.add_argument(
        "--render",
        metavar="FORMAT",
        help="render with dot to this format (svg, png, pdf) instead of printing DOT",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"output directory for --render (default: {DEFAULT_OUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print the underlying numbers as text instead of a graph",
    )
    parser.add_argument(
        "--root",
        default="magnelio.geo.shape.Shape",
        help="inheritance: dotted path of the root class",
    )
    parser.add_argument(
        "--package",
        help="classes: restrict to one subpackage (e.g. geo)",
    )
    parser.add_argument(
        "--members",
        action="store_true",
        help="classes: include fields and public methods",
    )
    parser.add_argument(
        "--deferred",
        action="store_true",
        help=(
            "packages: also draw imports made inside functions (dashed).  Off by "
            "default: Magnelio has 37 such edges and they bury the load-time graph."
        ),
    )
    args = parser.parse_args(argv)

    if args.view == "packages":
        if args.stats:
            _packages_stats()
            return 0
        return _emit(_packages_dot(args.deferred), "packages", args)
    if args.view == "inheritance":
        if args.stats:
            _inheritance_stats(args.root)
            return 0
        return _emit(_inheritance_dot(args.root), "inheritance", args)
    if args.view == "composition":
        if args.stats:
            _composition_stats()
            return 0
        return _emit(_composition_dot(), "composition", args)
    if args.stats:
        _classes_stats(args.package)
        return 0
    name = f"classes_{args.package}" if args.package else "classes"
    return _emit(_classes_dot(args.package, args.members), name, args)


if __name__ == "__main__":
    sys.exit(main())
