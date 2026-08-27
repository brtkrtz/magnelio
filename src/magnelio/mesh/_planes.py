"""Provenance of the grid planes a mesh was generated from.

Design: DD-200.  ``Mesh.from_geometry`` derives every grid plane from
a rule — a material face, a bounding-box extent, a thin sheet, a wire
vertex, a symmetry face, a user-forced position, a geometry edge — and
merges, snaps, floors and drops them before the axis lines are graded.
This module keeps that bookkeeping on the mesh: which plane came from
which rule and which shape, which requested planes never made it into
the grid and why.

The attribution is *post hoc* by position (:func:`attribute_planes`):
the merge helpers of the mesher stay untouched, and the raw
``(position, source)`` entries recorded next to the raw plane lists are
matched to the merged outcome afterwards.  Every merge stage moves an
entry by at most the clustering tolerance, so a window of twice that
tolerance finds the outcome; whatever finds none is reported as
*unplaced* — the invariant that catches a silent drop.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

_KINDS = ("face", "extent", "edge", "sheet", "wire", "symmetry", "forced")
# Kinds a WP-M3 floor merge may absorb into a neighbouring plane
# (material planes); an edge plane below the floor is *dropped* instead.
_ABSORBABLE = frozenset({"face", "extent", "sheet", "wire"})
_AXES = ("x", "y", "z")

__all__ = [
    "PlaneSource",
    "PlaneRecord",
    "GridPlanes",
    "attribute_planes",
    "shape_label",
]


def shape_label(index: int, shape) -> str:
    """``"#<index> <Class>(<material>)"``, plus ``'<name>'`` when the shape has one."""
    mat = getattr(shape, "material", None)
    mat_name = getattr(mat, "name", None)
    if mat_name is None and mat is not None:
        mat_name = str(mat)
    text = f"#{index} {type(shape).__name__.lstrip('_')}({mat_name or ''})"
    name = getattr(shape, "name", None)
    if name:
        text += f" {name!r}"
    return text


@dataclass(frozen=True, order=True)
class PlaneSource:
    """One reason for a grid plane.

    Attributes
    ----------
    kind : str
        ``"face"`` (analytic material face), ``"extent"`` (bounding-box
        extent of a shape), ``"edge"`` (a B-rep edge lying flat in the
        plane — a chamfer or fillet onset, a loft section), ``"sheet"``
        (the grid plane of a thin metallisation), ``"wire"`` (a
        thin-wire vertex), ``"symmetry"`` (a symmetry face with a
        position) or ``"forced"`` (``MeshControl.forced_planes``).
    shape : int or None
        Index of the contributing shape in the model's shape list;
        ``None`` for forced and symmetry planes.
    label : str
        Human-readable shape label (``"#1 Cylinder(pec)"``), or the face
        name of a symmetry plane (``"xmin"``).
    """

    kind: str
    shape: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown plane source kind {self.kind!r}; expected one of {_KINDS}")

    def __str__(self) -> str:
        return f"{self.kind} {self.label}" if self.label else self.kind

    @classmethod
    def parse(cls, text: str) -> PlaneSource:
        """Inverse of :meth:`__str__`."""
        kind, _, rest = text.strip().partition(" ")
        if kind == "symmetry":
            return cls(kind, None, rest)
        m = re.match(r"#(\d+)\b", rest)
        return cls(kind, int(m.group(1)) if m else None, rest)


@dataclass(frozen=True)
class PlaneRecord:
    """One grid plane (or one plane that did not make it into the grid).

    Attributes
    ----------
    position : float
        The merged plane coordinate [m] — the geometry position the
        grid line stands for, before absorber cells are appended and
        before a PMC face pulls its end node inwards.
    sources : tuple of PlaneSource
        Every rule and shape that asked for this plane, sorted.
    node : int or None
        Index of the plane's node in ``mesh.grid.<axis>``; ``None`` for
        planes that are not in the grid (dropped, absorbed, unplaced).
    singular : bool
        The plane holds a conductor edge with a field singularity and
        grades from the refined fine size.
    domain_end : bool
        First or last plane of the axis (the domain bounding box).
    h_fine : float or None
        The fine cell size [m] the grading starts from at this plane.
    moved_to : float or None
        Where the node actually sits when a PMC face pulled it inside
        the bounding box; ``None`` when the node is at ``position``.
    gap : float or None
        For a dropped edge plane: the cell it would have created [m].
    """

    position: float
    sources: tuple[PlaneSource, ...] = ()
    node: int | None = None
    singular: bool = False
    domain_end: bool = False
    h_fine: float | None = None
    moved_to: float | None = None
    gap: float | None = None

    @property
    def kinds(self) -> frozenset[str]:
        """The set of source kinds behind this plane."""
        return frozenset(s.kind for s in self.sources)


@dataclass(frozen=True)
class GridPlanes:
    """Provenance of every grid plane of a mesh, per axis.

    Returned as ``mesh.planes`` by :meth:`Mesh.from_geometry`
    (``None`` on meshes built from a grid or loaded from an older
    store).  ``print(mesh.planes)`` gives the report; :meth:`as_dict`
    a JSON-serialisable form used by the store and by the example-model
    pins of the test suite.

    Attributes
    ----------
    x, y, z : tuple of PlaneRecord
        The planes of each axis in ascending order — the interval
        boundaries of the graded grid.
    h_bulk : dict of str to tuple of float
        The bulk cell size [m] of every interval of each axis (one entry
        fewer than planes).
    dropped : dict of str to tuple of PlaneRecord
        Edge planes below the edge floor, reported by the mesher's
        warning; ``gap`` holds the cell they would have created.
    absorbed : dict of str to tuple of PlaneRecord
        Material planes merged into a neighbour by the hard
        ``min_cell_size`` floor.
    unplaced : dict of str to tuple of PlaneRecord
        Requested planes that match no outcome.  Empty in a consistent
        build; a non-empty list points at a rule that discards a plane
        without recording it.
    n_nodes : dict of str to int
        Node count of each axis of the final grid (absorber cells
        included).
    pml_cells : dict of str to int
        Absorber cells per domain face, as ``mesh.pml_cells``.
    feature_gap : float
        The clustering tolerance [m] the planes were merged with.
    """

    x: tuple[PlaneRecord, ...] = ()
    y: tuple[PlaneRecord, ...] = ()
    z: tuple[PlaneRecord, ...] = ()
    h_bulk: dict[str, tuple[float, ...]] = field(default_factory=dict)
    dropped: dict[str, tuple[PlaneRecord, ...]] = field(default_factory=dict)
    absorbed: dict[str, tuple[PlaneRecord, ...]] = field(default_factory=dict)
    unplaced: dict[str, tuple[PlaneRecord, ...]] = field(default_factory=dict)
    n_nodes: dict[str, int] = field(default_factory=dict)
    pml_cells: dict[str, int] = field(default_factory=dict)
    feature_gap: float = 0.0

    def __post_init__(self) -> None:
        # Every per-axis mapping lists all three axes, so two records
        # built from different sources compare equal.
        for name in ("h_bulk", "dropped", "absorbed", "unplaced"):
            d = dict(getattr(self, name))
            for axis in _AXES:
                d.setdefault(axis, ())
                d[axis] = tuple(d[axis])
            object.__setattr__(self, name, d)
        object.__setattr__(self, "n_nodes", {a: int(self.n_nodes.get(a, 0)) for a in _AXES})
        object.__setattr__(self, "pml_cells", dict(sorted(self.pml_cells.items())))

    # -- access -----------------------------------------------------------

    def records(self, axis: str) -> tuple[PlaneRecord, ...]:
        """The planes of one axis (``"x"``, ``"y"`` or ``"z"``)."""
        if axis not in _AXES:
            raise ValueError(f"axis must be 'x', 'y' or 'z'; got {axis!r}")
        return getattr(self, axis)

    # -- serialisation ------------------------------------------------------

    def _decimals(self) -> int:
        if self.feature_gap <= 0.0:
            return 12
        return max(0, 3 - math.floor(math.log10(self.feature_gap)))

    def as_dict(self, *, rounded: bool = True) -> dict:
        """JSON-serialisable form (deterministic key order).

        Parameters
        ----------
        rounded : bool, default True
            Round positions to three decimals below the feature gap and
            cell sizes to four significant digits — the form the
            example-model pins are written in.  ``False`` keeps every
            float verbatim (the store).
        """
        dec = self._decimals()

        def _pos(v):
            if v is None:
                return None
            return round(float(v), dec) if rounded else float(v)

        def _len(v):
            if v is None:
                return None
            return float(f"{v:.4g}") if rounded else float(v)

        def _rec(r: PlaneRecord, *, grid: bool) -> dict:
            d = {
                "position": _pos(r.position),
                "sources": [str(s) for s in r.sources],
            }
            if grid:
                d.update(
                    node=r.node,
                    singular=r.singular,
                    domain_end=r.domain_end,
                    h_fine=_len(r.h_fine),
                    moved_to=_pos(r.moved_to),
                )
            elif r.gap is not None:
                d["gap"] = _len(r.gap)
            return d

        axes = {}
        for axis in _AXES:
            axes[axis] = {
                "planes": [_rec(r, grid=True) for r in self.records(axis)],
                "h_bulk": [_len(h) for h in self.h_bulk.get(axis, ())],
                "dropped": [_rec(r, grid=False) for r in self.dropped.get(axis, ())],
                "absorbed": [_rec(r, grid=False) for r in self.absorbed.get(axis, ())],
                "unplaced": [_rec(r, grid=False) for r in self.unplaced.get(axis, ())],
            }
        return {
            "schema": 1,
            "feature_gap": float(f"{self.feature_gap:.6g}") if rounded else self.feature_gap,
            "n_nodes": {axis: int(self.n_nodes.get(axis, 0)) for axis in _AXES},
            "pml_cells": {k: int(v) for k, v in sorted(self.pml_cells.items())},
            "axes": axes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GridPlanes:
        """Inverse of :meth:`as_dict`."""

        def _rec(e: dict) -> PlaneRecord:
            return PlaneRecord(
                position=float(e["position"]),
                sources=tuple(sorted((PlaneSource.parse(s) for s in e["sources"]), key=str)),
                node=e.get("node"),
                singular=bool(e.get("singular", False)),
                domain_end=bool(e.get("domain_end", False)),
                h_fine=e.get("h_fine"),
                moved_to=e.get("moved_to"),
                gap=e.get("gap"),
            )

        axes = d.get("axes", {})
        kw: dict = {"h_bulk": {}, "dropped": {}, "absorbed": {}, "unplaced": {}}
        for axis in _AXES:
            a = axes.get(axis, {})
            kw[axis] = tuple(_rec(e) for e in a.get("planes", ()))
            kw["h_bulk"][axis] = tuple(float(h) for h in a.get("h_bulk", ()))
            for key in ("dropped", "absorbed", "unplaced"):
                kw[key][axis] = tuple(_rec(e) for e in a.get(key, ()))
        return cls(
            n_nodes={k: int(v) for k, v in d.get("n_nodes", {}).items()},
            pml_cells={k: int(v) for k, v in d.get("pml_cells", {}).items()},
            feature_gap=float(d.get("feature_gap", 0.0)),
            **kw,
        )

    # -- report -------------------------------------------------------------

    def summary(self, scale_mm: bool = True) -> str:
        """Multi-line human-readable report (used by ``str()``).

        Parameters
        ----------
        scale_mm : bool, default True
            Print lengths in millimetres; ``False`` prints metres.
        """
        unit = "mm" if scale_mm else "m"
        f = 1e3 if scale_mm else 1.0

        def _len(v: float, digits: int = 6) -> str:
            return f"{v * f:.{digits}g} {unit}"

        gap_text = f"{self.feature_gap * 1e6:.3g} µm" if scale_mm else _len(self.feature_gap, 3)
        nodes = ", ".join(f"{a} {self.n_nodes.get(a, 0)}" for a in _AXES)
        head = f"Grid planes — feature gap {gap_text}; nodes {nodes}"
        if self.pml_cells:
            head += "; PML " + ", ".join(f"{k} {v}" for k, v in sorted(self.pml_cells.items()))
        lines = [head]
        for axis in _AXES:
            recs = self.records(axis)
            h_bulk = self.h_bulk.get(axis, ())
            axis_line = f"{axis}: {len(recs)} planes"
            details = []
            if h_bulk:
                lo, hi = min(h_bulk), max(h_bulk)
                details.append(
                    f"h_bulk {_len(lo, 4)}" if lo == hi else f"h_bulk {_len(lo, 4)} … {_len(hi, 4)}"
                )
            fines = [r.h_fine for r in recs if r.h_fine is not None]
            h_fine_axis = max(fines) if fines else None
            if h_fine_axis is not None:
                details.append(f"h_fine {_len(h_fine_axis, 4)}")
            if details:
                axis_line += "  (" + ", ".join(details) + ")"
            lines.append(axis_line)
            pos_texts = [_len(r.position) for r in recs]
            width = max((len(t) for t in pos_texts), default=0)
            for i, (r, pos_text) in enumerate(zip(recs, pos_texts)):
                row = f"  [{i}]  {pos_text:>{width}s}   {_format_sources(r.sources)}"
                flags = []
                if r.domain_end:
                    flags.append("domain end")
                if r.singular:
                    flags.append("singular")
                if r.h_fine is not None and h_fine_axis is not None and r.h_fine < h_fine_axis:
                    flags.append(f"h_fine {_len(r.h_fine, 4)}")
                if r.moved_to is not None:
                    flags.append(f"-> node at {_len(r.moved_to)}")
                if flags:
                    row += "   " + ", ".join(flags)
                lines.append(row)
            for r in self.dropped.get(axis, ()):
                srcs = _format_sources(r.sources)
                cell = f"  (would make a {_len(r.gap, 4)} cell)" if r.gap is not None else ""
                lines.append(f"  dropped (edge floor): {_len(r.position)}  {srcs}{cell}")
            for r in self.absorbed.get(axis, ()):
                srcs = _format_sources(r.sources)
                lines.append(f"  absorbed (min_cell_size): {_len(r.position)}  {srcs}")
            for r in self.unplaced.get(axis, ()):
                srcs = _format_sources(r.sources)
                lines.append(f"  unplaced: {_len(r.position)}  {srcs}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def _sorted_sources(sources) -> tuple[PlaneSource, ...]:
    return tuple(sorted(set(sources), key=str))


_KIND_ORDER = {
    k: i for i, k in enumerate(("face", "sheet", "wire", "symmetry", "forced", "edge", "extent"))
}


def _format_sources(sources: tuple[PlaneSource, ...]) -> str:
    """``"face #0 Brick(air) #1 Cylinder(pec); extent #0 #1; edge #0"``.

    Grouped by kind; a shape's full label appears at its first mention
    in the row, later mentions are the bare index.
    """
    if not sources:
        return "(no source)"
    by_kind: dict[str, list[PlaneSource]] = {}
    for src in sorted(sources, key=lambda s: (_KIND_ORDER.get(s.kind, 99), str(s))):
        by_kind.setdefault(src.kind, []).append(src)
    seen: set[int] = set()
    parts = []
    for kind, group in by_kind.items():
        items = []
        for src in group:
            if src.shape is None:
                if src.label:
                    items.append(src.label)
                continue
            if src.shape in seen:
                items.append(f"#{src.shape}")
            else:
                seen.add(src.shape)
                items.append(src.label or f"#{src.shape}")
        parts.append(f"{kind} {' '.join(items)}".rstrip())
    return "; ".join(parts)


def attribute_planes(
    axis_planes: dict[str, list[float]],
    axis_is_singular: dict[str, list[bool]],
    plane_sources: dict[str, list[tuple[float, PlaneSource]]],
    dropped_edges: dict[str, list[tuple[float, float]]],
    absorbed_planes: dict[str, list[float]],
    feature_gap: float,
    window_factor: float = 2.0,
) -> tuple[
    dict[str, list[PlaneRecord]],
    dict[str, list[PlaneRecord]],
    dict[str, list[PlaneRecord]],
    dict[str, list[PlaneRecord]],
]:
    """Match the raw ``(position, source)`` entries to the merged outcome.

    Every raw entry is assigned to the nearest candidate within
    ``window_factor * feature_gap``: a final grid plane, a dropped edge
    plane (``edge`` sources only) or a floor-absorbed material plane
    (material-class sources only).  Entries that find no candidate are
    collected as *unplaced*.

    Returns
    -------
    grid, dropped, absorbed, unplaced : dict of str to list of PlaneRecord
        Per axis.  ``grid`` records carry ``position``, ``sources``,
        ``singular`` and ``domain_end``; the caller adds ``node``,
        ``h_fine`` and ``moved_to`` once the axis lines exist.
    """
    window = window_factor * feature_gap
    grid: dict[str, list[PlaneRecord]] = {}
    dropped: dict[str, list[PlaneRecord]] = {}
    absorbed: dict[str, list[PlaneRecord]] = {}
    unplaced: dict[str, list[PlaneRecord]] = {}
    for axis in _AXES:
        finals = [float(p) for p in axis_planes.get(axis, [])]
        drops = [(float(p), float(g)) for p, g in dropped_edges.get(axis, [])]
        absorbs = [float(p) for p in absorbed_planes.get(axis, [])]
        # candidate table: (position, bucket, index)
        cands: list[tuple[float, str, int]] = [(p, "grid", i) for i, p in enumerate(finals)]
        cands += [(p, "dropped", i) for i, (p, _g) in enumerate(drops)]
        cands += [(p, "absorbed", i) for i, p in enumerate(absorbs)]
        src_grid: list[set] = [set() for _ in finals]
        src_drop: list[set] = [set() for _ in drops]
        src_abs: list[set] = [set() for _ in absorbs]
        loose: list[tuple[float, PlaneSource]] = []
        for pos, src in plane_sources.get(axis, []):
            pos = float(pos)
            best = None
            best_d = math.inf
            for cand_pos, bucket, idx in cands:
                if bucket == "dropped" and src.kind != "edge":
                    continue
                if bucket == "absorbed" and src.kind not in _ABSORBABLE:
                    continue
                d = abs(pos - cand_pos)
                if d < best_d:
                    best_d, best = d, (bucket, idx)
            if best is None or best_d > window:
                loose.append((pos, src))
                continue
            bucket, idx = best
            {"grid": src_grid, "dropped": src_drop, "absorbed": src_abs}[bucket][idx].add(src)
        singular = list(axis_is_singular.get(axis, [])) or [False] * len(finals)
        grid[axis] = [
            PlaneRecord(
                position=p,
                sources=_sorted_sources(src_grid[i]),
                singular=bool(singular[i]) if i < len(singular) else False,
                domain_end=(i == 0 or i == len(finals) - 1),
            )
            for i, p in enumerate(finals)
        ]
        dropped[axis] = [
            PlaneRecord(position=p, sources=_sorted_sources(src_drop[i]), gap=g)
            for i, (p, g) in enumerate(drops)
        ]
        absorbed[axis] = [
            PlaneRecord(position=p, sources=_sorted_sources(src_abs[i]))
            for i, p in enumerate(absorbs)
        ]
        # Unplaced entries within the window of each other are one plane.
        loose.sort(key=lambda e: e[0])
        groups: list[list[tuple[float, PlaneSource]]] = []
        for pos, src in loose:
            if groups and abs(pos - groups[-1][0][0]) <= window:
                groups[-1].append((pos, src))
            else:
                groups.append([(pos, src)])
        unplaced[axis] = [
            PlaneRecord(
                position=sum(p for p, _s in g) / len(g),
                sources=_sorted_sources(s for _p, s in g),
            )
            for g in groups
        ]
    return grid, dropped, absorbed, unplaced


def with_node_data(
    record: PlaneRecord,
    *,
    node: int | None = None,
    h_fine: float | None = None,
    moved_to: float | None = None,
) -> PlaneRecord:
    """Copy *record* with the grid-side fields filled in."""
    return replace(record, node=node, h_fine=h_fine, moved_to=moved_to)
