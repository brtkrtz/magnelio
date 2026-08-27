"""Fuse prisms in their plane, and everything else only where it interferes.

The kernel's general fuser is not superlinear in the number of operands
but in their *interference*: a corporate feed network of 443 coplanar,
overlapping strips fuses in 16 s where the same 443 strips moved apart
fuse in 0.45 s.  Coplanar overlap is the fuser's worst case — every
split of a cap face has to be matched against every other cap piece in
the same plane — and at the same time the case a two-dimensional
Boolean solves trivially.  The board importer has always fused its
copper in the plane and extruded once; this module gives the same
route to every union the library runs.

Given the operands of a union, :func:`fuse_shapes`

1. classifies each operand as a *prism* along an axis when its B-Rep
   has planar faces normal to that axis on exactly two levels and every
   other face is ruled along it (planes containing the axis, cylinders
   about it) — true of bricks, cylinders, extruded profiles and
   imported plates alike, false of spheres, cones, chamfers and steps;
2. groups the prisms by (axis, interval), fuses the *bottom caps* of
   each group in the plane — bounding-box clusters, one planar fuse per
   cluster, seams removed — and raises each fused face once;
3. fuses what is left in space, but only within clusters of
   interfering bounding boxes; clusters that touch nothing are kept as
   they are.

The result is the same point set the general fuser would return, with
fewer faces: the seams a fuse leaves between coplanar operands describe
nothing and are gone.  Measured on the benchmark's patch arrays
(``benchmarks/bench_mesh_build.py``): 8 × 8 with feed network, 443
strips, 16.1 s → 0.74 s and 6 924 → 640 faces at the same volume to
nine digits.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# Direction cosines closer than this to 0 or 1 count as exactly
# perpendicular or parallel — the kernel builds axis-aligned primitives
# with exact unit vectors, so anything looser would admit real tilts.
_COSINE_TOLERANCE = 1e-12

# Two prism intervals count as the same when their levels agree to the
# kernel's own confusion tolerance (``Precision::Confusion``, 1e-7 in
# model units): faces that far apart are coincident to the fuser anyway.
_LEVEL_TOLERANCE = 1e-7

_AXES = (2, 0, 1)  # order of preference on a tie: z first, then x, y


def cluster_boxes(boxes: np.ndarray, tolerance: float) -> list[list[int]]:
    """Group boxes whose extents touch, transitively, by a sweep over the first axis.

    Parameters
    ----------
    boxes : ndarray, shape (n, 2 d)
        ``[lo_0, …, lo_{d-1}, hi_0, …, hi_{d-1}]`` per row, any ``d``.
    tolerance : float
        Boxes this far apart still count as touching.

    Returns
    -------
    list of list of int
        Row indices, one list per connected group.
    """
    n = len(boxes)
    if n == 0:
        return []
    d = boxes.shape[1] // 2
    lo, hi = boxes[:, :d], boxes[:, d:]
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: lo[i, 0])
    active: list[int] = []
    for i in order:
        active = [j for j in active if hi[j, 0] >= lo[i, 0] - tolerance]
        for j in active:
            reach = np.all(hi[j, 1:] >= lo[i, 1:] - tolerance)
            if reach and np.all(lo[j, 1:] <= hi[i, 1:] + tolerance):
                a, b = root(i), root(j)
                if a != b:
                    parent[b] = a
        active.append(i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _faces_of(shape) -> list:
    from OCC.Core.TopAbs import TopAbs_FACE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)  # pyright: ignore[reportArgumentType]
    while explorer.More():
        faces.append(topods.Face(explorer.Current()))
        explorer.Next()
    return faces


def _bounding_box(shape) -> np.ndarray:
    from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
    from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415

    box = Bnd_Box()
    brepbndlib.Add(shape, box, False)  # geometry only, no triangulation
    if box.IsVoid():  # an empty shape — a Boolean that removed everything
        return np.full(6, np.nan)
    return np.array(box.Get(), dtype=float)


def _compound(shapes: Sequence):
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound  # noqa: PLC0415

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    return compound


def prism_candidates(shape) -> dict[int, tuple[float, float, list]]:
    """Axes along which *shape* is a prism, with its interval and bottom caps.

    Returns ``{axis: (low, high, bottom_faces)}`` — empty for anything
    that is not a prism along any axis.  A brick qualifies on all three
    axes, a cylinder on its own axis only.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane  # noqa: PLC0415

    levels: dict[int, dict[float, list]] = {k: {} for k in range(3)}
    alive = {0, 1, 2}
    for face in _faces_of(shape):
        adaptor = BRepAdaptor_Surface(face)
        kind = adaptor.GetType()
        if kind == GeomAbs_Plane:
            plane = adaptor.Plane()
            n = plane.Axis().Direction()
            normal = (n.X(), n.Y(), n.Z())
            p = plane.Location()
            point = (p.X(), p.Y(), p.Z())
            for k in tuple(alive):
                c = abs(normal[k])
                if c > 1.0 - _COSINE_TOLERANCE:
                    levels[k].setdefault(_snap(point[k], levels[k]), []).append(face)
                elif c > _COSINE_TOLERANCE:
                    alive.discard(k)
        elif kind == GeomAbs_Cylinder:
            d = adaptor.Cylinder().Axis().Direction()
            direction = (d.X(), d.Y(), d.Z())
            for k in tuple(alive):
                if abs(direction[k]) < 1.0 - _COSINE_TOLERANCE:
                    alive.discard(k)
        else:
            return {}
        if not alive:
            return {}

    out = {}
    for k in alive:
        if len(levels[k]) != 2:
            continue
        low, high = sorted(levels[k])
        out[k] = (low, high, levels[k][low])
    return out


def _snap(value: float, known: dict[float, list]) -> float:
    for key in known:
        if abs(key - value) <= _LEVEL_TOLERANCE:
            return key
    return value


def _same_interval(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= _LEVEL_TOLERANCE and abs(a[1] - b[1]) <= _LEVEL_TOLERANCE


def _interval_groups(intervals: list[tuple[float, float]]) -> list[list[int]]:
    """Indices of intervals equal to the level tolerance, sorted once."""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and _same_interval(intervals[groups[-1][-1]], intervals[i]):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _choose_axis(candidates: list[dict]) -> int | None:
    """The axis on which the most operands share an interval with another."""
    best_axis, best_score = None, 0
    for k in _AXES:
        intervals = [c[k][:2] for c in candidates if k in c]
        score = sum(len(g) for g in _interval_groups(intervals) if len(g) > 1)
        if score > best_score:
            best_axis, best_score = k, score
    return best_axis


def _oriented_solid(shape):
    """A prism raised from a face inherits the face's orientation; fix the sign."""
    from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
    from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return shape.Reversed() if props.Mass() < 0.0 else shape


def _fuse_group_in_plane(
    members: list[tuple[object, list]],
    axis: int,
    low: float,
    high: float,
    fuse_faces: Callable,
    extrude: Callable,
):
    """Fuse one (axis, interval) group of prisms through their bottom caps.

    Members whose caps touch no other member's are returned as they
    are — no Boolean, no re-raising, bit-identical to their input.
    """
    from OCC.Core.TopAbs import TopAbs_FORWARD  # noqa: PLC0415

    in_plane = [k for k in range(3) if k != axis]
    caps = []  # (member index, face)
    for index, (_, faces) in enumerate(members):
        caps.extend((index, face) for face in faces)
    columns = [*in_plane, *(3 + k for k in in_plane)]
    boxes = np.array([_bounding_box(face)[columns] for _, face in caps])
    clusters = cluster_boxes(boxes, _LEVEL_TOLERANCE)
    owners = [{caps[i][0] for i in cluster} for cluster in clusters]
    # An operand meets another through any one of its caps; once it
    # does, all of its caps are re-raised so that nothing of it is lost.
    touched: set[int] = set()
    for owner_set in owners:
        if len(owner_set) > 1:
            touched |= owner_set
    parts = []
    direction = [0.0, 0.0, 0.0]
    direction[axis] = high - low
    for cluster, owner_set in zip(clusters, owners):
        if not owner_set & touched:
            continue
        faces = [caps[i][1].Oriented(TopAbs_FORWARD) for i in cluster]
        fused = fuse_faces(faces) if len(faces) > 1 else faces[0]
        parts.extend(_oriented_solid(extrude(face, tuple(direction))) for face in _faces_of(fused))
    parts.extend(members[i][0] for i in range(len(members)) if i not in touched)
    return parts


def fuse_shapes(shapes: list, fuse: Callable, fuse_faces: Callable, extrude: Callable):
    """Union of *shapes* — prisms in their plane, the rest where it interferes.

    Parameters
    ----------
    shapes : list of TopoDS_Shape
        Two or more operands.
    fuse : callable(list) -> TopoDS_Shape
        The general N-ary fuse, applied to interfering clusters of what
        the planar route did not absorb.
    fuse_faces : callable(list) -> TopoDS_Shape
        Fuses coplanar faces and removes the seams between them.
    extrude : callable(face, (dx, dy, dz)) -> TopoDS_Shape
        Raises a face into a solid.
    """
    candidates = [prism_candidates(s) for s in shapes]
    axis = _choose_axis(candidates)
    parts: list = []
    if axis is None:
        parts = list(shapes)
    else:
        prisms = [i for i, c in enumerate(candidates) if axis in c]
        parts.extend(shapes[i] for i in range(len(shapes)) if axis not in candidates[i])
        for group in _interval_groups([candidates[i][axis][:2] for i in prisms]):
            members = [(shapes[prisms[g]], candidates[prisms[g]][axis][2]) for g in group]
            if len(members) == 1:
                parts.append(members[0][0])
            else:
                low, high = candidates[prisms[group[0]]][axis][:2]
                parts.extend(_fuse_group_in_plane(members, axis, low, high, fuse_faces, extrude))

    boxes = [_bounding_box(p) for p in parts]
    kept = [i for i, b in enumerate(boxes) if not np.isnan(b[0])]
    if not kept:
        return _compound([])
    if len(kept) == 1:
        return parts[kept[0]]
    merged = []
    for cluster in cluster_boxes(np.array([boxes[i] for i in kept]), _LEVEL_TOLERANCE):
        members = [parts[kept[i]] for i in sorted(cluster)]
        merged.append(members[0] if len(members) == 1 else fuse(members))
    return merged[0] if len(merged) == 1 else _compound(merged)
