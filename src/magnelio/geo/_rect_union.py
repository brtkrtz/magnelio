"""Union of rectilinear coplanar faces by coordinate compression.

The kernel's fuse of many coplanar faces is superlinear in their
interference (443 caps 0.7 s, 1 787 caps 13 s flat, 2.6 s up the
bisection tree of :func:`magnelio.geo._prism_fuse.fuse_faces_tree`).
When every face is a straight, axis-aligned polygon — bricks, traces,
pads, the whole of a rectilinear layout — the union is a problem on a
grid: collect the distinct ``u`` and ``v`` coordinates of all vertices,
fill the cells of that grid by the winding number of the rings, and
walk the boundary of what is filled.  Every vertex of the result is a
vertex coordinate of the input; the point set is the kernel's, and the
1 787 caps take 0.2 s.

Two rings meeting at a corner need a rule the grid does not carry by
itself.  At such a node the boundary walk (interior on the left) has
two ways out; the choice decides whether two pieces touching at a
corner become two faces (the kernel's answer) or one self-touching
ring (invalid), and whether two holes touching at a corner stay two
wires (the kernel's answer) or merge.  Both follow from the connected
components of the filled cells: the two filled cells at the node are
of the *same* component exactly when the two empty ones are not — then
the walk keeps the empty cell on its right (turns right), otherwise it
keeps the filled cell on its left (turns left).
"""

from __future__ import annotations

import numpy as np

from magnelio.geo._polygon_clip import njit

# Direction codes of the boundary walk: +u, +v, -u, -v.
_DIRECTIONS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)


def rectilinear_rings(faces: list, tolerance: float):
    """The rings of *faces* when all are straight and axis-aligned, else ``None``.

    Returns ``(axis, level, rings)`` — the rings as ``(n, 2)`` arrays of
    ``(u, v)`` vertices in xyz order of the two in-plane axes, every
    face's outer ring counter-clockwise and its holes clockwise, so that
    the winding number of all rings together is non-zero exactly on the
    union.
    """
    from magnelio.geo._occ_backend import _planar_row  # noqa: PLC0415

    axis = level = None
    rings: list[np.ndarray] = []
    for face in faces:
        row = _planar_row(face)
        if row is None:
            return None
        face_axis, face_level, _, verts, offsets = row
        if axis is None:
            axis, level = face_axis, face_level
        elif face_axis != axis:
            return None
        face_rings = [verts[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)]
        areas = []
        for ring in face_rings:
            step = np.roll(ring, -1, axis=0) - ring
            if not np.all((np.abs(step[:, 0]) <= tolerance) | (np.abs(step[:, 1]) <= tolerance)):
                return None
            areas.append(_signed_area(ring))
        # The outer ring is the largest by magnitude; orient the face so
        # that it runs counter-clockwise and its holes clockwise.
        if areas[int(np.argmax(np.abs(areas)))] < 0.0:
            face_rings = [ring[::-1] for ring in face_rings]
        rings.extend(face_rings)
    if not rings:
        return None
    return axis, level, rings


def _signed_area(ring: np.ndarray) -> float:
    u, v = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))


def _compress(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Distinct coordinates (the smallest of each run within *tolerance*) and each value's index."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    first = np.ones(len(values), dtype=bool)
    first[1:] = np.diff(ordered) > tolerance
    index = np.empty(len(values), dtype=np.int64)
    index[order] = np.cumsum(first) - 1
    return ordered[first], index


def rectilinear_union(rings: list[np.ndarray], tolerance: float) -> list[tuple[np.ndarray, list]]:
    """Union of rectilinear *rings* as ``(outer, holes)`` loops of ``(u, v)`` vertices.

    Parameters
    ----------
    rings : list of ndarray
        ``(n, 2)`` rectilinear rings, outers counter-clockwise, holes
        clockwise (:func:`rectilinear_rings`).
    tolerance : float
        Coordinates this close are one grid line.

    Returns
    -------
    list of (ndarray, list of ndarray)
        One entry per connected piece of the union: its outer ring
        (counter-clockwise, collinear vertices dropped) and its holes
        (clockwise).  Pieces meeting at a corner are separate entries;
        holes meeting at a corner are separate rings sharing a vertex.
    """
    from scipy.ndimage import label  # noqa: PLC0415

    counts = np.array([len(r) for r in rings])
    verts = np.vstack(rings)
    us, iu = _compress(verts[:, 0], tolerance)
    vs, iv = _compress(verts[:, 1], tolerance)
    nu, nv = len(us) - 1, len(vs) - 1
    if nu < 1 or nv < 1:
        return []
    # Successor of every vertex within its ring.
    offsets = np.concatenate([[0], np.cumsum(counts)])
    nxt = np.arange(1, len(verts) + 1)
    nxt[offsets[1:] - 1] = offsets[:-1]
    # Winding number per cell: every edge along v toggles the cells on
    # its +u side by its direction — accumulated at the edge's line and
    # summed along v then along u.
    along_v = (iu == iu[nxt]) & (iv != iv[nxt])
    i_line = iu[along_v]
    j0, j1 = iv[along_v], iv[nxt][along_v]
    sign = np.where(j1 > j0, 1, -1).astype(np.int32)
    acc = np.zeros((nu + 1, nv + 1), dtype=np.int32)
    np.add.at(acc, (i_line, np.minimum(j0, j1)), sign)
    np.add.at(acc, (i_line, np.maximum(j0, j1)), -sign)
    winding = np.cumsum(np.cumsum(acc, axis=1), axis=0)[:nu, :nv]
    labels, n_pieces = label(winding != 0)
    if n_pieces == 0:
        return []
    pad = np.zeros((nu + 2, nv + 2), dtype=np.int32)
    pad[1:-1, 1:-1] = labels

    # Directed boundary edges with the filled cell on the left, as
    # (start node, direction code, label of that left cell).
    left, right = pad[:-1, 1:-1], pad[1:, 1:-1]  # (nu + 1, nv): cells beside a v-edge
    below, above = pad[1:-1, :-1], pad[1:-1, 1:]  # (nu, nv + 1): cells beside a u-edge
    starts, codes, owners = [], [], []
    for mask, code, shift, owner in (
        ((left != 0) & (right == 0), 1, (0, 0), left),  # +v along the left cell
        ((left == 0) & (right != 0), 3, (0, 1), right),  # -v along the right cell
        ((below != 0) & (above == 0), 2, (1, 0), below),  # -u along the cell below
        ((below == 0) & (above != 0), 0, (0, 0), above),  # +u along the cell above
    ):
        i, j = np.nonzero(mask)
        starts.append(np.column_stack([i + shift[0], j + shift[1]]))
        codes.append(np.full(len(i), code, dtype=np.int64))
        owners.append(owner[mask])
    start = np.vstack(starts)
    code = np.concatenate(codes)
    owner = np.concatenate(owners)
    end = start + _DIRECTIONS[code]

    # Successor edge: the one leaving this edge's end node — unique
    # except at a corner where two filled cells meet diagonally.
    stride = nv + 2
    start_key = start[:, 0] * stride + start[:, 1]
    order = np.argsort(start_key, kind="stable")
    sorted_key = start_key[order]
    end_key = end[:, 0] * stride + end[:, 1]
    lo = np.searchsorted(sorted_key, end_key, side="left")
    hi = np.searchsorted(sorted_key, end_key, side="right")
    succ = order[lo]
    forked = np.nonzero(hi - lo == 2)[0]
    if len(forked):
        node = end[forked]
        south_west = pad[node[:, 0], node[:, 1]]
        north_east = pad[node[:, 0] + 1, node[:, 1] + 1]
        south_east = pad[node[:, 0] + 1, node[:, 1]]
        north_west = pad[node[:, 0], node[:, 1] + 1]
        filled_a = np.where(south_west != 0, south_west, south_east)
        filled_b = np.where(south_west != 0, north_east, north_west)
        turn_right = filled_a == filled_b
        wanted = np.where(turn_right, (code[forked] + 3) % 4, (code[forked] + 1) % 4)
        first, second = order[lo[forked]], order[lo[forked] + 1]
        succ[forked] = np.where(code[first] == wanted, first, second)

    walk, loop_offsets = _cycles(succ)
    # Drop the vertices where the walk goes straight on.
    predecessor = np.roll(walk, 1)
    predecessor[loop_offsets[:-1]] = walk[loop_offsets[1:] - 1]
    corner = code[walk] != code[predecessor]

    pieces: dict[int, tuple[np.ndarray | None, list]] = {}
    for k in range(len(loop_offsets) - 1):
        edges = walk[loop_offsets[k] : loop_offsets[k + 1]]
        nodes = start[edges[corner[loop_offsets[k] : loop_offsets[k + 1]]]]
        ring = np.column_stack([us[nodes[:, 0]], vs[nodes[:, 1]]])
        entry = pieces.setdefault(int(owner[edges[0]]), (None, []))
        if _signed_area(ring) > 0.0:
            pieces[int(owner[edges[0]])] = (ring, entry[1])
        else:
            entry[1].append(ring)
    return [(outer, holes) for outer, holes in pieces.values() if outer is not None]


@njit(cache=True)
def _cycles(succ):
    """Edges in walk order and the offsets of the cycles of the permutation *succ*."""
    n = len(succ)
    visited = np.zeros(n, dtype=np.bool_)
    walk = np.empty(n, dtype=np.int64)
    offsets = np.empty(n + 1, dtype=np.int64)
    pos = 0
    loops = 0
    for e0 in range(n):
        if visited[e0]:
            continue
        offsets[loops] = pos
        loops += 1
        e = e0
        while not visited[e]:
            visited[e] = True
            walk[pos] = e
            pos += 1
            e = succ[e]
    offsets[loops] = pos
    return walk, offsets[: loops + 1]


def faces_from_pieces(pieces: list[tuple[np.ndarray, list]], axis: int, level: float) -> list:
    """Planar faces at ``axis = level`` from ``(outer, holes)`` loops of ``(u, v)`` vertices."""
    from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
    )
    from OCC.Core.gp import gp_Pnt  # noqa: PLC0415

    u, v = (k for k in range(3) if k != axis)

    def wire(ring: np.ndarray):
        polygon = BRepBuilderAPI_MakePolygon()
        for p in ring:
            xyz = [0.0, 0.0, 0.0]
            xyz[axis] = level
            xyz[u], xyz[v] = float(p[0]), float(p[1])
            polygon.Add(gp_Pnt(*xyz))
        polygon.Close()
        return polygon.Wire()

    faces = []
    for outer, holes in pieces:
        maker = BRepBuilderAPI_MakeFace(wire(outer), True)
        for hole in holes:
            maker.Add(wire(hole))
        faces.append(maker.Face())
    return faces


def fuse_rectilinear_faces(faces: list, tolerance: float):
    """The union of coplanar *faces* as a compound of faces; ``None`` unless all are rectilinear."""
    from magnelio.geo._prism_fuse import _compound  # noqa: PLC0415

    rows = rectilinear_rings(faces, tolerance)
    if rows is None:
        return None
    axis, level, rings = rows
    return _compound(faces_from_pieces(rectilinear_union(rings, tolerance), axis, level))
