"""Huygens box geometry shared by the surface monitors.

A Huygens box is an axis-aligned box of grid-node planes inside the
physical domain, carrying the tangential E and H that stand for
everything the box encloses.  Two monitors sample it: the far-field
transform accumulates a surface DFT on it, the field-surface monitor
records its time series.  Both need the same box, the same face
sampling and the same exclusions, so the placement lives here.

Sampling convention: each face lies on a node plane; the tangential
fields come from the sanctioned cell-centre interpolation of the two
adjacent cell layers, linearly combined onto the node plane.  The
surface stays exactly closed (faces meet at box edges without gaps or
overhangs) and second-order accurate on graded grids.
"""

# Design: DD-173 (far-field box), DD-226 (shared placement).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.monitors.base import _cell_centres
from magnelio.post._symmetry import mirror_spec_for_face
from magnelio.post.far_field import ImagePlane

_AXES = "xyz"
_FACE_NAMES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
# Tangential (E, H) component names per face axis.
_TANGENTIALS = {
    0: ("Ey", "Ez", "Hy", "Hz"),
    1: ("Ex", "Ez", "Hx", "Hz"),
    2: ("Ex", "Ey", "Hx", "Hy"),
}


@dataclass
class _BoxFace:
    """One sampled face of the Huygens box (internal)."""

    name: str
    axis: int
    sign: float  # outward normal component (±1)
    plane: float  # node-plane coordinate [m]
    slab: tuple  # (ix, iy, iz) slices, 2 cells thick along axis
    weight: float  # linear weight of the inner->outer layer pair
    tangent_axes: tuple  # (t1, t2)
    c1: np.ndarray  # patch centres along t1 [m]
    c2: np.ndarray
    w1: np.ndarray  # patch widths along t1 [m]
    w2: np.ndarray
    keep: np.ndarray | None = None  # (c1, c2) patch weights: 0 = excluded


def face_node_indices(
    mesh, *, margin_cells: int, zero_margin_faces=(), label: str = "Huygens box"
) -> tuple[list, list, list]:
    """Node indices of the automatic box, plus the faces it leaves open.

    The box sits ``margin_cells`` inside the physical domain on every
    absorbing face (the mesher-extended absorber cells excluded); a
    face closed with PEC or PMC is not crossed at all and is left to
    image theory.  ``zero_margin_faces`` are absorbing faces sampled at
    the absorber interface itself (a feed guide crosses them, DD-198).
    """
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        cpml_thickness_of,
    )

    grid = mesh.grid
    bc = mesh.boundary_conditions
    types = bc_type_entries(bc)
    pml = getattr(mesh, "pml_cells", None) or {}
    n_cells = (grid.Nx, grid.Ny, grid.Nz)

    lo_n = [0, 0, 0]
    hi_n = list(n_cells)
    open_faces: list[str] = []
    for face in _FACE_NAMES:
        axis = _AXES.index(face[0])
        at_low = face.endswith("min")
        kind = types[face]
        if kind in ("PEC", "PMC"):
            continue
        if kind != "CPML":
            raise ValueError(
                f"{label}: face {face!r} is {kind!r} — the fields of a "
                f"periodic model are not described by a Huygens box."
            )
        n_abs = pml.get(face) or cpml_thickness_of(bc)
        margin = 0 if face in zero_margin_faces else margin_cells
        idx = n_abs + margin
        if at_low:
            lo_n[axis] = idx
        else:
            hi_n[axis] = n_cells[axis] - idx
        open_faces.append(face)
    return lo_n, hi_n, open_faces


def image_planes_for(mesh, open_faces) -> list:
    """Image planes for the walls the box does not cross."""
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    grid = mesh.grid
    nodes = (grid.x, grid.y, grid.z)
    types = bc_type_entries(mesh.boundary_conditions)
    sym = symmetry_entries(mesh.boundary_conditions)
    out = []
    for face in _FACE_NAMES:
        if face in open_faces:
            continue
        axis = _AXES.index(face[0])
        kind = types[face]
        spec = mirror_spec_for_face(face, kind, nodes[axis])
        out.append(
            ImagePlane(
                axis=axis,
                position=spec.wall,
                kind=kind,
                at_low=face.endswith("min"),
                physical_halfspace=face not in sym,
            )
        )
    return out


def build_faces(grid, lo_n, hi_n, open_faces) -> list:
    """Assemble the sampled :class:`_BoxFace` records of the box."""
    nodes = (grid.x, grid.y, grid.z)
    centres = [_cell_centres(nodes[a]) for a in range(3)]
    widths = [np.asarray(d, dtype=float) for d in (grid.dx, grid.dy, grid.dz)]

    faces = []
    for face in open_faces:
        axis = _AXES.index(face[0])
        at_low = face.endswith("min")
        nn = lo_n[axis] if at_low else hi_n[axis]
        t1, t2 = (a for a in range(3) if a != axis)
        slices = [None, None, None]
        slices[axis] = slice(nn - 1, nn + 1)
        slices[t1] = slice(lo_n[t1], hi_n[t1])
        slices[t2] = slice(lo_n[t2], hi_n[t2])
        cc = centres[axis]
        w = float((nodes[axis][nn] - cc[nn - 1]) / (cc[nn] - cc[nn - 1]))
        faces.append(
            _BoxFace(
                name=face,
                axis=axis,
                sign=-1.0 if at_low else 1.0,
                plane=float(nodes[axis][nn]),
                slab=tuple(slices),
                weight=w,
                tangent_axes=(t1, t2),
                c1=centres[t1][lo_n[t1] : hi_n[t1]].copy(),
                c2=centres[t2][lo_n[t2] : hi_n[t2]].copy(),
                w1=widths[t1][lo_n[t1] : hi_n[t1]].copy(),
                w2=widths[t2][lo_n[t2] : hi_n[t2]].copy(),
            )
        )
    return faces


def exclude_pec_patches(mesh, faces, lo_n, port_footprints=None) -> None:
    """Zero the patch weights inside conductors and feed guides.

    A Huygens face cuts through whatever the model puts there.  A patch
    whose sampled cells are perfect conductor on both sides of the face
    carries no field and no source; a patch inside the footprint of a
    waveguide-port window on an absorbing face (DD-198) samples the
    guided wave of the feed, which is not an external source.  Both are
    left out.
    """
    material_id = getattr(mesh, "material_id", None)
    library = getattr(mesh, "material_library", None) or {}
    pec_ids = [mid for mid, mat in library.items() if getattr(mat, "is_pec", False)]
    footprints = port_footprints or {}
    for bf in faces:
        keep = np.ones((bf.c1.size, bf.c2.size), dtype=float)
        t1, t2 = bf.tangent_axes
        if material_id is not None and pec_ids:
            inner = list(bf.slab)
            outer = list(bf.slab)
            nn = bf.slab[bf.axis].stop - 1
            inner[bf.axis] = nn - 1
            outer[bf.axis] = nn
            pec_in = np.isin(material_id[tuple(inner)], pec_ids)
            pec_out = np.isin(material_id[tuple(outer)], pec_ids)
            keep[pec_in & pec_out] = 0.0
        for win in footprints.get(bf.name, ()):
            sl = [slice(None), slice(None)]
            for pos, t in enumerate((t1, t2)):
                if t in win:
                    lo, hi = (int(v) for v in win[t])
                    sl[pos] = slice(max(lo - lo_n[t], 0), max(hi - lo_n[t], 0))
            keep[tuple(sl)] = 0.0
        bf.keep = keep if (keep < 1.0).any() else None


# ---------------------------------------------------------------------------
# Native Yee sampling of a box face (DD-226)
# ---------------------------------------------------------------------------

# Yee position of each component per axis: "n" = node, "c" = cell centre.
_YEE = {
    "Ex": ("c", "n", "n"),
    "Ey": ("n", "c", "n"),
    "Ez": ("n", "n", "c"),
    "Hx": ("n", "c", "c"),
    "Hy": ("c", "n", "c"),
    "Hz": ("c", "c", "n"),
}


def _solver_dual_widths(d: np.ndarray) -> np.ndarray:
    """Dual widths in the solver convention (boundary = full end cell)."""
    from magnelio.monitors.base import _solver_dual_widths as impl  # noqa: PLC0415

    return impl(d)


def component_axis_coords(grid, comp: str, axis: int) -> np.ndarray:
    """Sample coordinates of *comp* along *axis* — nodes or cell centres."""
    nodes = np.asarray((grid.x, grid.y, grid.z)[axis], dtype=float)
    return nodes if _YEE[comp][axis] == "n" else _cell_centres(nodes)


def component_length(grid, comp: str) -> np.ndarray:
    """The FIT length that converts *comp* from a grid quantity to a field.

    ``E = e / l_primal`` along the component's own axis, ``H = h /
    l_dual`` with the solver's dual convention (DD-082/DD-085).
    """
    axis = _AXES.index(comp[1])
    d = np.asarray((grid.dx, grid.dy, grid.dz)[axis], dtype=float)
    return d if comp.startswith("E") else _solver_dual_widths(d)


def sample_component_plane(fields, grid, comp: str, axis: int, index: int, windows):
    """One component of a face, on its own Yee positions.

    Takes the layer ``index`` along *axis* out of the component's array
    and converts it to a physical field, without any averaging: the
    equivalent-source construction needs the values the update
    equations use, and a smoothed sample leaks at the level of its own
    smoothing error.  *windows* are inclusive index ranges per tangent
    axis.  Returns the ``(n1, n2)`` array in host memory.
    """
    comp_axis = _AXES.index(comp[1])
    if comp_axis == axis:
        raise ValueError(f"{comp} is normal to the {_AXES[axis]} face, not tangential")
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = index
    for t, (lo, hi) in windows.items():
        sl[t] = slice(lo, hi + 1)
    plane = getattr(fields, comp)[tuple(sl)]
    if hasattr(plane, "get"):  # GPU backend: one small transfer per sample
        plane = plane.get()
    plane = np.asarray(plane, dtype=float)

    lo, hi = windows[comp_axis]
    vec = np.asarray(component_length(grid, comp)[lo : hi + 1], dtype=float)
    t1, t2 = (a for a in range(3) if a != axis)
    return plane / (vec[:, None] if comp_axis == t1 else vec[None, :])


def snap_corners(grid, corners, lo_n, hi_n, *, label: str) -> tuple[list, list]:
    """Snap a user box onto grid-node planes inside the allowed range.

    *lo_n* / *hi_n* are the node indices the automatic box would use —
    the absorber-free range.  A face given outside it is clamped there,
    because a Huygens surface inside the absorber records a field the
    absorber is busy destroying.
    """
    nodes = (grid.x, grid.y, grid.z)
    try:
        p, q = corners
    except (TypeError, ValueError):
        raise ValueError(
            f"{label}: corners must be two opposite points "
            f"((x0, y0, z0), (x1, y1, z1)); got {corners!r}",
        ) from None
    lo, hi = list(lo_n), list(hi_n)
    for axis in range(3):
        a, b = p[axis], q[axis]
        if a is None and b is None:
            continue
        n = np.asarray(nodes[axis], dtype=float)
        vals = sorted(float(v) for v in (a, b) if v is not None and np.isfinite(v))
        if len(vals) == 2:
            i0 = int(np.clip(np.argmin(np.abs(n - vals[0])), lo_n[axis], hi_n[axis] - 1))
            i1 = int(np.clip(np.argmin(np.abs(n - vals[1])), i0 + 1, hi_n[axis]))
        else:
            i0 = int(np.clip(np.argmin(np.abs(n - vals[0])), lo_n[axis], hi_n[axis] - 1))
            i1 = hi_n[axis]
        lo[axis], hi[axis] = i0, i1
    return lo, hi


__all__ = [
    "_AXES",
    "_BoxFace",
    "_FACE_NAMES",
    "_TANGENTIALS",
    "build_faces",
    "component_axis_coords",
    "component_length",
    "exclude_pec_patches",
    "face_node_indices",
    "image_planes_for",
    "sample_component_plane",
    "snap_corners",
]
