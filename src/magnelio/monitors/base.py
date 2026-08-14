"""
Base utilities for field monitors.

Provides the :class:`MonitorRegion` helper that resolves a user-specified
corner-box specification into concrete grid index ranges after
snapping to the nearest cell centres.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.grid import GridLines

# ---------------------------------------------------------------------------
# Cell-centre interpolation helpers
# ---------------------------------------------------------------------------


def _solver_dual_widths(d: np.ndarray) -> np.ndarray:
    """Dual widths in the SOLVER convention: boundary = full end cell.

    Mirrors ``operators.material_matrices._build_avg_d`` — the h states
    at domain-boundary nodes carry the full first/last cell as their
    dual length (DD-082 B1 finding), so converting ``h -> H`` must
    divide by the same widths.
    """
    n = d.size
    out = np.empty(n + 1)
    out[0] = d[0]
    if n > 1:
        out[1:n] = 0.5 * (d[:-1] + d[1:])
    out[n] = d[-1]
    return out


def _interp_to_cell_centres(
    fields: FieldState, components: list[str], ix: slice, iy: slice, iz: slice, grid: GridLines
) -> dict[str, np.ndarray]:
    """Physical fields at cell centres within *ix, iy, iz* (DD-085).

    The solver states are FIT grid quantities (``e = E·l_primal`` [V],
    ``h = H·l_dual`` [A]); each staggered sample is converted to the
    physical field at its own position (``E = e/l``, ``H = h/l_dual``
    with the solver dual convention) and then averaged over its
    staggered neighbours so that the result lives at cell centres
    ``(xc[i], yc[j], zc[k])``.

    Parameters
    ----------
    fields : FieldState
        Current field snapshot (grid-quantity states).
    components : list[str]
        Subset of ``["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]``.
    ix, iy, iz : slice
        Cell-index slices (stop-exclusive) defining the sub-region.
    grid : GridLines
        Simulation grid providing the per-edge/per-face lengths.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping component name -> cell-centred array of shape
        ``(ix.stop - ix.start, iy.stop - iy.start, iz.stop - iz.start)``;
        E in [V/m], H in [A/m].
    """
    result = {}
    i0, i1 = ix.start, ix.stop
    j0, j1 = iy.start, iy.stop
    k0, k1 = iz.start, iz.stop

    # GPU backend: the field arrays are device arrays, which refuse
    # implicit mixing with NumPy operands.  Interpolate on the device
    # (the edge-length vectors are tiny host->device transfers) and
    # move only the region-sized results back at the end — for plane
    # monitors that is orders of magnitude cheaper than syncing the
    # full field state every recorded step.
    xp = np
    if type(getattr(fields, components[0])).__module__.partition(".")[0] == "cupy":
        import cupy as xp  # noqa: PLC0415

    dx_h = np.asarray(grid.dx, dtype=float)
    dy_h = np.asarray(grid.dy, dtype=float)
    dz_h = np.asarray(grid.dz, dtype=float)
    dx, dy, dz = xp.asarray(dx_h), xp.asarray(dy_h), xp.asarray(dz_h)

    need_h = any(c.startswith("H") for c in components)
    if need_h:
        dxa = xp.asarray(_solver_dual_widths(dx_h))
        dya = xp.asarray(_solver_dual_widths(dy_h))
        dza = xp.asarray(_solver_dual_widths(dz_h))

    for comp in components:
        arr = getattr(fields, comp)
        if comp == "Ex":
            # Ex[i,j,k] at (xc[i], y[j], z[k]) — average over j,k
            # neighbours; all four samples share the edge length dx[i]
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1]
                    + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0 + 1 : k1 + 1]
                )
                / dx[i0:i1, None, None]
            )
        elif comp == "Ey":
            # Ey[i,j,k] at (x[i], yc[j], z[k]) — average over i,k neighbours
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1]
                    + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0 + 1 : k1 + 1]
                )
                / dy[None, j0:j1, None]
            )
        elif comp == "Ez":
            # Ez[i,j,k] at (x[i], y[j], zc[k]) — average over i,j neighbours
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0 + 1 : j1 + 1, k0:k1]
                )
                / dz[None, None, k0:k1]
            )
        elif comp == "Hx":
            # Hx[i,j,k] at (x[i], yc[j], zc[k]) — the two staggered
            # samples sit at different x nodes: convert each first
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dxa[i0:i1, None, None]
                + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1] / dxa[i0 + 1 : i1 + 1, None, None]
            )
        elif comp == "Hy":
            # Hy[i,j,k] at (xc[i], y[j], zc[k]) — average over j neighbours
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dya[None, j0:j1, None]
                + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1] / dya[None, j0 + 1 : j1 + 1, None]
            )
        elif comp == "Hz":
            # Hz[i,j,k] at (xc[i], yc[j], z[k]) — average over k neighbours
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dza[None, None, k0:k1]
                + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1] / dza[None, None, k0 + 1 : k1 + 1]
            )
    if xp is not np:
        result = {comp: arr.get() for comp, arr in result.items()}
    return result


# ---------------------------------------------------------------------------
# MonitorRegion
# ---------------------------------------------------------------------------


@dataclass
class MonitorRegion:
    """Resolved grid region for a monitor.

    Created by :func:`resolve_region` from a pair of opposite corners and a
    :class:`~magnelio.mesh.grid.GridLines` object.

    Attributes
    ----------
    ix, iy, iz : slice
        Cell-index slices (stop-exclusive) into cell-centred arrays of shape
        ``(Nx, Ny, Nz)``.
    xc, yc, zc : np.ndarray
        Cell-centre coordinates within the region.
    ndim : int
        Effective dimensionality (0, 1, 2, or 3).
    """

    ix: slice
    iy: slice
    iz: slice
    xc: np.ndarray
    yc: np.ndarray
    zc: np.ndarray
    ndim: int


def _cell_centres(nodes: np.ndarray) -> np.ndarray:
    """Cell-centre coordinates from node positions."""
    return 0.5 * (nodes[:-1] + nodes[1:])


def _snap_range(cc: np.ndarray, lo: float, hi: float) -> slice:
    """Return a slice of cell indices whose centres fall within [lo, hi].

    For a zero-width range (lo == hi), snaps to the single nearest cell.
    """
    if lo == hi:
        idx = int(np.argmin(np.abs(cc - lo)))
        return slice(idx, idx + 1)
    mask = (cc >= lo) & (cc <= hi)
    indices = np.nonzero(mask)[0]
    if len(indices) == 0:
        idx = int(np.argmin(np.abs(cc - 0.5 * (lo + hi))))
        return slice(idx, idx + 1)
    return slice(int(indices[0]), int(indices[-1]) + 1)


def normalize_corners(corners) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Bring a corner specification into sorted ``(lo, hi)`` form.

    Parameters
    ----------
    corners : tuple of tuple or None
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        monitor box [m], in the same form
        as :meth:`Brick.from_corners`.  ``None`` for the whole domain.
        Per component, ``None`` means "unbounded on this side" — in
        the first corner it reads as ``-inf``, in the second as
        ``+inf``; ``math.inf`` / ``float("inf")`` say the same thing
        explicitly.  Corner order does not matter: each axis is
        sorted, so a box given "backwards" resolves the same.  An axis
        whose two values coincide is degenerate and selects a single
        cell — that is how a plane, a line or a point is expressed.

    Returns
    -------
    (lo, hi) : each a 3-tuple of float
        Componentwise lower and upper bounds, possibly infinite.
    """
    if corners is None:
        return ((-inf,) * 3, (inf,) * 3)
    try:
        p0, p1 = corners
    except (TypeError, ValueError):
        raise ValueError(
            "corners must be two opposite points "
            "((x0, y0, z0), (x1, y1, z1)) or None for the whole domain; "
            f"got {corners!r}",
        ) from None
    if len(p0) != 3 or len(p1) != 3:
        raise ValueError(
            f"each corner needs three coordinates; got {p0!r} and {p1!r}",
        )
    lo, hi = [], []
    for a, b in zip(p0, p1):
        a = -inf if a is None else float(a)
        b = inf if b is None else float(b)
        lo.append(min(a, b))
        hi.append(max(a, b))
    return tuple(lo), tuple(hi)


def _corners_array(corners) -> np.ndarray:
    """Corners as a ``(2, 3)`` float array for HDF5 storage.

    Normalised first, so ``None`` becomes ``∓inf`` and the axes are
    sorted — the stored form is canonical and reads back identically.
    """
    lo, hi = normalize_corners(corners)
    return np.asarray([lo, hi], dtype=float)


def _corners_from_array(arr) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Inverse of :func:`_corners_array`."""
    a = np.asarray(arr, dtype=float).reshape(2, 3)
    return (tuple(float(x) for x in a[0]), tuple(float(x) for x in a[1]))


def resolve_region(corners, grid: GridLines) -> MonitorRegion:
    """Snap a corner-box specification to cell-centre indices.

    Parameters
    ----------
    corners : tuple of tuple or None
        See :func:`normalize_corners`.
    grid : GridLines
        Simulation grid.

    Returns
    -------
    MonitorRegion
    """
    lo_all, hi_all = normalize_corners(corners)
    nodes = (grid.x, grid.y, grid.z)
    slices = []
    coords = []
    ndim = 0

    for lo, hi, n in zip(lo_all, hi_all, nodes):
        cc = _cell_centres(n)
        if lo == -inf and hi == inf:
            sl = slice(0, len(cc))
        else:
            # Clamp the open side onto the domain so _snap_range never
            # sees an infinity (its empty-range fallback averages the
            # bounds, which is meaningless for one).
            sl = _snap_range(
                cc,
                cc[0] if lo == -inf else lo,
                cc[-1] if hi == inf else hi,
            )
        slices.append(sl)
        coords.append(cc[sl])
        if sl.stop - sl.start > 1:
            ndim += 1

    return MonitorRegion(
        ix=slices[0],
        iy=slices[1],
        iz=slices[2],
        xc=coords[0],
        yc=coords[1],
        zc=coords[2],
        ndim=ndim,
    )


@dataclass
class PlaneView:
    """A 2D plotting plane resolved from a 2D or 3D monitor region.

    Attributes
    ----------
    free : list[tuple[int, np.ndarray]]
        The two in-plane axes as ``(axis_index, cell_centres)`` pairs,
        in ascending axis order.
    normal_idx : int
        Axis index (0/1/2) of the plane normal.
    normal_pos : float
        Position of the plane along the normal axis [m] (snapped to
        the nearest cell centre for 3D regions).
    slice_index : int or None
        Index to take along ``normal_idx`` in the (unsqueezed) spatial
        data arrays; ``None`` for 2D regions whose arrays are already
        two-dimensional.
    """

    free: list[tuple[int, np.ndarray]]
    normal_idx: int
    normal_pos: float
    slice_index: int | None

    def take2d(self, arr: np.ndarray) -> np.ndarray:
        """Reduce a spatial data array to the in-plane 2D array."""
        if self.slice_index is None:
            return arr
        return np.take(arr, self.slice_index, axis=self.normal_idx)


_AXES = ("x", "y", "z")


def resolve_plane_view(region: MonitorRegion, normal: str | None, position: float) -> PlaneView:
    """Resolve the 2D plane a monitor plot should show.

    For a 2D region the plane is fixed by the region itself; *normal*
    (if given) is validated against it and *position* is ignored.  For
    a 3D region *normal* selects the slice axis and *position* [m] is
    snapped to the nearest cell-centre plane.
    """
    coords = [region.xc, region.yc, region.zc]

    if region.ndim == 2:
        free = [(i, c) for i, c in enumerate(coords) if len(c) > 1]
        normal_idx = 3 - free[0][0] - free[1][0]
        if normal is not None and normal != _AXES[normal_idx]:
            raise ValueError(
                f"This 2D monitor lies in a plane with normal "
                f"'{_AXES[normal_idx]}'; got normal={normal!r}."
            )
        return PlaneView(
            free=free,
            normal_idx=normal_idx,
            normal_pos=float(coords[normal_idx][0]),
            slice_index=None,
        )

    if region.ndim != 3:
        raise NotImplementedError(f"Plotting for {region.ndim}D monitors not yet supported.")

    if normal is None:
        raise ValueError(
            "This monitor records a 3D volume; select a slice plane with "
            "normal=('x'|'y'|'z') and position=<coordinate in m>."
        )
    if normal not in _AXES:
        raise ValueError(f"normal must be 'x', 'y', or 'z'; got {normal!r}")
    normal_idx = _AXES.index(normal)
    cc = coords[normal_idx]
    k = int(np.argmin(np.abs(cc - position)))
    return PlaneView(
        free=[(i, coords[i]) for i in range(3) if i != normal_idx],
        normal_idx=normal_idx,
        normal_pos=float(cc[k]),
        slice_index=k,
    )


# ---------------------------------------------------------------------------
# Symmetry mirroring (DD-154)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MirrorSpec:
    """One symmetry plane a monitor region touches.

    Attributes
    ----------
    axis : int
        World axis index (0/1/2) of the mirror normal.
    wall : float
        Mirror-plane position [m] — the *physical* wall: on a PEC face
        the outermost grid line, on a PMC face half the boundary cell
        outside it (where the natural magnetic wall of the staggered
        grid sits; after the mesher's pull-in that is exactly the
        declared plane).
    kind : str
        Wall type, ``"PEC"`` or ``"PMC"``.
    at_low : bool
        True when the mirror sits at the low end of the axis (the
        mirrored copy prepends).
    """

    axis: int
    wall: float
    kind: str
    at_low: bool


def resolve_mirrors(region: MonitorRegion, mesh) -> tuple[MirrorSpec, ...]:
    """Symmetry planes the monitor region reaches.

    Read from the mesh's boundary declaration; a plane counts only when
    the region's cells extend to the domain wall on that side —
    mirroring a region that stops short of the plane would draw a
    detached copy.
    """
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    bc = getattr(mesh, "boundary_conditions", None)
    sym = symmetry_entries(bc)
    if not sym:
        return ()
    types = bc_type_entries(bc)
    grid = mesh.grid
    slices = (region.ix, region.iy, region.iz)
    axis_nodes = (grid.x, grid.y, grid.z)
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    out = []
    for face in sorted(sym):
        axis = _AXES.index(face[0])
        at_low = face.endswith("min")
        sl = slices[axis]
        if at_low and sl.start != 0:
            continue
        if not at_low and sl.stop != n_cells[axis]:
            continue
        n = np.asarray(axis_nodes[axis], dtype=float)
        kind = types[face]
        if kind == "PEC":
            wall = n[0] if at_low else n[-1]
        elif at_low:
            wall = n[0] - 0.5 * (n[1] - n[0])
        else:
            wall = n[-1] + 0.5 * (n[-1] - n[-2])
        out.append(MirrorSpec(axis=axis, wall=float(wall), kind=kind, at_low=at_low))
    return tuple(out)


def mirrors_to_jsonable(mirrors: tuple[MirrorSpec, ...]) -> list:
    """Mirror specs as a JSON-serialisable list for HDF5 attributes."""
    return [[m.axis, m.wall, m.kind, bool(m.at_low)] for m in mirrors]


def mirrors_from_jsonable(data) -> tuple[MirrorSpec, ...]:
    """Inverse of :func:`mirrors_to_jsonable` (``None`` → empty)."""
    if not data:
        return ()
    return tuple(
        MirrorSpec(axis=int(a), wall=float(w), kind=str(k), at_low=bool(lo)) for a, w, k, lo in data
    )


def component_mirror_key(component: str) -> tuple[str, int | None]:
    """Field kind and world component axis of a plot component name.

    ``"Ex"`` → ``("E", 0)``; ``"E"``/``"|H|"`` (magnitudes) →
    ``(field, None)`` — a magnitude is mirror-even, no sign involved.
    """
    if component in ("E", "H", "|E|", "|H|"):
        return component.strip("|"), None
    return component[0], _AXES.index(component[1])


def mirror_sign(field: str, comp_axis: int | None, mirror_axis: int, kind: str) -> float:
    """±1 continuation factor of a field component across a mirror.

    Across a magnetic (PMC) symmetry plane E continues like a polar
    vector (normal component odd, tangential even) and H like a
    pseudovector (normal even, tangential odd); across an electric
    (PEC) plane the roles swap.  Magnitudes (``comp_axis=None``) are
    always even.
    """
    if comp_axis is None:
        return 1.0
    is_normal = comp_axis == mirror_axis
    flips_normal = (kind == "PMC") if field == "E" else (kind == "PEC")
    return -1.0 if is_normal == flips_normal else 1.0


def mirror_extend(
    coords: np.ndarray,
    values: np.ndarray,
    spec: MirrorSpec,
    arr_axis: int,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend one axis of a data array across a mirror plane.

    Returns the extended cell-centre coordinates and the values with
    the sign-weighted mirrored copy prepended (``at_low``) or appended.
    """
    reflected = 2.0 * spec.wall - coords[::-1]
    flipped = sign * np.flip(values, axis=arr_axis)
    if spec.at_low:
        return (
            np.concatenate([reflected, coords]),
            np.concatenate([flipped, values], axis=arr_axis),
        )
    return (
        np.concatenate([coords, reflected]),
        np.concatenate([values, flipped], axis=arr_axis),
    )


def mirror_plane_arrays(
    pv: PlaneView,
    mirrors: tuple[MirrorSpec, ...],
    c0: np.ndarray,
    c1: np.ndarray,
    entries: list[tuple[np.ndarray | None, str, int | None]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Mirror a set of 2D plane slices across the in-plane symmetry planes.

    *entries* is a list of ``(values_2d_or_None, field, comp_axis)``;
    all arrays share the plane's ``(len(c0), len(c1))`` layout (free
    axes in ascending world-axis order).  Mirrors whose axis is the
    plane normal do not extend the image and are ignored here.
    """
    (i0, _), (i1, _) = pv.free
    arrays = [e[0] for e in entries]
    for spec in mirrors:
        if spec.axis == i0:
            arr_axis = 0
        elif spec.axis == i1:
            arr_axis = 1
        else:
            continue
        c_new = None
        for k, (arr, fld, comp_axis) in enumerate(entries):
            if arrays[k] is None:
                continue
            s = mirror_sign(fld, comp_axis, spec.axis, spec.kind)
            c_new, arrays[k] = mirror_extend(
                c0 if arr_axis == 0 else c1,
                arrays[k],
                spec,
                arr_axis,
                s,
            )
        if c_new is not None:
            if arr_axis == 0:
                c0 = c_new
            else:
                c1 = c_new
    return c0, c1, arrays


def _resolve_component(data: dict[str, np.ndarray], component: str) -> np.ndarray:
    """Resolve a component name to an array, supporting vector magnitudes.

    Individual components (``"Ex"``, ``"Hy"``, …) are returned directly.
    ``"|E|"`` and ``"|H|"`` compute the L2 norm of the available
    E- or H-field components: ``sqrt(|Ex|² + |Ey|² + |Ez|²)``.
    """
    if component in data:
        return data[component]

    if component in ("E", "|E|"):
        parts = [data[c] for c in ("Ex", "Ey", "Ez") if c in data]
    elif component in ("H", "|H|"):
        parts = [data[c] for c in ("Hx", "Hy", "Hz") if c in data]
    else:
        raise KeyError(
            f"Component '{component}' not recorded. Available: {list(data.keys())}, 'E', 'H'"
        )

    if not parts:
        raise KeyError(f"No field components available for '{component}'.")

    return np.sqrt(sum(np.abs(p) ** 2 for p in parts))


def _expand_field_list(fields: list[str]) -> list[str]:
    """Expand shorthand field names to individual components.

    ``"E"`` -> ``["Ex", "Ey", "Ez"]``, ``"H"`` -> ``["Hx", "Hy", "Hz"]``.
    """
    _VALID = {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    components: list[str] = []
    for f in fields:
        if f == "E":
            components.extend(["Ex", "Ey", "Ez"])
        elif f == "H":
            components.extend(["Hx", "Hy", "Hz"])
        elif f in _VALID:
            components.append(f)
        else:
            raise ValueError(
                f"Unknown field '{f}'. Valid: 'E', 'H', or individual components {sorted(_VALID)}"
            )
    return components
