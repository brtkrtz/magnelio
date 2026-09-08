"""PortSpecLumped + build_lumped_port — declarative spec + builder.

Mirror of the modal-port factory pattern: a plain dataclass holds the
user's intent (label, endpoints, internal impedance), and a builder
function resolves it onto a concrete mesh, returning a runtime operator
that implements the :class:`magnelio.ports.base.Port` protocol.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from magnelio.circuit.companion import ParallelRLC, SeriesRLC
from magnelio.circuit.rasterize import rasterize_curve, rasterize_points
from magnelio.ports._lumped.operator import LumpedElementOperator, PortOperatorLumped
from magnelio.ports._lumped.port_report import LumpedPortReport

_FACE_AXIS_SIDE = {
    "xmin": (0, "min"),
    "xmax": (0, "max"),
    "ymin": (1, "min"),
    "ymax": (1, "max"),
    "zmin": (2, "min"),
    "zmax": (2, "max"),
}


@dataclass
class PortSpecLumped:
    """Declarative description of a lumped discrete port / RLC element.

    Parameters
    ----------
    name : str
        Unique port identifier (used as recorder channel key).
    start, end : tuple[float, float, float], optional
        Endpoints in metres — the two-point short form of *path*.
        Exclusive with it; exactly one of the two forms is required.
        Under a clipping symmetry declaration they stay in full-model
        coordinates; ``Z0`` / ``element`` are full-model values
        throughout, and the builder derives the internally scaled
        half-model device.
    path : Curve or sequence of points, optional
        The port's path through the model: a
        :class:`~magnelio.geo.Curve` or a sequence of at least two
        ``(x, y, z)`` points [m] taken as polyline vertices.  It may
        run in any direction; the rasterised staircase carries an
        oblique path.  The chain must not visit an edge twice, so a
        self-crossing or doubled-back path is rejected — a two-terminal
        element is a series chain.
    samples_per_cell : int, default 4
        Path samples per smallest cell while rasterising.  Higher
        values only refine which edges a strongly curved path picks up.
    Z0 : float
        Power-wave reference impedance [Ω] (default 50 Ω).  Without an
        ``element`` it is also the internal Thévenin impedance — the
        classic discrete port.
    element : SeriesRLC or ParallelRLC, optional
        Trapezoidal companion element replacing the pure
        resistor as the port's internal impedance: an excited port
        becomes an RLC-backed source, an unexcited one a passive lumped
        RLC load.  ``None`` (default) means ``SeriesRLC(R=Z0)`` — the
        behaviour-identical classic port.  The element instance is
        deep-copied per run, so its transient state never leaks between
        excitations.
    """

    name: str
    start: Optional[tuple[float, float, float]] = None
    end: Optional[tuple[float, float, float]] = None
    Z0: float = 50.0
    element: Optional[Union[SeriesRLC, ParallelRLC]] = None
    path: object = None
    samples_per_cell: int = 4


def build_lumped_port(
    spec: PortSpecLumped,
    mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    dt: float,
) -> PortOperatorLumped:
    """Build a :class:`PortOperatorLumped` from *spec* and *mesh*.

    Resolves ``spec.start`` / ``spec.end`` to flat E-edge indices on
    *mesh*, computes the lossless update coefficient ``β_E = dt / M_eps``
    at those edges, and assembles a ready-to-use lumped port operator.

    Parameters
    ----------
    spec : PortSpecLumped
    mesh : Mesh
    m_eps : np.ndarray
        Diagonal of the FIT ``M_eps`` matrix on the flat E layout.
    m_mu : np.ndarray
        Diagonal of ``M_mu`` (unused for lumped ports; accepted for
        symmetry with :func:`magnelio.ports._modal.build_modal_port`).
    dt : float
        Solver time step [s].

    Returns
    -------
    PortOperatorLumped
        Runtime operator implementing :class:`magnelio.ports.base.Port`.

    Raises
    ------
    ValueError
        When ``spec.start`` and ``spec.end`` snap to nodes that do not
        differ along exactly one axis, or coincide entirely.
    """
    del m_mu  # unused — kept in signature to mirror build_modal_port

    what = f"PortSpecLumped {spec.name!r}"
    path = normalize_path(what, spec.start, spec.end, spec.path)
    path, report = _resolve_symmetry(what, path, mesh)
    flat_indices, ijk_list, dl_list, components, signs = _resolve_chain(
        what,
        path,
        mesh,
        spec.samples_per_cell,
    )
    _warn_dead_edges(what, mesh, flat_indices, components)

    m_eps_port = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_port

    # DD-172: the user's Z0 / element describe the full-model device.
    # The meshed half carries the internally scaled device (series cut:
    # Z/2, parallel cut: 2·Z); with that, recorded power waves and the
    # injection pick up the modal √2-per-plane convention through the
    # shared port_report plumbing.
    z_int = report.z_internal_scale if report is not None else 1.0

    return PortOperatorLumped(
        name=spec.name,
        Z0=spec.Z0 * z_int,
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        beta_E=beta_E,
        edge_components=components,
        edge_signs=signs,
        # Fresh companion state per built operator (operators are built
        # per excitation; the user's spec instance must stay pristine).
        element=_scaled_element(spec.element, z_int),
        port_report=report,
    )


def build_lumped_element(
    spec,
    mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    dt: float,
) -> LumpedElementOperator:
    """Build a passive :class:`LumpedElementOperator` from a declarative
    :class:`magnelio.circuit.LumpedElement` (DD-123).

    Same edge-chain resolution as :func:`build_lumped_port`, but the
    result is the plain element operator: no excitation is ever set, so
    it acts as a pure in-circuit load (``v_src = 0`` in the Thévenin
    update).  ``Z0 = 0`` — a passive element has no power-wave
    reference.

    Parameters
    ----------
    spec : magnelio.circuit.LumpedElement
    mesh : Mesh
    m_eps : np.ndarray
        Diagonal of the FIT ``M_eps`` matrix on the flat E layout.
    m_mu : np.ndarray
        Unused; accepted for symmetry with the port builders.
    dt : float
        Solver time step [s].
    """
    del m_mu

    what = f"LumpedElement {spec.name!r}"
    path = normalize_path(what, spec.start, spec.end, getattr(spec, "path", None))
    path, report = _resolve_symmetry(what, path, mesh)
    flat_indices, ijk_list, dl_list, components, signs = _resolve_chain(
        what,
        path,
        mesh,
        getattr(spec, "samples_per_cell", 4),
    )
    _warn_dead_edges(what, mesh, flat_indices, components)

    m_eps_elem = np.asarray(m_eps[flat_indices], dtype=float)
    beta_E = dt / m_eps_elem

    z_int = report.z_internal_scale if report is not None else 1.0

    return LumpedElementOperator(
        name=spec.name,
        Z0=0.0,
        element=_scaled_element(spec.element, z_int),
        flat_edge_indices=list(flat_indices),
        ijk_list=ijk_list,
        dl_list=dl_list,
        edge_components=components,
        edge_signs=signs,
        beta_E=beta_E,
        port_report=report,
    )


def normalize_path(what: str, start, end, path):
    """Bring the two declaration forms onto one representation.

    Returns a :class:`~magnelio.geo.Curve` unchanged, or a tuple of
    ``(x, y, z)`` points.  Exactly one of ``start``/``end`` and *path*
    may be given; the two-point form becomes a two-vertex polyline.
    """
    has_ends = start is not None or end is not None
    if has_ends and path is not None:
        raise ValueError(
            f"{what}: give either start/end or path=, not both.",
        )
    if has_ends:
        if start is None or end is None:
            raise ValueError(
                f"{what}: start and end must be given together "
                f"(or use path= for a multi-point path).",
            )
        return (_point(what, start), _point(what, end))
    if path is None:
        raise ValueError(
            f"{what}: a path is required — start/end, or path= with a "
            f"magnelio.geo.Curve or a sequence of (x, y, z) points [m].",
        )
    if hasattr(path, "_occ_shape"):  # a Curve, without importing geo here
        return path
    try:
        pts = tuple(_point(what, p) for p in path)
    except TypeError:
        raise TypeError(
            f"{what}: path= must be a magnelio.geo.Curve or a sequence of "
            f"(x, y, z) points [m]; got {type(path).__name__}.",
        ) from None
    if len(pts) < 2:
        raise ValueError(
            f"{what}: a point path needs at least two (x, y, z) triples [m]; "
            f"got {len(pts)} point(s).",
        )
    return pts


def declared_path(obj):
    """The declared path of a lumped port or element.

    Returns a :class:`~magnelio.geo.Curve` or a tuple of points,
    whichever form was declared, with the two-point short form folded
    into a two-vertex polyline.  Shared by the plots so a picture can
    never disagree with the builder about what was declared.
    """
    what = f"{type(obj).__name__} {getattr(obj, 'name', '?')!r}"
    return normalize_path(
        what,
        getattr(obj, "start", None),
        getattr(obj, "end", None),
        getattr(obj, "path", None),
    )


def _point(what: str, p) -> tuple[float, float, float]:
    try:
        q = tuple(float(c) for c in p)
    except (TypeError, ValueError):
        raise TypeError(
            f"{what}: expected an (x, y, z) point [m]; got {p!r}.",
        ) from None
    if len(q) != 3:
        raise ValueError(f"{what}: expected an (x, y, z) point [m]; got {p!r}.")
    return q


def _densify(points, grid, samples_per_cell: int) -> np.ndarray:
    """Resample a polyline so consecutive samples never skip a node.

    :func:`~magnelio.circuit.rasterize_points` requires that; bare
    polyline vertices do not satisfy it, and an oblique segment given
    by its two ends alone would rasterise to an L, not a staircase.
    Doing it here keeps a point path free of OCC — the reason DD-079
    deferred subsuming the old two-point rasteriser.
    """
    step = min(grid.dx_min, grid.dy_min, grid.dz_min) / float(samples_per_cell)
    pts = np.asarray(points, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def _resolve_chain(what: str, path, mesh, samples_per_cell: int):
    """Rasterise *path* onto the primary E-edges through the DD-076 kernel.

    Replaces the degenerate two-point rasteriser this module used to
    carry — the one DD-075 and DD-079 both marked for subsumption — so
    a lumped element and a voltage probe can no longer disagree about
    which edges a path occupies.

    Returns
    -------
    tuple
        ``(flat_indices, ijk_list, dl_list, components, signs)``.
    """
    grid = mesh.grid
    if int(samples_per_cell) < 2:
        raise ValueError(
            f"{what}: samples_per_cell must be >= 2 so the rasteriser cannot "
            f"skip a grid node; got {samples_per_cell!r}.",
        )
    if hasattr(path, "_occ_shape"):
        ep = rasterize_curve(path, grid, samples_per_cell=int(samples_per_cell))
    else:
        ep = rasterize_points(_densify(path, grid, int(samples_per_cell)), grid)

    if len(ep) == 0 or sum(ep.dls) == 0.0:
        raise ValueError(
            f"{what}: zero-length edge chain — the path collapses to a single "
            f"grid node.  Refine the mesh, or move the terminals apart.",
        )

    # A two-terminal element is a *series* chain: one current flows
    # through every edge once.  An edge visited twice would receive the
    # injection twice and its voltage would be counted twice, so a
    # self-crossing or doubled-back path is not an element.  (An
    # impressed current may do both — DD-227 folds them — which is why
    # the rasteriser itself allows it.)
    flat = np.asarray(ep.flat_indices, dtype=np.int64)
    uniq, counts = np.unique(flat, return_counts=True)
    if uniq.size != flat.size:
        n_rep = int((counts > 1).sum())
        raise ValueError(
            f"{what}: the path visits {n_rep} grid edge(s) more than once "
            f"(it crosses itself or doubles back).  A two-terminal element "
            f"is a series chain, so every edge may be traversed once.",
        )

    components = [{"x": 0, "y": 1, "z": 2}[a] for a in ep.axes]
    signs = [float(s) for s in ep.signs]
    return list(ep.flat_indices), list(ep.ijk), list(ep.dls), components, signs


def _warn_dead_edges(what: str, mesh, flat_indices, components) -> None:
    """Warn about chain edges the solver holds at zero.

    An edge inside a perfect conductor or tangential to a PEC wall is
    reset every step, so the element's injection there is discarded and
    the device is silently shorted along that edge.  ``SourceCurrentPath``
    reports the same condition rather than radiating less than asked.
    """
    mask = getattr(mesh, "pec_mask_edges", None)
    if mask is None:
        return
    grid = mesh.grid
    n_Ex = grid.Nx * (grid.Ny + 1) * (grid.Nz + 1)
    n_Ey = (grid.Nx + 1) * grid.Ny * (grid.Nz + 1)
    offset = (0, n_Ex, n_Ex + n_Ey)
    dead = 0
    for flat, comp in zip(flat_indices, components):
        local = flat - offset[comp]
        if 0 <= local < mask.shape[1] and bool(mask[comp, local]):
            dead += 1
    if dead:
        warnings.warn(
            f"{what}: {dead} of {len(flat_indices)} chain edge(s) are held at "
            f"zero by a perfect conductor (inside a PEC body, or tangential to "
            f"a PEC wall).  The element is shorted along them and does not act "
            f"as declared — move the path off the conductor, or leave a gap "
            f"for it.",
            UserWarning,
            stacklevel=3,
        )


def _scaled_element(
    element: Optional[Union[SeriesRLC, ParallelRLC]],
    z_scale: float,
) -> Optional[Union[SeriesRLC, ParallelRLC]]:
    """A fresh companion instance with all impedances scaled by *z_scale*.

    Both topologies scale identically (``R → R·s``, ``L → L·s``,
    ``C → C/s`` multiplies every branch impedance by *s*).  ``z_scale
    == 1`` reduces to the historic deep copy — fresh transient state,
    identical values.
    """
    if element is None:
        return None
    if z_scale == 1.0:
        return copy.deepcopy(element)
    return type(element)(
        R=None if element.R is None else element.R * z_scale,
        L=None if element.L is None else element.L * z_scale,
        C=None if element.C is None else element.C / z_scale,
    )


def _resolve_symmetry(what: str, path, mesh):
    """Relate a declared path to the declared symmetry planes.

    Validates the path against every symmetry plane of the mesh's
    boundary conditions, clips a plane-crossing path to the meshed
    half, and returns ``(path, report)`` with a
    :class:`LumpedPortReport` when at least one plane cuts it (``None``
    otherwise).  Meshes without symmetry declarations pass through
    untouched.

    The case table is DD-172's, lifted from a two-point chain to a
    polyline: a path lying *in* the plane, one *crossing* it (which
    must be mirror-symmetric about it, and needs an electric plane
    because a current normal to a magnetic plane mirrors anti-parallel),
    and one that merely ends on it under an as-built declaration.

    A ``Curve`` path cannot be clipped here — splitting it would need
    the OCC kernel and would not survive back into a ``Curve`` the
    rasteriser can sample the same way.  A curve that reaches a
    symmetry plane is therefore rejected with the point-path form as
    the way out; one that stays clear is passed through.
    """
    # Detection + validation matrix: DD-172.  The mirror-twin warning
    # follows the modal wording (ports/_modal/factory.py,
    # _with_symmetry_faces, DD-155).
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    bc = getattr(mesh, "boundary_conditions", None)
    sym = symmetry_entries(bc)
    if not sym:
        return path, None

    types = bc_type_entries(bc)
    grid = mesh.grid
    axis_coords = (grid.x, grid.y, grid.z)
    axis_widths = (grid.dx, grid.dy, grid.dz)
    is_curve = hasattr(path, "_occ_shape")
    pts = None if is_curve else [list(p) for p in path]

    faces: list[tuple[str, str, str]] = []
    for face in sorted(sym):
        axis, side = _FACE_AXIS_SIDE[face]
        position = sym[face]
        as_built = position is None
        coords = axis_coords[axis]
        wall = float(coords[0] if side == "min" else coords[-1]) if as_built else float(position)
        widths = axis_widths[axis]
        tol = 0.5 * float(widths[0] if side == "min" else widths[-1])
        kind = types[face]

        def _inward(c: float, _w=wall, _s=side) -> float:
            # Signed distance into the meshed half (positive = kept side).
            return (c - _w) if _s == "min" else (_w - c)

        if is_curve:
            lo, hi = path._analytic_bbox()[axis * 2 : axis * 2 + 2]
            near = min(_inward(float(lo)), _inward(float(hi)))
            if near <= tol:
                raise ValueError(
                    f"{what}: a Curve path reaches the symmetry plane on face "
                    f"{face!r} (plane at {wall:.6g} m along {'xyz'[axis]}).  "
                    f"Clipping a curve to the meshed half is not supported — "
                    f"declare the path as a sequence of points, which can be "
                    f"clipped, or keep the curve clear of the plane.",
                )
            continue

        d = [_inward(p[axis]) for p in pts]
        d_min, d_max = min(d), max(d)

        if d_max < -tol:
            raise ValueError(
                f"{what}: the path lies in the half-space removed by the "
                f"symmetry declaration on face {face!r} (plane at "
                f"{wall:.6g} m along {'xyz'[axis]}).  Declare the element "
                f"in the kept half, or drop the symmetry declaration.",
            )

        if all(abs(v) <= tol for v in d):
            # The whole path lies in the plane.
            if kind == "PEC":
                raise ValueError(
                    f"{what}: the path lies in the electric symmetry plane on "
                    f"face {face!r}; tangential edges inside an electric wall "
                    f"are shorted.  Move the element off the plane, or declare "
                    f"a magnetic symmetry plane if the fields support it.",
                )
            faces.append((face, kind, "containment"))
            continue

        if d_min < -tol:
            # The plane bisects the path.
            if kind == "PMC":
                raise ValueError(
                    f"{what}: the path crosses the magnetic symmetry plane on "
                    f"face {face!r}.  A current normal to a magnetic plane "
                    f"mirrors anti-parallel, so no full-model element "
                    f"corresponds to this declaration; a plane-crossing "
                    f"element needs an electric symmetry plane.",
                )
            _require_mirror_symmetric(what, face, pts, axis, wall, tol)
            pts = _clip_to_wall(pts, d, axis, wall, tol)
            faces.append((face, kind, "crossing"))
            continue

        if d_min <= tol:
            # A vertex sits on the plane and the body stays inside.
            on_terminal = abs(d[0]) <= tol or abs(d[-1]) <= tol
            if as_built and on_terminal:
                # ForceSymmetry*: the model is declared as built
                # (halved), so the half element ending on the plane IS
                # the crossing declaration.
                if kind == "PMC":
                    raise ValueError(
                        f"{what}: the path ends on the magnetic symmetry plane "
                        f"on face {face!r} along the plane normal.  Its mirror "
                        f"continuation crosses the plane, which a magnetic "
                        f"plane does not support (the mirrored current is "
                        f"anti-parallel).",
                    )
                faces.append((face, kind, "crossing"))
                continue
            raise ValueError(
                f"{what}: a path vertex lies on the symmetry plane on face "
                f"{face!r}.  Endpoints are declared in full-model coordinates "
                f"here — declare the full element crossing the plane, or "
                f"declare the boundary as as-built symmetry if the geometry "
                f"is meant to be halved.",
            )
        # else: the path is inside, away from this plane.

    out = path if is_curve else tuple(tuple(p) for p in pts)
    if not faces:
        warnings.warn(
            f"{what}: the path does not touch the declared symmetry "
            f"plane(s) {sorted(sym)} — in the full model this element "
            f"has a mirror twin, and the half-model run can only "
            f"realise the symmetric (in-phase) response of the twin "
            f"pair.  Model the element on the symmetry plane, or drop "
            f"the symmetry declaration if the twin is not intended.",
            UserWarning,
            stacklevel=3,
        )
        return out, None

    return out, LumpedPortReport(symmetry_faces=tuple(faces))


def _require_mirror_symmetric(what, face, pts, axis, wall, tol) -> None:
    """A plane-crossing element must be its own mirror image.

    Reflecting the path about the plane and reversing it must reproduce
    it; otherwise the meshed half does not stand for a full-model
    device.  For a two-point chain this reduces to the DD-172 endpoint
    test.
    """
    mirrored = []
    for p in reversed(pts):
        q = list(p)
        q[axis] = 2.0 * wall - q[axis]
        mirrored.append(q)
    worst = max(
        (abs(a[k] - b[k]) for a, b in zip(pts, mirrored) for k in range(3)),
        default=0.0,
    )
    if len(mirrored) != len(pts) or worst > tol:
        raise ValueError(
            f"{what}: the path crosses the symmetry plane on face {face!r} "
            f"asymmetrically (largest mismatch {worst:.6g} m against its own "
            f"mirror image).  An element crossing a symmetry plane must be "
            f"mirror-symmetric about it.",
        )


def _clip_to_wall(pts, d, axis, wall, tol):
    """Keep the meshed half, with the crossing vertex exactly on the wall."""
    kept: list[list[float]] = []

    def _push(q):
        if not kept or max(abs(q[k] - kept[-1][k]) for k in range(3)) > 1e-15:
            kept.append(q)

    for i, p in enumerate(pts):
        if d[i] >= -tol:
            _push(list(p))
        if i + 1 < len(pts) and (d[i] < 0.0) != (d[i + 1] < 0.0):
            t = d[i] / (d[i] - d[i + 1])
            q = [p[k] + t * (pts[i + 1][k] - p[k]) for k in range(3)]
            q[axis] = wall
            _push(q)
    return kept
