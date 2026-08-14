"""Declarative high-level port objects (WP4.1).

:class:`PortWaveguide` and :class:`PortAnalytical` describe *what* a
port is — a face and a mode count — without committing to a 2D solver
path.  :class:`AnalysisScatteringTD` (which owns the mesh and the
boundary conditions) resolves them into the concrete specs at
construction time via :func:`resolve_declarative_port`:

- ``PortWaveguide`` inspects the (BC-PEC-consolidated) mesh on the
  port face: **≥ 2 conductor groups** on the cross-section select the
  multi-conductor TEM/QTEM Laplace path
  (:class:`PortSpecMultiConductor` — TEM with the scalar ε of a
  homogeneous cross-section, QTEM ``epsilon_r=None`` for an
  inhomogeneous one); **a hollow cross-section** (0/1 groups) selects
  the TE/TM curl-curl path (:class:`PortSpecNumerical`).  The
  declarative mode-count semantics is uniform: "the
  ``n_modes`` lowest modes of whatever is on this face" — on a
  homogeneous conductor cross-section ``n_modes > K−1`` extends the
  TEM line modes by the lowest TE/TM modes (merged by cut-off in one
  operator); on an inhomogeneous one by the true hybrid ζ-pencil
  eigenpairs at ``f_calc`` (which must propagate there; otherwise
  the factory raises with guidance).
- ``PortAnalytical`` maps directly onto the closed-form specs
  (:class:`PortSpecCoax` / :class:`PortSpecRectWG`) — no mesh
  inspection involved.

``plane`` accepts a :class:`BoxFace` or a string (``"zmin"``,
``"z_min"``, case-insensitive).
"""

# Design: WP-U2/U3/U6 (uniform declarative mode-count semantics).

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from magnelio._operators.material_matrices import flatten_port_plane_pec_mask
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.auto_conductors import (
    extract_conductor_groups_from_mesh,
)
from magnelio.ports._modal.factory import (
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    _pec_faces_from_mask,
)
from magnelio.ports._modal.port_plane import (
    BoxFace,
    PortPlane,
    build_port_edge_pec_mask,
    resolve_port_edge_pec,
)

PlaneLike = Union[BoxFace, str]

_FACE_BY_KEY = {face.value.replace("_", ""): face for face in BoxFace}


def normalize_box_face(plane: PlaneLike) -> BoxFace:
    """Return a :class:`BoxFace` for ``plane``.

    Accepts ``BoxFace`` instances and strings in any common spelling:
    ``"zmin"``, ``"z_min"``, ``"Z_MIN"``, …
    """
    if isinstance(plane, BoxFace):
        return plane
    if isinstance(plane, str):
        key = plane.lower().replace("_", "").replace("-", "")
        if key in _FACE_BY_KEY:
            return _FACE_BY_KEY[key]
        raise ValueError(
            f"unknown port plane {plane!r}; expected one of {sorted(_FACE_BY_KEY)} or a BoxFace",
        )
    raise TypeError(
        f"plane must be a BoxFace or a face-name string; got {type(plane).__name__}",
    )


def window_from_corners(face: BoxFace, corners) -> tuple:
    """Project world-coordinate ``corners`` onto the tangential window.

    ``corners`` are two opposite 3D corner points (any order, ``None``
    components allowed); the result is the in-plane window
    ``((a1, b1), (a2, b2))`` in the global tangential-axis ordering the
    spec layer works with, with unbounded sides as ``±inf``.

    The components along the face's normal axis carry no information —
    *face* fixes that coordinate — so they may be ``None``; if both are
    given they must agree, since two differing values describe a box
    perpendicular to the port plane, which is a likely axis mix-up
    rather than a window.
    """
    try:
        p, q = corners
        p = tuple(p)
        q = tuple(q)
        if len(p) != 3 or len(q) != 3:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"corners must be two opposite 3D corner points "
            f"((x0, y0, z0), (x1, y1, z1)); got {corners!r}",
        ) from None
    n_axis = face.normal_axis
    pn, qn = p[n_axis], q[n_axis]
    if pn is not None and qn is not None and float(pn) != float(qn):
        raise ValueError(
            f"corners of a port on {face.value!r} must agree along the "
            f"face normal axis {'xyz'[n_axis]!r} (or leave it None); got "
            f"{pn!r} and {qn!r} — that describes a box perpendicular to "
            f"the port plane",
        )
    tang = [axis for axis in range(3) if axis != n_axis]
    window = []
    for axis in tang:
        lo, hi = p[axis], q[axis]
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        if hi < lo:
            lo, hi = hi, lo
        window.append((lo, hi))
    (a1, a2), (b1, b2) = window
    return ((a1, b1), (a2, b2))


def point_on_face(face: BoxFace, point) -> tuple[float, float]:
    """Project a world-coordinate ``point`` onto the tangential frame.

    ``point`` is an ``(x, y, z)`` triple; the result is the in-plane
    pair ``(a, b)`` in the global tangential-axis ordering the spec
    layer works with.  The component along the face's normal axis is
    fixed by *face* already and is ignored (``None`` is fine there);
    the two tangential components must be numbers.
    """
    try:
        p = tuple(point)
        if len(p) != 3:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"center must be an (x, y, z) world-coordinate point; got {point!r}",
        ) from None
    n_axis = face.normal_axis
    tang = [axis for axis in range(3) if axis != n_axis]
    coords = []
    for axis in tang:
        c = p[axis]
        if c is None:
            raise ValueError(
                f"center component along the tangential axis "
                f"{'xyz'[axis]!r} must be a number; got None in {point!r}",
            )
        coords.append(float(c))
    return (coords[0], coords[1])


@dataclass(frozen=True)
class PortWaveguide:
    """Generic declarative waveguide port: "solve whatever is on this face".

    The mode-solver path (TEM / QTEM / TE-TM) is selected from the mesh
    cross-section at analysis-construction time — see the module
    docstring for the rules.

    Parameters
    ----------
    name : str
        Unique port name.
    plane : BoxFace or str
        Bbox face the port lives on (``"zmin"``, ``BoxFace.Z_MIN``, …).
    corners : tuple of tuple, optional
        Sub-rectangle of the face, given as two opposite corners
        ``((x0, y0, z0), (x1, y1, z1))`` in world coordinates [m] —
        the same form as :meth:`~magnelio.geo.Brick.from_corners`.
        Corner order does not matter.  The component along the face's
        normal axis is fixed by *plane* already; write it as ``None``
        (or repeat the same value on both corners — differing values
        are rejected as a likely axis mix-up).  An oversized rectangle
        is clipped to the domain and snapped to the nearest grid
        nodes, and a tangential component may be ``None`` to reach the
        domain boundary on that side.  The window-boundary BCs follow
        the legacy edge rule: an edge on a domain boundary inherits
        that wall's BC, an interior edge inherits the port face's BC —
        so a port embedded in a PEC wall gets a PEC frame, which also
        counts as a conductor (ground) in the mode-path detection.
        ``None`` (default) covers the whole face.
    n_modes : int, default 1
        Number of modes to solve on the port.
    """

    name: str
    plane: PlaneLike
    corners: Optional[tuple] = None
    n_modes: int = 1

    def __post_init__(self) -> None:
        face = normalize_box_face(self.plane)  # fail fast on bad input
        if self.corners is not None:
            window_from_corners(face, self.corners)  # fail fast on bad input
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {self.n_modes}")


@dataclass(frozen=True)
class PortAnalytical:
    """Declarative port with a closed-form analytical reference mode.

    Parameters
    ----------
    name : str
        Unique port name.
    plane : BoxFace or str
        Bbox face the port lives on.
    family : {"coax", "rect_wg"}
        Which analytical family describes the cross-section.
    inner_radius, outer_radius : float, optional
        Coax conductor radii [m] (required for ``family="coax"``).
    width, height : float, optional
        Rectangular-waveguide cross-section [m] (required for
        ``family="rect_wg"``); ``width`` is the (usually broader)
        dimension along the lower-numbered global tangential axis.
    epsilon_r : float, default 1.0
        Relative permittivity of the (homogeneous) filling.
    center : tuple of float, default (0.0, 0.0, 0.0)
        Cross-section anchor as an ``(x, y, z)`` world-coordinate point
        [m] — the coax axis, or the lower-left corner of the rectangle.
        The component along the face's normal axis is fixed by *plane*
        already and is ignored (``None`` is fine there).
    n_modes : int, default 1
        Number of modes.
    """

    name: str
    plane: PlaneLike
    family: str = "coax"
    inner_radius: Optional[float] = None
    outer_radius: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    epsilon_r: float = 1.0
    center: tuple = (0.0, 0.0, 0.0)
    n_modes: int = 1

    def __post_init__(self) -> None:
        face = normalize_box_face(self.plane)
        point_on_face(face, self.center)  # fail fast on bad input
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1; got {self.n_modes}")
        if self.family == "coax":
            if self.inner_radius is None or self.outer_radius is None:
                raise ValueError(
                    "family='coax' requires inner_radius= and outer_radius=",
                )
        elif self.family == "rect_wg":
            if self.width is None or self.height is None:
                raise ValueError(
                    "family='rect_wg' requires width= and height=",
                )
        else:
            raise ValueError(
                f"unknown analytical port family {self.family!r}; expected 'coax' or 'rect_wg'",
            )


@dataclass(frozen=True)
class PortLumped:
    """Declarative lumped port on a straight interior edge path.

    The high-level spelling of the lumped Thévenin port: two endpoints
    and a reference impedance, optionally backed by an RLC companion
    element.  Resolved into a
    :class:`~magnelio.ports._lumped.PortSpecLumped` by the analysis.

    Parameters
    ----------
    name : str
        Unique port name.
    start, end : tuple of float
        Endpoints in metres; must differ along exactly one Cartesian
        axis after grid snapping.
    Z0 : float, default 50.0
        Power-wave reference impedance [Ω]; without *element* also the
        internal Thévenin impedance.
    element : SeriesRLC or ParallelRLC, optional
        Companion element replacing the pure resistor as the port's
        internal impedance.
    """

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    Z0: float = 50.0
    element: object | None = None


DeclarativePort = Union[PortWaveguide, PortAnalytical, PortLumped]


def resolve_declarative_port(
    port: DeclarativePort,
    mesh: Mesh,
):
    """Resolve a declarative port into a concrete spec.

    ``mesh`` must already carry every PEC source the port should see —
    in particular the BC-PEC consolidation of
    :class:`AnalysisScatteringTD` (``Mesh.with_pec_boundaries``) has to
    run *before* this resolution, since conductor detection reads
    ``mesh.pec_mask_edges``.
    """
    if isinstance(port, PortLumped):
        from magnelio.ports._lumped import PortSpecLumped  # noqa: PLC0415

        return PortSpecLumped(
            name=port.name,
            start=port.start,
            end=port.end,
            Z0=port.Z0,
            element=port.element,
        )

    face = normalize_box_face(port.plane)

    if isinstance(port, PortAnalytical):
        center_uv = point_on_face(face, port.center)
        if port.family == "coax":
            return PortSpecCoax(
                name=port.name,
                plane=face,
                inner_radius=port.inner_radius,
                outer_radius=port.outer_radius,
                epsilon_r=port.epsilon_r,
                center=center_uv,
                n_modes=port.n_modes,
            )
        return PortSpecRectWG(
            name=port.name,
            plane=face,
            width_a=port.width,
            height_b=port.height,
            epsilon_r=port.epsilon_r,
            center=center_uv,
            n_modes=port.n_modes,
        )

    if not isinstance(port, PortWaveguide):
        raise TypeError(
            f"cannot resolve port of type {type(port).__name__}",
        )

    # Conductor topology of the cross-section.  Detection must see the
    # same contour the mode solver will see: build_modal_port flattens
    # the port-plane PEC-mask slab in-line with the first interior slab
    # before extracting conductor groups, so replicate that flatten on
    # a mask copy (an all-PEC BC on the port face itself would
    # otherwise swallow the whole plane into one component).  For
    # sub-face ports the window-boundary Dirichlet ring (edge-BC rule,
    # read on the *unflattened* mask like build_modal_port does) joins
    # the detection, so an embedded port's PEC frame counts as a
    # conductor — e.g. the ground of a microstrip window.
    detection_mesh = dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(
            mesh.pec_mask_edges,
            mesh,
            face,
        ),
    )
    window = None if port.corners is None else window_from_corners(face, port.corners)
    plane = PortPlane.from_mesh(face, detection_mesh, window=window)
    extra_mask = None
    if window is not None:
        edge_pec = resolve_port_edge_pec(
            plane,
            mesh,
            _pec_faces_from_mask(mesh),
        )
        extra_mask = build_port_edge_pec_mask(plane, edge_pec)
    try:
        groups = extract_conductor_groups_from_mesh(
            plane,
            detection_mesh,
            extra_pec_edge_mask=extra_mask,
        )
    except ValueError:
        groups = None  # hollow (or PEC-free) cross-section

    eps_values = _cross_section_epsilons(
        mesh,
        face,
        u_window=plane.u_node_window,
        v_window=plane.v_node_window,
    )

    if groups is not None and len(groups) >= 2:
        homogeneous = len(eps_values) == 1
        scalar = homogeneous and _is_isotropic(next(iter(eps_values)))
        return PortSpecMultiConductor(
            name=port.name,
            plane=face,
            epsilon_r=(float(next(iter(eps_values))[0]) if scalar else None),
            n_modes=port.n_modes,
            window=window,
        )

    # Hollow cross-section: TE/TM curl-curl on a homogeneous filling.
    if len(eps_values) != 1 or not _is_isotropic(next(iter(eps_values))):
        raise ValueError(
            f"PortWaveguide {port.name!r}: hollow cross-section on "
            f"{face.value!r} has an inhomogeneous or anisotropic "
            f"filling ({sorted(eps_values)}); the TE/TM path requires "
            f"a homogeneous scalar permittivity — use an explicit "
            f"PortSpecNumerical / custom setup instead",
        )
    # mode_type=None: unified multi-mode port — the n_modes lowest
    # cut-offs across the TE *and* TM families, injected/recorded/
    # terminated by one operator (WP-R3, closes the former TE+TM
    # source-injection collision).
    return PortSpecNumerical(
        name=port.name,
        plane=face,
        n_modes=port.n_modes,
        epsilon_r=float(next(iter(eps_values))[0]),
        window=window,
    )


def _cross_section_epsilons(
    mesh: Mesh,
    face: BoxFace,
    u_window: tuple[int, int] | None = None,
    v_window: tuple[int, int] | None = None,
) -> set[tuple[float, float, float]]:
    """Distinct permittivity tuples of the non-PEC cells in the port slab.

    ``u_window`` / ``v_window`` clip the slab to a sub-face plane's
    cell windows (``PortPlane.u_node_window`` convention); ``None``
    scans the whole face.
    """
    n_axis = face.normal_axis
    n_cells = (mesh.grid.Nx, mesh.grid.Ny, mesh.grid.Nz)[n_axis]
    idx = n_cells - 1 if face.is_max else 0
    slab = np.take(mesh.material_id, idx, axis=n_axis)
    if u_window is not None and v_window is not None:
        # np.take keeps the remaining axes in ascending global-axis
        # order; map the (u, v) windows onto that order.
        wins = {face.u_axis: u_window, face.v_axis: v_window}
        a_ax, b_ax = sorted((face.u_axis, face.v_axis))
        slab = slab[
            wins[a_ax][0] : wins[a_ax][1],
            wins[b_ax][0] : wins[b_ax][1],
        ]
    out: set[tuple[float, float, float]] = set()
    for mid in np.unique(slab):
        mat = mesh.material_library[int(mid)]
        if mat.is_pec:
            continue
        out.add(tuple(float(e) for e in mat.epsilon))
    if not out:
        raise ValueError(
            f"port plane on {face.value!r} has no non-PEC cells — "
            f"the cross-section is entirely conductor",
        )
    return out


def _is_isotropic(eps: tuple[float, float, float]) -> bool:
    return eps[0] == eps[1] == eps[2]
