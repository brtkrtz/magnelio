"""Public modal-port specs and factory.

The integration tests for the modal-port pipeline used to repeat the same
half-dozen lines per port: pick the analytical solver, swap the
``(width_a, height_b)`` pair when constructing the X_MAX-side mode list
to match the local ``(u, v)`` frame, sample the modes onto the port
plane, build the operator with the full ``(m_eps, m_mu, dt, omega_calc)``
plumbing, and optionally wire a soft-source closure for the excitation.

This module exposes the public, face-agnostic description of a modal
port (``PortSpecCoax`` / ``PortSpecRectWG``) and a
single :func:`build_modal_port` factory that turns a spec + mesh +
material matrices into a ready-to-use :class:`PortOperatorModal`.

User-facing convention
----------------------

For both spec types, ``center`` (and for RectWG ``width_a`` /
``height_b``) are given in the *global* axis ordering: the two
tangential axes of the chosen ``BoxFace`` listed in *ascending* global
axis number.  Concretely, for a port on an X face (X_MIN or X_MAX):

- ``width_a``  is the dimension along global y,
- ``height_b`` is the dimension along global z,
- ``center``   is ``(c_y, c_z)``.

For a Y-face port the same rule yields ``(c_x, c_z)``, and for a Z-face
port ``(c_x, c_y)``.  This way the user's description does not depend
on whether the face is MIN or MAX — the factory handles the per-face
``(u, v)``-frame mapping (which has the (u, v) order reversed on MAX
faces so that ``u × v`` always points into the simulation domain).

For RectWG, this also means the lowest-cutoff mode returned at both
ends is the same physical mode: e.g. for a WR-90 with the broad side
along global y, the analytical solver receives ``(width_a_uv,
height_b_uv) = (WR90_A, WR90_B)`` at X_MIN and ``(WR90_B, WR90_A)`` at
X_MAX, and in both cases mode index 0 is the physical TE10 (cutoff
``c / (2 · WR90_A)``).
"""

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import (
    build_M_eps_vacuum,
    flatten_port_plane_mass,
    flatten_port_plane_mu,
    flatten_port_plane_pec_mask,
)
from magnelio.constants import C0
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.auto_conductors import (
    extract_conductor_groups_from_mesh,
)
from magnelio.ports._modal.coax import CoaxAnalyticalModeSolver
from magnelio.ports._modal.curl_curl_2d import (
    _hollow_pec_node_mask,
    build_2d_curl_curl,
    build_2d_gradient,
    build_2d_tm_curl_curl,
)
from magnelio.ports._modal.discrete import discretize_modes
from magnelio.ports._modal.mode import ModeType
from magnelio.ports._modal.numerical_2d import Numerical2DModeSolver
from magnelio.ports._modal.operator import (
    _DTBC_SLAB_DEFECT_TOL,
    PortOperatorModal,
    conformal_flux_patch_scale,
)
from magnelio.ports._modal.port_plane import (
    BoxFace,
    PortPlane,
    build_port_edge_pec_mask,
    magnetic_window_ends,
    resolve_port_edge_pec,
    window_domain_faces,
)
from magnelio.ports._modal.port_report import PortOperatorReport
from magnelio.ports._modal.rect import RectWGAnalyticalModeSolver
from magnelio.ports._modal.tem_laplace import (
    solve_qtem_laplace,
    solve_tem_laplace,
)


def _with_symmetry_faces(
    port_report: PortOperatorReport,
    plane,
    mesh,
    name: str = "",
) -> PortOperatorReport:
    """Record the symmetry planes cutting the port window (DD-154).

    A window edge on a domain face that is declared a symmetry plane
    makes this a half port; the report carries ``(face, wall_kind)``
    pairs so the publication layer can restore full-model impedance
    values.  Ports away from every symmetry plane pass through
    unchanged.
    """
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    bc = getattr(mesh, "boundary_conditions", None)
    sym = symmetry_entries(bc)
    if not sym:
        return port_report
    touched = set(window_domain_faces(plane, mesh.grid).values())
    types = bc_type_entries(bc)
    faces = tuple((f, types[f]) for f in sorted(touched & set(sym)))
    if not faces:
        # DD-155: a port away from every symmetry plane has a mirror
        # twin in the full model, and the half-model run can only
        # realise the symmetric (even) excitation of that twin pair.
        warnings.warn(
            f"port {name!r}: window does not touch the declared symmetry "
            f"plane(s) {sorted(sym)} — in the full model this port "
            f"has a mirror twin, and its S-parameters describe the "
            f"response under symmetric (in-phase) excitation of the "
            f"twin pair.  Model the port on the symmetry plane, or "
            f"drop the symmetry declaration if the twin is not "
            f"intended.",
            UserWarning,
            stacklevel=3,
        )
        return port_report
    return dataclasses.replace(port_report, symmetry_faces=faces)


@dataclass(frozen=True)
class PortSpecCoax:
    """Coaxial-line modal port (TEM, Phase 1).

    Parameters
    ----------
    name : str
        Port label, used by the recorder and S-parameter post-processing.
    plane : BoxFace
        Bbox face on which the port lives.
    inner_radius, outer_radius : float
        Inner and outer conductor radii [m].
    epsilon_r : float, default 1.0
        Relative permittivity of the dielectric.
    center : tuple[float, float], default (0.0, 0.0)
        Coax-axis location in the *global* tangential frame (lower-axis
        first).  See module docstring for the convention.
    n_modes : int, default 1
        Phase 1 supports only ``n_modes = 1`` (TEM).
    """

    name: str
    plane: BoxFace
    inner_radius: float
    outer_radius: float
    epsilon_r: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)
    n_modes: int = 1


@dataclass(frozen=True)
class PortSpecRectWG:
    """Rectangular-waveguide modal port (TE / TM, Phase 1).

    Parameters
    ----------
    name : str
        Port label.
    plane : BoxFace
        Bbox face on which the port lives.
    width_a : float
        Cross-section dimension along the *lower-numbered* tangential
        global axis [m].
    height_b : float
        Cross-section dimension along the *higher-numbered* tangential
        global axis [m].
    epsilon_r : float, default 1.0
        Relative permittivity.
    center : tuple[float, float], default (0.0, 0.0)
        Lower-left corner of the cross-section in the *global*
        tangential frame (lower-axis first).
    n_modes : int, default 1
        Number of modes returned by the analytical solver, ordered by
        ascending cutoff frequency.
    """

    name: str
    plane: BoxFace
    width_a: float
    height_b: float
    epsilon_r: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)
    n_modes: int = 1


@dataclass(frozen=True)
class PortSpecNumerical:
    """Numerical-mode-solver port for hollow homogeneously-filled cross-sections.

    Drives the numerical 2D curl-curl / node-Laplace eigenvalue solver
    (:class:`Numerical2DModeSolver`) for hollow waveguides whose
    cross-section has no analytical closed form (ridged, double-ridged,
    elliptical, circular with PEC bbox padding, …) or where a
    FIT-grid-native mode is preferred over an analytical projection.

    Scope: TE/TM modes on a hollow, homogeneously-filled cross-section.
    Multi-conductor TEM and inhomogeneous QTEM are served by
    :class:`PortSpecMultiConductor`.

    PEC walls are read from ``mesh.pec_mask_edges`` —
    the canonical 3D-mesh source.  Production setups define the wall
    geometry via :func:`Mesh.from_geometry` (OCC-meshed PEC body) or
    :meth:`Mesh.with_pec_boundaries` (BC-consolidated bbox faces); both
    populate ``mesh.pec_mask_edges`` automatically.  For bare
    ``Mesh.from_grid`` setups without either, the factory falls back to
    the standard hollow-waveguide assumption (all four lateral bbox
    faces are PEC walls).

    Parameters
    ----------
    name : str
        Port label.
    plane : BoxFace
        Bbox face on which the port lives.
    n_modes : int, default 1
        Number of modes returned by the eigsh solver, ordered by
        ascending cut-off frequency.
    epsilon_r : float, default 1.0
        Permittivity of the (homogeneous) cross-section filling.  Used
        for the H-profile bake-in and the sigma heuristic.
    mode_type : ModeType or None, default None
        ``None`` (default) solves **both** TE and TM families and
        keeps the ``n_modes`` lowest cut-offs, mixed types included —
        one operator injects/records/terminates every mode on the
        face (unified multi-mode port, WP-R3).  Pass ``ModeType.TE``
        or ``ModeType.TM`` to restrict to one family.
    window : tuple of two corner points, optional
        Sub-rectangle of the face as two opposite corners in *global*
        tangential-axis ordering (:meth:`PortPlane.from_mesh`
        convention).  ``None`` (default) covers the whole face.  The window-boundary BCs follow the
        legacy edge rule: a port edge on a domain boundary inherits
        that wall's BC, an interior edge inherits the port face's BC
        (both read from ``mesh.pec_mask_edges``).
    """

    name: str
    plane: BoxFace
    n_modes: int = 1
    epsilon_r: float = 1.0
    mode_type: Optional[ModeType] = None
    window: Optional[tuple] = None


@dataclass(frozen=True)
class BboxLateralConductor:
    """All four lateral bbox-wall nodes — the typical outer conductor.

    For a port plane on (say) X_MIN, the lateral walls are the four
    bbox faces whose normal axis is *tangential* to the port plane
    (i.e. ``Y_MIN``, ``Y_MAX``, ``Z_MIN``, ``Z_MAX``).  All primal 2D
    nodes that sit on any of these walls are taken to belong to this
    conductor — i.e. all nodes with ``i_u ∈ {0, Nu_node−1}`` *or*
    ``i_v ∈ {0, Nv_node−1}``.

    Useful for rectangular coax (the bbox itself acts as the outer
    conductor) and any other "outer = bbox boundary" topology.
    """


@dataclass(frozen=True)
class WallConductor:
    """Single bbox-wall PEC conductor (e.g. microstrip ground plane).

    ``face`` must be a bbox face whose normal axis is *tangential* to
    the port plane (a "lateral" face from the port plane's perspective).
    All primal 2D nodes lying on that wall are taken to belong to this
    conductor.  Mirrors the validation in
    :func:`_build_lateral_pec_edge_mask`.
    """

    face: BoxFace


@dataclass(frozen=True)
class RegionConductor:
    """Axis-aligned rectangular conductor region on the port plane.

    The two ranges are given in the *global* axis ordering — the same
    convention as :attr:`PortSpecCoax.center` and
    :attr:`PortSpecRectWG.width_a`.  Concretely:

    - X-face port (u/v axes are y, z): ``(y_range, z_range)``.
    - Y-face port (u/v axes are x, z): ``(x_range, z_range)``.
    - Z-face port (u/v axes are x, y): ``(x_range, y_range)``.

    The factory swaps the pair internally on MAX faces so the user
    description does not depend on whether the port is MIN or MAX.

    Primal 2D nodes whose physical (u, v) coordinate falls inside
    ``[range_a_lo, range_a_hi] × [range_b_lo, range_b_hi]`` (with a
    small tolerance relative to the port-plane extent) are taken to
    belong to this conductor.

    Typical uses: rectangular inner conductor of rect coax; the strip
    of a microstrip; a square bond pad.
    """

    axis_a_range: tuple[float, float]
    axis_b_range: tuple[float, float]


ConductorSpec = Union[BboxLateralConductor, WallConductor, RegionConductor]


@dataclass(frozen=True)
class PortSpecMultiConductor:
    """Multi-conductor numerical port spec (Phase 2b/c factory cleanup).

    Drives the TEM Laplace path
    (:func:`solve_tem_laplace`, when ``epsilon_r`` is set) or the QTEM
    dual-Laplace path (:func:`solve_qtem_laplace`, when ``epsilon_r``
    is ``None`` — the factory builds the vacuum reference mass via
    :func:`build_M_eps_vacuum`).  Returns the ``K − 1`` line modes
    (``K`` conductor groups): the single conductor mode for ``K = 2``,
    the modal basis of the line for ``K > 2`` — the capacitance-matrix
    eigenmodes (TEM) or the eigen-patterns of ``C v = ε_eff C_0 v``
    (QTEM), e.g. the even/odd pair of coupled lines, ordered by
    descending capacitance / ``ε_eff``.

    **Unified multi-mode port (WP-U2/WP-U6).**  On a homogeneous
    scalar filling (``epsilon_r`` set), ``n_modes > K − 1`` extends
    the port by the lowest TE/TM curl-curl modes of the same
    cross-section, merged by ascending cut-off (TEM channels first,
    ``f_c = 0``) — the exact continuum decomposition TEM ⊕ TE ⊕ TM;
    the discrete family cross-orthogonality is at solver tolerance
    (WP-U1).  On an inhomogeneous cross-section (``epsilon_r=None``,
    QTEM) no exact family split exists — the higher channels are the
    true hybrid eigenpairs of the ζ-pencil at ``f_calc``:
    profiles exact at ``f_calc``, dual-basis projections
    (the hybrids are not M_ε-orthogonal), termination per the standard
    defaults (certificates fail on inhomogeneous fillings → modal
    Mur-1st, loud notice; ``port_model="band"`` stays the
    reflection-critical opt-in for the tracked family).  Mode labels
    stay family-explicit (``TEM_lap00``, ``TE_num00``,
    ``QTEM_lap00``, ``HYB_zp00``, …).

    Parameters
    ----------
    name : str
        Port label.
    plane : BoxFace
        Bbox face on which the port lives.
    conductors : tuple[ConductorSpec, ...] or None, default None
        ``[ground, signal_1, signal_2, ...]`` — at least 2 entries
        when given.  ``conductors[0]`` is the gauge reference
        (φ = 0); each subsequent entry spawns one TEM/QTEM mode in
        input order.  When ``None``, the conductor groups are
        auto-derived from the mesh PEC mask on the port plane via
        :func:`extract_conductor_groups_from_mesh` — useful for
        OCC geometries (cylinders, arbitrary curved cross-sections)
        where the declarative ConductorSpec list is awkward.
    epsilon_r : float or None, default None
        Homogeneous-filling permittivity for the TEM path.  If
        ``None``, the QTEM dispatch is selected: the factory
        constructs the vacuum-reference mass matrix
        (:func:`build_M_eps_vacuum`) so the dual-Laplace solver can
        extract ``ε_eff = C' / C'_0``.
    n_modes : int, default 1
        Number of returned modes.  Up to ``K − 1`` these are the TEM
        (or QTEM) line modes; beyond that the port is extended by
        TE/TM curl-curl modes (homogeneous filling) or ζ-pencil
        hybrid modes (inhomogeneous filling) — see the class
        docstring.  The QTEM extension requires the requested modes
        to propagate at ``f_calc`` and raises with guidance
        otherwise.
    window : tuple of two corner points, optional
        Sub-rectangle of the face as two opposite corners in *global*
        tangential-axis ordering (:meth:`PortPlane.from_mesh`
        convention).  ``None`` (default) covers the whole face.  A PEC
        window boundary (via the legacy
        edge rule) joins the conductor-group graph as a boundary ring,
        so it can act as the ground conductor of an embedded port.

    Notes
    -----
    Conductor groups are auto-deduplicated in input order: nodes that
    belong to an earlier group are removed from later groups.  This
    matches the "ground first wins" semantic — a node on the ground
    plane is at φ = 0 even if a signal conductor's region spec
    nominally covers it.
    """

    name: str
    plane: BoxFace
    conductors: Optional[tuple[ConductorSpec, ...]] = None
    epsilon_r: Optional[float] = None
    n_modes: int = 1
    window: Optional[tuple] = None

    def __post_init__(self) -> None:
        if self.conductors is not None:
            if len(self.conductors) < 2:
                raise ValueError(
                    "PortSpecMultiConductor requires at least 2 conductors "
                    "(groups[0] = ground, groups[1:] = signal conductors); "
                    f"got {len(self.conductors)}."
                )
            max_modes = len(self.conductors) - 1
            if self.n_modes > max_modes:
                raise ValueError(
                    f"n_modes={self.n_modes} exceeds K-1 = {max_modes} for "
                    f"{len(self.conductors)} conductor groups."
                )
        if self.epsilon_r is not None and self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive (or None for QTEM).")
        if self.n_modes <= 0:
            raise ValueError("n_modes must be positive.")


PortSpec = Union[
    PortSpecCoax,
    PortSpecRectWG,
    PortSpecNumerical,
    PortSpecMultiConductor,
]


def _global_pair_to_uv(
    face: BoxFace,
    a_g: float,
    b_g: float,
) -> tuple[float, float]:
    """Map a (lower-tangent-axis, higher-tangent-axis) global pair to (u, v).

    The local ``(u, v)`` frame on a bbox face is chosen so that
    ``u × v`` points into the simulation domain (see
    ``PortPlane._UV_AXES``).  When that ordering matches the ascending
    global-axis order (``u_axis < v_axis``) the pair passes through; when
    it reverses (``u_axis > v_axis``, the case for X_MAX, Y_MIN, Z_MAX)
    the pair is swapped.
    """
    return (a_g, b_g) if face.u_axis < face.v_axis else (b_g, a_g)


def _build_lateral_pec_edge_mask(
    plane: PortPlane,
    mesh: Mesh,
    lateral_pec_faces: tuple[BoxFace, ...],
) -> np.ndarray:
    """Build the boolean PEC mask over ``[e_u | e_v]`` of a port plane.

    Each entry of ``lateral_pec_faces`` must be a bbox face whose
    normal axis is one of the port plane's tangential axes
    (``plane.face.u_axis`` or ``plane.face.v_axis``).  Edges lying on
    the wall coordinate (matching `_wall_pec_mask` from the unit tests)
    are marked True.

    Parameters
    ----------
    plane : PortPlane
        Port plane geometry.
    mesh : Mesh
        Mesh providing the bbox node coordinates.
    lateral_pec_faces : tuple[BoxFace, ...]
        Lateral PEC walls.

    Returns
    -------
    np.ndarray of bool, shape ``(N_u + N_v,)``
        True for E-edges tangent to a lateral PEC wall.

    Raises
    ------
    ValueError
        If a lateral face shares the port plane's normal axis (i.e.
        the face is the port plane itself or its opposite — neither is
        a *lateral* wall).
    """
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    mask = np.zeros(n_u + n_v, dtype=bool)
    if not lateral_pec_faces:
        return mask

    grid = mesh.grid
    nodes_per_axis = (grid.x, grid.y, grid.z)
    port_normal_axis = plane.face.normal_axis
    port_u_axis = plane.face.u_axis
    port_v_axis = plane.face.v_axis

    for face in lateral_pec_faces:
        n_axis = face.normal_axis
        if n_axis == port_normal_axis:
            raise ValueError(
                f"lateral_pec_faces entry {face!r} shares the port plane's "
                f"normal axis (axis {port_normal_axis}); a lateral wall must "
                f"be tangential to the port plane."
            )
        nodes = nodes_per_axis[n_axis]
        wall_coord = float(nodes[0] if not face.is_max else nodes[-1])
        extent = float(nodes[-1] - nodes[0])
        eps_tol = 1e-9 * extent if extent > 0 else 1e-15

        if n_axis == port_u_axis:
            # Wall normal along port u-axis ⇒ v-edges with their u-node on
            # the wall are tangential.  v-edge midpoints store (u_node,
            # v_centre) per ``PortPlane._build_uv_edges``.
            v_u = plane.v_edge_uv[:, 0]
            mask[n_u:] |= np.abs(v_u - wall_coord) < eps_tol
        elif n_axis == port_v_axis:
            # Wall normal along port v-axis ⇒ u-edges with their v-node on
            # the wall are tangential.  u-edge midpoints store (u_centre,
            # v_node).
            u_v = plane.u_edge_uv[:, 1]
            mask[:n_u] |= np.abs(u_v - wall_coord) < eps_tol
        else:  # pragma: no cover — should be unreachable per the check above
            raise RuntimeError(f"lateral face {face!r} has unexpected normal axis {n_axis}.")
    return mask


def _pec_faces_from_mask(mesh: Mesh) -> set[BoxFace]:
    """Bbox faces whose tangential edges are all PEC in the mesh mask.

    Input to :func:`resolve_port_edge_pec` for sub-face ports.  After
    the analysis-side BC-PEC consolidation (``Mesh.with_pec_boundaries``) a
    PEC boundary condition shows up here exactly like a geometric PEC
    wall (``background=pec``) — the mask is the single source of truth
    for "this wall is PEC".  Must be evaluated on the *unflattened*
    mask (the port-plane flatten erases the port face's own PEC-ness).
    """
    out: set[BoxFace] = set()
    for face in BoxFace:
        full = PortPlane.from_mesh(face, mesh)
        if _resolve_pec_edge_mask(full, mesh).all():
            out.add(face)
    return out


def _resolve_pec_edge_mask(plane: PortPlane, mesh: Mesh) -> np.ndarray:
    """Extract the boolean ``[e_u | e_v]`` PEC mask on the port plane.

    Per DD-046, the canonical source of all PEC information is
    ``mesh.pec_mask_edges``.  The 3D mask is sliced down to the port
    plane's tangential E-edges via ``plane.e_u_indices`` /
    ``plane.e_v_indices``.

    Fallback: when the resulting 2D mask is empty (typical for bare
    ``Mesh.from_grid`` setups without an OCC body or
    ``Mesh.with_pec_boundaries`` applied), all four lateral bbox faces
    are taken to be PEC walls — the standard hollow-waveguide
    assumption.  Production OCC runs populate
    ``mesh.pec_mask_edges`` and never reach the fallback.
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec_E_flat = np.concatenate(
        [
            mesh.pec_mask_edges[0, :n_Ex],
            mesh.pec_mask_edges[1, :n_Ey],
            mesh.pec_mask_edges[2, :n_Ez],
        ]
    )
    pec_mask = np.concatenate(
        [
            pec_E_flat[plane.e_u_indices],
            pec_E_flat[plane.e_v_indices],
        ]
    )
    if pec_mask.any():
        return pec_mask

    axis_to_min_face = {
        0: BoxFace.X_MIN,
        1: BoxFace.Y_MIN,
        2: BoxFace.Z_MIN,
    }
    axis_to_max_face = {
        0: BoxFace.X_MAX,
        1: BoxFace.Y_MAX,
        2: BoxFace.Z_MAX,
    }
    fallback_faces = (
        axis_to_min_face[plane.face.u_axis],
        axis_to_max_face[plane.face.u_axis],
        axis_to_min_face[plane.face.v_axis],
        axis_to_max_face[plane.face.v_axis],
    )
    return _build_lateral_pec_edge_mask(plane, mesh, fallback_faces)


def _build_tm_operators(
    plane: PortPlane,
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    c_3d,
    subface_edge_mask: Optional[np.ndarray],
):
    """TM eigenproblem inputs via :func:`build_2d_tm_curl_curl`.

    Passes the canonical 3D PEC mask when the port plane actually
    carries PEC information — the same trigger as the TE-path
    fallback in :func:`_resolve_pec_edge_mask` (which always returns
    a non-empty mask, so it cannot serve as the trigger itself);
    bare ``Mesh.from_grid`` setups fall back to the hollow-waveguide
    window boundary inside the builder.  Sub-face ports additionally
    Dirichlet-pin the normal-E edges on the window boundary — the
    port's virtual PEC frame, mirroring the ring OR-ed onto the
    transversal mask.
    """
    has_pec = bool(
        _flat_pec_on(mesh, plane.e_u_indices).any() or _flat_pec_on(mesh, plane.e_v_indices).any()
    )
    K_z, M_z, dirichlet, _ = build_2d_tm_curl_curl(
        plane,
        mesh.grid,
        m_eps,
        m_mu,
        c_3d,
        pec_mask_edges=mesh.pec_mask_edges if has_pec else None,
    )
    if subface_edge_mask is not None:
        dirichlet = dirichlet | _hollow_pec_node_mask(plane, mesh.grid)
    return K_z, M_z, dirichlet


def _complement_absorber_arrays(
    plane: PortPlane,
    mesh: Mesh,
    face,
    m_eps: np.ndarray,
    dt: float,
    subface_edge_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-edge Mur coefficients + live mask for the DD-096 complement
    absorber (``(r_u, r_v, live_u, live_v)``).

    ``r_p = (c_p dt - dx_n)/(c_p dt + dx_n)`` with the local phase
    velocity ``c_p = c0 / sqrt(eps_eff,p)`` from the exact per-edge
    ratio ``M_eps / M_eps_vacuum`` (the geometric ``A_dual/L`` factors
    cancel edge by edge — no free parameter).  Magnetic loading is not
    folded in: it would only shift the complement's absorption quality,
    not the stability class (WP-M2 exact evaluation).  The live mask
    zeroes the absorber on residual-PEC plane edges: lateral-wall edges
    (where the complement vanishes identically anyway — their interior
    companions are wall edges too) and sub-face window frames (whose
    interior companions are real field edges, so an unmasked absorber
    would write onto the virtual PEC frame).  Frozen conformal edges
    (``M_eps == 0``) join the dead set as well, mirroring the volume
    update's ``live_E = M_eps > 0`` convention: they carry no E update,
    and dividing through them would seed NaN coefficients.
    """
    m_vac = flatten_port_plane_mass(build_M_eps_vacuum(mesh), mesh, face)
    rs = []
    frozen = []
    for idx in (plane.e_u_indices, plane.e_v_indices):
        m_e = np.asarray(m_eps[idx], dtype=float)
        # DD-147/149 clamp degenerate conformal edges to exactly 0
        # without entering pec_mask_edges.
        dead = m_e <= 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            eps_eff = m_e / np.asarray(m_vac[idx], dtype=float)
            c_loc = C0 / np.sqrt(eps_eff)
            r = (c_loc * dt - plane.normal_dx) / (c_loc * dt + plane.normal_dx)
        rs.append(np.where(dead, 0.0, r))
        frozen.append(dead)
    pec = _flat_pec_on(mesh, np.concatenate([plane.e_u_indices, plane.e_v_indices]))
    pec = pec | np.concatenate(frozen)
    if subface_edge_mask is not None:
        pec = pec | subface_edge_mask
    n_u = plane.e_u_indices.size
    return (
        rs[0],
        rs[1],
        (~pec[:n_u]).astype(float),
        (~pec[n_u:]).astype(float),
    )


def validate_absorbing_face_window(
    face, plane: PortPlane, mesh: Mesh, *, whole_face: bool, absorbing: bool | None = None
) -> None:
    """Reject a modal port on an absorbing face unless it is a guide end.

    A port embedded in a CPML face is meaningful only as the end of a
    conductor-enclosed guide — the waveguide neck of a horn, a coax
    entering the box — because the absorber is switched off in the
    columns behind the window (DD-198) and the lateral edge of that
    switch-off must fall on conductor, not on free space, where it would
    scatter.  Two rules follow: the port needs a window, and every edge
    of the window ring must be a PEC edge of the mesh on the port slab.
    Faces that are not absorbing pass unchanged.

    Enforced where port and absorber meet — the time-domain solver's
    setup and the declarative ``PortWaveguide`` resolution; a port-only
    mode solve on an open cross-section (spec level) is not affected.
    *absorbing* may be given by a caller that already knows the face's
    boundary type; otherwise it is read from the mesh.
    """
    from magnelio.boundaries.boundary_conditions import bc_type_entries  # noqa: PLC0415

    key = face.value.replace("_", "")
    if absorbing is None:
        bc = getattr(mesh, "boundary_conditions", None)
        if bc is None:
            return
        try:
            absorbing = bc_type_entries(bc).get(key) == "CPML"
        except (TypeError, ValueError):
            return
    if not absorbing:
        return
    if whole_face:
        raise ValueError(
            f"port on the absorbing (CPML) face {key!r} covers the whole face. "
            f"A waveguide port in an absorbing wall must be the end of a "
            f"conductor-enclosed guide: give the port a window (corners=) "
            f"that matches the guide's cross-section, or declare the face "
            f"PEC."
        )
    ring = build_port_edge_pec_mask(
        plane, {"u_min": True, "u_max": True, "v_min": True, "v_max": True}
    )
    conductor = _flat_pec_on(mesh, np.concatenate([plane.e_u_indices, plane.e_v_indices]))
    missing = int(np.count_nonzero(ring & ~conductor))
    if missing:
        raise ValueError(
            f"port window on the absorbing (CPML) face {key!r} is not enclosed "
            f"by conductor: {missing} of {int(ring.sum())} window-ring edges "
            f"lie in free space.  The absorber is switched off behind the "
            f"window, so its lateral edge must fall on the guide's walls — "
            f"align the window corners with the conductor walls that reach "
            f"the face (the window snaps to grid nodes; refine the mesh if "
            f"a wall falls between nodes), or declare the face PEC."
        )


def _flat_pec_on(mesh: Mesh, indices: np.ndarray) -> np.ndarray:
    """Slice ``mesh.pec_mask_edges`` (flat-E ordering) at ``indices``."""
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    flat = np.concatenate(
        [
            mesh.pec_mask_edges[0, :n_Ex],
            mesh.pec_mask_edges[1, :n_Ey],
            mesh.pec_mask_edges[2, :n_Ez],
        ]
    )
    return flat[indices]


def _resolve_conductors(
    conductors: tuple[ConductorSpec, ...],
    plane: PortPlane,
    mesh: Mesh,
) -> list[np.ndarray]:
    """Translate a :class:`PortSpecMultiConductor.conductors` tuple into
    local 2D node-index arrays matching :func:`build_2d_gradient`'s
    ``primal_2d_node_indices`` basis (``i_u * Nv_node + i_v``).

    Performs:
    - geometric resolution of each :class:`ConductorSpec`,
    - empty-resolution check (a conductor that yields zero nodes
      probably reflects a typo in the user's coordinate ranges or a
      face-name mistake — fail loud),
    - "ground first wins" deduplication across input order,
    - post-deduplication empty-group check (a fully-shadowed signal
      conductor likewise fails loud).
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    nodes_per_axis = (mesh.grid.x, mesh.grid.y, mesh.grid.z)
    n_nodes_per_axis = (Nx + 1, Ny + 1, Nz + 1)

    n_axis = plane.face.normal_axis
    u_axis = plane.face.u_axis
    v_axis = plane.face.v_axis
    Nu_node = n_nodes_per_axis[u_axis]
    Nv_node = n_nodes_per_axis[v_axis]

    iu_arr, iv_arr = np.meshgrid(
        np.arange(Nu_node),
        np.arange(Nv_node),
        indexing="ij",
    )
    iu_flat = iu_arr.ravel()
    iv_flat = iv_arr.ravel()
    local_idx = iu_flat * Nv_node + iv_flat
    u_at_node = nodes_per_axis[u_axis][iu_flat]
    v_at_node = nodes_per_axis[v_axis][iv_flat]

    # Relative to the port-plane extents (DD-120) — the same form the
    # lateral-PEC-wall matching below already uses; an absolute 1e-9 m
    # would be a whole cell at optics scale.
    u_ext = float(u_at_node.max() - u_at_node.min()) if u_at_node.size else 0.0
    v_ext = float(v_at_node.max() - v_at_node.min()) if v_at_node.size else 0.0
    extent = max(u_ext, v_ext)
    eps_tol = 1e-9 * extent if extent > 0 else 1e-15
    raw_groups: list[np.ndarray] = []

    for k, cs in enumerate(conductors):
        if isinstance(cs, BboxLateralConductor):
            mask = (
                (iu_flat == 0)
                | (iu_flat == Nu_node - 1)
                | (iv_flat == 0)
                | (iv_flat == Nv_node - 1)
            )
        elif isinstance(cs, WallConductor):
            wall_axis = cs.face.normal_axis
            wall_is_max = cs.face.is_max
            if wall_axis == n_axis:
                raise ValueError(
                    f"WallConductor face {cs.face!r} shares the port "
                    f"plane's normal axis (axis {n_axis}); a wall "
                    f"must be lateral to the port plane."
                )
            if wall_axis == u_axis:
                mask = iu_flat == (Nu_node - 1 if wall_is_max else 0)
            else:  # wall_axis == v_axis
                mask = iv_flat == (Nv_node - 1 if wall_is_max else 0)
        elif isinstance(cs, RegionConductor):
            a_lo, a_hi = cs.axis_a_range
            b_lo, b_hi = cs.axis_b_range
            # Map global-axis ranges to (u, v) per the face's u/v
            # assignment.  axis_a is the lower-numbered global tangent
            # axis; axis_b is the higher.  For X_MIN (u=y, v=z):
            # u_axis < v_axis ⇒ pass through.  For X_MAX (u=z, v=y):
            # u_axis > v_axis ⇒ swap.
            if u_axis < v_axis:
                u_lo, u_hi = a_lo, a_hi
                v_lo, v_hi = b_lo, b_hi
            else:
                u_lo, u_hi = b_lo, b_hi
                v_lo, v_hi = a_lo, a_hi
            mask = (
                (u_at_node >= u_lo - eps_tol)
                & (u_at_node <= u_hi + eps_tol)
                & (v_at_node >= v_lo - eps_tol)
                & (v_at_node <= v_hi + eps_tol)
            )
        else:
            raise TypeError(f"unsupported ConductorSpec type: {type(cs).__name__}")

        nodes = local_idx[mask].astype(np.int64)
        if nodes.size == 0:
            raise ValueError(
                f"conductors[{k}] = {cs!r} yields zero 2D nodes on the "
                f"port plane — check coordinates/face."
            )
        raw_groups.append(nodes)

    # Deduplicate in input order ("ground first wins"): nodes belonging
    # to an earlier group are removed from later groups.  Re-check that
    # no group ends up empty after deduplication.
    seen = np.zeros(Nu_node * Nv_node, dtype=bool)
    final_groups: list[np.ndarray] = []
    for k, g in enumerate(raw_groups):
        keep_mask = ~seen[g]
        cleaned = g[keep_mask]
        if cleaned.size == 0:
            raise ValueError(
                f"conductors[{k}] = {conductors[k]!r} is fully shadowed "
                f"by earlier conductors — yields zero exclusive nodes "
                f"after deduplication."
            )
        seen[cleaned] = True
        final_groups.append(cleaned)
    return final_groups


def _validate_three_equidistant_cells(
    face: BoxFace,
    mesh: Mesh,
    *,
    rtol: float = 1e-6,
) -> None:
    """Enforce the §2.4 "three equidistant cells at the port" rule.

    The modal-port operator (DD-040) is implemented as a difference
    operator between the port plane and the next interior plane along
    the port-normal axis.  ``reference_waveguide_ports.md`` §2.4
    requires the **three cells immediately adjacent
    to the port** to be equidistant — otherwise the V/I projection
    silently scales by ~10⁶ while the time-domain wave still
    propagates physically.  ``Mesh.from_geometry`` guarantees this
    buffer on every face carrying a declared port (DD-109) — and on
    all six faces when the model declares none (DD-107 fallback) — so
    a failure here means the port landed on an unbuffered face (grid
    not from ``from_geometry``, port declared only at analysis time on
    a selectively buffered mesh, or a hard ``min_cell_size`` floor).

    Parameters
    ----------
    face : BoxFace
        Bbox face the port lives on; determines which axis and end
        of the grid to inspect.
    mesh : Mesh
        FIT mesh whose ``grid.dx``/``dy``/``dz`` give the per-cell
        widths along each axis.
    rtol : float, default 1e-6
        Allowed ``(max - min) / min`` ratio across the three port
        cells before the rule is considered violated.

    Raises
    ------
    ValueError
        If the three port-adjacent cells along the normal axis are
        not equidistant within ``rtol``, or if the grid has fewer
        than 3 cells along that axis.
    """
    grid = mesh.grid
    deltas_normal = (grid.dx, grid.dy, grid.dz)[face.normal_axis]
    if deltas_normal.size < 3:
        raise ValueError(
            f"port {face.value!r}: at least 3 cells required along the "
            f"port-normal axis; got {deltas_normal.size} "
            f"(reference_waveguide_ports.md §2.4)."
        )
    if face.is_max:
        port_cells = deltas_normal[-3:]
        slice_label = f"dx[{deltas_normal.size - 3}..{deltas_normal.size - 1}]"
    else:
        port_cells = deltas_normal[:3]
        slice_label = "dx[0..2]"
    cmin = float(np.min(port_cells))
    cmax = float(np.max(port_cells))
    if cmin <= 0.0:
        raise ValueError(
            f"port {face.value!r}: non-positive cell width in port-adjacent "
            f"slice {slice_label} = {port_cells.tolist()}."
        )
    if (cmax - cmin) / cmin > rtol:
        # Port declaration on the model: DD-109.
        raise ValueError(
            f"port {face.value!r}: the three cells immediately adjacent to "
            f"the port plane (axis {face.normal_axis}, slice {slice_label}) "
            f"must be equidistant within rtol={rtol:.0e} — "
            f"reference_waveguide_ports.md §2.4 'three equidistant cells'.  "
            f"Got widths [{', '.join(f'{c * 1e3:.4f} mm' for c in port_cells)}] "
            f"(ratio max/min = {cmax / cmin:.4f}).  Without this buffer the "
            f"modal-port V/I projection scales by orders of magnitude, and "
            f"it does so silently.  Declare the port on the model before "
            f"meshing (GeometryModel.add_port) so "
            f"Mesh.from_geometry buffers this face — unless a "
            f"min_cell_size floor overrides it."
        )


def _conductor_centroid_uv(
    nodes: np.ndarray,
    plane: PortPlane,
    mesh: Mesh,
) -> tuple[float, float]:
    """Return the (u, v) centroid of a conductor node group.

    ``nodes`` is a 1-D array of local 2D primal-node indices in the
    ``i_u * Nv_node + i_v`` raster used by
    :func:`extract_conductor_groups_from_mesh`.
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    Nv_node = (Nx, Ny, Nz)[plane.face.v_axis] + 1
    grid = mesh.grid
    u_coord = (grid.x, grid.y, grid.z)[plane.face.u_axis]
    v_coord = (grid.x, grid.y, grid.z)[plane.face.v_axis]
    ius = nodes // Nv_node
    ivs = nodes % Nv_node
    return float(np.mean(u_coord[ius])), float(np.mean(v_coord[ivs]))


def _validate_coax_geometry(
    plane: PortPlane,
    mesh: Mesh,
    conductor_groups: list[np.ndarray],
    spec_center_uv: tuple[float, float],
) -> None:
    """Cross-check PortSpecCoax against auto-detected conductors.

    Per DD-048 a PortSpecCoax runs both an analytical reference path
    and an operator-consistent FIT path.  The FIT path auto-detects
    conductor groups from the staircased PEC mask at the port plane;
    we verify that exactly two groups exist and that both centroids
    coincide with ``spec.center`` to within one mesh cell.  Otherwise
    the spec geometry does not match the meshed geometry and the
    factory raises rather than silently producing a wrong mode.
    """
    if len(conductor_groups) != 2:
        # Two-path (analytical + FIT) architecture: DD-048.
        raise ValueError(
            f"PortSpecCoax on {plane.face.value!r}: auto-detection found "
            f"{len(conductor_groups)} conductor group(s); expected 2 "
            f"(inner + outer).  Likely cause: mesh too coarse, inner "
            f"conductor not resolved, or the geometry on this face is "
            f"not a coax."
        )
    grid = mesh.grid
    du = (grid.dx, grid.dy, grid.dz)[plane.face.u_axis]
    dv = (grid.dx, grid.dy, grid.dz)[plane.face.v_axis]
    cell_tol = max(float(np.max(du)), float(np.max(dv)))
    u_c_spec, v_c_spec = spec_center_uv
    for k, group in enumerate(conductor_groups):
        u_c, v_c = _conductor_centroid_uv(group, plane, mesh)
        delta = math.hypot(u_c - u_c_spec, v_c - v_c_spec)
        if delta > cell_tol:
            label = "ground (outer)" if k == 0 else f"signal_{k} (inner)"
            # Two-path (analytical + FIT) architecture: DD-048.
            raise ValueError(
                f"PortSpecCoax on {plane.face.value!r}: detected "
                f"{label} centroid ({u_c * 1e3:.4f}, {v_c * 1e3:.4f}) mm "
                f"differs from spec.center "
                f"({u_c_spec * 1e3:.4f}, {v_c_spec * 1e3:.4f}) mm by "
                f"{delta * 1e3:.4f} mm — exceeds one mesh cell "
                f"({cell_tol * 1e3:.4f} mm).  Likely cause: mesh too "
                f"coarse around the conductor or coax displaced from "
                f"spec."
            )


_DEGENERATE_CUTOFF_RTOL = 1e-3


def _qtem_zeta_hybrid_modes(
    line_modes: list,
    plane,
    mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    c_3d,
    dt: float,
    f_calc: float,
    n_higher: int,
    name: str,
) -> list:
    """WP-U6: the ``n_higher`` lowest hybrid channels from the ζ-pencil.

    On an inhomogeneous cross-section no exact TEM/TE/TM split exists —
    the higher modes are hybrid with frequency-dependent profiles.  The
    correct discrete objects are the eigenpairs of the DD-056 ζ-pencil
    of the *production* matrices at ``f_calc``: this helper finds every
    mode propagating at ``f_calc``, drops the line-mode family (each
    Laplace mode's pencil counterpart, identified by the W_t overlap),
    and wraps the next ``n_higher`` eigenpairs — descending phase
    advance = ascending mode order — in ``DiscreteMode``-compatible
    ``Mode`` objects:

    * profile: the DD-056 real gauge (tangential trace real to ~1e-13),
      M_ε-normalised on the port plane;
    * ``omega_c = q_eff/dt`` — the frequency-local Klein-Gordon fit of
      the channel (metadata + the dispersive Mur ``v_p``; the DD-064
      lesson applies: it is a *local* estimate, never a band-wide
      cut-off claim, and no termination is fitted from it);
    * ``epsilon_r`` chosen so that ``Mode.gamma(2π f_calc)`` equals the
      exact discrete ``β = θ/dz`` of the eigenvalue — Mur-1st then uses
      the channel's true phase velocity at ``f_calc``;
    * ``mode_type = TEM`` with ``z_line = None``: the V/I norm is the
      frequency-flat ``η₀/√ε_r`` (QTEM convention).  A TE-form
      ``Z(ω)`` would diverge at the *estimated* ``f̂_c`` — exactly the
      DD-064 artificial-cut-off failure mode — so the smooth flat norm
      is deliberate.

    The hybrid profiles are mutually non-M_ε-orthogonal and overlap
    the line modes; the caller builds dual-basis projectors over ALL
    channels (the DD-056 machinery), so projections stay cross-talk
    free.  Termination follows DD-064 defaults: the pair-product /
    slab certificates fail on inhomogeneous fillings, so every channel
    runs modal Mur-1st with the loud verbose notice;
    ``port_model="band"`` remains the reflection-critical opt-in.

    Raises
    ------
    ValueError
        If fewer than ``n_higher`` hybrid modes propagate at
        ``f_calc`` (guidance: raise ``f_calc``/``f_max`` or reduce
        ``n_modes``), or if a profile fails the real-gauge check
        (degenerate complex pair).
    """
    from magnelio.constants import C0, ETA0
    from magnelio.ports._modal.mode import Mode
    from magnelio.ports._modal.tem_laplace import (
        travelling_wave_h_profiles,
    )
    from magnelio.ports._modal.zeta_pencil import (
        build_period_blocks,
        find_propagating_modes,
        make_channel,
        profile_reality,
    )

    _validate_three_equidistant_cells(plane.face, mesh)
    chain = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)
    w_dt = 2.0 * math.pi * f_calc * dt
    dz = float(plane.normal_dx)

    # The DC eps_eff underestimates the fundamental's phase advance
    # (normal dispersion) — 50 % arc margin.
    eps_hint = max(float(m.epsilon_r) for m in line_modes)
    theta0 = 2.0 * math.pi * f_calc * math.sqrt(eps_hint) / C0 * dz
    zp, pp = find_propagating_modes(chain, w_dt, 1.5 * theta0)

    # Drop the line-mode family: each Laplace mode's pencil
    # counterpart is the eigenvector with the largest W_t overlap.
    w_t = chain.w_period[: chain.n_t]
    keep = list(range(zp.size))
    for lap in line_modes:
        if not keep:
            break
        track = np.concatenate(
            [
                np.asarray(lap.discrete_e_u_profile)[chain.free_u],
                np.asarray(lap.discrete_e_v_profile)[chain.free_v],
            ]
        )
        ov = np.abs(track @ (w_t[:, None] * pp[: chain.n_t, keep]))
        keep.pop(int(np.argmax(ov)))
    if len(keep) < n_higher:
        raise ValueError(
            f"Modal port {name!r}: n_modes requests {n_higher} "
            f"hybrid channel(s) beyond the {len(line_modes)} QTEM "
            f"line mode(s), but only {len(keep)} additional mode(s) "
            f"propagate at f_calc = {f_calc / 1e9:.3f} GHz on this "
            f"cross-section.  Raise f_calc/f_max above the intended "
            f"band top or reduce n_modes (evanescent-at-f_calc "
            f"channels are not supported)."
        )

    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    me_u = np.asarray(m_eps, dtype=float)[plane.e_u_indices]
    me_v = np.asarray(m_eps, dtype=float)[plane.e_v_indices]
    nu_free = int(chain.free_u.sum())
    omega_calc = 2.0 * math.pi * f_calc

    hybrids: list = []
    for c, j in enumerate(keep[:n_higher]):
        ch = make_channel(complex(zp[j]), pp[:, j], chain, w_dt, dz)
        reality = profile_reality(ch.phi_trace, chain.n_t)
        if reality > 1e-6:
            raise ValueError(
                f"Modal port {name!r}: hybrid channel {c} tangential "
                f"profile not real after gauge fixing (residual "
                f"{reality:.2e}) — degenerate or complex mode pair; "
                f"not certified"
            )
        eu = np.zeros(n_u)
        ev = np.zeros(n_v)
        eu[chain.free_u] = ch.phi_trace[:nu_free].real
        ev[chain.free_v] = ch.phi_trace[nu_free : chain.n_t].real
        nrm = math.sqrt(float(np.dot(me_u, eu**2) + np.dot(me_v, ev**2)))
        eu /= nrm
        ev /= nrm
        theta = abs(np.angle(ch.zeta))
        beta = theta / dz
        omega_c = ch.q / dt
        # gamma(omega_calc) = j*beta exactly (Mode.gamma solves the
        # KG dispersion with this eps_r and omega_c).
        eps_r_kg = (C0 * beta) ** 2 / (omega_calc**2 - omega_c**2)
        h_u, h_v = travelling_wave_h_profiles(
            eu,
            ev,
            plane,
            m_mu,
            ETA0 / math.sqrt(eps_r_kg),
        )
        hybrids.append(
            Mode(
                name=f"HYB_zp{c:02d}",
                mode_type=ModeType.TEM,
                omega_c=omega_c,
                epsilon_r=eps_r_kg,
                field_evaluator=None,
                z_line=None,
                discrete_e_u_profile=eu,
                discrete_e_v_profile=ev,
                discrete_h_u_profile=h_u,
                discrete_h_v_profile=h_v,
            )
        )
    # Report in ascending frequency-local cut-off (the phase-advance
    # order at f_calc need not be monotone in f_c_hat — dispersion
    # differs per hybrid); relabel to match the final positions.
    hybrids.sort(key=lambda m: m.omega_c)
    hybrids = [dataclasses.replace(m, name=f"HYB_zp{c:02d}") for c, m in enumerate(hybrids)]
    return hybrids


def _port_chain_slab_defect(mesh, m_eps, m_mu, face) -> float:
    """Max relative slab deviation of the masses along the first feed cells.

    Certificate stage 2 (DD-067).  The transversal pair-product gate
    in ``PortOperatorModal._chain_params`` cannot see the
    *normal-face* M_mu (it enters the TE transversal curl-curl
    operator but forms no co-located pair) — the DD-066 conformal-coax
    finding: a 36 % boundary-slab Hz-M_mu deviation certified as
    "uniform" while the TE11 channel reflected at -42 dB.  This
    function measures the quantity the exact DTBC actually requires:
    every mass entry feeding the port's 2D mode solve must equal its
    continuation into the feed.  For each E- and H-component the
    port-side slab (after the factory flattens) is compared with the
    first and second interior slabs along the port normal — components
    sampled ON grid planes (size N+1 along the normal) compare slab
    indices (0,1) and (1,2), components sampled at cell layers
    (size N) compare layers (0,1) and (1,2).  Entries that are zero
    in both compared slabs are skipped (PEC-masked edges hold unread
    values only when masked in *all* compared slabs; donated faces
    with M_mu = 0 match by construction on an invariant feed).

    A z-translation-invariant feed measures ~1e-15 here.  Anything
    above ``_DTBC_SLAB_DEFECT_TOL`` sends the port's channels to Mur
    (the ``PortOperatorModal`` veto).
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    comps = [
        m_eps[:n_Ex].reshape(Nx, Ny + 1, Nz + 1),
        m_eps[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1),
        m_eps[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz),
        m_mu[:n_Hx].reshape(Nx + 1, Ny, Nz),
        m_mu[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz),
        m_mu[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1),
    ]
    axis = face.normal_axis
    defect = 0.0
    for arr in comps:
        n_along = arr.shape[axis]
        idx = (
            range(n_along - 1, max(n_along - 4, -1), -1)
            if face.is_max
            else range(0, min(3, n_along))
        )
        slabs = [np.take(arr, i, axis=axis) for i in idx]
        for a, b in zip(slabs, slabs[1:]):
            mask = (a != 0.0) | (b != 0.0)
            if not mask.any():
                continue
            denom = np.maximum(np.abs(a[mask]), np.abs(b[mask]))
            defect = max(defect, float(np.max(np.abs(a[mask] - b[mask]) / denom)))
    return defect


def _fix_degenerate_polarisation_gauge(
    modes: list,
    *,
    rtol: float = _DEGENERATE_CUTOFF_RTOL,
) -> list:
    """Deterministic basis inside degenerate numerical mode groups (WP-U4).

    An eigensolver returns an arbitrary orthogonal rotation of each
    degenerate eigenspace (e.g. the two coax TE11 polarisations), so
    per-channel example output and regression pins would depend on
    mesh and ARPACK start-vector noise.  Within every group of
    same-family *numerical* modes whose cut-offs agree within ``rtol``
    the basis is rotated to the principal axes of the u-edge energy
    form ``⟨e_u, e_u⟩`` on the group's span — ordered by descending
    u-energy, sign fixed by the largest-|.| ``e_u`` entry (falling
    back to ``e_v`` for u-degenerate axes).  An orthogonal rotation
    within a degenerate eigenspace preserves the M_ε-orthonormality
    and the per-channel physics; only the reporting basis becomes
    reproducible.  TEM channels (``omega_c = 0``) are excluded — the
    multi-conductor Gram-eigenbasis carries distinct line impedances
    and is already deterministic.  Analytical-path modes (no discrete
    profiles) pass through unchanged.
    """
    modes = list(modes)
    i = 0
    while i < len(modes):
        j = i + 1
        while (
            j < len(modes)
            and modes[i].omega_c > 0.0
            and modes[j].mode_type == modes[i].mode_type
            and modes[i].discrete_e_u_profile is not None
            and modes[j].discrete_e_u_profile is not None
            and abs(modes[j].omega_c - modes[i].omega_c) < rtol * abs(modes[j].omega_c)
        ):
            j += 1
        n_grp = j - i
        if n_grp >= 2:
            group = modes[i:j]
            E_u = np.stack([m.discrete_e_u_profile for m in group])
            E_v = np.stack([m.discrete_e_v_profile for m in group])
            H_u = np.stack([m.discrete_h_u_profile for m in group])
            H_v = np.stack([m.discrete_h_v_profile for m in group])
            form_u = E_u @ E_u.T
            vals, R = np.linalg.eigh(form_u)
            order = np.argsort(vals)[::-1]  # descending u-energy
            R = R[:, order]
            new_eu = R.T @ E_u
            new_ev = R.T @ E_v
            new_hu = R.T @ H_u
            new_hv = R.T @ H_v
            for k in range(n_grp):
                anchor = new_eu[k]
                if not np.any(anchor):
                    anchor = new_ev[k]
                if anchor[int(np.argmax(np.abs(anchor)))] < 0.0:
                    new_eu[k] = -new_eu[k]
                    new_ev[k] = -new_ev[k]
                    new_hu[k] = -new_hu[k]
                    new_hv[k] = -new_hv[k]
                modes[i + k] = dataclasses.replace(
                    group[k],
                    discrete_e_u_profile=new_eu[k],
                    discrete_e_v_profile=new_ev[k],
                    discrete_h_u_profile=new_hu[k],
                    discrete_h_v_profile=new_hv[k],
                )
        i = j
    return modes


def _warn_on_degenerate_cutoffs(
    modes: list,
    *,
    name: str,
    rtol: float = _DEGENERATE_CUTOFF_RTOL,
) -> None:
    """Emit a UserWarning if two modes share a cut-off within ``rtol``.

    Two propagating modes with (nearly) identical cut-off frequencies form
    a degenerate eigenpair on the port plane.  The classical Poynting-flux
    orthogonality, which rests on distinct propagation constants, breaks
    down at exact degeneracy: the modes pick up a cross-coupling at the
    modal port that does not vanish under mesh refinement.  Whether
    perturbing the geometry separates the cut-offs depends on the source
    of the degeneracy — TE_mn/TM_mn pairs in a homogeneously filled PEC
    rectangle are intrinsically degenerate (they share
    ``fc² = (mπ/a)² + (nπ/b)²`` regardless of a/b), and only an
    inhomogeneous filling or a non-rectangular cross-section can split
    them.  TE_mn/TE_nm pairs in a square cross-section (a = b), in
    contrast, do separate under a ~0.1 % size perturbation.

    The warning fires only if at least one of the two modes has a
    non-zero cut-off — pure TEM modes (``omega_c = 0``) are excluded.
    """
    sorted_modes = sorted(modes, key=lambda m: m.omega_c)
    for prev, curr in zip(sorted_modes, sorted_modes[1:]):
        if prev.omega_c <= 0.0 or curr.omega_c <= 0.0:
            continue
        delta_rel = abs(curr.omega_c - prev.omega_c) / curr.omega_c
        if delta_rel < rtol:
            f_prev = prev.omega_c / (2.0 * math.pi)
            f_curr = curr.omega_c / (2.0 * math.pi)
            warnings.warn(
                f"Modal port '{name}': modes '{prev.name}' "
                f"(fc = {f_prev / 1e9:.4f} GHz) and '{curr.name}' "
                f"(fc = {f_curr / 1e9:.4f} GHz) are degenerate within "
                f"{delta_rel * 100:.3f}% relative cut-off difference.  Their "
                f"power-flux inner product is not orthogonal under "
                f"degeneracy, so non-trivial cross-coupling at the modal "
                f"port is expected and does not vanish with mesh refinement. "
                f"Either restrict the operating band so the degenerate pair "
                f"stays evanescent, or break the underlying symmetry of the "
                f"cross-section (e.g. by inhomogeneous filling or by "
                f"choosing a non-rectangular geometry).",
                UserWarning,
                stacklevel=3,
            )


def build_modal_port(
    spec: PortSpec,
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    *,
    dt: float,
    f_calc: float,
) -> PortOperatorModal:
    """Build a :class:`PortOperatorModal` from a port spec.

    Parameters
    ----------
    spec : PortSpecCoax or PortSpecRectWG
        Face-agnostic port description (geometry, modes, optional
        excitation).
    mesh : Mesh
        FIT mesh; used to read the bbox-face coordinate and the edge
        layout via :meth:`PortPlane.from_mesh`.
    m_eps, m_mu : np.ndarray
        Diagonal-FIT material-matrix vectors.  Used for the M_ε
        Gram-Schmidt in :func:`discretize_modes`, the V/I-projection
        weights in the operator, and the V/I calibration.
    dt : float
        Solver time step [s].
    f_calc : float
        Mode-calculation frequency [Hz].  Sets the Mur reflection
        coefficient and the per-mode phase velocity used by the TF/SF
        soft source.

    Returns
    -------
    PortOperatorModal
        Operator ready to be passed to ``FITTimeDomainSolver`` via the
        unified ``ports=`` argument.
    """
    if not isinstance(
        spec,
        (
            PortSpecCoax,
            PortSpecRectWG,
            PortSpecNumerical,
            PortSpecMultiConductor,
        ),
    ):
        raise TypeError(f"unsupported port spec type: {type(spec).__name__}")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if f_calc <= 0.0:
        raise ValueError("f_calc must be positive")

    _validate_three_equidistant_cells(spec.plane, mesh)

    # Sub-face ports: the plane and its window-boundary Dirichlet mask
    # (legacy edge rule — domain-boundary edge inherits that wall's
    # BC, interior edge inherits the port face's BC) must be resolved
    # *before* the flatten below erases the port face's own PEC-ness
    # from the mask.
    plane = PortPlane.from_mesh(
        spec.plane,
        mesh,
        window=getattr(spec, "window", None),
    )
    subface_edge_mask: Optional[np.ndarray] = None
    if getattr(spec, "window", None) is not None:
        edge_pec = resolve_port_edge_pec(
            plane,
            mesh,
            _pec_faces_from_mask(mesh),
        )
        subface_edge_mask = build_port_edge_pec_mask(plane, edge_pec)

    # Flatten the port-plane mass slab AND the PEC-mask slab in-line
    # with the first interior slab.  Both flattens close longitudinal
    # jumps that the conformal corrections introduce at the bbox
    # boundary (where the cell neighbourhood is halved).  The PEC-mask
    # flatten is applied to a BUILDER-LOCAL mesh view so the downstream
    # ``extract_conductor_groups_from_mesh`` call below sees the same
    # conductor contour the volume wave will see — never written back
    # into the caller's mesh: a later operator build (second run, other
    # port) would then find its plane pre-stripped of the wall contour
    # and detect the wrong boundary conditions.
    m_eps = flatten_port_plane_mass(m_eps, mesh, spec.plane)
    m_mu = flatten_port_plane_mu(m_mu, mesh, spec.plane)
    mesh = dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(
            mesh.pec_mask_edges,
            mesh,
            spec.plane,
        ),
    )

    omega_calc = 2.0 * math.pi * f_calc
    # Set by the multi-conductor branch when zeta-pencil hybrid
    # channels join the port (WP-U6): their profiles are not
    # M_eps-orthogonal, so project_V goes dual-basis.
    qtem_multimode = False

    if isinstance(spec, PortSpecCoax):
        u_c, v_c = _global_pair_to_uv(
            spec.plane,
            spec.center[0],
            spec.center[1],
        )

        # Path (a): analytical reference (DD-048).  The closed-form 1/r
        # mode gives the continuous-geometry Z_line as a design target.
        ref_modes = CoaxAnalyticalModeSolver(
            inner_radius=spec.inner_radius,
            outer_radius=spec.outer_radius,
            epsilon_r=spec.epsilon_r,
            center=(u_c, v_c),
        ).solve(n_modes=spec.n_modes, f_calc=f_calc)
        z_line_ref = ref_modes[0].z_line

        # Path (b): operator-consistent FIT-mode (DD-048).  Auto-detect
        # the two conductor groups from the staircased PEC mask on the
        # port slice; cross-check centroids against spec.center; solve
        # the 2D Laplace problem on the same M_eps that drives the
        # 3D volume operator.
        conductor_groups = extract_conductor_groups_from_mesh(plane, mesh)
        _validate_coax_geometry(
            plane,
            mesh,
            conductor_groups,
            (u_c, v_c),
        )
        c_3d = build_curl_matrix(mesh.grid)
        g_3d = build_gradient_matrix(mesh.grid)
        _, M_2d, _ = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps,
            m_mu,
            c_3d,
        )
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
        modes = solve_tem_laplace(
            plane,
            g_2d,
            M_2d,
            conductor_groups,
            spec.epsilon_r,
            grid=mesh.grid,
            m_mu_flat=m_mu,
            boundary_conditions=mesh.boundary_conditions,
        )[: spec.n_modes]

        port_report = PortOperatorReport(
            z_line_num=modes[0].z_line,
            z_line_ref=z_line_ref,
        )
    elif isinstance(spec, PortSpecRectWG):
        u_c, v_c = _global_pair_to_uv(
            spec.plane,
            spec.center[0],
            spec.center[1],
        )
        width_a_uv, height_b_uv = _global_pair_to_uv(
            spec.plane,
            spec.width_a,
            spec.height_b,
        )

        # Path (a): analytical reference (DD-048).
        ref_modes = RectWGAnalyticalModeSolver(
            width_a=width_a_uv,
            height_b=height_b_uv,
            epsilon_r=spec.epsilon_r,
            center=(u_c, v_c),
        ).solve(n_modes=spec.n_modes, f_calc=f_calc)
        cutoff_ref = ref_modes[0].omega_c / (2.0 * math.pi)
        n_te = sum(1 for m in ref_modes if m.mode_type is ModeType.TE)
        n_tm = sum(1 for m in ref_modes if m.mode_type is ModeType.TM)

        # Path (b): operator-consistent FIT-mode (DD-048).  PEC edge
        # mask comes from the staircased mesh PEC mask on the port
        # slice (DD-046 / DD-050), so the 2D mode solver sees the same
        # wall contour as the 3D volume operator.
        pec_mask = _resolve_pec_edge_mask(plane, mesh)

        c_3d = build_curl_matrix(mesh.grid)
        K, M, primal_2d = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps,
            m_mu,
            c_3d,
        )
        te_modes: list = []
        tm_modes: list = []
        if n_te > 0:
            te_modes = Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                pec_edge_mask=pec_mask,
                epsilon_r=spec.epsilon_r,
                mode_type=ModeType.TE,
                m_mu_flat=m_mu,
            ).solve(n_modes=n_te, f_calc=f_calc)
        if n_tm > 0:
            g_3d = build_gradient_matrix(mesh.grid)
            g_2d_tm, _, _ = build_2d_gradient(
                plane,
                mesh.grid,
                g_3d,
            )
            L_node, M_node, pec_node_mask = _build_tm_operators(
                plane,
                mesh,
                m_eps,
                m_mu,
                c_3d,
                subface_edge_mask=None,
            )
            tm_modes = Numerical2DModeSolver(
                plane=plane,
                K=K,
                M=M,
                primal_2d_indices=primal_2d,
                pec_edge_mask=pec_mask,
                epsilon_r=spec.epsilon_r,
                mode_type=ModeType.TM,
                m_mu_flat=m_mu,
                g_2d=g_2d_tm,
                L_node=L_node,
                M_node=M_node,
                pec_node_mask=pec_node_mask,
            ).solve(n_modes=n_tm, f_calc=f_calc)
        modes = sorted(te_modes + tm_modes, key=lambda m: m.omega_c)
        modes = modes[: spec.n_modes]

        cutoff_num = modes[0].omega_c / (2.0 * math.pi)
        port_report = PortOperatorReport(
            cutoff_num=cutoff_num,
            cutoff_ref=cutoff_ref,
        )
    elif isinstance(spec, PortSpecNumerical):
        c_3d = build_curl_matrix(mesh.grid)
        K, M, primal_2d = build_2d_curl_curl(plane, mesh.grid, m_eps, m_mu, c_3d)
        pec_mask = _resolve_pec_edge_mask(plane, mesh)
        if subface_edge_mask is not None:
            pec_mask = pec_mask | subface_edge_mask
        solver_kwargs: dict = dict(
            plane=plane,
            K=K,
            M=M,
            primal_2d_indices=primal_2d,
            pec_edge_mask=pec_mask if pec_mask.any() else None,
            epsilon_r=spec.epsilon_r,
            m_mu_flat=m_mu,
        )
        # ``mode_type=None``: unified multi-mode port — solve both
        # families and keep the ``n_modes`` lowest cut-offs.  One
        # operator injects/records/terminates every mode on the face,
        # so the former two-operator TE+TM source-injection collision
        # cannot occur by construction (WP-R3, STATUS site 1).
        mode_types = (ModeType.TE, ModeType.TM) if spec.mode_type is None else (spec.mode_type,)
        modes = []
        for mt in mode_types:
            kwargs = dict(solver_kwargs, mode_type=mt)
            if mt is ModeType.TM:
                g_3d = build_gradient_matrix(mesh.grid)
                g_2d_tm, _, _ = build_2d_gradient(
                    plane,
                    mesh.grid,
                    g_3d,
                )
                L_node, M_node, pec_node_mask = _build_tm_operators(
                    plane,
                    mesh,
                    m_eps,
                    m_mu,
                    c_3d,
                    subface_edge_mask=subface_edge_mask,
                )
                kwargs.update(
                    g_2d=g_2d_tm,
                    L_node=L_node,
                    M_node=M_node,
                    pec_node_mask=pec_node_mask,
                )
            modes.extend(
                Numerical2DModeSolver(**kwargs).solve(
                    n_modes=spec.n_modes,
                    f_calc=f_calc,
                )
            )
        modes = sorted(modes, key=lambda m: m.omega_c)[: spec.n_modes]

        port_report = PortOperatorReport(
            cutoff_num=modes[0].omega_c / (2.0 * math.pi),
        )
    else:  # PortSpecMultiConductor
        c_3d = build_curl_matrix(mesh.grid)
        g_3d = build_gradient_matrix(mesh.grid)
        K_2d, M_2d, primal_2d = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps,
            m_mu,
            c_3d,
        )
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
        if spec.conductors is None:
            conductor_groups = extract_conductor_groups_from_mesh(
                plane,
                mesh,
                extra_pec_edge_mask=subface_edge_mask,
            )
        else:
            conductor_groups = _resolve_conductors(
                spec.conductors,
                plane,
                mesh,
            )
        max_line_modes = len(conductor_groups) - 1
        if spec.epsilon_r is None:
            # QTEM dispatch via dual-Laplace.
            m_eps_vac = build_M_eps_vacuum(mesh)
            m_eps_vac = flatten_port_plane_mass(m_eps_vac, mesh, spec.plane)
            _, M_2d_vacuum, _ = build_2d_curl_curl(
                plane,
                mesh.grid,
                m_eps_vac,
                m_mu,
                c_3d,
            )
            modes = solve_qtem_laplace(
                plane,
                g_2d,
                M_2d,
                M_2d_vacuum,
                conductor_groups,
                grid=mesh.grid,
                m_mu_flat=m_mu,
                boundary_conditions=mesh.boundary_conditions,
            )
            if spec.n_modes > max_line_modes:
                # WP-U6 unified multi-mode QTEM port: no exact
                # TEM/TE/TM split exists on an inhomogeneous
                # cross-section, so the higher channels are the true
                # hybrid eigenpairs of the DD-056 zeta pencil at
                # f_calc.  The line modes stay the Laplace QTEM
                # channels (the DD-064 default path unchanged);
                # projections go dual-basis below because the hybrid
                # profiles are not M_eps-orthogonal.
                modes = modes + _qtem_zeta_hybrid_modes(
                    modes,
                    plane,
                    mesh,
                    m_eps,
                    m_mu,
                    c_3d,
                    dt,
                    f_calc,
                    spec.n_modes - max_line_modes,
                    spec.name,
                )
                qtem_multimode = True
        else:
            modes = solve_tem_laplace(
                plane,
                g_2d,
                M_2d,
                conductor_groups,
                spec.epsilon_r,
                grid=mesh.grid,
                m_mu_flat=m_mu,
                boundary_conditions=mesh.boundary_conditions,
            )
            if spec.n_modes > max_line_modes:
                # WP-U2 unified multi-mode port: the K-1 TEM line
                # modes (omega_c = 0) are joined by the lowest TE/TM
                # curl-curl modes of the same cross-section, merged by
                # ascending cut-off.  For a homogeneous scalar filling
                # the continuum mode space decomposes exactly as
                # TEM (+) TE (+) TM, and the discrete families are
                # orthogonal at solver tolerance through the
                # production projections (WP-U1 measured 8e-16..2e-14
                # on coax and two-wire) — one operator serves all
                # channels, per-channel termination unchanged.
                n_higher = spec.n_modes - max_line_modes
                pec_mask = _resolve_pec_edge_mask(plane, mesh)
                if subface_edge_mask is not None:
                    pec_mask = pec_mask | subface_edge_mask
                solver_kwargs = dict(
                    plane=plane,
                    K=K_2d,
                    M=M_2d,
                    primal_2d_indices=primal_2d,
                    pec_edge_mask=pec_mask if pec_mask.any() else None,
                    epsilon_r=spec.epsilon_r,
                    m_mu_flat=m_mu,
                )
                L_node, M_node, pec_node_mask = _build_tm_operators(
                    plane,
                    mesh,
                    m_eps,
                    m_mu,
                    c_3d,
                    subface_edge_mask=subface_edge_mask,
                )
                higher = Numerical2DModeSolver(
                    **solver_kwargs,
                    mode_type=ModeType.TE,
                ).solve(n_modes=n_higher, f_calc=f_calc)
                higher += Numerical2DModeSolver(
                    **solver_kwargs,
                    mode_type=ModeType.TM,
                    g_2d=g_2d,
                    L_node=L_node,
                    M_node=M_node,
                    pec_node_mask=pec_node_mask,
                ).solve(n_modes=n_higher, f_calc=f_calc)
                higher = sorted(
                    higher,
                    key=lambda m: m.omega_c,
                )[:n_higher]
                modes = modes + higher
        modes = modes[: spec.n_modes]

        first_cutoff = min(
            (m.omega_c for m in modes if m.omega_c > 0.0),
            default=None,
        )
        port_report = PortOperatorReport(
            z_line_num=modes[0].z_line,
            cutoff_num=(first_cutoff / (2.0 * math.pi) if first_cutoff is not None else None),
        )

    modes = _fix_degenerate_polarisation_gauge(modes)
    _warn_on_degenerate_cutoffs(modes, name=spec.name)

    discrete = discretize_modes(modes, plane, m_eps)

    # WP-U6: dual-basis projectors over ALL channels of a multi-mode
    # QTEM port (Gram inverse in the port-plane M_eps metric, the
    # DD-056 machinery) — the hybrid profiles are not
    # M_eps-orthogonal to each other or to the Laplace line modes,
    # so primal projections would cross-talk (the DD-066 instability
    # class); reconstruction stays primal.  A K > 2 QTEM port without
    # hybrids gets the same treatment (DD-196): its modal channels
    # are orthogonal in the capacitance-corrected mass, which differs
    # from M_eps only at tangential window edges — the dual basis
    # absorbs that residual at no cost.
    dual_e_profiles = None
    if qtem_multimode or (spec.epsilon_r is None and len(discrete) > 1):
        me_u = np.asarray(m_eps, dtype=float)[plane.e_u_indices]
        me_v = np.asarray(m_eps, dtype=float)[plane.e_v_indices]
        prof_u = np.stack([dm.e_u_profile for dm in discrete])
        prof_v = np.stack([dm.e_v_profile for dm in discrete])
        gram = (prof_u * me_u[None, :]) @ prof_u.T + (prof_v * me_v[None, :]) @ prof_v.T
        ginv = np.linalg.inv(gram)
        dual_u = ginv @ prof_u
        dual_v = ginv @ prof_v
        dual_e_profiles = [(dual_u[c], dual_v[c]) for c in range(len(discrete))]

    # Certificate stage 2 (DD-067): slab consistency of the masses
    # feeding the mode solve along the first feed cells.  Above the
    # gate every channel of this port runs Mur — loud, so the user
    # knows the exact DTBC was withheld and why.
    chain_slab_defect = _port_chain_slab_defect(
        mesh,
        m_eps,
        m_mu,
        spec.plane,
    )
    if chain_slab_defect > _DTBC_SLAB_DEFECT_TOL:
        # DTBC certificate stage 2: DD-067.
        warnings.warn(
            f"Modal port {spec.name!r}: the feed-chain mass slabs "
            f"deviate by {chain_slab_defect:.2e} (relative, worst "
            f"entry) across the first cells behind the port plane — "
            f"the 2D port modes do not propagate as exact discrete "
            f"chain modes, so the exact DTBC is withheld and all "
            f"channels fall back to modal Mur-1st.  Typical causes: "
            f"a feed that is "
            f"not translation-invariant along the port normal."
        )
    op = PortOperatorModal(
        spec.name,
        plane,
        discrete,
        m_eps,
        m_mu,
        dt=dt,
        omega_calc=omega_calc,
        port_report=_with_symmetry_faces(port_report, plane, mesh, name=spec.name),
        chain_slab_defect=chain_slab_defect,
        dual_e_profiles=dual_e_profiles,
        flux_patch=conformal_flux_patch_scale(plane, mesh, m_eps),
        complement_absorber=_complement_absorber_arrays(
            plane,
            mesh,
            spec.plane,
            m_eps,
            dt,
            subface_edge_mask,
        ),
        magnetic_patch_ends=magnetic_window_ends(
            plane,
            mesh.grid,
            mesh.boundary_conditions,
        ),
    )
    return op


def build_cw_true_mode_port(
    spec: PortSpecMultiConductor,
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    *,
    dt: float,
    f_cw: float,
    n_channels: Optional[int] = None,
) -> PortOperatorModal:
    """Build a CW true-mode port for an inhomogeneous line (WP-R4a).

    Per-frequency production path (Option B, DD-056): the port's
    channels are the *true discrete modes* of the port cross-section
    at the drive frequency ``f_cw`` — eigenpairs of the quadratic
    zeta pencil built from the actual production matrices — each
    terminated by the frequency-local exact DTBC (the closed-form
    ``(r_eff, q_eff)`` scalar-chain fit, exact at ``f_cw`` by
    construction; pre-check gate session 88).  The returned operator
    carries ``op.cw_data`` (:class:`~magnelio.ports._modal.zeta_pencil.
    CWPortData`): per channel the eigenvalue ``zeta``, the fitted
    chain parameters and the exact V/I phasors of the incident /
    reflected discrete wave through the stored profiles — everything
    the CW lock-in a/b decomposition needs
    (:func:`~magnelio.ports._modal.zeta_pencil.cw_lockin_phasors` +
    :func:`~magnelio.ports._modal.zeta_pencil.cw_decompose`).

    Channel 0 is the fundamental (tracked from the first DC Laplace
    line mode — the conductor mode of a two-conductor line, the
    largest-``ε_eff`` modal channel of a multi-conductor one); further
    channels are the other modes propagating at ``f_cw``, ordered by
    descending phase advance.  Multi-channel projection is dual-basis (the true modes
    are not M_eps-orthogonal), reconstruction primal.

    Parameters
    ----------
    spec : PortSpecMultiConductor
        Port description; ``epsilon_r=None`` selects the QTEM
        Laplace bootstrap (the standard inhomogeneous case),
        a float value the homogeneous TEM bootstrap.  The spec's
        CW drive is set by the
        caller via ``op.set_excitation`` (the waveform is a ramped
        monochromatic tone, not a pulsed waveform).
    mesh, m_eps, m_mu, dt
        As in :func:`build_modal_port`.
    f_cw : float
        The CW drive frequency [Hz].  The port is exact at this
        frequency only; build one operator pair per frequency point.
    n_channels : int or None, default None
        ``None`` — one channel per mode propagating at ``f_cw``
        (at least the fundamental).  An integer demands exactly that
        many propagating channels and raises if fewer exist.

    Raises
    ------
    ValueError
        If the port-adjacent section violates the uniformity
        certificates of
        :func:`~magnelio.ports._modal.zeta_pencil.build_period_blocks`,
        if the frequency-local fit is not certified
        (``q_eff^2 < 0``), or if ``n_channels`` propagating modes do
        not exist at ``f_cw``.
    """
    from magnelio.constants import C0, ETA0
    from magnelio.ports._modal.mode import Mode
    from magnelio.ports._modal.tem_laplace import (
        travelling_wave_h_profiles,
    )
    from magnelio.ports._modal.zeta_pencil import (
        CWPortData,
        build_period_blocks,
        cw_wave_phasors,
        find_propagating_modes,
        make_channel,
        profile_reality,
    )

    if not isinstance(spec, PortSpecMultiConductor):
        raise TypeError(
            "build_cw_true_mode_port requires a PortSpecMultiConductor "
            f"spec; got {type(spec).__name__}"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if f_cw <= 0.0:
        raise ValueError("f_cw must be positive")
    if n_channels is not None and n_channels < 1:
        raise ValueError("n_channels must be >= 1 (or None for auto)")

    _validate_three_equidistant_cells(spec.plane, mesh)

    plane = PortPlane.from_mesh(
        spec.plane,
        mesh,
        window=getattr(spec, "window", None),
    )
    subface_edge_mask: Optional[np.ndarray] = None
    if getattr(spec, "window", None) is not None:
        edge_pec = resolve_port_edge_pec(
            plane,
            mesh,
            _pec_faces_from_mask(mesh),
        )
        subface_edge_mask = build_port_edge_pec_mask(plane, edge_pec)

    m_eps = flatten_port_plane_mass(m_eps, mesh, spec.plane)
    m_mu = flatten_port_plane_mu(m_mu, mesh, spec.plane)
    # Builder-local flatten — see build_modal_port for the rationale;
    # writing back would poison later operator builds on this mesh.
    mesh = dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(
            mesh.pec_mask_edges,
            mesh,
            spec.plane,
        ),
    )

    # DC bootstrap: the Laplace fundamental supplies the tracking
    # profile, the eps_eff continuation anchor and the reported
    # z_line.
    c_3d = build_curl_matrix(mesh.grid)
    g_3d = build_gradient_matrix(mesh.grid)
    K_2d, M_2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps, m_mu, c_3d)
    g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
    if spec.conductors is None:
        conductor_groups = extract_conductor_groups_from_mesh(
            plane,
            mesh,
            extra_pec_edge_mask=subface_edge_mask,
        )
    else:
        conductor_groups = _resolve_conductors(spec.conductors, plane, mesh)
    if spec.epsilon_r is None:
        m_eps_vac = build_M_eps_vacuum(mesh)
        m_eps_vac = flatten_port_plane_mass(m_eps_vac, mesh, spec.plane)
        _, M_2d_vacuum, _ = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps_vac,
            m_mu,
            c_3d,
        )
        laplace_modes = solve_qtem_laplace(
            plane,
            g_2d,
            M_2d,
            M_2d_vacuum,
            conductor_groups,
            grid=mesh.grid,
            m_mu_flat=m_mu,
            boundary_conditions=mesh.boundary_conditions,
        )
    else:
        laplace_modes = solve_tem_laplace(
            plane,
            g_2d,
            M_2d,
            conductor_groups,
            spec.epsilon_r,
            grid=mesh.grid,
            m_mu_flat=m_mu,
            boundary_conditions=mesh.boundary_conditions,
        )
    lap = laplace_modes[0]
    eps_eff_dc = float(lap.epsilon_r)

    t_solve0 = time.perf_counter()
    chain = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)
    w_dt = 2.0 * math.pi * f_cw * dt
    dz = float(plane.normal_dx)

    track_t = np.concatenate(
        [
            np.asarray(lap.discrete_e_u_profile)[chain.free_u],
            np.asarray(lap.discrete_e_v_profile)[chain.free_v],
        ]
    )

    # The DC eps_eff underestimates the fundamental's phase advance
    # (normal dispersion), so the arc hint carries a 30 % margin.
    theta0 = 2.0 * math.pi * f_cw * math.sqrt(eps_eff_dc) / C0 * dz
    zp, pp = find_propagating_modes(chain, w_dt, 1.3 * theta0)
    if zp.size == 0:
        raise ValueError(
            f"no propagating mode found at f_cw = {f_cw / 1e9:.3f} GHz on this port cross-section"
        )
    w_t = chain.w_period[: chain.n_t]
    ov = np.abs(track_t @ (w_t[:, None] * pp[: chain.n_t, :]))
    fund = int(np.argmax(ov))
    # Fundamental first, the rest stays in descending phase advance
    # (ascending mode order).
    picked = [fund] + [j for j in range(zp.size) if j != fund]
    if n_channels is not None:
        if len(picked) < n_channels:
            raise ValueError(
                f"only {len(picked)} propagating mode(s) at "
                f"f_cw = {f_cw / 1e9:.3f} GHz; n_channels="
                f"{n_channels} requested"
            )
        picked = picked[:n_channels]

    channels = [make_channel(complex(zp[j]), pp[:, j], chain, w_dt, dz) for j in picked]
    for c, ch in enumerate(channels):
        reality = profile_reality(ch.phi_trace, chain.n_t)
        if reality > 1e-6:
            raise ValueError(
                f"CW channel {c}: tangential profile not real after "
                f"gauge fixing (residual {reality:.2e}) — degenerate "
                "or complex mode pair; not certified"
            )

    # Real, W_t-normalised port profiles + travelling-wave H form.
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    me_u = np.asarray(m_eps, dtype=float)[plane.e_u_indices]
    me_v = np.asarray(m_eps, dtype=float)[plane.e_v_indices]
    prof_u = np.zeros((len(channels), n_u))
    prof_v = np.zeros((len(channels), n_v))
    modes: list[Mode] = []
    nu_free = int(chain.free_u.sum())
    for c, ch in enumerate(channels):
        eu = np.zeros(n_u)
        ev = np.zeros(n_v)
        eu[chain.free_u] = ch.phi_trace[:nu_free].real
        ev[chain.free_v] = ch.phi_trace[nu_free : chain.n_t].real
        nrm = math.sqrt(float(np.dot(me_u, eu**2) + np.dot(me_v, ev**2)))
        eu /= nrm
        ev /= nrm
        prof_u[c] = eu
        prof_v[c] = ev
        h_u, h_v = travelling_wave_h_profiles(
            eu,
            ev,
            plane,
            m_mu,
            ETA0 / math.sqrt(ch.eps_eff_hat),
        )
        modes.append(
            Mode(
                name=("QTEM_cw00" if c == 0 else f"HYB_cw{c:02d}"),
                mode_type=ModeType.TEM,
                omega_c=0.0 if c == 0 else ch.q / dt,
                epsilon_r=ch.eps_eff_hat,
                field_evaluator=None,
                z_line=lap.z_line if c == 0 else None,
                discrete_e_u_profile=eu,
                discrete_e_v_profile=ev,
                discrete_h_u_profile=h_u,
                discrete_h_v_profile=h_v,
            )
        )

    # Dual-basis projectors (Gram inverse in the port-plane W_t).
    gram = (prof_u * me_u[None, :]) @ prof_u.T + (prof_v * me_v[None, :]) @ prof_v.T
    ginv = np.linalg.inv(gram)
    dual_u = ginv @ prof_u
    dual_v = ginv @ prof_v
    dual_e_profiles = [(dual_u[c], dual_v[c]) for c in range(len(channels))]

    discrete = discretize_modes(modes, plane, m_eps)
    op = PortOperatorModal(
        spec.name,
        plane,
        discrete,
        m_eps,
        m_mu,
        dt=dt,
        omega_calc=2.0 * math.pi * f_cw,
        port_report=_with_symmetry_faces(
            PortOperatorReport(z_line_num=lap.z_line),
            plane,
            mesh,
            name=spec.name,
        ),
        chain_overrides={c: (ch.r, ch.q) for c, ch in enumerate(channels)},
        dual_e_profiles=dual_e_profiles,
        calibrate=False,
    )

    # Exact per-frequency phasors through the *stored* profiles.
    channels = [
        cw_wave_phasors(
            ch,
            chain,
            plane,
            m_eps,
            m_mu,
            c_3d,
            w_dt,
            h_u_prof=discrete[c].h_u_profile,
            h_v_prof=discrete[c].h_v_profile,
            proj_u=dual_u[c],
            proj_v=dual_v[c],
        )
        for c, ch in enumerate(channels)
    ]
    op.cw_data = CWPortData(
        f_cw=float(f_cw),
        w_dt=float(w_dt),
        channels=tuple(channels),
        solve_seconds=time.perf_counter() - t_solve0,
    )
    return op


def build_band_dtbc_port(
    spec: PortSpecMultiConductor,
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    *,
    dt: float,
    f_band: tuple[float, float],
    n_grid: int = 17,
    p: Optional[int] = None,
    svd_tol: float = 1e-8,
    n_channels: Optional[int] = None,
    n_kernel_init: int = 4096,
):
    """Build a broadband band-subspace DTBC port (WP-R4b).

    One operator terminates the whole band: the tracked mode-family
    traces over ``f_band`` span a real W-orthonormal subspace (rank
    ``p``), the exterior is Galerkin-projected onto it (``D~ = V^T W
    D V`` — passive by construction, DD-057) and closed by the exact
    small-system DTBC kernel (contour QZ at size 2p).  Pulsed
    broadband runs through this port are decomposed per frequency in
    postprocessing with the WP-R4a true-mode machinery
    (:func:`~magnelio.post.modal_sparameters.
    compute_band_s_parameters`); the returned operator carries
    ``op.band_data`` (:class:`~magnelio.ports._modal.band_dtbc.
    BandPortData`) with everything that decomposition needs.

    Channel 0 records the fundamental (tracked from the first DC
    Laplace line mode, see :func:`build_cw_true_mode_port`); further
    channels record the higher families cut on
    inside the band, each through a fixed mid-band reference profile
    (dual-basis projection — the per-frequency decomposition resolves
    the frequency dependence).  ``set_excitation(c, waveform)``
    launches a broadband pulse through the ghost plane of the
    projected exterior on channel ``c``'s reference profile.

    Parameters
    ----------
    spec : PortSpecMultiConductor
        Port description; ``epsilon_r=None`` selects the QTEM
        Laplace bootstrap.  Pulsed drives are set via ``op.set_excitation``.
    mesh, m_eps, m_mu, dt
        As in :func:`build_modal_port`.
    f_band : (float, float)
        Band ``(f_min, f_max)`` [Hz] the subspace is built for.  The
        S-parameter axis of the pulsed run must stay inside it.
    n_grid : int, default 17
        Frequency-grid points for the mode-family tracking.
    p : int or None
        Subspace rank; ``None`` selects it from ``svd_tol``.
    svd_tol : float, default 1e-8
        Relative singular-value threshold for the automatic rank
        (the subspace-capture certificate of the WP-R4b gate).
    n_channels : int or None
        Number of recording channels.  ``None`` — one per tracked
        family; an integer demands at least that many families.
    n_kernel_init : int, default 4096
        Initial ghost-kernel length (auto-extends past the run).

    Raises
    ------
    ValueError
        If the port section violates the uniformity certificates, no
        mode propagates at the band start, the projected blocks
        violate the palindromic symmetry certificate, or fewer than
        ``n_channels`` families exist.
    """
    from magnelio.constants import C0, ETA0
    from magnelio.ports._modal.band_dtbc import (
        BandDTBCBoundary,
        BandPortData,
        PortOperatorBandDTBC,
        band_subspace,
        galerkin_exterior,
        track_band_families,
    )
    from magnelio.ports._modal.mode import Mode
    from magnelio.ports._modal.tem_laplace import (
        travelling_wave_h_profiles,
    )
    from magnelio.ports._modal.zeta_pencil import build_period_blocks

    if not isinstance(spec, PortSpecMultiConductor):
        raise TypeError(
            "build_band_dtbc_port requires a PortSpecMultiConductor "
            f"spec; got {type(spec).__name__}"
        )
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    f_lo, f_hi = float(f_band[0]), float(f_band[1])
    if not (0.0 < f_lo < f_hi):
        raise ValueError(f"f_band must satisfy 0 < f_min < f_max, got {f_band}")
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3")
    if n_channels is not None and n_channels < 1:
        raise ValueError("n_channels must be >= 1 (or None for auto)")

    _validate_three_equidistant_cells(spec.plane, mesh)

    plane = PortPlane.from_mesh(
        spec.plane,
        mesh,
        window=getattr(spec, "window", None),
    )
    subface_edge_mask: Optional[np.ndarray] = None
    if getattr(spec, "window", None) is not None:
        edge_pec = resolve_port_edge_pec(
            plane,
            mesh,
            _pec_faces_from_mask(mesh),
        )
        subface_edge_mask = build_port_edge_pec_mask(plane, edge_pec)

    m_eps = flatten_port_plane_mass(m_eps, mesh, spec.plane)
    m_mu = flatten_port_plane_mu(m_mu, mesh, spec.plane)
    # Builder-local flatten — see build_modal_port for the rationale;
    # writing back would poison later operator builds on this mesh.
    mesh = dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(
            mesh.pec_mask_edges,
            mesh,
            spec.plane,
        ),
    )

    # DC bootstrap: tracking profile, eps_eff arc hint, z_line.
    c_3d = build_curl_matrix(mesh.grid)
    g_3d = build_gradient_matrix(mesh.grid)
    K_2d, M_2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps, m_mu, c_3d)
    g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
    if spec.conductors is None:
        conductor_groups = extract_conductor_groups_from_mesh(
            plane,
            mesh,
            extra_pec_edge_mask=subface_edge_mask,
        )
    else:
        conductor_groups = _resolve_conductors(spec.conductors, plane, mesh)
    if spec.epsilon_r is None:
        m_eps_vac = build_M_eps_vacuum(mesh)
        m_eps_vac = flatten_port_plane_mass(m_eps_vac, mesh, spec.plane)
        _, M_2d_vacuum, _ = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps_vac,
            m_mu,
            c_3d,
        )
        laplace_modes = solve_qtem_laplace(
            plane,
            g_2d,
            M_2d,
            M_2d_vacuum,
            conductor_groups,
            grid=mesh.grid,
            m_mu_flat=m_mu,
            boundary_conditions=mesh.boundary_conditions,
        )
    else:
        laplace_modes = solve_tem_laplace(
            plane,
            g_2d,
            M_2d,
            conductor_groups,
            spec.epsilon_r,
            grid=mesh.grid,
            m_mu_flat=m_mu,
            boundary_conditions=mesh.boundary_conditions,
        )
    lap = laplace_modes[0]
    eps_eff_dc = float(lap.epsilon_r)

    t_solve0 = time.perf_counter()
    chain_b = build_period_blocks(
        plane,
        mesh,
        m_eps,
        m_mu,
        c_3d,
        dt,
        pairing="boundary",
    )
    chain_i = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)
    dz = float(plane.normal_dx)

    track_t = np.concatenate(
        [
            np.asarray(lap.discrete_e_u_profile)[chain_b.free_u],
            np.asarray(lap.discrete_e_v_profile)[chain_b.free_v],
        ]
    )

    f_grid = np.linspace(f_lo, f_hi, int(n_grid))
    families = track_band_families(
        chain_b,
        dt,
        f_grid,
        track_t,
        eps_eff_dc,
        dz,
    )
    if n_channels is None:
        n_channels = len(families)
    elif len(families) < n_channels:
        raise ValueError(
            f"only {len(families)} mode families propagate in "
            f"[{f_lo / 1e9:.2f}, {f_hi / 1e9:.2f}] GHz; "
            f"n_channels={n_channels} requested"
        )

    V, sv = band_subspace(
        families,
        chain_b.w_period,
        p=p,
        svd_tol=svd_tol,
    )
    exterior = galerkin_exterior(chain_b, V)
    boundary = BandDTBCBoundary(exterior, n_kernel_init=n_kernel_init)

    # Fixed recording channels: one mid-band reference profile per
    # family (real e_t part in the fixed gauge, M_eps-normalised on
    # the plane).
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    me_u = np.asarray(m_eps, dtype=float)[plane.e_u_indices]
    me_v = np.asarray(m_eps, dtype=float)[plane.e_v_indices]
    nu_free = int(chain_b.free_u.sum())
    n_t = chain_b.n_t
    prof_u = np.zeros((n_channels, n_u))
    prof_v = np.zeros((n_channels, n_v))
    src_directions: list[tuple[np.ndarray, np.ndarray]] = []
    modes: list[Mode] = []
    for c in range(n_channels):
        fam = families[c]
        i_ref = fam.freqs.size // 2
        f_ref = float(fam.freqs[i_ref])
        zeta_ref = complex(fam.zetas[i_ref])
        phi_ref = fam.traces[:, i_ref]
        eu = np.zeros(n_u)
        ev = np.zeros(n_v)
        eu[chain_b.free_u] = phi_ref[:nu_free].real
        ev[chain_b.free_v] = phi_ref[nu_free:n_t].real
        nrm = math.sqrt(float(np.dot(me_u, eu**2) + np.dot(me_v, ev**2)))
        eu /= nrm
        ev /= nrm
        prof_u[c] = eu
        prof_v[c] = ev
        theta = abs(np.angle(zeta_ref))
        w_dt_ref = 2.0 * math.pi * f_ref * dt
        s_ratio = (math.sin(theta / 2.0) / (dz / 2.0)) / (math.sin(w_dt_ref / 2.0) / (dt / 2.0))
        eps_eff_ref = (C0 * s_ratio) ** 2
        h_u, h_v = travelling_wave_h_profiles(
            eu,
            ev,
            plane,
            m_mu,
            ETA0 / math.sqrt(eps_eff_ref),
        )
        modes.append(
            Mode(
                name=("QTEM_bb00" if c == 0 else f"HYB_bb{c:02d}"),
                mode_type=ModeType.TEM,
                omega_c=0.0 if c == 0 else 2.0 * math.pi * fam.f_first,
                epsilon_r=float(eps_eff_ref),
                field_evaluator=None,
                z_line=lap.z_line if c == 0 else None,
                discrete_e_u_profile=eu,
                discrete_e_v_profile=ev,
                discrete_h_u_profile=h_u,
                discrete_h_v_profile=h_v,
            )
        )
        # Frequency-tracked ghost-source directions: the family's
        # subspace coordinates over the grid, phase-aligned for a
        # smooth spectral synthesis in set_excitation.
        Uc = np.ascontiguousarray((exterior.VtW @ fam.traces).T)
        for i in range(1, Uc.shape[0]):
            ov = np.vdot(Uc[i - 1], Uc[i])
            if abs(ov) > 0.0:
                Uc[i] *= np.conj(ov) / abs(ov)
        src_directions.append((fam.freqs.copy(), Uc))

    # Dual-basis projectors (Gram inverse in the port-plane metric).
    gram = (prof_u * me_u[None, :]) @ prof_u.T + (prof_v * me_v[None, :]) @ prof_v.T
    ginv = np.linalg.inv(gram)
    dual_u = ginv @ prof_u
    dual_v = ginv @ prof_v
    dual_e_profiles = [(dual_u[c], dual_v[c]) for c in range(n_channels)]

    discrete = discretize_modes(modes, plane, m_eps)
    op = PortOperatorBandDTBC(
        spec.name,
        plane,
        chain_b,
        exterior,
        boundary,
        discrete,
        m_eps,
        m_mu,
        dt,
        src_directions=src_directions,
        dual_e_profiles=dual_e_profiles,
        port_report=_with_symmetry_faces(
            PortOperatorReport(z_line_num=lap.z_line),
            plane,
            mesh,
            name=spec.name,
        ),
    )
    op.band_data = BandPortData(
        f_band=(f_lo, f_hi),
        f_grid=f_grid,
        families=families,
        singular_values=sv,
        p=exterior.p,
        chain_inward=chain_i,
        chain_boundary=chain_b,
        exterior=exterior,
        plane=plane,
        m_eps=m_eps,
        m_mu=m_mu,
        c_3d=c_3d,
        dual_e_profiles=dual_e_profiles,
        eps_eff_dc=eps_eff_dc,
        z_line=lap.z_line,
        solve_seconds=time.perf_counter() - t_solve0,
    )
    return op
