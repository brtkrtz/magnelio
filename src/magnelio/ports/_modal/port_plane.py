"""Port-plane geometry and FIT-mesh edge mapping.

A ``PortPlane`` describes one face of the simulation bounding box where a
waveguide port lives.  It provides:

- a local ``(u, v)`` coordinate system whose right-hand-rule cross product
  ``u × v`` points into the simulation domain,
- the indices into the flat E and H vectors for the tangential primal
  and dual edges in the plane,
- the (u, v) mid-point coordinates and edge lengths needed to evaluate
  modal field profiles.

Yee-stagger property used throughout: tangential E-edges and tangential
dual-edges are co-located in (u, v).  ``e_u_indices[k]`` and
``h_v_indices[k]`` share the same midpoint, similarly
``e_v_indices[k]`` and ``h_u_indices[k]``.  This co-location lets the
projection ``V_m`` and ``I_m`` work on the same edge ordering without
interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.mesh.faces import BoxFace
from magnelio.mesh.mesher import Mesh


@dataclass(frozen=True)
class PortPlane:
    """Port-plane geometry on a hex-mesh face.

    The plane covers either the whole bbox face (default) or an
    axis-aligned sub-rectangle of it (``window=`` on :meth:`from_mesh`).

    Two parallel arrays of edges are exposed:

    - **u-edges**: primal edges along ``u``, holding the tangential
      ``E_u`` component.  Co-located dual edges hold ``H_v``.
    - **v-edges**: primal edges along ``v``, holding ``E_v``.  Co-located
      dual edges hold ``H_u``.

    All ``*_indices`` arrays index the flat E or H vectors of
    ``FieldState``.  Coordinates are in metres.

    Attributes
    ----------
    face : BoxFace
        Which bbox face the plane lies on.
    coordinate : float
        Position of the plane along the normal axis [m].
    e_u_indices : np.ndarray of int, shape (N_u,)
        Flat-E-vector indices for primal edges holding ``E_u`` at the
        port plane.
    h_v_indices : np.ndarray of int, shape (N_u,)
        Flat-H-vector indices for dual edges holding ``H_v``, co-located
        in ``(u, v)`` with the entries in ``e_u_indices``.
    u_edge_uv : np.ndarray of float, shape (N_u, 2)
        ``(u, v)`` midpoints of the u-edges.
    u_edge_lengths : np.ndarray of float, shape (N_u,)
        Primal edge lengths along ``u`` (= cell sizes of the u-cells).
    e_v_indices, h_u_indices, v_edge_uv, v_edge_lengths : analogous
        For v-edges.
    e_u_indices_interior : np.ndarray of int, shape (N_u,)
        Flat-E-vector indices for the same u-edges shifted **one cell
        inward** along the normal axis.  Used by the modal Mur-1st
        absorber to project ``V_m`` at the interior plane.
    e_v_indices_interior : np.ndarray of int, shape (N_v,)
        Same for v-edges.
    normal_dx : float
        Distance between the port plane and the interior plane along
        the normal axis [m].  Equal to the boundary cell's normal-axis
        size (e.g. ``mesh.grid.dx[0]`` for X_MIN).
    u_node_window, v_node_window : tuple[int, int]
        Inclusive grid-node index window ``(lo, hi)`` along the local
        u / v axis that the plane covers; cells span ``[lo, hi)``.
        For a whole-face plane this is ``(0, N_axis)``.
    u_bounds, v_bounds : tuple[float, float]
        Physical extents of the (clipped, grid-snapped) plane along
        the local u / v axis [m].
    """

    face: BoxFace
    coordinate: float

    e_u_indices: np.ndarray
    h_v_indices: np.ndarray
    u_edge_uv: np.ndarray
    u_edge_lengths: np.ndarray

    e_v_indices: np.ndarray
    h_u_indices: np.ndarray
    v_edge_uv: np.ndarray
    v_edge_lengths: np.ndarray

    e_u_indices_interior: np.ndarray
    e_v_indices_interior: np.ndarray
    normal_dx: float

    u_node_window: tuple[int, int]
    v_node_window: tuple[int, int]
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]

    @property
    def n_cells_u(self) -> int:
        """Number of grid cells the plane covers along local u."""
        return self.u_node_window[1] - self.u_node_window[0]

    @property
    def n_cells_v(self) -> int:
        """Number of grid cells the plane covers along local v."""
        return self.v_node_window[1] - self.v_node_window[0]

    @property
    def n_nodes_u(self) -> int:
        """Number of grid nodes the plane covers along local u."""
        return self.n_cells_u + 1

    @property
    def n_nodes_v(self) -> int:
        """Number of grid nodes the plane covers along local v."""
        return self.n_cells_v + 1

    @classmethod
    def from_mesh(
        cls,
        face: BoxFace,
        mesh: Mesh,
        window: tuple | None = None,
    ) -> "PortPlane":
        """Build a PortPlane on the given bbox face of ``mesh``.

        Parameters
        ----------
        face : BoxFace
            Bbox face the plane lies on.
        mesh : Mesh
            The 3D mesh.
        window : tuple of two corner points, optional
            Sub-rectangle of the face, given as two opposite corners
            ``((a1, b1), (a2, b2))`` in the *global* tangential-axis
            ordering — ``a`` along the lower-numbered tangential global
            axis, ``b`` along the higher-numbered one — in metres, in
            any corner order (``±inf`` reaches the domain boundary).
            The extents are clipped to the domain (an oversized window
            is legal) and snapped to the nearest grid nodes.  ``None``
            (default) covers the whole face.  The user-facing spelling
            is the world-coordinate ``corners=`` of
            :class:`~magnelio.ports.PortWaveguide`;
            :func:`~magnelio.ports.declarative.window_from_corners`
            projects it into this form.

        Raises
        ------
        ValueError
            When the window is malformed, degenerate, or collapses to
            fewer than one grid cell after clipping and snapping.
        """
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        x_n, y_n, z_n = mesh.grid.x, mesh.grid.y, mesh.grid.z
        x_c = 0.5 * (x_n[:-1] + x_n[1:])
        y_c = 0.5 * (y_n[:-1] + y_n[1:])
        z_c = 0.5 * (z_n[:-1] + z_n[1:])
        dx, dy, dz = mesh.grid.dx, mesh.grid.dy, mesh.grid.dz

        # Flat-vector offsets (must match FieldState layout)
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        E_OFF = {0: 0, 1: n_Ex, 2: n_Ex + n_Ey}

        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz
        H_OFF = {0: 0, 1: n_Hx, 2: n_Hx + n_Hy}

        # E-component shape lookup: shape[axis_of_E_component]
        # Ex:(Nx, Ny+1, Nz+1) | Ey:(Nx+1, Ny, Nz+1) | Ez:(Nx+1, Ny+1, Nz)
        E_SHAPE = {
            0: (Nx, Ny + 1, Nz + 1),
            1: (Nx + 1, Ny, Nz + 1),
            2: (Nx + 1, Ny + 1, Nz),
        }
        # H-component shape lookup
        # Hx:(Nx+1, Ny, Nz) | Hy:(Nx, Ny+1, Nz) | Hz:(Nx, Ny, Nz+1)
        H_SHAPE = {
            0: (Nx + 1, Ny, Nz),
            1: (Nx, Ny + 1, Nz),
            2: (Nx, Ny, Nz + 1),
        }

        n_axis = face.normal_axis
        u_axis = face.u_axis
        v_axis = face.v_axis

        # Node-index windows along the local u/v axes: whole face
        # unless a sub-face window clips them.
        u_window, v_window = _resolve_uv_windows(face, mesh, window)

        # E_u stored on E-component aligned with the u-axis (axis = u_axis).
        # E_v stored on E-component aligned with v_axis.
        # H_u on H-component aligned with u_axis.
        # H_v on H-component aligned with v_axis.

        # The plane is at:
        #  - normal-axis index 0 (for MIN faces) or N_axis (for MAX faces)
        #    in the E-component subarrays whose first dimension lies along
        #    the normal axis (Ey / Ez for X_MIN, etc.).
        # The corresponding H-components (Hy / Hz for X_MIN) sit half-a-cell
        # inside, indexed at:
        #  - 0 (for MIN) or N_axis - 1 (for MAX).
        N_axis = (Nx, Ny, Nz)[n_axis]
        plane_idx_E_port = 0 if not face.is_max else N_axis
        # Interior plane: one cell inward from the port plane.
        plane_idx_E_interior = 1 if not face.is_max else N_axis - 1
        plane_idx_H = 0 if not face.is_max else N_axis - 1

        # Distance between port and interior planes along the normal
        # axis = the boundary cell's normal-axis size.
        deltas_normal = (dx, dy, dz)[n_axis]
        normal_dx = float(deltas_normal[plane_idx_H])

        coordinate = (
            x_n[plane_idx_E_port]
            if n_axis == 0
            else (y_n[plane_idx_E_port] if n_axis == 1 else z_n[plane_idx_E_port])
        )

        # Cell-centre / node arrays per axis, used for midpoint coords.
        nodes = (x_n, y_n, z_n)
        centres = (x_c, y_c, z_c)
        deltas = (dx, dy, dz)

        # --- Build u-edges at port plane (E_u + co-located H_v dual) ---
        e_u_idx_port, h_v_idx, u_uv, u_len = _build_uv_edges(
            primal_axis=u_axis,
            secondary_axis=v_axis,
            normal_axis=n_axis,
            normal_index_E=plane_idx_E_port,
            normal_index_H=plane_idx_H,
            nodes=nodes,
            centres=centres,
            deltas=deltas,
            E_off=E_OFF[u_axis],
            E_shape=E_SHAPE[u_axis],
            H_off=H_OFF[v_axis],
            H_shape=H_SHAPE[v_axis],
            primal_cells=u_window,
            secondary_nodes=v_window,
        )
        # u-edges at interior plane (E_u only; same shape, different normal idx)
        e_u_idx_int, _, _, _ = _build_uv_edges(
            primal_axis=u_axis,
            secondary_axis=v_axis,
            normal_axis=n_axis,
            normal_index_E=plane_idx_E_interior,
            normal_index_H=plane_idx_H,
            nodes=nodes,
            centres=centres,
            deltas=deltas,
            E_off=E_OFF[u_axis],
            E_shape=E_SHAPE[u_axis],
            H_off=H_OFF[v_axis],
            H_shape=H_SHAPE[v_axis],
            primal_cells=u_window,
            secondary_nodes=v_window,
        )

        # --- Build v-edges at port plane (E_v + co-located H_u dual) ---
        e_v_idx_port, h_u_idx, v_uv_raw, v_len = _build_uv_edges(
            primal_axis=v_axis,
            secondary_axis=u_axis,
            normal_axis=n_axis,
            normal_index_E=plane_idx_E_port,
            normal_index_H=plane_idx_H,
            nodes=nodes,
            centres=centres,
            deltas=deltas,
            E_off=E_OFF[v_axis],
            E_shape=E_SHAPE[v_axis],
            H_off=H_OFF[u_axis],
            H_shape=H_SHAPE[u_axis],
            primal_cells=v_window,
            secondary_nodes=u_window,
        )
        # v-edges at interior plane (E_v only)
        e_v_idx_int, _, _, _ = _build_uv_edges(
            primal_axis=v_axis,
            secondary_axis=u_axis,
            normal_axis=n_axis,
            normal_index_E=plane_idx_E_interior,
            normal_index_H=plane_idx_H,
            nodes=nodes,
            centres=centres,
            deltas=deltas,
            E_off=E_OFF[v_axis],
            E_shape=E_SHAPE[v_axis],
            H_off=H_OFF[u_axis],
            H_shape=H_SHAPE[u_axis],
            primal_cells=v_window,
            secondary_nodes=u_window,
        )
        # v-edge raw uv has columns (v_coord, u_coord); swap to (u, v).
        v_uv = v_uv_raw[:, [1, 0]]

        u_nodes = nodes[u_axis]
        v_nodes = nodes[v_axis]
        return cls(
            face=face,
            coordinate=float(coordinate),
            e_u_indices=e_u_idx_port,
            h_v_indices=h_v_idx,
            u_edge_uv=u_uv,
            u_edge_lengths=u_len,
            e_v_indices=e_v_idx_port,
            h_u_indices=h_u_idx,
            v_edge_uv=v_uv,
            v_edge_lengths=v_len,
            e_u_indices_interior=e_u_idx_int,
            e_v_indices_interior=e_v_idx_int,
            normal_dx=normal_dx,
            u_node_window=u_window,
            v_node_window=v_window,
            u_bounds=(float(u_nodes[u_window[0]]), float(u_nodes[u_window[1]])),
            v_bounds=(float(v_nodes[v_window[0]]), float(v_nodes[v_window[1]])),
        )


def _resolve_uv_windows(
    face: BoxFace,
    mesh: Mesh,
    window: tuple | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve a sub-face window into inclusive node-index windows (u, v).

    ``window`` is two opposite corner points ``((a1, b1), (a2, b2))`` of
    the sub-rectangle, in the global tangential-axis ordering (``a`` =
    coordinate along the lower-numbered tangential axis, ``b`` = along
    the higher-numbered one); the corners may come in any order.
    ``None`` means the whole face.  The extents are clipped to the
    domain (an oversized window is legal) and snapped to the nearest grid
    nodes.
    """
    nodes = (mesh.grid.x, mesh.grid.y, mesh.grid.z)
    u_axis, v_axis = face.u_axis, face.v_axis

    if window is None:
        return (
            (0, nodes[u_axis].size - 1),
            (0, nodes[v_axis].size - 1),
        )

    if len(window) != 2 or any(len(r) != 2 for r in window):
        raise ValueError(
            f"window must be two corner points ((a1, b1), (a2, b2)) in "
            f"global tangential-axis order; got {window!r}",
        )
    (a1, b1), (a2, b2) = window
    range_a = (min(a1, a2), max(a1, a2))
    range_b = (min(b1, b2), max(b1, b2))
    # Global-order (a, b) -> local (u, v); reversed on faces whose
    # (u, v) frame swaps the tangential axes (u × v must point inward).
    range_u, range_v = (range_a, range_b) if u_axis < v_axis else (range_b, range_a)

    windows = []
    axis_names = "xyz"
    for axis, (lo, hi) in ((u_axis, range_u), (v_axis, range_v)):
        if not lo < hi:
            raise ValueError(
                f"window extent along {axis_names[axis]} is degenerate: both corners sit at {lo}",
            )
        ax_nodes = nodes[axis]
        lo_c = max(float(lo), float(ax_nodes[0]))
        hi_c = min(float(hi), float(ax_nodes[-1]))
        if not lo_c < hi_c:
            raise ValueError(
                f"window range ({lo}, {hi}) along {axis_names[axis]} lies "
                f"outside the domain "
                f"[{ax_nodes[0]}, {ax_nodes[-1]}]",
            )
        i_lo = int(np.argmin(np.abs(ax_nodes - lo_c)))
        i_hi = int(np.argmin(np.abs(ax_nodes - hi_c)))
        if i_hi <= i_lo:
            raise ValueError(
                f"window range ({lo}, {hi}) along {axis_names[axis]} "
                f"collapses to fewer than one grid cell after clipping "
                f"and snapping (nodes {i_lo}..{i_hi}); refine the mesh "
                f"or widen the window",
            )
        windows.append((i_lo, i_hi))

    return windows[0], windows[1]


def _build_uv_edges(
    *,
    primal_axis: int,
    secondary_axis: int,
    normal_axis: int,
    normal_index_E: int,
    normal_index_H: int,
    nodes: tuple,
    centres: tuple,
    deltas: tuple,
    E_off: int,
    E_shape: tuple,
    H_off: int,
    H_shape: tuple,
    primal_cells: tuple[int, int],
    secondary_nodes: tuple[int, int],
) -> tuple:
    """Build flat-E indices, flat-H indices, (u, v) midpoints, edge lengths.

    Constructs one parallel set of port-plane edges along ``primal_axis``
    (which is either the local-u or local-v axis).  The co-located
    dual-edge family along ``secondary_axis`` is also indexed.

    ``primal_cells`` is the inclusive *node* window along the primal
    axis — edges live on the cells ``[lo, hi)``.  ``secondary_nodes``
    is the inclusive node window along the secondary axis — edges sit
    on the nodes ``[lo, hi]``.  For a whole-face plane these are
    ``(0, N_cells_axis)``.

    For example, for X_MIN with u=y, v=z:
        primal_axis=1 (y), secondary_axis=2 (z), normal_axis=0 (x)
        E_y edges at i=0 → midpoints (y_centre[j], z_node[k])
        H_z duals at i=0 → midpoints (y_centre[j], z_node[k])  (co-located)
    """
    # Build (i, j, k) index lists for E-edges in the plane: cells
    # [lo, hi) along the primal axis (E_u edge sits on a u-cell),
    # nodes [lo, hi] along the secondary axis (E edge ends at v-node).
    # All E-edges share the same normal-axis index = normal_index_E.
    p_idx, s_idx = np.meshgrid(
        np.arange(primal_cells[0], primal_cells[1]),
        np.arange(secondary_nodes[0], secondary_nodes[1] + 1),
        indexing="ij",
    )

    # Primal axis: midpoint along axis = cell-centre in primal_axis dim.
    # Secondary axis: edge endpoint = node in secondary_axis dim.
    p_mid = centres[primal_axis][p_idx]
    s_mid = nodes[secondary_axis][s_idx]

    # Edge length along primal axis = cell size at p_idx
    p_len = deltas[primal_axis][p_idx]

    # Convert (n, p, s)-style indices into the global (i, j, k) tuple.
    i_jk_E = _make_ijk(
        normal_axis,
        primal_axis,
        secondary_axis,
        normal_index_E,
        p_idx,
        s_idx,
    )
    e_indices = (
        (E_off + i_jk_E[0] * E_shape[1] * E_shape[2] + i_jk_E[1] * E_shape[2] + i_jk_E[2])
        .ravel()
        .astype(np.int64)
    )

    # H-dual at the co-located (u, v) midpoint.  H-shape's primal-axis
    # dimension is N_cells (centre-staggered), secondary-axis dim is
    # N_nodes.  Same primal-axis cell-index, same secondary-axis node-index.
    i_jk_H = _make_ijk(
        normal_axis,
        primal_axis,
        secondary_axis,
        normal_index_H,
        p_idx,
        s_idx,
    )
    h_indices = (
        (H_off + i_jk_H[0] * H_shape[1] * H_shape[2] + i_jk_H[1] * H_shape[2] + i_jk_H[2])
        .ravel()
        .astype(np.int64)
    )

    # Stack midpoints into (u, v) — primal_axis's centre along its own
    # axis is the *u* component if primal_axis is the u-axis-of-face;
    # else it's the *v* component.  We don't know which here, so we
    # return midpoints in the order (axis_along_primal, axis_along_sec).
    # Caller must interpret this correctly.
    uv = np.stack([p_mid.ravel(), s_mid.ravel()], axis=1)

    return e_indices, h_indices, uv, p_len.ravel()


def _make_ijk(
    normal_axis: int,
    primal_axis: int,
    secondary_axis: int,
    n_idx: int,
    p_idx: np.ndarray,
    s_idx: np.ndarray,
) -> tuple:
    """Compose global (i, j, k) tuple from per-axis indices."""
    out: list = [None, None, None]
    out[normal_axis] = np.full_like(p_idx, n_idx)
    out[primal_axis] = p_idx
    out[secondary_axis] = s_idx
    return tuple(out)


_FACE_OF_AXIS_SIDE: dict[tuple[int, bool], BoxFace] = {
    (0, False): BoxFace.X_MIN,
    (0, True): BoxFace.X_MAX,
    (1, False): BoxFace.Y_MIN,
    (1, True): BoxFace.Y_MAX,
    (2, False): BoxFace.Z_MIN,
    (2, True): BoxFace.Z_MAX,
}


def resolve_port_edge_pec(
    plane: PortPlane,
    mesh: Mesh,
    pec_faces,
) -> dict[str, bool]:
    """Per-port-edge PEC flags via the legacy ``_edge_bc`` rule.
    For each of the four edges of the (possibly sub-face) port plane:
    - the edge lies **on a domain boundary** → it inherits that lateral
      wall's boundary condition;
    - the edge is **interior** (a sub-face window boundary inside the
      face) → it inherits the *port face's* boundary condition (the
      wall the port is embedded in).
    Only the PEC/non-PEC distinction matters for the 2D mode problem:
    a PEC edge is Dirichlet-eliminated, anything else (PMC, open) is
    the natural Neumann boundary.
    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry (carries the node windows).
    mesh : Mesh
        The 3D mesh (provides the per-axis cell counts that decide
        whether a window boundary coincides with the domain boundary).
    pec_faces : collection of BoxFace
        The set of bbox faces with a PEC boundary condition — e.g. the
        set ``AnalysisScatteringTD`` consolidates via
        ``with_pec_boundaries``.
    Returns
    -------
    dict[str, bool]
        Keys ``"u_min"``, ``"u_max"``, ``"v_min"``, ``"v_max"`` (the
        plane's local frame); ``True`` where the edge is PEC.
    """
    pec = set(pec_faces)
    n_cells = (mesh.Nx, mesh.Ny, mesh.Nz)
    face_pec = plane.face in pec
    out: dict[str, bool] = {}
    for name, axis, window, at_lo in (
        ("u_min", plane.face.u_axis, plane.u_node_window, True),
        ("u_max", plane.face.u_axis, plane.u_node_window, False),
        ("v_min", plane.face.v_axis, plane.v_node_window, True),
        ("v_max", plane.face.v_axis, plane.v_node_window, False),
    ):
        if at_lo:
            on_domain_boundary = window[0] == 0
        else:
            on_domain_boundary = window[1] == n_cells[axis]
        if on_domain_boundary:
            wall = _FACE_OF_AXIS_SIDE[(axis, not at_lo)]
            out[name] = wall in pec
        else:
            out[name] = face_pec
    return out


def build_port_edge_pec_mask(
    plane: PortPlane,
    edge_pec: dict[str, bool],
) -> np.ndarray:
    """Boolean ``[e_u | e_v]`` mask of edges tangent to PEC port edges.
    Marks the plane's own boundary edges (the window boundary for a
    sub-face plane) according to *edge_pec* from
    :func:`resolve_port_edge_pec`: the ``u_min``/``u_max`` port edges
    run along v, so the tangent edges are the v-edges whose u-node sits
    on that boundary; ``v_min``/``v_max`` analogously mark u-edges.
    The result has the same ``[e_u | e_v]`` basis as the factory's PEC
    resolver and is meant to be OR-ed onto it before the Dirichlet
    elimination in the 2D mode solve.
    """
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    mask_u = np.zeros((plane.n_cells_u, plane.n_nodes_v), dtype=bool)
    mask_v = np.zeros((plane.n_cells_v, plane.n_nodes_u), dtype=bool)
    if edge_pec.get("v_min", False):
        mask_u[:, 0] = True
    if edge_pec.get("v_max", False):
        mask_u[:, -1] = True
    if edge_pec.get("u_min", False):
        mask_v[:, 0] = True
    if edge_pec.get("u_max", False):
        mask_v[:, -1] = True
    out = np.concatenate([mask_u.ravel(), mask_v.ravel()])
    if out.size != n_u + n_v:
        raise ValueError(
            f"edge-raster size {out.size} does not match the plane's "
            f"[e_u | e_v] basis ({n_u} + {n_v})."
        )
    return out


_BBOX_FACE_NAMES = (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))


def _is_pmc_face(boundary_conditions, axis: int, at_lo: bool) -> bool:
    """True when the bbox face of ``axis`` (lo/hi side) is a PMC wall."""
    if boundary_conditions is None:
        return False
    face = _BBOX_FACE_NAMES[axis][0 if at_lo else 1]
    return getattr(boundary_conditions, face, None) == "PMC"


def window_domain_faces(plane: PortPlane, grid) -> dict[str, str]:
    """Bbox-face name of every window edge that lies on a domain boundary.

    Purely geometric (no boundary-condition knowledge): for each of the
    four lateral edges of the (possibly sub-face) port window, decide
    whether it coincides with a domain bbox face and name that face.
    Keys are the plane's local edges ``"u_min"``/``"u_max"``/``"v_min"``/
    ``"v_max"``, values are ``"xmin"`` … ``"zmax"``; interior window
    edges are absent.  This is the shared geometric core of
    :func:`magnetic_window_ends` and the symmetry-aware port report
    (DD-154 — a symmetry plane cutting the port window makes the port
    a half port).
    """
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    u_axis = plane.face.u_axis
    v_axis = plane.face.v_axis
    u_lo, u_hi = plane.u_node_window
    v_lo, v_hi = plane.v_node_window
    out: dict[str, str] = {}
    if u_lo == 0:
        out["u_min"] = _BBOX_FACE_NAMES[u_axis][0]
    if u_hi == n_cells[u_axis]:
        out["u_max"] = _BBOX_FACE_NAMES[u_axis][1]
    if v_lo == 0:
        out["v_min"] = _BBOX_FACE_NAMES[v_axis][0]
    if v_hi == n_cells[v_axis]:
        out["v_max"] = _BBOX_FACE_NAMES[v_axis][1]
    return out


def magnetic_window_ends(
    plane: PortPlane,
    grid,
    boundary_conditions,
) -> tuple[bool, bool, bool, bool]:
    """Window ends ``(u_lo, u_hi, v_lo, v_hi)`` that sit on a PMC bbox face.

    The natural magnetic wall of the staggered grid lies half the outer
    dual cell BEYOND the outermost grid line (the mesher's pulled-in
    line places that wall on the bbox face; a post-meshing declaration
    leaves it half a cell outside).  Quadratures that book physical
    widths on the port plane — the TEM/QTEM capacitance integrals and
    the physical-power Poynting patches — must therefore extend the end
    dual to the full boundary cell at exactly these window ends; at
    every other window end the wall (or the window cut) is ON the line
    and the half-cell convention stands.
    """
    if boundary_conditions is None:
        return (False, False, False, False)
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    u_axis = plane.face.u_axis
    v_axis = plane.face.v_axis
    u_lo, u_hi = plane.u_node_window
    v_lo, v_hi = plane.v_node_window
    return (
        u_lo == 0 and _is_pmc_face(boundary_conditions, u_axis, at_lo=True),
        u_hi == n_cells[u_axis] and _is_pmc_face(boundary_conditions, u_axis, at_lo=False),
        v_lo == 0 and _is_pmc_face(boundary_conditions, v_axis, at_lo=True),
        v_hi == n_cells[v_axis] and _is_pmc_face(boundary_conditions, v_axis, at_lo=False),
    )
