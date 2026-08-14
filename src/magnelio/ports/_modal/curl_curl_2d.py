"""2D curl-curl operator construction for the port-plane mode problem.

Builds the generalised eigenvalue problem ``K · ê = ω_c² · M · ê`` for
modes living on a port plane, by index-sliced restriction of the existing
3D FIT primal-curl matrix and the diagonal material mass matrices.

Per `reference_architecture_phase2_mode_solver.md` §3.2 (Phase 2a, step 1)
the 2D operators share their discretisation with the 3D operators by
construction — that is the property Reference §2.2 calls "in the same
metric" and is the reason the analytical-projection residue from Phase 1
(Reference §4.4) cannot occur in the numerical mode-solver path.

For a port plane on bbox face ``f`` (normal axis ``n``):

- **Primal 2D edges** (where ``E_u`` and ``E_v`` live) are the tangential
  primal E-edges at the plane — exposed by ``PortPlane`` as
  ``e_u_indices`` and ``e_v_indices``.
- **Normal 2D faces** (where the normal H-component lives) are H-faces
  at the plane along the normal axis (``H_x`` faces at ``i_n = 0`` for
  ``X_MIN``, etc.).  These are *not* the same as ``h_u_indices`` /
  ``h_v_indices`` (which are the *tangential* dual edges co-located with
  the primal E-edges, used for the modal V/I projection).

The primal 2D curl ``C_2D`` is then the row/column slice of the 3D
``C_primal``: rows = normal-H faces at the plane, columns = tangential
primal E-edges at the plane.  ``K = C_2D^T · diag(M_μ⁻¹) · C_2D`` is
symmetric positive-semidefinite (gradient null-space) and ``M = M_ε``
restricted to the same primal 2D edges is diagonal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from magnelio.mesh.grid import GridLines
from magnelio.ports._modal.port_plane import BoxFace, PortPlane


def _safe_mu_inv(m_mu: np.ndarray) -> np.ndarray:
    """Exact ``1/M_μ`` with 0 on frozen faces.

    ``M_μ = 0`` marks enlarged-cell-donated H-faces (WP-R5, ~fully
    inside PEC); the 3D update freezes their h exactly, so the 2D
    restriction removes them from the curl-curl operator the same way.
    """
    return np.where(m_mu > 0, 1.0 / np.where(m_mu > 0, m_mu, 1.0), 0.0)


def build_2d_curl_curl(
    plane: PortPlane,
    grid: GridLines,
    M_eps_diag: np.ndarray,
    M_mu_diag: np.ndarray,
    C_3d: sp.csr_matrix,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Restrict the 3D FIT operators to the 2D mode problem on ``plane``.

    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry (already built via ``PortPlane.from_mesh``).
    grid : GridLines
        Underlying FIT grid; needed only to compute the flat-H indices for
        the normal-H component at the plane (the indices of the H-faces
        whose row in the 3D primal-curl matrix becomes the 2D-curl row).
    M_eps_diag : np.ndarray, shape (n_E,)
        Diagonal of the 3D electric-mass matrix, in the canonical
        ``[Ex | Ey | Ez]`` ordering.
    M_mu_diag : np.ndarray, shape (n_H,)
        Diagonal of the 3D magnetic-mass matrix, in the canonical
        ``[Hx | Hy | Hz]`` ordering.
    C_3d : scipy.sparse.csr_matrix, shape (n_H, n_E)
        Full 3D primal-curl matrix from
        :func:`magnelio._operators.curl.build_curl_matrix`.

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (n_2d, n_2d)
        2D curl-curl stiffness ``C_2D^T · diag(M_μ⁻¹) · C_2D``.
        Symmetric and positive-semidefinite (a non-trivial gradient
        null-space exists in general — its dimension equals the number
        of interior-tangent-only nodes on the plane).
    M : scipy.sparse.csr_matrix, shape (n_2d, n_2d)
        2D mass matrix; the diagonal ``M_ε`` restricted to the
        primal 2D edges.
    primal_2d_indices : np.ndarray of int, shape (n_2d,)
        Concatenation of ``plane.e_u_indices`` and ``plane.e_v_indices``,
        in that order.  This is the basis ordering of the 2D edge
        vector ``ê`` in the global flat-E layout — callers use it to
        scatter eigenvectors back into a 3D edge state.

    Notes
    -----
    The eigenvalues of the generalised problem ``K · ê = λ · M · ê`` are
    ``λ = ω_c²`` for the modes on the cross-section (see Reference §2.1).
    The smallest non-zero eigenvalue equals the cutoff frequency squared
    of the lowest-cutoff TE/TM mode, modulo the discretisation error
    (≤ 1 % at typical resolutions per `reference_architecture_phase2_mode_solver.md`
    §4 Test 1N).

    Lateral PEC boundary conditions are *not* applied here — this
    function returns the raw operators on all primal 2D edges.  Callers
    apply Dirichlet BCs by row/column elimination of the wall-tangent
    edges before passing ``K, M`` to ``scipy.sparse.linalg.eigsh``.
    """
    h_n_indices = _normal_h_indices(
        plane.face,
        grid,
        u_window=plane.u_node_window,
        v_window=plane.v_node_window,
    )
    primal_2d = np.concatenate([plane.e_u_indices, plane.e_v_indices])

    # Slice rows then columns; both are CSR-friendly fancy-index ops.
    C_2d = C_3d[h_n_indices, :][:, primal_2d].tocsr()

    mu_inv = sp.diags(_safe_mu_inv(M_mu_diag[h_n_indices]), format="csr")
    K = (C_2d.T @ mu_inv @ C_2d).tocsr()
    M = sp.diags(M_eps_diag[primal_2d], format="csr")

    return K, M, primal_2d


def build_2d_tm_curl_curl(
    plane: PortPlane,
    grid: GridLines,
    M_eps_diag: np.ndarray,
    M_mu_diag: np.ndarray,
    C_3d: sp.csr_matrix,
    pec_mask_edges: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """Exact TM eigenproblem on the port plane's normal-E edges.

    The TM cut-off oscillation of the discrete system is the pure
    (e_n, h_t) resonance of one longitudinal cell: at ``beta = 0``
    the normal-E edges at the port-adjacent half-plane exchange
    energy with the co-located transversal H faces only, so the
    exact discrete TM eigenproblem is

        K_z · ê_n = ω̂_c² · M_z · ê_n,
        K_z = C_s^T · diag(M_μ⁻¹) · C_s,   M_z = M_ε[normal-E edges],

    with ``C_s`` the row/column slice of the 3D primal curl (rows =
    transversal H faces at the half-plane — ``plane.h_u_indices`` /
    ``h_v_indices`` —, columns = the normal-E edges).  This is the
    TM counterpart of :func:`build_2d_curl_curl` and shares its
    metric-exactness: the eigenvalue is the *discrete* cut-off of
    the 3D operator (``q = ω̂_c·dt`` in the Klein-Gordon chain), and
    the eigenvector reproduces the 3D transversal action to machine
    precision — including conformal cross-sections, where the
    former lumped node-Laplace (``build_2d_node_laplace``) deviated
    at the 1e-5 level (WP-R3 pre-check,
    ``validation/kg_dtbc_precheck_spike.py``).

    Given the DD-053 co-located pair identity (uniform
    ``M_ε·M_μ`` pair product across the section), the remaining
    modal-subspace closure conditions hold automatically and the
    transversal E profile is the *topological* gradient of the
    eigenvector, ``ê_t ∝ G_2d · ê_n`` — the construction already
    used by :meth:`Numerical2DModeSolver._solve_tm`.

    Parameters
    ----------
    plane, grid, M_eps_diag, M_mu_diag, C_3d
        As in :func:`build_2d_curl_curl`.
    pec_mask_edges : np.ndarray, optional
        ``mesh.pec_mask_edges`` (shape ``(3, n_max)``).  The
        Dirichlet mask for the eigenproblem is the 3D PEC state of
        the normal-E edges themselves — exactly the edges the 3D
        solver zeroes (post-DD-053 the tangential-surface re-masking
        guarantees conductor-footprint normal edges are included).
        ``None`` falls back to the hollow-waveguide window boundary
        (all four lateral walls), matching the bare-mesh fallback of
        the TE path.

    Returns
    -------
    K_z : scipy.sparse.csr_matrix, shape (n_n, n_n)
        TM stiffness on the normal-E edges.  Symmetric positive
        semidefinite; positive definite after Dirichlet elimination
        on a hollow cross-section.
    M_z : scipy.sparse.csr_matrix, shape (n_n, n_n)
        Diagonal ``M_ε`` restricted to the normal-E edges.
    dirichlet_mask : np.ndarray of bool, shape (n_n,)
        ``True`` where the eigenvector is constrained to zero (PEC
        or massless edges).
    normal_e_indices : np.ndarray of int, shape (n_n,)
        Flat 3D E indices of the normal-E edges, in the local
        (u, v) node-raster order of :func:`build_2d_gradient`'s
        ``primal_2d_node_indices`` — the 1:1 node correspondence
        that makes ``G_2d @ ê_n`` well-defined.
    """
    e_n = _normal_e_indices(
        plane.face,
        grid,
        u_window=plane.u_node_window,
        v_window=plane.v_node_window,
    )
    rows = np.concatenate([plane.h_u_indices, plane.h_v_indices])

    C_s = C_3d[rows, :][:, e_n].tocsr()
    mu_inv = sp.diags(_safe_mu_inv(M_mu_diag[rows]), format="csr")
    K_z = (C_s.T @ mu_inv @ C_s).tocsr()

    m_z = M_eps_diag[e_n]
    M_z = sp.diags(m_z, format="csr")

    if pec_mask_edges is None:
        dirichlet = _hollow_pec_node_mask(plane, grid)
    else:
        pec_flat = _flat_e_pec(pec_mask_edges, grid)
        dirichlet = pec_flat[e_n]
    dirichlet = dirichlet | (m_z <= 0.0)

    return K_z, M_z, dirichlet, e_n


def _flat_e_pec(pec_mask_edges: np.ndarray, grid: GridLines) -> np.ndarray:
    """Concatenate the (3, n_max) PEC edge mask into flat-E ordering."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    return np.concatenate(
        [
            pec_mask_edges[0, :n_Ex],
            pec_mask_edges[1, :n_Ey],
            pec_mask_edges[2, :n_Ez],
        ]
    )


def _normal_e_indices(
    face: BoxFace,
    grid: GridLines,
    u_window: tuple[int, int] | None = None,
    v_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Flat E indices of the normal-component edges at the port slab.

    One edge per port-plane node (the normal edge starting at that
    node and running into the domain), rastered in the same local
    (u, v) order as :func:`_plane_node_indices` so the two index
    sets correspond 1:1.  For MIN faces the edge layer along the
    normal axis is 0, for MAX faces ``N_axis - 1`` — the same
    half-plane as ``plane.h_u_indices`` / ``h_v_indices``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_axis = face.normal_axis
    N_axis = (Nx, Ny, Nz)[n_axis]
    layer = N_axis - 1 if face.is_max else 0

    u_lo, u_hi = (
        u_window
        if u_window is not None
        else (
            0,
            (Nx, Ny, Nz)[face.u_axis],
        )
    )
    v_lo, v_hi = (
        v_window
        if v_window is not None
        else (
            0,
            (Nx, Ny, Nz)[face.v_axis],
        )
    )
    iu_idx, iv_idx = np.meshgrid(
        np.arange(u_lo, u_hi + 1),
        np.arange(v_lo, v_hi + 1),
        indexing="ij",
    )

    out_ijk: list = [None, None, None]
    out_ijk[n_axis] = np.full_like(iu_idx, layer)
    out_ijk[face.u_axis] = iu_idx
    out_ijk[face.v_axis] = iv_idx
    i, j, k = out_ijk

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    if n_axis == 0:
        flat = i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
    elif n_axis == 1:
        flat = n_Ex + i * Ny * (Nz + 1) + j * (Nz + 1) + k
    else:
        flat = n_Ex + n_Ey + i * (Ny + 1) * Nz + j * Nz + k
    return flat.ravel().astype(np.int64)


def build_2d_gradient(
    plane: PortPlane,
    grid: GridLines,
    G_3d: sp.csr_matrix,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Restrict the 3D gradient operator to the 2D port-plane problem.

    Companion to :func:`build_2d_curl_curl` for the TEM Laplace path
    (`reference_architecture_phase2_mode_solver.md` §3.3, Phase 2b
    step 7).  The 2D primal nodes of the port plane are the 3D primal
    nodes whose normal-axis index sits at the plane (0 for MIN faces,
    ``N_axis`` for MAX faces).  Tangential primal 2D edges (those
    indexed by ``plane.e_u_indices`` / ``plane.e_v_indices``) connect
    *only* nodes on this plane, so the slice is self-contained.

    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry.
    grid : GridLines
        Underlying FIT grid; needed to enumerate the 2D primal-node
        indices in the global flat-node ordering.
    G_3d : scipy.sparse.csr_matrix, shape (n_E, n_nodes)
        Full 3D primal-gradient matrix from
        :func:`magnelio._operators.curl.build_gradient_matrix`.

    Returns
    -------
    G_2d : scipy.sparse.csr_matrix, shape (n_2d_edges, n_2d_nodes)
        2D discrete gradient (nodes → edges) with topological ±1
        entries.  Inherits exactness ``C_2d @ G_2d == 0`` from the 3D
        de Rham complex (where ``C_2d`` is the operator inside
        :func:`build_2d_curl_curl`).
    primal_2d_node_indices : np.ndarray of int, shape (n_2d_nodes,)
        Flat 3D node-vector indices for the 2D primal nodes on the
        plane, in local (u, v) raster ordering — i.e.
        ``out[i_u * Nv_node + i_v]`` is the flat 3D node index for the
        2D node at local ``(i_u, i_v)``.  Conductor groups passed to
        :func:`solve_tem_laplace` index into this basis.
    primal_2d_edge_indices : np.ndarray of int, shape (n_2d_edges,)
        Concatenation of ``plane.e_u_indices`` and
        ``plane.e_v_indices`` — identical to
        :func:`build_2d_curl_curl`'s ``primal_2d_indices`` output.
    """
    primal_2d_nodes = _plane_node_indices(
        plane.face,
        grid,
        u_window=plane.u_node_window,
        v_window=plane.v_node_window,
    )
    primal_2d_edges = np.concatenate([plane.e_u_indices, plane.e_v_indices])
    G_2d = G_3d[primal_2d_edges, :][:, primal_2d_nodes].tocsr()
    return G_2d, primal_2d_nodes, primal_2d_edges


def _hollow_pec_node_mask(plane: PortPlane, grid: GridLines) -> np.ndarray:
    """Default PEC mask for a hollow waveguide: all u/v boundary nodes.

    Returned in the same flat (u, v) raster order as
    :func:`_plane_node_indices`.  For a sub-face plane the boundary is
    the plane's own window boundary.
    """
    del grid  # sizes come from the plane's node windows
    Nu_node = plane.n_nodes_u
    Nv_node = plane.n_nodes_v
    iu_idx, iv_idx = np.meshgrid(np.arange(Nu_node), np.arange(Nv_node), indexing="ij")
    mask = (iu_idx == 0) | (iu_idx == Nu_node - 1) | (iv_idx == 0) | (iv_idx == Nv_node - 1)
    return mask.ravel()


def _plane_node_indices(
    face: BoxFace,
    grid: GridLines,
    u_window: tuple[int, int] | None = None,
    v_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Flat 3D node-vector indices for the 2D primal nodes on the port plane.

    The 3D node ordering is ``node(i, j, k) = i·(Ny+1)·(Nz+1) + j·(Nz+1) + k``
    (matches :func:`magnelio._operators.curl.build_gradient_matrix`).  The
    2D nodes on the plane are extracted in local (u, v) raster ordering
    so that ``out[i_u_rel * Nv_node + i_v_rel]`` is the global flat index.

    ``u_window`` / ``v_window`` are inclusive node-index windows along
    the local u / v axis (``PortPlane.u_node_window`` /
    ``v_node_window``); ``None`` covers the whole face.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_axis = face.normal_axis
    u_lo, u_hi = (
        u_window
        if u_window is not None
        else (
            0,
            (Nx, Ny, Nz)[face.u_axis],
        )
    )
    v_lo, v_hi = (
        v_window
        if v_window is not None
        else (
            0,
            (Nx, Ny, Nz)[face.v_axis],
        )
    )
    N_axis = (Nx, Ny, Nz)[n_axis]
    n_node_idx = 0 if not face.is_max else N_axis

    iu_idx, iv_idx = np.meshgrid(
        np.arange(u_lo, u_hi + 1), np.arange(v_lo, v_hi + 1), indexing="ij"
    )

    out_ijk: list = [None, None, None]
    out_ijk[n_axis] = np.full_like(iu_idx, n_node_idx)
    out_ijk[face.u_axis] = iu_idx
    out_ijk[face.v_axis] = iv_idx

    flat = out_ijk[0] * (Ny + 1) * (Nz + 1) + out_ijk[1] * (Nz + 1) + out_ijk[2]
    return flat.ravel().astype(np.int64)


def _normal_h_indices(
    face: BoxFace,
    grid: GridLines,
    u_window: tuple[int, int] | None = None,
    v_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Flat-H-vector indices for the normal-component H-faces at the plane.

    For ``X_MIN`` / ``X_MAX``: ``H_x`` faces at ``i_n ∈ {0, Nx}`` —
    shape-component is ``(Nx + 1, Ny, Nz)``.  For ``Y_MIN`` / ``Y_MAX``:
    ``H_y`` faces at ``j_n ∈ {0, Ny}`` — shape ``(Nx, Ny + 1, Nz)``.  For
    ``Z_MIN`` / ``Z_MAX``: ``H_z`` faces at ``k_n ∈ {0, Nz}`` — shape
    ``(Nx, Ny, Nz + 1)``.

    ``u_window`` / ``v_window`` are inclusive node-index windows along
    the plane's local u / v axis; the H-faces returned live on the cells
    ``[lo, hi)``.  ``None`` covers the whole face.  The raster order is
    along the *global* tangential axes of the relevant H-shape (not
    (u, v)) — permissible because callers only use the index set for
    row slicing of the 3D curl, and ``K = C᷀ᵀ·μ⁻¹·C᷀`` is invariant
    under row permutations.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz

    # Cell windows keyed by global axis (u_window belongs to the
    # face's u_axis, which is *not* the lower-numbered tangential axis
    # on MAX-type faces).
    cells = {
        face.u_axis: u_window if u_window is not None else (0, (Nx, Ny, Nz)[face.u_axis]),
        face.v_axis: v_window if v_window is not None else (0, (Nx, Ny, Nz)[face.v_axis]),
    }

    if face in (BoxFace.X_MIN, BoxFace.X_MAX):
        i_n = 0 if face is BoxFace.X_MIN else Nx
        # Hx[i_n, j, k] — flat = i_n * Ny * Nz + j * Nz + k
        j_idx, k_idx = np.meshgrid(np.arange(*cells[1]), np.arange(*cells[2]), indexing="ij")
        flat = i_n * Ny * Nz + j_idx * Nz + k_idx
        return flat.ravel().astype(np.int64)

    if face in (BoxFace.Y_MIN, BoxFace.Y_MAX):
        j_n = 0 if face is BoxFace.Y_MIN else Ny
        # Hy[i, j_n, k] — flat = i * (Ny + 1) * Nz + j_n * Nz + k
        i_idx, k_idx = np.meshgrid(np.arange(*cells[0]), np.arange(*cells[2]), indexing="ij")
        flat_local = i_idx * (Ny + 1) * Nz + j_n * Nz + k_idx
        return (flat_local + n_Hx).ravel().astype(np.int64)

    if face in (BoxFace.Z_MIN, BoxFace.Z_MAX):
        k_n = 0 if face is BoxFace.Z_MIN else Nz
        # Hz[i, j, k_n] — flat = i * Ny * (Nz + 1) + j * (Nz + 1) + k_n
        i_idx, j_idx = np.meshgrid(np.arange(*cells[0]), np.arange(*cells[1]), indexing="ij")
        flat_local = i_idx * Ny * (Nz + 1) + j_idx * (Nz + 1) + k_n
        return (flat_local + n_Hx + n_Hy).ravel().astype(np.int64)

    raise ValueError(f"Unhandled face: {face!r}")
