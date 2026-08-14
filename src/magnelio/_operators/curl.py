"""
Discrete differential operators for FIT on the Yee grid.

De Rham complex:  nodes →(G) edges →(C) faces →(D) volumes

    G : discrete gradient  — shape (n_E, n_nodes)
    C : discrete curl      — shape (n_H, n_E)

Exactness: C @ G = 0  (curl of gradient vanishes).

For a grid of size Nx × Ny × Nz:
    - n_nodes = (Nx+1)(Ny+1)(Nz+1)
    - n_E = 3 * Ne  (E edges: Ex, Ey, Ez components flattened)
    - n_H = 3 * Nf  (H faces: Hx, Hy, Hz components flattened)

See spec.md for mathematical details of the discrete operators.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from magnelio.mesh.grid import GridLines


def build_curl_matrix(grid: GridLines) -> sp.csr_matrix:
    """Build the discrete curl matrix C for the given grid.

    Returns a sparse CSR matrix of shape ``(3*Nf, 3*Ne)`` where each row
    has exactly two non-zero entries (+1 and -1).

    Args:
        grid: :class:`~magnelio.mesh.grid.GridLines` defining the mesh topology.

    Returns:
        C as a ``scipy.sparse.csr_matrix``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # Edge counts
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz

    # Face counts
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)

    n_E = n_Ex + n_Ey + n_Ez
    n_H = n_Hx + n_Hy + n_Hz

    rows = []
    cols = []
    data = []

    # Offsets for block indexing
    off_Ex = 0
    off_Ey = n_Ex
    off_Ez = n_Ex + n_Ey
    off_Hx = 0
    off_Hy = n_Hx
    off_Hz = n_Hx + n_Hy

    # --- Hx rows: (curl E)_x = dEz/dy - dEy/dz ---
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                row = off_Hx + i * Ny * Nz + j * Nz + k

                # +dEz/dy: Ez[i, j+1, k] - Ez[i, j, k]
                col_Ez_jp1 = off_Ez + i * (Ny + 1) * Nz + (j + 1) * Nz + k
                col_Ez_j = off_Ez + i * (Ny + 1) * Nz + j * Nz + k
                rows += [row, row]
                cols += [col_Ez_jp1, col_Ez_j]
                data += [+1.0, -1.0]

                # -dEy/dz: -(Ey[i, j, k+1] - Ey[i, j, k])
                col_Ey_kp1 = off_Ey + i * Ny * (Nz + 1) + j * (Nz + 1) + (k + 1)
                col_Ey_k = off_Ey + i * Ny * (Nz + 1) + j * (Nz + 1) + k
                rows += [row, row]
                cols += [col_Ey_kp1, col_Ey_k]
                data += [-1.0, +1.0]

    # --- Hy rows: (curl E)_y = dEx/dz - dEz/dx ---
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz):
                row = off_Hy + i * (Ny + 1) * Nz + j * Nz + k

                # +dEx/dz: Ex[i, j, k+1] - Ex[i, j, k]
                col_Ex_kp1 = off_Ex + i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + (k + 1)
                col_Ex_k = off_Ex + i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
                rows += [row, row]
                cols += [col_Ex_kp1, col_Ex_k]
                data += [+1.0, -1.0]

                # -dEz/dx: -(Ez[i+1, j, k] - Ez[i, j, k])
                col_Ez_ip1 = off_Ez + (i + 1) * (Ny + 1) * Nz + j * Nz + k
                col_Ez_i = off_Ez + i * (Ny + 1) * Nz + j * Nz + k
                rows += [row, row]
                cols += [col_Ez_ip1, col_Ez_i]
                data += [-1.0, +1.0]

    # --- Hz rows: (curl E)_z = dEy/dx - dEx/dy ---
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz + 1):
                row = off_Hz + i * Ny * (Nz + 1) + j * (Nz + 1) + k

                # +dEy/dx: Ey[i+1, j, k] - Ey[i, j, k]
                col_Ey_ip1 = off_Ey + (i + 1) * Ny * (Nz + 1) + j * (Nz + 1) + k
                col_Ey_i = off_Ey + i * Ny * (Nz + 1) + j * (Nz + 1) + k
                rows += [row, row]
                cols += [col_Ey_ip1, col_Ey_i]
                data += [+1.0, -1.0]

                # -dEx/dy: -(Ex[i, j+1, k] - Ex[i, j, k])
                col_Ex_jp1 = off_Ex + i * (Ny + 1) * (Nz + 1) + (j + 1) * (Nz + 1) + k
                col_Ex_j = off_Ex + i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
                rows += [row, row]
                cols += [col_Ex_jp1, col_Ex_j]
                data += [-1.0, +1.0]

    C = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_H, n_E),
        dtype=float,
    )
    return C


def curl_e_stencil(Ex, Ey, Ez, out_Hx, out_Hy, out_Hz):
    """Compute primal curl (C @ e) via direct stencil differences.

    Writes the result into pre-allocated output buffers (no allocation).
    Equivalent to ``C @ e_flat`` where C is from :func:`build_curl_matrix`,
    but significantly faster for large grids due to stride-regular memory
    access (no index indirection).

    Parameters
    ----------
    Ex, Ey, Ez : ndarray
        E-field component arrays (Yee-staggered shapes).
    out_Hx, out_Hy, out_Hz : ndarray
        Pre-allocated output buffers (Yee H-field shapes).
    """
    # (curl E)_x = dEz/dy - dEy/dz
    np.subtract(Ez[:, 1:, :], Ez[:, :-1, :], out=out_Hx)
    out_Hx -= Ey[:, :, 1:]
    out_Hx += Ey[:, :, :-1]

    # (curl E)_y = dEx/dz - dEz/dx
    np.subtract(Ex[:, :, 1:], Ex[:, :, :-1], out=out_Hy)
    out_Hy -= Ez[1:, :, :]
    out_Hy += Ez[:-1, :, :]

    # (curl E)_z = dEy/dx - dEx/dy
    np.subtract(Ey[1:, :, :], Ey[:-1, :, :], out=out_Hz)
    out_Hz -= Ex[:, 1:, :]
    out_Hz += Ex[:, :-1, :]


def curl_h_stencil(Hx, Hy, Hz, out_Ex, out_Ey, out_Ez):
    """Compute dual curl (C^T @ h) via direct stencil differences.

    Writes the result into pre-allocated output buffers.
    Boundary E-edges that have fewer than four H-face neighbours
    naturally receive fewer contributions (implicit zero padding).

    Parameters
    ----------
    Hx, Hy, Hz : ndarray
        H-field component arrays (Yee-staggered shapes).
    out_Ex, out_Ey, out_Ez : ndarray
        Pre-allocated output buffers (Yee E-field shapes).
    """
    # (C^T h)_Ex = dHz/dy - dHy/dz
    out_Ex[:] = 0.0
    out_Ex[:, :-1, :] += Hz
    out_Ex[:, 1:, :] -= Hz
    out_Ex[:, :, :-1] -= Hy
    out_Ex[:, :, 1:] += Hy

    # (C^T h)_Ey = dHx/dz - dHz/dx
    out_Ey[:] = 0.0
    out_Ey[:, :, :-1] += Hx
    out_Ey[:, :, 1:] -= Hx
    out_Ey[:-1, :, :] -= Hz
    out_Ey[1:, :, :] += Hz

    # (C^T h)_Ez = dHy/dx - dHx/dy
    out_Ez[:] = 0.0
    out_Ez[:-1, :, :] += Hy
    out_Ez[1:, :, :] -= Hy
    out_Ez[:, :-1, :] -= Hx
    out_Ez[:, 1:, :] += Hx


def build_gradient_matrix(grid: GridLines) -> sp.csr_matrix:
    """Build the discrete gradient matrix G (nodes → edges).

    Returns a sparse CSR matrix of shape ``(n_E, n_nodes)`` where each row
    has exactly two non-zero entries (−1 at the start node, +1 at the end
    node of the edge).

    Node ordering: ``node(i,j,k) = i·(Ny+1)·(Nz+1) + j·(Nz+1) + k``.

    The exactness property ``C @ G == 0`` holds by construction, where
    ``C`` is from :func:`build_curl_matrix`.

    Parameters
    ----------
    grid : GridLines
        Grid defining the mesh topology.

    Returns
    -------
    G : scipy.sparse.csr_matrix, shape ``(n_E, n_nodes)``
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_E = n_Ex + n_Ey + n_Ez
    n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)

    stride_i = (Ny + 1) * (Nz + 1)
    stride_j = Nz + 1

    all_rows = []
    all_cols_s = []  # start-node columns (coefficient −1)
    all_cols_e = []  # end-node columns   (coefficient +1)

    # --- Ex edges: connects node(i,j,k) → node(i+1,j,k) ----------------
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    flat = ii.ravel()
    jf, kf = jj.ravel(), kk.ravel()
    edge = flat * (Ny + 1) * (Nz + 1) + jf * (Nz + 1) + kf
    node_s = flat * stride_i + jf * stride_j + kf
    node_e = (flat + 1) * stride_i + jf * stride_j + kf
    all_rows.append(edge)
    all_cols_s.append(node_s)
    all_cols_e.append(node_e)

    # --- Ey edges: connects node(i,j,k) → node(i,j+1,k) ----------------
    off_Ey = n_Ex
    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), np.arange(Nz + 1), indexing="ij")
    flat = ii.ravel()
    jf, kf = jj.ravel(), kk.ravel()
    edge = off_Ey + flat * Ny * (Nz + 1) + jf * (Nz + 1) + kf
    node_s = flat * stride_i + jf * stride_j + kf
    node_e = flat * stride_i + (jf + 1) * stride_j + kf
    all_rows.append(edge)
    all_cols_s.append(node_s)
    all_cols_e.append(node_e)

    # --- Ez edges: connects node(i,j,k) → node(i,j,k+1) ----------------
    off_Ez = n_Ex + n_Ey
    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz), indexing="ij")
    flat = ii.ravel()
    jf, kf = jj.ravel(), kk.ravel()
    edge = off_Ez + flat * (Ny + 1) * Nz + jf * Nz + kf
    node_s = flat * stride_i + jf * stride_j + kf
    node_e = flat * stride_i + jf * stride_j + (kf + 1)
    all_rows.append(edge)
    all_cols_s.append(node_s)
    all_cols_e.append(node_e)

    rows = np.concatenate(all_rows)
    cols_s = np.concatenate(all_cols_s)
    cols_e = np.concatenate(all_cols_e)

    G = sp.csr_matrix(
        (
            np.concatenate([np.full(len(rows), -1.0), np.full(len(rows), +1.0)]),
            (np.concatenate([rows, rows]), np.concatenate([cols_s, cols_e])),
        ),
        shape=(n_E, n_nodes),
        dtype=float,
    )
    return G
