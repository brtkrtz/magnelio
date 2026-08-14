"""
Yee-cell, edge, and face index helpers.

Provides utilities for mapping between 3D (i, j, k) indices and flat array
indices for edge-based (E-field) and face-based (H-field) quantities.

Also provides PEC edge mask construction.
"""

from __future__ import annotations

import numpy as np

from magnelio.mesh.grid import GridLines


def edge_index_Ex(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Ex edge at cell (i, j, k).

    Ex lives on the x-edges of the primary grid.
    Grid: Ex shape is (Nx, Ny+1, Nz+1).
    """
    return i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k


def edge_index_Ey(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Ey edge at cell (i, j, k).

    Ey shape: (Nx+1, Ny, Nz+1).
    """
    return i * Ny * (Nz + 1) + j * (Nz + 1) + k


def edge_index_Ez(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Ez edge at cell (i, j, k).

    Ez shape: (Nx+1, Ny+1, Nz).
    """
    return i * (Ny + 1) * Nz + j * Nz + k


def face_index_Hx(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Hx face at (i, j, k).

    Hx shape: (Nx+1, Ny, Nz).
    """
    return i * Ny * Nz + j * Nz + k


def face_index_Hy(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Hy face at (i, j, k).

    Hy shape: (Nx, Ny+1, Nz).
    """
    return i * (Ny + 1) * Nz + j * Nz + k


def face_index_Hz(i: int, j: int, k: int, Nx: int, Ny: int, Nz: int) -> int:
    """Flat index of Hz face at (i, j, k).

    Hz shape: (Nx, Ny, Nz+1).
    """
    return i * Ny * (Nz + 1) + j * (Nz + 1) + k


def apply_thin_pec_sheet(
    mesh,
    axis: str,
    position: float,
    rect: "tuple[float, float, float, float] | None" = None,
) -> None:
    """Set tangential E edges at a grid-node plane to PEC (in-place).

    Implements the FIT model for an infinitely thin PEC conductor: only ONE
    grid-node plane is required, with no extra cell-layer thickness.

    Args:
        mesh:     Mesh to modify (pec_mask_edges is updated in-place).
        axis:     Normal to the sheet: 'x', 'y', or 'z'.
        position: Physical coordinate of the sheet [m].
        rect:     (u_min, v_min, u_max, v_max) — tangential extent in the two
                  axes perpendicular to ``axis``.  ``None`` = entire domain.
                  For axis='y': u→x, v→z.
                  For axis='x': u→y, v→z.
                  For axis='z': u→x, v→y.
    """
    grid = mesh.grid
    Ny, Nz = grid.Ny, grid.Nz
    pec = mesh.pec_mask_edges

    x_c = 0.5 * (grid.x[:-1] + grid.x[1:])  # cell centres (Nx,)
    y_c = 0.5 * (grid.y[:-1] + grid.y[1:])  # (Ny,)
    z_c = 0.5 * (grid.z[:-1] + grid.z[1:])  # (Nz,)

    if axis == "y":
        j_h = int(np.argmin(np.abs(grid.y - position)))

        if rect is None:
            x_min, z_min, x_max, z_max = grid.x[0], grid.z[0], grid.x[-1], grid.z[-1]
        else:
            x_min, z_min, x_max, z_max = rect

        i_mask = np.where((x_c >= x_min) & (x_c <= x_max))[0]  # x-cells for Ex
        k_mask_nodes = np.where((grid.z >= z_min) & (grid.z <= z_max))[0]  # z-nodes for Ex
        i_mask_nodes = np.where((grid.x >= x_min) & (grid.x <= x_max))[0]  # x-nodes for Ez
        k_mask = np.where((z_c >= z_min) & (z_c <= z_max))[0]  # z-cells for Ez

        # Ex[i, j_h, k]: flat = i*(Ny+1)*(Nz+1) + j_h*(Nz+1) + k
        ii, kk = np.meshgrid(i_mask, k_mask_nodes, indexing="ij")
        ex_flat = (ii * (Ny + 1) * (Nz + 1) + j_h * (Nz + 1) + kk).ravel()
        pec[0, ex_flat] = True

        # Ez[i, j_h, k]: flat = i*(Ny+1)*Nz + j_h*Nz + k
        ii2, kk2 = np.meshgrid(i_mask_nodes, k_mask, indexing="ij")
        ez_flat = (ii2 * (Ny + 1) * Nz + j_h * Nz + kk2).ravel()
        pec[2, ez_flat] = True

    elif axis == "x":
        i_h = int(np.argmin(np.abs(grid.x - position)))

        if rect is None:
            u_min, v_min, u_max, v_max = grid.y[0], grid.z[0], grid.y[-1], grid.z[-1]
        else:
            u_min, v_min, u_max, v_max = rect

        j_mask = np.where((y_c >= u_min) & (y_c <= u_max))[0]  # y-cells for Ey
        k_mask_nodes = np.where((grid.z >= v_min) & (grid.z <= v_max))[0]  # z-nodes for Ey
        j_mask_nodes = np.where((grid.y >= u_min) & (grid.y <= u_max))[0]  # y-nodes for Ez
        k_mask = np.where((z_c >= v_min) & (z_c <= v_max))[0]  # z-cells for Ez

        # Ey[i_h, j, k]: flat = i_h*Ny*(Nz+1) + j*(Nz+1) + k
        jj, kk = np.meshgrid(j_mask, k_mask_nodes, indexing="ij")
        ey_flat = (i_h * Ny * (Nz + 1) + jj * (Nz + 1) + kk).ravel()
        pec[1, ey_flat] = True

        # Ez[i_h, j, k]: flat = i_h*(Ny+1)*Nz + j*Nz + k
        jj2, kk2 = np.meshgrid(j_mask_nodes, k_mask, indexing="ij")
        ez_flat = (i_h * (Ny + 1) * Nz + jj2 * Nz + kk2).ravel()
        pec[2, ez_flat] = True

    elif axis == "z":
        k_h = int(np.argmin(np.abs(grid.z - position)))

        if rect is None:
            u_min, v_min, u_max, v_max = grid.x[0], grid.y[0], grid.x[-1], grid.y[-1]
        else:
            u_min, v_min, u_max, v_max = rect

        i_mask = np.where((x_c >= u_min) & (x_c <= u_max))[0]  # x-cells for Ex
        j_mask_nodes = np.where((grid.y >= v_min) & (grid.y <= v_max))[0]  # y-nodes for Ex
        i_mask_nodes = np.where((grid.x >= u_min) & (grid.x <= u_max))[0]  # x-nodes for Ey
        j_mask = np.where((y_c >= v_min) & (y_c <= v_max))[0]  # y-cells for Ey

        # Ex[i, j, k_h]: flat = i*(Ny+1)*(Nz+1) + j*(Nz+1) + k_h
        ii, jj = np.meshgrid(i_mask, j_mask_nodes, indexing="ij")
        ex_flat = (ii * (Ny + 1) * (Nz + 1) + jj * (Nz + 1) + k_h).ravel()
        pec[0, ex_flat] = True

        # Ey[i, j, k_h]: flat = i*Ny*(Nz+1) + j*(Nz+1) + k_h
        ii2, jj2 = np.meshgrid(i_mask_nodes, j_mask, indexing="ij")
        ey_flat = (ii2 * Ny * (Nz + 1) + jj2 * (Nz + 1) + k_h).ravel()
        pec[1, ey_flat] = True

    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def build_pec_mask_faces(
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
) -> np.ndarray:
    """Build boolean PEC mask for E-field edges.

    An edge is marked PEC if *all* adjacent cells that share that edge are PEC.
    For domain-boundary edges (only one adjacent cell) the single cell decides.

    Returns array of shape ``(3, n_max_edges)`` where axis-0 indexes
    Ex-, Ey-, Ez-edge components and ``True`` means E_tangential = 0.

    Uses vectorised NumPy operations — no Python loops over cells.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    # Boolean cell-PEC mask  shape (Nx, Ny, Nz)
    pec_cells = np.zeros((Nx, Ny, Nz), dtype=bool)
    for mid, mat in material_library.items():
        if mat.is_pec:
            pec_cells |= material_id == mid

    # Pad with True on all faces so boundary edges (one neighbour) count as PEC
    # when the single neighbour is PEC.  We pad with False for non-PEC boundary.
    # Strategy: edge is PEC iff the logical-AND of all neighbouring cells is True.
    # For Ex[i,j,k]: neighbours are cells (i, j-1, k-1), (i, j, k-1),
    #                                      (i, j-1, k  ), (i, j, k  )
    # We handle boundary by clamping indices and using max-of-neighbours instead.
    # Simpler practical rule (sufficient for structured grid):
    #   Ex[i,j,k] is PEC if pec_cells[i, clamp(j-1), clamp(k-1)]
    #   OR pec_cells[i, clamp(j), clamp(k-1)]
    #   OR pec_cells[i, clamp(j-1), clamp(k)] OR pec_cells[i, clamp(j), clamp(k)] is True.
    # (union = edge is PEC when *any* adjacent cell is PEC — conservative, correct for walls)

    def _clamp(arr, n):
        """Clamp 0..n inclusive → valid cell index 0..n-1."""
        return np.clip(arr, 0, n - 1)

    # ----- Ex edges: shape (Nx, Ny+1, Nz+1) -----
    jj = np.arange(Ny + 1)
    kk = np.arange(Nz + 1)
    j0 = _clamp(jj - 1, Ny)  # shape (Ny+1,)
    j1 = _clamp(jj, Ny)
    k0 = _clamp(kk - 1, Nz)  # shape (Nz+1,)
    k1 = _clamp(kk, Nz)

    # pec_cells has shape (Nx, Ny, Nz); broadcast over i,j,k
    ex_pec = (
        pec_cells[:, j0[:, None], k0[None, :]]  # (Nx, Ny+1, Nz+1)
        | pec_cells[:, j1[:, None], k0[None, :]]
        | pec_cells[:, j0[:, None], k1[None, :]]
        | pec_cells[:, j1[:, None], k1[None, :]]
    )

    # ----- Ey edges: shape (Nx+1, Ny, Nz+1) -----
    ii = np.arange(Nx + 1)
    i0 = _clamp(ii - 1, Nx)
    i1 = _clamp(ii, Nx)

    # Ey[i,j,k] is PEC iff any x-z-adjacent cell is PEC.  j passes through
    # unchanged (no clamping in j).  We avoid the NumPy non-contiguous
    # advanced-indexing pitfall (axes 0 and 2 fancy, axis 1 slice) which would
    # silently produce shape (Nx+1,1,Nz+1,Ny) instead of (Nx+1,Ny,Nz+1).
    # Instead: two sequential single-axis fancy index steps, each correct.
    ey_pec = (
        pec_cells[i0][:, :, k0]  # shape (Nx+1, Ny, Nz+1)
        | pec_cells[i1][:, :, k0]
        | pec_cells[i0][:, :, k1]
        | pec_cells[i1][:, :, k1]
    )

    # ----- Ez edges: shape (Nx+1, Ny+1, Nz) -----
    ez_pec = (
        pec_cells[i0[:, None, None], j0[None, :, None], :]  # (Nx+1, Ny+1, Nz)
        | pec_cells[i1[:, None, None], j0[None, :, None], :]
        | pec_cells[i0[:, None, None], j1[None, :, None], :]
        | pec_cells[i1[:, None, None], j1[None, :, None], :]
    )

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_max = max(n_Ex, n_Ey, n_Ez)

    pec_mask = np.zeros((3, n_max), dtype=bool)
    pec_mask[0, :n_Ex] = ex_pec.ravel()
    pec_mask[1, :n_Ey] = ey_pec.ravel()
    pec_mask[2, :n_Ez] = ez_pec.ravel()
    return pec_mask
