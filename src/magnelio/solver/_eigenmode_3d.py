"""
3D Eigenmode Solver.

Finds resonant modes of a 3D cavity by solving the discrete curl-curl
generalized eigenvalue problem:

    A · e = λ · B · e
    A = Cᵀ · Mμ⁻¹ · C   (curl-curl operator, shape n_E × n_E)
    B = M_eps             (diagonal ε-mass matrix, positive definite)
    λ = ω²               [(rad/s)²]

Boundary conditions
───────────────────
Each of the six domain faces can be set to ``"PEC"`` (default) or ``"PMC"``:

- **PEC face:** tangential E = 0 → those DOFs are removed from the system.
  With all-PEC walls the null space (gradient modes) is eliminated entirely.

- **PMC face:** tangential H = 0 → DOFs remain free (natural BC).
  PMC walls do *not* reduce the DOF count, so residual null-space modes may
  appear in the returned eigenvalues and are suppressed by the 1 MHz floor.

Solver backend
──────────────
Default: ARPACK shift-invert ``eigsh`` with SuperLU direct factorisation
of (A − σB).  Robust and accurate for problems up to ~100³ grid cells.

An experimental CHOLMOD Cholesky path (``solver="arpack-cholmod"``,
requires ``scikit-sparse``) uses tree-cotree gauging to eliminate the
gradient null space, making the cotree system positive definite.
**Limitation:** tree-cotree gauging introduces spurious low-frequency
modes; the shifted cotree matrix is only SPD when σ is below the
lowest spurious mode.  For non-cubic cavities with auto-estimated σ
this condition often fails.  Provide a small explicit ``sigma`` if
needed.  See DD-033.

An experimental AMG-preconditioned CG path (``solver="arpack-amg"``,
requires ``pyamg``) is available but **not recommended**: pyamg's scalar
smoothed-aggregation AMG does not achieve mesh-independent convergence
for the vector-valued curl-curl operator, making it 10–30× slower than
SuperLU at every tested size.  See DD-033 for details.

A LOBPCG path (``solver="lobpcg"``) uses the **folded spectrum** in
B-absorbed standard form: ``S_std = B⁻½ (A/σ−B) B⁻½``, then solves
``S_std² y = μ y`` (standard eigenproblem, no B in LOBPCG).  Modes
near σ have the smallest μ = (λ/σ−1)²; the null-space folds to μ≈1.
True eigenvalues are recovered via the Rayleigh quotient on the
original ``(A, B)`` pair.  No extra dependencies beyond SciPy.
Avoids the full LU factorisation that ARPACK + SuperLU needs, so
memory and time scale better for grids with 50k+ cells.

See design-decisions.md DD-007, DD-033.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigsh

from magnelio._operators.curl import build_curl_matrix, curl_e_stencil
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.constants import C0 as _C0  # noqa: E402
from magnelio.mesh.mesher import Mesh

_F_PHYSICAL_MIN = 1e6  # 1 MHz — modes below are null-space artefacts
_AMG_THRESHOLD = 50_000  # switch to AMG above this many free DOFs

# Auto-shift escalation step (in λ = ω²; a factor 2 in frequency).
# KB-011: the ε_r,max estimate assumes a *filled* cavity and is a
# lower bound — on a sparsely filled high-contrast cavity it lands so
# close to the curl-curl null space that ARPACK converges on gradient
# vectors and under-delivers physical modes.  The solve then retries
# with the shift raised by this factor, up to the ε_r = 1 estimate
# (the corresponding upper bound).
_SIGMA_RETRY_FACTOR = 4.0


def _has_pyamg() -> bool:
    """Check if pyamg is available."""
    try:
        import pyamg  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _has_cholmod() -> bool:
    """Check if scikit-sparse (CHOLMOD) is available."""
    try:
        from sksparse.cholmod import cholesky  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


# Canonical face names and their spatial axis / side
_FACE_AXIS = {
    "xmin": ("x", 0),
    "xmax": ("x", 1),
    "ymin": ("y", 0),
    "ymax": ("y", 1),
    "zmin": ("z", 0),
    "zmax": ("z", 1),
}


def _periodic_axes(bcs: dict[str, str]) -> list[str]:
    """Axes whose face pair is declared ``"Periodic"`` (validated pairwise)."""
    axes = []
    for axis in "xyz":
        lo = bcs.get(f"{axis}min", "PEC").upper() == "PERIODIC"
        hi = bcs.get(f"{axis}max", "PEC").upper() == "PERIODIC"
        if lo != hi:
            raise ValueError(
                f"Periodic boundaries come in pairs: {axis}min and {axis}max "
                f"must both be 'Periodic' (got {bcs.get(f'{axis}min', 'PEC')!r} "
                f"and {bcs.get(f'{axis}max', 'PEC')!r})."
            )
        if lo:
            axes.append(axis)
    for face, kind in bcs.items():
        if kind.upper() == "CPML":
            raise ValueError(
                f"Boundary {face!r} is CPML, which has no meaning for an "
                f"eigenmode problem; declare the face PEC, PMC or Periodic."
            )
    return axes


def _resolve_phase_advance(phase_advance_deg, periodic_axes: list[str]) -> dict[str, float]:
    """Normalise the user's phase advance into ``{axis: radians}``."""
    if not periodic_axes:
        if phase_advance_deg is not None:
            raise ValueError(
                "phase_advance_deg was given, but no face pair is declared "
                "'Periodic' — the phase advance belongs to a periodic axis."
            )
        return {}
    if phase_advance_deg is None:
        return dict.fromkeys(periodic_axes, 0.0)
    if isinstance(phase_advance_deg, dict):
        unknown = set(phase_advance_deg) - set(periodic_axes)
        if unknown:
            raise ValueError(
                f"phase_advance_deg names axes {sorted(unknown)!r} that are "
                f"not periodic (periodic axes: {periodic_axes!r})."
            )
        return {axis: np.deg2rad(float(phase_advance_deg.get(axis, 0.0))) for axis in periodic_axes}
    if len(periodic_axes) > 1:
        raise ValueError(
            f"{len(periodic_axes)} periodic axes ({periodic_axes!r}): pass "
            f"phase_advance_deg as a dict, one entry per axis."
        )
    return {periodic_axes[0]: np.deg2rad(float(phase_advance_deg))}


def _build_floquet_projector(grid, phases: dict[str, float]):
    """Bloch-periodic identification of the E-edge DOFs.

    Along a periodic axis the tangential edges in the far plane
    (index ``N``) are the edges in the near plane (index ``0``) times
    ``exp(-i*phi)``.  The returned sparse ``P`` of shape
    ``(n_E, n_kept)`` maps the kept (near-plane and interior) DOFs to
    the full set, so ``P^H A P`` and ``P^H B P`` are the reduced
    operators.  The material matrices book a *full* dual cell on every
    domain face (the mirror convention of the natural PMC wall), so the
    far plane must carry no metric of its own: its edge entries of
    ``M_eps`` and its face entries of ``1/M_mu`` are zeroed before the
    congruence, and the near plane's full dual cell stands for the
    identified pair.  ``P`` is real (entries ±1) when every phase is 0
    or π, complex otherwise.

    Returns ``(P, far_E, far_H)``: the projector, the mask of far-plane
    E-edges (rows of ``P`` that are images) and the mask of far-plane
    H-faces (the component along each periodic axis at index ``N``).
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    shapes = (
        (Nx, Ny + 1, Nz + 1),
        (Nx + 1, Ny, Nz + 1),
        (Nx + 1, Ny + 1, Nz),
    )
    extents = (Nx, Ny, Nz)
    n_E = sum(int(np.prod(shp)) for shp in shapes)
    is_real = all(abs(np.sin(phi)) < 1e-14 for phi in phases.values())
    factor = np.ones(n_E, dtype=float if is_real else complex)
    master = np.arange(n_E)
    offset = 0
    for comp, shp in enumerate(shapes):
        ii, jj, kk = np.meshgrid(*(np.arange(n) for n in shp), indexing="ij")
        idx = [ii.ravel(), jj.ravel(), kk.ravel()]
        fac = np.ones(ii.size, dtype=factor.dtype)
        for ax_name, phi in phases.items():
            ax = "xyz".index(ax_name)
            if ax == comp:
                continue  # the component along the axis has no far-plane copy
            far = idx[ax] == extents[ax]
            idx[ax] = np.where(far, 0, idx[ax])
            fac = np.where(
                far, fac * (np.cos(phi) - 1j * np.sin(phi) if not is_real else np.cos(phi)), fac
            )
        flat = (idx[0] * shp[1] + idx[1]) * shp[2] + idx[2]
        master[offset : offset + ii.size] = offset + flat
        factor[offset : offset + ii.size] = fac
        offset += ii.size
    kept = master == np.arange(n_E)
    kept_idx = np.where(kept)[0]
    column = np.full(n_E, -1)
    column[kept_idx] = np.arange(kept_idx.size)
    P = sp.csr_matrix(
        (factor, (np.arange(n_E), column[master])),
        shape=(n_E, kept_idx.size),
    )
    h_shapes = (
        (Nx + 1, Ny, Nz),
        (Nx, Ny + 1, Nz),
        (Nx, Ny, Nz + 1),
    )
    far_H = np.zeros(sum(int(np.prod(shp)) for shp in h_shapes), dtype=bool)
    offset = 0
    for comp, shp in enumerate(h_shapes):
        size = int(np.prod(shp))
        for ax_name in phases:
            ax = "xyz".index(ax_name)
            if ax == comp:
                block = np.zeros(shp, dtype=bool)
                index = [slice(None)] * 3
                index[ax] = extents[ax]
                block[tuple(index)] = True
                far_H[offset : offset + size] |= block.ravel()
        offset += size
    return P, ~kept, far_H


def _estimate_sigma(
    grid,
    bcs: dict[str, str],
    eps_r_max: float,
    phases: dict[str, float] | None = None,
) -> float | None:
    """Estimate ARPACK shift σ ≈ 0.75·λ₁ from grid and boundary conditions.

    The lowest physical eigenvalue λ₁ = ω₁² is approximated from the
    effective wavenumber k_min in each spatial direction:

    - PEC–PEC pair  →  half-wavelength resonance  →  k_min = π / L
    - PEC–PMC pair  →  quarter-wavelength          →  k_min = π / (2·L)
    - PMC–PMC pair  →  zero-variation (TEM-like)   →  k_min = 0
    - Periodic pair →  Bloch phase advance φ       →  k_min = φ / L

    The two smallest non-zero k_min² terms (corresponding to the lowest
    two-index mode, which uses the two largest spatial dimensions) are
    summed, divided by ε_r,max to account for dielectric loading, and
    multiplied by 0.75:

        σ = 0.75 · (π·c₀)² · (k_min,1² + k_min,2²) / ε_r,max

    Returns ``None`` when fewer than two non-zero k_min terms are found
    (e.g. fully PMC box); the caller must then require a user-supplied σ.
    """
    phases = phases or {}
    dims = {
        "x": float(grid.x[-1] - grid.x[0]),
        "y": float(grid.y[-1] - grid.y[0]),
        "z": float(grid.z[-1] - grid.z[0]),
    }

    k2_terms = []
    for axis in ("x", "y", "z"):
        L = dims[axis]
        bc0 = bcs.get(f"{axis}min", "PEC").upper()
        bc1 = bcs.get(f"{axis}max", "PEC").upper()
        both_pec = (bc0 == "PEC") and (bc1 == "PEC")
        both_pmc = (bc0 == "PMC") and (bc1 == "PMC")

        if bc0 == "PERIODIC":
            # Bloch axis: the lowest wavenumber is the phase advance per
            # period (zero for the 0-mode, which then adds no bound).
            phi = abs(phases.get(axis, 0.0))
            if phi > 0.0:
                k2_terms.append((phi / L) ** 2)
        elif both_pec:
            k2_terms.append((np.pi / L) ** 2)  # half-wave
        elif not both_pmc:  # one PEC + one PMC
            k2_terms.append((np.pi / (2.0 * L)) ** 2)  # quarter-wave
        # else: both PMC → k=0, skip (TEM-like, no lower bound)

    if len(k2_terms) < 2:
        return None  # cannot estimate — caller raises or falls back to user sigma

    k2_terms.sort()  # ascending — lowest mode uses two smallest k²
    lambda1_est = (_C0**2) * (k2_terms[0] + k2_terms[1]) / eps_r_max
    return 0.75 * lambda1_est


def _merge_physical_modes(
    vals: np.ndarray,
    vecs: np.ndarray,
    new_vals: np.ndarray,
    new_vecs: np.ndarray,
    b_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Union of physical eigenpairs across shift-invert attempts.

    Null-space artefacts (below ``_F_PHYSICAL_MIN``) are dropped from
    the candidates.  A candidate whose eigenvalue matches a kept mode
    within 1e-6 relative is tested against the span of that
    near-degenerate cluster in the B metric and only kept when its
    out-of-span residual exceeds 1/2 — so a degenerate partner found
    by a later attempt survives, while the same mode re-found under a
    different shift does not.  Eigenvectors are assumed B-normalised
    (ARPACK ``eigsh(..., M=B)`` convention).
    """
    lam = np.maximum(np.asarray(new_vals).real, 0.0)
    physical = np.sqrt(lam) / (2.0 * np.pi) >= _F_PHYSICAL_MIN
    new_vals = np.asarray(new_vals)[physical]
    new_vecs = np.asarray(new_vecs)[:, physical]
    if vals.size == 0:
        return new_vals, new_vecs

    keep_vals = list(vals)
    keep_vecs = [vecs[:, j] for j in range(vecs.shape[1])]
    for j in range(new_vals.size):
        cand = new_vecs[:, j]
        cluster = [
            k
            for k, v in enumerate(keep_vals)
            if abs(v - new_vals[j]) <= 1e-6 * max(abs(v), abs(new_vals[j]))
        ]
        if cluster:
            K = np.stack([keep_vecs[k] for k in cluster], axis=1)
            overlaps = K.conj().T @ (b_diag * cand)
            gram = K.conj().T @ (b_diag[:, None] * K)
            proj_sq = float((overlaps.conj() @ np.linalg.solve(gram, overlaps)).real)
            norm_sq = float((cand.conj() @ (b_diag * cand)).real)
            if norm_sq - proj_sq < 0.25 * norm_sq:
                continue
        keep_vals.append(float(new_vals[j]))
        keep_vecs.append(cand)
    return np.array(keep_vals), np.stack(keep_vecs, axis=1)


def _build_pec_dof_mask(grid, bcs: dict[str, str]) -> np.ndarray:
    """Return a boolean mask of tangential E-DOFs that are PEC-constrained.

    Only faces with ``bcs[face] == "PEC"`` (case-insensitive) contribute.
    PMC faces are left free (natural BC).

    DOF ordering: [Ex-edges, Ey-edges, Ez-edges] — same as build_curl_matrix.

    Tangential E-DOFs per face:
        xmin/xmax  →  Ey (k free) and Ez (j free) at i=0 / i=Nx
        ymin/ymax  →  Ex (k free) and Ez (i free) at j=0 / j=Ny
        zmin/zmax  →  Ex (j free) and Ey (i free) at k=0 / k=Nz
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_E = n_Ex + n_Ey + n_Ez

    mask = np.zeros(n_E, dtype=bool)

    def is_pec(face: str) -> bool:
        return bcs.get(face, "PEC").upper() == "PEC"

    # ── Ex DOFs (shape Nx, Ny+1, Nz+1) ──────────────────────────────────────
    # Tangential on y-faces (j=0 ymin, j=Ny ymax) and z-faces (k=0 zmin, k=Nz zmax)
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    flat_Ex = (ii * (Ny + 1) * (Nz + 1) + jj * (Nz + 1) + kk).ravel()
    bnd_Ex = (
        (is_pec("ymin") & (jj == 0))
        | (is_pec("ymax") & (jj == Ny))
        | (is_pec("zmin") & (kk == 0))
        | (is_pec("zmax") & (kk == Nz))
    ).ravel()
    mask[flat_Ex[bnd_Ex]] = True

    # ── Ey DOFs (shape Nx+1, Ny, Nz+1) ──────────────────────────────────────
    off_Ey = n_Ex
    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), np.arange(Nz + 1), indexing="ij")
    flat_Ey = off_Ey + (ii * Ny * (Nz + 1) + jj * (Nz + 1) + kk).ravel()
    bnd_Ey = (
        (is_pec("xmin") & (ii == 0))
        | (is_pec("xmax") & (ii == Nx))
        | (is_pec("zmin") & (kk == 0))
        | (is_pec("zmax") & (kk == Nz))
    ).ravel()
    mask[flat_Ey[bnd_Ey]] = True

    # ── Ez DOFs (shape Nx+1, Ny+1, Nz) ──────────────────────────────────────
    off_Ez = n_Ex + n_Ey
    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz), indexing="ij")
    flat_Ez = off_Ez + (ii * (Ny + 1) * Nz + jj * Nz + kk).ravel()
    bnd_Ez = (
        (is_pec("xmin") & (ii == 0))
        | (is_pec("xmax") & (ii == Nx))
        | (is_pec("ymin") & (jj == 0))
        | (is_pec("ymax") & (jj == Ny))
    ).ravel()
    mask[flat_Ez[bnd_Ez]] = True

    return mask


def _build_tree_edge_dofs(
    grid,
    bcs: dict[str, str],
    free_idx: np.ndarray,
) -> np.ndarray:
    """Return global edge-DOF indices forming a spanning tree of the super-node graph.

    Nodes connected by PEC edges are contracted into a single super-node
    (they share the same potential in the gradient null space).  The tree
    spans all super-nodes connected by free edges.  Its edge count equals
    dim(ker(A_f)), ensuring exact null-space elimination.

    After removing these DOFs the cotree subspace has a positive-definite
    curl-curl operator, enabling CHOLMOD Cholesky factorisation.
    """
    from collections import deque

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
    stride_i = (Ny + 1) * (Nz + 1)
    stride_j = Nz + 1

    free_set = set(free_idx.tolist())

    # ── Union-Find for PEC-connected super-nodes ──────────────────────────
    parent = np.arange(n_nodes, dtype=np.intp)
    rank = np.zeros(n_nodes, dtype=np.intp)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Union nodes connected by PEC edges
    pec_mask = _build_pec_dof_mask(grid, bcs)

    # Ex PEC edges
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz + 1):
                dof = i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
                if pec_mask[dof]:
                    union(i * stride_i + j * stride_j + k, (i + 1) * stride_i + j * stride_j + k)

    # Ey PEC edges
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz + 1):
                dof = n_Ex + i * Ny * (Nz + 1) + j * (Nz + 1) + k
                if pec_mask[dof]:
                    union(i * stride_i + j * stride_j + k, i * stride_i + (j + 1) * stride_j + k)

    # Ez PEC edges
    off_Ez = n_Ex + n_Ey
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            for k in range(Nz):
                dof = off_Ez + i * (Ny + 1) * Nz + j * Nz + k
                if pec_mask[dof]:
                    union(i * stride_i + j * stride_j + k, i * stride_i + j * stride_j + (k + 1))

    # ── Build super-node adjacency via free edges ─────────────────────────
    # adj[super_node] = [(neighbour_super_node, free_edge_dof), ...]
    adj: dict[int, list[tuple[int, int]]] = {}

    def _add_free_edge(n0: int, n1: int, dof: int) -> None:
        s0, s1 = find(n0), find(n1)
        if s0 == s1:
            return  # intra-component — skip
        adj.setdefault(s0, []).append((s1, dof))
        adj.setdefault(s1, []).append((s0, dof))

    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz + 1):
                dof = i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
                if dof in free_set:
                    _add_free_edge(
                        i * stride_i + j * stride_j + k, (i + 1) * stride_i + j * stride_j + k, dof
                    )

    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz + 1):
                dof = n_Ex + i * Ny * (Nz + 1) + j * (Nz + 1) + k
                if dof in free_set:
                    _add_free_edge(
                        i * stride_i + j * stride_j + k, i * stride_i + (j + 1) * stride_j + k, dof
                    )

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            for k in range(Nz):
                dof = off_Ez + i * (Ny + 1) * Nz + j * Nz + k
                if dof in free_set:
                    _add_free_edge(
                        i * stride_i + j * stride_j + k, i * stride_i + j * stride_j + (k + 1), dof
                    )

    # ── BFS spanning tree on super-node graph ─────────────────────────────
    visited_super: set[int] = set()
    tree_dofs: list[int] = []
    queue: deque[int] = deque()

    for snode in adj:
        if snode in visited_super:
            continue
        visited_super.add(snode)
        queue.append(snode)
        while queue:
            cur = queue.popleft()
            for nbr, dof in adj.get(cur, []):
                if nbr not in visited_super:
                    visited_super.add(nbr)
                    tree_dofs.append(dof)
                    queue.append(nbr)

    return np.array(tree_dofs, dtype=np.intp)


@dataclass
class EigenmodeSolver3D:
    """3D cavity eigenmode solver.

    Solves the FIT curl-curl generalized eigenvalue problem for resonant
    frequencies and E/H-field mode shapes of a closed cavity.

    Parameters
    ----------
    n_modes : int
        Number of physical resonant modes to return.
    boundary_conditions : dict[str, str]
        Dict mapping face names to ``"PEC"`` or ``"PMC"``.
        Valid keys: ``"xmin"``, ``"xmax"``, ``"ymin"``, ``"ymax"``,
        ``"zmin"``, ``"zmax"``.  Omitted faces default to ``"PEC"``.
    solver : str or None
        Factorisation backend for shift-invert.
        ``None``, ``"arpack"`` or ``"arpack-superlu"`` → SuperLU (default).
        ``"lobpcg"`` → LOBPCG with folded-spectrum transformation
        ``(A/σ − B)²``.  No external dependencies.  Recommended for
        large problems (50k+ cells).
        ``"arpack-cholmod"`` → CHOLMOD Cholesky with gradient
        regularisation (requires ``scikit-sparse``).  Typically 2–5×
        faster than SuperLU for large problems.
        ``"arpack-amg"`` → experimental AMG-CG (requires ``pyamg``,
        not recommended — see DD-033).
    sigma : float or None
        ARPACK shift σ [(rad/s)²].  ``None`` → auto-estimated from
        the cavity geometry and boundary conditions; when a solve
        returns fewer physical modes than requested (the partially
        filled high-contrast case, where the estimate lands near the
        curl-curl null space), the shift is escalated automatically
        and the physical eigenpairs of all attempts are merged.  An
        explicit value disables the escalation.  Either way a solve
        that still under-delivers emits a ``RuntimeWarning``.
    verbose : bool
        Print solver progress information.
    phase_advance_deg : float, dict or None
        Bloch phase advance [degrees] across each ``"Periodic"`` face
        pair: a number for a single periodic axis, ``{axis: degrees}``
        for several.  ``None`` means zero phase.  Phases other than 0
        and 180 degrees make the problem complex Hermitian, which only
        the SuperLU backend supports.
    """

    n_modes: int = 5
    boundary_conditions: dict[str, str] = field(default_factory=dict)
    solver: str | None = None
    sigma: float | None = None
    verbose: bool = False
    phase_advance_deg: float | dict[str, float] | None = None

    def solve(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for cavity eigenmodes.

        Parameters
        ----------
        mesh : Mesh
            Fully populated mesh.

        Returns
        -------
        freq_hz : np.ndarray
            Physical resonant frequencies [Hz], shape ``(n_modes,)``.
        E_modes : np.ndarray
            E-field eigenvectors, shape ``(n_E, n_modes)``.
        H_modes : np.ndarray
            H-field eigenvectors, shape ``(n_H, n_modes)``.
        """
        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        # DD-051 Scope: under the unified sub-cell pipeline the curved-
        # PEC formula is ``ε̄ · A_free / L_free`` (one stage, no ``/= f_L``
        # inflation).  The eigensolver may now consume the same M_eps
        # as the FIT-TD update; the historical apply_dm=False (DD-039)
        # is no longer needed.
        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        # M_mu = 0 marks enlarged-cell-donated faces (WP-R5); the
        # exact 1/M_mu = 0 removes them from the curl-curl operator.
        M_mu_inv = np.where(
            M_mu > 0,
            1.0 / np.where(M_mu > 0, M_mu, 1.0),
            0.0,
        )

        pec_mask = _build_pec_dof_mask(grid, self.boundary_conditions)

        # Merge with material-based PEC mask (internal PEC bodies).
        # mesh.pec_mask_edges has shape (3, n_max); flatten to match DOF order.
        mat_pec = mesh.pec_mask_edges
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        pec_mask[:n_Ex] |= mat_pec[0, :n_Ex]
        pec_mask[n_Ex : n_Ex + n_Ey] |= mat_pec[1, :n_Ey]
        pec_mask[n_Ex + n_Ey : n_Ex + n_Ey + n_Ez] |= mat_pec[2, :n_Ez]

        # Bloch-periodic identification (DD-182): far-plane edges are
        # near-plane edges times exp(-i*phi).  The reduced problem is
        # P^H A P, P^H B P on the kept DOFs; a kept edge is PEC when
        # any of its images is.
        periodic_axes = _periodic_axes(self.boundary_conditions)
        phases = _resolve_phase_advance(self.phase_advance_deg, periodic_axes)
        projector = None
        if periodic_axes:
            projector, far_E, far_H = _build_floquet_projector(grid, phases)
            pec_mask = (abs(projector).T @ pec_mask.astype(float)) > 0.0
            M_eps = np.where(far_E, 0.0, M_eps)
            M_mu_inv_operator = np.where(far_H, 0.0, M_mu_inv)
        else:
            M_mu_inv_operator = M_mu_inv
        is_complex = projector is not None and np.iscomplexobj(projector)

        free_idx = np.where(~pec_mask)[0]
        n_free = len(free_idx)

        # ── Backend dispatch ────────────────────────────────────────────────
        solver_key = (self.solver or "").lower()
        n_null_seen = 0  # null-space artefacts dropped before post-processing

        if is_complex and solver_key not in ("", "arpack", "arpack-superlu"):
            raise NotImplementedError(
                f"solver={self.solver!r} handles real symmetric problems only; "
                f"a Bloch phase advance other than 0 or 180 degrees makes the "
                f"eigenproblem complex Hermitian.  Use the default SuperLU backend."
            )
        if solver_key == "lobpcg":
            if projector is not None:
                raise NotImplementedError(
                    "solver='lobpcg' does not support periodic boundaries; "
                    "use the default SuperLU backend."
                )
            k_request = min(self.n_modes + 8, n_free - 2)
            eigenvalues, eigenvectors_free = self._solve_lobpcg(
                grid,
                M_mu_inv,
                M_eps,
                pec_mask,
                free_idx,
                n_free,
                k_request,
                mesh,
            )
        else:
            # Sparse matrix assembly (ARPACK backends only)
            C = build_curl_matrix(grid)
            M_mu_inv_diag = sp.diags(M_mu_inv_operator, 0, format="csr")
            A = C.T @ M_mu_inv_diag @ C

            M_eps_safe = np.where(M_eps > 0, M_eps, np.finfo(float).tiny)
            B = sp.diags(M_eps_safe, 0, format="csr")

            if projector is not None:
                P_H = projector.conj().T.tocsr()
                A = (P_H @ A @ projector).tocsr()
                B = (P_H @ B @ projector).tocsr()

            A_f = A[np.ix_(free_idx, free_idx)].tocsr()
            B_f = B[np.ix_(free_idx, free_idx)].tocsr()
            b_diag = B_f.diagonal()

            # KB-011: a shift near the curl-curl null space makes
            # ARPACK converge on gradient vectors and under-deliver
            # physical modes — silently, on a sparsely filled
            # high-contrast cavity.  Escalate the auto shift through
            # the ladder (filled-cavity lower bound → empty-cavity
            # upper bound), growing the subspace each attempt, and
            # keep the union of physical eigenpairs so a raised shift
            # never loses an already-found low mode.
            eigenvalues = np.empty(0)
            eigenvectors_free = np.empty((n_free, 0))
            ladder = self._sigma_ladder(grid, mesh)
            for attempt, sigma in enumerate(ladder):
                k_request = min(self.n_modes + 4 + 6 * attempt, n_free - 2)
                if solver_key == "arpack-cholmod":
                    vals, vecs = self._solve_arpack_cholmod(
                        A_f,
                        B_f,
                        sigma,
                        n_free,
                        k_request,
                        grid,
                        free_idx,
                    )
                elif solver_key == "arpack-amg":
                    vals, vecs = self._solve_arpack_amg(
                        A_f,
                        B_f,
                        sigma,
                        n_free,
                        k_request,
                    )
                else:
                    vals, vecs = self._solve_arpack(
                        A_f,
                        B_f,
                        sigma,
                        n_free,
                        k_request,
                    )
                eigenvalues, eigenvectors_free = _merge_physical_modes(
                    eigenvalues,
                    eigenvectors_free,
                    vals,
                    vecs,
                    b_diag,
                )
                freq_new = np.sqrt(np.maximum(np.asarray(vals).real, 0.0)) / (2.0 * np.pi)
                n_null_seen += int(np.count_nonzero(freq_new < _F_PHYSICAL_MIN))
                if eigenvalues.size >= self.n_modes:
                    break
                if self.verbose and attempt + 1 < len(ladder):
                    print(
                        f"  {eigenvalues.size}/{self.n_modes} physical "
                        f"modes at sigma={sigma:.3e}; retrying with a "
                        f"higher shift"
                    )

        # ── Post-processing ─────────────────────────────────────────────────
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors_free = eigenvectors_free[:, order]

        eigenvalues = np.maximum(eigenvalues.real, 0.0)
        freq_hz = np.sqrt(eigenvalues) / (2 * np.pi)

        physical = freq_hz >= _F_PHYSICAL_MIN
        n_null = n_null_seen + int(np.count_nonzero(~physical))
        freq_hz = freq_hz[physical][: self.n_modes]
        eigenvectors_free = eigenvectors_free[:, physical][:, : self.n_modes]

        # KB-011 defect 2: handing back fewer modes than requested
        # must never be silent.
        if freq_hz.size < self.n_modes:
            warnings.warn(
                f"Eigenmode solve returned {freq_hz.size} of "
                f"{self.n_modes} requested modes (discarded {n_null} "
                f"null-space artefact(s) below "
                f"{_F_PHYSICAL_MIN / 1e6:.0f} MHz).  The shift-invert "
                f"target sits too close to the curl-curl null space "
                f"for this material distribution; pass an explicit "
                f"shift near the expected fundamental, "
                f"sigma=(2*pi*f_estimate)**2.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Map E back to full n_E space (PEC DOFs remain zero)
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        n_E = n_Ex + n_Ey + n_Ez
        n_modes_out = eigenvectors_free.shape[1]

        if projector is not None:
            E_reduced = np.zeros((projector.shape[1], n_modes_out), dtype=eigenvectors_free.dtype)
            E_reduced[free_idx, :] = eigenvectors_free
            E_modes = np.asarray(projector @ E_reduced)
        else:
            E_modes = np.zeros((n_E, n_modes_out), dtype=eigenvectors_free.dtype)
            E_modes[free_idx, :] = eigenvectors_free

        # ── Compute H-field for each mode: h = (1/ω) M_mu_inv C e ──────────
        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz
        n_Hz = Nx * Ny * (Nz + 1)
        n_H = n_Hx + n_Hy + n_Hz

        mu_inv_x = M_mu_inv[:n_Hx].reshape(Nx + 1, Ny, Nz)
        mu_inv_y = M_mu_inv[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
        mu_inv_z = M_mu_inv[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)

        Hx_buf = np.empty((Nx + 1, Ny, Nz), dtype=E_modes.dtype)
        Hy_buf = np.empty((Nx, Ny + 1, Nz), dtype=E_modes.dtype)
        Hz_buf = np.empty((Nx, Ny, Nz + 1), dtype=E_modes.dtype)

        H_modes = np.zeros((n_H, n_modes_out), dtype=E_modes.dtype)
        for m in range(n_modes_out):
            omega = 2 * np.pi * freq_hz[m]
            e = E_modes[:, m]
            Ex = e[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
            Ey = e[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
            Ez = e[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)

            curl_e_stencil(Ex, Ey, Ez, Hx_buf, Hy_buf, Hz_buf)
            Hx_buf *= mu_inv_x / omega
            Hy_buf *= mu_inv_y / omega
            Hz_buf *= mu_inv_z / omega

            H_modes[:, m] = np.concatenate(
                [
                    Hx_buf.ravel(),
                    Hy_buf.ravel(),
                    Hz_buf.ravel(),
                ]
            )

        return freq_hz, E_modes, H_modes

    # ── Sigma computation ───────────────────────────────────────────────────

    def _compute_sigma(self, grid, mesh) -> float:
        """Return the (initial) ARPACK shift σ [(rad/s)²]."""
        return self._sigma_ladder(grid, mesh)[0]

    def _sigma_ladder(self, grid, mesh) -> list[float]:
        """Shift-invert targets, ascending.

        An explicit user ``sigma`` is a single-entry ladder (no
        escalation — the shift is a user decision).  The auto path
        starts at the filled-cavity estimate (global ε_r,max): raising
        ε anywhere only lowers eigenvalues, so that estimate bounds
        the true fundamental from below and escalating can never skip
        it.  Steps of ``_SIGMA_RETRY_FACTOR`` walk up to the ε_r = 1
        estimate, the matching upper bound — the interval a partially
        filled cavity lives in (KB-011).
        """
        if self.sigma is not None:
            return [self.sigma]
        eps_r_max = max(max(m.epsilon) for m in mesh.material_library.values() if not m.is_pec)
        phases = _resolve_phase_advance(
            self.phase_advance_deg, _periodic_axes(self.boundary_conditions)
        )
        sigma = _estimate_sigma(grid, self.boundary_conditions, eps_r_max, phases)
        if sigma is None:
            raise ValueError(
                "Cannot auto-estimate sigma for the given boundary "
                "conditions (need at least two faces with a PEC "
                "component).  Please provide sigma explicitly: "
                "sigma=0.75*(2*pi*f_est)**2"
            )
        ladder = [sigma]
        sigma_empty = sigma * eps_r_max
        while ladder[-1] * _SIGMA_RETRY_FACTOR < sigma_empty:
            ladder.append(ladder[-1] * _SIGMA_RETRY_FACTOR)
        if eps_r_max > 1.0 + 1e-12:
            ladder.append(sigma_empty)
        return ladder

    # ── ARPACK + SuperLU (small problems) ───────────────────────────────────

    def _solve_arpack(self, A_f, B_f, sigma, n_free, k_request):
        """Solve via ARPACK eigsh with SuperLU direct factorisation."""
        if self.verbose:
            print(f"ARPACK (SuperLU): n_free={n_free:,d}, k={k_request}, sigma={sigma:.3e}")

        eigenvalues, eigenvectors_free = eigsh(
            A_f,
            M=B_f,
            k=k_request,
            which="LM",
            sigma=sigma,
        )
        return eigenvalues, eigenvectors_free

    # ── LOBPCG + folded spectrum (large problems) ────────────────────────

    def _solve_lobpcg(self, grid, M_mu_inv, M_eps, pec_mask, free_idx, n_free, k_request, mesh):
        """Solve via LOBPCG on the folded spectrum.

        The curl-curl operator has a large gradient null space (λ=0).
        LOBPCG with ``largest=False`` would find these null-space modes
        first.  The **folded spectrum** ``F_std = S_std²`` transforms
        the problem so modes near the shift σ have the smallest μ:

            S_std = B⁻½ (A/σ − B) B⁻½   (symmetric)
            F_std y = μ y                  (standard eigenproblem)
            μ = (λ/σ − 1)²

        The B⁻½ transformation absorbs the mass matrix into the operator
        and avoids B-orthogonalisation issues from small M_ε entries.
        Original eigenvectors are recovered via ``x = B⁻½ y`` and true
        eigenvalues via the Rayleigh quotient ``xᵀ A x / (xᵀ B x)``.

        No full LU factorisation is needed — only sparse matvecs with A.
        """
        from scipy.sparse.linalg import lobpcg as scipy_lobpcg  # noqa: PLC0415

        # ── Sparse matrix assembly ─────────────────────────────────────
        C = build_curl_matrix(grid)
        M_mu_inv_diag = sp.diags(M_mu_inv, 0, format="csr")
        A = C.T @ M_mu_inv_diag @ C

        M_eps_safe = np.where(M_eps > 0, M_eps, np.finfo(float).tiny)
        A_f = A[np.ix_(free_idx, free_idx)].tocsr()
        B_diag = M_eps_safe[free_idx]
        B_f = sp.diags(B_diag, format="csr")

        # ── Folded spectrum in standard form ───────────────────────────
        # S = A/σ − B,  S_std = B^{-1/2} S B^{-1/2},  F_std = S_std²
        # σ = 3 × sigma_auto ≈ 2.25 λ₁ — captures modes up to ~4.5 λ₁.
        sigma = 3.0 * self._compute_sigma(grid, mesh)
        S_f = (A_f / sigma - B_f).tocsr()
        B_inv_sqrt = 1.0 / np.sqrt(B_diag)

        def folded_std_matvec(y):
            y = np.asarray(y).ravel()
            # v = S_std @ y = B^{-1/2} S B^{-1/2} y
            v = S_f @ (B_inv_sqrt * y)
            v = B_inv_sqrt * v
            # w = S_std @ v = S_std² @ y
            w = S_f @ (B_inv_sqrt * v)
            w = B_inv_sqrt * w
            return w

        F_op = LinearOperator(
            (n_free, n_free),
            matvec=folded_std_matvec,
            dtype=float,
        )

        # ── LOBPCG solve (standard eigenproblem, no B) ─────────────────
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n_free, k_request))

        if self.verbose:
            print(
                f"LOBPCG (folded): n_free={n_free:,d}, k={k_request}, "
                f"sigma={sigma:.3e}, nnz(S)={S_f.nnz:,d}"
            )

        eigenvalues_folded, Y = scipy_lobpcg(
            F_op,
            X0,
            largest=False,
            tol=1e-8,
            maxiter=500,
            verbosityLevel=1 if self.verbose else 0,
        )

        # ── Recover original eigenvectors and true eigenvalues ─────────
        # x = B^{-1/2} y (back-transformation)
        # λ = xᵀ A x / (xᵀ B x)  (Rayleigh quotient)
        eigenvectors = np.empty((n_free, len(eigenvalues_folded)))
        eigenvalues = np.empty(len(eigenvalues_folded))
        for i in range(len(eigenvalues_folded)):
            x = B_inv_sqrt * Y[:, i]
            eigenvectors[:, i] = x
            eigenvalues[i] = float(x @ (A_f @ x)) / float(x @ (B_f @ x))

        return eigenvalues, eigenvectors

    # ── ARPACK + CHOLMOD Cholesky (large problems) ───────────────────────────

    def _solve_arpack_cholmod(self, A_f, B_f, sigma, n_free, k_request, grid, free_idx):
        """Solve via ARPACK eigsh with CHOLMOD Cholesky factorisation.

        The shifted matrix (A − σB) is indefinite because PEC boundary
        conditions do not fully eliminate the discrete gradient null space.
        Tree-cotree gauging removes the null space:

        1. Build a spanning tree of the node graph using free edges.
           Tree edges form a basis for the gradient (null-space) subspace.
        2. Remove tree-edge DOFs → cotree DOFs carry physical modes only.
        3. (A_cotree − σ·B_cotree) is SPD → CHOLMOD Cholesky factorisation.
        4. ARPACK shift-invert on the reduced system.
        5. Map eigenvectors back (tree edges = 0).
        """
        from sksparse.cholmod import cholesky as cholmod_cholesky  # noqa: PLC0415

        tree_edge_dofs = _build_tree_edge_dofs(grid, self.boundary_conditions, free_idx)
        tree_set = set(tree_edge_dofs)
        cotree_free = np.array(
            [idx for idx in free_idx if idx not in tree_set],
            dtype=np.intp,
        )
        n_cotree = len(cotree_free)

        # Map cotree DOFs into the A_f/B_f index space
        free_to_local = {g: i for i, g in enumerate(free_idx)}
        cotree_local = np.array(
            [free_to_local[g] for g in cotree_free],
            dtype=np.intp,
        )

        A_ct = A_f[np.ix_(cotree_local, cotree_local)].tocsr()
        B_ct = B_f[np.ix_(cotree_local, cotree_local)].tocsr()

        S_ct = (A_ct - sigma * B_ct).tocsc()
        try:
            factor = cholmod_cholesky(S_ct)
        except Exception as exc:
            raise RuntimeError(
                f"CHOLMOD: (A_cotree − σ·B_cotree) is not SPD. "
                f"Tree-cotree gauging leaves spurious modes below σ = "
                f"{sigma:.3e}. Try providing a smaller sigma explicitly "
                f"(below the lowest cotree eigenvalue), or use the "
                f"default SuperLU backend."
            ) from exc

        if self.verbose:
            n_tree = len(tree_edge_dofs)
            print(
                f"ARPACK (CHOLMOD): n_free={n_free:,d}, "
                f"n_tree={n_tree:,d}, n_cotree={n_cotree:,d}, "
                f"k={k_request}, sigma={sigma:.3e}"
            )

        def opinv_matvec(b):
            return factor(b)

        OPinv = LinearOperator(
            (n_cotree, n_cotree),
            matvec=opinv_matvec,
            dtype=float,
        )

        k_ct = min(k_request, n_cotree - 2)
        eigenvalues, eigvecs_ct = eigsh(
            A_ct,
            M=B_ct,
            k=k_ct,
            which="LM",
            sigma=sigma,
            OPinv=OPinv,
        )

        # Map cotree eigenvectors back to full free-DOF space
        eigenvectors_free = np.zeros((n_free, eigvecs_ct.shape[1]))
        eigenvectors_free[cotree_local, :] = eigvecs_ct

        return eigenvalues, eigenvectors_free

    # ── ARPACK + AMG-CG (large problems) ────────────────────────────────────

    def _solve_arpack_amg(self, A_f, B_f, sigma, n_free, k_request):
        """Solve via ARPACK eigsh with AMG-preconditioned CG inner solve.

        Builds a smoothed-aggregation AMG hierarchy for the shifted matrix
        (A − σB) and uses it as preconditioner for CG.  The CG solution
        is supplied to ARPACK via the ``OPinv`` parameter.
        """
        import pyamg  # noqa: PLC0415

        S = (A_f - sigma * B_f).tocsr()

        ml = pyamg.smoothed_aggregation_solver(
            S,
            symmetry="symmetric",
            max_coarse=500,
        )
        M_prec = ml.aspreconditioner(cycle="V")

        if self.verbose:
            print(
                f"ARPACK (AMG-CG): n_free={n_free:,d}, k={k_request}, "
                f"sigma={sigma:.3e}, AMG levels={len(ml.levels)}"
            )

        def opinv_matvec(b):
            x, info = cg(S, b, M=M_prec, atol=0.0, rtol=1e-8, maxiter=500)
            if info != 0 and self.verbose:
                print(f"  AMG-CG inner: info={info}")
            return x

        OPinv = LinearOperator(
            (n_free, n_free),
            matvec=opinv_matvec,
            dtype=float,
        )

        eigenvalues, eigenvectors_free = eigsh(
            A_f,
            M=B_f,
            k=k_request,
            which="LM",
            sigma=sigma,
            OPinv=OPinv,
        )
        return eigenvalues, eigenvectors_free
