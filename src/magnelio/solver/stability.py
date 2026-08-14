"""
Courant stability condition for FIT time-domain simulation.

The Courant–Friedrichs–Lewy (CFL) condition ensures numerical stability
of the leapfrog scheme:

    dt ≤ safety / (c₀ · √(1/dx_min² + 1/dy_min² + 1/dz_min²))

where c₀ = 299_792_458 m/s and safety < 1.

See spec.md for the Courant stability analysis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from magnelio.mesh.grid import GridLines

if TYPE_CHECKING:
    import numpy as np

    from magnelio.mesh.mesher import Mesh

from magnelio.constants import C0  # noqa: E402

SAFETY_FACTORS = {
    "draft": 0.90,
    "normal": 0.95,
    "high": 0.99,
}

# Free-area floor of the conformal sub-cell formulae, duplicated from
# ``_operators.material_matrices`` rather than imported: this module
# stays free of cross-module imports on the hot CFL path (same reason
# the ``_build_*_local`` helpers below are copies).  Both mass matrices
# fall back to their bulk-staircase entry below this fraction, so both
# effective-material helpers must report the bulk value there — a
# mismatch would compute dt for a mass matrix the solver never uses.
_FREE_AREA_FLOOR = 0.01


def courant_dt(
    grid: GridLines,
    accuracy: str = "normal",
    min_effective_eps: float | None = None,
    min_effective_mu: float | None = None,
) -> float:
    """Compute the maximum stable time step for the given grid.

    Parameters
    ----------
    grid : GridLines
        The mesh grid.
    accuracy : str
        One of ``'draft'`` (0.90), ``'normal'`` (0.95), ``'high'`` (0.99).
    min_effective_eps : float or None
        Minimum effective ``ε_r`` across all active (non-PEC) edges,
        as returned by :func:`compute_min_effective_eps`.  When < 1,
        reduces dt to account for increased local wave speed at
        conformal boundary edges.
    min_effective_mu : float or None
        Minimum effective ``μ_r`` across all H-faces (conformal
        sub-cell faces have ``μ_eff < 1`` because the geometric
        ``A_face_free / A_face`` reduction lowers ``M_μ`` while the
        primal-face area in the geometric factor stays nominal).
        When < 1, reduces dt by ``√(μ_eff_min)`` symmetrically with
        the ε factor.

    Returns
    -------
    float
        Time step dt in seconds.

    Raises
    ------
    ValueError
        If *accuracy* is not recognized.
    """
    if accuracy not in SAFETY_FACTORS:
        raise ValueError(f"accuracy must be one of {list(SAFETY_FACTORS)!r}; got {accuracy!r}")
    safety = SAFETY_FACTORS[accuracy]

    eps_factor = math.sqrt(max(min_effective_eps, 1e-6)) if min_effective_eps is not None else 1.0
    mu_factor = math.sqrt(max(min_effective_mu, 1e-6)) if min_effective_mu is not None else 1.0

    dt_max = (
        eps_factor
        * mu_factor
        / (C0 * math.sqrt(1 / grid.dx_min**2 + 1 / grid.dy_min**2 + 1 / grid.dz_min**2))
    )
    return safety * dt_max


def compute_min_effective_eps(mesh: "Mesh") -> float:
    """Compute the minimum effective eps_r across all active edges.

    Driven by ``mesh.edge_material``:
        cat 1 (dielectric boundary) → ε_eff = eps_avg
        cat 2 (curved-PEC sub-cell) → ε_eff = eps_avg · L_primal / L_free
                                      (= heutige eps_avg / f_L), unless
                                      the dual face keeps less than 1 %
                                      free area — ``build_M_eps``
                                      freezes those edges, so they stay
                                      at the 1.0 default here and out
                                      of the minimum
        cat 0 / 3 / no edge_material → ε_eff = 1.0 (vacuum default)

    Returns 1.0 if no boundary information is present.
    """
    # Design: DD-051 (edge_material categories).
    import numpy as np

    em = mesh.edge_material
    if em is None:
        return 1.0

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_total = n_Ex + n_Ey + n_Ez

    pec = mesh.pec_mask_edges
    pec_flat = np.concatenate(
        [
            pec[0, :n_Ex],
            pec[1, :n_Ey],
            pec[2, :n_Ez],
        ]
    )
    active = ~pec_flat
    if not active.any():
        return 1.0

    # Start with eps = 1.0 for all active edges (staircase / vacuum default)
    eff_eps = np.ones(n_total)

    cat1 = (em.category == 1) & active
    if cat1.any():
        eff_eps[cat1] = em.eps_avg[cat1]

    cat2 = (em.category == 2) & active
    if cat2.any():
        # ε_eff(curved-PEC) = eps_avg · A_dual / A_free (algebraisch identisch
        # zur historischen eps_avg / f_L, weil A_dual / A_free = L_primal / L_free
        # in der Approximation A_free = A_dual · f_L).
        #
        # *Mirror the 1 % free-area floor that build_M_eps applies*
        # (DD-149): edges with ``f_A < 1%`` are frozen there
        # (``M_eps = 0``, ``beta_E = 0``), and a frozen edge cannot go
        # unstable — it must not enter this minimum at all.  Leaving
        # it at the 1.0 default does that, since any live edge is at
        # or below 1.0.  Reading the cat-2 reduction on such an edge
        # instead reports ~1e-15, which ``courant_dt``'s 1e-6 guard
        # turns into a dt three decades below the geometric limit —
        # the E-side twin of the collapse the μ floor prevents in
        # :func:`compute_min_effective_mu`.  NaN ``f_A`` compares
        # False and is floored, matching build_M_eps.
        applied_cat2 = cat2 & (em.f_A > _FREE_AREA_FLOOR)
        if applied_cat2.any():
            L_primal = _build_L_primal_E_local(grid)
            ratio = L_primal[applied_cat2] / em.L_free[applied_cat2]
            eff_eps[applied_cat2] = em.eps_avg[applied_cat2] * ratio

    # Backstop below the floor: an edge carrying no electric energy at
    # all cannot set the stability limit.  ``build_M_eps`` hands such an
    # edge ``M_eps = 0`` and the solver freezes it exactly as it freezes
    # the H side's donated faces (DD-147); a frozen edge cannot go
    # unstable, so counting it here only pins the minimum at 0 — and
    # ``courant_dt``'s 1e-6 guard then holds dt four decades below the
    # geometric limit, which reads as a run that never advances rather
    # than as an error.  The cat-2 free-area floor above now catches the
    # curved-PEC edges this was written for (DD-149); what remains is a
    # cat-1 edge whose dual face is entirely conductor, and any future
    # path that reaches zero by another route.
    live = active & (eff_eps > 0.0)
    if not live.any():
        return 1.0
    return float(np.nanmin(eff_eps[live]))


def compute_min_effective_mu(mesh: "Mesh") -> float:
    """Compute the minimum effective μ_r across all H-faces.

    Driven by ``mesh.face_material``:
        cat 1 (dielectric boundary) → μ_eff = mu_avg
        cat 2 (curved-PEC sub-cell) → μ_eff = mu_avg · (A_face_free / A_face)
                                      (the geometric reduction; equal
                                       to 1.0 for non-magnetic vacuum/PEC
                                       only when ``A_face_free = A_face``)
        cat 0 / no face_material   → μ_eff = 1.0 (vacuum default)

    Returns 1.0 if no boundary information is present.
    """
    # Design: DD-051 Variante A (face_material categories).
    import numpy as np

    fm = mesh.face_material
    if fm is None:
        return 1.0

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_total = (Nx + 1) * Ny * Nz + Nx * (Ny + 1) * Nz + Nx * Ny * (Nz + 1)

    # Start with μ_eff = 1.0 everywhere (staircase / vacuum).
    eff_mu = np.ones(n_total)

    cat1 = fm.category == 1
    if cat1.any():
        eff_mu[cat1] = fm.mu_avg[cat1]

    cat2 = fm.category == 2
    if cat2.any():
        A_face_full = _build_A_face_H_local(grid)
        # μ_eff = μ̄ · A_face_free / A_face — i.e. the same factor by
        # which build_M_mu reduces the cat-2 mass-matrix entry
        # relative to its bulk-staircase value.  *Mirror the 1%
        # A_face_free floor that build_M_mu applies*: faces with
        # ``A_face_free / A_face < 1%`` keep the bulk-staircase
        # value (μ_eff = mu_avg, ≥ 1.0) and contribute to the min
        # only via the bulk side, not via the cat-2 reduction.
        # Without this floor the min_eff_mu collapses to ~1e-6
        # (the build_M_mu floor's complement), which would shrink
        # ``dt`` to the courant_dt internal lower bound and turn a
        # 20-second test into a multi-hour one.
        ratio = fm.A_face_free[cat2] / np.maximum(A_face_full[cat2], 1e-30)
        applied_cat2 = cat2.copy()
        applied_cat2[cat2] = ratio > _FREE_AREA_FLOOR
        # Faces below the floor: μ_eff stays at 1.0 (bulk default).
        # Faces above the floor: μ_eff = mu_avg · ratio.
        if applied_cat2.any():
            ratio_applied = fm.A_face_free[applied_cat2] / np.maximum(
                A_face_full[applied_cat2],
                1e-30,
            )
            eff_mu[applied_cat2] = fm.mu_avg[applied_cat2] * ratio_applied

    return float(np.nanmin(eff_mu))


def _build_A_face_H_local(grid) -> "np.ndarray":
    """Local copy of geometry._subcell._build_A_face_H."""
    import numpy as np

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    A_hx = np.broadcast_to(
        (dy[:, None] * dz[None, :])[None, :, :],
        (Nx + 1, Ny, Nz),
    ).ravel()
    A_hy = np.broadcast_to(
        (dx[:, None] * dz[None, :])[:, None, :],
        (Nx, Ny + 1, Nz),
    ).ravel()
    A_hz = np.broadcast_to(
        (dx[:, None] * dy[None, :])[:, :, None],
        (Nx, Ny, Nz + 1),
    ).ravel()
    return np.concatenate([A_hx, A_hy, A_hz])


def _build_L_primal_E_local(grid) -> "np.ndarray":
    """Local copy of operators.material_matrices._build_L_primal_E.

    Duplicated to keep solver.stability free of cross-module imports
    on the hot CFL path; identical implementation.
    """
    import numpy as np

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    L = np.empty(n_Ex + n_Ey + n_Ez, dtype=np.float64)
    L[:n_Ex] = np.broadcast_to(
        dx[:, None, None],
        (Nx, Ny + 1, Nz + 1),
    ).ravel()
    L[n_Ex : n_Ex + n_Ey] = np.broadcast_to(
        dy[None, :, None],
        (Nx + 1, Ny, Nz + 1),
    ).ravel()
    L[n_Ex + n_Ey :] = np.broadcast_to(
        dz[None, None, :],
        (Nx + 1, Ny + 1, Nz),
    ).ravel()
    return L


def spectral_dt(
    mesh: "Mesh",
    accuracy: str = "normal",
    m_eps: "np.ndarray | None" = None,
    m_mu: "np.ndarray | None" = None,
) -> float:
    """Sharp CFL limit from the spectral radius of the update operator.

    The exact leapfrog stability limit is

        dt_max = 2 / sqrt(lambda_max(M_eps^-1 C^T M_mu^-1 C))

    restricted to the DOFs the solver actually updates: PEC-masked
    edges and frozen ``M_eps = 0`` edges are removed, frozen H-faces
    enter through the exact ``1/M_mu = 0``.  This replaces the
    ``sqrt(eps_min * mu_min)`` worst-case product of
    :func:`courant_dt`, which assumes the worst edge, the worst face
    and the smallest cell coincide — on conformal meshes they never
    do, and the product under-estimates the stable step by more than
    an order of magnitude (the measured limit sits at 67-75 % of the
    geometric Courant value where the heuristic reports 2-4 %).

    ``lambda_max`` is measured with a matrix-free Lanczos iteration on
    the symmetrised operator ``D^-1/2 A D^-1/2``.  If Lanczos does not
    converge, the row-sum (Gershgorin) upper bound on the same
    operator is used instead — strictly safe, typically within ~20 %
    of the exact value.  The measured ``lambda_max`` is cached on the
    mesh, so repeated calls (report + run) pay the eigensolve once.

    Parameters
    ----------
    mesh : Mesh
        Fully populated mesh.
    accuracy : str
        Safety factor selector, same map as :func:`courant_dt`.
    m_eps, m_mu : np.ndarray, optional
        Prebuilt mass matrices (``build_M_eps`` / ``build_M_mu``).
        Built from the mesh when omitted.

    Returns
    -------
    float
        Time step dt in seconds.
    """
    if accuracy not in SAFETY_FACTORS:
        raise ValueError(f"accuracy must be one of {list(SAFETY_FACTORS)!r}; got {accuracy!r}")
    safety = SAFETY_FACTORS[accuracy]

    cached = getattr(mesh, "_spectral_lambda_max", None)
    if cached is not None:
        return safety * 2.0 / math.sqrt(cached)

    lam = _measure_lambda_max(mesh, m_eps, m_mu)
    if lam is None or not math.isfinite(lam) or lam <= 0.0:
        # No live update operator (e.g. fully PEC-masked domain) —
        # nothing can go unstable; hand back the geometric value so
        # downstream step estimators keep working.
        return courant_dt(mesh.grid, accuracy)

    mesh._spectral_lambda_max = lam
    return safety * 2.0 / math.sqrt(lam)


def _measure_lambda_max(mesh, m_eps=None, m_mu=None):
    """lambda_max of the live symmetrised curl-curl update operator.

    Returns ``None`` when the mesh has no live E-DOF.  Falls back to
    the row-sum upper bound if the Lanczos iteration fails.
    """
    # Lazy imports: this module stays import-light for the plain
    # courant_dt path; the operator modules are only needed here.
    import numpy as np
    import scipy.sparse.linalg as spla

    from magnelio._operators.curl import build_curl_matrix
    from magnelio._operators.material_matrices import build_M_eps, build_M_mu

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    if m_eps is None:
        m_eps = build_M_eps(mesh)
    if m_mu is None:
        m_mu = build_M_mu(mesh)
    mmu_inv = np.where(m_mu > 0, 1.0 / np.where(m_mu > 0, m_mu, 1.0), 0.0)

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    pec_flat = np.concatenate([pec[0, :n_Ex], pec[1, :n_Ey], pec[2, :n_Ez]])
    live = (~pec_flat) & (m_eps > 0)
    if not live.any():
        return None
    live_idx = np.where(live)[0]
    n_live = live_idx.size

    C = build_curl_matrix(grid)
    inv_sqrt_d = 1.0 / np.sqrt(m_eps[live_idx])
    n_E = C.shape[1]

    def matvec(x):
        e = np.zeros(n_E)
        e[live_idx] = np.asarray(x).ravel() * inv_sqrt_d
        h = mmu_inv * (C @ e)
        return (C.T @ h)[live_idx] * inv_sqrt_d

    # Row-sum (Gershgorin) upper bound via the absolute-value factors:
    # row sums of |D^-1/2 C^T Mmu^-1 C D^-1/2| are bounded by the same
    # product with |C|, and for a symmetric matrix the largest row sum
    # of absolute values bounds the spectral radius.
    C_abs = C.copy()
    C_abs.data = np.abs(C_abs.data)
    e = np.zeros(n_E)
    e[live_idx] = inv_sqrt_d
    lam_upper = float(((C_abs.T @ (mmu_inv * (C_abs @ e)))[live_idx] * inv_sqrt_d).max())
    if lam_upper <= 0.0:
        return None

    if n_live < 8:
        # Too small for ARPACK; the upper bound is exact enough here.
        return lam_upper

    op = spla.LinearOperator((n_live, n_live), matvec=matvec, dtype=np.float64)
    try:
        # Fixed generic start vector (the DD-142 lesson): ARPACK's
        # default random start leaves a run-to-run residual in the
        # converged value, and dt must be bit-identical across rebuilds
        # of the same mesh — the project store resumes bit-exactly.
        v0 = np.random.default_rng(0).standard_normal(n_live)
        lam = float(
            spla.eigsh(
                op,
                k=1,
                which="LA",
                return_eigenvectors=False,
                tol=1e-8,
                maxiter=10_000,
                v0=v0,
            )[0]
        )
    except spla.ArpackError:
        return lam_upper
    if not math.isfinite(lam) or lam <= 0.0:
        return lam_upper
    # Lanczos converges from below; the safety factor covers the
    # 1e-8-tolerance residual.  A value above the certified upper
    # bound can only be rounding — never exceed it.
    return min(lam, lam_upper * (1.0 + 1e-12))


def estimate_total_steps(f_max: float, dt: float, periods: float = 10.0) -> int:
    """Estimate total time steps needed to resolve f_max.

    Simulation runs for ``periods / f_max`` seconds.

    Args:
        f_max:   Maximum frequency of interest [Hz].
        dt:      Time step [s].
        periods: Number of periods to simulate (default 10).

    Returns:
        Total number of time steps (integer, always >= 1).
    """
    T_sim = periods / f_max
    return max(1, math.ceil(T_sim / dt))
