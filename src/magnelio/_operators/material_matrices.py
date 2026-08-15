"""
Diagonal mass matrices M_eps, M_mu, M_sigma for FIT.

These encode the geometric averaging of material properties over each Yee
cell, weighted by edge length and face area.

For the primary (E) grid:
    M_eps[e] = ε₀ · εr[e] · A_face[e] / dl[e]

For the dual (H) grid:
    M_mu[f] = μ₀ · μr[f] · A_primal[f] / dl_dual[f]

In the diagonal sparse representation, these are stored as 1D arrays
(diagonals). Sparse diagonal matrices are constructed via
``scipy.sparse.diags(diag, 0, format='csr')``.

See spec.md for update equation context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh

if TYPE_CHECKING:
    from magnelio.materials.material import Material
    from magnelio.mesh.faces import BoxFace

from magnelio.constants import EPS0, MU0  # noqa: E402

# Sub-cell floor shared by both mass matrices: a conformal boundary
# element that retains less than this fraction of its free area counts
# as fully inside the conductor, and the sub-cell formula is not
# applied to it.  Below the floor that formula drives the entry to
# zero, and the reciprocal in the time-domain update is then either an
# infinity or, once it reaches the CFL helpers, a time step orders of
# magnitude below the geometric limit.  Refining a mesh past 1 % free
# area is not physically meaningful — the element has lost essentially
# all of its non-conductor share.  What replaces the formula differs
# per side (bulk staircase for H, frozen for E — see the two call
# sites).  ``compute_min_effective_eps`` and
# ``compute_min_effective_mu`` mirror this value; keep the three in
# step.
_FREE_AREA_FLOOR = 0.01

# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------


def _build_avg_d(d: np.ndarray, N: int) -> np.ndarray:
    """Vectorised dual-cell widths for boundary indices 0 .. N.

    Returns an array of length N+1 where:
    - result[0]     = d[0]
    - result[i]     = 0.5*(d[i-1] + d[i])  for 1 <= i <= N-1
    - result[N]     = d[N-1]

    For N == 0: returns [1.0] (degenerate single-node axis).
    """
    size = N + 1
    if N == 0:
        return np.ones(1)
    result = np.empty(size)
    result[0] = d[0]
    if N > 1:
        result[1:N] = 0.5 * (d[: N - 1] + d[1:N])
    result[N] = d[N - 1]
    return result


def _build_property_table(
    material_library: dict[int, "Material"],
    prop: str,
    component: int,
    pec_value: float | None = None,
) -> np.ndarray:
    """Build a lookup table ``table[mat_id] -> property_value``.

    Parameters
    ----------
    material_library : dict[int, Material]
        Maps material IDs to Material objects.
    prop : str
        Attribute name on Material ('epsilon', 'mu', 'sigma').
    component : int
        Tensor component (0=x, 1=y, 2=z).
    pec_value : float or None
        If given, PEC materials return this value instead of their
        natural property.  Used for sigma (PEC → 0.0 in M_sigma).
    """
    max_id = max(material_library.keys())
    table = np.zeros(max_id + 1)
    for mid, mat in material_library.items():
        if pec_value is not None and mat.is_pec:
            table[mid] = pec_value
        else:
            table[mid] = getattr(mat, prop)[component]
    return table


def flatten_port_plane_pec_mask(
    pec_mask: np.ndarray,
    mesh: Mesh,
    face: "BoxFace",
) -> np.ndarray:
    """Overwrite the port-plane tangential PEC mask with the first-interior slab.

    The conformal Dey-Mittra edge-shortening mechanism marks edges as
    PEC when they lie close enough to a PEC contour.  At the bbox
    boundary it has only half the cell neighbourhood available, so it
    shortens *more* edges than in the interior — empirically 18
    additional Ex/Ey edges per port slab on the conformal coax (444 vs
    426 interior).  That extra PEC mask makes the mode-solver see a
    *thicker* effective conductor than the volume wave does, producing
    a mode-profile mismatch that reflects ~27 % of the modal voltage
    even after the M_ε flatten.

    For a z-translation-invariant geometry the physically correct PEC
    mask is the *interior* one: every wavefront in the volume sees that
    contour, so the mode-solver and the FIT-TD update at the port plane
    must too.

    Parameters
    ----------
    pec_mask : np.ndarray, shape (3, n_max)
        Per-component PEC mask in the canonical Mesh layout.
    mesh : Mesh
        Source mesh; provides the grid sizes for per-component reshapes.
    face : BoxFace
        Bbox face whose two tangential E-components are flattened.

    Returns
    -------
    np.ndarray
        Fresh array (input is not mutated).  The boundary-slab tangential
        PEC mask values for ``face`` are replaced by the first-interior
        slab values; everything else is untouched.
    """
    from magnelio.mesh.faces import BoxFace as _BoxFace

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex_full = Nx * (Ny + 1) * (Nz + 1)
    n_Ey_full = (Nx + 1) * Ny * (Nz + 1)
    n_Ez_full = (Nx + 1) * (Ny + 1) * Nz

    out = np.array(pec_mask, copy=True)

    Ex = out[0, :n_Ex_full].reshape(Nx, Ny + 1, Nz + 1)
    Ey = out[1, :n_Ey_full].reshape(Nx + 1, Ny, Nz + 1)
    Ez = out[2, :n_Ez_full].reshape(Nx + 1, Ny + 1, Nz)

    if face is _BoxFace.X_MIN:
        Ey[0, :, :] = Ey[1, :, :]
        Ez[0, :, :] = Ez[1, :, :]
    elif face is _BoxFace.X_MAX:
        Ey[Nx, :, :] = Ey[Nx - 1, :, :]
        Ez[Nx, :, :] = Ez[Nx - 1, :, :]
    elif face is _BoxFace.Y_MIN:
        Ex[:, 0, :] = Ex[:, 1, :]
        Ez[:, 0, :] = Ez[:, 1, :]
    elif face is _BoxFace.Y_MAX:
        Ex[:, Ny, :] = Ex[:, Ny - 1, :]
        Ez[:, Ny, :] = Ez[:, Ny - 1, :]
    elif face is _BoxFace.Z_MIN:
        Ex[:, :, 0] = Ex[:, :, 1]
        Ey[:, :, 0] = Ey[:, :, 1]
    elif face is _BoxFace.Z_MAX:
        Ex[:, :, Nz] = Ex[:, :, Nz - 1]
        Ey[:, :, Nz] = Ey[:, :, Nz - 1]
    else:
        raise ValueError(f"Unhandled BoxFace: {face!r}")
    return out


def flatten_port_plane_mu(
    m_mu: np.ndarray,
    mesh: Mesh,
    face: "BoxFace",
) -> np.ndarray:
    """Overwrite the port-plane *normal* M_mu with the first-interior slab.

    The exact counterpart of :func:`flatten_port_plane_mass` for the
    magnetic mass: the normal H-faces (e.g. Hz for a z-port) live ON
    the bbox boundary plane, so the conformal Krietenstein/DD-053
    machinery builds their M_mu from a *halved* cell-neighbourhood —
    measured 36 % deviation from the first interior slab on the
    conformal RG-58 coax while the interior slabs agree to 1e-15
    (z-invariant feed).  The normal-face M_mu enters the TE
    transversal curl-curl operator directly, so the 2D port mode was
    solved against a *different* transversal operator than the volume
    propagates — the measured -42 dB TE11 port floor of DD-066's open
    item, invisible to the transversal pair-product chain certificate.

    The *transversal* H-faces (Hx/Hy for a z-port) sit at cell
    centres, one half-cell inside; they are built from a full
    neighbourhood and measure identical to the interior (8e-15) — not
    touched here.

    For a z-translation-invariant feed the physically correct normal
    M_mu is the interior value: every wavefront in the volume sees
    it, so the mode solver and the FIT-TD update at the port plane
    must too.

    Parameters
    ----------
    m_mu : np.ndarray, shape (n_H,)
        Flat-H M_mu diagonal in the canonical [Hx|Hy|Hz] ordering.
    mesh : Mesh
        Source mesh; provides the grid sizes for the reshapes.
    face : BoxFace
        Bbox face whose normal H-component slab is flattened.

    Returns
    -------
    np.ndarray
        Fresh array (input is not mutated).
    """
    from magnelio.mesh.faces import BoxFace as _BoxFace

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)
    if m_mu.size != n_Hx + n_Hy + n_Hz:
        raise ValueError(
            f"m_mu size {m_mu.size} does not match "
            f"n_Hx+n_Hy+n_Hz = {n_Hx + n_Hy + n_Hz} for this mesh."
        )
    out = np.array(m_mu, copy=True)
    Hx = out[:n_Hx].reshape(Nx + 1, Ny, Nz)
    Hy = out[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
    Hz = out[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)
    if face is _BoxFace.X_MIN:
        Hx[0, :, :] = Hx[1, :, :]
    elif face is _BoxFace.X_MAX:
        Hx[Nx, :, :] = Hx[Nx - 1, :, :]
    elif face is _BoxFace.Y_MIN:
        Hy[:, 0, :] = Hy[:, 1, :]
    elif face is _BoxFace.Y_MAX:
        Hy[:, Ny, :] = Hy[:, Ny - 1, :]
    elif face is _BoxFace.Z_MIN:
        Hz[:, :, 0] = Hz[:, :, 1]
    elif face is _BoxFace.Z_MAX:
        Hz[:, :, Nz] = Hz[:, :, Nz - 1]
    else:
        raise ValueError(f"Unhandled BoxFace: {face!r}")
    return out


def flatten_port_plane_mass(
    m_eps: np.ndarray,
    mesh: Mesh,
    face: "BoxFace",
) -> np.ndarray:
    """Overwrite the port-plane tangential M_eps with the first-interior slab.

    The conformal-area-weighted M_ε at the bbox boundary is built from a
    *halved* cell-neighbourhood (no cells exist beyond the bbox), so its
    values differ from the corresponding interior edges even for a
    z-translation-invariant geometry.  Measured edge by edge on a coax
    port plane: 350 edges with up to 10× M_ε deviation from the first
    interior slab, a 4 % slab-mass sum defect — *the* longitudinal mass
    jump that drives the |S|² ≈ 2 pathology of the conformal coax.

    For a true port plane on a z-translation-invariant geometry the
    physically correct M_ε is the *interior* value: every wavefront
    that propagates into the volume sees that value, so the mode-solver
    and the FIT-TD update at the port plane must too.  This function
    overwrites the boundary slab in-line with the first interior slab.

    Parameters
    ----------
    m_eps : np.ndarray, shape (n_E,)
        Flat-E M_eps diagonal in the canonical [Ex|Ey|Ez] ordering.
    mesh : Mesh
        Source mesh; provides the grid sizes for the per-component
        reshapes.
    face : BoxFace
        Bbox face whose two tangential E-components are flattened.

    Returns
    -------
    np.ndarray
        Fresh array (input is not mutated).  The boundary-slab tangential
        components for ``face`` are replaced by the first-interior-slab
        values; everything else is untouched.
    """
    from magnelio.mesh.faces import BoxFace as _BoxFace

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    if m_eps.size != n_Ex + n_Ey + n_Ez:
        raise ValueError(
            f"m_eps size {m_eps.size} does not match "
            f"n_Ex+n_Ey+n_Ez = {n_Ex + n_Ey + n_Ez} for this mesh."
        )
    out = np.array(m_eps, copy=True)
    Ex = out[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
    Ey = out[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
    Ez = out[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)
    if face is _BoxFace.X_MIN:
        Ey[0, :, :] = Ey[1, :, :]
        Ez[0, :, :] = Ez[1, :, :]
    elif face is _BoxFace.X_MAX:
        Ey[Nx, :, :] = Ey[Nx - 1, :, :]
        Ez[Nx, :, :] = Ez[Nx - 1, :, :]
    elif face is _BoxFace.Y_MIN:
        Ex[:, 0, :] = Ex[:, 1, :]
        Ez[:, 0, :] = Ez[:, 1, :]
    elif face is _BoxFace.Y_MAX:
        Ex[:, Ny, :] = Ex[:, Ny - 1, :]
        Ez[:, Ny, :] = Ez[:, Ny - 1, :]
    elif face is _BoxFace.Z_MIN:
        Ex[:, :, 0] = Ex[:, :, 1]
        Ey[:, :, 0] = Ey[:, :, 1]
    elif face is _BoxFace.Z_MAX:
        Ex[:, :, Nz] = Ex[:, :, Nz - 1]
        Ey[:, :, Nz] = Ey[:, :, Nz - 1]
    else:
        raise ValueError(f"Unhandled BoxFace: {face!r}")
    return out


def build_M_eps(mesh: Mesh) -> np.ndarray:
    """Build the diagonal of the electric permittivity mass matrix.

    Returns a 1D array of length ``n_Ex + n_Ey + n_Ez`` (total E-edge count).
    Units: [F/m * m²/m] = [F] = [s/Ω].

    The diagonal is ordered as [Ex-edges, Ey-edges, Ez-edges].

    Driven by the per-edge classification in ``mesh.edge_material``
    (DD-051).  Four categories:

        cat 0 (interior bulk)      — ``EPS0 · ε_r · A_dual / L_primal``
        cat 1 (dielectric boundary)— ``EPS0 · ε̄ · A_dual / L_primal``
        cat 2 (curved-PEC sub-cell)— ``EPS0 · ε̄ · A_dual / L_free``
                                     (≡ ``EPS0 · ε̄_free · A_free / L_free``);
                                     edges whose dual face retains less
                                     than 1 % free area lie inside the
                                     conductor and are frozen instead
                                     (``M_eps = 0``)
        cat 3 (interior PEC)       — masked; staircase value retained

    Enlarged-cell short edges contribute ``EPS0 · borrowed / L_primal``
    onto their donor neighbour.

    Parameters
    ----------
    mesh : Mesh
        Fully populated mesh.  ``mesh.edge_material`` may be ``None``
        for bare ``Mesh.from_grid`` setups; only category 0 is used in
        that case.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    mat_id = mesh.material_id  # shape (Nx, Ny, Nz)

    # Precompute dual-cell widths for boundary indices
    dx_avg = _build_avg_d(dx, Nx)  # length Nx+1
    dy_avg = _build_avg_d(dy, Ny)  # length Ny+1
    dz_avg = _build_avg_d(dz, Nz)  # length Nz+1

    # --- Cat-0 staircase: per-edge ε_r from material_id lookup --------
    eps_x = _build_property_table(mesh.material_library, "epsilon", 0)
    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    eps_ex = eps_x[mat_id[:, j_clamp][:, :, k_clamp]]
    geom_ex = dy_avg[None, :, None] * dz_avg[None, None, :] / dx[:, None, None]
    M_ex = (EPS0 * eps_ex * geom_ex).ravel()

    eps_y = _build_property_table(mesh.material_library, "epsilon", 1)
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    eps_ey = eps_y[mat_id[i_clamp][:, :, k_clamp]]
    geom_ey = dx_avg[:, None, None] * dz_avg[None, None, :] / dy[None, :, None]
    M_ey = (EPS0 * eps_ey * geom_ey).ravel()

    eps_z = _build_property_table(mesh.material_library, "epsilon", 2)
    eps_ez = eps_z[mat_id[i_clamp][:, j_clamp]]
    geom_ez = dx_avg[:, None, None] * dy_avg[None, :, None] / dz[None, None, :]
    M_ez = (EPS0 * eps_ez * geom_ez).ravel()

    M_eps = np.concatenate([M_ex, M_ey, M_ez])

    em = mesh.edge_material
    if em is None:
        return M_eps

    # --- Cat-1 / Cat-2 overrides driven by EdgeMaterialData ----------
    geom_E = _build_geom_E(grid)
    L_primal = _build_L_primal_E(grid)

    cat1 = em.category == 1  # dielectric boundary
    if cat1.any():
        M_eps[cat1] = EPS0 * em.eps_avg[cat1] * geom_E[cat1]

    cat2 = em.category == 2  # curved-PEC sub-cell
    if cat2.any():
        # M_eps = EPS0 · ε̄ · A_dual / L_free
        #       = EPS0 · ε̄ · geom_E · L_primal / L_free
        #
        # Floor (DD-149), the E-side mirror of the ``A_face_free /
        # A_face`` floor in :func:`build_M_mu`: edges whose dual face
        # has lost all but ``_FREE_AREA_FLOOR`` of its non-PEC area
        # are effectively fully inside the conductor.  ``eps_avg`` is
        # area-weighted over the *whole* dual face, so it carries that
        # loss directly (``eps_avg = f_A · ε̄_free``) and the cat-2
        # formula drives ``M_eps → 0`` there — after which
        # ``compute_min_effective_eps`` reports a permittivity of
        # ~1e-15, ``courant_dt``'s 1e-6 guard clamps it, and dt lands
        # three decades below the geometric limit.  DD-147 caught only
        # the exact ``eps_avg == 0`` case; a rounding remainder walks
        # straight past an equality test, which is what a coax feed on
        # a 0.25 mm grid produced (3 edges at ``f_A ~ 3e-15`` next to
        # 10 at exactly zero).  Falling back to the bulk-staircase
        # value is conservative — larger M_eps is more inertia, and
        # the edge stores no field worth resolving either way.
        #
        # Where the two sides differ: ``build_M_mu`` hands a floored
        # face the bulk-staircase value, because a floored H-face is
        # Faraday-dead (its circulation edges sit inside the PEC mask,
        # so ``C e = 0`` and h stays 0 either way — measured neutral to
        # 1e-15).  An E-edge has no such guarantee, so the bulk value
        # would let a curl drive an edge that lies inside a conductor,
        # where E = 0.  Freezing it is both the physical answer and the
        # one the solver already implements: ``M_eps = 0`` routes the
        # edge through the ``alpha_E = 1 / beta_E = 0`` branch of
        # ``FITTimeDomainSolver`` (DD-147).
        #
        # ``f_A`` is NaN on unprocessed edges, and NaN comparisons are
        # False, so those freeze too — they carry no sub-cell data that
        # could say otherwise.
        safe = cat2 & (em.f_A > _FREE_AREA_FLOOR)
        if safe.any():
            M_eps[safe] = EPS0 * em.eps_avg[safe] * geom_E[safe] * L_primal[safe] / em.L_free[safe]
        floored = cat2 & ~safe
        if floored.any():
            M_eps[floored] = 0.0

    # Cat-3 (interior PEC) keeps its staircase value; the edge is masked
    # by mesh.pec_mask_edges so the value is never read by the solver.

    # --- Enlarged-cell donor borrowing -------------------------------
    has_donor = em.enlarged_cell_donor >= 0
    if has_donor.any():
        donor_idx = np.nonzero(has_donor)[0]
        donors = em.enlarged_cell_donor[donor_idx]
        # borrowed = ε̄ · A_free [permittivity · m²];
        # contribution to donor M_eps is EPS0 · borrowed / L_primal_donor.
        contrib = EPS0 * em.enlarged_cell_area[donor_idx] / L_primal[donors]
        np.add.at(M_eps, donors, contrib)

    return M_eps


def build_M_eps_vacuum(mesh: Mesh) -> np.ndarray:
    """Build the vacuum-only diagonal of the electric permittivity mass matrix.

    Same shape and layout as :func:`build_M_eps`, but with all materials
    overridden to ``ε_r ≡ 1`` (vacuum) regardless of the mesh's actual
    material assignment.

    Used by the QTEM modal-port factory dispatch
    (:class:`magnelio.ports._modal.PortSpecMultiConductor` with
    ``epsilon_r=None``): the dual-Laplace path needs both the actual
    ``M_ε`` (with the real ε distribution) and a vacuum reference to
    extract ``ε_eff = C' / C'_0``.

    The vacuum override is non-invasive — the mesh's ``material_id`` and
    ``material_library`` are not mutated.  Conformal and Dey-Mittra
    overlays are skipped (they encode geometric corrections that are
    only meaningful in the presence of a non-vacuum reference material).

    Returns
    -------
    np.ndarray, shape (n_E,)
        Diagonal of the vacuum ``M_ε`` in the canonical
        ``[Ex | Ey | Ez]`` ordering.
    """
    return EPS0 * _build_geom_E(mesh.grid)


def build_M_mu(mesh: Mesh) -> np.ndarray:
    """Build the diagonal of the magnetic permeability mass matrix.

    Returns a 1D array of length ``n_Hx + n_Hy + n_Hz`` (total H-face count).
    Units: [H] = [V·s/A].
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    mat_id = mesh.material_id

    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    # --- Hx faces: shape (Nx+1, Ny, Nz) ---
    # M_mu = MU0 * mu_r * A_primal / L_dual = MU0 * mu_r * dy[j]*dz[k] / dx_avg[i]
    mu_x = _build_property_table(mesh.material_library, "mu", 0)
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    mu_hx = mu_x[mat_id[i_clamp]]  # (Nx+1, Ny, Nz)
    geom_hx = dy[None, :, None] * dz[None, None, :] / dx_avg[:, None, None]
    M_hx = (MU0 * mu_hx * geom_hx).ravel()

    # --- Hy faces: shape (Nx, Ny+1, Nz) ---
    mu_y = _build_property_table(mesh.material_library, "mu", 1)
    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    mu_hy = mu_y[mat_id[:, j_clamp]]  # (Nx, Ny+1, Nz)
    geom_hy = dx[:, None, None] * dz[None, None, :] / dy_avg[None, :, None]
    M_hy = (MU0 * mu_hy * geom_hy).ravel()

    # --- Hz faces: shape (Nx, Ny, Nz+1) ---
    mu_z = _build_property_table(mesh.material_library, "mu", 2)
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    mu_hz = mu_z[mat_id[:, :, k_clamp]]  # (Nx, Ny, Nz+1)
    geom_hz = dx[:, None, None] * dy[None, :, None] / dz_avg[None, None, :]
    M_hz = (MU0 * mu_hz * geom_hz).ravel()

    M_mu = np.concatenate([M_hx, M_hy, M_hz])

    # Boundary-face overrides from FaceMaterialData (DD-051 Variante A)
    fm = mesh.face_material
    if fm is None:
        return M_mu

    geom_H = _build_geom_H(grid)
    A_face = _build_A_face_H(grid)
    L_dual = _build_L_dual_H(grid)

    cat1 = fm.category == 1  # dielectric boundary
    if cat1.any():
        M_mu[cat1] = MU0 * fm.mu_avg[cat1] * geom_H[cat1]

    cat2 = fm.category == 2  # curved-PEC sub-cell
    if cat2.any():
        # Two value sources share this branch: the Krietenstein
        # geometric reduction (below) on genuinely 3D contours, and
        # the DD-053 LC-consistent pair value that
        # ``couple_face_material_pairs`` encodes into ``A_face_free``
        # at meshing time wherever a unique locally translation-
        # invariant ladder direction exists.
        #
        # Krietenstein: M_μ = MU0 · μ̄ · A_face_free / L_dual_free
        # — the geometric A_face shrinkage that vacuum+PEC μ-averaging
        # alone cannot capture (μ_r = 1 in both materials renders μ̄
        # invariant on a hollow PEC contour).  Without this term the
        # Faraday inertia at boundary faces is over-estimated, the
        # eigenvalues land too low, and the round-WG TE11 cut-off
        # converges only as O(h) instead of the textbook O(h²).
        #
        # Floor: faces with ``A_face_free / A_face < 1%`` are
        # effectively fully in PEC; the cat-2 formula on them yields
        # ``M_μ → 0`` and ``1 / M_μ → ∞``, which would alias as NaN
        # in the FIT-TD update.  Mesh refinement past 1% A_face_free
        # is not physically meaningful (the H-face has lost almost
        # all its non-PEC area) — falling back to the bulk-staircase
        # value at this floor is conservative and keeps M_μ finite.
        # Stability under standard ``dt = courant_dt(...)`` is
        # preserved because the helper now reads the cat-2-reduced
        # ``μ_eff`` via :func:`compute_min_effective_mu` and shrinks
        # ``dt`` accordingly.
        #
        # *Resolved (WP-R5, DD-058)*: the enlarged-cell donor
        # mechanism (``assign_h_face_donors`` — mass transfer onto a
        # neighbour face along the dual-edge axis, symmetric to the
        # E-edge donor in build_M_eps) exists but is DORMANT: the
        # trigger benchmark
        # ``validation/iris_cavity_donor_trigger.py``
        # (iris-loaded pillbox, 70-72 % floor share, mode field at
        # the aperture rim) measured it neutral to machine precision
        # (< 1e-15 relative eigenfrequency shift) — floored faces are
        # Faraday-dead: their circulation edges sit inside the PEC
        # mask, so ``C e = 0`` there and h stays 0 under either
        # treatment.  The staircase fallback therefore cannot inject
        # error; it merely keeps 1/M_μ finite.
        A_face_full = A_face
        safe = cat2 & (fm.A_face_free > _FREE_AREA_FLOOR * A_face_full)
        if safe.any():
            M_mu[safe] = MU0 * fm.mu_avg[safe] * fm.A_face_free[safe] / fm.L_dual_free[safe]

    # --- Enlarged-cell donor borrowing (WP-R5, M_μ mirror of the ---
    # --- E-edge donor in build_M_eps) ------------------------------
    # Floored cat-2 faces with an assigned donor are frozen
    # (M_μ = 0 ⇒ the update helpers use the exact 1/M_μ = 0, so h
    # stays 0: the face stores no energy and drives no EMF) and their
    # residual inertia μ̄·A_face_free moves onto the donor face along
    # the shared flux tube.  Floored faces *without* a donor keep the
    # bulk-staircase fallback from above.
    donor_arr = fm.enlarged_cell_donor
    if donor_arr is not None:
        donated = np.nonzero(donor_arr >= 0)[0]
        if donated.size > 0:
            receivers = donor_arr[donated]
            M_mu[donated] = 0.0
            np.add.at(
                M_mu,
                receivers,
                MU0 * fm.enlarged_cell_area[donated] / L_dual[receivers],
            )

    return M_mu


def build_M_sigma(mesh: Mesh) -> np.ndarray:
    """Build the diagonal of the electric conductivity matrix.

    Returns a 1D array of the same length as M_eps, with units [S] = [A/V].
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    mat_id = mesh.material_id

    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    # PEC materials get sigma=0 (PEC is handled by masking, not sigma)
    sig_x = _build_property_table(mesh.material_library, "sigma", 0, pec_value=0.0)
    sig_y = _build_property_table(mesh.material_library, "sigma", 1, pec_value=0.0)
    sig_z = _build_property_table(mesh.material_library, "sigma", 2, pec_value=0.0)

    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))

    # --- Sigma_x: shape (Nx, Ny+1, Nz+1) ---
    sig_ex = sig_x[mat_id[:, j_clamp][:, :, k_clamp]]
    geom_ex = dy_avg[None, :, None] * dz_avg[None, None, :] / dx[:, None, None]
    M_sx = (sig_ex * geom_ex).ravel()

    # --- Sigma_y: shape (Nx+1, Ny, Nz+1) ---
    sig_ey = sig_y[mat_id[i_clamp][:, :, k_clamp]]
    geom_ey = dx_avg[:, None, None] * dz_avg[None, None, :] / dy[None, :, None]
    M_sy = (sig_ey * geom_ey).ravel()

    # --- Sigma_z: shape (Nx+1, Ny+1, Nz) ---
    sig_ez = sig_z[mat_id[i_clamp][:, j_clamp]]
    geom_ez = dx_avg[:, None, None] * dy_avg[None, :, None] / dz[None, None, :]
    M_sz = (sig_ez * geom_ez).ravel()

    M_sigma = np.concatenate([M_sx, M_sy, M_sz])

    em = mesh.edge_material
    if em is None:
        return M_sigma

    # Cat-1 / Cat-2 overrides driven by EdgeMaterialData (DD-051).
    # Same kategorische Form as M_eps; sigma_avg uses the identical
    # A_dual normalisation, so cat 2 picks up the L_primal/L_free
    # factor on the curved-PEC sub-cell branch.
    geom_E = _build_geom_E(grid)
    L_primal = _build_L_primal_E(grid)

    cat1 = em.category == 1
    if cat1.any():
        valid = cat1 & ~np.isnan(em.sigma_avg)
        if valid.any():
            M_sigma[valid] = em.sigma_avg[valid] * geom_E[valid]

    cat2 = em.category == 2
    if cat2.any():
        valid = cat2 & ~np.isnan(em.sigma_avg)
        if valid.any():
            M_sigma[valid] = (
                em.sigma_avg[valid] * geom_E[valid] * L_primal[valid] / em.L_free[valid]
            )

    # Enlarged-cell donor borrowing for sigma is omitted: at PEC/air
    # boundaries the staircase sigma is 0, so the borrowing contribution
    # is zero in practice (matches historical _apply_dey_mittra_sigma).

    return M_sigma


def build_M_sigma_m(mesh: Mesh) -> np.ndarray:
    """Build the diagonal of the magnetic loss matrix (σ*, DD-081).

    Returns a 1D array of the same length as M_mu, with units [Ω].
    Staircase sampling identical to the bulk part of build_M_mu (same
    one-sided clamped cell lookup per H-face, same geometric factor).
    Conformal cat-1/cat-2 overrides (WP-C4, DD-093) mirror
    ``build_M_mu``'s form on faces where the classifier recorded
    ``FaceMaterialData.sigma_m_avg`` (NaN — including the DD-053
    pair-promoted faces — keeps the staircase value); the former
    "recorded non-goal" is retired.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    mat_id = mesh.material_id

    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    # PEC (incl. lossy metal) gets sigma_m=0 — handled by masking/loss models
    sgm_x = _build_property_table(mesh.material_library, "sigma_m", 0, pec_value=0.0)
    sgm_y = _build_property_table(mesh.material_library, "sigma_m", 1, pec_value=0.0)
    sgm_z = _build_property_table(mesh.material_library, "sigma_m", 2, pec_value=0.0)

    # --- Hx faces: shape (Nx+1, Ny, Nz) ---
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    sgm_hx = sgm_x[mat_id[i_clamp]]
    geom_hx = dy[None, :, None] * dz[None, None, :] / dx_avg[:, None, None]
    M_sx = (sgm_hx * geom_hx).ravel()

    # --- Hy faces: shape (Nx, Ny+1, Nz) ---
    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    sgm_hy = sgm_y[mat_id[:, j_clamp]]
    geom_hy = dx[:, None, None] * dz[None, None, :] / dy_avg[None, :, None]
    M_sy = (sgm_hy * geom_hy).ravel()

    # --- Hz faces: shape (Nx, Ny, Nz+1) ---
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    sgm_hz = sgm_z[mat_id[:, :, k_clamp]]
    geom_hz = dx[:, None, None] * dy[None, :, None] / dz_avg[None, None, :]
    M_sz = (sgm_hz * geom_hz).ravel()

    M_sigma_m = np.concatenate([M_sx, M_sy, M_sz])

    fm = mesh.face_material
    if fm is None or getattr(fm, "sigma_m_avg", None) is None:
        return M_sigma_m

    # Conformal overrides (WP-C4): the σ* booking follows build_M_mu's
    # categorical form face for face — cat 1 on the full primal face,
    # cat 2 on the SAME A_face_free / L_dual_free the μ̄ side uses
    # (incl. its 1 % floor fallback), so the μ/σ* pair stays a
    # consistently booked lossy face.
    geom_H = _build_geom_H(grid)
    A_face = _build_A_face_H(grid)

    cat1 = (fm.category == 1) & ~np.isnan(fm.sigma_m_avg)
    if cat1.any():
        M_sigma_m[cat1] = fm.sigma_m_avg[cat1] * geom_H[cat1]

    cat2 = (fm.category == 2) & ~np.isnan(fm.sigma_m_avg)
    safe = cat2 & (fm.A_face_free > 0.01 * A_face)
    if safe.any():
        M_sigma_m[safe] = fm.sigma_m_avg[safe] * fm.A_face_free[safe] / fm.L_dual_free[safe]

    return M_sigma_m


def _staircase_eps_edges(mesh: Mesh) -> np.ndarray:
    """Staircase per-edge ε_r lookup, concatenated [Ex | Ey | Ez].

    Mirrors the category-0 material lookup inside :func:`build_M_eps`
    (per-component tensor entry at the clamped owning cell), without
    the geometric factor.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    mat_id = mesh.material_id
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    eps_x = _build_property_table(mesh.material_library, "epsilon", 0)
    eps_y = _build_property_table(mesh.material_library, "epsilon", 1)
    eps_z = _build_property_table(mesh.material_library, "epsilon", 2)
    return np.concatenate(
        [
            eps_x[mat_id[:, j_clamp][:, :, k_clamp]].ravel(),
            eps_y[mat_id[i_clamp][:, :, k_clamp]].ravel(),
            eps_z[mat_id[i_clamp][:, j_clamp]].ravel(),
        ]
    )


def _staircase_mu_faces(mesh: Mesh) -> np.ndarray:
    """Staircase per-face μ_r lookup, concatenated [Hx | Hy | Hz].

    Mirrors the category-0 material lookup inside :func:`build_M_mu`.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    mat_id = mesh.material_id
    i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    mu_x = _build_property_table(mesh.material_library, "mu", 0)
    mu_y = _build_property_table(mesh.material_library, "mu", 1)
    mu_z = _build_property_table(mesh.material_library, "mu", 2)
    return np.concatenate(
        [
            mu_x[mat_id[i_clamp]].ravel(),
            mu_y[mat_id[:, j_clamp]].ravel(),
            mu_z[mat_id[:, :, k_clamp]].ravel(),
        ]
    )


def couple_face_material_pairs(mesh: Mesh, rtol: float = 1e-6) -> None:
    """LC-consistent conformal M_μ coupling (DD-053).  Mutates the mesh.

    The conformal ``M_ε`` on a curved-PEC edge is the physically
    correct flux-tube capacitance ``C = ε·f_A·A_dual/(f_L·L)``.  On a
    transmission line the per-section inductance is then fixed by the
    ladder identity ``L·C = εμ·d·d̃`` *independently of the tube
    shape* — the exact discrete travelling wave (DD-052) exists exactly
    when the co-located pair identity ``M_ε·M_μ = ε_pair·μ̄·ε0μ0·d·d̃``
    holds along the propagation axis.  The Krietenstein ``A_face_free``
    reduction is an independent B-flux-exclusion rule: correct where no
    ladder structure exists (genuinely 3D contours), the wrong LC
    partner on a line.

    Rule per H face (validated in
    ``validation/coax_pair_consistent_mu_spike.py``):

    * The two axes spanning the face are candidate ladder directions;
      the co-located E partners are the edges of the *other* spanning
      component on the two bounding planes.
    * A ladder is **valid** when its partners are unmasked and their
      two targets agree (relative ``rtol``) — that is the local
      translation-invariance test; partners on a domain-boundary plane
      are dropped in favour of the interior one (their conformal data
      is built from a halved cell neighbourhood, cf.
      :func:`flatten_port_plane_mass`).
    * One valid ladder, or two valid ladders that agree → the face
      mass is *defined* through the pair identity,
      ``M_μ := ε0μ0·ε_pair·μ̄·d·d̃ / M_ε[partner]`` with
      ``ε_pair = eps_avg/f_A`` (the material average over the free
      part of the dual face).  Bulk pairs reproduce their bulk value
      exactly, so the override is a no-op away from conformal
      contours.  When both agree, the one whose own two partners
      disagree less supplies the target: agreement at ``rtol`` still
      admits a spread of up to ``rtol``, and the better-conditioned
      ladder is the better estimator of the same quantity.  Picking by
      axis order instead would make the result depend on how the user
      happened to orient the model.
    * Two valid ladders that disagree → genuinely 3D neighbourhood, no
      exact wave to preserve: the Krietenstein value stays.

    The result is written into ``mesh.face_material`` as an equivalent
    ``A_face_free`` (category 2, ``L_dual_free = L_dual``), so
    ``build_M_mu``, the 2D mode solvers' ``m_mu_flat`` and
    ``compute_min_effective_mu`` / ``courant_dt`` all see it
    consistently.  Faces whose encoded value would fall below the
    ``build_M_mu`` 1 % ``A_face_free`` floor are left untouched.
    """
    em = mesh.edge_material
    fm = mesh.face_material
    if em is None or fm is None:
        return

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz

    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    A_face = _build_A_face_H(grid)
    L_dual = _build_L_dual_H(grid)

    pec = mesh.pec_mask_edges
    free_flat = ~np.concatenate(
        [
            pec[0, :n_Ex],
            pec[1, :n_Ey],
            pec[2, :n_Ez],
        ]
    )

    # eps_pair per edge: staircase ε_r, overridden by the conformal
    # free-region average eps_avg / f_A on cat-1/2 edges.
    eps_pair = _staircase_eps_edges(mesh)
    conf = np.isin(em.category, (1, 2)) & ~np.isnan(em.f_A) & (em.f_A > 1e-9)
    eps_pair[conf] = em.eps_avg[conf] / em.f_A[conf]

    mu_face = _staircase_mu_faces(mesh)
    has_mu = ~np.isnan(fm.mu_avg)
    mu_face[has_mu] = fm.mu_avg[has_mu]

    def _ladder(face_shape, p_dim, e_off, e_shape, d_p, d_avg_p):
        """Per-face (target, valid, resid) for the ladder along p_dim.

        ``resid`` is the ladder's own relative partner disagreement,
        ``inf`` where invalid — the quantity the caller ranks two valid
        ladders by.

        Interior faces are valid when both bounding-plane partners are
        free and agree (the local translation-invariance test).  Faces
        of the two domain-boundary slabs have one partner on a bbox
        plane whose conformal data is built from a halved cell
        neighbourhood (cf. :func:`flatten_port_plane_mass`); their
        validity is inherited from the adjacent interior face's
        two-partner test, with the interior partner's target — a
        single boundary partner alone cannot certify invariance
        (measured failure mode: bbox tangent points of a round
        contour, DD-049 geometry).
        """
        idx_e = (e_off + np.arange(int(np.prod(e_shape)))).reshape(e_shape)
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[p_dim] = slice(0, -1)
        sl_hi[p_dim] = slice(1, None)
        e1 = idx_e[tuple(sl_lo)]
        e2 = idx_e[tuple(sl_hi)]

        n_p = face_shape[p_dim]
        shape_p = [1, 1, 1]
        shape_p[p_dim] = n_p
        d_face = d_p.reshape(shape_p)
        dt1 = d_avg_p[:-1].reshape(shape_p)
        dt2 = d_avg_p[1:].reshape(shape_p)

        # m_eps vanishes on masked edges, so t1/t2 are allowed to come
        # out inf or nan here — the isfinite tests are what rejects
        # them.  The agreement test belongs in the same errstate block:
        # two masked partners give inf - inf, and numpy reports that as
        # an invalid subtract before the validity mask is ever applied.
        # The comparison itself is already correct (nan <= x is False,
        # and v1/v2 are False there anyway); only the warning is noise.
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (EPS0 * MU0 * eps_pair[e1] * d_face * dt1) / m_eps[e1]
            t2 = (EPS0 * MU0 * eps_pair[e2] * d_face * dt2) / m_eps[e2]
            v1 = free_flat[e1] & np.isfinite(t1)
            v2 = free_flat[e2] & np.isfinite(t2)

            scale = np.maximum(np.abs(t1), np.abs(t2))
            agree = np.abs(t1 - t2) <= rtol * scale
            resid = np.where(scale > 0.0, np.abs(t1 - t2) / scale, np.inf)
        valid = v1 & v2 & agree
        target = np.where(valid, t1, np.nan)
        resid = np.where(valid, resid, np.inf)

        if n_p >= 3:

            def _sl(idx):
                s = [slice(None)] * 3
                s[p_dim] = idx
                return tuple(s)

            lo, lo_in = _sl(0), _sl(1)
            valid[lo] = valid[lo_in] & v2[lo]
            target[lo] = np.where(valid[lo], t2[lo], np.nan)
            # A boundary slab inherits its interior neighbour's target,
            # so it inherits that neighbour's conditioning too.
            resid[lo] = np.where(valid[lo], resid[lo_in], np.inf)
            hi, hi_in = _sl(n_p - 1), _sl(n_p - 2)
            valid[hi] = valid[hi_in] & v1[hi]
            target[hi] = np.where(valid[hi], t1[hi], np.nan)
            resid[hi] = np.where(valid[hi], resid[hi_in], np.inf)
        return target, valid, resid

    # (face_offset, face_shape, [(p_dim, e_off, e_shape, d_p, d_avg_p)]*2)
    configs = [
        # Hx faces span (y, z): ladders along z (Ey partners) and
        # y (Ez partners).
        (
            0,
            (Nx + 1, Ny, Nz),
            [
                (2, n_Ex, (Nx + 1, Ny, Nz + 1), dz, dz_avg),
                (1, n_Ex + n_Ey, (Nx + 1, Ny + 1, Nz), dy, dy_avg),
            ],
        ),
        # Hy faces span (x, z): ladders along z (Ex) and x (Ez).
        (
            n_Hx,
            (Nx, Ny + 1, Nz),
            [
                (2, 0, (Nx, Ny + 1, Nz + 1), dz, dz_avg),
                (0, n_Ex + n_Ey, (Nx + 1, Ny + 1, Nz), dx, dx_avg),
            ],
        ),
        # Hz faces span (x, y): ladders along y (Ex) and x (Ey).
        (
            n_Hx + n_Hy,
            (Nx, Ny, Nz + 1),
            [
                (1, 0, (Nx, Ny + 1, Nz + 1), dy, dy_avg),
                (0, n_Ex, (Nx + 1, Ny, Nz + 1), dx, dx_avg),
            ],
        ),
    ]

    for f_off, f_shape, ladders in configs:
        (p_a, off_a, shape_a, dp_a, dav_a) = ladders[0]
        (p_b, off_b, shape_b, dp_b, dav_b) = ladders[1]
        t_a, v_a, r_a = _ladder(f_shape, p_a, off_a, shape_a, dp_a, dav_a)
        t_b, v_b, r_b = _ladder(f_shape, p_b, off_b, shape_b, dp_b, dav_b)

        agree_ab = np.abs(t_a - t_b) <= rtol * np.maximum(
            np.abs(t_a),
            np.abs(t_b),
        )
        # Both valid and agreeing: the better-conditioned ladder wins.
        # Both valid and disagreeing: neither, as before — a genuinely
        # 3D neighbourhood keeps its Krietenstein value.
        prefer_a = r_a <= r_b
        use_a = v_a & (~v_b | (agree_ab & prefer_a))
        use_b = v_b & (~v_a | (agree_ab & ~prefer_a))
        target = np.where(use_a, t_a, np.where(use_b, t_b, np.nan)).ravel()

        flat = f_off + np.nonzero(~np.isnan(target))[0]
        tgt = target[~np.isnan(target)]

        # The LC pair identity is M_ε·M_μ = ε0μ0·ε_pair·μ̄·d·d̃ (see
        # the function docstring): the ladder target carries the ε
        # side; the face's μ̄ multiplies here.  Session-126 fix — the
        # factor was missing, halving M_μ on μ_r = 2 uniform-ladder
        # faces (every historical DD-053 fixture has μ_r = 1, where
        # the omission is exactly invisible; caught by the WP-C5
        # rotated μ-slab reference).
        tgt = tgt * mu_face[flat]

        # Skip no-ops (bulk pairs reproduce the current value exactly).
        changed = np.abs(tgt - m_mu[flat]) > 1e-12 * m_mu[flat]
        flat, tgt = flat[changed], tgt[changed]
        if flat.size == 0:
            continue

        mu_f = mu_face[flat]
        with np.errstate(divide="ignore", invalid="ignore"):
            a_ff = tgt * L_dual[flat] / (MU0 * mu_f)
        # Finite gate: degenerate faces (e.g. sliver cells from
        # near-coincident planes) can carry mu_avg = 0; never write a
        # non-finite equivalent area into the mesh data.
        ok = np.isfinite(a_ff) & (a_ff > 0.011 * A_face[flat])
        flat, mu_f, a_ff = flat[ok], mu_f[ok], a_ff[ok]

        fm.category[flat] = 2
        fm.mu_avg[flat] = mu_f
        fm.A_face_free[flat] = a_ff
        fm.L_dual_free[flat] = L_dual[flat]


def assign_h_face_donors(mesh: Mesh, floor_ratio: float = 0.01) -> None:
    """Assign enlarged-cell donors for floored cat-2 H-faces (WP-R5).

    The M_μ mirror of the E-edge enlarged-cell donor (DD-051): a cat-2
    face whose free area has collapsed below ``floor_ratio · A_face``
    is effectively fully inside PEC.  Its Krietenstein value
    ``M_μ = μ0·μ̄·A_face_free/L_dual → 0`` would blow up ``1/M_μ``;
    the historical fallback restored the *bulk staircase* value, i.e.
    over-estimated the magnetic inertia by up to ``1/floor_ratio`` —
    harmless on smooth-walled geometries (round WG: the floored faces
    carry no mode energy) but a measured accuracy cap on deep PEC
    inclusions whose mode field concentrates exactly there
    (``validation/iris_cavity_donor_trigger.py``).

    Donor rule, mirroring :func:`_enlarged_cell` on the E side: the
    residual magnetic inertia ``μ̄ · A_face_free`` is borrowed onto a
    neighbour face along the dual-edge axis (the face normal — the
    two faces share the flux tube), preferring the neighbour with the
    larger free-area ratio.  Invalid receivers: floored faces and
    staircase interior-PEC faces (their h never evolves — borrowed
    inertia would silently vanish, the same failure mode the E-side
    donor blocks via the PEC mask).  The donated face is then frozen
    by ``build_M_mu`` (``M_μ = 0`` ⇒ exact ``1/M_μ = 0`` in the
    update — the face stores no energy and drives no EMF), and the
    receiver's mass grows by ``μ0 · borrowed / L_dual[receiver]``.
    Faces with no valid neighbour keep the staircase fallback.

    Must run *after* :func:`couple_face_material_pairs` (the DD-053
    pair pass finalises ``A_face_free`` first).  Mutates
    ``mesh.face_material`` in place.

    **Dormant in production** (developer gate, DD-051): the trigger
    benchmark ``validation/iris_cavity_donor_trigger.py``
    measured the mechanism exactly neutral even at > 70 % floor share
    on a deep PEC inclusion, because floored faces are Faraday-dead
    (their circulation edges sit inside the PEC mask — ``C e = 0``
    there, so h stays 0 under either treatment).  The mesher does not
    call this pass; wire it in ``Mesh.from_geometry`` step 4b if a
    future geometry meets the DD-051 trigger gate (> 70 % floor share
    *and* > 1–2 % convergence deficit *and* mode energy on the
    floored faces).
    """
    fm = mesh.face_material
    if fm is None:
        return

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)
    n_total = n_Hx + n_Hy + n_Hz

    donor = np.full(n_total, -1, dtype=np.int64)
    borrowed = np.zeros(n_total, dtype=np.float64)
    fm.enlarged_cell_donor = donor
    fm.enlarged_cell_area = borrowed

    A_face = _build_A_face_H(grid)
    cat2 = fm.category == 2
    with np.errstate(invalid="ignore"):
        floored = cat2 & ~(fm.A_face_free > floor_ratio * A_face)
    if not floored.any():
        return

    # Free-area ratio per candidate receiver: bulk / dielectric faces
    # count as 1.0, cat-2 faces by their (possibly reduced) free area.
    ratio = np.ones(n_total)
    ratio[cat2] = np.nan_to_num(
        fm.A_face_free[cat2] / A_face[cat2],
        nan=0.0,
        posinf=0.0,
    )

    # Staircase interior-PEC faces (cat 0 with every adjacent cell in
    # PEC) are dead receivers.
    pec_cell = _build_pec_cell_mask(mesh)
    dead = np.empty(n_total, dtype=bool)
    i_lo = np.clip(np.arange(Nx + 1) - 1, 0, max(Nx - 1, 0))
    i_hi = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
    dead[:n_Hx] = (pec_cell[i_lo] & pec_cell[i_hi]).ravel()
    j_lo = np.clip(np.arange(Ny + 1) - 1, 0, max(Ny - 1, 0))
    j_hi = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
    dead[n_Hx : n_Hx + n_Hy] = (pec_cell[:, j_lo] & pec_cell[:, j_hi]).ravel()
    k_lo = np.clip(np.arange(Nz + 1) - 1, 0, max(Nz - 1, 0))
    k_hi = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))
    dead[n_Hx + n_Hy :] = (pec_cell[:, :, k_lo] & pec_cell[:, :, k_hi]).ravel()
    dead &= fm.category == 0

    valid = (ratio > floor_ratio) & ~dead

    # Per-axis neighbour offsets along the face normal (= dual edge).
    def _neighbours(flat: int) -> list[int]:
        if flat < n_Hx:
            local, stride, n_p = flat, Ny * Nz, Nx + 1
        elif flat < n_Hx + n_Hy:
            local, stride, n_p = flat - n_Hx, Nz, Ny + 1
            # Hy: index = i·(Ny+1)·Nz + j·Nz + k; normal dim j has
            # stride Nz within each i-block.
        else:
            local, stride, n_p = flat - n_Hx - n_Hy, 1, Nz + 1
        p = (local // stride) % n_p
        out = []
        if p > 0:
            out.append(flat - stride)
        if p < n_p - 1:
            out.append(flat + stride)
        return out

    for flat in np.nonzero(floored)[0]:
        best, best_ratio = -1, -1.0
        for nbr in _neighbours(int(flat)):
            if valid[nbr] and ratio[nbr] > best_ratio:
                best, best_ratio = nbr, ratio[nbr]
        if best < 0:
            continue
        mu_val = fm.mu_avg[flat]
        if not np.isfinite(mu_val) or mu_val <= 0.0:
            mu_val = 1.0
        a_free = fm.A_face_free[flat]
        if not np.isfinite(a_free) or a_free < 0.0:
            a_free = 0.0
        donor[flat] = best
        borrowed[flat] = mu_val * a_free


def _build_pec_cell_mask(mesh: Mesh) -> np.ndarray:
    """Boolean (Nx, Ny, Nz) mask of cells filled with PEC material."""
    lib = mesh.material_library
    table = np.zeros(max(lib.keys()) + 1, dtype=bool)
    for mid, mat in lib.items():
        table[mid] = mat.is_pec
    return table[mesh.material_id]


# ---------------------------------------------------------------------------
# Geometric factor arrays for conformal overlays
# ---------------------------------------------------------------------------


def _build_L_primal_E(grid: GridLines) -> np.ndarray:
    """Build the primal-edge length per E-edge (Ex|Ey|Ez concatenated)."""
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


def _build_geom_E(grid: GridLines) -> np.ndarray:
    """Build A_dual / L_primal for all E-edges (Ex, Ey, Ez concatenated).

    Used by conformal overlays: ``M[e] = const * property * geom_E[e]``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    # Ex: dy_avg[j] * dz_avg[k] / dx[i]  — shape (Nx, Ny+1, Nz+1)
    gx = (dy_avg[None, :, None] * dz_avg[None, None, :] / dx[:, None, None]).ravel()
    # Ey: dx_avg[i] * dz_avg[k] / dy[j]  — shape (Nx+1, Ny, Nz+1)
    gy = (dx_avg[:, None, None] * dz_avg[None, None, :] / dy[None, :, None]).ravel()
    # Ez: dx_avg[i] * dy_avg[j] / dz[k]  — shape (Nx+1, Ny+1, Nz)
    gz = (dx_avg[:, None, None] * dy_avg[None, :, None] / dz[None, None, :]).ravel()

    return np.concatenate([gx, gy, gz])


def _build_A_face_H(grid: GridLines) -> np.ndarray:
    """Per-H-face primal-face area, concatenated [Hx | Hy | Hz]."""
    from magnelio.geo._subcell import _build_A_face_H as _impl  # noqa: PLC0415

    return _impl(grid)


def _build_L_dual_H(grid: GridLines) -> np.ndarray:
    """Per-H-face dual-edge length, concatenated [Hx | Hy | Hz]."""
    from magnelio.geo._subcell import _build_L_dual_H as _impl  # noqa: PLC0415

    return _impl(grid)


def _build_geom_H(grid: GridLines) -> np.ndarray:
    """Build A_primal / L_dual for all H-faces (Hx, Hy, Hz concatenated).

    Used by conformal mu overlay: ``M_mu[f] = MU0 * mu_r * geom_H[f]``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    dx_avg = _build_avg_d(dx, Nx)
    dy_avg = _build_avg_d(dy, Ny)
    dz_avg = _build_avg_d(dz, Nz)

    # Hx: dy[j]*dz[k] / dx_avg[i]  — shape (Nx+1, Ny, Nz)
    gx = (dy[None, :, None] * dz[None, None, :] / dx_avg[:, None, None]).ravel()
    # Hy: dx[i]*dz[k] / dy_avg[j]  — shape (Nx, Ny+1, Nz)
    gy = (dx[:, None, None] * dz[None, None, :] / dy_avg[None, :, None]).ravel()
    # Hz: dx[i]*dy[j] / dz_avg[k]  — shape (Nx, Ny, Nz+1)
    gz = (dx[:, None, None] * dy[None, :, None] / dz_avg[None, None, :]).ravel()

    return np.concatenate([gx, gy, gz])
