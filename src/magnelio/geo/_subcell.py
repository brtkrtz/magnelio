"""
Per-edge sub-cell classifier for E-edges (DD-051).

Replaces the two-stage override pipeline (compute_conformal_eps +
_apply_dey_mittra) with a single-pass four-category classifier that
produces, per E-edge, the unified data tuple

    (category, eps_avg, sigma_avg, A_free, L_free, donor, borrowed)

together with the PEC mask.  The four categories are:

    0 interior bulk        — dual-face inside one homogeneous material
    1 dielectric boundary  — dual-face crosses a dielectric/dielectric
                              interface; no PEC involved
    2 curved-PEC sub-cell  — dual-face crosses a PEC contour; both
                              eps_avg, A_free, L_free populated
    3 interior PEC         — dual-face fully inside PEC; edge masked

The mass-matrix builder (operators/material_matrices.py) drives directly
off the category, so the historical ``apply_dm`` switch and the
auxiliary ``_apply_dey_mittra`` overlay disappear.

Reference
---------
Krietenstein, Schuhmann, Thoma, Weiland, "The Perfect Boundary
Approximation Technique Facing the Big Challenge of High Precision
Field Computation," Proc. LINAC'98, Chicago, 1998.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.mesh.grid import GridLines

# Threshold below which a dielectric-boundary edge is masked as PEC to
# prevent 1/eps blow-up.  Mirrors the historical _CONFORMAL_EPS_THRESHOLD
# from filling.py.
_EPS_AVG_FLOOR = 0.1

# Category encoding.  Stored as int8 in EdgeMaterialData.category.
CAT_INTERIOR_BULK = 0
CAT_DIELECTRIC_BOUNDARY = 1
CAT_CURVED_PEC = 2
CAT_INTERIOR_PEC = 3


@dataclass
class FaceMaterialData:
    """Per-H-face sub-cell triple (M_mu mirror of EdgeMaterialData).

    The Krietenstein curved-PEC correction needs a *geometric*
    A_face_free reduction for H-faces, not just a material-μ
    average — on a curved PEC contour where μ_r = 1 in both vacuum
    and PEC, the bare μ̄-mean is identically 1.0 and degrades to the
    staircase value.  The missing piece is exactly the geometric
    A_face shrinkage that the M_ε side already gets through its
    Cat-2 ``A_dual / L_free`` formula.

    Three categories, mirroring ``EdgeMaterialData`` minus the
    interior-PEC case (H-faces are not masked — B is finite in PEC):

        0 interior bulk        — primal face inside one homogeneous
                                  material; staircase value
        1 dielectric boundary  — face crosses a dielectric/dielectric
                                  interface; area-weighted μ̄ over the
                                  full A_face (no PEC overlap)
        2 curved-PEC sub-cell  — face crosses a PEC contour; the free
                                  (non-PEC) area of the primal face is
                                  reduced; M_μ uses A_face_free in
                                  place of A_face.  On faces with a
                                  unique translation-invariant ladder,
                                  the DD-053 coupling pass
                                  (``couple_face_material_pairs``)
                                  later overwrites A_face_free with
                                  the LC-consistent pair value

    Attributes
    ----------
    category : np.ndarray, dtype int8, shape ``(n_H_total,)``
        One of {0, 1, 2} per face.
    mu_avg : np.ndarray, dtype float64, shape ``(n_H_total,)``
        Area-weighted ``μ_r`` over the primal face.  NaN for cat 0.
    A_face_free : np.ndarray, dtype float64, shape ``(n_H_total,)``
        Free (non-PEC) area of the primal face.  Equal to ``A_face``
        for cat 1; strictly less for cat 2.  NaN for cat 0.
    L_dual_free : np.ndarray, dtype float64, shape ``(n_H_total,)``
        Length of the dual edge outside PEC.  Equal to ``L_dual``
        when the dual edge does not enter the PEC region (the typical
        case on round-WG geometries with PEC at the bbox mantel and
        cell-centres in vacuum).  NaN for cat 0.
    enlarged_cell_donor : np.ndarray, dtype int64, shape ``(n_H_total,)``
        H-face enlarged-cell donor (the M_μ mirror of the E-edge
        donor): flat H-face index of the neighbour face along the
        dual-edge axis that absorbs the residual magnetic inertia of
        a floored cat-2 face (``A_face_free / A_face ≤ 1 %``), or
        ``-1``.  Assigned by
        :func:`~magnelio._operators.material_matrices.assign_h_face_donors`
        (which must run after the DD-053 coupling pass).  ``None``
        unless that pass was invoked explicitly — it is dormant in
        production (measured neutral, see the WP-R5 trigger
        benchmark); the staircase floor fallback then applies.
    enlarged_cell_area : np.ndarray, dtype float64, shape ``(n_H_total,)``
        Borrowed quantity ``μ̄ · A_face_free`` [permeability · m²] to
        be added to the donor's M_μ (divided by the donor's dual-edge
        length).  Zero where no borrowing.
    A_face_pec : np.ndarray or None, dtype float64, shape ``(n_H_total,)``
        GEOMETRIC PEC-covered area of every primal face [m²], never
        NaN (DD-087): classifier candidates carry the exact OCC value
        ``A_face·pec_frac`` (including exact 0); faces on degenerate
        section planes and non-candidates use the staircase cell rule
        (full when at least one neighbour cell is PEC-classified).
        Written BEFORE the DD-053 coupling pass — the pass rewrites
        only ``A_face_free``, so this field stays geometric.  Consumed
        by :func:`magnelio.mesh._surfaces.enumerate_pec_surfaces` for
        conformal wall areas; ``None`` on stores written pre-DD-087
        (consumers fall back to the staircase path).
    A_face_pec_jump : np.ndarray or None, dtype float64, shape ``(n_H_total,)``
        SIGNED jump of the PEC-covered area across every primal face's
        own plane [m²], never NaN (DD-087).  ``|jump|`` is the area of
        wall lying IN that plane — a flat, grid-aligned wall portion —
        and the sign names the non-PEC side of the face, the cell that
        owns it.  Zero wherever the PEC coverage is continuous across
        the plane, which is what tells a genuine flat wall apart from a
        face merely shadowed by a curved wall in front of it.  Lets
        :func:`magnelio.mesh._surfaces.enumerate_pec_surfaces` book a
        corner cell's lid and mantle separately; without it the
        divergence cell vector adds them as vectors and books
        ``|a+b| < |a|+|b|``.
    """

    category: np.ndarray
    mu_avg: np.ndarray
    A_face_free: np.ndarray
    L_dual_free: np.ndarray
    enlarged_cell_donor: np.ndarray | None = None
    enlarged_cell_area: np.ndarray | None = None
    A_face_pec: np.ndarray | None = None
    A_face_pec_jump: np.ndarray | None = None
    # WP-C1 (DD-093): per-material post-priority area fractions of the
    # primal faces for the dispersive/σ*-carrying material ids
    # (``prop='mu'`` conventions — PEC participates with its share).
    # ``fraction_mids`` is the id row order of ``material_fractions``
    # (shape ``(n_mids, n_H_total)``).  Only classifier-processed
    # (boundary) faces carry computed values; NaN marks "not
    # processed" — bulk faces AND faces later re-categorised without
    # an OCC statement (the DD-053 pair pass promotes uniform-ladder
    # faces to cat 2) — where consumers fall back to the staircase
    # cell lookup.  A computed 0 is a genuine zero share.  ``None`` on
    # meshes without such materials and on stores written pre-DD-093.
    fraction_mids: np.ndarray | None = None
    material_fractions: np.ndarray | None = None
    # WP-C4 (DD-093): area-weighted magnetic loss σ* over the primal
    # face, same conventions and NaN-marking as ``mu_avg`` (PEC claims
    # area with σ* = 0).  ``None`` on meshes without σ* materials and
    # on stores written pre-DD-093 — ``build_M_sigma_m`` then stays on
    # its staircase value bit-identically.
    sigma_m_avg: np.ndarray | None = None

    @property
    def fractions_by_mid(self) -> dict[int, np.ndarray] | None:
        """Dict view ``{mid: (n_H_total,) fraction array}`` or ``None``."""
        if self.fraction_mids is None or self.material_fractions is None:
            return None
        return {int(mid): self.material_fractions[i] for i, mid in enumerate(self.fraction_mids)}


@dataclass
class EdgeMaterialData:
    """Per-E-edge sub-cell triple from a single classification pass.

    Replaces the parallel ``ConformalData`` + ``DeyMittraData`` structures.
    Carries the full sub-cell triple ``(eps_avg, A_free, L_free)`` plus
    enlarged-cell donor info and the PEC mask in one container.

    Attributes
    ----------
    category : np.ndarray, dtype int8, shape ``(n_E_total,)``
        One of {0, 1, 2, 3} per edge — see module docstring.
    eps_avg : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Area-weighted relative permittivity of the dual face,
        ``(Σ_{non-PEC} ε_i · A_i) / A_dual``.  NaN for categories 0 and 3
        (bulk and interior PEC use the staircase / are masked).  PEC
        material contributes zero (D = 0 inside PEC).
    sigma_avg : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Area-weighted conductivity, same convention as ``eps_avg``.
    A_free : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Free dual-face area (PEC overlap subtracted).  Equal to A_dual
        for category 1.  NaN for categories 0 and 3.
    L_free : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Length of primal edge outside PEC.  Equal to L_primal for
        category 1.  NaN for categories 0 and 3.
    f_A : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Conformal free-area fraction of the dual face (non-PEC area /
        A_dual), from the same OCC tessellation as ``eps_avg`` — so
        ``eps_avg / f_A`` is exactly the material average over the
        *free* part of the dual face (the DD-053 ``eps_pair``).  NaN
        for categories 0 and 3.
    pec_mask : np.ndarray, dtype bool, shape ``(3, n_max)``
        E-edge PEC mask in the canonical ``Mesh.pec_mask_edges`` layout.
        Set True for category 3 edges, for short curved-PEC edges
        (f_L < eta) that the enlarged-cell technique borrows out, and
        for tangential-surface edges re-masked by the DD-053
        consistency rule (both endpoints on the same conductor).
    enlarged_cell_donor : np.ndarray, dtype int64, shape ``(n_E_total,)``
        Flat E-edge index of the neighbour edge that absorbs the
        borrowed area, or ``-1`` if no borrowing.
    enlarged_cell_area : np.ndarray, dtype float64, shape ``(n_E_total,)``
        Borrowed quantity ``ε_avg · A_free`` [permittivity · m²] to be
        added to the donor's M_eps.  Zero where no borrowing.
    """

    category: np.ndarray
    eps_avg: np.ndarray
    sigma_avg: np.ndarray
    A_free: np.ndarray
    L_free: np.ndarray
    f_A: np.ndarray
    pec_mask: np.ndarray
    enlarged_cell_donor: np.ndarray
    enlarged_cell_area: np.ndarray
    # WP-C1 (DD-093): per-material post-priority area fractions of the
    # dual faces for the dispersive material ids (E-side conventions —
    # PEC claims area without contributing, uncovered remainder counts
    # toward the background id 0, exactly the ``eps_avg`` budget:
    # ``Σᵢ fᵢ·εᵢ`` over all participating ids reproduces ``eps_avg`` on
    # fully sectioned faces).  ``fraction_mids`` is the id row order of
    # ``material_fractions`` (shape ``(n_mids, n_E_total)``).  Only
    # classifier-processed (boundary) edges carry computed values; NaN
    # marks "not processed" (bulk edges — their owning material
    # follows from the staircase cell lookup); a computed 0 is a
    # genuine zero share.  ``None`` on meshes without dispersive
    # materials and on stores written pre-DD-093.
    fraction_mids: np.ndarray | None = None
    material_fractions: np.ndarray | None = None

    @property
    def fractions_by_mid(self) -> dict[int, np.ndarray] | None:
        """Dict view ``{mid: (n_E_total,) fraction array}`` or ``None``."""
        if self.fraction_mids is None or self.material_fractions is None:
            return None
        return {int(mid): self.material_fractions[i] for i, mid in enumerate(self.fraction_mids)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _avg_d_val(d: np.ndarray, idx: int, N: int) -> float:
    """Average of d[idx-1] and d[idx], clamped to valid range."""
    if N == 0:
        return 1.0
    if idx == 0:
        return float(d[0])
    if idx >= N:
        return float(d[N - 1])
    return 0.5 * (float(d[idx - 1]) + float(d[idx]))


def _build_geom_E(grid: GridLines) -> np.ndarray:
    """Build A_dual / L_primal for all E-edges (Ex|Ey|Ez concatenated)."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    def _avg(d, N):
        if N == 0:
            return np.ones(1)
        out = np.empty(N + 1)
        out[0] = d[0]
        if N > 1:
            out[1:N] = 0.5 * (d[: N - 1] + d[1:N])
        out[N] = d[N - 1]
        return out

    dx_avg = _avg(dx, Nx)
    dy_avg = _avg(dy, Ny)
    dz_avg = _avg(dz, Nz)

    gx = (dy_avg[None, :, None] * dz_avg[None, None, :] / dx[:, None, None]).ravel()
    gy = (dx_avg[:, None, None] * dz_avg[None, None, :] / dy[None, :, None]).ravel()
    gz = (dx_avg[:, None, None] * dy_avg[None, :, None] / dz[None, None, :]).ravel()
    return np.concatenate([gx, gy, gz])


def _pick_neighbour(
    candidates: list[int],
    f_L: np.ndarray,
    eta: float,
    blocked: np.ndarray,
) -> int:
    """Pick the neighbour edge with the highest f_L >= eta (or NaN = 1.0).

    ``blocked`` marks edges that cannot receive donations — PEC-masked
    edges (their M_eps row is never read by the solver, so borrowed
    mass would silently vanish).
    """
    best = -1
    best_val = -1.0
    for c in candidates:
        if blocked[c]:
            continue
        val = f_L[c]
        if np.isnan(val):
            if 1.0 > best_val:
                best = c
                best_val = 1.0
        elif val >= eta and val > best_val:
            best = c
            best_val = val
    return best


def _edge_endpoint_nodes(grid: GridLines) -> tuple[np.ndarray, np.ndarray]:
    """Flat node indices of both endpoints for all E-edges [Ex|Ey|Ez].

    Node layout: ``(i, j, k) -> (i*(Ny+1) + j)*(Nz+1) + k``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    stride_j = Nz + 1
    stride_i = (Ny + 1) * stride_j

    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny + 1), np.arange(Nz + 1), indexing="ij")
    a_x = (ii * stride_i + jj * stride_j + kk).ravel()
    b_x = a_x + stride_i

    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), np.arange(Nz + 1), indexing="ij")
    a_y = (ii * stride_i + jj * stride_j + kk).ravel()
    b_y = a_y + stride_j

    ii, jj, kk = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), np.arange(Nz), indexing="ij")
    a_z = (ii * stride_i + jj * stride_j + kk).ravel()
    b_z = a_z + 1

    return (np.concatenate([a_x, a_y, a_z]), np.concatenate([b_x, b_y, b_z]))


def _masked_component_labels(
    node_a: np.ndarray,
    node_b: np.ndarray,
    masked_flat: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Connected-component label per node in the masked-edge graph.

    Nodes not touched by any masked edge are singleton components, so
    ``labels[a] == labels[b]`` for a two-endpoint edge holds exactly
    when both endpoints sit on the *same* conductor.
    """
    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    a = node_a[masked_flat]
    b = node_b[masked_flat]
    graph = sp.coo_matrix(
        (np.ones(a.size, dtype=np.int8), (a, b)),
        shape=(n_nodes, n_nodes),
    )
    _, labels = connected_components(graph, directed=False)
    return labels


def _bbox_face_edges(grid: GridLines, n_Ex: int, n_Ey: int, n_Ez: int) -> np.ndarray:
    """Flat mask of E-edges lying in one of the six bbox faces.

    An edge lies in a face when it is *tangential* to it; the two
    indices transverse to the edge axis are the ones that can reach a
    wall.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    mask = np.zeros(n_Ex + n_Ey + n_Ez, dtype=bool)

    ex = np.zeros((Nx, Ny + 1, Nz + 1), dtype=bool)
    ex[:, (0, Ny), :] = True
    ex[:, :, (0, Nz)] = True
    ey = np.zeros((Nx + 1, Ny, Nz + 1), dtype=bool)
    ey[(0, Nx), :, :] = True
    ey[:, :, (0, Nz)] = True
    ez = np.zeros((Nx + 1, Ny + 1, Nz), dtype=bool)
    ez[(0, Nx), :, :] = True
    ez[:, (0, Ny), :] = True

    mask[:n_Ex] = ex.ravel()
    mask[n_Ex : n_Ex + n_Ey] = ey.ravel()
    mask[n_Ex + n_Ey :] = ez.ravel()
    return mask


def _enlarged_cell(
    f_L: np.ndarray,
    A_free: np.ndarray,
    eps_avg: np.ndarray,
    grid: GridLines,
    n_Ex: int,
    n_Ey: int,
    n_Ez: int,
    eta: float,
    blocked: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine enlarged-cell donors and borrowed amounts.

    For each curved-PEC edge with ``f_L < eta`` (short edge), find a
    neighbour along the same axis with ``f_L >= eta`` and record the
    borrowed quantity ``eps_avg · A_free`` [permittivity · m²].

    The donor edge will receive this contribution divided by its primal
    length (``L_primal_donor``) added to its M_eps in build_M_eps.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    n_total = n_Ex + n_Ey + n_Ez
    donor = np.full(n_total, -1, dtype=np.int64)
    borrowed = np.zeros(n_total, dtype=np.float64)

    short = np.nonzero(~np.isnan(f_L) & (f_L < eta))[0]

    for flat in short:
        if flat < n_Ex:
            rem = int(flat)
            stride_j = Nz + 1
            stride_i = (Ny + 1) * stride_j
            i = rem // stride_i
            rem %= stride_i
            j = rem // stride_j
            k = rem % stride_j
            cands = []
            if i > 0:
                cands.append((i - 1) * stride_i + j * stride_j + k)
            if i < Nx - 1:
                cands.append((i + 1) * stride_i + j * stride_j + k)
        elif flat < n_Ex + n_Ey:
            local = int(flat - n_Ex)
            stride_k = Nz + 1
            stride_j = Ny * stride_k
            i = local // stride_j
            rem = local % stride_j
            j = rem // stride_k
            k = rem % stride_k
            cands = []
            if j > 0:
                cands.append(n_Ex + i * stride_j + (j - 1) * stride_k + k)
            if j < Ny - 1:
                cands.append(n_Ex + i * stride_j + (j + 1) * stride_k + k)
        else:
            local = int(flat - n_Ex - n_Ey)
            stride_k = Nz
            stride_j = (Ny + 1) * stride_k
            i = local // stride_j
            rem = local % stride_j
            j = rem // stride_k
            k = rem % stride_k
            cands = []
            if k > 0:
                cands.append(n_Ex + n_Ey + i * stride_j + j * stride_k + (k - 1))
            if k < Nz - 1:
                cands.append(n_Ex + n_Ey + i * stride_j + j * stride_k + (k + 1))

        nbr = _pick_neighbour(cands, f_L, eta, blocked)
        if nbr < 0:
            continue

        eps_val = eps_avg[flat]
        if np.isnan(eps_val):
            eps_val = 1.0
        a_free = A_free[flat]
        if np.isnan(a_free) or a_free <= 0:
            continue
        donor[flat] = nbr
        borrowed[flat] = eps_val * a_free

    return donor, borrowed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _thin_sheet_cell_seed(
    grid: GridLines,
    thin_sheet_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> np.ndarray:
    """Cells strictly overlapping any thin-sheet metal box (WP-M2 seed).

    ``material_id`` cannot see a sub-cell-thin PEC volume (no cell
    centre lies inside the metal), so boundary-cell detection must be
    seeded explicitly for the conformal eps/mu candidate selection.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    seed = np.zeros((Nx, Ny, Nz), dtype=bool)
    nodes = (grid.x, grid.y, grid.z)
    for lo, hi in thin_sheet_boxes:
        sel = []
        for d in range(3):
            n = nodes[d]
            # Cell [n_i, n_{i+1}] strictly overlaps [lo_d, hi_d].
            i0 = int(np.searchsorted(n, lo[d], side="right")) - 1
            i1 = int(np.searchsorted(n, hi[d], side="left"))
            i0 = max(i0, 0)
            i1 = min(i1, len(n) - 1)
            if i1 <= i0:
                # Zero-width range along a degenerate direction —
                # widen to the touching cell so the seed stays usable.
                i1 = min(i0 + 1, len(n) - 1)
            sel.append(slice(i0, i1))
        seed[sel[0], sel[1], sel[2]] = True
    return seed


def _edges_intersecting_box(
    grid: GridLines,
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> np.ndarray:
    """Flat E-edge mask: edges whose segment intersects the box (WP-M2).

    Along the edge's span axis the overlap test is *strict* (an edge
    merely touching the box at an endpoint carries no metal), along
    the two fixed axes it is *inclusive* (an edge lying in the box's
    boundary plane — e.g. tangential edges in the sheet plane — is
    inside the metal and must be a PEC-adjacency candidate).
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x, y, z = grid.x, grid.y, grid.z

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz

    def span_mask(n: np.ndarray, a: float, b: float) -> np.ndarray:
        return (n[:-1] < b) & (n[1:] > a)  # strict overlap

    def node_mask(n: np.ndarray, a: float, b: float) -> np.ndarray:
        return (n >= a) & (n <= b)  # inclusive

    ex = (
        span_mask(x, lo[0], hi[0])[:, None, None]
        & node_mask(y, lo[1], hi[1])[None, :, None]
        & node_mask(z, lo[2], hi[2])[None, None, :]
    )
    ey = (
        node_mask(x, lo[0], hi[0])[:, None, None]
        & span_mask(y, lo[1], hi[1])[None, :, None]
        & node_mask(z, lo[2], hi[2])[None, None, :]
    )
    ez = (
        node_mask(x, lo[0], hi[0])[:, None, None]
        & node_mask(y, lo[1], hi[1])[None, :, None]
        & span_mask(z, lo[2], hi[2])[None, None, :]
    )
    out = np.zeros(n_Ex + n_Ey + n_Ez, dtype=bool)
    out[:n_Ex] = ex.ravel()
    out[n_Ex : n_Ex + n_Ey] = ey.ravel()
    out[n_Ex + n_Ey :] = ez.ravel()
    return out


def _apply_longitudinal_eps(
    grid: GridLines,
    absorbed_planes: dict[str, list[float]],
    shapes_with_material: list[tuple[object, int]],
    material_library: dict,
    conf_eps: np.ndarray,
    conf_sigma: np.ndarray,
    conf_f_area: np.ndarray,
    pec_adj_flat: np.ndarray,
    section_cache: dict,
    scale: float = 1.0,
) -> None:
    """Series (harmonic) eps/sigma on edges crossing absorbed planes.

    The WP-M3 floor merge drops sub-floor material planes, so primal
    edges can cross a dielectric boundary mid-edge for the first time.
    The DD-051 dual-face average is *transverse-only* — a series stack
    along the edge needs the length-weighted **harmonic** mean
    (series capacitors); measured without it: ±2.6–3.7 % ε_eff error
    on a layered parallel plate, non-converging (locked to the floor).

    Per crossing edge the pass sections the dual face at each
    *segment* midpoint (segments = edge span split by the absorbed
    planes) through the same OCC backend as the conformal pass, then
    combines: ``ε̄ = L / Σ (L_seg / ε_seg)``; σ analogously (any
    σ_seg = 0 short-circuits to 0, the exact DC series limit).
    PEC-adjacent edges are skipped — the line-solid f_L path owns
    them.  Mutates ``conf_eps`` / ``conf_sigma`` / ``conf_f_area``
    in place.
    """
    from magnelio.geo._occ_backend import (  # noqa: PLC0415
        compute_face_material_areas,
    )

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x, y, z = grid.x, grid.y, grid.z
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])
    zm = 0.5 * (z[:-1] + z[1:])

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)

    h_min = min(grid.dx.min(), grid.dy.min(), grid.dz.min())
    # Purely h-relative (DD-120): chordal error 1 % of the smallest
    # cell at ANY model scale.  The old absolute clamps (1e-4 m cap,
    # 1e-7 m floor) made the tessellation error larger than the cell
    # below ~10 um cells; the OCC robustness floor now lives in
    # scaled units inside cross_section_polygons.
    deflection = h_min * 1e-2

    # Collect (flat_edge, segment) face specs across all axes, batch
    # the OCC sectioning once.
    face_specs_list: list[tuple[float, float, float, float, float]] = []
    face_axes_list: list[int] = []
    seg_edge: list[int] = []  # flat edge index per segment
    seg_len: list[float] = []  # segment length per segment

    axis_data = {
        "x": (0, x, xm),
        "y": (1, y, ym),
        "z": (2, z, zm),
    }

    for axis, planes in absorbed_planes.items():
        ax_idx, nodes, _mids = axis_data[axis]
        # Group absorbed planes by the cell they fall into (strictly
        # interior; a plane on a node crosses no edge).
        by_cell: dict[int, list[float]] = {}
        for p in planes:
            c = int(np.searchsorted(nodes, p)) - 1
            if 0 <= c < len(nodes) - 1 and nodes[c] < p < nodes[c + 1]:
                by_cell.setdefault(c, []).append(p)

        for c, crossings in by_cell.items():
            bounds = [float(nodes[c]), *sorted(crossings), float(nodes[c + 1])]
            seg_mids = [0.5 * (a + b) for a, b in zip(bounds, bounds[1:])]
            seg_lens = [b - a for a, b in zip(bounds, bounds[1:])]

            if axis == "x":
                for j in range(1, Ny):
                    for k in range(1, Nz):
                        flat = c * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
                        if pec_adj_flat[flat]:
                            continue
                        for smid, slen in zip(seg_mids, seg_lens):
                            face_specs_list.append(
                                (smid, ym[j - 1], zm[k - 1], ym[j], zm[k]),
                            )
                            face_axes_list.append(0)
                            seg_edge.append(flat)
                            seg_len.append(slen)
            elif axis == "y":
                for i in range(1, Nx):
                    for k in range(1, Nz):
                        flat = n_Ex + i * Ny * (Nz + 1) + c * (Nz + 1) + k
                        if pec_adj_flat[flat]:
                            continue
                        for smid, slen in zip(seg_mids, seg_lens):
                            face_specs_list.append(
                                (smid, xm[i - 1], zm[k - 1], xm[i], zm[k]),
                            )
                            face_axes_list.append(1)
                            seg_edge.append(flat)
                            seg_len.append(slen)
            else:  # z
                for i in range(1, Nx):
                    for j in range(1, Ny):
                        flat = n_Ex + n_Ey + i * (Ny + 1) * Nz + j * Nz + c
                        if pec_adj_flat[flat]:
                            continue
                        for smid, slen in zip(seg_mids, seg_lens):
                            face_specs_list.append(
                                (smid, xm[i - 1], ym[j - 1], xm[i], ym[j]),
                            )
                            face_axes_list.append(2)
                            seg_edge.append(flat)
                            seg_len.append(slen)

    if not face_specs_list:
        return

    face_specs = np.array(face_specs_list, dtype=np.float64)
    face_axes = np.array(face_axes_list, dtype=np.int32)
    eps_vals = compute_face_material_areas(
        shapes_with_material,
        material_library,
        face_specs,
        face_axes,
        prop="epsilon",
        deflection=deflection,
        section_cache=section_cache,
        scale=scale,
    )
    sigma_vals = compute_face_material_areas(
        shapes_with_material,
        material_library,
        face_specs,
        face_axes,
        prop="sigma",
        deflection=deflection,
        section_cache=section_cache,
        scale=scale,
    )

    # Harmonic combination per edge.
    seg_edge_arr = np.asarray(seg_edge, dtype=np.int64)
    seg_len_arr = np.asarray(seg_len, dtype=np.float64)
    for flat in np.unique(seg_edge_arr):
        sel = seg_edge_arr == flat
        eps_s = eps_vals[sel]
        if np.any(np.isnan(eps_s)) or np.any(eps_s <= 0):
            continue  # incomplete sectioning — leave the edge as-is
        lens = seg_len_arr[sel]
        L = lens.sum()
        conf_eps[flat] = L / np.sum(lens / eps_s)
        sig_s = sigma_vals[sel]
        if np.any(np.isnan(sig_s)) or np.any(sig_s <= 0):
            conf_sigma[flat] = 0.0
        else:
            conf_sigma[flat] = L / np.sum(lens / sig_s)
        if np.isnan(conf_f_area[flat]):
            conf_f_area[flat] = 1.0


def compute_subcell_data(
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
    shapes_with_material: list[tuple[object, int]],
    pec_solid=None,
    eta: float = 0.0,
    section_cache: dict | None = None,
    thin_sheet_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]]
    | None = None,
    absorbed_planes: dict[str, list[float]] | None = None,
    fraction_mids: np.ndarray | None = None,
    scale: float = 1.0,
) -> EdgeMaterialData:
    """Classify all E-edges into the four DD-051 categories.

    Single classification pass producing the per-edge tuple
    ``(category, eps_avg, sigma_avg, A_free, L_free, donor, borrowed)``
    plus the PEC mask in one container.

    Internally, the conformal eps/sigma averaging on the dual face and
    the line-solid f_L computation against ``pec_solid`` are run as two
    OCC operations *against the same effective PEC solid*.  This closes
    the consistency gap that two parallel pipelines (``compute_conformal_eps``
    and ``compute_edge_pec_fractions``) had under tessellation drift.

    Parameters
    ----------
    grid : GridLines
        The mesh grid.
    material_id : np.ndarray, shape ``(Nx, Ny, Nz)``
        Per-cell material assignment.
    material_library : dict[int, Material]
    shapes_with_material : list of (shape_obj, material_id)
        Ordered from lowest to highest priority.
    pec_solid : TopoDS_Shape or None
        Effective PEC solid produced by
        :func:`~magnelio.geo._occ_backend.build_effective_pec_solid`.
        ``None`` disables curved-PEC sub-cell handling — every PEC
        boundary edge falls back to category 3 (full PEC mask).
    eta : float
        Enlarged-cell short-edge threshold ``f_L < eta``.  ``0`` disables
        enlarged-cell borrowing; short edges then degrade to category 3.
    section_cache : dict or None
        Cross-section polygon cache shared across this and any sibling
        calls (e.g. the H-face mu pipeline) to avoid recomputing the
        same OCC sections.
    fraction_mids : np.ndarray or None
        WP-C1 (DD-093): dispersive material ids whose per-edge dual-face
        area fractions are recorded in the returned container (see
        ``EdgeMaterialData.material_fractions``).  ``None`` (the default
        for meshes without dispersive materials) attaches no container.

    Returns
    -------
    EdgeMaterialData
    """
    from magnelio.geo._filling import compute_conformal_eps  # noqa: PLC0415
    from magnelio.mesh.indexing import build_pec_mask_faces  # noqa: PLC0415

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    x, y, z = grid.x, grid.y, grid.z

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_total = n_Ex + n_Ey + n_Ez

    # Initialise outputs to category 0 (interior bulk) with NaN data.
    category = np.zeros(n_total, dtype=np.int8)
    eps_avg = np.full(n_total, np.nan, dtype=np.float64)
    sigma_avg = np.full(n_total, np.nan, dtype=np.float64)
    A_free = np.full(n_total, np.nan, dtype=np.float64)
    L_free = np.full(n_total, np.nan, dtype=np.float64)
    f_A = np.full(n_total, np.nan, dtype=np.float64)

    # WP-M2 thin-sheet seed: material_id is blind to sub-cell-thin PEC
    # volumes, so both candidate gates (boundary cells for the
    # conformal eps pass, PEC edge adjacency for the line-solid f_L
    # pass) are seeded explicitly from the metal boxes.
    seed_cells = None
    if thin_sheet_boxes:
        seed_cells = _thin_sheet_cell_seed(grid, thin_sheet_boxes)

    # ---- Step 1: Conformal eps / sigma on the full dual face ------------
    # Returns ε̄ = (Σ_{non-PEC} ε_i · A_i) / A_dual on boundary edges,
    # NaN elsewhere, plus the free-area fraction f_A of the dual face.
    # Same OCC backend as today; section_cache is shared with the mu
    # pipeline.
    conf_eps, conf_sigma, conf_f_area, conf_fractions = compute_conformal_eps(
        shapes_with_material,
        grid,
        material_id,
        material_library,
        section_cache=section_cache,
        extra_boundary_cells=seed_cells,
        fraction_mids=fraction_mids,
        scale=scale,
    )

    # ---- Step 2: Staircase PEC mask + flat PEC adjacency ---------------
    pec_mask = build_pec_mask_faces(grid, material_id, material_library)
    pec_adj_flat = np.concatenate(
        [
            pec_mask[0, :n_Ex],
            pec_mask[1, :n_Ey],
            pec_mask[2, :n_Ez],
        ]
    )
    if thin_sheet_boxes:
        for box_lo, box_hi in thin_sheet_boxes:
            pec_adj_flat |= _edges_intersecting_box(grid, box_lo, box_hi)

    # ---- Step 2b: Longitudinal (series/harmonic) eps on edges that
    # cross material planes absorbed by the WP-M3 floor merge.
    if absorbed_planes:
        if section_cache is None:
            section_cache = {}
        _apply_longitudinal_eps(
            grid,
            absorbed_planes,
            shapes_with_material,
            material_library,
            conf_eps,
            conf_sigma,
            conf_f_area,
            pec_adj_flat,
            section_cache,
            scale=scale,
        )

    # ---- Step 3: Line-solid f_L for PEC-adjacent boundary edges --------
    # Use the same pec_solid snapshot as build_effective_pec_solid produced
    # for the conformal eps pass — DD-051 consistency requirement.
    f_L = np.full(n_total, np.nan, dtype=np.float64)
    use_dm = pec_solid is not None
    if use_dm:
        dm_cand = ~np.isnan(conf_eps) & pec_adj_flat

        edges_parts: list[np.ndarray] = []
        flat_parts: list[np.ndarray] = []

        ex_cand = np.nonzero(dm_cand[:n_Ex])[0]
        if len(ex_cand) > 0:
            stride_j = Nz + 1
            stride_i = (Ny + 1) * stride_j
            ii = ex_cand // stride_i
            jj = (ex_cand % stride_i) // stride_j
            kk = ex_cand % stride_j
            starts = np.column_stack([x[ii], y[jj], z[kk]])
            ends = np.column_stack([x[ii + 1], y[jj], z[kk]])
            edges_parts.append(np.stack([starts, ends], axis=1))
            flat_parts.append(ex_cand)

        ey_cand = np.nonzero(dm_cand[n_Ex : n_Ex + n_Ey])[0]
        if len(ey_cand) > 0:
            stride_k = Nz + 1
            stride_j = Ny * stride_k
            ii = ey_cand // stride_j
            jj = (ey_cand % stride_j) // stride_k
            kk = ey_cand % stride_k
            starts = np.column_stack([x[ii], y[jj], z[kk]])
            ends = np.column_stack([x[ii], y[jj + 1], z[kk]])
            edges_parts.append(np.stack([starts, ends], axis=1))
            flat_parts.append(ey_cand + n_Ex)

        ez_cand = np.nonzero(dm_cand[n_Ex + n_Ey :])[0]
        if len(ez_cand) > 0:
            stride_k = Nz
            stride_j = (Ny + 1) * stride_k
            ii = ez_cand // stride_j
            jj = (ez_cand % stride_j) // stride_k
            kk = ez_cand % stride_k
            starts = np.column_stack([x[ii], y[jj], z[kk]])
            ends = np.column_stack([x[ii], y[jj], z[kk + 1]])
            edges_parts.append(np.stack([starts, ends], axis=1))
            flat_parts.append(ez_cand + n_Ex + n_Ey)

        if edges_parts:
            from magnelio.geo._occ_backend import (  # noqa: PLC0415
                compute_edge_pec_fractions,
            )

            all_edges = np.concatenate(edges_parts, axis=0)
            all_flat = np.concatenate(flat_parts)
            # Cell-relative tolerance (DD-120): 1e-4 of the smallest
            # cell reproduces the historical 1e-8 m at the typical
            # 100 um floor and stays proportionate at any scale.
            h_min_tol = 1e-4 * min(dx.min(), dy.min(), dz.min())
            f_L_3d = compute_edge_pec_fractions(
                [pec_solid], all_edges, tolerance=h_min_tol, scale=scale
            )
            f_L[all_flat] = f_L_3d

    # ---- Step 4: Geometric scaffolding for A_dual and L_primal --------
    # Pre-build A_dual = geom_E · L_primal on each axis for vectorised
    # writes.  L_primal per edge is just the cell width along the edge.
    geom_E = _build_geom_E(grid)

    # Per-axis primal lengths (broadcasted to per-edge).
    L_primal = np.empty(n_total, dtype=np.float64)
    L_primal[:n_Ex] = np.broadcast_to(
        dx[:, None, None],
        (Nx, Ny + 1, Nz + 1),
    ).ravel()
    L_primal[n_Ex : n_Ex + n_Ey] = np.broadcast_to(
        dy[None, :, None],
        (Nx + 1, Ny, Nz + 1),
    ).ravel()
    L_primal[n_Ex + n_Ey :] = np.broadcast_to(
        dz[None, None, :],
        (Nx + 1, Ny + 1, Nz),
    ).ravel()

    A_dual = geom_E * L_primal

    # ---- Step 5: Categorise per edge ----------------------------------
    has_conf = ~np.isnan(conf_eps)

    # Cat 1 — dielectric boundary: conformal eps present, no PEC adj.
    cat1 = has_conf & ~pec_adj_flat
    # Floor: edges where the dielectric eps_avg drops below threshold are
    # demoted to interior PEC (rare; protects against 1/eps blow-up).
    cat1_floor = cat1 & (conf_eps <= _EPS_AVG_FLOOR)
    cat1 &= ~cat1_floor

    # Cat 2 — curved PEC sub-cell: PEC adjacent + valid f_L (line-solid
    # crossed real PEC volume) + f_L >= eta (un-mask threshold).
    if use_dm:
        cat2_unmask = pec_adj_flat & ~np.isnan(f_L) & (f_L >= eta)
    else:
        cat2_unmask = np.zeros(n_total, dtype=bool)

    # Cat 3 — interior PEC: every other PEC-adjacent edge.  Includes:
    #   - edges where line-solid found no PEC crossing (f_L NaN under
    #     pec_solid is None ⇒ all PEC-adj edges fall here),
    #   - short curved-PEC edges (f_L < eta) that need enlarged-cell.
    cat3 = pec_adj_flat & ~cat2_unmask
    cat3 |= cat1_floor

    # Apply categories.
    category[cat1] = CAT_DIELECTRIC_BOUNDARY
    category[cat2_unmask] = CAT_CURVED_PEC
    category[cat3] = CAT_INTERIOR_PEC

    # Cat 1 data: A_free = A_dual, L_free = L_primal, eps/sigma from conformal.
    cat1_idx = np.nonzero(cat1)[0]
    eps_avg[cat1_idx] = conf_eps[cat1_idx]
    sigma_avg[cat1_idx] = conf_sigma[cat1_idx]
    A_free[cat1_idx] = A_dual[cat1_idx]
    L_free[cat1_idx] = L_primal[cat1_idx]
    f_A[cat1_idx] = conf_f_area[cat1_idx]

    # Cat 2 data: eps/sigma from conformal (averaged over A_dual; the
    # mass-matrix builder uses the identity ε̄ · A_dual ≡ Σ ε_i · A_i),
    # A_free = A_dual · f_L (curvature-tangent approximation; exact for
    # ebene PEC walls, leading-order accurate for curved walls under
    # mesh refinement), L_free = L_primal · f_L.
    cat2_idx = np.nonzero(cat2_unmask)[0]
    eps_avg[cat2_idx] = conf_eps[cat2_idx]
    sigma_avg[cat2_idx] = conf_sigma[cat2_idx]
    A_free[cat2_idx] = A_dual[cat2_idx] * f_L[cat2_idx]
    L_free[cat2_idx] = L_primal[cat2_idx] * f_L[cat2_idx]
    f_A[cat2_idx] = conf_f_area[cat2_idx]

    # ---- Step 6: PEC mask un-masking ----------------------------------
    # Apply un-masks for cat 1 and cat 2 (cat 3 stays masked).
    unmask_flat = np.nonzero(cat1 | cat2_unmask)[0]
    if len(unmask_flat) > 0:
        ex_sel = unmask_flat[unmask_flat < n_Ex]
        pec_mask[0, ex_sel] = False
        ey_sel = unmask_flat[(unmask_flat >= n_Ex) & (unmask_flat < n_Ex + n_Ey)]
        pec_mask[1, ey_sel - n_Ex] = False
        ez_sel = unmask_flat[unmask_flat >= n_Ex + n_Ey]
        pec_mask[2, ez_sel - n_Ex - n_Ey] = False

    # ---- Step 6b: tangential-surface re-masking (DD-053) ---------------
    # An unmasked E edge whose two endpoint nodes are connected through
    # the masked-edge graph runs tangentially along a PEC surface.  The
    # 2D mode solvers put both endpoints into the same Dirichlet
    # conductor group (the edge voltage is identically zero in every
    # mode), while the un-masked 3D edge would evolve a surface-
    # tangential E — e.g. the longitudinal e_z along a conformal coax
    # feed — that no transversal port profile can launch or record;
    # measured as the −32 dB conformal port floor
    # (validation/coax_pair_consistent_mu_spike.py).  Applying
    # the PEC surface condition on these edges makes the 3D update and
    # the mode solvers see the same conductor (the DD-050 line).  The
    # connected-COMPONENT check (rather than mere conductor adjacency)
    # spares edges that bridge different conductors across a resolved
    # gap — those legitimately carry voltage.  One pass reaches the
    # fixed point: re-masked edges connect only nodes that were already
    # in the same component.
    masked_flat = np.concatenate(
        [
            pec_mask[0, :n_Ex],
            pec_mask[1, :n_Ey],
            pec_mask[2, :n_Ez],
        ]
    )
    if masked_flat.any():
        node_a, node_b = _edge_endpoint_nodes(grid)
        labels = _masked_component_labels(
            node_a,
            node_b,
            masked_flat,
            (Nx + 1) * (Ny + 1) * (Nz + 1),
        )
        tangential = ~masked_flat & (labels[node_a] == labels[node_b])
        tang_idx = np.nonzero(tangential)[0]
        if len(tang_idx) > 0:
            category[tang_idx] = CAT_INTERIOR_PEC
            # Keep the category-3 data invariant (NaN) — the solver
            # never reads masked-edge data; step 7's borrowing replays
            # its own f_L/conf_eps copies, so this is purely cosmetic
            # consistency for downstream consumers.
            for arr in (eps_avg, sigma_avg, A_free, L_free, f_A):
                arr[tang_idx] = np.nan
            ex_sel = tang_idx[tang_idx < n_Ex]
            pec_mask[0, ex_sel] = True
            ey_sel = tang_idx[(tang_idx >= n_Ex) & (tang_idx < n_Ex + n_Ey)]
            pec_mask[1, ey_sel - n_Ex] = True
            ez_sel = tang_idx[tang_idx >= n_Ex + n_Ey]
            pec_mask[2, ez_sel - n_Ex - n_Ey] = True
            masked_flat[tang_idx] = True

    # ---- Step 7: Enlarged-cell borrowing for short cat-2 candidates ----
    # Short edges (f_L < eta) that did not make cat 2 un-masking borrow
    # their dual-face contribution onto a longer neighbour.  We replay
    # the f_L < eta selection here against the *original* f_L array so
    # that the borrowing is independent of the un-mask threshold result.
    donor = np.full(n_total, -1, dtype=np.int64)
    borrowed = np.zeros(n_total, dtype=np.float64)
    if use_dm and eta > 0:
        # Build temp eps_avg + A_free for the short edges.  These edges
        # are cat 3 in `category` (masked), but the borrowing logic
        # needs their conformal data.
        tmp_eps = np.where(np.isnan(conf_eps), 1.0, conf_eps)
        # A_free for short edges: same heuristic as cat 2 (≈ A_dual·f_L).
        tmp_A_free = np.where(
            ~np.isnan(f_L),
            A_dual * np.nan_to_num(f_L, nan=0.0),
            0.0,
        )
        # An edge lying IN a bbox face must never *receive* a donation:
        # a face later closed with PEC has its tangential edges masked
        # by ``Mesh.with_pec_boundaries``, i.e. after the donors are
        # picked, and the borrowed mass would vanish without a trace.
        # Donating *out of* such an edge stays allowed — ``blocked`` is
        # consulted only for the receiver.
        donor, borrowed = _enlarged_cell(
            f_L,
            tmp_A_free,
            tmp_eps,
            grid,
            n_Ex,
            n_Ey,
            n_Ez,
            eta,
            blocked=masked_flat | _bbox_face_edges(grid, n_Ex, n_Ey, n_Ez),
        )

    return EdgeMaterialData(
        category=category,
        eps_avg=eps_avg,
        sigma_avg=sigma_avg,
        A_free=A_free,
        L_free=L_free,
        f_A=f_A,
        pec_mask=pec_mask,
        enlarged_cell_donor=donor,
        enlarged_cell_area=borrowed,
        fraction_mids=(
            np.asarray(fraction_mids, dtype=np.int64) if fraction_mids is not None else None
        ),
        material_fractions=conf_fractions,
    )


def compute_subcell_data_mu(
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
    shapes_with_material: list[tuple[object, int]],
    section_cache: dict | None = None,
    thin_sheet_boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]]
    | None = None,
    fraction_mids: np.ndarray | None = None,
    scale: float = 1.0,
) -> FaceMaterialData:
    """Classify H-faces into the three M_μ sub-cell categories.

    ``fraction_mids`` (WP-C1, DD-093): dispersive-μ/σ*-carrying
    material ids whose per-face primal-face area fractions are
    recorded in the returned container (see
    ``FaceMaterialData.material_fractions``); ``None`` attaches none.

    Krietenstein-style sub-cell mass-matrix correction for the
    magnetic side: identifies primal faces whose A_face is partially
    covered by PEC and produces the geometric ``A_face_free``
    reduction needed to make the FIT update consistent with the
    continuous Faraday equation on a curved PEC contour.

    Parameters
    ----------
    grid : GridLines
    material_id : np.ndarray, shape ``(Nx, Ny, Nz)``
    material_library : dict[int, Material]
    shapes_with_material : list of (shape_obj, material_id)
        Ordered from lowest to highest priority.
    section_cache : dict or None
        Shared cross-section cache with the E-edge pipeline.

    Returns
    -------
    FaceMaterialData
    """
    from magnelio.geo._filling import compute_conformal_mu  # noqa: PLC0415

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)
    n_total = n_Hx + n_Hy + n_Hz

    # Pre-build A_face per H-face for the geometric reduction.  Same
    # layout as compute_conformal_mu's flat_indices: [Hx | Hy | Hz].
    A_face = _build_A_face_H(grid)
    L_dual = _build_L_dual_H(grid)

    category = np.zeros(n_total, dtype=np.int8)
    mu_avg = np.full(n_total, np.nan, dtype=np.float64)
    A_face_free = np.full(n_total, np.nan, dtype=np.float64)
    L_dual_free = np.full(n_total, np.nan, dtype=np.float64)

    frac_mids_arr = np.asarray(fraction_mids, dtype=np.int64) if fraction_mids is not None else None
    # WP-C4: conformal σ* only when some material carries it — the
    # no-σ* path stays container-free and bit-identical.
    need_sigma_m = any(
        not mat.is_pec and any(s != 0.0 for s in mat.sigma_m) for mat in material_library.values()
    )
    if not shapes_with_material:
        return FaceMaterialData(
            category=category,
            mu_avg=mu_avg,
            A_face_free=A_face_free,
            L_dual_free=L_dual_free,
            A_face_pec=_staircase_a_face_pec(
                grid,
                material_id,
                material_library,
            ),
            A_face_pec_jump=np.zeros(n_total, dtype=np.float64),
            fraction_mids=frac_mids_arr,
            material_fractions=(
                np.full((frac_mids_arr.size, n_total), np.nan)
                if frac_mids_arr is not None
                else None
            ),
        )

    # WP-M2 thin-sheet seed — see compute_subcell_data.
    seed_cells = None
    if thin_sheet_boxes:
        seed_cells = _thin_sheet_cell_seed(grid, thin_sheet_boxes)

    # DD-099 boundary-layer geometric seed: a conductor sliver squeezed
    # against the domain boundary captures no cell centre, so no
    # material_id contrast marks its cells and its wall never registers
    # in A_face_pec (the bbox-tangency void, BOUNDARY_WALL_PLAN WP-B0).
    # NON-PEC cells in the six boundary layers get their faces
    # classified for the GEOMETRIC channels only — the matrix channels
    # keep the contrast-gated candidate set (PEC cells are excluded:
    # seeding them registers the domain's whole PEC hull as phantom
    # jump-wall families; measured 0.850…0.953 → 0.740…1.057 on the
    # padded coax).  A registered domain-END plane (a PEC background
    # continues beyond every shape, so the shifted re-evaluation reads
    # it as a shorting lid) is CORRECT for a portless end; faces that
    # host ports get continuation semantics at enumeration time
    # (`_masked_face_pec_views`, WP-B1.2b).
    pec_cells = np.zeros(material_id.shape, dtype=bool)
    for _mid, _mat in material_library.items():
        if _mat.is_pec:
            pec_cells |= material_id == _mid
    geom_cells = np.zeros_like(pec_cells)
    for _ax in range(3):
        _sl = [slice(None)] * 3
        _sl[_ax] = 0
        geom_cells[tuple(_sl)] = True
        _sl[_ax] = -1
        geom_cells[tuple(_sl)] = True
    geom_cells &= ~pec_cells

    (mu_vals, pec_frac, pec_frac_geom, pec_frac_jump, conf_fractions, sigma_m_vals) = (
        compute_conformal_mu(
            shapes_with_material,
            grid,
            material_id,
            material_library,
            section_cache=section_cache,
            extra_boundary_cells=seed_cells,
            fraction_mids=fraction_mids,
            with_sigma_m=need_sigma_m,
            geom_only_cells=geom_cells,
            scale=scale,
        )
    )

    # Cat-1: dielectric boundary.  Marker: μ̄ defined AND no PEC overlap
    # (pec_frac ≈ 0).  A_face_free = A_face, L_dual_free = L_dual.
    has_mu = ~np.isnan(mu_vals)
    pec_overlap_threshold = 1e-9
    cat1 = has_mu & (np.nan_to_num(pec_frac, nan=0.0) <= pec_overlap_threshold)
    cat2 = has_mu & (np.nan_to_num(pec_frac, nan=0.0) > pec_overlap_threshold)

    # Apply categories.
    category[cat1] = 1
    category[cat2] = 2
    mu_avg[has_mu] = mu_vals[has_mu]
    A_face_free[cat1] = A_face[cat1]
    L_dual_free[cat1] = L_dual[cat1]
    # Cat-2: A_face_free = A_face · (1 - pec_frac).  L_dual_free
    # falls back to L_dual; the typical curved-PEC geometry has the
    # dual edge through the cell centre well clear of the PEC region,
    # so this is exact for the round-WG case.  A geometry where the
    # PEC contour cuts a dual edge would need a separate line-solid
    # call against the H dual-edge; that is an O(h)-correction on the
    # rare case and is queued for a follow-up if it ever surfaces.
    A_face_free[cat2] = A_face[cat2] * (1.0 - pec_frac[cat2])
    L_dual_free[cat2] = L_dual[cat2]

    # DD-087: geometric PEC area per face, NaN-free.  Exact OCC values
    # where the classifier computed a candidate face (pec_frac_geom
    # resolves degenerate section planes by the shifted re-evaluation,
    # including the exact 0 of a genuinely free face); non-candidates
    # fall back to the staircase cell rule.
    A_face_pec = _staircase_a_face_pec(grid, material_id, material_library)
    exact = ~np.isnan(pec_frac_geom)
    A_face_pec[exact] = A_face[exact] * pec_frac_geom[exact]
    # Flat wall lying in the face's own plane (signed; see the field
    # docstring).  Only classifier candidates can carry one: a wall
    # face separates cells of unequal material_id, so it is always
    # offered to OCC — non-candidates keep the initialised zero.
    A_face_pec_jump = A_face * pec_frac_jump

    return FaceMaterialData(
        category=category,
        mu_avg=mu_avg,
        A_face_free=A_face_free,
        L_dual_free=L_dual_free,
        A_face_pec=A_face_pec,
        A_face_pec_jump=A_face_pec_jump,
        fraction_mids=frac_mids_arr,
        material_fractions=conf_fractions,
        sigma_m_avg=sigma_m_vals,
    )


def _staircase_a_face_pec(
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
) -> np.ndarray:
    """Cell-rule PEC face areas: full when >= 1 neighbour cell is PEC.

    The staircase default for faces without a reliable OCC statement
    (DD-087): a face bordering a PEC-classified cell counts fully PEC
    (grid-snapped flat walls land exactly here — the wall area then
    telescopes into the adjacent air cell); domain padding is non-PEC.
    Layout: flat ``[Hx | Hy | Hz]``.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    pec = np.zeros((Nx, Ny, Nz), dtype=bool)
    for mid, mat in material_library.items():
        if mat.is_pec:
            pec |= material_id == mid

    afull = [
        np.broadcast_to((dy[:, None] * dz[None, :])[None, :, :], (Nx + 1, Ny, Nz)),
        np.broadcast_to((dx[:, None] * dz[None, :])[:, None, :], (Nx, Ny + 1, Nz)),
        np.broadcast_to((dx[:, None] * dy[None, :])[:, :, None], (Nx, Ny, Nz + 1)),
    ]
    parts = []
    for ax in range(3):
        pad = [(0, 0)] * 3
        pad[ax] = (1, 1)
        pp = np.pad(pec, pad, constant_values=False)
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(None, -1)
        sl_hi[ax] = slice(1, None)
        near_pec = pp[tuple(sl_lo)] | pp[tuple(sl_hi)]
        parts.append(np.where(near_pec, afull[ax], 0.0).ravel())
    return np.concatenate(parts)


def _build_A_face_H(grid: GridLines) -> np.ndarray:
    """Per-H-face primal-face area, concatenated [Hx | Hy | Hz]."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # Hx face area = dy * dz, shape (Nx+1, Ny, Nz)
    A_hx = np.broadcast_to(
        (dy[:, None] * dz[None, :])[None, :, :],
        (Nx + 1, Ny, Nz),
    ).ravel()
    # Hy face area = dx * dz, shape (Nx, Ny+1, Nz)
    A_hy = np.broadcast_to(
        (dx[:, None] * dz[None, :])[:, None, :],
        (Nx, Ny + 1, Nz),
    ).ravel()
    # Hz face area = dx * dy, shape (Nx, Ny, Nz+1)
    A_hz = np.broadcast_to(
        (dx[:, None] * dy[None, :])[:, :, None],
        (Nx, Ny, Nz + 1),
    ).ravel()
    return np.concatenate([A_hx, A_hy, A_hz])


def _build_L_dual_H(grid: GridLines) -> np.ndarray:
    """Per-H-face dual-edge length, concatenated [Hx | Hy | Hz]."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    def _avg(d, N):
        if N == 0:
            return np.ones(1)
        out = np.empty(N + 1)
        out[0] = d[0]
        if N > 1:
            out[1:N] = 0.5 * (d[: N - 1] + d[1:N])
        out[N] = d[N - 1]
        return out

    dx_avg = _avg(dx, Nx)
    dy_avg = _avg(dy, Ny)
    dz_avg = _avg(dz, Nz)

    # Hx dual edge along x: dx_avg, shape (Nx+1, Ny, Nz)
    L_hx = np.broadcast_to(dx_avg[:, None, None], (Nx + 1, Ny, Nz)).ravel()
    # Hy dual edge along y: dy_avg, shape (Nx, Ny+1, Nz)
    L_hy = np.broadcast_to(dy_avg[None, :, None], (Nx, Ny + 1, Nz)).ravel()
    # Hz dual edge along z: dz_avg, shape (Nx, Ny, Nz+1)
    L_hz = np.broadcast_to(dz_avg[None, None, :], (Nx, Ny, Nz + 1)).ravel()
    return np.concatenate([L_hx, L_hy, L_hz])
