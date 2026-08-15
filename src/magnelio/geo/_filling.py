"""
Cell classification and area-fraction computation from cross-section polygons.

This module provides two main operations:

1. **classify_cells_from_cross_sections** — replaces the per-cell
   ``point_in_shape`` loop with point-in-polygon tests on pre-computed
   2D cross-sections.  Output is the ``material_id`` array.

2. **compute_conformal_eps** — for each E-edge at a material boundary,
   computes the area-weighted effective permittivity via 3D face-solid
   intersection (thin-box volume method).
"""

from __future__ import annotations

import numpy as np

from magnelio.geo._polygon_clip import points_in_polygon
from magnelio.mesh.grid import GridLines

# Type alias for the cross-section cache produced by batch_cross_sections.
# {(axis, plane_idx): [(material_id, [polygons]), ...]}
CrossSectionCache = dict[tuple[str, int], list[tuple[int, list[np.ndarray]]]]


# ---------------------------------------------------------------------------
# Cell classification (replaces point_in_shape loop)
# ---------------------------------------------------------------------------


def classify_cells_from_cross_sections(
    cache: CrossSectionCache,
    grid: GridLines,
    background_id: int = 0,
) -> np.ndarray:
    """Assign material IDs to cells using cross-section polygon data.

    For each cell ``(i, j, k)``, the cell-centre point is tested against
    cross-section polygons at the cell-centre x-plane.  The last matching
    shape wins (same semantics as the previous ``point_in_shape`` loop).

    Parameters
    ----------
    cache : CrossSectionCache
        Pre-computed cross-sections, keyed by ``('x', i)`` for x-plane
        cross-sections at cell-centre positions.
    grid : GridLines
        The mesh grid.
    background_id : int
        Material ID for cells not covered by any shape (default: 0 = air).

    Returns
    -------
    np.ndarray
        Shape ``(Nx, Ny, Nz)``, dtype int32.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    material_id = np.full((Nx, Ny, Nz), background_id, dtype=np.int32)

    # Cell centres in transverse directions (y, z for x-plane cross-sections)
    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    zc = 0.5 * (grid.z[:-1] + grid.z[1:])

    # Pre-compute the (Ny, Nz) grids of point coordinates once — they
    # are reused for every x-plane.  ``indexing="ij"`` so YY[j,k] = yc[j],
    # ZZ[j,k] = zc[k], matching material_id[i, j, k] axis convention.
    YY, ZZ = np.meshgrid(yc, zc, indexing="ij")

    for i in range(Nx):
        key = ("x", i)
        if key not in cache:
            continue

        entries = cache[key]
        for mat_id, polygons in entries:
            # Even-odd rule: a point belongs to the shape iff it lies
            # inside an odd number of its contours.  Outer boundary
            # toggles a cell ON; an inner-boundary (hole) toggles it
            # back OFF.  Vectorised XOR over the (Ny, Nz) grid is
            # ~30-100× faster than the previous Python triple loop.
            mask = np.zeros((Ny, Nz), dtype=bool)
            for poly in polygons:
                mask ^= points_in_polygon(YY, ZZ, poly)
            if mask.any():
                material_id[i][mask] = mat_id

    return material_id


# ---------------------------------------------------------------------------
# Conformal M_eps: dual-face area fractions
# ---------------------------------------------------------------------------


def compute_conformal_eps(
    shapes_with_material: list[tuple[object, int]],
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
    section_cache: dict | None = None,
    extra_boundary_cells: np.ndarray | None = None,
    fraction_mids: np.ndarray | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute effective eps_r and sigma per E-edge via 3D face-solid intersection.

    For E-edges at material boundaries, constructs thin boxes representing
    each dual face and intersects them with the 3D material solids to compute
    exact area-weighted eps_r and sigma.

    PEC material area is excluded from the average (D = 0 inside PEC).

    Parameters
    ----------
    shapes_with_material : list of (shape_obj, material_id)
        Ordered from lowest to highest priority.
    grid : GridLines
        The mesh grid.
    material_id : np.ndarray
        Shape ``(Nx, Ny, Nz)``.
    material_library : dict
        ``{int: Material}`` mapping.

    fraction_mids : np.ndarray or None, default None
        WP-C1 (DD-093): material ids whose effective post-priority area
        fraction of every processed dual face is requested (the
        dispersive/σ*-carrying materials).  Fractions come from the
        SAME budget cascade as ``eps_avg`` (PEC claims area; uncovered
        remainder → background id 0), so on fully sectioned faces
        ``Σᵢ fᵢ·εᵢ`` over ALL participating ids reproduces ``eps_avg``
        exactly.

    Returns
    -------
    tuple of np.ndarray
        ``(conformal_eps, conformal_sigma, free_area_frac, fractions)``
        the first three of shape ``(n_Ex + n_Ey + n_Ez,)``.  NaN where
        staircase should be used.
        ``free_area_frac`` is the non-PEC fraction ``f_A`` of the dual
        face, from the same OCC tessellation as the weighted averages —
        so ``eps_avg / f_A`` recovers the material average over the
        *free* part of the dual face exactly (DD-053 ``eps_pair``).
        ``fractions`` is ``None`` when ``fraction_mids`` is ``None``,
        else shape ``(len(fraction_mids), n_E_total)`` — NaN on edges
        the conformal pass did not process (their dual face lies in
        bulk; the cat-0 fraction is 1 for the owning material, resolved
        by the consumer's staircase lookup); a computed 0 is a genuine
        zero share.
    """
    from magnelio.mesh._conformal import detect_boundary_cells  # noqa: PLC0415

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x, y, z = grid.x, grid.y, grid.z

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_total = n_Ex + n_Ey + n_Ez

    eps_result = np.full(n_total, np.nan, dtype=np.float64)
    sigma_result = np.full(n_total, np.nan, dtype=np.float64)
    f_area_result = np.full(n_total, np.nan, dtype=np.float64)
    # NaN = "not processed by the conformal pass" (bulk edges, and
    # entities later re-categorised by post-passes like DD-053) —
    # consumers fall back to the staircase lookup there.  A computed
    # zero is a genuine zero share.
    fractions_result = (
        np.full((len(fraction_mids), n_total), np.nan, dtype=np.float64)
        if fraction_mids is not None
        else None
    )

    if not shapes_with_material:
        return eps_result, sigma_result, f_area_result, fractions_result

    boundary = detect_boundary_cells(material_id)
    if extra_boundary_cells is not None:
        # WP-M2: cells intersecting a thin-sheet metal volume are
        # boundary cells even when material_id is blind to the
        # sub-cell-thin PEC (no cell centre lies inside the metal).
        boundary = boundary | extra_boundary_cells

    # Dual-face midpoints
    ym = 0.5 * (y[:-1] + y[1:])  # shape (Ny,)
    zm = 0.5 * (z[:-1] + z[1:])  # shape (Nz,)
    xm = 0.5 * (x[:-1] + x[1:])  # shape (Nx,)

    # Dual-face extents including the two domain walls: an edge lying in
    # a bbox face owns the truncated dual face between the wall and the
    # first dual line.  ``eps_avg`` and ``f_A`` stay intensive — the
    # consumer's ``A_dual`` is the full boundary cell (the mirror
    # convention of ``_build_avg_d``), so an average taken over the
    # truncated half needs no factor, and the material continues by
    # mirror symmetry, by extrusion, or not at all.
    xe = np.concatenate(([x[0]], xm, [x[-1]]))  # shape (Nx + 2,)
    ye = np.concatenate(([y[0]], ym, [y[-1]]))  # shape (Ny + 2,)
    ze = np.concatenate(([z[0]], zm, [z[-1]]))  # shape (Nz + 2,)

    # Boundary-edge masks (vectorised).  Padding the cell mask lets the
    # four-cell OR run over the boundary indices too: outside the domain
    # there is no cell, so the pad contributes nothing.
    bpad = np.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=bool)
    bpad[1:-1, 1:-1, 1:-1] = boundary

    b = bpad[1:-1, :, :]
    bnd_ex = (
        b[:, 0 : Ny + 1, 0 : Nz + 1]
        | b[:, 1 : Ny + 2, 0 : Nz + 1]
        | b[:, 0 : Ny + 1, 1 : Nz + 2]
        | b[:, 1 : Ny + 2, 1 : Nz + 2]
    )
    b = bpad[:, 1:-1, :]
    bnd_ey = (
        b[0 : Nx + 1, :, 0 : Nz + 1]
        | b[1 : Nx + 2, :, 0 : Nz + 1]
        | b[0 : Nx + 1, :, 1 : Nz + 2]
        | b[1 : Nx + 2, :, 1 : Nz + 2]
    )
    b = bpad[:, :, 1:-1]
    bnd_ez = (
        b[0 : Nx + 1, 0 : Ny + 1, :]
        | b[1 : Nx + 2, 0 : Ny + 1, :]
        | b[0 : Nx + 1, 1 : Ny + 2, :]
        | b[1 : Nx + 2, 1 : Ny + 2, :]
    )

    # Collect all boundary face specifications and their flat indices
    face_specs_list: list[tuple[float, float, float, float, float]] = []
    face_axes_list: list[int] = []
    flat_indices: list[int] = []

    # Ex edges: dual face at x=xm[i], extent [ye[j]..ye[j+1]] x [ze[k]..ze[k+1]]
    ex_ijk = np.argwhere(bnd_ex)
    for row in ex_ijk:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        face_specs_list.append((xm[i], ye[j], ze[k], ye[j + 1], ze[k + 1]))
        face_axes_list.append(0)
        flat_indices.append(i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k)

    # Ey edges: dual face at y=ym[j], extent [xe[i]..xe[i+1]] x [ze[k]..ze[k+1]]
    ey_ijk = np.argwhere(bnd_ey)
    for row in ey_ijk:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        face_specs_list.append((ym[j], xe[i], ze[k], xe[i + 1], ze[k + 1]))
        face_axes_list.append(1)
        flat_indices.append(n_Ex + i * Ny * (Nz + 1) + j * (Nz + 1) + k)

    # Ez edges: dual face at z=zm[k], extent [xe[i]..xe[i+1]] x [ye[j]..ye[j+1]]
    ez_ijk = np.argwhere(bnd_ez)
    for row in ez_ijk:
        i, j, k = int(row[0]), int(row[1]), int(row[2])
        face_specs_list.append((zm[k], xe[i], ye[j], xe[i + 1], ye[j + 1]))
        face_axes_list.append(2)
        flat_indices.append(n_Ex + n_Ey + i * (Ny + 1) * Nz + j * Nz + k)

    if not face_specs_list:
        return eps_result, sigma_result, f_area_result, fractions_result

    face_specs = np.array(face_specs_list, dtype=np.float64)
    face_axes = np.array(face_axes_list, dtype=np.int32)

    from magnelio.geo._occ_backend import compute_face_material_areas  # noqa: PLC0415

    # Cross-section tessellation deflection: scaled to the smallest cell,
    # bounded above to keep curved-boundary errors << cell size.
    h_min = min(grid.dx.min(), grid.dy.min(), grid.dz.min())
    # Purely h-relative (DD-120): chordal error 1 % of the smallest
    # cell at ANY model scale.  The old absolute clamps (1e-4 m cap,
    # 1e-7 m floor) made the tessellation error larger than the cell
    # below ~10 um cells; the OCC robustness floor now lives in
    # scaled units inside cross_section_polygons.
    deflection = h_min * 1e-2

    # Share section_cache between the eps and sigma calls (same face_specs).
    # If the caller didn't pass one, create a local cache for this pair.
    if section_cache is None:
        section_cache = {}
    pec_areas = np.zeros(len(face_specs), dtype=np.float64)
    face_fracs = (
        np.zeros((len(fraction_mids), len(face_specs)), dtype=np.float64)
        if fraction_mids is not None
        else None
    )
    eps_vals = compute_face_material_areas(
        shapes_with_material,
        material_library,
        face_specs,
        face_axes,
        prop="epsilon",
        deflection=deflection,
        section_cache=section_cache,
        pec_area_out=pec_areas,
        material_fraction_mids=fraction_mids,
        material_fractions_out=face_fracs,
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

    flat_arr = np.array(flat_indices, dtype=np.int64)
    valid = ~np.isnan(eps_vals)
    eps_result[flat_arr[valid]] = eps_vals[valid]
    valid_s = ~np.isnan(sigma_vals)
    sigma_result[flat_arr[valid_s]] = sigma_vals[valid_s]
    dual_areas = (face_specs[:, 3] - face_specs[:, 1]) * (face_specs[:, 4] - face_specs[:, 2])
    safe_areas = np.where(dual_areas > 0, dual_areas, 1.0)
    f_area_local = np.clip(1.0 - pec_areas / safe_areas, 0.0, 1.0)
    f_area_result[flat_arr[valid]] = f_area_local[valid]
    if fractions_result is not None:
        fractions_result[:, flat_arr[valid]] = face_fracs[:, valid]

    return eps_result, sigma_result, f_area_result, fractions_result


# ---------------------------------------------------------------------------
# Conformal M_mu: primal-face area fractions
# ---------------------------------------------------------------------------


def compute_conformal_mu(
    shapes_with_material: list[tuple[object, int]],
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
    section_cache: dict | None = None,
    extra_boundary_cells: np.ndarray | None = None,
    fraction_mids: np.ndarray | None = None,
    with_sigma_m: bool = False,
    geom_only_cells: np.ndarray | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compute effective mu_r and PEC-overlap area per H-face.

    ``with_sigma_m`` (WP-C4, DD-093): additionally area-average the
    magnetic loss σ* over the same primal faces (identical section
    set — the shared cache makes the second pass cheap) and append it
    as a sixth return value (``None`` when not requested; NaN where
    the staircase applies, mirroring ``mu_r``).  PEC claims area with
    σ* = 0; uncovered background is lossless.

    ``fraction_mids`` (WP-C1, DD-093): as in
    :func:`compute_conformal_eps` — per-material post-priority area
    fractions of the primal faces, appended as a fifth return value
    (``None`` when not requested; shape ``(n_mids, n_H_total)``, NaN
    on faces the conformal pass did not process).  Fractions follow
    the ``prop='mu'`` budget (PEC participates like any id and simply
    reports its share).

    For H-faces at material boundaries, constructs thin boxes representing
    each primal face and intersects them with the 3D material solids.

    The first output is the area-weighted ``μ̄`` over the full primal face
    (PEC contributes with ``μ_r = 1`` since B is finite in PEC).  The
    second output is the *PEC-overlap fraction* ``A_PEC / A_face``,
    needed for the Krietenstein curved-PEC sub-cell correction on
    M_μ — the Cat-2 face formula uses ``A_face_free = A_face·(1 - A_PEC/A_face)``
    instead of the full primal face area.  Without the geometric
    A_face reduction, μ̄ on a curved PEC contour stays 1.0 (vacuum =
    PEC = 1) and the staircase value is recovered, which is the
    pre-DD-051-Variante-A pathology that this output addresses.

    Parameters
    ----------
    shapes_with_material : list of (shape_obj, material_id)
        Ordered from lowest to highest priority.
    grid : GridLines
        The mesh grid.
    material_id : np.ndarray
        Shape ``(Nx, Ny, Nz)``.
    material_library : dict
        ``{int: Material}`` mapping.

    Returns
    -------
    mu_r : np.ndarray
        Shape ``(n_Hx + n_Hy + n_Hz,)`` with effective ``μ_r``.
        NaN where staircase should be used.
    pec_fraction : np.ndarray
        Same shape.  Per-face ``A_PEC / A_face`` ∈ [0, 1].  NaN where
        no boundary computation was done (staircase fallback).  On
        planes tangent to a material-boundary face the value follows
        the DD-106 min-convention (blocked only where embedded in PEC
        on both sides), which keeps ``A_face_free`` deterministic and
        translation-invariant along extruded feeds.
    pec_fraction_geom : np.ndarray
        Same shape.  The GEOMETRIC PEC fraction (DD-087): identical to
        ``pec_fraction`` except on degenerate section planes, where it
        books the max-convention side of the shifted re-evaluation
        (wall area lands in the adjacent non-PEC cell).  Feeds
        ``FaceMaterialData.A_face_pec`` — never the M_μ categories.
    pec_frac_jump : np.ndarray
        Same shape.  The fraction of the face occupied by wall lying IN
        the face's own plane (DD-087): the jump of the PEC overlap
        across the plane, zero (never NaN) wherever the overlap is
        continuous.  Feeds ``FaceMaterialData.A_face_pec_flat``, which
        separates a cell's grid-aligned wall portion from its curved
        one — the divergence cell vector cannot resolve both at once.
    """
    from magnelio.mesh._conformal import detect_boundary_cells  # noqa: PLC0415

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x, y, z = grid.x, grid.y, grid.z

    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    n_Hz = Nx * Ny * (Nz + 1)
    n_total = n_Hx + n_Hy + n_Hz

    result = np.full(n_total, np.nan, dtype=np.float64)
    pec_frac = np.full(n_total, np.nan, dtype=np.float64)
    pec_frac_geom = np.full(n_total, np.nan, dtype=np.float64)
    pec_frac_jump = np.zeros(n_total, dtype=np.float64)
    # NaN = "not processed" — see compute_conformal_eps.
    fractions_result = (
        np.full((len(fraction_mids), n_total), np.nan, dtype=np.float64)
        if fraction_mids is not None
        else None
    )
    sigma_m_result = np.full(n_total, np.nan, dtype=np.float64) if with_sigma_m else None

    if not shapes_with_material:
        return result, pec_frac, pec_frac_geom, pec_frac_jump, fractions_result, sigma_m_result

    # Share one section cache across the mu / sigma_m / geom-only
    # calls below (deterministic per-key entries at fixed deflection).
    if section_cache is None:
        section_cache = {}

    # Cell-level boundary mask: a cell whose neighbour has a different
    # material_id, OR a cell that is itself geometrically cut by a
    # material boundary even if all face-neighbour cell-centres share
    # the same material_id (which happens on extruded geometries —
    # e.g. round-WG with PEC mantel running parallel to x makes every
    # x-strip homogeneous in material_id, so the previous "Cell-Pair
    # material_id differs" detection missed every Hx face inside the
    # strip even though those faces *are* cut by the cylindrical PEC
    # contour).  detect_boundary_cells returns the staircase cells
    # adjacent to a material change; this picks up every cell whose
    # *primal* face is partially in PEC.
    boundary = detect_boundary_cells(material_id)
    if extra_boundary_cells is not None:
        # WP-M2: thin-sheet seed — see compute_conformal_eps.
        boundary = boundary | extra_boundary_cells
    # DD-099: geom-only candidates.  Their faces run through the OCC
    # classification in a SEPARATE call with a fresh section cache
    # (batching them with the contrast candidates changes the shifted
    # re-evaluation of degenerate section planes for the OLD faces —
    # measured full-coverage flips across all z planes of an extruded
    # coax), and ONLY the geometric PEC channels (pec_frac_geom /
    # pec_frac_jump -> A_face_pec / A_face_pec_flat) are written back —
    # mu, pec_frac, material fractions and sigma_m keep their NaN
    # staircase fallback, so the material matrices and every
    # contrast-candidate result are bit-identical with and without the
    # seed.
    geom = geom_only_cells & ~boundary if geom_only_cells is not None else np.zeros_like(boundary)

    face_specs_list: list[tuple[float, float, float, float, float]] = []
    face_axes_list: list[int] = []
    flat_indices: list[int] = []
    face_areas_list: list[float] = []
    geom_flags: list[bool] = []

    # Hx faces at x=x[i].  An interior face (1 <= i <= Nx-1) is a
    # candidate when either of its two adjacent cells is a boundary
    # cell.  Domain-boundary faces (i = 0 or i = Nx) are candidates
    # when their *single* adjacent cell is a boundary cell — those
    # are the port-plane H faces that the 2D mode solver consumes,
    # and on extruded curved-PEC geometries (round-WG) the Cylinder
    # contour cuts them just like every interior x-slice does.
    for i in range(0, Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                if i == 0:
                    is_bnd = bool(boundary[0, j, k])
                    is_geom = bool(geom[0, j, k])
                elif i == Nx:
                    is_bnd = bool(boundary[Nx - 1, j, k])
                    is_geom = bool(geom[Nx - 1, j, k])
                else:
                    is_bnd = bool(boundary[i - 1, j, k] or boundary[i, j, k])
                    is_geom = bool(geom[i - 1, j, k] or geom[i, j, k])
                if not (is_bnd or is_geom):
                    continue
                face_specs_list.append((x[i], y[j], z[k], y[j + 1], z[k + 1]))
                face_axes_list.append(0)
                flat_indices.append(i * Ny * Nz + j * Nz + k)
                face_areas_list.append(float((y[j + 1] - y[j]) * (z[k + 1] - z[k])))
                geom_flags.append(not is_bnd)

    # Hy faces at y=y[j]
    for j in range(0, Ny + 1):
        for i in range(Nx):
            for k in range(Nz):
                if j == 0:
                    is_bnd = bool(boundary[i, 0, k])
                    is_geom = bool(geom[i, 0, k])
                elif j == Ny:
                    is_bnd = bool(boundary[i, Ny - 1, k])
                    is_geom = bool(geom[i, Ny - 1, k])
                else:
                    is_bnd = bool(boundary[i, j - 1, k] or boundary[i, j, k])
                    is_geom = bool(geom[i, j - 1, k] or geom[i, j, k])
                if not (is_bnd or is_geom):
                    continue
                face_specs_list.append((y[j], x[i], z[k], x[i + 1], z[k + 1]))
                face_axes_list.append(1)
                flat_indices.append(n_Hx + i * (Ny + 1) * Nz + j * Nz + k)
                face_areas_list.append(float((x[i + 1] - x[i]) * (z[k + 1] - z[k])))
                geom_flags.append(not is_bnd)

    # Hz faces at z=z[k]
    for k in range(0, Nz + 1):
        for i in range(Nx):
            for j in range(Ny):
                if k == 0:
                    is_bnd = bool(boundary[i, j, 0])
                    is_geom = bool(geom[i, j, 0])
                elif k == Nz:
                    is_bnd = bool(boundary[i, j, Nz - 1])
                    is_geom = bool(geom[i, j, Nz - 1])
                else:
                    is_bnd = bool(boundary[i, j, k - 1] or boundary[i, j, k])
                    is_geom = bool(geom[i, j, k - 1] or geom[i, j, k])
                if not (is_bnd or is_geom):
                    continue
                face_specs_list.append((z[k], x[i], y[j], x[i + 1], y[j + 1]))
                face_axes_list.append(2)
                flat_indices.append(n_Hx + n_Hy + i * Ny * (Nz + 1) + j * (Nz + 1) + k)
                face_areas_list.append(float((x[i + 1] - x[i]) * (y[j + 1] - y[j])))
                geom_flags.append(not is_bnd)

    if not face_specs_list:
        return result, pec_frac, pec_frac_geom, pec_frac_jump, fractions_result, sigma_m_result

    from magnelio.geo._occ_backend import compute_face_material_areas  # noqa: PLC0415

    h_min = min(grid.dx.min(), grid.dy.min(), grid.dz.min())
    # Purely h-relative (DD-120): chordal error 1 % of the smallest
    # cell at ANY model scale.  The old absolute clamps (1e-4 m cap,
    # 1e-7 m floor) made the tessellation error larger than the cell
    # below ~10 um cells; the OCC robustness floor now lives in
    # scaled units inside cross_section_polygons.
    deflection = h_min * 1e-2
    # DD-106: planes on the domain hull are evaluated one-sided
    # (interior) for the matrix channel — see compute_face_material_areas.
    domain_bounds = (
        (float(x[0]), float(x[-1])),
        (float(y[0]), float(y[-1])),
        (float(z[0]), float(z[-1])),
    )

    geom_only = np.array(geom_flags, dtype=bool)
    main = ~geom_only
    face_specs = np.array(face_specs_list, dtype=np.float64)[main]
    face_axes = np.array(face_axes_list, dtype=np.int32)[main]
    face_areas = np.array(face_areas_list, dtype=np.float64)[main]
    flat_all = np.array(flat_indices, dtype=np.int64)

    if main.any():
        pec_areas = np.zeros(len(face_specs), dtype=np.float64)
        pec_areas_geom = np.zeros(len(face_specs), dtype=np.float64)
        pec_areas_jump = np.zeros(len(face_specs), dtype=np.float64)
        face_fracs = (
            np.zeros((len(fraction_mids), len(face_specs)), dtype=np.float64)
            if fraction_mids is not None
            else None
        )
        mu_vals = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop="mu",
            deflection=deflection,
            section_cache=section_cache,
            pec_area_out=pec_areas,
            pec_area_geom_out=pec_areas_geom,
            pec_area_jump_out=pec_areas_jump,
            material_fraction_mids=fraction_mids,
            material_fractions_out=face_fracs,
            domain_bounds=domain_bounds,
            scale=scale,
        )
        sigma_m_vals = None
        if with_sigma_m:
            sigma_m_vals = compute_face_material_areas(
                shapes_with_material,
                material_library,
                face_specs,
                face_axes,
                prop="sigma_m",
                deflection=deflection,
                section_cache=section_cache,
                domain_bounds=domain_bounds,
                scale=scale,
            )

        flat_arr = flat_all[main]
        valid = ~np.isnan(mu_vals)
        result[flat_arr[valid]] = mu_vals[valid]
        # Always populate pec_fraction for boundary faces, even when
        # the caller does not need it — staircase fallback edges keep
        # their NaN.
        safe_areas = np.where(face_areas > 0, face_areas, 1.0)
        pec_frac_local = pec_areas / safe_areas
        pec_frac[flat_arr[valid]] = pec_frac_local[valid]
        pec_frac_geom[flat_arr[valid]] = (pec_areas_geom / safe_areas)[valid]
        pec_frac_jump[flat_arr[valid]] = (pec_areas_jump / safe_areas)[valid]
        if fractions_result is not None:
            fractions_result[:, flat_arr[valid]] = face_fracs[:, valid]
        if sigma_m_result is not None:
            valid_sm = ~np.isnan(sigma_m_vals)
            sigma_m_result[flat_arr[valid_sm]] = sigma_m_vals[valid_sm]

    # DD-099 geom-only batch: separate classifier CALL (batching these
    # faces with the contrast candidates changes the shifted
    # re-evaluation of degenerate planes for the old faces, see the
    # candidate-mask comment above), writing the geometric channels
    # exclusively.  The section cache IS shared: entries are keyed by
    # (axis, plane_pos, shape) and deterministic at fixed deflection,
    # so a hit returns exactly what a fresh cache would recompute —
    # only the per-call face grouping (which the separate call
    # preserves) affects results.
    if geom_only.any():
        g_specs = np.array(face_specs_list, dtype=np.float64)[geom_only]
        g_axes = np.array(face_axes_list, dtype=np.int32)[geom_only]
        g_areas = np.array(face_areas_list, dtype=np.float64)[geom_only]
        g_flat = flat_all[geom_only]
        g_pec = np.zeros(len(g_specs), dtype=np.float64)
        g_geom = np.zeros(len(g_specs), dtype=np.float64)
        g_jump = np.zeros(len(g_specs), dtype=np.float64)
        g_mu = compute_face_material_areas(
            shapes_with_material,
            material_library,
            g_specs,
            g_axes,
            prop="mu",
            deflection=deflection,
            section_cache=section_cache,
            pec_area_out=g_pec,
            pec_area_geom_out=g_geom,
            pec_area_jump_out=g_jump,
            domain_bounds=domain_bounds,
            scale=scale,
        )
        g_valid = ~np.isnan(g_mu)
        g_safe = np.where(g_areas > 0, g_areas, 1.0)
        pec_frac_geom[g_flat[g_valid]] = (g_geom / g_safe)[g_valid]
        pec_frac_jump[g_flat[g_valid]] = (g_jump / g_safe)[g_valid]

    return result, pec_frac, pec_frac_geom, pec_frac_jump, fractions_result, sigma_m_result
