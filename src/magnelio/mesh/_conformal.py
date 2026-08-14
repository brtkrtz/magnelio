"""
PEC surface extraction, boundary cell detection, and thin-metallization
auto-detection helpers.

The per-edge sub-cell material data structures (``EdgeMaterialData``,
``FaceMaterialData``) live in :mod:`magnelio.geo._subcell` since
DD-051; this module no longer carries the legacy ``ConformalData`` and
``DeyMittraData`` classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.mesh.grid import GridLines


@dataclass
class PECSurfaceData:
    """Pre-computed PEC surface info for J_s = n x H post-processing.

    Used for surface current computation and quality-factor estimation:
    ``P_loss = (R_s / 2) * sum(|n x H|^2 * A_surface)``.

    Attributes
    ----------
    face_indices : np.ndarray
        Flat H-face indices on PEC surface, shape ``(n_surf,)``, dtype int.
    face_components : np.ndarray
        0 = Hx, 1 = Hy, 2 = Hz per face, shape ``(n_surf,)``, dtype int.
    outward_normals : np.ndarray
        Unit normal pointing away from PEC body, shape ``(n_surf, 3)``.
    surface_areas : np.ndarray
        Effective PEC surface area per face [m^2], shape ``(n_surf,)``.
    """

    face_indices: np.ndarray
    face_components: np.ndarray
    outward_normals: np.ndarray
    surface_areas: np.ndarray


def detect_boundary_cells(material_id: np.ndarray) -> np.ndarray:
    """Identify cells at material boundaries.

    A cell is a boundary cell if any of its 6 face-neighbours has a
    different ``material_id``.

    Parameters
    ----------
    material_id : np.ndarray
        Shape ``(Nx, Ny, Nz)``, dtype int.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(Nx, Ny, Nz)`` — True at boundaries.
    """
    boundary = np.zeros_like(material_id, dtype=bool)
    # x-neighbours
    diff_x = material_id[:-1] != material_id[1:]
    boundary[:-1] |= diff_x
    boundary[1:] |= diff_x
    # y-neighbours
    diff_y = material_id[:, :-1] != material_id[:, 1:]
    boundary[:, :-1] |= diff_y
    boundary[:, 1:] |= diff_y
    # z-neighbours
    diff_z = material_id[:, :, :-1] != material_id[:, :, 1:]
    boundary[:, :, :-1] |= diff_z
    boundary[:, :, 1:] |= diff_z
    return boundary


def extract_pec_surface(
    grid: GridLines,
    material_id: np.ndarray,
    material_library: dict,
) -> PECSurfaceData:
    """Extract H-faces on PEC/non-PEC boundaries (staircase approximation).

    For each face between a PEC cell and a non-PEC cell, records the face
    index, component, outward normal (pointing away from PEC into the
    non-PEC region), and the staircase face area.

    Parameters
    ----------
    grid : GridLines
        Mesh grid.
    material_id : np.ndarray
        Shape ``(Nx, Ny, Nz)``.
    material_library : dict
        ``{int: Material}`` mapping.

    Returns
    -------
    PECSurfaceData
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz

    pec_cells = np.zeros((Nx, Ny, Nz), dtype=bool)
    for mid, mat in material_library.items():
        if mat.is_pec:
            pec_cells |= material_id == mid

    indices: list[int] = []
    components: list[int] = []
    normals: list[tuple[float, float, float]] = []
    areas: list[float] = []

    # --- Hx faces at x-boundaries: shape (Nx+1, Ny, Nz) ---
    # Interior face at x[i] (i = 1..Nx-1): between cells (i-1,j,k) and (i,j,k)
    for i in range(1, Nx):
        diff = pec_cells[i - 1] != pec_cells[i]  # (Ny, Nz)
        if not diff.any():
            continue
        js, ks = np.nonzero(diff)
        for j, k in zip(js, ks):
            flat = i * Ny * Nz + j * Nz + k
            indices.append(flat)
            components.append(0)  # Hx
            area = dy[j] * dz[k]
            areas.append(area)
            # Normal points from PEC toward non-PEC
            if pec_cells[i - 1, j, k]:
                normals.append((1.0, 0.0, 0.0))  # PEC on left → normal +x
            else:
                normals.append((-1.0, 0.0, 0.0))  # PEC on right → normal -x

    # --- Hy faces at y-boundaries: shape (Nx, Ny+1, Nz) ---
    for j in range(1, Ny):
        diff = pec_cells[:, j - 1, :] != pec_cells[:, j, :]  # (Nx, Nz)
        if not diff.any():
            continue
        i_s, ks = np.nonzero(diff)
        for i, k in zip(i_s, ks):
            flat = i * (Ny + 1) * Nz + j * Nz + k
            indices.append(flat)
            components.append(1)  # Hy
            area = dx[i] * dz[k]
            areas.append(area)
            if pec_cells[i, j - 1, k]:
                normals.append((0.0, 1.0, 0.0))
            else:
                normals.append((0.0, -1.0, 0.0))

    # --- Hz faces at z-boundaries: shape (Nx, Ny, Nz+1) ---
    for k in range(1, Nz):
        diff = pec_cells[:, :, k - 1] != pec_cells[:, :, k]  # (Nx, Ny)
        if not diff.any():
            continue
        i_s, js = np.nonzero(diff)
        for i, j in zip(i_s, js):
            flat = i * Ny * (Nz + 1) + j * (Nz + 1) + k
            indices.append(flat)
            components.append(2)  # Hz
            area = dx[i] * dy[j]
            areas.append(area)
            if pec_cells[i, j, k - 1]:
                normals.append((0.0, 0.0, 1.0))
            else:
                normals.append((0.0, 0.0, -1.0))

    if not indices:
        return PECSurfaceData(
            face_indices=np.empty(0, dtype=int),
            face_components=np.empty(0, dtype=int),
            outward_normals=np.empty((0, 3), dtype=float),
            surface_areas=np.empty(0, dtype=float),
        )

    return PECSurfaceData(
        face_indices=np.array(indices, dtype=int),
        face_components=np.array(components, dtype=int),
        outward_normals=np.array(normals, dtype=float),
        surface_areas=np.array(areas, dtype=float),
    )


# ---------------------------------------------------------------------------
# Thin metallization auto-detection
# ---------------------------------------------------------------------------


@dataclass
class ThinSheetSpec:
    """Specification for an automatically detected thin PEC sheet."""

    axis: str  # sheet normal: 'x', 'y', or 'z'
    position: float  # substrate-side face = the ONE grid plane [m]
    rect: tuple[float, float, float, float] | None  # transverse extent, or None = full domain
    far_position: float | None = None  # far-side face, dropped from the critical planes
    shape: object | None = None  # source shape (stays in the sub-cell classifier list)


def _probe_eps(
    shapes: list,
    skip_shape,
    point: tuple[float, float, float],
    background,
    scale: float = 1.0,
    tolerance: float = 1e-7,
) -> float:
    """Effective permittivity of the material at *point*.

    Walks the shape list in reverse priority order (last shape wins,
    matching the cell-filling semantics) and classifies *point* against
    each candidate solid.  PEC neighbours and a PEC background count as
    eps = 1 — the substrate-side choice only needs a dielectric
    ordering, not a physical value for conductors.
    """
    from magnelio.geo._occ_backend import point_in_shape  # noqa: PLC0415

    for shape in reversed(shapes):
        if shape is skip_shape:
            continue
        try:
            (bmin, bmax) = shape.bounding_box(scale)
        except Exception:
            continue
        pad = 1e-12 * (1.0 + max(abs(c) for c in point))
        if not all(bmin[d] - pad <= point[d] <= bmax[d] + pad for d in range(3)):
            continue
        try:
            if point_in_shape(shape._occ_shape(scale), point, tolerance=tolerance, scale=scale):
                mat = shape.material
                return 1.0 if mat.is_pec else float(max(mat.epsilon))
        except Exception:
            continue
    if background is not None and not background.is_pec:
        return float(max(background.epsilon))
    return 1.0


def detect_thin_metallizations(
    shapes: list,
    min_cell_size: float,
    background=None,
    scale: float = 1.0,
) -> list[ThinSheetSpec]:
    """Identify PEC shapes thinner than the hard cell-size floor.

    Runs *before* grid-line generation (WP-M2): a PEC shape whose
    bounding box is thinner than ``min_cell_size`` along exactly one
    axis — and at least ``min_cell_size`` wide along the other two —
    is modelled as a thin sheet.  It gets ONE grid plane at its
    substrate-side face (carrying the tangential-E mask via
    ``apply_thin_pec_sheet``), while the metal volume stays in the
    DD-051 sub-cell classification of the adjacent cells, so the
    thickness effect enters through the conformal material matrices
    instead of a resolved cell layer.

    The substrate side is the face whose adjacent material has the
    higher permittivity, probed at the transverse centre just outside
    each face (PEC neighbours and a PEC background count as eps = 1).
    Ties pick the lower-coordinate face.

    The pre-WP-M2 variant of this function ran *after* grid generation
    and compared against the local cell size of the very grid that had
    already resolved the layer (both faces were critical planes) — a
    chicken-and-egg dead path that could only fire for layers thinner
    than ``min_feature_gap``.

    Parameters
    ----------
    shapes : list
        Geometry shapes in priority order (each with ``.material``,
        ``.bounding_box()``, ``._occ_shape()``).
    min_cell_size : float
        The hard cell-size floor [m]; the thin/resolved threshold.
    background : Material or None
        The model background material (used when a probe point lies
        outside every shape).

    Returns
    -------
    list of ThinSheetSpec
        One spec per detected sheet.  Detected shapes are *not*
        removed from the shape list — the caller keeps them in the
        sub-cell classifier list and only drops their far-side face
        from the critical planes.
    """
    thin_sheets: list[ThinSheetSpec] = []

    for shape in shapes:
        if not shape.material.is_pec:
            continue

        try:
            (bb_min, bb_max) = shape.bounding_box(scale)
        except Exception:
            continue
        extents = [bb_max[d] - bb_min[d] for d in range(3)]

        thin_axes = [d for d in range(3) if 0 < extents[d] < min_cell_size]
        if len(thin_axes) != 1:
            # Not a sheet: fully resolvable, or a sub-floor wire/point
            # (thin along 2+ axes) — the latter stays a normal shape
            # and is handled by the conformal sub-cell machinery.
            continue
        d = thin_axes[0]
        if any(extents[t] < min_cell_size for t in range(3) if t != d):
            continue  # degenerate transverse extent

        # Substrate side: probe the material just outside each face at
        # the transverse centre; the denser dielectric wins.
        centre = [0.5 * (bb_min[t] + bb_max[t]) for t in range(3)]
        delta = 0.5 * extents[d]
        p_lo = list(centre)
        p_lo[d] = bb_min[d] - delta
        p_hi = list(centre)
        p_hi[d] = bb_max[d] + delta
        # Cell-relative classification tolerance (DD-120): 1e-3 of the
        # hard floor reproduces the historical 1e-7 m at the typical
        # 100 um floor (this path only runs with min_cell_size set).
        probe_tol = 1e-3 * min_cell_size
        eps_lo = _probe_eps(
            shapes, shape, tuple(p_lo), background, scale=scale, tolerance=probe_tol
        )
        eps_hi = _probe_eps(
            shapes, shape, tuple(p_hi), background, scale=scale, tolerance=probe_tol
        )
        if eps_hi > eps_lo:
            position, far_position = bb_max[d], bb_min[d]
        else:
            position, far_position = bb_min[d], bb_max[d]

        axis = "xyz"[d]
        trans_axes = [a for a in "xyz" if a != axis]
        t0_idx = "xyz".index(trans_axes[0])
        t1_idx = "xyz".index(trans_axes[1])
        rect = (
            bb_min[t0_idx],
            bb_min[t1_idx],
            bb_max[t0_idx],
            bb_max[t1_idx],
        )

        thin_sheets.append(
            ThinSheetSpec(
                axis=axis,
                position=position,
                rect=rect,
                far_position=far_position,
                shape=shape,
            )
        )

    return thin_sheets


def rasterize_thin_sheet_footprint(mesh, spec: ThinSheetSpec, scale: float = 1.0) -> None:
    """Set tangential E edges on the sheet plane PEC — footprint-exact.

    The original WP-M2 rasterisation painted the spec's ``rect`` — the
    shape's *bounding box* — which is exact for a straight strip but
    silently shorts the whole box span for any non-rectangular layout
    (a ring resonator turned into a full metal plane).  Here the rect
    only pre-filters candidates; each candidate edge's midpoint is
    classified against the source shape's OCC solid, probed at the
    metal mid-thickness.  Edges on the lateral boundary count as metal
    (OCC ``ON`` state), matching the inclusive node selection of the
    rect path.

    Falls back to the rect fill when the spec carries no source shape
    or the OCC classification is unavailable.
    """
    from magnelio.geo._occ_backend import point_in_shape  # noqa: PLC0415
    from magnelio.mesh.indexing import apply_thin_pec_sheet  # noqa: PLC0415

    if spec.shape is None:
        apply_thin_pec_sheet(mesh, spec.axis, spec.position, spec.rect)
        return
    try:
        occ = spec.shape._occ_shape(scale)
    except Exception:
        apply_thin_pec_sheet(mesh, spec.axis, spec.position, spec.rect)
        return

    grid = mesh.grid
    Ny, Nz = grid.Ny, grid.Nz
    pec = mesh.pec_mask_edges

    if spec.far_position is not None:
        probe = 0.5 * (spec.position + spec.far_position)
        tol = 0.45 * abs(spec.far_position - spec.position)
    else:
        probe = spec.position
        tol = 1e-9 * (1.0 + abs(spec.position))

    nodes = {"x": grid.x, "y": grid.y, "z": grid.z}
    centres = {a: 0.5 * (nodes[a][:-1] + nodes[a][1:]) for a in "xyz"}
    h_idx = int(np.argmin(np.abs(nodes[spec.axis] - spec.position)))
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    if spec.rect is not None:
        u_min, v_min, u_max, v_max = spec.rect
    else:
        u_min, v_min = nodes[u_axis][0], nodes[v_axis][0]
        u_max, v_max = nodes[u_axis][-1], nodes[v_axis][-1]

    def _point(u, v):
        p = {spec.axis: probe, u_axis: u, v_axis: v}
        return (p["x"], p["y"], p["z"])

    def _flat(comp, i, j, k):
        if comp == "x":
            return (0, i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k)
        if comp == "y":
            return (1, i * Ny * (Nz + 1) + j * (Nz + 1) + k)
        return (2, i * (Ny + 1) * Nz + j * Nz + k)

    # The two tangential components: the u-directed edges run over
    # u-cells x v-nodes, the v-directed ones over u-nodes x v-cells.
    for comp, u_vals, v_vals in (
        (u_axis, centres[u_axis], nodes[v_axis]),
        (v_axis, nodes[u_axis], centres[v_axis]),
    ):
        u_sel = np.where((u_vals >= u_min) & (u_vals <= u_max))[0]
        v_sel = np.where((v_vals >= v_min) & (v_vals <= v_max))[0]
        for iu in u_sel:
            for iv in v_sel:
                if not point_in_shape(occ, _point(u_vals[iu], v_vals[iv]), tol, scale=scale):
                    continue
                idx = {spec.axis: h_idx, u_axis: int(iu), v_axis: int(iv)}
                comp_idx, flat = _flat(comp, idx["x"], idx["y"], idx["z"])
                pec[comp_idx, flat] = True
