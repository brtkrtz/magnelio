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
    classifiers=None,
) -> float:
    """Effective permittivity of the material at *point*.

    Walks the shape list in reverse priority order (last shape wins,
    matching the cell-filling semantics) and classifies *point* against
    each candidate solid.  PEC neighbours and a PEC background count as
    eps = 1 — the substrate-side choice only needs a dielectric
    ordering, not a physical value for conductors.  *classifiers* is a
    ``PointClassifierSet`` over *shapes* at *scale* and *tolerance*,
    shared between the probes of one detection pass; built here when
    not given.
    """
    from magnelio.geo._occ_backend import PointClassifierSet  # noqa: PLC0415

    if classifiers is None:
        classifiers = PointClassifierSet(shapes, scale=scale, tolerance=tolerance)
    hit = classifiers.first_containing(point, skip=skip_shape, reverse=True)
    if hit is not None:
        mat = shapes[hit].material
        return 1.0 if mat.is_pec else float(max(mat.epsilon))
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
    classifiers = None  # one loaded classifier per shape, shared by every probe

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
        if classifiers is None:
            from magnelio.geo._occ_backend import PointClassifierSet  # noqa: PLC0415

            classifiers = PointClassifierSet(shapes, scale=scale, tolerance=probe_tol)
        eps_lo = _probe_eps(
            shapes,
            shape,
            tuple(p_lo),
            background,
            scale=scale,
            tolerance=probe_tol,
            classifiers=classifiers,
        )
        eps_hi = _probe_eps(
            shapes,
            shape,
            tuple(p_hi),
            background,
            scale=scale,
            tolerance=probe_tol,
            classifiers=classifiers,
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


def _sheet_probe(spec: ThinSheetSpec) -> tuple[float, float]:
    """Probe position along the sheet normal and the membership tolerance.

    Mid-thickness when the far face is known, so the probe plane cuts
    the metal and never lies in a face; the tolerance is just under
    half the thickness, so a point laterally within it of the metal
    counts as metal — the inclusive boundary of the rect path.
    """
    if spec.far_position is not None:
        probe = 0.5 * (spec.position + spec.far_position)
        tol = 0.45 * abs(spec.far_position - spec.position)
    else:
        probe = spec.position
        tol = 1e-9 * (1.0 + abs(spec.position))
    return probe, tol


def _sheet_candidates(grid, spec: ThinSheetSpec) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The tangential E edges of the sheet plane inside the spec's rect.

    Returns one entry per tangential component ``(comp, iu_sel, iv_sel)``:
    the u-directed edges run over u-cells × v-nodes, the v-directed ones
    over u-nodes × v-cells; ``iu_sel``/``iv_sel`` index the candidate
    positions along the two transverse axes (inclusive rect test).
    """
    nodes = {"x": grid.x, "y": grid.y, "z": grid.z}
    centres = {a: 0.5 * (nodes[a][:-1] + nodes[a][1:]) for a in "xyz"}
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    if spec.rect is not None:
        u_min, v_min, u_max, v_max = spec.rect
    else:
        u_min, v_min = nodes[u_axis][0], nodes[v_axis][0]
        u_max, v_max = nodes[u_axis][-1], nodes[v_axis][-1]
    out = []
    for comp, u_vals, v_vals in (
        (u_axis, centres[u_axis], nodes[v_axis]),
        (v_axis, nodes[u_axis], centres[v_axis]),
    ):
        iu_sel = np.where((u_vals >= u_min) & (u_vals <= u_max))[0]
        iv_sel = np.where((v_vals >= v_min) & (v_vals <= v_max))[0]
        out.append((comp, iu_sel, iv_sel))
    return out


def _sheet_candidate_points(grid, spec: ThinSheetSpec, comp, iu_sel, iv_sel):
    """(UU, VV) coordinate grids of the candidate edge midpoints."""
    nodes = {"x": grid.x, "y": grid.y, "z": grid.z}
    centres = {a: 0.5 * (nodes[a][:-1] + nodes[a][1:]) for a in "xyz"}
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    u_vals = centres[u_axis] if comp == u_axis else nodes[u_axis]
    v_vals = nodes[v_axis] if comp == u_axis else centres[v_axis]
    return np.meshgrid(u_vals[iu_sel], v_vals[iv_sel], indexing="ij")


def _paint_sheet_edges(mesh, spec: ThinSheetSpec, comp, iu_sel, iv_sel, mask) -> None:
    """Set the masked candidate edges of component *comp* PEC (vectorised)."""
    grid = mesh.grid
    Ny, Nz = grid.Ny, grid.Nz
    nodes = {"x": grid.x, "y": grid.y, "z": grid.z}
    h_idx = int(np.argmin(np.abs(nodes[spec.axis] - spec.position)))
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    wu, wv = np.nonzero(mask)
    if wu.size == 0:
        return
    idx = {spec.axis: np.full(wu.size, h_idx), u_axis: iu_sel[wu], v_axis: iv_sel[wv]}
    i, j, k = idx["x"], idx["y"], idx["z"]
    if comp == "x":
        comp_idx, flat = 0, i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k
    elif comp == "y":
        comp_idx, flat = 1, i * Ny * (Nz + 1) + j * (Nz + 1) + k
    else:
        comp_idx, flat = 2, i * (Ny + 1) * Nz + j * Nz + k
    mesh.pec_mask_edges[comp_idx, flat] = True


def _rasterize_by_classifier(mesh, spec: ThinSheetSpec, scale: float = 1.0) -> None:
    """Footprint by solid classification of every candidate edge midpoint.

    The pre-section path, kept as the fallback for a sheet whose section
    fails: one ``BRepClass3d_SolidClassifier`` per sheet (its ``Load``
    is O(faces), so it is hoisted out of the loop) and one ``Perform``
    per candidate.  Still O(candidates × faces) — the reason the section
    path exists.
    """
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier  # noqa: PLC0415
    from OCC.Core.gp import gp_Pnt  # noqa: PLC0415
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON  # noqa: PLC0415

    from magnelio.geo._occ_backend import _scale3  # noqa: PLC0415

    occ = spec.shape._occ_shape(scale)
    probe, tol = _sheet_probe(spec)
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    classifier = BRepClass3d_SolidClassifier()
    classifier.Load(occ)
    for comp, iu_sel, iv_sel in _sheet_candidates(mesh.grid, spec):
        UU, VV = _sheet_candidate_points(mesh.grid, spec, comp, iu_sel, iv_sel)
        mask = np.zeros(UU.shape, dtype=bool)
        for a in range(UU.shape[0]):
            for b in range(UU.shape[1]):
                p = {spec.axis: probe, u_axis: UU[a, b], v_axis: VV[a, b]}
                classifier.Perform(gp_Pnt(*_scale3((p["x"], p["y"], p["z"]), scale)), tol * scale)
                mask[a, b] = classifier.State() in (TopAbs_IN, TopAbs_ON)
        _paint_sheet_edges(mesh, spec, comp, iu_sel, iv_sel, mask)


def rasterize_thin_sheet_footprint(mesh, spec: ThinSheetSpec, scale: float = 1.0) -> None:
    """Set tangential E edges on the sheet plane PEC — footprint-exact.

    The original WP-M2 rasterisation painted the spec's ``rect`` — the
    shape's *bounding box* — which is exact for a straight strip but
    silently shorts the whole box span for any non-rectangular layout
    (a ring resonator turned into a full metal plane).  Here the rect
    only pre-filters candidates; the footprint is **one section** of the
    source solid at the metal mid-thickness, and every candidate edge
    midpoint is tested against its contours by the even-odd rule (holes
    stay open) — the cell classifier's path, each contour on the points
    of its own bounding box (:func:`contour_mask`).  Edges on the
    lateral boundary count as metal (a band of the classification
    tolerance around the outline), matching the inclusive node
    selection of the rect path and the OCC ``ON`` state of the
    classifier.

    The section is asked of the shape's planar section engine first
    (:func:`magnelio.geo._occ_backend._section_engine`, the digest the
    ε/σ/µ passes built and cached on the shape): on a copper network
    of straight strips it answers in milliseconds where the kernel's
    section of the whole solid took a second (1.19 s for the 229
    contours of a 16 × 16 patch array's feed).  A plane the engine
    declines — through a vertex, tangent to a cylinder, a free-form
    face without facets — goes to ``cross_section_polygons`` as
    before.

    Classifying each midpoint against the solid instead costs
    O(candidates × faces): 15 ms per point on a 1 300-face copper
    network, minutes per sheet on a patch array.  That path stays as
    the fallback (``_rasterize_by_classifier``) for a section that
    fails or comes back empty.

    Falls back to the rect fill when the spec carries no source shape
    or the OCC evaluation is unavailable.
    """
    from magnelio.geo._filling import (  # noqa: PLC0415
        CLASSIFY_DEFLECTION_FRACTION,
        SECTION_NUDGE_FRACTION,
    )
    from magnelio.geo._occ_backend import cross_section_polygons  # noqa: PLC0415
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
    probe, tol = _sheet_probe(spec)
    nodes = {"x": grid.x, "y": grid.y, "z": grid.z}
    u_axis, v_axis = [a for a in "xyz" if a != spec.axis]
    h_min = min(float(np.diff(nodes[u_axis]).min()), float(np.diff(nodes[v_axis]).min()))
    # The chord budget of the cell classifier; the escape step must
    # stay inside the metal, so it is capped at half the tolerance.
    deflection = CLASSIFY_DEFLECTION_FRACTION * h_min
    nudge = SECTION_NUDGE_FRACTION * h_min
    if spec.far_position is not None:
        nudge = min(nudge, 0.5 * tol)
    contours = _sheet_section_by_engine(spec, probe, scale, deflection)
    if contours is None:
        try:
            contours = cross_section_polygons(
                occ,
                spec.axis,
                probe,
                deflection=deflection,
                scale=scale,
                exact_at_faces=spec.far_position is None,
                nudge=nudge,
                context=f"thin sheet {getattr(spec.shape, 'name', '') or ''}".rstrip(),
            )
        except Exception:  # noqa: BLE001 — the classifier path answers instead
            contours = []
    if not contours:
        _rasterize_by_classifier(mesh, spec, scale)
        return

    for comp, iu_sel, iv_sel in _sheet_candidates(grid, spec):
        UU, VV = _sheet_candidate_points(grid, spec, comp, iu_sel, iv_sel)
        mask = contour_mask(UU, VV, contours, tol)
        _paint_sheet_edges(mesh, spec, comp, iu_sel, iv_sel, mask)


def _sheet_section_by_engine(spec: ThinSheetSpec, probe: float, scale: float, deflection: float):
    """Contours of the sheet's solid at *probe* [m] from its section engine.

    The finest engine the material passes cached on the shape is
    reused — their chord budget is an order below this path's, and
    the digest's build is a third of a second on a 1 787-strip copper
    union, a millisecond on each of 192 finger bricks; only a shape
    no pass has sectioned gets an engine at *deflection*.  ``None``
    when the engine cannot digest the shape or declines the plane, so
    the caller takes the kernel section.
    """
    from magnelio.geo._occ_backend import _finest_section_engine, _section_engine  # noqa: PLC0415

    try:
        engine = _finest_section_engine(spec.shape, scale)
        if engine is None:
            engine = _section_engine(spec.shape, scale, deflection)
        if not engine.enabled:
            return None
        return engine.section("xyz".index(spec.axis), probe)
    except Exception:  # noqa: BLE001 — a shape the engine cannot digest: kernel section
        return None


def contour_mask(UU, VV, contours, tol: float) -> np.ndarray:
    """Even-odd membership of the grid points in *contours*, boundary band included.

    ``UU``/``VV`` are ``meshgrid(u_vals, v_vals, indexing="ij")`` of
    monotonic coordinate vectors.  Every contour is tested only on the
    block of points inside its bounding box padded by *tol*: outside
    it neither the even-odd test nor the band can be true, so the
    result equals the evaluation on the full grid — while a section of
    many small contours (the pads of an array) costs the sum of their
    boxes instead of contours × grid.
    """
    from magnelio.geo._polygon_clip import (  # noqa: PLC0415
        points_in_polygon,
        points_near_polygon_grid,
    )

    mask = np.zeros(UU.shape, dtype=bool)
    if UU.size == 0:
        return mask
    u_vals, v_vals = UU[:, 0], VV[0, :]
    windows = []
    for poly in contours:
        lo = poly.min(axis=0) - tol
        hi = poly.max(axis=0) + tol
        i0, i1 = np.searchsorted(u_vals, lo[0]), np.searchsorted(u_vals, hi[0], side="right")
        j0, j1 = np.searchsorted(v_vals, lo[1]), np.searchsorted(v_vals, hi[1], side="right")
        if i0 < i1 and j0 < j1:
            windows.append((poly, (slice(i0, i1), slice(j0, j1))))
    for poly, window in windows:
        mask[window] ^= points_in_polygon(UU[window], VV[window], poly)
    for poly, (wu, wv) in windows:
        mask[wu, wv] |= points_near_polygon_grid(u_vals[wu], v_vals[wv], poly, tol)
    return mask
