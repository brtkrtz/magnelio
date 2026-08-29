"""
Mesh: Top-level mesh object and MeshControl configuration.

Mesh.from_geometry() runs the two-scale grid-line generation algorithm
described in spec.md: cells start at ``h_fine`` next to material
interfaces and grow geometrically by ``MeshControl.growth_factor``
toward ``h_max`` in the bulk.
"""

# Design: DD-028 (two-scale grid-line generation; see design-decisions.md).

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from magnelio.mesh._quality import check_grading_undershoot, check_quality
from magnelio.mesh.grid import GridLines

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from magnelio.geo._subcell import EdgeMaterialData, FaceMaterialData
    from magnelio.materials.material import Material
    from magnelio.mesh._conformal import PECSurfaceData
    from magnelio.mesh._planes import GridPlanes
    from magnelio.mesh.faces import BoxFace


@dataclass
class MeshControl:
    """Parameters controlling grid-line generation.

    The mesher uses two cell-size scales per axis:

    * ``h_max`` (bulk) is wavelength-based,
      ``h_max = λ / min_nodes_per_wavelength``.  Cells in regions
      far from material interfaces are capped at ``h_max``.  With the
      default ``wavelength_rule="local"`` the wavelength is that of
      the densest material *in the slab* an axis interval spans, so
      air around a small dielectric is meshed at the air wavelength;
      ``"global"`` uses the densest material of the whole model
      everywhere.
    * ``h_fine`` (interface) is feature-based and **per axis**,
      ``h_fine = min_gap / min_cells_per_feature`` where
      ``min_gap`` is the smallest distance between adjacent material
      planes on *that* axis (axes without internal material
      boundaries stay at the wavelength size).  Cells adjacent to
      material interfaces use ``h_fine``.

    Between interfaces and the bulk, cells grow geometrically by
    ``growth_factor`` from ``h_fine`` toward ``h_max`` (graded spacing).
    Importantly, ``h_fine`` only applies *near* interfaces — it does
    *not* shrink the entire mesh down to feature size.

    Parameters
    ----------
    min_nodes_per_wavelength : int, default 20
        Cells per shortest wavelength in the densest material; sets ``h_max``.
    min_cells_per_feature : int, default 4
        Cells across the smallest geometry gap; sets ``h_fine``.  Set to
        0 to disable feature-based refinement (bulk-only meshing).
        Because cell counts are integers, the size actually generated
        may exceed ``h_fine`` by a few percent rather than add a
        cell; use ``min_cell_size`` for a hard floor.
    growth_factor : float, default 1.3
        Geometric ratio between adjacent cells in graded regions.
        Must be ``> 1.0``.  Smaller values yield smoother grading at
        the cost of more cells.
    max_cell_size : float or None, default None
        Hard upper bound on every cell, applied after ``h_max`` and
        ``h_fine`` are determined.  ``None`` disables the cap.
    min_cell_size : float or None, default None
        Hard lower bound on every cell.  ``None`` disables the floor.
    forced_planes : dict[str, list[float]], default empty
        Per-axis positions ``{"x": [...], "y": [...], "z": [...]}`` that
        the grid must include verbatim (e.g. probe points).
    conformal : bool, default True
        Enable conformal/Dey-Mittra material treatment for cells
        partially filled with PEC.
    dey_mittra_eta : float, default 0.4
        Stability cutoff for Dey-Mittra cells (fraction of full cell
        area below which cells are treated as PEC-only).
    min_feature_gap : float or None, default None
        Critical-plane clustering tolerance [m].  Adjacent critical
        planes closer than this are snapped to a single position before
        ``h_fine`` is computed.  Without this, sub-feature gaps from
        float coordinates (e.g. unaligned random geometry) collapse
        ``h_fine`` to absurdly small values and explode the mesh.  CAD
        geometries on a grid are unaffected.  ``None`` (default)
        resolves to ``1e-5 x`` the model bounding-box diagonal — the
        CSG float wiggle this tolerance absorbs is *relative*, so the
        default scales with the model, from meter-scale structures
        down to micron-scale optics.
    max_edge_refinement : float, default 4.0
        Geometry edges — the onset of a chamfer or fillet, a loft
        section, the equator or iris circle of a revolved profile —
        get a grid plane of their own wherever the edge lies flat in
        an axis-normal plane, so the feature occupies at least one
        cell layer and the cell's material average can see it.  A
        feature that varies *along* the grid edges inside one cell has
        no effect at all until it reaches the cell's midplane (a
        chamfer below half a cell height is invisible).  This ratio
        caps the refinement: an edge plane whose cell would be smaller
        than ``h_max / max_edge_refinement`` (or than
        ``min_cell_size``) is dropped with a warning naming it.  The
        time step follows the smallest cell, so the ratio also bounds
        the runtime cost of resolving small edges.  ``0`` disables
        edge planes altogether (material faces and silhouettes only).
    wavelength_rule : {"local", "global"}, default "local"
        Which wavelength sets the bulk cell size.  ``"local"``: each
        axis interval between grid planes is a slab of the domain, and
        the densest material whose bounding box reaches into that slab
        sets its bulk size — the air box around a small ceramic or a
        thin substrate is meshed at the air wavelength, the dielectric
        at its own.  ``"global"``: the densest material anywhere in the
        model sets one bulk size for the whole domain.  Feature
        refinement, grading and the edge floor are the same under both
        rules; the rules differ only far from material interfaces.
    singularity_refinement : float, default 1.0
        Refinement factor at conductor edges.  Where a metal body
        forms a wedge of less than 180° — the edges of a strip, a
        patch, an iris — the field and the surface current are
        singular (``r^(−1/3)`` at a 90° edge, ``r^(−1/2)`` at a knife
        edge) and the error of impedances and S-parameters converges
        only slowly with the cell size there.  The grid planes holding
        such an edge start their grading on both sides at
        ``h_fine / singularity_refinement`` instead of ``h_fine`` and
        grow by ``growth_factor`` from there.  Concave metal edges (the
        corners of a cavity), tangential edges (a fillet's onset) and
        dielectric edges are regular and not refined.  ``1`` disables
        the refinement.  The finer edge cells bound the time step:
        expect the run time to scale roughly with the factor.
    """

    # Design: WP-M4 (per-axis feature-based h_fine).

    min_nodes_per_wavelength: int = 20
    min_cells_per_feature: int = 4
    growth_factor: float = 1.3
    max_cell_size: float | None = None
    min_cell_size: float | None = None
    forced_planes: dict[str, list[float]] = field(default_factory=dict)
    conformal: bool = True
    dey_mittra_eta: float = 0.4
    min_feature_gap: float | None = None
    max_edge_refinement: float = 4.0
    wavelength_rule: str = "local"
    singularity_refinement: float = 1.0

    def __post_init__(self) -> None:
        if self.growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {self.growth_factor}")
        if not self.singularity_refinement >= 1.0:
            raise ValueError(
                f"singularity_refinement must be >= 1 (1 disables the refinement), "
                f"got {self.singularity_refinement}"
            )
        if self.wavelength_rule not in ("local", "global"):
            raise ValueError(
                f"wavelength_rule must be 'local' or 'global', got {self.wavelength_rule!r}"
            )
        if self.max_edge_refinement < 0.0:
            raise ValueError(
                f"max_edge_refinement must be >= 0 (0 disables edge planes), "
                f"got {self.max_edge_refinement}"
            )
        if self.min_nodes_per_wavelength < 2:
            raise ValueError(
                f"min_nodes_per_wavelength must be >= 2, got {self.min_nodes_per_wavelength}"
            )


def resolve_feature_gap(control: MeshControl, shapes) -> float:
    """Effective critical-plane clustering tolerance [m].

    ``control.min_feature_gap`` when the user set one; otherwise
    ``1e-5 x`` the analytic model bounding-box diagonal.  The CSG float
    wiggle the tolerance absorbs is relative to the coordinate
    magnitude, so the default must scale with the model.  Falls back to
    the historical 1e-6 m for an empty or degenerate shape set.
    """
    # Design: DD-120 (scale-relative default), DD-058 (CSG float wiggle).
    if control.min_feature_gap is not None:
        return control.min_feature_gap
    from magnelio.geo._scaling import analytic_bbox, box_diagonal, union_boxes  # noqa: PLC0415

    boxes = []
    for s in shapes:
        try:
            boxes.append(analytic_bbox(s))
        except ImportError:
            # Not an exotic shape but a missing pythonocc-core: raise the
            # backend's message instead of falling through to a default
            # gap and failing later on an empty grid (KB-024).
            raise
        except Exception:  # noqa: BLE001 — exotic shape
            continue
    if not boxes:
        return 1e-6
    diag = box_diagonal(union_boxes(boxes))
    if not math.isfinite(diag) or diag <= 0.0:
        return 1e-6
    return 1e-5 * diag


def _refractive_index(material) -> float | None:
    """``sqrt(max εr · max μr)`` of a propagating material; ``None`` for PEC / no material."""
    if material is None or material.is_pec:
        return None
    return math.sqrt(max(material.epsilon) * max(material.mu))


def _local_bulk_sizes(
    axis_planes: dict[str, list[float]],
    shapes,
    background,
    f_max: float,
    control: MeshControl,
    tol: float,
) -> dict[str, list[float]]:
    """Bulk cell size per axis interval (DD-192).

    ``wavelength_rule="global"``: every interval takes
    ``λ(n_max) / min_nodes_per_wavelength`` — the densest material
    anywhere sets one bulk size for the whole domain.

    ``wavelength_rule="local"``: an interval ``[p0, p1]`` on one axis is
    a slab of the domain.  The densest material whose analytic bounding
    box reaches into the slab by more than ``tol`` sets the slab's
    wavelength.  The background fills whatever no shape covers and so
    counts in every slab; a shape without an analytic bounding box
    counts everywhere (conservative — the slab is never meshed coarser
    than the material in it).  The bounding box is exact for bricks and
    conservative for curved or rotated bodies, which keeps the rule on
    the safe side there too.
    """
    from magnelio.constants import C0 as c0  # noqa: PLC0415
    from magnelio.geo._scaling import analytic_bbox  # noqa: PLC0415

    axes = ("x", "y", "z")
    n_floor = _refractive_index(background) or 1.0
    entries: list[tuple[float, tuple | None]] = []  # (n, bbox) — bbox None = everywhere
    for shape in shapes:
        n_shape = _refractive_index(shape.material)
        if n_shape is None or n_shape <= n_floor:
            continue
        try:
            box = analytic_bbox(shape)
        except Exception:  # noqa: BLE001 — exotic shape: counts everywhere
            box = None
        entries.append((n_shape, box))
    n_global = max([n_floor] + [n for n, _ in entries])

    def _h(n: float) -> float:
        return c0 / f_max / n / control.min_nodes_per_wavelength

    out: dict[str, list[float]] = {}
    for ax_i, axis in enumerate(axes):
        planes = axis_planes.get(axis, [])
        n_intervals = max(0, len(planes) - 1)
        if control.wavelength_rule == "global" or not entries:
            out[axis] = [_h(n_global)] * n_intervals
            continue
        sizes = []
        for k in range(n_intervals):
            p0, p1 = planes[k], planes[k + 1]
            n_slab = n_floor
            for n_shape, box in entries:
                if n_shape <= n_slab:
                    continue
                if box is None or (box[0][ax_i] < p1 - tol and box[1][ax_i] > p0 + tol):
                    n_slab = n_shape
            sizes.append(_h(n_slab))
        out[axis] = sizes
    return out


@dataclass
class Mesh:
    """Structured non-uniform hexahedral mesh with material assignments.

    See ``spec.md`` for the data-structure specification.
    """

    grid: GridLines
    material_id: np.ndarray  # shape (Nx, Ny, Nz), dtype int32
    material_library: dict[int, "Material"]
    pec_mask_edges: np.ndarray  # bool, shape (3, total_edges)
    edge_material: "EdgeMaterialData | None" = None
    face_material: "FaceMaterialData | None" = None
    pec_surface: "PECSurfaceData | None" = None
    # Design frequency (DD-186): the f_max ``from_geometry`` generated
    # this mesh for; ``None`` on the OCC-free ``from_grid`` path.  The
    # scattering analysis defaults its band to it and warns when asked
    # to exceed it.
    f_max: float | None = None
    # Boundary closure of the six bbox faces (DD-103).  Declared on the
    # model / grid the mesh was built from, carried here because the
    # mesh is what reaches the analysis, and read back by the analyses to
    # materialise the runtime BC objects.  A directly constructed Mesh
    # defaults to the all-PEC closure.
    boundary_conditions: object = None
    # Declarative ports (DD-109), declared on the GeometryModel before
    # meshing (or attached via with_ports).  Carried here because the
    # mesh is what reaches the analysis; AnalysisScatteringTD resolves
    # them when its own ports= is not given.  The mesher itself reads
    # only the port planes (per-face buffer cells).
    ports: tuple = ()
    # Passive lumped elements (DD-123), declared on the GeometryModel
    # via add_element.  Carried alongside the ports for the same
    # reason; the mesher itself never reads them.
    elements: tuple = ()
    # Field sources (DD-224), declared on the GeometryModel via
    # add_source and driven by an Excitation naming them.  Carried like
    # the ports and elements; the mesher itself never reads them.
    sources: tuple = ()
    # Provenance of every grid plane (DD-200): which rule and which
    # shape asked for it, which requested planes were dropped or
    # absorbed.  Set by ``from_geometry``; ``None`` on the ``from_grid``
    # path and on meshes loaded from stores written before this field.
    planes: "GridPlanes | None" = None
    # Pre-wall values of the edges the PEC closure overwrote, per face.
    # Lets a later closure change take a wall back off again (the OR
    # itself is lossy).  Not serialised: a reloaded mesh keeps the
    # closure it was stored with.
    _wall_backup: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            resolve_boundary_conditions,
        )

        self.boundary_conditions = resolve_boundary_conditions(
            self.boundary_conditions,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(
        cls,
        grid: GridLines,
        regions: list[tuple["Material", tuple[float, float, float, float, float, float]]]
        | None = None,
        background: "Material | None" = None,
        boundary_conditions=None,
    ) -> "Mesh":
        """Create a mesh from an explicit grid without requiring OCC.

        This is the OCC-free alternative to :meth:`from_geometry`. Material
        regions are specified as axis-aligned bounding boxes (AABB).

        Args:
            grid:       Pre-built :class:`~magnelio.mesh.grid.GridLines`.
            regions:    List of ``(material, (xmin, ymin, zmin, xmax, ymax, zmax))``
                        tuples. Regions are applied in order; later entries overwrite
                        earlier ones where they overlap.
            background: Material to fill all cells not covered by any region.
                        Defaults to ``Material.air()``.
            boundary_conditions: Closure of the six bbox faces
                        (:class:`~magnelio.boundaries.boundary_conditions.BoundaryConditions`
                        or dict); this is the OCC-free counterpart of
                        declaring it on the ``GeometryModel``.
                        Faces closed with PEC get their tangential
                        edges masked here, so the mode solvers and the
                        FIT update see the wall.  ``None`` closes every
                        face with PEC.  The grid is taken as given —
                        unlike :meth:`from_geometry`, this path cannot
                        extend it for CPML or pull a PMC line in.

        Returns:
            A fully populated :class:`Mesh`.

        Example::

            from magnelio.mesh.grid import GridLines
            from magnelio.mesh.mesher import Mesh
            from magnelio.materials.material import Material
            import numpy as np

            grid = GridLines(
                x=np.linspace(0, 10e-3, 11),
                y=np.linspace(0, 10e-3, 11),
                z=np.linspace(0, 10e-3, 11),
            )
            fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
            mesh = Mesh.from_grid(grid, regions=[(fr4, (0, 0, 0, 10e-3, 10e-3, 1.6e-3))])
        """
        from magnelio.materials.material import Material as Mat
        from magnelio.materials.material import resolve_material
        from magnelio.mesh.indexing import build_pec_mask_faces

        background = resolve_material(background, "Mesh.from_grid(background=...)")
        if background is None:
            background = Mat.air()

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        # Build material library and id array
        mat_library: dict[int, Mat] = {0: background}
        material_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)

        if regions:
            # Assign unique IDs to distinct material objects
            mat_to_id: dict[int, int] = {id(background): 0}
            next_id = 1

            # Cell-center coordinates for AABB lookup
            xc = 0.5 * (grid.x[:-1] + grid.x[1:])  # shape (Nx,)
            yc = 0.5 * (grid.y[:-1] + grid.y[1:])  # shape (Ny,)
            zc = 0.5 * (grid.z[:-1] + grid.z[1:])  # shape (Nz,)

            for mat, bbox in regions:
                mat = resolve_material(mat, "Mesh.from_grid regions material")
                xmin, ymin, zmin, xmax, ymax, zmax = bbox

                # Get or assign material ID
                obj_id = id(mat)
                if obj_id not in mat_to_id:
                    mat_to_id[obj_id] = next_id
                    mat_library[next_id] = mat
                    next_id += 1
                mid = mat_to_id[obj_id]

                # Mark cells whose centers fall inside AABB
                ix = np.where((xc >= xmin) & (xc < xmax))[0]
                iy = np.where((yc >= ymin) & (yc < ymax))[0]
                iz = np.where((zc >= zmin) & (zc < zmax))[0]

                if ix.size and iy.size and iz.size:
                    ixx, iyy, izz = np.meshgrid(ix, iy, iz, indexing="ij")
                    material_id[ixx, iyy, izz] = mid

        # PEC mask
        pec_mask = build_pec_mask_faces(grid, material_id, mat_library)

        # Boundary closure (DD-103): PEC faces mask their tangential
        # edges, exactly as on the from_geometry path.
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            bc_type_entries,
            resolve_boundary_conditions,
        )

        boundary_conditions = resolve_boundary_conditions(boundary_conditions)
        wall_backup: dict = {}
        _or_in_bbox_pec_walls(
            pec_mask,
            grid.Nx,
            grid.Ny,
            grid.Nz,
            [f for f, t in bc_type_entries(boundary_conditions).items() if t == "PEC"],
            backup=wall_backup,
        )

        mesh = cls(
            grid=grid,
            material_id=material_id,
            material_library=mat_library,
            pec_mask_edges=pec_mask,
            boundary_conditions=boundary_conditions,
            _wall_backup=wall_backup,
        )
        check_quality(mesh)
        return mesh

    @classmethod
    def from_geometry(
        cls,
        geometry,
        control: MeshControl,
        f_max: float,
    ) -> "Mesh":
        """Generate a mesh from a geometry model.

        Args:
            geometry:  A :class:`~magnelio.geo.GeometryModel` (CSG tree root).
            control:   :class:`MeshControl` parameters.
            f_max:     Maximum simulation frequency [Hz]. Determines target
                       cell size and is recorded on the mesh as its design
                       frequency (``mesh.f_max``); the scattering analysis
                       defaults its band to it.

        The boundary closure is read off
        ``geometry.boundary_conditions``; every mesh-time consequence
        follows from that one declaration:

        * **CPML** faces extend the grid by
          ``boundary_conditions.cpml_thickness_cells`` uniform cells, so
          the absorber has room to grade its profile.
        * **PMC** faces pull their outermost grid line inside the
          geometry bbox.  The natural magnetic wall of the FIT
          operators sits half a boundary cell *outside* that line (see
          :mod:`magnelio.boundaries.pmc`); moving the line to one third
          of the boundary cell lands the wall exactly ON the declared
          face and makes PMC cut-offs converge at
          O(dx**2).  Without it the wall sits half a boundary cell
          outside the bbox — an O(dx) bias, not an inconsistency.
        * **PEC** faces mask their tangential edges, giving the closed
          conducting chamber the mode solvers and the FIT update see.
          A PEC *background* fills the volume outside every shape but
          does **not** by itself close a face: a face declared PMC or
          CPML stays open through it (this supersedes the historical
          blanket bbox-wall forcing, which turned every symmetry
          plane and absorber into an electric wall).

        Returns:
            A fully populated :class:`Mesh`.

        Note:
            This method requires ``pythonocc-core`` for geometry queries.
            Import of OCC backend is deferred to this call.
        """
        # Design: WP-U0 stage 2 (PMC grid-line pull-in lands the natural
        # magnetic wall on the declared face).
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            bc_type_entries,
            cpml_thickness_of,
            resolve_boundary_conditions,
            symmetry_entries,
        )
        from magnelio.geo._occ_backend import (
            extract_critical_planes_per_shape,
            extract_feature_planes_per_shape,
        )
        from magnelio.mesh.indexing import build_pec_mask_faces

        # Collect shapes early (GeometryModel or plain list) — the
        # thin-sheet detection must run BEFORE grid-line generation.
        shapes = list(geometry)
        bg_material = getattr(geometry, "background", None)

        # DD-200: plane provenance.  The model's shape order labels every
        # plane source; taken before the wire split so wires keep the
        # index they have in the model.
        from magnelio.mesh._planes import (  # noqa: PLC0415
            GridPlanes,
            PlaneSource,
            attribute_planes,
            shape_label,
            with_node_data,
        )

        _shape_index = {id(s): i for i, s in enumerate(shapes)}
        _shape_labels = {id(s): shape_label(i, s) for i, s in enumerate(shapes)}

        def _src(kind: str, shape=None, label: str = "") -> PlaneSource:
            if shape is None:
                return PlaneSource(kind, None, label)
            return PlaneSource(kind, _shape_index.get(id(shape)), _shape_labels.get(id(shape), ""))

        boundary_conditions = resolve_boundary_conditions(
            getattr(geometry, "boundary_conditions", None),
        )
        _bc_types = bc_type_entries(boundary_conditions)
        pml_faces = [f for f, t in _bc_types.items() if t == "CPML"]
        pmc_faces = [f for f, t in _bc_types.items() if t == "PMC"]
        pec_wall_faces = [f for f, t in _bc_types.items() if t == "PEC"]
        pml_thickness_cells = cpml_thickness_of(boundary_conditions)

        # Thin wires (DD-080) are sub-cell objects, not solids: split
        # them off so the filling / classification / overlap layers
        # never see one.  Their effect (PEC edge chain + the paired
        # Holland (m, 1/m) material correction) is applied after mesh
        # assembly below.
        from magnelio.geo.wire import ThinWire as _ThinWire  # noqa: PLC0415

        wires = [s for s in shapes if isinstance(s, _ThinWire)]
        shapes = [s for s in shapes if not isinstance(s, _ThinWire)]

        if not shapes and not wires:
            raise ValueError(
                "The geometry model contains no shapes — there is "
                "nothing to mesh.  Add the solids with "
                "GeometryModel.add(...) before calling "
                "Mesh.from_geometry (a model carries only its "
                "background, boundary conditions and ports)."
            )

        # DD-120: one power-of-two scale factor for every OCC operation
        # of this mesh build, chosen from the analytic (OCC-free) model
        # bounding box.  1.0 for mm/meter-scale models (bit-identical
        # legacy path); >> 1 for micron-scale models, which moves the
        # geometry into the OCC kernel's comfortable precision range.
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(shapes + wires)
        # DD-120: the clustering tolerance scales with the model too.
        feature_gap = resolve_feature_gap(control, shapes + wires)

        # A standalone sheet has zero volume — meshing it as a thin
        # sheet (DD-035) is not yet wired.  Reject it up front, before any
        # grid work fails on the zero-thickness bounding box.
        from magnelio.geo._sheet import Sheet as _Sheet  # noqa: PLC0415

        if any(isinstance(s, _Sheet) for s in shapes):
            raise NotImplementedError(
                "A standalone sheet (a Face, a covered Curve or a Surface) "
                "cannot be meshed yet: thin-sheet physics is deferred. Use "
                "it as a profile for extruded()/revolved()/swept(), grow it "
                "into a solid with thickened(), or model the sheet as a thin "
                "solid."
            )

        # Step 0: Thin PEC metallization detection (WP-M2).  Only
        # active when the user sets the hard min_cell_size floor — the
        # floor is the thin/resolved threshold; without it the local
        # cell size is not known before the grid exists (the pre-WP-M2
        # post-grid detection was a chicken-and-egg dead path).  A
        # detected sheet gets ONE grid plane at its substrate-side
        # face; the far-side face is dropped from the critical planes
        # and the metal volume stays in the DD-051 sub-cell
        # classification of the adjacent cells.
        _thin_sheets: list = []
        if control.min_cell_size is not None and shapes:
            from magnelio.mesh._conformal import detect_thin_metallizations  # noqa: PLC0415

            _thin_sheets = detect_thin_metallizations(
                shapes,
                control.min_cell_size,
                background=bg_material,
                scale=geo_scale,
            )
        # Two sheets at one nominal height — a brick and a track that
        # came back from a Boolean, say — differ by kernel float wiggle
        # (~1e-19 m).  Sheet planes are anchors that survive every
        # merge verbatim, so left alone the pair makes a sliver cell
        # that collapses dt; unify them within the feature gap first
        # (a user-forced plane wins, otherwise the lowest sheet).
        _unify_thin_sheet_positions(_thin_sheets, control.forced_planes, feature_gap)
        _thin_by_shape = {id(spec.shape): spec for spec in _thin_sheets}

        # Step 1: Extract critical planes from OCC geometry (per shape,
        # so a thin sheet contributes exactly ONE plane along its thin
        # axis), then build the unified per-axis plane list (WP-M1):
        # forced planes are verbatim anchors (user positions win
        # bit-exactly), critical planes within control.min_feature_gap
        # of an anchor snap onto it, and the remaining critical planes
        # cluster to midpoints.  Without the anchor snap, CSG float
        # wiggle next to a forced node produces ~1e-18 m sliver cells
        # (DD-058); without the clustering, sub-µm gaps from float
        # coordinates collapse h_fine to absurdly small values and
        # explode the mesh.
        # Planes are (position, exact) pairs: exact = read from an
        # analytic face, False = bounding-box extent.  The clustering
        # snaps mixed clusters onto the exact members (KB-013 — a
        # Boolean-inflated bbox extent must not drag the grid line off
        # the material face it duplicates).
        critical_raw: dict[str, list[tuple[float, bool]]] = {"x": [], "y": [], "z": []}
        # DD-200: the raw (position, source) record kept next to every
        # raw plane list and attributed to the merged outcome below.
        plane_sources: dict[str, list[tuple[float, PlaneSource]]] = {"x": [], "y": [], "z": []}
        for shape, shape_planes in extract_critical_planes_per_shape(shapes, scale=geo_scale):
            spec = _thin_by_shape.get(id(shape))
            for axis in ("x", "y", "z"):
                if spec is not None and axis == spec.axis:
                    critical_raw[axis].append((spec.position, True))
                    plane_sources[axis].append((spec.position, _src("sheet", shape)))
                else:
                    critical_raw[axis].extend(shape_planes[axis])
                    plane_sources[axis].extend(
                        (p, _src("face" if exact else "extent", shape))
                        for p, exact in shape_planes[axis]
                    )
        # Step 1b (DD-191): geometry-edge planes — where a B-rep edge
        # lies flat in an axis-normal plane the body's cross-section
        # changes character along that axis (chamfer/fillet onset,
        # loft section, iris circle).  The face pass above cannot see
        # them (a chamfer is a cone), and a feature varying *along*
        # the grid edges inside one cell has no lever in the dual-face
        # material average until it crosses the cell's midplane.  They
        # are merged as a separate, soft class below: one cell per
        # feature, floored by max_edge_refinement, never outranking a
        # material plane.  A thin sheet's thin axis is exempt — the
        # sheet is ONE plane by construction.
        feature_raw: dict[str, list[float]] = {"x": [], "y": [], "z": []}
        if control.max_edge_refinement > 0 and shapes:
            for shape, shape_edges in extract_feature_planes_per_shape(shapes, scale=geo_scale):
                spec = _thin_by_shape.get(id(shape))
                for axis in ("x", "y", "z"):
                    if spec is not None and axis == spec.axis:
                        continue
                    feature_raw[axis].extend(shape_edges[axis])
                    plane_sources[axis].extend((p, _src("edge", shape)) for p in shape_edges[axis])
        # Step 1c (DD-194): conductor edges with a field singularity —
        # the sharp metal wedges of the model.  The planes holding one
        # are flagged after the merges below; their grading starts at
        # h_fine / singularity_refinement.  Positions only: an edge
        # never adds a plane of its own here (its plane is a material
        # face or a DD-191 edge plane already).
        singular_raw: dict[str, list[float]] = {"x": [], "y": [], "z": []}
        if control.singularity_refinement > 1.0 and shapes:
            from magnelio.geo._occ_backend import (  # noqa: PLC0415
                extract_singular_edge_planes,
            )

            singular_raw = extract_singular_edge_planes(shapes, bg_material, scale=geo_scale)
        # A thin wire contributes its curve's OCC vertex coordinates
        # (plus bbox extents) as material planes: every axis-aligned
        # polyline segment then lies exactly on grid lines — the wire's
        # transverse position is a vertex coordinate on both transverse
        # axes.  Arcs/splines/helices are covered by the rasteriser's
        # snap-displacement warning instead.
        if wires:
            from magnelio.geo._occ_backend import wire_vertex_points  # noqa: PLC0415

            for w in wires:
                (w_min, w_max) = w.bounding_box(geo_scale)
                w_pts = wire_vertex_points(w.curve._occ_shape(geo_scale), scale=geo_scale)
                for ax_i, axis in enumerate(("x", "y", "z")):
                    critical_raw[axis].extend((float(v), True) for v in w_pts[:, ax_i])
                    critical_raw[axis].extend(((w_min[ax_i], False), (w_max[ax_i], False)))
                    plane_sources[axis].extend((float(v), _src("wire", w)) for v in w_pts[:, ax_i])
                    plane_sources[axis].extend(
                        ((w_min[ax_i], _src("extent", w)), (w_max[ax_i], _src("extent", w)))
                    )
        # A source's total-field box (DD-224) wants its corners on grid
        # nodes, so the TF/SF split lands exactly where it was declared
        # instead of snapping to the nearest node; open sides (``None``
        # / ``±inf``) have no plane to ask for.
        for source in getattr(geometry, "sources", ()) or ():
            corners = getattr(source, "corners", None)
            if corners is None:
                continue
            for point in corners:
                for ax_i, axis in enumerate(("x", "y", "z")):
                    v = point[ax_i]
                    if v is None or not math.isfinite(float(v)):
                        continue
                    critical_raw[axis].append((float(v), True))
                    plane_sources[axis].append((float(v), _src("source", None, source.name)))

        # The far-side face of a thin sheet re-enters through the
        # *negative imprint* of the metal in the surrounding dielectric
        # (e.g. ``Difference(air, strip)`` contributes the cavity face
        # at the same position) — drop it globally.  A plane within
        # min_feature_gap of the far face would have clustered with it
        # anyway; after the drop it is represented by the sheet plane,
        # an offset below the floor scale that the conformal sub-cell
        # machinery absorbs.
        for spec in _thin_sheets:
            tol_drop = feature_gap

            def _not_far(p, _far=spec.far_position, _pos=spec.position, _tol=tol_drop):
                return abs(p - _far) > _tol or abs(p - _pos) <= _tol

            critical_raw[spec.axis] = [
                (p, exact) for (p, exact) in critical_raw[spec.axis] if _not_far(p)
            ]
            critical_raw[spec.axis].append((spec.position, True))
            plane_sources[spec.axis] = [
                (p, src) for (p, src) in plane_sources[spec.axis] if _not_far(p)
            ]
            plane_sources[spec.axis].append((spec.position, _src("sheet", spec.shape)))
            # The far face comes back through the imprint's *edges* too
            # (DD-191) — same global drop, or the edge floor would
            # report the sheet the mesher itself chose not to resolve.
            feature_raw[spec.axis] = [
                p for p in feature_raw[spec.axis] if abs(p - spec.far_position) > tol_drop
            ]
            # The sheet's knife edge is singular around the sheet
            # plane itself; the far face has no plane to flag.
            singular_raw[spec.axis] = [
                p for p in singular_raw[spec.axis] if abs(p - spec.far_position) > tol_drop
            ]
        # Anchors = user-forced planes + thin-sheet planes: both must
        # survive every merge stage verbatim (the sheet mask and any
        # user reference land on these exact nodes).
        axis_anchors: dict[str, list[float]] = {"x": [], "y": [], "z": []}
        for axis in ("x", "y", "z"):
            axis_anchors[axis] = list(control.forced_planes.get(axis, []))
            plane_sources[axis].extend((float(p), _src("forced")) for p in axis_anchors[axis])
        for spec in _thin_sheets:
            axis_anchors[spec.axis].append(spec.position)

        # Domain clip on symmetry faces (DD-154, vocabulary DD-159).
        # A symmetry face declared WITH a position ("SymmetryPEC"/"PMC",
        # default plane 0.0 or tuple form) clips the domain to the kept
        # half-space: the full geometry may be modelled and the mirror
        # half is simply never meshed.  Every critical plane on the
        # discarded side — including the clustering band around the
        # plane itself, so the position survives verbatim — is dropped,
        # and the symmetry plane enters as an exact face plane (it wins
        # the KB-013 clustering against bbox extents).  The wall
        # placement downstream needs no special case: a PMC symmetry
        # face is in pmc_faces and gets the step-2c pull-in, a PEC
        # symmetry face is in pec_wall_faces and gets its edge mask.
        # Without a position ("ForceSymmetry*") the declaration is
        # semantic only — the geometry already ends at the plane.
        for face, sym_pos in symmetry_entries(boundary_conditions).items():
            if sym_pos is None:
                continue
            axis, side = face[0], face[1:]
            if side == "min":

                def _kept(p, _pos=sym_pos, _gap=feature_gap):
                    return p > _pos + _gap
            else:

                def _kept(p, _pos=sym_pos, _gap=feature_gap):
                    return p < _pos - _gap

            critical_raw[axis] = [(p, e) for (p, e) in critical_raw[axis] if _kept(p)]
            critical_raw[axis].append((sym_pos, True))
            # A forced plane ON the symmetry plane stays an anchor
            # (only planes strictly beyond are dropped below).
            plane_sources[axis] = [
                (p, src)
                for (p, src) in plane_sources[axis]
                if _kept(p)
                or (src.kind == "forced" and (p >= sym_pos if side == "min" else p <= sym_pos))
            ]
            plane_sources[axis].append((sym_pos, _src("symmetry", label=face)))
            feature_raw[axis] = [p for p in feature_raw[axis] if _kept(p)]
            singular_raw[axis] = [p for p in singular_raw[axis] if _kept(p)]
            if side == "min":
                beyond = [p for p in axis_anchors[axis] if p < sym_pos]
            else:
                beyond = [p for p in axis_anchors[axis] if p > sym_pos]
            if beyond:
                warnings.warn(
                    f"symmetry face {face!r}: forced plane(s) at "
                    f"{sorted(beyond)!r} lie beyond the symmetry plane "
                    f"at {sym_pos!r} and are dropped with the clipped "
                    f"half-space.",
                    stacklevel=2,
                )
                axis_anchors[axis] = [p for p in axis_anchors[axis] if p not in beyond]

        from magnelio.constants import C0 as c0  # noqa: PLC0415

        # Determine max refractive index n = sqrt(εr·μr) over all shapes
        # and the background.  This sets the minimum wavelength:
        # λ_min = c₀ / (f_max · n_max).  PEC materials are excluded (they
        # have σ→∞, not a propagating medium).
        n_max = 1.0  # n = sqrt(εr·μr); floor = vacuum
        for mat in [s.material for s in shapes] + [bg_material]:
            n_mat = _refractive_index(mat)
            if n_mat is not None:
                n_max = max(n_max, n_mat)
        lambda_min = c0 / f_max / n_max
        h_wavelength = lambda_min / control.min_nodes_per_wavelength

        # Finest bulk cell size (the densest material's wavelength).
        # Under the local rule (DD-192) the per-interval bulk sizes are
        # derived below, once the grid planes are final; this global
        # value stays the reference for the edge floor, the feature
        # sentinel of h_fine and the undershoot check.
        h_max = h_wavelength

        # DD-191: the floor for edge planes.  h_max / ratio bounds the
        # time-step cost of resolving small edges; the hard floor
        # (when set) is a lower bound on every cell anyway.
        edge_floor = 0.0
        if control.max_edge_refinement > 0:
            edge_floor = h_max / control.max_edge_refinement
            if control.min_cell_size is not None:
                edge_floor = max(edge_floor, control.min_cell_size)

        axis_planes: dict[str, list[float]] = {}
        axis_is_material: dict[str, list[bool]] = {}
        axis_is_feature: dict[str, list[bool]] = {}
        _absorbed_planes: dict[str, list[float]] = {}
        _dropped_edges: dict[str, list[tuple[float, float]]] = {}
        for axis in ("x", "y", "z"):
            axis_planes[axis], axis_is_material[axis] = _merge_axis_planes(
                critical_raw.get(axis, []),
                axis_anchors[axis],
                feature_gap,
            )
            # DD-191: edge planes join as a soft class — dropped where
            # they duplicate a plane, where they would create a cell
            # below the edge floor, or where two of them crowd each
            # other (keep-first).  Every drop is reported: a feature
            # the grid cannot see must not vanish silently.
            (axis_planes[axis], axis_is_material[axis], axis_is_feature[axis], dropped) = (
                _merge_feature_planes(
                    axis_planes[axis],
                    axis_is_material[axis],
                    feature_raw[axis],
                    feature_gap,
                    edge_floor,
                )
            )
            if dropped:
                _dropped_edges[axis] = dropped
            # WP-M3 (a): the hard floor is a merge stage — no two
            # surviving planes closer than min_cell_size (anchor pairs
            # excepted, verbatim + warning).  Sub-floor PEC layers were
            # already converted to thin sheets in step 0; absorbed
            # dielectric boundaries are recorded so the classifier can
            # apply the longitudinal (series/harmonic) eps correction
            # on edges crossing them.
            if control.min_cell_size is not None:
                feature_set = {p for p, f in zip(axis_planes[axis], axis_is_feature[axis]) if f}
                (axis_planes[axis], axis_is_material[axis], absorbed) = _floor_merge_planes(
                    axis_planes[axis],
                    axis_is_material[axis],
                    axis_anchors[axis],
                    control.min_cell_size,
                )
                # Edge planes sit >= edge_floor >= min_cell_size from
                # every other plane, so the floor merge keeps them
                # verbatim; re-derive the flag on the survivors.
                axis_is_feature[axis] = [p in feature_set for p in axis_planes[axis]]
                if absorbed:
                    _absorbed_planes[axis] = absorbed
        if _dropped_edges:
            _warn_dropped_edge_planes(_dropped_edges, edge_floor, h_max, control)

        # DD-194: which of the final planes hold a singular conductor
        # edge.  Matched by position at the clustering tolerance (the
        # edge's plane is a material face or an edge plane that may
        # have been snapped); the domain's own end planes never — a
        # metal edge on a port face is the truncation, not geometry,
        # and the DD-107 buffer owns that interval.
        axis_is_singular: dict[str, list[bool]] = {}
        for axis in ("x", "y", "z"):
            planes = axis_planes[axis]
            marks = np.asarray(singular_raw[axis], dtype=float)
            axis_is_singular[axis] = [
                0 < i < len(planes) - 1
                and marks.size > 0
                and bool(np.min(np.abs(marks - p)) <= feature_gap)
                for i, p in enumerate(planes)
            ]

        # DD-200: attribute every raw source to its merged outcome.
        (_plane_recs, _plane_dropped, _plane_absorbed, _plane_unplaced) = attribute_planes(
            axis_planes,
            axis_is_singular,
            plane_sources,
            _dropped_edges,
            _absorbed_planes,
            feature_gap,
        )

        # DD-192: bulk cell size per axis interval.  An interval is a
        # slab of the domain; the densest material reaching into it
        # sets the wavelength the slab is meshed for.  ("global": every
        # interval takes h_wavelength.)
        h_max_axis = _local_bulk_sizes(
            axis_planes,
            shapes,
            bg_material,
            f_max,
            control,
            feature_gap,
        )

        # Step 2: Generate grid lines for each axis

        # Feature-based fine size for cells touching material interfaces
        # (DD-028): ensure the smallest geometry gap gets at least
        # min_cells_per_feature cells.  Only interior gaps count (need
        # >= 3 critical planes on an axis, i.e. at least one internal
        # material boundary).
        # Feature-based fine size PER AXIS (WP-M4): a small gap on one
        # axis must not refine the interface ramps of the other two.
        # (Pre-WP-M4 h_fine was the global minimum over all axes —
        # measured: one 0.635 mm substrate gap on z forced 0.32 mm
        # interface cells on x/y as well.)
        h_fine_axis = {"x": h_wavelength, "y": h_wavelength, "z": h_wavelength}
        if control.min_cells_per_feature > 0:
            for axis in ("x", "y", "z"):
                # Feature gaps are measured between *material* planes
                # only — forced-only planes (probe points etc.) are not
                # geometry features and must not drive h_fine.
                planes = [p for p, m in zip(axis_planes[axis], axis_is_material[axis]) if m]
                if len(planes) < 3:
                    continue  # no interior features on this axis
                min_gap = float("inf")
                for k in range(len(planes) - 1):
                    gap = planes[k + 1] - planes[k]
                    if gap > 0:
                        min_gap = min(min_gap, gap)
                if min_gap < float("inf"):
                    h_feature = min_gap / control.min_cells_per_feature
                    h_fine_axis[axis] = min(h_fine_axis[axis], h_feature)
        # DD-191: an edge plane asks for ONE cell across each interval it
        # bounds — enough for the cell's midplane to see the feature.
        # It enters the shared per-axis h_fine so the neighbouring
        # intervals ramp from that size (the generator grades from a
        # common h_fine, not from the adjacent interval's cell).
        for axis in ("x", "y", "z"):
            planes = axis_planes[axis]
            for k, is_feat in enumerate(axis_is_feature[axis]):
                if not is_feat:
                    continue
                for j in (k - 1, k + 1):
                    if 0 <= j < len(planes):
                        gap = abs(planes[j] - planes[k])
                        if gap > 0:
                            h_fine_axis[axis] = min(h_fine_axis[axis], gap)

        # DD-194: the fine size per plane — h_fine everywhere, divided
        # by the singularity factor on the planes holding a conductor
        # edge.  The generator grades each interval from its two ends'
        # own sizes.
        h_fine_planes: dict[str, list[float]] = {}
        for axis in ("x", "y", "z"):
            h_fine_planes[axis] = [
                h_fine_axis[axis] / control.singularity_refinement if s else h_fine_axis[axis]
                for s in axis_is_singular[axis]
            ]

        ports_declared = bool(getattr(geometry, "ports", ()))
        buffer_ends = _port_buffer_ends(getattr(geometry, "ports", ()))
        grid_lines = {}
        for axis in ("x", "y", "z"):
            # DD-191: a boundary interval bounded by an edge plane holds
            # one cell by design.  At a declared port face the DD-107
            # buffer still wins (the port needs its three equidistant
            # cells; the edge floor bounds them); at the port-blind
            # fallback faces the buffer is skipped there — tripling a
            # single-cell interval nobody asked to refine.
            feat = axis_is_feature[axis]
            end_floor: dict[str, float] = {}
            feature_end_floor = edge_floor if ports_declared else math.inf
            if len(feat) > 2 and feat[1]:
                end_floor["lo"] = feature_end_floor
            if len(feat) > 2 and feat[-2]:
                end_floor["hi"] = feature_end_floor
            grid_lines[axis] = _generate_axis_lines(
                axis_planes[axis],
                h_max=h_max_axis[axis],
                h_fine=h_fine_axis[axis],
                control=control,
                buffer_ends=buffer_ends[axis],
                end_floor=end_floor,
                h_fine_planes=h_fine_planes[axis],
            )

        # WP-M4 hard gate: a generated cell below min_feature_gap means
        # the plane clustering failed — the DD-058 silent-corruption
        # class (degenerate faces poison M_mu), which previously
        # sailed through with an aspect-ratio *warning*.  The only
        # legitimate sub-gap cells are user-anchor pairs (forced /
        # thin-sheet planes), which were kept verbatim with a warning.
        if feature_gap > 0:
            for axis in ("x", "y", "z"):
                arr = np.asarray(grid_lines[axis])
                d_arr = np.diff(arr)
                for idx in np.nonzero(d_arr < feature_gap)[0]:
                    lo_n, hi_n = float(arr[idx]), float(arr[idx + 1])
                    if lo_n in axis_anchors[axis] and hi_n in axis_anchors[axis]:
                        continue  # user-explicit pair (warned above)
                    raise RuntimeError(
                        f"mesher invariant violated: generated cell of "
                        f"size {d_arr[idx]:.3e} m on axis {axis!r} "
                        f"(nodes {lo_n!r}, {hi_n!r}) is below "
                        f"min_feature_gap = {feature_gap!r} "
                        f"and not a user-forced pair — plane clustering "
                        f"failed; please report this geometry."
                    )

        # The dt-setting cell may have come out finer than its interval
        # asked for (integer cell count per interval) — that costs time
        # steps and buys nothing.  Checked here, on the feature grid:
        # the PML extension below only appends coarse cells.  The
        # buffer info lets the check report DD-107 buffer cells that
        # bound dt on an otherwise wavelength-driven axis.
        check_grading_undershoot(
            grid_lines,
            axis_planes,
            axis_anchors,
            h_fine_axis,
            h_max,
            control,
            buffer_ends=buffer_ends,
            ports_declared=ports_declared,
            buffer_cells=_BOUNDARY_BUFFER_CELLS,
            h_fine_planes=h_fine_planes,
        )

        # DD-200: every plane is a node of its axis (the generators
        # clamp interval end points; the boundary buffer never moves a
        # plane).  Indices are taken before the absorber cells shift
        # them and before a PMC face pulls an end node inwards.
        _plane_nodes: dict[str, list[int]] = {}
        for axis in ("x", "y", "z"):
            arr = np.asarray(grid_lines[axis], dtype=float)
            span = float(arr[-1] - arr[0]) if arr.size > 1 else 1.0
            idx_list: list[int] = []
            for rec in _plane_recs[axis]:
                j = int(np.argmin(np.abs(arr - rec.position)))
                if abs(float(arr[j]) - rec.position) > 1e-9 * span:
                    raise RuntimeError(
                        f"mesher invariant violated: plane at {rec.position!r} on "
                        f"axis {axis!r} is not a grid node (nearest {float(arr[j])!r})."
                    )
                idx_list.append(j)
            _plane_nodes[axis] = idx_list
        _plane_moved: dict[str, dict[str, float]] = {}

        _face_axis = {
            "xmin": ("x", "min"),
            "xmax": ("x", "max"),
            "ymin": ("y", "min"),
            "ymax": ("y", "max"),
            "zmin": ("z", "min"),
            "zmax": ("z", "max"),
        }

        # Step 2b: Extend grid for PML faces
        _pml_cells: dict[str, int] = {}
        if pml_faces:
            # PML target depth = pml_thickness_cells × wavelength-based cell
            # size.  If the boundary cells are smaller (due to feature
            # resolution), use MORE cells at the boundary cell size to
            # maintain the same physical depth without a cell-size jump.
            for face in pml_faces:
                axis, side = _face_axis[face]
                # DD-192: the depth follows the bulk size of the
                # boundary slab the absorber continues.
                h_bulk_face = h_max_axis[axis][-1 if side == "max" else 0]
                d_pml_target = pml_thickness_cells * h_bulk_face
                nodes = grid_lines[axis]
                if side == "max":
                    h_bnd = nodes[-1] - nodes[-2]
                else:
                    h_bnd = nodes[1] - nodes[0]
                n_pml = max(
                    pml_thickness_cells,
                    math.ceil(d_pml_target / h_bnd),
                )
                if side == "max":
                    extension = [nodes[-1] + (i + 1) * h_bnd for i in range(n_pml)]
                    grid_lines[axis] = nodes + extension
                else:
                    extension = [nodes[0] - (n_pml - i) * h_bnd for i in range(n_pml)]
                    grid_lines[axis] = extension + nodes
                _pml_cells[face] = n_pml

        # Step 2c: Pull the outermost grid line inside the bbox on PMC
        # faces (WP-U0 stage 2).  The natural magnetic wall sits half a
        # boundary cell OUTSIDE the outermost primal line (full boundary
        # dual cell, see boundaries/pmc.py).  Moving the boundary node
        # to one third of the original boundary cell shrinks that cell
        # to 2d/3, and its outside half-cell (d/3) reaches exactly back
        # to the nominal bbox face: the wall lands ON the requested
        # geometry, leaving only the O(dx**2) dispersion error in PMC
        # cut-offs.  The 1.5 local cell-size ratio stays below the
        # quality-warning threshold.
        if pmc_faces:
            for face in pmc_faces:
                axis, side = _face_axis[face]
                nodes = grid_lines[axis]
                if side == "min":
                    moved = nodes[0] + (nodes[1] - nodes[0]) / 3.0
                    if nodes[0] in axis_anchors[axis]:
                        warnings.warn(
                            f"PMC face {face!r}: the user-forced plane at "
                            f"{nodes[0]!r} is the outermost grid line and "
                            f"is moved to {moved!r} to place the magnetic "
                            f"wall on the bbox face.",
                            stacklevel=2,
                        )
                    nodes[0] = moved
                    _plane_moved.setdefault(axis, {})["min"] = moved
                else:
                    moved = nodes[-1] - (nodes[-1] - nodes[-2]) / 3.0
                    if nodes[-1] in axis_anchors[axis]:
                        warnings.warn(
                            f"PMC face {face!r}: the user-forced plane at "
                            f"{nodes[-1]!r} is the outermost grid line and "
                            f"is moved to {moved!r} to place the magnetic "
                            f"wall on the bbox face.",
                            stacklevel=2,
                        )
                    nodes[-1] = moved
                    _plane_moved.setdefault(axis, {})["max"] = moved

        grid = GridLines(
            x=np.array(grid_lines["x"]),
            y=np.array(grid_lines["y"]),
            z=np.array(grid_lines["z"]),
        )

        # DD-200: freeze the provenance record with the grid-side data.
        def _frozen_axis(axis: str) -> tuple:
            recs = _plane_recs[axis]
            moved = _plane_moved.get(axis, {})
            offset = _pml_cells.get(f"{axis}min", 0)
            out = []
            for i, rec in enumerate(recs):
                moved_to = None
                if i == 0 and "min" in moved:
                    moved_to = moved["min"]
                elif i == len(recs) - 1 and "max" in moved:
                    moved_to = moved["max"]
                out.append(
                    with_node_data(
                        rec,
                        node=_plane_nodes[axis][i] + offset,
                        h_fine=float(h_fine_planes[axis][i]),
                        moved_to=moved_to,
                    )
                )
            return tuple(out)

        grid_planes = GridPlanes(
            x=_frozen_axis("x"),
            y=_frozen_axis("y"),
            z=_frozen_axis("z"),
            h_bulk={a: tuple(float(h) for h in h_max_axis[a]) for a in ("x", "y", "z")},
            dropped={a: tuple(_plane_dropped[a]) for a in ("x", "y", "z")},
            absorbed={a: tuple(_plane_absorbed[a]) for a in ("x", "y", "z")},
            unplaced={a: tuple(_plane_unplaced[a]) for a in ("x", "y", "z")},
            n_nodes={a: len(grid_lines[a]) for a in ("x", "y", "z")},
            pml_cells=dict(_pml_cells),
            feature_gap=feature_gap,
        )

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        # Step 3: Fill material IDs via cross-section polygons
        from magnelio.geo._filling import classify_cells_from_cross_sections
        from magnelio.geo._occ_backend import batch_cross_sections
        from magnelio.materials.material import Material as Mat

        bg = bg_material or Mat.air()
        material_library: dict[int, Mat] = {0: bg}
        material_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)

        # Overlap validation (GeometryModel only; plain lists are unchecked)
        if hasattr(geometry, "allow_overlaps") and not geometry.allow_overlaps and len(shapes) >= 2:
            geometry.validate()

        _cross_section_cache = None
        if shapes:
            # Assign stable IDs to each distinct material object.
            # ``shapes_with_material`` (cell-centre filling) excludes
            # thin sheets — their volume is thinner than any cell, and
            # a cell centre landing inside the metal would wrongly
            # resolve the sheet as a full PEC cell layer.
            # ``classifier_shapes_all`` (DD-051 sub-cell classification)
            # keeps them: the metal volume enters the conformal
            # material matrices through the effective PEC solid.
            mat_to_id: dict[int, int] = {id(bg): 0}
            next_id = 1

            shapes_with_material: list[tuple[object, int]] = []
            classifier_shapes_all: list[tuple[object, int]] = []
            for shape in shapes:
                mat = shape.material
                obj_id = id(mat)
                if obj_id not in mat_to_id:
                    mat_to_id[obj_id] = next_id
                    material_library[next_id] = mat
                    next_id += 1
                mid = mat_to_id[obj_id]
                classifier_shapes_all.append((shape, mid))
                if id(shape) not in _thin_by_shape:
                    shapes_with_material.append((shape, mid))

            # Cell-centre positions for cross-section planes.
            # classify_cells_from_cross_sections reads only ("x", i) keys
            # (one x-slice already fully classifies a cell via a 2D
            # point-in-polygon test over (y, z)) — so only x-plane
            # sections are computed here; y/z would be dead work.
            xc = 0.5 * (grid.x[:-1] + grid.x[1:])

            # Compute cross-sections at cell-centre planes.  Purely
            # h-relative (DD-120, the old 1e-4 m cap is gone).  The
            # chordal factor is deliberately coarser than the
            # conformal-area sites': cell-centre classification only
            # needs point-in-polygon fidelity to a fraction of a cell,
            # while the area integration feeds material matrices and
            # needs an order finer.  The degeneracy-escape step is
            # shared with those sites, so the two passes cannot end up
            # with different opinions about where the material is.
            from magnelio.geo._filling import (  # noqa: PLC0415
                CLASSIFY_DEFLECTION_FRACTION,
                SECTION_NUDGE_FRACTION,
            )

            h_min = min(grid.dx.min(), grid.dy.min(), grid.dz.min())
            deflection = h_min * CLASSIFY_DEFLECTION_FRACTION
            _cross_section_cache = batch_cross_sections(
                shapes_with_material,
                {"x": xc},
                deflection=deflection,
                scale=geo_scale,
                nudge=h_min * SECTION_NUDGE_FRACTION,
                material_library=material_library,
            )

            # Classify cells using x-plane cross-sections
            material_id = classify_cells_from_cross_sections(
                _cross_section_cache,
                grid,
                background_id=0,
            )

        # Step 3b: Extend material into PML cells.
        # The PML must preserve the waveguide cross-section so that modes
        # propagate smoothly into the absorbing region.  Each PML cell
        # inherits the material of the nearest interior cell on the same
        # transverse position.
        if _pml_cells:
            _axis_idx = {"x": 0, "y": 1, "z": 2}
            for face, n_pml in _pml_cells.items():
                ax, side = face[0], face[1:]
                ai = _axis_idx[ax]
                if side == "max":
                    src_slice = [slice(None)] * 3
                    src_slice[ai] = (
                        Nz - n_pml - 1
                        if ai == 2
                        else (Ny - n_pml - 1 if ai == 1 else Nx - n_pml - 1)
                    )
                    N_ax = [Nx, Ny, Nz][ai]
                    for offset in range(n_pml):
                        dst = [slice(None)] * 3
                        dst[ai] = N_ax - n_pml + offset
                        material_id[tuple(dst)] = material_id[tuple(src_slice)]
                else:  # min
                    src_slice = [slice(None)] * 3
                    src_slice[ai] = n_pml
                    for offset in range(n_pml):
                        dst = [slice(None)] * 3
                        dst[ai] = offset
                        material_id[tuple(dst)] = material_id[tuple(src_slice)]

        # Step 3c: Sub-cell classification + conformal mu (DD-051)
        edge_material_data = None
        face_material_data = None
        pec_surface_data = None
        pec_mask = None
        if control.conformal and _cross_section_cache is not None:
            from magnelio.geo._subcell import (  # noqa: PLC0415
                compute_subcell_data,
                compute_subcell_data_mu,
            )
            from magnelio.mesh._conformal import extract_pec_surface  # noqa: PLC0415

            # DD-049 fix: when the geometry's implicit background is PEC,
            # synthesize an explicit bbox-sized PEC brick at the lowest
            # priority before running the classifier.  Without this,
            # ``build_effective_pec_solid`` does not see the background-PEC
            # region and ``compute_edge_pec_fractions`` returns f_L = 1
            # ("entirely outside PEC") for edges fully embedded in the
            # background — which the un-mask threshold then wrongly admits,
            # leaving them with M_eps = 0 from the conformal pipeline that
            # correctly sees the dual face as fully PEC.
            # The classifier list includes thin sheets (WP-M2): their
            # metal volume enters the conformal matrices through the
            # effective PEC solid even though the cell-centre filling
            # never sees them.
            classifier_shapes = classifier_shapes_all
            if bg.is_pec and classifier_shapes_all:
                from magnelio.geo.primitives import Brick as _BboxBrick  # noqa: PLC0415

                _bg_bbox = _BboxBrick(
                    origin=(float(grid.x[0]), float(grid.y[0]), float(grid.z[0])),
                    size=(
                        float(grid.x[-1] - grid.x[0]),
                        float(grid.y[-1] - grid.y[0]),
                        float(grid.z[-1] - grid.z[0]),
                    ),
                    material=bg,
                )
                classifier_shapes = [(_bg_bbox, 0), *classifier_shapes_all]

            # Build effective PEC solid: shared by both the line-solid f_L
            # path and the thin-box A_free / ε̄ path inside compute_subcell_data
            # to close the consistency gap that two parallel pipelines had
            # under tessellation drift (DD-051).
            pec_solid = None
            if control.dey_mittra_eta > 0 and classifier_shapes:
                from magnelio.geo._occ_backend import (  # noqa: PLC0415
                    build_effective_pec_solid,
                )

                pec_solid = build_effective_pec_solid(
                    classifier_shapes,
                    material_library,
                    scale=geo_scale,
                )

            # Shared cross-section cache between the E-edge and H-face passes
            # — eps/sigma and mu reuse identical (axis, plane_pos, shape)
            # cache entries.
            section_cache: dict = {}

            # WP-M2: metal boxes of detected thin sheets seed the
            # classifier candidate gates (material_id cannot see
            # sub-cell-thin PEC volumes).
            _thin_boxes = None
            if _thin_sheets:
                _thin_boxes = []
                for spec in _thin_sheets:
                    (bb_min, bb_max) = spec.shape.bounding_box(geo_scale)
                    _thin_boxes.append((tuple(bb_min), tuple(bb_max)))

            # WP-C1 (DD-093): record per-material area fractions for
            # the dispersive/σ*-carrying ids only — the conformal ADE
            # and σ* builders consume them.  No such materials → no
            # container, no extra OCC work.
            _disp_mids_e = np.array(
                sorted(mid for mid, mat in material_library.items() if mat.dispersion is not None),
                dtype=np.int64,
            )
            _disp_mids_h = np.array(
                sorted(
                    mid
                    for mid, mat in material_library.items()
                    if mat.dispersion_mu is not None or any(s != 0.0 for s in mat.sigma_m)
                ),
                dtype=np.int64,
            )

            edge_material_data = compute_subcell_data(
                grid,
                material_id,
                material_library,
                classifier_shapes,
                pec_solid=pec_solid,
                eta=control.dey_mittra_eta,
                section_cache=section_cache,
                thin_sheet_boxes=_thin_boxes,
                absorbed_planes=_absorbed_planes or None,
                fraction_mids=_disp_mids_e if _disp_mids_e.size else None,
                scale=geo_scale,
            )
            face_material_data = compute_subcell_data_mu(
                grid,
                material_id,
                material_library,
                classifier_shapes,
                section_cache=section_cache,
                thin_sheet_boxes=_thin_boxes,
                fraction_mids=_disp_mids_h if _disp_mids_h.size else None,
                scale=geo_scale,
            )

            pec_surface_data = extract_pec_surface(
                grid,
                material_id,
                material_library,
            )

            # PEC mask from the sub-cell classifier (no post-hoc correction)
            pec_mask = edge_material_data.pec_mask

            # Step 3d (DD-198): the classifier saw the B-rep solids, which
            # end at the nominal bbox — inside the PML extension every
            # edge and face read as free space and a conductor touching
            # the absorbing wall lost its PEC mask there (KB-029).  Mirror
            # step 3b: the extension slabs take the first fully interior
            # slab's sub-cell data, the same translation-invariant
            # continuation the material ids already have.
            if _pml_cells:
                from magnelio.mesh._pml_extend import (  # noqa: PLC0415
                    extend_subcell_data_into_pml,
                )

                extend_subcell_data_into_pml(
                    pec_mask, edge_material_data, face_material_data, grid, _pml_cells
                )

        # Step 4: Build PEC masks (staircase fallback if no conformal)
        if pec_mask is None:
            pec_mask = build_pec_mask_faces(grid, material_id, material_library)

        # Step 4a: close the declared PEC faces (DD-103, supersedes the
        # background-driven rule of DD-049).  Force-masking the wall's
        # tangential E-edges is what makes a declared PEC face behave
        # like an explicit PEC bounding brick — in particular it keeps
        # the wall ONE connected component for the downstream
        # auto-conductor detection, which the staircase rule alone does
        # not: a dielectric touching the bbox at isolated tangent points
        # leaves those cell neighbours non-PEC and un-masks the edges
        # between them, fragmenting the wall.
        #
        # DD-049 keyed this on ``background.is_pec`` and then masked all
        # six faces.  That silently overrode every non-PEC closure a PEC
        # chamber might carry — a PMC symmetry plane became an electric
        # wall (breaking both the field symmetry and the TEM conductor
        # count on ports that touch it), and a CPML face became a mirror
        # in front of the absorber.  The closure now decides per face,
        # and the background is what it says it is: a volume filling.
        wall_backup: dict = {}
        _or_in_bbox_pec_walls(
            pec_mask,
            Nx,
            Ny,
            Nz,
            pec_wall_faces,
            backup=wall_backup,
        )

        mesh = cls(
            grid=grid,
            material_id=material_id,
            material_library=material_library,
            pec_mask_edges=pec_mask,
            edge_material=edge_material_data,
            face_material=face_material_data,
            pec_surface=pec_surface_data,
            boundary_conditions=boundary_conditions,
            f_max=f_max,
            ports=tuple(getattr(geometry, "ports", ()) or ()),
            elements=tuple(getattr(geometry, "elements", ()) or ()),
            sources=tuple(getattr(geometry, "sources", ()) or ()),
            planes=grid_planes,
            _wall_backup=wall_backup,
        )
        # Effective (resolved) DD-120 values, for diagnostics and the
        # stress-sentinel I3 invariant — the control object may carry
        # ``min_feature_gap=None``.
        mesh._resolved_feature_gap = feature_gap
        mesh._geo_scale = geo_scale

        # Step 4b: LC-consistent conformal M_mu coupling (DD-053).
        # Where the geometry is locally translation-invariant, the
        # transversal H-face mass is re-derived from the co-located
        # E-edge conformal capacitance through the pair identity, so
        # the exact discrete travelling wave (DD-052) survives on
        # conformal cross-sections.  Needs the assembled mesh (the
        # per-edge M_eps including enlarged-cell donors); mutates
        # mesh.face_material in place.
        # Thin wires, part 1 (DD-080): rasterise + PEC-mask the wire edge
        # chains BEFORE the DD-053 coupling pass, so no ladder is ever
        # certified *through* a wire edge.
        _wire_paths = []
        if wires:
            from magnelio.mesh._thin_wire import mask_thin_wires  # noqa: PLC0415

            _wire_paths = mask_thin_wires(mesh, wires, scale=geo_scale)

        if edge_material_data is not None and face_material_data is not None:
            from magnelio._operators.material_matrices import (  # noqa: PLC0415
                couple_face_material_pairs,
            )

            couple_face_material_pairs(mesh)
            # NOT wired here: the H-face enlarged-cell donor pass
            # (``assign_h_face_donors``, WP-R5).  The trigger benchmark
            # (``validation/iris_cavity_donor_trigger.py``)
            # measured the mechanism exactly neutral even at > 70 %
            # floor share on a deep PEC inclusion — floored cat-2 faces
            # are Faraday-dead (their circulation edges sit inside the
            # PEC mask), so the staircase fallback cannot inject error.
            # Wire the pass here (after the coupling pass) if a future
            # geometry meets the DD-051 trigger gate.

        # Thin wires, part 2 (DD-080): the paired Holland (m, 1/m)
        # correction — M_mu of the 4 encircling faces x m, M_eps of the
        # co-located radial edges x 1/m — AFTER the coupling pass, so a
        # conformal solid's (or DD-053 pair) value takes precedence.
        if wires:
            from magnelio.mesh._thin_wire import correct_thin_wire_materials  # noqa: PLC0415

            correct_thin_wire_materials(mesh, wires, _wire_paths)

        # Store PML cell counts (consumed by callers wiring CPML behind pml_faces)
        mesh._pml_cells = _pml_cells

        # Step 4b: Apply thin PEC sheets — footprint-exact (the bbox
        # rect would short the whole span for non-rectangular layouts).
        if _thin_sheets:
            from magnelio.mesh._conformal import rasterize_thin_sheet_footprint  # noqa: PLC0415

            for spec in _thin_sheets:
                rasterize_thin_sheet_footprint(mesh, spec, scale=geo_scale)

        # Step 4c: the wire and sheet masks were painted after step 3d,
        # so a thin conductor touching an absorbing face — a microstrip
        # reaching a port window in a CPML wall — had no mask in the
        # extension slabs and the port saw a hollow cross-section
        # (KB-034).  Mirror the interior slab once more, mask only: the
        # sub-cell material data was already continued in step 3d and
        # the sheet pass touches nothing but ``pec_mask_edges``.
        if _pml_cells and (_thin_sheets or wires):
            from magnelio.mesh._pml_extend import (  # noqa: PLC0415
                extend_subcell_data_into_pml,
            )

            extend_subcell_data_into_pml(mesh.pec_mask_edges, None, None, grid, _pml_cells)

        # Step 5: Quality checks
        check_quality(mesh)

        # Step 5b: DD-099 unregistered-wall warning (WP-B1.3), gated
        # on the scene actually declaring a lossy wall conductor
        # (DD-158): the registration itself always runs, but for
        # all-lossless scenes there is no loss surface to lose and the
        # warning would be noise.  Declared sources visible at mesh
        # time: lossy-metal materials and PECBoundary declarations
        # carrying their own wall material (dict form only — a
        # BoundaryConditions names types, not wall materials).  The
        # analysis-level ``wall_sigma`` fallback surfaces the same
        # warning later, at conductor-resolution time
        # (``resolve_wall_conductors``).
        _bc_raw = getattr(geometry, "boundary_conditions", None)
        _declares_conductors = any(
            getattr(_m, "is_lossy_metal", False) for _m in mesh.material_library.values()
        ) or (
            isinstance(_bc_raw, dict)
            and any(getattr(_v, "wall_sigma", None) is not None for _v in _bc_raw.values())
        )
        if _declares_conductors:
            from magnelio.mesh._surfaces import warn_unregistered_walls  # noqa: PLC0415

            warn_unregistered_walls(mesh, stacklevel=2)

        return mesh

    # ------------------------------------------------------------------
    # Boundary-condition consolidation
    # ------------------------------------------------------------------

    def with_pec_boundaries(
        self,
        faces: "Iterable[str]",
    ) -> "Mesh":
        """Consolidate PEC bbox boundary conditions into ``pec_mask_edges``.

        Returns a new :class:`Mesh` whose ``pec_mask_edges`` additionally
        marks every primal edge tangential to one of the listed bbox
        faces as PEC.  This is what makes a PEC *closure* and a
        geometric PEC block on that face indistinguishable to every
        downstream mesh consumer — the modal port factory,
        :func:`extract_conductor_groups_from_mesh`, the FIT solver's
        PEC-zeroing step, and any HDF5 export.

        Both mesh factories apply this to the PEC faces of their
        declared closure, so a mesh normally arrives already
        consolidated; the method remains for the component-level path that
        builds a :class:`Mesh` directly, and as the primitive both
        factories call.

        PMC, Periodic, and CPML faces are intentionally *not*
        consolidated:

        - **PMC** (``H_tan = 0``) translates to a Neumann condition for
          the modal Laplace / curl-curl problems, which is the natural
          (default) behaviour when no Dirichlet constraint is imposed —
          no mask change required.  Masking it instead would impose
          ``E_tan = 0``: the *opposite* symmetry.
        - **Periodic** ties opposite faces together; this is handled
          inside the FIT operator builders, not via the PEC mask.
        - **CPML** is an absorbing aux-variable layer; mode-solver
          semantics on a CPML lateral wall are an open architectural
          question deferred to a separate cleanup.

        Parameters
        ----------
        faces : iterable of str
            Bbox face names — any subset of ``{"xmin", "xmax", "ymin",
            "ymax", "zmin", "zmax"}``.  Duplicates and unknown names
            both raise.

        Returns
        -------
        Mesh
            New mesh sharing all data with ``self`` except for an
            updated ``pec_mask_edges``.  The original mesh is not
            mutated.
        """
        face_list = list(faces)
        new_mask = self.pec_mask_edges.copy()
        _or_in_bbox_pec_walls(new_mask, self.Nx, self.Ny, self.Nz, face_list)

        return Mesh(
            grid=self.grid,
            material_id=self.material_id,
            material_library=self.material_library,
            pec_mask_edges=new_mask,
            edge_material=self.edge_material,
            face_material=self.face_material,
            pec_surface=self.pec_surface,
            f_max=self.f_max,
            boundary_conditions=self.boundary_conditions,
            ports=self.ports,
            elements=self.elements,
            sources=self.sources,
            planes=self.planes,
        )

    def with_ports(self, ports) -> "Mesh":
        """Attach declarative ports to an already-built mesh.

        The late-declaration path for meshes not built through
        :meth:`from_geometry` (e.g. :meth:`from_grid`): the returned
        mesh carries *ports* exactly as if they had been declared on
        the :class:`~magnelio.geo.GeometryModel`, and
        ``AnalysisScatteringTD`` resolves them from the mesh.  The grid
        is taken as given — attaching ports here cannot regenerate the
        per-face buffer cells, so the port validator remains the
        backstop for faces that were never buffered.

        Parameters
        ----------
        ports : sequence of PortWaveguide / PortAnalytical
            Declarative ports with unique labels.

        Returns
        -------
        Mesh
            New mesh sharing all data with ``self`` plus the ports.
        """
        ports = tuple(ports)
        labels = [p.name for p in ports] + [e.name for e in self.elements]
        labels += [s.name for s in self.sources]
        if len(set(labels)) != len(labels):
            raise ValueError(f"port names must be unique; got {labels}")
        return dataclasses.replace(self, ports=ports)

    def with_elements(self, elements) -> "Mesh":
        """Attach passive lumped elements to an already-built mesh.

        The late-declaration path for meshes not built through
        :meth:`from_geometry`, mirroring :meth:`with_ports`:
        the returned mesh carries *elements* exactly as if they had
        been declared on the :class:`~magnelio.geo.GeometryModel` via
        ``add_element``, and ``AnalysisScatteringTD`` resolves them
        from the mesh.

        Parameters
        ----------
        elements : sequence of magnelio.circuit.LumpedElement
            Declarative elements; labels must be unique among the
            ports *and* elements of this mesh.

        Returns
        -------
        Mesh
            New mesh sharing all data with ``self`` plus the elements.
        """
        elements = tuple(elements)
        labels = [e.name for e in elements] + [p.name for p in self.ports]
        labels += [s.name for s in self.sources]
        if len(set(labels)) != len(labels):
            raise ValueError(f"element names must be unique; got {labels}")
        return dataclasses.replace(self, elements=elements)

    def with_sources(self, sources) -> "Mesh":
        """Attach field sources to an already-built mesh.

        The late-declaration path for meshes not built through
        :meth:`from_geometry`, mirroring :meth:`with_ports`: the
        returned mesh carries *sources* exactly as if they had been
        declared on the :class:`~magnelio.geo.GeometryModel` via
        ``add_source``, and an :class:`~magnelio.Excitation` drives
        them by name.

        Parameters
        ----------
        sources : sequence of magnelio.sources.Source
            Declarative sources; labels must be unique among the ports,
            elements *and* sources of this mesh.

        Returns
        -------
        Mesh
            New mesh sharing all data with ``self`` plus the sources.
        """
        sources = tuple(sources)
        labels = [s.name for s in sources] + [p.name for p in self.ports]
        labels += [e.name for e in self.elements]
        if len(set(labels)) != len(labels):
            raise ValueError(f"source names must be unique; got {labels}")
        return dataclasses.replace(self, sources=sources)

    def with_boundary_conditions(self, boundary_conditions) -> "Mesh":
        """Re-declare the boundary closure of an already-built mesh.

        Returns a new :class:`Mesh` carrying *boundary_conditions*, with
        the wall mask made to match: PEC faces get their tangential
        edges masked, and faces that are no longer PEC get the edges
        they had *before* any wall was forced on them — so this
        **replaces** the closure rather than adding to it, and a face
        can go from PEC back to PMC.

        The grid is taken as given: a closure attached here cannot grow
        a CPML extension or pull a PMC grid line in, so declare it on
        the :class:`~magnelio.geo.GeometryModel` when those matter.

        Notes
        -----
        Taking a wall back off needs the pre-wall edge values, which
        only exist for walls this process forced (``_wall_backup``).  A
        mesh reloaded from the project store has none, so on that path
        only re-declaring the *same* closure is meaningful — which is
        what the store does.
        """
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            bc_type_entries,
            resolve_boundary_conditions,
            symmetry_entries,
        )

        resolved = resolve_boundary_conditions(boundary_conditions)
        # A symmetry position clips the domain at mesh time (DD-154);
        # the grid is taken as given here, so only re-declaring the
        # position the mesh was built with is meaningful.
        old_sym = symmetry_entries(self.boundary_conditions)
        for f, pos in symmetry_entries(resolved).items():
            if pos is not None and old_sym.get(f) != pos:
                raise ValueError(
                    f"with_boundary_conditions cannot clip the domain: "
                    f"symmetry face {f!r} declares position {pos!r} but "
                    f"the mesh was built without it. Declare the "
                    f"symmetry on the GeometryModel and re-mesh.",
                )
        pec_faces = [f for f, t in bc_type_entries(resolved).items() if t == "PEC"]
        new_mask = self.pec_mask_edges.copy()
        drop = {f: v for f, v in self._wall_backup.items() if f not in pec_faces}
        if drop:
            _restore_bbox_pec_walls(new_mask, self.Nx, self.Ny, self.Nz, drop)
        backup = {f: v for f, v in self._wall_backup.items() if f in pec_faces}
        _or_in_bbox_pec_walls(
            new_mask,
            self.Nx,
            self.Ny,
            self.Nz,
            pec_faces,
            backup=backup,
        )
        return Mesh(
            grid=self.grid,
            material_id=self.material_id,
            material_library=self.material_library,
            pec_mask_edges=new_mask,
            edge_material=self.edge_material,
            face_material=self.face_material,
            pec_surface=self.pec_surface,
            f_max=self.f_max,
            boundary_conditions=resolved,
            ports=self.ports,
            elements=self.elements,
            sources=self.sources,
            planes=self.planes,
            _wall_backup=backup,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def Nx(self) -> int:
        return self.grid.Nx

    @property
    def Ny(self) -> int:
        return self.grid.Ny

    @property
    def Nz(self) -> int:
        return self.grid.Nz

    @property
    def pml_cells(self) -> dict[str, int]:
        """Number of absorber grid cells per domain face.

        Maps face names (``"xmin"`` … ``"zmax"``) to the number of grid
        cells the mesher appended *outside* the declared domain for that
        face's CPML layer.  Faces without an absorbing layer are absent.
        A mesh built without :meth:`from_geometry` reports an empty
        mapping.

        Returns
        -------
        dict of str to int
            Absorber cell count per face; a fresh copy on every access.
        """
        return dict(getattr(self, "_pml_cells", {}))

    def plot_section(self, normal: str, position: float, **kwargs):
        """Plot an axis-aligned section of the mesh: cells and grid lines by origin.

        Thin wrapper around :func:`magnelio.plots.plot_mesh_section`;
        see there for the keyword arguments — ``geometry=`` overlays
        the model's section outline, ``fill=`` picks the cell shading
        (PEC coverage by default, classification, or the permittivity
        the normal edges see), ``edges=True`` adds the PEC-masked and
        partially free edges.
        """
        from magnelio.post.plot_mesh import plot_mesh_section  # noqa: PLC0415

        return plot_mesh_section(self, normal, position, **kwargs)

    def __repr__(self) -> str:
        return f"Mesh(grid={self.grid!r}, n_materials={len(self.material_library)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _or_in_bbox_pec_walls(
    pec_mask: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    faces: "Iterable[str] | None" = None,
    backup: "dict | None" = None,
) -> None:
    """OR-in the bbox-face tangential E-edges into ``pec_mask`` in place.

    Mutates the input array.  Mirrors the per-face slicing used by
    :meth:`Mesh.with_pec_boundaries` so both code paths produce identical
    masks for the same face set.

    When *backup* is given, the pre-OR values of every wall edge this
    call touches are recorded into it (``{face: [array, ...]}``, first
    write per face wins).  That is what lets a later closure change
    *remove* a wall again — see
    :meth:`Mesh.with_boundary_conditions`.  Without it the OR is
    irreversible: the material information underneath is gone.
    """
    if faces is None:
        faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    n_per_axis = (
        Nx * (Ny + 1) * (Nz + 1),
        (Nx + 1) * Ny * (Nz + 1),
        (Nx + 1) * (Ny + 1) * Nz,
    )
    edge_shapes = (
        (Nx, Ny + 1, Nz + 1),
        (Nx + 1, Ny, Nz + 1),
        (Nx + 1, Ny + 1, Nz),
    )
    face_specs = {
        "xmin": [(1, np.s_[0, :, :]), (2, np.s_[0, :, :])],
        "xmax": [(1, np.s_[Nx, :, :]), (2, np.s_[Nx, :, :])],
        "ymin": [(0, np.s_[:, 0, :]), (2, np.s_[:, 0, :])],
        "ymax": [(0, np.s_[:, Ny, :]), (2, np.s_[:, Ny, :])],
        "zmin": [(0, np.s_[:, :, 0]), (1, np.s_[:, :, 0])],
        "zmax": [(0, np.s_[:, :, Nz]), (1, np.s_[:, :, Nz])],
    }
    faces = list(faces)
    for face in faces:
        if face not in face_specs:
            raise ValueError(
                f"unknown bbox face: {face!r}; valid choices are {sorted(face_specs)!r}."
            )
    # Snapshot BEFORE writing any wall: faces share edges along the bbox
    # corners, so a per-face save interleaved with the writes would
    # record a neighbouring wall's True instead of the material value.
    if backup is not None:
        for face in faces:
            if face in backup:
                continue
            backup[face] = [
                np.array(
                    pec_mask[axis, : n_per_axis[axis]].reshape(edge_shapes[axis])[sl],
                )
                for axis, sl in face_specs[face]
            ]
    for face in faces:
        for axis, sl in face_specs[face]:
            view = pec_mask[axis, : n_per_axis[axis]].reshape(edge_shapes[axis])
            view[sl] = True


def _restore_bbox_pec_walls(
    pec_mask: np.ndarray,
    Nx: int,
    Ny: int,
    Nz: int,
    backup: dict,
) -> None:
    """Undo :func:`_or_in_bbox_pec_walls` from its *backup*, in place."""
    n_per_axis = (
        Nx * (Ny + 1) * (Nz + 1),
        (Nx + 1) * Ny * (Nz + 1),
        (Nx + 1) * (Ny + 1) * Nz,
    )
    edge_shapes = (
        (Nx, Ny + 1, Nz + 1),
        (Nx + 1, Ny, Nz + 1),
        (Nx + 1, Ny + 1, Nz),
    )
    face_specs = {
        "xmin": [(1, np.s_[0, :, :]), (2, np.s_[0, :, :])],
        "xmax": [(1, np.s_[Nx, :, :]), (2, np.s_[Nx, :, :])],
        "ymin": [(0, np.s_[:, 0, :]), (2, np.s_[:, 0, :])],
        "ymax": [(0, np.s_[:, Ny, :]), (2, np.s_[:, Ny, :])],
        "zmin": [(0, np.s_[:, :, 0]), (1, np.s_[:, :, 0])],
        "zmax": [(0, np.s_[:, :, Nz]), (1, np.s_[:, :, Nz])],
    }
    for face, saved in backup.items():
        for (axis, sl), values in zip(face_specs[face], saved):
            view = pec_mask[axis, : n_per_axis[axis]].reshape(edge_shapes[axis])
            view[sl] = values


def _unify_thin_sheet_positions(specs: list, forced_planes: dict, tol: float) -> None:
    """Snap thin-sheet planes within *tol* of each other onto one position.

    Sheet planes are verbatim anchors of the plane merge, so two sheets
    whose substrate-side faces differ only by float wiggle would leave
    a sliver cell between two anchors.  Per axis, sheets are clustered
    by position (chain rule: each within *tol* of the previous one);
    a cluster takes the position of a user-forced plane within *tol*
    if there is one, otherwise its lowest member's.  The specs are
    updated in place — the sheet masks land on the shared node.
    """
    if tol <= 0:
        return
    for axis in ("x", "y", "z"):
        axis_specs = sorted((s for s in specs if s.axis == axis), key=lambda s: s.position)
        forced = [float(p) for p in forced_planes.get(axis, [])]
        clusters: list[list] = []
        for spec in axis_specs:
            if clusters and spec.position - clusters[-1][-1].position <= tol:
                clusters[-1].append(spec)
            else:
                clusters.append([spec])
        for cluster in clusters:
            rep = cluster[0].position
            for p in forced:
                if abs(p - rep) <= tol:
                    rep = p
                    break
            for spec in cluster:
                spec.position = rep


def _merge_axis_planes(
    critical: "Iterable[float | tuple[float, bool]]",
    forced: "Iterable[float]",
    tol: float,
) -> tuple[list[float], list[bool]]:
    """Cluster critical planes with forced planes as verbatim anchors.

    Forced (user-given) planes win bit-exactly: critical planes within
    ``tol`` of a forced anchor snap *onto* the anchor instead of
    surviving next to it.  Without the anchor snap, a CSG-returned
    tangent plane carrying ~1e-16 float wiggle next to a forced node
    produces a ~1e-18 m sliver cell whose degenerate faces poison
    M_mu (DD-058).  Critical planes away from every anchor cluster to
    midpoints as in :func:`_snap_planes`.  Forced-forced pairs closer
    than ``tol`` are respected verbatim with a loud warning — the user
    explicitly asked for both positions.

    Critical planes may carry a provenance flag ``(position, exact)``
    (bare floats count as exact): a cluster containing at least one
    exact face plane snaps to the exact members and ignores
    bounding-box extents for its position — see :func:`_snap_planes`
    (KB-013).

    Returns
    -------
    planes : list of float
        Strictly ascending merged plane positions.
    is_material : list of bool
        Parallel flags: True where the plane carries a material
        boundary (every critical-plane cluster; a forced anchor
        inherits the flag when a critical plane snapped onto it).
        Forced-only planes are False and must not drive feature-based
        refinement.
    """
    forced_sorted = sorted(set(forced))
    # Normalise to (position, exact) with an OR-merge on duplicate
    # positions — the same coordinate contributed as both a face plane
    # and a bbox extent is one exact plane.
    tagged: dict[float, bool] = {}
    for item in critical:
        p, exact = item if isinstance(item, tuple) else (item, True)
        tagged[p] = tagged.get(p, False) or exact
    crit_sorted = sorted(tagged.items())

    if not forced_sorted:
        snapped = _snap_planes(crit_sorted, tol)
        return snapped, [True] * len(snapped)

    if tol > 0:
        for a, b in zip(forced_sorted, forced_sorted[1:]):
            if b - a <= tol:
                warnings.warn(
                    f"forced planes at {a!r} and {b!r} are closer than "
                    f"min_feature_gap = {tol!r}; both are kept verbatim "
                    f"(user positions win) — expect a cell of size "
                    f"{b - a:.3e} m.",
                    stacklevel=3,
                )

    anchors = np.asarray(forced_sorted)
    anchor_is_material = [False] * len(forced_sorted)
    free_critical: list[tuple[float, bool]] = []
    for p, exact in crit_sorted:
        # Nearest anchor via bisection on the sorted anchor list.
        j = int(np.searchsorted(anchors, p))
        best_j = -1
        best_d = math.inf
        for cand in (j - 1, j):
            if 0 <= cand < len(anchors):
                d = abs(p - float(anchors[cand]))
                if d < best_d:
                    best_d = d
                    best_j = cand
        if best_j >= 0 and best_d <= tol:
            anchor_is_material[best_j] = True
        else:
            free_critical.append((p, exact))

    # Free critical planes are all farther than tol from every anchor,
    # so a cluster chain cannot cross an anchor (the crossing members
    # were snapped away) and every cluster midpoint stays > tol from
    # every anchor — no second pass needed.
    snapped = _snap_planes(free_critical, tol)

    merged = [(p, anchor_is_material[i]) for i, p in enumerate(forced_sorted)]
    merged += [(p, True) for p in snapped]
    merged.sort()
    planes = [p for p, _ in merged]
    flags = [m for _, m in merged]
    return planes, flags


def _floor_merge_planes(
    planes: list[float],
    is_material: list[bool],
    anchors: "Iterable[float]",
    floor: float,
) -> tuple[list[float], list[bool], list[float]]:
    """Enforce the hard ``min_cell_size`` floor on an axis plane list.

    No two surviving planes may be closer than ``floor`` — except
    anchor pairs (forced planes / thin-sheet planes), which are
    respected verbatim with a warning.  Non-anchor planes within
    ``floor`` of an anchor are dropped (the anchor wins); among the
    remaining non-anchor planes a keep-first scan drops every plane
    closer than ``floor`` to the previously kept one.  The survivor of
    a dropped run is a *real* material face (not a midpoint), so grid
    nodes stay on geometry boundaries.  The domain-end plane always
    survives (the computational domain must not shrink).

    Material flags are OR-merged onto the survivor: it now represents
    the absorbed boundaries for feature-gap purposes.

    Returns ``(kept, kept_material, absorbed)`` — ``absorbed`` lists
    the dropped *material* plane positions.  These are off-node
    dielectric boundaries now, and edges crossing them need the
    longitudinal (series/harmonic) eps correction: the DD-051
    dual-face average is transverse-only and cannot represent a
    series stack along the edge (measured session 92: ±2.6–3.7 %
    ε_eff error on a layered parallel plate without the correction).
    """
    if not planes or floor <= 0:
        return list(planes), list(is_material), []

    anchor_arr = np.asarray(sorted(set(anchors)), dtype=float)
    tol_eq = floor * 1e-9

    def _near_anchor(p: float) -> bool:
        if anchor_arr.size == 0:
            return False
        j = int(np.searchsorted(anchor_arr, p))
        for cand in (j - 1, j):
            if 0 <= cand < anchor_arr.size:
                if abs(p - float(anchor_arr[cand])) < floor - tol_eq:
                    return True
        return False

    def _is_anchor(p: float) -> bool:
        if anchor_arr.size == 0:
            return False
        j = int(np.searchsorted(anchor_arr, p))
        for cand in (j - 1, j):
            if 0 <= cand < anchor_arr.size:
                if p == float(anchor_arr[cand]):
                    return True
        return False

    kept: list[float] = []
    kept_material: list[bool] = []
    absorbed: list[float] = []
    for p, m in zip(planes, is_material):
        if _is_anchor(p):
            kept.append(p)
            kept_material.append(m)
            continue
        if _near_anchor(p):
            # Anchor wins; the anchor represents this boundary now.
            # (The anchor may come before OR after p in the scan, so
            # flag propagation to a preceding kept anchor is handled
            # here; a following anchor keeps its own flag — the
            # boundary it absorbs is within floor, below feature-gap
            # relevance at floor scale.)
            if kept and _is_anchor(kept[-1]) and p - kept[-1] < floor - tol_eq:
                kept_material[-1] = kept_material[-1] or m
            if m:
                absorbed.append(p)
            continue
        if kept and p - kept[-1] < floor - tol_eq:
            kept_material[-1] = kept_material[-1] or m
            if m:
                absorbed.append(p)
            continue
        kept.append(p)
        kept_material.append(m)

    # Domain-end rule: the last plane must survive.  If it was dropped,
    # remove kept non-anchor planes within floor below it and append it.
    if kept and kept[-1] != planes[-1]:
        p_end, m_end = planes[-1], is_material[-1]
        if m_end and p_end in absorbed:
            absorbed.remove(p_end)
        while kept and not _is_anchor(kept[-1]) and p_end - kept[-1] < floor - tol_eq:
            m_end = m_end or kept_material[-1]
            if kept_material[-1]:
                absorbed.append(kept[-1])
            kept.pop()
            kept_material.pop()
        if kept and _is_anchor(kept[-1]) and p_end - kept[-1] < floor - tol_eq:
            warnings.warn(
                f"domain-end plane at {p_end!r} is closer than "
                f"min_cell_size = {floor!r} to the anchor plane "
                f"{kept[-1]!r}; both are kept — expect a cell of size "
                f"{p_end - kept[-1]:.3e} m.",
                stacklevel=3,
            )
        kept.append(p_end)
        kept_material.append(m_end)

    for a, b in zip(kept, kept[1:]):
        if b - a < floor - tol_eq and _is_anchor(a) and _is_anchor(b):
            warnings.warn(
                f"anchor planes at {a!r} and {b!r} are closer than "
                f"min_cell_size = {floor!r}; both are kept verbatim — "
                f"expect a cell of size {b - a:.3e} m.",
                stacklevel=3,
            )

    return kept, kept_material, sorted(absorbed)


def _merge_feature_planes(
    planes: list[float],
    is_material: list[bool],
    feature: "Iterable[float]",
    tol: float,
    floor: float,
) -> tuple[list[float], list[bool], list[bool], list[tuple[float, float]]]:
    """Merge geometry-edge planes into an axis plane list (DD-191).

    ``planes`` / ``is_material`` are the merged material + forced
    planes of :func:`_merge_axis_planes`.  Edge planes are a *soft*
    class: they never move or outrank an existing plane.  Candidates
    within ``tol`` of each other cluster to their midpoint first
    (the same float-wiggle absorption as :func:`_snap_planes`); a
    candidate within ``tol`` of an existing plane is that plane and
    is silently absorbed; a candidate closer than ``floor`` to *any*
    existing plane, or to a previously kept edge plane (keep-first in
    ascending order), is dropped and reported.  Candidates outside
    the axis extent are dropped silently (the domain does not grow
    for an edge).

    Returns ``(planes, is_material, is_feature, dropped)`` with the
    three lists parallel and strictly ascending, and ``dropped`` a
    list of ``(position, gap)`` pairs — the cell size the dropped
    plane would have created — for the caller's warning.
    """
    cand = _snap_planes(sorted((float(p), True) for p in feature), tol)
    if not cand or not planes:
        return list(planes), list(is_material), [False] * len(planes), []
    fixed = np.asarray(planes, dtype=float)
    lo, hi = float(fixed[0]), float(fixed[-1])
    kept: list[float] = []
    dropped: list[tuple[float, float]] = []
    for p in cand:
        if p < lo or p > hi:
            continue
        j = int(np.searchsorted(fixed, p))
        d_fixed = min(abs(p - float(fixed[c])) for c in (j - 1, j) if 0 <= c < fixed.size)
        if d_fixed <= tol:
            continue  # duplicates a material / forced plane
        d_kept = p - kept[-1] if kept else math.inf
        gap = min(d_fixed, d_kept)
        if gap < floor:
            dropped.append((p, gap))
            continue
        kept.append(p)
    merged = [(p, m, False) for p, m in zip(planes, is_material)]
    merged += [(p, False, True) for p in kept]
    merged.sort()
    return (
        [p for p, _m, _f in merged],
        [m for _p, m, _f in merged],
        [f for _p, _m, f in merged],
        dropped,
    )


def _warn_dropped_edge_planes(
    dropped: dict[str, list[tuple[float, float]]],
    edge_floor: float,
    h_max: float,
    control: MeshControl,
) -> None:
    """One warning per mesh for the edge planes the floor removed (DD-191).

    Names the *coarsest* dropped plane — the one nearest to being
    resolved, and the one whose feature matters most — with the ratio
    that would keep it; the finer ones are counted.
    """
    axis, (pos, gap) = max(
        ((ax, item) for ax, items in dropped.items() for item in items),
        key=lambda t: t[1][1],
    )
    n = sum(len(v) for v in dropped.values())
    per_axis = ", ".join(f"{len(dropped[ax])} on {ax}" for ax in ("x", "y", "z") if ax in dropped)
    if control.min_cell_size is not None and control.min_cell_size >= edge_floor:
        binding = f"min_cell_size = {control.min_cell_size:.3g} m"
        remedy = "lower MeshControl(min_cell_size=...)"
    else:
        binding = f"h_max / max_edge_refinement = {h_max:.3g} m / {control.max_edge_refinement:g}"
        remedy = f"MeshControl(max_edge_refinement={math.ceil(h_max / gap * 10) / 10:g}) keeps it"
    warnings.warn(
        f"{n} geometry-edge plane{'s' if n != 1 else ''} ({per_axis}) below the "
        f"edge floor {edge_floor:.3g} m ({binding}) dropped.  The coarsest, at "
        f"{axis} = {pos:.6g} m, would create a {gap:.3g} m cell: the feature "
        f"there — a chamfer, fillet or section curve — is below the grid and "
        f"has no effect on the result until it spans half a cell; {remedy}, "
        f"or refine the mesh.",
        UserWarning,
        stacklevel=3,
    )


def _snap_planes(planes: list[tuple[float, bool]], tol: float) -> list[float]:
    """Cluster adjacent critical planes within ``tol`` to a single position.

    ``planes`` is ascending ``(position, exact)`` pairs — exact planes
    were read from an analytic face surface, non-exact ones are shape
    bounding-box extents.  The returned list is strictly sorted and
    every consecutive pair is separated by more than ``tol``.

    A cluster containing at least one exact plane collapses to the
    midpoint of its *exact* members only; bounding-box extents inside
    the cluster are absorbed without influencing the position.  OCCT
    Booleans on interpenetrating operands inflate the bounding box by
    ``Precision::Confusion`` beyond the true geometry, and averaging
    that phantom extent with the material face it duplicates would put
    the grid line tens of nanometres past the material surface —
    enough sliver fill to fail the DTBC slab certificate and silently
    drop every port channel on that face to Mur-1st (KB-013).
    Clusters without any exact member (silhouettes of tilted or
    free-form faces) keep the symmetric midpoint.

    This protects ``h_fine`` from float-noise gaps in user geometry —
    e.g. two random bricks with edges 1 µm apart should not force a
    0.25 µm grid spacing in the entire model.
    """
    if not planes or tol <= 0:
        return [p for p, _exact in planes]

    def collapse(members: list[tuple[float, bool]]) -> float:
        exact = [p for p, e in members if e]
        pool = exact if exact else [p for p, _e in members]
        return 0.5 * (pool[0] + pool[-1])

    out: list[float] = []
    cluster: list[tuple[float, bool]] = [planes[0]]
    for p, exact in planes[1:]:
        if p - cluster[-1][0] <= tol:
            cluster.append((p, exact))
        else:
            out.append(collapse(cluster))
            cluster = [(p, exact)]
    out.append(collapse(cluster))
    return out


def _port_buffer_ends(ports) -> dict[str, tuple[str, ...]]:
    """Axis ends that must carry the DD-107 equidistant-cell buffer.

    With declared ports (DD-109) only the faces that actually host one
    are buffered; without any declaration every domain face is (the
    port-blind fallback — ports may still arrive at analysis time).
    """
    all_ends = ("lo", "hi")
    if not ports:
        return {"x": all_ends, "y": all_ends, "z": all_ends}
    ends: dict[str, set] = {"x": set(), "y": set(), "z": set()}
    for port in ports:
        plane = getattr(port, "plane", None)
        if plane is None:
            continue  # interior port (e.g. lumped) — no face buffer
        face = _normalize_port_face(plane)
        ends["xyz"[face.normal_axis]].add("hi" if face.is_max else "lo")
    return {ax: tuple(sorted(v)) for ax, v in ends.items()}


def _normalize_port_face(plane) -> "BoxFace":
    """Accept a BoxFace or a face string ('zmin' / 'z_min') for a port plane."""
    from magnelio.mesh.faces import BoxFace  # noqa: PLC0415

    if isinstance(plane, BoxFace):
        return plane
    if isinstance(plane, str):
        key = plane.lower().replace("_", "")
        for face in BoxFace:
            if face.value.replace("_", "") == key:
                return face
    raise ValueError(
        f"unknown port plane {plane!r}; expected a BoxFace or one of "
        f"'xmin'/'xmax'/'ymin'/'ymax'/'zmin'/'zmax'"
    )


def _per_interval(h_max, n_intervals: int) -> list[float]:
    """Broadcast a scalar bulk size to one value per interval."""
    if isinstance(h_max, (int, float)):
        return [float(h_max)] * n_intervals
    values = [float(h) for h in h_max]
    if len(values) != n_intervals:
        raise ValueError(
            f"per-interval bulk sizes: expected {n_intervals} values, got {len(values)}"
        )
    return values


def _generate_axis_lines(
    critical_planes: list[float],
    h_max: "float | Sequence[float]",
    h_fine: float,
    control: MeshControl,
    buffer_ends: tuple[str, ...] = ("lo", "hi"),
    end_floor: dict[str, float] | None = None,
    h_fine_planes: "Sequence[float] | None" = None,
) -> list[float]:
    """Generate node positions for one axis from critical planes.

    Two-scale strategy: cells are ``h_fine`` at material interfaces (set
    by smallest geometry feature) and grow geometrically by
    ``control.growth_factor`` until they reach ``h_max`` (the bulk size,
    set by the wavelength criterion).  Beyond that, cells are uniform.
    ``h_max`` is one value for the whole axis or one per interval
    (DD-192: the slab's own wavelength).  ``h_fine_planes`` (DD-194)
    gives the fine size per plane — an interval grades from each end
    at that end's own size; ``None`` means ``h_fine`` at every plane.

    - Interior intervals (both endpoints are material interfaces): ramp
      from both ends, uniform middle when the interval is wide enough.
    - Boundary intervals (one endpoint is the domain wall): ramp from
      the interior interface, uniform fill toward the domain wall.
    - Single interval (whole domain, no interior interfaces): uniform
      with ``h_max``.

    ``control.max_cell_size`` and ``control.min_cell_size`` clamp both
    ``h_max`` and ``h_fine`` per interval.
    """
    if len(critical_planes) < 2:
        return critical_planes

    nodes: list[float] = [critical_planes[0]]
    n_intervals = len(critical_planes) - 1
    g = control.growth_factor
    min_cell = control.min_cell_size or 0.0
    h_max_list = _per_interval(h_max, n_intervals)
    fine_at = _fine_per_plane(h_fine, h_fine_planes, len(critical_planes))

    for i in range(n_intervals):
        p0 = critical_planes[i]
        p1 = critical_planes[i + 1]
        interval = p1 - p0

        if interval <= 0:
            continue

        # Per-interval h_max / h_fine, respecting MeshControl clamps.
        h_max_raw = min(h_max_list[i], interval)

        def _fine_eff(h: float, _h_max_raw=h_max_raw) -> float:
            h_eff = min(h, _h_max_raw)
            if control.max_cell_size is not None:
                h_eff = min(h_eff, control.max_cell_size)
            if min_cell > 0:
                h_eff = max(h_eff, min_cell)
            return h_eff

        h_max_eff = h_max_raw
        if control.max_cell_size is not None:
            h_max_eff = min(h_max_eff, control.max_cell_size)
        if min_cell > 0:
            h_max_eff = max(h_max_eff, min_cell)
        h_lo_eff = _fine_eff(fine_at[i])
        h_hi_eff = _fine_eff(fine_at[i + 1])

        is_first = i == 0
        is_last = i == n_intervals - 1

        if is_first and is_last:
            # No interior interface → pure uniform with h_max.
            n_sub = _n_uniform_floor(interval, h_max_eff, min_cell)
            sub_nodes = list(np.linspace(p0, p1, n_sub + 1))
        elif is_first:
            # p0 = domain wall, p1 = interior interface (fine end).
            sub_nodes = _grade_then_uniform(
                p_fine=p1,
                p_coarse=p0,
                h_fine=h_hi_eff,
                h_max=h_max_eff,
                g=g,
                min_cell=min_cell,
            )
        elif is_last:
            # p0 = interior interface (fine end), p1 = domain wall.
            sub_nodes = _grade_then_uniform(
                p_fine=p0,
                p_coarse=p1,
                h_fine=h_lo_eff,
                h_max=h_max_eff,
                g=g,
                min_cell=min_cell,
            )
        elif h_lo_eff == h_hi_eff:
            # Both ends are interior interfaces → symmetric ramps.
            sub_nodes = _grade_symmetric_to_uniform(
                p0=p0,
                p1=p1,
                h_fine=h_lo_eff,
                h_max=h_max_eff,
                g=g,
                min_cell=min_cell,
            )
        else:
            # Interior interval with a singular edge on one end only
            # (DD-194): ramps of different start sizes.
            sub_nodes = _grade_asymmetric_to_uniform(
                p0=p0,
                p1=p1,
                h_lo=h_lo_eff,
                h_hi=h_hi_eff,
                h_max=h_max_eff,
                g=g,
                min_cell=min_cell,
            )

        nodes.extend(sub_nodes[1:])  # skip first (already in nodes)

    return _enforce_boundary_buffer(
        nodes,
        critical_planes,
        h_max_list,
        h_fine,
        control,
        buffer_ends=buffer_ends,
        end_floor=end_floor,
        h_fine_planes=fine_at,
    )


def _fine_per_plane(
    h_fine: float, h_fine_planes: "Sequence[float] | None", n_planes: int
) -> list[float]:
    """Per-plane fine sizes: ``h_fine_planes`` verbatim, or ``h_fine``
    broadcast over the ``n_planes`` planes (DD-194)."""
    if h_fine_planes is None:
        return [float(h_fine)] * n_planes
    fine = [float(h) for h in h_fine_planes]
    if len(fine) != n_planes:
        raise ValueError(f"h_fine_planes has {len(fine)} entries for {n_planes} planes")
    return fine


def _axis_end_buffered(widths: "np.ndarray", end: str) -> bool:
    """Whether the axis already ends in >= _BOUNDARY_BUFFER_CELLS
    equidistant cells at the given end ('lo' or 'hi')."""
    n_buf = _BOUNDARY_BUFFER_CELLS
    if len(widths) < n_buf:
        return False
    tail = widths[-n_buf:] if end == "hi" else widths[:n_buf]
    lo = float(np.min(tail))
    hi = float(np.max(tail))
    return lo > 0 and (hi - lo) / lo <= 1e-9


def _enforce_boundary_buffer(
    nodes: list[float],
    critical_planes: list[float],
    h_max: "float | Sequence[float]",
    h_fine: float,
    control: MeshControl,
    buffer_ends: tuple[str, ...] = ("lo", "hi"),
    end_floor: dict[str, float] | None = None,
    h_fine_planes: "Sequence[float] | None" = None,
) -> list[float]:
    """DD-107 post-pass: guarantee the domain-face cell buffer.

    ``h_max`` is the axis bulk size or one value per interval; a
    regenerated boundary interval uses its own (DD-192), and grades
    from its interior plane's own fine size when ``h_fine_planes``
    is given (DD-194).

    ``end_floor`` maps an end (``"lo"`` / ``"hi"``) to an additional
    cell floor for its boundary interval — the DD-191 edge floor when
    that interval is bounded by a geometry-edge plane and a port is
    declared on the face (the buffer yields to it like to
    ``min_cell_size``), or ``inf`` to leave the interval alone (the
    port-blind fallback: a single-cell feature interval is not
    tripled for a port that does not exist).

    The buffer is a property of the axis' outermost CELLS, not of its
    outermost interval — a forced-planes grid can satisfy §2.4 across
    interval boundaries (uniform 1 mm raster), and rewriting its
    boundary interval would *destroy* the equidistance.  So: check the
    assembled axis first, and only where the buffer is violated
    regenerate the boundary interval with the buffer-enforcing
    profiles.  Critical planes never move.  If the regenerated
    interval still cannot host the buffer (interval too narrow, hard
    ``min_cell_size`` floor), the original grading is kept — the port
    validator reports the conflict if a port actually lands there.
    A single-cell axis is exempt (the deliberate degenerate-axis
    case, which can never host a waveguide port).
    """
    arr = np.asarray(nodes, dtype=np.float64)
    if len(arr) < 3 or len(critical_planes) < 2:
        return nodes
    widths = np.diff(arr)
    if len(widths) < 2:
        return nodes

    g = control.growth_factor
    min_cell = control.min_cell_size or 0.0
    n_buf = _BOUNDARY_BUFFER_CELLS
    h_max_list = _per_interval(h_max, len(critical_planes) - 1)
    fine_at = _fine_per_plane(h_fine, h_fine_planes, len(critical_planes))

    def _regen_interval(p0: float, p1: float, fine_at_hi: bool, min_cell: float):
        interval = p1 - p0
        # fine_at_hi marks the low boundary interval (wall at p0).
        h_max_eff = min(h_max_list[0] if fine_at_hi else h_max_list[-1], interval)
        h_fine_eff = min(fine_at[1] if fine_at_hi else fine_at[-2], h_max_eff)
        if control.max_cell_size is not None:
            h_max_eff = min(h_max_eff, control.max_cell_size)
            h_fine_eff = min(h_fine_eff, control.max_cell_size)
        if min_cell > 0:
            h_max_eff = max(h_max_eff, min_cell)
            h_fine_eff = max(h_fine_eff, min_cell)
        if len(critical_planes) == 2:
            # Whole-axis uniform: re-split at the buffer count when
            # the floor allows.
            if min_cell > 0 and interval / n_buf < min_cell * (1.0 - 1e-12):
                return None
            n_old = max(1, len(arr) - 1)
            n_sub = max(n_buf, n_old)
            return list(np.linspace(p0, p1, n_sub + 1))
        if fine_at_hi:
            return _grade_then_uniform(
                p_fine=p1,
                p_coarse=p0,
                h_fine=h_fine_eff,
                h_max=h_max_eff,
                g=g,
                min_cell=min_cell,
                boundary_buffer=True,
            )
        return _grade_then_uniform(
            p_fine=p0,
            p_coarse=p1,
            h_fine=h_fine_eff,
            h_max=h_max_eff,
            g=g,
            min_cell=min_cell,
            boundary_buffer=True,
        )

    result = arr
    for end in buffer_ends:
        widths = np.diff(result)
        if _axis_end_buffered(widths, end):
            continue
        min_cell_end = max(min_cell, (end_floor or {}).get(end, 0.0))
        if not math.isfinite(min_cell_end):
            continue
        if end == "lo":
            p0, p1 = critical_planes[0], critical_planes[1]
            sub = _regen_interval(p0, p1, fine_at_hi=True, min_cell=min_cell_end)
        else:
            p0, p1 = critical_planes[-2], critical_planes[-1]
            sub = _regen_interval(p0, p1, fine_at_hi=False, min_cell=min_cell_end)
        if sub is None:
            continue
        sub_w = np.diff(np.asarray(sub))
        wall_tail = sub_w[:n_buf] if end == "lo" else sub_w[-n_buf:]
        if len(sub_w) < n_buf or ((wall_tail.max() - wall_tail.min()) / wall_tail.min() > 1e-9):
            continue  # interval cannot host the buffer — keep as-is
        lo_mask = result >= p0 - 1e-15 * (1.0 + abs(p0))
        hi_mask = result <= p1 + 1e-15 * (1.0 + abs(p1))
        keep_before = result[~lo_mask]
        keep_after = result[~hi_mask]
        result = np.concatenate([keep_before, np.asarray(sub), keep_after])
    return list(result)


# Slack allowed when a cell count is chosen against h_fine (DD-105).
# h_fine is a convention -- min_gap / min_cells_per_feature -- not a
# physical constant, so overshooting it by a few percent is free.
# Refusing to is not: the count is an integer, so a miss by a fraction
# of a percent adds a cell and shrinks every cell in the interval,
# which costs time steps model-wide (the explicit loop takes one step,
# bounded by the smallest cell anywhere) and buys resolution nowhere.
# Never applied to h_max: that one is the user's wavelength criterion.
_H_FINE_TOL = 0.05

# Uniform-cell buffer adjacent to port-carrying domain faces (DD-107 /
# DD-109).  The modal-port operator is a difference operator between
# the port plane and the next interior plane; reference_waveguide_
# ports.md §2.4 requires the three cells adjacent to the port plane to
# be equidistant, or the V/I projection silently scales by orders of
# magnitude.  With ports declared on the GeometryModel before meshing
# (DD-109) the mesher buffers exactly the declared faces; a model
# without declarations falls back to buffering all six faces (DD-107 —
# ports may still arrive at analysis time).  Whether the boundary
# grading saturated before the domain wall used to decide this by
# accident (it did at 2 GHz on the slotline coupler and stopped doing
# so at 1.4 GHz).  The buffer is skipped only when the hard
# ``min_cell_size`` floor (WP-M3) makes three cells impossible; the
# port-side validator still guards that case.
_BOUNDARY_BUFFER_CELLS = 3


def _n_uniform_floor(interval: float, h: float, min_cell: float) -> int:
    """Uniform cell count for *interval*: target size ``h``, hard floor.

    ``ceil(interval / h)`` capped so that ``interval / n >= min_cell``
    (WP-M3 (b) — the uniform refit must not undershoot the floor).
    ``n = 1`` is always allowed: a sub-floor *interval* can only come
    from an anchor pair, which is respected verbatim by design.
    """
    n = max(1, math.ceil(interval / h * (1.0 - 1e-12)))
    if min_cell > 0:
        n_cap = max(1, int(interval / min_cell * (1.0 + 1e-12)))
        n = min(n, n_cap)
    return n


def _grade_then_uniform(
    p_fine: float,
    p_coarse: float,
    h_fine: float,
    h_max: float,
    g: float,
    min_cell: float = 0.0,
    boundary_buffer: bool = False,
) -> list[float]:
    """One-sided ramp from h_fine to ~h_max, then uniform fill.

    The ramp grows geometrically by ``g`` from ``h_fine`` adjacent to
    ``p_fine``.  The ramp's last cell and the uniform cells beyond are
    sized in a fix-point iteration so neighbouring cells differ by no
    more than the growth factor — no ramp-to-uniform back-step.
    ``min_cell`` is the hard cell-size floor (WP-M3): no generated
    cell may undershoot it.  Returned in ascending order.

    ``boundary_buffer`` (DD-107): the coarse end is a domain wall —
    end the profile in a uniform tail of ``_BOUNDARY_BUFFER_CELLS``
    cells (unless ``min_cell`` makes that impossible).  Off by
    default: the buffer is enforced per AXIS end, not per interval
    (``_enforce_boundary_buffer``), so the plain profile stays
    bit-identical wherever the assembled axis already satisfies it.
    """
    interval = abs(p_coarse - p_fine)
    lo, hi = min(p_fine, p_coarse), max(p_fine, p_coarse)
    if interval <= 0:
        return [lo, hi]

    h_fine = min(h_fine, h_max)
    n_buf = _BOUNDARY_BUFFER_CELLS

    # Trivial: h_fine ≈ h_max → single uniform run (DD-107: at least
    # the buffer count, unless the floor forbids).
    if h_fine >= h_max * (1.0 - 1e-10) or g <= 1.0 + 1e-10:
        n = _n_uniform_floor(interval, h_max, min_cell)
        if (
            boundary_buffer
            and n < n_buf
            and (min_cell <= 0 or interval / n_buf >= min_cell * (1.0 - 1e-12))
        ):
            n = n_buf
        return list(np.linspace(lo, hi, n + 1))

    def _build_ramp(target: float) -> list[float]:
        widths: list[float] = [h_fine]
        while widths[-1] < target * (1.0 - 1e-10):
            nxt = widths[-1] * g
            if nxt >= target:
                widths.append(target)
                break
            widths.append(nxt)
        return widths

    def _with_buffer_tail(
        ramp: list[float], rest: float, n_uniform: int, h_uniform: float
    ) -> list[float]:
        """Widths ``ramp + [h_uniform]*n_uniform``, upgraded to the
        DD-107 buffer: when fewer than ``n_buf`` uniform cells remain,
        ramp cells donate their length to a ``n_buf``-cell uniform
        tail.  The donor count is chosen to minimise the seam ratio
        against the surviving ramp cell (a fixed count is g-class for
        the default growth factor but overshoots for large ``g``).
        Falls back to the plain profile when the fine-end cell would
        be consumed or the floor would be undershot — the buffer is
        best-effort under WP-M3."""
        if n_uniform >= n_buf:
            return list(ramp) + [h_uniform] * n_uniform
        best = None
        for d in range(1, min(len(ramp) - 1, n_buf) + 1):
            h_u = (rest + sum(ramp[len(ramp) - d :])) / n_buf
            prev = ramp[len(ramp) - d - 1]
            seam = max(h_u / prev, prev / h_u)
            if best is None or seam < best[0]:
                best = (seam, d, h_u)
        if best is None or best[0] > g * (1.0 + 1e-9):
            # No donor split keeps the seam within the growth factor
            # (short ramp, h_fine ≈ h_max) — refit the whole interval.
            return _tailed_widths(interval, h_fine, g, min_cell, n_buf)
        _, d, h_u = best
        if min_cell > 0 and h_u < min_cell * (1.0 - 1e-12):
            # Donor tail undershoots the hard floor (WP-M3) — refit
            # the whole interval; _tailed_widths falls back to the
            # legacy profile when even that cannot satisfy the floor.
            return _tailed_widths(interval, h_fine, g, min_cell, n_buf)
        return list(ramp[: len(ramp) - d]) + [h_u] * n_buf

    # Two passes: first ramp to h_max, then if h_uniform is much smaller
    # than the ramp end, retarget the ramp to h_uniform·g so the
    # transition has neighbour ratio ≤ g.  A single retargeting suffices
    # in practice; iterating further would oscillate (smaller ramp →
    # bigger rest → more uniform cells → smaller h_uniform).
    ramp = _build_ramp(h_max)
    ramp_sum = sum(ramp)
    if ramp_sum >= interval:
        # Bulk too short for the full ramp — one-sided graded cells
        # over the whole interval, ending in the DD-107 buffer tail
        # when the coarse end is a domain wall.
        if boundary_buffer:
            widths = _tailed_widths(interval, h_fine, g, min_cell, n_buf)
            return _widths_to_nodes(widths, p_fine, p_coarse)
        n_legacy = _n_one_sided(interval, h_fine, g, min_cell=min_cell)
        # DD-193: keep the fine-end cell at h_fine and relax the ratio
        # instead of letting the integer count push h0 below h_fine.
        g_eff = g
        if _h0_one_sided(interval, n_legacy, g) < h_fine * (1.0 - 1e-9):
            g_eff = _ratio_for_exact_fill(interval, h_fine, n_legacy, g, symmetric=False)
        return _one_sided_subdivision(p_fine, p_coarse, n_legacy, g_eff)

    rest = interval - ramp_sum
    widths: list[float]
    if rest < 0.5 * h_max or rest < min_cell:
        # Tiny remainder — absorbed into the last ramp cell; DD-107
        # folds it into the buffer tail instead.
        widths = _with_buffer_tail(ramp, rest, 0, 0.0) if boundary_buffer else _absorb(ramp, rest)
    else:
        n_uniform = _n_uniform_floor(rest, h_max, min_cell)
        h_uniform = rest / n_uniform
        # If the ramp's last cell would overshoot h_uniform·g, retarget
        # the ramp to land at h_uniform·g instead of h_max.
        if ramp[-1] > h_uniform * g * (1.0 + 1e-10):
            new_target = h_uniform * g
            ramp = _build_ramp(new_target)
            rest = interval - sum(ramp)
            if rest < min_cell:
                widths = (
                    _with_buffer_tail(ramp, rest, 0, 0.0)
                    if boundary_buffer
                    else _absorb(ramp, rest)
                )
                return _widths_to_nodes(widths, p_fine, p_coarse)
            n_uniform = _n_uniform_floor(rest, h_max, min_cell)
            h_uniform = rest / n_uniform
        widths = (
            _with_buffer_tail(ramp, rest, n_uniform, h_uniform)
            if boundary_buffer
            else list(ramp) + [h_uniform] * n_uniform
        )

    return _widths_to_nodes(widths, p_fine, p_coarse)


def _absorb(ramp: list[float], rest: float) -> list[float]:
    """Tiny-remainder handling of the plain profile: absorb the
    remainder into the last ramp cell."""
    widths = list(ramp)
    widths[-1] += rest
    return widths


def _widths_to_nodes(
    widths: list[float],
    p_fine: float,
    p_coarse: float,
) -> list[float]:
    """Accumulate cell widths from ``p_fine`` toward ``p_coarse``.

    The last node is clamped to ``p_coarse`` against float
    accumulation; the result is returned in ascending order.
    """
    sign = 1.0 if p_coarse > p_fine else -1.0
    nodes = [p_fine]
    x = p_fine
    for w in widths:
        x += sign * w
        nodes.append(x)
    nodes[-1] = p_coarse

    if p_fine > p_coarse:
        nodes.reverse()
    return nodes


def _grade_symmetric_to_uniform(
    p0: float,
    p1: float,
    h_fine: float,
    h_max: float,
    g: float,
    min_cell: float = 0.0,
) -> list[float]:
    """Symmetric ramps from both ends to h_max with uniform middle.

    Cells start at ``h_fine`` next to ``p0`` and ``p1``, grow by ``g``
    until they reach ``h_max``, and the middle is filled with uniform
    cells of size ``h_max``.  Short intervals (where two full ramps plus
    a middle cell don't fit) fall back to legacy symmetric grading via
    :func:`_graded_subdivision`, which produces clean ratio-``g`` cells.
    ``min_cell`` is the hard cell-size floor (WP-M3): no generated
    cell may undershoot it.
    """
    interval = p1 - p0
    if interval <= 0:
        return [p0, p1]

    h_fine = min(h_fine, h_max)

    if h_fine >= h_max * (1.0 - 1e-10) or g <= 1.0 + 1e-10:
        n = _n_uniform_floor(interval, h_max, min_cell)
        return list(np.linspace(p0, p1, n + 1))

    # A "full ramp" reaches h_max in cells h_fine, h_fine·g, ..., h_max.
    full_ramp: list[float] = [h_fine]
    while full_ramp[-1] < h_max * (1.0 - 1e-10):
        nxt = full_ramp[-1] * g
        if nxt >= h_max:
            full_ramp.append(h_max)
            break
        full_ramp.append(nxt)

    full_ramp_sum = sum(full_ramp)
    middle = interval - 2.0 * full_ramp_sum

    if middle >= 0.5 * h_max:
        if middle < min_cell:
            # Middle too small for a floor-respecting cell — absorb
            # symmetrically into the two innermost ramp cells.
            left = list(full_ramp)
            left[-1] += 0.5 * middle
            full = left + list(reversed(left))
        else:
            # Long enough for full ramp + uniform middle + reversed ramp.
            n_middle = _n_uniform_floor(middle, h_max, min_cell)
            h_middle = middle / n_middle
            full = list(full_ramp) + [h_middle] * n_middle + list(reversed(full_ramp))
    else:
        # Short interval — legacy symmetric grading.  Find smallest n
        # such that h0 = interval / denom <= h_fine (within
        # _H_FINE_TOL), then back off while the refit undershoots the
        # hard floor (h0 grows as n shrinks, so the scan terminates at
        # n = 2 with h0 = interval / 2 >= min_cell for any
        # floor-respecting interval).
        n = 2
        while True:
            h0 = _h0_symmetric(interval, n, g)
            if h0 <= h_fine * (1.0 + _H_FINE_TOL):
                break
            n += 1
            if n > 10000:
                break
        if min_cell > 0:
            while n > 1 and _h0_symmetric(interval, n, g) < min_cell * (1.0 - 1e-12):
                n -= 1
        # DD-193: fine-end cells stay at h_fine; the ratio relaxes.
        g_eff = g
        if _h0_symmetric(interval, n, g) < h_fine * (1.0 - 1e-9):
            g_eff = _ratio_for_exact_fill(interval, h_fine, n, g, symmetric=True)
        return _graded_subdivision(p0, p1, n, g_eff)

    nodes = [p0]
    x = p0
    for w in full:
        x += w
        nodes.append(x)
    nodes[-1] = p1
    return nodes


def _full_ramp(h_fine: float, h_max: float, g: float) -> list[float]:
    """Cell widths ``h_fine, h_fine·g, …`` up to and including ``h_max``."""
    ramp: list[float] = [h_fine]
    while ramp[-1] < h_max * (1.0 - 1e-10):
        nxt = ramp[-1] * g
        if nxt >= h_max:
            ramp.append(h_max)
            break
        ramp.append(nxt)
    return ramp


def _grade_asymmetric_to_uniform(
    p0: float,
    p1: float,
    h_lo: float,
    h_hi: float,
    h_max: float,
    g: float,
    min_cell: float = 0.0,
) -> list[float]:
    """Ramps of different start sizes from both ends to h_max, uniform
    middle (DD-194: one end of the interval holds a singular edge).

    The long-interval profile is that of
    :func:`_grade_symmetric_to_uniform` with each ramp starting at its
    own size; a remainder below half a bulk cell is absorbed into the
    innermost ramp cells.  A short interval — the two full ramps do
    not fit — is filled by the tent of :func:`_two_ramp_fill`.
    ``min_cell`` is the hard cell-size floor (WP-M3).
    """
    interval = p1 - p0
    if interval <= 0:
        return [p0, p1]

    h_lo = min(h_lo, h_max)
    h_hi = min(h_hi, h_max)
    if h_lo == h_hi:
        return _grade_symmetric_to_uniform(p0, p1, h_lo, h_max, g, min_cell=min_cell)
    if g <= 1.0 + 1e-10:
        n = _n_uniform_floor(interval, h_max, min_cell)
        return list(np.linspace(p0, p1, n + 1))

    ramp_lo = _full_ramp(h_lo, h_max, g)
    ramp_hi = _full_ramp(h_hi, h_max, g)
    middle = interval - sum(ramp_lo) - sum(ramp_hi)

    if middle >= 0.5 * h_max and middle >= min_cell:
        n_middle = _n_uniform_floor(middle, h_max, min_cell)
        widths = ramp_lo + [middle / n_middle] * n_middle + list(reversed(ramp_hi))
    elif middle >= 0.0:
        # Both ramps fit but the remainder is below half a bulk cell
        # (or the floor): absorb it into the two innermost ramp cells
        # — each grows by less than a quarter cell, within the ratio.
        ramp_lo[-1] += 0.5 * middle
        ramp_hi[-1] += 0.5 * middle
        widths = ramp_lo + list(reversed(ramp_hi))
    else:
        widths = _two_ramp_fill(interval, h_lo, h_hi, g, min_cell)

    nodes = [p0]
    x = p0
    for w in widths:
        x += w
        nodes.append(x)
    nodes[-1] = p1
    return nodes


def _two_ramp_fill(
    interval: float,
    h_lo: float,
    h_hi: float,
    g: float,
    min_cell: float = 0.0,
) -> list[float]:
    """Cell widths (ascending position) of a short interval whose two
    ends ask for different fine sizes (DD-194).

    A tent: the smaller fine size is pinned exactly, the cells grow
    from it by one common ratio ``r ≤ g`` to a peak and shrink by the
    same ratio back to the other end, whose cell may be anything from
    the pinned size up to the DD-105 tolerance above its own — the
    time step is bound by the pinned cell already, so a smaller cell
    at the coarse end costs nothing.  The smallest count with such a
    profile wins; among its splits the one ending closest to the
    coarse end's own size.  Intervals below two pinned cells are one
    cell; where no tent fits exactly (a count gap), the one-sided
    ratio-``g`` refit from the pinned end applies (DD-193 rules).
    ``min_cell`` is the hard floor; every tent cell is at least the
    pinned size, which the caller has floored.
    """
    if h_lo > h_hi:
        return list(reversed(_two_ramp_fill(interval, h_hi, h_lo, g, min_cell)))
    if interval < 2.0 * h_lo:
        return [interval]
    cap = h_hi * (1.0 + _H_FINE_TOL)

    def _sum(n_lo: int, n_hi: int, r: float) -> float:
        if abs(r - 1.0) < 1e-12:
            return h_lo * (n_lo + n_hi)
        up = h_lo * (r**n_lo - 1.0) / (r - 1.0)
        peak = h_lo * r ** (n_lo - 1)
        down = peak * (1.0 - r ** (-n_hi)) / (r - 1.0) if n_hi else 0.0
        return up + down

    n_max = int(math.ceil(interval / h_lo)) + 1
    for n in range(2, n_max + 1):
        feasible = []
        for n_hi in range(0, n):
            n_lo = n - n_hi
            m = n_lo - 1 - n_hi  # coarse-end cell = h_lo · r^m
            if m < 0:
                continue
            if _sum(n_lo, n_hi, g) < interval or _sum(n_lo, n_hi, 1.0) > interval:
                continue
            lo, hi = 1.0, g
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _sum(n_lo, n_hi, mid) < interval:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-14:
                    break
            r = hi
            a_hi = h_lo * r**m
            if a_hi > cap:
                continue
            feasible.append((a_hi, n_hi, r))
        if feasible:
            _a_hi, n_hi, r = max(feasible)
            n_lo = n - n_hi
            widths = [h_lo * r**i for i in range(n_lo)]
            peak = widths[-1]
            widths += [peak * r ** (-(j + 1)) for j in range(n_hi)]
            return widths

    n = _n_one_sided(interval, h_lo, g, min_cell=min_cell)
    g_eff = g
    if _h0_one_sided(interval, n, g) < h_lo * (1.0 - 1e-9):
        g_eff = _ratio_for_exact_fill(interval, h_lo, n, g, symmetric=False)
    return list(np.diff(_one_sided_subdivision(0.0, interval, n, g_eff)))


def _graded_subdivision(p0: float, p1: float, n: int, g: float) -> list[float]:
    """Subdivide [p0, p1] into *n* cells with symmetric graded spacing.

    Cells grow from both endpoints toward the centre with ratio *g*.  The
    smallest cells are at the endpoints (material interfaces); the largest
    cell is at the midpoint.

    For ``g ≈ 1`` or ``n <= 2`` the result degenerates to uniform spacing.

    The starting cell size ``h0`` is derived from the constraint that cell
    widths sum to the interval length:

        sum = 2·h0·(g^n_half − 1)/(g − 1)  +  h0·g^n_half  (if n odd)

    where ``n_half = n // 2``.
    """
    if n <= 1:
        return list(np.linspace(p0, p1, n + 1))
    if abs(g - 1.0) < 1e-10:
        return list(np.linspace(p0, p1, n + 1))

    interval = p1 - p0
    n_half = n // 2

    # Sum of geometric series from one end: sum(g^i, i=0..n_half-1)
    series_sum = (g**n_half - 1.0) / (g - 1.0)

    # Extra centre cell for odd n
    centre_term = g**n_half if (n % 2 == 1) else 0.0
    denom = 2.0 * series_sum + centre_term
    h0 = interval / denom

    # Build left half (n_half cells from p0)
    nodes: list[float] = [p0]
    x = p0
    h = h0
    for _ in range(n_half):
        x += h
        nodes.append(x)
        h *= g

    # Optional centre cell
    if n % 2 == 1:
        x += h0 * (g**n_half)
        nodes.append(x)

    # Build right half as mirror from p1, then reverse to get left→right order
    right: list[float] = [p1]
    x = p1
    h = h0
    for _ in range(n_half):
        x -= h
        right.append(x)
        h *= g
    right.reverse()  # now left-to-right

    # Combine: nodes already contains left half up to midpoint;
    # right[1:] contains the right half (skip the mirrored midpoint).
    nodes.extend(right[1:])

    # Clamp endpoints exactly to avoid float accumulation drift
    nodes[0] = p0
    nodes[-1] = p1

    return nodes


def _ratio_for_exact_fill(interval: float, h0: float, n: int, g: float, symmetric: bool) -> float:
    """Growth ratio ``g' in [1, g]`` with which *n* cells starting at ``h0``
    fill *interval* exactly (DD-193).

    The integer-count refits (``_n_one_sided`` / the symmetric scan)
    fix the ratio at ``g`` and let the fine-end cell fall out of the
    count, anywhere between ``h_fine / g`` and ``h_fine`` — an
    undershoot of up to ``1 − 1/g`` that costs time steps and buys no
    resolution (DD-105).  Keeping ``h0 = h_fine`` and relaxing the
    ratio instead fills the interval with the same count, no fine-end
    undershoot and every neighbour ratio ``≤ g``.  Bisection on the
    monotone series sum; returns ``g`` when even ratio ``g`` cannot
    fill the interval from ``h0`` (the caller keeps its refit), and
    ``1`` when uniform cells of ``h0`` already overfill it.
    """

    def _total(r: float) -> float:
        if abs(r - 1.0) < 1e-12:
            return h0 * n
        if symmetric:
            n_half = n // 2
            series = (r**n_half - 1.0) / (r - 1.0)
            centre = (r**n_half) if (n % 2 == 1) else 0.0
            return h0 * (2.0 * series + centre)
        return h0 * (r**n - 1.0) / (r - 1.0)

    if _total(g) < interval:
        return g
    if _total(1.0) >= interval:
        return 1.0
    lo, hi = 1.0, g
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _total(mid) < interval:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return hi


def _h0_symmetric(interval: float, n: int, g: float) -> float:
    """Fine-end cell size of the symmetric ratio-``g`` refit with *n* cells."""
    if n <= 1:
        return interval
    if abs(g - 1.0) < 1e-10:
        return interval / n
    n_half = n // 2
    series = (g**n_half - 1.0) / (g - 1.0)
    centre = (g**n_half) if (n % 2 == 1) else 0.0
    return interval / (2.0 * series + centre)


def _h0_one_sided(interval: float, n: int, g: float) -> float:
    """Fine-end cell size of the one-sided ratio-``g`` refit with *n* cells."""
    if n <= 1:
        return interval
    if abs(g - 1.0) < 1e-10:
        return interval / n
    return interval * (g - 1.0) / (g**n - 1.0)


def _n_one_sided(
    interval: float,
    h_fine: float,
    g: float,
    min_cell: float = 0.0,
) -> int:
    """Number of cells for one-sided grading to cover *interval*.

    Returns the smallest *n* such that the fine-end cell size

        h0 = interval · (g − 1) / (g^n − 1)  ≤  h_fine · (1 + _H_FINE_TOL),

    backed off while the refit undershoots the hard floor
    ``min_cell`` (WP-M3; h0 grows as n shrinks, so the back-off
    terminates).  For g ≈ 1 falls back to a uniform count.
    """
    h_fine_tol = h_fine * (1.0 + _H_FINE_TOL)
    if abs(g - 1.0) < 1e-10:
        n = max(1, math.ceil(interval / h_fine_tol))
    else:
        n = max(1, math.ceil(math.log(1.0 + interval * (g - 1.0) / h_fine_tol) / math.log(g)))
    if min_cell > 0:
        while n > 1 and _h0_one_sided(interval, n, g) < min_cell * (1.0 - 1e-12):
            n -= 1
    return n


def _tailed_widths(
    interval: float,
    h_fine: float,
    g: float,
    min_cell: float,
    n_buf: int,
) -> list[float]:
    """Cell widths for a boundary interval too short for the full
    ramp-to-``h_max`` profile: geometric growth from the fine end over
    ``n_r`` cells, then a uniform ``n_buf``-cell tail at the coarse
    (domain-wall) end (DD-107).

    The tail size is a second free parameter next to the fine-end size
    ``h0`` — coupling it rigidly to the ramp end would coarsen the
    count granularity by a factor ``n_buf`` and reintroduce the
    DD-105 fine-end undershoot.  Instead: ``n_r`` is the smallest
    ramp count that can cover the interval with ``h0`` at the DD-105
    ``h_fine`` tolerance; then ``h0`` is kept there while the tail
    absorbs the remainder, and only if that opens a seam down-step
    beyond ``g`` is ``h0`` lowered just enough to close it — every
    neighbour ratio stays within ``g`` by construction, and the
    fine-end undershoot stays in the legacy refit's class.

    Backed off while the refit undershoots the hard floor
    ``min_cell`` (WP-M3).  If even the pure ``n_buf``-cell uniform
    split undershoots the floor, the buffer is dropped and the legacy
    one-sided refit applies — the floor wins.
    """
    h_fine_tol = h_fine * (1.0 + _H_FINE_TOL)

    def _series(n_r: int) -> float:
        if abs(g - 1.0) < 1e-10:
            return float(n_r)
        return (g**n_r - 1.0) / (g - 1.0)

    def _profile(n_r: int) -> tuple[float, float]:
        """(h0, h_u) filling the interval exactly with n_r ramp cells
        and n_buf tail cells, seams within g."""
        if n_r == 0:
            return 0.0, interval / n_buf
        s = _series(n_r)
        h0 = min(h_fine_tol, interval / (s + n_buf * g ** (n_r - 1)))
        h_u = (interval - h0 * s) / n_buf
        return h0, h_u

    # Smallest ramp count that reaches the interval with h0 at the
    # h_fine tolerance and the tail no more than one growth step above
    # the ramp end.
    n_r = 0
    while (n_r == 0 and n_buf * h_fine_tol < interval) or (
        n_r > 0 and h_fine_tol * (_series(n_r) + n_buf * g**n_r) < interval
    ):
        n_r += 1
        if n_r >= 500:
            break
    if min_cell > 0:
        while n_r > 0 and _profile(n_r)[0] < min_cell * (1.0 - 1e-12):
            n_r -= 1
        h0, h_u = _profile(n_r)
        if min(h0 if n_r > 0 else h_u, h_u) < min_cell * (1.0 - 1e-12):
            n_legacy = _n_one_sided(interval, h_fine, g, min_cell=min_cell)
            h0 = _h0_one_sided(interval, n_legacy, g)
            return [h0 * g**k for k in range(n_legacy)]
    h0, h_u = _profile(n_r)
    if n_r == 0:
        return [h_u] * n_buf
    return [h0 * g**k for k in range(n_r)] + [h_u] * n_buf


def _one_sided_subdivision(p_fine: float, p_coarse: float, n: int, g: float) -> list[float]:
    """Subdivide [min(p_fine, p_coarse), max(...)] into *n* cells.

    Smallest cell at *p_fine*, growing by factor *g* toward *p_coarse*.
    Result is always returned in ascending order.
    """
    if n <= 1:
        lo, hi = min(p_fine, p_coarse), max(p_fine, p_coarse)
        return [lo, hi]

    interval = abs(p_coarse - p_fine)

    if abs(g - 1.0) < 1e-10:
        nodes = list(np.linspace(min(p_fine, p_coarse), max(p_fine, p_coarse), n + 1))
        return nodes

    h0 = interval * (g - 1.0) / (g**n - 1.0)
    sign = 1.0 if p_coarse > p_fine else -1.0

    nodes: list[float] = [p_fine]
    x = p_fine
    h = h0
    for _ in range(n):
        x += sign * h
        nodes.append(x)
        h *= g

    nodes[-1] = p_coarse  # clamp against float accumulation

    if p_fine > p_coarse:
        nodes.reverse()

    return nodes
