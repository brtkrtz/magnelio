"""
pythonocc-core (OpenCASCADE) backend for geometry operations.

All functions in this module require ``pythonocc-core`` to be installed.
The import is deferred so that the rest of the package can be imported
without OCC being present.

See design-decisions.md DD-003 for the choice of pythonocc-core.
"""

from __future__ import annotations

import bisect
import contextlib
import math
import os
import sys
import warnings

import numpy as np


def _require_occ():
    """Import and return OCC modules, raising a helpful error if not installed."""
    try:
        from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
        from OCC.Core.BRepAlgoAPI import (  # noqa: PLC0415
            BRepAlgoAPI_Common,
            BRepAlgoAPI_Cut,
            BRepAlgoAPI_Fuse,
        )
        from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: PLC0415
        from OCC.Core.BRepPrimAPI import (  # noqa: PLC0415
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakeCone,
            BRepPrimAPI_MakeCylinder,
            BRepPrimAPI_MakeSphere,
            BRepPrimAPI_MakeTorus,
        )
        from OCC.Core.gp import (  # noqa: PLC0415
            gp_Ax1,
            gp_Ax2,
            gp_Dir,
            gp_Pnt,
            gp_Trsf,
            gp_Vec,
        )

        return {
            "MakeBox": BRepPrimAPI_MakeBox,
            "MakeSphere": BRepPrimAPI_MakeSphere,
            "MakeCylinder": BRepPrimAPI_MakeCylinder,
            "MakeCone": BRepPrimAPI_MakeCone,
            "MakeTorus": BRepPrimAPI_MakeTorus,
            "Fuse": BRepAlgoAPI_Fuse,
            "Common": BRepAlgoAPI_Common,
            "Cut": BRepAlgoAPI_Cut,
            "brepbndlib": brepbndlib,
            "Bnd_Box": Bnd_Box,
            "gp_Pnt": gp_Pnt,
            "gp_Vec": gp_Vec,
            "gp_Dir": gp_Dir,
            "gp_Ax1": gp_Ax1,
            "gp_Ax2": gp_Ax2,
            "gp_Trsf": gp_Trsf,
            "Transform": BRepBuilderAPI_Transform,
        }
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for geometry operations. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------

# OCC's Precision::Confusion() — geometry at or below this scale is
# degenerate for the OCC kernel (MakeBox etc. raise a cryptic
# Standard_DomainError; found by the WP-M5 mesher stress sentinel).
# The limit applies in *scaled model units* (DD-120): the automatic
# model scale moves sub-um models into the kernel's comfortable range,
# so the effective meter limit is 1e-7 / scale.
_OCC_PRECISION = 1e-7


def _scale3(p, scale: float):
    """Scale a 3-tuple of meters into scaled model units."""
    return (p[0] * scale, p[1] * scale, p[2] * scale)


def _check_dimensions(kind: str, scale: float = 1.0, **dims: float) -> None:
    for name, val in dims.items():
        if val * scale <= _OCC_PRECISION:
            limit = _OCC_PRECISION / scale
            raise ValueError(
                f"{kind} {name} = {val:.3e} m is at or below the OCC "
                f"geometric precision for this model ({limit:.0e} m at "
                f"model scale {scale:g}) — the CAD kernel cannot "
                f"represent it.  Model features below that limit "
                f"through materials/boundary conditions instead of "
                f"explicit solids."
            )


def make_brick(origin: tuple, size: tuple, scale: float = 1.0):
    """Build an OCC box shape (coordinates in scaled model units)."""
    occ = _require_occ()
    _check_dimensions("Brick", scale, size_x=size[0], size_y=size[1], size_z=size[2])
    pnt = occ["gp_Pnt"](*_scale3(origin, scale))
    shape = occ["MakeBox"](pnt, size[0] * scale, size[1] * scale, size[2] * scale).Shape()
    return shape


def make_sphere(center: tuple, radius: float, scale: float = 1.0):
    """Build an OCC sphere shape (coordinates in scaled model units)."""
    occ = _require_occ()
    _check_dimensions("Sphere", scale, radius=radius)
    pnt = occ["gp_Pnt"](*_scale3(center, scale))
    shape = occ["MakeSphere"](pnt, radius * scale).Shape()
    return shape


def _axis_with_height(axis, height):
    """Resolve axis letter/vector and fold a negative height into it.

    A negative height means "extrude along -axis from the origin"; OCC's
    prism builders want a positive height along an explicit direction.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

    d = normalize_axis(axis)
    if height < 0.0:
        return tuple(-c for c in d), -height
    return d, height


def make_cylinder(
    origin: tuple,
    radius: float,
    height: float,
    axis="z",
    inner_radius: float = 0.0,
    angle_deg=None,
    scale: float = 1.0,
):
    """Build an OCC cylinder shape (coordinates in scaled model units).

    With ``inner_radius`` the result is a tube, with ``angle_deg`` an
    angular segment; the two combine into an annular segment.  The plain
    solid cylinder keeps its original code path bit-for-bit.

    ``angle_deg`` is ``(start, end)`` measured right-handed about *axis*
    from :func:`~magnelio.geo._axes.reference_dir`, which the analytic
    bounding box reads from the same place.
    """
    occ = _require_occ()
    from OCC.Core.gp import gp_Ax2, gp_Dir  # noqa: PLC0415

    if inner_radius <= 0.0 and angle_deg is None:
        d, height = _axis_with_height(axis, height)
        _check_dimensions("Cylinder", scale, radius=radius, height=height)
        ax = gp_Ax2(occ["gp_Pnt"](*_scale3(origin, scale)), gp_Dir(*d))
        return occ["MakeCylinder"](ax, radius * scale, height * scale).Shape()

    from magnelio.geo._axes import cross, normalize_axis, reference_dir  # noqa: PLC0415

    # The angular frame is measured about the *declared* axis, so a
    # negative height moves the body without reversing the sweep.
    d = normalize_axis(axis)
    span_height = abs(height)
    _check_dimensions("Cylinder", scale, radius=radius, height=span_height)
    if inner_radius > 0.0:
        _check_dimensions("Cylinder", scale, wall=radius - inner_radius)
    base = tuple(o + min(0.0, height) * a for o, a in zip(origin, d))

    if angle_deg is None:
        ax = gp_Ax2(occ["gp_Pnt"](*_scale3(base, scale)), gp_Dir(*d))
        shape = occ["MakeCylinder"](ax, radius * scale, span_height * scale).Shape()
    else:
        start_deg, end_deg = angle_deg
        ref = reference_dir(d)
        turn = math.radians(start_deg)
        # ref is perpendicular to d, so Rodrigues reduces to this.
        x_dir = tuple(a * math.cos(turn) + b * math.sin(turn) for a, b in zip(ref, cross(d, ref)))
        ax = gp_Ax2(occ["gp_Pnt"](*_scale3(base, scale)), gp_Dir(*d), gp_Dir(*x_dir))
        shape = occ["MakeCylinder"](
            ax, radius * scale, span_height * scale, math.radians(end_deg - start_deg)
        ).Shape()

    if inner_radius > 0.0:
        # Overshoot axially so the bore's end caps do not land on the
        # outer ones, which would leave coincident faces in the cut.
        pad = 0.01 * span_height
        bore_base = tuple(o - pad * a for o, a in zip(base, d))
        bore_ax = gp_Ax2(occ["gp_Pnt"](*_scale3(bore_base, scale)), gp_Dir(*d))
        bore = occ["MakeCylinder"](
            bore_ax, inner_radius * scale, (span_height + 2.0 * pad) * scale
        ).Shape()
        shape = boolean_difference(shape, bore)
    return shape


def make_cone(
    origin: tuple,
    bottom_radius: float,
    top_radius: float,
    height: float,
    axis="z",
    scale: float = 1.0,
):
    """Build an OCC cone (or truncated cone) shape (scaled model units)."""
    occ = _require_occ()
    d, height = _axis_with_height(axis, height)
    _check_dimensions("Cone", scale, height=height)
    from OCC.Core.gp import gp_Ax2, gp_Dir  # noqa: PLC0415

    ax = gp_Ax2(occ["gp_Pnt"](*_scale3(origin, scale)), gp_Dir(*d))
    return occ["MakeCone"](ax, bottom_radius * scale, top_radius * scale, height * scale).Shape()


def make_torus(
    center: tuple, major_radius: float, minor_radius: float, axis="z", scale: float = 1.0
):
    """Build an OCC torus shape (coordinates in scaled model units)."""
    occ = _require_occ()
    from OCC.Core.gp import gp_Ax2, gp_Dir  # noqa: PLC0415

    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

    d = normalize_axis(axis)
    ax = gp_Ax2(occ["gp_Pnt"](*_scale3(center, scale)), gp_Dir(*d))
    return occ["MakeTorus"](ax, major_radius * scale, minor_radius * scale).Shape()


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------


def keep_operands_intact(op) -> None:
    """Forbid an OCCT Boolean operation from editing its input shapes (DD-146).

    ``BOPAlgo_Options::NonDestructive`` defaults to *false*: the kernel
    is free to raise the tolerance of the argument shapes, insert
    p-curves and shift vertices in place, and it does so on every call.
    Magnelio caches the built solid per shape (:func:`cached_occ_shape`),
    so those edits are permanent and they accumulate — a mesh build takes
    thousands of sections through the very same ``TopoDS_Shape``.
    Measured on a stripline-coupler assembly: edge tolerance grew
    1.1e-4 m -> 7.0e-3 m in one mesh build, after which cutting the
    solid against a half-space returned a fragment instead of the solid.

    Every Boolean the kernel runs on a cached shape must therefore call
    this before ``Build()``.  It costs nothing measurable (16.4 s vs.
    17.1 s on that mesh build) because OCCT only copies the sub-shapes
    it would otherwise have modified.
    """
    op.SetNonDestructive(True)


def _run_bop(op_cls, arguments: list, tools: list):
    """Run one OCCT Boolean operation with internal parallelism.

    The two-shape constructors (``Fuse(a, b)`` etc.) build immediately,
    before ``SetRunParallel`` could take effect — hence the explicit
    ``SetArguments``/``SetTools``/``Build`` sequence.  ``SetRunParallel``
    only changes the execution strategy, not the computed shape.
    """
    from OCC.Core.TopTools import TopTools_ListOfShape  # noqa: PLC0415

    args_list = TopTools_ListOfShape()
    for s in arguments:
        args_list.Append(s)
    tools_list = TopTools_ListOfShape()
    for s in tools:
        tools_list.Append(s)
    op = op_cls()
    op.SetArguments(args_list)
    op.SetTools(tools_list)
    op.SetRunParallel(True)
    keep_operands_intact(op)
    op.Build()
    if not op.IsDone():
        raise RuntimeError(
            f"OCC Boolean operation {op_cls.__name__} failed "
            f"({len(arguments)} argument(s), {len(tools)} tool(s))."
        )
    return op.Shape()


def boolean_union(shapes: list):
    """Fuse a list of OCC shapes into one.

    Single N-ary Boolean operation: all shapes enter one BOP pass.  The
    former pairwise loop re-processed the growing accumulated compound
    on every step — measured 28.3 s for 501 tools vs. 0.1 s N-ary, with
    bit-identical downstream cross-sections of the fused result.
    """
    occ = _require_occ()
    if len(shapes) == 1:
        return shapes[0]
    return _run_bop(occ["Fuse"], shapes[:1], shapes[1:])


def boolean_intersection(shape_a, shape_b):
    """Intersect two OCC shapes."""
    occ = _require_occ()
    return _run_bop(occ["Common"], [shape_a], [shape_b])


def boolean_difference(base, tool):
    """Subtract tool from base."""
    occ = _require_occ()
    return _run_bop(occ["Cut"], [base], [tool])


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


def bounding_box(
    shape, scale: float = 1.0
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return axis-aligned bounding box of an OCC shape.

    Args:
        shape: An OCC shape built at *scale* (scaled model units).
        scale: DD-120 model scale factor of the shape.

    Returns:
        ``(min_corner, max_corner)`` as two 3-tuples in meters.
    """
    occ = _require_occ()
    bbox = occ["Bnd_Box"]()
    # AddOptimal computes tight bounds from the actual geometry.  Plain
    # ``Add`` bounds a freeform (B-spline) surface by the convex hull of
    # its control poles, which over-sizes swept / lofted / revolved solids
    # by ~2x and would inflate the mesh grid ~8x in volume.  ``AddOptimal``
    # is exact for analytic primitives (box/cylinder/sphere/Boolean) — and
    # no slower there — so it is a safe drop-in.  ``(useTriangulation=False,
    # useShapeTolerance=False)`` = geometry-based bounds with no
    # shape-tolerance padding (no gap correction needed).
    occ["brepbndlib"].AddOptimal(shape, bbox, False, False)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return (xmin / scale, ymin / scale, zmin / scale), (xmax / scale, ymax / scale, zmax / scale)


# ---------------------------------------------------------------------------
# Critical plane extraction (for mesher)
# ---------------------------------------------------------------------------


def extract_critical_planes(geometry, scale: float = 1.0) -> dict[str, list[float]]:
    """Extract x, y, z critical plane positions from a CSG geometry tree.

    Critical planes are positions where a material boundary surface is
    tangent to an axis-normal plane — the outer shape bounding-box
    extents plus, for every *face* of the shape, the axis-normal tangent
    positions of the underlying surface (a plane's position when it is
    axis-normal; ``center ± radius`` for cylinders and spheres).
    Per-face extraction makes interior boundary faces visible to the
    mesher: a hole cut by ``Difference`` (behind which the model
    background material is exposed) contributes its own planes, so
    background-material regions receive the same grid-line snapping and
    feature-based refinement as explicitly added shapes.

    Tangent positions are taken from the analytic surface but only kept
    when they fall inside the trimmed face's own bounding box — a face
    trimmed by a Boolean cut must not contribute planes at its trim
    edges (those are topology artefacts, not material-boundary
    tangents), nor tangent positions of surface regions it does not
    actually cover.

    Args:
        geometry: A geometry model (list of CSG shapes or a GeometryModel).

    Returns:
        Dict with keys ``'x'``, ``'y'``, ``'z'``, each a list of floats.
    """
    critical: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    for _shape, planes in extract_critical_planes_per_shape(geometry, scale=scale):
        for axis in ("x", "y", "z"):
            critical[axis].extend(p for p, _exact in planes[axis])
    return critical


def extract_critical_planes_per_shape(
    geometry,
    scale: float = 1.0,
) -> list[tuple[object, dict[str, list[tuple[float, bool]]]]]:
    """Per-shape variant of :func:`extract_critical_planes`.

    Returns ``[(shape, {'x': [(pos, exact), ...], ...}), ...]`` so the
    mesher can filter a single shape's contribution (e.g. drop the
    far-side face of a thin PEC metallization, WP-M2) without losing an
    identical plane another shape legitimately contributes.

    Each plane carries a provenance flag: ``exact = True`` for planes
    read from an analytic face surface (a material boundary at that
    exact position), ``False`` for shape bounding-box extents.  The
    distinction matters to the plane clustering: OCCT Booleans on
    interpenetrating operands return a bounding box inflated by
    ``Precision::Confusion`` (1e-7 model units) beyond the true
    geometry, and averaging that phantom extent with the real face
    plane puts the grid line — and with it the domain boundary — tens
    of nanometres past the material surface (KB-013: the resulting
    sliver fill factor fails the DTBC slab certificate and silently
    downgrades every port channel on that face to Mur-1st).
    """
    result: list[tuple[object, dict[str, list[tuple[float, bool]]]]] = []

    shapes = list(geometry) if hasattr(geometry, "__iter__") else [geometry]

    for shape in shapes:
        critical: dict[str, list[tuple[float, bool]]] = {"x": [], "y": [], "z": []}
        # Outer bounding box (legacy behaviour): silhouette extents of
        # the whole shape, covers tilted/free-form faces conservatively.
        try:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = shape.bounding_box(scale)
        except Exception:
            # OCC unavailable — skip; the mesher uses domain bounds only.
            continue
        critical["x"].extend([(xmin, False), (xmax, False)])
        critical["y"].extend([(ymin, False), (ymax, False)])
        critical["z"].extend([(zmin, False), (zmax, False)])

        try:
            planes = _face_critical_planes(shape._occ_shape(scale))
        except Exception:
            result.append((shape, critical))
            continue
        for axis in ("x", "y", "z"):
            # _face_critical_planes works in scaled model units; the
            # power-of-two division back to meters is lossless.
            critical[axis].extend((p / scale, True) for p in planes[axis])
        result.append((shape, critical))

    return result


# Axis unit vectors used by _face_critical_planes.
_AXIS_DIRS = (
    ("x", (1.0, 0.0, 0.0)),
    ("y", (0.0, 1.0, 0.0)),
    ("z", (0.0, 0.0, 1.0)),
)


def _face_critical_planes(occ_shape) -> dict[str, list[float]]:
    """Axis-normal tangent positions of every face of an OCC shape.

    Analytic surface types are handled exactly:

    * plane — its position along the axis, when the face normal is
      axis-parallel;
    * cylinder — ``center_i ± R`` on both axes perpendicular to the
      cylinder axis, when the cylinder axis is itself axis-parallel;
    * sphere — ``center_i ± R`` on all three axes.

    Any other surface type (cones, tori, free-form) contributes nothing
    here; the caller's shape-bbox extents cover their silhouettes.

    Each candidate is kept only if it lies within the *trimmed* face's
    bounding box: a Boolean cut may leave only part of the analytic
    surface (e.g. a quarter arc), and tangent positions of the removed
    part must not become planes.

    Positions are in the shape's build units — for a DD-120-scaled
    shape the caller divides by the scale factor.  The absolute
    epsilons below therefore act in scaled units, the O(100) regime
    they were tuned for.
    """
    occ = _require_occ()
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415
    from OCC.Core.GeomAbs import (  # noqa: PLC0415
        GeomAbs_Cylinder,
        GeomAbs_Plane,
        GeomAbs_Sphere,
    )
    from OCC.Core.TopAbs import TopAbs_FACE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    align_tol = 1.0 - 1e-9  # |cos| threshold for "axis-parallel"

    planes: dict[str, list[float]] = {"x": [], "y": [], "z": []}
    explorer = TopExp_Explorer(occ_shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()

        surf = BRepAdaptor_Surface(face)
        stype = surf.GetType()

        # Candidate axis-normal tangent positions of the analytic surface
        candidates: dict[str, list[float]] = {"x": [], "y": [], "z": []}
        if stype == GeomAbs_Plane:
            pln = surf.Plane()
            n = pln.Axis().Direction()
            loc = pln.Location()
            n_xyz = (n.X(), n.Y(), n.Z())
            loc_xyz = (loc.X(), loc.Y(), loc.Z())
            for k, (axis, _) in enumerate(_AXIS_DIRS):
                if abs(n_xyz[k]) >= align_tol:
                    candidates[axis].append(loc_xyz[k])
        elif stype == GeomAbs_Cylinder:
            cyl = surf.Cylinder()
            d = cyl.Axis().Direction()
            c = cyl.Axis().Location()
            radius = cyl.Radius()
            d_xyz = (d.X(), d.Y(), d.Z())
            c_xyz = (c.X(), c.Y(), c.Z())
            if max(abs(v) for v in d_xyz) >= align_tol:
                for k, (axis, _) in enumerate(_AXIS_DIRS):
                    if abs(d_xyz[k]) < align_tol:
                        candidates[axis].extend(
                            [c_xyz[k] - radius, c_xyz[k] + radius],
                        )
        elif stype == GeomAbs_Sphere:
            sph = surf.Sphere()
            c = sph.Location()
            radius = sph.Radius()
            c_xyz = (c.X(), c.Y(), c.Z())
            for k, (axis, _) in enumerate(_AXIS_DIRS):
                candidates[axis].extend(
                    [c_xyz[k] - radius, c_xyz[k] + radius],
                )
        else:
            continue

        if not any(candidates.values()):
            continue

        # Trim check: keep candidates inside the trimmed face's bbox.
        bbox = occ["Bnd_Box"]()
        # useTriangulation=False: the box must come from the analytic
        # geometry.  A rendered/exported shape carries a coarse
        # triangulation on its cached faces, and a triangulation-based
        # box (enlarged by its deflection) admits tangent candidates of
        # surface regions the trimmed face does not cover — the mesh
        # then depends on whether the model was plotted first (KB-012).
        occ["brepbndlib"].Add(face, bbox, False)
        fxmin, fymin, fzmin, fxmax, fymax, fzmax = bbox.Get()
        gap = bbox.GetGap() + 1e-12
        face_lo = {"x": fxmin + gap, "y": fymin + gap, "z": fzmin + gap}
        face_hi = {"x": fxmax - gap, "y": fymax - gap, "z": fzmax - gap}
        for axis in ("x", "y", "z"):
            tol = 1e-9 + abs(gap)
            for pos in candidates[axis]:
                if face_lo[axis] - tol <= pos <= face_hi[axis] + tol:
                    planes[axis].append(pos)

    return planes


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def occ_translate(shape, vector: tuple, scale: float = 1.0):
    """Translate an OCC shape (vector in meters, applied in scaled units)."""
    occ = _require_occ()
    trsf = occ["gp_Trsf"]()
    trsf.SetTranslation(occ["gp_Vec"](*_scale3(vector, scale)))
    return occ["Transform"](shape, trsf, True).Shape()


def occ_rotate(shape, axis: tuple, angle_deg: float, origin: tuple, scale: float = 1.0):
    """Rotate an OCC shape (origin in meters, applied in scaled units)."""
    occ = _require_occ()
    import math  # noqa: PLC0415

    ax1 = occ["gp_Ax1"](occ["gp_Pnt"](*_scale3(origin, scale)), occ["gp_Dir"](*axis))
    trsf = occ["gp_Trsf"]()
    trsf.SetRotation(ax1, math.radians(angle_deg))
    return occ["Transform"](shape, trsf, True).Shape()


def occ_scale(shape, factor: float, center: tuple, scale: float = 1.0):
    """Uniformly scale an OCC shape (center in meters, applied in scaled units)."""
    occ = _require_occ()
    trsf = occ["gp_Trsf"]()
    trsf.SetScale(occ["gp_Pnt"](*_scale3(center, scale)), factor)
    return occ["Transform"](shape, trsf, True).Shape()


def occ_mirror(shape, normal: tuple, position: float, scale: float = 1.0):
    """Mirror an OCC shape across a plane (position in meters).

    The plane is ``p . normal == position`` with *normal* a unit vector,
    so ``position * normal`` is a point on it.  The transform has
    determinant −1; ``BRepBuilderAPI_Transform`` compensates by reversing
    the shape's orientation, which keeps the result a valid solid with
    positive volume.
    """
    occ = _require_occ()
    point = tuple(position * n for n in normal)
    ax2 = occ["gp_Ax2"](occ["gp_Pnt"](*_scale3(point, scale)), occ["gp_Dir"](*normal))
    trsf = occ["gp_Trsf"]()
    trsf.SetMirror(ax2)
    return occ["Transform"](shape, trsf, True).Shape()


# ---------------------------------------------------------------------------
# Point classification
# ---------------------------------------------------------------------------


def point_in_shape(
    shape, point: tuple[float, float, float], tolerance: float = 1e-7, scale: float = 1.0
) -> bool:
    """Return True if *point* is inside or on the surface of *shape*.

    Uses OCC BRepClass3d_SolidClassifier for robust solid classification.

    Args:
        shape:     An OCC TopoDS_Shape built at *scale* (must be a solid
                   or compound of solids).
        point:     (x, y, z) coordinates in meters.
        tolerance: Classification tolerance in meters.
        scale:     DD-120 model scale factor of the shape.

    Returns:
        True if the point is inside or on the boundary of the solid.
    """
    try:
        from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier  # noqa: PLC0415
        from OCC.Core.gp import gp_Pnt  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for point classification. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    classifier = BRepClass3d_SolidClassifier()
    classifier.Load(shape)
    classifier.Perform(gp_Pnt(*_scale3(point, scale)), tolerance * scale)
    state = classifier.State()
    return state in (TopAbs_IN, TopAbs_ON)


# ---------------------------------------------------------------------------
# Cross-section extraction
# ---------------------------------------------------------------------------


def _plane_lies_in_a_face(shape, axis_idx: int, position: float) -> bool:
    """Whether an axis-normal face of *shape* lies in the cutting plane.

    That is the degenerate case for ``BRepAlgoAPI_Section``: the
    intersection of the solid with the plane is a *face*, not the curves
    the section operator looks for, and on a Boolean result it returns
    only part of the seam structure.  Positions are in scaled units.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415
    from OCC.Core.GeomAbs import GeomAbs_Plane  # noqa: PLC0415
    from OCC.Core.TopAbs import TopAbs_FACE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() != GeomAbs_Plane:
            continue
        plane = surf.Plane()
        direction = plane.Axis().Direction()
        normal = (direction.X(), direction.Y(), direction.Z())
        if abs(normal[axis_idx]) < 1.0 - 1e-9:
            continue
        location = plane.Location()
        coords = (location.X(), location.Y(), location.Z())
        if abs(coords[axis_idx] - position) <= _OCC_PRECISION:
            return True
    return False


def _face_region_wires(shape, plane):
    """Boundary wires of ``shape`` ∩ *plane*, via a face-face Boolean.

    Slower than a section but valid when the plane lies in a face of the
    solid, where the intersection is two-dimensional.  Returns the wires
    of every resulting face, outer boundaries and holes alike.
    """
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common  # noqa: PLC0415
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace  # noqa: PLC0415
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    plane_face = BRepBuilderAPI_MakeFace(plane).Face()
    common = BRepAlgoAPI_Common()
    common.SetRunParallel(True)
    keep_operands_intact(common)
    common.SetArguments(_shape_list([shape]))
    common.SetTools(_shape_list([plane_face]))
    common.Build()
    if not common.IsDone():
        return []

    wires = []
    face_explorer = TopExp_Explorer(common.Shape(), TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        face_explorer.Next()
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wires.append(topods.Wire(wire_explorer.Current()))
            wire_explorer.Next()
    return wires


def _shape_list(shapes):
    """An OCC shape list from a Python sequence."""
    from OCC.Core.TopTools import TopTools_ListOfShape  # noqa: PLC0415

    out = TopTools_ListOfShape()
    for shape in shapes:
        out.Append(shape)
    return out


def _chain_section_edges(edges: list, u_idx: int, v_idx: int) -> list[list[tuple]]:
    """Group section edges into traversal chains on an endpoint graph.

    ``BRepBuilderAPI_MakeWire`` accepts an edge whenever it reaches any
    free end of the wire built so far — including a vertex that already
    joins two edges.  What comes back is a branched pseudo-wire, and
    ``BRepTools_WireExplorer`` then walks one arm of it and stops: the
    remaining edges sit inside the wire and are never tessellated.  They
    leave no open chain either, so nothing downstream can tell.  Measured
    on a plane grazing a lofted electrode: fourteen section edges, eight
    of them added to one wire, one visited — half the cross-section
    silently absent, and the booked area halved against both neighbouring
    planes.

    Chaining the edges here keeps every one of them.  Endpoints are
    merged at the kernel's own vertex tolerance rather than an invented
    one (it is the tolerance that grows in a near-tangency band — a
    factor 28 across the band measured above, so a fixed threshold either
    tears clean contours apart or fuses distinct ones).  A vertex where
    three or more edges meet ends the chain, unless one continuation
    carries straight on: a Boolean seam meeting a contour tangentially is
    the common case there, and of the branches the contour is the one
    with the smallest turn.

    Traversal direction follows the edges' parameterisation, which is
    what the wire path did through ``BRepTools_WireExplorer`` and what
    keeps the winding of separate contours agreeing.  The section edges'
    own orientation flags do not: measured across a grazing band, they
    run opposite on two mirror-image contours of one solid, and honouring
    them makes the signed areas of the pair cancel.

    Parameters
    ----------
    edges : list of TopoDS_Edge
        Section edges, in kernel order.
    u_idx, v_idx : int
        Coordinate indices spanning the section plane.

    Returns
    -------
    list of list of (TopoDS_Edge, bool)
        One chain per contour, in traversal order.  The flag marks an
        edge traversed against its own parameterisation — the same thing
        ``BRepTools_WireExplorer`` expresses through edge orientation.
    """
    from OCC.Core.BRep import BRep_Tool  # noqa: PLC0415
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve  # noqa: PLC0415
    from OCC.Core.gp import gp_Pnt, gp_Vec  # noqa: PLC0415
    from OCC.Core.TopAbs import TopAbs_VERTEX  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    n_edges = len(edges)
    if n_edges == 0:
        return []

    # Endpoint geometry.  Tangents point INTO the edge from each end, so
    # the arrival direction at a vertex is minus the tangent of the end
    # we leave through, and a candidate's tangent is where it would take
    # us — the two are directly comparable.
    pts = np.empty((2 * n_edges, 2))
    tangents = np.empty((2 * n_edges, 2))
    tols = np.empty(2 * n_edges)
    for i, edge in enumerate(edges):
        curve = BRepAdaptor_Curve(edge)
        for k, (param, sign) in enumerate(
            ((curve.FirstParameter(), 1.0), (curve.LastParameter(), -1.0))
        ):
            pnt, deriv = gp_Pnt(), gp_Vec()
            curve.D1(param, pnt, deriv)
            coords = (pnt.X(), pnt.Y(), pnt.Z())
            pts[2 * i + k] = (coords[u_idx], coords[v_idx])
            vec = np.array([deriv.Coord(u_idx + 1), deriv.Coord(v_idx + 1)]) * sign
            norm = float(np.hypot(*vec))
            tangents[2 * i + k] = vec / norm if norm > 0.0 else (1.0, 0.0)
        vertex_tols = []
        explorer = TopExp_Explorer(edge, TopAbs_VERTEX)
        while explorer.More():
            vertex_tols.append(BRep_Tool.Tolerance(topods.Vertex(explorer.Current())))
            explorer.Next()
        tols[2 * i : 2 * i + 2] = max(vertex_tols) if vertex_tols else 1e-7

    # Merge coincident endpoints into vertices.  The bucket grid is sized
    # by the largest tolerance so the 3x3 neighbourhood is exhaustive;
    # each pair is then admitted on its own tolerance, which keeps a
    # loose vertex from swallowing a tight one nearby.
    cell = max(2.0 * float(tols.max()), 1e-12)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, point in enumerate(pts):
        buckets.setdefault((int(point[0] // cell), int(point[1] // cell)), []).append(idx)
    parent = list(range(2 * n_edges))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (bu, bv), members in buckets.items():
        neighbours = [
            j for du in (-1, 0, 1) for dv in (-1, 0, 1) for j in buckets.get((bu + du, bv + dv), ())
        ]
        for a in members:
            for b in neighbours:
                if b <= a:
                    continue
                if float(np.hypot(*(pts[b] - pts[a]))) <= max(tols[a], tols[b]):
                    ra, rb = _find(a), _find(b)
                    if ra != rb:
                        parent[rb] = ra

    node = [_find(i) for i in range(2 * n_edges)]
    ends = [(node[2 * i], node[2 * i + 1]) for i in range(n_edges)]
    incident: dict[int, list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(ends):
        incident.setdefault(a, []).append((i, 0))
        incident.setdefault(b, []).append((i, 1))

    unused = set(range(n_edges))
    # Pass 1: run out the unambiguous stretches.  A vertex joining
    # exactly two edges leaves no choice, so following those needs no
    # geometry and cannot go wrong; every branch is left for pass 2.
    segments: list[dict] = []
    while unused:
        seed = min(unused)
        unused.discard(seed)
        run = [(seed, False)]
        head, tail = ends[seed]
        while len(incident[tail]) == 2:
            nxt = [(i, e) for i, e in incident[tail] if i in unused]
            if not nxt:
                break
            i, e = nxt[0]
            unused.discard(i)
            run.append((i, e == 1))
            tail = ends[i][1 - e]
            if tail == head:
                break
        if tail != head:
            while len(incident[head]) == 2:
                nxt = [(i, e) for i, e in incident[head] if i in unused]
                if not nxt:
                    break
                i, e = nxt[0]
                unused.discard(i)
                run.insert(0, (i, e == 0))
                head = ends[i][1 - e]
                if head == tail:
                    break
        segments.append({"run": run, "head": head, "tail": tail, "closed": head == tail})

    def _outward(segment: dict, side: int) -> np.ndarray:
        """Tangent leading from the segment's *side* vertex into it."""
        i, flipped = segment["run"][-1] if side else segment["run"][0]
        if side:
            return tangents[2 * i + (0 if flipped else 1)]
        return tangents[2 * i + (1 if flipped else 0)]

    # Pass 2: pair up segment ends that meet at a branch.  Two ends
    # continue one another when the traversal leaves the first along
    # the direction it enters the second, i.e. their inward tangents
    # oppose; anything turning by 90 degrees or more is a different
    # feature touching, and is left unpaired for the open-chain guard.
    at_vertex: dict[int, list[tuple[int, int]]] = {}
    for s, segment in enumerate(segments):
        if segment["closed"]:
            continue
        at_vertex.setdefault(segment["head"], []).append((s, 0))
        at_vertex.setdefault(segment["tail"], []).append((s, 1))

    joined: dict[tuple[int, int], tuple[int, int]] = {}
    for _vertex, entries in sorted(at_vertex.items()):
        if len(entries) < 2:
            continue
        scored = []
        for a in range(len(entries)):
            for b in range(a + 1, len(entries)):
                ea, eb = entries[a], entries[b]
                if ea[0] == eb[0]:
                    continue  # would close a segment onto itself
                score = -float(
                    np.dot(_outward(segments[ea[0]], ea[1]), _outward(segments[eb[0]], eb[1]))
                )
                if score > 0.0:
                    scored.append((-score, ea, eb))
        for _neg, ea, eb in sorted(scored):
            if ea in joined or eb in joined:
                continue
            joined[ea] = eb
            joined[eb] = ea

    # Walk the segment graph into chains.  Enter at a free end where
    # there is one — either end, since which of a segment's ends is its
    # head is an artefact of where pass 1 happened to start — and only
    # then pick up whatever is left, which is a cycle.
    chains: list[list[tuple]] = []
    done: set[int] = set()
    starts = [
        (s, side)
        for s in range(len(segments))
        for side in (0, 1)
        if (s, side) not in joined and not segments[s]["closed"]
    ]
    for s, side in starts + [(s, 0) for s in range(len(segments))]:
        if s in done:
            continue
        chain: list[tuple] = []
        cursor: tuple[int, int] | None = (s, side)
        while cursor is not None and cursor[0] not in done:
            index, entry = cursor
            done.add(index)
            run = segments[index]["run"]
            if entry:  # entered through the tail: traverse backwards
                chain.extend((edges[i], not flipped) for i, flipped in reversed(run))
            else:
                chain.extend((edges[i], flipped) for i, flipped in run)
            cursor = joined.get((index, 0 if entry else 1))
        chains.append(chain)
    return chains


def cross_section_polygons(
    shape,
    plane_normal: str,
    plane_position: float,
    deflection: float = 1e-4,
    scale: float = 1.0,
    exact_at_faces: bool = False,
    nudge: float | None = None,
    context: str = "",
) -> list[np.ndarray]:
    """Compute 2D polygon boundaries of a solid intersected with a plane.

    OCC's ``GCPnts_TangentialDeflection`` rejects ``deflection <= 0`` with
    a ``Standard_ConstructionError``; we clamp to ``1e-7`` *in scaled
    model units* (well above OCC's internal tolerance) to keep the
    function robust.  For auto-scaled models the clamp is unreachable —
    the DD-120 scale puts ``deflection * scale`` far above it.

    Intersects the OCC shape with an axis-aligned plane and returns the
    boundary curves as tessellated polygons in the plane's local (u, v)
    coordinate system.

    Parameters
    ----------
    shape : TopoDS_Shape
        An OCC solid (or compound of solids) built at *scale*.
    plane_normal : str
        Normal axis of the cutting plane: ``'x'``, ``'y'``, or ``'z'``.
    plane_position : float
        Position along the normal axis [m].
    deflection : float
        Chordal deflection for curve tessellation [m].
    scale : float
        DD-120 model scale factor of the shape.
    exact_at_faces : bool
        Handle a plane that lies *in* a planar face of the solid.  The
        section operator is degenerate there — on a Boolean result it
        returns only part of the seam structure, which shows up as
        missing material — so with this set such planes are answered by
        a face-face Boolean instead.  Off by default: the check plus the
        heavier Boolean are wasted work for callers that never sample a
        plane on a face, such as the mesher's cell-centre planes.
    nudge : float, optional
        Step length of the degeneracy-escape ladder [m] — how far the
        plane may be re-taken when the section comes back with an open
        chain (see the closedness contract below).  Defaults to
        *deflection*, which is only sound while the two happen to be
        comparable.  They are not: *deflection* is a chordal-accuracy
        budget and shrinks as far as the caller wants its tessellation
        to be faithful, whereas the escape distance has to clear a
        near-tangency band whose width is set by the geometry.  A
        caller sampling a grid should pass a fraction of its cell size
        — the position shift stays negligible against the cell it books
        into, and the ladder keeps the reach it needs.
    context : str
        Caller-supplied identification of *shape*, quoted in the
        open-chain warning so the user can tell which body of a scene
        is affected.

    Returns
    -------
    list of np.ndarray
        One closed contour per array, shape ``(N, 2)`` of (u, v) vertex
        coordinates [m], outer boundaries and holes mixed together.
        **Winding direction carries no meaning**: it follows whatever
        the kernel produced and is not consistent between axes (the
        ``y`` frame ``(u, v) = (x, z)`` is left-handed about its own
        normal, so contours come out mirrored there).  Consumers tell
        solid from hole by the even-odd rule — a point is inside the
        region iff it lies within an odd number of contours — which is
        what ``classify_cells_from_cross_sections`` and the geometry
        plot both use.  The signed-area consumer
        (``compute_face_material_areas``) additionally relies on the
        kernel's habit of pairing opposite windings on degenerate
        tangency bands (their contributions cancel); every returned
        contour is guaranteed CLOSED — open section chains mark a
        degenerate plane and are re-taken a nudge away, never
        implicitly closed (see ``_tessellate`` below).
    """
    plane_position = plane_position * scale
    nudge_step = (deflection if nudge is None else nudge) * scale
    deflection = max(deflection * scale, 1e-7)

    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve  # noqa: PLC0415
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section  # noqa: PLC0415
        from OCC.Core.BRepTools import BRepTools_WireExplorer  # noqa: PLC0415
        from OCC.Core.GCPnts import GCPnts_TangentialDeflection  # noqa: PLC0415
        from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_REVERSED  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for cross-section extraction. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    # Plane normal → gp_Dir and (u, v) axis indices
    _normal_map = {
        "x": (gp_Dir(1, 0, 0), 1, 2),  # u=y, v=z
        "y": (gp_Dir(0, 1, 0), 0, 2),  # u=x, v=z
        "z": (gp_Dir(0, 0, 1), 0, 1),  # u=x, v=y
    }
    if plane_normal not in _normal_map:
        raise ValueError(f"plane_normal must be 'x', 'y', or 'z'; got {plane_normal!r}")
    normal_dir, u_idx, v_idx = _normal_map[plane_normal]

    axis_idx = {"x": 0, "y": 1, "z": 2}[plane_normal]

    def _chain_of_wire(wire) -> list[tuple]:
        """A traversal chain from a genuine wire (the exact-in-face path)."""
        chain = []
        we = BRepTools_WireExplorer(wire)
        while we.More():
            edge = topods.Edge(we.Current())
            chain.append((edge, edge.Orientation() == TopAbs_REVERSED))
            we.Next()
        return chain

    def _chains_at(position: float) -> tuple[list[list[tuple]], bool]:
        """Section chains at *position* [scaled units]; ``(chains, from_faces)``."""
        origin = [0.0, 0.0, 0.0]
        origin[axis_idx] = position
        pln = gp_Pln(gp_Pnt(*origin), normal_dir)
        if exact_at_faces and _plane_lies_in_a_face(shape, axis_idx, position):
            return [_chain_of_wire(w) for w in _face_region_wires(shape, pln)], True
        # Intersect shape with plane. SetRunParallel enables OCCT's own
        # internal multi-threading for the Boolean-operation kernel — the
        # computed intersection is unaffected, only its execution strategy;
        # must be set before Init1/Init2/Build (the two-arg constructor
        # builds immediately, before the flag could take effect).
        section = BRepAlgoAPI_Section()
        section.SetRunParallel(True)
        keep_operands_intact(section)
        section.Init1(shape)
        section.Init2(pln)
        section.Build()
        if not section.IsDone():
            return [], False

        section_shape = section.Shape()

        # Collect section edges
        edges: list = []
        explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
        while explorer.More():
            edges.append(topods.Edge(explorer.Current()))
            explorer.Next()

        if not edges:
            return [], False

        # Group edges into separate contours (handles disjoint ones, e.g.
        # outer boundary + hole from a CSG difference).
        return _chain_section_edges(edges, u_idx, v_idx), False

    def _tessellate(chains: list) -> tuple[list[np.ndarray], int]:
        """Tessellated closed contours [scaled units] + open-chain count.

        A chain whose tessellated endpoints do not meet is an OPEN
        chain: on a degenerate section (a plane in the near-tangent
        band of a curved face of a tolerance-inflated Boolean solid,
        ``BRepAlgoAPI_Section`` drops edges) the kernel returns
        partial contours, and implicitly closing them books arbitrary
        coverage — measured: one 13-point chain spanning both coax
        bores of a stripline coupler, booking the complement of a
        bore-wall face and breaking the feed-chain slab invariance
        behind the port.  Closedness is judged against the
        tessellation scale and the contour perimeter, both far above
        genuine vertex-tolerance seams.
        """
        closed: list[np.ndarray] = []
        n_open = 0
        for chain in chains:
            points: list[tuple[float, float]] = []
            for edge, flipped in chain:
                curve = BRepAdaptor_Curve(edge)
                discretizer = GCPnts_TangentialDeflection(
                    curve,
                    math.radians(5.0),
                    deflection,
                )
                n_pts = discretizer.NbPoints()
                rng = range(n_pts, 0, -1) if flipped else range(1, n_pts + 1)
                for k in rng:
                    pnt = discretizer.Value(k)
                    coords = (pnt.X(), pnt.Y(), pnt.Z())
                    uv = (coords[u_idx], coords[v_idx])
                    if not points or (
                        abs(uv[0] - points[-1][0]) > 1e-12 or abs(uv[1] - points[-1][1]) > 1e-12
                    ):
                        points.append(uv)

            if len(points) < 3:
                continue
            arr = np.array(points)
            gap = float(np.hypot(*(arr[0] - arr[-1])))
            seg = np.hypot(*(arr[1:] - arr[:-1]).T)
            perimeter = float(seg.sum()) + gap
            if gap > max(8.0 * deflection, 5e-2 * perimeter):
                n_open += 1
                continue
            # Remove closing duplicate
            if gap < 1e-10:
                arr = arr[:-1]
            if len(arr) >= 3:
                closed.append(arr)
        return closed, n_open

    # Nudge-retry: an open chain marks the section itself as degenerate
    # at this position — re-take it a few *nudge steps* away
    # (deterministic sequence, reaching 8 steps out) rather than booking
    # a fantasy contour.  The step is the caller's to choose, and the
    # reach it buys is the whole point: too short and a near-tangency
    # band cannot be left at all, too long and the section answers about
    # a plane the caller never asked about.  A grid caller wants the
    # far end of the ladder to stay inside one cell.
    #
    # A nudge that comes back EMPTY where
    # the un-nudged section saw material is rejected too: it stepped
    # clear off a feature thinner than the nudge (a sub-tessellation
    # sliver), which would silently erase it.  The exact-in-face path
    # keeps its position semantics: no nudge, open chains are dropped.
    polygons_scaled: list[np.ndarray] = []
    base_closed: list[np.ndarray] = []
    base_seen_material = False
    base_open = 0
    for step in (0.0, 4.0, -4.0, 8.0, -8.0):
        chains, from_faces = _chains_at(plane_position + step * nudge_step)
        polygons_scaled, n_open = _tessellate(chains)
        if step == 0.0:
            base_closed = polygons_scaled
            base_open = n_open
            base_seen_material = bool(polygons_scaled) or n_open > 0
        if from_faces:
            break
        if n_open == 0 and (polygons_scaled or not base_seen_material):
            break
    else:
        warnings.warn(
            f"cross-section of {context or 'a solid'} at "
            f"{plane_normal}={plane_position / scale:.6g} m: {base_open} open "
            f"section chain(s) survived every retry out to "
            f"{8.0 * nudge_step / scale:.3g} m either side, and are dropped "
            f"rather than closed by guesswork — so that solid's coverage on "
            f"this one plane is incomplete ({len(base_closed)} closed contour(s) "
            f"kept). The plane is degenerate here: it grazes a curved face of a "
            f"Boolean solid, and such a near-tangency band is often wider than "
            f"the retry reach. A material boundary falling in the band keeps its "
            f"bulk classification but loses its sub-cell resolution there. "
            f"Changing the cell size near this plane moves it clear of the band.",
            UserWarning,
            stacklevel=2,
        )
        polygons_scaled = base_closed
        n_open = 0
    if n_open and from_faces:
        warnings.warn(
            f"cross-section of {context or 'a solid'} at "
            f"{plane_normal}={plane_position / scale:.6g} m: {n_open} open "
            f"face-region chain(s) dropped on the exact-in-face path.",
            UserWarning,
            stacklevel=2,
        )

    # Back to meters in one bulk divide (lossless: scale is a power of
    # two; identity for scale = 1).
    return [arr / scale for arr in polygons_scaled]


class _PlanarSectionEngine:
    """Exact axis-aligned plane sections of planar-faced solid regions.

    Answers the same question as :func:`cross_section_polygons` — the
    boundary polygons of ``shape`` ∩ plane in the plane's (u, v) frame —
    but without the per-plane Boolean, whose fixed cost (~1 ms per
    touched face pair in ``BRepAlgoAPI_Section``) dominates structured-
    grid meshing where thousands of parallel planes query the same
    solid.  Faces, straight edges and their adjacency are collected
    once; each section is then assembled from the exact edge/plane
    intersection points:

    - candidate faces/edges via bounding-box slabs (as in
      :class:`_PrefilteredLineSolid`),
    - one exact intersection point per transversally crossed straight
      edge,
    - per face, points sorted along the face/plane intersection line
      and paired by parity into segments directed along
      ``n_plane × n_outward`` (outer contours and holes are thereby
      consistently counter-rotating, the only orientation property the
      area accounting relies on),
    - segments stitched into closed chains through shared edge
      indices — exact, no tolerance matching, because both adjacent
      faces reference the *same* intersection point of their common
      edge.

    The fast path only fires when every candidate face is planar with
    a well-defined outward normal, every candidate edge is a straight
    line crossed transversally, and no vertex lies on the plane
    (within tolerance).  Everything else — curved faces or edges,
    tangencies, planes through vertices or coplanar faces (the DD-087
    degenerate planes) — makes :meth:`section` return ``None`` and must
    be delegated to :func:`cross_section_polygons`, which keeps every
    boundary-case semantic identical to the previous behaviour.
    """

    #: (u, v) axis indices per plane-normal axis, matching
    #: ``cross_section_polygons``'s ``_normal_map``.
    _UV = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def __init__(self, shape, scale: float = 1.0) -> None:
        self.enabled = False
        #: DD-120 model scale of the shape: the engine's internal arrays
        #: live entirely in scaled units; ``can_fast``/``section`` take
        #: meter positions and return meter polygons.
        self._scale = float(scale)
        #: Face count of the shape (0 until known) — the cost weight of
        #: one delegated OCC section for the pool-work estimate.
        self.face_count = 0
        try:
            self._build(shape)
        except Exception:  # noqa: BLE001 — fall back to the OCC path
            self.enabled = False

    def _build(self, shape) -> None:
        from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
        from OCC.Core.BRep import BRep_Tool  # noqa: PLC0415
        from OCC.Core.BRepAdaptor import (  # noqa: PLC0415
            BRepAdaptor_Curve,
            BRepAdaptor_Surface,
        )
        from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415
        from OCC.Core.GeomAbs import (  # noqa: PLC0415
            GeomAbs_Line,
            GeomAbs_Plane,
        )
        from OCC.Core.TopAbs import (  # noqa: PLC0415
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_FORWARD,
            TopAbs_REVERSED,
            TopAbs_VERTEX,
        )
        from OCC.Core.TopExp import topexp  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
        from OCC.Core.TopTools import (  # noqa: PLC0415
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_IndexedMapOfShape,
        )

        fmap = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_FACE, fmap)
        n_f = fmap.Size()
        self.face_count = n_f
        if n_f == 0:
            return

        f_planar = np.zeros(n_f, dtype=np.bool_)
        f_n = np.zeros((n_f, 3), dtype=np.float64)
        f_d = np.zeros(n_f, dtype=np.float64)
        f_lo = np.empty((n_f, 3), dtype=np.float64)
        f_hi = np.empty((n_f, 3), dtype=np.float64)
        for i in range(n_f):
            face = topods.Face(fmap.FindKey(i + 1))
            box = Bnd_Box()
            # Geometry-only box (no triangulation) — KB-012.
            brepbndlib.Add(face, box, False)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            f_lo[i] = (xmin, ymin, zmin)
            f_hi[i] = (xmax, ymax, zmax)
            ori = face.Orientation()
            if ori not in (TopAbs_FORWARD, TopAbs_REVERSED):
                continue
            surf = BRepAdaptor_Surface(face, False)
            if surf.GetType() != GeomAbs_Plane:
                continue
            pln = surf.Plane()
            # Parametric normal XDir × YDir (NOT Axis().Direction():
            # an indirect gp_Ax3 flips the latter against the
            # parametrisation the orientation convention refers to).
            xd = pln.Position().XDirection()
            yd = pln.Position().YDirection()
            n = np.cross(
                (xd.X(), xd.Y(), xd.Z()),
                (yd.X(), yd.Y(), yd.Z()),
            )
            n /= np.linalg.norm(n)
            if ori == TopAbs_REVERSED:
                n = -n
            loc = pln.Location()
            f_planar[i] = True
            f_n[i] = n
            f_d[i] = n[0] * loc.X() + n[1] * loc.Y() + n[2] * loc.Z()

        emap = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, emap)
        n_e = emap.Size()
        e_ok = np.zeros(n_e, dtype=np.bool_)
        e_p1 = np.zeros((n_e, 3), dtype=np.float64)
        e_p2 = np.zeros((n_e, 3), dtype=np.float64)
        e_f = np.zeros((n_e, 2), dtype=np.int64)
        e_lo = np.empty((n_e, 3), dtype=np.float64)
        e_hi = np.empty((n_e, 3), dtype=np.float64)
        max_tol = 1e-12
        for i in range(n_e):
            edge = topods.Edge(emap.FindKey(i + 1))
            # The bounding box must be valid for EVERY edge — it is what
            # routes planes near curved/degenerate edges to the OCC path.
            box = Bnd_Box()
            # Geometry-only box (no triangulation) — KB-012.
            brepbndlib.Add(edge, box, False)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            e_lo[i] = (xmin, ymin, zmin)
            e_hi[i] = (xmax, ymax, zmax)
            fids = []
            it = emap.FindFromIndex(i + 1)
            for f in it:
                fi = fmap.FindIndex(f) - 1
                if fi not in fids:
                    fids.append(fi)
            if len(fids) != 2:
                continue
            if BRep_Tool.Degenerated(edge):
                continue
            max_tol = max(max_tol, BRep_Tool.Tolerance(edge))
            curve = BRepAdaptor_Curve(edge)
            if curve.GetType() != GeomAbs_Line:
                continue
            p1 = curve.Value(curve.FirstParameter())
            p2 = curve.Value(curve.LastParameter())
            e_ok[i] = True
            e_p1[i] = (p1.X(), p1.Y(), p1.Z())
            e_p2[i] = (p2.X(), p2.Y(), p2.Z())
            e_f[i] = fids
        vex = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_VERTEX, vex)
        for i in range(vex.Size()):
            max_tol = max(
                max_tol,
                BRep_Tool.Tolerance(topods.Vertex(vex.FindKey(i + 1))),
            )

        self._f_planar = f_planar
        self._f_n = f_n
        self._f_d = f_d
        self._f_lo = f_lo
        self._f_hi = f_hi
        self._e_ok = e_ok
        self._e_p1 = e_p1
        self._e_p2 = e_p2
        self._e_f = e_f
        self._e_lo = e_lo
        self._e_hi = e_hi
        # On-plane tolerance: anchored at the shape's own vertex/edge
        # tolerances so any plane OCC might resolve by tolerance gluing
        # is delegated to the OCC path instead of answered exactly.
        self._tol = 2.0 * max_tol
        self.enabled = True

    def _screen(self, axis: int, pos: float):
        """Common fast-path admission checks.

        Returns ``(d1, d2)`` signed vertex/plane distances of the
        straight candidate edges, or ``None`` when the plane must be
        delegated (curved candidate face/edge, coplanar face, vertex on
        the plane).
        """
        tol = self._tol
        cf = (self._f_lo[:, axis] <= pos + tol) & (self._f_hi[:, axis] >= pos - tol)
        if not self._f_planar[cf].all():
            return None
        par = cf & (np.abs(self._f_n[:, axis]) >= 1.0 - 1e-9)
        if par.any() and (np.abs(pos * self._f_n[par, axis] - self._f_d[par]) <= tol).any():
            return None
        ce = (self._e_lo[:, axis] <= pos + tol) & (self._e_hi[:, axis] >= pos - tol)
        if not self._e_ok[ce].all():
            return None
        d1 = self._e_p1[:, axis] - pos
        d2 = self._e_p2[:, axis] - pos
        if (np.abs(d1[ce]) <= tol).any() or (np.abs(d2[ce]) <= tol).any():
            return None
        return d1, d2

    def can_fast(self, axis: int, pos: float) -> bool:
        """Whether :meth:`section` will (barring stitch anomalies)
        answer this plane (position in meters) without delegating."""
        return self._screen(axis, pos * self._scale) is not None

    def section(self, axis: int, pos: float) -> list[np.ndarray] | None:
        """Section polygons [m] for the plane at *pos* [m], or ``None``
        to delegate to the OCC path."""
        pos = pos * self._scale
        screened = self._screen(axis, pos)
        if screened is None:
            return None
        d1, d2 = screened
        trans = np.nonzero(self._e_ok & (d1 * d2 < 0.0))[0]
        if trans.size == 0:
            return []
        t = d1[trans] / (d1[trans] - d2[trans])
        pts = self._e_p1[trans] + t[:, None] * (self._e_p2[trans] - self._e_p1[trans])
        pts[:, axis] = pos
        u_idx, v_idx = self._UV[axis]
        uv = np.ascontiguousarray(pts[:, (u_idx, v_idx)])

        face_pts: dict[int, list[int]] = {}
        for k, ei in enumerate(trans):
            face_pts.setdefault(self._e_f[ei, 0], []).append(k)
            face_pts.setdefault(self._e_f[ei, 1], []).append(k)

        n_axis = np.zeros(3)
        n_axis[axis] = 1.0
        succ = np.full(uv.shape[0], -1, dtype=np.int64)
        for fi, ks in face_pts.items():
            if len(ks) % 2:
                return None
            t3 = np.cross(n_axis, self._f_n[fi])
            norm = np.linalg.norm(t3)
            if norm < 1e-12:
                return None
            t2 = (t3[u_idx], t3[v_idx])
            s = uv[ks] @ np.asarray(t2)
            order = np.argsort(s, kind="stable")
            s_sorted = s[order]
            if (np.diff(s_sorted) <= self._tol).any():
                return None
            for j in range(0, len(ks), 2):
                a = ks[order[j]]
                b = ks[order[j + 1]]
                if succ[a] != -1:
                    return None
                succ[a] = b
        if (succ == -1).any():
            return None
        counts = np.bincount(succ, minlength=uv.shape[0])
        if (counts != 1).any():
            return None

        polygons: list[np.ndarray] = []
        visited = np.zeros(uv.shape[0], dtype=np.bool_)
        for start in range(uv.shape[0]):
            if visited[start]:
                continue
            chain = []
            k = start
            while not visited[k]:
                visited[k] = True
                chain.append(k)
                k = succ[k]
            if k != start:
                return None
            points = [(float(uv[c, 0]), float(uv[c, 1])) for c in chain]
            # Consecutive-duplicate and closing-duplicate filtering,
            # matching cross_section_polygons.
            filtered: list[tuple[float, float]] = []
            for p in points:
                if not filtered or (
                    abs(p[0] - filtered[-1][0]) > 1e-12 or abs(p[1] - filtered[-1][1]) > 1e-12
                ):
                    filtered.append(p)
            if len(filtered) >= 3:
                if (
                    abs(filtered[0][0] - filtered[-1][0]) < 1e-10
                    and abs(filtered[0][1] - filtered[-1][1]) < 1e-10
                ):
                    filtered.pop()
                if len(filtered) >= 3:
                    # Back to meters (lossless power-of-two divide).
                    polygons.append(np.array(filtered) / self._scale)
        return polygons


# ---------------------------------------------------------------------------
# Batch cross-section computation for conformal material matrices
# ---------------------------------------------------------------------------


def batch_cross_sections(
    shapes_with_material: list[tuple[object, int]],
    plane_positions: dict[str, np.ndarray],
    deflection: float = 1e-4,
    scale: float = 1.0,
    nudge: float | None = None,
    material_library: dict | None = None,
) -> dict[tuple[str, int], list[tuple[int, list[np.ndarray]]]]:
    """Pre-compute cross-section polygons for multiple shapes at grid planes.

    For each (axis, plane_index) pair that intersects at least one shape,
    returns a list of ``(material_id, polygons)`` entries.

    Uses per-shape bounding boxes to skip planes that don't intersect.

    Parameters
    ----------
    shapes_with_material : list of (shape_obj, material_id)
        Each ``shape_obj`` must have ``._occ_shape()`` and ``.bounding_box()``
        methods.
    plane_positions : dict
        ``{'x': np.ndarray, 'y': ..., 'z': ...}`` — plane positions per axis.
    deflection : float
        Chordal deflection for polygon tessellation [m].
    nudge : float or None, default None
        Degeneracy-escape step handed to
        :func:`cross_section_polygons`; ``None`` leaves it tied to
        *deflection*.
    material_library : dict or None, default None
        Used only to name the solid in an open-chain warning.

    Returns
    -------
    dict
        ``{(axis, plane_idx): [(material_id, [polygons]), ...]}``
        Only entries with at least one polygon are included.
    """
    _axis_to_bbox_idx = {"x": (0, 3), "y": (1, 4), "z": (2, 5)}
    _axis_to_int = {"x": 0, "y": 1, "z": 2}
    cache: dict[tuple[str, int], list[tuple[int, list[np.ndarray]]]] = {}

    for shape_obj, mat_id in shapes_with_material:
        occ_shape = shape_obj._occ_shape(scale)
        name = getattr((material_library or {}).get(mat_id), "name", mat_id)
        context = f"the {name!r} solid"
        engine = _PlanarSectionEngine(occ_shape, scale=scale)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = shape_obj.bounding_box(scale)
        bbox_flat = (xmin, ymin, zmin, xmax, ymax, zmax)

        for axis, positions in plane_positions.items():
            lo_idx, hi_idx = _axis_to_bbox_idx[axis]
            lo = bbox_flat[lo_idx]
            hi = bbox_flat[hi_idx]

            for idx, pos in enumerate(positions):
                if pos < lo - deflection or pos > hi + deflection:
                    continue  # plane outside this shape's bounding box

                polys = engine.section(_axis_to_int[axis], float(pos)) if engine.enabled else None
                if polys is None:
                    polys = cross_section_polygons(
                        occ_shape,
                        axis,
                        float(pos),
                        deflection=deflection,
                        scale=scale,
                        nudge=nudge,
                        context=context,
                    )
                if polys:
                    key = (axis, idx)
                    if key not in cache:
                        cache[key] = []
                    cache[key].append((mat_id, polys))

    return cache


# ---------------------------------------------------------------------------
# Effective PEC solid construction (for 3D Dey-Mittra)
# ---------------------------------------------------------------------------


def build_effective_pec_solid(
    shapes_with_material: list[tuple[object, int]],
    material_library: dict,
    scale: float = 1.0,
):
    """Build the effective PEC solid respecting last-wins CSG ordering.

    Processes shapes in reverse priority order (last shape wins) and
    constructs an OCC solid representing the final PEC region after all
    material overrides are applied.

    Parameters
    ----------
    shapes_with_material : list of (shape_obj, material_id)
        Ordered list from lowest to highest priority.  Each shape_obj
        must have ``._occ_shape()`` returning a ``TopoDS_Shape``.
    material_library : dict
        ``{int: Material}`` mapping with ``is_pec`` attribute.

    Returns
    -------
    TopoDS_Shape or None
        Fused PEC solid, or ``None`` if no PEC region exists.
    """
    effective_pec = None
    higher_coverage = None  # union of all shapes with higher priority

    for shape_obj, mat_id in reversed(shapes_with_material):
        occ_shape = shape_obj._occ_shape(scale)
        is_pec = material_library[mat_id].is_pec

        if is_pec:
            # This shape's PEC contribution = its volume minus any
            # higher-priority shapes (which override it).
            if higher_coverage is not None:
                contribution = boolean_difference(occ_shape, higher_coverage)
            else:
                contribution = occ_shape

            if effective_pec is None:
                effective_pec = contribution
            else:
                effective_pec = boolean_union([effective_pec, contribution])

        # Add this shape to the higher-priority coverage for lower shapes
        if higher_coverage is None:
            higher_coverage = occ_shape
        else:
            higher_coverage = boolean_union([higher_coverage, occ_shape])

    return effective_pec


# ---------------------------------------------------------------------------
# 3D line-solid intersection for Dey-Mittra edge fractions (f_L)
# ---------------------------------------------------------------------------

# Per-call diagnostics of the face-bbox prefilter (overwritten by each
# ``compute_edge_pec_fractions`` call): edges processed, edges whose
# states could not be derived from crossing transitions, and midpoints
# that needed the O(faces) full-solid classifier as last resort.
_EDGE_FRACTION_STATS = {"edges": 0, "fallback_edges": 0, "classifier_points": 0}

# Probe directions of last resort for ``_PrefilteredLineSolid.point_state``
# after the cached axis-aligned probe lines: fixed oblique unit vectors
# that avoid tangencies with axis-aligned geometry.
_OBLIQUE_PROBES = (
    (0.7548776662466927, 0.5698402909980532, 0.32471795724474605),
    (-0.32471795724474605, 0.7548776662466927, -0.5698402909980532),
)


class _PrefilteredLineSolid:
    """Face-bbox prefiltered line-vs-solid intersections.

    Both ``BRepIntCurveSurface_Inter.Init(shape, ...)`` and
    ``BRepClass3d_SolidClassifier.Perform`` scan every face of the shape
    on every call, so per-edge queries against a fused solid cost O(faces)
    each.  This helper collects the faces and their bounding boxes once;
    a query line is then intersected only against the faces whose boxes
    its parameter window overlaps, through per-face
    ``IntCurvesFace_Intersector`` objects that are built once and cached
    (their construction digests the face restriction, which is the
    expensive part for faces with many wires).

    Inside/outside states are derived from crossing transitions instead
    of per-point classification.  Any hit that cannot be trusted for
    that bookkeeping (hit on a face border, i.e. ``State() !=
    TopAbs_IN``, or an orientation other than FORWARD/REVERSED) marks
    the query as not clean; the caller falls back to point
    classification.
    """

    def __init__(self, solid, tolerance: float):
        from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
        from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415
        from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt  # noqa: PLC0415
        from OCC.Core.IntCurvesFace import IntCurvesFace_Intersector  # noqa: PLC0415
        from OCC.Core.IntCurveSurface import (  # noqa: PLC0415
            IntCurveSurface_In,
            IntCurveSurface_Out,
        )
        from OCC.Core.TopAbs import (  # noqa: PLC0415
            TopAbs_FACE,
            TopAbs_FORWARD,
            TopAbs_IN,
            TopAbs_REVERSED,
        )
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415

        self._gp_Dir = gp_Dir
        self._gp_Lin = gp_Lin
        self._gp_Pnt = gp_Pnt
        self._tol = float(tolerance)
        self._Intersector = IntCurvesFace_Intersector
        self._IN = IntCurveSurface_In
        self._OUT = IntCurveSurface_Out
        self._TopAbs_IN = TopAbs_IN

        faces = []
        ori_ok: list[bool] = []
        lo: list[tuple[float, float, float]] = []
        hi: list[tuple[float, float, float]] = []
        exp = TopExp_Explorer(solid, TopAbs_FACE)
        while exp.More():
            face = topods.Face(exp.Current())
            box = Bnd_Box()
            # Geometry-only box (no triangulation) — KB-012.
            brepbndlib.Add(face, box, False)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            faces.append(face)
            ori_ok.append(face.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED))
            lo.append((xmin, ymin, zmin))
            hi.append((xmax, ymax, zmax))
            exp.Next()
        self._faces = faces
        self._ori_ok = ori_ok
        # Per-face intersectors, built lazily on first use: construction
        # digests the face restriction (all wires) once, so queries that
        # keep touching a complex face — a plate pierced by hundreds of
        # holes — stay cheap afterwards.
        self._face_ints: list = [None] * len(faces)
        if faces:
            self._flo = np.asarray(lo)
            self._fhi = np.asarray(hi)
        else:
            self._flo = np.empty((0, 3))
            self._fhi = np.empty((0, 3))

    def _line_candidates(self, p0, direction):
        """Faces whose bounding box the line touches, with the parameter
        interval [w_in, w_out] over which the line stays inside each box
        (slab method, vectorised over all faces).

        Returns ``(idx, w_in, w_out)`` sorted by ascending ``w_in``.
        """
        tol = self._tol
        w_in = np.full(len(self._faces), -np.inf)
        w_out = np.full(len(self._faces), np.inf)
        for ax in range(3):
            lo = self._flo[:, ax] - tol
            hi = self._fhi[:, ax] + tol
            d = direction[ax]
            if abs(d) > 1e-300:
                t1 = (lo - p0[ax]) / d
                t2 = (hi - p0[ax]) / d
                np.maximum(w_in, np.minimum(t1, t2), out=w_in)
                np.minimum(w_out, np.maximum(t1, t2), out=w_out)
            else:
                inside = (lo <= p0[ax]) & (p0[ax] <= hi)
                w_in[~inside] = np.inf
                w_out[~inside] = -np.inf
        keep = np.nonzero(w_in <= w_out)[0]
        order = np.argsort(w_in[keep], kind="stable")
        idx = keep[order]
        return idx, w_in[idx], w_out[idx]

    def _intersect_faces(self, face_indices, line, w_lo, w_hi):
        """Intersections of ``line`` with the given faces, W in
        [w_lo, w_hi].  Returns ``(hits, clean)``: ``hits`` is a list of
        ``(w, step)`` with step ``+1`` entering the solid along +W,
        ``-1`` leaving, ``0`` tangential or untrusted; ``clean`` is
        False when any hit cannot anchor transition bookkeeping.

        ``IntCurvesFace_Intersector`` reports transitions relative to
        the *oriented* face (verified against the whole-shape
        intersector: identical W and state, orientation-resolved
        transition), so no orientation correction is needed here.
        """
        hits: list[tuple[float, int]] = []
        clean = True
        face_ints = self._face_ints
        for fi in face_indices:
            it = face_ints[fi]
            if it is None:
                it = self._Intersector(self._faces[fi], self._tol)
                face_ints[fi] = it
            it.Perform(line, -1e100, 1e100)
            ok = self._ori_ok[fi]
            for ip in range(1, it.NbPnt() + 1):
                w = it.WParameter(ip)
                if w_lo <= w <= w_hi:
                    if not ok or it.State(ip) != self._TopAbs_IN:
                        clean = False
                        hits.append((w, 0))
                    else:
                        tr = it.Transition(ip)
                        if tr == self._IN:
                            step = 1
                        elif tr == self._OUT:
                            step = -1
                        else:
                            step = 0
                        hits.append((w, step))
        return hits, clean

    def window_hits(self, candidates, line, length):
        """Hits within the tolerance-inflated segment window [—tol,
        length + tol], given the line's candidate triple.  Returns
        ``(hits, clean)`` as in ``_intersect_faces``."""
        tol = self._tol
        idx, w_in, w_out = candidates
        m = (w_in <= length + tol) & (w_out >= -tol)
        if not m.any():
            return [], True
        return self._intersect_faces(idx[m], line, -tol, length + tol)

    def _nearest_beyond(self, candidates, line, w_from, forward: bool):
        """Nearest crossing strictly beyond ``w_from`` along the line
        (towards +W when ``forward``, else towards -W).

        Walks the candidate faces in bbox-interval order with an early
        break once no closer face can host a hit.  Returns ``(w, step)``
        of the nearest hit — step ``0`` marks it untrusted, and a
        trusted hit with an untrusted one within tolerance is demoted to
        untrusted — or ``None`` when the line reaches the outside of
        every face box without any hit (guaranteed outside there).
        """
        idx, w_in, w_out = candidates
        tol = self._tol
        if forward:
            m = np.nonzero(w_out >= w_from)[0]
            near = w_in[m]
        else:
            sel = np.nonzero(w_in <= w_from)[0]
            order = np.argsort(-w_out[sel], kind="stable")
            m = sel[order]
            near = -w_out[m]
            w_from = -w_from
        faces_sorted = idx[m]
        best_w = np.inf
        best_step = 0
        found = False
        pos = 0
        n_cand = len(faces_sorted)
        # Walk faces nearest-first in small groups: each group shares one
        # intersector Init (its fixed cost dominates single-face calls).
        while pos < n_cand:
            if found and near[pos] > best_w + tol:
                break
            end = min(pos + 8, n_cand)
            hits, _clean = self._intersect_faces(
                faces_sorted[pos:end],
                line,
                -np.inf,
                np.inf,
            )
            for w, step in hits:
                if not forward:
                    w = -w
                if w <= w_from + tol:
                    continue
                if not found or w < best_w - tol:
                    best_w = w
                    best_step = step
                    found = True
                elif w <= best_w + tol and (step == 0 or step != best_step):
                    # Ambiguous pile-up at the same parameter
                    best_step = 0
            pos = end
        if not found:
            return None
        if not forward:
            best_w = -best_w
        return best_w, best_step

    def uniform_state(self, candidates, line, length):
        """Outside/inside state of a segment with no crossing inside its
        tolerance-inflated window: True = outside, False = inside, None
        = undetermined (fall back to point classification)."""
        res = self._nearest_beyond(candidates, line, length + self._tol, True)
        if res is None:
            return True
        if res[1] != 0:
            return res[1] > 0
        res = self._nearest_beyond(candidates, line, -self._tol, False)
        if res is None:
            return True
        if res[1] != 0:
            return res[1] < 0
        return None

    def point_state(self, point, probe_dirs):
        """Classify a point against the solid via local ray probes.

        Casts rays along ``probe_dirs`` in order; the first direction
        whose nearest hit is trustworthy decides.  A hit within the
        tolerance of the ray origin means the point sits on the solid
        boundary — reported as inside, matching the ``TopAbs_ON``
        convention of ``BRepClass3d_SolidClassifier``.  Returns True =
        outside, False = inside (or on the boundary), None when every
        probe direction was spoiled by untrusted hits.
        """
        tol = self._tol
        for d in probe_dirs:
            dvec = np.asarray(d, dtype=np.float64)
            line = self._gp_Lin(
                self._gp_Pnt(float(point[0]), float(point[1]), float(point[2])),
                self._gp_Dir(float(dvec[0]), float(dvec[1]), float(dvec[2])),
            )
            cand = self._line_candidates(point, dvec)
            res = self._nearest_beyond(cand, line, -2.0 * tol, True)
            if res is None:
                return True
            w, step = res
            if step == 0:
                continue
            if w <= tol:
                return False
            return step > 0
        return None


def compute_edge_pec_fractions(
    pec_shapes: list,
    edges: np.ndarray,
    tolerance: float = 1e-8,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute f_L (fraction outside PEC) for edges via 3D line-solid intersection.

    For each edge, intersects the line segment with the fused PEC solid
    to find surface crossing parameters, then derives the inside/outside
    status of each resulting sub-segment from the orientation-corrected
    crossing transitions.  Faces are prefiltered per edge through their
    bounding boxes (see ``_PrefilteredLineSolid``), so the per-edge cost
    scales with the local face density instead of the total face count
    of the fused solid.  Edges whose crossings cannot anchor transition
    bookkeeping (tangential or border hits, inconsistent transitions)
    fall back to classifying each sub-segment midpoint against the full
    solid, which reproduces the pre-prefilter behaviour exactly.

    Parameters
    ----------
    pec_shapes : list
        OCC ``TopoDS_Shape`` objects representing PEC solids, built at
        *scale*.
    edges : np.ndarray
        Shape ``(N, 2, 3)`` — ``edges[i, 0]`` is the start point,
        ``edges[i, 1]`` is the end point (in metres).
    tolerance : float
        Geometric tolerance for intersection and point classification [m].
    scale : float
        DD-120 model scale factor of the PEC shapes.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, values in [0, 1].  ``f_L = 1`` means entirely
        outside PEC, ``f_L = 0`` means entirely inside PEC.
    """
    try:
        from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier  # noqa: PLC0415
        from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for 3D edge-PEC intersection. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    n_edges = len(edges)
    f_L = np.ones(n_edges, dtype=np.float64)

    _EDGE_FRACTION_STATS["edges"] = n_edges
    _EDGE_FRACTION_STATS["fallback_edges"] = 0
    _EDGE_FRACTION_STATS["classifier_points"] = 0

    if not pec_shapes or n_edges == 0:
        return f_L

    # Into scaled model units in one bulk multiply (f_L is a fraction,
    # so nothing needs converting back).  `edges * 1.0` keeps the s = 1
    # path bit-identical.
    edges = np.asarray(edges) * scale
    tolerance = tolerance * scale

    # Fuse PEC shapes into a single solid for clean face topology
    # (single N-ary Boolean pass; see boolean_union).
    pec_solid = boolean_union(pec_shapes)

    prefiltered = _PrefilteredLineSolid(pec_solid, tolerance)
    # Full-solid point classifier, built only when some edge needs the
    # fallback path (its Perform scans every face — O(faces) per call).
    classifier = None

    # Mesh edges are collinear in droves: every edge of a structured
    # grid lies on one of comparatively few carrier lines.  The full
    # crossing structure of such a line — candidate faces, transversal
    # crossing parameters, and whether transitions alternate cleanly —
    # is computed once and shared by every edge (and probe ray) on it.
    # Parity along the line is anchored for free: the line enters from
    # outside the solid, so the state before its first crossing is
    # "outside".
    line_cache: dict[tuple[int, float, float], dict] = {}

    def line_entry(ax: int, point) -> dict:
        key = (ax, float(point[(ax + 1) % 3]), float(point[(ax + 2) % 3]))
        entry = line_cache.get(key)
        if entry is not None:
            return entry
        origin = np.array(
            [float(point[0]), float(point[1]), float(point[2])],
        )
        dvec = np.zeros(3)
        dvec[ax] = 1.0
        axis_line = gp_Lin(
            gp_Pnt(origin[0], origin[1], origin[2]),
            gp_Dir(dvec[0], dvec[1], dvec[2]),
        )
        cand = prefiltered._line_candidates(origin, dvec)
        if cand[0].size:
            line_hits, line_clean = prefiltered._intersect_faces(
                cand[0],
                axis_line,
                -np.inf,
                np.inf,
            )
        else:
            line_hits, line_clean = [], True
        crossings = sorted((w, s) for w, s in line_hits if s != 0)
        ok = line_clean
        prev_step = 0
        for _w, step in crossings:
            if step == prev_step:
                ok = False
                break
            prev_step = step
        if crossings and crossings[0][1] != 1:
            # A bounded solid must be entered before it can be left.
            ok = False
        entry = {
            "origin_ax": origin[ax],
            "cand": cand,
            "ws": [w for w, _ in crossings],
            "ok": ok,
        }
        line_cache[key] = entry
        return entry

    def classify_point(point, axes_order):
        """Old-classifier semantics from cached carrier lines: a point
        within tolerance of a crossing sits on the boundary (``ON`` →
        PEC side, i.e. inside); otherwise crossing parity decides.
        Returns True = outside, False = inside, None = every probe line
        was untrusted."""
        for ax_probe in axes_order:
            probe = line_entry(int(ax_probe), point)
            if not probe["ok"]:
                continue
            w_rel = float(point[int(ax_probe)]) - probe["origin_ax"]
            ws = probe["ws"]
            pos = bisect.bisect_left(ws, w_rel - tolerance)
            if pos < len(ws) and ws[pos] <= w_rel + tolerance:
                return False
            return bisect.bisect(ws, w_rel) % 2 == 0
        return prefiltered.point_state(point, _OBLIQUE_PROBES)

    for i in range(n_edges):
        p0 = edges[i, 0]
        p1 = edges[i, 1]
        dx = p1 - p0
        length = float(np.linalg.norm(dx))

        if length < tolerance:
            # Degenerate edge — treat as PEC
            f_L[i] = 0.0
            continue

        direction = dx / length
        line = gp_Lin(
            gp_Pnt(float(p0[0]), float(p0[1]), float(p0[2])),
            gp_Dir(float(direction[0]), float(direction[1]), float(direction[2])),
        )

        # Axis-aligned edges share their carrier line's candidate faces;
        # anything else computes its own candidate intervals.
        nz_axes = np.nonzero(dx)[0]
        entry = None
        offset = 0.0
        if len(nz_axes) == 1 and dx[nz_axes[0]] > 0.0:
            ax = int(nz_axes[0])
            entry = line_entry(ax, p0)
            offset = float(p0[ax]) - entry["origin_ax"]
            idx_l, win_l, wout_l = entry["cand"]
            cut = np.searchsorted(win_l, offset + length + tolerance, side="right")
            in_window = wout_l[:cut] >= offset - tolerance
            window_faces = idx_l[:cut][in_window]
            if window_faces.size:
                hits, clean = prefiltered._intersect_faces(
                    window_faces,
                    line,
                    -tolerance,
                    length + tolerance,
                )
            else:
                hits, clean = [], True
        else:
            candidates = prefiltered._line_candidates(p0, direction)
            hits, clean = prefiltered.window_hits(candidates, line, length)

        # Collect surface crossing parameters within [0, length]
        raw_params = [max(0.0, min(w, length)) for w, _ in hits]

        # Sort and deduplicate crossing parameters
        raw_params.sort()
        params: list[float] = []
        for p in raw_params:
            if not params or p - params[-1] > tolerance:
                params.append(p)

        # Build segment boundaries: [0, p1, p2, ..., pN, length]
        boundaries = [0.0]
        for p in params:
            if p - boundaries[-1] > tolerance:
                boundaries.append(p)
        if length - boundaries[-1] > tolerance:
            boundaries.append(length)

        # Derive sub-segment states from the crossing transitions; any
        # inconsistency demotes the edge to midpoint classification.
        use_fallback = not clean
        trans_ws: list[float] = []
        first_outside = True
        if not use_fallback:
            trans = sorted((w, s) for w, s in hits if s != 0)
            if trans:
                for k in range(1, len(trans)):
                    if trans[k][1] != -trans[k - 1][1]:
                        use_fallback = True
                        break
                trans_ws = [w for w, _ in trans]
                first_outside = trans[0][1] > 0
            else:
                if entry is not None:
                    if entry["ok"]:
                        k = bisect.bisect(
                            entry["ws"],
                            offset + length / 2.0,
                        )
                        first_outside = k % 2 == 0
                    else:
                        use_fallback = True
                else:
                    uniform = prefiltered.uniform_state(
                        candidates,
                        line,
                        length,
                    )
                    if uniform is None:
                        use_fallback = True
                    else:
                        first_outside = uniform
        if use_fallback:
            _EDGE_FRACTION_STATS["fallback_edges"] += 1
            # Probe axes for local point classification: most-perpendicular
            # first (the decisive direction for edges lying inside a face
            # plane).
            probe_axes = np.argsort(np.abs(direction), kind="stable")

        # Sum the outside length over sub-segment midpoints
        outside_len = 0.0
        for j in range(len(boundaries) - 1):
            seg_start = boundaries[j]
            seg_end = boundaries[j + 1]
            seg_len = seg_end - seg_start
            if seg_len < tolerance:
                continue
            t_mid = (seg_start + seg_end) / 2
            if use_fallback:
                mid_pt = p0 + t_mid * direction
                state = classify_point(mid_pt, probe_axes)
                if state is None:
                    # Every probe ray was spoiled — full-solid classifier
                    # as the last resort (O(faces), rare).
                    _EDGE_FRACTION_STATS["classifier_points"] += 1
                    if classifier is None:
                        classifier = BRepClass3d_SolidClassifier()
                        classifier.Load(pec_solid)
                    classifier.Perform(
                        gp_Pnt(float(mid_pt[0]), float(mid_pt[1]), float(mid_pt[2])),
                        tolerance,
                    )
                    outside = classifier.State() not in (TopAbs_IN, TopAbs_ON)
                else:
                    outside = state
            else:
                k = bisect.bisect(trans_ws, t_mid)
                outside = first_outside if k % 2 == 0 else not first_outside
            if outside:
                outside_len += seg_len

        f_L[i] = max(0.0, min(outside_len / length, 1.0))

    return f_L


# ---------------------------------------------------------------------------
# Parallel cross-section prefill
# ---------------------------------------------------------------------------

# Minimum number of outstanding (axis, plane, shape) section queries
# before a process pool pays for its own startup.  Spinning up the
# spawn pool (workers importing NumPy + OCC) costs ~5 s wall per call;
# at the measured ~13 ms per fine-grained section query the pool only
# wins clearly above ~700 outstanding queries — below that, sequential
# is faster (measured: a 540-query mesh build was 6.9 s sequential vs
# 16.0 s pooled).
_SECTION_PARALLEL_MIN_QUERIES = 1024

# Alternative work-weighted trigger: queries that reach the prefill all
# delegate to the OCC Boolean (the planar engine answers its planes
# in-process, see compute_face_material_areas), and a delegated
# section costs ~40–80 µs per face of the queried shape (Boolean +
# wire assembly; measured on the 100-slot coupler: ~200 coplanar
# planes on a 420-face fuse = 5.1 s in one call).  The threshold sits
# at the ~5 s pool-startup break-even (~9 s of sequential work), so
# per-call delegation loads below it — the coupler cases — correctly
# stay sequential, while geometries another step up trigger the pool
# even though their raw query count stays far below
# _SECTION_PARALLEL_MIN_QUERIES.
#
# Both thresholds are *admission* tests only (DD-141): face count is a
# proxy for cost, and how good a proxy depends on the geometry class —
# on a row of small PEC posts it overestimates by an order of
# magnitude, because bbox prefiltering makes each section cheap despite
# the large fused face count.  Admitted batches are therefore timed on
# a sample before the pool is built; see ``_parallel_section_prefill``.
_SECTION_PARALLEL_MIN_FACE_WORK = 150_000

# Wall-clock cost of building the spawn pool: n fresh interpreters each
# importing NumPy + OCC.  Measured ~5 s for 8 workers on the reference
# box, and it is a property of the machine, not of the geometry — which
# is exactly why the *other* side of the comparison is measured per
# call rather than estimated.
_SECTION_POOL_STARTUP_S = 5.0

# Queries timed sequentially before deciding.  They are drawn evenly
# across the cost-sorted schedule (not from its front, which holds the
# deliberately most expensive work), and nothing is wasted: the sample
# fills the same cache the pool would have filled.
_SECTION_SAMPLE_QUERIES = 24

# The pool must beat sequential by this margin to be worth building.
# With n workers the parallel part costs t_seq/n, so break-even sits at
# t_seq > startup * n/(n-1) ≈ 1.14 * startup; the margin covers the
# scheduling and result-transfer overhead the model ignores.
_SECTION_POOL_SPEEDUP_MARGIN = 1.5

# Shapes broadcast to each worker process, keyed by shape index.
# Populated once per worker by ``_section_worker_init``.
_SECTION_WORKER_SHAPES: dict[int, object] = {}


def _section_worker_count() -> int:
    """Process count for the parallel section prefill.

    ``MAGNELIO_SECTION_WORKERS`` overrides: ``0`` or ``1`` disables
    pooling entirely, any larger integer sets the worker count.  The
    default is ``min(8, cpu_count)`` — measured saturation point of the
    pool (beyond 8 workers, OCCT's own ``SetRunParallel`` threads and
    pool startup overhead eat the gain).
    """
    env = os.environ.get("MAGNELIO_SECTION_WORKERS", "").strip()
    if env:
        try:
            return max(0, int(env))
        except ValueError:
            warnings.warn(
                f"MAGNELIO_SECTION_WORKERS={env!r} is not an integer; "
                "using the automatic worker count.",
                stacklevel=2,
            )
    return min(8, os.cpu_count() or 1)


def _section_worker_init(shape_blobs: list[tuple[int, bytes]]) -> None:
    """Worker initializer: deserialize the broadcast shapes once.

    Each worker is a fresh ``spawn`` interpreter (never ``fork``: the
    parent may already run OCCT/TBB threads via ``SetRunParallel``, and
    forking a multithreaded process inherits locked mutex state).
    """
    from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Shape  # noqa: PLC0415

    for si, blob in shape_blobs:
        shape = TopoDS_Shape()
        breptools.ReadFromString(blob, shape)
        _SECTION_WORKER_SHAPES[si] = shape


def _section_worker(
    task: tuple[str, float, int, float, float, float, str],
) -> list[np.ndarray]:
    axis_letter, plane_pos, si, deflection, scale, nudge, context = task
    # The broadcast shape blob was serialised at *scale*; passing the
    # same scale here keeps the worker's meter-in/meter-out contract
    # identical to the sequential path.
    return cross_section_polygons(
        _SECTION_WORKER_SHAPES[si],
        axis_letter,
        plane_pos,
        deflection=deflection,
        scale=scale,
        nudge=nudge,
        context=context,
    )


@contextlib.contextmanager
def _hidden_main_module():
    """Keep spawned workers from re-importing the caller's ``__main__``.

    ``spawn`` normally tells each child to re-create the parent's main
    module, which it does by re-executing the user's top-level script.
    A script without an ``if __name__ == "__main__":`` guard therefore
    runs its *entire* simulation once per worker before that worker
    accepts a single section task (the nested pool it then attempts is
    what raises the "bootstrapping phase" RuntimeError).

    Our workers never touch ``__main__``: the shapes travel as BRep
    blobs and ``_section_worker`` pickles by reference through this
    module.  ``multiprocessing.spawn.get_preparation_data`` only
    schedules the re-import when ``__main__`` exposes a ``__spec__``
    name or a ``__file__``, so clearing both for the lifetime of the
    pool makes the children skip it — guard or no guard in the script.

    The attributes are process-global while the pool lives, so a
    *concurrent* process launch from another thread would inherit the
    same suppression.  Nothing in magnelio starts processes elsewhere.
    """
    main = sys.modules.get("__main__")
    if main is None:
        yield
        return
    sentinel = object()
    saved_spec = getattr(main, "__spec__", sentinel)
    saved_file = getattr(main, "__file__", sentinel)
    try:
        main.__spec__ = None
        if saved_file is not sentinel:
            del main.__file__
        yield
    finally:
        if saved_spec is sentinel:
            if hasattr(main, "__spec__"):
                del main.__spec__
        else:
            main.__spec__ = saved_spec
        if saved_file is not sentinel:
            main.__file__ = saved_file


def _sample_and_admit(
    queries: list[tuple[int, float, int]],
    shapes_with_material: list[tuple[object, int]],
    deflection: float,
    section_cache: dict,
    annotate,
    scale: float,
    n_workers: int,
    nudge: float | None = None,
    contexts: list[str] | None = None,
) -> list[tuple[int, float, int]]:
    """Time a sample of *queries*; return what is left worth pooling.

    The sample is computed in-process and cached, so it is progress
    either way — the only question it answers is whether the remainder
    justifies a pool.  Samples are drawn with a stride across the
    cost-sorted schedule rather than from its front, which by
    construction holds the most expensive queries and would make every
    batch look worth parallelising.

    Returns the un-computed remainder when the pool should be built,
    or an empty list when the caller should simply fall through to the
    sequential path.
    """
    import time  # noqa: PLC0415

    n = len(queries)
    n_sample = min(_SECTION_SAMPLE_QUERIES, n)
    if n_sample == 0:
        return []
    stride = max(1, n // n_sample)
    sample_idx = list(range(0, n, stride))[:n_sample]

    axis_letter = {0: "x", 1: "y", 2: "z"}
    t0 = time.perf_counter()
    for i in sample_idx:
        axis, pos, si = queries[i]
        polys = cross_section_polygons(
            shapes_with_material[si][0]._occ_shape(scale),
            axis_letter[axis],
            pos,
            deflection=deflection,
            scale=scale,
            nudge=nudge,
            context=contexts[si] if contexts else "",
        )
        section_cache[(axis, pos, si)] = [annotate(p) for p in polys]
    elapsed = time.perf_counter() - t0

    remaining = [q for i, q in enumerate(queries) if i not in set(sample_idx)]
    if not remaining:
        return []
    per_query = elapsed / len(sample_idx)
    projected = per_query * len(remaining)
    # Break-even: startup + projected/n_workers < projected.
    if projected < _SECTION_POOL_STARTUP_S * _SECTION_POOL_SPEEDUP_MARGIN:
        return []
    return remaining


def _parallel_section_prefill(
    shapes_with_material: list[tuple[object, int]],
    queries: list[tuple[int, float, int]],
    deflection: float,
    section_cache: dict,
    annotate,
    scale: float = 1.0,
    nudge: float | None = None,
    contexts: list[str] | None = None,
) -> None:
    """Fill ``section_cache`` for ``queries`` using a process pool.

    Pure prefill: every entry stored here is exactly what the
    sequential path would compute for the same ``(axis, plane_pos,
    shape_index)`` key (`cross_section_polygons` is deterministic, and
    the shapes cross the process boundary via OCCT's own BRep
    serialisation, verified bit-identical).  On any failure the cache
    is simply left partially filled and the sequential path computes
    the remainder — parallelism is an execution strategy only, never a
    correctness dependency.

    A sample of the queries is timed in-process first (DD-141): the
    caller's admission test scores cost by face count, which is a good
    proxy on some geometry classes and an order-of-magnitude
    over-estimate on others, and building a pool that does not pay for
    itself costs more than the work it was meant to accelerate.
    """
    n_workers = _section_worker_count()
    if n_workers <= 1:
        return
    if contexts is None:
        contexts = [""] * len(shapes_with_material)
    import multiprocessing  # noqa: PLC0415

    # Already inside somebody else's worker (a user-level parameter
    # sweep, say): a nested pool would multiply the process count
    # instead of the throughput.  Stay sequential, silently.
    if multiprocessing.parent_process() is not None:
        return
    try:
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

        from OCC.Core.BRepTools import breptools  # noqa: PLC0415

        # Cost-aware schedule: per-section cost is wildly axis-
        # dependent (a plane normal to a geometry's layering axis cuts
        # every primitive; measured ~20x a cheap plane on the
        # 1002-primitive case).  The rarest axes are queried first —
        # expensive work lands at the front where every worker is busy
        # — and the chunks are fine enough that no straggler chunk
        # holds the pool at the end (73 -> 63 s on the stress case).
        # Order is an execution detail only: the cache is keyed, and
        # zip() below pairs each query with its own result.
        axis_counts = {0: 0, 1: 0, 2: 0}
        for axis, _, _ in queries:
            axis_counts[axis] += 1
        queries = sorted(
            queries,
            key=lambda q: (axis_counts[q[0]], q[0], q[1], q[2]),
        )

        queries = _sample_and_admit(
            queries,
            shapes_with_material,
            deflection,
            section_cache,
            annotate,
            scale,
            n_workers,
            nudge,
            contexts,
        )
        if not queries:
            return

        needed = sorted({si for (_, _, si) in queries})
        blobs = [
            (si, breptools.WriteToString(shapes_with_material[si][0]._occ_shape(scale)))
            for si in needed
        ]
        axis_letter = {0: "x", 1: "y", 2: "z"}
        tasks = [
            (axis_letter[axis], pos, si, deflection, scale, nudge, contexts[si])
            for (axis, pos, si) in queries
        ]
        ctx = multiprocessing.get_context("spawn")
        # The hide window has to span the whole pool lifetime, not just
        # its construction: ProcessPoolExecutor spawns its workers
        # lazily on the first submit.
        with (
            _hidden_main_module(),
            ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_section_worker_init,
                initargs=(blobs,),
            ) as ex,
        ):
            chunk = max(1, len(tasks) // (n_workers * 32))
            for key, polys in zip(
                queries,
                ex.map(_section_worker, tasks, chunksize=chunk),
            ):
                section_cache[key] = [annotate(p) for p in polys]
    except Exception as exc:
        warnings.warn(
            f"Parallel cross-section prefill failed ({exc!r}); "
            "falling back to sequential computation.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# 3D face-solid intersection for conformal material area fractions
# ---------------------------------------------------------------------------


def compute_face_material_areas(
    shapes_with_material: list[tuple[object, int]],
    material_library: dict,
    face_specs: np.ndarray,
    face_axes: np.ndarray,
    prop: str = "epsilon",
    deflection: float = 1e-4,
    section_cache: dict | None = None,
    pec_area_out: np.ndarray | None = None,
    pec_area_geom_out: np.ndarray | None = None,
    pec_area_jump_out: np.ndarray | None = None,
    material_fraction_mids: np.ndarray | None = None,
    material_fractions_out: np.ndarray | None = None,
    domain_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    | None = None,
    scale: float = 1.0,
    nudge: float | None = None,
) -> np.ndarray:
    """Compute area-weighted material property on rectangular faces.

    Section-based pipeline (the cross-section approach of commercial
    hexahedral meshers):

    1. Faces are grouped by ``(axis, plane_pos)``.
    2. For each unique plane, the cross-section polygons of every shape
       are computed once via :func:`cross_section_polygons` (OCC's
       ``BRepAlgoAPI_Section`` + tangential tessellation).
    3. For each face, the section polygons of each shape are clipped to
       the face rectangle (Sutherland-Hodgman) and their (signed) area
       summed.  Outer boundaries are CCW (+area), holes CW (−area), so
       the signed sum is the solid's intersection area on that face.

    This replaces the per-face ``BRepAlgoAPI_Common`` Boolean used
    previously, which scaled poorly: ~3 ms per face × 27 504 faces in
    the WR-90/coax test was ~80 s.  The section-based path scales with
    the number of unique planes (~hundreds) instead of the number of
    faces (~tens of thousands).

    Step 2 itself avoids the per-plane ``BRepAlgoAPI_Section`` Boolean
    (~1 ms fixed cost per touched face pair) wherever possible: a
    :class:`_PlanarSectionEngine` per shape answers planes whose
    candidate faces are planar with straight, transversally crossed
    edges exactly from cached face/edge data; only planes with curved
    candidates, tangencies or vertex/coplanar coincidences (including
    the DD-087 degenerate planes) fall back to the OCC Boolean, which
    keeps every boundary-case semantic unchanged.

    Materials are processed in reverse priority order (last shape wins).
    For ``prop='epsilon'`` and ``prop='sigma'``, PEC area is excluded
    from the weighted sum (D = 0 inside PEC) but still claims its area
    budget.  For ``prop='mu'``, PEC is included with ``mu_r = 1.0``.

    Parameters
    ----------
    shapes_with_material : list of (shape_obj, material_id)
        Ordered from lowest to highest priority.  Each ``shape_obj`` must
        have ``._occ_shape()`` and ``.bounding_box()`` methods.
    material_library : dict
        ``{int: Material}`` mapping with ``is_pec``, ``epsilon``, ``mu``,
        ``sigma`` attributes.
    face_specs : np.ndarray
        Shape ``(N, 5)``: each row is
        ``[plane_pos, u_min, v_min, u_max, v_max]``.
    face_axes : np.ndarray
        Shape ``(N,)``, int.  Normal axis per face: 0=x, 1=y, 2=z.
    prop : str
        Material property to average: ``'epsilon'``, ``'sigma'``,
        ``'mu'``, or ``'sigma_m'`` (WP-C4 — E-side PEC conventions,
        zero background like ``'sigma'``).
    deflection : float
        Tessellation chord-deflection used by ``cross_section_polygons``.
        Smaller values → more accurate curved-boundary areas at the cost
        of more polygon vertices.  ``1e-4`` (≈ 0.1 mm) is suitable for
        typical microwave geometries.
    nudge : float or None, default None
        Degeneracy-escape step handed to
        :func:`cross_section_polygons`; ``None`` leaves it tied to
        *deflection*.  A grid caller should pass a fraction of its cell
        size — see that function's parameter list for why the two must
        not share one knob.
    section_cache : dict or None, default None
        Optional shared cache, keyed by ``(axis, plane_pos, shape_index)``,
        of cross-section polygon lists.  When the same caller invokes
        this function multiple times with overlapping ``face_specs``
        (e.g. ε then σ over the same E-edges), passing the same dict
        avoids recomputing identical sections.  ``deflection`` must be
        consistent across calls sharing a cache; the caller is
        responsible for that invariant.  When ``None``, a call-local
        cache is used internally (results are unchanged; the dict is
        simply not shared across calls).

    Notes
    -----
    When a call needs at least ``_SECTION_PARALLEL_MIN_QUERIES``
    not-yet-cached cross-sections that the planar engine cannot answer
    (fast-path queries never count — a worker round-trip costs more
    than the in-process answer), they are precomputed in a
    ``spawn``-based process pool (``MAGNELIO_SECTION_WORKERS`` overrides
    the worker count; ``0`` disables pooling).  This is an execution
    strategy only: each worker runs the same deterministic
    :func:`cross_section_polygons` on an OCCT-BRep-serialised copy of
    the shape, and any pool failure falls back to the sequential path.
    pec_area_out : np.ndarray or None, default None
        Optional output buffer of shape ``(N,)``.  When provided, each
        entry is filled with the per-face PEC-overlap area
        (``A_PEC[fi]`` in m²) computed from the same OCC tessellation
        that drives the weighted average.  Used by the M_μ curved-PEC
        sub-cell correction (DD-051 Variante A) which needs the
        geometric ``A_face_free = A_face − A_PEC`` rather than the
        material-property mean.  Faces fully covered by non-PEC
        material get ``0.0``; faces fully covered by PEC get the
        full face area.  When ``None``, no PEC area is recorded.
        On degenerate planes (see below) the value follows the
        DD-106 min-convention: ``min(A_PEC(p+δ), A_PEC(p−δ))`` —
        a face is only blocked where it is *embedded* in PEC on both
        sides; a wall merely tangential to the face leaves it free
        (the staircase limit for perfectly gridded walls, and the
        only convention that is translation-invariant along an
        extruded feed).
    pec_area_geom_out : np.ndarray or None, default None
        Optional output buffer of shape ``(N,)`` for the GEOMETRIC
        PEC-overlap area (DD-087).  Identical to ``pec_area_out``
        except on DEGENERATE planes: planes tangent to a material-
        boundary face (grid-snapped flat solid faces, cylinder/sphere
        tangencies — detected per *face* of every shape, DD-106),
        where ``BRepAlgoAPI_Section`` is ill-posed.  There every
        channel is evaluated from sections re-taken a small step to
        either side of the plane; the geometric channel books the
        LARGER PEC area of the two sides ("shift towards the PEC" —
        the wall area lands in the adjacent non-PEC cell where the H
        samples live), while ``pec_area_out`` books the smaller one
        (min-convention above).
    pec_area_jump_out : np.ndarray or None, default None
        Optional output buffer of shape ``(N,)`` for the FLAT wall area
        lying IN the section plane (DD-087): the wall inside a face is
        the set of points with PEC on one side and non-PEC on the
        other, whose area is the JUMP ``|A_PEC(p+δ) − A_PEC(p−δ)|`` of
        the PEC overlap across the plane.  A plane coinciding with a
        flat solid face jumps by that face's area; a plane merely
        shadowed by a curved wall in front of it does not jump at all.
        This separates the two wall families a cell can hold at once
        (a lid AND a mantle), which the divergence cell vector alone
        cannot resolve.  Zero off the degenerate planes (the PEC
        overlap is continuous there).  Requires ``pec_area_geom_out``.
    material_fraction_mids : np.ndarray or None, default None
        WP-C1 (DD-093): material ids whose EFFECTIVE (post-priority)
        area fraction per face is requested — typically the few
        dispersive/σ*-carrying materials.  Uncovered area counts
        toward the background id 0 (the same convention as the
        property average).  Fractions are raw area shares of the SAME
        reverse-priority budget cascade that feeds the property
        average, so ``Σ_requested f_i ≤ 1`` with equality when the
        requested set covers every material on the face (PEC claims
        its share like any other id).  Requires
        ``material_fractions_out``.
    material_fractions_out : np.ndarray or None, default None
        Output buffer of shape ``(len(material_fraction_mids), N)``,
        pre-zeroed by the caller; filled with the per-face fractions.
        Rows follow ``material_fraction_mids`` order.  Faces the
        pipeline does not process (no section coverage) keep 0.
    domain_bounds : tuple of (lo, hi) pairs or None, default None
        Computational-domain extents per axis ``((x_lo, x_hi),
        (y_lo, y_hi), (z_lo, z_hi))``.  A degenerate plane coinciding
        with a domain bound is evaluated ONE-SIDED (interior side
        only) for the matrix channel — the face there represents the
        interior half-cell, and averaging with the fictitious outside
        would poison μ̄ on μ_r ≠ 1 port planes.  The geometric channel
        keeps both sides (a registered domain-end plane must still
        read as a shorting lid, DD-099).  ``None`` disables the
        one-sided rule.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``.  Area-weighted effective property value per face.
    """
    from magnelio.geo._polygon_clip import (  # noqa: PLC0415
        HAS_NUMBA,
        clip_polygon_to_rect,
        face_pec_area_kernel,
        face_property_kernel,
        face_shape_area_kernel,
        pack_annotated_sections,
        polygon_area,
    )

    n_faces = len(face_specs)
    result = np.full(n_faces, np.nan, dtype=np.float64)

    if (material_fractions_out is None) != (material_fraction_mids is None):
        raise ValueError(
            "material_fraction_mids and material_fractions_out must be passed together",
        )
    frac_mids = (
        np.asarray(material_fraction_mids, dtype=np.int64)
        if material_fraction_mids is not None
        else None
    )
    if frac_mids is not None and material_fractions_out.shape != (frac_mids.size, n_faces):
        raise ValueError(
            f"material_fractions_out must have shape ({frac_mids.size}, {n_faces})",
        )
    shape_mid_arr = np.array(
        [mid for _, mid in shapes_with_material],
        dtype=np.int64,
    )

    if n_faces == 0 or not shapes_with_material:
        return result

    is_mu = prop == "mu"
    pec_ids = {mid for mid, mat in material_library.items() if mat.is_pec}
    axis_letter = {0: "x", 1: "y", 2: "z"}
    # A scene's solids reach this layer as anonymous OCC shapes; the
    # material name is the one label that survives and it is enough to
    # tell a warning about the conductor from one about the vacuum.
    contexts = [
        f"the {getattr(material_library.get(mid), 'name', mid)!r} solid"
        for _, mid in shapes_with_material
    ]

    # Per-shape bounding boxes for fast plane-vs-shape rejection.
    # All bookkeeping below runs in meters: the section leaves
    # (engine / cross_section_polygons) convert at the boundary, so
    # cache keys, clip/area kernels and the DD-106 degenerate handling
    # are untouched by the DD-120 scaling.
    shape_bboxes = []
    for shape_obj, _ in shapes_with_material:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = shape_obj.bounding_box(scale)
        shape_bboxes.append((xmin, ymin, zmin, xmax, ymax, zmax))

    # Planar fast path (one engine per shape): planes whose candidate
    # faces are all planar with straight, transversally crossed edges
    # are sectioned exactly without the per-plane Boolean; every other
    # plane (curved candidates, tangencies, DD-087 degenerate planes)
    # delegates to cross_section_polygons unchanged.
    engines = [
        _PlanarSectionEngine(shape_obj._occ_shape(scale), scale=scale)
        for shape_obj, _ in shapes_with_material
    ]

    def _shape_sections(si, axis, pos):
        polys = engines[si].section(axis, pos) if engines[si].enabled else None
        if polys is None:
            polys = cross_section_polygons(
                shapes_with_material[si][0]._occ_shape(scale),
                axis_letter[axis],
                pos,
                deflection=deflection,
                scale=scale,
                nudge=nudge,
                context=contexts[si],
            )
        return polys

    # Step 1: group face indices by (axis, plane_pos).
    # The plane_pos values come from cell-centre arrays, so identical-
    # value floats compare equal — no rounding needed.
    plane_to_face_indices: dict[tuple[int, float], list[int]] = {}
    for fi in range(n_faces):
        key = (int(face_axes[fi]), float(face_specs[fi, 0]))
        plane_to_face_indices.setdefault(key, []).append(fi)

    # Step 2: compute cross-section polygons per (plane, shape) once.
    # Each polygon is annotated with (bbox, signed_area) for the face loop:
    # bbox lets us reject polygons fully outside a face's rectangle in O(1);
    # signed_area lets us skip Sutherland-Hodgman for polygons fully inside.
    # Layout: sections_per_plane[(axis, plane_pos)] = list of annotated lists,
    # one entry per shape (in shapes_with_material order). Empty list when
    # the shape's bounding box doesn't intersect the plane.
    # When section_cache is provided, lookups by (axis, plane_pos, shape_index)
    # avoid recomputation across calls (e.g. ε followed by σ on the same edges).
    def _annotate(poly: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
        u = poly[:, 0]
        v = poly[:, 1]
        bb = (float(u.min()), float(v.min()), float(u.max()), float(v.max()))
        return poly, bb, polygon_area(poly)

    sections_per_plane: dict[
        tuple[int, float],
        list[list[tuple[np.ndarray, tuple[float, float, float, float], float]]],
    ] = {}
    # Planes TANGENT to any material-boundary face (grid-snapped flat
    # solid faces, cylinder/sphere tangencies) make the exact-plane
    # ``BRepAlgoAPI_Section`` ill-posed — the intersection is 2D, not
    # 1D, and OCC returns coincident-face boundary wires, tessellation
    # fragments or nothing depending on the solid's topology (measured:
    # full-PEC faces reporting zero, DD-087; a slanted 0→1 front
    # through a translation-invariant wall, DD-106).  Detection is per
    # FACE of every shape via ``_face_critical_planes`` — the same
    # machinery whose output the mesher snaps grid lines to, so
    # exactly the constructed coincidences are caught (a bbox-only
    # test misses interior walls of Union/Difference solids).  Shape
    # bbox extents are kept as the conservative fallback for surface
    # types without analytic tangents (cones, tori, free-form).
    # Every channel on a degenerate plane is evaluated from sections
    # re-taken a small step to either side (the shift sits at the
    # tessellation tolerance, far below any cell size).
    if section_cache is None:
        section_cache = {}
    tangent_pos: list[list[float]] = [[], [], []]
    for si, (shape_obj, _) in enumerate(shapes_with_material):
        sbb = shape_bboxes[si]
        for ax in range(3):
            tangent_pos[ax].extend((sbb[ax], sbb[ax + 3]))
        t_key = ("__tangent_planes__", si)
        if t_key in section_cache:
            per_axis = section_cache[t_key]
        else:
            try:
                scaled_planes = _face_critical_planes(shape_obj._occ_shape(scale))
                per_axis = {axl: [p / scale for p in scaled_planes[axl]] for axl in ("x", "y", "z")}
            except Exception:  # noqa: BLE001 — bbox extents still apply
                per_axis = {"x": [], "y": [], "z": []}
            section_cache[t_key] = per_axis
        for ax, axl in enumerate(("x", "y", "z")):
            tangent_pos[ax].extend(per_axis[axl])
    tangent_arr = [np.unique(np.asarray(t, dtype=np.float64)) for t in tangent_pos]

    def _is_degenerate(axis: int, plane_pos: float) -> bool:
        tol = 1e-12 * (1.0 + abs(plane_pos))
        t = tangent_arr[axis]
        i = int(np.searchsorted(t, plane_pos))
        for j in (i - 1, i):
            if 0 <= j < t.size and abs(plane_pos - t[j]) <= tol:
                return True
        return False

    plane_pec_degenerate: dict[tuple[int, float], bool] = {
        (axis, plane_pos): _is_degenerate(axis, plane_pos)
        for (axis, plane_pos) in plane_to_face_indices
    }

    # Matrix-channel side selection (DD-106): both shifted sides,
    # except at a domain bound where only the interior side exists.
    def _matrix_sides(axis: int, plane_pos: float) -> tuple[bool, bool]:
        if domain_bounds is not None:
            lo, hi = domain_bounds[axis]
            tol = 1e-12 * (1.0 + abs(plane_pos))
            if abs(plane_pos - lo) <= tol:
                return True, False
            if abs(plane_pos - hi) <= tol:
                return False, True
        return True, True

    # Parallel prefill: collect every (axis, plane, shape) section this
    # call will need — the exact positions of regular planes plus the
    # shifted positions of degenerate planes (whose exact position is
    # never sectioned, DD-106) — and compute the cache misses in a
    # spawn-based process pool.  The sequential loops below then find
    # everything cached; their bookkeeping (priority order, degenerate
    # handling) is untouched.  Below the query threshold, or with
    # MAGNELIO_SECTION_WORKERS=0, nothing happens here at all.
    prefill_queries: dict[tuple[int, float, int], None] = {}
    for axis, plane_pos in plane_to_face_indices:
        if plane_pec_degenerate[(axis, plane_pos)]:
            positions = [plane_pos + deflection, plane_pos - deflection]
        else:
            positions = [plane_pos]
        for pos in positions:
            for si in range(len(shapes_with_material)):
                sbb = shape_bboxes[si]
                slack = 1e-12 * (1.0 + abs(pos))
                if pos < sbb[axis] - slack or pos > sbb[axis + 3] + slack:
                    continue
                key = (axis, pos, si)
                if key in section_cache:
                    continue
                # Planes the planar engine will answer in-process are
                # not worth a worker round-trip — only queries that
                # will delegate to the OCC Boolean count toward (and
                # are computed by) the pool.
                if engines[si].enabled and engines[si].can_fast(axis, pos):
                    continue
                prefill_queries[key] = None
    prefill_work = sum(engines[si].face_count for (_, _, si) in prefill_queries)
    if (
        len(prefill_queries) >= _SECTION_PARALLEL_MIN_QUERIES
        or prefill_work >= _SECTION_PARALLEL_MIN_FACE_WORK
    ):
        _parallel_section_prefill(
            shapes_with_material,
            list(prefill_queries),
            deflection,
            section_cache,
            _annotate,
            scale=scale,
            nudge=nudge,
            contexts=contexts,
        )

    def _sections_at(axis: int, pos: float) -> list:
        per_shape: list = []
        for si in range(len(shapes_with_material)):
            sbb = shape_bboxes[si]
            slack = 1e-12 * (1.0 + abs(pos))
            if pos < sbb[axis] - slack or pos > sbb[axis + 3] + slack:
                per_shape.append([])
                continue
            cache_key = (axis, pos, si)
            if cache_key in section_cache:
                annotated = section_cache[cache_key]
            else:
                polys = _shape_sections(si, axis, pos)
                annotated = [_annotate(p) for p in polys]
                section_cache[cache_key] = annotated
            per_shape.append(annotated)
        return per_shape

    for axis, plane_pos in plane_to_face_indices:
        if plane_pec_degenerate[(axis, plane_pos)]:
            continue  # DD-106: never section a degenerate plane exactly
        sections_per_plane[(axis, plane_pos)] = _sections_at(
            axis,
            plane_pos,
        )

    # Shifted sections for degenerate planes: BOTH side positions
    # ``plane ± deflection`` are sectioned and feed every channel
    # (DD-106).  The geometric channel books the LARGER PEC area
    # ("shift towards the PEC") — the wall area consistently lands in
    # the adjacent non-PEC cell, where the H samples live (the other
    # choice parks lids inside PEC cells whose samples are dead;
    # measured 0.39× wall sums on the pillbox).  The matrix channel
    # books the SMALLER one — a face is only blocked where it is
    # embedded in PEC on both sides — and averages the property
    # values of its sides (the staircase cell-pair mean in the
    # perfectly-gridded limit).
    shifted_sections_per_plane: dict[tuple[int, float], list] = {}
    for (axis, plane_pos), degen in plane_pec_degenerate.items():
        if not degen:
            continue
        shifted_sections_per_plane[(axis, plane_pos)] = [
            _sections_at(axis, plane_pos + deflection),
            _sections_at(axis, plane_pos - deflection),
        ]

    def _clip_area(annotated_polys, rect) -> float:
        u_min, v_min, u_max, v_max = rect
        area = 0.0
        for poly, (pu_min, pv_min, pu_max, pv_max), signed_area in annotated_polys:
            if pu_max < u_min or pu_min > u_max or pv_max < v_min or pv_min > v_max:
                continue
            if pu_min >= u_min and pu_max <= u_max and pv_min >= v_min and pv_max <= v_max:
                area += signed_area
                continue
            clipped = clip_polygon_to_rect(poly, rect)
            if len(clipped) >= 3:
                area += polygon_area(clipped)
        return abs(area)

    def _pec_area_from(per_shape_polys, rect, total_area) -> float:
        """PEC area on a face from a given section set (priority order,
        remaining-budget bookkeeping identical to the main loop)."""
        remaining = total_area
        pec_a = 0.0
        for si in range(len(shapes_with_material) - 1, -1, -1):
            if remaining <= 1e-30:
                break
            polys = per_shape_polys[si]
            if not polys:
                continue
            effective = min(_clip_area(polys, rect), remaining)
            if effective < 1e-30:
                continue
            if material_library[shapes_with_material[si][1]].is_pec:
                pec_a += effective
            remaining -= effective
        if remaining > 1e-30:
            bg_mat = material_library.get(0)
            if bg_mat is not None and bg_mat.is_pec:
                pec_a += remaining
        return pec_a

    # Step 3, batched (Numba): per plane, pack the annotated sections
    # into flat arrays once and run the accounting kernels over all of
    # the plane's faces.  The kernels mirror the fallback loop below
    # statement-for-statement (same reverse-priority budget, same
    # float-op order, the same njit'd clip/area leaf functions), so the
    # results are bit-identical — this is an execution strategy only.
    if HAS_NUMBA:
        bg_mat = material_library.get(0)
        bg_exists = bg_mat is not None
        bg_is_pec = bool(bg_exists and bg_mat.is_pec)
        # σ and σ* share the zero-background convention: an uncovered
        # face patch with no library background is lossless vacuum.
        prop_is_sigma = prop in ("sigma", "sigma_m")

        shape_arr_cache: dict[int, tuple] = {}

        def _shape_arrays(axis):
            if axis not in shape_arr_cache:
                n_s = len(shapes_with_material)
                val = np.zeros(n_s, dtype=np.float64)
                is_pec_arr = np.zeros(n_s, dtype=np.bool_)
                for si, (_, mid) in enumerate(shapes_with_material):
                    mat = material_library[mid]
                    is_pec_arr[si] = mid in pec_ids if not is_mu else mat.is_pec
                    if is_mu:
                        if not mat.is_pec:
                            val[si] = mat.mu[axis]
                    elif mid not in pec_ids:
                        val[si] = getattr(mat, prop)[axis]
                bg_val = float(getattr(bg_mat, prop)[axis]) if bg_exists else 0.0
                shape_arr_cache[axis] = (val, is_pec_arr, bg_val)
            return shape_arr_cache[axis]

        def _write_fractions(face_idx, rects, shape_eff, bg_rem):
            # WP-C1 (DD-093): effective per-shape areas from the same
            # budget cascade, aggregated to the requested material ids
            # (uncovered remainder → background id 0).
            areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
            safe = np.where(areas > 0.0, areas, 1.0)
            for mi, mid in enumerate(frac_mids):
                acc = shape_eff[:, shape_mid_arr == mid].sum(axis=1)
                if mid == 0:
                    acc = acc + bg_rem
                material_fractions_out[mi, face_idx] = acc / safe

        for (axis, plane_pos), face_idx_list in plane_to_face_indices.items():
            face_idx = np.asarray(face_idx_list, dtype=np.int64)
            rects = np.ascontiguousarray(
                face_specs[face_idx][:, 1:5],
                dtype=np.float64,
            )
            val, is_pec_arr, bg_val = _shape_arrays(axis)

            if plane_pec_degenerate[(axis, plane_pos)]:
                # DD-106: every channel from the two shifted sides.
                packed_sides = [
                    pack_annotated_sections(section_set)
                    for section_set in shifted_sections_per_plane[(axis, plane_pos)]
                ]
                res_sides = []
                pec_sides = []
                proc = np.zeros(face_idx.size, dtype=np.bool_)
                for packed_s in packed_sides:
                    res_b = result[face_idx].copy()  # NaN-prefilled
                    pec_b = np.zeros(face_idx.size, dtype=np.float64)
                    proc_b = np.zeros(face_idx.size, dtype=np.bool_)
                    face_property_kernel(
                        rects,
                        *packed_s,
                        val,
                        is_pec_arr,
                        is_mu,
                        bg_exists,
                        bg_is_pec,
                        bg_val,
                        prop_is_sigma,
                        res_b,
                        pec_b,
                        proc_b,
                    )
                    res_sides.append(res_b)
                    pec_sides.append(pec_b)
                    proc |= proc_b
                use_p, use_m = _matrix_sides(axis, plane_pos)
                if use_p and use_m:
                    res_c = 0.5 * (res_sides[0] + res_sides[1])
                    pec_c = np.minimum(pec_sides[0], pec_sides[1])
                elif use_p:
                    res_c, pec_c = res_sides[0], pec_sides[0]
                else:
                    res_c, pec_c = res_sides[1], pec_sides[1]
                result[face_idx] = res_c
                if pec_area_out is not None:
                    pec_area_out[face_idx[proc]] = pec_c[proc]

                if frac_mids is not None:
                    eff_sides = []
                    rem_sides = []
                    for packed_s in packed_sides:
                        shape_eff = np.zeros(
                            (face_idx.size, len(shapes_with_material)),
                            dtype=np.float64,
                        )
                        bg_rem = np.zeros(face_idx.size, dtype=np.float64)
                        face_shape_area_kernel(
                            rects,
                            *packed_s,
                            shape_eff,
                            bg_rem,
                        )
                        eff_sides.append(shape_eff)
                        rem_sides.append(bg_rem)
                    if use_p and use_m:
                        shape_eff = 0.5 * (eff_sides[0] + eff_sides[1])
                        bg_rem = 0.5 * (rem_sides[0] + rem_sides[1])
                    elif use_p:
                        shape_eff, bg_rem = eff_sides[0], rem_sides[0]
                    else:
                        shape_eff, bg_rem = eff_sides[1], rem_sides[1]
                    _write_fractions(face_idx, rects, shape_eff, bg_rem)

                if pec_area_geom_out is not None:
                    a_plus = np.zeros(face_idx.size, dtype=np.float64)
                    a_minus = np.zeros(face_idx.size, dtype=np.float64)
                    for buf, packed_s in ((a_plus, packed_sides[0]), (a_minus, packed_sides[1])):
                        face_pec_area_kernel(
                            rects,
                            *packed_s,
                            is_pec_arr,
                            bg_is_pec,
                            buf,
                        )
                    sel = face_idx[proc]
                    pec_area_geom_out[sel] = np.maximum(a_plus[proc], a_minus[proc])
                    if pec_area_jump_out is not None:
                        pec_area_jump_out[sel] = a_plus[proc] - a_minus[proc]
                continue

            packed = pack_annotated_sections(
                sections_per_plane[(axis, plane_pos)],
            )
            res_buf = result[face_idx].copy()  # NaN-prefilled
            pec_buf = np.zeros(face_idx.size, dtype=np.float64)
            proc = np.zeros(face_idx.size, dtype=np.bool_)
            face_property_kernel(
                rects,
                *packed,
                val,
                is_pec_arr,
                is_mu,
                bg_exists,
                bg_is_pec,
                bg_val,
                prop_is_sigma,
                res_buf,
                pec_buf,
                proc,
            )
            result[face_idx] = res_buf
            if pec_area_out is not None:
                pec_area_out[face_idx[proc]] = pec_buf[proc]

            if frac_mids is not None:
                shape_eff = np.zeros(
                    (face_idx.size, len(shapes_with_material)),
                    dtype=np.float64,
                )
                bg_rem = np.zeros(face_idx.size, dtype=np.float64)
                face_shape_area_kernel(rects, *packed, shape_eff, bg_rem)
                _write_fractions(face_idx, rects, shape_eff, bg_rem)

            if pec_area_geom_out is not None:
                sel = face_idx[proc]
                pec_area_geom_out[sel] = pec_buf[proc]
                if pec_area_jump_out is not None:
                    pec_area_jump_out[sel] = 0.0
        return result

    # Step 3 (fallback without Numba): per face, clip each shape's
    # section polygons against the face rectangle and accumulate the
    # area-weighted property.
    def _account(per_shape_polys, rect, total_area, axis):
        """One face's accounting pass over a section set: reverse-
        priority area budget, PEC bookkeeping, background fill.
        Returns ``(avg, pec_area, eff_by_shape, remaining)``."""
        u_min, v_min, u_max, v_max = rect
        remaining = total_area
        weighted_sum = 0.0
        pec_area = 0.0
        eff_by_shape = (
            np.zeros(len(shapes_with_material), dtype=np.float64) if frac_mids is not None else None
        )

        # Process shapes in reverse priority order (last wins).
        for si in range(len(shapes_with_material) - 1, -1, -1):
            if remaining <= 1e-30:
                break

            annotated_polys = per_shape_polys[si]
            if not annotated_polys:
                continue

            # Solid area on the face = signed sum of clipped polygon areas
            # (outer boundaries CCW, holes CW).  abs() handles tessellation
            # round-off near zero; the dominant sign is the outer boundary.
            #
            # Per-polygon bbox shortcuts:
            #   bbox fully outside face rect → 0 (skip).
            #   bbox fully inside face rect → reuse precomputed signed_area.
            #   partial overlap → run Sutherland-Hodgman.
            shape_area = 0.0
            for poly, (pu_min, pv_min, pu_max, pv_max), signed_area in annotated_polys:
                if pu_max < u_min or pu_min > u_max or pv_max < v_min or pv_min > v_max:
                    continue  # fully outside
                if pu_min >= u_min and pu_max <= u_max and pv_min >= v_min and pv_max <= v_max:
                    shape_area += signed_area  # fully inside
                    continue
                clipped = clip_polygon_to_rect(poly, rect)
                if len(clipped) >= 3:
                    shape_area += polygon_area(clipped)
            shape_area = abs(shape_area)

            effective = min(shape_area, remaining)
            if effective < 1e-30:
                continue

            mat_id = shapes_with_material[si][1]
            mat = material_library[mat_id]

            if is_mu:
                if mat.is_pec:
                    weighted_sum += 1.0 * effective
                    pec_area += effective
                else:
                    weighted_sum += mat.mu[axis] * effective
            else:
                if mat_id not in pec_ids:
                    val = getattr(mat, prop)[axis]
                    weighted_sum += val * effective
                else:
                    pec_area += effective
                # PEC: claims area, contributes 0 to weighted_sum (eps/sigma)
                # or 1.0 (mu).  Either way, accumulated into pec_area.

            if eff_by_shape is not None:
                eff_by_shape[si] = effective
            remaining -= effective

        # Uncovered area = background material (id 0 by convention).
        if remaining > 1e-30:
            bg_mat = material_library.get(0)
            if bg_mat is not None:
                if is_mu and bg_mat.is_pec:
                    weighted_sum += 1.0 * remaining
                    pec_area += remaining
                elif not is_mu and bg_mat.is_pec:
                    pec_area += remaining  # claims area, not in weighted_sum
                else:
                    weighted_sum += getattr(bg_mat, prop)[axis] * remaining
            else:
                if prop not in ("sigma", "sigma_m"):
                    weighted_sum += 1.0 * remaining

        return weighted_sum / total_area, pec_area, eff_by_shape, remaining

    def _write_fractions_fb(fi, total_area, eff_by_shape, remaining):
        for mi, mid in enumerate(frac_mids):
            acc = float(eff_by_shape[shape_mid_arr == mid].sum())
            if mid == 0 and remaining > 1e-30:
                acc += remaining
            material_fractions_out[mi, fi] = acc / total_area

    for fi in range(n_faces):
        plane_pos = float(face_specs[fi, 0])
        u_min = float(face_specs[fi, 1])
        v_min = float(face_specs[fi, 2])
        u_max = float(face_specs[fi, 3])
        v_max = float(face_specs[fi, 4])
        axis = int(face_axes[fi])

        total_area = (u_max - u_min) * (v_max - v_min)
        if total_area <= 0:
            continue

        rect = (u_min, v_min, u_max, v_max)

        if plane_pec_degenerate[(axis, plane_pos)]:
            # DD-106: every channel from the two shifted sides.
            plus, minus = shifted_sections_per_plane[(axis, plane_pos)]
            avg_p, pec_p, eff_p, rem_p = _account(plus, rect, total_area, axis)
            avg_m, pec_m, eff_m, rem_m = _account(minus, rect, total_area, axis)
            use_p, use_m = _matrix_sides(axis, plane_pos)
            if use_p and use_m:
                result[fi] = 0.5 * (avg_p + avg_m)
                pec_c = min(pec_p, pec_m)
            elif use_p:
                result[fi] = avg_p
                pec_c = pec_p
            else:
                result[fi] = avg_m
                pec_c = pec_m
            if pec_area_out is not None:
                pec_area_out[fi] = pec_c
            if frac_mids is not None:
                if use_p and use_m:
                    for mi, mid in enumerate(frac_mids):
                        acc_p = float(eff_p[shape_mid_arr == mid].sum())
                        if mid == 0 and rem_p > 1e-30:
                            acc_p += rem_p
                        acc_m = float(eff_m[shape_mid_arr == mid].sum())
                        if mid == 0 and rem_m > 1e-30:
                            acc_m += rem_m
                        material_fractions_out[mi, fi] = 0.5 * (acc_p + acc_m) / total_area
                elif use_p:
                    _write_fractions_fb(fi, total_area, eff_p, rem_p)
                else:
                    _write_fractions_fb(fi, total_area, eff_m, rem_m)
            if pec_area_geom_out is not None:
                a_plus = _pec_area_from(plus, rect, total_area)
                a_minus = _pec_area_from(minus, rect, total_area)
                pec_area_geom_out[fi] = max(a_plus, a_minus)
                if pec_area_jump_out is not None:
                    pec_area_jump_out[fi] = a_plus - a_minus
            continue

        avg, pec_area, eff_by_shape, remaining = _account(
            sections_per_plane[(axis, plane_pos)], rect, total_area, axis
        )
        result[fi] = avg
        if pec_area_out is not None:
            pec_area_out[fi] = pec_area
        if frac_mids is not None:
            _write_fractions_fb(fi, total_area, eff_by_shape, remaining)
        if pec_area_geom_out is not None:
            pec_area_geom_out[fi] = pec_area
            if pec_area_jump_out is not None:
                pec_area_jump_out[fi] = 0.0

    return result


# ---------------------------------------------------------------------------
# Edge and face selection utilities
# ---------------------------------------------------------------------------


def get_all_edges(shape):
    """Return all unique edges of an OCC shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        An OCC solid or compound.

    Returns
    -------
    list of TopoDS_Edge
        Unique edges (deduplicated via ``IsSame``).
    """
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for edge selection. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    edges = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        if not any(edge.IsSame(e) for e in edges):
            edges.append(edge)
        explorer.Next()
    return edges


def find_nearest_edge(shape, point, scale: float = 1.0):
    """Find the edge of *shape* nearest to a 3D point.

    Parameters
    ----------
    shape : TopoDS_Shape
        An OCC solid or compound.
    point : tuple of float
        (x, y, z) target point [m].

    Returns
    -------
    TopoDS_Edge
        The nearest edge.

    Raises
    ------
    ValueError
        If the shape has no edges.
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex  # noqa: PLC0415
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape  # noqa: PLC0415
        from OCC.Core.gp import gp_Pnt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for edge selection. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    all_edges = get_all_edges(shape)
    if not all_edges:
        raise ValueError("Shape has no edges.")

    vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(*_scale3(point, scale))).Shape()

    min_dist = float("inf")
    nearest = None
    for edge in all_edges:
        extrema = BRepExtrema_DistShapeShape(vertex_shape, edge)
        if extrema.IsDone() and extrema.NbSolution() > 0:
            d = extrema.Value()
            if d < min_dist:
                min_dist = d
                nearest = edge

    if nearest is None:
        raise ValueError("Could not find nearest edge.")
    return nearest


def find_nearest_face(shape, point, scale: float = 1.0):
    """Find the face of *shape* nearest to a 3D point.

    Parameters
    ----------
    shape : TopoDS_Shape
        An OCC solid or compound.
    point : tuple of float
        (x, y, z) target point [m].

    Returns
    -------
    TopoDS_Face
        The nearest face.

    Raises
    ------
    ValueError
        If the shape has no faces.
    """
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex  # noqa: PLC0415
        from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape  # noqa: PLC0415
        from OCC.Core.gp import gp_Pnt  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_FACE  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for face selection. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    faces = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        faces.append(topods.Face(face_explorer.Current()))
        face_explorer.Next()

    if not faces:
        raise ValueError("Shape has no faces.")

    vertex_shape = BRepBuilderAPI_MakeVertex(gp_Pnt(*_scale3(point, scale))).Shape()
    min_dist = float("inf")
    nearest_face = None
    for face in faces:
        extrema = BRepExtrema_DistShapeShape(vertex_shape, face)
        if extrema.IsDone() and extrema.NbSolution() > 0:
            d = extrema.Value()
            if d < min_dist:
                min_dist = d
                nearest_face = face

    if nearest_face is None:
        raise ValueError("Could not find nearest face.")
    return nearest_face


def resolve_faces(shape, points, scale: float = 1.0):
    """The faces of *shape* nearest to each of *points*, deduplicated.

    Accepts a single ``(x, y, z)`` point or a sequence of them; two
    points landing on the same face select it once.
    """
    if points is None:
        return []
    if not isinstance(points[0], (list, tuple)):
        points = [points]
    faces = []
    for point in points:
        face = find_nearest_face(shape, point, scale=scale)
        if not any(face.IsSame(seen) for seen in faces):
            faces.append(face)
    return faces


def face_plane_normal(face):
    """Unit normal of a planar OCC face, with a canonical sign.

    The sign of a face's own normal depends on how the face was built,
    which would make an offset direction derived from it unpredictable.
    The largest component is therefore forced positive, giving one
    reproducible direction per plane.
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415
        from OCC.Core.GeomAbs import GeomAbs_Plane  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for face queries.") from exc

    surf = BRepAdaptor_Surface(face)
    if surf.GetType() != GeomAbs_Plane:
        raise ValueError("Expected a planar face.")
    direction = surf.Plane().Axis().Direction()
    normal = [direction.X(), direction.Y(), direction.Z()]
    dominant = max(range(3), key=lambda i: abs(normal[i]))
    if normal[dominant] < 0.0:
        normal = [-c for c in normal]
    return tuple(normal)


def make_thick_solid(shape, opening_faces, thickness: float, scale: float = 1.0):
    """Hollow a solid to a constant wall thickness.

    Walls are built inward, so the outer surface of the result is the
    original one.  Faces in *opening_faces* are left out of the shell and
    become openings.

    Parameters
    ----------
    shape : TopoDS_Shape
        The solid to hollow.
    opening_faces : sequence of TopoDS_Face
        Faces to remove; may be empty for a closed internal void.
    thickness : float
        Wall thickness [m], positive.

    Returns
    -------
    TopoDS_Shape
        The hollowed solid.
    """
    try:
        from OCC.Core.BRepOffset import BRepOffset_Skin  # noqa: PLC0415
        from OCC.Core.BRepOffsetAPI import (  # noqa: PLC0415
            BRepOffsetAPI_MakeOffsetShape,
            BRepOffsetAPI_MakeThickSolid,
        )
        from OCC.Core.GeomAbs import GeomAbs_Arc  # noqa: PLC0415
        from OCC.Core.TopTools import TopTools_ListOfShape  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for shell.") from exc

    _check_dimensions("Shell", scale, thickness=thickness)
    offset = -thickness * scale  # negative: walls inward, outer surface kept
    failure = (
        f"Hollowing the solid failed at a wall thickness of "
        f"{thickness:.3e} m.  The offset surfaces stop being valid once "
        f"the wall approaches the smallest local dimension or curvature "
        f"radius of the solid — try a thinner wall."
    )

    def _result(maker):
        # IsDone() alone is not enough: an offset that cannot be built
        # still reports success and hands back a null shape.
        if not maker.IsDone():
            raise RuntimeError(failure)
        try:
            built = maker.Shape()
        except RuntimeError as exc:
            raise RuntimeError(failure) from exc
        if built is None or built.IsNull():
            raise RuntimeError(failure)
        return built

    opening_faces = list(opening_faces)
    if not opening_faces:
        # With no faces to remove there is nothing for MakeThickSolid to
        # open the body at, and it returns the shrunk solid rather than a
        # shell.  The sealed void is that solid subtracted from this one.
        maker = BRepOffsetAPI_MakeOffsetShape()
        maker.PerformByJoin(shape, offset, 1e-6, BRepOffset_Skin, False, False, GeomAbs_Arc, False)
        return boolean_difference(shape, _result(maker))

    faces = TopTools_ListOfShape()
    for face in opening_faces:
        faces.Append(face)
    maker = BRepOffsetAPI_MakeThickSolid()
    maker.MakeThickSolidByJoin(
        shape, faces, offset, 1e-6, BRepOffset_Skin, False, False, GeomAbs_Arc, False
    )
    built = _result(maker)
    # Walls thicker than the body meet in the middle; OCC reports success
    # and hands back the untouched solid, which would pass silently.
    if occ_volume(built) >= occ_volume(shape) * (1.0 - 1e-9):
        raise RuntimeError(
            f"A wall thickness of {thickness:.3e} m leaves no cavity in "
            f"this solid — the walls meet in the middle.  Use a thinner "
            f"wall."
        )
    return built


def occ_volume(shape) -> float:
    """Volume of an OCC solid in its own (scaled) units."""
    from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
    from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def find_edges_on_nearest_face(shape, point, scale: float = 1.0):
    """Find all edges of the face nearest to a 3D point.

    Parameters
    ----------
    shape : TopoDS_Shape
        An OCC solid or compound.
    point : tuple of float
        (x, y, z) target point [m].

    Returns
    -------
    list of TopoDS_Edge
        Unique edges of the nearest face.

    Raises
    ------
    ValueError
        If the shape has no faces.
    """
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for edge selection. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    nearest_face = find_nearest_face(shape, point, scale=scale)

    edges = []
    edge_explorer = TopExp_Explorer(nearest_face, TopAbs_EDGE)
    while edge_explorer.More():
        edge = topods.Edge(edge_explorer.Current())
        if not any(edge.IsSame(e) for e in edges):
            edges.append(edge)
        edge_explorer.Next()
    return edges


def resolve_edges(shape, *, near=None, face_near=None, edges=None, scale: float = 1.0):
    """Dispatch edge selection to the appropriate finder.

    Exactly one of *near*, *face_near*, or *edges* must be specified.

    Parameters
    ----------
    shape : TopoDS_Shape
        The OCC shape to select edges from.
    near : tuple or list of tuples, optional
        Point(s) near the desired edge(s).
    face_near : tuple, optional
        Point near the desired face (selects all its edges).
    edges : str, optional
        ``"all"`` to select every edge.

    Returns
    -------
    list of TopoDS_Edge

    Raises
    ------
    ValueError
        If zero or multiple selection modes are given, or *edges* has
        an invalid value.
    """
    modes = sum(x is not None for x in (near, face_near, edges))
    if modes != 1:
        raise ValueError("Exactly one of 'near', 'face_near', or 'edges' must be specified.")

    if edges is not None:
        if edges == "all":
            return get_all_edges(shape)
        raise ValueError(f"edges must be 'all'; got {edges!r}")

    if face_near is not None:
        return find_edges_on_nearest_face(shape, face_near, scale=scale)

    # near: single point (x,y,z) or list of points [(x,y,z), ...]
    if isinstance(near[0], (int, float)):
        return [find_nearest_edge(shape, near, scale=scale)]
    else:
        result = []
        for pt in near:
            edge = find_nearest_edge(shape, pt, scale=scale)
            if not any(edge.IsSame(e) for e in result):
                result.append(edge)
        return result


# ---------------------------------------------------------------------------
# Chamfer and fillet operations
# ---------------------------------------------------------------------------


def _find_face_for_edge(shape, edge):
    """Find a face adjacent to *edge* in *shape*."""
    try:
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for chamfer/fillet. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            e = topods.Edge(edge_explorer.Current())
            if e.IsSame(edge):
                return face
            edge_explorer.Next()
        face_explorer.Next()

    raise ValueError("No adjacent face found for edge.")


def make_chamfer(shape, edges, dist, scale: float = 1.0):
    """Apply chamfer to edges of an OCC shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        The solid to chamfer.
    edges : list of TopoDS_Edge
        Edges to chamfer (must belong to *shape*).
    dist : float or tuple of (float, float)
        Chamfer distance(s) [m].  A single float gives a symmetric chamfer.
        A tuple ``(d1, d2)`` gives an asymmetric chamfer where *d1* is
        measured on the reference face and *d2* on the opposite face.

    Returns
    -------
    TopoDS_Shape
        The chamfered solid.
    """
    try:
        from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for chamfer. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    chamfer_maker = BRepFilletAPI_MakeChamfer(shape)

    if isinstance(dist, (tuple, list)):
        for edge in edges:
            face = _find_face_for_edge(shape, edge)
            chamfer_maker.Add(float(dist[0]) * scale, float(dist[1]) * scale, edge, face)
    else:
        for edge in edges:
            chamfer_maker.Add(float(dist) * scale, edge)

    chamfer_maker.Build()
    if not chamfer_maker.IsDone():
        raise RuntimeError("OCC chamfer operation failed.")
    return chamfer_maker.Shape()


def make_fillet(shape, edges, radius, scale: float = 1.0):
    """Apply fillet to edges of an OCC shape.

    Parameters
    ----------
    shape : TopoDS_Shape
        The solid to fillet.
    edges : list of TopoDS_Edge
        Edges to fillet (must belong to *shape*).
    radius : float
        Fillet radius [m].

    Returns
    -------
    TopoDS_Shape
        The filleted solid.
    """
    try:
        from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for fillet. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    fillet_maker = BRepFilletAPI_MakeFillet(shape)
    for edge in edges:
        fillet_maker.Add(float(radius) * scale, edge)

    fillet_maker.Build()
    if not fillet_maker.IsDone():
        raise RuntimeError("OCC fillet operation failed.")
    return fillet_maker.Shape()


# ---------------------------------------------------------------------------
# Extrude and loft operations
# ---------------------------------------------------------------------------


def extract_face_wire(face):
    """Extract the outer wire of an OCC face.

    Parameters
    ----------
    face : TopoDS_Face
        An OCC face.

    Returns
    -------
    TopoDS_Wire
        The outer boundary wire.
    """
    try:
        from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for wire extraction. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    wire = breptools.OuterWire(face)
    if wire.IsNull():
        raise ValueError("Could not extract outer wire from face.")
    return wire


# Plane normal → (axis index, u index, v index).  Must match the (u, v)
# convention of ``cross_section_polygons`` so a Face and the cross-section
# of its extrusion share one coordinate frame.
_FACE_UV = {
    "x": (0, 1, 2),  # normal x: u=y, v=z
    "y": (1, 0, 2),  # normal y: u=x, v=z
    "z": (2, 0, 1),  # normal z: u=x, v=y
}


def make_face(normal: str, offset: float, points, scale: float = 1.0):
    """Build a planar OCC face from a polygon of ``(u, v)`` points.

    The polygon lies in the axis-normal plane at ``offset`` along *normal*;
    ``(u, v)`` map to the two in-plane axes following the same convention
    as :func:`cross_section_polygons` (normal ``x`` → u=y, v=z; ``y`` →
    u=x, v=z; ``z`` → u=x, v=y).  The polygon is closed automatically.

    Parameters
    ----------
    normal : str
        Plane normal axis: ``'x'``, ``'y'``, or ``'z'``.
    offset : float
        Position of the plane along the normal axis [m].
    points : sequence of (float, float)
        In-plane ``(u, v)`` vertices [m], at least 3, non-self-intersecting.

    Returns
    -------
    TopoDS_Face
        The planar face.
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
        )
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for face construction. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    if normal not in _FACE_UV:
        raise ValueError(f"normal must be 'x', 'y', or 'z'; got {normal!r}")
    pts = list(points)
    if len(pts) < 3:
        raise ValueError(f"A Face needs at least 3 points; got {len(pts)}.")

    axis_idx, u_idx, v_idx = _FACE_UV[normal]
    poly = BRepBuilderAPI_MakePolygon()
    for uv in pts:
        coord = [0.0, 0.0, 0.0]
        coord[axis_idx] = offset * scale
        coord[u_idx] = uv[0] * scale
        coord[v_idx] = uv[1] * scale
        poly.Add(occ["gp_Pnt"](*coord))
    poly.Close()
    if not poly.IsDone():
        raise ValueError(
            "OCC could not build a closed polygon from the given Face "
            "points (degenerate or duplicate vertices?)."
        )

    mkface = BRepBuilderAPI_MakeFace(poly.Wire(), True)  # True = planar only
    if not mkface.IsDone():
        raise ValueError(
            "OCC could not build a planar face from the Face polygon — "
            "the points must be coplanar and non-self-intersecting."
        )
    return mkface.Face()


def make_extrude(face, vector, scale: float = 1.0):
    """Extrude an OCC face along a direction vector to produce a solid.

    Parameters
    ----------
    face : TopoDS_Face
        The face to extrude.
    vector : tuple of float
        (dx, dy, dz) extrusion direction and length [m].

    Returns
    -------
    TopoDS_Shape
        The extruded solid.
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for extrude. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    prism = BRepPrimAPI_MakePrism(face, occ["gp_Vec"](*_scale3(vector, scale)))
    prism.Build()
    if not prism.IsDone():
        raise RuntimeError("OCC extrude (prism) operation failed.")
    return prism.Shape()


def make_loft(wires, is_solid=True, is_ruled=False):
    """Loft through a series of wire profiles to produce a solid.

    Parameters
    ----------
    wires : sequence of TopoDS_Wire
        Cross-section profiles in order (e.g. extracted via
        :func:`extract_face_wire`); at least two.
    is_solid : bool
        If True (default), produce a solid; otherwise a shell.
    is_ruled : bool
        If True, use ruled (straight-line) surfaces; otherwise smooth
        (spline) interpolation (default).

    Returns
    -------
    TopoDS_Shape
        The lofted solid (or shell).
    """
    try:
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for loft. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    wires = list(wires)
    if len(wires) < 2:
        raise ValueError(f"A loft needs at least 2 cross-sections; got {len(wires)}.")
    thru = BRepOffsetAPI_ThruSections(is_solid, is_ruled)
    for wire in wires:
        thru.AddWire(wire)
    thru.Build()
    if not thru.IsDone():
        raise RuntimeError("OCC loft (ThruSections) operation failed.")
    return thru.Shape()


def face_outward_normal(face):
    """Unit normal of *face*, pointing out of the solid it bounds.

    Unlike :func:`face_plane_normal`, which forces a reproducible sign
    for an offset direction, this reads the sign from the face's own
    orientation in its solid — the direction a blend has to leave along
    to meet the face at a right angle.

    Parameters
    ----------
    face : TopoDS_Face
        A face of a solid.  Planar faces give an exact normal; on a
        curved face the normal is taken at the middle of the parameter
        range, which is only representative.

    Returns
    -------
    tuple of float
        The outward unit normal ``(nx, ny, nz)``.

    Raises
    ------
    ValueError
        If the surface has no normal at the sampled point.
    """
    try:
        from OCC.Core.BRep import BRep_Tool  # noqa: PLC0415
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # noqa: PLC0415
        from OCC.Core.BRepTools import breptools  # noqa: PLC0415
        from OCC.Core.GeomAbs import GeomAbs_Plane  # noqa: PLC0415
        from OCC.Core.GeomLProp import GeomLProp_SLProps  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_REVERSED  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for face queries.") from exc

    surf = BRepAdaptor_Surface(face)
    if surf.GetType() == GeomAbs_Plane:
        direction = surf.Plane().Axis().Direction()
        normal = [direction.X(), direction.Y(), direction.Z()]
    else:
        u_min, u_max, v_min, v_max = breptools.UVBounds(face)
        props = GeomLProp_SLProps(
            BRep_Tool.Surface(face), 0.5 * (u_min + u_max), 0.5 * (v_min + v_max), 1, 1e-9
        )
        if not props.IsNormalDefined():
            raise ValueError(
                "The selected face has no well-defined normal, so there is "
                "no direction for the blend to leave it along."
            )
        direction = props.Normal()
        normal = [direction.X(), direction.Y(), direction.Z()]

    # The surface normal belongs to the underlying geometry; the face's
    # orientation is what turns it into an inward/outward statement.
    if face.Orientation() == TopAbs_REVERSED:
        normal = [-c for c in normal]
    return tuple(normal)


def make_tangent_blend(face_a, face_b, tension):
    """Blend two faces so the solid meets each of them at a right angle.

    Sweeps the outer wire of *face_a* into that of *face_b* along a cubic
    Bezier spine whose end tangents are the two outward face normals.
    ``BRepOffsetAPI_MakePipeShell`` keeps the profiles perpendicular to
    that spine, so the transition leaves both faces along their normal
    instead of meeting them at a crease.

    The spine is built from the faces as given, so it inherits whatever
    length unit they carry and needs no separate scale argument.

    Parameters
    ----------
    face_a, face_b : TopoDS_Face
        The two faces to bridge.
    tension : tuple of float
        Tangent stiffness ``(t_a, t_b)`` at each end, as a fraction of
        the centroid distance.  Larger values push the blend further
        along the normal before it turns.

    Returns
    -------
    TopoDS_Shape
        The blended solid.

    Raises
    ------
    ValueError
        If the two face centroids coincide, leaving no span to blend over.
    RuntimeError
        If the sweep fails or does not close into a solid.
    """
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
        )
        from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell  # noqa: PLC0415
        from OCC.Core.Geom import Geom_BezierCurve  # noqa: PLC0415
        from OCC.Core.gp import gp_Pnt  # noqa: PLC0415
        from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415
        from OCC.Core.TColgp import TColgp_Array1OfPnt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for a tangent blend. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    def centroid(face):
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        point = props.CentreOfMass()
        return (point.X(), point.Y(), point.Z())

    point_a, point_b = centroid(face_a), centroid(face_b)
    normal_a = face_outward_normal(face_a)
    normal_b = face_outward_normal(face_b)
    span = math.sqrt(sum((b - a) ** 2 for a, b in zip(point_a, point_b)))
    if span <= 0.0:
        raise ValueError(
            "The two faces share a centroid, so there is no span to blend "
            "over.  Pick faces that are apart from each other."
        )

    # Hermite end conditions written as a cubic Bezier: the two interior
    # control points sit along the outward normals, which is what makes
    # the spine — and with it the swept solid — leave each face squarely.
    tension_a, tension_b = tension
    control = [
        point_a,
        tuple(p + n * tension_a * span for p, n in zip(point_a, normal_a)),
        tuple(p + n * tension_b * span for p, n in zip(point_b, normal_b)),
        point_b,
    ]
    array = TColgp_Array1OfPnt(1, 4)
    for index, point in enumerate(control, start=1):
        array.SetValue(index, gp_Pnt(*point))
    edge = BRepBuilderAPI_MakeEdge(Geom_BezierCurve(array)).Edge()
    spine = BRepBuilderAPI_MakeWire(edge).Wire()

    pipe = BRepOffsetAPI_MakePipeShell(spine)
    # DD-144: corrected Frenet, not plain Frenet.  A plain Frenet frame
    # flips its normal at an inflection point, which a Bezier spine
    # between two arbitrarily posed faces can easily have.  On a planar
    # spine this agrees with a fixed binormal to every digit; it is the
    # non-coplanar case that needs the corrected frame.
    pipe.SetMode(False)
    pipe.Add(extract_face_wire(face_a), False, False)
    pipe.Add(extract_face_wire(face_b), False, False)
    pipe.Build()
    if not pipe.IsDone():
        raise RuntimeError("OCC tangent blend (MakePipeShell) operation failed.")
    if not pipe.MakeSolid():
        raise RuntimeError("OCC tangent blend did not close into a solid.")
    return pipe.Shape()


# ---------------------------------------------------------------------------
# Curves (1D loci) — polyline / arc / spline / helix → TopoDS_Wire
# ---------------------------------------------------------------------------


def make_polyline(points, scale: float = 1.0):
    """Build an open polyline wire through a sequence of 3D points."""
    occ = _require_occ()
    try:
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for curve construction.") from exc
    pts = list(points)
    if len(pts) < 2:
        raise ValueError(f"A polyline needs at least 2 points; got {len(pts)}.")
    poly = BRepBuilderAPI_MakePolygon()
    for p in pts:
        poly.Add(occ["gp_Pnt"](*_scale3(p, scale)))
    # Deliberately NOT closed — a Curve is an open locus.
    if not poly.IsDone():
        raise ValueError(
            "OCC could not build the polyline (duplicate or degenerate consecutive points?)."
        )
    return poly.Wire()


def make_arc(p_start, p_through, p_end, scale: float = 1.0):
    """Build a circular-arc wire through three 3D points."""
    occ = _require_occ()
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
        )
        from OCC.Core.GC import GC_MakeArcOfCircle  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for curve construction.") from exc
    arc = GC_MakeArcOfCircle(
        occ["gp_Pnt"](*_scale3(p_start, scale)),
        occ["gp_Pnt"](*_scale3(p_through, scale)),
        occ["gp_Pnt"](*_scale3(p_end, scale)),
    )
    if not arc.IsDone():
        raise ValueError(
            "OCC could not build a circular arc through the three points "
            "(are they collinear or coincident?)."
        )
    edge = BRepBuilderAPI_MakeEdge(arc.Value()).Edge()
    return BRepBuilderAPI_MakeWire(edge).Wire()


def make_spline(points, scale: float = 1.0):
    """Build a B-spline wire interpolating a sequence of 3D points."""
    occ = _require_occ()
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
        )
        from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline  # noqa: PLC0415
        from OCC.Core.TColgp import TColgp_Array1OfPnt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for curve construction.") from exc
    pts = list(points)
    if len(pts) < 2:
        raise ValueError(f"A spline needs at least 2 points; got {len(pts)}.")
    arr = TColgp_Array1OfPnt(1, len(pts))
    for i, p in enumerate(pts, 1):
        arr.SetValue(i, occ["gp_Pnt"](*_scale3(p, scale)))
    curve = GeomAPI_PointsToBSpline(arr).Curve()
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    return BRepBuilderAPI_MakeWire(edge).Wire()


def make_helix(
    radius, pitch, turns, origin=(0.0, 0.0, 0.0), axis="z", right_handed=True, scale: float = 1.0
):
    """Build an exact helical wire on a cylindrical surface.

    The helix is a straight line in the ``(angle, height)`` parameter space
    of a cylinder, realised as an edge on that surface — exact, not a
    sampled approximation.  ``breplib.BuildCurve3d`` is called so the edge
    carries a 3D curve (required by ``sweep``'s Frenet frame and for a tight
    bounding box).

    Parameters
    ----------
    radius : float
        Helix radius [m].
    pitch : float
        Axial rise per full turn [m].
    turns : float
        Number of turns (may be fractional).
    origin : tuple of float
        Base point on the axis [m].
    axis : str
        Axis direction: ``'x'``, ``'y'``, or ``'z'``.
    right_handed : bool
        If True (default) the helix ascends counter-clockwise about the
        axis; if False it is left-handed.
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeWire,
        )
        from OCC.Core.BRepLib import breplib  # noqa: PLC0415
        from OCC.Core.Geom import Geom_CylindricalSurface  # noqa: PLC0415
        from OCC.Core.Geom2d import Geom2d_Line  # noqa: PLC0415
        from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Dir2d, gp_Pnt2d  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for curve construction.") from exc
    _check_dimensions("Helix", scale, radius=radius)
    if pitch * scale <= _OCC_PRECISION:
        raise ValueError(f"Helix pitch must be positive; got {pitch:.3e} m.")
    if turns <= 0:
        raise ValueError(f"Helix turns must be positive; got {turns}.")
    _axis_dirs = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    if axis not in _axis_dirs:
        raise ValueError(f"Helix axis must be 'x', 'y', or 'z'; got {axis!r}")

    ax3 = gp_Ax3(occ["gp_Pnt"](*_scale3(origin, scale)), gp_Dir(*_axis_dirs[axis]))
    cyl = Geom_CylindricalSurface(ax3, radius * scale)
    u_sign = 1.0 if right_handed else -1.0
    line2d = Geom2d_Line(gp_Pnt2d(0.0, 0.0), gp_Dir2d(u_sign * 2.0 * math.pi, pitch * scale))
    # Unit 2D direction advances |u| by 2*pi per hypot(2*pi, pitch) of
    # parameter, so ``turns`` turns span this length.
    last = turns * math.hypot(2.0 * math.pi, pitch * scale)
    edge = BRepBuilderAPI_MakeEdge(line2d, cyl, 0.0, last).Edge()
    breplib.BuildCurve3d(edge)
    return BRepBuilderAPI_MakeWire(edge).Wire()


def _wire_edges(wire):
    """The edges of an OCC wire, in wire order."""
    from OCC.Core.TopAbs import TopAbs_EDGE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    edges = []
    explorer = TopExp_Explorer(wire, TopAbs_EDGE)
    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()
    return edges


def make_joined_wire(wires, tol: float):
    """Chain a sequence of OCC wires end to start into a single wire.

    ``BRepBuilderAPI_MakeWire.Add`` only fuses vertices that already lie
    within the vertices' own tolerance (``Precision::Confusion()``), which
    is far tighter than the public join tolerance.  Endpoints that agree
    exactly therefore take the fast path; a chain whose seams are merely
    *within tolerance* is healed once through ``ShapeFix_Wire``, which is
    not run unconditionally because it may reorder or reverse edges.

    Parameters
    ----------
    wires : sequence of TopoDS_Wire
        At least one wire, in chain order.
    tol : float
        Seam tolerance in **scaled model units**.

    Returns
    -------
    TopoDS_Wire
        The chained wire.
    """
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_DisconnectedWire,
            BRepBuilderAPI_EmptyWire,
            BRepBuilderAPI_MakeWire,
            BRepBuilderAPI_NonManifoldWire,
        )
        from OCC.Core.ShapeExtend import ShapeExtend_WireData  # noqa: PLC0415
        from OCC.Core.ShapeFix import ShapeFix_Wire  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for curve construction.") from exc

    wires = list(wires)
    if not wires:
        raise ValueError("Joining needs at least one curve.")
    if len(wires) == 1:
        return wires[0]

    maker = BRepBuilderAPI_MakeWire()
    for wire in wires:
        maker.Add(wire)
    if maker.IsDone():
        return maker.Wire()

    error = maker.Error()
    if error == BRepBuilderAPI_NonManifoldWire:
        raise ValueError(
            "Joining these curves produces a branching (non-manifold) "
            "wire: more than two segments meet at one point.  A chain "
            "must run through each junction exactly once."
        )
    if error == BRepBuilderAPI_EmptyWire:
        raise ValueError("Joining produced an empty wire — one of the curves is degenerate.")
    if error != BRepBuilderAPI_DisconnectedWire:
        raise ValueError(f"OCC could not chain the curves (wire error {error}).")

    # Disconnected: seams may still be inside the public tolerance, which
    # is wider than the kernel's vertex confusion.  Heal once.
    wire_data = ShapeExtend_WireData()
    for wire in wires:
        for edge in _wire_edges(wire):
            wire_data.Add(edge)
    fixer = ShapeFix_Wire()
    fixer.Load(wire_data)
    fixer.SetPrecision(tol)
    fixer.SetMaxTolerance(tol)
    if not fixer.FixConnected():
        raise ValueError(
            "The curves do not form a chain: consecutive segments must "
            "meet end to start.  Check the segment order and the "
            "coordinates of the seams."
        )
    healed = fixer.Wire()
    if healed.IsNull():
        raise ValueError("OCC could not chain the curves into a single wire.")
    return healed


def make_wire_face(wire):
    """Build the planar face bounded by a closed OCC wire.

    The free-boundary counterpart of :func:`make_face`, which is limited
    to axis-normal polygons: any closed planar wire — arcs, splines and
    straight segments mixed — becomes a face here.

    Parameters
    ----------
    wire : TopoDS_Wire
        A closed, planar, non-self-intersecting boundary.

    Returns
    -------
    TopoDS_Face
        The planar face.
    """
    try:
        from OCC.Core.BRepBuilderAPI import (  # noqa: PLC0415
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_NotPlanar,
        )
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for face construction.") from exc

    if not wire.Closed():
        raise ValueError(
            "covered() needs a closed curve; this one has two loose ends. "
            "Chain the segments with joined() so the last end meets the "
            "first start, or build the profile with Path(...).closed()."
        )
    mkface = BRepBuilderAPI_MakeFace(wire, True)  # True = planar only
    if not mkface.IsDone():
        if mkface.Error() == BRepBuilderAPI_NotPlanar:
            raise ValueError(
                "covered() needs a planar curve — these segments do not lie in one plane."
            )
        raise ValueError(
            "OCC could not build a face from this curve — is the boundary self-intersecting?"
        )
    return mkface.Face()


def _shape_wires(shape):
    """The wires of an OCC shape."""
    from OCC.Core.TopAbs import TopAbs_WIRE  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    out = []
    explorer = TopExp_Explorer(shape, TopAbs_WIRE)
    while explorer.More():
        out.append(topods.Wire(explorer.Current()))
        explorer.Next()
    return out


def wire_plane_normal(wire):
    """Canonical unit normal of a planar wire's plane, or None.

    ``None`` means the plane is not determined by the wire itself — a
    straight segment lies in infinitely many planes.
    """
    from OCC.Core.BRepLib import BRepLib_FindSurface  # noqa: PLC0415
    from OCC.Core.Geom import Geom_Plane  # noqa: PLC0415

    finder = BRepLib_FindSurface(wire, -1.0, True)  # True = only planes
    if not finder.Found():
        return None
    plane = Geom_Plane.DownCast(finder.Surface())
    if plane is None:
        return None
    direction = plane.Pln().Axis().Direction()
    normal = [direction.X(), direction.Y(), direction.Z()]
    dominant = max(range(3), key=lambda i: abs(normal[i]))
    if normal[dominant] < 0.0:
        normal = [-c for c in normal]
    return tuple(normal)


def _wire_is_straight(wire) -> bool:
    """Whether a wire's points are all collinear.

    Separates the two reasons a wire has no plane of its own: a straight
    run lies in infinitely many, a genuinely 3D path in none.
    """
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve  # noqa: PLC0415
    from OCC.Core.GCPnts import GCPnts_QuasiUniformAbscissa  # noqa: PLC0415

    comp = BRepAdaptor_CompCurve(wire)
    sampler = GCPnts_QuasiUniformAbscissa(comp, 12)
    if not sampler.IsDone() or sampler.NbPoints() < 3:
        return True
    points = []
    for i in range(1, sampler.NbPoints() + 1):
        p = comp.Value(sampler.Parameter(i))
        points.append((p.X(), p.Y(), p.Z()))
    first, last = points[0], points[-1]
    span = [b - a for a, b in zip(first, last)]
    length = math.sqrt(sum(c * c for c in span))
    if length <= 0.0:
        return True
    unit = [c / length for c in span]
    for point in points[1:-1]:
        rel = [b - a for a, b in zip(first, point)]
        along = sum(a * b for a, b in zip(rel, unit))
        perp = [c - along * u for c, u in zip(rel, unit)]
        if math.sqrt(sum(c * c for c in perp)) > 1e-9 * length:
            return False
    return True


def _offset_outline(spine_face, half_width, wire=None):
    """One closed offset contour of a spine, at distance *half_width*.

    *spine_face* anchors the plane; *wire* is the open spine to add to it
    (for a closed spine the face is the spine itself).
    """
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeOffset  # noqa: PLC0415
    from OCC.Core.GeomAbs import GeomAbs_Arc  # noqa: PLC0415

    maker = BRepOffsetAPI_MakeOffset()
    maker.Init(spine_face, GeomAbs_Arc, False)
    if wire is not None:
        maker.AddWire(wire)
    maker.Perform(half_width)
    built = maker.Shape()
    contours = _shape_wires(built) if built is not None else []
    if len(contours) != 1:
        raise ValueError(
            "The trace outline could not be built: widening this curve "
            "makes its own sides run into each other.  Reduce width, or "
            "open up the tightest bend and the smallest clearance along "
            "the path."
        )
    return contours[0]


def _end_halfspace(comp_curve, parameter, outward: float, size: float):
    """A large box covering everything beyond one end of a spine."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: PLC0415
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec  # noqa: PLC0415

    point, tangent = gp_Pnt(), gp_Vec()
    comp_curve.D1(parameter, point, tangent)
    if tangent.Magnitude() <= 1e-30:
        raise ValueError("Degenerate trace path — zero tangent at an end.")
    direction = gp_Dir(tangent.X() * outward, tangent.Y() * outward, tangent.Z() * outward)
    frame = gp_Ax2(point, direction)
    x_dir, y_dir = frame.XDirection(), frame.YDirection()
    # Shift the box laterally so it straddles the end plane; it then
    # covers the whole half-space beyond it.
    base = gp_Pnt(
        point.X() - 0.5 * size * (x_dir.X() + y_dir.X()),
        point.Y() - 0.5 * size * (x_dir.Y() + y_dir.Y()),
        point.Z() - 0.5 * size * (x_dir.Z() + y_dir.Z()),
    )
    return BRepPrimAPI_MakeBox(gp_Ax2(base, direction, x_dir), size, size, size).Shape()


def make_trace(wire, width: float, thickness: float, caps="round", normal=None, scale: float = 1.0):
    """Build a conductor track running along a planar centreline wire.

    The centreline is widened by ``width/2`` to each side within its own
    plane and the resulting outline extruded by *thickness* along the
    plane normal.

    Parameters
    ----------
    wire : TopoDS_Wire
        The planar centreline, open or closed.
    width, thickness : float
        Track width and metal thickness [m]; *thickness* is signed.
    caps : {"round", "flat"}
        End treatment of an open centreline.
    normal : sequence of float, optional
        Unit normal of the track's plane.  Required when the wire alone
        does not determine one (a straight centreline).
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve  # noqa: PLC0415
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace  # noqa: PLC0415
        from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Pln  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for trace construction.") from exc

    _check_dimensions("Trace", scale, width=width, thickness=abs(thickness))
    half_width = 0.5 * width * scale
    is_closed = wire.Closed()

    if normal is None:
        normal = wire_plane_normal(wire)
        if normal is None:
            if _wire_is_straight(wire):
                raise ValueError(
                    "traced() cannot tell which plane this track lies in: "
                    "a straight centreline lies in infinitely many.  Name "
                    "the plane with normal=."
                )
            raise ValueError(
                "traced() needs a planar centreline, but this curve does "
                "not lie in one plane.  A track has a single metal plane; "
                "build a non-planar run from planar pieces."
            )
    else:
        wire_normal = wire_plane_normal(wire)
        if wire_normal is not None:
            skew = abs(sum(a * b for a, b in zip(normal, wire_normal)))
            if skew < 1.0 - 1e-6:
                raise ValueError(
                    "traced(normal=...) does not match the plane this "
                    "curve lies in.  Leave normal= out to use the curve's "
                    "own plane."
                )

    if is_closed:
        spine = BRepBuilderAPI_MakeFace(wire, True)
        if not spine.IsDone():
            raise ValueError("traced() needs a planar centreline.")
        spine_face = spine.Face()
        outer = _offset_outline(spine_face, half_width)
        inner = _offset_outline(spine_face, -half_width)
        outline = BRepBuilderAPI_MakeFace(outer, True)
        outline.Add(inner.Reversed())
    else:
        comp = BRepAdaptor_CompCurve(wire)
        start = occ["gp_Pnt"]()
        comp.D0(comp.FirstParameter(), start)
        anchor = BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(start, gp_Dir(*normal)))).Face()
        outline = BRepBuilderAPI_MakeFace(_offset_outline(anchor, half_width, wire), True)
    if not outline.IsDone():
        raise ValueError("OCC could not build the trace outline from this curve.")

    solid = make_extrude(outline.Face(), tuple(thickness * c for c in normal), scale=scale)

    if not is_closed and caps == "flat":
        comp = BRepAdaptor_CompCurve(wire)
        size = 4.0 * _occ_bbox_diagonal(solid)
        for parameter, outward in (
            (comp.FirstParameter(), -1.0),
            (comp.LastParameter(), 1.0),
        ):
            solid = boolean_difference(solid, _end_halfspace(comp, parameter, outward, size))
    return solid


def _occ_bbox_diagonal(shape) -> float:
    """Diagonal of an OCC shape's bounding box, in its own units."""
    occ = _require_occ()
    box = occ["Bnd_Box"]()
    # Geometry-only box (no triangulation) — KB-012.
    occ["brepbndlib"].Add(shape, box, False)
    x_min, y_min, z_min, x_max, y_max, z_max = box.Get()
    return math.dist((x_min, y_min, z_min), (x_max, y_max, z_max))


def make_revolve(profile_face, axis_point, axis_dir, angle_rad, scale: float = 1.0):
    """Revolve a profile face about an axis to produce a solid.

    Parameters
    ----------
    profile_face : TopoDS_Face
        The planar profile to revolve.
    axis_point : tuple of float
        A point on the revolution axis [m].
    axis_dir : tuple of float
        Direction of the revolution axis (need not be normalised).
    angle_rad : float
        Revolution angle [radians]; ``2*pi`` for a full revolution.

    Returns
    -------
    TopoDS_Shape
        The revolved solid.
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol  # noqa: PLC0415
        from OCC.Core.gp import gp_Ax1, gp_Dir  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for revolve.") from exc
    ax1 = gp_Ax1(occ["gp_Pnt"](*_scale3(axis_point, scale)), gp_Dir(*axis_dir))
    rev = BRepPrimAPI_MakeRevol(profile_face, ax1, angle_rad)
    rev.Build()
    if not rev.IsDone():
        raise RuntimeError("OCC revolve (MakeRevol) operation failed.")
    return rev.Shape()


def _perp_dir(d):
    """A deterministic unit direction perpendicular to ``gp_Dir`` *d*.

    Used to fix the in-plane roll of the swept profile's frame.  Projects a
    global reference axis (switching axes when *d* is nearly parallel to it)
    into the plane normal to *d*.
    """
    from OCC.Core.gp import gp_Dir, gp_Vec  # noqa: PLC0415

    ref = gp_Dir(0.0, 0.0, 1.0)
    if abs(d.Dot(ref)) > 0.9:
        ref = gp_Dir(1.0, 0.0, 0.0)
    v = gp_Vec(ref) - gp_Vec(d) * ref.Dot(d)
    return gp_Dir(v)


def make_sweep(profile_face, spine_wire):
    """Sweep a planar profile face along a spine wire to produce a solid.

    ``BRepOffsetAPI_MakePipe`` uses the profile at the position it already
    occupies, so the profile is first rigidly moved so its centroid lands
    on the spine's start point and its plane normal aligns with the spine's
    start tangent (its in-plane roll fixed deterministically).  The result
    is a tube centred on the spine, oriented by the pipe's own trihedron
    along the path.

    Parameters
    ----------
    profile_face : TopoDS_Face
        A planar profile face.
    spine_wire : TopoDS_Wire
        The path to sweep along.

    Returns
    -------
    TopoDS_Shape
        The swept solid.
    """
    occ = _require_occ()
    try:
        from OCC.Core.BRepAdaptor import (  # noqa: PLC0415
            BRepAdaptor_CompCurve,
            BRepAdaptor_Surface,
        )
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform  # noqa: PLC0415
        from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
        from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe  # noqa: PLC0415
        from OCC.Core.GeomAbs import GeomAbs_Plane  # noqa: PLC0415
        from OCC.Core.gp import gp_Ax3, gp_Dir, gp_Trsf, gp_Vec  # noqa: PLC0415
        from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for sweep.") from exc

    # Profile plane normal + centroid.
    surf = BRepAdaptor_Surface(profile_face)
    if surf.GetType() != GeomAbs_Plane:
        raise ValueError("sweep profile must be a planar face.")
    normal = surf.Plane().Axis().Direction()
    gprops = GProp_GProps()
    brepgprop.SurfaceProperties(profile_face, gprops)
    centroid = gprops.CentreOfMass()

    # Spine start point + unit tangent (CompCurve handles multi-edge wires).
    comp = BRepAdaptor_CompCurve(spine_wire)
    start_pnt = occ["gp_Pnt"]()
    tangent_vec = gp_Vec()
    comp.D1(comp.FirstParameter(), start_pnt, tangent_vec)
    if tangent_vec.Magnitude() <= 1e-30:
        raise ValueError("degenerate spine — zero start tangent.")
    tangent = gp_Dir(tangent_vec)

    # Rigidly move the profile from its own frame to the spine-start frame.
    frame_from = gp_Ax3(centroid, normal, _perp_dir(normal))
    frame_to = gp_Ax3(start_pnt, tangent, _perp_dir(tangent))
    trsf = gp_Trsf()
    trsf.SetDisplacement(frame_from, frame_to)
    moved = BRepBuilderAPI_Transform(profile_face, trsf, True).Shape()

    pipe = BRepOffsetAPI_MakePipe(spine_wire, moved)
    pipe.Build()
    if not pipe.IsDone():
        raise RuntimeError("OCC sweep (MakePipe) operation failed.")
    return pipe.Shape()


# ---------------------------------------------------------------------------
# Pairwise overlap detection
# ---------------------------------------------------------------------------


def check_pairwise_overlaps(
    shapes: list,
    tolerance: float | None = None,
    materials: list | None = None,
    scale: float = 1.0,
) -> list[tuple[int, int, float]]:
    """Check all shape pairs for volumetric overlap.

    Uses bounding-box pre-filter to skip pairs whose AABBs do not
    intersect, then computes the Boolean intersection volume via
    ``BRepAlgoAPI_Common`` + ``BRepGProp.VolumeProperties`` for the
    remaining pairs.

    Parameters
    ----------
    shapes : list
        Each element must have ``._occ_shape()`` and ``.bounding_box()``.
    tolerance : float or None
        Intersection volumes below this threshold [m^3] are ignored.
        ``None`` (default) uses a scale-free per-pair threshold of
        ``1e-12 * min(AABB volume of the pair)`` — Boolean float dust
        stays invisible at any model scale, a real overlap of a tiny
        shape does not.  An explicit float is honored as an absolute
        volume in m^3.
    scale : float
        DD-120 model scale factor at which the shapes are built.
    materials : list or None
        Optional per-shape materials (same order as *shapes*).  When
        given, pairs whose materials are **value-equal** (``Material.__eq__``,
        not identity) are skipped *before* the Boolean — a same-material
        overlap is physically unambiguous (the overlap region gets that
        material either way), so it is allowed and costs no intersection
        computation.  ``None`` (default) reports every overlap.

    Returns
    -------
    list of (int, int, float)
        ``(i, j, volume)`` for each pair with nonzero overlap.
    """
    try:
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common  # noqa: PLC0415
        from OCC.Core.BRepGProp import brepgprop  # noqa: PLC0415
        from OCC.Core.GProp import GProp_GProps  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("pythonocc-core is required for overlap detection.") from exc

    n = len(shapes)
    if n < 2:
        return []

    # Pre-compute bounding boxes [m]
    bboxes = []
    aabb_volumes = []
    for s in shapes:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = s.bounding_box(scale)
        bboxes.append((xmin, ymin, zmin, xmax, ymax, zmax))
        aabb_volumes.append((xmax - xmin) * (ymax - ymin) * (zmax - zmin))

    # Pre-compute OCC shapes lazily (only when needed)
    occ_cache: dict[int, object] = {}

    def get_occ(idx):
        if idx not in occ_cache:
            occ_cache[idx] = shapes[idx]._occ_shape(scale)
        return occ_cache[idx]

    overlaps: list[tuple[int, int, float]] = []
    gprops = GProp_GProps()

    for i in range(n):
        for j in range(i + 1, n):
            # AABB rejection
            a, b = bboxes[i], bboxes[j]
            if (
                a[0] >= b[3]
                or b[0] >= a[3]
                or a[1] >= b[4]
                or b[1] >= a[4]
                or a[2] >= b[5]
                or b[2] >= a[5]
            ):
                continue

            # Same-material overlaps are allowed (value equality, not
            # identity) — skip before the expensive Boolean intersection.
            if materials is not None and materials[i] == materials[j]:
                continue

            # Explicit SetArguments/SetTools/Build, not the two-shape
            # constructor: that one builds immediately, before
            # ``keep_operands_intact`` could take effect — and this
            # Boolean runs on the user's cached solids (DD-146).
            common = BRepAlgoAPI_Common()
            common.SetArguments(_shape_list([get_occ(i)]))
            common.SetTools(_shape_list([get_occ(j)]))
            keep_operands_intact(common)
            common.Build()
            if not common.IsDone():
                continue

            brepgprop.VolumeProperties(common.Shape(), gprops)
            volume = abs(gprops.Mass()) / scale**3

            pair_tol = (
                tolerance
                if tolerance is not None
                else 1e-12 * min(aabb_volumes[i], aabb_volumes[j])
            )
            if volume > pair_tol:
                overlaps.append((i, j, volume))

    return overlaps


def sample_wire(wire, max_segment_length: float, scale: float = 1.0) -> np.ndarray:
    """Sample an ordered polyline of 3D points along a ``TopoDS_Wire``.

    The wire is treated as one composite curve — ``BRepAdaptor_CompCurve``
    resolves the edge order and per-edge orientation — and points are placed
    at quasi-uniform curvilinear abscissa no more than *max_segment_length*
    apart, so a downstream rasteriser never skips a grid node between
    consecutive samples.

    Parameters
    ----------
    wire : TopoDS_Wire
        The wire to sample, built at *scale* (e.g. ``Curve._occ_shape(scale)``).
    max_segment_length : float
        Maximum arc-length spacing between consecutive samples [m].
    scale : float
        DD-120 model scale factor of the wire.

    Returns
    -------
    np.ndarray, shape ``(N, 3)``
        Ordered points [m] from the wire start to its end, ``N >= 2``.
    """
    if max_segment_length <= 0.0:
        raise ValueError(f"max_segment_length must be positive; got {max_segment_length}")
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve  # noqa: PLC0415
        from OCC.Core.GCPnts import (  # noqa: PLC0415
            GCPnts_AbscissaPoint,
            GCPnts_QuasiUniformAbscissa,
        )
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for wire sampling. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    comp = BRepAdaptor_CompCurve(wire)
    length = GCPnts_AbscissaPoint.Length(comp)
    if length <= 0.0:
        raise ValueError("degenerate wire — zero arc length.")

    n = max(2, int(math.ceil(length / (max_segment_length * scale))) + 1)
    ua = GCPnts_QuasiUniformAbscissa(comp, n)
    if not ua.IsDone():
        raise RuntimeError("wire sampling failed (GCPnts_QuasiUniformAbscissa).")

    pts = np.empty((ua.NbPoints(), 3), dtype=float)
    for i in range(1, ua.NbPoints() + 1):
        pnt = comp.Value(ua.Parameter(i))
        pts[i - 1] = (pnt.X(), pnt.Y(), pnt.Z())
    return pts / scale


def wire_vertex_points(wire, scale: float = 1.0) -> np.ndarray:
    """The 3D positions of a ``TopoDS_Wire``'s topological vertices.

    A polyline wire yields its corner points (each interior vertex once
    per adjacent edge — duplicates are harmless for plane extraction);
    smooth curves yield only their end vertices.  Used by the mesher to
    anchor grid planes on a thin wire's axis-aligned segments.

    Returns
    -------
    np.ndarray, shape ``(N, 3)``
    """
    try:
        from OCC.Core.BRep import BRep_Tool  # noqa: PLC0415
        from OCC.Core.TopAbs import TopAbs_VERTEX  # noqa: PLC0415
        from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
        from OCC.Core.TopoDS import topods  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for wire vertex extraction. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc

    pts: list[tuple[float, float, float]] = []
    ex = TopExp_Explorer(wire, TopAbs_VERTEX)
    while ex.More():
        p = BRep_Tool.Pnt(topods.Vertex(ex.Current()))
        pts.append((p.X(), p.Y(), p.Z()))
        ex.Next()
    return np.asarray(pts, dtype=float).reshape(-1, 3) / scale
