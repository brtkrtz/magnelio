"""Automatic internal unit scaling for the OCC boundary (DD-120).

The public API is SI meters everywhere, but the OCC kernel has a fixed
model-unit precision (``Precision::Confusion()`` = 1e-7): geometry whose
features approach that scale degrades or fails outright.  This module
chooses a per-model **power-of-two** scale factor ``s`` that brings the
model into the kernel's comfortable range; the backend multiplies every
coordinate handed to OCC by ``s`` and converts every result back to
meters (lengths /s, areas /s^2, volumes /s^3).

Design constraints (DD-120):

* ``s`` is a pure function of the shape set — no global state, no
  model-owned cache.  It is threaded explicitly (``scale=`` keyword)
  through ``_occ_shape()`` and every backend entry point.
* The bounding box that determines ``s`` is computed **analytically**
  from primitive parameters and transform algebra, never via OCC —
  otherwise choosing ``s`` would itself require building OCC shapes at
  an unknown scale.  Conservative (containing) boxes are sufficient:
  only the order of magnitude matters.
* ``s`` is a power of two, so scaling and unscaling are lossless in
  IEEE-754 and round-trip bit-exactly.
* Models whose diagonal already sits in the identity band get ``s = 1``
  — existing meter/mm-scale models are bit-identical to the unscaled
  code path.
"""

from __future__ import annotations

import math

# Model diagonals [m] that need no scaling: s = 1.  The band is wide on
# purpose — OCC is comfortable anywhere within ~9 decades of its 1e-7
# precision, and s = 1 keeps every existing model bit-identical.
_IDENTITY_BAND = (1e-3, 1e4)

# Scaled-unit diagonal aimed for outside the identity band.  128 = 2^7
# puts typical feature sizes (1e-3..1e-1 of the diagonal) five to seven
# decades above Precision::Confusion().
_TARGET_DIAG = 128.0

Box = tuple[tuple[float, float, float], tuple[float, float, float]]


def choose_scale(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> float:
    """Power-of-two scale factor for a model with bounding box (lo, hi).

    Returns 1.0 inside the identity band and for degenerate boxes
    (zero, NaN, or infinite diagonal).
    """
    diag = math.dist(lo, hi)
    if not math.isfinite(diag) or diag <= 0.0:
        return 1.0
    if _IDENTITY_BAND[0] <= diag <= _IDENTITY_BAND[1]:
        return 1.0
    return 2.0 ** round(math.log2(_TARGET_DIAG / diag))


def fine_detail_scale(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> float:
    """Power-of-two scale for a model whose features are far below its size.

    :func:`choose_scale` leaves models in the identity band alone, on
    the assumption that their smallest features are within about three
    decades of their overall size.  A printed circuit board breaks that
    assumption by two decades — 35 µm of copper and 100 µm gaps on a
    100 mm board — and lands its Boolean operations three decades from
    the kernel's confusion tolerance, where fusing coplanar faces starts
    to drop slivers.  Scaling the diagonal to the same target the
    band-external case aims for restores the headroom; the factor is a
    power of two, so scaling back to meters afterwards is exact.
    """
    diag = math.dist(lo, hi)
    if not math.isfinite(diag) or diag <= 0.0:
        return 1.0
    return 2.0 ** round(math.log2(_TARGET_DIAG / diag))


def analytic_bbox(shape) -> Box:
    """Conservative OCC-free axis-aligned bounding box of *shape* [m].

    Dispatches to the shape's ``_analytic_bbox()`` method; a Group (any
    object exposing ``members()``) is bounded by the union of its
    members.  The box is guaranteed to *contain* the true shape, not to
    be tight — rotations, lofts and sweeps are bounded generously.
    """
    members = getattr(shape, "members", None)
    if members is not None:
        return union_boxes([analytic_bbox(m) for m in members()])
    return shape._analytic_bbox()


def model_scale(shapes) -> float:
    """The scale factor shared by all shapes of one model.

    Pure and cheap — recompute at every use site instead of caching, so
    a shape added after a first mesh build is always accounted for.
    """
    boxes = [analytic_bbox(s) for s in shapes]
    if not boxes:
        return 1.0
    return choose_scale(*union_boxes(boxes))


# ---------------------------------------------------------------------------
# Box algebra helpers for the per-class _analytic_bbox() implementations
# ---------------------------------------------------------------------------


def box_of_points(points) -> Box:
    """AABB of an iterable of 3D points."""
    pts = [tuple(p) for p in points]
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return lo, hi


def union_boxes(boxes) -> Box:
    """AABB containing every box in *boxes* (non-empty)."""
    lo = tuple(min(b[0][i] for b in boxes) for i in range(3))
    hi = tuple(max(b[1][i] for b in boxes) for i in range(3))
    return lo, hi


def intersect_boxes(box_a: Box, box_b: Box) -> Box:
    """AABB intersection; falls back to the union when disjoint.

    The fallback keeps the result conservative (containing) even for
    boxes that only appear disjoint because both inputs are themselves
    conservative over-estimates.
    """
    lo = tuple(max(box_a[0][i], box_b[0][i]) for i in range(3))
    hi = tuple(min(box_a[1][i], box_b[1][i]) for i in range(3))
    if any(lo[i] > hi[i] for i in range(3)):
        return union_boxes([box_a, box_b])
    return lo, hi


def translate_box(box: Box, vector) -> Box:
    lo, hi = box
    return (
        tuple(c + v for c, v in zip(lo, vector)),
        tuple(c + v for c, v in zip(hi, vector)),
    )


def pad_box(box: Box, pad: float) -> Box:
    lo, hi = box
    return tuple(c - pad for c in lo), tuple(c + pad for c in hi)


def box_diagonal(box: Box) -> float:
    return math.dist(box[0], box[1])


def corners_of_box(box: Box):
    """The 8 corner points of a box."""
    lo, hi = box
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def rotate_box(box: Box, axis, angle_deg: float, origin) -> Box:
    """AABB of *box* rotated about (*origin*, *axis*) by *angle_deg*.

    Rotating the 8 corners and re-boxing is conservative: the rotated
    box contains the rotated shape, and the AABB of its corners is the
    AABB of the rotated box.
    """
    ux, uy, uz = axis
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    def rot(p):
        px, py, pz = (c - o for c, o in zip(p, origin))
        dot = ux * px + uy * py + uz * pz
        cx, cy, cz = uy * pz - uz * py, uz * px - ux * pz, ux * py - uy * px
        rx = px * cos_a + cx * sin_a + ux * dot * (1.0 - cos_a)
        ry = py * cos_a + cy * sin_a + uy * dot * (1.0 - cos_a)
        rz = pz * cos_a + cz * sin_a + uz * dot * (1.0 - cos_a)
        return rx + origin[0], ry + origin[1], rz + origin[2]

    return box_of_points([rot(p) for p in corners_of_box(box)])


def mirror_box(box: Box, normal, position: float) -> Box:
    """AABB of *box* mirrored across the plane ``p . normal == position``.

    *normal* must be a unit vector.  Mirroring the 8 corners and
    re-boxing is exact for a plane normal to a coordinate axis and
    conservative otherwise — the same contract as :func:`rotate_box`.
    """

    def mir(p):
        dist = sum(c * n for c, n in zip(p, normal)) - position
        return tuple(c - 2.0 * dist * n for c, n in zip(p, normal))

    return box_of_points([mir(p) for p in corners_of_box(box)])


def scale_box(box: Box, factor: float, center) -> Box:
    def sc(p):
        return tuple(o + factor * (c - o) for c, o in zip(p, center))

    return box_of_points([sc(box[0]), sc(box[1])])


def axis_segment_box(origin, axis, length: float, radial: float) -> Box:
    """AABB of a cylinder-like solid: axis segment padded radially.

    Conservative for cylinders, cones and revolved solids whose points
    all lie within *radial* of the segment ``origin .. origin + length *
    axis`` (exact in the radial directions for axis-aligned axes).
    """
    end = tuple(o + length * a for o, a in zip(origin, axis))
    return pad_box(box_of_points([origin, end]), radial)


def sector_uv_points(start_deg: float, end_deg: float, r_outer: float, r_inner: float = 0.0):
    """In-plane points whose AABB is exactly that of a circular sector.

    Covers both the pie slice (``r_inner = 0``, apex included) and the
    annular slice.  The extremes of a sector sit either on the arc ends,
    on the apex/inner arc, or where the arc crosses a coordinate axis —
    all of which are enumerated here, so the resulting box is tight, not
    merely containing.
    """
    points = [] if r_inner > 0.0 else [(0.0, 0.0)]
    radii = (r_outer,) if r_inner <= 0.0 else (r_inner, r_outer)
    for radius in radii:
        for angle in (start_deg, end_deg):
            t = math.radians(angle)
            points.append((radius * math.cos(t), radius * math.sin(t)))
    quarter = math.ceil(start_deg / 90.0)
    while quarter * 90.0 <= end_deg:
        t = math.radians(quarter * 90.0)
        points.append((r_outer * math.cos(t), r_outer * math.sin(t)))
        quarter += 1
    return points


def circumcircle(p_a, p_b, p_c):
    """(center, radius) of the circle through three 3D points.

    Returns ``None`` for (near-)collinear points — the caller falls back
    to a padded point hull.
    """
    u = tuple(b - a for a, b in zip(p_a, p_b))
    v = tuple(c - a for a, c in zip(p_a, p_c))
    w = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    w2 = w[0] ** 2 + w[1] ** 2 + w[2] ** 2
    u2 = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
    v2 = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if w2 <= 1e-24 * u2 * v2 or w2 == 0.0:
        return None
    wxu = (
        w[1] * u[2] - w[2] * u[1],
        w[2] * u[0] - w[0] * u[2],
        w[0] * u[1] - w[1] * u[0],
    )
    vxw = (
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    )
    center = tuple(a + (v2 * cu + u2 * cv) / (2.0 * w2) for a, cu, cv in zip(p_a, wxu, vxw))
    radius = math.dist(center, p_a)
    return center, radius
