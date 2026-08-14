"""
2D polygon utilities for conformal material matrix computation.

Provides polygon area (shoelace), Sutherland-Hodgman clipping against
axis-aligned rectangles, and point-in-polygon classification.

All functions operate on (N, 2) NumPy arrays of vertex coordinates.

The hot-path kernels (``polygon_area``, ``_clip_edge``,
``clip_polygon_to_rect``, ``points_in_polygon``) are JIT-compiled with
Numba when available (profiling on a 1000-primitive CSG geometry:
``compute_face_material_areas`` calls ``clip_polygon_to_rect`` and
``_clip_edge`` millions of times over small polygons, where Python/NumPy
per-call dispatch overhead dominates over the actual arithmetic).
``njit`` is a no-op decorator when Numba is unavailable, so the exact
same code runs interpreted — same algorithm, same floating-point
results, either way.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range

    def njit(*args, **kwargs):  # noqa: ANN002, ANN003
        """No-op stand-in for ``numba.njit`` when Numba is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


@njit(cache=True)
def polygon_area(vertices: np.ndarray) -> float:
    """Signed area of a simple polygon via the shoelace formula.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (N, 2) — ordered vertex coordinates.  Counter-clockwise
        winding gives positive area; clockwise gives negative.

    Returns
    -------
    float
        Signed area.  Use ``abs(polygon_area(v))`` if unsigned area is needed.
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    # np.dot on contiguous arrays (Numba warns on the strided column
    # views vertices[:, 0]/[:, 1] otherwise; the copy is cheap at
    # typical polygon sizes and the summation order — hence the
    # floating-point result — is unaffected).
    x = np.ascontiguousarray(vertices[:, 0])
    y = np.ascontiguousarray(vertices[:, 1])
    # Shoelace without np.roll (avoids array allocation per call):
    # 2A = Σ (x_i * y_{i+1} - x_{i+1} * y_i) + wrap-around term
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


@njit(cache=True)
def _clip_edge(
    polygon: np.ndarray,
    axis: int,
    threshold: float,
    keep_ge: bool,
) -> np.ndarray:
    """Clip polygon against a single axis-aligned half-plane.

    Keeps vertices where ``coord[axis] >= threshold`` (if *keep_ge*)
    or ``coord[axis] <= threshold`` (otherwise).
    """
    n = len(polygon)
    # Pre-allocate buffer (each input vertex produces at most 2 output vertices)
    buf = np.empty((2 * n, 2), dtype=np.float64)
    count = 0
    other = 1 - axis  # the other coordinate axis

    vals = polygon[:, axis]

    for i in range(n):
        ni = (i + 1) % n
        c_val = vals[i]
        n_val = vals[ni]

        c_inside = (c_val >= threshold) if keep_ge else (c_val <= threshold)
        n_inside = (n_val >= threshold) if keep_ge else (n_val <= threshold)

        if c_inside:
            buf[count, 0] = polygon[i, 0]
            buf[count, 1] = polygon[i, 1]
            count += 1
            if not n_inside:
                d = n_val - c_val
                if abs(d) < 1e-30:
                    t = 0.5
                else:
                    t = (threshold - c_val) / d
                buf[count, axis] = threshold
                buf[count, other] = polygon[i, other] + t * (polygon[ni, other] - polygon[i, other])
                count += 1
        elif n_inside:
            d = n_val - c_val
            if abs(d) < 1e-30:
                t = 0.5
            else:
                t = (threshold - c_val) / d
            buf[count, axis] = threshold
            buf[count, other] = polygon[i, other] + t * (polygon[ni, other] - polygon[i, other])
            count += 1

    if count < 3:
        return np.empty((0, 2), dtype=np.float64)
    return buf[:count].copy()


@njit(cache=True)
def clip_polygon_to_rect(
    polygon: np.ndarray,
    rect: tuple[float, float, float, float],
) -> np.ndarray:
    """Clip a 2D polygon to an axis-aligned rectangle (Sutherland-Hodgman).

    Parameters
    ----------
    polygon : np.ndarray
        Shape (N, 2) — input polygon vertices.
    rect : tuple
        ``(u_min, v_min, u_max, v_max)`` — clipping rectangle bounds.

    Returns
    -------
    np.ndarray
        Shape (M, 2) — clipped polygon vertices, or empty (0, 2) array
        if the polygon is entirely outside the rectangle.
    """
    if len(polygon) < 3:
        return np.empty((0, 2), dtype=np.float64)

    u_min, v_min, u_max, v_max = rect
    output = polygon.astype(np.float64)

    # Clip against each of the four rectangle edges in turn: left, right,
    # bottom, top. Unrolled (rather than looped over a tuple list) for
    # Numba nopython compatibility.
    output = _clip_edge(output, 0, u_min, True)
    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float64)
    output = _clip_edge(output, 0, u_max, False)
    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float64)
    output = _clip_edge(output, 1, v_min, True)
    if len(output) == 0:
        return np.empty((0, 2), dtype=np.float64)
    output = _clip_edge(output, 1, v_max, False)

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return output


def _intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    axis: int,
    threshold: float,
) -> np.ndarray:
    """Intersection of segment p1→p2 with the line coord[axis] = threshold."""
    d = p2[axis] - p1[axis]
    if abs(d) < 1e-30:
        return 0.5 * (p1 + p2)
    t = (threshold - p1[axis]) / d
    return p1 + t * (p2 - p1)


def point_in_polygon(point: tuple[float, float], vertices: np.ndarray) -> bool:
    """Test whether a 2D point lies inside a simple polygon (ray-casting).

    Parameters
    ----------
    point : tuple
        ``(u, v)`` coordinates of the query point.
    vertices : np.ndarray
        Shape (N, 2) — polygon vertices in order (either winding).

    Returns
    -------
    bool
        True if the point is strictly inside or on the boundary.
    """
    if len(vertices) < 3:
        return False

    px, py = float(point[0]), float(point[1])
    n = len(vertices)
    inside = False

    # Standard ray-casting: cast horizontal ray to +u and count crossings.
    j = n - 1
    for i in range(n):
        yi = vertices[i, 1]
        yj = vertices[j, 1]
        xi = vertices[i, 0]
        xj = vertices[j, 0]

        # Check if the ray crosses edge (i, j)
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def points_in_polygon(
    px: np.ndarray,
    py: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    """Vectorised point-in-polygon test (ray-casting) for a batch of points.

    Parameters
    ----------
    px, py : np.ndarray
        Coordinate arrays of identical shape; each ``(px[..],py[..])`` is
        one query point.
    vertices : np.ndarray
        Shape ``(N, 2)`` — polygon vertices in order (either winding).

    Returns
    -------
    np.ndarray
        Boolean array of the same shape as ``px``/``py``; ``True`` where
        the point lies inside the polygon.

    Notes
    -----
    Two execution strategies with identical results:

    * With Numba: a ``parallel=True`` kernel, ``prange`` over points
      with the edge loop innermost — a single pass, no temporaries.
      Bit-identical to the NumPy path: the crossing decision uses the
      same mul/div/add chain (separate ufunc passes cannot
      FMA-contract, and Numba without fastmath does not either), and
      the output is boolean.  Measured 3–31x faster from 36 points up
      (the NumPy path pays one full-array temporary set per edge), so
      there is no small-batch threshold.  A naive *serial* ``@njit``
      had been measured 2.6x SLOWER than NumPy on the ~1e5-point
      per-plane grids — only the parallel kernel wins.
    * Without Numba: loop over polygon edges, each edge doing a NumPy
      comparison over all points — ``O(N_edges)`` Python overhead,
      ``O(N_points × N_edges)`` arithmetic.
    """
    n = len(vertices)
    if n < 3:
        return np.zeros(px.shape, dtype=bool)

    px = np.asarray(px)
    py = np.asarray(py)

    if HAS_NUMBA:
        out = np.empty(px.size, dtype=np.bool_)
        _points_in_polygon_parallel(
            np.ascontiguousarray(px, dtype=np.float64).ravel(),
            np.ascontiguousarray(py, dtype=np.float64).ravel(),
            np.ascontiguousarray(vertices, dtype=np.float64),
            out,
        )
        return out.reshape(px.shape)

    inside = np.zeros(px.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        yi = vertices[i, 1]
        yj = vertices[j, 1]
        xi = vertices[i, 0]
        xj = vertices[j, 0]

        # Avoid divide-by-zero for horizontal edges (yi == yj): the
        # crossing test (yi > py) != (yj > py) is False there, so the
        # divide-result is masked out anyway. np.errstate suppresses
        # the runtime warning.
        cross = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi
        flip = cross & (px < x_intersect)
        inside ^= flip
        j = i

    return inside


@njit(parallel=True, cache=True)
def _points_in_polygon_parallel(px, py, vertices, out):  # pragma: no cover
    n = vertices.shape[0]
    for p in prange(px.size):
        x = px[p]
        y = py[p]
        inside = False
        j = n - 1
        for i in range(n):
            yi = vertices[i, 1]
            yj = vertices[j, 1]
            if (yi > y) != (yj > y):
                xi = vertices[i, 0]
                xj = vertices[j, 0]
                if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
            j = i
        out[p] = inside


def line_polygon_intersection_length(
    polygon: np.ndarray,
    v_coord: float,
    u_min: float,
    u_max: float,
) -> float:
    """Total length of a horizontal line segment that lies inside a polygon.

    Computes the intersection of the polygon with the horizontal line
    ``v = v_coord``, restricted to ``u in [u_min, u_max]``.

    Parameters
    ----------
    polygon : np.ndarray
        Shape ``(N, 2)`` — polygon vertices in ``(u, v)`` coordinates.
    v_coord : float
        The v-coordinate of the horizontal scan line.
    u_min, u_max : float
        The u-interval of the line segment.

    Returns
    -------
    float
        Total length inside the polygon, in the range ``[0, u_max - u_min]``.
    """
    if len(polygon) < 3 or u_max <= u_min:
        return 0.0

    n = len(polygon)
    crossings: list[float] = []

    for i in range(n):
        ni = (i + 1) % n
        v0 = polygon[i, 1]
        v1 = polygon[ni, 1]

        # Skip horizontal edges and edges that don't straddle v_coord.
        # Use half-open interval [min, max) to avoid double-counting at vertices.
        if v0 == v1:
            continue
        if v0 < v1:
            if v_coord < v0 or v_coord >= v1:
                continue
        else:
            if v_coord < v1 or v_coord >= v0:
                continue

        # Linear interpolation for the u-coordinate at the crossing.
        t = (v_coord - v0) / (v1 - v0)
        u_cross = polygon[i, 0] + t * (polygon[ni, 0] - polygon[i, 0])
        crossings.append(u_cross)

    if len(crossings) < 2:
        return 0.0

    crossings.sort()

    # Pair crossings: inside between crossing[0..1], outside [1..2], etc.
    total = 0.0
    for k in range(0, len(crossings) - 1, 2):
        seg_lo = crossings[k]
        seg_hi = crossings[k + 1]
        # Clip to [u_min, u_max]
        lo = max(seg_lo, u_min)
        hi = min(seg_hi, u_max)
        if hi > lo:
            total += hi - lo

    return total


def pack_annotated_sections(per_shape):
    """Pack per-shape annotated polygon lists into flat arrays for the
    batched face kernels.
    Parameters
    ----------
    per_shape : list
        One entry per shape (priority order): a list of annotated
        polygons ``(vertices, (u_min, v_min, u_max, v_max), signed_area)``
        as built by ``compute_face_material_areas``.
    Returns
    -------
    tuple
        ``(verts, poly_off, poly_bbox, poly_signed, shape_off)`` —
        concatenated ``(V, 2)`` vertices, per-polygon vertex offsets
        ``(P+1,)``, bboxes ``(P, 4)``, signed areas ``(P,)``, and
        per-shape polygon offsets ``(S+1,)``.  Polygon order within a
        shape and shape order are preserved, so the kernels below visit
        them exactly as the Python loop does.
    """
    n_shapes = len(per_shape)
    shape_off = np.zeros(n_shapes + 1, dtype=np.int64)
    polys = []
    for si, annotated in enumerate(per_shape):
        shape_off[si + 1] = shape_off[si] + len(annotated)
        polys.extend(annotated)
    n_polys = len(polys)
    poly_off = np.zeros(n_polys + 1, dtype=np.int64)
    poly_bbox = np.empty((n_polys, 4), dtype=np.float64)
    poly_signed = np.empty(n_polys, dtype=np.float64)
    for p, (poly, bb, signed_area) in enumerate(polys):
        poly_off[p + 1] = poly_off[p] + len(poly)
        poly_bbox[p, 0] = bb[0]
        poly_bbox[p, 1] = bb[1]
        poly_bbox[p, 2] = bb[2]
        poly_bbox[p, 3] = bb[3]
        poly_signed[p] = signed_area
    if n_polys:
        verts = np.ascontiguousarray(
            np.concatenate([np.asarray(p[0], dtype=np.float64) for p in polys])
        )
    else:
        verts = np.empty((0, 2), dtype=np.float64)
    return verts, poly_off, poly_bbox, poly_signed, shape_off


@njit(cache=True)
def _shape_area_on_rect(
    verts, poly_off, poly_bbox, poly_signed, p_lo, p_hi, u_min, v_min, u_max, v_max
):  # pragma: no cover
    """Signed-sum area of one shape's section polygons on a face rect —
    the bbox-shortcut / Sutherland-Hodgman block of the Python loop."""
    shape_area = 0.0
    for p in range(p_lo, p_hi):
        pu_min = poly_bbox[p, 0]
        pv_min = poly_bbox[p, 1]
        pu_max = poly_bbox[p, 2]
        pv_max = poly_bbox[p, 3]
        if pu_max < u_min or pu_min > u_max or pv_max < v_min or pv_min > v_max:
            continue  # fully outside
        if pu_min >= u_min and pu_max <= u_max and pv_min >= v_min and pv_max <= v_max:
            shape_area += poly_signed[p]  # fully inside
            continue
        clipped = clip_polygon_to_rect(
            verts[poly_off[p] : poly_off[p + 1]],
            (u_min, v_min, u_max, v_max),
        )
        if len(clipped) >= 3:
            shape_area += polygon_area(clipped)
    return shape_area


@njit(parallel=True, cache=True)
def face_property_kernel(
    rects,
    verts,
    poly_off,
    poly_bbox,
    poly_signed,
    shape_off,
    shape_val,
    shape_is_pec,
    is_mu,
    bg_exists,
    bg_is_pec,
    bg_val,
    prop_is_sigma,
    result,
    pec_area,
    processed,
):  # pragma: no cover
    """Batched mirror of the per-face accounting loop in
    ``compute_face_material_areas`` (step 3): reverse-priority area
    budget, PEC bookkeeping, background fill.  Same control flow and
    float-op order per face — bit-identical results; ``prange`` over
    faces is safe because faces never share accumulators.
    """
    n_shapes = shape_off.size - 1
    for f in prange(rects.shape[0]):
        u_min = rects[f, 0]
        v_min = rects[f, 1]
        u_max = rects[f, 2]
        v_max = rects[f, 3]
        total_area = (u_max - u_min) * (v_max - v_min)
        if total_area <= 0:
            processed[f] = False
            continue
        processed[f] = True
        remaining = total_area
        weighted_sum = 0.0
        pec = 0.0
        for si in range(n_shapes - 1, -1, -1):
            if remaining <= 1e-30:
                break
            p_lo = shape_off[si]
            p_hi = shape_off[si + 1]
            if p_lo == p_hi:
                continue
            shape_area = abs(
                _shape_area_on_rect(
                    verts,
                    poly_off,
                    poly_bbox,
                    poly_signed,
                    p_lo,
                    p_hi,
                    u_min,
                    v_min,
                    u_max,
                    v_max,
                )
            )
            effective = min(shape_area, remaining)
            if effective < 1e-30:
                continue
            if is_mu:
                if shape_is_pec[si]:
                    weighted_sum += 1.0 * effective
                    pec += effective
                else:
                    weighted_sum += shape_val[si] * effective
            else:
                if not shape_is_pec[si]:
                    weighted_sum += shape_val[si] * effective
                else:
                    pec += effective
            remaining -= effective
        if remaining > 1e-30:
            if bg_exists:
                if is_mu and bg_is_pec:
                    weighted_sum += 1.0 * remaining
                    pec += remaining
                elif (not is_mu) and bg_is_pec:
                    pec += remaining
                else:
                    weighted_sum += bg_val * remaining
            else:
                if not prop_is_sigma:
                    weighted_sum += 1.0 * remaining
        result[f] = weighted_sum / total_area
        pec_area[f] = pec


@njit(parallel=True, cache=True)
def face_pec_area_kernel(
    rects, verts, poly_off, poly_bbox, poly_signed, shape_off, shape_is_pec, bg_is_pec, out
):  # pragma: no cover
    """Batched mirror of ``_pec_area_from`` in
    ``compute_face_material_areas``: PEC area per face from a section
    set with the same priority-order remaining-budget bookkeeping.
    Faces with non-positive area are left untouched (the caller never
    reads them).
    """
    n_shapes = shape_off.size - 1
    for f in prange(rects.shape[0]):
        u_min = rects[f, 0]
        v_min = rects[f, 1]
        u_max = rects[f, 2]
        v_max = rects[f, 3]
        total_area = (u_max - u_min) * (v_max - v_min)
        if total_area <= 0:
            continue
        remaining = total_area
        pec_a = 0.0
        for si in range(n_shapes - 1, -1, -1):
            if remaining <= 1e-30:
                break
            p_lo = shape_off[si]
            p_hi = shape_off[si + 1]
            if p_lo == p_hi:
                continue
            clip_a = abs(
                _shape_area_on_rect(
                    verts,
                    poly_off,
                    poly_bbox,
                    poly_signed,
                    p_lo,
                    p_hi,
                    u_min,
                    v_min,
                    u_max,
                    v_max,
                )
            )
            effective = min(clip_a, remaining)
            if effective < 1e-30:
                continue
            if shape_is_pec[si]:
                pec_a += effective
            remaining -= effective
        if remaining > 1e-30 and bg_is_pec:
            pec_a += remaining
        out[f] = pec_a


@njit(parallel=True, cache=True)
def face_shape_area_kernel(
    rects, verts, poly_off, poly_bbox, poly_signed, shape_off, shape_eff, bg_rem
):  # pragma: no cover
    """Batched mirror of the reverse-priority area budget in
    ``compute_face_material_areas``: EFFECTIVE (post-priority) area per
    (face, shape) plus the uncovered background remainder per face
    (WP-C1, DD-093).  Same control flow and float-op order as
    ``face_property_kernel``, so ``Σ_shapes eff + rem = total_area``
    holds with the identical budget cascade; only the outputs differ
    (raw areas instead of the property-weighted sum).  Faces with
    non-positive area are left untouched (the caller never reads
    them).
    """
    n_shapes = shape_off.size - 1
    for f in prange(rects.shape[0]):
        u_min = rects[f, 0]
        v_min = rects[f, 1]
        u_max = rects[f, 2]
        v_max = rects[f, 3]
        total_area = (u_max - u_min) * (v_max - v_min)
        if total_area <= 0:
            continue
        remaining = total_area
        for si in range(n_shapes - 1, -1, -1):
            if remaining <= 1e-30:
                break
            p_lo = shape_off[si]
            p_hi = shape_off[si + 1]
            if p_lo == p_hi:
                continue
            clip_a = abs(
                _shape_area_on_rect(
                    verts,
                    poly_off,
                    poly_bbox,
                    poly_signed,
                    p_lo,
                    p_hi,
                    u_min,
                    v_min,
                    u_max,
                    v_max,
                )
            )
            effective = min(clip_a, remaining)
            if effective < 1e-30:
                continue
            shape_eff[f, si] = effective
            remaining -= effective
        if remaining > 1e-30:
            bg_rem[f] = remaining
