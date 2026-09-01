"""Compiled batch of the planar section engine: every plane of an axis in one pass.

:meth:`_PlanarSectionEngine.section` answers one plane from the
engine's face/edge arrays with some fifty small NumPy calls — on a row
of posts, 15 487 planes at 245 µs each with one or two faces of
interest per plane, the per-call overhead and not the arithmetic.
:func:`section_planes` takes the sorted plane positions of one axis
and runs the same pipeline per plane in compiled loops: candidate
faces and edges from ``searchsorted`` windows on the sorted positions
(CSR, in face / edge index order), the admission screen, one crossing
per transversally crossed straight edge and two analytic ones per
circular edge, planar faces paired along their trace direction by
parity, cylindrical faces along their generatrices or section conic,
and the segments stitched into closed chains.  Every arithmetic step
is the scalar form of the array expression it replaces, in the same
order, so a plane comes out with the same vertices as the per-plane
path (the only known exception is the cylinder frame's ``rel @ y``,
which BLAS may fuse).  :func:`orient_annotate_packed` is the batch
form of ``orient_nested_contours`` plus the area pass's annotation.

The results are packed: ``status`` per plane (1 answered, 0
delegate), ``poly_ptr`` (polygons per plane), ``vert_ptr`` (vertices
per polygon) and ``verts`` [m].

``njit`` is a no-op decorator when Numba is unavailable; the kernels
are plain loops and give the same results either way.
"""

from __future__ import annotations

import math

import numpy as np

from magnelio.geo._polygon_clip import polygon_area

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # noqa: ANN002, ANN003
        """No-op stand-in for ``numba.njit`` when Numba is not installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


TWO_PI = 2.0 * math.pi

DELEGATE = 0
ANSWERED = 1

_EMPTY_VERTS = np.zeros((0, 2), dtype=np.float64)
_EMPTY_PTR = np.zeros(1, dtype=np.int64)


@njit(cache=True)
def atan2_elementwise(y, x):
    """``math.atan2`` element by element — the NumPy ufunc rounds
    differently in the last ulp, which would put the per-plane path
    off the compiled batch."""
    out = np.empty(y.size, dtype=np.float64)
    for i in range(y.size):
        out[i] = math.atan2(y[i], x[i])
    return out


@njit(cache=True)
def acos_elementwise(q):
    """``math.acos`` element by element (see :func:`atan2_elementwise`)."""
    out = np.empty(q.size, dtype=np.float64)
    for i in range(q.size):
        out[i] = math.acos(q[i])
    return out


@njit(cache=True)
def hypot_elementwise(a, b):
    """``math.hypot`` element by element (see :func:`atan2_elementwise`)."""
    out = np.empty(a.size, dtype=np.float64)
    for i in range(a.size):
        out[i] = math.hypot(a[i], b[i])
    return out


@njit(cache=True)
def _grow_rows(buf, needed):
    """*buf* ``(n, 2)`` with room for *needed* rows (doubling)."""
    if needed <= buf.shape[0]:
        return buf
    new = np.empty((max(needed, 2 * buf.shape[0]), buf.shape[1]), dtype=buf.dtype)
    new[: buf.shape[0]] = buf
    return new


@njit(cache=True)
def _grow_flat(buf, needed):
    """*buf* ``(n,)`` with room for *needed* entries (doubling)."""
    if needed <= buf.shape[0]:
        return buf
    new = np.empty(max(needed, 2 * buf.shape[0]), dtype=buf.dtype)
    new[: buf.shape[0]] = buf
    return new


@njit(cache=True)
def _windows(positions, lo, hi, tol):
    """Index ranges into sorted *positions* of the intervals ``[lo, hi]``
    widened past *tol* — a superset of the engine's ``pos + tol >= lo
    and pos - tol <= hi`` candidates, which every consumer re-tests
    exactly (the widened bounds round differently from the
    comparison)."""
    n = lo.size
    i0 = np.empty(n, dtype=np.int64)
    i1 = np.empty(n, dtype=np.int64)
    for k in range(n):
        a = lo[k] - (2.0 * tol + 1e-12 * (1.0 + abs(lo[k])))
        b = hi[k] + (2.0 * tol + 1e-12 * (1.0 + abs(hi[k])))
        i0[k] = np.searchsorted(positions, a, side="left")
        i1[k] = np.searchsorted(positions, b, side="right")
    return i0, i1


@njit(cache=True)
def _csr(n_planes, i0, i1):
    """Per-plane item lists (CSR) from per-item plane ranges; the items
    of a plane come out in ascending item index."""
    counts = np.zeros(n_planes + 1, dtype=np.int64)
    for k in range(i0.size):
        for p in range(i0[k], i1[k]):
            counts[p + 1] += 1
    ptr = np.cumsum(counts)
    fill = ptr[:-1].copy()
    items = np.empty(ptr[-1], dtype=np.int64)
    for k in range(i0.size):
        for p in range(i0[k], i1[k]):
            items[fill[p]] = k
            fill[p] += 1
    return ptr, items


@njit(cache=True)
def _direction(axis, uu, sense, x, y):
    """Segment direction n_plane × n_outward at azimuth *uu* of a
    cylindrical face — the cross product with the plane's unit axis
    vector written out."""
    cu = math.cos(uu)
    su = math.sin(uu)
    m0 = sense * (cu * x[0] + su * y[0])
    m1 = sense * (cu * x[1] + su * y[1])
    m2 = sense * (cu * x[2] + su * y[2])
    out = np.empty(3, dtype=np.float64)
    if axis == 0:
        out[0] = 0.0
        out[1] = -m2
        out[2] = m1
    elif axis == 1:
        out[0] = m2
        out[1] = 0.0
        out[2] = -m0
    else:
        out[0] = -m1
        out[1] = m0
        out[2] = 0.0
    return out


@njit(cache=True)
def _cylinder_face_pairs(
    f,
    axis,
    u_idx,
    v_idx,
    pos,
    tol,
    deflection,
    pts,
    crossings,
    cyl_c,
    cyl_a,
    cyl_x,
    cyl_y,
    cyl_r,
    cyl_sense,
    cyl_uv,
    first,
    second,
    n_pairs,
    int_start,
    int_len,
    int_pts,
    n_int,
):
    """Segments of one cylindrical face between its boundary crossings
    (the compiled form of ``_PlanarSectionEngine._cylinder_face_pairs``,
    step for step).  Appends to *first*/*second* from *n_pairs*, stores
    arc interiors in *int_pts* (per start crossing: ``int_start`` /
    ``int_len``).  Returns ``(ok, n_pairs, int_pts, n_int)``."""
    c = cyl_c[f]
    a = cyl_a[f]
    x = cyl_x[f]
    y = cyl_y[f]
    r = cyl_r[f]
    sense = cyl_sense[f]
    umin = cyl_uv[f, 0]
    umax = cyl_uv[f, 1]
    vmin = cyl_uv[f, 2]
    vmax = cyl_uv[f, 3]
    full = umax - umin >= TWO_PI - 1e-9
    k = crossings.size
    u = np.empty(k, dtype=np.float64)
    v = np.empty(k, dtype=np.float64)
    for i in range(k):
        q = crossings[i]
        r0 = pts[q, 0] - c[0]
        r1 = pts[q, 1] - c[1]
        r2 = pts[q, 2] - c[2]
        ry = r0 * y[0] + r1 * y[1] + r2 * y[2]
        rx = r0 * x[0] + r1 * x[1] + r2 * x[2]
        du = (math.atan2(ry, rx) - umin) % TWO_PI
        if du >= TWO_PI - 1e-12:
            du = 0.0
        u[i] = umin + du
        v[i] = r0 * a[0] + r1 * a[1] + r2 * a[2]
    a_n = x[axis]
    b_n = y[axis]
    c_n = a[axis]
    p = pos - c[axis]

    if abs(c_n) <= 1e-9:
        rho = r * math.hypot(a_n, b_n)
        q = p / rho
        if abs(q) >= 1.0:
            return False, n_pairs, int_pts, n_int
        phi = math.atan2(b_n, a_n)
        half = math.acos(q)
        for sgn in (1.0, -1.0):
            u_star = phi + sgn * half
            u_star = umin + np.fmod(u_star - umin, TWO_PI)
            if u_star < umin:
                u_star += TWO_PI
            if u_star > umax + 1e-9:
                continue
            g0 = -1
            g1 = -1
            n_group = 0
            for i in range(k):
                gap = abs((u[i] - u_star + math.pi) % TWO_PI - math.pi)
                if gap <= 1e-6:
                    n_group += 1
                    if g0 < 0:
                        g0 = i
                    else:
                        g1 = i
            if n_group != 2:
                return False, n_pairs, int_pts, n_int
            if v[g1] < v[g0]:
                lo, hi = g1, g0
            else:
                lo, hi = g0, g1
            if v[hi] - v[lo] <= tol:
                return False, n_pairs, int_pts, n_int
            d = _direction(axis, u_star, sense, x, y)
            forward = d[0] * a[0] + d[1] * a[1] + d[2] * a[2] > 0.0
            if forward:
                first[n_pairs] = crossings[lo]
                second[n_pairs] = crossings[hi]
            else:
                first[n_pairs] = crossings[hi]
                second[n_pairs] = crossings[lo]
            n_pairs += 1
        return True, n_pairs, int_pts, n_int

    order = np.argsort(u, kind="mergesort")
    if k == 0:
        return True, n_pairs, int_pts, n_int
    u_s = u[order]
    idx_s = crossings[order]
    n_arcs = k - 1
    if full:
        n_arcs += 1
    # Ellipse sagitta r du^2 / (8 |c_n|) at u = +-pi/2 -- exponent 1, the
    # compiled twin of the exact path in _occ_backend.
    du_max = min(
        math.radians(5.0) * abs(c_n),
        math.sqrt(8.0 * deflection * abs(c_n) / r),
    )
    v_margin = max(tol, 1e-9 * (vmax - vmin))
    for arc in range(n_arcs):
        if arc < k - 1:
            i = arc
            j = arc + 1
            wrap = 0.0
        else:
            i = k - 1
            j = 0
            wrap = TWO_PI
        ua = u_s[i]
        ub = u_s[j] + wrap
        if ub - ua <= 1e-12:
            continue
        um = 0.5 * (ua + ub)
        vm = (p - r * (a_n * math.cos(um) + b_n * math.sin(um))) / c_n
        if abs(vm - vmin) <= v_margin or abs(vm - vmax) <= v_margin:
            return False, n_pairs, int_pts, n_int
        if not (vmin < vm < vmax):
            continue
        v_prime = r * (a_n * math.sin(um) - b_n * math.cos(um)) / c_n
        sm = -math.sin(um)
        cm = math.cos(um)
        t0 = r * (sm * x[0] + cm * y[0]) + v_prime * a[0]
        t1 = r * (sm * x[1] + cm * y[1]) + v_prime * a[1]
        t2 = r * (sm * x[2] + cm * y[2]) + v_prime * a[2]
        d = _direction(axis, um, sense, x, y)
        forward = d[0] * t0 + d[1] * t1 + d[2] * t2 > 0.0
        n_seg = max(1, int(math.ceil((ub - ua) / du_max - 1e-9)))
        if n_seg > 100_000:
            return False, n_pairs, int_pts, n_int
        n_inner = n_seg - 1
        int_pts = _grow_rows(int_pts, n_int + n_inner)
        span = ub - ua
        for s in range(n_inner):
            uu = ua + (span * (s + 1)) / n_seg
            cu = math.cos(uu)
            su = math.sin(uu)
            vv = (p - r * (a_n * cu + b_n * su)) / c_n
            q0 = (c[0] + r * (cu * x[0] + su * y[0])) + vv * a[0]
            q1 = (c[1] + r * (cu * x[1] + su * y[1])) + vv * a[1]
            q2 = (c[2] + r * (cu * x[2] + su * y[2])) + vv * a[2]
            if axis == 0:
                q0 = pos
            elif axis == 1:
                q1 = pos
            else:
                q2 = pos
            if forward:
                row = n_int + s
            else:
                row = n_int + n_inner - 1 - s
            if u_idx == 0:
                int_pts[row, 0] = q0
            elif u_idx == 1:
                int_pts[row, 0] = q1
            else:
                int_pts[row, 0] = q2
            if v_idx == 1:
                int_pts[row, 1] = q1
            else:
                int_pts[row, 1] = q2
        if forward:
            start = idx_s[i]
            end = idx_s[j]
        else:
            start = idx_s[j]
            end = idx_s[i]
        first[n_pairs] = start
        second[n_pairs] = end
        n_pairs += 1
        int_start[start] = n_int
        int_len[start] = n_inner
        n_int += n_inner
    return True, n_pairs, int_pts, n_int


@njit(cache=True)
def _section_one(  # noqa: PLR0912, PLR0915 — one plane, step for step as ``section``
    axis,
    u_idx,
    v_idx,
    pos,
    tol,
    deflection,
    scale,
    faces,
    edges,
    v_hit,
    f_lo,
    f_hi,
    f_planar,
    f_n,
    f_d,
    f_ok,
    f_cyl_ok,
    tangent,
    tangent_ok,
    cyl_c,
    cyl_a,
    cyl_x,
    cyl_y,
    cyl_r,
    cyl_sense,
    cyl_uv,
    e_ok,
    e_any,
    e_p1,
    e_p2,
    e_f,
    e_lo,
    e_hi,
    c_ok,
    c_c,
    c_x,
    c_y,
    c_r,
    c_t,
):
    """One plane: ``(status, verts, ptr)`` — polygons [m] as
    ``verts[ptr[q]:ptr[q + 1]]``."""
    empty_v = np.empty((0, 2), dtype=np.float64)
    empty_p = np.zeros(1, dtype=np.int64)

    # --- admission screen (``_screen``) -----------------------------------
    n_cf = 0
    for idx in range(faces.size):
        f = faces[idx]
        if f_lo[f, axis] <= pos + tol and f_hi[f, axis] >= pos - tol:
            faces[n_cf] = f
            n_cf += 1
    for idx in range(n_cf):
        if not f_ok[faces[idx]]:
            return DELEGATE, empty_v, empty_p
    for idx in range(n_cf):
        f = faces[idx]
        if f_planar[f] and abs(f_n[f, axis]) >= 1.0 - 1e-9:
            if abs(pos * f_n[f, axis] - f_d[f]) <= tol:
                return DELEGATE, empty_v, empty_p
    for idx in range(n_cf):
        f = faces[idx]
        if f_cyl_ok[f] and abs(cyl_a[f, axis]) <= 1e-9:
            rho = cyl_r[f] * math.hypot(cyl_x[f, axis], cyl_y[f, axis])
            if abs(abs(pos - cyl_c[f, axis]) - rho) <= tol:
                return DELEGATE, empty_v, empty_p
    n_ce = 0
    for idx in range(edges.size):
        e = edges[idx]
        if e_lo[e, axis] <= pos + tol and e_hi[e, axis] >= pos - tol:
            edges[n_ce] = e
            n_ce += 1
    for idx in range(n_ce):
        if not e_any[edges[idx]]:
            return DELEGATE, empty_v, empty_p
    for idx in range(n_ce):
        e = edges[idx]
        if c_ok[e]:
            rho = c_r[e] * math.hypot(c_x[e, axis], c_y[e, axis])
            if abs(abs(pos - c_c[e, axis]) - rho) <= tol:
                return DELEGATE, empty_v, empty_p
    if v_hit:
        return DELEGATE, empty_v, empty_p

    # --- crossings: straight edges (edge order), then circles (+ then −)
    m_cap = 3 * n_ce + 1
    cr_edge = np.empty(m_cap, dtype=np.int64)
    pts = np.empty((m_cap, 3), dtype=np.float64)
    m = 0
    for idx in range(n_ce):
        e = edges[idx]
        if not e_ok[e]:
            continue
        d1 = e_p1[e, axis] - pos
        d2 = e_p2[e, axis] - pos
        if d1 * d2 < 0.0:
            t = d1 / (d1 - d2)
            cr_edge[m] = e
            for kk in range(3):
                pts[m, kk] = e_p1[e, kk] + t * (e_p2[e, kk] - e_p1[e, kk])
            m += 1
    for sgn in (1.0, -1.0):
        for idx in range(n_ce):
            e = edges[idx]
            if not c_ok[e]:
                continue
            a_coef = c_x[e, axis]
            b_coef = c_y[e, axis]
            rho = c_r[e] * math.hypot(a_coef, b_coef)
            if not rho > 0.0:
                continue
            q = (pos - c_c[e, axis]) / rho
            if not abs(q) < 1.0:
                continue
            phi = math.atan2(b_coef, a_coef)
            half = math.acos(q)
            tt = c_t[e, 0] + (phi + sgn * half - c_t[e, 0]) % TWO_PI
            if tt > c_t[e, 1] + 1e-12:
                continue
            ct = math.cos(tt)
            st = math.sin(tt)
            cr_edge[m] = e
            for kk in range(3):
                pts[m, kk] = c_c[e, kk] + c_r[e] * (ct * c_x[e, kk] + st * c_y[e, kk])
            m += 1
    if m == 0:
        return ANSWERED, empty_v, empty_p
    for q in range(m):
        pts[q, axis] = pos
    uv = np.empty((m, 2), dtype=np.float64)
    for q in range(m):
        uv[q, 0] = pts[q, u_idx]
        uv[q, 1] = pts[q, v_idx]

    # --- the two faces of every crossing ---------------------------------
    n2 = 2 * m
    rep = np.empty(n2, dtype=np.int64)
    face = np.empty(n2, dtype=np.int64)
    for q in range(m):
        rep[q] = q
        rep[m + q] = q
        face[q] = e_f[cr_edge[q], 0]
        face[m + q] = e_f[cr_edge[q], 1]
    n_pl = 0
    for q in range(n2):
        if f_planar[face[q]]:
            n_pl += 1
    n_cy = n2 - n_pl
    pl = np.empty(n_pl, dtype=np.int64)
    cy = np.empty(n_cy, dtype=np.int64)
    i_pl = 0
    i_cy = 0
    for q in range(n2):
        if f_planar[face[q]]:
            pl[i_pl] = q
            i_pl += 1
        else:
            cy[i_cy] = q
            i_cy += 1

    first = np.empty(4 * m + 8, dtype=np.int64)
    second = np.empty(4 * m + 8, dtype=np.int64)
    n_pairs = 0

    # --- planar faces: crossings sorted along the trace, paired by parity
    if n_pl:
        s = np.empty(n_pl, dtype=np.float64)
        pf = np.empty(n_pl, dtype=np.int64)
        for i in range(n_pl):
            q = pl[i]
            fq = face[q]
            if not tangent_ok[fq]:
                return DELEGATE, empty_v, empty_p
            s[i] = uv[rep[q], 0] * tangent[fq, u_idx] + uv[rep[q], 1] * tangent[fq, v_idx]
            pf[i] = fq
        o1 = np.argsort(s, kind="mergesort")
        o2 = np.argsort(pf[o1], kind="mergesort")
        order = o1[o2]
        if n_pl % 2:
            return DELEGATE, empty_v, empty_p
        i = 0
        while i < n_pl:
            fq = pf[order[i]]
            j = i
            while j < n_pl and pf[order[j]] == fq:
                j += 1
            if (j - i) % 2:
                return DELEGATE, empty_v, empty_p
            for kk in range(i, j - 1):
                if s[order[kk + 1]] - s[order[kk]] <= tol:
                    return DELEGATE, empty_v, empty_p
            i = j
        for i in range(0, n_pl, 2):
            first[n_pairs] = rep[pl[order[i]]]
            second[n_pairs] = rep[pl[order[i + 1]]]
            n_pairs += 1

    # --- cylindrical faces, face by face ----------------------------------
    int_start = np.full(m, -1, dtype=np.int64)
    int_len = np.zeros(m, dtype=np.int64)
    int_pts = np.empty((0, 2), dtype=np.float64)
    n_int = 0
    if n_cy:
        cf_ = np.empty(n_cy, dtype=np.int64)
        cr_ = np.empty(n_cy, dtype=np.int64)
        for i in range(n_cy):
            q = cy[i]
            if not f_cyl_ok[face[q]]:
                return DELEGATE, empty_v, empty_p
            cf_[i] = face[q]
            cr_[i] = rep[q]
        o1 = np.argsort(cr_, kind="mergesort")
        o2 = np.argsort(cf_[o1], kind="mergesort")
        order = o1[o2]
        # A seam crossing lists its face twice — once per side.
        keep_f = np.empty(n_cy, dtype=np.int64)
        keep_r = np.empty(n_cy, dtype=np.int64)
        n_keep = 0
        for i in range(n_cy):
            fq = cf_[order[i]]
            rq = cr_[order[i]]
            if n_keep and keep_f[n_keep - 1] == fq and keep_r[n_keep - 1] == rq:
                continue
            keep_f[n_keep] = fq
            keep_r[n_keep] = rq
            n_keep += 1
        int_pts = np.empty((64, 2), dtype=np.float64)
        i = 0
        while i < n_keep:
            fq = keep_f[i]
            j = i
            while j < n_keep and keep_f[j] == fq:
                j += 1
            ok, n_pairs, int_pts, n_int = _cylinder_face_pairs(
                fq,
                axis,
                u_idx,
                v_idx,
                pos,
                tol,
                deflection,
                pts,
                keep_r[i:j].copy(),
                cyl_c,
                cyl_a,
                cyl_x,
                cyl_y,
                cyl_r,
                cyl_sense,
                cyl_uv,
                first,
                second,
                n_pairs,
                int_start,
                int_len,
                int_pts,
                n_int,
            )
            if not ok:
                return DELEGATE, empty_v, empty_p
            i = j

    # --- stitch: every crossing starts one segment and ends one ----------
    succ = np.full(m, -1, dtype=np.int64)
    for i in range(n_pairs):
        if succ[first[i]] != -1:
            return DELEGATE, empty_v, empty_p
        succ[first[i]] = second[i]
    indeg = np.zeros(m, dtype=np.int64)
    for q in range(m):
        if succ[q] == -1:
            return DELEGATE, empty_v, empty_p
        indeg[succ[q]] += 1
    for q in range(m):
        if indeg[q] != 1:
            return DELEGATE, empty_v, empty_p

    # --- chains to polygons (``_chains_to_polygons``) ----------------------
    visited = np.zeros(m, dtype=np.bool_)
    out = np.empty((m + n_int + 8, 2), dtype=np.float64)
    ptr = np.empty(m + 2, dtype=np.int64)
    ptr[0] = 0
    n_out = 0
    n_poly = 0
    for start in range(m):
        if visited[start]:
            continue
        base = n_out
        k = start
        while not visited[k]:
            visited[k] = True
            # the crossing itself, then the interior of its segment
            x0 = uv[k, 0]
            y0 = uv[k, 1]
            if (
                n_out == base
                or abs(x0 - out[n_out - 1, 0]) > 1e-12
                or abs(y0 - out[n_out - 1, 1]) > 1e-12
            ):
                out[n_out, 0] = x0
                out[n_out, 1] = y0
                n_out += 1
            if int_start[k] >= 0:
                for s_ in range(int_len[k]):
                    x0 = int_pts[int_start[k] + s_, 0]
                    y0 = int_pts[int_start[k] + s_, 1]
                    if abs(x0 - out[n_out - 1, 0]) > 1e-12 or abs(y0 - out[n_out - 1, 1]) > 1e-12:
                        out[n_out, 0] = x0
                        out[n_out, 1] = y0
                        n_out += 1
            k = succ[k]
        if k != start:
            return DELEGATE, empty_v, empty_p
        n_pts = n_out - base
        if n_pts >= 3:
            if (
                abs(out[base, 0] - out[n_out - 1, 0]) < 1e-10
                and abs(out[base, 1] - out[n_out - 1, 1]) < 1e-10
            ):
                n_out -= 1
                n_pts -= 1
        if n_pts >= 3:
            n_poly += 1
            ptr[n_poly] = n_out
        else:
            n_out = base
    verts = out[:n_out].copy()
    # Back to meters (lossless power-of-two divide).
    for q in range(n_out):
        verts[q, 0] = verts[q, 0] / scale
        verts[q, 1] = verts[q, 1] / scale
    return ANSWERED, verts, ptr[: n_poly + 1].copy()


@njit(cache=True)
def section_planes(  # noqa: PLR0913
    axis,
    positions,
    tol,
    deflection,
    scale,
    f_lo,
    f_hi,
    f_planar,
    f_n,
    f_d,
    f_ok,
    f_cyl_ok,
    tangent,
    tangent_ok,
    cyl_c,
    cyl_a,
    cyl_x,
    cyl_y,
    cyl_r,
    cyl_sense,
    cyl_uv,
    e_ok,
    e_any,
    e_p1,
    e_p2,
    e_f,
    e_lo,
    e_hi,
    c_ok,
    c_c,
    c_x,
    c_y,
    c_r,
    c_t,
    v_sorted,
):
    """Sections of every plane of one axis (*positions* sorted, scaled
    units): ``(status, poly_ptr, vert_ptr, verts)`` — polygon *q* of
    plane *p* is ``verts[vert_ptr[q]:vert_ptr[q + 1]]`` [m] for *q* in
    ``poly_ptr[p]:poly_ptr[p + 1]``; ``status`` 0 marks a plane the
    engine delegates."""
    n_p = positions.size
    if axis == 0:
        u_idx, v_idx = 1, 2
    elif axis == 1:
        u_idx, v_idx = 0, 2
    else:
        u_idx, v_idx = 0, 1
    fi0, fi1 = _windows(positions, f_lo[:, axis].copy(), f_hi[:, axis].copy(), tol)
    f_ptr, f_items = _csr(n_p, fi0, fi1)
    ei0, ei1 = _windows(positions, e_lo[:, axis].copy(), e_hi[:, axis].copy(), tol)
    e_ptr, e_items = _csr(n_p, ei0, ei1)
    status = np.zeros(n_p, dtype=np.int8)
    poly_ptr = np.zeros(n_p + 1, dtype=np.int64)
    vert_ptr = np.zeros(2 * n_p + 16, dtype=np.int64)
    verts = np.empty((max(64, 16 * n_p), 2), dtype=np.float64)
    n_poly = 0
    n_vert = 0
    for pi in range(n_p):
        pos = positions[pi]
        v_hit = False
        margin = 2.0 * tol + 1e-12 * (1.0 + abs(pos))
        j0 = np.searchsorted(v_sorted, pos - margin, side="left")
        j1 = np.searchsorted(v_sorted, pos + margin, side="right")
        for j in range(j0, j1):
            if abs(v_sorted[j] - pos) <= tol:
                v_hit = True
                break
        st, pv, pp = _section_one(
            axis,
            u_idx,
            v_idx,
            pos,
            tol,
            deflection,
            scale,
            f_items[f_ptr[pi] : f_ptr[pi + 1]].copy(),
            e_items[e_ptr[pi] : e_ptr[pi + 1]].copy(),
            v_hit,
            f_lo,
            f_hi,
            f_planar,
            f_n,
            f_d,
            f_ok,
            f_cyl_ok,
            tangent,
            tangent_ok,
            cyl_c,
            cyl_a,
            cyl_x,
            cyl_y,
            cyl_r,
            cyl_sense,
            cyl_uv,
            e_ok,
            e_any,
            e_p1,
            e_p2,
            e_f,
            e_lo,
            e_hi,
            c_ok,
            c_c,
            c_x,
            c_y,
            c_r,
            c_t,
        )
        status[pi] = st
        if st == ANSWERED:
            k = pp.size - 1
            vert_ptr = _grow_flat(vert_ptr, n_poly + k + 1)
            verts = _grow_rows(verts, n_vert + pv.shape[0])
            for q in range(k):
                vert_ptr[n_poly + q + 1] = n_vert + pp[q + 1]
            verts[n_vert : n_vert + pv.shape[0]] = pv
            n_poly += k
            n_vert += pv.shape[0]
        poly_ptr[pi + 1] = n_poly
    return status, poly_ptr, vert_ptr[: n_poly + 1].copy(), verts[:n_vert].copy()


@njit(cache=True)
def _inside_votes(px, py, poly):
    """How many of the points ``(px, py)`` lie inside *poly* — the
    ray-casting rule of ``points_in_polygon``."""
    n = poly.shape[0]
    votes = 0
    for p in range(px.size):
        x = px[p]
        y = py[p]
        inside = False
        j = n - 1
        for i in range(n):
            yi = poly[i, 1]
            yj = poly[j, 1]
            if (yi > y) != (yj > y):
                xi = poly[i, 0]
                xj = poly[j, 0]
                if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
            j = i
        if inside:
            votes += 1
    return votes


@njit(cache=True)
def orient_annotate_packed(poly_ptr, vert_ptr, verts):
    """``orient_nested_contours`` plus the area pass's annotation for
    every plane of a packed batch: ``(verts, bbox, area)`` — the
    contours rewound by nesting parity (holes against their outer
    boundaries; bbox-coincident pairs untouched), per contour its
    ``(u_min, v_min, u_max, v_max)`` and signed area."""
    n_poly = vert_ptr.size - 1
    out = np.empty_like(verts)
    bbox = np.empty((n_poly, 4), dtype=np.float64)
    area = np.empty(n_poly, dtype=np.float64)
    for q in range(n_poly):
        a = vert_ptr[q]
        b = vert_ptr[q + 1]
        u0 = verts[a, 0]
        u1 = u0
        v0 = verts[a, 1]
        v1 = v0
        for i in range(a + 1, b):
            u0 = min(u0, verts[i, 0])
            u1 = max(u1, verts[i, 0])
            v0 = min(v0, verts[i, 1])
            v1 = max(v1, verts[i, 1])
        bbox[q, 0] = u0
        bbox[q, 1] = v0
        bbox[q, 2] = u1
        bbox[q, 3] = v1
    for pi in range(poly_ptr.size - 1):
        qa = poly_ptr[pi]
        qb = poly_ptr[pi + 1]
        n = qb - qa
        if n == 0:
            continue
        umin = bbox[qa, 0]
        umax = bbox[qa, 2]
        vmin = bbox[qa, 1]
        vmax = bbox[qa, 3]
        for q in range(qa + 1, qb):
            umin = min(umin, bbox[q, 0])
            umax = max(umax, bbox[q, 2])
            vmin = min(vmin, bbox[q, 1])
            vmax = max(vmax, bbox[q, 3])
        extent = max(umax - umin, vmax - vmin, 1e-300)
        tol = 1e-9 * extent
        for qi in range(qa, qb):
            depth = 0
            coincident = False
            for qj in range(qa, qb):
                if qj == qi:
                    continue
                same = True
                for kk in range(4):
                    if abs(bbox[qi, kk] - bbox[qj, kk]) > tol:
                        same = False
                        break
                if same:
                    coincident = True
                    continue
                nested = (
                    bbox[qi, 0] >= bbox[qj, 0] - tol
                    and bbox[qi, 1] >= bbox[qj, 1] - tol
                    and bbox[qi, 2] <= bbox[qj, 2] + tol
                    and bbox[qi, 3] <= bbox[qj, 3] + tol
                )
                if nested:
                    pi_ = verts[vert_ptr[qi] : vert_ptr[qi + 1]]
                    votes = _inside_votes(
                        np.ascontiguousarray(pi_[:, 0]),
                        np.ascontiguousarray(pi_[:, 1]),
                        verts[vert_ptr[qj] : vert_ptr[qj + 1]],
                    )
                    if 2 * votes > pi_.shape[0]:
                        depth += 1
            a = vert_ptr[qi]
            b = vert_ptr[qi + 1]
            flip = False
            if not coincident:
                want_positive = depth % 2 == 0
                flip = (polygon_area(verts[a:b]) >= 0.0) != want_positive
            if flip:
                for i in range(b - a):
                    out[a + i, 0] = verts[b - 1 - i, 0]
                    out[a + i, 1] = verts[b - 1 - i, 1]
            else:
                out[a:b] = verts[a:b]
            area[qi] = polygon_area(out[a:b])
    return out, bbox, area
