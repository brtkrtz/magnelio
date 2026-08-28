"""Kernels of the edge pass: axis-aligned line candidates, planar and cylindrical hits.

``njit`` is a no-op decorator when Numba is unavailable; the kernels are
plain loops and give the same results either way.
"""

from __future__ import annotations

import math

import numpy as np

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


@njit(cache=True)
def axis_line_candidates(flo, fhi, tol, p0, ax, d):
    """Rows whose tolerance-inflated box an axis-aligned line touches.

    The line runs through ``p0`` along axis ``ax`` with direction
    ``d = ±1``.  Returns ``(idx, w_in, w_out)`` sorted by ``w_in`` —
    the same values the general slab test produces (a division by ±1
    is exact), row order preserved among equal ``w_in``.
    """
    n = flo.shape[0]
    u = 0
    v = 0
    if ax == 0:
        u, v = 1, 2
    elif ax == 1:
        u, v = 0, 2
    else:
        u, v = 0, 1
    pu = p0[u]
    pv = p0[v]
    pa = p0[ax]
    keep = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(n):
        if (
            flo[i, u] - tol <= pu
            and pu <= fhi[i, u] + tol
            and flo[i, v] - tol <= pv
            and pv <= fhi[i, v] + tol
        ):
            keep[m] = i
            m += 1
    w_in = np.empty(m, dtype=np.float64)
    w_out = np.empty(m, dtype=np.float64)
    for k in range(m):
        i = keep[k]
        t1 = (flo[i, ax] - tol - pa) / d
        t2 = (fhi[i, ax] + tol - pa) / d
        if t1 <= t2:
            w_in[k] = t1
            w_out[k] = t2
        else:
            w_in[k] = t2
            w_out[k] = t1
    order = np.argsort(w_in, kind="mergesort")
    idx = np.empty(m, dtype=np.int64)
    w_in_s = np.empty(m, dtype=np.float64)
    w_out_s = np.empty(m, dtype=np.float64)
    for k in range(m):
        j = order[k]
        idx[k] = keep[j]
        w_in_s[k] = w_in[j]
        w_out_s[k] = w_out[j]
    return idx, w_in_s, w_out_s


@njit(cache=True)
def planar_point_state(pu, pv, verts, offsets, kinds, arcs, tol):
    """Classify a point against the rings of a planar face.

    ``verts`` holds the rings back to back, ring ``r`` occupying rows
    ``offsets[r]:offsets[r + 1]``; segment ``k`` runs from ``verts[k]``
    to the next vertex of its ring.  ``kinds[k]`` is ``0`` for a
    straight segment and ``+1`` / ``-1`` for a circular arc on the
    right / left half of its circle ``arcs[k, :3] = (cu, cv, r)``,
    whose angular span ``arcs[k, 3:5] = (psi_lo, psi_hi)`` is measured
    from that half's own direction (``atan2(v - cv, ±(u - cu))``).
    Such an arc is monotone in ``v``, so the even-odd rule treats it
    like a segment: it crosses the ray towards ``+u`` exactly when its
    ends straddle ``pv``, at ``cu ± sqrt(r² - (pv - cv)²)`` — no
    tessellation, no chord error.

    Returns ``2`` when the point is within ``tol`` of any ring segment
    (the kernel's ``ON``), else ``1`` when the even-odd rule over all
    rings puts it inside, else ``0``.
    """
    tol2 = tol * tol
    inside = False
    n_rings = offsets.shape[0] - 1
    for r in range(n_rings):
        start = offsets[r]
        stop = offsets[r + 1]
        n = stop - start
        if n < 2:
            continue
        j = stop - 1
        for i in range(start, stop):
            xi = verts[i, 0]
            yi = verts[i, 1]
            xj = verts[j, 0]
            yj = verts[j, 1]
            kind = kinds[j]
            if kind == 0:
                dx = xj - xi
                dy = yj - yi
                length2 = dx * dx + dy * dy
                if length2 > 0.0:
                    t = ((pu - xi) * dx + (pv - yi) * dy) / length2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                else:
                    t = 0.0
                ex = xi + t * dx - pu
                ey = yi + t * dy - pv
                if ex * ex + ey * ey <= tol2:
                    return 2
                if (yi > pv) != (yj > pv):
                    x_cross = (xj - xi) * (pv - yi) / (yj - yi) + xi
                    if pu < x_cross:
                        inside = not inside
            else:
                cu = arcs[j, 0]
                cv = arcs[j, 1]
                rad = arcs[j, 2]
                du = pu - cu
                dv = pv - cv
                psi = math.atan2(dv, kind * du)
                if arcs[j, 3] <= psi <= arcs[j, 4]:
                    if abs(math.sqrt(du * du + dv * dv) - rad) <= tol:
                        return 2
                else:
                    ei = (xi - pu) * (xi - pu) + (yi - pv) * (yi - pv)
                    ej = (xj - pu) * (xj - pu) + (yj - pv) * (yj - pv)
                    if ei <= tol2 or ej <= tol2:
                        return 2
                if (yi > pv) != (yj > pv):
                    h = rad * rad - dv * dv
                    if h < 0.0:
                        h = 0.0
                    x_cross = cu + kind * math.sqrt(h)
                    if pu < x_cross:
                        inside = not inside
            j = i
    return 1 if inside else 0


@njit(cache=True)
def _bisect_left(a, x, lo, hi):
    """First index in ``a[lo:hi]`` (as a global index) with ``a[i] >= x``."""
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _bisect_right(a, x, lo, hi):
    """First index in ``a[lo:hi]`` (as a global index) with ``a[i] > x``."""
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@njit(cache=True)
def axis_line_pairs(line_ax, line_pu, line_pv, flo, fhi, tol):
    """``(line, row)`` pairs of axis-aligned lines and the rows whose
    tolerance-inflated box they touch — :func:`axis_line_candidates` for
    many lines at once.

    A line ``i`` runs along axis ``line_ax[i]`` through the transverse
    point ``(line_pu[i], line_pv[i])`` (the other two axes in xyz
    order).  Per axis the lines are sorted by ``pu`` so every row
    tests only the lines inside its ``u`` range (binary search) and
    filters those by ``v``; the conditions are the ones of the
    single-line pass, so the pair set is the same.
    """
    n_lines = line_ax.shape[0]
    n_rows = flo.shape[0]
    out_line = np.empty(0, dtype=np.int64)
    out_row = np.empty(0, dtype=np.int64)
    for ax in range(3):
        if ax == 0:
            u, v = 1, 2
        elif ax == 1:
            u, v = 0, 2
        else:
            u, v = 0, 1
        n_sel = 0
        for i in range(n_lines):
            if line_ax[i] == ax:
                n_sel += 1
        if n_sel == 0:
            continue
        sel = np.empty(n_sel, dtype=np.int64)
        k = 0
        for i in range(n_lines):
            if line_ax[i] == ax:
                sel[k] = i
                k += 1
        pu_sel = np.empty(n_sel, dtype=np.float64)
        for k in range(n_sel):
            pu_sel[k] = line_pu[sel[k]]
        order = np.argsort(pu_sel, kind="mergesort")
        pu_s = np.empty(n_sel, dtype=np.float64)
        lines_s = np.empty(n_sel, dtype=np.int64)
        for k in range(n_sel):
            pu_s[k] = pu_sel[order[k]]
            lines_s[k] = sel[order[k]]
        # Two passes: count, then fill.
        count = 0
        for r in range(n_rows):
            a = _bisect_left(pu_s, flo[r, u] - tol, 0, n_sel)
            b = _bisect_right(pu_s, fhi[r, u] + tol, 0, n_sel)
            lo_v = flo[r, v] - tol
            hi_v = fhi[r, v] + tol
            for k in range(a, b):
                pv = line_pv[lines_s[k]]
                if lo_v <= pv and pv <= hi_v:
                    count += 1
        pair_line = np.empty(count, dtype=np.int64)
        pair_row = np.empty(count, dtype=np.int64)
        m = 0
        for r in range(n_rows):
            a = _bisect_left(pu_s, flo[r, u] - tol, 0, n_sel)
            b = _bisect_right(pu_s, fhi[r, u] + tol, 0, n_sel)
            lo_v = flo[r, v] - tol
            hi_v = fhi[r, v] + tol
            for k in range(a, b):
                line = lines_s[k]
                pv = line_pv[line]
                if lo_v <= pv and pv <= hi_v:
                    pair_line[m] = line
                    pair_row[m] = r
                    m += 1
        out_line = np.concatenate((out_line, pair_line))
        out_row = np.concatenate((out_row, pair_row))
    return out_line, out_row


@njit(cache=True)
def planar_pair_hits(
    pair_line,
    pair_row,
    line_pu,
    line_pv,
    line_origin,
    row_level,
    row_outward,
    verts,
    kinds,
    arcs,
    ring_offsets,
    row_ring_lo,
    row_ring_hi,
    tol,
):
    """In-house hits of axis-aligned lines on planar rows normal to them.

    For every pair the hit parameter is the row's level relative to
    the line's origin coordinate, the state the ring test at the line's
    transverse point (``ring_offsets[row_ring_lo[r]:row_ring_hi[r]]``
    are the row's rings in ``verts``), the step the outward sign
    against the line's ``+`` direction.  Returns ``(w, step,
    untrusted, valid)``; ``valid`` is False where the line misses the
    row.
    """
    n = pair_line.shape[0]
    w = np.empty(n, dtype=np.float64)
    step = np.zeros(n, dtype=np.int64)
    untrusted = np.zeros(n, dtype=np.bool_)
    valid = np.zeros(n, dtype=np.bool_)
    for k in range(n):
        line = pair_line[k]
        r = pair_row[k]
        state = planar_point_state(
            line_pu[line],
            line_pv[line],
            verts,
            ring_offsets[row_ring_lo[r] : row_ring_hi[r]],
            kinds,
            arcs,
            tol,
        )
        if state == 0:
            continue
        w[k] = (row_level[r] - line_origin[line]) / 1.0
        valid[k] = True
        if state == 2:
            untrusted[k] = True
        else:
            step[k] = 1 if row_outward[r] < 0.0 else -1
    return w, step, untrusted, valid


@njit(cache=True)
def cylinder_line_hits(p0, d, c, a, x, y, r, sense, uv, tol, out_w, out_step, out_untrusted):
    """Hits of the line ``p0 + w d`` (``d`` a unit vector) on a
    cylindrical face bounded by its own parameter lines — the in-house
    counterpart of the kernel's intersector for the rows of
    :class:`~magnelio.geo._occ_backend._PrefilteredLineSolid` that
    carry cylinder data.

    The face is P(u, v) = c + r (cos u X + sin u Y) + v A with the
    outward radial ``sense`` and the (u, v) box ``uv``; a full round
    (``uv[1] - uv[0] >= 2π``) has no seam boundary.  Writes up to two
    ``(w, step, untrusted)`` triples into the ``out_*`` arrays and
    returns their count: the roots of |q⊥ + w d⊥|² = r² (q = p0 − c,
    ⊥ the component normal to the axis), each kept when its (u, v)
    lies in the face — within ``tol`` (geometric: ``tol / r`` in u) of
    a rim or a generatrix boundary it is the kernel's ``ON``, i.e.
    untrusted.  The step is the sign of the outward normal against
    ``d`` (+1 entering the solid); a line touching the surface (its
    distance to the axis within ``tol`` of ``r``) reports one
    tangential hit, step 0; a line parallel to the axis — in the
    surface, inside or beside it — reports none, as the kernel does
    for a line in a planar row.
    """
    two_pi = 2.0 * np.pi
    q0 = p0[0] - c[0]
    q1 = p0[1] - c[1]
    q2 = p0[2] - c[2]
    qa = q0 * a[0] + q1 * a[1] + q2 * a[2]
    da = d[0] * a[0] + d[1] * a[1] + d[2] * a[2]
    qp0 = q0 - qa * a[0]
    qp1 = q1 - qa * a[1]
    qp2 = q2 - qa * a[2]
    dp0 = d[0] - da * a[0]
    dp1 = d[1] - da * a[1]
    dp2 = d[2] - da * a[2]
    aa = dp0 * dp0 + dp1 * dp1 + dp2 * dp2
    if aa <= 1e-24:
        return 0
    bb = qp0 * dp0 + qp1 * dp1 + qp2 * dp2
    # Distance of the line to the axis, from the perpendicular foot
    # (q⊥ − (q⊥·d⊥) d⊥ / |d⊥|²) — well conditioned where the
    # discriminant b² − a c cancels on a touching line.
    t_foot = bb / aa
    f0 = qp0 - t_foot * dp0
    f1 = qp1 - t_foot * dp1
    f2 = qp2 - t_foot * dp2
    dist = np.sqrt(f0 * f0 + f1 * f1 + f2 * f2)
    n_roots = 0
    w_a = 0.0
    w_b = 0.0
    if abs(dist - r) <= tol:
        n_roots = 1
        w_a = -t_foot
    elif dist < r:
        half = np.sqrt((r - dist) * (r + dist) / aa)
        n_roots = 2
        w_a = -t_foot - half
        w_b = -t_foot + half
    else:
        return 0
    umin = uv[0]
    umax = uv[1]
    vmin = uv[2]
    vmax = uv[3]
    span = umax - umin
    full = span >= two_pi - 1e-9
    tol_u = tol / r
    n_out = 0
    for k in range(n_roots):
        w = w_a if k == 0 else w_b
        rel0 = q0 + w * d[0]
        rel1 = q1 + w * d[1]
        rel2 = q2 + w * d[2]
        v = rel0 * a[0] + rel1 * a[1] + rel2 * a[2]
        if v < vmin - tol or v > vmax + tol:
            continue
        on = v <= vmin + tol or v >= vmax - tol
        if not full:
            ru = rel0 * x[0] + rel1 * x[1] + rel2 * x[2]
            rv = rel0 * y[0] + rel1 * y[1] + rel2 * y[2]
            du = np.arctan2(rv, ru) - umin
            du = du - two_pi * np.floor(du / two_pi)
            if du >= two_pi - tol_u:
                du -= two_pi
            if du < -tol_u or du > span + tol_u:
                continue
            if du <= tol_u or du >= span - tol_u:
                on = True
        out_w[n_out] = w
        if on:
            out_step[n_out] = 0
            out_untrusted[n_out] = True
        elif n_roots == 1:
            out_step[n_out] = 0
            out_untrusted[n_out] = False
        else:
            # Outward normal = sense · (rel − v A) / r; entering where it
            # points against the line direction.
            dn = (
                d[0] * (rel0 - v * a[0]) + d[1] * (rel1 - v * a[1]) + d[2] * (rel2 - v * a[2])
            ) * sense
            out_step[n_out] = 1 if dn < 0.0 else -1
            out_untrusted[n_out] = False
        n_out += 1
    return n_out


@njit(cache=True)
def cylinder_pair_hits(
    pair_line,
    pair_row,
    line_ax,
    line_pu,
    line_pv,
    line_origin,
    row_c,
    row_a,
    row_x,
    row_y,
    row_r,
    row_sense,
    row_uv,
    tol,
):
    """In-house hits of axis-aligned lines on cylindrical rows —
    :func:`cylinder_line_hits` for every ``(line, row)`` pair, the
    line through ``(pu, pv)`` along its axis with parameters relative
    to its origin coordinate.  Returns ``(line, row, w, step,
    untrusted)`` of the hits found, in pair order.
    """
    n = pair_line.shape[0]
    line_out = np.empty(2 * n, dtype=np.int64)
    row_out = np.empty(2 * n, dtype=np.int64)
    w_out = np.empty(2 * n, dtype=np.float64)
    step_out = np.empty(2 * n, dtype=np.int64)
    unt_out = np.empty(2 * n, dtype=np.bool_)
    p0 = np.empty(3, dtype=np.float64)
    d = np.zeros(3, dtype=np.float64)
    tmp_w = np.empty(2, dtype=np.float64)
    tmp_step = np.empty(2, dtype=np.int64)
    tmp_unt = np.empty(2, dtype=np.bool_)
    m = 0
    for k in range(n):
        line = pair_line[k]
        row = pair_row[k]
        ax = line_ax[line]
        if ax == 0:
            u, v = 1, 2
        elif ax == 1:
            u, v = 0, 2
        else:
            u, v = 0, 1
        p0[ax] = line_origin[line]
        p0[u] = line_pu[line]
        p0[v] = line_pv[line]
        d[0] = 0.0
        d[1] = 0.0
        d[2] = 0.0
        d[ax] = 1.0
        cnt = cylinder_line_hits(
            p0,
            d,
            row_c[row],
            row_a[row],
            row_x[row],
            row_y[row],
            row_r[row],
            row_sense[row],
            row_uv[row],
            tol,
            tmp_w,
            tmp_step,
            tmp_unt,
        )
        for j in range(cnt):
            line_out[m] = line
            row_out[m] = row
            w_out[m] = tmp_w[j]
            step_out[m] = tmp_step[j]
            unt_out[m] = tmp_unt[j]
            m += 1
    return line_out[:m], row_out[:m], w_out[:m], step_out[:m], unt_out[:m]


@njit(cache=True)
def line_flags(hit_offsets, hit_step, hit_untrusted):
    """Whether a line's sorted hits anchor parity: no untrusted hit,
    crossings alternate and the first one enters the solid."""
    n = hit_offsets.shape[0] - 1
    ok = np.ones(n, dtype=np.bool_)
    for line in range(n):
        prev = 0
        for k in range(hit_offsets[line], hit_offsets[line + 1]):
            if hit_untrusted[k]:
                ok[line] = False
                break
            s = hit_step[k]
            if s == 0:
                continue
            if prev == 0:
                if s != 1:
                    ok[line] = False
                    break
            elif s == prev:
                ok[line] = False
                break
            prev = s
    return ok


@njit(cache=True)
def axis_edge_windows(edge_line, edge_offset, edge_length, hit_offsets, hit_w, tol):
    """Global index range of the hits inside every edge's
    tolerance-inflated parameter window on its line."""
    n = edge_line.shape[0]
    lo = np.empty(n, dtype=np.int64)
    hi = np.empty(n, dtype=np.int64)
    for e in range(n):
        line = edge_line[e]
        a = hit_offsets[line]
        b = hit_offsets[line + 1]
        lo[e] = _bisect_left(hit_w, edge_offset[e] - tol, a, b)
        hi[e] = _bisect_right(hit_w, edge_offset[e] + edge_length[e] + tol, a, b)
    return lo, hi


@njit(cache=True)
def axis_edge_fractions(
    edge_line,
    edge_offset,
    edge_length,
    win_lo,
    win_hi,
    hit_w,
    hit_step,
    hit_untrusted,
    cross_offsets,
    cross_w,
    line_ok,
    seg_offsets,
    tol,
):
    """f_L of axis-aligned edges from their line's hits, and the
    sub-segments of the edges that need point classification.

    Follows the per-edge bookkeeping of the sequential pass: crossing
    parameters clipped to the edge, deduplicated within ``tol``,
    segment boundaries, states from alternating transitions (or the
    line's parity at the edge midpoint when nothing crosses).  An edge
    whose hits cannot anchor that — an untrusted hit in its window,
    non-alternating transitions, a line without parity — is marked
    ``fallback`` and its sub-segment midpoints (parameter along the
    edge) and lengths are written to ``seg_t``/``seg_len`` from
    ``seg_offsets[e]`` on, ``seg_count[e]`` of them.
    """
    n = edge_line.shape[0]
    f_l = np.ones(n, dtype=np.float64)
    fallback = np.zeros(n, dtype=np.bool_)
    seg_count = np.zeros(n, dtype=np.int64)
    seg_t = np.empty(seg_offsets[n], dtype=np.float64)
    seg_len = np.empty(seg_offsets[n], dtype=np.float64)
    for e in range(n):
        line = edge_line[e]
        offset = edge_offset[e]
        length = edge_length[e]
        lo = win_lo[e]
        hi = win_hi[e]
        n_win = hi - lo
        clean = True
        for k in range(lo, hi):
            if hit_untrusted[k]:
                clean = False
                break
        # Crossing parameters within [0, length], deduplicated.
        params = np.empty(n_win, dtype=np.float64)
        n_par = 0
        for k in range(lo, hi):
            p = hit_w[k] - offset
            if p > length:
                p = length
            if p < 0.0:
                p = 0.0
            if n_par == 0 or p - params[n_par - 1] > tol:
                params[n_par] = p
                n_par += 1
        bounds = np.empty(n_par + 2, dtype=np.float64)
        bounds[0] = 0.0
        n_b = 1
        for k in range(n_par):
            if params[k] - bounds[n_b - 1] > tol:
                bounds[n_b] = params[k]
                n_b += 1
        if length - bounds[n_b - 1] > tol:
            bounds[n_b] = length
            n_b += 1
        use_fallback = not clean
        first_outside = True
        trans_w = np.empty(n_win, dtype=np.float64)
        n_tr = 0
        if not use_fallback:
            trans_s = np.empty(n_win, dtype=np.int64)
            for k in range(lo, hi):
                s = hit_step[k]
                if s == 0:
                    continue
                # Insertion sort by (w, step), as the sequential pass sorts.
                wk = hit_w[k] - offset
                j = n_tr
                while j > 0 and (
                    trans_w[j - 1] > wk or (trans_w[j - 1] == wk and trans_s[j - 1] > s)
                ):
                    trans_w[j] = trans_w[j - 1]
                    trans_s[j] = trans_s[j - 1]
                    j -= 1
                trans_w[j] = wk
                trans_s[j] = s
                n_tr += 1
            if n_tr > 0:
                for k in range(1, n_tr):
                    if trans_s[k] != -trans_s[k - 1]:
                        use_fallback = True
                        break
                first_outside = trans_s[0] > 0
            elif line_ok[line]:
                a = cross_offsets[line]
                b = cross_offsets[line + 1]
                k = _bisect_right(cross_w, offset + length / 2.0, a, b) - a
                first_outside = k % 2 == 0
            else:
                use_fallback = True
        outside_len = 0.0
        pos = seg_offsets[e]
        for j in range(n_b - 1):
            seg_start = bounds[j]
            seg_end = bounds[j + 1]
            s_len = seg_end - seg_start
            if s_len < tol:
                continue
            t_mid = (seg_start + seg_end) / 2
            if use_fallback:
                seg_t[pos] = t_mid
                seg_len[pos] = s_len
                pos += 1
            else:
                k = _bisect_right(trans_w, t_mid, 0, n_tr)
                outside = first_outside if k % 2 == 0 else not first_outside
                if outside:
                    outside_len += s_len
        if use_fallback:
            fallback[e] = True
            seg_count[e] = pos - seg_offsets[e]
        else:
            frac = outside_len / length
            if frac > 1.0:
                frac = 1.0
            if frac < 0.0:
                frac = 0.0
            f_l[e] = frac
    return f_l, fallback, seg_count, seg_t, seg_len


@njit(cache=True)
def classify_on_lines(pt_line, w_rel, cross_offsets, cross_w, touch_offsets, touch_w, line_ok, tol):
    """State of points on their probe lines: ``0`` outside, ``1``
    inside (within ``tol`` of a crossing or of a trusted tangential
    hit counts as inside — the classifier's ``ON``), ``2`` undecided
    (the line has no parity)."""
    n = pt_line.shape[0]
    state = np.empty(n, dtype=np.int64)
    for i in range(n):
        line = pt_line[i]
        if not line_ok[line]:
            state[i] = 2
            continue
        a = cross_offsets[line]
        b = cross_offsets[line + 1]
        w = w_rel[i]
        pos = _bisect_left(cross_w, w - tol, a, b)
        ta = touch_offsets[line]
        tb = touch_offsets[line + 1]
        tpos = _bisect_left(touch_w, w - tol, ta, tb)
        if pos < b and cross_w[pos] <= w + tol:
            state[i] = 1
        elif tpos < tb and touch_w[tpos] <= w + tol:
            state[i] = 1
        else:
            k = _bisect_right(cross_w, w, a, b) - a
            state[i] = 0 if k % 2 == 0 else 1
    return state


@njit(cache=True)
def boundary_on_lines(
    pt_line,
    w_rel,
    line_pu,
    line_pv,
    hit_offsets,
    hit_w,
    hit_untrusted,
    hit_row,
    face_verts,
    face_kinds,
    face_arcs,
    face_ring_offsets,
    row_face_lo,
    row_face_hi,
    tol,
):
    """Whether a point lies on the solid's boundary by the hits of its
    line: a hit within ``tol`` of the point's parameter that is trusted
    (the face's interior), or an in-house hit whose point lies in or
    on the row's *face* by the face's own rings
    (``face_ring_offsets[row_face_lo[r]:row_face_hi[r]]`` in
    ``face_verts``; a tile border reads ``ON`` on the tile but is not an
    outline — the point is still on the face).  Kernel hits
    (``hit_row < 0``) do not qualify."""
    n = pt_line.shape[0]
    on = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        line = pt_line[i]
        a = hit_offsets[line]
        b = hit_offsets[line + 1]
        w = w_rel[i]
        for k in range(_bisect_left(hit_w, w - tol, a, b), b):
            if hit_w[k] > w + tol:
                break
            if not hit_untrusted[k]:
                on[i] = True
                break
            r = hit_row[k]
            if r < 0 or row_face_hi[r] <= row_face_lo[r]:
                continue
            state = planar_point_state(
                line_pu[line],
                line_pv[line],
                face_verts,
                face_ring_offsets[row_face_lo[r] : row_face_hi[r]],
                face_kinds,
                face_arcs,
                tol,
            )
            if state != 0:
                on[i] = True
                break
    return on


@njit(cache=True)
def segment_fractions(seg_offsets, seg_count, seg_len, seg_outside, edge_length):
    """f_L of the fallback edges: outside length over the edge length,
    summed in sub-segment order."""
    n = seg_count.shape[0]
    f_l = np.empty(n, dtype=np.float64)
    for e in range(n):
        outside_len = 0.0
        for k in range(seg_offsets[e], seg_offsets[e] + seg_count[e]):
            if seg_outside[k]:
                outside_len += seg_len[k]
        frac = outside_len / edge_length[e]
        if frac > 1.0:
            frac = 1.0
        if frac < 0.0:
            frac = 0.0
        f_l[e] = frac
    return f_l
