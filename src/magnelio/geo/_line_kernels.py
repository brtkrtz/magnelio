"""Kernels of the edge pass: axis-aligned line candidates and planar hits.

``njit`` is a no-op decorator when Numba is unavailable; the kernels are
plain loops and give the same results either way.
"""

from __future__ import annotations

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
def planar_point_state(pu, pv, verts, offsets, tol):
    """Classify a point against the rings of a planar face.

    ``verts`` holds the rings back to back, ring ``r`` occupying rows
    ``offsets[r]:offsets[r + 1]``.  Returns ``2`` when the point is
    within ``tol`` of any ring segment (the kernel's ``ON``), else ``1``
    when the even-odd rule over all rings puts it inside, else ``0``.
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
            j = i
    return 1 if inside else 0
