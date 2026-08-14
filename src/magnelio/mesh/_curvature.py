"""Curvature pullback factors for the conformal wall booking (DD-098).

Conformal wall losses read tangential H a distance ``d1 = O(h)`` off
the wall and inherit a first-order position bias (1/r-amplified on
curved conductors).  At a PEC wall the tangential curl component
vanishes for all time (``E_tan == 0``) and ``H_n == 0``, which yields
the exact, mode-free identity

    dH_tan/dn |wall = -W . H_tan

with ``W = grad_tan(n_hat)`` the shape operator of the wall (n_hat
pointing out of the conductor; convex-from-air walls have positive
normal curvatures, concave walls negative).  Integrating along the
normal with the offset-surface curvature gives the multiplicative
pullback per booking entry

    c_b = max(1 + kappa_b * d1_b, 0)

which is exact at every distance for 1/r line fields (coax class)
and first-order exact in general; every conformal booking weight is
scaled by ``c_b**2``.  ``kappa_b`` is the normal curvature of the
entry's wall family along the booked component direction, estimated
from the rotation of neighbouring wall planes; ``d1_b`` the signed
distance of the booked H sample from the family plane.  Flat wall
families whose neighbourhood is flat carry exactly ``kappa = 0`` —
grid-aligned flat scenes are bit-exact no-ops.

Everything is computed at enumeration time from existing mesh
channels (``A_face_pec``/``A_face_pec_jump``); no mesher or store
change is involved.  Derivation, gates and measurements: internal
dossier ``investigations/wall_curvature/{DERIVATION,MEASUREMENTS}.md``.
"""

from __future__ import annotations

import numpy as np

_FACES = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
_COS_GATE = float(np.cos(np.pi / 4))
_EYE = np.eye(3)


def _invert_face_offset(cell_lo, cell_hi, n_hat, face, target):
    """Solve ``area{n_hat . x <= p} == target`` on one cell face.

    The covered area of a half-plane clip of a rectangle is piecewise
    quadratic in the plane offset (quadratic / linear / quadratic
    between the sorted corner values), so the inversion is closed
    form.  Returns ``(p, dA/dp)`` or ``None`` when the face cannot
    pin the offset (plane parallel to the face).
    """
    ax, side = _FACES[face]
    ua, va = (ax + 1) % 3, (ax + 2) % 3
    ca = cell_hi[ax] if side else cell_lo[ax]
    nu, nv = n_hat[ua], n_hat[va]
    wu = cell_hi[ua] - cell_lo[ua]
    wv = cell_hi[va] - cell_lo[va]
    # corner-value gaps as exact products (a sorted-value difference
    # cancels catastrophically for near-axis in-plane normals)
    du, dv = abs(nu) * wu, abs(nv) * wv
    span = du + dv
    if span <= 0.0:
        return None
    s1 = min(nu * cell_lo[ua], nu * cell_hi[ua]) + min(nv * cell_lo[va], nv * cell_hi[va])
    a_full = wu * wv
    g1 = min(du, dv)  # width of the quadratic pieces
    d = abs(nu * nv)
    if g1 <= 0.0 or d <= 0.0:
        # in-plane normal along one axis: the ramp is linear
        q = s1 + (target / a_full) * span
        return q + n_hat[ax] * ca, a_full / span
    a2 = g1 * g1 / (2.0 * d)  # area at the first piece boundary
    a3 = a_full - a2
    if target <= a2:
        q = s1 + np.sqrt(2.0 * d * target)
        sens = np.sqrt(2.0 * target / d)
    elif target >= a3:
        q = s1 + span - np.sqrt(2.0 * d * (a_full - target))
        sens = np.sqrt(2.0 * (a_full - target) / d)
    else:
        slope = g1 / d  # constant chord on the middle piece
        q = s1 + g1 + (target - a2) / slope
        sens = slope
    return q + n_hat[ax] * ca, sens


class CurvatureFactors:
    """Per-entry curvature pullback factors of one conformal mesh.

    Parameters mirror the intermediate arrays of
    ``_conformal_solid_surfaces`` (per-component face views, per-cell
    wall vectors and their flat-family split, live-cell mask); the
    class reconstructs every wall family's plane, fits the shape
    operator from neighbouring planes (angle-gated symmetric
    least-squares on the Gauss-map rotation) and serves the squared
    factor per booked sample via :meth:`scale`.
    """

    def __init__(self, grid, a_pec, a_full, a_jump, w, w_flat, live) -> None:
        x, y, z = grid.x, grid.y, grid.z
        xc, yc, zc = (0.5 * (v[:-1] + v[1:]) for v in (x, y, z))
        # H-sample position arrays per component lattice along each axis
        self._ax_pos = [(x, yc, zc), (xc, y, zc), (xc, yc, z)]
        self._fam: dict[tuple, list] = {}

        ii, jj, kk = np.nonzero(live)
        for i, j, k in zip(ii, jj, kk):
            key = (int(i), int(j), int(k))
            cell_lo = np.array([x[i], y[j], z[k]])
            cell_hi = np.array([x[i + 1], y[j + 1], z[k + 1]])
            centre = 0.5 * (cell_lo + cell_hi)
            recs = []
            # flat families: the wall lies IN a jumped grid plane
            for ax in range(3):
                planes = (x, y, z)[ax]
                idx_hi = [i, j, k]
                idx_hi[ax] += 1
                j_hi = a_jump[ax][tuple(idx_hi)]
                j_lo = a_jump[ax][key]
                for jmp, sgn, plane in (
                    (j_hi, -1.0, planes[key[ax] + 1]),
                    (-j_lo, 1.0, planes[key[ax]]),
                ):
                    if jmp <= 0.0:
                        continue
                    n_hat = np.zeros(3)
                    n_hat[ax] = sgn
                    p = sgn * plane
                    foot = centre - (float(n_hat @ centre) - p) * n_hat
                    recs.append({"n": n_hat, "p": p, "foot": foot, "area": float(jmp), "fit": True})
            # curved family: divergence-identity normal + offset from
            # the covered-area inversion on cut, non-jump faces
            w_c = np.array([w[a][key] - w_flat[a][key] for a in range(3)])
            a_c = float(np.linalg.norm(w_c))
            if a_c > 0.0:
                rec = self._curved_plane(key, cell_lo, cell_hi, w_c, a_c, a_pec, a_full, a_jump)
                if rec is None:
                    # no invertible face (sliver): book unscaled and
                    # keep the fake plane out of every fit
                    recs.append(
                        {
                            "n": -w_c / a_c,
                            "p": float((-w_c / a_c) @ centre),
                            "foot": centre,
                            "area": a_c,
                            "fit": False,
                        }
                    )
                else:
                    n_hat, p = rec
                    foot = centre - (float(n_hat @ centre) - p) * n_hat
                    recs.append({"n": n_hat, "p": p, "foot": foot, "area": a_c, "fit": True})
            if recs:
                self._fam[key] = recs
        self._fit_all()

    @staticmethod
    def _curved_plane(key, cell_lo, cell_hi, w_c, a_c, a_pec, a_full, a_jump):
        """(n_hat, p) of the curved family.  The normal comes from the
        FLAT-CORRECTED wall vector — a jumped face stores the full
        covered area and would tilt the combined-w normal into the
        flat family (DD-097 par. 1.1 / DD-098 gate 2)."""
        i, j, k = key
        n_hat = -w_c / a_c
        eps = 1e-9
        est, sens = [], []
        for f, (ax, side) in enumerate(_FACES):
            idx = [i, j, k]
            idx[ax] += side
            if a_jump[ax][tuple(idx)] != 0.0:
                continue
            area = a_pec[ax][tuple(idx)]
            full = float(np.asarray(a_full[ax])[tuple(idx)])
            if area <= eps * full or area >= (1.0 - eps) * full:
                continue
            r = _invert_face_offset(cell_lo, cell_hi, n_hat, f, area)
            if r is None:
                continue
            est.append(r[0])
            sens.append(max(r[1], 0.0))
        if not est:
            return None
        est_a, sens_a = np.array(est), np.array(sens)
        stot = float(sens_a.sum())
        p = est_a[int(np.argmax(sens_a))] if stot == 0.0 else float(np.sum(est_a * sens_a) / stot)
        return n_hat, p

    def _fit_all(self) -> None:
        """Shape operator per family from neighbouring wall planes.

        Neighbour planes within the 3x3x3 stencil sample the Gauss map
        one cell apart; families whose normals differ by more than 45
        degrees belong to a different wall (material edges) and are
        excluded.  A coplanar-only neighbourhood yields exactly zero
        curvature (flat walls are exact no-ops); no usable neighbour
        leaves the family unscaled."""
        for (i, j, k), recs in self._fam.items():
            for rec in recs:
                kap = np.zeros(3)
                rec["kap"] = kap
                if not rec["fit"]:
                    continue
                n0 = rec["n"]
                e1 = np.cross(n0, _EYE[int(np.argmin(np.abs(n0)))])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(n0, e1)
                rows, rhs = [], []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            for nb in self._fam.get((i + di, j + dj, k + dk), ()):
                                if nb is rec or not nb["fit"]:
                                    continue
                                if float(nb["n"] @ n0) < _COS_GATE:
                                    continue
                                dy = nb["foot"] - rec["foot"]
                                t1 = float(dy @ e1)
                                t2 = float(dy @ e2)
                                if t1 * t1 + t2 * t2 < 1e-24:
                                    continue
                                dn = nb["n"] - n0
                                rows.append([t1, t2, 0.0])
                                rows.append([0.0, t1, t2])
                                rhs.append(float(dn @ e1))
                                rhs.append(float(dn @ e2))
                if not rows:
                    continue
                sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
                w2 = np.array([[sol[0], sol[1]], [sol[1], sol[2]]])
                for c in range(3):
                    tt = _EYE[c] - float(n0[c]) * n0
                    nt = float(np.linalg.norm(tt))
                    if nt < 1e-3:
                        continue
                    t = np.array([float(tt @ e1), float(tt @ e2)]) / nt
                    kap[c] = float(t @ w2 @ t)

    def scale(self, pos3, ax, face_idx):
        """Squared pullback factor per wall cell for one booked face.

        Parameters
        ----------
        pos3 : tuple of int arrays
            Wall-cell indices (i, j, k) of the booking loop.
        ax : int
            Booked H component / face-normal axis.
        face_idx : list of int arrays
            Index triple of the booked H face per cell.

        Returns
        -------
        ndarray
            Per-cell weight scale: the family-area-weighted mean of
            ``c_b**2`` over the cell's wall families (1.0 wherever no
            curvature information exists).
        """
        n = pos3[0].shape[0]
        out = np.ones(n)
        pos = self._ax_pos[ax]
        for s in range(n):
            key = (int(pos3[0][s]), int(pos3[1][s]), int(pos3[2][s]))
            recs = self._fam.get(key)
            if not recs:
                continue
            x_s = np.array([pos[a][int(face_idx[a][s])] for a in range(3)])
            num = 0.0
            den = 0.0
            for rec in recs:
                area = rec["area"]
                den += area
                kap = rec["kap"][ax]
                if kap == 0.0:
                    num += area
                    continue
                d1 = float(rec["n"] @ x_s) - rec["p"]
                c = 1.0 + kap * d1
                num += area * (c * c if c > 0.0 else 0.0)
            if den > 0.0:
                out[s] = num / den
        return out
