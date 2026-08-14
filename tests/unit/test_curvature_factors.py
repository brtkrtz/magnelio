"""Unit tests for the DD-098 curvature pullback (mesh/curvature.py).

The closed-form covered-area inversion is checked against a brute
polygon clip; the factor application (clamp, family mixing, exact
no-op) is checked on hand-built family records.  End-to-end behaviour
on real conformal meshes is covered by the integration suite
(test_sibc_validation, test_wall_loss_q).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.mesh._curvature import (
    _FACES,
    CurvatureFactors,
    _invert_face_offset,
)

rng = np.random.default_rng(20260728)


# ---------------------------------------------------------------------------
# brute-force reference: polygon half-plane clip
# ---------------------------------------------------------------------------


def _clip_area(u0, u1, v0, v1, a, b, c):
    poly = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    out = []
    n = len(poly)
    d = [a * u + b * v - c for u, v in poly]
    for i in range(n):
        j = (i + 1) % n
        if d[i] <= 0.0:
            out.append(poly[i])
        if (d[i] < 0.0) != (d[j] < 0.0):
            t = d[i] / (d[i] - d[j])
            out.append(
                (
                    poly[i][0] + t * (poly[j][0] - poly[i][0]),
                    poly[i][1] + t * (poly[j][1] - poly[i][1]),
                )
            )
    if len(out) < 3:
        return 0.0
    u = np.array([p[0] for p in out])
    v = np.array([p[1] for p in out])
    return 0.5 * abs(np.dot(u, np.roll(v, -1)) - np.dot(v, np.roll(u, -1)))


def _face_area(cell_lo, cell_hi, n_hat, p, face):
    ax, side = _FACES[face]
    ua, va = (ax + 1) % 3, (ax + 2) % 3
    ca = cell_hi[ax] if side else cell_lo[ax]
    return _clip_area(
        cell_lo[ua], cell_hi[ua], cell_lo[va], cell_hi[va], n_hat[ua], n_hat[va], p - n_hat[ax] * ca
    )


def test_offset_inversion_matches_brute_clip():
    n_checked = 0
    for _ in range(3000):
        d = rng.uniform(0.5, 2.0, size=3)
        cell_lo = rng.uniform(-1.0, 1.0, size=3)
        cell_hi = cell_lo + d
        v = rng.normal(size=3)
        if rng.random() < 0.2:  # near-axis normals stress the branches
            v = np.eye(3)[rng.integers(3)] + rng.normal(size=3) * 1e-7
        n_hat = v / np.linalg.norm(v)
        centre = 0.5 * (cell_lo + cell_hi)
        p = float(n_hat @ centre + rng.uniform(-0.4, 0.4) * float((n_hat * d).sum()))
        for face in range(6):
            area = _face_area(cell_lo, cell_hi, n_hat, p, face)
            full = _face_area(cell_lo, cell_hi, n_hat, 1e9, face)
            if area <= 1e-9 * full or area >= (1 - 1e-9) * full:
                continue
            r = _invert_face_offset(cell_lo, cell_hi, n_hat, face, area)
            assert r is not None
            p_rec, sens = r
            # p to ~1e-8 h (conditioning: barely-clipped faces amplify
            # the brute-clip rounding by 1/chord); the sharp invariant
            # is the area round trip
            assert p_rec == pytest.approx(p, abs=1e-8 * float(d.max()))
            a_rt = _face_area(cell_lo, cell_hi, n_hat, p_rec, face)
            assert a_rt == pytest.approx(area, rel=1e-9, abs=1e-14)
            assert sens > 0.0
            n_checked += 1
    assert n_checked > 3000  # the draw actually exercised the inversion


def test_offset_inversion_axis_aligned_plane():
    # exactly axis-aligned in-plane normal: the linear-ramp branch
    cell_lo = np.zeros(3)
    cell_hi = np.array([1.0, 2.0, 3.0])
    n_hat = np.array([0.0, 1.0, 0.0])
    p = 0.75
    face = 0  # x-lo face, in-plane axes (y, z)
    area = _face_area(cell_lo, cell_hi, n_hat, p, face)
    p_rec, sens = _invert_face_offset(cell_lo, cell_hi, n_hat, face, area)
    assert p_rec == pytest.approx(p, abs=1e-12)
    assert sens == pytest.approx(3.0)  # dA/dp = chord length 3.0


# ---------------------------------------------------------------------------
# factor application on hand-built family records
# ---------------------------------------------------------------------------


def _factors_with(recs):
    cf = CurvatureFactors.__new__(CurvatureFactors)
    x = np.array([0.0, 1.0])
    xc = np.array([0.5])
    cf._ax_pos = [(x, xc, xc), (xc, x, xc), (xc, xc, x)]
    cf._fam = {(0, 0, 0): recs}
    return cf


def _rec(n, p, area, kap):
    return {
        "n": np.asarray(n, dtype=float),
        "p": p,
        "area": area,
        "foot": np.zeros(3),
        "fit": True,
        "kap": np.asarray(kap, dtype=float),
    }


def test_scale_linear_factor_and_exact_noop():
    # wall plane x = 0.2, normal +x, curvature along y only
    kap_y = -0.5
    rec = _rec([1.0, 0.0, 0.0], 0.2, 1.0, [0.0, kap_y, 0.0])
    cf = _factors_with([rec])
    pos3 = (np.array([0]), np.array([0]), np.array([0]))
    # Hy sample sits at x = 0.5 -> d1 = 0.3
    s_y = cf.scale(pos3, 1, [np.array([0]), np.array([0]), np.array([0])])
    assert s_y[0] == pytest.approx((1.0 + kap_y * 0.3) ** 2, rel=1e-14)
    # kappa == 0 along z: EXACTLY 1.0 (bit-exact no-op path)
    s_z = cf.scale(pos3, 2, [np.array([0]), np.array([0]), np.array([0])])
    assert s_z[0] == 1.0


def test_scale_clamps_to_zero_beyond_curvature_centre():
    rec = _rec([1.0, 0.0, 0.0], 0.2, 1.0, [0.0, -5.0, 0.0])
    cf = _factors_with([rec])
    pos3 = (np.array([0]), np.array([0]), np.array([0]))
    # d1 = 0.3, kappa*d1 = -1.5 -> c clamped to 0
    s = cf.scale(pos3, 1, [np.array([0]), np.array([0]), np.array([0])])
    assert s[0] == 0.0


def test_scale_mixes_families_by_area():
    flat = _rec([0.0, 0.0, 1.0], 0.0, 3.0, [0.0, 0.0, 0.0])
    curved = _rec([1.0, 0.0, 0.0], 0.2, 1.0, [0.0, -0.5, 0.0])
    cf = _factors_with([flat, curved])
    pos3 = (np.array([0]), np.array([0]), np.array([0]))
    s = cf.scale(pos3, 1, [np.array([0]), np.array([0]), np.array([0])])
    c2 = (1.0 + (-0.5) * 0.3) ** 2
    assert s[0] == pytest.approx((3.0 * 1.0 + 1.0 * c2) / 4.0, rel=1e-14)
