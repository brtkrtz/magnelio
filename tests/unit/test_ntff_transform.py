"""DD-173: the NTFF transform against the analytic Hertzian dipole.

The exact near fields of a z-directed current element are sampled on a
closed synthetic box and pushed through the surface-equivalence
transform.  Gates: the sin²θ pattern with D = 1.5, the analytic
radiated power, and — the highest-risk item — the complex phase of
E_theta, which pins the e^{-jωt}/e^{+jωt} conjugation convention.
Image theory is pinned by exactness: for a mirror-symmetric field the
image expansion of a half (quarter) box reproduces the full-box
transform to machine precision, because the mirrored patches ARE the
discarded samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.constants import C0, EPS0, MU0
from magnelio.post.far_field import (
    FarFieldResult,
    ImagePlane,
    SurfacePatchSet,
    ntff_transform,
)

ETA0 = float(np.sqrt(MU0 / EPS0))
F0 = 1e9
K0 = 2.0 * np.pi * F0 / C0
IDL = 1.0e-3  # current-moment I*dl [A*m]


def hertzian_fields(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact fields of a z-directed Hertzian dipole at the origin.

    Balanis 4-2 in the e^{+jωt} convention, conjugated to the
    library's e^{-jωt} phasors.
    """
    x, y, z = pts.T
    r = np.sqrt(x**2 + y**2 + z**2)
    rho = np.sqrt(x**2 + y**2)
    ct, st = z / r, rho / r
    phi = np.arctan2(y, x)
    kr = K0 * r
    ex = np.exp(-1j * kr)
    E_r = ETA0 * IDL * ct / (2.0 * np.pi * r**2) * (1.0 + 1.0 / (1j * kr)) * ex
    E_t = 1j * ETA0 * K0 * IDL * st / (4.0 * np.pi * r) * (1.0 + 1.0 / (1j * kr) - 1.0 / kr**2) * ex
    H_p = 1j * K0 * IDL * st / (4.0 * np.pi * r) * (1.0 + 1.0 / (1j * kr)) * ex
    r_hat = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
    t_hat = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
    p_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    E = E_r[:, None] * r_hat + E_t[:, None] * t_hat
    H = H_p[:, None] * p_hat
    return np.conj(E), np.conj(H)


_FACES = {
    "xmin": (0, -1.0),
    "xmax": (0, 1.0),
    "ymin": (1, -1.0),
    "ymax": (1, 1.0),
    "zmin": (2, -1.0),
    "zmax": (2, 1.0),
}


def box_patches(lo, hi, n, skip=()) -> list[SurfacePatchSet]:
    """Midpoint-sampled analytic patches on the faces of a box.

    ``n`` is the patch count per axis — a scalar, or a per-axis triple
    so a half box can keep the full box's patch grid (which makes the
    image-expansion comparisons exact rather than discretisation-close).
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    counts = (n, n, n) if np.isscalar(n) else tuple(n)
    sets = []
    for face, (axis, sign) in _FACES.items():
        if face in skip:
            continue
        t1, t2 = [a for a in range(3) if a != axis]
        n1, n2 = counts[t1], counts[t2]
        c1 = np.linspace(lo[t1], hi[t1], n1 + 1)
        c2 = np.linspace(lo[t2], hi[t2], n2 + 1)
        m1 = 0.5 * (c1[:-1] + c1[1:])
        m2 = 0.5 * (c2[:-1] + c2[1:])
        g1, g2 = np.meshgrid(m1, m2, indexing="ij")
        n_p = n1 * n2
        centers = np.zeros((n_p, 3))
        centers[:, t1] = g1.ravel()
        centers[:, t2] = g2.ravel()
        centers[:, axis] = hi[axis] if sign > 0 else lo[axis]
        normals = np.zeros((n_p, 3))
        normals[:, axis] = sign
        area = (c1[1] - c1[0]) * (c2[1] - c2[0])
        E, H = hertzian_fields(centers)
        sets.append(
            SurfacePatchSet(
                centers=centers,
                normals=normals,
                areas=np.full(n_p, area),
                E=E,
                H=H,
            )
        )
    return sets


L = 0.08  # box half-extent [m] (~0.27 lambda)
N_PATCH = 32
P_ANALYTIC = ETA0 * K0**2 * IDL**2 / (12.0 * np.pi)


@pytest.fixture(scope="module")
def free_dipole() -> FarFieldResult:
    return ntff_transform(box_patches((-L, -L, -L), (L, L, L), N_PATCH), [], F0)


class TestFreeDipole:
    def test_directivity_and_pattern(self, free_dipole):
        res = free_dipole
        d = res.directivity
        i_eq = np.argmin(np.abs(res.theta - np.pi / 2))
        assert d[i_eq, 0] == pytest.approx(1.5, rel=5e-3)
        # sin^2 theta shape, phi-independent.
        ref = 1.5 * np.sin(res.theta) ** 2
        assert np.max(np.abs(d - ref[:, None])) < 1.5 * 5e-3
        # phi-independence holds to the surface-discretisation level.
        assert np.max(np.abs(d - d[:, :1])) < 1e-3 * 1.5

    def test_radiated_power(self, free_dipole):
        assert free_dipole.P_rad == pytest.approx(P_ANALYTIC, rel=5e-3)

    def test_phase_convention_is_pinned(self, free_dipole):
        # Library convention (e^{-jωt}): A_theta = -j η k I dl sinθ/(4π).
        res = free_dipole
        i_eq = np.argmin(np.abs(res.theta - np.pi / 2))
        expected = -1j * ETA0 * K0 * IDL / (4.0 * np.pi)
        assert res.E_theta[i_eq, 0] == pytest.approx(expected, rel=1e-2)

    def test_e_phi_vanishes(self, free_dipole):
        res = free_dipole
        scale = np.max(np.abs(res.E_theta))
        assert np.max(np.abs(res.E_phi)) < 1e-3 * scale


class TestGroundPlane:
    """Vertical dipole on an infinite electric ground plane."""

    def test_monopole_limit(self):
        res = ntff_transform(
            box_patches((-L, -L, 0.0), (L, L, L), N_PATCH, skip=("zmin",)),
            [ImagePlane(axis=2, position=0.0, kind="PEC", at_low=True, physical_halfspace=True)],
            F0,
        )
        # Upper-hemisphere pattern identical to the free dipole, lower
        # hemisphere masked; half the free power radiates.
        assert res.physical_mask is not None
        below = res.theta > np.pi / 2 + 1e-9
        assert not res.physical_mask[below, :].any()
        assert res.U[below, :].max() == 0.0
        assert res.P_rad == pytest.approx(0.5 * P_ANALYTIC, rel=1e-2)
        i_eq = np.argmin(np.abs(res.theta - np.pi / 2))
        assert res.directivity[i_eq, 0] == pytest.approx(3.0, rel=2e-2)


class TestSymmetryComposition:
    """Image expansion == the discarded samples, to machine precision."""

    def test_half_box_pec_symmetry(self, free_dipole):
        res = ntff_transform(
            box_patches((-L, -L, 0.0), (L, L, L), (N_PATCH, N_PATCH, N_PATCH // 2), skip=("zmin",)),
            [ImagePlane(axis=2, position=0.0, kind="PEC", at_low=True, physical_halfspace=False)],
            F0,
        )
        scale = np.max(np.abs(free_dipole.E_theta))
        assert np.max(np.abs(res.E_theta - free_dipole.E_theta)) < 1e-10 * scale
        assert res.physical_mask is None

    def test_quarter_box_pec_plus_pmc(self, free_dipole):
        res = ntff_transform(
            box_patches(
                (0.0, -L, 0.0),
                (L, L, L),
                (N_PATCH // 2, N_PATCH, N_PATCH // 2),
                skip=("zmin", "xmin"),
            ),
            [
                ImagePlane(axis=2, position=0.0, kind="PEC", at_low=True, physical_halfspace=False),
                ImagePlane(axis=0, position=0.0, kind="PMC", at_low=True, physical_halfspace=False),
            ],
            F0,
        )
        scale = np.max(np.abs(free_dipole.E_theta))
        assert np.max(np.abs(res.E_theta - free_dipole.E_theta)) < 1e-10 * scale


class TestResultInterface:
    def test_cut_folds_the_full_polar_trace(self, free_dipole):
        angles, trace = free_dipole.cut(plane="phi", angle=0.0, quantity="directivity")
        assert angles[0] == 0.0 and angles[-1] == pytest.approx(2.0 * np.pi)
        assert trace.shape == angles.shape
        # Forward and folded-back halves agree for the phi-symmetric dipole.
        n = free_dipole.theta.size
        assert np.allclose(trace[:n], 1.5 * np.sin(free_dipole.theta) ** 2, atol=1e-2)

    def test_gain_requires_accepted_power(self, free_dipole):
        with pytest.raises(ValueError, match="accepted"):
            _ = free_dipole.gain
        free_dipole.accepted_power = free_dipole.P_rad
        try:
            eff = free_dipole.radiation_efficiency
            assert eff == pytest.approx(1.0)
            i_eq = np.argmin(np.abs(free_dipole.theta - np.pi / 2))
            assert free_dipole.gain[i_eq, 0] == pytest.approx(1.5, rel=5e-3)
        finally:
            free_dipole.accepted_power = None
