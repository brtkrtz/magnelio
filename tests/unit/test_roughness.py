"""Surface-roughness models (DD-088).

Gates the two K(f) multipliers against their closed forms and physical
limits, the Cannonball parameter set against the published worked
example, and the Material/store plumbing that carries them.
"""

import numpy as np
import pytest

from magnelio import Material
from magnelio.materials import Hammerstad, Huray
from magnelio.materials.roughness import skin_depth
from magnelio.post.wall_loss import surface_resistance

SIGMA_CU = 5.8e7
MU0 = 1.2566370614e-6


# ══════════════════════════════════════════════════════════════════════
# Skin depth
# ══════════════════════════════════════════════════════════════════════


def test_skin_depth_matches_surface_resistance():
    # R_s = 1/(sigma*delta) is the identity tying the two together.
    f = np.array([1e8, 1e9, 1e10])
    d = skin_depth(f, SIGMA_CU)
    assert np.allclose(d, 1.0 / (SIGMA_CU * surface_resistance(f, SIGMA_CU)), rtol=1e-12)
    # Copper at 1 GHz: the textbook 2.09 um.
    assert skin_depth(1e9, SIGMA_CU) == pytest.approx(2.09e-6, rel=2e-2)


# ══════════════════════════════════════════════════════════════════════
# Hammerstad
# ══════════════════════════════════════════════════════════════════════


def test_hammerstad_closed_form():
    rq, f = 1e-6, 5e9
    d = skin_depth(f, SIGMA_CU)
    expected = 1.0 + (2.0 / np.pi) * np.arctan(1.4 * (rq / d) ** 2)
    assert float(Hammerstad(rq).factor(f, SIGMA_CU)) == pytest.approx(
        expected,
        rel=1e-14,
    )


def test_hammerstad_smooth_limit_is_one():
    # Rq -> 0 at fixed f, and f -> 0 at fixed Rq: both leave K == 1.
    f = np.array([1e8, 1e9, 1e10])
    assert np.all(Hammerstad(0.0).factor(f, SIGMA_CU) == 1.0)
    assert Hammerstad(1e-6).factor(1e-6, SIGMA_CU) == pytest.approx(1.0, abs=1e-9)


def test_hammerstad_saturates_at_two():
    # Rq >> delta: arctan -> pi/2, so K -> 2 from below and never past it.
    k = Hammerstad(1e-3).factor(1e10, SIGMA_CU)
    assert k < 2.0
    assert k == pytest.approx(2.0, rel=1e-6)


def test_hammerstad_monotonic_in_frequency():
    f = np.logspace(8, 11, 40)
    k = Hammerstad(1e-6).factor(f, SIGMA_CU)
    assert np.all(np.diff(k) > 0.0)
    assert np.all(k >= 1.0)


# ══════════════════════════════════════════════════════════════════════
# Huray
# ══════════════════════════════════════════════════════════════════════


def test_huray_closed_form():
    # Bracken DesignCon 2012 eq. (5), one sphere class.
    a, cov, f = 0.5e-6, 4.0, 8e9
    d = skin_depth(f, SIGMA_CU)
    expected = 1.0 + 1.5 * cov / (1.0 + d / a + d**2 / (2.0 * a**2))
    assert float(Huray(a, cov).factor(f, SIGMA_CU)) == pytest.approx(
        expected,
        rel=1e-14,
    )


def test_huray_low_frequency_limit_is_base_ratio():
    # delta >> a: the field does not resolve the spheres.
    assert Huray(0.5e-6, 4.0).factor(1.0, SIGMA_CU) == pytest.approx(1.0, abs=1e-9)
    assert Huray(0.5e-6, 4.0, base_ratio=1.3).factor(1.0, SIGMA_CU) == (
        pytest.approx(1.3, abs=1e-9)
    )


def test_huray_high_frequency_limit_follows_the_profile():
    # delta << a: every sphere contributes its full surface.
    #    K -> base_ratio + 1.5*coverage — no saturation at 2.
    cov = 4.0
    k = float(Huray(0.5e-6, cov).factor(1e18, SIGMA_CU))
    assert k == pytest.approx(1.0 + 1.5 * cov, rel=1e-3)
    assert k > 2.0


def test_huray_monotonic_in_frequency():
    f = np.logspace(8, 12, 40)
    k = Huray(0.3e-6, 4.9).factor(f, SIGMA_CU)
    assert np.all(np.diff(k) > 0.0)
    assert np.all(k >= 1.0)


def test_huray_smooth_limit_is_one():
    assert np.all(Huray(0.5e-6, 0.0).factor(np.array([1e9, 1e10]), SIGMA_CU) == 1.0)


# ══════════════════════════════════════════════════════════════════════
# Cannonball parameter set vs the published worked example
# ══════════════════════════════════════════════════════════════════════


def test_cannonball_parameters_vs_published_equations():
    # Cannonball stack (Simonovich; the vendor-implemented form is Polar
    # AP8195, co-signed by the author): 14 spheres (9+4+1) on a square
    # tile of side 6a with the stack height identified with Rz, giving
    # eq. 2  r = Rz/16.73  and  eq. 3  A_flat = (6r)^2.
    for rz in (3.175e-6, 4.443e-6, 1e-6):
        m = Huray.cannonball(rz)
        assert m.radius == pytest.approx(rz / 16.73, rel=1e-12)
        assert 36.0 * m.radius**2 == pytest.approx((6.0 * m.radius) ** 2)

    cov = Huray.cannonball(3.175e-6).coverage
    assert cov == pytest.approx(14.0 * 4.0 * np.pi / 36.0, rel=1e-12)
    # coverage is a pure geometry constant — Rz sets the radius alone.
    assert Huray.cannonball(1e-6).coverage == cov
    assert Huray.cannonball(1e-6).base_ratio == 1.0


def test_cannonball_constant_is_the_closed_form():
    # The published 16.73 is the rounded 4*sqrt(3)*(1+sqrt(2)) — check
    # that the constant we carry is that number and not a typo.
    assert 4.0 * np.sqrt(3.0) * (1.0 + np.sqrt(2.0)) == pytest.approx(
        16.73,
        abs=5e-3,
    )
    assert Huray.cannonball(16.73e-6).radius == pytest.approx(1e-6, rel=1e-12)


def test_cannonball_typical_pcb_factor():
    # Rz = 4.443 um copper at 10 GHz: a well-conditioned regime
    # (delta/a ~ 2.5), K ~ 2.1 — the roughness doubles the conductor
    # loss, which is why the model exists.
    k = float(Huray.cannonball(4.443e-6).factor(10e9, SIGMA_CU))
    assert 1.9 < k < 2.3


# ══════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("bad", [-1e-6, float("inf"), float("nan")])
def test_hammerstad_rejects_bad_height(bad):
    with pytest.raises(ValueError, match="rms_height"):
        Hammerstad(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0, "coverage": 1.0},
        {"radius": -1e-6, "coverage": 1.0},
        {"radius": 1e-6, "coverage": -1.0},
        {"radius": 1e-6, "coverage": 1.0, "base_ratio": -1.0},
    ],
)
def test_huray_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        Huray(**kwargs)


def test_cannonball_rejects_bad_rz():
    with pytest.raises(ValueError, match="rz"):
        Huray.cannonball(0.0)


# ══════════════════════════════════════════════════════════════════════
# Material plumbing
# ══════════════════════════════════════════════════════════════════════


def test_lossy_metal_carries_roughness_and_compares_by_value():
    rough = Hammerstad(1e-6)
    a = Material.lossy_metal("copper", SIGMA_CU, roughness=rough)
    b = Material.lossy_metal("copper", SIGMA_CU, roughness=Hammerstad(1e-6))
    smooth = Material.lossy_metal("copper", SIGMA_CU)
    assert a.roughness == rough
    assert a == b
    assert a != smooth
    assert a != Material.lossy_metal("copper", SIGMA_CU, roughness=Huray(1e-6, 4.0))
    # The field solve is untouched: still plain PEC.
    assert a.is_pec and a.is_lossy_metal


def test_roughness_requires_a_conductivity_to_act_on():
    # A roughness on a dielectric (or on plain PEC) has no R_s to
    # multiply — reject it at construction, not at loss-evaluation time.
    with pytest.raises(ValueError, match="lossy metal"):
        Material(name="fr4", epsilon=(4.3,) * 3, roughness=Hammerstad(1e-6))
    with pytest.raises(ValueError, match="lossy metal"):
        Material(name="pec", is_pec=True, roughness=Hammerstad(1e-6))


def test_surface_resistance_multiplies_by_k():
    f = np.array([1e9, 1e10])
    rough = Huray.cannonball(4.443e-6)
    r_smooth = surface_resistance(f, SIGMA_CU)
    r_rough = surface_resistance(f, SIGMA_CU, roughness=rough)
    assert np.allclose(r_rough / r_smooth, rough.factor(f, SIGMA_CU), rtol=1e-14)
    assert np.all(r_rough > r_smooth)


def test_material_store_roundtrip_is_schema_additive():
    from magnelio.io.project import _material_from_dict, _material_to_dict

    for rough in (
        Hammerstad(1.2e-6),
        Huray(0.3e-6, 4.9, base_ratio=1.1),
        Huray.cannonball(3.175e-6),
    ):
        mat = Material.lossy_metal("cu", SIGMA_CU, mu=1.0, roughness=rough)
        back = _material_from_dict(_material_to_dict(mat))
        assert back == mat
        assert back.roughness == rough

    # Smooth metals write no key at all, and a store written before the
    # roughness channel loads unchanged.
    smooth = Material.lossy_metal("cu", SIGMA_CU)
    d = _material_to_dict(smooth)
    assert "roughness" not in d
    assert _material_from_dict(d) == smooth
