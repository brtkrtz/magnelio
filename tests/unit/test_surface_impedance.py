"""Causal rational surface impedance (SIBC_PLAN.md WP-D2).

Gates the Foster-ladder NNLS fit against the smooth closed form and the
DD-088 roughness targets, the Kramers-Kronig completion against the
case where the exact answer is known (smooth: X = R), passivity of the
coefficients by construction, and the acceptance-loop failure path.
A-priori targets from the internal dossier
investigations/sibc/DERIVATION.md §2/§7.
"""

import numpy as np
import pytest

from magnelio.materials import Huray
from magnelio.materials.surface_impedance import (
    fit_surface_impedance,
    kk_reactance,
    smooth_surface_impedance,
)
from magnelio.post.wall_loss import surface_resistance

SIGMA_CU = 5.8e7
F_LO, F_HI = 1e8, 1e11


# ══════════════════════════════════════════════════════════════════════
# Closed form
# ══════════════════════════════════════════════════════════════════════


def test_smooth_closed_form_matches_surface_resistance():
    # Re Z_s == Im Z_s == R_s ties the new closed form to the DD-082
    # single source of R_s.
    f = np.logspace(8, 11, 20)
    z = smooth_surface_impedance(f, SIGMA_CU)
    r_s = surface_resistance(f, SIGMA_CU)
    assert np.allclose(z.real, r_s, rtol=1e-12)
    assert np.allclose(z.imag, r_s, rtol=1e-12)


def test_smooth_closed_form_mu_scaling():
    # Z ~ sqrt(mu): mu = 4 doubles the impedance.
    f = np.array([1e9, 1e10])
    z1 = smooth_surface_impedance(f, SIGMA_CU, mu=1.0)
    z4 = smooth_surface_impedance(f, SIGMA_CU, mu=4.0)
    assert np.allclose(z4, 2.0 * z1, rtol=1e-12)


# ══════════════════════════════════════════════════════════════════════
# Kramers-Kronig completion
# ══════════════════════════════════════════════════════════════════════


def test_kk_quadrature_against_smooth_closed_form():
    # The one case with an exact answer: the Hilbert transform of
    # R_s ~ sqrt(f) is X = R exactly.  WP-D2 target from the dossier:
    # <= 2e-4 (the WP-D1 probe-grade quadrature stood at 1.3e-3).
    f_eval = np.logspace(np.log10(F_LO), np.log10(F_HI), 25)
    x = kk_reactance(
        f_eval,
        lambda f: surface_resistance(f, SIGMA_CU),
        F_LO,
        F_HI,
    )
    x_exact = surface_resistance(f_eval, SIGMA_CU)
    assert np.max(np.abs(x - x_exact) / x_exact) <= 2e-4


# ══════════════════════════════════════════════════════════════════════
# Smooth fit
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def smooth_fit():
    return fit_surface_impedance(SIGMA_CU, f_lo=F_LO, f_hi=F_HI, tol=1e-3)


def test_smooth_fit_meets_tolerance(smooth_fit):
    # Independent dense-grid verification, not the fit's own bookkeeping.
    f = np.logspace(np.log10(F_LO), np.log10(F_HI), 777)
    z = smooth_fit.impedance(f)
    r_exact = surface_resistance(f, SIGMA_CU)
    assert np.max(np.abs(z.real - r_exact) / r_exact) <= 1e-3
    # The reactance rides along without ever being fitted directly —
    # the causality dividend (dossier §2); same class, small headroom.
    assert np.max(np.abs(z.imag - r_exact) / r_exact) <= 2e-3
    assert smooth_fit.rel_err_re <= 1e-3


def test_fit_coefficients_passive_by_construction(smooth_fit):
    # c0 >= 0, every (b_p, c_p) > 0: the elementary per-branch
    # dissipation identity needs nothing else (dossier §5).
    assert smooth_fit.c0 >= 0.0
    assert len(smooth_fit.branches) >= 4
    for b_p, c_p in smooth_fit.branches:
        assert b_p > 0.0
        assert c_p > 0.0


def test_fit_positive_real_far_beyond_band(smooth_fit):
    # Accuracy is band-limited, passivity is global: Re Z >= 0 over
    # twelve decades.
    f = np.logspace(3, 15, 4000)
    assert np.all(smooth_fit.impedance(f).real >= 0.0)


def test_fit_instantaneous_resistance(smooth_fit):
    # Z(inf) = c0 + sum c_p, approached from below (all branches are
    # high-pass).
    z_hf = smooth_fit.impedance(np.array([1e18]))[0]
    assert z_hf.real == pytest.approx(smooth_fit.r_instantaneous, rel=1e-6)
    assert smooth_fit.r_instantaneous > 0.0


def test_fit_mu_scaling():
    # sqrt(mu) covariance survives the fit machinery end to end.
    fit1 = fit_surface_impedance(SIGMA_CU, f_lo=1e9, f_hi=1e10, tol=1e-3)
    fit4 = fit_surface_impedance(SIGMA_CU, mu=4.0, f_lo=1e9, f_hi=1e10, tol=1e-3)
    f = np.logspace(9, 10, 30)
    assert np.allclose(fit4.impedance(f), 2.0 * fit1.impedance(f), rtol=5e-3)


# ══════════════════════════════════════════════════════════════════════
# Causal roughness fit
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def rough_fit():
    return fit_surface_impedance(
        SIGMA_CU,
        roughness=Huray.cannonball(6e-6),
        f_lo=F_LO,
        f_hi=F_HI,
        tol=1e-3,
    )


def test_rough_fit_reproduces_k_times_rs(rough_fit):
    # The loss-carrying real part must be K(f) * R_s per frequency —
    # frequency-SHAPED (K spans ~1.04 -> 5.2 on this band), not scaled.
    huray = Huray.cannonball(6e-6)
    f = np.logspace(np.log10(F_LO), np.log10(F_HI), 777)
    r_target = surface_resistance(f, SIGMA_CU, roughness=huray)
    z = rough_fit.impedance(f)
    assert np.max(np.abs(z.real - r_target) / r_target) <= 1e-3
    # K really varied over the band (guards the fixture itself)
    k = huray.factor(f, SIGMA_CU)
    assert k[0] < 1.1 and k[-1] > 4.0


def test_rough_reactance_leads_the_resistance(rough_fit):
    # Physics of the completed reactance: the roughness excess DR >= 0
    # adds inductive reactance everywhere (X > R_smooth, including the
    # inductive tail BELOW the transition), and where K(f) rises the
    # local power law is steeper than sqrt, so the reactance LEADS the
    # rough resistance there (measured X/R_rough up to ~1.5 on this
    # band) — a Hilbert pair R ~ w^p has X/R = tan(p pi/2), > 1 for
    # p > 1/2.  X must also stay monotone on a monotone target.
    f = np.logspace(np.log10(F_LO), np.log10(F_HI), 100)
    x = rough_fit.impedance(f).imag
    r_smooth = surface_resistance(f, SIGMA_CU)
    r_rough = surface_resistance(f, SIGMA_CU, roughness=Huray.cannonball(6e-6))
    assert np.all(x > r_smooth)
    assert np.max(x / r_rough) > 1.2
    assert np.all(np.diff(x) > 0.0)


def test_rough_smooth_limit_anchor():
    # Vanishing roughness must reproduce the smooth fit within the
    # combined fit tolerances.
    tiny = fit_surface_impedance(
        SIGMA_CU,
        roughness=Huray.cannonball(1e-9),
        f_lo=1e9,
        f_hi=1e10,
        tol=1e-3,
    )
    smooth = fit_surface_impedance(SIGMA_CU, f_lo=1e9, f_hi=1e10, tol=1e-3)
    f = np.logspace(9, 10, 50)
    assert np.allclose(tiny.impedance(f), smooth.impedance(f), rtol=5e-3)


# ══════════════════════════════════════════════════════════════════════
# Acceptance loop and dataclass plumbing
# ══════════════════════════════════════════════════════════════════════


def test_unreachable_tolerance_raises():
    with pytest.raises(ValueError, match="not reached"):
        fit_surface_impedance(SIGMA_CU, f_lo=F_LO, f_hi=F_HI, tol=1e-9, max_branches=6)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="sigma"):
        fit_surface_impedance(-1.0)
    with pytest.raises(ValueError, match="f_lo"):
        fit_surface_impedance(SIGMA_CU, f_lo=1e10, f_hi=1e9)


def test_fit_equality_and_hash():
    # Frozen dataclass identity is physical (band, material, ladder),
    # not bookkeeping: the achieved-error fields are compare=False.
    fit_a = fit_surface_impedance(SIGMA_CU, f_lo=1e9, f_hi=1e10, tol=1e-3)
    fit_b = fit_surface_impedance(SIGMA_CU, f_lo=1e9, f_hi=1e10, tol=1e-3)
    assert fit_a == fit_b
    assert hash(fit_a) == hash(fit_b)
    fit_c = fit_surface_impedance(2.0e7, f_lo=1e9, f_hi=1e10, tol=1e-3)
    assert fit_a != fit_c


def test_branch_count_grows_with_tolerance():
    # The acceptance loop actually adapts: a coarse tolerance needs
    # fewer branches than a tight one.
    coarse = fit_surface_impedance(SIGMA_CU, f_lo=F_LO, f_hi=F_HI, tol=1e-2)
    tight = fit_surface_impedance(SIGMA_CU, f_lo=F_LO, f_hi=F_HI, tol=5e-4)
    assert len(coarse.branches) < len(tight.branches)
    assert coarse.rel_err_re <= 1e-2
    assert tight.rel_err_re <= 5e-4
