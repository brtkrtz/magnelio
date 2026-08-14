"""DD-086 gates: vector fitting of tabulated eps(f) onto the DD-083
pole-residue form.

The fit enforces pole stability (VF flipping); the DispersionModel
constructor is the mandatory PASSIVITY acceptance filter — an accurate
but active fit is rejected, never silently shipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.materials import DispersionModel
from magnelio.materials.vector_fit import vector_fit


def _table_err(model: DispersionModel, f: np.ndarray, ref: np.ndarray) -> float:
    fit = model.evaluate(2.0 * np.pi * f)
    return float(np.max(np.abs(fit - ref)) / np.max(np.abs(ref)))


# ── round trips on synthetic tables ──────────────────────────────────


def test_debye_round_trip_machine_precision():
    """A single-relaxation table is recovered essentially exactly
    (measured 3e-16) with the minimal order."""
    f = np.logspace(8, 11, 60)
    ref = DispersionModel.debye(eps_inf=2.1, delta_eps=2.2, tau=1e-10)
    eps = ref.evaluate(2.0 * np.pi * f)

    m = DispersionModel.from_table(f, eps)
    assert len(m.poles) == 1
    assert _table_err(m, f, eps) < 1e-12
    assert m.eps_inf == pytest.approx(2.1, rel=1e-9)


def test_lorentz_round_trip():
    """A resonant table lands on one conjugate pair (measured 2.8e-6;
    the residual is the auto-order acceptance, not the pair position)."""
    f = np.logspace(9, 11.3, 80)
    ref = DispersionModel.lorentz(
        eps_inf=2.0,
        delta_eps=1.5,
        omega0=2 * np.pi * 20e9,
        delta=2 * np.pi * 2e9,
    )
    eps = ref.evaluate(2.0 * np.pi * f)

    m = DispersionModel.from_table(f, eps)
    assert _table_err(m, f, eps) < 1e-3
    # off-grid: the fit tracks the underlying model over the band
    f_dense = np.logspace(9.05, 11.25, 500)
    err = np.max(
        np.abs(m.evaluate(2 * np.pi * f_dense) - ref.evaluate(2 * np.pi * f_dense))
    ) / np.max(np.abs(eps))
    assert err < 1e-3


def test_djordjevic_sarkar_round_trip():
    """The wideband DS relaxation continuum (measured: 13 poles at
    9.1e-4 on-grid, 4.2e-4 off-grid) — the smooth-relaxation start."""
    ref = DispersionModel.djordjevic_sarkar(
        eps_r=4.3,
        tan_delta=0.02,
        f_ref=1e9,
        f1=1e6,
        f2=1e11,
    )
    f = np.logspace(7, 10.5, 100)
    eps = ref.evaluate(2.0 * np.pi * f)

    m = DispersionModel.from_table(f, eps)
    assert _table_err(m, f, eps) < 1e-3
    f_dense = np.logspace(7.2, 10.3, 500)
    err = np.max(
        np.abs(m.evaluate(2 * np.pi * f_dense) - ref.evaluate(2 * np.pi * f_dense))
    ) / np.max(np.abs(eps))
    assert err < 1e-3


def test_eps_prime_tan_delta_input_form():
    """(eps', tan_delta) pairs are accepted and give the same model
    class as the complex form."""
    ref = DispersionModel.djordjevic_sarkar(
        eps_r=4.3,
        tan_delta=0.02,
        f_ref=1e9,
        f1=1e6,
        f2=1e11,
    )
    f = np.logspace(7, 10.5, 100)
    eps = ref.evaluate(2.0 * np.pi * f)

    m = DispersionModel.from_table(f, (eps.real, -eps.imag / eps.real))
    assert _table_err(m, f, eps) < 1e-3


# ── noise robustness ─────────────────────────────────────────────────


def test_noisy_table_recovers_underlying_model():
    """1 % multiplicative noise: with the tolerance set above the noise
    max-norm the automatic order finds the SMOOTH low-order model
    through the noise (measured: 1 pole, 8e-4 vs the clean model)."""
    f = np.logspace(8, 11, 60)
    ref = DispersionModel.debye(eps_inf=2.1, delta_eps=2.2, tau=1e-10)
    eps = ref.evaluate(2.0 * np.pi * f)
    rng = np.random.default_rng(3)
    noisy = eps.real * (1 + 0.01 * rng.standard_normal(f.size)) + 1j * eps.imag * (
        1 + 0.01 * rng.standard_normal(f.size)
    )

    m = DispersionModel.from_table(f, noisy, tol=0.04)
    assert _table_err(m, f, eps) < 0.01  # vs the CLEAN model


# ── passivity / rejection paths ──────────────────────────────────────


def test_active_table_rejected_with_clear_message():
    """Gain data (eps'' < 0) is rejected up front with the offending
    frequency in the message."""
    f = np.logspace(8, 11, 60)
    eps = DispersionModel.debye(2.1, 2.2, tau=1e-10).evaluate(2 * np.pi * f)
    with pytest.raises(ValueError, match="non-passive table"):
        DispersionModel.from_table(f, np.conj(eps))


def test_overfit_explicit_order_hits_passivity_filter():
    """An explicit order that overfits noise into an active pole set is
    rejected by the DD-083 constructor (the acceptance filter), with the
    offending frequency — never silently shipped."""
    f = np.logspace(8, 11, 60)
    eps = DispersionModel.debye(2.1, 2.2, tau=1e-10).evaluate(2 * np.pi * f)
    rng = np.random.default_rng(3)
    noisy = eps.real * (1 + 0.01 * rng.standard_normal(f.size)) + 1j * eps.imag * (
        1 + 0.01 * rng.standard_normal(f.size)
    )
    with pytest.raises(ValueError, match="non-passive"):
        DispersionModel.from_table(f, noisy, n_poles=6)


def test_auto_order_cap_error_is_actionable():
    """The auto-order cap failure names tolerance, best error and the
    explicit-n_poles escape hatch."""
    f = np.logspace(8, 11, 60)
    eps = DispersionModel.debye(2.1, 2.2, tau=1e-10).evaluate(2 * np.pi * f)
    rng = np.random.default_rng(7)
    noisy = eps * (1 + 0.05 * rng.standard_normal(f.size))
    with pytest.raises(ValueError, match="explicit n_poles"):
        DispersionModel.from_table(f, noisy, tol=1e-4, max_poles=4)


def test_input_validation():
    f = np.logspace(8, 11, 60)
    eps = np.full(60, 2.0 + 0j)
    with pytest.raises(ValueError, match="ascending"):
        DispersionModel.from_table(f[::-1], eps)
    with pytest.raises(ValueError, match="does not match"):
        DispersionModel.from_table(f, eps[:-1])
    with pytest.raises(ValueError, match=">= 4"):
        DispersionModel.from_table(f[:3], eps[:3])


# ── fit-core properties ──────────────────────────────────────────────


def test_vector_fit_flips_poles_stable():
    """The relocated pole set is always left-half-plane (the VF
    stability rule), for both start sets."""
    f = np.logspace(8, 11, 50)
    eps = DispersionModel.debye(2.0, 3.0, tau=5e-11).evaluate(2 * np.pi * f)
    for start in ("real", "complex"):
        _, poles, _ = vector_fit(2 * np.pi * f, eps, 6, start=start)
        assert all(a.real <= 0.0 for a, r in poles), start


def test_real_poles_carry_real_residues():
    """The real-coefficient formulation gives real residues on real
    poles by construction (a DD-083 constructor requirement)."""
    f = np.logspace(7, 10.5, 100)
    ref = DispersionModel.djordjevic_sarkar(
        eps_r=4.3,
        tan_delta=0.02,
        f_ref=1e9,
        f1=1e6,
        f2=1e11,
    )
    eps = ref.evaluate(2.0 * np.pi * f)
    _, poles, _ = vector_fit(2 * np.pi * f, eps, 12, start="real")
    for a, r in poles:
        if a.imag == 0.0:
            assert r.imag == 0.0


def test_f_band_defaults_to_table_span():
    f = np.logspace(8, 11, 60)
    eps = DispersionModel.debye(2.1, 2.2, tau=1e-10).evaluate(2 * np.pi * f)
    m = DispersionModel.from_table(f, eps)
    assert m.f_band == (pytest.approx(f[0]), pytest.approx(f[-1]))
