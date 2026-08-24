"""Unit tests for the reference-plane shift (post.deembed).

Synthetic S-matrices with hand-set line parameters — the factor
algebra, both dispersion tiers and the error paths, without a solver
run.  The end-to-end exactness certificate lives in
``tests/integration/test_deembed_line.py`` and
``validation/deembed_uniform_line.py``.
"""

import numpy as np
import pytest

from magnelio.post.deembed import deembed_s_params
from magnelio.post.sparameter_result import SParameterResult

DT = 1.0e-12
R = 0.5
DZ = 1.0e-3
CHANNELS = (("p1", 0), ("p2", 0))
LINE_PARAMS = {("p1", 0): (R, 0.0, None), ("p2", 0): (R, 0.0, None)}
NORMAL_DX = {"p1": DZ, "p2": DZ}


def _s_params(seed: int = 7) -> SParameterResult:
    f = np.linspace(1e9, 40e9, 33)
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((f.size, 2, 2)) + 1j * rng.standard_normal((f.size, 2, 2))
    return SParameterResult(f_axis=f, channels=CHANNELS, excitations=CHANNELS, matrix=matrix)


def _tem_phase_advance(f: np.ndarray, d: float) -> np.ndarray:
    # Discrete TEM chain dispersion: sin(w dt/2) = r sin(beta_hat dz/2).
    w = 2.0 * np.pi * f
    beta_hat = (2.0 / DZ) * np.arcsin(np.sin(w * DT / 2.0) / R)
    return np.exp(1j * beta_hat * d)


def _deembed(s, distances, **overrides):
    kwargs = {
        "dt": DT,
        "port_line_params": LINE_PARAMS,
        "port_normal_dx": NORMAL_DX,
        "port_modes": None,
    }
    kwargs.update(overrides)
    return deembed_s_params(s, distances, **kwargs)


class TestFactorAlgebra:
    def test_zero_distance_is_identity(self):
        s = _s_params()
        de = _deembed(s, {"p1": 0.0, "p2": 0.0})
        np.testing.assert_allclose(de.matrix, s.matrix, rtol=0, atol=0)

    def test_original_untouched(self):
        s = _s_params()
        before = s.matrix.copy()
        _deembed(s, {"p1": 3e-3})
        np.testing.assert_array_equal(s.matrix, before)

    def test_transmission_gets_discrete_phase_advance_once(self):
        s = _s_params()
        d = 3.0 * DZ
        de = _deembed(s, {"p1": d})
        expected = _tem_phase_advance(s.f_axis, d)
        np.testing.assert_allclose(de.S("p2", "p1") / s.S("p2", "p1"), expected, atol=1e-6)
        # The row side of the untouched port is unshifted.
        np.testing.assert_allclose(de.S("p2", "p2") / s.S("p2", "p2"), 1.0, atol=1e-6)

    def test_reflection_gets_factor_twice(self):
        s = _s_params()
        d = 2.5 * DZ
        de = _deembed(s, {"p1": d})
        expected = _tem_phase_advance(s.f_axis, d) ** 2
        np.testing.assert_allclose(de.S("p1", "p1") / s.S("p1", "p1"), expected, atol=1e-6)

    def test_split_shift_matches_single_on_transmission(self):
        s = _s_params()
        L = 6.0 * DZ
        single = _deembed(s, {"p1": L})
        split = _deembed(s, {"p1": L / 2, "p2": L / 2})
        np.testing.assert_allclose(split.S("p2", "p1"), single.S("p2", "p1"), rtol=1e-9)

    def test_negative_distance_inverts(self):
        s = _s_params()
        d = 4.0 * DZ
        forth = _deembed(s, {"p1": d})
        back = _deembed(forth, {"p1": -d})
        np.testing.assert_allclose(back.matrix, s.matrix, rtol=1e-9)

    def test_propagating_magnitude_exactly_invariant(self):
        # On-circle evaluation: |lambda| = 1 exactly in the passband,
        # so the shift is a pure phase there — no offset bias that
        # would grow with d/dz.
        s = _s_params()
        de = _deembed(s, {"p1": 50.0 * DZ, "p2": 50.0 * DZ})
        np.testing.assert_allclose(np.abs(de.matrix), np.abs(s.matrix), rtol=1e-12)

    def test_matches_dtbc_lambda_symbol(self):
        # The on-circle branch logic must agree with the canonical
        # off-circle root of the DTBC machinery to its offset bias.
        from magnelio.ports._modal.dtbc import lambda_symbol
        from magnelio.post.deembed import _chain_lambda_log

        w_dt = np.linspace(1e-3, 2.0, 400)
        for q in (0.0, 0.4):
            z = (1.0 + 1e-8) * np.exp(1j * w_dt)
            expected = np.log(lambda_symbol(z, R, q))
            got = _chain_lambda_log(w_dt, R, q)
            np.testing.assert_allclose(got, expected, atol=1e-6)


class TestDispersionTiers:
    def test_continuum_fallback_uses_mode_gamma(self):
        class _Mode:
            def gamma(self, omega):
                return 1j * omega / 3e8

        s = _s_params()
        d = 2e-3
        de = _deembed(
            s,
            {"p1": d},
            port_line_params=None,
            port_normal_dx=None,
            port_modes={"p1": [_Mode()], "p2": [_Mode()]},
        )
        w = 2.0 * np.pi * s.f_axis
        expected = np.exp(1j * w / 3e8 * d)
        np.testing.assert_allclose(de.S("p2", "p1") / s.S("p2", "p1"), expected, rtol=1e-9)

    def test_evanescent_discrete_factor_grows(self):
        # Klein-Gordon channel below cut-off: real growth exp(+alpha d).
        s = _s_params()
        q = 0.5  # discrete cut-off well above the band start
        lp = {key: (R, q, 50.0) for key in CHANNELS}
        de = _deembed(s, {"p1": 10.0 * DZ}, port_line_params=lp)
        f_c = q / DT / (2.0 * np.pi)
        below = s.f_axis < 0.5 * f_c
        assert below.any()
        ratio = np.abs(de.S("p2", "p1") / s.S("p2", "p1"))[below]
        assert (ratio > 1.0).all()


class TestErrors:
    def test_unknown_port_raises(self):
        with pytest.raises(ValueError, match="unknown port"):
            _deembed(_s_params(), {"nope": 1e-3})

    def test_non_finite_distance_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _deembed(_s_params(), {"p1": float("nan")})

    def test_lumped_channel_raises(self):
        class _LumpedStub:
            def z_modal(self, omega):
                return 50.0 + 0j

        with pytest.raises(ValueError, match="lumped"):
            _deembed(
                _s_params(),
                {"p2": 1e-3},
                port_line_params=None,
                port_normal_dx=None,
                port_modes={"p2": [_LumpedStub()]},
            )

    def test_missing_records_raise(self):
        with pytest.raises(ValueError, match="feed-line dispersion"):
            _deembed(
                _s_params(),
                {"p1": 1e-3},
                port_line_params=None,
                port_normal_dx=None,
                port_modes=None,
            )
