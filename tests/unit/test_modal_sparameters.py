"""Tests for magnelio.post.modal_sparameters.compute_s_parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.ports._modal.mode import Mode, ModeType
from magnelio.post import compute_s_parameters
from magnelio.signals.signal_1d import Signal1D

# ----------------------------------------------------------------------
# Lightweight Mode fixtures (no FIT mesh involved — only z_modal is used
# by compute_s_parameters, so the field_evaluator is a stub).
# ----------------------------------------------------------------------


def _stub_field(u, v):
    z = np.zeros_like(u)
    return z, z, z, z


def _tem_mode(z_line: float = 50.0) -> Mode:
    return Mode(
        name="TEM",
        mode_type=ModeType.TEM,
        omega_c=0.0,
        epsilon_r=1.0,
        field_evaluator=_stub_field,
        z_line=z_line,
    )


def _te10_mode(f_c: float = 6.557e9) -> Mode:
    """WR-90-like TE10 with f_cutoff ≈ 6.557 GHz (z_line=None → uses z_wave)."""
    return Mode(
        name="TE10",
        mode_type=ModeType.TE,
        omega_c=2.0 * math.pi * f_c,
        epsilon_r=1.0,
        field_evaluator=_stub_field,
        z_line=None,
    )


def _gauss_signal(
    N: int, dt: float, t_offset: float, t0: float, sigma: float, label: str = ""
) -> Signal1D:
    """Gaussian pulse centred at ``t0``, sampled at ``k·dt + t_offset``."""
    t_phys = np.arange(N) * dt + t_offset
    values = np.exp(-((t_phys - t0) ** 2) / (2.0 * sigma**2))
    return Signal1D(t=np.arange(N) * dt, values=values, dt=dt, label=label)


def _empty_reference(N: int, dt: float) -> Signal1D:
    """Reference excitation Gaussian — concrete shape unimportant for ratios."""
    t = np.arange(N) * dt
    sigma = 30 * dt
    values = np.exp(-((t - 50 * dt) ** 2) / (2.0 * sigma**2))
    return Signal1D(t=t, values=values, dt=dt, label="ref")


# ----------------------------------------------------------------------
# Limit-case behaviours
# ----------------------------------------------------------------------


class TestLimitCases:
    """Open / short / matched-load — closed-form S targets."""

    def test_open_reflection_gives_S_plus_one(self):
        """V = pulse, I = 0  →  a = b = V/(2√Z)  →  S = +1."""
        N, dt = 256, 5e-12
        t = np.arange(N) * dt
        V_vals = np.exp(-((t - 50 * dt) ** 2) / (2.0 * (20 * dt) ** 2))
        V_sig = Signal1D(t=t, values=V_vals, dt=dt, label="V")
        I_sig = Signal1D(t=t, values=np.zeros(N), dt=dt, label="I")
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode()]}
        f_axis = np.linspace(1e9, 20e9, 21)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
        )
        S_pp = S[("p", 0)]
        np.testing.assert_allclose(S_pp.real, 1.0, atol=1e-9)
        np.testing.assert_allclose(S_pp.imag, 0.0, atol=1e-9)

    def test_short_reflection_gives_S_minus_one(self):
        """V = 0, I = pulse  →  a = -b  →  S = -1."""
        N, dt = 256, 5e-12
        t = np.arange(N) * dt
        I_vals = np.exp(-((t - 50 * dt) ** 2) / (2.0 * (20 * dt) ** 2))
        V_sig = Signal1D(t=t, values=np.zeros(N), dt=dt, label="V")
        I_sig = Signal1D(t=t, values=I_vals, dt=dt, label="I")
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode()]}
        f_axis = np.linspace(1e9, 20e9, 21)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=I_sig,
            f_axis=f_axis,
        )
        S_pp = S[("p", 0)]
        # Short-circuit: |S|=1, phase = π (S ≈ -1).  Magnitude is the
        # tight diagnostic; phase has been verified to land at π.
        np.testing.assert_allclose(np.abs(S_pp), 1.0, atol=1e-9)
        np.testing.assert_allclose(S_pp.real, -1.0, atol=1e-9)

    def test_matched_outgoing_wave_gives_S_zero(self):
        """V = Z·I (in phase, with Yee stagger pre-applied)  →  b = 0  →  S = 0.

        Construct a Gaussian pulse ``f(t)``.  V samples it at integer
        time ``(k+1)·dt`` (post-E-update); I samples it at half-integer
        ``(k+1/2)·dt`` (pre-H-update), then divides by Z so V/I = Z
        physically.  The recorder convention drops the half-step offset
        and the post-processor must re-introduce it via ``exp(+jω·dt/2)``
        on the I spectrum.

        Frequency band restricted to ``f ≤ 5 GHz`` (sigma_t = 80 ps gives
        sigma_f ≈ 2 GHz).  Beyond that the Gaussian's analytical form
        runs into discretisation artefacts that swamp the cancellation
        residual — that is a sampling problem, not a correction problem.
        """
        N, dt = 512, 2e-12
        Z = 75.0
        t0 = 200 * dt
        sigma = 40 * dt

        V_sig = _gauss_signal(N, dt, t_offset=dt, t0=t0, sigma=sigma)
        I_vals = np.exp(-((np.arange(N) * dt + 0.5 * dt - t0) ** 2) / (2.0 * sigma**2)) / Z
        I_sig = Signal1D(t=np.arange(N) * dt, values=I_vals, dt=dt)
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode(z_line=Z)]}
        f_axis = np.linspace(0.5e9, 5e9, 10)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
        )
        S_pp = S[("p", 0)]
        # Pulse-band cancellation: |S| ≤ 5e-3 in the well-sampled band.
        assert np.max(np.abs(S_pp)) < 5e-3, (
            f"Forward-wave cancellation residual too large: "
            f"max |S| = {np.max(np.abs(S_pp)):.2e} at f = "
            f"{f_axis[np.argmax(np.abs(S_pp))] / 1e9:.2f} GHz."
        )

    def test_matched_outgoing_wave_without_stagger_correction_does_not_cancel(self):
        """Diagnostic: skipping the dt/2 phase correction breaks cancellation.

        Same setup as the matched-wave test, but compute the bad
        ``a, b`` manually without phase-correcting I.  At the same band
        where the corrected version achieves ``|S| < 5e-3`` the
        un-corrected version must show a clearly larger residual,
        proving the correction is what yields the cancellation.
        """
        N, dt = 512, 2e-12
        Z = 75.0
        t0 = 200 * dt
        sigma = 40 * dt
        V_sig = _gauss_signal(N, dt, t_offset=dt, t0=t0, sigma=sigma)
        I_vals = np.exp(-((np.arange(N) * dt + 0.5 * dt - t0) ** 2) / (2.0 * sigma**2)) / Z
        I_sig = Signal1D(t=np.arange(N) * dt, values=I_vals, dt=dt)
        # Compare at f = 5 GHz, where the in-band corrected residual is
        # well below 1% and the un-corrected residual is well above it.
        f_target = np.array([5e9])
        V_f = V_sig.at_frequencies(f_target)
        I_f_uncorrected = I_sig.at_frequencies(f_target)
        a_bad = (V_f / math.sqrt(Z) + math.sqrt(Z) * I_f_uncorrected) / 2.0
        b_bad = (V_f / math.sqrt(Z) - math.sqrt(Z) * I_f_uncorrected) / 2.0
        S_bad = abs(b_bad / a_bad)[0]
        assert S_bad > 0.01, (
            f"Sanity: without correction, |S| should exceed 1% at 5 GHz "
            f"with dt=2 ps; got {S_bad:.2e}.  If this fails, the test "
            f"setup is degenerate and the comparison test loses meaning."
        )


# ----------------------------------------------------------------------
# Spatial half-cell de-stagger (port_normal_dx)
# ----------------------------------------------------------------------


class TestSpatialDeStagger:
    """V at z'=0 vs I at z'=normal_dx/2 — the two-plane decomposition.

    A forward TEM wave reaches the I sampling plane (half a normal
    cell inside the domain) delayed by ``d/(2·v_p)``.  Without the
    spatial correction the naive decomposition leaks
    ``≈ sin(β·d/2)`` of the forward wave into ``b``; with
    ``port_normal_dx`` given, the two-plane system cancels it.
    """

    N = 512
    DT = 2e-12
    Z = 75.0
    D_NORMAL = 1.0e-3  # boundary-cell size along the normal [m]
    C0 = 299_792_458.0  # eps_r = 1 in _tem_mode → v_p = c0

    def _forward_wave_signals(self):
        t0 = 200 * self.DT
        sigma = 40 * self.DT
        V_sig = _gauss_signal(self.N, self.DT, t_offset=self.DT, t0=t0, sigma=sigma)
        # I: Yee half-step later in time AND half a cell inside, i.e.
        # delayed by the one-way travel time d/(2·v_p).
        t_I = np.arange(self.N) * self.DT + 0.5 * self.DT - self.D_NORMAL / (2.0 * self.C0)
        I_vals = np.exp(-((t_I - t0) ** 2) / (2.0 * sigma**2)) / self.Z
        I_sig = Signal1D(t=np.arange(self.N) * self.DT, values=I_vals, dt=self.DT)
        return V_sig, I_sig

    def test_forward_wave_cancels_with_port_normal_dx(self):
        V_sig, I_sig = self._forward_wave_signals()
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode(z_line=self.Z)]}
        f_axis = np.linspace(0.5e9, 5e9, 10)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
            port_normal_dx={"p": self.D_NORMAL},
        )
        S_pp = S[("p", 0)]
        assert np.max(np.abs(S_pp)) < 5e-3, (
            f"De-staggered forward-wave residual too large: "
            f"max |S| = {np.max(np.abs(S_pp)):.2e} at f = "
            f"{f_axis[np.argmax(np.abs(S_pp))] / 1e9:.2f} GHz."
        )

    def test_forward_wave_without_spatial_correction_leaks(self):
        """Sanity anchor: same signals, no port_normal_dx → |S| ≈ β·d/4."""
        V_sig, I_sig = self._forward_wave_signals()
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode(z_line=self.Z)]}
        f_axis = np.array([5e9])

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
        )
        leak = float(np.abs(S[("p", 0)][0]))
        beta = 2.0 * math.pi * 5e9 / self.C0
        expected = beta * self.D_NORMAL / 4.0
        assert leak > 0.01, (
            f"Sanity: without spatial correction the leak at 5 GHz "
            f"should exceed 1% for d = 1 mm; got {leak:.2e}.  If this "
            f"fails, the setup is degenerate."
        )
        assert abs(leak - expected) < 0.3 * expected, (
            f"Leak magnitude {leak:.3e} deviates from the β·d/4 model "
            f"{expected:.3e} by more than 30 % — sampling artefact?"
        )


class TestDiscreteDeStagger:
    """Exact discrete travelling wave — port_line_params cancels it exactly.

    Synthesises V/I samples of the exact leapfrog TEM chain wave
    (discrete dispersion ``sin(ω·dt/2) = r·sin(β̂·dz/2)``, V/I
    magnitude exactly ``Z`` at all frequencies) on exact DFT bins.
    With the discrete half-cell factor the b-channel cancels to float
    noise; the continuum de-stagger leaves the grid-dispersion leak
    ``≈ (β·dz/2)³·(1−r²)/6``.
    """

    N = 4096
    DT = 2e-12
    Z = 50.0
    R_CHAIN = 0.55
    C0 = 299_792_458.0
    BINS = (20, 60, 120)

    def _discrete_wave_signals(self):
        n = np.arange(self.N)
        V_vals = np.zeros(self.N)
        I_vals = np.zeros(self.N)
        for m_bin in self.BINS:
            w_dt = 2.0 * math.pi * m_bin / self.N
            beta_dz_half = math.asin(math.sin(w_dt / 2.0) / self.R_CHAIN)
            # V: e at t^{n+1};  I: h at t^{n+1/2}, half a cell inside
            # (phase retarded by beta_hat*dz/2 for the inward wave).
            V_vals += np.cos(w_dt * (n + 1.0))
            I_vals += np.cos(w_dt * (n + 0.5) - beta_dz_half) / self.Z
        t = n * self.DT
        return (
            Signal1D(t=t, values=V_vals, dt=self.DT),
            Signal1D(t=t, values=I_vals, dt=self.DT),
        )

    def _s_at_bins(self, port_line_params):
        V_sig, I_sig = self._discrete_wave_signals()
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode(z_line=self.Z)]}
        f_axis = np.array(
            [m / (self.N * self.DT) for m in self.BINS],
            dtype=float,
        )
        dz = self.C0 * self.DT / self.R_CHAIN
        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
            port_normal_dx={"p": dz},
            port_line_params=port_line_params,
        )
        return np.abs(S[("p", 0)])

    def test_discrete_factor_cancels_to_float_noise(self):
        # Residual bounded by the deliberate contour offset of the
        # symbol evaluation (1e-8 off the unit circle → ~ -166 dB),
        # far below the -100 dB acceptance line.
        s_abs = self._s_at_bins({("p", 0): (self.R_CHAIN, 0.0)})
        assert np.max(s_abs) < 1e-7, f"discrete de-stagger residual {np.max(s_abs):.2e}"

    def test_continuum_factor_leaves_dispersion_leak(self):
        """Sanity anchor: same wave, continuum θ → leak ≈ (β̂dz/2)³(1−r²)/6."""
        s_abs = self._s_at_bins(None)
        w_dt = 2.0 * math.pi * self.BINS[-1] / self.N
        theta_half = math.asin(math.sin(w_dt / 2.0) / self.R_CHAIN)
        expected = theta_half**3 / 6.0 * (1.0 - self.R_CHAIN**2)
        leak = float(s_abs[-1])
        assert leak > 1e-6
        assert abs(leak - expected) < 0.5 * expected


# ----------------------------------------------------------------------
# Multi-port / multi-mode dispatch
# ----------------------------------------------------------------------


class TestRouting:
    def test_multi_port_dict_keys_match_recorder_keys(self):
        N, dt = 128, 1e-11
        t = np.arange(N) * dt
        V = Signal1D(t=t, values=np.zeros(N), dt=dt)
        I = Signal1D(t=t, values=np.zeros(N), dt=dt)
        # Non-trivial excited channel so a ≠ 0.
        V_exc = Signal1D(t=t, values=np.exp(-((t - 30 * dt) ** 2) / (10 * dt) ** 2), dt=dt)
        recorder = {
            ("p1", 0): (V_exc, I),
            ("p1", 1): (V, I),
            ("p2", 0): (V, I),
            ("p2", 1): (V, I),
        }
        port_modes = {"p1": [_tem_mode(), _tem_mode()], "p2": [_tem_mode(), _tem_mode()]}
        f_axis = np.linspace(1e9, 10e9, 5)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p1", 0),
            reference_signal=V_exc,
            f_axis=f_axis,
        )
        assert set(S.keys()) == set(recorder.keys())
        # Excited self-channel: V=V_exc, I=0 → S = +1 (open).
        np.testing.assert_allclose(S[("p1", 0)].real, 1.0, atol=1e-9)
        # Other channels: V=0, I=0 → b=0 → S=0.
        np.testing.assert_allclose(np.abs(S[("p1", 1)]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(S[("p2", 0)]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(S[("p2", 1)]), 0.0, atol=1e-12)


# ----------------------------------------------------------------------
# Cutoff-band behaviour
# ----------------------------------------------------------------------


class TestCutoffBand:
    """Below TE-mode cutoff Z is imaginary — formula must remain numerically sane."""

    def test_returns_finite_S_below_cutoff(self):
        """Open-circuit (V=pulse, I=0) → S=+1 even where ω < ω_c."""
        N, dt = 256, 5e-12
        t = np.arange(N) * dt
        V_vals = np.exp(-((t - 50 * dt) ** 2) / (2.0 * (20 * dt) ** 2))
        V_sig = Signal1D(t=t, values=V_vals, dt=dt)
        I_sig = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_te10_mode(f_c=6.557e9)]}
        # Span below and above cutoff.
        f_axis = np.linspace(1e9, 12e9, 12)

        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
        )
        S_pp = S[("p", 0)]
        assert np.all(np.isfinite(S_pp.real))
        assert np.all(np.isfinite(S_pp.imag))
        # Open-circuit: |S| = 1 everywhere — the formula is Z-blind for
        # the I=0 case (the √Z factors in V/(√Z) cancel between a and b).
        np.testing.assert_allclose(np.abs(S_pp), 1.0, atol=1e-9)

    def test_modulus_bounded_in_cutoff_for_passive_open(self):
        """|S| ≤ 1 for the open-circuit case — sanity for the architecture."""
        N, dt = 256, 5e-12
        t = np.arange(N) * dt
        V_vals = np.exp(-((t - 50 * dt) ** 2) / (2.0 * (20 * dt) ** 2))
        V_sig = Signal1D(t=t, values=V_vals, dt=dt)
        I_sig = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_te10_mode(f_c=6.557e9)]}
        f_axis = np.linspace(1e9, 20e9, 30)
        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
        )
        assert np.max(np.abs(S[("p", 0)])) <= 1.0 + 1e-9


# ----------------------------------------------------------------------
# Numerical floor (a_threshold)
# ----------------------------------------------------------------------


class TestAThreshold:
    def test_zero_excitation_yields_nan_everywhere(self):
        """V=I=0 on excited channel  →  ``a_peak = 0``  →  S ≡ NaN.

        This is the early-exit branch that protects against ``0/0``.
        It is the cleanest synthetic NaN test: when the only signal in
        the recorder is identically zero, the threshold mechanism cannot
        recover anything meaningful.
        """
        N, dt = 64, 1e-11
        t = np.arange(N) * dt
        zero = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (zero, zero), ("q", 0): (zero, zero)}
        port_modes = {"p": [_tem_mode()], "q": [_tem_mode()]}
        f_axis = np.linspace(1e9, 10e9, 7)
        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=zero,
            f_axis=f_axis,
        )
        for key in recorder:
            assert np.all(np.isnan(S[key]))

    def test_threshold_filters_low_a_below_relative_floor(self):
        """A heterogeneous setup: V_excited has narrow-band spectrum.

        Construct ``V_excited`` from a tone-burst (narrow band around
        f₀=5 GHz).  ``I=0`` on the excited port keeps the math 1-D:
        ``a = V/(2√Z)``.  Out-of-band, ``|a| ≪ |a|_max`` and the
        threshold guard substitutes NaN.  Validates that the relative-
        floor mechanism actually fires when the spectrum has real gaps.
        """
        N, dt = 4096, 5e-12
        t = np.arange(N) * dt
        f0 = 5e9
        sigma = 50 * dt
        # Tone burst: cosine inside a narrow Gaussian envelope.
        envelope = np.exp(-((t - 200 * dt) ** 2) / (2.0 * sigma**2))
        V_vals = np.cos(2 * math.pi * f0 * t) * envelope
        V_sig = Signal1D(t=t, values=V_vals, dt=dt)
        I_sig = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (V_sig, I_sig)}
        port_modes = {"p": [_tem_mode()]}
        # Sample around the band centre AND well outside.
        f_axis = np.array([1e9, f0, 2 * f0, 50e9, 80e9])
        S = compute_s_parameters(
            recorder,
            port_modes,
            ("p", 0),
            reference_signal=V_sig,
            f_axis=f_axis,
            a_threshold=1e-3,
        )
        S_pp = S[("p", 0)]
        # In-band (f0): valid, |S| = 1 (open-circuit).
        assert not np.isnan(S_pp[1])
        # Far out-of-band (50, 80 GHz): well below threshold → NaN.
        assert np.isnan(S_pp[3])
        assert np.isnan(S_pp[4])


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


class TestValidation:
    def test_empty_recorder_signals_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_s_parameters(
                {},
                {"p": [_tem_mode()]},
                ("p", 0),
                reference_signal=Signal1D(t=np.arange(4) * 1e-12, values=np.ones(4), dt=1e-12),
                f_axis=np.array([1e9]),
            )

    def test_missing_excited_channel_raises(self):
        N, dt = 16, 1e-11
        t = np.arange(N) * dt
        V = Signal1D(t=t, values=np.ones(N), dt=dt)
        I = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (V, I)}
        with pytest.raises(KeyError, match="excited channel"):
            compute_s_parameters(
                recorder,
                {"p": [_tem_mode()]},
                ("not_a_port", 0),
                reference_signal=V,
                f_axis=np.array([1e9]),
            )

    def test_zero_or_negative_frequency_raises(self):
        N, dt = 16, 1e-11
        t = np.arange(N) * dt
        V = Signal1D(t=t, values=np.ones(N), dt=dt)
        I = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 0): (V, I)}
        with pytest.raises(ValueError, match="positive frequencies"):
            compute_s_parameters(
                recorder,
                {"p": [_tem_mode()]},
                ("p", 0),
                reference_signal=V,
                f_axis=np.array([0.0, 1e9]),
            )
        with pytest.raises(ValueError, match="positive frequencies"):
            compute_s_parameters(
                recorder,
                {"p": [_tem_mode()]},
                ("p", 0),
                reference_signal=V,
                f_axis=np.array([-1e9, 1e9]),
            )

    def test_recorder_port_missing_in_port_modes_raises(self):
        N, dt = 16, 1e-11
        t = np.arange(N) * dt
        V = Signal1D(t=t, values=np.ones(N), dt=dt)
        I = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p_unknown", 0): (V, I)}
        with pytest.raises(KeyError, match="not found in port_modes"):
            compute_s_parameters(
                recorder,
                {"p": [_tem_mode()]},
                ("p_unknown", 0),
                reference_signal=V,
                f_axis=np.array([1e9]),
            )

    def test_mode_index_out_of_range_raises(self):
        N, dt = 16, 1e-11
        t = np.arange(N) * dt
        V = Signal1D(t=t, values=np.ones(N), dt=dt)
        I = Signal1D(t=t, values=np.zeros(N), dt=dt)
        recorder = {("p", 5): (V, I)}
        with pytest.raises(ValueError, match="mode_idx"):
            compute_s_parameters(
                recorder,
                {"p": [_tem_mode()]},
                ("p", 5),
                reference_signal=V,
                f_axis=np.array([1e9]),
            )
