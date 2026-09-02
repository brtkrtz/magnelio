"""Power-wave S-parameter post-processing for the modal port pipeline.

Consumes the ``{(port_label, mode_idx): (V_signal, I_signal)}`` mapping
returned by :meth:`magnelio.ports.PortSignalRecorder.finalize` and
the per-port :class:`~magnelio.ports._modal.Mode` list, and produces the
S-parameter spectrum for one excited (port, mode) pair.

Implements power-wave S-parameters with sqrt(W) excitation
normalisation — the post-processing half of the modal-port rewrite
described in ``reference_architecture_waveguide_ports.md`` §3.7 and §4.2.

Key numerical concerns
----------------------

1. **Yee half-step stagger between V and I (temporal).**  The recorder
   samples ``V`` from ``e`` at ``t^{n+1}`` (post-E-update) and ``I``
   from ``h`` at ``t^{n+1/2}`` (pre-H-update), but stores both with the
   naive time axis ``t = arange(N)·dt``.  Carrier-test derivation: with
   ``V_phys = exp(jω₀t)``, ``I_phys = exp(jω₀t)/Z``, the DFT samples
   give ``V_FFT ∝ exp(jω₀·dt)`` and ``I_FFT ∝ exp(jω₀·dt/2)/Z``.  For
   a matched outgoing wave (``V_phys/I_phys = Z``) to come out of the
   formula as ``b = 0``, we therefore multiply ``I_FFT`` by
   ``exp(+jω·dt/2)`` — this rotates I forward by half a step so it
   aligns with the V samples.  The common ``exp(jω·dt)`` factor on V
   cancels in the S-parameter ratio ``b/a`` and is therefore not
   removed explicitly.

1b. **Yee half-cell stagger between V and I (spatial).**  ``V`` is
   projected from the tangential E-edges *on* the port plane, but the
   co-located dual H-faces sit half a normal cell *inside* the domain
   (``PortPlane.plane_idx_H``), so ``I`` is sampled at
   ``z' = normal_dx/2`` while ``V`` is sampled at ``z' = 0``.  A
   travelling wave therefore reaches the I plane phase-shifted by
   ``θ = γ_m(ω)·normal_dx/2`` relative to the V plane, and the naive
   ``(V ∓ Z·I)/2`` decomposition leaks ``≈ sin(θ)`` of the forward
   wave into ``b`` — with λ/20 meshing this is a −22 dB ``|S11|`` floor at
   the band edge (``θ = π/40``), *independent of f_max*.  When the
   caller provides ``port_normal_dx``, we instead solve the exact
   two-plane system

       V(0)/√Z        = a + b
       √Z·I(d/2)      = a·e^{−θ} − b·e^{+θ},    θ = γ_m(ω)·d/2

   for the power waves *at the port plane*:

       a = (V/√Z·e^{+θ} + √Z·I) / (e^{+θ} + e^{−θ})
       b = (V/√Z·e^{−θ} − √Z·I) / (e^{+θ} + e^{−θ})

   which reduces to the standard form for ``θ → 0`` and stays
   well-defined below cut-off (``γ`` real → cosh in the denominator).

1c. **Discrete vs continuum propagation factor.**  The continuum
   ``θ = γ_m(ω)·d/2`` carries the grid-dispersion gap
   ``≈ (β·d/2)³·(1−r²)/6`` against the *discrete* travelling wave the
   solver actually propagates — a measured-``|S11|`` cap near −70 dB on a
   λ/20 mesh, independent of the absorber.  For modes whose port
   termination certifies the exact 1D chain (DTBC), the caller
   passes the discrete line parameters ``(r, q)`` via
   ``port_line_params`` and the de-stagger uses the exact discrete
   half-cell factor ``λ^{1/2}(e^{jω·dt})`` instead
   (:func:`~magnelio.ports._modal.dtbc.destagger_theta`).  For the
   discrete TEM chain the V/I magnitude ratio is frequency-independent
   (the travelling-wave calibration is exact at all frequencies), so
   with this factor the a/b decomposition is exact for the discrete
   wave — the measurement chain no longer floors the reflection.

2. **Power-wave normalisation.**  Per §6 of
   ``reference_waveguide_ports.md``, the standard form is:

       a = (V/√Z + √Z·I) / 2          [unit: √W]
       b = (V/√Z − √Z·I) / 2          [unit: √W]

   With this normalisation, ``|a|² = P_inj`` and ``|S| ≤ 1`` holds for
   every passive network — independent of any Z-mismatch between
   excited and observed ports.  Equivalent to the ``(V ± Z·I)/2``
   power-wave form when we further divide by ``√Z`` on each side.

3. **Modes below cut-off.**  Commercial EM suites return S-parameters
   for every requested frequency, even where the mode is evanescent
   (``Z`` then imaginary).  We follow the same convention: the formalism is
   well-defined, the magnitude carries diagnostic value (in particular,
   spurious ``|S| > 1`` under cut-off would flag an operator pathology).
   Only the numerical division ``b / a_excited`` is guarded — frequencies
   where ``|a_excited|`` falls below ``a_threshold`` (default ``1e-12``
   relative to the spectrum peak) yield ``NaN`` to avoid amplifying
   round-off in the tails of the excitation spectrum.
"""

# Design: DD-042 (power-wave S-parameters, sqrt(W) excitation normalisation),
# WP-R2 (discrete de-stagger for DTBC-certified chains).

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.signal.windows import tukey

from magnelio.ports._modal.dtbc import destagger_theta, dtbc_wave_impedance
from magnelio.ports._modal.mode import Mode
from magnelio.signals.signal_1d import Signal1D

_TUKEY_ALPHA_DEFAULT = 0.05

# Conditioning guards for the spectral power-wave decomposition on a
# *full* rfft axis (time-domain de-stagger).  The exact discrete
# two-plane denominator is a cosh (>= 2, never ill-conditioned), but
# the continuum-theta fallback has a cos-denominator zero crossing
# when the half-cell reaches a quarter wavelength, and the TE/TM wave
# impedance is singular at the (discrete) cut-off.  Bins failing the
# guards fall back to the co-located frozen-Z form — they carry no
# pulse energy on a band-limited excitation (see
# :func:`destaggered_power_waves`).
_DENOM_GUARD = 0.05
_Z_RATIO_GUARD = 1.0e3


def channel_reference_impedance(
    mode,
    omega: np.ndarray,
    dt: float,
    line_params: tuple | None = None,
) -> np.ndarray:
    """Reference impedance ``Z(ω)`` of one recorded channel (complex).

    The impedance the power-wave split values the channel's V/I
    against: the exact discrete wave impedance of the leapfrog chain on
    a certified Klein-Gordon channel (``line_params`` carrying ``z0``
    — concern 1d of the module docstring: the continuum value misses
    the discrete wave's V/I by O((ω·dt)², (β·dz)²), a −40…−60 dB
    measured floor), and ``mode.z_modal(ω)`` otherwise — the
    frequency-flat quasi-static line impedance of a quasi-TEM channel,
    the wave impedance of a hollow-pipe mode, the Thévenin impedance
    of a lumped port.
    """
    omega = np.asarray(omega, dtype=float)
    if line_params is not None and len(line_params) > 2 and line_params[2] is not None:
        return dtbc_wave_impedance(
            omega * dt,
            line_params[1],
            line_params[2],
            mode.mode_type.value,
        )
    return np.array([mode.z_modal(float(w)) for w in omega], dtype=complex)


def spectral_power_waves(
    V_f: np.ndarray,
    I_f: np.ndarray,
    omega: np.ndarray,
    dt: float,
    mode,
    *,
    normal_dx: float | None = None,
    line_params: tuple | None = None,
    guarded: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-domain power-wave split of one recorded channel.

    The shared decomposition core of :func:`compute_s_parameters` and
    :func:`destaggered_power_waves`: reference impedance (exact
    discrete wave impedance on certified Klein-Gordon channels,
    ``mode.z_modal`` otherwise), then the two-plane spatial de-stagger
    (exact discrete ``λ^{1/2}`` factor with ``line_params``, continuum
    ``e^{−γ·d/2}`` with ``normal_dx`` alone, co-located ``(V ∓ Z·I)/2``
    without either) — concerns 1b–1d of the module docstring.

    Parameters
    ----------
    V_f, I_f : np.ndarray
        Complex V/I spectra of the channel.  ``I_f`` must already
        carry the temporal Yee half-step alignment
        ``exp(+jω·dt/2)`` (concern 1).
    omega : np.ndarray
        Angular frequencies of the spectra [rad/s], same shape.
    dt : float
        Solver time step [s] (discrete factors evaluate at ``ω·dt``).
    mode : Mode or mode-shaped stub
        Supplies ``z_modal(omega)`` and — on the continuum de-stagger
        path — ``gamma(omega)``; ``mode_type.value`` selects the
        discrete wave-impedance branch.
    normal_dx : float, optional
        Port-normal boundary cell size [m]; enables the two-plane
        spatial de-stagger.
    line_params : tuple, optional
        ``(r, q)`` or ``(r, q, z0)`` of the certified discrete chain
        (``PortOperatorModal.dtbc_line_params``); enables the exact
        discrete factors.
    guarded : bool, default False
        NaN out bins where the decomposition is singular or
        ill-conditioned (continuum cos-denominator zero beyond the
        trust band, wave impedance collapsing/diverging at a
        cut-off).  ``False`` — the S-parameter convention: modes
        below cut-off keep their (diagnostic) values, exactly the
        historical behaviour of :func:`compute_s_parameters`.
        ``True`` — the full-rfft-axis convention of
        :func:`destaggered_power_waves`, whose caller masks the NaN
        bins and falls back to the co-located split.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Complex ``a(ω)``, ``b(ω)``.
    """
    omega = np.asarray(omega, dtype=float)
    Z = channel_reference_impedance(mode, omega, dt, line_params)
    with np.errstate(divide="ignore", invalid="ignore"):
        if guarded:
            # Cut-off guard: |Z| collapsing to 0 or diverging turns
            # V/sqrt(Z) resp. sqrt(Z)*I into pure amplification of
            # round-off; NaN those bins out for the caller's fallback.
            z_mag = np.abs(Z)
            z_finite = z_mag[np.isfinite(z_mag) & (z_mag > 0.0)]
            if z_finite.size:
                z_med = float(np.median(z_finite))
                bad_z = (
                    ~np.isfinite(z_mag)
                    | (z_mag < z_med / _Z_RATIO_GUARD)
                    | (z_mag > z_med * _Z_RATIO_GUARD)
                )
                Z = np.where(bad_z, np.nan, Z)
        sqrt_Z = np.sqrt(Z)

        if normal_dx is not None:
            # Spatial half-cell de-stagger (concern 1b): solve the
            # V(0) / I(normal_dx/2) two-plane system for a, b at the
            # port plane.
            if line_params is not None:
                # Concern 1c: exact discrete half-cell factor of the
                # certified 1D chain.
                r_chain, q_chain = line_params[0], line_params[1]
                theta = destagger_theta(omega * dt, r_chain, q_chain)
            else:
                theta = (
                    0.5
                    * float(normal_dx)
                    * np.array(
                        [mode.gamma(float(w)) for w in omega],
                        dtype=complex,
                    )
                )
            exp_p = np.exp(+theta)
            exp_m = np.exp(-theta)
            denom = exp_p + exp_m
            if guarded:
                denom = np.where(
                    np.abs(denom) < _DENOM_GUARD,
                    np.nan,
                    denom,
                )
            a = (V_f / sqrt_Z * exp_p + sqrt_Z * I_f) / denom
            b = (V_f / sqrt_Z * exp_m - sqrt_Z * I_f) / denom
        else:
            a = (V_f / sqrt_Z + sqrt_Z * I_f) / 2.0
            b = (V_f / sqrt_Z - sqrt_Z * I_f) / 2.0
    return a, b


def destaggered_power_waves(
    V_sig: Signal1D,
    I_sig: Signal1D,
    mode,
    *,
    z_ref: float,
    normal_dx: float | None = None,
    line_params: tuple | None = None,
) -> tuple[Signal1D, Signal1D]:
    """Exact-as-possible time-domain power waves ``a(t)``, ``b(t)``.

    The frequency-domain counterpart of the naive time-domain split
    ``(V/√Z ± √Z·I)/2``: both records are taken to the rfft axis, the
    per-frequency corrections of :func:`compute_s_parameters` are
    applied bin by bin (temporal Yee half-step, spatial two-plane
    de-stagger — exact discrete ``λ^{1/2}`` on certified chains —
    and the exact discrete wave impedance), and the result is
    transformed back.  This removes the ``β·dz/4``-level incident
    leak of the co-located split (a derivative-of-pulse-shaped ghost
    in ``b`` at −22 dB on λ/20 meshes) down to the port floor.

    Out-of-band bins where the exact decomposition is singular
    (continuum cos-denominator zero beyond the trust band, TE/TM wave
    impedance at cut-off, DC) fall back to the frozen-``z_ref``
    co-located split.  On a band-limited excitation those bins carry
    no pulse energy, so the fallback is inert; it merely keeps the
    inverse transform bounded.

    Parameters
    ----------
    V_sig, I_sig : Signal1D
        Recorded V/I of one channel on the naive shared time axis
        (:meth:`PortSignalRecorder.finalize` output).
    mode : Mode or mode-shaped stub
        As in :func:`spectral_power_waves`.
    z_ref : float
        Real reference impedance [Ω] of the fallback split (the
        frozen ``z_modal(2π·f_ref)`` of the caller).
    normal_dx, line_params
        As in :func:`spectral_power_waves`.

    Returns
    -------
    (Signal1D, Signal1D)
        ``a(t)``, ``b(t)`` on the input time axis [√W].
    """
    if z_ref <= 0.0:
        raise ValueError(f"z_ref must be positive, got {z_ref}")
    V = np.asarray(V_sig.values, dtype=float)
    I = np.asarray(I_sig.values, dtype=float)
    n = V.size
    dt = float(V_sig.dt)
    omega = 2.0 * math.pi * np.fft.rfftfreq(n, dt)

    V_f = np.fft.rfft(V)
    I_f = np.fft.rfft(I)
    # Temporal Yee half-step alignment (concern 1), exact per bin — but only
    # for modally-sampled I.  Lumped ports sample I co-temporally with V, so
    # their channels opt out via ``mode.i_cotemporal`` (DD-075, F3), the same
    # guard ``compute_s_parameters`` uses.
    if not getattr(mode, "i_cotemporal", False):
        I_f = I_f * np.exp(+0.5j * omega * dt)

    a_f, b_f = spectral_power_waves(
        V_f,
        I_f,
        omega,
        dt,
        mode,
        normal_dx=normal_dx,
        line_params=line_params,
        guarded=True,
    )

    # Frozen-Z co-located fallback for the singular bins.
    sqrt_zr = math.sqrt(z_ref)
    a_0 = 0.5 * (V_f / sqrt_zr + sqrt_zr * I_f)
    b_0 = 0.5 * (V_f / sqrt_zr - sqrt_zr * I_f)
    good = np.isfinite(a_f) & np.isfinite(b_f)
    a_f = np.where(good, a_f, a_0)
    b_f = np.where(good, b_f, b_0)

    a_t = np.fft.irfft(a_f, n=n)
    b_t = np.fft.irfft(b_f, n=n)
    return (
        Signal1D(t=V_sig.t, values=a_t, dt=dt, label="a"),
        Signal1D(t=V_sig.t, values=b_t, dt=dt, label="b"),
    )


def compute_s_parameters(
    recorder_signals: dict[tuple[str, int], tuple[Signal1D, Signal1D]],
    port_modes: dict[str, list[Mode]],
    excited: tuple[str, int],
    reference_signal: Signal1D,
    f_axis: np.ndarray,
    *,
    a_threshold: float = 1e-12,
    taper_signals: bool = False,
    port_normal_dx: dict[str, float] | None = None,
    port_line_params: dict[tuple[str, int], tuple[float, float]] | None = None,
    return_incident: bool = False,
    return_reference: bool = False,
    port_reference_scale: dict[str, float] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """Compute power-wave S-parameters from recorded V/I time-series.

    With ``return_incident=True`` the result is the pair ``(S, a)`` where
    ``a`` is the incident power-wave spectrum of the excited channel on
    ``f_axis`` [√W · s] — the actual incident wave the run launched,
    which the monitors' "per 1 W incident" normalisation refers to (a
    TE/TM channel's wave impedance varies with frequency, so a
    frequency-flat excitation waveform does not launch frequency-flat
    incident power).

    Parameters
    ----------
    recorder_signals : dict
        Output of :meth:`PortSignalRecorder.finalize`.  Keys are
        ``(port_label, mode_idx)``; values are ``(V_signal, I_signal)``.
        Both signals share the same naive time axis ``t = arange(N)·dt``;
        the half-step Yee stagger between V and I is corrected internally
        by this function.
    port_modes : dict[str, list[Mode]]
        Per-port ordered list of :class:`Mode` objects.  Mode index in
        each list must align with the recorder's ``mode_idx``.  Used to
        evaluate the modal reference impedance ``Z(ω)`` at every
        ``f_axis`` point.
    excited : (str, int)
        ``(port_label, mode_idx)`` of the source.  Defines the
        denominator ``a_excited(f)`` of the S-parameter ratio.
    reference_signal : Signal1D
        Original user excitation waveform ``s(t)``.  Currently unused in
        the S-parameter math itself (a_excited is extracted from V/I
        for self-consistency); kept on the signature as a sanity-check
        anchor and so the API does not need to change when
        ``PortOperatorModal`` adopts the §4.3 power-normalised injection.
    f_axis : np.ndarray
        Target frequencies [Hz], shape ``(Nf,)``.  Must be > 0.
    a_threshold : float, default 1e-12
        Relative threshold below which ``|a_excited(f)| / max(|a_excited|)``
        is treated as numerical floor and S is reported as ``NaN``.
    taper_signals : bool, default False
        If ``True``, multiply every recorder ``V`` and ``I`` array with
        a symmetric Tukey window of ``alpha = 0.05`` (i.e. the first and
        last 2.5 % of samples taper to zero, the inner 95 % stay
        untouched) before the DFT.  Suppresses the rectangular-window
        sidelobes caused by a non-vanishing residual at the truncation
        edge — useful when ``energy_stop_db`` terminates the run while
        the tail of the transient is still ringing above the FFT noise
        floor.  Off by default; the rectangular window is the
        unbiased reference for closed-form mode validation.
    port_normal_dx : dict[str, float] or None, default None
        Per-port boundary-cell size along the port normal
        (``PortOperatorModal.plane.normal_dx``), keyed by port name.
        When given for a port, the spatial half-cell stagger of the I
        sampling plane is compensated exactly (concern 1b above) for
        every channel of that port.  Ports missing from the dict (and
        the ``None`` default) fall back to the co-located ``(V ∓ Z·I)/2``
        decomposition — correct for lumped ports, where V and I live on
        the same cell, and the historical behaviour for modal ports.

        Residual accuracy after de-stagger: the historical grading
        floor (measured: growth-1.4 transversal grading at ≈ −23 dB)
        was a V/I measurement error of the pointwise H-voltage
        convention and is resolved by the travelling-wave port
        profiles — graded transversal port meshes are
        first-class.  For TEM modes, the pair (exact DTBC termination +
        the ``port_line_params`` discrete de-stagger below) removes
        the remaining absorber and measurement floors entirely
        (straight-line benchmarks −131 … −159 dB).  TE/TM modes on
        certified uniform chains run the Klein-Gordon DTBC,
        which removed the former Mur-1st ≈ −19 dB near-cutoff peak
        structurally; only analytical-path modes remain on Mur-1st.
    port_line_params : dict[(str, int), tuple] or None, default None
        Per-channel discrete line parameters ``(r, q)`` or
        ``(r, q, z0)`` — the modal Courant number, discrete
        cut-off × dt, and static impedance constant of the exact 1D
        chain certified by the port termination
        (``PortOperatorModal.dtbc_line_params``).  Channels present
        here (and whose port is in ``port_normal_dx``) are
        de-staggered with the exact discrete half-cell factor
        ``λ^{1/2}`` instead of the continuum ``e^{−γ·d/2}``
        (concern 1c above).  Channels carrying a non-``None`` ``z0``
        (certified Klein-Gordon chains) additionally replace
        the continuum ``z_modal(ω)`` by the exact discrete wave
        impedance
        :func:`~magnelio.ports._modal.dtbc.dtbc_wave_impedance` —
        the continuum impedance misses the discrete wave's V/I by
        O((ω·dt)², (β·dz)²), a −40…−60 dB measured-``|S11|`` cap on
        λ/20 meshes.  Missing channels fall back to the continuum
        forms.

    return_reference : bool, default False
        Also return the per-channel real reference impedance
        ``Z(f_axis)`` the power waves are defined against
        (:func:`channel_reference_impedance`, real part), scaled per
        ``port_reference_scale``.  Appended after the incident
        amplitude when both flags are set: ``(S, a)``, ``(S, z)`` or
        ``(S, a, z)``.
    port_reference_scale : dict[str, float], optional
        Half-window → full-model factor per port for *line* impedances
        (a port cut by a symmetry plane solves half the cross-section;
        the published reference is the full-model value).  Applied to
        channels whose mode carries a ``z_line``; wave impedances are
        intensive and untouched.

    Returns
    -------
    dict[(str, int), np.ndarray]
        Mapping ``(port_label, mode_idx) → S(f_axis)`` of shape
        ``(Nf,)``, complex.  One entry per channel in ``recorder_signals``.

    Raises
    ------
    KeyError
        If ``excited`` is not a key in ``recorder_signals``, or if a
        recorder channel references a port not present in ``port_modes``.
    ValueError
        If ``f_axis`` contains a non-positive frequency, or if a mode
        index is out of range for its port.

    Notes
    -----
    **Degenerate mode pairs.**  Two modes with identical
    cut-offs (coax TE11 polarisations, TE_mn/TM_mn in a rectangle)
    span a degenerate subspace: the individual per-channel ``S_ij``
    entries within the pair depend on the (mesh-dependent) basis the
    eigensolver picked, and the cross-coupling between the partners
    does not vanish with refinement — ``build_modal_port`` warns.
    The basis-independent transmission observable is the *total*
    power into the degenerate subspace,
    ``Σ_k |S(out_k, in)|²`` summed over the pair — the convention the
    ``examples/straight_waveguide_*.py`` acceptance scripts print.
    Reflection ``S_mm`` of the excited channel itself is
    basis-independent for a symmetric discretisation.
    """
    # Design: WP-R3 (discrete Klein-Gordon wave impedance via z0),
    # WP-U4 (degenerate mode-pair semantics).
    if not recorder_signals:
        raise ValueError("recorder_signals is empty")
    if excited not in recorder_signals:
        raise KeyError(
            f"excited channel {excited!r} not in recorder_signals; "
            f"available: {sorted(recorder_signals.keys())}"
        )
    f_axis = np.asarray(f_axis, dtype=float)
    if np.any(f_axis <= 0.0):
        raise ValueError(
            "f_axis must contain only positive frequencies "
            "(power-wave decomposition undefined at DC)"
        )

    omega = 2.0 * math.pi * f_axis

    # Yee half-step phase correction for MODALLY-sampled I (V ∼ e at
    # t^{n+1}, I ∼ h at t^{n+1/2}).  Lumped ports sample I co-temporally
    # with V — ``PortOperatorLumped.project_I`` returns the t^{n+1}
    # Thévenin current and ignores h — so their channels opt out via
    # ``mode.i_cotemporal`` (DD-075, F3); applying the shift there would
    # over-rotate I by ω·dt/2 and cap the achievable match at ~π·f·dt/2.
    dt = next(iter(recorder_signals.values()))[0].dt
    i_phase_correction = np.exp(+1j * omega * dt / 2.0)

    # Sanity: the unused-but-still-referenced reference_signal.  Force
    # an early evaluation so a malformed input fails here rather than
    # silently downstream.
    _ = reference_signal.at_frequencies(f_axis)

    if taper_signals:
        N = len(next(iter(recorder_signals.values()))[0].values)
        window = tukey(N, alpha=_TUKEY_ALPHA_DEFAULT, sym=True)
    else:
        window = None

    a_channels: dict[tuple[str, int], np.ndarray] = {}
    b_channels: dict[tuple[str, int], np.ndarray] = {}
    z_channels: dict[tuple[str, int], np.ndarray] = {}

    for key, (V_sig, I_sig) in recorder_signals.items():
        label, mode_idx = key
        if label not in port_modes:
            raise KeyError(
                f"port {label!r} from recorder not found in port_modes "
                f"(available: {sorted(port_modes.keys())})"
            )
        modes = port_modes[label]
        if not (0 <= mode_idx < len(modes)):
            raise ValueError(
                f"mode_idx {mode_idx} out of range for port {label!r} (has {len(modes)} modes)"
            )
        mode = modes[mode_idx]

        if window is not None:
            V_sig = Signal1D(
                t=V_sig.t,
                values=V_sig.values * window,
                dt=V_sig.dt,
                label=V_sig.label,
            )
            I_sig = Signal1D(
                t=I_sig.t,
                values=I_sig.values * window,
                dt=I_sig.dt,
                label=I_sig.label,
            )

        V_f = V_sig.at_frequencies(f_axis)
        I_f = I_sig.at_frequencies(f_axis)
        if not getattr(mode, "i_cotemporal", False):
            I_f = I_f * i_phase_correction

        line_params = port_line_params.get(key) if port_line_params is not None else None
        normal_dx = port_normal_dx.get(label) if port_normal_dx is not None else None
        a_channels[key], b_channels[key] = spectral_power_waves(
            V_f,
            I_f,
            omega,
            dt,
            mode,
            normal_dx=normal_dx,
            line_params=line_params,
        )
        if return_reference:
            z = channel_reference_impedance(mode, omega, dt, line_params).real
            if getattr(mode, "z_line", None) is not None and port_reference_scale:
                z = z * float(port_reference_scale.get(label, 1.0))
            z_channels[key] = z

    def _pack(S_out, a_out):
        out = [S_out]
        if return_incident:
            out.append(a_out)
        if return_reference:
            out.append(z_channels)
        return out[0] if len(out) == 1 else tuple(out)

    a_excited = a_channels[excited]
    a_peak = float(np.max(np.abs(a_excited)))
    if a_peak == 0.0:
        # Fully zero excitation: S is undefined everywhere.
        S_nan = {key: np.full_like(f_axis, np.nan, dtype=complex) for key in recorder_signals}
        return _pack(S_nan, a_excited)

    valid = np.abs(a_excited) >= (a_threshold * a_peak)
    safe_a = np.where(valid, a_excited, 1.0 + 0j)

    S: dict[tuple[str, int], np.ndarray] = {}
    for key, b in b_channels.items():
        S_k = b / safe_a
        S_k = np.where(valid, S_k, np.nan + 1j * np.nan)
        S[key] = S_k
    return _pack(S, a_excited)


def compute_band_s_parameters(
    recorder_signals: dict[tuple[str, int], tuple[Signal1D, Signal1D]],
    ports: list,
    excited: tuple[str, int],
    f_axis: np.ndarray,
    *,
    m_eps: np.ndarray | None = None,
    m_mu: np.ndarray | None = None,
    c_3d=None,
    a_threshold: float = 1e-12,
    return_reference: bool = False,
    port_reference_scale: dict[str, float] | None = None,
    search: str = "track",
) -> dict[tuple[str, int], np.ndarray]:
    """S-parameters of a pulsed broadband run through band DTBC ports.

    The per-frequency true-mode decomposition applied to a
    single pulsed record: per ``f_axis`` point the true
    discrete modes of each port cross-section are solved on the
    stored inward chain, their exact V/I responses through the
    port's *fixed* recording profiles are synthesised
    (:mod:`~magnelio.ports._modal.dispersion` — the de-stagger and
    the discrete wave impedance are contained in the phasors), and
    the recorded spectra are decomposed by the joint linear system

        V_c(f) = sum_j a_j v_in[c, j] + b_j v_out[c, j]
        I_c(f) = sum_j a_j i_in[c, j] + b_j i_out[c, j]

    over all recording channels ``c`` and all modes ``j`` propagating
    at ``f``, every mode's phasors scaled to unit power so ``a`` and
    ``b`` are power waves (DD-244).  ``S[(port, c)] = b_(port, c) /
    a_excited``.

    Cost note: with ``search="track"`` (default) each frequency point
    runs one sparse factorisation per tracked mode and port; the full
    five-shift arc search runs at the first and last axis point and
    wherever a channel is unassigned.

    Parameters
    ----------
    recorder_signals : dict
        Output of :meth:`PortSignalRecorder.finalize`, keyed by
        ``(port_label, mode_idx)``.
    ports : list
        The run's band ports, either as built
        ``PortOperatorBandDTBC`` instances (each carrying
        ``band_data``) or as
        :class:`~magnelio.ports._modal.band_dtbc.BandDecomposition`
        records — the detached form a project store reads back.
    excited : (str, int)
        ``(port_label, channel)`` of the pulsed source.
    f_axis : np.ndarray
        Evaluation frequencies [Hz]; must lie inside every port's
        subspace band (content outside the band is not certified).
    m_eps, m_mu, c_3d : optional
        The mesh-side operators, needed only for records that predate
        the stored plane masses and curl slice (DD-244); a current
        record is self-contained.
    a_threshold : float, default 1e-12
        Relative ``|a_excited|`` floor below which S is NaN.
    return_reference : bool, default False
        Also return the per-channel reference impedance ``Z(f)`` the
        power waves are defined against (real, NaN where the channel
        has no mode), as ``(S, z_ref)``.
    port_reference_scale : dict[str, float], optional
        Half-window → full-model factor per port applied to the
        returned references (a port cut by a symmetry plane).
    search : {"track", "full"}
        Mode continuation strategy of
        :func:`~magnelio.ports._modal.dispersion.solve_port_dispersion`.

    Returns
    -------
    dict[(str, int), np.ndarray]
        Complex S spectrum per (port, channel) against the excited
        channel.  Channels whose family does not propagate at a
        frequency carry NaN there.
    """
    # Design: WP-R4a (per-frequency true-mode decomposition; cost-watch
    # numbers measured there), DD-244 (continuation, power waves).
    from magnelio.ports._modal.band_dtbc import BandDecomposition
    from magnelio.ports._modal.dispersion import (
        decompose_power_waves,
        solve_port_dispersion,
    )

    # Accept built operators (live runs) or detached records (a project
    # store read).  Legacy records without stored masses fall back to
    # the mesh-side operators cached on the first built port.
    if m_eps is None or m_mu is None or c_3d is None:
        bd0 = getattr(ports[0], "band_data", None) if ports else None
        if bd0 is not None:
            m_eps = bd0.m_eps if m_eps is None else m_eps
            m_mu = bd0.m_mu if m_mu is None else m_mu
            c_3d = bd0.c_3d if c_3d is None else c_3d
    ports = [
        p if isinstance(p, BandDecomposition) else BandDecomposition.from_operator(p) for p in ports
    ]

    f_axis = np.asarray(f_axis, dtype=float)
    n_f = f_axis.size
    labels = [p.name for p in ports]
    if excited[0] not in labels:
        raise ValueError(f"excited port {excited[0]!r} not among {labels}")

    a_all: dict[tuple[str, int], np.ndarray] = {}
    b_all: dict[tuple[str, int], np.ndarray] = {}
    z_all: dict[tuple[str, int], np.ndarray] = {}
    for port in ports:
        n_ch = port.n_modes
        disp = solve_port_dispersion(
            port,
            f_axis,
            m_eps=m_eps,
            m_mu=m_mu,
            c_3d=c_3d,
            search=search,
        )
        # Spectra of the recorded projections at the axis points
        # (direct DFT; the common source-spectrum factor cancels in
        # b/a).  I is rotated by e^{+j w dt/2} — the recorder's Yee
        # half-step convention, identical to cw_lockin_phasors.
        V0, _ = recorder_signals[(port.name, 0)]
        t_axis = V0.t
        dt = float(t_axis[1] - t_axis[0])
        n_steps = t_axis.size
        w_dt_axis = 2.0 * math.pi * f_axis * dt
        dft = np.exp(
            -1j * np.outer(w_dt_axis, np.arange(n_steps)),
        )
        Vh = np.empty((n_ch, n_f), complex)
        Ih = np.empty((n_ch, n_f), complex)
        for c in range(n_ch):
            V_sig, I_sig = recorder_signals[(port.name, c)]
            Vh[c] = dft @ V_sig.values
            Ih[c] = (dft @ I_sig.values) * np.exp(0.5j * w_dt_axis)
        a, b = decompose_power_waves(disp, Vh, Ih)
        z_scale = float(port_reference_scale.get(port.name, 1.0)) if port_reference_scale else 1.0
        for c in range(n_ch):
            a_all[(port.name, c)] = a[c]
            b_all[(port.name, c)] = b[c]
            z_all[(port.name, c)] = disp.z_line[c] * z_scale

        # DD-235.  Frequencies where a mode propagates but reaches no
        # recording channel.  The system above is written over every
        # channel, so its right-hand side carries the response of *all*
        # modes present at the port; a mode left out of the basis has
        # its content absorbed by the modes that remain.  A channel
        # without a mode stays NaN and is visible; a mode without a
        # channel is not, which is what this collects (at the axis
        # points the full arc search visited).
        if disp.incomplete.size:
            f_lo, f_hi = float(disp.incomplete.min()), float(disp.incomplete.max())
            span = f"{f_lo:.4g} Hz" if f_lo == f_hi else f"{f_lo:.4g}-{f_hi:.4g} Hz"
            warnings.warn(
                f"Port {port.name!r} carries modes that propagate but match "
                f"no recording channel at {disp.incomplete.size} of "
                f"{disp.n_full_searches} searched frequencies ({span}); their "
                "response is absorbed into the channels that were matched, so "
                "S is biased there rather than flagged. Raise the port's "
                "n_modes to cover every propagating mode of its cross-section, "
                "or keep the axis below the next cut-on.",
                stacklevel=2,
            )

    a_exc = a_all[excited]
    finite = np.isfinite(a_exc.real)
    peak = float(np.abs(a_exc[finite]).max()) if finite.any() else 0.0
    valid = finite & (np.abs(np.where(finite, a_exc, 0.0)) >= a_threshold * max(peak, 1e-300))
    safe_a = np.where(valid, a_exc, 1.0)
    S: dict[tuple[str, int], np.ndarray] = {}
    for key, b in b_all.items():
        S[key] = np.where(valid, b / safe_a, np.nan + 1j * np.nan)
    return (S, z_all) if return_reference else S
