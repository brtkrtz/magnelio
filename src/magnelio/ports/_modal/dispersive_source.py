"""Rank-r dispersive source for a modal port (DD-248).

The shipped modal source imprints one frozen quasi-static profile with
one propagation delay: ``e_t(x, t) = phi_qs(x) a(t)``, the incident
half of the TF/SF split in :meth:`PortOperatorModal.update_e`.  The
grid, however, carries a mode whose transverse shape *and* phase
velocity move with frequency, and on an inhomogeneous cross-section
that mismatch is one of the two terms that cap the reported |S11| (the
other, the quasi-static power-wave split, is repaired by the
per-frequency decomposition of ``dispersion.py``).

This module synthesises the replacement: the true mode's transverse
field is solved along the band, decomposed by SVD into ``r`` profiles
``u_k(x)`` with frequency responses ``c_k(f)``, and each profile is
driven by its own waveform ``w_k(t) = irfft(a(f) c_k(f))``.  The
interior plane takes the same family propagated one cell,
``c_k(f) zeta(f)``.  Both are projected onto the channel's dual profile
so the scalar TF/SF subtraction stays exact.

Two properties of the synthesis are not obvious and are certified by
``tests/unit/test_dispersive_source.py``:

* **The coefficients must not be truncated at the solved band.**  The
  excitation spectrum generally reaches past it, and zeroing ``c_k``
  outside makes ``irfft`` ring — the incident wave then never decays
  and the run is evaluated on a signal that is still moving.  The
  coefficients are continued by their edge value and ``zeta`` by
  holding ``eps_eff`` (``beta`` proportional to ``f``), which is its
  physical low-frequency law.
* **The gauge must be anchored.**  ``normalize_gauge`` phases each
  profile on its own largest component, and which component that is can
  change with frequency; the resulting jump survives the SVD (the rank
  is unchanged) and destroys the time signal.  Every profile is instead
  phased so its overlap with the channel's recording profile is real
  positive, which is smooth in frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["DispersiveSourceTerms", "synthesise_dispersive_source"]

#: Rank is grown until the worst profile error over the band is below
#: this, unless the caller pins it.  -60 dB sits about 40 dB under the
#: port floors the shipped ports reach, with the measured 17-21 dB of
#: conversion from profile error to reflected amplitude on top.
DEFAULT_TARGET_DB = -60.0

#: Never grow past this without being asked: beyond it the far-port
#: floor caps what any source can buy, and each term costs a waveform
#: and an AXPY per step.
DEFAULT_MAX_RANK = 4


@dataclass(frozen=True)
class DispersiveSourceTerms:
    """Profiles and waveforms of one channel's rank-r source.

    Attributes
    ----------
    profiles_u, profiles_v : np.ndarray
        ``(rank, n_u)`` and ``(rank, n_v)`` transverse profiles on the
        port plane's primal edges.
    waveform : np.ndarray
        ``(rank, n_steps)`` — the port plane's own drive per profile.
    waveform_interior : np.ndarray
        ``(rank, n_steps)`` — the same family one cell inside, i.e. the
        incident wave the TF/SF split subtracts at the interior plane.
    projected, projected_interior : np.ndarray
        ``(n_steps,)`` — the two waveform families projected onto the
        channel's dual profile: the scalars that replace
        ``s(t)`` and ``s(t - tau)`` in the shipped source.
    rank : int
    profile_error_db : float
        Worst relative profile error over the band at this rank.
    """

    profiles_u: np.ndarray
    profiles_v: np.ndarray
    waveform: np.ndarray
    waveform_interior: np.ndarray
    projected: np.ndarray
    projected_interior: np.ndarray
    rank: int
    profile_error_db: float


def _true_mode_fields(record, f_axis, m_eps, m_mu, channel):
    """Transverse E of one channel's true mode per frequency.

    Returns ``(profiles, zeta, f_used)`` with ``profiles`` of shape
    ``(n_u + n_v, n_f)``, unit-power scaled.  Profiles are attributed to
    channels by overlap with the recording profiles rather than by call
    order: the solver emits one field evaluation per assigned channel
    per frequency and its continuation branch iterates a mapping, so the
    order can change between frequencies.
    """
    from magnelio.ports._modal import dispersion as _disp

    seen: list[tuple[float, np.ndarray]] = []
    orig = _disp._wave_fields
    dt = float(record.chain_inward.dt)

    def spy(phi, z_j, chain, curl_slice, n_u, n_v, w_dt, *a, **kw):
        out = orig(phi, z_j, chain, curl_slice, n_u, n_v, w_dt, *a, **kw)
        f = float(w_dt) / (2.0 * math.pi * dt)
        seen.append((f, np.concatenate([np.asarray(out[0]), np.asarray(out[1])]).copy()))
        return out

    _disp._wave_fields = spy
    try:
        disp = _disp.solve_port_dispersion(record, f_axis, m_eps=m_eps, m_mu=m_mu)
    finally:
        _disp._wave_fields = orig

    refs = []
    for c in range(int(record.n_modes)):
        r = np.concatenate(
            [
                np.asarray(record.e_u_profiles[c], dtype=float),
                np.asarray(record.e_v_profiles[c], dtype=float),
            ]
        )
        refs.append(r / np.linalg.norm(r))

    f_got: list[float] = []
    p_got: list[np.ndarray] = []
    for f, vec in seen:
        nv = np.linalg.norm(vec)
        if nv == 0.0:
            continue
        if int(np.argmax([abs(np.vdot(r, vec)) / nv for r in refs])) == channel:
            f_got.append(f)
            p_got.append(vec)
    if not p_got:
        raise ValueError(f"no propagating mode found for channel {channel} on the given band")

    order = np.argsort(f_got)
    f_used = np.asarray(f_got)[order]
    prof = np.stack([p_got[i] for i in order], axis=1)

    power = np.asarray(disp.power)[channel]
    zeta = np.asarray(disp.zeta)[channel]
    f_disp = np.asarray(disp.f_axis)
    ok = np.isfinite(power) & (power > 0.0) & np.isfinite(zeta)
    if not ok.any():
        raise ValueError(f"channel {channel} carries no propagating mode on the given band")
    pw = np.interp(f_used, f_disp[ok], power[ok])
    prof = prof / np.sqrt(np.abs(pw))[None, :]
    zeta_used = np.interp(f_used, f_disp[ok], zeta[ok].real) + 1j * np.interp(
        f_used, f_disp[ok], zeta[ok].imag
    )
    # Smooth, physically anchored gauge (module docstring).
    ref = refs[channel]
    ov = ref @ prof
    prof = prof * np.where(np.abs(ov) > 0.0, np.conj(ov) / np.maximum(np.abs(ov), 1e-300), 1.0)
    return prof, zeta_used, f_used, ref


def synthesise_dispersive_source(
    record,
    channel,
    waveform_samples,
    dt,
    f_band,
    *,
    m_eps=None,
    m_mu=None,
    n_band_points=81,
    rank=None,
    target_db=DEFAULT_TARGET_DB,
    max_rank=DEFAULT_MAX_RANK,
    dual_projector=None,
):
    """Build the rank-r source terms of one channel.

    Parameters
    ----------
    record : BandDecomposition
        The port's dispersion record.
    channel : int
        Channel index within the port.
    waveform_samples : array_like
        The drive ``a(t)`` sampled on the run's own time steps; its
        length sets the length of every synthesised waveform.
    dt : float
        Time step [s].
    f_band : tuple of float
        ``(f_lo, f_hi)`` over which the true modes are solved.  It need
        not cover the excitation — the coefficients are continued
        outside it, never truncated.
    rank : int, optional
        Pin the rank.  When ``None`` the rank grows until the worst
        profile error is under ``target_db``, capped at ``max_rank``.
    dual_projector : callable, optional
        ``f(profile_u, profile_v) -> (v_port, v_interior)``, the port's
        own dual projection of a transverse profile.  Required; the
        operator passes its bound method.

    Returns
    -------
    DispersiveSourceTerms
    """
    if dual_projector is None:
        raise ValueError("dual_projector is required")
    a_t = np.asarray(waveform_samples, dtype=float).ravel()
    n_t = a_t.size
    if n_t < 8:
        raise ValueError("waveform_samples too short to synthesise a source")
    f_lo, f_hi = (float(x) for x in f_band)
    if not (0.0 < f_lo < f_hi):
        raise ValueError(f"f_band must satisfy 0 < f_lo < f_hi; got {f_band}")

    f_solve = np.linspace(f_lo, f_hi, int(n_band_points))
    prof, zeta, f_used, _ref = _true_mode_fields(record, f_solve, m_eps, m_mu, int(channel))

    u_all, s_all, vh_all = np.linalg.svd(prof, full_matrices=False)
    norms = np.linalg.norm(prof, axis=0)

    def err_at(r):
        approx = (u_all[:, :r] * s_all[:r]) @ vh_all[:r, :]
        return float((np.linalg.norm(prof - approx, axis=0) / norms).max())

    if rank is None:
        r = 1
        r_cap = min(int(max_rank), s_all.size)
        while r < r_cap and 20.0 * math.log10(max(err_at(r), 1e-300)) > target_db:
            r += 1
    else:
        r = max(1, min(int(rank), s_all.size))
    err_db = 20.0 * math.log10(max(err_at(r), 1e-300))

    u = u_all[:, :r]
    c_solved = s_all[:r, None] * np.conj(vh_all[:r, :])

    f_fine = np.fft.rfftfreq(n_t, dt)
    a_f = np.fft.rfft(a_t)
    # Continue, never truncate (module docstring).
    c = np.empty((r, f_fine.size), dtype=complex)
    for k in range(r):
        c[k] = np.interp(f_fine, f_used, c_solved[k].real) + 1j * np.interp(
            f_fine, f_used, c_solved[k].imag
        )
    beta = -np.unwrap(np.angle(zeta))
    beta_fine = np.interp(f_fine, f_used, beta)
    lo = f_fine < f_used[0]
    hi = f_fine > f_used[-1]
    beta_fine[lo] = (beta[0] / f_used[0]) * f_fine[lo]
    beta_fine[hi] = (beta[-1] / f_used[-1]) * f_fine[hi]
    zeta_fine = np.exp(-1j * beta_fine)

    w = np.empty((r, n_t))
    w_int = np.empty((r, n_t))
    for k in range(r):
        w[k] = np.fft.irfft(a_f * c[k], n=n_t)
        w_int[k] = np.fft.irfft(a_f * c[k] * zeta_fine, n=n_t)

    n_u = int(record.e_u_profiles[channel].size)
    prof_u = np.ascontiguousarray(u[:n_u, :].real.T)
    prof_v = np.ascontiguousarray(u[n_u:, :].real.T)
    d_face = np.empty(r)
    d_int = np.empty(r)
    for k in range(r):
        d_face[k], d_int[k] = dual_projector(prof_u[k], prof_v[k])

    return DispersiveSourceTerms(
        profiles_u=prof_u,
        profiles_v=prof_v,
        waveform=w,
        waveform_interior=w_int,
        projected=d_face @ w,
        projected_interior=d_int @ w_int,
        rank=r,
        profile_error_db=err_db,
    )
