"""Per-frequency dispersion of a port cross-section (DD-244).

One port, one frequency axis: the true discrete modes of the port's
uniform feed chain at every axis point, the reference impedance and
propagation constant each channel carries there, and the exact
incident/outgoing ``(V, I)`` phasors of every mode through the port's
*recording* profiles.  Three consumers share it:

* the band S-parameter decomposition
  (:func:`~magnelio.post.modal_sparameters.compute_band_s_parameters`),
  which fits the recorded spectra to these phasors;
* the port report's dispersion sweep (``PortReport.dispersion``), the
  per-frequency ``Z``, ``ε_eff`` and ``γ`` of an inhomogeneous line;
* the de-embedding of a modal run on quasi-TEM feeds, which needs the
  discrete ``ζ(f)`` of the channel rather than the quasi-static
  continuum ``γ``.

The eigenproblem is the ζ-pencil of
:mod:`~magnelio.ports._modal.zeta_pencil`; this module adds the
*continuation* along the axis.  A full arc search
(:func:`~magnelio.ports._modal.zeta_pencil.find_propagating_modes`,
five shift-invert factorisations) runs at the first and the last axis
point and wherever a recording channel has no mode; in between every
tracked mode is continued from its own previous eigenvalue with one
factorisation, which is what makes a 201-point axis affordable.

Reference impedance and power.  The phasors of the unit incident wave
give ``V`` at the port plane and ``I`` at the interior half-plane
(the recorder's sampling convention).  Brought to the same plane with
the wave's own half-cell factor ``√ζ``, their ratio is real on a
lossless line and is the impedance the wave carries::

    Z(f) = V_in · √ζ / I_in,     P(f) = ½ Re(V_in · conj(I_in / √ζ)).

``P`` is the power of the unit wave through the channel's own
profiles, and the decomposition scales every mode's phasors by
``1/√P`` so the fitted amplitudes are power waves in √W — which is
what makes ``S21`` between two ports of *different* cross-sections a
power ratio rather than a ratio of two unrelated eigenvector norms.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from magnelio.constants import C0
from magnelio.ports._modal import zeta_pencil as _zp
from magnelio.ports._modal.zeta_pencil import (
    _PROP_TOL,
    PeriodChain,
    PortCurlSlice,
    build_port_curl_slice,
    normalize_gauge,
    solve_zeta_modes,
)

# Minimum W-overlap for a continued eigenvector to count as the same
# mode (the band tracker's threshold), and for a mode to be assigned to
# a recording channel.
_OVERLAP_MIN = 0.5


@dataclass(frozen=True)
class PortDispersion:
    """The true discrete modes of one port along a frequency axis.

    Arrays are indexed ``[channel, frequency]``; a channel that has no
    propagating mode at a frequency (below its cut-on) is NaN there.
    ``zeta`` is the incident-wave eigenvalue in the inward indexing
    (``Im ζ ≤ 0`` travels into the domain); ``gamma = −ln ζ / dz`` is
    the continuum-equivalent propagation constant, ``α + jβ``.

    ``z_ref`` and ``power`` are taken through the channel's *own*
    recording profiles, half-window values on a port cut by a symmetry
    plane — the publication layer applies the symmetry scale.
    ``z_line`` is the power–current impedance ``2P/|I|²`` of the true
    mode from its own fields — the discrete Poynting flux through the
    port plane and the Ampère loop around the signal conductor — where
    the record carries the conductor; on a port with several signal
    conductors, or a hollow pipe, it falls back to ``z_ref``.

    The phasor blocks ``v_in`` … ``i_out`` are ``[frequency,
    recording channel, mode]`` with the mode index equal to the channel
    the mode was assigned to; unassigned columns are NaN.
    """

    f_axis: np.ndarray
    dz: float
    dt: float
    zeta: np.ndarray
    z_ref: np.ndarray
    power: np.ndarray
    epsilon_eff: np.ndarray
    z_line: np.ndarray
    v_in: np.ndarray = field(repr=False)
    i_in: np.ndarray = field(repr=False)
    v_out: np.ndarray = field(repr=False)
    i_out: np.ndarray = field(repr=False)
    incomplete: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_full_searches: int = 0

    @property
    def n_channels(self) -> int:
        return int(self.zeta.shape[0])

    @property
    def gamma(self) -> np.ndarray:
        """``α + jβ`` per channel [1/m]; NaN where the channel has no mode."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return -np.log(self.zeta) / self.dz

    @property
    def beta(self) -> np.ndarray:
        """Phase constant [rad/m]."""
        return self.gamma.imag

    @property
    def alpha(self) -> np.ndarray:
        """Attenuation constant [Np/m] — zero on a lossless chain to roundoff."""
        return self.gamma.real

    @property
    def assigned(self) -> np.ndarray:
        """Boolean ``[channel, frequency]`` — where a mode was found."""
        return np.isfinite(self.zeta.real)


def _chain_masses(record, m_eps, m_mu):
    """Plane-local masses in the recorder's convention.

    A record written since DD-244 carries them (the builder-local
    flattened values the operator projects with); older band records
    and the CW path slice the run's arrays.
    """
    plane = record.plane
    if getattr(record, "me_u", None) is not None:
        return (
            np.asarray(record.me_u, dtype=float),
            np.asarray(record.me_v, dtype=float),
            np.asarray(record.mh_u, dtype=float),
            np.asarray(record.mh_v, dtype=float),
        )
    if m_eps is None or m_mu is None:
        raise ValueError(
            f"port {record.name!r}: the dispersion record carries no plane masses "
            "and no M_eps/M_mu were given"
        )
    me = np.asarray(m_eps, dtype=float)
    mh = np.asarray(m_mu, dtype=float)
    return (
        me[plane.e_u_indices],
        me[plane.e_v_indices],
        mh[plane.h_u_indices],
        mh[plane.h_v_indices],
    )


def _curl_slice_of(record, m_mu, c_3d, mh_u, mh_v) -> PortCurlSlice:
    """The port's curl restriction — stored on the record, or built."""
    cs = getattr(record, "curl_slice", None)
    if cs is None:
        if c_3d is None or m_mu is None:
            raise ValueError(
                f"port {record.name!r}: the dispersion record carries no curl "
                "slice and no 3D curl was given"
            )
        cs = build_port_curl_slice(record.chain_inward, record.plane, m_mu, c_3d)
    mh_rows = np.concatenate([mh_u, mh_v])
    if mh_rows.size == cs.mh_rows.size and not np.array_equal(mh_rows, cs.mh_rows):
        cs = dataclasses.replace(cs, mh_rows=mh_rows)
    return cs


def _propagating(zs: np.ndarray, ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the on-circle branches, one per conjugate pair, ``Im ζ ≤ 0``."""
    prop = np.abs(np.abs(zs) - 1.0) <= _PROP_TOL
    zs, ps = zs[prop], ps[:, prop]
    keep: list[int] = []
    for j in range(zs.size):
        if any(abs(zs[j] - zs[i]) < 1e-9 or abs(zs[j] - np.conj(zs[i])) < 1e-9 for i in keep):
            continue
        keep.append(j)
    zs = zs[keep]
    ps = ps[:, keep]
    flip = zs.imag > 0.0
    zs = np.where(flip, np.conj(zs), zs)
    ps[:, flip] = np.conj(ps[:, flip])
    return zs, ps


def _wave_fields(
    phi: np.ndarray,
    zeta: complex,
    chain: PeriodChain,
    curl_slice: PortCurlSlice,
    n_u: int,
    n_v: int,
    w_dt: float,
):
    """Plane E and dual-face H of the unit incident wave.

    The synthesis of :func:`~magnelio.ports._modal.zeta_pencil.
    cw_wave_phasors`, split from its projections so one wave serves
    every recording channel of the port.
    """
    n_t = chain.n_t
    z_half = np.exp(0.5j * w_dt)
    e_syn = np.empty(curl_slice.n_period + n_t, dtype=complex)
    e_syn[: curl_slice.n_period] = phi
    e_syn[curl_slice.n_period :] = zeta * phi[:n_t]
    h_rows = -(chain.dt * (curl_slice.c_sub @ e_syn) / curl_slice.mh_rows) / (z_half - 1.0 / z_half)
    e_u_full = np.zeros(n_u, dtype=complex)
    e_v_full = np.zeros(n_v, dtype=complex)
    nu_free = int(chain.free_u.sum())
    e_u_full[chain.free_u] = phi[:nu_free]
    e_v_full[chain.free_v] = phi[nu_free:n_t]
    return e_u_full, e_v_full, h_rows[: curl_slice.n_h_u], h_rows[curl_slice.n_h_u :]


def solve_port_dispersion(
    record,
    f_axis: np.ndarray,
    *,
    m_eps: np.ndarray | None = None,
    m_mu: np.ndarray | None = None,
    c_3d: sp.spmatrix | None = None,
    search: str = "track",
) -> PortDispersion:
    """Solve the port's true discrete modes along ``f_axis``.

    Parameters
    ----------
    record : BandDecomposition
        The port's chain, plane, recording profiles and family hint
        (:class:`~magnelio.ports._modal.band_dtbc.BandDecomposition`,
        from a band or a modal operator).
    f_axis : array_like
        Frequencies [Hz], positive; any order.
    m_eps, m_mu, c_3d : optional
        The run's mesh-side operators, needed only for a record that
        predates the stored plane masses and curl slice.
    search : {"track", "full"}
        ``"track"`` continues every mode from its previous eigenvalue
        (one factorisation per tracked mode) and falls back to the
        full arc search at the first and last point and wherever a
        channel is unassigned or a tracked mode is lost.  ``"full"``
        runs the arc search at every point.

    Returns
    -------
    PortDispersion
    """
    if search not in ("track", "full"):
        raise ValueError(f"search must be 'track' or 'full', got {search!r}")
    f_in = np.asarray(f_axis, dtype=float).ravel()
    if f_in.size == 0 or np.any(f_in <= 0.0):
        raise ValueError("f_axis must contain positive frequencies")
    order = np.argsort(f_in, kind="stable")
    f_sorted = f_in[order]

    chain: PeriodChain = record.chain_inward
    plane = record.plane
    dt = float(chain.dt)
    dz = float(plane.normal_dx)
    n_t = chain.n_t
    w = chain.w_period
    w_t = w[:n_t]
    n_ch = int(record.n_modes)
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)

    me_u, me_v, mh_u, mh_v = _chain_masses(record, m_eps, m_mu)
    curl_slice = _curl_slice_of(record, m_mu, c_3d, mh_u, mh_v)

    # Recording channels: tangential e_t traces for the mode-to-channel
    # assignment, and the projection weights the recorder applies.
    ch_traces = []
    for e_u, e_v in zip(record.e_u_profiles, record.e_v_profiles):
        tr = np.concatenate([np.asarray(e_u)[chain.free_u], np.asarray(e_v)[chain.free_v]])
        ch_traces.append(tr / math.sqrt(float(np.dot(w_t, tr**2))))
    proj = [
        (me_u * np.asarray(du, dtype=float), me_v * np.asarray(dv, dtype=float))
        for du, dv in record.dual_e_profiles
    ]
    h_w = [
        (mh_u * np.asarray(hu, dtype=float), mh_v * np.asarray(hv, dtype=float))
        for hu, hv in zip(record.h_u_profiles, record.h_v_profiles)
    ]

    # Power–current impedance of the true mode: the Ampère loop around
    # every node of the one signal conductor (interior dual edges cancel
    # pairwise), as the transposed 2D gradient of the rotated dual
    # voltages; a K > 1 port carries no single modal current here.
    g_2d = getattr(record, "g_2d", None)
    signal_nodes = getattr(record, "signal_nodes", None)
    loop_t = None
    if g_2d is not None and signal_nodes is not None and len(signal_nodes) == 1:
        nodes = np.asarray(signal_nodes[0], dtype=np.int64)
        loop_t = sp.csr_matrix(g_2d)[:, nodes].sum(axis=1).A1  # per edge: Σ_n ±1

    fam_f = np.asarray(record.family_freqs, dtype=float)
    fam_theta = np.abs(np.angle(np.asarray(record.family_zetas, dtype=complex)))

    n_f = f_sorted.size
    zeta = np.full((n_ch, n_f), np.nan + 0j)
    z_ref = np.full((n_ch, n_f), np.nan)
    power = np.full((n_ch, n_f), np.nan)
    z_line = np.full((n_ch, n_f), np.nan)
    eps_eff = np.full((n_ch, n_f), np.nan)
    v_in = np.full((n_f, n_ch, n_ch), np.nan + 0j)
    i_in = np.full((n_f, n_ch, n_ch), np.nan + 0j)
    v_out = np.full((n_f, n_ch, n_ch), np.nan + 0j)
    i_out = np.full((n_f, n_ch, n_ch), np.nan + 0j)
    incomplete: list[float] = []
    n_full = 0

    # Continuation state: per channel the (ζ, φ, f) of its last mode.
    tracked: dict[int, tuple[complex, np.ndarray, float]] = {}

    for k, f in enumerate(f_sorted):
        w_dt = 2.0 * math.pi * f * dt
        full = search == "full" or k == 0 or k == n_f - 1 or len(tracked) < n_ch
        zp = pp = None
        if not full:
            targets = []
            for zeta_prev, _phi_prev, f_prev in tracked.values():
                # θ grows with f on every guided mode; extrapolate the
                # phase advance linearly for the shift.
                targets.append(complex(np.exp(np.log(zeta_prev) * (f / f_prev))))
            zs, ps = solve_zeta_modes(chain, w_dt, targets, k=4)
            zs, ps = _propagating(zs, ps)
            matched: dict[int, int] = {}
            taken = np.zeros(zs.size, dtype=bool)
            for c, (_z, phi_prev, _f) in tracked.items():
                if zs.size == 0:
                    break
                ov = np.abs(np.conj(phi_prev) @ (w[:, None] * ps))
                ov[taken] = -1.0
                j = int(np.argmax(ov))
                if ov[j] < _OVERLAP_MIN:
                    continue
                taken[j] = True
                matched[c] = j
            if len(matched) == len(tracked):
                zp, pp = zs, ps
                assigned = list(matched.items())
            else:
                full = True
        if full:
            n_full += 1
            hint = 1.3 * float(np.interp(f, fam_f, fam_theta))
            # Looked up on the module so the search can be substituted.
            zp, pp = _zp.find_propagating_modes(chain, w_dt, hint)
            if zp.size == 0:
                tracked = {}
                continue
            taken = np.zeros(zp.size, dtype=bool)
            assigned = []
            for c in range(n_ch):
                ov = np.abs(ch_traces[c] @ (w_t[:, None] * pp[:n_t, :]))
                ov[taken] = -1.0
                j = int(np.argmax(ov))
                if ov[j] < _OVERLAP_MIN:
                    continue
                taken[j] = True
                assigned.append((c, j))
            if zp.size > len(assigned):
                incomplete.append(float(f))
        tracked = {}
        for c_mode, j in assigned:
            z_j = complex(zp[j])
            phi = normalize_gauge(pp[:, j], n_t)
            tracked[c_mode] = (z_j, phi, float(f))
            e_u_full, e_v_full, h_u_wave, h_v_wave = _wave_fields(
                phi, z_j, chain, curl_slice, n_u, n_v, w_dt
            )
            for c_rec in range(n_ch):
                pu, pv = proj[c_rec]
                hu, hv = h_w[c_rec]
                vi = complex(np.dot(pu, e_u_full) + np.dot(pv, e_v_full))
                ii = complex(np.dot(hu, h_u_wave) + np.dot(hv, h_v_wave))
                v_in[k, c_rec, c_mode] = vi
                i_in[k, c_rec, c_mode] = ii
                # The reflected wave is the conjugate eigenpair; the
                # leapfrog H phasor's purely imaginary denominator
                # flips sign under conjugation (i_out = -conj(i_in)).
                v_out[k, c_rec, c_mode] = np.conj(vi)
                i_out[k, c_rec, c_mode] = -np.conj(ii)
            sq = np.sqrt(z_j)
            vi = v_in[k, c_mode, c_mode]
            ii = i_in[k, c_mode, c_mode]
            zeta[c_mode, k] = z_j
            z_ref[c_mode, k] = float((vi * sq / ii).real)
            power[c_mode, k] = float(0.5 * (vi * np.conj(ii / sq)).real)
            if loop_t is not None:
                # ê at the plane against ĥ at the half-plane brought to
                # the plane: P = ½ Re Σ (ê_u ĥ_v* − ê_v ĥ_u*), I = ∮ ĥ.
                h_v0 = h_v_wave / sq
                h_u0 = h_u_wave / sq
                p_true = 0.5 * float(
                    (np.dot(e_u_full, np.conj(h_v0)) - np.dot(e_v_full, np.conj(h_u0))).real
                )
                rot = np.concatenate([-h_v0, h_u0])
                i_true = complex(np.dot(loop_t, rot))
                if abs(i_true) > 0.0 and p_true > 0.0:
                    z_line[c_mode, k] = 2.0 * p_true / abs(i_true) ** 2
            if not np.isfinite(z_line[c_mode, k]):
                z_line[c_mode, k] = z_ref[c_mode, k]
            theta = abs(float(np.angle(z_j)))
            s_ratio = (math.sin(theta / 2.0) / (dz / 2.0)) / (math.sin(w_dt / 2.0) / (dt / 2.0))
            eps_eff[c_mode, k] = (C0 * s_ratio) ** 2

    inv = np.empty_like(order)
    inv[order] = np.arange(n_f)
    return PortDispersion(
        f_axis=f_in.copy(),
        dz=dz,
        dt=dt,
        zeta=zeta[:, inv],
        z_ref=z_ref[:, inv],
        power=power[:, inv],
        z_line=z_line[:, inv],
        epsilon_eff=eps_eff[:, inv],
        v_in=v_in[inv],
        i_in=i_in[inv],
        v_out=v_out[inv],
        i_out=i_out[inv],
        incomplete=np.asarray(incomplete, dtype=float),
        n_full_searches=n_full,
    )


def decompose_power_waves(
    disp: PortDispersion,
    V_hat: np.ndarray,
    I_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit recorded channel spectra to the port's true-mode phasors.

    Solves, at every frequency, the joint least-squares system over all
    recording channels ``c`` and all assigned modes ``j``::

        V_c = Σ_j a_j v_in[c, j] + b_j v_out[c, j]
        I_c = Σ_j a_j i_in[c, j] + b_j i_out[c, j]

    with every mode's phasors scaled by ``1/√P_j`` so ``a``, ``b`` are
    power waves in √W (module docstring).

    Parameters
    ----------
    disp : PortDispersion
    V_hat, I_hat : np.ndarray, shape (n_channels, n_f)
        Recorded channel spectra; ``I_hat`` already carries the Yee
        half-step rotation ``e^{+jω dt/2}``.

    Returns
    -------
    (a, b) : np.ndarray, shape (n_channels, n_f)
        NaN on channels without a mode at that frequency.
    """
    n_ch, n_f = disp.zeta.shape
    a = np.full((n_ch, n_f), np.nan + 0j)
    b = np.full((n_ch, n_f), np.nan + 0j)
    for k in range(n_f):
        modes = np.flatnonzero(disp.assigned[:, k])
        if modes.size == 0:
            continue
        scale = 1.0 / np.sqrt(disp.power[modes, k])
        v_in = disp.v_in[k][:, modes] * scale[None, :]
        v_out = disp.v_out[k][:, modes] * scale[None, :]
        i_in = disp.i_in[k][:, modes] * scale[None, :]
        i_out = disp.i_out[k][:, modes] * scale[None, :]
        lhs = np.block([[v_in, v_out], [i_in, i_out]])
        rhs = np.concatenate([V_hat[:, k], I_hat[:, k]])
        ab, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        n_j = modes.size
        a[modes, k] = ab[:n_j]
        b[modes, k] = ab[n_j:]
    return a, b
