"""WP-R1 gate: exact discrete transparent boundary condition, offline.

Validates the reflection-free-plan method core (WP-R1; plan retired
to git history, see DD-054) on the synthetic
modal 1D lattice only — no solver code.  Per port mode the interior
update on a uniform feed line is the leapfrog Klein-Gordon chain

    u^{n+1}_k = 2u^n_k - u^{n-1}_k
                + r^2 (u^n_{k+1} - 2u^n_k + u^n_{k-1}) - q^2 u^n_k

(r = modal Courant number, q = discrete cut-off * dt; TEM: q = 0).
The exact DTBC is the ghost relation  u_ghost(z) = lambda(z) * u_bnd(z)
with  lambda = A - sqrt(A^2 - 1),  A(z) = 1 + (z - 2 + 1/z + q^2)/(2 r^2),
branch |lambda| <= 1 outside the unit circle.  The kernel l_m (Laurent
coefficients of lambda around infinity, l_0 = 0) is generated to
machine precision by contour integration; the reflection of an
approximated boundary symbol lambda~ is exactly

    Gamma(w) = (lambda~ - lambda) / (1/lambda - lambda~)
    at z = e^{i w dt}.

Findings (session 84; all numbers n_kernel = 4096 unless stated):

1.  **The Gamma formula is exact.**  Monochromatic steady-state
    lock-in measurements on the chain agree with the formula to
    0.1 dB at every tested frequency, TEM and KG, across the band
    (e.g. r=0.9/q=0.2: -79.1 vs -79.1 dB at the 1.01 f_c edge bin,
    -124.9 vs -124.8 dB mid-band).  The a-priori bound IS the floor.
2.  **Truncation converges as n_kernel^{-3/2}** (-27 dB per 8x), with
    the maximum always at the lower band edge (the lambda branch
    point: cut-off for KG, DC for TEM).  Raw truncation reaches
    -100 dB at 1.01 f_c with n_kernel ~ 2.6e5 — fine offline,
    motivating a compressed (rational / sum-of-exponentials) form in
    production.
3.  **Truncation is weakly active** (max |lambda~| - 1 ~ +1e-3 on the
    unit circle): noise-excited long runs show ~+1 dB/1e5-step drift
    on TEM chains.  Production fits must be passivity-enforced; a
    certified fit with error eps can always be made passive by
    scaling with (1 - eps) at total error <= 2 eps, so passive
    approximations at the -100 dB level exist constructively.
4.  Near-DC / evanescent robustness: a drive band pushed below the
    evaluation band (down to w*dt = 0.005, resp. below cut-off)
    stays bounded and decays.

Earlier pulse-based reflectometry attempts are recorded as failed
measurement methodology (displacement sources double-integrate on the
chain — near-DC blow-up; finite-record leakage floors the FFT ratio);
the CW lock-in method replaced them.

Run:  python validation/dtbc_kernel_spike.py
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Symbol and kernel
# ---------------------------------------------------------------------------


def lam_exact(z: np.ndarray, r: float, q: float) -> np.ndarray:
    """Outgoing root lambda(z) with |lambda| <= 1 for |z| >= 1."""
    A = 1.0 + (z - 2.0 + 1.0 / z + q * q) / (2.0 * r * r)
    root = np.sqrt(A * A - 1.0)
    lam_minus = A - root
    lam_plus = A + root
    return np.where(np.abs(lam_minus) <= np.abs(lam_plus), lam_minus, lam_plus)


def dtbc_kernel(r: float, q: float, n_kernel: int) -> np.ndarray:
    """Kernel l_m, m = 0 .. n_kernel-1, via contour integration.

    lambda(z) = sum_m l_m z^{-m};  on z = rho e^{i theta} the l_m are
    rho^m times the Fourier coefficients of lambda along the circle.
    rho > 1 keeps the branch selection unambiguous; it is chosen so
    that rho^{n_kernel} stays O(e^4) (bounded roundoff amplification).
    """
    n_fft = 8 * n_kernel
    rho = math.exp(4.0 / n_kernel)
    theta = 2.0 * np.pi * np.arange(n_fft) / n_fft
    z = rho * np.exp(1j * theta)
    coeff = np.fft.ifft(lam_exact(z, r, q))
    m = np.arange(n_kernel)
    return np.real(coeff[:n_kernel]) * rho**m


def lam_truncated(kern: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the truncated series sum_m l_m z^{-m}."""
    m = np.arange(kern.size)
    return np.array([np.sum(kern * zz ** (-m)) for zz in z])


def apriori_reflection(r: float, q: float, kern: np.ndarray, w_dt: np.ndarray) -> np.ndarray:
    """|Gamma(w)| of the truncated kernel on z = e^{i w dt}.

    The contour sits at |z| = 1 + 1e-8: far enough off the circle to
    keep the outgoing-root selection unambiguous against roundoff,
    close enough that the induced bias (~1e-8 / v_g) stays below
    -130 dB even at the 1.01 f_c evaluation edge.
    """
    z = np.exp(1j * w_dt) * (1.0 + 1e-8)
    lam = lam_exact(z, r, q)
    lam_t = lam_truncated(kern, z)
    return np.abs(lam_t - lam) / np.abs(1.0 / lam - lam_t)


# ---------------------------------------------------------------------------
# Synthetic chain with DTBC ends
# ---------------------------------------------------------------------------


def run_chain(
    K: int,
    n_steps: int,
    r: float,
    q: float,
    kern: np.ndarray | None,
    k_src: int,
    probes: tuple[int, ...],
    src: np.ndarray,
) -> np.ndarray:
    """Leapfrog KG chain; kern=None -> plain interior ends (reference).

    Returns the probe time series, shape (n_steps, len(probes)).  With
    a kernel, both ends carry the DTBC ghost update (right ghost from
    the history of u[K-1], left ghost from the history of u[0] — the
    same kernel by symmetry).
    """
    u_prev = np.zeros(K)
    u = np.zeros(K)
    hist_l = np.zeros(n_steps + 1)
    hist_r = np.zeros(n_steps + 1)
    out = np.zeros((n_steps, len(probes)))
    r2, q2 = r * r, q * q
    n_k = 0 if kern is None else kern.size
    kflip = None if kern is None else kern[1:][::-1].copy()
    probes_arr = np.array(probes, dtype=int)

    for n in range(n_steps):
        if kern is not None and n > 0:
            lo = max(0, n - (n_k - 1))
            ghost_l = float(np.dot(kflip[-(n - lo) :], hist_l[lo:n]))
            ghost_r = float(np.dot(kflip[-(n - lo) :], hist_r[lo:n]))
        else:
            ghost_l = ghost_r = 0.0

        lap = np.empty(K)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        lap[0] = u[1] - 2.0 * u[0] + ghost_l
        lap[-1] = ghost_r - 2.0 * u[-1] + u[-2]

        u_next = 2.0 * u - u_prev + r2 * lap - q2 * u
        u_next[k_src] += src[n]
        u_prev, u = u, u_next

        hist_l[n + 1] = u[0]
        hist_r[n + 1] = u[K - 1]
        out[n] = u[probes_arr]
    return out


def gaussian_pulse(n_steps: int, w_lo: float, w_hi: float) -> np.ndarray:
    """Band-limited displacement drive (modulated Gaussian).

    Second time difference applied: a displacement source on the
    leapfrog chain acts as a double integrator (response ~ 1/w^2),
    which would blow residual DC content up into a dominant near-DC
    ring; the w^2 of the second difference cancels that exactly.
    """
    n = np.arange(n_steps)
    w0 = 0.5 * (w_lo + w_hi)
    sig = 6.0 / max(w_hi - w_lo, 1e-9)
    t0 = 6.0 * sig
    env = np.exp(-0.5 * ((n - t0) / sig) ** 2)
    pulse = env * np.cos(w0 * (n - t0))
    out = np.zeros(n_steps)
    out[2:] = np.diff(pulse, 2)
    return out


def cw_gamma(
    r: float,
    q: float,
    w: float,
    kern: np.ndarray,
    K: int = 240,
    n_settle: int = 12000,
    n_avg: int = 4096,
) -> float:
    """Steady-state |Gamma| via lock-in standing-wave decomposition.

    CW drive (ramped, displacement pre-emphasised), complex phasors at
    two sites near the right boundary via lock-in over the final
    n_avg samples, then the incident/reflected split against the
    exact lattice wavenumber lambda(w).
    """
    n_steps = n_settle + n_avg
    n = np.arange(n_steps)
    ramp = np.minimum(n / 3000.0, 1.0)
    carrier = ramp * np.cos(w * n)
    src = np.zeros(n_steps)
    src[2:] = np.diff(carrier, 2)

    sites = (K - 12, K - 11)
    rec = run_chain(K, n_steps, r, q, kern, 40, sites, src)
    t = np.arange(n_avg)
    ph = np.exp(-1j * w * (n_steps - n_avg + t))
    U = 2.0 * np.mean(rec[-n_avg:] * ph[:, None], axis=0)

    z = np.exp(1j * w) * (1.0 + 1e-8)
    lam = lam_exact(np.array([z]), r, q)[0]
    k1, k2 = sites
    A = np.array([[lam**k1, lam ** (-k1)], [lam**k2, lam ** (-k2)]])
    a, b = np.linalg.solve(A, U)
    return float(abs(b / a) * abs(lam) ** (-2 * (K - 1)))


# ---------------------------------------------------------------------------
# Working-point evaluation
# ---------------------------------------------------------------------------


def evaluate(r: float, q: float, *, n_kernel: int = 4096, K: int = 240) -> None:
    w_c = 2.0 * math.asin(q / 2.0) if q > 0 else 0.0
    w_half = 2.0 * math.asin(math.sqrt(r * r * 0.5 + q * q / 4.0))
    # Practical band: from 1.01 f_c up to beta*dz = pi/2 (4 cells per
    # wavelength — coarser than any physical mesh).
    w_lo_eval = max(1.01 * w_c, 0.02)
    w_hi_eval = w_half

    kern = dtbc_kernel(r, q, n_kernel)

    # (1) a-priori bound over the band
    w_chk = np.linspace(w_lo_eval, w_hi_eval, 101)
    gam = apriori_reflection(r, q, kern, w_chk)
    bound_db = 20.0 * math.log10(max(float(gam.max()), 1e-300))

    # (2) CW verification: measured steady-state |Gamma| vs formula at
    # 5 frequencies including both band edges (lock-in-grid snapped).
    devs, meas_edge = [], None
    for frac in (0.0, 0.1, 0.35, 0.65, 1.0):
        w_raw = w_lo_eval + frac * (w_hi_eval - w_lo_eval)
        m_bin = max(1, round(w_raw * 4096 / (2.0 * np.pi)))
        w = 2.0 * np.pi * m_bin / 4096
        g_meas = cw_gamma(r, q, w, kern, K=K)
        g_pred = float(apriori_reflection(r, q, kern, np.array([w]))[0])
        devs.append(
            abs(20.0 * math.log10(max(g_meas, 1e-300)) - 20.0 * math.log10(max(g_pred, 1e-300)))
        )
        if frac == 0.0:
            meas_edge = 20.0 * math.log10(max(g_meas, 1e-300))

    # (3) passivity of the truncated kernel on the unit circle
    th = np.linspace(1e-4, np.pi, 2001)
    lam_t_circ = lam_truncated(kern, np.exp(1j * th))
    passivity = float(np.abs(lam_t_circ).max()) - 1.0

    # (4) near-DC / evanescent robustness: drive deliberately below
    # the evaluation band (the hardest content for the boundary),
    # judge boundedness of the late tail.
    n_long = 100_000
    src_ev = np.zeros(n_long)
    src_ev[:16384] = gaussian_pulse(16384, 0.005, w_hi_eval)
    probe_ev = run_chain(K, n_long, r, q, kern, 60, (150,), src_ev)[:, 0]
    peak = float(np.abs(probe_ev).max())
    tail = float(np.abs(probe_ev[-5_000:]).max())
    tail_db = 20.0 * math.log10(max(tail / peak, 1e-300))

    # (5) strict growth probe: broadband noise burst, no further
    # source, probe-amplitude trend over 1e5 steps.
    rng = np.random.default_rng(7)
    src_n = np.zeros(n_long)
    src_n[:64] = 1e-6 * rng.standard_normal(64)
    probe_st = run_chain(K, n_long, r, q, kern, 60, (150,), src_n)[:, 0]
    early = float(np.abs(probe_st[5_000:15_000]).max())
    late = float(np.abs(probe_st[-10_000:]).max())
    stab_db = 20.0 * math.log10(max(late / max(early, 1e-300), 1e-300))

    fc_txt = f"{w_c:.3f}" if q > 0 else "  DC "
    print(
        f"  r={r:4.2f} q={q:4.2f} (w_c*dt {fc_txt}, band "
        f"[{w_lo_eval:.3f},{w_hi_eval:.3f}]): a-priori {bound_db:7.1f} dB"
        f" | CW@edge {meas_edge:7.1f} dB, max dev {max(devs):4.2f} dB"
        f" | passivity {passivity:+.1e}"
        f" | near-DC tail {tail_db:6.1f} dB | noise-run {stab_db:6.1f} dB"
    )


def kernel_convergence(r: float, q: float) -> None:
    """A-priori bound vs kernel length (band as in evaluate)."""
    w_c = 2.0 * math.asin(q / 2.0) if q > 0 else 0.0
    w_half = 2.0 * math.asin(math.sqrt(r * r * 0.5 + q * q / 4.0))
    w_chk = np.linspace(max(1.01 * w_c, 0.02), w_half, 101)
    parts = []
    for n_k in (4096, 32768, 262144):
        gam = apriori_reflection(r, q, dtbc_kernel(r, q, n_k), w_chk)
        i_max = int(np.argmax(gam))
        parts.append(
            f"n={n_k}: {20 * math.log10(float(gam.max())):6.1f} dB @ w*dt={w_chk[i_max]:.3f}"
        )
    print(f"  r={r:4.2f} q={q:4.2f}:  " + "  |  ".join(parts))


def main() -> None:
    print("DTBC offline gate (truncated exact kernel, n_kernel = 4096)")
    print(
        "CW@edge = steady-state measurement at the 1.01 f_c edge bin; "
        "max dev = |measured - formula| over 5 band points"
    )
    for r, q in ((0.30, 0.0), (0.70, 0.0), (0.99, 0.0), (0.50, 0.30), (0.90, 0.20), (0.70, 0.50)):
        evaluate(r, q)
    print(
        "\nkernel-truncation convergence of the a-priori bound (location of the max in brackets):"
    )
    for r, q in ((0.70, 0.0), (0.99, 0.0), (0.50, 0.30), (0.90, 0.20)):
        kernel_convergence(r, q)


if __name__ == "__main__":
    main()
