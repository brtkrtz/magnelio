"""WP-R4b certificate gate — band-subspace-projected DTBC
(reflection-free plan WP-R4b, retired to git history, see DD-057;
R1-style spike BEFORE any solver
code, sessions-63-65 ground rule; falls back to WP-R4a if a
certificate fails at the -100 dB level).

Question: can a *broadband* TD termination for inhomogeneous lines be
certified to |S11| < -100 dB when the exact matrix DTBC kernel is
compressed onto the band subspace of the tracked mode families
(WP-R4 finding: raw kernel tail FULL-rank, but the profile family
over the band has rank 5-7 at 1e-8)?

Method core — exact a-priori reflection of ANY boundary symbol
----------------------------------------------------------------

The matrix generalisation of the R1 gate formula.  At ``z = e^{i w
dt}`` (``sig_hat`` real) the quadratic pencil's finite spectrum
splits into N_t outgoing branches (|zeta| < 1, or on-circle with
``Im zeta < 0``) and their N_t incoming partners.  For a unit
incident mode ``(zeta_0, phi_0)`` and a boundary that closes the
chain with the ghost relation ``e_t(K+1) = Lambda~_t(z) x_K``, the
total steady field is

    x_k = phi_0 zeta_0^k + sum_j c_j chi_j mu_j^k        (K = 0)

with ``(mu_j, chi_j)`` the incoming branches, and the ONLY equations
that differ from the exact continuation are the ghost rows:

    sum_j c_j (mu_j chi_{j,t} - Lambda~_t chi_j)
        = Lambda~_t phi_0 - zeta_0 phi_{0,t},

an N_t x N_t solve per frequency.  ``|Gamma| = |c_{j0}|`` at the
incident mode's mirror branch (both W-normalised).  This evaluates
the exact modal reflection of any candidate symbol on a dense grid
before any TD run — certificates (i) and (ii) become computable
a-priori.  Validation inside this spike: the exact ``Lambda`` gives
|Gamma| at machine zero, and the WP-R4a frequency-local scalar
symbol reproduces the WP-R4 single-profile numbers (-10..-60 dB at
the band ends) — two independent anchors.

Candidate production form (certified object of this gate)
----------------------------------------------------------

    Lambda~_t(z) = U U^T W_t  Lambda_t(z)  V V^T W          (+ head)

with ``V`` the W-orthonormal basis of the mode-family traces over
the band (fundamental + second family above its cut-on, SVD rank p)
and ``U`` the W_t-orthonormal basis of their e_t parts.  In the time
domain this is a p-channel convolution: ``ghost^n = U sum_m q_m
(V^T W x_K^{n-m})`` with the projected kernel ``q_m = U^T W_t L_m V``
— O(p^2 n_hist + p N) per step instead of O(N_t N n_hist).  The
SOE/rational compression of the p x p channel kernel is the
*implementation* WP's concern (the R2 precedent: exact first,
compress later); this gate certifies the projection ceiling.

Results (session 89) — gate PASSED
-----------------------------------

* Evaluator anchors: the exact ``Lambda`` measures |Gamma| ~ 1e-7
  (the _RHO_OFF evaluation floor, as designed); the WP-R4a scalar
  symbol frozen at band centre reproduces the WP-R4 single-profile
  refutation (-11.6 / -73.8 / -32.4 dB across the layered band,
  -45.4 / -219.6 / -51.6 dB block).
* Certificate (i) — subspace coupling, a-priori over the band
  (evaluation floored near -130 dB by _RHO_OFF; the TD leg goes
  deeper):

      layered  p = 4 / 6 / 8 / 10 / 12:
          -37.4 / -68.4 / -113.7 / -127.1 / -129.0 dB
        (p = 12 detail: fundamental -129.0 .. -148.5 dB; second
         family -113.4 dB AT its cut-on grid point, -126.1 .. -142.7
         from 1.05x up — the criterion applies from 1.01*f_c_hat)
      block    p = 4 already -133.6 dB (single family, fast SVD
        decay 1e-2 per vector; layered needs p ~ 10-12 because two
        real families span Re/Im trace parts)

* Certificate (ii) — TD floors through the coupled boundary
  (CW lock-in on the chain, kernel exact within the run):

      layered  1.28 / 4.40 / 7.52 GHz (fundamental):
          -128.8 / -154.0 / -174.0 dB
      layered  6.38 GHz (second mode incident, two propagating):
          -116.3 dB
      block    2.27 / 4.15 / 6.03 GHz: -228.0 / -234.5 / -237.5 dB

* Certificate (iii) — passivity.  The NAIVE kernel projection
  ``U U^T W_t Lambda V V^T W`` is weakly ACTIVE (negative control:
  noise-probe window maxima grow 2.3x (block) to 66x (layered) at
  4094 steps) — projecting the kernel destroys the half-lattice
  passivity.  The Galerkin exterior is lossless BY CONSTRUCTION
  (W-symmetric projection; the coupled interior+exterior system is
  block-symmetric) and measures decaying: window-max ratios <= 1.12
  early-transient / <= 0.99 late, final/peak 0.91; out-of-band
  |Gamma| <= 1.2e-6.

Verdict: gate passed with the production form FIXED as the
Galerkin-projected exterior + exact small-system DTBC.  Everything
the production implementation needs runs at size p (subspace from
the WP-R4a sparse mode-family solves; contour QZ at 2p = 20-24
instead of 2N — seconds, not hours, at production cross-sections).
The implementation WP carries: subspace/kernel construction in
``zeta_pencil``, a p-channel boundary operator (projected boundary
period + small-kernel convolution, the PortOperatorModal pattern),
pulsed broadband S-parameters with the WP-R4a per-frequency
decomposition in post-processing, and the R2-style kernel
auto-extension (SOE compression only if profiling demands it).

Run:  python validation/qtem_band_dtbc_certificate_spike.py
      [--case layered|block|all] [--fast]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtem_cw_precheck_spike import (  # noqa: E402
    chain_fit,
    normalize_gauge,
)
from qtem_dtbc_method_spike import (  # noqa: E402
    lockin_two_wave,
    make_case,
    matrix_kernel,
    stable_solvent,
)

# Off-circle offset for the exact-DtN evaluation (the on-circle
# dichotomy is ambiguous for propagating branches; the induced bias
# floors the certificate EVALUATION near -130 dB — deep enough to
# certify the -100 dB criterion, and the TD leg validates below it).
_RHO_OFF = 1e-8

from magnelio._operators.curl import build_curl_matrix  # noqa: E402
from magnelio.ports._modal.dtbc import lambda_symbol  # noqa: E402

_ONC_TOL = 1e-7


# ----------------------------------------------------------------------
# On-circle branch machinery
# ----------------------------------------------------------------------


def all_branches(D, sig_hat):
    """Finite eigenpairs at real ``sig_hat``, W-unnormalised."""
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    Ap = np.zeros((2 * n, 2 * n), dtype=complex)
    Bp = np.zeros((2 * n, 2 * n), dtype=complex)
    Ap[:n, n:] = np.eye(n)
    Ap[n:, :n] = -D_m1
    Ap[n:, n:] = -(D_0 - sig_hat * np.eye(n))
    Bp[:n, :n] = np.eye(n)
    Bp[n:, n:] = D_p1
    vals, vecs = sla.eig(Ap, Bp)
    fin = np.isfinite(vals) & (np.abs(vals) > 1e-12)
    return vals[fin], vecs[:n, fin]


def split_branches(vals, vecs, w_eps):
    """(outgoing, incoming) finite branch sets, W-normalised.

    Outgoing (toward the exterior, +k): |zeta| < 1 - tol, or
    on-circle with Im zeta < 0.  Incoming: the complement.
    """
    norms = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(vecs), w_eps, vecs)))
    vecs = vecs / norms[None, :]
    a = np.abs(vals)
    out = (a < 1.0 - _ONC_TOL) | ((np.abs(a - 1.0) <= _ONC_TOL) & (vals.imag < 0.0))
    return (vals[out], vecs[:, out]), (vals[~out], vecs[:, ~out])


def exact_lambda_t(D, w_dt, n_t):
    """Exact DtN (e_t rows) just off the circle (ordered-QZ solvent)."""
    z = (1.0 + _RHO_OFF) * np.exp(1j * w_dt)
    sig = 2.0 - z - 1.0 / z
    lam = stable_solvent(D[0], D[1], D[2], complex(sig))
    return lam[:n_t, :]


def modal_reflection(lam_t, D, sig_hat, w_eps, n_t, phi_inc, zeta_inc):
    """Exact |Gamma| of the boundary symbol for one incident mode."""
    vals, vecs = all_branches(D, sig_hat)
    _, (mi, chi) = split_branches(vals, vecs, w_eps)
    if mi.size != n_t:
        raise RuntimeError(f"incoming set has {mi.size} branches, expected {n_t}")
    lhs = (mi[None, :] * chi[:n_t, :]) - lam_t @ chi
    rhs = lam_t @ phi_inc - zeta_inc * phi_inc[:n_t]
    c, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    j0 = int(np.argmin(np.abs(mi - 1.0 / zeta_inc)))
    if abs(mi[j0] - 1.0 / zeta_inc) > 1e-6:
        raise RuntimeError("mirror branch of the incident mode not found in the incoming set")
    return float(abs(c[j0])), c, mi


def band_modes(D, dt, w_eps, n_t, f_grid):
    """Tracked mode families over the band.

    Returns ``{family: [(f, zeta, phi)]}`` for the fundamental (0)
    and — where propagating — the second family (1), tracked by
    W-overlap continuation from the lowest frequency.
    """
    track = {0: None, 1: None}
    fams: dict[int, list] = {0: [], 1: []}
    for f in f_grid:
        sig_hat = 2.0 - 2.0 * math.cos(2.0 * math.pi * f * dt)
        vals, vecs = all_branches(D, sig_hat)
        (zo, po), _ = split_branches(vals, vecs, w_eps)
        on = np.abs(np.abs(zo) - 1.0) <= _ONC_TOL
        zp, pp = zo[on], po[:, on]
        if zp.size == 0:
            continue
        order = np.argsort(-np.abs(np.angle(zp)))
        zp, pp = zp[order], pp[:, order]
        for fam in (0, 1):
            if track[fam] is None:
                pick = fam if zp.size > fam else None
                if fam == 1 and zp.size < 2:
                    pick = None
            else:
                ov = np.abs(np.conj(track[fam]) @ (w_eps[:, None] * pp))
                pick = int(np.argmax(ov)) if zp.size else None
                if pick is not None and ov[pick] < 0.5:
                    pick = None
            if pick is None:
                continue
            phi = normalize_gauge(pp[:, pick], n_t)
            track[fam] = phi
            fams[fam].append((f, complex(zp[pick]), phi))
    return fams


def band_subspace(fams, w_eps, n_t, p):
    """(V, U): REAL W-orthonormal trace / W_t-orthonormal e_t bases.
    Real span of the family (Re phi, Im phi columns) — the TD kernel
    must be real, so the subspace is built real; the rank roughly
    doubles against the complex family rank (e_t is ~real, e_z ~pure
    imaginary in the fixed gauge, so both parts carry content).
    """
    cols = [part for fam in fams.values() for (_, _, phi) in fam for part in (phi.real, phi.imag)]
    A = np.column_stack(cols)
    sw = np.sqrt(w_eps)
    Uw, s, _ = np.linalg.svd(sw[:, None] * A, full_matrices=False)
    s = np.where(s > 0, s, 1e-300)
    V = Uw[:, :p] / sw[:, None]
    swt = np.sqrt(w_eps[:n_t])
    Ut, st, _ = np.linalg.svd(swt[:, None] * A[:n_t, :], full_matrices=False)
    U = Ut[:, : min(p, Ut.shape[1])] / swt[:, None]
    return V, U, s / s[0]


def projected_symbol(lam_t, V, U, w_eps, n_t):
    """Naive kernel projection ``U U^T W_t Lambda V V^T W``.

    Kept as the measured NEGATIVE control: this form is weakly
    ACTIVE in the time domain (part 4) — projecting the *kernel*
    does not preserve the passivity of the half-lattice.
    """
    PU = U @ (U.T * w_eps[None, :n_t])
    PV = V @ (V.T * w_eps[None, :])
    return PU @ lam_t @ PV


def galerkin_exterior(D, V, w_eps):
    """Galerkin-projected exterior blocks (p x p) + the coupling map.

    ``D~ = V^T W D V`` inherits the palindromic W-symmetry
    (``(W D_p1)^T = W D_m1``  =>  ``D~_m1^T = D~_p1``), so the
    projected half-line is itself a lossless lattice: its exact
    small-system DTBC is passive BY CONSTRUCTION, and the coupled
    full-interior + projected-exterior system is block-symmetric
    lossless — the structural fix for the active naive kernel
    projection.  Bonus: every contour/eigen operation runs at size
    2p instead of 2N, which makes the broadband form cheap at
    production scale.
    """
    VtW = (V * w_eps[:, None]).T
    Dt = tuple(VtW @ Dk @ V for Dk in D)
    return Dt, VtW


def galerkin_symbol_t(D, Dt, V, VtW, z, n_t):
    """Effective boundary symbol of the Galerkin exterior.

    Interface at period K: periods < K are the full interior, the
    boundary period K and everything beyond live in the subspace.
    Eliminating the projected half-line (small exact solvent) gives

        e_t(K) = -V_t [ (D~_0 - sig) + D~_p1 Lam_s(z) ]^{-1}
                      V^T W D_m1  x_{K-1},

    directly usable in :func:`modal_reflection` (the boundary period
    is one plane earlier — a relabeling).
    """
    sig = 2.0 - z - 1.0 / z
    lam_s = stable_solvent(Dt[0], Dt[1], Dt[2], complex(sig))
    p = Dt[1].shape[0]
    G = np.linalg.inv((Dt[1] - sig * np.eye(p)) + Dt[2] @ lam_s)
    return -(V[:n_t, :] @ G @ (VtW @ D[0]))


# ----------------------------------------------------------------------
# Part 1+2 — evaluator anchors + projection ceiling
# ----------------------------------------------------------------------


def part12(name, D, dt, w_eps, n_t, f_band, f_ref, p_list, fast):
    print(f"  [{name}] part 1 — reflection-evaluator anchors")
    f_grid = np.linspace(f_band[0], f_band[1], 9 if fast else 25)
    fams = band_modes(D, dt, w_eps, n_t, f_grid)
    n2 = len(fams[1])
    print(
        f"    families: fundamental {len(fams[0])} pts, "
        f"second {n2} pts" + (f" (cut-on ~{fams[1][0][0] / 1e9:.2f} GHz)" if n2 else "")
    )

    # Anchor A: exact Lambda -> |Gamma| at machine zero.
    f, zeta, phi = fams[0][len(fams[0]) // 2]
    sig_hat = 2.0 - 2.0 * math.cos(2.0 * math.pi * f * dt)
    lam_ex = exact_lambda_t(D, 2.0 * math.pi * f * dt, n_t)
    g_ex, _, _ = modal_reflection(lam_ex, D, sig_hat, w_eps, n_t, phi, zeta)
    # Anchor B: the WP-R4a frequency-local scalar symbol, frozen at
    # f_ref, evaluated across the band — must reproduce the WP-R4
    # single-profile refutation numbers.
    fr, zr, pr = min(fams[0], key=lambda t: abs(t[0] - f_ref))
    w_dt_r = 2.0 * math.pi * fr * dt
    r2f, _, q2f = chain_fit(zr, pr, D, 2.0 - 2.0 * math.cos(w_dt_r), w_eps, w_dt_r)
    r_fit, q_fit = math.sqrt(r2f), math.sqrt(max(q2f, 0.0))
    prt = pr[:n_t].real
    prt = prt / math.sqrt(float(np.dot(w_eps[:n_t], prt**2)))
    outer = np.outer(prt, prt * w_eps[:n_t])
    print(f"    anchor A (exact Lambda): |Gamma| = {g_ex:.2e}")
    print(f"    anchor B (R4a scalar frozen at {fr / 1e9:.2f} GHz):")
    for f, zeta, phi in (fams[0][0], fams[0][len(fams[0]) // 2], fams[0][-1]):
        w_dt = 2.0 * math.pi * f * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        lam_sc = complex(lambda_symbol((1.0 + 1e-12) * np.exp(1j * w_dt), r_fit, q_fit))
        lam_t = np.zeros((n_t, D[0].shape[0]), dtype=complex)
        lam_t[:, :n_t] = lam_sc * outer
        g, _, _ = modal_reflection(lam_t, D, sig_hat, w_eps, n_t, phi, zeta)
        print(f"      f {f / 1e9:5.2f} GHz  |Gamma| {20 * math.log10(max(g, 1e-300)):7.1f} dB")

    print(f"  [{name}] part 2 — certificate (i): projection ceiling")
    _, _, sv = band_subspace(fams, w_eps, n_t, 1)
    print("    family singular values: " + " ".join(f"{s:.1e}" for s in sv[:12]))
    results = {}
    f_eval = np.linspace(f_band[0], f_band[1], 9 if fast else 21)
    for p in p_list:
        V, U, _ = band_subspace(fams, w_eps, n_t, p)
        Dt, VtW = galerkin_exterior(D, V, w_eps)
        worst = -1e9
        rows = []
        for f in f_eval:
            w_dt = 2.0 * math.pi * f * dt
            sig_hat = 2.0 - 2.0 * math.cos(w_dt)
            z = (1.0 + _RHO_OFF) * np.exp(1j * w_dt)
            lam_p = galerkin_symbol_t(D, Dt, V, VtW, z, n_t)
            gs = []
            for fam in (0, 1):
                sel = min(fams[fam], key=lambda t: abs(t[0] - f)) if fams[fam] else None
                if sel is None or abs(sel[0] - f) > 1e-3 * f:
                    continue
                g, _, _ = modal_reflection(lam_p, D, sig_hat, w_eps, n_t, sel[2], sel[1])
                gs.append(20 * math.log10(max(g, 1e-300)))
            if gs:
                rows.append((f, gs))
                worst = max(worst, max(gs))
        results[p] = worst
        print(f"    p = {p:2d}: max in-band |Gamma| {worst:7.1f} dB")
    # Detail at the best p: per-frequency table incl. 2nd family.
    p_best = min(results, key=lambda p: results[p])
    V, U, _ = band_subspace(fams, w_eps, n_t, p_best)
    Dt, VtW = galerkin_exterior(D, V, w_eps)
    print(f"    detail (p = {p_best}):")
    for fam in (0, 1):
        if not fams[fam]:
            continue
        vals = []
        for f, zeta, phi in fams[fam][:: max(len(fams[fam]) // 6, 1)]:
            w_dt = 2.0 * math.pi * f * dt
            sig_hat = 2.0 - 2.0 * math.cos(w_dt)
            lam_p = galerkin_symbol_t(D, Dt, V, VtW, (1.0 + _RHO_OFF) * np.exp(1j * w_dt), n_t)
            g, _, _ = modal_reflection(lam_p, D, sig_hat, w_eps, n_t, phi, zeta)
            vals.append((f, 20 * math.log10(max(g, 1e-300))))
        print(f"      family {fam}: " + "  ".join(f"{f / 1e9:.2f}GHz {g:.1f}" for f, g in vals))
    return fams, p_best, results[p_best]


# ----------------------------------------------------------------------
# Part 3 — TD validation of the projected kernel
# ----------------------------------------------------------------------


def projected_kernel(D, n_kernel, n_t, V, U, w_eps):
    """p-channel kernel q_m = U^T W_t L_m V from the contour FFT."""
    L, cert = matrix_kernel(D[0], D[1], D[2], n_kernel, n_t, verbose=False)
    q = np.einsum("ti,mtj,jk->mik", U * w_eps[:n_t, None], L, V)
    return q, L, cert


def run_chain_cw_projected(
    D, dt, q_m, V, U, w_eps, n_t, K, phi, w, n_steps, sigma_steps, fit_planes
):
    """R4-spike CW chain run with the PROJECTED boundary kernel."""
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    hist_u = np.zeros((n_steps + 1, q_m.shape[2]))
    traces = np.empty((n_steps, len(fit_planes), n))
    t0 = 5.0 * sigma_steps
    Dp1_et = D_p1[:, :n_t]
    VtW = (V * w_eps[:, None]).T.real
    for nn in range(n_steps):
        m_max = min(nn, q_m.shape[0] - 1)
        if m_max >= 1:
            hseg = hist_u[nn - 1 :: -1][:m_max]
            ghost = U.real @ np.einsum("mij,mj->i", q_m[1 : m_max + 1].real, hseg)
        else:
            ghost = np.zeros(n_t)
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += ghost @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        amp = 0.5 * (1.0 + float(erf(((nn + 1) - t0) / (math.sqrt(2.0) * sigma_steps))))
        x_nx[0] = amp * np.real(phi * np.exp(1j * w * dt * (nn + 1)))
        x_pr, x_cu = x_cu, x_nx
        hist_u[nn + 1] = VtW @ x_cu[K - 1]
        traces[nn] = x_cu[fit_planes]
    return traces


def small_kernel(Dt, n_kernel):
    """Exact DTBC kernel of the p-dim Galerkin exterior (2p-size QZ)."""
    p = Dt[1].shape[0]
    return matrix_kernel(Dt[0], Dt[1], Dt[2], n_kernel, p, verbose=False)


def run_chain_cw_galerkin(
    D, Dt, VtW, V, dt, Lt, n_t, K, phi, w, n_steps, sigma_steps, fit_planes, noise=None
):
    """CW/noise chain run with the Galerkin-exterior boundary.

    Periods 0..K-1 are the full interior; the boundary period K is a
    p-dim projected state ``xt`` evolved explicitly (coupled through
    the block-symmetric interface), and its own exterior is closed by
    the exact small-system DTBC kernel ``Lt``.
    """
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    p = Dt[1].shape[0]
    Dp1_et = D_p1[:, :n_t]
    V_t = V[:n_t, :].real
    Cm = (VtW @ D_m1).real
    Dt0, Dtp = Dt[1].real, Dt[2].real
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    xt_pr = np.zeros(p)
    xt_cu = np.zeros(p)
    hist = np.zeros((n_steps + 1, p))
    traces = np.empty((n_steps, len(fit_planes), n))
    energy = np.empty(n_steps)
    t0 = 5.0 * sigma_steps
    for nn in range(n_steps):
        m_max = min(nn, Lt.shape[0] - 1)
        if m_max >= 1:
            hseg = hist[nn - 1 :: -1][:m_max]
            ghost_s = np.einsum("mij,mj->i", Lt[1 : m_max + 1], hseg)
        else:
            ghost_s = np.zeros(p)
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += (V_t @ xt_cu) @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        xt_nx = 2.0 * xt_cu - xt_pr - (Cm @ x_cu[K - 1] + Dt0 @ xt_cu + Dtp @ ghost_s)
        if noise is None:
            amp = 0.5 * (1.0 + float(erf(((nn + 1) - t0) / (math.sqrt(2.0) * sigma_steps))))
            x_nx[0] = amp * np.real(phi * np.exp(1j * w * dt * (nn + 1)))
        else:
            x_nx[0, :n_t] = 0.0
            if nn < 300:
                g = noise[nn]
                x_nx[K // 2 - 1] += g @ D_p1.T
                x_nx[K // 2] += g @ D_0.T
                x_nx[K // 2 + 1] += g @ D_m1.T
        x_pr, x_cu = x_cu, x_nx
        xt_pr, xt_cu = xt_cu, xt_nx
        hist[nn + 1] = xt_cu
        traces[nn] = x_cu[fit_planes]
        energy[nn] = math.hypot(float(np.linalg.norm(x_cu)), float(np.linalg.norm(xt_cu)))
    return traces, energy


def part3(name, D, dt, w_eps, n_t, fams, p, fast):
    print(f"  [{name}] part 3 — TD validation (Galerkin exterior, p = {p})")
    V, _, _ = band_subspace(fams, w_eps, n_t, p)
    Dt, VtW = galerkin_exterior(D, V, w_eps)
    n_kernel = 1024 if fast else 4096
    Lt, cert = small_kernel(Dt, n_kernel)
    print(
        f"    small kernel (2p = {2 * p}): solvent residual "
        f"{cert['residual']:.1e}, n_kernel {n_kernel}"
    )
    K = 40
    fit_planes = list(range(K - 16, K - 4))
    sigma_steps = n_kernel // 24
    n_steps = n_kernel - 2
    n_win = n_kernel // 4
    picks = [fams[0][1], fams[0][len(fams[0]) // 2], fams[0][-2]]
    if fams[1]:
        picks.append(fams[1][max(len(fams[1]) // 2, 1) - 1])
    for f, zeta, phi in picks:
        w = 2.0 * math.pi * f
        w_dt = w * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        lam_p = galerkin_symbol_t(D, Dt, V, VtW, (1.0 + _RHO_OFF) * np.exp(1j * w_dt), n_t)
        g_ap, _, _ = modal_reflection(lam_p, D, sig_hat, w_eps, n_t, phi, zeta)
        others = []
        for fam in (0, 1):
            sel = min(fams[fam], key=lambda t: abs(t[0] - f)) if fams[fam] else None
            if sel and abs(sel[1] - zeta) > 1e-9 and abs(sel[0] - f) < 1e-3 * f:
                others.append((sel[1], sel[2]))
        traces, _ = run_chain_cw_galerkin(
            D, Dt, VtW, V, dt, Lt, n_t, K, phi, w, n_steps, sigma_steps, fit_planes
        )
        ba, res_fit, res_mod = lockin_two_wave(
            traces, w, dt, fit_planes, zeta, phi, w_eps, n_win, others=others
        )
        print(
            f"    f {f / 1e9:5.2f} GHz  a-priori "
            f"{20 * math.log10(max(g_ap, 1e-300)):7.1f} dB   "
            f"TD |b/a| {20 * math.log10(max(ba, 1e-300)):7.1f} dB"
            f"   (fit res {res_fit:.1e}/{res_mod:.1e})"
        )


# ----------------------------------------------------------------------
# Part 4 — passivity / stability probe + out-of-band bound
# ----------------------------------------------------------------------


def part4(name, D, dt, w_eps, n_t, fams, p, f_band, fast):
    V, U, _ = band_subspace(fams, w_eps, n_t, p)
    Dt, VtW = galerkin_exterior(D, V, w_eps)
    # Out-of-band reflection bounded (passivity indicator on the
    # symbol level): fundamental incidence below/above the band.
    f_lo, f_hi = f_band
    worst = -1e9
    for f in np.linspace(0.3 * f_lo, 1.15 * f_hi, 11):
        w_dt = 2.0 * math.pi * f * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        vals, vecs = all_branches(D, sig_hat)
        (zo, po), _ = split_branches(vals, vecs, w_eps)
        on = np.abs(np.abs(zo) - 1.0) <= _ONC_TOL
        if not np.any(on):
            continue
        zp, pp = zo[on], po[:, on]
        j = int(np.argmax(np.abs(np.angle(zp))))
        lam_p = galerkin_symbol_t(D, Dt, V, VtW, (1.0 + _RHO_OFF) * np.exp(1j * w_dt), n_t)
        g, _, _ = modal_reflection(
            lam_p, D, sig_hat, w_eps, n_t, normalize_gauge(pp[:, j], n_t), complex(zp[j])
        )
        worst = max(worst, g)
    print(
        f"  [{name}] part 4 — out-of-band max |Gamma| {worst:.2e} (<= 1 = no over-unity reflection)"
    )

    def window_stats(energy, n_steps):
        n_w = 8
        wmax = np.array(
            [energy[i * n_steps // n_w : (i + 1) * n_steps // n_w].max() for i in range(n_w)]
        )
        r = wmax[1:] / wmax[:-1]
        return r.max(), r[-1], energy[-1] / energy.max()

    n_kernel = 1024 if fast else 4096
    n_steps = n_kernel - 2
    K = 40
    rng = np.random.default_rng(5)
    noise = 1e-3 * rng.standard_normal((300, D[1].shape[0]))
    # Certified form: Galerkin exterior (structurally lossless).
    Lt, _ = small_kernel(Dt, n_kernel)
    _, energy = run_chain_cw_galerkin(
        D, Dt, VtW, V, dt, Lt, n_t, K, None, 0.0, n_steps, 1.0, [K // 2], noise=noise
    )
    rmax, rlast, fp = window_stats(energy, n_steps)
    print(
        f"    Galerkin exterior:      window-max ratios max "
        f"{rmax:.4f} / last {rlast:.4f}, final/peak {fp:.2e}"
    )
    # NEGATIVE control: naive kernel projection (weakly active).
    q_m, _, _ = projected_kernel(D, n_kernel, n_t, V, U, w_eps)
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    hist_u = np.zeros((n_steps + 1, q_m.shape[2]))
    energy = np.empty(n_steps)
    Dp1_et = D_p1[:, :n_t]
    VtWr = (V * w_eps[:, None]).T.real
    for nn in range(n_steps):
        m_max = min(nn, q_m.shape[0] - 1)
        if m_max >= 1:
            hseg = hist_u[nn - 1 :: -1][:m_max]
            ghost = U.real @ np.einsum("mij,mj->i", q_m[1 : m_max + 1].real, hseg)
        else:
            ghost = np.zeros(n_t)
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += ghost @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        x_nx[0, :n_t] = 0.0  # PEC far end
        if nn < 300:
            g = noise[nn]
            x_nx[K // 2 - 1] += g @ D_p1.T
            x_nx[K // 2] += g @ D_0.T
            x_nx[K // 2 + 1] += g @ D_m1.T
        x_pr, x_cu = x_cu, x_nx
        hist_u[nn + 1] = VtWr @ x_cu[K - 1]
        energy[nn] = float(np.linalg.norm(x_cu))
    rmax, rlast, fp = window_stats(energy, n_steps)
    print(
        f"    naive kernel proj (neg. control): window-max ratios "
        f"max {rmax:.4f} / last {rlast:.4f}, final/peak {fp:.2e}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def run_case(name, fast):
    case, _ = make_case(name, 24)
    C = build_curl_matrix(case.grid)
    A = (sp.diags(1.0 / case.m_eps) @ C.T @ sp.diags(1.0 / case.m_mu) @ C).tocsr()
    k0 = case.Nz // 2
    sub = A[case.period(k0), :]
    B = tuple(sub[:, case.period(k0 + d)].toarray() for d in (-1, 0, 1))
    D = tuple(case.dt**2 * b for b in B)
    w_eps = case.m_eps[case.period(k0)]
    print(f"[{name}] N = {case.n} (N_t = {case.n_t}), dt = {case.dt * 1e12:.3f} ps")
    if name == "layered":
        f_band, f_ref = (1.0e9, 7.8e9), 4.2e9
    else:
        f_band, f_ref = (2.1e9, 6.2e9), 4.2e9
    p_list = [4, 6, 8, 10, 12] if not fast else [6, 10]
    fams, p_best, ceil = part12(name, D, case.dt, w_eps, case.n_t, f_band, f_ref, p_list, fast)
    part3(name, D, case.dt, w_eps, case.n_t, fams, p_best, fast)
    part4(name, D, case.dt, w_eps, case.n_t, fams, p_best, f_band, fast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["layered", "block", "all"])
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    for name in ["layered", "block"] if args.case == "all" else [args.case]:
        run_case(name, args.fast)
        print()


if __name__ == "__main__":
    main()
