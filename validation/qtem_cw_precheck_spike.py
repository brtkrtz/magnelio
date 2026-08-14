"""WP-R4a pre-check gate — frequency-local exact termination for CW
runs on inhomogeneous lines (reflection-free plan WP-R4a, retired
to git history, see DD-056; run
before any production code under the sessions-63-65 ground rule).

Question: the WP-R4 spike established that no *fixed* real transversal
profile terminates a QTEM/hybrid line broadband (best scalar symbol:
-10..-60 dB at the band ends).  The developer-selected production form
Option B measures CW, one frequency per run — so the port only has to
be exact *at the drive frequency*.  Can the existing scalar production
machinery (single real profile + ``DTBCTermination``) carry the true
discrete mode at one frequency to < -100 dB, with everything derived
from the per-frequency zeta eigenproblem?

Derivation (the analytic path to -100 dB)
-----------------------------------------

At the drive frequency ``w`` the true discrete mode of the uniform
continuation is the eigenpair ``(zeta, phi)`` of the quadratic pencil

    [ zeta^2 D_p1 + zeta (D_0 - sig_hat) + D_m1 ] phi = 0,
    sig_hat = 2 - 2 cos(w dt)   (real on the unit circle),

with the dimensionless period blocks ``D = dt^2 B`` of the WP-R4
spike.  Three facts make the frequency-local scalar termination exact
at ``w``:

1. **Real port profile.**  The blocks are real and ``sig_hat`` is
   real, so eigenvalues come in conjugate pairs, and the palindromic
   W-structure (``W D_p1 = (W D_m1)^T``) pairs ``zeta`` with
   ``1/zeta``.  For a propagating mode (|zeta| = 1) both pairings
   coincide: ``conj(phi)`` is the eigenvector of ``1/zeta = conj(zeta)``
   — the incoming wave.  Combined with the z-reflection symmetry of
   the trace ``x_k = (e_t(k), e_z(k+1/2))`` this fixes the gauge
   ``phi_t`` real, ``phi_z in i*zeta^{-1/2}*R`` (Part 1 measures the
   residual).  The tangential profile the port stores is therefore a
   real vector — the production ``DiscreteMode`` form.

2. **Exact scalar-chain fit at w.**  The scalar KG chain with
   parameters (r, q) has ``sig_hat = r^2 (2 - zeta - 1/zeta) + q^2``.
   Matching the true ``zeta = e^{-i theta}`` at ``w`` exactly is ONE
   real equation (the relation is real on the circle),

       q^2 = 4 [ sin^2(w dt / 2) - r^2 sin^2(theta / 2) ],

   leaving r free.  r is spent on matching the *derivative*
   ``d zeta / d sig_hat`` (group delay — fastest transient settling,
   least sensitivity to the lock-in bin), available from the same
   eigenvector by Hellmann-Feynman with the palindromic left
   eigenvector ``psi = W conj(phi)``:

       d zeta / d sig_hat = zeta (phi^H W phi)
                            / (phi^H W (2 zeta D_p1 + D_0 - sig_hat) phi),
       r^2 = (phi^H W (2 zeta D_p1 + D_0 - sig_hat) phi)
             / ((1/zeta - zeta) (phi^H W phi)).

   Both are closed-form in the eigenpair — no tuning, no extra solve.
   By construction ``lambda(e^{i w dt}; r, q) = zeta`` to machine
   precision, so the a-priori reflection ``|Gamma(w)| = |lambda -
   zeta| / |1/zeta - lambda|`` of the R1 gate is float-noise AT the
   drive frequency; off-frequency content is transient by definition
   of the CW measurement.  Passivity: (r, q) real with q^2 >= 0 and
   0 < r <= 1 makes the kernel the exact symbol of a passive
   half-lattice (R1) — q^2 >= 0 is normal dispersion (eps_eff rising
   with f, true for dielectric-loaded lines; measured per point and
   enforced in production).

3. **Steady-state consistency of the profile overwrite.**  The pure
   outgoing wave ``x_k = Re(a phi zeta^k e^{i w t})`` satisfies the
   interior update exactly AND the port constraint (its e_t trace at
   every plane is proportional to phi_t; the modal amplitude
   ``u_k = <phi_t, e_t(k)>_W`` obeys the fitted scalar chain at
   ``(w, zeta)`` exactly, and the ghost convolution reproduces
   ``lambda * u_K = zeta * u_K``).  The port-plane overwrite is
   therefore compatible with a zero-reflection steady state; the
   measurement window starts after the transients drain.

   Above the second propagating cut-off, every propagating mode needs
   its own channel (a wiped propagating mode reflects totally).  The
   per-frequency eigenvectors of different branches are NOT
   M_eps-orthogonal on the e_t block, so multi-channel projection
   must be **dual-basis** (Gram-inverse applied to the projectors,
   reconstruction stays primal) — measured here, required for the
   production operator.

What this spike measures (all offline, reduced vector chain —
exact-arithmetic equal to the 3D update by the WP-R4 Part-1 identity):

* Part 1 — per-frequency fit table: r_eff, q_eff^2 (sign!), profile
  reality ``||Im phi_t|| / ||phi_t||``, a-priori ``|lambda - zeta|``
  and ``|Gamma(w)|``, Hellmann-Feynman r^2 vs finite differences.
* Part 2 — CW floors of the production-equivalent boundary (profile
  overwrite + ``DTBCTermination`` per channel, production time
  alignment, TF/SF ghost injection): |b/a| at the drive frequency by
  the true-mode two-wave lock-in fit, across the band, including the
  two-mode band with and without the second channel.
* Part 3 — sparse shift-invert zeta solve (LinearOperator around a
  factored ``A - sigma B`` of the linearised pencil) with frequency
  continuation: agreement with the dense on-circle solve, and the
  cost-watch preview on scaled-up cross-sections.
* Part 4 — stability probe: noise drive through the frequency-local
  port, late-window growth factor (< 1 = decaying).

Results (session 88) — gate PASSED
----------------------------------

* Part 1: q_eff^2 > 0 at every band point (normal dispersion, as
  derived); r_eff 0.446..0.529 (layered) / 0.481..0.486 (block),
  always < 1; profile reality ||Im phi_t||/||phi_t|| <= 2.5e-13;
  fit exactness |lambda(r,q) - zeta| <= 2.5e-12, a-priori
  |Gamma(w)| <= -209 dB; Hellmann-Feynman r^2 matches finite
  differences to <= 2.1e-8 (FD truncation at the flattest band
  point).
* Part 2 (criterion): production-equivalent CW floors, |b/a| at the
  drive frequency:

      layered  1.0 / 2.1 / 4.2 GHz (single mode):
          -135.9 / -140.0 / -140.4 dB
      layered  6.2 / 7.8 GHz (second mode propagating, dual channel):
          -143.6 / -146.4 dB
      block    2.1 / 4.2 / 6.2 GHz: -154.2 / -150.6 / -141.3 dB

  All 35-54 dB below the -100 dB line.  Honest scope note: on the
  *uniform* acceptance line the fundamental-only variant measures the
  SAME floor as the dual-channel port at 6.2/7.8 GHz — clean modal
  injection excites no second-mode content and nothing on a uniform
  line converts between modes, so the acceptance geometry does not
  exercise the extra channel.  The multi-channel port (with the
  dual-basis projection measured here) is nevertheless the production
  form: content scattered into other propagating modes by a real
  device MUST find a matched channel at the port, or it reflects
  totally off the profile wipe.
* Part 3: sparse shift-invert (factored ``A - sigma B`` +
  LinearOperator ARPACK) matches the dense on-circle solve to
  |dzeta| <= 1.2e-13 with profile overlaps 1 - O(1e-16); frequency
  continuation tracks the fundamental through the second cut-on.
  Cost preview (scaled block cross-sections, one factor + k=8
  eigensolve per frequency point, this host): N = 469 / 1801 / 3997
  / 7057 -> 0.01-0.07 / 0.05 / 0.15-0.21 / ~0.3 s, ARPACK-dominated.
  The cost-watch deliverable proper (production-sized port
  cross-sections vs 3D run time) is measured in the acceptance
  benchmark.
* Part 4: no exponential drift — the noise-probe window maxima
  fluctuate +-2..9 % around a constant level with final/peak
  0.77..0.93 (16k-step spike run; a 32k-step diagnostic run shows
  the same bounded fluctuation), against a PEC reference that
  conserves to 0.999995.  Non-modal noise content is *trapped* by a
  finite-channel port (the profile wipe reflects it — neutral, not
  active); at the drive frequency itself every propagating branch
  has a matched channel, which is all a CW measurement needs (the
  Part-2 floors include the ramp transient).  Probe-methodology
  lesson recorded in-line: a raw state kick excites the static
  gradient modes (null space of A), whose leapfrog Jordan block
  drifts linearly — the probe must force within the image of the
  update operator, or PEC references appear to "grow" (n/n0 window
  ratio 4/3 measured, port-independent).

Verdict: gate passed — the frequency-local scalar reduction of the
exact matrix DtN is analytically exact at the drive frequency and
measures at the R2/R3 float-noise class through the
production-equivalent boundary.  Production legs fixed by this gate:
real phi_t profile, closed-form (r_eff, q_eff) from the eigenpair
(q^2 >= 0 enforced), dual-basis projection for multi-channel ports,
sparse continuation solver.

Run:  python validation/qtem_cw_precheck_spike.py
      [--case layered|block|all] [--fast] [--cost]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtem_dtbc_method_spike import (  # noqa: E402
    LineCase,
    _edge_eps_block,
    flat_e_pec,
    lockin_two_wave,
    make_case,
)

from magnelio._operators.curl import build_curl_matrix  # noqa: E402
from magnelio._operators.material_matrices import (  # noqa: E402
    build_M_eps,
    build_M_mu,
)
from magnelio.mesh.grid import GridLines  # noqa: E402
from magnelio.mesh.mesher import Mesh  # noqa: E402
from magnelio.ports._modal.dtbc import (  # noqa: E402
    DTBCTermination,
    lambda_symbol,
)
from magnelio.solver.stability import courant_dt  # noqa: E402

# ----------------------------------------------------------------------
# On-circle eigensolve (dense reference) + frequency-local fit
# ----------------------------------------------------------------------


def _linearize(D_m1, D_0, D_p1, sig_hat):
    n = D_0.shape[0]
    Ap = np.zeros((2 * n, 2 * n), dtype=complex)
    Bp = np.zeros((2 * n, 2 * n), dtype=complex)
    Ap[:n, n:] = np.eye(n)
    Ap[n:, :n] = -D_m1
    Ap[n:, n:] = -(D_0 - sig_hat * np.eye(n))
    Bp[:n, :n] = np.eye(n)
    Bp[n:, n:] = D_p1
    return Ap, Bp


def circle_modes_exact(D, sig_hat):
    """All finite outgoing eigenpairs at real ``sig_hat`` (|z| = 1).

    Outgoing = strictly evanescent (|zeta| < 1 - tol) or propagating
    with ``Im zeta < 0`` (the ``zeta = e^{-i beta dz}``, ``z =
    e^{+i w dt}`` convention).  On the circle the pencil is real, so
    the incoming partner of every propagating mode is the conjugate
    pair member — discarded here.
    """
    D_m1, D_0, D_p1 = D
    Ap, Bp = _linearize(D_m1, D_0, D_p1, complex(sig_hat))
    vals, vecs = sla.eig(Ap, Bp)
    fin = np.isfinite(vals)
    vals, vecs = vals[fin], vecs[:, fin]
    tol = 1e-7
    out = (np.abs(vals) < 1.0 - tol) | ((np.abs(vals) <= 1.0 + tol) & (vals.imag < 0.0))
    vals, vecs = vals[out], vecs[:, out]
    order = np.argsort(-np.abs(vals))
    n = D_0.shape[0]
    return vals[order], vecs[:n, order]


def pick_mode(vals, vecs, w_eps, phi_track=None):
    """Fundamental (largest |zeta|) or tracked-by-W-overlap eigenpair.

    Returns the picked pair (full-W normalised, phase fixed on the
    largest e_t component) and the other *propagating* pairs for
    channel/fit bases.
    """
    norms = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(vecs), w_eps, vecs)))
    if phi_track is None:
        pick = 0
    else:
        ov = np.abs(np.conj(phi_track) @ (w_eps[:, None] * vecs)) / norms
        pick = int(np.argmax(ov))
    zeta = vals[pick]
    phi = vecs[:, pick] / norms[pick]
    others = [
        (vals[m], vecs[:, m] / norms[m])
        for m in range(vals.size)
        if m != pick and abs(vals[m]) > 1.0 - 1e-6
    ]
    return zeta, phi, others


def normalize_gauge(phi, n_t):
    """Phase-fix so the largest-|.| e_t component is real positive."""
    ref = phi[np.argmax(np.abs(phi[:n_t]))]
    return phi * (np.conj(ref) / abs(ref))


def chain_fit(zeta, phi, D, sig_hat, w_eps, w_dt):
    """Closed-form frequency-local (r^2, q^2) from one eigenpair.

    r^2 by Hellmann-Feynman derivative matching (see module
    docstring); q^2 by the exact on-circle identity — the fit hits
    ``lambda(e^{i w dt}; r, q) = zeta`` to machine precision by
    construction, independent of any r^2 bias.
    """
    D_m1, D_0, D_p1 = D
    qp = (2.0 * zeta) * (D_p1 @ phi) + D_0 @ phi - sig_hat * phi
    num = np.vdot(phi, w_eps * qp)
    den = (1.0 / zeta - zeta) * np.vdot(phi, w_eps * phi)
    r2 = num / den
    theta = -np.angle(zeta)
    q2 = 4.0 * (math.sin(w_dt / 2.0) ** 2 - r2.real * math.sin(theta / 2.0) ** 2)
    return float(r2.real), float(r2.imag), float(q2)


def et_profile(phi, n_t, w_t):
    """Real, W_t-normalised tangential port profile + reality residual."""
    p = phi[:n_t]
    imag_rel = float(np.linalg.norm(p.imag) / np.linalg.norm(p))
    prof = p.real / math.sqrt(float(np.dot(w_t, p.real**2)))
    return prof, imag_rel


def part1(name, D, dt, w_eps, n_t, f_list, dz):
    """Fit table: reality, q^2 sign, fit exactness, HF-vs-FD r^2."""
    D_m1, D_0, D_p1 = D
    w_t = w_eps[:n_t]
    print(f"  [{name}] part 1 — frequency-local fit (r_eff, q_eff^2, exactness)")
    phi_lo = None
    rows = []
    for f in f_list:
        w_dt = 2.0 * math.pi * f * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        vals, vecs = circle_modes_exact(D, sig_hat)
        zeta, phi, others = pick_mode(vals, vecs, w_eps, phi_track=phi_lo)
        phi = normalize_gauge(phi, n_t)
        if phi_lo is None:
            phi_lo = phi
        r2, r2_im, q2 = chain_fit(zeta, phi, D, sig_hat, w_eps, w_dt)
        _, imag_rel = et_profile(phi, n_t, w_t)
        # Fit exactness at the drive frequency: the kernel's symbol at
        # z = e^{i w dt} vs the true zeta (a-priori |Gamma| formula).
        z_c = (1.0 + 1e-12) * np.exp(1j * w_dt)
        lam = complex(lambda_symbol(z_c, math.sqrt(r2), math.sqrt(max(q2, 0.0))))
        mism = abs(lam - zeta)
        gam = mism / abs(1.0 / zeta - lam)
        # FD cross-check of the Hellmann-Feynman r^2.
        df = 1e-4 * f
        th = []
        for fs in (f - df, f + df):
            w_s = 2.0 * math.pi * fs * dt
            sg = 2.0 - 2.0 * math.cos(w_s)
            vs, ve = circle_modes_exact(D, sg)
            zt, _, _ = pick_mode(vs, ve, w_eps, phi_track=phi)
            th.append(-np.angle(zt))
        dtheta_dw = (th[1] - th[0]) / (2.0 * 2.0 * math.pi * df)
        theta = -np.angle(zeta)
        r2_fd = dt * math.sin(w_dt) / (math.sin(theta) * dtheta_dw)
        print(
            f"    f {f / 1e9:5.2f} GHz  r_eff {math.sqrt(r2):.6f}"
            f"  q_eff^2 {q2:+.3e}"
            f"  Im r^2 {r2_im:.1e}"
            f"  |Im phi_t| {imag_rel:.1e}"
            f"  |lam-zeta| {mism:.1e}"
            f"  |Gamma(w)| {20 * math.log10(max(gam, 1e-300)):6.0f} dB"
            f"  r^2 HF/FD-1 {r2 / r2_fd - 1.0:+.1e}"
        )
        rows.append((f, zeta, phi, r2, q2, others))
    return phi_lo, rows


# ----------------------------------------------------------------------
# Part 2 — production-equivalent CW run on the reduced chain
# ----------------------------------------------------------------------


class Channel:
    """One frequency-local port channel: real e_t profile + (r, q)."""

    def __init__(self, phi, zeta, n_t, w_t, D, sig_hat, w_eps, w_dt):
        r2, _, q2 = chain_fit(zeta, phi, D, sig_hat, w_eps, w_dt)
        if q2 < 0.0:
            raise RuntimeError(
                f"q_eff^2 = {q2:.3e} < 0 (anomalous dispersion) — gate assumption violated"
            )
        self.r = math.sqrt(r2)
        self.q = math.sqrt(q2)
        self.profile, self.imag_rel = et_profile(phi, n_t, w_t)
        self.zeta = zeta
        self.phi = phi


def run_cw_production_like(
    D, dt, w_t, channels, K, w, n_steps, sigma_steps, fit_planes, exc_channel=0
):
    """CW drive through the production-equivalent boundary.

    Both line ends are ports: plane 0 excites channel ``exc_channel``
    via the DTBC ghost injection, plane K absorbs.  Per step (the
    ``PortOperatorModal.update_e`` order): interior leapfrog first,
    then each port advances its per-channel ``DTBCTermination`` with
    the *previous-level* interior projection and overwrites the port
    plane's e_t with the primal profile expansion.  Projection is
    dual-basis (Gram-inverse) so non-orthogonal channels don't
    cross-talk.
    """
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    n_t = w_t.size
    prof = np.stack([c.profile for c in channels])  # (n_ch, n_t)
    gram = (prof * w_t[None, :]) @ prof.T
    proj = np.linalg.solve(gram, prof * w_t[None, :])  # dual projectors
    term_a = [DTBCTermination(c.r, c.q) for c in channels]
    term_b = [DTBCTermination(c.r, c.q) for c in channels]
    Dp1_et = D_p1[:, :n_t]

    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    b_et = np.zeros(n_t)  # e_t(K)
    traces = np.empty((n_steps, len(fit_planes), n))
    t0 = 5.0 * sigma_steps
    for nn in range(n_steps):
        # Interior projections at the current level (the operator's
        # ``V_interior_prev``: field one solver step before the new
        # port value).
        u_int_a = proj @ x_cu[1, :n_t]
        u_int_b = proj @ x_cu[K - 1, :n_t]
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += b_et @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        amp = 0.5 * (1.0 + float(erf((nn - t0) / (math.sqrt(2.0) * sigma_steps))))
        src = amp * math.sin(w * dt * nn)  # s(t^{n+1} - dt)
        vals_a = [
            term_a[c].advance(float(u_int_a[c]), src if c == exc_channel else 0.0)
            for c in range(len(channels))
        ]
        vals_b = [term_b[c].advance(float(u_int_b[c]), 0.0) for c in range(len(channels))]
        x_nx[0, :n_t] = np.asarray(vals_a) @ prof
        b_et = np.asarray(vals_b) @ prof
        x_pr, x_cu = x_cu, x_nx
        traces[nn] = x_cu[fit_planes]
    return traces


def part2(name, D, dt, w_eps, n_t, rows, fast):
    print(f"  [{name}] part 2 — production-equivalent CW floors (criterion: < -100 dB)")
    w_t = w_eps[:n_t]
    K = 48
    fit_planes = list(range(K // 2 - 6, K // 2 + 6))
    n_steps = 3000 if fast else 9000
    sigma_steps = n_steps // 12
    n_win = 2048 if not fast else 700
    for f, zeta, phi, r2, q2, others in rows:
        w = 2.0 * math.pi * f
        w_dt = w * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        chans = [Channel(phi, zeta, n_t, w_t, D, sig_hat, w_eps, w_dt)]
        for zt, pf in others:
            chans.append(Channel(normalize_gauge(pf, n_t), zt, n_t, w_t, D, sig_hat, w_eps, w_dt))
        variants = [("all-ch", chans)]
        if len(chans) > 1:
            variants.append(("fund-only", chans[:1]))
        for tag, ch in variants:
            traces = run_cw_production_like(D, dt, w_t, ch, K, w, n_steps, sigma_steps, fit_planes)
            ba, res_fit, res_mod = lockin_two_wave(
                traces, w, dt, fit_planes, zeta, phi, w_eps, n_win, others=others
            )
            db = 20.0 * math.log10(max(ba, 1e-300))
            gram_off = 0.0
            if len(ch) > 1:
                prof = np.stack([c.profile for c in ch])
                g = (prof * w_t[None, :]) @ prof.T
                gram_off = float(np.abs(g - np.diag(np.diag(g))).max())
            print(
                f"    f {f / 1e9:5.2f} GHz  {tag:9s}"
                f"  ({len(ch)} ch, gram-off {gram_off:.2f})"
                f"   |b/a| {db:8.1f} dB"
                f"   fit res {res_fit:.1e} / {res_mod:.1e}"
            )


# ----------------------------------------------------------------------
# Part 3 — sparse shift-invert with frequency continuation + cost
# ----------------------------------------------------------------------


def sparse_blocks(case, C):
    """Sparse period blocks (B_m1, B_0, B_p1) on the free DOFs."""
    A = (sp.diags(1.0 / case.m_eps) @ C.T @ sp.diags(1.0 / case.m_mu) @ C).tocsr()
    k0 = case.Nz // 2
    sub = A[case.period(k0), :]
    return tuple(sub[:, case.period(k0 + d)].tocsc() for d in (-1, 0, 1))


def sparse_zeta_solve(D_sp, sig_hat, zeta_target, k=8):
    """Outgoing eigenpairs near ``zeta_target`` by sparse shift-invert.

    Linearised pencil ``A y = zeta B y`` with singular ``B`` (rank
    deficiency = the pencil's infinite eigenvalues), so scipy's
    generalised path is bypassed: factor ``A - sigma B`` once and run
    ARPACK on the standard operator ``(A - sigma B)^{-1} B`` whose
    eigenvalues are ``1 / (zeta - sigma)``.
    """
    D_m1, D_0, D_p1 = D_sp
    n = D_0.shape[0]
    eye = sp.identity(n, format="csc")
    A_lin = sp.bmat([[None, eye], [-D_m1, -(D_0 - sig_hat * eye)]], format="csc")
    B_lin = sp.block_diag([eye, D_p1], format="csc")
    t0 = time.time()
    lu = spla.splu((A_lin - zeta_target * B_lin).astype(complex).tocsc())
    t_fac = time.time() - t0

    def matvec(v):
        return lu.solve(B_lin @ v)

    op = spla.LinearOperator((2 * n, 2 * n), matvec=matvec, dtype=complex)
    t0 = time.time()
    mu, vecs = spla.eigs(op, k=k, which="LM")
    t_eig = time.time() - t0
    zeta = zeta_target + 1.0 / mu
    order = np.argsort(np.abs(zeta - zeta_target))
    return zeta[order], vecs[:n, order], t_fac, t_eig


def make_block_scaled(n_xy, nz=8):
    """Block case at n_xy x (n_xy+1) transversal cells (cost scaling)."""
    d = 1.0e-3
    grid = GridLines(
        x=np.arange(n_xy + 1) * d,
        y=np.arange(n_xy + 2) * d,
        z=np.arange(nz + 1) * d,
    )
    mesh = Mesh.from_grid(grid).with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    eps_edge = _edge_eps_block(grid, 4.0, 0.4 * n_xy * d, 0.4 * n_xy * d)
    m_eps = build_M_eps(mesh) * eps_edge
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, "normal")
    return LineCase(f"block{n_xy}", grid, m_eps, m_mu, flat_e_pec(mesh), dt)


def part3(name, case, C, D_dense, dt, w_eps, rows, do_cost):
    print(f"  [{name}] part 3 — sparse shift-invert vs dense, continuation")
    D_sp = tuple(dt * dt * b for b in sparse_blocks(case, C))
    zeta_prev = None
    for f, zeta_ref, phi_ref, _, _, _ in rows:
        w_dt = 2.0 * math.pi * f * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        target = zeta_prev if zeta_prev is not None else 0.995 * np.exp(-0.05j)
        zs, vs, t_fac, t_eig = sparse_zeta_solve(D_sp, sig_hat, target)
        ov = np.abs(np.conj(phi_ref) @ (w_eps[:, None] * vs))
        nrm = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(vs), w_eps, vs)))
        pick = int(np.argmax(ov / nrm))
        dz_err = abs(zs[pick] - zeta_ref)
        print(
            f"    f {f / 1e9:5.2f} GHz  |zeta_sp - zeta_dense| "
            f"{dz_err:.1e}   overlap 1-{1.0 - ov[pick] / nrm[pick]:.1e}"
            f"   ({t_fac * 1e3:.0f} + {t_eig * 1e3:.0f} ms)"
        )
        zeta_prev = zs[pick]
    if not do_cost:
        return
    print(
        "    cost preview on scaled block cross-sections "
        "(factor + k=8 eigensolve per frequency point):"
    )
    for n_xy in (12, 24, 36, 48):
        cs = make_block_scaled(n_xy)
        Cs = build_curl_matrix(cs.grid)
        D_s = tuple(cs.dt * cs.dt * b for b in sparse_blocks(cs, Cs))
        w_dt = 2.0 * math.pi * 4.2e9 * cs.dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        t0 = time.time()
        zs, _, t_fac, t_eig = sparse_zeta_solve(D_s, sig_hat, 0.99 * np.exp(-0.1j))
        t_tot = time.time() - t0
        print(
            f"      n_t {n_xy}x{n_xy + 1}  N {cs.n:6d}"
            f"   total {t_tot:7.2f} s"
            f"  (factor {t_fac:.2f} s, eigs {t_eig:.2f} s)"
        )


# ----------------------------------------------------------------------
# Part 4 — stability probe
# ----------------------------------------------------------------------


def part4(name, D, dt, w_eps, n_t, rows, fast):
    """Noise drive at plane 0, free run; late-window growth < 1."""
    f, zeta, phi, r2, q2, others = rows[-1]
    w_dt = 2.0 * math.pi * f * dt
    sig_hat = 2.0 - 2.0 * math.cos(w_dt)
    w_t = w_eps[:n_t]
    chans = [Channel(phi, zeta, n_t, w_t, D, sig_hat, w_eps, w_dt)]
    for zt, pf in others:
        chans.append(Channel(normalize_gauge(pf, n_t), zt, n_t, w_t, D, sig_hat, w_eps, w_dt))
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    K = 48
    n_steps = 4000 if fast else 16000
    prof = np.stack([c.profile for c in chans])
    gram = (prof * w_t[None, :]) @ prof.T
    proj = np.linalg.solve(gram, prof * w_t[None, :])
    term_a = [DTBCTermination(c.r, c.q) for c in chans]
    term_b = [DTBCTermination(c.r, c.q) for c in chans]
    Dp1_et = D_p1[:, :n_t]
    rng = np.random.default_rng(11)
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    b_et = np.zeros(n_t)
    energy = np.empty(n_steps)
    for nn in range(n_steps):
        u_int_a = proj @ x_cu[1, :n_t]
        u_int_b = proj @ x_cu[K - 1, :n_t]
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += b_et @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        vals_a = [t.advance(float(u_int_a[c]), 0.0) for c, t in enumerate(term_a)]
        vals_b = [t.advance(float(u_int_b[c]), 0.0) for c, t in enumerate(term_b)]
        x_nx[0, :n_t] = np.asarray(vals_a) @ prof
        b_et = np.asarray(vals_b) @ prof
        if nn < 300:
            # Kernel-free forcing (image of the update operator): a
            # raw state kick also excites the static gradient modes
            # (null space of A), whose leapfrog Jordan block drifts
            # LINEARLY — a secular artefact of the probe, not boundary
            # activity (measured: PEC ends show the same n/n0 growth).
            g = 1e-3 * rng.standard_normal(n)
            x_nx[K // 2 - 1] += g @ D_p1.T
            x_nx[K // 2] += g @ D_0.T
            x_nx[K // 2 + 1] += g @ D_m1.T
        x_pr, x_cu = x_cu, x_nx
        energy[nn] = float(np.linalg.norm(x_cu))
    # Window-maxima statistics: a finite-channel port traps non-modal
    # noise content (neutral standing waves), so the honest activity
    # metric is the drift of the window maxima, not window-to-window
    # decay.  Exponential boundary activity would show as a
    # systematically rising trend.
    n_w = 8
    wmax = np.array(
        [energy[i * n_steps // n_w : (i + 1) * n_steps // n_w].max() for i in range(n_w)]
    )
    ratios = wmax[1:] / wmax[:-1]
    print(
        f"  [{name}] part 4 — stability probe ({len(chans)} ch at "
        f"{f / 1e9:.1f} GHz, {n_steps} steps): window-max ratios "
        f"max {ratios.max():.4f} / last {ratios[-1]:.4f}, "
        f"final/peak {energy[-1] / energy.max():.2e}"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def run_case(name, fast, do_cost):
    case, _ = make_case(name, 24)
    C = build_curl_matrix(case.grid)
    dz = float(case.grid.z[1] - case.grid.z[0])
    print(f"[{name}] N = {case.n} (N_t = {case.n_t}), dt = {case.dt * 1e12:.3f} ps")
    A = (sp.diags(1.0 / case.m_eps) @ C.T @ sp.diags(1.0 / case.m_mu) @ C).tocsr()
    k0 = case.Nz // 2
    sub = A[case.period(k0), :]
    B = tuple(sub[:, case.period(k0 + d)].toarray() for d in (-1, 0, 1))
    D = tuple(case.dt**2 * b for b in B)
    w_eps = case.m_eps[case.period(k0)]
    if name == "layered":
        f_list = [1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.8e9]
    else:
        f_list = [2.1e9, 4.2e9, 6.2e9]
    if fast:
        f_list = f_list[1:4]
    _, rows = part1(name, D, case.dt, w_eps, case.n_t, f_list, dz)
    part2(name, D, case.dt, w_eps, case.n_t, rows, fast)
    part3(name, case, C, D, case.dt, w_eps, rows, do_cost)
    part4(name, D, case.dt, w_eps, case.n_t, rows, fast)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["layered", "block", "all"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cost", action="store_true", help="run the scaled-cross-section cost preview")
    args = ap.parse_args()
    cases = ["layered", "block"] if args.case == "all" else [args.case]
    for name in cases:
        run_case(name, args.fast, args.cost)
        print()


if __name__ == "__main__":
    main()
