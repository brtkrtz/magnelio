"""WP-R4 method-selection spike — exact matrix DTBC for inhomogeneous
lines (reflection-free plan WP-R4 gate; plan retired to git
history, see DD-056).

Question (sessions-63-65 ground rule: analytic path to |S11| < -100 dB
before any solver code): what terminates a *transversally
inhomogeneous* straight line (QTEM / hybrid modes) reflection-free?
The plan names two candidates — the discrete quadratic eigenproblem in
``zeta = e^{-i beta dz}`` per frequency (true discrete hybrid modes)
and multi-modal DtN / mode-matching.  This spike shows they are the
same construction and validates it offline.

Derivation
----------

**Reduction.**  On a z-uniform section (uniform ``dz``, z-uniform
materials and PEC masks) group the free E DOFs per period,

    x_k = [ e_t(k) ; e_z(k+1/2) ]  in R^N,

e_t = tangential E edge voltages on plane k (N_t of them), e_z =
normal E edge voltages on the half-plane above.  The lossless FIT
leapfrog in second-order form, ``e^{n+1} = 2 e^n - e^{n-1}
- dt^2 A e^n`` with ``A = M_eps^{-1} C^T M_mu^{-1} C``, is *exactly*
block-tridiagonal in k with k-independent blocks:

    x_k^{n+1} = 2 x_k^n - x_k^{n-1}
                - dt^2 (B_m1 x_{k-1}^n + B_0 x_k^n + B_p1 x_{k+1}^n).

No approximation: the blocks are index slices of A (verified here to
machine precision against the raw 3D leapfrog, Part 1).  In the
``W = diag(M_eps)`` metric ``W B_p1 = (W B_m1)^T`` and ``W B_0``
symmetric (A is W-self-adjoint).  Inhomogeneity of the cross-section
only destroys the *frequency-independent* diagonalisation (DD-052);
the chain structure itself survives.  Structurally,

    B_p1 has non-zero columns ONLY on the e_t block

(e_z(k+1/2) belongs to period k; nothing at period k reaches
e_t/e_z of period k+1 except through h_t(k+1/2), which reads
e_t(k+1) only).  The exterior therefore acts on the interior solely
through the ghost trace ``e_t(K+1)``.

**Exact DtN.**  Z-transform in time: with
``sigma(z) = (2 - z - 1/z)/dt^2`` (on the circle
``sigma = (2 sin(w dt/2)/dt)^2 = w_hat^2``) the recurrence becomes

    B_p1 x_{k+1} + (B_0 - sigma I) x_k + B_m1 x_{k-1} = 0.

Bloch solutions ``x_k = zeta^k phi`` solve the quadratic eigenproblem

    [ zeta^2 B_p1 + zeta (B_0 - sigma) + B_m1 ] phi = 0,

a T-palindromic-type pencil in the W metric: eigenvalues come in
(zeta, 1/zeta) pairs — the outgoing/incoming pairing.  ``B_p1`` and
``B_m1`` have rank N_t, so the 2N-eigenvalue count is N_z zeros +
N_z infinities + 2 N_t finite non-zero.  For ``|z| > 1`` the
half-lattice is strictly dissipative (same passivity argument as the
scalar R1 branch selection), so no ``|zeta| = 1`` eigenvalue exists
and the spectrum splits into exactly N stable (|zeta| < 1, zeros
included) and N anti-stable.  The semi-infinite outgoing solution
space is the stable deflating subspace of the linearisation

    zeta [ I  0    ] [x_k    ]   [ 0     I           ] [x_k    ]
         [ 0  B_p1 ] [x_{k+1}] = [ -B_m1 -(B_0-sigma)] [x_{k+1}],

computed per contour point by ordered QZ; with ``[X1; X2]`` an
orthonormal basis of it, the exact discrete transparent boundary is
the ghost relation

    x_{K+1} = Lambda(z) x_K,      Lambda = X2 X1^{-1},

the matrix generalisation of the scalar ``lambda = A - sqrt(A^2-1)``
(and its literal restriction: on a homogeneous cross-section Lambda
diagonalises on the transversal eigenbasis with eigenvalues
``lambda_symbol(z; r, q_i)`` — cross-checked here).  Only the e_t
rows are needed.  ``Lambda(z) -> 0`` as ``z -> infinity`` (the
solvent behaves like ``B_m1/sigma``), so the kernel

    L_m = rho^m * ifft( Lambda(rho e^{i theta}) )_m,   rho > 1,

has ``L_0 = 0`` — the ghost value depends on strictly past boundary
samples and slots into the explicit leapfrog exactly like the scalar
DTBC (same contour parameters as ``dtbc.dtbc_kernel``: 8x
oversampling, ``rho^{n_kernel} = e^4``).  Within a run of
``n <= n_kernel`` steps the boundary is the EXACT DtN of the uniform
continuation (the R2 auto-extension argument); truncation/compression
bounds only concern the production form.

**Why one real profile cannot get there.**  Any single-profile method
(today's QTEM path) freezes phi at one frequency.  The true
fundamental profile drifts with frequency (hybrid mode); the best any
scalar-chain symbol calibrated at f_ref can do is measured here two
ways: the scalar-symbol mismatch ``|lambda_sc(w) - zeta(w)| /
|1/zeta(w) - lambda_sc(w)|`` with (r_eff, q_eff) solved exactly at
f_ref (closed form: ``r^2 = Im d / (2 Im c)``, ``q^2 = 2 r^2 Re c -
Re d`` with ``c = (zeta + 1/zeta)/2 - 1``, ``d = z - 2 + 1/z``), and
the frozen-profile power deficit ``1 - |<phi_ref, phi(w)>_W|^2 /
(||phi_ref||^2 ||phi(w)||^2)``.  Both sit orders of magnitude above
the criterion (Part 2) — the WP-R4 plan caveat, quantified.

**Production compression path (rank argument).**  The kernel family
``{L_m}`` is numerically low-rank for m beyond a short head: strongly
evanescent branches (|zeta| << 1) die within a few steps, leaving the
few near-band branches.  The spike measures the singular-value decay
of the stacked tail — the a-priori certificate that a production
form 'short dense head + few long scalar channels' exists with a
computable bound (neglected branches decay like |zeta_cut|^m).  The
certified-compression design itself is the implementation WP, not
this gate.

Cases
-----

* ``layered`` — parallel plate (PEC y-walls, magnetic x-walls,
  Nx = 1), lower half filled with eps_r = 4.  Genuinely dispersive
  QTEM line with an analytic DC anchor: eps_eff(0) = 2*4/(1+4) = 1.6
  (series capacitors).  Precision case for Parts 1-3.
* ``block`` — 2D-inhomogeneous: eps_r = 4 block over the lower-left
  quadrant of a 4 mm x 5 mm parallel-plate cross-section (5x5 cells).
  Confirms nothing in the construction relied on the layered
  1D-ness.
* ``uniform`` — homogeneous filling: Lambda(z) eigenvalues must
  reproduce ``lambda_symbol(z; r, 0)`` (TEM branch) to machine
  precision — cross-validation against the R1-R3 scalar machinery.

The transversal metric here is a hand-built z-uniform staircase
(arithmetic interface averaging); the method consumes only
(B_m1, B_0, B_p1) extracted from *whatever* production M_eps/M_mu,
so metric details (DD-053 conformal coupling etc.) are orthogonal to
this gate and re-enter only in the production floors of the
implementation WP.

Results (session 87) — gate PASSED
----------------------------------

* Part 1 (reduction): recurrence residual of the period chain on the
  raw 3D leapfrog 2.8e-15 (layered) / 2.0e-15 (block) relative;
  locality and the B_p1 e_z-columns exactly zero, translation
  invariance ~2e-15, W-symmetry ~1.5e-16.  Uniform cross-check: the
  TEM branch of Lambda(z) reproduces ``lambda_symbol(z; r, 0)`` to
  1.5e-10 (eigenvalue conditioning of the non-normal solvent; the
  end-to-end TD floors below are the sharper certificate).
* Part 2 (single-profile refutation): the layered fundamental drifts
  eps_eff_hat 1.6035 -> 1.8966 across 1-7.9 GHz (DC anchor: series
  formula 2*4/(1+4) = 1.6); the best scalar symbol calibrated at
  4.2 GHz reflects -10.2 / -31.2 / -41.8 / -32.1 dB at the other
  band points (block: -49.9 / -60.5 dB), and the frozen-profile
  power deficit sits at -11..-19 dB(pwr) (block -29).  No single
  real transversal profile reaches -100 dB — the plan caveat,
  quantified.  Second propagating mode enters at ~5 GHz (layered);
  tracking by profile continuation, multi-wave fit basis.
* Part 3 (exact matrix DTBC): solvent residual <= 2.2e-12, ||L_0||
  ~1e-14, kernel imaginary part 6e-17.  CW lock-in on the reduced
  vector chain (exact-arithmetic equal to the 3D update by Part 1),
  joint two-wave fit with the true zeta(w), incl. band points with a
  second propagating mode in the fit basis:

      layered  1.0 / 2.1 / 4.2 / 6.2 / 7.8 GHz:
          |b/a| = -128.9 / -135.2 / -140.6 / -143.8 / -147.1 dB
      block    2.1 / 4.2 / 6.2 GHz:
          |b/a| = -133.6 / -139.3 / -142.3 dB

  (fit residuals 3e-10..1e-9 lock-in / 1e-8..1e-7 modal; all
  28-47 dB below the -100 dB line).  Stability probe: noise drive,
  free run to the kernel horizon — late-window growth 0.63 / 0.93
  (< 1, decaying); residual energy drains slowly (near-cut-off
  content, v_g -> 0 — physics, not boundary activity; within-run
  kernel exact => passive).
* Compression findings (production path): the RAW kernel tail is
  full-rank at every tolerance (N_t at m >= 256 — the z -> 1 branch
  point carries the whole electrostatic subspace), so 'few long
  scalar channels' does NOT fall out of the kernel itself.  The
  *band subspace* is small: the fundamental profile family over the
  band has rank 7 (layered) / 5 (block) at 1e-8 — a certified
  production compression must be built on the band content (e.g.
  projected DTBC on the family subspace + short dense head), or
  production measurement runs per frequency (CW) with the
  frequency-local exact termination.  Design + certificates belong
  to the implementation WP.

Verdict: gate passed.  The two plan candidates coincide (the
per-frequency zeta eigenproblem IS the spectral form of the
multi-modal DtN); the exact matrix DTBC carries inhomogeneous lines
to the same float-noise class as R2/R3.  Proposed acceptance for the
implementation WP (developer decision): |S11| < -100 dB on straight
inhomogeneous lines, CW lock-in through the production solver with
the true discrete mode per frequency (profile, zeta and discrete V/I
impedance from the same eigenvector); full band for the QTEM
fundamental, from 1.01*f_c_hat for higher hybrid modes.  TD
injection of a fixed profile stays legitimate — the exact ghost
propagation distributes it onto the true modes, and the
per-frequency decomposition sorts it out in post-processing.

Run:  python validation/qtem_dtbc_method_spike.py
      [--case layered|block|uniform|all] [--fast]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.special import erf

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.dtbc import lambda_symbol
from magnelio.solver.stability import courant_dt

C0 = 299_792_458.0


# ----------------------------------------------------------------------
# Cases — z-uniform inhomogeneous lines on hand-scaled staircase metric
# ----------------------------------------------------------------------


class LineCase:
    """z-uniform line: mesh-level arrays + per-period flat indices."""

    def __init__(self, name, grid, m_eps, m_mu, pec_flat, dt):
        self.name = name
        self.grid = grid
        self.m_eps = m_eps
        self.m_mu = m_mu
        self.pec_flat = pec_flat
        self.dt = dt
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        self.Nz = Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        # Plane-0 flat bases, k-fastest (stride 1 along z) per family.
        i, j = np.meshgrid(np.arange(Nx), np.arange(Ny + 1), indexing="ij")
        ex0 = ((i * (Ny + 1) + j) * (Nz + 1)).ravel()
        i, j = np.meshgrid(np.arange(Nx + 1), np.arange(Ny), indexing="ij")
        ey0 = (n_Ex + (i * Ny + j) * (Nz + 1)).ravel()
        i, j = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), indexing="ij")
        ez0 = (n_Ex + n_Ey + (i * (Ny + 1) + j) * Nz).ravel()
        # Free (non-PEC) DOFs; z-uniformity of the masks is asserted.
        free_ex = ~pec_flat[ex0]
        free_ey = ~pec_flat[ey0]
        free_ez = ~pec_flat[ez0]
        for k in (1, Nz // 2):
            assert np.array_equal(free_ex, ~pec_flat[ex0 + k])
            assert np.array_equal(free_ey, ~pec_flat[ey0 + k])
        for k in (1, Nz // 2):
            assert np.array_equal(free_ez, ~pec_flat[ez0 + k])
        self.et0 = np.concatenate([ex0[free_ex], ey0[free_ey]])
        self.ez0 = ez0[free_ez]
        self.n_t = self.et0.size
        self.n = self.n_t + self.ez0.size

    def period(self, k):
        """Flat E indices of period k: [e_t(k); e_z(k+1/2)]."""
        return np.concatenate([self.et0 + k, self.ez0 + k])


def _edge_eps_layered(grid, eps_r, y_if):
    """Per-edge relative eps for a layer y < y_if (arith. interface avg)."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    y_n = grid.y
    y_c = 0.5 * (y_n[:-1] + y_n[1:])
    tol = 1e-12

    def node_eps(y):
        out = np.ones_like(y)
        out[y < y_if - tol] = eps_r
        out[np.abs(y - y_if) < tol] = 0.5 * (1.0 + eps_r)
        return out

    ex = np.broadcast_to(node_eps(y_n)[None, :, None], (Nx, Ny + 1, Nz + 1)).ravel()
    ey = np.broadcast_to(
        np.where(y_c < y_if - tol, eps_r, 1.0)[None, :, None], (Nx + 1, Ny, Nz + 1)
    ).ravel()
    ez = np.broadcast_to(node_eps(y_n)[None, :, None], (Nx + 1, Ny + 1, Nz)).ravel()
    return np.concatenate([ex, ey, ez])


def _edge_eps_block(grid, eps_r, x_if, y_if):
    """Per-edge eps for a block x < x_if, y < y_if (adjacent-cell avg)."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x_n, y_n = grid.x, grid.y
    x_c = 0.5 * (x_n[:-1] + x_n[1:])
    y_c = 0.5 * (y_n[:-1] + y_n[1:])
    cell = np.where((x_c[:, None] < x_if) & (y_c[None, :] < y_if), eps_r, 1.0)  # (Nx, Ny)
    padx = np.pad(cell, ((1, 1), (0, 0)), mode="edge")
    pady = np.pad(cell, ((0, 0), (1, 1)), mode="edge")
    padxy = np.pad(cell, ((1, 1), (1, 1)), mode="edge")
    # Ex edge (i, j): dual face spans y — average cells (i, j-1), (i, j).
    ex2d = 0.5 * (pady[:, :-1] + pady[:, 1:])  # (Nx, Ny+1)
    # Ey edge (i, j): average cells (i-1, j), (i, j).
    ey2d = 0.5 * (padx[:-1, :] + padx[1:, :])  # (Nx+1, Ny)
    # Ez edge at node (i, j): average the four adjacent cells.
    ez2d = 0.25 * (
        padxy[:-1, :-1] + padxy[1:, :-1] + padxy[:-1, 1:] + padxy[1:, 1:]
    )  # (Nx+1, Ny+1)
    ex = np.broadcast_to(ex2d[:, :, None], (Nx, Ny + 1, Nz + 1)).ravel()
    ey = np.broadcast_to(ey2d[:, :, None], (Nx + 1, Ny, Nz + 1)).ravel()
    ez = np.broadcast_to(ez2d[:, :, None], (Nx + 1, Ny + 1, Nz)).ravel()
    return np.concatenate([ex, ey, ez])


def flat_e_pec(mesh):
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    return np.concatenate([pec[0, :n_Ex], pec[1, :n_Ey], pec[2, :n_Ez]])


def make_case(name, nz):
    """Build a spike case; magnetic x-walls (natural FIT boundary)."""
    if name in ("layered", "uniform"):
        ny, dy, dx, dz = 8, 1.0e-3, 10.0e-3, 1.0e-3
        grid = GridLines(
            x=np.array([0.0, dx]),
            y=np.arange(ny + 1) * dy,
            z=np.arange(nz + 1) * dz,
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
        if name == "layered":
            eps_edge = _edge_eps_layered(grid, 4.0, 4.0e-3)
        else:
            eps_edge = np.ones_like(build_M_eps(mesh))
    elif name == "block":
        nx, ny, d, dz = 4, 5, 1.0e-3, 1.0e-3
        grid = GridLines(
            x=np.arange(nx + 1) * d,
            y=np.arange(ny + 1) * d,
            z=np.arange(nz + 1) * dz,
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
        eps_edge = _edge_eps_block(grid, 4.0, 2.0e-3, 2.0e-3)
    else:
        raise ValueError(name)
    m_eps = build_M_eps(mesh) * eps_edge
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, "normal")  # vacuum CFL; eps >= 1 is safer
    return LineCase(name, grid, m_eps, m_mu, flat_e_pec(mesh), dt), mesh


# ----------------------------------------------------------------------
# Part 1 — block extraction + reduction identity on the raw 3D leapfrog
# ----------------------------------------------------------------------


def extract_blocks(case, C):
    """(B_m1, B_0, B_p1) as dense free-DOF blocks + structure checks."""
    A = (sp.diags(1.0 / case.m_eps) @ C.T @ sp.diags(1.0 / case.m_mu) @ C).tocsr()
    k0 = case.Nz // 2
    rows = case.period(k0)
    sub = A[rows, :]
    blocks = {d: sub[:, case.period(k0 + d)].toarray() for d in (-1, 0, 1)}
    # Locality: nothing outside the three periods (on free columns of
    # any other period; PEC columns are irrelevant — those DOFs are 0).
    far = np.concatenate([case.period(k0 + d) for d in (-3, -2, 2, 3)])
    loc = np.abs(sub[:, far]).max() if far.size else 0.0
    # Translation invariance: same extraction two periods away.
    rows2 = case.period(k0 + 2)
    sub2 = A[rows2, :]
    inv = max(
        np.abs(sub2[:, case.period(k0 + 2 + d)].toarray() - blocks[d]).max() for d in (-1, 0, 1)
    ) / max(np.abs(blocks[0]).max(), 1e-300)
    # Ghost structure: B_p1 columns vanish on the e_z block.
    ez_cols = float(np.abs(blocks[1][:, case.n_t :]).max())
    # W-symmetry: W B_p1 = (W B_m1)^T.
    w = case.m_eps[case.period(k0)]
    wsym = (
        np.abs(w[:, None] * blocks[1] - (w[:, None] * blocks[-1]).T).max()
        / np.abs(w[:, None] * blocks[1]).max()
    )
    return (
        blocks[-1],
        blocks[0],
        blocks[1],
        dict(locality=float(loc), invariance=float(inv), ez_cols=ez_cols, w_symmetry=float(wsym)),
    )


def leapfrog_residual(case, C, n_steps=200):
    """Recurrence residual of the period chain on the raw 3D update."""
    rng = np.random.default_rng(7)
    beta_e = case.dt / case.m_eps
    beta_h = case.dt / case.m_mu
    pec_idx = np.where(case.pec_flat)[0]
    e = np.zeros(case.m_eps.size)
    k0 = case.Nz // 2
    for k in range(k0 - 4, k0 + 5):
        idx = case.period(k)
        e[idx] = rng.standard_normal(idx.size) * math.exp(-(((k - k0) / 2.5) ** 2))
    e[pec_idx] = 0.0
    h = np.zeros(C.shape[0])
    ks = np.arange(2, case.Nz - 3)
    traces = np.empty((n_steps + 1, ks.size, case.n))
    idx_kn = np.array([case.period(k) for k in ks])
    traces[0] = e[idx_kn]
    for n in range(n_steps):
        e += beta_e * (C.T @ h)
        e[pec_idx] = 0.0
        traces[n + 1] = e[idx_kn]
        h -= beta_h * (C @ e)
    B_m1, B_0, B_p1, checks = extract_blocks(case, C)
    x = traces
    res = (
        x[2:, 1:-1]
        - 2.0 * x[1:-1, 1:-1]
        + x[:-2, 1:-1]
        + case.dt**2
        * (
            np.einsum("nkj,ij->nki", x[1:-1, :-2], B_m1)
            + np.einsum("nkj,ij->nki", x[1:-1, 1:-1], B_0)
            + np.einsum("nkj,ij->nki", x[1:-1, 2:], B_p1)
        )
    )
    rel = float(np.abs(res).max() / np.abs(x).max())
    return (B_m1, B_0, B_p1), checks, rel


# ----------------------------------------------------------------------
# Solvent Lambda(z) and matrix kernel
# ----------------------------------------------------------------------


def stable_solvent(D_m1, D_0, D_p1, sig_hat):
    """Stable solvent Lambda of D_p1 L^2 + (D_0 - sig_hat) L + D_m1 = 0.

    The blocks are the *dimensionless* ``D = dt^2 B`` (entries O(r^2))
    and ``sig_hat = dt^2 sigma = 2 - z - 1/z`` — same eigenstructure,
    but the linearisation mixes identity blocks with O(1) blocks
    instead of O(1/(eps mu dx^2)) ~ 1e23, which QZ cannot balance.
    """
    n = D_0.shape[0]
    Ap = np.zeros((2 * n, 2 * n), dtype=complex)
    Bp = np.zeros((2 * n, 2 * n), dtype=complex)
    Ap[:n, n:] = np.eye(n)
    Ap[n:, :n] = -D_m1
    Ap[n:, n:] = -(D_0 - sig_hat * np.eye(n))
    Bp[:n, :n] = np.eye(n)
    Bp[n:, n:] = D_p1
    _, _, alpha, beta, _, Z = sla.ordqz(Ap, Bp, sort="iuc", output="complex")
    n_stable = int(np.sum(np.abs(alpha) < np.abs(beta)))
    if n_stable != n:
        raise RuntimeError(f"dichotomy violated: {n_stable} stable of {2 * n}")
    X1 = Z[:n, :n]
    X2 = Z[n:, :n]
    lam = np.linalg.solve(X1.T, X2.T).T
    return lam


def solvent_residual(D_m1, D_0, D_p1, sig_hat, lam):
    r = D_p1 @ lam @ lam + (D_0 - sig_hat * np.eye(D_0.shape[0])) @ lam + D_m1
    return float(np.linalg.norm(r) / max(np.linalg.norm(D_m1), 1e-300))


def matrix_kernel(D_m1, D_0, D_p1, n_kernel, n_t, verbose=True):
    """Kernel L_m (e_t rows only), contour parameters as dtbc_kernel."""
    n = D_0.shape[0]
    n_fft = 8 * n_kernel
    rho = math.exp(4.0 / n_kernel)
    half = n_fft // 2 + 1
    spec = np.empty((half, n_t, n), dtype=complex)
    res_max = 0.0
    t0 = time.time()
    for jj in range(half):
        z = rho * np.exp(2j * np.pi * jj / n_fft)
        sig_hat = 2.0 - z - 1.0 / z
        lam = stable_solvent(D_m1, D_0, D_p1, sig_hat)
        if jj % (max(half // 16, 1)) == 0:
            res_max = max(res_max, solvent_residual(D_m1, D_0, D_p1, sig_hat, lam))
            if verbose:
                el = time.time() - t0
                print(f"      contour {jj}/{half}  ({el:.0f} s)", flush=True)
        spec[jj] = lam[:n_t, :]
    full = np.empty((n_fft, n_t, n), dtype=complex)
    full[:half] = spec
    full[half:] = np.conj(spec[1:-1][::-1])
    coeff = np.fft.ifft(full, axis=0)[:n_kernel]
    imag_max = float(np.abs(coeff.imag).max())
    L = coeff.real * (rho ** np.arange(n_kernel))[:, None, None]
    return L, dict(residual=res_max, imag=imag_max, l0=float(np.abs(L[0]).max()))


# ----------------------------------------------------------------------
# Part 2 — unit-circle modes, dispersion, frozen-profile bound
# ----------------------------------------------------------------------

_RHO_OFF = 1.0 + 1e-8


def circle_modes(D_m1, D_0, D_p1, dt, w):
    """Stable eigenpairs at z = (1+eps) e^{i w dt}, sorted by |zeta|."""
    n = D_0.shape[0]
    z = _RHO_OFF * np.exp(1j * w * dt)
    sig_hat = 2.0 - z - 1.0 / z
    Ap = np.zeros((2 * n, 2 * n), dtype=complex)
    Bp = np.zeros((2 * n, 2 * n), dtype=complex)
    Ap[:n, n:] = np.eye(n)
    Ap[n:, :n] = -D_m1
    Ap[n:, n:] = -(D_0 - sig_hat * np.eye(n))
    Bp[:n, :n] = np.eye(n)
    Bp[n:, n:] = D_p1
    vals, vecs = sla.eig(Ap, Bp)
    fin = np.isfinite(vals)
    vals, vecs = vals[fin], vecs[:, fin]
    stab = np.abs(vals) < 1.0
    vals, vecs = vals[stab], vecs[:, stab]
    order = np.argsort(-np.abs(vals))
    return vals[order], vecs[: D_0.shape[0], order]


def fundamental(D_m1, D_0, D_p1, dt, w, w_eps, phi_track=None):
    """Fundamental outgoing mode, M_eps-normalised.

    Without ``phi_track``: largest |zeta| (valid in the single-mode
    band).  With ``phi_track``: the stable mode of maximal W-overlap
    with the tracking profile — profile continuation, needed above
    the second propagating cut-off where |zeta| no longer orders the
    branches.  Also returns the other propagating modes (|zeta| >
    0.999) for the multi-wave fit basis.
    """
    vals, vecs = circle_modes(D_m1, D_0, D_p1, dt, w)
    norms = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(vecs), w_eps, vecs)))
    if phi_track is None:
        pick = 0
    else:
        ov = np.abs(np.conj(phi_track) @ (w_eps[:, None] * vecs)) / norms
        pick = int(np.argmax(ov))
    zeta, phi = vals[pick], vecs[:, pick] / norms[pick]
    phi = phi * np.exp(-1j * np.angle(phi[np.argmax(np.abs(phi))]))
    others = [
        (vals[m], vecs[:, m] / norms[m])
        for m in range(vals.size)
        if m != pick and abs(vals[m]) > 0.999
    ]
    gap = abs(vals[1]) / abs(vals[0]) if vals.size > 1 else 0.0
    return zeta, phi, gap, others


def scalar_best_fit(zeta_ref, w_ref, dt):
    """(r^2, q^2) of the scalar chain matching zeta exactly at w_ref."""
    z = _RHO_OFF * np.exp(1j * w_ref * dt)
    c = 0.5 * (zeta_ref + 1.0 / zeta_ref) - 1.0
    d = z - 2.0 + 1.0 / z
    r2 = d.imag / (2.0 * c.imag)
    q2 = 2.0 * r2 * c.real - d.real
    return r2, q2


def part2(name, D, dt, w_eps, f_list, f_ref, dz):
    D_m1, D_0, D_p1 = D
    print(f"  [{name}] part 2 — dispersion / frozen-profile bound")
    # Track the fundamental from the lowest band frequency upward —
    # above the second propagating cut-off |zeta| no longer orders it.
    _, phi_lo, _, _ = fundamental(D_m1, D_0, D_p1, dt, 2 * math.pi * min(f_list), w_eps)
    zeta_ref, phi_ref, _, _ = fundamental(
        D_m1, D_0, D_p1, dt, 2 * math.pi * f_ref, w_eps, phi_track=phi_lo
    )
    r2, q2 = scalar_best_fit(zeta_ref, 2 * math.pi * f_ref, dt)
    r_eff = math.sqrt(abs(r2))
    q_arg = q2 if q2 > 0 else 0.0
    print(f"    scalar fit at {f_ref / 1e9:.2f} GHz: r_eff {r_eff:.6f}  q_eff^2 {q2:.3e}")
    for f in f_list:
        w = 2 * math.pi * f
        zeta, phi, gap, _ = fundamental(D_m1, D_0, D_p1, dt, w, w_eps, phi_track=phi_lo)
        beta = -np.angle(zeta) / dz
        eps_eff = (
            2.0
            / dz
            * math.sin(abs(np.angle(zeta)) / 2.0)
            / (2.0 / dt * math.sin(w * dt / 2.0))
            * C0
        ) ** 2
        ov = abs(np.vdot(phi_ref, w_eps * phi)) ** 2
        drift = max(1.0 - ov, 0.0)
        lam_sc = complex(lambda_symbol(_RHO_OFF * np.exp(1j * w * dt), r_eff, math.sqrt(q_arg)))
        gam = abs(lam_sc - zeta) / abs(1.0 / zeta - lam_sc)
        gam_db = 20 * math.log10(max(gam, 1e-300))
        drift_db = 10 * math.log10(max(drift, 1e-300))
        print(
            f"    f {f / 1e9:6.2f} GHz  beta_hat {beta:9.2f}"
            f"  eps_eff_hat {eps_eff:7.4f}"
            f"  scalar-symbol floor {gam_db:7.1f} dB"
            f"  profile deficit {drift_db:7.1f} dB(pwr)"
            f"  |z2/z1| {gap:.3f}"
        )
    # Band-subspace rank of the fundamental profile family — the
    # honest 'few channels' quantifier for a production compression
    # (the raw kernel tail is NOT low-rank: the z -> 1 branch point
    # carries the whole electrostatic subspace, see part 3 print).
    f_dense = np.linspace(min(f_list), max(f_list), 33)
    fam = np.column_stack(
        [
            fundamental(D_m1, D_0, D_p1, dt, 2 * math.pi * f, w_eps, phi_track=phi_lo)[1]
            for f in f_dense
        ]
    )
    sv = np.linalg.svd(np.sqrt(w_eps)[:, None] * fam, compute_uv=False)
    ranks = {tol: int(np.sum(sv > tol * sv[0])) for tol in (1e-4, 1e-6, 1e-8, 1e-10)}
    print(f"    fundamental profile-family rank over band: {ranks}")
    return phi_lo


# ----------------------------------------------------------------------
# Part 3 — matrix-DTBC chain, CW lock-in, two-wave fit
# ----------------------------------------------------------------------


def run_chain_cw(D, dt, L_et, n_t, K, phi, w, n_steps, sigma_steps, fit_planes):
    """Hard CW drive at period 0, matrix DTBC at period K-1."""
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    hist_K = np.zeros((n_steps + 1, n))
    traces = np.empty((n_steps, len(fit_planes), n))
    t0 = 5.0 * sigma_steps
    Dp1_et = D_p1[:, :n_t]  # only e_t columns are non-zero
    for nn in range(n_steps):
        # Ghost e_t(K) at time level nn from the boundary history.
        m_max = min(nn, L_et.shape[0] - 1)
        if m_max >= 1:
            hseg = hist_K[nn - 1 :: -1][:m_max]
            ghost = np.einsum("mij,mj->i", L_et[1 : m_max + 1], hseg)
        else:
            ghost = np.zeros(n_t)
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += ghost @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        amp = 0.5 * (1.0 + float(erf(((nn + 1) - t0) / (math.sqrt(2.0) * sigma_steps))))
        # +i w t convention (z = e^{+i w dt}): x_0 = Re(phi e^{i w t})
        # launches the outgoing eigen-branch phi zeta^k cleanly.
        x_nx[0] = amp * np.real(phi * np.exp(1j * w * dt * (nn + 1)))
        x_pr, x_cu = x_cu, x_nx
        hist_K[nn + 1] = x_cu[K - 1]
        traces[nn] = x_cu[fit_planes]
    return traces


def lockin_two_wave(traces, w, dt, fit_planes, zeta, phi, w_eps, n_win, others=()):
    """LSQ lock-in phasors -> joint two-wave vector fit -> |b/a|.

    In the ``e^{+i w t}`` convention the steady state at the fit
    planes is ``u_k = a phi zeta^k + b conj(phi) zeta^{-k}`` (the
    incoming eigenvector of the near-real pencil is the conjugate of
    the outgoing one), so a joint least-squares fit over all planes
    and DOFs in the M_eps metric yields |b/a| directly — no scalar
    projection, no overlap-factor understatement.
    """
    n_steps = traces.shape[0]
    n_grid = np.arange(n_steps - n_win, n_steps)
    t_grid = (n_grid + 1) * dt
    basis = np.column_stack([np.cos(w * t_grid), np.sin(w * t_grid)])
    sig = traces[n_grid].reshape(n_win, -1)
    coef, *_ = np.linalg.lstsq(basis, sig, rcond=None)
    res_fit = float(np.linalg.norm(sig - basis @ coef) / max(np.linalg.norm(sig), 1e-300))
    # x = Re(u e^{+i w t}) = Re(u) cos - Im(u) sin -> u = c_cos - i c_sin
    ph = (coef[0] - 1j * coef[1]).reshape(len(fit_planes), -1)
    kk = np.asarray(fit_planes)
    sw = np.sqrt(w_eps)

    def wave_cols(zt, pf):
        ca = (zt**kk)[:, None] * pf[None, :] * sw[None, :]
        cb = (zt ** (-kk.astype(float)))[:, None] * np.conj(pf)[None, :] * sw[None, :]
        return ca.ravel(), cb.ravel()

    cols = list(wave_cols(zeta, phi))
    for zt, pf in others:  # other propagating modes, both ways
        cols.extend(wave_cols(zt, pf))
    G = np.column_stack(cols)
    rhs = (ph * sw[None, :]).ravel()
    ab, *_ = np.linalg.lstsq(G, rhs, rcond=None)
    res_mod = float(np.linalg.norm(rhs - G @ ab) / max(np.linalg.norm(rhs), 1e-300))
    a, b = ab[0], ab[1]
    return abs(b / a), res_fit, res_mod


def snap_frequency(f, dt):
    """Snap to an integer number of steps per period (clean lock-in)."""
    n_per = max(int(round(1.0 / (f * dt))), 4)
    return 1.0 / (n_per * dt), n_per


def part3(name, D, dt, w_eps, f_list, dz, n_kernel, n_t, K, phi_lo):
    D_m1, D_0, D_p1 = D
    print(f"  [{name}] part 3 — matrix DTBC, kernel n = {n_kernel}, chain K = {K}")
    L, cert = matrix_kernel(D_m1, D_0, D_p1, n_kernel, n_t, verbose=True)
    print(
        f"    solvent residual {cert['residual']:.2e}   "
        f"||L_0|| {cert['l0']:.2e}   kernel imag {cert['imag']:.2e}"
    )
    fit_planes = list(range(K - 16, K - 4))
    sigma_steps = n_kernel // 24
    n_steps = n_kernel - 2
    n_win = n_kernel // 4
    out = []
    for f_raw in f_list:
        f, n_per = snap_frequency(f_raw, dt)
        w = 2 * math.pi * f
        zeta, phi, _, others = fundamental(D_m1, D_0, D_p1, dt, w, w_eps, phi_track=phi_lo)
        traces = run_chain_cw(D, dt, L, n_t, K, phi, w, n_steps, sigma_steps, fit_planes)
        ba, res_fit, res_mod = lockin_two_wave(
            traces, w, dt, fit_planes, zeta, phi, w_eps, n_win, others=others
        )
        db = 20 * math.log10(max(ba, 1e-300))
        print(
            f"    f {f / 1e9:6.3f} GHz ({n_per} steps/per, "
            f"{len(others)} other propagating)"
            f"   |b/a| {db:8.1f} dB"
            f"   fit res {res_fit:.1e} / {res_mod:.1e}"
        )
        out.append((f, db))
    # Stability probe: noisy drive, then free run to the kernel
    # horizon.  Late-window growth factor is the passivity check;
    # residual energy itself decays only slowly (near-cut-off content
    # has v_g -> 0 — physics, not boundary activity).
    rng = np.random.default_rng(3)
    drive = rng.standard_normal((300, D_0.shape[0]))
    e_hist = stability_probe(D, L, n_t, K, drive, n_steps)
    q2 = e_hist[n_steps // 2 : 3 * n_steps // 4].max()
    q3 = e_hist[3 * n_steps // 4 :].max()
    print(
        f"    stability probe: late-window growth {q3 / q2:.6f} "
        f"(< 1 = decaying), energy final/peak "
        f"{e_hist[-1] / e_hist.max():.2e}"
    )
    # Raw kernel-tail rank — measured NEGATIVE finding: the tail is
    # full-rank (the z -> 1 branch point carries the electrostatic
    # subspace), so production compression must target the band
    # subspace (part 2 rank), not the raw kernel.
    for m0 in (32, 256):
        tail = L[m0:].reshape(-1, L.shape[2])
        sv = np.linalg.svd(tail, compute_uv=False)
        head = np.linalg.norm(L[:m0].reshape(-1, L.shape[2]), 2)
        ranks = {tol: int(np.sum(sv > tol * head)) for tol in (1e-8, 1e-10, 1e-12)}
        print(f"    raw kernel tail (m >= {m0}) rank vs head: {ranks}  (N_t = {n_t})")
    return out


def stability_probe(D, L_et, n_t, K, drive, n_steps):
    D_m1, D_0, D_p1 = D
    n = D_0.shape[0]
    x_pr = np.zeros((K, n))
    x_cu = np.zeros((K, n))
    hist_K = np.zeros((n_steps + 1, n))
    energy = np.empty(n_steps)
    Dp1_et = D_p1[:, :n_t]
    for nn in range(n_steps):
        m_max = min(nn, L_et.shape[0] - 1)
        if m_max >= 1:
            hseg = hist_K[nn - 1 :: -1][:m_max]
            ghost = np.einsum("mij,mj->i", L_et[1 : m_max + 1], hseg)
        else:
            ghost = np.zeros(n_t)
        y = x_cu @ D_0.T
        y[:-1] += x_cu[1:] @ D_p1.T
        y[1:] += x_cu[:-1] @ D_m1.T
        y[K - 1] += ghost @ Dp1_et.T
        x_nx = 2.0 * x_cu - x_pr - y
        if nn < drive.shape[0]:
            x_nx[0] = 1e-3 * drive[nn]
        else:
            x_nx[0] = 0.0
        x_pr, x_cu = x_cu, x_nx
        hist_K[nn + 1] = x_cu[K - 1]
        energy[nn] = float(np.linalg.norm(x_cu))
    return energy


# ----------------------------------------------------------------------
# Uniform cross-check against the scalar symbol
# ----------------------------------------------------------------------


def uniform_crosscheck(D, dt, dz):
    D_m1, D_0, D_p1 = D
    r = C0 * dt / dz
    dev = 0.0
    for w_dt in (0.05, 0.4, 1.2):
        z = math.exp(4.0 / 4096) * np.exp(1j * w_dt)
        sig_hat = 2.0 - z - 1.0 / z
        lam = stable_solvent(D_m1, D_0, D_p1, complex(sig_hat))
        ev = np.linalg.eigvals(lam)
        lam_sc = complex(lambda_symbol(z, r, 0.0))
        dev = max(dev, float(np.min(np.abs(ev - lam_sc))))
    return dev


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def run_case(name, fast):
    nz = 24
    case, mesh = make_case(name, nz)
    C = build_curl_matrix(case.grid)
    dz = float(case.grid.z[1] - case.grid.z[0])
    print(
        f"[{name}] N = {case.n} (N_t = {case.n_t}), dt = "
        f"{case.dt * 1e12:.3f} ps, dz = {dz * 1e3:.2f} mm"
    )
    B, checks, rel = leapfrog_residual(case, C)
    print(
        f"  part 1 — reduction: recurrence residual {rel:.2e}   "
        f"locality {checks['locality']:.1e}   invariance "
        f"{checks['invariance']:.1e}"
    )
    print(
        f"            B_p1 e_z-columns {checks['ez_cols']:.1e}   "
        f"W-symmetry {checks['w_symmetry']:.1e}"
    )
    dt2 = case.dt**2
    D = tuple(dt2 * b for b in B)  # dimensionless blocks, O(r^2)
    w_eps = case.m_eps[case.period(case.Nz // 2)]
    if name == "uniform":
        dev = uniform_crosscheck(D, case.dt, dz)
        print(f"  scalar-symbol cross-check (TEM branch): min|eig(Lambda) - lambda_sc| = {dev:.2e}")
        return
    if name == "layered":
        f_list = [1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.9e9]
        f_ref = 4.2e9
    else:
        f_list = [2.1e9, 4.2e9, 6.2e9]
        f_ref = 4.2e9
    phi_lo = part2(name, D, case.dt, w_eps, f_list, f_ref, dz)
    n_kernel = 1024 if fast else (4096 if name == "layered" else 2048)
    K = 40 if name == "layered" else 32
    f3 = f_list if not fast else f_list[1:2]
    part3(name, D, case.dt, w_eps, f3, dz, n_kernel, case.n_t, K, phi_lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["layered", "block", "uniform", "all"])
    ap.add_argument(
        "--fast", action="store_true", help="small kernel / single frequency (smoke test)"
    )
    args = ap.parse_args()
    cases = ["uniform", "layered", "block"] if args.case == "all" else [args.case]
    for name in cases:
        run_case(name, args.fast)
        print()


if __name__ == "__main__":
    main()
