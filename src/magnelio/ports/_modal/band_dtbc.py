"""Broadband band-subspace DTBC for inhomogeneous lines (WP-R4b).

Production form fixed by the WP-R4b certificate gate (session 89,
``validation/qtem_band_dtbc_certificate_spike.py``): the
**Galerkin-projected exterior**.  Let ``V`` be a real W-orthonormal
basis (W = ``M_eps`` over the period trace) of the tracked mode-family
traces over the frequency band (SVD rank ``p``, spike-measured 5-12).
The exterior of the port — the uniform continuation of the feed line —
is replaced by the projected half-line with period blocks

    D~_k = V^T W D_k V          (k = -1, 0, +1;  p x p),

which inherits the palindromic W-symmetry (``(W D_p1)^T = W D_m1`` =>
``D~_m1^T = D~_p1``).  The projected half-line is therefore itself a
lossless lattice: its exact small-system DTBC is passive **by
construction**, and the coupled full-interior + projected-boundary-
period system is block-symmetric lossless.  (The naive alternative —
projecting the *kernel*, ``U U^T W_t Lambda V V^T W`` — is weakly
active and was refuted in the gate; it is not implemented here.)

Chain orientation
-----------------

Everything in this module runs on the ``pairing="boundary"`` trace of
:func:`~magnelio.ports._modal.zeta_pencil.build_period_blocks`:
period ``p`` (inward plane indexing) holds ``(e_t(p), e_z(p-1/2))``,
so

* interior periods ``p >= 1`` consist of real mesh DOFs and couple to
  the boundary period 0 exclusively through the port-plane ``e_t(0)``
  (``D_m1`` has ``e_t`` columns only in this pairing);
* the boundary period 0 owns the virtual exterior half-plane
  ``e_z(-1/2)`` — it lives as the projected p-dimensional state
  ``xt`` inside :class:`BandDTBCBoundary`, never on the mesh;
* the exterior ghost period sits at ``p = -1``: outgoing radiation
  decays toward ``p -> -inf`` (multipliers ``|zeta| > 1`` of the
  pencil), so the ghost kernel is the exact DTBC kernel of the
  *swapped* projected pencil ``zeta^2 D~_m1 + zeta (D~_0 - sig) +
  D~_p1`` (contour QZ at size 2p — cheap at production scale).

Excitation is prescribed at the ghost period (the R2 scheme):
``ghost^n = s^n + sum_m L^out_m (xt - xt_inc)^{n-m}`` with
``xt_inc^n = sum_m L^in_m s^{n-m}``.  Unlike the scalar case the
incoming propagator differs from the outgoing one — the incident wave
decays toward ``+p`` (multipliers ``|zeta| < 1``), so ``L^in`` is the
kernel of the *unswapped* projected pencil.  Both kernels
auto-extend past the run length (powers of two), so within every run
the projected boundary is exact — no truncation, no passivity
question beyond the structural one settled above.

Certificates
------------

* :func:`galerkin_exterior` verifies the palindromic W-symmetry of
  the projected blocks at roundoff level and then enforces it exactly
  (the passivity-by-construction argument becomes exact in floating
  point).
* :func:`band_subspace` reports the family singular values; the rank
  cut is the certified subspace-capture parameter.
* :func:`band_apriori_reflection` evaluates the exact frequency-
  domain modal reflection of the Galerkin boundary (the matrix
  generalisation of the R1 |Gamma| formula: incoming-branch ansatz,
  N_t x N_t solve per frequency).  It requires the *dense* branch
  set of the full pencil — O((2N)^3) per frequency — and is meant
  for gate-sized cross-sections (tests, benchmark toys), not for
  routine port builds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.special import erfc, erfcinv

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.mesher import Mesh  # noqa: F401  (type reference)
from magnelio.ports._modal.discrete import DiscreteMode
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.port_report import PortOperatorReport
from magnelio.ports._modal.zeta_pencil import (
    PeriodChain,
    find_propagating_modes,
    normalize_gauge,
)

# On-circle tolerance for classifying propagating branches in the
# dense certificate evaluator (matches the spike).
_ONC_TOL = 1e-7

# Off-circle offset for certificate evaluations (the on-circle
# dichotomy is ambiguous for propagating branches; the induced bias
# floors the certificate EVALUATION near -130 dB — deep enough to
# certify the -100 dB criterion; TD floors validate below it).
_RHO_OFF = 1e-8

# Relative roundoff budget for the palindromic-symmetry certificate of
# the projected exterior blocks.  A genuine symmetry violation (wrong
# pairing, inconsistent metric) shows up at O(1).
_SYMMETRY_RTOL = 1e-10

# W-overlap threshold for mode-family continuation across the band
# grid: below this the branch is treated as a new family (fresh
# cut-on), not a continuation.
_TRACK_OVERLAP_MIN = 0.5


# ----------------------------------------------------------------------
# Exact solvent + matrix kernel (contour QZ)
# ----------------------------------------------------------------------


def stable_solvent(
    D_m1: np.ndarray,
    D_0: np.ndarray,
    D_p1: np.ndarray,
    sig_hat: complex,
) -> np.ndarray:
    """Stable solvent ``L`` of ``D_p1 L^2 + (D_0 - sig_hat) L + D_m1 = 0``.

    Ordered QZ on the linearised pencil; the blocks must be the
    *dimensionless* ``D = dt^2 B`` (entries O(r^2)) — the raw FIT
    scale ~1e23 breaks the pencil balancing (WP-R4 finding).  For
    ``|z| > 1`` (``sig_hat = 2 - z - 1/z``) exactly ``n`` of the
    ``2n`` finite eigenvalues are stable (dichotomy), and the solvent
    propagates solutions decaying toward ``+k``: ``x_{k+1} = L x_k``.

    Raises
    ------
    RuntimeError
        If the stable/unstable dichotomy is violated (evaluation too
        close to the unit circle, or a defective pencil).
    """
    n = D_0.shape[0]
    Ap = np.zeros((2 * n, 2 * n), dtype=complex)
    Bp = np.zeros((2 * n, 2 * n), dtype=complex)
    Ap[:n, n:] = np.eye(n)
    Ap[n:, :n] = -D_m1
    Ap[n:, n:] = -(D_0 - sig_hat * np.eye(n))
    Bp[:n, :n] = np.eye(n)
    Bp[n:, n:] = D_p1
    _, _, alpha, beta, _, Z = sla.ordqz(
        Ap,
        Bp,
        sort="iuc",
        output="complex",
    )
    n_stable = int(np.sum(np.abs(alpha) < np.abs(beta)))
    if n_stable != n:
        raise RuntimeError(
            f"solvent dichotomy violated: {n_stable} stable of {2 * n}",
        )
    X1 = Z[:n, :n]
    X2 = Z[n:, :n]
    return np.linalg.solve(X1.T, X2.T).T


def solvent_residual(
    D_m1: np.ndarray,
    D_0: np.ndarray,
    D_p1: np.ndarray,
    sig_hat: complex,
    lam: np.ndarray,
) -> float:
    """Relative residual of a solvent candidate (certificate probe)."""
    r = D_p1 @ lam @ lam + (D_0 - sig_hat * np.eye(D_0.shape[0])) @ lam + D_m1
    return float(
        np.linalg.norm(r) / max(np.linalg.norm(D_m1), 1e-300),
    )


# ----------------------------------------------------------------------
# Contour loop: parallel over processes above a measured size threshold
# ----------------------------------------------------------------------

# Spawning a pool costs a fixed ~1.2 s (measured, 8 spawned
# interpreters), while the loop itself divides cleanly by the worker
# count.  Below roughly three seconds of serial work the constant wins,
# so the loop stays sequential there.  Calibrated from the per-point
# cost of one solvent evaluation: ~0.061 ms at p = 4, ~0.31 ms at p = 12
# (internal record, section 13d), i.e. ~p^1.5 over the range that
# matters.
_CONTOUR_PARALLEL_MIN_WORK = 4.0e5
_CONTOUR_MAX_WORKERS = 8

_CONTOUR_BLOCKS: dict = {}


def _contour_worker_init(D_m1, D_0, D_p1):  # pragma: no cover - subprocess
    """Pin BLAS to one thread per worker and keep the blocks resident."""
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    _CONTOUR_BLOCKS["abc"] = (D_m1, D_0, D_p1)


def _contour_worker(task):  # pragma: no cover - subprocess
    """Solvents over one contiguous run of contour points."""
    j0, j1, n_fft, rho = task
    D_m1, D_0, D_p1 = _CONTOUR_BLOCKS["abc"]
    n = D_0.shape[0]
    out = np.empty((j1 - j0, n, n), dtype=complex)
    for i, j in enumerate(range(j0, j1)):
        z = rho * np.exp(2j * math.pi * j / n_fft)
        out[i] = stable_solvent(D_m1, D_0, D_p1, 2.0 - z - 1.0 / z)
    return out


def _contour_workers(n_points: int, p: int) -> int:
    """Worker count for a contour loop, 1 when a pool would not pay."""
    import multiprocessing
    import os

    if n_points * p**1.5 < _CONTOUR_PARALLEL_MIN_WORK:
        return 1
    # Already inside somebody else's worker (a parameter sweep, say):
    # a nested pool multiplies the process count, not the throughput.
    if multiprocessing.parent_process() is not None:
        return 1
    return max(1, min(_CONTOUR_MAX_WORKERS, (os.cpu_count() or 1) - 1))


def _contour_spectrum(D_m1, D_0, D_p1, n_fft: int, rho: float, half: int) -> np.ndarray:
    """The stable solvent at every contour point, in parallel when it pays.

    Bit-identical to the sequential loop: the points are independent and
    each is solved by the same call, so splitting them changes nothing
    but the wall clock.
    """
    n = D_0.shape[0]
    n_workers = _contour_workers(half, n)
    if n_workers > 1:
        try:
            import multiprocessing  # noqa: PLC0415
            from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

            edges = np.linspace(0, half, 4 * n_workers + 1).astype(int)
            tasks = [
                (int(edges[i]), int(edges[i + 1]), n_fft, rho)
                for i in range(4 * n_workers)
                if edges[i + 1] > edges[i]
            ]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_contour_worker_init,
                initargs=(D_m1, D_0, D_p1),
            ) as ex:
                return np.concatenate(list(ex.map(_contour_worker, tasks)), axis=0)
        except (OSError, RuntimeError, ImportError):
            # No usable process pool (restricted sandbox, frozen app,
            # exhausted file descriptors): the sequential loop is always
            # available and gives the same numbers.
            pass
    spec = np.empty((half, n, n), dtype=complex)
    for j in range(half):
        z = rho * np.exp(2j * math.pi * j / n_fft)
        spec[j] = stable_solvent(D_m1, D_0, D_p1, 2.0 - z - 1.0 / z)
    return spec


def matrix_dtbc_kernel(
    D_m1: np.ndarray,
    D_0: np.ndarray,
    D_p1: np.ndarray,
    n_kernel: int,
) -> tuple[np.ndarray, dict]:
    """Exact matrix DTBC kernel ``L_m`` of a uniform vector chain.

    Laurent coefficients of the stable solvent around infinity via
    contour integration — the matrix form of
    :func:`~magnelio.ports._modal.dtbc.dtbc_kernel`, same contour
    parameters (8x oversampling, ``rho^{n_kernel} = e^4``), with the
    conjugate symmetry of the real chain halving the solvent count.
    ``L_0 = 0`` analytically (the solvent vanishes at infinity).

    Parameters
    ----------
    D_m1, D_0, D_p1 : np.ndarray
        Dense dimensionless period blocks (size p x p for the
        projected exterior; the contour QZ is O(p^3) per point).
    n_kernel : int
        Number of kernel samples.

    Returns
    -------
    L : np.ndarray
        Real kernel of shape ``(n_kernel, n, n)``.
    cert : dict
        ``residual`` — max sampled solvent residual, ``imag`` — max
        imaginary part discarded by the real cast, ``l0`` — max
        |L_0| (all should sit at roundoff level).
    """
    if n_kernel < 2:
        raise ValueError("n_kernel must be >= 2")
    n = D_0.shape[0]
    n_fft = 8 * n_kernel
    rho = math.exp(4.0 / n_kernel)
    half = n_fft // 2 + 1
    spec = _contour_spectrum(D_m1, D_0, D_p1, n_fft, rho, half)
    res_max = 0.0
    probe_every = max(half // 16, 1)
    for j in range(0, half, probe_every):
        z = rho * np.exp(2j * np.pi * j / n_fft)
        res_max = max(
            res_max,
            solvent_residual(D_m1, D_0, D_p1, 2.0 - z - 1.0 / z, spec[j]),
        )
    full = np.empty((n_fft, n, n), dtype=complex)
    full[:half] = spec
    full[half:] = np.conj(spec[1:-1][::-1])
    coeff = np.fft.ifft(full, axis=0)[:n_kernel]
    imag_max = float(np.abs(coeff.imag).max())
    L = coeff.real * (rho ** np.arange(n_kernel))[:, None, None]
    return L, dict(
        residual=res_max,
        imag=imag_max,
        l0=float(np.abs(L[0]).max()),
    )


# ----------------------------------------------------------------------
# Band mode families and subspace
# ----------------------------------------------------------------------


@dataclass
class BandModeFamily:
    """One tracked propagating mode family over the band grid.

    ``freqs[i]``, ``zetas[i]``, ``traces[:, i]`` belong together;
    traces are W-normalised and gauge-fixed
    (:func:`~magnelio.ports._modal.zeta_pencil.normalize_gauge`).
    """

    freqs: np.ndarray
    zetas: np.ndarray
    traces: np.ndarray = field(repr=False)

    @property
    def f_first(self) -> float:
        return float(self.freqs[0])

    def nearest(self, f: float) -> tuple[float, complex, np.ndarray]:
        """(f_grid, zeta, trace) of the grid point closest to ``f``."""
        i = int(np.argmin(np.abs(self.freqs - f)))
        return float(self.freqs[i]), complex(self.zetas[i]), self.traces[:, i]


def track_band_families(
    chain: PeriodChain,
    dt: float,
    f_grid: np.ndarray,
    track_t: np.ndarray,
    eps_eff_hint: float,
    dz: float,
    k: int = 8,
) -> list[BandModeFamily]:
    """Track the propagating mode families across the band grid.

    Per grid frequency all propagating modes are found by the sparse
    arc-target eigensolve
    (:func:`~magnelio.ports._modal.zeta_pencil.find_propagating_modes`)
    and assigned to families by W-overlap continuation with each
    family's previous profile (threshold ``0.5``); unmatched branches
    open new families (fresh cut-ons).  The returned list puts the
    fundamental — the family with maximal ``e_t`` overlap with
    ``track_t`` at its first point — first, the rest ordered by their
    first propagating grid frequency.

    Parameters
    ----------
    chain : PeriodChain
        Period blocks at the port (either pairing).
    dt : float
        Solver time step [s].
    f_grid : np.ndarray
        Ascending band frequencies [Hz].
    track_t : np.ndarray
        Tangential bootstrap profile (free DOFs) identifying the
        fundamental — typically the DC Laplace solution.
    eps_eff_hint : float
        Effective-permittivity lower bound for the fundamental's
        phase-advance arc hint (the DC value; normal dispersion is
        covered by the 1.3 margin and by continuation updates).
    dz : float
        Port-normal cell size [m].

    Raises
    ------
    ValueError
        If no propagating mode exists at the lowest grid frequency.
    """
    from magnelio.constants import C0

    w = chain.w_period
    w_t = w[: chain.n_t]
    families: list[dict] = []
    eps_run = float(eps_eff_hint)
    for f in np.asarray(f_grid, dtype=float):
        w_dt = 2.0 * math.pi * f * dt
        theta0 = 2.0 * math.pi * f * math.sqrt(eps_run) / C0 * dz
        zp, pp = find_propagating_modes(chain, w_dt, 1.3 * theta0, k=k)
        if zp.size == 0:
            if not families:
                raise ValueError(
                    f"no propagating mode at the lowest band frequency {f / 1e9:.3f} GHz",
                )
            continue
        assigned = np.zeros(zp.size, dtype=bool)
        for fam in families:
            ov = np.abs(
                np.conj(fam["last_phi"]) @ (w[:, None] * pp),
            )
            ov[assigned] = -1.0
            j = int(np.argmax(ov))
            if ov[j] < _TRACK_OVERLAP_MIN:
                continue
            assigned[j] = True
            phi = normalize_gauge(pp[:, j], chain.n_t)
            fam["last_phi"] = phi
            fam["freqs"].append(f)
            fam["zetas"].append(complex(zp[j]))
            fam["traces"].append(phi)
        for j in np.flatnonzero(~assigned):
            phi = normalize_gauge(pp[:, j], chain.n_t)
            families.append(
                dict(
                    last_phi=phi,
                    freqs=[f],
                    zetas=[complex(zp[j])],
                    traces=[phi],
                )
            )
        # Continuation update of the arc hint from the fundamental
        # candidate (largest phase advance at this frequency).
        theta_max = float(np.abs(np.angle(zp)).max())
        s_ratio = (math.sin(theta_max / 2.0) / (dz / 2.0)) / (math.sin(w_dt / 2.0) / (dt / 2.0))
        eps_run = max(eps_run, (C0 * s_ratio) ** 2)

    out = [
        BandModeFamily(
            freqs=np.array(fam["freqs"]),
            zetas=np.array(fam["zetas"]),
            traces=np.stack(fam["traces"], axis=1),
        )
        for fam in families
    ]
    track_n = track_t / math.sqrt(
        float(np.dot(w_t, np.abs(track_t) ** 2)),
    )
    ov0 = [float(np.abs(np.conj(track_n) @ (w_t * fam.traces[: chain.n_t, 0]))) for fam in out]
    i_fund = int(np.argmax(ov0))
    rest = sorted(
        (i for i in range(len(out)) if i != i_fund),
        key=lambda i: out[i].f_first,
    )
    return [out[i_fund]] + [out[i] for i in rest]


def continue_family_to_dc(
    chain: PeriodChain,
    dt: float,
    fam: BandModeFamily,
    track_t: np.ndarray,
    eps_eff_hint: float,
    dz: float,
    n_extra: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue the fundamental family below its grid, down to DC.

    The QTEM fundamental propagates at every ``f > 0`` and converges
    to the static Laplace mode as ``f -> 0`` — measured on the shielded
    microstrip, the W-overlap between ``track_t`` and the tracked
    trace is 0.999972 at 2.25 GHz and 1.000000 (six digits) at 20 MHz.
    The DC limit itself needs no eigensolve: it *is* ``track_t``,
    with a vanishing longitudinal part and ``zeta = 1`` (no phase
    advance per period).

    This continuation exists for the *excitation direction* only.  It
    deliberately leaves the family, the subspace and the recording
    channels untouched: the Galerkin boundary built over the tracked
    band is already accurate far below it (a boundary built on
    2.25-18 GHz absorbs the fundamental at -83.4 dB at 74.6 MHz), so
    the band limit of a band run is bookkeeping about where the
    excitation direction is tabulated, not a property of the
    boundary.

    Parameters
    ----------
    chain : PeriodChain
        Period blocks at the port (boundary pairing).
    dt : float
        Solver time step [s].
    fam : BandModeFamily
        The fundamental family (first entry of
        :func:`track_band_families`).
    track_t : np.ndarray
        Tangential DC profile (free DOFs) — the Laplace solution.
    eps_eff_hint : float
        DC effective permittivity, for the phase-advance arc hint.
    dz : float
        Port-normal cell size [m].
    n_extra : int
        Number of continuation points, including the DC endpoint.

    Returns
    -------
    freqs : np.ndarray
        Ascending frequencies from 0 up to (excluding) ``fam.freqs[0]``.
    traces : np.ndarray
        Matching traces, W-normalised and gauge-fixed like the
        family's own, column ``0`` being the static limit.
    """
    from magnelio.constants import C0

    n_t = chain.n_t
    w = chain.w_period
    w_t = w[:n_t]
    f_lo = float(fam.freqs[0])
    n_extra = max(int(n_extra), 1)
    f_below = np.linspace(0.0, f_lo, n_extra + 1)[:-1]

    # DC endpoint in closed form: the Laplace trace, no longitudinal
    # part, scaled to the family's W-normalisation.
    dc = np.zeros(w.size, dtype=complex)
    dc[:n_t] = np.asarray(track_t, dtype=float)
    scale = math.sqrt(float(np.dot(w, np.abs(fam.traces[:, 0]) ** 2))) / math.sqrt(
        float(np.dot(w_t, np.abs(track_t) ** 2)),
    )
    cols = [normalize_gauge(dc * scale, n_t)]

    # The rest is measured, continued upward toward the tracked grid
    # so each solve is seeded by the arc of the point below it.
    prev = cols[0]
    for f in f_below[1:]:
        w_dt = 2.0 * math.pi * f * dt
        theta0 = 2.0 * math.pi * f * math.sqrt(eps_eff_hint) / C0 * dz
        zp, pp = find_propagating_modes(chain, w_dt, 1.3 * theta0, k=4)
        if zp.size == 0:
            raise ValueError(
                f"the fundamental family does not continue to {f / 1e9:.4g} GHz "
                f"below its tracked band — the DC excitation anchor needs it "
                f"down to 0",
            )
        ov = np.abs(np.conj(prev) @ (w[:, None] * pp))
        j = int(np.argmax(ov))
        prev = normalize_gauge(pp[:, j], n_t)
        cols.append(prev)

    return f_below, np.stack(cols, axis=1)


def band_subspace(
    families: list[BandModeFamily],
    w: np.ndarray,
    p: int | None = None,
    svd_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Real W-orthonormal band subspace of the tracked families.

    The TD kernel is real, so the subspace is the *real* span of the
    family traces (``Re phi`` and ``Im phi`` columns; in the fixed
    gauge ``e_t`` is ~real and ``e_z`` ~imaginary, so both parts
    carry content and the rank roughly doubles the complex family
    rank — WP-R4b spike).

    Parameters
    ----------
    families : list[BandModeFamily]
        Tracked families (:func:`track_band_families`).
    w : np.ndarray
        W metric (``M_eps`` over the period trace).
    p : int or None
        Subspace rank.  ``None`` selects the smallest rank whose
        discarded singular values fall below ``svd_tol`` relative to
        the largest.
    svd_tol : float
        Relative singular-value threshold for the automatic rank.

    Returns
    -------
    V : np.ndarray
        ``(n, p)`` real, ``V^T W V = I``.
    sv : np.ndarray
        All relative singular values of the family matrix (the
        subspace-capture certificate; ``sv[p:]`` is the discarded
        tail).
    """
    cols = [
        part
        for fam in families
        for i in range(fam.freqs.size)
        for part in (fam.traces[:, i].real, fam.traces[:, i].imag)
    ]
    A = np.column_stack(cols)
    sw = np.sqrt(np.asarray(w, dtype=float))
    Uw, s, _ = np.linalg.svd(sw[:, None] * A, full_matrices=False)
    sv = s / s[0]
    if p is None:
        p = int(np.sum(sv >= svd_tol))
    p = max(1, min(p, Uw.shape[1]))
    V = Uw[:, :p] / sw[:, None]
    return V, sv


# ----------------------------------------------------------------------
# Galerkin-projected exterior
# ----------------------------------------------------------------------


@dataclass
class BandExterior:
    """Projected exterior blocks + the coupling maps.

    ``Dt_m1 = Dt_p1^T`` and ``Dt_0 = Dt_0^T`` exactly (verified at
    roundoff level, then enforced) — the projected half-line is a
    lossless lattice and its exact DTBC is passive by construction.
    """

    V: np.ndarray = field(repr=False)
    VtW: np.ndarray = field(repr=False)
    Dt_m1: np.ndarray = field(repr=False)
    Dt_0: np.ndarray = field(repr=False)
    Dt_p1: np.ndarray = field(repr=False)
    sym_residual: float = 0.0

    @property
    def p(self) -> int:
        return self.Dt_0.shape[0]


def galerkin_exterior(chain: PeriodChain, V: np.ndarray) -> BandExterior:
    """Galerkin-project the exterior period blocks onto the subspace.

    ``D~_k = V^T W D_k V``.  The palindromic W-symmetry of the full
    blocks makes ``D~_m1 = D~_p1^T`` and ``D~_0`` symmetric; this is
    verified against ``_SYMMETRY_RTOL`` (a violation means the trace
    pairing or the metric is inconsistent — the boundary would not be
    passive by construction) and then enforced exactly.

    Raises
    ------
    ValueError
        If the symmetry certificate fails.
    """
    w = np.asarray(chain.w_period, dtype=float)
    VtW = (V * w[:, None]).T
    Dm = VtW @ (chain.D_m1 @ V)
    D0 = VtW @ (chain.D_0 @ V)
    Dp = VtW @ (chain.D_p1 @ V)
    ref = max(float(np.linalg.norm(Dp)), 1e-300)
    res = (
        max(
            float(np.linalg.norm(Dp - Dm.T)),
            float(np.linalg.norm(D0 - D0.T)),
        )
        / ref
    )
    if res > _SYMMETRY_RTOL:
        raise ValueError(
            "projected exterior blocks violate the palindromic "
            f"W-symmetry (residual {res:.2e} > {_SYMMETRY_RTOL:.0e}) "
            "— the Galerkin boundary would not be passive by "
            "construction",
        )
    D0 = 0.5 * (D0 + D0.T)
    Dm = Dp.T.copy()
    return BandExterior(
        V=V,
        VtW=VtW,
        Dt_m1=Dm,
        Dt_0=D0,
        Dt_p1=Dp,
        sym_residual=res,
    )


# ----------------------------------------------------------------------
# Boundary state machine
# ----------------------------------------------------------------------


class BandDTBCBoundary:
    """Projected boundary-period state + exact small-system DTBC.

    Owns the p-dimensional boundary state ``xt`` (the Galerkin
    coordinates of the boundary period: port-plane ``e_t`` plus the
    virtual exterior ``e_z`` half-plane), the scattered/source
    histories and the two ghost kernels (module docstring).  One
    :meth:`advance` call performs one leapfrog step of the projected
    boundary period:

        xt^{n+1} = 2 xt^n - xt^{n-1}
                   - (coup^n + D~_0 xt^n + D~_m1 ghost^n)

    with ``ghost^n = s^n + sum_{m>=1} L^out_m (xt - xt_inc)^{n-m}``
    and ``xt_inc^n = sum_{m>=1} L^in_m s^{n-m}``; ``coup^n`` is the
    projected coupling to the first full interior period, supplied by
    the caller.  Kernels auto-extend past the run length (powers of
    two) — within a run the boundary is exact.

    Parameters
    ----------
    exterior : BandExterior
        Projected exterior blocks (symmetry-certified).
    n_kernel_init : int, default 2048
        Initial kernel length; grows on demand.
    """

    def __init__(
        self,
        exterior: BandExterior,
        *,
        n_kernel_init: int = 2048,
    ) -> None:
        self._Dm1 = exterior.Dt_m1
        self._D0 = exterior.Dt_0
        self._Dp1 = exterior.Dt_p1
        self._p = exterior.p
        self._n_kernel = int(n_kernel_init)
        self._need_in_kernel = False
        self._rebuild_kernels()

        self._xt = np.zeros(self._p)
        self._xt_prev = np.zeros(self._p)
        self._n = 0
        self._w_hist = np.zeros((self._n_kernel, self._p))
        self._s_hist = np.zeros((self._n_kernel, self._p))

    def _rebuild_kernels(self) -> None:
        # Ghost kernel: outgoing radiation decays toward -p, i.e. the
        # stable solvent of the SWAPPED projected pencil.
        L_out, cert_out = matrix_dtbc_kernel(
            self._Dp1,
            self._D0,
            self._Dm1,
            self._n_kernel,
        )
        self._kflip_out = L_out[1:][::-1].copy()
        self.kernel_certificate = dict(out=cert_out)
        # Incoming kernel: the incident wave prescribed at the ghost
        # decays toward +p — the UNSWAPPED pencil.  Only excited
        # ports need it; built lazily (halves the contour-QZ cost of
        # passive ports).
        self._kflip_in: np.ndarray | None = None
        if self._need_in_kernel:
            self._build_in_kernel()

    def _build_in_kernel(self) -> None:
        L_in, cert_in = matrix_dtbc_kernel(
            self._Dm1,
            self._D0,
            self._Dp1,
            self._n_kernel,
        )
        self._kflip_in = L_in[1:][::-1].copy()
        self.kernel_certificate["into"] = cert_in
        self._need_in_kernel = True

    def require_in_kernel(self) -> None:
        """Ensure the incoming (excitation) kernel exists."""
        if self._kflip_in is None:
            self._build_in_kernel()

    def _ensure_capacity(self, n: int) -> None:
        if n + 1 > self._n_kernel:
            while self._n_kernel < n + 1:
                self._n_kernel *= 2
            self._rebuild_kernels()
        if n + 1 > self._w_hist.shape[0]:
            grow = max(2 * self._w_hist.shape[0], n + 1)
            for name in ("_w_hist", "_s_hist"):
                old = getattr(self, name)
                new = np.zeros((grow, self._p))
                new[: old.shape[0]] = old
                setattr(self, name, new)

    @property
    def p(self) -> int:
        return self._p

    @property
    def step_count(self) -> int:
        """Completed boundary steps ``n`` (chain time of the state)."""
        return self._n

    @property
    def xt(self) -> np.ndarray:
        """Current boundary state ``xt^n`` (last value returned)."""
        return self._xt

    def advance(
        self,
        coup: np.ndarray,
        src: np.ndarray | None = None,
    ) -> np.ndarray:
        """One boundary leapfrog step; returns ``xt^{n+1}``.

        Parameters
        ----------
        coup : np.ndarray
            Projected interior coupling ``(V^T W D_p1) x_1^n`` at the
            boundary's own time level (the full trace of the first
            interior period one solver step ago).
        src : np.ndarray or None
            Incident p-vector ``s^n`` prescribed at the ghost period;
            ``None`` for a passive port.
        """
        n = self._n
        self._ensure_capacity(n)
        if src is not None and self._kflip_in is None:
            raise RuntimeError(
                "ghost source supplied but the incoming kernel was "
                "never built — call require_in_kernel() (the port "
                "operator does this in set_excitation)",
            )
        if src is None:
            src = np.zeros(self._p)

        if n == 0:
            xt_inc = np.zeros(self._p)
            conv_w = np.zeros(self._p)
        else:
            # einsum keeps the reduction single-threaded SIMD — the
            # DD-054 BLAS-threading lesson applies per matrix row.
            kf_out = self._kflip_out[self._kflip_out.shape[0] - n :]
            conv_w = np.einsum("mij,mj->i", kf_out, self._w_hist[:n])
            if self._kflip_in is not None:
                kf_in = self._kflip_in[self._kflip_in.shape[0] - n :]
                xt_inc = np.einsum("mij,mj->i", kf_in, self._s_hist[:n])
            else:
                xt_inc = np.zeros(self._p)

        ghost = src + conv_w
        self._w_hist[n] = self._xt - xt_inc
        self._s_hist[n] = src

        xt_new = 2.0 * self._xt - self._xt_prev - (coup + self._D0 @ self._xt + self._Dm1 @ ghost)
        self._xt_prev = self._xt
        self._xt = xt_new
        self._n = n + 1
        return xt_new

    def reset_state(self) -> None:
        """Zero the boundary state and histories for a fresh run.

        Keeps the (expensive, contour-QZ) kernels and the projected
        blocks — the boundary behaves as freshly built at a fraction
        of the construction cost.  Grown history/kernel capacity is
        retained.
        """
        self._xt = np.zeros(self._p)
        self._xt_prev = np.zeros(self._p)
        self._n = 0
        self._w_hist[:] = 0.0
        self._s_hist[:] = 0.0

    def state_dict(self) -> dict:
        """Checkpoint the projected boundary state and both histories.

        The convolution reaches back over the *whole* record, so unlike
        a ring-buffer boundary this state grows with the run: the two
        histories are ``n x p`` each.  That is small in absolute terms
        (12288 steps at p = 12 is 2.4 MB together) and it is the entire
        memory of the boundary — the kernels and projected blocks are
        constant and rebuilt from the recipe.

        Only the filled prefix is written; the rest of the capacity is
        zero by construction and would just pad the checkpoint.
        """
        n = self._n
        return {
            "xt": self._xt.copy(),
            "xt_prev": self._xt_prev.copy(),
            "n": int(n),
            "w_hist": self._w_hist[:n].copy(),
            "s_hist": self._s_hist[:n].copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state written by :meth:`state_dict` (bit-exact resume).

        The kernels are *not* in the checkpoint: they are a deterministic
        function of the projected exterior blocks, which the resuming run
        rebuilds from the same recipe.  That determinism is a property
        the pencil eigensolve only acquired with a fixed ARPACK start
        vector (KB-037) — without it the rebuilt subspace differs from
        the recorded one and this restore would be silently inconsistent.
        """
        n = int(sd["n"])
        p = int(np.asarray(sd["xt"]).size)
        if p != self._p:
            raise ValueError(
                f"checkpoint holds a boundary state of rank {p}, but the "
                f"rebuilt band port has rank {self._p} — the subspace it "
                f"was recorded in is not the one being restored into",
            )
        self._ensure_capacity(n)
        self._xt = np.asarray(sd["xt"], dtype=float).copy()
        self._xt_prev = np.asarray(sd["xt_prev"], dtype=float).copy()
        self._w_hist[:] = 0.0
        self._s_hist[:] = 0.0
        if n:
            self._w_hist[:n] = np.asarray(sd["w_hist"], dtype=float)
            self._s_hist[:n] = np.asarray(sd["s_hist"], dtype=float)
            if np.any(self._s_hist[:n]) and self._kflip_in is None:
                raise RuntimeError(
                    "checkpoint carries a ghost source history but the "
                    "incoming kernel was never built — set the excitation "
                    "before loading the state",
                )
        self._n = n


@dataclass
class BandPortData:
    """Everything the pulsed-broadband postprocessing needs per port.

    Attached by ``build_band_dtbc_port`` as ``op.band_data`` and
    consumed by :func:`~magnelio.post.modal_sparameters.
    compute_band_s_parameters`: the inward-paired chain feeds the
    per-frequency true-mode solves and phasor synthesis (the WP-R4a
    machinery), the families supply the arc hints and channel
    assignment, and the stored matrices/profiles reproduce the
    recorder's projection exactly.
    """

    f_band: tuple[float, float]
    f_grid: np.ndarray
    families: list[BandModeFamily]
    singular_values: np.ndarray
    p: int
    chain_inward: PeriodChain
    chain_boundary: PeriodChain
    exterior: BandExterior = field(repr=False)
    plane: PortPlane = field(repr=False)
    m_eps: np.ndarray = field(repr=False)
    m_mu: np.ndarray = field(repr=False)
    c_3d: sp.spmatrix = field(repr=False)
    dual_e_profiles: list = field(repr=False)
    eps_eff_dc: float = 0.0
    z_line: float | None = None
    solve_seconds: float = 0.0


@dataclass
class BandDecomposition:
    """The band postprocessing input of one port, detached from its operator.

    :func:`~magnelio.post.modal_sparameters.compute_band_s_parameters`
    reads a live :class:`PortOperatorBandDTBC` for two different kinds
    of thing: quantities that belong to *this port* — its chain, its
    plane, its recording profiles, its tracked family — and quantities
    that are properties of the *mesh* and merely happen to be cached on
    the port (``M_eps``, ``M_mu``, the 3D curl).  Only the first kind
    identifies a run; the second is rebuilt from the grid by the same
    three builders that produced it.

    Separating them is what makes a band run readable from a project
    store: the port-side data serialises to a few hundred KiB of plain
    arrays (measured 0.26 KiB per free tangential DOF, linear in the
    cross-section), while the mesh-side operators never need to be
    written at all.  Crucially, none of the expensive construction —
    the contour-QZ ghost kernels, the SVD subspace — appears here: it
    drives the time stepping and has no part in the decomposition.
    """

    name: str
    n_modes: int
    chain_inward: PeriodChain
    plane: PortPlane = field(repr=False)
    family_freqs: np.ndarray = field(repr=False, default=None)
    family_zetas: np.ndarray = field(repr=False, default=None)
    e_u_profiles: list = field(repr=False, default_factory=list)
    e_v_profiles: list = field(repr=False, default_factory=list)
    h_u_profiles: list = field(repr=False, default_factory=list)
    h_v_profiles: list = field(repr=False, default_factory=list)
    dual_e_profiles: list = field(repr=False, default_factory=list)

    @classmethod
    def from_operator(cls, op) -> "BandDecomposition":
        """Extract the decomposition input from a built band port."""
        bd = getattr(op, "band_data", None)
        if bd is None:
            raise ValueError(
                f"port {op.name!r} carries no band_data — build it with build_band_dtbc_port",
            )
        fam0 = bd.families[0]
        return cls(
            name=op.name,
            n_modes=int(op.n_modes),
            chain_inward=bd.chain_inward,
            plane=bd.plane,
            family_freqs=np.asarray(fam0.freqs, dtype=float),
            family_zetas=np.asarray(fam0.zetas, dtype=complex),
            e_u_profiles=[np.asarray(dm.e_u_profile) for dm in op.discrete_modes],
            e_v_profiles=[np.asarray(dm.e_v_profile) for dm in op.discrete_modes],
            h_u_profiles=[np.asarray(dm.h_u_profile) for dm in op.discrete_modes],
            h_v_profiles=[np.asarray(dm.h_v_profile) for dm in op.discrete_modes],
            dual_e_profiles=[(np.asarray(du), np.asarray(dv)) for du, dv in bd.dual_e_profiles],
        )


# ----------------------------------------------------------------------
# Port operator (Port protocol)
# ----------------------------------------------------------------------


def band_source_spectrum(
    f_span: tuple[float, float],
    f_subspace: tuple[float, float],
    dt: float,
    n_syn: int,
    *,
    skirt: float = 1e-7,
) -> np.ndarray:
    """Erfc-product flat-band source spectrum on the rfft grid.

    The scalar spectrum behind :meth:`PortOperatorBandDTBC.
    set_excitation_band`: flat over ``f_span``, Gaussian-class
    roll-offs reaching ``skirt`` at the subspace band edges
    ``f_subspace``, hard zero outside, and a linear phase centring
    the pulse in the synthesis window.  Extracted as a module
    function so callers (the analyses) can synthesise the *scalar*
    reference waveform ``irfft(W_hat, n_syn)`` of a band excitation
    identically to the port's internal source.

    Parameters
    ----------
    f_span : (float, float)
        Flat measurement span [Hz]; must lie strictly inside
        ``f_subspace``.
    f_subspace : (float, float)
        The channel's tracked subspace band [Hz]
        (``op.channel_band(mode_idx)``).
    dt : float
        Solver time step [s].
    n_syn : int
        Synthesis window length in steps.
    skirt : float, default 1e-7
        Relative spectral amplitude at the subspace band edges.

    Returns
    -------
    np.ndarray
        Complex spectrum on ``np.fft.rfftfreq(n_syn, dt)``.
    """
    f_lo, f_hi = float(f_subspace[0]), float(f_subspace[1])
    f1, f2 = float(f_span[0]), float(f_span[1])
    if not (f_lo <= f1 < f2 < f_hi):
        raise ValueError(
            f"f_span {f_span} must lie strictly inside the subspace band ({f_lo:.3e}, {f_hi:.3e})",
        )
    if f_lo > 0.0 and f1 <= f_lo:
        raise ValueError(
            f"f_span {f_span} must lie strictly inside the subspace band ({f_lo:.3e}, {f_hi:.3e})",
        )
    x_skirt = float(erfcinv(2.0 * skirt))
    sig_hi = (f_hi - f2) / (math.sqrt(2.0) * x_skirt)
    f_bins = np.fft.rfftfreq(n_syn, dt)
    if f_lo > 0.0:
        sig_lo = (f1 - f_lo) / (math.sqrt(2.0) * x_skirt)
        env_lo = erfc((f1 - f_bins) / (math.sqrt(2.0) * sig_lo))
    else:
        # DC-anchored channel: there is no band edge below f1 to roll
        # off against, so the spectrum stays flat down to the first
        # bin.  This is what unchains the pulse length from f_span[0]
        # — the duration is then set by the upper roll-off alone.
        env_lo = np.full(f_bins.size, 2.0)
    env = 0.25 * env_lo * erfc((f_bins - f2) / (math.sqrt(2.0) * sig_hi))
    env[f_bins > f_hi] = 0.0
    if f_lo > 0.0:
        env[f_bins < f_lo] = 0.0
    t_c = 0.5 * n_syn * dt
    return env * np.exp(-2j * np.pi * f_bins * t_c)


class PortOperatorBandDTBC:
    """Broadband band-subspace DTBC port on one face of the FIT mesh.

    Implements the :class:`magnelio.ports.base.Port` protocol.  The
    port plane's tangential E is forced into the subspace image
    ``V_t xt`` each step (the modal-overwrite pattern of
    ``PortOperatorModal``; the unprojected remainder was certified
    not to cap the floor — WP-R4b gate, certificate ii), the
    projected boundary period evolves through
    :class:`BandDTBCBoundary`, and the recorder-facing V/I channels
    project with fixed per-family profiles (dual-basis for
    multi-channel ports) whose frequency-resolved a/b decomposition
    happens in postprocessing
    (:func:`~magnelio.post.modal_sparameters.
    compute_band_s_parameters`).

    Built by ``build_band_dtbc_port``; not intended for direct
    construction.

    Parameters
    ----------
    name : str
        Port identifier.
    plane : PortPlane
        Port-plane geometry.
    chain : PeriodChain
        Boundary-paired period blocks (``pairing="boundary"``).
    exterior : BandExterior
        Galerkin-projected exterior (symmetry-certified).
    boundary : BandDTBCBoundary
        Boundary state machine built on ``exterior``.
    discrete_modes : list[DiscreteMode]
        Fixed recording channels (one per tracked family).
    m_eps_flat, m_mu_flat : np.ndarray
        Production mass-matrix diagonals (port-plane-flattened
        ``m_eps``).
    dt : float
        Solver time step [s].
    src_directions : list of (np.ndarray, np.ndarray)
        Per channel ``(freqs, U)`` with ``U`` of shape
        ``(n_grid, p)`` complex: the gauge-aligned subspace
        coordinates ``V^T W phi_f`` of the channel's family over the
        tracking grid.  :meth:`set_excitation` synthesises the
        frequency-tracked ghost source from them — a fixed source
        direction launches, away from its reference frequency, a
        wave whose profile deficit against the true mode excites an
        evanescent interface halo at the measurement plane at the
        *profile-drift* level (measured −40 dB class at band edges);
        tracking the family direction per frequency pushes the halo
        down to the subspace-capture level.
    dual_e_profiles : list of (np.ndarray, np.ndarray) or None
        Dual-basis projection profiles per channel (Gram inverse);
        ``None`` falls back to the primal profiles.
    """

    def __init__(
        self,
        name: str,
        plane: PortPlane,
        chain: PeriodChain,
        exterior: BandExterior,
        boundary: BandDTBCBoundary,
        discrete_modes: list[DiscreteMode],
        m_eps_flat: np.ndarray,
        m_mu_flat: np.ndarray,
        dt: float,
        src_directions: list[tuple[np.ndarray, np.ndarray]],
        dual_e_profiles: (list[tuple[np.ndarray, np.ndarray] | None] | None) = None,
        port_report: PortOperatorReport | None = None,
    ) -> None:
        if chain.pairing != "boundary":
            raise ValueError(
                "PortOperatorBandDTBC requires a boundary-paired "
                f"chain, got pairing={chain.pairing!r}",
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.name = name
        self.plane = plane
        self.port_report = port_report
        self.discrete_modes = list(discrete_modes)
        self._n_modes = len(self.discrete_modes)
        self._dt = float(dt)
        self._chain = chain
        self._exterior = exterior
        self._boundary = boundary
        if len(src_directions) != self._n_modes:
            raise ValueError(
                "src_directions must have one (freqs, U) entry per "
                f"mode ({self._n_modes}), got {len(src_directions)}",
            )
        self._src_directions = src_directions
        if dual_e_profiles is not None and len(dual_e_profiles) != self._n_modes:
            raise ValueError(
                "dual_e_profiles must have one entry per mode "
                f"({self._n_modes}), got {len(dual_e_profiles)}",
            )
        self._dual_e_profiles = dual_e_profiles

        # Projection metric slices (recorder-facing V/I channels).
        self._me_u_port = np.asarray(m_eps_flat[plane.e_u_indices], dtype=float)
        self._me_v_port = np.asarray(m_eps_flat[plane.e_v_indices], dtype=float)
        self._me_u_int = np.asarray(m_eps_flat[plane.e_u_indices_interior], dtype=float)
        self._me_v_int = np.asarray(m_eps_flat[plane.e_v_indices_interior], dtype=float)
        self._mh_u = np.asarray(m_mu_flat[plane.h_u_indices], dtype=float)
        self._mh_v = np.asarray(m_mu_flat[plane.h_v_indices], dtype=float)

        # Reconstruction maps: subspace coordinates -> full port-plane
        # tangential arrays (masked edges stay zero).
        V = exterior.V
        nu_free = int(chain.free_u.sum())
        n_u = int(plane.e_u_indices.size)
        n_v = int(plane.e_v_indices.size)
        self._Wu = np.zeros((n_u, exterior.p))
        self._Wv = np.zeros((n_v, exterior.p))
        self._Wu[chain.free_u, :] = V[:nu_free, :]
        self._Wv[chain.free_v, :] = V[nu_free : chain.n_t, :]

        # Projected coupling to the first full interior period:
        # coup = (V^T W D_p1) x_1.
        self._x1_idx = chain.period(1)
        self._M_coup = np.ascontiguousarray(
            (chain.D_p1.T @ exterior.VtW.T).T,
        )
        self._x1_prev = np.zeros(self._x1_idx.size)

        self._excitation_mode: int | None = None
        self._excitation_waveform = None
        self._src_series: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def subspace_rank(self) -> int:
        return self._exterior.p

    def channel_band(self, mode_idx: int) -> tuple[float, float]:
        """Tracked subspace band ``(f_lo, f_hi)`` [Hz] of a channel.

        The frequency range over which the channel family's subspace
        direction is tabulated; ``set_excitation_band`` spans must lie
        inside it, strictly so at the upper end where the roll-off
        needs room.  A DC-anchored channel reports ``0.0`` as its
        lower bound: the direction is known down to the static limit,
        so a span may reach the first bin and no lower roll-off is
        synthesised.
        """
        if not (0 <= mode_idx < self._n_modes):
            raise ValueError(
                f"mode_idx {mode_idx} out of range [0, {self._n_modes})",
            )
        freqs, _ = self._src_directions[mode_idx]
        return float(freqs[0]), float(freqs[-1])

    # ------------------------------------------------------------------
    # Excitation lifecycle
    # ------------------------------------------------------------------

    def set_excitation(
        self,
        mode_idx: int,
        waveform_fn,
        n_syn: int = 8192,
    ) -> None:
        """Activate the frequency-tracked ghost source on a channel.

        The scalar ``waveform_fn`` must be a finite pulse; it is
        sampled over ``n_syn`` steps and re-synthesised with the
        channel family's tracked subspace direction per frequency,

            s_hat(f) = w_hat(f) * u(f),    u(f) = V^T W phi_f,

        band-limited to the tracked grid range (cosine roll-off over
        one grid spacing outside it).  The incoming kernel then
        launches, per in-band frequency, the projected image of the
        *true* discrete mode — a fixed source direction would excite
        the interface at the profile-drift level instead (class
        docstring).  Out-of-band pulse content is deliberately not
        launched: the subspace does not certify it.

        Parameters
        ----------
        mode_idx : int
            Channel index in ``[0, n_modes)``.
        waveform_fn : Callable[[float], float]
            Pulse amplitude vs time [s]; must have decayed within
            ``n_syn`` solver steps.
        n_syn : int, default 8192
            Synthesis window length in steps (also the maximum
            source duration).

        Raises
        ------
        ValueError
            If the waveform has not decayed at the synthesis-window
            end (the source would be truncated mid-pulse).
        """
        if not (0 <= mode_idx < self._n_modes):
            raise ValueError(
                f"mode_idx {mode_idx} out of range [0, {self._n_modes})",
            )
        w = np.array([float(waveform_fn(n * self._dt)) for n in range(n_syn)])
        w_peak = float(np.abs(w).max())
        if w_peak > 0.0 and float(np.abs(w[-16:]).max()) > 1e-9 * w_peak:
            raise ValueError(
                "excitation waveform has not decayed within the "
                f"synthesis window ({n_syn} steps); pass a larger "
                "n_syn or a shorter pulse",
            )
        W_hat = np.fft.rfft(w)
        self._synthesize_source(mode_idx, W_hat, n_syn)
        self._excitation_mode = mode_idx
        self._excitation_waveform = waveform_fn

    def set_excitation_band(
        self,
        mode_idx: int,
        f_span: tuple[float, float],
        *,
        n_syn: int = 8192,
        skirt: float = 1e-7,
        amplitude: float = 1.0,
    ) -> None:
        """Flat-spectrum broadband pulse over ``f_span`` on a channel.

        The natural excitation of the band port: the erfc-product
        window (a rectangle convolved with Gaussians — flat over
        ``f_span``, Gaussian-class roll-offs sized so the spectrum
        has fallen to ``skirt`` at the subspace band edges).  Its
        time series is sinc-times-Gaussian, i.e. Gaussian-compact; a
        merely C^1 piecewise window decays like ``t^-3`` and is still
        at 1e-4 of peak at the window end (measured), which the
        compactness gate rejects.  The pulse is centred in the
        synthesis window; run the solver for at least ``n_syn`` steps
        plus the ring-down.

        Parameters
        ----------
        mode_idx : int
            Channel index.
        f_span : (float, float)
            Flat measurement span [Hz]; the S-parameter axis should
            stay inside it.  Must leave room for the upper roll-off
            within the port's subspace band
            (:meth:`channel_band`); the lower one exists only where
            that band starts above zero.
        n_syn : int, default 8192
            Synthesis window length in steps.
        skirt : float, default 1e-7
            Relative spectral amplitude at the subspace band edges.
        amplitude : float, default 1.0
            Linear amplitude factor on the synthesised pulse.  The
            caller uses this to inject at full-model power on a port
            cut by symmetry planes; the decomposition itself is
            amplitude-invariant.

        Raises
        ------
        ValueError
            If ``f_span`` leaves no roll-off room inside the subspace
            band, or the synthesised source is not compact.
        """
        if not (0 <= mode_idx < self._n_modes):
            raise ValueError(
                f"mode_idx {mode_idx} out of range [0, {self._n_modes})",
            )
        freqs, _ = self._src_directions[mode_idx]
        W_hat = amplitude * band_source_spectrum(
            f_span,
            (float(freqs[0]), float(freqs[-1])),
            self._dt,
            n_syn,
            skirt=skirt,
        )
        self._synthesize_source(mode_idx, W_hat, n_syn)
        self._excitation_mode = mode_idx
        self._excitation_waveform = None

    def _synthesize_source(
        self,
        mode_idx: int,
        W_hat: np.ndarray,
        n_syn: int,
    ) -> None:
        """Frequency-tracked p-vector source series from a spectrum.

        Multiplies the scalar spectrum by the channel family's
        subspace direction per rfft bin (cubic interpolation over the
        tracking grid — the linear error ~df^2 rides at the port
        plane as launch halo), band-limits to the tabulated range
        with a cosine safety taper (a DC-anchored channel is
        tabulated down to the first bin, so only the upper edge
        tapers), and enforces temporal compactness: a
        source that is still active at the window end would step to
        zero and kick broadband grid modes the band boundary does not
        absorb (measured: near-Nyquist ringing at 1e-5 for thousands
        of steps).
        """
        self._boundary.require_in_kernel()
        freqs, U = self._src_directions[mode_idx]
        f_bins = np.fft.rfftfreq(n_syn, self._dt)
        f_lo, f_hi = float(freqs[0]), float(freqs[-1])
        # Largest gap of the tracking grid — with a DC anchor the grid
        # is no longer exactly uniform, and the taper wants the coarse
        # spacing, not the average.
        df = float(np.max(np.diff(freqs))) if freqs.size > 1 else (f_hi - f_lo)
        S_hat = np.zeros((f_bins.size, self._exterior.p), dtype=complex)
        idx = np.flatnonzero((f_bins >= f_lo - df) & (f_bins <= f_hi + df))
        if idx.size:
            f_in = np.clip(f_bins[idx], f_lo, f_hi)
            if freqs.size >= 4:
                u_in = CubicSpline(freqs, U, axis=0)(f_in)
            else:
                u_in = np.stack(
                    [
                        np.interp(f_in, freqs, U[:, c].real)
                        + 1j * np.interp(f_in, freqs, U[:, c].imag)
                        for c in range(U.shape[1])
                    ],
                    axis=1,
                )
            taper = np.ones(idx.size)
            below = f_bins[idx] < f_lo
            above = f_bins[idx] > f_hi
            taper[below] = 0.5 * (1.0 + np.cos(np.pi * (f_lo - f_bins[idx][below]) / df))
            taper[above] = 0.5 * (1.0 + np.cos(np.pi * (f_bins[idx][above] - f_hi) / df))
            S_hat[idx] = (W_hat[idx] * taper)[:, None] * u_in
        s = np.fft.irfft(S_hat, n=n_syn, axis=0)
        peak = float(np.abs(s).max())
        n_edge = max(n_syn // 32, 16)
        tail = float(np.abs(s[-n_edge:]).max())
        if peak > 0.0 and tail > 1e-6 * peak:
            raise ValueError(
                "synthesised ghost source is not compact: "
                f"|s| = {tail / peak:.1e} of peak at the synthesis-"
                "window end.  The excitation spectrum must fit inside "
                "the port's subspace band "
                f"({f_lo / 1e9:.2f}-{f_hi / 1e9:.2f} GHz) — content "
                "outside it is band-limited away, which spreads the "
                "pulse over the whole window.  Use a narrower pulse, "
                "set_excitation_band, or build the port with a wider "
                "f_band.",
            )
        self._src_series = s

    def clear_excitation(self) -> None:
        """Deactivate the source — the port becomes a passive absorber."""
        self._excitation_mode = None
        self._excitation_waveform = None
        self._src_series = None

    def initialize_state(self, e: np.ndarray) -> None:
        """Capture the interior trace of a non-zero initial condition.

        Exactness of the transparent boundary assumes the exterior
        (and the boundary period itself) is quiescent at ``t = 0``;
        an interior IC that has not reached the port satisfies this.
        """
        self._x1_prev = np.asarray(e, dtype=float)[self._x1_idx].copy()

    def reset_state(self) -> None:
        """Reset the run state for a fresh solver run.

        Zeros the boundary state machine and the interior-trace
        memory and deactivates the source, while keeping everything
        expensive — subspace, projected exterior, contour-QZ kernels.
        The analysis reuses one built port across the per-excitation runs
        of a multi-excitation S-matrix through this.
        """
        self._boundary.reset_state()
        self._x1_prev = np.zeros(self._x1_idx.size)
        self.clear_excitation()

    def state_dict(self) -> dict:
        """Checkpoint the run state (bit-exact resume).

        Exactly what :meth:`reset_state` clears, minus the excitation:
        the projected boundary state machine and the interior trace the
        boundary consumes one step later.  The waveform is re-bound by
        the resuming caller through :meth:`set_excitation`, as on the
        modal port, so only the retardation state is stored here.
        """
        return {
            "x1_prev": self._x1_prev.copy(),
            "boundary": self._boundary.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state written by :meth:`state_dict`."""
        x1 = np.asarray(sd["x1_prev"], dtype=float)
        if x1.size != self._x1_idx.size:
            raise ValueError(
                f"checkpoint holds an interior trace of {x1.size} values "
                f"for port {self.name!r}, which the rebuilt run traces "
                f"with {self._x1_idx.size}",
            )
        self._x1_prev = x1.copy()
        self._boundary.load_state_dict(sd["boundary"])

    # ------------------------------------------------------------------
    # Projections (recorder-facing fixed channels)
    # ------------------------------------------------------------------

    def project_V(self, e: np.ndarray) -> np.ndarray:
        """``V_m`` at the port plane (dual-basis when configured)."""
        return self._project_V_at(
            e,
            self.plane.e_u_indices,
            self.plane.e_v_indices,
            self._me_u_port,
            self._me_v_port,
        )

    def project_V_interior(self, e: np.ndarray) -> np.ndarray:
        """``V_m`` at the one-cell-inside companion plane."""
        return self._project_V_at(
            e,
            self.plane.e_u_indices_interior,
            self.plane.e_v_indices_interior,
            self._me_u_int,
            self._me_v_int,
        )

    def project_I(self, h: np.ndarray) -> np.ndarray:
        """``I_m = <h_m, h>_Mmu`` at the port plane's dual edges."""
        h_u = h[self.plane.h_u_indices]
        h_v = h[self.plane.h_v_indices]
        I = np.empty(self._n_modes)
        for m, dm in enumerate(self.discrete_modes):
            I[m] = float(np.dot(self._mh_u, dm.h_u_profile * h_u)) + float(
                np.dot(self._mh_v, dm.h_v_profile * h_v)
            )
        return I

    def _project_V_at(self, e, e_u_idx, e_v_idx, me_u, me_v):
        e_u = e[e_u_idx]
        e_v = e[e_v_idx]
        V = np.empty(self._n_modes)
        for m, dm in enumerate(self.discrete_modes):
            dual = self._dual_e_profiles[m] if self._dual_e_profiles is not None else None
            p_u, p_v = (
                dual
                if dual is not None
                else (
                    dm.e_u_profile,
                    dm.e_v_profile,
                )
            )
            V[m] = float(np.dot(me_u, p_u * e_u)) + float(np.dot(me_v, p_v * e_v))
        return V

    # ------------------------------------------------------------------
    # FIT-solver hook
    # ------------------------------------------------------------------

    def update_e(self, fields: FieldState, t: float, dt: float) -> None:
        """One projected-boundary leapfrog step; write the port plane.

        Called as the last E-side step (``fields.e_flat`` at
        ``t^{n+1}``).  The boundary update consumes the interior
        trace at its own time level ``t^n`` — captured from the field
        state at the previous call — and the source at chain time
        ``t^n = t - dt`` (the ghost-plane convention of the scalar
        DTBC).  The port plane's tangential E is then overwritten
        with the reconstruction ``V_t xt^{n+1}``.
        """
        del dt
        e = fields.e_flat

        src = None
        if self._src_series is not None:
            n = self._boundary.step_count
            if n < self._src_series.shape[0]:
                src = self._src_series[n]

        coup = self._M_coup @ self._x1_prev
        xt = self._boundary.advance(coup, src)

        e[self.plane.e_u_indices] = self._Wu @ xt
        e[self.plane.e_v_indices] = self._Wv @ xt

        self._x1_prev = e[self._x1_idx].copy()


# ----------------------------------------------------------------------
# Dense a-priori reflection certificate (gate-sized sections)
# ----------------------------------------------------------------------


def _all_branches(D, sig_hat):
    """Finite eigenpairs of the full pencil at real ``sig_hat``."""
    D_m1, D_0, D_p1 = (Dk.toarray() if sp.issparse(Dk) else np.asarray(Dk) for Dk in D)
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


def _scattered_branches(D, sig_hat, w):
    """W-normalised into-domain branch set (the scattered ansatz).

    Waves radiated from the boundary back into the domain propagate
    or decay toward ``+p``: ``|mu| < 1`` strictly, or on-circle with
    ``Im mu < 0`` (the zeta-pencil incident-wave convention).
    """
    vals, vecs = _all_branches(D, sig_hat)
    norms = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(vecs), w, vecs)))
    vecs = vecs / norms[None, :]
    a = np.abs(vals)
    into = (a < 1.0 - _ONC_TOL) | ((np.abs(a - 1.0) <= _ONC_TOL) & (vals.imag < 0.0))
    return vals[into], vecs[:, into]


def galerkin_boundary_symbol(
    chain: PeriodChain,
    exterior: BandExterior,
    z: complex,
) -> np.ndarray:
    """Effective ``e_t`` boundary symbol of the Galerkin exterior.

    Eliminating the projected half-line (small exact solvent of the
    swapped projected pencil) closes the boundary period, giving the
    ghost map seen by the last full interior period:

        e_t(0) = -V_t [ (D~_0 - sig) + D~_m1 Lam_out ]^{-1}
                     (V^T W D_p1)  x_1,

    directly comparable against the exact DtN in
    :func:`band_apriori_reflection`.
    """
    sig = 2.0 - z - 1.0 / z
    lam_out = stable_solvent(
        exterior.Dt_p1,
        exterior.Dt_0,
        exterior.Dt_m1,
        complex(sig),
    )
    p = exterior.p
    G = np.linalg.inv(
        (exterior.Dt_0 - sig * np.eye(p)) + exterior.Dt_m1 @ lam_out,
    )
    if sp.issparse(chain.D_p1):
        coup = np.asarray((chain.D_p1.T @ exterior.VtW.T).T)
    else:
        coup = exterior.VtW @ chain.D_p1
    return -(exterior.V[: chain.n_t, :] @ G @ coup)


def band_apriori_reflection(
    chain: PeriodChain,
    exterior: BandExterior,
    dt: float,
    points: list[tuple[float, complex, np.ndarray]],
) -> list[float]:
    """Exact a-priori modal reflection |Gamma| of the Galerkin boundary.

    The matrix generalisation of the R1 gate formula (WP-R4b spike),
    relabelled to the production orientation (exterior at ``-p``).
    The port-incident wave is the conjugate partner of the family
    eigenpair (travelling toward the port); the scattered field is
    expanded over all into-domain branches; the only equations
    differing from the exact continuation are the ghost rows
    ``x_{0,t} = Lam~_t x_1``:

        sum_j c_j (chi_{j,t} - mu_j Lam~_t chi_j)
            = zeta_inc Lam~_t phi_inc - phi_inc,t,

    an ``N_t x N_t`` solve per point; ``|Gamma|`` is the coefficient
    of the incident mode's mirror branch (the family eigenpair
    itself).  Requires the *dense* branch set (O((2N)^3) per point) —
    intended for gate-sized cross-sections.

    Parameters
    ----------
    chain : PeriodChain
        Boundary-paired chain.
    exterior : BandExterior
        Galerkin exterior under test.
    dt : float
        Solver time step [s].
    points : list of (f, zeta, phi)
        Family grid points (zeta-pencil incident-wave gauge,
        ``Im zeta <= 0``), e.g. from :func:`track_band_families`.

    Returns
    -------
    list[float]
        ``|Gamma|`` per point.
    """
    D = (chain.D_m1, chain.D_0, chain.D_p1)
    w = np.asarray(chain.w_period, dtype=float)
    n_t = chain.n_t
    out: list[float] = []
    for f, zeta_fam, phi_fam in points:
        w_dt = 2.0 * math.pi * f * dt
        sig_hat = 2.0 - 2.0 * math.cos(w_dt)
        z = (1.0 + _RHO_OFF) * np.exp(1j * w_dt)
        lam_t = galerkin_boundary_symbol(chain, exterior, complex(z))
        mu, chi = _scattered_branches(D, sig_hat, w)
        if mu.size != n_t:
            raise RuntimeError(
                f"scattered branch set has {mu.size} members, expected N_t = {n_t}",
            )
        # Port-incident wave = conjugate partner (travels toward -p).
        zeta_inc = np.conj(zeta_fam)
        phi_inc = np.conj(phi_fam)
        lhs = chi[:n_t, :] - mu[None, :] * (lam_t @ chi)
        rhs = zeta_inc * (lam_t @ phi_inc) - phi_inc[:n_t]
        c, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        j0 = int(np.argmin(np.abs(mu - zeta_fam)))
        if abs(mu[j0] - zeta_fam) > 1e-6:
            raise RuntimeError(
                "mirror branch of the incident mode not found in the scattered set",
            )
        out.append(float(abs(c[j0])))
    return out
