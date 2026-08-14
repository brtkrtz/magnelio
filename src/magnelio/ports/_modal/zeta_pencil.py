"""Per-frequency true discrete modes of an inhomogeneous line (WP-R4a).

On a z-uniform feed section the 3D FIT update in second-order form is
exactly a block-tridiagonal vector chain over the period trace
``x_p = (e_t(p), e_z(p+1/2))`` (WP-R4 method spike, session 87), and
the true discrete modes at one frequency are the eigenpairs of the
quadratic pencil

    [ zeta^2 D_p1 + zeta (D_0 - sig_hat) + D_m1 ] phi = 0,
    sig_hat = 2 - 2 cos(w dt),  D = dt^2 * (period blocks of A),

with ``A = M_eps^{-1} C^T M_mu^{-1} C``.  This module provides the
production machinery of the CW-per-frequency path (Option B,
developer decision 2026-07-09):

* :func:`build_period_blocks` — sparse period blocks extracted from
  the *actual* production matrices at the port (any ``BoxFace``),
  with a translation-invariance certificate of the port-adjacent
  uniform section (the exterior the transparent condition emulates).
* :func:`solve_zeta_modes` — sparse shift-invert eigensolve of the
  linearised pencil around a target ``zeta`` (frequency continuation
  supplies the target; ARPACK on the factored standard operator
  ``(A_lin - sigma B_lin)^{-1} B_lin``, since ``B_lin`` is singular).
* :func:`chain_fit` — the closed-form frequency-local scalar-chain
  parameters ``(r_eff, q_eff)``: ``r^2`` by Hellmann-Feynman
  derivative matching with the palindromic left eigenvector
  ``psi = W conj(phi)``, ``q^2`` by the exact on-circle identity
  ``q^2 = 4 [sin^2(w dt/2) - r^2 sin^2(theta/2)]``.  By construction
  ``lambda(e^{i w dt}; r, q) = zeta`` to machine precision, so the
  existing scalar :class:`~magnelio.ports._modal.dtbc.DTBCTermination`
  terminates the true mode *at the drive frequency* exactly — the
  frequency-local exact termination (pre-check gate
  ``validation/qtem_cw_precheck_spike.py``, session 88).
* :func:`cw_wave_phasors` — the exact (V, I) response of the unit
  incident / outgoing discrete wave through the *stored* projection
  profiles and the actual 3D curl (synthetic Bloch field on the two
  port-adjacent planes; ``H = -dt (C E) / (M_mu (z^{1/2} -
  z^{-1/2}))`` from the leapfrog H update).  Consumed by
  :func:`cw_decompose` — the per-frequency a/b decomposition of the
  CW lock-in measurement (the R2/R3 de-stagger and discrete wave
  impedance are *contained* in these phasors; no closed-form scalar
  impedance is needed).

Gauge: on the unit circle the pencil is real with real ``sig_hat``,
so propagating eigenpairs come in conjugate pairs ``(zeta, phi)`` /
``(1/zeta, conj(phi))`` — incident and reflected wave.  The
z-reflection symmetry of the trace fixes a gauge with ``phi_t`` real
(residual ~1e-13, checked); the stored port profile is that real
vector, in the production ``DiscreteMode`` form.

Plane indexing is *inward*: ``p = 0`` is the port plane, ``p``
increases into the domain, ``e_z(p+1/2)`` sits between planes ``p``
and ``p+1``.  A wave with ``Im zeta < 0`` travels toward ``+p``
(into the domain) — the incident wave; its conjugate partner is the
reflected wave.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from magnelio.constants import C0
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.port_plane import PortPlane

# A propagating branch on the (slightly off-)circle solve: |zeta|
# within this tolerance of 1.  Evanescent branches of interest to a
# CW port do not exist (they carry no power at the drive frequency).
_PROP_TOL = 1e-6

# Translation-invariance certificate tolerance for the port-adjacent
# period blocks (relative).  Roundoff sits many orders below; a
# genuinely non-uniform feed section violates it at the material-
# contrast level.
_INVARIANCE_RTOL = 1e-9


@dataclass(frozen=True)
class PeriodChain:
    """Sparse period blocks + index bookkeeping at one port.

    Attributes
    ----------
    D_m1, D_0, D_p1 : scipy.sparse.csc_matrix
        Dimensionless period blocks ``dt^2 * B`` on the free DOFs of
        one period, inward plane indexing.
    w_period : np.ndarray
        ``M_eps`` diagonal over the period trace (the W metric).
    n_t : int
        Number of free tangential DOFs (``e_t`` block size).
    free_u, free_v : np.ndarray of bool
        Free-edge masks over the plane's ``e_u`` / ``e_v`` arrays —
        map the pencil's ``e_t`` block back onto full plane profiles.
    et_indices : np.ndarray
        Flat E indices of the free tangential edges at the port plane
        (periods shift by ``et_step``).
    ez_indices : np.ndarray
        Flat E indices of the free normal-E edges of period 0.
    et_step, ez_step : np.ndarray / int
        Flat-index offsets for one period inward.  ``et_step`` is a
        scalar on z-normal faces (both tangential families share the
        normal stride) and a per-edge array on x-/y-normal faces,
        where ``e_u`` and ``e_v`` are different E components with
        different flat-layout strides.
    dt : float
        Solver time step the blocks were scaled with.
    pairing : str
        Trace convention.  ``"inward"`` (default) pairs
        ``(e_t(p), e_z(p+1/2))`` — the WP-R4/R4a convention, where
        the coupling block toward the *domain interior* (``D_p1``)
        has ``e_t`` columns only.  ``"boundary"`` pairs
        ``(e_t(p), e_z(p-1/2))`` so the coupling block toward the
        *exterior* (``D_m1``) has ``e_t`` columns only: interior
        periods touch the port-side boundary period exclusively
        through the port-plane ``e_t``, and the boundary period's
        ``e_z`` half-plane lies outside the mesh (virtual) — the
        trace layout of the WP-R4b Galerkin boundary operator, whose
        projected period-0 state owns that virtual half-plane.
    """

    D_m1: sp.csc_matrix
    D_0: sp.csc_matrix
    D_p1: sp.csc_matrix
    w_period: np.ndarray
    n_t: int
    free_u: np.ndarray
    free_v: np.ndarray
    et_indices: np.ndarray
    ez_indices: np.ndarray
    et_step: int | np.ndarray
    ez_step: int
    dt: float
    pairing: str = "inward"

    def period(self, p: int) -> np.ndarray:
        """Flat E indices of period ``p`` (inward from the port).

        With ``pairing="boundary"`` the ``e_z`` indices of period 0
        are virtual (outside the mesh); callers must only gather
        periods ``p >= 1`` from a field vector in that pairing.
        """
        ez_p = p if self.pairing == "inward" else p - 1
        return np.concatenate(
            [
                self.et_indices + p * self.et_step,
                self.ez_indices + ez_p * self.ez_step,
            ]
        )


@dataclass(frozen=True)
class CWChannel:
    """One frequency-local port channel at the CW frequency.

    ``zeta`` is the incident-wave eigenvalue (``Im zeta <= 0``,
    travelling into the domain); the reflected wave is the conjugate
    pair.  ``r``/``q`` are the certified frequency-local chain
    parameters consumed by ``DTBCTermination``; the ``v_*``/``i_*``
    phasors are the exact projection responses of the unit incident
    (``_in``) and reflected (``_out``) discrete wave through the
    stored profiles, in the recorder's sampling convention (V at the
    port plane / integer steps, I at the interior half-plane rotated
    by ``e^{+j w dt/2}``).
    """

    zeta: complex
    r: float
    q: float
    eps_eff_hat: float
    phi_trace: np.ndarray = field(repr=False)
    v_in: complex = 0.0
    i_in: complex = 0.0
    v_out: complex = 0.0
    i_out: complex = 0.0


@dataclass(frozen=True)
class CWPortData:
    """Per-frequency mode data of a CW true-mode port.

    Attached to the operator by ``build_cw_true_mode_port`` as
    ``op.cw_data``; consumed by the CW lock-in postprocessing
    (:func:`cw_lockin_phasors` + :func:`cw_decompose`).  ``channels``
    aligns with the operator's mode indices.
    """

    f_cw: float
    w_dt: float
    channels: tuple[CWChannel, ...]
    solve_seconds: float


def _flat_e_layout(mesh: Mesh):
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    shapes = {
        0: (Nx, Ny + 1, Nz + 1),
        1: (Nx + 1, Ny, Nz + 1),
        2: (Nx + 1, Ny + 1, Nz),
    }
    offsets = {0: 0, 1: n_Ex, 2: n_Ex + n_Ey}
    return shapes, offsets, n_Ex + n_Ey + n_Ez


def _flat_e_pec(mesh: Mesh) -> np.ndarray:
    shapes, _, n_e = _flat_e_layout(mesh)
    sizes = [int(np.prod(shapes[a])) for a in (0, 1, 2)]
    return np.concatenate(
        [
            mesh.pec_mask_edges[0, : sizes[0]],
            mesh.pec_mask_edges[1, : sizes[1]],
            mesh.pec_mask_edges[2, : sizes[2]],
        ]
    )


def _normal_e_indices(plane: PortPlane, mesh: Mesh) -> tuple[np.ndarray, int]:
    """Flat indices of period-0 normal-E edges + the per-period step.

    The normal-E edge of period 0 spans the cell between the port
    plane and the first interior plane; transversally the edges sit
    on the plane's (u, v) node window.
    """
    shapes, offsets, _ = _flat_e_layout(mesh)
    n_axis = plane.face.normal_axis
    u_axis = plane.face.u_axis
    v_axis = plane.face.v_axis
    shape = shapes[n_axis]
    n_cells = shape[n_axis]

    cell0 = 0 if not plane.face.is_max else n_cells - 1
    step_cells = 1 if not plane.face.is_max else -1

    iu, iv = np.meshgrid(
        np.arange(plane.u_node_window[0], plane.u_node_window[1] + 1),
        np.arange(plane.v_node_window[0], plane.v_node_window[1] + 1),
        indexing="ij",
    )
    ijk: list = [None, None, None]
    ijk[n_axis] = np.full_like(iu, cell0)
    ijk[u_axis] = iu
    ijk[v_axis] = iv
    flat = (
        (offsets[n_axis] + ijk[0] * shape[1] * shape[2] + ijk[1] * shape[2] + ijk[2])
        .ravel()
        .astype(np.int64)
    )

    stride = (shape[1] * shape[2], shape[2], 1)[n_axis]
    return flat, int(step_cells * stride)


def _uniform_step(port_idx: np.ndarray, interior_idx: np.ndarray) -> int:
    """One-plane-inward flat-index offset of an edge family."""
    d = np.asarray(interior_idx, dtype=np.int64) - np.asarray(port_idx, dtype=np.int64)
    if d.size == 0:
        return 0
    if not np.all(d == d[0]):
        raise RuntimeError(
            "port-plane edge family has a non-constant normal stride "
            "— flat-index layout assumption violated"
        )
    return int(d[0])


def build_period_blocks(
    plane: PortPlane,
    mesh: Mesh,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    c_3d: sp.spmatrix,
    dt: float,
    pairing: str = "inward",
) -> PeriodChain:
    """Extract the dimensionless period blocks at a port.

    The blocks are index slices of ``A = M_eps^{-1} C^T M_mu^{-1} C``
    built from the *actual* production matrices — DD-053 conformal
    coupling, PEC masks and all — restricted to the free DOFs of the
    port-adjacent period, scaled by ``dt^2``.  Only the rows of the
    extraction periods are formed (no global matrix product).

    ``pairing`` selects the trace convention (see
    :class:`PeriodChain`): ``"inward"`` extracts at periods 1/2,
    ``"boundary"`` at periods 2/3 (its period-1 columns already reach
    the virtual ``e_z`` of period 0 otherwise); both stay within the
    four certified cells.

    Certificates enforced here (the CW analogue of the DD-054 pair
    gate):

    * at least four equidistant cells along the port normal
      (extraction reaches plane 4);
    * the PEC mask is z-uniform over those planes;
    * the blocks extracted one period deeper agree to
      ``_INVARIANCE_RTOL`` — the feed section is translation
      invariant, so the pencil's uniform continuation IS the
      exterior the port emulates.

    Raises
    ------
    ValueError
        If the port-adjacent section violates a certificate.
    """
    if pairing not in ("inward", "boundary"):
        raise ValueError(f"pairing must be 'inward' or 'boundary', got {pairing!r}")
    grid = mesh.grid
    deltas = (grid.dx, grid.dy, grid.dz)[plane.face.normal_axis]
    if deltas.size < 4:
        raise ValueError(
            f"CW true-mode port requires at least 4 cells along the port normal; got {deltas.size}"
        )
    port_cells = deltas[:4] if not plane.face.is_max else deltas[-4:]
    cmin, cmax = float(np.min(port_cells)), float(np.max(port_cells))
    if (cmax - cmin) / cmin > 1e-6:
        raise ValueError(
            "CW true-mode port requires 4 equidistant cells adjacent "
            f"to the port plane (got widths {port_cells.tolist()}); "
            "the frequency-local termination emulates the uniform "
            "continuation of exactly this section"
        )

    pec_flat = _flat_e_pec(mesh)
    et_u_step = _uniform_step(plane.e_u_indices, plane.e_u_indices_interior)
    et_v_step = _uniform_step(plane.e_v_indices, plane.e_v_indices_interior)
    ez_idx, ez_step = _normal_e_indices(plane, mesh)

    free_u = ~pec_flat[plane.e_u_indices]
    free_v = ~pec_flat[plane.e_v_indices]
    free_z = ~pec_flat[ez_idx]
    # z-uniformity of the masks over the certificate range.
    for p in range(1, 5):
        if not (
            np.array_equal(free_u, ~pec_flat[plane.e_u_indices + p * et_u_step])
            and np.array_equal(free_v, ~pec_flat[plane.e_v_indices + p * et_v_step])
        ):
            raise ValueError(
                "PEC mask is not z-uniform over the port-adjacent "
                f"section (plane {p}) — the feed line must be "
                "straight and uniform at a CW true-mode port"
            )
    for p in range(1, 4):
        if not np.array_equal(free_z, ~pec_flat[ez_idx + p * ez_step]):
            raise ValueError(
                "PEC mask (normal-E edges) is not z-uniform over the port-adjacent section"
            )

    # Combined tangential index family.  On z-normal faces both
    # families share the normal stride and a scalar step suffices; on
    # x-/y-normal faces e_u and e_v are different E components with
    # different flat-layout strides, so the step is carried per edge
    # (``et_indices + p * et_step`` stays elementwise; KB-009).
    et_idx = np.concatenate(
        [
            plane.e_u_indices[free_u],
            plane.e_v_indices[free_v],
        ]
    ).astype(np.int64)
    if et_idx.size == 0:
        raise ValueError("port plane has no free tangential edges; cannot build a period chain")
    if et_u_step == et_v_step:
        et_step: int | np.ndarray = int(et_u_step)
    else:
        et_step = np.concatenate(
            [
                np.full(int(free_u.sum()), et_u_step, dtype=np.int64),
                np.full(int(free_v.sum()), et_v_step, dtype=np.int64),
            ]
        )
    ez_free = ez_idx[free_z]

    chain_proto = dict(
        w_period=None,
        n_t=int(et_idx.size),
        free_u=free_u,
        free_v=free_v,
        et_indices=et_idx,
        ez_indices=ez_free,
        et_step=et_step,
        ez_step=ez_step,
        dt=float(dt),
        pairing=pairing,
    )

    def period(p: int) -> np.ndarray:
        ez_p = p if pairing == "inward" else p - 1
        return np.concatenate([et_idx + p * et_step, ez_free + ez_p * ez_step])

    # Exact 1/M_μ with 0 on frozen (enlarged-cell-donated, WP-R5)
    # faces — the same removal the 3D update applies.
    m_mu_arr = np.asarray(m_mu, dtype=float)
    inv_mh = sp.diags(
        np.where(
            m_mu_arr > 0,
            1.0 / np.where(m_mu_arr > 0, m_mu_arr, 1.0),
            0.0,
        )
    )
    c_csc = c_3d.tocsc()

    def blocks_at(p0: int):
        rows = period(p0)
        # Row slice of A without forming A: rows of C^T = cols of C.
        ct_rows = c_csc[:, rows].T.tocsr()
        a_rows = (
            sp.diags(1.0 / np.asarray(m_eps, dtype=float)[rows]) @ ct_rows @ inv_mh @ c_3d
        ).tocsc()
        return tuple((dt * dt) * a_rows[:, period(p0 + d)] for d in (-1, 0, 1))

    p_lo = 1 if pairing == "inward" else 2
    D1 = blocks_at(p_lo)
    D2 = blocks_at(p_lo + 1)
    ref = max(abs(D1[1]).max(), 1e-300)
    dev = max(abs(D2[i] - D1[i]).max() for i in range(3)) / ref
    if dev > _INVARIANCE_RTOL:
        raise ValueError(
            "port-adjacent section is not translation invariant "
            f"(period-block deviation {dev:.2e} > "
            f"{_INVARIANCE_RTOL:.0e}) — the CW true-mode port needs "
            "a uniform straight feed section"
        )

    chain_proto["w_period"] = np.asarray(m_eps, dtype=float)[period(p_lo)]
    return PeriodChain(D_m1=D1[0].tocsc(), D_0=D1[1].tocsc(), D_p1=D1[2].tocsc(), **chain_proto)


def solve_zeta_modes(
    chain: PeriodChain,
    w_dt: float,
    zeta_targets: list[complex],
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of the on-circle pencil near the given targets.

    One sparse LU factorisation + ARPACK run per target; results are
    merged and deduplicated.  Returns ``(zetas, phis)`` with ``phis``
    columns W-normalised over the full period trace.
    """
    D_m1, D_0, D_p1 = chain.D_m1, chain.D_0, chain.D_p1
    n = D_0.shape[0]
    sig_hat = 2.0 - 2.0 * math.cos(w_dt)
    eye = sp.identity(n, format="csc")
    A_lin = sp.bmat([[None, eye], [-D_m1, -(D_0 - sig_hat * eye)]], format="csc")
    B_lin = sp.block_diag([eye, D_p1], format="csc")

    zs: list[complex] = []
    vs: list[np.ndarray] = []
    for target in zeta_targets:
        lu = spla.splu((A_lin - complex(target) * B_lin).astype(complex).tocsc())
        op = spla.LinearOperator(
            (2 * n, 2 * n), matvec=lambda v: lu.solve(B_lin @ v), dtype=complex
        )
        k_eff = min(k, 2 * n - 2)
        mu, vecs = spla.eigs(op, k=k_eff, which="LM")
        zeta = complex(target) + 1.0 / mu
        for j in range(zeta.size):
            if any(abs(zeta[j] - z0) < 1e-9 for z0 in zs):
                continue
            zs.append(complex(zeta[j]))
            vs.append(vecs[:n, j])
    zetas = np.array(zs)
    phis = np.stack(vs, axis=1) if vs else np.empty((n, 0), complex)
    norms = np.sqrt(np.abs(np.einsum("in,i,in->n", np.conj(phis), chain.w_period, phis)))
    phis = phis / norms[None, :]
    return zetas, phis


def find_propagating_modes(
    chain: PeriodChain,
    w_dt: float,
    theta_hint: float,
    k: int = 8,
    arc_fractions: tuple = (1.0, 0.55, 0.3, 0.15, 0.06),
) -> tuple[np.ndarray, np.ndarray]:
    """All propagating modes at ``w_dt`` on the arc ``(0, theta_hint]``.

    Every mode propagating at a given frequency has a phase advance
    below the fundamental's (the fundamental carries the largest
    ``beta``), so shift-invert targets spread along the unit-circle
    arc up to ``theta_hint`` (the fundamental estimate plus margin)
    cover the whole family — a single target near the fundamental
    misses higher modes whose ``zeta`` hides among the evanescent
    cluster near ``+1``.  Conjugate partners are deduplicated and the
    result is gauge-flipped to ``Im zeta <= 0`` (incident-wave
    convention), ordered by descending phase advance.
    """
    theta_hint = min(theta_hint, math.pi * 0.9)
    targets = [np.exp(-1j * max(theta_hint * f, 1e-3)) for f in arc_fractions]
    zs, ps = solve_zeta_modes(chain, w_dt, targets, k=k)
    prop = np.abs(np.abs(zs) - 1.0) <= _PROP_TOL
    zs, ps = zs[prop], ps[:, prop]
    keep: list[int] = []
    for j in range(zs.size):
        if any(abs(zs[j] - zs[i]) < 1e-9 or abs(zs[j] - np.conj(zs[i])) < 1e-9 for i in keep):
            continue
        keep.append(j)
    keep.sort(key=lambda j: -abs(np.angle(zs[j])))
    zetas = zs[keep]
    phis = ps[:, keep]
    flip = zetas.imag > 0.0
    zetas = np.where(flip, np.conj(zetas), zetas)
    phis[:, flip] = np.conj(phis[:, flip])
    return zetas, phis


def normalize_gauge(phi: np.ndarray, n_t: int) -> np.ndarray:
    """Phase-fix: largest-|.| tangential component real positive."""
    ref = phi[np.argmax(np.abs(phi[:n_t]))]
    return phi * (np.conj(ref) / abs(ref))


def chain_fit(
    zeta: complex,
    phi: np.ndarray,
    chain: PeriodChain,
    w_dt: float,
) -> tuple[float, float]:
    """Frequency-local scalar-chain parameters ``(r, q)``.

    Closed-form from one eigenpair (module docstring); the fit
    satisfies ``lambda(e^{i w dt}; r, q) = zeta`` to machine
    precision.  Raises when the certified assumptions are violated.

    Raises
    ------
    ValueError
        ``q_eff^2 < 0`` (anomalous dispersion — the fitted chain
        would be unstable; not certified) or ``r_eff`` outside
        ``(0, 1]``.
    """
    sig_hat = 2.0 - 2.0 * math.cos(w_dt)
    w = chain.w_period
    qp = (2.0 * zeta) * (chain.D_p1 @ phi) + chain.D_0 @ phi - sig_hat * phi
    num = np.vdot(phi, w * qp)
    den = (1.0 / zeta - zeta) * np.vdot(phi, w * phi)
    r2 = float((num / den).real)
    theta = abs(np.angle(zeta))
    q2 = 4.0 * (math.sin(w_dt / 2.0) ** 2 - r2 * math.sin(theta / 2.0) ** 2)
    # q2 is a difference of two near-equal O(sin^2) terms; on uniform
    # lines (exact TEM symbol, q = 0) cancellation noise can land a few
    # 1e-16 below zero.  Clamp within the cancellation scale — the
    # anomalous-dispersion rejection below stays for real negatives.
    q2_scale = 4.0 * math.sin(w_dt / 2.0) ** 2
    if -1e-12 * q2_scale <= q2 < 0.0:
        q2 = 0.0
    if 1.0 < r2 <= 1.0 + 1e-12:
        r2 = 1.0
    if q2 < 0.0:
        raise ValueError(
            f"frequency-local fit yields q_eff^2 = {q2:.3e} < 0 "
            "(anomalous dispersion at this frequency); the "
            "frequency-local KG termination is not certified here"
        )
    if not (0.0 < r2 <= 1.0):
        raise ValueError(f"frequency-local fit yields r_eff^2 = {r2:.3e} outside (0, 1]")
    return math.sqrt(r2), math.sqrt(q2)


def make_channel(
    zeta: complex,
    phi: np.ndarray,
    chain: PeriodChain,
    w_dt: float,
    dz: float,
) -> CWChannel:
    """Package one eigenpair as a gauge-fixed incident-wave channel."""
    # Incident wave travels inward: Im zeta <= 0 in the inward
    # indexing.  Flip to the conjugate partner if needed.
    if zeta.imag > 0.0:
        zeta = np.conj(zeta)
        phi = np.conj(phi)
    phi = normalize_gauge(phi, chain.n_t)
    r, q = chain_fit(zeta, phi, chain, w_dt)
    theta = abs(np.angle(zeta))
    s_ratio = (math.sin(theta / 2.0) / (dz / 2.0)) / (math.sin(w_dt / 2.0) / (chain.dt / 2.0))
    eps_eff_hat = (C0 * s_ratio) ** 2
    return CWChannel(zeta=complex(zeta), r=r, q=q, eps_eff_hat=float(eps_eff_hat), phi_trace=phi)


def profile_reality(phi: np.ndarray, n_t: int) -> float:
    """``||Im phi_t|| / ||phi_t||`` after gauge fixing."""
    p = phi[:n_t]
    return float(np.linalg.norm(p.imag) / np.linalg.norm(p))


def cw_wave_phasors(
    channel: CWChannel,
    chain: PeriodChain,
    plane: PortPlane,
    m_eps: np.ndarray,
    m_mu: np.ndarray,
    c_3d: sp.spmatrix,
    w_dt: float,
    h_u_prof: np.ndarray,
    h_v_prof: np.ndarray,
    proj_u: np.ndarray,
    proj_v: np.ndarray,
) -> CWChannel:
    """Exact (V, I) phasors of the unit incident/reflected wave.

    Synthesises the Bloch field of the channel on the two
    port-adjacent planes (``e_t(0)``, ``e_t(1) = zeta e_t(0)``,
    ``e_z(1/2)``), applies the *actual* 3D curl to obtain the
    travelling wave's H at the port's dual faces via the leapfrog
    relation ``H = -dt (C E) / (M_mu (z^{1/2} - z^{-1/2}))``, and
    projects with the stored profiles exactly as the recorder does
    (V with the — possibly dual-basis — projection weights, I with
    the stored H profile and ``M_mu`` weights).  The reflected wave
    is the conjugate partner.

    Returns a copy of ``channel`` with the four phasors filled.
    """
    if chain.pairing != "inward":
        raise ValueError(
            "cw_wave_phasors requires an inward-paired chain (the "
            "synthetic Bloch field is laid out on periods 0/1)"
        )
    n_t = chain.n_t
    n_u = int(plane.e_u_indices.size)
    phi = channel.phi_trace
    zeta = channel.zeta
    z_half = np.exp(0.5j * w_dt)

    e_syn = np.zeros(c_3d.shape[1], dtype=complex)
    per0 = chain.period(0)
    per1 = chain.period(1)
    e_syn[per0[:n_t]] = phi[:n_t]
    e_syn[per0[n_t:]] = phi[n_t:]
    e_syn[per1[:n_t]] = zeta * phi[:n_t]

    h_syn = -(chain.dt * (c_3d @ e_syn) / np.asarray(m_mu, dtype=float)) / (z_half - 1.0 / z_half)

    # Map the trace's free-DOF e_t back onto the full plane arrays.
    e_u_full = np.zeros(n_u, dtype=complex)
    e_v_full = np.zeros(plane.e_v_indices.size, dtype=complex)
    nu_free = int(chain.free_u.sum())
    e_u_full[chain.free_u] = phi[:nu_free]
    e_v_full[chain.free_v] = phi[nu_free:n_t]

    me_u = np.asarray(m_eps, dtype=float)[plane.e_u_indices]
    me_v = np.asarray(m_eps, dtype=float)[plane.e_v_indices]
    mh_u = np.asarray(m_mu, dtype=float)[plane.h_u_indices]
    mh_v = np.asarray(m_mu, dtype=float)[plane.h_v_indices]

    v_in = complex(np.dot(me_u * proj_u, e_u_full) + np.dot(me_v * proj_v, e_v_full))
    h_u_wave = h_syn[plane.h_u_indices]
    h_v_wave = h_syn[plane.h_v_indices]
    i_in = complex(np.dot(mh_u * h_u_prof, h_u_wave) + np.dot(mh_v * h_v_prof, h_v_wave))

    # Reflected wave = the conjugate eigenpair.  Its E projection is
    # the plain conjugate, but the leapfrog H phasor carries the
    # purely imaginary denominator ``z^{1/2} - z^{-1/2} = 2i
    # sin(w dt/2)``, which flips sign under conjugation —
    # ``i_out = -conj(i_in)``, the discrete form of the reversed-wave
    # current sign of transmission-line theory.
    return CWChannel(
        zeta=channel.zeta,
        r=channel.r,
        q=channel.q,
        eps_eff_hat=channel.eps_eff_hat,
        phi_trace=channel.phi_trace,
        v_in=v_in,
        i_in=i_in,
        v_out=np.conj(v_in),
        i_out=-np.conj(i_in),
    )


def cw_lockin_phasors(
    v_values: np.ndarray,
    i_values: np.ndarray,
    w_dt: float,
    n_win: int,
) -> tuple[complex, complex, float]:
    """Two-quadrature lock-in of recorded V/I over the last ``n_win``.

    Returns ``(V, I, res)`` with I rotated by ``e^{+j w dt/2}`` (the
    production Yee half-step convention) and ``res`` the relative fit
    residual of the V channel — the steady-state quality gauge.
    """
    n_steps = v_values.size
    n_grid = np.arange(n_steps - n_win, n_steps)
    basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
    cv, *_ = np.linalg.lstsq(basis, v_values[n_grid], rcond=None)
    ci, *_ = np.linalg.lstsq(basis, i_values[n_grid], rcond=None)
    res = float(
        np.linalg.norm(v_values[n_grid] - basis @ cv)
        / max(np.linalg.norm(v_values[n_grid]), 1e-300)
    )
    V = complex(cv[0] - 1j * cv[1])
    I = complex(ci[0] - 1j * ci[1]) * np.exp(0.5j * w_dt)
    return V, I, res


def cw_decompose(
    V: complex,
    I: complex,
    channel: CWChannel,
) -> tuple[complex, complex]:
    """Solve the exact two-wave system for ``(a, b)`` at one port.

    ``V = a v_in + b v_out``, ``I = a i_in + b i_out`` with the
    channel's exact phasors; ``b/a`` is the port reflection of the
    true discrete mode at the drive frequency.
    """
    m = np.array([[channel.v_in, channel.v_out], [channel.i_in, channel.i_out]], dtype=complex)
    ab = np.linalg.solve(m, np.array([V, I], dtype=complex))
    return complex(ab[0]), complex(ab[1])
