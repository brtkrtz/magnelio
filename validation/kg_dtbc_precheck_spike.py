"""WP-R3 pre-check spike — TE/TM Klein-Gordon separation and the
discrete measurement chain (reflection-free plan WP-R3 gate; plan
retired to git history, see DD-055).

Two questions must be answered offline, before any production code,
under the sessions-63-65 ground rule (analytic path to |S11| < -100 dB
or no code):

1. **Separation** — is the longitudinal dynamics of a production
   TE/TM port mode the *exact* 1D leapfrog Klein-Gordon chain?  The
   2D TE eigenproblem is the index-sliced restriction of the 3D FIT
   operators (``build_2d_curl_curl``), so ``Mode.omega_c`` should be
   the exact discrete cut-off (``q = omega_c * dt``); the TM path
   runs a *separately constructed* node-Laplace whose eigenvalue must
   be shown to equal the effective mass of the 3D (e_t, e_z, h_t)
   elimination.  Checked by running the raw 3D leapfrog on a modal
   initial condition and measuring the KG-chain residual of the
   projected amplitude to machine precision — on WR-90 (TE10, TM11)
   and on the conformal round waveguide (TE11, TM01).

   TM subtlety: eliminating (w = e_z amplitude, g = h_t amplitude)
   leaves the discrete invariant
   ``(w_{k+1/2} - w_{k-1/2}) - (wg/ug1) u_k = const``; a pure-e_t
   initial condition therefore obeys the *modified* chain
   ``d2t u = r^2 d2z u - q^2 (u - u0)`` (the static gradient part
   does not propagate).  A quiescent start — the production case,
   excitation entering through the port — has ``u0 = 0`` and the
   pure KG chain.  The spike verifies the modified identity, which
   certifies r^2 and q^2 for any start.

2. **Measurement chain** — the a/b decomposition of
   ``compute_s_parameters`` uses the *continuum* wave impedance
   ``z_modal(omega)``.  For TEM the calibrated V/I of the discrete
   wave is frequency-flat, so one-point calibration is exact at all
   frequencies (DD-054); for a Klein-Gordon mode it is *not*: the
   discrete travelling wave obeys (derived from the chain symbol,
   verified here)

       Z_TE(omega) = K0 * s / sqrt(s^2 - (q/2)^2),
       Z_TM(omega) = K0 * sqrt(s^2 - (q/2)^2) / s,
       s = sin(omega dt / 2),
       K0 = r * nV / (dt * nI * |gu0|),

   the continuum relations under omega -> (2/dt) sin(omega dt/2),
   beta -> (2/dz) sin(beta_hat dz/2).  ``nV`` / ``nI`` are the
   M_eps / M_mu norms of the discrete profiles and ``gu0`` the
   single scalar coupling of the reduced chain — all *static*
   quantities of the port build (no calibration anchor, no fit).
   The gap of the continuum Z(omega) against this is
   O((omega dt)^2, (beta dz)^2) — a -40 to -60 dB measured-|S11|
   floor on lambda/20 meshes that no absorber can remove.

Method for (2): the scalar couplings of the reduced 1D system
(u = e_t amplitude; g = h_t; p = h_z for TE / w = e_z for TM) are
measured by applying the sparse 3D operators to single-plane modal
profiles (rank-1 application; rank-1-ness remainders reported as
certificates that the modal subspace is invariant).  The frequency
response is solved from the symbol algebra, cross-checked against
the closed forms above, and validated end-to-end by CW lock-in runs
of the coupled chain through a bit-faithful replica of the
production sampling (recorder staggering, ``e^{+j omega dt/2}``
rotation, two-plane de-stagger with ``destagger_theta``): for a
pure outgoing wave the measured |b/a| must reach the float-noise /
lock-in floor with the discrete Z and stay at the continuum-gap
level with ``z_modal``.

Results (session 86) — gate PASSED, in two stages:

* With the *original* lumped node-Laplace TM solve
  (``build_2d_node_laplace``): TE separation exact everywhere
  (chain residual 6e-16 uniform WR-90 TE10, 7e-16 conformal round
  TE11; q = Mode.omega_c*dt to 1e-15; pair spread ~1e-16 incl.
  conformal — DD-053 answers the transversal side, TE needs no
  further analogue).  TM however was *inconsistent with the 3D
  metric*: q^2 off by 9e-10 (uniform staircase) to 1.7e-5
  (conformal), e_t profile leakage 5.7e-5, rank-1 closure broken at
  4.7e-3, lock-in floor capped at -77 dB — the anticipated "DD-053
  analogue for TM" question, answered YES.
* With the exact restriction ``build_2d_tm_curl_curl`` (rows =
  transversal H faces at the port-adjacent half-plane, columns =
  normal-E edges, same 3D matrices — implemented as the production
  fix): TM matches TE — chain residuals 1e-15, q^2 to 4e-15,
  rank-1 closure 1e-13, conformal TM01 lock-in -77 -> -132 dB.

Measurement chain: the K0 static formula matches the coupled-chain
symbol to 1e-15 (1e-10 pre-fix TM); the closed-form Z matches the
symbol solve exactly (the apparent ~1e-4 near-edge deviation scales
linearly with the unit-circle evaluation offset — pure branch-point
evaluation artefact; the production form evaluates the real closed
form on the circle, offset-free).  Lock-in |b/a| of a pure outgoing
wave through the bit-faithful production sampling chain:

    WR-90 TE10   1.02..1.6 f_c:  -153..-165 dB   (continuum Z: -48..-72)
    WR-90 TM11   1.02..1.6 f_c:  -152..-165 dB   (continuum Z: -33..-56)
    round TE11   1.2 f_c:        -132 dB         (continuum Z: -116)
    round TM01   1.2 f_c:        -132 dB         (continuum Z: -110)

The continuum z_wave alone would cap measured floors far above the
-100 dB criterion; the discrete impedance removes the cap entirely.
The a-priori composition — exact DTBC (auto-extended kernel, R1/R2)
+ exact discrete de-stagger + exact discrete wave impedance — leaves
no term above float noise for certified uniform chains.

Run:  python validation/kg_dtbc_precheck_spike.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh.grid import GridLines
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecNumerical,
    build_modal_port,
)
from magnelio.ports._modal.dtbc import destagger_theta, lambda_symbol
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

C0 = 299_792_458.0
_EDGE_OFFSET = 1e-8


# ----------------------------------------------------------------------
# Flat-index helpers (Z_MIN port planes; stride-1 along z asserted)
# ----------------------------------------------------------------------


def flat_e_pec(mesh) -> np.ndarray:
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    return np.concatenate(
        [
            pec[0, :n_Ex],
            pec[1, :n_Ey],
            pec[2, :n_Ez],
        ]
    )


def ez_plane_base(grid) -> np.ndarray:
    """Flat E indices of the Ez (port-normal) family at half-plane 0."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i, j = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), indexing="ij")
    return (n_Ex + n_Ey + (i * (Ny + 1) + j) * Nz).ravel()


def hz_plane_base(grid) -> np.ndarray:
    """Flat H indices of the Hz (port-normal) family at plane 0."""
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    i, j = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    return (n_Hx + n_Hy + (i * Ny + j) * (Nz + 1)).ravel()


class ZChainIndex:
    """Per-plane flat index arrays for a Z_MIN port plane.

    All six FIT component blocks are k-fastest (stride 1 along z), so
    the plane-k index array is the plane-0 array plus k.  Asserted
    against ``PortPlane.e_*_indices_interior``.
    """

    def __init__(self, plane, grid):
        assert plane.face is BoxFace.Z_MIN
        assert np.array_equal(plane.e_u_indices_interior, plane.e_u_indices + 1)
        assert np.array_equal(plane.e_v_indices_interior, plane.e_v_indices + 1)
        self.grid = grid
        self.e_u0 = plane.e_u_indices
        self.e_v0 = plane.e_v_indices
        self.h_u0 = plane.h_u_indices
        self.h_v0 = plane.h_v_indices
        self.e_n0 = ez_plane_base(grid)
        self.h_n0 = hz_plane_base(grid)

    def e_t(self, k):
        return self.e_u0 + k, self.e_v0 + k

    def h_t(self, m):
        """Transversal H at half-plane m + 1/2 (m = 0 is plane_idx_H)."""
        return self.h_u0 + m, self.h_v0 + m

    def e_n(self, m):
        """Normal E (e_z) at half-plane m + 1/2."""
        return self.e_n0 + m

    def h_n(self, k):
        """Normal H (h_z) co-located with E plane k."""
        return self.h_n0 + k


# ----------------------------------------------------------------------
# Raw 3D leapfrog with per-plane modal projections
# ----------------------------------------------------------------------


class Leapfrog3D:
    """Bit-faithful lossless FIT leapfrog on the sparse operators.

    Update order matches the production solver:
    ``e += dt/M_eps * C^T h`` (then PEC zeroing), then
    ``h -= dt/M_mu * C e``.  Samplers run after the E update, i.e.
    on (e at t^{n+1}, h at t^{n+1/2}) — the production recorder's
    sampling point.
    """

    def __init__(self, mesh, dt):
        self.grid = mesh.grid
        self.dt = dt
        self.C = build_curl_matrix(mesh.grid)
        self.CT = self.C.T.tocsr()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        self.m_eps = m_eps
        self.m_mu = m_mu
        self.beta_e = dt / np.where(m_eps > 0, m_eps, 1.0)
        self.beta_h = dt / np.where(m_mu > 0, m_mu, 1.0)
        self.pec_idx = np.where(flat_e_pec(mesh))[0]

    def run(self, e0, n_steps, samplers):
        e = e0.copy()
        e[self.pec_idx] = 0.0
        h = np.zeros(self.C.shape[0])
        out = {name: [fn(e, h)] for name, fn in samplers.items()}
        for _ in range(n_steps):
            e += self.beta_e * (self.CT @ h)
            e[self.pec_idx] = 0.0
            for name, fn in samplers.items():
                out[name].append(fn(e, h))
            h -= self.beta_h * (self.C @ e)
        return {name: np.array(v) for name, v in out.items()}


def plane_projector(idx: ZChainIndex, prof_u, prof_v, n_planes, which="e"):
    """l2 amplitude + squared-remainder sampler over all planes."""
    if which == "e":
        base_u, base_v = idx.e_u0, idx.e_v0
    else:
        base_u, base_v = idx.h_u0, idx.h_v0
    ks = np.arange(n_planes)
    iu = base_u[None, :] + ks[:, None]
    iv = base_v[None, :] + ks[:, None]
    prof = np.concatenate([prof_u, prof_v])
    den = float(prof @ prof)

    def sample(e, h):
        f = e if which == "e" else h
        vals = np.concatenate([f[iu], f[iv]], axis=1)
        amp = (vals @ prof) / den
        rem2 = np.einsum("ij,ij->i", vals, vals) - den * amp**2
        return np.concatenate([amp, np.maximum(rem2, 0.0)])

    return sample, den


# ----------------------------------------------------------------------
# Geometries
# ----------------------------------------------------------------------


def make_wr90(nz=64, dz=1.1e-3):
    a, b = 22.86e-3, 10.16e-3
    grid = GridLines(
        x=np.linspace(0.0, a, 21),
        y=np.linspace(0.0, b, 10),
        z=np.arange(nz + 1) * dz,
    )
    mesh = Mesh.from_grid(grid)  # all-PEC closure is the default
    dt = courant_dt(grid, "normal")
    return mesh, dt


def make_round_wg(n_t_nodes=23, nz=28, dz=1.5e-3):
    R = 10.0e-3
    s_bbox = 2.4 * R
    length = nz * dz
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(-s_bbox / 2, -s_bbox / 2, 0.0), size=(s_bbox, s_bbox, length), material=pec
    )
    inner = Cylinder(origin=(0.0, 0.0, 0.0), radius=R, height=length, axis="z", material=vacuum)
    model = GeometryModel()
    model.add(Difference(bbox, inner))
    model.add(inner)
    t_nodes = np.linspace(-s_bbox / 2, s_bbox / 2, n_t_nodes).tolist()
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        conformal=True,
        max_cell_size=4.0 * s_bbox / (n_t_nodes - 1),
        forced_planes={
            "x": t_nodes,
            "y": t_nodes,
            "z": (np.arange(nz + 1) * dz).tolist(),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=14.0e9)
    # Close the open cylinder ends so the spike cavity is PEC-bounded
    # along z (the identity test needs modal end planes, not the
    # free evolution of an uncovered bbox face).
    mesh = mesh.with_boundary_conditions(
        {
            "zmin": "PEC",
            "zmax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PMC",
            "ymax": "PMC",
        }
    )
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    return mesh, dt


# ----------------------------------------------------------------------
# Part 1 — KG-chain separation on the raw 3D update
# ----------------------------------------------------------------------


def pair_courant(idx: ZChainIndex, lf: Leapfrog3D, dm, k):
    """Modal-weighted pair-product Courant number on plane k."""
    iu, iv = idx.e_t(k)
    hu, hv = idx.h_t(k)
    pair = np.concatenate(
        [
            lf.m_eps[iu] * lf.m_mu[hv],
            lf.m_eps[iv] * lf.m_mu[hu],
        ]
    )
    weight = np.concatenate(
        [
            lf.m_eps[iu] * dm.e_u_profile**2,
            lf.m_eps[iv] * dm.e_v_profile**2,
        ]
    )
    active = weight > 1e-12 * weight.max()
    r_pairs = lf.dt / np.sqrt(pair[active])
    w = weight[active]
    r_mean = float(w @ r_pairs / w.sum())
    spread = float(math.sqrt(w @ (r_pairs - r_mean) ** 2 / w.sum()) / r_mean)
    return r_mean, spread


def separation_check(name, mesh, dt, mode_type, f_calc, n_steps=400):
    """Run the modal-IC cavity test; report chain residual and fits."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    spec = PortSpecNumerical(name="p1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=mode_type)
    lf = Leapfrog3D(mesh, dt)  # snapshots PEC before factory
    op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)
    dm = op.discrete_modes[0]
    omega_c = dm.mode.omega_c
    q = omega_c * dt

    nz = mesh.Nz
    idx = ZChainIndex(op.plane, mesh.grid)

    k_mid = nz // 2
    r_mid, spread_mid = pair_courant(idx, lf, dm, k_mid)
    r_planes = np.array([pair_courant(idx, lf, dm, k)[0] for k in range(1, nz - 1)])
    r_axial_spread = float(np.ptp(r_planes) / r_mid)

    # Modal IC: e_t bump centred in the guide.
    sigma_k = max(3.0, nz / 10.0)
    e0 = np.zeros(lf.CT.shape[0])
    for k in range(nz + 1):
        f = math.exp(-(((k - k_mid) / sigma_k) ** 2))
        if f < 1e-14:
            continue
        iu, iv = idx.e_t(k)
        e0[iu] += f * dm.e_u_profile
        e0[iv] += f * dm.e_v_profile

    et_sampler, den_e = plane_projector(idx, dm.e_u_profile, dm.e_v_profile, nz + 1, "e")
    ht_sampler, den_h = plane_projector(idx, dm.h_u_profile, dm.h_v_profile, nz, "h")
    en_base = idx.e_n0

    def ez_max_sampler(e, h):
        del h
        ks = np.arange(nz)
        vals = e[en_base[None, :] + ks[:, None]]
        return float(np.abs(vals).max())

    out = lf.run(
        e0,
        n_steps,
        {
            "et": et_sampler,
            "ht": ht_sampler,
            "ez": ez_max_sampler,
        },
    )

    et = out["et"]
    u = et[:, : nz + 1]
    u_max = float(np.abs(u).max())
    et_leak = math.sqrt(float(et[:, nz + 1 :].max())) / (u_max * math.sqrt(den_e))
    ht = out["ht"]
    g_max = float(np.abs(ht[:, :nz]).max())
    ht_leak = math.sqrt(float(ht[:, nz:].max())) / max(g_max * math.sqrt(den_h), 1e-300)
    e_val_max = u_max * float(np.abs(np.concatenate([dm.e_u_profile, dm.e_v_profile])).max())
    ez_rel = float(np.array(out["ez"]).max()) / e_val_max

    ks = np.arange(1, nz)
    d2t = u[2:, ks] - 2.0 * u[1:-1, ks] + u[:-2, ks]
    d2z = u[1:-1, ks + 1] - 2.0 * u[1:-1, ks] + u[1:-1, ks - 1]
    uc = u[1:-1, ks]
    src = u[0, ks][None, :] * np.ones((u.shape[0] - 2, 1)) if mode_type is ModeType.TM else 0.0

    # A-priori chain: r from the pair product, q from the 2D
    # eigenvalue.
    resid = d2t - r_mid**2 * d2z + q * q * (uc - src)
    rel = float(np.abs(resid).max()) / u_max

    # Best-fit (r^2, q^2) as an independent diagnostic.
    A = np.column_stack([d2z.ravel(), -(uc - src).ravel()])
    coef, *_ = np.linalg.lstsq(A, d2t.ravel(), rcond=None)
    r2_fit, q2_fit = float(coef[0]), float(coef[1])

    print(f"  {name}")
    print(f"    r = {r_mid:.12f}  (pair spread {spread_mid:.2e}, axial {r_axial_spread:.2e})")
    print(f"    q = {q:.12f}  (omega_c/2pi = {omega_c / 2 / math.pi / 1e9:.6f} GHz)")
    print(f"    KG-chain residual (a-priori r, q): {rel:.3e}")
    print(
        f"    fit: r2/r2_ap-1 = {r2_fit / r_mid**2 - 1:+.3e}   "
        f"q2/q2_ap-1 = {q2_fit / (q * q) - 1:+.3e}"
    )
    print(f"    leakage: e_t {et_leak:.3e}   h_t {ht_leak:.3e}   e_z {ez_rel:.3e}")
    return {
        "dm": dm,
        "lf": lf,
        "idx": idx,
        "r": r_mid,
        "q": q,
        "residual": rel,
    }


# ----------------------------------------------------------------------
# Part 2 — coupled-chain couplings, discrete Z, lock-in b/a
# ----------------------------------------------------------------------


def rank1_coeff(vec, prof):
    """(coefficient, relative remainder) of vec against prof (l2)."""
    den = float(prof @ prof)
    c = float(vec @ prof) / den
    rem = vec - c * prof
    rel = float(np.linalg.norm(rem)) / max(float(np.linalg.norm(vec)), 1e-300)
    return c, rel


def extract_couplings(res, mode_type, k0=None):
    """Scalar couplings of the reduced 1D system, from the operators.

    The reduced system integrated by the 1D chain replica is

        e-step:  u_k += dt * (ug0 * g_{k-1/2} + ug1 * g_{k+1/2}
                              + up * p_k)              [TE]
                 w_m += dt * wg * g_m                  [TM]
        h-step:  g_m += dt * (gu0 * u_m + gu1 * u_{m+1} + gw * w_m)
                 p_k += dt * pu * u_k                  [TE]

    with m the half-plane index.  All coefficients are measured by
    applying the 3D update operators to single-plane profiles; the
    reported remainders certify that the modal subspace is invariant
    under the 3D update.
    """
    lf, idx, dm = res["lf"], res["idx"], res["dm"]
    nz = lf.grid.Nz
    if k0 is None:
        k0 = nz // 2
    prof_e = np.concatenate([dm.e_u_profile, dm.e_v_profile])
    prof_h = np.concatenate([dm.h_u_profile, dm.h_v_profile])

    def slab(k, pu_, pv_, which):
        v = np.zeros(lf.CT.shape[0] if which == "e" else lf.C.shape[0])
        iu, iv = idx.e_t(k) if which == "e" else idx.h_t(k)
        v[iu] = pu_
        v[iv] = pv_
        return v

    def e_response(h_vec):
        """e-step response incl. the solver's PEC zeroing (without
        it, wall-tangential rows corrupt the rank-1 remainders)."""
        x = lf.beta_e * (lf.CT @ h_vec)
        x[lf.pec_idx] = 0.0
        return x

    def h_slice(vec, m):
        hu, hv = idx.h_t(m)
        return np.concatenate([vec[hu], vec[hv]])

    def e_slice(vec, k):
        iu, iv = idx.e_t(k)
        return np.concatenate([vec[iu], vec[iv]])

    out = {}
    rems = []

    # h-step response to a u slab: dh = -beta_h * (C e).
    y = -lf.beta_h * (lf.C @ slab(k0, dm.e_u_profile, dm.e_v_profile, "e"))
    c, rel = rank1_coeff(h_slice(y, k0), prof_h)
    out["gu0"] = c / lf.dt
    rems.append(rel)
    c, rel = rank1_coeff(h_slice(y, k0 - 1), prof_h)
    out["gu1"] = c / lf.dt
    rems.append(rel)

    if mode_type is ModeType.TE:
        hz = y[idx.h_n(k0)]
        prof_p = hz / float(np.linalg.norm(hz))
        out["pu"] = float(hz @ prof_p) / lf.dt
        v = np.zeros(lf.C.shape[0])
        v[idx.h_n(k0)] = prof_p
        x = e_response(v)
        c, rel = rank1_coeff(e_slice(x, k0), prof_e)
        out["up"] = c / lf.dt
        rems.append(rel)

    # e-step response to a g slab: de = +beta_e * (C^T h).
    x = e_response(slab(k0, dm.h_u_profile, dm.h_v_profile, "h"))
    c, rel = rank1_coeff(e_slice(x, k0), prof_e)
    out["ug1"] = c / lf.dt
    rems.append(rel)
    c, rel = rank1_coeff(e_slice(x, k0 + 1), prof_e)
    out["ug0"] = c / lf.dt
    rems.append(rel)

    if mode_type is ModeType.TM:
        ez = x[idx.e_n(k0)]
        prof_w = ez / float(np.linalg.norm(ez))
        out["wg"] = float(ez @ prof_w) / lf.dt
        v = np.zeros(lf.CT.shape[0])
        v[idx.e_n(k0)] = prof_w
        y2 = -lf.beta_h * (lf.C @ v)
        c, rel = rank1_coeff(h_slice(y2, k0), prof_h)
        out["gw"] = c / lf.dt
        rems.append(rel)

    # Static projection norms: V = nV * u, I = nI * g.
    iu, iv = idx.e_t(k0)
    hu, hv = idx.h_t(k0)
    out["nV"] = float(lf.m_eps[iu] @ dm.e_u_profile**2 + lf.m_eps[iv] @ dm.e_v_profile**2)
    out["nI"] = float(lf.m_mu[hu] @ dm.h_u_profile**2 + lf.m_mu[hv] @ dm.h_v_profile**2)
    out["rank1_max"] = max(rems)

    # Consistency with part 1.
    dt2 = lf.dt**2
    out["r2_c"] = -dt2 * out["ug0"] * out["gu1"]
    out["r2_c2"] = -dt2 * out["ug1"] * out["gu0"]
    out["q2_c"] = (
        -dt2 * out["up"] * out["pu"] if mode_type is ModeType.TE else -dt2 * out["gw"] * out["wg"]
    )
    return out


def discrete_VI_ratio(w_dt, cpl, dt, r, q, mode_type):
    """I(omega)/V(omega) of the discrete outgoing wave from the
    coupled-chain symbol algebra, at the production sampling points
    and *after* the e^{+j omega dt/2} temporal rotation (i.e. the
    remaining spatial factor is exactly lambda^{1/2})."""
    z = (1.0 + _EDGE_OFFSET) * np.exp(1j * np.asarray(w_dt))
    lam = lambda_symbol(z, r, q)
    ls = np.sqrt(lam)
    s2 = np.sqrt(z) - 1.0 / np.sqrt(z)
    num = dt * (cpl["gu0"] + cpl["gu1"] * lam) / ls
    if mode_type is ModeType.TE:
        G = num / s2
    else:
        G = num / (s2 - dt * dt * cpl["gw"] * cpl["wg"] / s2)
    return (cpl["nI"] / cpl["nV"]) * G * ls


def closed_form_Z(w_dt, r, q, K0, mode_type):
    s = np.sin(np.asarray(w_dt) / 2.0)
    rad = np.sqrt((s**2 - (q / 2.0) ** 2).astype(complex))
    if mode_type is ModeType.TE:
        return K0 * s / rad
    return K0 * rad / s


def chain_lockin(cpl, dt, r, q, mode_type, w_dt, n_sites, k_m, sigma_steps, n_meas0, n_win):
    """CW lock-in on the coupled 1D chain replica.

    Returns complex (V, I) amplitudes fitted over the window
    [n_meas0, n_meas0 + n_win) against e^{+j w n dt} on the *naive*
    recorder time axis (exactly what compute_s_parameters sees),
    plus the fit residual as the measurement floor.
    """
    del r, q  # dynamics fully encoded in the couplings
    n_steps = n_meas0 + n_win + 2
    t0 = 5.0 * sigma_steps

    u = np.zeros(n_sites + 1)
    g = np.zeros(n_sites)
    is_te = mode_type is ModeType.TE
    aux = np.zeros(n_sites + 1 if is_te else n_sites)

    ug0, ug1 = cpl["ug0"], cpl["ug1"]
    gu0, gu1 = cpl["gu0"], cpl["gu1"]
    if is_te:
        up, pu = cpl["up"], cpl["pu"]
    else:
        gw, wg = cpl["gw"], cpl["wg"]

    v_rec = np.empty(n_steps)
    i_rec = np.empty(n_steps)
    for n in range(n_steps):
        # e-step (u, and w for TM, use g at t^{n+1/2}).
        du = dt * (ug0 * g[:-1] + ug1 * g[1:])
        if is_te:
            u[1:-1] += du + dt * up * aux[1:-1]
        else:
            u[1:-1] += du
            aux += dt * wg * g
        amp = 0.5 * (1.0 + float(erf(((n + 1) - t0) / (math.sqrt(2.0) * sigma_steps))))
        u[0] = amp * math.sin(w_dt * (n + 1))
        u[-1] = 0.0
        # Production recorder point: e at t^{n+1}, h at t^{n+1/2}.
        v_rec[n] = cpl["nV"] * u[k_m]
        i_rec[n] = cpl["nI"] * g[k_m]
        # h-step (uses u at t^{n+1}).
        if is_te:
            aux += dt * pu * u
        g += dt * (gu0 * u[:-1] + gu1 * u[1:])
        if not is_te:
            g += dt * gw * aux

    n_grid = np.arange(n_meas0, n_meas0 + n_win)
    basis = np.column_stack([np.cos(w_dt * n_grid), np.sin(w_dt * n_grid)])
    cv, *_ = np.linalg.lstsq(basis, v_rec[n_grid], rcond=None)
    ci, *_ = np.linalg.lstsq(basis, i_rec[n_grid], rcond=None)
    res_fit = float(
        np.linalg.norm(v_rec[n_grid] - basis @ cv) / max(np.linalg.norm(v_rec[n_grid]), 1e-300)
    )
    V = cv[0] - 1j * cv[1]
    I = ci[0] - 1j * ci[1]
    return V, I, res_fit


def measurement_chain_check(name, res, mode_type, facs=(1.02, 1.05, 1.2, 1.6)):
    lf = res["lf"]
    dt, r, q = lf.dt, res["r"], res["q"]
    cpl = extract_couplings(res, mode_type)
    print(f"  {name}")
    print(f"    rank-1 application remainder: {cpl['rank1_max']:.3e}")
    print(
        f"    couplings vs part 1:  "
        f"r2 {cpl['r2_c'] / r**2 - 1:+.3e} / "
        f"{cpl['r2_c2'] / r**2 - 1:+.3e}   "
        f"q2 {cpl['q2_c'] / q**2 - 1:+.3e}"
    )
    w_co = 2.0 * math.asin(q / 2.0)
    K0_static = abs(r * cpl["nV"] / (dt * cpl["nI"] * cpl["gu0"]))
    w_ref = np.array([min(3.0 * w_co, 0.5 * math.pi)])
    z_ref = (1.0 + _EDGE_OFFSET) * np.exp(1j * w_ref)
    ls_ref = np.sqrt(lambda_symbol(z_ref, r, q))
    Z_ref = (ls_ref / discrete_VI_ratio(w_ref, cpl, dt, r, q, mode_type))[0].real
    Z_cf_ref = closed_form_Z(w_ref, r, q, K0_static, mode_type)[0].real
    print(
        f"    Z(w_ref): symbol {Z_ref:.6f}  closed-form(K0 static) "
        f"{Z_cf_ref:.6f}  ratio-1 {Z_ref / Z_cf_ref - 1:+.3e}"
    )
    w_band = w_co * np.array([1.001, 1.01, 1.05, 1.2, 1.5, 1.9])
    w_band = w_band[w_band < 0.9 * math.pi]
    z_band = (1.0 + _EDGE_OFFSET) * np.exp(1j * w_band)
    ls_band = np.sqrt(lambda_symbol(z_band, r, q))
    Z_sym = ls_band / discrete_VI_ratio(w_band, cpl, dt, r, q, mode_type)
    Z_cf = closed_form_Z(w_band, r, q, K0_static, mode_type)
    shape_dev = float(np.abs(Z_sym / Z_cf - 1.0).max())
    print(f"    closed-form Z deviation across band: {shape_dev:.3e}")
    print("    lock-in |b/a| (pure outgoing wave):  w/w_co   discrete Z    continuum Z    fit-res")
    for fac in facs:
        w = fac * w_co
        if w >= 0.95 * math.pi:
            continue
        period = 2.0 * math.pi / w
        dw = w - w_co
        sigma_steps = min(max(6.0 / dw, 8.0 * period), 200.0 * period)
        sb = math.sqrt(max(math.sin(w / 2.0) ** 2 - (q / 2) ** 2, 1e-30)) / r
        v_g = r * r * math.sin(2.0 * math.asin(min(sb, 1.0))) / math.sin(w)
        k_m = 40
        n_meas0 = int(10.0 * sigma_steps + 40 * period + k_m / max(v_g, 1e-3))
        n_win = int(20 * period)
        n_total = n_meas0 + n_win + 2
        n_sites = max(int(0.75 * n_total * max(v_g, 1e-3)) + 2 * k_m, 4000)
        V, I, res_fit = chain_lockin(
            cpl, dt, r, q, mode_type, w, n_sites, k_m, sigma_steps, n_meas0, n_win
        )
        I_rot = I * np.exp(1j * w / 2.0)
        theta = destagger_theta(np.array([w]), r, q)[0]
        ep, em = np.exp(theta), np.exp(-theta)
        ratios = []
        for which in ("disc", "cont"):
            if which == "disc":
                Z = closed_form_Z(np.array([w]), r, q, K0_static, mode_type)[0]
            else:
                Z = res["dm"].mode.z_modal(w / dt)
            sz = np.sqrt(Z)
            a = (V / sz * ep + sz * I_rot) / (ep + em)
            b = (V / sz * em - sz * I_rot) / (ep + em)
            ratios.append(20.0 * math.log10(max(abs(b / a), 1e-300)))
        print(f"      {fac:5.2f}   {ratios[0]:9.1f} dB   {ratios[1]:9.1f} dB     {res_fit:.1e}")
    return {"K0": K0_static, "shape_dev": shape_dev}


def main():
    print("=" * 72)
    print("Part 1 — KG-chain separation of production 2D modes (3D leapfrog)")
    print("=" * 72)
    mesh, dt = make_wr90()
    res_te = separation_check("WR-90 TE10 (uniform, eigsh path)", mesh, dt, ModeType.TE, 10.0e9)
    mesh, dt = make_wr90()
    res_tm = separation_check("WR-90 TM11 (uniform, normal-E path)", mesh, dt, ModeType.TM, 20.0e9)
    mesh, dt = make_round_wg()
    dz_all = np.diff(mesh.grid.z)
    n_uni = int((np.abs(dz_all / dz_all[len(dz_all) // 2] - 1.0) < 1e-12).sum())
    print(
        f"  [round WG mesh {mesh.Nx}x{mesh.Ny}x{mesh.Nz}, uniform z planes {n_uni}/{dz_all.size}]"
    )
    res_te11 = separation_check(
        "round WG TE11 (conformal, eigsh path)", mesh, dt, ModeType.TE, 13.0e9
    )
    mesh, dt = make_round_wg()
    res_tm01 = separation_check(
        "round WG TM01 (conformal, normal-E path)", mesh, dt, ModeType.TM, 13.0e9
    )
    print()
    print("=" * 72)
    print("Part 2 — discrete wave impedance + production measurement chain")
    print("=" * 72)
    measurement_chain_check("WR-90 TE10", res_te, ModeType.TE)
    measurement_chain_check("WR-90 TM11", res_tm, ModeType.TM)
    # Round WG: the lock-in replica depends only on (r, q, couplings)
    # — the sampling-chain certificate is covered by the WR-90 runs.
    # The tiny conformal r (forced-plane snapping yields a small dt)
    # makes near-edge lock-ins expensive; one mid-band point each.
    measurement_chain_check("round WG TE11", res_te11, ModeType.TE, facs=(1.2,))
    measurement_chain_check("round WG TM01", res_tm01, ModeType.TM, facs=(1.2,))


if __name__ == "__main__":
    main()
