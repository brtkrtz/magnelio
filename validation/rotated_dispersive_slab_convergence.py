#!/usr/bin/env python
"""WP-C5 (DD-093): rotated dispersive-slab cavity vs exact 1D reference.

The DD-051 rotation trick applied to the conformal dispersive/σ*
boundaries: a PEC-walled 2D cavity (a × b in the xy-plane, one cell
thick in z between PMC lids) holding a dispersive slab is rotated as a
WHOLE about z relative to the grid — the physics is rotation-
invariant, the exact planar layered-cavity reference stays valid, and
the grid sees oblique dispersive interfaces along the whole contour.
(Developer decision session 126: the cavity observable replaces the
plan's |S21| wording — waveguide ports are axis-bound.)

Mode family: TE_z (H = ẑ·h(y'), E ⊥ ẑ) with NO variation along the
slab plane — under PMC lids this is an exact eigenfamily for ALL
three channels, because the interface conditions (H_z continuous,
E_x' ∝ (1/ε)·h' continuous) are compatible with a pure y'-profile
even when ε or μ jumps.  Characteristic equation of the fundamental:

    (k1/ε1)·tan(k1 t) + (k2/ε2)·tan(k2 (b − t)) = 0,

with k_i(ω) = ω√(ε_i(ω) μ_i(ω))/c0 from the SAME pole-residue models
the solver integrates (σ* enters as μ_eff = μ_r + σ*/(jωμ0)); complex
roots give (f, γ) for the lossy channels.

Observable: complex frequency of the ring-down of the modal amplitude
a(t) = <IC, e(t)> after seeding the exact lossless mode profile
E_x'(y') ∝ (1/ε_i)·h_i'(y') rotated onto the grid edges.

Configs per resolution: conformal (the DD-093 default) vs staircase
(the pre-plan ADE/σ* booking, reproduced by dropping the WP-C1
fraction containers from the meshed Edge/FaceMaterialData before
operator construction — the static conformal ε/μ̄ stays, which is
exactly the pre-plan state).

Run:  python validation/rotated_dispersive_slab_convergence.py
        [--channel eps|mu|sigma_m] [--cells 2e-3 1.5e-3 1e-3 0.75e-3]
        [--steps 16000] [--backend numpy|auto] [--rot 30]
"""

from __future__ import annotations

import argparse
import cmath
import math
import time
import warnings
from pathlib import Path

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio.boundaries.pec import PECBoundary
from magnelio.boundaries.pmc import PMCBoundary
from magnelio.geo import Brick, GeometryModel
from magnelio.geo.transforms import rotate
from magnelio.materials import DispersionModel
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

C0 = 299_792_458.0
MU0 = 4e-7 * math.pi

# Cavity: a (x') × b (y') in-plane; slab occupies y' ∈ [0, t].
A_X, B_Y = 40e-3, 20e-3
T_SLAB = 8e-3
ROT_Z_DEG = 30.0
F_MAX = 8.0e9

EPS_INF, DELTA_EPS, TAU = 2.0, 1.0, 1.0e-11
MU_INF, DELTA_MU = 2.0, 1.0
SIGMA_M = 400.0  # Ω/m — Q ≈ 400 class: cleanly fittable decay

OUTPUT_DIR = Path(__file__).parent / "rotated_dispersive_slab_results"


# ---------------------------------------------------------------------------
# Channel definitions: slab material + its exact (eps(ω), mu(ω)) laws
# ---------------------------------------------------------------------------


def _eval_model(model: DispersionModel, f: complex) -> complex:
    """Pole-residue evaluation at COMPLEX frequency (the root search
    walks off the real axis; ``model.evaluate`` is real-ω only).
    Conjugate-pair poles are stored once — add the partner's term."""
    jw = 1j * 2 * math.pi * f
    val = complex(model.eps_inf)
    for a, r in model.poles:
        val += r / (jw - a)
        if a.imag != 0.0:
            val += r.conjugate() / (jw - a.conjugate())
    return val


def _channel(name: str):
    if name == "eps":
        model = DispersionModel.debye(EPS_INF, DELTA_EPS, TAU, f_band=(1e8, 2e10))
        mat = Material.dispersive("slab", model)

        def eps_f(f):
            return _eval_model(model, f)

        def mu_f(f):
            return 1.0 + 0.0j
    elif name == "mu":
        model = DispersionModel.debye(MU_INF, DELTA_MU, TAU, f_band=(1e8, 2e10))
        mat = Material.dispersive_mu("slab", model, epsilon=2.0)

        def eps_f(f):
            return 2.0 + 0.0j

        def mu_f(f):
            return _eval_model(model, f)
    elif name == "sigma_m":
        mat = Material(name="slab", epsilon=(2.0,) * 3, mu=(2.0,) * 3, sigma_m=(SIGMA_M,) * 3)

        def eps_f(f):
            return 2.0 + 0.0j

        def mu_f(f):
            return 2.0 + SIGMA_M / (1j * 2 * math.pi * f * MU0)
    else:
        raise ValueError(name)
    return mat, eps_f, mu_f


# ---------------------------------------------------------------------------
# Exact 1D layered-cavity reference (TE_z family under PMC lids)
# ---------------------------------------------------------------------------


def _char(f: complex, eps_f, mu_f) -> complex:
    e1, m1 = eps_f(f), mu_f(f)
    k1 = 2 * math.pi * f * cmath.sqrt(e1 * m1) / C0
    k2 = 2 * math.pi * f / C0
    return (k1 / e1) * cmath.tan(k1 * T_SLAB) + (k2 / 1.0) * cmath.tan(k2 * (B_Y - T_SLAB))


def reference_root(eps_f, mu_f) -> complex:
    """Complex fundamental root: real-axis sign-change candidates on
    the lossless magnitudes, each verified by bisection (tan-pole flips
    diverge instead of converging), then complex Newton."""

    def F_real(f: float) -> float:
        e1 = eps_f(complex(f)).real
        m1 = mu_f(complex(f)).real
        k1 = 2 * math.pi * f * math.sqrt(abs(e1 * m1)) / C0
        k2 = 2 * math.pi * f / C0
        return (k1 / e1) * math.tan(k1 * T_SLAB) + k2 * math.tan(k2 * (B_Y - T_SLAB))

    fs = np.linspace(1.0e9, 8e9, 4000)
    vals = np.array([F_real(f) for f in fs])
    idx = np.nonzero(np.diff(np.sign(vals)) != 0)[0]
    f0 = None
    for i in idx:
        lo, hi = float(fs[i]), float(fs[i + 1])
        flo = F_real(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = F_real(mid)
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        mid = 0.5 * (lo + hi)
        if abs(F_real(mid)) < 1.0:  # genuine root, not a tan pole
            f0 = complex(mid)
            break
    if f0 is None:
        raise RuntimeError("no root bracketed")
    df = 1e3
    for _ in range(200):
        F = _char(f0, eps_f, mu_f)
        dF = (_char(f0 + df, eps_f, mu_f) - _char(f0 - df, eps_f, mu_f)) / (2 * df)
        step = F / dF
        f0 = f0 - step
        if abs(step) < 1e-3:
            break
    return f0


def lossless_profile(f: float, eps_f, mu_f):
    """E_x'(y') of the (lossless-magnitude) fundamental — the IC seed.

    H_z: h1 = cos(k1 y'), h2 = C·cos(k2 (b − y')) (PEC walls at
    y' = 0, b enforce E_x' ∝ h' = 0 there); E_x' ∝ (1/ε_i) h_i'.
    Continuity of E_x' fixes the branch amplitudes; the overall scale
    is arbitrary.
    """
    e1 = eps_f(complex(f)).real
    m1 = mu_f(complex(f)).real
    k1 = 2 * math.pi * f * math.sqrt(abs(e1 * m1)) / C0
    k2 = 2 * math.pi * f / C0
    # E_x' branch shapes (up to sign): s1 = (k1/e1) sin(k1 y'),
    # s2 = (k2/1) sin(k2 (b−y')).  Match at y' = t.
    s1t = (k1 / e1) * math.sin(k1 * T_SLAB)
    s2t = k2 * math.sin(k2 * (B_Y - T_SLAB))
    C = s1t / s2t if abs(s2t) > 1e-12 else 1.0

    def prof(y):
        y = np.asarray(y)
        inside = (k1 / e1) * np.sin(k1 * np.clip(y, 0, T_SLAB))
        outside = C * k2 * np.sin(k2 * np.clip(B_Y - y, 0, B_Y))
        return np.where(y <= T_SLAB, inside, outside)

    return prof


# ---------------------------------------------------------------------------
# Rotated thin cavity mesh + modal IC + ring-down march
# ---------------------------------------------------------------------------


def _rot_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def build_mesh(h: float, slab_mat: Material, rot_deg: float) -> Mesh:
    """Rotated PEC FRAME (not a PEC background): a background-PEC mesh
    marks the DOMAIN-boundary edges PEC, which would turn the z lids
    into electric walls and kill the TE_z family (measured: the
    28-GHz z-half-wave mode took over).  With an explicit rotated
    frame the lid edges inside the cavity stay free — the PMC natural
    boundary then applies."""
    from magnelio.geo import Difference

    thick = 2 * h  # z: two cells between PMC lids
    w = 3e-3  # PEC frame thickness
    frame = Brick(
        origin=(-A_X / 2 - w, -B_Y / 2 - w, 0),
        size=(A_X + 2 * w, B_Y + 2 * w, thick),
        material=Material.pec(),
    )
    hole = Brick(origin=(-A_X / 2, -B_Y / 2, 0), size=(A_X, B_Y, thick), material=Material.air())
    slab = Brick(origin=(-A_X / 2, -B_Y / 2, 0), size=(A_X, T_SLAB, thick), material=slab_mat)
    air = Brick(
        origin=(-A_X / 2, -B_Y / 2 + T_SLAB, 0),
        size=(A_X, B_Y - T_SLAB, thick),
        material=Material.air(),
    )
    rot = [rotate(s, (0, 0, 1), rot_deg) for s in (frame, hole, slab, air)]
    model = GeometryModel()
    model.add(Difference(rot[0], rot[1]))
    model.add(rot[2])
    model.add(rot[3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Mesh.from_geometry(
            model,
            MeshControl(
                max_cell_size=h,
                # h must be the ONLY resolution driver for the order
                # study — the default λ/10 rule (1.87 mm in the slab
                # at 8 GHz) would otherwise pin every h ≥ 1.5 mm to
                # the same grid.
                min_nodes_per_wavelength=4,
                min_cells_per_feature=0,
                forced_planes={"z": np.array([0.0, h, thick])},
            ),
            f_max=F_MAX,
        )


def _edge_geometry(grid):
    """Per-E-edge midpoints (3, n) and axis lengths (n,) in flat order."""
    x, y, z = grid.x, grid.y, grid.z
    xm, ym, zm = (0.5 * (v[:-1] + v[1:]) for v in (x, y, z))
    dx, dy, dz = (np.diff(v) for v in (x, y, z))
    mids, lens, axes = [], [], []
    for pts, L_arr, ax in (
        ((xm, y, z), dx[:, None, None], 0),
        ((x, ym, z), dy[None, :, None], 1),
        ((x, y, zm), dz[None, None, :], 2),
    ):
        X, Y, Z = np.meshgrid(*pts, indexing="ij")
        L = np.broadcast_to(L_arr, X.shape)
        mids.append(np.stack([X, Y, Z]).reshape(3, -1))
        lens.append(L.ravel())
        axes.append(np.full(L.size, ax, dtype=int))
    return (np.concatenate(mids, axis=1), np.concatenate(lens), np.concatenate(axes))


def modal_ic(mesh, prof, rot_deg: float) -> np.ndarray:
    """IC vector: the rotated TE_z mode's E field (x'-polarised,
    profile along y') projected onto the grid edges (e = E·l)."""
    R = _rot_matrix(rot_deg)
    mids, lens, axes = _edge_geometry(mesh.grid)
    local = R.T @ mids
    xl, yl = local[0], local[1]
    inside = (np.abs(xl) < A_X / 2 - 1e-9) & (yl > -B_Y / 2 + 1e-9) & (yl < B_Y / 2 - 1e-9)
    e_dir = R[:, 0]  # rotated x' polarisation
    comp = e_dir[axes]
    ic = np.where(inside, prof(yl + B_Y / 2) * comp * lens, 0.0)
    return ic


class _ModalProbe:
    """Diagnostics hook recording a(t) = <ic, e>."""

    def __init__(self, ic):
        self.ic = ic
        self.trace: list[float] = []

    def record(self, fields, t):
        e = fields.e_flat
        e = e.get() if hasattr(e, "get") else e
        self.trace.append(float(np.dot(self.ic, e)))


def ring_down(mesh, ic, n_steps, backend) -> tuple[np.ndarray, float]:
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    probe = _ModalProbe(ic)
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={
            "zmin": PMCBoundary("zmin", mesh.grid),
            "zmax": PMCBoundary("zmax", mesh.grid),
            **{f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax")},
        },
        dt=dt,
        total_time_steps=n_steps,
        verbose=False,
        backend=backend,
        precision="double",
        diagnostics=[probe],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.setup()
        solver._fields.e_flat[:] = solver._xp.asarray(ic)
        solver.run()
    return np.asarray(probe.trace), dt


def extract_mode(trace: np.ndarray, dt: float) -> tuple[float, float]:
    """(f, γ) via FFT peak + complex demodulation of the tail."""
    n0 = len(trace) // 8
    sig = trace[n0:]
    w = np.hanning(sig.size)
    spec = np.abs(np.fft.rfft(sig * w))
    fax = np.fft.rfftfreq(sig.size, d=dt)
    i = int(np.argmax(spec))
    if 0 < i < spec.size - 1:
        denom = spec[i - 1] - 2 * spec[i] + spec[i + 1]
        delta = 0.5 * (spec[i - 1] - spec[i + 1]) / denom if denom else 0.0
    else:
        delta = 0.0
    f_hat = float(fax[i] + delta * (fax[1] - fax[0]))

    t = np.arange(sig.size) * dt
    z = sig * np.exp(-2j * math.pi * f_hat * t)
    win = max(int(2.0 / (f_hat * dt)), 4)
    kernel = np.ones(win) / win
    env = np.convolve(z, kernel, mode="valid")
    m = np.abs(env)
    keep = m > 1e-3 * m.max()
    tt = np.arange(env.size)[keep] * dt
    gamma_fit = np.polyfit(tt, np.log(m[keep]), 1)[0]
    phase = np.unwrap(np.angle(env[keep]))
    df = np.polyfit(tt, phase, 1)[0] / (2 * math.pi)
    return f_hat + df, -gamma_fit


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def strip_fraction_containers(mesh) -> None:
    """Reproduce the pre-plan (staircase ADE/σ*) state on this mesh."""
    em, fm = mesh.edge_material, mesh.face_material
    if em is not None:
        em.material_fractions = None
        em.fraction_mids = None
    if fm is not None:
        fm.material_fractions = None
        fm.fraction_mids = None
        fm.sigma_m_avg = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="eps", choices=["eps", "mu", "sigma_m"])
    ap.add_argument("--cells", type=float, nargs="+", default=[2e-3, 1.5e-3, 1e-3, 0.75e-3])
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--rot", type=float, default=ROT_Z_DEG)
    args = ap.parse_args()

    mat, eps_f, mu_f = _channel(args.channel)
    root = reference_root(eps_f, mu_f)
    # e^{jωt}: a decaying resonance has Im(f̂) > 0, amplitude e^{−γt}
    # with γ = 2π·Im(f̂).
    f_ref, gamma_ref = root.real, 2 * math.pi * root.imag
    print(
        f"[{args.channel}] exact reference: f = {f_ref / 1e9:.6f} GHz, "
        f"gamma = {gamma_ref:.4e} 1/s "
        f"(Q = {math.pi * f_ref / max(gamma_ref, 1e-30):.1f})"
    )
    prof = lossless_profile(f_ref, eps_f, mu_f)

    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for h in args.cells:
        for config in ("staircase", "conformal"):
            t0 = time.perf_counter()
            mesh = build_mesh(h, mat, args.rot)
            if config == "staircase":
                strip_fraction_containers(mesh)
            ic = modal_ic(mesh, prof, args.rot)
            trace, dt = ring_down(mesh, ic, args.steps, args.backend)
            f_sim, g_sim = extract_mode(trace, dt)
            err_f = (f_sim - f_ref) / f_ref
            err_g = (g_sim - gamma_ref) / max(abs(gamma_ref), 1e-30)
            wall = time.perf_counter() - t0
            print(
                f"[{args.channel}] h={h * 1e3:5.2f} mm {config:>10}: "
                f"mesh ({mesh.Nx},{mesh.Ny},{mesh.Nz}) "
                f"f = {f_sim / 1e9:.6f} GHz (err {err_f:+.3e})  "
                f"gamma = {g_sim:.4e} (err {err_g:+.3e})  "
                f"[{wall:.0f} s]",
                flush=True,
            )
            rows.append((h, config == "conformal", f_sim, g_sim, err_f, err_g))

    arr = np.array(rows, dtype=float)
    np.savez(
        OUTPUT_DIR / f"{args.channel}.npz",
        rows=arr,
        f_ref=f_ref,
        gamma_ref=gamma_ref,
        columns="h,conformal,f_sim,gamma_sim,err_f,err_g",
    )
    for conformal in (0.0, 1.0):
        sel = arr[arr[:, 1] == conformal]
        name = "conformal" if conformal else "staircase"
        if len(sel) >= 2:
            p = np.polyfit(np.log(sel[:, 0]), np.log(np.abs(sel[:, 4]) + 1e-16), 1)[0]
            print(
                f"[{args.channel}] {name}: |err_f| order ≈ {p:.2f}, "
                f"finest |err_f| = {abs(sel[-1, 4]):.3e}"
            )


if __name__ == "__main__":
    main()
