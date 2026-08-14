"""WP-R5 trigger benchmark: iris-loaded pillbox cavity (H-face donor).

DD-051 left the enlarged-cell donor mechanism for H-faces as an open
follow-up with explicit trigger conditions (STATUS construction site 2):

    (a) ``subcell_floor_histogram``-style floor share > 70 % of the
        cat-2 H-faces on the geometry (faces with
        ``A_face_free / A_face < 1 %`` fall back to staircase M_μ),
    (b) a curved-PEC convergence benchmark lands more than 1–2 % off,
    (c) the mode energy is concentrated near the PEC boundary.

Any one condition is necessary, the three together are sufficient.
Round-WG / rect-WG / coax / smooth cavities do NOT hit the trigger
(measured: sub-percent accuracy at ~48 % floor share, because the
floored faces carry no mode energy there).  This script builds the
geometry class DD-051 anticipated WOULD trigger: a pillbox cavity
loaded with a thin PEC iris whose small circular aperture is a deep
curved PEC inclusion — the aperture rim cuts transversal H-faces at
grazing angles (→ high floor share) exactly where the coupled
TM010-pair mode concentrates its field (→ energy at the rim).

Observables
-----------
* The two lowest eigenmodes of the coupled two-cell cavity (the
  TM010-like 0-mode and π-mode) at transversal resolutions
  ``N_T_LIST``, staircase vs conformal branch.
* Reference: Richardson extrapolation ``f(h) = f_inf + C·h^p`` per
  branch (grid search over p, linear LSQ for (f_inf, C)).  The same
  extrapolation methodology is validated on the *iris-free* pillbox
  against the analytic TM010 (``f = j01·c0 / (2πR)``) on the same
  mesh family — that anchor calibrates how much the extrapolated
  reference can be trusted.
* Floor share of cat-2 H-faces (total and restricted to the iris
  slab) — trigger condition (a).
* Fraction of magnetic / electric mode energy within ``2h`` of the
  aperture rim circle — trigger condition (c).

Measured verdict (session 91, DD-058)
-------------------------------------
Trigger (a) fires — floor share 70.3–71.8 % (iris slab 81.6–91.7 %) —
and the E-field concentrates at the rim (8.7–28.8 % within 2h), but
trigger (b) does NOT: the conformal branch lands at −0.17…−0.23 %
(staircase +1.0 %) against the cross-branch extrapolated reference,
well inside the 1–2 % line.  The donor mechanism itself is neutral to
machine precision (< 1e-15 relative eigenfrequency shift with ~5000
donated faces): floored cat-2 faces are *Faraday-dead* — their
circulation edges sit inside the PEC mask, so ``C e = 0`` there and h
stays 0 under staircase fallback and donor freeze alike.  The donor
pass (``assign_h_face_donors``) therefore stays dormant in
production; this benchmark is the sentinel to re-run (adapted) on any
future deep-inclusion geometry before wiring it.

Usage::

    <python-in-magnelio-env> validation/iris_cavity_donor_trigger.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from magnelio import AnalysisEigenmode, Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.geo._subcell import _build_A_face_H

C0 = 299_792_458.0
J01 = 2.404825557695773  # first zero of J0

# ---------------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------------

# Radii historically chosen off every forced-grid multiple at all
# N_T_LIST resolutions (h = 1.5/1.0/0.75/0.5 mm from the fixed 24 mm
# bbox): the cylinder tangent planes (±R, ±a) come back from the CSG
# solids with float wiggle, and before the WP-M1 unified plane
# clustering a tangent plane within float distance of — but not
# bit-exactly on — a forced node produced a ~1e-18 m sliver cell whose
# degenerate faces poison M_μ (measured here twice at R = 10 mm /
# a = 3 mm).  WP-M1 snaps critical planes onto forced anchors, so
# R = 10 mm / a = 3 mm now meshes sliver-free at every n_t; the
# off-multiple radii stay to keep the DD-058 recorded numbers
# reproducible.  ``build_mesh`` still guards against regressions.
R = 9.7e-3  # pillbox radius
L_HALF = 8.0e-3  # half-cell length (each side of the iris)
T_IRIS = 2.0e-3  # iris thickness
A_APERTURE = 3.1e-3  # aperture radius (deep PEC inclusion: a << R)
S_BBOX = 24.0e-3
X_PAD = 2.0e-3
L_TOT = 2.0 * L_HALF + T_IRIS

F_TM010 = J01 * C0 / (2.0 * math.pi * R)  # analytic pillbox anchor
F_MAX = 14.0e9

# n_t = 49 (~340k free DOFs) exceeds the SuperLU factorisation
# memory on a 30 GB machine; 4 levels up to n_t = 33 keep the
# Richardson fit over a 2× h-range with tractable eigensolves.
N_T_LIST = (17, 21, 25, 33)


# ---------------------------------------------------------------------------
# Geometry + mesh builders
# ---------------------------------------------------------------------------


def _geometry(with_iris: bool) -> GeometryModel:
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(-X_PAD, -S_BBOX / 2, -S_BBOX / 2),
        size=(L_TOT + 2 * X_PAD, S_BBOX, S_BBOX),
        material=pec,
    )
    cavity = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=R,
        height=L_TOT,
        axis="x",
        material=vacuum,
    )
    model = GeometryModel()
    model.add(Difference(bbox, cavity))
    if not with_iris:
        model.add(cavity)
        return model
    # Volume-disjoint partition (the geometry validator rejects
    # overlapping solids): two vacuum half-cells, the PEC iris ring,
    # and the vacuum aperture channel through it.
    slab = Cylinder(
        origin=(L_HALF, 0.0, 0.0),
        radius=R,
        height=T_IRIS,
        axis="x",
        material=pec,
    )
    hole = Cylinder(
        origin=(L_HALF, 0.0, 0.0),
        radius=A_APERTURE,
        height=T_IRIS,
        axis="x",
        material=vacuum,
    )
    model.add(Difference(cavity, slab))  # both half-cells (vacuum)
    model.add(Difference(slab, hole))  # iris ring (PEC)
    model.add(hole)  # aperture channel (vacuum)
    return model


def _seg(x0: float, x1: float, h: float) -> list[float]:
    n = max(2, int(round((x1 - x0) / h)) + 1)
    return np.linspace(x0, x1, n).tolist()


def build_mesh(
    n_t: int,
    with_iris: bool,
    conformal: bool,
    donors: bool = False,
) -> Mesh:
    h_t = S_BBOX / (n_t - 1)
    y_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, n_t).tolist()
    z_nodes = y_nodes

    # x planes as exact linspace segments snapped to the geometry
    # planes (end walls + iris faces) — accumulated arange would
    # produce sliver cells at the snap planes (session-88 lesson).
    x_nodes: list[float] = []
    breaks = [-X_PAD, 0.0, L_HALF, L_HALF + T_IRIS, L_TOT, L_TOT + X_PAD]
    for x0, x1 in zip(breaks[:-1], breaks[1:]):
        seg = _seg(x0, x1, h_t)
        if x_nodes:
            seg = seg[1:]
        x_nodes.extend(seg)

    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=4.0 * h_t,
        conformal=conformal,
        forced_planes={"x": x_nodes, "y": y_nodes, "z": z_nodes},
    )
    mesh = Mesh.from_geometry(_geometry(with_iris), control, f_max=F_MAX)
    d_min = min(mesh.grid.dx.min(), mesh.grid.dy.min(), mesh.grid.dz.min())
    if d_min < 1e-6:
        raise RuntimeError(
            f"sliver cell detected (d_min = {d_min:.3e} m) — a geometry "
            f"tangent plane collided with a forced grid node; adjust the "
            f"radii/resolution pairing (n_t = {n_t})."
        )
    if donors and mesh.face_material is not None:
        # The donor pass is dormant in production (this benchmark's
        # measured verdict); invoke it explicitly for the conf+donor
        # branch.
        from magnelio._operators.material_matrices import (
            assign_h_face_donors,
        )

        assign_h_face_donors(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Trigger (a): cat-2 H-face floor share
# ---------------------------------------------------------------------------


def _face_positions(grid) -> np.ndarray:
    """Face-centre coordinates for all H-faces, [Hx | Hy | Hz], (n, 3)."""
    x, y, z = grid.x, grid.y, grid.z
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])

    def mesh3(a, b, c):
        A, B, C = np.meshgrid(a, b, c, indexing="ij")
        return np.column_stack([A.ravel(), B.ravel(), C.ravel()])

    return np.concatenate(
        [
            mesh3(x, yc, zc),  # Hx faces (Nx+1, Ny, Nz)
            mesh3(xc, y, zc),  # Hy faces (Nx, Ny+1, Nz)
            mesh3(xc, yc, z),  # Hz faces (Nx, Ny, Nz+1)
        ]
    )


def _edge_positions(grid) -> np.ndarray:
    """Edge-centre coordinates for all E-edges, [Ex | Ey | Ez], (n, 3)."""
    x, y, z = grid.x, grid.y, grid.z
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])

    def mesh3(a, b, c):
        A, B, C = np.meshgrid(a, b, c, indexing="ij")
        return np.column_stack([A.ravel(), B.ravel(), C.ravel()])

    return np.concatenate(
        [
            mesh3(xc, y, z),  # Ex edges (Nx, Ny+1, Nz+1)
            mesh3(x, yc, z),  # Ey edges (Nx+1, Ny, Nz+1)
            mesh3(x, y, zc),  # Ez edges (Nx+1, Ny+1, Nz)
        ]
    )


def floor_share(mesh: Mesh) -> tuple[float, float, int]:
    """Return (total floor share %, iris-slab floor share %, n_cat2)."""
    fm = mesh.face_material
    A_face = _build_A_face_H(mesh.grid)
    cat2 = fm.category == 2
    n_cat2 = int(cat2.sum())
    if n_cat2 == 0:
        return math.nan, math.nan, 0
    floored = cat2 & (fm.A_face_free <= 0.01 * A_face)
    pos = _face_positions(mesh.grid)
    slab = np.abs(pos[:, 0] - (L_HALF + T_IRIS / 2)) <= T_IRIS / 2 + 1e-12
    n_slab = int((cat2 & slab).sum())
    share_total = 100.0 * int(floored.sum()) / n_cat2
    share_slab = 100.0 * int((floored & slab).sum()) / n_slab if n_slab else math.nan
    return share_total, share_slab, n_cat2


# ---------------------------------------------------------------------------
# Trigger (c): mode-energy concentration at the aperture rim
# ---------------------------------------------------------------------------


def rim_energy_fraction(mesh: Mesh, mode, h_t: float) -> tuple[float, float]:
    """Fractions of magnetic / electric mode energy within 2h of the rim.

    Rim = circle (radius ``A_APERTURE``) swept over the iris barrel;
    distance measured as ``sqrt((r_t - a)^2 + max(|x - x_c| - t/2, 0)^2)``.
    Energies are the exact FIT quadratic forms ``0.5·hᵀM_μh`` and
    ``0.5·eᵀM_εe`` restricted to the rim neighbourhood.
    """
    grid = mesh.grid

    def rim_mask(pos):
        r_t = np.hypot(pos[:, 1], pos[:, 2])
        dx = np.maximum(np.abs(pos[:, 0] - (L_HALF + T_IRIS / 2)) - T_IRIS / 2, 0.0)
        return np.hypot(r_t - A_APERTURE, dx) <= 2.0 * h_t

    h_flat = np.concatenate(
        [
            mode.Hx.ravel(),
            mode.Hy.ravel(),
            mode.Hz.ravel(),
        ]
    )
    e_flat = np.concatenate(
        [
            mode.Ex.ravel(),
            mode.Ey.ravel(),
            mode.Ez.ravel(),
        ]
    )
    w_h = build_M_mu(mesh) * h_flat**2
    w_e = build_M_eps(mesh) * e_flat**2

    m_h = rim_mask(_face_positions(grid))
    m_e = rim_mask(_edge_positions(grid))
    return float(w_h[m_h].sum() / w_h.sum()), float(w_e[m_e].sum() / w_e.sum())


# ---------------------------------------------------------------------------
# Trigger (b): eigenfrequency convergence
# ---------------------------------------------------------------------------


def lowest_modes(mesh: Mesh, n: int):
    """Return the n lowest physical eigenfrequencies + their FieldStates."""
    result = AnalysisEigenmode(mesh=mesh, n_modes=n + 3, verbose=False).run()
    phys = np.nonzero(result.frequencies > 1e6)[0]
    order = phys[np.argsort(result.frequencies[phys])][:n]
    return (
        [float(result.frequencies[i]) for i in order],
        [result.modes[i] for i in order],
    )


def richardson(h_arr: np.ndarray, f_arr: np.ndarray):
    """Fit f = f_inf + C·h^p by grid search over p + linear LSQ.

    Returns (f_inf, C, p, rms_residual).
    """
    best = None
    for p in np.linspace(0.5, 3.0, 251):
        X = np.column_stack([np.ones_like(h_arr), h_arr**p])
        coef, res, *_ = np.linalg.lstsq(X, f_arr, rcond=None)
        r = f_arr - X @ coef
        rms = float(np.sqrt(np.mean(r**2)))
        if best is None or rms < best[3]:
            best = (float(coef[0]), float(coef[1]), float(p), rms)
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 76)
    print("WP-R5 trigger benchmark: iris-loaded pillbox cavity")
    print(
        f"  R = {R * 1e3:.1f} mm, half-cell {L_HALF * 1e3:.1f} mm, "
        f"iris t = {T_IRIS * 1e3:.1f} mm, aperture a = {A_APERTURE * 1e3:.1f} mm"
    )
    print(f"  analytic pillbox TM010 anchor: {F_TM010 / 1e9:.6f} GHz")
    print("=" * 76)

    h_list = np.array([S_BBOX / (n_t - 1) for n_t in N_T_LIST])

    branches = (
        ("staircase", False, False),
        ("conformal", True, False),  # pre-donor: staircase floor fallback
        ("conf+donor", True, True),  # WP-R5 H-face enlarged-cell donors
    )

    # ---- Anchor: iris-free pillbox, methodology validation ------------
    print("\n--- Anchor: iris-free pillbox (analytic TM010) ---")
    anchor = {}
    for branch, conformal, donors in branches:
        f_vals = []
        for n_t in N_T_LIST:
            t0 = time.perf_counter()
            mesh = build_mesh(
                n_t,
                with_iris=False,
                conformal=conformal,
                donors=donors,
            )
            freqs, _ = lowest_modes(mesh, 1)
            dt_s = time.perf_counter() - t0
            err = (freqs[0] - F_TM010) / F_TM010
            f_vals.append(freqs[0])
            print(
                f"  {branch:>10s} n_t={n_t:2d}: f = {freqs[0] / 1e9:9.6f} GHz  "
                f"err = {err * 100:+7.3f} %   ({dt_s:5.1f} s)"
            )
        f_inf, _, p, rms = richardson(h_list, np.array(f_vals))
        err_inf = (f_inf - F_TM010) / F_TM010
        anchor[branch] = err_inf
        print(
            f"  {branch:>10s} extrapolated: f_inf = {f_inf / 1e9:9.6f} GHz  "
            f"err = {err_inf * 100:+7.3f} %  (p = {p:.2f}, "
            f"rms = {rms / 1e6:.2f} MHz)"
        )

    # ---- Iris cavity ---------------------------------------------------
    print("\n--- Iris-loaded cavity: TM010 0/π pair ---")
    data: dict[str, dict] = {}
    for branch, conformal, donors in branches:
        f0_vals, fpi_vals = [], []
        for n_t in N_T_LIST:
            t0 = time.perf_counter()
            mesh = build_mesh(
                n_t,
                with_iris=True,
                conformal=conformal,
                donors=donors,
            )
            freqs, modes = lowest_modes(mesh, 2)
            dt_s = time.perf_counter() - t0
            f0_vals.append(freqs[0])
            fpi_vals.append(freqs[1])
            line = (
                f"  {branch:>10s} n_t={n_t:2d}: "
                f"f_0 = {freqs[0] / 1e9:9.6f}  "
                f"f_pi = {freqs[1] / 1e9:9.6f} GHz"
            )
            if conformal and not donors:
                share_tot, share_slab, n_cat2 = floor_share(mesh)
                h_t = S_BBOX / (n_t - 1)
                rim_h0, rim_e0 = rim_energy_fraction(mesh, modes[0], h_t)
                line += (
                    f"  floor {share_tot:5.1f} % "
                    f"(iris slab {share_slab:5.1f} %, n_cat2 {n_cat2}) "
                    f" rim-energy H {rim_h0 * 100:5.1f} % / "
                    f"E {rim_e0 * 100:5.1f} %"
                )
            if donors:
                d = mesh.face_material.enlarged_cell_donor
                n_donated = int((d >= 0).sum()) if d is not None else 0
                line += f"  donated {n_donated}"
            print(line + f"   ({dt_s:5.1f} s)")
        data[branch] = {"f0": np.array(f0_vals), "fpi": np.array(fpi_vals)}

    # ---- Extrapolated references + trigger (b) evaluation --------------
    print("\n--- Richardson extrapolation and branch disagreement ---")
    names = [b[0] for b in branches]
    for key, label in (("f0", "0-mode"), ("fpi", "pi-mode")):
        refs = {}
        for branch in names:
            f_inf, _, p, rms = richardson(h_list, data[branch][key])
            refs[branch] = f_inf
            print(
                f"  {label:>7s} {branch:>10s}: f_inf = {f_inf / 1e9:9.6f} GHz "
                f"(p = {p:.2f}, rms = {rms / 1e6:.2f} MHz)"
            )
        f_ref = float(np.mean([refs[b] for b in names]))
        spread = (max(refs.values()) - min(refs.values())) / f_ref
        print(
            f"  {label:>7s} f_ref (mean of extrapolations) = "
            f"{f_ref / 1e9:9.6f} GHz, branch spread {spread * 100:.3f} %"
        )
        for branch in names:
            err = (data[branch][key][-1] - f_ref) / f_ref
            print(f"  {label:>7s} {branch:>10s} finest-mesh error vs f_ref: {err * 100:+7.3f} %")

    print("\nTrigger reading (DD-051 / STATUS site 2):")
    print("  (a) floor share > 70 %   (b) finest-mesh error > 1-2 %   (c) rim energy concentration")
    print(
        "  -> evaluate the numbers above; all three together are sufficient for the H-face donor."
    )


if __name__ == "__main__":
    main()
