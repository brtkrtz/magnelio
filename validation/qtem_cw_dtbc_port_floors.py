"""WP-R4a acceptance: CW true-mode port floors on inhomogeneous lines.

The QTEM acceptance criterion (developer decision 2026-07-09):
``|S11| < -100 dB`` by CW lock-in through the *production* solver
with the true discrete mode per frequency — profile, zeta and
discrete V/I response from the same eigenproblem — full band for the
QTEM fundamental, from ``1.01 * f_c_hat`` for higher hybrid modes.

Measurement chain per frequency point (all production API):
``build_cw_true_mode_port`` (zeta-pencil channels + frequency-local
DTBC + exact phasors) -> ``FITTimeDomainSolver`` CW run with a
ramped monochromatic drive -> ``PortSignalRecorder`` ->
``cw_lockin_phasors`` -> ``cw_decompose`` -> ``S11 = b/a`` at the
excited port.  ``|S21|`` (``b_2/a_1``) is printed as a convention
anchor (~0 dB on a matched uniform line).

Geometries (the two WP-R4 spike cases as real 3D production meshes,
plus a microstrip-class section):

* layered   — parallel plate, lower half eps_r = 4 (PEC y-walls,
              natural x-walls), 1 mm grid, 48 mm line.
* block     — 2D-inhomogeneous: eps_r = 4 block over the lower-left
              quadrant of a 4 x 5 mm cross-section, 48 mm line.
* microstrip — shielded microstrip, eps_r = 4.3 substrate
              (h = 0.8 mm), 1.6 mm strip (0.2 mm thick PEC), 8 x 4.8
              mm box, 40 mm line; conductors auto-detected from the
              mesh PEC mask.

Cost-watch deliverable (developer caveat 2026-07-09): the measured
per-frequency mode-solve cost (``op.cw_data.solve_seconds`` — period
blocks + sparse shift-invert eigensolve + fit + phasors) against the
3D CW run time, plus scaling on a refined microstrip cross-section.

Results (session 88), CW lock-in |S11| (|S21| = 0.00 dB throughout):

    layered     1.0-7.8 GHz fundamental:   -244.6 .. -196.5 dB
    layered     2nd mode (f_c_hat 8.4465 GHz),
                1.01/1.05/1.2 f_c_hat:     -176.3 / -194.6 / -200.6 dB
    block       2.1-6.2 GHz fundamental:   -250.2 .. -225.2 dB
    microstrip  1.0-7.8 GHz fundamental:   -250.8 .. -206.5 dB

All 76-150 dB below the -100 dB line.  Cost-watch: mode solve
41 / 31 / 433 ms per port vs 3D runs of 2.7 / 1.9 / 196 s per point
(share ~3 % on the toy lines whose 3D run lasts seconds, 0.9 % on
the production-sized microstrip); cross-section scaling N = 2 710 /
11 132 / 45 112 -> 86 ms / 0.49 s / 3.6 s (ARPACK-dominated).

Run:  python validation/qtem_cw_dtbc_port_floors.py
      [--case layered|block|microstrip|all] [--fast] [--cost]
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
from scipy.special import erf

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    build_cw_true_mode_port,
    cw_decompose,
    cw_lockin_phasors,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

# ----------------------------------------------------------------------
# Geometries
# ----------------------------------------------------------------------


def _forced(vals):
    return [float(v) for v in vals]


def _segments(*breaks_and_counts):
    """Grid raster through exact geometry planes.

    ``_segments((x0, x1, n1), (x1, x2, n2), ...)`` concatenates
    ``linspace`` segments; the segment endpoints are set bit-exactly,
    so every geometry snap plane coincides with a raster node (a
    float-accumulated ``arange`` misses them by ~1e-19 m and the
    mesher then inserts sliver cells that collapse the CFL dt).
    """
    out = []
    for lo, hi, n in breaks_and_counts:
        seg = np.linspace(lo, hi, n + 1)
        out.extend(seg if not out else seg[1:])
    return _forced(out)


def layered_mesh(nz=48):
    w, hy, h_if, dz = 10.0e-3, 8.0e-3, 4.0e-3, 1.0e-3
    length = nz * dz
    diel = Material(name="diel", epsilon=(4.0,) * 3)
    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(w, h_if, length), material=diel))
    model.add(Brick(origin=(0, h_if, 0), size=(w, hy - h_if, length), material=Material.air()))
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=5.1e-3,
        forced_planes={
            "x": _segments((0.0, w, 2)),
            "y": _segments((0.0, h_if, 4), (h_if, hy, 4)),
            "z": _segments((0.0, length, nz)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    mesh = mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(mesh.grid, "normal")


def block_mesh(nz=48):
    wx, hy, dz = 4.0e-3, 5.0e-3, 1.0e-3
    length = nz * dz
    diel = Material(name="diel", epsilon=(4.0,) * 3)
    block = Brick(origin=(0, 0, 0), size=(2.0e-3, 2.0e-3, length), material=diel)
    air_box = Brick(origin=(0, 0, 0), size=(wx, hy, length), material=Material.air())
    model = GeometryModel()
    model.add(Difference(air_box, block))
    model.add(block)
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=1.1e-3,
        forced_planes={
            "x": _segments((0.0, 2.0e-3, 2), (2.0e-3, wx, 2)),
            "y": _segments((0.0, 2.0e-3, 2), (2.0e-3, hy, 3)),
            "z": _segments((0.0, length, nz)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    mesh = mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(mesh.grid, "normal")


def microstrip_mesh(nz=40, d_t=0.2e-3):
    w_strip, t_strip, h_sub = 1.6e-3, 0.2e-3, 0.8e-3
    w_box, h_box, dz = 8.0e-3, 4.8e-3, 1.0e-3
    length = nz * dz
    sub = Material(name="substrate", epsilon=(4.3,) * 3)
    strip = Brick(
        origin=(-w_strip / 2, h_sub, 0), size=(w_strip, t_strip, length), material=Material.pec()
    )
    air_cap = Brick(
        origin=(-w_box / 2, h_sub, 0), size=(w_box, h_box - h_sub, length), material=Material.air()
    )
    model = GeometryModel()
    model.add(Brick(origin=(-w_box / 2, 0, 0), size=(w_box, h_sub, length), material=sub))
    model.add(Difference(air_cap, strip))
    model.add(strip)

    def n_of(a, b):
        return max(int(round((b - a) / d_t)), 1)

    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=2.0 * d_t,
        forced_planes={
            "x": _segments(
                (-w_box / 2, -w_strip / 2, n_of(-w_box / 2, -w_strip / 2)),
                (-w_strip / 2, w_strip / 2, n_of(-w_strip / 2, w_strip / 2)),
                (w_strip / 2, w_box / 2, n_of(w_strip / 2, w_box / 2)),
            ),
            "y": _segments(
                (0.0, h_sub, n_of(0.0, h_sub)),
                (h_sub, h_sub + t_strip, n_of(h_sub, h_sub + t_strip)),
                (h_sub + t_strip, h_box, n_of(h_sub + t_strip, h_box)),
            ),
            "z": _segments((0.0, length, nz)),
        },
    )
    mesh = Mesh.from_geometry(model, control, f_max=8.0e9)
    mesh = mesh.with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )
    return mesh, courant_dt(mesh.grid, "normal")


# ----------------------------------------------------------------------
# CW lock-in measurement through the production solver
# ----------------------------------------------------------------------


def cw_point(mesh, dt, f, exc_channel=0, n_channels=None, verbose_channels=False):
    """One CW frequency point: build ports, run, decompose."""
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    spec1 = PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=None)
    spec2 = PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=None)
    op1 = build_cw_true_mode_port(spec1, mesh, m_eps, m_mu, dt=dt, f_cw=f, n_channels=n_channels)
    op2 = build_cw_true_mode_port(spec2, mesh, m_eps, m_mu, dt=dt, f_cw=f, n_channels=n_channels)
    ch = op1.cw_data.channels[exc_channel]
    assert op1.termination_kinds[exc_channel] == "dtbc"

    w_dt = op1.cw_data.w_dt
    period = 2.0 * math.pi / w_dt
    theta = abs(np.angle(ch.zeta))
    v_g = ch.r**2 * math.sin(theta) / math.sin(w_dt)
    nz = mesh.Nz
    # Near a channel cut-off the ramp must narrow spectrally with the
    # gap to the cut-off, or its below-cut-off leakage never drains
    # (the R3 band-edge lock-in lesson).
    gap = w_dt - 2.0 * math.asin(min(ch.q / 2.0, 1.0))
    sigma = max(8.0 / max(gap, 1e-12), 6.0 * period)
    n_win = int(40 * period)
    n_meas0 = int(10.0 * sigma + 20.0 * period + 3.0 * nz / max(v_g, 1e-3))
    n_steps = n_meas0 + n_win + 2

    t0 = 5.0 * sigma * dt
    sig_t = sigma * dt
    w_phys = w_dt / dt

    def waveform(t: float) -> float:
        amp = 0.5 * (1.0 + float(erf((t - t0) / (math.sqrt(2.0) * sig_t))))
        return amp * math.sin(w_phys * t)

    op1.set_excitation(exc_channel, waveform)
    recorder = PortSignalRecorder(dt=dt, ports=[op1, op2])
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        ports=[op1, op2],
        recorder=recorder,
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    t_run0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*neither a BoundaryCondition.*",
        )
        solver.run()
    t_run = time.perf_counter() - t_run0
    signals = recorder.finalize(n_steps_actual=n_steps)

    V1, I1 = signals[("port1", exc_channel)]
    Vp, Ip, res_fit = cw_lockin_phasors(V1.values, I1.values, w_dt, n_win)
    a1, b1 = cw_decompose(Vp, Ip, ch)
    V2, I2 = signals[("port2", exc_channel)]
    Vp2, Ip2, _ = cw_lockin_phasors(V2.values, I2.values, w_dt, n_win)
    a2, b2 = cw_decompose(Vp2, Ip2, op2.cw_data.channels[exc_channel])
    s11 = abs(b1 / a1)
    s21 = abs(b2 / a1)
    info = dict(
        s11_db=20.0 * math.log10(max(s11, 1e-300)),
        s21_db=20.0 * math.log10(max(s21, 1e-300)),
        res_fit=res_fit,
        n_steps=n_steps,
        t_run=t_run,
        solve_seconds=op1.cw_data.solve_seconds,
        n_ch=len(op1.cw_data.channels),
        eps_eff=ch.eps_eff_hat,
        r=ch.r,
        q=ch.q,
    )
    if verbose_channels:
        for c, cc in enumerate(op1.cw_data.channels):
            print(
                f"        ch{c}: zeta {cc.zeta:.6f}  r {cc.r:.4f}"
                f"  q^2 {cc.q**2:+.3e}"
                f"  eps_eff_hat {cc.eps_eff_hat:.4f}"
            )
    return info


def fundamental_sweep(name, mesh, dt, f_list):
    print(f"  {name} — fundamental, CW lock-in (criterion: < -100 dB full band)")
    costs = []
    for f in f_list:
        info = cw_point(mesh, dt, f)
        costs.append((info["solve_seconds"], info["t_run"]))
        print(
            f"    f {f / 1e9:6.2f} GHz  |S11| {info['s11_db']:8.1f} dB"
            f"   |S21| {info['s21_db']:6.2f} dB"
            f"   eps_eff_hat {info['eps_eff']:.4f}"
            f"   ({info['n_ch']} ch, {info['n_steps']} steps,"
            f" fit-res {info['res_fit']:.1e})"
        )
    return costs


def higher_mode_points(name, mesh, dt, f_lo, f_hi, ratios):
    """Locate the second cut-on by bisection, measure channel 1."""
    from magnelio.ports._modal.port_plane import PortPlane
    from magnelio.ports._modal.zeta_pencil import (
        build_period_blocks,
        find_propagating_modes,
    )

    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    from magnelio._operators.material_matrices import (
        flatten_port_plane_mass,
        flatten_port_plane_pec_mask,
    )

    m_eps_f = flatten_port_plane_mass(m_eps, mesh, BoxFace.Z_MIN)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, BoxFace.Z_MIN),
    )
    plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
    c_3d = build_curl_matrix(mesh.grid)
    chain = build_period_blocks(plane, mesh, m_eps_f, m_mu, c_3d, dt)

    def n_prop(f):
        w_dt = 2.0 * math.pi * f * dt
        theta_hint = 2.0 * math.pi * f * 2.0 / 3e8 * plane.normal_dx
        zs, _ = find_propagating_modes(chain, w_dt, theta_hint)
        return zs.size

    lo, hi = f_lo, f_hi
    if n_prop(lo) >= 2 or n_prop(hi) < 2:
        print(f"    (cut-on not bracketed in [{lo / 1e9:.1f}, {hi / 1e9:.1f}] GHz — skipping)")
        return
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_prop(mid) >= 2:
            hi = mid
        else:
            lo = mid
    f_c2 = hi
    print(
        f"  {name} — second mode, discrete cut-on "
        f"f_c_hat = {f_c2 / 1e9:.4f} GHz "
        f"(criterion: < -100 dB from 1.01*f_c_hat)"
    )
    for ratio in ratios:
        f = ratio * f_c2
        info = cw_point(mesh, dt, f, exc_channel=1, n_channels=2)
        print(
            f"    f/f_c_hat {ratio:5.3f}  |S11| "
            f"{info['s11_db']:8.1f} dB   |S21| {info['s21_db']:6.2f}"
            f" dB   ({info['n_steps']} steps,"
            f" fit-res {info['res_fit']:.1e})"
        )


# ----------------------------------------------------------------------
# Cost-watch
# ----------------------------------------------------------------------


def cost_watch(name, mesh, dt, costs):
    solve = np.array([c[0] for c in costs])
    run = np.array([c[1] for c in costs])
    share = 100.0 * solve / run
    print(
        f"  {name}: mode-solve {solve.mean() * 1e3:7.1f} ms/point "
        f"(min {solve.min() * 1e3:.1f} / max {solve.max() * 1e3:.1f})"
        f"   3D run {run.mean():6.1f} s/point"
        f"   share {share.mean():.2f} %"
    )


def cost_scaling():
    from magnelio._operators.material_matrices import (
        flatten_port_plane_mass,
        flatten_port_plane_pec_mask,
    )
    from magnelio.ports._modal.port_plane import PortPlane
    from magnelio.ports._modal.zeta_pencil import (
        build_period_blocks,
        solve_zeta_modes,
    )

    print("  microstrip cross-section refinement (mode solve only, 4.2 GHz):")
    for d_t in (0.2e-3, 0.1e-3, 0.05e-3):
        mesh, dt = microstrip_mesh(nz=8, d_t=d_t)
        m_eps = flatten_port_plane_mass(build_M_eps(mesh), mesh, BoxFace.Z_MIN)
        object.__setattr__(
            mesh,
            "pec_mask_edges",
            flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, BoxFace.Z_MIN),
        )
        m_mu = build_M_mu(mesh)
        plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
        c_3d = build_curl_matrix(mesh.grid)
        t0 = time.perf_counter()
        chain = build_period_blocks(plane, mesh, m_eps, m_mu, c_3d, dt)
        t_blocks = time.perf_counter() - t0
        w_dt = 2.0 * math.pi * 4.2e9 * dt
        t0 = time.perf_counter()
        solve_zeta_modes(chain, w_dt, [0.99 * np.exp(-0.1j)])
        t_solve = time.perf_counter() - t0
        n = chain.D_0.shape[0]
        print(
            f"    d_t {d_t * 1e3:5.2f} mm  N {n:6d}"
            f"   blocks {t_blocks * 1e3:7.1f} ms"
            f"   eigensolve {t_solve * 1e3:7.1f} ms"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["layered", "block", "microstrip", "all"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--cost", action="store_true")
    args = ap.parse_args()
    cases = ["layered", "block", "microstrip"] if args.case == "all" else [args.case]

    print("WP-R4a acceptance — CW true-mode port floors (|S11| < -100 dB, production solver):")
    for name in cases:
        if name == "layered":
            mesh, dt = layered_mesh()
            f_list = [4.2e9] if args.fast else [1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.8e9]
            costs = fundamental_sweep("layered", mesh, dt, f_list)
            cost_watch("layered", mesh, dt, costs)
            if not args.fast:
                mesh, dt = layered_mesh()
                higher_mode_points("layered", mesh, dt, 5.0e9, 13.0e9, (1.01, 1.05, 1.2))
        elif name == "block":
            mesh, dt = block_mesh()
            f_list = [4.2e9] if args.fast else [2.1e9, 4.2e9, 6.2e9]
            costs = fundamental_sweep("block", mesh, dt, f_list)
            cost_watch("block", mesh, dt, costs)
        else:
            mesh, dt = microstrip_mesh()
            f_list = [4.2e9] if args.fast else [1.0e9, 2.1e9, 4.2e9, 6.2e9, 7.8e9]
            costs = fundamental_sweep("microstrip", mesh, dt, f_list)
            cost_watch("microstrip", mesh, dt, costs)
    if args.cost:
        print()
        print("Cost-watch scaling (developer caveat 2026-07-09):")
        cost_scaling()


if __name__ == "__main__":
    main()
