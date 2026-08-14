"""
Benchmark: Plane-wave pulse propagation and timing accuracy.

A Gaussian pulse is injected via the TF/SF PlaneWaveSource (+z, x-polarised)
into a PEC-walled domain.  A 0D field monitor records Ex at the midpoint.

Acceptance criteria:
  1. Probe records a non-trivial peak |Ex| > 1e-4 V/m (wave reached probe).
  2. The peak arrives within ±10 % of the expected propagation delay z/c₀
     (numerical phase velocity ≈ c₀ within FIT dispersion).
"""

import datetime
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

C0 = 299_792_458.0
TIMING_TOLERANCE = 0.10  # ±10 % of propagation delay


def run_benchmark():
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh
    from magnelio.monitors.field_time import MonitorFieldTime
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt
    from magnelio.sources.plane_wave import PlaneWaveSource

    # Uniform grid — chosen so numerical dispersion < 1 % at f_max
    Nx, Ny, Nz = 8, 8, 24
    L_xy, L_z = 8e-3, 24e-3

    grid = GridLines(
        x=np.linspace(0, L_xy, Nx + 1),
        y=np.linspace(0, L_xy, Ny + 1),
        z=np.linspace(0, L_z, Nz + 1),
    )
    mesh = Mesh.from_grid(grid)

    f_max = 10e9
    dt = courant_dt(grid, accuracy="normal")

    # TF/SF box: 2 cells inset from all faces
    x, y, z = grid.x, grid.y, grid.z
    tf_box = ((x[2], y[2], z[2]), (x[Nx - 2], y[Ny - 2], z[Nz - 2]))

    src = PlaneWaveSource(
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        corners=tf_box,
        f_max=f_max,
        waveform="gaussian",
    )

    # Probe at z = L_z/2, inside TF box: a 0D field monitor sampling Ex
    # (x-polarised plane wave) at every time step.
    z_probe = L_z / 2
    p = (L_xy / 2, L_xy / 2, z_probe)
    probe = MonitorFieldTime(corners=(p, p), interval=dt, fields=["Ex"], name="probe_mid")

    # PEC on all 6 faces (standard FIT boundary)
    bcs = {f: PECBoundary(f, grid) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    # Run long enough for the pulse to reach the probe
    t0 = 4.0 / f_max  # Gaussian peak at source z=0
    t_prop = z_probe / C0  # propagation delay to probe
    t_total = t0 + t_prop + 3.0 / f_max
    n_steps = int(math.ceil(t_total / dt)) + 10

    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        sources=[src],
        monitors=[probe],
        total_time_steps=n_steps,
        dt=dt,
        energy_stop_db=None,
        verbose=False,
    )
    solver.run()

    sig = probe.data["Ex"]
    peak_Ex = float(np.max(np.abs(sig)))
    i_peak = int(np.argmax(np.abs(sig)))
    t_peak_measured = float(probe.t[i_peak])

    t_peak_expected = t0 + t_prop
    timing_error = abs(t_peak_measured - t_peak_expected) / t_peak_expected

    print(f"  n_steps={n_steps}, dt={dt * 1e12:.3f} ps, h={L_z / Nz * 1e3:.1f} mm")
    print(f"  Probe peak |Ex|: {peak_Ex:.4e} V/m")
    print(
        f"  Peak arrival: {t_peak_measured * 1e12:.1f} ps  "
        f"(expected {t_peak_expected * 1e12:.1f} ps,  error {timing_error * 100:.1f}%)"
    )

    passed = peak_Ex > 1e-4 and timing_error < TIMING_TOLERANCE
    print(f"PASSED: {passed}")

    report = {
        "case": "waveguide_transmission",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "grid_cells": [Nx, Ny, Nz],
        "n_steps": n_steps,
        "probe_z_mm": z_probe * 1e3,
        "probe_peak_Ex_Vm": peak_Ex,
        "t_peak_measured_ps": t_peak_measured * 1e12,
        "t_peak_expected_ps": t_peak_expected * 1e12,
        "timing_error_pct": timing_error * 100,
        "timing_tolerance_pct": TIMING_TOLERANCE * 100,
        "passed": passed,
    }

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench_waveguide_transmission.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {out_path}")

    assert passed, (
        f"Benchmark FAILED. peak_Ex={peak_Ex:.2e} V/m, "
        f"timing error={timing_error * 100:.1f}% > {TIMING_TOLERANCE * 100:.0f}%"
    )
    return report


if __name__ == "__main__":
    run_benchmark()
