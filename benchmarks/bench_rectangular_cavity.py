"""
Benchmark: Rectangular PEC cavity eigenmodes.

Acceptance criterion: all resonant frequencies within 2% of analytical value.

Output: JSON report written to benchmarks/results/bench_rectangular_cavity.json
"""

import datetime
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D

C0 = 299_792_458.0
TOLERANCE_PCT = 2.0


def analytical_freq(m, n, p, a, b, c):
    return (C0 / 2) * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)


def run_benchmark():
    a, b, c = 30e-3, 20e-3, 15e-3  # mm cavity
    Nx, Ny, Nz = 20, 14, 10

    grid = GridLines(
        x=np.linspace(0, a, Nx + 1),
        y=np.linspace(0, b, Ny + 1),
        z=np.linspace(0, c, Nz + 1),
    )
    air = Material.air()
    material_id = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    pec_mask = np.zeros((3, Nx * (Ny + 1) * (Nz + 1)), dtype=bool)
    mesh = Mesh(
        grid=grid, material_id=material_id, material_library={0: air}, pec_mask_edges=pec_mask
    )

    solver = EigenmodeSolver3D(n_modes=8)
    freq_hz, _ = solver.solve(mesh)

    # Remove static modes
    freq_phys = sorted(freq_hz[freq_hz > 1e6].tolist())[:3]

    analytical = sorted(
        [
            analytical_freq(1, 0, 1, a, b, c),
            analytical_freq(0, 1, 1, a, b, c),
            analytical_freq(1, 1, 0, a, b, c),
        ]
    )

    errors = []
    for f_num, f_ana in zip(freq_phys, analytical):
        err_pct = abs(f_num - f_ana) / f_ana * 100
        errors.append(err_pct)
        print(f"  f_num={f_num / 1e9:.4f} GHz, f_ana={f_ana / 1e9:.4f} GHz, error={err_pct:.2f}%")

    passed = all(e < TOLERANCE_PCT for e in errors)

    report = {
        "case": "rectangular_cavity",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cavity_mm": [a * 1e3, b * 1e3, c * 1e3],
        "grid_cells": [Nx, Ny, Nz],
        "mode_errors_pct": errors,
        "tolerance_pct": TOLERANCE_PCT,
        "passed": passed,
    }

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench_rectangular_cavity.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    print(f"PASSED: {passed}")

    assert passed, f"Benchmark FAILED. Mode errors: {errors}"
    return report


if __name__ == "__main__":
    run_benchmark()
