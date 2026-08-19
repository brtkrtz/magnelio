#!/usr/bin/env python
"""Benchmark: cost of TF/SF plane-wave injection vs. total-field box size.

The plane-wave source corrects the six faces of its total-field box after
every E and every H update.  Its cost therefore follows the box *surface*,
not the cell count of the domain — a box wrapped around the whole model
costs far more per step than a box hugging the scatterer, on the same grid.

Reports milliseconds per time step against a source-free baseline on the
same mesh, and normalises the difference by the number of box boundary
cells so the two backends can be compared directly.

Usage:
    mamba run --no-capture-output -n mio python \
        benchmarks/bench_plane_wave_tfsf.py [--backend numpy|cupy] [--cells 1.0]
"""

# Reproduces the measurement table in design-decisions.md DD-177.

import argparse
import time

import numpy as np

from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt
from magnelio.sources.plane_wave import PlaneWaveSource

# Box extents as a fraction of the domain, plus the full-domain box.
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
N_STEPS = 40


def build(n_cells_M: float):
    """Cubic vacuum domain of roughly *n_cells_M* million cells."""
    n = int(round((n_cells_M * 1e6) ** (1 / 3)))
    lines = np.linspace(0.0, 1e-2, n + 1)
    grid = GridLines(x=lines, y=lines, z=lines)
    return Mesh.from_grid(grid), grid


def source_for(grid, fraction: float, f_max: float):
    """A +z plane wave whose TF box covers *fraction* of the domain."""
    if fraction >= 1.0:
        corners = None  # default extent: two cells inside each face
    else:
        half = fraction / 2
        span = np.asarray(grid.x)[-1]
        lo, hi = (0.5 - half) * span, (0.5 + half) * span
        corners = ((lo, lo, lo), (hi, hi, hi))
    return PlaneWaveSource(
        direction=(0, 0, 1),
        polarization=(1, 0, 0),
        corners=corners,
        waveform="gaussian",
        f_max=f_max,
    )


def time_run(mesh, grid, dt, sources, backend: str) -> float:
    """Milliseconds per time step."""
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        sources=sources,
        total_time_steps=N_STEPS,
        dt=dt,
        verbose=False,
        backend=backend,
    )
    t0 = time.perf_counter()
    solver.run()
    return (time.perf_counter() - t0) / N_STEPS * 1e3


def boundary_cells(src) -> int:
    """Cells on the six faces of the snapped TF/SF box."""
    ix0, ix1, iy0, iy1, iz0, iz1 = src._box
    nx, ny, nz = ix1 - ix0, iy1 - iy0, iz1 - iz0
    return 2 * (nx * ny + ny * nz + nx * nz)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--cells", type=float, default=1.0, help="domain size [10^6 cells]")
    args = ap.parse_args()

    mesh, grid = build(args.cells)
    dt = courant_dt(grid, accuracy="normal")
    f_max = 0.2 * 299_792_458.0 / float(np.diff(np.asarray(grid.x)).max())

    n_cells = mesh.Nx * mesh.Ny * mesh.Nz
    print(
        f"backend {args.backend}   grid {mesh.Nx}x{mesh.Ny}x{mesh.Nz} = {n_cells / 1e6:.2f} M cells"
    )
    base = time_run(mesh, grid, dt, [], args.backend)
    print(f"no source: {base:.2f} ms/step\n")
    print(f"{'box':>8} {'ms/step':>9} {'injection':>10} {'face cells':>11} {'us/cell':>9}")
    for fraction in FRACTIONS:
        src = source_for(grid, fraction, f_max)
        total = time_run(mesh, grid, dt, [src], args.backend)
        n_face = boundary_cells(src)
        cost = total - base
        print(
            f"{fraction:8.2f} {total:9.2f} {cost:9.2f} {n_face:11d} "
            f"{cost * 1e3 / max(n_face, 1):9.3f}"
        )


if __name__ == "__main__":
    main()
