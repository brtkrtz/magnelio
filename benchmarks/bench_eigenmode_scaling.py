#!/usr/bin/env python
"""Benchmark: 3D eigenmode solver scaling (ARPACK vs matrix-free shift-invert).

Measures solve time, peak RAM, and eigenfrequency error vs. analytical
for an all-PEC rectangular cavity across increasing grid sizes.

Usage:
    mamba run --no-capture-output -n mio python \
        benchmarks/bench_eigenmode_scaling.py
"""

import math
import time
import tracemalloc

import numpy as np

from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D

C0 = 299_792_458.0

# Cavity dimensions (distinct lengths → non-degenerate modes)
A, B, C_DIM = 30e-3, 20e-3, 15e-3

# Analytical modes for this cavity
ANALYTICAL_MODES = sorted(
    [
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 1),
    ]
)


def f_analytical(m, n, p):
    return (C0 / 2) * math.sqrt((m / A) ** 2 + (n / B) ** 2 + (p / C_DIM) ** 2)


ANALYTICAL_FREQS = np.array([f_analytical(*mode) for mode in ANALYTICAL_MODES])


def run_benchmark(N, n_modes=5, solver_backend=None):
    """Run eigenmode solve on an N^3 grid. Return dict with results."""
    Nx = N
    Ny = max(2, round(N * B / A))
    Nz = max(2, round(N * C_DIM / A))
    n_cells = Nx * Ny * Nz

    grid = GridLines(
        x=np.linspace(0, A, Nx + 1),
        y=np.linspace(0, B, Ny + 1),
        z=np.linspace(0, C_DIM, Nz + 1),
    )
    mesh = Mesh.from_grid(grid)

    solver = EigenmodeSolver3D(n_modes=n_modes, solver=solver_backend)

    tracemalloc.start()
    t0 = time.perf_counter()
    freq_hz, E_modes, H_modes = solver.solve(mesh)
    t_solve = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    n_E = n_Ex + n_Ey + n_Ez

    freq_phys = sorted(freq_hz[freq_hz > 1e6].tolist())
    n_compare = min(len(freq_phys), len(ANALYTICAL_FREQS))
    if n_compare > 0:
        errors = [
            abs(freq_phys[i] - ANALYTICAL_FREQS[i]) / ANALYTICAL_FREQS[i] * 100
            for i in range(n_compare)
        ]
        mean_err = np.mean(errors)
        max_err = np.max(errors)
    else:
        mean_err = max_err = float("nan")

    return {
        "N": N,
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "cells": n_cells,
        "n_E": n_E,
        "n_modes_found": len(freq_phys),
        "solve_s": t_solve,
        "peak_MB": peak_bytes / 1e6,
        "mean_err_pct": mean_err,
        "max_err_pct": max_err,
    }


def _print_header(title):
    print(f"\n=== {title} ===")
    print(f"Cavity: {A * 1e3:.0f} x {B * 1e3:.0f} x {C_DIM * 1e3:.0f} mm, all-PEC, n_modes=5")
    print()
    print(
        f"{'Grid':>12s} | {'Cells':>10s} | {'n_E DOFs':>10s} | "
        f"{'Solve [s]':>10s} | {'Peak RAM':>10s} | "
        f"{'f_err mean':>10s} | {'f_err max':>10s}"
    )
    print("-" * 88)


def _print_row(r):
    print(
        f"{r['Nx']}x{r['Ny']}x{r['Nz']:>3d} | "
        f"{r['cells']:>10,d} | {r['n_E']:>10,d} | "
        f"{r['solve_s']:>10.2f} | {r['peak_MB']:>8.0f} MB | "
        f"{r['mean_err_pct']:>9.1f}% | {r['max_err_pct']:>9.1f}%"
    )


def main():
    arpack_sizes = [10, 20, 30, 40, 50]
    lobpcg_sizes = [10, 20, 30, 40, 50, 70, 100]

    # ARPACK baseline
    _print_header("ARPACK eigsh (shift-invert, SuperLU)")
    arpack_results = {}
    for N in arpack_sizes:
        try:
            r = run_benchmark(N, solver_backend="arpack")
            _print_row(r)
            arpack_results[N] = r
        except Exception as e:
            print(f"{'N=' + str(N):>12s} | FAILED: {e}")

    # Matrix-free shift-invert
    _print_header("Matrix-free shift-invert (GMRES + Jacobi preconditioner)")
    lobpcg_results = {}
    for N in lobpcg_sizes:
        try:
            r = run_benchmark(N, solver_backend="lobpcg")
            _print_row(r)
            lobpcg_results[N] = r
        except Exception as e:
            print(f"{'N=' + str(N):>12s} | FAILED: {e}")

    # Comparison
    common = sorted(set(arpack_results) & set(lobpcg_results))
    if common:
        print("\n=== Comparison: Matrix-free vs. ARPACK ===")
        print(
            f"{'Grid':>12s} | {'ARPACK [s]':>10s} | {'MF-SI [s]':>10s} | "
            f"{'Speedup':>8s} | {'ARPACK RAM':>10s} | {'MF-SI RAM':>10s} | "
            f"{'RAM saved':>9s}"
        )
        print("-" * 85)
        for N in common:
            a = arpack_results[N]
            l = lobpcg_results[N]
            speedup = a["solve_s"] / l["solve_s"] if l["solve_s"] > 0 else float("inf")
            ram_ratio = 1.0 - l["peak_MB"] / a["peak_MB"] if a["peak_MB"] > 0 else 0.0
            print(
                f"{a['Nx']}x{a['Ny']}x{a['Nz']:>3d} | "
                f"{a['solve_s']:>10.2f} | {l['solve_s']:>10.2f} | "
                f"{speedup:>7.1f}x | "
                f"{a['peak_MB']:>8.0f} MB | {l['peak_MB']:>8.0f} MB | "
                f"{ram_ratio * 100:>7.0f}%"
            )

    print()
    print("Analytical reference frequencies:")
    for mode, freq in zip(ANALYTICAL_MODES, ANALYTICAL_FREQS):
        print(f"  TE/TM{mode} = {freq / 1e9:.4f} GHz")


if __name__ == "__main__":
    main()
