#!/usr/bin/env python
"""Convergence benchmark: rotated rectangular PEC cavity eigenfrequency.

Meshes a PEC rectangular cavity (air brick rotated 30 deg around x and y)
at several resolutions and compares staircase vs conformal vs conformal+DM
eigenfrequency accuracy.

Outputs:
  - Table of results (resolution, frequency, error, timing)
  - Log-log convergence plot with fitted order
  - Saves results to rotated_cavity_results/ as .npz and .png

Analytical reference:
    Fundamental mode (1,1,0) of rectangular PEC cavity a x b x c:
        f = (c0/2) * sqrt((1/a)^2 + (1/b)^2)
    with a=40 mm, b=30 mm, c=20 mm  =>  f ~ 6.2457 GHz.

    The rotation does not affect the eigenfrequency — only the
    mesh approximation quality, which is what this benchmark probes.
"""

import math
import time
from pathlib import Path

import numpy as np

from magnelio import AnalysisEigenmode, Material, Mesh, MeshControl
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.geo.transforms import rotate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C0 = 299_792_458.0

# Cavity dimensions [m] — all different for a non-degenerate fundamental mode
A = 40e-3  # 40 mm
B = 30e-3  # 30 mm
D = 20e-3  # 20 mm  (avoid shadowing speed-of-light C)

# Rotation: 30 deg around x, then 30 deg around y
ROT_X_DEG = 30.0
ROT_Y_DEG = 30.0

# Analytical fundamental mode (1,1,0)
F_ANALYTICAL = (C0 / 2) * math.sqrt((1 / A) ** 2 + (1 / B) ** 2)

CELL_SIZES = [4e-3, 3e-3, 2.5e-3, 2e-3, 1.5e-3, 1.25e-3]
CONFIGS = {
    "staircase": {"conformal": False, "dey_mittra_eta": 0.0},
    "conformal": {"conformal": True, "dey_mittra_eta": 0.0},
    "conformal+DM": {"conformal": True, "dey_mittra_eta": 0.4},
}

OUTPUT_DIR = Path(__file__).parent / "rotated_cavity_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_cavity_mesh(
    h: float,
    conformal: bool,
    dey_mittra_eta: float,
) -> Mesh:
    """Build a rotated rectangular PEC cavity mesh at resolution *h*.

    Geometry: PEC outer shell with an air brick (rotated 30 deg / 30 deg)
    carved out via CSG Difference.  Padding >= 5 cells between cavity
    surface and domain boundary (avoids KB-003).
    """
    pec = Material.pec()
    air = Material.air()

    # Air cavity centred at origin, then rotated
    cavity = Brick(
        origin=(-A / 2, -B / 2, -D / 2),
        size=(A, B, D),
        material=air,
    )
    cavity = rotate(cavity, (1, 0, 0), ROT_X_DEG)
    cavity = rotate(cavity, (0, 1, 0), ROT_Y_DEG)

    # PEC outer box with padding
    bb_lo, bb_hi = cavity.bounding_box()
    pad = max(5 * h, 10e-3)
    outer = Brick(
        origin=(bb_lo[0] - pad, bb_lo[1] - pad, bb_lo[2] - pad),
        size=(
            bb_hi[0] - bb_lo[0] + 2 * pad,
            bb_hi[1] - bb_lo[1] + 2 * pad,
            bb_hi[2] - bb_lo[2] + 2 * pad,
        ),
        material=pec,
    )

    model = GeometryModel()
    model.add(Difference(outer, cavity))
    model.add(cavity)

    ctrl = MeshControl(
        min_nodes_per_wavelength=4,
        max_cell_size=h,
        conformal=conformal,
        dey_mittra_eta=dey_mittra_eta,
    )
    return Mesh.from_geometry(model, ctrl, f_max=2 * F_ANALYTICAL)


def find_fundamental(mesh: Mesh) -> float:
    """Run eigenmode solver and return the fundamental frequency [Hz].

    Filters out near-DC null-space modes (f < 1 MHz) and returns the
    lowest physical eigenfrequency.
    """
    result = AnalysisEigenmode(mesh=mesh, n_modes=5, verbose=False).run()
    f_phys = np.sort(result.frequencies[result.frequencies > 1e6])
    if len(f_phys) < 1:
        return np.nan
    return float(f_phys[0])


def fit_convergence_order(h_arr: np.ndarray, err_arr: np.ndarray):
    """Fit p in |error| ~ h^p via least-squares on log-log data."""
    valid = err_arr > 0
    if valid.sum() < 2:
        return np.nan, np.nan
    log_h = np.log(h_arr[valid])
    log_e = np.log(err_arr[valid])
    p, log_C = np.polyfit(log_h, log_e, 1)
    return p, np.exp(log_C)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Rotated Rectangular PEC Cavity Convergence Benchmark")
    print(f"Cavity: {A * 1e3:.0f} x {B * 1e3:.0f} x {D * 1e3:.0f} mm")
    print(f"Rotation: {ROT_X_DEG} deg around x, {ROT_Y_DEG} deg around y")
    print(f"f_analytical (mode 1,1,0) = {F_ANALYTICAL / 1e9:.6f} GHz")
    print(f"Cell sizes: {[h * 1e3 for h in CELL_SIZES]} mm")
    print(f"Configurations: {list(CONFIGS.keys())}")
    print()

    results = {}
    for config_name, config in CONFIGS.items():
        print(f"--- {config_name} ---")
        freqs = []
        errors = []
        times_mesh = []
        times_solve = []
        grid_sizes = []

        for h in CELL_SIZES:
            # Mesh
            t0 = time.perf_counter()
            mesh = build_cavity_mesh(h, **config)
            t_mesh = time.perf_counter() - t0

            Nx, Ny, Nz = mesh.grid.Nx, mesh.grid.Ny, mesh.grid.Nz
            n_cells = Nx * Ny * Nz

            # Solve
            t0 = time.perf_counter()
            f_num = find_fundamental(mesh)
            t_solve = time.perf_counter() - t0

            err = abs(f_num - F_ANALYTICAL) / F_ANALYTICAL
            freqs.append(f_num)
            errors.append(err)
            times_mesh.append(t_mesh)
            times_solve.append(t_solve)
            grid_sizes.append((Nx, Ny, Nz))

            print(
                f"  h={h * 1e3:5.2f}mm  "
                f"grid={Nx:3d}x{Ny:3d}x{Nz:3d} ({n_cells:7d} cells)  "
                f"f={f_num / 1e9:8.5f} GHz  "
                f"err={err * 100:6.2f}%  "
                f"t_mesh={t_mesh:5.1f}s  t_solve={t_solve:5.1f}s"
            )

        h_arr = np.array(CELL_SIZES)
        err_arr = np.array(errors)
        p, C_fit = fit_convergence_order(h_arr, err_arr)
        print(f"  Convergence order: p = {p:.2f}")
        print()

        results[config_name] = {
            "h": h_arr,
            "freq": np.array(freqs),
            "error": err_arr,
            "t_mesh": np.array(times_mesh),
            "t_solve": np.array(times_solve),
            "grid_sizes": grid_sizes,
            "conv_order": p,
            "conv_C": C_fit,
        }

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_DIR / "results.npz",
        h=np.array(CELL_SIZES),
        f_analytical=F_ANALYTICAL,
        **{f"{k}_freq": v["freq"] for k, v in results.items()},
        **{f"{k}_error": v["error"] for k, v in results.items()},
        **{f"{k}_t_mesh": v["t_mesh"] for k, v in results.items()},
        **{f"{k}_t_solve": v["t_solve"] for k, v in results.items()},
    )
    print(f"Results saved to {OUTPUT_DIR / 'results.npz'}")

    # --- Plot ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        markers = {"staircase": "s", "conformal": "o", "conformal+DM": "D"}
        colors = {
            "staircase": "#d62728",
            "conformal": "#1f77b4",
            "conformal+DM": "#2ca02c",
        }

        for name, data in results.items():
            h_mm = data["h"] * 1e3
            err_pct = data["error"] * 100
            p = data["conv_order"]
            ax1.loglog(
                h_mm,
                err_pct,
                marker=markers[name],
                color=colors[name],
                label=f"{name} (p={p:.1f})",
                linewidth=1.5,
                markersize=7,
            )
            if not np.isnan(p):
                h_fit = np.linspace(h_mm.min() * 0.8, h_mm.max() * 1.2, 50)
                err_fit = data["conv_C"] * (h_fit * 1e-3) ** p * 100
                ax1.loglog(h_fit, err_fit, "--", color=colors[name], alpha=0.4)

        ax1.set_xlabel("Cell size h [mm]")
        ax1.set_ylabel("Relative frequency error [%]")
        ax1.set_title(
            f"Rotated PEC Cavity "
            f"({A * 1e3:.0f}x{B * 1e3:.0f}x{D * 1e3:.0f} mm, "
            f"{ROT_X_DEG}/{ROT_Y_DEG} deg)"
        )
        ax1.legend()
        ax1.grid(True, which="both", alpha=0.3)

        for name, data in results.items():
            n_cells = [nx * ny * nz for nx, ny, nz in data["grid_sizes"]]
            ax2.plot(
                n_cells,
                data["t_solve"],
                marker=markers[name],
                color=colors[name],
                label=name,
                linewidth=1.5,
                markersize=7,
            )
        ax2.set_xlabel("Number of cells")
        ax2.set_ylabel("Eigenmode solve time [s]")
        ax2.set_title("Solver Timing")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
        print(f"Plot saved to {OUTPUT_DIR / 'convergence.png'}")

    except ImportError:
        print("matplotlib not available -- skipping plot.")


if __name__ == "__main__":
    main()
