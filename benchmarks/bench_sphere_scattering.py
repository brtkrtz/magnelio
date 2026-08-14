"""
Benchmark: Plane-wave scattering from a dielectric/PEC sphere (Mie theory).

Acceptance criterion: Radar Cross Section (RCS) error < 2 dB vs Mie series.
"""

import datetime
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

TOLERANCE_DB = 2.0


def mie_rcs_pec_sphere(R: float, freq: float) -> float:
    """Approximate RCS of a PEC sphere (Rayleigh + Mie for ka ~ 1).

    This is a placeholder. Full Mie series requires summing partial waves.
    """
    c0 = 299_792_458.0
    k = 2 * math.pi * freq / c0
    ka = k * R
    if ka < 0.1:
        # Rayleigh regime: RCS = (4π/3)² · k⁴ · R⁶ ... simplified
        return 4 * math.pi * R**2 * (ka**4)
    else:
        # Geometric optics approximation: RCS ≈ π R²
        return math.pi * R**2


def run_benchmark():
    print("Sphere scattering benchmark: requires TF/SF source + OCC sphere — stub for v1.0")

    R = 10e-3
    freq = 5e9
    rcs_ana = mie_rcs_pec_sphere(R, freq)
    print(f"  Approximate Mie RCS at {freq / 1e9:.1f} GHz: {10 * math.log10(rcs_ana):.2f} dBm²")

    report = {
        "case": "sphere_scattering",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sphere_radius_mm": R * 1e3,
        "freq_ghz": freq / 1e9,
        "rcs_analytical_dbm2": 10 * math.log10(rcs_ana),
        "tolerance_db": TOLERANCE_DB,
        "passed": None,
        "note": "Stub — requires TF/SF plane wave + OCC sphere geometry",
    }

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench_sphere_scattering.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {out_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
