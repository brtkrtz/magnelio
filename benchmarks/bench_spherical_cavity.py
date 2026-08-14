"""
Benchmark: Spherical PEC cavity eigenmodes.

Acceptance criterion: lowest resonant frequency within 2% of analytical value.

Analytical (TM_011): f_011 = 2.744 * c₀ / (2π · R)
"""

import datetime
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

C0 = 299_792_458.0
TOLERANCE_PCT = 2.0


def analytical_tm011(R: float) -> float:
    """Lowest TM mode of a spherical PEC cavity."""
    # First zero of j_1(x): x_11 = 2.7437...
    x11 = 2.7437
    return C0 * x11 / (2 * math.pi * R)


def run_benchmark():
    print("Spherical cavity benchmark: requires OCC geometry — stub for v1.0 completion")

    R = 15e-3
    f_ana = analytical_tm011(R)
    print(f"  Analytical TM011: {f_ana / 1e9:.4f} GHz")

    report = {
        "case": "spherical_cavity",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "radius_mm": R * 1e3,
        "analytical_tm011_ghz": f_ana / 1e9,
        "passed": None,
        "note": "Stub — requires OCC geometry + full mesher",
    }

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "bench_spherical_cavity.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {out_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
