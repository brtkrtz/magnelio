"""WP7.2 regression: TE10 port quality on a graded transversal grid.

The ``numerical_2d`` TE/TM path shared the pointwise ``h = ±e/η``
H-voltage convention with the TEM/QTEM Laplace solvers; on grids that
are graded along the mode's *E-field direction* (here the WR-90 height
z — TE10's E_z voltage samples then sit on faces of varying area) the
I projection mismeasured the travelling wave exactly like the WP5.3
parallel-plate case.  Grading along the *width* (across the field
family) is NOT exposed: there the legacy calibration's implied test
field coincides with the travelling-wave form because the co-located
face areas stay constant — measured before == after to all digits.

Measured on this fixture (WR-90, 30 mm, z graded growth-1.4, 10 cells,
band 8.5–12 GHz):

    pointwise era:   median |S11| = −20.3 dB, |S21| dev 0.075 dB
    travelling-wave: median |S11| = −26.3 dB, |S21| dev 0.0006 dB

The in-band *max* (−17 dB near cut-off) is the genuine modal-Mur limit
(DD-047) and is excluded from the bound; the median and the |S21|
unitarity deviation are the discriminating regression quantities.
"""

from __future__ import annotations

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal.mode import ModeType

WR90_A = 22.86e-3
WR90_B = 10.16e-3
LENGTH = 30e-3


def _graded_axis(lo: float, hi: float, n_cells: int, growth: float) -> np.ndarray:
    """Symmetric grading: fine at both ends, coarse in the middle."""
    half = n_cells // 2
    d = growth ** np.arange(half, dtype=float)
    if n_cells % 2 == 0:
        d_all = np.concatenate([d, d[::-1]])
    else:
        d_all = np.concatenate([d, [d[-1] * growth], d[::-1]])
    d_all = d_all / d_all.sum() * (hi - lo)
    return lo + np.concatenate([[0.0], np.cumsum(d_all)])


def test_wr90_te10_graded_height():
    """Median |S11| < −24 dB and |S21| within 0.01 dB on graded z."""
    grid = GridLines(
        x=np.linspace(0.0, LENGTH, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=_graded_axis(0.0, WR90_B, 10, 1.4),
    )
    mesh = Mesh.from_grid(grid)
    specs = [
        PortSpecNumerical(name="port1", plane=BoxFace.X_MIN, mode_type=ModeType.TE, n_modes=1),
        PortSpecNumerical(name="port2", plane=BoxFace.X_MAX, mode_type=ModeType.TE, n_modes=1),
    ]
    analysis = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
                "xmin": "PMC",
                "xmax": "PMC",
            }
        ),
        ports=specs,
        f_max=12.4e9,
        f_min=8.2e9,
        verbose=False,
    )
    f_axis = np.linspace(8.5e9, 12.0e9, 36)
    result = analysis.run(f_axis=f_axis, excited=["port1"])

    s11_db = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-30)
    s21_db = 20 * np.log10(np.abs(result.S("port2", "port1")) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    med = float(np.median(s11_db))
    assert med < -24.0, (
        f"graded-z WR-90 TE10 |S11| regression: median in band = "
        f"{med:.2f} dB (bound: -24 dB; pointwise-era median was "
        f"-20.3 dB, uniform baseline ≈ -28 dB)"
    )
    s21_dev = float(np.max(np.abs(s21_db)))
    assert s21_dev < 0.01, (
        f"|S21| deviates from 0 dB by {s21_dev:.4f} dB (bound: 0.01 dB; pointwise era: 0.075 dB)"
    )
