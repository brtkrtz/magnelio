"""Regression test for BC-PEC consolidation on a parallel-plate line (WP2.2, finding F3).

A 20 mm parallel-plate line defined *without any geometric PEC object*:
the plates are pure boundary conditions (``ymin``/``ymax`` = PEC), the
open sides are PMC, and the mesh is a bare vacuum ``Mesh.from_grid``.
The TEM MultiConductor ports on the z-faces auto-derive their conductor
groups from ``mesh.pec_mask_edges`` — which is empty unless
``AnalysisScatteringTD`` consolidates the BC-PEC faces into the mesh
via :meth:`Mesh.with_pec_boundaries` (DD-050 PEC equivalence).

Before WP2.2 this setup raised ``ValueError: port plane on 'z_min' has
no PEC edges``; the notebooks worked around it with explicit helper PEC
bricks.  The same step also materialises string-valued BC dict entries
(``{"ymin": "PEC"}``) into runtime BC instances — previously a silent
no-op in the solver loop.
"""

from __future__ import annotations

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor

WIDTH_A = 10e-3
GAP_B = 5e-3
LENGTH = 20e-3
F_MAX = 10e9


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


def _build_analysis(*, graded_y: bool = False) -> AnalysisScatteringTD:
    y = (
        _graded_axis(-GAP_B / 2, GAP_B / 2, 12, 1.4)
        if graded_y
        else np.linspace(-GAP_B / 2, GAP_B / 2, 6)
    )
    grid = GridLines(
        x=np.linspace(-WIDTH_A / 2, WIDTH_A / 2, 11),
        y=y,
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 41),
    )
    mesh = Mesh.from_grid(grid)

    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=specs,
        f_max=F_MAX,
        verbose=False,
    )


def test_parallel_plate_bc_pec_only():
    """|S11| stays below −60 dB across 0.25–10 GHz, |S21| flat at 0 dB.

    Measured after BC-PEC consolidation (uniform 11×6×41 vacuum mesh,
    QTEM auto-conductor ports): max in-band |S11| = −71.9 dB, |S21|
    within ±0.009 dB.

    WP-R2 (exact DTBC termination + discrete de-stagger): max in-band
    |S11| = −138.7 dB, median −164.0 dB.  The bound guards the
    −100 dB reflection-free acceptance criterion with ~20 dB margin.
    """
    analysis = _build_analysis()
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    result = analysis.run(f_axis=f_axis, excited=["port1"])

    S11 = result.S("port1", "port1")
    S21 = result.S("port2", "port1")
    s11_db = 20 * np.log10(np.abs(S11) + 1e-30)
    s21_db = 20 * np.log10(np.abs(S21) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    assert s11_db.max() < -120.0, (
        f"parallel-plate |S11| regression: max in band = "
        f"{s11_db.max():.2f} dB at f = "
        f"{f_axis[np.argmax(s11_db)] / 1e9:.2f} GHz (bound: -120 dB; "
        f"reflection-free acceptance line is -100 dB)"
    )
    assert np.max(np.abs(s21_db)) < 0.1, (
        f"|S21| deviates from 0 dB by {np.max(np.abs(s21_db)):.3f} dB (bound: 0.1 dB)"
    )


def test_parallel_plate_graded_transversal():
    """WP7.2 regression: graded transversal grid keeps the |S11| floor.

    Growth-1.4 symmetric grading of the y (gap) axis — 12 cells, aspect
    ~2.7:1 across the cross-section.  With the pointwise ``h = ±e/η``
    H-voltage convention this floor sat at −21.4 dB (the WP5.3 grading
    finding: a pure V/I *measurement* error, V/I(TW) = 223 Ω against
    z_line = 188 Ω).  With the travelling-wave-consistent H profiles
    (per-face ``1/M_μ`` weights + direct M-metric calibration) the
    measured floor was −64.1 dB max / −75.7 dB median.

    WP-R2 (exact DTBC termination + discrete de-stagger): max in-band
    |S11| = −136.1 dB, median −158.1 dB — the transversal-grading
    penalty is gone entirely.  The bound guards the −100 dB
    reflection-free acceptance criterion with ~15 dB margin.
    """
    analysis = _build_analysis(graded_y=True)
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    result = analysis.run(f_axis=f_axis, excited=["port1"])

    s11_db = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-30)
    s21_db = 20 * np.log10(np.abs(result.S("port2", "port1")) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    assert s11_db.max() < -115.0, (
        f"graded parallel-plate |S11| regression: max in band = "
        f"{s11_db.max():.2f} dB at f = "
        f"{f_axis[np.argmax(s11_db)] / 1e9:.2f} GHz (bound: -115 dB; "
        f"reflection-free acceptance line is -100 dB)"
    )
    assert np.max(np.abs(s21_db)) < 0.1, (
        f"|S21| deviates from 0 dB by {np.max(np.abs(s21_db)):.3f} dB (bound: 0.1 dB)"
    )
