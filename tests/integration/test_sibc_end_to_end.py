"""End-to-end high-level-API SIBC gates (WP-D5).

Copper-class parallel-plate TEM line (BC plates, PMC side walls) with
``wall_model="sibc"``: the field solution itself carries the conductor
loss, so |S21| shows the closed-form attenuation ``alpha =
R_s/(eta0*b)`` directly — the self-consistency the perturbative chain
cannot give.

Measured (session 115, sigma = 5.8e3 S/m, 2-10 GHz, 10x5x60 cells):

* ``-ln|S21| / (alpha*L)`` = 0.981 … 0.997 across the band (the O(h)
  residual class of the DD-082 chain; the TEM plate has no H-position
  bias, so the residual is fit error + discrete dispersion).
* |S11| = -38 … -50 dB — exactly the DERIVATION.md §7 recorded
  limitation: the DTBC port is exact for the LOSSLESS chain, and
  uniform wall damping perturbs the true mode by O(Z_s/eta0)
  (2.6/377 = -43 dB class).  Physical-line reflections of interest sit
  far above this floor.
* The SIBC-accounted ``MonitorWallLoss`` fraction exceeds the
  power-wave balance ``1-|S21|^2-|S11|^2`` by the discrete/physical
  metric ratio of the uniform-H_x rim strip: PMC side walls make H_x
  uniform up to the x rims, where the state convention books FULL end
  cells (WP-D3 record) — (Nx+1)/Nx = 1.10 on this grid, measured
  1.10-1.11.  A closed PEC cross-section has H_tan -> 0 at the rims
  and no such strip; a finer grid shrinks it O(h).
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortSpecMultiConductor
from magnelio.post.wall_loss import surface_resistance

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
ETA0 = np.sqrt(MU0 / EPS0)
SIGMA = 5.8e3  # scaled down from copper so alpha*L >> DFT ripple

W_A, GAP_B, LENGTH = 10e-3, 5e-3, 30e-3
FREQS = np.linspace(2e9, 10e9, 9)


def _analysis(wall_model, monitors=()):
    grid = GridLines(
        x=np.linspace(0, W_A, 11),
        y=np.linspace(0, GAP_B, 6),
        z=np.linspace(0, LENGTH, 61),
    )
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecMultiConductor(name="p1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="p2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=10e9,
        f_min=2e9,
        monitors=monitors,
        verbose=False,
        wall_model=wall_model,
        wall_sigma=SIGMA if wall_model == "sibc" else None,
    )


def test_parallel_plate_sibc_alpha_and_monitor():
    """|S21| carries the closed-form conductor attenuation; the wall
    monitor reports the SIBC's own extraction (windows measured above)."""
    mon = MonitorWallLoss(
        freqs=FREQS,
        reference_plane=("z", 2e-3),
        sigma=SIGMA,
        bc_faces=("ymin", "ymax"),
    )
    ana = _analysis("sibc", monitors=(mon,))
    res = ana.run(f_axis=FREQS, excited=["p1"])
    s21 = np.abs(res.S("p2", "p1"))
    s11 = np.abs(res.S("p1", "p1"))

    alpha = surface_resistance(FREQS, SIGMA) / (ETA0 * GAP_B)
    ratio = -np.log(s21) / (alpha * LENGTH)
    # Window re-measured under the spectral time step (dt moved 1.29 ->
    # 1.31 ps): the interior of the band stays in the 0.988-0.999 class,
    # but the 2 GHz band edge — where the excitation carries the least
    # energy and alpha*L is smallest — sits at a converged 0.9607
    # (longer runs do not move it).  The physics anchor is the
    # dt-parameterised SIBC unit layer (test_surface_impedance,
    # test_sibc_operator); this end-to-end window only needs to catch a
    # broken chain, not re-measure the fit.
    assert ratio.min() > 0.95 and ratio.max() < 1.005, f"alpha ratio {ratio} (measured 0.96-0.999)"

    # Lossy-wall port floor: the recorded DERIVATION.md §7 limitation —
    # O(Z_s/eta0) mismatch class, far below physical reflections.
    assert (20 * np.log10(s11) < -35).all()

    # Monitor runs the SIBC accounting (spec wired by the analysis) and
    # tracks the power-wave balance up to the rim-strip metric ratio.
    assert mon.sibc is ana._sibc_spec()
    balance = 1.0 - s21**2 - s11**2
    ratio_mon = mon.dissipated_fraction["total"] / balance
    # Ceiling re-measured under the spectral time step: the 2 GHz
    # band-edge point sits at 1.139 (same least-excitation-energy
    # sensitivity as the alpha window above); the interior stays at
    # the rim-strip 1.10-1.11 class.
    assert ratio_mon.min() > 1.0 and ratio_mon.max() < 1.15, (
        f"monitor/balance {ratio_mon} (measured 1.10-1.14, rim strip 11/10)"
    )
    # both plates split the loss evenly
    frac = mon.dissipated_fraction
    np.testing.assert_allclose(frac["ymin"], frac["ymax"], rtol=1e-6)


def test_perturbative_default_is_lossless_in_field_solve():
    """Suite guard: the default wall model leaves the field solve
    lossless — |S21| stays at the DTBC 0 dB class, far above the SIBC
    run's attenuation."""
    res = _analysis("perturbative").run(f_axis=FREQS, excited=["p1"])
    s21 = np.abs(res.S("p2", "p1"))
    assert (np.abs(s21 - 1.0) < 5e-3).all()
