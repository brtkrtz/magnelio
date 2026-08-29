"""TD wall-loss monitor gates (DD-082, B3).

The monitor's ``dissipated_fraction`` is P_loss/P_flow — both quadratic
in the run's field states, so the global mesh-dependent state scale
cancels (scale-free; no source renormalisation needed).

Measured (session 105):
- parallel plate (alpha = R_s/(eta0*b), copper): fraction/(2*alpha*L)
  = 1.0000..1.0014 over 2-10 GHz (max 0.14 %).
- TE10 waveguide (Pozar alpha_c closed form, 1.4-2.3 fc): max 1.3 % at
  3000 steps (the residual is the half-cell H sampling + staircase
  dispersion, converging O(h^2); 8000 steps: 1.1 %).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    Mesh,
)
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortSpecMultiConductor, PortSpecNumerical
from magnelio.ports._modal.mode import ModeType
from magnelio.post.wall_loss import surface_resistance

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
C0 = 299_792_458.0
ETA0 = np.sqrt(MU0 / EPS0)
SIGMA_CU = 5.8e7


def test_parallel_plate_conductor_loss():
    """Copper-plate TEM line: dissipated fraction vs 2*alpha*L with the
    closed form alpha = R_s/(eta0*b)."""
    w_a, gap_b, length = 10e-3, 5e-3, 30e-3
    freqs = np.linspace(2e9, 10e9, 9)
    grid = GridLines(
        x=np.linspace(0, w_a, 11),
        y=np.linspace(0, gap_b, 6),
        z=np.linspace(0, length, 61),
    )
    mon = MonitorWallLoss(
        freqs=freqs,
        normal="z",
        position=2e-3,
        sigma=SIGMA_CU,
        bc_faces=("ymin", "ymax"),
    )
    ana = AnalysisScatteringTD(
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
        monitors=(mon,),
        verbose=False,
    )
    ana.run(f_axis=freqs, excited=["p1"])

    frac = mon.dissipated_fraction
    alpha = surface_resistance(freqs, SIGMA_CU) / (ETA0 * gap_b)
    ratio = frac["total"] / (2 * alpha * length)
    assert np.abs(ratio - 1).max() < 0.003, f"plate loss ratio {ratio} (measured max 0.0014)"
    # the two plates split the loss exactly evenly
    np.testing.assert_allclose(frac["ymin"], frac["ymax"], rtol=1e-9)
    # power_loss scales linearly with P_in
    p1 = mon.power_loss(P_in=1.0)["total"]
    p2 = mon.power_loss(P_in=2.5)["total"]
    np.testing.assert_allclose(p2, 2.5 * p1, rtol=1e-12)


def test_te10_waveguide_conductor_loss():
    """Copper rectangular waveguide, TE10: dissipated fraction vs the
    closed-form alpha_c (all four walls, band 1.4-2.3 fc)."""
    a, b, length = 20e-3, 10e-3, 40e-3
    fc = C0 / (2 * a)
    freqs = np.linspace(1.4 * fc, 2.3 * fc, 10)
    grid = GridLines(
        x=np.linspace(0, a, 21),
        y=np.linspace(0, b, 11),
        z=np.linspace(0, length, 41),
    )
    mon = MonitorWallLoss(
        freqs=freqs,
        normal="z",
        position=2e-3,
        sigma=SIGMA_CU,
        bc_faces=("xmin", "xmax", "ymin", "ymax"),
    )
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
            },
        ),
        ports=[
            PortSpecNumerical(name="p1", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TE),
            PortSpecNumerical(name="p2", plane=BoxFace.Z_MAX, n_modes=1, mode_type=ModeType.TE),
        ],
        f_max=freqs[-1],
        monitors=(mon,),
        verbose=False,
    )
    # Bounded run: sub-cutoff pulse content decays extremely slowly in a
    # closed waveguide — the energy criterion is the wrong stop here.
    ana.run(f_axis=freqs, excited=["p1"], total_time_steps=3000)

    k = 2 * np.pi * freqs / C0
    beta = np.sqrt(k**2 - (np.pi / a) ** 2)
    Rs = surface_resistance(freqs, SIGMA_CU)
    alpha_c = Rs * (2 * b * np.pi**2 + a**3 * k**2) / (a**3 * b * beta * k * ETA0)
    ratio = mon.dissipated_fraction["total"] / (2 * alpha_c * length)
    assert np.abs(ratio - 1).max() < 0.025, f"TE10 loss ratio {ratio} (measured max 0.013)"


def test_missing_sigma_raises_at_attach():
    grid = GridLines(
        x=np.linspace(0, 1e-2, 6), y=np.linspace(0, 5e-3, 4), z=np.linspace(0, 2e-2, 11)
    )
    mon = MonitorWallLoss(freqs=[1e9], normal="z", position=5e-3, bc_faces=("ymin",))
    with pytest.raises(ValueError, match="no conductivity"):
        mon.attach(Mesh.from_grid(grid))


def test_bad_reference_axis_raises():
    with pytest.raises(ValueError, match="normal must be"):
        MonitorWallLoss(freqs=[1e9], normal="w", position=0.0)
