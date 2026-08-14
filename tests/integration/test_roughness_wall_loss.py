"""Roughness end-to-end through the wall-loss chain (DD-088).

Roughness is pure postprocessing: it multiplies R_s by K(f) wherever a
loss model evaluates a wall, so ONE lossless solve serves every
roughness variant.  These gates verify that the multiplier actually
reaches the walls — through both channels (caller-supplied for PEC
walls, material-carried for lossy metals) and in both consumers
(eigenmode Q at f0, monitor DFT bins per frequency).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import (
    AnalysisScatteringTD,
    Material,
)
from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.materials import Hammerstad, Huray
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortSpecMultiConductor
from magnelio.post.wall_loss import surface_resistance, wall_loss_Q

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
C0 = 299_792_458.0
ETA0 = np.sqrt(MU0 / EPS0)

A, B, D = 20e-3, 10e-3, 30e-3
SIGMA_CU = 5.8e7

# Rq = 1 um copper near 9 GHz sits between Hammerstad's limits
# (delta ~ 0.7 um -> K ~ 1.79): neither the smooth nor the saturated end.
ROUGH = Hammerstad(1e-6)


def _q_te101_closed_form(roughness=None) -> float:
    f0 = C0 / 2 * np.sqrt((1 / A) ** 2 + (1 / D) ** 2)
    k = 2 * np.pi * f0 / C0
    Rs = float(surface_resistance(f0, SIGMA_CU, roughness=roughness))
    return (
        (k * A * D) ** 3
        * B
        * ETA0
        / (2 * np.pi**2 * Rs * (2 * A**3 * B + 2 * B * D**3 + A**3 * D + A * D**3))
    )


@pytest.fixture(scope="module")
def _cavity_bc():
    grid = GridLines(
        x=np.linspace(0, A, 21),
        y=np.linspace(0, B, 11),
        z=np.linspace(0, D, 31),
    )
    return AnalysisEigenmode(mesh=Mesh.from_grid(grid), n_modes=1, verbose=False).run()


def test_te101_q_divides_by_k(_cavity_bc):
    """Caller-supplied roughness on PEC BC walls: every wall takes the
    same K(f0), so Q must divide by it exactly."""
    q_smooth = wall_loss_Q(_cavity_bc, sigma=SIGMA_CU)
    q_rough = wall_loss_Q(_cavity_bc, sigma=SIGMA_CU, roughness=ROUGH)

    k = float(ROUGH.factor(q_smooth.frequency, SIGMA_CU))
    assert 1.7 < k < 1.9, f"fixture off its design point: K = {k}"
    assert q_rough.Q == pytest.approx(q_smooth.Q / k, rel=1e-12)
    assert q_rough.P_loss == pytest.approx(q_smooth.P_loss * k, rel=1e-12)
    # Losses only — the mode itself is untouched.
    assert q_rough.frequency == q_smooth.frequency
    assert q_rough.W == q_smooth.W
    # ... and it reaches every wall tag, not just the total.
    for tag in q_smooth.per_tag:
        assert q_rough.per_tag[tag] == pytest.approx(
            q_smooth.per_tag[tag] * k,
            rel=1e-12,
        )


def test_te101_q_rough_vs_closed_form(_cavity_bc):
    """The rough Q against the closed form evaluated with K*R_s — the
    same 1 % envelope the smooth gate holds."""
    q = wall_loss_Q(_cavity_bc, sigma=SIGMA_CU, roughness=ROUGH)
    assert q.Q == pytest.approx(_q_te101_closed_form(ROUGH), rel=0.01)


def test_material_carried_roughness_reaches_the_walls():
    """A lossy metal brings its own roughness: the cavity as an air
    volume in a rough-copper shell against the K-scaled closed form."""
    h = 1e-3
    grid = GridLines(
        x=np.linspace(-h, A + h, 23),
        y=np.linspace(-h, B + h, 13),
        z=np.linspace(-h, D + h, 33),
    )
    cu = Material.lossy_metal("copper", sigma=SIGMA_CU, roughness=ROUGH)
    mesh = Mesh.from_grid(
        grid,
        regions=[(Material.air(), (0, 0, 0, A, B, D))],
        background=cu,
    )
    res = AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()

    # No sigma=, no roughness=: both come from the material.
    q = wall_loss_Q(res)
    assert q.Q == pytest.approx(_q_te101_closed_form(ROUGH), rel=0.01)
    (tag,) = q.per_tag
    assert mesh.material_library[tag].roughness == ROUGH


def test_plate_line_alpha_with_roughness():
    """Copper parallel plate, one solve and two monitors: the rough
    fraction is the smooth one shaped by K(f) per DFT bin, and it tracks
    the closed form alpha = K*R_s/(eta0*b)."""
    w_a, gap_b, length = 10e-3, 5e-3, 30e-3
    freqs = np.linspace(2e9, 10e9, 9)
    # A PCB-grade foil: K rises 1.33 -> 2.11 across the band, so a
    # constant multiplier could not pass the per-bin comparison.
    rough = Huray.cannonball(4.443e-6)

    grid = GridLines(
        x=np.linspace(0, w_a, 11),
        y=np.linspace(0, gap_b, 6),
        z=np.linspace(0, length, 61),
    )
    common = dict(reference_plane=("z", 2e-3), sigma=SIGMA_CU, bc_faces=("ymin", "ymax"))
    mon_smooth = MonitorWallLoss(freqs=freqs, name="smooth", **common)
    mon_rough = MonitorWallLoss(freqs=freqs, name="rough", roughness=rough, **common)
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
        monitors=(mon_smooth, mon_rough),
        verbose=False,
    )
    ana.run(f_axis=freqs, excited=["p1"])

    k = rough.factor(freqs, SIGMA_CU)
    assert k.min() < 1.4 and 1.9 < k.max() < 2.3, f"fixture off its design point: K = {k}"
    # Same H bins, only R_s differs -> the ratio IS K(f), bin by bin.
    f_smooth = mon_smooth.dissipated_fraction["total"]
    f_rough = mon_rough.dissipated_fraction["total"]
    np.testing.assert_allclose(f_rough / f_smooth, k, rtol=1e-12)

    # ... and the rough fraction still matches the closed form.
    alpha = k * surface_resistance(freqs, SIGMA_CU) / (ETA0 * gap_b)
    ratio = f_rough / (2 * alpha * length)
    assert np.abs(ratio - 1).max() < 0.003, (
        f"rough plate loss ratio {ratio} (smooth gate measures 0.0014)"
    )
