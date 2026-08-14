"""Eigenmode wall-loss Q gates (DD-082, B2).

Rectangular-cavity TE101 against the closed form (Pozar Eq. 6.46,
cross-checked in-session against an independent explicit surface
integration of the analytic mode — both give Q = 7568.9 for the copper
20x10x30 mm cavity).

Measured (session 105): BC-wall cavity Q = 7585.3 at 1 mm cells
(+0.22 %), 7573.0 at 0.5 mm (+0.05 %) — O(h^2) convergence; the
material-wall (lossy-metal shell) cavity reproduces the BC-wall Q to
5 digits on the identical interior grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.post.wall_loss import surface_resistance, wall_loss_Q

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6
C0 = 299_792_458.0
ETA0 = np.sqrt(MU0 / EPS0)

A, B, D = 20e-3, 10e-3, 30e-3
SIGMA_CU = 5.8e7


def _q_te101_closed_form() -> float:
    f0 = C0 / 2 * np.sqrt((1 / A) ** 2 + (1 / D) ** 2)
    k = 2 * np.pi * f0 / C0
    Rs = float(surface_resistance(f0, SIGMA_CU))
    return (
        (k * A * D) ** 3
        * B
        * ETA0
        / (2 * np.pi**2 * Rs * (2 * A**3 * B + 2 * B * D**3 + A**3 * D + A * D**3))
    )


def test_te101_q_bc_walls():
    """Domain-BC PEC walls: Q vs closed form within 1 %."""
    grid = GridLines(
        x=np.linspace(0, A, 21),
        y=np.linspace(0, B, 11),
        z=np.linspace(0, D, 31),
    )
    mesh = Mesh.from_grid(grid)
    res = AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()
    q = wall_loss_Q(res, sigma=SIGMA_CU)

    q_ref = _q_te101_closed_form()
    assert q.frequency == pytest.approx(9.0076e9, rel=2e-3)
    assert q.Q == pytest.approx(q_ref, rel=0.01), (
        f"TE101 Q = {q.Q:.1f} vs closed form {q_ref:.1f} (measured +0.22 %)"
    )
    # symmetric per-tag breakdown, 1/Q additivity
    assert q.Q_of("xmin") == pytest.approx(q.Q_of("xmax"), rel=1e-6)
    assert q.Q_of("zmin") == pytest.approx(q.Q_of("zmax"), rel=1e-6)
    inv_sum = sum(1.0 / q.Q_of(t) for t in q.per_tag)
    assert 1.0 / q.Q == pytest.approx(inv_sum, rel=1e-12)


def test_te101_q_material_walls_match_bc_walls():
    """The identical cavity built as an air volume inside a lossy-metal
    shell (material-PEC surface path, sigma from the material) gives the
    same Q as the BC-wall run."""
    h = 1e-3
    grid_bc = GridLines(
        x=np.linspace(0, A, 21),
        y=np.linspace(0, B, 11),
        z=np.linspace(0, D, 31),
    )
    res_bc = AnalysisEigenmode(mesh=Mesh.from_grid(grid_bc), n_modes=1, verbose=False).run()
    q_bc = wall_loss_Q(res_bc, sigma=SIGMA_CU)

    grid_mat = GridLines(
        x=np.linspace(-h, A + h, 23),
        y=np.linspace(-h, B + h, 13),
        z=np.linspace(-h, D + h, 33),
    )
    cu = Material.lossy_metal("copper", sigma=SIGMA_CU)
    mesh = Mesh.from_grid(
        grid_mat,
        regions=[(Material.air(), (0, 0, 0, A, B, D))],
        background=cu,
    )
    res = AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()
    q_mat = wall_loss_Q(res)  # sigma comes from the lossy metal

    assert q_mat.frequency == pytest.approx(q_bc.frequency, rel=1e-4)
    assert q_mat.Q == pytest.approx(q_bc.Q, rel=1e-4)
    # exactly one wall tag: the copper shell's material id
    (tag,) = q_mat.per_tag
    assert mesh.material_library[tag].is_lossy_metal


def test_plain_pec_wall_without_sigma_raises():
    grid = GridLines(
        x=np.linspace(0, A, 11),
        y=np.linspace(0, B, 6),
        z=np.linspace(0, D, 16),
    )
    res = AnalysisEigenmode(mesh=Mesh.from_grid(grid), n_modes=1, verbose=False).run()
    with pytest.raises(ValueError, match="no conductivity"):
        wall_loss_Q(res)
