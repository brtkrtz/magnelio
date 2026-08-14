"""Unit tests for PEC wall-surface enumeration (DD-082, B1)."""

import numpy as np
import pytest

from magnelio.materials.material import Material
from magnelio.mesh._surfaces import enumerate_pec_surfaces
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh

D = 1e-3


def _grid(nx=8, ny=6, nz=5):
    return GridLines(x=np.arange(nx + 1) * D, y=np.arange(ny + 1) * D, z=np.arange(nz + 1) * D)


class TestSolidWalls:
    def test_brick_surface_area(self):
        """A free-standing PEC brick exposes its exact 6-face area."""
        pec = Material.lossy_metal("cu", sigma=5.8e7)
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 1 * D, 6 * D, 5 * D, 3 * D))],
        )
        surfs = enumerate_pec_surfaces(mesh)
        assert len(surfs) == 1
        bx, by, bz = 4 * D, 3 * D, 2 * D
        assert surfs[0].area == pytest.approx(2 * (bx * by + bx * bz + by * bz))
        assert surfs[0].tag == 1

    def test_weights_tile_each_component_once(self):
        """Per tangential component the weights sum to the full area."""
        pec = Material.lossy_metal("cu", sigma=5.8e7)
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 1 * D, 6 * D, 5 * D, 3 * D))],
        )
        surf = enumerate_pec_surfaces(mesh)[0]
        assert surf.weight.sum() == pytest.approx(2 * surf.area)

    def test_brick_flush_at_domain_boundary_skipped(self):
        """A solid face lying ON the domain boundary has no interior air
        cell — only the 5 exposed faces are enumerated."""
        pec = Material.pec()
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 0.0, 6 * D, 5 * D, 2 * D))],
        )
        surf = enumerate_pec_surfaces(mesh)[0]
        bx, by, bz = 4 * D, 3 * D, 2 * D
        assert surf.area == pytest.approx(
            bx * by + 2 * (bx * bz + by * bz)  # top + 4 sides, no bottom
        )

    def test_two_metals_tagged_separately(self):
        m1 = Material.lossy_metal("cu", sigma=5.8e7)
        m2 = Material.lossy_metal("steel", sigma=1.4e6, mu=100.0)
        mesh = Mesh.from_grid(
            _grid(),
            regions=[
                (m1, (1 * D, 1 * D, 1 * D, 3 * D, 3 * D, 3 * D)),
                (m2, (5 * D, 1 * D, 1 * D, 7 * D, 3 * D, 3 * D)),
            ],
        )
        surfs = enumerate_pec_surfaces(mesh)
        assert len(surfs) == 2
        cube = 6 * (2 * D) ** 2
        for s in surfs:
            assert s.area == pytest.approx(cube)
        assert {s.tag for s in surfs} == {1, 2}

    def test_touching_same_metal_faces_not_counted(self):
        """The shared face between two touching bricks of one metal is
        interior — not part of the exposed surface."""
        m = Material.pec()
        mesh = Mesh.from_grid(
            _grid(),
            regions=[
                (m, (1 * D, 1 * D, 1 * D, 3 * D, 3 * D, 3 * D)),
                (m, (3 * D, 1 * D, 1 * D, 5 * D, 3 * D, 3 * D)),
            ],
        )
        surf = enumerate_pec_surfaces(mesh)[0]
        # merged 4x2x2 brick
        bx, by, bz = 4 * D, 2 * D, 2 * D
        assert surf.area == pytest.approx(2 * (bx * by + bx * bz + by * bz))


class TestBoundaryWalls:
    def test_domain_wall_area(self):
        mesh = Mesh.from_grid(_grid())
        surfs = enumerate_pec_surfaces(mesh, bc_pec_faces=("zmin", "xmax"))
        areas = {s.tag: s.area for s in surfs}
        assert areas["zmin"] == pytest.approx(8 * D * 6 * D)
        assert areas["xmax"] == pytest.approx(6 * D * 5 * D)

    def test_metal_flush_on_wall_not_double_counted(self):
        """Where a PEC solid touches a PEC domain wall, the wall face is
        inside metal and must not contribute wall samples."""
        pec = Material.pec()
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 0.0, 6 * D, 5 * D, 2 * D))],
        )
        surfs = enumerate_pec_surfaces(mesh, bc_pec_faces=("zmin",))
        areas = {s.tag: s.area for s in surfs}
        assert areas["zmin"] == pytest.approx(8 * D * 6 * D - 4 * D * 3 * D)

    def test_unknown_face_raises(self):
        mesh = Mesh.from_grid(_grid())
        with pytest.raises(ValueError, match="unknown boundary face"):
            enumerate_pec_surfaces(mesh, bc_pec_faces=("bottom",))


class TestSampleGeometry:
    def test_inv_l_dual_values(self):
        """1/l_dual per sample follows the component's own axis (graded).

        Boundary entries use the SOLVER state convention (full first/
        last cell, matching material_matrices._build_avg_d) — that is
        the length the boundary H states carry.
        """
        x = np.array([0.0, 1.0, 2.5, 4.5]) * 1e-3
        grid = GridLines(x=x, y=np.arange(4) * D, z=np.arange(4) * D)
        mesh = Mesh.from_grid(grid)
        surf = [s for s in enumerate_pec_surfaces(mesh, bc_pec_faces=("zmin",))][0]
        dx = np.diff(x)
        dx_dual = np.array([dx[0], (dx[0] + dx[1]) / 2, (dx[1] + dx[2]) / 2, dx[2]])
        hx = surf.comp == 0
        idx = np.unravel_index(surf.flat_idx[hx], (grid.Nx + 1, grid.Ny, grid.Nz))
        np.testing.assert_allclose(
            surf.inv_l_dual[hx],
            1.0 / dx_dual[idx[0]],
            rtol=1e-14,
        )

    def test_h_tan_sq_sum_uniform_field(self):
        """With H = const the weighted sum equals |H|^2 * area per
        tangential component (state = H * l_dual)."""
        mesh = Mesh.from_grid(_grid())
        surf = enumerate_pec_surfaces(mesh, bc_pec_faces=("zmin",))[0]
        Nx, Ny, Nz = mesh.grid.Nx, mesh.grid.Ny, mesh.grid.Nz
        H0 = 3.0
        dx_dual = np.full(Nx + 1, D)  # state convention: full cell at ends
        Hx = np.broadcast_to((H0 * dx_dual)[:, None, None], (Nx + 1, Ny, Nz)).copy()
        Hy = np.zeros((Nx, Ny + 1, Nz))
        Hz = np.zeros((Nx, Ny, Nz + 1))
        total = surf.h_tan_sq_sum(Hx, Hy, Hz)
        assert total == pytest.approx(H0**2 * surf.area)
