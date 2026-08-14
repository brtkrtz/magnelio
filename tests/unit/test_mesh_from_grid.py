"""Unit tests for Mesh.from_grid(boundary_conditions=_BC_OPEN),
build_pec_mask_faces(), and apply_thin_pec_sheet()."""

import numpy as np

from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import apply_thin_pec_sheet
from magnelio.mesh.mesher import Mesh

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}


def _grid(Nx=8, Ny=8, Nz=8, L=8e-3):
    return GridLines(
        x=np.linspace(0, L, Nx + 1),
        y=np.linspace(0, L, Ny + 1),
        z=np.linspace(0, L, Nz + 1),
    )


class TestMeshFromGrid:
    def test_creates_mesh(self):
        mesh = Mesh.from_grid(_grid(), boundary_conditions=_BC_OPEN)
        assert mesh.Nx == 8
        assert mesh.material_id.shape == (8, 8, 8)

    def test_default_background_is_air(self):
        mesh = Mesh.from_grid(_grid(), boundary_conditions=_BC_OPEN)
        assert 0 in mesh.material_library
        assert mesh.material_library[0].name == "air"
        assert np.all(mesh.material_id == 0)

    def test_single_material_region(self):
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        grid = _grid(Nx=10, Ny=10, Nz=10, L=10e-3)
        mesh = Mesh.from_grid(
            grid, regions=[(fr4, (0, 0, 0, 10e-3, 10e-3, 3e-3))], boundary_conditions=_BC_OPEN
        )

        # First 3 z-layers should be FR4
        assert mesh.material_id[5, 5, 0] != 0  # inside FR4 region
        assert mesh.material_id[5, 5, 5] == 0  # outside FR4 region

    def test_overlapping_regions_later_wins(self):
        mat_a = Material(name="A", epsilon=(2.0, 2.0, 2.0))
        mat_b = Material(name="B", epsilon=(4.0, 4.0, 4.0))
        grid = _grid(Nx=10, Ny=10, Nz=10, L=10e-3)
        mesh = Mesh.from_grid(
            grid,
            regions=[
                (mat_a, (0, 0, 0, 10e-3, 10e-3, 10e-3)),  # whole domain
                (mat_b, (3e-3, 3e-3, 3e-3, 7e-3, 7e-3, 7e-3)),  # inner cube
            ],
            boundary_conditions=_BC_OPEN,
        )
        id_b = None
        for mid, mat in mesh.material_library.items():
            if mat.name == "B":
                id_b = mid
        assert id_b is not None
        # Center cell should be B
        assert mesh.material_id[5, 5, 5] == id_b

    def test_custom_background(self):
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        mesh = Mesh.from_grid(_grid(), background=fr4, boundary_conditions=_BC_OPEN)
        assert mesh.material_library[0].name == "FR4"

    def test_pec_region_marks_edges(self):
        pec = Material.pec()
        grid = _grid(Nx=6, Ny=6, Nz=6, L=6e-3)
        mesh = Mesh.from_grid(
            grid,
            regions=[(pec, (1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3))],
            boundary_conditions=_BC_OPEN,
        )
        assert mesh.pec_mask_edges.sum() > 0

    def test_no_pec_material_all_false(self):
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        mesh = Mesh.from_grid(
            _grid(), regions=[(fr4, (0, 0, 0, 8e-3, 8e-3, 8e-3))], boundary_conditions=_BC_OPEN
        )
        assert not mesh.pec_mask_edges.any()


class TestApplyThinPecSheet:
    """Tests for apply_thin_pec_sheet() — DD-017 edge-mask model."""

    def _make_mesh(self, Nx=8, Ny=8, Nz=8, L=8e-3):
        grid = GridLines(
            x=np.linspace(0, L, Nx + 1),
            y=np.linspace(0, L, Ny + 1),
            z=np.linspace(0, L, Nz + 1),
        )
        return Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)

    def test_ex_edges_at_sheet_are_pec(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        # Place sheet at y = grid.y[4] (middle node)
        j_h = 4
        y_pos = grid.y[j_h]
        x_min, x_max = grid.x[2], grid.x[6]
        z_min, z_max = grid.z[0], grid.z[-1]
        apply_thin_pec_sheet(mesh, axis="y", position=y_pos, rect=(x_min, z_min, x_max, z_max))

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        x_c = 0.5 * (grid.x[:-1] + grid.x[1:])
        # Check that all Ex edges in the trace region at j_h are PEC
        for i in range(Nx):
            if x_c[i] < x_min or x_c[i] > x_max:
                continue
            for k in range(Nz + 1):
                flat = i * (Ny + 1) * (Nz + 1) + j_h * (Nz + 1) + k
                assert mesh.pec_mask_edges[0, flat], f"Ex[{i},{j_h},{k}] should be PEC"

    def test_ez_edges_at_sheet_are_pec(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        j_h = 4
        y_pos = grid.y[j_h]
        x_min, x_max = grid.x[2], grid.x[6]
        z_min, z_max = grid.z[0], grid.z[-1]
        apply_thin_pec_sheet(mesh, axis="y", position=y_pos, rect=(x_min, z_min, x_max, z_max))

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        z_c = 0.5 * (grid.z[:-1] + grid.z[1:])
        for i in range(Nx + 1):
            if grid.x[i] < x_min or grid.x[i] > x_max:
                continue
            for k in range(Nz):
                if z_c[k] < z_min or z_c[k] > z_max:
                    continue
                flat = i * (Ny + 1) * Nz + j_h * Nz + k
                assert mesh.pec_mask_edges[2, flat], f"Ez[{i},{j_h},{k}] should be PEC"

    def test_ey_edges_not_affected(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        j_h = 4
        y_pos = grid.y[j_h]
        apply_thin_pec_sheet(mesh, axis="y", position=y_pos)
        # Ey (component 1) should not be touched at all
        assert not mesh.pec_mask_edges[1].any(), "Ey must not be affected by y-axis sheet"

    def test_edges_outside_rect_not_affected(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        j_h = 4
        y_pos = grid.y[j_h]
        x_min, x_max = grid.x[2], grid.x[4]  # restrict to x[2]..x[4]
        z_min, z_max = grid.z[0], grid.z[-1]
        apply_thin_pec_sheet(mesh, axis="y", position=y_pos, rect=(x_min, z_min, x_max, z_max))

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        x_c = 0.5 * (grid.x[:-1] + grid.x[1:])
        # Check that Ex edges outside [x_min, x_max] at j_h are NOT PEC
        for i in range(Nx):
            if x_c[i] >= x_min and x_c[i] <= x_max:
                continue
            for k in range(Nz + 1):
                flat = i * (Ny + 1) * (Nz + 1) + j_h * (Nz + 1) + k
                assert not mesh.pec_mask_edges[0, flat], (
                    f"Ex[{i},{j_h},{k}] outside rect should NOT be PEC"
                )

    def test_edges_at_wrong_y_not_affected(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        j_h = 4
        y_pos = grid.y[j_h]
        apply_thin_pec_sheet(mesh, axis="y", position=y_pos)

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        # Check a different j-node (j=2) — Ex there should still be False
        j_other = 2
        for i in range(Nx):
            for k in range(Nz + 1):
                flat = i * (Ny + 1) * (Nz + 1) + j_other * (Nz + 1) + k
                assert not mesh.pec_mask_edges[0, flat], (
                    f"Ex[{i},{j_other},{k}] at wrong y should NOT be PEC"
                )

    def test_axis_x_zeros_ey_and_ez(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        i_h = 4
        x_pos = grid.x[i_h]
        apply_thin_pec_sheet(mesh, axis="x", position=x_pos)

        # Ex (component 0) must not be touched
        assert not mesh.pec_mask_edges[0].any(), "Ex must not be affected by x-axis sheet"
        # Ey (component 1) and Ez (component 2) must have some PEC entries
        assert mesh.pec_mask_edges[1].any(), "Ey must be PEC for x-axis sheet"
        assert mesh.pec_mask_edges[2].any(), "Ez must be PEC for x-axis sheet"

    def test_axis_z_zeros_ex_and_ey(self):
        mesh = self._make_mesh()
        grid = mesh.grid
        k_h = 4
        z_pos = grid.z[k_h]
        apply_thin_pec_sheet(mesh, axis="z", position=z_pos)

        # Ez (component 2) must not be touched
        assert not mesh.pec_mask_edges[2].any(), "Ez must not be affected by z-axis sheet"
        # Ex (component 0) and Ey (component 1) must have some PEC entries
        assert mesh.pec_mask_edges[0].any(), "Ex must be PEC for z-axis sheet"
        assert mesh.pec_mask_edges[1].any(), "Ey must be PEC for z-axis sheet"

    def test_pec_mask_shape_unchanged(self):
        mesh = self._make_mesh()
        original_shape = mesh.pec_mask_edges.shape
        apply_thin_pec_sheet(mesh, axis="y", position=mesh.grid.y[4])
        assert mesh.pec_mask_edges.shape == original_shape, "Shape must not change"


class TestPecMaskFaces:
    def test_full_pec_mesh_all_edges_marked(self):
        """If the whole domain is PEC, all E-edges should be marked."""
        from magnelio.mesh.indexing import build_pec_mask_faces

        Nx, Ny, Nz = 3, 3, 3
        grid = _grid(Nx=Nx, Ny=Ny, Nz=Nz, L=3e-3)
        pec = Material.pec()
        mid = np.zeros((Nx, Ny, Nz), dtype=np.int32)
        lib = {0: pec}
        mask = build_pec_mask_faces(grid, mid, lib)

        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        assert mask[0, :n_Ex].all()
        assert mask[1, :n_Ey].all()
        assert mask[2, :n_Ez].all()

    def test_air_mesh_no_edges_marked(self):
        from magnelio.mesh.indexing import build_pec_mask_faces

        grid = _grid()
        air = Material.air()
        mid = np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=np.int32)
        mask = build_pec_mask_faces(grid, mid, {0: air})
        assert not mask.any()
