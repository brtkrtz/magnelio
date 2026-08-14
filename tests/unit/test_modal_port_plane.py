"""Tests for magnelio.ports._modal.port_plane — port plane geometry on FIT mesh."""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.port_plane import BoxFace, PortPlane

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _small_mesh(Nx: int = 4, Ny: int = 3, Nz: int = 2) -> Mesh:
    """Equispaced (Nx, Ny, Nz)-cell box, 1 m × 1 m × 1 m with offset origin."""
    grid = GridLines(
        x=np.linspace(0.0, 1.0, Nx + 1),
        y=np.linspace(2.0, 3.0, Ny + 1),
        z=np.linspace(-1.0, 1.0, Nz + 1),
    )
    return Mesh.from_grid(grid)


# ---------------------------------------------------------------------
# BoxFace
# ---------------------------------------------------------------------


class TestBoxFace:
    def test_normal_axis(self):
        assert BoxFace.X_MIN.normal_axis == 0
        assert BoxFace.X_MAX.normal_axis == 0
        assert BoxFace.Y_MIN.normal_axis == 1
        assert BoxFace.Y_MAX.normal_axis == 1
        assert BoxFace.Z_MIN.normal_axis == 2
        assert BoxFace.Z_MAX.normal_axis == 2

    def test_is_max(self):
        for face in (BoxFace.X_MAX, BoxFace.Y_MAX, BoxFace.Z_MAX):
            assert face.is_max
        for face in (BoxFace.X_MIN, BoxFace.Y_MIN, BoxFace.Z_MIN):
            assert not face.is_max

    def test_inward_sign(self):
        assert BoxFace.X_MIN.inward_sign == 1
        assert BoxFace.X_MAX.inward_sign == -1

    def test_uv_axes_satisfy_inward_cross(self):
        """u_axis × v_axis should equal inward_normal."""
        # Cross product of unit vectors e_u × e_v gives e_n with sign:
        # +1 if (u, v, n) is a cyclic permutation of (0, 1, 2), else -1.
        # We require the result to equal inward_sign · normal_axis_unit.
        for face in BoxFace:
            u, v, n = face.u_axis, face.v_axis, face.normal_axis
            sign = _epsilon_ijk(u, v, n)
            assert sign == face.inward_sign, (
                f"{face}: u={u}, v={v}, n={n}, ε_uvn={sign}, "
                f"expected inward_sign={face.inward_sign}"
            )


def _epsilon_ijk(i: int, j: int, k: int) -> int:
    """Levi-Civita symbol for (i, j, k) ∈ {0, 1, 2}^3."""
    perm = (i, j, k)
    if i == j or j == k or i == k:
        return 0
    even = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    return 1 if perm in even else -1


# ---------------------------------------------------------------------
# PortPlane.from_mesh — basic structural checks
# ---------------------------------------------------------------------


class TestPortPlaneFromMeshShape:
    @pytest.mark.parametrize(
        "face,expected_n_u,expected_n_v",
        [
            # X_MIN: u=y(Ny cells), v=z(Nz nodes); v: u_nodes·v_cells
            (BoxFace.X_MIN, 3 * (2 + 1), (3 + 1) * 2),  # Ny·(Nz+1), (Ny+1)·Nz
            (BoxFace.X_MAX, 2 * (3 + 1), (2 + 1) * 3),  # u=z, v=y → Nz·(Ny+1), (Nz+1)·Ny
            (BoxFace.Y_MIN, 2 * (4 + 1), (2 + 1) * 4),  # u=z, v=x
            (BoxFace.Y_MAX, 4 * (2 + 1), (4 + 1) * 2),  # u=x, v=z
            (BoxFace.Z_MIN, 4 * (3 + 1), (4 + 1) * 3),  # u=x, v=y
            (BoxFace.Z_MAX, 3 * (4 + 1), (3 + 1) * 4),  # u=y, v=x
        ],
    )
    def test_edge_counts(self, face, expected_n_u, expected_n_v):
        mesh = _small_mesh()
        plane = PortPlane.from_mesh(face, mesh)
        assert plane.e_u_indices.shape == (expected_n_u,)
        assert plane.h_v_indices.shape == (expected_n_u,)
        assert plane.u_edge_uv.shape == (expected_n_u, 2)
        assert plane.u_edge_lengths.shape == (expected_n_u,)
        assert plane.e_v_indices.shape == (expected_n_v,)
        assert plane.h_u_indices.shape == (expected_n_v,)
        assert plane.v_edge_uv.shape == (expected_n_v, 2)
        assert plane.v_edge_lengths.shape == (expected_n_v,)

    @pytest.mark.parametrize(
        "face,expected_coord",
        [
            (BoxFace.X_MIN, 0.0),
            (BoxFace.X_MAX, 1.0),
            (BoxFace.Y_MIN, 2.0),
            (BoxFace.Y_MAX, 3.0),
            (BoxFace.Z_MIN, -1.0),
            (BoxFace.Z_MAX, 1.0),
        ],
    )
    def test_coordinate(self, face, expected_coord):
        mesh = _small_mesh()
        plane = PortPlane.from_mesh(face, mesh)
        assert plane.coordinate == pytest.approx(expected_coord, abs=1e-12)


# ---------------------------------------------------------------------
# PortPlane indexing — verify the right field components are addressed
# ---------------------------------------------------------------------


class TestPortPlaneIndexing:
    def test_x_min_e_u_addresses_ey_at_i0(self):
        """e_u_indices[k] should land on Ey[0, j, k]."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        fields = FieldState.zeros(Nx, Ny, Nz)
        # Set every Ey[0, j, k] to a unique tag; everything else stays 0.
        for j in range(Ny):
            for k in range(Nz + 1):
                fields.Ey[0, j, k] = 1000 * j + k
        # Check that flat-index lookup retrieves these tags.
        e_flat = fields.e_flat
        # Ordering by ij convention in _build_uv_edges:
        # idx[j*(Nz+1) + k] in the e_u_indices array corresponds to Ey[0, j, k].
        for j in range(Ny):
            for k in range(Nz + 1):
                idx_in_array = j * (Nz + 1) + k
                assert e_flat[plane.e_u_indices[idx_in_array]] == 1000 * j + k

    def test_x_min_h_v_addresses_hz_at_i0(self):
        """h_v_indices[k] should land on Hz[0, j, k] (co-located with Ey)."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        fields = FieldState.zeros(Nx, Ny, Nz)
        for j in range(Ny):
            for k in range(Nz + 1):
                fields.Hz[0, j, k] = 1.0 + 100 * j + k
        h_flat = fields.h_flat
        for j in range(Ny):
            for k in range(Nz + 1):
                idx_in_array = j * (Nz + 1) + k
                assert h_flat[plane.h_v_indices[idx_in_array]] == 1.0 + 100 * j + k

    def test_x_max_e_u_addresses_ez_at_iNx(self):
        """For X_MAX: u=z, primal-u edges are Ez at i=Nx."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        fields = FieldState.zeros(Nx, Ny, Nz)
        # u_axis = 2 (z), so E_u = E_z. Ez shape: (Nx+1, Ny+1, Nz).
        for j in range(Ny + 1):
            for k in range(Nz):
                fields.Ez[Nx, j, k] = 700 + 10 * j + k
        e_flat = fields.e_flat
        # _build_uv_edges meshgrid is (primal_axis_cells, secondary_axis_nodes)
        # with primal_axis=z (Nz cells), secondary_axis=y (Ny+1 nodes).
        # Ordering: idx[k*(Ny+1) + j] corresponds to Ez[Nx, j, k].
        for k in range(Nz):
            for j in range(Ny + 1):
                idx_in_array = k * (Ny + 1) + j
                assert e_flat[plane.e_u_indices[idx_in_array]] == 700 + 10 * j + k

    def test_x_max_uses_inner_cell_for_h(self):
        """X_MAX: H tangential is at the last cell-centre (i=Nx-1)."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        fields = FieldState.zeros(Nx, Ny, Nz)
        # u=z, v=y. H_v = H_y. Hy shape: (Nx, Ny+1, Nz). Use Hy[Nx-1, :, :].
        # H_v co-located with E_u (v-edges below).
        # Actually the test should be: u-edges' co-located H_v is Hy.
        # For X_MAX with u=z, v=y: H_v is the v-component = H_y.
        # Hy[Nx-1, j, k] is at (x_centre[Nx-1], y_node[j], z_centre[k]).
        for j in range(Ny + 1):
            for k in range(Nz):
                fields.Hy[Nx - 1, j, k] = -1.0 - j - 0.1 * k
        h_flat = fields.h_flat
        # u-edges meshgrid is (Nz cells × Ny+1 nodes).
        for k in range(Nz):
            for j in range(Ny + 1):
                idx_in_array = k * (Ny + 1) + j
                expected = -1.0 - j - 0.1 * k
                assert h_flat[plane.h_v_indices[idx_in_array]] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------
# PortPlane co-location — u-edge midpoints match h_v dual midpoints
# ---------------------------------------------------------------------


class TestPortPlaneColocation:
    @pytest.mark.parametrize("face", list(BoxFace))
    def test_u_edge_uv_finite(self, face):
        mesh = _small_mesh()
        plane = PortPlane.from_mesh(face, mesh)
        assert np.all(np.isfinite(plane.u_edge_uv))
        assert np.all(np.isfinite(plane.v_edge_uv))

    def test_x_min_u_edge_midpoints(self):
        """X_MIN: u=y, v=z; u-edge midpoint = (y_centre[j], z_node[k])."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        y_c = 0.5 * (mesh.grid.y[:-1] + mesh.grid.y[1:])
        for j in range(Ny):
            for k in range(Nz + 1):
                idx = j * (Nz + 1) + k
                u, v = plane.u_edge_uv[idx]
                assert u == pytest.approx(y_c[j], abs=1e-12)
                assert v == pytest.approx(mesh.grid.z[k], abs=1e-12)

    def test_x_min_v_edge_midpoints(self):
        """X_MIN: v-edge midpoint = (y_node[j], z_centre[k])."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        z_c = 0.5 * (mesh.grid.z[:-1] + mesh.grid.z[1:])
        # v-edges meshgrid: (Nz cells, Ny+1 nodes), so idx = k*(Ny+1) + j
        for k in range(Nz):
            for j in range(Ny + 1):
                idx = k * (Ny + 1) + j
                u, v = plane.v_edge_uv[idx]
                assert u == pytest.approx(mesh.grid.y[j], abs=1e-12)
                assert v == pytest.approx(z_c[k], abs=1e-12)


# ---------------------------------------------------------------------
# PortPlane edge lengths
# ---------------------------------------------------------------------


class TestPortPlaneEdgeLengths:
    def test_x_min_u_edge_length_is_dy(self):
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        for j in range(Ny):
            for k in range(Nz + 1):
                idx = j * (Nz + 1) + k
                assert plane.u_edge_lengths[idx] == pytest.approx(
                    mesh.grid.dy[j],
                    abs=1e-12,
                )

    def test_x_min_v_edge_length_is_dz(self):
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        for k in range(Nz):
            for j in range(Ny + 1):
                idx = k * (Ny + 1) + j
                assert plane.v_edge_lengths[idx] == pytest.approx(
                    mesh.grid.dz[k],
                    abs=1e-12,
                )


# ---------------------------------------------------------------------
# Smoke test: all six faces construct without error on non-square mesh
# ---------------------------------------------------------------------


class TestAllFacesSmoke:
    @pytest.mark.parametrize("face", list(BoxFace))
    def test_all_faces_buildable(self, face):
        mesh = _small_mesh(Nx=5, Ny=4, Nz=3)
        plane = PortPlane.from_mesh(face, mesh)
        assert isinstance(plane, PortPlane)
        # All indices must be valid (in range of flat E and H vectors)
        n_E = (5 * 5 * 4) + (6 * 4 * 4) + (6 * 5 * 3)
        n_H = (6 * 4 * 3) + (5 * 5 * 3) + (5 * 4 * 4)
        assert plane.e_u_indices.max() < n_E
        assert plane.e_v_indices.max() < n_E
        assert plane.h_u_indices.max() < n_H
        assert plane.h_v_indices.max() < n_H
        assert plane.e_u_indices.min() >= 0
        # Interior indices: same shape as port indices, in range
        assert plane.e_u_indices_interior.shape == plane.e_u_indices.shape
        assert plane.e_v_indices_interior.shape == plane.e_v_indices.shape
        assert plane.e_u_indices_interior.max() < n_E
        assert plane.e_u_indices_interior.min() >= 0
        # Indices must differ from port plane (one cell shifted)
        assert not np.array_equal(plane.e_u_indices, plane.e_u_indices_interior)


class TestPortPlaneInterior:
    def test_x_min_interior_at_i1(self):
        """X_MIN port at i=0, interior at i=1.  Ey edge index shift = Ny·(Nz+1)."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        # Ey shape (Nx+1, Ny, Nz+1) → flat-stride for i is Ny·(Nz+1).
        stride_i = Ny * (Nz + 1)
        np.testing.assert_array_equal(
            plane.e_u_indices_interior,
            plane.e_u_indices + stride_i,
        )

    def test_x_max_interior_at_iNx_minus_1(self):
        """X_MAX port at i=Nx, interior at i=Nx-1.  Stride is negative."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        # u_axis = z (axis 2), so e_u uses Ez component (Nx+1, Ny+1, Nz).
        stride_i = (Ny + 1) * Nz
        np.testing.assert_array_equal(
            plane.e_u_indices_interior,
            plane.e_u_indices - stride_i,
        )

    @pytest.mark.parametrize(
        "face,expected_dx_index",
        [
            (BoxFace.X_MIN, "dx[0]"),
            (BoxFace.X_MAX, "dx[Nx-1]"),
            (BoxFace.Y_MIN, "dy[0]"),
            (BoxFace.Y_MAX, "dy[Ny-1]"),
            (BoxFace.Z_MIN, "dz[0]"),
            (BoxFace.Z_MAX, "dz[Nz-1]"),
        ],
    )
    def test_normal_dx_is_boundary_cell_size(self, face, expected_dx_index):
        # Use a non-uniform mesh so we can distinguish "first" vs "last" cells.
        grid = GridLines(
            x=np.array([0.0, 0.5, 1.0, 1.7, 2.5]),  # dx = [0.5, 0.5, 0.7, 0.8]
            y=np.array([0.0, 0.3, 0.7, 1.0]),  # dy = [0.3, 0.4, 0.3]
            z=np.array([0.0, 0.6, 1.0]),  # dz = [0.6, 0.4]
        )
        mesh = Mesh.from_grid(grid)
        plane = PortPlane.from_mesh(face, mesh)
        deltas = {0: grid.dx, 1: grid.dy, 2: grid.dz}[face.normal_axis]
        expected = float(deltas[-1] if face.is_max else deltas[0])
        assert plane.normal_dx == pytest.approx(expected, abs=1e-12)

    def test_x_min_interior_e_addresses_correct_cell(self):
        """Set Ey[1, j, k] = unique tags; verify e_u_indices_interior reads them."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        fields = FieldState.zeros(Nx, Ny, Nz)
        for j in range(Ny):
            for k in range(Nz + 1):
                fields.Ey[1, j, k] = 5000 + 10 * j + k
        e_flat = fields.e_flat
        for j in range(Ny):
            for k in range(Nz + 1):
                idx_in_array = j * (Nz + 1) + k
                assert e_flat[plane.e_u_indices_interior[idx_in_array]] == 5000 + 10 * j + k


def _domain_window_for(face: BoxFace) -> tuple:
    """Full-domain corner-pair window of _small_mesh in global axis ordering."""
    extents = [(0.0, 1.0), (2.0, 3.0), (-1.0, 1.0)]
    a_ax, b_ax = sorted((face.u_axis, face.v_axis))
    (a_lo, a_hi), (b_lo, b_hi) = extents[a_ax], extents[b_ax]
    return ((a_lo, b_lo), (a_hi, b_hi))


class TestSubFaceWholeFaceEquivalence:
    @pytest.mark.parametrize("face", list(BoxFace))
    def test_domain_window_equals_default(self, face):
        mesh = _small_mesh()
        full = PortPlane.from_mesh(face, mesh)
        boxed = PortPlane.from_mesh(face, mesh, window=_domain_window_for(face))
        for attr in (
            "e_u_indices",
            "e_v_indices",
            "h_u_indices",
            "h_v_indices",
            "u_edge_uv",
            "v_edge_uv",
            "u_edge_lengths",
            "v_edge_lengths",
            "e_u_indices_interior",
            "e_v_indices_interior",
        ):
            np.testing.assert_array_equal(
                getattr(boxed, attr),
                getattr(full, attr),
                err_msg=attr,
            )
        assert boxed.u_node_window == full.u_node_window
        assert boxed.v_node_window == full.v_node_window

    @pytest.mark.parametrize("face", list(BoxFace))
    def test_oversized_window_clips_to_domain(self, face):
        """Legacy behaviour: an oversized window is clipped, not an error."""
        mesh = _small_mesh()
        full = PortPlane.from_mesh(face, mesh)
        boxed = PortPlane.from_mesh(
            face,
            mesh,
            window=((-1e3, -1e3), (1e3, 1e3)),
        )
        np.testing.assert_array_equal(boxed.e_u_indices, full.e_u_indices)
        np.testing.assert_array_equal(boxed.e_v_indices, full.e_v_indices)
        assert boxed.u_bounds == full.u_bounds
        assert boxed.v_bounds == full.v_bounds

    def test_whole_face_windows_and_counts(self):
        mesh = _small_mesh()  # Nx=4, Ny=3, Nz=2
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)  # u=y, v=z
        assert plane.u_node_window == (0, 3)
        assert plane.v_node_window == (0, 2)
        assert (plane.n_cells_u, plane.n_cells_v) == (3, 2)
        assert (plane.n_nodes_u, plane.n_nodes_v) == (4, 3)
        assert plane.u_bounds == (2.0, 3.0)
        assert plane.v_bounds == (-1.0, 1.0)


class TestSubFaceGeometry:
    def test_x_min_sub_rectangle(self):
        """X_MIN (u=y, v=z): window is (y_range, z_range) in global order."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)  # y nodes at 2, 2.333.., 2.666.., 3
        y_n, z_n = mesh.grid.y, mesh.grid.z
        sub = PortPlane.from_mesh(
            BoxFace.X_MIN,
            mesh,
            window=((y_n[1], z_n[0]), (y_n[3], z_n[1])),
        )
        assert sub.u_node_window == (1, 3)
        assert sub.v_node_window == (0, 1)
        assert sub.e_u_indices.shape == (4,)
        assert sub.e_v_indices.shape == (3,)
        assert sub.h_v_indices.shape == (4,)
        assert sub.h_u_indices.shape == (3,)
        assert sub.u_bounds == (pytest.approx(y_n[1]), pytest.approx(y_n[3]))
        assert sub.v_bounds == (pytest.approx(z_n[0]), pytest.approx(z_n[1]))

    def test_x_min_sub_indices_address_expected_ey(self):
        """Sub-face e_u_indices must land exactly on Ey[0, j, k] in-window."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        y_n, z_n = mesh.grid.y, mesh.grid.z
        sub = PortPlane.from_mesh(
            BoxFace.X_MIN,
            mesh,
            window=((y_n[1], z_n[0]), (y_n[3], z_n[1])),
        )
        fields = FieldState.zeros(Nx, Ny, Nz)
        for j in range(Ny):
            for k in range(Nz + 1):
                fields.Ey[0, j, k] = 1000 * j + k
        e_flat = fields.e_flat
        expected = [1000 * j + k for j in (1, 2) for k in (0, 1)]
        assert [e_flat[i] for i in sub.e_u_indices] == expected

    def test_x_min_sub_interior_shift(self):
        """Interior indices shift one cell inward, same as whole-face."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        y_n, z_n = mesh.grid.y, mesh.grid.z
        sub = PortPlane.from_mesh(
            BoxFace.X_MIN,
            mesh,
            window=((y_n[1], z_n[0]), (y_n[3], z_n[1])),
        )
        stride_i = Ny * (Nz + 1)  # Ey i-stride
        np.testing.assert_array_equal(
            sub.e_u_indices_interior,
            sub.e_u_indices + stride_i,
        )

    def test_x_max_sub_respects_uv_swap(self):
        """X_MAX has u=z, v=y; window stays in global (y_range, z_range)."""
        Nx, Ny, Nz = 4, 3, 2
        mesh = _small_mesh(Nx, Ny, Nz)
        y_n, z_n = mesh.grid.y, mesh.grid.z
        sub = PortPlane.from_mesh(
            BoxFace.X_MAX,
            mesh,
            window=((y_n[0], z_n[1]), (y_n[2], z_n[2])),
        )
        assert sub.u_node_window == (1, 2)
        assert sub.v_node_window == (0, 2)
        assert sub.u_bounds == (pytest.approx(z_n[1]), pytest.approx(z_n[2]))
        assert sub.v_bounds == (pytest.approx(y_n[0]), pytest.approx(y_n[2]))
        assert np.all(sub.u_edge_uv[:, 0] >= z_n[1] - 1e-12)
        assert np.all(sub.u_edge_uv[:, 0] <= z_n[2] + 1e-12)
        assert np.all(sub.u_edge_uv[:, 1] >= y_n[0] - 1e-12)
        assert np.all(sub.u_edge_uv[:, 1] <= y_n[2] + 1e-12)

    def test_sub_edges_are_subset_of_whole_face(self):
        mesh = _small_mesh()
        full = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
        sub = PortPlane.from_mesh(
            BoxFace.Z_MIN,
            mesh,
            window=((0.25, 2.0), (0.75, 2.5)),
        )
        assert set(sub.e_u_indices) < set(full.e_u_indices)
        assert set(sub.e_v_indices) < set(full.e_v_indices)
        assert set(sub.h_u_indices) < set(full.h_u_indices)
        assert set(sub.h_v_indices) < set(full.h_v_indices)

    def test_window_snaps_to_nearest_nodes(self):
        """Off-grid window coordinates snap to the closest grid nodes."""
        mesh = _small_mesh()  # x nodes at 0, 0.25, 0.5, 0.75, 1
        sub = PortPlane.from_mesh(
            BoxFace.Z_MIN,
            mesh,
            window=((0.23, 2.0), (0.80, 3.0)),
        )
        assert sub.u_node_window == (1, 3)  # 0.23 → 0.25, 0.80 → 0.75
        assert sub.u_bounds == (pytest.approx(0.25), pytest.approx(0.75))


class TestSubFaceValidation:
    def test_corner_order_irrelevant(self):
        """Corners may come in any order — same window either way."""
        mesh = _small_mesh()
        p1 = PortPlane.from_mesh(
            BoxFace.X_MIN,
            mesh,
            window=((2.2, -1.0), (2.8, 1.0)),
        )
        p2 = PortPlane.from_mesh(
            BoxFace.X_MIN,
            mesh,
            window=((2.8, 1.0), (2.2, -1.0)),
        )
        assert p1.u_node_window == p2.u_node_window
        assert p1.v_node_window == p2.v_node_window

    def test_degenerate_corner_extent_raises(self):
        mesh = _small_mesh()
        with pytest.raises(ValueError, match="degenerate"):
            PortPlane.from_mesh(
                BoxFace.X_MIN,
                mesh,
                window=((2.5, -1.0), (2.5, 1.0)),
            )

    def test_window_outside_domain_raises(self):
        mesh = _small_mesh()
        with pytest.raises(ValueError, match="outside the domain"):
            PortPlane.from_mesh(
                BoxFace.X_MIN,
                mesh,
                window=((5.0, -1.0), (6.0, 1.0)),
            )

    def test_degenerate_after_snapping_raises(self):
        """A range inside one cell collapses onto a single node."""
        mesh = _small_mesh()  # y nodes spaced 1/3
        with pytest.raises(ValueError, match="fewer than one grid cell"):
            PortPlane.from_mesh(
                BoxFace.X_MIN,
                mesh,
                window=((2.30, -1.0), (2.36, 1.0)),
            )

    def test_malformed_window_raises(self):
        mesh = _small_mesh()
        with pytest.raises(ValueError, match="two corner points"):
            PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=(1.0, 2.0, 3.0))
