"""Sub-face port-edge BC rule + windowed 2D mode solve (WP6.2).

Covers:

1. ``resolve_port_edge_pec`` — the legacy ``_edge_bc`` rule: a port
   edge on a domain boundary inherits that wall's BC, an interior
   (sub-face window) edge inherits the port face's BC.
2. ``build_port_edge_pec_mask`` — the ``[e_u | e_v]`` Dirichlet mask
   for the plane's own (window) boundary edges.
3. Windowed 2D operators — de Rham exactness ``C_2d @ G_2d == 0`` on a
   sub-face plane, and the physical anchor: the TE10 cutoff of a
   WR-90-sized sub-face window embedded in a larger vacuum face equals
   ``c / (2a)`` of the *window*, not of the full face.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    Numerical2DModeSolver,
    PortPlane,
)
from magnelio.ports._modal.curl_curl_2d import (
    build_2d_curl_curl,
    build_2d_gradient,
)
from magnelio.ports._modal.port_plane import (
    build_port_edge_pec_mask,
    resolve_port_edge_pec,
)

C0 = 2.99792458e8
WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _mesh(Nx=2, Ny=6, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0) -> Mesh:
    grid = GridLines(
        x=np.linspace(0.0, Lx, Nx + 1),
        y=np.linspace(0.0, Ly, Ny + 1),
        z=np.linspace(0.0, Lz, Nz + 1),
    )
    return Mesh.from_grid(grid)


# ---------------------------------------------------------------------
# resolve_port_edge_pec — the legacy _edge_bc rule
# ---------------------------------------------------------------------


class TestResolvePortEdgePec:
    def test_whole_face_inherits_lateral_wall_bcs(self):
        """X_MIN whole face (u=y, v=z): edges are the four lateral walls."""
        mesh = _mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        pec = resolve_port_edge_pec(
            plane,
            mesh,
            {BoxFace.Y_MIN, BoxFace.Z_MAX},
        )
        assert pec == {
            "u_min": True,  # y_min wall is PEC
            "u_max": False,  # y_max is not
            "v_min": False,  # z_min is not
            "v_max": True,  # z_max is PEC
        }

    def test_interior_edges_inherit_port_face_bc(self):
        """A centred sub-face window: all four edges take the face's BC."""
        mesh = _mesh()
        window = ((1 / 6, 0.25), (5 / 6, 0.75))  # interior on both axes
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        all_pec = resolve_port_edge_pec(plane, mesh, {BoxFace.X_MIN})
        assert all_pec == {
            "u_min": True,
            "u_max": True,
            "v_min": True,
            "v_max": True,
        }
        none_pec = resolve_port_edge_pec(plane, mesh, {BoxFace.Y_MIN})
        assert none_pec == {
            "u_min": False,
            "u_max": False,
            "v_min": False,
            "v_max": False,
        }

    def test_mixed_wall_and_interior_edges(self):
        """Window flush with y_min: u_min is the wall, the rest interior."""
        mesh = _mesh()
        window = ((0.0, 0.25), (0.5, 0.75))
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        # Wall PEC, port face not PEC: only the wall-touching edge is PEC.
        pec = resolve_port_edge_pec(plane, mesh, {BoxFace.Y_MIN})
        assert pec == {
            "u_min": True,
            "u_max": False,
            "v_min": False,
            "v_max": False,
        }
        # Port face PEC, wall not: exactly the complement.
        pec = resolve_port_edge_pec(plane, mesh, {BoxFace.X_MIN})
        assert pec == {
            "u_min": False,
            "u_max": True,
            "v_min": True,
            "v_max": True,
        }

    def test_max_face_uv_swap(self):
        """X_MAX has u=z, v=y: u_min must map to the z_min wall."""
        mesh = _mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        pec = resolve_port_edge_pec(plane, mesh, {BoxFace.Z_MIN})
        assert pec == {
            "u_min": True,
            "u_max": False,
            "v_min": False,
            "v_max": False,
        }


# ---------------------------------------------------------------------
# build_port_edge_pec_mask — window-boundary Dirichlet mask
# ---------------------------------------------------------------------


class TestBuildPortEdgePecMask:
    def test_no_pec_edges_gives_empty_mask(self):
        mesh = _mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        mask = build_port_edge_pec_mask(
            plane,
            {
                "u_min": False,
                "u_max": False,
                "v_min": False,
                "v_max": False,
            },
        )
        assert mask.shape == (plane.e_u_indices.size + plane.e_v_indices.size,)
        assert not mask.any()

    def test_marked_edges_lie_on_the_claimed_boundary(self):
        """Coordinates of masked edges sit exactly on the window bounds."""
        mesh = _mesh(Ny=8, Nz=6)
        window = ((0.25, 1 / 3), (0.75, 5 / 6))
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        n_u = plane.e_u_indices.size

        mask = build_port_edge_pec_mask(
            plane,
            {
                "u_min": True,
                "u_max": False,
                "v_min": False,
                "v_max": False,
            },
        )
        # u_min edge runs along v → tangent edges are v-edges whose
        # u-node is at u_bounds[0]; count = n_cells_v.
        assert mask[:n_u].sum() == 0
        marked_v = np.nonzero(mask[n_u:])[0]
        assert marked_v.size == plane.n_cells_v
        assert np.allclose(
            plane.v_edge_uv[marked_v, 0],
            plane.u_bounds[0],
            atol=1e-12,
        )

        mask = build_port_edge_pec_mask(
            plane,
            {
                "u_min": False,
                "u_max": False,
                "v_min": False,
                "v_max": True,
            },
        )
        # v_max edge runs along u → u-edges with v-node at v_bounds[1].
        assert mask[n_u:].sum() == 0
        marked_u = np.nonzero(mask[:n_u])[0]
        assert marked_u.size == plane.n_cells_u
        assert np.allclose(
            plane.u_edge_uv[marked_u, 1],
            plane.v_bounds[1],
            atol=1e-12,
        )

    def test_all_edges_marks_full_window_ring(self):
        mesh = _mesh(Ny=8, Nz=6)
        window = ((0.25, 1 / 3), (0.75, 5 / 6))
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        mask = build_port_edge_pec_mask(
            plane,
            {
                "u_min": True,
                "u_max": True,
                "v_min": True,
                "v_max": True,
            },
        )
        assert mask.sum() == 2 * plane.n_cells_u + 2 * plane.n_cells_v


# ---------------------------------------------------------------------
# Windowed 2D operators — structure + physical anchor
# ---------------------------------------------------------------------


class TestSubFaceOperators:
    def test_de_rham_exactness_on_sub_face(self):
        """C_2d @ G_2d == 0 must hold on the window slice too."""
        mesh = _mesh(Ny=8, Nz=6)
        window = ((0.25, 1 / 3), (0.75, 5 / 6))
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        C = build_curl_matrix(mesh.grid)
        G = build_gradient_matrix(mesh.grid)
        K, M, primal_2d = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)
        G_2d, nodes_2d, edges_2d = build_2d_gradient(plane, mesh.grid, G)

        assert nodes_2d.size == plane.n_nodes_u * plane.n_nodes_v
        np.testing.assert_array_equal(edges_2d, primal_2d)
        assert K.shape == (edges_2d.size, edges_2d.size)
        # Gradient fields are curl-free: K @ (G_2d @ phi) == 0.
        rng = np.random.default_rng(42)
        phi = rng.standard_normal(nodes_2d.size)
        residual = K @ (G_2d @ phi)
        assert np.abs(residual).max() < 1e-8 * max(np.abs(K.data).max(), 1.0)

    def test_te10_cutoff_of_embedded_window(self):
        """WR-90-sized window centred in a 2× face → window TE10 cutoff.

        The face is 2a × 2b of vacuum with no PEC anywhere in the mesh;
        the window's PEC boundary comes entirely from the edge-BC rule
        (interior edges of a port embedded in a PEC wall).  The lowest
        mode must be the *window's* TE10 at c/(2a) — the full face's
        TE10 (c/(4a)) must not appear.
        """
        Ny, Nz = 48, 24  # 2a × 2b face → 24 × 12 cells in the window
        mesh = _mesh(
            Nx=2,
            Ny=Ny,
            Nz=Nz,
            Lx=5e-3,
            Ly=2 * WR90_A,
            Lz=2 * WR90_B,
        )
        window = (
            (WR90_A / 2, WR90_B / 2),
            (3 * WR90_A / 2, 3 * WR90_B / 2),
        )
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh, window=window)
        assert (plane.n_cells_u, plane.n_cells_v) == (24, 12)

        # All four window edges are interior; the port face is PEC.
        edge_pec = resolve_port_edge_pec(plane, mesh, {BoxFace.X_MIN})
        assert all(edge_pec.values())
        pec_mask = build_port_edge_pec_mask(plane, edge_pec)

        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        C = build_curl_matrix(mesh.grid)
        K, M, primal_2d = build_2d_curl_curl(plane, mesh.grid, M_eps, M_mu, C)

        solver = Numerical2DModeSolver(
            plane=plane,
            K=K,
            M=M,
            primal_2d_indices=primal_2d,
            pec_edge_mask=pec_mask,
            mode_type=ModeType.TE,
            m_mu_flat=M_mu,
        )
        modes = solver.solve(n_modes=1, f_calc=10e9)
        f_c = modes[0].omega_c / (2.0 * np.pi)

        f_te10_window = C0 / (2.0 * WR90_A)
        f_te10_face = C0 / (2.0 * 2.0 * WR90_A)
        assert f_c == pytest.approx(f_te10_window, rel=1e-2)
        assert abs(f_c - f_te10_face) > 0.4 * f_te10_face


# ---------------------------------------------------------------------
# _complement_absorber_arrays — frozen zero-M_eps edges
# ---------------------------------------------------------------------


class TestComplementAbsorberFrozenEdges:
    """A conformal edge clamped to ``M_eps == 0`` without a PEC-mask
    entry must join the absorber's dead set with a finite coefficient —
    mirroring the volume update's ``live_E = M_eps > 0`` convention.
    Previously such an edge produced a NaN Mur coefficient on a *live*
    edge, which an active complement absorber would have injected into
    the fields."""

    def test_zero_eps_edge_is_dead_and_all_coefficients_finite(self):
        import warnings

        from magnelio.ports._modal.factory import _complement_absorber_arrays

        grid = GridLines(
            x=np.linspace(0.0, 1.0, 3),
            y=np.linspace(0.0, 1.0, 7),
            z=np.linspace(0.0, 1.0, 5),
        )
        # PMC on the port face: a PEC face would put every plane edge
        # into the wall mask and the dead-set assertion would be vacuous.
        mesh = Mesh.from_grid(grid, boundary_conditions={"xmin": "PMC"})
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        m_eps = np.array(build_M_eps(mesh), dtype=float)

        target = int(plane.e_u_indices[plane.e_u_indices.size // 2])
        m_eps[target] = 0.0
        dt = 1e-12

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r_u, r_v, live_u, live_v = _complement_absorber_arrays(
                plane,
                mesh,
                BoxFace.X_MIN,
                m_eps,
                dt,
                None,
            )

        assert np.all(np.isfinite(r_u))
        assert np.all(np.isfinite(r_v))
        pos = int(np.nonzero(plane.e_u_indices == target)[0][0])
        assert live_u[pos] == 0.0
        assert r_u[pos] == 0.0
        # The neighbouring interior edge with untouched M_eps stays live.
        assert live_u[pos + 1] == 1.0
        assert live_u.sum() > 0
        assert live_v.sum() > 0
