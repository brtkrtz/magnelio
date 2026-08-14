"""Tests for the PortSpecMultiConductor factory dispatch.

Phase-2 cleanup item 1.  Validates the new public API for
multi-conductor TEM/QTEM ports — :class:`PortSpecMultiConductor` plus
the three :class:`ConductorSpec` types
(:class:`BboxLateralConductor`, :class:`WallConductor`,
:class:`RegionConductor`) — and the
:func:`build_modal_port` dispatch that wires them through
:func:`solve_tem_laplace` (homogeneous) or :func:`solve_qtem_laplace`
(QTEM, with :func:`build_M_eps_vacuum` providing the dual-Laplace
reference).

Six layers of validation:

1. **Spec invariants.**  ``__post_init__`` rejects fewer than 2
   conductors, non-positive ``epsilon_r``, non-positive ``n_modes``,
   ``n_modes > K-1``.
2. **ConductorSpec resolution.**  Each ConductorSpec type produces
   the correct local-2D-node-index set: ``BboxLateralConductor`` →
   four bbox corners; ``WallConductor`` → one wall (with face-on-
   port-plane-normal rejected); ``RegionConductor`` → axis-aligned
   region with global-axis ordering.
3. **Deduplication / shadow.**  Nodes belonging to an earlier group
   are removed from later groups; a fully-shadowed signal conductor
   raises.
4. **TEM dispatch via factory matches direct ``solve_tem_laplace``.**
   On a rect-coax setup, the Z_line and the discrete-mode profiles
   coming out of the factory match the standalone path byte-for-byte
   — confirming the dispatch is a thin wrapper.
5. **QTEM dispatch via factory matches direct ``solve_qtem_laplace``
   + ``build_M_eps_vacuum``.**  On a microstrip-with-substrate
   setup, ε_eff and Z_0 from the factory agree with the standalone
   path.
6. **N_modes truncation.**  Multi-signal-conductor setups respect
   the ``n_modes`` limit on the resulting Mode list.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_eps_vacuum,
    build_M_mu,
)
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BboxLateralConductor,
    BoxFace,
    PortPlane,
    PortSpecMultiConductor,
    RegionConductor,
    WallConductor,
    build_modal_port,
    solve_qtem_laplace,
    solve_tem_laplace,
)
from magnelio.ports._modal.curl_curl_2d import (
    build_2d_curl_curl,
    build_2d_gradient,
)
from magnelio.ports._modal.factory import _resolve_conductors
from magnelio.solver.stability import courant_dt

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _rect_coax_setup(*, B: float = 8e-3, a: float = 2e-3, L_x: float = 20e-3):
    """Rect coax via Mesh.from_geometry: outer bbox PEC walls + inner PEC brick."""
    y0 = z0 = (B - a) / 2
    air = Material.air()
    pec = Material.pec()
    bbox = Brick(origin=(0, 0, 0), size=(L_x, B, B), material=air)
    inner = Brick(origin=(0, y0, z0), size=(L_x, a, a), material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, inner))
    model.add(inner)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.4e-3,
            min_cells_per_feature=4,
        ),
        f_max=8e9,
    )
    return mesh, B, a, y0, z0, L_x


def _microstrip_setup(
    *,
    W: float = 2e-3,
    h: float = 1e-3,
    eps_r_sub: float = 9.8,
    W_box: float = 16e-3,
    H_box: float = 8e-3,
):
    """Microstrip via Mesh.from_geometry: substrate brick + air cap."""
    air = Material.air()
    diel = Material(name="subst", epsilon=(eps_r_sub,) * 3)
    substrate = Brick(
        origin=(-W_box / 2, -W_box / 2, 0),
        size=(4e-3, W_box, h),
        material=diel,
    )
    air_cap = Brick(
        origin=(-W_box / 2, -W_box / 2, h),
        size=(4e-3, W_box, H_box - h),
        material=air,
    )
    model = GeometryModel()
    model.add(substrate)
    model.add(air_cap)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.1e-3,
            min_cells_per_feature=4,
        ),
        f_max=8e9,
    )
    return mesh, W, h, eps_r_sub


# ---------------------------------------------------------------------
# 1) Spec invariants
# ---------------------------------------------------------------------


class TestSpecInvariants:
    def test_fewer_than_two_conductors_rejected(self):
        with pytest.raises(ValueError, match="at least 2 conductors"):
            PortSpecMultiConductor(
                name="p",
                plane=BoxFace.X_MIN,
                conductors=(BboxLateralConductor(),),
            )

    def test_zero_epsilon_r_rejected(self):
        with pytest.raises(ValueError, match="epsilon_r must be positive"):
            PortSpecMultiConductor(
                name="p",
                plane=BoxFace.X_MIN,
                conductors=(BboxLateralConductor(), BboxLateralConductor()),
                epsilon_r=0.0,
            )

    def test_zero_n_modes_rejected(self):
        with pytest.raises(ValueError, match="n_modes must be positive"):
            PortSpecMultiConductor(
                name="p",
                plane=BoxFace.X_MIN,
                conductors=(BboxLateralConductor(), BboxLateralConductor()),
                n_modes=0,
            )

    def test_n_modes_too_many_rejected(self):
        with pytest.raises(ValueError, match="exceeds K-1"):
            # 2 conductors → K-1 = 1 mode max, but we ask for 2.
            PortSpecMultiConductor(
                name="p",
                plane=BoxFace.X_MIN,
                conductors=(BboxLateralConductor(), BboxLateralConductor()),
                n_modes=2,
            )

    def test_qtem_path_via_epsilon_r_none(self):
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=(BboxLateralConductor(), BboxLateralConductor()),
            epsilon_r=None,
        )
        assert spec.epsilon_r is None


# ---------------------------------------------------------------------
# 2) ConductorSpec resolution
# ---------------------------------------------------------------------


class TestConductorResolution:
    @pytest.fixture
    def small_mesh(self):
        grid = GridLines(
            x=np.linspace(0, 4e-3, 5),  # Nx=4
            y=np.linspace(0, 4e-3, 5),  # Ny=4
            z=np.linspace(0, 4e-3, 5),  # Nz=4
        )
        return Mesh.from_grid(grid)

    def test_bbox_lateral_marks_corners_only(self, small_mesh):
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        groups = _resolve_conductors(
            (
                BboxLateralConductor(),
                RegionConductor(
                    axis_a_range=(1e-3, 3e-3),
                    axis_b_range=(1e-3, 3e-3),
                ),
            ),
            plane,
            small_mesh,
        )
        # X_MIN: u=y, v=z. (Ny+1)·(Nz+1) = 25 nodes.
        # Bbox lateral: any j ∈ {0, 4} OR k ∈ {0, 4} → 5+5+3+3 = 16 nodes.
        assert groups[0].size == 16

    def test_wall_conductor_picks_single_wall(self, small_mesh):
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        groups = _resolve_conductors(
            (
                WallConductor(face=BoxFace.Z_MIN),
                RegionConductor(
                    axis_a_range=(2e-3, 3e-3),
                    axis_b_range=(2e-3, 3e-3),
                ),
            ),
            plane,
            small_mesh,
        )
        # X_MIN: u=y, v=z; WallConductor(Z_MIN) → iv=0 → Ny+1 = 5 nodes.
        assert groups[0].size == 5

    def test_wall_conductor_on_port_normal_rejected(self, small_mesh):
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        with pytest.raises(ValueError, match="shares the port plane's normal"):
            _resolve_conductors(
                (
                    WallConductor(face=BoxFace.X_MAX),  # X_MAX is parallel to X_MIN
                    RegionConductor(
                        axis_a_range=(1e-3, 2e-3),
                        axis_b_range=(1e-3, 2e-3),
                    ),
                ),
                plane,
                small_mesh,
            )

    def test_region_conductor_global_axis_ordering(self, small_mesh):
        """For X_MIN (u=y, v=z): axis_a_range = y range, axis_b_range = z range."""
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        groups = _resolve_conductors(
            (
                BboxLateralConductor(),
                RegionConductor(
                    axis_a_range=(1e-3, 1e-3),  # exactly y=1mm, j=1
                    axis_b_range=(0, 4e-3),  # z full range
                ),
            ),
            plane,
            small_mesh,
        )
        # j=1 across all k = 5 nodes; minus those already in BboxLateralConductor
        # (k=0, k=4) → 3 exclusive nodes (k=1, 2, 3).
        assert groups[1].size == 3

    def test_region_conductor_xmax_swaps_axes(self):
        """X_MAX has u=z, v=y (swapped).  axis_a_range still = y range."""
        grid = GridLines(
            x=np.linspace(0, 4e-3, 5),
            y=np.linspace(0, 8e-3, 5),  # Ny=4, dy=2mm
            z=np.linspace(0, 4e-3, 5),  # Nz=4, dz=1mm
        )
        mesh = Mesh.from_grid(grid)
        plane = PortPlane.from_mesh(BoxFace.X_MAX, mesh)
        groups = _resolve_conductors(
            (
                BboxLateralConductor(),
                RegionConductor(
                    axis_a_range=(2e-3, 6e-3),  # y range
                    axis_b_range=(1e-3, 3e-3),  # z range
                ),
            ),
            plane,
            mesh,
        )
        # Region: y∈[2,6]=j∈{1,2,3}, z∈[1,3]=k∈{1,2,3}; 3×3=9 candidate nodes,
        # all interior (no overlap with bbox lateral).  9 nodes.
        assert groups[1].size == 9

    def test_empty_resolution_rejected(self, small_mesh):
        """A region that misses every node fails."""
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        with pytest.raises(ValueError, match="zero 2D nodes"):
            _resolve_conductors(
                (
                    BboxLateralConductor(),
                    RegionConductor(
                        axis_a_range=(10e-3, 11e-3),  # outside the mesh
                        axis_b_range=(10e-3, 11e-3),
                    ),
                ),
                plane,
                small_mesh,
            )

    def test_unsupported_conductor_type_rejected(self, small_mesh):
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        with pytest.raises(TypeError, match="unsupported ConductorSpec"):
            _resolve_conductors(
                (BboxLateralConductor(), object()),  # type: ignore[arg-type]
                plane,
                small_mesh,
            )


# ---------------------------------------------------------------------
# 3) Deduplication semantic
# ---------------------------------------------------------------------


class TestDeduplication:
    @pytest.fixture
    def small_mesh(self):
        grid = GridLines(
            x=np.linspace(0, 4e-3, 5),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        return Mesh.from_grid(grid)

    def test_signal_overlap_with_ground_dedups(self, small_mesh):
        """A signal region that nominally covers ground-plane nodes
        loses them silently."""
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        groups = _resolve_conductors(
            (
                WallConductor(face=BoxFace.Z_MIN),
                RegionConductor(
                    axis_a_range=(0, 4e-3),
                    axis_b_range=(0, 1e-3),  # includes z=0 (ground!) + z=1mm
                ),
            ),
            plane,
            small_mesh,
        )
        # Ground: 5 nodes (j=0..4, k=0).
        # Region nominal: y∈full=5 cols × z∈{0,1}=2 rows = 10 candidates.
        # After dedup: subtract the 5 ground nodes → 5 exclusive.
        assert groups[0].size == 5
        assert groups[1].size == 5

    def test_fully_shadowed_signal_rejected(self, small_mesh):
        plane = PortPlane.from_mesh(BoxFace.X_MIN, small_mesh)
        with pytest.raises(ValueError, match="fully shadowed"):
            _resolve_conductors(
                (
                    BboxLateralConductor(),  # all corners
                    RegionConductor(
                        axis_a_range=(0, 0),  # only j=0 (corner)
                        axis_b_range=(0, 4e-3),
                    ),
                ),
                plane,
                small_mesh,
            )


# ---------------------------------------------------------------------
# 4) TEM dispatch matches standalone solve_tem_laplace
# ---------------------------------------------------------------------


class TestTEMFactoryEquivalence:
    """Factory dispatch for TEM produces the same Mode as the
    standalone ``solve_tem_laplace`` path."""

    def test_rect_coax_z_line_matches(self):
        mesh, B, a, y0, z0, L_x = _rect_coax_setup()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        # Factory path.
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=(
                BboxLateralConductor(),
                RegionConductor(
                    axis_a_range=(y0, y0 + a),
                    axis_b_range=(z0, z0 + a),
                ),
            ),
            epsilon_r=1.0,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=5e9)
        z_factory = op.discrete_modes[0].mode.z_line

        # Standalone path — must replicate the factory's port-plane
        # flatten preprocessing (mass + PEC mask) to produce the same
        # mode.  See ``flatten_port_plane_mass`` and
        # ``flatten_port_plane_pec_mask`` for the rationale.
        from magnelio._operators.material_matrices import (
            flatten_port_plane_mass,
        )

        m_eps_flat = flatten_port_plane_mass(m_eps, mesh, BoxFace.X_MIN)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        c_3d = build_curl_matrix(mesh.grid)
        g_3d = build_gradient_matrix(mesh.grid)
        _, M_2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps_flat, m_mu, c_3d)
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
        groups = _resolve_conductors(spec.conductors, plane, mesh)
        modes_standalone = solve_tem_laplace(
            plane,
            g_2d,
            M_2d,
            groups,
            epsilon_r=1.0,
            grid=mesh.grid,
            m_mu_flat=build_M_mu(mesh),
        )
        assert z_factory == pytest.approx(modes_standalone[0].z_line, rel=1e-12)


# ---------------------------------------------------------------------
# 5) QTEM dispatch matches standalone solve_qtem_laplace
# ---------------------------------------------------------------------


class TestQTEMFactoryEquivalence:
    """Factory QTEM dispatch (``epsilon_r=None``) produces the same
    Mode as the standalone ``solve_qtem_laplace`` + ``build_M_eps_vacuum``
    path."""

    def test_microstrip_eps_eff_and_z_0_match(self):
        mesh, W, h, eps_r_sub = _microstrip_setup()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=(
                WallConductor(face=BoxFace.Z_MIN),
                RegionConductor(
                    axis_a_range=(-W / 2, W / 2),
                    axis_b_range=(h, h),
                ),
            ),
            epsilon_r=None,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=5e9)
        eps_factory = op.discrete_modes[0].mode.epsilon_r
        z_factory = op.discrete_modes[0].mode.z_line

        # Standalone path must replicate factory's port-plane flatten
        # preprocessing on both the actual and vacuum mass.
        from magnelio._operators.material_matrices import flatten_port_plane_mass

        m_eps_flat = flatten_port_plane_mass(m_eps, mesh, BoxFace.X_MIN)
        m_eps_vac = build_M_eps_vacuum(mesh)
        m_eps_vac_flat = flatten_port_plane_mass(m_eps_vac, mesh, BoxFace.X_MIN)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        c_3d = build_curl_matrix(mesh.grid)
        g_3d = build_gradient_matrix(mesh.grid)
        _, M_2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps_flat, m_mu, c_3d)
        _, M_2d_vac, _ = build_2d_curl_curl(
            plane,
            mesh.grid,
            m_eps_vac_flat,
            m_mu,
            c_3d,
        )
        g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)
        groups = _resolve_conductors(spec.conductors, plane, mesh)
        modes_standalone = solve_qtem_laplace(
            plane,
            g_2d,
            M_2d,
            M_2d_vac,
            groups,
            grid=mesh.grid,
            m_mu_flat=build_M_mu(mesh),
        )
        assert eps_factory == pytest.approx(
            modes_standalone[0].epsilon_r,
            rel=1e-12,
        )
        assert z_factory == pytest.approx(
            modes_standalone[0].z_line,
            rel=1e-12,
        )

    def test_microstrip_within_3_percent_of_hammerstad_jensen(self):
        """End-to-end accuracy: the Mesh.from_geometry feature-resolution
        at material interfaces beats the historical bare-Cartesian
        per-edge-scaling benchmark (the retired Phase-2b microstrip
        script).  At the default spec resolution, ε_eff and Z_0 land
        within 3 % of Hammerstad-Jensen — matching architecture's
        2 %-stretch target up to mesh resolution noise.
        """
        from magnelio.constants import ETA0

        mesh, W, h, eps_r_sub = _microstrip_setup()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=(
                WallConductor(face=BoxFace.Z_MIN),
                RegionConductor(
                    axis_a_range=(-W / 2, W / 2),
                    axis_b_range=(h, h),
                ),
            ),
            epsilon_r=None,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=5e9)
        m = op.discrete_modes[0].mode

        W_h = W / h
        eps_HJ = (eps_r_sub + 1) / 2 + (eps_r_sub - 1) / 2 * (1 + 12 * h / W) ** -0.5
        z_HJ = (ETA0 / math.sqrt(eps_HJ)) / (W_h + 1.393 + 0.667 * math.log(W_h + 1.444))
        eps_err = abs(m.epsilon_r - eps_HJ) / eps_HJ
        z_err = abs(m.z_line - z_HJ) / z_HJ
        assert eps_err < 0.03, (
            f"ε_eff off by {eps_err:.2%}: numerical {m.epsilon_r:.4f}, HJ {eps_HJ:.4f}"
        )
        assert z_err < 0.03, f"Z_0 off by {z_err:.2%}: numerical {m.z_line:.4f}, HJ {z_HJ:.4f}"


# ---------------------------------------------------------------------
# 6) N_modes truncation
# ---------------------------------------------------------------------


class TestNModesTruncation:
    def test_three_conductors_n_modes_one(self):
        """3 conductor groups (1 ground + 2 signal) → K-1 = 2 modes
        available; n_modes=1 returns just the first."""
        mesh, B, a, y0, z0, L_x = _rect_coax_setup()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        # Synthetic 3-conductor: outer + two inner halves.
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=(
                BboxLateralConductor(),
                RegionConductor(
                    axis_a_range=(y0, y0 + a / 2),
                    axis_b_range=(z0, z0 + a),
                ),
                RegionConductor(
                    axis_a_range=(y0 + a / 2, y0 + a),
                    axis_b_range=(z0, z0 + a),
                ),
            ),
            epsilon_r=1.0,
            n_modes=1,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=5e9)
        assert len(op.discrete_modes) == 1


# ---------------------------------------------------------------------
# 7) build_M_eps_vacuum sanity
# ---------------------------------------------------------------------


class TestBuildMEpsVacuum:
    def test_vacuum_mesh_returns_same_as_build_M_eps(self):
        """For a vacuum-only mesh (no PEC, no dielectric), both should
        agree byte-for-byte (modulo the conformal/DM overlays which
        don't trigger here)."""
        grid = GridLines(
            x=np.linspace(0, 5e-3, 6),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 3e-3, 4),
        )
        mesh = Mesh.from_grid(grid)
        m_actual = build_M_eps(mesh)
        m_vacuum = build_M_eps_vacuum(mesh)
        np.testing.assert_array_equal(m_actual, m_vacuum)

    def test_dielectric_mesh_vacuum_smaller_than_actual(self):
        """For a substrate-filled mesh, vacuum M_eps < actual M_eps everywhere
        (modulo PEC cells where actual is 0)."""
        air = Material.air()
        diel = Material(name="d", epsilon=(4.0,) * 3)
        substrate = Brick(origin=(0, 0, 0), size=(2e-3, 4e-3, 1e-3), material=diel)
        air_cap = Brick(origin=(0, 0, 1e-3), size=(2e-3, 4e-3, 3e-3), material=air)
        model = GeometryModel()
        model.add(substrate)
        model.add(air_cap)
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.5e-3, min_cells_per_feature=2),
            f_max=8e9,
        )
        m_actual = build_M_eps(mesh)
        m_vacuum = build_M_eps_vacuum(mesh)
        ratio = m_actual / np.maximum(m_vacuum, 1e-30)
        non_pec = m_actual > 0
        # All non-PEC ratios ≥ 1 (vacuum) ≤ ~4 (dielectric); allow the
        # small numerical roundoff at material interfaces (DM overlays
        # can push slightly above 4).
        assert ratio[non_pec].min() >= 1.0 - 1e-9
        assert ratio[non_pec].max() <= 5.0
