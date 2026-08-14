"""Unit tests for GridLines and MeshControl."""

import numpy as np
import pytest

from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import (
    MeshControl,
    _generate_axis_lines,
    _merge_axis_planes,
    _n_one_sided,
    _one_sided_subdivision,
)


class TestGridLines:
    def test_basic_creation(self):
        x = np.linspace(0, 1e-2, 11)
        y = np.linspace(0, 5e-3, 6)
        z = np.linspace(0, 2e-3, 5)
        grid = GridLines(x=x, y=y, z=z)
        assert grid.Nx == 10
        assert grid.Ny == 5
        assert grid.Nz == 4

    def test_cell_sizes(self):
        x = np.array([0.0, 1e-3, 3e-3, 6e-3])
        y = np.array([0.0, 1e-3, 2e-3])
        z = np.array([0.0, 1e-3])
        grid = GridLines(x=x, y=y, z=z)
        np.testing.assert_allclose(grid.dx, [1e-3, 2e-3, 3e-3])
        assert grid.dx_min == pytest.approx(1e-3)

    def test_non_monotonic_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            GridLines(x=np.array([0, 2e-3, 1e-3]), y=np.array([0, 1]), z=np.array([0, 1]))

    def test_min_two_nodes(self):
        with pytest.raises(ValueError):
            GridLines(x=np.array([0.0]), y=np.array([0, 1]), z=np.array([0, 1]))

    def test_n_cells(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 3)
        grid = GridLines(x=x, y=y, z=z)
        assert grid.n_cells == 4 * 3 * 2

    def test_courant_dt(self):
        x = np.linspace(0, 1e-2, 11)  # dx = 1e-3
        y = np.linspace(0, 1e-2, 11)
        z = np.linspace(0, 1e-2, 11)
        grid = GridLines(x=x, y=y, z=z)
        c0 = 299_792_458.0
        expected = 1.0 / (c0 * np.sqrt(3) / 1e-3)
        assert grid.courant_dt_max == pytest.approx(expected, rel=1e-6)


class TestMeshControl:
    def test_defaults(self):
        ctrl = MeshControl()
        assert ctrl.min_nodes_per_wavelength == 20
        assert ctrl.growth_factor == 1.3

    def test_growth_factor_must_exceed_one(self):
        with pytest.raises(ValueError):
            MeshControl(growth_factor=0.9)

    def test_forced_planes(self):
        ctrl = MeshControl(forced_planes={"z": [0.0, 1.6e-3]})
        assert ctrl.forced_planes["z"] == [0.0, 1.6e-3]


# ── _graded_subdivision ───────────────────────────────────────────────────────


class TestGradedSubdivision:
    """Tests for the symmetric graded mesh subdivision helper."""

    def _sub(self, p0, p1, n, g):
        from magnelio.mesh.mesher import _graded_subdivision

        return _graded_subdivision(p0, p1, n, g)

    def test_n1_trivial(self):
        nodes = self._sub(0.0, 1.0, 1, 1.3)
        assert nodes == pytest.approx([0.0, 1.0])

    def test_uniform_g1(self):
        nodes = self._sub(0.0, 1.0, 4, 1.0)
        assert nodes == pytest.approx(np.linspace(0, 1, 5).tolist())

    def test_sum_equals_interval(self):
        """Cell widths must sum exactly to the interval."""
        for n in (2, 3, 4, 5, 6, 7):
            nodes = self._sub(0.0, 1.0, n, 1.3)
            assert len(nodes) == n + 1
            assert sum(np.diff(nodes)) == pytest.approx(1.0, rel=1e-10)

    def test_endpoints_exact(self):
        nodes = self._sub(2e-3, 8e-3, 6, 1.4)
        assert nodes[0] == pytest.approx(2e-3, abs=1e-15)
        assert nodes[-1] == pytest.approx(8e-3, abs=1e-15)

    def test_growth_ratio_left(self):
        """First two cell widths must have ratio ≈ g."""
        g = 1.3
        nodes = self._sub(0.0, 1.0, 6, g)
        widths = np.diff(nodes)
        assert widths[1] / widths[0] == pytest.approx(g, rel=1e-6)

    def test_growth_ratio_right(self):
        """Last two cell widths must have ratio ≈ 1/g (shrinking toward p1)."""
        g = 1.3
        nodes = self._sub(0.0, 1.0, 6, g)
        widths = np.diff(nodes)
        assert widths[-1] / widths[-2] == pytest.approx(1.0 / g, rel=1e-6)

    def test_symmetric_even(self):
        """For even n: first cell width == last cell width."""
        nodes = self._sub(0.0, 1.0, 4, 1.5)
        widths = np.diff(nodes)
        assert widths[0] == pytest.approx(widths[-1], rel=1e-10)

    def test_symmetric_odd(self):
        """For odd n: first cell width == last cell width."""
        nodes = self._sub(0.0, 1.0, 5, 1.3)
        widths = np.diff(nodes)
        assert widths[0] == pytest.approx(widths[-1], rel=1e-10)

    def test_centre_cell_largest_even(self):
        """The two central cells should be the largest."""
        nodes = self._sub(0.0, 1.0, 6, 1.4)
        widths = np.diff(nodes)
        assert widths[2] == pytest.approx(max(widths), rel=1e-10)

    def test_centre_cell_largest_odd(self):
        """The single central cell should be the largest."""
        nodes = self._sub(0.0, 1.0, 5, 1.4)
        widths = np.diff(nodes)
        assert widths[2] == pytest.approx(max(widths), rel=1e-10)

    def test_n2_two_equal_cells(self):
        """n=2 with grading: symmetric means two equal cells."""
        nodes = self._sub(0.0, 2.0, 2, 1.5)
        widths = np.diff(nodes)
        assert widths[0] == pytest.approx(widths[1], rel=1e-10)


# ── _one_sided_subdivision / _n_one_sided ─────────────────────────────────────


class TestOneSidedSubdivision:
    """Tests for the one-sided graded mesh subdivision helpers."""

    def test_sum_equals_interval_fine_at_left(self):
        nodes = _one_sided_subdivision(0.0, 1.0, 5, 1.3)
        assert sum(np.diff(nodes)) == pytest.approx(1.0, rel=1e-10)

    def test_sum_equals_interval_fine_at_right(self):
        # p_fine = 1.0 (right), p_coarse = 0.0 (left)
        nodes = _one_sided_subdivision(1.0, 0.0, 5, 1.3)
        assert sum(np.diff(nodes)) == pytest.approx(1.0, rel=1e-10)

    def test_endpoints_fine_at_left(self):
        nodes = _one_sided_subdivision(0.0, 2e-3, 4, 1.3)
        assert nodes[0] == pytest.approx(0.0, abs=1e-15)
        assert nodes[-1] == pytest.approx(2e-3, abs=1e-15)

    def test_endpoints_fine_at_right(self):
        nodes = _one_sided_subdivision(2e-3, 0.0, 4, 1.3)
        assert nodes[0] == pytest.approx(0.0, abs=1e-15)
        assert nodes[-1] == pytest.approx(2e-3, abs=1e-15)

    def test_result_ascending_fine_at_left(self):
        nodes = _one_sided_subdivision(0.0, 1.0, 6, 1.3)
        assert all(nodes[i] < nodes[i + 1] for i in range(len(nodes) - 1))

    def test_result_ascending_fine_at_right(self):
        nodes = _one_sided_subdivision(1.0, 0.0, 6, 1.3)
        assert all(nodes[i] < nodes[i + 1] for i in range(len(nodes) - 1))

    def test_widths_monotone_growing_fine_at_left(self):
        """Cells grow from p_fine=left toward p_coarse=right."""
        nodes = _one_sided_subdivision(0.0, 1.0, 6, 1.3)
        widths = np.diff(nodes)
        assert all(widths[i] <= widths[i + 1] for i in range(len(widths) - 1))

    def test_widths_monotone_growing_fine_at_right(self):
        """Cells grow from p_fine=right toward p_coarse=left, so widths
        are decreasing left→right."""
        nodes = _one_sided_subdivision(1.0, 0.0, 6, 1.3)
        widths = np.diff(nodes)
        assert all(widths[i] >= widths[i + 1] for i in range(len(widths) - 1))

    def test_smallest_cell_below_h_fine(self):
        h_fine = 0.4e-3
        nodes = _one_sided_subdivision(0.0, 6e-3, _n_one_sided(6e-3, h_fine, 1.3), 1.3)
        assert np.diff(nodes)[0] <= h_fine + 1e-15

    def test_n1_trivial(self):
        nodes = _one_sided_subdivision(0.0, 1.0, 1, 1.3)
        assert nodes == pytest.approx([0.0, 1.0])

    def test_uniform_g1(self):
        nodes = _one_sided_subdivision(0.0, 1.0, 4, 1.0)
        assert nodes == pytest.approx(np.linspace(0, 1, 5).tolist())


class TestAxisLinesGrading:
    """Verify the two-scale grading: h_fine at interfaces, h_max in bulk."""

    def _ctrl(self, g=1.3):
        return MeshControl(min_nodes_per_wavelength=20, growth_factor=g)

    def test_first_interval_coarse_at_left_wall(self):
        """Domain wall at p0=0 must be coarser than interior interface at p1."""
        ctrl = self._ctrl()
        # 20 mm boundary interval, h_max = 2 mm, h_fine = 0.2 mm
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 40e-3],
            h_max=2e-3,
            h_fine=0.2e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        j_iface = int(np.argmin(np.abs(np.array(nodes) - 20e-3)))
        cell_at_wall = widths[0]
        cell_at_iface = widths[j_iface - 1]
        # Order-of-magnitude separation, not floating-point luck
        assert cell_at_wall > cell_at_iface * 5

    def test_last_interval_coarse_at_right_wall(self):
        ctrl = self._ctrl()
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 40e-3],
            h_max=2e-3,
            h_fine=0.2e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        j_iface = int(np.argmin(np.abs(np.array(nodes) - 20e-3)))
        cell_at_right_wall = widths[-1]
        cell_at_iface = widths[j_iface]
        assert cell_at_right_wall > cell_at_iface * 5

    def test_fine_cell_at_interface(self):
        """The first cell after the interior interface is ≈ h_fine."""
        ctrl = self._ctrl()
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 40e-3],
            h_max=2e-3,
            h_fine=0.2e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        j_iface = int(np.argmin(np.abs(np.array(nodes) - 20e-3)))
        # Cells touching the interface, on both sides
        assert widths[j_iface - 1] == pytest.approx(0.2e-3, rel=0.5)
        assert widths[j_iface] == pytest.approx(0.2e-3, rel=0.5)

    def test_bulk_cells_reach_h_max(self):
        """Cells far from any interface should be close to h_max."""
        ctrl = self._ctrl()
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 40e-3],
            h_max=2e-3,
            h_fine=0.2e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        # The largest cell width should be close to h_max
        assert max(widths) == pytest.approx(2e-3, rel=0.3)
        # And reasonably many cells should be at the bulk size
        bulk_cells = [w for w in widths if w > 1.5e-3]
        assert len(bulk_cells) >= 5

    def test_interior_interval_symmetric(self):
        """Middle interval: first and last cells must be equal (symmetric grading)."""
        ctrl = self._ctrl()
        # Four planes — middle interval [2 mm, 18 mm], wide enough for grading
        nodes = _generate_axis_lines(
            [0.0, 2e-3, 18e-3, 20e-3],
            h_max=1e-3,
            h_fine=0.1e-3,
            control=ctrl,
        )
        mid_nodes = [n for n in nodes if 2e-3 <= n <= 18e-3]
        widths = np.diff(mid_nodes)
        assert widths[0] == pytest.approx(widths[-1], rel=1e-6)
        # And both ends are close to h_fine
        assert widths[0] == pytest.approx(0.1e-3, rel=0.5)

    def test_single_interval_uniform(self):
        """Two planes only → single interval, no interior interface → uniform."""
        ctrl = self._ctrl()
        nodes = _generate_axis_lines(
            [0.0, 5e-3],
            h_max=0.5e-3,
            h_fine=0.5e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        assert max(widths) == pytest.approx(min(widths), rel=1e-10)

    def test_h_fine_equals_h_max_yields_uniform(self):
        """When h_fine == h_max, every interval is uniform with that size."""
        ctrl = self._ctrl()
        nodes = _generate_axis_lines(
            [0.0, 1.6e-3, 4.8e-3],
            h_max=0.4e-3,
            h_fine=0.4e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        assert max(widths) == pytest.approx(min(widths), rel=1e-9)
        assert max(widths) == pytest.approx(0.4e-3, rel=1e-9)

    def test_no_backward_jump_at_boundary(self):
        """Cells must grow monotonically toward the domain wall — no
        ramp peaking high then collapsing back to small uniform cells.

        Concretely: every consecutive pair of cells must satisfy
        ``ratio <= growth_factor`` (with a small numeric slack), in
        either direction across the whole axis.
        """
        ctrl = self._ctrl(g=1.3)
        # Mimic the WR-90/coax x-axis: 9.4 mm bulk, 1.42/1.27/1.42 mm
        # interior gaps, 9.4 mm bulk again.
        planes = [-11.43e-3, -2.055e-3, -0.635e-3, 0.635e-3, 2.055e-3, 11.43e-3]
        nodes = _generate_axis_lines(
            planes,
            h_max=1.676e-3,
            h_fine=0.317e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        ratios = widths[1:] / widths[:-1]
        max_ratio = max(np.max(ratios), np.max(1.0 / ratios))
        # Allow up to 1.6× (= growth_factor with minor tail-merge slack)
        assert max_ratio < 1.6, f"max consecutive ratio {max_ratio:.3f} too large"


# ── _merge_axis_planes (WP-M1 unified plane clustering) ──────────────────────


class TestMergeAxisPlanes:
    """Forced planes are verbatim anchors; critical planes snap onto them."""

    TOL = 1e-6

    def test_no_forced_matches_snap_planes(self):
        critical = [0.0, 0.5e-6, 5e-3, 10e-3]
        planes, is_material = _merge_axis_planes(critical, [], self.TOL)
        # First two cluster to their midpoint, rest verbatim.
        assert planes == [0.25e-6, 5e-3, 10e-3]
        assert is_material == [True, True, True]

    def test_critical_snaps_onto_anchor_bit_exactly(self):
        anchor = 5e-3
        wiggled = 5e-3 + 3e-7  # within tol of the anchor
        planes, is_material = _merge_axis_planes(
            [0.0, wiggled, 10e-3],
            [anchor],
            self.TOL,
        )
        assert anchor in planes  # anchor survives bit-exactly
        assert wiggled not in planes  # wiggled critical is gone
        assert planes == [0.0, anchor, 10e-3]
        # The anchor inherited the material flag from the snapped plane.
        assert is_material[planes.index(anchor)] is True

    def test_forced_only_plane_is_not_material(self):
        planes, is_material = _merge_axis_planes(
            [0.0, 10e-3],
            [5e-3],
            self.TOL,
        )
        assert planes == [0.0, 5e-3, 10e-3]
        assert is_material == [True, False, True]

    def test_multiple_criticals_snap_onto_same_anchor(self):
        anchor = 1e-3
        planes, is_material = _merge_axis_planes(
            [anchor - 4e-7, anchor + 4e-7, 5e-3],
            [anchor, 0.0],
            self.TOL,
        )
        assert planes == [0.0, anchor, 5e-3]
        assert is_material == [False, True, True]

    def test_forced_forced_pair_below_tol_warns_and_keeps_both(self):
        a, b = 1e-3, 1e-3 + 5e-7
        with pytest.warns(UserWarning, match="closer than"):
            planes, _ = _merge_axis_planes([], [a, b], self.TOL)
        assert planes == [a, b]

    def test_free_criticals_still_cluster_to_midpoint(self):
        # Two criticals within tol of each other but far from the anchor.
        planes, _ = _merge_axis_planes(
            [2e-3, 2e-3 + 4e-7],
            [0.0],
            self.TOL,
        )
        assert len(planes) == 2
        assert planes[0] == 0.0
        assert planes[1] == pytest.approx(2e-3 + 2e-7, abs=1e-12)

    def test_tol_zero_passthrough(self):
        planes, is_material = _merge_axis_planes(
            [0.0, 1e-9, 5e-3],
            [5e-3, 9e-3],
            0.0,
        )
        # Exact duplicate (5e-3) dedupes onto the anchor; nothing else moves.
        assert planes == [0.0, 1e-9, 5e-3, 9e-3]
        assert is_material == [True, True, True, False]

    def test_result_strictly_ascending(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            critical = sorted(rng.uniform(0, 10e-3, size=20))
            forced = sorted(rng.uniform(0, 10e-3, size=5))
            planes, flags = _merge_axis_planes(critical, forced, self.TOL)
            arr = np.asarray(planes)
            assert np.all(np.diff(arr) > 0)
            assert len(flags) == len(planes)
            for f in forced:
                assert f in planes  # anchors always survive verbatim


# ── Plane provenance: face planes outrank bbox extents (KB-013) ───────────────


class TestPlaneProvenance:
    """A cluster snaps onto its face planes; bbox extents cannot drag it.

    OCCT Booleans on interpenetrating operands return a bounding box
    inflated by ``Precision::Confusion`` (1e-7 model units) beyond the
    true geometry.  Averaging that phantom extent with the material
    face put the domain boundary 50 nm past the geometry, whose sliver
    fill factor (1.11e-5) failed the DTBC slab certificate and sent
    every port channel on that face to Mur-1st (KB-013).
    """

    TOL = 1e-6
    FACE = 75e-3
    BBOX = 75e-3 + 1e-7  # Precision::Confusion inflation

    def test_mixed_cluster_snaps_onto_the_face_plane(self):
        planes, is_material = _merge_axis_planes(
            [(0.0, True), (self.FACE, True), (self.BBOX, False)],
            [],
            self.TOL,
        )
        assert planes == [0.0, self.FACE]  # bit-exact, no midpoint shift
        assert is_material == [True, True]

    def test_bbox_only_cluster_keeps_the_midpoint(self):
        lo, hi = 5e-3, 5e-3 + 4e-7
        planes, _ = _merge_axis_planes([(lo, False), (hi, False)], [], self.TOL)
        assert planes == [pytest.approx(0.5 * (lo + hi), abs=1e-15)]

    def test_two_face_planes_midpoint_among_faces_only(self):
        a, b = self.FACE - 2e-7, self.FACE + 2e-7
        planes, _ = _merge_axis_planes(
            [(a, True), (b, True), (self.BBOX + 4e-7, False)],
            [],
            self.TOL,
        )
        assert planes == [pytest.approx(0.5 * (a + b), abs=1e-15)]

    def test_interpenetrating_union_grid_ends_on_the_face(self):
        """End-to-end: the coupler's failure geometry, minimal.

        A side stub unioned into a main cylinder it interpenetrates —
        the union's bbox ymax is inflated beyond the stub's analytic
        end face, and the grid (with it the domain boundary) must land
        on the face, not between face and phantom extent.
        """
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Cylinder, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        air = Material.air()
        y_end = 75e-3
        vac = Cylinder(
            radius=45e-3, origin=(0.0, 0.0, -30e-3), axis="z", height=60e-3, material=air
        )
        stub = Cylinder(origin=(0.0, 0.0, 0.0), axis="y", height=y_end, radius=3.5e-3, material=air)
        vac += stub

        model = GeometryModel(background=Material.pec())
        model.add(vac)
        mesh = Mesh.from_geometry(model, MeshControl(), f_max=1e9)
        assert abs(float(mesh.grid.y[-1]) - y_end) < 1e-12


# ── Forced-plane anchoring end-to-end (DD-058 sliver factory) ─────────────────


class TestForcedPlaneAnchoring:
    """CSG tangent planes colliding with forced nodes must not produce slivers."""

    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepPrimAPI")

    def test_cylinder_tangent_on_forced_node_no_sliver(self):
        """Session-91 collision: cylinder tangent plane vs forced grid node.

        R = 10 mm tangent planes (±R on y/z) land within float distance of
        the forced 1-mm-grid nodes; before WP-M1 the verbatim union produced
        ~1e-18 m cells whose degenerate faces poison M_mu (DD-058).
        """
        self._occ()
        from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        R = 10e-3
        s_bbox = 24e-3
        pec = Material.pec()
        vacuum = Material.air()

        bbox = Brick(
            origin=(0.0, -s_bbox / 2, -s_bbox / 2),
            size=(4e-3, s_bbox, s_bbox),
            material=pec,
        )
        cavity = Cylinder(
            origin=(0.0, 0.0, 0.0),
            radius=R,
            height=4e-3,
            axis="x",
            material=vacuum,
        )
        model = GeometryModel()
        model.add(Difference(bbox, cavity))
        model.add(cavity)

        # Forced nodes on a 1 mm grid: ±10 mm coincide with ±R tangents.
        t_nodes = np.linspace(-s_bbox / 2, s_bbox / 2, 25).tolist()
        control = MeshControl(
            min_nodes_per_wavelength=8,
            min_cells_per_feature=0,
            growth_factor=1.5,
            max_cell_size=4e-3,
            forced_planes={"y": t_nodes, "z": t_nodes},
        )
        mesh = Mesh.from_geometry(model, control, f_max=14e9)

        d_min = min(
            mesh.grid.dx.min(),
            mesh.grid.dy.min(),
            mesh.grid.dz.min(),
        )
        assert d_min > 1e-6, (
            f"sliver cell (d_min = {d_min:.3e} m): tangent plane survived next to a forced node"
        )
        # Every forced node survives bit-exactly (the linspace values,
        # which differ from the ±R literals in the last bit — exactly
        # the collision this test exercises).
        for t in t_nodes:
            assert t in mesh.grid.y
            assert t in mesh.grid.z

    def test_forced_plane_wins_over_wiggled_brick_face(self):
        """A brick face 1e-10 m off a forced plane snaps onto it bit-exactly."""
        self._occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        forced = 1.0e-3
        wiggle = 1e-10
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, forced + wiggle), material=fr4))
        ctrl = MeshControl(
            min_nodes_per_wavelength=10,
            forced_planes={"z": [forced]},
        )
        mesh = Mesh.from_geometry(m, ctrl, f_max=10e9)
        assert forced in mesh.grid.z
        assert mesh.grid.dz.min() > 1e-6


# ── WP-M3: hard min_cell_size ─────────────────────────────────────────────────


class TestFloorMergePlanes:
    """_floor_merge_planes — the WP-M3 (a) merge stage."""

    FLOOR = 100e-6

    def _merge(self, planes, anchors=()):
        from magnelio.mesh.mesher import _floor_merge_planes

        flags = [True] * len(planes)
        kept, kept_flags, _ = _floor_merge_planes(
            planes,
            flags,
            anchors,
            self.FLOOR,
        )
        return kept, kept_flags

    def test_keep_first_in_run(self):
        planes, flags = self._merge([0.0, 80e-6, 0.5e-3, 1e-3])
        assert planes == [0.0, 0.5e-3, 1e-3]
        assert flags == [True, True, True]

    def test_anchor_wins_over_nearby_critical(self):
        anchor = 0.5e-3
        planes, _ = self._merge(
            [0.0, anchor - 60e-6, anchor, 1e-3],
            anchors=[anchor],
        )
        assert planes == [0.0, anchor, 1e-3]

    def test_domain_end_always_survives(self):
        # 240 um chain at a 100 um floor: keep-first keeps 0 and 160,
        # the end rule then sacrifices 160 for the domain-end plane.
        planes, _ = self._merge([0.0, 80e-6, 160e-6, 240e-6])
        assert planes[0] == 0.0
        assert planes[-1] == 240e-6
        d = np.diff(planes)
        assert np.all(d >= self.FLOOR * (1 - 1e-9))

    def test_anchor_pair_below_floor_warns_and_survives(self):
        a, b = 0.5e-3, 0.5e-3 + 40e-6
        with pytest.warns(UserWarning, match="closer than"):
            planes, _ = self._merge([0.0, a, b, 1e-3], anchors=[a, b])
        assert a in planes and b in planes

    def test_result_respects_floor(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            raw = np.sort(rng.uniform(0, 5e-3, size=25))
            planes, flags = self._merge(list(raw))
            d = np.diff(planes)
            assert np.all(d >= self.FLOOR * (1 - 1e-9))
            assert planes[0] == raw[0]
            assert planes[-1] == raw[-1]
            assert len(flags) == len(planes)


class TestHardMinCellSize:
    """WP-M3 acceptance: no generated cell below the floor."""

    def test_axis_lines_property(self):
        """Randomised control parameters: every cell >= min_cell_size."""
        rng = np.random.default_rng(1234)
        for _ in range(300):
            n_planes = int(rng.integers(2, 8))
            planes = np.sort(rng.uniform(0.0, 20e-3, size=n_planes))
            # Pre-merge the planes as from_geometry would (the floor
            # merge guarantees interval >= floor).
            floor = float(rng.uniform(20e-6, 1e-3))
            from magnelio.mesh.mesher import _floor_merge_planes

            merged, _, _ = _floor_merge_planes(
                list(planes),
                [True] * len(planes),
                (),
                floor,
            )
            if len(merged) < 2:
                continue
            ctrl = MeshControl(
                growth_factor=float(rng.uniform(1.05, 2.5)),
                min_cell_size=floor,
            )
            h_max = float(rng.uniform(floor, 5e-3))
            h_fine = float(rng.uniform(0.1 * floor, h_max))
            nodes = np.asarray(
                _generate_axis_lines(
                    merged,
                    h_max=h_max,
                    h_fine=h_fine,
                    control=ctrl,
                )
            )
            d = np.diff(nodes)
            assert np.all(d > 0), "non-monotone or zero interval"
            assert np.all(d >= floor * (1 - 1e-9)), (
                f"sub-floor cell {d.min():.3e} at floor {floor:.3e} "
                f"(g={ctrl.growth_factor:.3f}, h_max={h_max:.3e}, "
                f"h_fine={h_fine:.3e})"
            )
            # Endpoints survive exactly.
            assert nodes[0] == merged[0]
            assert nodes[-1] == merged[-1]

    def test_from_geometry_property(self):
        """Randomised brick stacks: the meshed grid respects the floor."""
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        rng = np.random.default_rng(99)
        for _ in range(8):
            floor = float(rng.uniform(50e-6, 400e-6))
            model = GeometryModel(allow_overlaps=True)
            n_bricks = int(rng.integers(2, 5))
            for b in range(n_bricks):
                origin = rng.uniform(0.0, 4e-3, size=3)
                size = rng.uniform(0.2e-3, 6e-3, size=3)
                eps = float(rng.uniform(1.0, 10.0))
                model.add(
                    Brick(
                        origin=tuple(origin),
                        size=tuple(size),
                        material=Material(name=f"m{b}", epsilon=(eps,) * 3),
                    )
                )
            ctrl = MeshControl(
                min_cells_per_feature=int(rng.integers(0, 5)),
                growth_factor=float(rng.uniform(1.1, 2.0)),
                min_cell_size=floor,
                conformal=False,
            )
            mesh = Mesh.from_geometry(model, ctrl, f_max=10e9)
            for d in (mesh.grid.dx, mesh.grid.dy, mesh.grid.dz):
                assert np.all(d >= floor * (1 - 1e-9)), (
                    f"sub-floor cell {d.min():.3e} at floor {floor:.3e}"
                )

    def test_absorbed_plane_gets_harmonic_eps(self):
        """Edges crossing a floor-absorbed dielectric boundary carry the
        length-weighted harmonic (series) eps, not the transverse
        dual-face average (WP-M3 longitudinal correction)."""
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        h1, eps1 = 0.5e-3, 4.3
        h2, eps2 = 60e-6, 8.0
        h3 = 1.0e-3
        w = 2.0e-3
        floor = 100e-6

        m = GeometryModel()
        m.add(
            Brick(
                origin=(0, 0, 0),
                size=(1e-3, w, h1),
                material=Material(name="d1", epsilon=(eps1,) * 3),
            )
        )
        m.add(
            Brick(
                origin=(0, 0, h1),
                size=(1e-3, w, h2),
                material=Material(name="d2", epsilon=(eps2,) * 3),
            )
        )
        m.add(Brick(origin=(0, 0, h1 + h2), size=(1e-3, w, h3), material=Material.air()))
        ctrl = MeshControl(
            min_cells_per_feature=2,
            growth_factor=1.3,
            max_cell_size=floor,
            min_cell_size=floor,
        )
        mesh = Mesh.from_geometry(m, ctrl, f_max=10e9)

        gz = np.asarray(mesh.grid.z)
        # Keep-first: the lower layer face survives, the upper is absorbed.
        assert h1 in gz
        assert not np.any(np.abs(gz - (h1 + h2)) < 1e-9)

        k = int(np.argmin(np.abs(gz - h1)))
        dz = gz[k + 1] - gz[k]
        assert gz[k + 1] > h1 + h2  # the crossing cell spans the layer
        expected = dz / (h2 / eps2 + (dz - h2) / 1.0)

        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        em = mesh.edge_material
        eps_avg = em.eps_avg[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)
        sel = eps_avg[1:-1, 1:-1, k]
        np.testing.assert_allclose(sel, expected, rtol=1e-3)

    def test_session_91_refit_case_restored(self):
        """The measured 91.3/70.2 um ramp cells at a 100 um floor."""
        # Interior interval wide enough for the symmetric refit to
        # produce sub-floor ramp cells before WP-M3.
        ctrl = MeshControl(growth_factor=1.3, min_cell_size=100e-6)
        nodes = np.asarray(
            _generate_axis_lines(
                [0.0, 0.635e-3, 3.635e-3],
                h_max=1.4e-3,
                h_fine=100e-6,
                control=ctrl,
            )
        )
        d = np.diff(nodes)
        assert np.all(d >= 100e-6 * (1 - 1e-9))


# ── WP-M4: helper audit ───────────────────────────────────────────────────────


class TestPerAxisHFine:
    """h_fine is per axis: a small gap on z must not refine x/y."""

    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepPrimAPI")

    def test_z_gap_does_not_refine_x(self):
        self._occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        # Two thin z-layers (0.5 mm gap) inside a wide 20 mm domain;
        # no interior x/y features.
        fr4 = Material(name="FR4", epsilon=(4.4,) * 3)
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(20e-3, 20e-3, 0.5e-3), material=fr4))
        m.add(Brick(origin=(0, 0, 0.5e-3), size=(20e-3, 20e-3, 5e-3), material=Material.air()))
        ctrl = MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=4)
        mesh = Mesh.from_geometry(m, ctrl, f_max=10e9)

        # z is feature-refined (0.5 mm / 4 = 125 um cells near the
        # interface); x has no interior features and must stay at the
        # wavelength scale — nowhere near the z feature size.
        assert mesh.grid.dz.min() < 0.2e-3
        assert mesh.grid.dx.min() > 0.5e-3


class TestPMLExtension:
    """PML grid extension: depth, uniformity, material continuation."""

    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepPrimAPI")

    def _mesh(self, pml_faces):
        from magnelio.boundaries.boundary_conditions import BoundaryConditions
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        fr4 = Material(name="FR4", epsilon=(4.4,) * 3)
        m = GeometryModel(
            boundary_conditions=BoundaryConditions(
                cpml_thickness_cells=8,
                **{f: "CPML" for f in (pml_faces or ())},
            )
        )
        m.add(Brick(origin=(0, 0, 0), size=(5e-3, 5e-3, 2e-3), material=fr4))
        m.add(Brick(origin=(0, 0, 2e-3), size=(5e-3, 5e-3, 8e-3), material=Material.air()))
        ctrl = MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=2)
        return Mesh.from_geometry(m, ctrl, f_max=10e9)

    def test_zmax_extension_uniform_and_deep_enough(self):
        self._occ()
        mesh_ref = self._mesh(None)
        mesh_pml = self._mesh(["zmax"])
        n_pml = mesh_pml._pml_cells["zmax"]
        assert n_pml >= 8
        assert mesh_pml.Nz == mesh_ref.Nz + n_pml
        # Extension cells are uniform at the boundary cell width.
        dz = mesh_pml.grid.dz
        ext = dz[-n_pml:]
        np.testing.assert_allclose(ext, ext[0], rtol=1e-9)
        assert ext[0] == pytest.approx(dz[-n_pml - 1], rel=1e-9)

    def test_material_continued_into_pml(self):
        self._occ()
        mesh_pml = self._mesh(["zmax"])
        n_pml = mesh_pml._pml_cells["zmax"]
        interior = mesh_pml.material_id[:, :, -n_pml - 1]
        for off in range(1, n_pml + 1):
            np.testing.assert_array_equal(
                mesh_pml.material_id[:, :, -off],
                interior,
            )

    def test_zmin_extension(self):
        self._occ()
        mesh_ref = self._mesh(None)
        mesh_pml = self._mesh(["zmin"])
        n_pml = mesh_pml._pml_cells["zmin"]
        assert mesh_pml.Nz == mesh_ref.Nz + n_pml
        dz = mesh_pml.grid.dz
        np.testing.assert_allclose(dz[:n_pml], dz[n_pml], rtol=1e-9)


class TestDegenerateAxis:
    """N = 1 cells along an axis must produce a valid mesh."""

    def test_single_cell_axis(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 0.5e-3, 10e-3), material=Material.air()))
        ctrl = MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=0)
        mesh = Mesh.from_geometry(m, ctrl, f_max=10e9)
        assert mesh.Ny == 1
        assert mesh.material_id.shape == (mesh.Nx, 1, mesh.Nz)
        assert mesh.pec_mask_edges.dtype == bool


class TestRampFixpointProperty:
    """Node generators: exact endpoints, floor, bounded neighbour ratio.

    The measured worst neighbour-cell ratio is 1.5*g (the sub-h_max/2
    remainder absorbed into the last ramp cell); pinned here over
    randomised parameters.
    """

    def test_generators_property(self):
        from magnelio.mesh.mesher import (
            _grade_symmetric_to_uniform,
            _grade_then_uniform,
        )

        rng = np.random.default_rng(21)
        for _ in range(500):
            interval = float(rng.uniform(1e-4, 5e-2))
            g = float(rng.uniform(1.05, 2.5))
            h_max = float(rng.uniform(1e-5, interval))
            h_fine = float(rng.uniform(0.05 * h_max, h_max))
            mc = float(
                rng.choice(
                    [0.0, rng.uniform(0.2, 1.0) * h_fine],
                )
            )
            for nodes in (
                _grade_then_uniform(0.0, interval, h_fine, h_max, g, min_cell=mc),
                _grade_symmetric_to_uniform(0.0, interval, h_fine, h_max, g, min_cell=mc),
            ):
                arr = np.asarray(nodes)
                assert arr[0] == 0.0 and arr[-1] == interval
                d = np.diff(arr)
                assert np.all(d > 0)
                if mc > 0:
                    assert d.min() >= mc * (1 - 1e-9)
                if len(d) > 1:
                    r = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]).max()
                    assert r <= 1.5 * g * (1 + 1e-9)


class TestBoundaryBufferCells:
    """DD-107: every domain face gets >= 3 equidistant adjacent cells.

    Modal ports live on domain faces but are declared after meshing,
    so the mesher cannot know which faces carry one — the buffer is
    enforced everywhere (reference_waveguide_ports.md §2.4).  The
    single-cell degenerate axis is exempt (pinned by
    TestDegenerateAxis); a hard min_cell_size floor that makes three
    cells impossible wins over the buffer.
    """

    @staticmethod
    def _tail_uniform(widths, end):
        tail = widths[-3:] if end == "hi" else widths[:3]
        return (max(tail) - min(tail)) / min(tail) <= 1e-9

    def test_ramp_shorter_than_boundary_interval(self):
        # The beamcoupler regression in miniature: the ramp from the
        # fine interface saturates before the wall → buffer was always
        # present.  Shrink the interval so the ramp no longer fits.
        ctrl = MeshControl(min_nodes_per_wavelength=10, growth_factor=1.3)
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 40e-3],
            h_max=21.4e-3,
            h_fine=0.5e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        assert self._tail_uniform(widths, "lo")
        assert self._tail_uniform(widths, "hi")

    def test_saturated_boundary_interval_unchanged_path(self):
        # Long interval, ramp saturates → the uniform run provides the
        # buffer as before.
        ctrl = MeshControl(min_nodes_per_wavelength=20, growth_factor=1.3)
        nodes = _generate_axis_lines(
            [0.0, 20e-3, 120e-3],
            h_max=2e-3,
            h_fine=0.2e-3,
            control=ctrl,
        )
        widths = np.diff(nodes)
        assert self._tail_uniform(widths, "lo")
        assert self._tail_uniform(widths, "hi")

    def test_property_random_boundary_intervals(self):
        from magnelio.mesh.mesher import _grade_then_uniform

        rng = np.random.default_rng(7)
        for _ in range(500):
            interval = float(rng.uniform(1e-4, 5e-2))
            g = float(rng.uniform(1.05, 2.5))
            h_max = float(rng.uniform(1e-5, interval))
            h_fine = float(rng.uniform(0.05 * h_max, h_max))
            mc = float(
                rng.choice(
                    [0.0, rng.uniform(0.2, 1.0) * h_fine],
                )
            )
            nodes = _grade_then_uniform(
                0.0,
                interval,
                h_fine,
                h_max,
                g,
                min_cell=mc,
                boundary_buffer=True,
            )
            d = np.diff(np.asarray(nodes))
            if len(d) < 3:
                # Only the hard floor may prevent three cells.
                assert mc > 0 and interval / 3 < mc * (1 + 1e-9)
                continue
            tail = d[-3:]
            assert (tail.max() - tail.min()) / tail.min() <= 1e-9

    def test_min_cell_floor_wins_over_buffer(self):
        from magnelio.mesh.mesher import _grade_then_uniform

        # 1 mm interval, floor 0.4 mm: three cells would need
        # 0.333 mm < floor — the buffer must yield, cells stay legal.
        nodes = _grade_then_uniform(
            0.0,
            1e-3,
            h_fine=0.5e-3,
            h_max=0.9e-3,
            g=1.3,
            min_cell=0.4e-3,
            boundary_buffer=True,
        )
        d = np.diff(np.asarray(nodes))
        assert d.min() >= 0.4e-3 * (1 - 1e-9)

    def test_mesh_from_geometry_all_faces(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Brick, Difference, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        air = Material.air()
        pec = Material.pec()
        outer = Brick(origin=(1e-3, 2e-3, 0.5e-3), size=(6e-3, 4e-3, 7e-3), material=air)
        bar = Brick(origin=(2e-3, 3e-3, 1e-3), size=(1e-3, 1e-3, 6e-3), material=pec)
        m = GeometryModel(background=pec)
        m.add((Difference(outer, bar), bar))
        mesh = Mesh.from_geometry(
            m,
            MeshControl(min_nodes_per_wavelength=10),
            f_max=20e9,
        )
        for arr in (mesh.grid.dx, mesh.grid.dy, mesh.grid.dz):
            w = np.asarray(arr)
            assert len(w) >= 3
            assert self._tail_uniform(w, "lo"), w[:4]
            assert self._tail_uniform(w, "hi"), w[-4:]


class TestSliverGate:
    """A sub-min_feature_gap cell not caused by user anchors raises."""

    def test_injected_sliver_raises(self, monkeypatch):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        import magnelio.mesh.mesher as mesher_mod
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material

        orig = mesher_mod._generate_axis_lines

        def broken(planes, h_max, h_fine, control, **kwargs):
            nodes = orig(planes, h_max=h_max, h_fine=h_fine, control=control, **kwargs)
            if len(nodes) >= 2:
                # Inject a 1e-12 m sliver — the DD-058 corruption class.
                nodes = sorted(set(nodes) | {nodes[0] + 1e-12})
            return nodes

        monkeypatch.setattr(mesher_mod, "_generate_axis_lines", broken)
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(5e-3, 5e-3, 5e-3), material=Material.air()))
        with pytest.raises(RuntimeError, match="mesher invariant"):
            mesher_mod.Mesh.from_geometry(
                m,
                MeshControl(min_nodes_per_wavelength=10),
                f_max=10e9,
            )

    def test_forced_pair_below_gap_is_allowed(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(5e-3, 5e-3, 5e-3), material=Material.air()))
        # Explicit min_feature_gap: under the DD-120 bbox-relative
        # default (~8.7e-8 for this cube) a 5e-7 pair is simply
        # resolvable and no longer a below-gap special case.
        ctrl = MeshControl(
            min_nodes_per_wavelength=10,
            forced_planes={"z": [2e-3, 2e-3 + 5e-7]},
            min_feature_gap=1e-6,
        )
        with pytest.warns(UserWarning, match="closer than"):
            mesh = Mesh.from_geometry(m, ctrl, f_max=10e9)
        gz = np.asarray(mesh.grid.z)
        assert 2e-3 in gz and (2e-3 + 5e-7) in gz


# ── Mesh.from_geometry() eps_max / h_target ───────────────────────────────────


class TestFromGeometryEpsMax:
    """Verify that h_target accounts for the maximum refractive index."""

    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepPrimAPI")

    def test_fr4_gives_finer_mesh_than_air(self):
        """A domain containing FR4 (εr=4.4) should produce finer cells than air."""
        self._occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        ctrl = MeshControl(min_nodes_per_wavelength=10, max_cell_size=None)

        # Air-only mesh
        m_air = GeometryModel()
        m_air.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, 1e-3), material=Material.air()))
        mesh_air = Mesh.from_geometry(m_air, ctrl, f_max=10e9)

        # FR4 mesh (εr=4.4 → n=sqrt(4.4)≈2.1 → cells ≈2.1× finer)
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        m_fr4 = GeometryModel()
        m_fr4.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, 1e-3), material=fr4))
        mesh_fr4 = Mesh.from_geometry(m_fr4, ctrl, f_max=10e9)

        # FR4 mesh must have more cells along x
        assert mesh_fr4.Nx > mesh_air.Nx

    def test_pec_ignored_for_wavelength_h_target(self):
        """PEC shapes should not affect wavelength-based h_target.

        However, PEC geometry does contribute to feature-based resolution
        (DD-028) since the PEC boundary creates a critical plane.  To isolate
        the wavelength effect, disable feature-based resolution.
        """
        self._occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        ctrl = MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=0)

        # Air + PEC: PEC should not make wavelength h_target smaller.
        # PEC brick uses same x-extent as Air to keep critical planes
        # identical — otherwise the extra critical plane changes interval
        # splitting, which is a separate (valid) effect unrelated to h_target.
        m = GeometryModel(allow_overlaps=True)
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, 1e-3), material=Material.air()))
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, 1e-3), material=Material.pec()))
        mesh_with_pec = Mesh.from_geometry(m, ctrl, f_max=10e9)

        m_air = GeometryModel()
        m_air.add(Brick(origin=(0, 0, 0), size=(10e-3, 1e-3, 1e-3), material=Material.air()))
        mesh_air = Mesh.from_geometry(m_air, ctrl, f_max=10e9)

        # Same number of cells: PEC did not change wavelength-based h_target
        assert mesh_with_pec.Nx == mesh_air.Nx
