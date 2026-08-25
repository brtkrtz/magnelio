"""Singularity refinement at conductor edges (DD-194).

Three layers: the geometry pass that finds the planes holding a
singular conductor edge, the asymmetric grading profile that starts
an interval at two different fine sizes, and the mesher end to end
(``MeshControl.singularity_refinement``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio.geo import GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh import mesher
from magnelio.mesh.mesher import (
    Mesh,
    MeshControl,
    _grade_asymmetric_to_uniform,
    _grade_symmetric_to_uniform,
    _two_ramp_fill,
)

MM = 1e-3


def _ratios(widths):
    w = np.asarray(widths)
    r = w[1:] / w[:-1]
    return np.maximum(r, 1.0 / r)


# ---------------------------------------------------------------------------
# Geometry: which edges are singular
# ---------------------------------------------------------------------------


class TestSingularEdgePlanes:
    """A sharp edge is singular when the wedge below 180° at it is metal."""

    @pytest.fixture(autouse=True)
    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepOffset")

    @staticmethod
    def _planes(shapes, background=None):
        from magnelio.geo._occ_backend import extract_singular_edge_planes

        found = extract_singular_edge_planes(shapes, background)
        return {ax: sorted({round(p / MM, 6) for p in found[ax]}) for ax in "xyz"}

    @staticmethod
    def _brick(material, origin=(0, 0, 0), size=(10 * MM, 4 * MM, 1 * MM)):
        from magnelio.geo.primitives import Brick

        return Brick(origin=origin, size=size, material=material)

    def test_pec_brick_every_edge_is_convex(self):
        planes = self._planes([self._brick(Material.pec())], Material.air())
        assert planes == {"x": [0.0, 10.0], "y": [0.0, 4.0], "z": [0.0, 1.0]}

    def test_lossy_metal_counts_as_metal(self):
        cu = Material.lossy_metal("cu", sigma=5.8e7)
        planes = self._planes([self._brick(cu)], Material.air())
        assert planes == {"x": [0.0, 10.0], "y": [0.0, 4.0], "z": [0.0, 1.0]}

    def test_dielectric_edges_are_not_singular(self):
        fr4 = Material("fr4", epsilon=(4.3,) * 3)
        planes = self._planes([self._brick(fr4)], Material.air())
        assert planes == {"x": [], "y": [], "z": []}

    def test_cavity_corners_are_regular(self):
        # A vacuum body in a PEC background: its convex edges are the
        # concave corners of the cavity — no singularity.
        planes = self._planes([self._brick(Material.air())], Material.pec())
        assert planes == {"x": [], "y": [], "z": []}

    def test_iris_rim_of_a_cavity_is_singular(self):
        from magnelio.geo.primitives import Cylinder

        body = (
            Cylinder(origin=(0, 0, 0), radius=5 * MM, height=10 * MM, axis="z", material="air")
            + Cylinder(
                origin=(0, 0, 10 * MM), radius=2 * MM, height=4 * MM, axis="z", material="air"
            )
            + Cylinder(
                origin=(0, 0, 14 * MM), radius=5 * MM, height=10 * MM, axis="z", material="air"
            )
        )
        # The concave edges of the vacuum body are the iris rims; the
        # metal outside them is the sharp wedge.
        assert self._planes([body], Material.pec()) == {"x": [], "y": [], "z": [10.0, 14.0]}
        # ... and nothing at all without metal around the body.
        assert self._planes([body], Material.air()) == {"x": [], "y": [], "z": []}

    def test_ridge_in_a_cavity(self):
        box = self._brick(Material.air(), size=(10 * MM, 10 * MM, 10 * MM))
        ridge = self._brick(Material.air(), origin=(4 * MM, 4 * MM, 5 * MM), size=(2 * MM,) * 3)
        planes = self._planes([box - ridge], Material.pec())
        assert planes == {"x": [4.0, 6.0], "y": [4.0, 6.0], "z": [5.0, 7.0]}

    def test_cylinder_has_singular_rims_only(self):
        from magnelio.geo.primitives import Cylinder

        rod = Cylinder(origin=(0, 0, 0), radius=3 * MM, height=8 * MM, axis="z", material="pec")
        planes = self._planes([rod], Material.air())
        assert planes == {"x": [], "y": [], "z": [0.0, 8.0]}

    def test_fillet_onset_is_tangential(self):
        rounded = self._brick(Material.pec()).filleted(edges="all", radius=0.3 * MM)
        assert self._planes([rounded], Material.air()) == {"x": [], "y": [], "z": []}

    def test_chamfer_edges_stay_singular(self):
        chamfered = self._brick(Material.pec()).chamfered(edges="all", distance=0.3 * MM)
        planes = self._planes([chamfered], Material.air())
        assert planes["z"] == pytest.approx([0.0, 0.3, 0.7, 1.0])

    def test_metal_next_to_dielectric_marks_the_metal_edges_only(self):
        fr4 = Material("fr4", epsilon=(4.3,) * 3)
        slab = self._brick(fr4, size=(5 * MM, 4 * MM, 1 * MM))
        metal = self._brick(Material.pec(), origin=(5 * MM, 0, 0), size=(5 * MM, 4 * MM, 1 * MM))
        planes = self._planes([slab, metal], Material.air())
        assert planes["x"] == [5.0, 10.0]

    def test_without_any_metal_the_pass_is_free(self, monkeypatch):
        fr4 = Material("fr4", epsilon=(4.3,) * 3)
        slab = self._brick(fr4)
        called = []
        monkeypatch.setattr(slab, "_occ_shape", lambda *a, **k: called.append(1))
        assert self._planes([slab], Material.air()) == {"x": [], "y": [], "z": []}
        assert not called


# ---------------------------------------------------------------------------
# Profile: two different fine sizes on one interval
# ---------------------------------------------------------------------------


class TestAsymmetricGrading:
    G = 1.3

    def test_long_interval_has_both_ramps_and_a_uniform_middle(self):
        nodes = _grade_asymmetric_to_uniform(0.0, 20e-3, 0.1e-3, 0.4e-3, 1e-3, self.G)
        w = np.diff(nodes)
        assert nodes[0] == 0.0 and nodes[-1] == 20e-3
        assert w[0] == pytest.approx(0.1e-3) and w[-1] == pytest.approx(0.4e-3)
        assert w.max() <= 1e-3 * (1 + 1e-9)
        assert _ratios(w).max() <= self.G * (1 + 1e-9)
        # The steeper side needs more cells to reach the bulk size.
        n_lo = int(np.argmax(w >= 1e-3 * (1 - 1e-9)))
        n_hi = len(w) - 1 - int(np.argmax(w[::-1] >= 1e-3 * (1 - 1e-9)))
        assert n_lo > len(w) - 1 - n_hi

    def test_equal_ends_reduce_to_the_symmetric_profile(self):
        sym = _grade_symmetric_to_uniform(0.0, 5e-3, 0.2e-3, 1e-3, self.G)
        asym = _grade_asymmetric_to_uniform(0.0, 5e-3, 0.2e-3, 0.2e-3, 1e-3, self.G)
        assert asym == sym

    @pytest.mark.parametrize("interval", [0.9e-3, 1.4e-3, 2.1e-3, 3.3e-3, 5.0e-3])
    def test_short_interval_pins_the_fine_cell_and_caps_the_other_end(self, interval):
        h_lo, h_hi = 0.1e-3, 0.3e-3
        nodes = _grade_asymmetric_to_uniform(0.0, interval, h_lo, h_hi, 1e-3, self.G)
        w = np.diff(nodes)
        assert w.sum() == pytest.approx(interval)
        assert w[0] == pytest.approx(h_lo)
        assert h_lo * (1 - 1e-9) <= w[-1] <= h_hi * 1.05
        assert w.max() <= 1e-3 * 1.25
        assert _ratios(w).max() <= self.G * (1 + 1e-6)

    def test_tent_is_mirrored_when_the_upper_end_is_the_fine_one(self):
        lo_first = _two_ramp_fill(2.1e-3, 0.1e-3, 0.3e-3, self.G)
        hi_first = _two_ramp_fill(2.1e-3, 0.3e-3, 0.1e-3, self.G)
        assert hi_first == list(reversed(lo_first))

    def test_tent_ratios_never_exceed_g(self):
        h_lo = 0.1e-3
        for h_hi in (h_lo * 2, h_lo * self.G**2, h_lo * 4):
            for interval in np.linspace(2 * h_lo, 4e-3, 60):
                w = np.asarray(_two_ramp_fill(float(interval), h_lo, h_hi, self.G))
                assert w.sum() == pytest.approx(interval)
                assert _ratios(w).max() <= self.G * (1 + 1e-6), (h_hi, interval)
                assert w[-1] <= h_hi * 1.05 * (1 + 1e-9) or len(w) <= 3, (h_hi, interval)

    def test_tent_beats_a_capped_one_sided_ramp_on_a_medium_interval(self):
        # 25 µm edge cell, 50 µm interface cell, 1 mm bulk, 5 mm apart:
        # the two full ramps do not fit; a ramp capped at 50 µm would
        # need ~100 cells, the tent a quarter of that.
        w = np.asarray(_two_ramp_fill(5e-3, 25e-6, 50e-6, self.G))
        assert len(w) < 30
        assert w[0] == pytest.approx(25e-6)
        assert w[-1] <= 52.5e-6 * (1 + 1e-9)
        assert _ratios(w).max() <= self.G * (1 + 1e-6)

    def test_interval_below_two_fine_cells_is_one_cell(self):
        assert _two_ramp_fill(0.15e-3, 0.1e-3, 0.3e-3, self.G) == [0.15e-3]

    def test_floor_is_never_undershot(self):
        for interval in np.linspace(0.2e-3, 2e-3, 30):
            w = np.asarray(_two_ramp_fill(float(interval), 0.1e-3, 0.3e-3, self.G, 0.1e-3))
            assert w.min() >= 0.1e-3 * (1 - 1e-12), interval
            assert w.sum() == pytest.approx(interval)

    def test_axis_lines_take_a_fine_size_per_plane(self):
        control = MeshControl(growth_factor=self.G)
        planes = [0.0, 2e-3, 6e-3, 10e-3]
        fine = [0.4e-3, 0.1e-3, 0.4e-3, 0.4e-3]
        nodes = mesher._generate_axis_lines(
            planes, h_max=1e-3, h_fine=0.4e-3, control=control, h_fine_planes=fine
        )
        w = np.diff(nodes)
        k = int(np.argmin(np.abs(np.asarray(nodes) - 2e-3)))
        assert 0.1e-3 * (1 - 1e-9) <= w[k - 1] <= 0.1e-3 * 1.05 * (1 + 1e-9)
        assert 0.1e-3 * (1 - 1e-9) <= w[k] <= 0.1e-3 * 1.05 * (1 + 1e-9)
        assert _ratios(w).max() <= self.G * (1 + 1e-6)
        same = mesher._generate_axis_lines(planes, h_max=1e-3, h_fine=0.4e-3, control=control)
        assert same == mesher._generate_axis_lines(
            planes, h_max=1e-3, h_fine=0.4e-3, control=control, h_fine_planes=[0.4e-3] * 4
        )
        with pytest.raises(ValueError, match="h_fine_planes"):
            mesher._generate_axis_lines(
                planes, h_max=1e-3, h_fine=0.4e-3, control=control, h_fine_planes=[0.1e-3]
            )


# ---------------------------------------------------------------------------
# Mesher end to end
# ---------------------------------------------------------------------------


def test_mesh_control_validates_the_factor():
    assert MeshControl().singularity_refinement == 1.0
    assert MeshControl(singularity_refinement=2.5).singularity_refinement == 2.5
    with pytest.raises(ValueError, match="singularity_refinement"):
        MeshControl(singularity_refinement=0.5)
    with pytest.raises(ValueError, match="singularity_refinement"):
        MeshControl(singularity_refinement=float("nan"))


class TestMicrostripRefinement:
    """A strip on a substrate: the strip's edges refine x and y, not z."""

    H_SUB, W_STRIP, T_STRIP = 0.8 * MM, 1.2 * MM, 0.2 * MM
    W_BOX, H_BOX, L = 8.0 * MM, 5.0 * MM, 6.0 * MM
    F_MAX = 15e9

    @pytest.fixture(autouse=True)
    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepOffset")

    def _model(self):
        from magnelio.geo.primitives import Brick

        fr4 = Material("fr4", epsilon=(4.3,) * 3)
        substrate = Brick(
            origin=(-self.W_BOX / 2, 0, 0), size=(self.W_BOX, self.H_SUB, self.L), material=fr4
        )
        air = Brick(
            origin=(-self.W_BOX / 2, self.H_SUB, 0),
            size=(self.W_BOX, self.H_BOX - self.H_SUB, self.L),
            material="air",
        )
        strip = Brick(
            origin=(-self.W_STRIP / 2, self.H_SUB, 0),
            size=(self.W_STRIP, self.T_STRIP, self.L),
            material="pec",
        )
        model = GeometryModel()
        model.add(substrate)
        model.add(air - strip)
        model.add(strip)
        return model

    def _mesh(self, k, **kw):
        control = MeshControl(min_nodes_per_wavelength=20, singularity_refinement=k, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return Mesh.from_geometry(self._model(), control, f_max=self.F_MAX)

    @staticmethod
    def _cells_at(lines, position):
        arr = np.asarray(lines)
        d = np.diff(arr)
        i = int(np.argmin(np.abs(arr - position)))
        assert abs(arr[i] - position) < 1e-12
        return d[i - 1], d[i]

    def test_factor_one_is_the_plain_mesh(self):
        ref = self._mesh(1.0)
        plain = Mesh.from_geometry(
            self._model(), MeshControl(min_nodes_per_wavelength=20), f_max=self.F_MAX
        )
        for ax in "xyz":
            assert np.array_equal(getattr(ref.grid, ax), getattr(plain.grid, ax))

    @pytest.mark.parametrize("k", [2.0, 3.0])
    def test_edge_cells_are_h_fine_over_k_on_both_sides(self, k):
        ref = self._mesh(1.0)
        fine = self._mesh(k)
        # h_fine per axis: the strip's width on x, its thickness on y
        # (min_cells_per_feature = 4).
        h_fine = {"x": self.W_STRIP / 4, "y": self.T_STRIP / 4}
        for ax, positions in (
            ("x", (-self.W_STRIP / 2, self.W_STRIP / 2)),
            ("y", (self.H_SUB, self.H_SUB + self.T_STRIP)),
        ):
            for pos in positions:
                want = h_fine[ax] / k
                pair = self._cells_at(getattr(fine.grid, ax), pos)
                for a in pair:
                    # 5 % DD-105 band above; the DD-107 buffer refit at
                    # a domain face may trim the fine cell within the
                    # legacy refit class (below 1 - 1/g).
                    assert want / 1.3 <= a <= want * 1.06, (ax, pos, a)
                assert max(pair) >= want * 0.99, (ax, pos, pair)
                for b in self._cells_at(getattr(ref.grid, ax), pos):
                    assert b >= h_fine[ax] / 1.3  # the plain mesh sits at h_fine
            w = np.diff(getattr(fine.grid, ax))
            assert _ratios(w).max() <= 1.3 * (1 + 1e-6)
        # The strip runs the full length: its ends are the domain
        # faces, which are never refined — z is untouched.
        assert np.array_equal(ref.grid.z, fine.grid.z)

    def test_the_substrate_edge_alone_is_not_refined(self):
        # Without the strip the y = h_sub plane is a dielectric face.
        from magnelio.geo.primitives import Brick

        fr4 = Material("fr4", epsilon=(4.3,) * 3)
        model = GeometryModel()
        model.add(
            Brick(
                origin=(-self.W_BOX / 2, 0, 0),
                size=(self.W_BOX, self.H_SUB, self.L),
                material=fr4,
            )
        )
        model.add(
            Brick(
                origin=(-self.W_BOX / 2, self.H_SUB, 0),
                size=(self.W_BOX, self.H_BOX - self.H_SUB, self.L),
                material="air",
            )
        )
        control = MeshControl(min_nodes_per_wavelength=20)
        plain = Mesh.from_geometry(model, control, f_max=self.F_MAX)
        refined = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=20, singularity_refinement=3.0),
            f_max=self.F_MAX,
        )
        for ax in "xyz":
            assert np.array_equal(getattr(plain.grid, ax), getattr(refined.grid, ax))

    def test_hard_floor_caps_the_refinement(self):
        floor = 0.12 * MM
        fine = self._mesh(4.0, min_cell_size=floor)
        for ax in "xy":
            assert np.diff(getattr(fine.grid, ax)).min() >= floor * (1 - 1e-12)

    def test_no_undershoot_warning_for_the_deliberate_edge_cells(self):
        # _mesh() runs with warnings as errors: the DD-105 check must
        # know the edge cells are asked for.
        self._mesh(3.0)


class TestCavityIris:
    """PEC background with a vacuum body: the iris rim refines, the corners do not."""

    @pytest.fixture(autouse=True)
    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepOffset")

    def _mesh(self, k):
        from magnelio.geo.primitives import Cylinder

        body = (
            Cylinder(origin=(0, 0, 0), radius=5 * MM, height=10 * MM, axis="z", material="air")
            + Cylinder(
                origin=(0, 0, 10 * MM), radius=2 * MM, height=4 * MM, axis="z", material="air"
            )
            + Cylinder(
                origin=(0, 0, 14 * MM), radius=5 * MM, height=10 * MM, axis="z", material="air"
            )
        )
        model = GeometryModel(background="pec")
        model.add(body)
        control = MeshControl(min_nodes_per_wavelength=12, singularity_refinement=k)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return Mesh.from_geometry(model, control, f_max=20e9)

    def test_iris_planes_refine_and_the_rest_stays(self):
        ref, fine = self._mesh(1.0), self._mesh(2.0)
        z_ref, z_fine = np.asarray(ref.grid.z), np.asarray(fine.grid.z)
        for pos in (10 * MM, 14 * MM):
            i_ref = int(np.argmin(np.abs(z_ref - pos)))
            i_fine = int(np.argmin(np.abs(z_fine - pos)))
            d_ref = np.diff(z_ref)[i_ref - 1 : i_ref + 1]
            d_fine = np.diff(z_fine)[i_fine - 1 : i_fine + 1]
            assert np.all(d_fine <= d_ref * 0.55)
        # x/y planes come from the cylinder silhouettes (no sharp edge
        # lies flat there) — identical under both factors.
        assert np.array_equal(ref.grid.x, fine.grid.x)
        assert np.array_equal(ref.grid.y, fine.grid.y)


def test_undershoot_check_knows_the_refined_planes():
    from magnelio.mesh._quality import check_grading_undershoot

    control = MeshControl(min_nodes_per_wavelength=20)
    planes = [0.0, 2e-3, 6e-3, 10e-3]
    fine = [0.4e-3, 0.1e-3, 0.4e-3, 0.4e-3]
    lines = mesher._generate_axis_lines(
        planes, h_max=1e-3, h_fine=0.4e-3, control=control, h_fine_planes=fine
    )
    grid = {"x": lines, "y": [0.0, 1e-3, 2e-3], "z": [0.0, 1e-3, 2e-3]}
    axis_planes = {"x": planes, "y": [0.0, 2e-3], "z": [0.0, 2e-3]}
    h_fine_axis = {"x": 0.4e-3, "y": 1e-3, "z": 1e-3}
    kwargs = dict(
        axis_anchors={"x": set(), "y": set(), "z": set()},
        h_fine_axis=h_fine_axis,
        h_max=1e-3,
        control=control,
    )
    with pytest.warns(UserWarning, match="below the"):
        check_grading_undershoot(grid, axis_planes, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_grading_undershoot(
            grid, axis_planes, h_fine_planes={"x": fine, "y": [1e-3] * 2, "z": [1e-3] * 2}, **kwargs
        )


def test_symmetry_clip_drops_singular_marks_beyond_the_plane():
    pytest.importorskip("OCC.Core.BRepOffset")
    from magnelio.geo.primitives import Brick

    fr4 = Material("fr4", epsilon=(4.3,) * 3)

    def model(bc):
        m = GeometryModel(boundary_conditions=bc)
        m.add(Brick(origin=(-4 * MM, 0, 0), size=(8 * MM, 0.8 * MM, 4 * MM), material=fr4))
        air = Brick(origin=(-4 * MM, 0.8 * MM, 0), size=(8 * MM, 3 * MM, 4 * MM), material="air")
        strip = Brick(
            origin=(-0.6 * MM, 0.8 * MM, 0), size=(1.2 * MM, 0.2 * MM, 4 * MM), material="pec"
        )
        m.add(air - strip)
        m.add(strip)
        return m

    control = MeshControl(min_nodes_per_wavelength=20, singularity_refinement=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        half = Mesh.from_geometry(model({"xmin": "SymmetryPMC"}), control, f_max=15e9)
    x = np.asarray(half.grid.x)
    assert x[0] >= 0.0 and x[0] < 0.1 * MM  # PMC pull-in owns the first node
    d = np.diff(x)
    i = int(np.argmin(np.abs(x - 0.6 * MM)))
    assert abs(x[i] - 0.6 * MM) < 1e-12
    # h_fine on x is the half-width 0.6 mm / 4; the kept edge plane
    # halves it on both sides (the buffered symmetry face may trim the
    # inner cell in the DD-107 refit class).
    assert d[i] == pytest.approx(0.075 * MM, rel=0.06)
    assert 0.075 * MM / 1.3 <= d[i - 1] <= 0.075 * MM * 1.06


def test_thin_sheet_keeps_the_knife_edge_refinement():
    """A sub-floor PEC layer becomes a thin sheet: its transverse edges refine."""
    pytest.importorskip("OCC.Core.BRepOffset")
    from magnelio.geo.primitives import Brick

    fr4 = Material("fr4", epsilon=(4.3,) * 3)
    m = GeometryModel()
    m.add(Brick(origin=(-4 * MM, 0, 0), size=(8 * MM, 0.8 * MM, 4 * MM), material=fr4))
    air = Brick(origin=(-4 * MM, 0.8 * MM, 0), size=(8 * MM, 3 * MM, 4 * MM), material="air")
    strip = Brick(
        origin=(-0.6 * MM, 0.8 * MM, 0), size=(1.2 * MM, 0.035 * MM, 4 * MM), material="pec"
    )
    m.add(air - strip)
    m.add(strip)
    kw = dict(min_nodes_per_wavelength=20, min_cell_size=0.05 * MM)
    ref = Mesh.from_geometry(m, MeshControl(**kw), f_max=15e9)
    fine = Mesh.from_geometry(m, MeshControl(singularity_refinement=2.0, **kw), f_max=15e9)
    # The sheet is ONE plane: the far face never enters the grid.
    assert not np.any(np.abs(np.asarray(fine.grid.y) - 0.835 * MM) < 1e-9)
    for grid in (ref, fine):
        assert np.any(np.abs(np.asarray(grid.grid.y) - 0.8 * MM) < 1e-12)
    x_ref, x_fine = np.asarray(ref.grid.x), np.asarray(fine.grid.x)
    i = int(np.argmin(np.abs(x_fine - 0.6 * MM)))
    j = int(np.argmin(np.abs(x_ref - 0.6 * MM)))
    assert np.diff(x_fine)[i] <= 0.55 * np.diff(x_ref)[j]
