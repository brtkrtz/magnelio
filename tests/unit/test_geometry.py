"""Unit tests for geometry module (CSG primitives, Boolean operations, transforms, OCC backend)."""

import math

import pytest

from magnelio.materials.material import Material


def _air():
    return Material.air()


def _occ():
    """Skip test if pythonocc-core is not installed."""
    return pytest.importorskip("OCC.Core.BRepPrimAPI")


# ── Class creation (no OCC needed) ───────────────────────────────────────────


class TestPrimitiveCreate:
    def test_brick_create(self):
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(1e-3, 2e-3, 3e-3), material=_air())
        assert b.size == (1e-3, 2e-3, 3e-3)

    def test_sphere_create(self):
        from magnelio.geo.primitives import Sphere

        s = Sphere(center=(0, 0, 0), radius=5e-3, material=_air())
        assert s.radius == 5e-3

    def test_cylinder_create(self):
        from magnelio.geo.primitives import Cylinder

        c = Cylinder(origin=(0, 0, 0), radius=1e-3, height=5e-3, axis="z", material=_air())
        assert c.axis == "z"

    def test_cone_create(self):
        from magnelio.geo.primitives import Cone

        c = Cone(origin=(0, 0, 0), bottom_radius=2e-3, top_radius=0.0, height=5e-3, material=_air())
        assert c.top_radius == 0.0

    def test_torus_create(self):
        from magnelio.geo.primitives import Torus

        t = Torus(center=(0, 0, 0), major_radius=10e-3, minor_radius=2e-3, material=_air())
        assert t.major_radius == 10e-3

    def test_brick_default_name_none(self):
        from magnelio.geo.primitives import Brick

        b = Brick(material=_air())
        assert b.name is None

    def test_brick_custom_name(self):
        from magnelio.geo.primitives import Brick

        b = Brick(material=_air(), name="substrate")
        assert b.name == "substrate"


# ── Boolean operation material inheritance (no OCC needed) ───────────────────


class TestBooleanMaterials:
    def test_difference_inherits_base_material(self):
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick, Cylinder

        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        base = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 2e-3), material=fr4)
        hole = Cylinder(origin=(5e-3, 5e-3, 0), radius=0.5e-3, height=2e-3, material=_air())
        diff = Difference(base, hole)
        assert diff.material is fr4

    def test_union_inherits_first_material(self):
        from magnelio.geo.operations import Union
        from magnelio.geo.primitives import Brick

        mat1 = Material(name="A")
        mat2 = Material(name="B")
        u = Union(Brick(material=mat1), Brick(material=mat2))
        assert u.material is mat1

    def test_union_explicit_material(self):
        from magnelio.geo.operations import Union
        from magnelio.geo.primitives import Brick

        mat_out = Material(name="merged")
        u = Union(Brick(material=_air()), Brick(material=_air()), material=mat_out)
        assert u.material is mat_out

    def test_intersection_inherits_a_material(self):
        from magnelio.geo.operations import Intersection
        from magnelio.geo.primitives import Brick

        mat_a = Material(name="A")
        inter = Intersection(Brick(material=mat_a), Brick(material=_air()))
        assert inter.material is mat_a

    def test_difference_explicit_material(self):
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick

        mat_out = Material(name="out")
        diff = Difference(Brick(material=_air()), Brick(material=_air()), material=mat_out)
        assert diff.material is mat_out

    def test_difference_multi_tool(self):
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick

        base = Brick(material=_air())
        t1 = Brick(material=_air())
        t2 = Brick(material=_air())
        diff = Difference(base, t1, t2)
        assert diff.tools == (t1, t2)
        assert diff.material is base.material

    def test_difference_single_tool_backward_compat(self):
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick

        base = Brick(material=_air())
        tool = Brick(material=_air())
        diff = Difference(base, tool)
        assert diff.tools == (tool,)


# ── Construction solids (no material) ────────────────────────────────────────


class TestConstructionSolids:
    def test_primitive_without_material_is_construction(self):
        from magnelio.geo.primitives import Cylinder

        assert Cylinder(radius=1e-3, height=2e-3).material is None

    def test_cut_tool_needs_no_material(self):
        from magnelio.geo.primitives import Brick, Cylinder

        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        board = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 2e-3), material=fr4)
        via = board - Cylinder(origin=(5e-3, 5e-3, -1), radius=0.5e-3, height=2.0)
        assert via.material is fr4

    def test_model_rejects_construction_solid(self):
        from magnelio.geo import Cylinder, GeometryModel

        with pytest.raises(ValueError, match="construction body"):
            GeometryModel().add(Cylinder(radius=1e-3, height=2e-3, name="cut_tool"))

    def test_rejection_names_the_shape(self):
        from magnelio.geo import Brick, GeometryModel

        with pytest.raises(ValueError, match="Brick 'tool'"):
            GeometryModel().add(Brick(name="tool"))

    def test_boolean_result_off_a_construction_base_is_rejected(self):
        from magnelio.geo import Brick, GeometryModel

        # The base is the construction body here, so the Difference has
        # nothing to inherit — the model must not accept it.
        bad = Brick() - Brick(material=_air())
        with pytest.raises(ValueError, match="construction body"):
            GeometryModel().add(bad)

    def test_group_member_without_material_is_rejected(self):
        from magnelio.geo import Brick, GeometryModel, Group

        with pytest.raises(ValueError, match="construction body"):
            GeometryModel().add(Group(Brick(material=_air()), Brick()))

    def test_transformed_construction_solid_stays_material_less(self):
        from magnelio.geo import Brick, GeometryModel

        moved = Brick().translated((1e-3, 0, 0))
        assert moved.material is None
        with pytest.raises(ValueError, match="construction body"):
            GeometryModel().add(moved)


# ── GeometryModel ─────────────────────────────────────────────────────────────


class TestGeometryModel:
    def test_empty_model(self):
        from magnelio.geo import GeometryModel

        m = GeometryModel()
        assert len(m) == 0

    def test_add_returns_self(self):
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        ret = m.add(Brick(material=_air()))
        assert ret is m

    def test_add_multiple(self):
        from magnelio.geo import Brick, GeometryModel, Sphere

        m = GeometryModel()
        m.add(Brick(material=_air())).add(Sphere(material=_air()))
        assert len(m) == 2

    def test_iter(self):
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        b1 = Brick(material=_air())
        b2 = Brick(material=_air())
        m.add(b1).add(b2)
        assert list(m) == [b1, b2]

    def test_repr(self):
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        m.add(Brick(material=_air()))
        assert "1" in repr(m)

    def test_bounding_box_empty_raises(self):
        from magnelio.geo import GeometryModel

        with pytest.raises(ValueError, match="empty"):
            GeometryModel().bounding_box()

    def test_default_background_is_air(self):
        from magnelio.geo import GeometryModel

        m = GeometryModel()
        assert m.background.name == "air"
        assert not m.background.is_pec

    def test_custom_background(self):
        from magnelio.geo import GeometryModel

        pec = Material.pec()
        m = GeometryModel(background=pec)
        assert m.background.is_pec
        assert m.background.name == "PEC"

    def test_allow_overlaps_default_false(self):
        from magnelio.geo import GeometryModel

        m = GeometryModel()
        assert m.allow_overlaps is False

    def test_allow_overlaps_explicit(self):
        from magnelio.geo import GeometryModel

        m = GeometryModel(allow_overlaps=True)
        assert m.allow_overlaps is True

    def test_add_list(self):
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        shapes = [Brick(material=_air()), Brick(material=_air())]
        m.add(shapes)
        assert len(m) == 2

    def test_repr_shows_background(self):
        from magnelio.geo import GeometryModel

        m = GeometryModel(background=Material.pec())
        assert "PEC" in repr(m)


# ── Overlap detection ─────────────────────────────────────────────────────────


class TestOverlapDetection:
    """Verify that GeometryModel.validate() detects overlapping shapes."""

    def test_overlapping_bricks_raises(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel, GeometryOverlapError

        # Different materials — a genuinely ambiguous overlap (same-material
        # overlaps are allowed, see TestSameMaterialOverlap).
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.pec()))
        m.add(Brick(origin=(2e-3, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air()))
        with pytest.raises(GeometryOverlapError, match="overlapping"):
            m.validate()

    def test_non_overlapping_bricks_ok(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air()))
        m.add(Brick(origin=(4e-3, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air()))
        m.validate()  # should not raise

    def test_single_shape_ok(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air()))
        m.validate()  # should not raise

    def test_allow_overlaps_skips_check(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel(allow_overlaps=True)
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.pec()))
        m.add(Brick(origin=(2e-3, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air()))
        # allow_overlaps=True means validate() is not called automatically,
        # but we can still call it explicitly to check (different materials).
        from magnelio.geo import GeometryOverlapError

        with pytest.raises(GeometryOverlapError):
            m.validate()

    def test_csg_difference_no_overlap(self):
        """Difference(A, B) + B should not overlap (non-overlapping CSG)."""
        _occ()
        from magnelio.geo import Brick, Difference, GeometryModel

        pec = Material.pec()
        air = Material.air()
        outer = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=pec)
        cavity = Brick(origin=(2e-3, 2e-3, 2e-3), size=(6e-3, 6e-3, 6e-3), material=air)
        shell = Difference(outer, cavity)
        m = GeometryModel()
        m.add(shell)
        m.add(cavity)
        m.validate()  # should not raise


class TestSameMaterialOverlap:
    """2a: overlaps between value-equal materials are allowed."""

    def _brick(self, origin, material):
        from magnelio.geo import Brick

        return Brick(origin=origin, size=(4e-3, 4e-3, 4e-3), material=material)

    def test_two_pec_instances_allowed(self):
        """Two distinct Material.pec() instances are the same material."""
        _occ()
        from magnelio.geo import GeometryModel

        m = GeometryModel()
        m.add(self._brick((0, 0, 0), Material.pec()))
        m.add(self._brick((2e-3, 0, 0), Material.pec()))
        m.validate()  # must not raise

    def test_two_air_instances_allowed(self):
        _occ()
        from magnelio.geo import GeometryModel

        m = GeometryModel()
        m.add(self._brick((0, 0, 0), Material.air()))
        m.add(self._brick((2e-3, 0, 0), Material.air()))
        m.validate()  # must not raise

    def test_different_material_still_raises(self):
        _occ()
        from magnelio.geo import GeometryModel, GeometryOverlapError

        m = GeometryModel()
        m.add(self._brick((0, 0, 0), Material.pec()))
        m.add(self._brick((2e-3, 0, 0), Material.air()))
        with pytest.raises(GeometryOverlapError):
            m.validate()

    def test_different_name_same_physics_raises(self):
        """Name is part of material identity — differently-named raises."""
        _occ()
        from magnelio.geo import GeometryModel, GeometryOverlapError

        m = GeometryModel()
        m.add(self._brick((0, 0, 0), Material.from_isotropic("A", epsilon=2.0)))
        m.add(self._brick((2e-3, 0, 0), Material.from_isotropic("B", epsilon=2.0)))
        with pytest.raises(GeometryOverlapError):
            m.validate()

    def test_mixed_only_reports_different_pair(self):
        """Same-material pair skipped; the different-material pair still raises."""
        _occ()
        from magnelio.geo import GeometryModel, GeometryOverlapError

        pec = Material.pec()
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        m = GeometryModel()
        m.add(self._brick((0, 0, 0), pec))  # 0
        m.add(self._brick((2e-3, 0, 0), Material.pec()))  # 1 overlaps 0, same mat → ok
        m.add(self._brick((3e-3, 0, 0), fr4))  # 2 overlaps 0,1 → different
        with pytest.raises(GeometryOverlapError) as exc:
            m.validate()
        # The reported pairs involve shape 2 (FR4), never the 0&1 same-mat pair.
        assert "shape_2" in str(exc.value) or "FR4" in str(exc.value)

    def test_same_material_overlap_meshes(self):
        """A same-material overlap fills correctly end-to-end."""
        _occ()
        import numpy as np

        from magnelio.geo import GeometryModel
        from magnelio.mesh.mesher import Mesh, MeshControl

        m = GeometryModel()
        m.add(self._brick((0, 0, 0), Material.pec()))
        m.add(self._brick((2e-3, 0, 0), Material.pec()))
        mesh = Mesh.from_geometry(
            m, MeshControl(min_nodes_per_wavelength=4, max_cell_size=2e-3), f_max=10e9
        )
        pec_ids = [i for i, mm in mesh.material_library.items() if mm.is_pec]
        assert pec_ids and np.any(np.isin(mesh.material_id, pec_ids))


# ── OCC-dependent tests ───────────────────────────────────────────────────────
# All tests below are skipped automatically if pythonocc-core is not installed.


class TestOCCPrimitives:
    """Verify that _occ_shape() and bounding_box() return correct results."""

    def test_brick_occ_shape_runs(self):
        _occ()
        from magnelio.geo.primitives import Brick

        shape = Brick(origin=(0, 0, 0), size=(10e-3, 5e-3, 2e-3), material=_air())
        s = shape._occ_shape()
        assert s is not None

    def test_brick_bounding_box_exact(self):
        _occ()
        from magnelio.geo.primitives import Brick

        shape = Brick(origin=(1e-3, 2e-3, 3e-3), size=(4e-3, 5e-3, 6e-3), material=_air())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = shape.bounding_box()
        assert xmin == pytest.approx(1e-3, abs=1e-6)
        assert ymin == pytest.approx(2e-3, abs=1e-6)
        assert zmin == pytest.approx(3e-3, abs=1e-6)
        assert xmax == pytest.approx(5e-3, abs=1e-6)
        assert ymax == pytest.approx(7e-3, abs=1e-6)
        assert zmax == pytest.approx(9e-3, abs=1e-6)

    def test_sphere_bounding_box_approx(self):
        _occ()
        from magnelio.geo.primitives import Sphere

        r = 3e-3
        shape = Sphere(center=(0, 0, 0), radius=r, material=_air())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = shape.bounding_box()
        assert xmin == pytest.approx(-r, abs=1e-6)
        assert xmax == pytest.approx(+r, abs=1e-6)

    def test_cylinder_z_bounding_box(self):
        _occ()
        from magnelio.geo.primitives import Cylinder

        shape = Cylinder(origin=(0, 0, 0), radius=2e-3, height=8e-3, axis="z", material=_air())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = shape.bounding_box()
        assert zmin == pytest.approx(0.0, abs=1e-6)
        assert zmax == pytest.approx(8e-3, abs=1e-6)
        assert xmin == pytest.approx(-2e-3, abs=1e-6)
        assert xmax == pytest.approx(+2e-3, abs=1e-6)

    def test_cone_occ_shape_runs(self):
        _occ()
        from magnelio.geo.primitives import Cone

        shape = Cone(
            origin=(0, 0, 0), bottom_radius=3e-3, top_radius=0, height=6e-3, material=_air()
        )
        assert shape._occ_shape() is not None

    def test_torus_occ_shape_runs(self):
        _occ()
        from magnelio.geo.primitives import Torus

        shape = Torus(center=(0, 0, 0), major_radius=5e-3, minor_radius=1e-3, material=_air())
        assert shape._occ_shape() is not None

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_cylinder_axis_runs(self, axis):
        _occ()
        from magnelio.geo.primitives import Cylinder

        shape = Cylinder(origin=(0, 0, 0), radius=1e-3, height=3e-3, axis=axis, material=_air())
        assert shape._occ_shape() is not None


class TestOCCBooleanOps:
    def test_difference_shape_runs(self):
        _occ()
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick, Cylinder

        base = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 5e-3), material=_air())
        hole = Cylinder(origin=(5e-3, 5e-3, 0), radius=1e-3, height=5e-3, material=_air())
        diff = Difference(base, hole)
        assert diff._occ_shape() is not None

    def test_occ_shape_is_cached(self):
        """_occ_shape must return the same object on repeated calls so that
        OCC Boolean operations are not re-evaluated by mesh-time hot loops.
        Regression for the 25k-call cache miss observed in coax2rectwg."""
        _occ()
        from magnelio.geo.operations import Difference, Intersection, Union
        from magnelio.geo.primitives import Brick, Cylinder

        base = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 5e-3), material=_air())
        hole = Cylinder(origin=(5e-3, 5e-3, 0), radius=1e-3, height=5e-3, material=_air())
        # Primitives, Difference, Union, Intersection all cache.
        for shape in [
            base,
            hole,
            Difference(base, hole),
            Union(base, hole),
            Intersection(base, hole),
        ]:
            s1 = shape._occ_shape()
            s2 = shape._occ_shape()
            assert s1 is s2, f"{type(shape).__name__}._occ_shape not cached"

    def test_difference_bbox_smaller_than_base(self):
        _occ()
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick

        base = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 5e-3), material=_air())
        sub = Brick(origin=(5e-3, 0, 0), size=(5e-3, 10e-3, 5e-3), material=_air())
        diff = Difference(base, sub)
        (xmin, _, _), (xmax, _, _) = diff.bounding_box()
        assert xmax <= 10e-3 + 1e-6

    def test_difference_multi_tool_occ(self):
        """Multiple tools subtracted in one Difference operation."""
        _occ()
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick, Cylinder

        base = Brick(origin=(0, 0, 0), size=(20e-3, 20e-3, 5e-3), material=_air())
        h1 = Cylinder(origin=(5e-3, 10e-3, 0), radius=1e-3, height=5e-3, material=_air())
        h2 = Cylinder(origin=(15e-3, 10e-3, 0), radius=1e-3, height=5e-3, material=_air())
        diff = Difference(base, h1, h2)
        assert diff._occ_shape() is not None
        # BBox should still span the full base extent
        (xmin, _, _), (xmax, _, _) = diff.bounding_box()
        assert xmin == pytest.approx(0.0, abs=1e-6)
        assert xmax == pytest.approx(20e-3, abs=1e-6)

    def test_union_bbox_covers_both(self):
        _occ()
        from magnelio.geo.operations import Union
        from magnelio.geo.primitives import Brick

        b1 = Brick(origin=(0, 0, 0), size=(5e-3, 5e-3, 5e-3), material=_air())
        b2 = Brick(origin=(6e-3, 0, 0), size=(4e-3, 5e-3, 5e-3), material=_air())
        u = Union(b1, b2)
        (xmin, _, _), (xmax, _, _) = u.bounding_box()
        assert xmin == pytest.approx(0.0, abs=1e-6)
        assert xmax == pytest.approx(10e-3, abs=1e-6)

    def test_intersection_bbox_inside_both(self):
        _occ()
        from magnelio.geo.operations import Intersection
        from magnelio.geo.primitives import Brick

        b1 = Brick(origin=(0, 0, 0), size=(8e-3, 8e-3, 8e-3), material=_air())
        b2 = Brick(origin=(4e-3, 0, 0), size=(8e-3, 8e-3, 8e-3), material=_air())
        inter = Intersection(b1, b2)
        (xmin, _, _), (xmax, _, _) = inter.bounding_box()
        assert xmin == pytest.approx(4e-3, abs=1e-6)
        assert xmax == pytest.approx(8e-3, abs=1e-6)


class TestOCCTransforms:
    def test_translate_shifts_bbox(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        b = Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 2e-3), material=_air())
        moved = translate(b, (3e-3, 4e-3, 5e-3))
        (xmin, ymin, zmin), (xmax, ymax, zmax) = moved.bounding_box()
        assert xmin == pytest.approx(3e-3, abs=1e-6)
        assert ymin == pytest.approx(4e-3, abs=1e-6)
        assert zmin == pytest.approx(5e-3, abs=1e-6)
        assert xmax == pytest.approx(5e-3, abs=1e-6)

    def test_translate_material_inherited(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        mat = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=mat)
        moved = translate(b, (1e-3, 0, 0))
        assert moved.material is mat

    def test_rotate_90deg_z(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import rotate

        # Brick at (1,0,0)..(3,1,1) mm; rotate 90° around z
        b = Brick(origin=(1e-3, 0, 0), size=(2e-3, 1e-3, 1e-3), material=_air())
        rotated = rotate(b, axis=(0, 0, 1), angle_deg=90, origin=(0, 0, 0))
        (_, ymin, _), (_, ymax, _) = rotated.bounding_box()
        # After 90° rotation, x→y, so y range should be ~[1e-3, 3e-3]
        assert ymin == pytest.approx(1e-3, abs=1e-6)
        assert ymax == pytest.approx(3e-3, abs=1e-6)

    def test_scale_doubles_bbox(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import scale

        b = Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 2e-3), material=_air())
        scaled = scale(b, factor=2.0, center=(0, 0, 0))
        (xmin, _, _), (xmax, _, _) = scaled.bounding_box()
        assert xmin == pytest.approx(0.0, abs=1e-6)
        assert xmax == pytest.approx(4e-3, abs=1e-6)

    def test_translate_material_via_property(self):
        _occ()
        from magnelio.geo.primitives import Sphere
        from magnelio.geo.transforms import rotate, scale, translate

        mat = Material(name="M")
        s = Sphere(center=(0, 0, 0), radius=1e-3, material=mat)
        assert translate(s, (1, 0, 0)).material is mat
        assert rotate(s, (0, 0, 1), 45).material is mat
        assert scale(s, 2.0).material is mat

    # -- repeat / copy / unite -------------------------------------------------

    def test_translate_repeat(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        copies = translate(b, (2e-3, 0, 0), repeat=3)
        assert isinstance(copies, list)
        assert len(copies) == 3
        # Copy 1 at 2mm, copy 2 at 4mm, copy 3 at 6mm
        for i, s in enumerate(copies, 1):
            (xmin, _, _), _ = s.bounding_box()
            assert xmin == pytest.approx(2e-3 * i, abs=1e-6)

    def test_translate_repeat_with_copy(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        result = translate(b, (2e-3, 0, 0), repeat=2, copy=True)
        assert isinstance(result, list)
        assert len(result) == 3  # original + 2 copies
        assert result[0] is b  # first element is the original

    def test_translate_repeat_unite(self):
        _occ()
        from magnelio.geo.operations import Union
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        result = translate(b, (2e-3, 0, 0), repeat=3, unite=True)
        assert isinstance(result, Union)
        assert result._occ_shape() is not None

    def test_rotate_repeat(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import rotate

        b = Brick(origin=(1e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        copies = rotate(b, axis=(0, 0, 1), angle_deg=90, repeat=3)
        assert isinstance(copies, list)
        assert len(copies) == 3  # at 90°, 180°, 270°

    def test_rotate_repeat_with_copy(self):
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import rotate

        b = Brick(origin=(1e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        result = rotate(b, axis=(0, 0, 1), angle_deg=90, repeat=3, copy=True)
        assert isinstance(result, list)
        assert len(result) == 4  # original + 3 copies
        assert result[0] is b

    def test_translate_default_unchanged(self):
        """Default translate(shape, vec) returns single shape (no list)."""
        _occ()
        from magnelio.geo.primitives import Brick
        from magnelio.geo.transforms import translate

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        result = translate(b, (1e-3, 0, 0))
        assert not isinstance(result, (list, tuple))
        assert result.material is b.material


class TestOCCExtractCriticalPlanes:
    def _approx_in(self, value, lst, abs_tol=1e-6):
        """Return True if any element of lst is within abs_tol of value."""
        return any(abs(v - value) < abs_tol for v in lst)

    def test_brick_planes_correct(self):
        _occ()
        from magnelio.geo._occ_backend import extract_critical_planes
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(1e-3, 2e-3, 3e-3), size=(4e-3, 5e-3, 6e-3), material=_air())
        planes = extract_critical_planes([b])
        assert self._approx_in(1e-3, planes["x"])
        assert self._approx_in(5e-3, planes["x"])
        assert self._approx_in(2e-3, planes["y"])
        assert self._approx_in(7e-3, planes["y"])

    def test_two_bricks_planes_merged(self):
        _occ()
        from magnelio.geo._occ_backend import extract_critical_planes
        from magnelio.geo.primitives import Brick

        b1 = Brick(origin=(0, 0, 0), size=(5e-3, 5e-3, 5e-3), material=_air())
        b2 = Brick(origin=(5e-3, 0, 0), size=(5e-3, 5e-3, 5e-3), material=_air())
        planes = extract_critical_planes([b1, b2])
        xs = sorted(planes["x"])
        assert xs[0] == pytest.approx(0.0, abs=1e-6)
        assert xs[-1] == pytest.approx(10e-3, abs=1e-6)

    def test_geometry_model_planes(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.geo._occ_backend import extract_critical_planes

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 2e-3), material=_air()))
        planes = extract_critical_planes(m)
        assert any(abs(v - 4e-3) < 1e-6 for v in planes["x"])


class TestGeometryQueriesIgnoreTriangulation:
    """Geometric queries must not read rendering state (KB-012).

    ``JupyterRenderer.DisplayShape`` (``model.plot()``) tessellates the
    cached solids in place, and ``BRepBndLib::Add`` prefers a present
    triangulation — whose node box, enlarged by the tessellation
    deflection, differs from the analytic face box by whole tenths of
    a millimetre.  The plane-extraction trim filter then admits
    tangent candidates of surface regions the trimmed face does not
    cover, and the same model meshed after a ``plot()`` got a
    different grid (``N_y`` 68 -> 75 on the coupler).  All bbox reads
    that feed meshing or classification pass
    ``useTriangulation=False`` now.
    """

    def test_critical_planes_invariant_under_tessellation(self):
        _occ()
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

        from magnelio.geo._occ_backend import extract_critical_planes_per_shape
        from magnelio.geo.primitives import Brick, Cylinder

        # A cylinder patch trimmed 10 µm short of its x = +5 mm tangent:
        # the analytic face box rejects the tangent candidate, while a
        # coarse triangulation box (delta ~0.04 mm at 2 mm deflection)
        # would admit it.
        cyl = Cylinder(origin=(0.0, 0.0, 0.0), axis="z", height=10e-3, radius=5e-3, material=_air())
        cut = Brick(origin=(4.99e-3, -10e-3, -1e-3), size=(10e-3, 20e-3, 12e-3), material=_air())
        shape = cyl - cut

        ((_, before),) = extract_critical_planes_per_shape([shape], scale=1.0)
        x_faces = [p for p, exact in before["x"] if exact]
        assert not any(abs(v - 5e-3) < 1e-6 for v in x_faces), (
            "the tangent of the removed region must not be a plane"
        )

        # Tessellate the cached solid in place, as DisplayShape does.
        BRepMesh_IncrementalMesh(shape._occ_shape(1.0), 2e-3)

        ((_, after),) = extract_critical_planes_per_shape([shape], scale=1.0)
        assert after == before


class TestOCCPointInShape:
    def test_point_inside_brick(self):
        _occ()
        from magnelio.geo._occ_backend import point_in_shape
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=_air())
        assert point_in_shape(b._occ_shape(), (5e-3, 5e-3, 5e-3)) is True

    def test_point_outside_brick(self):
        _occ()
        from magnelio.geo._occ_backend import point_in_shape
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=_air())
        assert point_in_shape(b._occ_shape(), (20e-3, 5e-3, 5e-3)) is False

    def test_point_inside_sphere(self):
        _occ()
        from magnelio.geo._occ_backend import point_in_shape
        from magnelio.geo.primitives import Sphere

        s = Sphere(center=(0, 0, 0), radius=5e-3, material=_air())
        assert point_in_shape(s._occ_shape(), (0, 0, 0)) is True
        assert point_in_shape(s._occ_shape(), (4e-3, 0, 0)) is True
        assert point_in_shape(s._occ_shape(), (6e-3, 0, 0)) is False

    def test_point_inside_cylinder(self):
        _occ()
        from magnelio.geo._occ_backend import point_in_shape
        from magnelio.geo.primitives import Cylinder

        c = Cylinder(origin=(0, 0, 0), radius=2e-3, height=5e-3, axis="z", material=_air())
        assert point_in_shape(c._occ_shape(), (0, 0, 2.5e-3)) is True
        assert point_in_shape(c._occ_shape(), (3e-3, 0, 2.5e-3)) is False

    def test_point_outside_after_difference(self):
        _occ()
        from magnelio.geo._occ_backend import point_in_shape
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick, Cylinder

        # Box with cylindrical hole along z at centre
        base = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=_air())
        hole = Cylinder(origin=(5e-3, 5e-3, 0), radius=2e-3, height=10e-3, material=_air())
        diff = Difference(base, hole)
        s = diff._occ_shape()
        # Far from hole: inside
        assert point_in_shape(s, (1e-3, 1e-3, 5e-3)) is True
        # Inside the hole: outside
        assert point_in_shape(s, (5e-3, 5e-3, 5e-3)) is False


class TestCrossSectionPolygons:
    def test_brick_rectangle(self):
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(2e-3, 3e-3, 4e-3), material=_air())
        polys = cross_section_polygons(b._occ_shape(), "z", 1e-3)
        assert len(polys) == 1
        p = polys[0]
        assert p.shape[1] == 2
        assert p.shape[0] == 4  # rectangle
        np.testing.assert_allclose(p[:, 0].min(), 0.0, atol=1e-9)
        np.testing.assert_allclose(p[:, 0].max(), 2e-3, atol=1e-9)
        np.testing.assert_allclose(p[:, 1].min(), 0.0, atol=1e-9)
        np.testing.assert_allclose(p[:, 1].max(), 3e-3, atol=1e-9)

    def test_cylinder_circle(self):
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.primitives import Cylinder

        r = 5e-3
        c = Cylinder(origin=(0, 0, 0), radius=r, height=20e-3, axis="z", material=_air())
        polys = cross_section_polygons(c._occ_shape(), "z", 10e-3)
        assert len(polys) == 1
        p = polys[0]
        radii = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        np.testing.assert_allclose(radii, r, atol=1e-6)
        assert p.shape[0] >= 20  # sufficient tessellation

    def test_csg_difference_two_contours(self):
        _occ()
        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick

        outer = Brick(origin=(-5e-3, -5e-3, 0), size=(10e-3, 10e-3, 20e-3), material=_air())
        inner = Brick(origin=(-1e-3, -1e-3, 0), size=(2e-3, 2e-3, 20e-3), material=_air())
        diff = Difference(outer, inner)
        polys = cross_section_polygons(diff._occ_shape(), "z", 5e-3)
        assert len(polys) == 2  # outer boundary + inner hole

    def test_no_intersection_returns_empty(self):
        _occ()
        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        polys = cross_section_polygons(b._occ_shape(), "z", 5e-3)
        assert polys == []

    def test_x_normal_plane(self):
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.primitives import Brick

        b = Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 3e-3), material=_air())
        polys = cross_section_polygons(b._occ_shape(), "x", 2e-3)
        assert len(polys) == 1
        p = polys[0]
        # x-normal: u=y, v=z
        np.testing.assert_allclose(p[:, 0].min(), 0.0, atol=1e-9)
        np.testing.assert_allclose(p[:, 0].max(), 2e-3, atol=1e-9)
        np.testing.assert_allclose(p[:, 1].min(), 0.0, atol=1e-9)
        np.testing.assert_allclose(p[:, 1].max(), 3e-3, atol=1e-9)


# ── Cross-section cell classification (even-odd rule, KB-002 fix) ────────────


class TestClassifyCellsEvenOdd:
    """Verify that classify_cells_from_cross_sections uses the even-odd rule
    so that CSG Difference shapes with holes are classified correctly."""

    def test_difference_hole_is_background(self):
        """Cells inside the hole of a Difference shape must be background."""
        _occ()
        import numpy as np

        from magnelio.geo._filling import classify_cells_from_cross_sections
        from magnelio.geo._occ_backend import batch_cross_sections
        from magnelio.geo.operations import Difference
        from magnelio.geo.primitives import Brick
        from magnelio.mesh.grid import GridLines

        pec = Material.pec()
        air = Material.air()

        # PEC brick [0, 10] x [0, 10] x [0, 10] mm with air hole [3, 7]^3
        outer = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=pec)
        inner = Brick(origin=(3e-3, 3e-3, 3e-3), size=(4e-3, 4e-3, 4e-3), material=air)
        shell = Difference(outer, inner)

        # Coarse grid: 5 cells per axis → cell size 2 mm
        x = np.linspace(0, 10e-3, 6)
        y = np.linspace(0, 10e-3, 6)
        z = np.linspace(0, 10e-3, 6)
        grid = GridLines(x=x, y=y, z=z)

        shapes_with_material = [(shell, 1)]  # material_id=1 for PEC shell
        xc = 0.5 * (x[:-1] + x[1:])
        cache = batch_cross_sections(shapes_with_material, {"x": xc}, deflection=1e-4)
        mat_id = classify_cells_from_cross_sections(cache, grid, background_id=0)

        # Cell centres at 1, 3, 5, 7, 9 mm
        # Cells at 5 mm (index 2) are inside the hole → must be background (0)
        assert mat_id[2, 2, 2] == 0, "Cell inside hole should be background"
        # Cells at 1 mm (index 0) are inside the shell → must be PEC (1)
        assert mat_id[0, 0, 0] == 1, "Cell in shell wall should be PEC"

    def test_simple_shape_unaffected(self):
        """A simple Brick (no holes) should still classify correctly."""
        _occ()
        import numpy as np

        from magnelio.geo._filling import classify_cells_from_cross_sections
        from magnelio.geo._occ_backend import batch_cross_sections
        from magnelio.geo.primitives import Brick
        from magnelio.mesh.grid import GridLines

        mat = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        brick = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=mat)

        x = np.linspace(0, 4e-3, 3)
        y = np.linspace(0, 4e-3, 3)
        z = np.linspace(0, 4e-3, 3)
        grid = GridLines(x=x, y=y, z=z)

        shapes_with_material = [(brick, 1)]
        xc = 0.5 * (x[:-1] + x[1:])
        cache = batch_cross_sections(shapes_with_material, {"x": xc}, deflection=1e-4)
        mat_id = classify_cells_from_cross_sections(cache, grid, background_id=0)

        # All cells inside the brick should be material 1
        assert np.all(mat_id == 1), "All cells inside simple brick should have mat_id=1"


# ── Mesh.from_geometry() integration ─────────────────────────────────────────


class TestMeshFromGeometry:
    def test_brick_fills_material(self):
        """A single Brick should fill the correct cells with its material."""
        _occ()
        import numpy as np

        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        # 4×4×4 mm box of FR4
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=fr4))

        ctrl = MeshControl(
            min_nodes_per_wavelength=4,  # coarse — fast test
            max_cell_size=2e-3,
        )
        mesh = Mesh.from_geometry(model, ctrl, f_max=10e9)

        # FR4 material should appear in the library
        fr4_ids = [mid for mid, m in mesh.material_library.items() if m.name == "FR4"]
        assert len(fr4_ids) == 1, "FR4 not found in material_library"
        fr4_id = fr4_ids[0]

        # At least some cells should be FR4
        assert np.any(mesh.material_id == fr4_id), "No cells assigned to FR4"

    def test_grid_contains_critical_planes(self):
        """Grid lines must include the brick face positions (within OCC bbox tolerance)."""
        _occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        mat = Material.air()
        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(6e-3, 4e-3, 2e-3), material=mat))

        ctrl = MeshControl(min_nodes_per_wavelength=4, max_cell_size=3e-3)
        mesh = Mesh.from_geometry(model, ctrl, f_max=10e9)

        # 0 and 6e-3 must appear in grid lines on x (within 1 µm)
        assert any(abs(v) < 1e-6 for v in mesh.grid.x)
        assert any(abs(v - 6e-3) < 1e-6 for v in mesh.grid.x)

    def test_background_is_air(self):
        """Cells outside all shapes should be material_id 0 (air)."""
        _occ()
        import numpy as np

        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        # Small sphere inside a larger air domain by using forced_planes
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        model = GeometryModel()
        # Brick occupies only [1,3]mm in x; domain [0,4]mm via forced_planes
        model.add(Brick(origin=(1e-3, 0, 0), size=(2e-3, 4e-3, 4e-3), material=fr4))

        ctrl = MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=2e-3,
            forced_planes={"x": [0.0, 4e-3]},
        )
        mesh = Mesh.from_geometry(model, ctrl, f_max=10e9)

        # material_id 0 = air must be present somewhere
        assert np.any(mesh.material_id == 0)

    def test_difference_material_assignment(self):
        """Cells inside a subtracted volume should remain air (not FR4)."""
        _occ()
        import numpy as np

        from magnelio.geo import Brick, GeometryModel
        from magnelio.geo.operations import Difference
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        base = Brick(origin=(0, 0, 0), size=(6e-3, 6e-3, 6e-3), material=fr4)
        # Remove right half
        tool = Brick(origin=(3e-3, 0, 0), size=(3e-3, 6e-3, 6e-3), material=Material.air())
        diff = Difference(base, tool)

        model = GeometryModel()
        model.add(diff)

        ctrl = MeshControl(min_nodes_per_wavelength=4, max_cell_size=1.5e-3)
        mesh = Mesh.from_geometry(model, ctrl, f_max=10e9)

        fr4_ids = [mid for mid, m in mesh.material_library.items() if m.name == "FR4"]
        assert fr4_ids, "FR4 not in library"
        fr4_id = fr4_ids[0]

        xc = 0.5 * (mesh.grid.x[:-1] + mesh.grid.x[1:])
        # Cells with x > 3e-3 should not be FR4 (they were subtracted)
        right_half = mesh.material_id[xc > 3.5e-3, :, :]
        assert not np.any(right_half == fr4_id), "Subtracted region still has FR4"

    def test_feature_resolution_rect_coax(self):
        """Feature-based resolution ensures >=4 cells across inner conductor."""
        _occ()
        import numpy as np

        from magnelio.geo import Brick, GeometryModel
        from magnelio.geo.operations import Difference
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        a, b, L = 2e-3, 10e-3, 10e-3
        ptfe = Material.from_isotropic("PTFE", epsilon=2.1)
        pec = Material.pec()

        model = GeometryModel()
        ptfe_full = Brick(origin=(-b / 2, -b / 2, 0), size=(b, b, L), material=ptfe)
        inner = Brick(origin=(-a / 2, -a / 2, 0), size=(a, a, L), material=pec)
        model.add(Difference(ptfe_full, inner, material=ptfe))
        model.add(inner)

        # Default MeshControl — no max_cell_size override
        mesh = Mesh.from_geometry(model, MeshControl(), f_max=10e9)
        grid = mesh.grid

        # Inner conductor spans [-1mm, +1mm]. Count cells fully inside.
        tol = 1e-9
        cells_inner_x = int(np.sum((grid.x[:-1] >= -a / 2 - tol) & (grid.x[1:] <= a / 2 + tol)))
        assert cells_inner_x >= 4, f"Expected >= 4 cells in inner conductor, got {cells_inner_x}"


# ── Background-region feature resolution (WP2.1, finding F1) ─────────────────


class TestBackgroundFeatureResolution:
    """Regions exposed by ``Difference`` (background material behind a
    hole) must contribute critical planes and feature refinement, even
    without an explicit shape covering them.
    """

    def test_difference_hole_contributes_critical_planes(self):
        """Interior hole faces must appear as critical planes."""
        _occ()
        from magnelio.geo import Brick, Difference
        from magnelio.geo._occ_backend import extract_critical_planes
        from magnelio.materials.material import Material

        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        base = Brick(origin=(0, 0, 0), size=(6e-3, 6e-3, 6e-3), material=fr4)
        hole = Brick(origin=(2e-3, 2e-3, 0), size=(2e-3, 2e-3, 6e-3), material=Material.air())

        critical = extract_critical_planes([Difference(base, hole)])
        for pos in (0.0, 2e-3, 4e-3, 6e-3):
            assert any(abs(v - pos) < 1e-9 for v in critical["x"]), (
                f"critical x-plane at {pos} missing: {sorted(set(critical['x']))}"
            )

    def test_background_pec_coax_inner_conductor_resolved(self):
        """F1 repro: coax without explicit inner-PEC shape.

        The inner conductor exists only as a ``Difference`` hole exposing
        the PEC background.  The mesher must still resolve it: grid lines
        snapped to ±r_i, >= 3 cells across D_i, and at least one grid
        node strictly inside the inner conductor.
        """
        _occ()
        import numpy as np

        from magnelio.geo import Cylinder, Difference, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh, MeshControl

        D_i, D_a, L = 0.41e-3, 5.0e-3, 10.0e-3
        r_i = D_i / 2
        pec = Material.pec()
        diel = Material.from_isotropic("alumina", epsilon=9.0)

        out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
        in_cyl = Cylinder(origin=(0, 0, 0), radius=r_i, height=L, axis="z", material=pec)
        model = GeometryModel(background=pec)
        model.add(Difference(out_cyl, in_cyl))  # no explicit inner shape

        mesh = Mesh.from_geometry(model, MeshControl(), f_max=10e9)
        grid = mesh.grid

        # Grid lines snapped to the inner-conductor tangent planes ±r_i
        for pos in (-r_i, r_i):
            assert any(abs(v - pos) < 1e-6 for v in grid.x)
            assert any(abs(v - pos) < 1e-6 for v in grid.y)

        # >= 3 cells across D_i on both transversal axes
        tol = 1e-9
        for nodes in (grid.x, grid.y):
            cells_across = int(np.sum((nodes[:-1] >= -r_i - tol) & (nodes[1:] <= r_i + tol)))
            assert cells_across >= 3, f"Expected >= 3 cells across D_i, got {cells_across}"

        # At least one grid node strictly inside the inner conductor
        xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
        assert np.any(xx**2 + yy**2 < r_i**2), "No grid node inside the inner conductor"


# ── 3D edge-PEC intersection (compute_edge_pec_fractions) ────────────────────


class TestEdgePecFractions:
    def test_edge_outside_box(self):
        """Edge completely outside PEC box → f_L = 1."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        pec_box = Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 2e-3), material=_air())
        edges = np.array([[[5e-3, 1e-3, 1e-3], [8e-3, 1e-3, 1e-3]]])
        f_L = compute_edge_pec_fractions([pec_box._occ_shape()], edges)
        assert f_L.shape == (1,)
        assert f_L[0] == pytest.approx(1.0, abs=1e-6)

    def test_edge_inside_box(self):
        """Edge completely inside PEC box → f_L = 0."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        pec_box = Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=_air())
        edges = np.array([[[2e-3, 5e-3, 5e-3], [8e-3, 5e-3, 5e-3]]])
        f_L = compute_edge_pec_fractions([pec_box._occ_shape()], edges)
        assert f_L[0] == pytest.approx(0.0, abs=1e-6)

    def test_edge_half_in_box(self):
        """Edge halfway inside PEC box → f_L ≈ 0.5."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        # Box from x=0..4mm. Edge from x=-2..+2mm (half outside, half inside).
        pec_box = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air())
        edges = np.array([[[-2e-3, 2e-3, 2e-3], [2e-3, 2e-3, 2e-3]]])
        f_L = compute_edge_pec_fractions([pec_box._occ_shape()], edges)
        assert f_L[0] == pytest.approx(0.5, abs=0.01)

    def test_edge_through_cylinder(self):
        """Edge through a cylinder — compare with analytical chord length."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Cylinder

        # Cylinder: radius=5mm, along z, centred at origin
        r = 5e-3
        h = 10e-3
        cyl = Cylinder(origin=(0, 0, 0), radius=r, height=h, axis="z", material=_air())

        # Edge along y at z=h/2, from y=-8mm to y=+8mm, x=0
        # Chord at x=0: enters at y=-5mm, exits at y=+5mm
        # Total length = 16mm, inside = 10mm, outside = 6mm → f_L = 6/16
        y0, y1 = -8e-3, 8e-3
        edges = np.array([[[0.0, y0, h / 2], [0.0, y1, h / 2]]])
        f_L = compute_edge_pec_fractions([cyl._occ_shape()], edges)
        expected = (16e-3 - 10e-3) / 16e-3  # 6/16 = 0.375
        assert f_L[0] == pytest.approx(expected, abs=0.01)

    def test_edge_through_offset_cylinder(self):
        """Edge through cylinder at offset — analytical chord length."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Cylinder

        # Cylinder radius=5mm at origin, edge along y at x=3mm
        # Chord half-length = sqrt(25 - 9) = 4mm → chord = 8mm
        r = 5e-3
        h = 10e-3
        x_off = 3e-3
        cyl = Cylinder(origin=(0, 0, 0), radius=r, height=h, axis="z", material=_air())

        y0, y1 = -8e-3, 8e-3
        total = y1 - y0
        chord = 2 * np.sqrt(r**2 - x_off**2)
        edges = np.array([[[x_off, y0, h / 2], [x_off, y1, h / 2]]])
        f_L = compute_edge_pec_fractions([cyl._occ_shape()], edges)
        expected = (total - chord) / total
        assert f_L[0] == pytest.approx(expected, abs=0.01)

    def test_no_pec_shapes(self):
        """No PEC shapes → all f_L = 1."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions

        edges = np.array([[[0, 0, 0], [1e-3, 0, 0]]])
        f_L = compute_edge_pec_fractions([], edges)
        assert f_L[0] == pytest.approx(1.0)

    def test_empty_edges(self):
        """Empty edge array → empty result."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        pec_box = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        edges = np.empty((0, 2, 3))
        f_L = compute_edge_pec_fractions([pec_box._occ_shape()], edges)
        assert len(f_L) == 0

    def test_multiple_edges_batch(self):
        """Multiple edges in one call — verify batch processing."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        # Box from (0,0,0) to (4,4,4) mm
        pec_box = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air())
        edges = np.array(
            [
                [[5e-3, 2e-3, 2e-3], [8e-3, 2e-3, 2e-3]],  # outside
                [[1e-3, 2e-3, 2e-3], [3e-3, 2e-3, 2e-3]],  # inside
                [[-2e-3, 2e-3, 2e-3], [2e-3, 2e-3, 2e-3]],  # half in
            ]
        )
        f_L = compute_edge_pec_fractions([pec_box._occ_shape()], edges)
        assert f_L[0] == pytest.approx(1.0, abs=0.01)
        assert f_L[1] == pytest.approx(0.0, abs=0.01)
        assert f_L[2] == pytest.approx(0.5, abs=0.01)

    def test_two_pec_shapes_fused(self):
        """Two overlapping PEC boxes — fused correctly."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_edge_pec_fractions
        from magnelio.geo.primitives import Brick

        # Two overlapping boxes: (0..3mm) and (2..5mm) along x → fused 0..5mm
        b1 = Brick(origin=(0, 0, 0), size=(3e-3, 4e-3, 4e-3), material=_air())
        b2 = Brick(origin=(2e-3, 0, 0), size=(3e-3, 4e-3, 4e-3), material=_air())
        # Edge from x=-2mm to x=+7mm at y=z=2mm, total=9mm, inside=5mm
        edges = np.array([[[-2e-3, 2e-3, 2e-3], [7e-3, 2e-3, 2e-3]]])
        f_L = compute_edge_pec_fractions(
            [b1._occ_shape(), b2._occ_shape()],
            edges,
        )
        expected = (9e-3 - 5e-3) / 9e-3
        assert f_L[0] == pytest.approx(expected, abs=0.02)


class TestFaceMaterialAreasAnalytic:
    """Validate compute_face_material_areas against closed-form solutions."""

    def test_pec_cylinder_eps_face_average(self):
        """A PEC cylinder of radius R intersects a square face: the area-weighted
        effective εᵣ on that face is (face_area − π·R²) / face_area, since PEC
        claims the disk area but contributes 0 to the εᵣ weighted sum and the
        remainder fills with the air background (εᵣ = 1).
        """
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_face_material_areas
        from magnelio.geo.primitives import Cylinder
        from magnelio.materials.material import Material

        R = 2e-3
        face_half = 5e-3  # face spans [-5, +5] mm in u and v
        face_area = (2 * face_half) ** 2
        pec = Material.pec()
        air = Material.air()
        cyl = Cylinder(
            origin=(0, 0, -10e-3),
            radius=R,
            height=20e-3,
            axis="z",
            material=pec,
        )
        material_library = {0: air, 1: pec}
        shapes_with_material = [(cyl, 1)]
        face_specs = np.array([[0.0, -face_half, -face_half, face_half, face_half]])
        face_axes = np.array([2], dtype=np.int32)  # z-normal
        eps = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop="epsilon",
        )
        expected = (face_area - math.pi * R * R) / face_area
        assert eps[0] == pytest.approx(expected, rel=1e-3), (
            f"got {eps[0]:.6f}, expected {expected:.6f}"
        )

    def test_dielectric_cylinder_eps_average(self):
        """A dielectric cylinder (εᵣ = 4) inside an air box: face εᵣ is the
        area-weighted blend (eps_diel·A_diel + 1·A_air) / face_area.
        """
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_face_material_areas
        from magnelio.geo.primitives import Cylinder
        from magnelio.materials.material import Material

        R = 1.5e-3
        eps_r = 4.0
        face_half = 4e-3
        face_area = (2 * face_half) ** 2
        diel = Material.from_isotropic(name="DIEL", epsilon=eps_r)
        air = Material.air()
        cyl = Cylinder(
            origin=(0, 0, -10e-3),
            radius=R,
            height=20e-3,
            axis="z",
            material=diel,
        )
        material_library = {0: air, 1: diel}
        shapes_with_material = [(cyl, 1)]
        face_specs = np.array([[0.0, -face_half, -face_half, face_half, face_half]])
        face_axes = np.array([2], dtype=np.int32)
        eps = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop="epsilon",
        )
        A_diel = math.pi * R * R
        expected = (eps_r * A_diel + 1.0 * (face_area - A_diel)) / face_area
        assert eps[0] == pytest.approx(expected, rel=1e-3), (
            f"got {eps[0]:.6f}, expected {expected:.6f}"
        )

    def test_face_fully_outside_shape(self):
        """A face completely outside the geometry: returns the background εᵣ."""
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import compute_face_material_areas
        from magnelio.geo.primitives import Cylinder
        from magnelio.materials.material import Material

        diel = Material.from_isotropic(name="DIEL", epsilon=4.0)
        air = Material.air()
        cyl = Cylinder(
            origin=(0, 0, -10e-3),
            radius=1e-3,
            height=20e-3,
            axis="z",
            material=diel,
        )
        material_library = {0: air, 1: diel}
        shapes_with_material = [(cyl, 1)]
        face_specs = np.array([[0.0, 100e-3, 100e-3, 110e-3, 110e-3]])
        face_axes = np.array([2], dtype=np.int32)
        eps = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop="epsilon",
        )
        assert eps[0] == pytest.approx(1.0, rel=1e-9)


class TestTangentPlaneClassification:
    """DD-106: deterministic conventions on planes tangent to a
    material-boundary face, where the exact-plane section is ill-posed.

    Matrix channel (pec_area_out → A_face_free → M_μ categories):
    min-convention — a face is blocked only where it is embedded in
    PEC on both sides; a merely tangential wall leaves it free.
    Geometric channel: max-convention (wall booked into the adjacent
    non-PEC cell) with the flat-wall jump registered.  Both must be
    translation-invariant along an extruded feed.
    """

    def _run(self, shapes_with_material, material_library, plane_y, n_z, domain_bounds=None):
        import numpy as np

        from magnelio.geo._occ_backend import compute_face_material_areas

        dz = 1e-3
        face_specs = np.array(
            [[plane_y, 0.0, k * dz, 1e-3, (k + 1) * dz] for k in range(n_z)],
            dtype=np.float64,
        )
        face_axes = np.full(n_z, 1, dtype=np.int32)
        pec_area = np.zeros(n_z)
        pec_geom = np.zeros(n_z)
        pec_jump = np.zeros(n_z)
        mu = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop="mu",
            deflection=1e-5,
            pec_area_out=pec_area,
            pec_area_geom_out=pec_geom,
            pec_area_jump_out=pec_jump,
            domain_bounds=domain_bounds,
        )
        area = 1e-3 * dz
        return mu, pec_area / area, pec_geom / area, pec_jump / area

    def test_interior_union_tangent_plane_invariant(self):
        """A Union's interior flat face on the section plane (no shape
        bbox witnesses it — the historical detection gap): the wall
        behind an air chamber, pierced by one slot.  Along the
        invariant stretches the matrix channel must be exactly free
        and the geometric channel must book the wall; the slot window
        opens both.  Pre-DD-106 this returned a slanted 0→1 front.
        """
        _occ()
        import numpy as np

        from magnelio.geo.operations import Union
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        pec = Material.pec()
        air = Material.air()
        chamber = Brick.from_corners((0, 0, 0), (10e-3, 5e-3, 40e-3), material=air)
        slot = Brick.from_corners((0, 5e-3, 10e-3), (3e-3, 6e-3, 12e-3), material=air)
        vac = Union(chamber, slot)
        mu, pec_frac, geom_frac, jump_frac = self._run([(vac, 1)], {0: pec, 1: air}, 5e-3, 40)
        slot_cells = np.zeros(40, dtype=bool)
        slot_cells[10:12] = True
        # Matrix: tangential wall never blocks — free everywhere.
        assert np.allclose(pec_frac, 0.0, atol=1e-12)
        assert np.allclose(mu, 1.0, atol=1e-12)
        # Geometric: wall booked except in the slot window, jump too.
        assert np.allclose(geom_frac[~slot_cells], 1.0, atol=1e-9)
        assert np.allclose(geom_frac[slot_cells], 0.0, atol=1e-9)
        assert np.allclose(jump_frac[~slot_cells], 1.0, atol=1e-9)
        assert np.allclose(jump_frac[slot_cells], 0.0, atol=1e-9)

    def test_bbox_tangent_plane_matches_interior_case(self):
        """The same wall built from separate bricks (bbox-tangent
        detection fires): identical channel values as the Union case —
        the convention must not depend on how the solid was composed.
        """
        _occ()
        import numpy as np

        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        pec = Material.pec()
        air = Material.air()
        chamber = Brick.from_corners((0, 0, 0), (10e-3, 5e-3, 40e-3), material=air)
        slot = Brick.from_corners((0, 5e-3, 10e-3), (3e-3, 6e-3, 12e-3), material=air)
        mu, pec_frac, geom_frac, jump_frac = self._run(
            [(chamber, 1), (slot, 1)], {0: pec, 1: air}, 5e-3, 40
        )
        slot_cells = np.zeros(40, dtype=bool)
        slot_cells[10:12] = True
        assert np.allclose(pec_frac, 0.0, atol=1e-12)
        assert np.allclose(geom_frac[~slot_cells], 1.0, atol=1e-9)
        assert np.allclose(geom_frac[slot_cells], 0.0, atol=1e-9)
        assert np.allclose(jump_frac[~slot_cells], 1.0, atol=1e-9)

    def test_face_embedded_in_pec_is_blocked(self):
        """Two stacked PEC bricks share the section plane: the face is
        embedded in PEC on both sides — min-convention blocks it.
        """
        _occ()
        import numpy as np

        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        pec = Material.pec()
        air = Material.air()
        lower = Brick.from_corners((0, 0, 0), (10e-3, 5e-3, 40e-3), material=pec)
        upper = Brick.from_corners((0, 5e-3, 0), (10e-3, 10e-3, 40e-3), material=pec)
        mu, pec_frac, geom_frac, jump_frac = self._run(
            [(lower, 1), (upper, 1)], {0: air, 1: pec}, 5e-3, 40
        )
        assert np.allclose(pec_frac, 1.0, atol=1e-12)
        assert np.allclose(mu, 1.0, atol=1e-12)
        # No wall lies in this plane — the PEC overlap is continuous.
        assert np.allclose(jump_frac, 0.0, atol=1e-9)

    def test_dielectric_tangent_interface_side_mean(self):
        """A μ_r = 2 slab tangent to the plane from below, air above:
        the face value is the staircase cell-pair mean (μ̄ = 1.5).  On
        the domain hull the same plane is evaluated one-sided — the
        interior μ_r survives instead of being averaged with the
        fictitious outside.
        """
        _occ()
        import numpy as np

        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        air = Material.air()
        mag = Material.from_isotropic(name="MAG", mu=2.0)
        slab = Brick.from_corners((0, 0, 0), (10e-3, 5e-3, 40e-3), material=mag)
        shapes = [(slab, 1)]
        lib = {0: air, 1: mag}
        mu_mid, pec_mid, _, _ = self._run(shapes, lib, 5e-3, 5)
        assert np.allclose(mu_mid, 1.5, atol=1e-9)
        assert np.allclose(pec_mid, 0.0, atol=1e-12)
        bounds = ((0.0, 10e-3), (0.0, 5e-3), (0.0, 40e-3))
        mu_dom, pec_dom, _, _ = self._run(shapes, lib, 5e-3, 5, domain_bounds=bounds)
        assert np.allclose(mu_dom, 2.0, atol=1e-9)
        assert np.allclose(pec_dom, 0.0, atol=1e-12)


class TestParallelSectionPrefill:
    """The process-pool section prefill is bit-identical to sequential."""

    def test_worker_count_env_override(self, monkeypatch):
        from magnelio.geo._occ_backend import _section_worker_count

        monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", "0")
        assert _section_worker_count() == 0
        monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", "3")
        assert _section_worker_count() == 3
        monkeypatch.delenv("MAGNELIO_SECTION_WORKERS")
        auto = _section_worker_count()
        assert 1 <= auto <= 8
        monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", "not-a-number")
        with pytest.warns(UserWarning, match="not an integer"):
            assert _section_worker_count() == auto

    def test_pool_bit_identical_to_sequential(self, monkeypatch):
        """Same eps/PEC-area results, bit for bit, pooled vs. sequential.

        The threshold is patched to 1 so the small fixture reaches the
        prefill, and the pool-startup constant to 0 so the DD-141
        sample gate admits it (this fixture is deliberately cheap, and
        the gate would otherwise — correctly — keep it sequential and
        leave this test comparing sequential against itself).  2
        workers keep the spawn startup cheap.
        """
        _occ()
        import numpy as np

        from magnelio.geo import _occ_backend as occ_backend
        from magnelio.geo._occ_backend import compute_face_material_areas
        from magnelio.geo.primitives import Cylinder, Sphere
        from magnelio.materials.material import Material

        pec = Material.pec()
        diel = Material.from_isotropic(name="DIEL", epsilon=4.0)
        air = Material.air()
        cyl = Cylinder(
            origin=(0, 0, -10e-3),
            radius=2e-3,
            height=20e-3,
            axis="z",
            material=pec,
        )
        sph = Sphere(center=(3e-3, 0, 0), radius=2.5e-3, material=diel)
        material_library = {0: air, 1: pec, 2: diel}
        shapes_with_material = [(cyl, 1), (sph, 2)]
        # 30 z-normal faces at distinct plane positions through both shapes
        z_planes = np.linspace(-2.4e-3, 2.4e-3, 30)
        face_specs = np.array([[z, -5e-3, -5e-3, 5e-3, 5e-3] for z in z_planes])
        face_axes = np.full(len(z_planes), 2, dtype=np.int32)

        def run():
            pec_area = np.zeros(len(z_planes))
            pec_geom = np.zeros(len(z_planes))
            pec_jump = np.zeros(len(z_planes))
            eps = compute_face_material_areas(
                shapes_with_material,
                material_library,
                face_specs,
                face_axes,
                prop="epsilon",
                pec_area_out=pec_area,
                pec_area_geom_out=pec_geom,
                pec_area_jump_out=pec_jump,
            )
            return eps, pec_area, pec_geom, pec_jump

        monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", "0")
        seq = run()
        monkeypatch.setattr(occ_backend, "_SECTION_PARALLEL_MIN_QUERIES", 1)
        monkeypatch.setattr(occ_backend, "_SECTION_POOL_STARTUP_S", 0.0)
        monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", "2")
        pooled = run()
        for a_seq, a_pool in zip(seq, pooled):
            np.testing.assert_array_equal(a_seq, a_pool)

    def test_sample_gate_keeps_cheap_batches_sequential(self, monkeypatch):
        """DD-141: a batch that would not repay the pool is not pooled.

        The admission test upstream scores cost by face count, which
        over-estimates by an order of magnitude on some geometry
        classes; the sample decides on measured time instead.  Cheap
        queries must come back as "nothing left to pool" — with the
        sample's own results already in the cache, so the work is not
        repeated.
        """
        _occ()

        from magnelio.geo import _occ_backend as occ_backend
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        brick = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.pec())
        shapes = [(brick, 1)]
        queries = [(2, 0.5e-3 * (k + 1), 0) for k in range(6)]
        cache: dict = {}

        remaining = occ_backend._sample_and_admit(
            list(queries),
            shapes,
            1e-4,
            cache,
            lambda p: p,
            1.0,
            8,
        )
        assert remaining == []
        # Sampling is progress, not overhead: what it computed is cached.
        assert cache
        assert set(cache) <= set(queries)

    def test_sample_gate_admits_expensive_batches(self, monkeypatch):
        """A batch whose projected cost clears the bar is handed on."""
        _occ()

        from magnelio.geo import _occ_backend as occ_backend
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        brick = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.pec())
        shapes = [(brick, 1)]
        queries = [(2, 0.5e-3 + 1e-5 * k, 0) for k in range(60)]
        cache: dict = {}

        # "The pool is free": every non-empty remainder then clears the
        # bar, which isolates the accounting from the machine's speed.
        monkeypatch.setattr(occ_backend, "_SECTION_POOL_STARTUP_S", 0.0)
        remaining = occ_backend._sample_and_admit(
            list(queries),
            shapes,
            1e-4,
            cache,
            lambda p: p,
            1.0,
            8,
        )
        assert remaining
        # The sample and the remainder partition the batch exactly —
        # nothing computed twice, nothing dropped.
        assert len(remaining) + len(cache) == len(queries)
        assert not (set(remaining) & set(cache))

    def test_sample_is_spread_not_taken_from_the_front(self, monkeypatch):
        """The schedule is cost-sorted, so a head sample would mislead.

        Expensive queries are deliberately scheduled first; timing only
        those would make every batch look worth parallelising.
        """
        _occ()

        from magnelio.geo import _occ_backend as occ_backend
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        brick = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.pec())
        shapes = [(brick, 1)]
        n = 4 * occ_backend._SECTION_SAMPLE_QUERIES
        queries = [(2, 0.5e-3 + 1e-6 * k, 0) for k in range(n)]
        cache: dict = {}
        monkeypatch.setattr(occ_backend, "_SECTION_POOL_STARTUP_S", 0.0)
        occ_backend._sample_and_admit(list(queries), shapes, 1e-4, cache, lambda p: p, 1.0, 8)
        sampled = sorted(queries.index(q) for q in cache)
        assert len(sampled) == occ_backend._SECTION_SAMPLE_QUERIES
        # Spread across the whole batch, not clustered at its head.
        assert max(sampled) > n // 2

    def test_unguarded_script_does_not_reexecute(self, tmp_path):
        """A user script without ``if __name__ == "__main__":`` stays sane.

        ``spawn`` asks each worker to rebuild the parent's main module
        by re-running the script — which used to replay the whole
        simulation once per worker and then warn about the nested pool
        it tried to start.  This can only be observed from a real child
        process, hence the subprocess.  The marker must appear exactly
        once and the cache must come back fully pooled.
        """
        _occ()
        import os
        import subprocess
        import sys
        from pathlib import Path

        import magnelio

        script = tmp_path / "unguarded.py"
        script.write_text(
            "import os\n"
            "print('MARK', os.getpid(), flush=True)\n"
            "import numpy as np\n"
            "from magnelio.geo import _occ_backend as occ_backend\n"
            "from magnelio.geo.primitives import Cylinder\n"
            "from magnelio.materials.material import Material\n"
            "occ_backend._SECTION_PARALLEL_MIN_QUERIES = 1\n"
            "cyl = Cylinder(origin=(0, 0, -10e-3), radius=2e-3,\n"
            "               height=20e-3, axis='z', material=Material.pec())\n"
            "q = [(2, float(z), 0) for z in np.linspace(-2e-3, 2e-3, 8)]\n"
            "cache = {}\n"
            "occ_backend._parallel_section_prefill(\n"
            "    [(cyl, 1)], q, 1e-4, cache,\n"
            "    lambda p: (p, (0.0, 0.0, 0.0, 0.0), 0.0))\n"
            "print('FILLED', len(cache), len(q), flush=True)\n"
        )
        env = dict(os.environ)
        env["MAGNELIO_SECTION_WORKERS"] = "2"
        src_root = str(Path(magnelio.__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join([src_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=str(tmp_path),
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert out.count("MARK") == 1, f"main module re-executed:\n{out}"
        assert "bootstrapping phase" not in out, out
        assert "FILLED 8 8" in out, out


class TestOccPrecisionGuard:
    """Solids at/below the OCC kernel precision are rejected clearly.
    Found by the WP-M5 mesher stress sentinel: a 100 nm brick dies
    deep inside OCC with a cryptic Standard_DomainError; the geometry
    layer now raises an informative ValueError instead.
    """

    def _occ(self):
        return pytest.importorskip("OCC.Core.BRepPrimAPI")

    def test_sub_precision_brick_raises(self):
        self._occ()
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-7), material=Material.pec())
        with pytest.raises(ValueError, match="OCC geometric precision"):
            b._occ_shape()

    def test_sub_precision_cylinder_raises(self):
        self._occ()
        from magnelio.geo.primitives import Cylinder
        from magnelio.materials.material import Material

        c = Cylinder(origin=(0, 0, 0), radius=5e-8, height=1e-3, axis="z", material=Material.pec())
        with pytest.raises(ValueError, match="OCC geometric precision"):
            c._occ_shape()

    def test_normal_sizes_unaffected(self):
        self._occ()
        from magnelio.geo.primitives import Brick
        from magnelio.materials.material import Material

        b = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 2e-7), material=Material.pec())
        assert b._occ_shape() is not None


# ── 1a: Brick.from_corners ───────────────────────────────────────────────────


class TestBrickFromCorners:
    """Two-corner constructor normalises min/max per axis (WP 1a)."""

    def test_normalises_swapped_corners(self):
        from magnelio.geo.primitives import Brick

        # p1 holds the larger x/z, p2 the larger y — deliberately unsorted.
        b = Brick.from_corners((3e-3, 0.0, 2e-3), (0.0, 4e-3, 0.0), material=_air())
        assert b.origin == (0.0, 0.0, 0.0)
        assert b.size == (3e-3, 4e-3, 2e-3)

    def test_already_sorted_corners(self):
        from magnelio.geo.primitives import Brick

        b = Brick.from_corners((1e-3, 2e-3, 3e-3), (5e-3, 7e-3, 9e-3), material=_air())
        assert b.origin == (1e-3, 2e-3, 3e-3)
        # size = hi - lo is a plain FP subtraction (no rounding by design).
        assert b.size == pytest.approx((4e-3, 5e-3, 6e-3), abs=1e-15)

    def test_material_and_name_carried(self):
        from magnelio.geo.primitives import Brick

        pec = Material.pec()
        b = Brick.from_corners((0, 0, 0), (1e-3, 1e-3, 1e-3), material=pec, name="pad")
        assert b.material is pec
        assert b.name == "pad"

    def test_result_is_plain_brick(self):
        """from_corners populates the same fields as the normal constructor."""
        from magnelio.geo.primitives import Brick

        a = Brick.from_corners((2e-3, 0, 0), (0, 2e-3, 2e-3), material=_air())
        b = Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 2e-3), material=_air())
        assert a.origin == b.origin and a.size == b.size

    def test_bounding_box_exact(self):
        _occ()
        from magnelio.geo.primitives import Brick

        b = Brick.from_corners((1e-3, 2e-3, 3e-3), (5e-3, 7e-3, 9e-3), material=_air())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = b.bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((1e-3, 2e-3, 3e-3), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((5e-3, 7e-3, 9e-3), abs=1e-6)


class TestBrickFromRanges:
    """Per-axis range constructor: exactly two of (a1, a2, da) per axis."""

    def test_two_bounds_per_axis(self):
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=0, x2=3e-3, y1=0, y2=4e-3, z1=0, z2=2e-3)
        assert b.origin == (0.0, 0.0, 0.0)
        assert b.size == pytest.approx((3e-3, 4e-3, 2e-3), abs=1e-15)

    def test_bound_and_extent(self):
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=1e-3, dx=2e-3, y1=0, dy=4e-3, z1=0, dz=2e-3)
        assert b.origin == pytest.approx((1e-3, 0.0, 0.0), abs=1e-15)
        assert b.size == pytest.approx((2e-3, 4e-3, 2e-3), abs=1e-15)

    def test_upper_bound_and_extent_grows_downwards(self):
        """x2 + dx without x1 puts the box below the given plane."""
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=0, dx=1e-3, y1=0, dy=1e-3, z2=0.0, dz=1.6e-3)
        assert b.origin[2] == pytest.approx(-1.6e-3, abs=1e-15)
        assert b.size[2] == pytest.approx(1.6e-3, abs=1e-15)

    def test_mixed_spellings_on_different_axes(self):
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=0, x2=3e-3, y1=0, dy=4e-3, z2=2e-3, dz=2e-3)
        assert b.origin == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)
        assert b.size == pytest.approx((3e-3, 4e-3, 2e-3), abs=1e-15)

    def test_negative_extent_is_normalised(self):
        """A negative extent extrudes backwards, like Cylinder's height."""
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=5e-3, dx=-2e-3, y1=0, dy=1e-3, z1=0, dz=1e-3)
        assert b.origin[0] == pytest.approx(3e-3, abs=1e-15)
        assert b.size[0] == pytest.approx(2e-3, abs=1e-15)

    def test_swapped_bounds_are_normalised(self):
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=3e-3, x2=0, y1=4e-3, y2=0, z1=2e-3, z2=0)
        assert b.origin == (0.0, 0.0, 0.0)
        assert b.size == pytest.approx((3e-3, 4e-3, 2e-3), abs=1e-15)

    def test_agrees_with_from_corners(self):
        from magnelio.geo.primitives import Brick

        a = Brick.from_ranges(x1=1e-3, x2=5e-3, y1=2e-3, y2=7e-3, z1=3e-3, z2=9e-3)
        b = Brick.from_corners((1e-3, 2e-3, 3e-3), (5e-3, 7e-3, 9e-3))
        assert a.origin == b.origin and a.size == b.size

    def test_material_and_name_carried(self):
        from magnelio.geo.primitives import Brick

        pec = Material.pec()
        b = Brick.from_ranges(x1=0, dx=1e-3, y1=0, dy=1e-3, z1=0, dz=1e-3, material=pec, name="pad")
        assert b.material is pec
        assert b.name == "pad"

    def test_material_is_optional(self):
        """A cutting slab spanning the domain needs no material."""
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=-1, x2=1, y1=-1, y2=1, z1=-1, z2=0)
        assert b.material is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x1": 0},  # one keyword: under-determined
            {},  # none at all
            {"x1": 0, "x2": 1, "dx": 1},  # all three, even if consistent
        ],
    )
    def test_rejects_wrong_number_of_keywords(self, kwargs):
        from magnelio.geo.primitives import Brick

        full = {"y1": 0, "y2": 1, "z1": 0, "z2": 1}
        with pytest.raises(ValueError, match=r"exactly two of x1, x2, dx"):
            Brick.from_ranges(**kwargs, **full)

    def test_error_names_the_offending_axis(self):
        from magnelio.geo.primitives import Brick

        with pytest.raises(ValueError, match=r"the z axis"):
            Brick.from_ranges(x1=0, x2=1, y1=0, y2=1, z1=0)

    def test_bounding_box_exact(self):
        _occ()
        from magnelio.geo.primitives import Brick

        b = Brick.from_ranges(x1=1e-3, dx=4e-3, y2=7e-3, dy=5e-3, z1=3e-3, z2=9e-3)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = b.bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((1e-3, 2e-3, 3e-3), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((5e-3, 7e-3, 9e-3), abs=1e-6)


# ── 1f: Group — material-preserving bundle ───────────────────────────────────


class TestGroupClass:
    """Group is a heterogeneous compound node (WP 1f)."""

    def test_members_preserve_materials(self):
        from magnelio.geo import Brick, Group

        pec, air = Material.pec(), Material.air()
        g = Group(Brick(material=pec, name="a"), Brick(material=air, name="b"))
        assert [m.material.name for m in g.members()] == ["PEC", "air"]

    def test_members_flatten_nested(self):
        from magnelio.geo import Brick, Group

        pin = Brick(material=Material.pec(), name="pin")
        diel = Brick(material=Material.air(), name="diel")
        shell = Brick(material=Material.pec(), name="shell")
        g = Group(Group(pin, diel), shell, name="sma")
        assert list(g.members()) == [pin, diel, shell]

    def test_no_single_material(self):
        """A Group deliberately exposes no .material (it is heterogeneous)."""
        from magnelio.geo import Brick, Group

        g = Group(Brick(material=Material.pec()), Brick(material=Material.air()))
        assert not hasattr(g, "material")

    def test_no_occ_shape(self):
        """A Group has no single solid — no _occ_shape method."""
        from magnelio.geo import Brick, Group

        g = Group(Brick(material=Material.pec()))
        assert not hasattr(g, "_occ_shape")

    def test_bounding_box_encloses_members(self):
        _occ()
        from magnelio.geo import Brick, Group

        b1 = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        b2 = Brick(origin=(0, 3e-3, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = Group(b1, b2).bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((0, 0, 0), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((1e-3, 4e-3, 1e-3), abs=1e-6)

    def test_empty_group_bbox_raises(self):
        from magnelio.geo import Group

        with pytest.raises(ValueError, match="empty"):
            Group().bounding_box()


class TestGroupTransforms:
    """Transforms distribute over members; group= aggregates copies."""

    def test_group_flag_aggregates_copies(self):
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import translate

        res = translate(Brick(material=_air()), (1e-3, 0, 0), repeat=3, group=True)
        assert isinstance(res, Group)
        assert len(list(res.members())) == 3

    def test_group_flag_with_copy_includes_original(self):
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import translate

        b = Brick(material=_air())
        res = translate(b, (1e-3, 0, 0), repeat=2, copy=True, group=True)
        assert isinstance(res, Group)
        members = list(res.members())
        assert len(members) == 3 and members[0] is b

    def test_unite_and_group_mutually_exclusive(self):
        from magnelio.geo import Brick
        from magnelio.geo.transforms import translate

        with pytest.raises(ValueError, match="either unite"):
            translate(Brick(material=_air()), (1e-3, 0, 0), repeat=2, unite=True, group=True)

    def test_translate_distributes_and_preserves_material(self):
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import translate

        pec, air = Material.pec(), Material.air()
        g = Group(Brick(material=pec), Brick(material=air))
        gt = translate(g, (2e-3, 0, 0))
        assert isinstance(gt, Group)
        assert [m.material.name for m in gt.members()] == ["PEC", "air"]

    def test_nested_distribution_preserves_all_materials(self):
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import rotate

        pec, air = Material.pec(), Material.air()
        g = Group(Group(Brick(material=pec), Brick(material=air)), Brick(material=pec))
        gr = rotate(g, axis=(0, 0, 1), angle_deg=30)
        assert [m.material.name for m in gr.members()] == ["PEC", "air", "PEC"]

    def test_translate_group_bbox_shifted(self):
        _occ()
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import translate

        b1 = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        b2 = Brick(origin=(0, 3e-3, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        gt = translate(Group(b1, b2), (2e-3, 0, 0))
        for m in gt.members():
            (xmin, _, _), _ = m.bounding_box()
            assert xmin == pytest.approx(2e-3, abs=1e-6)

    def test_scale_group_distributes(self):
        _occ()
        from magnelio.geo import Brick, Group
        from magnelio.geo.transforms import scale

        b1 = Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        b2 = Brick(origin=(2e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material=_air())
        gs = scale(Group(b1, b2), 2.0, center=(0, 0, 0))
        assert isinstance(gs, Group)
        (_, _, _), (xmax, _, _) = gs.bounding_box()
        # Farthest corner was at x=3mm; scaling ×2 about origin → 6mm.
        assert xmax == pytest.approx(6e-3, abs=1e-6)


class TestGroupCSGRejection:
    """Boolean ops reject a Group operand (no single material/solid)."""

    def test_union_rejects_group(self):
        from magnelio.geo import Brick, Group, Union

        g = Group(Brick(material=Material.pec()))
        with pytest.raises(TypeError, match="Group"):
            Union(g, Brick(material=_air()))

    def test_intersection_rejects_group(self):
        from magnelio.geo import Brick, Group, Intersection

        g = Group(Brick(material=Material.pec()))
        with pytest.raises(TypeError, match="Group"):
            Intersection(g, Brick(material=_air()))

    def test_difference_rejects_group_base(self):
        from magnelio.geo import Brick, Difference, Group

        g = Group(Brick(material=Material.pec()))
        with pytest.raises(TypeError, match="Group"):
            Difference(g, Brick(material=_air()))

    def test_difference_rejects_group_tool(self):
        from magnelio.geo import Brick, Difference, Group

        g = Group(Brick(material=Material.pec()))
        with pytest.raises(TypeError, match="Group"):
            Difference(Brick(material=_air()), g)


class TestGroupInModel:
    """A Group flattens into leaf members at GeometryModel.add."""

    def test_add_flattens_group(self):
        from magnelio.geo import Brick, GeometryModel, Group

        m = GeometryModel()
        m.add(Group(Brick(material=_air()), Brick(material=_air())))
        assert len(m) == 2

    def test_add_flattens_nested_group(self):
        from magnelio.geo import Brick, GeometryModel, Group

        m = GeometryModel()
        m.add(Group(Group(Brick(material=_air()), Brick(material=_air())), Brick(material=_air())))
        assert len(m) == 3

    def test_add_list_containing_group_flattens(self):
        from magnelio.geo import Brick, GeometryModel, Group

        m = GeometryModel()
        m.add([Brick(material=_air()), Group(Brick(material=_air()), Brick(material=_air()))])
        assert len(m) == 3

    def test_model_never_sees_group(self):
        from magnelio.geo import Brick, GeometryModel, Group

        m = GeometryModel()
        m.add(Group(Brick(material=_air()), Brick(material=_air())))
        assert not any(isinstance(s, Group) for s in m.shapes)

    def test_multi_material_group_meshes(self):
        """End-to-end: a two-material Group fills both materials (WP 1f target)."""
        _occ()
        import numpy as np

        from magnelio.geo import Brick, GeometryModel, Group
        from magnelio.mesh.mesher import Mesh, MeshControl

        pec = Material.pec()
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4))
        conn = Group(
            Brick(origin=(0, 0, 0), size=(2e-3, 4e-3, 4e-3), material=pec),
            Brick(origin=(2e-3, 0, 0), size=(2e-3, 4e-3, 4e-3), material=fr4),
        )
        model = GeometryModel()
        model.add(conn)
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=4, max_cell_size=1e-3),
            f_max=10e9,
        )
        lib = mesh.material_library
        pec_ids = [i for i, mm in lib.items() if mm.name == "PEC"]
        fr4_ids = [i for i, mm in lib.items() if mm.name == "FR4"]
        assert pec_ids and fr4_ids, "both Group materials must reach the library"
        assert np.any(mesh.material_id == pec_ids[0])
        assert np.any(mesh.material_id == fr4_ids[0])


# ── 1b: standalone Face ──────────────────────────────────────────────────────


class TestFaceClass:
    """Standalone planar Face with optional material (WP 1b)."""

    _RECT = [(0, 0), (4e-3, 0), (4e-3, 3e-3), (0, 3e-3)]

    def test_default_material_none_and_position_zero(self):
        from magnelio.geo import Face

        f = Face(normal="z", points=self._RECT)
        assert f.material is None
        assert f.position == 0.0

    def test_stores_material_and_name(self):
        from magnelio.geo import Face

        pec = Material.pec()
        f = Face(normal="z", points=self._RECT, material=pec, name="sheet")
        assert f.material is pec and f.name == "sheet"

    def test_bad_normal_raises(self):
        from magnelio.geo import Face

        with pytest.raises(ValueError, match="normal"):
            Face(normal="q", points=self._RECT)

    def test_too_few_points_raises(self):
        from magnelio.geo import Face

        with pytest.raises(ValueError, match="at least 3"):
            Face(normal="z", points=[(0, 0), (1e-3, 0)])

    def test_bbox_z_normal(self):
        """z-normal: (u, v) = (x, y); plane at offset in z."""
        _occ()
        from magnelio.geo import Face

        (xmin, ymin, zmin), (xmax, ymax, zmax) = Face(
            normal="z", points=self._RECT, position=2e-3
        ).bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((0, 0, 2e-3), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((4e-3, 3e-3, 2e-3), abs=1e-6)

    def test_bbox_y_normal(self):
        """y-normal: (u, v) = (x, z); plane at offset in y."""
        _occ()
        from magnelio.geo import Face

        (xmin, ymin, zmin), (xmax, ymax, zmax) = Face(
            normal="y", points=self._RECT, position=2e-3
        ).bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((0, 2e-3, 0), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((4e-3, 2e-3, 3e-3), abs=1e-6)

    def test_bbox_x_normal(self):
        """x-normal: (u, v) = (y, z); plane at offset in x."""
        _occ()
        from magnelio.geo import Face

        (xmin, ymin, zmin), (xmax, ymax, zmax) = Face(
            normal="x", points=self._RECT, position=2e-3
        ).bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((2e-3, 0, 0), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((2e-3, 4e-3, 3e-3), abs=1e-6)

    def test_occ_shape_cached(self):
        _occ()
        from magnelio.geo import Face

        f = Face(normal="z", points=self._RECT)
        assert f._occ_shape() is f._occ_shape()


# ── 1d: extrude a standalone Face ────────────────────────────────────────────


class TestExtrudeFace:
    """extrude() accepts a standalone Face as the profile (WP 1d)."""

    _RECT = [(0, 0), (4e-3, 0), (4e-3, 3e-3), (0, 3e-3)]

    def test_construction_face_requires_material(self):
        """A material-less Face needs an explicit material= to extrude."""
        from magnelio.geo import Face
        from magnelio.geo.modifications import extrude

        with pytest.raises(ValueError, match="requires an explicit material"):
            extrude(Face(normal="z", points=self._RECT), vector=(0, 0, 5e-3))

    def test_solid_still_requires_face_near(self):
        """The solid form of extrude still needs face_near."""
        from magnelio.geo import Brick
        from magnelio.geo.modifications import extrude

        b = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=_air())
        with pytest.raises(ValueError, match="face_near"):
            extrude(b, vector=(0, 0, 2e-3))

    def test_extruded_face_bbox(self):
        _occ()
        from magnelio.geo import Face
        from magnelio.geo.modifications import extrude

        solid = extrude(
            Face(normal="z", points=self._RECT, position=2e-3),
            vector=(0, 0, 5e-3),
            material=Material.pec(),
        )
        (xmin, ymin, zmin), (xmax, ymax, zmax) = solid.bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((0, 0, 2e-3), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((4e-3, 3e-3, 7e-3), abs=1e-6)

    def test_extruded_solid_material_explicit(self):
        _occ()
        from magnelio.geo import Face
        from magnelio.geo.modifications import extrude

        pec = Material.pec()
        solid = extrude(Face(normal="z", points=self._RECT), vector=(0, 0, 5e-3), material=pec)
        assert solid.material is pec

    def test_extruded_solid_inherits_face_material(self):
        _occ()
        from magnelio.geo import Face
        from magnelio.geo.modifications import extrude

        pec = Material.pec()
        solid = extrude(Face(normal="z", points=self._RECT, material=pec), vector=(0, 0, 5e-3))
        assert solid.material is pec

    def test_extrude_nonrectangular_profile(self):
        """An L-shaped profile extrudes to a valid solid (general polygon)."""
        _occ()
        from magnelio.geo import Face
        from magnelio.geo.modifications import extrude

        lprof = [(0, 0), (6e-3, 0), (6e-3, 2e-3), (2e-3, 2e-3), (2e-3, 5e-3), (0, 5e-3)]
        solid = extrude(
            Face(normal="z", points=lprof), vector=(0, 0, 3e-3), material=Material.pec()
        )
        (_, _, _), (xmax, ymax, zmax) = solid.bounding_box()
        assert (xmax, ymax, zmax) == pytest.approx((6e-3, 5e-3, 3e-3), abs=1e-6)

    def test_face_rejected_by_mesher(self):
        """A standalone Face in a model raises up front (thin-sheet deferred)."""
        _occ()
        from magnelio.geo import Face, GeometryModel
        from magnelio.mesh.mesher import Mesh, MeshControl

        m = GeometryModel()
        m.add(Face(normal="z", points=self._RECT, material=Material.pec()))
        with pytest.raises(NotImplementedError, match="thin-sheet"):
            Mesh.from_geometry(
                m, MeshControl(min_nodes_per_wavelength=4, max_cell_size=2e-3), f_max=10e9
            )

    def test_extruded_face_meshes(self):
        """End-to-end: an extruded Face profile fills its material (WP 1d target)."""
        _occ()
        import numpy as np

        from magnelio.geo import Face, GeometryModel
        from magnelio.geo.modifications import extrude
        from magnelio.mesh.mesher import Mesh, MeshControl

        pec = Material.pec()
        model = GeometryModel()
        model.add(extrude(Face(normal="z", points=self._RECT), vector=(0, 0, 6e-3), material=pec))
        mesh = Mesh.from_geometry(
            model, MeshControl(min_nodes_per_wavelength=4, max_cell_size=1e-3), f_max=10e9
        )
        pec_ids = [i for i, mm in mesh.material_library.items() if mm.name == "PEC"]
        assert pec_ids and np.any(mesh.material_id == pec_ids[0])


# ── 1c: abstract Curve ───────────────────────────────────────────────────────


class TestCurveClass:
    """One OCC-backed 3D locus with 4 constructors, no material (WP 1c)."""

    def test_no_material(self):
        """A Curve is 1D — never a physical object, so no .material."""
        from magnelio.geo import Curve

        c = Curve.polyline([(0, 0, 0), (1e-3, 0, 0)])
        assert not hasattr(c, "material")

    def test_polyline_too_few_points_raises(self):
        from magnelio.geo import Curve

        with pytest.raises(ValueError, match="at least 2"):
            Curve.polyline([(0, 0, 0)])

    def test_spline_too_few_points_raises(self):
        from magnelio.geo import Curve

        with pytest.raises(ValueError, match="at least 2"):
            Curve.spline([(0, 0, 0)])

    def test_helix_bad_axis_raises(self):
        from magnelio.geo import Curve

        with pytest.raises(ValueError, match="axis"):
            Curve.helix(radius=1e-3, pitch=1e-3, turns=3, axis="q")

    def test_helix_nonpositive_pitch_raises(self):
        from magnelio.geo import Curve

        with pytest.raises(ValueError, match="pitch"):
            Curve.helix(radius=1e-3, pitch=0.0, turns=3)

    def test_polyline_bbox(self):
        _occ()
        from magnelio.geo import Curve

        (xmin, ymin, zmin), (xmax, ymax, zmax) = Curve.polyline(
            [(0, 0, 0), (2e-3, 0, 0), (2e-3, 1e-3, 0)]
        ).bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((0, 0, 0), abs=1e-6)
        assert (xmax, ymax, zmax) == pytest.approx((2e-3, 1e-3, 0), abs=1e-6)

    def test_arc_bbox_semicircle(self):
        _occ()
        from magnelio.geo import Curve

        # Arc through (1,0)->(0,1)->(-1,0) mm: upper half of the unit circle.
        (xmin, ymin, _), (xmax, ymax, _) = Curve.arc(
            (1e-3, 0, 0), (0, 1e-3, 0), (-1e-3, 0, 0)
        ).bounding_box()
        assert (xmin, ymin) == pytest.approx((-1e-3, 0), abs=1e-6)
        assert (xmax, ymax) == pytest.approx((1e-3, 1e-3), abs=1e-6)

    def test_spline_occ_shape_runs(self):
        _occ()
        from magnelio.geo import Curve

        assert Curve.spline([(0, 0, 0), (1e-3, 2e-3, 0), (3e-3, 0, 0)])._occ_shape() is not None

    def test_helix_bbox_tight(self):
        """Helix bbox hugs the radius and spans pitch*turns in height.

        Regression for the AddOptimal bounding box: plain brepbndlib.Add
        bounds the helical B-spline by its control poles, roughly doubling
        the x/y extent.
        """
        _occ()
        from magnelio.geo import Curve

        r, pitch, turns = 2e-3, 1e-3, 3.0
        (xmin, ymin, zmin), (xmax, ymax, zmax) = Curve.helix(
            radius=r, pitch=pitch, turns=turns
        ).bounding_box()
        # Tight: x/y within ~1% of ±r (would be ~±2r with the loose bbox).
        assert xmax == pytest.approx(r, abs=0.02 * r)
        assert xmin == pytest.approx(-r, abs=0.02 * r)
        assert ymax == pytest.approx(r, abs=0.02 * r)
        assert zmin == pytest.approx(0.0, abs=1e-6)
        assert zmax == pytest.approx(pitch * turns, abs=1e-6)

    def test_helix_left_handed_runs(self):
        _occ()
        from magnelio.geo import Curve

        assert (
            Curve.helix(radius=1e-3, pitch=1e-3, turns=2, right_handed=False)._occ_shape()
            is not None
        )

    def test_occ_shape_cached(self):
        _occ()
        from magnelio.geo import Curve

        c = Curve.helix(radius=1e-3, pitch=1e-3, turns=2)
        assert c._occ_shape() is c._occ_shape()


# ── 1e: revolve a profile ────────────────────────────────────────────────────


class TestRevolve:
    """revolve() sweeps a Face profile about an axis (WP 1e)."""

    # Rect ring profile in the y=0 plane: x in [3,4] mm, z in [0,1] mm.
    def _ring_profile(self, material=None):
        from magnelio.geo import Face

        return Face(
            normal="y", points=[(3e-3, 0), (4e-3, 0), (4e-3, 1e-3), (3e-3, 1e-3)], material=material
        )

    def test_construction_face_requires_material(self):
        from magnelio.geo.modifications import revolve

        with pytest.raises(ValueError, match="requires an explicit material"):
            revolve(self._ring_profile(), axis="z")

    def test_full_revolution_pappus_volume(self):
        """360° revolution volume matches Pappus' theorem."""
        _occ()
        import math

        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        from magnelio.geo.modifications import revolve

        ring = revolve(self._ring_profile(), axis="z", angle_deg=360, material=Material.pec())
        props = GProp_GProps()
        brepgprop.VolumeProperties(ring._occ_shape(), props)
        # Pappus: V = 2*pi*R_centroid*A, R_centroid=3.5mm, A=1e-6 m^2.
        expected = 2 * math.pi * 3.5e-3 * 1e-6
        assert props.Mass() == pytest.approx(expected, rel=1e-3)

    def test_partial_revolution_scales(self):
        """90° revolution is a quarter of the full volume."""
        _occ()
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        from magnelio.geo.modifications import revolve

        pec = Material.pec()

        def vol(angle):
            s = revolve(self._ring_profile(), axis="z", angle_deg=angle, material=pec)
            p = GProp_GProps()
            brepgprop.VolumeProperties(s._occ_shape(), p)
            return p.Mass()

        assert vol(90) == pytest.approx(vol(360) / 4, rel=1e-3)

    def test_full_revolution_bbox(self):
        _occ()
        from magnelio.geo.modifications import revolve

        ring = revolve(self._ring_profile(), axis="z", angle_deg=360, material=Material.pec())
        (xmin, ymin, zmin), (xmax, ymax, zmax) = ring.bounding_box()
        assert (xmin, ymin, zmin) == pytest.approx((-4e-3, -4e-3, 0), abs=1e-5)
        assert (xmax, ymax, zmax) == pytest.approx((4e-3, 4e-3, 1e-3), abs=1e-5)

    def test_material_inherited_from_face(self):
        _occ()
        from magnelio.geo.modifications import revolve

        pec = Material.pec()
        ring = revolve(self._ring_profile(material=pec), axis="z", angle_deg=360)
        assert ring.material is pec

    def test_axis_as_vector(self):
        _occ()
        from magnelio.geo.modifications import revolve

        ring = revolve(self._ring_profile(), axis=(0, 0, 1), angle_deg=180, material=Material.pec())
        assert ring._occ_shape() is not None

    def test_revolved_solid_meshes(self):
        """End-to-end: a revolved ring fills its material."""
        _occ()
        import numpy as np

        from magnelio.geo import GeometryModel
        from magnelio.geo.modifications import revolve
        from magnelio.mesh.mesher import Mesh, MeshControl

        pec = Material.pec()
        model = GeometryModel()
        model.add(revolve(self._ring_profile(), axis="z", angle_deg=360, material=pec))
        mesh = Mesh.from_geometry(
            model, MeshControl(min_nodes_per_wavelength=4, max_cell_size=1e-3), f_max=10e9
        )
        pec_ids = [i for i, mm in mesh.material_library.items() if mm.name == "PEC"]
        assert pec_ids and np.any(mesh.material_id == pec_ids[0])


# ── 1e: sweep a profile along a spine ────────────────────────────────────────


class TestSweep:
    """sweep() sweeps a Face profile along a Curve spine (WP 1e)."""

    def _square(self, half, material=None):
        from magnelio.geo import Face

        return Face(
            normal="z",
            points=[(-half, -half), (half, -half), (half, half), (-half, half)],
            material=material,
        )

    def _vol(self, shape):
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        p = GProp_GProps()
        brepgprop.VolumeProperties(shape._occ_shape(), p)
        return p.Mass()

    def test_construction_profile_requires_material(self):
        from magnelio.geo import Curve
        from magnelio.geo.modifications import sweep

        with pytest.raises(ValueError, match="requires an explicit material"):
            sweep(self._square(0.5e-3), Curve.polyline([(0, 0, 0), (0, 0, 5e-3)]))

    def test_straight_spine_exact_volume(self):
        """A straight sweep has volume exactly profile_area × length."""
        _occ()
        from magnelio.geo import Curve
        from magnelio.geo.modifications import sweep

        s = sweep(
            self._square(0.5e-3), Curve.polyline([(0, 0, 0), (0, 0, 5e-3)]), material=Material.pec()
        )
        assert self._vol(s) == pytest.approx((1e-3) ** 2 * 5e-3, rel=1e-4)

    def test_coil_volume_and_tight_bbox(self):
        """A helix sweep ~ area×arclength, and its bbox hugs the helix radius.

        The tight bbox is the AddOptimal payoff — the swept B-spline solid
        would otherwise report ~±2× the radius.
        """
        _occ()
        import math

        from magnelio.geo import Curve
        from magnelio.geo.modifications import sweep

        r, pitch, turns, half = 2e-3, 1e-3, 3.0, 0.3e-3
        coil = sweep(
            self._square(half),
            Curve.helix(radius=r, pitch=pitch, turns=turns),
            material=Material.pec(),
        )
        arclen = turns * math.hypot(2 * math.pi * r, pitch)
        assert self._vol(coil) == pytest.approx((2 * half) ** 2 * arclen, rel=0.02)
        (xmin, _, zmin), (xmax, _, zmax) = coil.bounding_box()
        # Tight: outer edge ~ r + profile diagonal ≈ 2.42mm; a loose
        # (control-poles) bbox would be ~2r = 4mm.  1.5*r cleanly separates.
        assert xmax < 1.5 * r and xmin > -1.5 * r
        assert zmin == pytest.approx(-half, abs=1e-4)
        assert zmax == pytest.approx(pitch * turns + half, abs=1e-4)

    def test_arc_bend_volume(self):
        """Sweep along a quarter-circle arc ~ area × arclength."""
        _occ()
        import math

        from magnelio.geo import Curve
        from magnelio.geo.modifications import sweep

        R, half = 5e-3, 0.4e-3
        arc = Curve.arc((R, 0, 0), (R / math.sqrt(2), R / math.sqrt(2), 0), (0, R, 0))
        s = sweep(self._square(half), arc, material=Material.pec())
        assert self._vol(s) == pytest.approx((2 * half) ** 2 * (2 * math.pi * R / 4), rel=0.02)

    def test_material_inherited_from_face(self):
        from magnelio.geo import Curve
        from magnelio.geo.modifications import sweep

        pec = Material.pec()
        s = sweep(self._square(0.5e-3, material=pec), Curve.polyline([(0, 0, 0), (0, 0, 5e-3)]))
        assert s.material is pec

    def test_coil_meshes(self):
        """End-to-end: a swept coil fills its material on a non-inflated grid."""
        _occ()
        import numpy as np

        from magnelio.geo import Curve, GeometryModel
        from magnelio.geo.modifications import sweep
        from magnelio.mesh.mesher import Mesh, MeshControl

        pec = Material.pec()
        model = GeometryModel()
        model.add(
            sweep(
                self._square(0.3e-3), Curve.helix(radius=2e-3, pitch=1.5e-3, turns=2), material=pec
            )
        )
        mesh = Mesh.from_geometry(
            model, MeshControl(min_nodes_per_wavelength=4, max_cell_size=0.5e-3), f_max=10e9
        )
        pec_ids = [i for i, mm in mesh.material_library.items() if mm.name == "PEC"]
        assert pec_ids and np.any(mesh.material_id == pec_ids[0])


class TestBatchedFaceKernels:
    """The Numba face-accounting kernels are bit-identical to the
    Python fallback loop in compute_face_material_areas — including the
    PEC budget, the degenerate-plane (DD-087) geometric/jump route and
    the background fill.
    """

    def _run(self, prop):
        import numpy as np

        from magnelio.geo._occ_backend import compute_face_material_areas
        from magnelio.geo.primitives import Brick, Cylinder, Sphere
        from magnelio.materials.material import Material

        pec = Material.pec()
        diel = Material.from_isotropic(name="DIEL", epsilon=4.0)
        air = Material.air()
        cyl = Cylinder(
            origin=(0, 0, -10e-3),
            radius=2e-3,
            height=20e-3,
            axis="z",
            material=pec,
        )
        sph = Sphere(center=(3e-3, 0, 0), radius=2.5e-3, material=diel)
        brick = Brick(origin=(-4e-3, -4e-3, -3e-3), size=(2e-3, 2e-3, 4e-3), material=pec)
        material_library = {0: air, 1: pec, 2: diel}
        shapes_with_material = [(brick, 1), (cyl, 1), (sph, 2)]
        z_planes = np.concatenate(
            [
                np.linspace(-2.4e-3, 2.4e-3, 12),
                [1e-3, 2.5e-3],  # 1e-3 = exact brick top -> degenerate plane
            ]
        )
        face_specs = np.array([[z, -5e-3, -5e-3, 5e-3, 5e-3] for z in z_planes])
        face_axes = np.full(len(z_planes), 2, dtype=np.int32)
        pec_area = np.zeros(len(z_planes))
        pec_geom = np.zeros(len(z_planes))
        pec_jump = np.zeros(len(z_planes))
        vals = compute_face_material_areas(
            shapes_with_material,
            material_library,
            face_specs,
            face_axes,
            prop=prop,
            pec_area_out=pec_area,
            pec_area_geom_out=pec_geom,
            pec_area_jump_out=pec_jump,
        )
        return vals, pec_area, pec_geom, pec_jump

    @pytest.mark.parametrize("prop", ["epsilon", "sigma", "mu"])
    def test_kernel_bit_identical_to_fallback(self, monkeypatch, prop):
        _occ()
        import numpy as np

        import magnelio.geo._polygon_clip as pc

        if not pc.HAS_NUMBA:
            pytest.skip("Numba not available")
        kernel = self._run(prop)
        monkeypatch.setattr(pc, "HAS_NUMBA", False)
        fallback = self._run(prop)
        monkeypatch.undo()
        for a_k, a_f in zip(kernel, fallback):
            np.testing.assert_array_equal(a_k, a_f)
        vals, pec_area, pec_geom, pec_jump = kernel
        assert np.all(np.isfinite(vals))
        assert pec_area.max() > 0  # PEC budget taken
        assert np.abs(pec_jump).max() > 0  # degenerate plane jumped


# ── Curve chaining, covering and tracing ─────────────────────────────────────


def _volume(shape, scale=1.0):
    """Volume of a shape's OCC solid, in meters^3."""
    from magnelio.geo._occ_backend import occ_volume

    return abs(occ_volume(shape._occ_shape(scale)))


def _flat_box(box):
    """A bounding box as one flat tuple, for approximate comparison."""
    return (*box[0], *box[1])


def _area(shape, scale=1.0):
    """Surface area of a shape's OCC face, in meters^2."""
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape._occ_shape(scale), props)
    return abs(props.Mass())


class TestCurveJoined:
    """Curve.joined() chains segments into one profile."""

    def test_chain_is_open_and_traversable(self):
        from magnelio.geo import Curve

        _occ()
        back = Curve.polyline([(0, -5e-3, 0), (0, 5e-3, 0)])
        side = Curve.polyline([(0, 5e-3, 0), (5e-3, 5e-3, 0)])
        chain = back.joined(side)
        assert chain.is_closed is False
        assert chain._occ_shape(1.0).Closed() is False

    def test_chain_closes_on_itself(self):
        from magnelio.geo import Curve

        _occ()
        back = Curve.polyline([(0, -5e-3, 0), (0, 5e-3, 0)])
        front = Curve.arc((0, 5e-3, 0), (5e-3, 0, 0), (0, -5e-3, 0))
        assert back.joined(front).is_closed is True

    def test_seam_within_tolerance_joins(self):
        """A gap far below one part per million of the profile joins."""
        from magnelio.geo import Curve

        _occ()
        a = Curve.polyline([(0, 0, 0), (10e-3, 0, 0)])
        b = Curve.polyline([(10e-3 + 1e-12, 0, 0), (10e-3, 5e-3, 0)])
        assert a.joined(b)._occ_shape(1.0) is not None

    def test_real_gap_names_the_segment(self):
        from magnelio.geo import Curve

        _occ()
        a = Curve.polyline([(0, 0, 0), (10e-3, 0, 0)])
        b = Curve.polyline([(10e-3 + 1e-4, 0, 0), (10e-3, 5e-3, 0)])
        with pytest.raises(ValueError, match=r"segment 1 starts .* away"):
            a.joined(b)

    def test_tolerance_is_relative_to_the_profile(self):
        """The same shape at micrometre scale behaves identically."""
        from magnelio.geo import Curve

        _occ()
        for size in (10e-3, 10e-9):
            a = Curve.polyline([(0, 0, 0), (size, 0, 0)])
            good = Curve.polyline([(size * (1 + 1e-9), 0, 0), (size, size / 2, 0)])
            bad = Curve.polyline([(size * (1 + 1e-2), 0, 0), (size, size / 2, 0)])
            a.joined(good)
            with pytest.raises(ValueError, match="away from where"):
                a.joined(bad)

    def test_non_curve_argument_rejected(self):
        from magnelio.geo import Brick, Curve

        with pytest.raises(TypeError, match="takes Curve segments"):
            Curve.polyline([(0, 0, 0), (1e-3, 0, 0)]).joined(Brick())

    def test_chain_of_chains_flattens(self):
        from magnelio.geo import Curve
        from magnelio.geo._occ_backend import _wire_edges

        _occ()
        a = Curve.polyline([(0, 0, 0), (1e-3, 0, 0)])
        b = Curve.polyline([(1e-3, 0, 0), (2e-3, 0, 0)])
        c = Curve.polyline([(2e-3, 0, 0), (3e-3, 0, 0)])
        nested = a.joined(b).joined(c)
        assert len(nested._segments) == 3
        assert len(_wire_edges(nested._occ_shape(1.0))) == 3


class TestCurveCovered:
    """Curve.covered() turns a closed profile into a planar sheet."""

    def _square(self, side=4e-3):
        from magnelio.geo import Path

        return (
            Path((0, 0, 0))
            .line_to((side, 0, 0))
            .line_to((side, side, 0))
            .line_to((0, side, 0))
            .closed()
        )

    def test_open_curve_rejected_eagerly(self):
        from magnelio.geo import Curve

        with pytest.raises(ValueError, match="needs a closed curve"):
            Curve.polyline([(0, 0, 0), (1e-3, 0, 0)]).covered()

    def test_square_area(self):
        _occ()
        assert _area(self._square().covered()) == pytest.approx(16e-6, rel=1e-12)

    def test_half_disc_extrudes_to_exact_volume(self):
        from magnelio.geo import Curve

        _occ()
        r, h = 5e-3, 20e-3
        back = Curve.polyline([(0, -r, 0), (0, r, 0)])
        front = Curve.arc((0, r, 0), (r, 0, 0), (0, -r, 0))
        rod = back.joined(front).covered().extruded(vector=(0, 0, h), material=_air())
        assert _volume(rod) == pytest.approx(math.pi * r * r / 2 * h, rel=1e-12)

    def test_construction_sheet_needs_material_to_extrude(self):
        _occ()
        with pytest.raises(ValueError, match="requires an explicit material"):
            self._square().covered().extruded(vector=(0, 0, 1e-3))

    def test_non_planar_profile_rejected_by_the_kernel(self):
        from magnelio.geo import Path

        _occ()
        skew = (
            Path((0, 0, 0))
            .line_to((4e-3, 0, 0))
            .line_to((4e-3, 4e-3, 4e-3))
            .line_to((0, 4e-3, 0))
            .closed()
        )
        with pytest.raises(ValueError, match="planar"):
            skew.covered()._occ_shape(1.0)

    def test_revolved_and_swept_accept_a_covered_sheet(self):
        from magnelio.geo import Curve

        _occ()
        profile = (
            Curve.polyline([(2e-3, 0, 0), (4e-3, 0, 0), (4e-3, 0, 1e-3), (2e-3, 0, 1e-3)])
            .joined(Curve.polyline([(2e-3, 0, 1e-3), (2e-3, 0, 0)]))
            .covered()
        )
        ring = profile.revolved(axis="z", material=_air())
        assert _volume(ring) == pytest.approx(math.pi * (16e-6 - 4e-6) * 1e-3, rel=1e-6)

        spine = Curve.polyline([(0, 0, 0), (0, 0, 10e-3)])
        tube = self._square().covered().swept(spine, material=_air())
        assert _volume(tube) == pytest.approx(16e-6 * 10e-3, rel=1e-6)

    def test_analytic_box_contains_the_kernel_box(self):
        _occ()
        sheet = self._square().covered()
        lo, hi = sheet._analytic_bbox()
        occ_lo, occ_hi = sheet.bounding_box()
        assert all(lo[i] <= occ_lo[i] + 1e-12 for i in range(3))
        assert all(hi[i] >= occ_hi[i] - 1e-12 for i in range(3))

    def test_standalone_sheet_is_rejected_by_the_mesher(self):
        from magnelio.geo import GeometryModel
        from magnelio.mesh.mesher import Mesh, MeshControl

        _occ()
        model = GeometryModel()
        model.add(self._square().covered(material=_air()))
        with pytest.raises(NotImplementedError, match="planar sheet"):
            Mesh.from_geometry(model, MeshControl(), f_max=10e9)


class TestCurveTraced:
    """Curve.traced() widens a centreline into a conductor track."""

    W, T, L = 0.6e-3, 35e-6, 10e-3

    def test_flat_caps_give_the_plain_rectangle(self):
        from magnelio.geo import Curve

        _occ()
        line = Curve.polyline([(0, 0, 0), (self.L, 0, 0)])
        track = line.traced(
            width=self.W, thickness=self.T, caps="flat", normal="z", material=_air()
        )
        assert _volume(track) == pytest.approx(self.L * self.W * self.T, rel=1e-9)

    def test_round_caps_add_a_disc(self):
        from magnelio.geo import Curve

        _occ()
        line = Curve.polyline([(0, 0, 0), (self.L, 0, 0)])
        track = line.traced(width=self.W, thickness=self.T, normal="z", material=_air())
        half = 0.5 * self.W
        expected = (self.L * self.W + math.pi * half * half) * self.T
        assert _volume(track) == pytest.approx(expected, rel=1e-9)

    def test_corner_is_rounded_outside_and_mitred_inside(self):
        from magnelio.geo import Curve

        _occ()
        corner = Curve.polyline([(0, 0, 0), (self.L, 0, 0), (self.L, self.L, 0)])
        track = corner.traced(width=self.W, thickness=self.T, caps="flat", material=_air())
        half = 0.5 * self.W
        expected = (2 * self.L * self.W - half * half + math.pi * half * half / 4) * self.T
        assert _volume(track) == pytest.approx(expected, rel=1e-9)

    def test_closed_centreline_gives_a_ring(self):
        from magnelio.geo import Path

        _occ()
        loop = (
            Path((0, 0, 0))
            .line_to((self.L, 0, 0))
            .line_to((self.L, self.L, 0))
            .line_to((0, self.L, 0))
            .closed()
        )
        track = loop.traced(width=self.W, thickness=self.T, material=_air())
        half = 0.5 * self.W
        expected = (4 * self.L * self.W - 4 * half * half + math.pi * half * half) * self.T
        assert _volume(track) == pytest.approx(expected, rel=1e-9)

    def test_straight_centreline_needs_a_normal(self):
        from magnelio.geo import Curve

        _occ()
        line = Curve.polyline([(0, 0, 0), (self.L, 0, 0)])
        with pytest.raises(ValueError, match="lies in infinitely many"):
            line.traced(width=self.W, thickness=self.T, material=_air())._occ_shape(1.0)

    def test_non_planar_centreline_rejected(self):
        from magnelio.geo import Curve

        _occ()
        skew = Curve.polyline([(0, 0, 0), (self.L, 0, 0), (self.L, self.L, 0), (0, 0, self.L)])
        with pytest.raises(ValueError, match="does not lie in one plane"):
            skew.traced(width=self.W, thickness=self.T, material=_air())._occ_shape(1.0)

    def test_width_beyond_the_clearance_is_reported(self):
        from magnelio.geo import Curve

        _occ()
        hairpin = Curve.polyline([(0, 0, 0), (2e-3, 0, 0), (2e-3, 0.2e-3, 0), (0, 0.2e-3, 0)])
        with pytest.raises(ValueError, match="run into each other"):
            hairpin.traced(width=2e-3, thickness=self.T, material=_air())._occ_shape(1.0)

    def test_analytic_box_contains_the_kernel_box(self):
        from magnelio.geo import Curve

        _occ()
        corner = Curve.polyline([(0, 0, 0), (self.L, 0, 0), (self.L, self.L, 0)])
        track = corner.traced(width=self.W, thickness=self.T, material=_air())
        lo, hi = track._analytic_bbox()
        occ_lo, occ_hi = track.bounding_box()
        assert all(lo[i] <= occ_lo[i] + 1e-12 for i in range(3))
        assert all(hi[i] >= occ_hi[i] - 1e-12 for i in range(3))


class TestPath:
    """The Path builder draws the same curves, with less repetition."""

    def test_matches_the_explicit_chain(self):
        from magnelio.geo import Curve, Path
        from magnelio.geo._occ_backend import sample_wire

        _occ()
        built = (
            Path((0, 0, 0)).line_to((4e-3, 0, 0)).arc_to((4e-3, 4e-3, 0), via=(5e-3, 2e-3, 0))
        ).curve()
        explicit = Curve.polyline([(0, 0, 0), (4e-3, 0, 0)]).joined(
            Curve.arc((4e-3, 0, 0), (5e-3, 2e-3, 0), (4e-3, 4e-3, 0))
        )
        a = sample_wire(built._occ_shape(1.0), 1e-4)
        b = sample_wire(explicit._occ_shape(1.0), 1e-4)
        assert a.shape == b.shape
        assert abs(a - b).max() < 1e-12

    def test_is_immutable_so_prefixes_branch(self):
        from magnelio.geo import Path

        _occ()
        prefix = Path((0, 0, 0)).line_to((4e-3, 0, 0))
        up = prefix.line_to((4e-3, 4e-3, 0)).curve()
        down = prefix.line_to((4e-3, -4e-3, 0)).curve()
        assert prefix.current == (4e-3, 0.0, 0.0)
        assert up.bounding_box()[1][1] > 0
        assert down.bounding_box()[0][1] < 0

    def test_closed_appends_the_missing_segment(self):
        from magnelio.geo import Path

        _occ()
        path = Path((0, 0, 0)).line_to((4e-3, 0, 0)).line_to((4e-3, 4e-3, 0))
        assert path.curve().is_closed is False
        assert path.closed().is_closed is True

    def test_arc_center_takes_the_short_way_by_default(self):
        from magnelio.geo import Path

        _occ()
        minor = Path((1e-3, 0, 0)).arc_to((0, 1e-3, 0), center=(0, 0, 0)).curve()
        major = Path((1e-3, 0, 0)).arc_to((0, 1e-3, 0), center=(0, 0, 0), major=True).curve()
        assert minor.bounding_box()[0][0] == pytest.approx(0.0, abs=1e-9)
        assert major.bounding_box()[0][0] == pytest.approx(-1e-3, abs=1e-9)

    def test_arc_normal_fixes_the_turning_direction(self):
        from magnelio.geo import Path

        _occ()
        ccw = Path((1e-3, 0, 0)).arc_to((0, 1e-3, 0), center=(0, 0, 0), normal="z").curve()
        cw = Path((1e-3, 0, 0)).arc_to((0, 1e-3, 0), center=(0, 0, 0), normal=(0, 0, -1)).curve()
        assert ccw.bounding_box()[0][0] == pytest.approx(0.0, abs=1e-9)
        assert cw.bounding_box()[0][0] == pytest.approx(-1e-3, abs=1e-9)

    def test_arc_normal_draws_a_half_circle(self):
        """The case center= alone cannot resolve: diametrically opposite ends."""
        from magnelio.geo import Path

        _occ()
        cap = Path((0, 1e-3, 0)).arc_to((0, -1e-3, 0), center=(0, 0, 0), normal="z").curve()
        assert cap.bounding_box()[0][0] == pytest.approx(-1e-3, abs=1e-9)

    def test_antipodal_ends_without_normal_are_rejected(self):
        from magnelio.geo import Path

        with pytest.raises(ValueError, match="diametrically opposite"):
            Path((1e-3, 0, 0)).arc_to((-1e-3, 0, 0), center=(0, 0, 0))

    def test_centre_must_be_equidistant(self):
        from magnelio.geo import Path

        with pytest.raises(ValueError, match="equidistant"):
            Path((1e-3, 0, 0)).arc_to((0, 2e-3, 0), center=(0, 0, 0))

    def test_ends_must_lie_in_the_normal_plane(self):
        from magnelio.geo import Path

        with pytest.raises(ValueError, match="perpendicular to the normal"):
            Path((1e-3, 0, 0)).arc_to((0, 0, 1e-3), center=(0, 0, 0), normal="z")

    def test_exactly_one_of_via_or_center(self):
        from magnelio.geo import Path

        with pytest.raises(ValueError, match="exactly one of via"):
            Path((1e-3, 0, 0)).arc_to((0, 1e-3, 0))

    def test_empty_path_cannot_close(self):
        from magnelio.geo import Path

        with pytest.raises(ValueError, match="no segments yet"):
            Path((0, 0, 0)).closed()


class TestCylinderSegment:
    """Cylinder gains a bore and an angular segment."""

    R, RI, H, PHI = 12e-3, 10e-3, 30e-3, 20.0

    def test_defaults_are_the_plain_cylinder(self):
        from magnelio.geo import Cylinder

        _occ()
        plain = Cylinder(radius=self.R, height=self.H, material=_air())
        assert _volume(plain) == pytest.approx(math.pi * self.R**2 * self.H, rel=1e-12)

    def test_tube_volume(self):
        from magnelio.geo import Cylinder

        _occ()
        tube = Cylinder(radius=self.R, inner_radius=self.RI, height=self.H, material=_air())
        expected = math.pi * (self.R**2 - self.RI**2) * self.H
        assert _volume(tube) == pytest.approx(expected, rel=1e-12)

    def test_pie_segment_volume(self):
        from magnelio.geo import Cylinder

        _occ()
        pie = Cylinder(radius=self.R, height=self.H, angle_deg=self.PHI, material=_air())
        expected = math.radians(self.PHI) / 2 * self.R**2 * self.H
        assert _volume(pie) == pytest.approx(expected, rel=1e-12)

    def test_annular_segment_volume(self):
        from magnelio.geo import Cylinder

        _occ()
        seg = Cylinder(
            radius=self.R,
            inner_radius=self.RI,
            height=self.H,
            angle_deg=(0, self.PHI),
            material=_air(),
        )
        expected = math.radians(self.PHI) / 2 * (self.R**2 - self.RI**2) * self.H
        assert _volume(seg) == pytest.approx(expected, rel=1e-12)

    def test_scalar_angle_starts_at_zero(self):
        from magnelio.geo import Cylinder

        _occ()
        scalar = Cylinder(radius=self.R, height=self.H, angle_deg=self.PHI, material=_air())
        pair = Cylinder(radius=self.R, height=self.H, angle_deg=(0, self.PHI), material=_air())
        assert _flat_box(scalar.bounding_box()) == pytest.approx(_flat_box(pair.bounding_box()))

    def test_angles_turn_like_the_rotated_verb(self):
        from magnelio.geo import Cylinder

        _occ()
        turned = Cylinder(radius=self.R, height=self.H, angle_deg=(0, 20), material=_air()).rotated(
            "z", 30.0
        )
        shifted = Cylinder(radius=self.R, height=self.H, angle_deg=(30, 50), material=_air())
        assert _flat_box(turned.bounding_box()) == pytest.approx(
            _flat_box(shifted.bounding_box()), abs=1e-12
        )

    def test_segment_box_is_tight(self):
        from magnelio.geo import Cylinder

        _occ()
        seg = Cylinder(
            radius=self.R,
            inner_radius=self.RI,
            height=self.H,
            angle_deg=(0, self.PHI),
            material=_air(),
        )
        assert _flat_box(seg._analytic_bbox()) == pytest.approx(
            _flat_box(seg.bounding_box()), abs=1e-9
        )

    def test_segment_box_contains_on_every_axis(self):
        from magnelio.geo import Cylinder

        _occ()
        for axis in ("x", "y", "z", (1.0, 1.0, 0.0)):
            seg = Cylinder(
                radius=self.R, height=self.H, axis=axis, angle_deg=(0, 90), material=_air()
            )
            lo, hi = seg._analytic_bbox()
            occ_lo, occ_hi = seg.bounding_box()
            assert all(lo[i] <= occ_lo[i] + 1e-12 for i in range(3))
            assert all(hi[i] >= occ_hi[i] - 1e-12 for i in range(3))

    def test_negative_height_moves_without_turning(self):
        from magnelio.geo import Cylinder

        _occ()
        down = Cylinder(radius=self.R, height=-self.H, angle_deg=(0, self.PHI), material=_air())
        lo, hi = down.bounding_box()
        assert lo[2] == pytest.approx(-self.H)
        assert hi[2] == pytest.approx(0.0, abs=1e-12)
        expected = math.radians(self.PHI) / 2 * self.R**2 * self.H
        assert _volume(down) == pytest.approx(expected, rel=1e-12)

    def test_bore_must_leave_a_wall(self):
        from magnelio.geo import Cylinder

        with pytest.raises(ValueError, match="must be smaller than radius"):
            Cylinder(radius=1e-3, inner_radius=1e-3)

    def test_negative_bore_rejected(self):
        from magnelio.geo import Cylinder

        with pytest.raises(ValueError, match="must not be negative"):
            Cylinder(radius=1e-3, inner_radius=-1e-4)

    @pytest.mark.parametrize("span", [(0.0, 0.0), (0.0, 400.0), (10.0, 5.0)])
    def test_impossible_sweeps_rejected(self, span):
        from magnelio.geo import Cylinder

        with pytest.raises(ValueError, match="more than 0 and at most"):
            Cylinder(radius=1e-3, angle_deg=span)


class TestSegmentCriticalPlanes:
    """A segment must not contribute the tangent planes it does not reach."""

    def test_quarter_segment_drops_the_far_tangent(self):
        from magnelio.geo import Cylinder
        from magnelio.geo._occ_backend import extract_critical_planes

        _occ()
        # 90..180 deg about z spans -x and +y only.
        seg = Cylinder(radius=10e-3, height=5e-3, angle_deg=(90, 180), material=_air())
        planes = extract_critical_planes([seg])
        assert any(abs(p - (-10e-3)) < 1e-9 for p in planes["x"])
        assert not any(abs(p - 10e-3) < 1e-9 for p in planes["x"])

    def test_tube_contributes_both_radii(self):
        from magnelio.geo import Cylinder
        from magnelio.geo._occ_backend import extract_critical_planes

        _occ()
        tube = Cylinder(radius=10e-3, inner_radius=6e-3, height=5e-3, material=_air())
        planes = extract_critical_planes([tube])
        for expected in (-10e-3, 10e-3, -6e-3, 6e-3):
            assert any(abs(p - expected) < 1e-9 for p in planes["x"])


# ── Public volume query ──────────────────────────────────────────────────────


class TestShapeVolume:
    """Shape.volume() reports what the kernel actually built."""

    def test_brick(self):
        from magnelio.geo import Brick

        _occ()
        brick = Brick(size=(2e-3, 3e-3, 4e-3), material=_air())
        assert brick.volume() == pytest.approx(24e-9, rel=1e-12)

    def test_boolean_difference_reports_what_is_left(self):
        from magnelio.geo import Brick, Cylinder

        _occ()
        plate = Brick(size=(10e-3, 10e-3, 1e-3), material=_air())
        bore = Cylinder(origin=(5e-3, 5e-3, -1.0), radius=2e-3, height=3.0)
        expected = (100e-6 - math.pi * 4e-6) * 1e-3
        assert (plate - bore).volume() == pytest.approx(expected, rel=1e-9)

    def test_annular_segment(self):
        from magnelio.geo import Cylinder

        _occ()
        seg = Cylinder(
            radius=12e-3, inner_radius=10e-3, height=30e-3, angle_deg=(0, 20), material=_air()
        )
        expected = math.radians(20.0) / 2 * (144e-6 - 100e-6) * 30e-3
        assert seg.volume() == pytest.approx(expected, rel=1e-12)

    def test_planar_sheet_has_none(self):
        from magnelio.geo import Face

        _occ()
        sheet = Face(normal="z", points=[(0, 0), (1e-3, 0), (1e-3, 1e-3)])
        assert sheet.volume() == 0.0

    def test_group_adds_its_members(self):
        from magnelio.geo import Brick, Group

        _occ()
        bundle = Group(
            Brick(size=(1e-3, 1e-3, 1e-3), material=_air()),
            Brick(origin=(5e-3, 0, 0), size=(2e-3, 1e-3, 1e-3), material=_air()),
        )
        assert bundle.volume() == pytest.approx(3e-9, rel=1e-12)

    @pytest.mark.parametrize("size", [1e-9, 1e-3, 1e3])
    def test_survives_any_model_scale(self, size):
        """The automatic scale must cancel exactly out of the result."""
        from magnelio.geo import Brick

        _occ()
        brick = Brick(size=(2 * size, 3 * size, 4 * size), material=_air())
        assert brick.volume() == pytest.approx(24 * size**3, rel=1e-12)


# ── Cross-sections that land on a face ───────────────────────────────────────


class TestSectionAtFace:
    """A plane lying in a face is degenerate for the section operator."""

    T = 35e-6

    def _strip(self):
        """Two bricks fused end to end — the seam is what goes wrong."""
        from magnelio.geo import Brick

        return Brick(size=(10e-3, 2e-3, self.T), material=_air()) + Brick(
            origin=(10e-3, 0, 0), size=(10e-3, 2e-3, self.T), material=_air()
        )

    @staticmethod
    def _area(polygons):
        """Total enclosed area by the even-odd rule (holes subtract)."""
        signed = []
        for poly in polygons:
            u, v = poly[:, 0], poly[:, 1]
            n = len(u)
            signed.append(
                0.5 * sum(u[i] * v[(i + 1) % n] - u[(i + 1) % n] * v[i] for i in range(n))
            )
        largest = max(abs(a) for a in signed)
        return sum(abs(a) if abs(a) == largest else -abs(a) for a in signed)

    def test_plain_section_loses_material_at_a_seam(self):
        """The behaviour that motivates the option — the seam eats area.

        *How much* it eats is not a contract.  A plane lying in a face
        makes the plain section ill-posed; historically what OCC
        returned there moved with kernel settings (half the strip, a
        quarter after DD-146 — implicitly closed partial chains,
        neither the right answer).  Since the open-chain guard, such
        chains are dropped with a warning instead of being implicitly
        closed, so the plain path now comes back short *loudly* — and
        that shortfall is why ``exact_at_faces`` exists.  No
        production caller takes this path on such a plane: plotting
        opts in (DD-137) and the mesher re-takes degenerate planes a
        step to either side.
        """
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        with pytest.warns(UserWarning, match="open section chain"):
            polys = cross_section_polygons(self._strip()._occ_shape(1.0), "z", 0.0, 1e-5)
        assert (self._area(polys) if polys else 0.0) < 40e-6

    def test_exact_at_faces_recovers_it(self):
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        polys = cross_section_polygons(
            self._strip()._occ_shape(1.0), "z", 0.0, 1e-5, exact_at_faces=True
        )
        assert self._area(polys) == pytest.approx(40e-6, rel=1e-9)

    def test_the_escape_step_is_not_the_chordal_budget(self):
        """The retry needs a length of its own (DD-167).

        Same seam plane, same tessellation: the escape has to land
        inside the 35 µm strip to find a clean section.  Tied to the
        chordal budget it steps 40 µm and lands in thin air, so the
        retry is rejected and the material is dropped; given its own,
        smaller step it lands at 16 µm and recovers the strip whole.
        Neither value is more 'accurate' than the other — they answer
        different questions, which is the whole point.
        """
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        occ_shape = self._strip()._occ_shape(1.0)
        with pytest.warns(UserWarning, match="open section chain"):
            cross_section_polygons(occ_shape, "z", 0.0, 1e-5, nudge=1e-5)
        polys = cross_section_polygons(occ_shape, "z", 0.0, 1e-5, nudge=4e-6)
        assert self._area(polys) == pytest.approx(40e-6, rel=1e-9)

    @pytest.mark.parametrize("position", [0.0, T])
    def test_both_faces_of_the_solid(self, position):
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        polys = cross_section_polygons(
            self._strip()._occ_shape(1.0), "z", position, 1e-5, exact_at_faces=True
        )
        assert self._area(polys) == pytest.approx(40e-6, rel=1e-9)

    def test_interior_planes_are_untouched_by_the_option(self):
        """Away from a face the option must change nothing at all."""
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        occ_shape = self._strip()._occ_shape(1.0)
        plain = cross_section_polygons(occ_shape, "z", self.T / 2, 1e-5)
        exact = cross_section_polygons(occ_shape, "z", self.T / 2, 1e-5, exact_at_faces=True)
        assert len(plain) == len(exact)
        for a, b in zip(plain, exact):
            assert a.shape == b.shape
            assert abs(a - b).max() == 0.0

    def test_holes_survive_a_face_cut(self):
        from magnelio.geo import Brick, Cylinder
        from magnelio.geo._occ_backend import cross_section_polygons

        _occ()
        plate = Brick(size=(10e-3, 10e-3, self.T), material=_air()) - Cylinder(
            origin=(5e-3, 5e-3, -1.0), radius=2e-3, height=3.0
        )
        polys = cross_section_polygons(plate._occ_shape(1.0), "z", 0.0, 1e-5, exact_at_faces=True)
        assert len(polys) == 2
        assert self._area(polys) == pytest.approx(100e-6 - math.pi * 4e-6, rel=1e-3)

    def test_the_geometry_plot_uses_it(self):
        """The whole point: the plot must show the whole strip."""
        import matplotlib

        matplotlib.use("Agg")
        from magnelio import plots

        _occ()
        _fig, ax = plots.plot_cross_section([self._strip()], "z", 0.0)
        spans = [patch.get_xy()[:, 0] for patch in ax.patches]
        assert min(s.min() for s in spans) == pytest.approx(0.0, abs=1e-9)
        assert max(s.max() for s in spans) == pytest.approx(20.0, abs=1e-6)


class TestMesherEscapeReach:
    """Both mesh passes must step off a degenerate plane equally far."""

    def test_classification_and_conformal_areas_share_the_escape(self, monkeypatch):
        """The regression this guards is a silent disagreement (DD-167).

        The two passes tessellate at deliberately different chordal
        budgets, and while the escape hung off that budget the finer
        pass could not leave bands the coarser one cleared — cells
        classified as conductor whose material matrices saw nothing
        there.  The escape is one number now, and this asserts it.
        """
        from magnelio.geo import Cylinder, GeometryModel
        from magnelio.geo import _occ_backend as ob
        from magnelio.geo._filling import SECTION_NUDGE_FRACTION
        from magnelio.mesh.mesher import Mesh, MeshControl

        _occ()
        seen: list[float | None] = []
        original = ob.cross_section_polygons

        def recording(*args, **kwargs):
            seen.append(kwargs.get("nudge"))
            return original(*args, **kwargs)

        monkeypatch.setattr(ob, "cross_section_polygons", recording)

        model = GeometryModel()
        model.add(Cylinder(radius=2e-3, height=4e-3, axis="z", material=_air()))
        mesh = Mesh.from_geometry(
            model, MeshControl(min_nodes_per_wavelength=4, max_cell_size=1e-3), f_max=10e9
        )

        h_min = min(mesh.grid.dx.min(), mesh.grid.dy.min(), mesh.grid.dz.min())
        assert seen, "a curved solid must reach the section operator"
        assert all(n == pytest.approx(h_min * SECTION_NUDGE_FRACTION) for n in seen)
        # The far end of the ladder must stay inside one cell, or the
        # section answers about a plane nobody asked about.
        assert 8.0 * h_min * SECTION_NUDGE_FRACTION < h_min
