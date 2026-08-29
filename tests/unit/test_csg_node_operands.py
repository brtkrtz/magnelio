"""A Difference served from its operands: box, planes, sections, PEC solid.

With every tool inside a box-shaped base, ``base − tools`` is the box
with the tools' union removed inside it; the mesher reads its box, its
face and edge planes, the convexity of its edges and its section
contours off the operands and never builds the kernel cut.  Every
answer must equal the kernel route's (``MAGNELIO_CSG_NODES=0``) up to
the rounding of a different polygon decomposition.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.geo import Brick, Cylinder, Difference, GeometryModel, Union
from magnelio.geo import _occ_backend as ob
from magnelio.geo._polygon_clip import points_in_polygon, polygon_area
from magnelio.materials import Material
from magnelio.mesh.mesher import Mesh, MeshControl

pytest.importorskip("OCC")

MM = 1e-3
AIR = Material.air()
PEC = Material.pec()


def _tools(material=PEC):
    """Pieces inside a 10 × 10 × 5 mm box: a strip on the floor, a floating
    block, a post standing on the floor and a pad under the ceiling."""
    return [
        Brick(origin=(2 * MM, 4 * MM, 0.0), size=(6 * MM, 2 * MM, 0.5 * MM), material=material),
        Brick(origin=(3 * MM, 1 * MM, 2 * MM), size=(1 * MM, 1 * MM, 1 * MM), material=material),
        Cylinder(
            origin=(7 * MM, 7.5 * MM, 0.0),
            radius=0.8 * MM,
            height=2 * MM,
            axis="z",
            material=material,
        ),
        Brick(origin=(1 * MM, 8 * MM, 4 * MM), size=(2 * MM, 1 * MM, 1 * MM), material=material),
    ]


def _housing(tools=None, material=AIR):
    box = Brick(origin=(0, 0, 0), size=(10 * MM, 10 * MM, 5 * MM), material=material)
    return Difference(box, *(tools if tools is not None else _tools()))


def _forbid_cut(monkeypatch):
    def forbidden(self, scale=1.0):
        raise AssertionError("the kernel cut of a Difference was built")

    monkeypatch.setattr(Difference, "_occ_shape", forbidden)


def _planes_set(planes):
    return {axis: sorted(set(planes[axis])) for axis in "xyz"}


class TestOperandRoute:
    def test_tools_inside_a_box_base(self):
        assert ob._operand_route(_housing(), 1.0)

    def test_a_tool_sticking_out_keeps_the_kernel(self):
        tool = Brick(origin=(8 * MM, 4 * MM, MM), size=(4 * MM, MM, MM), material=PEC)
        assert not ob._operand_route(_housing([tool]), 1.0)

    def test_a_base_that_is_not_its_box_keeps_the_kernel(self):
        base = Cylinder(origin=(0, 0, 0), radius=5 * MM, height=5 * MM, axis="z", material=AIR)
        node = Difference(base, Brick(origin=(-MM, -MM, MM), size=(2 * MM, 2 * MM, MM)))
        assert not ob._operand_route(node, 1.0)

    def test_differences_only(self):
        assert not ob._operand_route(Union(*_tools()[:2]), 1.0)
        assert not ob._operand_route(_tools()[0], 1.0)

    def test_the_switch_restores_the_kernel_route(self, monkeypatch):
        monkeypatch.setenv("MAGNELIO_CSG_NODES", "0")
        assert not ob._operand_route(_housing(), 1.0)
        assert isinstance(ob._section_engine(_housing(), 1.0, 1e-5), ob._PlanarSectionEngine)

    def test_box_and_planes_are_the_cuts(self):
        node = _housing()
        cut = node._occ_shape(1.0)
        assert ob._screen_bbox(node, 1.0) == ob.bounding_box(cut, 1.0)
        assert _planes_set(ob._node_face_critical_planes(node, 1.0)) == _planes_set(
            ob._face_critical_planes(cut)
        )
        assert _planes_set(ob._node_edge_feature_planes(node, 1.0)) == _planes_set(
            ob._edge_feature_planes(cut)
        )


class TestSingularEdges:
    @staticmethod
    def _planes(shapes, background):
        found = ob.extract_singular_edge_planes(shapes, background, scale=1.0)
        return {axis: sorted({round(p / MM, 9) for p in found[axis]}) for axis in "xyz"}

    @staticmethod
    def _both_routes(monkeypatch, build, background):
        monkeypatch.setenv("MAGNELIO_CSG_NODES", "0")
        kernel = TestSingularEdges._planes([build()], background)
        monkeypatch.setenv("MAGNELIO_CSG_NODES", "1")
        node = build()
        assert ob._operand_route(node, 1.0)
        operands = TestSingularEdges._planes([node], background)
        return kernel, operands

    def test_void_pockets_in_a_pec_background(self, monkeypatch):
        # The pockets are construction solids: their walls are metal
        # (the background), so every concave edge of the air body — a
        # pocket rim inside the box — is singular; the pocket edges on
        # the floor and the ceiling are convex wedges of air and are not.
        kernel, operands = self._both_routes(monkeypatch, lambda: _housing(_tools(None)), PEC)
        assert operands == kernel
        assert 0.5 in kernel["z"] and 4.0 in kernel["z"]
        assert 0.0 not in kernel["z"] and 5.0 not in kernel["z"]

    def test_metal_block_with_pockets(self, monkeypatch):
        # A PEC block with dielectric pockets: its convex edges are its
        # own plus those of the pockets open to its faces.
        kernel, operands = self._both_routes(
            monkeypatch, lambda: _housing(_tools(None), material=PEC), AIR
        )
        assert operands == kernel
        assert kernel["z"] == [0.0, 5.0]
        assert 2.0 in kernel["x"] and 8.0 in kernel["x"]


class TestSections:
    def test_engine_and_delegation_never_build_the_cut(self, monkeypatch):
        node = _housing()
        _forbid_cut(monkeypatch)
        engine = ob._section_engine(node, 1.0, 1e-5)
        assert isinstance(engine, ob._CsgSectionEngine) and engine.enabled
        # A plane through the post's seam vertices: the tools' engine
        # declines, the base's answers — the tools take the kernel
        # section, the node's solid stays unbuilt.
        y = 7.5 * MM
        assert engine._base.can_fast(1, y) and not engine.can_fast(1, y)
        assert engine.section(1, y) is None
        polys = ob._delegated_section(
            node, engine, "y", y, deflection=1e-5, scale=1.0, nudge=None, context=""
        )
        assert polys

        def inside(u, v):
            hit = False
            for poly in polys:
                hit ^= bool(points_in_polygon(np.array([u]), np.array([v]), poly)[0])
            return hit

        assert inside(1 * MM, 4 * MM)  # air
        assert not inside(7 * MM, 1 * MM)  # in the post
        assert inside(7 * MM, 3 * MM)  # above the post
        assert not inside(12 * MM, 3 * MM)  # outside the box

    def test_sections_are_the_base_and_the_reversed_tools(self):
        node = _housing()
        engine = ob._section_engine(node, 1.0, 1e-5)
        z = 0.25 * MM  # through the strip and the post
        polys = engine.section(2, z)
        assert polys is not None and len(polys) == 3
        areas = sorted(polygon_area(p) for p in polys)
        assert areas[-1] == pytest.approx(100 * MM * MM)
        assert areas[0] < 0 and areas[1] < 0
        annotated = engine.annotated_sections(2, [z])[0]
        assert sorted(a for _, _, a in annotated) == pytest.approx(areas)


class TestMeshEquivalence:
    @staticmethod
    def _mesh(monkeypatch, flag):
        monkeypatch.setenv("MAGNELIO_CSG_NODES", flag)
        model = GeometryModel(background="pec")
        node = _housing()
        model.add(node)
        for tool in node.tools:
            model.add(tool)
        control = MeshControl(min_nodes_per_wavelength=10, min_cell_size=0.25 * MM)
        return Mesh.from_geometry(model, control, f_max=20e9)

    def test_arrays_match_the_kernel_route(self, monkeypatch):
        kernel = self._mesh(monkeypatch, "0")
        operands = self._mesh(monkeypatch, "1")
        np.testing.assert_array_equal(kernel.material_id, operands.material_id)
        for name in ("x", "y", "z"):
            np.testing.assert_array_equal(getattr(kernel.grid, name), getattr(operands.grid, name))
        # Areas differ by the rounding of a different polygon
        # decomposition (the box's rectangle minus the tools' contours
        # against the cut's notched contour): 1e-13 of a cell face.
        fk, fo = kernel.face_material, operands.face_material
        for name in ("A_face_pec", "A_face_free", "A_face_pec_jump", "mu_avg"):
            a, b = np.nan_to_num(getattr(fk, name)), np.nan_to_num(getattr(fo, name))
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-13 * np.abs(a).max())
        ek, eo = kernel.edge_material, operands.edge_material
        for name in ("eps_avg", "f_A"):
            a, b = np.nan_to_num(getattr(ek, name)), np.nan_to_num(getattr(eo, name))
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_the_build_never_touches_the_cut(self, monkeypatch):
        _forbid_cut(monkeypatch)
        mesh = self._mesh(monkeypatch, "1")
        assert mesh.material_id.max() > 0


class TestEffectivePecSolid:
    @staticmethod
    def _shapes():
        housing = Brick(origin=(-MM, -MM, -MM), size=(12 * MM, 12 * MM, 7 * MM), material=PEC)
        node = _housing()
        return [(housing, 1), (node, 0), *((tool, 1) for tool in node.tools)]

    def test_volume_matches_without_the_cut(self, monkeypatch):
        library = {0: AIR, 1: PEC}
        monkeypatch.setenv("MAGNELIO_CSG_NODES", "0")
        reference = ob.occ_volume(ob.build_effective_pec_solid(self._shapes(), library, scale=1.0))
        monkeypatch.setenv("MAGNELIO_CSG_NODES", "1")
        _forbid_cut(monkeypatch)
        solid = ob.build_effective_pec_solid(self._shapes(), library, scale=1.0)
        assert ob.occ_volume(solid) == pytest.approx(reference, rel=1e-9)
