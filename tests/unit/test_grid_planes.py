"""Grid-plane provenance (DD-200): attribution, records, report, carriage."""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.mesh._planes import (
    GridPlanes,
    PlaneRecord,
    PlaneSource,
    attribute_planes,
    shape_label,
)

GAP = 1e-6


def _src(kind, shape=None, label=""):
    return PlaneSource(kind, shape, label)


class TestAttribute:
    """attribute_planes matches raw entries to the merged outcome by position."""

    def _run(self, finals, sources, *, dropped=(), absorbed=(), singular=None):
        return attribute_planes(
            {"x": list(finals), "y": [], "z": []},
            {"x": list(singular or [False] * len(finals)), "y": [], "z": []},
            {"x": list(sources), "y": [], "z": []},
            {"x": list(dropped), "y": [], "z": []},
            {"x": list(absorbed), "y": [], "z": []},
            GAP,
        )

    def test_snapped_cluster_lands_on_the_midpoint_plane(self):
        a, b = _src("face", 0, "#0 A"), _src("extent", 1, "#1 B")
        grid, _d, _a, unplaced = self._run(
            [0.0, 5e-3], [(0.0, a), (5e-3 - 0.4 * GAP, a), (5e-3 + 0.4 * GAP, b)]
        )
        assert [r.position for r in grid["x"]] == [0.0, 5e-3]
        assert set(grid["x"][1].sources) == {a, b}
        assert grid["x"][0].domain_end and grid["x"][1].domain_end
        assert unplaced["x"] == []

    def test_window_is_twice_the_feature_gap(self):
        a = _src("face", 0, "#0 A")
        grid, _d, _a, unplaced = self._run([0.0, 5e-3], [(1.9 * GAP, a), (5e-3 + 2.1 * GAP, a)])
        assert grid["x"][0].sources == (a,)
        assert grid["x"][1].sources == ()
        assert len(unplaced["x"]) == 1
        assert unplaced["x"][0].position == pytest.approx(5e-3 + 2.1 * GAP)
        assert unplaced["x"][0].sources == (a,)

    def test_edge_entry_matches_a_dropped_plane(self):
        e = _src("edge", 2, "#2 C")
        grid, dropped, _a, unplaced = self._run(
            [0.0, 5e-3], [(2e-3, e)], dropped=[(2e-3 + 0.5 * GAP, 40e-6)]
        )
        assert dropped["x"][0].sources == (e,)
        assert dropped["x"][0].gap == pytest.approx(40e-6)
        assert unplaced["x"] == []
        assert all(r.sources == () for r in grid["x"])

    def test_material_entry_matches_an_absorbed_plane_but_edge_does_not(self):
        f, e = _src("face", 0, "#0 A"), _src("edge", 0, "#0 A")
        _g, _d, absorbed, unplaced = self._run([0.0, 5e-3], [(2e-3, f), (2e-3, e)], absorbed=[2e-3])
        assert absorbed["x"][0].sources == (f,)
        assert len(unplaced["x"]) == 1 and unplaced["x"][0].sources == (e,)

    def test_nearest_candidate_wins(self):
        a = _src("face", 0, "#0 A")
        grid, _d, _a, _u = self._run([0.0, 3 * GAP, 5e-3], [(1.6 * GAP, a)])
        assert grid["x"][1].sources == (a,)
        assert grid["x"][0].sources == ()

    def test_singular_flag_is_carried(self):
        grid, _d, _a, _u = self._run([0.0, 1e-3, 5e-3], [], singular=[False, True, False])
        assert [r.singular for r in grid["x"]] == [False, True, False]

    def test_unplaced_entries_within_the_window_form_one_record(self):
        a, b = _src("face", 0, "#0 A"), _src("face", 1, "#1 B")
        _g, _d, _a, unplaced = self._run([0.0, 5e-3], [(2e-3, a), (2e-3 + GAP, b)])
        assert len(unplaced["x"]) == 1
        assert unplaced["x"][0].sources == (a, b)


class TestSourceStrings:
    def test_str_parse_round_trip(self):
        for src in (
            _src("face", 3, "#3 Cylinder(pec)"),
            _src("extent", 12, "#12 ImportedSolid(ptfe) 'window'"),
            _src("edge", 0, "#0 Difference(air)"),
            _src("sheet", 1, "#1 Brick(pec)"),
            _src("wire", 4, "#4 ThinWire(pec)"),
            _src("symmetry", None, "xmin"),
            _src("forced"),
        ):
            assert PlaneSource.parse(str(src)) == src

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown plane source kind"):
            PlaneSource("port")

    def test_shape_label(self):
        class Mat:
            name = "pec"

        class Brick:
            material = Mat()
            name = None

        class Named(Brick):
            name = "lid"

        assert shape_label(2, Brick()) == "#2 Brick(pec)"
        assert shape_label(3, Named()) == "#3 Named(pec) 'lid'"

    def _planes(self):
        a, b = _src("face", 0, "#0 A"), _src("forced")
        rec = [
            PlaneRecord(0.0, (a, b), node=0, domain_end=True, h_fine=1e-4),
            PlaneRecord(1e-3, (a,), node=7, singular=True, h_fine=2.5e-5),
            PlaneRecord(5e-3, (a,), node=12, domain_end=True, h_fine=1e-4, moved_to=4.99e-3),
        ]
        return GridPlanes(
            x=tuple(rec),
            y=(PlaneRecord(0.0, (a,), node=0, domain_end=True),),
            z=(),
            h_bulk={"x": (5e-4, 5e-4), "y": (), "z": ()},
            dropped={"x": (PlaneRecord(2e-3, (_src("edge", 1, "#1 B"),), gap=4e-5),)},
            absorbed={"x": (PlaneRecord(3e-3, (_src("face", 1, "#1 B"),)),)},
            unplaced={"x": (PlaneRecord(4e-3, (_src("extent", 1, "#1 B"),)),)},
            n_nodes={"x": 13, "y": 1, "z": 0},
            pml_cells={"zmin": 8},
            feature_gap=1.234567e-7,
        )

    def test_dict_round_trip_is_exact_when_unrounded(self):
        gp = self._planes()
        d = gp.as_dict(rounded=False)
        assert GridPlanes.from_dict(d) == gp

    def test_dict_is_deterministic_and_rounded(self):
        gp = self._planes()
        d = gp.as_dict()
        assert d["schema"] == 1
        assert list(d["axes"]) == ["x", "y", "z"]
        assert d["axes"]["x"]["planes"][0]["sources"] == ["face #0 A", "forced"]
        assert d["axes"]["x"]["planes"][2]["moved_to"] == pytest.approx(4.99e-3)
        assert d["axes"]["x"]["dropped"][0]["gap"] == 4e-5
        assert d["n_nodes"] == {"x": 13, "y": 1, "z": 0}
        assert d["pml_cells"] == {"zmin": 8}
        # positions rounded three decimals below the feature gap (0.12 µm -> 1e-10 m)
        gp2 = GridPlanes(x=(PlaneRecord(1.00000000049e-3, ()),), feature_gap=1.234567e-7)
        assert gp2.as_dict()["axes"]["x"]["planes"][0]["position"] == 1.0e-3

    def test_summary_lists_planes_and_leftovers(self):
        text = str(self._planes())
        head, *rows = text.splitlines()
        assert head.startswith(
            "Grid planes — feature gap 0.123 µm; nodes x 13, y 1, z 0; PML zmin 8"
        )
        assert rows[0].startswith("x: 3 planes")
        assert "[0]" in rows[1] and "domain end" in rows[1] and "face #0 A; forced" in rows[1]
        assert "singular" in rows[2] and "h_fine 0.025 mm" in rows[2]
        assert "-> node at 4.99 mm" in rows[3]
        assert any(
            r.startswith("  dropped (edge floor): 2 mm") and "0.04 mm cell" in r for r in rows
        )
        assert any(r.startswith("  absorbed (min_cell_size): 3 mm") for r in rows)
        assert any(r.startswith("  unplaced: 4 mm") for r in rows)
        assert (
            " m " in self._planes().summary(scale_mm=False).splitlines()[1]
            or "m" in (self._planes().summary(scale_mm=False).splitlines()[2])
        )


# ---------------------------------------------------------------------------
# End to end through Mesh.from_geometry (OCC)
# ---------------------------------------------------------------------------


def _kinds_at(planes, axis, position, tol):
    return {
        s.kind for r in planes.records(axis) if abs(r.position - position) <= tol for s in r.sources
    }


class TestFromGeometry:
    @pytest.fixture(autouse=True)
    def _occ(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")

    @staticmethod
    def _mesh(model, **control):
        from magnelio.mesh.mesher import Mesh, MeshControl

        control.setdefault("min_nodes_per_wavelength", 6)
        return Mesh.from_geometry(model, MeshControl(**control), f_max=10e9)

    @staticmethod
    def _brick_model(bc=None, **kw):
        from magnelio.geo import Brick, GeometryModel

        m = GeometryModel(boundary_conditions=bc) if bc else GeometryModel()
        m.add(Brick(origin=(-5e-3, -4e-3, 0.0), size=(10e-3, 8e-3, 6e-3), material="air", **kw))
        return m

    def test_brick_faces_are_face_and_extent_planes(self):
        mesh = self._mesh(self._brick_model())
        gp = mesh.planes
        assert gp is not None
        for axis, (lo, hi) in (("x", (-5e-3, 5e-3)), ("y", (-4e-3, 4e-3)), ("z", (0.0, 6e-3))):
            recs = gp.records(axis)
            assert [r.position for r in recs] == pytest.approx([lo, hi])
            # a brick's straight edges lie in its faces: the edge pass
            # duplicates the face planes and is absorbed silently
            assert all(r.kinds >= {"face", "extent"} for r in recs)
            assert all(s.label == "#0 Brick(air)" for r in recs for s in r.sources)
            assert recs[0].domain_end and recs[1].domain_end
            assert recs[0].node == 0 and recs[1].node == len(getattr(mesh.grid, axis)) - 1
            assert gp.n_nodes[axis] == len(getattr(mesh.grid, axis))
            assert len(gp.h_bulk[axis]) == 1
            assert gp.unplaced[axis] == () and gp.dropped[axis] == () and gp.absorbed[axis] == ()

    def test_forced_plane_on_a_face_keeps_both_sources(self):
        mesh = self._mesh(self._brick_model(), forced_planes={"z": [0.0, 3e-3]})
        z = mesh.planes.records("z")
        assert [r.position for r in z] == pytest.approx([0.0, 3e-3, 6e-3])
        assert z[0].kinds >= {"face", "extent", "forced"}
        assert z[1].kinds == {"forced"} and not z[1].domain_end
        assert mesh.grid.z[z[1].node] == pytest.approx(3e-3)

    def test_inscribed_cylinder_puts_no_edge_plane_through_its_axis(self):
        """KB-028: the Boolean's split lines along the touching lines are not geometry."""
        from magnelio.geo import Brick, Cylinder, Difference, GeometryModel

        R, H = 15e-3, 20e-3
        block = Brick(origin=(-R, -R, 0), size=(2 * R, 2 * R, H), material="pec")
        hole = Cylinder(origin=(0, 0, 0), radius=R, height=H, axis="z", material="air")
        model = GeometryModel()
        model.add(Difference(block, hole))
        model.add(hole)
        mesh = self._mesh(model, max_cell_size=5e-3)
        gp = mesh.planes
        for axis in "xy":
            assert [r.position for r in gp.records(axis)] == pytest.approx([-R, R])
            assert gp.unplaced[axis] == ()
        assert _kinds_at(gp, "z", 0.0, gp.feature_gap) >= {"face", "edge"}

    def test_dropped_edge_plane_is_recorded_with_its_gap(self):
        from magnelio.geo import Cylinder, GeometryModel

        R, H, c = 4e-3, 6e-3, 0.05e-3
        puck = Cylinder(origin=(0, 0, 0), radius=R, height=H, axis="z", material="air").chamfered(
            edges="all", distance=c
        )
        model = GeometryModel()
        model.add(puck)
        with pytest.warns(UserWarning, match="edge plane"):
            mesh = self._mesh(model, min_nodes_per_wavelength=4)
        gp = mesh.planes
        dropped = gp.dropped["z"]
        assert {round(r.position, 9) for r in dropped} == {round(c, 9), round(H - c, 9)}
        assert all(r.kinds == {"edge"} and r.gap == pytest.approx(c) for r in dropped)
        assert gp.unplaced["z"] == ()

    def test_absorbed_material_plane_is_recorded(self):
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material

        h1, h2, h3, w = 0.5e-3, 60e-6, 1.0e-3, 2.0e-3
        m = GeometryModel()
        m.add(
            Brick(
                origin=(0, 0, 0),
                size=(1e-3, w, h1),
                material=Material(name="d1", epsilon=(4.3,) * 3),
            )
        )
        m.add(
            Brick(
                origin=(0, 0, h1),
                size=(1e-3, w, h2),
                material=Material(name="d2", epsilon=(8.0,) * 3),
            )
        )
        m.add(Brick(origin=(0, 0, h1 + h2), size=(1e-3, w, h3), material=Material.air()))
        mesh = self._mesh(m, min_cell_size=100e-6, max_edge_refinement=0)
        gp = mesh.planes
        assert len(gp.absorbed["z"]) == 1
        absorbed = gp.absorbed["z"][0]
        assert absorbed.kinds == {"face", "extent"}
        assert {s.shape for s in absorbed.sources} == {1, 2} or {
            s.shape for s in absorbed.sources
        } == {
            0,
            1,
        }
        assert gp.unplaced["z"] == ()

    def test_thin_sheet_plane_carries_the_sheet_source(self):
        from magnelio.geo import Brick, GeometryModel

        t = 20e-6
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 1e-3), material="air"))
        m.add(Brick(origin=(1e-3, 1e-3, 1e-3), size=(2e-3, 2e-3, t), material="pec"))
        m.add(Brick(origin=(0, 0, 1e-3 + t), size=(4e-3, 4e-3, 1e-3), material="air"))
        mesh = self._mesh(m, min_cell_size=100e-6)
        z = mesh.planes.records("z")
        sheet = [r for r in z if "sheet" in r.kinds]
        assert len(sheet) == 1
        assert {s.shape for s in sheet[0].sources if s.kind == "sheet"} == {1}
        # the far face is dropped globally and must not surface as unplaced
        assert mesh.planes.unplaced["z"] == ()

    def test_cpml_offsets_the_node_index(self):
        mesh = self._mesh(self._brick_model(bc={"xmin": "CPML", "zmax": "CPML"}))
        gp = mesh.planes
        assert gp.pml_cells == mesh.pml_cells and gp.pml_cells["xmin"] > 0
        x = gp.records("x")
        assert x[0].node == gp.pml_cells["xmin"]
        assert mesh.grid.x[x[0].node] == pytest.approx(-5e-3)
        assert mesh.grid.x[x[1].node] == pytest.approx(5e-3)
        z = gp.records("z")
        assert z[1].node == len(mesh.grid.z) - 1 - gp.pml_cells["zmax"]

    def test_pmc_pull_in_is_reported_as_moved(self):
        mesh = self._mesh(self._brick_model(bc={"xmin": "PMC", "xmax": "PMC"}))
        x = mesh.planes.records("x")
        assert x[0].position == pytest.approx(-5e-3)
        assert x[0].moved_to == pytest.approx(float(mesh.grid.x[0]))
        assert x[-1].moved_to == pytest.approx(float(mesh.grid.x[-1]))
        assert mesh.grid.x[0] > -5e-3

    def test_symmetry_clip_records_the_symmetry_source(self):
        mesh = self._mesh(self._brick_model(bc={"xmin": "SymmetryPMC"}))
        x = mesh.planes.records("x")
        assert x[0].position == pytest.approx(0.0)
        assert "symmetry" in x[0].kinds
        assert [s.label for s in x[0].sources if s.kind == "symmetry"] == ["xmin"]
        assert x[-1].position == pytest.approx(5e-3)
        assert mesh.planes.unplaced["x"] == ()

    def test_thin_wire_vertices_are_wire_planes(self):
        from magnelio.geo import Brick, Curve, GeometryModel, ThinWire

        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material="air"))
        wire = ThinWire(Curve.polyline([(2e-3, 5e-3, 5e-3), (8e-3, 5e-3, 5e-3)]), radius=0.1e-3)
        m.add(wire)
        mesh = self._mesh(m, min_nodes_per_wavelength=4)
        gp = mesh.planes
        assert _kinds_at(gp, "x", 2e-3, gp.feature_gap) >= {"wire"}
        assert _kinds_at(gp, "y", 5e-3, gp.feature_gap) >= {"wire"}
        assert {s.shape for r in gp.records("x") for s in r.sources if s.kind == "wire"} == {1}
        assert all(gp.unplaced[a] == () for a in "xyz")


class TestCarriage:
    """Trap: field-by-field Mesh rebuilds must carry ``planes``."""

    @pytest.fixture(autouse=True)
    def _occ(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")

    @pytest.fixture
    def mesh(self):
        return TestFromGeometry._mesh(TestFromGeometry._brick_model())

    def test_with_boundary_conditions_and_pec_boundaries(self, mesh):
        assert mesh.with_boundary_conditions({"zmax": "PEC"}).planes is mesh.planes
        assert mesh.with_pec_boundaries(["zmax"]).planes is mesh.planes

    def test_with_ports_and_elements(self, mesh):
        assert mesh.with_ports(()).planes is mesh.planes
        assert mesh.with_elements(()).planes is mesh.planes

    def test_from_grid_has_no_planes(self):
        from magnelio.mesh.grid import GridLines
        from magnelio.mesh.mesher import Mesh

        grid = GridLines(
            x=np.linspace(0, 1e-3, 4), y=np.linspace(0, 1e-3, 4), z=np.linspace(0, 1e-3, 4)
        )
        assert Mesh.from_grid(grid, [], background="air").planes is None


class TestSourcePlanes:
    """A source's total-field box asks for grid planes at its corners (DD-224)."""

    @pytest.fixture(autouse=True)
    def _occ(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")

    def test_box_corners_become_source_planes(self):
        from magnelio import sources
        from magnelio.geo import Brick, GeometryModel
        from magnelio.mesh.mesher import Mesh, MeshControl

        m = GeometryModel()
        m.add(Brick(origin=(-5e-3, -4e-3, 0.0), size=(10e-3, 8e-3, 6e-3), material="air"))
        m.add_source(
            sources.SourcePlaneWave(
                name="pw",
                direction=(0, 0, 1),
                polarization=(1, 0, 0),
                corners=((-3.3e-3, None, 1.1e-3), (3.3e-3, None, 4.9e-3)),
            )
        )
        mesh = Mesh.from_geometry(m, MeshControl(min_nodes_per_wavelength=6), f_max=10e9)
        for axis, positions in (("x", (-3.3e-3, 3.3e-3)), ("z", (1.1e-3, 4.9e-3))):
            for pos in positions:
                recs = [r for r in mesh.planes.records(axis) if abs(r.position - pos) < 1e-9]
                assert len(recs) == 1, (axis, pos)
                assert "source" in recs[0].kinds
                assert any(s.kind == "source" and s.label == "pw" for s in recs[0].sources)
                assert getattr(mesh.grid, axis)[recs[0].node] == pytest.approx(pos)
        # the open y sides asked for nothing
        assert not any("source" in r.kinds for r in mesh.planes.records("y"))
        assert PlaneSource.parse("source pw") == PlaneSource("source", None, "pw")
