"""CAD import gates (DD-178): STEP and BREP into the geometry API.

The fixtures are written at test time with the same kernel that reads
them back, which is what makes the unit contract checkable: a file
whose unit, names and colours are known by construction.  The single
most important gate is the unit one — a millimetre file has to arrive
in meters, because everything downstream (mesh, frequencies, port
impedances) is metric and a factor of 1000 is silent otherwise.

A file exported by a real CAD system is exercised in the integration
suite; here the kernel writes what a CAD system would.
"""

from __future__ import annotations

import pytest

from magnelio import GeometryModel, Material
from magnelio.geo import Brick, Curve, ImportedSheet, ImportedSolid, Solid
from magnelio.io import export_brep, export_step, import_brep, import_step, write_brep
from magnelio.io.cad import _resolve_materials, _unit_factor

PEC = Material.pec()
PTFE = Material(name="PTFE", epsilon=(2.1, 2.1, 2.1))

pytest.importorskip("OCC", reason="CAD import requires pythonocc-core")


# ── fixture writers ──────────────────────────────────────────────────


def _box(dx, dy, dz, at=(0.0, 0.0, 0.0)):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    return BRepPrimAPI_MakeBox(gp_Pnt(*at), dx, dy, dz).Shape()


def _write_step(path, parts, *, unit="MM", assembly=None):
    """Write a STEP file from ``(shape, name, rgb)`` parts.

    *assembly*, when given, is a list of ``(part_index, dx, name[, rgb])``
    component placements referring to the parts, so the file carries a
    real assembly structure instead of free-standing solids.
    """
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

    doc = TDocStd_Document("XCAF")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())

    labels = []
    for shape, name, rgb in parts:
        label = shape_tool.AddShape(shape, False)
        labels.append(label)
        if name is not None:
            TDataStd_Name.Set(label, name)
        if rgb is not None:
            color_tool.SetColor(label, Quantity_Color(*rgb, Quantity_TOC_RGB), XCAFDoc_ColorSurf)

    if assembly:
        root = shape_tool.NewShape()
        TDataStd_Name.Set(root, "assembly")
        for placement in assembly:
            index, dx, name, *instance_color = placement
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(dx, 0.0, 0.0))
            component = shape_tool.AddComponent(root, labels[index], TopLoc_Location(trsf))
            if name is not None:
                TDataStd_Name.Set(component, name)
            if instance_color:
                color_tool.SetColor(
                    component,
                    Quantity_Color(*instance_color[0], Quantity_TOC_RGB),
                    XCAFDoc_ColorSurf,
                )
        shape_tool.UpdateAssemblies()

    Interface_Static.SetCVal("write.step.unit", unit)
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def _simple_step(tmp_path, **kwargs):
    return _write_step(
        tmp_path / "parts.step",
        [
            (_box(10.0, 4.0, 2.0), "substrate", (0.2, 0.6, 0.3)),
            (_box(3.0, 3.0, 3.0), "pin", None),
        ],
        **kwargs,
    )


# ── units ────────────────────────────────────────────────────────────


class TestUnits:
    """What the file was drawn in must not reach the caller."""

    def test_millimetre_file_arrives_in_meters(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path)).members())
        assert parts[0].bounding_box()[1] == pytest.approx((10e-3, 4e-3, 2e-3))

    def test_the_declared_unit_decides_the_size(self, tmp_path):
        # Same coordinates, different unit declaration: the kernel's
        # writer converts the numbers when it changes the unit, so only
        # patching the declaration in place isolates the reader's unit
        # handling from everything else.
        path = _simple_step(tmp_path)
        as_meters = tmp_path / "meters.step"
        as_meters.write_text(
            path.read_text().replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)")
        )
        millimetre = next(iter(import_step(path).members()))
        metre = next(iter(import_step(as_meters).members()))
        assert metre.bounding_box()[1][0] == pytest.approx(1000.0 * millimetre.bounding_box()[1][0])
        assert metre.bounding_box()[1][0] == pytest.approx(10.0)

    def test_written_unit_round_trips(self, tmp_path):
        # Writing the same model in another unit is a conversion, not a
        # reinterpretation: the part keeps its true size.
        for unit in ("M", "CM", "INCH"):
            part = next(iter(import_step(_simple_step(tmp_path, unit=unit)).members()))
            assert part.bounding_box()[1] == pytest.approx((10e-3, 4e-3, 2e-3))

    def test_reader_unit_setting_is_restored(self, tmp_path):
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.STEPControl import STEPControl_Reader

        STEPControl_Reader()  # registers the STEP statics
        Interface_Static.SetCVal("xstep.cascade.unit", "INCH")
        import_step(_simple_step(tmp_path))
        # Process-global setting: leaking "M" would silently re-scale
        # every later STEP read in the same session, ours or not.
        assert Interface_Static.CVal("xstep.cascade.unit") == "INCH"
        Interface_Static.SetCVal("xstep.cascade.unit", "MM")

    def test_unit_names_and_factors_agree(self):
        assert _unit_factor("mm") == pytest.approx(1e-3)
        assert _unit_factor("MM") == pytest.approx(1e-3)
        assert _unit_factor(1e-3) == pytest.approx(1e-3)

    def test_unknown_unit_lists_the_known_ones(self):
        with pytest.raises(ValueError) as excinfo:
            _unit_factor("furlong")
        assert "'mm'" in str(excinfo.value)

    def test_negative_unit_is_rejected(self):
        with pytest.raises(ValueError):
            _unit_factor(-1.0)

    def test_non_finite_unit_is_rejected(self):
        for unit in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="finite and positive"):
                _unit_factor(unit)


# ── names, colours, assemblies ───────────────────────────────────────


class TestMetadata:
    """Names and colours are what STEP has over BREP."""

    def test_solid_names_come_from_the_file(self, tmp_path):
        names = [s.name for s in import_step(_simple_step(tmp_path)).members()]
        assert names == ["substrate", "pin"]

    def test_display_colour_is_read(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path)).members())
        assert parts[0].color == pytest.approx((0.2, 0.6, 0.3), abs=1e-6)
        assert parts[1].color is None

    def test_unnamed_solids_get_unique_synthetic_names(self, tmp_path):
        path = _write_step(
            tmp_path / "anon.step",
            [(_box(1.0, 1.0, 1.0), None, None), (_box(2.0, 1.0, 1.0), None, None)],
        )
        names = [s.name for s in import_step(path).members()]
        assert names == ["solid_1", "solid_2"]
        assert len(set(names)) == 2

    def test_group_is_named_after_the_file(self, tmp_path):
        assert import_step(_simple_step(tmp_path)).name == "parts"

    def test_assembly_placements_are_baked_into_the_solids(self, tmp_path):
        path = _write_step(
            tmp_path / "asm.step",
            [(_box(2.0, 2.0, 2.0), "cube", None)],
            assembly=[(0, 0.0, "cube_a"), (0, 10.0, "cube_b")],
        )
        parts = {s.name: s for s in import_step(path).members()}
        assert set(parts) == {"cube_a", "cube_b"}
        assert parts["cube_a"].bounding_box()[0][0] == pytest.approx(0.0)
        assert parts["cube_b"].bounding_box()[0][0] == pytest.approx(10e-3)

    def test_multi_solid_part_names_are_suffixed(self, tmp_path):
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Compound

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, _box(1.0, 1.0, 1.0))
        builder.Add(compound, _box(1.0, 1.0, 1.0, at=(5.0, 0.0, 0.0)))
        path = _write_step(tmp_path / "pair.step", [(compound, "twin", None)])
        assert [s.name for s in import_step(path).members()] == ["twin_1", "twin_2"]


# ── material assignment ──────────────────────────────────────────────


class TestMaterialMapping:
    """Materials are not in the file, so they are assigned by name."""

    def test_single_material_broadcasts(self, tmp_path):
        parts = import_step(_simple_step(tmp_path), PEC).members()
        assert all(s.material is PEC for s in parts)

    def test_literal_names_map_one_by_one(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path), {"substrate": PTFE, "pin": PEC}).members())
        assert [s.material.name for s in parts] == ["PTFE", "PEC"]

    def test_wildcard_matches_a_family(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path), {"*": PTFE}).members())
        assert all(s.material is PTFE for s in parts)

    def test_literal_beats_wildcard(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path), {"*": PTFE, "pin": PEC}).members())
        assert [s.material.name for s in parts] == ["PTFE", "PEC"]

    def test_conflicting_wildcards_are_an_error(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            import_step(_simple_step(tmp_path), {"p*": PEC, "*n": PTFE})
        assert "two patterns" in str(excinfo.value)

    def test_same_material_through_two_wildcards_is_fine(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path), {"p*": PEC, "*n": PEC}).members())
        assert parts[1].material is PEC

    def test_key_matching_nothing_names_the_available_solids(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            import_step(_simple_step(tmp_path), {"Substrate": PEC})
        message = str(excinfo.value)
        assert "'substrate'" in message and "'pin'" in message

    def test_unmapped_solids_are_construction_bodies(self, tmp_path):
        parts = list(import_step(_simple_step(tmp_path), {"pin": PEC}).members())
        assert parts[0].material is None
        with pytest.raises(ValueError) as excinfo:
            GeometryModel().add(parts[0])
        assert "construction body" in str(excinfo.value)

    def test_wrong_materials_type_is_rejected(self, tmp_path):
        with pytest.raises(TypeError) as excinfo:
            import_step(_simple_step(tmp_path), 1.0)
        assert "Material" in str(excinfo.value)

    def test_builtin_name_string_broadcasts(self, tmp_path):
        # DD-185: "pec" is the canonical PEC instance, applied to
        # every solid like the explicit broadcast form.
        parts = import_step(_simple_step(tmp_path), "PEC")
        assert all(part.material.is_pec for part in parts.members())

    def test_duplicate_solid_names_are_all_matched(self):
        assigned = _resolve_materials(["ring", "ring"], {"ring": PEC})
        assert assigned == [PEC, PEC]


# ── the imported solid is a full citizen ─────────────────────────────


class TestImportedSolid:
    """An imported solid has to behave like any other shape."""

    def test_verbs_and_booleans_work(self, tmp_path):
        part = next(iter(import_step(_simple_step(tmp_path), PEC).members()))
        assert isinstance(part, ImportedSolid)
        assert part.volume() == pytest.approx(10e-3 * 4e-3 * 2e-3)
        moved = part.translated((1e-3, 0.0, 0.0))
        assert moved.bounding_box()[0][0] == pytest.approx(1e-3)
        assert moved.material is PEC
        cut = part - Brick(origin=(0, 0, 0), size=(10e-3, 4e-3, 1e-3))
        assert cut.volume() == pytest.approx(10e-3 * 4e-3 * 1e-3)

    def test_group_can_be_added_to_a_model(self, tmp_path):
        model = GeometryModel().add(import_step(_simple_step(tmp_path), PEC))
        assert len(model.shapes) == 2

    def test_scale_choice_uses_the_meter_space_shape(self, tmp_path):
        # DD-120: a millimetre-sized part is built at a large model
        # scale; the stored BRep stays in meters either way.
        part = next(iter(import_step(_simple_step(tmp_path), PEC).members()))
        low, high = part._analytic_bbox()
        assert high[0] == pytest.approx(10e-3)
        assert part._occ_shape(1000.0) is not part._occ_shape(1.0)


# ── healing ──────────────────────────────────────────────────────────


class TestHealing:
    """Repair passes run by default and never fail the import."""

    def test_healing_keeps_the_geometry(self, tmp_path):
        healed = next(iter(import_step(_simple_step(tmp_path), PEC, heal=True).members()))
        raw = next(iter(import_step(_simple_step(tmp_path), PEC, heal=False).members()))
        assert healed.volume() == pytest.approx(raw.volume())

    def test_unify_merges_faces_without_changing_the_volume(self, tmp_path):
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        # Two boxes fused across a shared plane by the kernel's plain
        # fuser: the seam splits what is geometrically one face into
        # several (the library's own union removes such seams itself).
        from magnelio.geo._occ_backend import _require_occ, _run_bop

        fused = _run_bop(
            _require_occ()["Fuse"], [_box(2.0, 2.0, 2.0)], [_box(2.0, 2.0, 2.0, at=(2.0, 0.0, 0.0))]
        )
        path = tmp_path / "fused.brep"
        from OCC.Core.BRepTools import breptools

        assert breptools.Write(fused, str(path))

        def faces(shape):
            count, explorer = 0, TopExp_Explorer(shape, TopAbs_FACE)
            while explorer.More():
                count += 1
                explorer.Next()
            return count

        plain = next(iter(import_brep(path, unit="mm", material=PEC).members()))
        as_written = _write_step(tmp_path / "f.step", [(fused, "block", None)])
        merged = next(iter(import_step(as_written, PEC).members()))
        unified = next(iter(import_step(as_written, PEC, unify=True).members()))
        assert unified.volume() == pytest.approx(plain.volume())
        assert faces(unified._shape) < faces(merged._shape)


# ── BREP ─────────────────────────────────────────────────────────────


class TestImportBrep:
    """BREP carries no unit, so the caller has to supply one."""

    def _brep(self, tmp_path):
        path = tmp_path / "part.brep"
        write_brep([Brick(origin=(0, 0, 0), size=(10e-3, 4e-3, 2e-3), material=PEC)], path)
        return path

    def test_unit_is_mandatory(self, tmp_path):
        with pytest.raises(TypeError):
            import_brep(self._brep(tmp_path))  # pyright: ignore[reportCallIssue]

    def test_meters_round_trip_exactly(self, tmp_path):
        part = next(iter(import_brep(self._brep(tmp_path), unit="m", material=PEC).members()))
        assert part.bounding_box()[1] == pytest.approx((10e-3, 4e-3, 2e-3))

    def test_unit_name_and_factor_agree(self, tmp_path):
        by_name = next(iter(import_brep(self._brep(tmp_path), unit="mm").members()))
        by_factor = next(iter(import_brep(self._brep(tmp_path), unit=1e-3).members()))
        assert by_name.bounding_box()[1] == pytest.approx(by_factor.bounding_box()[1])

    def test_solids_are_named_after_the_file(self, tmp_path):
        assert next(iter(import_brep(self._brep(tmp_path), unit="m").members())).name == "part"

    def test_several_solids_are_numbered(self, tmp_path):
        path = tmp_path / "two.brep"
        write_brep(
            [
                Brick(origin=(0, 0, 0), size=(1e-3, 1e-3, 1e-3), material=PEC),
                Brick(origin=(5e-3, 0, 0), size=(1e-3, 1e-3, 1e-3), material=PEC),
            ],
            path,
        )
        assert [s.name for s in import_brep(path, unit="m").members()] == ["two_1", "two_2"]

    def test_material_is_optional(self, tmp_path):
        part = next(iter(import_brep(self._brep(tmp_path), unit="m").members()))
        assert part.material is None


# ── failure modes ────────────────────────────────────────────────────


class TestFailures:
    """A bad file has to say what is wrong with it."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_step(tmp_path / "nope.step")
        with pytest.raises(FileNotFoundError):
            import_brep(tmp_path / "nope.brep", unit="m")

    def test_not_a_step_file(self, tmp_path):
        path = tmp_path / "junk.step"
        path.write_text("this is not a STEP file\n")
        with pytest.raises(ValueError) as excinfo:
            import_step(path)
        assert "STEP" in str(excinfo.value)

    def test_surface_model_is_imported_as_sheet(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(tmp_path / "sheet.step", [(face, "plate", None)])
        (sheet,) = import_step(path).members()
        assert isinstance(sheet, ImportedSheet)
        assert sheet.name == "plate"
        assert sheet.bounding_box()[1][:2] == pytest.approx((1e-3, 1e-3))
        physical_sheet = next(import_step(path, {"plate": PEC}).members())
        from magnelio.mesh.mesher import Mesh, MeshControl

        with pytest.raises(NotImplementedError, match="sheet"):
            Mesh.from_geometry(GeometryModel().add(physical_sheet), MeshControl(), f_max=10e9)

    def test_mixed_surface_and_solid_are_both_imported(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(
            tmp_path / "mixed.step",
            [(_box(1.0, 1.0, 1.0), "block", None), (face, "plate", None)],
        )
        parts = list(import_step(path).members())
        assert [s.name for s in parts] == ["block", "plate"]
        assert isinstance(parts[0], ImportedSolid)
        assert isinstance(parts[1], ImportedSheet)


class TestExchange:
    def test_compound_keeps_free_faces_without_solid_face_duplicates(self, tmp_path):
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt
        from OCC.Core.TopoDS import TopoDS_Compound

        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        builder.Add(compound, _box(2.0, 3.0, 4.0))
        builder.Add(compound, BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape())
        other_plane = gp_Pln(gp_Pnt(0, 0, 5), gp_Dir(0, 0, 1))
        builder.Add(compound, BRepBuilderAPI_MakeFace(other_plane, -1.0, 1.0, -1.0, 1.0).Shape())
        source = _write_step(tmp_path / "mixed_compound.step", [(compound, "part", None)])
        found = list(import_step(source).members())
        assert len(found) == 3
        assert [type(s) for s in found] == [ImportedSolid, ImportedSheet, ImportedSheet]

    def test_placed_sheet_instances_and_duplicate_names(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(
            tmp_path / "placed.step",
            [(face, "prototype", (0.3, 0.4, 0.5))],
            assembly=[(0, 0.0, "panel"), (0, 10.0, "panel")],
        )
        found = list(import_step(path, {"panel": PEC}).members())
        assert [s.name for s in found] == ["panel", "panel"]
        assert [s.bounding_box()[0][0] for s in found] == pytest.approx([-1e-3, 9e-3])
        assert [s.color for s in found] == [pytest.approx((0.3, 0.4, 0.5), abs=1e-6)] * 2
        assert all(s.material is PEC for s in found)
        exported = tmp_path / "placed_again.step"
        export_step(exported, found)
        replay = list(import_step(exported).members())
        assert len(replay) == 2
        assert [s.bounding_box()[0][0] for s in replay] == pytest.approx([-1e-3, 9e-3])

    def test_instance_sheet_colours_override_prototype(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(
            tmp_path / "instance_colors.step",
            [(face, "prototype", (0.0, 1.0, 0.0))],
            assembly=[
                (0, 0.0, "red", (1.0, 0.0, 0.0)),
                (0, 10.0, "blue", (0.0, 0.0, 1.0)),
            ],
        )
        found = list(import_step(path).members())
        assert [s.name for s in found] == ["red", "blue"]
        assert found[0].color == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
        assert found[1].color == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
        exported = tmp_path / "instance_colors_again.step"
        export_step(exported, found)
        replay = list(import_step(exported).members())
        assert [s.name for s in replay] == ["red", "blue"]
        assert replay[0].color == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
        assert replay[1].color == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)

    def test_nested_sheet_assembly_placements_accumulate(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln, gp_Trsf, gp_Vec
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
        from OCC.Core.TDataStd import TDataStd_Name
        from OCC.Core.TDocStd import TDocStd_Document
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool

        doc = TDocStd_Document("XCAF")
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1, 1, -1, 1).Shape()
        leaf = shape_tool.AddShape(face, False)
        middle = shape_tool.NewShape()
        root = shape_tool.NewShape()
        inner = gp_Trsf()
        inner.SetTranslation(gp_Vec(3, 0, 0))
        outer = gp_Trsf()
        outer.SetTranslation(gp_Vec(7, 0, 0))
        shape_tool.AddComponent(middle, leaf, TopLoc_Location(inner))
        component = shape_tool.AddComponent(root, middle, TopLoc_Location(outer))
        TDataStd_Name.Set(component, "assembly")
        shape_tool.UpdateAssemblies()
        Interface_Static.SetCVal("write.step.unit", "MM")
        writer = STEPCAFControl_Writer()
        writer.Transfer(doc)
        path = tmp_path / "nested.step"
        assert writer.Write(str(path)) == IFSelect_RetDone
        (sheet,) = import_step(path).members()
        assert sheet.bounding_box()[0][0] == pytest.approx(9e-3)

    def test_curved_trimmed_face_remains_sheet(self, tmp_path):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCC.Core.GeomAbs import GeomAbs_Cylinder
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer

        cylinder = BRepPrimAPI_MakeCylinder(2.0, 5.0).Shape()
        explorer = TopExp_Explorer(cylinder, TopAbs_FACE)
        curved = None
        while explorer.More():
            candidate = explorer.Current()
            if BRepAdaptor_Surface(candidate).GetType() == GeomAbs_Cylinder:
                curved = candidate
                break
            explorer.Next()
        assert curved is not None
        source = _write_step(tmp_path / "curved.step", [(curved, "wall", None)])
        sheet = next(import_step(source).members())
        assert isinstance(sheet, ImportedSheet)
        assert sheet.bounding_box()[1][2] == pytest.approx(5e-3)
        target = tmp_path / "curved.brep"
        export_brep(target, sheet, unit="mm")
        assert isinstance(next(import_brep(target, unit="mm").members()), ImportedSheet)

    def test_free_curve_is_reported_and_cannot_be_exported(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
        from OCC.Core.gp import gp_Pnt

        edge = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0)).Shape()
        source = _write_step(tmp_path / "curve.step", [(edge, "line", None)])
        with pytest.warns(UserWarning, match="line"):
            with pytest.raises(ValueError, match="no solid or free face"):
                import_step(source)
        brep = tmp_path / "curve.brep"
        write_brep([edge], brep)
        with pytest.warns(UserWarning, match="unsupported free curves"):
            with pytest.raises(ValueError, match="no solid or free face"):
                import_brep(brep, unit="mm")
        with pytest.raises(TypeError, match="Solid and Sheet"):
            export_step(tmp_path / "curve_out.step", Curve.line((0, 0, 0), (1, 0, 0)))

    def test_imported_sheet_can_be_given_resolved_thickness(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(tmp_path / "sheet.step", [(face, "plate", None)])
        sheet = next(import_step(path, {"plate": PEC}).members())
        slab = sheet.thickened(0.2e-3)
        assert isinstance(slab, Solid)
        assert slab.material is PEC
        assert slab.volume() == pytest.approx(0.8e-9, rel=1e-3)

        import numpy as np

        from magnelio import Mesh, open_project
        from magnelio.io.project import ProjectStore
        from magnelio.mesh import GridLines

        grid = GridLines(
            x=np.linspace(-2e-3, 2e-3, 3),
            y=np.linspace(-2e-3, 2e-3, 3),
            z=np.linspace(-1e-3, 1e-3, 3),
        )
        ProjectStore.create(tmp_path / "sheet_project", Mesh.from_grid(grid), geometry=[slab])
        (replayed,) = open_project(tmp_path / "sheet_project").geometry
        assert replayed.material == PEC
        assert replayed.volume() == pytest.approx(slab.volume())

    def test_step_round_trip_mixed_categories_units_and_metadata(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        source = _write_step(
            tmp_path / "source.step",
            [(_box(2.0, 3.0, 4.0), "block", (0.2, 0.6, 0.3)), (face, "plate", None)],
        )
        imported = import_step(source, {"block": PEC, "plate": PTFE})
        for unit in ("m", "cm", "mm", "in"):
            target = tmp_path / f"target_{unit}.step"
            export_step(target, imported, unit=unit)
            found = list(import_step(target).members())
            assert [type(s) for s in found] == [ImportedSolid, ImportedSheet]
            assert [s.name for s in found] == ["block", "plate"]
            assert found[0].color == pytest.approx((0.2, 0.6, 0.3), abs=1e-6)
            assert found[0].bounding_box()[1] == pytest.approx((2e-3, 3e-3, 4e-3))
            assert found[1].bounding_box()[1][:2] == pytest.approx((1e-3, 1e-3))
            assert all(s.material is None for s in found)

    def test_brep_round_trip_and_overwrite(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        source = _write_step(
            tmp_path / "source.step",
            [(_box(2.0, 3.0, 4.0), "block", None), (face, "plate", None)],
        )
        imported = import_step(source)
        target = tmp_path / "target.brep"
        export_brep(target, imported, unit="cm")
        with pytest.raises(FileExistsError):
            export_brep(target, imported, unit="cm")
        export_brep(target, imported, unit="cm", overwrite=True)
        found = list(import_brep(target, unit="cm").members())
        assert [type(s) for s in found] == [ImportedSolid, ImportedSheet]
        assert found[0].bounding_box()[1] == pytest.approx((2e-3, 3e-3, 4e-3))
        assert found[1].bounding_box()[1][:2] == pytest.approx((1e-3, 1e-3))

    def test_step_export_overwrite_and_unit_setting_restored(self, tmp_path):
        from OCC.Core.Interface import Interface_Static

        body = Brick(origin=(0, 0, 0), size=(0.01, 0.02, 0.03), material=PEC, name="body")
        previous_write = Interface_Static.CVal("write.step.unit")
        previous_cascade = Interface_Static.CVal("xstep.cascade.unit")
        Interface_Static.SetCVal("write.step.unit", "INCH")
        Interface_Static.SetCVal("xstep.cascade.unit", "CM")
        try:
            target = tmp_path / "body.step"
            export_step(target, body, unit="mm")
            assert Interface_Static.CVal("write.step.unit") == "INCH"
            assert Interface_Static.CVal("xstep.cascade.unit") == "CM"
            assert next(import_step(target).members()).bounding_box()[1] == pytest.approx(
                (0.01, 0.02, 0.03)
            )
            with pytest.raises(FileExistsError):
                export_step(target, body)
            export_step(target, body, overwrite=True)
        finally:
            if previous_write is not None:
                Interface_Static.SetCVal("write.step.unit", previous_write)
            if previous_cascade is not None:
                Interface_Static.SetCVal("xstep.cascade.unit", previous_cascade)

    @pytest.mark.parametrize("size", [1e-9, 1e-3, 1.0])
    def test_export_respects_geometry_model_scale(self, tmp_path, size):
        from OCC.Core.BRepCheck import BRepCheck_Analyzer

        body = Brick(origin=(0, 0, 0), size=(size, size, size), name="scaled")
        step = tmp_path / "scaled.step"
        brep = tmp_path / "scaled.brep"
        export_step(step, body, unit="mm")
        export_brep(brep, body, unit="mm")
        for restored in (
            import_step(step),
            import_brep(brep, unit="mm"),
            import_brep(brep, unit="mm", heal=True),
        ):
            (member,) = restored.members()
            assert member.bounding_box()[1] == pytest.approx((size,) * 3, rel=1e-6, abs=1e-12)
            assert BRepCheck_Analyzer(member._occ_shape()).IsValid()
