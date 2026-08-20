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

import warnings

import pytest

from magnelio import GeometryModel, Material
from magnelio.geo import Brick, ImportedSolid
from magnelio.io import import_brep, import_step, write_brep
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

    *assembly*, when given, is a list of ``(part_index, dx, name)``
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
        for index, dx, name in assembly:
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(dx, 0.0, 0.0))
            component = shape_tool.AddComponent(root, labels[index], TopLoc_Location(trsf))
            if name is not None:
                TDataStd_Name.Set(component, name)
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
            import_step(_simple_step(tmp_path), "PEC")
        assert "Material" in str(excinfo.value)

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

        # Two boxes fused across a shared plane: the seam splits what is
        # geometrically one face into several.
        from magnelio.geo._occ_backend import boolean_union

        fused = boolean_union([_box(2.0, 2.0, 2.0), _box(2.0, 2.0, 2.0, at=(2.0, 0.0, 0.0))])
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

    def test_surface_model_is_reported_not_silently_empty(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(tmp_path / "sheet.step", [(face, "plate", None)])
        with pytest.raises(ValueError) as excinfo, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import_step(path)
        assert "no solid" in str(excinfo.value)

    def test_skipped_surface_bodies_are_warned_about(self, tmp_path):
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
        from OCC.Core.gp import gp_Pln

        face = BRepBuilderAPI_MakeFace(gp_Pln(), -1.0, 1.0, -1.0, 1.0).Shape()
        path = _write_step(
            tmp_path / "mixed.step",
            [(_box(1.0, 1.0, 1.0), "block", None), (face, "plate", None)],
        )
        with pytest.warns(UserWarning, match="plate"):
            parts = list(import_step(path).members())
        assert [s.name for s in parts] == ["block"]
