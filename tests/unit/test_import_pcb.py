"""Board import end to end: fabrication data in, stacked solids out.

The fixtures are written here at test time, so what every layer of the
board contains is known by construction and the volumes can be checked
against arithmetic rather than against a previous run.  Three things
are worth the trouble of measuring: the layers land where the stackup
says, the holes are removed from everything they pass through, and the
barrel that fills a plated hole neither overlaps the layers it joins
nor leaves a gap between them.
"""

from __future__ import annotations

import json
import math

import pytest

from magnelio import Material
from magnelio.io import import_pcb

pytest.importorskip("OCC", reason="board import requires pythonocc-core")

MM = 1e-3
PEC = Material.pec()


# ── fixture writers ──────────────────────────────────────────────────


def _mm(value: float) -> str:
    """A length in the 4.6 coordinate format the fixtures use."""
    return f"{round(value * 1e6):d}"


def _gerber(*body: str) -> str:
    return "\n".join(["%FSLAX46Y46*%", "%MOMM*%", "G01*", "G75*", *body, "M02*"]) + "\n"


def _rectangle(x0, y0, x1, y1, *, region=False):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    body = [f"X{_mm(corners[0][0])}Y{_mm(corners[0][1])}D02*"]
    body += [f"X{_mm(x)}Y{_mm(y)}D01*" for x, y in corners[1:]]
    return ["G36*", *body, "G37*"] if region else body


def _drill(function: str, diameter: float, *hits, unit="METRIC"):
    lines = [
        "M48",
        f"; #@! TF.FileFunction,{function}",
        "FMAT,2",
        unit,
        f"T1C{diameter:.3f}",
        "%",
        "G90",
        "G05",
        "T1",
    ]
    lines += [f"X{x:.4f}Y{y:.4f}" for x, y in hits]
    return "\n".join([*lines, "T0", "M30"]) + "\n"


BOARD_X, BOARD_Y = 20.0, 10.0
BOARD_AREA = BOARD_X * BOARD_Y
COPPER_T, CORE_T = 0.035, 1.53
VIA_D, NPTH_D = 0.4, 2.0


def _two_layer(tmp_path, *, stackup=None, files=None, drills=True, epsilon="4.5"):
    """A two-layer board: a track and a pad on top, a ground plane below."""
    stack = stackup or [
        {"Type": "Copper", "Thickness": COPPER_T, "Name": "F.Cu"},
        {
            "Type": "Dielectric",
            "Thickness": CORE_T,
            "Material": "FR4",
            "DielectricConstant": epsilon,
            "LossTangent": "0.02",
            "Name": "F.Cu/B.Cu",
        },
        {"Type": "Copper", "Thickness": COPPER_T, "Name": "B.Cu"},
    ]
    attributes = [
        {"Path": "mini-F_Cu.gbr", "FileFunction": "Copper,L1,Top"},
        {"Path": "mini-B_Cu.gbr", "FileFunction": "Copper,L2,Bot"},
        {"Path": "mini-Edge_Cuts.gbr", "FileFunction": "Profile,NP"},
    ]
    if drills:
        attributes += [
            {"Path": "mini-PTH.drl", "FileFunction": "Plated,1,2,PTH"},
            {"Path": "mini-NPTH.drl", "FileFunction": "NonPlated,1,2,NPTH"},
        ]
    (tmp_path / "mini.gbrjob").write_text(
        json.dumps(
            {
                "GeneralSpecs": {"ProjectId": {"Name": "mini"}},
                "FilesAttributes": files if files is not None else attributes,
                "MaterialStackup": stack,
            }
        )
    )
    (tmp_path / "mini-Edge_Cuts.gbr").write_text(
        _gerber("%ADD10C,0.05*%", "D10*", *_rectangle(0.0, 0.0, BOARD_X, BOARD_Y))
    )
    (tmp_path / "mini-F_Cu.gbr").write_text(
        _gerber(
            "%TF.FileFunction,Copper,L1,Top*%",
            "%ADD10C,0.3*%",
            "%ADD11C,0.9*%",
            "D10*",
            f"X{_mm(2.0)}Y{_mm(5.0)}D02*",
            f"X{_mm(10.0)}Y{_mm(5.0)}D01*",
            "D11*",
            f"X{_mm(10.0)}Y{_mm(5.0)}D03*",
        )
    )
    (tmp_path / "mini-B_Cu.gbr").write_text(
        _gerber(
            "%TF.FileFunction,Copper,L2,Bot*%",
            *_rectangle(0.5, 0.5, BOARD_X - 0.5, BOARD_Y - 0.5, region=True),
        )
    )
    if drills:
        (tmp_path / "mini-PTH.drl").write_text(_drill("Plated,1,2,PTH", VIA_D, (10.0, 5.0)))
        (tmp_path / "mini-NPTH.drl").write_text(_drill("NonPlated,1,2,NPTH", NPTH_D, (18.0, 8.0)))
    return tmp_path


def _members(board):
    return {solid.name: solid for solid in board.members()}


def _import(tmp_path, *args, **kwargs):
    with pytest.warns(UserWarning, match="loss tangent"):
        return import_pcb(tmp_path, *args, **kwargs)


# ── the stack ────────────────────────────────────────────────────────


def test_every_stackup_layer_arrives_as_one_solid(tmp_path):
    board = _import(_two_layer(tmp_path))

    assert [solid.name for solid in board.members()] == [
        "F.Cu",
        "dielectric_1",
        "B.Cu",
        "via_1",
    ]
    assert board.name == "mini"


def test_the_layers_sit_where_the_stackup_puts_them(tmp_path):
    members = _members(_import(_two_layer(tmp_path)))

    def extent(name):
        lo, hi = members[name].bounding_box()
        return lo[2], hi[2]

    assert extent("F.Cu") == pytest.approx((0.0, COPPER_T * MM), abs=1e-12)
    assert extent("dielectric_1") == pytest.approx((-CORE_T * MM, 0.0), abs=1e-12)
    assert extent("B.Cu") == pytest.approx((-(CORE_T + COPPER_T) * MM, -CORE_T * MM), abs=1e-12)


def test_the_substrate_fills_the_outline_minus_its_holes(tmp_path):
    substrate = _members(_import(_two_layer(tmp_path)))["dielectric_1"]
    holes = math.pi / 4 * (NPTH_D**2 + VIA_D**2)

    assert substrate.volume() == pytest.approx((BOARD_AREA - holes) * CORE_T * MM**3)


def test_a_plated_hole_becomes_a_barrel_through_the_whole_board(tmp_path):
    via = _members(_import(_two_layer(tmp_path)))["via_1"]
    height = 2 * COPPER_T + CORE_T

    assert via.volume() == pytest.approx(math.pi / 4 * VIA_D**2 * height * MM**3)
    lo, hi = via.bounding_box()
    assert (lo[2], hi[2]) == pytest.approx((-(CORE_T + COPPER_T) * MM, COPPER_T * MM), abs=1e-12)


def test_holes_are_cut_from_the_copper_they_pass_through(tmp_path):
    plane = _members(_import(_two_layer(tmp_path)))["B.Cu"]
    zone = (BOARD_X - 1.0) * (BOARD_Y - 1.0)
    holes = math.pi / 4 * (NPTH_D**2 + VIA_D**2)

    assert plane.volume() == pytest.approx((zone - holes) * COPPER_T * MM**3)


def test_an_unplated_hole_leaves_no_metal_behind(tmp_path):
    """It is absent material; only a plated hole becomes a solid."""
    board = _import(_two_layer(tmp_path))

    assert sum(1 for solid in board.members() if solid.name.startswith("via_")) == 1


def test_the_solids_tile_the_board_without_overlapping(tmp_path):
    """The barrel has to fill the circles cut for it, exactly."""
    from magnelio.geo._occ_backend import boolean_union

    board = _import(_two_layer(tmp_path))
    solids = list(board.members())
    combined = boolean_union([solid._occ_shape() for solid in solids])

    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.VolumeProperties(combined, props)
    total = sum(solid.volume() for solid in solids)

    assert props.Mass() == pytest.approx(total, rel=1e-9)


# ── materials ────────────────────────────────────────────────────────


def test_copper_is_a_perfect_conductor_and_the_substrate_its_dielectric(tmp_path):
    members = _members(_import(_two_layer(tmp_path)))

    assert members["F.Cu"].material == PEC
    assert members["via_1"].material == PEC
    assert members["dielectric_1"].material.epsilon == pytest.approx((4.5, 4.5, 4.5))


def test_the_copper_material_can_be_replaced_wholesale(tmp_path):
    metal = Material.lossy_metal(name="copper", sigma=5.8e7)
    members = _members(_import(_two_layer(tmp_path), copper=metal))

    assert members["F.Cu"].material == metal
    assert members["B.Cu"].material == metal
    assert members["via_1"].material == metal
    assert members["dielectric_1"].material != metal


def test_a_named_material_beats_the_default(tmp_path):
    laminate = Material(name="RO4350B", epsilon=(3.66,) * 3)
    members = _members(_import(_two_layer(tmp_path), {"dielectric_1": laminate}))

    assert members["dielectric_1"].material == laminate
    assert members["F.Cu"].material == PEC


def test_a_wildcard_reaches_every_barrel(tmp_path):
    gold = Material.lossy_metal(name="gold", sigma=4.1e7)
    members = _members(_import(_two_layer(tmp_path), {"via_*": gold}))

    assert members["via_1"].material == gold


def test_a_material_key_that_matches_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="matches none of the solids"):
        import_pcb(_two_layer(tmp_path), {"In1.Cu": PEC})


def test_a_substrate_without_a_permittivity_is_not_silently_vacuum(tmp_path):
    root = _two_layer(tmp_path, epsilon=None)
    document = json.loads((root / "mini.gbrjob").read_text())
    del document["MaterialStackup"][1]["DielectricConstant"]
    (root / "mini.gbrjob").write_text(json.dumps(document))

    with pytest.warns(UserWarning, match="no dielectric constant"):
        board = import_pcb(root)

    assert _members(board)["dielectric_1"].material is None


def test_a_loss_tangent_is_reported_and_not_modelled(tmp_path):
    """One number at an unrecorded frequency is not a causal material."""
    with pytest.warns(UserWarning, match=r"loss tangent .*not.*modelled"):
        board = import_pcb(_two_layer(tmp_path))

    assert _members(board)["dielectric_1"].material.dispersion is None


# ── blind vias ───────────────────────────────────────────────────────


def _four_layer(tmp_path):
    """Four copper layers with a via that only joins the top two."""
    stack = [
        {"Type": "Copper", "Thickness": COPPER_T, "Name": "F.Cu"},
        {"Type": "Dielectric", "Thickness": 0.2, "DielectricConstant": "4.5", "Name": "prepreg"},
        {"Type": "Copper", "Thickness": 0.018, "Name": "In1.Cu"},
        {"Type": "Dielectric", "Thickness": 1.0, "DielectricConstant": "4.5", "Name": "core"},
        {"Type": "Copper", "Thickness": 0.018, "Name": "In2.Cu"},
        {"Type": "Dielectric", "Thickness": 0.2, "DielectricConstant": "4.5", "Name": "prepreg"},
        {"Type": "Copper", "Thickness": COPPER_T, "Name": "B.Cu"},
    ]
    files = [
        {"Path": f"mini-{layer}.gbr", "FileFunction": f"Copper,L{number},Inr"}
        for number, layer in enumerate(("F_Cu", "In1_Cu", "In2_Cu", "B_Cu"), start=1)
    ]
    files += [
        {"Path": "mini-Edge_Cuts.gbr", "FileFunction": "Profile,NP"},
        {"Path": "mini-blind.drl", "FileFunction": "Plated,1,2,Blind"},
    ]
    _two_layer(tmp_path, stackup=stack, files=files, drills=False)
    plane = _gerber(*_rectangle(0.5, 0.5, BOARD_X - 0.5, BOARD_Y - 0.5, region=True))
    for layer in ("In1_Cu", "In2_Cu"):
        (tmp_path / f"mini-{layer}.gbr").write_text(plane)
    (tmp_path / "mini-B_Cu.gbr").write_text(plane)
    (tmp_path / "mini-blind.drl").write_text(_drill("Plated,1,2,Blind", VIA_D, (10.0, 5.0)))
    return tmp_path


def test_a_blind_via_stops_at_the_layer_it_reaches(tmp_path):
    board = import_pcb(_four_layer(tmp_path))
    members = _members(board)
    lo, hi = members["via_1"].bounding_box()

    assert hi[2] == pytest.approx(COPPER_T * MM, abs=1e-12)
    assert lo[2] == pytest.approx(-(0.2 + 0.018) * MM, abs=1e-12)


def test_a_blind_via_leaves_the_layers_below_it_untouched(tmp_path):
    members = _members(import_pcb(_four_layer(tmp_path)))
    zone = (BOARD_X - 1.0) * (BOARD_Y - 1.0)

    assert members["B.Cu"].volume() == pytest.approx(zone * COPPER_T * MM**3)
    assert members["In2.Cu"].volume() == pytest.approx(zone * 0.018 * MM**3)
    assert members["In1.Cu"].volume() == pytest.approx(
        (zone - math.pi / 4 * VIA_D**2) * 0.018 * MM**3
    )


# ── refusals ─────────────────────────────────────────────────────────


def test_a_folder_without_a_job_file_says_what_is_missing(tmp_path):
    (tmp_path / "mini-F_Cu.gbr").write_text(_gerber())

    with pytest.raises(FileNotFoundError, match="job file"):
        import_pcb(tmp_path)


def test_a_set_without_an_outline_is_an_error(tmp_path):
    root = _two_layer(tmp_path)
    document = json.loads((root / "mini.gbrjob").read_text())
    document["FilesAttributes"] = [
        entry for entry in document["FilesAttributes"] if "Profile" not in entry["FileFunction"]
    ]
    (root / "mini.gbrjob").write_text(json.dumps(document))

    with pytest.raises(ValueError, match="no board outline"):
        import_pcb(root)


def test_a_missing_copper_file_is_an_error(tmp_path):
    root = _two_layer(tmp_path)
    (root / "mini-B_Cu.gbr").unlink()

    with pytest.raises(FileNotFoundError, match="mini-B_Cu.gbr"):
        import_pcb(root)


def test_a_copper_layer_the_job_file_does_not_plot_is_an_error(tmp_path):
    root = _two_layer(tmp_path)
    document = json.loads((root / "mini.gbrjob").read_text())
    document["FilesAttributes"] = [
        entry for entry in document["FilesAttributes"] if "L2" not in entry["FileFunction"]
    ]
    (root / "mini.gbrjob").write_text(json.dumps(document))

    with pytest.raises(ValueError, match="names no Gerber file"):
        import_pcb(root)


def test_the_group_can_be_renamed(tmp_path):
    board = _import(_two_layer(tmp_path), name="dut")

    assert board.name == "dut"


def test_a_drill_file_of_unknown_plating_says_so(tmp_path):
    """Silently dropping a barrel would change the model, not the picture."""
    root = _two_layer(tmp_path)
    document = json.loads((root / "mini.gbrjob").read_text())
    for entry in document["FilesAttributes"]:
        if entry["Path"].endswith("PTH.drl"):
            entry["FileFunction"] = "Other,drill"
    (root / "mini.gbrjob").write_text(json.dumps(document))
    (root / "mini-PTH.drl").write_text(
        "\n".join(
            ["M48", "FMAT,2", "METRIC", "T1C0.400", "%", "G90", "G05", "T1", "X10.0Y5.0", "M30"]
        )
    )

    with pytest.warns(UserWarning, match="whether its holes are plated"):
        board = import_pcb(root)

    assert not any(solid.name.startswith("via_") for solid in board.members())


def test_a_drill_file_the_job_file_omits_is_still_read(tmp_path):
    """A set whose job file lists no drill file must not lose its holes."""
    root = _two_layer(tmp_path)
    document = json.loads((root / "mini.gbrjob").read_text())
    document["FilesAttributes"] = [
        entry for entry in document["FilesAttributes"] if not entry["Path"].endswith(".drl")
    ]
    (root / "mini.gbrjob").write_text(json.dumps(document))

    board = _import(root)
    substrate = _members(board)["dielectric_1"]
    holes = math.pi / 4 * (NPTH_D**2 + VIA_D**2)

    assert substrate.volume() == pytest.approx((BOARD_AREA - holes) * CORE_T * MM**3)


def test_a_gerber_assigned_to_the_wrong_layer_is_an_error(tmp_path):
    """The layers would stack in the wrong order — a different board."""
    root = _two_layer(tmp_path)
    document = json.loads((root / "mini.gbrjob").read_text())
    for entry in document["FilesAttributes"]:
        if entry["Path"] == "mini-F_Cu.gbr":
            entry["FileFunction"] = "Copper,L2,Bot"
        elif entry["Path"] == "mini-B_Cu.gbr":
            entry["FileFunction"] = "Copper,L1,Top"
    (root / "mini.gbrjob").write_text(json.dumps(document))

    with pytest.raises(ValueError, match="the file itself says it is layer L2"):
        import_pcb(root)
