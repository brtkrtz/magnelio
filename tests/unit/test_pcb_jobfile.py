"""Gerber job file reading: the stackup contract of a PCB import.

Everything a board import does in three dimensions rests on this file.
The Gerber drawings say where copper is; only the job file says how
thick it is, how far apart the layers sit, and what the dielectric is —
so the gates here are the ones that keep a board from arriving at the
wrong size or with a silently invented substrate.
"""

from __future__ import annotations

import json

import pytest

from magnelio.io._gbrjob import find_job_file, read_gbrjob


def _job(stackup, *, files=None, project="board"):
    """A job file document with *stackup* as its material stackup."""
    return {
        "Header": {"GenerationSoftware": {"Vendor": "test", "Application": "test"}},
        "GeneralSpecs": {"ProjectId": {"Name": project}, "BoardThickness": 1.6},
        "FilesAttributes": files if files is not None else [],
        "MaterialStackup": stackup,
    }


def _two_layer(**overrides):
    """The stackup of an ordinary two-layer board, 35 µm on 1.53 mm."""
    stack = [
        {"Type": "Legend", "Name": "Top Silk Screen"},
        {"Type": "SolderMask", "Thickness": 0.01, "Name": "Top Solder Mask"},
        {"Type": "Copper", "Thickness": 0.035, "Name": "F.Cu"},
        {
            "Type": "Dielectric",
            "Thickness": 1.53,
            "Material": "FR4",
            "DielectricConstant": "4.5",
            "LossTangent": "0.02",
            "Name": "F.Cu/B.Cu",
        },
        {"Type": "Copper", "Thickness": 0.035, "Name": "B.Cu"},
        {"Type": "SolderMask", "Thickness": 0.01, "Name": "Bottom Solder Mask"},
    ]
    stack[2].update(overrides)
    return stack


def _write(tmp_path, document, name="board.gbrjob"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ── stackup ──────────────────────────────────────────────────────────


def test_layers_are_copper_and_dielectric_in_meters(tmp_path):
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer())))

    assert [layer.name for layer in stackup.layers] == ["F.Cu", "dielectric_1", "B.Cu"]
    assert [layer.kind for layer in stackup.layers] == ["copper", "dielectric", "copper"]
    assert stackup.layers[0].thickness == pytest.approx(35e-6)
    assert stackup.layers[1].thickness == pytest.approx(1.53e-3)
    assert stackup.thickness() == pytest.approx(1.6e-3)
    assert stackup.project == "board"


def test_coatings_carry_no_layer(tmp_path):
    """Mask and legend are in the fabrication stack, not in the model."""
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer())))

    assert all("Mask" not in layer.name and "Silk" not in layer.name for layer in stackup.layers)
    assert len(stackup.layers) == 3


def test_dielectric_properties_survive_as_numbers(tmp_path):
    """A job file writes these as strings; they have to arrive as floats."""
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer())))
    dielectric = stackup.layers[1]

    assert dielectric.epsilon == pytest.approx(4.5)
    assert dielectric.loss_tangent == pytest.approx(0.02)
    assert dielectric.material == "FR4"


def test_copper_layers_are_numbered_from_the_top(tmp_path):
    stack = [
        {"Type": "Copper", "Thickness": 0.035, "Name": "F.Cu"},
        {"Type": "Dielectric", "Thickness": 0.2, "DielectricConstant": 4.5, "Name": "FR4"},
        {"Type": "Copper", "Thickness": 0.018, "Name": "In1.Cu"},
        {"Type": "Dielectric", "Thickness": 1.0, "DielectricConstant": 4.5, "Name": "FR4"},
        {"Type": "Copper", "Thickness": 0.018, "Name": "In2.Cu"},
        {"Type": "Dielectric", "Thickness": 0.2, "DielectricConstant": 4.5, "Name": "FR4"},
        {"Type": "Copper", "Thickness": 0.035, "Name": "B.Cu"},
    ]
    stackup = read_gbrjob(_write(tmp_path, _job(stack)))

    assert [layer.number for layer in stackup.copper_layers] == [1, 2, 3, 4]
    assert [layer.name for layer in stackup.copper_layers] == ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
    # Repeated dielectric names would collide as material keys.
    names = [layer.name for layer in stackup.layers]
    assert len(set(names)) == len(names)


def test_elevations_put_the_substrate_surface_at_zero(tmp_path):
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer())))
    (top_lo, top_hi), (die_lo, die_hi), (bot_lo, bot_hi) = stackup.elevations()

    assert (top_lo, top_hi) == pytest.approx((0.0, 35e-6))
    assert (die_lo, die_hi) == pytest.approx((-1.53e-3, 0.0))
    assert (bot_lo, bot_hi) == pytest.approx((-1.565e-3, -1.53e-3))


def test_layers_touch_without_gap_or_overlap(tmp_path):
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer())))
    elevations = stackup.elevations()

    for (upper_bottom, _), (_, lower_top) in zip(elevations, elevations[1:]):
        assert upper_bottom == pytest.approx(lower_top)


# ── file roles ───────────────────────────────────────────────────────


def test_file_roles_resolve_next_to_the_job_file(tmp_path):
    files = [
        {"Path": "board-F_Cu.gbr", "FileFunction": "Copper,L1,Top"},
        {"Path": "board-B_Cu.gbr", "FileFunction": "Copper,L2,Bot"},
        {"Path": "board-Edge_Cuts.gbr", "FileFunction": "Profile,NP"},
        {"Path": "board-PTH.drl", "FileFunction": "Plated,1,2,PTH"},
        {"Path": "board-NPTH.drl", "FileFunction": "NonPlated,1,2,NPTH"},
        {"Path": "board-F_Mask.gbr", "FileFunction": "Soldermask,Top"},
    ]
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer(), files=files)))

    assert stackup.copper_files == {
        1: tmp_path / "board-F_Cu.gbr",
        2: tmp_path / "board-B_Cu.gbr",
    }
    assert stackup.outline_file == tmp_path / "board-Edge_Cuts.gbr"
    plated, non_plated = stackup.drill_files
    assert (plated.path, plated.plated, plated.span) == (tmp_path / "board-PTH.drl", True, (1, 2))
    assert (non_plated.path, non_plated.plated) == (tmp_path / "board-NPTH.drl", False)


def test_a_blind_via_file_declares_its_span(tmp_path):
    files = [{"Path": "blind.drl", "FileFunction": "Plated,1,2,Blind"}]
    stackup = read_gbrjob(_write(tmp_path, _job(_two_layer(), files=files)))

    assert stackup.drill_files[0].span == (1, 2)


# ── locating the job file ────────────────────────────────────────────


def test_directory_with_one_job_file_is_enough(tmp_path):
    path = _write(tmp_path, _job(_two_layer()))

    assert find_job_file(tmp_path) == path
    assert read_gbrjob(tmp_path).project == "board"


def test_missing_job_file_says_how_to_get_one(tmp_path):
    (tmp_path / "board-F_Cu.gbr").write_text("G04*\nM02*\n")

    with pytest.raises(FileNotFoundError, match="job file"):
        find_job_file(tmp_path)


def test_two_job_files_ask_which_one(tmp_path):
    _write(tmp_path, _job(_two_layer()), name="a.gbrjob")
    _write(tmp_path, _job(_two_layer()), name="b.gbrjob")

    with pytest.raises(ValueError, match="more than one job file"):
        find_job_file(tmp_path)


def test_absent_path_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        find_job_file(tmp_path / "nowhere")


# ── refusals ─────────────────────────────────────────────────────────


def test_missing_thickness_is_an_error_naming_the_layer(tmp_path):
    stack = _two_layer()
    del stack[2]["Thickness"]
    path = _write(tmp_path, _job(stack))

    with pytest.raises(ValueError, match=r"no thickness for copper layer 'F\.Cu'"):
        read_gbrjob(path)


def test_zero_thickness_is_an_error(tmp_path):
    path = _write(tmp_path, _job(_two_layer(Thickness=0.0)))

    with pytest.raises(ValueError, match="positive thickness"):
        read_gbrjob(path)


def test_stackup_free_job_file_is_an_error(tmp_path):
    document = _job(_two_layer())
    del document["MaterialStackup"]
    path = _write(tmp_path, document)

    with pytest.raises(ValueError, match="no material stackup"):
        read_gbrjob(path)


def test_unknown_layer_type_is_an_error(tmp_path):
    stack = _two_layer()
    stack[2]["Type"] = "Unobtainium"
    path = _write(tmp_path, _job(stack))

    with pytest.raises(ValueError, match="unknown type 'Unobtainium'"):
        read_gbrjob(path)


def test_broken_json_reports_where(tmp_path):
    path = tmp_path / "board.gbrjob"
    path.write_text('{"MaterialStackup": [', encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_gbrjob(path)


def test_non_numeric_dielectric_constant_is_an_error(tmp_path):
    stack = _two_layer()
    stack[3]["DielectricConstant"] = "not a number"
    path = _write(tmp_path, _job(stack))

    with pytest.raises(ValueError, match="DielectricConstant"):
        read_gbrjob(path)
