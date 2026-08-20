"""Excellon reading: which holes exist, how wide, and plated or not.

Two properties of a drill file change the physics of the imported
board, and neither can be recovered later: the plating, which decides
whether a hole is a copper barrel joining layers or absent material,
and the layer span, which decides how deep that barrel goes.  The rest
of the gates guard the format's number handling, where the same digits
mean different lengths depending on a keyword in the header.
"""

from __future__ import annotations

import pytest

from magnelio.io._excellon import Hit, Slot, parse_excellon

MM = 1e-3

_KICAD_PTH = """M48
;DRILL file {KiCad} date Thu 01 Jan 2026
;FORMAT={-:-/ absolute / metric / decimal}
; #@! TF.CreationDate,2026-01-01T00:00:00+01:00
; #@! TF.FileFunction,Plated,1,2,PTH
; #@! TF.FilePolarity,Positive
FMAT,2
METRIC
; #@! TA.AperFunction,Plated,PTH,ViaDrill
T1C0.300
; #@! TA.AperFunction,Plated,PTH,ComponentDrill
T2C0.800
%
G90
G05
T1
X10.0Y-20.0
X12.5Y-20.0
T2
X5.0Y-5.0
T0
M30
"""


# ── the ordinary case ────────────────────────────────────────────────


def test_hits_carry_the_diameter_of_their_tool():
    drill = parse_excellon(_KICAD_PTH, source="board-PTH.drl")

    assert len(drill.holes) == 3
    assert all(isinstance(hole, Hit) for hole in drill.holes)
    assert [hole.diameter for hole in drill.holes] == pytest.approx([0.3 * MM, 0.3 * MM, 0.8 * MM])
    assert drill.holes[0].at == pytest.approx((10.0 * MM, -20.0 * MM))


def test_plating_and_span_come_from_the_file_attribute():
    drill = parse_excellon(_KICAD_PTH)

    assert drill.plated is True
    assert drill.span == (1, 2)


def test_a_non_plated_file_says_so():
    text = _KICAD_PTH.replace(
        "TF.FileFunction,Plated,1,2,PTH", "TF.FileFunction,NonPlated,1,2,NPTH"
    )

    assert parse_excellon(text).plated is False


def test_a_file_without_the_attribute_claims_nothing():
    """Guessing here would turn a via into a hole, or the reverse."""
    text = "\n".join(line for line in _KICAD_PTH.splitlines() if "TF.FileFunction" not in line)
    drill = parse_excellon(text)

    assert drill.plated is None
    assert drill.span is None


def test_a_blind_via_file_spans_two_inner_layers():
    text = _KICAD_PTH.replace("TF.FileFunction,Plated,1,2,PTH", "TF.FileFunction,Plated,2,3,Buried")

    assert parse_excellon(text).span == (2, 3)


# ── numbers ──────────────────────────────────────────────────────────


def _drill(*body, unit="METRIC"):
    return "\n".join(["M48", "FMAT,2", unit, "T1C1.0", "%", "G90", "G05", "T1", *body, "M30"])


def test_leading_and_trailing_suppression_are_named_the_other_way_round():
    """``LZ`` says the leading zeros are *present*, so the field pads right."""
    trailing_present = parse_excellon(_drill("X010000Y-005000", unit="METRIC,TZ"))
    leading_present = parse_excellon(_drill("X01Y-005", unit="METRIC,LZ"))

    assert trailing_present.holes[0].at == pytest.approx((10.0 * MM, -5.0 * MM))
    assert leading_present.holes[0].at == pytest.approx((10.0 * MM, -5.0 * MM))


def test_a_decimal_point_overrides_any_suppression_rule():
    drill = parse_excellon(_drill("X10.0Y-5.0", unit="METRIC,TZ"))

    assert drill.holes[0].at == pytest.approx((10.0 * MM, -5.0 * MM))


def test_inch_files_default_to_two_by_four_digits():
    drill = parse_excellon(_drill("X010000Y000000", unit="INCH,TZ"))

    assert drill.holes[0].at[0] == pytest.approx(1.0 * 0.0254)
    assert drill.holes[0].diameter == pytest.approx(1.0 * 0.0254)


def test_an_explicit_format_is_honoured():
    drill = parse_excellon(_drill("X01000000", unit="METRIC,TZ,0000.0000"))

    assert drill.holes[0].at[0] == pytest.approx(100.0 * MM)


def test_coordinates_are_modal():
    drill = parse_excellon(_drill("X10.0Y-5.0", "X12.0"))

    assert drill.holes[1].at == pytest.approx((12.0 * MM, -5.0 * MM))


# ── slots ────────────────────────────────────────────────────────────


def test_a_g85_slot_runs_between_its_two_points():
    drill = parse_excellon(_drill("X1.0Y2.0G85X3.0Y2.0"))
    (slot,) = drill.holes

    assert isinstance(slot, Slot)
    assert slot.start == pytest.approx((1.0 * MM, 2.0 * MM))
    assert slot.end == pytest.approx((3.0 * MM, 2.0 * MM))
    assert slot.diameter == pytest.approx(1.0 * MM)
    assert slot.at == pytest.approx((2.0 * MM, 2.0 * MM))


def test_a_routed_slot_is_the_same_slot_spelled_differently():
    """Writers pick either spelling for an oval hole; both must arrive."""
    drill = parse_excellon(_drill("G00X1.0Y2.0", "M15", "G01X3.0Y2.0", "M16", "G05"))
    (slot,) = drill.holes

    assert isinstance(slot, Slot)
    assert slot.start == pytest.approx((1.0 * MM, 2.0 * MM))
    assert slot.end == pytest.approx((3.0 * MM, 2.0 * MM))


def test_a_routing_move_with_the_tool_up_cuts_nothing():
    drill = parse_excellon(_drill("G00X1.0Y2.0", "G00X9.0Y9.0", "G05"))

    assert drill.holes == ()


# ── refusals ─────────────────────────────────────────────────────────


def test_a_hole_before_any_tool_is_an_error():
    text = "\n".join(["M48", "FMAT,2", "METRIC", "T1C1.0", "%", "G90", "X1.0Y1.0", "M30"])

    with pytest.raises(ValueError, match="before any tool was selected"):
        parse_excellon(text)


def test_a_tool_without_a_diameter_is_an_error():
    text = "\n".join(["M48", "FMAT,2", "METRIC", "%", "G90", "T1", "X1.0Y1.0", "M30"])

    with pytest.raises(ValueError, match="never given a diameter"):
        parse_excellon(text)


def test_format_one_is_refused():
    with pytest.raises(ValueError, match="format 2"):
        parse_excellon("M48\nFMAT,1\nMETRIC\n%\nM30\n")


def test_circular_routing_is_refused():
    with pytest.raises(ValueError, match="circular routed slot"):
        parse_excellon(_drill("G00X1.0Y1.0", "M15", "G02X2.0Y2.0I1.0J0"))


def test_incremental_coordinates_are_refused():
    with pytest.raises(ValueError, match="[Ii]ncremental"):
        parse_excellon(_drill("ICI,ON"))


def test_a_file_without_a_unit_is_an_error():
    with pytest.raises(ValueError, match="no unit"):
        parse_excellon("M48\nFMAT,2\n%\nM30\n")


def test_errors_name_the_line():
    with pytest.raises(ValueError, match=r"board\.drl, line 9"):
        parse_excellon(_drill("Q17"), source="board.drl")


def test_everything_after_the_end_of_program_is_ignored():
    drill = parse_excellon(_drill("X1.0Y1.0") + "\nX99.0Y99.0\n")

    assert len(drill.holes) == 1
