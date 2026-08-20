"""Gerber reading: the format traps that silently change a board.

Gerber is a plotting language whose defaults are hostile — coordinates
without a decimal point, a zero-suppression flag that shifts everything
by decades, apertures defined by little programs.  Every gate here
guards a way a layout could arrive looking plausible and being wrong,
so the fixtures are hand-written files whose copper is known by
construction.
"""

from __future__ import annotations

import pytest

from magnelio.io._gerber import (
    ArcStroke,
    Circle,
    Flash,
    MacroAperture,
    MacroCircle,
    MacroOutline,
    Obround,
    Rect,
    Region,
    RegularPolygon,
    Stroke,
    parse_gerber,
)

MM = 1e-3


def _gerber(*body, unit="MM", fmt="LAX46Y46"):
    """A minimal well-formed file wrapped around *body*."""
    return "\n".join([f"%FS{fmt}*%", f"%MO{unit}*%", *body, "M02*"]) + "\n"


def _only(layer):
    assert len(layer.objects) == 1
    dark, obj = layer.objects[0]
    assert dark
    return obj


# ── coordinates ──────────────────────────────────────────────────────


def test_coordinates_arrive_in_meters(tmp_path):
    layer = parse_gerber(_gerber("%ADD10C,0.5*%", "D10*", "X1000000Y-2500000D03*"))
    flash = _only(layer)

    assert isinstance(flash, Flash)
    assert flash.at == pytest.approx((1.0 * MM, -2.5 * MM))
    assert flash.aperture == Circle(diameter=0.5 * MM)


def test_inch_files_are_converted(tmp_path):
    layer = parse_gerber(_gerber("%ADD10C,0.1*%", "D10*", "X1000000Y0000000D03*", unit="IN"))
    flash = _only(layer)

    assert flash.at[0] == pytest.approx(1.0 * 0.0254)
    assert flash.aperture.diameter == pytest.approx(0.1 * 0.0254)


def test_trailing_zero_suppression_pads_to_the_right(tmp_path):
    """``X15`` in a 2.4 format is 15.0000, not 0.0015 — and ``Y2`` is 20."""
    layer = parse_gerber(_gerber("%ADD10C,0.5*%", "D10*", "X15Y2D03*", fmt="TAX24Y24"))

    assert _only(layer).at == pytest.approx((15.0 * MM, 20.0 * MM))


def test_coordinates_are_modal(tmp_path):
    layer = parse_gerber(
        _gerber("%ADD10C,0.5*%", "D10*", "X1000000Y1000000D03*", "X2000000D03*", "Y3000000D03*")
    )
    points = [obj.at for _, obj in layer.objects]

    assert points == pytest.approx([(1 * MM, 1 * MM), (2 * MM, 1 * MM), (2 * MM, 3 * MM)])


def test_incremental_coordinates_are_refused(tmp_path):
    with pytest.raises(ValueError, match="[Ii]ncremental"):
        parse_gerber(_gerber("D02*", fmt="LIX46Y46"))


# ── apertures ────────────────────────────────────────────────────────


def test_standard_apertures_and_their_holes(tmp_path):
    layer = parse_gerber(
        _gerber(
            "%ADD10C,0.6X0.3*%",
            "%ADD11R,1.0X2.0*%",
            "%ADD12O,1.0X2.0X0.4*%",
            "%ADD13P,1.2X6X30*%",
            "D10*",
            "X0Y0D03*",
            "D11*",
            "X0Y0D03*",
            "D12*",
            "X0Y0D03*",
            "D13*",
            "X0Y0D03*",
        )
    )
    apertures = [obj.aperture for _, obj in layer.objects]

    assert apertures[0] == Circle(diameter=0.6 * MM, hole=0.3 * MM)
    assert apertures[1] == Rect(width=1.0 * MM, height=2.0 * MM, hole=None)
    assert apertures[2] == Obround(width=1.0 * MM, height=2.0 * MM, hole=0.4 * MM)
    assert apertures[3] == RegularPolygon(diameter=1.2 * MM, vertices=6, rotation=30.0, hole=None)


def test_selecting_an_undefined_aperture_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="D10 is selected but never defined"):
        parse_gerber(_gerber("D10*", "X0Y0D03*"))


def test_rectangular_hole_is_refused(tmp_path):
    with pytest.raises(ValueError, match="rectangular hole"):
        parse_gerber(_gerber("%ADD10C,0.6X0.3X0.3*%"))


# ── tracks ───────────────────────────────────────────────────────────


def test_a_draw_is_a_stroke_of_the_aperture_width(tmp_path):
    layer = parse_gerber(_gerber("%ADD10C,0.25*%", "D10*", "X0Y0D02*", "X1000000Y0D01*"))
    stroke = _only(layer)

    assert isinstance(stroke, Stroke)
    assert stroke.start == pytest.approx((0.0, 0.0))
    assert stroke.end == pytest.approx((1.0 * MM, 0.0))
    assert stroke.width == pytest.approx(0.25 * MM)


def test_drawing_with_a_rectangle_is_refused(tmp_path):
    """A rectangular pen sweeps a shape no tool agrees on; it is deprecated."""
    with pytest.raises(ValueError, match="drawing a track"):
        parse_gerber(_gerber("%ADD10R,1.0X2.0*%", "D10*", "X0Y0D02*", "X1000000Y0D01*"))


def test_a_draw_before_a_move_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="before the current point"):
        parse_gerber(_gerber("%ADD10C,0.25*%", "D10*", "X1000000Y0D01*"))


def test_arcs_take_their_centre_from_the_offset(tmp_path):
    layer = parse_gerber(
        _gerber(
            "%ADD10C,0.25*%",
            "D10*",
            "G75*",
            "G03*",
            "X1000000Y0D02*",
            "X0Y1000000I-1000000J0D01*",
        )
    )
    arc = _only(layer)

    assert isinstance(arc, ArcStroke)
    assert arc.center == pytest.approx((0.0, 0.0))
    assert arc.clockwise is False
    assert arc.end == pytest.approx((0.0, 1.0 * MM))


def test_single_quadrant_arcs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="single quadrant"):
        parse_gerber(_gerber("G74*"))


def test_an_arc_without_multi_quadrant_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="multi quadrant"):
        parse_gerber(_gerber("%ADD10C,0.25*%", "D10*", "G03*", "X0Y0D02*", "X0Y0I100J0D01*"))


# ── regions ──────────────────────────────────────────────────────────


def test_a_region_becomes_one_closed_contour(tmp_path):
    layer = parse_gerber(
        _gerber(
            "G36*",
            "X0Y0D02*",
            "X2000000Y0D01*",
            "X2000000Y1000000D01*",
            "X0Y1000000D01*",
            "X0Y0D01*",
            "G37*",
        )
    )
    region = _only(layer)

    assert isinstance(region, Region)
    assert len(region.contours) == 1
    assert len(region.contours[0]) == 4
    assert region.contours[0][0].start == pytest.approx((0.0, 0.0))
    assert region.contours[0][-1].end == pytest.approx((0.0, 0.0))


def test_an_unclosed_region_contour_is_closed(tmp_path):
    """Writers disagree over the closing segment; the area is the same."""
    layer = parse_gerber(
        _gerber(
            "G36*",
            "X0Y0D02*",
            "X2000000Y0D01*",
            "X2000000Y1000000D01*",
            "X0Y1000000D01*",
            "G37*",
        )
    )
    contour = _only(layer).contours[0]

    assert contour[-1].end == pytest.approx(contour[0].start)


def test_a_move_inside_a_region_starts_a_new_contour(tmp_path):
    layer = parse_gerber(
        _gerber(
            "G36*",
            "X0Y0D02*",
            "X2000000Y0D01*",
            "X0Y2000000D01*",
            "X0Y0D01*",
            "X3000000Y3000000D02*",
            "X4000000Y3000000D01*",
            "X3000000Y4000000D01*",
            "X3000000Y3000000D01*",
            "G37*",
        )
    )

    assert len(_only(layer).contours) == 2


def test_region_needs_no_aperture(tmp_path):
    """The aperture is irrelevant inside G36; requiring one would reject
    perfectly good zone-only layers."""
    layer = parse_gerber(_gerber("G36*", "X0Y0D02*", "X1000000Y0D01*", "X0Y1000000D01*", "G37*"))

    assert isinstance(_only(layer), Region)


def test_an_unclosed_region_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="never closed"):
        parse_gerber(_gerber("G36*", "X0Y0D02*", "X1000000Y0D01*"))


# ── polarity ─────────────────────────────────────────────────────────


def test_clear_polarity_is_kept_in_order(tmp_path):
    layer = parse_gerber(
        _gerber(
            "%ADD10C,2.0*%",
            "D10*",
            "X0Y0D03*",
            "%LPC*%",
            "X0Y0D03*",
            "%LPD*%",
            "X0Y0D03*",
        )
    )

    assert [dark for dark, _ in layer.objects] == [True, False, True]


# ── aperture macros ──────────────────────────────────────────────────

_ROUNDRECT = [
    "%AMRoundRect*",
    "0 Rectangle with rounded corners*",
    "$5=$1x2*",
    "4,1,4,-1.0,-0.5,1.0,-0.5,1.0,0.5,-1.0,0.5,-1.0,-0.5,0*",
    "1,1,$5,-1.0,-0.5*",
    "1,1,$5,1.0,-0.5*",
    "1,1,$5,1.0,0.5*",
    "1,1,$5,-1.0,0.5*",
    "%",
]


def test_a_macro_aperture_is_evaluated_at_definition(tmp_path):
    layer = parse_gerber(_gerber(*_ROUNDRECT, "%ADD10RoundRect,0.25*%", "D10*", "X0Y0D03*"))
    aperture = _only(layer).aperture

    assert isinstance(aperture, MacroAperture)
    assert aperture.name == "RoundRect"
    outline, *corners = aperture.primitives
    assert isinstance(outline, MacroOutline)
    assert len(outline.points) == 5
    assert outline.points[0] == pytest.approx((-1.0 * MM, -0.5 * MM))
    # $5 = $1 x 2, with $1 = 0.25 handed over by the AD command.
    assert all(isinstance(c, MacroCircle) for c in corners)
    assert corners[0].diameter == pytest.approx(0.5 * MM)


def test_macro_arithmetic_follows_the_format_not_python(tmp_path):
    """``x`` multiplies, and precedence is the usual one."""
    body = ["%AMBox*", "21,1,$1+$2x2,$1,0,0,0*", "%"]
    layer = parse_gerber(_gerber(*body, "%ADD10Box,1.0X0.5*%", "D10*", "X0Y0D03*"))
    (line,) = _only(layer).aperture.primitives

    assert line.width == pytest.approx(2.0 * MM)
    assert line.height == pytest.approx(1.0 * MM)


def test_a_clear_macro_primitive_keeps_its_exposure(tmp_path):
    body = ["%AMRing*", "1,1,2.0,0,0*", "1,0,1.0,0,0*", "%"]
    layer = parse_gerber(_gerber(*body, "%ADD10Ring*%", "D10*", "X0Y0D03*"))
    outer, inner = _only(layer).aperture.primitives

    assert outer.exposure is True
    assert inner.exposure is False


def test_unsupported_macro_primitives_say_which(tmp_path):
    body = ["%AMTherm*", "7,0,0,2.0,1.0,0.3,0*", "%"]

    with pytest.raises(ValueError, match="thermal primitive"):
        parse_gerber(_gerber(*body, "%ADD10Therm*%"))


def test_an_undefined_macro_template_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="undefined template 'Nope'"):
        parse_gerber(_gerber("%ADD10Nope,1.0*%"))


# ── attributes and refusals ──────────────────────────────────────────


def test_file_attributes_are_kept(tmp_path):
    layer = parse_gerber(_gerber("%TF.FileFunction,Copper,L1,Top,Signal*%", "%TF.Part,Single*%"))

    assert layer.file_function == ("Copper", "L1", "Top", "Signal")
    assert layer.attributes[".Part"] == ("Single",)


def test_object_attributes_are_tolerated(tmp_path):
    layer = parse_gerber(_gerber("%TO.N,GND*%", "%ADD10C,0.5*%", "D10*", "X0Y0D03*", "%TD*%"))

    assert isinstance(_only(layer), Flash)


def test_step_and_repeat_is_refused(tmp_path):
    with pytest.raises(ValueError, match="step-and-repeat"):
        parse_gerber(_gerber("%SRX2Y3I5.0J4.0*%"))


def test_negative_images_are_refused(tmp_path):
    with pytest.raises(ValueError, match="negative image"):
        parse_gerber(_gerber("%IPNEG*%"))


def test_errors_name_the_line(tmp_path):
    with pytest.raises(ValueError, match="board.gbr, line 4"):
        parse_gerber(_gerber("%ADD10C,0.5*%", "Q17*"), source="board.gbr")


def test_a_file_without_a_unit_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no unit"):
        parse_gerber("%FSLAX46Y46*%\nM02*\n")


def test_everything_after_the_end_of_file_is_ignored(tmp_path):
    layer = parse_gerber(_gerber("%ADD10C,0.5*%", "D10*", "X0Y0D03*") + "garbage without a star")

    assert len(layer.objects) == 1


def test_comments_and_deprecated_prefixes_are_tolerated(tmp_path):
    layer = parse_gerber(
        _gerber("G04 This is a comment*", "%ADD10C,0.5*%", "G54D10*", "G01X0Y0D03*")
    )

    assert isinstance(_only(layer), Flash)


def test_the_coordinate_resolution_is_reported(tmp_path):
    """Chaining loose segments needs a node tolerance the file defines."""
    layer = parse_gerber(_gerber("%ADD10C,0.5*%", "D10*", "X0Y0D03*"))

    assert layer.resolution == pytest.approx(1e-6 * MM)
    assert parse_gerber(_gerber("%ADD10C,1*%", fmt="LAX24Y24")).resolution == pytest.approx(
        1e-4 * MM
    )


def test_aperture_numbers_are_not_capped_at_three_digits(tmp_path):
    """A dense board runs past D999; the code is a number, not a field."""
    layer = parse_gerber(_gerber("%ADD1234C,0.5*%", "D1234*", "X0Y0D03*"))

    assert _only(layer).aperture == Circle(diameter=0.5 * MM)
