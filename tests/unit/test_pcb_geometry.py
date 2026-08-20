"""Board records into kernel geometry: areas, nesting and the outline.

Every gate here measures an area or a volume against a number worked
out by hand, because that is the only way to catch the failure mode
this stage really has: a Boolean that drops a sliver, a contour read
with the wrong nesting or an arc built the short way round all produce
a shape that looks like a board and is not the one in the file.
"""

from __future__ import annotations

import math

import pytest

from magnelio.io import _pcb_geom as geom
from magnelio.io._gerber import parse_gerber

pytest.importorskip("OCC", reason="board geometry requires pythonocc-core")

MM = 1e-3
# Construction scale: a board's features sit four decades below its
# size, so the kernel works on a magnified copy (see fine_detail_scale).
SCALE = 1024.0


def _gerber(*body, fmt="LAX46Y46"):
    return "\n".join([f"%FS{fmt}*%", "%MOMM*%", *body, "M02*"]) + "\n"


def _area(shape) -> float:
    """Face area of a shape, back in square meters."""
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape, props)
    return props.Mass() / SCALE**2


def _volume(shape) -> float:
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass() / SCALE**3


def _copper(*body, fmt="LAX46Y46"):
    return geom.layer_shape(parse_gerber(_gerber(*body, fmt=fmt)), SCALE)


# ── apertures ────────────────────────────────────────────────────────


def test_a_round_pad_covers_its_circle():
    assert _area(_copper("%ADD10C,1.0*%", "D10*", "X0Y0D03*")) == pytest.approx(math.pi / 4 * MM**2)


def test_a_pad_with_a_hole_is_a_ring():
    shape = _copper("%ADD10C,1.0X0.4*%", "D10*", "X0Y0D03*")

    assert _area(shape) == pytest.approx(math.pi / 4 * (1.0 - 0.16) * MM**2)


def test_a_rectangular_pad_covers_its_rectangle():
    assert _area(_copper("%ADD10R,2.0X1.0*%", "D10*", "X0Y0D03*")) == pytest.approx(2.0 * MM**2)


def test_an_obround_pad_is_a_rectangle_with_round_ends():
    shape = _copper("%ADD10O,2.0X1.0*%", "D10*", "X0Y0D03*")

    assert _area(shape) == pytest.approx((1.0 * 1.0 + math.pi / 4 * 1.0) * MM**2)


def test_a_polygon_pad_covers_its_polygon():
    shape = _copper("%ADD10P,2.0X6X0*%", "D10*", "X0Y0D03*")
    hexagon = 6 * 0.5 * 1.0**2 * math.sin(math.pi / 3)

    assert _area(shape) == pytest.approx(hexagon * MM**2)


def test_a_macro_pad_adds_and_subtracts_its_primitives():
    """A ring built as a circle with a cleared circle inside it."""
    shape = _copper(
        "%AMRing*", "1,1,2.0,0,0*", "1,0,1.0,0,0*", "%", "%ADD10Ring*%", "D10*", "X0Y0D03*"
    )

    assert _area(shape) == pytest.approx(math.pi / 4 * (4.0 - 1.0) * MM**2)


def test_a_macro_pad_rotates_about_the_macro_origin():
    """The rotation moves an off-centre primitive, it does not spin it."""
    shape = _copper("%AMOff*", "1,1,1.0,2.0,0,90*", "%", "%ADD10Off*%", "D10*", "X0Y0D03*")
    (lo, hi) = _bounds(shape)

    assert lo == pytest.approx((-0.5 * MM, 1.5 * MM), abs=1e-9)
    assert hi == pytest.approx((0.5 * MM, 2.5 * MM), abs=1e-9)


def _bounds(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box, False)
    box.SetGap(0.0)
    x_min, y_min, _, x_max, y_max, _ = box.Get()
    return (x_min / SCALE, y_min / SCALE), (x_max / SCALE, y_max / SCALE)


# ── tracks ───────────────────────────────────────────────────────────


def test_a_track_is_a_rectangle_with_round_ends():
    shape = _copper("%ADD10C,0.2*%", "D10*", "X0Y0D02*", "X1000000Y0D01*")
    expected = 1.0 * 0.2 + math.pi / 4 * 0.2**2

    assert _area(shape) == pytest.approx(expected * MM**2)


def test_an_arc_track_sweeps_the_annulus_it_runs_through():
    shape = _copper(
        "%ADD10C,0.2*%",
        "D10*",
        "G75*",
        "G03*",
        "X5000000Y0D02*",
        "X0Y5000000I-5000000J0D01*",
    )
    expected = (math.pi / 2 * 5.0) * 0.2 + math.pi / 4 * 0.2**2

    assert _area(shape) == pytest.approx(expected * MM**2)


def test_a_full_circle_arc_is_an_annulus():
    shape = _copper(
        "%ADD10C,0.2*%", "D10*", "G75*", "G03*", "X5000000Y0D02*", "X5000000Y0I-5000000J0D01*"
    )
    expected = math.pi / 4 * (10.2**2 - 9.8**2)

    assert _area(shape) == pytest.approx(expected * MM**2)


def test_an_arc_tighter_than_the_track_is_refused():
    """Its outline would fold through itself; no reading of it is right."""
    with pytest.raises(ValueError, match="covers its own centre"):
        _copper(
            "%ADD10C,2.0*%",
            "D10*",
            "G75*",
            "G03*",
            "X500000Y0D02*",
            "X0Y500000I-500000J0D01*",
        )


# ── merging a layer ──────────────────────────────────────────────────


def test_overlapping_pads_merge_into_one_area():
    shape = _copper("%ADD10C,1.0*%", "D10*", "X0Y0D03*", "X500000Y0D03*")
    lens = 2 * 0.25 * math.acos(0.5) - 0.25 * math.sqrt(0.75)
    expected = 2 * math.pi / 4 - lens

    assert _area(shape) == pytest.approx(expected * MM**2)


def test_disjoint_pads_keep_their_full_area():
    shape = _copper("%ADD10C,1.0*%", "D10*", "X0Y0D03*", "X9000000Y0D03*")

    assert _area(shape) == pytest.approx(2 * math.pi / 4 * MM**2)


def test_disjoint_pads_are_not_put_through_a_boolean(monkeypatch):
    """The Boolean is the cost of a copper layer; isolated pads skip it."""
    calls = []
    original = geom.boolean_union
    monkeypatch.setattr(geom, "boolean_union", lambda s: calls.append(len(s)) or original(s))

    _copper("%ADD10C,1.0*%", "D10*", "X0Y0D03*", "X9000000Y0D03*", "X18000000Y0D03*")

    assert calls == []


def test_clear_polarity_removes_what_was_drawn_before_it():
    shape = _copper(
        "%ADD10C,2.0*%", "%ADD11C,1.0*%", "D10*", "X0Y0D03*", "%LPC*%", "D11*", "X0Y0D03*"
    )

    assert _area(shape) == pytest.approx(math.pi / 4 * (4.0 - 1.0) * MM**2)


def test_clear_polarity_does_not_remove_what_comes_after_it():
    shape = _copper(
        "%ADD10C,2.0*%",
        "%ADD11C,1.0*%",
        "D10*",
        "X0Y0D03*",
        "%LPC*%",
        "D11*",
        "X0Y0D03*",
        "%LPD*%",
        "D11*",
        "X0Y0D03*",
    )

    assert _area(shape) == pytest.approx(math.pi / 4 * 4.0 * MM**2)


# ── regions ──────────────────────────────────────────────────────────


def test_a_region_covers_the_area_it_encloses():
    shape = _copper(
        "G36*", "X0Y0D02*", "X2000000Y0D01*", "X2000000Y1000000D01*", "X0Y1000000D01*", "G37*"
    )

    assert _area(shape) == pytest.approx(2.0 * MM**2)


def test_a_contour_inside_another_is_a_hole():
    shape = _copper(
        "G36*",
        "X0Y0D02*",
        "X4000000Y0D01*",
        "X4000000Y4000000D01*",
        "X0Y4000000D01*",
        "X0Y0D01*",
        "X1000000Y1000000D02*",
        "X1000000Y3000000D01*",
        "X3000000Y3000000D01*",
        "X3000000Y1000000D01*",
        "X1000000Y1000000D01*",
        "G37*",
    )

    assert _area(shape) == pytest.approx((16.0 - 4.0) * MM**2)


def test_a_region_contour_may_mix_lines_and_arcs():
    shape = _copper(
        "G75*",
        "G36*",
        "X0Y0D02*",
        "G01*",
        "X2000000Y0D01*",
        "G03*",
        "X2000000Y2000000I0J1000000D01*",
        "G01*",
        "X0Y2000000D01*",
        "X0Y0D01*",
        "G37*",
    )
    expected = 2.0 * 2.0 + math.pi / 2 * 1.0**2

    assert _area(shape) == pytest.approx(expected * MM**2)


# ── the board outline ────────────────────────────────────────────────


def _outline(*body, fmt="LAX46Y46"):
    return geom.outline_faces(parse_gerber(_gerber(*body, fmt=fmt)), SCALE)


_RECTANGLE = (
    "%ADD10C,0.1*%",
    "D10*",
    "X0Y0D02*",
    "X10000000Y0D01*",
    "X10000000Y8000000D01*",
    "X0Y8000000D01*",
    "X0Y0D01*",
)


def test_the_outline_becomes_the_area_it_encloses():
    """The profile is drawn as a line; the board is what it encloses."""
    (face,) = _outline(*_RECTANGLE)

    assert _area(face) == pytest.approx(80.0 * MM**2)


def test_the_outline_segments_may_arrive_in_any_order():
    scrambled = (
        "%ADD10C,0.1*%",
        "D10*",
        "X0Y8000000D02*",
        "X0Y0D01*",
        "X10000000Y8000000D02*",
        "X0Y8000000D01*",
        "X0Y0D02*",
        "X10000000Y0D01*",
        "X10000000Y0D02*",
        "X10000000Y8000000D01*",
    )
    (face,) = _outline(*scrambled)

    assert _area(face) == pytest.approx(80.0 * MM**2)


def test_a_loop_inside_the_outline_is_a_cut_out():
    cutout = (
        "X2000000Y2000000D02*",
        "X4000000Y2000000D01*",
        "X4000000Y4000000D01*",
        "X2000000Y4000000D01*",
        "X2000000Y2000000D01*",
    )
    (face,) = _outline(*_RECTANGLE, *cutout)

    assert _area(face) == pytest.approx((80.0 - 4.0) * MM**2)


def test_two_separate_outlines_are_two_boards():
    second = (
        "X20000000Y0D02*",
        "X24000000Y0D01*",
        "X24000000Y4000000D01*",
        "X20000000Y4000000D01*",
        "X20000000Y0D01*",
    )
    faces = _outline(*_RECTANGLE, *second)

    assert len(faces) == 2
    assert sum(_area(face) for face in faces) == pytest.approx((80.0 + 16.0) * MM**2)


def test_an_outline_with_a_rounded_corner_keeps_its_arc():
    rounded = (
        "%ADD10C,0.1*%",
        "D10*",
        "G75*",
        "X0Y0D02*",
        "X9000000Y0D01*",
        "G03*",
        "X10000000Y1000000I0J1000000D01*",
        "G01*",
        "X10000000Y8000000D01*",
        "X0Y8000000D01*",
        "X0Y0D01*",
    )
    (face,) = _outline(*rounded)
    expected = 80.0 - (1.0 - math.pi / 4)

    assert _area(face) == pytest.approx(expected * MM**2)


def test_an_open_outline_is_an_error():
    """Chaining silently past a loose end would give a plausible board."""
    open_edge = _RECTANGLE[:-1]

    with pytest.raises(ValueError, match="not a set of closed loops"):
        _outline(*open_edge)


def test_a_branching_outline_is_an_error():
    branch = (*_RECTANGLE, "X0Y0D02*", "X-2000000Y0D01*")

    with pytest.raises(ValueError, match="must not branch"):
        _outline(*branch)


def test_an_empty_outline_layer_is_an_error():
    with pytest.raises(ValueError, match="draws nothing"):
        _outline("%ADD10C,0.1*%", "D10*")


# ── from faces to solids ─────────────────────────────────────────────


def test_a_layer_is_extruded_to_the_thickness_it_is_given():
    (face,) = _outline(*_RECTANGLE)
    solid = geom.extrude(face, -1.53 * MM, 0.0, SCALE)

    assert _volume(solid) == pytest.approx(80.0 * 1.53 * MM**3)
    (lo, hi) = _z_extent(solid)
    assert (lo, hi) == pytest.approx((-1.53 * MM, 0.0))


def _z_extent(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box, False)
    box.SetGap(0.0)
    _, _, z_min, _, _, z_max = box.Get()
    return z_min / SCALE, z_max / SCALE


def test_a_drilled_hole_is_removed_from_the_layer():
    (face,) = _outline(*_RECTANGLE)
    holes = [geom.circle_face((5.0 * MM, 4.0 * MM), 0.8 * MM, SCALE)]
    solid = geom.extrude(geom.cut(face, holes), -1.53 * MM, 0.0, SCALE)
    expected = (80.0 - math.pi / 4 * 0.64) * 1.53

    assert _volume(solid) == pytest.approx(expected * MM**3)


def test_a_slot_is_removed_along_its_whole_length():
    (face,) = _outline(*_RECTANGLE)
    slot = geom.slot_face((4.0 * MM, 4.0 * MM), (6.0 * MM, 4.0 * MM), 1.0 * MM, SCALE)
    solid = geom.extrude(geom.cut(face, [slot]), -1.0 * MM, 0.0, SCALE)
    expected = (80.0 - (2.0 * 1.0 + math.pi / 4)) * 1.0

    assert _volume(solid) == pytest.approx(expected * MM**3)


def test_copper_is_clipped_to_the_board():
    """A track drawn over the edge must not leave metal in mid-air."""
    (board,) = _outline(*_RECTANGLE)
    copper = _copper("%ADD10C,1.0*%", "D10*", "X-500000Y4000000D02*", "X500000Y4000000D01*")
    clipped = geom.clip(copper, board)

    assert _area(clipped) < _area(copper)
    assert _area(clipped) == pytest.approx((0.5 * 1.0 + math.pi / 8) * MM**2)
