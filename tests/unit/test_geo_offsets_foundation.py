"""Independent geometry checks for planar and curved sheet offsets."""

import math
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo

pytest.importorskip("OCC.Core.BRepOffsetAPI")


def _ring(outer=2.0, inner=1.0):
    return geo.Profile.from_wires(
        geo.Curve.circle((0, 0, 0), outer),
        [geo.Curve.circle((0, 0, 0), inner)],
        material="pec",
    )


def test_directed_line_offsets_have_opposite_sides_and_uncapped_ends():
    source = geo.Curve.line((0, 0, 0), (2, 0, 0))
    (left,) = source.offset(0.2, normal="z")
    (right,) = source.offset(-0.2, normal="z")
    assert left.bounding_box()[0] == pytest.approx((0, 0.2, 0))
    assert left.bounding_box()[1] == pytest.approx((2, 0.2, 0))
    assert right.bounding_box()[0] == pytest.approx((0, -0.2, 0))
    assert right.bounding_box()[1] == pytest.approx((2, -0.2, 0))
    assert left.length == pytest.approx(2)
    assert right.length == pytest.approx(2)
    assert source.bounding_box()[0] == (0, 0, 0)


def test_open_corner_has_round_outer_join_and_sharp_inner_join():
    source = geo.Curve.polyline([(0, 0, 0), (2, 0, 0), (2, 2, 0)])
    (inner,) = source.offset(0.2, normal="z")
    (outer,) = source.offset(-0.2, normal="z")
    assert inner.length == pytest.approx(3.6)
    assert outer.length == pytest.approx(4 + math.pi * 0.2 / 2)
    assert inner.bounding_box()[1][0] == pytest.approx(1.8)
    assert outer.bounding_box()[1][0] == pytest.approx(2.2)


def test_open_arc_and_joined_line_arc_keep_requested_side():
    arc = geo.Curve.arc((1, 0, 0), (0, 1, 0), (-1, 0, 0))
    assert arc.offset(0.2, normal="z")[0].length == pytest.approx(math.pi * 0.8)
    assert arc.offset(-0.2, normal="z")[0].length == pytest.approx(math.pi * 1.2)
    joined = geo.Curve.line((-1, 0, 0), (1, 0, 0)).joined(
        geo.Curve.arc((1, 0, 0), (2, 1, 0), (1, 2, 0))
    )
    assert joined.offset(0.2, normal="z")[0].length == pytest.approx(2 + math.pi * 0.8)
    assert joined.offset(-0.2, normal="z")[0].length == pytest.approx(2 + math.pi * 1.2)


def test_closed_circle_uses_traversal_and_plane_normal_for_side():
    source = geo.Curve.circle((0, 0, 0), 1)
    assert source.offset(0.2, normal="z")[0].length == pytest.approx(2 * math.pi * 0.8)
    assert source.offset(-0.2, normal="z")[0].length == pytest.approx(2 * math.pi * 1.2)
    assert source.offset(1.0, normal="z") == ()
    assert source.offset(1.2, normal="z") == ()
    assert source.offset(0.2, normal=(0, 0, -1))[0].length == pytest.approx(2 * math.pi * 1.2)


def test_annular_dilation_erosion_and_hole_collapse_follow_region_rule():
    ring = _ring()
    (grown,) = ring.offset(0.2)
    (eroded,) = ring.offset(-0.2)
    assert grown.area == pytest.approx(math.pi * (2.2**2 - 0.8**2))
    assert eroded.area == pytest.approx(math.pi * (1.8**2 - 1.2**2))
    assert grown.material == ring.material
    assert grown is not ring
    assert ring.area == pytest.approx(3 * math.pi)
    (closed_hole,) = ring.offset(1.0)
    assert closed_hole.area == pytest.approx(9 * math.pi)
    assert ring.offset(-1.2) == ()


def test_ellipse_offset_respects_curvature_and_hole_closure():
    ellipse = geo.Curve.ellipse((0, 0, 0), (2, 1), major_axis="x")
    profile = geo.Profile.from_wires(ellipse)
    perimeter = ellipse.length
    assert profile.offset(-0.2)[0].area == pytest.approx(
        profile.area - perimeter * 0.2 + math.pi * 0.2**2, rel=3e-5
    )
    with pytest.raises(ValueError, match="curvature cusp"):
        ellipse.offset(0.6, normal="z")
    holed = geo.Profile.from_wires(geo.Curve.circle((0, 0, 0), 3), [ellipse])
    assert holed.offset(0.6)[0].area == pytest.approx(
        math.pi * 3.6**2 - profile.offset(-0.6)[0].area, rel=1e-5
    )
    assert holed.offset(1.0)[0].area == pytest.approx(math.pi * 4**2)


def test_rectangle_dilation_uses_quarter_circle_corners():
    profile = geo.Profile.rectangle((0, 0, 0), (2, 2))
    assert profile.offset(0.2)[0].area == pytest.approx(4 + 8 * 0.2 + math.pi * 0.2**2)
    assert profile.offset(-0.2)[0].area == pytest.approx(1.6**2)
    assert profile.offset(-1.0) == ()


def test_neck_erosion_returns_both_surviving_regions():
    profile = geo.Profile.polygon(
        [
            (-3, -1, 0),
            (-1, -1, 0),
            (-1, -0.2, 0),
            (1, -0.2, 0),
            (1, -1, 0),
            (3, -1, 0),
            (3, 1, 0),
            (1, 1, 0),
            (1, 0.2, 0),
            (-1, 0.2, 0),
            (-1, 1, 0),
            (-3, 1, 0),
        ]
    )
    pieces = profile.offset(-0.5)
    assert len(pieces) == 2
    assert pieces[0].area == pytest.approx(pieces[1].area)
    assert all(piece.area > 0 for piece in pieces)


@pytest.mark.parametrize("scale", [1e-9, 1, 1e3])
def test_profile_offset_respects_model_scale(scale):
    ring = _ring(2 * scale, scale)
    (grown,) = ring.offset(0.2 * scale)
    assert grown.area / scale**2 == pytest.approx(math.pi * (2.2**2 - 0.8**2))


def test_oblique_plane_and_reflection_preserve_profile_area():
    profile = geo.Profile.rectangle((0, 0, 0), (2, 2))
    (base,) = profile.offset(0.2)
    placed = profile.rotated("x", 37).mirrored("y")
    (result,) = placed.offset(0.2)
    assert result.area == pytest.approx(base.area)


def test_curved_sheet_moves_true_trim_and_keeps_source_independent():
    owner = geo.Cylinder(radius=1, height=2)
    sheet = owner.face(near=(1, 0, 1)).detached()
    (outside,) = sheet.offset(0.1)
    (inside,) = sheet.offset(-0.1)
    assert isinstance(outside, geo.Surface)
    assert outside.bounding_box()[1][0] == pytest.approx(1.1, abs=2e-6)
    assert inside.bounding_box()[1][0] == pytest.approx(0.9, abs=2e-6)
    assert outside.bounding_box()[1][2] == pytest.approx(2, abs=2e-6)
    assert sheet.bounding_box()[1][0] == pytest.approx(1, abs=2e-6)


def test_curved_sheet_fold_and_tolerance_fail():
    sheet = geo.Cylinder(radius=1, height=2).face(near=(1, 0, 1)).detached()
    with pytest.raises(ValueError, match="folds"):
        sheet.offset(-1.2)
    with pytest.raises(ValueError, match="positive"):
        sheet.offset(0.1, tolerance=0)


def test_bounded_spherical_patch_offsets_with_its_trimmed_rim():
    patch = geo.Surface.parametric(
        lambda u, v: (np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)),
        u=(-0.6, 0.6),
        v=(-0.4, 0.4),
        samples=(12, 12),
    )
    (outer,) = patch.offset(0.1)
    (inner,) = patch.offset(-0.1)
    assert outer.bounding_box()[1][0] == pytest.approx(1.1, abs=2e-6)
    assert inner.bounding_box()[1][0] == pytest.approx(0.9, abs=2e-6)
    assert outer.bounding_box()[1][2] == pytest.approx(1.1 * math.sin(0.4), abs=2e-4)
    with pytest.raises(ValueError, match="singular normal"):
        geo.Sphere(radius=1).face(near=(0, 0, 1)).detached().offset(0.1)


def test_project_store_replays_body_built_from_offset_profile(tmp_path):
    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    (clearance,) = _ring().offset(0.1)
    body = clearance.extruded(vector=(0, 0, 0.1))
    lines = np.linspace(-3, 3, 3)
    grid = GridLines(x=lines, y=lines, z=lines)
    ProjectStore.create(tmp_path / "offset", mio.Mesh.from_grid(grid), geometry=[body])
    (restored,) = mio.open_project(tmp_path / "offset").geometry
    assert restored.volume() == pytest.approx(body.volume())


def test_project_store_replays_body_built_from_offset_sheet(tmp_path):
    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    wall = geo.Cylinder(radius=1, height=2).face(near=(1, 0, 1)).detached()
    (shifted,) = wall.offset(0.1)
    body = shifted.thickened(0.05, material="pec")
    lines = np.linspace(-2, 2, 3)
    grid = GridLines(x=lines, y=lines, z=lines)
    ProjectStore.create(tmp_path / "sheet-offset", mio.Mesh.from_grid(grid), geometry=[body])
    (restored,) = mio.open_project(tmp_path / "sheet-offset").geometry
    assert restored.volume() == pytest.approx(body.volume())


def test_invalid_plane_and_nonplanar_curve_fail_at_call():
    line = geo.Curve.line((0, 0, 0), (1, 0, 0))
    with pytest.raises(ValueError, match="perpendicular"):
        line.offset(0.1, normal="x")
    spatial = geo.Curve.polyline([(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)])
    with pytest.raises(ValueError, match="plane"):
        spatial.offset(0.1, normal="z")


def test_methods_recipe_and_public_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/methods/geometry.md").read_text()
    section = text.split("## Offset curves and sheets", 1)[1].split("\n## ", 1)[0]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, re.S):
        exec(block, namespace)
    assert len(namespace["clearance"]) == 1
    runpy.run_path(str(root / "examples/tutorials/plot_27_offset_geometry.py"))
