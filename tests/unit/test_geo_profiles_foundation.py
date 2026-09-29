"""Analytic and invalid-topology gates for the curve/profile foundation."""

import math

import numpy as np
import pytest
from scipy.special import ellipe

from magnelio import geo

pytest.importorskip("OCC.Core.BRepPrimAPI")


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3])
def test_exact_primitives_at_model_scales(scale):
    line = geo.Curve.line((0, 0, 0), (3 * scale, 4 * scale, 0))
    circle = geo.Curve.circle((scale, 2 * scale, 3 * scale), 2 * scale, normal=(1, 2, 3))
    ellipse = geo.Curve.ellipse((0, 0, 0), (3 * scale, 2 * scale), major_axis="x")
    assert line.length == pytest.approx(5 * scale, rel=1e-12)
    assert circle.is_closed and ellipse.is_closed
    assert circle.length == pytest.approx(4 * math.pi * scale, rel=1e-12)
    assert ellipse.length == pytest.approx(12 * scale * ellipe(1 - 4 / 9), rel=1e-8)
    assert geo.Profile.from_wires(ellipse).area == pytest.approx(6 * math.pi * scale**2, rel=1e-12)
    assert geo.Profile.circle((0, 0, 0), scale).area == pytest.approx(math.pi * scale**2)


def test_ellipse_minor_first_and_oblique_orientation():
    ellipse = geo.Curve.ellipse((1, 2, 3), (2, 3), major_axis="y", normal="x")
    assert geo.Profile.from_wires(ellipse).area == pytest.approx(6 * math.pi)
    assert ellipse.bounding_box()[0] == pytest.approx((1, 0, 0), abs=1e-8)
    assert ellipse.bounding_box()[1] == pytest.approx((1, 4, 6), abs=1e-8)
    from OCC.Core.BRepAdaptor import BRepAdaptor_CompCurve

    adaptor = BRepAdaptor_CompCurve(ellipse._occ_shape())
    first = adaptor.Value(adaptor.FirstParameter())
    assert ellipse._ends[0] == pytest.approx((first.X(), first.Y(), first.Z()))
    assert ellipse._ends[0] == pytest.approx((1, 2, 6))


def test_rectangle_world_orientation_and_boundary_detachment():
    profile = geo.Profile.rectangle(
        (1, 2, 3), (4, 2), normal=(1, 1, 1), x_direction=(1, -1, 0), material="pec", name="section"
    )
    placement = geo.Translation((4, 2, -1)) @ geo.Rotation("y", 37) @ geo.Mirror("x") @ geo.Scale(2)
    placed = placement @ profile
    assert profile.area == pytest.approx(8)
    assert placed.area == pytest.approx(32)
    assert placed.material is profile.material and placed.name == "section"
    (boundary,) = placed.boundary()
    assert boundary.is_closed and boundary.length == pytest.approx(24)
    detached = geo.Profile.from_wires(boundary)
    assert detached.area == pytest.approx(placed.area)
    assert detached.bounding_box() == pytest.approx(np.array(placed.bounding_box()))
    assert profile.area == pytest.approx(8)


def annulus(center=(0, 0, 0), normal="z", scale=1):
    return geo.Profile.from_wires(
        geo.Curve.circle(center, 2 * scale, normal=normal),
        [geo.Curve.circle(center, scale, normal=normal)],
        material="pec",
        name="ring",
    )


def test_holes_survive_all_profile_operations():
    ring = annulus()
    assert ring.area == pytest.approx(3 * math.pi)
    assert len(ring.boundary()) == 2
    extrusion = ring.extruded((0, 0, 5))
    assert extrusion.volume() == pytest.approx(15 * math.pi)
    sweep = ring.swept(geo.Curve.arc((8, 0, 0), (0, 8, 0), (-8, 0, 0)))
    assert sweep.volume() == pytest.approx(24 * math.pi**2, rel=1e-7)
    revolved = annulus((8, 0, 0), "y").revolved("z")
    assert revolved.volume() == pytest.approx(48 * math.pi**2, rel=1e-10)
    loft = geo.Loft(ring, annulus((0, 0, 5), scale=2), blend="ruled")
    assert loft.volume() == pytest.approx(35 * math.pi, rel=1e-10)
    for solid in (extrusion, sweep, revolved, loft):
        assert isinstance(solid, geo.Solid)
        assert solid.material is ring.material
    assert geo.Loft(ring, ring.translated((0, 0, 5))).volume() == pytest.approx(15 * math.pi)


def test_multiple_holes_and_winding_independence():
    outer = geo.Profile.rectangle((0, 0, 0), (10, 10)).boundary()[0]
    a = geo.Curve.circle((-2, 0, 0), 1, normal=(0, 0, -1))
    b = geo.Curve.ellipse((2, 0, 0), (1, 2), major_axis="y")
    profile = geo.Profile.from_wires(outer, [a, b])
    area = 100 - 3 * math.pi
    assert profile.area == pytest.approx(area)
    assert profile.extruded((0, 0, 3), material="pec").volume() == pytest.approx(3 * area)
    spine = geo.Curve.arc((20, 0, 0), (0, 20, 0), (-20, 0, 0))
    assert profile.swept(spine, material="pec").volume() == pytest.approx(
        20 * math.pi * area, rel=1e-7
    )
    radial = profile.rotated("x", 90).translated((20, 0, 0))
    # Pappus with unequal hole areas: first radial moment is 20*A - 2*pi.
    assert radial.revolved("z", material="pec").volume() == pytest.approx(
        2 * math.pi * (20 * area - 2 * math.pi), rel=1e-10
    )
    solid = geo.Loft(profile, profile.translated((0, 0, 3)), material="pec")
    assert solid.volume() == pytest.approx(3 * area)
    assert len((geo.Mirror("x") @ profile).boundary()) == 3


@pytest.mark.parametrize(
    "hole",
    [
        geo.Curve.circle((4, 0, 0), 1),
        geo.Curve.circle((1, 0, 0), 1),
        geo.Curve.circle((1.5, 0, 0), 1),
        geo.Curve.circle((0, 0, 1), 1),
        geo.Curve.circle((0, 0, 0), 1, normal="x"),
    ],
)
def test_invalid_hole_placement_rejected_at_factory(hole):
    with pytest.raises(ValueError):
        geo.Profile.from_wires(geo.Curve.circle((0, 0, 0), 2), [hole])


@pytest.mark.parametrize(
    "centers,radii",
    [
        (((0, 0, 0), (0, 0, 0)), (1, 0.5)),
        (((-0.5, 0, 0), (0.5, 0, 0)), (1, 1)),
        (((-1, 0, 0), (1, 0, 0)), (1, 1)),
    ],
)
def test_nested_intersecting_and_touching_holes_rejected(centers, radii):
    with pytest.raises(ValueError, match="touch|intersect|nest"):
        geo.Profile.from_wires(
            geo.Curve.circle((0, 0, 0), 4), [geo.Curve.circle(c, r) for c, r in zip(centers, radii)]
        )


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)],
        [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        [(0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0)],
        [(0, 0, 0), (0, 0, 0), (1, 0, 0)],
    ],
)
def test_invalid_polygon_rejected_at_factory(points):
    with pytest.raises(ValueError):
        geo.Profile.polygon(points)


def test_polygon_explicit_closure_and_concavity():
    points = [(0, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    assert geo.Profile.polygon(points).area == pytest.approx(5)
    assert geo.Profile.polygon([*points, points[0]]).area == pytest.approx(5)


def test_category_and_argument_errors():
    assert not hasattr(geo, "Face")
    assert not hasattr(geo.Curve, "covered")
    with pytest.raises(TypeError, match="Curve"):
        geo.Profile.from_wires(geo.Brick())
    with pytest.raises(ValueError, match="closed"):
        geo.Profile.from_wires(geo.Curve.line((0, 0, 0), (1, 0, 0)))
    with pytest.raises(TypeError, match="Curve"):
        geo.Profile.from_wires(geo.Curve.circle((0, 0, 0), 1), holes=None)
    with pytest.raises(ValueError, match="radius"):
        geo.Curve.circle((0, 0, 0), -1)
    with pytest.raises(ValueError, match="distinct"):
        geo.Curve.line((0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="parallel"):
        geo.Profile.rectangle((0, 0, 0), (1, 1), x_direction="z")
    with pytest.raises(ValueError, match="parallel"):
        geo.Curve.ellipse((0, 0, 0), (1, 2), major_axis="z")
    with pytest.raises(ValueError, match="finite"):
        geo.Curve.ellipse((0, 0, 0), (math.nan, 1), major_axis="x")
    with pytest.raises(ValueError, match="number of holes"):
        geo.Loft(annulus(), geo.Profile.circle((0, 0, 5), 2))


def test_profile_solids_round_trip_through_project_store(tmp_path):
    import magnelio as mio
    from magnelio.io.project import ProjectStore
    from magnelio.mesh import GridLines

    ring = annulus(scale=1e-3)
    solid = geo.Translation((5e-3, 0, 0)) @ geo.Loft(
        ring, ring.translated((0, 0, 5e-3)), name="placed_ring"
    )
    model = mio.GeometryModel()
    model.add(solid)
    grid = GridLines(
        x=np.linspace(0, 0.01, 5), y=np.linspace(-0.005, 0.005, 5), z=np.linspace(0, 0.01, 5)
    )
    ProjectStore.create(tmp_path / "ring", mio.Mesh.from_grid(grid), geometry=model)
    (restored,) = mio.open_project(tmp_path / "ring").geometry
    assert restored.name == solid.name
    assert restored.material == solid.material
    assert restored.volume() == pytest.approx(15 * math.pi * 1e-9)
    assert np.array(restored.bounding_box()) == pytest.approx(np.array(solid.bounding_box()))
    from magnelio.geo._occ_backend import cross_section_polygons

    loops = cross_section_polygons(restored._occ_shape(), "z", 2.5e-3)
    assert len(loops) == 2


def test_documented_profile_recipes_execute():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    methods = (root / "docs/methods/geometry.md").read_text()
    methods = methods.split("## Exact curves and planar profiles", 1)[1].split("\n## ", 1)[0]
    sources = [
        (methods, 30 * math.pi * 1e-9, "tube"),
        (
            (root / "docs/migration-geometry.md").read_text().split("## Uniform operations", 1)[0],
            60e-9,
            "body",
        ),
    ]
    for text, expected, name in sources:
        for snippet in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
            namespace = {}
            exec(compile(snippet, "geometry documentation", "exec"), namespace)
            assert namespace[name].volume() == pytest.approx(expected, rel=1e-12)
