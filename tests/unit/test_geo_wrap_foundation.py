"""WP6.9 chart wrapping and tangent transition contracts."""

import json
import math
import re
import runpy
from pathlib import Path

import numpy as np
import pytest

from magnelio import geo
from magnelio.geo._topology_store import from_recipe, to_recipe

pytest.importorskip("OCC.Core.BRepFeat")


def _cylinder():
    length = math.pi / 2
    target = geo.Surface.parametric(
        lambda u, v: (np.sin(u), v, 1 - np.cos(u)),
        u=(0, length),
        v=(-0.3, 0.3),
        samples=(65, 9),
    )
    return geo.Wrap(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, length),
        v=(-0.3, 0.3),
        max_strain=0.01,
    ), length


def test_cylinder_wrap_retains_explicit_normal_layer_spacing():
    wrap, length = _cylinder()
    source = geo.Brick(origin=(0, -0.2, 0), size=(length, 0.4, 0.05), material="pec")
    result = wrap @ source
    expected = 0.4 * math.pi / 4 * (1 - 0.95**2)
    assert result.volume() == pytest.approx(expected, rel=2e-5)
    assert result.material == source.material
    assert source.volume() == pytest.approx(0.4 * 0.05 * length)


def test_wrap_maps_openings_and_distinct_material_layers():
    wrap, length = _cylinder()
    base = geo.Brick(origin=(0, -0.2, 0), size=(length, 0.4, 0.04), material="pec")
    trace = geo.Brick(origin=(0.2, -0.05, 0.04), size=(length - 0.4, 0.1, 0.01), material="air")
    hole = geo.Cylinder(origin=(length / 2, 0, -0.1), radius=0.05, height=0.2)
    group = wrap @ geo.Group(base - hole, trace, name="layers")
    board, conductor = tuple(group.members())
    assert group.name == "layers"
    assert board.material.name == "PEC"
    assert conductor.material.name == "air"
    assert board.volume() < (wrap @ base).volume()
    assert conductor.volume() > 0


def test_wrap_accepts_double_curvature_with_declared_strain():
    target = geo.Surface.parametric(
        lambda u, v: (u, v, 0.02 * np.sin(np.pi * u) * np.cos(np.pi * v)),
        u=(0, 1),
        v=(-0.3, 0.3),
        samples=(65, 33),
    )
    wrap = geo.Wrap(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, 1),
        v=(-0.3, 0.3),
        max_strain=0.1,
        tolerance=1e-4,
    )
    source = geo.Brick(origin=(0.1, -0.2, 0), size=(0.8, 0.4, 0.01), material="pec")
    result = wrap @ source
    assert result.volume() > 0
    assert result.bounding_box()[1][2] > source.bounding_box()[1][2]


def test_conical_chart_has_explicit_circumferential_stretch():
    target = geo.Surface.parametric(
        lambda u, v: ((1 + 0.1 * u) * np.cos(v), (1 + 0.1 * u) * np.sin(v), u),
        u=(0, 1),
        v=(-0.2, 0.2),
        samples=(25, 25),
    )
    wrap = geo.Wrap(
        target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, 1),
        v=(-0.2, 0.2),
        max_strain=0.15,
        tolerance=1e-4,
    )
    source = geo.Brick(origin=(0.1, -0.1, 0), size=(0.8, 0.2, 0.01))
    result = wrap @ source
    assert result.volume() > 0
    assert result.bounding_box()[1][0] > 1


def test_non_developable_chart_rejects_independently_known_metric_distortion():
    amplitude = 0.1
    target = geo.Surface.parametric(
        lambda u, v: (u, v, amplitude * u * v),
        u=(0, 1),
        v=(-0.3, 0.3),
        samples=(25, 25),
    )
    source = geo.Brick(origin=(0, -0.3, 0), size=(1, 0.6, 0.005))
    maximum_stretch = math.sqrt(1 + amplitude**2 * (1 + 0.3**2)) - 1
    assert 0.005 < maximum_stretch < 0.006
    kwargs = dict(origin=(0, 0, 0), along="x", across="y", u=(0, 1), v=(-0.3, 0.3))
    accepted = geo.Wrap(target, max_strain=0.006, tolerance=1e-4, **kwargs)
    assert (accepted @ source).volume() > 0
    rejected = geo.Wrap(target, max_strain=0.004, tolerance=1e-4, **kwargs)
    with pytest.raises(ValueError, match="strain"):
        (rejected @ source).volume()


def test_wrap_rejects_uncovered_source_and_excess_strain():
    wrap, length = _cylinder()
    too_wide = geo.Brick(origin=(0, -0.4, 0), size=(length, 0.8, 0.02))
    with pytest.raises(ValueError, match="transverse extent"):
        (wrap @ too_wide).volume()
    too_long = geo.Brick(origin=(-0.1, -0.2, 0), size=(length + 0.2, 0.4, 0.02))
    with pytest.raises(ValueError, match="longitudinal extent"):
        (wrap @ too_long).volume()
    strict = geo.Wrap(
        wrap.target,
        origin=(0, 0, 0),
        along="x",
        across="y",
        u=(0, 0.8 * length),
        v=(-0.3, 0.3),
        max_strain=0.01,
    )
    source = geo.Brick(origin=(0, -0.2, 0), size=(0.8 * length, 0.4, 0.02))
    with pytest.raises(ValueError, match="strain"):
        (strict @ source).volume()


def test_named_wrapped_body_replays_from_chart():
    wrap, length = _cylinder()
    source = geo.Brick(origin=(0, -0.2, 0), size=(length, 0.4, 0.02), material="pec").tag_face(
        "outer", normal=(0, 0, 1)
    )
    result = wrap @ source
    recipe = json.loads(json.dumps(to_recipe(result)))
    assert any(node["operation"] == "wrap" for node in recipe["nodes"])
    restored = from_recipe(recipe)
    assert restored.volume() == pytest.approx(result.volume(), rel=1e-6)
    assert restored.face(name="outer") is not None


def test_wrapped_sheet_has_no_physical_thickness():
    wrap, length = _cylinder()
    source = geo.Profile.rectangle(
        (length / 2, 0, 0), (length / 2, 0.2), normal="z", x_direction="x"
    )
    result = wrap @ source
    assert isinstance(result, geo.Sheet)
    assert result.bounding_box()[1][2] > 0
    assert result.thickened(0.01, material="pec").volume() > 0


def test_asymmetric_tangent_transition_matches_both_wall_normals():
    start = geo.Brick(origin=(-1, -0.5, 0), size=(2, 1, 1), material="pec")
    end = geo.Brick(origin=(-1.5, -0.75, 3), size=(3, 1.5, 1), material="pec")
    transition = start.face(normal="z").lofted(end.face(normal=(0, 0, -1)), blend="tangent")
    walls = (
        (
            max(transition.faces(), key=lambda face: face.centroid[0]),
            ((1, 0, 1), (1.5, 0, 3)),
            (1, 0, 0),
        ),
        (
            max(transition.faces(), key=lambda face: face.centroid[1]),
            ((0, 0.5, 1), (0, 0.75, 3)),
            (0, 1, 0),
        ),
    )
    for face, points, expected in walls:
        for point in points:
            normal = np.asarray(face.normal_at(point))
            angle = math.acos(np.clip(np.dot(normal, expected), -1, 1))
            assert angle < 1e-5


def test_methods_recipe_and_public_tutorial_execute():
    import matplotlib

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[2]
    content = (root / "docs/methods/geometry.md").read_text()
    section = content.split("## Wrap a flat component onto a curved patch", 1)[1].split("\n## ", 1)[
        0
    ]
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, re.S):
        exec(block, namespace)
    assert all(member.volume() > 0 for member in namespace["wrapped"].members())
    runpy.run_path(str(root / "examples/tutorials/plot_29_wrap_and_g1_transition.py"))
