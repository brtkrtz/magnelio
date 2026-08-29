"""PointClassifierSet: the bounding-box screen and the per-shape classifier cache.

The set must answer exactly what the per-call ``point_in_shape`` loop
answered, in both walking orders and with a skipped shape, while
loading every solid's classifier once.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.geo import Brick, Cylinder, Difference
from magnelio.geo._occ_backend import PointClassifierSet, point_in_shape
from magnelio.materials import Material

pytest.importorskip("OCC")

AIR = Material.from_isotropic(name="air", epsilon=1.0)
PEC = Material.pec()


def _model():
    """Two overlapping bricks, a cylinder and a pocketed body — ambiguous points included."""
    a = Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=AIR)
    b = Brick(origin=(2e-3, 0, 0), size=(4e-3, 4e-3, 4e-3), material=PEC)
    c = Cylinder(origin=(3e-3, 2e-3, 4e-3), radius=1e-3, height=2e-3, axis="z", material=AIR)
    d = Difference(
        Brick(origin=(0, 0, -3e-3), size=(6e-3, 4e-3, 3e-3), material=AIR),
        Brick(origin=(1e-3, 1e-3, -2e-3), size=(1e-3, 1e-3, 1e-3), material=PEC),
    )
    return [a, b, c, d]


def _reference(shapes, point, skip=None, reverse=False, tolerance=1e-7):
    order = reversed(list(enumerate(shapes))) if reverse else enumerate(shapes)
    for k, shape in order:
        if shape is skip:
            continue
        if point_in_shape(shape._occ_shape(1.0), point, tolerance=tolerance):
            return k
    return None


@pytest.mark.parametrize("reverse", [False, True])
def test_matches_the_per_call_loop(reverse):
    shapes = _model()
    classifiers = PointClassifierSet(shapes, scale=1.0)
    rng = np.random.default_rng(7)
    points = rng.uniform((-1e-3, -1e-3, -4e-3), (7e-3, 5e-3, 7e-3), size=(300, 3))
    special = [(1.5e-3, 1.5e-3, -1.5e-3), (3e-3, 2e-3, 2e-3), (3e-3, 2e-3, 5e-3)]
    points = np.vstack([points, special])
    for p in map(tuple, points):
        assert classifiers.first_containing(p, reverse=reverse) == _reference(
            shapes, p, reverse=reverse
        )


def test_skip_is_by_identity():
    shapes = _model()
    classifiers = PointClassifierSet(shapes, scale=1.0)
    p = (3e-3, 2e-3, 2e-3)  # inside both bricks
    assert classifiers.first_containing(p) == 0
    assert classifiers.first_containing(p, skip=shapes[0]) == 1
    assert classifiers.first_containing(p, reverse=True) == 1
    assert classifiers.first_containing(p, reverse=True, skip=shapes[1]) == 0


def test_screen_is_padded_by_the_tolerance():
    shapes = _model()
    tol = 1e-6
    classifiers = PointClassifierSet(shapes, scale=1.0, tolerance=tol)
    on = (4e-3 + 0.5 * tol, 2e-3, 2e-3)  # half a tolerance outside brick a, inside b
    assert 0 in classifiers.candidates(on)
    assert classifiers.contains(0, on)  # ON within tolerance
    off = (-2 * tol, 2e-3, 2e-3)
    assert 0 not in classifiers.candidates(off)
    assert classifiers.first_containing(off) is None


def test_every_classifier_loads_once(monkeypatch):
    from OCC.Core import BRepClass3d

    loads = []
    original = BRepClass3d.BRepClass3d_SolidClassifier

    class Counting(original):
        def Load(self, shape):  # noqa: N802 — kernel spelling
            loads.append(shape)
            return original.Load(self, shape)

    monkeypatch.setattr(BRepClass3d, "BRepClass3d_SolidClassifier", Counting)
    shapes = _model()
    classifiers = PointClassifierSet(shapes, scale=1.0)
    for x in np.linspace(0.5e-3, 5.5e-3, 40):
        classifiers.first_containing((x, 2e-3, 2e-3))
        classifiers.first_containing((x, 2e-3, 2e-3), reverse=True)
    assert len(loads) == 2  # bricks a and b; the cylinder and the pocketed body were screened


def test_shapes_the_kernel_cannot_handle_are_skipped():
    class Broken:
        def bounding_box(self, scale):
            raise RuntimeError("no box")

        def _occ_shape(self, scale):
            raise RuntimeError("no shape")

    class BoxedButUnbuildable(Broken):
        def bounding_box(self, scale):
            return (0.0, 0.0, 0.0), (1e-2, 1e-2, 1e-2)

    shapes = [Broken(), BoxedButUnbuildable(), *_model()]
    classifiers = PointClassifierSet(shapes, scale=1.0)
    assert classifiers.first_containing((1e-3, 1e-3, 1e-3)) == 2
    assert classifiers.first_containing((1e-3, 1e-3, 1e-3), reverse=True) == 2


def _csg_model():
    """A pocketed slab (brick and cylinder pockets), a union, an intersection, a nested cut."""
    from magnelio.geo import Intersection, Union

    slab = Brick(origin=(0, 0, 0), size=(6e-3, 4e-3, 2e-3), material=AIR)
    pocket = Brick(origin=(1e-3, 1e-3, 1e-3), size=(1e-3, 1e-3, 2e-3), material=PEC)
    bore = Cylinder(origin=(4e-3, 2e-3, 0.5e-3), radius=0.6e-3, height=1e-3, axis="z", material=PEC)
    pocketed = Difference(slab, pocket, bore)
    union = Union(
        Brick(origin=(0, 0, 3e-3), size=(3e-3, 4e-3, 1e-3), material=PEC),
        Brick(origin=(2e-3, 0, 3e-3), size=(4e-3, 4e-3, 1e-3), material=PEC),
        material=PEC,
    )
    inter = Intersection(
        Brick(origin=(0, 0, 5e-3), size=(4e-3, 4e-3, 1e-3), material=AIR),
        Brick(origin=(2e-3, 1e-3, 5e-3), size=(4e-3, 2e-3, 1e-3), material=AIR),
    )
    nested = Difference(
        union,
        Difference(
            Brick(origin=(2e-3, 1e-3, 3e-3), size=(2e-3, 2e-3, 1e-3), material=PEC),
            Brick(origin=(2.5e-3, 1.5e-3, 3e-3), size=(1e-3, 1e-3, 1e-3), material=PEC),
        ),
    )
    return [pocketed, union, inter, nested]


def test_csg_nodes_are_answered_from_their_operands():
    """Difference / Union / Intersection states follow the point-set identities,
    including the ON band on the pocket walls, and match the kernel's answer on
    the Boolean solid."""
    tol = 1e-6
    shapes = _csg_model()
    classifiers = PointClassifierSet(shapes, scale=1.0, tolerance=tol)
    rng = np.random.default_rng(11)
    points = list(map(tuple, rng.uniform((-1e-3, -1e-3, -1e-3), (7e-3, 5e-3, 7e-3), size=(400, 3))))
    # The pocket wall at x = 1e-3, from both sides, within and beyond the band;
    # the bore wall; the cut-out of the nested difference (the inner brick is
    # metal again); the union's seam plane.
    points += [
        (1e-3 - 0.5 * tol, 1.5e-3, 1.5e-3),
        (1e-3 + 0.5 * tol, 1.5e-3, 1.5e-3),
        (1e-3 + 3 * tol, 1.5e-3, 1.5e-3),
        (1e-3 - 3 * tol, 1.5e-3, 1.5e-3),
        (4e-3 + 0.6e-3 + 0.5 * tol, 2e-3, 1e-3),
        (4e-3 + 0.6e-3 - 0.5 * tol, 2e-3, 1e-3),
        (4e-3, 2e-3, 1e-3),
        (3e-3, 2e-3, 3.5e-3),
        (2.2e-3, 1.2e-3, 3.5e-3),
        (2e-3, 2e-3, 3.5e-3),
        (3e-3, 2e-3, 5.5e-3),
        (1e-3, 2e-3, 5.5e-3),
    ]
    for p in points:
        for k, shape in enumerate(shapes):
            expected = point_in_shape(shape._occ_shape(1.0), p, tolerance=tol)
            assert classifiers.contains(k, p) == expected, (k, p)


def test_only_leaf_classifiers_are_loaded(monkeypatch):
    from OCC.Core import BRepClass3d

    loads = []
    original = BRepClass3d.BRepClass3d_SolidClassifier

    class Counting(original):
        def Load(self, shape):  # noqa: N802 — kernel spelling
            loads.append(shape)
            return original.Load(self, shape)

    monkeypatch.setattr(BRepClass3d, "BRepClass3d_SolidClassifier", Counting)
    shapes = _csg_model()
    pocketed = shapes[0]
    classifiers = PointClassifierSet(shapes, scale=1.0, tolerance=1e-6)
    assert classifiers.contains(0, (0.5e-3, 0.5e-3, 0.5e-3))  # slab, no pocket box holds it
    assert len(loads) == 1
    assert not classifiers.contains(0, (1.5e-3, 1.5e-3, 1.5e-3))  # in the brick pocket
    assert len(loads) == 2
    assert classifiers.contains(0, (1.5e-3, 1.5e-3, 0.5e-3))  # below the pocket, same boxes
    assert len(loads) == 2
    assert all(not s.IsSame(pocketed._occ_shape(1.0)) for s in loads)
    # The nested cut: union pieces and the inner difference's operands, never a Boolean solid.
    assert classifiers.contains(3, (3e-3, 2e-3, 3.5e-3))
    assert all(not s.IsSame(node._occ_shape(1.0)) for s in loads for node in (shapes[1], shapes[3]))


def test_a_node_with_an_unboxable_operand_falls_back_to_its_own_solid():
    class Unboxable(Brick):
        def bounding_box(self, scale=None):
            raise RuntimeError("no box")

    hole = Unboxable(origin=(1e-3, 1e-3, 1e-3), size=(1e-3, 1e-3, 2e-3), material=PEC)
    slab = Brick(origin=(0, 0, 0), size=(6e-3, 4e-3, 2e-3), material=AIR)
    shapes = [Difference(slab, hole)]
    classifiers = PointClassifierSet(shapes, scale=1.0, tolerance=1e-6)
    assert classifiers.contains(0, (0.5e-3, 0.5e-3, 0.5e-3))
    assert not classifiers.contains(0, (1.5e-3, 1.5e-3, 1.5e-3))
    assert classifiers._operand_boxes[id(shapes[0])] is None
