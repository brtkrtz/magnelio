"""Unit tests for the DD-120 automatic unit scaling (geo/_scaling.py)."""

import math

import numpy as np
import pytest

from magnelio.geo._scaling import (
    analytic_bbox,
    box_of_points,
    choose_scale,
    model_scale,
    union_boxes,
)
from magnelio.materials.material import Material


def _air():
    return Material.air()


def _occ():
    """Skip test if pythonocc-core is not installed."""
    return pytest.importorskip("OCC.Core.BRepPrimAPI")


# ── choose_scale properties (no OCC needed) ──────────────────────────────────


class TestChooseScale:
    def test_identity_band(self):
        # mm-scale device (WG90-like) and metre-scale device: s must be
        # exactly 1.0 so existing models stay bit-identical.
        assert choose_scale((0, 0, 0), (22.86e-3, 10.16e-3, 40e-3)) == 1.0
        assert choose_scale((0, 0, 0), (1.0, 1.0, 1.0)) == 1.0
        assert choose_scale((0, 0, 0), (1e-3, 0, 0)) == 1.0
        assert choose_scale((0, 0, 0), (1e4 / math.sqrt(3),) * 3) == 1.0

    def test_power_of_two(self):
        for diag in (1e-6, 1e-4, 3.7e-5, 1e5, 2.3e7):
            s = choose_scale((0, 0, 0), (diag, 0, 0))
            exponent = math.log2(s)
            assert exponent == round(exponent), f"s={s} is not a power of two"

    def test_target_range(self):
        # Outside the identity band the scaled diagonal must land within
        # a factor sqrt(2) of the 128 target.
        for diag in (1e-6, 1e-4, 5e-4, 2e4, 1e7):
            s = choose_scale((0, 0, 0), (diag, 0, 0))
            assert 64.0 <= s * diag <= 256.0

    def test_degenerate(self):
        assert choose_scale((0, 0, 0), (0, 0, 0)) == 1.0
        assert choose_scale((0, 0, 0), (math.nan, 0, 0)) == 1.0
        assert choose_scale((0, 0, 0), (math.inf, 0, 0)) == 1.0

    def test_fiber_scale(self):
        # 125 um coating radius, 100 um simulated length (a realistic
        # optics-scale domain): features land decades above the OCC
        # precision in scaled units.
        s = choose_scale((-125e-6, -125e-6, 0.0), (125e-6, 125e-6, 100e-6))
        assert s > 1.0
        assert 10e-9 * s > 1e-6  # a 10 nm feature clears 1e-7 comfortably


class TestModelScale:
    def test_empty(self):
        assert model_scale([]) == 1.0

    def test_mm_model_is_identity(self):
        from magnelio.geo.primitives import Brick

        shapes = [Brick(origin=(0, 0, 0), size=(1e-3, 2e-3, 3e-3), material=_air())]
        assert model_scale(shapes) == 1.0

    def test_micron_model_scales(self):
        from magnelio.geo.primitives import Cylinder

        shapes = [
            Cylinder(origin=(0, 0, 0), radius=62.5e-6, height=1e-4, axis="z", material=_air())
        ]
        s = model_scale(shapes)
        assert s > 1.0 and math.log2(s) == round(math.log2(s))


# ── analytic bbox containment vs. the OCC BRep bbox ──────────────────────────


def _shape_zoo():
    from magnelio.geo.curves import Curve
    from magnelio.geo.modifications import extrude, revolve, sweep
    from magnelio.geo.operations import Difference, Group, Intersection, Union
    from magnelio.geo.primitives import Brick, Cone, Cylinder, Face, Sphere, Torus
    from magnelio.geo.surfaces import Surface
    from magnelio.geo.transforms import rotate, scale, translate

    air = _air()
    brick = Brick(origin=(1e-3, -2e-3, 0.0), size=(4e-3, 3e-3, 2e-3), material=air)
    sphere = Sphere(center=(2e-3, 1e-3, -1e-3), radius=1.5e-3, material=air)
    cyl_z = Cylinder(origin=(0, 0, 0), radius=1e-3, height=5e-3, axis="z", material=air)
    cyl_vec = Cylinder(
        origin=(1e-3, 1e-3, 1e-3), radius=0.5e-3, height=-3e-3, axis=(1, 1, 0), material=air
    )
    cone = Cone(
        origin=(0, 0, 0), bottom_radius=2e-3, top_radius=0.5e-3, height=4e-3, axis="x", material=air
    )
    torus = Torus(center=(0, 0, 0), major_radius=3e-3, minor_radius=0.5e-3, axis="y", material=air)
    face = Face(normal="z", points=((0, 0), (2e-3, 0), (2e-3, 1e-3), (0, 1e-3)), position=0.5e-3)
    dish = Surface.parametric(
        lambda r, phi: (r * np.cos(phi), r * np.sin(phi), r * r / 4e-3),
        u=(0.0, 2e-3),
        v=(0.0, 2 * np.pi),
        samples=(12, 24),
    )
    zoo = [
        ("surface", dish),
        ("surface_extruded", extrude(dish, vector=(0, 0, -0.3e-3), material=air)),
        ("surface_rotated", rotate(dish, "x", 30.0)),
        ("brick", brick),
        ("sphere", sphere),
        ("cylinder_z", cyl_z),
        ("cylinder_vec_neg", cyl_vec),
        ("cone_x", cone),
        ("torus_y", torus),
        ("face", face),
        ("union", Union(brick, sphere)),
        ("intersection", Intersection(brick, sphere)),
        ("difference", Difference(brick, sphere)),
        ("group", Group(brick, sphere)),
        ("translated", translate(brick, (1e-3, -1e-3, 2e-3))),
        ("rotated", rotate(brick, (1, 2, 3), 37.0, origin=(1e-3, 0, 0))),
        ("scaled", scale(sphere, 2.5, center=(0, 0, 0))),
        ("rotated_union", rotate(Union(brick, sphere), "z", 45.0)),
        ("extruded", extrude(face, vector=(0, 0, 3e-3), material=air)),
        (
            "revolved",
            revolve(
                Face(normal="z", points=((1e-3, 0), (2e-3, 0), (2e-3, 1e-3))),
                axis="x",
                angle_deg=270.0,
                material=air,
            ),
        ),
        ("polyline", Curve.polyline([(0, 0, 0), (1e-3, 2e-3, 0), (0, 2e-3, 3e-3)])),
        ("arc", Curve.arc((0, 0, 0), (1e-3, 1e-3, 0), (2e-3, 0, 0))),
        ("spline", Curve.spline([(0, 0, 0), (1e-3, 2e-3, 0), (3e-3, 0, 1e-3)])),
        ("helix", Curve.helix(radius=1e-3, pitch=0.5e-3, turns=4, axis="z")),
        (
            "sweep_helix",
            sweep(
                Face(normal="z", points=((-1e-4, -1e-4), (1e-4, -1e-4), (1e-4, 1e-4))),
                Curve.helix(radius=1e-3, pitch=0.5e-3, turns=3, axis="z"),
                material=air,
            ),
        ),
    ]
    return zoo


class TestAnalyticBboxContainment:
    @pytest.mark.parametrize(
        "name,shape", _shape_zoo(), ids=lambda p: p if isinstance(p, str) else ""
    )
    def test_contains_occ_bbox(self, name, shape):
        _occ()
        lo_a, hi_a = analytic_bbox(shape)
        lo_o, hi_o = shape.bounding_box()
        # OCC pads its bbox by the shape tolerance (~1e-7); allow that
        # much slack in the containment check.
        eps = 1e-6 + 1e-6 * math.dist(lo_o, hi_o)
        for i in range(3):
            assert lo_a[i] <= lo_o[i] + eps, f"{name}: axis {i} lower bound not contained"
            assert hi_a[i] >= hi_o[i] - eps, f"{name}: axis {i} upper bound not contained"


class TestBoxHelpers:
    def test_union_and_points(self):
        box = union_boxes(
            [box_of_points([(0, 0, 0), (1, 1, 1)]), box_of_points([(-1, 2, 0), (0, 3, 0)])]
        )
        assert box == ((-1, 0, 0), (1, 3, 1))


class TestScaledOccBuilds:
    def test_sub_100nm_brick_builds_at_model_scale(self):
        # The DD-062 wall: a 20 nm brick was unrepresentable.  At the
        # DD-120 model scale it builds; the bare meter-space build
        # still raises with the informative guard.
        _occ()
        from magnelio.geo._occ_backend import make_brick
        from magnelio.geo.primitives import Brick

        brick = Brick(origin=(0, 0, 0), size=(20e-9, 1e-6, 1e-6), material=_air())
        s = model_scale([brick])
        assert s > 1.0
        lo, hi = brick.bounding_box(s)
        assert hi[0] - lo[0] == pytest.approx(20e-9, rel=1e-9)
        with pytest.raises(ValueError, match="OCC geometric precision"):
            make_brick((0, 0, 0), (20e-9, 1e-6, 1e-6), scale=1.0)

    def test_overlap_detection_at_micron_scale(self):
        # A genuine micron-scale overlap is reported with the correct
        # meter-volume; the relative default tolerance ignores float
        # dust but not real intersections.
        _occ()
        from magnelio.geo._occ_backend import check_pairwise_overlaps
        from magnelio.geo.primitives import Cylinder

        air = _air()
        pec = Material.pec()
        outer = Cylinder(origin=(0, 0, 0), radius=62.5e-6, height=100e-6, axis="z", material=air)
        inner = Cylinder(origin=(0, 0, 0), radius=4.5e-6, height=100e-6, axis="z", material=pec)
        s = model_scale([outer, inner])
        overlaps = check_pairwise_overlaps([outer, inner], materials=[air, pec], scale=s)
        assert len(overlaps) == 1
        _, _, volume = overlaps[0]
        assert volume == pytest.approx(math.pi * 4.5e-6**2 * 100e-6, rel=1e-6)

    def test_disjoint_micron_shapes_report_nothing(self):
        _occ()
        from magnelio.geo._occ_backend import check_pairwise_overlaps
        from magnelio.geo.primitives import Brick

        air = _air()
        pec = Material.pec()
        a = Brick(origin=(0, 0, 0), size=(1e-6, 1e-6, 1e-6), material=air)
        b = Brick(origin=(1e-6, 0, 0), size=(1e-6, 1e-6, 1e-6), material=pec)
        s = model_scale([a, b])
        assert check_pairwise_overlaps([a, b], materials=[air, pec], scale=s) == []

    def test_cross_section_roundtrip_at_scale(self):
        # Section polygons of a scaled shape come back in meters.
        _occ()
        import numpy as np

        from magnelio.geo._occ_backend import cross_section_polygons
        from magnelio.geo.primitives import Cylinder

        cyl = Cylinder(origin=(0, 0, 0), radius=62.5e-6, height=100e-6, axis="z", material=_air())
        s = model_scale([cyl])
        polys = cross_section_polygons(cyl._occ_shape(s), "z", 50e-6, deflection=1e-8, scale=s)
        assert polys
        r_max = max(np.hypot(p[:, 0], p[:, 1]).max() for p in polys)
        assert r_max == pytest.approx(62.5e-6, rel=1e-6)


class TestSectionPoolAtScale:
    def test_pool_matches_sequential_at_micron_scale(self, monkeypatch):
        # The spawn-pool workers receive BRep blobs serialised at the
        # model scale; a scale mismatch would silently poison the
        # section cache.  Force the pool on a micron model and compare
        # against the sequential path.
        _occ()
        import numpy as np

        import magnelio.geo._occ_backend as backend
        from magnelio.geo.primitives import Cylinder

        air = _air()
        cyl = Cylinder(origin=(0, 0, 0), radius=62.5e-6, height=100e-6, axis="z", material=air)
        s = model_scale([cyl])
        material_library = {1: air}
        n = 8
        positions = np.linspace(5e-6, 95e-6, n)
        face_specs = np.array([[p, -70e-6, -70e-6, 70e-6, 70e-6] for p in positions])
        face_axes = np.full(n, 2, dtype=np.int32)

        def run(workers: str):
            monkeypatch.setenv("MAGNELIO_SECTION_WORKERS", workers)
            return backend.compute_face_material_areas(
                [(cyl, 1)],
                material_library,
                face_specs,
                face_axes,
                prop="epsilon",
                deflection=1e-8,
                scale=s,
            )

        seq = run("0")
        monkeypatch.setattr(backend, "_SECTION_PARALLEL_MIN_QUERIES", 1)
        monkeypatch.setattr(backend, "_SECTION_PARALLEL_MIN_FACE_WORK", 1)
        par = run("2")
        np.testing.assert_array_equal(seq, par)


class TestNanoscaleGuards:
    """DD-120 audit: the 1e-30 zero-guards sit far below nm-scale data.

    Face areas at 10 nm cells are ~1e-16 m^2 and cell volumes ~1e-24
    m^3 — 12+ decades above the exact-zero guards in the polygon
    clipper and the CFL path.  These tests pin that headroom.
    """

    def test_polygon_clip_at_nm_scale(self):
        import numpy as np

        from magnelio.geo._polygon_clip import clip_polygon_to_rect, polygon_area

        a = 10e-9  # 10 nm square
        poly = np.array([(0.0, 0.0), (a, 0.0), (a, a), (0.0, a)])
        assert polygon_area(poly) == pytest.approx(a * a, rel=1e-12)
        clipped = clip_polygon_to_rect(poly, (0.0, 0.0, a / 2, a / 2))
        assert polygon_area(clipped) == pytest.approx(a * a / 4.0, rel=1e-9)

    def test_courant_dt_at_nm_cells(self):
        import numpy as np

        from magnelio.mesh.grid import GridLines
        from magnelio.solver.stability import courant_dt

        n = np.linspace(0.0, 100e-9, 11)
        grid = GridLines(x=n, y=n, z=n)
        dt = courant_dt(grid)
        assert 0.0 < dt < 1e-16  # ~1.9e-17 s for 10 nm cells
