"""Tests for geometry.polygon_clip — area, clipping, point-in-polygon."""

import numpy as np
import numpy.testing as npt
import pytest

from magnelio.geo._polygon_clip import (
    clip_polygon_to_rect,
    line_polygon_intersection_length,
    point_in_polygon,
    polygon_area,
)

# ---------------------------------------------------------------------------
# polygon_area
# ---------------------------------------------------------------------------


class TestPolygonArea:
    def test_unit_square_ccw(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert polygon_area(sq) == pytest.approx(1.0)

    def test_unit_square_cw(self):
        sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert polygon_area(sq) == pytest.approx(-1.0)

    def test_triangle(self):
        tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        assert polygon_area(tri) == pytest.approx(6.0)

    def test_degenerate_line(self):
        line = np.array([[0, 0], [1, 1]], dtype=float)
        assert polygon_area(line) == 0.0

    def test_empty(self):
        assert polygon_area(np.empty((0, 2))) == 0.0

    def test_regular_hexagon(self):
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
        expected = 3 * np.sqrt(3) / 2  # area of regular hexagon with r=1
        assert abs(polygon_area(hex_verts)) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# clip_polygon_to_rect
# ---------------------------------------------------------------------------


class TestClipPolygonToRect:
    def test_polygon_fully_inside(self):
        sq = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
        clipped = clip_polygon_to_rect(sq, (0, 0, 1, 1))
        assert abs(polygon_area(clipped)) == pytest.approx(0.36)

    def test_polygon_fully_outside(self):
        sq = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=float)
        clipped = clip_polygon_to_rect(sq, (0, 0, 1, 1))
        assert len(clipped) == 0

    def test_unit_square_clipped_to_half(self):
        sq = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        clipped = clip_polygon_to_rect(sq, (0, 0, 1, 1))
        assert abs(polygon_area(clipped)) == pytest.approx(1.0)

    def test_triangle_clipped_to_rect(self):
        tri = np.array([[0, 0], [2, 0], [1, 2]], dtype=float)
        clipped = clip_polygon_to_rect(tri, (0, 0, 1, 1))
        assert abs(polygon_area(clipped)) > 0
        # All clipped vertices must be inside the rect
        assert np.all(clipped[:, 0] >= 0 - 1e-12)
        assert np.all(clipped[:, 0] <= 1 + 1e-12)
        assert np.all(clipped[:, 1] >= 0 - 1e-12)
        assert np.all(clipped[:, 1] <= 1 + 1e-12)

    def test_circle_clipped_area(self):
        """A circle inscribed in [0,2]x[0,2] clipped to [0,1]x[0,1]."""
        n = 200
        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        circle = np.column_stack([1 + np.cos(angles), 1 + np.sin(angles)])
        clipped = clip_polygon_to_rect(circle, (0, 0, 1, 1))
        # Quarter circle area = pi/4 ≈ 0.785
        assert abs(polygon_area(clipped)) == pytest.approx(np.pi / 4, rel=0.01)

    def test_empty_polygon(self):
        clipped = clip_polygon_to_rect(np.empty((0, 2)), (0, 0, 1, 1))
        assert len(clipped) == 0

    def test_degenerate_two_vertices(self):
        clipped = clip_polygon_to_rect(np.array([[0, 0], [1, 1]]), (0, 0, 1, 1))
        assert len(clipped) == 0

    def test_rect_touching_edge(self):
        """Polygon edge lies exactly on clipping boundary."""
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        clipped = clip_polygon_to_rect(sq, (0, 0, 1, 1))
        assert abs(polygon_area(clipped)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# point_in_polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    @pytest.fixture()
    def unit_square(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_center_inside(self, unit_square):
        assert point_in_polygon((0.5, 0.5), unit_square) is True

    def test_outside(self, unit_square):
        assert point_in_polygon((2.0, 0.5), unit_square) is False

    def test_above(self, unit_square):
        assert point_in_polygon((0.5, 1.5), unit_square) is False

    def test_below(self, unit_square):
        assert point_in_polygon((0.5, -0.5), unit_square) is False

    def test_triangle_inside(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=float)
        assert point_in_polygon((2, 1), tri) is True

    def test_triangle_outside(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=float)
        assert point_in_polygon((0, 3), tri) is False

    def test_concave_polygon(self):
        """L-shaped polygon: inside the notch should be outside."""
        L = np.array(
            [
                [0, 0],
                [2, 0],
                [2, 1],
                [1, 1],
                [1, 2],
                [0, 2],
            ],
            dtype=float,
        )
        assert point_in_polygon((0.5, 0.5), L) is True
        assert point_in_polygon((1.5, 1.5), L) is False  # inside the notch

    def test_empty_polygon(self):
        assert point_in_polygon((0, 0), np.empty((0, 2))) is False

    def test_circle_center(self):
        n = 100
        angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        assert point_in_polygon((0, 0), circle) is True
        assert point_in_polygon((2, 0), circle) is False


# ---------------------------------------------------------------------------
# line_polygon_intersection_length
# ---------------------------------------------------------------------------


class TestLinePolygonIntersectionLength:
    RECT = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=float)

    def test_full_inside(self):
        length = line_polygon_intersection_length(self.RECT, 1.5, 0, 4)
        assert length == pytest.approx(4.0)

    def test_partial_clip_right(self):
        length = line_polygon_intersection_length(self.RECT, 1.5, 2, 4)
        assert length == pytest.approx(2.0)

    def test_partial_clip_left(self):
        length = line_polygon_intersection_length(self.RECT, 1.5, -1, 2)
        assert length == pytest.approx(2.0)

    def test_line_outside(self):
        length = line_polygon_intersection_length(self.RECT, 5.0, 0, 4)
        assert length == pytest.approx(0.0)

    def test_line_below(self):
        length = line_polygon_intersection_length(self.RECT, -1.0, 0, 4)
        assert length == pytest.approx(0.0)

    def test_empty_polygon(self):
        empty = np.empty((0, 2), dtype=float)
        assert line_polygon_intersection_length(empty, 1.0, 0, 4) == 0.0

    def test_degenerate_segment(self):
        assert line_polygon_intersection_length(self.RECT, 1.5, 2, 2) == 0.0

    def test_circle_diameter(self):
        n = 256
        t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        circle = np.column_stack([np.cos(t), np.sin(t)])
        length = line_polygon_intersection_length(circle, 0.0, -2, 2)
        assert length == pytest.approx(2.0, abs=0.01)

    def test_circle_chord(self):
        n = 256
        t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        circle = np.column_stack([np.cos(t), np.sin(t)])
        length = line_polygon_intersection_length(circle, 0.5, -2, 2)
        expected = 2 * np.sqrt(1.0 - 0.25)  # 2*sqrt(0.75)
        assert length == pytest.approx(expected, abs=0.02)

    def test_circle_half_segment(self):
        n = 256
        t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
        circle = np.column_stack([np.cos(t), np.sin(t)])
        length = line_polygon_intersection_length(circle, 0.0, 0, 2)
        assert length == pytest.approx(1.0, abs=0.01)

    def test_triangle(self):
        tri = np.array([[0, 0], [4, 0], [2, 4]], dtype=float)
        length = line_polygon_intersection_length(tri, 2.0, -1, 5)
        assert length == pytest.approx(2.0)


class TestPointsInPolygonVectorised:
    """Verify the vectorised batched form matches the scalar reference."""

    def test_matches_scalar_on_unit_square(self):
        """Random batch of points: vectorised result == scalar result."""
        from magnelio.geo._polygon_clip import points_in_polygon

        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        rng = np.random.default_rng(42)
        px = rng.uniform(-0.5, 1.5, size=(20, 30))
        py = rng.uniform(-0.5, 1.5, size=(20, 30))
        from magnelio.geo._polygon_clip import point_in_polygon

        ref = np.array(
            [
                [point_in_polygon((px[i, j], py[i, j]), sq) for j in range(px.shape[1])]
                for i in range(px.shape[0])
            ]
        )
        vec = points_in_polygon(px, py, sq)
        assert vec.shape == px.shape
        np.testing.assert_array_equal(vec, ref)

    def test_handles_horizontal_edges(self):
        """Polygons with horizontal edges (y_i == y_j) must not divide-by-zero."""
        from magnelio.geo._polygon_clip import points_in_polygon

        L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
        px = np.array([[0.5, 1.5], [1.5, 0.5]])
        py = np.array([[0.5, 0.5], [1.5, 1.5]])
        result = points_in_polygon(px, py, L)
        np.testing.assert_array_equal(result, [[True, True], [False, True]])

    def test_empty_polygon(self):
        """Less than 3 vertices: all points return False."""
        from magnelio.geo._polygon_clip import points_in_polygon

        empty = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = points_in_polygon(np.array([0.5]), np.array([0.5]), empty)
        assert result.tolist() == [False]


class TestPointsInPolygonParallelKernel:
    """Bitwise gate: the Numba parallel kernel vs the NumPy edge-loop.
    The parallel kernel is an execution strategy only (same mul/div/add
    chain per crossing test, boolean output) — its result must equal
    the NumPy fallback path exactly, including on degenerate polygons.
    """

    @staticmethod
    def _numpy_path(monkeypatch, px, py, poly):
        import magnelio.geo._polygon_clip as pc

        monkeypatch.setattr(pc, "HAS_NUMBA", False)
        return pc.points_in_polygon(px, py, poly)

    def test_bitwise_equal_random_polygons(self, monkeypatch):
        import magnelio.geo._polygon_clip as pc

        if not pc.HAS_NUMBA:
            pytest.skip("Numba not available")
        rng = np.random.default_rng(1234)
        for n_vert in (3, 8, 64, 257):
            ang = np.sort(rng.uniform(0, 2 * np.pi, n_vert))
            r = rng.uniform(0.3, 1.0, n_vert)
            poly = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
            px = rng.uniform(-1.2, 1.2, 5000)
            py = rng.uniform(-1.2, 1.2, 5000)
            kernel = pc.points_in_polygon(px, py, poly)
            ref = self._numpy_path(monkeypatch, px, py, poly)
            monkeypatch.undo()
            npt.assert_array_equal(kernel, ref)

    def test_bitwise_equal_degenerate_features(self, monkeypatch):
        """Horizontal edges, duplicate vertices, CW winding, on-edge points."""
        import magnelio.geo._polygon_clip as pc

        if not pc.HAS_NUMBA:
            pytest.skip("Numba not available")
        polys = [
            np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float),
            np.array([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float),
        ]
        g = np.linspace(0.0, 2.0, 21)
        PX, PY = np.meshgrid(g, g, indexing="ij")
        for poly in polys:
            kernel = pc.points_in_polygon(PX, PY, poly)
            ref = self._numpy_path(monkeypatch, PX, PY, poly)
            monkeypatch.undo()
            assert kernel.shape == PX.shape
            npt.assert_array_equal(kernel, ref)

    def test_noncontiguous_input(self):
        """Sliced (non-contiguous) coordinate views dispatch correctly."""
        from magnelio.geo._polygon_clip import points_in_polygon

        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        base = np.linspace(-0.5, 1.5, 400).reshape(20, 20)
        px = base[::2, ::2]
        py = base.T[::2, ::2]
        result = points_in_polygon(px, py, sq)
        assert result.shape == px.shape
        expected = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
        inside_strict = (px > 0) & (px < 1) & (py > 0) & (py < 1)
        assert np.all(result[inside_strict])
        assert not np.any(result[~expected])


class TestPointsNearPolygon:
    """The boundary band that makes polygon membership inclusive."""

    def test_band_around_unit_square(self):
        from magnelio.geo._polygon_clip import points_near_polygon

        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        px = np.array([0.5, 0.5, 0.5, -0.05, -0.2, 1.05, 0.5])
        py = np.array([0.0, 0.05, 0.5, -0.05, 0.5, 1.05, -0.11])
        near = points_near_polygon(px, py, sq, 0.1)
        # on the edge, within the band, deep inside, outer corner within
        # the band (diagonal 0.07), too far outside, outer corner within
        # the band, just outside the band
        assert near.tolist() == [True, True, False, True, False, True, False]

    def test_preserves_shape_and_chunks(self):
        from magnelio.geo._polygon_clip import points_near_polygon

        tri = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
        rng = np.random.default_rng(7)
        px = rng.uniform(-1, 3, size=(40, 50))
        py = rng.uniform(-1, 3, size=(40, 50))
        whole = points_near_polygon(px, py, tri, 0.15)
        chunked = points_near_polygon(px, py, tri, 0.15, chunk=64)
        assert whole.shape == px.shape
        np.testing.assert_array_equal(whole, chunked)
        assert whole.any() and not whole.all()

    def test_degenerate_inputs(self):
        from magnelio.geo._polygon_clip import points_near_polygon

        one = np.array([[0.0, 0.0]])
        assert not points_near_polygon(np.array([0.0]), np.array([0.0]), one, 1.0).any()
        repeated = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        near = points_near_polygon(np.array([1.0, 1.5]), np.array([1.0, 1.0]), repeated, 0.2)
        assert near.tolist() == [True, False]
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert not points_near_polygon(np.array([0.0]), np.array([0.0]), sq, 0.0).any()


def _random_polygons(rng):
    """Closed random outlines: a star, a thin sliver, a rectangle with a repeated vertex."""
    n = 9
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 3.0, n)
    star = np.column_stack((3.0 + radii * np.cos(angles), 3.0 + radii * np.sin(angles)))
    sliver = np.array([(0.2, 0.2), (7.9, 0.25), (7.8, 0.35)])
    rect = np.array([(1.0, 4.0), (5.0, 4.0), (5.0, 4.0), (5.0, 6.5), (1.0, 6.5)])
    return [star, sliver, rect]


@pytest.mark.parametrize("numba", [True, False])
def test_points_near_polygon_grid_matches_the_point_version(monkeypatch, numba):
    from magnelio.geo import _polygon_clip as pc

    if numba and not pc.HAS_NUMBA:
        pytest.skip("numba not installed")
    monkeypatch.setattr(pc, "HAS_NUMBA", numba)
    rng = np.random.default_rng(5)
    u = np.sort(rng.uniform(0.0, 8.0, 211))
    v = np.sort(rng.uniform(0.0, 7.0, 157))
    UU, VV = np.meshgrid(u, v, indexing="ij")
    for poly in _random_polygons(rng):
        for tol in (1e-12, 0.01, 0.07, 0.5, 3.0):
            got = pc.points_near_polygon_grid(u, v, poly, tol)
            expected = pc.points_near_polygon(UU, VV, poly, tol)
            np.testing.assert_array_equal(got, expected)
            assert got.dtype == bool and got.shape == UU.shape
    assert not pc.points_near_polygon_grid(u, v, _random_polygons(rng)[0], 0.0).any()
    assert pc.points_near_polygon_grid(u[:0], v, _random_polygons(rng)[0], 0.1).shape == (0, 157)
    assert not pc.points_near_polygon_grid(u, v, np.zeros((1, 2)), 0.1).any()
