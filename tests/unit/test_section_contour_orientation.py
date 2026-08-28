"""Section contours are wound by nesting parity before the area kernels.

The kernels sum signed areas per shape; a hole therefore has to run
against its outer boundary.  ``BRepAlgoAPI_Section`` does not orient
contours that way — a tube's bore and rim came back with the same
winding and every dual face at the bore wall was booked fully PEC, the
conformal correction at the inner wall of any hollow body silently lost.
"""

from __future__ import annotations

import numpy as np

import magnelio as mio
from magnelio import geo
from magnelio.geo._polygon_clip import orient_nested_contours, polygon_area


def _square(c, half, ccw=True):
    p = np.array(
        [(c - half, c - half), (c + half, c - half), (c + half, c + half), (c - half, c + half)]
    )
    return p if ccw else p[::-1]


class TestOrientNestedContours:
    def test_hole_runs_against_its_outer_boundary(self):
        outer = _square(0.0, 1.0, ccw=True)
        hole = _square(0.0, 0.5, ccw=True)
        a, b = orient_nested_contours([outer, hole])
        assert polygon_area(a) > 0 and polygon_area(b) < 0

    def test_island_in_a_hole_is_positive_again(self):
        outer = _square(0.0, 1.0, ccw=False)
        hole = _square(0.0, 0.5, ccw=False)
        island = _square(0.0, 0.2, ccw=False)
        signs = [np.sign(polygon_area(p)) for p in orient_nested_contours([outer, hole, island])]
        assert signs == [1, -1, 1]

    def test_disjoint_contours_are_all_positive(self):
        a, b = orient_nested_contours([_square(-2.0, 0.5, ccw=False), _square(2.0, 0.5, ccw=True)])
        assert polygon_area(a) > 0 and polygon_area(b) > 0

    def test_coincident_pair_keeps_its_windings(self):
        a = _square(0.0, 1.0, ccw=True)
        b = _square(0.0, 1.0, ccw=False)
        out = orient_nested_contours([a, b])
        assert polygon_area(out[0]) > 0 and polygon_area(out[1]) < 0

    def test_many_holes_with_islands_shuffled(self):
        rng = np.random.default_rng(1)
        outer = _square(0.0, 60.0, ccw=False)
        holes, islands = [], []
        for i in range(8):
            for j in range(8):
                c = (-42.0 + 12.0 * i, -42.0 + 12.0 * j)
                hole = np.array(
                    [
                        (c[0] - 4, c[1] - 4),
                        (c[0] + 4, c[1] - 4),
                        (c[0] + 4, c[1] + 4),
                        (c[0] - 4, c[1] + 4),
                    ]
                )
                holes.append(hole if rng.random() < 0.5 else hole[::-1])
                if (i + j) % 3 == 0:
                    island = np.array(
                        [
                            (c[0] - 1, c[1] - 1),
                            (c[0] + 1, c[1] - 1),
                            (c[0] + 1, c[1] + 1),
                            (c[0] - 1, c[1] + 1),
                        ]
                    )
                    islands.append(island if rng.random() < 0.5 else island[::-1])
        polys = [outer, *holes, *islands]
        expected = [1] + [-1] * len(holes) + [1] * len(islands)
        order = rng.permutation(len(polys))
        out = orient_nested_contours([polys[k] for k in order])
        assert [int(np.sign(polygon_area(p))) for p in out] == [expected[k] for k in order]

    def test_empty_and_single(self):
        assert orient_nested_contours([]) == []
        (only,) = orient_nested_contours([_square(0.0, 1.0, ccw=False)])
        assert polygon_area(only) > 0


class TestHollowBodyDualFaces:
    """A PEC tube in air: dual faces at the bore wall keep their free
    area (the kernel path, analytic surfaces)."""

    R_OUT, R_IN, H = 10e-3, 6e-3, 6e-3

    def test_bore_wall_fractions_follow_the_exact_geometry(self):
        tube = geo.Cylinder(
            origin=(0, 0, 0),
            radius=self.R_OUT,
            height=self.H,
            axis="z",
            inner_radius=self.R_IN,
            material="pec",
        )
        box = geo.Brick(origin=(-15e-3, -15e-3, -3e-3), size=(30e-3, 30e-3, 12e-3), material="air")
        model = mio.GeometryModel()
        model.add(geo.Difference(box, tube))
        model.add(tube)
        mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=1.5e-3), f_max=20e9)
        nx, ny, nz = mesh.Nx, mesh.Ny, mesh.Nz
        n_ex = nx * (ny + 1) * (nz + 1)
        n_ey = (nx + 1) * ny * (nz + 1)
        gx, gy, gz = mesh.grid.x, mesh.grid.y, mesh.grid.z
        xc = 0.5 * (gx[:-1] + gx[1:])
        yc = 0.5 * (gy[:-1] + gy[1:])
        f_a = mesh.edge_material.f_A
        cat = mesh.edge_material.category
        diffs = []
        for i in range(1, nx):
            for j in range(1, ny):
                for k in range(nz):
                    e = n_ex + n_ey + i * (ny + 1) * nz + j * nz + k
                    z = 0.5 * (gz[k] + gz[k + 1])
                    if cat[e] not in (1, 2) or not np.isfinite(f_a[e]) or not 0 < z < self.H:
                        continue
                    xs = (np.arange(60) + 0.5) / 60 * (xc[i] - xc[i - 1]) + xc[i - 1]
                    ys = (np.arange(60) + 0.5) / 60 * (yc[j] - yc[j - 1]) + yc[j - 1]
                    xx, yy = np.meshgrid(xs, ys, indexing="ij")
                    r2 = xx * xx + yy * yy
                    exact = 1.0 - ((r2 <= self.R_OUT**2) & (r2 >= self.R_IN**2)).mean()
                    diffs.append(abs(f_a[e] - exact))
        diffs = np.asarray(diffs)
        assert diffs.size > 500
        # Was: mean 0.12, max 1.0 (bore-wall faces fully PEC).
        assert diffs.mean() <= 5e-3, diffs.mean()
        assert diffs.max() <= 5e-2, diffs.max()
