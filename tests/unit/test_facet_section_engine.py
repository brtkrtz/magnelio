"""Facet path of the planar section engine (free-form faces).

A shape with at least one free-form face (B-spline, extrusion, ...) is
represented by its triangulation at the section deflection and sectioned
without the kernel Boolean.  The gate is the same as for the exact planar
path: the |area| multiset of the section polygons against
``cross_section_polygons`` — the kernel's exact curves are themselves
tessellated at the same deflection, so both sides carry the same
chord-error class and agree to a fraction of it.
"""

from __future__ import annotations

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo
from magnelio.geo._occ_backend import _PlanarSectionEngine, cross_section_polygons
from magnelio.geo._polygon_clip import polygon_area

DEFLECTION = 1e-4


def _paraboloid(r, phi):
    x = 0.05 + r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, (x * x + y * y) / (4 * 0.12)


@pytest.fixture(scope="module")
def dish_solid():
    sheet = geo.Surface.parametric(_paraboloid, u=(0.0, 0.06), v=(0.0, 2 * np.pi), samples=(16, 32))
    return sheet.extruded(vector=(0.0, 0.0, -0.02), material="pec")


@pytest.fixture(scope="module")
def horn():
    def rect(x, a, b):
        return geo.Face(
            normal="x",
            points=((-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)),
            position=x,
        )

    return geo.Loft(rect(0.0, 0.02, 0.01), rect(0.06, 0.06, 0.045), blend="ruled", material="pec")


def _engine(shape, deflection=DEFLECTION):
    return _PlanarSectionEngine(shape._occ_shape(1.0), scale=1.0, deflection=deflection)


def _areas(polys):
    return sorted(abs(polygon_area(p)) for p in polys)


def _planes(shape, n=6, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = shape.bounding_box()
    for axis in range(3):
        for f in rng.uniform(0.05, 0.95, n):
            yield axis, float(lo[axis] + f * (hi[axis] - lo[axis]))


def _perimeter(poly):
    return float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).sum())


def _gate(shape, n=6, exact=False):
    """Engine against the kernel path: every |area| within the deflection
    times the polygon's perimeter (both are chord-accurate to the
    deflection); with *exact* the engine agrees to 1e-9 relative (planar
    B-spline faces, where neither side approximates anything)."""
    eng = _engine(shape)
    assert eng.enabled and eng.facetted
    occ = shape._occ_shape(1.0)
    for axis, pos in _planes(shape, n):
        mine = eng.section(axis, pos)
        assert mine is not None, (axis, pos)
        ref = cross_section_polygons(occ, "xyz"[axis], pos, deflection=DEFLECTION, scale=1.0)
        a, b = _areas(mine), _areas(ref)
        assert len(a) == len(b), (axis, pos, len(a), len(b))
        tol = 1e-9 * max(b) if exact else DEFLECTION * max(_perimeter(p) for p in ref)
        for x, y in zip(a, b):
            assert abs(x - y) <= tol, (axis, pos, x, y)


def _slab_area(shape, axis, pos, width=2e-5):
    """Kernel reference without tessellation: the volume of a thin slab
    of the solid around the plane, divided by the slab width."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.GProp import GProp_GProps

    lo = [-1.0, -1.0, -1.0]
    hi = [1.0, 1.0, 1.0]
    lo[axis] = pos - width / 2
    hi[axis] = pos + width / 2
    box = BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()
    props = GProp_GProps()
    brepgprop.VolumeProperties(BRepAlgoAPI_Common(shape._occ_shape(1.0), box).Shape(), props, 1e-9)
    return props.Mass() / width


class TestAdmission:
    def test_free_form_shape_is_facetted(self, dish_solid):
        eng = _engine(dish_solid)
        assert eng.enabled and eng.facetted

    def test_analytic_shapes_keep_their_paths(self):
        brick = geo.Brick(origin=(0, 0, 0), size=(1, 1, 1), material="pec")
        eng = _engine(brick)
        assert eng.enabled and not eng.facetted
        cyl = geo.Cylinder(origin=(0, 0, 0), radius=0.5, height=1.0, axis="z", material="pec")
        assert not _engine(cyl).facetted

    def test_without_deflection_the_kernel_keeps_free_form(self, dish_solid):
        eng = _engine(dish_solid, deflection=None)
        assert not eng.facetted

    def test_environment_switch_disables_the_facet_path(self, dish_solid, monkeypatch):
        monkeypatch.setenv("MAGNELIO_FACET_SECTIONS", "0")
        eng = _engine(dish_solid)
        assert not eng.facetted


class TestPolygonGate:
    def test_paraboloid_prism_matches_the_kernel(self, dish_solid):
        _gate(dish_solid)

    def test_paraboloid_prism_matches_the_exact_area(self, dish_solid):
        # Lifted and refined polygons follow the exact surface: the
        # area agrees with the untessellated kernel reference to a few
        # 1e-4 on every cut, including the shallow z-cuts.
        eng = _engine(dish_solid)
        for axis, pos in ((2, 0.004), (2, 0.017), (2, 0.0207), (0, 0.106), (1, 0.03)):
            mine = sum(_areas(eng.section(axis, pos)))
            ref = _slab_area(dish_solid, axis, pos)
            assert abs(mine - ref) <= 5e-4 * ref, (axis, pos, mine, ref)

    def test_ruled_loft_is_exact(self, horn):
        # Ruled quads between parallel rectangles are planar: the
        # triangles reproduce them exactly and the kernel's B-spline
        # section curves are straight lines.
        _gate(horn, exact=True)

    def test_boolean_with_free_form_faces(self, dish_solid):
        box = geo.Brick(origin=(-0.05, -0.1, -0.05), size=(0.2, 0.2, 0.1), material="air")
        _gate(geo.Difference(box, dish_solid))

    def test_plane_through_a_triangulation_vertex(self, dish_solid):
        eng = _engine(dish_solid)
        occ = dish_solid._occ_shape(1.0)
        for axis in range(3):
            # An interior vertex: the middle of the edge table.
            pos = float(eng._e_p1[eng._e_p1.shape[0] // 2, axis])
            mine = eng.section(axis, pos)
            assert mine is not None
            ref = cross_section_polygons(occ, "xyz"[axis], pos, deflection=DEFLECTION, scale=1.0)
            a, b = _areas(mine), _areas(ref)
            assert len(a) == len(b)
            tol = DEFLECTION * max(_perimeter(p) for p in ref)
            for x, y in zip(a, b):
                assert abs(x - y) <= tol

    def test_outside_the_shape_is_empty(self, dish_solid):
        eng = _engine(dish_solid)
        lo, hi = dish_solid.bounding_box()
        assert eng.section(2, hi[2] + 0.1) == []
        assert eng.section(0, lo[0] - 0.1) == []


def _edges_touching(cells: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Mask over the flat edge array of every edge bordering a marked
    cell (Ex, Ey, Ez blocks in ``pec_mask_edges`` order)."""
    ex = np.zeros((nx, ny + 1, nz + 1), dtype=bool)
    ey = np.zeros((nx + 1, ny, nz + 1), dtype=bool)
    ez = np.zeros((nx + 1, ny + 1, nz), dtype=bool)
    for i, j, k in zip(*np.nonzero(cells)):
        ex[i, j : j + 2, k : k + 2] = True
        ey[i : i + 2, j, k : k + 2] = True
        ez[i : i + 2, j : j + 2, k] = True
    return np.concatenate([ex.ravel(), ey.ravel(), ez.ravel()])


class TestMeshGate:
    """The mesher's arrays with the facet path against the exact geometry.

    The reference is analytic, not the kernel path: on this dish the
    kernel's section contours of a z-plane — the annulus between the top
    and bottom surface — come back with the SAME winding, so the area
    kernels add the hole to the disc and every dual face inside the hole
    is booked fully PEC.  The facet path's contours counter-rotate by
    construction.
    """

    R_DISH = 10e-3
    CURV = 20e-3  # z = r² / CURV
    T = 2e-3

    @classmethod
    def _mesh(cls):
        small = geo.Surface.parametric(
            lambda r, phi: (r * np.cos(phi), r * np.sin(phi), r * r / cls.CURV),
            u=(0.0, cls.R_DISH),
            v=(0.0, 2 * np.pi),
            samples=(12, 24),
        )
        reflector = small.extruded(vector=(0.0, 0.0, -cls.T), material="pec")
        box = geo.Brick(origin=(-15e-3, -15e-3, -5e-3), size=(30e-3, 30e-3, 15e-3), material="air")
        model = mio.GeometryModel()
        model.add(geo.Difference(box, reflector))
        model.add(reflector)
        return mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=1.5e-3), f_max=20e9)

    @classmethod
    def _exact_free_fraction(cls, x0, x1, y0, y1, z, n=60):
        """Free (non-PEC) area fraction of a z-normal dual face: the PEC
        is the extruded dish, r² / CURV - T <= z <= r² / CURV, r <= R."""
        xs = (np.arange(n) + 0.5) / n * (x1 - x0) + x0
        ys = (np.arange(n) + 0.5) / n * (y1 - y0) + y0
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        r2 = xx * xx + yy * yy
        pec = (r2 <= cls.R_DISH**2) & (r2 / cls.CURV - cls.T <= z) & (z <= r2 / cls.CURV)
        return 1.0 - pec.mean()

    def test_grid_and_cells_agree_with_the_kernel_path(self, monkeypatch):
        facet = self._mesh()
        monkeypatch.setenv("MAGNELIO_FACET_SECTIONS", "0")
        kernel = self._mesh()
        np.testing.assert_array_equal(facet.grid.x, kernel.grid.x)
        np.testing.assert_array_equal(facet.grid.y, kernel.grid.y)
        np.testing.assert_array_equal(facet.grid.z, kernel.grid.z)
        # Both classifications are chord-accurate; a cell centre within
        # that band of the surface may land on either side.
        cells = facet.material_id != kernel.material_id
        assert cells.sum() <= 2e-3 * cells.size

    def test_dual_face_fractions_follow_the_exact_geometry(self):
        mesh = self._mesh()
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
                    if cat[e] not in (1, 2) or not np.isfinite(f_a[e]):
                        continue
                    z = 0.5 * (gz[k] + gz[k + 1])
                    exact = self._exact_free_fraction(xc[i - 1], xc[i], yc[j - 1], yc[j], z)
                    diffs.append(abs(f_a[e] - exact))
        diffs = np.asarray(diffs)
        assert diffs.size > 200
        # 60 x 60 samples resolve a fraction to ~1e-2 on a face the
        # boundary crosses; the mean is far below that.
        assert diffs.mean() <= 5e-3, diffs.mean()
        assert diffs.max() <= 5e-2, diffs.max()
