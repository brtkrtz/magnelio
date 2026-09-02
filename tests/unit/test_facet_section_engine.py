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


# ---------------------------------------------------------------------------
# A free-form face must not move the trace of an analytic one
# ---------------------------------------------------------------------------

#: Cell size and the section deflection the mesher derives from it.
CELL = 2.5e-4
BLEND_DEFLECTION = CELL * 1e-2
BORE_R = 3.5e-3
BORE_LO = 0.0
BORE_HI = 20e-3


def _cap(y: float, side: float, x0: float = 0.0):
    half = 0.5 * side
    return geo.Face(
        normal="y",
        points=(
            (x0 - half, -half),
            (x0 + half, -half),
            (x0 + half, half),
            (x0 - half, half),
        ),
        position=y,
    )


def _cell_areas(polys, window) -> np.ndarray:
    """Covered area of every cell of a ``CELL`` grid over *window*.

    *window* is ``(u0, v0, u1, v1)`` in the section plane's own
    coordinates — the quantity ``compute_face_material_areas`` books
    per dual face, computed here directly from the section polygons.
    """
    from magnelio.geo._polygon_clip import clip_polygon_to_rect

    u0, v0, u1, v1 = window
    nu = int(round((u1 - u0) / CELL))
    nv = int(round((v1 - v0) / CELL))
    out = np.zeros((nu, nv))
    for i in range(nu):
        for j in range(nv):
            rect = (u0 + i * CELL, v0 + j * CELL, u0 + (i + 1) * CELL, v0 + (j + 1) * CELL)
            total = 0.0
            for poly in polys:
                clipped = clip_polygon_to_rect(poly, rect)
                if clipped is not None and len(clipped) >= 3:
                    total += abs(polygon_area(clipped))
            out[i, j] = total
    return out


class TestFreeFormReachIsLocal:
    """A body a cell holds none of must not change that cell's masses.

    The facet path represents a whole shape by its triangulation as soon
    as one of its faces is free-form, so a blend fused onto a cylinder
    took the cylinder's own faces off their exact geometry too — and a
    triangulated cylinder cuts differently depending on where the plane
    falls between its node rows.  Cells nowhere near the blend then
    moved by up to a per-cent, and a port sitting on them lost its exact
    termination with nothing warning about it.
    """

    @pytest.fixture(scope="class")
    def shapes(self):
        """The cylinder alone, and the same cylinder sharing its solid
        with a free-form blend that stops well below the window used."""
        cylinder = geo.Cylinder(
            radius=BORE_R,
            origin=(0.0, 0.0, BORE_LO),
            axis="y",
            height=BORE_HI - BORE_LO,
            material="pec",
        )
        blend = geo.Loft(_cap(-6e-3, 8e-3), _cap(0.5e-3, 4e-3), blend="ruled", material="pec")
        return cylinder, cylinder + blend

    @staticmethod
    def _engines(shapes):
        plain, fused = shapes
        a = _engine(plain, BLEND_DEFLECTION)
        b = _engine(fused, BLEND_DEFLECTION)
        assert not a.facetted and b.facetted
        return a, b

    def test_the_blend_shares_the_solid_and_facets_it(self, shapes):
        plain, fused = self._engines(shapes)
        assert plain.enabled and fused.enabled

    def test_cells_away_from_the_blend_are_unchanged(self, shapes):
        """The acceptance criterion: identical masses where the body is
        not.  Cuts along the cylinder axis, 8 mm and further above the
        blend's last material."""
        plain, fused = self._engines(shapes)
        window = (8e-3, -4e-3, 18e-3, 4e-3)
        worst = 0.0
        for pos in (0.5e-3, 1.3e-3, 2.1e-3, 3.4e-3):
            a = _cell_areas(plain.section(0, pos), window)
            b = _cell_areas(fused.section(0, pos), window)
            assert (a > 0.5 * CELL * CELL).any()
            # Deviations are measured against the cell, the way a
            # conformal fraction reads them.
            worst = max(worst, float(np.abs(b - a).max()) / (CELL * CELL))
        assert worst <= 1e-12, worst

    def test_the_trace_repeats_along_the_axis(self, shapes):
        """Translational invariance — what the exact port termination
        consumes.  A cut across the axis is a circle at every height, so
        the cells it books must not depend on the height."""
        _, fused = self._engines(shapes)
        window = (-4e-3, -4e-3, 4e-3, 4e-3)
        ref = None
        for pos in (10e-3, 10.13e-3, 10.25e-3, 10.37e-3):
            cells = _cell_areas(fused.section(1, pos), window)
            if ref is None:
                ref = cells
                assert (ref > 0.5 * CELL * CELL).any()
                continue
            assert float(np.abs(cells - ref).max()) / (CELL * CELL) <= 1e-12

    def test_the_repeating_trace_is_the_right_circle(self, shapes):
        """Invariance must not be bought with a wrong radius: the area
        of the cross cut stays inside the chord budget of the section."""
        _, fused = self._engines(shapes)
        polys = fused.section(1, 10e-3)
        area = sum(abs(polygon_area(p)) for p in polys)
        assert abs(area / (np.pi * BORE_R**2) - 1.0) <= 1e-3, area


# ---------------------------------------------------------------------------
# The same, on a cylinder whose axis is oblique to the grid
# ---------------------------------------------------------------------------

#: Axis (0, 1, 1)/sqrt(2).  Two things the axis-aligned fixture above
#: cannot see follow from it: every cut normal to y is a genuine ellipse
#: rather than a circle, so translational invariance is no longer almost
#: free; and one cell of plane travel in y moves the trace by exactly one
#: cell in z (the plane advances by CELL*sqrt(2) along the axis, whose z
#: component is 1/sqrt(2)), so the invariance can be read off a shifted
#: window with no interpolation.
TILT_AXIS = (0.0, 2.0**-0.5, 2.0**-0.5)
TILT_HEIGHT = 60e-3
#: Cut well above the blend's last material, and far from either cap.
TILT_CUT = 10e-3
TILT_WINDOW = (-4e-3, 4.5e-3, 4e-3, 15.5e-3)


class TestObliqueFreeFormReachIsLocal:
    """:class:`TestFreeFormReachIsLocal` on an oblique conic run.

    With the axis on a grid direction every cross cut is the same
    circle, so the trace repeats as soon as the compression fires at
    all, and the fixture above misses that invariant by 1.3e-05 of a
    cell without the fix.  Tilted, the cut is a genuine ellipse whose
    node rows fall differently on every plane, and the same measurement
    misses it by 3.3e-04 — the size a port termination on such a body
    would see.

    Both shapes carry the same free-form neighbour, so both are on the
    facet path and the blend is the only difference between them: what
    is compared is the reach of the blend, not the reach of the path.
    """

    @pytest.fixture(scope="class")
    def engines(self):
        cylinder = geo.Cylinder(
            radius=BORE_R,
            origin=(0.0, 0.0, 0.0),
            axis=TILT_AXIS,
            height=TILT_HEIGHT,
            material="pec",
        )
        # Free-form, far enough away to share nothing but the path.
        neighbour = geo.Loft(
            _cap(-6e-3, 4e-3, 30e-3),
            _cap(0.5e-3, 2e-3, 30e-3),
            blend="ruled",
            material="pec",
        )
        blend = geo.Loft(_cap(-6e-3, 8e-3), _cap(0.5e-3, 4e-3), blend="ruled", material="pec")
        plain = _engine(cylinder + neighbour, BLEND_DEFLECTION)
        fused = _engine(cylinder + neighbour + blend, BLEND_DEFLECTION)
        assert plain.enabled and plain.facetted
        assert fused.enabled and fused.facetted
        return plain, fused

    def test_cells_away_from_the_blend_are_unchanged(self, engines):
        """Identical masses where the blend is not, on planes that fall
        at four different places between the triangulation's node rows.
        """
        plain, fused = engines
        worst = 0.0
        for pos in (TILT_CUT, TILT_CUT + 0.09e-3, TILT_CUT + 0.17e-3, TILT_CUT + 0.23e-3):
            a = _cell_areas(plain.section(1, pos), TILT_WINDOW)
            b = _cell_areas(fused.section(1, pos), TILT_WINDOW)
            assert (a > 0.5 * CELL * CELL).any()
            worst = max(worst, float(np.abs(b - a).max()) / (CELL * CELL))
        assert worst <= 1e-12, worst

    def test_the_trace_repeats_along_the_cylinder_axis(self, engines):
        """Translational invariance along the body's own axis: one cell
        of plane travel in y is one cell of trace travel in z, so the
        cells of a window that follows the trace must not move."""
        _, fused = engines
        u0, v0, u1, v1 = TILT_WINDOW
        ref = None
        for k in range(4):
            window = (u0, v0 + k * CELL, u1, v1 + k * CELL)
            cells = _cell_areas(fused.section(1, TILT_CUT + k * CELL), window)
            if ref is None:
                ref = cells
                assert (ref > 0.5 * CELL * CELL).any()
                continue
            worst = float(np.abs(cells - ref).max()) / (CELL * CELL)
            assert worst <= 1e-9, worst

    def test_the_repeating_trace_is_the_right_ellipse(self, engines):
        """Invariance must not be bought with a wrong conic: the cut
        normal to y has the area of the ellipse the axis tilt implies."""
        _, fused = engines
        polys = fused.section(1, TILT_CUT)
        area = sum(abs(polygon_area(p)) for p in polys)
        exact = np.pi * BORE_R**2 / abs(TILT_AXIS[1])
        assert abs(area / exact - 1.0) <= 1e-3, area


# ---------------------------------------------------------------------------
# Sphere, cone and torus faces of a facetted shape
# ---------------------------------------------------------------------------

#: The three surfaces the exact engine does not speak, each with its
#: axis on y, and the planes of the cylinder fixtures above: cuts along
#: y at four x positions, cuts across y at four heights around 10 mm.
QUADRIC_PLANES = {
    0: ((0.5e-3, 1.3e-3, 2.1e-3, 3.4e-3), (4e-3, -4e-3, 16e-3, 4e-3)),
    1: ((10e-3, 10.13e-3, 10.25e-3, 10.37e-3), (-4e-3, -4e-3, 4e-3, 4e-3)),
}
TORUS_R = 1.2e-3


def _quadric_body(name):
    if name == "sphere":
        return geo.Sphere(center=(0.0, 10e-3, 0.0), radius=BORE_R, material="pec")
    if name == "cone":
        return geo.Cone(
            origin=(0.0, 0.0, 0.0),
            bottom_radius=4.5e-3,
            top_radius=3e-3,
            height=20e-3,
            axis="y",
            material="pec",
        )
    return geo.Torus(
        center=(0.0, 10e-3, 0.0),
        major_radius=BORE_R,
        minor_radius=TORUS_R,
        axis="y",
        material="pec",
    )


def _quadric_residual(name, body, p):
    """Implicit equation of *body*'s curved surface at the 3D points *p*
    (zero on the surface), in meters."""
    if name == "sphere":
        return np.linalg.norm(p - np.array(body.center), axis=1) - body.radius
    rel = p - np.array(body.origin if name == "cone" else body.center)
    h = rel[:, 1]
    rho = np.hypot(rel[:, 0], rel[:, 2])
    if name == "cone":
        slope = (body.top_radius - body.bottom_radius) / body.height
        return rho - (body.bottom_radius + h * slope)
    return np.hypot(rho - body.major_radius, h) - body.minor_radius


def _converged_reference(body, axis, pos):
    """Kernel section tessellated three decades below the deflection
    (built at scale 1e3 to get under the kernel path's 1e-7 clamp)."""
    return cross_section_polygons(
        body._occ_shape(1e3), "xyz"[axis], pos, deflection=BLEND_DEFLECTION * 1e-3, scale=1e3
    )


class TestQuadricFacesOfAFacettedShape:
    """Sphere, cone and torus faces are answered from their own surface.

    The DD-240 repair was cylinders only; these three kept the one-step
    parametric lift of DD-199.  What was open under KB-042 turned out to
    be two smaller things: the lift needs the face parameters, which do
    not describe a point at a sphere's pole (the crossing there stayed
    on the chord, a full deflection off the surface), and the difference
    to the kernel path is the kernel's own tessellation at the
    deflection, not a reach of the free-form neighbour.  The projection
    onto the implicit surface fixes the first and the tests here pin
    both: a far body does not reach, every vertex lies on the surface
    to rounding, the pole included, and the areas are the closed forms.
    """

    @pytest.fixture(scope="class", params=["sphere", "cone", "torus"])
    def case(self, request):
        name = request.param
        body = _quadric_body(name)
        far = geo.Loft(_cap(-6e-3, 4e-3, 40e-3), _cap(0.5e-3, 2e-3, 40e-3), blend="ruled")
        farther = geo.Loft(_cap(-6e-3, 4e-3, 45e-3), _cap(0.5e-3, 2e-3, 45e-3), blend="ruled")
        a = _engine(body + far, BLEND_DEFLECTION)
        b = _engine(body + farther, BLEND_DEFLECTION)
        assert a.enabled and a.facetted and b.enabled and b.facetted
        assert not _engine(body, BLEND_DEFLECTION).facetted
        return name, body, a, b

    def test_a_far_body_does_not_reach(self, case):
        """The KB-042 claim, measured: the cells of the quadric are the
        same whichever far free-form body shares its solid."""
        _, _, a, b = case
        worst = 0.0
        for axis, (positions, window) in QUADRIC_PLANES.items():
            for pos in positions:
                cells_a = _cell_areas(a.section(axis, pos), window)
                cells_b = _cell_areas(b.section(axis, pos), window)
                assert (cells_a > 0.5 * CELL * CELL).any()
                worst = max(worst, float(np.abs(cells_a - cells_b).max()) / (CELL * CELL))
        assert worst <= 1e-12, worst

    def test_every_vertex_lies_on_the_surface(self, case):
        """The projection's contract: no second-order residual, no
        unprojected pole.  Before it, the sphere's worst vertex sat a
        full deflection off (2.4e-6 m) and the torus' 1.5e-9 m."""
        name, body, a, _ = case
        worst = 0.0
        for axis, (positions, _) in QUADRIC_PLANES.items():
            u_idx, v_idx = a._UV[axis]
            for pos in positions:
                for poly in a.section(axis, pos):
                    p = np.empty((len(poly), 3))
                    p[:, axis] = pos
                    p[:, u_idx] = poly[:, 0]
                    p[:, v_idx] = poly[:, 1]
                    if name == "cone":  # the planar caps are not on the cone
                        on_cap = (np.abs(p[:, 1]) <= 1e-9) | (np.abs(p[:, 1] - 20e-3) <= 1e-9)
                        p = p[~on_cap]
                    worst = max(worst, float(np.abs(_quadric_residual(name, body, p)).max()))
        assert worst <= 1e-12 * BORE_R, worst

    def test_the_areas_are_the_closed_forms(self, case):
        """Chord-accurate at the refinement budget, a tenth of the
        deflection: sphere circles, cone circles, torus annuli.  A
        circle of radius rho tessellated at sagitta s books (4/3) s/rho
        less than its area; the gate is twice that at the smallest
        radius of curvature of the trace."""
        name, body, a, _ = case
        for axis, (positions, _) in QUADRIC_PLANES.items():
            for pos in positions:
                # Signed: the hole of an annulus counter-rotates.
                area = abs(sum(polygon_area(p) for p in a.section(axis, pos)))
                if name == "sphere":
                    off = pos if axis == 0 else pos - 10e-3
                    rho = (BORE_R**2 - off**2) ** 0.5
                    exact = np.pi * rho**2
                elif name == "cone" and axis == 1:
                    rho = 4.5e-3 - 1.5e-3 * pos / 20e-3
                    exact = np.pi * rho**2
                elif name == "torus" and axis == 1:
                    half = (TORUS_R**2 - (pos - 10e-3) ** 2) ** 0.5
                    rho = BORE_R - half
                    exact = 4 * np.pi * BORE_R * half
                else:
                    continue
                gate = 2 * (4 / 3) * (BLEND_DEFLECTION / 10) / rho
                assert abs(area / exact - 1.0) <= gate, (name, axis, pos, area / exact - 1.0)

    def test_the_sphere_pole_is_projected_like_any_other_point(self):
        """The plane through both poles, cell by cell against a converged
        kernel reference: 2.24e-3 of a cell with the parametric lift
        (the pole cell), 6.3e-4 with the projection — the level of the
        planes that miss the poles."""
        body = _quadric_body("sphere")
        far = geo.Loft(_cap(-6e-3, 4e-3, 40e-3), _cap(0.5e-3, 2e-3, 40e-3), blend="ruled")
        eng = _engine(body + far, BLEND_DEFLECTION)
        window = QUADRIC_PLANES[1][1]
        cells = _cell_areas(eng.section(1, 10e-3), window)
        truth = _cell_areas(_converged_reference(body, 1, 10e-3), window)
        assert (truth > 0.5 * CELL * CELL).any()
        assert float(np.abs(cells - truth).max()) / (CELL * CELL) <= 1e-3


# ---------------------------------------------------------------------------
# A conic run whose exact arc cannot be built
# ---------------------------------------------------------------------------

#: A thin, long body whose axis sits a fraction of a degree off the grid
#: — a bond wire, a via, a barrel out of an imported STEP file.  The
#: trace of a plane nearly parallel to that axis is a conic so stretched
#: that tessellating it at the in-plane sagitta budget runs into the
#: segment cap of ``_cylinder_arc``, which then declines.
#:
#: Retuned 2026-09-01 from ``WIRE_TILT = 3e-3`` / ``WIRE_LENGTH = 100e-3``
#: / blend at 80-95 mm.  Those constants reached the segment cap only
#: through the sagitta term's wrong ``abs(c_n) ** 3``; with the exponent
#: at its derived 1 the same body tessellates in 12001 segments and
#: builds.  The decline now comes from the ``radians(5) * abs(c_n)``
#: angular cap, which needs ``abs(c_n) < span / 100_000 / radians(5)``
#: -- 3.6e-4 for the half turn this plane cuts -- and the run's reach
#: ``WIRE_R / WIRE_TILT`` grows with it, so the blend moves out to match.
WIRE_R = 0.2e-3
WIRE_LENGTH = 900e-3
WIRE_TILT = 3e-4  # sine of the angle between the axis and z, ~0.017 deg
WIRE_DEFLECTION = 2e-6
#: The blend sits above the conic run, which reaches WIRE_R / WIRE_TILT.
WIRE_BLEND_LO = 750e-3
WIRE_BLEND_HI = 850e-3


def _wire_cylinder():
    return geo.Cylinder(
        radius=WIRE_R,
        origin=(0.0, 0.0, 0.0),
        axis=(0.0, WIRE_TILT, (1.0 - WIRE_TILT**2) ** 0.5),
        height=WIRE_LENGTH,
        material="pec",
    )


def _wire_run_area(polys):
    """|area| of the section polygons below the blend — the conic run."""
    return sum(abs(polygon_area(p)) for p in polys if float(p[:, 1].max()) <= WIRE_BLEND_LO)


class TestConicRunSurvivesAnUnbuildableArc:
    """A run the exact conic cannot replace keeps its own crossings.

    Compression drops the crossings of a run and rewires its ends across
    the arc that takes their place.  When no arc can be built the run
    must be left as it is: rewiring first and only then asking for the
    arc turns the whole conic into a straight chord, and a chord between
    the two ends of a half turn of this body encloses nothing at all.
    """

    @pytest.fixture(scope="class")
    def shape(self):
        def square(z, side):
            half = 0.5 * side
            return geo.Face(
                normal="z",
                points=((-half, -half), (half, -half), (half, half), (-half, half)),
                position=z,
            )

        blend = geo.Loft(
            square(WIRE_BLEND_LO, 8e-4),
            square(WIRE_BLEND_HI, 4e-4),
            blend="ruled",
            material="pec",
        )
        return _wire_cylinder() + blend

    def test_the_fixture_drives_the_arc_into_declining(self, shape, monkeypatch):
        """Guard on the guard: without a declined arc the test below
        would pass on any code."""
        engine = _engine(shape, WIRE_DEFLECTION)
        assert engine.enabled and engine.facetted
        declined = []
        original = _PlanarSectionEngine._cylinder_arc

        def spy(self, *args, **kwargs):
            arc = original(self, *args, **kwargs)
            declined.append(arc is None)
            return arc

        monkeypatch.setattr(_PlanarSectionEngine, "_cylinder_arc", spy)
        engine.section(1, 0.0)
        assert declined and any(declined), declined

    def test_the_run_does_not_collapse_to_a_chord(self, shape):
        """The area of the run stays that of the triangulated conic; a
        chord across it would enclose a sliver of no area at all."""
        engine = _engine(shape, WIRE_DEFLECTION)
        area = _wire_run_area(engine.section(1, 0.0))
        exact = _wire_run_area(
            cross_section_polygons(
                _wire_cylinder()._occ_shape(1.0),
                "y",
                0.0,
                deflection=WIRE_DEFLECTION,
                scale=1.0,
            )
        )
        assert exact > 0.0
        assert abs(area / exact - 1.0) <= 2e-2, (area, exact)


# ---------------------------------------------------------------------------
# A plane tangent to a cylinder
# ---------------------------------------------------------------------------

TANGENT_R = 2.3e-3
TANGENT_BORE_R = 0.7e-3
TANGENT_HEIGHT = 10e-3
TANGENT_DEFLECTION = 2.5e-6
#: Axis offsets along the plane normal.  The second one puts the
#: tangency half an ulp off in ``pos - c`` (4.3e-19), which an exact
#: comparison misses.
TANGENT_OFFSETS = (0.0, 2.1e-3)


def _cross_drilled(offset: float, bore_axis: str):
    """A cross-drilled cylinder sharing its solid with a free-form
    blend, its axis moved by *offset* along *bore_axis*."""

    def along(value: float, third: float):
        return (0.0, value, third) if bore_axis == "y" else (value, 0.0, third)

    body = geo.Cylinder(
        radius=TANGENT_R,
        origin=along(offset, -0.5 * TANGENT_HEIGHT),
        axis="z",
        height=TANGENT_HEIGHT,
        material="pec",
    )
    bore = geo.Cylinder(
        radius=TANGENT_BORE_R,
        origin=along(offset - 2.0 * TANGENT_R, 0.0),
        axis=bore_axis,
        height=4.0 * TANGENT_R,
        material="pec",
    )

    def square(z, side):
        half = 0.5 * side
        return geo.Face(
            normal="z",
            points=((-half, -half), (half, -half), (half, half), (-half, half)),
            position=z,
        )

    blend = geo.Loft(
        square(0.3 * TANGENT_HEIGHT, 1.2 * TANGENT_R),
        square(0.9 * TANGENT_HEIGHT, 0.6 * TANGENT_R),
        blend="ruled",
        material="pec",
    )
    return (body - bore) + blend


class TestTangentPlaneBooksNothing:
    """A plane touching the outer cylinder along one generatrix.

    Its trace on the solid is that line and nothing else — no area at
    all — but the triangulation renders the touch as a run of crossings
    whose ends lie apart, and the conic compression then puts a whole
    turn of the *bore* in its place: 1.539e-6 m^2 of material, the
    bore's own disc, on a plane the solid only touches.  The kernel and
    the exact planar engine both answer nothing there, so the facet path
    delegates the plane the way the analytic screen does.
    """

    @pytest.mark.parametrize("bore_axis", ["x", "y"])
    @pytest.mark.parametrize("offset", TANGENT_OFFSETS)
    def test_the_engine_declines_the_tangent_plane(self, bore_axis, offset):
        shape = _cross_drilled(offset, bore_axis)
        engine = _engine(shape, TANGENT_DEFLECTION)
        assert engine.enabled and engine.facetted
        axis = "xyz".index(bore_axis)
        pos = offset + TANGENT_R
        assert not engine.can_fast(axis, pos)
        assert engine.section(axis, pos) is None

    @pytest.mark.parametrize("bore_axis", ["x", "y"])
    @pytest.mark.parametrize("offset", TANGENT_OFFSETS)
    def test_the_kernel_books_nothing_there(self, bore_axis, offset):
        """What the delegated plane answers: no area at all."""
        shape = _cross_drilled(offset, bore_axis)
        polys = cross_section_polygons(
            shape._occ_shape(1.0),
            bore_axis,
            offset + TANGENT_R,
            deflection=TANGENT_DEFLECTION,
            scale=1.0,
        )
        assert sum(_areas(polys)) == 0.0

    def test_the_fixture_manufactures_the_circle_without_the_screen(self, monkeypatch):
        """Guard on the guard: the plane really does reach the conic
        compression, so the test above would fail without the screen."""
        monkeypatch.setattr(
            _PlanarSectionEngine,
            "_cylinder_tangency",
            lambda self, axis, pos: False,
            raising=False,
        )
        engine = _engine(_cross_drilled(0.0, "y"), TANGENT_DEFLECTION)
        area = sum(_areas(engine.section(1, TANGENT_R)))
        assert abs(area / (np.pi * TANGENT_BORE_R**2) - 1.0) <= 1e-2, area

    @pytest.mark.parametrize("pos", [-1.7e-3, 0.0, 1.1e-3])
    def test_planes_across_the_body_are_untouched(self, pos):
        """The screen takes the tangent plane and nothing else: a cut
        through the body keeps its bore hole and its own area."""
        shape = _cross_drilled(0.0, "y")
        engine = _engine(shape, TANGENT_DEFLECTION)
        polys = engine.section(1, pos)
        assert polys is not None
        ref = cross_section_polygons(
            shape._occ_shape(1.0), "y", pos, deflection=TANGENT_DEFLECTION, scale=1.0
        )
        assert len(polys) == len(ref)
        area, exact = sum(_areas(polys)), sum(_areas(ref))
        assert exact > 0.0
        # Both sides tessellate at the same deflection, so they agree to
        # a fraction of the chord-error class (measured 2.9e-5).
        assert abs(area / exact - 1.0) <= 1e-3, (area, exact)
