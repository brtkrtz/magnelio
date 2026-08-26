"""Tests for ``geo.Surface`` — curved sheets from parametric maps.

The fixture is an offset paraboloid dish (focal length F, aperture D,
centre x_c off the axis) parametrised in polar coordinates about the
aperture centre, the shape the Cassegrain tutorial builds.  Closed
forms: the prism of the dish along z has the volume of the projected
disc times the length; the paraboloid surface area over the disc is a
1D integral; every sample lies on z = (x² + y²)/(4F).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo
from magnelio.geo._sheet import PlanarSheet, Sheet

F, D, XC = 0.18, 0.24, 0.15
T = 5e-3


def _dish(r, phi):
    x = XC + r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, (x * x + y * y) / (4 * F)


def _area_exact():
    rr = np.linspace(0.0, D / 2, 400)
    pp = np.linspace(0.0, 2 * np.pi, 800)
    R, P = np.meshgrid(rr, pp, indexing="ij")
    X = XC + R * np.cos(P)
    Y = R * np.sin(P)
    integrand = np.sqrt(1.0 + (X * X + Y * Y) / (4 * F * F)) * R
    return np.trapezoid(np.trapezoid(integrand, pp, axis=1), rr)


def _occ_area(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.SurfaceProperties(shape._occ_shape(), props)
    return props.Mass()


@pytest.fixture(scope="module")
def dish():
    return geo.Surface.parametric(_dish, u=(0.0, D / 2), v=(0.0, 2 * np.pi))


class TestParametricSampling:
    def test_is_a_sheet_but_not_planar(self, dish):
        assert isinstance(dish, Sheet)
        assert not isinstance(dish, PlanarSheet)
        assert dish.material is None

    def test_samples_lie_on_the_surface(self, dish):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
        from OCC.Core.gp import gp_Pnt

        surf = BRepAdaptor_Surface(dish._occ_shape()).Surface().Surface()
        worst = 0.0
        for r in np.linspace(0.0, D / 2, 7):
            for phi in np.linspace(0.1, 2 * np.pi - 0.1, 9):
                x, y, z = _dish(r, phi)
                worst = max(
                    worst, GeomAPI_ProjectPointOnSurf(gp_Pnt(x, y, z), surf).LowerDistance()
                )
        # 32 x 32 samples on a 240 mm dish: micrometres between samples.
        assert worst < 5e-6

    def test_area_matches_the_paraboloid(self, dish):
        assert _occ_area(dish) == pytest.approx(_area_exact(), rel=1e-4)

    def test_scalar_map_is_accepted(self):
        sheet = geo.Surface.parametric(
            lambda u, v: (u, v, u * v), u=(0.0, 1e-2), v=(0.0, 1e-2), samples=(4, 3)
        )
        assert len(sheet.points) == 4 and len(sheet.points[0]) == 3
        assert sheet.points[-1][-1] == pytest.approx((1e-2, 1e-2, 1e-4))

    def test_grid_shape_and_parameter_order(self):
        sheet = geo.Surface.parametric(
            lambda u, v: (u, 2 * v, 0 * u), u=(0.0, 1.0), v=(0.0, 1.0), samples=(3, 5)
        )
        assert len(sheet.points) == 3 and all(len(row) == 5 for row in sheet.points)
        assert sheet.points[2][0] == pytest.approx((1.0, 0.0, 0.0))
        assert sheet.points[0][4] == pytest.approx((0.0, 2.0, 0.0))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"u": (0.0, 0.0), "v": (0.0, 1.0)}, "non-empty interval"),
            ({"u": (0.0, 1.0), "v": 3.0}, "pair"),
            ({"u": (0.0, 1.0), "v": (0.0, 1.0), "samples": (1, 4)}, "samples"),
            ({"u": (0.0, 1.0), "v": (0.0, 1.0), "samples": 7}, "pair of integers"),
        ],
    )
    def test_bad_arguments_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            geo.Surface.parametric(lambda u, v: (u, v, 0 * u), **kwargs)

    def test_non_finite_map_is_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            geo.Surface.parametric(lambda u, v: (u, v, 1.0 / u), u=(0.0, 1.0), v=(0.0, 1.0))

    def test_map_must_return_three_coordinates(self):
        with pytest.raises(ValueError, match="three coordinates"):
            geo.Surface.parametric(lambda u, v: (u, v), u=(0.0, 1.0), v=(0.0, 1.0))

    def test_not_callable_is_rejected(self):
        with pytest.raises(TypeError, match="callable"):
            geo.Surface.parametric(3.0, u=(0.0, 1.0), v=(0.0, 1.0))

    def test_point_grid_constructor_validates(self):
        with pytest.raises(ValueError, match="2 x 2"):
            geo.Surface(points=(((0, 0, 0), (1, 0, 0)),))
        with pytest.raises(ValueError, match="same length"):
            geo.Surface(points=(((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0), (2, 1, 0))))


class TestExtrudeAndThicken:
    def test_extruded_volume_is_projected_disc_times_length(self, dish):
        solid = dish.extruded(vector=(0.0, 0.0, -T), material="pec")
        assert solid.volume() == pytest.approx(math.pi * (D / 2) ** 2 * T, rel=1e-4)
        assert solid.material.is_pec

    def test_extruding_a_construction_sheet_needs_a_material(self, dish):
        with pytest.raises(ValueError, match="material"):
            dish.extruded(vector=(0.0, 0.0, -T))

    def test_thickened_forward_grows_along_the_dominant_normal(self, dish):
        shell = dish.thickened(thickness=T, material="pec")
        assert shell.volume() == pytest.approx(_area_exact() * T, rel=0.03)
        (_, _, z_lo), (_, _, z_hi) = shell.bounding_box()
        dish_lo, dish_hi = dish.bounding_box()
        # Forward = +z here (the dish opens upward): the shell rises above
        # the sheet's rim and keeps its vertex.
        assert z_hi > dish_hi[2] + 0.8 * T
        assert z_lo == pytest.approx(dish_lo[2], abs=1e-4)

    def test_thickened_backward_grows_the_other_way(self, dish):
        shell = dish.thickened(thickness=T, direction="backward", material="pec")
        (_, _, z_lo), _ = shell.bounding_box()
        assert z_lo < (XC - D / 2) ** 2 / (4 * F) - 0.5 * T

    def test_symmetric_is_planar_only(self, dish):
        with pytest.raises(ValueError, match="planar sheets only"):
            dish.thickened(thickness=T, direction="symmetric", material="pec").volume()

    def test_thicken_refuses_a_folded_offset_loudly(self):
        # The kernel's offset folds on this dense grid; the check must
        # catch it instead of returning a body of absurd volume.
        dense = geo.Surface.parametric(_dish, u=(0.0, D / 2), v=(0.0, 2 * np.pi), samples=(48, 96))
        with pytest.raises(ValueError, match="extruded"):
            dense.thickened(thickness=T, material="pec").volume()

    def test_shelled_refuses_a_sheet(self, dish):
        with pytest.raises(TypeError, match="thickened"):
            dish.shelled(thickness=T)


class TestTransformsKeepSheets:
    def test_rotated_surface_is_still_a_sheet(self, dish):
        turned = dish.rotated("y", 25.0)
        assert isinstance(turned, Sheet) and not isinstance(turned, PlanarSheet)
        solid = turned.extruded(vector=(0.0, 0.0, -T), material="pec")
        assert solid.volume() > 0.9 * math.pi * (D / 2) ** 2 * T * math.cos(math.radians(25.0))

    def test_translated_scaled_mirrored_keep_the_marker(self, dish):
        for sheet in (
            dish.translated((0.0, 0.0, 1e-2)),
            dish.scaled(0.5),
            dish.mirrored("x"),
        ):
            assert isinstance(sheet, Sheet)
            assert sheet.extruded(vector=(0.0, 0.0, -T), material="pec").volume() > 0.0

    def test_rotated_face_thickens(self):
        face = geo.Face(normal="z", points=((0, 0), (1e-2, 0), (1e-2, 5e-3)))
        slab = face.rotated("x", 30.0).thickened(thickness=1e-3, material="pec")
        assert isinstance(face.rotated("x", 30.0), PlanarSheet)
        assert slab.volume() == pytest.approx(0.5 * 1e-2 * 5e-3 * 1e-3, rel=1e-9)


class TestModelAndMesh:
    def test_standalone_sheet_cannot_be_meshed(self, dish):
        model = mio.GeometryModel()
        with pytest.raises(ValueError, match="construction body"):
            model.add(dish)
        model.add(geo.Surface(points=dish.points, material="pec"))
        with pytest.raises(NotImplementedError, match="Surface"):
            mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=2e-2), f_max=3e9)

    def test_thin_curved_pec_dish_meshes(self, dish):
        small = geo.Surface.parametric(
            lambda r, phi: (r * np.cos(phi), r * np.sin(phi), r * r / 20e-3),
            u=(0.0, 10e-3),
            v=(0.0, 2 * np.pi),
            samples=(12, 24),
        )
        reflector = small.extruded(vector=(0.0, 0.0, -1e-3), material="pec")
        box = geo.Brick(origin=(-15e-3, -15e-3, -5e-3), size=(30e-3, 30e-3, 15e-3), material="air")
        model = mio.GeometryModel()
        model.add(geo.Difference(box, reflector))
        model.add(reflector)
        mesh = mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=1.5e-3), f_max=20e9)
        assert mesh.Nx * mesh.Ny * mesh.Nz > 0
        assert mesh.pec_mask_edges.any()

    def test_store_round_trip_returns_the_solid(self, dish, tmp_path):
        from magnelio.io.project import read_brep, write_brep

        reflector = dish.extruded(vector=(0.0, 0.0, -T), material="pec")
        path = tmp_path / "dish.brep"
        write_brep([reflector], path)
        (topo,) = read_brep(path)
        back = geo.ImportedSolid(topo, "pec")
        assert back.volume() == pytest.approx(reflector.volume(), rel=1e-6)
