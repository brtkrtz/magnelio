"""The conductor surface as patches, and the current on it (DD-273).

The enumeration that turns a wall into patches with normals, the
identity that ties those patches back to the wall-loss booking they
came from, and what the current does with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material, MeshControl
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh._surfaces import enumerate_pec_surfaces, enumerate_wall_patches
from magnelio.mesh.mesher import Mesh

AIR, PEC = Material.air(), Material.pec()


def _brick_mesh(cell=1.5e-3) -> Mesh:
    """A PEC cube in air — axis-aligned, so the staircase path books it."""
    outer = Brick(origin=(0, 0, 0), size=(20e-3, 20e-3, 20e-3), material=AIR)
    inner = Brick(origin=(6e-3, 6e-3, 6e-3), size=(8e-3, 8e-3, 8e-3), material=PEC)
    model = GeometryModel()
    model.add(Difference(outer, inner))
    model.add(inner)
    return Mesh.from_geometry(
        model, MeshControl(min_nodes_per_wavelength=8, min_cell_size=cell), f_max=20e9
    )


def _rod_mesh(cell=0.5e-3, radius=4e-3, height=10e-3) -> Mesh:
    """A PEC cylinder through the domain — the sub-cell path books it."""
    outer = Brick(origin=(-10e-3, -10e-3, 0), size=(20e-3, 20e-3, height), material=AIR)
    rod = Cylinder(origin=(0, 0, 0), radius=radius, height=height, axis=(0, 0, 1), material=PEC)
    model = GeometryModel()
    model.add(Difference(outer, rod))
    model.add(rod)
    return Mesh.from_geometry(
        model, MeshControl(min_nodes_per_wavelength=12, max_cell_size=cell), f_max=20e9
    )


def _random_h(mesh, seed=4, complex_=False):
    g = mesh.grid
    rng = np.random.default_rng(seed)
    shapes = ((g.Nx + 1, g.Ny, g.Nz), (g.Nx, g.Ny + 1, g.Nz), (g.Nx, g.Ny, g.Nz + 1))
    out = [rng.standard_normal(s) for s in shapes]
    if complex_:
        out = [a + 1j * rng.standard_normal(a.shape) for a in out]
    return out


class TestTheEnumeration:
    def test_a_staircase_cube_is_six_flat_faces(self):
        patches = enumerate_wall_patches(_brick_mesh())
        assert not patches.conformal
        np.testing.assert_allclose(patches.area, 6 * (8e-3) ** 2, rtol=1e-12)
        axes, counts = np.unique(np.rint(patches.normals).astype(int), axis=0, return_counts=True)
        assert len(axes) == 6, "one normal per face of the cube"
        assert len(set(counts)) == 1, "the faces are the same size"
        # every normal is a unit axis vector
        np.testing.assert_allclose(np.linalg.norm(patches.normals, axis=1), 1.0)

    def test_a_cylinder_gets_its_true_normals(self):
        """The point of the sub-cell path: not a staircase axis."""
        mesh = _rod_mesh(cell=0.4e-3)
        patches = enumerate_wall_patches(mesh)
        assert patches.conformal
        mantle = patches.select(np.abs(patches.normals[:, 2]) < 0.5)
        r = np.linalg.norm(mantle.centres[:, :2], axis=1)
        radial = mantle.centres[:, :2] / r[:, None]
        alignment = np.sum(mantle.normals[:, :2] * radial, axis=1)
        # A staircase normal would read 0.92 at 45 degrees.
        assert alignment.min() > 0.99
        np.testing.assert_allclose(mantle.area, 2 * np.pi * 4e-3 * 10e-3, rtol=0.01)

    def test_the_normal_points_out_of_the_metal(self):
        patches = enumerate_wall_patches(_brick_mesh())
        centre = np.array([10e-3, 10e-3, 10e-3])  # centre of the cube
        outward = np.sum(patches.normals * (patches.centres - centre), axis=1)
        assert np.all(outward > 0.0)


class TestTheIdentityWithTheLoss:
    """The patches are the wall-loss booking, split by patch."""

    @pytest.mark.parametrize("mesh_of", [_brick_mesh, _rod_mesh])
    @pytest.mark.parametrize("complex_", [False, True])
    def test_area_and_loss_are_reproduced(self, mesh_of, complex_):
        mesh = mesh_of()
        surfaces = enumerate_pec_surfaces(mesh)
        patches = enumerate_wall_patches(mesh)
        h = _random_h(mesh, complex_=complex_)
        np.testing.assert_allclose(patches.area, sum(s.area for s in surfaces), rtol=1e-12)
        np.testing.assert_allclose(
            patches.h_tan_sq(*h).sum(),
            sum(s.h_tan_sq_sum(*h) for s in surfaces),
            rtol=1e-12,
        )

    def test_selecting_a_subset_keeps_its_samples(self):
        mesh = _rod_mesh()
        patches = enumerate_wall_patches(mesh)
        h = _random_h(mesh)
        half = np.zeros(len(patches), dtype=bool)
        half[::2] = True
        part = patches.select(half)
        assert len(part) == int(half.sum())
        np.testing.assert_allclose(part.h_tan_sq(*h), patches.h_tan_sq(*h)[half], rtol=1e-12)
        np.testing.assert_allclose(part.area, patches.areas[half].sum(), rtol=1e-12)

    def test_the_loss_path_is_untouched_by_the_tracking(self):
        """Booking patch ids must not change a single weight."""
        mesh = _rod_mesh()
        plain = enumerate_pec_surfaces(mesh)
        tracked, _blocks = enumerate_pec_surfaces(mesh, with_patches=True)
        assert len(plain) == len(tracked)
        for a, b in zip(plain, tracked):
            np.testing.assert_array_equal(a.weight, b.weight)
            np.testing.assert_array_equal(a.flat_idx, b.flat_idx)
            np.testing.assert_array_equal(a.comp, b.comp)
            assert a.area == b.area


class TestTheCurrent:
    def _current(self, mesh, complex_=False):
        from magnelio.fields import FieldState

        g = mesh.grid
        state = FieldState.zeros(g, dtype=complex if complex_ else float)
        for name, arr in zip(("Hx", "Hy", "Hz"), _random_h(mesh, complex_=complex_)):
            setattr(state._raw, name, arr)
        return state.surface_current(mesh, exclude_faces=()), state, mesh

    def test_the_current_is_perpendicular_to_the_normal(self):
        js, _state, _mesh = self._current(_rod_mesh())
        along = np.sum(js.vector(0) * js.normals, axis=1)
        np.testing.assert_allclose(along, 0.0, atol=1e-9 * np.abs(js.vector(0)).max())

    def test_its_magnitude_is_the_booked_one(self):
        """|J_s|² A is the patch's share of the loss integral, exactly."""
        js, state, mesh = self._current(_rod_mesh())
        patches = enumerate_wall_patches(mesh)
        h = [getattr(state._raw, c) for c in ("Hx", "Hy", "Hz")]
        np.testing.assert_allclose(js.magnitude(0) ** 2 * js.areas, patches.h_tan_sq(*h), rtol=1e-9)

    def test_power_loss_is_the_wall_loss_integral(self):
        js, state, mesh = self._current(_rod_mesh())
        surfaces = enumerate_pec_surfaces(mesh)
        h = [getattr(state._raw, c) for c in ("Hx", "Hy", "Hz")]
        r_s = 0.026
        expected = 0.5 * r_s * sum(s.h_tan_sq_sum(*h) for s in surfaces)
        np.testing.assert_allclose(js.power_loss(r_s), expected, rtol=1e-9)

    def test_a_complex_frame_carries_no_further_half(self):
        js, state, mesh = self._current(_rod_mesh(), complex_=True)
        surfaces = enumerate_pec_surfaces(mesh)
        h = [getattr(state._raw, c) for c in ("Hx", "Hy", "Hz")]
        r_s = 0.026
        np.testing.assert_allclose(
            js.power_loss(r_s), r_s * sum(s.h_tan_sq_sum(*h) for s in surfaces), rtol=1e-9
        )

    def test_one_conductor_can_be_selected(self):
        js, _state, _mesh = self._current(_brick_mesh())
        tag = js.tags[0]
        part = js.select(tag)
        assert len(part) == int((js.tags == tag).sum())
        assert "patches" in repr(part)


class TestWhatItRefuses:
    def test_a_field_without_h_refuses(self):
        from magnelio.fields import FieldRecording, FieldState

        mesh = _brick_mesh()
        g = mesh.grid
        zero = FieldState.zeros(g)
        rec = FieldRecording(g, [0.0], Ez=np.zeros((1, *zero.Ez.shape)))
        with pytest.raises(KeyError, match="all three magnetic components"):
            rec.surface_current(mesh)

    def test_a_field_on_another_grid_refuses(self):
        from magnelio.fields import FieldState
        from magnelio.mesh.grid import GridLines

        mesh = _brick_mesh()
        other = FieldState.zeros(GridLines(x=[0, 1e-3], y=[0, 1e-3], z=[0, 1e-3]))
        with pytest.raises(ValueError, match="mesh's own grid"):
            other.surface_current(mesh)

    def test_a_mesh_without_metal_refuses(self):
        from magnelio.fields import FieldState

        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=AIR))
        mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=6), f_max=20e9)
        state = FieldState.zeros(mesh.grid)
        with pytest.raises(ValueError, match="no conductor surface"):
            state.surface_current(mesh)


class TestThePicture:
    """The viewer draws it — and the group has to be registered, or the
    actor is built, filled and then never shown (it was, once)."""

    def _scene(self, **kwargs):
        import pyvista as pv

        pv.OFF_SCREEN = True
        from magnelio.fields import FieldState
        from magnelio.post.plot_3d import _build_scene

        mesh = _brick_mesh()
        state = FieldState.zeros(mesh.grid)
        for name, arr in zip(("Hx", "Hy", "Hz"), _random_h(mesh)):
            setattr(state._raw, name, arr)
        js = state.surface_current(mesh, exclude_faces=())
        model = GeometryModel()
        model.add(Brick(origin=(6e-3, 6e-3, 6e-3), size=(8e-3, 8e-3, 8e-3), material=PEC))
        scene = _build_scene(
            model,
            mesh=None,
            cut=None,
            flip=False,
            show_ports=False,
            show_wires=False,
            show_grid=False,
            show_labels=False,
            size=(400, 300),
            render_edges=False,
            edge_color="#202020",
            quality=1.0,
            scale_mm=True,
            camera="iso",
            off_screen=True,
            surface_current=js,
            **kwargs,
        )
        return scene, js

    def test_the_arrows_are_built_and_visible(self):
        scene, js = self._scene()
        overlay = next(o for o in scene.overlays if o.name == "surface_current")
        assert overlay.polydata.n_points > 0
        assert overlay.actor.GetVisibility() == 1, "the group must be a known display group"
        assert "|J_s| (A/m)" in overlay.polydata.point_data
        assert overlay.polydata.active_scalars_name == "|J_s| (A/m)"

    def test_density_thins_the_arrows_out(self):
        dense, _js = self._scene()
        sparse, _js = self._scene(current_density=4)
        n_dense = next(o for o in dense.overlays if o.name == "surface_current").polydata.n_points
        n_sparse = next(o for o in sparse.overlays if o.name == "surface_current").polydata.n_points
        assert n_sparse < n_dense
