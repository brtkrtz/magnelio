"""DD-192: the bulk cell size follows the wavelength of the slab, not the model."""

import math

import numpy as np
import pytest

from magnelio.constants import C0
from magnelio.mesh.mesher import (
    MeshControl,
    _generate_axis_lines,
    _local_bulk_sizes,
    _refractive_index,
)


def _occ():
    pytest.importorskip("OCC")


F_MAX = 10e9
MNPW = 20
H_AIR = C0 / F_MAX / MNPW
H_FR4 = H_AIR / math.sqrt(4.3)


def _ceramic_in_air(eps_r=4.3, background=None):
    from magnelio.geo import Brick, GeometryModel
    from magnelio.materials.material import Material

    puck = Brick(origin=(-5e-3, -5e-3, -1e-3), size=(10e-3, 10e-3, 2e-3), material="air")
    box = Brick(origin=(-40e-3, -40e-3, -40e-3), size=(80e-3, 80e-3, 80e-3), material="air")
    ceramic = Material(name="ceramic", epsilon=(eps_r,) * 3)
    m = GeometryModel(background=background) if background is not None else GeometryModel()
    m.add(box - puck)
    m.add(
        Brick(origin=(-5e-3, -5e-3, -1e-3), size=(10e-3, 10e-3, 2e-3), material=ceramic),
    )
    return m


def _cells_outside(nodes, lo, hi):
    d = np.diff(nodes)
    mid = 0.5 * (nodes[1:] + nodes[:-1])
    return d[(mid < lo) | (mid > hi)], d[(mid > lo) & (mid < hi)]


class TestRefractiveIndex:
    def test_pec_and_none_have_no_wavelength(self):
        from magnelio.materials.material import Material

        assert _refractive_index(None) is None
        assert _refractive_index(Material.pec()) is None

    def test_anisotropic_takes_the_densest_component(self):
        from magnelio.materials.material import Material

        m = Material(name="aniso", epsilon=(2.0, 8.0, 2.0))
        assert _refractive_index(m) == pytest.approx(math.sqrt(8.0))


class TestLocalBulkSizes:
    """The pure per-interval rule, without OCC."""

    def _shape(self, eps_r, lo, hi):
        from magnelio.materials.material import Material

        class _Box:
            material = Material(name="d", epsilon=(eps_r,) * 3)

            def _analytic_bbox(self):
                return (lo, hi)

        return _Box()

    def test_slab_takes_the_densest_material_reaching_into_it(self):
        planes = {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 3.0], "z": [0.0, 3.0]}
        s = self._shape(4.0, (1.0, 0.0, 0.0), (2.0, 3.0, 3.0))
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW)
        out = _local_bulk_sizes(planes, [s], None, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR, H_AIR / 2.0, H_AIR])
        # y and z: the box spans the whole axis → dense everywhere.
        assert out["y"] == pytest.approx([H_AIR / 2.0])
        assert out["z"] == pytest.approx([H_AIR / 2.0])

    def test_touching_at_a_plane_does_not_count(self):
        planes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
        s = self._shape(4.0, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW)
        out = _local_bulk_sizes(planes, [s], None, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR, H_AIR / 2.0])

    def test_global_rule_uses_the_densest_material_everywhere(self):
        planes = {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 3.0], "z": [0.0, 3.0]}
        s = self._shape(4.0, (1.0, 0.0, 0.0), (2.0, 3.0, 3.0))
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW, wavelength_rule="global")
        out = _local_bulk_sizes(planes, [s], None, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR / 2.0] * 3)

    def test_background_counts_in_every_slab(self):
        from magnelio.materials.material import Material

        planes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
        s = self._shape(4.0, (1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        bg = Material(name="oil", epsilon=(2.25,) * 3)
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW)
        out = _local_bulk_sizes(planes, [s], bg, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR / 1.5, H_AIR / 2.0])

    def test_shape_without_bbox_counts_everywhere(self):
        from magnelio.materials.material import Material

        class _Exotic:
            material = Material(name="d", epsilon=(9.0,) * 3)

            def _analytic_bbox(self):
                raise RuntimeError("no analytic box")

        planes = {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW)
        out = _local_bulk_sizes(planes, [_Exotic()], None, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR / 3.0] * 2)

    def test_pec_shapes_do_not_set_a_wavelength(self):
        from magnelio.materials.material import Material

        class _Metal:
            material = Material.pec()

            def _analytic_bbox(self):
                return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        planes = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW)
        out = _local_bulk_sizes(planes, [_Metal()], None, F_MAX, ctrl, tol=1e-9)
        assert out["x"] == pytest.approx([H_AIR])


class TestAxisLinesPerInterval:
    def test_per_interval_bulk_sizes_are_honoured(self):
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW, min_cells_per_feature=0)
        planes = [0.0, 10e-3, 20e-3, 30e-3]
        nodes = np.asarray(
            _generate_axis_lines(planes, h_max=[2e-3, 1e-3, 2e-3], h_fine=1e-3, control=ctrl)
        )
        d = np.diff(nodes)
        mid = 0.5 * (nodes[1:] + nodes[:-1])
        assert d[(mid > 10e-3) & (mid < 20e-3)].max() <= 1e-3 * (1 + 1e-9)
        assert d[mid < 10e-3].max() > 1.5e-3
        assert d[mid > 20e-3].max() > 1.5e-3

    def test_scalar_bulk_size_still_works(self):
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW, min_cells_per_feature=0)
        a = _generate_axis_lines([0.0, 30e-3], h_max=2e-3, h_fine=2e-3, control=ctrl)
        b = _generate_axis_lines([0.0, 30e-3], h_max=[2e-3], h_fine=2e-3, control=ctrl)
        assert a == b

    def test_wrong_count_raises(self):
        ctrl = MeshControl()
        with pytest.raises(ValueError, match="per-interval bulk sizes"):
            _generate_axis_lines([0.0, 1.0, 2.0], h_max=[1.0], h_fine=1.0, control=ctrl)


class TestMeshControl:
    def test_default_is_local(self):
        assert MeshControl().wavelength_rule == "local"

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError, match="wavelength_rule"):
            MeshControl(wavelength_rule="slab")


class TestFromGeometry:
    def test_air_around_a_ceramic_is_meshed_at_the_air_wavelength(self):
        _occ()
        from magnelio.mesh.mesher import Mesh

        ctrl = MeshControl(min_nodes_per_wavelength=MNPW, min_cells_per_feature=4)
        mesh = Mesh.from_geometry(_ceramic_in_air(), ctrl, f_max=F_MAX)
        slabs = ((mesh.grid.x, -5e-3, 5e-3), (mesh.grid.y, -5e-3, 5e-3), (mesh.grid.z, -1e-3, 1e-3))
        for nodes, lo, hi in slabs:
            outside, inside = _cells_outside(nodes, lo, hi)
            assert outside.max() == pytest.approx(H_AIR, rel=1e-6)
            assert inside.max() <= H_FR4 * (1 + 1e-9)

    def test_global_rule_meshes_the_air_at_the_ceramic_wavelength(self):
        _occ()
        from magnelio.mesh.mesher import Mesh

        ctrl = MeshControl(
            min_nodes_per_wavelength=MNPW, min_cells_per_feature=4, wavelength_rule="global"
        )
        mesh = Mesh.from_geometry(_ceramic_in_air(), ctrl, f_max=F_MAX)
        for nodes in (mesh.grid.x, mesh.grid.y, mesh.grid.z):
            assert np.diff(nodes).max() <= H_FR4 * (1 + 1e-9)

    def test_local_rule_saves_cells(self):
        _occ()
        from magnelio.mesh.mesher import Mesh

        ctrl_local = MeshControl(min_nodes_per_wavelength=MNPW)
        ctrl_global = MeshControl(min_nodes_per_wavelength=MNPW, wavelength_rule="global")
        local = Mesh.from_geometry(_ceramic_in_air(), ctrl_local, f_max=F_MAX)
        glob = Mesh.from_geometry(_ceramic_in_air(), ctrl_global, f_max=F_MAX)
        n_local = local.Nx * local.Ny * local.Nz
        n_glob = glob.Nx * glob.Ny * glob.Nz
        assert n_local < 0.3 * n_glob

    def test_homogeneous_model_is_identical_under_both_rules(self):
        _occ()
        from magnelio.geo import Brick, GeometryModel
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        ptfe = Material(name="PTFE", epsilon=(2.1,) * 3)
        rod = Brick(origin=(8e-3, 4e-3, 0), size=(4e-3, 2e-3, 30e-3), material="pec")
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(20e-3, 10e-3, 30e-3), material=ptfe) - rod)
        m.add(rod)
        ctrl_local = MeshControl(min_nodes_per_wavelength=MNPW)
        ctrl_global = MeshControl(min_nodes_per_wavelength=MNPW, wavelength_rule="global")
        a = Mesh.from_geometry(m, ctrl_local, f_max=F_MAX)
        b = Mesh.from_geometry(m, ctrl_global, f_max=F_MAX)
        for ax in "xyz":
            assert np.array_equal(getattr(a.grid, ax), getattr(b.grid, ax))

    def test_dense_background_sets_the_wavelength_everywhere(self):
        _occ()
        from magnelio.materials.material import Material
        from magnelio.mesh.mesher import Mesh

        oil = Material(name="oil", epsilon=(2.25,) * 3)
        ctrl = MeshControl(min_nodes_per_wavelength=MNPW, min_cells_per_feature=4)
        mesh = Mesh.from_geometry(_ceramic_in_air(background=oil), ctrl, f_max=F_MAX)
        outside, _ = _cells_outside(mesh.grid.x, -5e-3, 5e-3)
        assert outside.max() <= H_AIR / 1.5 * (1 + 1e-9)
