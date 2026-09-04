"""Port-plane refinement (DD-244): ``refine_port_modes`` and ``MeshControl.subdivide``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio.analysis import AnalysisScatteringTD
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports import PortWaveguide, refine_port_modes

F_MAX = 8.0e9


def _rect_coax(*, B: float = 8e-3, b_air: float = 6e-3, a: float = 2e-3, L_x: float = 6e-3):
    """Square coax with a PEC outer body and an inner brick (IPC-2141A Z_line)."""
    pec = Material.pec()
    air = Material.air()
    y0_air = z0_air = (B - b_air) / 2
    y0_inner = z0_inner = (B - a) / 2
    bbox = Brick(origin=(0, 0, 0), size=(L_x, B, B), material=pec)
    air_region = Brick(origin=(0, y0_air, z0_air), size=(L_x, b_air, b_air), material=air)
    inner = Brick(origin=(0, y0_inner, z0_inner), size=(L_x, a, a), material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, air_region))
    model.add(Difference(air_region, inner))
    model.add(inner)
    model.add_port(PortWaveguide(name="p", plane="xmin"))
    return model, a, b_air


def _ipc_2141a_z_line(B: float, a: float) -> float:
    return 60.0 * math.log(1.0787 * B / a)


def _hollow_guide():
    """A short WR-90-like PEC-walled air guide with one waveguide port."""
    guide = GeometryModel(background="pec")
    guide.add(Brick(origin=(0, 0, 0), size=(22.86e-3, 10.16e-3, 12e-3), material=Material.air()))
    guide.add_port(PortWaveguide(name="wg", plane="zmin", n_modes=1))
    control = MeshControl(min_nodes_per_wavelength=6, conformal=False, max_cell_size=2e-3)
    mesh = Mesh.from_geometry(guide, control, f_max=12e9)
    return guide, control, mesh


@pytest.fixture(scope="module")
def coax():
    model, a, b_air = _rect_coax()
    control = MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=False,
        max_cell_size=0.4e-3,
    )
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)
    return model, control, mesh, a, b_air


class TestSubdivide:
    def test_every_plane_survives_and_every_cell_splits(self, coax):
        model, control, mesh, *_ = coax
        fine = Mesh.from_geometry(
            model,
            MeshControl(**{**control.__dict__, "subdivide": {"y": 2, "z": 3}}),
            f_max=F_MAX,
        )
        assert fine.Nx == mesh.Nx
        assert fine.Ny == 2 * mesh.Ny
        assert fine.Nz == 3 * mesh.Nz
        assert np.allclose(fine.grid.x, mesh.grid.x)
        assert np.allclose(fine.grid.y[::2], mesh.grid.y)
        assert np.allclose(fine.grid.z[::3], mesh.grid.z)
        # Materials of the coarse cells replicate into their sub-cells
        # (axis-aligned bricks: no partially filled cell moves).
        coarse = mesh.material_id
        assert np.array_equal(fine.material_id, np.repeat(np.repeat(coarse, 2, axis=1), 3, axis=2))

    def test_validation(self):
        with pytest.raises(ValueError, match="subdivide"):
            MeshControl(subdivide={"x": 0})
        with pytest.raises(ValueError, match="subdivide"):
            MeshControl(subdivide={"w": 2})


class TestRefinePortModes:
    def test_level_zero_is_the_users_port_grid(self, coax):
        model, control, mesh, *_ = coax
        z_user = AnalysisScatteringTD(mesh=mesh, verbose=False).solve_ports()["p"].z_line_num
        report = refine_port_modes(model, control, mesh, "p", levels=1)
        assert report.n_levels == 1
        assert report.levels[0].n_cells_port_plane == mesh.Ny * mesh.Nz
        assert report.levels[0].value == pytest.approx(z_user, rel=1e-12)
        assert report.extrapolated is None
        assert report.value == pytest.approx(z_user, rel=1e-12)

    def test_ladder_converges_toward_the_reference(self, coax):
        model, control, mesh, a, b_air = coax
        report = refine_port_modes(model, control, mesh, "p", levels=3, tol=1e-6)
        z_ipc = _ipc_2141a_z_line(b_air, a)
        cells = [lv.n_cells_port_plane for lv in report.levels]
        assert cells[1] == 4 * cells[0] and cells[2] == 4 * cells[1]
        errors = [abs(lv.value - z_ipc) for lv in report.levels]
        assert errors[2] < errors[0]
        assert math.isnan(report.levels[0].rel_change)
        assert report.levels[2].rel_change < report.levels[1].rel_change
        assert report.extrapolated is not None
        assert report.order is not None
        assert abs(report.value - z_ipc) / z_ipc < 0.2
        assert "level 2" in str(report)
        assert len(report.reports) == 3

    def test_tolerance_stops_the_ladder(self, coax):
        model, control, mesh, *_ = coax
        report = refine_port_modes(model, control, mesh, "p", levels=4, tol=0.5)
        assert report.converged
        assert report.n_levels == 2

    def test_other_targets(self, coax):
        model, control, mesh, *_ = coax
        eps = refine_port_modes(model, control, mesh, "p", levels=2, target="epsilon_eff")
        assert all(lv.value == pytest.approx(1.0, abs=1e-9) for lv in eps.levels)
        assert eps.converged

    def test_auto_target_follows_the_mode_family(self, coax):
        """A TEM line converges its impedance, a hollow guide its cut-off.

        The default used to be ``z_line`` outright, and a rectangular-to-
        circular taper's round port answered with ``has no line
        impedance`` -- the one port a taper's user wants to converge.
        """
        model, control, mesh, *_ = coax
        tem = refine_port_modes(model, control, mesh, "p", levels=1)
        assert tem.target == "z_line"

        guide, g_control, g_mesh = _hollow_guide()
        te = refine_port_modes(guide, g_control, g_mesh, "wg", levels=1)
        assert te.target == "f_cutoff"
        f_c = AnalysisScatteringTD(mesh=g_mesh, verbose=False).solve_ports()["wg"].modes[0]
        assert te.levels[0].value == pytest.approx(f_c.f_cutoff, rel=1e-12)
        assert "f_cutoff" in str(te)

    def test_z_line_on_a_te_mode_names_the_way_out(self):
        guide, control, mesh = _hollow_guide()
        with pytest.raises(ValueError, match="target='f_cutoff'"):
            refine_port_modes(guide, control, mesh, "wg", levels=1, target="z_line")

    def test_rejections(self, coax):
        model, control, mesh, *_ = coax
        with pytest.raises(ValueError, match="not declared"):
            refine_port_modes(model, control, mesh, "nope")
        with pytest.raises(ValueError, match="target"):
            refine_port_modes(model, control, mesh, "p", target="v_phase")
        with pytest.raises(ValueError, match="levels"):
            refine_port_modes(model, control, mesh, "p", levels=0)
        with pytest.raises(ValueError, match="slab_cells"):
            refine_port_modes(model, control, mesh, "p", slab_cells=2)
