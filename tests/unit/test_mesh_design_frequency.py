"""DD-186: the mesh records the f_max it was generated for.

``Mesh.from_geometry`` stamps its ``f_max`` on the mesh; the scattering
analysis defaults its band to it, warns when asked to exceed it, and a
``from_grid`` mesh (no design frequency) keeps requiring an explicit
value.  The attribute survives the wall-rewrite copies and the project
store.
"""

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh, MeshControl
from magnelio.io.project import ProjectStore, open_project
from magnelio.mesh.grid import GridLines

F_MESH = 30e9


def _grid_mesh() -> Mesh:
    grid = GridLines(
        x=np.linspace(0, 4e-3, 5),
        y=np.linspace(0, 4e-3, 5),
        z=np.linspace(0, 4e-3, 5),
    )
    return Mesh.from_grid(grid)


class TestFromGrid:
    def test_no_design_frequency(self):
        assert _grid_mesh().f_max is None

    def test_analysis_requires_explicit_f_max(self):
        with pytest.raises(ValueError, match="f_max is required"):
            AnalysisScatteringTD(mesh=_grid_mesh(), verbose=False)

    def test_store_roundtrip_none(self, tmp_path):
        ProjectStore.create(tmp_path / "proj", _grid_mesh())
        assert open_project(tmp_path / "proj").mesh.f_max is None


@pytest.fixture(scope="module")
def occ_mesh() -> Mesh:
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio.geo import Brick, GeometryModel  # noqa: PLC0415
    from magnelio.ports import PortWaveguide  # noqa: PLC0415

    model = GeometryModel(background="pec")
    model.add(Brick(material="air", origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3)))
    model.add_port(PortWaveguide(name="p1", plane="zmin"))
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=10),
        f_max=F_MESH,
    )


class TestFromGeometry:
    def test_records_design_frequency(self, occ_mesh):
        assert occ_mesh.f_max == F_MESH

    def test_wall_rewrite_copy_keeps_it(self, occ_mesh):
        assert occ_mesh.with_pec_boundaries(["zmax"]).f_max == F_MESH

    def test_analysis_defaults_to_mesh_f_max(self, occ_mesh):
        ana = AnalysisScatteringTD(mesh=occ_mesh, verbose=False)
        assert ana.f_max == F_MESH

    def test_explicit_f_max_overrides(self, occ_mesh):
        ana = AnalysisScatteringTD(mesh=occ_mesh, f_max=F_MESH / 2, verbose=False)
        assert ana.f_max == F_MESH / 2

    def test_exceeding_design_frequency_warns(self, occ_mesh):
        with pytest.warns(UserWarning, match="exceeds the design frequency"):
            AnalysisScatteringTD(mesh=occ_mesh, f_max=2 * F_MESH, verbose=False)

    def test_below_design_frequency_is_silent(self, occ_mesh):
        import warnings as _warnings  # noqa: PLC0415

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            AnalysisScatteringTD(mesh=occ_mesh, f_max=F_MESH / 2, verbose=False)

    def test_store_roundtrip(self, occ_mesh, tmp_path):
        ProjectStore.create(tmp_path / "proj", occ_mesh)
        assert open_project(tmp_path / "proj").mesh.f_max == F_MESH
