"""DD-109: ports declared on the model before meshing.

Covers the declaration API (GeometryModel.add_port, Mesh.with_ports),
the port hand-off model -> mesh -> analysis, and the per-face buffer:
declared faces get the three equidistant cells, undeclared faces keep
the plain grading, and a model without declarations falls back to the
DD-107 all-face buffer.
"""

import numpy as np
import pytest

from magnelio import Material, Mesh, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.mesh.mesher import _port_buffer_ends
from magnelio.ports import PortWaveguide

pytest.importorskip("OCC.Core.BRepPrimAPI")

PEC = Material.pec()
AIR = Material.air()


def _model_with_feature() -> GeometryModel:
    """10 mm air box with a small PEC insert that forces wall grading."""
    box = Brick(material=AIR, origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3))
    insert = Brick(material=PEC, origin=(4.5e-3, 4.5e-3, 4.5e-3), size=(1e-3, 1e-3, 1e-3))
    model = GeometryModel(background=PEC)
    model.add(box - insert)
    model.add(insert)
    return model


def _mesh(model: GeometryModel) -> Mesh:
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=10),
        f_max=30e9,
    )


def _is_buffered(deltas: np.ndarray, end: str) -> bool:
    tail = deltas[:3] if end == "lo" else deltas[-3:]
    return (tail.max() - tail.min()) / tail.min() <= 1e-9


class TestAddPort:
    def test_add_port_chains_and_stores(self):
        model = GeometryModel()
        port = PortWaveguide(name="p1", plane="zmin")
        assert model.add_port(port) is model
        assert model.ports == [port]

    def test_duplicate_name_raises(self):
        model = GeometryModel()
        model.add_port(PortWaveguide(name="p1", plane="zmin"))
        with pytest.raises(ValueError, match="duplicate port name"):
            model.add_port(PortWaveguide(name="p1", plane="zmax"))

    def test_spec_level_port_rejected(self):
        from magnelio.ports._modal import PortSpecRectWG

        model = GeometryModel()
        with pytest.raises(TypeError, match="declarative"):
            model.add_port(
                PortSpecRectWG(
                    name="p1",
                    plane="zmin",
                    width_a=22.86e-3,
                    height_b=10.16e-3,
                )
            )


class TestBufferEnds:
    def test_no_ports_buffers_everything(self):
        ends = _port_buffer_ends(())
        assert ends == {"x": ("lo", "hi"), "y": ("lo", "hi"), "z": ("lo", "hi")}

    def test_declared_ports_select_faces(self):
        ends = _port_buffer_ends(
            (
                PortWaveguide(name="a", plane="zmin"),
                PortWaveguide(name="b", plane="xmax"),
            )
        )
        assert ends == {"x": ("hi",), "y": (), "z": ("lo",)}

    def test_bad_plane_raises(self):
        class _P:
            plane = "diagonal"
            name = "p"

        with pytest.raises(ValueError, match="unknown port plane"):
            _port_buffer_ends((_P(),))


class TestMeshCarriesPorts:
    def test_from_geometry_carries_declarations(self):
        model = _model_with_feature()
        port = PortWaveguide(name="p1", plane="zmin")
        model.add_port(port)
        mesh = _mesh(model)
        assert mesh.ports == (port,)

    def test_with_ports_attaches(self):
        mesh = _mesh(_model_with_feature())
        port = PortWaveguide(name="p1", plane="zmin")
        mesh2 = mesh.with_ports([port])
        assert mesh2.ports == (port,)
        assert mesh.ports == ()  # original untouched

    def test_with_ports_duplicate_name_raises(self):
        mesh = _mesh(_model_with_feature())
        with pytest.raises(ValueError, match="unique"):
            mesh.with_ports(
                [
                    PortWaveguide(name="p", plane="zmin"),
                    PortWaveguide(name="p", plane="zmax"),
                ]
            )

    def test_boundary_condition_rewrite_keeps_ports(self):
        model = _model_with_feature()
        model.add_port(PortWaveguide(name="p1", plane="zmin"))
        mesh = _mesh(model)
        rewired = mesh.with_boundary_conditions({"xmin": "PEC"})
        assert rewired.ports == mesh.ports


class TestPerFaceBuffer:
    def test_declared_face_is_buffered(self):
        model = _model_with_feature()
        model.add_port(PortWaveguide(name="p1", plane="zmin"))
        mesh = _mesh(model)
        assert _is_buffered(mesh.grid.dz, "lo")

    def test_no_declaration_buffers_all_faces(self):
        mesh = _mesh(_model_with_feature())
        for deltas in (mesh.grid.dx, mesh.grid.dy, mesh.grid.dz):
            assert _is_buffered(deltas, "lo")
            assert _is_buffered(deltas, "hi")

    def test_declared_grid_matches_legacy_on_port_face(self):
        """The buffered axis end is bit-identical to the all-face path."""
        declared = _model_with_feature()
        declared.add_port(PortWaveguide(name="p1", plane="zmin"))
        legacy = _model_with_feature()
        z_declared = _mesh(declared).grid.z
        z_legacy = _mesh(legacy).grid.z
        np.testing.assert_array_equal(z_declared[:4], z_legacy[:4])


class TestAnalysisPickup:
    def test_missing_ports_error_mentions_add_port(self):
        from magnelio import AnalysisScatteringTD

        mesh = _mesh(_model_with_feature())
        with pytest.raises(ValueError, match="add_port"):
            AnalysisScatteringTD(mesh=mesh, f_max=10e9, verbose=False)
