"""DD-123: declarative passive LumpedElement — declaration to solver wiring.

The element rides the ports' infrastructure (edge chain, trapezoidal
companion, solver update hook) but must stay invisible to everything
port-like: no S-matrix column, no excitation, no recording.  These
tests cover the declaration surface, the mesh carriage (including the
explicit-reconstruction traps), the builder, and the analysis wiring;
the physics anchor lives in
``validation/lumped_element_shunt_certificate.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.circuit import LumpedElement, ParallelRLC, SeriesRLC
from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortLumped
from magnelio.ports._lumped import LumpedElementOperator, build_lumped_element

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


# Interior by one cell in x and z: on the domain edge every chain edge
# is tangential to the default PEC wall and the builder rightly warns
# that the element is shorted along it.
def _element(label="iso", start=(1e-3, 0.0, 1e-3), end=(1e-3, 4e-3, 1e-3)):
    return LumpedElement(name=label, start=start, end=end, element=SeriesRLC(R=100.0))


def _grid(n=5, span=4e-3):
    lin = np.linspace(0.0, span, n)
    return GridLines(x=lin, y=lin, z=lin)


class TestDeclaration:
    def test_element_field_must_be_companion(self):
        with pytest.raises(TypeError, match="SeriesRLC or ParallelRLC"):
            LumpedElement(name="iso", start=(0, 0, 0), end=(0, 1e-3, 0), element=100.0)

    def test_add_element_rejects_non_elements(self):
        model = GeometryModel()
        with pytest.raises(TypeError, match="LumpedElement"):
            model.add_element(SeriesRLC(R=100.0))

    def test_names_share_one_namespace_with_ports(self):
        model = GeometryModel()
        model.add_port(
            PortLumped(name="feed", start=(0, 0, 0), end=(0, 1e-3, 0), Z0=50.0),
        )
        with pytest.raises(ValueError, match="duplicate"):
            model.add_element(_element(label="feed"))
        model.add_element(_element(label="iso"))
        with pytest.raises(ValueError, match="duplicate"):
            model.add_port(
                PortLumped(name="iso", start=(0, 0, 0), end=(0, 1e-3, 0), Z0=50.0),
            )
        with pytest.raises(ValueError, match="duplicate"):
            model.add_element(_element(label="iso"))


class TestMeshCarriage:
    def test_from_geometry_carries_elements(self):
        model = GeometryModel()
        model.add(
            Brick(
                origin=(0, 0, 0),
                size=(4e-3, 4e-3, 4e-3),
                material=Material.from_isotropic("air", epsilon=1.0),
            )
        )
        model.add_element(_element())
        from magnelio.mesh.mesher import MeshControl

        mesh = Mesh.from_geometry(model, MeshControl(), f_max=10e9)
        assert len(mesh.elements) == 1
        assert mesh.elements[0].name == "iso"

    def test_with_boundary_conditions_preserves_elements(self):
        # Trap: with_boundary_conditions reconstructs Mesh(...) field by
        # field — a forgotten elements= silently drops the declaration.
        mesh = Mesh.from_grid(_grid()).with_elements([_element()])
        rebuilt = mesh.with_boundary_conditions({f: "PMC" for f in FACES})
        assert len(rebuilt.elements) == 1

    def test_with_elements_checks_port_collision(self):
        mesh = Mesh.from_grid(_grid()).with_ports(
            [PortLumped(name="iso", start=(0, 0, 0), end=(0, 1e-3, 0), Z0=50.0)],
        )
        with pytest.raises(ValueError, match="unique"):
            mesh.with_elements([_element(label="iso")])

    def test_mesh_h5_round_trip_with_rlc_elements(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from magnelio.io.project import _load_mesh, _save_mesh

        elem = LumpedElement(
            name="iso",
            start=(0.0, 0.0, 0.0),
            end=(0.0, 4e-3, 0.0),
            element=ParallelRLC(R=100.0, C=1e-12),
        )
        # A PortLumped with an RLC companion rides along: its serialiser
        # used to asdict() the companion including the init=False state
        # fields, which the constructor rejected on reload.
        port = PortLumped(
            name="feed",
            start=(0.0, 0.0, 0.0),
            end=(0.0, 4e-3, 0.0),
            Z0=50.0,
            element=SeriesRLC(R=50.0, L=2e-9),
        )
        mesh = Mesh.from_grid(_grid()).with_ports([port]).with_elements([elem])
        path = tmp_path / "mesh.h5"
        with h5py.File(path, "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(path, "r") as f:
            back = _load_mesh(f)

        (e,) = back.elements
        assert e.name == "iso"
        assert isinstance(e.element, ParallelRLC)
        assert e.element.R == 100.0 and e.element.C == 1e-12
        (p,) = back.ports
        assert isinstance(p.element, SeriesRLC)
        assert p.element.R == 50.0 and p.element.L == 2e-9


class TestBuilder:
    def _mesh(self):
        return Mesh.from_grid(_grid())

    def test_builds_passive_operator(self):
        mesh = self._mesh()
        n_e = mesh.pec_mask_edges.size // 3 * 3  # flat E layout size proxy
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        op = build_lumped_element(_element(), mesh, m_eps, None, dt=1e-12)
        assert isinstance(op, LumpedElementOperator)
        assert type(op) is LumpedElementOperator  # not the port subclass
        assert op.Z0 == 0.0
        assert op.name == "iso"
        assert len(op.flat_edge_indices) == 4  # 4 cells across the 5-node span
        del n_e

    def test_companion_state_is_deep_copied(self):
        mesh = self._mesh()
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        spec = _element()
        op = build_lumped_element(spec, mesh, m_eps, None, dt=1e-12)
        assert op.element is not spec.element

    def test_oblique_path_is_a_staircase(self):
        """DD-269: a diagonal element is carried by a staircase chain.

        It used to be rejected outright ("exactly one Cartesian axis").
        The chain now mixes Ex and Ey edges, and its length is the
        staircase's, not the chord's.
        """
        # A larger grid so the diagonal stays clear of the PEC walls.
        mesh = Mesh.from_grid(_grid(n=9, span=8e-3))
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        diag = LumpedElement(
            name="diag",
            start=(1e-3, 1e-3, 1e-3),
            end=(5e-3, 5e-3, 1e-3),
            element=SeriesRLC(R=100.0),
        )
        op = build_lumped_element(diag, mesh, m_eps, None, dt=1e-12)
        assert set(op.edge_components) == {0, 1}  # Ex and Ey
        assert len(op.flat_edge_indices) == 8  # 4 + 4 steps, not 4
        assert all(s == 1.0 for s in op.edge_signs)

    def test_direction_is_honoured(self):
        """Reversing the terminals reverses the chain's polarity.

        The old two-point rasteriser sorted the endpoints and always
        produced ``+1`` signs, so a port declared "backwards" silently
        had the polarity of its forward twin.
        """
        mesh = self._mesh()
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        fwd = build_lumped_element(_element(), mesh, m_eps, None, dt=1e-12)
        rev = build_lumped_element(
            _element(start=(1e-3, 4e-3, 1e-3), end=(1e-3, 0.0, 1e-3)),
            mesh,
            m_eps,
            None,
            dt=1e-12,
        )
        assert sorted(fwd.flat_edge_indices) == sorted(rev.flat_edge_indices)
        assert all(s == 1.0 for s in fwd.edge_signs)
        assert all(s == -1.0 for s in rev.edge_signs)

    def test_self_crossing_path_rejected(self):
        """A two-terminal element is a series chain: one pass per edge."""
        mesh = self._mesh()
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        doubled = LumpedElement(
            name="back",
            path=[(1e-3, 0.0, 1e-3), (1e-3, 4e-3, 1e-3), (1e-3, 2e-3, 1e-3)],
            element=SeriesRLC(R=100.0),
        )
        with pytest.raises(ValueError, match="more than once"):
            build_lumped_element(doubled, mesh, m_eps, None, dt=1e-12)

    def test_multi_point_path(self):
        """A polyline element bends inside the model."""
        mesh = self._mesh()
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        bent = LumpedElement(
            name="bend",
            path=[(1e-3, 1e-3, 1e-3), (1e-3, 3e-3, 1e-3), (3e-3, 3e-3, 1e-3)],
            element=SeriesRLC(R=100.0),
        )
        op = build_lumped_element(bent, mesh, m_eps, None, dt=1e-12)
        assert len(op.flat_edge_indices) == 4
        assert set(op.edge_components) == {0, 1}

    def test_path_and_endpoints_are_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            build_lumped_element(
                LumpedElement(
                    name="both",
                    start=(0.0, 0.0, 0.0),
                    end=(0.0, 4e-3, 0.0),
                    path=[(0.0, 0.0, 0.0), (0.0, 4e-3, 0.0)],
                    element=SeriesRLC(R=100.0),
                ),
                self._mesh(),
                __import__(
                    "magnelio._operators.material_matrices", fromlist=["build_M_eps"]
                ).build_M_eps(self._mesh()),
                None,
                dt=1e-12,
            )

    def test_pec_shorted_chain_warns(self):
        """An edge held at zero cannot carry the element's injection."""
        mesh = self._mesh()
        from magnelio._operators.material_matrices import build_M_eps

        m_eps = build_M_eps(mesh)
        on_wall = _element(start=(0.0, 0.0, 0.0), end=(0.0, 4e-3, 0.0))
        with pytest.warns(UserWarning, match="held at zero"):
            build_lumped_element(on_wall, mesh, m_eps, None, dt=1e-12)


class TestAnalysisWiring:
    GAP, WIDTH, LENGTH = 4e-3, 12e-3, 30e-3
    BC = {
        "xmin": "PMC",
        "xmax": "PMC",
        "ymin": "PEC",
        "ymax": "PEC",
        "zmin": "PEC",
        "zmax": "PEC",
    }

    def _analysis(self, elements):
        from magnelio.analysis.scattering_td import AnalysisScatteringTD
        from magnelio.mesh import BoxFace
        from magnelio.ports import PortSpecMultiConductor

        grid = GridLines(
            x=np.linspace(-self.WIDTH / 2, self.WIDTH / 2, 7),
            y=np.linspace(-self.GAP / 2, self.GAP / 2, 5),
            z=np.linspace(-self.LENGTH / 2, self.LENGTH / 2, 61),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=dict(self.BC))
        if elements:
            mesh = mesh.with_elements(elements)
        return AnalysisScatteringTD(
            mesh=mesh,
            ports=[
                PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
                PortSpecMultiConductor(name="m2", plane=BoxFace.Z_MAX, n_modes=1),
            ],
            f_max=6e9,
            verbose=False,
            backend="numpy",
        )

    def _shunt(self, label="shunt"):
        return LumpedElement(
            name=label,
            start=(0.0, -self.GAP / 2, 0.0),
            end=(0.0, self.GAP / 2, 0.0),
            element=SeriesRLC(R=100.0),
        )

    def test_element_acts_but_stays_out_of_the_s_matrix(self):
        f_axis = np.array([1e9])
        kwargs = dict(f_axis=f_axis, excited=["m1"], total_time_steps=4000, energy_stop_db=None)
        res_bare = self._analysis([]).run(**kwargs)
        res_shunt = self._analysis([self._shunt()]).run(**kwargs)

        assert sorted(res_shunt.channels) == sorted(res_bare.channels)
        assert all("shunt" not in str(c) for c in res_shunt.channels)
        # A matched bare line barely reflects; a 100 ohm gap shunt does.
        s11_bare = float(np.abs(res_bare.S("m1", "m1"))[0])
        s11_shunt = float(np.abs(res_shunt.S("m1", "m1"))[0])
        assert s11_shunt > s11_bare + 0.1

    def test_elements_resolved_from_mesh_and_validated(self):
        from magnelio.analysis.scattering_td import AnalysisScatteringTD

        ana = self._analysis([self._shunt()])
        assert len(ana.elements) == 1

        with pytest.raises(TypeError, match="LumpedElement"):
            bad = self._analysis([])
            AnalysisScatteringTD(
                mesh=bad.mesh,
                ports=bad.ports,
                elements=[SeriesRLC(R=100.0)],
                f_max=6e9,
                verbose=False,
            )

    def test_element_name_may_not_shadow_a_port(self):
        from magnelio.analysis.scattering_td import AnalysisScatteringTD

        base = self._analysis([])
        with pytest.raises(ValueError, match="unique together"):
            AnalysisScatteringTD(
                mesh=base.mesh,
                ports=base.ports,
                elements=[self._shunt(label="m1")],
                f_max=6e9,
                verbose=False,
            )
