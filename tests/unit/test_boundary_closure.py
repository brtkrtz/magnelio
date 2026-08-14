"""DD-103: the boundary closure is declared on the model, once.

Regression cover for the defect DD-103 replaced: a PEC *background*
used to force a PEC wall onto all six bbox faces, silently overriding
whatever closure the analysis declared.  A PMC symmetry plane became an
electric wall — wrong field symmetry, and on a port touching that plane
the inner conductor merged with the outer one, so the TEM path was lost
and the port resolved as a hollow TE/TM guide.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio.boundaries.boundary_conditions import (
    BoundaryConditions,
    bc_type_entries,
    cpml_thickness_of,
)
from magnelio.constants import ETA0
from magnelio.geo import Brick, Difference, GeometryModel, Union
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal.port_plane import BoxFace, PortPlane

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _wall_mask(mesh: Mesh, face: BoxFace) -> np.ndarray:
    """The tangential-edge PEC mask on one bbox face."""
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    flat = np.concatenate(
        [
            mesh.pec_mask_edges[0, :n_Ex],
            mesh.pec_mask_edges[1, :n_Ey],
            mesh.pec_mask_edges[2, :n_Ez],
        ]
    )
    plane = PortPlane.from_mesh(face, mesh)
    return np.concatenate([flat[plane.e_u_indices], flat[plane.e_v_indices]])


class TestDeclaration:
    def test_default_closes_every_face_with_pec(self):
        bc = BoundaryConditions()
        assert set(bc.to_dict().values()) == {"PEC"}
        assert cpml_thickness_of(bc) == 8

    def test_partial_dict_is_canonicalised(self):
        model = GeometryModel(boundary_conditions={"xmin": "PMC"})
        assert isinstance(model.boundary_conditions, BoundaryConditions)
        assert model.boundary_conditions.xmin == "PMC"
        assert model.boundary_conditions.xmax == "PEC"

    def test_invalid_type_rejected_at_declaration(self):
        with pytest.raises(ValueError, match="not valid"):
            GeometryModel(boundary_conditions={"xmin": "PErC"})

    def test_unknown_face_rejected(self):
        with pytest.raises(ValueError, match="unknown boundary face"):
            bc_type_entries({"frobnicate": "PEC"})


class TestWallMask:
    """The declaration decides the wall — not the background material."""

    def _shell_mesh(self, closure):
        """PEC background around a vacuum brick: a closed chamber."""
        pec = Material.pec()
        air = Material.air()
        model = GeometryModel(background=pec, boundary_conditions=closure)
        model.add(Brick(origin=(0.0, 0.0, 0.0), size=(4e-3, 4e-3, 8e-3), material=air))
        return Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=6, max_cell_size=1e-3),
            f_max=10e9,
        )

    def test_pec_background_does_not_close_a_pmc_face(self):
        """The DD-049 regression: bg=pec used to mask ALL six faces."""
        mesh = self._shell_mesh({"xmin": "PMC"})
        assert not _wall_mask(mesh, BoxFace.X_MIN).all()
        assert _wall_mask(mesh, BoxFace.X_MAX).all()
        assert _wall_mask(mesh, BoxFace.Y_MIN).all()

    def test_pec_background_does_not_close_a_cpml_face(self):
        """Same defect, absorbing flavour: a mirror in front of the PML."""
        mesh = self._shell_mesh({"zmax": "CPML"})
        assert not _wall_mask(mesh, BoxFace.Z_MAX).all()
        assert _wall_mask(mesh, BoxFace.Z_MIN).all()

    def test_all_pec_declaration_closes_the_chamber(self):
        mesh = self._shell_mesh(None)
        for face in BoxFace:
            assert _wall_mask(mesh, face).all(), face


class TestRedeclaration:
    """``with_boundary_conditions`` replaces a closure, it does not add."""

    def _grid(self):
        lin = np.linspace(0.0, 4e-3, 5)
        return GridLines(x=lin, y=lin, z=lin)

    def test_taking_a_wall_back_off(self):
        plate = {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
        closed = Mesh.from_grid(self._grid())
        direct = Mesh.from_grid(self._grid(), boundary_conditions=plate)
        assert np.array_equal(
            closed.with_boundary_conditions(plate).pec_mask_edges,
            direct.pec_mask_edges,
        )

    def test_path_independent(self):
        plate = {
            "ymin": "PEC",
            "ymax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
        opened = Mesh.from_grid(
            self._grid(),
            boundary_conditions={f: "PMC" for f in FACES},
        )
        closed = Mesh.from_grid(self._grid())
        assert np.array_equal(
            opened.with_boundary_conditions(plate).pec_mask_edges,
            closed.with_boundary_conditions(plate).pec_mask_edges,
        )

    def test_round_trip_restores_the_mask(self):
        closed = Mesh.from_grid(self._grid())
        there_and_back = closed.with_boundary_conditions(
            {f: "PMC" for f in FACES}
        ).with_boundary_conditions({f: "PEC" for f in FACES})
        assert np.array_equal(
            there_and_back.pec_mask_edges,
            closed.pec_mask_edges,
        )


class TestPersistence:
    """The closure round-trips with the mesh (it is mesh state now)."""

    def test_mesh_h5_round_trip(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from magnelio.io.project import _load_mesh, _save_mesh

        lin = np.linspace(0.0, 4e-3, 5)
        closure = BoundaryConditions(
            xmin="PMC",
            zmax="CPML",
            cpml_thickness_cells=12,
        )
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            boundary_conditions=closure,
        )
        path = tmp_path / "mesh.h5"
        with h5py.File(path, "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(path, "r") as f:
            back = _load_mesh(f)

        assert bc_type_entries(back.boundary_conditions) == bc_type_entries(closure)
        assert cpml_thickness_of(back.boundary_conditions) == 12
        assert np.array_equal(back.pec_mask_edges, mesh.pec_mask_edges)


class TestSymmetryPlanePort:
    """The reported defect, end to end.

    Rectangular coax halved by a magnetic symmetry plane at x = 0: the
    inner conductor touches that plane, the outer conductor is the PEC
    background.  With the plane closed as PEC the two conductors merge
    into one component and the port falls back to the hollow TE/TM
    path; as a declared PMC face they stay separate and the port is
    TEM.
    """

    def _model(self, closure):
        pec = Material.pec()
        air = Material.air()
        model = GeometryModel(background=pec, boundary_conditions=closure)
        outer = Brick(origin=(0.0, 0.0, 0.0), size=(6e-3, 6e-3, 10e-3), material=air)
        inner = Brick(origin=(0.0, 2e-3, 0.0), size=(2e-3, 2e-3, 10e-3), material=pec)
        model.add(Union(Difference(outer, inner)))
        model.add(inner)
        return model

    def _mesh(self, closure):
        return Mesh.from_geometry(
            self._model(closure),
            MeshControl(min_nodes_per_wavelength=6, max_cell_size=0.5e-3),
            f_max=10e9,
        )

    def test_port_on_a_pmc_symmetry_plane_resolves_as_tem(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.ports._modal import PortSpecMultiConductor
        from magnelio.ports.declarative import (
            PortWaveguide,
            resolve_declarative_port,
        )

        mesh = self._mesh({"xmin": "PMC"})
        spec = resolve_declarative_port(
            PortWaveguide(name="p", plane="zmin"),
            mesh,
        )
        assert isinstance(spec, PortSpecMultiConductor), (
            "half-model coax behind a magnetic symmetry plane must keep "
            "its two conductors and stay on the TEM path"
        )


class TestMagneticWallCapacitance:
    """z_line bookkeeping at tangential PMC window edges.

    The natural magnetic wall of the staggered grid sits half the outer
    dual cell beyond the outermost grid line, so the TEM capacitance
    quadrature must book the full boundary dual cell at a declared PMC
    face.  On the parallel-plate line the discrete TEM mode is exact,
    which makes ``z_line`` a machine-precision observable on ANY
    resolution.  Regression: the quadrature used to stop at the
    outermost line and reported ``η0·b/(a - 2d/3)`` on the on-model
    path — an O(h) bias of ``2/(3·Nx)``.
    """

    A = 10e-3
    B = 5e-3
    LENGTH = 20e-3
    F_MAX = 10e9

    def _z_line(self, declare_on_model, max_cell_size=None):
        from magnelio.analysis import AnalysisScatteringTD
        from magnelio.ports.declarative import PortWaveguide

        closure = BoundaryConditions(xmin="PMC", xmax="PMC")
        model = GeometryModel(
            boundary_conditions=closure if declare_on_model else None,
        )
        model.add(
            Brick(
                origin=(-self.A / 2, -self.B / 2, -self.LENGTH / 2),
                size=(self.A, self.B, self.LENGTH),
                material=Material.air(),
            )
        )
        model.add_port(PortWaveguide(name="p1", plane="zmin", n_modes=1))
        model.add_port(PortWaveguide(name="p2", plane="zmax", n_modes=1))
        mesh = Mesh.from_geometry(
            model,
            MeshControl(max_cell_size=max_cell_size),
            f_max=self.F_MAX,
        )
        if not declare_on_model:
            mesh = mesh.with_boundary_conditions(closure)
        analysis = AnalysisScatteringTD(mesh=mesh, f_max=self.F_MAX, verbose=False)
        return analysis.solve_ports()["p1"].modes[0].z_line

    def test_parallel_plate_z_line_exact_even_on_the_coarse_default_mesh(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        z_ref = ETA0 * self.B / self.A
        assert self._z_line(declare_on_model=True) == pytest.approx(z_ref, rel=1e-9)

    def test_post_meshing_declaration_reports_the_simulated_width(self):
        # Declaring PMC after meshing keeps the outermost lines in
        # place, so the effective walls sit half a cell OUTSIDE the
        # requested faces: the simulated line is a + d wide and z_line
        # must say so — exactly.
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        d = 0.3125e-3
        z_ref = ETA0 * self.B / (self.A + d)
        z = self._z_line(declare_on_model=False, max_cell_size=d)
        assert z == pytest.approx(z_ref, rel=1e-9)
