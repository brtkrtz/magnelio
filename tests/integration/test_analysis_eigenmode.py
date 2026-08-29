"""Integration tests for the AnalysisEigenmode high-level API.

Verifies that AnalysisEigenmode.run() returns an EigenmodeResult with correct
frequencies and FieldState mode shapes for a rectangular PEC cavity, and the
KB-011 behaviour on a sparsely filled high-contrast cavity (auto-shift
escalation, loud under-delivery).

The project-store eigenmode round-trip lives in
``tests/unit/test_project_store.py`` (DD-070).
"""

import math
import warnings

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.eigenmode_result import EigenmodeResult

C0 = 299_792_458.0


def _cavity_mesh(a, b, c, Nx, Ny, Nz):
    grid = GridLines(
        x=np.linspace(0, a, Nx + 1),
        y=np.linspace(0, b, Ny + 1),
        z=np.linspace(0, c, Nz + 1),
    )
    return Mesh.from_grid(grid)


def f_analytical(m, n, p, a, b, c):
    return (C0 / 2) * math.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / c) ** 2)


class TestEigenAnalysis:
    def test_returns_eigenmode_result(self):
        a, b, c = 30e-3, 20e-3, 15e-3
        mesh = _cavity_mesh(a, b, c, 10, 8, 6)

        analysis = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False)
        result = analysis.run()

        assert isinstance(result, EigenmodeResult)
        assert result.n_modes == 3
        assert len(result.modes) == 3
        assert result.mesh is mesh

    def test_frequencies_match_analytical(self):
        a, b, c = 30e-3, 20e-3, 15e-3
        mesh = _cavity_mesh(a, b, c, 20, 14, 10)

        result = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()

        analytical = sorted(
            [
                f_analytical(1, 0, 1, a, b, c),
                f_analytical(0, 1, 1, a, b, c),
                f_analytical(1, 1, 0, a, b, c),
            ]
        )

        for i, (f_num, f_ana) in enumerate(zip(result.frequencies, analytical)):
            err = abs(f_num - f_ana) / f_ana * 100
            assert err < 2.0, (
                f"Mode {i}: {f_num / 1e9:.4f} vs {f_ana / 1e9:.4f} GHz, error={err:.2f}%"
            )

    def test_modes_are_fieldstate(self):
        mesh = _cavity_mesh(30e-3, 20e-3, 15e-3, 10, 8, 6)
        result = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()

        for i, mode in enumerate(result.modes):
            assert isinstance(mode, FieldState), f"Mode {i} is not FieldState"
            # E-field should be non-zero
            assert np.linalg.norm(mode.e_flat) > 0, f"Mode {i}: E=0"
            # H-field should be non-zero
            assert np.linalg.norm(mode.h_flat) > 0, f"Mode {i}: H=0"

    def test_fieldstate_shapes(self):
        Nx, Ny, Nz = 10, 8, 6
        mesh = _cavity_mesh(30e-3, 20e-3, 15e-3, Nx, Ny, Nz)
        result = AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()

        mode = result.modes[0]
        assert mode.Ex.shape == (Nx, Ny + 1, Nz + 1)
        assert mode.Ey.shape == (Nx + 1, Ny, Nz + 1)
        assert mode.Ez.shape == (Nx + 1, Ny + 1, Nz)
        assert mode.Hx.shape == (Nx + 1, Ny, Nz)
        assert mode.Hy.shape == (Nx, Ny + 1, Nz)
        assert mode.Hz.shape == (Nx, Ny, Nz + 1)

    def test_solver_info(self):
        mesh = _cavity_mesh(30e-3, 20e-3, 15e-3, 10, 8, 6)
        result = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()

        assert "backend" in result.solver_info
        assert "n_modes_found" in result.solver_info
        assert result.solver_info["n_modes_found"] == 3

    def test_repr(self):
        mesh = _cavity_mesh(30e-3, 20e-3, 15e-3, 10, 8, 6)
        result = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()
        s = repr(result)
        assert "EigenmodeResult" in s
        assert "n_modes=3" in s
        assert "GHz" in s


@pytest.fixture(scope="module")
def puck_mesh():
    """Sparsely filled high-contrast cavity (the KB-011 fixture, coarse).

    A ceramic ring (eps_r = 45, ~1 % of the volume) in an air-filled
    PEC box.  The filled-cavity auto shift lands a factor ~30 below
    the true fundamental here, so the first shift-invert attempt
    under-delivers and the escalation path is exercised.
    """
    from magnelio.geo import Brick, Cylinder, Difference, GeometryModel

    air = Material.from_isotropic(name="air", epsilon=1.0)
    ceramic = Material.from_isotropic(name="ceramic", epsilon=45.0)
    w, h = 20.0e-3, 6.0e-3
    puck = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=4.0e-3,
        height=h,
        inner_radius=2.0e-3,
        axis="z",
        material=ceramic,
    )
    box = Brick(origin=(-w / 2, -w / 2, 0.0), size=(w, w, h), material=air)
    model = GeometryModel()
    model.add(Difference(box, puck))
    model.add(puck)
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=3.0e9)
    return mesh.with_boundary_conditions(
        {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )


class TestSparseHighContrastCavity:
    """KB-011: no silent under-delivery on high-contrast cavities."""

    def test_auto_shift_escalates_to_full_mode_count(self, puck_mesh):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = AnalysisEigenmode(mesh=puck_mesh, n_modes=6, verbose=False).run()
        freqs = np.asarray(result.frequencies)
        assert freqs.size == 6
        assert np.all(np.diff(freqs) >= 0)
        # The ring-loaded fundamental on this grid (air bore since the
        # nested-contour winding of the section engine; the solid puck
        # sits at 2.23 GHz), and its degenerate first pair (equal to
        # solver tolerance, not just to the mesh).
        assert freqs[0] == pytest.approx(2.6566e9, rel=2e-3)
        assert freqs[1] == pytest.approx(freqs[2], rel=1e-6)

    def test_under_delivery_warns(self, puck_mesh):
        # An explicit shift pinned far below the fundamental starves
        # ARPACK on null-space vectors; that must be loud, and an
        # explicit sigma must stay a single attempt (no escalation —
        # the request may still grow at the same shift).  At 100 MHz
        # the third grow (k = 46) happens to deliver on this grid;
        # from 50 MHz down every grow returns artefacts only.
        sigma_bad = (2.0 * math.pi * 3.0e7) ** 2
        with pytest.warns(RuntimeWarning, match="of 6 requested"):
            result = AnalysisEigenmode(
                mesh=puck_mesh, n_modes=6, sigma=sigma_bad, verbose=False
            ).run()
        assert len(result.frequencies) < 6
        assert result.solver_info["n_modes_found"] < 6


class TestSharedAnalysisArguments:
    """``AnalysisEigenmode`` on the shared analysis base (DD-224 Phase C)."""

    @pytest.fixture
    def small_mesh(self):
        grid = GridLines(
            x=np.linspace(0, 10e-3, 6),
            y=np.linspace(0, 6e-3, 4),
            z=np.linspace(0, 14e-3, 8),
        )
        return Mesh.from_grid(grid)

    def test_rejects_other_meshes_methods(self, small_mesh):
        with pytest.raises(ValueError, match="tetrahedral"):
            AnalysisEigenmode(mesh=small_mesh, method="fem")

    def test_rejects_gpu_and_single_precision(self, small_mesh):
        with pytest.raises(ValueError, match="CPU"):
            AnalysisEigenmode(mesh=small_mesh, backend="cupy")
        with pytest.raises(ValueError, match="double"):
            AnalysisEigenmode(mesh=small_mesh, precision="single")

    def test_rejects_unknown_solver_and_n_modes(self, small_mesh):
        with pytest.raises(ValueError, match="solver must be"):
            AnalysisEigenmode(mesh=small_mesh, solver="pardiso")
        with pytest.raises(ValueError, match="n_modes"):
            AnalysisEigenmode(mesh=small_mesh, n_modes=0)

    def test_params_reach_the_project(self, small_mesh, tmp_path):
        project = AnalysisEigenmode(
            mesh=small_mesh,
            n_modes=1,
            verbose=False,
            params={"gap_mm": 1.5},
            project=tmp_path / "swept",
        ).run()
        assert project.params == {"gap_mm": 1.5}

    def test_field_is_a_public_field_state(self, small_mesh):
        from magnelio.fields import FieldState as PublicFieldState

        result = AnalysisEigenmode(mesh=small_mesh, n_modes=1, verbose=False).run()
        field = result.field(0)
        assert isinstance(field, PublicFieldState)
        assert field.grid is small_mesh.grid
        # the raw mode is a grid quantity; the public one is V/m
        assert np.abs(field.Ey).max() > 0.0
        with pytest.raises(IndexError):
            result.field(result.n_modes)
