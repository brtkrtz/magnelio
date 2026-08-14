"""AnalysisEigenmode — high-level eigenmode analysis workflow.

Takes a mesh and boundary conditions, runs the 3D eigenmode solver, and
returns an :class:`~magnelio.solver.eigenmode_result.EigenmodeResult` with
resonant frequencies and E/H field patterns.

Example
-------
>>> result = AnalysisEigenmode(mesh=mesh, n_modes=5).run()
>>> print(result.frequencies / 1e9)  # GHz
>>> result.modes[0].Ex  # E_x of lowest mode
"""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D
from magnelio.solver.eigenmode_result import EigenmodeResult


@dataclass
class AnalysisEigenmode:
    """High-level eigenmode analysis for 3D cavities.

    Parameters
    ----------
    mesh : Mesh
        The simulation mesh, carrying the boundary closure it was built
        with (declared on the ``GeometryModel`` or on
        ``Mesh.from_grid``).  Only PEC and PMC are meaningful for an
        eigenmode problem.
    n_modes : int
        Number of physical resonant modes to find (default 5).
    solver : str or None
        Inner-solve strategy.  ``None`` → auto-dispatch (SuperLU for small,
        AMG-CG for large).  ``"arpack-amg"`` forces AMG, ``"arpack-superlu"``
        forces SuperLU.
    sigma : float or None
        ARPACK shift σ [(rad/s)²].  ``None`` → auto-estimated from
        geometry and materials; if a solve returns fewer physical
        modes than requested, the shift is raised automatically and
        the modes found so far are kept.  Pass an explicit value
        (``sigma=(2*pi*f_estimate)**2``) to pin the shift; a run that
        still returns fewer than ``n_modes`` modes emits a
        ``RuntimeWarning`` either way.
    verbose : bool
        Print solver progress information.
    """

    mesh: Mesh
    n_modes: int = 5
    solver: str | None = None
    sigma: float | None = None
    verbose: bool = True
    project: object | None = None
    geometry: object | None = None

    @property
    def boundary_conditions(self) -> dict[str, str]:
        """The mesh's boundary closure as ``{face: type_str}``."""
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            bc_type_entries,
        )

        return bc_type_entries(self.mesh.boundary_conditions)

    def run(self):
        """Execute the eigenmode analysis.

        Returns
        -------
        EigenmodeResult or Project
            Frequencies [Hz] and E/H field patterns for each mode.  When
            ``project`` is set, the model and the eigenmode result are
            written into the project directory and a read-only
            :class:`~magnelio.io.project.Project` reader is returned
            instead (its ``.eigenmodes`` yields the ``EigenmodeResult``).
            Eigenmode analysis has no time-marching state, so it is a
            one-shot result — the streaming/resume machinery does not
            apply.
        """
        grid = self.mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        eigen_solver = EigenmodeSolver3D(
            n_modes=self.n_modes,
            boundary_conditions=self.boundary_conditions,
            solver=self.solver,
            sigma=self.sigma,
            verbose=self.verbose,
        )

        freq_hz, E_modes, H_modes = eigen_solver.solve(self.mesh)

        modes = EigenmodeResult._modes_from_flat(
            E_modes,
            H_modes,
            Nx,
            Ny,
            Nz,
        )

        solver_info = {
            "backend": eigen_solver.solver or "auto",
            "n_modes_requested": self.n_modes,
            "n_modes_found": len(freq_hz),
            "grid": f"{Nx}x{Ny}x{Nz}",
            # Omitted faces default to PEC (solver convention); wall-loss
            # postprocessing reads this to enumerate PEC domain walls.
            "boundary_conditions": dict(self.boundary_conditions),
        }

        result = EigenmodeResult(
            frequencies=freq_hz,
            modes=modes,
            mesh=self.mesh,
            solver_info=solver_info,
        )
        if self.project is not None:
            return self._write_project(result)
        return result

    def _write_project(self, result):
        """Write the model + eigenmode result into the project dir.

        Pointing ``project`` at an existing directory keeps the shared
        model and (re)writes the eigenmode result.  Returns a read-only
        :class:`~magnelio.io.project.Project` reader.
        """
        from pathlib import Path  # noqa: PLC0415

        from magnelio.io.project import ProjectStore, open_project  # noqa: PLC0415

        path = Path(self.project)
        setup = {
            "analysis": "AnalysisEigenmode",
            "n_modes": int(self.n_modes),
            "boundary_conditions": dict(self.boundary_conditions),
        }
        if (path / "project.json").exists():
            store = ProjectStore(path)
        else:
            store = ProjectStore.create(
                path,
                self.mesh,
                geometry=self.geometry,
                setup=setup,
            )
        store.write_eigenmodes(result)
        return open_project(path)

    def __repr__(self) -> str:
        return (
            f"AnalysisEigenmode(n_modes={self.n_modes}, "
            f"bcs={self.boundary_conditions or 'all-PEC'})"
        )
