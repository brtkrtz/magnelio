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

from magnelio.analysis._base import _AnalysisBase
from magnelio.solver._eigenmode_3d import EigenmodeSolver3D
from magnelio.solver.eigenmode_result import EigenmodeResult

_SOLVERS = (None, "arpack-amg", "arpack-superlu")


@dataclass
class AnalysisEigenmode(_AnalysisBase):
    """High-level eigenmode analysis for 3D cavities.

    Parameters
    ----------
    mesh : Mesh
        The simulation mesh, carrying the boundary closure it was built
        with (declared on the ``GeometryModel`` or on
        ``Mesh.from_grid``).  PEC, PMC and Periodic faces are
        meaningful for an eigenmode problem; CPML is rejected.
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
    verbose : bool, optional
        Print solver progress.  ``None`` (the default) follows
        :func:`magnelio.set_verbosity`.
    project, geometry, params
        Project-store hooks: ``project`` names a directory the model
        and the result are written into, ``geometry`` the model to
        store with it, ``params`` a free dict recorded alongside.
    backend, precision, method
        The arguments every analysis shares.  The eigenmode solve runs
        in double precision on the CPU (ARPACK), so only the defaults
        ``"auto"`` / ``None`` are accepted here.
    phase_advance_deg : float, dict or None
        Bloch phase advance [degrees] across the mesh's ``"Periodic"``
        face pair — the phase by which the field in one period leads
        the next.  Sweeping it from 0 to 180 traces the dispersion
        diagram of the infinite periodic structure.  A number serves a
        single periodic axis; with several, pass ``{axis: degrees}``.
        ``None`` (default) is zero phase.  Phases other than 0 and 180
        degrees give complex mode fields and need the default
        (SuperLU) solver.
    """

    n_modes: int = 5
    sigma: float | None = None
    phase_advance_deg: float | dict[str, float] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in ("auto", "numpy"):
            raise ValueError(
                f"backend={self.backend!r}: the eigenmode solve runs on the CPU "
                f"(ARPACK); leave backend at 'auto'",
            )
        if self.precision not in (None, "double"):
            raise ValueError(
                f"precision={self.precision!r}: the eigenmode solve is double "
                f"precision; leave precision at None",
            )
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}; got {self.solver!r}")
        if not isinstance(self.n_modes, int) or self.n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer; got {self.n_modes!r}")

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
            verbose=self._verbose,
            phase_advance_deg=self.phase_advance_deg,
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
            "phase_advance_deg": self.phase_advance_deg,
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
            "params": dict(self.params or {}),
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
