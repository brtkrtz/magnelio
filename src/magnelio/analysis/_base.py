"""The private base every problem class shares (DD-224).

Holds the arguments that mean the same on every analysis — the mesh,
the compute backend and precision, the discretisation ``method`` and
algebraic ``solver`` selectors, the project-store hooks — and validates
them once, so a new problem class only adds what is specific to its
question.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.mesh.mesher import Mesh

_METHODS = ("auto", "fit", "fem", "bem")
_BACKENDS = ("auto", "numpy", "cupy")
_PRECISIONS = (None, "single", "double")


@dataclass
class _AnalysisBase:
    """Arguments common to every ``Analysis*`` class.

    ``method`` names the discretisation: ``"auto"`` follows the mesh's
    element type (hexahedral → ``"fit"``); ``"fem"`` and ``"bem"`` are
    reserved for tetrahedral and surface meshes and are rejected on
    today's hexahedral meshes.  ``solver`` names the algebraic solver
    where one exists (the eigenmode analysis); a time-domain march has
    none.
    """

    mesh: Mesh
    verbose: bool | None = None
    project: object | None = None
    geometry: object | None = None
    params: dict | None = None
    backend: str = "auto"
    precision: str | None = None
    method: str = "auto"
    solver: str | None = None

    @property
    def _verbose(self) -> bool:
        """The effective verbosity: the local setting, else the global one.

        ``verbose=None`` (the default) means "follow
        :func:`magnelio.set_verbosity`", so the process-wide setting
        reaches nested work — a port refinement that meshes and solves
        ports per rung passes its own setting down instead of silencing
        the inner calls.  ``True``/``False`` override it locally.
        """
        from magnelio._progress import get_verbosity  # noqa: PLC0415

        return get_verbosity() if self.verbose is None else bool(self.verbose)

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"backend must be 'auto', 'numpy' or 'cupy'; got {self.backend!r}",
            )
        if self.precision not in _PRECISIONS:
            raise ValueError(
                f"precision must be 'single', 'double' or None; got {self.precision!r}",
            )
        if self.method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS}; got {self.method!r}",
            )
        if self.method in ("fem", "bem"):
            raise ValueError(
                f"method={self.method!r} needs a "
                f"{'tetrahedral' if self.method == 'fem' else 'surface'} mesh; "
                f"this mesh is hexahedral, which is discretised by 'fit' (method='auto').",
            )
        if self.solver is not None and not isinstance(self.solver, str):
            raise TypeError(f"solver must be a string or None; got {type(self.solver).__name__}")
        if self.params is not None and not isinstance(self.params, dict):
            raise TypeError(f"params must be a dict or None; got {type(self.params).__name__}")

    @property
    def resolved_method(self) -> str:
        """The discretisation this analysis runs: ``"fit"`` on a hexahedral mesh."""
        return "fit" if self.method == "auto" else self.method


__all__ = ["_AnalysisBase"]
