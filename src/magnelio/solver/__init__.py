"""Solver components — the FIT time-domain engine and eigenmode results."""

from magnelio.solver.eigenmode_result import EigenmodeResult
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt, spectral_dt

__all__ = [
    "FITTimeDomainSolver",
    "EigenmodeResult",
    "courant_dt",
    "spectral_dt",
]
