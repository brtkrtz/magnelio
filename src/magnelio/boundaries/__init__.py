"""Boundary-condition components.
``BoundaryConditions`` (the per-face closure declaration) lives in the
core ``magnelio`` namespace; this component holds the concrete
boundary classes for custom setups.
"""

from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.boundaries.periodic import PeriodicBoundary
from magnelio.boundaries.pmc import PMCBoundary

__all__ = [
    "PECBoundary",
    "PMCBoundary",
    "CPMLBoundary",
    "PeriodicBoundary",
]
