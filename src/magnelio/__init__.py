"""
Magnelio — Python library for full-wave 3D electromagnetic field
simulation.

This namespace is the **core**: the model container and run vocabulary
(``GeometryModel``, ``Material``, ``Mesh``/``MeshControl``,
``BoundaryConditions``, ``Excitation``), the problem classes
(``Analysis*``), the project-store entry points (``open_project``,
``resume``) and the progress-output default (``set_verbosity``).  Every other
public name lives in exactly one domain namespace —
``magnelio.geo``, ``magnelio.materials``, ``magnelio.mesh``,
``magnelio.boundaries``, ``magnelio.ports``, ``magnelio.sources``,
``magnelio.monitors``, ``magnelio.fields``, ``magnelio.circuit``,
``magnelio.signals``, ``magnelio.solver``, ``magnelio.analysis``,
``magnelio.post``, ``magnelio.plots``, ``magnelio.io``,
``magnelio.constants`` — and
underscore modules are internal.
"""

from magnelio._progress import get_verbosity, set_verbosity
from magnelio._version import __version__

# Problem classes + project store entry points
from magnelio.analysis import AnalysisEigenmode, AnalysisScatteringTD, AnalysisTD, resume

# Run vocabulary — one port or source bound to a waveform (DD-224)
from magnelio.analysis.excitation import Excitation

# Model vocabulary — boundary closure (DD-103)
from magnelio.boundaries.boundary_conditions import BoundaryConditions

# Model vocabulary — geometry container
from magnelio.geo import GeometryModel
from magnelio.io.project import open_project

# Model vocabulary — materials
from magnelio.materials.material import Material

# Model vocabulary — mesh
from magnelio.mesh.mesher import Mesh, MeshControl

__all__ = [
    "__version__",
    "GeometryModel",
    "Material",
    "Mesh",
    "MeshControl",
    "BoundaryConditions",
    "Excitation",
    "AnalysisTD",
    "AnalysisScatteringTD",
    "AnalysisEigenmode",
    "resume",
    "open_project",
    "set_verbosity",
    "get_verbosity",
]
