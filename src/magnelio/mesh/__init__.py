"""Mesh components.
``Mesh`` and ``MeshControl`` live in the core ``magnelio``
namespace; this component holds the grid container and face
identification for custom setups.
"""

from magnelio.mesh._planes import GridPlanes, PlaneRecord, PlaneSource
from magnelio.mesh.faces import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl

__all__ = [
    "GridLines",
    "GridPlanes",
    "PlaneRecord",
    "PlaneSource",
    "BoxFace",
]
