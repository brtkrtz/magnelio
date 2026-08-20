"""Project store and CAD file import.

``open_project`` lives in the core ``magnelio`` namespace; this
component holds the reader/writer classes and geometry file I/O.
The one-shot save_project/load_project (io/hdf5.py) was removed;
the store supersedes it.

``import_step`` / ``import_brep`` read geometry drawn in a CAD
system into the geometry API.
"""

from magnelio.io.cad import import_brep, import_step
from magnelio.io.project import (
    LoadedGeometry,
    Project,
    ProjectStore,
    open_project,
    read_brep,
    write_brep,
)

__all__ = [
    "ProjectStore",
    "Project",
    "LoadedGeometry",
    "read_brep",
    "write_brep",
    "import_step",
    "import_brep",
]
