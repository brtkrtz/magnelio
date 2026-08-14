"""Project-store components.
``open_project`` lives in the core ``magnelio`` namespace; this
component holds the reader/writer classes and geometry file I/O.
The one-shot save_project/load_project (io/hdf5.py) was removed;
the store supersedes it.
"""

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
]
