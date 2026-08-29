"""Sources — model objects that inject fields, declared before meshing.

A source is declared on the :class:`~magnelio.GeometryModel` with
``add_source`` and driven at run time by an :class:`~magnelio.Excitation`
that names it (the waveform and amplitude live there).  Ports are
sources *and* loads and stay in :mod:`magnelio.ports`.
"""

from magnelio.sources.field_incident import SourceFieldIncident
from magnelio.sources.plane_wave import SourcePlaneWave

__all__ = [
    "SourceFieldIncident",
    "SourcePlaneWave",
]
