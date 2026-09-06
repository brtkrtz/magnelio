"""Fields — the public container for a field snapshot on a grid.

:class:`FieldState` couples the six Yee-staggered field components to
the grid lines they live on, so a field can be read at its own sample
positions or at arbitrary points, sliced and plotted, and handed from
one analysis to another (an eigenmode into a time-domain start, a
monitor snapshot into a plot).

:class:`FieldRecording` and :class:`FieldSpectrum` are series of such
snapshots on one grid — over time, and over frequency as complex
patterns — with the same vocabulary frame by frame.

:class:`SurfaceRecording` is the other coupling object: the tangential
fields on a closed box over time, written by
:class:`~magnelio.monitors.MonitorFieldSurface` and replayed by
:class:`~magnelio.sources.SourceFieldSurface`.
"""

from magnelio.fields.series import FieldRecording, FieldSpectrum
from magnelio.fields.state import FieldState
from magnelio.fields.surface import ComponentRecord, FaceRecord, SurfaceRecording

__all__ = [
    "ComponentRecord",
    "FaceRecord",
    "FieldRecording",
    "FieldSpectrum",
    "FieldState",
    "SurfaceRecording",
]
