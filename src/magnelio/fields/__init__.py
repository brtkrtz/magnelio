"""Fields — the public container for a field snapshot on a grid.

:class:`FieldState` couples the six Yee-staggered field components to
the grid lines they live on, so a field can be read at its own sample
positions or at arbitrary points, sliced and plotted, and handed from
one analysis to another (an eigenmode into a time-domain start, a
monitor snapshot into a plot).
"""

from magnelio.fields.state import FieldState

__all__ = ["FieldState"]
