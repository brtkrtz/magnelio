"""Plotting components — the free functions behind the ``.plot()`` methods.

The primary plotting path is the methods on the objects themselves
(``model.plot()``, ``model.plot_cross_section()``, ``report.plot()``,
``monitor.plot()``, ``result.plot_s()``); this module is their public
home for direct use on objects you assembled yourself.
"""

from magnelio.post.plot_field import (
    plot_field_scalar,
    plot_field_vector,
)
from magnelio.post.plot_geometry import (
    plot_cross_section,
    show_geometry,
)
from magnelio.post.plot_pattern import (
    plot_pattern_3d,
    plot_pattern_cut,
)

__all__ = [
    "plot_cross_section",
    "show_geometry",
    "plot_field_scalar",
    "plot_field_vector",
    "plot_pattern_cut",
    "plot_pattern_3d",
]
