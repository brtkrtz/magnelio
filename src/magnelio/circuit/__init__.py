"""Circuit — lumped elements, the shared curve rasteriser, edge tools.
``SeriesRLC``/``ParallelRLC`` are trapezoidal companion models; a port
carries them via ``PortLumped(..., element=...)`` (the port supplies
the endpoints, the element the terminal relation), and a passive
in-circuit load is declared as a :class:`LumpedElement` on the
geometry model via ``add_element``.
"""

from magnelio.circuit.companion import ParallelRLC, SeriesRLC
from magnelio.circuit.element import LumpedElement
from magnelio.circuit.rasterize import EdgePath, integrate_E, rasterize_curve

__all__ = [
    "SeriesRLC",
    "ParallelRLC",
    "LumpedElement",
    "EdgePath",
    "rasterize_curve",
    "integrate_E",
]
