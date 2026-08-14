"""LumpedElement — declarative passive two-terminal circuit element.

The port-free counterpart of :class:`magnelio.ports.PortLumped`:
the same straight interior edge path and the same trapezoidal
companion models (:class:`SeriesRLC` / :class:`ParallelRLC`), but as a
pure passive load — no excitation, no S-matrix column, no recording.
Declared on the :class:`~magnelio.geo.GeometryModel` via
``add_element`` and carried by the mesh to the analysis, exactly like
declarative ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from magnelio.circuit.companion import ParallelRLC, SeriesRLC


@dataclass
class LumpedElement:
    """Passive lumped RLC element on a straight interior edge path.

    Parameters
    ----------
    name : str
        Unique identifier; shares one namespace with the port names
        of the model it is added to.
    start, end : tuple of float
        Endpoints in metres; must differ along exactly one Cartesian
        axis after grid snapping.
    element : SeriesRLC or ParallelRLC
        Trapezoidal companion model providing the terminal relation,
        e.g. ``SeriesRLC(R=100.0)`` for an ideal 100 Ω resistor.

    Examples
    --------
    >>> from magnelio import circuit
    >>> iso = circuit.LumpedElement(
    ...     name="iso",
    ...     start=(0.0, 0.8e-3, 10e-3),
    ...     end=(0.5e-3, 0.8e-3, 10e-3),
    ...     element=circuit.SeriesRLC(R=100.0),
    ... )
    """

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    element: Union[SeriesRLC, ParallelRLC]

    def __post_init__(self) -> None:
        if not isinstance(self.element, (SeriesRLC, ParallelRLC)):
            raise TypeError(
                f"LumpedElement {self.name!r}: element must be a SeriesRLC "
                f"or ParallelRLC companion model, got "
                f"{type(self.element).__name__}."
            )
