"""LumpedElement — declarative passive two-terminal circuit element.

The port-free counterpart of :class:`magnelio.ports.PortLumped`:
the same interior edge path and the same trapezoidal
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
    """Passive lumped RLC element on an interior edge path.

    Parameters
    ----------
    name : str
        Unique identifier; shares one namespace with the port names
        of the model it is added to.
    start, end : tuple of float, optional
        Endpoints in metres — the two-point short form of *path*, and
        exclusive with it.  Under a clipping symmetry declaration they
        stay in full-model coordinates: an element crossing an electric
        symmetry plane is clipped to the meshed half automatically.
    path : Curve or sequence of points, optional
        The element's path: a :class:`~magnelio.geo.Curve`, or a
        sequence of at least two ``(x, y, z)`` points [m] read as
        polyline vertices.  Any direction is allowed — an oblique path
        is carried by a staircase of grid edges.  The path must not
        visit a grid edge twice: a two-terminal element is a series
        chain.
    samples_per_cell : int, default 4
        Path samples per smallest cell while rasterising.
    element : SeriesRLC or ParallelRLC
        Trapezoidal companion model providing the terminal relation,
        e.g. ``SeriesRLC(R=100.0)`` for an ideal 100 Ω resistor.
        Always the full-model values: under symmetry the solver
        internally scales the companion to the meshed half.

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
    start: tuple[float, float, float] | None = None
    end: tuple[float, float, float] | None = None
    element: Union[SeriesRLC, ParallelRLC] = None
    path: object = None
    samples_per_cell: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.element, (SeriesRLC, ParallelRLC)):
            raise TypeError(
                f"LumpedElement {self.name!r}: element must be a SeriesRLC "
                f"or ParallelRLC companion model, got "
                f"{type(self.element).__name__}."
            )
