"""Source — the contract every model source fulfils.

A source is declared on the :class:`~magnelio.GeometryModel` before
meshing (:meth:`~magnelio.GeometryModel.add_source`), travels with the
:class:`~magnelio.Mesh` (``mesh.sources``) and is driven by an
:class:`~magnelio.Excitation` that names it.  The solver-facing part of
the contract mirrors the port operators: an excitation is bound with
:meth:`Source.set_excitation`, the source is attached to a solver once,
and it injects into the fields every step.
"""

# Design: DD-224.

from __future__ import annotations

from abc import ABC, abstractmethod

from magnelio.signals.waveforms import Waveform


class Source(ABC):
    """Abstract base of every model source.

    Concrete sources are dataclasses with at least a ``name`` field and
    publish

    ``amplitude_unit``
        The unit of :attr:`~magnelio.Excitation.amplitude` for this
        source (``"V/m"`` for an incident field, …).
    ``excitable``
        Whether an excitation may name it.
    ``has_waveform``
        Whether the excitation drives it with a time function; an
        initial field has none — its excitation carries only the
        amplitude.
    ``writes_initial_field``
        Whether :meth:`attach` writes the solver's field state rather
        than injecting every step.
    """

    amplitude_unit: str = "1"
    excitable: bool = True
    has_waveform: bool = True
    # Whether ``attach`` writes into the solver's field state, so the
    # port operators must capture its trace before the first step.
    writes_initial_field: bool = False

    # -- excitation binding (solver-facing, like Port.set_excitation) --

    @abstractmethod
    def set_excitation(
        self,
        waveform: Waveform | None,
        *,
        amplitude: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        """Bind the waveform (and weight) the source injects.

        A source without a waveform (``has_waveform = False``) takes
        ``None`` and the amplitude only.
        """

    @abstractmethod
    def clear_excitation(self) -> None:
        """Drop the bound waveform — the source injects nothing."""

    # -- solver hooks --

    @abstractmethod
    def attach(self, solver) -> None:
        """Cache solver coefficients; called once from the solver's ``setup()``."""

    @abstractmethod
    def inject_E(self, fields, t_E: float) -> None:
        """Apply the source's E-side contribution after the E update."""

    @abstractmethod
    def inject_H(self, fields, t_H: float) -> None:
        """Apply the source's H-side contribution after the H update."""


__all__ = ["Source"]
