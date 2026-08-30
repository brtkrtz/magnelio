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

import math
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


class _WaveformDriven(Source):
    """The waveform binding every time-driven source shares.

    ``set_excitation`` stores the unit-peak :class:`Waveform`, the
    amplitude (in the source's own :attr:`~Source.amplitude_unit`) and
    the delay; :meth:`_drive` evaluates ``A · w(t − delay)``.  A source
    whose drive is *not* a time function (an initial field) implements
    the binding itself instead.

    Methods only — the three state fields are declared by each concrete
    dataclass, so nothing leaks into a subclass's dataclass defaults.
    """

    _waveform: Waveform | None
    _amplitude: float
    _delay: float

    def set_excitation(
        self,
        waveform: Waveform,
        *,
        amplitude: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        """Bind the waveform the source injects, with its weight.

        Parameters
        ----------
        waveform : Waveform
            Unit-peak time function of the drive.
        amplitude : float, default 1.0
            Peak drive in the source's :attr:`~Source.amplitude_unit`.
        delay : float, default 0.0
            Time offset [s] of the waveform.
        """
        if not isinstance(waveform, Waveform):
            raise TypeError(
                f"{type(self).__name__}.set_excitation takes a magnelio.signals.Waveform; "
                f"got {type(waveform).__name__}",
            )
        amplitude = float(amplitude)
        delay = float(delay)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite; got {amplitude!r}")
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay must be a non-negative finite time [s]; got {delay!r}")
        self._waveform = waveform
        self._amplitude = amplitude
        self._delay = delay

    def clear_excitation(self) -> None:
        self._waveform = None
        self._amplitude = 1.0
        self._delay = 0.0

    @property
    def waveform(self) -> Waveform | None:
        """The bound waveform, or ``None`` before :meth:`set_excitation`."""
        return self._waveform

    def _require_waveform(self) -> Waveform:
        if self._waveform is None:
            raise ValueError(
                f"source {self.name!r} has no waveform: bind one with "
                f"set_excitation(waveform, amplitude=..., delay=...) before the run",
            )
        return self._waveform

    def _drive(self, t):
        """Drive ``A · w(t − delay)`` at time(s) *t* [s]."""
        w = self._require_waveform()
        return self._amplitude * w(t - self._delay)


__all__ = ["Source"]
