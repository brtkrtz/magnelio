"""The scattering-result contract shared by RAM and store results.

One interface, two implementations: the in-RAM
:class:`~magnelio.analysis.scattering_td.ScatteringTDResult` returned by
``AnalysisScatteringTD.run()`` without a project, and the store-backed
:class:`~magnelio.io.project.Project` reader returned with one.  User
scripts must work identically against either, so the contract is pinned
here — as a :class:`typing.Protocol` for typing, as
:class:`ScatteringResultMixin` for the accessors that derive from
``S(...)``, and cross-checked by
``tests/integration/test_result_contract.py`` running the same
assertions over both implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RunSettings:
    """Settings a finished run was produced with, readable off the result.

    All fields are optional: the in-RAM result fills what the analysis
    knew at run time; the store-backed reader fills what the project
    recorded (older stores may lack individual entries).
    """

    f_max: float | None = None
    f_min: float | None = None
    n_freq: int | None = None
    dt: float | None = None
    n_actual_steps: int | None = None
    accuracy: str | None = None
    energy_stop_db: float | None = None
    port_signal_stop_db: float | None = None
    taper_signals: bool | None = None
    # Why the marching ended ("energy", "port_signal",
    # "port_signal_stall", "runtime_cap", "steps", "aborted") and the
    # port |V|-envelope level below peak it reached — the provenance of
    # a truncated record ("stall"/"cap" runs stopped short of their
    # criterion; the level bounds the truncation residual).
    stop_reason: str | None = None
    final_port_signal_db: float | None = None
    precision: str | None = None
    backend: str | None = None
    port_model_used: str | None = None


class ScatteringResultMixin:
    """Accessors derived purely from ``S(...)`` — shared verbatim."""

    def phase(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
        deg: bool = True,
        unwrap: bool = True,
        f_axis=None,
    ) -> np.ndarray:
        """Phase of one S-parameter over the frequency axis.

        Parameters
        ----------
        out_port, in_port : str
            Observed / excited port names (modes via ``mode_out`` /
            ``mode_in``), as in :meth:`S`.
        deg : bool, default True
            Return degrees; ``False`` returns radians.
        unwrap : bool, default True
            Unwrap 2π discontinuities along the frequency axis.
        f_axis : array-like, optional
            Custom frequency axis; recomputed from the recorded
            signals like :meth:`S`.

        Returns
        -------
        np.ndarray
            Phase per frequency point.
        """
        s = self.S(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
            f_axis=f_axis,
        )
        ph = np.angle(s)
        if unwrap:
            ph = np.unwrap(ph)
        return np.degrees(ph) if deg else ph

    def plot_s(self, *pairs, db=True, floor_db=-200.0, ax=None):
        """Plot S-parameter magnitudes over frequency.

        Parameters
        ----------
        *pairs : tuple
            Channels to plot, each ``(out_port, in_port)`` or
            ``(out_port, in_port, mode_out, mode_in)``.  Without
            arguments every recorded channel of every excitation is
            plotted.
        db : bool, default True
            Magnitude in dB (with *floor_db*) instead of linear.
        floor_db : float, default -200.0
            Clip floor for the dB display.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created otherwise.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if not pairs:
            pairs = tuple(
                (out_port, in_port, mode_out, mode_in)
                for (in_port, mode_in) in self.excitations
                for (out_port, mode_out) in self.channels
            )
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        f_ghz = np.asarray(self.f_axis) / 1e9
        multi_mode = any(len(p) == 4 and (p[2] or p[3]) for p in pairs)
        for p in pairs:
            out_port, in_port = p[0], p[1]
            mode_out = p[2] if len(p) > 2 else 0
            mode_in = p[3] if len(p) > 3 else 0
            if db:
                y = self.db(
                    out_port, in_port, mode_out=mode_out, mode_in=mode_in, floor_db=floor_db
                )
            else:
                y = np.abs(self.S(out_port, in_port, mode_out=mode_out, mode_in=mode_in))
            label = f"S({out_port} ← {in_port})"
            if multi_mode:
                label = f"S({out_port}:{mode_out} ← {in_port}:{mode_in})"
            ax.plot(f_ghz, y, label=label)
        ax.set_xlabel("f / GHz")
        ax.set_ylabel("|S| / dB" if db else "|S|")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax

    def _channel_cutoffs(self) -> dict | None:
        """Per-channel cut-off frequency [Hz], or ``None`` if unknown.

        Backs the export warning about propagating modes left out of a
        Touchstone file.  Both implementations override it from their
        port-mode records; the fallback keeps the exports working for
        any other holder of the contract.
        """
        return None

    def _warn_export(self, channels) -> None:
        """Warn about propagating modes the export would leave out."""
        from magnelio.post.sparameter_result import (  # noqa: PLC0415
            warn_unexported_modes,
        )

        s_params = self.s_params
        exported = s_params.export_channels(channels)
        warn_unexported_modes(
            exported,
            s_params.channels,
            self._channel_cutoffs(),
            float(np.max(np.asarray(s_params.f_axis, dtype=float))),
            stacklevel=4,
        )

    def to_touchstone(self, path, *, channels=None) -> None:
        """Write the S-matrix as a Touchstone ``.sNp`` file.

        Exports the square sub-matrix over the excited channels — one
        Touchstone port per channel, so a multi-mode port occupies one
        port per mode.  Channels that were observed but never excited
        are dropped from rows and columns alike; they carry a
        reflection-free boundary throughout the run, so the export is
        the network seen with them *matched*, the same quantity a
        network analyser measures with its unused ports terminated.

        Warns when a port that *is* exported carries propagating modes
        that the export leaves out: the file then looks like a
        complete N-port while the mode conversion at that port is
        missing from it.

        The ``.sNp`` extension must agree with the number of exported
        channels — Touchstone records the port count nowhere else — so
        a mismatch raises instead of writing an unreadable file.  A
        path without an extension gets the matching one.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.  ``<name>.s{N}p``, or ``<name>`` to have the
            extension filled in.
        channels : sequence of str or (str, int), optional
            Select the exported sub-network explicitly, e.g.
            ``["port1", "port3"]`` to cut a two-port out of a fully
            excited three-port.  A bare port name means mode 0.  Every
            entry must have been excited.
        """
        self._warn_export(channels)
        self.s_params.to_touchstone(path, channels=channels)

    def to_skrf(self, name: str = "magnelio", *, channels=None):
        """Return the S-matrix as a ``skrf.Network``.

        Requires scikit-rf (extra ``magnelio[interop]``).  Same
        sub-matrix, channel selection and warning as
        :meth:`to_touchstone`.

        Parameters
        ----------
        name : str, optional
            Network name.
        channels : sequence of str or (str, int), optional
            Explicit channel selection, as in :meth:`to_touchstone`.

        Returns
        -------
        skrf.Network
        """
        self._warn_export(channels)
        return self.s_params.to_skrf(name=name, channels=channels)


@runtime_checkable
class ScatteringResult(Protocol):
    """Everything a scattering result guarantees, RAM- or store-backed.

    A script written against this protocol runs unchanged whether
    ``AnalysisScatteringTD.run()`` returned an in-RAM
    :class:`~magnelio.analysis.ScatteringTDResult` (no project) or a
    :class:`~magnelio.io.project.Project` reader (with one).  The
    protocol is :func:`~typing.runtime_checkable`, so ``isinstance(res,
    ScatteringResult)`` holds for both.

    The members below are what both implementations provide; each
    implementation documents its own behaviour, and the two are
    cross-checked by running one shared set of assertions over both.

    - :attr:`f_axis` — frequency axis of the S-matrix [Hz]
    - :attr:`channels` / :attr:`excitations` — the observed and excited
      ``(port, mode)`` pairs
    - :attr:`settings` — the run's :class:`RunSettings`
    - :meth:`S`, :meth:`db`, :meth:`phase` — one S-parameter over
      frequency, as a complex number, in dB, or as a phase
    - :meth:`a`, :meth:`b` — incident and outgoing power waves in time

    Three further accessors come from :class:`ScatteringResultMixin`
    and are shared verbatim rather than reimplemented: ``plot_s`` and
    the ``to_touchstone`` / ``to_skrf`` exports.
    """

    @property
    def f_axis(self) -> np.ndarray:
        """Frequency axis of the S-matrix [Hz], ascending."""
        ...

    @property
    def channels(self) -> tuple:
        """Observed ``(port_name, mode_idx)`` pairs, in S-matrix order."""
        ...

    @property
    def excitations(self) -> tuple:
        """Excited ``(port_name, mode_idx)`` pairs — the S-matrix columns present."""
        ...

    @property
    def settings(self) -> RunSettings:
        """The :class:`RunSettings` this result was produced with."""
        ...

    def S(self, out_port, in_port, *, mode_out=0, mode_in=0, f_axis=None) -> np.ndarray:
        """One complex S-parameter over the frequency axis."""
        ...

    def db(
        self, out_port, in_port, *, mode_out=0, mode_in=0, floor_db=-200.0, f_axis=None
    ) -> np.ndarray:
        """One S-parameter in decibels, floored at *floor_db*."""
        ...

    def phase(
        self, out_port, in_port, *, mode_out=0, mode_in=0, deg=True, unwrap=True, f_axis=None
    ) -> np.ndarray:
        """Phase of one S-parameter over the frequency axis."""
        ...

    def a(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Incident power-wave time series ``a(t)`` at one channel."""
        ...

    def b(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Outgoing power-wave time series ``b(t)`` at one channel."""
        ...


__all__ = ["RunSettings", "ScatteringResult", "ScatteringResultMixin"]
