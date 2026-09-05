"""The stored-energy trace as a picture: dB below the peak over time.

The number a running simulation is watched by is the stored energy in
dB below its peak — the figure on the progress line, the figure the
energy criterion fires on.  The plot shows the same figure over the
whole run, so the picture and the line agree: the rising edge sits
below zero too, the curve touches 0 dB once at the peak, and the stop
criterion is a horizontal line the curve has to cross.
"""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np

__all__ = ["plot_energy_traces"]


def plot_energy_traces(
    traces: Mapping[str, np.ndarray],
    *,
    energy_stop_db: float | None = None,
    x: str = "time",
    ax=None,
    title: str | None = None,
):
    """Plot stored-energy traces in dB below their peak.

    Parameters
    ----------
    traces : mapping of label to structured array
        Each value an energy trace with ``step``, ``time`` and
        ``energy`` fields, as :attr:`~magnelio.analysis.TDResult.energy_trace`
        holds it; the key is the legend label.  Empty traces are
        skipped with a warning.
    energy_stop_db : float, optional
        The energy criterion of the run [dB below peak]; drawn as a
        dashed line when given.
    x : {"time", "step"}, default "time"
        Time in nanoseconds, or the leapfrog step.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created otherwise.
    title : str, optional
        Axes title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if x not in ("time", "step"):
        raise ValueError(f"x must be 'time' or 'step'; got {x!r}")
    if ax is None:
        # pyplot only when a figure has to be made: a caller drawing
        # into its own axes — the live monitor renders off the main
        # thread — must not have a GUI backend started on its behalf.
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tiny = np.finfo(float).tiny
    drawn = 0
    for label, trace in traces.items():
        trace = np.asarray(trace)
        if trace.size == 0:
            warnings.warn(f"energy trace {label!r} is empty; skipped", UserWarning, stacklevel=2)
            continue
        energy = np.asarray(trace["energy"], dtype=float)
        peak = float(energy.max())
        if not peak > 0.0:
            warnings.warn(
                f"energy trace {label!r} never rose above zero; skipped", UserWarning, stacklevel=2
            )
            continue
        level = 10.0 * np.log10(np.maximum(energy, tiny) / peak)
        xs = np.asarray(trace["time"], dtype=float) * 1e9 if x == "time" else trace["step"]
        ax.plot(xs, level, label=str(label))
        drawn += 1
    if energy_stop_db is not None:
        ax.axhline(
            -float(energy_stop_db),
            linestyle="--",
            color="0.4",
            linewidth=1.0,
            label=f"stop criterion: −{float(energy_stop_db):.0f} dB",
        )
    ax.set_xlabel("t / ns" if x == "time" else "time step")
    ax.set_ylabel("stored energy / dB below peak")
    ax.grid(True, alpha=0.3)
    if drawn > 1 or energy_stop_db is not None:
        ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax
