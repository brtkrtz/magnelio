"""
Matplotlib plotting helpers for field monitors (0D / 1D line plots).

2D scalar, contour, and vector plots are handled by the unified
:mod:`magnelio.post.plot_field` module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


def plot_time_0d(
    t: np.ndarray,
    values: np.ndarray,
    component: str,
    name: str,
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Line plot for a 0D time-domain monitor."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(t * 1e9, values, label=component)
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(component)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_time_1d(
    coord: np.ndarray,
    values: np.ndarray,
    component: str,
    axis_label: str,
    name: str,
    ax: "matplotlib.axes.Axes | None" = None,
    scale_mm: bool = True,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Line plot for a 1D time-domain monitor at a specific time step."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    scale = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"

    ax.plot(coord * scale, values, label=component)
    ax.set_xlabel(f"{axis_label} [{unit}]")
    ax.set_ylabel(component)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_freq_0d(
    freqs: np.ndarray,
    values: np.ndarray,
    component: str,
    name: str,
    what: str = "abs",
    ax: "matplotlib.axes.Axes | None" = None,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Line plot for a 0D frequency-domain monitor."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if what == "abs":
        y = np.abs(values)
        ylabel = f"|{component}|"
    elif what == "phase":
        y = np.angle(values, deg=True)
        ylabel = f"Phase({component}) [deg]"
    elif what == "real":
        y = values.real
        ylabel = f"Re({component})"
    elif what == "imag":
        y = values.imag
        ylabel = f"Im({component})"
    else:
        raise ValueError(f"Unknown what={what!r}. Use 'abs', 'phase', 'real', 'imag'.")

    ax.plot(freqs * 1e-9, y, label=ylabel)
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax
