"""Field series — frames of a field on one grid, over time or over frequency.

:class:`FieldRecording` holds the frames of a time-domain field, each
a :class:`~magnelio.fields.FieldState` on the same grid, with the time
base of the electric field and the half-step-shifted one of the
magnetic field; :class:`FieldSpectrum` holds complex frames, one per
frequency.  Both keep the Yee staggering of every sample — nothing is
averaged until it is asked for — so a frame can be read at its own
positions, put back into a run as an initial field, sliced, plotted,
or shown in 3D.

The recorded components may be a subset of the six; a frame fills the
others with zeros and :attr:`components` says which were recorded.
"""

# Design: DD-259 step 1 (containers behind the raw-recording monitors).

from __future__ import annotations

from typing import Any

import numpy as np

from magnelio._fields.field_arrays import FieldArrays
from magnelio.fields.state import FieldState, _yee_shapes
from magnelio.mesh.grid import GridLines

__all__ = ["FieldRecording", "FieldSpectrum"]

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


class _FieldSeries:
    """Frames of a field on one grid, stacked along a leading axis."""

    _kind = "field"
    _label_unit = ""

    def _init(self, grid: GridLines, labels, components: dict, *, complex_only: bool) -> None:
        if not isinstance(grid, GridLines):
            raise TypeError(f"grid must be a magnelio.mesh.GridLines; got {type(grid).__name__}")
        labels = np.atleast_1d(np.asarray(labels, dtype=float))
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("a series needs at least one frame")
        unknown = sorted(set(components) - set(_COMPONENTS))
        if unknown:
            raise KeyError(f"unknown component(s) {unknown}; expected a subset of {_COMPONENTS}")
        if not components:
            raise ValueError("a series needs at least one recorded component")
        shapes = _yee_shapes(grid.Nx, grid.Ny, grid.Nz)
        lengths = FieldState.zeros(grid)._lengths()
        raw = {}
        for name, value in components.items():
            a = np.asarray(value)
            expected = (labels.size, *shapes[name])
            if a.shape != expected:
                raise ValueError(
                    f"{name} must have shape (n_frames, *Yee shape) = {expected}; got {a.shape}"
                )
            if complex_only:
                a = a.astype(complex)
            raw[name] = a * lengths[name][None]
        self._grid = grid
        self._labels = labels
        self._raw = raw
        self._dual = None
        self._frame_cache: tuple[int, FieldArrays] | None = None

    @classmethod
    def _from_raw(cls, grid: GridLines, labels, raw: dict[str, np.ndarray], dual=None, **kwargs):
        """Wrap stacked grid quantities without conversion (internal).

        *dual*: the dual widths of the ``h`` samples when the grid is a
        region cut from a larger one (see :meth:`FieldState._from_raw`).
        """
        self = cls.__new__(cls)
        self._grid = grid
        self._labels = np.atleast_1d(np.asarray(labels, dtype=float))
        self._raw = {name: np.asarray(a) for name, a in raw.items()}
        self._dual = dual
        self._frame_cache = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def _lengths(self) -> dict[str, np.ndarray]:
        probe = FieldState.zeros(self._grid)
        probe._dual = self._dual
        return probe._lengths()

    # ── vocabulary ───────────────────────────────────────────────────────

    @property
    def grid(self) -> GridLines:
        """The grid lines every frame refers to."""
        return self._grid

    @property
    def n_frames(self) -> int:
        return int(self._labels.size)

    def __len__(self) -> int:
        return self.n_frames

    @property
    def components(self) -> tuple[str, ...]:
        """The recorded components, in Yee order."""
        return tuple(c for c in _COMPONENTS if c in self._raw)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Cells of the grid, ``(Nx, Ny, Nz)``."""
        g = self._grid
        return (g.Nx, g.Ny, g.Nz)

    @property
    def is_complex(self) -> bool:
        return any(np.iscomplexobj(a) for a in self._raw.values())

    def positions(self, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The 1-D coordinate vectors ``(x, y, z)`` of a component's samples."""
        return FieldState.zeros(self._grid).positions(component)

    @property
    def cell_centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The 1-D cell-centre coordinates ``(xc, yc, zc)`` of :meth:`cell_centred`."""
        return FieldState.zeros(self._grid).cell_centres

    # ── frames ───────────────────────────────────────────────────────────

    def _check_index(self, i: int) -> int:
        i = int(i)
        if not (-self.n_frames <= i < self.n_frames):
            raise IndexError(f"frame {i} out of range for {self.n_frames} frames")
        return i % self.n_frames

    def _nearest(self, value: float) -> int:
        return int(np.argmin(np.abs(self._labels - float(value))))

    def _frame_arrays(self, i: int) -> FieldArrays:
        """The grid quantities of frame *i*, zeros for unrecorded components."""
        i = self._check_index(i)
        if self._frame_cache is not None and self._frame_cache[0] == i:
            return self._frame_cache[1]
        shapes = _yee_shapes(*self.shape)
        dtype = complex if self.is_complex else float
        arrays = {
            c: (self._raw[c][i] if c in self._raw else np.zeros(shapes[c], dtype=dtype))
            for c in _COMPONENTS
        }
        fa = FieldArrays(**arrays)
        self._frame_cache = (i, fa)
        return fa

    def frame(self, i: int) -> FieldState:
        """Frame *i* as a :class:`~magnelio.fields.FieldState`.

        Components that were not recorded are zero in the frame; see
        :attr:`components`.
        """
        return FieldState._from_raw(self._grid, self._frame_arrays(i), dual=self._dual)

    def component(self, name: str) -> np.ndarray:
        """The physical samples of one recorded component, ``(n_frames, *Yee shape)``."""
        if name not in self._raw:
            raise KeyError(
                f"component {name!r} was not recorded; recorded: {list(self.components)}"
            )
        return np.asarray(self._raw[name]) / self._lengths()[name][None]

    def cell_centred(self, components=None, corners=None, frame: int | None = None) -> dict:
        """Components averaged onto the cell centres, per frame.

        Parameters
        ----------
        components : sequence of str, optional
            Subset of the six names; default the recorded ones.
        corners : tuple of tuple, optional
            Two opposite corners [m] of a sub-box; default the whole grid.
        frame : int, optional
            One frame; default all, stacked along a leading axis.

        Returns
        -------
        dict[str, np.ndarray]
            ``{name: array}`` shaped ``(nx, ny, nz)`` for one frame,
            ``(n_frames, nx, ny, nz)`` for all.
        """
        names = list(self.components if components is None else components)
        if frame is not None:
            return self.frame(frame).cell_centred(names, corners)
        per_frame = [self.frame(i).cell_centred(names, corners) for i in range(self.n_frames)]
        return {c: np.stack([d[c] for d in per_frame], axis=0) for c in names}

    # ── pictures ─────────────────────────────────────────────────────────

    def _label_text(self, i: int) -> str:
        return ""

    def _snapshot(self, i: int, phase: float | None) -> FieldState:
        fs = self.frame(i)
        if phase is not None and fs.is_complex:
            fs = fs.scaled(np.exp(1j * np.deg2rad(float(phase)))).real()
        return fs

    def _plot(self, i: int, phase: float | None, component: str, kwargs: dict):
        fig, ax = self._snapshot(i, phase).plot(component, **kwargs)
        label = self._label_text(i)
        if label and kwargs.get("title") is None:
            ax.set_title(f"{ax.get_title()}, {label}")
        return fig, ax

    def show(self, component: str = "E", **kwargs):
        """Interactive 3D view of the series on a cutting plane.

        The geometry viewer with the field laid on its cut and a frame
        slider over the series; see :func:`magnelio.plots.show_field`
        for the arguments.
        """
        from magnelio.post.field_3d import show_field  # noqa: PLC0415

        return show_field(self, component, **kwargs)

    def _repr_body(self) -> str:
        g = self._grid
        comps = ",".join(self.components)
        kind = "complex" if self.is_complex else "real"
        return f"{self.n_frames} frames, grid {g.Nx}x{g.Ny}x{g.Nz} cells, {comps}, {kind}"


class FieldRecording(_FieldSeries):
    """Frames of a time-domain field on one grid.

    Parameters
    ----------
    grid : GridLines
        The grid the samples live on (a monitor's region as its own grid).
    times : array_like
        Instants [s] of the electric-field frames, ascending.
    dt : float, optional
        The time step [s].  The magnetic field of a leapfrog march is
        sampled half a step *after* the electric field; :attr:`times_h`
        states those instants.  ``None`` when the recording did not
        come from a march.
    **components : array_like
        Physical field arrays, E in V/m and H in A/m, each shaped
        ``(n_frames, *Yee shape)`` — a subset of ``Ex Ey Ez Hx Hy Hz``.
    """

    _kind = "time"

    def __init__(self, grid: GridLines, times, *, dt: float | None = None, **components) -> None:
        self._init(grid, times, components, complex_only=False)
        if np.any(np.diff(self._labels) <= 0.0):
            raise ValueError("times must be strictly increasing")
        self.dt = None if dt is None else float(dt)

    @property
    def times(self) -> np.ndarray:
        """Instants [s] of the electric-field frames."""
        return self._labels

    @property
    def times_h(self) -> np.ndarray:
        """Instants [s] of the magnetic-field frames: ``times + dt/2``."""
        return self._labels + (0.5 * self.dt if self.dt else 0.0)

    def index_of(self, t: float) -> int:
        """Index of the frame nearest to *t* [s]."""
        return self._nearest(t)

    def at_time(self, t: float) -> FieldState:
        """The frame nearest to *t* [s]."""
        return self.frame(self._nearest(t))

    def _label_text(self, i: int) -> str:
        return f"t = {self._labels[i] * 1e9:.4g} ns"

    def plot(
        self, component: str = "E", *, t: float | None = None, frame: int | None = None, **kwargs
    ):
        """Plot one frame on a slice plane.

        Parameters
        ----------
        component : str
            ``"E"``/``"H"`` for a vector plot or magnitude, ``"Ez"``, …
            for one component.
        t : float, optional
            Instant [s]; the nearest frame is drawn.
        frame : int, optional
            Frame index (default 0; exclusive with *t*).
        **kwargs
            Passed to :meth:`magnelio.fields.FieldState.plot` — the
            plane (``normal``, ``position``), ``plot_type``, ``ax``,
            ``geometry`` and the rest.

        Returns
        -------
        fig, ax
        """
        if t is not None and frame is not None:
            raise ValueError("give t or frame, not both")
        i = self._nearest(t) if t is not None else (0 if frame is None else frame)
        return self._plot(self._check_index(i), None, component, kwargs)

    def __repr__(self) -> str:
        t = self._labels
        span = (
            f"{t[0] * 1e9:.4g} – {t[-1] * 1e9:.4g} ns" if t.size > 1 else f"t = {t[0] * 1e9:.4g} ns"
        )
        return f"FieldRecording({self._repr_body()}, {span})"


class FieldSpectrum(_FieldSeries):
    """Complex frames of a field on one grid, one per frequency.

    Parameters
    ----------
    grid : GridLines
    frequencies : array_like
        Frequencies [Hz] of the frames.
    **components : array_like
        Complex field arrays, E in V/m and H in A/m (per √W for a
        monitor's pattern), each shaped ``(n_frames, *Yee shape)``.
    """

    _kind = "frequency"

    def __init__(self, grid: GridLines, frequencies, **components) -> None:
        self._init(grid, frequencies, components, complex_only=True)

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies [Hz] of the frames."""
        return self._labels

    @property
    def f(self) -> np.ndarray:
        """Alias of :attr:`frequencies`."""
        return self._labels

    @property
    def is_complex(self) -> bool:
        return True

    def index_of(self, f: float) -> int:
        """Index of the frame nearest to *f* [Hz]."""
        return self._nearest(f)

    def at_frequency(self, f: float) -> FieldState:
        """The (complex) frame nearest to *f* [Hz]."""
        return self.frame(self._nearest(f))

    def snapshot(self, f: float | None = None, *, frame: int | None = None, phase: float = 0.0):
        """The real field ``Re(F · exp(j·phase))`` of one frame, *phase* in degrees."""
        i = self._nearest(f) if f is not None else (0 if frame is None else frame)
        return self._snapshot(self._check_index(i), phase)

    def _label_text(self, i: int) -> str:
        return f"f = {self._labels[i] * 1e-9:.4g} GHz"

    def plot(
        self,
        component: str = "E",
        *,
        f: float | None = None,
        frame: int | None = None,
        phase: float | None = None,
        **kwargs,
    ):
        """Plot one frame on a slice plane.

        Parameters
        ----------
        component : str
            ``"E"``/``"H"`` for a vector plot or magnitude, ``"Ez"``, …
            for one component.
        f : float, optional
            Frequency [Hz]; the nearest frame is drawn.
        frame : int, optional
            Frame index (default 0; exclusive with *f*).
        phase : float, optional
            Instant of the complex pattern in degrees,
            ``Re(F · exp(j·phase))``.  Default: the instant of maximum
            energy on the slice (see :meth:`FieldState.plot`).
        **kwargs
            Passed to :meth:`magnelio.fields.FieldState.plot`.

        Returns
        -------
        fig, ax
        """
        if f is not None and frame is not None:
            raise ValueError("give f or frame, not both")
        i = self._nearest(f) if f is not None else (0 if frame is None else frame)
        return self._plot(self._check_index(i), phase, component, kwargs)

    def __repr__(self) -> str:
        fr = self._labels
        span = (
            f"{fr[0] * 1e-9:.4g} – {fr[-1] * 1e-9:.4g} GHz"
            if fr.size > 1
            else f"f = {fr[0] * 1e-9:.4g} GHz"
        )
        return f"FieldSpectrum({self._repr_body()}, {span})"


def _series_kind(series: Any) -> str:
    return getattr(series, "_kind", "field")
