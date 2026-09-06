"""
MonitorFieldTime — records field snapshots at specified time points.

Supports 0D (point), 1D (line), 2D (plane), and 3D (volume) regions:
the region is a box given by two opposite corners, and an axis whose
two corner values coincide degenerates to a single cell layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from magnelio.monitors.base import (
    MonitorRegion,
    _expand_field_list,
    _region_dual,
    _region_slices,
    _take_raw,
    region_grid,
    resolve_mirrors,
    resolve_region,
)


@dataclass
class MonitorFieldTime:
    """Record field snapshots at specified time points.

    Give either an explicit list of *times* or a recording *interval*.
    The interval form is the one to use when the run length is decided
    by a stop criterion rather than by you: it keeps sampling for as
    long as the simulation lasts, with no end time to guess.

    Parameters
    ----------
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        recorded box [m] — the same form
        as :meth:`Brick.from_corners`.  Corner order does not matter.
        An axis whose two values coincide is degenerate and records a
        single cell layer: that is how a plane, a line or a point is
        expressed.  A component may be ``None`` (or ``±math.inf``) to
        reach the domain boundary on that side.  Omit *corners*
        entirely for the whole domain.
    times : array_like, optional
        Explicit recording time points [s].  Mutually exclusive with
        *interval*.
    interval : float, optional
        Record every *interval* seconds until the run ends.  Mutually
        exclusive with *times*.  Note that an open-ended monitor on a
        long run accumulates snapshots: give the analysis a
        ``project=`` so they stream to disk instead of filling RAM.
    start : float, default 0.0
        First recording time [s] of the *interval* form.
    fields : list[str]
        Field groups or components to record.  ``"E"`` expands to
        ``["Ex", "Ey", "Ez"]``, ``"H"`` to ``["Hx", "Hy", "Hz"]``.
    name : str
        Monitor label (must be unique within a simulation).

    Examples
    --------
    A plane at z = 5 mm spanning the whole cross-section, at a fixed
    set of instants:

    >>> mon = MonitorFieldTime(
    ...     corners=((None, None, 5e-3), (None, None, 5e-3)),
    ...     times=np.arange(0, 10e-9, 0.5e-9),
    ...     fields=["E"],
    ...     name="E_xy_plane",
    ... )

    A box, sampled every 0.5 ns however long the run turns out to be:

    >>> mon = MonitorFieldTime(
    ...     corners=((0, 0, -20e-3), (5e-3, 5e-3, 20e-3)),
    ...     interval=0.5e-9,
    ...     fields=["E"],
    ...     name="E_box",
    ... )

    The whole domain, same cadence:

    >>> mon = MonitorFieldTime(interval=0.5e-9, fields=["E"])
    """

    corners: object = None
    times: np.ndarray | None = None
    interval: float | None = None
    start: float = 0.0
    fields: list[str] = field(default_factory=lambda: ["E"])
    name: str = ""

    # --- internal state (set by attach / record) ---
    _region: MonitorRegion | None = field(default=None, repr=False, init=False)
    _components: list[str] = field(default_factory=list, repr=False, init=False)
    _snapshots: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False, init=False)
    _recorded_times: list[float] = field(default_factory=list, repr=False, init=False)
    _next_idx: int = field(default=0, repr=False, init=False)
    # Frames recorded so far, streamed ones included (the store's
    # truncation point on resume; ``_next_idx`` counts consumed targets).
    _n_recorded: int = field(default=0, repr=False, init=False)
    _dt: float = field(default=0.0, repr=False, init=False)
    # Symmetry planes the region touches (DD-154) — plots mirror the
    # recorded half across them on read.
    _mirrors: tuple = field(default=(), repr=False, init=False)
    # The region as its own grid, the dual widths of its h samples and
    # the per-component slices of the solver arrays (DD-259): a snapshot
    # is the grid quantities on the Yee positions, nothing averaged.
    _subgrid: object = field(default=None, repr=False, init=False)
    _dual: tuple | None = field(default=None, repr=False, init=False)
    _slices: dict = field(default_factory=dict, repr=False, init=False)

    @classmethod
    def from_ranges(
        cls,
        *,
        x1=None,
        x2=None,
        dx=None,
        y1=None,
        y2=None,
        dy=None,
        z1=None,
        z2=None,
        dz=None,
        **kwargs,
    ):
        """Build the same monitor from one coordinate range per axis.

        The range spelling of ``corners=``, as in
        :meth:`~magnelio.geo.Brick.from_ranges`: each axis takes up to
        two of its three keywords — the two bounds (``x1``, ``x2``) or
        a bound and an extent (``x1``, ``dx`` / ``x2``, ``dx``).  Here
        an axis may also be open: give nothing for the whole domain
        extent, or a single bound to reach the domain boundary on the
        other side.  All remaining keyword arguments are forwarded to
        the constructor.

        Examples
        --------
        >>> mon = MonitorFieldTime.from_ranges(z1=5e-3, z2=5e-3, interval=0.5e-9, fields=["E"])
        """
        from magnelio.geo._ranges import corners_from_ranges  # noqa: PLC0415

        return cls(
            corners=corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz),
            **kwargs,
        )

    def __post_init__(self) -> None:
        if (self.times is None) == (self.interval is None):
            raise ValueError(
                "give either times= (explicit instants) or interval= "
                "(record every interval seconds until the run ends), "
                "not both and not neither",
            )
        if self.interval is not None:
            if not self.interval > 0.0:
                raise ValueError(
                    f"interval must be positive; got {self.interval}",
                )
            self.interval = float(self.interval)
            self.start = float(self.start)
        else:
            self.times = np.asarray(self.times, dtype=float)
            if self.times.ndim != 1 or len(self.times) == 0:
                raise ValueError("times must be a non-empty 1D array")
            self.times = np.sort(self.times)
        self._components = _expand_field_list(self.fields)
        if not self.name:
            self.name = f"field_time_{id(self):x}"

    def _target(self, k: int) -> float | None:
        """The k-th recording time, or ``None`` past the last one.

        The interval form never runs out — it is the caller's stop
        criterion that ends the run, not the monitor's schedule.
        """
        if self.interval is not None:
            return self.start + k * self.interval
        return float(self.times[k]) if k < len(self.times) else None

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        """Snap monitor region to the simulation grid.

        Called once by the solver during setup.
        """
        self._region = resolve_region(self.corners, mesh.grid)
        self._grid = mesh.grid  # per-edge lengths for the physical fields
        self._mirrors = resolve_mirrors(self._region, mesh)
        r = self._region
        self._subgrid = region_grid(mesh.grid, r)
        self._dual = _region_dual(mesh.grid, r.ix, r.iy, r.iz)
        self._slices = {c: _region_slices(r.ix, r.iy, r.iz, c) for c in self._components}
        self._ops = None
        self._snapshots = []
        self._recorded_times = []
        self._next_idx = 0
        self._n_recorded = 0

    def attach_operators(self, mesh, M_eps, M_mu) -> None:
        """Take the region's cut of the solver's material diagonals (DD-260).

        Called by the solver after :meth:`attach`, with its own ``M_ε``
        and ``M_μ`` diagonals; the recording then states its energy and
        flux (:meth:`~magnelio.fields.FieldRecording.energy`).
        """
        from magnelio.fields._operators import region_operators  # noqa: PLC0415

        if self._region is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")
        self._ops = region_operators(mesh, self._region, M_eps, M_mu)

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Record a snapshot when the electric field's instant meets a target.

        Called at every time step by the solver, after the H update of
        step *n*: ``e`` then stands at ``t + dt`` and ``h`` half a step
        later.  The snapshot is a copy of the grid quantities on the
        region's Yee positions — no averaging — stamped with the
        electric instant; the recording states the magnetic one as
        ``times + dt/2``.
        """
        if self._region is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")

        self._dt = dt
        t_e = t + dt

        # Fast path: schedule exhausted (explicit-times form only — the
        # interval form has no last target)
        t_target = self._target(self._next_idx)
        if t_target is None:
            return

        if t_e + 0.5 * dt < t_target:
            return  # not yet

        # One frame per step.  A schedule finer than the time step has
        # no more information to record, and two frames sharing an
        # instant would leave a spectrum dividing by a zero interval; the
        # further targets this step passes are consumed without a frame.
        self._snapshots.append(_take_raw(fields, self._components, self._slices))
        self._recorded_times.append(float(t_e))
        self._n_recorded += 1
        self._next_idx += 1
        t_target = self._target(self._next_idx)
        while t_target is not None and t_e + 0.5 * dt >= t_target:
            self._next_idx += 1
            t_target = self._target(self._next_idx)

    def finalize(self) -> None:
        """Called after the simulation completes."""
        pass  # nothing to do for time-domain monitors

    # ------------------------------------------------------------------
    # Streaming write-through (DD-070, WP-S9)
    # ------------------------------------------------------------------

    def pop_pending(self) -> tuple[list[float], dict[str, np.ndarray]]:
        """Drain the snapshots recorded since the last call (streaming).

        Returns the pending recorded times and, per component, the pending
        snapshots — grid quantities on the region's Yee positions, shape
        ``(k, *Yee shape)`` — stacked along a leading time axis, then
        clears the in-RAM snapshot buffer — so a project-backed run stays
        **memory-bounded** (the run sink flushes each batch to disk
        instead of the monitor holding every snapshot).  ``_next_idx`` (target-time progress) is
        *kept*, so recording continues at the right time point.  The in-RAM
        path never calls this, so its ``data``/``t`` accumulation is
        unchanged.
        """
        if not self._snapshots:
            return [], {}
        times = list(self._recorded_times)
        out = {}
        for comp in self._components:
            arrays = [s[comp] for s in self._snapshots if comp in s]
            if arrays:
                out[comp] = np.stack(arrays, axis=0)
        self._snapshots = []
        self._recorded_times = []
        return times, out

    def state_dict(self) -> dict:
        """Checkpoint the schedule cursor for a bit-exact resume.

        Two counters: the targets consumed (``next_idx``) and the frames
        recorded (``n_recorded``) — they differ when the schedule is
        finer than the time step.  The frames themselves live in the
        run's ``results.h5`` (streamed), and the region is re-resolved on
        attach; ``n_recorded`` drives the monitor-stream truncation on
        resume.
        """
        return {"next_idx": int(self._next_idx), "n_recorded": int(self._n_recorded)}

    def load_state_dict(self, sd: dict) -> None:
        """Restore the schedule cursor (see :meth:`state_dict`)."""
        self._next_idx = int(sd["next_idx"])
        self._n_recorded = int(sd.get("n_recorded", sd["next_idx"]))

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def t(self) -> np.ndarray:
        """Recorded instants [s] of the electric field."""
        return np.array(self._recorded_times)

    @property
    def recording(self):
        """The snapshots as a :class:`~magnelio.fields.FieldRecording`.

        Every frame keeps the Yee staggering of the region; the magnetic
        field's instants are ``recording.times_h``.  Raises when nothing
        has been recorded — on a project-backed run the frames stream to
        the store and are read back through the project's monitors.
        """
        from magnelio.fields.series import FieldRecording  # noqa: PLC0415

        if self._region is None or self._subgrid is None:
            raise RuntimeError(f"monitor {self.name!r} is not attached to a mesh")
        if not self._snapshots:
            raise RuntimeError(
                f"monitor {self.name!r} holds no snapshots — on a project run they stream "
                "to the store; open the project and use its monitors instead"
            )
        raw = {
            c: np.stack([snap[c] for snap in self._snapshots], axis=0)
            for c in self._components
            if all(c in snap for snap in self._snapshots)
        }
        return FieldRecording._from_raw(
            self._subgrid,
            np.asarray(self._recorded_times, dtype=float),
            raw,
            dual=self._dual,
            ops=self._ops,
            dt=float(self._dt) if self._dt else None,
        )

    # ------------------------------------------------------------------
    # Pictures — drawn from the recording (DD-259)
    # ------------------------------------------------------------------

    def _view(self):
        from magnelio.monitors._frame_plots import SeriesView  # noqa: PLC0415

        return SeriesView(
            self.recording,
            name=self.name,
            mirrors=self._mirrors,
            grid=getattr(self, "_grid", None),
        )

    def plot(
        self,
        component: str = "E",
        t: float | None = None,
        t_index: int | None = None,
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        ax=None,
        scale_mm: bool = True,
        cmap: str | None = None,
        geometry=None,
        flip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        density: int = 20,
        normalize_arrows: bool = False,
        threshold: float = 0.02,
        quiver_scale: float | None = None,
        colorbar: bool = True,
    ):
        """Plot one frame of the recording.

        A point region draws its value over time (ignores *t* /
        *t_index*); a line region the values along its axis at one
        instant; a plane region the frame on its own plane; a volume
        the plane selected with *normal* and *position*.  Only the
        drawn layer is averaged onto cell centres.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for a vector plot or magnitude,
            ``"Ex"``, ``"Hy"``, … for a single component.
        t : float, optional
            Instant [s]; the nearest frame is drawn.
        t_index : int, optional
            Frame index (overrides *t*).
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for a volume (required there); for a
            plane region it may name the plane's own normal.
        position : float
            Slice-plane position along *normal* [m]; snapped to the
            nearest cell-centre plane.
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        ax : matplotlib.axes.Axes, optional
        scale_mm : bool
        cmap : str or None
            Colourmap (None = auto-select).
        geometry : GeometryModel, optional
            Cross-section overlay of the plane.
        flip : bool
            Swap horizontal and vertical axes.
        vmin, vmax : float, optional
            Colour limits (scalar) or arrow clipping (vector).
        density : int
            Target arrows per axis (vector mode).
        normalize_arrows : bool
            Unit-length arrows, colour = magnitude.
        threshold : float
            Suppress arrows below this fraction of peak.
        quiver_scale : float or None
            Fixed quiver scale override.
        colorbar : bool, default True
            Draw the colour bar; ``False`` for a panel of a shared figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from magnelio.monitors._frame_plots import plot_frame  # noqa: PLC0415

        view = self._view()
        return plot_frame(
            view,
            component,
            view.index_of(t, t_index),
            normal=normal,
            position=position,
            plot_type=plot_type,
            ax=ax,
            scale_mm=scale_mm,
            cmap=cmap,
            geometry=geometry,
            flip=flip,
            vmin=vmin,
            vmax=vmax,
            density=density,
            normalize_arrows=normalize_arrows,
            threshold=threshold,
            quiver_scale=quiver_scale,
            colorbar=colorbar,
        )

    def interact(
        self,
        component: str = "E",
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        scale_mm: bool = True,
        cmap: str | None = None,
        geometry=None,
        flip: bool = False,
        density: int = 20,
        threshold: float = 0.02,
        vmax: float | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """A notebook slider over the recorded frames (needs ipywidgets).

        The colour range (scalar) or arrow scale (vector) is fixed over
        every frame for visual stability.  The arguments are those of
        :meth:`plot`; a volume needs its slice plane (*normal*,
        *position*), and the slider then runs over time on it.
        """
        from magnelio.monitors._frame_plots import interact as _interact  # noqa: PLC0415

        return _interact(
            self._view(),
            component,
            normal=normal,
            position=position,
            plot_type=plot_type,
            scale_mm=scale_mm,
            cmap=cmap,
            geometry=geometry,
            flip=flip,
            density=density,
            threshold=threshold,
            vmax=vmax,
            figsize=figsize,
        )

    def show(self, component: str = "E", **kwargs):
        """Interactive 3D view of the recorded field on a cutting plane.

        The geometry viewer with the field laid on its cut, a time slider
        over the recorded frames, and the position slider walking through
        the region.  See :func:`magnelio.plots.show_field` for the
        arguments — the frame (``t=`` or ``frame=``), the plane
        (``normal``, ``position``), ``geometry`` and ``mesh`` overlays,
        and the rendering ``mode``.
        """
        from magnelio.post.field_3d import show_field  # noqa: PLC0415

        return show_field(self, component, **kwargs)

    def __repr__(self) -> str:
        shape = "unattached"
        if self._region is not None:
            r = self._region
            shape = f"{r.ix.stop - r.ix.start}x{r.iy.stop - r.iy.start}x{r.iz.stop - r.iz.start}"
        schedule = (
            f"every {self.interval:g} s from {self.start:g} s"
            if self.interval is not None
            else f"n_times={len(self.times)}"
        )
        return (
            f"MonitorFieldTime(name={self.name!r}, "
            f"fields={self.fields}, "
            f"{schedule}, "
            f"region={shape})"
        )
