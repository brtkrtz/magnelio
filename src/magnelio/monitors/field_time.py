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
    _AXES,
    MonitorRegion,
    PlaneView,
    _expand_field_list,
    _interp_to_cell_centres,
    _resolve_component,
    component_mirror_key,
    mirror_extend,
    mirror_plane_arrays,
    mirror_sign,
    plane_slab_halfwidth,
    resolve_mirrors,
    resolve_plane_view,
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
    _dt: float = field(default=0.0, repr=False, init=False)
    # Symmetry planes the region touches (DD-154) — plots mirror the
    # recorded half across them on read.
    _mirrors: tuple = field(default=(), repr=False, init=False)

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
        self._snapshots = []
        self._recorded_times = []
        self._next_idx = 0

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Record a snapshot if *t* matches a requested time point.

        Called at every time step by the solver.
        """
        if self._region is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")

        self._dt = dt

        # Fast path: schedule exhausted (explicit-times form only — the
        # interval form has no last target)
        t_target = self._target(self._next_idx)
        if t_target is None:
            return

        # Check if current time matches the next requested recording time
        if t + 0.5 * dt < t_target:
            return  # not yet

        # Record — may need to catch up if dt is large
        while t_target is not None and t + 0.5 * dt >= t_target:
            r = self._region
            snap = _interp_to_cell_centres(
                fields,
                self._components,
                r.ix,
                r.iy,
                r.iz,
                self._grid,
            )
            # Squeeze singleton dimensions for lower-dimensional monitors
            snap = {k: np.squeeze(v) for k, v in snap.items()}
            self._snapshots.append(snap)
            self._recorded_times.append(float(t))
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
        snapshots stacked along a leading time axis, then clears the in-RAM
        snapshot buffer — so a project-backed run stays **memory-bounded**
        (the run sink flushes each batch to disk instead of the monitor
        holding every snapshot).  ``_next_idx`` (target-time progress) is
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
        """Checkpoint the target-time cursor for a bit-exact resume.

        Only ``_next_idx`` is state a continuation must restore — the
        recorded snapshots themselves live in the run's ``results.h5``
        (streamed), and the region is re-resolved on attach.  ``_next_idx``
        equals the number of snapshots recorded so far, so it also drives
        the monitor-stream truncation on resume.
        """
        return {"next_idx": int(self._next_idx)}

    def load_state_dict(self, sd: dict) -> None:
        """Restore the target-time cursor (see :meth:`state_dict`)."""
        self._next_idx = int(sd["next_idx"])

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def t(self) -> np.ndarray:
        """Actually recorded time points [s]."""
        return np.array(self._recorded_times)

    @property
    def data(self) -> dict[str, np.ndarray]:
        """All snapshots stacked along a leading time axis.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are component names (e.g. ``"Ex"``).  Values have shape
            ``(n_times, <spatial dims>)``.  For a 0D monitor the spatial
            dims are empty, giving shape ``(n_times,)``.
        """
        if not self._snapshots:
            return {}
        out = {}
        for comp in self._components:
            arrays = [s[comp] for s in self._snapshots if comp in s]
            if arrays:
                out[comp] = np.stack(arrays, axis=0)
        return out

    def component(self, name: str) -> np.ndarray:
        """Return recorded data for a single component.

        Parameters
        ----------
        name : str
            Component name, e.g. ``"Ez"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_times, <spatial dims>)``.
        """
        d = self.data
        if name not in d:
            raise KeyError(f"Component '{name}' not recorded. Available: {list(d.keys())}")
        return d[name]

    @property
    def region(self) -> MonitorRegion | None:
        """Resolved grid region (available after :meth:`attach`)."""
        return self._region

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _make_overlay(self, geometry, pv: PlaneView):
        """Build CrossSectionOverlay for the resolved plotting plane."""
        if geometry is None:
            return None
        from magnelio.post.plot_field import (  # noqa: PLC0415
            CrossSectionOverlay,
        )

        (i0, _), (i1, _) = pv.free
        return CrossSectionOverlay(
            geometry=geometry,
            normal=_AXES[pv.normal_idx],
            position=pv.normal_pos,
            mirrors=tuple(
                (0 if m.axis == i0 else 1, m.wall, m.at_low)
                for m in self._mirrors
                if m.axis in (i0, i1)
            ),
            slab=plane_slab_halfwidth(getattr(self, "_grid", None), pv.normal_idx, pv.normal_pos),
        )

    @staticmethod
    def _slice_note(pv: PlaneView, scale_mm: bool) -> str:
        """Title suffix naming the slice plane (3D monitors only)."""
        if pv.slice_index is None:
            return ""
        if scale_mm:
            return f", {_AXES[pv.normal_idx]}={pv.normal_pos * 1e3:.3g} mm"
        return f", {_AXES[pv.normal_idx]}={pv.normal_pos:.3g} m"

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

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
        **kwargs,
    ):
        """Plot recorded field data.

        For 0D monitors: line plot over time (ignores *t* / *t_index*).
        For 1D monitors: line plot at a specific time.
        For 2D monitors: colour-map, contour, or quiver plot at a time.
        For 3D monitors: the same plane plots on a slice selected with
        *normal* and *position*.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for vector magnitude or vector plot.
            ``"Ex"``, ``"Hy"``, … for a single component (scalar only).
        t : float, optional
            Time point [s].  Nearest recorded time is used.
        t_index : int, optional
            Time index (overrides *t*).
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for 3D monitors (required there).  For a
            2D monitor it may be given for validation but is redundant.
        position : float
            Slice-plane position along *normal* [m]; snapped to the
            nearest cell-centre plane (3D monitors only).
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        ax : matplotlib.axes.Axes, optional
        scale_mm : bool
        cmap : str or None
            Colourmap (None = auto-select).
        geometry : list, optional
            Geometry objects for cross-section overlay (2D only).
        flip : bool
            Swap horizontal and vertical axes (2D only).
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

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        r = self._region
        if r is None:
            raise RuntimeError("Monitor not attached / no data.")

        data = self.data
        ndim = r.ndim
        is_field_group = component in ("E", "H")

        # --- 0D: line plot over time ---
        if ndim == 0:
            from magnelio.monitors.plotting import plot_time_0d  # noqa: PLC0415

            arr = _resolve_component(data, component)
            return plot_time_0d(self.t, arr, component, self.name, ax=ax)

        # Resolve time index
        if t_index is None:
            if t is not None:
                t_index = int(np.argmin(np.abs(self.t - t)))
            else:
                t_index = 0

        # --- 1D: line plot ---
        if ndim == 1:
            from magnelio.monitors.plotting import plot_time_1d  # noqa: PLC0415

            arr = _resolve_component(data, component)
            vals = arr[t_index]
            title = f"{self.name} — {component}, t={self.t[t_index]:.3e} s"
            axes_labels = ["x", "y", "z"]
            coords = [r.xc, r.yc, r.zc]
            for i, c in enumerate(coords):
                if len(c) > 1:
                    fld, comp_axis = component_mirror_key(component)
                    for spec in self._mirrors:
                        if spec.axis != i:
                            continue
                        c, vals = mirror_extend(
                            c,
                            vals,
                            spec,
                            0,
                            mirror_sign(fld, comp_axis, spec.axis, spec.kind),
                        )
                    return plot_time_1d(
                        c,
                        vals,
                        component,
                        axes_labels[i],
                        title,
                        ax=ax,
                        scale_mm=scale_mm,
                    )

        # --- 2D / 3D: plane plot (3D via normal/position slice) ---
        pv = resolve_plane_view(r, normal, position)
        (i0, c0), (i1, c1) = pv.free
        note = self._slice_note(pv, scale_mm)

        from magnelio.post.plot_field import (  # noqa: PLC0415
            plot_field_scalar,
            plot_field_vector,
        )

        overlay = self._make_overlay(geometry, pv)

        if plot_type == "vector":
            # component must be field group
            field_group = component if is_field_group else "E"
            comp_u = f"{field_group}{_AXES[i0]}"
            comp_v = f"{field_group}{_AXES[i1]}"
            comp_w = f"{field_group}{_AXES[pv.normal_idx]}"
            if comp_u not in data or comp_v not in data:
                raise KeyError(
                    f"Need both {comp_u} and {comp_v} recorded.  Available: {list(data.keys())}"
                )
            u_arr = pv.take2d(data[comp_u][t_index])
            v_arr = pv.take2d(data[comp_v][t_index])
            w_arr = pv.take2d(data[comp_w][t_index]) if comp_w in data else None
            c0, c1, (u_arr, v_arr, w_arr) = mirror_plane_arrays(
                pv,
                self._mirrors,
                c0,
                c1,
                [
                    (u_arr, field_group, i0),
                    (v_arr, field_group, i1),
                    (w_arr, field_group, pv.normal_idx),
                ],
            )
            title = f"{self.name} — {field_group}-field, t={self.t[t_index]:.3e} s{note}"
            return plot_field_vector(
                c0,
                c1,
                u_arr,
                v_arr,
                w=w_arr,
                xlabel=_AXES[i0],
                ylabel=_AXES[i1],
                wlabel=_AXES[pv.normal_idx],
                title=title,
                ax=ax,
                scale_mm=scale_mm,
                cmap=cmap or "viridis",
                density=density,
                normalize_arrows=normalize_arrows,
                vmax=vmax,
                threshold=threshold,
                quiver_scale=quiver_scale,
                flip=flip,
                geometry=overlay,
            )

        # Scalar plot (color / contour)
        arr = _resolve_component(data, component)
        is_amplitude = is_field_group
        vals = pv.take2d(arr[t_index])
        fld, comp_axis = component_mirror_key(component)
        c0, c1, (vals,) = mirror_plane_arrays(
            pv,
            self._mirrors,
            c0,
            c1,
            [(vals, fld, comp_axis)],
        )
        title = f"{self.name} — {component}, t={self.t[t_index]:.3e} s{note}"

        if is_amplitude:
            effective_cmap = cmap or "viridis"
            sym = False
            if vmin is None:
                vmin = 0.0
        else:
            effective_cmap = cmap or "RdBu_r"
            sym = True

        return plot_field_scalar(
            c0,
            c1,
            vals,
            xlabel=_AXES[i0],
            ylabel=_AXES[i1],
            title=title,
            clabel=component,
            ax=ax,
            scale_mm=scale_mm,
            cmap=effective_cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric=sym,
            plot_type=plot_type,
            flip=flip,
            geometry=overlay,
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
        """Interactive time-step slider for field snapshots (Jupyter notebook).

        Requires ``ipywidgets``.  The colour range (scalar) or arrow
        scale (vector) is fixed across all time steps for visual stability.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for vector / amplitude plots.
            ``"Ex"``, ``"Ez"``, … for individual component scalar plots.
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for 3D monitors (required there); the
            slider then runs over time at a fixed slice plane.
        position : float
            Slice-plane position along *normal* [m] (3D monitors only).
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        scale_mm : bool
            Use millimetres for spatial axes.
        cmap : str or None
            Colormap (None = auto-select).
        geometry : list, optional
            Geometry objects for cross-section overlay (2D only).
        flip : bool
            Swap horizontal and vertical axes (2D only).
        density : int
            Target arrows per axis (vector mode only).
        threshold : float
            Suppress arrows below this fraction of peak (vector mode only).
        vmax : float or None
            Clip arrow length at this magnitude (vector mode only).
        figsize : (float, float) or None
            Figure size in inches ``(width, height)``.
        """
        import ipywidgets as widgets  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from IPython.display import clear_output, display  # noqa: PLC0415

        r = self._region
        data = self.data

        if r is None:
            raise RuntimeError("Monitor not attached / no data.")
        if r.ndim == 0:
            raise TypeError("0D monitors have no time axis to slide — use plot() instead.")

        is_field_group = component in ("E", "H")

        # Resolve the plotting plane once (validates normal/position for
        # 3D monitors); slicing across the leading time axis needs the
        # spatial slice axis shifted by one.
        pv = resolve_plane_view(r, normal, position) if r.ndim >= 2 else None

        def _all_times_2d(arr):
            if pv is None or pv.slice_index is None:
                return arr
            return np.take(arr, pv.slice_index, axis=pv.normal_idx + 1)

        if plot_type == "vector":
            # --- Vector mode ---
            if pv is None:
                raise ValueError(
                    "Vector interact requires a 2D monitor or a 3D monitor with a slice plane."
                )

            (i0, c0), (i1, _c1) = pv.free

            field_group = component if is_field_group else "E"
            comp_u = f"{field_group}{_AXES[i0]}"
            comp_v = f"{field_group}{_AXES[i1]}"
            comp_w = f"{field_group}{_AXES[pv.normal_idx]}"
            if comp_u not in data or comp_v not in data:
                raise KeyError(f"Need both {comp_u} and {comp_v} recorded.")

            # Pre-compute fixed arrow scale across all time steps
            # (matching plot_field_vector's auto-scale reference: full
            # 3D magnitude when the normal component is recorded)
            all_mag2 = _all_times_2d(data[comp_u]) ** 2 + _all_times_2d(data[comp_v]) ** 2
            if comp_w in data:
                all_mag2 = all_mag2 + _all_times_2d(data[comp_w]) ** 2
            all_mag = np.sqrt(all_mag2)
            global_max_mag = float(np.max(all_mag)) if np.any(all_mag > 0) else 1.0
            effective_max = min(global_max_mag, vmax) if vmax is not None else global_max_mag
            sc = 1e3 if scale_mm else 1.0
            sx = max(1, len(c0) // density)
            xs = c0[::sx]
            dx = float(np.mean(np.diff(xs))) * sc if len(xs) > 1 else 1.0
            fixed_scale = effective_max / dx if effective_max > 0 else 1.0

            def _render(t_index):
                with out:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=figsize)
                    self.plot(
                        component=component,
                        t_index=t_index,
                        normal=normal,
                        position=position,
                        plot_type="vector",
                        ax=ax,
                        scale_mm=scale_mm,
                        density=density,
                        cmap=cmap,
                        geometry=geometry,
                        flip=flip,
                        quiver_scale=fixed_scale,
                        threshold=threshold,
                        vmax=vmax,
                    )
                    plt.show()
        else:
            # --- Scalar mode ---
            arr = _all_times_2d(_resolve_component(data, component))
            is_amplitude = is_field_group

            if is_amplitude:
                effective_cmap = cmap or "viridis"
            else:
                effective_cmap = cmap or "RdBu_r"

            global_vmax = float(np.max(np.abs(arr))) if np.any(arr != 0) else 1.0
            fixed_vmin = 0.0 if is_amplitude else -global_vmax

            def _render(t_index):
                with out:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=figsize)
                    self.plot(
                        component=component,
                        t_index=t_index,
                        normal=normal,
                        position=position,
                        plot_type=plot_type,
                        ax=ax,
                        scale_mm=scale_mm,
                        cmap=effective_cmap,
                        geometry=geometry,
                        vmin=fixed_vmin,
                        vmax=global_vmax,
                        flip=flip,
                    )
                    plt.show()

        # Slider with human-readable time labels
        times_ns = self.t * 1e9
        options = [(f"{t_ns:.3f} ns", i) for i, t_ns in enumerate(times_ns)]
        slider = widgets.SelectionSlider(
            options=options,
            value=0,
            description="Time:",
            continuous_update=False,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="60%"),
        )

        out = widgets.Output()

        slider.observe(lambda change: _render(change["new"]), names="value")
        display(widgets.VBox([slider, out]))
        _render(0)

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
