"""
MonitorFieldFrequency — records frequency-domain fields via running DFT.

Accumulates the discrete Fourier transform of E and/or H during the
simulation.  The result is a set of complex-valued field arrays at each
requested frequency.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from magnelio.monitors._dft import DFTAccumulator, divide_by_spectrum, source_spectrum
from magnelio.monitors.base import (
    MonitorRegion,
    _corners_array,
    _expand_field_list,
    _region_dual,
    _region_slices,
    _take_raw,
    region_grid,
    resolve_mirrors,
    resolve_region,
)

# Sub-sampling limits, in samples per period of the *highest* requested
# frequency (DD-140).  Unlike a time monitor's recording interval, a DFT
# interval is genuine under-sampling: the running sum is a Riemann
# integral of an oscillating integrand, so too few samples per period do
# not coarsen the output — they corrupt the bins.  Nyquist (2) is the
# theoretical floor and far too optimistic for the integral; below
# ``_MIN_SAMPLES_PER_PERIOD`` the monitor refuses, and between that and
# ``_SAFE_SAMPLES_PER_PERIOD`` it warns.
_MIN_SAMPLES_PER_PERIOD = 4.0
_SAFE_SAMPLES_PER_PERIOD = 10.0


@dataclass
class MonitorFieldFrequency:
    """Record complex-valued fields at specified frequencies via running DFT.

    Parameters
    ----------
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        recorded box [m] — the same form
        as :meth:`Brick.from_corners`.  Corner order does not matter.
        An axis whose two values coincide is degenerate and records a
        single cell layer (plane, line, point).  A component may be
        ``None`` (or ``±math.inf``) to reach the domain boundary on
        that side.  Omit *corners* entirely for the whole domain.
    freqs : array_like
        Target frequencies [Hz].
    fields : list[str]
        Field groups or components to record.  ``"E"`` expands to
        ``["Ex", "Ey", "Ez"]``, ``"H"`` to ``["Hx", "Hy", "Hz"]``.
    interval : float, optional
        Seconds between DFT contributions.  The default (``None``)
        accumulates at **every** time step, which for a whole-volume
        monitor is arithmetic comparable to the solver itself and can
        double a run's wall-clock time.  Sub-sampling cuts that cost
        proportionally: the recorded step count, and with it the
        cell-centre interpolation and the complex accumulation, drop by
        the same factor.

        Unlike :class:`~magnelio.monitors.MonitorFieldTime`, where the
        interval only decides how many snapshots are kept, this one is
        real under-sampling of an oscillating integrand.  Two
        conditions must hold, and only the first can be checked here:

        * the interval must resolve the monitor's own highest
          frequency — below four samples per period the run is
          rejected, below ten it warns;
        * the *fields* must carry nothing above the resulting Nyquist
          frequency, or that content folds onto the requested bins.
          The monitor cannot know the excitation bandwidth, so this is
          the caller's judgement: an interval chosen from ``f_max`` of
          the analysis rather than from the monitor's own frequencies
          is always safe.

        Rounded **down** to a whole number of time steps (at least
        one), so the realised spacing never exceeds the one asked for;
        the integration weight follows exactly, so the result stays in
        the same units and ``renormalize`` is unaffected.
    name : str
        Monitor label (must be unique within a simulation).

    Examples
    --------
    >>> mon = MonitorFieldFrequency(
    ...     corners=((None, None, 5e-3), (None, None, 5e-3)),
    ...     freqs=np.linspace(1e9, 10e9, 50),
    ...     fields=["E", "H"],
    ...     name="EH_xy_5GHz",
    ... )

    A whole-volume monitor on a band that ends at 3.4 GHz, sampled at
    20 points per period of that top frequency instead of every step:

    >>> mon = MonitorFieldFrequency(
    ...     freqs=[2.87e9, 2.91e9],
    ...     fields=["E"],
    ...     interval=1.0 / (20 * 3.4e9),
    ...     name="E_volume",
    ... )
    """

    freqs: np.ndarray
    corners: object = None
    fields: list[str] = field(default_factory=lambda: ["E"])
    interval: float | None = None
    name: str = ""

    # --- internal ---
    _region: MonitorRegion | None = field(default=None, repr=False, init=False)
    _components: list[str] = field(default_factory=list, repr=False, init=False)
    _accumulators: dict[str, DFTAccumulator] = field(default_factory=dict, repr=False, init=False)
    _e_components: list[str] = field(default_factory=list, repr=False, init=False)
    _h_components: list[str] = field(default_factory=list, repr=False, init=False)
    _source_spectrum: np.ndarray | None = field(default=None, repr=False, init=False)
    # |a(f)| / |W(f)| of the excited channel on the monitor frequencies:
    # the incident power wave the run launched per unit excitation
    # waveform (1 for lumped and TEM ports, Z(f_calc)/Z(f)-shaped for
    # TE/TM ports).  Runtime wiring by the analysis, kept in the dump.
    _incident_amplitude: np.ndarray | None = field(default=None, repr=False, init=False)
    _step_stride: int | None = field(default=None, repr=False, init=False)
    # Symmetry planes the region touches (DD-154) — plots mirror the
    # recorded half across them on read.
    _mirrors: tuple = field(default=(), repr=False, init=False)
    # The region as its own grid, the dual widths of its h samples and
    # the per-component slices of the solver arrays (DD-259): the bins
    # accumulate the grid quantities on the Yee positions.
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
        >>> mon = MonitorFieldFrequency.from_ranges(x1=0, dx=5e-3, freqs=[2.9e9], fields=["E"])
        """
        from magnelio.geo._ranges import corners_from_ranges  # noqa: PLC0415

        return cls(
            corners=corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz),
            **kwargs,
        )

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float)
        if self.freqs.ndim != 1 or len(self.freqs) == 0:
            raise ValueError("freqs must be a non-empty 1D array")
        if self.interval is not None:
            if not self.interval > 0.0:
                raise ValueError(f"interval must be positive; got {self.interval}")
            self.interval = float(self.interval)
        self._components = _expand_field_list(self.fields)
        self._e_components = [c for c in self._components if c.startswith("E")]
        self._h_components = [c for c in self._components if c.startswith("H")]
        if not self.name:
            self.name = f"field_freq_{id(self):x}"

    def _resolve_stride(self, dt: float) -> int:
        """Recording stride in time steps, validated against ``freqs``.

        Resolved on the first recorded step, the first moment ``dt`` is
        known to a monitor; a rejected interval therefore stops the run
        immediately rather than after it has been paid for.
        """
        if self.interval is None:
            return 1
        # Round DOWN: the interval is an upper bound on the sample
        # spacing, so rounding up would sample coarser than asked for —
        # and an interval derived from a samples-per-period rule would
        # then trip the very margin it was chosen to keep.
        stride = max(1, int(self.interval / dt * (1.0 + 1e-9)))
        f_top = float(np.max(self.freqs))
        if f_top <= 0.0:
            return stride
        per_period = 1.0 / (f_top * stride * dt)
        if per_period < _MIN_SAMPLES_PER_PERIOD:
            raise ValueError(
                f"Monitor {self.name!r}: interval={self.interval:.4g} s is "
                f"{stride} time steps, leaving {per_period:.2f} samples per "
                f"period at the highest requested frequency "
                f"({f_top / 1e9:.4g} GHz).  A DFT accumulator integrates an "
                f"oscillating signal, so this does not coarsen the result — "
                f"it corrupts it.  Use interval <= "
                f"{1.0 / (_SAFE_SAMPLES_PER_PERIOD * f_top):.4g} s "
                f"({_SAFE_SAMPLES_PER_PERIOD:.0f} samples per period), and "
                f"size it from the excitation's highest frequency rather "
                f"than the monitor's when they differ."
            )
        if per_period < _SAFE_SAMPLES_PER_PERIOD:
            warnings.warn(
                f"Monitor {self.name!r}: {per_period:.1f} samples per period "
                f"at {f_top / 1e9:.4g} GHz — the DFT integration error grows "
                f"quadratically as this falls.  interval <= "
                f"{1.0 / (_SAFE_SAMPLES_PER_PERIOD * f_top):.4g} s restores "
                f"the {_SAFE_SAMPLES_PER_PERIOD:.0f}-samples-per-period "
                f"margin.",
                UserWarning,
                stacklevel=3,
            )
        return stride

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        """Snap to grid and allocate one DFT accumulator per staggered component."""
        from magnelio.fields.state import _yee_shapes  # noqa: PLC0415

        self._region = resolve_region(self.corners, mesh.grid)
        self._grid = mesh.grid  # per-edge lengths for the physical fields
        self._mirrors = resolve_mirrors(self._region, mesh)
        r = self._region
        self._subgrid = region_grid(mesh.grid, r)
        self._dual = _region_dual(mesh.grid, r.ix, r.iy, r.iz)
        self._slices = {c: _region_slices(r.ix, r.iy, r.iz, c) for c in self._components}
        shapes = _yee_shapes(self._subgrid.Nx, self._subgrid.Ny, self._subgrid.Nz)
        self._accumulators = {}
        for comp in self._components:
            self._accumulators[comp] = DFTAccumulator(self.freqs, shapes[comp])

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Accumulate DFT contribution from the current time step.

        E-fields are at time ``t = n * dt`` and H-fields at
        ``t_H = (n + 0.5) * dt``.  The correct time is passed to the
        DFT accumulator for each component so that the Leapfrog
        staggering is handled automatically.

        The bins accumulate the grid quantities on the region's Yee
        positions; nothing is averaged until the spectrum is read.  The
        time stamps are the ones the port recorder uses for the same
        step (``t`` for the sample of ``e`` taken after step *n*), so a
        renormalised pattern is phase-consistent with the run's
        S-parameters.

        With an ``interval``, steps off the stride return before the
        copy — which is where a whole-volume monitor spends its time, so
        the saving is proportional.  The stride is keyed on the absolute
        step index, so a resumed run samples the same instants as an
        uninterrupted one.
        """
        if self._region is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")

        if self._step_stride is None:
            self._step_stride = self._resolve_stride(dt)
        stride = self._step_stride
        if stride > 1 and n % stride:
            return
        # The Riemann weight is the sample spacing actually used, so the
        # bins keep their units and ``renormalize`` is unaffected.  The
        # leapfrog half-step below stays on the *solver* dt: it is where
        # H physically sits, not a property of the sampling.
        dt_weight = stride * dt

        raw = _take_raw(fields, self._components, self._slices)
        # E-field components at time t_E = t (= n * dt)
        for comp in self._e_components:
            self._accumulators[comp].accumulate(raw[comp], t, dt_weight)
        # H-field components at time t_H = t + dt/2 (= (n + 0.5) * dt)
        t_h = t + 0.5 * dt
        for comp in self._h_components:
            self._accumulators[comp].accumulate(raw[comp], t_h, dt_weight)

    def finalize(self) -> None:
        """Called after the simulation completes (no-op for DFT monitors)."""
        pass

    # ------------------------------------------------------------------
    # Result persistence (DD-070 follow-up)
    # ------------------------------------------------------------------

    def result_dump(self) -> dict:
        """The DFT result + geometry needed to persist and reload it.

        Unlike a time monitor, the accumulator does not stream append-only:
        it is a fixed-size running sum, *both* the live result (a partial
        DFT, readable from the first dump) and the resume state (reloaded to
        keep integrating).  The bins are the raw complex sums — renormalising
        to 1 W stays a reader/user step, exactly as for the in-RAM monitor.
        The reader/hydrator needs the region coordinates and frequencies too,
        so they travel with the bins.
        """
        if self._region is None:
            raise RuntimeError("monitor not attached; nothing to dump")
        from magnelio.monitors.base import mirrors_to_jsonable  # noqa: PLC0415

        g = self._subgrid
        dual = self._dual
        return {
            "components": list(self._components),
            "fields": list(self.fields),
            "interval": self.interval,
            "symmetry": mirrors_to_jsonable(self._mirrors),
            "freqs": np.asarray(self.freqs, dtype=float),
            "corners": _corners_array(self.corners),
            # The region's own grid lines (nodes) and the dual widths of
            # its h samples — what rebuilds the FieldSpectrum (DD-259).
            "grid_x": np.asarray(g.x, dtype=float),
            "grid_y": np.asarray(g.y, dtype=float),
            "grid_z": np.asarray(g.z, dtype=float),
            "dual_x": np.asarray(dual[0], dtype=float),
            "dual_y": np.asarray(dual[1], dtype=float),
            "dual_z": np.asarray(dual[2], dtype=float),
            "bins": {comp: self._accumulators[comp].result for comp in self._components},
            "incident_amplitude": (
                np.ones(len(self.freqs))
                if self._incident_amplitude is None
                else np.asarray(self._incident_amplitude, dtype=float)
            ),
        }

    def load_result_dump(self, dump: dict) -> None:
        """Restore the DFT accumulators from a :meth:`result_dump` (resume).

        The monitor must already be attached (fresh zero accumulators of the
        right shape); this overwrites their bins in place so the resumed run
        keeps integrating from the checkpointed partial DFT.
        """
        bins = dump["bins"]
        for comp in self._components:
            if comp in bins:
                self._accumulators[comp]._bins[...] = np.asarray(bins[comp])
        if "incident_amplitude" in dump:
            self._incident_amplitude = np.asarray(dump["incident_amplitude"], dtype=float)

    # ------------------------------------------------------------------
    # Source renormalization
    # ------------------------------------------------------------------

    def renormalize(self, source_signal) -> None:
        """Normalize DFT data to 1 W incident CW power.

        A monitor that took part in a scattering run is renormalised
        for you when the run ends, and so is one read back from a
        project store — call this only for a monitor filled by hand, or
        to divide by a reference other than the run's own excitation.

        Divides each DFT frequency bin by the source-waveform spectrum.
        The excitation waveform *is* the incident power-wave
        amplitude ``a(t)`` in √W for lumped, TEM and quasi-TEM feeds, so
        the renormalised fields are exactly the fields of a **1 W CW
        excitation** at each monitor frequency (gated by
        ``test_port_units.py::test_frequency_monitor_fields_per_1w_cw``).
        A TE/TM feed launches a frequency-dependent power per unit
        waveform (its wave impedance varies across the band); the
        analysis wires that ratio into the monitor after the run so the
        same statement holds there.

        The source spectrum is computed in the same Fourier convention
        as the internal DFT accumulator (``exp(+jωt)`` with ``dt``
        integration weight), so the division is consistent.

        Repeating the call is harmless: the accumulated bins are never
        modified, so this only replaces the divisor.

        Parameters
        ----------
        source_signal : Signal1D
            Excitation waveform of the run — pass
            ``result.reference_signal``.
        """
        self._source_spectrum = source_spectrum(
            source_signal.values,
            source_signal.dt,
            self.freqs,
        )

    @property
    def is_renormalized(self) -> bool:
        """Whether 1 W renormalization has been applied."""
        return self._source_spectrum is not None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _apply_renorm(self, arr: np.ndarray) -> np.ndarray:
        """Divide *arr* (freq axis 0) by the source spectrum.

        And by the launched incident amplitude ratio where the analysis
        wired one — the per-1-W reference of a TE/TM-fed run.
        """
        out = divide_by_spectrum(arr, self._source_spectrum)
        if self._incident_amplitude is not None:
            ratio = np.asarray(self._incident_amplitude, dtype=float)
            out = out / ratio.reshape(-1, *([1] * (out.ndim - 1)))
        return out

    def _set_incident_amplitude(self, f_axis, ratio) -> None:
        # Runtime wiring by the analysis: |a(f)| / |W(f)| of the excited
        # channel interpolated onto the monitor frequencies.
        self._incident_amplitude = np.interp(
            self.freqs, np.asarray(f_axis, dtype=float), np.asarray(ratio, dtype=float)
        )

    @property
    def f(self) -> np.ndarray:
        """Frequency array [Hz]."""
        return self.freqs

    def _require_source(self) -> None:
        """Refuse to hand out raw bins under the name of physical fields."""
        if self._source_spectrum is None:
            raise RuntimeError(
                f"monitor {self.name!r}: .spectrum needs a source reference.  "
                f"Without one the accumulated bins are the raw DFT of the "
                f"transient — the field folded with the excitation spectrum, "
                f"in field units x seconds — not fields per 1 W CW.  A monitor "
                f"that took part in a scattering run is renormalised "
                f"automatically; if this one was filled by hand, call "
                f".renormalize(result.reference_signal).  Read .spectrum_raw for "
                f"the raw transform itself."
            )

    def _spectrum(self, renormalised: bool):
        from magnelio.fields.series import FieldSpectrum  # noqa: PLC0415

        if self._region is None or self._subgrid is None:
            raise RuntimeError(f"monitor {self.name!r} is not attached to a mesh")
        raw = {
            comp: (self._apply_renorm(acc.result) if renormalised else acc.result)
            for comp, acc in self._accumulators.items()
        }
        return FieldSpectrum._from_raw(
            self._subgrid, np.asarray(self.freqs, dtype=float), raw, dual=self._dual
        )

    @property
    def spectrum(self):
        """The pattern as a :class:`~magnelio.fields.FieldSpectrum`, per 1 W CW.

        Every frame keeps the Yee staggering of the region.  Raises
        without a source reference (see :meth:`renormalize`);
        :attr:`spectrum_raw` is the undivided transform.
        """
        self._require_source()
        return self._spectrum(True)

    @property
    def spectrum_raw(self):
        """The raw transform as a :class:`~magnelio.fields.FieldSpectrum`."""
        return self._spectrum(False)

    # ------------------------------------------------------------------
    # Pictures — drawn from the spectrum (DD-259)
    # ------------------------------------------------------------------

    def _view(self):
        from magnelio.monitors._frame_plots import SeriesView  # noqa: PLC0415

        return SeriesView(
            self.spectrum,
            name=self.name,
            mirrors=self._mirrors,
            grid=getattr(self, "_grid", None),
        )

    def plot(
        self,
        component: str = "E",
        f: float | None = None,
        f_index: int | None = None,
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        phase: float = 0.0,
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
    ):
        """Plot the pattern at one frequency.

        A point region draws its magnitude over frequency (ignores *f*
        / *f_index*); a line region the values along its axis; a plane
        region the pattern on its own plane; a volume the plane selected
        with *normal* and *position*.  Only the drawn layer is averaged
        onto cell centres.  The pattern is per 1 W CW (see
        :attr:`spectrum`).

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for a vector plot or the envelope
            magnitude, ``"Ex"``, ``"Hz"``, … for a single component.
        f : float, optional
            Frequency [Hz]; the nearest frame is drawn.
        f_index : int, optional
            Frame index (overrides *f*).
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for a volume (required there).
        position : float
            Slice-plane position along *normal* [m]; snapped to the
            nearest cell-centre plane.
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        phase : float
            Instant of the complex pattern in degrees,
            ``Re(F · exp(j·phase))``; a group magnitude is the envelope
            and ignores it.
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
            view.index_of(f, f_index),
            phase=phase,
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
        )

    def interact(
        self,
        component: str = "E",
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        phase: float = 0.0,
        scale_mm: bool = True,
        cmap: str | None = None,
        geometry=None,
        flip: bool = False,
        density: int = 20,
        threshold: float = 0.02,
        vmax: float | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """A notebook slider over the frequencies (needs ipywidgets).

        The colour range (scalar) or arrow scale (vector) is fixed over
        every frequency for visual stability.  The arguments are those
        of :meth:`plot`.
        """
        from magnelio.monitors._frame_plots import interact as _interact  # noqa: PLC0415

        return _interact(
            self._view(),
            component,
            normal=normal,
            position=position,
            plot_type=plot_type,
            phase=phase,
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
        """Interactive 3D view of the DFT field on a cutting plane.

        The geometry viewer with the field laid on its cut, a phase
        slider for the complex pattern, a frequency slider when several
        bins were recorded, and the position slider walking through the
        region.  See :func:`magnelio.plots.show_field` for the arguments
        — the bin (``f=`` or ``frame=``), ``phase``, the plane
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
        return (
            f"MonitorFieldFrequency(name={self.name!r}, "
            f"fields={self.fields}, "
            f"n_freqs={len(self.freqs)}, "
            f"region={shape})"
        )


def renormalize_all(monitors, source_signal) -> None:
    """Hand *source_signal* to every frequency monitor in *monitors*.

    A run knows its own excitation; its monitors would otherwise hold
    bins that no caller can interpret.  Applied once per run, at the
    point where the reference waveform is sampled, so that ``.data``
    speaks physical units from the moment the run returns.  Monitors of
    other kinds are ignored.
    """
    from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

    for mon in monitors or ():
        if isinstance(mon, (MonitorFieldFrequency, MonitorFarFieldFrequency)):
            mon.renormalize(source_signal)
