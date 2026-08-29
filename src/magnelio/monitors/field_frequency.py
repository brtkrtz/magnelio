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
    _AXES,
    MonitorRegion,
    PlaneView,
    _corners_array,
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
        """Snap to grid and allocate DFT accumulators."""
        self._region = resolve_region(self.corners, mesh.grid)
        self._grid = mesh.grid  # per-edge lengths for the physical fields
        self._mirrors = resolve_mirrors(self._region, mesh)
        r = self._region
        spatial_shape = (
            r.ix.stop - r.ix.start,
            r.iy.stop - r.iy.start,
            r.iz.stop - r.iz.start,
        )
        self._accumulators = {}
        for comp in self._components:
            self._accumulators[comp] = DFTAccumulator(self.freqs, spatial_shape)

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Accumulate DFT contribution from the current time step.

        E-fields are at time ``t = n * dt`` and H-fields at
        ``t_H = (n + 0.5) * dt``.  The correct time is passed to the
        DFT accumulator for each component so that the Leapfrog
        staggering is handled automatically.

        With an ``interval``, steps off the stride return before the
        cell-centre interpolation — which is where a whole-volume
        monitor spends its time, so the saving is proportional.  The
        stride is keyed on the absolute step index, so a resumed run
        samples the same instants as an uninterrupted one.
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

        r = self._region

        # E-field components at time t_E = t (= n * dt)
        if self._e_components:
            e_data = _interp_to_cell_centres(
                fields, self._e_components, r.ix, r.iy, r.iz, self._grid
            )
            for comp, arr in e_data.items():
                self._accumulators[comp].accumulate(arr, t, dt_weight)

        # H-field components at time t_H = t + dt/2 (= (n + 0.5) * dt)
        if self._h_components:
            h_data = _interp_to_cell_centres(
                fields, self._h_components, r.ix, r.iy, r.iz, self._grid
            )
            t_h = t + 0.5 * dt
            for comp, arr in h_data.items():
                self._accumulators[comp].accumulate(arr, t_h, dt_weight)

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

        r = self._region
        return {
            "components": list(self._components),
            "fields": list(self.fields),
            "interval": self.interval,
            "symmetry": mirrors_to_jsonable(self._mirrors),
            "freqs": np.asarray(self.freqs, dtype=float),
            "corners": _corners_array(self.corners),
            "grid_x": np.asarray(r.xc, dtype=float),
            "grid_y": np.asarray(r.yc, dtype=float),
            "grid_z": np.asarray(r.zc, dtype=float),
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

    def _squeeze_spatial(self, arr: np.ndarray) -> np.ndarray:
        """Squeeze length-1 spatial dimensions, keep frequency axis 0."""
        spatial_squeeze = []
        for ax in range(1, arr.ndim):
            if arr.shape[ax] == 1:
                spatial_squeeze.append(ax)
        if spatial_squeeze:
            arr = np.squeeze(arr, axis=tuple(spatial_squeeze))
        return arr

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
                f"monitor {self.name!r}: .data needs a source reference.  "
                f"Without one the accumulated bins are the raw DFT of the "
                f"transient — the field folded with the excitation spectrum, "
                f"in field units x seconds — not fields per 1 W CW.  A monitor "
                f"that took part in a scattering run is renormalised "
                f"automatically; if this one was filled by hand, call "
                f".renormalize(result.reference_signal).  Read .data_raw for "
                f"the raw bins themselves."
            )

    @property
    def data(self) -> dict[str, np.ndarray]:
        """Recorded fields per 1 W incident CW power.

        Each bin is divided by the spectrum of the run's excitation, so
        E is in V/m and H in A/m, both per √W of incident power.  Raises
        if no source reference is available (see :meth:`renormalize`);
        :attr:`data_raw` returns the undivided bins instead.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are component names.  Values have shape
            ``(n_freqs, <spatial dims>)``, complex128.
            For a 0D monitor the spatial dims are squeezed away,
            giving shape ``(n_freqs,)``.
        """
        self._require_source()
        out = {}
        for comp in self._components:
            acc = self._accumulators.get(comp)
            if acc is not None:
                arr = self._apply_renorm(acc.result)  # shape (Nf, nx, ny, nz)
                out[comp] = self._squeeze_spatial(arr)
        return out

    @property
    def data_raw(self) -> dict[str, np.ndarray]:
        """Raw DFT bins, in field units x seconds.

        The running sum ``Σ field(t_n)·exp(+jω t_n)·dt`` as accumulated,
        i.e. the field folded with the spectrum of the excitation
        waveform.  Always returns these, whether or not a source
        reference is set — :attr:`data` is the physical counterpart.

        Returns
        -------
        dict[str, np.ndarray]
            Same layout as :attr:`data`.
        """
        out = {}
        for comp in self._components:
            acc = self._accumulators.get(comp)
            if acc is not None:
                arr = self._squeeze_spatial(acc.result)
                out[comp] = arr
        return out

    def component(self, name: str) -> np.ndarray:
        """Return DFT data for a single component.

        Parameters
        ----------
        name : str
            Component name, e.g. ``"Ez"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_freqs, <spatial dims>)``, complex128.
        """
        d = self.data
        if name not in d:
            raise KeyError(f"Component '{name}' not recorded. Available: {list(d.keys())}")
        return d[name]

    @property
    def region(self) -> MonitorRegion | None:
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

    @staticmethod
    def _apply_phase(arr_complex: np.ndarray, phase_deg: float) -> np.ndarray:
        """Extract instantaneous value at *phase_deg* from complex phasor."""
        if phase_deg == 0.0:
            return arr_complex.real
        phasor = np.exp(1j * np.deg2rad(phase_deg))
        return (arr_complex * phasor).real

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

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
        **kwargs,
    ):
        """Plot DFT field data.

        For 0D monitors: line plot over frequency (ignores *f* / *f_index*).
        For 2D monitors: colour-map, contour, or quiver plot at a frequency.
        For 3D monitors: the same plane plots on a slice selected with
        *normal* and *position*.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for vector / amplitude plots.
            ``"Ex"``, ``"Hz"``, … for a single component (scalar only).
        f : float, optional
            Frequency [Hz].  Nearest is used.
        f_index : int, optional
            Frequency index (overrides *f*).
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for 3D monitors (required there).  For a
            2D monitor it may be given for validation but is redundant.
        position : float
            Slice-plane position along *normal* [m]; snapped to the
            nearest cell-centre plane (3D monitors only).
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        phase : float
            Phase angle [degrees] for extracting the instantaneous field
            from complex phasors: ``Re(F · exp(j·phase·π/180))``.
            Ignored for amplitude plots (``component="E"``/``"H"``).
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

        # --- 0D: line plot over frequency ---
        if ndim == 0:
            from magnelio.monitors.plotting import plot_freq_0d  # noqa: PLC0415

            arr = _resolve_component(data, component)
            # For 0D freq plots, show magnitude by default
            what = "abs"
            return plot_freq_0d(
                self.freqs,
                arr,
                component,
                self.name,
                what=what,
                ax=ax,
            )

        # Resolve frequency index
        if f_index is None:
            if f is not None:
                f_index = int(np.argmin(np.abs(self.freqs - f)))
            else:
                f_index = 0

        # --- 1D: line plot along the free axis ---
        if ndim == 1:
            from magnelio.monitors.plotting import plot_time_1d  # noqa: PLC0415

            arr = _resolve_component(data, component)
            if is_field_group:
                vals = np.abs(arr[f_index]) if np.iscomplexobj(arr) else arr[f_index]
                label = f"|{component}|"
                title = f"{self.name} — |{component}|, f={self.freqs[f_index]:.4e} Hz"
            else:
                vals = self._apply_phase(arr[f_index], phase)
                label = component
                title = (
                    f"{self.name} — {component}, f={self.freqs[f_index]:.4e} Hz, phase={phase:.0f}°"
                )
            axes_labels = ["x", "y", "z"]
            for i, c in enumerate((r.xc, r.yc, r.zc)):
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
                        label,
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
            field_group = component if is_field_group else "E"
            comp_u = f"{field_group}{_AXES[i0]}"
            comp_v = f"{field_group}{_AXES[i1]}"
            comp_w = f"{field_group}{_AXES[pv.normal_idx]}"
            if comp_u not in data or comp_v not in data:
                raise KeyError(
                    f"Need both {comp_u} and {comp_v} recorded.  Available: {list(data.keys())}"
                )
            u_arr = self._apply_phase(pv.take2d(data[comp_u][f_index]), phase)
            v_arr = self._apply_phase(pv.take2d(data[comp_v][f_index]), phase)
            w_arr = (
                self._apply_phase(pv.take2d(data[comp_w][f_index]), phase)
                if comp_w in data
                else None
            )
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
            title = (
                f"{self.name} — {field_group}-field, "
                f"f={self.freqs[f_index]:.4e} Hz, "
                f"phase={phase:.0f}°{note}"
            )
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

        if is_amplitude:
            # |E| is phase-independent and always real+non-negative
            vals = np.abs(arr[f_index]) if np.iscomplexobj(arr) else arr[f_index]
            effective_cmap = cmap or "viridis"
            sym = False
            if vmin is None:
                vmin = 0.0
            title = f"{self.name} — |{component}|, f={self.freqs[f_index]:.4e} Hz{note}"
        else:
            vals = self._apply_phase(arr[f_index], phase)
            effective_cmap = cmap or "RdBu_r"
            sym = True
            title = (
                f"{self.name} — {component}, "
                f"f={self.freqs[f_index]:.4e} Hz, phase={phase:.0f}°{note}"
            )
        vals = pv.take2d(vals)
        fld, comp_axis = component_mirror_key(component)
        c0, c1, (vals,) = mirror_plane_arrays(
            pv,
            self._mirrors,
            c0,
            c1,
            [(vals, fld, comp_axis)],
        )

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
        """Interactive frequency slider for DFT field snapshots (Jupyter notebook).

        Requires ``ipywidgets``.  The colour range (scalar) or arrow
        scale (vector) is fixed across all frequencies for visual stability.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for vector / amplitude plots.
            ``"Ex"``, ``"Ez"``, … for individual component scalar plots.
        normal : {"x", "y", "z"}, optional
            Slice-plane normal for 3D monitors (required there); the
            slider then runs over frequency at a fixed slice plane.
        position : float
            Slice-plane position along *normal* [m] (3D monitors only).
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        phase : float
            Phase angle [degrees] for instantaneous field extraction.
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
            raise TypeError("0D monitors have no frequency axis to slide — use plot() instead.")

        is_field_group = component in ("E", "H")

        # Resolve the plotting plane once (validates normal/position for
        # 3D monitors); slicing across the leading frequency axis needs
        # the spatial slice axis shifted by one.
        pv = resolve_plane_view(r, normal, position) if r.ndim >= 2 else None

        def _all_freqs_2d(arr):
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

            # Pre-compute fixed arrow scale (at given phase), matching
            # plot_field_vector's auto-scale reference: full 3D
            # magnitude when the normal component is recorded
            all_u = self._apply_phase(_all_freqs_2d(data[comp_u]), phase)
            all_v = self._apply_phase(_all_freqs_2d(data[comp_v]), phase)
            all_mag2 = all_u**2 + all_v**2
            if comp_w in data:
                all_mag2 = all_mag2 + self._apply_phase(_all_freqs_2d(data[comp_w]), phase) ** 2
            all_mag = np.sqrt(all_mag2)
            global_max_mag = float(np.max(all_mag)) if np.any(all_mag > 0) else 1.0
            effective_max = min(global_max_mag, vmax) if vmax is not None else global_max_mag
            sc = 1e3 if scale_mm else 1.0
            sx = max(1, len(c0) // density)
            xs = c0[::sx]
            dx = float(np.mean(np.diff(xs))) * sc if len(xs) > 1 else 1.0
            fixed_scale = effective_max / dx if effective_max > 0 else 1.0

            def _render(f_index):
                with out:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=figsize)
                    self.plot(
                        component=component,
                        f_index=f_index,
                        normal=normal,
                        position=position,
                        plot_type="vector",
                        phase=phase,
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
            arr = _all_freqs_2d(_resolve_component(data, component))
            is_amplitude = is_field_group

            if is_amplitude:
                derived = np.abs(arr) if np.iscomplexobj(arr) else arr
                effective_cmap = cmap or "viridis"
            else:
                derived = self._apply_phase(arr, phase)
                effective_cmap = cmap or "RdBu_r"

            global_vmax = float(np.max(np.abs(derived))) if np.any(derived != 0) else 1.0
            fixed_vmin = 0.0 if is_amplitude else -global_vmax
            fixed_vmax = global_vmax

            def _render(f_index):
                with out:
                    clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=figsize)
                    self.plot(
                        component=component,
                        f_index=f_index,
                        normal=normal,
                        position=position,
                        plot_type=plot_type,
                        phase=phase,
                        ax=ax,
                        scale_mm=scale_mm,
                        cmap=effective_cmap,
                        geometry=geometry,
                        vmin=fixed_vmin,
                        vmax=fixed_vmax,
                        flip=flip,
                    )
                    plt.show()

        # Slider with human-readable frequency labels
        freqs_ghz = self.freqs * 1e-9
        options = [(f"{fg:.4g} GHz", i) for i, fg in enumerate(freqs_ghz)]
        slider = widgets.SelectionSlider(
            options=options,
            value=0,
            description="Freq:",
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
