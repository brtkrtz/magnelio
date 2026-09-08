"""Slice plots and frame sliders over a field series (DD-259 step 3).

The two field monitors and the two store readers draw through the
functions here.  The series supplies the frames — a frame's layer is
averaged onto cell centres for the plane being drawn and for nothing
else — and the caller supplies the region's bookkeeping: the symmetry
planes the region touches (DD-154) and the run's grid, so the geometry
overlay states the thickness of the cell layer a picture stands for
(DD-175).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from magnelio.monitors.base import (
    _AXES,
    MonitorRegion,
    PlaneView,
    _resolve_component,
    component_mirror_key,
    mirror_extend,
    mirror_plane_arrays,
    mirror_sign,
    plane_slab_halfwidth,
    resolve_plane_view,
)

_GROUPS = ("E", "H")


def region_of_grid(grid) -> MonitorRegion:
    """The whole of *grid* as a monitor region (cell slices from zero)."""
    x, y, z = (np.asarray(a, dtype=float) for a in (grid.x, grid.y, grid.z))
    centres = [0.5 * (a[:-1] + a[1:]) for a in (x, y, z)]
    return MonitorRegion(
        ix=slice(0, grid.Nx),
        iy=slice(0, grid.Ny),
        iz=slice(0, grid.Nz),
        xc=centres[0],
        yc=centres[1],
        zc=centres[2],
        ndim=sum(1 for c in centres if c.size > 1),
    )


def at_phase(values: np.ndarray, phase: float | None) -> np.ndarray:
    """``Re(F · exp(-j·phase))`` of a complex array (degrees); real data passes.

    The library's phasors follow the ``e^{-j w t}`` convention — the
    running DFT accumulates ``sum F(t) e^{+j w t} dt``, so the instant
    of a pattern at time *t* is ``Re(F e^{-j w t})``.  *phase* is that
    ``w t`` in degrees: it advances with time, and a travelling wave
    moves the way it ran in the simulation.
    """
    # Design: DD-267 (the sign; it was +j, which ran every animation
    # backwards).  The same rule in post.field_3d._FieldView._instant
    # and fields.series._FieldSeries._snapshot.
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=float)
    if phase is None or phase == 0.0:
        return np.real(values)
    return np.real(values * np.exp(-1j * np.deg2rad(float(phase))))


@dataclass
class SeriesView:
    """A field series with the bookkeeping its pictures need.

    Attributes
    ----------
    series : FieldRecording or FieldSpectrum
    name : str
        The monitor's name, for titles.
    mirrors : tuple of MirrorSpec
        Symmetry planes the region touches; the pictures show the full
        structure.
    grid : GridLines, optional
        The run's grid — the geometry overlay takes the thickness of the
        displayed cell layer from it.  Default: the series' own grid.
    """

    series: Any
    name: str = ""
    mirrors: tuple = ()
    grid: Any = None

    @property
    def region(self) -> MonitorRegion:
        return region_of_grid(self.series.grid)

    @property
    def slab_grid(self):
        return self.grid if self.grid is not None else self.series.grid

    def label(self, index: int) -> str:
        return self.series._label_text(index)

    def index_of(self, value: float | None, index: int | None) -> int:
        """The frame index from an explicit index or a label value (time / frequency)."""
        if index is not None:
            return self.series._check_index(index)
        if value is not None:
            return self.series._nearest(value)
        return 0

    # ── the data of a picture ────────────────────────────────────────────

    def trace(self, component: str, phase: float | None = None) -> np.ndarray:
        """A point region's value over every frame, ``(n_frames,)``."""
        s = self.series
        comps = self._comps(component)
        cc = s.cell_centred(comps)
        data = {c: np.asarray(a).reshape(s.n_frames) for c, a in cc.items()}
        if component in _GROUPS:
            return _resolve_component(data, component)
        return at_phase(data[component], phase)

    def line(self, component: str, index: int, phase: float | None = None):
        """A line region's values at one frame: ``(free_axis, coords, values)``."""
        s = self.series
        comps = self._comps(component)
        cc = s.cell_centred(comps, frame=index)
        free = [a for a, n in enumerate(s.shape) if n > 1]
        axis = free[0] if free else 0
        data = {c: np.asarray(a).reshape(s.shape[axis]) for c, a in cc.items()}
        if component in _GROUPS:
            vals = _resolve_component(data, component)
        else:
            vals = at_phase(data[component], phase)
        coords = s.cell_centres[axis]
        fld, comp_axis = component_mirror_key(component)
        for spec in self.mirrors:
            if spec.axis != axis:
                continue
            coords, vals = mirror_extend(
                coords, vals, spec, 0, mirror_sign(fld, comp_axis, spec.axis, spec.kind)
            )
        return axis, coords, vals

    def plane(self, normal: str | None, position: float) -> PlaneView:
        return resolve_plane_view(self.region, normal, position)

    def layer(self, index: int, pv: PlaneView, comps, phase: float | None = None) -> dict:
        """The cell layer of one frame on the plane *pv*, mirrored across the in-plane planes."""
        k = 0 if pv.slice_index is None else int(pv.slice_index)
        raw = self.series.cell_centred_layer(index, pv.normal_idx, k, list(comps))
        return {c: at_phase(a, phase) for c, a in raw.items()}

    def _comps(self, component: str) -> list[str]:
        if component in _GROUPS:
            recorded = list(self.series.components)
            comps = [f"{component}{a}" for a in _AXES if f"{component}{a}" in recorded]
            if not comps:
                raise KeyError(f"no {component} component recorded; recorded: {recorded}")
            return comps
        if component not in self.series.components:
            raise KeyError(
                f"component {component!r} not recorded; recorded: {list(self.series.components)}"
            )
        return [component]

    def overlay(self, geometry, pv: PlaneView):
        if geometry is None:
            return None
        from magnelio.post.plot_field import CrossSectionOverlay  # noqa: PLC0415

        (i0, _), (i1, _) = pv.free
        return CrossSectionOverlay(
            geometry=geometry,
            normal=_AXES[pv.normal_idx],
            position=pv.normal_pos,
            mirrors=tuple(
                (0 if m.axis == i0 else 1, m.wall, m.at_low)
                for m in self.mirrors
                if m.axis in (i0, i1)
            ),
            slab=plane_slab_halfwidth(self.slab_grid, pv.normal_idx, pv.normal_pos),
        )

    @staticmethod
    def slice_note(pv: PlaneView, scale_mm: bool) -> str:
        if pv.slice_index is None:
            return ""
        if scale_mm:
            return f", {_AXES[pv.normal_idx]}={pv.normal_pos * 1e3:.3g} mm"
        return f", {_AXES[pv.normal_idx]}={pv.normal_pos:.3g} m"


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------


def plot_frame(
    view: SeriesView,
    component: str,
    index: int,
    *,
    phase: float | None = None,
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
    """Draw one frame of *view*: a trace (point), a line, or a plane.

    A point region ignores *index* and draws the value over every
    frame; a line region draws the values along its axis at the frame;
    a plane region draws the frame on its own plane and a volume on the
    plane selected with *normal* and *position*.  Complex series (a
    spectrum) are drawn at *phase* — component plots as
    ``Re(F·exp(-j·phase))``, group magnitudes as the envelope.
    """
    s = view.series
    region = view.region
    is_group = component in _GROUPS
    kind = s._kind

    if region.ndim == 0:
        from magnelio.monitors.plotting import plot_freq_0d, plot_time_0d  # noqa: PLC0415

        if kind == "frequency":
            arr = view.trace(component) if is_group else s.cell_centred([component])[component]
            arr = np.asarray(arr).reshape(s.n_frames)
            return plot_freq_0d(s.frequencies, arr, component, view.name, what="abs", ax=ax)
        return plot_time_0d(s.times, view.trace(component, phase), component, view.name, ax=ax)

    index = view.index_of(None, index)
    if region.ndim == 1:
        from magnelio.monitors.plotting import plot_time_1d  # noqa: PLC0415

        if kind == "frequency" and is_group:
            axis, coords, vals = view.line(component, index)
            label = f"|{component}|"
        else:
            axis, coords, vals = view.line(component, index, phase)
            label = component
        title = f"{view.name} — {label}, {view.label(index)}"
        if kind == "frequency" and not is_group and phase:
            title += f", phase={phase:.0f}°"
        return plot_time_1d(coords, vals, label, _AXES[axis], title, ax=ax, scale_mm=scale_mm)

    pv = view.plane(normal, position)
    (i0, c0), (i1, c1) = pv.free
    note = view.slice_note(pv, scale_mm)
    overlay = view.overlay(geometry, pv)

    from magnelio.post.plot_field import plot_field_scalar, plot_field_vector  # noqa: PLC0415

    if plot_type == "vector":
        group = component if is_group else "E"
        comps = view._comps(group)
        comp_u, comp_v, comp_w = (f"{group}{_AXES[a]}" for a in (i0, i1, pv.normal_idx))
        if comp_u not in comps or comp_v not in comps:
            raise KeyError(
                f"Need both {comp_u} and {comp_v} recorded.  Available: {list(s.components)}"
            )
        layer = view.layer(index, pv, comps, phase)
        u_arr, v_arr = layer[comp_u], layer[comp_v]
        w_arr = layer.get(comp_w)
        c0, c1, (u_arr, v_arr, w_arr) = mirror_plane_arrays(
            pv,
            view.mirrors,
            c0,
            c1,
            [(u_arr, group, i0), (v_arr, group, i1), (w_arr, group, pv.normal_idx)],
        )
        title = f"{view.name} — {group}-field, {view.label(index)}"
        if kind == "frequency" and phase:
            title += f", phase={phase:.0f}°"
        return plot_field_vector(
            c0,
            c1,
            u_arr,
            v_arr,
            w=w_arr,
            xlabel=_AXES[i0],
            ylabel=_AXES[i1],
            wlabel=_AXES[pv.normal_idx],
            title=title + note,
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
            colorbar=colorbar,
        )

    comps = view._comps(component)
    if is_group:
        # The envelope of a complex group, the magnitude of a real one.
        k = 0 if pv.slice_index is None else int(pv.slice_index)
        raw = s.cell_centred_layer(index, pv.normal_idx, k, comps)
        vals = np.sqrt(sum(np.abs(np.asarray(a)) ** 2 for a in raw.values()))
        label = component
    else:
        vals = view.layer(index, pv, comps, phase)[component]
        label = component
    fld, comp_axis = component_mirror_key(component)
    c0, c1, (vals,) = mirror_plane_arrays(pv, view.mirrors, c0, c1, [(vals, fld, comp_axis)])
    title = f"{view.name} — {label}, {view.label(index)}"
    if kind == "frequency" and not is_group and phase:
        title += f", phase={phase:.0f}°"
    if is_group:
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
        title=title + note,
        clabel=label,
        ax=ax,
        scale_mm=scale_mm,
        cmap=effective_cmap,
        vmin=vmin,
        vmax=vmax,
        symmetric=sym,
        plot_type=plot_type,
        flip=flip,
        geometry=overlay,
        colorbar=colorbar,
    )


# ---------------------------------------------------------------------------
# A slider over the frames
# ---------------------------------------------------------------------------


def interact(
    view: SeriesView,
    component: str = "E",
    *,
    normal: str | None = None,
    position: float = 0.0,
    plot_type: str = "vector",
    phase: float | None = None,
    scale_mm: bool = True,
    cmap: str | None = None,
    geometry=None,
    flip: bool = False,
    density: int = 20,
    threshold: float = 0.02,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    """A notebook slider over the frames of *view* (needs ipywidgets).

    The colour range (scalar) or the arrow scale (vector) is fixed over
    every frame of the drawn plane, for visual stability — computed
    plane by plane, so a volume recording is never averaged whole.
    """
    import ipywidgets as widgets  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from IPython.display import clear_output, display  # noqa: PLC0415

    s = view.series
    region = view.region
    if region.ndim == 0:
        raise TypeError("a point region has no frames to slide over — use plot() instead.")
    is_group = component in _GROUPS
    pv = view.plane(normal, position) if region.ndim >= 2 else None
    n = s.n_frames

    def layer_values(i):
        if pv is None:
            return view.line(component, i, phase)[2]
        comps = view._comps(component if (is_group or plot_type != "vector") else "E")
        return view.layer(i, pv, comps, phase)

    if plot_type == "vector":
        if pv is None:
            raise ValueError(
                "a vector slider needs a plane region, or a volume with a slice plane."
            )
        (i0, c0), (i1, _c1) = pv.free
        group = component if is_group else "E"
        comps = view._comps(group)
        comp_u, comp_v, comp_w = (f"{group}{_AXES[a]}" for a in (i0, i1, pv.normal_idx))
        if comp_u not in comps or comp_v not in comps:
            raise KeyError(f"Need both {comp_u} and {comp_v} recorded.")
        global_max = 0.0
        for i in range(n):
            layer = view.layer(i, pv, comps, phase)
            mag2 = layer[comp_u] ** 2 + layer[comp_v] ** 2
            if comp_w in layer:
                mag2 = mag2 + layer[comp_w] ** 2
            global_max = max(global_max, float(np.sqrt(mag2).max()) if mag2.size else 0.0)
        global_max = global_max if global_max > 0.0 else 1.0
        effective_max = min(global_max, vmax) if vmax is not None else global_max
        sc = 1e3 if scale_mm else 1.0
        sx = max(1, len(c0) // density)
        xs = c0[::sx]
        dx = float(np.mean(np.diff(xs))) * sc if len(xs) > 1 else 1.0
        fixed_scale = effective_max / dx if effective_max > 0 else 1.0
        fixed = {"quiver_scale": fixed_scale, "vmax": vmax}
    else:
        global_vmax = 0.0
        for i in range(n):
            vals = layer_values(i)
            if isinstance(vals, dict):
                vals = (
                    np.sqrt(sum(np.abs(a) ** 2 for a in vals.values()))
                    if is_group
                    else vals[component]
                )
            global_vmax = max(global_vmax, float(np.abs(vals).max()) if np.size(vals) else 0.0)
        global_vmax = global_vmax if global_vmax > 0.0 else 1.0
        fixed = {"vmin": 0.0 if is_group else -global_vmax, "vmax": global_vmax}

    out = widgets.Output()

    def _render(i):
        with out:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            plot_frame(
                view,
                component,
                i,
                phase=phase,
                normal=normal,
                position=position,
                plot_type=plot_type,
                ax=ax,
                scale_mm=scale_mm,
                cmap=cmap,
                geometry=geometry,
                flip=flip,
                density=density,
                threshold=threshold,
                **fixed,
            )
            plt.show()

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n - 1,
        step=1,
        description=view.series._kind,
        continuous_update=False,
    )
    label = widgets.Label(value=view.label(0))

    def _on_change(change):
        label.value = view.label(int(change["new"]))
        _render(int(change["new"]))

    slider.observe(_on_change, names="value")
    display(widgets.VBox([widgets.HBox([slider, label]), out]))
    _render(0)
    return slider
