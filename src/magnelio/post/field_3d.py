"""3D view of a field or a field monitor on the viewer's cutting plane.

The geometry viewer (:mod:`magnelio.post.plot_3d`) opens a model along
an axis-aligned plane; this module lays the field on that plane.  The
cell layer the cut exposes carries one value per cell — the magnitude
of E or H, or one signed component — as a coloured sheet, and, for a
field group, arrows on an even lattice over the layer.  The viewer's
position slider therefore walks through a recorded volume; a frame
slider (time or frequency) and a phase slider (complex data) come with
the source.

The values are the cell-centred physical fields of the exposed layer
alone, computed when the cut moves — a field picture stands for a cell
layer (DD-175).  Symmetry planes are not mirrored: the 3D view shows
the modelled half, as the geometry view does.

The source is abstracted as :class:`_FieldFrames` — region nodes, frame
labels and a layer loader — so the storage underneath the monitors can
change without touching the view.
"""

# Design: DD-259 (step 0 of the raw-monitor plan); the scene, cut and
# widget machinery is DD-190's.

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from magnelio.post import plot_3d as _viewer

__all__ = ["show_field"]

_AXES = ("x", "y", "z")
_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_PLOT_TYPES = ("vector", "color")

# The field sheet lies two hairs into the removed half — one more than
# the grid sheet — so that it draws over the grid when both are shown;
# the arrows sit one further hair out so they never fight the sheet.
_SHEET_HAIRS = 2.0
_ARROW_HAIRS = 3.0
_HAIR = 1e-3


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


@dataclass
class _FieldFrames:
    """What the view needs from a field source.

    Attributes
    ----------
    nodes : (x, y, z)
        Node coordinates [m] of the recorded region, ``n + 1`` per axis.
    components : tuple of str
        The recorded component names.
    labels : np.ndarray
        One label per frame: times [s], frequencies [Hz], or ``[0.0]``
        for a single field.
    kind : {"time", "frequency", "field"}
    is_complex : bool
    layer : callable
        ``layer(frame, axis, k, comps)`` returns ``{comp: array}`` with
        the cell-centred physical values of cell layer *k* along *axis*,
        shaped ``(nu, nv)`` over the two in-plane axes in ascending order.
    pec : np.ndarray or None
        ``(nx, ny, nz)`` mask of the cells buried in a perfect conductor;
        those are cut out of the sheet.
    name : str
    """

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray]
    components: tuple[str, ...]
    labels: np.ndarray
    kind: str
    is_complex: bool
    layer: Callable[[int, int, int, list[str]], dict[str, np.ndarray]]
    pec: np.ndarray | None = None
    name: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n.size - 1) for n in self.nodes)  # type: ignore[return-value]

    @property
    def n_frames(self) -> int:
        return int(self.labels.size)


def _region_nodes(grid, region) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(grid.x, dtype=float)[region.ix.start : region.ix.stop + 1],
        np.asarray(grid.y, dtype=float)[region.iy.start : region.iy.stop + 1],
        np.asarray(grid.z, dtype=float)[region.iz.start : region.iz.stop + 1],
    )


def _pec_cells(mesh, nodes, region=None) -> np.ndarray | None:
    """Mask of the region's cells buried in PEC, or ``None`` without a mesh."""
    if mesh is None:
        return None
    grid = mesh.grid
    if region is None:
        from magnelio.monitors.base import resolve_region  # noqa: PLC0415

        region = resolve_region(None, grid)
    own = _region_nodes(grid, region)
    for a, (mine, theirs) in enumerate(zip(nodes, own)):
        if mine.shape != theirs.shape or not np.allclose(mine, theirs, rtol=1e-9, atol=0.0):
            raise ValueError(
                f"mesh does not match the field's grid along {_AXES[a]}: "
                f"{theirs.size} vs {mine.size} nodes"
            )
    ids = np.asarray(mesh.material_id)
    pec_ids = [
        int(mid)
        for mid, mat in getattr(mesh, "material_library", {}).items()
        if getattr(mat, "is_pec", False)
    ]
    if not pec_ids:
        return None
    return np.isin(ids, pec_ids)[region.ix, region.iy, region.iz]


def _frames_from_field(fs, mesh) -> _FieldFrames:
    from magnelio.monitors.base import _interp_to_cell_centres  # noqa: PLC0415

    grid = fs._grid
    nodes = (
        np.asarray(grid.x, dtype=float),
        np.asarray(grid.y, dtype=float),
        np.asarray(grid.z, dtype=float),
    )
    full = [slice(0, grid.Nx), slice(0, grid.Ny), slice(0, grid.Nz)]

    def layer(frame, axis, k, comps):
        slabs = list(full)
        slabs[axis] = slice(k, k + 1)
        data = _interp_to_cell_centres(fs._raw, list(comps), *slabs, grid)
        return {c: np.squeeze(np.asarray(a), axis=axis) for c, a in data.items()}

    return _FieldFrames(
        nodes=nodes,
        components=_COMPONENTS,
        labels=np.zeros(1),
        kind="field",
        is_complex=bool(fs.is_complex),
        layer=layer,
        pec=_pec_cells(mesh, nodes),
        name="field",
    )


def _frames_from_time_monitor(mon, mesh) -> _FieldFrames:
    region = mon._region
    if region is None or mon._grid is None:
        raise RuntimeError(f"monitor {mon.name!r} is not attached to a mesh")
    if not mon._snapshots:
        raise RuntimeError(
            f"monitor {mon.name!r} holds no snapshots — on a project run they stream "
            "to the store; open the project and use its monitors instead"
        )
    nx, ny, nz = (s.stop - s.start for s in (region.ix, region.iy, region.iz))
    nodes = _region_nodes(mon._grid, region)
    snapshots = mon._snapshots

    def layer(frame, axis, k, comps):
        snap = snapshots[frame]
        return {c: np.take(np.asarray(snap[c]).reshape(nx, ny, nz), k, axis=axis) for c in comps}

    return _FieldFrames(
        nodes=nodes,
        components=tuple(mon._components),
        labels=np.asarray(mon.t, dtype=float),
        kind="time",
        is_complex=False,
        layer=layer,
        pec=_pec_cells(mesh, nodes, region),
        name=mon.name,
    )


def _frames_from_freq_monitor(mon, mesh) -> _FieldFrames:
    region = mon._region
    if region is None or mon._grid is None:
        raise RuntimeError(f"monitor {mon.name!r} is not attached to a mesh")
    nx, ny, nz = (s.stop - s.start for s in (region.ix, region.iy, region.iz))
    nodes = _region_nodes(mon._grid, region)
    freqs = np.asarray(mon.f, dtype=float)
    data = {c: np.asarray(a).reshape(freqs.size, nx, ny, nz) for c, a in mon.data.items()}

    def layer(frame, axis, k, comps):
        return {c: np.take(data[c][frame], k, axis=axis) for c in comps}

    return _FieldFrames(
        nodes=nodes,
        components=tuple(data),
        labels=freqs,
        kind="frequency",
        is_complex=True,
        layer=layer,
        pec=_pec_cells(mesh, nodes, region),
        name=mon.name,
    )


def _frames_from_loaded_time(reader, mesh) -> _FieldFrames:
    """Frames over a store reader; one frame is read from HDF5 at a time."""
    import h5py  # noqa: PLC0415

    path = reader._run_dir / "results.h5"
    with h5py.File(path, "r", swmr=True) as f:
        mg = f["monitors"][reader.name]
        nodes = tuple(np.asarray(mg[f"grid_{a}"][()], dtype=float) for a in _AXES)
    n = int(reader._n)
    cache: dict = {}

    def frame_arrays(frame: int, comps) -> dict[str, np.ndarray]:
        key = frame
        held = cache.get(key)
        if held is not None and all(c in held for c in comps):
            return held
        with h5py.File(path, "r", swmr=True) as f:
            mg = f["monitors"][reader.name]
            arrays = {
                c: np.transpose(np.asarray(mg[c][frame]), (2, 1, 0)) for c in comps
            }  # (nz, ny, nx) -> (nx, ny, nz)
        cache.clear()
        cache[key] = arrays
        return arrays

    def layer(frame, axis, k, comps):
        arrays = frame_arrays(frame, comps)
        return {c: np.take(arrays[c], k, axis=axis) for c in comps}

    region = None
    if mesh is not None:
        from magnelio.monitors.base import resolve_region  # noqa: PLC0415

        region = resolve_region(reader.corners, mesh.grid)
    return _FieldFrames(
        nodes=nodes,  # type: ignore[arg-type]
        components=tuple(reader._components),
        labels=np.asarray(reader.t, dtype=float)[:n],
        kind="time",
        is_complex=False,
        layer=layer,
        pec=_pec_cells(mesh, nodes, region),
        name=reader.name,
    )


def _frames_of(source, mesh) -> _FieldFrames:
    from magnelio.fields.state import FieldState  # noqa: PLC0415
    from magnelio.monitors.field_frequency import MonitorFieldFrequency  # noqa: PLC0415
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415

    if isinstance(source, FieldState):
        return _frames_from_field(source, mesh)
    if isinstance(source, MonitorFieldTime):
        return _frames_from_time_monitor(source, mesh)
    if isinstance(source, MonitorFieldFrequency):
        return _frames_from_freq_monitor(source, mesh)
    kind = type(source).__name__
    if kind == "_LoadedFieldMonitor":
        return _frames_from_loaded_time(source, mesh)
    if kind == "_LoadedFreqMonitor":
        return _frames_from_freq_monitor(source._hydrate(), mesh)
    raise TypeError(
        "show_field needs a FieldState, a MonitorFieldTime, a MonitorFieldFrequency "
        f"or a project's monitor reader; got {kind}"
    )


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------


def _available_components(recorded) -> list[str]:
    out = []
    for group in ("E", "H"):
        if all(f"{group}{a}" in recorded for a in _AXES):
            out.append(group)
    out.extend(c for c in _COMPONENTS if c in recorded)
    return out


def _layer_index(nodes_display, axis: int, cut) -> int | None:
    """Index of the cell layer the cut exposes, or ``None`` outside the region."""
    n = nodes_display[axis]
    if n.size < 2 or not (n[0] <= cut.position <= n[-1]):
        return None
    if cut.flip:
        return min(int(np.searchsorted(n, cut.position, side="left")), n.size - 2)
    return max(int(np.searchsorted(n, cut.position, side="left")) - 1, 0)


def _layer_sheet(nodes_display, axis: int, k: int, position: float, keep):
    """Cell faces of layer *k* as a polydata sheet in the plane *position*.

    Quads are ordered ``iu + nu * iv`` (u fastest), the ``order="F"``
    ravel of a ``(nu, nv)`` layer array; *keep* (same shape, or
    ``None``) drops cells from the sheet.  Returns ``None`` when no cell
    is left.
    """
    import pyvista as pv  # noqa: PLC0415

    u_axis, v_axis = (a for a in range(3) if a != axis)
    u, v = nodes_display[u_axis], nodes_display[v_axis]
    nu, nv = u.size - 1, v.size - 1
    uu, vv = np.meshgrid(u, v, indexing="xy")  # (nv+1, nu+1)
    points = np.empty(((nu + 1) * (nv + 1), 3), dtype=float)
    points[:, u_axis] = uu.ravel()
    points[:, v_axis] = vv.ravel()
    points[:, axis] = position
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="xy")
    p0 = (iu + (nu + 1) * iv).ravel()
    if keep is not None:
        sel = np.asarray(keep, dtype=bool).ravel(order="F")
        p0 = p0[sel]
        if p0.size == 0:
            return None
    faces = np.column_stack([np.full(p0.size, 4), p0, p0 + 1, p0 + nu + 2, p0 + nu + 1]).ravel()
    return pv.PolyData(points, faces=faces)


@dataclass
class _FieldView:
    """The field on the cut: state, actors and the notebook controls."""

    frames: _FieldFrames
    component: str
    plot_type: str
    frame: int
    phase: float
    vmax_fixed: float | None
    cmap: str | None
    unit_scale: float
    density: int
    threshold: float
    opacity: float
    arrow_color: str
    nodes_display: tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False)
    sheet_actor: Any = None
    arrow_actor: Any = None
    _layer_cache: dict = field(default_factory=dict, repr=False)
    _vmax_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.nodes_display = tuple(n * self.unit_scale for n in self.frames.nodes)  # type: ignore[assignment]

    # ── vocabulary ───────────────────────────────────────────────────────

    @property
    def is_group(self) -> bool:
        return self.component in ("E", "H")

    @property
    def has_arrows(self) -> bool:
        return self.plot_type == "vector" and self.is_group

    @property
    def group(self) -> str:
        return self.component[:1]

    def comps(self) -> list[str]:
        if self.is_group:
            return [f"{self.component}{a}" for a in _AXES]
        return [self.component]

    @property
    def unit(self) -> str:
        unit = "V/m" if self.group == "E" else "A/m"
        return f"{unit} per √W" if self.frames.kind == "frequency" else unit

    @property
    def bar_title(self) -> str:
        label = f"|{self.component}|" if self.is_group else self.component
        return f"{label} ({self.unit})"

    def frame_label(self) -> str:
        value = float(self.frames.labels[self.frame])
        if self.frames.kind == "time":
            return f"t = {value * 1e9:.4g} ns"
        if self.frames.kind == "frequency":
            return f"f = {value * 1e-9:.4g} GHz"
        return ""

    # ── values ───────────────────────────────────────────────────────────

    def _layer(self, axis: int, k: int) -> dict[str, np.ndarray]:
        key = (self.frame, axis, k, tuple(self.comps()))
        held = self._layer_cache.get(key)
        if held is None:
            held = self.frames.layer(self.frame, axis, k, self.comps())
            self._layer_cache.clear()
            self._layer_cache[key] = held
        return held

    def values(self, axis: int, k: int) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Scalar of the layer (magnitude or signed component) and its vectors."""
        layer = self._layer(axis, k)
        if self.frames.is_complex:
            phasor = np.exp(1j * np.deg2rad(self.phase))
            layer = {c: np.real(np.asarray(a) * phasor) for c, a in layer.items()}
        else:
            layer = {c: np.real(np.asarray(a, dtype=float)) for c, a in layer.items()}
        if self.is_group:
            mag = np.sqrt(sum(a**2 for a in layer.values()))
            return mag, layer
        return layer[self.component], None

    def vmax(self) -> float:
        """Colour and arrow ceiling: fixed, or the peak over every frame and layer."""
        if self.vmax_fixed is not None:
            return float(self.vmax_fixed)
        held = self._vmax_cache.get(self.component)
        if held is not None:
            return held
        comps = self.comps()
        axis = int(np.argmin(self.frames.shape))
        best = 0.0
        pec = self.frames.pec
        for fr in range(self.frames.n_frames):
            for k in range(self.frames.shape[axis]):
                layer = self.frames.layer(fr, axis, k, comps)
                if self.is_group:
                    v = np.sqrt(sum(np.abs(np.asarray(a)) ** 2 for a in layer.values()))
                else:
                    v = np.abs(np.asarray(layer[self.component]))
                if pec is not None:
                    # Cells inside a conductor are cut out of the sheet;
                    # they must not set its colour scale either.
                    v = v[~np.take(pec, k, axis=axis)]
                if v.size:
                    best = max(best, float(np.nanmax(v)))
        best = best if best > 0.0 else 1.0
        self._vmax_cache[self.component] = best
        return best

    # ── actors ───────────────────────────────────────────────────────────

    def _hide(self) -> None:
        for actor in (self.sheet_actor, self.arrow_actor):
            if actor is not None:
                actor.SetVisibility(False)

    def apply(self, scene, shown) -> None:
        """(Re)build the sheet and arrows for the scene's cut state."""
        cut = scene.cut
        if cut.axis is None:
            self._hide()
            return
        axis = _viewer._AXIS_INDEX[cut.axis]
        k = _layer_index(self.nodes_display, axis, cut)
        if k is None:
            self._hide()
            return
        n = self.nodes_display[axis]
        hair = _HAIR * (n[k + 1] - n[k]) * (-1.0 if cut.flip else 1.0)
        keep = None if self.frames.pec is None else ~np.take(self.frames.pec, k, axis=axis)
        scalar, vectors = self.values(axis, k)
        vmax = self.vmax()
        self._draw_sheet(scene, axis, k, cut.position + _SHEET_HAIRS * hair, keep, scalar, vmax)
        if self.sheet_actor is not None:
            self.sheet_actor.SetVisibility("field" in shown)
        if self.has_arrows and vectors is not None:
            self._draw_arrows(
                scene, axis, k, cut.position + _ARROW_HAIRS * hair, keep, vectors, vmax
            )
            if self.arrow_actor is not None:
                self.arrow_actor.SetVisibility("arrows" in shown)
        elif self.arrow_actor is not None:
            scene.plotter.remove_actor(self.arrow_actor, reset_camera=False, render=False)
            self.arrow_actor = None

    def _draw_sheet(self, scene, axis, k, position, keep, scalar, vmax) -> None:
        pl = scene.plotter
        sheet = _layer_sheet(self.nodes_display, axis, k, position, keep)
        if sheet is None:
            self._hide()
            return
        values = np.asarray(scalar, dtype=float).ravel(order="F")
        if keep is not None:
            values = values[np.asarray(keep, dtype=bool).ravel(order="F")]
        sheet.cell_data["field"] = values
        clim = (0.0, vmax) if self.is_group else (-vmax, vmax)
        cmap = self.cmap or ("viridis" if self.is_group else "RdBu_r")
        # Rebuilt rather than swapped, for the browser renderer's sake
        # (DD-190): a fresh actor carries fresh identities.
        self.sheet_actor = pl.add_mesh(
            sheet,
            scalars="field",
            cmap=cmap,
            clim=clim,
            opacity=self.opacity,
            lighting=False,
            show_edges=False,
            name="field_cut",
            reset_camera=False,
            render=False,
            scalar_bar_args={
                "title": self.bar_title,
                "vertical": True,
                "n_labels": 5,
                "fmt": "%.3g",
                # Leave room for the labels: the default bar sits so far
                # right that "2.74e+03" runs off the window edge.
                "position_x": 0.84,
                "position_y": 0.08,
                "width": 0.05,
                "height": 0.45,
            },
        )
        mapper = self.sheet_actor.mapper
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray("field")
        # A replaced actor's mapper is fed through a small pipeline whose
        # output stays empty until it executes; run it now so that any
        # reader of the mapper's input — the browser serialiser, a test —
        # sees the sheet without a render in between.
        mapper.Update()

    def _draw_arrows(self, scene, axis, k, position, keep, vectors, vmax) -> None:
        import pyvista as pv  # noqa: PLC0415

        from magnelio.post.plot_field import _arrow_grid, _resample  # noqa: PLC0415

        pl = scene.plotter
        if self.arrow_actor is not None:
            pl.remove_actor(self.arrow_actor, reset_camera=False, render=False)
            self.arrow_actor = None
        u_axis, v_axis = (a for a in range(3) if a != axis)
        u, v = self.nodes_display[u_axis], self.nodes_display[v_axis]
        uc, vc = 0.5 * (u[:-1] + u[1:]), 0.5 * (v[:-1] + v[1:])
        us_grid, vs_grid = _arrow_grid(uc, vc, self.density)
        comps = self.comps()
        arrays = [vectors[comps[u_axis]], vectors[comps[v_axis]], vectors[comps[axis]]]
        (au, av, aw), live = _resample(uc, vc, us_grid, vs_grid, arrays, keep)
        mag = np.sqrt(au**2 + av**2 + aw**2)
        mask = live & np.isfinite(mag) & (mag >= self.threshold * vmax)
        if not np.any(mask):
            return
        uu, vv = np.meshgrid(us_grid, vs_grid, indexing="ij")
        points = np.empty((int(mask.sum()), 3), dtype=float)
        points[:, u_axis] = uu[mask]
        points[:, v_axis] = vv[mask]
        points[:, axis] = position
        vec = np.empty_like(points)
        vec[:, u_axis] = au[mask]
        vec[:, v_axis] = av[mask]
        vec[:, axis] = aw[mask]
        spacing = min(
            float(us_grid[1] - us_grid[0]) if us_grid.size > 1 else np.inf,
            float(vs_grid[1] - vs_grid[0]) if vs_grid.size > 1 else np.inf,
        )
        if not np.isfinite(spacing):
            spacing = 0.1 * _viewer._diag(scene.bounds)
        cloud = pv.PolyData(points)
        cloud["vec"] = vec / np.maximum(mag[mask], 1e-300)[:, None]
        cloud["mag"] = np.minimum(mag[mask], vmax)
        glyphs = cloud.glyph(
            orient="vec",
            scale="mag",
            factor=spacing / vmax,
            geom=pv.Arrow(
                tip_length=0.3,
                tip_radius=0.1,
                shaft_radius=0.035,
                tip_resolution=8,
                shaft_resolution=8,
            ),
        )
        if glyphs.n_cells == 0:
            return
        self.arrow_actor = pl.add_mesh(
            glyphs,
            color=self.arrow_color,
            name="field_arrows",
            reset_camera=False,
            render=False,
        )
        self.arrow_actor.mapper.Update()

    # ── notebook controls ────────────────────────────────────────────────

    def attach_controls(self, server, key: str, refresh) -> Callable[[], None]:
        """Register the frame / phase / component handlers; return the widget builder."""
        from trame.widgets import html  # noqa: PLC0415
        from trame.widgets import vuetify3 as vuetify  # noqa: PLC0415

        state = server.state
        k_frame, k_phase, k_comp = f"{key}_frame", f"{key}_phase", f"{key}_comp"
        k_label = f"{key}_frame_label"
        state[k_frame] = int(self.frame)
        state[k_phase] = float(self.phase)
        state[k_comp] = self.component
        state[k_label] = self.frame_label()
        state[f"{key}_comps"] = _available_components(self.frames.components)
        n_frames = self.frames.n_frames

        @state.change(k_frame)
        def _on_frame(**kwargs):
            fr = int(kwargs[k_frame])
            if fr == self.frame or not (0 <= fr < n_frames):
                return
            self.frame = fr
            with state:
                state[k_label] = self.frame_label()
            refresh()

        @state.change(k_phase)
        def _on_phase(**kwargs):
            ph = float(kwargs[k_phase])
            if ph == self.phase:
                return
            self.phase = ph
            refresh()

        @state.change(k_comp)
        def _on_comp(**kwargs):
            comp = kwargs[k_comp]
            if comp == self.component or comp not in state[f"{key}_comps"]:
                return
            self.component = comp
            refresh()

        def items() -> None:
            if n_frames > 1:
                vuetify.VSlider(
                    v_model=(k_frame, state[k_frame]),
                    min=0,
                    max=n_frames - 1,
                    step=1,
                    hide_details=True,
                    density="compact",
                    style="width: 160px; margin-left: 12px;",
                )
                html.Span(f"{{{{ {k_label} }}}}", style="margin-left: 6px; white-space: nowrap;")
            if self.frames.is_complex:
                vuetify.VSlider(
                    v_model=(k_phase, state[k_phase]),
                    min=0,
                    max=360,
                    step=5,
                    thumb_label=True,
                    hide_details=True,
                    density="compact",
                    style="width: 120px; margin-left: 12px;",
                )
                html.Span("phase °", style="margin-left: 4px; white-space: nowrap;")
            vuetify.VSelect(
                v_model=(k_comp, state[k_comp]),
                items=(f"{key}_comps", state[f"{key}_comps"]),
                label="Field",
                density="compact",
                hide_details=True,
                variant="plain",
                style="width: 80px; margin-left: 8px;",
            )

        return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _default_normal(shape) -> str:
    """The axis with the fewest cells; ``z``, then ``y``, on a tie."""
    smallest = min(shape)
    for a in (2, 1, 0):
        if shape[a] == smallest:
            return _AXES[a]
    return "z"  # pragma: no cover


def _resolve_frame(frames: _FieldFrames, frame, t, f) -> int:
    given = [name for name, v in (("frame", frame), ("t", t), ("f", f)) if v is not None]
    if len(given) > 1:
        raise ValueError(f"give one of frame, t or f; got {given}")
    if t is not None:
        if frames.kind != "time":
            raise ValueError("t= selects a frame of a time monitor only")
        return int(np.argmin(np.abs(frames.labels - float(t))))
    if f is not None:
        if frames.kind != "frequency":
            raise ValueError("f= selects a frame of a frequency monitor only")
        return int(np.argmin(np.abs(frames.labels - float(f))))
    if frame is None:
        return 0
    fr = int(frame)
    if not (0 <= fr < frames.n_frames):
        raise IndexError(f"frame {fr} out of range for {frames.n_frames} frames")
    return fr


def show_field(
    source,
    component: str = "E",
    *,
    normal: str | None = None,
    position: float | None = None,
    flip: bool = False,
    plot_type: str = "vector",
    frame: int | None = None,
    t: float | None = None,
    f: float | None = None,
    phase: float = 0.0,
    vmax: float | None = None,
    cmap: str | None = None,
    density: int = 20,
    threshold: float = 0.02,
    opacity: float = 1.0,
    arrow_color: str = "#303030",
    geometry=None,
    mesh=None,
    show_ports: bool = True,
    show_wires: bool = True,
    show_grid: bool = False,
    show_labels: bool = True,
    mode: str | None = None,
    size: tuple[int, int] | None = None,
    quality: float = 1.0,
    scale_mm: bool = True,
    camera: Any = "iso",
):
    """Interactive 3D view of a field, or a field monitor, on a cutting plane.

    The view is the geometry viewer's (:func:`~magnelio.plots.show_geometry`)
    with the field laid on the cutting plane: the cell layer the cut
    exposes as a coloured sheet — the magnitude of a field group, or one
    signed component — and, for a field group, arrows on an even lattice
    over that layer.  Moving the position slider walks the cut through
    the recorded volume; a time or frequency monitor adds a frame slider,
    complex data a phase slider, and a selector switches the field.  The
    values are the cell-centred physical fields of the exposed layer,
    computed for that layer when the cut moves.

    Parameters
    ----------
    source : FieldState, MonitorFieldTime, MonitorFieldFrequency, or a project's monitor reader
        What to show.  A monitor must have recorded (or be read from a
        project); a :class:`~magnelio.fields.FieldState` is one frame.
    component : str
        ``"E"`` or ``"H"`` for the magnitude sheet (with arrows when
        *plot_type* is ``"vector"``); ``"Ex"``, ``"Hy"``, … for one
        signed component on a diverging colour scale.
    normal : {"x", "y", "z"}, optional
        Normal of the initial cutting plane.  Default: the axis along
        which the region has the fewest cells — for a plane monitor, its
        own normal.
    position : float, optional
        Initial plane position along *normal* [m].  Default: the middle
        of the recorded region.
    flip : bool, default False
        Which half the cut removes (see the geometry viewer).
    plot_type : {"vector", "color"}
        Arrows over the magnitude sheet, or the sheet alone.  A single
        component is always drawn as a sheet.
    frame : int, optional
        Initial frame (time or frequency index).  Default 0.
    t, f : float, optional
        Initial frame by time [s] (time monitors) or frequency [Hz]
        (frequency monitors); the nearest recorded one is used.
    phase : float, default 0.0
        Phase [degrees] at which a complex field is shown:
        ``Re(F · exp(j·phase))``.
    vmax : float, optional
        Ceiling of the colour scale and of the arrow length.  Default:
        the peak over every frame and layer of the region, so that the
        colours stay comparable while sliding through time.
    cmap : str, optional
        Colour map; default ``"viridis"`` for a magnitude, ``"RdBu_r"``
        for a signed component.
    density : int, default 20
        Arrows along the longer in-plane axis.
    threshold : float, default 0.02
        Arrows below this fraction of *vmax* are not drawn.
    opacity : float, default 1.0
        Opacity of the field sheet.
    arrow_color : str
        Colour of the arrows.
    geometry : GeometryModel, optional
        Draw the model's solids and features with the field.
    mesh : Mesh, optional
        The mesh the field was computed on.  Cells buried in a perfect
        conductor are cut out of the sheet, so the solids' cut faces
        show through where the field is not defined; with ``show_grid``
        the grid cells are drawn on the cut as well.
    show_ports, show_wires, show_labels : bool, default True
        As in the geometry viewer.
    show_grid : bool, default False
        With *mesh*: draw the grid cells on the cut under the field.
    mode, size, quality, scale_mm, camera
        As in :func:`~magnelio.plots.show_geometry`.

    Returns
    -------
    pyvista.Plotter or None
        The plotter when ``mode="none"``; otherwise the view is displayed
        as a side effect.

    Notes
    -----
    Symmetry planes are not mirrored in the 3D view; it shows the
    modelled half of the model, as the geometry view does.  Every
    sample is a cell-centre average of the staggered components — the
    picture stands for a layer of cells, not for a plane.
    """
    frames = _frames_of(source, mesh)
    if plot_type not in _PLOT_TYPES:
        raise ValueError(f"plot_type must be one of {_PLOT_TYPES}; got {plot_type!r}")
    available = _available_components(frames.components)
    if component not in available:
        raise KeyError(
            f"component {component!r} is not available from {frames.name!r}; recorded: {available}"
        )
    if normal is None:
        normal = _default_normal(frames.shape)
    elif normal not in _viewer._AXIS_INDEX:
        raise ValueError(f"normal must be 'x', 'y', or 'z'; got {normal!r}")
    axis = _viewer._AXIS_INDEX[normal]
    n = frames.nodes[axis]
    if position is None:
        position = 0.5 * float(n[0] + n[-1])
    elif not (n[0] <= position <= n[-1]):
        warnings.warn(
            f"position {position:g} m lies outside the recorded region "
            f"[{n[0]:g}, {n[-1]:g}] m along {normal}; the field sheet starts hidden",
            stacklevel=2,
        )

    unit_scale = 1e3 if scale_mm else 1.0
    view = _FieldView(
        frames=frames,
        component=component,
        plot_type=plot_type,
        frame=_resolve_frame(frames, frame, t, f),
        phase=float(phase),
        vmax_fixed=vmax,
        cmap=cmap,
        unit_scale=unit_scale,
        density=int(density),
        threshold=float(threshold),
        opacity=float(opacity),
        arrow_color=arrow_color,
    )
    extent = []
    for nodes in view.nodes_display:
        extent += [float(nodes[0]), float(nodes[-1])]

    notebook, mode, off_screen = _viewer._resolve_mode(mode)
    scene = _viewer._build_scene(
        geometry,
        mesh=mesh,
        cut=(normal, float(position)),
        flip=flip,
        show_ports=show_ports,
        show_wires=show_wires,
        show_grid=show_grid,
        show_labels=show_labels,
        size=size,
        render_edges=False,
        edge_color="#202020",
        quality=quality,
        scale_mm=scale_mm,
        camera=camera,
        off_screen=off_screen or (notebook and mode not in (None, "none")),
        field_view=view,
        extent=tuple(extent),
    )
    return _viewer._display(scene, mode, notebook)
