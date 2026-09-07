"""3D view of a field or a field monitor on the viewer's cutting plane.

The geometry viewer (:mod:`magnelio.post.plot_3d`) opens a model along
an axis-aligned plane; this module lays the field on that plane and,
on request, into the volume.  The cell layer the cut exposes carries
one value per cell — the magnitude of E or H, or one signed component —
as a coloured sheet, and, for a field group, arrows on an even lattice
over the layer.  The volume behind the cut can carry arrows on a 3D
lattice, coloured by magnitude, and isosurfaces of the magnitude (or
the ±level of a signed component), both clipped to the kept half like
the solids.  The viewer's position slider therefore walks through a
recorded volume; a frame slider (time or frequency), a phase slider
(complex data), a level slider (isosurfaces) and a density slider
(arrows) come with the source, in a second toolbar row.

The values are the cell-centred physical fields of the exposed layer
alone, computed when the cut moves — a field picture stands for a cell
layer (DD-175) — and of the whole region when a volume representation
is shown.  With a mesh that declares symmetry planes the frames are
continued across them, so the view shows the whole model like every
other field picture (DD-154); without one it shows the modelled part.

The source is abstracted as :class:`_FieldFrames` — region nodes, frame
labels, a layer loader and a volume loader — so the storage underneath
the monitors can change without touching the view.
"""

# Design: DD-259 (step 0 of the raw-monitor plan), DD-261 (the volume
# representations, the arrow style and the second toolbar row), DD-263
# (mirroring, eigenmode frames, glyph styles, the phase play, the fixed
# label widths); the scene, cut and widget machinery is DD-190's.

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
# The shortest arrow drawn, as a fraction of the lattice spacing: a
# decaying field keeps readable arrows instead of vanishing ones.
_ARROW_FLOOR = 0.3
_ISO_OPACITY = 0.5
_VOLUME_CHOICES = ("arrows", "isosurface", "both")
_VOLUME_GROUPS = ("volume arrows", "isosurface")
_GLYPH_CHOICES = ("arrow", "cone")
# Degrees the phase advances per tick of its play button.
_PHASE_STEP = 10.0


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
        One label per frame: times [s], frequencies [Hz], the modes'
        eigenfrequencies [Hz], or ``[0.0]`` for a single field.
    kind : {"time", "frequency", "mode", "field"}
    is_complex : bool
    layer : callable
        ``layer(frame, axis, k, comps)`` returns ``{comp: array}`` with
        the cell-centred physical values of cell layer *k* along *axis*,
        shaped ``(nu, nv)`` over the two in-plane axes in ascending order.
    pec : np.ndarray or None
        ``(nx, ny, nz)`` mask of the cells buried in a perfect conductor;
        those are cut out of the sheet.
    name : str
    volume : callable
        ``volume(frame, comps)`` returns ``{comp: array}`` with the
        cell-centred physical values of the whole region, shaped
        ``(nx, ny, nz)`` — what the volume representations draw.
    mirrored : bool
        True when the frames are continued across the model's symmetry
        planes: the nodes span the whole model then.
    """

    nodes: tuple[np.ndarray, np.ndarray, np.ndarray]
    components: tuple[str, ...]
    labels: np.ndarray
    kind: str
    is_complex: bool
    layer: Callable[[int, int, int, list[str]], dict[str, np.ndarray]]
    pec: np.ndarray | None = None
    name: str = ""
    volume: Callable[[int, list[str]], dict[str, np.ndarray]] | None = None
    mirrored: bool = False

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
    if getattr(mesh, "material_id", None) is None:
        return None  # a bare grid holder, no materials to cut out
    ids = np.asarray(mesh.material_id)
    pec_ids = [
        int(mid)
        for mid, mat in getattr(mesh, "material_library", {}).items()
        if getattr(mat, "is_pec", False)
    ]
    if not pec_ids:
        return None
    return np.isin(ids, pec_ids)[region.ix, region.iy, region.iz]


def _region_in(grid, nodes):
    """The cell slices of *grid* that carry the region with these *nodes*."""
    from magnelio.monitors.base import MonitorRegion  # noqa: PLC0415

    slices = []
    for own, mine in zip((grid.x, grid.y, grid.z), nodes):
        own = np.asarray(own, dtype=float)
        i0 = int(np.searchsorted(own, mine[0] - 1e-12 * max(1.0, abs(mine[0]))))
        slices.append(slice(i0, i0 + mine.size - 1))
    centres = [0.5 * (m[:-1] + m[1:]) for m in nodes]
    return MonitorRegion(
        ix=slices[0],
        iy=slices[1],
        iz=slices[2],
        xc=centres[0],
        yc=centres[1],
        zc=centres[2],
        ndim=sum(1 for m in nodes if m.size > 2),
    )


def _mirror_specs(mesh, nodes) -> tuple:
    """The symmetry planes the region with *nodes* reaches, from the mesh's declaration.

    Empty without a mesh, without a declaration, or when the region
    stops short of every declared plane (the monitors' own rule,
    :func:`~magnelio.monitors.base.resolve_mirrors`).
    """
    if mesh is None or getattr(mesh, "boundary_conditions", None) is None:
        return ()
    from magnelio.monitors.base import resolve_mirrors  # noqa: PLC0415

    return tuple(resolve_mirrors(_region_in(mesh.grid, nodes), mesh))


def _mirrored_mask(mask, nodes, mirrored_nodes, mirrors) -> np.ndarray:
    """*mask* over the reduced region's cells, laid onto the mirrored region's.

    Every mirrored cell centre is folded back across the planes (in the
    reverse order they were applied) onto the reduced region and takes
    that cell's value; the cell a magnetic wall bisects, whose centre
    lies on the wall, takes the wall cell's.
    """
    index = []
    for a in range(3):
        c = 0.5 * (mirrored_nodes[a][:-1] + mirrored_nodes[a][1:])
        for spec in reversed(mirrors):
            if int(spec.axis) != a:
                continue
            fold = np.abs(c - spec.wall)
            c = spec.wall + fold if spec.at_low else spec.wall - fold
        n = nodes[a]
        index.append(np.clip(np.searchsorted(n, c, side="right") - 1, 0, n.size - 2))
    return mask[np.ix_(*index)]


def _frames_from_states(
    get_state,
    *,
    n_frames: int,
    labels,
    kind: str,
    is_complex: bool,
    components,
    name: str,
    mesh,
    mirror: bool,
) -> _FieldFrames:
    """Frames over per-frame :class:`~magnelio.fields.FieldState` accessors.

    *get_state(i)* returns frame *i*; one frame is held at a time, so a
    store reader's monitor is never loaded whole.  With *mirror* and a
    mesh that declares symmetry planes the region reaches, every frame
    is continued across them (:meth:`FieldState.mirrored`, exact on
    electric and magnetic walls) and the PEC mask follows.
    """
    from magnelio.fields._interp import _interp_to_cell_centres  # noqa: PLC0415

    first = get_state(0)
    grid0 = first._grid
    nodes0 = tuple(np.asarray(v, dtype=float) for v in (grid0.x, grid0.y, grid0.z))
    region = _region_in(mesh.grid, nodes0) if mesh is not None else None
    pec = _pec_cells(mesh, nodes0, region)
    mirrors = _mirror_specs(mesh, nodes0) if mirror else ()
    cache: dict = {}

    def state(i):
        fs = cache.get(i)
        if fs is None:
            fs = first if (i == 0 and not cache) else get_state(i)
            if mirrors:
                fs = fs.mirrored(*mirrors)
            cache.clear()
            cache[i] = fs
        return fs

    grid = state(0)._grid
    nodes = tuple(np.asarray(v, dtype=float) for v in (grid.x, grid.y, grid.z))
    if pec is not None and mirrors:
        pec = _mirrored_mask(pec, nodes0, nodes, mirrors)
    full = [slice(0, grid.Nx), slice(0, grid.Ny), slice(0, grid.Nz)]

    def layer(frame, axis, k, comps):
        fs = state(frame)
        slabs = list(full)
        slabs[axis] = slice(k, k + 1)
        data = _interp_to_cell_centres(fs._raw, list(comps), *slabs, fs._grid, dual=fs._dual)
        return {c: np.squeeze(np.asarray(a), axis=axis) for c, a in data.items()}

    def volume(frame, comps):
        fs = state(frame)
        data = _interp_to_cell_centres(fs._raw, list(comps), *full, fs._grid, dual=fs._dual)
        return {c: np.asarray(a) for c, a in data.items()}

    return _FieldFrames(
        nodes=nodes,  # type: ignore[arg-type]
        components=tuple(components),
        labels=np.asarray(labels, dtype=float).reshape(-1)[:n_frames],
        kind=kind,
        is_complex=bool(is_complex),
        layer=layer,
        pec=pec,
        name=name,
        volume=volume,
        mirrored=bool(mirrors),
    )


def _frames_from_series(series, mesh, mirror=True, name=None) -> _FieldFrames:
    """Frames over a :class:`~magnelio.fields.FieldRecording` / ``FieldSpectrum``."""
    return _frames_from_states(
        series.frame,
        n_frames=series.n_frames,
        labels=series._labels,
        kind=series._kind,
        is_complex=series.is_complex,
        components=series.components,
        name=type(series).__name__ if name is None else name,
        mesh=mesh,
        mirror=mirror,
    )


def _frames_from_field(fs, mesh, mirror=True) -> _FieldFrames:
    return _frames_from_states(
        lambda _i: fs,
        n_frames=1,
        labels=np.zeros(1),
        kind="field",
        is_complex=fs.is_complex,
        components=_COMPONENTS,
        name="field",
        mesh=mesh,
        mirror=mirror,
    )


def _frames_from_loaded_time(reader, mesh, mirror=True) -> _FieldFrames:
    """Frames over a store reader; one frame is read from HDF5 at a time."""
    return _frames_from_states(
        reader.frame,
        n_frames=int(np.asarray(reader.t).size),
        labels=np.asarray(reader.t, dtype=float),
        kind="time",
        is_complex=False,
        components=reader.components,
        name=reader.name,
        mesh=mesh,
        mirror=mirror,
    )


def _mode_state(result, i: int):
    """Mode *i* of an eigenmode result at the instant of maximum energy.

    An eigenvector's global phase is arbitrary; a complex (Bloch) mode
    is turned so its real part carries the maximum of the energy over
    all instants — the rule of the result's own ``plot`` — and the
    phase slider turns it further from there.
    """
    from magnelio._fields.field_arrays import FieldArrays  # noqa: PLC0415
    from magnelio.fields.state import FieldState  # noqa: PLC0415

    fs = result.field(i)
    if not fs.is_complex:
        return fs
    moment = sum(np.sum(np.asarray(fs.component(c), dtype=complex) ** 2) for c in _COMPONENTS)
    rotation = np.exp(-0.5j * np.angle(moment)) if moment != 0 else 1.0
    raw = fs._raw
    turned = FieldArrays(**{c: np.asarray(getattr(raw, c)) * rotation for c in _COMPONENTS})
    return FieldState._from_raw(fs._grid, turned, dual=fs._dual)


def _frames_from_eigenmodes(result, mesh, mirror=True) -> _FieldFrames:
    """Frames over the modes of an :class:`~magnelio.solver.eigenmode_result.EigenmodeResult`."""
    if result.n_modes == 0:
        raise ValueError("the eigenmode result holds no mode to show")
    return _frames_from_states(
        lambda i: _mode_state(result, i),
        n_frames=result.n_modes,
        labels=np.asarray(result.frequencies, dtype=float),
        kind="mode",
        is_complex=any(np.iscomplexobj(m.Ex) for m in result.modes),
        components=_COMPONENTS,
        name="eigenmodes",
        mesh=result.mesh if mesh is None else mesh,
        mirror=mirror,
    )


def _frames_of(source, mesh, mirror: bool = True) -> _FieldFrames:
    from magnelio.fields.series import _FieldSeries  # noqa: PLC0415
    from magnelio.fields.state import FieldState  # noqa: PLC0415
    from magnelio.monitors.field_frequency import MonitorFieldFrequency  # noqa: PLC0415
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
    from magnelio.solver.eigenmode_result import EigenmodeResult  # noqa: PLC0415

    if isinstance(source, FieldState):
        return _frames_from_field(source, mesh, mirror)
    if isinstance(source, _FieldSeries):
        return _frames_from_series(source, mesh, mirror)
    if isinstance(source, MonitorFieldTime):
        return _frames_from_series(source.recording, mesh, mirror, name=source.name)
    if isinstance(source, MonitorFieldFrequency):
        return _frames_from_series(source.spectrum, mesh, mirror, name=source.name)
    if isinstance(source, EigenmodeResult):
        return _frames_from_eigenmodes(source, mesh, mirror)
    kind = type(source).__name__
    if kind == "_LoadedFieldMonitor":
        return _frames_from_loaded_time(source, mesh, mirror)
    if kind == "_LoadedFreqMonitor":
        return _frames_from_series(source._hydrate().spectrum, mesh, mirror, name=source.name)
    raise TypeError(
        "show_field needs a FieldState, a MonitorFieldTime, a MonitorFieldFrequency, "
        f"an EigenmodeResult or a project's monitor reader; got {kind}"
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


def _lattice3(centres, density: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic 3D arrow raster over the cell-centre extents.

    *density* counts arrows along the longest axis; the others get the
    count that keeps the spacing equal (the 3D counterpart of
    :func:`~magnelio.post.plot_field._arrow_grid`).
    """
    centres = tuple(np.asarray(c, dtype=float) for c in centres)
    spans = [float(c[-1] - c[0]) if c.size > 1 else 0.0 for c in centres]
    span = max(spans)
    if span <= 0.0 or density < 2:
        return centres
    step = span / (density - 1)
    out = []
    for c, sp in zip(centres, spans):
        if sp <= 0.0:
            out.append(c)
        else:
            out.append(np.linspace(c[0], c[-1], max(2, int(round(sp / step)) + 1)))
    return tuple(out)  # type: ignore[return-value]


def _resample3(centres, raster, arrays, valid):
    """Trilinear resampling of cell-centre volumes onto the arrow raster.

    Invalid cells (a cell buried in a conductor) are dropped from the
    stencil instead of read as zeros, so no field is smeared into or
    out of the metal; a raster point whose stencil is more than half
    invalid yields NaN and is reported as not live.
    """
    idx, frac = [], []
    for c, r in zip(centres, raster):
        c = np.asarray(c, dtype=float)
        r = np.asarray(r, dtype=float)
        if c.size < 2:
            idx.append(np.zeros(r.size, dtype=int))
            frac.append(np.zeros(r.size))
            continue
        i = np.clip(np.searchsorted(c, r, side="right") - 1, 0, c.size - 2)
        idx.append(i)
        frac.append(np.clip((r - c[i]) / np.diff(c)[i], 0.0, 1.0))
    n_cells = [np.asarray(c).size for c in centres]
    total = [np.zeros(tuple(r.size for r in raster)) for _ in arrays]
    norm = np.zeros(tuple(r.size for r in raster))
    for corner in range(8):
        ii, ww = [], []
        for a in range(3):
            up = (corner >> a) & 1
            ii.append(np.minimum(idx[a] + up, n_cells[a] - 1))
            ww.append(frac[a] if up else 1.0 - frac[a])
        w = ww[0][:, None, None] * ww[1][None, :, None] * ww[2][None, None, :]
        sel = np.ix_(*ii)
        if valid is not None:
            w = w * np.asarray(valid, dtype=float)[sel]
        norm += w
        for k, a in enumerate(arrays):
            total[k] += w * np.asarray(a, dtype=float)[sel]
    live = norm > 0.5
    safe = np.where(live, norm, 1.0)
    return [np.where(live, t / safe, np.nan) for t in total], live


@dataclass
class _FieldView:
    """The field on the cut and in the volume: state, actors and the notebook controls."""

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
    arrow_color: str | None
    fps: float = 4.0
    volume_start: str | None = None
    iso_level: float = 0.5
    levels: tuple[float, ...] | None = None
    glyph: str = "arrow"
    glyph_width: float = 1.0
    nodes_display: tuple[np.ndarray, np.ndarray, np.ndarray] = field(init=False)
    _play_task: Any = field(default=None, repr=False)
    _phase_task: Any = field(default=None, repr=False)
    sheet_actor: Any = None
    arrow_actor: Any = None
    volume_actor: Any = None
    iso_actor: Any = None
    # The polydata behind the actors live as long as the view: every
    # frame and every cut change is written *into* them (DD-259 step 0
    # finding d), never swapped or re-created.
    _sheet_pd: Any = field(default=None, repr=False)
    _arrow_pd: Any = field(default=None, repr=False)
    _volume_pd: Any = field(default=None, repr=False)
    _iso_pd: Any = field(default=None, repr=False)
    _iso_grid: Any = field(default=None, repr=False)
    _bar_title: str = field(default="", repr=False)
    _painted: dict = field(default_factory=dict, repr=False)
    _layer_cache: dict = field(default_factory=dict, repr=False)
    _volume_cache: dict = field(default_factory=dict, repr=False)
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
    def can_iso(self) -> bool:
        """Isosurfaces need a volume: at least two cells along every axis."""
        return self.frames.volume is not None and min(self.frames.shape) >= 2

    @property
    def has_volume(self) -> bool:
        return self.frames.volume is not None

    def groups(self) -> set[str]:
        """The toolbar groups this view offers."""
        out = {"field"}
        if self.has_arrows:
            out.add("arrows")
            if self.has_volume:
                out.add("volume arrows")
        if self.can_iso:
            out.add("isosurface")
        return out

    def hidden_at_start(self) -> set[str]:
        """Groups hidden until the Show menu turns them on."""
        wanted = set()
        if self.volume_start in ("arrows", "both"):
            wanted.add("volume arrows")
        if self.volume_start in ("isosurface", "both"):
            wanted.add("isosurface")
        hidden = set(_VOLUME_GROUPS) - wanted
        if "volume arrows" in wanted:
            # Arrows in the volume replace the arrows on the cut.
            hidden.add("arrows")
        return hidden

    @property
    def group(self) -> str:
        return self.component[:1]

    def comps(self) -> list[str]:
        if self.is_group:
            return [f"{self.component}{a}" for a in _AXES]
        return [self.component]

    @property
    def unit(self) -> str:
        if self.frames.kind == "mode":
            return "a.u."  # an eigenvector carries no absolute amplitude
        unit = "V/m" if self.group == "E" else "A/m"
        return f"{unit} per √W" if self.frames.kind == "frequency" else unit

    @property
    def bar_title(self) -> str:
        label = f"|{self.component}|" if self.is_group else self.component
        return f"{label} ({self.unit})"

    def _label_format(self) -> tuple[str, float, str, int] | None:
        """``(prefix, scale, unit, decimals)`` of the frame labels, or ``None``.

        The decimals are fixed once per series from the smallest step
        between frames, so every label has the same width and the
        toolbar does not reflow while the frames play.
        """
        kind = self.frames.kind
        if kind == "time":
            prefix, scale, unit = "t", 1e9, "ns"
        elif kind in ("frequency", "mode"):
            prefix, scale, unit = "f", 1e-9, "GHz"
        else:
            return None
        values = np.asarray(self.frames.labels, dtype=float) * scale
        if values.size < 2:
            return prefix, scale, unit, -1  # one frame: general format
        steps = np.diff(np.unique(values))
        step = float(steps.min()) if steps.size else 0.0
        decimals = 0 if step <= 0.0 else int(np.clip(np.ceil(-np.log10(step)) + 1, 0, 3))
        return prefix, scale, unit, decimals

    def frame_label(self, frame: int | None = None) -> str:
        fmt = self._label_format()
        if fmt is None:
            return ""
        prefix, scale, unit, decimals = fmt
        i = self.frame if frame is None else int(frame)
        value = float(self.frames.labels[i]) * scale
        number = f"{value:.4g}" if decimals < 0 else f"{value:.{decimals}f}"
        if self.frames.kind == "mode":
            return f"mode {i}  {number} {unit}"
        return f"{prefix} = {number} {unit}"

    def label_chars(self) -> int:
        """Characters of the widest frame label — the span's minimum width."""
        return max((len(self.frame_label(i)) for i in range(self.frames.n_frames)), default=0)

    @property
    def colour_map(self) -> str:
        return self.cmap or ("viridis" if self.is_group else "RdBu_r")

    def colour_range(self, vmax: float) -> tuple[float, float]:
        return (0.0, vmax) if self.is_group else (-vmax, vmax)

    # ── values ───────────────────────────────────────────────────────────

    def _layer(self, axis: int, k: int) -> dict[str, np.ndarray]:
        key = (self.frame, axis, k, tuple(self.comps()))
        held = self._layer_cache.get(key)
        if held is None:
            held = self.frames.layer(self.frame, axis, k, self.comps())
            self._layer_cache.clear()
            self._layer_cache[key] = held
        return held

    def _volume(self) -> dict[str, np.ndarray]:
        key = (self.frame, tuple(self.comps()))
        held = self._volume_cache.get(key)
        if held is None:
            if self.frames.volume is None:
                raise RuntimeError("this field source offers no volume")
            held = self.frames.volume(self.frame, self.comps())
            self._volume_cache.clear()
            self._volume_cache[key] = held
        return held

    def _instant(self, data: dict) -> dict[str, np.ndarray]:
        """Real values of *data* at the view's phase (complex sources)."""
        if self.frames.is_complex:
            phasor = np.exp(1j * np.deg2rad(self.phase))
            return {c: np.real(np.asarray(a) * phasor) for c, a in data.items()}
        return {c: np.real(np.asarray(a, dtype=float)) for c, a in data.items()}

    def _scalar_and_vectors(self, data: dict):
        if self.is_group:
            mag = np.sqrt(sum(a**2 for a in data.values()))
            return mag, data
        return data[self.component], None

    def values(self, axis: int, k: int) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Scalar of the layer (magnitude or signed component) and its vectors."""
        return self._scalar_and_vectors(self._instant(self._layer(axis, k)))

    def volume_values(self) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Scalar of the whole region ``(nx, ny, nz)`` and its vectors."""
        return self._scalar_and_vectors(self._instant(self._volume()))

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

    def iso_values(self, vmax: float) -> list[float]:
        """The isosurface levels in field units."""
        if self.levels is not None:
            levels = [float(v) for v in self.levels]
        else:
            levels = [float(self.iso_level) * vmax]
        if not self.is_group:
            levels = sorted({-v for v in levels} | set(levels))
        return levels

    # ── actors ───────────────────────────────────────────────────────────

    def _hide_cut(self) -> None:
        for actor in (self.sheet_actor, self.arrow_actor):
            if actor is not None:
                actor.SetVisibility(False)

    def _hide(self) -> None:
        self._hide_cut()
        for actor in (self.volume_actor, self.iso_actor):
            if actor is not None:
                actor.SetVisibility(False)

    def apply(self, scene, shown) -> None:
        """(Re)build the sheet, the arrows and the volume for the scene's cut state."""
        vmax = self.vmax()
        cut = scene.cut
        axis = None if cut.axis is None else _viewer._AXIS_INDEX[cut.axis]
        k = None if axis is None else _layer_index(self.nodes_display, axis, cut)
        if k is None:
            self._hide_cut()
        else:
            n = self.nodes_display[axis]
            hair = _HAIR * (n[k + 1] - n[k]) * (-1.0 if cut.flip else 1.0)
            keep = None if self.frames.pec is None else ~np.take(self.frames.pec, k, axis=axis)
            scalar, vectors = self.values(axis, k)
            if not self._draw_sheet(
                scene, axis, k, cut.position + _SHEET_HAIRS * hair, keep, scalar, vmax
            ):
                self._hide_cut()
            else:
                self.sheet_actor.SetVisibility("field" in shown)
                drawn = False
                if self.has_arrows and vectors is not None:
                    drawn = self._draw_arrows(
                        scene, axis, k, cut.position + _ARROW_HAIRS * hair, keep, vectors, vmax
                    )
                if self.arrow_actor is not None:
                    self.arrow_actor.SetVisibility(drawn and "arrows" in shown)

        want_volume = "volume arrows" in shown and self.has_arrows and self.has_volume
        want_iso = "isosurface" in shown and self.can_iso
        if want_volume or want_iso:
            scalar3, vectors3 = self.volume_values()
        drawn = False
        if want_volume and vectors3 is not None:
            drawn = self._draw_volume_arrows(scene, vectors3, vmax)
        if self.volume_actor is not None:
            self.volume_actor.SetVisibility(drawn)
        drawn = False
        if want_iso:
            drawn = self._draw_isosurfaces(scene, scalar3, vmax)
        if self.iso_actor is not None:
            self.iso_actor.SetVisibility(drawn)

    _BAR_KWARGS = {
        "vertical": True,
        "n_labels": 5,
        "fmt": "%.3g",
        # Leave room for the labels: the default bar sits so far right
        # that "2.74e+03" runs off the window edge.
        "position_x": 0.84,
        "position_y": 0.08,
        "width": 0.05,
        "height": 0.45,
    }

    def _paint(self, name: str, actor, vmax: float) -> None:
        """Keep an actor's colour map and range in step with the component."""
        cmap, clim = self.colour_map, self.colour_range(vmax)
        if self._painted.get(name) == (cmap, clim):
            return
        mapper = actor.mapper
        mapper.lookup_table.cmap = cmap
        mapper.scalar_range = clim
        mapper.lookup_table.scalar_range = clim
        self._painted[name] = (cmap, clim)

    def _draw_sheet(self, scene, axis, k, position, keep, scalar, vmax) -> bool:
        """Write the layer into the sheet; ``False`` when no cell is left to show.

        The actor, its mapper and its polydata are created once and then
        updated in place.  The browser renderer keys every object by its
        identity and takes a changed object only when it is newer than
        the one it holds — a modified polydata always is, a swapped-in
        one need not be — and it keeps what it was sent, so a fresh actor
        per frame piles up until the page stalls (measured: a 50-frame
        movie froze after about 75 frames).
        """
        pl = scene.plotter
        sheet = _layer_sheet(self.nodes_display, axis, k, position, keep)
        if sheet is None:
            return False
        values = np.asarray(scalar, dtype=float).ravel(order="F")
        if keep is not None:
            values = values[np.asarray(keep, dtype=bool).ravel(order="F")]
        sheet.cell_data["field"] = values
        clim = self.colour_range(vmax)
        cmap = self.colour_map
        if self.sheet_actor is None:
            self._sheet_pd = sheet
            self.sheet_actor = pl.add_mesh(
                self._sheet_pd,
                scalars="field",
                cmap=cmap,
                clim=clim,
                opacity=self.opacity,
                lighting=False,
                show_edges=False,
                name="field_cut",
                reset_camera=False,
                render=False,
                scalar_bar_args={"title": self.bar_title, **self._BAR_KWARGS},
            )
            self._bar_title = self.bar_title
        else:
            self._sheet_pd.copy_from(sheet)
            self._sheet_pd.cell_data.active_scalars_name = "field"
        mapper = self.sheet_actor.mapper
        if self._bar_title != self.bar_title:
            # The component changed: new colours, new range, new title.
            mapper.lookup_table.cmap = cmap
            mapper.scalar_range = clim
            mapper.lookup_table.scalar_range = clim
            pl.remove_scalar_bar(self._bar_title, render=False)
            pl.add_scalar_bar(title=self.bar_title, mapper=mapper, render=False, **self._BAR_KWARGS)
            self._bar_title = self.bar_title
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray("field")
        # The mapper is fed through a small pipeline whose output stays
        # stale until it executes; run it now so that any reader of the
        # mapper's input — the browser serialiser, a test — sees the
        # current sheet without a render in between.
        mapper.Update()
        return True

    def _glyphs(self, points, vec, mag, spacing, vmax):
        """Arrow glyphs at *points*: coloured by magnitude, length scaled with a floor."""
        import pyvista as pv  # noqa: PLC0415

        cloud = pv.PolyData(points)
        cloud["vec"] = vec / np.maximum(mag, 1e-300)[:, None]
        cloud["mag"] = np.minimum(mag, vmax)
        cloud["len"] = spacing * np.maximum(np.minimum(mag, vmax) / vmax, _ARROW_FLOOR)
        glyphs = cloud.glyph(orient="vec", scale="len", factor=1.0, geom=self._glyph_geometry())
        return glyphs if glyphs.n_cells else None

    def _glyph_geometry(self):
        """The unit glyph, centred on its sample point and pointing along +x."""
        import pyvista as pv  # noqa: PLC0415

        w = float(self.glyph_width)
        if self.glyph == "cone":
            return pv.Cone(
                center=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), height=1.0, radius=0.12 * w
            )
        return pv.Arrow(
            start=(-0.5, 0.0, 0.0),
            tip_length=0.3,
            tip_radius=0.1 * w,
            shaft_radius=0.035 * w,
            tip_resolution=8,
            shaft_resolution=8,
        )

    def _place_arrows(self, scene, name: str, glyphs, vmax: float):
        """Create or refresh a glyph actor; returns the actor."""
        pl = scene.plotter
        is_volume = name == "field_volume_arrows"
        actor = self.volume_actor if is_volume else self.arrow_actor
        if actor is None:
            kwargs = {"name": name, "reset_camera": False, "render": False}
            if self.arrow_color is None:
                kwargs.update(
                    scalars="mag",
                    cmap=self.colour_map,
                    clim=self.colour_range(vmax),
                    show_scalar_bar=False,
                )
            else:
                kwargs["color"] = self.arrow_color
            actor = pl.add_mesh(glyphs, **kwargs)
            if is_volume:
                self._volume_pd, self.volume_actor = glyphs, actor
            else:
                self._arrow_pd, self.arrow_actor = glyphs, actor
            self._painted[name] = (self.colour_map, self.colour_range(vmax))
        else:
            (self._volume_pd if is_volume else self._arrow_pd).copy_from(glyphs)
            if self.arrow_color is None:
                self._paint(name, actor, vmax)
        if self.arrow_color is None:
            actor.mapper.SetScalarModeToUsePointFieldData()
            actor.mapper.SelectColorArray("mag")
        actor.mapper.Update()
        return actor

    def _draw_arrows(self, scene, axis, k, position, keep, vectors, vmax) -> bool:
        """Write the layer's arrows into the glyph polydata; ``False`` when none."""
        from magnelio.post.plot_field import _arrow_grid, _resample  # noqa: PLC0415

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
            return False
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
        glyphs = self._glyphs(points, vec, mag[mask], spacing, vmax)
        if glyphs is None:
            return False
        self._place_arrows(scene, "field_arrows", glyphs, vmax)
        return True

    def _draw_volume_arrows(self, scene, vectors, vmax) -> bool:
        """Arrows on a 3D lattice over the kept half of the region; ``False`` when none."""
        centres = tuple(0.5 * (n[:-1] + n[1:]) for n in self.nodes_display)
        raster = _lattice3(centres, self.density)
        comps = self.comps()
        arrays = [vectors[c] for c in comps]
        keep = None if self.frames.pec is None else ~self.frames.pec
        (ax_, ay_, az_), live = _resample3(centres, raster, arrays, keep)
        mag = np.sqrt(ax_**2 + ay_**2 + az_**2)
        mask = live & np.isfinite(mag) & (mag >= self.threshold * vmax)
        xx, yy, zz = np.meshgrid(*raster, indexing="ij")
        cut = scene.cut
        if cut.axis is not None:
            coord = (xx, yy, zz)[_viewer._AXIS_INDEX[cut.axis]]
            mask &= (coord >= cut.position) if cut.flip else (coord <= cut.position)
        if not np.any(mask):
            return False
        points = np.stack([xx[mask], yy[mask], zz[mask]], axis=1)
        vec = np.stack([ax_[mask], ay_[mask], az_[mask]], axis=1)
        steps = [float(r[1] - r[0]) for r in raster if r.size > 1]
        spacing = min(steps) if steps else 0.1 * _viewer._diag(scene.bounds)
        glyphs = self._glyphs(points, vec, mag[mask], spacing, vmax)
        if glyphs is None:
            return False
        self._place_arrows(scene, "field_volume_arrows", glyphs, vmax)
        return True

    def _draw_isosurfaces(self, scene, scalar, vmax) -> bool:
        """Contour the region's scalar and clip it to the kept half; ``False`` when empty."""
        import pyvista as pv  # noqa: PLC0415

        pl = scene.plotter
        values = np.asarray(scalar, dtype=float)
        if self.frames.pec is not None:
            # No field inside a conductor: the surface closes on the metal.
            values = np.where(self.frames.pec, 0.0, values)
        if self._iso_grid is None:
            self._iso_grid = pv.RectilinearGrid(*self.nodes_display)
        self._iso_grid.cell_data["field"] = values.ravel(order="F")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points = self._iso_grid.cell_data_to_point_data()
            contour = points.contour(isosurfaces=self.iso_values(vmax), scalars="field")
        if contour.n_cells == 0:
            return False
        clipped = _viewer._clip_body(contour, scene.cut, closed=False)
        if clipped is None or clipped.n_cells == 0:
            return False
        clim, cmap = self.colour_range(vmax), self.colour_map
        if self.iso_actor is None:
            self._iso_pd = clipped
            self.iso_actor = pl.add_mesh(
                self._iso_pd,
                scalars="field",
                cmap=cmap,
                clim=clim,
                opacity=_ISO_OPACITY,
                # No ``smooth_shading``: PyVista would hand the mapper a
                # copy with normals, and the view's polydata — written
                # into on every frame — would no longer be what is drawn
                # (the contour filter computes normals itself).
                show_scalar_bar=False,
                name="field_iso",
                reset_camera=False,
                render=False,
            )
            self._painted["field_iso"] = (cmap, clim)
        else:
            self._iso_pd.copy_from(clipped)
            self._paint("field_iso", self.iso_actor, vmax)
        mapper = self.iso_actor.mapper
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("field")
        mapper.Update()
        return True

    # ── notebook controls ────────────────────────────────────────────────

    def next_frame(self) -> int:
        """The frame after the current one, wrapping around."""
        return (self.frame + 1) % max(self.frames.n_frames, 1)

    def attach_controls(self, server, key: str, refresh) -> Callable[[], None]:
        """Register the frame / phase / component / play / level / density handlers.

        Returns the builder of the field controls, which draws them as a
        second row under the viewer's toolbar.
        """
        import asyncio  # noqa: PLC0415

        from trame.widgets import client, html  # noqa: PLC0415
        from trame.widgets import vuetify3 as vuetify  # noqa: PLC0415

        state = server.state
        k_frame, k_phase, k_comp = f"{key}_frame", f"{key}_phase", f"{key}_comp"
        k_label, k_play, k_play_phase = f"{key}_frame_label", f"{key}_play", f"{key}_play_phase"
        k_level, k_density = f"{key}_level", f"{key}_density"
        state[k_frame] = int(self.frame)
        state[k_phase] = float(self.phase)
        state[k_comp] = self.component
        state[k_label] = self.frame_label()
        state[k_play] = False
        state[k_play_phase] = False
        state[k_level] = int(round(100.0 * self.iso_level))
        state[k_density] = int(self.density)
        state[f"{key}_comps"] = _available_components(self.frames.components)
        n_frames = self.frames.n_frames
        label_chars = self.label_chars()

        @state.change(k_frame)
        def _on_frame(**kwargs):
            fr = int(kwargs[k_frame])
            if fr == self.frame or not (0 <= fr < n_frames):
                return
            self.frame = fr
            with state:
                state[k_label] = self.frame_label()
            refresh()

        def paced(advance):
            # A loop on the server's event loop: each step goes through
            # the slider's own handler, so the picture follows.  Paced
            # on the clock, so a slow frame shortens the pause instead
            # of piling up behind the websocket.
            async def run() -> None:
                period = 1.0 / max(float(self.fps), 0.1)
                loop = asyncio.get_running_loop()
                try:
                    while True:
                        t0 = loop.time()
                        with state:
                            advance()
                        await asyncio.sleep(max(period - (loop.time() - t0), 0.02))
                except asyncio.CancelledError:  # pragma: no cover - stop button
                    pass

            return run

        def advance_frame() -> None:
            state[k_frame] = self.next_frame()

        def advance_phase() -> None:
            state[k_phase] = (float(state[k_phase]) + _PHASE_STEP) % 360.0

        def toggle(attr: str, playing: bool, advance, other: str, k_other: str) -> None:
            # The two play buttons exclude each other: starting one stops
            # the other, so the picture never runs two clocks.
            task = getattr(self, attr)
            if playing and task is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No event loop (a script, a test): nothing can drive
                    # the slider; put the button back.
                    with state:
                        state[k_play if attr == "_play_task" else k_play_phase] = False
                    return
                if getattr(self, other) is not None:
                    with state:
                        state[k_other] = False
                setattr(self, attr, loop.create_task(paced(advance)()))
            elif not playing and task is not None:
                task.cancel()
                setattr(self, attr, None)

        @state.change(k_play)
        def _on_play(**kwargs):
            toggle("_play_task", bool(kwargs[k_play]), advance_frame, "_phase_task", k_play_phase)

        @state.change(k_play_phase)
        def _on_play_phase(**kwargs):
            toggle("_phase_task", bool(kwargs[k_play_phase]), advance_phase, "_play_task", k_play)

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

        @state.change(k_level)
        def _on_level(**kwargs):
            level = float(kwargs[k_level]) / 100.0
            if not (0.0 < level < 1.0) or level == self.iso_level:
                return
            self.iso_level = level
            refresh()

        @state.change(k_density)
        def _on_density(**kwargs):
            density = int(kwargs[k_density])
            if density < 2 or density == self.density:
                return
            self.density = density
            refresh()

        def play_button(k_flag: str, tooltip: str) -> None:
            with vuetify.VBtn(
                icon=True,
                size="small",
                variant="text",
                click=f"{k_flag} = !{k_flag}",
                style="margin-left: 8px;",
            ):
                vuetify.VIcon("mdi-play", v_if=(f"!{k_flag}",))
                vuetify.VIcon("mdi-pause", v_else=True)
                vuetify.VTooltip(tooltip, activator="parent", location="bottom")

        def readout(text: str, chars: int) -> None:
            # A fixed-width readout: the toolbar must not reflow while
            # the frames play.
            html.Span(
                text,
                style=f"margin-left: 6px; white-space: nowrap; min-width: {chars}ch; "
                "font-variant-numeric: tabular-nums;",
            )

        def items() -> None:
            # A second row under the viewer's toolbar.  PyVista's menu is
            # a card of fixed height whose rows do not wrap; the card is
            # told to grow and its row to wrap where this row is present.
            client.Style(
                ".v-card:has(.mio-field-row) { height: auto !important; "
                "overflow: visible !important; }\n"
                ".v-card:has(.mio-field-row) > .v-row { align-items: flex-start !important; }\n"
                ".v-card:has(.mio-field-row) > .v-row > .v-row { flex-wrap: wrap !important; }\n"
            )
            with html.Div(
                classes="mio-field-row",
                style="flex-basis: 100%; display: flex; align-items: center; "
                "flex-wrap: nowrap; padding: 2px 4px 2px 0;",
            ):
                if n_frames > 1:
                    play_button(k_play, "Play / pause the frames")
                    vuetify.VSlider(
                        v_model=(k_frame, state[k_frame]),
                        min=0,
                        max=n_frames - 1,
                        step=1,
                        hide_details=True,
                        density="compact",
                        style="width: 160px; margin-left: 4px;",
                    )
                    readout(f"{{{{ {k_label} }}}}", label_chars)
                if self.frames.is_complex:
                    play_button(k_play_phase, "Play / pause the phase")
                    vuetify.VSlider(
                        v_model=(k_phase, state[k_phase]),
                        min=0,
                        max=360,
                        step=5,
                        hide_details=True,
                        density="compact",
                        style="width: 120px; margin-left: 4px;",
                    )
                    readout(f"phase = {{{{ Math.round({k_phase}) }}}}°", 12)
                vuetify.VSelect(
                    v_model=(k_comp, state[k_comp]),
                    items=(f"{key}_comps", state[f"{key}_comps"]),
                    label="Field",
                    density="compact",
                    hide_details=True,
                    variant="plain",
                    style="width: 80px; margin-left: 8px;",
                )
                if self.can_iso and self.levels is None:
                    vuetify.VSlider(
                        v_model=(k_level, state[k_level]),
                        min=5,
                        max=95,
                        step=5,
                        hide_details=True,
                        density="compact",
                        style="width: 110px; margin-left: 12px;",
                    )
                    readout(f"iso {{{{ {k_level} }}}} %", 8)
                if self.has_arrows:
                    vuetify.VSlider(
                        v_model=(k_density, state[k_density]),
                        min=5,
                        max=40,
                        step=1,
                        hide_details=True,
                        density="compact",
                        style="width: 110px; margin-left: 12px;",
                    )
                    readout(f"{{{{ {k_density} }}}} arrows", 9)

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
    arrow_color: str | None = None,
    fps: float = 4.0,
    volume: str | None = None,
    levels=None,
    iso_level: float = 0.5,
    glyph: str = "arrow",
    glyph_width: float = 1.0,
    mirror: bool = True,
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
    over that layer.  The volume behind the cut can carry the field too
    (*volume*): arrows on a 3D lattice, coloured by magnitude, and
    isosurfaces of the magnitude (the ±level of a signed component),
    both clipped to the kept half like the solids; the *Show* menu turns
    either on and off.  Moving the position slider walks the cut through
    the recorded volume; a time or frequency monitor adds a frame slider,
    an eigenmode result a mode slider, complex data a phase slider, a
    selector switches the field, and sliders set the isosurface level
    and the arrow density.  The values are the cell-centred physical
    fields of the exposed layer, computed for that layer when the cut
    moves — and of the whole region when a volume representation is
    shown.  With a mesh that declares symmetry planes the field is
    continued across them, so the picture is the whole model.

    Parameters
    ----------
    source : FieldState, monitor, EigenmodeResult, or a project's monitor reader
        What to show.  A monitor must have recorded (or be read from a
        project); a :class:`~magnelio.fields.FieldState` is one frame; an
        eigenmode result's modes are its frames.
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
        Arrows along the longer in-plane axis of the cut, and along the
        longest axis of the region in the volume; the other axes get
        the count that keeps the lattice even.
    threshold : float, default 0.02
        Arrows below this fraction of *vmax* are not drawn.
    opacity : float, default 1.0
        Opacity of the field sheet.
    arrow_color : str, optional
        One colour for every arrow.  Default: the arrows are coloured by
        their magnitude on the sheet's colour scale.  Either way an
        arrow's length grows with its magnitude, from three tenths of
        the lattice spacing up to one spacing, so a decaying field keeps
        readable arrows.
    fps : float, default 4.0
        Frames per second of the toolbar's play button (notebook widget);
        the effective rate is bounded by how fast the browser receives a
        frame — a volume representation costs the whole region per frame.
    volume : {"arrows", "isosurface", "both"}, optional
        What to draw in the volume behind the cut at first.  ``"arrows"``
        replaces the arrows on the cut by arrows on a 3D lattice over the
        kept half; ``"isosurface"`` adds translucent surfaces of the
        magnitude at *iso_level* (or at *levels*); ``"both"`` draws
        both.  Default: nothing in the volume — the *Show* menu offers
        both representations whenever the source is a volume.
    levels : sequence of float, optional
        Isosurface levels in field units (V/m or A/m).  Default: one
        surface at *iso_level* of the colour ceiling, movable with the
        toolbar's level slider; given levels are fixed.  A signed
        component gets each level with both signs.
    iso_level : float, default 0.5
        The isosurface level as a fraction of *vmax* when *levels* is
        not given.
    glyph : {"arrow", "cone"}
        The shape of the field vectors, centred on their sample points.
    glyph_width : float, default 1.0
        Thickness of the vectors relative to the default.
    mirror : bool, default True
        Continue the field across the model's symmetry planes, so the
        view shows the whole model (as every other field picture does).
        Needs *mesh* (an eigenmode result brings its own), whose
        boundary declaration names the planes; a region that stops
        short of a plane is not mirrored across it.  ``False`` shows the
        modelled part only.
    geometry : GeometryModel, optional
        Draw the model's solids and features with the field.
    mesh : Mesh, optional
        The mesh the field was computed on.  Cells buried in a perfect
        conductor are cut out of the sheet, so the solids' cut faces
        show through where the field is not defined; with ``show_grid``
        the grid cells are drawn on the cut as well; and the symmetry
        planes are read from it (*mirror*).
    show_ports, show_wires, show_labels : bool, default True
        As in the geometry viewer.
    show_grid : bool, default False
        With *mesh*: draw the grid cells on the cut under the field.
    mode, size, quality, scale_mm, camera
        As in :func:`~magnelio.plots.show_geometry`; *size* sets the
        widget's height in the notebook, the toolbar's pop-out button
        opens the same view in a browser tab of its own.

    Returns
    -------
    pyvista.Plotter or None
        The plotter when ``mode="none"``; otherwise the view is displayed
        as a side effect.

    Notes
    -----
    **Controls** (notebook widget).  The first toolbar row is the
    geometry viewer's — camera buttons (reset, isometric, along x/y/z,
    parallel or perspective projection, screenshot, pop-out, help),
    *Cut* / position / *Flip* / undo / reset, and the *Show* menu with
    *Field on cut*, *Vectors on cut*, *Field vectors* (in the volume)
    and *Isosurfaces* beside the geometry's groups.  The second row
    holds the field: play and frame slider with the frame's time,
    frequency or mode; play and phase slider for complex data; the
    *Field* selector; the isosurface level and the arrow density.  The
    mouse in the browser: left drag orbits, middle drag (or shift +
    left) pans, right drag or the wheel zooms, ctrl + left rolls; the
    help button lists the same.

    Every sample is a cell-centre average of the staggered components
    — the picture stands for a layer of cells, not for a plane — and
    the isosurfaces interpolate those cell values to the nodes before
    they are contoured.
    """
    if glyph not in _GLYPH_CHOICES:
        raise ValueError(f"glyph must be one of {_GLYPH_CHOICES}; got {glyph!r}")
    if not (float(glyph_width) > 0.0):
        raise ValueError(f"glyph_width must be positive; got {glyph_width!r}")
    frames = _frames_of(source, mesh, bool(mirror))
    if plot_type not in _PLOT_TYPES:
        raise ValueError(f"plot_type must be one of {_PLOT_TYPES}; got {plot_type!r}")
    if volume is not None and volume not in _VOLUME_CHOICES:
        raise ValueError(f"volume must be one of {_VOLUME_CHOICES} or None; got {volume!r}")
    if levels is not None:
        levels = tuple(float(v) for v in np.atleast_1d(np.asarray(levels, dtype=float)))
        if not levels or any(not np.isfinite(v) or v <= 0.0 for v in levels):
            raise ValueError(f"levels must be positive field values; got {levels}")
    if not (0.0 < float(iso_level) < 1.0):
        raise ValueError(f"iso_level must lie in (0, 1); got {iso_level!r}")
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
        fps=float(fps),
        volume_start=volume,
        iso_level=float(iso_level),
        levels=levels,
        glyph=glyph,
        glyph_width=float(glyph_width),
    )
    if volume is not None and not view.has_volume:
        raise ValueError(f"{frames.name!r} offers no volume to draw {volume!r} in")
    if volume in ("isosurface", "both") and not view.can_iso:
        raise ValueError(
            f"isosurfaces need at least two cells along every axis; {frames.name!r} "
            f"spans {frames.shape} cells"
        )
    extent = []
    for nodes in view.nodes_display:
        extent += [float(nodes[0]), float(nodes[-1])]

    notebook, mode, off_screen = _viewer._resolve_mode(mode)
    if mesh is None and frames.kind == "mode":
        mesh = source.mesh
    if frames.mirrored:
        # The frames span the whole model now; the mesh does not.
        mesh = None
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
