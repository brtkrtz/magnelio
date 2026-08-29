"""MonitorFieldSurface — tangential fields on a Huygens box over time.

The monitor records the tangential E and H on a closed box of grid-node
planes, sampled on the box's own patch centres, as a time series.  What
it records is everything the box encloses, in the form the equivalence
principle asks for: a :class:`~magnelio.fields.SurfaceRecording` that
:class:`~magnelio.sources.SourceFieldSurface` replays as an equivalent
source in a second model.

Sampling follows the shared Huygens convention
(:mod:`magnelio.monitors._huygens`): the tangential fields come from the
cell-centre interpolation of the two cell layers adjacent to each node
plane, so all four tangential components of a face live on the same
points and the surface stays exactly closed.

Recording rate: the leapfrog step is far below the Nyquist rate of the
model's bandwidth — a 12-cells-per-wavelength grid runs its CFL step
roughly an order of magnitude finer than 1/(2·f_max).  The monitor
therefore records on its own schedule (``oversample`` samples per
period at ``f_max``) and the source interpolates in between; storing
every step would cost that same order of magnitude in disk for no
information.
"""

# Design: DD-226.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from magnelio.monitors._huygens import (
    _FACE_NAMES,
    _TANGENTIALS,
    build_faces,
    component_axis_coords,
    face_node_indices,
    image_planes_for,
    sample_component_plane,
    snap_corners,
)


@dataclass
class MonitorFieldSurface:
    """Record the tangential fields on a closed Huygens box.

    The recording is the model's radiated field in transportable form:
    hand it to :class:`~magnelio.sources.SourceFieldSurface` and a
    second model is driven by this one's radiator without meshing it
    again.

    Parameters
    ----------
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        box [m].  Omit it to take the largest box the physical domain
        allows (``margin_cells`` inside the absorber) — right for a
        model built around one radiator, while a box drawn close around
        the radiator keeps the recording small.  Faces are snapped to
        the nearest grid-node plane and clamped out of the absorber.
    interval : float, optional
        Recording interval [s].  Defaults to ``1 / (oversample ·
        f_max)`` of the mesh — the samples the bandwidth actually
        carries.  Give it explicitly only to override that.
    oversample : float, default 8.0
        Samples per period at ``f_max`` when *interval* is not given.
        Well above Nyquist on purpose: the replay interpolates the
        recording linearly in time, and that error falls as the square
        of the sample spacing, not with the sampling theorem.  Measured
        on a sphere's scattered field, the replay reaches its spatial
        floor at eight samples per period and stays 1.2 % short of it
        at four.
    name : str
        Monitor name (store key).
    margin_cells : int, default 3
        Clearance in grid cells between the box and the absorber (or
        domain edge).  At least 1, so the two-layer node-plane
        sampling never reads absorber cells.

    Notes
    -----
    The recording is held in memory until the run ends and is written
    with :meth:`SurfaceRecording.save`; size is faces × patches ×
    samples, which a box drawn close around the radiator keeps small.

    The equivalence principle needs a **closed** surface: the box must
    enclose every source of the field it stands for.  A domain face
    closed with PEC/PMC — a ground plane — leaves the box open there,
    and the recording is then valid only for a second model that
    continues that wall (the replay says so).

    Examples
    --------
    >>> from magnelio import monitors
    >>> box = monitors.MonitorFieldSurface(name="antenna_box")
    """

    corners: tuple | None = None
    interval: float | None = None
    oversample: float = 8.0
    name: str = "field_surface"
    margin_cells: int = 3

    _grid: object = field(default=None, repr=False, init=False)
    _faces: list = field(default_factory=list, repr=False, init=False)
    _image_planes: list = field(default_factory=list, repr=False, init=False)
    _open_faces: tuple = field(default=(), repr=False, init=False)
    _port_footprints: dict = field(default_factory=dict, repr=False, init=False)
    # Recorded samples: face name -> component -> list of (n1, n2) arrays.
    _samples: dict = field(default_factory=dict, repr=False, init=False)
    _times: list = field(default_factory=list, repr=False, init=False)
    _dt: float = field(default=0.0, repr=False, init=False)
    _interval: float = field(default=0.0, repr=False, init=False)
    _next_idx: int = field(default=0, repr=False, init=False)
    _bounds: tuple = field(default=(), repr=False, init=False)
    _plan: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"monitor name must be a non-empty string; got {self.name!r}")
        if self.margin_cells < 1:
            raise ValueError(
                "margin_cells must be at least 1: the node-plane sampling "
                "reads one cell outside each box face."
            )
        if self.interval is not None:
            self.interval = float(self.interval)
            if not self.interval > 0.0:
                raise ValueError(f"interval must be positive [s]; got {self.interval}")
        if not self.oversample > 2.0:
            raise ValueError(
                f"oversample must exceed 2 (the Nyquist rate of f_max); got {self.oversample}"
            )

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        """Place the box on *mesh* and start a fresh recording."""
        label = f"field-surface monitor {self.name!r}"
        grid = mesh.grid
        self._grid = grid

        lo_n, hi_n, open_faces = face_node_indices(
            mesh,
            margin_cells=self.margin_cells,
            zero_margin_faces=tuple(self._port_footprints),
            label=label,
        )
        if not open_faces:
            raise ValueError(
                f"{label}: every domain face is a wall — a fully enclosed "
                f"model radiates nothing a second model could be driven by."
            )
        for axis in range(3):
            if hi_n[axis] - lo_n[axis] < 2:
                raise ValueError(
                    f"{label}: after excluding the absorber and "
                    f"{self.margin_cells} margin cell(s), only "
                    f"{hi_n[axis] - lo_n[axis]} cell(s) remain along "
                    f"{'xyz'[axis]} — the model needs more physical volume "
                    f"around the radiator."
                )

        if self.corners is not None:
            lo_n, hi_n = snap_corners(grid, self.corners, lo_n, hi_n, label=label)
            # A wall the box now stands clear of is no longer an image
            # plane: the surface closes on that side by itself.
            n_cells = (grid.Nx, grid.Ny, grid.Nz)
            open_faces = [
                f
                for f in _FACE_NAMES
                if f in open_faces
                or (
                    lo_n["xyz".index(f[0])] > 0
                    if f.endswith("min")
                    else hi_n["xyz".index(f[0])] < n_cells["xyz".index(f[0])]
                )
            ]
        self._faces = build_faces(grid, lo_n, hi_n, open_faces)
        self._warn_if_conductor_crosses(mesh, label)
        self._image_planes = image_planes_for(mesh, open_faces)
        self._open_faces = tuple(open_faces)
        nodes = (grid.x, grid.y, grid.z)
        self._bounds = tuple((float(nodes[a][lo_n[a]]), float(nodes[a][hi_n[a]])) for a in range(3))

        if self.interval is not None:
            self._interval = self.interval
        else:
            f_max = float(getattr(mesh, "f_max", 0.0) or 0.0)
            if f_max <= 0.0:
                raise ValueError(
                    f"{label}: the mesh carries no f_max, so the recording "
                    f"rate cannot be derived — give interval= [s]."
                )
            self._interval = 1.0 / (self.oversample * f_max)

        # Sampling plan, one entry per face and tangential component:
        # where its Yee positions are and which layer(s) to take.  E is
        # needed on the node plane, H half a cell outside it — by the
        # spacing of the grid that later replays the recording, so both
        # adjacent cell-centre layers are kept.
        self._plan = {}
        for bf in self._faces:
            nn = bf.slab[bf.axis].stop - 1
            t1, t2 = bf.tangent_axes
            windows = {t1: (lo_n[t1], hi_n[t1]), t2: (lo_n[t2], hi_n[t2])}
            entries = {}
            for comp in _TANGENTIALS[bf.axis]:
                c1 = component_axis_coords(grid, comp, t1)[windows[t1][0] : windows[t1][1] + 1]
                c2 = component_axis_coords(grid, comp, t2)[windows[t2][0] : windows[t2][1] + 1]
                # Along the face normal, a tangential E sits on the node
                # plane and a tangential H on the cell centres either
                # side of it — the component's own Yee position says so.
                along = component_axis_coords(grid, comp, bf.axis)
                if comp.startswith("E"):
                    layers = (nn,)
                    normals = (float(along[nn]),)
                else:
                    layers = (nn - 1, nn)
                    normals = (float(along[nn - 1]), float(along[nn]))
                entries[comp] = (windows, layers, normals, c1, c2)
            self._plan[bf.name] = entries

        self._samples = {
            bf.name: {comp: [] for comp in _TANGENTIALS[bf.axis]} for bf in self._faces
        }
        self._times = []
        self._next_idx = 0

    def _warn_if_conductor_crosses(self, mesh, label: str) -> None:
        """Warn when a conductor cuts the box.

        The equivalence principle asks for a surface that encloses the
        sources.  A conductor running through a face carries current
        across it, and the recording then stands for only part of the
        radiator — the replay is short of exactly that part.  A ground
        plane is the licit case and is not this: it closes a domain
        face, so the box is left open there instead of cut.
        """
        material_id = getattr(mesh, "material_id", None)
        library = getattr(mesh, "material_library", None) or {}
        pec_ids = [mid for mid, mat in library.items() if getattr(mat, "is_pec", False)]
        if material_id is None or not pec_ids:
            return
        for bf in self._faces:
            nn = bf.slab[bf.axis].stop - 1
            sl = list(bf.slab)
            sl[bf.axis] = nn - 1 if bf.sign < 0 else nn
            if np.isin(material_id[tuple(sl)], pec_ids).any():
                warnings.warn(
                    f"{label}: a conductor crosses the {bf.name} face of the "
                    f"box.  The recorded surface then stands for only part "
                    f"of the radiator — move the box clear of it, or leave "
                    f"that side to a ground plane.",
                    stacklevel=3,
                )

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Sample the box faces when the schedule asks for it.

        The solver calls this after the H update of step *n*, where
        ``e`` stands at ``t + dt`` and ``h`` half a step later.  Both
        stamps are kept: the recording stores the E time base and the
        half-step offset, and the replay interpolates each field at its
        own instant.

        At most one sample per step — a schedule finer than the time
        step has no more information to record, and two samples sharing
        an instant would leave the replay dividing by a zero interval.
        """
        del n
        if self._grid is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")
        if self._interval < dt:
            warnings.warn(
                f"field-surface monitor {self.name!r}: interval "
                f"{self._interval:.3e} s is finer than the time step "
                f"{dt:.3e} s and carries no extra information; recording "
                f"every step instead.",
                stacklevel=2,
            )
            self._interval = dt
        self._dt = dt
        t_E = t + dt
        if t_E + 0.5 * dt < self._next_idx * self._interval:
            return
        for bf in self._faces:
            for comp, (windows, layers, _normals, _c1, _c2) in self._plan[bf.name].items():
                planes = [
                    sample_component_plane(fields, self._grid, comp, bf.axis, k, windows)
                    for k in layers
                ]
                vals = planes[0] if len(planes) == 1 else np.stack(planes, axis=0)
                self._samples[bf.name][comp].append(vals)
        self._times.append(float(t_E))
        self._next_idx = int(math.floor((t_E + 0.5 * dt) / self._interval)) + 1

    def finalize(self) -> None:
        """Nothing to settle — the samples are the result."""

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Checkpoint the schedule cursor for a bit-exact resume."""
        return {"next_idx": int(self._next_idx)}

    def load_state_dict(self, sd: dict) -> None:
        """Restore the schedule cursor (see :meth:`state_dict`)."""
        self._next_idx = int(sd["next_idx"])

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def t(self) -> np.ndarray:
        """Recorded sample times [s] of the E time base."""
        return np.asarray(self._times, dtype=float)

    def recording(self):
        """The recorded surface as a :class:`~magnelio.fields.SurfaceRecording`.

        Hand the result to
        :meth:`~magnelio.sources.SourceFieldSurface.from_recording` to
        drive a second model with it.
        """
        from magnelio.fields.surface import (  # noqa: PLC0415
            ComponentRecord,
            FaceRecord,
            SurfaceRecording,
        )

        if self._grid is None:
            raise RuntimeError(
                f"field-surface monitor {self.name!r} has not run yet — "
                f"attach it to an analysis and run before reading it."
            )
        faces = {}
        for bf in self._faces:
            components = {}
            for comp, (_w, _layers, normals, c1, c2) in self._plan[bf.name].items():
                arrays = self._samples[bf.name][comp]
                if not arrays:
                    continue
                components[comp] = ComponentRecord(
                    c1=np.asarray(c1, dtype=float),
                    c2=np.asarray(c2, dtype=float),
                    normals=normals,
                    values=np.stack(arrays, axis=0),
                )
            faces[bf.name] = FaceRecord(
                name=bf.name,
                axis=bf.axis,
                sign=bf.sign,
                plane=bf.plane,
                tangent_axes=bf.tangent_axes,
                components=components,
            )
        return SurfaceRecording(
            name=self.name,
            faces=faces,
            times=np.asarray(self._times, dtype=float),
            half_step=0.5 * self._dt,
            bounds=self._bounds,
            open_faces=self._open_faces,
        )

    def __repr__(self) -> str:
        n = len(self._times)
        return f"MonitorFieldSurface(name={self.name!r}, faces={len(self._faces)}, samples={n})"


__all__ = ["MonitorFieldSurface"]
