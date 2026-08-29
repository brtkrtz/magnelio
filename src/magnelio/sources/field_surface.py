"""SourceFieldSurface — a recorded Huygens surface replayed as a source.

The equivalence principle in its operational form: the tangential E and
H a :class:`~magnelio.monitors.MonitorFieldSurface` recorded on a closed
box stand for everything that box enclosed.  Replayed on a box in a
second model, they radiate that same field outwards and nothing
inwards — the radiator itself never has to be meshed again.

Mechanically this is the TF/SF construction of
:class:`~magnelio.sources.SourceFieldIncident` with its roles swapped.
An incident field is *total* inside the box and *scattered* outside; an
equivalent source is the reverse — scattered (that is, whatever the
second model sends back) inside, total outside.  The two differ by the
sign of every face correction, which is why this class is a subclass
and not a second implementation.

Placement: the box may be moved anywhere in the second model and turned
by a multiple of 90°.  Free angles are not available and are refused
rather than rounded — a TF/SF box is spanned by grid-node planes, so a
turned recording would meet the target grid obliquely, and the replay
would be reading a surface that has no samples where the patches sit.
"""

# Design: DD-226.

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from magnelio.sources.field_incident import SourceFieldIncident

_AXES = "xyz"


def _rotation_matrix(rotation) -> np.ndarray:
    """Signed permutation matrix for ``(axis, degrees)`` or ``None``.

    Only multiples of 90° are representable on a Cartesian grid; a free
    angle raises rather than silently snapping to the nearest quarter
    turn.
    """
    if rotation is None:
        return np.eye(3)
    try:
        axis, degrees = rotation
    except (TypeError, ValueError):
        raise ValueError(
            f"rotation must be (axis, degrees) — e.g. ('z', 90); got {rotation!r}",
        ) from None
    if axis not in _AXES:
        raise ValueError(f"rotation axis must be 'x', 'y' or 'z'; got {axis!r}")
    degrees = float(degrees)
    quarters = degrees / 90.0
    if abs(quarters - round(quarters)) > 1e-9:
        raise ValueError(
            f"rotation must be a multiple of 90°; got {degrees}°.  A TF/SF "
            f"box is spanned by grid-node planes, so a free angle would put "
            f"the recorded surface obliquely across the target grid.",
        )
    k = int(round(quarters)) % 4
    a = _AXES.index(axis)
    u, v = (i for i in range(3) if i != a)
    cos, sin = ((1, 0), (0, 1), (-1, 0), (0, -1))[k]
    r = np.zeros((3, 3))
    r[a, a] = 1.0
    r[u, u] = cos
    r[u, v] = -sin
    r[v, u] = sin
    r[v, v] = cos
    return r


@dataclass
class SourceFieldSurface(SourceFieldIncident):
    """Replay a recorded Huygens surface as an equivalent source.

    Parameters
    ----------
    recording : SurfaceRecording
        The surface a :class:`~magnelio.monitors.MonitorFieldSurface`
        recorded, from :meth:`MonitorFieldSurface.recording` or
        :meth:`from_file`.
    name : str
        Source name; an :class:`~magnelio.Excitation` names it to set
        the scale factor and the delay.
    position : tuple of float, optional
        Where the centre of the recorded box goes in this model [m].
        Defaults to the position it had in the recording.
    rotation : tuple, optional
        ``(axis, degrees)`` turn applied to the recording, e.g.
        ``("z", 90)``.  Multiples of 90° only.

    Notes
    -----
    The excitation that drives this source carries no waveform — the
    time function is the recording.  Its ``amplitude`` scales the
    replayed field (dimensionless, 1 = as recorded) and its ``delay``
    shifts it in time.

    The replayed field is only as complete as the recording: a
    recording whose box was left open at a PEC/PMC wall is valid only
    in a model that continues that wall, and one that was stopped
    before the fields had decayed ends where its samples end (the
    replay holds the last sample, it does not extrapolate).

    Memory: the recording is resampled onto this model's patches once,
    at attach time — about as much again as the recording itself.

    Examples
    --------
    >>> src = sources.SourceFieldSurface(          # doctest: +SKIP
    ...     recording=rec, name="antenna", position=(0.0, 0.0, 0.2)
    ... )
    """

    recording: object = None
    position: tuple | None = None
    rotation: tuple | None = None

    _needs_field = False
    has_waveform = False
    amplitude_unit = "1"

    _rot: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _rot_inv: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _shift: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    # Per patch record: the resampled series and its time base flag.
    _series_E: tuple = dc_field(default=(), repr=False, init=False)
    _series_H: tuple = dc_field(default=(), repr=False, init=False)

    def __post_init__(self) -> None:
        if self.recording is None:
            raise TypeError(
                f"{type(self).__name__} needs recording=, a SurfaceRecording "
                f"from MonitorFieldSurface.recording() or "
                f"SourceFieldSurface.from_file(...)",
            )
        if not hasattr(self.recording, "faces"):
            raise TypeError(
                f"recording must be a magnelio.fields.SurfaceRecording; "
                f"got {type(self.recording).__name__}",
            )
        self._rot = _rotation_matrix(self.rotation)
        self._rot_inv = self._rot.T
        centre = np.asarray(self.recording.centre, dtype=float)
        if self.position is None:
            target = centre
        else:
            target = np.asarray(self.position, dtype=float).reshape(3)
            if not np.all(np.isfinite(target)):
                raise ValueError(
                    f"position must be three finite coordinates; got {self.position!r}"
                )
        self._shift = target
        if self.corners is None:
            half = 0.5 * np.abs(self._rot @ np.asarray(self.recording.size, dtype=float))
            self.corners = (
                tuple(target - half),
                tuple(target + half),
            )
        super().__post_init__()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path, **kwargs):
        """Build the source from a recording file.

        *path* is what :meth:`SurfaceRecording.save` wrote in the
        recording run; the two models share nothing else.

        Examples
        --------
        >>> src = sources.SourceFieldSurface.from_file(   # doctest: +SKIP
        ...     "antenna.h5", name="antenna", position=(0, 0, 0.2)
        ... )
        """
        from magnelio.fields.surface import SurfaceRecording  # noqa: PLC0415

        return cls(recording=SurfaceRecording.load(path), **kwargs)

    # ------------------------------------------------------------------
    # Excitation binding — a recording carries its own time function
    # ------------------------------------------------------------------

    def set_excitation(self, waveform=None, *, amplitude: float = 1.0, delay: float = 0.0) -> None:
        """Bind the scale factor and delay this source replays with.

        Takes no waveform: the time function is the recording itself.
        """
        if waveform is not None:
            raise TypeError(
                f"source {self.name!r} replays a recorded surface and has no "
                f"waveform — give the excitation an amplitude (scale factor) "
                f"and a delay only.",
            )
        amplitude = float(amplitude)
        delay = float(delay)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite; got {amplitude!r}")
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay must be a non-negative finite time [s]; got {delay!r}")
        self._waveform = None
        self._amplitude = amplitude
        self._delay = delay

    # ------------------------------------------------------------------
    # Attach
    # ------------------------------------------------------------------

    def attach(self, solver) -> None:
        """Snap the box, build the face corrections and resample the recording."""
        self._attach_grid(solver)
        patches_E, patches_H = self._build_patches()
        # An equivalent source is the incident construction with its
        # regions exchanged: total field outside the box, scattered
        # inside.  Every face correction changes sign with that swap.
        self._patches_E = tuple(self._negate(r) for r in patches_E)
        self._patches_H = tuple(self._negate(r) for r in patches_H)
        self._series_E = tuple(self._resample(r) for r in self._patches_E)
        self._series_H = tuple(self._resample(r) for r in self._patches_H)
        self._attached = True

    @staticmethod
    def _negate(record):
        comp, index, coef, fcomp, coords = record
        return (comp, index, -coef, fcomp, coords)

    def _face_of(self, coords) -> str:
        """Which box face a patch's sample points lie on.

        The patch is flat: exactly one axis is constant across it, and
        its value identifies the low or high face of that axis.  The
        Yee stagger can put the samples half a cell outside the node
        plane, so the nearer of the two bounds decides.
        """
        lo = np.asarray([c[0] for c in self._box_bounds()], dtype=float)
        hi = np.asarray([c[1] for c in self._box_bounds()], dtype=float)
        best = None
        for axis in range(3):
            values = np.asarray(coords[axis], dtype=float)
            if values.size > 1 and float(np.ptp(values)) > 0.0:
                continue
            v = float(values.flat[0])
            for side, bound in (("min", lo[axis]), ("max", hi[axis])):
                d = abs(v - bound)
                if best is None or d < best[0]:
                    best = (d, f"{_AXES[axis]}{side}")
        if best is None:
            raise ValueError(
                f"source {self.name!r}: a face correction is not flat — the "
                f"TF/SF box could not be matched to the recording.",
            )
        return best[1]

    def _box_bounds(self) -> list:
        """The snapped TF/SF box as ``[(x0, x1), (y0, y1), (z0, z1)]`` [m]."""
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        g = self._grid
        return [
            (float(g.x[ix0]), float(g.x[ix1])),
            (float(g.y[iy0]), float(g.y[iy1])),
            (float(g.z[iz0]), float(g.z[iz1])),
        ]

    def _resample(self, record):
        """The recording on one patch: ``(series, magnetic)``.

        The whole time series is interpolated onto the patch's sample
        points once, so the time loop is left with a linear blend of
        two slices.
        """
        _comp, _index, _coef, fcomp, coords = record
        face_dst = self._face_of(coords)
        axis_dst = _AXES.index(face_dst[0])
        side = face_dst[1:]

        # Target -> recording frame.
        n_dst = np.zeros(3)
        n_dst[axis_dst] = -1.0 if side == "min" else 1.0
        n_src = self._rot_inv @ n_dst
        axis_src = int(np.argmax(np.abs(n_src)))
        face_src = f"{_AXES[axis_src]}{'min' if n_src[axis_src] < 0 else 'max'}"
        if face_src not in self.recording.faces:
            raise ValueError(
                f"source {self.name!r}: the recording has no {face_src!r} face "
                f"(recorded: {sorted(self.recording.faces)}) — it was left open "
                f"at a wall, so it can only drive a model that continues that "
                f"wall on this side.",
            )
        face = self.recording.faces[face_src]

        # Sample points of this patch, carried into the recording frame.
        pts = np.broadcast_arrays(*(np.asarray(c, dtype=float) for c in coords))
        p_dst = np.stack([p.ravel() for p in pts], axis=0)
        centre = np.asarray(self.recording.centre, dtype=float)
        p_src = self._rot_inv @ (p_dst - self._shift[:, None]) + centre[:, None]
        t1, t2 = face.tangent_axes
        u = p_src[t1].reshape(pts[0].shape)
        v = p_src[t2].reshape(pts[0].shape)
        # The patch's own normal coordinate, carried into the recording
        # frame: an E patch lands on the node plane, an H patch half a
        # cell of *this* grid outside it.
        normal = float(np.asarray(p_src[axis_src]).flat[0])

        # Which recorded component the requested target component is.
        group, comp_axis = fcomp[0], _AXES.index(fcomp[1])
        e_dst = np.zeros(3)
        e_dst[comp_axis] = 1.0
        e_src = self._rot_inv @ e_dst
        axis_rec = int(np.argmax(np.abs(e_src)))
        sign = float(np.sign(e_src[axis_rec]))
        comp_rec = f"{group}{_AXES[axis_rec]}"

        series = sign * face.resample(comp_rec, u, v, normal=normal)
        series = np.asarray(series, dtype=self._dtype)
        return (self._xp.asarray(series), group == "H")

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def _replay(self, patches, series, fields, t: float) -> None:
        """Add every face correction, reading the recording at *t*."""
        xp = self._xp
        t_query = t - self._delay
        scale = self._scalar(self._amplitude)
        for record, (data, magnetic) in zip(patches, series):
            i0, i1, w = self.recording.time_weights(t_query, magnetic=magnetic)
            i0, i1, w = int(i0), int(i1), float(w)
            wave = data[i0] if i0 == i1 else (1.0 - w) * data[i0] + w * data[i1]
            comp, index, coef, _fcomp, _coords = record
            getattr(fields, comp)[index] += coef * (scale * xp.asarray(wave))

    def inject_E(self, fields, t_E: float) -> None:
        """Apply the E-side corrections after the E update."""
        if self._attached:
            self._replay(self._patches_E, self._series_E, fields, t_E - self._dt / 2)

    def inject_H(self, fields, t_H: float) -> None:
        """Apply the H-side corrections after the H update."""
        if self._attached:
            self._replay(self._patches_H, self._series_H, fields, t_H - self._dt / 2)


__all__ = ["SourceFieldSurface"]
