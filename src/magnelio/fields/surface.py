"""SurfaceRecording — tangential fields on a closed box over time.

The transportable form of what a model radiates.
:class:`~magnelio.monitors.MonitorFieldSurface` writes one,
:class:`~magnelio.sources.SourceFieldSurface` replays it as an
equivalent source in a second model, and the store carries it between
the two runs.

The recording holds, per box face, the four tangential field
components on the face's patch centres and one time base.  E and H are
half a leapfrog step apart (``half_step``); each is interpolated at its
own instant, so the replay reproduces the stagger the recording was
taken with rather than averaging it away.
"""

# Design: DD-226.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AXES = "xyz"


def _interp_weights(
    nodes: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation indices and weight for *query* in *nodes*.

    Returns ``(i0, i1, w)`` with ``value = (1 - w)·f[i0] + w·f[i1]``.
    Queries outside the sampled range are clamped to the end value: a
    replay patch may sit half a cell beyond the recorded patch centres,
    and the recorded field is the honest limit there — extrapolating a
    wave field off the end of its sampling is not.
    """
    nodes = np.asarray(nodes, dtype=float)
    query = np.asarray(query, dtype=float)
    if nodes.size == 1:
        zero = np.zeros(query.shape, dtype=int)
        return zero, zero, np.zeros(query.shape, dtype=float)
    i1 = np.clip(np.searchsorted(nodes, query), 1, nodes.size - 1)
    i0 = i1 - 1
    span = nodes[i1] - nodes[i0]
    w = np.clip((query - nodes[i0]) / span, 0.0, 1.0)
    return i0, i1, w


@dataclass
class ComponentRecord:
    """One tangential component of one box face, on its own Yee positions.

    Attributes
    ----------
    c1, c2 : np.ndarray
        Sample coordinates [m] along the face's two tangent axes — node
        or cell-centre lines, whichever the component sits on.
    normals : tuple of float
        Normal coordinate(s) [m] of the sampled layer(s).  One for E,
        which the face corrections need on the node plane; two for H,
        which they need half a cell outside the face — by the spacing
        of the *replaying* grid, so the replay interpolates between
        these two.
    values : np.ndarray
        ``(n_t, n1, n2)`` for one layer, ``(n_t, 2, n1, n2)`` for two.
    """

    c1: np.ndarray
    c2: np.ndarray
    normals: tuple
    values: np.ndarray

    def at(self, u, v, normal: float | None = None) -> np.ndarray:
        """The whole time series at in-plane points ``(u, v)``.

        Bilinear in the face and, for a two-layer record, linear in the
        normal — exact on the recorded positions, so a replay on the
        recording's own grid reproduces it rather than smoothing it.
        """
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        f = self.values
        if len(self.normals) == 2:
            lo, hi = self.normals
            w = (
                0.0
                if (normal is None or hi == lo)
                else float(np.clip((float(normal) - lo) / (hi - lo), 0.0, 1.0))
            )
            f = f[:, 0] * (1.0 - w) + f[:, 1] * w
        i0, i1, wu = _interp_weights(self.c1, u)
        j0, j1, wv = _interp_weights(self.c2, v)
        a = f[:, i0, j0] * ((1 - wu) * (1 - wv))
        a += f[:, i1, j0] * (wu * (1 - wv))
        a += f[:, i0, j1] * ((1 - wu) * wv)
        a += f[:, i1, j1] * (wu * wv)
        return a


@dataclass
class FaceRecord:
    """One face of a :class:`SurfaceRecording`.

    Attributes
    ----------
    name : str
        Face name (``"xmin"`` … ``"zmax"``) in the recording's own frame.
    axis : int
        Axis index of the face normal (0/1/2).
    sign : float
        Outward normal component along *axis* (±1).
    plane : float
        Node-plane coordinate of the face [m].
    tangent_axes : tuple of int
        The two in-plane axis indices, in ascending order.
    components : dict[str, ComponentRecord]
        The four tangential components — E in [V/m], H in [A/m].
    """

    name: str
    axis: int
    sign: float
    plane: float
    tangent_axes: tuple
    components: dict

    def resample(self, comp: str, u, v, normal: float | None = None) -> np.ndarray:
        """The time series of *comp* at in-plane points ``(u, v)``."""
        if comp not in self.components:
            raise KeyError(
                f"face {self.name!r} carries {sorted(self.components)}, not {comp!r}",
            )
        return self.components[comp].at(u, v, normal)


@dataclass
class SurfaceRecording:
    """Tangential fields on a closed box, sampled over time.

    Attributes
    ----------
    name : str
        The recording monitor's name.
    faces : dict[str, FaceRecord]
        The recorded faces, keyed by name.
    times : np.ndarray
        Sample instants [s] of the E time base.
    half_step : float
        Time offset [s] of the H samples against *times* (half the
        leapfrog step of the recording run).
    bounds : tuple
        ``((x0, x1), (y0, y1), (z0, z1))`` — the box extent [m] in the
        recording's own frame.
    open_faces : tuple of str
        The faces that were actually recorded.  Fewer than six means
        the box was left open at a PEC/PMC wall, and the recording is
        valid only where that wall continues.

    Examples
    --------
    >>> rec = monitor.recording()             # doctest: +SKIP
    >>> rec.duration, rec.interval            # doctest: +SKIP
    """

    name: str
    faces: dict
    times: np.ndarray
    half_step: float
    bounds: tuple
    open_faces: tuple

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.half_step = float(self.half_step)

    @property
    def duration(self) -> float:
        """Recorded time span [s]."""
        return float(self.times[-1] - self.times[0]) if self.times.size > 1 else 0.0

    @property
    def interval(self) -> float:
        """Sample interval [s] (the recording is uniformly sampled)."""
        return float(self.times[1] - self.times[0]) if self.times.size > 1 else 0.0

    @property
    def closed(self) -> bool:
        """Whether all six faces were recorded."""
        return len(self.open_faces) == 6

    @property
    def centre(self) -> tuple:
        """Centre of the box [m] in the recording's own frame."""
        return tuple(0.5 * (lo + hi) for lo, hi in self.bounds)

    @property
    def size(self) -> tuple:
        """Edge lengths of the box [m]."""
        return tuple(hi - lo for lo, hi in self.bounds)

    def time_weights(self, t, *, magnetic: bool = False):
        """Interpolation indices and weight for instant(s) *t* [s].

        *magnetic* shifts the query by the recorded half step, so an H
        component is read on the time base it was sampled on.
        """
        base = self.times + self.half_step if magnetic else self.times
        return _interp_weights(base, np.asarray(t, dtype=float))

    def __repr__(self) -> str:
        n = self.times.size
        return (
            f"SurfaceRecording(name={self.name!r}, faces={len(self.faces)}, "
            f"samples={n}, duration={self.duration:.3e} s)"
        )

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Write the recording to an HDF5 file.

        The file is the exchange format between the two runs: the model
        that recorded it and the model that replays it need share
        nothing else, not even a project.

        Examples
        --------
        >>> rec.save("antenna.h5")                    # doctest: +SKIP
        """
        import h5py  # noqa: PLC0415

        with h5py.File(str(path), "w") as f:
            f.attrs["magnelio_object"] = "SurfaceRecording"
            self.to_h5(f.create_group("recording"))

    @classmethod
    def load(cls, path) -> SurfaceRecording:
        """Read a recording written by :meth:`save`."""
        import h5py  # noqa: PLC0415

        with h5py.File(str(path), "r") as f:
            if f.attrs.get("magnelio_object") != "SurfaceRecording":
                raise ValueError(
                    f"{path} does not hold a magnelio surface recording "
                    f"(written by SurfaceRecording.save)",
                )
            return cls.from_h5(f["recording"])

    def to_h5(self, group) -> None:
        """Write the recording into an open HDF5 *group*."""
        group.attrs["name"] = self.name
        group.attrs["half_step"] = float(self.half_step)
        group.attrs["open_faces"] = list(self.open_faces)
        group.create_dataset("times", data=self.times)
        group.create_dataset("bounds", data=np.asarray(self.bounds, dtype=float))
        fg = group.create_group("faces")
        for name, fr in self.faces.items():
            g = fg.create_group(name)
            g.attrs["axis"] = int(fr.axis)
            g.attrs["sign"] = float(fr.sign)
            g.attrs["plane"] = float(fr.plane)
            g.attrs["tangent_axes"] = list(int(a) for a in fr.tangent_axes)
            for comp, cr in fr.components.items():
                cg = g.create_group(comp)
                cg.attrs["normals"] = [float(v) for v in cr.normals]
                cg.create_dataset("c1", data=cr.c1)
                cg.create_dataset("c2", data=cr.c2)
                cg.create_dataset("values", data=np.asarray(cr.values, dtype=float))

    @classmethod
    def from_h5(cls, group) -> SurfaceRecording:
        """Read a recording written by :meth:`to_h5`."""
        faces = {}
        for name, g in group["faces"].items():
            faces[name] = FaceRecord(
                name=name,
                axis=int(g.attrs["axis"]),
                sign=float(g.attrs["sign"]),
                plane=float(g.attrs["plane"]),
                tangent_axes=tuple(int(a) for a in g.attrs["tangent_axes"]),
                components={
                    comp: ComponentRecord(
                        c1=np.asarray(cg["c1"]),
                        c2=np.asarray(cg["c2"]),
                        normals=tuple(float(v) for v in cg.attrs["normals"]),
                        values=np.asarray(cg["values"]),
                    )
                    for comp, cg in g.items()
                },
            )
        bounds = tuple(tuple(float(v) for v in row) for row in np.asarray(group["bounds"]))
        return cls(
            name=str(group.attrs["name"]),
            faces=faces,
            times=np.asarray(group["times"]),
            half_step=float(group.attrs["half_step"]),
            bounds=bounds,
            open_faces=tuple(str(f) for f in group.attrs["open_faces"]),
        )


__all__ = ["ComponentRecord", "FaceRecord", "SurfaceRecording"]
