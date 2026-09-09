"""The surface current of a conductor, from a recorded magnetic field.

On a perfect conductor the tangential magnetic field at the wall *is*
the surface current density, ``J_s = n x H`` [A/m], with ``n`` the
outward normal.  The normal component of H vanishes there, so the cross
product with the full vector is the same thing as with its tangential
part, and ``|J_s| = |H_tan|``.

Magnitude and direction come from different places, and that is the
point of the design (DD-273).  The **magnitude** is taken from the
booking the wall loss already uses (``sum(weight * |H|^2)`` per patch,
DD-087): those weights compensate the deliberate displacement of the
sampling faces away from the wall, so the number is the one the loss
integral trusts, and ``(R_s/2)*integral|J_s|^2 dA`` reproduces
:class:`~magnelio.monitors.MonitorWallLoss` exactly.  The **direction**
is ``n x H`` with the weight-averaged H of the same samples and the
patch's own outward normal, which on a curved conductor is the
direction of the sub-cell wall vector rather than a staircase axis.

Taking the magnitude from a mean of samples instead would be wrong by
Jensen's inequality alone; taking it from a cell-centre average of the
neighbouring cell — the first thing one writes — puts the sample half a
cell off the wall and loses half of a thin conductor's current.

On a conductor described by a surface impedance rather than a perfect
one the same formula gives the current integrated through the skin
depth: the same amperes per metre, a different physical picture.
"""

# Design: DD-273 (the surface current as a derived view).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceCurrent:
    """Surface current density on a conductor's patches, per frame.

    Attributes
    ----------
    values : np.ndarray
        ``J_s`` [A/m], shape ``(n_frames, n_patches, 3)`` — complex for
        a spectrum's phasors, real for a time-domain frame.
    positions : np.ndarray
        Patch positions [m], shape ``(n_patches, 3)``.
    normals : np.ndarray
        Outward unit normals, shape ``(n_patches, 3)``.
    areas : np.ndarray
        Conducting area of each patch [m²], shape ``(n_patches,)``.
    tags : np.ndarray
        Material id (or boundary face name) of the conductor each patch
        belongs to.
    labels : np.ndarray
        Frame labels of the series it came from — times [s] or
        frequencies [Hz].
    conformal : bool
        Whether the normals are the sub-cell ones.
    """

    values: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    tags: np.ndarray
    labels: np.ndarray
    conformal: bool

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def area(self) -> float:
        """Total conducting area covered [m²]."""
        return float(self.areas.sum())

    def magnitude(self, frame: int | None = None) -> np.ndarray:
        """``|J_s|`` [A/m]: ``(n_patches,)`` for one frame, all frames otherwise."""
        mag = np.sqrt(np.sum(np.abs(self.values) ** 2, axis=-1))
        return mag if frame is None else mag[int(frame)]

    def vector(self, frame: int = 0) -> np.ndarray:
        """The current density vector of one frame, ``(n_patches, 3)``."""
        return self.values[int(frame)]

    def current_through(self, frame: int = 0) -> float:
        """``∮|J_s| dA`` [A·m] — the area integral of the magnitude.

        Divide by the length of the surface along the current to get
        amperes: on one cell layer of a transmission line that is the
        layer's thickness, and the result is the conductor current.
        """
        return float(np.sum(self.magnitude(frame) * self.areas))

    def power_loss(self, surface_resistance, frame: int = 0) -> float:
        """Ohmic loss ``(R_s/2)·∮|J_s|² dA`` [W] of one frame.

        The perturbative wall loss: the current a perfect conductor
        carries, dissipated in a real one of surface resistance *R_s*
        [Ω] (a scalar, or one value per patch).  A complex frame is an
        RMS phasor — per 1 W CW for a monitor's spectrum — so its time
        average carries no further half; a real frame's amplitude does.

        This is the identity :class:`~magnelio.monitors.MonitorWallLoss`
        integrates during a march, on the same booking, so the two agree
        to round-off.
        """
        mag2 = self.magnitude(frame) ** 2
        total = float(np.sum(np.asarray(surface_resistance) * mag2 * self.areas))
        half = 1.0 if np.iscomplexobj(self.values) else 0.5
        return half * total

    def select(self, tag) -> "SurfaceCurrent":
        """The patches of one conductor, by material id or face name."""
        mask = self.tags == tag
        return SurfaceCurrent(
            values=self.values[:, mask],
            positions=self.positions[mask],
            normals=self.normals[mask],
            areas=self.areas[mask],
            tags=self.tags[mask],
            labels=self.labels,
            conformal=self.conformal,
        )

    def show(self, geometry=None, *, frame: int = 0, density: int = 1, **kwargs):
        """Interactive 3D view of the current over the model.

        One arrow per wall patch, coloured by ``|J_s|`` and scaled to
        the patch size, drawn over the geometry.  A complex frame — a
        spectrum's phasor — is shown at its zero-phase instant, and its
        colour is the RMS magnitude.

        Parameters
        ----------
        geometry : GeometryModel, optional
            Drawn under the arrows; without it only the current shows.
        frame : int
            Which frame to draw.
        density : int
            Draw every n-th patch.  A fine mesh otherwise turns into a
            mat of overlapping arrows; the colour scale is unaffected.
        **kwargs
            Passed to :func:`magnelio.plots.show_geometry` — ``mesh=``,
            ``cut=``, ``mode=``, ``size=`` and the rest.

        Returns
        -------
        The viewer, as :func:`magnelio.plots.show_geometry` returns it.
        """
        from magnelio.post.plot_3d import show_geometry  # noqa: PLC0415

        return show_geometry(
            geometry,
            surface_current=self,
            current_frame=frame,
            current_density=density,
            **kwargs,
        )

    def __repr__(self) -> str:
        kind = "complex" if np.iscomplexobj(self.values) else "real"
        tags = sorted({str(t) for t in self.tags})
        return (
            f"SurfaceCurrent({len(self)} patches, {self.n_frames} frames, {kind}, "
            f"area {self.area * 1e6:.4g} mm², conductors {tags}"
            f"{'' if self.conformal else ', staircase normals'})"
        )


def _frames_of(series):
    """``(n_frames, labels, getter)`` for a series or a single state."""
    if hasattr(series, "n_frames"):
        return (
            int(series.n_frames),
            np.asarray(getattr(series, "_labels", np.zeros(series.n_frames)), dtype=float),
            series.frame,
        )
    return 1, np.zeros(1), lambda _i: series


def _warn_on_boundary_coverage(mesh, excluded) -> None:
    """Ask once whether a boundary face carries a port's cross-section.

    A conductor leaving the model through a boundary — a feed line
    ending on a port plane — has its *cross-section* in that face, and
    that is not a wall: the structure continues through it.  Booking it
    adds surface carrying no current of the model (measured on a coax:
    +9.5 % on the dissipated power, DD-273).

    Whether a given face is that, or simply where the domain ends
    inside metal, cannot be decided from the mesh: a port sits on a PEC
    face and is substituted at run time, and both cases look alike in
    the coverage.  So the question is asked once, and answering it —
    ``exclude_faces=("zmin",)`` or an explicit ``exclude_faces=()`` —
    settles it.
    """
    import warnings  # noqa: PLC0415

    fm = getattr(mesh, "face_material", None)
    if fm is None or getattr(fm, "A_face_pec_jump", None) is None:
        return
    from magnelio.mesh._surfaces import _face_pec_views  # noqa: PLC0415

    _a_pec, a_full, jumps = _face_pec_views(mesh)
    n_cells = (mesh.grid.Nx, mesh.grid.Ny, mesh.grid.Nz)
    faces = []
    for axis, name in enumerate("xyz"):
        for face, plane in ((f"{name}min", 0), (f"{name}max", n_cells[axis])):
            idx = [slice(None)] * 3
            idx[axis] = plane
            covered = np.abs(jumps[axis][tuple(idx)]) > 1e-9 * a_full[axis][tuple(idx)]
            if np.any(covered):
                faces.append(face)
    if not faces:
        return
    warnings.warn(
        f"conductor meets the domain boundary at {tuple(faces)}, and the wall booking "
        f"counts that coverage as surface.  Where a port sits, it is the feed's "
        f"cross-section instead — the structure continues through the face — and "
        f"counting it overstates the current and the loss.  Name the port planes with "
        f"exclude_faces=(...), or pass exclude_faces=() to say there are none.",
        UserWarning,
        stacklevel=4,
    )


def from_series(series, mesh, *, tag=None, exclude_faces=None, **kwargs) -> SurfaceCurrent:
    """``J_s = n × H`` on the conductor patches of *mesh*.

    Parameters
    ----------
    series : FieldRecording, FieldSpectrum or FieldState
        Must carry all three magnetic components on the mesh's own grid.
    mesh : Mesh
        The mesh whose conductor surface is meant.
    tag : optional
        Restrict to one conductor (material id, or a boundary face name
        such as ``"zmin"``).
    exclude_faces : tuple of str, optional
        Domain boundary faces whose conductor coverage is *not* a wall
        — a port plane above all, where the feed's cross-section shows
        and the structure continues through the face.  Leaving this
        unset on a model whose conductor reaches a boundary asks once,
        as a warning, which faces those are; ``()`` answers "none".
    **kwargs
        Passed to
        :func:`~magnelio.mesh._surfaces.enumerate_wall_patches` — e.g.
        ``bc_pec_faces=("zmin",)`` to include a PEC boundary wall.

    Returns
    -------
    SurfaceCurrent
    """
    from magnelio.mesh._surfaces import enumerate_wall_patches  # noqa: PLC0415

    have = tuple(getattr(series, "components", ("Hx", "Hy", "Hz")))
    missing = [c for c in ("Hx", "Hy", "Hz") if c not in have]
    if missing:
        raise KeyError(
            f"the surface current is n x H, so all three magnetic components are needed; "
            f"{missing} were not recorded (fields=['H'] on the monitor)"
        )
    grid = series.grid if hasattr(series, "grid") else series._grid
    g = mesh.grid
    if (grid.Nx, grid.Ny, grid.Nz) != (g.Nx, g.Ny, g.Nz):
        raise ValueError(
            f"the surface current needs the magnetic field on the mesh's own grid "
            f"({g.Nx}x{g.Ny}x{g.Nz} cells); this field covers {grid.Nx}x{grid.Ny}x{grid.Nz} "
            f"— record the whole domain, or use a monitor without corners"
        )
    excluded = () if exclude_faces is None else tuple(exclude_faces)
    if exclude_faces is None:
        _warn_on_boundary_coverage(mesh, excluded)
    patches = enumerate_wall_patches(mesh, masked_boundary_faces=excluded, **kwargs)
    if tag is not None:
        patches = patches.select(patches.tags == tag)
    if len(patches) == 0:
        raise ValueError(
            "the mesh holds no conductor surface in the field region"
            + ("" if tag is None else f" with tag {tag!r}")
        )

    n_frames, labels, frame_of = _frames_of(series)
    out = None
    for i in range(n_frames):
        state = frame_of(i)
        # The booking expects the solver's grid quantities (h = H·l_dual,
        # DD-085) and divides by the dual length itself — handing it the
        # physical field would multiply H by 1/l_dual a second time.
        h = [np.asarray(getattr(state._raw, c)) for c in ("Hx", "Hy", "Hz")]
        # Magnitude from the loss booking, direction from the same
        # samples: |J_s|² A is exactly the patch's loss integrand.
        mag = np.sqrt(np.maximum(patches.h_tan_sq(*h), 0.0) / np.maximum(patches.areas, 1e-300))
        direction = np.cross(patches.normals, patches.h_vectors(*h))
        scale = np.linalg.norm(direction, axis=1)
        unit = np.zeros_like(direction)
        live = scale > 0.0
        unit[live] = direction[live] / scale[live, None]
        js = unit * mag[:, None]
        if out is None:
            out = np.empty((n_frames, len(patches), 3), dtype=js.dtype)
        out[i] = js
    return SurfaceCurrent(
        values=out,
        positions=patches.centres,
        normals=patches.normals,
        areas=patches.areas,
        tags=patches.tags,
        labels=labels.reshape(-1)[:n_frames],
        conformal=patches.conformal,
    )
