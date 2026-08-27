"""Near-to-far-field transform on a closed Huygens surface.

The frequency-domain surface-equivalence transform: tangential fields
on a closed surface become equivalent currents ``J = n × H`` and
``M = −n × E``, whose radiation vectors

.. math::

    \\mathbf N = \\oint \\mathbf J\\, e^{+jk\\,\\hat r\\cdot r'}\\,dS',
    \\qquad
    \\mathbf L = \\oint \\mathbf M\\, e^{+jk\\,\\hat r\\cdot r'}\\,dS'

give the far-zone field per direction.  Boundary faces the surface
cannot cross — a ground plane, a symmetry plane — enter through image
theory: every surface patch is mirrored across each such plane with
the field-component signs of the shared symmetry table
(:func:`magnelio.post._symmetry.mirror_sign`), which *is* the
image-current sign table.

Convention note: the textbook formulas above assume the
``e^{+j\\omega t}`` time convention, while the solver's running DFT
accumulates ``\\sum F(t)\\,e^{+j\\omega t}\\,dt`` — phasors of the
``e^{-j\\omega t}`` convention.  The transform therefore conjugates its
inputs once at the entrance, applies the textbook formulas verbatim,
and conjugates the result back, so the returned complex pattern is a
phasor in the same convention as every other frequency-domain quantity
of the library.
"""

# Design: DD-173 (far-field monitor and NTFF transform; spherical
# convention, image theory, symmetry composition).

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from magnelio.constants import C0, EPS0, MU0
from magnelio.post._symmetry import mirror_sign

_ETA0 = float(np.sqrt(MU0 / EPS0))
_DIR_CHUNK = 256


@dataclass
class SurfacePatchSet:
    """Sampled tangential fields on one part of a Huygens surface.

    All arrays share the leading patch count ``n``.  Produced by
    ``MonitorFarField`` (one set per box face) or assembled directly
    from analytic fields in tests.

    Attributes
    ----------
    centers : (n, 3) float
        Patch centre positions [m].
    normals : (n, 3) float
        Outward unit normals.
    areas : (n,) float
        Patch areas [m²].
    E, H : (n, 3) complex
        Tangential (or full — only the tangential part contributes)
        E [V/m] and H [A/m] phasors at one frequency, in the library's
        ``e^{-jωt}`` convention.
    """

    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    E: np.ndarray
    H: np.ndarray


@dataclass(frozen=True)
class ImagePlane:
    """A boundary plane handled by image theory.

    Attributes
    ----------
    axis : int
        World axis index (0/1/2) of the plane normal.
    position : float
        Plane coordinate [m] along that axis.
    kind : str
        ``"PEC"`` or ``"PMC"`` — selects the image-current signs.
    at_low : bool
        True when the physical domain lies on the high side of the
        plane (a ``*min`` face).
    physical_halfspace : bool
        True for a real boundary (ground plane): only the domain side
        of the plane is physical and the pattern is masked beyond it.
        False for a symmetry plane: the mirror half exists physically
        and the full sphere is meaningful.
    """

    axis: int
    position: float
    kind: str
    at_low: bool
    physical_halfspace: bool


def _mirror_patches(patches: SurfacePatchSet, plane: ImagePlane) -> SurfacePatchSet:
    """The image copy of *patches* across *plane* (image theory)."""
    centers = patches.centers.copy()
    centers[:, plane.axis] = 2.0 * plane.position - centers[:, plane.axis]
    normals = patches.normals.copy()
    normals[:, plane.axis] = -normals[:, plane.axis]
    E = patches.E.copy()
    H = patches.H.copy()
    for c in range(3):
        E[:, c] *= mirror_sign("E", c, plane.axis, plane.kind)
        H[:, c] *= mirror_sign("H", c, plane.axis, plane.kind)
    return SurfacePatchSet(
        centers=centers,
        normals=normals,
        areas=patches.areas.copy(),
        E=E,
        H=H,
    )


def _concat(sets: Sequence[SurfacePatchSet]) -> SurfacePatchSet:
    return SurfacePatchSet(
        centers=np.concatenate([s.centers for s in sets], axis=0),
        normals=np.concatenate([s.normals for s in sets], axis=0),
        areas=np.concatenate([s.areas for s in sets], axis=0),
        E=np.concatenate([s.E for s in sets], axis=0),
        H=np.concatenate([s.H for s in sets], axis=0),
    )


def surface_power(patches: Sequence[SurfacePatchSet]) -> float:
    """Real power [W] leaving the Huygens surface, from the same samples.

    ``Re ∮ (E × H*) · n̂ dS`` over the patches — the Poynting flux the
    surface fields carry outward, in the library's effective-phasor
    units (no ½).  For a lossless exterior this equals the radiated
    power exactly, so it is the natural closure check for the
    transform, independent of any port bookkeeping.
    """
    total = 0.0
    for s in patches:
        S = np.real(np.cross(s.E, np.conj(s.H)))
        total += float((np.einsum("pc,pc->p", S, s.normals) * s.areas).sum())
    return total


def default_angular_grid() -> tuple[np.ndarray, np.ndarray]:
    """The default (θ, φ) evaluation grid: 2° spacing, ISO convention."""
    return np.linspace(0.0, np.pi, 91), np.linspace(0.0, 2.0 * np.pi, 181)


def ntff_transform(
    patches: Sequence[SurfacePatchSet],
    image_planes: Sequence[ImagePlane],
    f: float,
    theta: Optional[np.ndarray] = None,
    phi: Optional[np.ndarray] = None,
    accepted_power: Optional[float] = None,
    surface_power: Optional[float] = None,
) -> "FarFieldResult":
    """Far-field pattern from tangential surface fields at one frequency.

    Parameters
    ----------
    patches : sequence of SurfacePatchSet
        The Huygens surface, in as many pieces as convenient.
    image_planes : sequence of ImagePlane
        Boundary planes the surface does not cross; each doubles the
        patch set with its mirror image.
    f : float
        Frequency [Hz].
    theta, phi : array_like, optional
        Spherical evaluation angles [rad]; θ from the +z axis, φ from
        the +x axis in the xy-plane.  Default: 2° grids over the full
        sphere.
    accepted_power : float, optional
        Accepted input power [W] for the gain normalisation; usually
        wired by the analysis.
    surface_power : float, optional
        Real power leaving the surface, :func:`surface_power` of the
        same patches; carried on the result for the closure check.

    Returns
    -------
    FarFieldResult
    """
    if theta is None or phi is None:
        th_d, ph_d = default_angular_grid()
        theta = th_d if theta is None else np.asarray(theta, dtype=float)
        phi = ph_d if phi is None else np.asarray(phi, dtype=float)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))

    surf = _concat(list(patches))
    # Library phasors are e^{-jωt}; the textbook algebra below is
    # e^{+jωt}.  Conjugate in, conjugate out.
    surf = SurfacePatchSet(
        centers=surf.centers,
        normals=surf.normals,
        areas=surf.areas,
        E=np.conj(surf.E),
        H=np.conj(surf.H),
    )
    expanded = [surf]
    for plane in image_planes:
        expanded += [_mirror_patches(s, plane) for s in expanded]
    surf = _concat(expanded) if len(expanded) > 1 else expanded[0]

    J = np.cross(surf.normals, surf.H)
    M = -np.cross(surf.normals, surf.E)
    wJ = J * surf.areas[:, None]
    wM = M * surf.areas[:, None]

    k = 2.0 * np.pi * float(f) / C0
    n_th, n_ph = theta.size, phi.size
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    # Direction grid (θ outer, φ inner), flattened to (n_dirs, 3).
    r_hat = np.stack(
        [
            np.outer(st, cp).ravel(),
            np.outer(st, sp).ravel(),
            np.outer(ct, np.ones_like(sp)).ravel(),
        ],
        axis=1,
    )
    th_hat = np.stack(
        [
            np.outer(ct, cp).ravel(),
            np.outer(ct, sp).ravel(),
            np.outer(-st, np.ones_like(sp)).ravel(),
        ],
        axis=1,
    )
    ph_hat = np.stack(
        [
            np.outer(np.ones_like(ct), -sp).ravel(),
            np.outer(np.ones_like(ct), cp).ravel(),
            np.zeros(n_th * n_ph),
        ],
        axis=1,
    )

    n_dirs = r_hat.shape[0]
    N = np.empty((n_dirs, 3), dtype=complex)
    L = np.empty((n_dirs, 3), dtype=complex)
    for lo in range(0, n_dirs, _DIR_CHUNK):
        hi = min(lo + _DIR_CHUNK, n_dirs)
        phase = np.exp(1j * k * (r_hat[lo:hi] @ surf.centers.T))
        N[lo:hi] = phase @ wJ
        L[lo:hi] = phase @ wM

    N_th = np.einsum("dc,dc->d", N, th_hat)
    N_ph = np.einsum("dc,dc->d", N, ph_hat)
    L_th = np.einsum("dc,dc->d", L, th_hat)
    L_ph = np.einsum("dc,dc->d", L, ph_hat)

    # Far-zone amplitude A = r · e^{+jkr} · E(r) [V].
    pre = -1j * k / (4.0 * np.pi)
    A_th = pre * (_ETA0 * N_th + L_ph)
    A_ph = pre * (_ETA0 * N_ph - L_th)

    mask = None
    physical = [p for p in image_planes if p.physical_halfspace]
    if physical:
        mask = np.ones(n_dirs, dtype=bool)
        for plane in physical:
            inward = 1.0 if plane.at_low else -1.0
            mask &= inward * r_hat[:, plane.axis] >= -1e-12
        mask = mask.reshape(n_th, n_ph)

    return FarFieldResult(
        f=float(f),
        theta=theta,
        phi=phi,
        E_theta=np.conj(A_th).reshape(n_th, n_ph),
        E_phi=np.conj(A_ph).reshape(n_th, n_ph),
        accepted_power=accepted_power,
        surface_power=surface_power,
        physical_mask=mask,
        physical_sphere_fraction=0.5 ** len(physical),
    )


@dataclass
class FarFieldResult:
    """Far-field pattern of one frequency.

    The complex patterns are the r-independent far-zone amplitudes:
    the physical field at distance r is
    ``E(r) = (E_theta θ̂ + E_phi φ̂) · e^{-jkr} / r``.  Like every
    frequency-domain field of the library they are effective (RMS)
    phasors per √W of incident CW power, which makes
    ``realized_gain`` the directly measured quantity and keeps the
    intensity free of a peak-phasor ½.

    Attributes
    ----------
    f : float
        Frequency [Hz].
    theta, phi : ndarray
        Evaluation angles [rad]; θ from +z (ISO convention), φ from +x.
    E_theta, E_phi : (n_theta, n_phi) complex ndarray
        Far-zone amplitudes [V].
    accepted_power : float or None
        Accepted input power [W] behind :attr:`gain`; ``None`` until
        wired by the analysis.
    surface_power : float or None
        Real power [W] the recorded surface fields carry out of the
        Huygens box (``Re ∮ E × H* · n̂ dS``), in the same full-model
        watts as :attr:`P_rad` — a monitor on a symmetry half model
        counts the mirrored half as well.  For a lossless exterior it
        equals the radiated power; :attr:`power_balance` compares the
        two.  ``None`` for a result built without it.
    physical_mask : (n_theta, n_phi) bool ndarray or None
        False in directions behind an infinite ground plane; ``None``
        when the whole sphere is physical.
    physical_sphere_fraction : float
        Solid-angle share of the physical region (0.5 per ground
        plane).  The image expansion makes the pattern mirror-symmetric
        about every such plane, so the physical-region power integral
        is exactly this fraction of the smooth full-sphere integral —
        which avoids the half-cell quadrature bias a hard mask edge
        would cost.
    """

    f: float
    theta: np.ndarray
    phi: np.ndarray
    E_theta: np.ndarray
    E_phi: np.ndarray
    accepted_power: Optional[float] = None
    surface_power: Optional[float] = None
    physical_mask: Optional[np.ndarray] = None
    physical_sphere_fraction: float = 1.0

    @property
    def _U_unmasked(self) -> np.ndarray:
        # Library phasors are effective (RMS) amplitudes — the per-1-W-CW
        # normalisation makes P = |V|²/Z without a ½ (see the port-units
        # gate) — so the intensity carries no extra factor either.
        return (np.abs(self.E_theta) ** 2 + np.abs(self.E_phi) ** 2) / _ETA0

    @property
    def U(self) -> np.ndarray:
        """Radiation intensity [W/sr], zero behind a ground plane."""
        u = self._U_unmasked
        if self.physical_mask is not None:
            u = np.where(self.physical_mask, u, 0.0)
        return u

    @property
    def P_rad(self) -> float:
        """Total radiated power [W]: ∮ U dΩ over the physical sphere."""
        integrand = self._U_unmasked * np.sin(self.theta)[:, None]
        full = float(np.trapezoid(np.trapezoid(integrand, self.phi, axis=1), self.theta))
        return full * self.physical_sphere_fraction

    @property
    def directivity(self) -> np.ndarray:
        """Directivity 4π U / P_rad (linear)."""
        return 4.0 * np.pi * self.U / self.P_rad

    @property
    def realized_gain(self) -> np.ndarray:
        """Realized gain 4π U / P_incident (linear).

        The pattern of a solver run is per √W of incident power, so
        the reference is exactly 1 W and mismatch loss is included.
        """
        return 4.0 * np.pi * self.U

    @property
    def gain(self) -> np.ndarray:
        """IEEE gain 4π U / P_accepted (linear)."""
        if self.accepted_power is None:
            raise ValueError(
                "gain needs the accepted input power, which only a "
                "scattering run provides; use realized_gain (per "
                "incident watt) or directivity instead, or set "
                "accepted_power on the result."
            )
        return self.realized_gain / self.accepted_power

    @property
    def radiation_efficiency(self) -> float:
        """P_rad / P_accepted — 1.0 for a lossless model."""
        if self.accepted_power is None:
            raise ValueError(
                "radiation_efficiency needs the accepted input power, "
                "which only a scattering run provides; set "
                "accepted_power on the result to evaluate it."
            )
        return self.P_rad / self.accepted_power

    @property
    def power_balance(self) -> float:
        """P_rad / surface_power — 1.0 when the transform closes.

        The surface fields carry a definite real power out of the box;
        the far-field pattern must radiate the same power when the
        exterior is lossless.  A ratio below about 0.97 means the box
        samples the radiator's near zone too closely for the transform
        (measured 0.93 for a microstrip patch with the box top 0.3 λ
        above it, 1.00 at 0.7 λ); the pattern amplitude, and with it
        the realized gain, is then low by that factor while the
        self-normalised directivity is unaffected.
        """
        if self.surface_power is None:
            raise ValueError(
                "power_balance needs the surface power of the recording "
                "box, which only a far-field monitor provides; set "
                "surface_power on the result to evaluate it."
            )
        return self.P_rad / self.surface_power

    def _quantity(self, quantity: str) -> np.ndarray:
        table = {
            "realized_gain": lambda: self.realized_gain,
            "gain": lambda: self.gain,
            "directivity": lambda: self.directivity,
            "U": lambda: self.U,
        }
        try:
            return table[quantity]()
        except KeyError:
            raise ValueError(
                f"unknown quantity {quantity!r}; expected one of {sorted(table)}"
            ) from None

    def cut(
        self,
        *,
        plane: str = "phi",
        angle: float = 0.0,
        quantity: str = "realized_gain",
    ) -> tuple[np.ndarray, np.ndarray]:
        """One pattern cut for polar plotting.

        Parameters
        ----------
        plane : {"phi", "theta"}
            ``"phi"`` fixes the azimuth: the cut runs over θ at the
            azimuth *angle* (and continues over the back half at
            *angle* + π, giving the full 0…2π polar trace).
            ``"theta"`` fixes the polar angle and runs over φ.
        angle : float
            The fixed angle [rad].
        quantity : str
            ``"realized_gain"`` (default), ``"gain"``,
            ``"directivity"`` or ``"U"``.

        Returns
        -------
        angles, values : ndarray
            Cut angles [rad] and the (linear) quantity along the cut.
        """
        values = self._quantity(quantity)
        if plane == "theta":
            i = int(np.argmin(np.abs(self.theta - angle)))
            return self.phi, values[i, :]
        if plane != "phi":
            raise ValueError(f"plane must be 'phi' or 'theta', got {plane!r}")
        j_front = int(np.argmin(np.abs(self.phi - (angle % (2.0 * np.pi)))))
        j_back = int(np.argmin(np.abs(self.phi - ((angle + np.pi) % (2.0 * np.pi)))))
        # Front half: θ = 0…π at φ = angle; back half: θ folds back
        # through π…2π at the opposite azimuth.
        angles = np.concatenate([self.theta, 2.0 * np.pi - self.theta[-2::-1]])
        trace = np.concatenate([values[:, j_front], values[-2::-1, j_back]])
        return angles, trace

    def plot_cut(
        self,
        *,
        plane: str = "phi",
        angle: float = 0.0,
        quantity: str = "realized_gain",
        **kwargs,
    ):
        """Polar plot of one pattern cut (see :meth:`cut`).

        Extra keyword arguments go to
        :func:`magnelio.plots.plot_pattern_cut` (``db=``, ``floor_db=``,
        ``ax=``, ``label=``, ``title=``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.projections.polar.PolarAxes
        """
        from magnelio.post.plot_pattern import plot_pattern_cut  # noqa: PLC0415

        angles, trace = self.cut(plane=plane, angle=angle, quantity=quantity)
        return plot_pattern_cut(angles, trace, **kwargs)

    def plot_3d(self, *, quantity: str = "realized_gain", **kwargs):
        """3D radiation surface of the pattern.

        Extra keyword arguments go to
        :func:`magnelio.plots.plot_pattern_3d` (``db=``, ``floor_db=``,
        ``ax=``, ``cmap=``, ``title=``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        """
        from magnelio.post.plot_pattern import plot_pattern_3d  # noqa: PLC0415

        return plot_pattern_3d(self.theta, self.phi, self._quantity(quantity), **kwargs)
