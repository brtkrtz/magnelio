"""Conductor surface-roughness models for the wall-loss chain.

A rough conductor dissipates more than a smooth one of the same
footprint: the current follows a longer path over the profile, and once
the skin depth falls below the profile height the field starts to see
individual protrusions.  Both effects are carried by ONE scalar,
frequency-dependent multiplier on the surface resistance,

    R_s,rough(f) = K(f) * R_s,smooth(f),   K(f) >= 1,

which is exactly how the perturbative power-loss chain
consumes them: ``P_loss = 1/2 * K(f) * R_s(f) * sum(w * |H_tan|^2)``.
No solver change — the field solution stays the lossless PEC one.

Two models, both industry standard, both functions of the ratio of the
roughness scale to the skin depth ``delta = 1/sqrt(pi*f*mu*sigma)``:

* :class:`Hammerstad` — the classical curve fit on the RMS profile
  height.  Cheap, needs one datasheet number, saturates at 2.
* :class:`Huray` — the physics-based "snowball" model: loss of a sphere
  cluster on a flat base, derived from the analytic field solution
  around a conducting sphere.  More accurate on the strongly roughened
  foils where Hammerstad's saturation is the wrong asymptote.

Both are frozen and value-comparable, so they join ``Material``
equality and the project store without further machinery.

Note on causality
-----------------
``K(f)`` is real-valued: it multiplies the loss but leaves the reactive
part of the surface impedance untouched, which breaks the Hilbert
relation between them.  That makes it non-causal *as a time-domain
impedance boundary condition* [Bracken, DesignCon 2012].  It is not a
problem here, because the perturbative chain evaluates dissipated power
per frequency (an eigenmode's ``f0``, or a monitor's DFT bins) and never
forms a time-domain impedance.  A future self-consistent surface
impedance in the update would need the complex-valued causal form.

References
----------
Huray et al., "Fundamentals of a 3-D snowball model for surface
roughness power losses", IEEE SPI 2007.
Bracken, "A Causal Huray Model for Surface Roughness", DesignCon 2012 —
eq. (5) is the loss-factor form implemented here.
Hammerstad and Jensen, "Accurate models for microstrip computer-aided
design", IEEE MTT-S 1980.
Simonovich, "Practical method for modeling conductor surface roughness
using close packing of equal spheres", Signal Integrity Journal 2016 —
the Cannonball parameter set of :meth:`Huray.cannonball`.
"""

# Design: DD-088 (surface-roughness models), DD-082/DD-087 (perturbative
# power-loss chain).

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from magnelio.constants import MU0  # noqa: E402


def skin_depth(f, sigma: float, mu: float = 1.0) -> np.ndarray:
    """Skin depth ``delta(f) = 1/sqrt(pi*f*mu0*mu_r*sigma)`` [m].

    Parameters
    ----------
    f : float or array_like
        Frequency [Hz].
    sigma : float
        Conductivity [S/m].
    mu : float, optional
        Relative permeability (default 1).
    """
    return 1.0 / np.sqrt(np.pi * np.asarray(f, dtype=float) * MU0 * mu * sigma)


class SurfaceRoughness(ABC):
    """Frequency-dependent multiplier on a conductor's surface resistance."""

    @abstractmethod
    def factor(self, f, sigma: float, mu: float = 1.0) -> np.ndarray:
        """Roughness correction factor ``K(f) >= 1``.

        Parameters
        ----------
        f : float or array_like
            Frequency [Hz].
        sigma : float
            Conductivity [S/m] of the metal (enters through the skin
            depth — the roughness scale only matters relative to it).
        mu : float, optional
            Relative permeability of the metal (default 1).

        Returns
        -------
        numpy.ndarray
            ``K`` at each frequency.
        """


@dataclass(frozen=True)
class Hammerstad(SurfaceRoughness):
    """Hammerstad-Jensen roughness correction (the classical curve fit).

    ``K(f) = 1 + (2/pi) * arctan(1.4 * (Rq/delta(f))^2)``

    Fitted to measured microstrip loss, and adequate for low to moderate
    roughness.  Its ceiling is structural, not a fit artefact: the
    arctan saturates, so ``K -> 2`` however rough the profile gets.
    Strongly roughened foils exceed that ceiling in reality — use
    :class:`Huray` there.

    Parameters
    ----------
    rms_height : float
        RMS roughness Rq [m] of the profile.  ``0`` is the smooth limit
        (``K == 1`` at every frequency).
    """

    rms_height: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.rms_height < float("inf")):
            raise ValueError(
                f"Hammerstad requires a finite rms_height >= 0, got: {self.rms_height!r}"
            )

    def factor(self, f, sigma: float, mu: float = 1.0) -> np.ndarray:
        d = skin_depth(f, sigma, mu)
        return 1.0 + (2.0 / np.pi) * np.arctan(1.4 * (self.rms_height / d) ** 2)


@dataclass(frozen=True)
class Huray(SurfaceRoughness):
    """Huray "snowball" roughness correction (physics-based).

    A tile of flat base area ``A_flat`` carries a cluster of conducting
    spheres of radius ``a``.  Superposing the analytic loss of a single
    sphere in a uniform tangential field over the cluster gives
    [Bracken 2012, eq. (5)]

        K(f) = base_ratio + (3/2) * coverage
                            / (1 + delta/a + delta^2/(2*a^2))

    with ``coverage = N*4*pi*a^2 / A_flat`` the total sphere surface per
    unit tile area.  The asymptotes are the model's physical content:
    at low frequency ``delta >> a`` the field does not resolve the
    spheres and ``K -> base_ratio``; at high frequency ``delta << a``
    every sphere contributes its full surface and ``K -> base_ratio +
    1.5*coverage``.  Unlike :class:`Hammerstad` that ceiling follows
    the actual profile instead of saturating at 2.

    Parameters
    ----------
    radius : float
        Sphere ("snowball") radius a [m].  Must be > 0.
    coverage : float
        Total sphere surface area per unit flat tile area,
        ``N*4*pi*a^2/A_flat`` (dimensionless).  ``0`` is the smooth
        limit.
    base_ratio : float, optional
        Area of the matte base relative to a flat surface,
        ``A_matte/A_flat`` (default 1 — a flat base, the Cannonball
        assumption).  It scales the base plate's own loss.

    Notes
    -----
    Multiple sphere classes (radii ``a_i`` with coverages ``c_i``) are
    additive in the original model; sum the ``factor`` contributions of
    one :class:`Huray` per class and subtract the duplicated
    ``base_ratio`` terms, or use the single-class fit that the
    :meth:`cannonball` parameter set provides.
    """

    radius: float
    coverage: float
    base_ratio: float = 1.0

    # Cannonball stack: 14 equal spheres (9 + 4 + 1) on a square tile of
    # side 6a, its height set equal to the datasheet Rz.
    _CANNONBALL_SPHERES = 14
    _CANNONBALL_TILE_SIDE = 6.0
    _CANNONBALL_RZ_PER_RADIUS = 16.73  # = 4*sqrt(3)*(1 + sqrt(2))

    def __post_init__(self) -> None:
        if not (0.0 < self.radius < float("inf")):
            raise ValueError(f"Huray requires a finite radius > 0, got: {self.radius!r}")
        if not (0.0 <= self.coverage < float("inf")):
            raise ValueError(f"Huray requires a finite coverage >= 0, got: {self.coverage!r}")
        if not (0.0 <= self.base_ratio < float("inf")):
            raise ValueError(f"Huray requires a finite base_ratio >= 0, got: {self.base_ratio!r}")

    @classmethod
    def cannonball(cls, rz: float) -> "Huray":
        """Huray parameters from the datasheet Rz via the Cannonball stack.

        Foil datasheets publish the 10-point mean roughness Rz, not the
        sphere radius and tile area the Huray model wants.  The
        Cannonball model closes that gap with a fixed geometry: 14 equal
        spheres stacked 9 + 4 + 1 on a square tile of side ``6a``, the
        stack height identified with Rz, giving ``a = Rz/16.73`` and
        ``coverage = 14*4*pi*a^2/(6a)^2 = 56*pi/36``.  The coverage is
        therefore a pure constant — Rz sets the radius alone.

        Parameters
        ----------
        rz : float
            10-point mean roughness Rz [m] from the foil datasheet.

        Returns
        -------
        Huray
        """
        if not (0.0 < rz < float("inf")):
            raise ValueError(f"cannonball requires a finite rz > 0, got: {rz!r}")
        radius = rz / cls._CANNONBALL_RZ_PER_RADIUS
        coverage = cls._CANNONBALL_SPHERES * 4.0 * np.pi / cls._CANNONBALL_TILE_SIDE**2
        return cls(radius=radius, coverage=float(coverage))

    def factor(self, f, sigma: float, mu: float = 1.0) -> np.ndarray:
        d = skin_depth(f, sigma, mu)
        a = self.radius
        return self.base_ratio + 1.5 * self.coverage / (1.0 + d / a + d**2 / (2.0 * a**2))
