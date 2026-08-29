"""
SourcePlaneWave — a plane wave injected on a total-field/scattered-field box.

The TF/SF formulation divides the domain into a Total-Field (TF) region
containing the incident wave, and a Scattered-Field (SF) region outside.
Corrections are applied at the 6 faces of the TF/SF box after each E and H
update to inject the incident plane wave.  The waveform and the peak
field come from the :class:`~magnelio.Excitation` that drives the source
(bound through :meth:`SourceFieldIncident.set_excitation`).

The face machinery lives in :class:`~magnelio.sources.SourceFieldIncident`;
this class supplies the analytic plane wave, whose retardation folds
into a delay table so the waveform is evaluated on a handful of values
per step.  The H-side corrections carry the sign of the kernel's
``H = a·H − β·curl`` form (an earlier implementation had it inverted —
measured 0.39× TF amplitude with massive SF leakage; now SF leakage
sits at the numeric dispersion floor).

Axis-aligned propagation only (k in {±x, ±y, ±z}).
"""

# Design: DD-085 (FIT grid-quantity states; the pre-DD-085 implementation had
# the H-side correction sign inverted), DD-177 (the face corrections are a
# precomputed coefficient table, not a Python loop over boundary cells),
# DD-224 (waveform and amplitude live on the excitation; the source is a
# model object; the general incident field is the base class).

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

# Free-space constants
from magnelio.constants import C0 as _C0  # noqa: E402
from magnelio.constants import ETA0 as _ETA0
from magnelio.sources.field_incident import SourceFieldIncident

# Axis-aligned unit vectors
_AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def _classify_axis(d: np.ndarray) -> tuple[int, int]:
    """Return (axis_index, sign) for an axis-aligned direction vector.

    Returns e.g. (2, +1) for +z propagation.
    Raises ValueError if not axis-aligned.
    """
    abs_d = np.abs(d)
    ax = int(np.argmax(abs_d))
    if abs_d[ax] < 0.999:
        raise NotImplementedError(
            f"Oblique plane-wave incidence is not supported; direction must be "
            f"axis-aligned, got {d}"
        )
    return ax, int(np.sign(d[ax]))


@dataclass
class SourcePlaneWave(SourceFieldIncident):
    """Plane-wave illumination on a total-field/scattered-field box.

    Declared on the model with :meth:`~magnelio.GeometryModel.add_source`
    and driven by an :class:`~magnelio.Excitation` naming it; the
    excitation's ``amplitude`` is the peak incident field in V/m and
    its ``waveform`` the time function.  Propagation must be along a
    grid axis.

    Parameters
    ----------
    name : str
        Source name — the handle an :class:`~magnelio.Excitation` uses.
    direction : tuple of float
        Propagation direction ``(kx, ky, kz)``; normalised, must be
        axis-aligned.
    polarization : tuple of float
        E-field polarization vector; its component along *direction*
        is projected out and the rest normalised.
    corners : tuple of tuple, optional
        Two opposite corners of the total-field region [m]; see
        :class:`SourceFieldIncident`.  ``None`` (default) leaves a
        two-cell scattered-field shell inside every domain face.

    Examples
    --------
    >>> from magnelio import sources
    >>> pw = sources.SourcePlaneWave(name="pw", direction=(0, 0, 1), polarization=(1, 0, 0))
    """

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)

    _needs_field = False

    # --- internal state (set by attach) ---
    _prop_axis: int = dc_field(default=0, repr=False, init=False)
    _prop_sign: int = dc_field(default=1, repr=False, init=False)
    _k_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _e_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _h_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)

    # ── initialisation ────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        super().__post_init__()
        d = np.array(self.direction, dtype=float)
        norm_d = np.linalg.norm(d)
        if not norm_d > 0.0:
            raise ValueError(f"direction must be a non-zero vector; got {self.direction}")
        # Normalise only when needed, so a stored (already unit) vector
        # reloads bit-identically.
        if abs(norm_d - 1.0) > 1e-12:
            d /= norm_d
        self.direction = tuple(float(v) for v in d)

        p = np.array(self.polarization, dtype=float)
        if abs(np.dot(p, d)) > 1e-12:
            p -= np.dot(p, d) * d
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-10:
            raise ValueError(
                "polarization must not be parallel to direction; "
                f"got direction={self.direction}, polarization={self.polarization}"
            )
        if abs(norm_p - 1.0) > 1e-12:
            p /= norm_p
        self.polarization = tuple(float(v) for v in p)

    # ── incident field evaluation ─────────────────────────────────────────

    def _waveform_at(self, t: float, pos_along_k: float) -> float:
        """Evaluate the drive A·w(t − r·k̂/c₀ − delay) at a point along k̂."""
        t_ret = t - pos_along_k / _C0
        return self._drive(t_ret)

    def incident_E(self, r: np.ndarray, t: float) -> np.ndarray:
        """Incident E-field vector [V/m] at position *r* and time *t*.

        E_inc(r, t) = E0 · ê · f(t − k̂·r / c₀)
        """
        pos = np.dot(self._k_hat, r)
        return self._e_hat * self._waveform_at(t, pos)

    def incident_H(self, r: np.ndarray, t: float) -> np.ndarray:
        """Incident H-field vector [A/m] at position *r* and time *t*.

        H_inc = (1/η₀) · (k̂ × ê) · f(t − k̂·r / c₀) = (E0/η₀) · ĥ · f(…)
        """
        pos = np.dot(self._k_hat, r)
        return self._h_hat * self._waveform_at(t, pos)

    # ── attach to solver ──────────────────────────────────────────────────

    def attach(self, solver) -> None:
        """Cache solver coefficients and snap TF/SF box to grid nodes.

        Called once from ``FITTimeDomainSolver.setup()``.  Requires a
        bound waveform (:meth:`set_excitation`).
        """
        self._require_waveform()
        self._attach_grid(solver)

        # Direction classification
        k = np.array(self.direction, dtype=float)
        self._k_hat = k
        self._prop_axis, self._prop_sign = _classify_axis(k)

        # Polarisation & H-direction (unit incident field; the amplitude
        # multiplies the drive)
        e = np.array(self.polarization, dtype=float)
        self._e_hat = e
        h = np.cross(k, e)
        self._h_hat = h / _ETA0

        self._patches_E, self._patches_H = self._build_patches()
        self._attached = True

    # ── TF/SF face corrections ────────────────────────────────────────────
    #
    # The plane wave folds more than the base class: the incident
    # component on a face is ``unit[fcomp] · f(t − k̂·r/c₀)``, so the unit
    # vector's component joins the coefficient and the retardation
    # ``k̂·r/c₀`` becomes a constant delay table — a scalar on the two
    # faces normal to k and a 1-D array on the other four.  The waveform
    # is then evaluated on a handful of values per step, never on the
    # face itself.

    def _patch(self, comp, index, beta, metric, fcomp, sign, coords):
        """Fold one face correction into a ``(comp, index, delay, coef)`` record.

        Returns ``None`` when the incident field has no component on this
        face, so the wave never touches a face it cannot excite.
        """
        unit = self._e_hat if fcomp[0] == "E" else self._h_hat
        factor = sign * unit["xyz".index(fcomp[1])]
        if factor == 0.0:
            return None
        xp = self._xp
        coef = beta * xp.asarray(np.asarray(factor * metric, dtype=self._dtype))
        pos = 0.0
        for k_c, c in zip(self._k_hat, coords):
            if k_c != 0.0:
                pos = pos + k_c * c
        delay = np.asarray(pos, dtype=np.float64) / _C0
        return (comp, index, xp.asarray(delay) if delay.ndim else float(delay), coef)

    # ── TF/SF injection ───────────────────────────────────────────────────

    def _apply(self, patches, fields, t: float) -> None:
        """Add every face correction at time level *t*."""
        for comp, index, delay, coef in patches:
            wave = self._drive(t - delay)
            wave = (
                wave.astype(self._dtype, copy=False)
                if getattr(wave, "ndim", 0)
                else self._scalar(wave)
            )
            getattr(fields, comp)[index] += coef * wave


__all__ = ["SourcePlaneWave"]
