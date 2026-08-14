"""
Plane-wave source using the Total-Field / Scattered-Field (TF/SF) technique.

The TF/SF formulation divides the domain into a Total-Field (TF) region
containing the incident wave, and a Scattered-Field (SF) region outside.
Corrections are applied at the 6 faces of the TF/SF box after each E and H
update to inject the incident plane wave.

The solver states are FIT grid quantities: the incident
samples are converted per edge/face (``e = E_inc·l_primal``,
``h = H_inc·l_dual`` with the solver dual convention), so ``amplitude``
is the physical peak E-field in V/m on any grid.  The H-side
corrections carry the sign of the kernel's ``H = a·H − β·curl`` form
(an earlier implementation had it inverted — measured 0.39× TF
amplitude with massive SF leakage; now SF leakage sits at the numeric
dispersion floor).

v1.0 supports axis-aligned propagation only (k in {±x, ±y, ±z}).
"""

# Design: DD-085 (FIT grid-quantity states; the pre-DD-085 implementation had
# the H-side correction sign inverted).

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

# Free-space constants
from magnelio.constants import C0 as _C0  # noqa: E402
from magnelio.constants import ETA0 as _ETA0

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
            "Oblique plane-wave incidence is not supported in v1.0; "
            f"direction must be axis-aligned, got {d}"
        )
    return ax, int(np.sign(d[ax]))


@dataclass
class PlaneWaveSource:
    """Plane-wave excitation via TF/SF formulation.

    Parameters
    ----------
    direction : tuple of float
        Propagation direction unit vector ``(kx, ky, kz)``.
    polarization : tuple of float
        E-field polarization unit vector (⊥ *direction*).
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        total-field region [m] — the same form as
        :meth:`~magnelio.geo.Brick.from_corners`.  Corner order does
        not matter, and a component may be ``None`` (or ``±math.inf``)
        to fall back to the default extent on that side (two bulk
        cells inside the domain boundary — the scattered-field shell
        the TF/SF split needs).  Snapped to the nearest grid nodes.
        ``None`` (default) uses the default extent on all six sides.
    amplitude : float
        Peak E-field amplitude [V/m].
    waveform : {"gaussian", "sine"}
        Excitation waveform.
    f_center : float, optional
        Center frequency for the sine waveform [Hz].
    f_max : float, optional
        Bandwidth parameter for the Gaussian waveform [Hz].
    """

    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    polarization: tuple[float, float, float] = (1.0, 0.0, 0.0)
    corners: tuple[tuple, tuple] | None = None
    amplitude: float = 1.0
    waveform: str = "gaussian"
    f_center: float | None = None
    f_max: float | None = None

    # --- internal state (set by attach) ---
    _attached: bool = dc_field(default=False, repr=False, init=False)
    _prop_axis: int = dc_field(default=0, repr=False, init=False)
    _prop_sign: int = dc_field(default=1, repr=False, init=False)
    _k_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _e_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _h_hat: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    # TF/SF box node indices: (ix0, ix1, iy0, iy1, iz0, iz1)
    _box: tuple[int, ...] | None = dc_field(default=None, repr=False, init=False)
    # Cached solver arrays
    _beta_E: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _beta_H: np.ndarray | None = dc_field(default=None, repr=False, init=False)
    _n_Ex: int = dc_field(default=0, repr=False, init=False)
    _n_Ey: int = dc_field(default=0, repr=False, init=False)
    _n_Hx: int = dc_field(default=0, repr=False, init=False)
    _n_Hy: int = dc_field(default=0, repr=False, init=False)
    _grid: object = dc_field(default=None, repr=False, init=False)
    _dt: float = dc_field(default=0.0, repr=False, init=False)
    _Nx: int = dc_field(default=0, repr=False, init=False)
    _Ny: int = dc_field(default=0, repr=False, init=False)
    _Nz: int = dc_field(default=0, repr=False, init=False)

    # ── initialisation ────────────────────────────────────────────────────

    @classmethod
    def from_ranges(
        cls,
        *,
        x1=None,
        x2=None,
        dx=None,
        y1=None,
        y2=None,
        dy=None,
        z1=None,
        z2=None,
        dz=None,
        **kwargs,
    ):
        """Build the same source from one coordinate range per axis.

        The range spelling of ``corners=``, as in
        :meth:`~magnelio.geo.Brick.from_ranges`: each axis takes up to
        two of its three keywords — the two bounds (``x1``, ``x2``) or
        a bound and an extent (``x1``, ``dx`` / ``x2``, ``dx``).  Here
        an axis may also be open: give nothing for the whole domain
        extent, or a single bound to reach the domain boundary on the
        other side.  All remaining keyword arguments are forwarded to
        the constructor.

        Examples
        --------
        >>> src = PlaneWaveSource.from_ranges(x1=1e-3, x2=9e-3, y1=1e-3, y2=9e-3, z1=1e-3, z2=19e-3)
        """
        from magnelio.geo._ranges import corners_from_ranges  # noqa: PLC0415

        return cls(
            corners=corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz),
            **kwargs,
        )

    def __post_init__(self) -> None:
        d = np.array(self.direction, dtype=float)
        d /= np.linalg.norm(d)
        self.direction = tuple(d)

        p = np.array(self.polarization, dtype=float)
        p -= np.dot(p, d) * d
        norm_p = np.linalg.norm(p)
        if norm_p < 1e-10:
            raise ValueError(
                "polarization must not be parallel to direction; "
                f"got direction={self.direction}, polarization={self.polarization}"
            )
        p /= norm_p
        self.polarization = tuple(p)

    # ── waveform ──────────────────────────────────────────────────────────

    def excitation(self, t: float) -> float:
        """Scalar waveform value at time *t* [s]."""
        if self.waveform == "gaussian":
            f_max = self.f_max or 1e9
            t0 = 4.0 / f_max
            sigma = 2.0 / (math.pi * f_max)
            return self.amplitude * math.exp(-((t - t0) ** 2) / (2 * sigma**2))
        elif self.waveform == "sine":
            f = self.f_center or 1e9
            return self.amplitude * math.sin(2 * math.pi * f * t)
        raise ValueError(f"Unknown waveform: {self.waveform!r}")

    # ── incident field evaluation ─────────────────────────────────────────

    def _waveform_at(self, t: float, pos_along_k: float) -> float:
        """Evaluate waveform f(t − r·k̂/c₀) at a point along k̂."""
        t_ret = t - pos_along_k / _C0
        return self.excitation(t_ret)

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

        Called once from ``FITTimeDomainSolver.setup()``.
        """
        grid = solver.mesh.grid
        self._grid = grid
        self._dt = solver.dt
        self._Nx = solver.mesh.Nx
        self._Ny = solver.mesh.Ny
        self._Nz = solver.mesh.Nz

        # Copy flat coefficient arrays
        self._beta_E = solver._beta_E
        self._beta_H = solver._beta_H
        self._n_Ex = solver._n_Ex
        self._n_Ey = solver._n_Ey
        self._n_Hx = solver._n_Hx
        self._n_Hy = solver._n_Hy

        # Direction classification
        k = np.array(self.direction, dtype=float)
        self._k_hat = k
        self._prop_axis, self._prop_sign = _classify_axis(k)

        # Polarisation & H-direction
        e = np.array(self.polarization, dtype=float)
        self._e_hat = e * self.amplitude
        h = np.cross(k, e)
        self._h_hat = h * (self.amplitude / _ETA0)

        # DD-085: the solver states are FIT grid quantities
        # (e = E·l_primal, h = H·l_dual), so the injected incident
        # samples are converted per edge/face.  Dual widths use the
        # solver convention (_build_avg_d: boundary = full end cell).
        from magnelio._operators.material_matrices import (  # noqa: PLC0415
            _build_avg_d,
        )

        self._dx = np.asarray(grid.dx, dtype=float)
        self._dy = np.asarray(grid.dy, dtype=float)
        self._dz = np.asarray(grid.dz, dtype=float)
        self._dx_avg = _build_avg_d(self._dx, self._Nx)
        self._dy_avg = _build_avg_d(self._dy, self._Ny)
        self._dz_avg = _build_avg_d(self._dz, self._Nz)

        # Snap TF/SF box to grid nodes
        self._box = self._snap_box(grid)
        self._attached = True

    def _snap_box(self, grid) -> tuple[int, int, int, int, int, int]:
        """Snap the TF-region corners to nearest grid node indices.

        Returns (ix0, ix1, iy0, iy1, iz0, iz1) such that the TF region
        spans cells [ix0, ix1) × [iy0, iy1) × [iz0, iz1) in node indexing.

        The corners are normalised per axis (order does not matter, the
        shared ``corners=`` contract), and ``None``/``±inf`` components
        fall back to the default extent on that side.  A box that leaves
        no scattered-field shell is clamped inward — that is a property
        of the TF/SF split, not a silent reinterpretation of the input.
        """
        if self.corners is None:
            # Default: 2 cells inset from each face
            return (2, self._Nx - 1, 2, self._Ny - 1, 2, self._Nz - 1)

        p, q = self.corners
        lo, hi = [], []
        for a, b, default_lo, default_hi in zip(
            p, q, (2, 2, 2), (self._Nx - 1, self._Ny - 1, self._Nz - 1)
        ):
            aa = None if a is None or not np.isfinite(a) else float(a)
            bb = None if b is None or not np.isfinite(b) else float(b)
            if aa is not None and bb is not None and bb < aa:
                aa, bb = bb, aa
            lo.append((aa, default_lo))
            hi.append((bb, default_hi))

        x, y, z = np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z)
        n_hi = (self._Nx, self._Ny, self._Nz)
        idx = []
        for axis, (nodes, n) in enumerate(zip((x, y, z), n_hi)):
            a, default_lo = lo[axis]
            b, default_hi = hi[axis]
            i0 = default_lo if a is None else int(np.searchsorted(nodes, a).clip(1, n - 1))
            i1 = default_hi if b is None else int(np.searchsorted(nodes, b).clip(i0 + 1, n))
            idx += [i0, i1]

        return tuple(idx)

    # ── beta_E / beta_H lookup helpers ────────────────────────────────────

    def _beta_E_ex(self, i: int, j: int, k: int) -> float:
        """β_E for Ex edge at (i, j, k)."""
        Ny1, Nz1 = self._Ny + 1, self._Nz + 1
        return self._beta_E[i * Ny1 * Nz1 + j * Nz1 + k]

    def _beta_E_ey(self, i: int, j: int, k: int) -> float:
        """β_E for Ey edge at (i, j, k)."""
        Nz1 = self._Nz + 1
        return self._beta_E[self._n_Ex + i * self._Ny * Nz1 + j * Nz1 + k]

    def _beta_E_ez(self, i: int, j: int, k: int) -> float:
        """β_E for Ez edge at (i, j, k)."""
        Ny1 = self._Ny + 1
        return self._beta_E[self._n_Ex + self._n_Ey + i * Ny1 * self._Nz + j * self._Nz + k]

    def _beta_H_hx(self, i: int, j: int, k: int) -> float:
        """β_H for Hx face at (i, j, k)."""
        return self._beta_H[i * self._Ny * self._Nz + j * self._Nz + k]

    def _beta_H_hy(self, i: int, j: int, k: int) -> float:
        """β_H for Hy face at (i, j, k)."""
        Nz = self._Nz
        return self._beta_H[self._n_Hx + i * (self._Ny + 1) * Nz + j * Nz + k]

    def _beta_H_hz(self, i: int, j: int, k: int) -> float:
        """β_H for Hz face at (i, j, k)."""
        return self._beta_H[
            self._n_Hx + self._n_Hy + i * self._Ny * (self._Nz + 1) + j * (self._Nz + 1) + k
        ]

    # ── TF/SF injection (stubs — filled in A2) ───────────────────────────

    def inject_E(self, fields, t_E: float) -> None:
        """Apply TF/SF E-field corrections after E update.

        Uses H_inc at t = t_E − dt/2 (half-step behind E).
        """
        if not self._attached:
            return
        self._inject_E_impl(fields, t_E)

    def inject_H(self, fields, t_H: float) -> None:
        """Apply TF/SF H-field corrections after H update.

        Uses E_inc at t = t_H − dt/2 (the E time level just computed).
        """
        if not self._attached:
            return
        self._inject_H_impl(fields, t_H)

    def _inject_E_impl(self, fields, t_E: float) -> None:
        """TF/SF E-corrections on all 6 box faces.

        After the E update, E edges on the TF/SF boundary used an H neighbour
        from the wrong region.  We add/subtract the missing H_inc contribution.

        Timing: H_inc is evaluated at t_H = t_E − dt/2.
        """
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        g = self._grid
        x, y, z = np.asarray(g.x), np.asarray(g.y), np.asarray(g.z)
        t_H = t_E - self._dt / 2  # H time level
        hh = self._h_hat  # (3,)  H_inc amplitude vector

        # ── z-min face (k = iz0): E inside used H[…, iz0-1] from SF ──────
        # C^T row for Ex[i,j,iz0]: …+ Hy[i,j,iz0-1] − Hy[i,j,iz0]…
        # Hy[i,j,iz0-1] is in SF → missing +Hy_inc  → add β_E · Hy_inc
        if hh[1] != 0.0:  # Hy_inc component
            for i in range(ix0, ix1):
                for j in range(iy0, iy1 + 1):
                    z_hy = (z[iz0 - 1] + z[iz0]) / 2  # Hy dual z-pos
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[j], z_hy])
                    Hy_inc = float(np.dot(hh, [0, 1, 0])) * self._waveform_at(
                        t_H, np.dot(self._k_hat, r)
                    )
                    fields.Ex[i, j, iz0] += self._beta_E_ex(i, j, iz0) * Hy_inc * self._dy_avg[j]

        # C^T row for Ey[i,j,iz0]: …+ Hx[i,j,iz0] − Hx[i,j,iz0-1]…
        # Hx[i,j,iz0-1] is in SF → missing −Hx_inc → subtract
        if hh[0] != 0.0:  # Hx_inc component
            for i in range(ix0, ix1 + 1):
                for j in range(iy0, iy1):
                    z_hx = (z[iz0 - 1] + z[iz0]) / 2
                    r = np.array([x[i], y[j] + (y[j + 1] - y[j]) / 2, z_hx])
                    Hx_inc = float(np.dot(hh, [1, 0, 0])) * self._waveform_at(
                        t_H, np.dot(self._k_hat, r)
                    )
                    fields.Ey[i, j, iz0] -= self._beta_E_ey(i, j, iz0) * Hx_inc * self._dx_avg[i]

        # ── z-max face (k = iz1): E inside used H[…, iz1] from SF ────────
        # C^T for Ex[i,j,iz1]: …+ Hy[i,j,iz1-1] − Hy[i,j,iz1]…
        # Hy[i,j,iz1] is in SF → missing −Hy_inc → subtract
        if hh[1] != 0.0:
            for i in range(ix0, ix1):
                for j in range(iy0, iy1 + 1):
                    z_hy = (z[iz1] + z[iz1 + 1]) / 2 if iz1 < self._Nz else z[iz1]
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[j], z_hy])
                    Hy_inc = hh[1] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ex[i, j, iz1] -= self._beta_E_ex(i, j, iz1) * Hy_inc * self._dy_avg[j]

        # C^T for Ey[i,j,iz1]: …+ Hx[i,j,iz1] − Hx[i,j,iz1-1]…
        # Hx[i,j,iz1] is in SF → missing +Hx_inc → add
        if hh[0] != 0.0:
            for i in range(ix0, ix1 + 1):
                for j in range(iy0, iy1):
                    z_hx = (z[iz1] + z[iz1 + 1]) / 2 if iz1 < self._Nz else z[iz1]
                    r = np.array([x[i], y[j] + (y[j + 1] - y[j]) / 2, z_hx])
                    Hx_inc = hh[0] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ey[i, j, iz1] += self._beta_E_ey(i, j, iz1) * Hx_inc * self._dx_avg[i]

        # ── x-min face (i = ix0): E inside used H[ix0-1,…] from SF ───────
        # C^T for Ey[ix0,j,k]: …− Hz[ix0,j,k] + Hz[ix0-1,j,k]…
        # Hz[ix0-1,j,k] is in SF → missing +Hz_inc → add
        if hh[2] != 0.0:
            for j in range(iy0, iy1):
                for k in range(iz0, iz1 + 1):
                    x_hz = (x[ix0 - 1] + x[ix0]) / 2
                    r = np.array([x_hz, y[j] + (y[j + 1] - y[j]) / 2, z[k]])
                    Hz_inc = hh[2] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ey[ix0, j, k] += self._beta_E_ey(ix0, j, k) * Hz_inc * self._dz_avg[k]

        # C^T for Ez[ix0,j,k]: …+ Hy[ix0,j,k] − Hy[ix0-1,j,k]…
        # Hy[ix0-1,j,k] is in SF → missing −Hy_inc → subtract
        if hh[1] != 0.0:
            for j in range(iy0, iy1 + 1):
                for k in range(iz0, iz1):
                    x_hy = (x[ix0 - 1] + x[ix0]) / 2
                    r = np.array([x_hy, y[j], z[k] + (z[k + 1] - z[k]) / 2])
                    Hy_inc = hh[1] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ez[ix0, j, k] -= self._beta_E_ez(ix0, j, k) * Hy_inc * self._dy_avg[j]

        # ── x-max face (i = ix1): E inside used H[ix1,…] from SF ─────────
        # C^T for Ey[ix1,j,k]: …− Hz[ix1,j,k] + Hz[ix1-1,j,k]…
        # Hz[ix1,j,k] is in SF → missing −Hz_inc → subtract
        if hh[2] != 0.0:
            for j in range(iy0, iy1):
                for k in range(iz0, iz1 + 1):
                    x_hz = (x[ix1] + x[ix1 + 1]) / 2 if ix1 < self._Nx else x[ix1]
                    r = np.array([x_hz, y[j] + (y[j + 1] - y[j]) / 2, z[k]])
                    Hz_inc = hh[2] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ey[ix1, j, k] -= self._beta_E_ey(ix1, j, k) * Hz_inc * self._dz_avg[k]

        # C^T for Ez[ix1,j,k]: …+ Hy[ix1,j,k] − Hy[ix1-1,j,k]…
        # Hy[ix1,j,k] is in SF → missing +Hy_inc → add
        if hh[1] != 0.0:
            for j in range(iy0, iy1 + 1):
                for k in range(iz0, iz1):
                    x_hy = (x[ix1] + x[ix1 + 1]) / 2 if ix1 < self._Nx else x[ix1]
                    r = np.array([x_hy, y[j], z[k] + (z[k + 1] - z[k]) / 2])
                    Hy_inc = hh[1] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ez[ix1, j, k] += self._beta_E_ez(ix1, j, k) * Hy_inc * self._dy_avg[j]

        # ── y-min face (j = iy0): E inside used H[…,iy0-1,…] from SF ─────
        # C^T for Ex[i,iy0,k]: …+ Hz[i,iy0,k] − Hz[i,iy0-1,k]…
        # Hz[i,iy0-1,k] is in SF → missing −Hz_inc → subtract
        if hh[2] != 0.0:
            for i in range(ix0, ix1):
                for k in range(iz0, iz1 + 1):
                    y_hz = (y[iy0 - 1] + y[iy0]) / 2
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y_hz, z[k]])
                    Hz_inc = hh[2] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ex[i, iy0, k] -= self._beta_E_ex(i, iy0, k) * Hz_inc * self._dz_avg[k]

        # C^T for Ez[i,iy0,k]: …− Hx[i,iy0,k] + Hx[i,iy0-1,k]…
        # Hx[i,iy0-1,k] is in SF → missing +Hx_inc → add
        if hh[0] != 0.0:
            for i in range(ix0, ix1 + 1):
                for k in range(iz0, iz1):
                    y_hx = (y[iy0 - 1] + y[iy0]) / 2
                    r = np.array([x[i], y_hx, z[k] + (z[k + 1] - z[k]) / 2])
                    Hx_inc = hh[0] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ez[i, iy0, k] += self._beta_E_ez(i, iy0, k) * Hx_inc * self._dx_avg[i]

        # ── y-max face (j = iy1): E inside used H[…,iy1,…] from SF ───────
        # C^T for Ex[i,iy1,k]: …+ Hz[i,iy1,k] − Hz[i,iy1-1,k]…
        # Hz[i,iy1,k] is in SF → missing +Hz_inc → add
        if hh[2] != 0.0:
            for i in range(ix0, ix1):
                for k in range(iz0, iz1 + 1):
                    y_hz = (y[iy1] + y[iy1 + 1]) / 2 if iy1 < self._Ny else y[iy1]
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y_hz, z[k]])
                    Hz_inc = hh[2] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ex[i, iy1, k] += self._beta_E_ex(i, iy1, k) * Hz_inc * self._dz_avg[k]

        # C^T for Ez[i,iy1,k]: …− Hx[i,iy1,k] + Hx[i,iy1-1,k]…
        # Hx[i,iy1,k] is in SF → missing −Hx_inc → subtract
        if hh[0] != 0.0:
            for i in range(ix0, ix1 + 1):
                for k in range(iz0, iz1):
                    y_hx = (y[iy1] + y[iy1 + 1]) / 2 if iy1 < self._Ny else y[iy1]
                    r = np.array([x[i], y_hx, z[k] + (z[k + 1] - z[k]) / 2])
                    Hx_inc = hh[0] * self._waveform_at(t_H, np.dot(self._k_hat, r))
                    fields.Ez[i, iy1, k] -= self._beta_E_ez(i, iy1, k) * Hx_inc * self._dx_avg[i]

    def _inject_H_impl(self, fields, t_H: float) -> None:
        """TF/SF H-corrections on all 6 box faces.

        After the H update, H faces on the TF/SF boundary used an E neighbour
        from the wrong region.  We add/subtract the missing E_inc contribution.

        Timing: E_inc is evaluated at t_E = t_H − dt/2.
        """
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        g = self._grid
        x, y, z = np.asarray(g.x), np.asarray(g.y), np.asarray(g.z)
        t_E = t_H - self._dt / 2  # E time level
        eh = self._e_hat

        # ── z-min face (k = iz0): Hy[i,j,iz0-1] (SF) used Ex[i,j,iz0] (TF)
        # C row for Hy: …+ Ex[i,j,k+1] − Ex[i,j,k]…
        # Hy[i,j,iz0-1] row uses +Ex[i,j,iz0] from TF → missing −Ex_inc
        if eh[0] != 0.0:
            for i in range(ix0, ix1):
                for j in range(iy0, iy1 + 1):
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[j], z[iz0]])
                    Ex_inc = eh[0] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hy[i, j, iz0 - 1] += (
                        self._beta_H_hy(i, j, iz0 - 1) * Ex_inc * self._dx[i]
                    )

        # Hx[i,j,iz0-1] (SF) row uses −Ey[i,j,iz0] from TF → missing +Ey_inc
        if eh[1] != 0.0:
            for i in range(ix0, ix1 + 1):
                for j in range(iy0, iy1):
                    r = np.array([x[i], y[j] + (y[j + 1] - y[j]) / 2, z[iz0]])
                    Ey_inc = eh[1] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hx[i, j, iz0 - 1] -= (
                        self._beta_H_hx(i, j, iz0 - 1) * Ey_inc * self._dy[j]
                    )

        # ── z-max face (k = iz1): Hy[i,j,iz1] (SF) used Ex[i,j,iz1] (TF)
        # Hy[i,j,iz1] row uses −Ex[i,j,iz1] from TF → need +Ex_inc correction
        # Actually: Hy row k=iz1 uses +Ex[i,j,iz1+1]−Ex[i,j,iz1]
        # Ex[i,j,iz1] is on TF boundary → Hy[i,j,iz1] in SF uses −Ex[i,j,iz1]
        # → missing +Ex_inc
        if eh[0] != 0.0:
            for i in range(ix0, ix1):
                for j in range(iy0, iy1 + 1):
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[j], z[iz1]])
                    Ex_inc = eh[0] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hy[i, j, iz1] -= self._beta_H_hy(i, j, iz1) * Ex_inc * self._dx[i]

        # Hx[i,j,iz1] (SF) row uses +Ey[i,j,iz1] from TF → missing −Ey_inc
        if eh[1] != 0.0:
            for i in range(ix0, ix1 + 1):
                for j in range(iy0, iy1):
                    r = np.array([x[i], y[j] + (y[j + 1] - y[j]) / 2, z[iz1]])
                    Ey_inc = eh[1] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hx[i, j, iz1] += self._beta_H_hx(i, j, iz1) * Ey_inc * self._dy[j]

        # ── x-min face (i = ix0): Hz[ix0-1,j,k] (SF) used Ey[ix0,j,k] (TF)
        # Hz row: +Ey[i+1,j,k] − Ey[i,j,k]
        # Hz[ix0-1,j,k] uses +Ey[ix0,j,k] from TF → missing −Ey_inc
        if eh[1] != 0.0:
            for j in range(iy0, iy1):
                for k in range(iz0, iz1 + 1):
                    r = np.array([x[ix0], y[j] + (y[j + 1] - y[j]) / 2, z[k]])
                    Ey_inc = eh[1] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hz[ix0 - 1, j, k] += (
                        self._beta_H_hz(ix0 - 1, j, k) * Ey_inc * self._dy[j]
                    )

        # Hy[ix0-1,j,k] (SF) row uses −Ez[ix0,j,k] from TF → +Ez_inc
        if eh[2] != 0.0:
            for j in range(iy0, iy1 + 1):
                for k in range(iz0, iz1):
                    r = np.array([x[ix0], y[j], z[k] + (z[k + 1] - z[k]) / 2])
                    Ez_inc = eh[2] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hy[ix0 - 1, j, k] -= (
                        self._beta_H_hy(ix0 - 1, j, k) * Ez_inc * self._dz[k]
                    )

        # ── x-max face (i = ix1): Hz[ix1,j,k] (SF) used −Ey[ix1,j,k] from TF
        # Hz row at i=ix1: +Ey[ix1+1,j,k] − Ey[ix1,j,k]
        # Ey[ix1] is TF boundary → Hz[ix1] in SF uses −Ey[ix1] → missing +Ey_inc
        if eh[1] != 0.0:
            for j in range(iy0, iy1):
                for k in range(iz0, iz1 + 1):
                    r = np.array([x[ix1], y[j] + (y[j + 1] - y[j]) / 2, z[k]])
                    Ey_inc = eh[1] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hz[ix1, j, k] -= self._beta_H_hz(ix1, j, k) * Ey_inc * self._dy[j]

        # Hy[ix1,j,k] (SF) row uses +Ez[ix1,j,k] from TF → −Ez_inc
        # Hy row at i=ix1: −Ez[ix1+1,j,k] + Ez[ix1,j,k]
        # Wait — Hy index range is i∈[0,Nx). ix1 could be Nx.
        # Hy[ix1-1,j,k] is inside TF; we need Hy[ix1,j,k] but only if ix1<Nx
        # Actually Hy is at (i,j,k) for i∈[0,Nx), and its row uses
        # −Ez[i+1,j,k]+Ez[i,j,k]. For the x-max boundary Ez[ix1,j,k] is on
        # the TF boundary. The SF-side H face that uses it is Hy[ix1,j,k]
        # (if ix1 < Nx). Its row: +Ex[ix1,j,k+1]−Ex[ix1,j,k]−Ez[ix1+1,j,k]+Ez[ix1,j,k]
        # The +Ez[ix1,j,k] term pulls from TF → must subtract Ez_inc
        if eh[2] != 0.0 and ix1 < self._Nx:
            for j in range(iy0, iy1 + 1):
                for k in range(iz0, iz1):
                    r = np.array([x[ix1], y[j], z[k] + (z[k + 1] - z[k]) / 2])
                    Ez_inc = eh[2] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hy[ix1, j, k] += self._beta_H_hy(ix1, j, k) * Ez_inc * self._dz[k]

        # ── y-min face (j = iy0): Hz[i,iy0-1,k] (SF) used −Ex[i,iy0,k] (TF)
        # Hz row: −Ex[i,j+1,k] + Ex[i,j,k]
        # Hz[i,iy0-1,k] uses −Ex[i,iy0,k] from TF → missing +Ex_inc
        if eh[0] != 0.0:
            for i in range(ix0, ix1):
                for k in range(iz0, iz1 + 1):
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[iy0], z[k]])
                    Ex_inc = eh[0] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hz[i, iy0 - 1, k] -= (
                        self._beta_H_hz(i, iy0 - 1, k) * Ex_inc * self._dx[i]
                    )

        # Hx[i,iy0-1,k] (SF) row uses +Ez[i,iy0,k] from TF → −Ez_inc
        # Hx row: +Ez[i,j+1,k]−Ez[i,j,k]−Ey[i,j,k+1]+Ey[i,j,k]
        # Hx[i,iy0-1,k] uses +Ez[i,iy0,k] from TF → missing −Ez_inc
        if eh[2] != 0.0:
            for i in range(ix0, ix1 + 1):
                for k in range(iz0, iz1):
                    r = np.array([x[i], y[iy0], z[k] + (z[k + 1] - z[k]) / 2])
                    Ez_inc = eh[2] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hx[i, iy0 - 1, k] += (
                        self._beta_H_hx(i, iy0 - 1, k) * Ez_inc * self._dz[k]
                    )

        # ── y-max face (j = iy1): Hz[i,iy1,k] (SF) used +Ex[i,iy1,k] (TF)
        # Hz row at j=iy1: −Ex[i,iy1+1,k]+Ex[i,iy1,k]  (but iy1+1 may be out)
        # Actually Hz[i,iy1,k] has row: +Ey[i+1,iy1,k]−Ey[i,iy1,k]−Ex[i,iy1+1,k]+Ex[i,iy1,k]
        # Wait, Hz index j∈[0,Ny). If iy1=Ny this face doesn't exist.
        # For iy1<Ny: Hz[i,iy1,k] uses +Ex[i,iy1,k] which is TF boundary → −Ex_inc
        if eh[0] != 0.0 and iy1 < self._Ny:
            for i in range(ix0, ix1):
                for k in range(iz0, iz1 + 1):
                    r = np.array([x[i] + (x[i + 1] - x[i]) / 2, y[iy1], z[k]])
                    Ex_inc = eh[0] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hz[i, iy1, k] += self._beta_H_hz(i, iy1, k) * Ex_inc * self._dx[i]

        # Hx[i,iy1,k] (SF) uses −Ez[i,iy1,k] from TF → +Ez_inc
        # Hx row: +Ez[i,j+1,k]−Ez[i,j,k]. At j=iy1: −Ez[i,iy1,k]
        # Wait, Hx[i,iy1,k] — Hx index j∈[0,Ny). iy1 could be Ny.
        # For iy1<Ny: Hx[i,iy1,k] uses −Ez[i,iy1,k] from TF → +Ez_inc
        if eh[2] != 0.0 and iy1 < self._Ny:
            for i in range(ix0, ix1 + 1):
                for k in range(iz0, iz1):
                    r = np.array([x[i], y[iy1], z[k] + (z[k + 1] - z[k]) / 2])
                    Ez_inc = eh[2] * self._waveform_at(t_E, np.dot(self._k_hat, r))
                    fields.Hx[i, iy1, k] -= self._beta_H_hx(i, iy1, k) * Ez_inc * self._dz[k]

    def __repr__(self) -> str:
        return (
            f"PlaneWaveSource(dir={self.direction}, pol={self.polarization}, "
            f"waveform={self.waveform!r})"
        )
