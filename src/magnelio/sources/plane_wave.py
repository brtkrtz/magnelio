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
# the H-side correction sign inverted), DD-177 (the face corrections are a
# precomputed coefficient table, not a Python loop over boundary cells).

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
    # Array module, field dtype and its scalar type — the injection writes
    # into the solver's own arrays, so it must match both.
    _xp: object = dc_field(default=None, repr=False, init=False)
    _dtype: object = dc_field(default=None, repr=False, init=False)
    _scalar: object = dc_field(default=None, repr=False, init=False)
    # Face corrections, precomputed in attach()
    _patches_E: tuple = dc_field(default=(), repr=False, init=False)
    _patches_H: tuple = dc_field(default=(), repr=False, init=False)

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

    def excitation(self, t):
        """Waveform value at time *t* [s].

        Accepts a scalar or an array of times and follows the input: the
        TF/SF injection evaluates a whole box face in one call.
        """
        if self.waveform == "gaussian":
            f_max = self.f_max or 1e9
            t0 = 4.0 / f_max
            sigma = 2.0 / (math.pi * f_max)
            return self.amplitude * np.exp(-((t - t0) ** 2) / (2 * sigma**2))
        elif self.waveform == "sine":
            f = self.f_center or 1e9
            return self.amplitude * np.sin(2 * math.pi * f * t)
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

        # The injection writes into the solver's field arrays: same device,
        # same dtype.  ``_beta_E`` / ``_beta_H`` have already been moved to
        # the device at this point, so the folded coefficients are built
        # there and the time loop needs no host transfer.
        self._xp = solver._xp
        self._dtype = solver._real_dtype
        self._scalar = np.dtype(self._dtype).type

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
        self._patches_E, self._patches_H = self._build_patches()
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

    # ── TF/SF face corrections ────────────────────────────────────────────
    #
    # Each of the six box faces carries two corrections, one per tangential
    # field component, and all twelve have the same form:
    #
    #     field[face] += beta * metric * A * f(t - k.r/c0)
    #
    # Only ``t`` changes between time steps.  ``beta * metric * A`` is
    # therefore folded into one coefficient array per face at attach time,
    # which leaves the time loop a single array expression per face instead
    # of a Python iteration per boundary cell.  The retardation ``k.r/c0``
    # is constant as well, and for axis-aligned propagation it collapses to
    # a scalar on the two faces normal to k and to a 1-D array on the other
    # four — so the waveform is evaluated on a handful of values per step,
    # never on the face itself.

    def _beta_views(self) -> dict:
        """The six beta blocks reshaped onto their Yee component grids.

        ``_beta_E`` and ``_beta_H`` are flat concatenations; the reshapes
        are views, so this costs nothing and stays on whichever device the
        solver put them.
        """
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
        n_Ex, n_Ey = self._n_Ex, self._n_Ey
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        n_Hx, n_Hy = self._n_Hx, self._n_Hy
        n_Hz = Nx * Ny * (Nz + 1)
        bE, bH = self._beta_E, self._beta_H
        return {
            "Ex": bE[:n_Ex].reshape(Nx, Ny + 1, Nz + 1),
            "Ey": bE[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1),
            "Ez": bE[n_Ex + n_Ey : n_Ex + n_Ey + n_Ez].reshape(Nx + 1, Ny + 1, Nz),
            "Hx": bH[:n_Hx].reshape(Nx + 1, Ny, Nz),
            "Hy": bH[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz),
            "Hz": bH[n_Hx + n_Hy : n_Hx + n_Hy + n_Hz].reshape(Nx, Ny, Nz + 1),
        }

    def _patch(self, comp, index, beta, metric, factor, coords):
        """Fold one face correction into a ``(comp, index, delay, coef)`` record.

        *beta* is the coefficient block sliced to the face, *metric* the
        edge or face length broadcast against it, *factor* the signed
        incident amplitude the face carries, and *coords* the three
        position arrays of the face, each already shaped for broadcasting.
        Returns ``None`` when the incident field has no component on this
        face, so a source never touches a face it cannot excite.
        """
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

    def _build_patches(self) -> tuple[tuple, tuple]:
        """Assemble the coefficient records for all six TF/SF box faces.

        Each pair of comments names the curl-operator row being repaired:
        an update on one side of the box read a neighbour living on the
        other side, so the incident contribution it wrongly included — or
        wrongly left out — is put back here.
        """
        ix0, ix1, iy0, iy1, iz0, iz1 = self._box
        Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
        g = self._grid
        x, y, z = np.asarray(g.x), np.asarray(g.y), np.asarray(g.z)
        xc, yc, zc = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2, (z[:-1] + z[1:]) / 2
        b = self._beta_views()
        eh, hh = self._e_hat, self._h_hat
        dxa, dya, dza = self._dx_avg, self._dy_avg, self._dz_avg
        dx, dy, dz = self._dx, self._dy, self._dz

        # Cell ranges (i in [i0, i1)) and node ranges (i in [i0, i1]); which
        # one applies follows from the Yee position of the component.
        ix_c, ix_n = slice(ix0, ix1), slice(ix0, ix1 + 1)
        iy_c, iy_n = slice(iy0, iy1), slice(iy0, iy1 + 1)
        iz_c, iz_n = slice(iz0, iz1), slice(iz0, iz1 + 1)

        # Dual position of the H layer just outside each face.  A face flush
        # with the domain boundary has no layer beyond it — the node
        # position is the honest limit there.
        x_lo = (x[ix0 - 1] + x[ix0]) / 2
        y_lo = (y[iy0 - 1] + y[iy0]) / 2
        z_lo = (z[iz0 - 1] + z[iz0]) / 2
        x_hi = (x[ix1] + x[ix1 + 1]) / 2 if ix1 < Nx else x[ix1]
        y_hi = (y[iy1] + y[iy1 + 1]) / 2 if iy1 < Ny else y[iy1]
        z_hi = (z[iz1] + z[iz1 + 1]) / 2 if iz1 < Nz else z[iz1]

        # Column/row vectors for broadcasting onto a face.  The two varying
        # axes keep their grid order, so the lower one takes the column.
        X_c, X_n = xc[ix_c], x[ix_n]
        Y_c, Y_n = yc[iy_c], y[iy_n]
        Z_c, Z_n = zc[iz_c], z[iz_n]

        p = self._patch
        E, H = [], []

        # ── z-min face (k = iz0) ──────────────────────────────────────────
        # C^T row for Ex[i,j,iz0] reads +Hy[i,j,iz0-1] from the SF region,
        # so the incident Hy it lacks is added back.
        E.append(
            p(
                "Ex",
                (ix_c, iy_n, iz0),
                b["Ex"][ix_c, iy_n, iz0],
                dya[iy_n][None, :],
                +hh[1],
                (X_c[:, None], Y_n[None, :], z_lo),
            )
        )
        # C^T row for Ey[i,j,iz0] reads -Hx[i,j,iz0-1] from the SF region.
        E.append(
            p(
                "Ey",
                (ix_n, iy_c, iz0),
                b["Ey"][ix_n, iy_c, iz0],
                dxa[ix_n][:, None],
                -hh[0],
                (X_n[:, None], Y_c[None, :], z_lo),
            )
        )

        # ── z-max face (k = iz1) ──────────────────────────────────────────
        E.append(
            p(
                "Ex",
                (ix_c, iy_n, iz1),
                b["Ex"][ix_c, iy_n, iz1],
                dya[iy_n][None, :],
                -hh[1],
                (X_c[:, None], Y_n[None, :], z_hi),
            )
        )
        E.append(
            p(
                "Ey",
                (ix_n, iy_c, iz1),
                b["Ey"][ix_n, iy_c, iz1],
                dxa[ix_n][:, None],
                +hh[0],
                (X_n[:, None], Y_c[None, :], z_hi),
            )
        )

        # ── x-min face (i = ix0) ──────────────────────────────────────────
        E.append(
            p(
                "Ey",
                (ix0, iy_c, iz_n),
                b["Ey"][ix0, iy_c, iz_n],
                dza[iz_n][None, :],
                +hh[2],
                (x_lo, Y_c[:, None], Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix0, iy_n, iz_c),
                b["Ez"][ix0, iy_n, iz_c],
                dya[iy_n][:, None],
                -hh[1],
                (x_lo, Y_n[:, None], Z_c[None, :]),
            )
        )

        # ── x-max face (i = ix1) ──────────────────────────────────────────
        E.append(
            p(
                "Ey",
                (ix1, iy_c, iz_n),
                b["Ey"][ix1, iy_c, iz_n],
                dza[iz_n][None, :],
                -hh[2],
                (x_hi, Y_c[:, None], Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix1, iy_n, iz_c),
                b["Ez"][ix1, iy_n, iz_c],
                dya[iy_n][:, None],
                +hh[1],
                (x_hi, Y_n[:, None], Z_c[None, :]),
            )
        )

        # ── y-min face (j = iy0) ──────────────────────────────────────────
        E.append(
            p(
                "Ex",
                (ix_c, iy0, iz_n),
                b["Ex"][ix_c, iy0, iz_n],
                dza[iz_n][None, :],
                -hh[2],
                (X_c[:, None], y_lo, Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix_n, iy0, iz_c),
                b["Ez"][ix_n, iy0, iz_c],
                dxa[ix_n][:, None],
                +hh[0],
                (X_n[:, None], y_lo, Z_c[None, :]),
            )
        )

        # ── y-max face (j = iy1) ──────────────────────────────────────────
        E.append(
            p(
                "Ex",
                (ix_c, iy1, iz_n),
                b["Ex"][ix_c, iy1, iz_n],
                dza[iz_n][None, :],
                +hh[2],
                (X_c[:, None], y_hi, Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix_n, iy1, iz_c),
                b["Ez"][ix_n, iy1, iz_c],
                dxa[ix_n][:, None],
                -hh[0],
                (X_n[:, None], y_hi, Z_c[None, :]),
            )
        )

        # ── H side ────────────────────────────────────────────────────────
        # Mirror image of the above: the SF-side H face just outside the box
        # read a TF-side E edge.  A face flush with the domain boundary has
        # no H layer beyond it, so there is nothing to correct there.

        # z-min: Hy[i,j,iz0-1] read +Ex[i,j,iz0], Hx[i,j,iz0-1] read -Ey[i,j,iz0]
        H.append(
            p(
                "Hy",
                (ix_c, iy_n, iz0 - 1),
                b["Hy"][ix_c, iy_n, iz0 - 1],
                dx[ix_c][:, None],
                +eh[0],
                (X_c[:, None], Y_n[None, :], z[iz0]),
            )
        )
        H.append(
            p(
                "Hx",
                (ix_n, iy_c, iz0 - 1),
                b["Hx"][ix_n, iy_c, iz0 - 1],
                dy[iy_c][None, :],
                -eh[1],
                (X_n[:, None], Y_c[None, :], z[iz0]),
            )
        )

        # z-max
        if iz1 < Nz:
            H.append(
                p(
                    "Hy",
                    (ix_c, iy_n, iz1),
                    b["Hy"][ix_c, iy_n, iz1],
                    dx[ix_c][:, None],
                    -eh[0],
                    (X_c[:, None], Y_n[None, :], z[iz1]),
                )
            )
            H.append(
                p(
                    "Hx",
                    (ix_n, iy_c, iz1),
                    b["Hx"][ix_n, iy_c, iz1],
                    dy[iy_c][None, :],
                    +eh[1],
                    (X_n[:, None], Y_c[None, :], z[iz1]),
                )
            )

        # x-min: Hz[ix0-1,j,k] read +Ey[ix0,j,k], Hy[ix0-1,j,k] read -Ez[ix0,j,k]
        H.append(
            p(
                "Hz",
                (ix0 - 1, iy_c, iz_n),
                b["Hz"][ix0 - 1, iy_c, iz_n],
                dy[iy_c][:, None],
                +eh[1],
                (x[ix0], Y_c[:, None], Z_n[None, :]),
            )
        )
        H.append(
            p(
                "Hy",
                (ix0 - 1, iy_n, iz_c),
                b["Hy"][ix0 - 1, iy_n, iz_c],
                dz[iz_c][None, :],
                -eh[2],
                (x[ix0], Y_n[:, None], Z_c[None, :]),
            )
        )

        # x-max
        if ix1 < Nx:
            H.append(
                p(
                    "Hz",
                    (ix1, iy_c, iz_n),
                    b["Hz"][ix1, iy_c, iz_n],
                    dy[iy_c][:, None],
                    -eh[1],
                    (x[ix1], Y_c[:, None], Z_n[None, :]),
                )
            )
            H.append(
                p(
                    "Hy",
                    (ix1, iy_n, iz_c),
                    b["Hy"][ix1, iy_n, iz_c],
                    dz[iz_c][None, :],
                    +eh[2],
                    (x[ix1], Y_n[:, None], Z_c[None, :]),
                )
            )

        # y-min: Hz[i,iy0-1,k] read -Ex[i,iy0,k], Hx[i,iy0-1,k] read +Ez[i,iy0,k]
        H.append(
            p(
                "Hz",
                (ix_c, iy0 - 1, iz_n),
                b["Hz"][ix_c, iy0 - 1, iz_n],
                dx[ix_c][:, None],
                -eh[0],
                (X_c[:, None], y[iy0], Z_n[None, :]),
            )
        )
        H.append(
            p(
                "Hx",
                (ix_n, iy0 - 1, iz_c),
                b["Hx"][ix_n, iy0 - 1, iz_c],
                dz[iz_c][None, :],
                +eh[2],
                (X_n[:, None], y[iy0], Z_c[None, :]),
            )
        )

        # y-max
        if iy1 < Ny:
            H.append(
                p(
                    "Hz",
                    (ix_c, iy1, iz_n),
                    b["Hz"][ix_c, iy1, iz_n],
                    dx[ix_c][:, None],
                    +eh[0],
                    (X_c[:, None], y[iy1], Z_n[None, :]),
                )
            )
            H.append(
                p(
                    "Hx",
                    (ix_n, iy1, iz_c),
                    b["Hx"][ix_n, iy1, iz_c],
                    dz[iz_c][None, :],
                    -eh[2],
                    (X_n[:, None], y[iy1], Z_c[None, :]),
                )
            )

        return tuple(r for r in E if r is not None), tuple(r for r in H if r is not None)

    # ── TF/SF injection ───────────────────────────────────────────────────

    def _apply(self, patches, fields, t: float) -> None:
        """Add every face correction at time level *t*."""
        for comp, index, delay, coef in patches:
            wave = self.excitation(t - delay)
            wave = (
                wave.astype(self._dtype, copy=False)
                if getattr(wave, "ndim", 0)
                else self._scalar(wave)
            )
            getattr(fields, comp)[index] += coef * wave

    def inject_E(self, fields, t_E: float) -> None:
        """Apply TF/SF E-field corrections after the E update.

        Uses H_inc at t = t_E - dt/2 (half a step behind E).
        """
        if self._attached:
            self._apply(self._patches_E, fields, t_E - self._dt / 2)

    def inject_H(self, fields, t_H: float) -> None:
        """Apply TF/SF H-field corrections after the H update.

        Uses E_inc at t = t_H - dt/2 (the E time level just computed).
        """
        if self._attached:
            self._apply(self._patches_H, fields, t_H - self._dt / 2)

    def __repr__(self) -> str:
        return (
            f"PlaneWaveSource(dir={self.direction}, pol={self.polarization}, "
            f"waveform={self.waveform!r})"
        )
