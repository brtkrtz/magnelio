"""SourceFieldIncident — an incident field injected on a total-field/scattered-field box.

The domain is split by a virtual box: inside it the fields are the
*total* field (incident plus scattered), outside only the *scattered*
field remains.  Consistency corrections on the six box faces inject
the incident wave into the total-field region after every E and H
update.  This module holds the whole TF/SF machinery — the box, its
snapping to the grid, the binding of the waveform an excitation
supplies, and the face corrections as a precomputed coefficient
table — for *any* incident field given as a function of position and
time.  :class:`~magnelio.sources.SourcePlaneWave` specialises it
with the analytic plane wave, whose retardation collapses to a delay
table so the waveform is evaluated on a handful of values per step.

The solver states are FIT grid quantities: the incident samples are
converted per edge/face (``e = E_inc·l_primal``, ``h = H_inc·l_dual``
with the solver dual convention), so amplitudes are physical (V/m) on
any grid.
"""

# Design: DD-013 (TF/SF), DD-085 (grid-quantity states; the pre-DD-085
# implementation had the H-side correction sign inverted), DD-153
# (corners vocabulary), DD-177 (the face corrections are a precomputed
# coefficient table, not a Python loop over boundary cells), DD-224
# (sources on the model, waveform on the excitation; general incident
# field in Phase C).

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from magnelio.signals.waveforms import Waveform
from magnelio.sources.base import Source

_E_COMPONENTS = ("Ex", "Ey", "Ez")
_H_COMPONENTS = ("Hx", "Hy", "Hz")


@dataclass
class SourceFieldIncident(Source):
    """An incident field on a total-field/scattered-field box.

    Declared on the model with :meth:`~magnelio.GeometryModel.add_source`
    and driven by an :class:`~magnelio.Excitation` naming it.  The
    incident field is any function of position and time — a Gaussian
    beam, a focused wave, a tabulated field — evaluated on the six box
    faces every step; the plane wave has its own fast path in
    :class:`~magnelio.sources.SourcePlaneWave`.

    Parameters
    ----------
    name : str
        Source name — the handle an :class:`~magnelio.Excitation` uses.
    field : callable
        ``field(x, y, z, t, drive) -> ((Ex, Ey, Ez), (Hx, Hy, Hz))``,
        the incident field at the positions *x*, *y*, *z* (arrays of
        one shape, the samples of one box face) and time *t* [s]:
        E in V/m, H in A/m, each component broadcastable to the
        position shape.  ``drive(τ)`` is the excitation's time
        function ``amplitude · waveform(τ − delay)``, vectorised over
        *τ*, so a retarded field is ``drive(t − k̂·r / c₀)``.  The
        field must itself solve the free-space Maxwell equations
        (a superposition of plane waves); an inconsistent E/H pair
        leaks into the scattered-field region.
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        total-field region [m] — the same form as
        :meth:`~magnelio.geo.Brick.from_corners`.  Corner order does
        not matter, and a component may be ``None`` (or ``±math.inf``)
        to fall back to the default extent on that side (two bulk
        cells inside the domain boundary — the scattered-field shell
        the TF/SF split needs).  Snapped to the nearest grid nodes.
        ``None`` (default) uses the default extent on all six sides.

    Notes
    -----
    The amplitude of the incident field is the excitation's
    ``amplitude`` in ``V/m`` (:attr:`amplitude_unit`); the waveform is
    the excitation's ``waveform``.  Neither is part of the source.

    Examples
    --------
    A plane wave along +z spelled out as a general field (the
    dedicated class is the faster way to say this):

    >>> ETA0 = magnelio.constants.ETA0
    >>> def pw(x, y, z, t, drive):
    ...     f = drive(t - z / C0)
    ...     return (f, 0.0, 0.0), (0.0, f / ETA0, 0.0)
    >>> src = SourceFieldIncident(name="inc", field=pw)
    """

    name: str
    corners: tuple[tuple, tuple] | None = None
    field: Callable | None = None

    amplitude_unit = "V/m"
    _needs_field = True

    # --- excitation binding (set_excitation) ---
    _waveform: Waveform | None = dc_field(default=None, repr=False, init=False)
    _amplitude: float = dc_field(default=1.0, repr=False, init=False)
    _delay: float = dc_field(default=0.0, repr=False, init=False)
    # --- solver state (attach) ---
    _attached: bool = dc_field(default=False, repr=False, init=False)
    _Nx: int = dc_field(default=0, repr=False, init=False)
    _Ny: int = dc_field(default=0, repr=False, init=False)
    _Nz: int = dc_field(default=0, repr=False, init=False)
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
    # Array module, field dtype and its scalar type — the injection writes
    # into the solver's own arrays, so it must match both.
    _xp: object = dc_field(default=None, repr=False, init=False)
    _dtype: object = dc_field(default=None, repr=False, init=False)
    _scalar: object = dc_field(default=None, repr=False, init=False)
    # Face corrections, precomputed in attach()
    _patches_E: tuple = dc_field(default=(), repr=False, init=False)
    _patches_H: tuple = dc_field(default=(), repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"source name must be a non-empty string; got {self.name!r}")
        if self._needs_field:
            if self.field is None:
                raise TypeError(
                    f"{type(self).__name__} needs field=, a callable "
                    f"field(x, y, z, t, drive) -> ((Ex, Ey, Ez), (Hx, Hy, Hz)); "
                    f"for a plane wave use SourcePlaneWave",
                )
            if not callable(self.field):
                raise TypeError(f"field must be callable; got {type(self.field).__name__}")
        elif self.field is not None:
            raise TypeError(f"{type(self).__name__} defines its own incident field; drop field=")

    # ── construction helpers ─────────────────────────────────────────────

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
        >>> src = SourcePlaneWave.from_ranges(name="pw", x1=1e-3, x2=9e-3, z1=1e-3, z2=19e-3)
        """
        from magnelio.geo._ranges import corners_from_ranges  # noqa: PLC0415

        return cls(
            corners=corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz),
            **kwargs,
        )

    # ── excitation binding ───────────────────────────────────────────────

    def set_excitation(
        self,
        waveform: Waveform,
        *,
        amplitude: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        """Bind the waveform the source injects, with its weight.

        Parameters
        ----------
        waveform : Waveform
            Unit-peak time function of the incident field.
        amplitude : float, default 1.0
            Peak incident field [V/m].
        delay : float, default 0.0
            Time offset [s] of the waveform.
        """
        if not isinstance(waveform, Waveform):
            raise TypeError(
                f"{type(self).__name__}.set_excitation takes a magnelio.signals.Waveform; "
                f"got {type(waveform).__name__}",
            )
        amplitude = float(amplitude)
        delay = float(delay)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite; got {amplitude!r}")
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay must be a non-negative finite time [s]; got {delay!r}")
        self._waveform = waveform
        self._amplitude = amplitude
        self._delay = delay

    def clear_excitation(self) -> None:
        self._waveform = None
        self._amplitude = 1.0
        self._delay = 0.0

    @property
    def waveform(self) -> Waveform | None:
        """The bound waveform, or ``None`` before :meth:`set_excitation`."""
        return self._waveform

    def _require_waveform(self) -> Waveform:
        if self._waveform is None:
            raise ValueError(
                f"source {self.name!r} has no waveform: bind one with "
                f"set_excitation(waveform, amplitude=..., delay=...) before the run",
            )
        return self._waveform

    def _drive(self, t):
        """Incident amplitude ``A · w(t − delay)`` at time(s) *t* [s]."""
        w = self._require_waveform()
        return self._amplitude * w(t - self._delay)

    # ── TF/SF box ────────────────────────────────────────────────────────

    def _snap_box(self, grid, pml_cells=None) -> tuple[int, int, int, int, int, int]:
        """Snap the TF-region corners to nearest grid node indices.

        Returns (ix0, ix1, iy0, iy1, iz0, iz1) such that the TF region
        spans cells [ix0, ix1) × [iy0, iy1) × [iz0, iz1) in node indexing.

        The corners are normalised per axis (order does not matter, the
        shared ``corners=`` contract), and ``None``/``±inf`` components
        fall back to the default extent on that side: two bulk cells
        inside the physical domain, i.e. past the absorber cells the
        mesher appended (``pml_cells``, per face).  A box that leaves no
        scattered-field shell in the bulk is clamped inward — that is a
        property of the TF/SF split, not a silent reinterpretation of
        the input.
        """
        pml = pml_cells or {}
        n_hi = (self._Nx, self._Ny, self._Nz)
        # Bulk range per axis: the first and last node outside the absorber.
        bulk_lo = [int(pml.get(f"{ax}min", 0) or 0) for ax in "xyz"]
        bulk_hi = [n - int(pml.get(f"{ax}max", 0) or 0) for ax, n in zip("xyz", n_hi)]
        default_lo = [b + 2 for b in bulk_lo]
        default_hi = [b - 1 for b in bulk_hi]
        if self.corners is None:
            return (
                default_lo[0],
                default_hi[0],
                default_lo[1],
                default_hi[1],
                default_lo[2],
                default_hi[2],
            )

        p, q = self.corners
        lo, hi = [], []
        for a, b, d_lo, d_hi in zip(p, q, default_lo, default_hi):
            aa = None if a is None or not np.isfinite(a) else float(a)
            bb = None if b is None or not np.isfinite(b) else float(b)
            if aa is not None and bb is not None and bb < aa:
                aa, bb = bb, aa
            lo.append((aa, d_lo))
            hi.append((bb, d_hi))

        x, y, z = np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z)
        idx = []
        for axis, nodes in enumerate((x, y, z)):
            a, d_lo = lo[axis]
            b, d_hi = hi[axis]
            # One bulk cell of scattered field on each side at least.
            i_min, i_max = bulk_lo[axis] + 1, bulk_hi[axis] - 1
            i0 = d_lo if a is None else int(np.searchsorted(nodes, a).clip(i_min, i_max - 1))
            i1 = d_hi if b is None else int(np.searchsorted(nodes, b).clip(i0 + 1, i_max))
            idx += [i0, i1]

        return tuple(idx)

    # ── attach to solver ──────────────────────────────────────────────────

    def _attach_grid(self, solver) -> None:
        """Cache the solver's grid, coefficients and device conventions."""
        grid = solver.mesh.grid
        self._grid = grid
        self._dt = solver.dt
        self._Nx = solver.mesh.Nx
        self._Ny = solver.mesh.Ny
        self._Nz = solver.mesh.Nz

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

        # Snap TF/SF box to grid nodes, past the mesher's absorber cells
        self._box = self._snap_box(grid, getattr(solver.mesh, "pml_cells", None))

    def attach(self, solver) -> None:
        """Cache solver coefficients, snap the box and fold the face table.

        Called once from ``FITTimeDomainSolver.setup()``.  Requires a
        bound waveform (:meth:`set_excitation`).
        """
        self._require_waveform()
        self._attach_grid(solver)
        self._patches_E, self._patches_H = self._build_patches()
        self._attached = True

    # ── TF/SF face corrections ────────────────────────────────────────────
    #
    # Each of the six box faces carries two corrections, one per tangential
    # field component, and all twelve have the same form:
    #
    #     field[face] += beta * metric * sign * F_inc(r, t)
    #
    # where ``F_inc`` is one component of the incident field.  Only ``t``
    # changes between time steps, so ``beta * metric * sign`` is folded
    # into one coefficient array per face at attach time, which leaves the
    # time loop a single array expression per face instead of a Python
    # iteration per boundary cell.  The incident component on the face is
    # what the source evaluates each step — the user callable here, a
    # retarded scalar in the plane-wave subclass.

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

    def _patch(self, comp, index, beta, metric, fcomp, sign, coords):
        """Fold one face correction into a ``(comp, index, coef, fcomp, coords)`` record.

        *beta* is the coefficient block sliced to the face, *metric* the
        edge or face length broadcast against it, *fcomp* the incident
        component the face carries with *sign*, and *coords* the three
        position arrays of the face, each already shaped for
        broadcasting.  The user callable sees the positions as full
        face arrays.
        """
        xp = self._xp
        coef = beta * xp.asarray(np.asarray(sign * metric, dtype=self._dtype))
        X, Y, Z = np.broadcast_arrays(*(np.asarray(c, dtype=float) for c in coords))
        return (comp, index, coef, fcomp, (X, Y, Z))

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
                "Hy",
                +1.0,
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
                "Hx",
                -1.0,
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
                "Hy",
                -1.0,
                (X_c[:, None], Y_n[None, :], z_hi),
            )
        )
        E.append(
            p(
                "Ey",
                (ix_n, iy_c, iz1),
                b["Ey"][ix_n, iy_c, iz1],
                dxa[ix_n][:, None],
                "Hx",
                +1.0,
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
                "Hz",
                +1.0,
                (x_lo, Y_c[:, None], Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix0, iy_n, iz_c),
                b["Ez"][ix0, iy_n, iz_c],
                dya[iy_n][:, None],
                "Hy",
                -1.0,
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
                "Hz",
                -1.0,
                (x_hi, Y_c[:, None], Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix1, iy_n, iz_c),
                b["Ez"][ix1, iy_n, iz_c],
                dya[iy_n][:, None],
                "Hy",
                +1.0,
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
                "Hz",
                -1.0,
                (X_c[:, None], y_lo, Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix_n, iy0, iz_c),
                b["Ez"][ix_n, iy0, iz_c],
                dxa[ix_n][:, None],
                "Hx",
                +1.0,
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
                "Hz",
                +1.0,
                (X_c[:, None], y_hi, Z_n[None, :]),
            )
        )
        E.append(
            p(
                "Ez",
                (ix_n, iy1, iz_c),
                b["Ez"][ix_n, iy1, iz_c],
                dxa[ix_n][:, None],
                "Hx",
                -1.0,
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
                "Ex",
                +1.0,
                (X_c[:, None], Y_n[None, :], z[iz0]),
            )
        )
        H.append(
            p(
                "Hx",
                (ix_n, iy_c, iz0 - 1),
                b["Hx"][ix_n, iy_c, iz0 - 1],
                dy[iy_c][None, :],
                "Ey",
                -1.0,
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
                    "Ex",
                    -1.0,
                    (X_c[:, None], Y_n[None, :], z[iz1]),
                )
            )
            H.append(
                p(
                    "Hx",
                    (ix_n, iy_c, iz1),
                    b["Hx"][ix_n, iy_c, iz1],
                    dy[iy_c][None, :],
                    "Ey",
                    +1.0,
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
                "Ey",
                +1.0,
                (x[ix0], Y_c[:, None], Z_n[None, :]),
            )
        )
        H.append(
            p(
                "Hy",
                (ix0 - 1, iy_n, iz_c),
                b["Hy"][ix0 - 1, iy_n, iz_c],
                dz[iz_c][None, :],
                "Ez",
                -1.0,
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
                    "Ey",
                    -1.0,
                    (x[ix1], Y_c[:, None], Z_n[None, :]),
                )
            )
            H.append(
                p(
                    "Hy",
                    (ix1, iy_n, iz_c),
                    b["Hy"][ix1, iy_n, iz_c],
                    dz[iz_c][None, :],
                    "Ez",
                    +1.0,
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
                "Ex",
                -1.0,
                (X_c[:, None], y[iy0], Z_n[None, :]),
            )
        )
        H.append(
            p(
                "Hx",
                (ix_n, iy0 - 1, iz_c),
                b["Hx"][ix_n, iy0 - 1, iz_c],
                dz[iz_c][None, :],
                "Ez",
                +1.0,
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
                    "Ex",
                    +1.0,
                    (X_c[:, None], y[iy1], Z_n[None, :]),
                )
            )
            H.append(
                p(
                    "Hx",
                    (ix_n, iy1, iz_c),
                    b["Hx"][ix_n, iy1, iz_c],
                    dz[iz_c][None, :],
                    "Ez",
                    -1.0,
                    (X_n[:, None], y[iy1], Z_c[None, :]),
                )
            )

        return tuple(r for r in E if r is not None), tuple(r for r in H if r is not None)

    # ── incident field evaluation ─────────────────────────────────────────

    def _incident_on_face(self, coords, t: float):
        """The user field on one face at *t*.

        Every face correction owns its sample positions (the Yee
        staggering puts the two tangential components of one box face
        on different points), so the callable is evaluated once per
        correction and per step — the price of a field given as a
        function.  The plane wave's delay table is the cheap path.
        """
        X, Y, Z = coords
        try:
            E, H = self.field(X, Y, Z, t, self._drive)
        except TypeError as exc:
            raise TypeError(
                f"source {self.name!r}: field(x, y, z, t, drive) must return "
                f"((Ex, Ey, Ez), (Hx, Hy, Hz)); {exc}",
            ) from exc
        value = (tuple(E), tuple(H))
        if len(value[0]) != 3 or len(value[1]) != 3:
            raise ValueError(
                f"source {self.name!r}: field() must return two 3-tuples "
                f"((Ex, Ey, Ez), (Hx, Hy, Hz))",
            )
        return value

    def _incident_component(self, fcomp: str, coords, t: float):
        """One incident component on a face — an array of the face shape."""
        E, H = self._incident_on_face(coords, t)
        group, axis = fcomp[0], "xyz".index(fcomp[1])
        value = np.asarray((E if group == "E" else H)[axis], dtype=float)
        return np.broadcast_to(value, coords[0].shape)

    # ── TF/SF injection ───────────────────────────────────────────────────

    def _apply(self, patches, fields, t: float) -> None:
        """Add every face correction at time level *t*."""
        xp = self._xp
        for comp, index, coef, fcomp, coords in patches:
            wave = xp.asarray(self._incident_component(fcomp, coords, t), dtype=self._dtype)
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


__all__ = ["SourceFieldIncident"]
