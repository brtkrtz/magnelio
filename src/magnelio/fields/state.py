"""FieldState — E and H on a grid, with the positions of every sample.

The solver keeps its fields as FIT grid quantities (``e = E·l`` on the
primal edges, ``h = H·l_dual`` on the dual edges through the primal
faces); that layout is the internal ``_fields.FieldArrays`` and carries
no grid.  This class adds the grid lines and the Yee offset
convention, so a user sees physical fields with known positions:

============  ==================================  ==========================
Component     Sample position                     Array shape
============  ==================================  ==========================
``Ex``        ``(xc[i], y[j], z[k])``             ``(Nx, Ny+1, Nz+1)``
``Ey``        ``(x[i], yc[j], z[k])``             ``(Nx+1, Ny, Nz+1)``
``Ez``        ``(x[i], y[j], zc[k])``             ``(Nx+1, Ny+1, Nz)``
``Hx``        ``(x[i], yc[j], zc[k])``            ``(Nx+1, Ny, Nz)``
``Hy``        ``(xc[i], y[j], zc[k])``            ``(Nx, Ny+1, Nz)``
``Hz``        ``(xc[i], yc[j], z[k])``            ``(Nx, Ny, Nz+1)``
============  ==================================  ==========================

with ``x``/``y``/``z`` the grid nodes and ``xc``/``yc``/``zc`` the cell
centres.  E is in V/m, H in A/m.
"""

# Design: DD-224 (public field container, Phase C), DD-085 (grid
# quantities vs physical fields), DD-002 (Yee layout).

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from magnelio._fields.field_arrays import FieldArrays
from magnelio.fields._interp import _interp_to_cell_centres, _solver_dual_widths
from magnelio.mesh.grid import GridLines

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_AXES = ("x", "y", "z")

# Dual widths in the solver convention (boundary node = full end cell).
_dual_widths = _solver_dual_widths


def _yee_shapes(Nx: int, Ny: int, Nz: int) -> dict[str, tuple[int, int, int]]:
    return {
        "Ex": (Nx, Ny + 1, Nz + 1),
        "Ey": (Nx + 1, Ny, Nz + 1),
        "Ez": (Nx + 1, Ny + 1, Nz),
        "Hx": (Nx + 1, Ny, Nz),
        "Hy": (Nx, Ny + 1, Nz),
        "Hz": (Nx, Ny, Nz + 1),
    }


def _continued(values, a: int, at_low: bool, shared: bool, sign: float, on_centre: bool):
    """*values* continued across a symmetry plane along axis *a* (DD-154).

    Node-sampled quantities (``on_centre`` false) share the wall sample
    when the wall is a grid line; cell-sampled ones gain the cell the
    wall bisects, whose sample is the wall value — the neighbour's for
    an even quantity, zero for an odd one.
    """
    values = np.asarray(values)
    flipped = sign * np.flip(values, axis=a)
    if not on_centre:
        if shared:
            flipped = np.delete(flipped, -1 if at_low else 0, axis=a)
        parts = [flipped, values]
    elif shared:
        parts = [flipped, values]
    else:
        near = np.take(values, [0 if at_low else -1], axis=a)
        wall = near if sign > 0 else 0.0 * near
        parts = [flipped, wall, values]
    if not at_low:
        parts = parts[::-1]
    return np.concatenate(parts, axis=a)


class FieldState:
    """Electric and magnetic field on a grid, sampled at the Yee positions.

    Build one from physical component arrays (the constructor), from a
    function of position (:meth:`from_function`) or as zeros
    (:meth:`zeros`); analyses hand them out (an eigenmode's
    :meth:`~magnelio.solver.eigenmode_result.EigenmodeResult.field`).
    The arrays are copied on construction.

    Parameters
    ----------
    grid : GridLines
        The grid the samples live on.
    Ex, Ey, Ez : array_like
        Electric field [V/m] on the primal edges, Yee shapes (see the
        module table).
    Hx, Hy, Hz : array_like
        Magnetic field [A/m] on the dual edges through the primal faces.

    Examples
    --------
    >>> field = FieldState.from_function(
    ...     grid, E=lambda x, y, z: (0 * x, 0 * y, np.sin(np.pi * x / L))
    ... )
    >>> field.Ez.shape, field.positions("Ez")[0][:3]
    """

    @staticmethod
    def _check_grid(grid) -> GridLines:
        if not isinstance(grid, GridLines):
            raise TypeError(f"grid must be a magnelio.mesh.GridLines; got {type(grid).__name__}")
        return grid

    def __init__(self, grid: GridLines, Ex, Ey, Ez, Hx, Hy, Hz) -> None:
        self._grid = self._check_grid(grid)
        shapes = _yee_shapes(grid.Nx, grid.Ny, grid.Nz)
        arrays = {}
        for name, value in zip(_COMPONENTS, (Ex, Ey, Ez, Hx, Hy, Hz)):
            a = np.asarray(value)
            if a.shape != shapes[name]:
                raise ValueError(
                    f"{name} must have the Yee shape {shapes[name]} on this grid; got {a.shape}",
                )
            arrays[name] = a
        dtype = np.result_type(*(a.dtype for a in arrays.values()), np.float64)
        lengths = self._lengths()
        raw = {name: arrays[name].astype(dtype) * lengths[name] for name in _COMPONENTS}
        self._raw = FieldArrays(**raw)
        self._dual = None
        self._ops = None
        self._h_lead = 0.0

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def _from_raw(
        cls, grid: GridLines, raw: FieldArrays, dual=None, ops=None, h_lead: float = 0.0
    ) -> FieldState:
        """Wrap solver grid quantities without conversion (internal).

        *dual* names the dual widths the ``h`` samples were formed with
        when they differ from the solver convention on *grid* — a region
        cut from a larger grid (:func:`~magnelio.fields._interp._region_dual`).
        *ops* are the region's material operators
        (:class:`~magnelio.fields._operators.RegionOperators`) when the
        field can state its energy and flux; *h_lead* the lead [s] of
        the magnetic samples over the electric ones (half a step for
        the state of a march).
        """
        self = cls.__new__(cls)
        self._grid = grid
        self._raw = raw
        self._dual = dual
        self._ops = ops
        self._h_lead = float(h_lead)
        return self

    @classmethod
    def zeros(cls, grid: GridLines, dtype=float) -> FieldState:
        """A zero field on *grid*."""
        cls._check_grid(grid)
        raw = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz, dtype=dtype)
        return cls._from_raw(grid, raw)

    @classmethod
    def from_function(
        cls,
        grid: GridLines,
        *,
        E: Callable | None = None,
        H: Callable | None = None,
    ) -> FieldState:
        """Sample vector functions of position on the Yee positions.

        Parameters
        ----------
        grid : GridLines
        E, H : callable, optional
            ``fn(x, y, z) -> (fx, fy, fz)`` with *x*, *y*, *z* arrays of
            one shape (the sample positions of the component being
            filled) and each returned component broadcastable to it.
            The function is called once per component with that
            component's own positions; ``None`` leaves the field zero.
        """
        self = cls.zeros(grid)
        lengths = self._lengths()
        for group, fn in (("E", E), ("H", H)):
            if fn is None:
                continue
            if not callable(fn):
                raise TypeError(f"{group} must be callable; got {type(fn).__name__}")
            for k, axis in enumerate(_AXES):
                name = f"{group}{axis}"
                X, Y, Z = self._meshgrid(name)
                value = np.asarray(fn(X, Y, Z)[k])
                sample = np.broadcast_to(value, X.shape)
                if np.iscomplexobj(sample) and not np.iscomplexobj(self._raw.e_flat):
                    self._raw = FieldArrays(
                        **{c: getattr(self._raw, c).astype(complex) for c in _COMPONENTS},
                    )
                setattr(self._raw, name, sample * lengths[name])
        return self

    # ── grid and positions ───────────────────────────────────────────────

    @property
    def grid(self) -> GridLines:
        """The grid lines the samples refer to."""
        return self._grid

    @property
    def is_complex(self) -> bool:
        """Whether the samples are complex (a Bloch mode with a phase advance)."""
        return bool(np.iscomplexobj(self._raw.e_flat))

    def _nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self._grid
        return (
            np.asarray(g.x, dtype=float),
            np.asarray(g.y, dtype=float),
            np.asarray(g.z, dtype=float),
        )

    def _centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = self._nodes()
        return 0.5 * (x[:-1] + x[1:]), 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])

    def positions(self, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The 1-D coordinate vectors ``(x, y, z)`` of a component's samples.

        ``component(name)[i, j, k]`` sits at ``(x[i], y[j], z[k])``.
        """
        self._check_component(component)
        nodes, centres = self._nodes(), self._centres()
        group, axis = component[0], _AXES.index(component[1])
        # E lives on edges: centred along its own axis, on nodes across.
        # H lives on faces: on nodes along its own axis, centred across.
        on_centre = [(a == axis) if group == "E" else (a != axis) for a in range(3)]
        return tuple(centres[a] if on_centre[a] else nodes[a] for a in range(3))

    def _meshgrid(self, component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.meshgrid(*self.positions(component), indexing="ij")

    @property
    def cell_centres(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The 1-D cell-centre coordinates ``(xc, yc, zc)`` of :meth:`cell_centred`."""
        return self._centres()

    def _lengths(self) -> dict[str, np.ndarray]:
        """The edge lengths that turn a physical sample into a grid quantity."""
        g = self._grid
        dx, dy, dz = (np.asarray(d, dtype=float) for d in (g.dx, g.dy, g.dz))
        dual = getattr(self, "_dual", None)
        if dual is None:
            dxa, dya, dza = _dual_widths(dx), _dual_widths(dy), _dual_widths(dz)
        else:
            dxa, dya, dza = (np.asarray(d, dtype=float) for d in dual)
        return {
            "Ex": dx[:, None, None],
            "Ey": dy[None, :, None],
            "Ez": dz[None, None, :],
            "Hx": dxa[:, None, None],
            "Hy": dya[None, :, None],
            "Hz": dza[None, None, :],
        }

    # ── components ───────────────────────────────────────────────────────

    @staticmethod
    def _check_component(component: str) -> None:
        if component not in _COMPONENTS:
            raise KeyError(f"component must be one of {_COMPONENTS}; got {component!r}")

    def component(self, name: str) -> np.ndarray:
        """The physical samples of one component (a new array)."""
        self._check_component(name)
        raw = getattr(self._raw, name)
        if type(raw).__module__.partition(".")[0] == "cupy":
            raw = raw.get()
        return np.asarray(raw) / self._lengths()[name]

    @property
    def Ex(self) -> np.ndarray:
        """E_x [V/m] on the x-edges, shape ``(Nx, Ny+1, Nz+1)``."""
        return self.component("Ex")

    @property
    def Ey(self) -> np.ndarray:
        """E_y [V/m] on the y-edges, shape ``(Nx+1, Ny, Nz+1)``."""
        return self.component("Ey")

    @property
    def Ez(self) -> np.ndarray:
        """E_z [V/m] on the z-edges, shape ``(Nx+1, Ny+1, Nz)``."""
        return self.component("Ez")

    @property
    def Hx(self) -> np.ndarray:
        """H_x [A/m] on the x-faces, shape ``(Nx+1, Ny, Nz)``."""
        return self.component("Hx")

    @property
    def Hy(self) -> np.ndarray:
        """H_y [A/m] on the y-faces, shape ``(Nx, Ny+1, Nz)``."""
        return self.component("Hy")

    @property
    def Hz(self) -> np.ndarray:
        """H_z [A/m] on the z-faces, shape ``(Nx, Ny, Nz+1)``."""
        return self.component("Hz")

    # ── sampling ─────────────────────────────────────────────────────────

    def at(self, points) -> tuple[np.ndarray, np.ndarray]:
        """E and H at arbitrary points, interpolated per component.

        Each component is interpolated trilinearly between its own
        samples, so the staggering is honoured exactly; within half a
        cell of the domain boundary — where a component has no sample
        on one side — the interpolation is continued linearly.

        Parameters
        ----------
        points : array_like, shape ``(n, 3)`` or ``(3,)``
            Positions [m] inside the grid's bounding box.

        Returns
        -------
        E, H : np.ndarray, shape ``(n, 3)``
            The field vectors at the points.
        """
        from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3); got {pts.shape}")
        x, y, z = self._nodes()
        lo, hi = (x[0], y[0], z[0]), (x[-1], y[-1], z[-1])
        tol = 1e-9 * max(hi[a] - lo[a] for a in range(3))
        if np.any(pts < np.array(lo) - tol) or np.any(pts > np.array(hi) + tol):
            raise ValueError("points must lie inside the grid's bounding box")
        out = {}
        for name in _COMPONENTS:
            values = self.component(name)
            interp = RegularGridInterpolator(
                self.positions(name),
                values,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            out[name] = interp(pts)
        E = np.stack([out["Ex"], out["Ey"], out["Ez"]], axis=-1)
        H = np.stack([out["Hx"], out["Hy"], out["Hz"]], axis=-1)
        return E, H

    def cell_centred(self, components=None, corners=None) -> dict[str, np.ndarray]:
        """Components averaged onto the cell centres (:attr:`cell_centres`).

        Parameters
        ----------
        components : sequence of str, optional
            Subset of the six names; default all.
        corners : tuple of tuple, optional
            Two opposite corners [m] of a sub-box; default the whole grid.

        Returns
        -------
        dict[str, np.ndarray]
            ``{name: array}`` with shape ``(nx, ny, nz)`` of the box.
        """
        from magnelio.monitors.base import resolve_region  # noqa: PLC0415

        names = list(_COMPONENTS if components is None else components)
        for name in names:
            self._check_component(name)
        region = resolve_region(corners, self._grid)
        return _interp_to_cell_centres(
            self._raw, names, region.ix, region.iy, region.iz, self._grid, dual=self._dual
        )

    # ── arithmetic ───────────────────────────────────────────────────────

    def scaled(self, factor) -> FieldState:
        """A copy multiplied by a (possibly complex) scalar."""
        raw = FieldArrays(**{c: getattr(self._raw, c) * factor for c in _COMPONENTS})
        return self._like(raw)

    def real(self) -> FieldState:
        """The real part (the field of a complex mode at its zero-phase instant)."""
        if not self.is_complex:
            return self
        raw = FieldArrays(**{c: np.real(getattr(self._raw, c)) for c in _COMPONENTS})
        return self._like(raw)

    def _like(self, raw: FieldArrays, grid: GridLines | None = None, dual=None, ops=None):
        """A field on the same grid with the same operators and lead, new samples."""
        return type(self)._from_raw(
            grid if grid is not None else self._grid,
            raw,
            dual=dual if grid is not None else self._dual,
            ops=ops if grid is not None else self._ops,
            h_lead=self._h_lead,
        )

    # ── energy and flux (DD-260) ─────────────────────────────────────────

    def energy(self) -> float:
        """The stored energy [J] of the field, full-model booking.

        ``½ eᵀMε e + ½ hᵀMμ h`` on the field's own samples, with the
        material operators of the region the field carries — a
        monitor's frame does (live or read back from a project), an
        eigenmode or a field assembled by hand does not, and a
        ``RuntimeError`` says so.  For a frame of a march, whose
        magnetic samples lead the electric ones by half a step, the
        magnetic term is the one the leapfrog conserves —
        ``h(t−dt/2)·Mμ·h(t+dt/2)``, formed from the frame alone through
        the discrete Faraday law — so a recording of the whole domain
        reproduces the run's energy trace.  A complex frame is read as
        an RMS phasor — a spectrum's frames per 1 W CW are — and gives
        the time-averaged energy.  Where
        the region is cut out of the domain, the dual patches on its
        boundary count with the half that lies inside, so the energies
        of two adjoining regions add up to that of their union.  A
        model solved on a symmetry-reduced domain reports the joules of
        the whole model.

        Returns
        -------
        float
        """
        from magnelio.fields._operators import energy, no_operators  # noqa: PLC0415

        if self._ops is None:
            raise no_operators("this field")
        return energy(self._raw, self._ops, self._grid, self._dual, self._h_lead)

    def flux(self, normal: str, position: float) -> float:
        """Poynting flux [W] through the field's cross-section at a plane.

        The FIT pairing ``Σ e·h`` of the samples on the plane — the
        identity :class:`~magnelio.monitors.MonitorFluxTime` records
        during a march, here on a recorded frame, over the region's
        extent — positive along the axis; the plane snaps to the
        nearest grid node; the pairing takes the electric samples on
        that node plane and the magnetic ones in the cell above it, as
        the flux monitor does.  A complex frame is read as an RMS
        phasor — a spectrum's frames per 1 W CW are — and gives the
        time-averaged real power ``Re Σ e·h*``, so on a matched line
        a spectrum's flux is the transmitted watts per watt incident.
        Needs the region's operators like
        :meth:`energy`; a model solved on a symmetry-reduced domain
        reports the watts of the whole cross-section.

        Parameters
        ----------
        normal : {"x", "y", "z"}
            Normal axis of the plane.
        position : float
            Position [m] along that axis.

        Returns
        -------
        float
        """
        from magnelio.fields._operators import flux, no_operators, plane_index  # noqa: PLC0415

        if self._ops is None:
            raise no_operators("this field")
        if normal not in _AXES:
            raise ValueError(f"normal must be 'x', 'y' or 'z'; got {normal!r}")
        axis = _AXES.index(normal)
        return flux(
            self._raw,
            self._ops,
            self._grid,
            self._dual,
            axis,
            plane_index(self._grid, axis, position),
        )

    # ── symmetry ─────────────────────────────────────────────────────────

    def mirrored(self, *mirrors) -> FieldState:
        """The field continued across symmetry planes, on the extended grid.

        A model declared with symmetry planes is solved on the reduced
        domain; this returns the field of the whole structure — the
        simulated part plus its mirror images — with every component
        continued by its own rule: across a magnetic (PMC) plane E
        continues like a polar vector (normal component odd, tangential
        even) and H like a pseudovector, across an electric (PEC) plane
        the roles swap.  The staggering is kept: a sample sitting on
        the wall appears once, and the cell a magnetic wall bisects
        gets its centre sample from the wall value (zero for an odd
        component, the neighbour's value for an even one).

        Parameters
        ----------
        *mirrors : MirrorSpec
            One plane each (:class:`magnelio.post._symmetry.MirrorSpec`:
            axis, wall position, ``"PEC"``/``"PMC"``, side).  A plane
            must bound the grid: on the wall for PEC, half the boundary
            cell outside it for PMC.
        """
        out = self
        for spec in mirrors:
            out = out._mirrored_once(spec)
        return out

    def _mirrored_once(self, spec) -> FieldState:
        from magnelio.post._symmetry import mirror_sign  # noqa: PLC0415

        a = int(spec.axis)
        nodes = list(self._nodes())
        n = nodes[a]
        tol = 1e-9 * abs(float(n[-1] - n[0]))
        edge = float(n[0] if spec.at_low else n[-1])
        outside = (edge - spec.wall) if spec.at_low else (spec.wall - edge)
        if outside < -tol:
            raise ValueError(
                f"the {spec.kind} plane at {_AXES[a]} = {spec.wall:g} m lies inside the grid "
                f"({edge:g} m is its {'low' if spec.at_low else 'high'} end)"
            )
        reflected = 2.0 * float(spec.wall) - n[::-1]
        shared = outside <= tol  # the wall is a grid line: its samples appear once
        if spec.at_low:
            nodes[a] = np.concatenate([reflected[:-1] if shared else reflected, n])
        else:
            nodes[a] = np.concatenate([n, reflected[1:] if shared else reflected])
        grid = GridLines(*nodes)

        def on_centre_of(name: str) -> bool:
            group, caxis = name[0], _AXES.index(name[1])
            return (caxis == a) if group == "E" else (caxis != a)

        comps = {}
        for name in _COMPONENTS:
            group, caxis = name[0], _AXES.index(name[1])
            sign = mirror_sign(group, caxis, a, spec.kind)
            comps[name] = _continued(
                self.component(name), a, spec.at_low, shared, sign, on_centre_of(name)
            )
        if self._ops is None:
            return FieldState(grid, **comps)

        # The operators continue like even quantities, the dual widths
        # with the nodes; the mirrored end takes the kind of the end it
        # mirrors, and the plane crossed is no symmetry plane any more.
        from magnelio.fields._operators import RegionOperators  # noqa: PLC0415

        ops = self._ops
        dual = list(self._lengths_dual())
        dual[a] = _continued(dual[a], 0, spec.at_low, shared, 1.0, False)
        m_eps = {
            c: _continued(ops.m_eps[c], a, spec.at_low, shared, 1.0, on_centre_of(c))
            for c in ops.m_eps
        }
        m_mu = {
            c: _continued(ops.m_mu[c], a, spec.at_low, shared, 1.0, on_centre_of(c))
            for c in ops.m_mu
        }
        ends = list(ops.ends)
        lo, hi = ends[a]
        ends[a] = (hi, hi) if spec.at_low else (lo, lo)
        face = f"{_AXES[a]}{'min' if spec.at_low else 'max'}"
        new_ops = RegionOperators(
            m_eps=m_eps,
            m_mu=m_mu,
            ends=tuple(ends),
            symmetry=tuple(f for f in ops.symmetry if f != face),
        )
        out = FieldState.zeros(grid)
        out._dual = tuple(dual)
        lengths = out._lengths()
        raw = FieldArrays(**{c: np.asarray(comps[c]) * lengths[c] for c in _COMPONENTS})
        return self._like(raw, grid=grid, dual=tuple(dual), ops=new_ops)

    def _lengths_dual(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The dual widths of the h samples along each axis."""
        if self._dual is not None:
            return tuple(np.asarray(d, dtype=float) for d in self._dual)
        g = self._grid
        return tuple(_dual_widths(np.asarray(d, dtype=float)) for d in (g.dx, g.dy, g.dz))

    # ── plotting ─────────────────────────────────────────────────────────

    def plot(
        self,
        component: str = "E",
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        ax=None,
        scale_mm: bool = True,
        cmap: str | None = None,
        geometry=None,
        flip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        density: int = 20,
        normalize_arrows: bool = False,
        threshold: float = 0.02,
        title: str | None = None,
        unit: str | None = None,
    ):
        """Plot the field on a slice plane.

        The staggered components are averaged onto cell centres and
        rendered on the plane selected by *normal* and *position*.  A
        complex field is drawn as the real snapshot at the instant of
        maximum energy on the slice.

        Parameters
        ----------
        component : str
            ``"E"`` or ``"H"`` for a vector plot / vector magnitude;
            ``"Ex"``, ``"Hy"``, … for a single component (scalar only).
        normal : {"x", "y", "z"}, optional
            Normal axis of the slice plane; default the thinnest axis.
        position : float
            Plane position along *normal* [m]; snapped to the nearest
            cell-centre plane.
        plot_type : str
            ``"vector"``, ``"color"`` or ``"contour"``.
        ax : matplotlib.axes.Axes, optional
        scale_mm : bool
            Axis coordinates in mm instead of m.
        cmap : str, optional
        geometry : GeometryModel, optional
            Cross-section overlay of the slice plane.
        flip : bool
            Swap horizontal and vertical axes.
        vmin, vmax : float, optional
            Colour limits (scalar) or arrow clipping (vector).
        density : int
            Target arrows per axis (vector).
        normalize_arrows : bool
            Unit-length arrows, colour = magnitude (vector).
        threshold : float
            Suppress arrows below this fraction of the peak (vector).
        title : str, optional
            Plot title; default names the component and the plane.
        unit : str, optional
            Colour-bar unit label; default ``"V/m"`` / ``"A/m"``.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from magnelio.monitors.base import (  # noqa: PLC0415
            _AXES as _AX,
        )
        from magnelio.monitors.base import (  # noqa: PLC0415
            _resolve_component,
            plane_slab_halfwidth,
            resolve_plane_view,
            resolve_region,
        )
        from magnelio.post.plot_field import (  # noqa: PLC0415
            CrossSectionOverlay,
            plot_field_scalar,
            plot_field_vector,
        )
        from magnelio.solver.eigenmode_result import _real_snapshot  # noqa: PLC0415

        grid = self._grid
        region = resolve_region(None, grid)
        pv = resolve_plane_view(region, normal, position)
        (i0, c0), (i1, c1) = pv.free

        is_magnitude = component in ("E", "H", "|E|", "|H|")
        is_field_group = component in ("E", "H")
        field_group = component.strip("|")[:1]
        if not is_magnitude and component not in _COMPONENTS:
            raise KeyError(f"component must be E/H (or |E|/|H|) or one of {_COMPONENTS}.")
        comps = [f"{field_group}{a}" for a in _AX] if is_magnitude else [component]

        slabs = [region.ix, region.iy, region.iz]
        if pv.slice_index is not None:
            base = slabs[pv.normal_idx].start
            slabs[pv.normal_idx] = slice(base + pv.slice_index, base + pv.slice_index + 1)
        data = _interp_to_cell_centres(self._raw, comps, *slabs, grid, dual=self._dual)
        data = {c: np.squeeze(a, axis=pv.normal_idx) for c, a in data.items()}
        data = _real_snapshot(data)

        overlay = None
        if geometry is not None:
            overlay = CrossSectionOverlay(
                geometry=geometry,
                normal=_AX[pv.normal_idx],
                position=pv.normal_pos,
                slab=plane_slab_halfwidth(grid, pv.normal_idx, pv.normal_pos),
            )

        pos_txt = (
            f"{_AX[pv.normal_idx]}={pv.normal_pos * 1e3:.3g} mm"
            if scale_mm
            else f"{_AX[pv.normal_idx]}={pv.normal_pos:.3g} m"
        )
        if unit is None:
            unit = "V/m" if field_group == "E" else "A/m"

        if plot_type == "vector":
            if not is_field_group:
                raise ValueError("Vector plots need component='E' or 'H'.")
            return plot_field_vector(
                c0,
                c1,
                data[f"{field_group}{_AX[i0]}"],
                data[f"{field_group}{_AX[i1]}"],
                w=data[f"{field_group}{_AX[pv.normal_idx]}"],
                xlabel=_AX[i0],
                ylabel=_AX[i1],
                wlabel=_AX[pv.normal_idx],
                title=title if title is not None else f"{field_group}-field, {pos_txt}",
                clabel=f"|{field_group}| ({unit})",
                ax=ax,
                scale_mm=scale_mm,
                cmap=cmap or "viridis",
                density=density,
                normalize_arrows=normalize_arrows,
                vmax=vmax,
                threshold=threshold,
                flip=flip,
                geometry=overlay,
            )

        vals = _resolve_component(data, field_group if is_magnitude else component)
        if is_magnitude:
            label = f"|{field_group}|"
            effective_cmap = cmap or "viridis"
            sym = False
            if vmin is None:
                vmin = 0.0
        else:
            label = component
            effective_cmap = cmap or "RdBu_r"
            sym = True
        return plot_field_scalar(
            c0,
            c1,
            vals,
            xlabel=_AX[i0],
            ylabel=_AX[i1],
            title=title if title is not None else f"{label}, {pos_txt}",
            clabel=f"{label} ({unit})",
            ax=ax,
            scale_mm=scale_mm,
            cmap=effective_cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric=sym,
            plot_type=plot_type,
            flip=flip,
            geometry=overlay,
        )

    def show(self, component: str = "E", **kwargs):
        """Interactive 3D view of the field on a cutting plane.

        The geometry viewer with the field laid on its cut: the exposed
        cell layer as a coloured sheet (``"E"``/``"H"`` magnitude, or one
        signed component such as ``"Ez"``) with arrows for a field
        group, walked through the volume by the position slider.  See
        :func:`magnelio.plots.show_field` for the arguments — the plane
        (``normal``, ``position``), ``geometry`` and ``mesh`` overlays,
        ``phase`` for a complex field, and the rendering ``mode``.
        """
        from magnelio.post.field_3d import show_field  # noqa: PLC0415

        return show_field(self, component, **kwargs)

    def __repr__(self) -> str:
        g = self._grid
        kind = "complex" if self.is_complex else "real"
        return f"FieldState(grid={g.Nx}x{g.Ny}x{g.Nz} cells, {kind})"


__all__ = ["FieldState"]
