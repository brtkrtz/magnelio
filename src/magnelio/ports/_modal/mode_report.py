"""User-facing mode reports for modal ports (WP5.1, finding F5).

:func:`build_modal_port` solves the port's 2D mode problem and attaches
a :class:`PortOperatorReport` (scalar z_line / cutoff numbers) to the operator —
but the operator itself is an internal solver object.  This module
wraps one built operator into a :class:`PortReport`: a small,
picklable-ish view object exposing the per-port impedance/cutoff
numbers plus one :class:`ModeReport` per solved mode, each of which can
evaluate the closed-form frequency relations (``z_modal``, ``z_wave``,
``gamma``) and plot its transverse profile on the port plane — all
without running a time-domain simulation.

The analysis obtains these via :meth:`AnalysisScatteringTD.solve_ports`;
component-level scripts can call :meth:`PortReport.from_operator` on any
operator they built by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from magnelio.ports._modal.discrete import DiscreteMode
from magnelio.ports._modal.mode import ModeType
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.port_report import PortOperatorReport
from magnelio.post._symmetry import mirror_extend, mirror_sign, mirror_spec_for_face

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ModeReport:
    """One solved port mode, inspectable without a TD run.

    Attributes
    ----------
    port_name : str
        Label of the port the mode belongs to.
    name : str
        Mode identifier from the 2D solver (e.g. ``"TE10"``,
        ``"TEM_lap00"``).
    mode_type : ModeType
        TEM / TE / TM classification.
    f_cutoff : float
        Cut-off frequency [Hz]; ``0.0`` for TEM modes.
    z_line : float or None
        Frequency-independent line impedance ``Z₀ = 2P/(I·I*)`` for
        multi-conductor (TEM/QTEM) modes; ``None`` for hollow-pipe
        modes, whose reference impedance is the frequency-dependent
        wave impedance (use :meth:`z_modal`).
    epsilon_eff : float or None
        Effective relative permittivity of a line mode — the ``ε_eff``
        of a quasi-TEM mode (``C'/C'_0``, one value per mode of a
        coupled line: even and odd travel at different speeds), the
        filling ``ε_r`` of a homogeneous TEM mode; ``None`` for
        hollow-pipe modes.  The phase velocity is ``c / √ε_eff``.
    termination : {'dtbc', 'mur'} or None
        How the channel closes the domain at the port plane.
        ``'dtbc'`` is the exact discrete transparent boundary
        condition, reflection-free to roundoff; ``'mur'`` is the
        first-order absorber, a reflection floor of order −30 dB.  A
        channel qualifies for the exact one when the feed cross-section
        is a uniform discrete chain; ``chain_spread`` is that
        measurement.  ``None`` where the port reports no per-channel
        termination (lumped ports carry no modes at all).
    chain_spread : float or None
        Weighted RMS spread of the per-pair modal Courant number over
        the feed cross-section — 0 on a perfectly uniform chain, and
        the quantity ``chain_floor_db`` turns into a reflection.  A
        value just above the acceptance threshold means a
        cross-section that was meant to be uniform and was not, and
        warns; a large one is an inhomogeneous line that never
        qualified for the scalar chain.  ``None``
        when the test does not apply: a mode with a
        closed-form field evaluator is ineligible by construction, and
        so is every channel of a port whose feed masses fail the slab
        consistency check upstream (both warn on their own).
    """

    port_name: str
    name: str
    mode_type: ModeType
    f_cutoff: float
    z_line: float | None
    _discrete: DiscreteMode = field(repr=False)
    _plane: PortPlane = field(repr=False)
    # In-plane symmetry planes the port window is cut by (DD-154), so
    # plot() can show the full cross-section instead of the solved half.
    _mirrors: tuple = field(default=(), repr=False)
    # Dual edge lengths of the plane's H faces, needed to turn the H
    # profile's dual voltages back into A/m (see _field_profiles).
    _h_dual_lengths: tuple = field(default=(), repr=False)
    epsilon_eff: float | None = None
    termination: str | None = None
    chain_spread: float | None = None

    @property
    def chain_floor_db(self) -> float | None:
        """Reflection the feed cross-section's non-uniformity can cost [dB].

        An upper bound, not an estimate.  The exact termination is
        built for the weighted-mean chain, so a spread across the
        cross-section leaves a residual mismatch; measured through the
        production chain, the worst case rises linearly with the
        spread at about a seventh of it, and this bound takes the
        coefficient as one.  A channel terminated by the exact
        boundary contributes at most this much reflection on top of
        whatever else limits it — compare it against the floor the
        port itself reaches.

        ``None`` where no spread was measured (see ``chain_spread``),
        and meaningless for a channel on the first-order absorber,
        whose floor is set by the absorber instead.
        """
        if self.chain_spread is None or self.chain_spread <= 0.0:
            return None
        return 20.0 * math.log10(self.chain_spread)

    def z_modal(self, f: float) -> complex:
        """Power-wave reference impedance at frequency ``f`` [Hz]."""
        return self._discrete.mode.z_modal(2.0 * math.pi * f)

    def z_wave(self, f: float) -> complex:
        """Modal wave impedance at frequency ``f`` [Hz]."""
        return self._discrete.mode.z_wave(2.0 * math.pi * f)

    def gamma(self, f: float) -> complex:
        """Propagation constant ``γ = α + jβ`` at frequency ``f`` [Hz]."""
        return self._discrete.mode.gamma(2.0 * math.pi * f)

    def _field_profiles(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        """Physical field at the edge midpoints, V/m or A/m.

        The 2D mode solvers return FIT *grid quantities*, not field
        samples: the primal profile is the edge voltage
        ``ê = E · l_primal`` (the gradient behind ``ê = -∇φ`` is
        topological), the dual profile the face voltage
        ``ĥ = H · l_dual``.  Both scale with a per-edge length, so on a
        graded mesh reading them as a field tilts every cell-centre
        vector and biases its magnitude — with an extra bias wherever
        the conductor contour forces a locally different spacing.

        Analytical mode families sample their closed-form evaluator
        directly and already carry V/m and A/m.
        """
        dm = self._discrete
        raw = (dm.e_u_profile, dm.e_v_profile) if field == "E" else (dm.h_u_profile, dm.h_v_profile)
        if dm.mode.field_evaluator is not None:
            return np.asarray(raw[0], dtype=float), np.asarray(raw[1], dtype=float)
        lengths = (
            (self._plane.u_edge_lengths, self._plane.v_edge_lengths)
            if field == "E"
            else self._h_dual_lengths
        )
        if len(lengths) != 2:
            return np.asarray(raw[0], dtype=float), np.asarray(raw[1], dtype=float)
        out = []
        for values, length in zip(raw, lengths):
            values = np.asarray(values, dtype=float)
            length = np.asarray(length, dtype=float)
            out.append(np.divide(values, length, out=np.zeros_like(values), where=length > 0.0))
        return out[0], out[1]

    def plot(
        self,
        *,
        field: str = "E",
        ax: "matplotlib.axes.Axes | None" = None,
        density: int = 20,
        normalize_arrows: bool = True,
        threshold: float = 0.0,
        flip: bool = False,
        scale_mm: bool = True,
        title: str | None = None,
        geometry=None,
    ) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
        """Quiver plot of the transverse mode profile on the port plane.

        The discrete edge profiles (the B-orthonormal basis vectors the
        FIT operator actually injects and projects with) are averaged
        from their staggered edge positions onto the port-plane cell
        centres and rendered via
        :func:`~magnelio.post.plot_field.plot_field_vector`.

        Parameters
        ----------
        field : {"E", "H"}, default "E"
            Which transverse field profile to draw.
        ax : matplotlib.axes.Axes, optional
            Target axes; a new figure is created when omitted.
        density : int, default 20
            Number of arrows along the longer axis of the cross-section;
            the shorter one gets the count that keeps the spacing equal.
            The raster is deliberately independent of the computational
            grid, so a locally refined region does not show up as a
            cluster of arrows — but it also does not gain any.  Raise
            this to read a feature that the default spacing steps over,
            such as the field in a thin gap.
        normalize_arrows : bool, default True
            Unit-length arrows with magnitude encoded in colour
            (port-mode style).
        threshold : float, default 0.0
            Suppress arrows below this fraction of the peak magnitude.
        flip : bool, default False
            Swap the horizontal and vertical plot axes.
        scale_mm : bool, default True
            Axis coordinates in mm instead of m.
        title : str, optional
            Axes title; default names port, mode, and field.
        geometry : GeometryModel, optional
            Geometry model for a cross-section overlay of the port
            plane (conductors filled, air regions as dashed outlines).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from magnelio.post.plot_field import CrossSectionOverlay, plot_field_vector

        if field == "E":
            prof_u, prof_v = self._field_profiles("E")
            comp_u = _edge_grid(prof_u, self._plane.u_edge_uv)
            comp_v = _edge_grid(prof_v, self._plane.v_edge_uv)
            # E_u lives at (u-centre, v-node): average along v.
            # E_v lives at (u-node, v-centre): average along u.
            u_cc = _avg_nonzero(comp_u[0][:, :-1], comp_u[0][:, 1:])
            v_cc = _avg_nonzero(comp_v[0][:-1, :], comp_v[0][1:, :])
            valid = _live_cells(comp_u[0], comp_v[0])
            uc, vc = comp_u[1], comp_v[2]
            # E_u resolves the v-nodes, E_v the u-nodes.
            u_node_axis = 1
            u_nodes, v_nodes = comp_v[1], comp_u[2]
        elif field == "H":
            # H_u is co-located with the v-edges, H_v with the u-edges.
            prof_u, prof_v = self._field_profiles("H")
            comp_u = _edge_grid(prof_u, self._plane.v_edge_uv)
            comp_v = _edge_grid(prof_v, self._plane.u_edge_uv)
            u_cc = _avg_nonzero(comp_u[0][:-1, :], comp_u[0][1:, :])
            v_cc = _avg_nonzero(comp_v[0][:, :-1], comp_v[0][:, 1:])
            valid = _live_cells(comp_v[0], comp_u[0])
            uc, vc = comp_v[1], comp_u[2]
            u_node_axis = 0
            u_nodes, v_nodes = comp_u[1], comp_v[2]
        else:
            raise ValueError(f"field must be 'E' or 'H'; got {field!r}")

        uc, vc, u_cc, v_cc, valid = _extend_to_window(
            uc,
            vc,
            u_cc,
            v_cc,
            valid,
            comp_u[0],
            comp_v[0],
            u_node_axis,
            (float(u_nodes[0]), float(u_nodes[-1])),
            (float(v_nodes[0]), float(v_nodes[-1])),
        )

        face = self._plane.face
        u_name = _AXIS_NAMES[face.u_axis]
        v_name = _AXIS_NAMES[face.v_axis]
        uc, vc, u_cc, v_cc, valid = _apply_mirrors(
            self._mirrors, face, field, uc, vc, u_cc, v_cc, valid
        )
        if title is None:
            title = f"{self.port_name}: {self.name} ({self.mode_type.value}) {field} profile"
            if self._mirrors:
                title += " — full model"

        overlay = None
        if geometry is not None:
            # Slice half a boundary cell inward: exactly on the bbox
            # face the OCC section is tangent to the solids' end faces
            # (ill-defined), and a port requires an extruded cross
            # section there anyway.  Half the faces order (u, v)
            # descending (u x v points inward) — swap_axes corrects the
            # ascending-axis convention of the cross-section renderer.
            overlay = CrossSectionOverlay(
                geometry=geometry,
                normal=_AXIS_NAMES[face.normal_axis],
                position=self._plane.coordinate + face.inward_sign * 0.5 * self._plane.normal_dx,
                swap_axes=face.u_axis > face.v_axis,
                mirrors=_overlay_mirrors(self._mirrors, face),
            )

        return plot_field_vector(
            uc,
            vc,
            u_cc,
            v_cc,
            valid=valid,
            xlabel=u_name,
            ylabel=v_name,
            title=title,
            ax=ax,
            scale_mm=scale_mm,
            density=density,
            normalize_arrows=normalize_arrows,
            threshold=threshold,
            flip=flip,
            geometry=overlay,
        )


def _extend_to_window(
    uc: np.ndarray,
    vc: np.ndarray,
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    valid: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
    u_node_axis: int,
    u_bounds: tuple[float, float],
    v_bounds: tuple[float, float],
):
    """Grow the cell-centre picture out to the port-window boundary.

    Destaggering lands both components on cell centres, so the picture
    spans centre to centre and loses *half a cell* on each of the four
    sides — invisible on a uniform mesh, up to a tenth of the frame
    where the mesh is graded, and a seam in the middle of a mirrored
    full-model plot.

    Nothing is missing from the solution: each component is staggered
    along one axis only and therefore carries a genuine value on the two
    window boundary lines of its *other* axis.  Those go in as they are;
    the partner component, which has no sample out there, is carried out
    by its nearest interior value — so a boundary arrow's direction is
    exact in one component and first-order in the other.

    An added line is valid exactly where the interior line it continues
    is: what decides whether there is anything to draw out there is
    whether the neighbourhood is metal, not how large the field happens
    to be.  Reading the genuine component instead — blank it out where
    it vanishes — confuses *on* a conductor with *in* one: an electric
    wall forces the tangential half to zero and leaves the normal half
    at its maximum, and the frame of a port's 2D mode problem is such a
    wall all the way round, so that rule dropped the outermost ring of
    arrows on every port.  A boundary arrow is therefore exact in its
    tangential component (identically zero on a wall) and first-order in
    its normal one, which draws it standing perpendicular on the wall —
    the boundary condition made visible.  A window reaching into a
    conductor still stays blank: there the interior line is invalid too.
    """
    nu, nv = u_cc.shape

    def _pad(a):
        out = np.empty((nu + 2, nv + 2), dtype=a.dtype)
        out[1:-1, 1:-1] = a
        return out

    ue, ve, va = _pad(u_cc), _pad(v_cc), _pad(valid)
    # Genuine boundary lines where the component resolves the nodes,
    # nearest-value continuation for its partner.
    if u_node_axis == 1:
        genuine_v_edge = (grid_u[:, 0], grid_u[:, -1])  # u-component on the v-bounds
        genuine_u_edge = (grid_v[0, :], grid_v[-1, :])  # v-component on the u-bounds
        ue[1:-1, 0], ue[1:-1, -1] = genuine_v_edge
        ve[0, 1:-1], ve[-1, 1:-1] = genuine_u_edge
        ue[0, 1:-1], ue[-1, 1:-1] = u_cc[0, :], u_cc[-1, :]
        ve[1:-1, 0], ve[1:-1, -1] = v_cc[:, 0], v_cc[:, -1]
    else:
        genuine_u_edge = (grid_u[0, :], grid_u[-1, :])
        genuine_v_edge = (grid_v[:, 0], grid_v[:, -1])
        ue[0, 1:-1], ue[-1, 1:-1] = genuine_u_edge
        ve[1:-1, 0], ve[1:-1, -1] = genuine_v_edge
        ue[1:-1, 0], ue[1:-1, -1] = u_cc[:, 0], u_cc[:, -1]
        ve[0, 1:-1], ve[-1, 1:-1] = v_cc[0, :], v_cc[-1, :]
    va[1:-1, 0], va[1:-1, -1] = valid[:, 0], valid[:, -1]
    va[0, 1:-1], va[-1, 1:-1] = valid[0, :], valid[-1, :]
    for arr in (ue, ve):
        arr[0, 0], arr[0, -1] = arr[1, 0], arr[1, -1]
        arr[-1, 0], arr[-1, -1] = arr[-2, 0], arr[-2, -1]
    va[0, 0], va[0, -1] = va[0, 1] & va[1, 0], va[0, -2] & va[1, -1]
    va[-1, 0], va[-1, -1] = va[-1, 1] & va[-2, 0], va[-1, -2] & va[-2, -1]
    uc = np.concatenate([[u_bounds[0]], uc, [u_bounds[1]]])
    vc = np.concatenate([[v_bounds[0]], vc, [v_bounds[1]]])
    return uc, vc, ue, ve, va


def _live_cells(grid_u: np.ndarray, grid_v: np.ndarray) -> np.ndarray:
    """Cells the mode profile actually lives in.

    *grid_u* is the edge grid staggered along v (one extra column),
    *grid_v* the one staggered along u.  A cell whose four bounding
    edges are all exact ``0.0`` is buried in a conductor: it carries no
    solved degree of freedom, so it must not enter the plot's
    interpolation stencil at all.  Everything else is live, including
    the partially filled cells at the conductor contour.
    """
    live_u = grid_u != 0.0
    live_v = grid_v != 0.0
    return (live_u[:, :-1] | live_u[:, 1:]) | (live_v[:-1, :] | live_v[1:, :])


def _overlay_mirrors(mirrors: tuple, face) -> tuple:
    """Mirror specs in the ``CrossSectionOverlay`` (slot, wall, at_low) form.

    The overlay indexes its in-plane axes in *ascending world order*,
    not in the port plane's ``(u, v)`` order — its ``swap_axes`` flag
    already carries the difference.
    """
    first = min(face.u_axis, face.v_axis)
    return tuple((0 if m.axis == first else 1, m.wall, m.at_low) for m in mirrors)


def _apply_mirrors(mirrors: tuple, face, field: str, uc, vc, u_cc, v_cc, valid):
    """Extend the plotted arrays across the port window's symmetry planes.

    Each plane doubles the picture along its axis; the component signs
    follow the usual PEC/PMC continuation rules, and the live-cell mask
    is carried along so a mirrored conductor stays blank.
    """
    for spec in mirrors:
        if spec.axis not in (face.u_axis, face.v_axis):
            continue  # normal-axis plane: does not extend a transverse profile
        arr_axis = 0 if spec.axis == face.u_axis else 1
        coords = uc if arr_axis == 0 else vc
        new_c, u_cc = mirror_extend(
            coords, u_cc, spec, arr_axis, mirror_sign(field, face.u_axis, spec.axis, spec.kind)
        )
        _, v_cc = mirror_extend(
            coords, v_cc, spec, arr_axis, mirror_sign(field, face.v_axis, spec.axis, spec.kind)
        )
        _, mask = mirror_extend(coords, valid.astype(float), spec, arr_axis, 1.0)
        valid = mask > 0.5
        if arr_axis == 0:
            uc = new_c
        else:
            vc = new_c
    return uc, vc, u_cc, v_cc, valid


def resolve_port_mirrors(plane: PortPlane, mesh) -> tuple:
    """In-plane symmetry planes the port window is cut by (DD-154).

    Only planes whose normal lies *in* the port plane extend the
    picture — a symmetry plane parallel to the port face is the face
    itself and mirrors the structure along the propagation direction,
    which the transverse profile does not show.  A plane counts only
    when the window actually reaches it; a sub-face window stopping
    short of the wall would otherwise get a detached mirror image.
    """
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        symmetry_entries,
    )

    bc = getattr(mesh, "boundary_conditions", None)
    sym = symmetry_entries(bc)
    if not sym:
        return ()
    types = bc_type_entries(bc)
    grid = mesh.grid
    axis_nodes = (grid.x, grid.y, grid.z)
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    windows = {
        plane.face.u_axis: plane.u_node_window,
        plane.face.v_axis: plane.v_node_window,
    }
    out = []
    for face_name in sorted(sym):
        axis = _AXIS_NAMES.index(face_name[0])
        if axis not in windows:
            continue
        lo, hi = windows[axis]
        at_low = face_name.endswith("min")
        if at_low and lo != 0:
            continue
        if not at_low and hi != n_cells[axis]:
            continue
        out.append(mirror_spec_for_face(face_name, types[face_name], axis_nodes[axis]))
    return tuple(out)


def _avg_nonzero(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average two staggered neighbours, counting only nonzero ones.

    Mode profiles carry exact ``0.0`` on every non-DOF edge (inside or
    on a conductor).  A plain two-point average halves the magnitude
    and rotates the direction of every cell-centre vector whose
    stencil touches a conductor; averaging only the live contributors
    removes that bias while leaving interior cells (both nonzero)
    untouched.
    """
    n = (a != 0).astype(float) + (b != 0).astype(float)
    return np.where(n > 0, (a + b) / np.maximum(n, 1.0), 0.0)


def _edge_grid(
    values: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter flat edge samples onto their regular (u, v) tensor grid.

    The port plane spans the full bbox face, so the edge midpoints form
    a complete tensor grid whose coordinates were computed once from
    the shared node/centre arrays — ``np.unique`` therefore matches
    them exactly.  Returns ``(grid, u_coords, v_coords)``.
    """
    us = np.unique(uv[:, 0])
    vs = np.unique(uv[:, 1])
    grid = np.empty((us.size, vs.size), dtype=float)
    grid[np.searchsorted(us, uv[:, 0]), np.searchsorted(vs, uv[:, 1])] = values
    return grid, us, vs


@dataclass(frozen=True)
class PortReport:
    """Per-port mode-solution report (no TD run required).

    Attributes
    ----------
    name : str
        Port label.
    modes : tuple[ModeReport, ...]
        One entry per solved mode, in cut-off-ascending solver order
        (the same indexing as ``excited=[(port, mode_idx)]``).  Empty
        for lumped ports.
    report : PortOperatorReport or None
        The two-path scalar summary attached by
        :func:`build_modal_port`; exposes ``z_line_num`` /
        ``z_line_ref`` / ``cutoff_num`` / ``cutoff_ref``.  For lumped
        ports a synthetic report carrying ``z_line_num = Z0``.
    """

    name: str
    modes: tuple[ModeReport, ...]
    report: PortOperatorReport | None = None

    @property
    def z_line_num(self) -> float | None:
        # Full-model value: a port cut by declared symmetry planes is a
        # half window whose solved impedance carries a factor 2 per
        # plane; the publication layer removes it (DD-154) so the user
        # always reads the impedance of the full structure.
        if self.report is None or self.report.z_line_num is None:
            return None
        return self.report.z_line_num * self.report.z_line_full_scale

    @property
    def z_line_ref(self) -> float | None:
        return self.report.z_line_ref if self.report else None

    @property
    def cutoff_num(self) -> float | None:
        return self.report.cutoff_num if self.report else None

    @property
    def cutoff_ref(self) -> float | None:
        return self.report.cutoff_ref if self.report else None

    @classmethod
    def from_operator(cls, op, mesh=None) -> "PortReport":
        """Build a report from a built port operator.

        Modal operators contribute one :class:`ModeReport` per
        ``discrete_modes`` entry; lumped operators (no mode solve)
        yield an empty mode tuple and a synthetic ``PortOperatorReport`` with
        ``z_line_num = Z0``.

        Passing the *mesh* lets the mode plots resolve the symmetry
        planes cutting the port window, so they show the full
        cross-section instead of the solved half; without it they show
        the solved window.
        """
        if hasattr(op, "discrete_modes"):
            # Publish full-model line impedances (DD-154): the Mode
            # objects keep their raw half-window values (they normalise
            # the injection/recording), the report entries carry the
            # symmetry scale.
            scale = op.port_report.z_line_full_scale if op.port_report else 1.0
            mirrors = resolve_port_mirrors(op.plane, mesh) if mesh is not None else ()
            dual_lengths = getattr(op, "h_dual_lengths", ())
            # Which termination each channel actually got, and the
            # uniform-chain measurement behind the choice (DD-228).
            # The operator decides this at construction; publishing it
            # makes ``solve_ports()`` the place to see it before a run.
            n = len(op.discrete_modes)
            kinds = list(getattr(op, "termination_kinds", None) or [])
            terminations = kinds + [None] * (n - len(kinds))
            spreads = list(getattr(op, "_dtbc_pair_spread", None) or [])
            spreads = spreads + [None] * (n - len(spreads))
            modes = tuple(
                ModeReport(
                    port_name=op.name,
                    name=dm.mode.name,
                    mode_type=dm.mode.mode_type,
                    f_cutoff=dm.mode.omega_c / (2.0 * math.pi),
                    z_line=(None if dm.mode.z_line is None else dm.mode.z_line * scale),
                    epsilon_eff=(
                        float(dm.mode.epsilon_r) if dm.mode.mode_type is ModeType.TEM else None
                    ),
                    _discrete=dm,
                    _plane=op.plane,
                    _mirrors=mirrors,
                    _h_dual_lengths=dual_lengths,
                    termination=terminations[m],
                    chain_spread=spreads[m],
                )
                for m, dm in enumerate(op.discrete_modes)
            )
            return cls(name=op.name, modes=modes, report=op.port_report)
        return cls(
            name=op.name,
            modes=(),
            report=PortOperatorReport(z_line_num=float(op.Z0)),
        )

    def summary(self) -> str:
        """Multi-line human-readable summary (used by ``str()``)."""
        lines = [f"Port {self.name!r} — {len(self.modes)} mode(s)"]
        if self.report is not None and self.report.symmetry_faces:
            cuts = ", ".join(f"{face} ({kind})" for face, kind in self.report.symmetry_faces)
            lines.append(f"  cut by symmetry plane(s) {cuts} — impedances are full-model values")
        if self.z_line_num is not None:
            z = f"  z_line = {self.z_line_num:.2f} Ω (numerical)"
            if self.z_line_ref is not None:
                delta = self.report.z_line_delta_relative
                z += f", {self.z_line_ref:.2f} Ω (reference, Δ {100.0 * delta:+.2f} %)"
            lines.append(z)
        if self.cutoff_num is not None:
            c = f"  f_cutoff = {self.cutoff_num / 1e9:.4f} GHz (numerical)"
            if self.cutoff_ref is not None:
                c += f", {self.cutoff_ref / 1e9:.4f} GHz (reference)"
            lines.append(c)
        for i, m in enumerate(self.modes):
            entry = (
                f"  [{i}] {m.name:<12s} {m.mode_type.value:<3s} f_c = {m.f_cutoff / 1e9:.4f} GHz"
            )
            if m.z_line is not None:
                entry += f"  z_line = {m.z_line:.2f} Ω"
            if m.epsilon_eff is not None:
                entry += f"  ε_eff = {m.epsilon_eff:.3f}"
            if m.termination is not None:
                entry += f"  termination = {m.termination}"
                if m.termination == "dtbc" and m.chain_floor_db is not None:
                    entry += f" (chain floor <= {m.chain_floor_db:.0f} dB)"
                elif m.chain_spread is not None:
                    entry += f" (chain spread {m.chain_spread:.1e})"
            lines.append(entry)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
