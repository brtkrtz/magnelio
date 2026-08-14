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
    """

    port_name: str
    name: str
    mode_type: ModeType
    f_cutoff: float
    z_line: float | None
    _discrete: DiscreteMode = field(repr=False)
    _plane: PortPlane = field(repr=False)

    def z_modal(self, f: float) -> complex:
        """Power-wave reference impedance at frequency ``f`` [Hz]."""
        return self._discrete.mode.z_modal(2.0 * math.pi * f)

    def z_wave(self, f: float) -> complex:
        """Modal wave impedance at frequency ``f`` [Hz]."""
        return self._discrete.mode.z_wave(2.0 * math.pi * f)

    def gamma(self, f: float) -> complex:
        """Propagation constant ``γ = α + jβ`` at frequency ``f`` [Hz]."""
        return self._discrete.mode.gamma(2.0 * math.pi * f)

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
            Target number of arrows per axis.
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
            comp_u = _edge_grid(self._discrete.e_u_profile, self._plane.u_edge_uv)
            comp_v = _edge_grid(self._discrete.e_v_profile, self._plane.v_edge_uv)
            # E_u lives at (u-centre, v-node): average along v.
            # E_v lives at (u-node, v-centre): average along u.
            u_cc = _avg_nonzero(comp_u[0][:, :-1], comp_u[0][:, 1:])
            v_cc = _avg_nonzero(comp_v[0][:-1, :], comp_v[0][1:, :])
            uc, vc = comp_u[1], comp_v[2]
        elif field == "H":
            # H_u is co-located with the v-edges, H_v with the u-edges.
            comp_u = _edge_grid(self._discrete.h_u_profile, self._plane.v_edge_uv)
            comp_v = _edge_grid(self._discrete.h_v_profile, self._plane.u_edge_uv)
            u_cc = _avg_nonzero(comp_u[0][:-1, :], comp_u[0][1:, :])
            v_cc = _avg_nonzero(comp_v[0][:, :-1], comp_v[0][:, 1:])
            uc, vc = comp_v[1], comp_u[2]
        else:
            raise ValueError(f"field must be 'E' or 'H'; got {field!r}")

        face = self._plane.face
        u_name = _AXIS_NAMES[face.u_axis]
        v_name = _AXIS_NAMES[face.v_axis]
        if title is None:
            title = f"{self.port_name}: {self.name} ({self.mode_type.value}) {field} profile"

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
            )

        return plot_field_vector(
            uc,
            vc,
            u_cc,
            v_cc,
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
    def from_operator(cls, op) -> "PortReport":
        """Build a report from a built port operator.

        Modal operators contribute one :class:`ModeReport` per
        ``discrete_modes`` entry; lumped operators (no mode solve)
        yield an empty mode tuple and a synthetic ``PortOperatorReport`` with
        ``z_line_num = Z0``.
        """
        if hasattr(op, "discrete_modes"):
            # Publish full-model line impedances (DD-154): the Mode
            # objects keep their raw half-window values (they normalise
            # the injection/recording), the report entries carry the
            # symmetry scale.
            scale = op.port_report.z_line_full_scale if op.port_report else 1.0
            modes = tuple(
                ModeReport(
                    port_name=op.name,
                    name=dm.mode.name,
                    mode_type=dm.mode.mode_type,
                    f_cutoff=dm.mode.omega_c / (2.0 * math.pi),
                    z_line=(None if dm.mode.z_line is None else dm.mode.z_line * scale),
                    _discrete=dm,
                    _plane=op.plane,
                )
                for dm in op.discrete_modes
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
            lines.append(entry)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
