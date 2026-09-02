"""Two-path mode reporting for modal ports.

A :class:`PortOperatorReport` carries the user-visible information about both
the *reference* and the *operator-consistent* mode of a modal port,
per DD-048.  It is attached to :class:`PortOperatorModal` via the
``port_report`` attribute and populated by :func:`build_modal_port`.

Field semantics
---------------

``z_line_num`` / ``cutoff_num``
    Properties of the mode that drives the FIT-TD update operator
    (path b in DD-048).  Solved on the 2D transversal slice of the
    3D mesh.  Always populated for modes with a defined Z_line / f_c.

``z_line_ref`` / ``cutoff_ref``
    Properties of the mesh-independent reference mode (path a in
    DD-048).  Source: closed-form analytical solver where one is
    available, or :func:`solve_modes_refined` when the spec activated
    ``reference_refinement``.  ``None`` if no reference path was
    requested.

``refinement_log``
    Per-level :class:`ModeRefinementReport` from
    :func:`solve_modes_refined`, populated only when the spec used
    that path.  ``None`` for analytical-reference specs and for
    numerical specs with ``reference_refinement = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from magnelio.ports._modal.refinement import ModeRefinementReport


@dataclass(frozen=True)
class PortOperatorReport:
    z_line_num: Optional[float] = None
    z_line_ref: Optional[float] = None
    cutoff_num: Optional[float] = None
    cutoff_ref: Optional[float] = None
    refinement_log: "Optional[ModeRefinementReport]" = None
    # Symmetry planes cutting the port window (DD-154), as
    # ((face, wall_kind), ...) pairs, e.g. (("ymin", "PMC"),).  The
    # numeric fields above stay the raw half-window solver values; the
    # publication layer (PortReport) applies z_line_full_scale so the
    # user sees full-model quantities.
    symmetry_faces: tuple = ()
    # The line impedance of an inhomogeneous (quasi-TEM) cross-section
    # is the frequency-flat quasi-static value of the Laplace mode, and
    # the summary labels it so: the impedance the discrete wave carries
    # at the top of the band sits a few percent above it (DD-239).
    quasi_static: bool = False

    @property
    def z_line_full_scale(self) -> float:
        """Half-window → full-model scale for the line impedance.

        Each cutting magnetic symmetry plane halves the window and its
        capacitance, so the two halves sit in parallel
        (``z_full = z_half / 2``); an electric symmetry plane puts them
        in series (``z_full = 2 · z_half``).
        """
        scale = 1.0
        for _, kind in self.symmetry_faces:
            scale *= 0.5 if kind == "PMC" else 2.0
        return scale

    @property
    def power_wave_full_scale(self) -> float:
        """Half-window → full-model scale for modal wave amplitudes.

        The port modes are power-normalised on the half window, so a
        recorded amplitude of 1 √W accounts for the power crossing the
        meshed half only; the mirror half carries the same power again.
        Each cutting symmetry plane therefore scales the full-model
        wave amplitude by √2 (and the excitation by 1/√2 so that a
        declared injected power is a full-model watt).
        """
        return 2.0 ** (0.5 * len(self.symmetry_faces))

    @property
    def z_line_delta_relative(self) -> Optional[float]:
        # The reference path solves the continuous full geometry, so
        # the comparison uses the full-model numeric value.
        if self.z_line_ref is None or self.z_line_ref == 0.0:
            return None
        return (self.z_line_num * self.z_line_full_scale - self.z_line_ref) / self.z_line_ref
