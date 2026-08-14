"""Mode-solver Protocol for the modal waveguide-port pipeline.

A mode solver consumes a port-cross-section specification and produces a
list of ``Mode`` objects sorted by ascending cut-off frequency.  The
Protocol form keeps Phase-1 analytical solvers and any future Phase-2
numerical 2D solver behind the same call surface.
"""

from __future__ import annotations

from typing import Protocol

from magnelio.ports._modal.mode import Mode


class ModeSolver(Protocol):
    """Common surface for analytical and numerical mode solvers."""

    def solve(self, n_modes: int, f_calc: float = 0.0) -> list[Mode]:
        """Compute the lowest ``n_modes`` modes for the cross-section.

        Parameters
        ----------
        n_modes : int
            Number of modes to return, sorted by ascending ``omega_c``.
            Ties broken by mode_type (TE before TM at the same cut-off).
        f_calc : float, default 0.0
            Mode calculation frequency [Hz].  Required for dispersive
            wave-impedance modes (TE/TM): the H-component amplitude in
            the field evaluator is baked in at ``Z(2π f_calc)``.  Ignored
            by non-dispersive (TEM) solvers.

        Returns
        -------
        list[Mode]
            Modes with field evaluators Poynting-normalised to 1 W at
            ``f_calc``.
        """
        ...
