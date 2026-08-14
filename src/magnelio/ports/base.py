"""Common Port protocol for the FIT-TD solver.

A Port couples a region of the simulation domain — a 0-D lumped gap
(``PortOperatorLumped``) or a 2-D modal cross-section
(``PortOperatorModal``) — to the leapfrog solver.  All ports share:

* an E-side hook ``update_e(fields, t, dt)`` that runs after the curl-
  driven E-update, after PEC re-enforcement, and after CPML / source
  corrections — i.e. as the *last* E-side step before the recorder
  reads V and I and the H-update begins;
* per-mode V/I projections (``project_V``, ``project_I``) that produce
  scalar time-series readouts for the unified ``PortSignalRecorder``;
* an excitation interface (``set_excitation``, ``clear_excitation``)
  that switches a port between active source and passive monitor.

The interface is mode-indexed throughout: lumped ports report a single
mode (``n_modes == 1``); modal ports report ``n_modes`` channels in a
fixed order.  This matches the S-parameter convention where
``S(port_to, port_from, mode_to=0, mode_from=0)`` collapses to
``S(port_to, port_from)`` for single-mode ports.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from magnelio._fields.field_arrays import FieldState


@runtime_checkable
class Port(Protocol):
    """Common interface for all FIT-TD port operators.

    Implementations: ``PortOperatorLumped`` (n_modes = 1, lumped),
    ``PortOperatorModal`` (n_modes ≥ 1, modal cross-section).
    """

    name: str
    n_modes: int

    def project_V(self, e: np.ndarray) -> np.ndarray:
        """Per-mode voltage at the current E-field.

        Parameters
        ----------
        e : np.ndarray
            Flat E vector at ``t^{n+1}`` (post-PEC, post-``update_e``).

        Returns
        -------
        np.ndarray
            Shape ``(n_modes,)``, dtype float.
        """
        ...

    def project_I(self, h: np.ndarray) -> np.ndarray:
        """Per-mode current at the current H-field.

        Parameters
        ----------
        h : np.ndarray
            Flat H vector at ``t^{n+1/2}`` (pre-H-update).

        Returns
        -------
        np.ndarray
            Shape ``(n_modes,)``, dtype float.

        Notes
        -----
        Lumped ports return the internally cached Thévenin current from
        the most recent ``update_e`` call; ``h`` is ignored.  Modal
        ports project the H field onto the modal basis.
        """
        ...

    def update_e(self, fields: FieldState, t: float, dt: float) -> None:
        """E-side hook, called after PEC / CPML / source corrections.

        Lumped ports run the semi-implicit Thévenin update on their
        edges here.  Modal ports run the Mur-1st absorber + optional
        TF/SF source injection.  Both modify ``fields`` in place.

        ``t`` is the time at which the post-update E lives (i.e.
        ``t^{n+1}``); ``dt`` is the solver time step.
        """
        ...

    def set_excitation(
        self,
        mode_idx: int,
        waveform_fn: Callable[[float], float],
    ) -> None:
        """Activate an excitation on the given mode.

        Parameters
        ----------
        mode_idx : int
            Index in ``[0, n_modes)``.
        waveform_fn : Callable[[float], float]
            Source amplitude as a function of time [s].
        """
        ...

    def clear_excitation(self) -> None:
        """Deactivate all excitations — port becomes a passive monitor."""
        ...
