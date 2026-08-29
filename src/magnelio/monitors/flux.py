"""
Flux monitors — Poynting flux through mesh-aligned cross-sections.

:class:`MonitorFluxTime` records the instantaneous power P(t) at every
time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np


def _boundary_half_weights(
    n_nodes: int,
    magnetic_lo: bool = False,
    magnetic_hi: bool = False,
) -> np.ndarray:
    """Per-node weights for the FIT flux identity ``P = Σ e·h`` (DD-085).

    The states are grid quantities (``e = E·l_primal``, ``h = H·l_dual``),
    so the Poynting sum needs no area weights — the lengths in the states
    ARE the area patch.  The only correction: boundary h states carry the
    full first/last cell as dual length (solver convention) while the
    physical flux patch is the half cell, so those terms count ×½
    (the DD-082 MonitorWallLoss precedent).  Exception: at a PMC bbox
    face the magnetic wall sits half the outer cell BEYOND the outermost
    grid line, the physical patch is the full boundary cell, and the
    solver convention is already correct — weight 1 (the same
    wall-position booking as the port capacitance and Poynting patches).
    """
    w = np.ones(n_nodes)
    if n_nodes > 1:
        w[0] = 1.0 if magnetic_lo else 0.5
        w[-1] = 1.0 if magnetic_hi else 0.5
    return w


@dataclass
class MonitorFluxTime:
    """Integrate normal Poynting flux through a mesh-aligned surface.

    The flux is always integrated over the **full** cross-section of
    the domain (a partial-aperture flux is not what this monitor
    measures), so the surface is fully described by an axis-aligned
    plane.

    Parameters
    ----------
    normal : str
        Normal axis of the cross-section plane (``"x"``, ``"y"`` or
        ``"z"``).
    position : float
        Position of the plane along that axis [m]; snapped to the
        nearest grid node.
    name : str
        Monitor label.

    Examples
    --------
    >>> flux = MonitorFluxTime(normal="z", position=5e-3, name="flux_z")
    """

    normal: str
    position: float
    name: str = ""

    # --- internal ---
    _axis: str = field(default="", repr=False, init=False)
    _pos: float = field(default=0.0, repr=False, init=False)
    _k: int = field(default=0, repr=False, init=False)
    _w_1: np.ndarray | None = field(default=None, repr=False, init=False)
    _w_2: np.ndarray | None = field(default=None, repr=False, init=False)
    _times: list[float] = field(default_factory=list, repr=False, init=False)
    _power: list[float] = field(default_factory=list, repr=False, init=False)
    _next_idx: int = field(default=0, repr=False, init=False)
    _sym_factor: float = field(default=1.0, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.normal not in ("x", "y", "z"):
            raise ValueError(
                f"MonitorFluxTime normal must be 'x', 'y' or 'z'; got {self.normal!r}",
            )
        try:
            pos = float(self.position)
        except (TypeError, ValueError):
            raise ValueError(
                f"MonitorFluxTime position must be a finite number [m]; got {self.position!r}",
            ) from None
        if not isfinite(pos):
            raise ValueError(
                f"MonitorFluxTime position must be a finite number [m]; got {self.position!r}",
            )
        self._axis = self.normal
        self._pos = pos
        if not self.name:
            self.name = f"flux_{self._axis}_{id(self):x}"

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        """Snap to nearest grid node and pre-compute the flux weights.

        Full-model booking on a symmetric run: every symmetry
        plane whose axis lies IN the cross-section halves the meshed
        aperture, so the recorded flux doubles per such plane.  This is
        source-independent because the sources themselves declare
        full-model amplitudes — a modal port injects
        ×1/√2 per cutting plane (fields at full-model level, half the
        full-model power into the meshed half), and a plane wave is
        field-normalised anyway.  A plane parallel to the monitor
        surface leaves the aperture whole (factor 1).  Under the
        earlier half-window-normalised excitation this ×2 was a
        factor-2 error (measured in
        ``validation/symmetry_full_vs_half_certificate.py``).
        """
        # Design: DD-155 (full-model power semantics — sources declare
        # full-model amplitudes; the pre-DD-155 excitation was
        # half-window-normalised).
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            symmetry_entries,
        )

        grid = mesh.grid
        bc = getattr(mesh, "boundary_conditions", None)
        n_cutting = sum(1 for face in symmetry_entries(bc) if face[0] != self._axis)
        self._sym_factor = float(2**n_cutting)

        def _pmc(face: str) -> bool:
            return bc is not None and getattr(bc, face, None) == "PMC"

        def _weights(n_nodes: int, axis: str) -> np.ndarray:
            return _boundary_half_weights(
                n_nodes,
                magnetic_lo=_pmc(f"{axis}min"),
                magnetic_hi=_pmc(f"{axis}max"),
            )

        # DD-085: grid-quantity states make the flux the plain FIT sum
        # P = Σ e·h; the "weights" are 1 except boundary-h halving.
        if self._axis == "z":
            nodes = grid.z
            self._k = int(np.argmin(np.abs(nodes - self._pos)))
            self._k = min(self._k, grid.Nz - 1)
            # (Ex·Hy): Hy on y-nodes; (Ey·Hx): Hx on x-nodes
            self._w_1 = np.broadcast_to(_weights(grid.Ny + 1, "y")[None, :], (grid.Nx, grid.Ny + 1))
            self._w_2 = np.broadcast_to(_weights(grid.Nx + 1, "x")[:, None], (grid.Nx + 1, grid.Ny))

        elif self._axis == "x":
            nodes = grid.x
            self._k = int(np.argmin(np.abs(nodes - self._pos)))
            self._k = min(self._k, grid.Nx - 1)
            # (Ey·Hz): Hz on z-nodes; (Ez·Hy): Hy on y-nodes
            self._w_1 = np.broadcast_to(_weights(grid.Nz + 1, "z")[None, :], (grid.Ny, grid.Nz + 1))
            self._w_2 = np.broadcast_to(_weights(grid.Ny + 1, "y")[:, None], (grid.Ny + 1, grid.Nz))

        else:  # y
            nodes = grid.y
            self._k = int(np.argmin(np.abs(nodes - self._pos)))
            self._k = min(self._k, grid.Ny - 1)
            # (Ez·Hx): Hx on x-nodes; (Ex·Hz): Hz on z-nodes
            self._w_1 = np.broadcast_to(_weights(grid.Nx + 1, "x")[:, None], (grid.Nx + 1, grid.Nz))
            self._w_2 = np.broadcast_to(_weights(grid.Nz + 1, "z")[None, :], (grid.Nx, grid.Nz + 1))

        self._times = []
        self._power = []
        self._next_idx = 0

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Record instantaneous Poynting flux at this time step."""
        k = self._k

        # GPU backend: device field arrays refuse implicit mixing with
        # the NumPy weight planes — move the weights to the device once
        # (they are read-only after attach).
        if type(fields.Ex).__module__.partition(".")[0] == "cupy":
            import cupy  # noqa: PLC0415

            if not isinstance(self._w_1, cupy.ndarray):
                self._w_1 = cupy.asarray(self._w_1)
                self._w_2 = cupy.asarray(self._w_2)

        if self._axis == "z":
            # S_z = Ex·Hy − Ey·Hx
            P = float(np.sum(fields.Ex[:, :, k] * fields.Hy[:, :, k] * self._w_1)) - float(
                np.sum(fields.Ey[:, :, k] * fields.Hx[:, :, k] * self._w_2)
            )
        elif self._axis == "x":
            # S_x = Ey·Hz − Ez·Hy
            P = float(np.sum(fields.Ey[k, :, :] * fields.Hz[k, :, :] * self._w_1)) - float(
                np.sum(fields.Ez[k, :, :] * fields.Hy[k, :, :] * self._w_2)
            )
        else:  # y
            # S_y = Ez·Hx − Ex·Hz
            P = float(np.sum(fields.Ez[:, k, :] * fields.Hx[:, k, :] * self._w_1)) - float(
                np.sum(fields.Ex[:, k, :] * fields.Hz[:, k, :] * self._w_2)
            )

        self._times.append(t)
        self._power.append(self._sym_factor * P)
        self._next_idx += 1

    def finalize(self) -> None:
        """Called after the simulation completes (no-op)."""
        pass

    # ------------------------------------------------------------------
    # Streaming write-through (DD-070 follow-up)
    # ------------------------------------------------------------------

    def pop_pending(self) -> tuple[list[float], list[float]]:
        """Drain the ``(time, power)`` samples recorded since the last call.

        Like :meth:`MonitorFieldTime.pop_pending`, this returns the pending
        samples and clears the in-RAM buffer so a project-backed run stays
        memory-bounded (the run sink appends each batch to ``results.h5``).
        ``_next_idx`` (the total recorded count) is *kept*, so it drives the
        flux-stream truncation on resume.  The in-RAM path never calls this,
        so its ``.power`` / ``.t`` / ``.total_energy`` accumulation is
        unchanged.
        """
        if not self._power:
            return [], []
        times = list(self._times)
        power = list(self._power)
        self._times = []
        self._power = []
        return times, power

    def state_dict(self) -> dict:
        """Checkpoint the recorded-sample count for a bit-exact resume.

        Only the running count is state a continuation must restore — the
        samples themselves live in the run's ``results.h5`` (streamed).
        ``_next_idx`` equals the number of samples recorded so far (flux
        records every step), so it also drives the flux-stream truncation
        on resume.
        """
        return {"next_idx": int(self._next_idx)}

    def load_state_dict(self, sd: dict) -> None:
        """Restore the recorded-sample count (see :meth:`state_dict`)."""
        self._next_idx = int(sd["next_idx"])

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def t(self) -> np.ndarray:
        """Time axis [s]."""
        return np.asarray(self._times)

    @property
    def power(self) -> np.ndarray:
        """Instantaneous Poynting flux [W] vs. time."""
        return np.asarray(self._power)

    @property
    def total_energy(self) -> float:
        """Time-integrated Poynting energy [J]."""
        t = self.t
        p = self.power
        if len(t) < 2:
            return 0.0
        return float(np.trapezoid(p, t))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, ax=None):
        """Plot instantaneous Poynting flux vs. time.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(self.t * 1e9, self.power, label=self.name)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Power [W]")
        ax.set_title(f"{self.name} — Poynting flux (S_{self._axis})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"MonitorFluxTime(name={self.name!r}, axis={self._axis!r}, n_steps={len(self._times)})"
        )
