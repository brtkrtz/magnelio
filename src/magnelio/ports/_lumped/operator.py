"""LumpedElementOperator / PortOperatorLumped — Thévenin lumped elements.

``LumpedElementOperator`` (DD-077/DD-078, 3b part 2) is the general
lumped two-terminal element on a chain of grid edges: a trapezoidal
companion model (:class:`magnelio.circuit.SeriesRLC` /
:class:`~magnelio.circuit.ParallelRLC`) in series with an optional
independent Thévenin source, implementing the unified
:class:`magnelio.ports.base.Port` protocol with ``n_modes = 1``.  The
edge chain carries per-edge field components and orientation signs, so
an :class:`magnelio.circuit.EdgePath` staircase from the canonical
rasteriser (DD-076) can drive it as well as the axis-aligned two-point
chain of the classic discrete port.

``PortOperatorLumped`` is the thin special case ``SeriesRLC(R=Z0)``
(DD-077's unification): same constructor surface as before; with a pure
resistance the companion contributes ``r_eq = Z0`` and ``v_hist = 0``
exactly, so the update is arithmetically identical to the historic
implementation (bit-identity gated).

Per-step semantics:

* :meth:`update_e` runs the semi-implicit Thévenin update on the port
  edges and caches the corrected total voltage and lumped current.
  With the companion relation ``V_elem = r_eq·i + v_hist`` and the
  local half-step term ``Σβ``, KVL gives

      i = (v_src − v_hist − v_total) / (r_eq + Σβ).

  Called after PEC / CPML / source E-corrections, just before the
  recorder reads V/I.
* :meth:`project_V` recomputes the (signed) line integral from the
  post-update E field (matching the recorder's expectation that V is
  read at ``t^{n+1}``).
* :meth:`project_I` returns the cached Thévenin current; the H field
  argument is ignored (lumped ports do not project H).

Excitation units (DD-078): the user waveform is the incident power-wave
amplitude ``a(t)`` in √W; :meth:`set_excitation` realises it as the
Thévenin source ``v_src = 2√Z0·a(t)`` (``Z0`` = the port's power-wave
reference impedance; ``Z0 = 0`` keeps the raw waveform).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from magnelio._fields.field_arrays import FieldState
from magnelio.circuit.companion import ParallelRLC, SeriesRLC

_COMPONENT_OF_DIRECTION = {"x": 0, "y": 1, "z": 2}


@dataclass
class LumpedElementOperator:
    """General lumped element (companion + optional source) on grid edges.

    Construction parameters are filled by a builder; do not instantiate
    this class directly.

    Parameters
    ----------
    name : str
        Recorder channel key.
    Z0 : float
        Power-wave reference impedance [Ω] for a/b decomposition and
        the √W source convention (DD-078).
    element : SeriesRLC or ParallelRLC
        Trapezoidal companion model — the element's internal impedance.
    flat_edge_indices, ijk_list, dl_list : lists
        Edge chain in the flat ``Ex|Ey|Ez`` layout.
    edge_components : list[int]
        Field component per edge (0 = Ex, 1 = Ey, 2 = Ez).
    edge_signs : list[float]
        Orientation sign per edge (±1, EdgePath convention).
    beta_E : np.ndarray
        Lossless update coefficient ``dt / M_eps`` per edge.
    port_report : LumpedPortReport, optional
        Symmetry planes cutting the chain.  ``Z0`` and ``element``
        then hold the internally scaled half-model device; the
        recorder and the injection read the report's full-model scales
        through the shared ``port_report`` plumbing.
    """

    name: str
    Z0: float
    element: SeriesRLC | ParallelRLC
    flat_edge_indices: list[int]
    ijk_list: list[tuple[int, int, int]]
    dl_list: list[float]
    edge_components: list[int]
    edge_signs: list[float]
    beta_E: np.ndarray = field(repr=False)
    port_report: object | None = field(default=None, repr=False)

    # Derived in __post_init__
    _beta_sum: float = field(default=0.0, repr=False, init=False)
    _L: float = field(default=0.0, repr=False, init=False)

    # Excitation state
    _waveform_fn: Callable[[float], float] | None = field(
        default=None,
        repr=False,
        init=False,
    )

    # Last-step cache for project_V / project_I
    _last_V: float = field(default=0.0, repr=False, init=False)
    _last_I: float = field(default=0.0, repr=False, init=False)

    def __post_init__(self) -> None:
        self._beta_sum = float(np.sum(self.beta_E))
        self._L = float(sum(self.dl_list))

    @property
    def n_modes(self) -> int:
        return 1

    def project_V(self, e: np.ndarray) -> np.ndarray:
        # DD-085: the states are FIT grid quantities (edge voltages),
        # so the gap voltage is the plain signed sum along the chain.
        v = 0.0
        for flat, sign in zip(self.flat_edge_indices, self.edge_signs):
            v += sign * float(e[flat])
        return np.array([v], dtype=float)

    def project_I(self, h: np.ndarray) -> np.ndarray:
        del h
        return np.array([self._last_I], dtype=float)

    def update_e(self, fields: FieldState, t: float, dt: float) -> None:
        comps = (fields.Ex, fields.Ey, fields.Ez)

        # DD-085 grid-quantity form: edge states are voltages, so the
        # line integral is the plain signed sum and the injected lumped
        # current enters the FIT update as β·i directly.  (The historic
        # field-interpretation ``Σ e·dl`` / ``β·i/dl`` pair was only
        # consistent on uniform grids.)
        v_total = 0.0
        for (i, j, k), c, sign in zip(
            self.ijk_list,
            self.edge_components,
            self.edge_signs,
        ):
            v_total += sign * float(comps[c][i, j, k])

        v_src = self._waveform_fn(t) if self._waveform_fn is not None else 0.0
        r_eq = self.element.r_eq(dt)
        v_hist = self.element.v_hist(dt)
        z_eff = r_eq + self._beta_sum
        i_port = (v_src - v_hist - v_total) / z_eff

        for (i, j, k), c, sign, beta in zip(
            self.ijk_list,
            self.edge_components,
            self.edge_signs,
            self.beta_E,
        ):
            comps[c][i, j, k] += sign * beta * i_port

        self.element.advance(i_port, r_eq * i_port + v_hist, dt)
        self._last_V = v_total + i_port * self._beta_sum
        self._last_I = i_port

    def set_excitation(
        self,
        mode_idx: int,
        waveform_fn: Callable[[float], float],
    ) -> None:
        if mode_idx != 0:
            raise ValueError(
                f"{type(self).__name__} has only one mode (index 0); got mode_idx={mode_idx}.",
            )
        # DD-078: the user waveform is the incident power-wave amplitude
        # a(t) in √W.  With the Thévenin identity a = v_src/(2√Z0), the
        # source voltage that realises it is v_src = 2√Z0·a(t) — this
        # makes lumped ports commensurate with the physically-scaled
        # modal ports (Z0 = 0 keeps the raw waveform: no power-wave
        # reference exists for an ideal source).
        src_scale = 2.0 * math.sqrt(self.Z0) if self.Z0 > 0.0 else 1.0
        if src_scale != 1.0:

            def _scaled_wf(t: float, _fn=waveform_fn, _s=src_scale) -> float:
                return _s * _fn(t)

            waveform_fn = _scaled_wf
        self._waveform_fn = waveform_fn

    def clear_excitation(self) -> None:
        self._waveform_fn = None

    def state_dict(self) -> dict:
        """Checkpoint the V/I cache + companion state (DD-070, WP-S6)."""
        return {
            "last_V": float(self._last_V),
            "last_I": float(self._last_I),
            "element": self.element.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore state written by :meth:`state_dict` (bit-exact resume).

        Checkpoints written before the companion unification carry no
        ``element`` group; the pure-R history is stateless, so loading
        them remains bit-exact.
        """
        self._last_V = float(sd["last_V"])
        self._last_I = float(sd["last_I"])
        if "element" in sd:
            self.element.load_state_dict(sd["element"])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"Z0={self.Z0:.1f}Ω, element={self.element!r}, "
            f"n_edges={len(self.flat_edge_indices)}, L={self._L:.3e}m)"
        )


class PortOperatorLumped(LumpedElementOperator):
    """Lumped port on an axis-aligned two-point edge chain.

    The thin ``SeriesRLC(R=Z0)`` special case of
    :class:`LumpedElementOperator` (DD-077 unification): with a pure
    resistance the companion is ``r_eq = Z0``, ``v_hist = 0`` and the
    update is arithmetically identical to the historic discrete port.
    A non-trivial ``element`` turns the same port into a lumped RLC
    (source or passive load); ``Z0`` stays the power-wave reference.

    Construction parameters are filled by :func:`build_lumped_port`;
    do not instantiate this class directly.
    """

    def __init__(
        self,
        name: str,
        Z0: float,
        direction: str,
        flat_edge_indices: list[int],
        ijk_list: list[tuple[int, int, int]],
        dl_list: list[float],
        beta_E: np.ndarray,
        element: SeriesRLC | ParallelRLC | None = None,
        port_report=None,
    ) -> None:
        component = _COMPONENT_OF_DIRECTION[direction]
        n = len(flat_edge_indices)
        super().__init__(
            name=name,
            Z0=Z0,
            element=element if element is not None else SeriesRLC(R=Z0),
            flat_edge_indices=flat_edge_indices,
            ijk_list=ijk_list,
            dl_list=dl_list,
            edge_components=[component] * n,
            edge_signs=[1.0] * n,
            beta_E=beta_E,
            port_report=port_report,
        )
        self.direction = direction

    def __repr__(self) -> str:
        return (
            f"PortOperatorLumped(name={self.name!r}, "
            f"direction={self.direction!r}, Z0={self.Z0:.1f}Ω, "
            f"element={self.element!r}, "
            f"n_edges={len(self.flat_edge_indices)}, L={self._L:.3e}m)"
        )
