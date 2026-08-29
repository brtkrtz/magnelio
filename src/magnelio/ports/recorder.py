"""PortSignalRecorder — unified V/I time-series recorder for any Port.

Operates on any sequence of objects that satisfy the
:class:`magnelio.ports.base.Port` protocol — ``PortOperatorLumped``,
``PortOperatorModal``, or any future port type.  Channels are addressed
by ``(port_name, mode_idx)`` tuples; the recorder queries
``port.project_V(e)`` and ``port.project_I(h)`` per call to
:meth:`record`.

``finalize(n_steps_actual=...)`` trims the recorded buffers to the
actual number of leapfrog steps the solver ran — this matters when the
solver terminates early (e.g. via ``energy_stop_db``): without trimming,
zero-padded tails would drag artefacts into ``compute_s_parameters``
and corrupt the FFT.
"""

from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np

from magnelio.signals.signal_1d import Signal1D

# Device-staging block budget per port (WP-G1): the ring buffer holds
# up to this many bytes of raw port-plane samples before a drain is
# forced.  Capacity in steps is clamped so tiny planes do not stage
# unboundedly long and huge planes still amortise the transfer.
_STAGE_BLOCK_BYTES = 8 << 20
_STAGE_MIN_STEPS = 16
_STAGE_MAX_STEPS = 4096


class _DevicePortStage:
    """Device-side sample ring buffer for one stageable port (WP-G1).

    On the CuPy backend the per-step host round trips of
    ``project_V``/``project_I`` (four blocking D2H gathers per port per
    step) dominate the GPU small-grid floor.  This stage instead
    gathers the raw port-plane samples device-side each step — two
    fancy-index kernels, no synchronisation — and hands back one host
    block per drain, after which the port's *unchanged* host dot
    products materialise V/I (``project_V_samples`` /
    ``project_I_samples``).  Bit-identical by construction: the
    concatenated gather moves the same float64 values the per-array
    gathers would.
    """

    def __init__(self, port, e, h) -> None:
        e_idx, h_idx = port.record_gather_indices
        # Resolve the array module from the field array itself: CuPy in
        # production; plain NumPy for ndarray subclasses that mimic the
        # device interface (unit tests exercise the staging machinery
        # without a CUDA device that way).
        self._xp = (
            np
            if isinstance(e, np.ndarray)
            else (importlib.import_module(type(e).__module__.split(".")[0]))
        )
        self._e_idx = self._xp.asarray(e_idx)
        self._h_idx = self._xp.asarray(h_idx)
        step_bytes = e_idx.size * e.dtype.itemsize + h_idx.size * h.dtype.itemsize
        self.capacity = int(
            np.clip(
                _STAGE_BLOCK_BYTES // max(step_bytes, 1),
                _STAGE_MIN_STEPS,
                _STAGE_MAX_STEPS,
            )
        )
        self._buf_e = self._xp.empty((self.capacity, e_idx.size), dtype=e.dtype)
        self._buf_h = self._xp.empty((self.capacity, h_idx.size), dtype=h.dtype)
        self._n = 0

    @property
    def full(self) -> bool:
        return self._n >= self.capacity

    @property
    def n_staged(self) -> int:
        return self._n

    def stage(self, e, h) -> None:
        self._buf_e[self._n] = e[self._e_idx]
        self._buf_h[self._n] = h[self._h_idx]
        self._n += 1

    def drain(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(e_block, h_block)`` host arrays and reset the buffer."""
        k = self._n
        self._n = 0
        blk_e = self._buf_e[:k]
        blk_h = self._buf_h[:k]
        if hasattr(blk_e, "get"):  # CuPy: one D2H per array
            return blk_e.get(), blk_h.get()
        return np.array(blk_e), np.array(blk_h)


class PortSignalRecorder:
    """Per-port, per-mode V and I time-series recorder.

    Parameters
    ----------
    dt : float
        Solver time step [s].
    ports : Iterable[Port]
        Port operators whose V and I time series are recorded.  Each
        must expose ``name``, ``n_modes``, ``project_V(e)`` and
        ``project_I(h)``.  Names must be unique across the iterable.

    Notes
    -----
    Sign convention follows the modal orthonormal basis: positive
    ``V_m`` means the E field at the port plane has a positive
    component along the discrete mode profile ``ê_m``.  Lumped
    (discrete) ports report the line integral of E over the port
    edges.  Power-wave decomposition (forward / reflected) is the
    responsibility of :func:`compute_s_parameters`, not of the recorder.
    """

    def __init__(
        self,
        dt: float,
        ports: Iterable,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        ports_list = list(ports)
        labels = [p.name for p in ports_list]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"port names must be unique; got {labels}",
            )

        self._dt = float(dt)
        self._ports = ports_list
        # DD-078: per-port physical amplitude scale (modal ports expose
        # record_scale = κ per mode; ports without it record raw units —
        # lumped ports are already in volts, band ports keep their
        # phasor-referenced convention).  DD-155 composes the full-model
        # power-wave scale on top: a port cut by symmetry planes records
        # V/I ×√2 per plane, so every consumer of the recorded signals
        # (a/b split, S-parameters, stores) sees full-model amplitudes.
        # A factor of exactly 1 keeps ``None`` — non-symmetric runs stay
        # bit-identical.
        self._scales: list[np.ndarray | float | None] = []
        for p in ports_list:
            scale = getattr(p, "record_scale", None)
            if scale is not None:
                scale = np.asarray(scale, dtype=float)
            report = getattr(p, "port_report", None)
            sym = getattr(report, "power_wave_full_scale", 1.0) if report is not None else 1.0
            if sym != 1.0:
                scale = sym if scale is None else scale * sym
            self._scales.append(scale)

        self._V_buffers: dict[tuple[str, int], list[float]] = {}
        self._I_buffers: dict[tuple[str, int], list[float]] = {}
        for p in ports_list:
            for m in range(p.n_modes):
                self._V_buffers[(p.name, m)] = []
                self._I_buffers[(p.name, m)] = []

        self._n_steps = 0
        # WP-G1 device staging: ``None`` until the first ``record``
        # call decides per backend; a list parallel to ``_ports`` with
        # ``_DevicePortStage`` entries (or ``None`` for ports recorded
        # immediately) afterwards.  ``_staged`` is True when at least
        # one port stages.
        self._stage_list: list[_DevicePortStage | None] | None = None
        self._staged = False

    @property
    def n_steps_recorded(self) -> int:
        return self._n_steps

    @property
    def channels(self) -> list[tuple[str, int]]:
        return list(self._V_buffers.keys())

    def tail(
        self,
        start: int,
    ) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
        """Return the V/I samples ``[start:n_steps_recorded)`` per channel.

        The streaming project sink calls this at every
        flush to append only the newly recorded tail to disk, so the
        recorder stays the single in-RAM buffer while a separate reader
        follows the run live.

        Parameters
        ----------
        start : int
            First sample index to return (the count already flushed).

        Returns
        -------
        dict[(str, int), (np.ndarray, np.ndarray)]
            ``(V_tail, I_tail)`` per ``(port_name, mode_idx)`` channel;
            empty arrays when no new samples exist.
        """
        # Design: DD-070 (streaming project sink).
        self._drain_stages()
        n = self._n_steps
        return {
            key: (
                np.asarray(self._V_buffers[key][start:n], dtype=float),
                np.asarray(self._I_buffers[key][start:n], dtype=float),
            )
            for key in self._V_buffers
        }

    def record(self, e: np.ndarray, h: np.ndarray) -> None:
        """Append one V and I sample per ``(port, mode)`` channel.

        On the CuPy backend, ports exposing the staged-recording
        interface (``record_gather_indices`` +
        ``project_V_samples``/``project_I_samples``) have their raw
        port-plane samples gathered into a device ring buffer instead
        of taking four blocking D2H round trips here; the
        buffers are drained — and the identical host dot products run —
        at :meth:`tail`, :meth:`finalize`, or when a buffer fills.  The
        NumPy path records immediately, exactly as before.
        """
        # Design: WP-G1 (device-side staged recording).
        if self._stage_list is None:
            self._init_stages(e, h)
        if self._staged:
            for p, scale, stage in zip(self._ports, self._scales, self._stage_list):
                if stage is not None:
                    stage.stage(e, h)
                else:
                    self._append_sample(p, scale, p.project_V(e), p.project_I(h))
            self._n_steps += 1
            if any(s is not None and s.full for s in self._stage_list):
                self._drain_stages()
            return
        for p, scale in zip(self._ports, self._scales):
            V = p.project_V(e)
            I = p.project_I(h)
            if scale is not None:
                V = V * scale
                I = I * scale
            for m in range(p.n_modes):
                self._V_buffers[(p.name, m)].append(float(V[m]))
                self._I_buffers[(p.name, m)].append(float(I[m]))
        self._n_steps += 1

    def _init_stages(self, e, h) -> None:
        """Decide once, at the first sample, which ports stage (WP-G1)."""
        if hasattr(e, "get"):
            self._stage_list = [
                _DevicePortStage(p, e, h) if hasattr(p, "record_gather_indices") else None
                for p in self._ports
            ]
        else:
            self._stage_list = [None] * len(self._ports)
        self._staged = any(s is not None for s in self._stage_list)

    def _append_sample(self, p, scale, V, I) -> None:
        if scale is not None:
            V = V * scale
            I = I * scale
        for m in range(p.n_modes):
            self._V_buffers[(p.name, m)].append(float(V[m]))
            self._I_buffers[(p.name, m)].append(float(I[m]))

    def _drain_stages(self) -> None:
        """Materialise all staged samples into the host V/I lists."""
        if not self._staged:
            return
        for p, scale, stage in zip(self._ports, self._scales, self._stage_list):
            if stage is None or stage.n_staged == 0:
                continue
            e_block, h_block = stage.drain()
            for k in range(e_block.shape[0]):
                self._append_sample(
                    p,
                    scale,
                    p.project_V_samples(e_block[k]),
                    p.project_I_samples(h_block[k]),
                )

    def finalize(
        self,
        n_steps_actual: int | None = None,
    ) -> dict[tuple[str, int], tuple[Signal1D, Signal1D]]:
        """Convert all buffers to ``Signal1D`` pairs, optionally trimmed.

        Parameters
        ----------
        n_steps_actual : int, optional
            Actual number of leapfrog steps the solver ran.  When
            provided and smaller than :attr:`n_steps_recorded`, V and I
            buffers are truncated to ``n_steps_actual`` samples to drop
            the zero-padded tail.  ``None`` (default) keeps all
            recorded samples.

        Returns
        -------
        dict[(str, int), (Signal1D, Signal1D)]
            Mapping ``(port_name, mode_idx) -> (V_signal, I_signal)``.
            Both signals in a pair share the same time axis
            ``t = arange(N) * dt``.
        """
        if n_steps_actual is not None and n_steps_actual < 0:
            raise ValueError("n_steps_actual must be non-negative")

        self._drain_stages()
        n_kept = (
            self._n_steps
            if n_steps_actual is None
            else min(
                n_steps_actual,
                self._n_steps,
            )
        )
        t = np.arange(n_kept) * self._dt

        result: dict[tuple[str, int], tuple[Signal1D, Signal1D]] = {}
        for key in self._V_buffers:
            label, m = key
            V_buf = self._V_buffers[key][:n_kept]
            I_buf = self._I_buffers[key][:n_kept]
            V_signal = Signal1D(
                t=t,
                values=np.asarray(V_buf, dtype=float),
                dt=self._dt,
                label=f"{label}_mode{m}_V",
            )
            I_signal = Signal1D(
                t=t,
                values=np.asarray(I_buf, dtype=float),
                dt=self._dt,
                label=f"{label}_mode{m}_I",
            )
            result[key] = (V_signal, I_signal)
        return result

    def __repr__(self) -> str:
        return (
            f"PortSignalRecorder(dt={self._dt:.3e}, "
            f"channels={len(self._V_buffers)}, steps={self._n_steps})"
        )
