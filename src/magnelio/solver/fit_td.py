"""
FIT Time-Domain Solver — Leapfrog (Yee) scheme.

Implements the half-step leapfrog update:

    E^{n+1} = α_E · E^n + β_E · C^T · H^{n+1/2}
    H^{n+3/2} = α_H · H^{n+1/2} - β_H · C · E^{n+1}

PEC boundaries are enforced after each E update by zeroing PEC edges.
CPML auxiliary fields are updated alongside the main update loop.

See spec.md for the FIT update equations.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from magnelio._backend.array_api import (
    auto_fallback_reason,
    backend_summary,
    copy_into,
    resolve_backend,
    resolve_precision,
)
from magnelio._fields.field_arrays import FieldState
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    build_M_sigma,
    build_M_sigma_m,
    flatten_port_plane_mass,
    flatten_port_plane_mu,
    flatten_port_plane_pec_mask,
)
from magnelio._operators.numba_kernels import (
    pack_block_ids,
    update_E_fused,
    update_E_fused_cuda,
    update_E_stencil,
    update_H_fused,
    update_H_fused_cuda,
    update_H_stencil,
)
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.boundaries.pmc import PMCBoundary
from magnelio.mesh.mesher import Mesh
from magnelio.solver._dispersion import DispersionOperator
from magnelio.solver._gpu_graphs import CudaGraphPhases, graphs_enabled
from magnelio.solver._sibc import SIBCOperator
from magnelio.solver._tile_skip import (
    PMC_SHELL,
    build_tile_skip_plan,
    tile_skip_enabled,
)

# Port-signal stall watchdog (KB-008 fix, DD-122): arming floor below the
# run peak, and the minimum window length in envelope checks.  Both are
# deliberately internal — the only public runtime knob is the cap itself
# (``max_time_steps``), which the projection is measured against.
_STALL_ARM_DB = 40.0
_STALL_MIN_WINDOW_CHECKS = 10

# One banner per process: the "auto" device probe is cached per process
# (resolve_backend), so the choice cannot change between runs and
# repeating the line for every excitation run of an S-parameter analysis
# would be noise.
_AUTO_BANNER_SHOWN = False


class _SignalStallDetector:
    """Watchdog that proves the port-signal stop threshold unreachable.

    Band-edge (waveguide cut-off) content decays *algebraically* — its
    ``|V|``-envelope slope in dB per step tends to zero — so a dB-below-peak
    threshold sitting just under the plateau level is never crossed and
    an unbounded run marches until the runtime cap.  This detector
    watches the envelope samples the stop criterion already polls: once
    armed (level below ``arm_db`` under the run peak), it fits a
    straight line through the last ``window`` samples and extrapolates
    the step at which the threshold would be crossed.  If that step lies
    beyond ``cap_step``, the run can stop *now* — by construction the
    same outcome the cap would deliver, minus the wasted marching.  A
    genuine exponential ring-down (constant dB-per-step slope) that
    reaches the threshold before the cap is never interrupted.
    """

    def __init__(self, arm_db: float, window: int, cap_step: int) -> None:
        self.arm_db = float(arm_db)
        self.window = max(int(window), 2)
        self.cap_step = int(cap_step)
        self.slope_db_per_step: float | None = None
        self._samples: list[tuple[int, float]] = []

    def reset(self) -> None:
        """Drop the window (new envelope peak — the decay starts over)."""
        self._samples.clear()

    def observe(self, n: int, sig_db: float, threshold_db: float) -> bool:
        """Feed one envelope sample; ``True`` = threshold provably out
        of reach before the cap.

        ``sig_db`` and ``threshold_db`` are dB relative to the running
        peak (both negative, ``threshold_db < sig_db``).
        """
        if sig_db > -self.arm_db:
            # Not armed (or the envelope recovered above the arming
            # floor): restart the window so the fit never mixes regimes.
            self._samples.clear()
            return False
        self._samples.append((n, float(sig_db)))
        if len(self._samples) > self.window:
            del self._samples[0]
        elif len(self._samples) < self.window:
            return False
        steps = np.array([s[0] for s in self._samples], dtype=np.float64)
        vals = np.array([s[1] for s in self._samples], dtype=np.float64)
        # Least-squares line through the window — robust against the
        # envelope beating of multi-mode ring-down.
        slope = float(np.polyfit(steps, vals, 1)[0])
        self.slope_db_per_step = slope
        remaining = threshold_db - float(vals[-1])
        if remaining >= 0.0:  # already past the threshold — not our call
            return False
        if slope >= 0.0:
            return True  # flat or rising: unreachable at any horizon
        return n + remaining / slope > self.cap_step


@dataclass
class FITTimeDomainSolver:
    """FIT leapfrog solver.

    Args:
        mesh:                The simulation mesh.
        boundary_conditions: Dict mapping face names to BC objects.
        ports:               List of objects implementing the
                             :class:`magnelio.ports.base.Port` protocol
                             (``PortOperatorLumped``,
                             ``PortOperatorModal``).  Excitation is
                             configured on each operator via
                             ``set_excitation`` before ``run()``.
        sources:             List of source objects (e.g. plane-wave TF/SF).
        total_time_steps:    Number of leapfrog steps, or ``None`` for an
                             unbounded run that marches until the energy
                             criterion (or a graceful stop) ends it —
                             ``None`` requires ``energy_stop_db``.
        dt:                  Time step [s].
        verbose:             Print progress during simulation.
        energy_stop_db:      Stop when energy decays by this many dB
                             below peak.
        port_signal_stop_db: Stop when the modal-port ``|V|`` envelope
                             decays by this many dB below its run peak
                             — see the field comment.
        max_time_steps:      Runtime cap for unbounded runs (absolute
                             step bound; warns and stops when no
                             criterion fired) — see the field comment.
        recorder:            Optional :class:`PortSignalRecorder` —
                             receives V/I from each port at every step.
        backend:             ``"auto"`` (default) runs on the GPU via
                             CuPy when CuPy and a CUDA device are
                             available and falls back to the NumPy CPU
                             backend with a one-time notice otherwise;
                             ``"cupy"`` requires the GPU (raises with a
                             clear message when unavailable);
                             ``"numpy"`` forces the CPU backend.
    """

    mesh: Mesh
    boundary_conditions: dict = field(default_factory=dict)
    ports: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    total_time_steps: int | None = 1000
    dt: float = 1e-12
    verbose: bool = True
    energy_stop_db: float | None = None
    # Stop when every modal port's |V| envelope (windowed max between
    # checks — zero-crossing-proof) has decayed by this many dB below its
    # run peak (DD-096).  The robust termination for shielded lossless
    # structures, where TM-cut-off (k_z = 0) cavity modes hold the
    # stored energy at a plateau no port-plane boundary scheme can
    # reach, so ``energy_stop_db`` alone never fires; the S-parameter
    # deliverable depends only on the port signals this criterion
    # watches.  Either criterion may bound an unbounded run; whichever
    # fires first ends it.
    port_signal_stop_db: float | None = None
    # Arming step for the port-signal criterion (DD-114): it may only
    # fire at step >= this value.  Guards against the quiet gap between
    # the excitation leaving the driven port and the (possibly heavily
    # attenuated) response reaching the far ports — the envelope can
    # transiently sit far below the incident peak there.  The analyses
    # pass the auto-sized step estimate (2·t0 + 25 diagonal transits),
    # so every signal has long arrived once the criterion arms.
    # ``None`` → armed from step 0 (historical DD-096 behaviour).
    port_signal_min_steps: int | None = None
    # Runtime cap for *unbounded* runs (DD-122): absolute step bound at
    # which the march is ended with a RuntimeWarning and
    # ``stop_reason == "runtime_cap"`` when no stop criterion has fired.
    # The industry-standard backstop against criterion-defeating
    # ring-down (band-edge plateaus, unexpectedly high Q).  Also the
    # horizon of the stall watchdog: the port-signal criterion stops
    # early ("port_signal_stall") once the fitted envelope slope proves
    # the threshold unreachable before this cap.  ``None`` disables both
    # (march forever, watch it live / resume).  Ignored on bounded runs
    # — an explicit ``total_time_steps`` is a user decision that wins.
    max_time_steps: int | None = None
    # Steps between energy checks (+ live-stream flushes).  ``None`` →
    # derived from ``total_time_steps`` (``min(100, n/20)``, or 100 when
    # unbounded).  Set it to decouple the energy-check cadence from the
    # step cap — an unbounded run then stops at the same step a bounded
    # one would (DD-070, WP-S7).
    energy_check_interval: int | None = None
    recorder: object | None = None  # PortSignalRecorder | None
    diagnostics: list = field(default_factory=list)
    monitors: list = field(default_factory=list)
    sink: object | None = None  # streaming project sink (DD-070)
    backend: str = "auto"  # "auto" | "numpy" | "cupy"
    # Time-loop scalar precision (plan WP1 / DD-094): "single" (float32 —
    # matches commercial FIT/FDTD tools, halves memory + lifts GPU
    # throughput on consumer FP64-crippled cards) or "double" (float64,
    # opt-in for high-Q / high-dynamic-range).  ``None`` (the default)
    # resolves to MAGNELIO_PRECISION else "single"; an explicit value wins
    # over the env (mirrors backend="cupy" vs MAGNELIO_BACKEND).  Orthogonal
    # to ``backend``.  DFT/port/eigenmode/geometry stay double regardless.
    precision: str | None = None  # None | "single" | "double"
    # TD surface-impedance walls (WP-D4): an SIBCSpec (surfaces + fits)
    # or None for the untouched PEC path.  The analysis wires this behind its
    # wall_model switch (WP-D5); tests hand-build the spec.
    sibc: object | None = None  # SIBCSpec | None

    # Internal — set by setup()
    _fields: FieldState | None = field(default=None, repr=False, init=False)
    _curl_bufs: tuple | None = field(default=None, repr=False, init=False)
    # Dead-tile skip (TILE_SKIP_PLAN): per-component packed live-tile
    # id device arrays for the fused CUDA kernels, one-time zeroing
    # indices, and the skip statistics; all None when skipping is off
    # (CPU backend, MAGNELIO_TILE_SKIP=0, sources/unsafe BCs, or no
    # dead tiles).
    _tile_blocks_E: dict | None = field(default=None, repr=False, init=False)
    _tile_blocks_H: dict | None = field(default=None, repr=False, init=False)
    _tile_zero_E: object | None = field(default=None, repr=False, init=False)
    _tile_zero_H: object | None = field(default=None, repr=False, init=False)
    _tile_skip_stats: dict | None = field(default=None, repr=False, init=False)
    _alpha_E: np.ndarray | None = field(default=None, repr=False, init=False)
    _beta_E: np.ndarray | None = field(default=None, repr=False, init=False)
    _alpha_H: np.ndarray | None = field(default=None, repr=False, init=False)
    _beta_H: np.ndarray | None = field(default=None, repr=False, init=False)
    _pec_mask_E: np.ndarray | None = field(default=None, repr=False, init=False)
    # True when a BC type may write E onto PEC edges without a beta_E
    # factor (periodic / unknown types) — keeps one per-step
    # e[pec_idx] = 0 re-enforcement alive; set in setup().
    _pec_reenforce_after_bc: bool = field(default=False, repr=False, init=False)
    # ADE pole-current operator for dispersive materials (DD-084);
    # None on meshes without a dispersive material.
    _dispersion: object | None = field(default=None, repr=False, init=False)
    _dispersion_mu: object | None = field(default=None, repr=False, init=False)
    # TD-SIBC wall operator (WP-D4); None without an SIBC spec.
    _sibc: object | None = field(default=None, repr=False, init=False)
    # CUDA-graph dispatcher of the two device phases (WP-G3); set per
    # run() on the CuPy backend, None on CPU or MAGNELIO_GPU_GRAPHS=0.
    _gpu_graphs: object | None = field(default=None, repr=False, init=False)
    # Time-loop dtypes resolved from ``precision`` in setup() (plan WP1).
    # ``_real_dtype`` is the field/coefficient scalar; ``_complex_dtype``
    # pairs with it for field-local complex state (ADE pole currents).
    _real_dtype: object = field(default=None, repr=False, init=False)
    _complex_dtype: object = field(default=None, repr=False, init=False)
    # Flat edge-count offsets (set in setup)
    _n_Ex: int = field(default=0, repr=False, init=False)
    _n_Ey: int = field(default=0, repr=False, init=False)
    _n_Hx: int = field(default=0, repr=False, init=False)
    _n_Hy: int = field(default=0, repr=False, init=False)
    # Energy monitoring (set in setup, filled in run)
    _M_eps_diag: np.ndarray | None = field(default=None, repr=False, init=False)
    _M_mu_diag: np.ndarray | None = field(default=None, repr=False, init=False)
    _peak_energy: float = field(default=0.0, repr=False, init=False)
    _peak_signal: float = field(default=0.0, repr=False, init=False)
    _actual_steps: int = field(default=0, repr=False, init=False)
    # Why the last run() ended (DD-122): "steps" (bounded run completed),
    # "energy", "port_signal", "port_signal_stall", "runtime_cap", or
    # "aborted".  ``_final_signal_db`` is the |V|-envelope level below
    # peak at the stop (None when the port criterion never sampled).
    _stop_reason: str | None = field(default=None, repr=False, init=False)
    _final_signal_db: float | None = field(default=None, repr=False, init=False)
    # First step of the marching loop — nonzero after load_state_dict()
    # so a resumed run continues where the checkpoint left off (DD-070).
    _resume_step: int = field(default=0, repr=False, init=False)
    # Cooperative graceful-stop flags (DD-070, WP-S7): a SIGINT handler
    # (or request_stop()) sets _stop_requested; the loop breaks at the
    # next top-of-iteration — a consistent leapfrog pair — and marks
    # _aborted so the caller can persist a resumable checkpoint.
    _stop_requested: bool = field(default=False, repr=False, init=False)
    _aborted: bool = field(default=False, repr=False, init=False)
    _energy_trace: np.ndarray | None = field(default=None, repr=False, init=False)
    # Faces that host a modal port (PEC skipped on these faces, DD-021)
    _port_faces: set = field(default_factory=set, repr=False, init=False)

    def setup(self) -> None:
        """Pre-compute operators and allocate field arrays."""
        self._warn_on_uncovered_bbox_faces()

        mesh = self.mesh
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        dt = self.dt

        xp = resolve_backend(self.backend)
        self._xp = xp
        self._use_gpu = xp is not np
        if self.verbose and self.backend == "auto":
            global _AUTO_BANNER_SHOWN
            if not _AUTO_BANNER_SHOWN:
                reason = auto_fallback_reason() if not self._use_gpu else None
                suffix = f" — no usable GPU ({reason})" if reason else ""
                print(f"  FIT-TD | backend 'auto': {backend_summary(xp)}{suffix}")
                _AUTO_BANNER_SHOWN = True
        real_dtype, complex_dtype = resolve_precision(self.precision)
        self._real_dtype = real_dtype
        self._complex_dtype = complex_dtype
        self._fields = FieldState.zeros(Nx, Ny, Nz, xp=xp, dtype=real_dtype)

        # Curl buffers: only needed for stencil fallback path.
        # Fused paths (CUDA GPU / Numba CPU) need no temporaries.
        use_cuda_gpu = update_E_fused_cuda is not None and self._use_gpu
        use_numba_cpu = update_E_fused is not None and not self._use_gpu
        if use_cuda_gpu or use_numba_cpu:
            self._curl_bufs = None
        else:
            self._curl_bufs = (
                xp.empty((Nx, Ny + 1, Nz + 1), dtype=real_dtype),  # curl_Ex
                xp.empty((Nx + 1, Ny, Nz + 1), dtype=real_dtype),  # curl_Ey
                xp.empty((Nx + 1, Ny + 1, Nz), dtype=real_dtype),  # curl_Ez
                xp.empty((Nx + 1, Ny, Nz), dtype=real_dtype),  # curl_Hx
                xp.empty((Nx, Ny + 1, Nz), dtype=real_dtype),  # curl_Hy
                xp.empty((Nx, Ny, Nz + 1), dtype=real_dtype),  # curl_Hz
            )

        M_eps = build_M_eps(mesh)
        M_mu = build_M_mu(mesh)
        M_sigma = build_M_sigma(mesh)

        # Flatten the boundary M_eps slab, the normal-face M_mu slab AND
        # the PEC mask slab on every modal port operator's face, so the
        # FIT-TD update at the port plane uses the same mass and
        # conductor contour the mode-solver computed its mode against.
        # See ``flatten_port_plane_mass`` / ``flatten_port_plane_mu`` /
        # ``flatten_port_plane_pec_mask`` docstrings for the rationale.
        modal_port_faces = []
        for op in self.ports:
            face = getattr(getattr(op, "plane", None), "face", None)
            if face is not None:
                modal_port_faces.append(face)
                M_eps = flatten_port_plane_mass(M_eps, mesh, face)
                M_mu = flatten_port_plane_mu(M_mu, mesh, face)
        if modal_port_faces:
            new_pec = mesh.pec_mask_edges
            for face in modal_port_faces:
                new_pec = flatten_port_plane_pec_mask(new_pec, mesh, face)
            # Solver-local view ONLY.  Writing the flattened mask back
            # into the caller's mesh (the historical object.__setattr__)
            # poisoned every LATER operator build on the same mesh: the
            # 2D mode solver reads pec_mask_edges, so a second run() on
            # a multi-face port setup computed its port modes against a
            # plane stripped of its wall/conductor contour and silently
            # projected the arriving field onto the wrong mode.
            mesh = replace(mesh, pec_mask_edges=new_pec)

        # Store diagonals for energy monitoring (DD-019)
        self._M_eps_diag = M_eps
        self._M_mu_diag = M_mu

        # Exponential (Crank–Nicolson) update coefficients for lossy E.
        # Dispersive materials (DD-084) add their trapezoidal pole-current
        # coefficient W to BOTH sides of the semi-implicit update — the
        # (dt/2)·W term appears in the numerator AND the denominator (the
        # pole current couples to e^{n+1} − e^n, not to the midpoint), so
        # with no dispersive material the expressions below reduce to the
        # exact lossless/σ-only coefficients (adding nothing, not 0.0).
        self._dispersion = DispersionOperator.from_mesh(mesh, dt, side="E")
        # M_eps = 0 marks an edge with no electric energy — an edge lying
        # wholly inside a conductor that the classifier left cat-2 and
        # unmasked (DD-147).  Mirror the H side below: alpha_E = 1 /
        # beta_E = 0 freezes e there, which is what masking the edge
        # would have done.  Dividing straight through instead put NaN in
        # both coefficients, and the NaN spread over the whole grid on
        # the first step — a run that produced NaN power waves without
        # ever failing.
        live_E = M_eps > 0
        if self._dispersion is not None:
            W_disp = 0.5 * dt * self._dispersion.W
            denom = np.where(live_E, M_eps + 0.5 * dt * M_sigma + W_disp, 1.0)
            self._alpha_E = np.where(
                live_E,
                (M_eps - 0.5 * dt * M_sigma + W_disp) / denom,
                1.0,
            )
        else:
            denom = np.where(live_E, M_eps + 0.5 * dt * M_sigma, 1.0)
            self._alpha_E = np.where(live_E, (M_eps - 0.5 * dt * M_sigma) / denom, 1.0)
        self._beta_E = np.where(live_E, dt / denom, 0.0)

        # Lossy H update, mirroring the E side (σ*, DD-081).
        # M_mu = 0 marks enlarged-cell-donated faces (WP-R5): the
        # exact beta_H = 0 freezes h there — the face's inertia lives
        # on its donor — so σ* is dropped on those faces too.  With
        # σ* = 0 everywhere this reduces bit-exactly to the lossless
        # coefficients (M_mu/M_mu == 1.0, dt/(M_mu+0) == dt/M_mu).
        M_sigma_m = np.where(M_mu > 0, build_M_sigma_m(mesh), 0.0)
        # TD-SIBC walls (WP-D4): the instantaneous part G_f·R_inst of the
        # restored wall-edge voltage multiplies h_mid exactly like a
        # magnetic surface conductivity, so it folds as a plain addition
        # to the M_sigma_m diagonal (DERIVATION.md §3); the branch
        # history is added after the H kernel below.  With no spec the
        # operator is None and M_sigma_m is untouched — the PEC path
        # stays bit-identical (the DD-084 add-nothing rule, Gate A).
        if self.sibc is not None:
            self._sibc = SIBCOperator.from_spec(
                self.sibc,
                mesh.grid,
                dt,
                frozen=(M_mu <= 0),
            )
        else:
            self._sibc = None
        if self._sibc is not None:
            M_sigma_m = M_sigma_m + self._sibc.W
        # mu(omega) poles (DD-089) fold their W_m into BOTH sides exactly
        # like the E-side W — same derivation, same masking as sigma_m
        # (donor faces are excluded from the subsets AND carry W_m = 0,
        # so their alpha_H/beta_H stay the frozen 1.0/0.0).  With no
        # mu-dispersive material these expressions are the DD-081 ones,
        # array-equal.
        self._dispersion_mu = DispersionOperator.from_mesh(
            mesh,
            dt,
            side="H",
            frozen=(M_mu <= 0),
        )
        if self._dispersion_mu is not None:
            W_mu = np.where(M_mu > 0, 0.5 * dt * self._dispersion_mu.W, 0.0)
            denom_H = np.where(
                M_mu > 0,
                M_mu + 0.5 * dt * M_sigma_m + W_mu,
                1.0,
            )
            self._alpha_H = np.where(
                M_mu > 0,
                (M_mu - 0.5 * dt * M_sigma_m + W_mu) / denom_H,
                1.0,
            )
        else:
            denom_H = np.where(M_mu > 0, M_mu + 0.5 * dt * M_sigma_m, 1.0)
            self._alpha_H = np.where(
                M_mu > 0,
                (M_mu - 0.5 * dt * M_sigma_m) / denom_H,
                1.0,
            )
        self._beta_H = np.where(M_mu > 0, dt / denom_H, 0.0)

        # Precision (plan WP1): the per-step kernel multipliers cast down to
        # the field dtype — the fused CUDA kernel takes a single scalar_t for
        # both fields and coefficients, and the Numba path stays consistent.
        # The /denom arithmetic above ran in float64, so CFL/timestep
        # resolution is never single; only the stored result is cast.
        self._alpha_E = self._alpha_E.astype(self._real_dtype)
        self._beta_E = self._beta_E.astype(self._real_dtype)
        self._alpha_H = self._alpha_H.astype(self._real_dtype)
        self._beta_H = self._beta_H.astype(self._real_dtype)
        # ``_M_eps_diag``/``_M_mu_diag`` feed ONLY the energy monitor (the
        # coefficients above were already built from the float64 ``M_eps`` /
        # ``M_mu`` locals), so they carry the field dtype too (WP1b — a
        # full-grid array each, as large as the fields; the biggest remaining
        # memory lever).  The energy reduction forces float64 accumulation
        # (see run()), so the stop criterion is unaffected.
        self._M_eps_diag = self._M_eps_diag.astype(self._real_dtype)
        self._M_mu_diag = self._M_mu_diag.astype(self._real_dtype)

        # PEC edge mask — flat boolean array of length n_E
        pec = mesh.pec_mask_edges  # shape (3, n_max)
        self._n_Ex = Nx * (Ny + 1) * (Nz + 1)
        self._n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Ez = (Nx + 1) * (Ny + 1) * Nz
        self._pec_mask_E = np.concatenate(
            [
                pec[0, : self._n_Ex],
                pec[1, : self._n_Ey],
                pec[2, :n_Ez],
            ]
        )
        # Integer index array for O(n_pec) enforcement instead of O(n_total)
        if self._pec_mask_E.any():
            self._pec_idx_E = np.where(self._pec_mask_E)[0]
        else:
            self._pec_idx_E = None

        # PEC edges are frozen degrees of freedom: exact alpha_E =
        # beta_E = 0 makes the update kernel itself hold them at +0.0
        # (0*e + 0*curl == +0.0 + ±0.0 == +0.0 in IEEE once e starts at
        # 0), and every consumer that could repopulate them scales with
        # beta_E — the CPML correction is Δe = beta_E[edge]·ψ, TF/SF
        # E-injection is beta_E-weighted — so they vanish there too.
        # This is the WP-R5 "exact beta_H = 0 freezes donor faces"
        # technique applied to the E side; it replaces the former
        # unconditional per-step e[pec_idx] = 0 scatter (measured
        # 0.83 ms/step at 379k cells / 343k PEC edges — the single
        # largest per-step cost after the curl kernels, see
        # PERFORMANCE_PROFILING_PLAN.md Workstream 2).
        if self._pec_idx_E is not None:
            self._alpha_E[self._pec_idx_E] = 0.0
            self._beta_E[self._pec_idx_E] = 0.0

        # One per-step re-enforcement remains ONLY for BC types that
        # write E directly without a beta_E factor: PeriodicBoundary
        # copies whole E slices (and unknown/user BC types are treated
        # conservatively).  PEC (writes exact zeros), PMC (E no-op) and
        # CPML (beta_E-scaled) are provably safe without it.
        _pec_safe_bcs = (PECBoundary, PMCBoundary, CPMLBoundary)
        self._pec_reenforce_after_bc = self._pec_idx_E is not None and any(
            not isinstance(bc, _pec_safe_bcs) for bc in self.boundary_conditions.values()
        )

        # Dead-tile skip plan (TILE_SKIP_PLAN): analysed on the final
        # host-side coefficients, before their GPU transfer below.
        # The plan self-disables on TF/SF field sources and on BC
        # types outside the safe list; PMC faces suppress curl-dead
        # H skipping in their outermost layer.
        self._tile_blocks_E = self._tile_blocks_H = None
        self._tile_zero_E = self._tile_zero_H = None
        self._tile_skip_stats = None
        if update_E_fused_cuda is not None and self._use_gpu and tile_skip_enabled():
            plan = build_tile_skip_plan(
                Nx=Nx,
                Ny=Ny,
                Nz=Nz,
                alpha_E=self._alpha_E,
                beta_E=self._beta_E,
                alpha_H=self._alpha_H,
                beta_H=self._beta_H,
                has_field_sources=bool(self.sources),
                has_unsafe_bcs=any(
                    not isinstance(bc, _pec_safe_bcs) for bc in self.boundary_conditions.values()
                ),
                boundary_shell_faces={
                    face: PMC_SHELL
                    for face, bc in self.boundary_conditions.items()
                    if isinstance(bc, PMCBoundary)
                },
            )
            if plan is not None and plan.stats["total"] > 0.0:
                self._tile_blocks_E = {
                    n: xp.asarray(pack_block_ids(plan.live_blocks[n], plan.block_grids[n]))
                    for n in ("Ex", "Ey", "Ez")
                }
                self._tile_blocks_H = {
                    n: xp.asarray(pack_block_ids(plan.live_blocks[n], plan.block_grids[n]))
                    for n in ("Hx", "Hy", "Hz")
                }
                self._tile_zero_E = xp.asarray(plan.dead_zero_idx_E)
                self._tile_zero_H = xp.asarray(plan.dead_zero_idx_H)
                self._tile_skip_stats = plan.stats
                if self.verbose:
                    print(f"Tile skip: {plan.stats['total']:.1%} of kernel elements in dead tiles")

        # H-face offsets
        self._n_Hx = (Nx + 1) * Ny * Nz
        self._n_Hy = Nx * (Ny + 1) * Nz

        # Transfer coefficients and indices to GPU
        if self._use_gpu:
            self._alpha_E = xp.asarray(self._alpha_E)
            self._beta_E = xp.asarray(self._beta_E)
            self._alpha_H = xp.asarray(self._alpha_H)
            self._beta_H = xp.asarray(self._beta_H)
            self._M_eps_diag = xp.asarray(self._M_eps_diag)
            self._M_mu_diag = xp.asarray(self._M_mu_diag)
            if self._pec_idx_E is not None:
                self._pec_idx_E = xp.asarray(self._pec_idx_E)

        # Bind the ADE pole states to the final beta / backend
        if self._dispersion is not None:
            self._dispersion.bind(self._beta_E, xp)  # DD-084
        if self._dispersion_mu is not None:
            self._dispersion_mu.bind(self._beta_H, xp)  # DD-089
        if self._sibc is not None:
            self._sibc.bind(self._beta_H, xp)  # WP-D4

        # Determine which faces host a modal port (DD-021).
        # PEC is skipped on these faces; the modal operator's
        # ``update_e`` writes the field there from the mode expansion.
        # An absorbing face hands the columns behind each port window
        # to the port (DD-198): the CPML is switched off there.
        self._port_faces = set()
        port_windows: dict[str, list[dict]] = {}
        for op in self.ports:
            plane = getattr(op, "plane", None)
            face = getattr(plane, "face", None)
            value = getattr(face, "value", None)
            if isinstance(value, str):
                # BoxFace values are "x_min" etc.; BC keys are "xmin".
                key = value.replace("_", "")
                self._port_faces.add(key)
                u_win = getattr(plane, "u_node_window", None)
                v_win = getattr(plane, "v_node_window", None)
                if u_win is not None and v_win is not None:
                    bc = self.boundary_conditions.get(key)
                    if hasattr(bc, "set_port_windows"):
                        from magnelio.ports._modal.factory import (  # noqa: PLC0415
                            validate_absorbing_face_window,
                        )

                        n_cells = (Nx, Ny, Nz)
                        whole = (
                            u_win[0] == 0
                            and u_win[1] == n_cells[int(face.u_axis)]
                            and v_win[0] == 0
                            and v_win[1] == n_cells[int(face.v_axis)]
                        )
                        validate_absorbing_face_window(
                            face, plane, mesh, whole_face=whole, absorbing=True
                        )
                    port_windows.setdefault(key, []).append(
                        {
                            int(face.u_axis): tuple(int(i) for i in u_win),
                            int(face.v_axis): tuple(int(i) for i in v_win),
                        }
                    )

        # Initialize CPML and pass PEC mask for PEC-in-PML stability.
        # xp is forwarded so CPML state lives on this solver's backend
        # (the module-global get_xp() is no longer consulted here).
        for face_key, bc in self.boundary_conditions.items():
            if hasattr(bc, "initialize"):
                bc.initialize(dt, xp=xp, dtype=real_dtype)
            if hasattr(bc, "set_pec_mask"):
                bc.set_pec_mask(
                    self._pec_mask_E,
                    Nx,
                    Ny,
                    Nz,
                    material_id=mesh.material_id,
                    material_library=mesh.material_library,
                    xp=xp,
                )
            if hasattr(bc, "set_port_windows") and port_windows.get(face_key):
                bc.set_port_windows(port_windows[face_key], xp=xp)

        # Attach sources (TF/SF plane wave needs solver coefficients)
        for src in self.sources:
            if hasattr(src, "attach"):
                src.attach(self)

        # A source may have written an initial field into the state.  The
        # modal ports difference their plane against the previous step, so
        # they must capture that field once before the first one — with
        # ``V_prev = 0`` the Mur formula would read the initial condition
        # as a step and reflect it (DD-224).  Done after *all* sources, so
        # superposed initial fields are seen as one.
        if any(getattr(src, "writes_initial_field", False) for src in self.sources):
            for op in self.ports:
                if hasattr(op, "initialize_state"):
                    op.initialize_state(self._fields.e_flat)

        # Attach diagnostic probes (after BCs are initialized so CPML sigma is available)
        for probe in self.diagnostics:
            if hasattr(probe, "attach"):
                probe.attach(mesh)

        # Attach monitors
        for mon in self.monitors:
            if hasattr(mon, "attach"):
                mon.attach(mesh)

    _ALL_FACES = frozenset({"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})

    def _warn_on_uncovered_bbox_faces(self) -> None:
        """Warn if any bbox face has no boundary mechanism configured.

        Each of the six bbox faces should have a boundary closure: a
        :class:`BoundaryCondition` (PEC/PMC/CPML/Periodic), a legacy
        port (``port.face``), or a :class:`PortOperatorModal`
        (``port_operator.plane.face``).  Faces without any closure are
        evolved freely by the curl operator using boundary-edge
        neighbours, which makes the simulation effectively open at
        those faces — almost always a setup mistake (the foot-gun
        that produced session 46's misdiagnosis).

        The warning is silenced when the solver is configured with
        zero BCs *and* zero ports (typical for solver-internals unit
        tests that never run a long evolution).  It also silenced
        when full coverage is reached.  Anything in between — some
        coverage but not full — fires.
        """
        covered: set[str] = set(self.boundary_conditions.keys())
        for op in self.ports:
            plane = getattr(op, "plane", None)
            face = getattr(plane, "face", None)
            value = getattr(face, "value", None)
            if isinstance(value, str):
                # BoxFace values are "x_min" etc.; BC keys are "xmin".
                covered.add(value.replace("_", ""))

        if not covered:
            # No BCs and no ports: likely a solver-internals unit test
            # that doesn't care about boundary closure.  Silent.
            return

        missing = sorted(self._ALL_FACES - covered)
        if not missing:
            return

        warnings.warn(
            f"FITTimeDomainSolver: bbox face(s) {missing} have neither a "
            f"BoundaryCondition (PEC/PMC/CPML/Periodic) nor a modal port. "
            f"Tangential E on these faces will be evolved freely by the "
            f"curl operator — the domain is effectively open there, "
            f"which is almost always a setup mistake. "
            f"Pass `boundary_conditions={{...}}`.",
            UserWarning,
            stacklevel=3,
        )

    def run(self) -> FieldState:
        """Execute the leapfrog time-stepping loop."""
        if self._fields is None:
            self.setup()

        fields = self._fields
        pec_idx = self._pec_idx_E
        pec_reenforce = self._pec_reenforce_after_bc
        Nx, Ny, Nz = self.mesh.Nx, self.mesh.Ny, self.mesh.Nz
        n_steps = self.total_time_steps
        if n_steps is None and self.energy_stop_db is None and self.port_signal_stop_db is None:
            raise ValueError(
                "total_time_steps=None (unbounded run) needs energy_stop_db "
                "or port_signal_stop_db to provide a stop criterion",
            )
        unbounded = n_steps is None
        # Runtime cap (DD-122): bounds unbounded runs only; an explicit
        # total_time_steps wins.  Absolute step count, so a resumed run
        # needs a cap past its checkpoint.
        cap_steps = self.max_time_steps if unbounded else None
        if cap_steps is not None and cap_steps <= self._resume_step:
            raise ValueError(
                f"max_time_steps={cap_steps} does not advance past the "
                f"current step {self._resume_step}; pass a larger cap "
                f"(absolute step count) or None to march uncapped",
            )
        total_str = "∞" if unbounded else str(n_steps)
        self._stop_reason = None
        self._final_signal_db = None
        dt = self.dt

        # Flat material coefficients (needed by CPML update_E / update_H)
        alpha_E = self._alpha_E
        beta_E = self._beta_E
        alpha_H = self._alpha_H
        beta_H = self._beta_H

        # 3D component views (zero-copy into flat backing store)
        Ex, Ey, Ez = fields.Ex, fields.Ey, fields.Ez
        Hx, Hy, Hz = fields.Hx, fields.Hy, fields.Hz

        # Dispatch: CUDA fused (GPU) > Numba fused (CPU) > stencil (fallback)
        use_gpu = self._use_gpu
        use_cuda = update_E_fused_cuda is not None and use_gpu
        use_numba = update_E_fused is not None and not use_gpu
        if not use_cuda and not use_numba:
            cEx, cEy, cEz, cHx, cHy, cHz = self._curl_bufs

        # Reshaped material coefficient views (zero-copy into flat arrays)
        n_Ex, n_Ey = self._n_Ex, self._n_Ey
        n_Hx, n_Hy = self._n_Hx, self._n_Hy
        aEx = alpha_E[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
        aEy = alpha_E[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
        aEz = alpha_E[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)
        bEx = beta_E[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
        bEy = beta_E[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
        bEz = beta_E[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)
        aHx = alpha_H[:n_Hx].reshape(Nx + 1, Ny, Nz)
        aHy = alpha_H[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
        aHz = alpha_H[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)
        bHx = beta_H[:n_Hx].reshape(Nx + 1, Ny, Nz)
        bHy = beta_H[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
        bHz = beta_H[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)

        # Port faces where PEC is replaced by the modal operator (DD-021)
        port_faces = self._port_faces

        # ADE dispersion hooks (DD-084 on E, DD-089 on H) and the
        # TD-SIBC wall hook (WP-D4)
        dispersion = self._dispersion
        dispersion_mu = self._dispersion_mu
        sibc = self._sibc

        # Energy monitoring setup (DD-019).  The cadence can be pinned
        # (``energy_check_interval``) so an unbounded run checks energy on
        # the same grid a bounded one would — else derive it from the cap.
        energy_stop = self.energy_stop_db
        check_interval = (
            self.energy_check_interval
            if self.energy_check_interval is not None
            else (100 if unbounded else max(1, min(100, n_steps // 20)))
        )
        M_eps_diag = self._M_eps_diag
        M_mu_diag = self._M_mu_diag
        # Resume-aware: a loaded checkpoint restores the pre-decay peak,
        # so the energy-stop threshold is rebuilt here instead of waiting
        # for a new peak that a decaying resumed run will never reach.
        peak_energy = self._peak_energy
        energy_threshold = (
            peak_energy * 10 ** (-energy_stop / 10)
            if (energy_stop is not None and peak_energy > 0.0)
            else 0.0
        )
        energy_falling = False
        energy_trace: list[tuple[int, float]] = []
        # H half-step buffer of the energy evaluation (allocated on first use)
        h_prev = None
        # Port-signal stop criterion (DD-096): ports accumulate the |V|
        # envelope between checks; resume-aware like the energy peak.
        signal_stop = self.port_signal_stop_db
        signal_min_steps = self.port_signal_min_steps
        peak_signal = self._peak_signal
        signal_ports = (
            [op for op in self.ports if hasattr(op, "poll_signal_absmax")]
            if signal_stop is not None
            else []
        )
        if signal_stop is not None and not signal_ports:
            raise ValueError(
                "port_signal_stop_db needs at least one modal port",
            )
        # Stall watchdog (DD-122): only on capped unbounded runs — the
        # cap is the horizon the slope projection is measured against,
        # and cap=None is the explicit march-forever opt-out.  The
        # window spans half the transit estimate of physical time
        # (port_signal_min_steps carries the estimate on the analysis
        # path), so a slowly-ringing high-Q structure is judged on a
        # structure-scaled window, not a fixed check count.
        stall = None
        if signal_stop is not None and cap_steps is not None:
            window = max(
                _STALL_MIN_WINDOW_CHECKS,
                (signal_min_steps or 0) // (2 * check_interval),
            )
            stall = _SignalStallDetector(_STALL_ARM_DB, window, cap_steps)
        last_sig_db: float | None = None

        # Direct references to flat backing arrays (zero-copy)
        e = fields.e_flat
        h = fields.h_flat

        # Tile-skip invariant: the provably-zero elements must BE zero
        # before the march — unconditional normalisation covers resumed
        # checkpoints (skipped tiles are never written again).
        tile_blocks_E = self._tile_blocks_E
        tile_blocks_H = self._tile_blocks_H
        if self._tile_zero_E is not None:
            e[self._tile_zero_E] = 0.0
            h[self._tile_zero_H] = 0.0

        # ── Device-phase closures (WP-G3) ────────────────────────────
        # The two contiguous device-only step segments, shared verbatim
        # by the eager path (CPU and GPU) and the CUDA-graph capture:
        # capture records exactly the kernels the eager path would
        # launch, on the same pointers, so replays are bit-identical.

        def e_phase_device():
            # Stash e^n / h^n on the dispersive subsets — the pole
            # recursion couples to f^{n+1} − f^n (DD-084/DD-089).
            # Placed at the top of the iteration so f_prev is the final
            # field of the previous step, all corrections
            # (BC/ports/sources) included — on the H side those run
            # AFTER its kernel, so this is the only correct place.
            if dispersion is not None:
                dispersion.save_field(e)
            if dispersion_mu is not None:
                dispersion_mu.save_field(h)
            # SIBC midpoint drive needs h^{n+1/2} — same placement
            # rationale as the ADE stashes above (WP-D4).
            if sibc is not None:
                sibc.save_field(h)

            # ── E update ─────────────────────────────────────────────
            if use_cuda:
                update_E_fused_cuda(
                    Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz, blocks=tile_blocks_E
                )
            elif use_numba:
                update_E_fused(Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz)
            else:
                update_E_stencil(
                    Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz, cEx, cEy, cEz
                )

            # Complete the constitutive update on dispersive edges: the
            # kernel applied the W-folded alpha/beta; subtracting the
            # pole-history current finishes the implicit ADE solution,
            # then the pole states advance (DD-084).
            if dispersion is not None:
                dispersion.update_field(e)

            # No explicit PEC zeroing here: alpha_E = beta_E = 0 on PEC
            # edges makes the kernel hold them at exact +0.0 (setup).

            # ── BC E-corrections (skip PEC/PMC on port faces, DD-021) ─
            # Two passes: (1) CPML outer-face PEC + PML correction,
            #              (2) re-apply PEC after all CPML E-corrections.
            # PMC is an H-constraint and runs after the H-update below;
            # ``apply_E`` is a no-op on PMC, ``apply_H`` is a no-op on
            # PEC.
            for face, bc in self.boundary_conditions.items():
                if face in port_faces and hasattr(bc, "apply_E"):
                    pass
                elif hasattr(bc, "apply_E"):
                    bc.apply_E(fields)
                if hasattr(bc, "update_E"):  # CPML
                    bc.update_E(fields, beta_E)
            # Re-enforce PEC after all CPML E-corrections
            for face, bc in self.boundary_conditions.items():
                if face in port_faces:
                    continue
                if hasattr(bc, "apply_E"):
                    bc.apply_E(fields)

            # Global PEC re-enforcement — needed ONLY when a BC type
            # writes E without a beta_E factor (periodic slice copies,
            # unknown user BCs; see setup).  PEC/PMC/CPML never
            # repopulate PEC edges since beta_E = 0 there.
            if pec_reenforce:
                e[pec_idx] = 0.0

        def h_phase_device():
            # ── H update ─────────────────────────────────────────────
            if use_cuda:
                update_H_fused_cuda(
                    Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz, blocks=tile_blocks_H
                )
            elif use_numba:
                update_H_fused(Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz)
            else:
                update_H_stencil(
                    Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz, cHx, cHy, cHz
                )

            # Complete the constitutive update on mu-dispersive faces —
            # the H-side mirror of the E hook above (DD-089).
            if dispersion_mu is not None:
                dispersion_mu.update_field(h)

            # Complete the SIBC wall damping on the booked faces: the
            # kernel applied the W-folded alpha_H/beta_H; adding the
            # beta_H-weighted branch history finishes the implicit
            # solve, then the branch states advance on the midpoint
            # (WP-D4, DERIVATION.md §3).
            if sibc is not None:
                sibc.update_field(h)

            # CPML H-corrections
            for bc in self.boundary_conditions.values():
                if hasattr(bc, "update_H"):
                    bc.update_H(fields, beta_H)

            # ── PMC enforcement after H-update (skip on port faces) ──
            for face, bc in self.boundary_conditions.items():
                if face in port_faces:
                    continue
                if hasattr(bc, "apply_H"):
                    bc.apply_H(fields)

        # CUDA-graph capture of the two device phases (WP-G3): CuPy
        # backend only, MAGNELIO_GPU_GRAPHS=0 disables; capture failure
        # falls back to the eager path with one warning.
        gpu_graphs = CudaGraphPhases() if (use_cuda and graphs_enabled()) else None
        self._gpu_graphs = gpu_graphs

        # Cooperative graceful stop starts clear for this run (DD-070).
        self._stop_requested = False
        self._aborted = False

        # Unbounded runs march an open-ended counter (a stop criterion or
        # a graceful stop ends them) — bounded by the runtime cap when
        # one is set (DD-122); bounded runs a plain range.
        if unbounded:
            step_counter = (
                itertools.count(self._resume_step)
                if cap_steps is None
                else range(self._resume_step, cap_steps)
            )
        else:
            step_counter = range(self._resume_step, n_steps)
        for n in step_counter:
            # Break at the top of an iteration — the previous iteration
            # left a consistent leapfrog pair (E^{n}, H^{n+1/2}) and
            # _resume_step == n, so a checkpoint here is bit-exact (WP-S7).
            if self._stop_requested:
                break
            t = n * dt

            # ── E phase (device-only segment; WP-G3 graph site) ────────
            if gpu_graphs is not None:
                gpu_graphs.run_phase("E", e_phase_device)
            else:
                e_phase_device()

            # ── Source injection — TF/SF E-correction ──────────────────
            for src in self.sources:
                if hasattr(src, "inject_E"):
                    src.inject_E(fields, t + dt)

            # ── Port operators: unified update_e hook ───────────────────
            # ``e`` is at t^{n+1}, ``h`` at t^{n+1/2}.  Each port runs its
            # own E-side correction in place: lumped (Thévenin) for
            # ``PortOperatorLumped``, modal Mur + TF/SF for
            # ``PortOperatorModal``.  Excitation has been configured on
            # the operator beforehand via ``set_excitation``.
            for op in self.ports:
                op.update_e(fields, t + dt, dt)

            # ── V/I recording (Yee half-step stagger) ───────────────────
            # Recorded here so ``V_m`` sees the operator-corrected ``e``
            # at t^{n+1} and ``I_m`` sees ``h`` at t^{n+1/2}.
            if self.recorder is not None:
                self.recorder.record(e, h)

            # The stored energy below is evaluated from ``e`` at t^{n+1}
            # and the two ``h`` half-steps that straddle it, so the
            # H-side sample taken *before* this step's H update is kept
            # (only on the steps that actually evaluate; the buffer is
            # half a field state and is allocated on first use).
            if n % check_interval == 0:
                if h_prev is None:
                    h_prev = self._xp.empty_like(h)
                h_prev[:] = h

            # ── H phase (device-only segment; WP-G3 graph site) ────────
            if gpu_graphs is not None:
                gpu_graphs.run_phase("H", h_phase_device)
            else:
                h_phase_device()

            # Source injection — TF/SF H-correction
            for src in self.sources:
                if hasattr(src, "inject_H"):
                    src.inject_H(fields, t + dt + dt / 2)

            # Diagnostic probes — record after H update (E and H approximately co-temporal)
            for probe in self.diagnostics:
                if hasattr(probe, "record"):
                    probe.record(fields, t)

            # Monitors — record after H update
            for mon in self.monitors:
                mon.record(fields, n, t, dt)

            # Step n is fully done (E^{n+1}, H^{n+3/2} written, V/I recorded);
            # advance the completed-step count so a checkpoint taken at the
            # flush below carries the right n_completed (DD-070, WP-S7).
            self._resume_step = n + 1

            # Energy monitoring, status display, and early stopping (DD-019).
            # xp.sum(..., dtype=float64) forces double accumulation even when
            # the fields and M diagonals are float32 (WP1b): the reduction over
            # the whole grid keeps its dynamic range, so the energy-decay stop
            # criterion behaves identically to the double path.
            if n % check_interval == 0:
                # Leapfrog energy (DD-225): ``e`` sits at t^{n+1} while
                # ``h`` sits half a step later, so the naive
                # ½(e·M_ε·e + h·M_μ·h) oscillates at 2f with relative
                # amplitude sin(ω·dt/2) — 10 % on a 12-cells-per-
                # wavelength grid, and aliased into a ragged zig-zag by
                # the check cadence.  Pairing the two H half-steps that
                # straddle t^{n+1} gives the quantity the leapfrog
                # actually conserves.  Should a pathological medium make
                # it non-positive (the pairing is positive definite only
                # under the CFL limit), the naive form stands in, so the
                # decay stop always sees a usable number.
                energy_E = float(self._xp.sum(M_eps_diag * e * e, dtype=np.float64))
                energy_H = float(self._xp.sum(M_mu_diag * h_prev * h, dtype=np.float64))
                current_energy = 0.5 * (energy_E + energy_H)
                if not current_energy > 0.0:
                    current_energy = 0.5 * (
                        energy_E + float(self._xp.sum(M_mu_diag * h * h, dtype=np.float64))
                    )
                energy_trace.append((n, current_energy))

                # Stream the newly recorded V/I tail + this energy sample
                # to the project store (DD-070); a separate reader process
                # follows the run live via HDF5-SWMR.
                if self.sink is not None:
                    self.sink.flush(energy=(n, n * dt, current_energy))

                if current_energy > peak_energy:
                    peak_energy = current_energy
                    energy_falling = False
                    if energy_stop is not None:
                        energy_threshold = peak_energy * 10 ** (-energy_stop / 10)
                elif peak_energy > 0.0:
                    if not energy_falling:
                        energy_falling = True

                    # Early stopping
                    if energy_stop is not None and current_energy < energy_threshold:
                        self._stop_reason = "energy"
                        self._final_signal_db = last_sig_db
                        self._peak_energy = peak_energy
                        self._peak_signal = peak_signal
                        self._actual_steps = n + 1
                        self._resume_step = n + 1
                        self._energy_trace = self._build_energy_trace(
                            energy_trace,
                            dt,
                        )
                        if self.verbose:
                            energy_db = 10 * np.log10(max(current_energy, 1e-300) / peak_energy)
                            print(
                                f"\r  FIT-TD | time step {n + 1}/{total_str} "
                                f"| stored energy [dB] {energy_db:.1f}/"
                                f"{-energy_stop:.0f} "
                                f"| done (energy criterion)          "
                            )
                        for mon in self.monitors:
                            if hasattr(mon, "finalize"):
                                mon.finalize()
                        if self.sink is not None:
                            self.sink.flush()  # final V/I tail past last check
                        return fields

                # Port-signal stop (DD-096): the polled value is the
                # per-channel |V| envelope over the steps since the
                # last check, so a zero crossing at poll time cannot
                # fake a decayed signal.
                if signal_stop is not None:
                    sig = max(op.poll_signal_absmax() for op in signal_ports)
                    if sig > peak_signal:
                        peak_signal = sig
                        last_sig_db = 0.0
                        if stall is not None:
                            stall.reset()
                    elif peak_signal > 0.0:
                        sig_db = 20.0 * np.log10(max(sig, 1e-300) / peak_signal)
                        last_sig_db = sig_db
                        armed = signal_min_steps is None or n + 1 >= signal_min_steps
                        if armed and sig < peak_signal * 10.0 ** (-signal_stop / 20.0):
                            self._stop_reason = "port_signal"
                            self._final_signal_db = sig_db
                            self._peak_energy = peak_energy
                            self._peak_signal = peak_signal
                            self._actual_steps = n + 1
                            self._resume_step = n + 1
                            self._energy_trace = self._build_energy_trace(
                                energy_trace,
                                dt,
                            )
                            if self.verbose:
                                print(
                                    f"\r  FIT-TD | time step {n + 1}/{total_str} "
                                    f"| port signal [dB] {sig_db:.1f}/"
                                    f"{-signal_stop:.0f} "
                                    f"| done (port-signal criterion)     "
                                )
                            for mon in self.monitors:
                                if hasattr(mon, "finalize"):
                                    mon.finalize()
                            if self.sink is not None:
                                self.sink.flush()
                            return fields
                        # Stall watchdog (DD-122): the criterion did not
                        # fire — check whether its threshold is provably
                        # out of reach before the runtime cap (band-edge
                        # plateaus decay algebraically and hold the
                        # envelope just above the threshold forever).
                        if armed and stall is not None and stall.observe(n, sig_db, -signal_stop):
                            self._stop_reason = "port_signal_stall"
                            self._final_signal_db = sig_db
                            self._peak_energy = peak_energy
                            self._peak_signal = peak_signal
                            self._actual_steps = n + 1
                            self._resume_step = n + 1
                            self._energy_trace = self._build_energy_trace(
                                energy_trace,
                                dt,
                            )
                            slope_window = (
                                (stall.slope_db_per_step or 0.0) * stall.window * check_interval
                            )
                            warnings.warn(
                                f"port-signal stop criterion "
                                f"({-signal_stop:.0f} dB below peak) is "
                                f"unreachable before the runtime cap: the "
                                f"|V| envelope has stalled at {sig_db:.1f} dB "
                                f"(decaying {slope_window:.2e} dB over the "
                                f"last {stall.window * check_interval} "
                                f"steps) — typically band-edge content near "
                                f"a waveguide cut-off, which decays "
                                f"algebraically rather than exponentially.  "
                                f"Accepting the stall level as the "
                                f"effective floor and stopping at step "
                                f"{n + 1}.  The recorded signals carry a "
                                f"truncation residual of about this level "
                                f"(taper_signals=True bounds its spectral "
                                f"leakage); raise max_time_steps or pass "
                                f"total_time_steps to march further.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            if self.verbose:
                                print(
                                    f"\r  FIT-TD | time step {n + 1}/{total_str} "
                                    f"| port signal [dB] {sig_db:.1f}/"
                                    f"{-signal_stop:.0f} "
                                    f"| done (port signal stalled)       "
                                )
                            for mon in self.monitors:
                                if hasattr(mon, "finalize"):
                                    mon.finalize()
                            if self.sink is not None:
                                self.sink.flush()
                            return fields

                # Status display: absolute stored energy while the system
                # is still filling, decay in dB below the run peak once
                # the energy has passed its maximum.
                if self.verbose and energy_stop is not None:
                    if energy_falling and peak_energy > 0 and current_energy > 0:
                        energy_db = 10 * np.log10(max(current_energy, 1e-300) / peak_energy)
                        status = f"stored energy [dB] {energy_db:.1f}/{-energy_stop:.0f}"
                    else:
                        status = f"stored energy {current_energy:.3e} J"
                    print(
                        f"\r  FIT-TD | time step {n}/{total_str} | {status}          ",
                        end="",
                        flush=True,
                    )
                elif self.verbose:
                    pct = 100.0 * n / n_steps
                    print(
                        f"\r  FIT-TD {pct:5.1f}% ({n}/{n_steps})",
                        end="",
                        flush=True,
                    )

        # Graceful stop (top-of-loop break): the state is consistent at
        # _resume_step, so drain the V/I tail and persist a resume
        # checkpoint before marking the run aborted (DD-070, WP-S7).
        if self._stop_requested:
            self._aborted = True
            self._stop_reason = "aborted"
            self._final_signal_db = last_sig_db
            self._peak_energy = peak_energy
            self._peak_signal = peak_signal
            self._actual_steps = self._resume_step
            self._energy_trace = self._build_energy_trace(energy_trace, dt)
            for mon in self.monitors:
                if hasattr(mon, "finalize"):
                    mon.finalize()
            if self.verbose:
                print(
                    f"\r  FIT-TD | graceful stop at step {self._resume_step}/{total_str}          "
                )
            if self.sink is not None:
                self.sink.flush()
                self.sink.write_checkpoint()
            return fields

        # Loop exhausted: a bounded run completed its requested steps; an
        # unbounded run can only get here by hitting the runtime cap
        # (cap=None marches an open-ended counter) — the backstop case
        # where no stop criterion fired (DD-122).
        end_step = cap_steps if unbounded else n_steps
        self._peak_energy = peak_energy
        self._peak_signal = peak_signal
        self._actual_steps = end_step
        self._resume_step = end_step
        self._energy_trace = self._build_energy_trace(energy_trace, dt)
        self._final_signal_db = last_sig_db
        if unbounded:
            self._stop_reason = "runtime_cap"
            level = (
                f" (port signal at {last_sig_db:.1f} dB below peak)"
                if last_sig_db is not None
                else ""
            )
            warnings.warn(
                f"unbounded run hit the runtime cap of {cap_steps} steps "
                f"before any stop criterion fired{level}; the recorded "
                f"signals are truncated at this step.  Resume the run "
                f"(project store), raise max_time_steps, or pass an "
                f"explicit total_time_steps; max_time_steps=None removes "
                f"the cap entirely.",
                RuntimeWarning,
                stacklevel=2,
            )
            if self.verbose:
                print(
                    f"\r  FIT-TD | time step {end_step}/{total_str} "
                    f"| done (runtime cap)               "
                )
        else:
            self._stop_reason = "steps"
            if self.verbose:
                print(
                    f"\r  FIT-TD | time step {n_steps}/{n_steps} | done (n_periods limit)          "
                )

        for mon in self.monitors:
            if hasattr(mon, "finalize"):
                mon.finalize()
        if self.sink is not None:
            self.sink.flush()  # final V/I tail past the last check interval

        return fields

    # ------------------------------------------------------------------
    # Checkpoint / resume (DD-070, WP-S6/WP-S7)
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Ask the marching loop to stop gracefully at the next step.

        Cooperative and thread-safe by construction (a single bool flag):
        the loop checks it at the top of each iteration, so the in-flight
        step completes and the break lands on a consistent leapfrog pair.
        The run then persists a resume checkpoint (if a sink is attached)
        and returns with :attr:`_aborted` set.  Wire a ``SIGINT`` handler
        to this for Ctrl-C, or call it from a monitor / GUI.
        """
        self._stop_requested = True

    def state_dict(self) -> dict:
        """Capture the full leapfrog state for a bit-exact resume.

        Gathers everything the marching loop mutates — the E/H field, the
        completed-step count and pre-decay energy peak, and every
        stateful boundary (CPML ψ) and port (Mur previous-values, TF/SF
        source buffer, exact DTBC convolution history), keyed by face and
        port name.  Constant operators (material matrices, absorbing-
        boundary kernels) are re-derived on the resuming solver, never
        stored.
        """
        if self._fields is None:
            raise RuntimeError("nothing to checkpoint: run setup()/run() first")
        return {
            "n_completed": int(self._resume_step),
            "peak_energy": float(self._peak_energy),
            # DD-096 port-signal criterion peak; schema-additive.
            "peak_signal": float(self._peak_signal),
            "e": self._fields.e_flat.copy(),
            "h": self._fields.h_flat.copy(),
            "boundaries": {
                face: bc.state_dict()
                for face, bc in self.boundary_conditions.items()
                if hasattr(bc, "state_dict")
            },
            "ports": {op.name: op.state_dict() for op in self.ports if hasattr(op, "state_dict")},
            "monitors": {
                mon.name: mon.state_dict()
                for mon in self.monitors
                if hasattr(mon, "state_dict") and getattr(mon, "name", "")
            },
            # ADE pole currents (DD-084 on E, DD-089 on H); schema-additive
            # — absent on meshes without dispersive materials and in older
            # checkpoints.
            **(
                {"dispersion": self._dispersion.state_dict()}
                if self._dispersion is not None
                else {}
            ),
            **(
                {"dispersion_mu": self._dispersion_mu.state_dict()}
                if self._dispersion_mu is not None
                else {}
            ),
            # TD-SIBC branch states (WP-D4); schema-additive like the
            # ADE keys above.
            **({"sibc": self._sibc.state_dict()} if self._sibc is not None else {}),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a checkpoint captured by :meth:`state_dict`.

        The solver must be constructed identically (same mesh, dt, ports
        and boundary conditions) to the one that produced ``state`` —
        those fix the constant operators; this call only repopulates the
        evolving state.  Call before :meth:`run`, which then continues the
        marching loop from ``n_completed``.
        """
        if self._fields is None:
            self.setup()
        copy_into(self._fields.e_flat, state["e"])
        copy_into(self._fields.h_flat, state["h"])
        self._peak_energy = float(state["peak_energy"])
        # Pre-DD-096 checkpoints carry no signal peak; the criterion
        # then re-peaks from the resumed signals.
        self._peak_signal = float(state.get("peak_signal", 0.0))
        self._resume_step = int(state["n_completed"])
        for face, bsd in state["boundaries"].items():
            self.boundary_conditions[face].load_state_dict(bsd)
        port_by_name = {op.name: op for op in self.ports}
        for label, psd in state["ports"].items():
            port_by_name[label].load_state_dict(psd)
        # Field-monitor cursors (WP-S9): restore the target-time index so a
        # resumed run continues recording at the right point.  Older
        # checkpoints (pre-WP-S9) carry no "monitors" group.
        mon_by_name = {getattr(mon, "name", ""): mon for mon in self.monitors}
        for name, msd in state.get("monitors", {}).items():
            mon = mon_by_name.get(name)
            if mon is not None and hasattr(mon, "load_state_dict"):
                mon.load_state_dict(msd)
        # ADE pole currents (DD-084 on E, DD-089 on H) — keys absent on
        # non-dispersive meshes and in older checkpoints.
        if "dispersion" in state and self._dispersion is not None:
            self._dispersion.load_state_dict(state["dispersion"])
        if "dispersion_mu" in state and self._dispersion_mu is not None:
            self._dispersion_mu.load_state_dict(state["dispersion_mu"])
        # TD-SIBC branch states (WP-D4) — key absent on non-SIBC runs
        # and in older checkpoints.
        if "sibc" in state and self._sibc is not None:
            self._sibc.load_state_dict(state["sibc"])

    @staticmethod
    def _build_energy_trace(
        trace: list[tuple[int, float]],
        dt: float,
    ) -> np.ndarray:
        """Convert accumulated (step, energy) list to a structured array.

        Returns
        -------
        np.ndarray
            Structured array with fields ``'step'`` (int), ``'time'`` (float),
            and ``'energy'`` (float).
        """
        if not trace:
            return np.array(
                [],
                dtype=[("step", int), ("time", float), ("energy", float)],
            )
        steps = np.array([t[0] for t in trace], dtype=int)
        energies = np.array([t[1] for t in trace], dtype=float)
        out = np.empty(
            len(trace),
            dtype=[
                ("step", int),
                ("time", float),
                ("energy", float),
            ],
        )
        out["step"] = steps
        out["time"] = steps * dt
        out["energy"] = energies
        return out
