"""AnalysisTD — the general time-domain analysis.

One ``run()`` marches the FIT leapfrog once with any number of
excitations applied *simultaneously* — port channels and model
sources, each with its own waveform, amplitude and delay — and
returns a :class:`TDResult`: the recorded port signals, the sampled
excitation signals, the energy trace and the monitors.  No
S-parameters: those are the business of
:class:`~magnelio.AnalysisScatteringTD`, which derives from this class
and drives one channel per run.

The class also holds the machinery every time-domain problem class
shares — port-operator construction, the excitation binding, the
run-length estimate, the stop-criterion resolution, the project-store
streaming and the resume — so a derived class adds only the question
it answers.
"""

# Design: DD-224 (Analysis<Problem><Formulation>, the excitation triad,
# the shared transient engine); DD-063/DD-064 (modal port pipeline),
# DD-070 (project store, resume), DD-103 (closure on the mesh),
# DD-114/DD-122 (stop criteria, runtime cap), DD-155 (symmetry scale).

from __future__ import annotations

import dataclasses
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence, Union

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio._progress import Reporter, format_clock, format_seconds
from magnelio.analysis._base import _AnalysisBase
from magnelio.analysis._recipe import build_recipe, excitation_to_dict, recipe_kwargs
from magnelio.analysis.excitation import Excitation
from magnelio.analysis.result_interface import RunSettings
from magnelio.boundaries.boundary_conditions import (
    BoundaryConditions,
    bc_type_entries,
    cpml_thickness_of,
    materialize_boundary,
    symmetry_entries,
)
from magnelio.boundaries.pec import PECBoundary
from magnelio.constants import C0
from magnelio.ports._lumped import PortSpecLumped, build_lumped_element, build_lumped_port
from magnelio.ports._modal.factory import (
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.ports._modal.mode_report import PortReport
from magnelio.ports._modal.operator import _DTBC_PAIR_MARGINAL_TOL
from magnelio.ports.declarative import (
    PortAnalytical,
    PortLumped,
    PortWaveguide,
    resolve_declarative_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.post.modal_sparameters import destaggered_power_waves
from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import (
    Waveform,
    WaveformGaussian,
    WaveformGaussianModulated,
)
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import spectral_dt
from magnelio.sources.base import Source

if TYPE_CHECKING:
    from magnelio.io.project import Project

PortSpec = Union[
    PortSpecCoax,
    PortSpecRectWG,
    PortSpecNumerical,
    PortSpecMultiConductor,
    PortSpecLumped,
    PortWaveguide,
    PortAnalytical,
]

ExcitationSpec = Union[Excitation, str, tuple[str, int]]

# Auto runtime cap for unbounded runs (DD-122), in units of the
# auto-sized step estimate (itself ~25 diagonal transits): ~10³
# transits total.  In ring-down terms the cap accommodates a loaded Q
# of about 900·(structure size/wavelength) before a 60-dB decay is cut
# short — chosen to cover realistic narrow-band filters while still
# bounding a criterion-defeating run to minutes, not hours.
_RUNTIME_CAP_ESTIMATES = 40


# ═════════════════════════════════════════════════════════════════════
# Helpers shared with the scattering analysis
# ═════════════════════════════════════════════════════════════════════


def _excitation_key(exc: Excitation) -> tuple[str, int]:
    """``(name, mode)`` — the identity of an excitation in a run."""
    return (exc.source, int(exc.mode))


def _drive_function(exc: Excitation):
    """``A · w(t − delay)`` as a callable; the bare waveform when A = 1, delay = 0.

    The bare waveform keeps the default scattering drive bit-identical
    to the one the analysis used before excitations carried weights.
    """
    waveform = exc.waveform
    amplitude = float(exc.amplitude)
    delay = float(exc.effective_delay())
    if amplitude == 1.0 and delay == 0.0:
        return waveform

    def _drive(t, _w=waveform, _a=amplitude, _d=delay):
        return _a * _w(t - _d)

    return _drive


def _impulse_drive(amplitude: float):
    """The "signal" of an initial field: its amplitude at t = 0, nothing after."""
    amplitude = float(amplitude)

    def _drive(t, _a=amplitude):
        return _a if float(t) == 0.0 else 0.0

    return _drive


def _excitation_scale(op) -> float:
    """Injection amplitude scale for full-model power semantics (DD-155).

    ``1/√2`` per symmetry plane cutting the port window (modal) or the
    lumped edge chain, so a declared excitation of 1 √W injects one
    full-model watt (half of it into the meshed half-space).  Ports
    without a report and ports away from every symmetry plane
    return 1.0.
    """
    report = getattr(op, "port_report", None)
    if report is None:
        return 1.0
    return 1.0 / report.power_wave_full_scale


def _sampled_signal(fn, n_steps: int, dt: float, label: str = "excitation") -> Signal1D:
    """A drive function sampled on the run's own step axis."""
    t_axis = np.arange(n_steps) * dt
    return Signal1D(
        t=t_axis,
        values=np.array([fn(float(t)) for t in t_axis], dtype=float),
        dt=dt,
        label=label,
    )


def _renormalize_freq_monitors(monitors, reference_signal) -> None:
    """Give a finished run's excitation to its frequency monitors.

    Their DFT bins are the field folded with that waveform's spectrum;
    dividing it out is what turns them into fields per 1 W CW.
    """
    from magnelio.monitors.field_frequency import renormalize_all  # noqa: PLC0415

    renormalize_all(monitors, reference_signal)


def _port_stage(name: str, index: int, total: int) -> str:
    """Phase label for one port build; the count only helps above one."""
    return f"port {name!r}" if total == 1 else f"port {name!r} ({index}/{total})"


@dataclass(frozen=True)
class _LumpedModeStub:
    """Mode-shaped stub for ``PortOperatorLumped`` in compute_s_parameters.

    ``compute_s_parameters`` only ever calls ``mode.z_modal(omega)`` on
    the per-port Mode list to evaluate the reference impedance for
    power-wave decomposition.  A lumped port has a frequency-
    independent Thévenin impedance, so this stub returns ``Z0`` for any
    omega.  Constructed by the analyses internally; not part of the
    public API.
    """

    z0: float

    def z_modal(self, omega: float) -> complex:
        del omega
        return complex(self.z0)

    @property
    def i_cotemporal(self) -> bool:
        """Lumped I is sampled co-temporally with V (DD-075, F3).

        ``PortOperatorLumped.project_I`` returns the Thévenin current
        cached in ``update_e`` at ``t^{n+1}`` (the ``h`` argument is
        ignored), so V and I share the integer time level.  Modal ports
        sample ``I ∼ h`` at ``t^{n+1/2}`` instead; ``compute_s_parameters``
        reads this flag to skip the Yee half-step ``exp(+jω·dt/2)``
        correction on lumped channels only.
        """
        return True


def _cutoffs_from_port_modes(port_modes: dict | None) -> dict | None:
    """``{(port, mode): f_cutoff [Hz]}`` from a per-port Mode list.

    Modes without a cut-off (lumped stubs) are left out; the export
    warning treats a missing entry as "unknown", not as "DC".
    """
    if not port_modes:
        return None
    out = {}
    for port, modes in port_modes.items():
        for index, mode in enumerate(modes):
            omega_c = getattr(mode, "omega_c", None)
            if omega_c is not None:
                out[(port, index)] = float(omega_c) / (2.0 * math.pi)
    return out or None


def _power_wave_signal(
    channels: dict,
    port_modes: dict | None,
    port_normal_dx: dict | None,
    port_line_params: dict | None,
    port: str,
    mode: int,
    f_ref: float,
    sign: float,
    destagger: bool,
) -> Signal1D:
    """The incident (``sign > 0``) or outgoing power wave of one channel.

    Shared by :class:`TDResult` and the scattering result: the modal
    reference impedance is evaluated at ``f_ref``, the recorded V/I are
    split into ``(V ± Z·I)/(2√Z)`` — de-staggered with the port's
    certified discrete line parameters by default.
    """
    if port_modes is None:
        raise ValueError(
            "this result carries no port_modes; time-domain power waves are unavailable",
        )
    chan = (port, mode)
    if chan not in channels:
        raise KeyError(
            f"channel {chan!r} not recorded; available: {sorted(channels.keys())}",
        )
    if port not in port_modes:
        raise KeyError(
            f"port {port!r} not in port_modes (available: {sorted(port_modes.keys())})",
        )
    modes = port_modes[port]
    if not 0 <= mode < len(modes):
        raise ValueError(
            f"mode index {mode} out of range for port {port!r} with {len(modes)} mode(s)",
        )
    Z = complex(modes[mode].z_modal(2.0 * math.pi * float(f_ref)))
    if abs(Z.imag) > 1e-9 * abs(Z):
        raise ValueError(
            f"z_modal({f_ref:.4g} Hz) = {Z:.4g} is not real "
            f"(mode evanescent at f_ref?); pass an in-band f_ref=",
        )
    sqrt_z = math.sqrt(Z.real)

    V_sig, I_sig = channels[chan]

    if destagger:
        line_params = port_line_params.get(chan) if port_line_params is not None else None
        normal_dx = port_normal_dx.get(port) if port_normal_dx is not None else None
        a_sig, b_sig = destaggered_power_waves(
            V_sig,
            I_sig,
            modes[mode],
            z_ref=Z.real,
            normal_dx=normal_dx,
            line_params=line_params,
        )
        sig = a_sig if sign > 0 else b_sig
        name = "a" if sign > 0 else "b"
        return Signal1D(
            t=sig.t,
            values=sig.values,
            dt=sig.dt,
            label=f"{name}({port},{mode})",
        )

    V = V_sig.values
    I = I_sig.values
    # Temporal Yee alignment: stored I[n] is sampled dt/2 before
    # V[n]; the midpoint of consecutive samples lands on V's
    # instants.  The final sample keeps its own value — the run has
    # decayed there (energy_stop_db), so the O(dt) hold is inert.
    I_aligned = np.empty_like(I)
    I_aligned[:-1] = 0.5 * (I[:-1] + I[1:])
    I_aligned[-1] = I[-1]

    name = "a" if sign > 0 else "b"
    return Signal1D(
        t=V_sig.t,
        values=0.5 * (V / sqrt_z + sign * sqrt_z * I_aligned),
        dt=V_sig.dt,
        label=f"{name}({port},{mode})",
    )


# ═════════════════════════════════════════════════════════════════════
# TDResult
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TDResult:
    """Result of one :meth:`AnalysisTD.run` call — one leapfrog march.

    Attributes
    ----------
    excitations : tuple of Excitation
        The excitations that drove the run, with the waveform each one
        resolved to (the per-mode default where none was given).
    dt : float
        Solver time step [s].
    n_steps : int
        Number of leapfrog steps executed; every time series below has
        this many samples.
    signals : dict[(str, int), (Signal1D, Signal1D)]
        Recorded modal voltage and current ``(V, I)`` of every port
        channel, on the time axis ``t``.  ``V`` is the *total* modal
        voltage; use :meth:`a` / :meth:`b` for the incident/outgoing
        split.
    excitation_signals : dict[(str, int), Signal1D]
        The drive of every excitation sampled on the same axis —
        amplitude and delay included — keyed like ``excitations``.
    energy_trace : np.ndarray or None
        Stored electromagnetic energy at the solver's check cadence, a
        structured array with ``step``, ``time`` and ``energy`` fields.
    monitors : dict[str, Monitor]
        The run's monitors by name, holding their recorded data.
        Frequency-domain monitors keep the *raw* transient bins: with
        several waveforms in one run there is no single reference
        spectrum to divide out, so :meth:`renormalize` is your call.
    port_modes, port_normal_dx, port_line_params : dict or None
        The port records behind :meth:`a` / :meth:`b`.
    settings : RunSettings or None
        The settings this run was produced with, including why the
        marching stopped.
    name : str or None
        The run's name in a project store; ``None`` for an in-RAM run.
    started, finished : datetime or None
        Wall-clock stamps (UTC) of the march's start and end.
    elapsed : float or None
        Wall time of the march [s]; for a run read back from a project
        store the sum over every march of the run, resumes included.
    """

    excitations: tuple
    dt: float
    n_steps: int
    signals: dict
    excitation_signals: dict
    energy_trace: np.ndarray | None = None
    monitors: dict = field(default_factory=dict)
    port_modes: dict | None = None
    port_normal_dx: dict | None = None
    port_line_params: dict | None = None
    settings: RunSettings | None = None
    name: str | None = None
    started: object | None = None
    finished: object | None = None
    elapsed: float | None = None

    @property
    def t(self) -> np.ndarray:
        """Time axis of every recorded series [s]."""
        return np.arange(self.n_steps) * self.dt

    @property
    def stop_reason(self) -> str | None:
        """Why the marching ended (``"energy"``, ``"port_signal"``, ``"steps"``, …)."""
        return None if self.settings is None else self.settings.stop_reason

    def _key(self, name: str, mode: int) -> tuple[str, int]:
        key = (name, int(mode))
        if key not in self.excitation_signals:
            raise KeyError(
                f"no excitation {key!r} in this run; available: {sorted(self.excitation_signals)}",
            )
        return key

    def excitation_signal(self, name: str, mode: int = 0) -> Signal1D:
        """The sampled drive of the excitation naming ``name`` (and ``mode`` on a port)."""
        return self.excitation_signals[self._key(name, mode)]

    def signal(self, port: str, mode: int = 0, kind: str = "V") -> Signal1D:
        """A recorded port signal: modal voltage (``"V"``) or current (``"I"``)."""
        if kind not in ("V", "I"):
            raise ValueError(f"kind must be 'V' or 'I'; got {kind!r}")
        chan = (port, int(mode))
        if chan not in self.signals:
            raise KeyError(
                f"channel {chan!r} not recorded; available: {sorted(self.signals)}",
            )
        return self.signals[chan][0 if kind == "V" else 1]

    def _default_f_ref(self) -> float:
        waveforms = [e.waveform for e in self.excitations if e.waveform is not None]
        if not waveforms:
            raise ValueError("pass f_ref=: the run carries no waveform to take a band from")
        f_lo = min(float(w.f_min) for w in waveforms)
        f_hi = max(float(w.f_max) for w in waveforms)
        return 0.5 * (f_lo + f_hi)

    def a(self, port: str, mode: int = 0, *, f_ref: float | None = None, destagger: bool = True):
        """Incident power wave ``a(t)`` [√W] of one port channel.

        Parameters
        ----------
        port, mode
            The channel.
        f_ref : float, optional
            Frequency [Hz] at which the modal reference impedance is
            evaluated; default the centre of the excitations' band.
        destagger : bool, default True
            Use the port's certified discrete line parameters for the
            V/I half-cell alignment (the S-parameter convention).
        """
        return _power_wave_signal(
            self.signals,
            self.port_modes,
            self.port_normal_dx,
            self.port_line_params,
            port,
            int(mode),
            self._default_f_ref() if f_ref is None else float(f_ref),
            +1.0,
            destagger,
        )

    def b(self, port: str, mode: int = 0, *, f_ref: float | None = None, destagger: bool = True):
        """Outgoing power wave ``b(t)`` [√W] of one port channel (see :meth:`a`)."""
        return _power_wave_signal(
            self.signals,
            self.port_modes,
            self.port_normal_dx,
            self.port_line_params,
            port,
            int(mode),
            self._default_f_ref() if f_ref is None else float(f_ref),
            -1.0,
            destagger,
        )

    def renormalize(self, name: str, mode: int = 0) -> None:
        """Divide the frequency-domain monitors by one excitation's spectrum.

        Turns the raw transient bins of every ``MonitorFieldFrequency``
        and ``MonitorFarFieldFrequency`` into the response per unit of
        the named excitation — fields per 1 √W incident on a port, per
        1 V/m of an incident plane wave.  Meaningful when that
        excitation is the only one, or the only one with energy at the
        monitor frequencies.
        """
        _renormalize_freq_monitors(self.monitors.values(), self.excitation_signal(name, mode))

    def plot_signals(self, ax=None, *, kind: str = "V", **kwargs):
        """Plot every recorded port signal of one kind over time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes; a new figure when omitted.
        kind : {"V", "I"}, default "V"
            Modal voltage or current.
        **kwargs
            Forwarded to ``ax.plot``.

        Returns
        -------
        tuple
            The matplotlib ``(figure, axes)`` pair.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        for (port, mode), _ in sorted(self.signals.items()):
            sig = self.signal(port, mode, kind)
            ax.plot(sig.t * 1e9, sig.values, label=f"{port} mode {mode}", **kwargs)
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("modal voltage [V]" if kind == "V" else "modal current [A]")
        if self.signals:
            ax.legend()
        return fig, ax

    # ── how a result introduces itself (DD-254) ───────────────────────

    def _summary_rows(self) -> list[tuple[str, object]]:
        from magnelio._repr import fmt_db  # noqa: PLC0415
        from magnelio.post._energy import db_below_peak  # noqa: PLC0415

        keys = [f"{n}:{m}" if m else n for n, m in self.excitation_signals]
        duration = self.n_steps * self.dt * 1e9
        return [
            ("excitations", ", ".join(keys) or "—"),
            ("channels", ", ".join(f"{p}:{m}" for p, m in sorted(self.signals)) or "—"),
            ("steps", f"{self.n_steps} ({duration:.3g} ns)"),
            ("dt", self.dt),
            ("stop reason", self.stop_reason),
            ("energy", f"{fmt_db(db_below_peak(self.energy_trace))} below peak"),
            ("elapsed", format_seconds(self.elapsed)),
            ("started", self.started),
            ("monitors", ", ".join(sorted(self.monitors)) or "—"),
        ]

    def _title(self) -> str:
        return f"TDResult {self.name!r}" if self.name else "TDResult"

    def __repr__(self) -> str:
        from magnelio._repr import kv_block  # noqa: PLC0415

        return kv_block(self._title(), self._summary_rows())

    def _repr_html_(self) -> str:
        from magnelio._repr import html_kv  # noqa: PLC0415

        return html_kv(self._title(), self._summary_rows())


# ═════════════════════════════════════════════════════════════════════
# The prepared run (operators bound, drives armed)
# ═════════════════════════════════════════════════════════════════════


@dataclass
class _PreparedRun:
    """Everything a march needs, built once per ``run()`` (or resume)."""

    excitations: tuple  # resolved Excitation objects (waveform filled in)
    operators: list
    element_ops: list
    sources: list  # the excited model sources, waveform bound
    recorder: PortSignalRecorder | None
    drives: dict  # (name, mode) -> unscaled full-model drive function
    n_steps_estimate: int
    port_modes: dict
    port_normal_dx: dict
    port_line_params: dict
    # DD-244: dispersion records of the quasi-TEM modal ports (their
    # feed chains, for the exact de-embedding) and the half-window →
    # full-model factor of every port's line impedance.
    port_dispersion: dict = field(default_factory=dict)
    port_reference_scale: dict = field(default_factory=dict)

    @property
    def keys(self) -> list:
        return list(self.drives)

    @property
    def reference_fn(self):
        """The drive of the first excitation — the store's ``reference`` stream."""
        return next(iter(self.drives.values()))

    @property
    def drive_items(self) -> list:
        return list(self.drives.items())


# ═════════════════════════════════════════════════════════════════════
# AnalysisTD
# ═════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisTD(_AnalysisBase):
    """General time-domain analysis: simultaneous excitations, one march.

    Parameters
    ----------
    mesh : Mesh
        The mesh; carries the boundary closure and the ports, elements
        and sources declared on the model.
    f_max : float, optional
        Upper band edge of the analysis [Hz].  ``None`` (default) uses
        the design frequency the mesh was generated for (``mesh.f_max``);
        a mesh without one (``Mesh.from_grid``) requires an explicit
        value.  Parameterises the port-mode calculation and the default
        waveforms (a Gaussian over ``[0, f_max]`` for sources and
        TEM/lumped ports, a modulated Gaussian above a mode's cut-off);
        a waveform reaching above it warns.
    ports : list of port spec, optional
        ``None`` (default) uses the declarative ports the mesh carries;
        passing ``ports=`` overrides them completely.  May be empty when
        the run is driven by sources alone.
    elements : list of LumpedElement, optional
        Passive lumped circuit elements; ``None`` uses the mesh's.
    sources : list of Source, optional
        Model sources an excitation may name; ``None`` uses the mesh's
        (``model.add_source``).  Only the sources an excitation names
        inject anything.
    monitors : iterable, default ()
        Field monitors recorded during the march.
    port_model : {"modal"}, default "modal"
        The port pipeline.  The band-subspace pipeline decomposes
        S-parameters and belongs to the scattering analysis.
    port_source : {"frozen", "dispersive"}, default "frozen"
        What the port imprints.  ``"frozen"`` drives one quasi-static
        mode profile with one propagation delay.  ``"dispersive"``
        drives a low-rank family solved along the band, so the launched
        field is the mode the grid actually carries at each frequency —
        worth 15-20 dB of reflection on an inhomogeneous line such as a
        microstrip, and nothing at all on a hollow guide, whose profile
        does not move with frequency.  It costs one waveform and one
        plane write per rank term per step, about 1 % of the march.
    wall_model, wall_sigma, wall_mu, wall_roughness
        Conductor-loss model of the run, as on
        :class:`~magnelio.AnalysisScatteringTD`.
    verbose, project, geometry, params, backend, precision, method, solver
        The arguments every analysis takes; ``project=`` streams the
        run into a project store and returns its reader.

    Examples
    --------
    >>> result = AnalysisTD(mesh=mesh).run(  # doctest: +SKIP
    ...     excitations=[
    ...         Excitation("port1", waveform=signals.WaveformGaussianModulated(2e9, 8e9)),
    ...         Excitation("pw", waveform=signals.WaveformSine(f=5e9), amplitude=100.0),
    ...     ],
    ...     t_end=20e-9,
    ... )
    >>> result.signal("port1").values  # doctest: +SKIP
    """

    f_max: float | None = None
    ports: Sequence[PortSpec] | None = None
    elements: Sequence | None = None
    sources: Sequence | None = None
    monitors: tuple = field(default_factory=tuple)
    port_model: str = "modal"
    port_source: str = "frozen"
    wall_model: str = "perturbative"
    wall_sigma: float | None = None
    wall_mu: float = 1.0
    wall_roughness: object = None

    _SUPPORTED_SPEC_TYPES = (
        PortSpecCoax,
        PortSpecRectWG,
        PortSpecNumerical,
        PortSpecMultiConductor,
        PortSpecLumped,
        PortWaveguide,
        PortAnalytical,
        PortLumped,
    )
    _PORT_MODELS = ("modal",)
    _PORT_SOURCES = ("frozen", "dispersive")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.solver is not None:
            raise ValueError(
                f"solver={self.solver!r}: a time-domain march has no algebraic "
                f"solver to choose; solver= selects the eigenvalue solver of "
                f"AnalysisEigenmode",
            )
        self._mur_notice_printed = False
        # DD-186: the mesh records the f_max it was generated for; the
        # analysis band defaults to it.
        mesh_f_max = getattr(self.mesh, "f_max", None)
        if self.f_max is None:
            if mesh_f_max is None:
                raise ValueError(
                    "f_max is required: this mesh carries no design "
                    "frequency (it was not built by Mesh.from_geometry), "
                    "so the analysis band cannot be inferred. Pass "
                    "f_max= explicitly."
                )
            self.f_max = float(mesh_f_max)
        if self.f_max <= 0.0:
            raise ValueError(f"f_max must be positive; got {self.f_max}")
        if mesh_f_max is not None and self.f_max > float(mesh_f_max):
            warnings.warn(
                f"analysis f_max = {self.f_max:.4g} Hz exceeds the design "
                f"frequency this mesh was generated for "
                f"({float(mesh_f_max):.4g} Hz): the grid undersamples the "
                f"upper band. Re-mesh with the analysis f_max for a "
                f"resolved result.",
                UserWarning,
                stacklevel=3,
            )
        if self.port_model not in self._PORT_MODELS:
            raise ValueError(
                f"port_model must be one of {self._PORT_MODELS}; got {self.port_model!r}",
            )
        if self.port_source not in self._PORT_SOURCES:
            raise ValueError(
                f"port_source must be one of {self._PORT_SOURCES}; got {self.port_source!r}",
            )
        if self.wall_model not in ("perturbative", "sibc"):
            raise ValueError(
                f"wall_model must be 'perturbative' or 'sibc'; got {self.wall_model!r}",
            )
        self._sibc_spec_cache = None
        if self.wall_model == "sibc":
            has_lossy_metal = any(
                getattr(mat, "is_lossy_metal", False) for mat in self.mesh.material_library.values()
            )
            # DD-099: PECBoundary declarations carrying their own wall
            # material are a conductor source too.
            if not has_lossy_metal and self.wall_sigma is None and not self._bc_wall_overrides():
                raise ValueError(
                    "wall_model='sibc' needs a conductor to model: no "
                    "lossy-metal material in the mesh and no wall_sigma= "
                    "override for plain-PEC walls.  Give conductor walls "
                    "a Material.lossy_metal, or pass wall_sigma= (and "
                    "wall_mu=/wall_roughness=) for PEC walls.",
                )
        self._ports_from_mesh = self.ports is None
        if self.ports is None:
            # DD-109: ports declared on the GeometryModel travel with
            # the mesh; the analysis picks them up when no ports= of
            # its own is given.
            self.ports = list(self.mesh.ports)
        else:
            self.ports = list(self.ports)
        labels = [self._spec_label(s) for s in self.ports]
        if len(set(labels)) != len(labels):
            raise ValueError(f"port names must be unique; got {labels}")
        for s in self.ports:
            if not isinstance(s, self._SUPPORTED_SPEC_TYPES):
                raise TypeError(
                    f"{type(self).__name__} does not support spec type "
                    f"{type(s).__name__}; supported: "
                    f"{[t.__name__ for t in self._SUPPORTED_SPEC_TYPES]}",
                )
        # Resolve declarative ports into concrete specs.
        # PortWaveguide's conductor detection reads mesh.pec_mask_edges
        # (WP4.1), which already carries the declared PEC walls — both
        # mesh factories consolidate the closure they were given
        # (DD-103), so there is nothing left to fold in here.
        self.ports = [
            resolve_declarative_port(p, self.mesh)
            if isinstance(p, (PortWaveguide, PortAnalytical, PortLumped))
            else p
            for p in self.ports
        ]
        # Passive lumped elements (DD-123): same travel-with-the-mesh
        # convention as ports, one shared label namespace (the solver
        # keys per-operator checkpoint state by label).
        if self.elements is None:
            self.elements = list(getattr(self.mesh, "elements", ()) or ())
        else:
            self.elements = list(self.elements)
        from magnelio.circuit import LumpedElement  # noqa: PLC0415

        for e in self.elements:
            if not isinstance(e, LumpedElement):
                raise TypeError(
                    f"elements= takes magnelio.circuit.LumpedElement "
                    f"instances; got {type(e).__name__}",
                )
        # Model sources (DD-224): declared on the model, carried by the
        # mesh, named by excitations.
        if self.sources is None:
            self.sources = list(getattr(self.mesh, "sources", ()) or ())
        else:
            self.sources = list(self.sources)
        for src in self.sources:
            if not isinstance(src, Source):
                raise TypeError(
                    f"sources= takes magnelio.sources.Source instances; got {type(src).__name__}",
                )
        all_labels = labels + [e.name for e in self.elements] + [s.name for s in self.sources]
        if len(set(all_labels)) != len(all_labels):
            raise ValueError(
                f"port, element and source names must be unique together; got {all_labels}",
            )
        self._check_excitable()

    def _check_excitable(self) -> None:
        """Raise unless the analysis has something an excitation may name."""
        if not self.ports and not self.sources:
            raise ValueError(
                "nothing to excite: declare ports (GeometryModel.add_port) or "
                "sources (GeometryModel.add_source) before meshing, or pass "
                "ports= / sources= to the analysis",
            )

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def boundary_conditions(self):
        """The mesh's boundary closure (declared on the model).

        Read-only view: the closure is fixed when the mesh is built,
        because its consequences (CPML grid extension, PMC grid-line
        placement, PEC wall mask) are baked into the grid and the mask
        by then.  Re-declaring it here could only contradict them.
        """
        return self.mesh.boundary_conditions

    @property
    def cpml_thickness_cells(self) -> int:
        """CPML depth [cells] of the mesh's closure."""
        return cpml_thickness_of(self.mesh.boundary_conditions)

    def solve_ports(self) -> dict[str, PortReport]:
        """Solve every port's 2D mode problem without a TD run.

        Builds each port operator exactly as :meth:`run` would (same
        mesh, same material matrices, mode calculation at ``f_max``)
        and returns its mode solution as a
        :class:`~magnelio.ports._modal.mode_report.PortReport` —
        line impedance (numerical and, where available, analytical
        reference), cut-off frequencies, mode types, and per-mode
        ``modes[m].plot()`` transverse-profile plots.  Use this to
        inspect and validate the port modes *before* spending time on
        the 3D time-domain simulation.

        Lumped ports appear with an empty mode tuple and
        ``z_line_num = Z0``.

        Returns
        -------
        dict[str, PortReport]
            Keyed by port name, in ``ports`` order.
        """
        rep = Reporter("setup", self._verbose)
        rep.stage("material matrices")
        m_eps = build_M_eps(self.mesh)
        m_mu = build_M_mu(self.mesh)
        # dt only parameterises the operator's Mur coefficients, which
        # the report does not expose — the spectral value keeps the
        # construction identical to run() (the measured lambda_max is
        # cached on the mesh, so run() pays no second eigensolve).
        # It is a Lanczos iteration over the whole update operator and
        # dominates a standalone solve_ports (62 % of it on a 2 M-cell
        # model), which is why it gets a phase of its own.
        rep.stage("CFL eigenvalue")
        dt = spectral_dt(self.mesh, "normal", m_eps=m_eps, m_mu=m_mu)
        from magnelio.ports._modal.factory import (  # noqa: PLC0415
            build_port_dispersion_record,
        )

        curl_cache: dict = {}

        def _factory_for(op):
            def _build():
                if "c_3d" not in curl_cache:
                    from magnelio._operators.curl import build_curl_matrix  # noqa: PLC0415

                    curl_cache["c_3d"] = build_curl_matrix(self.mesh.grid)
                return build_port_dispersion_record(
                    op, self.mesh, m_eps, m_mu, self.f_max, c_3d=curl_cache["c_3d"]
                )

            return _build

        reports = {}
        n_ports = len(self.ports)
        for i, spec in enumerate(self.ports, start=1):
            rep.stage(_port_stage(spec.name, i, n_ports))
            op = self._build_operator(spec, m_eps, m_mu, dt, self.f_max)
            reports[spec.name] = PortReport.from_operator(
                op,
                mesh=self.mesh,
                dispersion_factory=_factory_for(op) if hasattr(op, "discrete_modes") else None,
            )
        rep.finish()
        return reports

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        excitations: Iterable[ExcitationSpec] | ExcitationSpec | None = None,
        *,
        name: str | None = None,
        t_end: float | None = None,
        accuracy: str = "normal",
        energy_stop_db: float | None = 70.0,
        total_time_steps: int | None = None,
        port_signal_stop_db: float | str | None = "auto",
        max_time_steps: int | str | None = "auto",
        checkpoint_interval: int | None = None,
    ) -> "TDResult | Project":
        """March once with every excitation applied simultaneously.

        Parameters
        ----------
        excitations : iterable of Excitation, str or (str, int)
            What drives the run: :class:`~magnelio.Excitation` objects,
            or the shorthands ``"port1"`` / ``("port1", 1)`` for a port
            channel at unit amplitude with the default waveform.  Every
            entry names a port declared on the model or a source, and
            each ``(name, mode)`` may appear once.
        name : str, optional
            Run name in the project store (``project=``); default
            ``run_<n>``.  A name already taken in the project — by any
            run, including a scattering channel run — is an error.
        t_end : float, optional
            Physical duration of the run [s]; exclusive with
            ``total_time_steps``.  Required when a waveform has no end
            (``WaveformSine``, ``WaveformStep`` without ``fall_time``):
            a continuous-wave drive never decays, so the energy and
            port-signal criteria are disabled for such a run.
        accuracy : {"draft", "normal", "high"}, default "normal"
            Courant safety factor.
        energy_stop_db : float, default 70.0
            Stop when the stored energy has decayed by this many dB
            below its peak (``None`` disables).
        total_time_steps : int, optional
            Exact leapfrog step count; default unbounded until a stop
            criterion fires, backstopped by ``max_time_steps``.
        port_signal_stop_db : float, None or "auto", default "auto"
            Stop when every modal port's ``|V|`` envelope has decayed by
            this many dB below its run peak.  ``"auto"`` resolves to
            60 dB when a modal port is present and to disabled otherwise.
        max_time_steps : int, None or "auto", default "auto"
            Runtime cap for unbounded runs (40× the auto step estimate).
        checkpoint_interval : int, optional
            Minimum steps between resume checkpoints on the project path.

        Returns
        -------
        TDResult or Project
            The in-RAM result; with ``project=`` the store's
            :class:`~magnelio.io.project.Project` reader instead — the
            same object :func:`~magnelio.open_project` returns, whose
            ``result(name)`` rebuilds the :class:`TDResult`.
        """
        if self.port_model != "modal":
            raise NotImplementedError(
                "AnalysisTD runs on the modal port pipeline; the band pipeline "
                "decomposes S-parameters and belongs to AnalysisScatteringTD",
            )
        exc_list = self._resolve_excitations(excitations)
        port_signal_stop_db = self._resolve_port_signal_stop(port_signal_stop_db)
        if isinstance(max_time_steps, str) and max_time_steps != "auto":
            raise ValueError(
                f"max_time_steps must be an int (steps), None or 'auto'; got {max_time_steps!r}",
            )
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError(f"name must be a non-empty string; got {name!r}")

        self._wire_wall_monitors()
        self._wire_far_field_ports()
        bc_objects = self._resolve_bc()

        # The whole call is timed, setup included — that is the number
        # a user waits for (DD-253).
        self._start_run_clock()
        # The setup ahead of the time-domain loop is not free: the
        # CFL eigenvalue is a Lanczos iteration over the whole update
        # operator, and each port is a 2D mode solve.  Name them, or
        # the run looks stalled before step 1 appears.
        self._setup_reporter = Reporter("setup", self._verbose)
        self._setup_reporter.stage("material matrices")
        m_eps = build_M_eps(self.mesh)
        m_mu = build_M_mu(self.mesh)
        self._setup_reporter.stage("CFL eigenvalue")
        dt = spectral_dt(self.mesh, accuracy, m_eps=m_eps, m_mu=m_mu)
        self._setup_reporter.finish()

        prepared = self._prepare_run(exc_list, m_eps, m_mu, dt)
        total_time_steps, energy_stop_db, port_signal_stop_db = self._duration_rules(
            prepared, dt, t_end, total_time_steps, energy_stop_db, port_signal_stop_db
        )
        stops = dict(
            total_time_steps=total_time_steps,
            energy_stop_db=energy_stop_db,
            port_signal_stop_db=port_signal_stop_db,
            max_time_steps=max_time_steps,
        )
        if self.project is None:
            monitors = list(self.monitors)
            solver = self._build_solver(prepared, bc_objects, dt, monitors=monitors, **stops)
            solver.run()
            result = self._collect_result(
                prepared, solver, dt, monitors, accuracy=accuracy, t_end=t_end, **stops
            )
            self._report_finished()
            return result
        return self._run_to_store(
            prepared,
            bc_objects,
            dt,
            name=name,
            checkpoint_interval=checkpoint_interval,
            t_end=t_end,
            **stops,
        )

    # ------------------------------------------------------------------
    # Excitations
    # ------------------------------------------------------------------

    def _resolve_excitations(self, excitations) -> list[Excitation]:
        """Coerce the shorthands, check names and uniqueness."""
        if excitations is None:
            raise TypeError(
                f"{type(self).__name__}.run() needs excitations=[...]: the "
                f"ports or sources to drive (an Excitation, a name, or a "
                f"(name, mode) pair each)",
            )
        if isinstance(excitations, (str, Excitation)) or (
            isinstance(excitations, tuple)
            and len(excitations) == 2
            and isinstance(excitations[0], str)
        ):
            excitations = [excitations]
        port_labels = {self._spec_label(s) for s in self.ports}
        source_by_name = {s.name: s for s in self.sources}
        out: list[Excitation] = []
        seen: set = set()
        for spec in excitations:
            exc = Excitation.coerce(spec)
            if exc.source in port_labels:
                pass  # the mode index is checked against the built operator
            elif exc.source in source_by_name:
                src = source_by_name[exc.source]
                if not getattr(src, "excitable", True):
                    raise ValueError(f"source {exc.source!r} cannot be excited")
                if exc.mode != 0:
                    raise ValueError(
                        f"Excitation({exc.source!r}, mode={exc.mode}): a source has no "
                        f"modes; leave mode at 0",
                    )
            else:
                raise ValueError(
                    f"excitation names {exc.source!r}, which is neither a port "
                    f"{sorted(port_labels)} nor a source {sorted(source_by_name)}",
                )
            key = _excitation_key(exc)
            if key in seen:
                raise ValueError(f"duplicate excitation of {key!r}")
            seen.add(key)
            if exc.waveform is not None and exc.waveform.f_max > self.f_max:
                warnings.warn(
                    f"waveform f_max = {exc.waveform.f_max:.4g} Hz of excitation "
                    f"{key!r} exceeds the analysis band f_max = {self.f_max:.4g} Hz: "
                    f"the pulse carries energy the grid does not resolve.",
                    UserWarning,
                    stacklevel=3,
                )
            out.append(exc)
        if not out:
            raise ValueError("excitations must not be empty")
        return out

    def _resolve_port_signal_stop(self, port_signal_stop_db):
        if isinstance(port_signal_stop_db, str):
            if port_signal_stop_db != "auto":
                raise ValueError(
                    f"port_signal_stop_db must be a float [dB], None or "
                    f"'auto'; got {port_signal_stop_db!r}",
                )
            # DD-114: on by default wherever a modal port can feed the
            # |V|-envelope criterion; a run without one has no modal
            # envelope to watch and keeps the energy criterion alone.
            return 60.0 if any(not isinstance(s, PortSpecLumped) for s in self.ports) else None
        return port_signal_stop_db

    @staticmethod
    def _duration_rules(
        prepared: _PreparedRun,
        dt: float,
        t_end: float | None,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        port_signal_stop_db: float | None,
    ) -> tuple:
        """``t_end`` → steps; a continuous-wave drive needs a length and no decay stop."""
        if t_end is not None:
            if total_time_steps is not None:
                raise ValueError(
                    "t_end= (seconds) and total_time_steps= (steps) both fix the run "
                    "length; pass one of them",
                )
            t_end = float(t_end)
            if not t_end > 0.0:
                raise ValueError(f"t_end must be a positive duration [s]; got {t_end!r}")
            # Round-off guard: 4e-9 / 1e-12 is 4000.0000000000005 in
            # binary, and a duration that is a whole number of steps
            # must not gain a step for it.
            total_time_steps = int(math.ceil(t_end / dt - 1e-9))
        cw = [
            _excitation_key(e)
            for e in prepared.excitations
            if e.waveform is not None and math.isinf(float(e.waveform.t_end))
        ]
        if cw:
            if total_time_steps is None:
                raise ValueError(
                    f"excitation(s) {cw} drive a continuous-wave waveform, which "
                    f"never decays: pass t_end= (seconds) or total_time_steps= to "
                    f"fix the run length",
                )
            # No decay to wait for — the stop criteria would never fire.
            energy_stop_db = None
            port_signal_stop_db = None
        return total_time_steps, energy_stop_db, port_signal_stop_db

    def _band_f_min(self) -> float:
        """Lower edge the default waveforms respect (0 — the scattering analysis has one)."""
        return 0.0

    def _resolve_waveform(
        self,
        waveform: Waveform | None,
        excited_op,
        mode_idx: int,
    ) -> Waveform:
        """Return the explicit override or derive a per-mode waveform.

        Auto rule: the effective lower band edge is ``max(f_cutoff,
        f_min)``, where ``f_cutoff`` is the excited mode's cut-off
        frequency (zero for TEM modes and lumped ports).  A zero edge
        yields a DC-inclusive ``WaveformGaussian``; a positive one a
        band-limited ``WaveformGaussianModulated`` over ``[edge,
        f_max]`` — for TE/TM modes this keeps the pulse spectrum above
        cut-off, where a DC-inclusive pulse would put ~half its energy
        below cut-off (total reflection, slow Mur-ABC ringing).
        """
        modes = self._modes_for_operator(excited_op)
        if not 0 <= mode_idx < len(modes):
            raise ValueError(
                f"excited mode index {mode_idx} out of range for port "
                f"{excited_op.name!r} with {len(modes)} mode(s)",
            )
        if waveform is not None:
            return waveform
        mode = modes[mode_idx]
        # _LumpedModeStub carries no omega_c — lumped ports are DC-capable.
        f_cutoff = getattr(mode, "omega_c", 0.0) / (2.0 * math.pi)
        f_min = self._band_f_min()
        eff_f_min = max(f_cutoff, f_min)
        if eff_f_min >= self.f_max:
            mode_label = getattr(mode, "name", f"mode {mode_idx}")
            raise ValueError(
                f"f_max = {self.f_max:.4g} Hz does not exceed the lower "
                f"band edge {eff_f_min:.4g} Hz of excited mode "
                f"{mode_label!r} on port {excited_op.name!r} "
                f"(cut-off {f_cutoff:.4g} Hz, f_min {f_min:.4g} Hz); "
                f"increase f_max or pass an explicit waveform=",
            )
        if eff_f_min <= 0.0:
            return WaveformGaussian(f_max=self.f_max)
        return WaveformGaussianModulated(f_min=eff_f_min, f_max=self.f_max)

    def _mur_fallback_notice(
        self,
        operators: Sequence[object],
        dt: float,
    ) -> str | None:
        """Explain the modal Mur-1st fallback, or return ``None``.

        A channel whose feed cross-section is not a uniform discrete
        chain cannot carry the exact transparent boundary and is
        terminated by modal Mur-1st instead.  That is a deliberate
        trade, not a defect, so the notice reads as a balance sheet —
        one line of trade, one line of alternative, the channels listed
        above them.  It prints on every quasi-TEM run at the default
        ``verbose``, so it stays short; the detail is the *Ports*
        chapter of the methods guide.  Subclasses that own the
        alternative price it (see ``AnalysisScatteringTD``).

        The three ways a channel arrives here are physically distinct
        and are named apart: a genuinely inhomogeneous cross-section
        (the user's model — a microstrip is one), a cross-section that
        was meant to be uniform and missed the gate by jitter (the
        port has already warned with the mesh-side cause), and a mode
        with no measured spread at all (analytical evaluator, or a
        feed-chain veto that decided first).
        """
        lines: list[str] = []
        for op in operators:
            kinds = list(getattr(op, "termination_kinds", None) or [])
            spreads = list(getattr(op, "chain_spreads", None) or [])
            modes = list(getattr(op, "discrete_modes", None) or [])
            for m, kind in enumerate(kinds):
                if kind != "mur":
                    continue
                spread = spreads[m] if m < len(spreads) else None
                name = modes[m].mode.name if m < len(modes) else f"mode {m}"
                if spread is None:
                    why = "no discrete chain parameters for this mode"
                elif spread > _DTBC_PAIR_MARGINAL_TOL:
                    why = f"inhomogeneous cross-section, chain spread {spread:.2e}"
                else:
                    why = (
                        f"chain spread {spread:.2e} above the uniform-chain "
                        f"gate (see the port's own warning)"
                    )
                lines.append(f"  {op.name} [{m}] {name} — {why}")
        if not lines:
            return None

        head = f"[{type(self).__name__}] modal Mur-1st termination on {len(lines)} channel(s):"
        body = (
            "  |S11| floor of order -30 dB on those channels, |S21| within "
            "0.01 dB; the run keeps its runtime and the power waves a()/b()."
        )
        return "\n".join([head, *lines, body, self._mur_fallback_alternative(dt)])

    def _mur_fallback_alternative(self, dt: float) -> str:
        """Name the reflection-free alternative and where it lives.

        ``AnalysisTD`` has no port-pipeline switch of its own — the
        band-subspace DTBC is an ``AnalysisScatteringTD`` option — so
        the base notice points at the class that owns it rather than
        at a keyword this one would reject.
        """
        return '  Reflection-free alternative: AnalysisScatteringTD(port_model="band").'

    def _prepare_run(
        self,
        excitations: Sequence[Excitation],
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
    ) -> _PreparedRun:
        """Build the operators, bind every excitation, size the run.

        Operators are built fresh per run — clean Mur/DTBC state, and
        the per-mode source-history buffers freshly sized.  The port
        metadata depends only on the operators (not on the marching),
        so the streaming sink can declare the run's HDF5 layout before
        step 0.  Elements (DD-123) join the solver's operator list but
        stay out of the recorder and the port metadata.
        """
        rep = getattr(self, "_setup_reporter", None)
        operators = []
        n_ports = len(self.ports)
        for i, spec in enumerate(self.ports, start=1):
            if rep is not None:
                rep.stage(_port_stage(spec.name, i, n_ports))
            operators.append(self._build_operator(spec, m_eps, m_mu, dt, self.f_max))
        if rep is not None:
            rep.finish()
        element_ops = [
            build_lumped_element(e, self.mesh, m_eps, m_mu, dt=dt) for e in self.elements
        ]

        if self._verbose and not self._mur_notice_printed:
            # DD-231: the balance sheet, once per analysis.
            notice = self._mur_fallback_notice(operators, dt)
            if notice:
                self._note("setup", notice)
            self._mur_notice_printed = True

        label_to_op = {op.name: op for op in operators}
        source_by_name = {s.name: s for s in self.sources}
        for op in operators:
            op.clear_excitation()
        for src in self.sources:
            src.clear_excitation()

        resolved: list[Excitation] = []
        drives: dict = {}
        sources: list = []
        for exc in excitations:
            if exc.source in label_to_op:
                op = label_to_op[exc.source]
                waveform = self._resolve_waveform(exc.waveform, op, exc.mode)
                exc = dataclasses.replace(exc, waveform=waveform)
                fn = _drive_function(exc)
                # DD-155: a port cut by symmetry planes injects ×1/√2 per
                # plane so the declared amplitude is a full-model √W —
                # the meshed half then carries exactly half the
                # full-model power and the fields sit at full-model
                # level.  The scale must act exactly once: the sampled
                # excitation signal keeps the *unscaled* drive (the
                # full-model reference the monitors renormalise
                # against); the recorder restores ×√2 per plane on the
                # read side.
                exc_scale = _excitation_scale(op)
                if exc_scale != 1.0:
                    op.set_excitation(exc.mode, lambda t, _fn=fn, _s=exc_scale: _s * _fn(t))
                else:
                    op.set_excitation(exc.mode, fn)
            else:
                src = source_by_name[exc.source]
                if not getattr(src, "has_waveform", True):
                    # An initial field exists at t = 0: the excitation
                    # is its amplitude alone, and the recorded
                    # "signal" is that amplitude at the first sample.
                    if exc.waveform is not None:
                        raise ValueError(
                            f"Excitation({exc.source!r}): source {exc.source!r} is an "
                            f"initial field and takes no waveform",
                        )
                    if exc.delay != 0.0 or exc.phase != 0.0:
                        raise ValueError(
                            f"Excitation({exc.source!r}): an initial field cannot be "
                            f"delayed or phased (got delay={exc.delay!r}, phase={exc.phase!r})",
                        )
                    fn = _impulse_drive(exc.amplitude)
                    src.set_excitation(None, amplitude=exc.amplitude)
                else:
                    waveform = exc.waveform
                    if waveform is None:
                        waveform = WaveformGaussian(f_max=self.f_max)
                    exc = dataclasses.replace(exc, waveform=waveform)
                    fn = _drive_function(exc)
                    src.set_excitation(
                        waveform, amplitude=exc.amplitude, delay=exc.effective_delay()
                    )
                sources.append(src)
            resolved.append(exc)
            drives[_excitation_key(exc)] = fn

        n_steps_estimate = self._estimate_steps(self.mesh.grid, self._pulse_duration(resolved), dt)
        port_dispersion = self._dispersion_records(operators, m_eps, m_mu)
        if self.port_source == "dispersive":
            self._bind_dispersive_sources(
                resolved, drives, label_to_op, port_dispersion, dt, n_steps_estimate
            )
        recorder = PortSignalRecorder(dt=dt, ports=operators) if operators else None
        port_modes = {op.name: self._modes_for_operator(op) for op in operators}
        # Spatial de-stagger of the I sampling plane (modal ports only;
        # lumped ports have no plane and keep the co-located formula).
        port_normal_dx = {
            op.name: op.plane.normal_dx
            for op in operators
            if getattr(op, "plane", None) is not None
        }
        # Discrete de-stagger (WP-R2): modes whose termination certified
        # the exact 1D chain carry (r, q) for the exact half-cell factor.
        port_line_params = {
            (op.name, m): params
            for op in operators
            for m, params in getattr(op, "dtbc_line_params", {}).items()
        }
        return _PreparedRun(
            excitations=tuple(resolved),
            operators=operators,
            element_ops=element_ops,
            sources=sources,
            recorder=recorder,
            drives=drives,
            n_steps_estimate=n_steps_estimate,
            port_modes=port_modes,
            port_normal_dx=port_normal_dx,
            port_line_params=port_line_params,
            port_dispersion=port_dispersion,
            port_reference_scale=self._reference_scales(operators),
        )

    def _bind_dispersive_sources(self, resolved, drives, label_to_op, records, dt, n_steps) -> None:
        """Synthesise and attach a rank-r source per excited modal mode.

        A channel that carries no propagating mode over the band, or a
        port with no dispersion record, keeps the frozen source and says
        so: the dispersive source is an accuracy option, not a
        precondition for the run.
        """
        from magnelio.ports._modal.dispersive_source import synthesise_dispersive_source

        f_hi = float(self.f_max) if self.f_max else None
        if not f_hi:
            warnings.warn(
                "port_source='dispersive' needs f_max to size its band; keeping the frozen source",
                UserWarning,
                stacklevel=2,
            )
            return
        band = (max(f_hi / 50.0, 1.0e6), 1.2 * f_hi)
        t_grid = np.arange(int(n_steps)) * dt
        for exc in resolved:
            op = label_to_op.get(exc.source)
            if op is None or getattr(op, "plane", None) is None:
                continue
            record = records.get(op.name)
            if record is None:
                continue
            fn = drives.get(_excitation_key(exc))
            if fn is None:
                continue
            try:
                terms = synthesise_dispersive_source(
                    record,
                    int(exc.mode),
                    np.array([float(fn(float(tt))) for tt in t_grid]),
                    dt,
                    band,
                    dual_projector=op.dual_projection_of,
                )
                op.set_excitation_dispersive(int(exc.mode), terms)
            except ValueError as err:
                warnings.warn(
                    f"port '{op.name}' mode {exc.mode} keeps the frozen source: {err}",
                    UserWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _reference_scales(operators) -> dict:
        """Half-window → full-model factor of each modal port's line impedance."""
        out = {}
        for op in operators:
            scale = getattr(getattr(op, "port_report", None), "z_line_full_scale", None)
            if scale is not None:
                out[op.name] = float(scale)
        return out

    def _dispersion_records(self, operators, m_eps, m_mu) -> dict:
        """Dispersion records of the quasi-TEM modal ports (DD-244).

        A quasi-TEM channel runs on the modal Mur absorber and carries
        no certified line parameters, so its de-embedding would fall
        back to the quasi-static continuum ``γ``; the record lets the
        result solve the feed's true discrete ``ζ(f)`` instead.  A port
        whose feed section is not a certified uniform chain is skipped
        — its de-embedding keeps the continuum fallback, and the port
        has already warned about the section.
        """
        from magnelio.ports._modal.factory import (  # noqa: PLC0415
            build_port_dispersion_record,
        )

        records = {}
        c_3d = None
        for op in operators:
            report = getattr(op, "port_report", None)
            if report is None or not getattr(report, "quasi_static", False):
                continue
            if not hasattr(op, "discrete_modes"):
                continue
            if c_3d is None:
                from magnelio._operators.curl import build_curl_matrix  # noqa: PLC0415

                c_3d = build_curl_matrix(self.mesh.grid)
            try:
                records[op.name] = build_port_dispersion_record(
                    op, self.mesh, m_eps, m_mu, self.f_max, c_3d=c_3d
                )
            except ValueError:
                continue
        return records

    # ------------------------------------------------------------------
    # Run length and stop criteria
    # ------------------------------------------------------------------

    @staticmethod
    def _pulse_duration(excitations: Sequence[Excitation]) -> float:
        """``max_i(delay_i + t_end_i)`` over the finite waveforms [s].

        A continuous-wave waveform (``t_end = inf``) contributes
        nothing: its run is bounded explicitly, and the estimate then
        only sets the check cadence and the checkpoint stride.
        """
        ends = [
            float(e.effective_delay()) + float(e.waveform.t_end)
            for e in excitations
            if e.waveform is not None and not math.isinf(float(e.waveform.t_end))
        ]
        return max(ends) if ends else 0.0

    @staticmethod
    def _estimate_steps(
        grid,
        t_pulse: float,
        dt: float,
        n_traversals: int = 25,
    ) -> int:
        """Auto-derive the run *scale* — a generous estimate of the
        needed length, not a tight one.

        Heuristic: ``t_pulse + n_traversals · t_diag``, where
        ``t_pulse`` is the time the last excitation has died out
        (``max_i(delay_i + t_end_i)``; ``8 / f_max`` for a Gaussian
        pulse) and ``t_diag = ‖bbox‖ / v_safe`` uses ``v_safe = 0.5·c₀``
        to keep a margin against dispersion (group velocity in hollow
        WG drops below c₀ near cutoff).

        Role.  This is **not** a default cap: the default
        run is unbounded and stops on ``energy_stop_db`` (see
        :meth:`_resolve_runtime`).  The estimate sets the energy-check
        cadence (``min(100, estimate/20)``) and the checkpoint interval,
        and serves as a *soft* fallback cap only when the energy criterion
        is disabled *and* no explicit ``total_time_steps`` is given.
        ``n_traversals = 25`` is intentionally generous — TEM lines reach
        70 dB after ~3-4 traversals; moderate-Q TE/TM modes converge
        within ~5 round-trips against their Mur-1 floor.  A high-Q
        resonator that never decays 70 dB simply keeps marching (the
        motivating resume case) unless an explicit cap is set.
        """
        if not t_pulse >= 0.0:
            raise ValueError(f"t_pulse must be non-negative; got {t_pulse}")
        Lx = float(grid.x[-1] - grid.x[0])
        Ly = float(grid.y[-1] - grid.y[0])
        Lz = float(grid.z[-1] - grid.z[0])
        L_diag = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        v_safe = 0.5 * C0
        t_diag = L_diag / v_safe if L_diag > 0.0 else 0.0
        t_total = t_pulse + n_traversals * t_diag
        return int(math.ceil(t_total / dt))

    def _resolve_runtime(
        self,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        estimate: int,
        port_signal_stop_db: float | None = None,
    ) -> tuple[int | None, int | None]:
        """Resolve ``(solver total_time_steps, energy_check_interval)``.

        The default is an **unbounded, energy-gated** march: the
        auto-sized ``estimate`` sets only the energy-check cadence, never a
        cap, so a converging run is bounded by ``energy_stop_db`` and never
        by a heuristic timeout.  The cadence matches the historical bounded
        default (``min(100, estimate/20)``), so the energy-stop step — and
        hence the S-parameters — are unchanged for well-absorbed runs; only
        a structure that never decays past ``energy_stop_db`` now keeps
        going (watch it live / ``resume()``, don't restart).  An explicit
        ``total_time_steps`` restores a hard cap (the solver then derives
        its own cadence from it); disabling the energy criterion *and*
        leaving the cap open falls back to the estimate as a soft cap.
        """
        if total_time_steps is not None:
            return total_time_steps, None
        if energy_stop_db is not None or port_signal_stop_db is not None:
            return None, max(1, min(100, estimate // 20))
        return estimate, None

    @staticmethod
    def _resolve_cap(
        max_time_steps: int | str | None,
        total_time_steps: int | None,
        estimate: int,
        start_step: int = 0,
    ) -> int | None:
        """Resolve the runtime cap (absolute step bound) for one run.

        ``None`` on bounded runs — an explicit ``total_time_steps`` is a
        user decision the cap must not override.  ``"auto"`` grants each
        launch (or resume segment) its own budget of
        ``_RUNTIME_CAP_ESTIMATES`` step estimates past ``start_step``;
        an explicit int is passed through as an absolute bound, and
        ``None`` disables the cap (and with it the stall watchdog).
        """
        if total_time_steps is not None:
            return None
        if max_time_steps == "auto":
            return start_step + _RUNTIME_CAP_ESTIMATES * int(estimate)
        return max_time_steps

    # ------------------------------------------------------------------
    # The march
    # ------------------------------------------------------------------

    def _build_solver(
        self,
        prepared: _PreparedRun,
        bc_objects: dict,
        dt: float,
        *,
        monitors: list,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        port_signal_stop_db: float | None,
        max_time_steps: int | str | None,
        sink=None,
        start_step: int = 0,
    ) -> FITTimeDomainSolver:
        """The solver of one march — identical on every path (RAM, store, resume)."""
        solver_steps, check_interval = self._resolve_runtime(
            total_time_steps,
            energy_stop_db,
            prepared.n_steps_estimate,
            port_signal_stop_db=port_signal_stop_db,
        )
        return FITTimeDomainSolver(
            mesh=self.mesh,
            boundary_conditions=bc_objects,
            ports=prepared.operators + prepared.element_ops,
            sources=list(prepared.sources),
            recorder=prepared.recorder,
            total_time_steps=solver_steps,
            energy_check_interval=check_interval,
            dt=dt,
            energy_stop_db=energy_stop_db,
            port_signal_stop_db=port_signal_stop_db,
            port_signal_min_steps=prepared.n_steps_estimate,
            max_time_steps=self._resolve_cap(
                max_time_steps, total_time_steps, prepared.n_steps_estimate, start_step
            ),
            verbose=self._verbose,
            monitors=monitors,
            sink=sink,
            backend=self.backend,
            precision=self.precision,
            sibc=self._sibc_spec(),
        )

    def _run_settings(self, **kwargs) -> RunSettings:
        """Assemble the result-contract settings for this run."""
        return RunSettings(
            f_max=self.f_max,
            precision=self.precision,
            backend=self.backend,
            **kwargs,
        )

    def _collect_result(
        self,
        prepared: _PreparedRun,
        solver: FITTimeDomainSolver,
        dt: float,
        monitors: list,
        *,
        accuracy: str,
        t_end: float | None,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        port_signal_stop_db: float | None,
        max_time_steps=None,
        name: str | None = None,
    ) -> TDResult:
        """Turn a finished in-RAM march into a :class:`TDResult`."""
        del max_time_steps
        n_actual = solver._actual_steps or prepared.n_steps_estimate
        signals = (
            prepared.recorder.finalize(n_steps_actual=n_actual)
            if prepared.recorder is not None
            else {}
        )
        excitation_signals = self._sample_drives(prepared, n_actual, dt)
        return TDResult(
            excitations=prepared.excitations,
            dt=dt,
            n_steps=n_actual,
            signals=signals,
            excitation_signals=excitation_signals,
            energy_trace=getattr(solver, "_energy_trace", None),
            started=solver._started,
            finished=solver._finished,
            elapsed=solver._elapsed,
            monitors={m.name: m for m in monitors if getattr(m, "name", None)},
            port_modes=prepared.port_modes,
            port_normal_dx=prepared.port_normal_dx,
            port_line_params=prepared.port_line_params,
            settings=self._run_settings(
                dt=dt,
                n_actual_steps=n_actual,
                accuracy=accuracy,
                energy_stop_db=energy_stop_db,
                port_signal_stop_db=port_signal_stop_db,
                stop_reason=solver._stop_reason,
                final_port_signal_db=solver._final_signal_db,
                port_model_used="modal",
                t_end=t_end,
                excitations=tuple(prepared.keys),
            ),
            name=name,
        )

    @staticmethod
    def _sample_drives(prepared: _PreparedRun, n_steps: int, dt: float) -> dict:
        single = len(prepared.drives) == 1
        return {
            key: _sampled_signal(
                fn, n_steps, dt, label="excitation" if single else f"excitation{key}"
            )
            for key, fn in prepared.drives.items()
        }

    # ------------------------------------------------------------------
    # Project store
    # ------------------------------------------------------------------

    def _store_setup(self, dt: float) -> dict:
        return {
            "analysis": type(self).__name__,
            "f_max": float(self.f_max),
            "dt": float(dt),
            "port_names": [self._spec_label(s) for s in self.ports],
            "port_model": "modal",
            # Reconstruction recipe (DD-070, WP-S8): the resolved port
            # specs, waveform and monitors, so resume() rebuilds the
            # exact same operators from the store (path-only API).
            "recipe": build_recipe(self),
            "params": dict(self.params or {}),
        }

    def _open_store(self, dt: float):
        """Create the project (model written once) or reopen it for a fill-in."""
        from pathlib import Path  # noqa: PLC0415

        from magnelio.io.project import ProjectStore, open_project  # noqa: PLC0415

        path = Path(self.project)
        setup = self._store_setup(dt)
        if (path / "project.json").exists():
            kind = open_project(path).setup.get("analysis")
            if kind != setup["analysis"]:
                raise ValueError(
                    f"project {path} was written by {kind!r}; a "
                    f"{setup['analysis']} run cannot be added to it (one project, "
                    f"one analysis kind) — point project= at a new directory",
                )
            store = ProjectStore(path)  # fill-in: keep the model
        else:
            store = ProjectStore.create(
                path,
                self.mesh,
                geometry=self.geometry,
                setup=setup,
            )
        self._warn_unstreamed_monitors()
        return store, path

    def _warn_unstreamed_monitors(self) -> None:
        # DD-070 streams MonitorFieldTime + MonitorFluxTime +
        # MonitorFieldFrequency + MonitorFarFieldFrequency; warn if a real
        # magnelio data monitor of another kind is present, so its
        # absence from the reader is not a silent surprise.
        from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415
        from magnelio.monitors.field_frequency import (  # noqa: PLC0415
            MonitorFieldFrequency,
        )
        from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
        from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

        _streamed = (
            MonitorFieldTime,
            MonitorFluxTime,
            MonitorFieldFrequency,
            MonitorFarFieldFrequency,
        )
        not_streamed = sorted(
            {
                type(m).__name__
                for m in self.monitors
                if not isinstance(m, _streamed)
                and type(m).__module__.startswith("magnelio.monitors")
            }
        )
        if not_streamed and self._verbose:
            self._note(
                "run",
                f"monitor type(s) {not_streamed} are not streamed to the "
                f"project store; they run in RAM on this pass but are absent "
                f"from the reader and from a resume.",
            )

    @staticmethod
    def _next_run_name(existing: Iterable[str], name: str | None) -> str:
        existing = set(existing)
        if name is not None:
            if name in existing:
                raise ValueError(
                    f"run name {name!r} is already taken in this project; existing "
                    f"runs: {sorted(existing)}",
                )
            return name
        n = len(existing) + 1
        while f"run_{n}" in existing:
            n += 1
        return f"run_{n}"

    def _run_to_store(
        self,
        prepared: _PreparedRun,
        bc_objects: dict,
        dt: float,
        *,
        name: str | None,
        checkpoint_interval: int | None,
        t_end: float | None,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        port_signal_stop_db: float | None,
        max_time_steps: int | str | None,
    ):
        """Stream one march into the project store; return its reader."""
        from magnelio.io.project import open_project  # noqa: PLC0415

        del t_end
        store, path = self._open_store(dt)
        store.mark_analysis_started()
        run_name = self._next_run_name(open_project(path).runs, name)
        excitation_dicts = [excitation_to_dict(e) for e in prepared.excitations]
        ckpt_interval = (
            checkpoint_interval
            if checkpoint_interval is not None
            else max(1, prepared.n_steps_estimate // 8)
        )
        # The SAME monitor objects go to the sink (which drains their
        # snapshots to disk) and the solver (which records into them) —
        # a shared reference, so the drained data is exactly what the
        # solver wrote this run (WP-S9).
        run_monitors = list(self.monitors)
        sink = store.open_run(
            run_name,
            excitations=excitation_dicts,
            excited=None,
            dt=dt,
            f_axis=None,
            channels=prepared.recorder.channels if prepared.recorder is not None else [],
            port_modes=prepared.port_modes,
            port_normal_dx=prepared.port_normal_dx,
            port_line_params=prepared.port_line_params,
            excitation_fns=prepared.drive_items,
            recorder=prepared.recorder,
            port_model="modal",
            energy_stop_db=energy_stop_db,
            port_signal_stop_db=port_signal_stop_db,
            total_time_steps=total_time_steps,
            taper_signals=False,
            monitors=run_monitors,
            grid=self.mesh.grid,
        )
        solver = self._build_solver(
            prepared,
            bc_objects,
            dt,
            monitors=run_monitors,
            total_time_steps=total_time_steps,
            energy_stop_db=energy_stop_db,
            port_signal_stop_db=port_signal_stop_db,
            max_time_steps=max_time_steps,
            sink=sink,
        )
        sink.enable_checkpoints(solver.state_dict, ckpt_interval)
        self._drive_streamed_solver(solver, sink, run_name, path)
        self._note("run", f"streamed run {run_name!r} to project {path}")
        store.mark_analysis_finished(self._report_finished())
        return open_project(path)

    # ── the run clock and the lines around a run (DD-253) ─────────────

    def _note(self, label: str, text: str) -> None:
        """One standalone line under *label*, obeying the verbosity setting."""
        rep = Reporter(label, self._verbose)
        rep.note(text)
        rep.close()

    def _start_run_clock(self) -> None:
        """Open the ``run`` reporter and start the wall clock of this call.

        Created before the setup reporter so the nested reporters
        stand down onto it; a reporter left open by an earlier call
        that raised is closed first.
        """
        self._close_run_clock()
        self._run_reporter = Reporter("run", self._verbose)
        self._run_t0 = time.perf_counter()

    def _close_run_clock(self) -> None:
        rep = getattr(self, "_run_reporter", None)
        if rep is not None:
            rep.close()
            self._run_reporter = None

    def _report_finished(self, n_runs: int = 1) -> float:
        """Print ``finished in …`` and return the elapsed wall time [s]."""
        elapsed = time.perf_counter() - getattr(self, "_run_t0", time.perf_counter())
        rep = getattr(self, "_run_reporter", None)
        if rep is not None:
            runs = f" ({n_runs} runs)" if n_runs > 1 else ""
            rep.note(f"finished in {format_seconds(elapsed)}{runs}")
        self._close_run_clock()
        return elapsed

    def _drive_streamed_solver(self, solver, sink, run_label: str, path) -> None:
        """Run a sink-attached solver under a cooperative Ctrl-C trap (WP-S7).

        Shared by the first-run and the resume paths.  Traps ``SIGINT``
        on the main thread into :meth:`FITTimeDomainSolver.request_stop`
        so a Ctrl-C finishes the in-flight step and checkpoints a
        consistent, resumable state instead of tearing the run down
        mid-step; the previous handler is restored in ``finally``.  Also
        traps ``SIGUSR1`` (POSIX) into the sink's ``request_checkpoint``
        — a snapshot-and-continue signal that writes a resume checkpoint
        at the next flush *without* stopping the march (send ``kill -USR1
        <pid>``).  Finalises the run ``done`` on normal completion
        (writes the run-longer checkpoint) or ``aborted`` on a graceful
        stop, re-raising ``KeyboardInterrupt`` in the latter case (Ctrl-C
        still stops the program — but now leaves a resumable project
        behind).
        """
        import signal  # noqa: PLC0415
        import threading  # noqa: PLC0415

        trap = threading.current_thread() is threading.main_thread()
        prev_int = prev_usr1 = None
        has_usr1 = hasattr(signal, "SIGUSR1")  # POSIX only (no-op on Windows)
        if trap:
            prev_int = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda *_: solver.request_stop())
            # SIGUSR1 = snapshot-and-continue: write a resume checkpoint at
            # the next flush *without* stopping the march (DD-070 follow-up).
            # The handler only sets a flag; the sink writes at the next
            # consistent flush point, so it is as resumable as a periodic one.
            if has_usr1:
                prev_usr1 = signal.getsignal(signal.SIGUSR1)
                signal.signal(
                    signal.SIGUSR1,
                    lambda *_: sink.request_checkpoint(),
                )
        try:
            solver.run()
        except BaseException:
            # Any *other* failure: leave the run non-done, the partial
            # record + last checkpoint (periodic or on-demand) stay on disk.
            sink.close(state="aborted", elapsed=solver._elapsed)
            self._close_run_clock()
            raise
        finally:
            if trap:
                signal.signal(signal.SIGINT, prev_int)
                if has_usr1:
                    signal.signal(signal.SIGUSR1, prev_usr1)

        if solver._aborted:
            sink.close(state="aborted", stop_reason="aborted", elapsed=solver._elapsed)
            self._note(
                "run",
                f"run {run_label!r} aborted at step {solver._resume_step} after "
                f"{format_clock(solver._elapsed or 0.0)}; resume checkpoint saved to {path}",
            )
            self._close_run_clock()
            raise KeyboardInterrupt(
                f"run {run_label} aborted at step {solver._resume_step}; resume from {path}",
            )
        # Book why the run ended (and the achieved |V| level) into the
        # run index — the honest provenance of the derived results,
        # resume-safe like the launch criteria (DD-122).
        sink.close(
            state="done",
            stop_reason=solver._stop_reason,
            final_port_signal_db=solver._final_signal_db,
            elapsed=solver._elapsed,
        )

    @classmethod
    def from_project(cls, project, *, verbose: bool = True):
        """Reconstruct the analysis that produced a project.

        Reads the reconstruction recipe (``setup['recipe']``) and rebuilds
        the analysis on the stored mesh.  The persisted mesh is already
        PEC-consolidated and the recipe's port specs are already resolved,
        so the constructor's PEC consolidation is idempotent (it ORs the
        same bbox walls into an unchanged mask) and its declarative-port
        resolution a no-op — the reconstructed analysis is functionally
        identical to the one that ran, which is what makes a rebuilt-then-
        resumed run bit-exact on deterministically-built (e.g. TEM) ports.

        Field monitors are rebuilt from the recipe; their data lives in
        the monitor write-through streams.
        """
        # Design: WP-S9 (monitor write-through), WP-S8 (reconstruction recipe).
        from magnelio.io.project import Project, open_project  # noqa: PLC0415

        proj = project if isinstance(project, Project) else open_project(project)
        kind = proj.setup.get("analysis")
        if kind != cls.__name__:
            raise ValueError(
                f"project {proj.path} was written by {kind!r}, not by {cls.__name__}",
            )
        recipe = proj.setup.get("recipe")
        if recipe is None:
            raise ValueError(
                f"project {proj.path} carries no reconstruction recipe; it "
                f"was written by an older magnelio (or not by "
                f"{cls.__name__}) "
                f"and cannot be rebuilt for resume.  Re-run it with a current "
                f"magnelio to enable resume.",
            )
        return cls(mesh=proj.mesh, verbose=verbose, **recipe_kwargs(recipe))

    # ------------------------------------------------------------------
    # Operators, boundaries, walls
    # ------------------------------------------------------------------

    def _build_operator(
        self,
        spec: PortSpec,
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
        f_calc: float,
    ):
        """Dispatch to the right builder for the spec type."""
        if isinstance(spec, PortSpecLumped):
            return build_lumped_port(spec, self.mesh, m_eps, m_mu, dt=dt)
        return build_modal_port(
            spec,
            self.mesh,
            m_eps,
            m_mu,
            dt=dt,
            f_calc=f_calc,
        )

    @staticmethod
    def _modes_for_operator(op):
        """Return the per-port Mode list compute_s_parameters expects.

        Modal operators carry a ``discrete_modes`` attribute whose
        items each expose ``.mode`` (a real ``Mode`` instance with
        ``z_modal``).  Lumped operators carry only ``Z0``; we wrap
        that in a ``_LumpedModeStub`` of the same shape.
        """
        if hasattr(op, "discrete_modes"):
            return [dm.mode for dm in op.discrete_modes]
        return [_LumpedModeStub(z0=op.Z0)]

    def _pec_bc_faces(self) -> list[str]:
        """Bbox faces whose declared closure is PEC."""
        return [
            face
            for face, bc_type in bc_type_entries(
                self.boundary_conditions,
            ).items()
            if bc_type == "PEC"
        ]

    def _port_face_names(self) -> set[str]:
        """Bbox faces hosting a modal port, in BC-key form (``"zmin"``)."""
        faces: set[str] = set()
        for s in self.ports:
            value = getattr(getattr(s, "plane", None), "value", None)
            if isinstance(value, str):
                faces.add(value.replace("_", ""))
        return faces

    def _sibc_band(self) -> tuple[float, float]:
        """The band the SIBC ladders are fitted over, ``(f_lo, f_hi)`` [Hz]."""
        return self.f_max / 201.0, float(self.f_max)

    def _sibc_spec(self):
        """Build (once) the SIBC wall spec of this analysis (WP-D5).

        ``None`` on ``wall_model="perturbative"``.  Otherwise the
        WP-D3/WP-D2 chain on the consolidated mesh: enumerate the wall
        update topology (PEC boundary faces minus port faces — port
        planes stay lossless), resolve each tag's conductor, and fit
        one passive ``Z_s`` ladder per distinct conductor over the
        analysis band.  Cached — the mesh and band are fixed per
        analysis, so every excitation (and a resume) reuses the same
        spec.
        """
        if self.wall_model != "sibc":
            return None
        if self._sibc_spec_cache is None:
            from magnelio.materials.surface_impedance import (  # noqa: PLC0415
                fit_wall_impedances,
            )
            from magnelio.mesh._surfaces import (  # noqa: PLC0415
                enumerate_sibc_surfaces,
                resolve_wall_conductors,
            )
            from magnelio.solver._sibc import SIBCSpec  # noqa: PLC0415

            port_faces = self._port_face_names()
            # DD-154: a symmetry face is a mirror plane, not a physical
            # conductor wall — it must not dissipate.
            sym_faces = set(symmetry_entries(self.boundary_conditions))
            bc_faces = tuple(
                f for f in self._pec_bc_faces() if f not in port_faces and f not in sym_faces
            )
            surfaces = enumerate_sibc_surfaces(
                self.mesh,
                bc_pec_faces=bc_faces,
                masked_boundary_faces=self._non_wall_boundary_faces(),
            )
            if not surfaces:
                raise ValueError(
                    "wall_model='sibc': the mesh exposes no conductor "
                    "walls (no PEC/lossy-metal solid surfaces and no "
                    "PEC boundary faces off the port planes).",
                )
            resolved = resolve_wall_conductors(
                self.mesh,
                surfaces,
                sigma=self.wall_sigma,
                mu=self.wall_mu,
                roughness=self.wall_roughness,
                overrides=self._bc_wall_overrides() or None,
            )
            f_lo, f_hi = self._sibc_band()
            fits = fit_wall_impedances(resolved, float(f_lo), float(f_hi))
            self._sibc_spec_cache = SIBCSpec(
                surfaces=tuple(surfaces),
                fits=fits,
            )
        return self._sibc_spec_cache

    def _wire_far_field_ports(self) -> None:
        """Tell far-field monitors where feed guides cross absorbing faces.

        A waveguide port in a CPML face (DD-198) sits at the end of a
        guide that the Huygens box cuts; the monitor leaves the guide's
        interior out of the equivalent surface.  Runtime wiring like the
        accepted-power curve — refreshed per run, not serialised.
        """
        from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415
        from magnelio.ports._modal.port_plane import PortPlane  # noqa: PLC0415

        ff = [m for m in self.monitors if isinstance(m, MonitorFarFieldFrequency)]
        if not ff:
            return
        types = bc_type_entries(self.boundary_conditions)
        footprints: dict[str, list[dict]] = {}
        for spec in self.ports:
            face = getattr(spec, "plane", None)
            value = getattr(face, "value", None)
            if not isinstance(value, str):
                continue
            key = value.replace("_", "")
            if types.get(key) != "CPML" or getattr(spec, "window", None) is None:
                continue
            plane = PortPlane.from_mesh(face, self.mesh, window=spec.window)
            footprints.setdefault(key, []).append(
                {
                    int(face.u_axis): tuple(int(i) for i in plane.u_node_window),
                    int(face.v_axis): tuple(int(i) for i in plane.v_node_window),
                }
            )
        for mon in ff:
            mon._port_footprints = footprints

    def _wire_wall_monitors(self) -> None:
        """Point every MonitorWallLoss at the SIBC accounting (WP-D5).

        On ``wall_model="sibc"`` the monitor must report the SIBC's own
        dissipated power (same faces, weights and Z_s as the operator —
        no double counting); perturbative monitors receive the
        masked-face list (port planes and non-PEC BC faces must not act
        as registered boundary walls).  Called at run/resume time,
        never serialised.
        """
        from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

        masked = self._non_wall_boundary_faces()
        overrides = self._bc_wall_overrides()
        spec = self._sibc_spec() if self.wall_model == "sibc" else None
        for mon in self.monitors:
            if isinstance(mon, MonitorWallLoss):
                mon.masked_faces = masked
                mon.wall_overrides = overrides
                if spec is not None:
                    mon.sibc = spec

    def _bc_wall_overrides(self) -> dict:
        """Per-face wall-conductor overrides ``face -> (σ, μ_r,
        roughness)`` from ``PECBoundary`` declarations that carry
        their own wall material (the boundary condition
        carries the wall model).

        Only the dict form can carry them — a ``BoundaryConditions``
        names types, not wall materials."""
        bc = self.boundary_conditions
        entries = bc if isinstance(bc, dict) else {}
        return {
            face: (value.wall_sigma, value.wall_mu, value.wall_roughness)
            for face, value in entries.items()
            if isinstance(value, PECBoundary) and value.wall_sigma is not None
        }

    def _non_wall_boundary_faces(self) -> tuple[str, ...]:
        """Faces whose boundary-plane PEC coverage must not book walls.

        Port planes (the conductor cross-section where a line
        exits the domain is not a wall), faces whose declared BC is
        not PEC, and declared symmetry faces (DD-154 — a mirror plane
        is not a physical conductor wall; the full model has no wall
        there, so it must not dissipate).  Undeclared faces default to
        PEC in the FIT update and stay wall-eligible.

        Masked faces get CONTINUATION semantics at enumeration time
        (`_masked_face_pec_views`): the adjacent interior plane's
        coverage, zero jump — no wall books there and the target gate
        keeps its previous exclusions.  This also strips the
        historically booked contrast-sampled port-plane cross-section
        families (~1 % on the padded coax) — the "port planes stay
        lossless" rule now holds for the conformal cell booking too.
        """
        non_pec = {
            face
            for face, bc_type in bc_type_entries(
                self.boundary_conditions,
            ).items()
            if bc_type != "PEC"
        }
        sym_faces = set(symmetry_entries(self.boundary_conditions))
        return tuple(sorted(non_pec | sym_faces | self._port_face_names()))

    def _resolve_bc(self) -> dict:
        """The runtime BC object for each of the six faces.

        Instances declared by the user pass through untouched (a
        ``PECBoundary`` may carry this face's wall material);
        type strings — and faces the declaration left out, which close
        with the PEC default — are materialised here, because the
        solver dispatches on ``apply_E``/``apply_H`` attributes and a
        raw string would be a silent no-op.
        """
        bc = self.boundary_conditions
        declared = bc.to_dict() if isinstance(bc, BoundaryConditions) else bc
        out = {}
        for face, bc_type in bc_type_entries(bc).items():
            value = declared.get(face)
            out[face] = (
                value
                if value is not None and not isinstance(value, str)
                else materialize_boundary(
                    face,
                    bc_type,
                    self.mesh.grid,
                    cpml_thickness_cells=self.cpml_thickness_cells,
                )
            )
        return out

    @staticmethod
    def _spec_label(spec) -> str:
        return spec.name

    def __repr__(self) -> str:
        labels = [self._spec_label(s) for s in self.ports]
        sources = [s.name for s in self.sources]
        return f"{type(self).__name__}(ports={labels}, sources={sources}, f_max={self.f_max:.3e})"


# ═════════════════════════════════════════════════════════════════════
# Resume (DD-070, WP-S8) — shared by every time-domain problem class
# ═════════════════════════════════════════════════════════════════════


def _load_freq_accumulators(project_path, run_name, monitors, n_completed):
    """Restore MonitorFieldFrequency accumulators from fields_freq.h5 (resume).

    The DFT result is the monitor's own ``fields_freq.h5``, dumped at each
    checkpoint tagged with the step it reflects.  If that step matches the
    checkpoint's ``n_completed`` the accumulators are reloaded so the resumed
    run keeps integrating bit-exactly; on a mismatch (a hard crash *between*
    the checkpoint and the frequency-result write) it raises rather than
    silently integrating from a wrong partial DFT.  A no-op when the run
    carries no frequency monitors.
    """
    from pathlib import Path  # noqa: PLC0415

    from magnelio.monitors.field_frequency import (  # noqa: PLC0415
        MonitorFieldFrequency,
    )

    freq_mons = [m for m in monitors if isinstance(m, MonitorFieldFrequency)]
    if not freq_mons:
        return
    ff = Path(project_path) / "runs" / run_name / "fields_freq.h5"
    if not ff.exists():
        raise ValueError(
            f"resume needs {ff} for frequency monitor(s) "
            f"{[m.name for m in freq_mons]}, but it is missing",
        )
    import h5py  # noqa: PLC0415

    with h5py.File(ff, "r") as f:
        freq_step = int(f.attrs["n_completed"])
        if freq_step != int(n_completed):
            raise ValueError(
                f"fields_freq.h5 is at step {freq_step} but the checkpoint is "
                f"at {n_completed} — likely a hard crash during a checkpoint "
                f"dump.  The frequency accumulator cannot be resumed "
                f"consistently; restart the run or drop the frequency monitor.",
            )
        for mon in freq_mons:
            if mon.name not in f:
                continue
            bins_grp = f[mon.name]["bins"]
            bins = {comp: bins_grp[comp][()] for comp in bins_grp}
            mon.load_result_dump({"bins": bins})


def _load_wall_loss_accumulators(project_path, run_name, monitors, n_completed):
    """Restore MonitorWallLoss accumulators from wall_loss.h5 (resume).

    The wall-loss result file follows the frequency monitor's contract
    exactly (DD-082 addendum): dumped at each checkpoint, tagged with the
    step it reflects, reloaded only when that step matches the
    checkpoint's ``n_completed`` — a mismatch means a hard crash between
    the two writes, and raising beats integrating on from a wrong partial
    DFT.  A no-op when the run carries no wall-loss monitors.
    """
    from pathlib import Path  # noqa: PLC0415

    from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

    wl_mons = [m for m in monitors if isinstance(m, MonitorWallLoss)]
    if not wl_mons:
        return
    wl = Path(project_path) / "runs" / run_name / "wall_loss.h5"
    if not wl.exists():
        raise ValueError(
            f"resume needs {wl} for wall-loss monitor(s) "
            f"{[m.name for m in wl_mons]}, but it is missing",
        )
    import h5py  # noqa: PLC0415

    with h5py.File(wl, "r") as f:
        wl_step = int(f.attrs["n_completed"])
        if wl_step != int(n_completed):
            raise ValueError(
                f"wall_loss.h5 is at step {wl_step} but the checkpoint is at "
                f"{n_completed} — likely a hard crash during a checkpoint "
                f"dump.  The wall-loss accumulator cannot be resumed "
                f"consistently; restart the run or drop the monitor.",
            )
        for mon in wl_mons:
            if mon.name not in f:
                continue
            raw = f[mon.name]["raw"]
            hg = raw["h_bins"]
            mon.load_result_dump(
                {
                    "h_bins": [hg[str(i)][()] for i in range(len(hg))],
                    "ref_bins": {k: raw["ref_bins"][k][()] for k in raw["ref_bins"]},
                }
            )


def _load_far_field_accumulators(project_path, run_name, monitors, n_completed):
    """Restore MonitorFarFieldFrequency accumulators from far_field.h5 (resume).

    Same contract as the frequency and wall-loss dumps: reloaded only
    when the file's step matches the checkpoint's ``n_completed``.  A
    no-op when the run carries no far-field monitors.
    """
    from pathlib import Path  # noqa: PLC0415

    from magnelio.io.project import _read_far_field_dump  # noqa: PLC0415
    from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

    ff_mons = [m for m in monitors if isinstance(m, MonitorFarFieldFrequency)]
    if not ff_mons:
        return
    ff = Path(project_path) / "runs" / run_name / "far_field.h5"
    if not ff.exists():
        raise ValueError(
            f"resume needs {ff} for far-field monitor(s) "
            f"{[m.name for m in ff_mons]}, but it is missing",
        )
    import h5py  # noqa: PLC0415

    with h5py.File(ff, "r") as f:
        ff_step = int(f.attrs["n_completed"])
        if ff_step != int(n_completed):
            raise ValueError(
                f"far_field.h5 is at step {ff_step} but the checkpoint is at "
                f"{n_completed} — likely a hard crash during a checkpoint "
                f"dump.  The far-field accumulator cannot be resumed "
                f"consistently; restart the run or drop the monitor.",
            )
    run_dir = Path(project_path) / "runs" / run_name
    for mon in ff_mons:
        mon.load_result_dump(_read_far_field_dump(run_dir, mon.name))


def _resume_transient(
    analysis: AnalysisTD,
    proj,
    run_name: str,
    excitations: Sequence[Excitation],
    *,
    energy_stop_db: float | None,
    total_time_steps: int | None,
    port_signal_stop_db: float | str | None,
    max_time_steps: int | str | None,
    checkpoint_interval: int | None,
    verbose: bool,
    prepare=None,
):
    """Continue one project-backed march from its checkpoint (WP-S8).

    Rebuilds the run's operators from the reconstructed ``analysis``,
    binds the stored excitations, loads the latest ``checkpoint.h5``
    into a freshly-built solver, reopens ``results.h5`` (truncated back
    to the checkpoint step), and marches on with the (optionally
    overridden) stop criterion — so a resumed run is bit-identical to an
    uninterrupted run of the same total length on a
    deterministically-built line (the DTBC seam injects nothing; the
    checkpoint carries the full CPML ψ + DTBC convolution history,
    DD-070).  Returns ``(solver, prepared, run_monitors)`` for the
    caller's post-processing.
    """
    from magnelio.io.project import ProjectStore  # noqa: PLC0415

    run_meta = proj._run_info(run_name)
    ckpt = proj.checkpoint_state(run_name)
    if ckpt is None:
        raise ValueError(
            f"run {run_name!r} has no checkpoint.h5 to resume from — it was "
            f"streamed without resume checkpoints, or aborted before the "
            f"first checkpoint interval.  Re-run it with checkpoint_interval= "
            f"set (the default writes ~8 per run plus a final one).",
        )
    n_completed = int(ckpt["n_completed"])
    dt = float(run_meta["dt"])
    state = run_meta.get("state")
    original_esd = run_meta.get("energy_stop_db")

    # Criterion resolution.  Passing *no* knob inherits the run's
    # original criterion (finish an aborted run to its launch target);
    # passing *any* uses exactly what was given, with the other knobs
    # left at their disabled/unbounded defaults — the knobs are
    # independent, as in ``run``, so ``energy_stop_db=`` alone
    # switches a formerly bounded run to an energy-gated continuation
    # without the old cap silently re-blocking it.
    if energy_stop_db is None and total_time_steps is None and port_signal_stop_db is None:
        energy_stop_db = original_esd
        total_time_steps = run_meta.get("total_time_steps")
        port_signal_stop_db = run_meta.get("port_signal_stop_db")
    port_signal_stop_db = analysis._resolve_port_signal_stop(port_signal_stop_db)

    # The resume must actually advance past the checkpoint, else it is a
    # no-op (or, on a done energy-gated run, an immediate re-stop).
    if total_time_steps is not None:
        if total_time_steps <= n_completed:
            raise ValueError(
                f"resume target total_time_steps={total_time_steps} is not "
                f"past the checkpoint step {n_completed}; pass a larger "
                f"total_time_steps= to continue (or an energy_stop_db=).",
            )
    elif energy_stop_db is None and port_signal_stop_db is None:
        raise ValueError(
            f"run {run_name!r} has no stop criterion to resume with; pass "
            f"energy_stop_db= (dB below peak), port_signal_stop_db=, or "
            f"total_time_steps=.",
        )
    elif (
        state == "done"
        and original_esd is not None
        and energy_stop_db is not None
        and energy_stop_db <= original_esd
        # A cap- or stall-truncated run is "done" without having reached
        # its criterion — resuming it with the same criterion is the
        # intended escape hatch, not an immediate re-stop.  Legacy
        # projects carry no stop_reason and keep the historical guard.
        and run_meta.get("stop_reason") in (None, "energy")
    ):
        raise ValueError(
            f"run {run_name!r} already reached energy_stop_db={original_esd} "
            f"(decayed that far below peak), so resuming at "
            f"energy_stop_db={energy_stop_db} would stop immediately.  Pass a "
            f"deeper (larger) energy_stop_db= or a total_time_steps= to run "
            f"it longer.",
        )
    if isinstance(max_time_steps, str) and max_time_steps != "auto":
        raise ValueError(
            f"max_time_steps must be an int (steps), None or 'auto'; got {max_time_steps!r}",
        )

    mesh = analysis.mesh
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    bc_objects = analysis._resolve_bc()

    # Rebuild the operators + excitations + a fresh recorder exactly as
    # the first run did (same mesh, specs, dt, f_calc) — the fresh
    # recorder restarts at local index 0, i.e. global step n_completed.
    # ``prepare`` lets a pipeline with its own construction (the band
    # scattering path) supply the operators while reusing everything
    # below — sink reopening, checkpoint load, monitor restoration.
    prepared = (
        prepare(m_eps, m_mu)
        if prepare is not None
        else analysis._prepare_run(list(excitations), m_eps, m_mu, dt)
    )
    ckpt_interval = (
        checkpoint_interval
        if checkpoint_interval is not None
        else max(1, prepared.n_steps_estimate // 8)
    )

    # Monitors rebuilt from the recipe; their append streams are truncated
    # back to each monitor's checkpointed sample count so the resumed run
    # appends onward without a gap or duplicate.  SIBC wall model (WP-D5):
    # the spec is rebuilt from the recipe-restored analysis (same mesh,
    # band and overrides ⇒ same operator).
    analysis._wire_wall_monitors()
    analysis._wire_far_field_ports()
    run_monitors = list(analysis.monitors)
    stream_keep = {
        name: int(msd["next_idx"])
        for name, msd in ckpt.get("monitors", {}).items()
        if "next_idx" in msd
    }
    store = ProjectStore(proj.path)
    sink = store.reopen_run(
        run_name,
        recorder=prepared.recorder,
        excitation_fns=prepared.drive_items,
        dt=dt,
        n_keep=n_completed,
        step_offset=n_completed,
        monitors=run_monitors,
        grid=mesh.grid,
        monitor_keep=stream_keep,
        flux_keep=stream_keep,
    )
    solver = analysis._build_solver(
        prepared,
        bc_objects,
        dt,
        monitors=run_monitors,
        total_time_steps=total_time_steps,
        energy_stop_db=energy_stop_db,
        port_signal_stop_db=port_signal_stop_db,
        max_time_steps=max_time_steps,
        sink=sink,
        start_step=n_completed,
    )
    # Build the constant operators, then load the evolving state; the
    # excitation closures armed by _prepare_run survive the load
    # (state_dict restores only the source-history buffers, not the
    # closures).
    solver.setup()
    solver.load_state_dict(ckpt)
    # Frequency monitors: reload the DFT accumulator from fields_freq.h5 (its
    # own result file), which the checkpoint's E/H does not carry — with a
    # step-check against the checkpoint (DD-070 follow-up).
    _load_freq_accumulators(proj.path, run_name, run_monitors, n_completed)
    # Wall-loss monitors: same contract, own result file (DD-082 addendum).
    _load_wall_loss_accumulators(proj.path, run_name, run_monitors, n_completed)
    # Far-field monitors: same contract, own result file (DD-173).
    _load_far_field_accumulators(proj.path, run_name, run_monitors, n_completed)
    sink.enable_checkpoints(solver.state_dict, ckpt_interval)

    target = "∞" if total_time_steps is None else str(total_time_steps)
    analysis._note(
        "run",
        f"resuming run {run_name!r} from step {n_completed} to {target} "
        f"(energy_stop_db={energy_stop_db}, port_signal_stop_db={port_signal_stop_db})",
    )
    analysis._drive_streamed_solver(solver, sink, run_name, proj.path)
    return solver, prepared, run_monitors


def _resume_td(
    proj,
    name=None,
    *,
    energy_stop_db: float | None = None,
    total_time_steps: int | None = None,
    port_signal_stop_db: float | str | None = None,
    max_time_steps: int | str | None = "auto",
    checkpoint_interval: int | None = None,
    verbose: bool = True,
):
    """Back :func:`magnelio.resume` for ``setup['analysis'] == "AnalysisTD"``."""
    from magnelio.analysis._recipe import excitation_from_dict  # noqa: PLC0415
    from magnelio.io.project import ProjectStore, open_project  # noqa: PLC0415

    run_name = proj._run_name_for_excited(name)
    analysis = AnalysisTD.from_project(proj, verbose=verbose)
    excitations = [excitation_from_dict(d) for d in proj._run_excitations(run_name)]
    analysis._start_run_clock()
    store = ProjectStore(proj.path)
    store.mark_analysis_started()
    _resume_transient(
        analysis,
        proj,
        run_name,
        excitations,
        energy_stop_db=energy_stop_db,
        total_time_steps=total_time_steps,
        port_signal_stop_db=port_signal_stop_db,
        max_time_steps=max_time_steps,
        checkpoint_interval=checkpoint_interval,
        verbose=verbose,
    )
    store.mark_analysis_finished(analysis._report_finished())
    return open_project(proj.path)


__all__ = ["AnalysisTD", "TDResult"]
