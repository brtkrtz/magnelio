"""AnalysisScatteringTD — high-level FIT-TD S-parameter workflow.

Single-physics-question, multi-port-list-input class in the same style
as :class:`magnelio.analysis.AnalysisEigenmode`: takes a mesh, a list of
declarative port specs, the boundary closure, and the analysis band
(``f_max``, optionally ``f_min`` / ``n_freq``); returns a
:class:`ScatteringTDResult` carrying the requested S-matrix columns
plus the per-excitation V/I time series and the sampled excitation
waveform — everything a notebook needs for both S-parameter plots and
time-domain inspection.

``f_max`` is the single frequency source of the analysis: it sizes the
default frequency axis (``linspace(max(f_min, f_max/n_freq), f_max,
n_freq)`` — power waves are undefined at DC, hence the ``f_max/n_freq``
lower bound), fixes the mode-calculation frequency of the port builder
(which only parameterises the Mur velocity of analytical-path modes —
DTBC-terminated and band-pipeline channels are exact per frequency and
do not depend on it), and caps the default excitation waveform, which
is derived per excited mode: DC-inclusive Gaussian for TEM/lumped,
band-limited modulated Gaussian over ``[max(f_cutoff, f_min), f_max]``
for TE/TM.  An explicit ``excitation=ExcitationSpec(...)`` stays
available as an override, as does ``run(f_axis=...)`` for a custom
frequency axis.

Port pipeline dispatch.  The default is the **modal
pipeline** (``port_model="modal"``): exact DTBC on certified uniform
chains, modal Mur-1st on inhomogeneous QTEM/hybrid channels —
measured −26…−39 dB ``|S11|`` on a realistic shielded microstrip at
``|S21|`` errors below 0.01 dB, seconds of runtime, full time-domain
power-wave access.  That trade (speed + TD signals over deep
reflection floors) is the accepted production default for QTEM lines
(developer decision, 2026-07-10); a verbose notice names the
Mur-fallback channels.  For reflection-critical work,
``port_model="band"`` (or ``"auto"``) engages the broadband
band-subspace DTBC pipeline: mode families tracked over the
band, one reflection-free operator per port, one pulsed record per
excitation, per-frequency true-mode decomposition — measured floors
−159…−231 dB, at kernel-build cost, a longer record, a strictly
positive lower band edge, and no time-domain power waves.

One ``run()`` call produces the S-parameter columns for *every*
``(port, mode)`` pair listed in ``excited``.  Each excited pair drives
one independent FIT-TD simulation; the resulting columns are merged
via :meth:`SParameterResult.merge` and exposed through the wrapper's
``s_params`` field.

Example
-------
>>> result = AnalysisScatteringTD(
...     mesh=mesh,          # carries the boundary closure
...     ports=[PortSpecCoax(name="port1", ...),
...            PortSpecCoax(name="port2", ...)],
...     f_max=10e9,
... ).run(excited=["port1"])
>>> S11 = result.S("port1", "port1")
>>> S21 = result.S("port2", "port1")

This is the high-level API: the user describes a problem
(scattering analysis on a multi-port network) and gets a result
back.  The component path — building specs, operators, recorder, solver
by hand and calling ``compute_s_parameters`` — remains available for
custom workflows this class does not cover (simultaneous multi-port
drive, nonlinear materials, hand-tuned excitations).  Lumped discrete
ports *are* covered here (``PortSpecLumped``, see "Supported port
specs" below).

Supported port specs
--------------------
* Declarative (WP4.1) — ``PortWaveguide`` ("solve whatever
  is on this face"; the TEM/QTEM/TE-TM path is selected from the
  mesh cross-section at construction time) and ``PortAnalytical``
  (closed-form coax / rect-WG reference).  Resolved into the concrete
  specs below via :func:`resolve_declarative_port`, reading the
  mesh's consolidated PEC mask — so a face closed with PEC counts as
  a conductor, and one closed with PMC (a symmetry plane) does not.
* Modal — ``PortSpecCoax``, ``PortSpecRectWG``, ``PortSpecNumerical``,
  ``PortSpecMultiConductor``.  Built via :func:`build_modal_port`,
  the operator carries one or more ``DiscreteMode`` instances whose
  ``Mode.z_modal(omega)`` provides the reference impedance.
* Lumped — ``PortSpecLumped``.  Built via
  :func:`build_lumped_port`; the analysis synthesises an internal
  ``_LumpedModeStub`` with constant ``z_modal(omega) = Z0`` for the
  power-wave decomposition.

Note
----
The per-spec ``excitation`` field on modal specs is ignored —
``AnalysisScatteringTD`` uses its own ``excitation`` argument and
applies it via ``set_excitation`` per excited pair.  Component-level
scripts that build operators directly from specs continue to honour
``spec.excitation``.
"""

# Design: DD-063/DD-064 (port pipeline dispatch, modal default for QTEM lines),
# DD-057 (band-subspace DTBC pipeline), DD-103 (boundary closure carried on the
# mesh).

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence, Union

import numpy as np
from scipy.special import erfcinv

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.analysis._recipe import build_scattering_recipe, recipe_kwargs
from magnelio.analysis.result_interface import RunSettings, ScatteringResultMixin
from magnelio.boundaries.boundary_conditions import (
    BoundaryConditions,
    bc_type_entries,
    cpml_thickness_of,
    materialize_boundary,
    symmetry_entries,
)
from magnelio.boundaries.pec import PECBoundary
from magnelio.constants import C0
from magnelio.mesh.mesher import Mesh
from magnelio.ports._lumped import PortSpecLumped, build_lumped_element, build_lumped_port
from magnelio.ports._modal.band_dtbc import band_source_spectrum
from magnelio.ports._modal.factory import (
    ExcitationSpec,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    build_band_dtbc_port,
    build_modal_port,
)
from magnelio.ports._modal.mode_report import PortReport
from magnelio.ports.declarative import (
    PortAnalytical,
    PortLumped,
    PortWaveguide,
    resolve_declarative_port,
)
from magnelio.ports.recorder import PortSignalRecorder
from magnelio.post.modal_sparameters import (
    compute_band_s_parameters,
    compute_s_parameters,
    destaggered_power_waves,
)
from magnelio.post.sparameter_result import SParameterResult
from magnelio.signals.signal_1d import Signal1D
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import (
    spectral_dt,
)

PortSpec = Union[
    PortSpecCoax,
    PortSpecRectWG,
    PortSpecNumerical,
    PortSpecMultiConductor,
    PortSpecLumped,
    PortWaveguide,
    PortAnalytical,
]

ExcitedSpec = Union[str, tuple[str, int]]

# Auto runtime cap for unbounded runs (DD-122), in units of the
# auto-sized step estimate (itself ~25 diagonal transits): ~10³
# transits total.  In ring-down terms the cap accommodates a loaded Q
# of about 900·(structure size/wavelength) before a 60-dB decay is cut
# short — chosen to cover realistic narrow-band filters while still
# bounding a criterion-defeating run to minutes, not hours.
_RUNTIME_CAP_ESTIMATES = 40


def _scattering_run_name(excited_chan: tuple[str, int]) -> str:
    """Canonical run-directory name for one excited (port, mode) pair."""
    return f"{excited_chan[0]}_mode{excited_chan[1]}"


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


def _sampled_excitation(waveform_fn, n_steps: int, dt: float) -> Signal1D:
    """The run's excitation waveform sampled on its own step axis."""
    t_axis = np.arange(n_steps) * dt
    return Signal1D(
        t=t_axis,
        values=np.array([waveform_fn(float(t)) for t in t_axis], dtype=float),
        dt=dt,
        label="excitation",
    )


def _renormalize_freq_monitors(monitors, reference_signal) -> None:
    """Give a finished run's excitation to its frequency monitors.

    Their DFT bins are the field folded with that waveform's spectrum;
    dividing it out is what turns them into fields per 1 W CW.  Doing it
    here, once per run, is what lets ``MonitorFieldFrequency.data`` state
    a unit instead of depending on whether the caller remembered.
    """
    from magnelio.monitors.field_frequency import renormalize_all  # noqa: PLC0415

    renormalize_all(monitors, reference_signal)


@dataclass(frozen=True)
class _LumpedModeStub:
    """Mode-shaped stub for ``PortOperatorLumped`` in compute_s_parameters.

    ``compute_s_parameters`` only ever calls ``mode.z_modal(omega)`` on
    the per-port Mode list to evaluate the reference impedance for
    power-wave decomposition.  A lumped port has a frequency-
    independent Thévenin impedance, so this stub returns ``Z0`` for any
    omega.  Constructed by ``AnalysisScatteringTD`` internally; not
    part of the public API.
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


@dataclass(frozen=True)
class ScatteringTDResult(ScatteringResultMixin):
    """Result of one :meth:`AnalysisScatteringTD.run` call.

    Attributes
    ----------
    s_params : SParameterResult
        Frequency-domain S-matrix columns for every excited pair.
    signals : dict[(str, int), dict[(str, int), (Signal1D, Signal1D)]]
        Outer key: the excited ``(port_name, mode_idx)`` pair.  Inner
        dict: every observed channel for that excitation, mapped to a
        ``(V_signal, I_signal)`` pair on the same time axis.  Buffers
        are already trimmed to the actual leapfrog count via
        :meth:`PortSignalRecorder.finalize`.
    reference_signal : Signal1D
        The excitation waveform sampled on the same time axis as the
        recorded signals.  Useful for monitor renormalisation
        (``MonitorFieldFrequency.renormalize``) and for
        incident/reflected decomposition in time-domain plots.  On
        multi-excitation runs with auto-derived waveforms this is the
        waveform of the *longest* run — per-mode waveforms can differ
        (cut-off-dependent band); the S-parameters are unaffected.
    dt : float
        Solver time step [s].
    n_actual_steps : int
        Number of leapfrog steps actually executed.  Equals
        ``len(reference_signal.values)``; can be smaller than the
        configured ``total_time_steps`` when ``energy_stop_db``
        triggered an early termination.
    port_modes : dict[str, list] or None
        Per-port ordered Mode list (lumped ports carry a
        ``_LumpedModeStub``), as used by the S-parameter
        post-processing.  Backs the time-domain power-wave accessors
        ``a(...)`` / ``b(...)``.
    port_normal_dx : dict[str, float] or None
        Per-port boundary cell size along the port normal, as passed
        to the S-parameter de-stagger.  Backs the default
        (``destagger=True``) time-domain power waves.
    port_line_params : dict[(str, int), tuple] or None
        Per-channel certified discrete line parameters
        (``PortOperatorModal.dtbc_line_params``), as passed to the
        S-parameter de-stagger.  Backs the default time-domain power
        waves; channels missing here use the continuum factors.
    port_model_used : str or None
        Which port pipeline produced this result: ``"modal"`` (the
        per-spec modal port operators) or ``"band"``
        (the broadband band-subspace DTBC pipeline).
    settings : RunSettings or None
        The settings this run was produced with — frequency range,
        time step, why the marching stopped, precision and backend.

    Notes
    -----
    The accessors :meth:`S`, :meth:`db` and :attr:`f_axis` delegate to
    ``s_params``, so a script can write ``result.S("port2", "port1")``
    without unpacking anything.

    Use :meth:`a` / :meth:`b` — not ``V − s(t)`` — for the
    incident/reflected split in time-domain plots: ``V`` is the *total*
    modal voltage and ``s(t)`` is the source waveform, not the launched
    wave, so the difference is not the reflected wave.
    """

    s_params: SParameterResult
    signals: dict
    reference_signal: Signal1D
    dt: float
    n_actual_steps: int
    port_modes: dict | None = None
    port_normal_dx: dict | None = None
    port_line_params: dict | None = None
    port_model_used: str | None = None
    # Per-excitation reference waveforms (multi-excitation modal runs
    # auto-derive per-mode waveforms); backs the f_axis= recompute.
    reference_signals: dict | None = None
    # Settings the run was produced with (result contract).
    settings: "RunSettings | None" = None

    @property
    def f_axis(self) -> np.ndarray:
        """Frequency axis of the S-matrix [Hz], ascending."""
        return self.s_params.f_axis

    @property
    def channels(self) -> tuple:
        """Observed ``(port_name, mode_idx)`` pairs, in S-matrix order."""
        return self.s_params.channels

    @property
    def excitations(self) -> tuple:
        """Excited ``(port_name, mode_idx)`` pairs — the S-matrix columns present."""
        return self.s_params.excitations

    def _channel_cutoffs(self) -> dict | None:
        """Per-channel cut-off frequency [Hz] from the port-mode records."""
        return _cutoffs_from_port_modes(self.port_modes)

    def _s_params_on(self, f_axis) -> SParameterResult:
        """Recompute the S-matrix on a custom frequency axis.

        Runs the same per-excitation pipeline as the original run,
        using the per-excitation reference waveforms where recorded
        (``reference_signals``) and the single ``reference_signal``
        otherwise.
        """
        from magnelio.post.modal_sparameters import (  # noqa: PLC0415
            compute_s_parameters,
        )

        f_axis = np.asarray(f_axis, dtype=float)
        cols = []
        for excited, sigs in self.signals.items():
            ref = (self.reference_signals or {}).get(
                excited,
                self.reference_signal,
            )
            s_dict = compute_s_parameters(
                recorder_signals=sigs,
                port_modes=self.port_modes,
                excited=excited,
                reference_signal=ref,
                f_axis=f_axis,
                port_normal_dx=self.port_normal_dx,
                port_line_params=self.port_line_params,
            )
            cols.append(
                SParameterResult.from_single_excitation(
                    s_dict,
                    excited,
                    f_axis,
                )
            )
        return cols[0] if len(cols) == 1 else SParameterResult.merge(cols)

    def S(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
        f_axis=None,
    ) -> np.ndarray:
        """One complex S-parameter over the frequency axis.

        Parameters
        ----------
        out_port, in_port : str
            Observed and excited port names.  ``S("port2", "port1")``
            is the transmission from port 1 to port 2; equal labels give
            a reflection.
        mode_out, mode_in : int
            Mode index at the observed / excited port (default 0, the
            fundamental).  Only needed on multi-mode ports.
        f_axis : array-like, optional
            Custom frequency axis [Hz].  The S-matrix is then recomputed
            from the recorded time signals on that axis instead of being
            read off the run's own axis — use it to resolve a narrow
            feature without re-running the simulation.  Frequencies
            outside the excitation's band carry no usable signal.

        Returns
        -------
        numpy.ndarray
            Complex S-parameter, one entry per frequency.

        Raises
        ------
        KeyError
            If the pair was never recorded — ``in_port`` must be among
            :attr:`excitations` and ``out_port`` among :attr:`channels`.
        """
        source = self.s_params if f_axis is None else self._s_params_on(f_axis)
        return source.S(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
        )

    def db(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
        floor_db: float = -200.0,
        f_axis=None,
    ) -> np.ndarray:
        """One S-parameter in decibels — ``20 log10 |S|``.

        Parameters
        ----------
        out_port, in_port : str
            Observed and excited port names, as in :meth:`S`.
        mode_out, mode_in : int
            Mode indices at the observed / excited port (default 0).
        floor_db : float, default -200.0
            Values below this are clamped to it, so an exact zero
            (an unexcited channel, a perfect null) yields a finite
            number instead of ``-inf`` and stays plottable.
        f_axis : array-like, optional
            Custom frequency axis; recomputed as in :meth:`S`.

        Returns
        -------
        numpy.ndarray
            Magnitude in dB, one entry per frequency.
        """
        source = self.s_params if f_axis is None else self._s_params_on(f_axis)
        return source.db(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
            floor_db=floor_db,
        )

    def a(
        self,
        port: str,
        mode: int = 0,
        *,
        excited: Union[str, tuple[str, int], None] = None,
        f_ref: float | None = None,
        destagger: bool = True,
    ) -> Signal1D:
        """Incident power-wave time series ``a(t)`` [√W].

        With ``destagger=True`` (default) the decomposition runs on
        the rfft axis with the same per-frequency corrections as
        :func:`compute_s_parameters` — temporal Yee half-step
        rotation, spatial two-plane de-stagger (exact discrete
        ``λ^{1/2}`` factor on DTBC-certified channels, continuum
        ``e^{−γ·d/2}`` otherwise), and the exact discrete wave
        impedance where certified — then transforms back.  ``b(t)``
        then shows the *true* outgoing wave down to the port floor.
        ``destagger=False`` restores the historical co-located split
        ``(V/√Z ± √Z·I)/2`` with midpoint-averaged I, which leaks
        ``≈ β·dz/4`` of the incident pulse into ``b`` (a
        derivative-of-pulse ghost, −22 dB at λ/20 meshes).

        Parameters
        ----------
        port : str
            Observed port name.
        mode : int, default 0
            Observed mode index.
        excited : str or (str, int), optional
            Which excitation run to read.  May be omitted when the
            result holds exactly one excitation.
        f_ref : float, optional
            Frequency [Hz] at which the frozen reference impedance
            ``Z = z_modal(2π·f_ref)`` is evaluated.  Defaults to the
            centre of the result's frequency axis.  With
            ``destagger=False`` this Z parameterises the whole
            decomposition; with ``destagger=True`` it only backs the
            out-of-band fallback bins (see
            :func:`~magnelio.post.modal_sparameters.
            destaggered_power_waves`).  Raises if Z is not real at
            ``f_ref`` (mode evanescent there).
        destagger : bool, default True
            Apply the exact frequency-domain de-stagger.

        Returns
        -------
        Signal1D
            On the same time axis as the recorded V.
        """
        return self._power_wave(
            port,
            mode,
            excited,
            f_ref,
            sign=+1.0,
            destagger=destagger,
        )

    def b(
        self,
        port: str,
        mode: int = 0,
        *,
        excited: Union[str, tuple[str, int], None] = None,
        f_ref: float | None = None,
        destagger: bool = True,
    ) -> Signal1D:
        """Outgoing power-wave time series ``b(t)`` [√W].

        See :meth:`a` for conventions, stagger handling, and the
        ``f_ref`` / ``destagger`` semantics.
        """
        return self._power_wave(
            port,
            mode,
            excited,
            f_ref,
            sign=-1.0,
            destagger=destagger,
        )

    def _power_wave(
        self,
        port: str,
        mode: int,
        excited: Union[str, tuple[str, int], None],
        f_ref: float | None,
        sign: float,
        destagger: bool = False,
    ) -> Signal1D:
        if self.port_modes is None:
            raise ValueError(
                "this ScatteringTDResult carries no port_modes; "
                "time-domain power waves are unavailable",
            )
        if self.port_model_used == "band":
            # Per-frequency true-mode decomposition of the band pipeline: DD-057.
            raise ValueError(
                "time-domain power waves are not available on band-"
                "DTBC results: the band port's recorded V/I channels "
                "are fixed subspace projections whose incident/"
                "outgoing split is defined per frequency through the "
                "true-mode phasors — a scalar (V ∓ Z·I)/2 "
                "split has no calibrated Z and would show the "
                "incident wave in b.  Inspect the raw V/I via "
                "result.signals and take S-parameters from result.S; "
                "a per-frequency-synthesised time-domain split is a "
                "possible future extension.",
            )
        key = self._resolve_excited_key(excited)
        channels = self.signals[key]
        chan = (port, mode)
        if chan not in channels:
            raise KeyError(
                f"channel {chan!r} not recorded for excitation {key!r}; "
                f"available: {sorted(channels.keys())}",
            )
        if port not in self.port_modes:
            raise KeyError(
                f"port {port!r} not in port_modes (available: {sorted(self.port_modes.keys())})",
            )
        modes = self.port_modes[port]
        if not 0 <= mode < len(modes):
            raise ValueError(
                f"mode index {mode} out of range for port {port!r} with {len(modes)} mode(s)",
            )
        if f_ref is None:
            f_axis = self.f_axis
            f_ref = 0.5 * (float(f_axis[0]) + float(f_axis[-1]))
        Z = complex(modes[mode].z_modal(2.0 * math.pi * float(f_ref)))
        if abs(Z.imag) > 1e-9 * abs(Z):
            raise ValueError(
                f"z_modal({f_ref:.4g} Hz) = {Z:.4g} is not real "
                f"(mode evanescent at f_ref?); pass an in-band f_ref=",
            )
        sqrt_z = math.sqrt(Z.real)

        V_sig, I_sig = channels[chan]

        if destagger:
            line_params = (
                self.port_line_params.get(chan) if self.port_line_params is not None else None
            )
            normal_dx = self.port_normal_dx.get(port) if self.port_normal_dx is not None else None
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

    def _resolve_excited_key(
        self,
        excited: Union[str, tuple[str, int], None],
    ) -> tuple[str, int]:
        keys = list(self.signals.keys())
        if excited is None:
            if len(keys) == 1:
                return keys[0]
            raise ValueError(
                f"result holds {len(keys)} excitations {sorted(keys)}; pass excited= to select one",
            )
        key = (excited, 0) if isinstance(excited, str) else tuple(excited)
        if key not in self.signals:
            raise KeyError(
                f"excitation {key!r} not in result; available: {sorted(keys)}",
            )
        return key


@dataclass
class AnalysisScatteringTD:
    """High-level FIT-TD scattering analysis for multi-port networks.

    Parameters
    ----------
    mesh : Mesh
    ports : list of port spec, optional
        ``None`` (default) uses the declarative ports the mesh carries
        (declared before meshing via
        :meth:`~magnelio.geo.GeometryModel.add_port`).
        Passing ``ports=`` overrides the mesh declarations completely:
        declarative ports (``PortWaveguide`` / ``PortAnalytical``)
        and/or specs (``PortSpecCoax`` / ``PortSpecRectWG`` /
        ``PortSpecNumerical`` / ``PortSpecMultiConductor`` /
        ``PortSpecLumped``), freely mixed.  Declarative ports are
        resolved into concrete specs at construction time
        (``self.ports`` holds the resolved list).  Names must be
        unique.
    elements : list of LumpedElement, optional
        Passive lumped circuit elements.  ``None`` (default)
        uses the declarative elements the mesh carries (declared
        before meshing via
        :meth:`~magnelio.geo.GeometryModel.add_element`); passing
        ``elements=`` overrides them completely.  Elements act on the
        fields as pure loads: they are never excited, never recorded,
        and do not appear in the S-matrix.  Their labels share the
        port-label namespace.
    f_max : float
        Upper band edge of the analysis [Hz].  Single frequency source:
        sizes the default frequency axis (see ``f_axis``), sets the
        mode-calculation frequency of the modal port builder, and
        parameterises the default excitation waveform.
    f_min : float, default 0.0
        Lower band edge [Hz].  The default frequency axis never starts
        below ``f_max / n_freq`` (power waves are undefined at DC), so
        ``f_min = 0`` yields a first point at ``f_max / n_freq``.
        On the band pipeline set a real lower edge: the pulsed band
        drive needs spectral roll-off room *below* the first axis
        point, so an axis reaching toward DC forces an
        ``O(1/f_axis[0])``-long pulse — the auto-sizing raises with
        the recommended value instead of hanging in the kernel build
        (see ``_band_setup``).
    n_freq : int, default 201
        Number of points on the default frequency axis.
    excitation : ExcitationSpec, optional
        Source waveform applied to every excited ``(port, mode)``.
        Default (``None``): derived *per excited mode* — the lower band
        edge is ``max(f_cutoff, f_min)`` with ``f_cutoff`` the excited
        mode's cut-off frequency (zero for TEM and lumped ports).  A
        zero lower edge yields a DC-inclusive ``gaussian``; a positive
        one a band-limited ``modulated_gaussian`` over
        ``[max(f_cutoff, f_min), f_max]``, which keeps the pulse energy
        of TE/TM excitations above cut-off.  Raises at run time when
        ``f_max`` does not exceed the excited mode's cut-off.
        Per-spec ``excitation`` fields are ignored.
    monitors : iterable, default ()
        Field monitors forwarded to every ``FITTimeDomainSolver`` run.
    verbose : bool, default True
        Print solver progress.
    port_model : {"modal", "band", "auto"}, default "modal"
        Which port pipeline terminates and measures the run.

        ``"modal"`` (production default) — one modal
        port operator per spec: exact DTBC on
        every certified uniform chain (homogeneous TEM, numerical
        TE/TM — port floors −124…−166 dB), modal Mur-1st on the
        rest.  On a transversally *inhomogeneous* line (microstrip,
        layered substrates) the QTEM fundamental fails the
        uniform-chain certificate and falls back to Mur — measured
        −26…−39 dB ``|S11|`` on a realistic shielded microstrip at
        ``|S21|`` errors below 0.01 dB.  Fast (seconds), full
        time-domain power-wave access, no lower-band-edge
        restriction; the accepted trade for routine QTEM work.
        A verbose notice
        names the Mur-fallback channels.

        ``"band"`` — the broadband band-subspace DTBC pipeline:
        the mode family is
        tracked over the analysis band, one operator terminates the
        whole band reflection-free (measured microstrip floors
        −171…−211 dB), and one pulsed record per excitation is
        decomposed per frequency with the true mode at every axis
        point.  Requires every port to be a
        ``PortSpecMultiConductor``, a strictly positive band
        (``f_min``!), and pays a kernel-build phase plus a longer
        record — for reflection-critical work, not the routine
        default.  ``result.a()/b()`` are unavailable on band
        results.

        ``"auto"`` — build the modal operators and inspect their
        termination certificates: certified-everywhere runs use the
        modal pipeline, any Mur-fallback multi-conductor channel
        switches the run to the band pipeline.  Mixing an
        uncertified multi-conductor port with non-multi-conductor
        specs raises.
    band_options : dict, optional
        Expert overrides for the band pipeline.  Recognised keys:
        ``f_band`` ((f_lo, f_hi) [Hz] subspace band; default derived
        from the frequency axis with 25 % roll-off guard),
        ``n_grid`` (mode-family tracking points, default 25),
        ``n_syn`` (synthesis-window steps; default auto-sized from
        the roll-off widths and doubled on a compactness failure),
        ``n_kernel_init`` (initial ghost-kernel length; default the
        next power of two above the planned run),
        ``svd_tol`` (subspace-rank threshold, default 1e-8),
        ``skirt`` (spectral amplitude at the band edges, 1e-7).
    project : str or Path, optional
        When set, ``run()`` writes the model and each excitation's
        results into this project directory and returns a
        read-only :class:`~magnelio.io.project.Project` reader instead of
        an in-memory :class:`ScatteringTDResult`.  Pointing at an
        existing project *adds* the new excitations as runs (fill-in)
        without rewriting the model.  Modal pipeline only in this
        version (band + project is a follow-up).
    geometry : GeometryModel, optional
        Source geometry to persist (exact BREP + tessellated STL) when
        ``project`` is set.  Optional — the mesh alone suffices for
        post-processing; geometry adds ParaView overlays and re-meshing.
    backend : {"auto", "numpy", "cupy"}, default "auto"
        Array backend of every FIT-TD run.  ``"auto"`` uses the GPU
        (CuPy) when CuPy and a CUDA device are available and falls back
        to the NumPy CPU backend with a one-time notice; ``"cupy"``
        requires the GPU; ``"numpy"`` forces the CPU.  Not persisted in
        the project recipe — a resumed run re-resolves ``"auto"`` on
        the machine it runs on.
    precision : {"single", "double", None}, default None
        Scalar precision of the FIT-TD time loop (fields + update
        coefficients).  ``None`` resolves to the ``MAGNELIO_PRECISION``
        environment variable else ``"single"`` (float32), the production
        default — it matches commercial FIT/FDTD tools, halves memory and
        lifts GPU throughput on consumer FP64-crippled cards, and its
        ~1e-7 field floor sits three-to-four orders of magnitude below the
        discretisation error.  ``"double"`` (float64) is the opt-in for
        high-Q (Q ≳ 1e4-1e5) or high-dynamic-range studies.  Orthogonal to
        ``backend``.  The DFT/S-parameter accumulators, the modal-port
        solve and the geometry pipeline stay double regardless.
        Persisted in the project recipe so a resumed run keeps its
        precision.
    wall_model : {"perturbative", "sibc"}, default "perturbative"
        Conductor-loss model of the TD runs.

        ``"perturbative"`` (default) — walls are lossless PEC in the
        field solve; conductor loss is evaluated after the fact by
        ``wall_loss_Q`` / :class:`MonitorWallLoss`.
        Nothing changes against earlier versions.

        ``"sibc"`` — broadband Leontovich surface impedance in the
        update itself: every conductor wall (lossy-metal solids with
        their own σ/µ/roughness; plain-PEC solids and PEC boundary
        walls with the ``wall_sigma`` override) damps the field
        through a passive rational ``Z_s(ω)`` fit over the analysis
        band ``[f_axis[0], f_max]`` — S-parameters become lossy and
        self-consistent (loaded Q, attenuation in ``|S21|``).  Requires
        lossy-metal materials and/or ``wall_sigma``; raises loudly
        otherwise.  Port planes stay lossless (modes of the lossless
        cross-section — a lossy line's α shows up in S21 over length,
        not in the port model); faces hosting ports carry no wall.
        ``AnalysisEigenmode`` keeps the perturbative route (non-goal).
    wall_sigma : float, optional
        Conductivity [S/m] for SIBC walls that are not lossy metals
        (plain-PEC solids and PEC boundary-condition walls) — the same
        override rule as ``wall_loss_Q`` / ``MonitorWallLoss``.
        Lossy-metal solids always use their own material values.
    wall_mu : float, default 1.0
        Relative permeability accompanying ``wall_sigma``.
    wall_roughness : SurfaceRoughness, optional
        Roughness model for the same walls ``wall_sigma``
        applies to; enters the SIBC as the causally completed
        ``K(f)·R_s`` fit.
    """

    # Design: SIBC_PLAN WP-D5 (wall_model="sibc"), WP-D2 (causal K(f)·R_s fit).

    mesh: Mesh
    f_max: float
    ports: Sequence[PortSpec] | None = None
    elements: Sequence | None = None
    f_min: float = 0.0
    n_freq: int = 201
    excitation: ExcitationSpec | None = None
    monitors: tuple = field(default_factory=tuple)
    verbose: bool = True
    port_model: str = "modal"
    band_options: dict | None = None
    project: object | None = None
    geometry: object | None = None
    backend: str = "auto"
    precision: str | None = None
    wall_model: str = "perturbative"
    wall_sigma: float | None = None
    wall_mu: float = 1.0
    wall_roughness: object = None
    params: dict | None = None

    _BAND_OPTION_KEYS = frozenset(
        {
            "f_band",
            "n_grid",
            "n_syn",
            "n_kernel_init",
            "svd_tol",
            "skirt",
        }
    )

    # Auto-sizing budget for the band synthesis window; beyond it the
    # kernel build (4·n_kernel single-threaded contour-QZ solves) and
    # the record length leave the interactive regime, and _band_setup
    # raises with the recommended f_min instead of hanging.  Explicit
    # band_options["n_syn"] bypasses the gate.
    _BAND_AUTO_N_SYN_MAX = 131072

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

    def __post_init__(self) -> None:
        self._mur_notice_printed = False
        if self.f_max <= 0.0:
            raise ValueError(f"f_max must be positive; got {self.f_max}")
        if not 0.0 <= self.f_min < self.f_max:
            raise ValueError(
                f"f_min must satisfy 0 <= f_min < f_max; "
                f"got f_min={self.f_min}, f_max={self.f_max}",
            )
        if self.n_freq < 2:
            raise ValueError(f"n_freq must be >= 2; got {self.n_freq}")
        if self.port_model not in ("auto", "modal", "band"):
            raise ValueError(
                f"port_model must be 'auto', 'modal' or 'band'; got {self.port_model!r}",
            )
        if self.backend not in ("auto", "numpy", "cupy"):
            raise ValueError(
                f"backend must be 'auto', 'numpy' or 'cupy'; got {self.backend!r}",
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
        if self.band_options is not None:
            unknown = set(self.band_options) - self._BAND_OPTION_KEYS
            if unknown:
                raise ValueError(
                    f"unknown band_options keys {sorted(unknown)}; "
                    f"recognised: {sorted(self._BAND_OPTION_KEYS)}",
                )
        if self.ports is None:
            # DD-109: ports declared on the GeometryModel travel with
            # the mesh; the analysis picks them up when no ports= of
            # its own is given.
            self.ports = list(self.mesh.ports)
            if not self.ports:
                raise ValueError(
                    "no ports: declare them before meshing via "
                    "GeometryModel.add_port(...) (they travel with the "
                    "mesh), or pass ports= to the analysis",
                )
        if not self.ports:
            raise ValueError("ports must be non-empty")
        labels = [self._spec_label(s) for s in self.ports]
        if len(set(labels)) != len(labels):
            raise ValueError(f"port names must be unique; got {labels}")
        for s in self.ports:
            if not isinstance(s, self._SUPPORTED_SPEC_TYPES):
                raise TypeError(
                    f"AnalysisScatteringTD does not support spec type "
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
        all_labels = labels + [e.name for e in self.elements]
        if len(set(all_labels)) != len(all_labels):
            raise ValueError(
                f"port and element names must be unique together; got {all_labels}",
            )

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

    @property
    def f_axis(self) -> np.ndarray:
        """Default frequency axis [Hz] derived from ``f_max``/``f_min``/``n_freq``.

        ``linspace(max(f_min, f_max/n_freq), f_max, n_freq)`` — the
        lower bound keeps the axis strictly positive because power
        waves (and hence S-parameters) are undefined at DC.
        """
        f_start = max(self.f_min, self.f_max / self.n_freq)
        return np.linspace(f_start, self.f_max, self.n_freq)

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
        m_eps = build_M_eps(self.mesh)
        m_mu = build_M_mu(self.mesh)
        # dt only parameterises the operator's Mur coefficients, which
        # the report does not expose — the spectral value keeps the
        # construction identical to run() (the measured lambda_max is
        # cached on the mesh, so run() pays no second eigensolve).
        dt = spectral_dt(self.mesh, "normal", m_eps=m_eps, m_mu=m_mu)
        return {
            spec.name: PortReport.from_operator(
                self._build_operator(spec, m_eps, m_mu, dt, self.f_max),
                mesh=self.mesh,
            )
            for spec in self.ports
        }

    def run(
        self,
        f_axis: np.ndarray | None = None,
        excited: Iterable[ExcitedSpec] | None = None,
        accuracy: str = "normal",
        energy_stop_db: float | None = 70.0,
        total_time_steps: int | None = None,
        taper_signals: bool = False,
        checkpoint_interval: int | None = None,
        port_signal_stop_db: float | str | None = "auto",
        max_time_steps: int | str | None = "auto",
    ) -> ScatteringTDResult:
        """Run one FIT-TD simulation per excited (port, mode) and merge.

        Parameters
        ----------
        f_axis : np.ndarray, optional
            Target frequencies [Hz], shape ``(Nf,)``.  Strictly positive.
            Default: the constructor-derived axis (see the ``f_axis``
            property).
        excited : iterable, optional
            ``[(port_name, mode_idx), ...]`` or a list of bare
            ``port_name`` strings (mode 0 implied).  Default: the first
            port at mode 0.
        accuracy : {"draft", "normal", "high"}, default "normal"
            Courant safety factor.
        energy_stop_db : float, default 70.0
            Stop each TD run when stored EM energy has decayed by this
            many dB below peak.  Calibrated against the well-absorbed
            TEM-line case: at 70 dB the truncation residual on V/I
            falls below ~7e-4 of peak, which keeps the rectangular-DFT
            sidelobes on ``|S21|`` below ~0.02 dB; the port floors
            themselves sit at the DTBC level (−130 dB class on
            certified lines).  More aggressive cuts
            (40 dB) leave a residual of ~1 % that produces
            ``|S21| > 1`` artefacts until either the run is extended
            or the rectangular window is replaced (see
            ``taper_signals``).  Set to ``None`` to disable the early
            stop; the run is then bounded by ``total_time_steps`` (an
            explicit value, or the auto-sized estimate as a fallback cap
            when both are left open).  Not applicable on the band
            pipeline (see Notes).
        total_time_steps : int, optional
            Exact leapfrog step count.  Default (``None``): the run is
            **unbounded** and marches until a stop criterion fires
            (``energy_stop_db`` or ``port_signal_stop_db``, whichever
            first), backstopped only by the generous ``max_time_steps``
            runtime cap — the auto-sized step estimate sets the
            energy-check cadence, never the length, so a slowly decaying
            structure keeps resolving instead of being cut off at a
            heuristic timeout.  Pass an explicit value to force a
            fixed-length run (it also disables the runtime cap and the
            stall watchdog: an explicit length is a user decision).
            The check cadence is unchanged from the historical bounded
            default, so a well-absorbed run stops at the very same step
            (its S-parameters are unchanged).
        taper_signals : bool, default False
            Apply a symmetric Tukey window (``alpha = 0.05``) to every
            recorded V/I time-series before the S-parameter DFT.
            Suppresses the rectangular-window sidelobes caused by a
            non-vanishing residual at the truncation edge — set to
            ``True`` if the default rectangular DFT shows ``|S|>1`` ripple
            on a passive structure and you do not want to push
            ``energy_stop_db`` higher.  On the project-store path the
            flag is recorded per run and the reader applies the same
            window when it derives the S-parameters.
        port_signal_stop_db : float, None or "auto", default "auto"
            Stop criterion: end each TD run when the modal-port
            ``|V|`` envelope has decayed by this many dB below its run peak.
            The robust termination for shielded lossless structures
            whose stored energy plateaus on TM-cut-off cavity content
            that no port can reach (``energy_stop_db`` then never
            fires); the S-parameters depend only on the port signals
            this criterion watches.  Whichever criterion fires first
            ends the run.  ``"auto"`` (default) resolves to
            60 dB when at least one modal port is present and to
            disabled on lumped-only runs (the criterion needs a modal
            ``|V|`` envelope to watch); ``None`` disables it explicitly.
            The criterion only arms once the auto-sized step estimate
            is reached, so it cannot fire in the quiet gap
            before the response reaches the far ports — runs that stop
            on the energy criterion earlier are untouched by it.
            Band-edge (cut-off) ring-down can hold the envelope at a
            plateau just above the threshold — it decays algebraically,
            so the threshold is then unreachable; a stall watchdog
            detects this (envelope slope provably too flat to reach the
            threshold before ``max_time_steps``), accepts the plateau as
            the effective floor, and stops with a ``RuntimeWarning``
            stating the achieved level.  Not applicable on the band
            pipeline.
        max_time_steps : int, None or "auto", default "auto"
            Runtime cap for unbounded runs: if no stop criterion has
            fired by this step, the run ends with a ``RuntimeWarning``
            (results truncated, resumable on the project-store path).
            ``"auto"`` sizes the cap at 40× the auto step estimate —
            roughly 10³ diagonal transits, far beyond any converging
            run; in ring-down terms it accommodates a loaded Q of
            about ``900 · (structure size / wavelength)`` before a
            60-dB decay is cut short.  ``None`` removes the cap (march
            forever — watch it live or resume; also disables the stall
            watchdog, which projects against this cap).  Ignored when
            ``total_time_steps`` is set.
        checkpoint_interval : int, optional
            Only on the project-store path (``project=``): minimum number
            of leapfrog steps between periodic resume checkpoints
            (``runs/<name>/checkpoint.h5``), overwritten via temp+rename
            so a live reader never sees a partial file.
            Default: about an eighth of the auto-sized run length (≈ eight
            checkpoints).  A final checkpoint is always written on normal
            completion (enables run-longer) and on a Ctrl-C graceful
            abort (enables resume).  Ignored on the in-RAM path.

        Returns
        -------
        ScatteringTDResult
            Wrapper carrying the merged S-matrix (``.s_params``,
            single-column for one excited pair, K-column for K), the
            per-excitation V/I time series (``.signals``), and the
            sampled reference waveform (``.reference_signal``).  Use
            ``result.S("p_out", "p_in")`` for direct S-parameter
            access.

        Notes
        -----
        When the run resolves to the band pipeline (``port_model``,
        constructor docstring) the record is a fixed-length pulsed
        broadband run: ``energy_stop_db`` and ``taper_signals`` do
        not apply (the band decomposition needs the complete
        synthesis window plus ring-down and DFTs the rectangular
        record; a truncation-quality warning fires if the record has
        not rung down), ``total_time_steps`` remains the manual
        override for the record length, and the built band ports are
        reused across excitations (state reset, kernels kept).
        """
        f_axis = self.f_axis if f_axis is None else np.asarray(f_axis, dtype=float)

        excited_list = self._resolve_excited(excited)

        if isinstance(port_signal_stop_db, str):
            if port_signal_stop_db != "auto":
                raise ValueError(
                    f"port_signal_stop_db must be a float [dB], None or "
                    f"'auto'; got {port_signal_stop_db!r}",
                )
            # DD-114: on by default wherever a modal port can feed the
            # |V|-envelope criterion; a lumped-only run has no modal
            # envelope to watch and keeps the energy criterion alone.
            port_signal_stop_db = (
                60.0 if any(not isinstance(s, PortSpecLumped) for s in self.ports) else None
            )

        if isinstance(max_time_steps, str) and max_time_steps != "auto":
            raise ValueError(
                f"max_time_steps must be an int (steps), None or 'auto'; got {max_time_steps!r}",
            )

        # SIBC runs report wall loss through the operator's own
        # accounting — point the monitors at the spec before any solver
        # attaches them (WP-D5; a no-op on the perturbative default).
        self._wire_wall_monitors()

        bc_objects = self._resolve_bc()

        m_eps = build_M_eps(self.mesh)
        m_mu = build_M_mu(self.mesh)
        dt = spectral_dt(self.mesh, accuracy, m_eps=m_eps, m_mu=m_mu)

        if self._resolve_port_model(m_eps, m_mu, dt) == "band":
            if self.project is not None:
                # Streaming wired for the modal pipeline in WP-S2; band is a
                # follow-up.
                raise NotImplementedError(
                    "the project store does not support the band pipeline "
                    "yet; use port_model='modal' or omit project=",
                )
            return self._run_band(
                excited_list=excited_list,
                f_axis=f_axis,
                m_eps=m_eps,
                m_mu=m_mu,
                dt=dt,
                total_time_steps=total_time_steps,
                bc_objects=bc_objects,
            )

        if self.project is not None:
            return self._run_streamed(
                excited_list,
                m_eps,
                m_mu,
                dt,
                bc_objects,
                f_axis,
                total_time_steps,
                energy_stop_db,
                checkpoint_interval,
                port_signal_stop_db=port_signal_stop_db,
                taper_signals=taper_signals,
                max_time_steps=max_time_steps,
            )

        per_excitation: list[tuple[SParameterResult, dict, Signal1D, int, dict, dict, dict]] = []
        for excited_chan in excited_list:
            # excitation / step count resolve inside per excitation:
            # the auto-waveform depends on the excited mode's cut-off,
            # which is only known once the port operator is built.
            per_excitation.append(
                self._run_one_excitation(
                    excited_chan=excited_chan,
                    excitation=self.excitation,
                    m_eps=m_eps,
                    m_mu=m_mu,
                    dt=dt,
                    total_time_steps=total_time_steps,
                    bc_objects=bc_objects,
                    f_calc=self.f_max,
                    f_axis=f_axis,
                    energy_stop_db=energy_stop_db,
                    taper_signals=taper_signals,
                    port_signal_stop_db=port_signal_stop_db,
                    max_time_steps=max_time_steps,
                ),
            )

        # port_modes / de-stagger parameters are identical across runs
        # (same builders on the same mesh); take them from the first.
        port_modes = per_excitation[0][4]
        port_normal_dx = per_excitation[0][5]
        port_line_params = per_excitation[0][6]
        if len(per_excitation) == 1:
            s_params, signals, ref_sig, n_actual = per_excitation[0][:4]
            signals_by_excitation = {excited_list[0]: signals}
        else:
            s_params = SParameterResult.merge(
                [item[0] for item in per_excitation],
            )
            signals_by_excitation = {
                excited_list[k]: per_excitation[k][1] for k in range(len(per_excitation))
            }
            # reference_signal is single-valued on the result; pick the
            # longest run's.  With auto-derived excitations the waveform
            # can differ across excited modes (cut-off-dependent band) —
            # the S-parameters are unaffected (each run normalises
            # against its own reference internally), but treat this
            # signal as representative only for the longest run.
            longest = max(per_excitation, key=lambda it: it[3])
            _, _, ref_sig, n_actual = longest[:4]

        return ScatteringTDResult(
            s_params=s_params,
            signals=signals_by_excitation,
            reference_signal=ref_sig,
            dt=dt,
            n_actual_steps=n_actual,
            port_modes=port_modes,
            port_normal_dx=port_normal_dx,
            port_line_params=port_line_params,
            port_model_used="modal",
            reference_signals={
                excited_list[k]: per_excitation[k][2] for k in range(len(per_excitation))
            },
            settings=self._run_settings(
                dt=dt,
                n_actual_steps=n_actual,
                accuracy=accuracy,
                energy_stop_db=energy_stop_db,
                port_signal_stop_db=port_signal_stop_db,
                taper_signals=taper_signals,
                port_model_used="modal",
            ),
        )

    def _run_settings(self, **kwargs) -> RunSettings:
        """Assemble the result-contract settings for this run."""
        return RunSettings(
            f_max=self.f_max,
            f_min=self.f_min,
            n_freq=self.n_freq,
            precision=self.precision,
            backend=self.backend,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_excitation_run(
        self,
        excited_chan: tuple[str, int],
        excitation: ExcitationSpec | None,
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
        f_calc: float,
    ) -> tuple:
        """Build the operators, excitation, recorder and static per-run
        metadata shared by the in-RAM and streaming modal paths.

        Returns ``(operators, element_ops, waveform_fn, n_steps_estimate,
        recorder, port_modes, port_normal_dx, port_line_params)``.
        ``n_steps_estimate`` is the auto-sized run length
        (:meth:`_estimate_steps`); the actual loop bound + energy-check
        cadence are resolved per path by :meth:`_resolve_runtime` (the
        default is unbounded, energy-gated).  The port metadata depends
        only on the operators (not on the time-marching), so the
        streaming sink can declare the run's HDF5 layout before step 0.
        ``element_ops`` (DD-123) join the solver's operator list but
        stay out of the recorder and all port metadata.
        """
        # Build all operators fresh per excitation — clean Mur state on
        # both ports, and per-mode source-history buffer freshly sized.
        operators = [self._build_operator(spec, m_eps, m_mu, dt, f_calc) for spec in self.ports]
        # Passive elements rebuild per excitation too: fresh companion
        # (L/C history) state, exactly like a port's internal element.
        element_ops = [
            build_lumped_element(e, self.mesh, m_eps, m_mu, dt=dt) for e in self.elements
        ]

        if self.verbose and not self._mur_notice_printed:
            mur_channels = [
                (op.name, m)
                for op in operators
                for m, kind in enumerate(getattr(op, "termination_kinds", []))
                if kind == "mur"
            ]
            if mur_channels:
                print(
                    f"[AnalysisScatteringTD] channels {mur_channels} "
                    f"run the modal Mur-1st fallback (−30 dB-class "
                    f"|S11| on inhomogeneous lines; DD-064 accepted "
                    f"default).  port_model='band' provides the "
                    f"reflection-free pipeline.",
                )
            self._mur_notice_printed = True

        label_to_op = {op.name: op for op in operators}
        excited_op = label_to_op[excited_chan[0]]
        excitation = self._resolve_excitation(
            excitation,
            excited_op,
            excited_chan[1],
        )
        n_steps_estimate = self._estimate_steps(self.mesh.grid, excitation, dt)
        for op in operators:
            op.clear_excitation()
        waveform_fn = excitation.build_waveform()
        # DD-155: a port cut by symmetry planes injects ×1/√2 per plane
        # so the declared waveform amplitude is a full-model √W — the
        # meshed half then carries exactly half the full-model power and
        # the fields sit at full-model level.  The scale must act
        # exactly once: reference_signal keeps sampling the *unscaled*
        # waveform_fn (the full-model reference the monitors renormalise
        # against); the recorder restores ×√2 per plane on the read side.
        exc_scale = _excitation_scale(excited_op)
        if exc_scale != 1.0:
            excited_op.set_excitation(
                excited_chan[1],
                lambda t: exc_scale * waveform_fn(t),
            )
        else:
            excited_op.set_excitation(excited_chan[1], waveform_fn)

        recorder = PortSignalRecorder(dt=dt, ports=operators)
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
        return (
            operators,
            element_ops,
            waveform_fn,
            n_steps_estimate,
            recorder,
            port_modes,
            port_normal_dx,
            port_line_params,
        )

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

    def _run_one_excitation(
        self,
        excited_chan: tuple[str, int],
        excitation: ExcitationSpec | None,
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
        total_time_steps: int | None,
        bc_objects: dict,
        f_calc: float,
        f_axis: np.ndarray,
        energy_stop_db: float | None,
        taper_signals: bool,
        port_signal_stop_db: float | None = None,
        max_time_steps: int | str | None = "auto",
    ) -> tuple:
        """In-RAM modal run: solve, then compute this column's S-parameters."""
        (
            operators,
            element_ops,
            waveform_fn,
            n_steps_estimate,
            recorder,
            port_modes,
            port_normal_dx,
            port_line_params,
        ) = self._prepare_excitation_run(
            excited_chan,
            excitation,
            m_eps,
            m_mu,
            dt,
            f_calc,
        )
        solver_steps, check_interval = self._resolve_runtime(
            total_time_steps,
            energy_stop_db,
            n_steps_estimate,
            port_signal_stop_db=port_signal_stop_db,
        )
        solver = FITTimeDomainSolver(
            mesh=self.mesh,
            boundary_conditions=bc_objects,
            ports=operators + element_ops,
            recorder=recorder,
            total_time_steps=solver_steps,
            energy_check_interval=check_interval,
            dt=dt,
            energy_stop_db=energy_stop_db,
            port_signal_stop_db=port_signal_stop_db,
            port_signal_min_steps=n_steps_estimate,
            max_time_steps=self._resolve_cap(max_time_steps, total_time_steps, n_steps_estimate),
            verbose=self.verbose,
            monitors=list(self.monitors),
            backend=self.backend,
            precision=self.precision,
            sibc=self._sibc_spec(),
        )
        solver.run()

        n_actual = solver._actual_steps or n_steps_estimate
        signals = recorder.finalize(n_steps_actual=n_actual)

        reference_signal = _sampled_excitation(waveform_fn, n_actual, dt)
        _renormalize_freq_monitors(self.monitors, reference_signal)

        s_dict = compute_s_parameters(
            recorder_signals=signals,
            port_modes=port_modes,
            excited=excited_chan,
            reference_signal=reference_signal,
            f_axis=f_axis,
            taper_signals=taper_signals,
            port_normal_dx=port_normal_dx,
            port_line_params=port_line_params,
        )
        s_params = SParameterResult.from_single_excitation(
            s_dict,
            excited_chan,
            f_axis,
        )
        self._wire_far_field_monitors(s_dict, f_axis)
        energy_trace = getattr(solver, "_energy_trace", None)
        return (
            s_params,
            signals,
            reference_signal,
            n_actual,
            port_modes,
            port_normal_dx,
            port_line_params,
            energy_trace,
        )

    def _run_streamed(
        self,
        excited_list: list[tuple[str, int]],
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
        bc_objects: dict,
        f_axis: np.ndarray,
        total_time_steps: int | None,
        energy_stop_db: float | None,
        checkpoint_interval: int | None,
        port_signal_stop_db: float | None = None,
        taper_signals: bool = False,
        max_time_steps: int | str | None = "auto",
    ) -> object:
        """Stream one FIT-TD run per excitation into the project store.

        Writes the shared model once, then runs each excitation with a
        live :class:`~magnelio.io.project._ScatteringRunSink` attached to
        the solver: V/I, the reference waveform and the energy trace are
        flushed to disk (HDF5-SWMR) as the run proceeds, so a separate
        post-processing script can follow it live.  Each
        run also writes a periodic resume ``checkpoint.h5`` and, on Ctrl-C,
        stops gracefully at a consistent step and checkpoints (WP-S7).
        Pointing ``project`` at an existing directory adds the new runs
        (fill-in) without rewriting the model.  Returns a read-only
        :class:`~magnelio.io.project.Project` reader; the S-matrix is
        derived on read, never stored.
        """
        from pathlib import Path  # noqa: PLC0415

        from magnelio.io.project import ProjectStore, open_project  # noqa: PLC0415

        path = Path(self.project)
        setup = {
            "analysis": "AnalysisScatteringTD",
            "f_max": float(self.f_max),
            "f_min": float(self.f_min),
            "n_freq": int(self.n_freq),
            "dt": float(dt),
            "port_names": [self._spec_label(s) for s in self.ports],
            "port_model": "modal",
            # Reconstruction recipe (DD-070, WP-S8): the resolved port
            # specs + boundary closure + excitation, so resume() rebuilds
            # the exact same operators from the store (path-only API).
            "recipe": build_scattering_recipe(self),
            "params": dict(self.params or {}),
        }
        if (path / "project.json").exists():
            store = ProjectStore(path)  # fill-in: keep the model
        else:
            store = ProjectStore.create(
                path,
                self.mesh,
                geometry=self.geometry,
                setup=setup,
            )

        # Pre-register every planned excitation as ``pending`` so the
        # project status cannot flicker to "done" in the gap between
        # sequential runs — a live watcher polling ``status`` sees
        # "running" until the *last* planned run finishes.
        store.register_planned_runs((_scattering_run_name(chan), chan) for chan in excited_list)

        # DD-070 streams MonitorFieldTime + MonitorFluxTime +
        # MonitorFieldFrequency; warn if a real magnelio data monitor of
        # another kind is present, so its absence from the reader is not a
        # silent surprise (test control monitors are not magnelio monitors, so
        # they do not trip this).
        from magnelio.monitors.far_field import MonitorFarField  # noqa: PLC0415
        from magnelio.monitors.field_frequency import (  # noqa: PLC0415
            MonitorFieldFrequency,
        )
        from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
        from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

        _streamed = (
            MonitorFieldTime,
            MonitorFluxTime,
            MonitorFieldFrequency,
            MonitorFarField,
        )
        not_streamed = sorted(
            {
                type(m).__name__
                for m in self.monitors
                if not isinstance(m, _streamed)
                and type(m).__module__.startswith("magnelio.monitors")
            }
        )
        if not_streamed and self.verbose:
            print(
                f"[AnalysisScatteringTD] monitor type(s) {not_streamed} are "
                f"not yet streamed to the project store (DD-070); they run in "
                f"RAM on this pass but are absent from the reader and from a "
                f"resume.",
            )

        for excited_chan in excited_list:
            (
                operators,
                element_ops,
                waveform_fn,
                n_steps_estimate,
                recorder,
                port_modes,
                port_normal_dx,
                port_line_params,
            ) = self._prepare_excitation_run(
                excited_chan,
                self.excitation,
                m_eps,
                m_mu,
                dt,
                self.f_max,
            )
            # Same runtime resolution as the in-RAM path (default:
            # unbounded, energy-gated; the estimate only sets the check
            # cadence), so `project=` never changes the answer — the
            # energy-stop step is identical.
            solver_steps, check_interval = self._resolve_runtime(
                total_time_steps,
                energy_stop_db,
                n_steps_estimate,
                port_signal_stop_db=port_signal_stop_db,
            )
            ckpt_interval = (
                checkpoint_interval
                if checkpoint_interval is not None
                else max(1, n_steps_estimate // 8)
            )
            # The SAME monitor objects go to the sink (which drains their
            # snapshots to disk) and the solver (which records into them) —
            # a shared reference, so the drained data is exactly what the
            # solver wrote this run (WP-S9).
            run_monitors = list(self.monitors)
            sink = store.open_scattering_run(
                _scattering_run_name(excited_chan),
                excited=excited_chan,
                dt=dt,
                f_axis=f_axis,
                channels=recorder.channels,
                port_modes=port_modes,
                port_normal_dx=port_normal_dx,
                port_line_params=port_line_params,
                waveform_fn=waveform_fn,
                recorder=recorder,
                port_model="modal",
                energy_stop_db=energy_stop_db,
                port_signal_stop_db=port_signal_stop_db,
                total_time_steps=total_time_steps,
                taper_signals=taper_signals,
                monitors=run_monitors,
                grid=self.mesh.grid,
            )
            solver = FITTimeDomainSolver(
                mesh=self.mesh,
                boundary_conditions=bc_objects,
                ports=operators + element_ops,
                recorder=recorder,
                total_time_steps=solver_steps,
                energy_check_interval=check_interval,
                dt=dt,
                energy_stop_db=energy_stop_db,
                port_signal_stop_db=port_signal_stop_db,
                port_signal_min_steps=n_steps_estimate,
                max_time_steps=self._resolve_cap(
                    max_time_steps, total_time_steps, n_steps_estimate
                ),
                verbose=self.verbose,
                monitors=run_monitors,
                sink=sink,
                backend=self.backend,
                precision=self.precision,
                sibc=self._sibc_spec(),
            )
            sink.enable_checkpoints(solver.state_dict, ckpt_interval)
            self._drive_streamed_solver(solver, sink, excited_chan, path)
            # The caller keeps these monitor objects (the sink drained
            # copies to disk), so they get the run's excitation too.
            _renormalize_freq_monitors(
                run_monitors,
                _sampled_excitation(waveform_fn, solver._actual_steps or solver_steps, dt),
            )

        if self.verbose:
            print(
                f"[AnalysisScatteringTD] streamed {len(excited_list)} run(s) to project {path}",
            )
        return open_project(path)

    def _drive_streamed_solver(self, solver, sink, excited_chan, path) -> None:
        """Run a sink-attached solver under a cooperative Ctrl-C trap (WP-S7).

        Shared by the first-run (:meth:`_run_streamed`) and the resume
        path.  Traps ``SIGINT`` on the main thread into
        :meth:`FITTimeDomainSolver.request_stop` so a Ctrl-C finishes the
        in-flight step and checkpoints a consistent, resumable state
        instead of tearing the run down mid-step; the previous handler is
        restored in ``finally``.  Also traps ``SIGUSR1`` (POSIX) into
        :meth:`_ScatteringRunSink.request_checkpoint` — a snapshot-and-
        continue signal that writes a resume checkpoint at the next flush
        *without* stopping the march (send ``kill -USR1 <pid>``).  Finalises
        the run ``done`` on normal
        completion (writes the run-longer checkpoint) or ``aborted`` on a
        graceful stop, re-raising ``KeyboardInterrupt`` in the latter case
        (Ctrl-C still stops the program — but now leaves a resumable
        project behind).
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
            sink.close(state="aborted")
            raise
        finally:
            if trap:
                signal.signal(signal.SIGINT, prev_int)
                if has_usr1:
                    signal.signal(signal.SIGUSR1, prev_usr1)

        if solver._aborted:
            sink.close(state="aborted", stop_reason="aborted")
            print(
                f"\n[AnalysisScatteringTD] run "
                f"{excited_chan[0]}_mode{excited_chan[1]} aborted at "
                f"step {solver._resume_step}; resume checkpoint saved "
                f"to {path}",
            )
            raise KeyboardInterrupt(
                f"scattering run aborted at step {solver._resume_step}; resume from {path}",
            )
        # Book why the run ended (and the achieved |V| level) into the
        # run index — the honest provenance of the derived S-parameters,
        # resume-safe like the launch criteria (DD-122).
        sink.close(
            state="done",
            stop_reason=solver._stop_reason,
            final_port_signal_db=solver._final_signal_db,
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

        Field monitors are *not* part of the recipe and are not restored
        (their data lives in the memory-efficient monitor write-through
        streams instead); they do not affect the leapfrog update, so V/I
        and S-parameters are unchanged.
        """
        # Design: WP-S9 (monitor write-through), WP-S8 (reconstruction recipe).
        from magnelio.io.project import Project, open_project  # noqa: PLC0415

        proj = project if isinstance(project, Project) else open_project(project)
        recipe = proj.setup.get("recipe")
        if recipe is None:
            raise ValueError(
                f"project {proj.path} carries no reconstruction recipe; it "
                f"was written by an older magnelio (or not by "
                f"AnalysisScatteringTD) "
                f"and cannot be rebuilt for resume.  Re-run it with a current "
                f"magnelio to enable resume.",
            )
        return cls(mesh=proj.mesh, verbose=verbose, **recipe_kwargs(recipe))

    def _resolve_port_model(
        self,
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
    ) -> str:
        """Resolve ``port_model`` to the pipeline that runs.

        ``"auto"`` builds the modal operator of every
        ``PortSpecMultiConductor`` and reads its per-mode termination
        certificates: any recorded channel on Mur (the inhomogeneous-
        QTEM fallback) switches the run to the band pipeline — the
        production S-parameter path must not silently degrade to a
        −30 dB-class absorber.  Certified-everywhere runs keep the
        cheaper modal pipeline.
        """
        if self.port_model == "modal":
            return "modal"
        mc = [s for s in self.ports if isinstance(s, PortSpecMultiConductor)]
        all_mc = len(mc) == len(self.ports)
        if self.port_model == "band":
            if not all_mc:
                raise ValueError(
                    "port_model='band' requires every port to be a "
                    "PortSpecMultiConductor (the band decomposition "
                    "needs band DTBC ports on every face); got "
                    f"{[type(s).__name__ for s in self.ports]}",
                )
            return "band"
        # "auto": probe the multi-conductor termination certificates.
        if not mc:
            return "modal"
        mur_channels: list[tuple[str, int]] = []
        for spec in mc:
            op = build_modal_port(
                spec,
                self.mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_calc=self.f_max,
            )
            kinds = getattr(op, "termination_kinds", [])
            mur_channels.extend((spec.name, m) for m, kind in enumerate(kinds) if kind == "mur")
        if not mur_channels:
            return "modal"
        if not all_mc:
            raise ValueError(
                f"channels {mur_channels} fail the DTBC uniform-chain "
                f"certificate (inhomogeneous cross-section) and need "
                f"the band pipeline, but the port list mixes in "
                f"non-multi-conductor specs "
                f"({[type(s).__name__ for s in self.ports]}).  The "
                f"band decomposition needs band ports on every face — "
                f"assemble it from components, or pass "
                f"port_model='modal' to accept the Mur fallback "
                f"(−30 dB-class |S11| floor) on those channels.",
            )
        if self.verbose:
            print(
                f"[AnalysisScatteringTD] channels {mur_channels} are "
                f"not DTBC-certifiable (inhomogeneous cross-section); "
                f"using the band-subspace DTBC pipeline (DD-057)",
            )
        return "band"

    def _band_setup(
        self,
        f_axis: np.ndarray,
        dt: float,
        total_time_steps: int | None,
    ) -> dict:
        """Derive the band-pipeline parameters for one run.

        Defaults follow the band-pipeline acceptance benchmark: the subspace
        band pads the measurement span by 25 % on both sides (floored
        at a quarter of the lowest axis frequency — the band must
        stay positive and the QTEM fundamental propagates at any
        f > 0), the synthesis window is sized so the erfc-product
        pulse (Gaussian roll-off σ_t = 1/(2π·σ_f)) has decayed at the
        window end, and the ghost kernels are pre-sized past the
        planned record so no mid-run contour-QZ rebuild triggers.

        Cost gate: the lower roll-off room is bounded by ``f_axis[0]``
        itself, so an axis reaching toward DC (the default axis with
        ``f_min = 0`` starts at ``f_max/n_freq``) forces a pulse of
        ``O(1/f_axis[0])`` duration — at production dt this is
        hundreds of thousands of steps, and the kernel build (4·n_k
        single-threaded QZ solves) inherits it.  The auto-sizing
        raises above ``_BAND_AUTO_N_SYN_MAX`` with the recommended
        ``f_min``; explicit ``band_options["n_syn"]`` or
        ``total_time_steps`` bypass the gate deliberately.
        """
        opts = dict(self.band_options or {})
        f1 = float(f_axis[0])
        f2 = float(f_axis[-1])
        if not (0.0 < f1 < f2):
            raise ValueError(
                f"the band pipeline needs a strictly positive "
                f"frequency span; got axis [{f1:.4g}, {f2:.4g}] Hz",
            )
        width = f2 - f1
        f_band = opts.get("f_band") or (
            max(f1 - 0.25 * width, 0.25 * f1),
            f2 + 0.25 * width,
        )
        skirt = float(opts.get("skirt", 1e-7))
        n_syn = opts.get("n_syn")
        if n_syn is None:
            # Compactness pre-sizing: the narrower roll-off dominates
            # the pulse duration.  x_skirt = erfcinv(2·skirt); the
            # gate rejects tails above 1e-6 of peak in the last
            # n_syn/32 samples, so budget 13 Gaussian time constants
            # across the window (centre-to-edge 6.5 σ_t ≈ 3e-10 of
            # peak) and round up to a power of two.  A failing gate
            # still auto-doubles at excitation time.
            x_skirt = float(erfcinv(2.0 * skirt))
            sig_f = min(f1 - f_band[0], f_band[1] - f2) / (math.sqrt(2.0) * x_skirt)
            sigma_t = 1.0 / (2.0 * math.pi * sig_f)
            n_min = int(math.ceil(13.0 * sigma_t / dt))
            n_syn = max(8192, 1 << (n_min - 1).bit_length())
            if n_syn > self._BAND_AUTO_N_SYN_MAX:
                # Recommended axis start that fits the budget: invert
                # the sizing chain (gap = 0.75·f1 under the default
                # f_band) for n_min = budget.
                f1_rec = (
                    13.0
                    * math.sqrt(2.0)
                    * x_skirt
                    / (2.0 * math.pi * 0.75 * self._BAND_AUTO_N_SYN_MAX * dt)
                )
                raise ValueError(
                    f"band-pipeline auto-sizing: the frequency axis "
                    f"starts at {f1:.4g} Hz, leaving only "
                    f"{f1 - f_band[0]:.4g} Hz of spectral roll-off "
                    f"room below it — the band-limited pulse then "
                    f"needs ~{13.0 * sigma_t:.3g} s ≈ {n_min} steps "
                    f"at dt = {dt:.3g} s (auto budget "
                    f"{self._BAND_AUTO_N_SYN_MAX}), and the ghost-kernel "
                    f"build scales with it (single-threaded contour-"
                    f"QZ).  Measuring close to DC with a pulsed band "
                    f"port is inherently long.  Options: raise the "
                    f"axis start to >= {f1_rec:.3g} Hz (constructor "
                    f"f_min= or an explicit run(f_axis=...); higher "
                    f"starts shorten pulse and kernel build "
                    f"proportionally), or accept the cost "
                    f"deliberately via band_options="
                    f"{{'n_syn': {n_syn}}} (plus total_time_steps).",
                )
        n_syn = int(n_syn)

        grid = self.mesh.grid
        L_diag = math.sqrt(
            float(grid.x[-1] - grid.x[0]) ** 2
            + float(grid.y[-1] - grid.y[0]) ** 2
            + float(grid.z[-1] - grid.z[0]) ** 2
        )
        ring_down = max(
            4096,
            int(
                math.ceil(
                    8.0 * L_diag / (0.5 * C0) / dt,
                )
            ),
        )
        n_steps = int(total_time_steps) if total_time_steps is not None else n_syn + ring_down
        n_kernel = opts.get("n_kernel_init")
        if n_kernel is None:
            n_kernel = max(16384, 1 << (n_steps - 1).bit_length())
        return dict(
            f_band=(float(f_band[0]), float(f_band[1])),
            f_span=(f1, f2),
            n_grid=int(opts.get("n_grid", 25)),
            svd_tol=float(opts.get("svd_tol", 1e-8)),
            skirt=skirt,
            n_syn=n_syn,
            ring_down=ring_down,
            n_steps=n_steps,
            n_steps_user=total_time_steps is not None,
            n_kernel_init=int(n_kernel),
        )

    def _run_band(
        self,
        excited_list: list[tuple[str, int]],
        f_axis: np.ndarray,
        m_eps: np.ndarray,
        m_mu: np.ndarray,
        dt: float,
        total_time_steps: int | None,
        bc_objects: dict,
    ) -> ScatteringTDResult:
        """Band-subspace DTBC run: build once, one pulsed run per
        excitation, per-frequency true-mode decomposition.
        """
        if np.any(np.diff(f_axis) <= 0.0):
            raise ValueError("f_axis must be sorted strictly ascending")
        cfg = self._band_setup(f_axis, dt, total_time_steps)

        if self.verbose:
            print(
                f"[AnalysisScatteringTD] building "
                f"{len(self.ports)} band ports: band "
                f"{cfg['f_band'][0] / 1e9:.3g}-"
                f"{cfg['f_band'][1] / 1e9:.3g} GHz, "
                f"n_grid = {cfg['n_grid']}, n_kernel = "
                f"{cfg['n_kernel_init']} (mode-family tracking + "
                f"contour-QZ ghost kernels, single-threaded — this "
                f"is the expensive phase before the TD run)",
                flush=True,
            )
        operators = [
            build_band_dtbc_port(
                spec,
                self.mesh,
                m_eps,
                m_mu,
                dt=dt,
                f_band=cfg["f_band"],
                n_grid=cfg["n_grid"],
                svd_tol=cfg["svd_tol"],
                n_channels=spec.n_modes,
                n_kernel_init=cfg["n_kernel_init"],
            )
            for spec in self.ports
        ]
        label_to_op = {op.name: op for op in operators}
        if self.verbose:
            op0 = operators[0]
            print(
                f"[AnalysisScatteringTD] band ports built: "
                f"subspace rank p = {op0.subspace_rank}, "
                f"n_syn = {cfg['n_syn']}, run = {cfg['n_steps']} steps",
            )

        per_excitation = []
        for excited_chan in excited_list:
            for op in operators:
                op.reset_state()
            excited_op = label_to_op[excited_chan[0]]
            n_syn, ref_time = self._set_band_excitation(
                excited_op,
                excited_chan[1],
                cfg,
                dt,
            )
            n_steps = cfg["n_steps"] if cfg["n_steps_user"] else n_syn + cfg["ring_down"]

            # Passive elements (DD-123): rebuilt per excitation for a
            # fresh companion (L/C history) state; never recorded.
            element_ops = [
                build_lumped_element(e, self.mesh, m_eps, m_mu, dt=dt) for e in self.elements
            ]
            recorder = PortSignalRecorder(dt=dt, ports=operators)
            solver = FITTimeDomainSolver(
                mesh=self.mesh,
                boundary_conditions=bc_objects,
                ports=operators + element_ops,
                recorder=recorder,
                total_time_steps=n_steps,
                dt=dt,
                verbose=self.verbose,
                monitors=list(self.monitors),
                backend=self.backend,
                precision=self.precision,
                sibc=self._sibc_spec(),
            )
            solver.run()
            signals = recorder.finalize(n_steps_actual=n_steps)

            # Record-quality contract: the DFT decomposition assumes
            # complete ring-down inside the record (DD-057).
            v_exc = signals[excited_chan][0].values
            peak = float(np.abs(v_exc).max())
            tail = float(np.abs(v_exc[-256:]).max())
            if peak > 0.0 and tail > 1e-4 * peak:
                warnings.warn(
                    f"band-run record ends at {tail / peak:.1e} of "
                    f"peak on {excited_chan} (contract: < 1e-4); "
                    f"S-parameters may carry truncation ripple — "
                    f"increase total_time_steps",
                    UserWarning,
                    stacklevel=3,
                )

            s_dict = compute_band_s_parameters(
                signals,
                operators,
                excited_chan,
                f_axis,
            )
            s_params = SParameterResult.from_single_excitation(
                s_dict,
                excited_chan,
                f_axis,
            )

            t_axis = np.arange(n_steps) * dt
            ref_values = np.zeros(n_steps)
            ref_values[: min(n_syn, n_steps)] = ref_time[: min(n_syn, n_steps)]
            reference_signal = Signal1D(
                t=t_axis,
                values=ref_values,
                dt=dt,
                label="excitation",
            )
            _renormalize_freq_monitors(self.monitors, reference_signal)
            per_excitation.append(
                (s_params, signals, reference_signal, n_steps),
            )

        port_modes = {op.name: self._modes_for_operator(op) for op in operators}
        port_normal_dx = {op.name: op.plane.normal_dx for op in operators}
        if len(per_excitation) == 1:
            s_params, signals, ref_sig, n_actual = per_excitation[0]
            signals_by_excitation = {excited_list[0]: signals}
        else:
            s_params = SParameterResult.merge(
                [item[0] for item in per_excitation],
            )
            signals_by_excitation = {
                excited_list[k]: per_excitation[k][1] for k in range(len(per_excitation))
            }
            longest = max(per_excitation, key=lambda it: it[3])
            _, _, ref_sig, n_actual = longest

        return ScatteringTDResult(
            s_params=s_params,
            signals=signals_by_excitation,
            reference_signal=ref_sig,
            dt=dt,
            n_actual_steps=n_actual,
            port_modes=port_modes,
            port_normal_dx=port_normal_dx,
            port_line_params={},
            port_model_used="band",
            settings=self._run_settings(
                dt=dt,
                n_actual_steps=n_actual,
                port_model_used="band",
            ),
        )

    def _set_band_excitation(
        self,
        op,
        mode_idx: int,
        cfg: dict,
        dt: float,
    ) -> tuple[int, np.ndarray]:
        """Arm the band port's source; returns (n_syn, scalar series).

        The default drive is the flat-spectrum erfc-product band pulse
        over the measurement span; an explicit ``excitation=``
        override is sampled and launched through the frequency-tracked
        ghost source instead.  Both paths retry with a doubled
        synthesis window when the compactness gate rejects the pulse
        (fine meshes need longer windows for the same physical
        roll-off).  The returned scalar time series is the
        result's ``reference_signal`` (the band pulse has no
        closed-form waveform).
        """
        n_syn = cfg["n_syn"]
        # DD-155 full-model power semantics: inject ×1/√2 per symmetry
        # plane cutting the port window; the returned reference series
        # stays unscaled (the full-model waveform the read side and the
        # monitors normalise against).
        exc_scale = _excitation_scale(op)
        last_err: Exception | None = None
        for _ in range(4):
            try:
                if self.excitation is None:
                    op.set_excitation_band(
                        mode_idx,
                        cfg["f_span"],
                        n_syn=n_syn,
                        skirt=cfg["skirt"],
                        amplitude=exc_scale,
                    )
                    ref = np.fft.irfft(
                        band_source_spectrum(
                            cfg["f_span"],
                            op.channel_band(mode_idx),
                            dt,
                            n_syn,
                            skirt=cfg["skirt"],
                        ),
                        n=n_syn,
                    )
                else:
                    waveform_fn = self.excitation.build_waveform()
                    if exc_scale != 1.0:
                        op.set_excitation(
                            mode_idx,
                            lambda t, fn=waveform_fn: exc_scale * fn(t),
                            n_syn=n_syn,
                        )
                    else:
                        op.set_excitation(mode_idx, waveform_fn, n_syn=n_syn)
                    t = np.arange(n_syn) * dt
                    ref = np.array(
                        [waveform_fn(float(tt)) for tt in t],
                    )
                return n_syn, ref
            except ValueError as err:
                last_err = err
                n_syn *= 2
        raise ValueError(
            f"band excitation failed even at n_syn = {n_syn // 2}: {last_err}",
        ) from last_err

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

    def _resolve_excitation(
        self,
        excitation: ExcitationSpec | None,
        excited_op,
        mode_idx: int,
    ) -> ExcitationSpec:
        """Return the explicit override or derive a per-mode waveform.

        Auto rule (ports the legacy ``waveform_for_mode`` selection):
        the effective lower band edge is ``max(f_cutoff, f_min)``,
        where ``f_cutoff`` is the excited mode's cut-off frequency
        (zero for TEM modes and lumped ports).  A zero edge yields a
        DC-inclusive ``gaussian``; a positive one a band-limited
        ``modulated_gaussian`` over ``[edge, f_max]`` — for TE/TM
        modes this keeps the pulse spectrum above cut-off, where a
        DC-inclusive pulse would put ~half its energy below cut-off
        (total reflection, slow Mur-ABC ringing).
        """
        if excitation is not None:
            return excitation
        modes = self._modes_for_operator(excited_op)
        if not 0 <= mode_idx < len(modes):
            raise ValueError(
                f"excited mode index {mode_idx} out of range for port "
                f"{excited_op.name!r} with {len(modes)} mode(s)",
            )
        mode = modes[mode_idx]
        # _LumpedModeStub carries no omega_c — lumped ports are DC-capable.
        f_cutoff = getattr(mode, "omega_c", 0.0) / (2.0 * math.pi)
        eff_f_min = max(f_cutoff, self.f_min)
        if eff_f_min >= self.f_max:
            mode_label = getattr(mode, "name", f"mode {mode_idx}")
            raise ValueError(
                f"f_max = {self.f_max:.4g} Hz does not exceed the lower "
                f"band edge {eff_f_min:.4g} Hz of excited mode "
                f"{mode_label!r} on port {excited_op.name!r} "
                f"(cut-off {f_cutoff:.4g} Hz, f_min {self.f_min:.4g} Hz); "
                f"increase f_max or pass an explicit excitation=",
            )
        waveform = "gaussian" if eff_f_min <= 0.0 else "modulated_gaussian"
        return ExcitationSpec(
            f_min=eff_f_min,
            f_max=self.f_max,
            waveform=waveform,
        )

    def _resolve_excited(
        self,
        excited: Iterable[ExcitedSpec] | None,
    ) -> list[tuple[str, int]]:
        if excited is None:
            first_label = self._spec_label(self.ports[0])
            return [(first_label, 0)]
        out: list[tuple[str, int]] = []
        labels = {self._spec_label(s) for s in self.ports}
        for entry in excited:
            if isinstance(entry, str):
                key = (entry, 0)
            else:
                if (
                    not isinstance(entry, tuple)
                    or len(entry) != 2
                    or not isinstance(entry[0], str)
                    or not isinstance(entry[1], int)
                ):
                    raise TypeError(
                        f"excited entry must be a port name or a "
                        f"(port, mode_idx) tuple; got {entry!r}",
                    )
                key = (entry[0], entry[1])
            if key[0] not in labels:
                raise ValueError(
                    f"excited port {key[0]!r} not in ports {sorted(labels)}",
                )
            out.append(key)
        if len(set(out)) != len(out):
            raise ValueError(
                f"duplicate excited entries: {out}",
            )
        return out

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

    def _sibc_spec(self):
        """Build (once) the SIBC wall spec of this analysis (WP-D5).

        ``None`` on ``wall_model="perturbative"``.  Otherwise the
        WP-D3/WP-D2 chain on the consolidated mesh: enumerate the wall
        update topology (PEC boundary faces minus port faces — port
        planes stay lossless), resolve each tag's conductor, and fit
        one passive ``Z_s`` ladder per distinct conductor over the
        analysis band ``[f_axis[0], f_max]``.  Cached — the mesh and
        band are fixed per analysis, so every excitation (and a
        resume) reuses the same spec.
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
            f_axis = self.f_axis
            fits = fit_wall_impedances(
                resolved,
                float(f_axis[0]),
                float(self.f_max),
            )
            self._sibc_spec_cache = SIBCSpec(
                surfaces=tuple(surfaces),
                fits=fits,
            )
        return self._sibc_spec_cache

    def _wire_far_field_monitors(self, s_dict, f_axis) -> None:
        """Hand the run's accepted-power curve to far-field monitors.

        ``1 − Σ|S|²`` over every recorded channel of the excited run —
        the reference behind ``FarFieldResult.gain`` and
        ``radiation_efficiency``.  Runtime wiring like the wall-loss
        accounting: not part of the recipe, refreshed per run.
        """
        from magnelio.monitors.far_field import MonitorFarField  # noqa: PLC0415

        ff = [m for m in self.monitors if isinstance(m, MonitorFarField)]
        if not ff:
            return
        accepted = np.ones(len(f_axis))
        for s in s_dict.values():
            accepted = accepted - np.abs(np.asarray(s)) ** 2
        for mon in ff:
            mon._set_accepted_power(f_axis, accepted)

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

    @staticmethod
    def _estimate_steps(
        grid,
        excitation: ExcitationSpec,
        dt: float,
        n_traversals: int = 25,
    ) -> int:
        """Auto-derive the run *scale* — a generous estimate of the
        needed length, not a tight one.

        Heuristic: ``2·t0_pulse + n_traversals · t_diag``, where
        ``t0_pulse = 4 / bandwidth`` covers a Gaussian centred at
        ``4 / bandwidth`` plus a 1-sigma tail, and
        ``t_diag = ‖bbox‖ / v_safe`` uses ``v_safe = 0.5·c₀`` to keep
        a margin against dispersion (group velocity in hollow WG drops
        below c₀ near cutoff).

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
        if excitation.f_max <= 0.0:
            raise ValueError("excitation.f_max must be positive")
        bandwidth = max(
            excitation.f_max - max(excitation.f_min, 0.0),
            excitation.f_max,  # lower-bound: at least one f_max-period
        )
        t0_pulse = 4.0 / bandwidth
        Lx = float(grid.x[-1] - grid.x[0])
        Ly = float(grid.y[-1] - grid.y[0])
        Lz = float(grid.z[-1] - grid.z[0])
        L_diag = math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
        v_safe = 0.5 * C0
        t_diag = L_diag / v_safe if L_diag > 0.0 else 0.0
        t_total = 2.0 * t0_pulse + n_traversals * t_diag
        return int(math.ceil(t_total / dt))

    def __repr__(self) -> str:
        labels = [self._spec_label(s) for s in self.ports]
        return f"AnalysisScatteringTD(ports={labels}, f_max={self.f_max:.3e})"


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
    """Restore MonitorFarField accumulators from far_field.h5 (resume).

    Same contract as the frequency and wall-loss dumps: reloaded only
    when the file's step matches the checkpoint's ``n_completed``.  A
    no-op when the run carries no far-field monitors.
    """
    from pathlib import Path  # noqa: PLC0415

    from magnelio.io.project import _read_far_field_dump  # noqa: PLC0415
    from magnelio.monitors.far_field import MonitorFarField  # noqa: PLC0415

    ff_mons = [m for m in monitors if isinstance(m, MonitorFarField)]
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


def _resume_scattering(
    proj,
    excited=None,
    *,
    energy_stop_db: float | None = None,
    total_time_steps: int | None = None,
    port_signal_stop_db: float | str | None = None,
    max_time_steps: int | str | None = "auto",
    checkpoint_interval: int | None = None,
    verbose: bool = True,
):
    """Continue a project-backed scattering run from its checkpoint (WP-S8).

    Backs :func:`magnelio.resume` for ``setup['analysis'] ==
    "AnalysisScatteringTD"``.  Rebuilds the run's operators from the store
    recipe, loads the latest ``checkpoint.h5`` into a freshly-built solver,
    reopens ``results.h5`` (truncated back to the checkpoint step), and
    marches on with the (optionally overridden) stop criterion — so a
    resumed run is bit-identical to an uninterrupted run of the same total
    length on a deterministically-built line (the DTBC seam injects
    nothing; the checkpoint carries the full CPML ψ + DTBC convolution
    history, DD-070).

    Parameters mirror the run-time knobs of
    :meth:`AnalysisScatteringTD.run`; ``energy_stop_db`` /
    ``total_time_steps`` / ``port_signal_stop_db`` default to the run's
    *original* criterion (stored at launch), so a Ctrl-C-aborted run
    finishes to its target with a bare ``resume(project)``.  To run a
    *completed* run longer, pass a deeper ``energy_stop_db`` or
    ``port_signal_stop_db`` (a larger dB-below-peak) or a larger
    ``total_time_steps``.  ``max_time_steps="auto"`` grants the resumed
    segment a fresh runtime-cap budget past the checkpoint (an explicit
    int is an absolute step bound, ``None`` removes cap and stall
    watchdog); it is not inherited — each segment gets its own.
    """
    from magnelio.io.project import ProjectStore, open_project  # noqa: PLC0415

    run_name = proj._run_name_for_excited(excited)
    run_meta = proj.runs[run_name]
    excited_chan = (run_meta["excited"][0], int(run_meta["excited"][1]))

    ckpt = proj.checkpoint_state(excited_chan)
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
    # independent, as in :meth:`run`, so ``energy_stop_db=`` alone
    # switches a formerly bounded run to an energy-gated continuation
    # without the old cap silently re-blocking it.
    if energy_stop_db is None and total_time_steps is None and port_signal_stop_db is None:
        energy_stop_db = original_esd
        total_time_steps = run_meta.get("total_time_steps")
        port_signal_stop_db = run_meta.get("port_signal_stop_db")

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

    analysis = AnalysisScatteringTD.from_project(proj, verbose=verbose)
    mesh = analysis.mesh
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    bc_objects = analysis._resolve_bc()

    # Rebuild the operators + excitation + a fresh recorder exactly as the
    # first run did (same mesh, specs, dt, f_calc) — the fresh recorder
    # restarts at local index 0, i.e. global step n_completed.
    (
        operators,
        element_ops,
        waveform_fn,
        n_steps_estimate,
        recorder,
        port_modes,
        port_normal_dx,
        port_line_params,
    ) = analysis._prepare_excitation_run(
        excited_chan,
        analysis.excitation,
        m_eps,
        m_mu,
        dt,
        analysis.f_max,
    )
    solver_steps, check_interval = analysis._resolve_runtime(
        total_time_steps,
        energy_stop_db,
        n_steps_estimate,
        port_signal_stop_db=port_signal_stop_db,
    )
    if isinstance(max_time_steps, str) and max_time_steps != "auto":
        raise ValueError(
            f"max_time_steps must be an int (steps), None or 'auto'; got {max_time_steps!r}",
        )
    cap_steps = analysis._resolve_cap(
        max_time_steps,
        total_time_steps,
        n_steps_estimate,
        start_step=n_completed,
    )
    ckpt_interval = (
        checkpoint_interval if checkpoint_interval is not None else max(1, n_steps_estimate // 8)
    )

    # Monitors rebuilt from the recipe; their append streams are truncated
    # back to each monitor's checkpointed sample count so the resumed run
    # appends onward without a gap or duplicate.  Both MonitorFieldTime and
    # MonitorFluxTime checkpoint a next_idx; a shared map is safe because
    # each truncation loop only reads the names in its own HDF5 group.
    # (MonitorFieldFrequency carries no next_idx — its DFT accumulator lives
    # in fields_freq.h5 and is restored separately.)
    # SIBC wall model (WP-D5): rebuild the spec from the recipe-restored
    # analysis (same mesh, band and overrides ⇒ same operator), and point
    # the rebuilt wall-loss monitors at it before they attach.
    analysis._wire_wall_monitors()
    run_monitors = list(analysis.monitors)
    stream_keep = {
        name: int(msd["next_idx"])
        for name, msd in ckpt.get("monitors", {}).items()
        if "next_idx" in msd
    }
    store = ProjectStore(proj.path)
    sink = store.reopen_scattering_run(
        run_name,
        recorder=recorder,
        waveform_fn=waveform_fn,
        dt=dt,
        n_keep=n_completed,
        step_offset=n_completed,
        monitors=run_monitors,
        grid=mesh.grid,
        monitor_keep=stream_keep,
        flux_keep=stream_keep,
    )
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bc_objects,
        ports=operators + element_ops,
        recorder=recorder,
        total_time_steps=solver_steps,
        energy_check_interval=check_interval,
        dt=dt,
        energy_stop_db=energy_stop_db,
        port_signal_stop_db=port_signal_stop_db,
        port_signal_min_steps=n_steps_estimate,
        max_time_steps=cap_steps,
        verbose=verbose,
        monitors=run_monitors,
        sink=sink,
        sibc=analysis._sibc_spec(),
    )
    # Build the constant operators, then load the evolving state; the
    # excitation closure armed by _prepare_excitation_run survives the load
    # (state_dict restores only the source-history buffer, not the closure).
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

    if verbose:
        target = "∞" if solver_steps is None else str(solver_steps)
        print(
            f"[AnalysisScatteringTD] resuming run {run_name} from step "
            f"{n_completed} to {target} "
            f"(energy_stop_db={energy_stop_db}, "
            f"port_signal_stop_db={port_signal_stop_db})",
        )
    analysis._drive_streamed_solver(solver, sink, excited_chan, proj.path)
    # ``_actual_steps`` counts from step zero across the resume, so the
    # sampled waveform spans the whole run, not just the appended tail.
    _renormalize_freq_monitors(
        run_monitors,
        _sampled_excitation(waveform_fn, solver._actual_steps or target, dt),
    )
    return open_project(proj.path)
