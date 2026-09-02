"""The scattering-result contract shared by RAM and store results.

One interface, two implementations: the in-RAM
:class:`~magnelio.analysis.scattering_td.ScatteringTDResult` returned by
``AnalysisScatteringTD.run()`` without a project, and the store-backed
:class:`~magnelio.io.project.Project` reader returned with one.  User
scripts must work identically against either, so the contract is pinned
here — as a :class:`typing.Protocol` for typing, as
:class:`ScatteringResultMixin` for the accessors that derive from
``S(...)``, and cross-checked by
``tests/integration/test_result_contract.py`` running the same
assertions over both implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from magnelio.post.sparameter_result import SDerivedAccessors


@dataclass(frozen=True)
class RunSettings:
    """Settings a finished run was produced with, readable off the result.

    All fields are optional: the in-RAM result fills what the analysis
    knew at run time; the store-backed reader fills what the project
    recorded (older stores may lack individual entries).
    """

    f_max: float | None = None
    f_min: float | None = None
    n_freq: int | None = None
    dt: float | None = None
    n_actual_steps: int | None = None
    accuracy: str | None = None
    energy_stop_db: float | None = None
    port_signal_stop_db: float | None = None
    taper_signals: bool | None = None
    # Why the marching ended ("energy", "port_signal",
    # "port_signal_stall", "runtime_cap", "steps", "aborted") and the
    # port |V|-envelope level below peak it reached — the provenance of
    # a truncated record ("stall"/"cap" runs stopped short of their
    # criterion; the level bounds the truncation residual).
    stop_reason: str | None = None
    final_port_signal_db: float | None = None
    precision: str | None = None
    backend: str | None = None
    port_model_used: str | None = None
    # Requested physical duration [s] of a general time-domain run
    # (``AnalysisTD.run(t_end=…)``); ``None`` when the run length came
    # from a stop criterion or a step count.
    t_end: float | None = None
    # The ``(name, mode)`` keys of the run's excitations — the channels
    # driven one per run on a scattering analysis, the simultaneous
    # drives of a general time-domain run.
    excitations: tuple | None = None


class ScatteringResultMixin(SDerivedAccessors):
    """Accessors derived purely from ``S(...)`` — shared verbatim.

    ``phase`` and ``plot_s`` come from
    :class:`~magnelio.post.sparameter_result.SDerivedAccessors`, the
    same base a plain :class:`SParameterResult` (e.g. a de-embedded
    matrix) uses; this mixin adds the members that need the run's port
    records: the export warning, :meth:`to_touchstone` / :meth:`to_skrf`
    and :meth:`deembed`.
    """

    def deembed(self, distances):
        """Shift port reference planes; return the de-embedded S-matrix.

        Removes the feed-line propagation between a port plane and its
        new reference plane: ``result.deembed({"port1": d})`` moves
        port1's reference plane the distance ``d`` [m] from the port
        plane *into* the domain, and every S-parameter touching that
        port is multiplied by the inverse line propagation factor over
        ``d`` — reflections twice, transmissions once per shifted end.
        A negative distance moves the plane outward (adds line
        length); ports not named keep their plane.

        The shift uses the discrete dispersion of the port's uniform
        feed chain — the same grid propagation the solver applied — so
        de-embedding a uniform feed line removes its phase down to the
        accuracy floor of the run itself, including the
        grid-dispersion part that an analytic ``exp(-jβd)`` would
        leave behind on coarse meshes.  It assumes the cross-section
        stays that of the port over the shifted length.  A quasi-TEM
        channel (microstrip, CPW — an inhomogeneous cross-section on
        modal Mur) carries no certified line parameters; the run keeps
        the port's dispersion record instead, and the shift uses the
        true discrete modes of the feed solved at every frequency of
        the axis, so the line's physical dispersion is removed as
        well (the first call on such a result spends a few seconds on
        that solve).  Only a channel with neither — a feed section
        that is not a uniform chain behind the port — falls back to
        the mode's continuum ``γ(f)``, for a quasi-TEM mode the
        frequency-flat quasi-static one.

        Below its cut-off a channel's factor grows exponentially with
        distance, so de-embedded values there keep the diagnostic
        character the raw ones have.  Lumped ports carry no feed-line
        dispersion; naming one raises.

        Parameters
        ----------
        distances : dict[str, float]
            Per-port shift distance [m], positive into the domain.

        Returns
        -------
        SParameterResult
            A new result referenced at the shifted planes — the
            original is untouched.  It answers ``S`` / ``db`` /
            ``phase`` / ``plot_s`` and the Touchstone / scikit-rf
            exports.
        """
        from magnelio.post.deembed import deembed_s_params  # noqa: PLC0415

        dt, line_params, normal_dx, port_modes, *rest = self._deembed_data()
        port_dispersion = rest[0] if rest else None
        return deembed_s_params(
            self.s_params,
            distances,
            dt=dt,
            port_line_params=line_params,
            port_normal_dx=normal_dx,
            port_modes=port_modes,
            port_dispersion=port_dispersion,
        )

    def _deembed_data(self) -> tuple:
        """``(dt, port_line_params, port_normal_dx, port_modes[, port_dispersion])``.

        Both implementations override this from their run records; the
        base raises so any other holder of the contract fails loudly
        instead of de-embedding with nothing.
        """
        raise NotImplementedError(
            "this result does not expose the port line records de-embedding needs."
        )

    def reference_impedance(self, port: str, mode: int = 0) -> np.ndarray:
        """Reference impedance [Ω] of one channel along the frequency axis.

        The real impedance the channel's power waves — and so its row
        and column of the S-matrix — are defined against: the line
        impedance of a TEM or quasi-TEM port mode as the grid carries
        it, the wave impedance of a hollow-pipe mode (which varies
        with frequency), the Thévenin impedance of a lumped port.
        Full-model values on ports cut by a symmetry plane.  See
        :meth:`renormalize` for moving to a common reference.
        """
        return self.s_params.reference_impedance(port, mode)

    def renormalize(self, z_ref):
        """Re-reference the S-matrix to new port impedances.

        The raw S-matrix is measured against each port mode's own
        impedance on the grid (:meth:`reference_impedance`) — a
        uniform line is *matched* there whatever its impedance came
        out at.  This returns the same network against ``z_ref``
        instead, typically ``renormalize(50)``: what a network
        analyser with 50 Ω reference planes would read, and what a
        circuit simulator expects before it cascades this block with
        others.  It acts on the square matrix over the excited
        channels: a channel that was observed but never excited stays
        matched to its own impedance and is left out, as the exports
        leave it out.

        Whether the re-referenced mismatch is real is a modelling
        question: a 49 Ω grid line feeding a 50 Ω system does reflect,
        while a line *meant* to be the 50 Ω one shows a discretisation
        artefact — converge its impedance on the port plane first
        (``refine_port_modes``).

        Parameters
        ----------
        z_ref : float or dict
            New real reference impedance [Ω] for every channel, or a
            mapping ``{port_name: Z}`` / ``{(port, mode): Z}``, each
            value a scalar or an array on the frequency axis.

        Returns
        -------
        SParameterResult
            A new result on the same channels; the original is
            untouched.  It answers ``S`` / ``db`` / ``phase`` /
            ``plot_s`` and the Touchstone / scikit-rf exports.
        """
        return self.s_params.renormalize(z_ref)

    def _channel_cutoffs(self) -> dict | None:
        """Per-channel cut-off frequency [Hz], or ``None`` if unknown.

        Backs the export warning about propagating modes left out of a
        Touchstone file.  Both implementations override it from their
        port-mode records; the fallback keeps the exports working for
        any other holder of the contract.
        """
        return None

    def _warn_export(self, channels) -> None:
        """Warn about propagating modes the export would leave out."""
        from magnelio.post.sparameter_result import (  # noqa: PLC0415
            warn_unexported_modes,
        )

        s_params = self.s_params
        exported = s_params.export_channels(channels)
        warn_unexported_modes(
            exported,
            s_params.channels,
            self._channel_cutoffs(),
            float(np.max(np.asarray(s_params.f_axis, dtype=float))),
            stacklevel=4,
        )

    def to_touchstone(self, path, *, channels=None, z_ref=None) -> None:
        """Write the S-matrix as a Touchstone ``.sNp`` file.

        Exports the square sub-matrix over the excited channels — one
        Touchstone port per channel, so a multi-mode port occupies one
        port per mode.  Channels that were observed but never excited
        are dropped from rows and columns alike; they carry a
        reflection-free boundary throughout the run, so the export is
        the network seen with them *matched*, the same quantity a
        network analyser measures with its unused ports terminated.

        The option line's ``R`` states the reference impedance the
        data refer to.  Touchstone 1.x holds one constant value for
        all ports, so pass ``z_ref`` (typically ``z_ref=50``) to
        renormalise first when the ports' own references differ or
        vary with frequency; without it such a file is written with a
        nominal ``R 50``, a warning, and each port's actual reference
        in the header.

        Warns when a port that *is* exported carries propagating modes
        that the export leaves out: the file then looks like a
        complete N-port while the mode conversion at that port is
        missing from it.

        The ``.sNp`` extension must agree with the number of exported
        channels — Touchstone records the port count nowhere else — so
        a mismatch raises instead of writing an unreadable file.  A
        path without an extension gets the matching one.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.  ``<name>.s{N}p``, or ``<name>`` to have the
            extension filled in.
        channels : sequence of str or (str, int), optional
            Select the exported sub-network explicitly, e.g.
            ``["port1", "port3"]`` to cut a two-port out of a fully
            excited three-port.  A bare port name means mode 0.  Every
            entry must have been excited.
        z_ref : float or dict, optional
            Renormalise to this reference before writing, as in
            :meth:`renormalize`.
        """
        self._warn_export(channels)
        self.s_params.to_touchstone(path, channels=channels, z_ref=z_ref)

    def to_skrf(self, name: str = "magnelio", *, channels=None, z_ref=None):
        """Return the S-matrix as a ``skrf.Network``.

        Requires scikit-rf (extra ``magnelio[interop]``).  Same
        sub-matrix, channel selection and warning as
        :meth:`to_touchstone`; the network's ``z0`` carries each
        channel's reference impedance per frequency.

        Parameters
        ----------
        name : str, optional
            Network name.
        channels : sequence of str or (str, int), optional
            Explicit channel selection, as in :meth:`to_touchstone`.
        z_ref : float or dict, optional
            Renormalise first, as in :meth:`renormalize`.

        Returns
        -------
        skrf.Network
        """
        self._warn_export(channels)
        return self.s_params.to_skrf(name=name, channels=channels, z_ref=z_ref)


@runtime_checkable
class ScatteringResult(Protocol):
    """Everything a scattering result guarantees, RAM- or store-backed.

    A script written against this protocol runs unchanged whether
    ``AnalysisScatteringTD.run()`` returned an in-RAM
    :class:`~magnelio.analysis.ScatteringTDResult` (no project) or a
    :class:`~magnelio.io.project.Project` reader (with one).  The
    protocol is :func:`~typing.runtime_checkable`, so ``isinstance(res,
    ScatteringResult)`` holds for both.

    The members below are what both implementations provide; each
    implementation documents its own behaviour, and the two are
    cross-checked by running one shared set of assertions over both.

    - :attr:`f_axis` — frequency axis of the S-matrix [Hz]
    - :attr:`channels` / :attr:`excitations` — the observed and excited
      ``(port, mode)`` pairs
    - :attr:`settings` — the run's :class:`RunSettings`
    - :meth:`S`, :meth:`db`, :meth:`phase` — one S-parameter over
      frequency, as a complex number, in dB, or as a phase
    - :meth:`a`, :meth:`b` — incident and outgoing power waves in time

    Further accessors come from :class:`ScatteringResultMixin` and are
    shared verbatim rather than reimplemented: ``plot_s``, ``deembed``
    (reference-plane shift) and the ``to_touchstone`` / ``to_skrf``
    exports.
    """

    @property
    def f_axis(self) -> np.ndarray:
        """Frequency axis of the S-matrix [Hz], ascending."""
        ...

    @property
    def channels(self) -> tuple:
        """Observed ``(port_name, mode_idx)`` pairs, in S-matrix order."""
        ...

    @property
    def excitations(self) -> tuple:
        """Excited ``(port_name, mode_idx)`` pairs — the S-matrix columns present."""
        ...

    @property
    def settings(self) -> RunSettings:
        """The :class:`RunSettings` this result was produced with."""
        ...

    def S(self, out_port, in_port, *, mode_out=0, mode_in=0, f_axis=None) -> np.ndarray:
        """One complex S-parameter over the frequency axis."""
        ...

    def db(
        self, out_port, in_port, *, mode_out=0, mode_in=0, floor_db=-200.0, f_axis=None
    ) -> np.ndarray:
        """One S-parameter in decibels, floored at *floor_db*."""
        ...

    def phase(
        self, out_port, in_port, *, mode_out=0, mode_in=0, deg=True, unwrap=True, f_axis=None
    ) -> np.ndarray:
        """Phase of one S-parameter over the frequency axis."""
        ...

    def a(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Incident power-wave time series ``a(t)`` at one channel."""
        ...

    def b(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Outgoing power-wave time series ``b(t)`` at one channel."""
        ...


__all__ = ["RunSettings", "ScatteringResult", "ScatteringResultMixin"]
