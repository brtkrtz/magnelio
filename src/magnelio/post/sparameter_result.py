"""Multi-port S-parameter result wrapper.

Phase 2c steps 11 + 12 (`reference_architecture_phase2_mode_solver.md`
§5).

:func:`compute_s_parameters` produces ``dict[(port_name, mode_idx),
ndarray]`` for one excited channel — a single column of the S-matrix.
This module wraps that output (and aggregates multiple such columns)
into a structured :class:`SParameterResult` with named-port access
(``S.S("port2", "port1")``-style) and a 3D matrix
``(n_frequencies, n_channels, n_excitations)`` suitable for plotting,
multi-port reasoning, and the magic-tee / branch-line use cases the
Phase-2 architecture document §11–§13 anticipates.

The wrapper is intentionally a thin data class — it does *not* re-run
the FIT-TD simulation or the S-parameter extraction, only re-shapes
the data.  Use :func:`compute_s_parameters` for each of the K
excitation runs (one per independent excited channel) and combine the
results via :meth:`SParameterResult.merge` (or the
:meth:`from_multiple_excitations` class-method shortcut) into a
single multi-port S-matrix.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_TOUCHSTONE_SUFFIX = re.compile(r"^\.s(\d+)p$", re.IGNORECASE)


def _touchstone_path(path, channels: Sequence[tuple[str, int]]) -> Path:
    """Resolve the output path, checking the ``.sNp`` port count.

    Touchstone 1.x carries the port count *only* in the file
    extension — the file body has no field for it — so an extension
    that disagrees with the matrix produces a file no reader can
    interpret correctly.  A missing extension is filled in; a
    disagreeing one is an error rather than a silent rename, so a
    downstream script still finds the file it asked for.
    """
    out = Path(path)
    n_ports = len(channels)
    suffix = out.suffix
    if not suffix:
        return out.with_suffix(f".s{n_ports}p")
    match = _TOUCHSTONE_SUFFIX.match(suffix)
    if match is None:
        raise ValueError(
            f"{suffix!r} is not a Touchstone extension; expected "
            f"'.s{n_ports}p' for this {n_ports}-port matrix (or pass a "
            f"path without an extension to have it filled in)."
        )
    declared = int(match.group(1))
    if declared != n_ports:
        raise ValueError(
            f"file name declares a {declared}-port network ('{suffix}') "
            f"but the export covers {n_ports} channel(s): "
            f"{list(channels)}. Rename to '.s{n_ports}p', or excite / "
            f"select (channels=) the channels you meant — Touchstone "
            f"records the port count nowhere but in the extension, so "
            f"writing this file would make it unreadable."
        )
    return out


def warn_unexported_modes(
    exported: Sequence[tuple[str, int]],
    channels: Sequence[tuple[str, int]],
    cutoffs: dict[tuple[str, int], float] | None,
    f_max: float,
    *,
    stacklevel: int = 3,
) -> None:
    """Warn about propagating modes dropped at an *exported* port.

    A channel omitted at a port that is not exported at all is a
    deliberate cut through the network — a one-port reflection export
    of a two-port, say — and the result describes that sub-network
    correctly, with the omitted port matched.  Higher modes dropped at
    a port that *is* exported are the subtle case: the file then looks
    like a complete N-port while the mode conversion at that port is
    missing from it, so cascading it in a circuit simulator quietly
    loses the power scattered into the omitted modes.

    Only modes that propagate inside the exported band can carry that
    power.  Evanescent ones cannot — solving for more modes than one
    excites, so that the evanescent content is represented at the port
    plane, is ordinary practice and draws no warning.  Does nothing
    when the cut-off frequencies are unknown.
    """
    if not cutoffs:
        return
    exported = tuple(exported)
    exported_set = set(exported)
    exported_ports = {port for port, _ in exported}
    seen: list[str] = []
    for port, _ in channels:
        if port in exported_ports and port not in seen:
            seen.append(port)
    for port in seen:
        dropped = [m for (p, m) in channels if p == port and (p, m) not in exported_set]
        propagating = []
        for mode in dropped:
            f_c = cutoffs.get((port, mode))
            if f_c is not None and f_c < f_max:
                propagating.append(mode)
        if not propagating:
            continue
        f_low = min(cutoffs[(port, mode)] for mode in propagating)
        kept = [m for (p, m) in exported if p == port]
        warnings.warn(
            f"port {port!r}: mode(s) {propagating} are left out of the "
            f"export but propagate above {f_low / 1e9:.4g} GHz, so the "
            f"exported channel(s) {kept} do not describe the mode "
            f"conversion at this port. The data are valid for the "
            f"exported channels with the others matched, but the file "
            f"is not a complete model of the component — excite and "
            f"export those modes to cascade it.",
            UserWarning,
            stacklevel=stacklevel,
        )


class SDerivedAccessors:
    """Accessors derived purely from ``S(...)``/``db(...)``.

    Shared verbatim between :class:`SParameterResult` and the
    scattering-result contract
    (:class:`~magnelio.analysis.result_interface.ScatteringResultMixin`),
    so a de-embedded or otherwise post-processed S-matrix answers the
    same ``phase``/``plot_s`` calls as the run result it came from.
    Hosts require ``S``, ``db``, ``f_axis``, ``channels`` and
    ``excitations``.
    """

    def phase(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
        deg: bool = True,
        unwrap: bool = True,
        f_axis=None,
    ) -> np.ndarray:
        """Phase of one S-parameter over the frequency axis.

        Parameters
        ----------
        out_port, in_port : str
            Observed / excited port names (modes via ``mode_out`` /
            ``mode_in``), as in :meth:`S`.
        deg : bool, default True
            Return degrees; ``False`` returns radians.
        unwrap : bool, default True
            Unwrap 2π discontinuities along the frequency axis.
        f_axis : array-like, optional
            Custom frequency axis, on hosts whose :meth:`S` can
            recompute the spectrum (run results); a plain
            :class:`SParameterResult` holds one fixed axis and
            rejects it.

        Returns
        -------
        np.ndarray
            Phase per frequency point.
        """
        kwargs = {} if f_axis is None else {"f_axis": f_axis}
        s = self.S(out_port, in_port, mode_out=mode_out, mode_in=mode_in, **kwargs)
        ph = np.angle(s)
        if unwrap:
            ph = np.unwrap(ph)
        return np.degrees(ph) if deg else ph

    def plot_s(self, *pairs, db=True, floor_db=-200.0, ax=None):
        """Plot S-parameter magnitudes over frequency.

        Parameters
        ----------
        *pairs : tuple
            Channels to plot, each ``(out_port, in_port)`` or
            ``(out_port, in_port, mode_out, mode_in)``.  Without
            arguments every recorded channel of every excitation is
            plotted.
        db : bool, default True
            Magnitude in dB (with *floor_db*) instead of linear.
        floor_db : float, default -200.0
            Clip floor for the dB display.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; a new figure is created otherwise.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if not pairs:
            pairs = tuple(
                (out_port, in_port, mode_out, mode_in)
                for (in_port, mode_in) in self.excitations
                for (out_port, mode_out) in self.channels
            )
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        f_ghz = np.asarray(self.f_axis) / 1e9
        multi_mode = any(len(p) == 4 and (p[2] or p[3]) for p in pairs)
        for p in pairs:
            out_port, in_port = p[0], p[1]
            mode_out = p[2] if len(p) > 2 else 0
            mode_in = p[3] if len(p) > 3 else 0
            if db:
                y = self.db(
                    out_port, in_port, mode_out=mode_out, mode_in=mode_in, floor_db=floor_db
                )
            else:
                y = np.abs(self.S(out_port, in_port, mode_out=mode_out, mode_in=mode_in))
            label = f"S({out_port} ← {in_port})"
            if multi_mode:
                label = f"S({out_port}:{mode_out} ← {in_port}:{mode_in})"
            ax.plot(f_ghz, y, label=label)
        ax.set_xlabel("f / GHz")
        ax.set_ylabel("|S| / dB" if db else "|S|")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax


@dataclass(frozen=True)
class SParameterResult(SDerivedAccessors):
    """Structured multi-port S-parameter spectrum.

    A single :class:`SParameterResult` holds the full S-matrix
    ``S[f, observed, excited]`` of a multi-port network at a fixed
    list of frequencies, along with the channel labels and the
    subset of channels that were actually excited.

    Channels are addressed by ``(port_name, mode_idx)`` tuples; the
    convenience accessors (:meth:`S`, :meth:`db`) take the more
    ergonomic ``port_name`` strings plus optional mode indices.

    Attributes
    ----------
    f_axis : np.ndarray
        Real, strictly positive frequencies [Hz], shape ``(Nf,)``.
    channels : tuple of (str, int)
        Observed channels in canonical order.  Same ordering as the
        rows of :attr:`matrix`.
    excitations : tuple of (str, int)
        Excited channels in canonical order — a subset of
        :attr:`channels`.  Same ordering as the columns of
        :attr:`matrix`.
    matrix : np.ndarray, shape (Nf, n_channels, n_excitations)
        Complex S-matrix.  ``matrix[k, i, j]`` is ``S(observed=i,
        excited=j)`` at frequency ``f_axis[k]``.

    Notes
    -----
    The wrapper enforces channel-key consistency at construction; the
    matrix shape must match ``(Nf, len(channels), len(excitations))``.

    Single-excitation runs produce an SParameterResult with
    ``n_excitations = 1``; that is the natural form for one Phase-2
    FIT-TD simulation result.  Use :meth:`merge` to combine K such
    results into the full K-excitation S-matrix.
    """

    f_axis: np.ndarray
    channels: tuple[tuple[str, int], ...]
    excitations: tuple[tuple[str, int], ...]
    matrix: np.ndarray
    # Per-channel real reference impedance [Ω] on ``f_axis`` — the
    # impedance the power waves of that channel are defined against
    # (DD-244).  ``None`` when the producer did not record it; NaN at
    # frequencies where a channel carries no propagating mode.
    reference_impedances: dict[tuple[str, int], np.ndarray] | None = None

    def __post_init__(self) -> None:
        f = np.asarray(self.f_axis)
        if f.ndim != 1 or f.size == 0:
            raise ValueError(f"f_axis must be 1D with at least one frequency; got shape {f.shape}.")
        if np.any(f <= 0.0):
            raise ValueError("f_axis must contain only positive frequencies.")
        n_ch = len(self.channels)
        n_ex = len(self.excitations)
        if self.reference_impedances is not None:
            refs = {}
            for key, z in self.reference_impedances.items():
                key = (key[0], int(key[1]))
                if key not in self.channels:
                    raise ValueError(
                        f"reference_impedances names channel {key!r}, which is not observed."
                    )
                z = np.asarray(z, dtype=float)
                if z.shape != (f.size,):
                    raise ValueError(
                        f"reference_impedances[{key!r}] has shape {z.shape}; "
                        f"expected ({f.size},) to match f_axis."
                    )
                refs[key] = z
            missing = [c for c in self.channels if c not in refs]
            if missing:
                raise ValueError(f"reference_impedances lacks channels {missing}.")
            object.__setattr__(self, "reference_impedances", refs)
        if n_ch == 0:
            raise ValueError("channels must be non-empty.")
        if n_ex == 0:
            raise ValueError("excitations must be non-empty.")
        if len(set(self.channels)) != n_ch:
            raise ValueError("channels must be unique.")
        if len(set(self.excitations)) != n_ex:
            raise ValueError("excitations must be unique.")
        # Excitations must be a subset of observed channels.
        observed = set(self.channels)
        for exc in self.excitations:
            if exc not in observed:
                raise ValueError(f"excitation {exc!r} not in channels {sorted(observed)}.")
        m = np.asarray(self.matrix)
        expected = (f.size, n_ch, n_ex)
        if m.shape != expected:
            raise ValueError(
                f"matrix shape {m.shape} does not match expected "
                f"{expected} = (n_frequencies, n_channels, n_excitations)."
            )

    # ------------------------------------------------------------------
    # How the matrix introduces itself (DD-254): size and labels, never
    # the 201 × n × k complex numbers behind them.
    # ------------------------------------------------------------------

    def _summary_rows(self) -> list[tuple[str, object]]:
        f = np.asarray(self.f_axis)
        return [
            ("channels", list(self.channels)),
            ("excitations", list(self.excitations)),
            ("frequency", f"{f[0] / 1e9:.4g}–{f[-1] / 1e9:.4g} GHz ({f.size} points)"),
            ("matrix", np.asarray(self.matrix)),
            ("reference impedances", self.reference_impedances is not None),
        ]

    def __repr__(self) -> str:
        from magnelio._repr import kv_block  # noqa: PLC0415

        return kv_block("SParameterResult", self._summary_rows())

    def _repr_html_(self) -> str:
        from magnelio._repr import html_kv  # noqa: PLC0415

        return html_kv("SParameterResult", self._summary_rows())

    # ------------------------------------------------------------------
    # Shape / introspection
    # ------------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        return int(self.f_axis.size)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_excitations(self) -> int:
        return len(self.excitations)

    @property
    def port_names(self) -> tuple[str, ...]:
        """Unique port names in first-occurrence order across :attr:`channels`."""
        seen: set[str] = set()
        out: list[str] = []
        for label, _ in self.channels:
            if label not in seen:
                seen.add(label)
                out.append(label)
        return tuple(out)

    @property
    def is_complete(self) -> bool:
        """True iff every observed channel was also excited (K×K matrix)."""
        return set(self.excitations) == set(self.channels)

    def channel_index(self, port: str, mode: int = 0) -> int:
        """Index of ``(port, mode)`` into :attr:`channels`."""
        key = (port, mode)
        try:
            return self.channels.index(key)
        except ValueError:
            raise KeyError(
                f"channel ({port!r}, {mode}) not in result; available: {list(self.channels)}"
            )

    def excitation_index(self, port: str, mode: int = 0) -> int:
        """Index of ``(port, mode)`` into :attr:`excitations`."""
        key = (port, mode)
        try:
            return self.excitations.index(key)
        except ValueError:
            raise KeyError(
                f"excitation ({port!r}, {mode}) not present; this result "
                f"only carries excitations {list(self.excitations)}."
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def S(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
    ) -> np.ndarray:
        """Return the spectrum ``S(out_port, in_port)``.

        Conventionally ``S_{out,in}`` is the wave amplitude received at
        the *out* port-mode in response to a unit-amplitude wave
        injected at the *in* port-mode.

        Parameters
        ----------
        out_port, in_port : str
            Port labels.  Must match channel keys.
        mode_out, mode_in : int, optional
            Mode indices on each port (default 0).
        """
        i = self.channel_index(out_port, mode_out)
        j = self.excitation_index(in_port, mode_in)
        return self.matrix[:, i, j].copy()

    def db(
        self,
        out_port: str,
        in_port: str,
        *,
        mode_out: int = 0,
        mode_in: int = 0,
        floor_db: float = -200.0,
    ) -> np.ndarray:
        """Return ``20·log10|S(out, in)|`` with a floor at ``floor_db``.

        The floor avoids ``-inf`` at frequencies where the
        S-parameter is at the FFT round-off (``|S| ≈ 1e-12`` or
        smaller).  Default ``-200 dB`` is below any physically
        meaningful response.
        """
        s = self.S(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
        )
        mag = np.abs(s)
        # Replace NaN (under-threshold from compute_s_parameters) with
        # the floor; clip the rest at the floor to avoid -inf.
        mag_safe = np.where(np.isnan(mag), 0.0, mag)
        floor_lin = 10.0 ** (floor_db / 20.0)
        clipped = np.maximum(mag_safe, floor_lin)
        return 20.0 * np.log10(clipped)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def export_channels(
        self,
        channels: Sequence[str | tuple[str, int]] | None = None,
    ) -> tuple[tuple[str, int], ...]:
        """The channel set an export covers, in canonical channel order.

        Default (``channels=None``): every excited channel.  Rows and
        columns of the exported matrix are that same set, so the
        result is a square sub-matrix in which every entry was
        measured — nothing is padded or inferred.

        The sub-matrix is a valid network description in its own
        right.  Channels left out of it are not open circuits: each
        one is terminated by its own reflection-free port boundary
        throughout the run, which is exactly the matched-termination
        condition the definition of S-parameters asks for.  The export
        is therefore the network *seen with the omitted channels
        matched* — the same quantity a vector network analyser
        measures with its unused ports terminated.

        Parameters
        ----------
        channels : sequence of str or (str, int), optional
            Explicit channel selection, e.g. to cut a two-port out of
            a fully excited three-port.  A bare port name means mode
            0.  Every entry must have been excited.

        Returns
        -------
        tuple of (str, int)
        """
        if channels is None:
            excited = set(self.excitations)
            return tuple(chan for chan in self.channels if chan in excited)

        sel: list[tuple[str, int]] = []
        for entry in channels:
            chan = (entry, 0) if isinstance(entry, str) else (entry[0], int(entry[1]))
            if chan not in self.channels:
                raise ValueError(
                    f"channel {chan!r} is not in this result; available: {list(self.channels)}"
                )
            if chan not in self.excitations:
                raise ValueError(
                    f"channel {chan!r} was never excited, so its column "
                    f"of the S-matrix was not measured; excited "
                    f"channels are {list(self.excitations)}."
                )
            if chan in sel:
                raise ValueError(f"channel {chan!r} listed twice in channels=.")
            sel.append(chan)
        if not sel:
            raise ValueError("channels= is empty; pass at least one channel.")
        order = {chan: i for i, chan in enumerate(self.channels)}
        sel.sort(key=lambda chan: order[chan])
        return tuple(sel)

    def _export_matrix(
        self,
        channels: Sequence[str | tuple[str, int]] | None = None,
    ) -> tuple[tuple[tuple[str, int], ...], np.ndarray]:
        """``(channels, S)`` of the square sub-matrix to export."""
        sel = self.export_channels(channels)
        rows = [self.channels.index(chan) for chan in sel]
        cols = [self.excitations.index(chan) for chan in sel]
        return sel, self.matrix[:, rows, :][:, :, cols]

    # ------------------------------------------------------------------
    # Reference impedance
    # ------------------------------------------------------------------

    def reference_impedance(self, port: str, mode: int = 0) -> np.ndarray:
        """Reference impedance [Ω] of one channel along :attr:`f_axis`.

        The real impedance the channel's power waves are defined
        against: the line impedance of a TEM or quasi-TEM channel, the
        wave impedance of a hollow-pipe mode (which varies with
        frequency), the Thévenin impedance of a lumped port.  On a
        port cut by a symmetry plane it is the full-model value.

        Raises
        ------
        ValueError
            If the result carries no reference impedances.
        """
        if self.reference_impedances is None:
            raise ValueError(
                "this S-parameter result carries no reference impedances; "
                "results of a scattering run and of a project store do."
            )
        key = (port, int(mode))
        if key not in self.reference_impedances:
            raise KeyError(f"channel {key!r} not in result; available: {list(self.channels)}")
        return self.reference_impedances[key].copy()

    def renormalize(self, z_ref) -> "SParameterResult":
        """Re-reference the S-matrix to new port impedances.

        A scattering result is measured against each channel's own
        reference impedance — the impedance its port mode carries on
        the grid, :meth:`reference_impedance`.  This returns the same
        network described against ``z_ref`` instead: the S-matrix a
        network analyser with ``z_ref`` reference planes would read,
        or the one a circuit simulator expects when it cascades this
        block with others on a common impedance.

        The transformation is the exact power-wave re-referencing for
        real reference impedances (Kurokawa): with ``ρ_i = (Z_i −
        Z'_i)/(Z_i + Z'_i)`` and ``c_i = (Z_i + Z'_i)/(2√(Z_i Z'_i))``,

            S' = C (S + ρ)(I + ρ S)⁻¹ C⁻¹ ,

        applied per frequency to the square matrix over the *excited*
        channels (:meth:`export_channels`).  A channel that was
        observed but never excited cannot be re-referenced — its own
        reflection would enter every other entry once its reference
        moves — and is left out of the result exactly as the exports
        leave it out: it stays terminated by its own reflection-free
        boundary, matched to its own impedance.

        Note what the operation means physically.  A line whose grid
        impedance came out at 49 Ω against a 50 Ω design is *matched*
        in the raw result and shows a −40 dB reflection after
        re-referencing to 50 Ω — that mismatch is real if the line is
        part of the device and will be fed from 50 Ω, and a
        discretisation artefact if the line is meant to be the 50 Ω
        one; converge the port's impedance first
        (``refine_port_modes``) before reading it either way.

        Parameters
        ----------
        z_ref : float or dict
            New real reference impedance [Ω]: one value for every
            channel, or a mapping ``{port_name: Z}`` / ``{(port,
            mode): Z}``; each value may also be an array on
            :attr:`f_axis`.  Channels not named keep their impedance.

        Returns
        -------
        SParameterResult
            A new result on the same channels, carrying ``z_ref`` as
            its reference impedances; the original is untouched.

        Raises
        ------
        ValueError
            If the result carries no reference impedances, or a new
            impedance is not positive.
        """
        if self.reference_impedances is None:
            raise ValueError("cannot renormalize: this result carries no reference impedances.")
        if not self.is_complete:
            return self._sub_result(self.export_channels(None)).renormalize(z_ref)
        n_f = self.n_frequencies
        new_refs = {key: z.copy() for key, z in self.reference_impedances.items()}
        if isinstance(z_ref, dict):
            for key, val in z_ref.items():
                targets = (
                    [c for c in self.channels if c[0] == key]
                    if isinstance(key, str)
                    else [(key[0], int(key[1]))]
                )
                if not targets or any(t not in new_refs for t in targets):
                    raise KeyError(
                        f"z_ref names {key!r}, not a channel of this result "
                        f"({list(self.channels)})."
                    )
                for t in targets:
                    new_refs[t] = np.broadcast_to(np.asarray(val, dtype=float), (n_f,)).copy()
        else:
            for key in new_refs:
                new_refs[key] = np.broadcast_to(np.asarray(z_ref, dtype=float), (n_f,)).copy()
        for key, z in new_refs.items():
            if np.any(z[np.isfinite(z)] <= 0.0):
                raise ValueError(f"reference impedance for {key!r} must be positive.")

        # Square matrix in channel order: rows are self.channels, the
        # columns follow self.excitations — permute them to match.
        cols = [self.excitations.index(chan) for chan in self.channels]
        S = self.matrix[:, :, cols]
        z_old = np.stack([self.reference_impedances[c] for c in self.channels], axis=1)
        z_new = np.stack([new_refs[c] for c in self.channels], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = (z_old - z_new) / (z_old + z_new)
            c = (z_old + z_new) / (2.0 * np.sqrt(z_old * z_new))
        n = len(self.channels)
        eye = np.eye(n)
        out = np.empty_like(S)
        for k in range(n_f):
            if not np.all(np.isfinite(rho[k])) or not np.all(np.isfinite(S[k])):
                out[k] = np.nan + 1j * np.nan
                continue
            rho_k = np.diag(rho[k])
            # (S + ρ)(I + ρS)^{-1} without forming the inverse.
            rhs = np.linalg.solve((eye + rho_k @ S[k]).T, (S[k] + rho_k).T).T
            out[k] = (c[k][:, None] * rhs) / c[k][None, :]
        inv_cols = [self.channels.index(chan) for chan in self.excitations]
        return SParameterResult(
            f_axis=self.f_axis.copy(),
            channels=self.channels,
            excitations=self.excitations,
            matrix=out[:, :, inv_cols],
            reference_impedances=new_refs,
        )

    def _uniform_reference(self, chans) -> float | None:
        """The one flat reference impedance of ``chans``, or ``None``."""
        if self.reference_impedances is None:
            return None
        z = np.stack([self.reference_impedances[c] for c in chans], axis=1)
        if not np.all(np.isfinite(z)):
            return None
        z0 = float(z[0, 0])
        if z0 <= 0.0 or not np.allclose(z, z0, rtol=1e-9, atol=0.0):
            return None
        return z0

    def _reference_lines(self, chans) -> list[str]:
        """Comment lines naming each exported channel's reference."""
        if self.reference_impedances is None:
            return ["! reference impedances: not recorded"]
        f = np.asarray(self.f_axis, dtype=float)
        lines = []
        for k, chan in enumerate(chans, start=1):
            z = self.reference_impedances[chan]
            finite = np.isfinite(z)
            if not finite.any():
                lines.append(f"! port {k} reference impedance: undefined on this axis")
                continue
            lo, hi = float(np.min(z[finite])), float(np.max(z[finite]))
            if np.isclose(lo, hi, rtol=1e-9, atol=0.0):
                lines.append(f"! port {k} reference impedance: {lo:.6g} Ohm (constant)")
            else:
                k0 = int(np.flatnonzero(finite)[0])
                k1 = int(np.flatnonzero(finite)[-1])
                lines.append(
                    f"! port {k} reference impedance: {z[k0]:.6g} Ohm at {f[k0]:.6g} Hz "
                    f"to {z[k1]:.6g} Ohm at {f[k1]:.6g} Hz (frequency dependent)"
                )
        return lines

    def _sub_result(self, chans) -> "SParameterResult":
        """The square result over ``chans`` (their matched sub-network)."""
        rows = [self.channels.index(chan) for chan in chans]
        cols = [self.excitations.index(chan) for chan in chans]
        refs = None
        if self.reference_impedances is not None:
            refs = {c: self.reference_impedances[c] for c in chans}
        return SParameterResult(
            f_axis=self.f_axis.copy(),
            channels=tuple(chans),
            excitations=tuple(chans),
            matrix=self.matrix[:, rows, :][:, :, cols],
            reference_impedances=refs,
        )

    def _for_export(self, channels, z_ref):
        """``(channels, S, R, source)`` of an export, renormalised when asked.

        ``R`` is the reference impedance the Touchstone option line can
        state: the value every exported channel shares when all are
        constant and equal, else ``None``.
        """
        sel = self.export_channels(channels)
        src = self._sub_result(sel)
        if z_ref is not None:
            src = src.renormalize(z_ref)
        chans, s = src._export_matrix(None)
        return chans, s, src._uniform_reference(chans), src

    def to_touchstone(
        self,
        path,
        *,
        channels: Sequence[str | tuple[str, int]] | None = None,
        z_ref: float | dict | None = None,
    ) -> None:
        """Write the S-matrix as a Touchstone ``.sNp`` file.

        Exports the square sub-matrix over
        :meth:`export_channels` — by default every excited channel,
        with the unexcited ones matched (see that method for what the
        reduction does and does not describe).  Touchstone ports are
        the exported *channels* in canonical order — a multi-mode port
        occupies one Touchstone port per mode; the mapping is recorded
        in the file's comment header.

        Data are power-wave S-parameters, and the option line's ``R``
        states the impedance they refer to.  Touchstone 1.x can state
        one constant value for all ports, so the file is exact when
        every exported channel shares one frequency-flat reference —
        pass ``z_ref`` to renormalise to such a value first (typically
        ``z_ref=50``).  Without it, channels whose references differ
        or vary with frequency (hollow-pipe modes) are written with a
        nominal ``R 50`` and a warning; the header then lists each
        port's actual reference, and a reader that renormalises on
        ``R`` would be wrong.  :meth:`to_skrf` carries per-port,
        per-frequency references and needs no such choice.

        The ``.sNp`` extension must agree with the exported port
        count: Touchstone 1.x records the port count nowhere else, so
        a mismatch is rejected.  A path without an extension gets the
        matching one.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.  ``<name>.s{N}p``, or ``<name>`` to have the
            extension filled in.
        channels : sequence of str or (str, int), optional
            Explicit channel selection, as in :meth:`export_channels`.
        z_ref : float or dict, optional
            Renormalise to this reference before writing, as in
            :meth:`renormalize`.
        """
        chans, s, r_common, src = self._for_export(channels, z_ref)
        n = len(chans)
        out = _touchstone_path(path, chans)
        lines = ["! magnelio S-parameter export (power-wave S-parameters)"]
        for k, chan in enumerate(chans, start=1):
            lines.append(f"! port {k} = channel {chan[0]!r} mode {chan[1]}")
        lines.extend(src._reference_lines(chans))
        if r_common is None:
            if src.reference_impedances is not None:
                warnings.warn(
                    "Touchstone 1.x states one constant reference impedance for "
                    "all ports, but the exported channels' references differ or "
                    "vary with frequency (see the file header); writing the "
                    "nominal 'R 50' without renormalising.  Pass z_ref= (e.g. "
                    "z_ref=50) to renormalise first, or use to_skrf(), which "
                    "carries per-port, per-frequency references.",
                    stacklevel=2,
                )
            lines.append("# Hz S RI R 50")
        else:
            lines.append(f"# Hz S RI R {r_common:.10g}")
        for k, f in enumerate(np.asarray(self.f_axis, dtype=float)):
            if n <= 2:
                # Touchstone 1.x two-port order is S11 S21 S12 S22
                # (column-major).
                entries = [s[k, i, j] for j in range(n) for i in range(n)]
                vals = " ".join(f"{v.real:.12e} {v.imag:.12e}" for v in entries)
                lines.append(f"{f:.12e} {vals}")
            else:
                for i in range(n):
                    row = " ".join(
                        f"{s[k, i, j].real:.12e} {s[k, i, j].imag:.12e}" for j in range(n)
                    )
                    lines.append(f"{f:.12e} {row}" if i == 0 else row)
        out.write_text("\n".join(lines) + "\n", encoding="ascii")

    def to_skrf(
        self,
        name: str = "magnelio",
        *,
        channels: Sequence[str | tuple[str, int]] | None = None,
        z_ref: float | dict | None = None,
    ):
        """Return the S-matrix as a ``skrf.Network``.

        Requires ``scikit-rf`` (install extra ``magnelio[interop]``).
        Exports the same square sub-matrix as
        :meth:`to_touchstone` — by default every excited channel, with
        the unexcited ones matched; multi-mode ports map to one
        network port per channel, in canonical channel order.  The
        network's ``z0`` carries each channel's reference impedance
        per frequency, so scikit-rf's own ``renormalize`` and
        cascading operate on the right references.

        Parameters
        ----------
        name : str, optional
            Network name.
        channels : sequence of str or (str, int), optional
            Explicit channel selection, as in :meth:`export_channels`.
        z_ref : float or dict, optional
            Renormalise first, as in :meth:`renormalize`.

        Returns
        -------
        skrf.Network
        """
        try:
            import skrf  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "to_skrf() requires scikit-rf — install it via "
                "'pip install scikit-rf' (or the magnelio[interop] extra)"
            ) from exc
        chans, s, _r, src = self._for_export(channels, z_ref)
        freq = skrf.Frequency.from_f(
            np.asarray(self.f_axis, dtype=float),
            unit="hz",
        )
        if src.reference_impedances is not None:
            z0 = np.stack([src.reference_impedances[c] for c in chans], axis=1)
            ntw = skrf.Network(frequency=freq, s=s, z0=z0, name=name)
        else:
            ntw = skrf.Network(frequency=freq, s=s, name=name)
        ntw.port_names = [f"{p}:{m}" for (p, m) in chans]
        return ntw

    @classmethod
    def from_single_excitation(
        cls,
        s_dict: dict[tuple[str, int], np.ndarray],
        excited: tuple[str, int],
        f_axis: np.ndarray,
        *,
        channel_order: tuple[tuple[str, int], ...] | None = None,
        reference_impedances: dict | None = None,
    ) -> "SParameterResult":
        """Wrap a single-column :func:`compute_s_parameters` dict.

        Parameters
        ----------
        s_dict : dict[(str, int), np.ndarray]
            Direct output of :func:`compute_s_parameters`.  Each value
            is a frequency-domain S-parameter spectrum of shape
            ``(Nf,)``.
        excited : (str, int)
            ``(port_name, mode_idx)`` of the source for this run.
            Must be a key of ``s_dict``.
        f_axis : np.ndarray
            Frequency axis the spectra were sampled at, shape ``(Nf,)``.
        channel_order : tuple of (str, int), optional
            Override the canonical ordering of observed channels.
            Default: keys of ``s_dict`` in their dict-iteration order
            (insertion-order in CPython ≥ 3.7).
        reference_impedances : dict, optional
            Per-channel real reference impedance on ``f_axis``.
        """
        if not s_dict:
            raise ValueError("s_dict is empty.")
        if excited not in s_dict:
            raise KeyError(f"excited {excited!r} not in s_dict; available: {sorted(s_dict.keys())}")
        if channel_order is None:
            channel_order = tuple(s_dict.keys())
        else:
            channel_order = tuple(channel_order)
            for key in channel_order:
                if key not in s_dict:
                    raise KeyError(f"channel_order entry {key!r} not in s_dict.")
            if len(channel_order) != len(s_dict):
                raise ValueError(
                    "channel_order must enumerate every key of s_dict "
                    f"({len(s_dict)} keys, channel_order has "
                    f"{len(channel_order)})."
                )

        f = np.asarray(f_axis, dtype=float)
        n_ch = len(channel_order)
        matrix = np.empty((f.size, n_ch, 1), dtype=complex)
        for i, key in enumerate(channel_order):
            spectrum = np.asarray(s_dict[key])
            if spectrum.shape != (f.size,):
                raise ValueError(
                    f"s_dict[{key!r}] has shape {spectrum.shape}; "
                    f"expected ({f.size},) to match f_axis."
                )
            matrix[:, i, 0] = spectrum

        return cls(
            f_axis=f.copy(),
            channels=channel_order,
            excitations=(excited,),
            matrix=matrix,
            reference_impedances=reference_impedances,
        )

    @classmethod
    def from_multiple_excitations(
        cls,
        runs: list[tuple[tuple[str, int], dict[tuple[str, int], np.ndarray]]],
        f_axis: np.ndarray,
        *,
        channel_order: tuple[tuple[str, int], ...] | None = None,
    ) -> "SParameterResult":
        """Aggregate K single-excitation runs into a multi-column result.

        Each ``runs[k] = (excited_k, s_dict_k)`` contributes one column
        of the S-matrix at column index ``k``.  All ``s_dict_k`` must
        share the same observed-channel set (otherwise the resulting
        matrix would have NaN holes — refuse rather than fill).

        Parameters
        ----------
        runs : list of ((str, int), dict[(str, int), np.ndarray])
            List of (excited, s_dict) pairs, in the order to assemble
            the columns.  All s_dicts must share the same key set.
        f_axis : np.ndarray
            Frequency axis common to all runs, shape ``(Nf,)``.
        channel_order : tuple of (str, int), optional
            Override the canonical observed-channel ordering.  Must
            enumerate every key shared by all s_dicts.  Default: keys
            of the first s_dict in iteration order.
        """
        if not runs:
            raise ValueError("runs list is empty.")

        first_keys = set(runs[0][1].keys())
        for k, (_, s_dict) in enumerate(runs):
            if set(s_dict.keys()) != first_keys:
                raise ValueError(
                    f"runs[{k}] has channel set "
                    f"{sorted(set(s_dict.keys()))}, but runs[0] has "
                    f"{sorted(first_keys)}.  All single-excitation runs "
                    f"must observe the same channels."
                )

        if channel_order is None:
            channel_order = tuple(runs[0][1].keys())
        else:
            channel_order = tuple(channel_order)
            for key in channel_order:
                if key not in first_keys:
                    raise KeyError(f"channel_order entry {key!r} not in s_dict keys.")
            if len(channel_order) != len(first_keys):
                raise ValueError(
                    "channel_order must enumerate every shared key "
                    f"({len(first_keys)} keys, channel_order has "
                    f"{len(channel_order)})."
                )

        excitations = tuple(exc for exc, _ in runs)
        if len(set(excitations)) != len(excitations):
            raise ValueError(f"runs contain duplicate excitations: {excitations}")
        for exc in excitations:
            if exc not in first_keys:
                raise ValueError(
                    f"excitation {exc!r} not present in observed-channel set {sorted(first_keys)}."
                )

        f = np.asarray(f_axis, dtype=float)
        n_ch = len(channel_order)
        n_ex = len(excitations)
        matrix = np.empty((f.size, n_ch, n_ex), dtype=complex)
        for j, (_, s_dict) in enumerate(runs):
            for i, key in enumerate(channel_order):
                spectrum = np.asarray(s_dict[key])
                if spectrum.shape != (f.size,):
                    raise ValueError(
                        f"runs[{j}] s_dict[{key!r}] shape {spectrum.shape} != ({f.size},)."
                    )
                matrix[:, i, j] = spectrum

        return cls(
            f_axis=f.copy(),
            channels=channel_order,
            excitations=excitations,
            matrix=matrix,
        )

    @classmethod
    def merge(
        cls,
        results: list["SParameterResult"],
    ) -> "SParameterResult":
        """Combine K single-excitation :class:`SParameterResult` objects.

        All inputs must share the same ``f_axis`` and the same
        ``channels`` ordering.  Each input's single excitation becomes
        one column of the merged matrix; column order follows the
        input list.
        """
        if not results:
            raise ValueError("results list is empty.")
        first = results[0]
        for k, r in enumerate(results):
            if not np.array_equal(r.f_axis, first.f_axis):
                raise ValueError(f"results[{k}].f_axis differs from results[0].f_axis.")
            if r.channels != first.channels:
                raise ValueError(
                    f"results[{k}].channels {r.channels} differs from "
                    f"results[0].channels {first.channels}."
                )

        all_excitations: list[tuple[str, int]] = []
        for r in results:
            all_excitations.extend(r.excitations)
        if len(set(all_excitations)) != len(all_excitations):
            raise ValueError(f"merge: duplicate excitations across inputs: {all_excitations}")

        n_ex_total = sum(r.n_excitations for r in results)
        matrix = np.empty(
            (first.n_frequencies, first.n_channels, n_ex_total),
            dtype=complex,
        )
        col = 0
        for r in results:
            matrix[:, :, col : col + r.n_excitations] = r.matrix
            col += r.n_excitations

        refs = first.reference_impedances
        if refs is not None:
            for k, r in enumerate(results[1:], start=1):
                if r.reference_impedances is None:
                    refs = None
                    break
                for key, z in refs.items():
                    if not np.allclose(
                        r.reference_impedances[key], z, rtol=1e-9, atol=0.0, equal_nan=True
                    ):
                        raise ValueError(
                            f"results[{k}] references channel {key!r} to a different "
                            "impedance than results[0]; the runs are not on the same ports."
                        )
        return cls(
            f_axis=first.f_axis.copy(),
            channels=first.channels,
            excitations=tuple(all_excitations),
            matrix=matrix,
            reference_impedances=refs,
        )
