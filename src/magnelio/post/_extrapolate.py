"""Continue a truncated port record past the end of the march.

A time-domain run has to stop somewhere.  What it leaves behind on a
high-Q structure is a record whose transient is still ringing, and the
DFT of such a record carries the truncation as ripple on every
S-parameter — the effect
:func:`~magnelio.analysis.scattering_td._warn_on_truncated_band_record`
warns about.  The remedy is older than FDTD: past the excitation the
record is the structure's own free decay, a sum of damped exponentials
whose poles are the structure's resonances, so a handful of poles
fitted to what *was* recorded continues it for as long as one likes.

The fit is the matrix pencil on a Hankel matrix of the record: an SVD
picks the order from the singular-value decay, a generalised eigenvalue
problem gives the per-step growth factors, and the residues follow by
least squares.  All the channels of one excitation share one pole set —
they are one structure's resonances, seen through different ports — so
the Hankel blocks are stacked and the pencil solved once.  Poles on or
outside the unit circle cannot be a passive structure's free decay and
are dropped.

The quality number is an out-of-sample one: the model is fitted on the
first half of the window and asked to predict the second, and the
relative rms of that prediction is what
:class:`ExtrapolationReport` reports.  A Prony fit always looks perfect
on the data it was fitted to; only the prediction says whether the
poles are the structure's or the noise's.
"""

# Design: DD-272 (extrapolating a truncated record to infinite decay).

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Above this out-of-sample error the fit is not describing resonances.
_RESIDUAL_WARN = 0.3


@dataclass(frozen=True)
class ExtrapolationReport:
    """What the continuation of one excitation's records did.

    Attributes
    ----------
    order : int
        Model order the singular values selected (or the one asked
        for): the number of poles fitted, two per resonance.
    dropped : int
        Poles discarded for lying on or outside the unit circle — a
        growing exponential, which a passive structure's free decay
        cannot contain.
    residual : float
        Relative rms of the out-of-sample prediction: the model fitted
        on the first half of the fit window, measured against the
        recorded second half.  Below about 1e-2 the poles are the
        structure's; approaching 1 the fit is describing noise.
    fit_start : float
        Time [s] the fit window opens — after the excitation has died
        away, since only then is the record a free decay.
    added : float
        Time [s] appended to every record.
    capped : bool
        True when *max_factor* stopped the continuation before the
        model had decayed by ``decay_db`` — the record is then still
        truncated, only less so, and the cap should be raised.
    frequencies, decay_times : np.ndarray
        The fitted resonances: frequency [Hz] and time constant [s] of
        every pole with a positive imaginary part, strongest residue
        first.  A pole pair with a decay time far beyond the run is
        exactly the one the truncation cut off.
    """

    order: int
    dropped: int
    residual: float
    fit_start: float
    added: float
    capped: bool
    frequencies: np.ndarray
    decay_times: np.ndarray

    def __repr__(self) -> str:
        head = (
            f"ExtrapolationReport(order={self.order}, dropped={self.dropped}, "
            f"residual={self.residual:.3g}, added={self.added * 1e9:.4g} ns"
            f"{', capped' if self.capped else ''})"
        )
        lines = [head]
        for f, tau in zip(self.frequencies[:8], self.decay_times[:8]):
            lines.append(f"  {f / 1e9:10.5g} GHz   tau = {tau * 1e9:10.5g} ns")
        if self.frequencies.size > 8:
            lines.append(f"  … {self.frequencies.size - 8} more")
        return "\n".join(lines)


def _hankel_stack(rows: np.ndarray, pencil: int) -> np.ndarray:
    """The stacked Hankel matrix of several equally sampled signals."""
    n = rows.shape[1]
    idx = np.arange(n - pencil)[:, None] + np.arange(pencil + 1)[None, :]
    return np.vstack([row[idx] for row in rows])


def fit_poles(rows, *, pencil: int | None = None, order: int | None = None, tol: float = 1e-4):
    """Common discrete-time poles of several decaying records.

    Parameters
    ----------
    rows : array_like, shape (n_signals, n_samples)
        The records, sampled on one time axis.
    pencil : int, optional
        Pencil parameter L (default N/3, the usual robust choice).
    order : int, optional
        Model order; default from the singular-value decay.
    tol : float
        Relative singular-value threshold for that choice.

    Returns
    -------
    z : np.ndarray
        Per-step growth factors, unfiltered.
    order : int
        The order used.
    """
    x = np.atleast_2d(np.asarray(rows, dtype=float))
    n = x.shape[1]
    if n < 8:
        raise ValueError(f"a record of {n} samples is too short to continue")
    L = int(pencil) if pencil is not None else n // 3
    L = max(2, min(L, n - 2))
    y = _hankel_stack(x, L)
    _u, s, vh = np.linalg.svd(y, full_matrices=False)
    if order is None:
        keep = int(np.sum(s > float(tol) * s[0]))
    else:
        keep = int(order)
    keep = max(1, min(keep, L - 1))
    v = vh[:keep].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(v[:-1]) @ v[1:])
    return z, keep


def residues(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Least-squares residues of one record for the given poles."""
    n = np.asarray(values).size
    basis = z[None, :] ** np.arange(n)[:, None]
    r, *_ = np.linalg.lstsq(basis, np.asarray(values), rcond=None)
    return r


def continuation(n_from: int, n_extra: int, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """The model's samples ``n_from … n_from + n_extra`` (real part)."""
    basis = z[None, :] ** (n_from + np.arange(n_extra))[:, None]
    return np.real(basis @ r)


def _stable(z: np.ndarray) -> np.ndarray:
    """The poles a passive free decay can hold: strictly inside the circle."""
    return z[np.abs(z) < 1.0 - 1e-12]


def excitation_end(reference, floor_db: float = 60.0) -> int:
    """Index past which the excitation waveform is below *floor_db* of its peak.

    Only after this is the record a free decay, which is what the pole
    model describes; fitting across the driven part would try to
    express the source's own shape in the structure's resonances.
    """
    v = np.abs(np.asarray(reference, dtype=float))
    peak = float(v.max()) if v.size else 0.0
    if peak <= 0.0:
        return 0
    above = np.nonzero(v > peak * 10.0 ** (-float(floor_db) / 20.0))[0]
    return int(above[-1]) + 1 if above.size else 0


def _out_of_sample_residual(rows: np.ndarray, *, order: int | None, tol: float) -> float:
    """Fit on the first half of *rows*, measure against the second.

    The only honest quality number for a pole fit: a Prony model
    reproduces the samples it was fitted to almost by construction, so
    what says whether the poles are the structure's is a prediction it
    has not seen.
    """
    half = rows.shape[1] // 2
    if half < 8:
        return float("nan")
    z, _ = fit_poles(rows[:, :half], order=order, tol=tol)
    z = _stable(z)
    if z.size == 0:
        return float("inf")
    err = 0.0
    ref = 0.0
    for row in rows:
        r = residues(row[:half], z)
        pred = continuation(half, rows.shape[1] - half, z, r)
        err += float(np.sum((pred - row[half:]) ** 2))
        ref += float(np.sum(row[half:] ** 2))
    return float(np.sqrt(err / ref)) if ref > 0.0 else float("nan")


def _samples_to_add(z: np.ndarray, r_max: float, peak: float, decay_db: float, cap: int) -> int:
    """How far to continue: until the slowest pole is *decay_db* below the peak.

    The slowest pole sets the horizon; the amplitude that has to fall is
    the largest residue of the record.  Capped, because a pole a hair
    inside the unit circle would otherwise ask for an unbounded record.
    """
    if z.size == 0 or peak <= 0.0 or r_max <= 0.0:
        return 0
    slowest = float(np.max(np.abs(z)))
    if slowest <= 0.0 or slowest >= 1.0:
        return cap
    target = peak * 10.0 ** (-float(decay_db) / 20.0)
    n = np.log(target / r_max) / np.log(slowest)
    if not np.isfinite(n) or n <= 0.0:
        return 0
    return int(min(cap, np.ceil(n)))


def _resonances(z: np.ndarray, r: np.ndarray, dt: float):
    """Frequency [Hz] and decay time [s] of every pole with ω > 0, strongest first."""
    s = np.log(z.astype(complex)) / float(dt)
    keep = s.imag > 0.0
    s, weight = s[keep], np.abs(r)[keep] if r.size == z.size else np.abs(s.imag)[keep]
    order = np.argsort(-weight)
    s = s[order]
    with np.errstate(divide="ignore"):
        tau = np.where(s.real < 0.0, -1.0 / s.real, np.inf)
    return s.imag / (2.0 * np.pi), tau


def extend_records(
    records,
    dt: float,
    *,
    fit_from: int = 0,
    order: int | None = None,
    tol: float = 1e-4,
    decay_db: float = 80.0,
    max_factor: float = 100.0,
):
    """Continue several records of one excitation past their end.

    Parameters
    ----------
    records : sequence of np.ndarray
        The recorded samples, all on one time axis.  Every one is
        continued; they share one pole set, being one structure's
        resonances seen through different ports.
    dt : float
        Time step [s].
    fit_from : int
        First sample of the fit window — past the excitation
        (:func:`excitation_end`), where the record is a free decay.
    order, tol : int or None, float
        Model order, or the relative singular-value threshold that
        chooses it.
    decay_db : float
        Continue until the model has fallen this far below the peak of
        the record.
    max_factor : float
        Cap on the continuation, in multiples of the recorded length.
        It has to be generous: the case this method exists for is a run
        cut short, where the decay left to model is many times what was
        recorded.

    Returns
    -------
    extended : list of np.ndarray
    report : ExtrapolationReport
    """
    rows = np.array([np.asarray(v, dtype=float) for v in records])
    if rows.ndim != 2:
        raise ValueError("every record must be a 1-D array of the same length")
    window = rows[:, int(fit_from) :]
    if window.shape[1] < 16:
        raise ValueError(
            f"the fit window holds {window.shape[1]} samples; a pole fit needs at least 16 "
            "(the run stopped almost as soon as the excitation ended — there is no free "
            "decay to continue)"
        )
    z_all, used = fit_poles(window, order=order, tol=tol)
    z = _stable(z_all)
    if z.size == 0:
        raise ValueError(
            "no decaying pole survived the fit: every fitted pole grows with time, which a "
            "passive structure's free decay cannot do (the window is probably noise)"
        )
    residual = _out_of_sample_residual(window, order=order, tol=tol)
    res = [residues(row, z) for row in window]
    peak = float(np.max(np.abs(rows))) if rows.size else 0.0
    r_max = float(max(np.max(np.abs(r)) for r in res))
    cap = int(max_factor * rows.shape[1])
    n_extra = _samples_to_add(z, r_max, peak, decay_db, cap)
    capped = n_extra >= cap
    extended = [
        np.concatenate([row, continuation(window.shape[1], n_extra, z, r)])
        for row, r in zip(rows, res)
    ]
    if residual > _RESIDUAL_WARN:
        warnings.warn(
            f"the pole model predicts the recorded second half of its own fit window "
            f"with a relative error of {residual:.2f} — at that level it is describing "
            f"noise or a delay line, not a few resonances, and the continuation is not "
            f"an extrapolation to trust.  A record dominated by discrete echoes rather "
            f"than by ringing has no small pole set to find; run longer instead.",
            UserWarning,
            stacklevel=3,
        )
    freqs, taus = _resonances(z, res[int(np.argmax([np.max(np.abs(r)) for r in res]))], dt)
    report = ExtrapolationReport(
        order=int(used),
        dropped=int(z_all.size - z.size),
        residual=float(residual),
        fit_start=float(fit_from) * float(dt),
        added=float(n_extra) * float(dt),
        capped=bool(capped),
        frequencies=freqs,
        decay_times=taus,
    )
    return extended, report
