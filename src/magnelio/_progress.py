"""Progress reporting for the operations that take real time.

Three operations dominate the wall time of a typical run and used to
produce no output at all until the time-domain loop started: the mesh
build, the CFL eigenvalue (a Lanczos iteration over the whole update
operator), and the per-port mode solve.  On a 2 M-cell model that is
some 46 s of silence before the first line appears, which is long
enough that a *hung* process is indistinguishable from a working one.

This module holds the one reporter every long-running operation talks
to, so that the reporting policy — where the verbosity setting comes
from, whether the line is overwritten or appended, how often it
refreshes, who stays silent — lives in one place instead of being
re-decided at each print site.

Policy
------
* **Verbosity** is a process-wide default (:func:`set_verbosity`) that
  any object overrides locally with its own ``verbose=`` argument.  The
  default carries into nested work: a port refinement that meshes and
  solves ports per rung passes its own setting down instead of
  silencing the inner calls outright.
* **A terminal gets one overwritten line** (``\\r``), the way the
  time-domain loop has always reported.  Anything else — a log file, a
  CI job, a captured pipe — gets whole lines at a slow cadence, because
  carriage returns collapse a log into one unreadable row.
* **Worker processes stay silent.**  The section engine and the band
  kernel run over process pools; eight workers writing to one terminal
  interleave into noise.  Only the parent reports.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TextIO

__all__ = ["Reporter", "current_reporter", "get_verbosity", "set_verbosity"]

# Minimum seconds between two refreshes of a progress line.  A terminal
# can take a few per second without flicker; a log file wants one line
# per phase plus the occasional heartbeat on a long one.
_TTY_INTERVAL = 0.1
_LOG_INTERVAL = 30.0

# A phase that finishes faster than this is not worth a line of its
# own.  Without it a small model — every tutorial, every test —
# prints one `done (0.0 s)` per phase and says nothing with any of
# them, which is the same defect as a `Tile skip: 0.0%` notice.
_MIN_REPORTED = 0.5

_verbosity: bool = True

# The reporter of the operation currently running.  The mesh build's
# expensive phase reaches the section engine through four call levels
# that have no interest in reporting — threading a parameter through
# all of them would put an output concern into signatures that exist
# for geometry.  A context variable keeps it out of them and is safe
# across threads and async tasks, which a module global is not.
_current: ContextVar["Reporter | None"] = ContextVar("magnelio_reporter", default=None)


def current_reporter() -> "Reporter | None":
    """The reporter of the operation in progress, if one is reporting.

    Returns ``None`` outside a reported operation and inside a worker
    process, so a caller can simply do ``rep = current_reporter()`` and
    guard on it.
    """
    rep = _current.get()
    return rep if rep is not None and rep.enabled else None


def set_verbosity(value: bool) -> None:
    """Set the process-wide default for solver progress output.

    Every analysis, mesh build and port solve reports its progress
    unless told otherwise.  This sets the default they all read; an
    individual object still overrides it with its own ``verbose=``
    argument.

    Parameters
    ----------
    value : bool
        ``True`` to report progress (the default), ``False`` to run
        silently.

    Examples
    --------
    >>> import magnelio as mio
    >>> mio.set_verbosity(False)          # a batch sweep, no output
    >>> mio.set_verbosity(True)           # back to the default
    """
    global _verbosity
    _verbosity = bool(value)


def get_verbosity() -> bool:
    """Return the process-wide default for solver progress output.

    Returns
    -------
    bool
    """
    return _verbosity


def _in_worker_process() -> bool:
    """True inside a multiprocessing worker, where output would interleave."""
    import multiprocessing  # noqa: PLC0415

    return multiprocessing.parent_process() is not None


class Reporter:
    """Progress line for one long-running operation.

    A reporter owns a label (``"mesh"``, ``"ports"``, ``"FIT-TD"``) and
    prints ``  <label> | <phase> | <detail>``.  On a terminal the line
    is overwritten in place; elsewhere each update is a whole line at a
    slow cadence.

    Parameters
    ----------
    label : str
        Operation name, printed at the start of every line.
    verbose : bool, optional
        Local override.  ``None`` (the default) reads the process-wide
        setting at construction time.
    stream : file-like, optional
        Destination; defaults to ``sys.stdout``.
    """

    def __init__(
        self,
        label: str,
        verbose: bool | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.label = label
        self._stream = stream if stream is not None else sys.stdout
        wanted = get_verbosity() if verbose is None else bool(verbose)
        self.enabled = wanted and not _in_worker_process()
        # isatty() is absent on some file-likes (and raises on a closed
        # one); anything that cannot answer is treated as a log.
        try:
            self._tty = bool(self._stream.isatty())
        except (AttributeError, ValueError):
            self._tty = False
        self._interval = _TTY_INTERVAL if self._tty else _LOG_INTERVAL
        self._phase: str | None = None
        self._phase_t0 = 0.0
        self._last_emit = time.perf_counter()
        self._t0 = self._last_emit
        self._token = _current.set(self)
        self._counter = 0
        self._dirty = False  # an unterminated \r line is pending

    # ── line plumbing ───────────────────────────────────────────────

    def _write(self, text: str, *, overwrite: bool) -> None:
        if overwrite and self._tty:
            # Trailing blanks erase the tail of a longer previous line.
            self._stream.write(f"\r  {text}          ")
        elif self._dirty:
            # A final line replaces the running one in place, rather
            # than leaving a half-written row above it.
            self._stream.write(f"\r  {text}          \n")
            self._dirty = False
        else:
            self._stream.write(f"  {text}\n")
        self._stream.flush()
        self._dirty = overwrite and self._tty

    def _erase_line(self) -> None:
        """Drop the running line without leaving it on screen."""
        if self._dirty:
            self._stream.write("\r" + " " * 78 + "\r")
            self._stream.flush()
            self._dirty = False

    def _end_line(self) -> None:
        if self._dirty:
            self._stream.write("\n")
            self._stream.flush()
            self._dirty = False

    def _due(self, now: float) -> bool:
        return now - self._last_emit >= self._interval

    # ── public surface ──────────────────────────────────────────────

    def note(self, text: str) -> None:
        """Print one standalone line, never overwritten."""
        if not self.enabled:
            return
        self._write(f"{self.label} | {text}", overwrite=False)

    @contextmanager
    def phase(self, name: str):
        """Announce a phase, and report its duration when it ends.

        The phase name prefixes every :meth:`step` line inside it.  A
        phase that raises still closes its line, so an exception
        traceback does not start mid-row.
        """
        if not self.enabled:
            yield self
            return
        self._phase = name
        self._phase_t0 = time.perf_counter()
        self._last_emit = self._phase_t0
        self._counter = 0
        self._write(f"{self.label} | {name}", overwrite=True)
        try:
            yield self
        except BaseException:
            self._end_line()
            self._phase = None
            raise
        dt = time.perf_counter() - self._phase_t0
        if dt >= _MIN_REPORTED:
            self._write(f"{self.label} | {name} | done ({dt:.1f} s)", overwrite=False)
        else:
            self._erase_line()
        self._phase = None

    def stage(self, name: str) -> None:
        """Begin a phase in a linear sequence, closing the previous one.

        The context-manager form (:meth:`phase`) suits a block that
        nests; a long procedure that walks through named steps in one
        scope uses this instead, so that instrumenting it does not
        re-indent the body.  :meth:`finish` closes the last stage.
        """
        if not self.enabled:
            return
        self._close_stage()
        self._phase = name
        self._phase_t0 = time.perf_counter()
        self._last_emit = self._phase_t0
        self._counter = 0
        self._write(f"{self.label} | {name}", overwrite=True)

    def _close_stage(self) -> None:
        if self._phase is None:
            return
        dt = time.perf_counter() - self._phase_t0
        if dt >= _MIN_REPORTED:
            self._write(f"{self.label} | {self._phase} | done ({dt:.1f} s)", overwrite=False)
        else:
            # On a terminal the announcement disappears with the phase;
            # in a log it has already been written and stands as the
            # only live sign that the phase ran at all.
            self._erase_line()
        self._phase = None

    def finish(self, text: str = "") -> None:
        """Close the running stage and, optionally, print a summary line.

        The summary carries the wall time of the whole operation, so
        that a run with several reported phases needs no mental
        addition.
        """
        if not self.enabled:
            return
        self._close_stage()
        if text:
            total = time.perf_counter() - self._t0
            if total >= _MIN_REPORTED:
                text = f"{text} ({total:.1f} s total)"
            self._write(f"{self.label} | {text}", overwrite=False)
        self._end_line()
        self._deactivate()

    def step(self, done: int, total: int, detail: str = "") -> None:
        """Report ``done`` of ``total`` units inside the current phase.

        Refreshes at most every :data:`_TTY_INTERVAL` seconds on a
        terminal, every :data:`_LOG_INTERVAL` seconds elsewhere, so a
        tight loop can call this on every iteration.  The final unit
        always prints.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if not self._due(now) and done < total:
            return
        self._last_emit = now
        pct = 100.0 * done / total if total else 100.0
        head = f"{self.label} | {self._phase} | " if self._phase else f"{self.label} | "
        tail = f" | {detail}" if detail else ""
        self._write(f"{head}{done}/{total} ({pct:.0f} %){tail}", overwrite=True)

    def tick(self, count: int, detail: str = "") -> None:
        """Report an open-ended count — work whose total is not known.

        Iterative solvers converge when they converge; ``count`` is a
        sign of life, not a fraction of the way there.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if not self._due(now):
            return
        self._last_emit = now
        head = f"{self.label} | {self._phase} | " if self._phase else f"{self.label} | "
        tail = f" {detail}" if detail else ""
        self._write(f"{head}{count}{tail}", overwrite=True)

    def advance(self, n: int = 1, detail: str = "") -> None:
        """Count *n* more units of open-ended work inside this phase.

        A phase built from several passes over different quantities has
        no denominator that stays meaningful across them — a percentage
        would run to 100 and start again, which reads as a stall and a
        restart.  The running total does not lie about how far along
        the phase is, because it does not claim to know.
        """
        if not self.enabled:
            return
        self._counter += n
        self.tick(self._counter, detail)

    def line(self, text: str) -> None:
        """Overwrite the running line with *text*, subject to the cadence.

        For callers that compose their own status text — the
        time-domain loop reports a step count against a stop criterion,
        which is neither a fraction nor a plain count.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if not self._due(now):
            return
        self._last_emit = now
        self._write(f"{self.label} | {text}", overwrite=True)

    def final(self, text: str) -> None:
        """Close the running line with *text*, always printed.

        This is the last line of the operation, so the reporter stands
        down here as :meth:`finish` would.  A march has several exits
        — each stop criterion, the graceful stop, the runtime cap —
        and every one of them ends on this call.
        """
        if not self.enabled:
            return
        self._write(f"{self.label} | {text}", overwrite=False)
        self._last_emit = time.perf_counter()
        self._deactivate()

    def close(self) -> None:
        """Terminate a pending overwritten line and stand down."""
        if self.enabled:
            self._end_line()
        self._deactivate()

    def _deactivate(self) -> None:
        """Restore whichever reporter was current before this one."""
        if self._token is None:
            return
        try:
            _current.reset(self._token)
        except ValueError:
            # Reset from a different context than the set; the value
            # there is not ours to restore.
            pass
        self._token = None
