"""Analysis workflows — the high-level problem-class API.

Each ``Analysis*`` class solves a specific physical question in one
``run()`` call:

* :class:`AnalysisEigenmode` — resonant eigenmodes of a closed cavity.
* :class:`AnalysisScatteringTD` — S-parameters of a multi-port network
  via FIT time-domain simulation.

:func:`resume` continues a time-domain run persisted in a project store
 from its last checkpoint — see the function docstring.
"""

from magnelio.analysis.eigenmode import AnalysisEigenmode
from magnelio.analysis.result_interface import (
    RunSettings,
    ScatteringResult,
    ScatteringResultMixin,
)
from magnelio.analysis.scattering_td import AnalysisScatteringTD, ScatteringTDResult

__all__ = [
    "ScatteringTDResult",
    "RunSettings",
    "ScatteringResult",
]


def resume(
    project,
    excited=None,
    *,
    energy_stop_db: float | None = None,
    total_time_steps: int | None = None,
    port_signal_stop_db: float | str | None = None,
    max_time_steps: int | str | None = "auto",
    checkpoint_interval: int | None = None,
    verbose: bool = True,
):
    """Continue a time-domain run in a project store from its checkpoint.

    Opens ``project`` (a path or an already-open
    :class:`~magnelio.io.project.Project`), rebuilds the run's operators
    from the stored reconstruction recipe, loads the latest
    ``checkpoint.h5``, and marches on — appending to the same
    ``results.h5`` streams.  The resume injects no seam:
    the checkpoint carries the full leapfrog state (E/H, CPML ψ, exact
    DTBC convolution history, Mur previous-values), so a resumed run is
    bit-identical to an uninterrupted run of the same total length on a
    deterministically-built line.

    Parameters
    ----------
    project : str or Path or Project
        The project directory (or an open reader) to continue.
    excited : str or (str, int), optional
        Which run to resume, by its excited ``(port, mode)`` pair.  May
        be omitted when the project holds exactly one run.
    energy_stop_db : float, optional
        New stop threshold, dB below the *original* peak energy.  Default
        (``None``): inherit the run's launch criterion — so a Ctrl-C-
        aborted run finishes to its original target with a bare
        ``resume(project)``.  To run a completed run *longer*, pass a
        deeper (larger) value than it originally used.
    total_time_steps : int, optional
        New global step cap (not a delta — the target leapfrog count).
        Default (``None``): inherit the run's launch value.  Must exceed
        the checkpoint step.
    port_signal_stop_db : float or "auto", optional
        New port-signal stop threshold, dB below the run's peak modal
        ``|V|`` envelope (see :meth:`AnalysisScatteringTD.run`).  Default
        (``None``): inherit the run's launch criterion together with the
        other two knobs; passing any knob explicitly disables the
        others, exactly as at launch.
    max_time_steps : int, None or "auto", default "auto"
        Runtime cap for the resumed segment (see
        :meth:`AnalysisScatteringTD.run`).  ``"auto"`` grants a fresh
        cap budget past the checkpoint step; an explicit int is an
        absolute step bound; ``None`` removes the cap and the stall
        watchdog.  Not inherited — each segment gets its own.
    checkpoint_interval : int, optional
        Minimum steps between resume checkpoints for the continued run
        (as in :meth:`AnalysisScatteringTD.run`).
    verbose : bool, default True
        Print solver progress.

    Returns
    -------
    Project
        A reader over the (now-extended) project.

    Raises
    ------
    ValueError
        If the project carries no reconstruction recipe (written by an
        older version without one), the
        run has no checkpoint, or the effective criterion would not
        advance past the checkpoint step.
    NotImplementedError
        If the project's analysis kind has no time-marching state to
        resume (e.g. an eigenmode result).
    """
    from magnelio.io.project import Project, open_project

    proj = project if isinstance(project, Project) else open_project(project)
    kind = proj.setup.get("analysis")
    if kind == "AnalysisScatteringTD":
        from magnelio.analysis.scattering_td import _resume_scattering

        return _resume_scattering(
            proj,
            excited,
            energy_stop_db=energy_stop_db,
            total_time_steps=total_time_steps,
            port_signal_stop_db=port_signal_stop_db,
            max_time_steps=max_time_steps,
            checkpoint_interval=checkpoint_interval,
            verbose=verbose,
        )
    if kind is None:
        raise ValueError(
            f"project {proj.path} has no analysis metadata; cannot resume.",
        )
    raise NotImplementedError(
        f"resume is not defined for analysis kind {kind!r}: only time-domain "
        f"runs (AnalysisScatteringTD) carry the leapfrog state a resume "
        f"continues.  An eigenmode result has none — re-run AnalysisEigenmode "
        f"with more modes / a different sigma instead.",
    )
