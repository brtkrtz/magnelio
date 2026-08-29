"""SourceFieldInitial — E and H at t = 0, the start of a transient run.

An initial field turns a time-domain march into an initial-value
problem: the cavity rings down from an eigenmode, a field recorded
elsewhere continues in a new geometry.  The source carries a
:class:`~magnelio.fields.FieldState` and writes it into the solver's
state once, in :meth:`attach`; the leapfrog half-step of H is derived
from the discrete Faraday law so that a mode of the discrete operator
starts as exactly that mode.

Nothing about the model restricts this.  Where the run carries state
besides the fields — an absorber's convolutions, the pole currents of
a dispersive material, the branch currents of a surface-impedance
wall, a port's boundary history — that state starts at zero, which is
the quiescent condition: the absorber is empty, the material
unpolarised, the wall carries no current, the exterior of a port was
quiet.  That is a well-defined initial-value problem, and a stable one
(measured), but it is not the *steady* state a mode would have built
up, so a start of that kind is worth a thought.  Whether the field you
load is the one you meant is your call; the field the source is fed is
exactly the field it marches.
"""

# Design: DD-224 (sources on the model, Phase C).

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from magnelio.fields import FieldState
from magnelio.mesh.grid import GridLines
from magnelio.signals.waveforms import Waveform
from magnelio.sources.base import Source


def _to_host(a):
    return a.get() if type(a).__module__.partition(".")[0] == "cupy" else np.asarray(a)


def _same_grid(a: GridLines, b: GridLines) -> bool:
    return all(
        p.shape == q.shape and np.allclose(p, q, rtol=0.0, atol=1e-12 * max(1.0, abs(q[-1])))
        for p, q in ((a.x, b.x), (a.y, b.y), (a.z, b.z))
    )


@dataclass
class SourceFieldInitial(Source):
    """The field at t = 0 of a time-domain run.

    Declared on the model with :meth:`~magnelio.GeometryModel.add_source`
    (or passed as ``sources=`` to the analysis) and named by an
    :class:`~magnelio.Excitation` whose ``amplitude`` scales the field
    (unit ``"1"``); it has no waveform, delay or phase.  Build one with
    :meth:`from_project`, :meth:`from_function` or :meth:`from_arrays`.

    Parameters
    ----------
    name : str
        Source name — the handle an :class:`~magnelio.Excitation` uses.
    field : FieldState
        E [V/m] and H [A/m] at t = 0 on the Yee positions.  A field on
        a grid other than the run's is resampled onto the run's grid
        by trilinear interpolation per component (which does not
        preserve the discrete divergence exactly — prefer the run's
        own grid).

    Examples
    --------
    >>> ring = sources.SourceFieldInitial.from_project("cavity_modes", name="mode0", mode=0)
    >>> mesh = mio.Mesh.from_geometry(model, f_max=f_max).with_sources([ring])
    >>> result = mio.AnalysisTD(mesh=mesh).run(excitations=["mode0"], t_end=50e-9)
    """

    name: str
    field: FieldState

    amplitude_unit = "1"
    has_waveform = False
    writes_initial_field = True

    _amplitude: float = dc_field(default=1.0, repr=False, init=False)
    _attached: bool = dc_field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"source name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.field, FieldState):
            raise TypeError(
                f"field must be a magnelio.fields.FieldState; got {type(self.field).__name__}",
            )
        if self.field.is_complex:
            raise ValueError(
                "an initial field must be real: a Bloch mode with a phase advance "
                "other than 0 or 180 degrees has no real time-domain start "
                "(take .real() for its zero-phase snapshot if that is what you want)",
            )

    # ── constructors ─────────────────────────────────────────────────────

    @classmethod
    def from_project(
        cls,
        project,
        *,
        name: str,
        mode: int = 0,
        phase_deg: float = 0.0,
    ) -> SourceFieldInitial:
        """The eigenmode *mode* of a project's stored eigenmode result.

        The mode oscillates as ``E(t) = E_m cos(ωt)``,
        ``H(t) = −H_m sin(ωt)`` with the stored patterns ``E_m``, ``H_m``;
        the start is taken at the phase *phase_deg* of that cycle, so
        the default ``0`` starts at the instant of maximum electric
        field with H = 0.

        Parameters
        ----------
        project : str, Path or Project
            A project directory written by ``AnalysisEigenmode(project=…)``,
            or the opened :class:`~magnelio.io.project.Project`.
        name : str
            Source name.
        mode : int
            Mode index (ascending in frequency).
        phase_deg : float
            Phase ``ωt`` [degrees] of the cycle at which the run starts.
        """
        from magnelio.io.project import Project, open_project  # noqa: PLC0415

        proj = project if isinstance(project, Project) else open_project(project)
        result = proj.eigenmodes
        if result is None:
            raise ValueError(f"project {proj.path} holds no eigenmode result")
        pattern = result.field(mode)
        if pattern.is_complex:
            raise ValueError(
                f"mode {mode} is a complex Bloch mode (phase advance other than 0 or "
                f"180 degrees) and has no real time-domain start",
            )
        phi = math.radians(float(phase_deg))
        c, s = math.cos(phi), -math.sin(phi)
        return cls(
            name=name,
            field=FieldState(
                pattern.grid,
                Ex=pattern.Ex * c,
                Ey=pattern.Ey * c,
                Ez=pattern.Ez * c,
                Hx=pattern.Hx * s,
                Hy=pattern.Hy * s,
                Hz=pattern.Hz * s,
            ),
        )

    @classmethod
    def from_function(cls, grid: GridLines, *, name: str, E=None, H=None) -> SourceFieldInitial:
        """E and H as vector functions of position, sampled on *grid*.

        See :meth:`magnelio.fields.FieldState.from_function`.
        """
        return cls(name=name, field=FieldState.from_function(grid, E=E, H=H))

    @classmethod
    def from_arrays(
        cls,
        grid: GridLines,
        *,
        name: str,
        Ex=None,
        Ey=None,
        Ez=None,
        Hx=None,
        Hy=None,
        Hz=None,
    ) -> SourceFieldInitial:
        """E and H from component arrays on the Yee positions of *grid*.

        A component left ``None`` is zero.  Shapes follow the table in
        :mod:`magnelio.fields`.
        """
        zeros = FieldState.zeros(grid)
        comps = {}
        for k, v in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), (Ex, Ey, Ez, Hx, Hy, Hz)):
            comps[k] = zeros.component(k) if v is None else v
        return cls(name=name, field=FieldState(grid, **comps))

    # ── excitation binding ───────────────────────────────────────────────

    def set_excitation(
        self,
        waveform: Waveform | None,
        *,
        amplitude: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        """Bind the amplitude the field is scaled by; there is no waveform."""
        if waveform is not None:
            raise ValueError(
                f"source {self.name!r} is an initial field and takes no waveform: "
                f"Excitation({self.name!r}, amplitude=…) is the whole drive",
            )
        amplitude = float(amplitude)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite; got {amplitude!r}")
        if float(delay) != 0.0:
            raise ValueError(
                f"source {self.name!r} is an initial field: it exists at t = 0 and "
                f"cannot be delayed (got delay = {delay!r})",
            )
        self._amplitude = amplitude

    def clear_excitation(self) -> None:
        self._amplitude = 1.0

    @property
    def amplitude(self) -> float:
        """The bound scale factor (1 before :meth:`set_excitation`)."""
        return self._amplitude

    # ── solver hooks ─────────────────────────────────────────────────────

    def _resampled(self, grid: GridLines) -> FieldState:
        """The field on *grid* — itself when the grids coincide.

        Each component is interpolated between its own samples, so the
        Yee staggering is honoured; one interpolator per component,
        never a full six-component evaluation per component.
        """
        if _same_grid(self.field.grid, grid):
            return self.field

        from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

        target = FieldState.zeros(grid)
        components = {}
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            interp = RegularGridInterpolator(
                self.field.positions(name),
                self.field.component(name),
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            X, Y, Z = np.meshgrid(*target.positions(name), indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            components[name] = interp(pts).reshape(X.shape)
        return FieldState(grid, **components)

    def attach(self, solver) -> None:
        """Write the scaled field into the solver's state.

        ``e(0)`` is the field on the primal edges (PEC edges zeroed).
        The march holds H half a step *ahead* of E when it enters its
        first E update, so the start is ``h(+dt/2)``, half a discrete
        Faraday step from ``h(0)``: ``h(dt/2) = h(0) − ½·β_H·(C e(0))``.
        For an eigenmode of the discrete operator started at its E
        maximum (``h(0) = 0``) this is the exact leapfrog state of that
        mode, so the run is a pure oscillation of it.

        The contribution is *added* to the solver's state, which the
        solver allocated as zeros: two initial fields excited in the
        same run superpose, as the linear start they are.
        """
        from magnelio._operators.curl import curl_e_stencil  # noqa: PLC0415

        grid = solver.mesh.grid
        field = self._resampled(grid)
        raw = field._raw
        scale = self._amplitude

        e0 = np.asarray(_to_host(raw.e_flat), dtype=np.float64) * scale
        h0 = np.asarray(_to_host(raw.h_flat), dtype=np.float64) * scale
        pec = getattr(solver, "_pec_mask_E", None)
        if pec is not None:
            e0[_to_host(pec).astype(bool)] = 0.0

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        cx = np.empty((Nx + 1, Ny, Nz))
        cy = np.empty((Nx, Ny + 1, Nz))
        cz = np.empty((Nx, Ny, Nz + 1))
        curl_e_stencil(
            e0[:n_Ex].reshape(Nx, Ny + 1, Nz + 1),
            e0[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1),
            e0[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz),
            cx,
            cy,
            cz,
        )
        curl = np.concatenate([cx.ravel(), cy.ravel(), cz.ravel()])
        beta_H = np.asarray(_to_host(solver._beta_H), dtype=np.float64)
        h_half = h0 - 0.5 * beta_H * curl

        xp = solver._xp
        dtype = solver._real_dtype
        fields = solver._fields
        fields.e_flat[:] += xp.asarray(e0.astype(dtype, copy=False))
        fields.h_flat[:] += xp.asarray(h_half.astype(dtype, copy=False))
        self._attached = True

    def inject_E(self, fields, t_E: float) -> None:
        """Nothing to inject — the field was written at t = 0."""

    def inject_H(self, fields, t_H: float) -> None:
        """Nothing to inject — the field was written at t = 0."""

    # ── store payload ────────────────────────────────────────────────────

    def _store_payload(self) -> dict:
        """The field's arrays for ``mesh.h5`` (grid quantities plus grid lines)."""
        raw = self.field._raw
        g = self.field.grid
        out = {"x": np.asarray(g.x), "y": np.asarray(g.y), "z": np.asarray(g.z)}
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            out[c] = _to_host(getattr(raw, c))
        return out

    @classmethod
    def _from_store_payload(cls, d: dict, payload: dict) -> SourceFieldInitial:
        from magnelio._fields.field_arrays import FieldState as _Raw  # noqa: PLC0415

        grid = GridLines(x=payload["x"], y=payload["y"], z=payload["z"])
        raw = _Raw(**{c: np.asarray(payload[c]) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")})
        return cls(name=d["name"], field=FieldState._from_raw(grid, raw))

    def __repr__(self) -> str:
        return f"SourceFieldInitial(name={self.name!r}, field={self.field!r})"


__all__ = ["SourceFieldInitial"]
