"""Wall-loss monitor — perturbative conductor losses from a TD run.

Accumulates a running DFT of the tangential-H samples on PEC walls
(``mesh/surfaces.py`` enumeration) plus the E/H states of one reference
cross-section, and evaluates the frequency-domain wall loss as a
FRACTION of the power flowing through the reference plane:

    fraction(f) = P_loss(f) / P_flow(f)
    P_loss = 1/2 * R_s(f) * sum( w * |H_tan(f)|^2 )
    P_flow = 1/2 * Re( sum e_hat x h_hat^* )      (FIT identity, no areas)

Both P_loss and P_flow are quadratic in the run's field states, so the
global mesh-dependent state scale (the M_eps-basis scale,
pinned at the port recorders but NOT in the volume states) cancels
exactly — the monitor is scale-free and needs no source renormalisation.
Multiply by the incident power to get Watts (``power_loss(P_in=…)``).

Storage is surface-only (~N^2): wall samples + one cross-section, never
a volume.

On an SIBC run (``sibc=`` set) the monitor switches to
the SIBC's OWN loss accounting: the sampled faces and weights are the
operator's update topology (``SIBCSurface`` — its ``h_tan_sq_sum`` IS
the wall term's power booking) and ``R_s(f)`` is the real part of the
same rational ``Z_s`` fit the solver damps with, so the reported loss
is exactly what the SIBC extracted from the field solution — no double
counting against the perturbative chain, and no post-hoc sampling
mismatch.  Reader interface and ``wall_loss.h5`` shape are unchanged.
"""

# Design: DD-082 (perturbative wall-loss chain), DD-078 (M_eps-basis state
# scale), SIBC_PLAN WP-D5 (SIBC-run loss accounting).

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from magnelio.mesh._surfaces import enumerate_pec_surfaces
from magnelio.post.wall_loss import surface_resistance


@dataclass
class MonitorWallLoss:
    """Frequency-domain wall-loss monitor for time-domain runs.

    Parameters
    ----------
    freqs : array_like
        Evaluation frequencies [Hz].
    normal : str
        Normal axis (``"x"``, ``"y"`` or ``"z"``) of the power-reference
        cross-section — a plane between the excited port and the lossy
        walls.
    position : float
        Position of that plane along its normal axis [m]; snapped to
        the nearest grid node.
    sigma : float, optional
        Conductivity [S/m] for walls that are not lossy metals
        (plain-PEC solids and PEC boundary walls); lossy-metal solids
        use their own material values.
    mu : float, optional
        Relative permeability accompanying ``sigma`` (default 1).
    roughness : SurfaceRoughness, optional
        Surface-roughness model for the same walls ``sigma``
        applies to; lossy-metal solids always use their own.  It raises
        R_s per DFT bin, so the reported fraction is frequency-shaped by
        K(f) rather than scaled by a constant.
    bc_faces : tuple[str, ...]
        Domain-boundary faces to treat as PEC walls (``"xmin"`` …).
        Port faces must not be listed.
    name : str
        Monitor label.
    sibc : SIBCSpec, optional
        When set (an SIBC run), the monitor reports the SIBC's
        own dissipated power: surfaces come from the spec's update
        topology and ``R_s(f) = Re Z_s(f)`` from its fits.  ``sigma`` /
        ``mu`` / ``roughness`` / ``bc_faces`` are ignored in that mode
        (the spec already resolved them).  Wired automatically by
        ``AnalysisScatteringTD`` on ``wall_model="sibc"`` runs; not
        part of the recipe (re-derived on resume).
    """

    freqs: np.ndarray
    normal: str
    position: float
    sigma: float | None = None
    mu: float = 1.0
    roughness: object = None
    bc_faces: tuple[str, ...] = ()
    name: str = "wall_loss"
    sibc: object = None
    # DD-099: faces whose registered boundary-plane coverage must not
    # book walls (port planes, non-PEC BCs), and per-face wall
    # conductor overrides from PECBoundary declarations.  Wired at run
    # time by the analysis, like ``sibc`` — not part of the recipe.
    masked_faces: tuple[str, ...] = ()
    wall_overrides: dict = None

    _mesh: object = field(default=None, repr=False, init=False)
    _surfaces: list = field(default_factory=list, repr=False, init=False)
    _omega: np.ndarray | None = field(default=None, repr=False, init=False)
    _h_bins: list = field(default_factory=list, repr=False, init=False)
    _k_ref: int = field(default=0, repr=False, init=False)
    _ref_bins: dict = field(default_factory=dict, repr=False, init=False)
    _gather_idx: list | None = field(default=None, repr=False, init=False)
    # Full-model factor on the dissipated fraction (DD-154): each
    # declared symmetry plane doubles the walls (mirror half) AND the
    # reference cross-section it cuts — those cancel; only a plane
    # parallel to the reference cross-section leaves the reference
    # power unchanged and contributes a factor 2.
    _sym_fraction_factor: float = field(default=1.0, repr=False, init=False)

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float)
        if self.normal not in ("x", "y", "z"):
            raise ValueError(f"normal must be 'x', 'y' or 'z'; got {self.normal!r}")
        self.position = float(self.position)

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            symmetry_entries,
        )
        from magnelio.post.wall_loss import _resolve_surface_materials

        self._mesh = mesh
        self._omega = 2.0 * np.pi * self.freqs

        # DD-154: symmetry faces are mirror planes, not conductor
        # walls — they must not dissipate.  The analysis already masks
        # them; a user-listed bc_face on a symmetry plane is dropped
        # loudly here.
        sym = symmetry_entries(getattr(mesh, "boundary_conditions", None))
        sym_listed = tuple(f for f in self.bc_faces if f in sym)
        if sym_listed:
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"MonitorWallLoss {self.name!r}: bc_faces {sym_listed!r} "
                f"are declared symmetry planes — a mirror plane is not "
                f"a physical wall and books no loss; dropped.",
                stacklevel=2,
            )
            self.bc_faces = tuple(f for f in self.bc_faces if f not in sym)
        ref_axis = self.normal
        n_parallel = sum(1 for face in sym if face[0] == ref_axis)
        self._sym_fraction_factor = float(2**n_parallel)
        if self.sibc is not None:
            # SIBC accounting (WP-D5): the operator's own faces, weights
            # and Z_s fits.  Fail early on a fit-less tag, mirroring the
            # perturbative resolution below.
            self._surfaces = list(self.sibc.surfaces)
            if not self._surfaces:
                raise ValueError("MonitorWallLoss: the SIBC spec carries no wall surfaces")
            for s in self._surfaces:
                if s.tag not in self.sibc.fits:
                    raise ValueError(
                        f"MonitorWallLoss: SIBC spec has no impedance fit for wall tag {s.tag!r}"
                    )
            self._resolved = None
        else:
            self._surfaces = enumerate_pec_surfaces(
                mesh,
                bc_pec_faces=self.bc_faces,
                masked_boundary_faces=self.masked_faces,
            )
            if not self._surfaces:
                raise ValueError("MonitorWallLoss: no PEC wall surfaces found")
            # Fail early on missing conductivities (not at evaluation time).
            self._resolved = _resolve_surface_materials(
                mesh,
                self._surfaces,
                self.sigma,
                self.mu,
                self.roughness,
                overrides=self.wall_overrides or None,
            )
        nf = len(self.freqs)
        self._h_bins = [np.zeros((nf, len(s.comp)), dtype=complex) for s in self._surfaces]
        self._gather_idx = None  # rebuilt lazily on the first record

        axis, pos = self.normal, self.position
        grid = mesh.grid
        nodes = {"x": grid.x, "y": grid.y, "z": grid.z}[axis]
        n_cells = {"x": grid.Nx, "y": grid.Ny, "z": grid.Nz}[axis]
        self._k_ref = min(int(np.argmin(np.abs(np.asarray(nodes) - pos))), n_cells - 1)
        self._ref_bins = {}  # filled lazily with the four state slabs

        # Boundary correction for the FIT power identity: the H states
        # carry the SOLVER dual length (full first/last cell, matching
        # build_M_mu), while the physical dual patch at a domain
        # boundary is the half cell — ratio 1/2 at the two ends of the
        # H component's own axis, 1 inside.
        def _edge_corr(n_nodes: int) -> np.ndarray:
            c = np.ones(n_nodes)
            c[0] = 0.5
            c[-1] = 0.5
            return c

        cx = _edge_corr(grid.Nx + 1)
        cy = _edge_corr(grid.Ny + 1)
        cz = _edge_corr(grid.Nz + 1)
        if axis == "z":
            # pair 1: Ex*Hy on (Nx, Ny+1) — Hy's own axis is y (axis 1)
            # pair 2: Ey*Hx on (Nx+1, Ny) — Hx's own axis is x (axis 0)
            self._ref_corr = (cy[None, :], cx[:, None])
        elif axis == "x":
            # pair 1: Ey*Hz on (Ny, Nz+1) — Hz axis z; pair 2: Ez*Hy — Hy axis y
            self._ref_corr = (cz[None, :], cy[:, None])
        else:
            # pair 1: Ez*Hx on (Nx+1, Nz) — Hx axis x; pair 2: Ex*Hz — Hz axis z
            self._ref_corr = (cx[:, None], cz[None, :])

    def _ref_slabs(self, fields):
        """The four (E1, H2, E2, H1) state slabs of the reference plane."""
        axis = self.normal
        k = self._k_ref
        if axis == "z":  # S_z = Ex*Hy - Ey*Hx
            return (fields.Ex[:, :, k], fields.Hy[:, :, k], fields.Ey[:, :, k], fields.Hx[:, :, k])
        if axis == "x":  # S_x = Ey*Hz - Ez*Hy
            return (fields.Ey[k, :, :], fields.Hz[k, :, :], fields.Ez[k, :, :], fields.Hy[k, :, :])
        # y: S_y = Ez*Hx - Ex*Hz
        return (fields.Ez[:, k, :], fields.Hx[:, k, :], fields.Ex[:, k, :], fields.Hz[:, k, :])

    def record(self, fields, n: int, t: float, dt: float) -> None:
        if self._mesh is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")
        phase_e = np.exp(1j * self._omega * t) * dt
        phase_h = np.exp(1j * self._omega * (t + 0.5 * dt)) * dt

        h_arrays = (fields.Hx, fields.Hy, fields.Hz)
        # GPU backend: device field arrays refuse implicit mixing with
        # NumPy operands, and fancy-indexing them with host index arrays
        # raises.  Gather on the device with device-resident index
        # arrays and transfer only the per-surface sample vectors — the
        # accumulators stay host-side (surface-scale, tiny per step).
        cp = None
        if type(h_arrays[0]).__module__.partition(".")[0] == "cupy":
            import cupy as cp  # noqa: PLC0415
        if self._gather_idx is None:
            asarray = np.asarray if cp is None else cp.asarray
            self._gather_idx = [
                [
                    (c, surf.comp == c, asarray(surf.flat_idx[surf.comp == c]))
                    for c in range(3)
                    if (surf.comp == c).any()
                ]
                for surf in self._surfaces
            ]
        for surf, bins, gathers in zip(self._surfaces, self._h_bins, self._gather_idx):
            vals = np.empty(len(surf.comp))
            for c, sel, idx in gathers:
                sample = h_arrays[c].reshape(-1)[idx]
                vals[sel] = sample if cp is None else cp.asnumpy(sample)
            bins += phase_h[:, None] * vals[None, :]

        e1, h2, e2, h1 = self._ref_slabs(fields)
        if cp is not None:
            # One plane per component — the same region-sized transfer
            # the field monitors make per recorded step.
            e1, h2, e2, h1 = (cp.asnumpy(a) for a in (e1, h2, e2, h1))
        if not self._ref_bins:
            nf = len(self.freqs)
            for key, arr in (("e1", e1), ("h2", h2), ("e2", e2), ("h1", h1)):
                self._ref_bins[key] = np.zeros((nf, *arr.shape), dtype=complex)
        for key, arr, ph in (
            ("e1", e1, phase_e),
            ("h2", h2, phase_h),
            ("e2", e2, phase_e),
            ("h1", h1, phase_h),
        ):
            self._ref_bins[key] += ph.reshape(-1, 1, 1) * np.asarray(arr)

    def finalize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Result persistence (DD-082 addendum; the DD-070 Freq pattern)
    # ------------------------------------------------------------------

    def result_dump(self) -> dict:
        """The result + the accumulators needed to persist and resume it.

        Like a MonitorFieldFrequency's DFT this is a fixed-size running
        sum, not an append stream — but unlike it, the RESULT is a
        reduction (P_loss/P_flow per tag) rather than the accumulators
        themselves.  So the dump carries both:

        * ``fraction`` — what a reader serves.  Recomputing it from the
          raw bins would need the mesh, the surface enumeration and the
          material resolution, i.e. a second place that produces (and
          could get wrong) the same number; writing what
          :attr:`dissipated_fraction` returns makes reader == monitor
          true by construction.
        * ``h_bins``/``ref_bins`` — the raw accumulators, the resume
          source.

        ``tags`` travels as its own list because tags are heterogeneous
        (material ids are ints, BC walls are face-name strings) and the
        arrays are stored in its order.
        """
        if self._mesh is None:
            raise RuntimeError("monitor not attached; nothing to dump")
        frac = self.dissipated_fraction
        tags = [s.tag for s in self._surfaces]
        return {
            "freqs": np.asarray(self.freqs, dtype=float),
            "tags": tags,
            "fraction": [np.asarray(frac[t], dtype=float) for t in tags],
            "total": np.asarray(frac["total"], dtype=float),
            "h_bins": [np.asarray(b) for b in self._h_bins],
            "ref_bins": {k: np.asarray(v) for k, v in self._ref_bins.items()},
        }

    def load_result_dump(self, dump: dict) -> None:
        """Restore the accumulators from a :meth:`result_dump` (resume).

        The monitor must already be attached (fresh zero accumulators of
        the right shape).  ``_ref_bins`` is normally filled lazily on the
        first ``record``; loading it here pre-populates it, and ``record``
        then adds onto the restored slabs instead of re-zeroing them.
        """
        for bins, saved in zip(self._h_bins, dump["h_bins"]):
            bins[...] = np.asarray(saved)
        self._ref_bins = {k: np.asarray(v).copy() for k, v in dump["ref_bins"].items()}

    # ------------------------------------------------------------------
    # Results (all scale-free ratios)
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        return self.freqs

    @property
    def reference_power(self) -> np.ndarray:
        """P_flow(f) through the reference plane, in (state scale)^2 W.

        FIT identity: P = 1/2 Re( sum e_hat*conj(h_hat) ) over the
        staggered patch pairs — no area weights in the grid-quantity
        basis.  Only meaningful relative to :meth:`raw_power_loss`.
        """
        b = self._ref_bins
        c1, c2 = self._ref_corr
        # Sum the two tangential patch families; e1 pairs with h2 (+),
        # e2 with h1 (−) — the Poynting sign convention of MonitorFluxTime.
        # c1/c2 rescale boundary H states to the physical half-cell patch.
        p = 0.5 * (
            np.sum(b["e1"] * np.conj(b["h2"]) * c1[None, :, :], axis=(1, 2))
            - np.sum(b["e2"] * np.conj(b["h1"]) * c2[None, :, :], axis=(1, 2))
        )
        return p.real

    def raw_power_loss(self) -> dict:
        """Per-tag wall loss in (state scale)^2 W (pairs with
        :attr:`reference_power`).

        Perturbative mode uses the roughness-corrected surface resistance; SIBC
        mode the real part of the operator's own rational fit — the loss
        the solver actually extracted per bin.
        """
        out = {}
        for surf, bins in zip(self._surfaces, self._h_bins):
            if self._resolved is None:
                R_s = self.sibc.fits[surf.tag].impedance(self.freqs).real
            else:
                sig, mur, rough = self._resolved[surf.tag]
                R_s = surface_resistance(self.freqs, sig, mur, rough)
            h_phys2 = (np.abs(bins) * surf.inv_l_dual[None, :]) ** 2
            out[surf.tag] = 0.5 * R_s * np.sum(surf.weight[None, :] * h_phys2, axis=1)
        return out

    @property
    def dissipated_fraction(self) -> dict:
        """Per-tag ``P_loss(f) / P_flow(f)`` (scale-free), plus ``"total"``.

        Full-model semantics on a symmetric run: losses double
        per symmetry plane and so does the reference power for planes
        cutting the reference cross-section — those cancel; a symmetry
        plane parallel to the reference cross-section contributes the
        remaining factor 2.
        """
        # Design: DD-154 (symmetry-plane full-model semantics).
        p_ref = self.reference_power
        s = self._sym_fraction_factor
        out = {tag: s * p / p_ref for tag, p in self.raw_power_loss().items()}
        out["total"] = sum(out.values())
        return out

    def power_loss(self, P_in: float = 1.0) -> dict:
        """Per-tag wall loss [W] for *P_in* Watts through the reference
        plane, plus ``"total"``."""
        return {tag: frac * P_in for tag, frac in self.dissipated_fraction.items()}

    def __repr__(self) -> str:
        return (
            f"MonitorWallLoss(name={self.name!r}, n_freqs={len(self.freqs)}, "
            f"tags={[s.tag for s in self._surfaces]})"
        )
