"""MonitorFarFieldFrequency — running surface DFT on a closed Huygens box.

The monitor places an axis-aligned box a few cells inside the physical
domain, accumulates the DFT of the tangential E and H on its faces
during the time loop, and hands the frequency-domain surface fields to
the near-to-far-field transform on demand.  Domain faces the box
cannot cross — a PEC/PMC boundary (ground plane) or a declared
symmetry plane — are omitted from the surface and booked as image
planes for the transform.

Sampling: each face lies on a grid-node plane; the tangential fields
are taken from the sanctioned cell-centre interpolation of the two
adjacent cell layers and linearly combined onto the node plane.  That
keeps the surface exactly closed (faces meet at box edges without
gaps or overhangs) and second-order accurate on graded grids.
"""

# Design: DD-173 (far-field monitor and NTFF transform).

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from magnelio.monitors._dft import DFTAccumulator, divide_by_spectrum, source_spectrum
from magnelio.monitors._huygens import (
    _AXES,
    _TANGENTIALS,
    _BoxFace,
    build_faces,
    exclude_pec_patches,
    face_node_indices,
    image_planes_for,
)
from magnelio.monitors.base import _interp_to_cell_centres
from magnelio.post.far_field import (
    FarFieldResult,
    ImagePlane,
    SurfacePatchSet,
    ntff_transform,
    surface_power,
)

# Closure tolerance of the radiated power against the surface power;
# beyond it the box samples the near zone too closely (see result()).
_CLOSURE_TOLERANCE = 0.05


@dataclass
class MonitorFarFieldFrequency:
    """Far-field (antenna pattern) monitor at fixed frequencies on an automatic Huygens box.

    Records the surface DFT of the tangential fields on a closed box
    placed ``margin_cells`` inside the physical domain (the absorber
    layers are excluded automatically).  After a scattering run,
    :meth:`result` returns the far-field pattern at one of the
    requested frequencies.

    Domain faces closed with PEC or PMC — a ground plane — and
    declared symmetry planes are handled by image theory: the box is
    left open there and the mirror images of the recorded surface
    complete it.  For a plain PEC/PMC boundary the pattern is masked
    to the physical half-space; a symmetry plane keeps the full
    sphere.

    A feed guide that crosses the box — a waveguide port in an
    absorbing face — is handled the way a waveguide-fed antenna is
    usually treated: the box face it crosses is sampled at the
    absorber interface, the patches inside the guide (no external
    source) and inside conductors are left out, and the currents on
    the guide's outer wall beyond the box, which the absorber removes,
    are the approximation.  The normalisation is to the incident power
    the run actually launched at each frequency, which for a TE/TM feed
    differs from the excitation waveform by the mode's wave impedance.

    Parameters
    ----------
    freqs : array_like
        Frequencies [Hz] to record.
    name : str
        Monitor name (store key).
    margin_cells : int, default 3
        Clearance in grid cells between the box and the absorber (or
        domain edge).  At least 1, so the two-layer node-plane
        sampling never reads absorber cells.

    Examples
    --------
    >>> from magnelio import monitors
    >>> ff = monitors.MonitorFarFieldFrequency(freqs=[2.45e9], name="pattern")
    """

    freqs: np.ndarray
    name: str = "far_field"
    margin_cells: int = 3

    _grid: object = field(default=None, repr=False, init=False)
    _faces: list = field(default_factory=list, repr=False, init=False)
    _image_planes: list = field(default_factory=list, repr=False, init=False)
    # Footprints of waveguide-port windows on absorbing faces, per face
    # name: ``[{axis: (lo, hi)}]`` inclusive node windows (DD-198).
    # Set by the analysis before attach; the guide interior is no
    # external source and is left out of the Huygens surface.
    _port_footprints: dict = field(default_factory=dict, repr=False, init=False)
    _acc: dict = field(default_factory=dict, repr=False, init=False)
    _source_spectrum: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    # |a(f)| / |W(f)|: the incident power wave the run launched per unit
    # excitation waveform, on the monitor frequencies (1 where the
    # excited channel's impedance is frequency-flat).  Runtime wiring.
    _incident_amplitude: Optional[np.ndarray] = field(default=None, repr=False, init=False)
    _accepted_power: Optional[np.ndarray] = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.freqs = np.atleast_1d(np.asarray(self.freqs, dtype=float))
        if self.freqs.size == 0:
            raise ValueError("freqs must contain at least one frequency")
        if self.margin_cells < 1:
            raise ValueError(
                "margin_cells must be at least 1: the node-plane sampling "
                "reads one cell outside each box face."
            )

    # ------------------------------------------------------------------
    # Monitor protocol
    # ------------------------------------------------------------------

    def attach(self, mesh) -> None:
        """Place the box on *mesh* and allocate fresh DFT accumulators."""
        label = f"far-field monitor {self.name!r}"
        grid = mesh.grid
        self._grid = grid

        lo_n, hi_n, open_faces = face_node_indices(
            mesh,
            margin_cells=self.margin_cells,
            zero_margin_faces=tuple(self._port_footprints),
            label=label,
        )
        self._image_planes = image_planes_for(mesh, open_faces)

        if not open_faces:
            raise ValueError(
                f"{label}: every domain face is a wall; a closed cavity has no radiated field."
            )
        for axis in range(3):
            if hi_n[axis] - lo_n[axis] < 2:
                raise ValueError(
                    f"{label}: after excluding the absorber and "
                    f"{self.margin_cells} margin cell(s), only "
                    f"{hi_n[axis] - lo_n[axis]} cell(s) remain along "
                    f"{_AXES[axis]} — the model needs more physical volume "
                    f"around the radiator."
                )

        self._faces = build_faces(grid, lo_n, hi_n, open_faces)
        self._exclude_metal_and_feeds(mesh, lo_n)

        self._acc = {}
        for bf in self._faces:
            shape = (bf.c1.size, bf.c2.size)
            self._acc[bf.name] = {
                comp: DFTAccumulator(self.freqs, shape) for comp in _TANGENTIALS[bf.axis]
            }

    def _exclude_metal_and_feeds(self, mesh, lo_n) -> None:
        """Zero the patch weights inside conductors and feed guides.

        The exclusions are shared with the other Huygens-box monitors
        (:mod:`magnelio.monitors._huygens`): a patch that is perfect
        conductor on both sides of the face carries no field and no
        source, and a patch inside a waveguide-port window (DD-198)
        samples the feed rather than an external source.  What remains
        is the usual approximation for a waveguide-fed radiator.
        """
        exclude_pec_patches(mesh, self._faces, lo_n, self._port_footprints)

    def record(self, fields, n: int, t: float, dt: float) -> None:
        """Accumulate this step's surface DFT contribution.

        E samples are at time ``t``, H samples at ``t + dt/2`` (the
        leapfrog stagger); each goes into its accumulator with its own
        time stamp.
        """
        del n
        if self._grid is None:
            raise RuntimeError("Monitor not attached. Call attach() first.")
        for bf in self._faces:
            comps = list(_TANGENTIALS[bf.axis])
            ix, iy, iz = bf.slab
            slab = _interp_to_cell_centres(fields, comps, ix, iy, iz, self._grid)
            for comp in comps:
                arr = np.moveaxis(slab[comp], bf.axis, 0)
                vals = (1.0 - bf.weight) * arr[0] + bf.weight * arr[1]
                t_comp = t + 0.5 * dt if comp.startswith("H") else t
                self._acc[bf.name][comp].accumulate(vals, t_comp, dt)

    def finalize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Normalisation (wired by the analysis after each run)
    # ------------------------------------------------------------------

    def renormalize(self, source_signal) -> None:
        """Normalize the surface DFT to 1 W incident CW power.

        Called for you at the end of a scattering run; call directly
        only for hand-driven solver runs.  Stores the excitation
        spectrum as the divisor — the accumulated bins stay untouched,
        so repeating the call just replaces the reference.
        """
        self._source_spectrum = source_spectrum(
            source_signal.values,
            source_signal.dt,
            self.freqs,
        )

    @property
    def is_renormalized(self) -> bool:
        """Whether the 1 W renormalization has been applied."""
        return self._source_spectrum is not None

    def _set_incident_amplitude(self, f_axis, ratio) -> None:
        # Runtime wiring by the analysis: |a(f)| / |W(f)| of the excited
        # channel, so the per-1-W normalisation refers to the incident
        # power actually launched at each frequency (TE/TM ports).
        self._incident_amplitude = np.interp(
            self.freqs, np.asarray(f_axis, dtype=float), np.asarray(ratio, dtype=float)
        )

    def _set_accepted_power(self, f_axis, accepted) -> None:
        # Runtime wiring by the analysis (not part of the recipe): the
        # run's accepted-power curve, interpolated onto the monitor
        # frequencies, feeds FarFieldResult.gain.
        self._accepted_power = np.interp(
            self.freqs, np.asarray(f_axis, dtype=float), np.asarray(accepted, dtype=float)
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        return self.freqs

    def _freq_index(self, f: Optional[float], f_index: Optional[int]) -> int:
        if f_index is not None:
            return int(f_index)
        if f is None:
            if self.freqs.size == 1:
                return 0
            raise ValueError(
                f"this monitor recorded {self.freqs.size} frequencies; "
                f"pass f= or f_index= to pick one."
            )
        idx = int(np.argmin(np.abs(self.freqs - f)))
        if abs(self.freqs[idx] - f) > 1e-6 * max(abs(f), 1.0):
            raise ValueError(
                f"frequency {f:.6g} Hz was not recorded; available: "
                f"{np.array2string(self.freqs, precision=6)}"
            )
        return idx

    def _patch_sets(self, idx: int) -> list[SurfacePatchSet]:
        if self._source_spectrum is None:
            raise ValueError(
                "far-field data is only meaningful per watt of incident "
                "power; run the monitor through a scattering analysis, or "
                "call renormalize(reference_signal) for a hand-driven run."
            )
        spectrum = self._source_spectrum[idx : idx + 1]
        incident = 1.0 if self._incident_amplitude is None else float(self._incident_amplitude[idx])
        sets = []
        for bf in self._faces:
            comps = _TANGENTIALS[bf.axis]
            vals = {
                c: divide_by_spectrum(self._acc[bf.name][c].result[idx : idx + 1], spectrum)[0]
                / incident
                for c in comps
            }
            g1, g2 = np.meshgrid(bf.c1, bf.c2, indexing="ij")
            n_p = g1.size
            centers = np.zeros((n_p, 3))
            centers[:, bf.axis] = bf.plane
            centers[:, bf.tangent_axes[0]] = g1.ravel()
            centers[:, bf.tangent_axes[1]] = g2.ravel()
            normals = np.zeros((n_p, 3))
            normals[:, bf.axis] = bf.sign
            areas = np.outer(bf.w1, bf.w2)
            if bf.keep is not None:
                areas = areas * bf.keep
            areas = areas.ravel()
            E = np.zeros((n_p, 3), dtype=complex)
            H = np.zeros((n_p, 3), dtype=complex)
            E[:, bf.tangent_axes[0]] = vals[comps[0]].ravel()
            E[:, bf.tangent_axes[1]] = vals[comps[1]].ravel()
            H[:, bf.tangent_axes[0]] = vals[comps[2]].ravel()
            H[:, bf.tangent_axes[1]] = vals[comps[3]].ravel()
            sets.append(SurfacePatchSet(centers=centers, normals=normals, areas=areas, E=E, H=H))
        return sets

    def result(
        self,
        f: Optional[float] = None,
        *,
        f_index: Optional[int] = None,
        theta: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
    ) -> FarFieldResult:
        """The far-field pattern at one recorded frequency.

        Parameters
        ----------
        f : float, optional
            Frequency [Hz]; must be one of :attr:`freqs` (omit for a
            single-frequency monitor).
        f_index : int, optional
            Index into :attr:`freqs`, alternative to *f*.
        theta, phi : array_like, optional
            Spherical evaluation grids [rad]; defaults to 2° over the
            full sphere.

        Returns
        -------
        FarFieldResult
        """
        idx = self._freq_index(f, f_index)
        accepted = None
        if self._accepted_power is not None:
            accepted = float(self._accepted_power[idx])
        sets = self._patch_sets(idx)
        # The flux through the recorded box; a symmetry plane's mirror
        # half exists physically and radiates the same again, so the
        # surface power is booked in full-model watts like P_rad.  A
        # real ground plane doubles nothing: P_rad is the half-space
        # power and so is the flux through the open box.
        n_sym = sum(1 for p in self._image_planes if not p.physical_halfspace)
        p_surface = surface_power(sets) * 2.0**n_sym
        result = ntff_transform(
            sets,
            self._image_planes,
            float(self.freqs[idx]),
            theta=theta,
            phi=phi,
            accepted_power=accepted,
            surface_power=p_surface,
        )
        # Closure check: the surface fields carry a definite real power
        # out of the box, and the pattern must radiate the same power
        # for a lossless exterior.  A shortfall means the box samples
        # the radiator's near zone too closely — every face of the box
        # sits at the absorbing boundary, so the cure is more clearance
        # (measured on a microstrip patch: 0.93 with the domain top
        # 0.3 λ above the copper, 1.00 at 0.7 λ).  The pattern amplitude
        # is then low by that factor; directivity is self-normalised.
        scale = max(abs(result.P_rad), abs(p_surface))
        if scale > 0.0 and abs(result.P_rad - p_surface) > _CLOSURE_TOLERANCE * scale:
            balance = result.P_rad / p_surface if p_surface > 0.0 else float("inf")
            f_ghz = float(self.freqs[idx]) / 1e9
            warnings.warn(
                f"far-field monitor {self.name!r} at {f_ghz:.4g} GHz: the "
                f"pattern radiates {balance:.3f} of the power leaving the "
                f"recording box (surface_power {p_surface:.4g} W, P_rad "
                f"{result.P_rad:.4g} W).  The box sits at the absorbing "
                f"boundary and samples the radiator's near zone too "
                f"closely; realized gain and gain are off by that factor "
                f"(directivity is not).  Give the model more clearance to "
                f"the absorbing faces — half a wavelength or more between "
                f"the radiator and the boundary restores the balance.",
                stacklevel=2,
            )
        return result

    def plot_cut(self, f: Optional[float] = None, *, f_index: Optional[int] = None, **kwargs):
        """Polar cut of the pattern at one recorded frequency.

        Keyword arguments beyond *f*/*f_index* go to
        :meth:`FarFieldResult.plot_cut` (``plane=``, ``angle=``,
        ``quantity=`` and the drawing options).
        """
        return self.result(f, f_index=f_index).plot_cut(**kwargs)

    def plot_3d(self, f: Optional[float] = None, *, f_index: Optional[int] = None, **kwargs):
        """3D radiation surface at one recorded frequency.

        Keyword arguments beyond *f*/*f_index* go to
        :meth:`FarFieldResult.plot_3d`.
        """
        return self.result(f, f_index=f_index).plot_3d(**kwargs)

    # ------------------------------------------------------------------
    # Persistence (the MonitorWallLoss result_dump pattern)
    # ------------------------------------------------------------------

    def result_dump(self) -> dict:
        """The accumulators plus the box geometry, for store and resume.

        The face geometry and image planes travel with the bins so a
        reader can rebuild the transform inputs without the mesh —
        reader == monitor by construction.
        """
        if self._grid is None:
            raise RuntimeError("monitor not attached; nothing to dump")
        faces = []
        for bf in self._faces:
            faces.append(
                {
                    "name": bf.name,
                    "axis": int(bf.axis),
                    "sign": float(bf.sign),
                    "plane": float(bf.plane),
                    "c1": np.asarray(bf.c1),
                    "c2": np.asarray(bf.c2),
                    "w1": np.asarray(bf.w1),
                    "w2": np.asarray(bf.w2),
                    "keep": (np.ones((bf.c1.size, bf.c2.size)) if bf.keep is None else bf.keep),
                    "bins": {
                        comp: np.asarray(self._acc[bf.name][comp].result)
                        for comp in _TANGENTIALS[bf.axis]
                    },
                }
            )
        planes = [
            {
                "axis": int(p.axis),
                "position": float(p.position),
                "kind": p.kind,
                "at_low": bool(p.at_low),
                "physical_halfspace": bool(p.physical_halfspace),
            }
            for p in self._image_planes
        ]
        dump = {
            "name": self.name,
            "freqs": np.asarray(self.freqs, dtype=float),
            "margin_cells": int(self.margin_cells),
            "faces": faces,
            "image_planes": planes,
        }
        if self._source_spectrum is not None:
            dump["source_spectrum"] = np.asarray(self._source_spectrum)
        if self._accepted_power is not None:
            dump["accepted_power"] = np.asarray(self._accepted_power)
        if self._incident_amplitude is not None:
            dump["incident_amplitude"] = np.asarray(self._incident_amplitude)
        return dump

    @classmethod
    def from_result_dump(cls, dump: dict) -> "MonitorFarFieldFrequency":
        """Rebuild a result-serving monitor from a :meth:`result_dump`.

        The store reader's path: the dump carries the box geometry and
        image planes, so no mesh is needed — the rebuilt monitor
        answers :meth:`result` but cannot :meth:`record` (it is not
        attached to a grid).
        """
        mon = cls(
            freqs=np.asarray(dump["freqs"], dtype=float),
            name=str(dump.get("name", "far_field")),
            margin_cells=int(dump.get("margin_cells", 3)),
        )
        mon._faces = []
        mon._acc = {}
        for saved in dump["faces"]:
            axis = int(saved["axis"])
            name = str(saved["name"])
            c1 = np.asarray(saved["c1"], dtype=float)
            c2 = np.asarray(saved["c2"], dtype=float)
            bf = _BoxFace(
                name=name,
                axis=axis,
                sign=float(saved["sign"]),
                plane=float(saved["plane"]),
                slab=(None, None, None),  # reader: result-only, no record
                weight=0.0,
                tangent_axes=tuple(a for a in range(3) if a != axis),
                c1=c1,
                c2=c2,
                w1=np.asarray(saved["w1"], dtype=float),
                w2=np.asarray(saved["w2"], dtype=float),
                keep=(None if "keep" not in saved else np.asarray(saved["keep"], dtype=float)),
            )
            mon._faces.append(bf)
            mon._acc[name] = {}
            for comp, bins in saved["bins"].items():
                acc = DFTAccumulator(mon.freqs, (c1.size, c2.size))
                acc.result[...] = np.asarray(bins)
                mon._acc[name][comp] = acc
        mon._image_planes = [
            ImagePlane(
                axis=int(p["axis"]),
                position=float(p["position"]),
                kind=str(p["kind"]),
                at_low=bool(p["at_low"]),
                physical_halfspace=bool(p["physical_halfspace"]),
            )
            for p in dump["image_planes"]
        ]
        if "source_spectrum" in dump:
            mon._source_spectrum = np.asarray(dump["source_spectrum"])
        if "accepted_power" in dump:
            mon._accepted_power = np.asarray(dump["accepted_power"])
        if "incident_amplitude" in dump:
            mon._incident_amplitude = np.asarray(dump["incident_amplitude"])
        return mon

    def load_result_dump(self, dump: dict) -> None:
        """Restore accumulators written by :meth:`result_dump` (resume).

        The monitor must already be attached to the same mesh, so the
        fresh accumulators have the dumped shapes.
        """
        for saved in dump["faces"]:
            acc = self._acc[str(saved["name"])]
            for comp, bins in saved["bins"].items():
                acc[comp].result[...] = np.asarray(bins)
        if "source_spectrum" in dump:
            self._source_spectrum = np.asarray(dump["source_spectrum"])
        if "accepted_power" in dump:
            self._accepted_power = np.asarray(dump["accepted_power"])
        if "incident_amplitude" in dump:
            self._incident_amplitude = np.asarray(dump["incident_amplitude"])

    def __repr__(self) -> str:
        n_freqs = self.freqs.size
        return (
            f"MonitorFarFieldFrequency(name={self.name!r}, n_freqs={n_freqs}, "
            f"margin_cells={self.margin_cells})"
        )
