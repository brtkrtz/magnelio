"""Perturbative wall-loss (power-loss) postprocessing.

Evaluates conductor losses on PEC-classified walls from the tangential
magnetic field of a lossless solve:

    P_loss = 1/2 * R_s(omega) * integral |H_tan|^2 dA,
    R_s    = sqrt(omega * mu0 * mu_r / (2 * sigma))

For eigenmodes this yields the wall-loss quality factor
Q = omega * W / P_loss — the standard perturbation result, accurate for
good conductors (skin depth much smaller than wall thickness and
curvature radius).

Surface resistances come from the materials themselves for lossy-metal
solids (``Material.lossy_metal``); plain-PEC solids and PEC
boundary-condition walls have no conductivity of their own and take the
caller-supplied ``sigma`` / ``mu``.
"""

# Design: DD-082 (perturbative wall-loss chain), DD-081 (lossy-metal
# materials).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.constants import MU0  # noqa: E402
from magnelio.mesh._surfaces import enumerate_pec_surfaces, resolve_wall_conductors


def surface_resistance(
    f,
    sigma: float,
    mu: float = 1.0,
    roughness=None,
) -> np.ndarray:
    """Skin-effect surface resistance R_s(f) [Ohm] for a good conductor.

    With a ``roughness`` model the smooth value is multiplied by
    its frequency-dependent factor K(f) >= 1.
    """
    # Roughness models: DD-088.
    r_s = np.sqrt(2.0 * np.pi * np.asarray(f) * MU0 * mu / (2.0 * sigma))
    if roughness is not None:
        r_s = r_s * roughness.factor(f, sigma, mu)
    return r_s


def _resolve_surface_materials(mesh, surfaces, sigma, mu, roughness=None, overrides=None):
    """Per-tag (sigma, mu, roughness) — the shared DD-082 tag rule
    (single source: ``mesh.surfaces.resolve_wall_conductors``, which
    the SIBC setup uses too).  ``overrides`` carries the DD-099
    per-face wall materials of ``PECBoundary`` declarations."""
    return resolve_wall_conductors(
        mesh,
        surfaces,
        sigma=sigma,
        mu=mu,
        roughness=roughness,
        overrides=overrides,
    )


@dataclass
class WallLossQ:
    """Wall-loss Q of one eigenmode.

    Attributes
    ----------
    frequency : float
        Mode frequency [Hz].
    Q : float
        Total wall-loss quality factor.
    P_loss : float
        Total wall loss [W] at the mode's stored amplitude.
    W : float
        Time-averaged stored energy [J] at the mode's stored amplitude.
    per_tag : dict
        ``tag -> P_loss`` breakdown [W]; tags are material ids (solids)
        and face names (PEC boundary walls).
    """

    frequency: float
    Q: float
    P_loss: float
    W: float
    per_tag: dict

    def Q_of(self, tag) -> float:
        """Partial Q of one wall tag (losses of the others switched off)."""
        return 2.0 * np.pi * self.frequency * self.W / self.per_tag[tag]

    def __repr__(self) -> str:
        return (
            f"WallLossQ(f={self.frequency / 1e9:.4f} GHz, Q={self.Q:.1f}, "
            f"tags={list(self.per_tag)})"
        )


def wall_loss_Q(
    result,
    mode: int = 0,
    *,
    sigma: float | None = None,
    mu: float = 1.0,
    roughness=None,
) -> WallLossQ:
    """Wall-loss Q factor of an eigenmode (perturbative power-loss).

    Parameters
    ----------
    result : EigenmodeResult
        A lossless eigenmode solution (fields + mesh).  PEC domain
        walls are read from ``solver_info['boundary_conditions']``
        (omitted faces default to PEC, the solver convention).
    mode : int
        Mode index into ``result.modes``.
    sigma : float, optional
        Conductivity [S/m] for walls that are not lossy metals
        (plain-PEC solids and PEC boundary walls).  Lossy-metal solids
        always use their own material values.
    mu : float, optional
        Relative permeability accompanying ``sigma`` (default 1).
    roughness : SurfaceRoughness, optional
        Surface-roughness model for the same walls ``sigma``
        applies to; lossy-metal solids always use their own.  ``None``
        (default) is a perfectly smooth conductor.

    Returns
    -------
    WallLossQ
    """
    mesh = result.mesh
    f0 = float(result.frequencies[mode])
    omega = 2.0 * np.pi * f0
    fs = result.modes[mode]

    from magnelio.boundaries.pec import PECBoundary  # noqa: PLC0415

    bcs = result.solver_info.get("boundary_conditions", {})
    all_faces = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")

    def _is_pec(value) -> bool:
        return isinstance(value, PECBoundary) or (isinstance(value, str) and value.upper() == "PEC")

    bc_pec = tuple(face for face in all_faces if _is_pec(bcs.get(face, "PEC")))
    # DD-099: PECBoundary declarations may carry their own wall
    # material (the boundary condition carries the wall model).
    overrides = {
        face: (v.wall_sigma, v.wall_mu, v.wall_roughness)
        for face in bc_pec
        if isinstance(v := bcs.get(face), PECBoundary) and v.wall_sigma is not None
    }
    # DD-099: declared non-PEC faces must not book registered boundary
    # walls (an eigen run has no ports to exclude) — they get the
    # continuation semantics of `_masked_face_pec_views`.
    surfaces = enumerate_pec_surfaces(
        mesh,
        bc_pec_faces=bc_pec,
        masked_boundary_faces=tuple(f for f in all_faces if f not in bc_pec),
    )
    if not surfaces:
        raise ValueError("no PEC wall surfaces found on this mesh")
    resolved = _resolve_surface_materials(
        mesh,
        surfaces,
        sigma,
        mu,
        roughness,
        overrides=overrides or None,
    )

    # Stored energy: time-averaged total of the peak-amplitude mode,
    # W = 1/4 (e^T M_eps e + h^T M_mu h) — equal halves at resonance.
    e = np.concatenate([fs.Ex.ravel(), fs.Ey.ravel(), fs.Ez.ravel()])
    h = np.concatenate([fs.Hx.ravel(), fs.Hy.ravel(), fs.Hz.ravel()])
    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    W = 0.25 * (float((M_eps * e) @ e) + float((M_mu * h) @ h))

    per_tag = {}
    for surf in surfaces:
        sig, mur, rough = resolved[surf.tag]
        R_s = float(surface_resistance(f0, sig, mur, rough))
        per_tag[surf.tag] = 0.5 * R_s * surf.h_tan_sq_sum(fs.Hx, fs.Hy, fs.Hz)

    P_total = float(sum(per_tag.values()))
    return WallLossQ(
        frequency=f0,
        Q=omega * W / P_total,
        P_loss=P_total,
        W=W,
        per_tag=per_tag,
    )
