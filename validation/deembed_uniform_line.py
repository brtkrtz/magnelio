"""DD-187 certificate: de-embedding a uniform line to its own floor.

A matched uniform line between two waveguide ports carries nothing but
grid propagation, so ``result.deembed({"p1": L})`` must cancel the
transmission down to the accuracy floor of the run itself — the claim
that the reference-plane shift uses the *exact discrete* chain
dispersion (the ``lambda(z)`` root of DD-054/DD-055), with no
half-cell or half-step reference-plane offset.

Two cases on the same 10×2×20 mm brick, 8 cells/λ at 20 GHz
(λ/8 — deliberately coarse, so the continuum comparison shows the
grid dispersion a textbook ``exp(-jβL)`` would leave behind):

- TEM (PMC side walls): full band.
- TE10 (PEC box, discrete cut-off 14.82 GHz): evaluated above
  1.1 × f_c; below cut-off S values are diagnostic.

Reference measurement (2026-08-24, internal record
``investigations/port-deembedding/MEASUREMENTS.md``):

    TEM : |S21_de - 1| max  -119.7 dB  (run's own S11 floor -123 dB),
          continuum residual 9.7 deg of phase (-15.4 dB).
    TE10: |S21_de - e^{0}| max -67.4 dB (run floor: S11 -76.7 dB),
          continuum residual 3.2 deg (-25.0 dB).
"""

from __future__ import annotations

import numpy as np

import magnelio as mio
from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.ports import PortWaveguide

F_MAX = 20e9
A, B, L = 10e-3, 2e-3, 20e-3
C0 = 299792458.0


def run_case(tem: bool) -> None:
    bc = mio.BoundaryConditions(xmin="PMC", xmax="PMC") if tem else None
    model = GeometryModel(boundary_conditions=bc) if bc else GeometryModel()
    model.add(
        Brick(
            origin=(-A / 2, -B / 2, -L / 2),
            size=(A, B, L),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    model.add_port(PortWaveguide(name="p1", plane="zmin"))
    model.add_port(PortWaveguide(name="p2", plane="zmax"))
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    result = AnalysisScatteringTD(
        mesh=mesh,
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
    ).run(excited=["p1"], energy_stop_db=60.0)

    f = np.asarray(result.f_axis)
    r, q, _z0 = result.port_line_params[("p1", 0)]
    f_c = q / result.dt / (2.0 * np.pi)
    band = (f <= F_MAX) & (f >= 1.1 * f_c)

    de = result.deembed({"p1": L})
    s21_de = de.S("p2", "p1")[band]
    err_disc = np.abs(s21_de - 1.0)
    phi_disc = np.degrees(np.abs(np.angle(s21_de)))

    # Continuum comparison: cancel with exp(-gamma L) instead.
    w = 2.0 * np.pi * f[band]
    kc = 2.0 * np.pi * f_c / C0
    gam = np.sqrt((kc**2 - (w / C0) ** 2).astype(complex))
    gam = np.where(gam.imag < 0, -gam, gam)
    s21_cont = result.S("p2", "p1")[band] * np.exp(gam * L)
    err_cont = np.abs(s21_cont - 1.0)
    phi_cont = np.degrees(np.abs(np.angle(s21_cont)))

    s11 = np.abs(result.S("p1", "p1")[band])
    name = "TEM (PMC walls)" if tem else "TE10 (PEC box)"
    print(f"--- {name}: f_c = {f_c / 1e9:.3f} GHz, {band.sum()} bins ---")
    print(
        f"discrete shift : max|S21_de - 1| = {20 * np.log10(err_disc.max()):7.1f} dB, "
        f"max phase {phi_disc.max():.3e} deg"
    )
    print(
        f"continuum shift: max|S21_de - 1| = {20 * np.log10(err_cont.max()):7.1f} dB, "
        f"max phase {phi_cont.max():.3e} deg"
    )
    print(f"run's own floor: max|S11| = {20 * np.log10(s11.max()):7.1f} dB")


if __name__ == "__main__":
    run_case(tem=True)
    run_case(tem=False)
