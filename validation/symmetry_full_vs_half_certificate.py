"""Full- vs. half-model certificate for symmetry planes (DD-154).

A shielded microstrip line with a dielectric block over the trace
(non-trivial S11/S21), mirror-symmetric in x.  The FULL geometry is
built twice and run twice:

* ``full`` — no symmetry declared; the whole cross-section is meshed.
* ``half`` — ``{"xmin": "SymmetryPMC"}``; the mesher clips the domain
  to x >= 0 and the mirror half is never meshed.

Certificate quantities (full vs. half):

* ``z_line`` — the published full-model port impedance (stage D
  restores the half-window factor 2);
* ``S11``/``S21`` over the centre band (S-parameters need no
  correction on a symmetric port pair — the full-model wave scale
  cancels in b/a);
* the peak Poynting flux through a cross-section between port and
  block — full-model watts on both runs (DD-155): the half-model
  excitation injects ×1/√2 (half the full-model power into the meshed
  half) and the monitor books ×2 per cutting plane.  Historical note:
  under the pre-DD-155 half-window-normalised excitation the raw
  half-model flux already matched the full model, and this
  certificate measured a monitor-side ×2 as a factor-2 error;
* the peak of the incident power wave ``a(t)`` on the excited port —
  the recorder restores ×√2 per cutting plane, so both runs report
  the same full-model √W amplitude;
* a 1-W-renormalised field probe near the strip: the injection scale
  puts the half-model fields at full-model level, and
  ``renormalize(result.reference_signal)`` divides by the *unscaled*
  full-model waveform — under the old semantics this probe read √2
  high on the half model.

The two meshes are NOT identical on x >= 0 (the full model grades
through x = 0, the half model ends there), so the certificate bounds
are discretisation-level, not machine precision.

Run from ``magnelio/``:

    CUPY_ACCELERATORS= python validation/symmetry_full_vs_half_certificate.py
"""

from __future__ import annotations

import numpy as np

import magnelio as mio
from magnelio import geo, ports
from magnelio.monitors import MonitorFieldFrequency, MonitorFluxTime

F_MAX = 12e9
H_SUB = 0.8e-3
W_STRIP = 1.2e-3
T_STRIP = 0.2e-3
W_BOX = 8.0e-3
H_BOX = 5.0e-3
L = 16.0e-3
EPS_R = 4.3

# Certificate bounds (discretisation-level: the two grids differ on
# the shared half-space).
TOL_Z_REL = 0.02
TOL_S_ABS = 0.05
TOL_FLUX_REL = 0.05
TOL_A_REL = 0.01
TOL_FIELD_REL = 0.05
F_PROBE = 6e9


def run_case(symmetric: bool) -> dict:
    pec = mio.Material.pec()
    air = mio.Material.air()
    fr4 = mio.Material.from_isotropic(name="FR4", epsilon=EPS_R)
    blocker = mio.Material.from_isotropic(name="blocker", epsilon=10.0)

    bc = {"xmin": "SymmetryPMC"} if symmetric else None
    model = mio.GeometryModel(boundary_conditions=bc, allow_overlaps=True)
    model.add(geo.Brick(origin=(-W_BOX / 2, 0.0, 0.0), size=(W_BOX, H_SUB, L), material=fr4))
    air_cap = geo.Brick(
        origin=(-W_BOX / 2, H_SUB, 0.0),
        size=(W_BOX, H_BOX - H_SUB, L),
        material=air,
    )
    strip = geo.Brick(
        origin=(-W_STRIP / 2, H_SUB, 0.0),
        size=(W_STRIP, T_STRIP, L),
        material=pec,
    )
    model.add(geo.Difference(air_cap, strip))
    model.add(strip)
    # Reflecting block over the trace, mirror-symmetric in x
    # (last-wins overlap semantics carve it out of the air cap).
    model.add(
        geo.Brick(
            origin=(-1.5e-3, H_SUB + T_STRIP, L / 2 - 2e-3),
            size=(3e-3, 1.5e-3, 4e-3),
            material=blocker,
        )
    )
    model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))

    mesh = mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_nodes_per_wavelength=20),
        f_max=F_MAX,
    )
    flux = MonitorFluxTime(plane=("z", L / 4), name="flux_feed")
    # 1-W field probe beside the strip edge, inside the meshed half of
    # both runs (the point snaps to slightly different cells on the two
    # grids — a discretisation-level comparison).
    probe_at = (1.5e-3, H_SUB + 0.6e-3, L / 4)
    probe = MonitorFieldFrequency(
        corners=(probe_at, probe_at),
        freqs=[F_PROBE],
        fields=["E"],
        name="e_probe",
    )
    analysis = mio.AnalysisScatteringTD(
        mesh=mesh,
        f_max=F_MAX,
        monitors=(flux, probe),
        verbose=False,
    )
    report = analysis.solve_ports()["port1"]
    result = analysis.run(excited=["port1"])
    probe.renormalize(result.reference_signal)
    e_mag = float(
        np.sqrt(sum(np.abs(probe.component(c)[0]) ** 2 for c in ("Ex", "Ey", "Ez"))),
    )
    f = result.f_axis
    return {
        "n_cells": mesh.Nx * mesh.Ny * mesh.Nz,
        "z_line": float(report.z_line_num),
        "f": np.asarray(f),
        "S11": np.asarray(result.S("port1", "port1")),
        "S21": np.asarray(result.S("port2", "port1")),
        "p_peak": float(np.max(np.abs(flux.power))),
        "a_peak": float(np.max(np.abs(result.a("port1").values))),
        "e_probe": e_mag,
    }


def main() -> None:
    full = run_case(symmetric=False)
    half = run_case(symmetric=True)

    # Compare on the centre band (the band edges carry the excitation
    # roll-off).
    n = len(full["f"])
    sel_full = slice(n // 4, 3 * n // 4)
    f_band = full["f"][sel_full]
    s11_half = np.interp(f_band, half["f"], np.abs(half["S11"]))
    s21_half = np.interp(f_band, half["f"], np.abs(half["S21"]))
    d_s11 = float(np.max(np.abs(np.abs(full["S11"][sel_full]) - s11_half)))
    d_s21 = float(np.max(np.abs(np.abs(full["S21"][sel_full]) - s21_half)))
    d_z = abs(half["z_line"] - full["z_line"]) / full["z_line"]
    d_p = abs(half["p_peak"] - full["p_peak"]) / full["p_peak"]
    d_a = abs(half["a_peak"] - full["a_peak"]) / full["a_peak"]
    d_e = abs(half["e_probe"] - full["e_probe"]) / full["e_probe"]

    print(f"cells         full {full['n_cells']:8d}   half {half['n_cells']:8d}")
    print(f"z_line [ohm]  full {full['z_line']:8.3f}   half {half['z_line']:8.3f}   d {d_z:.2e}")
    print(f"flux peak [W] full {full['p_peak']:.6e}   half {half['p_peak']:.6e}   d {d_p:.2e}")
    print(f"a peak [sqW]  full {full['a_peak']:.6e}   half {half['a_peak']:.6e}   d {d_a:.2e}")
    print(f"|E| 1W probe  full {full['e_probe']:.6e}   half {half['e_probe']:.6e}   d {d_e:.2e}")
    print(f"max |d|S11||  {d_s11:.3e}   (band {f_band[0] / 1e9:.1f}-{f_band[-1] / 1e9:.1f} GHz)")
    print(f"max |d|S21||  {d_s21:.3e}")

    assert d_z < TOL_Z_REL, f"z_line deviates {d_z:.3e} (tol {TOL_Z_REL})"
    assert d_s11 < TOL_S_ABS, f"S11 deviates {d_s11:.3e} (tol {TOL_S_ABS})"
    assert d_s21 < TOL_S_ABS, f"S21 deviates {d_s21:.3e} (tol {TOL_S_ABS})"
    assert d_p < TOL_FLUX_REL, f"flux peak deviates {d_p:.3e} (tol {TOL_FLUX_REL})"
    assert d_a < TOL_A_REL, f"a peak deviates {d_a:.3e} (tol {TOL_A_REL})"
    assert d_e < TOL_FIELD_REL, f"1W field probe deviates {d_e:.3e} (tol {TOL_FIELD_REL})"
    print("CERTIFICATE PASSED")


if __name__ == "__main__":
    main()
