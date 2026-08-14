"""Certificate: passive LumpedElement vs the closed shunt-impedance form (DD-123).

A parallel-plate TEM line (PMC side walls, exact multi-conductor DTBC
ports, |S11| floors < -120 dB) carries a passive lumped element as a
mid-line shunt across the plate gap.  For a shunt impedance Z on a line
of impedance Z0 the exact S-parameters at the element plane are

    S11 = -Z0 / (Z0 + 2 Z),      S21 = 2 Z / (Z0 + 2 Z).

Three gates:

1. **Quasi-static anchor** — a pure 100 ohm SeriesRLC shunt matches
   |S11|/|S21| of the closed form to < 1e-3 at beta*d ~ 0.05 (element
   path electrically short).
2. **Electrical-length envelope** — the deviation grows ~ (beta*d)^2
   as the element path becomes electrically long (measured: halving
   the path shrinks the 4 GHz deviation by ~4.7x).  This is the real
   physics of a distributed current filament, not a discretisation
   error (grid refinement does not change it); the practical rule is
   the same as for a physical SMD: keep the element path short
   against the wavelength.
3. **Resonance wiring** — a series-LC-R shunt produces its |S21|
   minimum at f_res = 1/(2 pi sqrt(LC)), validating the trapezoidal
   L/C companion state inside the 3D time loop (not just the pure-R
   path).

Run from the repository root:

    mamba run --no-capture-output -n mio python \
        validation/lumped_element_shunt_certificate.py
"""

import numpy as np

from magnelio import circuit
from magnelio.analysis.scattering_td import AnalysisScatteringTD
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortSpecMultiConductor

WIDTH, LENGTH = 16e-3, 60e-3
BC = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}


def run_shunt(gap: float, element, f_axis: np.ndarray):
    """S11/S21 columns of a mid-line gap shunt on the TEM plate."""
    grid = GridLines(
        x=np.linspace(-WIDTH / 2, WIDTH / 2, 9),
        y=np.linspace(-gap / 2, gap / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 121),
    )
    shunt = circuit.LumpedElement(
        name="shunt",
        start=(0.0, -gap / 2, 0.0),
        end=(0.0, gap / 2, 0.0),
        element=element,
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=dict(BC)).with_elements([shunt])
    ana = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="m2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=6e9,
        verbose=False,
        backend="numpy",
    )
    z0 = ana.solve_ports()["m1"].z_line_num
    res = ana.run(
        f_axis=f_axis,
        excited=["m1"],
        total_time_steps=40000,
        energy_stop_db=None,
    )
    return z0, res.S("m1", "m1"), res.S("m2", "m1")


def main() -> None:
    ok = True

    # Gate 1: quasi-static anchor (beta*d = 0.05 at 0.5 GHz, gap 5 mm).
    r_sh = 100.0
    f_axis = np.array([0.5e9, 2e9, 4e9])
    z0, s11, s21 = run_shunt(5e-3, circuit.SeriesRLC(R=r_sh), f_axis)
    s11_exact = abs(-z0 / (z0 + 2 * r_sh))
    s21_exact = 2 * r_sh / (z0 + 2 * r_sh)
    d11 = abs(abs(s11[0]) - s11_exact)
    d21 = abs(abs(s21[0]) - s21_exact)
    print(f"gate 1  z0 = {z0:.3f} ohm, R = {r_sh:.0f} ohm shunt @ 0.5 GHz")
    print(f"        |S11| {abs(s11[0]):.5f} vs {s11_exact:.5f}  (d = {d11:.1e})")
    print(f"        |S21| {abs(s21[0]):.5f} vs {s21_exact:.5f}  (d = {d21:.1e})")
    gate1 = d11 < 1e-3 and d21 < 1e-3
    ok &= gate1
    print(f"        -> {'PASSED' if gate1 else 'FAILED'} (tolerance 1e-3)")

    # Gate 2: (beta*d)^2 envelope — halve the element path, deviation
    # at 4 GHz must shrink by ~4x (accept 3x..7x).
    dev_full = abs(s21[2]) - s21_exact
    z0h, _, s21h = run_shunt(2.5e-3, circuit.SeriesRLC(R=r_sh), f_axis)
    dev_half = abs(s21h[2]) - 2 * r_sh / (z0h + 2 * r_sh)
    ratio = dev_full / dev_half
    print(f"gate 2  dS21(4 GHz): path 5 mm {dev_full:+.4f}, path 2.5 mm {dev_half:+.4f}")
    print(f"        ratio = {ratio:.2f} (quadratic expectation ~4)")
    gate2 = 3.0 < ratio < 7.0
    ok &= gate2
    print(f"        -> {'PASSED' if gate2 else 'FAILED'}")

    # Gate 3: series-RLC resonance — |S21| minimum at f_res.  Short
    # path (2.5 mm) and a large element L so the path's own series
    # inductance (~1 nH here, the gate-2 physics again) stays a small
    # perturbation on f_res.
    f_res = 3.0e9
    l_res = 20e-9
    c_res = 1.0 / ((2 * np.pi * f_res) ** 2 * l_res)
    f_fine = np.linspace(1e9, 5e9, 81)
    _, _, s21_rlc = run_shunt(
        2.5e-3,
        circuit.SeriesRLC(R=10.0, L=l_res, C=c_res),
        f_fine,
    )
    f_min = f_fine[int(np.argmin(np.abs(s21_rlc)))]
    print(f"gate 3  series RLC (10 ohm, {l_res * 1e9:.0f} nH, {c_res * 1e12:.2f} pF)")
    print(f"        |S21| minimum at {f_min / 1e9:.3f} GHz (target {f_res / 1e9:.3f} GHz)")
    gate3 = abs(f_min - f_res) / f_res < 0.05
    ok &= gate3
    print(f"        -> {'PASSED' if gate3 else 'FAILED'} (tolerance 5 %)")

    print()
    print("CERTIFICATE PASSED" if ok else "CERTIFICATE FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
