"""Scale-invariance certificate for the DD-120 geometry scaling.

Maxwell's equations are exactly scale invariant: shrinking every length
by k and raising every frequency by 1/k leaves the S-parameters
unchanged.  This certificate runs the parallel-plate and coax example
fixtures at geometric scales 1x, 1e-3x and 1e-6x (frequencies scaled
inversely, materials non-dispersive) and compares |S| over the common
normalized frequency axis.

Any deviation beyond floating-point noise is a scale artefact of the
geometry/mesh pipeline — the physics core cannot produce one.  The
meshing arithmetic is not bit-equivariant across scales (the user
coordinates themselves round differently at 1e-3x), so the pass bound
is a measured numerical-noise level, not zero.

Run:
    ~/.local/share/mamba/envs/mio/bin/python validation/scale_invariance_certificate.py
"""

import sys

import numpy as np

import magnelio as mio
from magnelio import geo, ports

PASS_BOUND = 1e-6  # measured headroom: deviations land at ~1e-10..1e-8


def run_parallel_plate(k: float):
    a, b, length = 10.0e-3 * k, 5.0e-3 * k, 20.0e-3 * k
    f_max = 40.0e9 / k
    n_modes = 2

    model = mio.GeometryModel(
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    model.add(
        geo.Brick(
            origin=(-a / 2, -b / 2, -length / 2),
            size=(a, b, length),
            material=mio.Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=n_modes))
    model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=n_modes))

    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=15), f_max=f_max)
    analysis = mio.AnalysisScatteringTD(mesh=mesh, f_max=f_max, verbose=False)
    result = analysis.run(excited=[("port1", m) for m in range(n_modes)])

    curves = {}
    for m in range(n_modes):
        curves[f"S11_m{m}"] = result.S("port1", "port1", mode_out=m, mode_in=m)
        curves[f"S21_m{m}"] = result.S("port2", "port1", mode_out=m, mode_in=m)
    return np.asarray(result.f_axis) * k, curves, getattr(mesh, "_geo_scale", None)


def run_coax(k: float):
    r_i, r_a, length = 0.405e-3 * k, 1.475e-3 * k, 8.0e-3 * k
    f_max = 25.0e9 / k

    pec = mio.Material.pec()
    diel = mio.Material.from_isotropic(name="polyethylene", epsilon=2.25)
    outer = geo.Cylinder(origin=(0, 0, 0), radius=r_a, height=length, axis="z", material=diel)
    inner = geo.Cylinder(origin=(0, 0, 0), radius=r_i, height=length, axis="z", material=pec)

    model = mio.GeometryModel(background=pec)
    model.add(geo.Difference(outer, inner))
    model.add(inner)
    model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=1))

    mesh = mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_nodes_per_wavelength=15, max_cell_size=0.12e-3 * k),
        f_max=f_max,
    )
    analysis = mio.AnalysisScatteringTD(mesh=mesh, f_max=f_max, verbose=False)
    result = analysis.run(excited=[("port1", 0)])

    curves = {
        "S11_m0": result.S("port1", "port1", mode_out=0, mode_in=0),
        "S21_m0": result.S("port2", "port1", mode_out=0, mode_in=0),
    }
    return np.asarray(result.f_axis) * k, curves, getattr(mesh, "_geo_scale", None)


def compare(name: str, runner) -> float:
    """Run at all scales, return the worst |S| deviation vs. the 1x run."""
    print(f"\n=== {name} ===")
    f_ref, ref_curves, s_ref = runner(1.0)
    print(f"  k=1e+00: geo scale s = {s_ref}")
    worst = 0.0
    for k in (1e-3, 1e-6):
        f_k, curves, s_k = runner(k)
        print(f"  k={k:.0e}: geo scale s = {s_k}")
        for key, ref in ref_curves.items():
            cur = curves[key]
            # Common normalized frequency axis (interp guards against a
            # step-count difference from the adaptive stop criterion).
            lo = max(f_ref[0], f_k[0])
            hi = min(f_ref[-1], f_k[-1])
            f_common = np.linspace(lo, hi, 401)
            dev = np.abs(
                np.interp(f_common, f_ref, np.abs(ref)) - np.interp(f_common, f_k, np.abs(cur))
            ).max()
            print(f"    {key}: max ||S|(k) - |S|(1)| = {dev:.3e}")
            worst = max(worst, dev)
    return worst


def main() -> None:
    worst_pp = compare("parallel plate (TEM + TE10)", run_parallel_plate)
    worst_cx = compare("coax RG-58 (TEM)", run_coax)
    worst = max(worst_pp, worst_cx)
    print(f"\nworst |S| deviation across scales: {worst:.3e} (bound {PASS_BOUND:.0e})")
    if worst > PASS_BOUND:
        print("FAIL")
        sys.exit(1)
    print("PASS: S-parameters are scale invariant to numerical precision.")


if __name__ == "__main__":
    main()
