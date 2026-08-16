"""
Performance profiling of the FIT-TD solver loop (modal-port API).

Successor to the legacy ``profile_solver.py`` deleted in ``7e22bc4``
during the modal-port rewrite; rebuilt on the public high-level API
(``AnalysisScatteringTD`` + ``PortWaveguide``) per
PERFORMANCE_PROFILING_PLAN.md Workstream 2.

Geometry: PTFE-filled rectangular coaxial line (square outer PEC wall,
square inner PEC conductor), two single-mode waveguide ports on the z
faces.  The homogeneous two-conductor cross-section takes the TEM
Laplace path and passes the uniform-chain certificate, so the ports
terminate with the exact DTBC (DD-054/064) — the per-step port update
profiled here is the production reflection-free path.

Cases (selectable, all share one geometry/grid per preset):

- ``baseline``       non-dispersive PTFE everywhere: plain E/H leapfrog
                     + DTBC port update, no ADE kernels.
- ``dispersive``     the middle half of the line filled with a two-term
                     Debye dielectric whose eps_inf matches PTFE: adds
                     the E-side trapezoidal pole-residue ADE
                     (DD-083/084) and nothing else.
- ``dispersive-mu``  the middle half filled with a mu(omega) Debye
                     material (eps static 2.1): adds the H-side ADE
                     (DD-089) and nothing else.

Each preset runs a fixed ``total_time_steps`` (energy stop disabled),
so per-step costs are directly comparable across the three cases.

Usage
-----
    # Quick smoke test (~40k cells):
    mamba run --no-capture-output -n mio python benchmarks/profile_solver.py small

    # Medium / large:
    mamba run --no-capture-output -n mio python benchmarks/profile_solver.py medium
    mamba run --no-capture-output -n mio python benchmarks/profile_solver.py large

    # One case only, custom step count, cProfile dump:
    mamba run --no-capture-output -n mio python benchmarks/profile_solver.py small \
        --case dispersive --steps 2000 --profile-out results/x.prof

The script prints a per-phase wall-time summary (mesh / run), the
per-step cost, and a cProfile top-30 by cumulative time for each case.
Profiles are saved to ``benchmarks/results/profile_solver_<case>_<preset>.prof``.
"""

import argparse
import cProfile
import io
import pathlib
import pstats
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials import DispersionModel
from magnelio.ports import PortWaveguide

# -- Cross-section (rectangular coax, notebook-08 lineage) -------------------

a = 2e-3  # inner conductor side length [m]
b = 10e-3  # outer conductor side length [m]
eps_r = 2.1  # PTFE

f_max = 10e9
n_freq = 201

PRESETS = {
    #          L [m]      steps   ~cells
    "small": {"L": 50e-3, "steps": 4000},  # ~10k
    "medium": {"L": 200e-3, "steps": 6000},  # ~40k
    "large": {"L": 500e-3, "steps": 8000},  # ~100k
    "xlarge": {"L": 2000e-3, "steps": 8000},  # ~400k
}

CASES = ("baseline", "dispersive", "dispersive-mu")


def _materials(case: str):
    """Return (end_material, middle_material) for the dielectric fill."""
    ptfe = Material.from_isotropic("PTFE", epsilon=eps_r)
    if case == "baseline":
        return ptfe, ptfe
    if case == "dispersive":
        # Two-term Debye, eps_inf matched to PTFE so grid/CFL stay
        # identical to the baseline; relaxations in-band (3.2 / 1.1 GHz).
        model = DispersionModel.debye(
            eps_inf=eps_r,
            delta_eps=[0.3, 0.2],
            tau=[5.0e-11, 1.5e-10],
        )
        return ptfe, Material.dispersive("debye-fill", model)
    if case == "dispersive-mu":
        model = DispersionModel.debye(
            eps_inf=1.0,  # mu_inf for the H-side ADE
            delta_eps=[0.5],
            tau=[5.0e-11],
        )
        return ptfe, Material.dispersive_mu("mu-debye-fill", model, epsilon=eps_r)
    raise ValueError(f"unknown case {case!r}")


def build_geometry(L: float, case: str) -> GeometryModel:
    """Coax line; the middle half carries the case's fill material."""
    end_mat, mid_mat = _materials(case)
    pec = Material.pec()
    # Boundaries are declared on the model, not on the analysis; a closed
    # PEC box is the default but stays explicit here so the profiled
    # configuration is readable from the benchmark alone.
    model = GeometryModel(
        background=pec,
        boundary_conditions={f"{ax}{end}": "PEC" for ax in "xyz" for end in ("min", "max")},
    )

    inner = Brick(origin=(-a / 2, -a / 2, 0.0), size=(a, a, L), material=pec)
    z_cuts = (0.0, L / 4, 3 * L / 4, L)
    fills = (end_mat, mid_mat, end_mat)
    for z0, z1, mat in zip(z_cuts[:-1], z_cuts[1:], fills):
        section = Brick(origin=(-b / 2, -b / 2, z0), size=(b, b, z1 - z0), material=mat)
        model.add(Difference(section, inner, material=mat, name=f"fill-{z0 * 1e3:.0f}mm"))
    model.add(inner)
    return model


def run_case(
    case: str,
    preset_name: str,
    steps: int,
    profile_out: pathlib.Path | None,
    backend: str = "numpy",
) -> dict:
    preset = PRESETS[preset_name]
    L = preset["L"]

    print(
        f"\n=== case={case}  preset={preset_name}  "
        f"L={L * 1e3:.0f} mm  steps={steps}  backend={backend} ==="
    )

    model = build_geometry(L, case)

    t0 = time.perf_counter()
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=20),
        f_max=f_max,
    )
    t_mesh = time.perf_counter() - t0
    n_cells = mesh.Nx * mesh.Ny * mesh.Nz
    print(f"mesh: {mesh.Nx} x {mesh.Ny} x {mesh.Nz} = {n_cells:,} cells ({t_mesh:.2f} s)")

    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin"),
            PortWaveguide(name="port2", plane="zmax"),
        ],
        f_max=f_max,
        n_freq=n_freq,
        verbose=False,
        backend=backend,
    )

    profiler = cProfile.Profile() if profile_out is not None else None
    t1 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    analysis.run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=steps,
    )
    if profiler is not None:
        profiler.disable()
    t_run = time.perf_counter() - t1

    if profiler is not None:
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_out)

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        print(stream.getvalue())

    print(
        f"phases: mesh {t_mesh:.2f} s | run() total {t_run:.2f} s "
        f"| {t_run / steps * 1e3:.3f} ms/step "
        f"| {n_cells * steps / t_run / 1e6:.1f} Mcell-steps/s"
    )
    if profile_out is not None:
        print(f"profile written to {profile_out}")
    return {
        "case": case,
        "preset": preset_name,
        "cells": n_cells,
        "steps": steps,
        "t_mesh": t_mesh,
        "t_run": t_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(PRESETS))
    parser.add_argument("--case", choices=CASES + ("all",), default="all")
    parser.add_argument(
        "--steps", type=int, default=None, help="override the preset's total_time_steps"
    )
    parser.add_argument(
        "--profile-out",
        type=pathlib.Path,
        default=None,
        help="cProfile dump path (single-case runs only)",
    )
    parser.add_argument(
        "--no-profile", action="store_true", help="skip cProfile for unbiased wall-clock timings"
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy", "auto"),
        default="numpy",
        help="array backend; defaults to numpy so the recorded CPU baselines stay comparable",
    )
    args = parser.parse_args()

    cases = CASES if args.case == "all" else (args.case,)
    if args.profile_out is not None and len(cases) != 1:
        parser.error("--profile-out requires a single --case")

    steps = args.steps or PRESETS[args.preset]["steps"]
    results_dir = pathlib.Path(__file__).parent / "results"

    summary = []
    for case in cases:
        out = (
            None
            if args.no_profile
            else args.profile_out or (results_dir / f"profile_solver_{case}_{args.preset}.prof")
        )
        summary.append(run_case(case, args.preset, steps, out, backend=args.backend))

    print(f"\n{'case':<16}{'cells':>10}{'steps':>8}{'mesh [s]':>10}{'run [s]':>10}{'ms/step':>10}")
    for r in summary:
        print(
            f"{r['case']:<16}{r['cells']:>10,}{r['steps']:>8}"
            f"{r['t_mesh']:>10.2f}{r['t_run']:>10.2f}"
            f"{r['t_run'] / r['steps'] * 1e3:>10.3f}"
        )


if __name__ == "__main__":
    main()
