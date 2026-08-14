"""
Profiling: CSG discretization scaling at ~1000 geometric primitives.

Motivation
----------
Real accelerator-component geometries (beam couplers, complex RF windows)
are routinely built from thousands of CSG primitives.  Discretization of
such a model must complete in minutes, not hours.  This script builds a
synthetic but representative stress case entirely from the public API
(``magnelio.geo`` + ``magnelio.mesh``) and profiles it with ``cProfile``
to find out which phase — CSG boolean tree evaluation (OCC) or per-cell
material classification (mesher) — dominates wall-clock time at this scale.

Test geometry
-------------
A biconvex "lens" is built from two overlapping spheres via
:class:`~magnelio.geo.Intersection` (2 primitives).  ``translate`` with
``repeat=500, copy=True`` places 501 lenses along a line (1002 sphere
primitives total).  A single multi-tool :class:`~magnelio.geo.Difference`
carves the whole lens row out of an enclosing dielectric block — this is
the ``Difference(base, *many_tools)`` path added in DD-038, which fuses
all tools into one compound before a single ``BRepAlgoAPI_Cut`` (see
``GeometryModel`` docstring / design-decisions.md DD-038).

Usage
-----
    mamba run --no-capture-output -n mio python benchmarks/profile_csg_scaling.py
"""

from __future__ import annotations

import cProfile
import pathlib
import pstats
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from magnelio.geo import Brick, Difference, GeometryModel, Intersection, Sphere
from magnelio.geo.transforms import translate
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl

N_LENSES = 500  # -> 501 lenses, 1002 sphere primitives
SPHERE_R = 4.0e-3  # m
LENS_OFFSET = 5.0e-3  # center-to-center distance of the two spheres (< 2R -> overlap)
PITCH = 8.0e-3  # spacing between successive lenses along y
F_MAX = 5.0e9  # Hz


def build_geometry() -> GeometryModel:
    dielectric = Material(name="alumina", epsilon=(9.8, 9.8, 9.8))  # RF window material

    sphere_a = Sphere(center=(0.0, 0.0, 0.0), radius=SPHERE_R, material=dielectric)
    sphere_b = Sphere(center=(LENS_OFFSET, 0.0, 0.0), radius=SPHERE_R, material=dielectric)
    lens = Intersection(sphere_a, sphere_b, material=dielectric)

    lenses = translate(lens, (0.0, PITCH, 0.0), repeat=N_LENSES, copy=True)
    print(f"  built {len(lenses)} lens copies ({2 * len(lenses)} sphere primitives)")

    row_length = N_LENSES * PITCH
    block = Brick(
        origin=(-10.0e-3, -5.0e-3, -10.0e-3),
        size=(20.0e-3, row_length + 10.0e-3, 20.0e-3),
        material=dielectric,
    )
    carved = Difference(block, *lenses)

    model = GeometryModel(background=Material.air())
    model.add(carved)
    return model


def main() -> None:
    print(f"CSG scaling profile: {N_LENSES + 1} lenses, ~{2 * (N_LENSES + 1)} sphere primitives")

    t0 = time.perf_counter()
    profiler = cProfile.Profile()

    profiler.enable()
    model = build_geometry()
    t_geom = time.perf_counter()

    control = MeshControl()
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)
    t_mesh = time.perf_counter()
    profiler.disable()

    print(f"\n  geometry construction : {t_geom - t0:8.1f} s")
    print(f"  Mesh.from_geometry     : {t_mesh - t_geom:8.1f} s")
    print(f"  total                  : {t_mesh - t0:8.1f} s")
    print(
        f"  grid                   : {mesh.Nx} x {mesh.Ny} x {mesh.Nz}"
        f" = {mesh.Nx * mesh.Ny * mesh.Nz:,} cells"
    )

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    prof_path = out_dir / "profile_csg_scaling.prof"
    profiler.dump_stats(str(prof_path))

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    print("\n  --- top 25 by cumulative time ---")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
