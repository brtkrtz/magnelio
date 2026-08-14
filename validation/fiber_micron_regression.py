"""Micron-scale fiber geometry regression (DD-120).

A step-index fiber cross-section — core (r = 4.5 um), cladding
(r = 62.5 um) and polymer coating (r = 125 um) as concentric solids
with coordinates in the 1e-6..1e-4 m range — exercised through the
full meshing pipeline.

Before DD-120 this geometry died twice over: ``min_feature_gap = 1e-6``
clustered *every* transverse critical plane of the micron structure
into a single position (silent geometry annihilation), and the OCC
kernel operated at a tolerance of ~1e-7 m — a tenth of the core
radius.  With the automatic power-of-two model scaling plus the
bbox-relative clustering tolerance the fiber must now mesh with all
transverse feature planes intact.

Checks (DD-062 invariant style):

  R1  the mesher chooses a scale factor > 1 (power of two)
  R2  every material-boundary radius appears as grid planes on BOTH
      transverse axes (+/- r within the resolved feature gap)
  R3  grid axes are strictly monotone, no sliver cell below the
      resolved feature gap
  R4  all four materials (background + 3 dielectrics) survive into
      material_id

Run:
    ~/.local/share/mamba/envs/mio/bin/python validation/fiber_micron_regression.py
"""

import math
import sys

import numpy as np

import magnelio as mio
from magnelio import geo

R_CORE = 4.5e-6
R_CLAD = 62.5e-6
R_COAT = 125e-6
LENGTH = 20e-6
F_MAX = 5e12  # 5 THz keeps the mesh small; the *geometry* is the test


def build_model() -> mio.GeometryModel:
    core_mat = mio.Material.from_isotropic(name="core", epsilon=2.11)
    clad_mat = mio.Material.from_isotropic(name="cladding", epsilon=2.085)
    coat_mat = mio.Material.from_isotropic(name="coating", epsilon=2.40)

    core = geo.Cylinder(origin=(0, 0, 0), radius=R_CORE, height=LENGTH, axis="z", material=core_mat)
    clad = geo.Cylinder(origin=(0, 0, 0), radius=R_CLAD, height=LENGTH, axis="z", material=clad_mat)
    coat = geo.Cylinder(origin=(0, 0, 0), radius=R_COAT, height=LENGTH, axis="z", material=coat_mat)

    model = mio.GeometryModel()
    model.add(geo.Difference(coat, clad))
    model.add(geo.Difference(clad, core))
    model.add(core)
    return model


def main() -> None:
    model = build_model()
    mesh = mio.Mesh.from_geometry(
        model,
        mio.MeshControl(min_nodes_per_wavelength=10),
        f_max=F_MAX,
    )
    print(f"mesh: ({mesh.Nx}, {mesh.Ny}, {mesh.Nz})")

    failures: list[str] = []

    # R1 — auto scale chosen, power of two.
    s = getattr(mesh, "_geo_scale", None)
    gap = getattr(mesh, "_resolved_feature_gap", None)
    print(f"geo scale s = {s}, resolved min_feature_gap = {gap:.3e} m")
    if not (s and s > 1.0 and math.log2(s) == round(math.log2(s))):
        failures.append(f"R1: expected power-of-two scale > 1, got {s!r}")

    # R2 — feature planes survive on both transverse axes.
    for axis_name, nodes in (("x", np.asarray(mesh.grid.x)), ("y", np.asarray(mesh.grid.y))):
        for r in (R_CORE, R_CLAD, R_COAT):
            for pos in (-r, r):
                if np.abs(nodes - pos).min() > max(gap, 1e-9):
                    failures.append(
                        f"R2: plane {pos:+.3e} missing on axis {axis_name} "
                        f"(nearest {nodes[np.abs(nodes - pos).argmin()]:+.3e})"
                    )
        n_distinct = len(np.unique(nodes))
        if n_distinct < 3:
            failures.append(f"R2: axis {axis_name} collapsed to {n_distinct} planes")

    # R3 — monotone axes, no sliver below the resolved gap.
    for axis_name, d in (("x", mesh.grid.dx), ("y", mesh.grid.dy), ("z", mesh.grid.dz)):
        d = np.asarray(d)
        if not np.all(d > 0):
            failures.append(f"R3: axis {axis_name} non-monotone")
        if d.min() < gap * (1 - 1e-9):
            failures.append(f"R3: sliver cell {d.min():.3e} m below gap {gap:.3e} on {axis_name}")

    # R4 — all materials present.
    present = {int(v) for v in np.unique(mesh.material_id)}
    names = sorted(mesh.material_library[m].name for m in present)
    print(f"materials in mesh: {names}")
    for want in ("core", "cladding", "coating"):
        if want not in names:
            failures.append(f"R4: material {want!r} missing from material_id")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nPASS: micron fiber meshes with all feature planes intact.")


if __name__ == "__main__":
    main()
