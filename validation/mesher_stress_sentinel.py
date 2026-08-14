"""Mesher stress sentinel — permanent gate for every mesher change (WP-M5).

Randomised and pathological geometries through ``Mesh.from_geometry``,
each checked against machine-checkable mesh invariants:

I1  monotone axes, strictly positive cell sizes;
I2  no cell below the hard ``min_cell_size`` floor (when set);
I3  no cell below ``min_feature_gap`` (the DD-058 sliver class; the
    mesher itself raises on violations since WP-M4 — the sentinel
    treats any such RuntimeError as a FAIL, not a crash);
I4  per-axis growth ratio <= 1.5 * growth_factor (the pinned WP-M4
    generator contract) — checked only when feature refinement is
    active (``min_cells_per_feature > 0``): with it, every interval
    starts at the shared per-axis h_fine, so the bound holds across
    interval boundaries too; with ``min_cells_per_feature = 0`` the
    two-scale design never promised smoothing between adjacent
    critical-plane intervals (measured here: 3.2-7.6x jumps on the
    tangent-cylinder family — user-accepted coarse meshing, covered
    by the quality warning);
I5  M_eps / M_mu finite (no NaN/inf — the DD-058 spectrum-poisoning
    class), M_mu > 0 where not frozen;
I6  production ``courant_dt`` within budget of the floor-implied
    bound (guards against effective-eps/mu collapse);
I7  thin-sheet cases: the sheet plane is a grid node and no node lies
    inside the metal layer.

Case families (the session-88/91/92 lessons as permanent regressions):

* tangent cylinders on AND off forced-grid multiples (the WP-M1
  sliver factory);
* CSG-wiggled brick faces vs forced nodes (1e-16 … 1e-9 offsets);
* thin PEC layers at 0.1x … 2x the floor (WP-M2 detection threshold
  sweep — below the floor -> ONE plane, above -> resolved);
* near-coincident dielectric faces (noise scale to 1.5x floor —
  WP-M3 floor merge + longitudinal harmonic eps);
* rotated bricks (tilted faces contribute only bbox planes);
* seeded random brick/cylinder soups with random floors.

Run with: ``mamba run --no-capture-output -n mio python
validation/mesher_stress_sentinel.py [--fast]``.
Exit code 0 = all invariants hold.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import (
    Brick,
    Cylinder,
    Difference,
    GeometryModel,
)
from magnelio.geo.transforms import rotate
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.stability import courant_dt

C0 = 299_792_458.0


# ----------------------------------------------------------------------
# Invariant checks
# ----------------------------------------------------------------------


def check_invariants(
    mesh: Mesh,
    control: MeshControl,
    label: str,
    thin_sheet_planes: dict[str, list[float]] | None = None,
    metal_intervals: dict[str, list[tuple[float, float]]] | None = None,
) -> list[str]:
    """Return a list of invariant violations (empty = clean)."""
    bad: list[str] = []
    grid = mesh.grid
    floor = control.min_cell_size

    for name, nodes, d in (
        ("x", grid.x, grid.dx),
        ("y", grid.y, grid.dy),
        ("z", grid.z, grid.dz),
    ):
        # I1 — monotone, positive.
        if not np.all(np.diff(nodes) > 0):
            bad.append(f"I1 {name}: non-monotone axis")
        # I2 — hard floor.
        if floor is not None and d.min() < floor * (1 - 1e-9):
            bad.append(f"I2 {name}: cell {d.min():.3e} m below floor {floor:.3e}")
        # I3 — sliver scale (resolved DD-120 value: the control may
        # carry min_feature_gap=None for the bbox-relative default).
        feature_gap = getattr(mesh, "_resolved_feature_gap", None)
        if feature_gap is None:
            feature_gap = control.min_feature_gap or 0.0
        if d.min() < feature_gap * (1 - 1e-9):
            bad.append(
                f"I3 {name}: sliver cell {d.min():.3e} m below min_feature_gap {feature_gap:.3e}"
            )
        # I4 — growth-ratio contract (per axis, pinned WP-M4 bound).
        # Only when feature refinement is active: every interval then
        # starts at the shared per-axis h_fine, so the within-interval
        # bound extends across interval boundaries.
        if control.min_cells_per_feature > 0 and len(d) > 1:
            r = np.maximum(d[1:] / d[:-1], d[:-1] / d[1:]).max()
            if r > 1.5 * control.growth_factor * (1 + 1e-9):
                bad.append(
                    f"I4 {name}: growth ratio {r:.3f} exceeds "
                    f"1.5*g = {1.5 * control.growth_factor:.3f}"
                )

    # I5 — finite material matrices (the DD-058 poisoning class).
    m_eps = np.asarray(build_M_eps(mesh))
    m_mu = np.asarray(build_M_mu(mesh))
    if not np.all(np.isfinite(m_eps)):
        bad.append(f"I5: non-finite entries in M_eps ({np.count_nonzero(~np.isfinite(m_eps))})")
    if not np.all(np.isfinite(m_mu)):
        bad.append(f"I5: non-finite entries in M_mu ({np.count_nonzero(~np.isfinite(m_mu))})")
    if np.any(m_mu < 0):
        bad.append(f"I5: negative M_mu entries ({np.count_nonzero(m_mu < 0)})")

    # I6 — production dt within budget of the floor-implied bound.
    if floor is not None:
        dt = (
            courant_dt(grid, accuracy="normal", mesh=mesh)
            if _courant_takes_mesh()
            else courant_dt(grid, "normal")
        )
        n_max = max(
            math.sqrt(max(mat.epsilon) * max(mat.mu))
            for mat in mesh.material_library.values()
            if not mat.is_pec
        )
        dt_bound = floor / (C0 * math.sqrt(3.0))  # vacuum floor bound
        # Budget: the material-loaded dt may be n_max slower plus the
        # safety factor courant_dt applies; 0.5 covers both.
        if dt < 0.5 * dt_bound / n_max:
            bad.append(
                f"I6: courant_dt {dt:.3e} below budget {0.5 * dt_bound / n_max:.3e} (floor-implied)"
            )

    # I7 — thin-sheet plane placement.
    if thin_sheet_planes:
        for axis, positions in thin_sheet_planes.items():
            nodes = np.asarray(getattr(grid, axis))
            for p in positions:
                if p not in nodes:
                    bad.append(f"I7 {axis}: sheet plane {p!r} missing")
    if metal_intervals:
        for axis, spans in metal_intervals.items():
            nodes = np.asarray(getattr(grid, axis))
            for lo, hi in spans:
                inside = nodes[(nodes > lo + 1e-12) & (nodes < hi - 1e-12)]
                if inside.size:
                    bad.append(
                        f"I7 {axis}: node(s) inside metal layer ({lo!r}, {hi!r}): {inside[:3]}"
                    )

    return bad


def _courant_takes_mesh() -> bool:
    import inspect

    return "mesh" in inspect.signature(courant_dt).parameters


# ----------------------------------------------------------------------
# Case families
# ----------------------------------------------------------------------


def cases_tangent_cylinders(rng, fast):
    """Cylinder tangent planes vs forced grid nodes (WP-M1 class)."""
    s_bbox = 24e-3
    radii = [10e-3, 9.7e-3]  # on- and off-multiple of the 1 mm grid
    if not fast:
        radii += [float(r) for r in rng.uniform(6e-3, 11.5e-3, size=3)]
    for R in radii:
        pec, vac = Material.pec(), Material.air()
        bbox = Brick(
            origin=(0.0, -s_bbox / 2, -s_bbox / 2), size=(4e-3, s_bbox, s_bbox), material=pec
        )
        cav = Cylinder(origin=(0, 0, 0), radius=R, height=4e-3, axis="x", material=vac)
        m = GeometryModel()
        m.add(Difference(bbox, cav))
        m.add(cav)
        t_nodes = np.linspace(-s_bbox / 2, s_bbox / 2, 25).tolist()
        ctrl = MeshControl(
            min_nodes_per_wavelength=8,
            min_cells_per_feature=0,
            growth_factor=1.5,
            max_cell_size=4e-3,
            forced_planes={"y": t_nodes, "z": t_nodes},
        )
        yield f"tangent-cyl R={R * 1e3:.3f}mm", m, ctrl, 14e9, None, None


def cases_csg_wiggle(rng, fast):
    """Brick faces float-wiggled against forced nodes."""
    offsets = [1e-16, 1e-12, 1e-9]
    if fast:
        offsets = [1e-12]
    for off in offsets:
        fr4 = Material(name="FR4", epsilon=(4.4,) * 3)
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 5e-3, 2e-3 + off), material=fr4))
        m.add(Brick(origin=(0, 0, 2e-3 + off), size=(10e-3, 5e-3, 4e-3), material=Material.air()))
        ctrl = MeshControl(
            min_nodes_per_wavelength=10,
            min_cells_per_feature=2,
            forced_planes={"z": [2e-3]},
        )
        yield f"csg-wiggle {off:.0e}", m, ctrl, 10e9, None, None


def cases_thin_layers(rng, fast):
    """PEC layers at 0.1x … 2x the floor on a substrate."""
    floor = 100e-6
    factors = [0.1, 0.35, 0.9, 1.1, 2.0]
    if fast:
        factors = [0.35, 1.1]
    for f in factors:
        t = f * floor
        sub = Material(name="sub", epsilon=(4.3,) * 3)
        pec, air = Material.pec(), Material.air()
        h_sub, w_dom, w_strip = 0.635e-3, 6e-3, 1.8e-3
        strip = Brick(origin=(-w_strip / 2, 0, h_sub), size=(w_strip, 4e-3, t), material=pec)
        air_b = Brick(origin=(-w_dom / 2, 0, h_sub), size=(w_dom, 4e-3, 3e-3), material=air)
        m = GeometryModel()
        m.add(Brick(origin=(-w_dom / 2, 0, 0), size=(w_dom, 4e-3, h_sub), material=sub))
        m.add(Difference(air_b, strip))
        m.add(strip)
        ctrl = MeshControl(min_cells_per_feature=2, min_cell_size=floor)
        thin = {"z": [h_sub]} if t < floor else None
        metal = {"z": [(h_sub, h_sub + t)]} if t < floor else None
        yield (f"thin-layer t={f:.2f}x floor", m, ctrl, 10e9, thin, metal)


def cases_near_coincident_dielectric(rng, fast):
    """Dielectric face pairs from noise scale to 1.5x the floor.

    The smallest gap sits above the OCC kernel precision (1e-7 m) —
    solids at or below that scale cannot be built at all and are
    rejected with a clear error by the geometry layer (found by this
    sentinel; guard in ``occ_backend._check_dimensions``).
    """
    floor = 100e-6
    gaps = [0.5e-6, 2e-6, 0.5 * floor, 0.9 * floor, 1.5 * floor]
    if fast:
        gaps = [2e-6, 0.9 * floor]
    for gap in gaps:
        d1 = Material(name="d1", epsilon=(4.3,) * 3)
        d2 = Material(name="d2", epsilon=(8.0,) * 3)
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 1e-3), material=d1))
        m.add(Brick(origin=(0, 0, 1e-3), size=(4e-3, 4e-3, gap), material=d2))
        m.add(Brick(origin=(0, 0, 1e-3 + gap), size=(4e-3, 4e-3, 2e-3), material=Material.air()))
        ctrl = MeshControl(min_cells_per_feature=2, min_cell_size=floor)
        yield f"near-diel gap={gap * 1e6:.1f}um", m, ctrl, 10e9, None, None


def cases_rotated_bricks(rng, fast):
    """Tilted faces: only bbox silhouette planes, no tangent artefacts."""
    angles = [7.0, 33.0] if not fast else [33.0]
    for ang in angles:
        fr4 = Material(name="FR4", epsilon=(4.4,) * 3)
        inner = Brick(origin=(2e-3, 2e-3, 2e-3), size=(4e-3, 3e-3, 2e-3), material=fr4)
        inner = rotate(inner, (0, 0, 1), ang, origin=(4e-3, 3.5e-3, 3e-3))
        m = GeometryModel(allow_overlaps=True)
        m.add(Brick(origin=(0, 0, 0), size=(10e-3, 8e-3, 6e-3), material=Material.air()))
        m.add(inner)
        ctrl = MeshControl(min_nodes_per_wavelength=10, min_cells_per_feature=2)
        yield f"rotated-brick {ang:.0f}deg", m, ctrl, 10e9, None, None


def cases_random_soup(rng, fast):
    """Seeded random brick/cylinder soups with random floors."""
    n_cases = 3 if fast else 10
    for i in range(n_cases):
        floor = float(rng.uniform(50e-6, 500e-6))
        m = GeometryModel(allow_overlaps=True)
        n_shapes = int(rng.integers(2, 6))
        for s in range(n_shapes):
            eps = float(rng.uniform(1.0, 12.0))
            mat = Material(name=f"m{s}", epsilon=(eps,) * 3)
            if rng.random() < 0.3:
                m.add(
                    Cylinder(
                        origin=tuple(rng.uniform(1e-3, 6e-3, size=3)),
                        radius=float(rng.uniform(0.3e-3, 2e-3)),
                        height=float(rng.uniform(0.5e-3, 4e-3)),
                        axis=str(rng.choice(["x", "y", "z"])),
                        material=mat,
                    )
                )
            else:
                m.add(
                    Brick(
                        origin=tuple(rng.uniform(0.0, 5e-3, size=3)),
                        size=tuple(rng.uniform(0.3e-3, 5e-3, size=3)),
                        material=mat,
                    )
                )
        ctrl = MeshControl(
            min_cells_per_feature=int(rng.integers(0, 5)),
            growth_factor=float(rng.uniform(1.15, 2.0)),
            min_cell_size=floor,
        )
        yield f"random-soup #{i} floor={floor * 1e6:.0f}um", m, ctrl, 10e9, None, None


# ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reduced case counts (CI smoke)")
    args = ap.parse_args()

    rng = np.random.default_rng(20260710)
    families = (
        cases_tangent_cylinders,
        cases_csg_wiggle,
        cases_thin_layers,
        cases_near_coincident_dielectric,
        cases_rotated_bricks,
        cases_random_soup,
    )

    n_pass = 0
    failures: list[tuple[str, list[str]]] = []
    t0 = time.time()
    for family in families:
        for label, model, ctrl, f_max, thin, metal in family(rng, args.fast):
            t_case = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mesh = Mesh.from_geometry(model, ctrl, f_max=f_max)
            except Exception as exc:  # noqa: BLE001 — a crash IS a finding
                failures.append((label, [f"mesher raised: {exc!r}"]))
                print(f"  FAIL  {label}: raised {type(exc).__name__}")
                continue
            bad = check_invariants(mesh, ctrl, label, thin_sheet_planes=thin, metal_intervals=metal)
            n_cells = mesh.grid.n_cells
            dt_s = time.time() - t_case
            if bad:
                failures.append((label, bad))
                print(f"  FAIL  {label} ({n_cells} cells, {dt_s:.1f}s)")
                for b in bad:
                    print(f"        {b}")
            else:
                n_pass += 1
                print(f"  pass  {label} ({n_cells} cells, {dt_s:.1f}s)")

    total = n_pass + len(failures)
    print(f"\n{n_pass}/{total} cases clean in {time.time() - t0:.1f}s")
    if failures:
        raise SystemExit(f"FAIL: {len(failures)} case(s) violated mesh invariants.")
    print("PASS: all mesh invariants hold.")


if __name__ == "__main__":
    main()
