"""Reproduce the DD-269 staircase-inductance law on one refinement pair.

An oblique lumped path is carried by a staircase of grid edges, and the
staircase links more flux than the chord it stands for.  DD-269 measured
that excess as a local inductance per unit chord length,

    dL' = 58.17 nH/m * x^0.614,   x = staircase/chord - 1,

over ten points and three loop families.  This certificate re-runs the
cheapest of them so the number stays honest as the code moves.

Method (the fixture matters as much as the physics).  A square with side
vector ``(p, q)*delta`` has all four corners on lattice points and area
``(p^2+q^2)*delta^2``, so a Pythagorean triple gives two *congruent,
area-matched* squares at different orientations, both lattice-exact:
``(5m, 0)`` axis-parallel against ``(4m, 3m)`` at 36.87 degrees, whose
staircase is 1.4x its chord.  Nothing has to be forced onto the mesher,
the grid stays uniform, and both orientations get the same grid --
asserted here, because the mesher anchors a source path's endpoints on
grid planes, so a naively rotated loop rotates the grid with it and the
grid's own equivalent radius shifts the reading by the same order as the
effect being measured.

An impressed current (``SourceCurrentPath``) drives each loop and the
induced EMF is read back along the same rasterised chain.  Both runs
share one excitation, so the source spectrum cancels in the ratio and no
DFT convention enters -- which also makes the reading immune to a
truncated time window, as long as both runs use the same pinned step
count.

Acceptance: the excess at 36.87 degrees, 25 cells per side, is
3.89 % +- 0.25 (measured 3.893 %), and the ratio must stay real
(|Im/Re| < 1e-3) and flat across the band -- a growing imaginary part
would mean radiation or boundary asymmetry had entered and the number
must not be believed.

Run from the repository root::

    python validation/oblique_lumped_staircase_certificate.py
"""

import numpy as np

import magnelio as mio
from magnelio import geo, monitors, signals, sources
from magnelio.circuit import rasterize_curve

SIDE_CELLS = 25
DELTA = 200e-6
SIDE = SIDE_CELLS * DELTA  # 5 mm
HALF = 8e-3
MARGIN = 1.5e-3
F_MESH = 20e9
FREQS = np.array([1e9, 2e9, 3e9])
STEPS = 2250

FAMILY = {"axis": (5, 0), "oblique": (4, 3)}
TARGET, TOLERANCE = 3.893, 0.25  # per cent
OPEN = dict.fromkeys(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), "CPML")


def loop_points(kind):
    p, q = (v * SIDE_CELLS // 5 for v in FAMILY[kind])
    corners = [(0, 0), (p, q), (p - q, q + p), (-q, p)]
    # Integer centring: a half-cell shift would put the corners off the
    # lattice, the mesher would anchor planes there, and the two grids
    # would differ -- the confound this fixture exists to remove.
    cx = round(sum(c[0] for c in corners) / 4.0)
    cy = round(sum(c[1] for c in corners) / 4.0)
    pts = [((c[0] - cx) * DELTA, (c[1] - cy) * DELTA, 0.0) for c in corners]
    return [*pts, pts[0]]


def run(kind):
    pts = loop_points(kind)
    model = mio.GeometryModel(boundary_conditions=OPEN)
    model.add(geo.Brick(origin=(-HALF, -HALF, -HALF), size=(2 * HALF,) * 3, material="air"))
    model.add_source(sources.SourceCurrentPath(name="loop", path=pts))
    mesh = mio.Mesh.from_geometry(
        model, mio.MeshControl(max_cell_size=DELTA, min_cell_size=DELTA), f_max=F_MESH
    )

    reach = np.hypot(*FAMILY[kind]) * (SIDE_CELLS // 5) * DELTA / np.sqrt(2.0) + MARGIN
    mon = monitors.MonitorFieldFrequency(
        corners=((-reach, -reach, -MARGIN), (reach, reach, MARGIN)),
        freqs=FREQS,
        fields=["E"],
        name="E_loop",
    )
    mio.AnalysisTD(mesh=mesh, monitors=[mon], verbose=False).run(
        excitations=[
            mio.Excitation("loop", waveform=signals.WaveformGaussian(f_max=F_MESH), amplitude=1.0)
        ],
        total_time_steps=STEPS,
        energy_stop_db=None,
    )
    spec = mon.spectrum_raw
    path = rasterize_curve(geo.Curve.polyline(pts), spec.grid, samples_per_cell=4)
    emf = []
    for f in FREQS:
        state = spec.at_frequency(f)
        comp = {"x": state.Ex, "y": state.Ey, "z": state.Ez}
        # integrate_E cannot take a complex frame (KB-047), so the sum is
        # spelled out here; it is the same signed line integral.
        emf.append(
            sum(
                s * comp[a][i, j, k] * dl
                for a, (i, j, k), s, dl in zip(path.axes, path.ijk, path.signs, path.dls)
            )
        )
    return mesh, path, np.asarray(emf)


def main() -> int:
    grids, emf, lengths = {}, {}, {}
    for kind in FAMILY:
        mesh, path, v = run(kind)
        g = mesh.grid
        grids[kind] = (g.x.copy(), g.y.copy(), g.z.copy())
        emf[kind] = v
        chord = 4.0 * np.hypot(*FAMILY[kind]) * (SIDE_CELLS // 5) * DELTA
        lengths[kind] = path.length / chord
        print(f"{kind:>8}: grid {g.Nx}^3, staircase/chord {lengths[kind]:.4f}")

    for axis, a, b in zip("xyz", grids["axis"], grids["oblique"]):
        if a.shape != b.shape or float(np.abs(a - b).max()) > 1e-15:
            print(f"FAIL: the two orientations do not share the grid ({axis})")
            return 1
    print("both orientations on one identical grid: OK")

    r = emf["oblique"] / emf["axis"]
    excess = 100.0 * (r.real - 1.0)
    im = np.abs(r.imag / r.real)
    print(f"\n{'f [GHz]':>9} {'excess [%]':>12} {'|Im/Re|':>10}")
    for f, e, i in zip(FREQS, excess, im):
        print(f"{f / 1e9:9.1f} {e:12.3f} {i:10.1e}")

    ok = True
    if im.max() > 1e-3:
        print(f"\nFAIL: ratio not real (max |Im/Re| = {im.max():.1e})")
        ok = False
    spread = excess.max() - excess.min()
    if spread > 0.1:
        print(f"FAIL: excess not flat across the band (spread {spread:.3f} points)")
        ok = False
    got = float(excess[1])
    if abs(got - TARGET) > TOLERANCE:
        print(f"FAIL: excess {got:.3f} % is outside {TARGET} +- {TOLERANCE}")
        ok = False
    print(
        f"\nexcess at 2 GHz {got:.3f} % against the DD-269 record {TARGET} % "
        f"(tolerance {TOLERANCE})"
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
