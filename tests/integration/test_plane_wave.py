"""
Integration test: PlaneWaveSource TF/SF injection smoke test.

Runs a small FIT-TD simulation with a +z-propagating, x-polarised Gaussian
pulse injected via the TF/SF formulation.  After ~25 steps the wave front
should have entered the TF box and Ex field values inside should be non-zero.
Outside (SF region) fields should remain near zero until the wave reaches the
domain boundary.
"""

import numpy as np
import pytest


def _build_mesh(Nx=8, Ny=8, Nz=16):
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh

    L_xy = 8e-3
    L_z = 16e-3
    grid = GridLines(
        x=np.linspace(0, L_xy, Nx + 1),
        y=np.linspace(0, L_xy, Ny + 1),
        z=np.linspace(0, L_z, Nz + 1),
    )
    return Mesh.from_grid(grid), grid


def test_plane_wave_smoke():
    """TF/SF: Ex is non-zero inside TF box after sufficient steps."""
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt
    from magnelio.sources.plane_wave import PlaneWaveSource

    mesh, grid = _build_mesh(Nx=8, Ny=8, Nz=16)
    dt = courant_dt(grid, accuracy="normal")

    # TF/SF box: inner region of the domain (2 cells inset each side)
    x, y, z = grid.x, grid.y, grid.z
    tf_box = ((x[2], y[2], z[2]), (x[6], y[6], z[14]))

    f_max = 20e9
    src = PlaneWaveSource(
        direction=(0.0, 0.0, 1.0),
        polarization=(1.0, 0.0, 0.0),
        corners=tf_box,
        f_max=f_max,
        waveform="gaussian",
    )

    bcs = {face: PECBoundary(face) for face in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    # Run enough steps for the wave peak to enter the TF box.
    # Wave peak at t0=4/f_max travels at c0; TF box z_min ≈ z[2].
    # Steps needed: (t0 + z[2]/c0) / dt  (plus some margin)
    import math

    c0 = 299_792_458.0
    t_arrive = 4.0 / f_max + z[2] / c0
    n_steps = int(math.ceil(t_arrive / dt)) + 10

    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        sources=[src],
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
    )
    fields = solver.run()

    # Inside the TF box Ex must be non-zero
    Ex_inside = fields.Ex[2:6, 2:7, 3:14]
    assert np.max(np.abs(Ex_inside)) > 1e-4, (
        f"Ex inside TF box is too small: max={np.max(np.abs(Ex_inside)):.3e}"
    )


def _tfsf_amplitude_and_leakage(direction, polarization, n=24, inset=6):
    """Run a plane wave in empty space; return (peak TF amplitude, peak SF leakage).

    The TF/SF split is only correct if the incident field reaches full
    amplitude inside the box and cancels outside it.  Both halves are
    sensitive to a sign or an index slipping on any one of the six faces,
    which a "something arrived" assertion is not.
    """
    import math

    from magnelio.boundaries.pec import PECBoundary
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt
    from magnelio.sources.plane_wave import PlaneWaveSource

    L = 24e-3
    # The incident field is retarded from the origin, so the domain is laid
    # out on the far side of it: k.r >= 0 everywhere, and the pulse enters
    # the grid from rest instead of switching on part-way up its flank.
    axes = [
        np.linspace(0.0, L, n + 1) if direction[a] >= 0 else np.linspace(-L, 0.0, n + 1)
        for a in range(3)
    ]
    grid = GridLines(x=axes[0], y=axes[1], z=axes[2])
    mesh = Mesh.from_grid(grid)
    dt = courant_dt(grid, accuracy="normal")

    f_max = 6e9  # ~30 cells per wavelength: dispersion well below the leakage floor
    src = PlaneWaveSource(
        direction=direction,
        polarization=polarization,
        corners=(
            tuple(ax[inset] for ax in axes),
            tuple(ax[n - inset] for ax in axes),
        ),
        f_max=f_max,
        waveform="gaussian",
    )
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    # Stop with the pulse peak at the centre of the box, which it reaches at
    # t0 + k.r_centre / c0.
    c0 = 299_792_458.0
    centre = np.array([0.5 * (ax[0] + ax[-1]) for ax in axes])
    n_steps = int(math.ceil((4.0 / f_max + float(np.dot(direction, centre)) / c0) / dt))
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        sources=[src],
        total_time_steps=n_steps,
        dt=dt,
        verbose=False,
        precision="double",
    )
    fields = solver.run()

    # Sample the polarisation component; it carries the full amplitude.
    comp = ("Ex", "Ey", "Ez")[int(np.argmax(np.abs(polarization)))]
    arr = getattr(fields, comp)
    e = np.abs(arr.get() if hasattr(arr, "get") else np.asarray(arr))  # device or host
    # Grid voltages, not field samples: divide by the edge length (uniform here).
    e = e / (L / n)

    core = slice(inset + 1, n - inset)
    inside = float(e[core, core, core].max())
    shell = e.copy()
    shell[inset:-inset, inset:-inset, inset:-inset] = 0.0
    # The outermost layer sits on the PEC wall, where the reflected field
    # is not part of the SF/TF statement.
    outside = float(shell[1:-1, 1:-1, 1:-1].max())
    return inside, outside


@pytest.mark.parametrize(
    "direction,polarization",
    [
        ((0, 0, 1), (1, 0, 0)),
        ((0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((-1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0)),
    ],
)
def test_tfsf_full_amplitude_inside_and_quiet_outside(direction, polarization):
    """Every propagation axis reaches unit amplitude in TF and stays quiet in SF."""
    inside, outside = _tfsf_amplitude_and_leakage(direction, polarization)
    # Measured 0.9999 and 1.5e-5 on every axis; the thresholds leave a factor
    # of ten so a genuine sign or index slip cannot hide in the margin.
    assert inside == pytest.approx(1.0, abs=1e-3), f"TF amplitude {inside:.4f}"
    assert outside < 1e-4, f"SF leakage {outside:.3e}"
