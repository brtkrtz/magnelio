"""
Integration test: PlaneWaveSource TF/SF injection smoke test.

Runs a small FIT-TD simulation with a +z-propagating, x-polarised Gaussian
pulse injected via the TF/SF formulation.  After ~25 steps the wave front
should have entered the TF box and Ex field values inside should be non-zero.
Outside (SF region) fields should remain near zero until the wave reaches the
domain boundary.
"""

import numpy as np


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
