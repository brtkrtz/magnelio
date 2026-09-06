"""Energy and flux from a recording, against the march itself (DD-260).

A whole-domain recording of a ring-down reproduces the solver's energy
trace; a recording's flux through a plane is the flux monitor's record;
a frequency monitor's spectrum states the power a matched line carries
per watt incident; and all of it survives the project store.
"""

from __future__ import annotations

import numpy as np

import magnelio as mio
from magnelio import monitors, sources
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.stability import spectral_dt

A, B, D = 22.86e-3, 10.16e-3, 30e-3  # WR-90 cross-section, TE101 near 8.2 GHz
F_MAX = 12e9


def _mesh() -> Mesh:
    grid = GridLines(x=np.linspace(0, A, 13), y=np.linspace(0, B, 7), z=np.linspace(0, D, 17))
    return Mesh.from_grid(grid)


def _ringdown(mesh, *, t_end, project=None):
    eig = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()
    mode = sources.SourceFieldInitial(name="mode0", field=eig.field(0))
    dt = spectral_dt(mesh, "normal")
    every_step = np.arange(0.0, t_end, dt)
    volume = monitors.MonitorFieldTime(
        name="volume", corners=((0, 0, 0), (A, B, D)), fields=["E", "H"], times=every_step
    )
    z0 = mesh.grid.z[8]
    # The flux pairing at node k takes the magnetic samples of cell k
    # (DD-085), so the one-cell layer that can state it is the cell
    # above the plane: declared at that cell's centre.
    zc = 0.5 * (mesh.grid.z[8] + mesh.grid.z[9])
    layer = monitors.MonitorFieldTime(
        name="layer", corners=((0, 0, zc), (A, B, zc)), fields=["E", "H"], times=every_step
    )
    flux = monitors.MonitorFluxTime(normal="z", position=z0, name="flux")
    result = mio.AnalysisTD(
        mesh=mesh.with_sources([mode]),
        monitors=[volume, layer, flux],
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        project=project,
    ).run(excitations=["mode0"], t_end=t_end, energy_stop_db=None)
    return result, z0


def _check_against_the_march(result, z0):
    rec = result.monitors["volume"].recording
    # A project-backed run hands back the project: its trace is a call.
    trace = result.energy_trace() if callable(result.energy_trace) else result.energy_trace
    dt = rec.dt
    # The energy sample of step n is formed from the frame taken after
    # step n: e = E^{n+1} at (n+1)·dt, h_prev/h the H half-steps around it.
    steps = [int(n) for n in trace["step"] if (n + 1) * dt <= rec.times[-1] + 0.5 * dt]
    assert len(steps) >= 3, "the run recorded too few energy samples to compare"
    idx = np.array([rec.index_of((n + 1) * dt) for n in steps])
    np.testing.assert_allclose(rec.times[idx], (np.array(steps) + 1) * dt, rtol=0, atol=1e-3 * dt)
    energies = rec.energy()
    expected = np.array([trace["energy"][list(trace["step"]).index(n)] for n in steps])
    np.testing.assert_allclose(energies[idx], expected, rtol=1e-9)

    # The recorded flux is the flux monitor's record, frame by frame,
    # from the whole-domain recording and from a one-cell layer alike.
    power = np.asarray(result.monitors["flux"].power)
    n = min(power.size, rec.n_frames)
    np.testing.assert_array_equal(rec.flux("z", z0)[:n], power[:n])
    layer = result.monitors["layer"].recording
    assert layer.grid.Nz == 1
    np.testing.assert_array_equal(layer.flux("z", z0)[:n], power[:n])
    return rec


def test_recording_reproduces_the_energy_trace_and_the_flux_monitor():
    mesh = _mesh()
    result, z0 = _ringdown(mesh, t_end=2e-9)
    rec = _check_against_the_march(result, z0)
    assert rec._ops is not None and rec._ops.ends == (("wall", "wall"),) * 3


def test_the_store_carries_the_operators(tmp_path):
    mesh = _mesh()
    result, z0 = _ringdown(mesh, t_end=1.5e-9, project=tmp_path / "ring")
    _check_against_the_march(result, z0)
    project = mio.open_project(tmp_path / "ring")
    (run,) = project.runs.values()
    rec = run.monitors["volume"].recording
    assert rec._ops is not None
    np.testing.assert_array_equal(rec.energy(), result.monitors["volume"].recording.energy())


def test_spectrum_flux_is_the_transmitted_power():
    """On a matched TEM line the time-averaged flux per 1 W incident is |S21|²."""
    from magnelio import AnalysisScatteringTD
    from magnelio.mesh import BoxFace
    from magnelio.ports import PortSpecMultiConductor

    gap, width, length = 5e-3, 16e-3, 60e-3
    grid = GridLines(
        x=np.linspace(-width / 2, width / 2, 9),
        y=np.linspace(-gap / 2, gap / 2, 6),
        z=np.linspace(-length / 2, length / 2, 121),
    )
    bc = {"xmin": "PMC", "xmax": "PMC", "ymin": "PEC", "ymax": "PEC", "zmin": "PEC", "zmax": "PEC"}
    freqs = np.array([1e9, 2e9])
    plane = monitors.MonitorFieldFrequency(
        corners=((None, None, 0.0), (None, None, 0.0)), freqs=freqs, fields=["E", "H"], name="plane"
    )
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid).with_boundary_conditions(bc),
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="m2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=6e9,
        monitors=(plane,),
        verbose=False,
    )
    res = ana.run(f_axis=freqs, excited=["m1"], total_time_steps=20000, energy_stop_db=None)
    spectrum = plane.spectrum
    power = spectrum.flux("z", 0.0)
    s21 = np.abs(res.S("m2", "m1", f_axis=freqs)) ** 2
    assert np.all(power > 0.0), f"power flows towards m2; got {power}"
    np.testing.assert_allclose(power, s21, rtol=5e-3)
    assert np.all(spectrum.energy() > 0.0)


def _te201(x, y, z):
    """TE201 of the box: E_y ∝ sin(2πx/A) sin(πz/D), zero on the plane x = A/2."""
    e_y = np.sin(2 * np.pi * np.clip(x, 0.0, A) / A) * np.sin(np.pi * np.clip(z, 0.0, D) / D)
    return np.zeros_like(e_y), e_y, np.zeros_like(e_y)


def test_symmetry_half_books_the_full_model():
    """The half box behind an electric symmetry plane reports the whole box's joules.

    Same spacing on both grids, the same analytic field sampled on
    both, and the plane on a grid line — so the half's samples are the
    full's, and its energy must be the full's once the dual patches on
    the plane count by half and the booking doubles.
    """
    full = _mesh()
    half = Mesh.from_grid(
        GridLines(x=np.linspace(A / 2, A, 7), y=np.linspace(0, B, 7), z=np.linspace(0, D, 17))
    ).with_boundary_conditions({"xmin": "ForceSymmetryPEC"})
    from magnelio.solver.fit_td import FITTimeDomainSolver

    # One time step for both grids: the leapfrog invariant depends on it.
    dt = min(spectral_dt(full, "normal"), spectral_dt(half, "normal"))
    energies = {}
    for name, mesh in (("full", full), ("half", half)):
        mode = sources.SourceFieldInitial.from_function(mesh.grid, name="m", E=_te201)
        mode.set_excitation(None)
        volume = monitors.MonitorFieldTime(name="v", corners=None, fields=["E", "H"], times=[0.0])
        solver = FITTimeDomainSolver(
            mesh=mesh,
            sources=[mode],
            monitors=[volume],
            total_time_steps=3,
            dt=dt,
            verbose=False,
            backend="numpy",
        )
        solver.run()
        rec = volume.recording
        energies[name] = float(rec.energy()[0])
        if name == "half":
            assert rec._ops.symmetry == ("xmin",)
            assert rec._ops.ends[0] == ("cut", "wall")
    assert energies["full"] > 0.0
    np.testing.assert_allclose(energies["half"], energies["full"], rtol=1e-9)
