"""Eigenmode ring-down: an initial field started from a stored mode (DD-224 Phase C).

The eigenmode solver and the time-domain march discretise the same
curl-curl operator, so a mode fed back as ``SourceFieldInitial`` must
ring at the frequency the eigensolver reported — the sharpest check
that the initial state (E on the primal edges, H half a leapfrog step
ahead) is consistent.
"""

import numpy as np

import magnelio as mio
from magnelio import monitors, sources
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh

A, B, D = 22.86e-3, 10.16e-3, 30e-3  # WR-90 cross-section, TE101 near 8.2 GHz
F_MAX = 12e9  # the band the monitors and the step estimate work in


def _mesh():
    grid = GridLines(
        x=np.linspace(0, A, 13),
        y=np.linspace(0, B, 7),
        z=np.linspace(0, D, 17),
    )
    return Mesh.from_grid(grid)


def _ringdown_frequency(result, monitor_name="probe"):
    """Peak of the probe spectrum [Hz], parabolically interpolated."""
    mon = result.monitors[monitor_name]
    sig = mon.component("Ey").ravel()
    t = mon.t
    spectrum = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))
    df = 1.0 / (sig.size * (t[1] - t[0]))
    k = int(spectrum.argmax())
    if 0 < k < spectrum.size - 1:
        y0, y1, y2 = np.log(spectrum[k - 1 : k + 2])
        return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * df
    return k * df


def _run_ringdown(mesh, source, t_end=8e-9):
    from magnelio.solver.stability import spectral_dt

    dt = spectral_dt(mesh, "normal")
    probe = monitors.MonitorFieldTime(
        name="probe",
        corners=((A / 2, B / 2, D / 2), (A / 2, B / 2, D / 2)),
        fields=["Ey"],
        times=np.arange(0.0, t_end, dt),
    )
    analysis = mio.AnalysisTD(
        mesh=mesh.with_sources([source]),
        monitors=[probe],
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
    )
    return analysis.run(excitations=[source.name], t_end=t_end, energy_stop_db=None)


def test_ringdown_matches_the_eigenfrequency(tmp_path):
    """Mode 0 out of a project, back in as the initial field."""
    mesh = _mesh()
    project = mio.AnalysisEigenmode(
        mesh=mesh, n_modes=1, verbose=False, project=tmp_path / "modes"
    ).run()
    f_eigen = float(project.eigenmodes.frequencies[0])

    source = sources.SourceFieldInitial.from_project(project, name="mode0")
    result = _run_ringdown(mesh, source)

    f_td = _ringdown_frequency(result)
    rel = abs(f_td / f_eigen - 1.0)
    assert rel < 5e-3, (
        f"ring-down {f_td / 1e9:.4f} GHz vs eigen {f_eigen / 1e9:.4f} GHz ({rel:.2%})"
    )


def test_amplitude_scales_and_energy_is_conserved():
    """The excitation's amplitude scales the field; a lossless box keeps its energy."""
    mesh = _mesh()
    result_eigen = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()
    field = result_eigen.field(0)
    source = sources.SourceFieldInitial(name="mode0", field=field)

    single = _run_ringdown(mesh, source, t_end=2e-9)
    doubled = mio.AnalysisTD(
        mesh=mesh.with_sources([source]),
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
    ).run(
        excitations=[mio.Excitation("mode0", amplitude=2.0)],
        t_end=2e-9,
        energy_stop_db=None,
    )
    energy_single = single.energy_trace["energy"]
    energy_double = doubled.energy_trace["energy"]
    np.testing.assert_allclose(energy_double, 4.0 * energy_single, rtol=1e-9)
    # A PEC box without losses holds its energy, and the recorded trace is
    # the quantity the leapfrog conserves — so every sample, not just a
    # running mean, sits on the starting value.
    spread = float(np.ptp(energy_single) / energy_single[0])
    assert spread < 5e-3, f"stored energy varied by {spread:.2%} over the run"
    assert single.excitation_signals[("mode0", 0)].values[0] == 1.0


# ── coupled resonator: an initial field next to a waveguide port ────────────

W_IRIS, T_IRIS, L_WG = 12e-3, 2e-3, 30e-3
TAN_DELTA = 5e-4
SIGMA = 2 * np.pi * 8.15e9 * 8.8541878128e-12 * TAN_DELTA


def _coupled_model(tan_delta):
    from magnelio import geo, ports

    sigma = SIGMA if tan_delta else 0.0
    fill = mio.Material.from_isotropic("fill", epsilon=1.0, sigma=sigma)
    model = mio.GeometryModel(background="pec")
    model.add(geo.Brick.from_corners((0, 0, 0), (A, B, D), material=fill))
    z0, z1 = D, D + T_IRIS
    model.add(
        geo.Brick.from_corners(
            ((A - W_IRIS) / 2, 0.0, z0), ((A + W_IRIS) / 2, B, z1), material=fill
        )
    )
    model.add(geo.Brick.from_corners((0, 0, z1), (A, B, z1 + L_WG), material=fill))
    model.add_port(ports.PortWaveguide(name="out", plane="zmax", n_modes=1))
    return model


def _te101(x, y, z):
    inside = (z >= 0.0) & (z <= D)
    e_y = np.where(
        inside,
        np.sin(np.pi * np.clip(x, 0.0, A) / A) * np.sin(np.pi * np.clip(z, 0.0, D) / D),
        0.0,
    )
    return np.zeros_like(e_y), e_y, np.zeros_like(e_y)


def _coupled_ringdown(tan_delta, t_end=25e-9):
    mesh = mio.Mesh.from_geometry(
        _coupled_model(tan_delta), mio.MeshControl(min_nodes_per_wavelength=10), f_max=12e9
    )
    source = sources.SourceFieldInitial.from_function(mesh.grid, name="mode0", E=_te101)
    result = mio.AnalysisTD(mesh=mesh.with_sources([source]), verbose=False, backend="numpy").run(
        excitations=["mode0"], t_end=t_end, energy_stop_db=None
    )
    trace = result.energy_trace
    t, w = trace["time"], trace["energy"]
    keep = w > w.max() * 1e-4
    slope = np.polyfit(t[keep], np.log(w[keep]), 1)[0]
    signal = np.asarray(result.signal("out").values)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    df = 1.0 / (signal.size * result.dt)
    k = int(spectrum.argmax())
    y0, y1, y2 = np.log(spectrum[k - 1 : k + 2])
    f0 = (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * df
    return f0, -2 * np.pi * f0 / slope, result


def test_initial_field_rings_out_through_a_waveguide_port():
    """The mode charges the cavity, the port carries the energy away."""
    f_loaded, q_ext, result = _coupled_ringdown(0.0)
    signal = np.asarray(result.signal("out").values)
    assert np.abs(signal).max() > 0.0, "nothing reached the port"
    # The iris pulls the resonance below the sealed TE101 (8.215 GHz here).
    assert 7.0e9 < f_loaded < 8.2e9, f"loaded resonance at {f_loaded / 1e9:.4f} GHz"
    # A lossless model with a port loses energy only through it.
    assert 50.0 < q_ext < 1000.0, f"Q_ext = {q_ext:.1f}"
    trace = result.energy_trace
    assert trace["energy"][-1] < 0.5 * trace["energy"][0], "the cavity did not ring down"


def test_q_budget_sum_rule():
    """1/Q_L = 1/Q_fill + 1/Q_ext with a homogeneously filled resonator."""
    f_ext, q_ext, _ = _coupled_ringdown(0.0)
    f_loaded, q_loaded, _ = _coupled_ringdown(TAN_DELTA)
    # A conductivity is a frequency-dependent loss tangent.
    q_fill = 2 * np.pi * f_loaded * 8.8541878128e-12 / SIGMA
    predicted = 1.0 / (1.0 / q_fill + 1.0 / q_ext)
    rel = abs(q_loaded / predicted - 1.0)
    assert rel < 0.03, (
        f"Q_L {q_loaded:.1f} vs sum rule {predicted:.1f} "
        f"(Q_fill {q_fill:.1f}, Q_ext {q_ext:.1f}): {rel:.2%}"
    )


def test_discrete_port_loads_the_mode_from_t_zero():
    """A pure-R port is a resistor without memory: no start problem at all.

    The same initial mode is marched with and without a discrete port
    across the cavity height.  Without it the box is lossless and holds
    its energy; with it the mode is damped and the port carries a
    signal from the first steps.
    """
    from magnelio import ports

    mesh = _mesh()
    source = sources.SourceFieldInitial.from_function(mesh.grid, name="mode0", E=_te101_closed)

    def march(port_specs):
        return mio.AnalysisTD(
            mesh=mesh.with_sources([source]),
            ports=port_specs,
            f_max=F_MAX,
            verbose=False,
            backend="numpy",
        ).run(excitations=["mode0"], t_end=5e-9, energy_stop_db=None)

    free = march(None)
    loaded = march(
        [ports.PortLumped(name="probe", start=(A / 2, 0.0, D / 2), end=(A / 2, B, D / 2), Z0=50.0)]
    )

    w_free = free.energy_trace["energy"]
    w_loaded = loaded.energy_trace["energy"]
    assert w_free[-1] > 0.9 * w_free[0], "the unloaded box should hold its energy"
    damping_db = 10 * np.log10(w_loaded[0] / w_loaded[-1])
    assert damping_db > 15.0, f"the port damped the mode by only {damping_db:.1f} dB"
    assert np.abs(np.asarray(loaded.signal("probe").values)).max() > 0.0


def _te101_closed(x, y, z):
    """TE101 of the sealed box (the module's own cavity)."""
    e_y = np.sin(np.pi * np.clip(x, 0.0, A) / A) * np.sin(np.pi * np.clip(z, 0.0, D) / D)
    return np.zeros_like(e_y), e_y, np.zeros_like(e_y)


def test_surface_impedance_walls_start_from_zero_and_give_the_right_q():
    """SIBC branch currents start quiescent — the decay still reads the wall Q.

    The perturbative surface-resistance evaluation on the eigenmode and
    the transient decay of the same mode are independent routes to the
    same loss, so their agreement gates the start.
    """
    from magnelio import post

    mesh = _mesh()
    eig = mio.AnalysisEigenmode(mesh=mesh, n_modes=1, verbose=False).run()
    f0 = float(eig.frequencies[0])
    q_perturbative = post.wall_loss_Q(eig, 0, sigma=5.8e7).Q

    source = sources.SourceFieldInitial.from_function(mesh.grid, name="mode0", E=_te101_closed)
    result = mio.AnalysisTD(
        mesh=mesh.with_sources([source]),
        f_max=F_MAX,
        wall_model="sibc",
        wall_sigma=5.8e7,
        verbose=False,
        backend="numpy",
    ).run(excitations=["mode0"], t_end=150e-9, energy_stop_db=None)

    w = result.energy_trace["energy"]
    assert w.max() <= 1.0000001 * w[0], "the SIBC start grew the stored energy"
    t = result.energy_trace["time"]
    q = -2 * np.pi * f0 / np.polyfit(t, np.log(w), 1)[0]
    rel = abs(q / q_perturbative - 1.0)
    assert rel < 0.02, f"ring-down Q_wall {q:.1f} vs perturbative {q_perturbative:.1f} ({rel:.2%})"
