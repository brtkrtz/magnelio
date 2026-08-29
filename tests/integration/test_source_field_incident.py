"""The general incident field (DD-224 Phase C) against the analytic plane wave.

``SourceFieldIncident`` evaluates a user field on the six TF/SF box
faces every step; ``SourcePlaneWave`` folds the same wave into a delay
table.  Spelling the plane wave out as a general field must therefore
reproduce the specialised source to round-off, and an incident field
that no single ``SourcePlaneWave`` can express — two waves crossing at
right angles — must match the two sources driven together and leave
the scattered-field region quiet.

Note that the incident field has to solve the free-space Maxwell
equations itself: a transversally tapered "beam" spelled as
``E_x·exp(-r²/w²)`` with ``H_y = E_x/η₀`` is not a solution (it lacks
the longitudinal components the taper implies) and leaks out of the
box at the level of the total field — measured 116 %.
"""

import numpy as np

import magnelio as mio
from magnelio import geo, monitors, signals, sources
from magnelio.constants import C0, ETA0

F_MAX = 15e9
L = 12e-3


def _model():
    model = mio.GeometryModel(
        boundary_conditions=dict.fromkeys(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), "CPML"),
    )
    model.add(geo.Brick(origin=(-L / 2, -L / 2, -L / 2), size=(L, L, L), material="air"))
    return model


def _run(source, probes):
    """One march of *source* (or a list of them); ``probes`` maps a name to a box."""
    src_list = source if isinstance(source, list) else [source]
    model = _model()
    for src in src_list:
        model.add_source(src)
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=10), f_max=F_MAX)
    monitor_list = [
        monitors.MonitorFieldTime(
            name=name,
            corners=corners,
            fields=["E"],
            # the Gaussian peaks at 267 ps; the box face is 13 ps away
            times=[280e-12, 310e-12],
        )
        for name, corners in probes.items()
    ]
    analysis = mio.AnalysisTD(mesh=mesh, monitors=monitor_list, verbose=False, backend="numpy")
    result = analysis.run(
        excitations=[
            mio.Excitation(src.name, waveform=signals.WaveformGaussian(f_max=F_MAX))
            for src in src_list
        ],
        t_end=320e-12,
        energy_stop_db=None,
    )
    return {name: result.monitors[name].data for name in probes}


def _plane_wave_field(x, y, z, t, drive):
    """The +z plane wave of SourcePlaneWave(direction=+z, polarization=+x)."""
    f = drive(t - z / C0)
    zero = np.zeros_like(np.asarray(f, dtype=float))
    return (f, zero, zero), (zero, f / ETA0, zero)


def test_general_field_reproduces_the_plane_wave():
    """The same wave through the general path and the analytic one."""
    box = ((-4e-3, -4e-3, -4e-3), (4e-3, 4e-3, 4e-3))
    probes = {"inside": ((-2e-3, -2e-3, -2e-3), (2e-3, 2e-3, 2e-3))}
    analytic = _run(
        sources.SourcePlaneWave(
            name="pw", direction=(0, 0, 1), polarization=(1, 0, 0), corners=box
        ),
        probes,
    )["inside"]
    general = _run(
        sources.SourceFieldIncident(name="pw", field=_plane_wave_field, corners=box),
        probes,
    )["inside"]
    peak = np.abs(analytic["Ex"]).max()
    assert peak > 0.1, f"the probe saw no wave (peak {peak:.3g} V/m)"
    for comp in ("Ex", "Ey", "Ez"):
        err = np.abs(general[comp] - analytic[comp]).max() / peak
        assert err < 1e-12, f"{comp}: general field deviates by {err:.3g} of the peak"


def _crossed_field(x, y, z, t, drive):
    """Two plane waves at right angles — an exact free-space solution.

    +z with E along x, and +x with E along y; each carries
    ``H = k̂ × Ê / η₀``.
    """
    f_z = drive(t - z / C0)
    f_x = drive(t - x / C0)
    zero = np.zeros_like(np.asarray(f_z + f_x, dtype=float))
    E = (f_z + zero, f_x + zero, zero)
    H = (zero, f_z / ETA0 + zero, f_x / ETA0 + zero)
    return E, H


def test_crossed_waves_match_two_simultaneous_sources():
    """A field no single plane-wave source can express, against the pair.

    The TF/SF corrections are linear in the incident field, so two
    ``SourcePlaneWave`` sources excited in the same run must reproduce
    what one general source computes from their sum — and the
    scattered-field shell must stay at the dispersion floor, because
    the superposition solves Maxwell exactly.
    """
    box = ((-4e-3, -4e-3, -4e-3), (4e-3, 4e-3, 4e-3))
    probes = {
        "inside": ((-2e-3, -2e-3, -2e-3), (2e-3, 2e-3, 2e-3)),
        "shell": ((-5e-3, -5e-3, -5e-3), (5e-3, -4.5e-3, 5e-3)),
    }
    general = _run(
        sources.SourceFieldIncident(name="crossed", field=_crossed_field, corners=box),
        probes,
    )
    pair = _run(
        [
            sources.SourcePlaneWave(
                name="pw_z", direction=(0, 0, 1), polarization=(1, 0, 0), corners=box
            ),
            sources.SourcePlaneWave(
                name="pw_x", direction=(1, 0, 0), polarization=(0, 1, 0), corners=box
            ),
        ],
        probes,
    )
    peak = max(np.abs(pair["inside"][c]).max() for c in ("Ex", "Ey", "Ez"))
    assert peak > 0.1, f"the probe saw no wave (peak {peak:.3g} V/m)"
    for comp in ("Ex", "Ey", "Ez"):
        err = np.abs(general["inside"][comp] - pair["inside"][comp]).max() / peak
        assert err < 1e-12, f"{comp}: general field deviates by {err:.3g} of the peak"
    leak = max(np.abs(general["shell"][c]).max() for c in ("Ex", "Ey", "Ez")) / peak
    assert leak < 0.02, f"scattered-field leak {leak:.3%} of the total field"
