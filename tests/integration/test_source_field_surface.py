"""Huygens coupling (DD-226): record a surface, replay it as a source.

The equivalence principle says the tangential fields on a closed box
stand for everything the box encloses.  Two properties pin that down:
replayed in a second model, the recording must reproduce the field
*outside* the box and leave the *inside* quiet.

On the recording's own grid the replay is exact — the interpolation
reads its own samples — which is the sharp form of the test.  Across
grids it is limited by the interpolation, and the numbers quoted in
DD-226 come from the internal record `investigations/api-blueprint/
phase-d/` (an internal dossier, kept outside the public repository).
"""

import numpy as np
import pytest

import magnelio as mio
from magnelio import geo, monitors, signals, sources

F_MAX = 15e9
L = 24e-3
BOX = ((-6e-3, -6e-3, -6e-3), (6e-3, 6e-3, 6e-3))
OUTSIDE = ((7e-3, -1e-3, -1e-3), (9e-3, 1e-3, 1e-3))
INSIDE = ((-1e-3, -1e-3, -1e-3), (1e-3, 1e-3, 1e-3))
TIMES = list(np.arange(20e-12, 200e-12, 5e-12))
T_END = 210e-12


def _air():
    m = mio.GeometryModel(
        boundary_conditions=dict.fromkeys(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), "CPML"),
    )
    m.add(geo.Brick(origin=(-L / 2, -L / 2, -L / 2), size=(L, L, L), material="air"))
    return m


def _mesh(model):
    return mio.Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=12), f_max=F_MAX)


def _blob(x, y, z):
    """A localised pulse — an outgoing field with a radiator inside the box."""
    a = np.exp(-(x**2 + y**2 + z**2) / (1.5e-3) ** 2)
    return (a, np.zeros_like(a), np.zeros_like(a))


def _probes():
    return [
        monitors.MonitorFieldTime(name="out", corners=OUTSIDE, fields=["E"], times=TIMES),
        monitors.MonitorFieldTime(name="in", corners=INSIDE, fields=["E"], times=TIMES),
    ]


def _flat(a):
    a = np.asarray(a)
    return a.reshape(a.shape[0], -1).mean(axis=1)


@pytest.fixture(scope="module")
def recorded():
    """Record the blob's outgoing field, and the field outside the box.

    A zero-amplitude incident source contributes the same grid planes
    the replay run's own box will ask for, so both runs march on one
    grid and the replay can be held to the exact standard.
    """
    model = _air()
    grid = _mesh(model).grid
    model.add_source(sources.SourceFieldInitial.from_function(grid, name="blob", E=_blob))
    model.add_source(
        sources.SourcePlaneWave(
            name="pad", direction=(0, 0, 1), polarization=(1, 0, 0), corners=BOX
        )
    )
    mesh = _mesh(model)
    surface = monitors.MonitorFieldSurface(name="box", corners=BOX, interval=1e-13)
    analysis = mio.AnalysisTD(
        mesh=mesh, monitors=[surface, *_probes()], verbose=False, backend="numpy"
    )
    result = analysis.run(
        excitations=[
            mio.Excitation("blob", amplitude=1.0),
            mio.Excitation("pad", waveform=signals.WaveformGaussian(f_max=F_MAX), amplitude=0.0),
        ],
        t_end=T_END,
        energy_stop_db=None,
    )
    return (
        surface.recording(),
        _flat(result.monitors["out"].data["Ex"]),
        _flat(result.monitors["in"].data["Ex"]),
    )


def _replay(recording, **source_kwargs):
    model = _air()
    grid = _mesh(model).grid
    # Same two sources as the recording run, so the grid is the same;
    # the blob is switched off and the surface drives instead.
    model.add_source(sources.SourceFieldInitial.from_function(grid, name="blob", E=_blob))
    model.add_source(
        sources.SourceFieldSurface(recording=recording, name="box", corners=BOX, **source_kwargs)
    )
    analysis = mio.AnalysisTD(mesh=_mesh(model), monitors=_probes(), verbose=False, backend="numpy")
    result = analysis.run(
        excitations=[
            mio.Excitation("blob", amplitude=0.0),
            mio.Excitation("box", amplitude=1.0),
        ],
        t_end=T_END,
        energy_stop_db=None,
    )
    return (
        _flat(result.monitors["out"].data["Ex"]),
        _flat(result.monitors["in"].data["Ex"]),
    )


def test_the_recording_is_closed_and_sampled(recorded):
    recording, _out, _in = recorded
    assert recording.closed
    assert len(recording.faces) == 6
    assert recording.times.size > 50
    for face in recording.faces.values():
        assert set(face.components) == {
            f"{kind}{'xyz'[a]}" for kind in "EH" for a in range(3) if a != face.axis
        }


def test_replay_reproduces_the_field_outside_the_box(recorded):
    """On the recording's own grid the replay reads its own samples."""
    recording, out_ref, _in_ref = recorded
    out, _inside = _replay(recording)
    peak = np.abs(out_ref).max()
    assert np.abs(out).max() == pytest.approx(peak, rel=1e-3)
    # Single-precision round-off, measured at -84 dB.
    assert np.abs(out - out_ref).max() < 1e-3 * peak


def test_replay_leaves_the_inside_quiet(recorded):
    """Whatever radiated inside is replaced by the surface, not duplicated."""
    recording, out_ref, in_ref = recorded
    _out, inside = _replay(recording)
    assert np.abs(in_ref).max() > np.abs(out_ref).max()  # the radiator was in there
    assert np.abs(inside).max() < 1e-3 * np.abs(in_ref).max()


def test_amplitude_scales_the_replay(recorded):
    """The excitation carries no waveform, only a scale factor."""
    recording, _out_ref, _in_ref = recorded
    model = _air()
    grid = _mesh(model).grid
    model.add_source(sources.SourceFieldInitial.from_function(grid, name="blob", E=_blob))
    model.add_source(sources.SourceFieldSurface(recording=recording, name="box", corners=BOX))
    analysis = mio.AnalysisTD(mesh=_mesh(model), monitors=_probes(), verbose=False, backend="numpy")
    result = analysis.run(
        excitations=[
            mio.Excitation("blob", amplitude=0.0),
            mio.Excitation("box", amplitude=0.5),
        ],
        t_end=T_END,
        energy_stop_db=None,
    )
    half = _flat(result.monitors["out"].data["Ex"])
    full, _inside = _replay(recording)
    assert np.abs(half).max() == pytest.approx(0.5 * np.abs(full).max(), rel=2e-3)


def test_file_round_trip_drives_the_same_field(recorded, tmp_path):
    """The exchange format is the recording file, not a shared session."""
    recording, _out_ref, _in_ref = recorded
    path = tmp_path / "surface.h5"
    recording.save(path)
    reloaded = sources.SourceFieldSurface.from_file(path, name="box", corners=BOX).recording
    out_direct, _a = _replay(recording)
    out_file, _b = _replay(reloaded)
    assert np.allclose(out_file, out_direct)
