"""SurfaceRecording — interpolation, round-trip and the placement rules."""

import numpy as np
import pytest

from magnelio import sources
from magnelio.fields import ComponentRecord, FaceRecord, SurfaceRecording
from magnelio.sources.field_surface import _rotation_matrix

_AXES = "xyz"


def _recording(n_t=5):
    """A three-face stub with a linear ramp per component."""
    c1 = np.linspace(-1.0, 1.0, 5)
    c2 = np.linspace(-2.0, 2.0, 4)
    faces = {}
    for name, axis in (("xmin", 0), ("ymax", 1), ("zmin", 2)):
        comps = {}
        for comp in ("Ey", "Ez", "Hy", "Hz") if axis == 0 else ("Ex", "Ez", "Hx", "Hz"):
            if comp[1] == _AXES[axis]:
                continue
            if comp.startswith("E"):
                values = np.tile(c1[None, :, None] + c2[None, None, :], (n_t, 1, 1))
                normals = (0.0,)
            else:
                base = c1[None, None, :, None] + c2[None, None, None, :]
                values = np.tile(base, (n_t, 2, 1, 1))
                values[:, 1] += 1.0
                normals = (0.0, 1.0)
            comps[comp] = ComponentRecord(c1=c1, c2=c2, normals=normals, values=values)
        faces[name] = FaceRecord(
            name=name,
            axis=axis,
            sign=-1.0 if name.endswith("min") else 1.0,
            plane=0.0,
            tangent_axes=tuple(a for a in range(3) if a != axis),
            components=comps,
        )
    return SurfaceRecording(
        name="rec",
        faces=faces,
        times=np.arange(n_t) * 1e-12,
        half_step=0.5e-12,
        bounds=((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)),
        open_faces=("xmin", "ymax", "zmin"),
    )


def test_interpolation_is_exact_on_the_recorded_positions():
    """A replay on the recording's own grid reads its samples, not a blur."""
    rec = _recording()
    face = rec.faces["zmin"]
    cr = face.components["Ex"]
    got = face.resample("Ex", cr.c1[:, None], cr.c2[None, :])
    assert np.allclose(got, cr.values)


def test_in_plane_interpolation_is_bilinear():
    rec = _recording()
    face = rec.faces["zmin"]
    got = face.resample("Ex", np.array([0.25]), np.array([0.5]))
    assert np.allclose(got, 0.75)


def test_h_interpolates_between_its_two_layers():
    """The magnetic samples span the normal, so any dual plane is reachable."""
    rec = _recording()
    face = rec.faces["zmin"]
    a = face.resample("Hx", np.array([0.0]), np.array([0.0]), normal=0.0)
    b = face.resample("Hx", np.array([0.0]), np.array([0.0]), normal=1.0)
    mid = face.resample("Hx", np.array([0.0]), np.array([0.0]), normal=0.5)
    assert np.allclose(b - a, 1.0)
    assert np.allclose(mid, 0.5 * (a + b))


def test_normal_query_is_clamped_not_extrapolated():
    rec = _recording()
    face = rec.faces["zmin"]
    far = face.resample("Hx", np.array([0.0]), np.array([0.0]), normal=17.0)
    edge = face.resample("Hx", np.array([0.0]), np.array([0.0]), normal=1.0)
    assert np.allclose(far, edge)


def test_time_weights_put_h_on_its_own_base():
    """E and H are half a step apart and are read on their own stamps."""
    rec = _recording()
    i0, i1, w = rec.time_weights(1e-12)
    assert (int(i0), int(i1), float(w)) == (0, 1, 1.0)
    i0, i1, w = rec.time_weights(1.5e-12, magnetic=True)
    assert (int(i0), int(i1), float(w)) == (0, 1, 1.0)


def test_geometry_properties():
    rec = _recording()
    assert rec.centre == (0.0, 0.0, 0.0)
    assert rec.size == (2.0, 4.0, 6.0)
    assert not rec.closed
    assert rec.interval == pytest.approx(1e-12)
    assert rec.duration == pytest.approx(4e-12)


def test_save_load_round_trip(tmp_path):
    rec = _recording()
    path = tmp_path / "rec.h5"
    rec.save(path)
    back = SurfaceRecording.load(path)
    assert back.name == rec.name
    assert set(back.faces) == set(rec.faces)
    assert back.open_faces == rec.open_faces
    assert np.allclose(back.times, rec.times)
    assert back.half_step == pytest.approx(rec.half_step)
    for name, face in rec.faces.items():
        other = back.faces[name]
        assert other.tangent_axes == face.tangent_axes
        for comp, cr in face.components.items():
            assert np.allclose(other.components[comp].values, cr.values)
            assert other.components[comp].normals == cr.normals


def test_load_rejects_a_foreign_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1.0])
    with pytest.raises(ValueError, match="surface recording"):
        SurfaceRecording.load(path)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_rotation_matrix_is_a_signed_permutation():
    r = _rotation_matrix(("z", 90))
    assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
    assert np.allclose(r @ np.array([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0])
    assert np.allclose(_rotation_matrix(("x", 360)), np.eye(3))
    assert np.allclose(_rotation_matrix(None), np.eye(3))


def test_free_rotation_angles_are_refused():
    """A tilted box has no samples where the target grid's patches sit."""
    with pytest.raises(ValueError, match="multiple of 90"):
        _rotation_matrix(("z", 30))


def test_rotation_axis_is_validated():
    with pytest.raises(ValueError, match="rotation axis"):
        _rotation_matrix(("w", 90))


def test_source_needs_a_recording():
    with pytest.raises(TypeError, match="needs recording="):
        sources.SourceFieldSurface(name="s")


def test_source_box_follows_the_recording_and_the_position():
    rec = _recording()
    src = sources.SourceFieldSurface(recording=rec, name="s", position=(1.0, 2.0, 3.0))
    lo, hi = src.corners
    assert np.allclose(lo, (0.0, 0.0, 0.0))
    assert np.allclose(hi, (2.0, 4.0, 6.0))


def test_rotation_turns_the_box_with_it():
    rec = _recording()
    src = sources.SourceFieldSurface(recording=rec, name="s", rotation=("z", 90))
    lo, hi = src.corners
    # x and y extents (2 and 4) swap under the quarter turn.
    assert np.allclose(np.asarray(hi) - np.asarray(lo), (4.0, 2.0, 6.0))


def test_source_takes_no_waveform():
    """The recording is the time function; an excitation only scales it."""
    rec = _recording()
    src = sources.SourceFieldSurface(recording=rec, name="s")
    from magnelio import signals

    with pytest.raises(TypeError, match="has no\n?\\s*waveform|no waveform"):
        src.set_excitation(signals.WaveformGaussian(f_max=1e9))
    src.set_excitation(None, amplitude=2.0, delay=1e-12)
    assert src.has_waveform is False
