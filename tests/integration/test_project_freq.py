"""Frequency-monitor persistence — DD-070 follow-up.

A :class:`MonitorFieldFrequency` accumulates a running DFT — a fixed-size
sum, not an append stream — so it cannot live in the SWMR ``results.h5``.
It gets its own ``fields_freq.h5``, written whole and atomically at each
checkpoint: the post-processing result the user plots *and*, transparently,
the resume source for the accumulator.  ``Project.monitors`` resolves it by
name into a ``_LoadedFreqMonitor``.  These tests pin:

* **Parity** — the reader reproduces the in-RAM monitor's raw DFT bins
  bit-for-bit (renormalisation stays a user step, as in RAM).
* **Live (partial) result** — a checkpointed run already exposes a
  converging partial DFT, distinct from the finished one.
* **Resume continuity** — the accumulator reloaded across a
  checkpoint/resume seam integrates on to a value bit-identical to an
  uninterrupted run (the checkpoint's E/H does not carry the DFT; it comes
  from ``fields_freq.h5``, step-checked against the checkpoint).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl, open_project, resume
from magnelio.geo import Brick
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency
from magnelio.ports import PortWaveguide

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9
N_TOTAL = 300
FREQS = np.linspace(4.0e9, 10.0e9, 3)


def _tem_analysis(project=None):
    """A parallel-plate TEM two-port with an xy-plane frequency monitor."""
    from magnelio.geo import GeometryModel  # noqa: PLC0415

    model = GeometryModel(
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    model.add(
        Brick(
            origin=(-A / 2, -B / 2, -LZ / 2),
            size=(A, B, LZ),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
        project=project,
        geometry=model,
    )
    analysis.monitors = (
        MonitorFieldFrequency(
            corners=((None, None, 0.0), (None, None, 0.0)),
            freqs=FREQS,
            fields=["E"],
            name="EHfreq",
        ),
    )
    return analysis


def _ref_freq_data():
    """Run in-RAM and return the monitor's raw DFT data (un-renormalised)."""
    an = _tem_analysis()
    an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    mon = an.monitors[0]
    return {c: v.copy() for c, v in mon.spectrum.cell_centred(squeeze=True).items()}


def _assert_freq_matches(reader_mon, ref_data, tag):
    assert set(reader_mon.components) == set(ref_data), f"{tag}: comps differ"
    assert np.array_equal(reader_mon.f, FREQS), f"{tag}: freq axis differs"
    for comp, ref in ref_data.items():
        got = reader_mon.spectrum.cell_centred([comp], squeeze=True)[comp]
        assert got.shape == ref.shape, f"{tag} {comp}: shape {got.shape} != {ref.shape}"
        assert np.array_equal(got, ref), (
            f"{tag} {comp}: not bit-exact, max|Δ|={float(np.max(np.abs(got - ref))):.3e}"
        )


# ═════════════════════════════════════════════════════════════════════
# Parity
# ═════════════════════════════════════════════════════════════════════


def test_streamed_freq_matches_in_ram(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data = _ref_freq_data()
    assert ref_data["Ez"].shape[0] == FREQS.size  # (n_freqs, <spatial>)

    p = tmp_path / "pp"
    proj = _tem_analysis(project=p).run(
        excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL
    )

    rmon = proj.monitors["EHfreq"]
    _assert_freq_matches(rmon, ref_data, "streamed")
    # It really is the frequency reader, resolved by name from fields_freq.h5.
    assert type(rmon).__name__ == "_LoadedFreqMonitor"
    assert (p / "runs" / "port1_mode0" / "fields_freq.h5").exists()

    # The plotting path (_hydrate) rebuilds a real monitor — region +
    # accumulators — whose data still matches, so .plot() works off-store.
    hyd = rmon._hydrate()
    for comp, ref in ref_data.items():
        assert np.array_equal(hyd.spectrum.cell_centred([comp], squeeze=True)[comp], ref), (
            f"hydrate {comp}"
        )
    assert hyd._region is not None and hyd._region.ndim == 2


# ═════════════════════════════════════════════════════════════════════
# Live (partial) result + resume continuity
# ═════════════════════════════════════════════════════════════════════


def test_freq_partial_then_bit_exact_across_resume(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data = _ref_freq_data()

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
        checkpoint_interval=40,
    )
    # A partial DFT is already readable — same shape, but not yet the full sum.
    partial = open_project(p).monitors["EHfreq"]
    pz = partial.spectrum.cell_centred(["Ez"], squeeze=True)["Ez"]
    assert pz.shape == ref_data["Ez"].shape
    assert not np.array_equal(pz, ref_data["Ez"]), "partial DFT should differ from the finished one"

    proj = resume(p, excited=("port1", 0), total_time_steps=N_TOTAL, verbose=False)
    _assert_freq_matches(proj.monitors["EHfreq"], ref_data, "resume")


# ═════════════════════════════════════════════════════════════════════
# Sub-sampled accumulation (DD-140)
# ═════════════════════════════════════════════════════════════════════


def test_interval_survives_the_store_round_trip(tmp_path):
    """The sampling the bins were integrated with travels with them.

    Without it a stored DFT cannot be told apart from one accumulated
    every step, and the reader would rehydrate a monitor that resumes
    on a different stride than the run it continues.
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")

    # A stride of 2, expressed relative to the step the run will use
    # (the stride resolution floors interval / dt, so 2.5 dt lands on
    # 2) — coarse enough to change the bins, still ~12 samples per
    # period at the monitor's 10 GHz top frequency (no margin
    # warning).  An absolute interval broke once when the time step
    # grew under it and the stride quietly became 1, making the
    # sub-sampled bins identical to the every-step reference.
    from magnelio.solver.stability import spectral_dt

    an = _tem_analysis(project=str(tmp_path / "proj"))
    interval = 2.5 * spectral_dt(an.mesh, "normal")
    an.monitors = (
        MonitorFieldFrequency(
            corners=((None, None, 0.0), (None, None, 0.0)),
            freqs=FREQS,
            fields=["E"],
            interval=interval,
            name="EHfreq",
        ),
    )
    an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)

    project = open_project(tmp_path / "proj")
    mon = project.monitors["EHfreq"]
    assert mon.interval == pytest.approx(interval, rel=1e-12)
    assert mon._hydrate().interval == pytest.approx(interval, rel=1e-12)

    # A sub-sampled run is a different measurement from an every-step
    # one — close, but not the same numbers, which is what makes
    # recording the interval necessary rather than decorative.  Compare
    # on the component that actually carries the mode: Ez is tangential
    # to the PEC plane the monitor sits on, so it holds only roundoff.
    ref = _ref_freq_data()
    comp = max(ref, key=lambda c: float(np.max(np.abs(ref[c]))))
    got = mon.spectrum.cell_centred([comp], squeeze=True)[comp]
    scale = float(np.max(np.abs(ref[comp])))
    assert not np.array_equal(got, ref[comp])
    assert float(np.max(np.abs(got - ref[comp]))) < 1e-2 * scale


def test_stored_bins_without_interval_read_as_every_step(tmp_path):
    """Files written before the field existed stay readable."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")

    an = _tem_analysis(project=str(tmp_path / "proj"))
    an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    project = open_project(tmp_path / "proj")
    assert project.monitors["EHfreq"].interval is None


# ═════════════════════════════════════════════════════════════════════
# Phasor convention across releases (DD-268)
# ═════════════════════════════════════════════════════════════════════


def _demodernise_freq_file(path):
    """Rewrite fields_freq.h5 the way releases up to v0.7.0 wrote it.

    Their running DFT summed ``e^{+jwt}``, so every bin is the conjugate
    of the one written today, and no file carried the convention stamp.
    """
    import h5py

    with h5py.File(path, "r+") as f:
        assert f.attrs["phasor_convention"] == "exp(+jwt)"
        del f.attrs["phasor_convention"]
        for name in f:
            bg = f[name]["bins"]
            for comp in bg:
                bg[comp][...] = np.conj(bg[comp][()])


def test_legacy_phasor_file_reads_conjugated(tmp_path):
    """A store of the e^{-jwt} era hands out today's phasors, exactly."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data = _ref_freq_data()

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL
    )
    _demodernise_freq_file(p / "runs" / "port1_mode0" / "fields_freq.h5")

    _assert_freq_matches(open_project(p).monitors["EHfreq"], ref_data, "legacy")


def test_legacy_partial_file_resumes_bit_exact(tmp_path):
    """A partial sum of the old era continues into a run of the new one."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data = _ref_freq_data()

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
        checkpoint_interval=40,
    )
    ff = p / "runs" / "port1_mode0" / "fields_freq.h5"
    _demodernise_freq_file(ff)

    proj = resume(p, excited=("port1", 0), total_time_steps=N_TOTAL, verbose=False)
    _assert_freq_matches(proj.monitors["EHfreq"], ref_data, "legacy resume")

    # The finished run rewrote the file, so it is stamped again.
    import h5py

    with h5py.File(ff, "r") as f:
        assert f.attrs["phasor_convention"] == "exp(+jwt)"


def test_unknown_phasor_convention_is_rejected(tmp_path):
    """An unreadable stamp fails loudly rather than degrading silently."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    import h5py

    from magnelio.io._schema import ProjectSchemaError

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL
    )
    with h5py.File(p / "runs" / "port1_mode0" / "fields_freq.h5", "r+") as f:
        f.attrs["phasor_convention"] = "exp(-jwt-ish)"

    with pytest.raises(ProjectSchemaError, match="phasor convention"):
        open_project(p).monitors["EHfreq"].spectrum
