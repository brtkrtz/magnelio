"""ParaView session export on a real streamed run (DD-115, DD-262).

A project-backed TEM run with a plane time monitor, a volume time
monitor and a frequency monitor; ``export_paraview()`` must leave a
ready-to-open session in the run directory: per-monitor VTK series over
time, the frequency DFT as a ``.vtr``-per-frequency series (values and
cell ordering gated against ``fields_freq.h5``), the per-solid
``geometry.vtm``, and the generated ``paraview_open.py`` whose embedded
config carries slice planes and a positive glyph clip cap.  The run
itself writes none of it.  The ``pvpython`` state bake and reload run
as separate, environment-gated tests.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl, open_project
from magnelio.geo import Brick
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldFrequency, MonitorFieldTime
from magnelio.ports import PortWaveguide
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9
N_TOTAL = 300
FREQS = (6.0e9, 10.0e9)


def _tem_analysis(project):
    """Parallel-plate TEM two-port with time + frequency monitors."""
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
            name="line",
        )
    )
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=F_MAX)
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    times = np.linspace(0.15, 0.9, 4) * N_TOTAL * dt
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
        MonitorFieldTime(
            corners=((None, None, 0.0), (None, None, 0.0)),
            times=times,
            fields=["E"],
            name="Eplane",
        ),
        MonitorFieldTime(times=times, fields=["E", "H"], name="Evol"),
        MonitorFieldFrequency(
            corners=((None, None, 0.0), (None, None, 0.0)),
            freqs=FREQS,
            fields=["E"],
            name="Efreq",
        ),
    )
    return analysis


@pytest.fixture(scope="module")
def session_run(tmp_path_factory):
    """One streamed run, kept with the analysis whose monitors fed it."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    pytest.importorskip("vtk")
    p = tmp_path_factory.mktemp("pv") / "pp"
    analysis = _tem_analysis(p)
    analysis.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    open_project(p).export_paraview(bake_state=False)
    return p, analysis


@pytest.fixture(scope="module")
def session_project(session_run):
    return session_run[0]


def _config_from_script(script_path):
    text = script_path.read_text(encoding="utf-8")
    ns: dict = {}
    exec(text[: text.index("def build")], ns)
    return ns["CONFIG"]


def test_session_artifacts_written(session_project):
    run_dir = session_project / "runs" / "port1_mode0"
    assert (session_project / "geometry.vtm").exists()
    assert not (session_project / "geometry.stl").exists()
    assert (run_dir / "paraview_open.py").exists()
    pv = run_dir / "paraview"
    assert (pv / "Eplane.pvd").exists()
    assert (pv / "Evol.pvd").exists()
    assert (pv / "Efreq.pvd").exists()
    vtrs = sorted((pv / "Efreq").glob("f_*.vtr"))
    assert len(vtrs) == len(FREQS)
    # One frame per recorded instant for the time monitors (DD-259).
    reader = open_project(session_project).monitors_for(("port1", 0))["Eplane"]
    assert len(sorted((pv / "Eplane").glob("t_*.vtr"))) == reader.t.size > 0
    assert not (run_dir / "fields.xdmf").exists()


def test_script_config(session_project):
    run_dir = session_project / "runs" / "port1_mode0"
    config = _config_from_script(run_dir / "paraview_open.py")
    assert config["geometry"] == "../../geometry.vtm"
    assert [m["name"] for m in config["materials"]] == ["air"]

    mons = {m["name"]: m for m in config["monitors"]}
    assert set(mons) == {"Eplane", "Evol", "Efreq"}
    # Plane monitors carry their normal; the volume monitor carries the
    # three slice planes with a deterministic default.
    assert mons["Eplane"]["planar_normal"] == "z"
    assert mons["Eplane"]["slice_axes"] == []
    assert mons["Evol"]["slice_axes"] == ["x", "y", "z"]
    assert mons["Evol"]["default_axis"] == "y"  # shortest extent
    # Every monitor recorded a complete E triple -> glyph spec with a
    # positive, finite clip cap and a bounded full-arrow length.
    for name, arrays in (
        ("Eplane", ["E"]),
        ("Evol", ["E"]),
        # A frequency monitor offers phase 0 AND phase -90 degrees; the
        # real part alone hides the field wherever it is mostly imaginary.
        ("Efreq", ["E_re", "E_im"]),
    ):
        glyph = mons[name]["glyph"]
        assert glyph is not None and glyph["arrays"] == arrays
        assert 0.0 < glyph["cap"] < np.inf
        assert glyph["length"] == pytest.approx(mons[name]["l_ref"])
        # Length factor is dimensionless and bounded: cap maps to exactly 1.
        assert 0.2 <= glyph["exponent"] <= 1.0
        # Volume glyphs prune the field-free cells below a fraction of the cap.
        assert 0.0 < glyph["threshold"] < glyph["cap"]


def test_store_and_ram_monitor_agree(session_run):
    """Both sides divide out the excitation on their own, so both must
    land on the same numbers — neither having been asked to (DD-170).

    The absolute scale is gated against analytic physics elsewhere
    (``test_port_units``); what this pins is that the two channels
    cannot drift apart, which they would the moment one of them sampled
    the reference waveform over a different span than the other.
    """
    path, analysis = session_run
    live = next(m for m in analysis.monitors if m.name == "Efreq")
    stored = open_project(path).monitors["Efreq"]

    assert live.is_renormalized, "the streamed run must renormalise the caller's monitor"
    for comp in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            stored.spectrum.cell_centred([comp], squeeze=True)[comp],
            live.spectrum.cell_centred([comp], squeeze=True)[comp],
            rtol=1e-12,
        )


def test_freq_vtr_matches_the_monitor(session_project):
    """The renderer and ``.data`` must report the same quantity (DD-170).

    Gated against the monitor rather than against the raw HDF5 bins: the
    export divides by the run's excitation spectrum exactly as the
    monitor does, so a value read in ParaView means what the same value
    means in a script.  A drift between the two channels shows up here as
    the excitation spectrum itself, orders of magnitude wide.
    """
    import vtk  # noqa: PLC0415
    from vtk.util import numpy_support as ns  # noqa: PLC0415

    run_dir = session_project / "runs" / "port1_mode0"
    mon = open_project(session_project).monitors["Efreq"]
    bins = {c: mon.spectrum.cell_centred([c], squeeze=True)[c] for c in ("Ex", "Ey", "Ez")}
    assert np.asarray(mon.freqs) == pytest.approx(list(FREQS))

    # Cell ordering: monitor-native (nx, ny[, nz]) -> VTK x-fastest, which
    # is the full axis reversal either way (the monitor squeezes the
    # collapsed axis, the raw bins keep it).
    def cells(a):
        return np.ascontiguousarray(a.T).ravel()

    for fi in range(len(FREQS)):
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(run_dir / "paraview" / "Efreq" / f"f_{fi:04d}.vtr"))
        reader.Update()
        grid = reader.GetOutput()
        cd = grid.GetCellData()
        for comp in ("Ex", "Ey", "Ez"):
            expect = cells(bins[comp][fi])
            got_re = ns.vtk_to_numpy(cd.GetArray(f"{comp}_re"))
            got_im = ns.vtk_to_numpy(cd.GetArray(f"{comp}_im"))
            np.testing.assert_array_equal(got_re, expect.real)
            np.testing.assert_array_equal(got_im, expect.imag)
        # Vector + complex-magnitude convenience arrays.
        e_re = ns.vtk_to_numpy(cd.GetArray("E_re"))
        assert e_re.shape[1] == 3
        mag = ns.vtk_to_numpy(cd.GetArray("|E|"))
        stack = np.stack([cells(bins[c][fi]) for c in ("Ex", "Ey", "Ez")], axis=-1)
        np.testing.assert_allclose(mag, np.sqrt(np.sum(np.abs(stack) ** 2, axis=-1)), rtol=1e-12)


def test_export_paraview_regenerates(session_project):
    proj = open_project(session_project)
    out = proj.export_paraview(glyph_percentile=90.0, bake_state=False)
    assert out["state"] is None
    assert sorted(out["monitors"]) == ["Efreq", "Eplane", "Evol"]
    config = _config_from_script(out["script"])
    assert {m["name"] for m in config["monitors"]} == {"Eplane", "Evol", "Efreq"}


@pytest.mark.skipif(shutil.which("pvpython") is None, reason="pvpython not installed")
def test_pvsm_bake(session_project, monkeypatch):
    monkeypatch.setenv("MAGNELIO_PVSM_BAKE", "1")
    proj = open_project(session_project)
    out = proj.export_paraview()
    state = out["state"]
    assert state is not None and state.exists()
    text = state.read_text(encoding="utf-8", errors="replace")
    assert "ServerManagerState" in text
    # The baked pipeline carries the pre-built session (DD-262): one
    # Python filter per monitor with its scripts, one cut with its
    # arrows, the hidden volume arrows, the plane-linked geometry cut
    # and the registered slice<->clip link that makes them drag together.
    for marker in (
        "Evol_field",
        "Evol_slice",
        "Evol_arrows",
        "Evol_volume",
        "Evol_volume_arrows",
        "Efreq_field",
        "Efreq_arrows_im",
        # Planar monitors go through the same filter — no branch is left
        # on the computational grid.
        "Eplane_field",
        "Eplane_arrows",
        "geometry_cut_Evol",
        '<ProxyLink name="plane_Evol"',
        'type="ProgrammableFilter"',
        'name="InformationScript"',
        "E_len",
    ):
        assert marker in text, marker
    # The proxy chain of DD-115 is gone from the pipeline browser.
    for gone in ("Evol_lattice", "Evol_slice_y", "Evol_E_dir", "Efreq_E_im_dir", "Evol_points"):
        assert gone not in text, gone


_RELOAD_PROBE = """
import sys
import numpy
if not hasattr(numpy, "in1d"):
    numpy.in1d = numpy.isin
from paraview import simple
simple.LoadState(sys.argv[1])
field = simple.FindSource("Evol_field")
# ParaView names a loaded reader after its file.
reader = simple.FindSource("Evol.pvd") or simple.FindSource("Evol")
sl = simple.FindSource("Evol_slice")
arrows = simple.FindSource("Evol_arrows")
for proxy in (reader, field, sl, arrows):
    proxy.UpdatePipeline()
print("SLICE_POINTS", sl.GetDataInformation().GetNumberOfPoints())
print("ARROW_CELLS", arrows.GetDataInformation().GetNumberOfCells())
print("TIMES_MATCH", list(field.TimestepValues) == list(reader.TimestepValues))
"""


@pytest.mark.skipif(shutil.which("pvpython") is None, reason="pvpython not installed")
def test_pvsm_reloads_with_field_on_the_cut(session_project, tmp_path, monkeypatch):
    """A second ParaView finds the cut populated and the time axis intact.

    The one test that catches an extent negotiation or preamble failure
    inside the Python filter: the state file may carry every proxy and
    still show an empty cut.
    """
    import subprocess  # noqa: PLC0415

    monkeypatch.setenv("MAGNELIO_PVSM_BAKE", "1")
    state = open_project(session_project).export_paraview()["state"]
    assert state is not None
    probe = tmp_path / "probe.py"
    probe.write_text(_RELOAD_PROBE, encoding="utf-8")
    proc = subprocess.run(
        [shutil.which("pvpython"), "--force-offscreen-rendering", str(probe), str(state)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr[-2000:]
    values = dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
    assert int(values["SLICE_POINTS"]) > 0
    assert int(values["ARROW_CELLS"]) > 0
    assert values["TIMES_MATCH"] == "True"


def test_run_close_writes_no_paraview_artefacts(tmp_path):
    """The run leaves the store alone; the session is a step the user takes (DD-262)."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    pytest.importorskip("vtk")
    p = tmp_path / "pp"
    _tem_analysis(p).run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=60)
    run_dir = p / "runs" / "port1_mode0"
    assert (run_dir / "results.h5").exists()
    assert not (run_dir / "paraview").exists()
    assert not (run_dir / "paraview_open.py").exists()
    assert not (p / "geometry.vtm").exists()
    out = open_project(p).export_paraview(bake_state=False)
    assert (run_dir / "paraview").is_dir()
    assert (run_dir / "paraview_open.py").exists()
    assert (p / "geometry.vtm").exists()
    assert _config_from_script(out["script"])["geometry"] == "../../geometry.vtm"
