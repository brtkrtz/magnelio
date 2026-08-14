"""ParaView session export on a real streamed run (DD-115).

A project-backed TEM run with a plane time monitor, a volume time
monitor and a frequency monitor must leave a ready-to-open session in
the run directory: per-monitor XDMF descriptors, the frequency DFT as a
``.vtr``-per-frequency series (values and cell ordering gated against
``fields_freq.h5``), the per-solid ``geometry.vtm``, and the generated
``paraview_open.py`` whose embedded config carries slice planes and a
positive glyph clip cap.  The ``pvpython`` state bake runs as a
separate, environment-gated test.
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
def session_project(tmp_path_factory):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    pytest.importorskip("vtk")
    p = tmp_path_factory.mktemp("pv") / "pp"
    _tem_analysis(p).run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    return p


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
    assert (pv / "Eplane.xdmf").exists()
    assert (pv / "Evol.xdmf").exists()
    assert (pv / "Efreq.pvd").exists()
    vtrs = sorted((pv / "Efreq").glob("f_*.vtr"))
    assert len(vtrs) == len(FREQS)
    # Per-monitor descriptors reference the run's results.h5 one level up.
    assert "../results.h5:/monitors/Eplane/" in (pv / "Eplane.xdmf").read_text()


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


def test_freq_vtr_matches_h5(session_project):
    import h5py  # noqa: PLC0415
    import vtk  # noqa: PLC0415
    from vtk.util import numpy_support as ns  # noqa: PLC0415

    run_dir = session_project / "runs" / "port1_mode0"
    with h5py.File(run_dir / "fields_freq.h5", "r") as f:
        bins = {c: f["Efreq"]["bins"][c][()] for c in ("Ex", "Ey", "Ez")}
        freqs = f["Efreq"]["freqs"][()]
    assert freqs == pytest.approx(list(FREQS))

    for fi in range(len(FREQS)):
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(run_dir / "paraview" / "Efreq" / f"f_{fi:04d}.vtr"))
        reader.Update()
        grid = reader.GetOutput()
        cd = grid.GetCellData()
        # Cell ordering: monitor-native (nx, ny, nz) -> VTK x-fastest.
        for comp in ("Ex", "Ey", "Ez"):
            expect = np.ascontiguousarray(bins[comp][fi].transpose(2, 1, 0)).ravel()
            got_re = ns.vtk_to_numpy(cd.GetArray(f"{comp}_re"))
            got_im = ns.vtk_to_numpy(cd.GetArray(f"{comp}_im"))
            np.testing.assert_array_equal(got_re, expect.real)
            np.testing.assert_array_equal(got_im, expect.imag)
        # Vector + complex-magnitude convenience arrays.
        e_re = ns.vtk_to_numpy(cd.GetArray("E_re"))
        assert e_re.shape[1] == 3
        mag = ns.vtk_to_numpy(cd.GetArray("|E|"))
        stack = np.stack(
            [
                np.ascontiguousarray(bins[c][fi].transpose(2, 1, 0)).ravel()
                for c in ("Ex", "Ey", "Ez")
            ],
            axis=-1,
        )
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
    # The baked pipeline carries the pre-built session: slice planes,
    # glyph arrows, the plane-linked geometry cuts, and the registered
    # slice<->clip plane links that make them drag together.
    for marker in (
        "Evol_slice_y",
        "Evol_arrows_E_y",
        "Efreq_E_im_dir",
        # Every arrow set sits on an evenly spaced lattice: one sized for
        # in-plane density feeding the cuts, a coarser one for the
        # whole-volume view behind its field threshold.
        "Evol_lattice",
        "Evol_lattice_volume",
        "Evol_volume_E",
        "Evol_volume_region_E",
        # Planar monitors are resampled too — no branch is left on the
        # computational grid.
        "Eplane_lattice",
        "geometry_cut_Evol_y",
        "Efreq",
        '<ProxyLink name="plane_Evol_y"',
    ):
        assert marker in text, marker


def test_run_close_warns_not_raises_on_broken_export(tmp_path, monkeypatch):
    """A failing viz export must downgrade to a warning, not kill the run."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    import magnelio.io.paraview as pv  # noqa: PLC0415

    def boom(*a, **k):
        raise RuntimeError("synthetic viz failure")

    monkeypatch.setattr(pv, "export_run_visualization", boom)
    with pytest.warns(UserWarning, match="ParaView session export failed"):
        _tem_analysis(tmp_path / "pp").run(
            excited=[("port1", 0)], energy_stop_db=None, total_time_steps=60
        )
