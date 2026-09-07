"""Unit gates for the ParaView session exporter (DD-115).

Covers the pieces that need no solver run: the deterministic material
table (block colouring contract between ``export_vtm`` and the script
generator), the per-solid ``.vtm`` writer, the slice-plane spec derived
from monitor node axes, and the generated ``paraview_open.py`` (must
compile, must embed the config verbatim, must not bake state when the
suite-wide ``MAGNELIO_PVSM_BAKE=0`` pin is active).

The eigenmode session (DD-139) is gated here too: it needs an
eigensolve rather than a time-domain run, which is cheap enough for a
unit test on a coarse air box.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pytest

from magnelio import Material
from magnelio.io.paraview import (
    _FIELD_SCRIPT,
    _INFO_SCRIPT,
    _SCRIPT_HEADER,
    _length_exponent,
    _magnitude_stats,
    _material_table,
    _mirror_signature,
    _mirror_signs,
    _monitor_geometry,
    _pick_vector,
    _prepare_mirroring,
    _sanitize,
    bake_pvsm,
    write_paraview_script,
)


def _config_of(script_path):
    """The config literal embedded in a generated session script."""
    text = script_path.read_text(encoding="utf-8")
    ns: dict = {}
    exec(text[: text.index("def build")], ns)  # noqa: S102 — our own generated file
    return ns["CONFIG"]


def test_sanitize():
    assert _sanitize("E_plane") == "E_plane"
    assert _sanitize("E field / xy") == "E_field_xy"
    assert _sanitize("///") == "unnamed"


def test_material_table_dedupes_and_disambiguates():
    fr4 = Material.from_isotropic("FR4", epsilon=4.3)
    fr4_red = Material.from_isotropic("FR4", epsilon=4.3)
    fr4_red.color = (1.0, 0.0, 0.0)
    table, idx = _material_table([fr4, Material.pec(), fr4, fr4_red])
    # Same name+colour collapses to one entry; same name with a
    # different colour gets its own, disambiguated label.
    assert idx == [0, 1, 0, 2]
    assert [t["name"] for t in table] == ["FR4", "PEC", "FR4 #2"]
    assert all(len(t["rgba"]) == 4 for t in table)


def test_material_table_air_gets_translucent_blue():
    # Auto-coloured air/vacuum must be *visible* in the 3D session (the
    # cavity volume is the device): translucent, blue-dominant — unlike
    # the 2D cross-sections, which draw it as a dashed outline.
    air = Material.from_isotropic("air", epsilon=1.0)
    table, _idx = _material_table([air, Material.pec()])
    r, g, b, a = table[0]["rgba"]
    assert 0.0 < a < 0.5
    assert b > r and b > g
    # An explicit colour and an invisible material both win over the default.
    tinted = Material.from_isotropic("air", epsilon=1.0)
    tinted.color = (1.0, 0.0, 0.0)
    tinted.alpha = 0.5
    hidden = Material.from_isotropic("air", epsilon=1.0)
    hidden.visible = False
    table2, _idx2 = _material_table([tinted, hidden])
    assert table2[0]["rgba"] == [1.0, 0.0, 0.0, 0.5]
    assert table2[1]["rgba"][3] == 0.0


def test_pick_vector_prefers_complete_e_triple():
    assert _pick_vector(["Ex", "Ey", "Ez", "Hx"]) == ("E", ["Ex", "Ey", "Ez"])
    assert _pick_vector(["Hx", "Hy", "Hz"]) == ("H", ["Hx", "Hy", "Hz"])
    assert _pick_vector(["Ex", "Ey"]) is None


def test_monitor_geometry_3d_slices_normal_to_shortest_extent():
    spec = _monitor_geometry([0.0, 1.0, 2.0], [0.0, 0.25, 0.5], [0.0, 2.0, 4.0, 6.0])
    assert spec["slice_axes"] == ["x", "y", "z"]
    assert spec["default_axis"] == "y"
    assert spec["planar_normal"] is None
    assert spec["center"] == [1.0, 0.25, 3.0]
    assert spec["l_ref"] > 0


def test_length_exponent_fits_reference_to_its_length_fraction():
    # The exponent must map the typical magnitude onto 45 % of the full
    # arrow length; a reference already there needs no compression.
    e = _length_exponent(0.21, 1.0)
    assert 0.21**e == pytest.approx(0.45, rel=1e-6)
    assert _length_exponent(0.45, 1.0) == 1.0
    # Degenerate inputs never produce a compressing exponent.
    assert _length_exponent(0.0, 1.0) == 1.0
    assert _length_exponent(2.0, 1.0) == 1.0


def test_magnitude_stats_ignores_dead_cells():
    """A field-free majority must not drag the length reference up.

    Half the cells are exactly zero (PEC interior, quiet volume) and the
    live ones span 20x.  Taking the reference over all cells would put it
    next to the cap and leave the typical arrow invisible — the measured
    defect behind the exponent fit.
    """
    live = np.concatenate([np.full(400, 1.0), np.full(100, 20.0)])
    vals = np.concatenate([np.zeros(500), live])
    cap, exponent = _magnitude_stats([[vals]], 98.0)
    assert cap == pytest.approx(20.0)
    # Reference = p60 of the live cells = 1.0, i.e. cap/20 -> compressed.
    assert exponent == pytest.approx(np.log(0.45) / np.log(1.0 / 20.0), rel=1e-6)
    assert 0.2 <= exponent < 1.0


def test_magnitude_stats_all_zero():
    cap, exponent = _magnitude_stats([[np.zeros(10)]], 98.0)
    assert cap == 0.0 and exponent == 1.0


def test_lattice_dims_keep_one_spacing_for_every_axis():
    from magnelio.io.paraview import _lattice_dims

    ext = [0.0594, 0.0683, 0.800]
    dims = _lattice_dims(ext, 0.005)
    steps = [L / (n - 1) for L, n in zip(ext, dims)]
    # One spacing everywhere: a fixed count per axis would make the
    # spacing directional on an elongated region — the very bias that
    # placing arrows on the computational grid produced.
    assert max(steps) / min(steps) < 1.25
    # A degenerate axis collapses to a single layer, not to two.
    assert _lattice_dims([0.1, 0.0, 0.1], 0.01)[1] == 1


def test_section_and_volume_steps_serve_their_view():
    from magnelio.io.paraview import _section_step, _volume_step

    ext = [0.0594, 0.0683, 0.800]
    s_sec = _section_step(ext)
    # The largest section (the two longest extents) lands on the target.
    assert 0.0683 * 0.800 / s_sec**2 == pytest.approx(2000, rel=1e-6)
    # The volume lattice is coarser: glyphing every point of the section
    # lattice in 3D would bury the field under its own arrows.
    s_vol = _volume_step(ext)
    assert s_vol > s_sec
    # A planar region has no volume lattice.
    assert _volume_step([0.1, 0.0, 0.1]) is None
    # ... but still gets a section spacing from its area.
    assert _section_step([0.1, 0.0, 0.1]) == pytest.approx((0.1 * 0.1 / 2000) ** 0.5)


def test_monitor_geometry_carries_lattice_dims():
    spec = _monitor_geometry([0.0, 1.0, 2.0], [0.0, 0.25, 0.5], [0.0, 2.0, 4.0, 6.0])
    assert all(n >= 2 for n in spec["resample_dims"])
    assert spec["resample_dims_volume"] is not None
    # A planar monitor is glyphed on its own lattice and needs no volume one.
    planar = _monitor_geometry([0.0, 1.0, 2.0], [0.5, 0.6], [0.0, 2.0, 4.0])
    assert planar["resample_dims_volume"] is None
    assert planar["resample_dims"][1] == 1  # the degenerate axis


def test_monitor_geometry_planar():
    spec = _monitor_geometry([0.0, 1.0, 2.0], [0.5, 0.6], [0.0, 2.0, 4.0])
    assert spec["slice_axes"] == []
    assert spec["planar_normal"] == "y"


def test_written_script_compiles_and_embeds_config(tmp_path):
    config = {
        "geometry": "../../geometry.vtm",
        "materials": [{"name": "PEC", "rgba": [0.65, 0.65, 0.65, 1.0]}],
        "monitors": [
            {
                "name": "E_vol",
                "kind": "time",
                "data": "paraview/E_vol.pvd",
                "reader": "pvd",
                "glyph": {
                    "arrays": ["E"],
                    "cap": 123.4,
                    "exponent": 0.5,
                    "length": 1e-3,
                    "threshold": 2.468,
                },
                "center": [0.0, 0.0, 0.0],
                "slice_axes": ["x", "y", "z"],
                "default_axis": "y",
                "planar_normal": None,
                "resample_dims": [40, 30, 20],
                "resample_dims_volume": [20, 15, 10],
                "l_ref": 1e-3,
            }
        ],
    }
    script = write_paraview_script(tmp_path / "paraview_open.py", config)
    text = script.read_text(encoding="utf-8")
    compile(text, str(script), "exec")
    # The per-monitor preparation is one Python filter whose scripts
    # travel inside the session file (DD-262).
    assert "simple.ProgrammableFilter" in text
    assert "FIELD_SCRIPT" in text and "INFO_SCRIPT" in text
    # The config must round-trip verbatim through the embedded literal
    # (executing only the header stops at the paraview import inside
    # build(), which is not reached).
    ns: dict = {}
    exec(text[: text.index("def build")], ns)
    assert ns["CONFIG"] == config


def test_bake_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGNELIO_PVSM_BAKE", "0")
    script = tmp_path / "s.py"
    script.write_text("raise SystemExit(1)\n")
    assert bake_pvsm(script, tmp_path / "s.pvsm") is False


def test_the_baking_paraview_is_named_in_the_script(tmp_path):
    """A state file is bound to the release that wrote it (DD-265).

    Nothing in a ``.pvsm`` says which ParaView it came from, and opening
    it with another one loses whole branches of the pipeline without an
    error a caller would recognise.  The version therefore goes into the
    header of the script that sits beside it.
    """
    from magnelio.io.paraview import _STAMP_NONE, _STAMP_PREFIX, _stamp_version

    script = tmp_path / "paraview_open.py"
    script.write_text(_SCRIPT_HEADER, encoding="utf-8")
    assert _STAMP_PREFIX + _STAMP_NONE in script.read_text(encoding="utf-8")

    # What the session script prints back after SaveState.
    _stamp_version(script, "paraview version 9.9.9")
    header = script.read_text(encoding="utf-8")
    assert _STAMP_PREFIX + "9.9.9" in header
    assert _STAMP_NONE not in header


def test_the_bake_interpreter_can_be_named(tmp_path, monkeypatch):
    """A machine with two ParaViews must not bake for the wrong one."""
    from magnelio.io.paraview import resolve_pvpython

    monkeypatch.delenv("MAGNELIO_PVPYTHON", raising=False)
    assert resolve_pvpython(sys.executable) == sys.executable
    monkeypatch.setenv("MAGNELIO_PVPYTHON", sys.executable)
    assert resolve_pvpython() == sys.executable
    assert resolve_pvpython("no-such-pvpython-anywhere") is None


def test_export_vtm_blocks_names_materials(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    vtk = pytest.importorskip("vtk")

    from magnelio import GeometryModel
    from magnelio.geo import Brick, Cylinder
    from magnelio.io.paraview import export_vtm

    model = GeometryModel()
    model.add(
        Brick(
            origin=(0, 0, 0),
            size=(10e-3, 5e-3, 1e-3),
            material=Material.from_isotropic("FR4", epsilon=4.3),
            name="substrate",
        )
    )
    model.add(
        Cylinder(
            origin=(5e-3, 2.5e-3, 1e-3),
            radius=1e-3,
            height=2e-3,
            axis="z",
            material=Material.pec(),
            name="post",
        )
    )

    out = tmp_path / "geometry.vtm"
    table = export_vtm(out, list(model))
    assert out.exists()
    assert [t["name"] for t in table] == ["FR4", "PEC"]

    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(out))
    reader.Update()
    mb = reader.GetOutput()
    assert mb.GetNumberOfBlocks() == 2
    names = [mb.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) for i in range(2)]
    assert names == ["substrate", "post"]
    for i in range(2):
        blk = mb.GetBlock(i)
        assert blk.GetNumberOfCells() > 0
        arr = blk.GetCellData().GetArray("MaterialIndex")
        assert arr is not None
        assert arr.GetRange() == (float(i), float(i))
    # The curved solid tessellates finer than a box's 12 triangles.
    assert mb.GetBlock(1).GetNumberOfCells() > 12


def test_export_vtm_completes_the_model_across_symmetry_planes(tmp_path):
    """The file holds the whole model, not the simulated half (DD-265).

    The completion used to be ParaView's `Reflect` filter, whose proxy
    was renamed between 6.0 and 6.1 — a state file naming the old one
    lost its entire geometry branch on the newer release.  Doing it here
    keeps the session's pipeline to proxies that do not move.
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    vtk = pytest.importorskip("vtk")

    from magnelio import GeometryModel
    from magnelio.geo import Brick
    from magnelio.io.paraview import export_vtm

    # A quarter block in the first quadrant, mirrored back over both walls.
    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(4e-3, 3e-3, 1e-3), material="pec", name="quarter"))

    half = tmp_path / "half.vtm"
    whole = tmp_path / "whole.vtm"
    export_vtm(half, list(model))
    export_vtm(
        whole,
        list(model),
        mirrors=[("x", 0.0, True, "PEC"), ("y", 0.0, True, "PMC")],
    )

    def bounds(path):
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        return reader.GetOutput().GetBlock(0).GetBounds()

    bh, bw = bounds(half), bounds(whole)
    assert bh[0] == pytest.approx(0.0, abs=1e-12)
    assert bh[2] == pytest.approx(0.0, abs=1e-12)
    # Both walls sit at zero, so the model reaches as far below as above.
    assert bw[0] == pytest.approx(-4e-3)
    assert bw[1] == pytest.approx(4e-3)
    assert bw[2] == pytest.approx(-3e-3)
    assert bw[3] == pytest.approx(3e-3)
    # The out-of-plane extent is untouched.
    assert bw[4:] == pytest.approx(bh[4:])


def _eigen_project(tmp_path, n_modes=4, export=True):
    """A small air-box eigenmode project, written to *tmp_path*.

    With *export* the ParaView session is written as well — on request,
    as the store no longer does it by itself (DD-262).
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    pytest.importorskip("vtk")

    from magnelio import GeometryModel, Mesh, MeshControl
    from magnelio.analysis.eigenmode import AnalysisEigenmode
    from magnelio.geo import Brick

    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(30e-3, 20e-3, 15e-3), material=Material.air()))
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=12e9)
    path = tmp_path / "cavity"
    project = AnalysisEigenmode(
        mesh=mesh,
        n_modes=n_modes,
        verbose=False,
        project=str(path),
        geometry=model,
    ).run()
    if export:
        project.export_paraview_eigenmodes(bake_state=False)
    return project


class TestEigenmodeExport:
    """Eigenmodes reach ParaView without a driven run (DD-139)."""

    def test_writing_eigenmodes_exports_nothing(self, tmp_path):
        # The solver writes the result; the ParaView files are the
        # user's call (DD-262).
        project = _eigen_project(tmp_path, export=False)
        assert (project.path / "eigenmodes.h5").exists()
        assert not (project.path / "paraview_open.py").exists()
        assert not (project.path / "paraview").exists()
        assert not (project.path / "geometry.vtm").exists()

    def test_export_generates_the_session(self, tmp_path):
        project = _eigen_project(tmp_path)
        assert (project.path / "paraview_open.py").exists()
        assert (project.path / "paraview" / "eigenmodes.pvd").exists()
        n = len(project.eigenmodes.frequencies)
        vtrs = sorted((project.path / "paraview" / "eigenmodes").glob("mode_*.vtr"))
        assert len(vtrs) == n

    def test_export_writes_geometry_vtm_when_missing(self, tmp_path):
        project = _eigen_project(tmp_path, export=False)
        assert (project.path / "geometry.brep").exists()
        assert not (project.path / "geometry.vtm").exists()
        project.export_paraview_eigenmodes(bake_state=False)
        assert (project.path / "geometry.vtm").exists()
        assert (project.path / "geometry" / "geometry_0.vtp").exists()
        config = _config_of(project.path / "paraview_open.py")
        assert config["geometry"] == "geometry.vtm"
        assert config["materials"][0]["name"] == "air"

    def test_time_axis_is_the_mode_index_not_the_frequency(self, tmp_path):
        # Degenerate pairs share a frequency exactly; two datasets at one
        # timestep would hide each other.
        project = _eigen_project(tmp_path)
        pvd = (project.path / "paraview" / "eigenmodes.pvd").read_text()
        steps = [int(s) for s in re.findall(r'timestep="(\d+)"', pvd)]
        assert steps == list(range(len(project.eigenmodes.frequencies)))

    def test_fields_are_peak_normalised_with_the_divisor_kept(self, tmp_path):
        import vtk
        from vtk.util import numpy_support as ns

        project = _eigen_project(tmp_path)
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(project.path / "paraview" / "eigenmodes" / "mode_0000.vtr"))
        reader.Update()
        out = reader.GetOutput()

        cd = out.GetCellData()
        assert {cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())} >= {
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "E",
            "H",
            "|E|",
            "|H|",
        }
        for field in ("E", "H"):
            vec = ns.vtk_to_numpy(cd.GetArray(field))
            assert float(np.sqrt((vec**2).sum(axis=1)).max()) == pytest.approx(1.0, rel=1e-12)

        fd = out.GetFieldData()
        stored = {
            fd.GetArrayName(i): float(ns.vtk_to_numpy(fd.GetArray(i))[0])
            for i in range(fd.GetNumberOfArrays())
        }
        assert stored["f_Hz"] == pytest.approx(float(project.eigenmodes.frequencies[0]), rel=1e-12)
        assert stored["mode_index"] == 0.0
        # The eigenvector's own scale is far from 1 — that it is recorded
        # is what makes the normalisation reversible rather than lossy.
        assert stored["E_peak_before_normalisation"] > 0.0
        assert stored["H_peak_before_normalisation"] > 0.0

    def test_cells_match_the_project_grid(self, tmp_path):
        import vtk

        project = _eigen_project(tmp_path)
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(project.path / "paraview" / "eigenmodes" / "mode_0000.vtr"))
        reader.Update()
        grid = project.grid
        assert reader.GetOutput().GetDimensions() == (
            len(grid.x),
            len(grid.y),
            len(grid.z),
        )

    def test_regenerating_is_idempotent_and_returns_paths(self, tmp_path):
        project = _eigen_project(tmp_path)
        before = (project.path / "paraview" / "eigenmodes" / "mode_0000.vtr").read_bytes()
        written = project.export_paraview_eigenmodes(bake_state=False)
        assert written["monitors"] == ["eigenmodes"]
        assert written["script"].exists()
        after = (project.path / "paraview" / "eigenmodes" / "mode_0000.vtr").read_bytes()
        assert after == before

    def test_project_without_eigenmodes_exports_nothing(self, tmp_path):
        from magnelio import GeometryModel, Mesh, MeshControl
        from magnelio.geo import Brick
        from magnelio.io.paraview import export_eigenmode_visualization
        from magnelio.io.project import ProjectStore

        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(10e-3, 10e-3, 10e-3), material=Material.air()))
        mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=6), f_max=10e9)
        store = ProjectStore.create(tmp_path / "empty", mesh, setup={"analysis": "none"})
        assert export_eigenmode_visualization(store.path) == {}


class TestMirrorSigns:
    """A mirrored half must carry the continuation the monitors use.

    The session's field filter builds the mirrored copy in numpy, one
    sign per array and component (DD-262), so no renderer guesswork is
    left to correct: the export resolves every array against
    :func:`~magnelio.post._symmetry.mirror_sign`, the same function the
    monitor plots continue their data with.
    """

    def test_array_names_resolve_to_field_and_component(self):
        assert _mirror_signature("Ex") == ("E", 0)
        assert _mirror_signature("Hz_im") == ("H", 2)
        assert _mirror_signature("Ey_re") == ("E", 1)
        # The vectors carry no single component.
        assert _mirror_signature("E") == ("E", None)
        assert _mirror_signature("H_re") == ("H", None)
        # Magnitudes are even across every mirror, and so is anything
        # that is not a field at all.
        assert _mirror_signature("|E|") is None
        assert _mirror_signature("MaterialIndex") is None

    def test_every_field_array_gets_its_sign(self):
        arrays = ["Ex", "Ey", "Ez", "E", "|E|", "Hx", "H", "E_re", "H_im"]
        mirrors = [["y", 0.0, True, "PEC"]]
        (plane,) = _mirror_signs(arrays, mirrors)
        # Across an electric wall E continues with its normal component
        # even and both tangential ones odd, H the other way round; the
        # magnitude is even and needs no entry.
        assert plane == {
            "Ex": -1.0,
            "Ey": 1.0,
            "Ez": -1.0,
            "E": [-1.0, 1.0, -1.0],
            "Hx": 1.0,
            "H": [1.0, -1.0, 1.0],
            "E_re": [-1.0, 1.0, -1.0],
            "H_im": [1.0, -1.0, 1.0],
        }

    def test_single_components_carry_their_full_continuation(self):
        from magnelio.post._symmetry import mirror_sign

        arrays = [f"{f}{a}" for f in "EH" for a in "xyz"]
        for kind in ("PEC", "PMC"):
            (plane,) = _mirror_signs(arrays, [["y", 0.0, True, kind]])
            for field in ("E", "H"):
                for comp, axis in enumerate("xyz"):
                    assert plane[f"{field}{axis}"] == mirror_sign(field, comp, 1, kind)

    def test_the_two_wall_types_disagree_on_every_entry(self):
        arrays = ["Ex", "Ey", "Ez", "E"]
        pec, pmc = _mirror_signs(arrays, [["x", 0.0, True, "PEC"], ["x", 0.0, True, "PMC"]])
        assert pec == {"Ex": 1.0, "Ey": -1.0, "Ez": -1.0, "E": [1.0, -1.0, -1.0]}
        assert pmc == {"Ex": -1.0, "Ey": 1.0, "Ez": 1.0, "E": [-1.0, 1.0, 1.0]}

    def test_a_monitor_collapsed_onto_a_mirrored_axis_gains_a_layer(self):
        # Mirroring turns its single cell layer into two, and a lattice
        # of one would sample the seam between them.
        spec = {
            "field_arrays": ["Ey"],
            "resample_dims": [1, 40, 40],
            "resample_dims_volume": None,
        }
        _prepare_mirroring([spec], [["x", 0.0, True, "PEC"]])
        assert spec["resample_dims"] == [2, 40, 40]
        # Every other axis keeps its spacing: the arrow count is a
        # target for the displayed picture, which is the mirrored one.
        _prepare_mirroring([spec], [["y", 0.0, True, "PEC"]])
        assert spec["resample_dims"] == [2, 40, 40]

    def test_the_raw_array_listing_does_not_reach_the_session(self):
        spec = {"field_arrays": ["Ex", "E"], "resample_dims": [4, 4, 4]}
        _prepare_mirroring([spec], [["z", 0.0, True, "PEC"]])
        assert "field_arrays" not in spec
        assert spec["mirror_signs"] == [{"Ex": -1.0, "E": [-1.0, -1.0, 1.0]}]

    def test_without_symmetry_nothing_is_signed(self):
        spec = {"field_arrays": ["Ex", "E"], "resample_dims": [4, 4, 4]}
        _prepare_mirroring([spec], [])
        assert spec["mirror_signs"] == []


class TestSymmetryReachesTheSession:
    """The declared planes travel into the generated script (DD-169)."""

    def test_the_wall_type_travels_with_the_plane(self, tmp_path):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        pytest.importorskip("vtk")

        from magnelio import GeometryModel, Mesh, MeshControl
        from magnelio.analysis.eigenmode import AnalysisEigenmode
        from magnelio.boundaries.boundary_conditions import BoundaryConditions
        from magnelio.geo import Brick

        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(30e-3, 20e-3, 15e-3), material=Material.air()))
        model.boundary_conditions = BoundaryConditions(
            xmin="ForceSymmetryPMC", ymin="ForceSymmetryPEC"
        )
        mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=6), f_max=12e9)
        project = AnalysisEigenmode(
            mesh=mesh,
            n_modes=2,
            verbose=False,
            project=str(tmp_path / "half"),
            geometry=model,
        ).run()
        project.export_paraview_eigenmodes(bake_state=False)

        config = _config_of(project.path / "paraview_open.py")
        planes = {p[0]: p[3] for p in config["symmetry"]}
        assert planes == {"x": "PMC", "y": "PEC"}
        # Two planes of opposite type continue the same field with
        # opposite parities: E polar across the magnetic wall (normal
        # odd), axial across the electric one; H the other way round.
        signs = config["monitors"][0]["mirror_signs"]
        assert signs[0]["E"] == [-1.0, 1.0, 1.0]
        assert signs[1]["E"] == [-1.0, 1.0, -1.0]
        assert signs[0]["H"] == [1.0, -1.0, -1.0]
        assert signs[1]["H"] == [1.0, -1.0, 1.0]


# ═════════════════════════════════════════════════════════════════════
# The monitor's Python filter, run outside ParaView (DD-262)
# ═════════════════════════════════════════════════════════════════════


class _FilterStub:
    """What the field script asks of ``self`` inside a ProgrammableFilter."""

    def __init__(self, inp, out):
        self._inp, self._out = inp, out

    def GetInputDataObject(self, _port, _index):  # noqa: N802 — VTK spelling
        return self._inp

    def GetOutputDataObject(self, _port):  # noqa: N802 — VTK spelling
        return self._out


def _frame_grid(nx=4, ny=3, nz=2, dx=1e-3):
    """A monitor frame as the export writes it: cell data on the node grid.

    ``Ex = 1`` everywhere, ``Ey = y`` (the cell centre's coordinate),
    ``Ez = 0``; the vector ``E`` beside the three components.
    """
    import vtk
    from vtk.util import numpy_support as ns

    x, y, z = (np.arange(n + 1, dtype=float) * dx for n in (nx, ny, nz))
    rg = vtk.vtkRectilinearGrid()
    rg.SetDimensions(nx + 1, ny + 1, nz + 1)
    setters = (rg.SetXCoordinates, rg.SetYCoordinates, rg.SetZCoordinates)
    for setter, nodes in zip(setters, (x, y, z)):
        setter(ns.numpy_to_vtk(nodes, deep=True))
    cy = 0.5 * (y[1:] + y[:-1])
    ex = np.ones((nz, ny, nx))
    ey = np.broadcast_to(cy[None, :, None], (nz, ny, nx)).copy()
    ez = np.zeros((nz, ny, nx))
    vec = np.stack([ex, ey, ez], axis=-1).reshape(-1, 3)
    for name, data in (("E", vec), ("Ex", ex.ravel()), ("Ey", ey.ravel()), ("Ez", ez.ravel())):
        arr = ns.numpy_to_vtk(np.ascontiguousarray(data), deep=True)
        arr.SetName(name)
        rg.GetCellData().AddArray(arr)
    return rg


def _run_field_script(cfg, mode, grid=None):
    """Execute the filter script the way ParaView does, shadowed builtins included."""
    import vtk

    grid = grid if grid is not None else _frame_grid()
    out = vtk.vtkPolyData() if mode == "volume" else vtk.vtkImageData()
    scope = {"self": _FilterStub(grid, out), "CFG": cfg, "MODE": mode}
    # ParaView's preamble: the array reductions of numpy_interface take
    # the names of the builtins, which is what a script must survive.
    exec("from vtkmodules.numpy_interface.algorithms import *", scope)  # noqa: S102
    exec(_FIELD_SCRIPT, scope)  # noqa: S102
    return out


def _point_array(dataset, name):
    from vtk.util import numpy_support as ns

    arr = dataset.GetPointData().GetArray(name)
    assert arr is not None, name
    return ns.vtk_to_numpy(arr)


def _cfg(**overrides):
    cfg = {
        "dims": [9, 7, 5],
        "dims_volume": [5, 4, 3],
        "arrays": ["E"],
        "cap": 1.2,
        "exponent": 0.5,
        "threshold": 0.0,
        "symmetry": [],
        "signs": [],
    }
    cfg.update(overrides)
    return cfg


class TestFieldScript:
    """The per-monitor filter: cell frame in, even lattice with arrow lengths out."""

    def test_section_is_the_lattice_with_magnitude_and_length(self):
        pytest.importorskip("vtk")
        cfg = _cfg(cap=0.5)
        out = _run_field_script(cfg, "section")
        assert tuple(out.GetDimensions()) == tuple(cfg["dims"])
        assert out.GetPointData().GetArray("vtkValidPointMask") is None
        vec = _point_array(out, "E")
        mag = _point_array(out, "E_mag")
        length = _point_array(out, "E_len")
        np.testing.assert_allclose(mag, np.sqrt(np.sum(vec * vec, axis=-1)), rtol=1e-12)
        # |E| >= 1 everywhere (Ex = 1) and the cap is 0.5: every arrow
        # saturates at the full length.
        assert np.all(mag >= 1.0)
        np.testing.assert_allclose(length, 1.0)
        # The lattice spans the frame's bounds (VTK pulls the samples a
        # millionth of the span inside, so no point sits on the face).
        assert out.GetBounds() == pytest.approx((0.0, 4e-3, 0.0, 3e-3, 0.0, 2e-3), abs=1e-7)

    def test_length_law_compresses_below_the_cap(self):
        pytest.importorskip("vtk")
        cfg = _cfg(cap=4.0, exponent=0.5)
        out = _run_field_script(cfg, "section")
        mag = _point_array(out, "E_mag")
        length = _point_array(out, "E_len")
        np.testing.assert_allclose(length, np.sqrt(np.minimum(mag, 4.0) / 4.0), rtol=1e-12)

    def test_electric_wall_on_a_grid_line_mirrors_without_a_seam_cell(self):
        pytest.importorskip("vtk")
        # E across an electric wall: normal component even, tangential
        # odd — the sign table the export resolves through mirror_sign.
        cfg = _cfg(
            symmetry=[["x", 0.0, True, "PEC"]],
            signs=[{"E": [1.0, -1.0, -1.0], "Ex": 1.0, "Ey": -1.0, "Ez": -1.0}],
            cap=100.0,
            exponent=1.0,
        )
        out = _run_field_script(cfg, "section")
        # 2n - 1 nodes: the wall's node reflects onto itself.
        assert out.GetBounds()[:2] == pytest.approx((-4e-3, 4e-3), abs=1e-7)
        nx, ny, nz = out.GetDimensions()
        vec = _point_array(out, "E").reshape(nz, ny, nx, 3)
        ex, ey = vec[..., 0], vec[..., 1]
        np.testing.assert_allclose(ex, np.flip(ex, axis=2), atol=1e-12)
        np.testing.assert_allclose(ey, -np.flip(ey, axis=2), atol=1e-12)
        # On the wall itself the odd component vanishes; the even one
        # keeps its value.
        assert np.all(np.abs(ey[:, :, nx // 2]) < 1e-12)
        np.testing.assert_allclose(ex[:, :, nx // 2], 1.0)
        # The single component agrees with the vector's.
        np.testing.assert_allclose(_point_array(out, "Ey").reshape(nz, ny, nx), ey, atol=1e-12)

    def test_magnetic_wall_half_a_cell_out_gets_a_cell_astride_it(self):
        pytest.importorskip("vtk")
        # E across a magnetic wall: normal component odd, tangential even.
        cfg = _cfg(
            symmetry=[["x", -0.5e-3, True, "PMC"]],
            signs=[{"E": [-1.0, 1.0, 1.0], "Ex": -1.0, "Ey": 1.0, "Ez": 1.0}],
            cap=100.0,
            exponent=1.0,
        )
        out = _run_field_script(cfg, "section")
        # 2(n + 1) nodes: the reflected grid stops one cell short of
        # the original, the cell between takes the wall value.
        assert out.GetBounds()[:2] == pytest.approx((-5e-3, 4e-3), abs=1e-7)
        nx, ny, nz = out.GetDimensions()
        vec = _point_array(out, "E").reshape(nz, ny, nx, 3)
        ex, ey = vec[..., 0], vec[..., 1]
        np.testing.assert_allclose(ex, -np.flip(ex, axis=2), atol=1e-12)
        np.testing.assert_allclose(ey, np.flip(ey, axis=2), atol=1e-12)
        assert np.all(np.abs(ex[:, :, nx // 2]) < 1e-12)  # the wall sample

    def test_volume_is_the_thresholded_point_cloud(self):
        pytest.importorskip("vtk")
        cfg = _cfg(threshold=0.0)
        out = _run_field_script(cfg, "volume")
        assert out.GetNumberOfPoints() == int(np.prod(cfg["dims_volume"]))
        assert out.GetNumberOfVerts() == out.GetNumberOfPoints()
        for name in ("E", "E_mag", "E_len", "Ex"):
            assert out.GetPointData().GetArray(name) is not None, name
        assert out.GetPointData().GetArray("vtkValidPointMask") is None
        # Ey grows with y: a threshold between its extremes keeps the
        # upper part of the lattice only.
        ey = _point_array(out, "E")[:, 1]
        cut = float(np.median(np.sqrt(1.0 + ey * ey)))
        kept = _run_field_script(_cfg(threshold=cut), "volume")
        expect = int(np.count_nonzero(np.sqrt(1.0 + ey * ey) >= cut))
        assert 0 < kept.GetNumberOfPoints() == expect < out.GetNumberOfPoints()
        assert _run_field_script(_cfg(threshold=1e9), "volume").GetNumberOfPoints() == 0

    def test_info_script_declares_the_lattice_extent(self):
        # The one line ParaView needs to run the field script once per
        # update instead of four times.
        assert "WHOLE_EXTENT" in _INFO_SCRIPT
        assert 'CFG["dims"]' in _INFO_SCRIPT
