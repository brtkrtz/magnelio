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

import numpy as np
import pytest

from magnelio import Material
from magnelio.io.paraview import (
    _length_exponent,
    _magnitude_stats,
    _material_table,
    _monitor_geometry,
    _pick_vector,
    _sanitize,
    bake_pvsm,
    write_paraview_script,
)


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
                "data": "paraview/E_vol.xdmf",
                "reader": "xdmf",
                "glyph": {
                    "arrays": ["E"],
                    "cap": 123.4,
                    "exponent": 0.5,
                    "length": 1e-3,
                },
                "center": [0.0, 0.0, 0.0],
                "slice_axes": ["x", "y", "z"],
                "default_axis": "y",
                "planar_normal": None,
                "l_ref": 1e-3,
            }
        ],
    }
    script = write_paraview_script(tmp_path / "paraview_open.py", config)
    text = script.read_text(encoding="utf-8")
    compile(text, str(script), "exec")
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


def _eigen_project(tmp_path, n_modes=4):
    """A small air-box eigenmode project, written to *tmp_path*."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    pytest.importorskip("vtk")

    from magnelio import GeometryModel, Mesh, MeshControl
    from magnelio.analysis.eigenmode import AnalysisEigenmode
    from magnelio.geo import Brick

    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(30e-3, 20e-3, 15e-3), material=Material.air()))
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=12e9)
    path = tmp_path / "cavity"
    return AnalysisEigenmode(
        mesh=mesh,
        n_modes=n_modes,
        verbose=False,
        project=str(path),
        geometry=model,
    ).run()


class TestEigenmodeExport:
    """Eigenmodes reach ParaView without a driven run (DD-139)."""

    def test_writing_eigenmodes_generates_the_session(self, tmp_path):
        project = _eigen_project(tmp_path)
        assert (project.path / "paraview_open.py").exists()
        assert (project.path / "paraview" / "eigenmodes.pvd").exists()
        n = len(project.eigenmodes.frequencies)
        vtrs = sorted((project.path / "paraview" / "eigenmodes").glob("mode_*.vtr"))
        assert len(vtrs) == n

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
