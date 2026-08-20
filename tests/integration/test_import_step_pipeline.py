"""A STEP file all the way through the pipeline (DD-178).

The unit gates pin what the reader extracts from a file; this one pins
that what comes out is ordinary geometry.  Three properties:

* **Equivalence** — a box imported from STEP meshes exactly like the
  same box built with the CSG API.  Anything the import got wrong about
  units, placement or orientation shows up as a different grid or a
  different material fill.
* **Store round-trip** — a model built from imported solids saves and
  reloads with its names, materials and display colours intact.
* **It runs** — the imported geometry carries a waveguide port and a
  short time-domain run through to an S-parameter.

A file exported by a real CAD system is exercised on top, when one is
available under ``tests/fixtures``; the rest is written by the kernel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, GeometryModel, Material, MeshControl, open_project
from magnelio.geo import Brick
from magnelio.io import import_step
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortWaveguide

pytest.importorskip("OCC", reason="CAD import requires pythonocc-core")

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 25.0e9
AIR = Material.from_isotropic(name="air", epsilon=1.0)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _guide_step(path):
    """A waveguide-shaped air box, written in millimetres as CAD does."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
    from OCC.Core.TDataStd import TDataStd_Name
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

    doc = TDocStd_Document("XCAF")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    box = BRepPrimAPI_MakeBox(
        gp_Pnt(-A / 2 * 1e3, -B / 2 * 1e3, -LZ / 2 * 1e3), A * 1e3, B * 1e3, LZ * 1e3
    ).Shape()
    label = shape_tool.AddShape(box, False)
    TDataStd_Name.Set(label, "guide")
    color_tool.SetColor(label, Quantity_Color(0.1, 0.4, 0.8, Quantity_TOC_RGB), XCAFDoc_ColorSurf)
    Interface_Static.SetCVal("write.step.unit", "MM")
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def _model(shape):
    model = GeometryModel(
        boundary_conditions={f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax")}
        | {"zmin": "PEC", "zmax": "PEC"}
    )
    model.add(shape)
    return model


def _mesh(model):
    return Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=F_MAX)


def test_imported_box_meshes_like_the_modelled_one(tmp_path):
    imported = import_step(_guide_step(tmp_path / "guide.step"), {"guide": AIR})
    built = Brick(origin=(-A / 2, -B / 2, -LZ / 2), size=(A, B, LZ), material=AIR)

    from_file = _mesh(_model(imported))
    from_api = _mesh(_model(built))

    for axis in ("x", "y", "z"):
        np.testing.assert_allclose(
            getattr(from_file.grid, axis), getattr(from_api.grid, axis), atol=1e-12
        )
    np.testing.assert_array_equal(from_file.material_id, from_api.material_id)


def test_store_round_trip_keeps_names_materials_and_colours(tmp_path):
    imported = import_step(_guide_step(tmp_path / "guide.step"), {"guide": AIR})
    model = _model(imported)
    mesh = _mesh(model)

    from magnelio.io.project import ProjectStore

    ProjectStore.create(tmp_path / "proj", mesh=mesh, geometry=model)
    loaded = list(open_project(tmp_path / "proj").geometry)

    assert [s.name for s in loaded] == ["guide"]
    assert loaded[0].material.name == "air"
    assert loaded[0].color == pytest.approx((0.1, 0.4, 0.8), abs=1e-6)
    original = list(model)[0]
    np.testing.assert_allclose(loaded[0].bounding_box(), original.bounding_box(), atol=1e-12)


def test_imported_geometry_carries_a_run(tmp_path):
    imported = import_step(_guide_step(tmp_path / "guide.step"), {"guide": AIR})
    analysis = AnalysisScatteringTD(
        mesh=_mesh(_model(imported)),
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
    )
    f = np.linspace(16e9, 24e9, 9)  # above the TE10 cutoff of 15 GHz
    result = analysis.run(f_axis=f, total_time_steps=1500)
    s21 = np.abs(result.S("port2", "port1"))
    # A short, lossless piece of waveguide passes what it is fed.
    assert np.all(s21 > 0.8)


@pytest.mark.skipif(
    not (FIXTURES / "freecad_part.step").exists(),
    reason="no CAD-exported fixture available",
)
def test_a_real_cad_export_imports():
    parts = list(import_step(FIXTURES / "freecad_part.step").members())
    assert parts
    for part in parts:
        assert part.name  # a real exporter names its products
        low, high = part.bounding_box()
        assert all(h > lo for h, lo in zip(high, low))
        assert part.volume() > 0.0
        # Machined parts are centimetres to decimetres.  Ignoring the
        # file's millimetre declaration would report metres here, which
        # is the one import error that produces no other symptom.
        assert all(h - lo < 10.0 for lo, h in zip(low, high))
