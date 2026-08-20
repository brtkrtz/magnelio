"""A board all the way from fabrication data into a mesh.

The unit gates pin what the readers extract and what the geometry
engine builds.  This one pins the property the whole design rests on:
that 35 µm of copper does not have to be resolved by the grid.

The mesher recognises a perfectly conducting layer thinner than the
cell-size floor and gives it a single grid plane on its substrate side,
carrying the metal's thickness in the sub-cell material fractions of
the neighbouring cells instead of in a layer of cells of its own.  A
board import that produced copper the mesher did not recognise that way
would still mesh — and would demand cells a hundred times smaller than
the physics needs, on every board.  So the recognition is asserted, not
assumed, and the floor it requires is what the documentation tells the
user to set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from magnelio import GeometryModel, Material, MeshControl, open_project
from magnelio.io import import_pcb
from magnelio.mesh.mesher import Mesh

pytest.importorskip("OCC", reason="board import requires pythonocc-core")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
KICAD_BOARD = FIXTURES / "kicad_board"

MM = 1e-3
COPPER_T = 0.035 * MM
# The floor the documentation asks for: comfortably above the metal, so
# the layer is a thin sheet rather than a resolved stack of cells.
MIN_CELL = 0.2 * MM
F_MAX = 5.0e9


# ── a microstrip line, written as a fabrication export ───────────────


def _mm(value: float) -> str:
    return f"{round(value * 1e6):d}"


def _gerber(*body: str) -> str:
    return "\n".join(["%FSLAX46Y46*%", "%MOMM*%", "G01*", "G75*", *body, "M02*"]) + "\n"


def _loop(x0, y0, x1, y1, *, region=False):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    body = [f"X{_mm(corners[0][0])}Y{_mm(corners[0][1])}D02*"]
    body += [f"X{_mm(x)}Y{_mm(y)}D01*" for x, y in corners[1:]]
    return ["G36*", *body, "G37*"] if region else body


def _microstrip(root: Path) -> Path:
    """A 50 Ω line on 1.53 mm FR4 over a ground plane, with two vias."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "line.gbrjob").write_text(
        json.dumps(
            {
                "GeneralSpecs": {"ProjectId": {"Name": "line"}},
                "FilesAttributes": [
                    {"Path": "line-F_Cu.gbr", "FileFunction": "Copper,L1,Top"},
                    {"Path": "line-B_Cu.gbr", "FileFunction": "Copper,L2,Bot"},
                    {"Path": "line-Edge_Cuts.gbr", "FileFunction": "Profile,NP"},
                    {"Path": "line-PTH.drl", "FileFunction": "Plated,1,2,PTH"},
                ],
                "MaterialStackup": [
                    {"Type": "Copper", "Thickness": 0.035, "Name": "F.Cu"},
                    {
                        "Type": "Dielectric",
                        "Thickness": 1.53,
                        "Material": "FR4",
                        "DielectricConstant": "4.5",
                        "Name": "core",
                    },
                    {"Type": "Copper", "Thickness": 0.035, "Name": "B.Cu"},
                ],
            }
        )
    )
    (root / "line-Edge_Cuts.gbr").write_text(
        _gerber("%ADD10C,0.05*%", "D10*", *_loop(0.0, 0.0, 20.0, 10.0))
    )
    (root / "line-F_Cu.gbr").write_text(
        _gerber(
            "%ADD10C,2.9*%",
            "D10*",
            f"X{_mm(1.45)}Y{_mm(5.0)}D02*",
            f"X{_mm(18.55)}Y{_mm(5.0)}D01*",
        )
    )
    (root / "line-B_Cu.gbr").write_text(_gerber(*_loop(0.0, 0.0, 20.0, 10.0, region=True)))
    (root / "line-PTH.drl").write_text(
        "M48\n; #@! TF.FileFunction,Plated,1,2,PTH\nFMAT,2\nMETRIC\nT1C0.400\n%\nG90\nG05\nT1\n"
        "X3.0Y1.0\nX17.0Y9.0\nT0\nM30\n"
    )
    return root


AIR = Material.from_isotropic(name="air", epsilon=1.0)


def _model(board):
    """The board in a box of air, as a microstrip has to be modelled."""
    from magnelio.geo import Brick

    model = GeometryModel(
        boundary_conditions={
            face: "PEC" for face in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        }
    )
    model.add(board)
    # Air above the top copper, not around it: the board occupies its
    # own volume, and the cells beside the traces fall to the model
    # background, which is air as well.
    model.add(
        Brick(
            origin=(0.0, 0.0, COPPER_T),
            size=(20.0 * MM, 10.0 * MM, 3.0 * MM),
            material=AIR,
        )
    )
    return model


def _mesh(model):
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8, min_cell_size=MIN_CELL),
        f_max=F_MAX,
    )


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    root = _microstrip(tmp_path_factory.mktemp("fab"))
    return import_pcb(root)


# ── the thin-sheet coupling ──────────────────────────────────────────


def test_imported_copper_is_recognised_as_a_thin_sheet(board):
    """Otherwise every board would need cells thinner than its copper."""
    from magnelio.mesh._conformal import detect_thin_metallizations

    shapes = list(_model(board))
    sheets = detect_thin_metallizations(shapes, MIN_CELL)
    detected = {spec.shape.name for spec in sheets}

    assert detected == {"F.Cu", "B.Cu"}


def test_a_thin_copper_layer_gets_one_grid_plane_not_two(board):
    """Two planes would resolve the metal and drive the cell size."""
    mesh = _mesh(_model(board))
    inside_copper = [z for z in mesh.grid.z if 0.0 < z < COPPER_T]

    assert inside_copper == []
    assert min(np.diff(mesh.grid.z)) >= MIN_CELL - 1e-12


def test_the_board_fills_the_grid_it_is_meshed_on(board):
    """A model whose materials never land in a cell meshes and is empty."""
    mesh = _mesh(_model(board))
    names = {mesh.material_library[key].name for key in np.unique(mesh.material_id)}

    assert {"air", "FR4"} <= names
    # The substrate faces are grid planes; the outer copper faces are
    # not, which is the thin-sheet treatment showing in the grid.
    assert np.min(np.abs(mesh.grid.z)) == pytest.approx(0.0, abs=1e-12)
    assert np.min(np.abs(mesh.grid.z + 1.53 * MM)) == pytest.approx(0.0, abs=1e-12)


def test_the_substrate_reaches_the_mesh_as_a_dielectric(board):
    mesh = _mesh(_model(board))
    permittivities = {m.name: max(m.epsilon) for m in mesh.material_library.values()}

    assert permittivities.get("FR4") == pytest.approx(4.5)


# ── the model survives a store round trip ────────────────────────────


def test_store_round_trip_keeps_the_layer_names_and_materials(board, tmp_path):
    from magnelio.io.project import ProjectStore

    model = _model(board)
    ProjectStore.create(tmp_path / "proj", mesh=_mesh(model), geometry=model)
    loaded = list(open_project(tmp_path / "proj").geometry)

    assert [solid.name for solid in loaded][:5] == [
        "F.Cu",
        "dielectric_1",
        "B.Cu",
        "via_1",
        "via_2",
    ]
    assert loaded[0].material.is_pec
    assert loaded[1].material.epsilon == pytest.approx((4.5, 4.5, 4.5))


def test_a_material_given_at_import_reaches_the_mesh(tmp_path_factory):
    root = _microstrip(tmp_path_factory.mktemp("fab_ro"))
    laminate = Material(name="RO4350B", epsilon=(3.66,) * 3)
    mesh = _mesh(_model(import_pcb(root, {"dielectric_1": laminate})))
    permittivities = {m.name: max(m.epsilon) for m in mesh.material_library.values()}

    assert permittivities.get("RO4350B") == pytest.approx(3.66)


# ── a set exported by a real layout tool ─────────────────────────────


@pytest.mark.skipif(
    not KICAD_BOARD.exists(), reason="no layout-tool-exported fabrication set available"
)
def test_a_real_fabrication_export_imports():
    board = import_pcb(KICAD_BOARD)
    solids = list(board.members())

    assert solids
    for solid in solids:
        assert solid.name
        low, high = solid.bounding_box()
        assert all(h > lo for lo, h in zip(low, high))
        assert solid.volume() > 0.0
    # A board is centimetres across and millimetres thick; reading the
    # coordinate format wrong is the one error with no other symptom.
    low, high = board.bounding_box()
    assert 1e-3 < high[0] - low[0] < 1.0
    assert 1e-5 < high[2] - low[2] < 1e-2
