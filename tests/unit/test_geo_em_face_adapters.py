"""Owned topology adapters for existing rectangular EM consumers (WP6.11)."""

import json

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl, geo, monitors, ports
from magnelio._fields.field_arrays import FieldArrays
from magnelio.geo._topology_store import from_recipe, to_recipe
from magnelio.io.project import (
    ProjectStore,
    _declarative_port_from_dict,
    _declarative_port_to_dict,
    open_project,
)
from magnelio.mesh.grid import GridLines


def _box(scale=1):
    return geo.Brick(size=(2 * scale, 3 * scale, 4 * scale), material="air").tag_face(
        "end", normal="z"
    )


@pytest.mark.parametrize("scale", [1e-9, 1e-3, 1])
def test_rectangular_named_face_matches_explicit_declarations(scale):
    box = _box(scale)
    model = geo.GeometryModel(background=Material.pec()).add(box)
    face = box.face("end")
    selected_port = ports.PortWaveguide.from_face(face, model=model, name="feed")
    explicit_port = ports.PortWaveguide(
        name="feed", plane="zmax", corners=((0, 0, 4 * scale), (2 * scale, 3 * scale, 4 * scale))
    )
    assert selected_port == explicit_port
    assert _declarative_port_from_dict(_declarative_port_to_dict(selected_port)) == explicit_port

    selected_monitor = monitors.MonitorFieldFrequency.from_face(
        face, freqs=[1e9], fields=["Ex"], name="fields"
    )
    explicit_monitor = monitors.MonitorFieldFrequency(
        corners=explicit_port.corners, freqs=[1e9], fields=["Ex"], name="fields"
    )
    assert selected_monitor.corners == explicit_monitor.corners
    np.testing.assert_array_equal(selected_monitor.freqs, explicit_monitor.freqs)

    restored = from_recipe(json.loads(json.dumps(to_recipe(box))))
    restored_model = geo.GeometryModel(background=Material.pec()).add(restored)
    assert (
        ports.PortWaveguide.from_face(restored.face("end"), model=restored_model, name="feed")
        == explicit_port
    )


def test_moved_owner_requires_new_selection_and_model_membership():
    original = _box(1e-3)
    placed = geo.Translation((0, 0, 5e-3)) @ original
    model = geo.GeometryModel().add(placed)
    with pytest.raises(ValueError, match="owner must already be added"):
        ports.PortWaveguide.from_face(original.face("end"), model=model, name="stale")
    selected = ports.PortWaveguide.from_face(placed.face("end"), model=model, name="moved")
    assert selected.corners[0][2] == pytest.approx(9e-3)
    assert selected.plane == "zmax"


def test_interior_face_is_eligible_for_field_recording():
    internal = geo.Brick(origin=(1, 1, 1), size=(2, 2, 2), material="air")
    internal = internal.tag_face("slice", normal="z")
    monitor = monitors.MonitorFieldFrequency.from_face(
        internal.face("slice"), freqs=[2e9], name="interior"
    )
    assert monitor.corners == ((1, 1, 3), (3, 3, 3))
    assert monitor.fields == ["E"]


def test_adapter_rejects_ambiguous_curved_oblique_holes_and_nonrectangular_faces():
    box = _box()
    with pytest.raises(TypeError, match="singular FaceRef"):
        monitors.MonitorFieldFrequency.from_face(box.faces(), freqs=[1e9])
    with pytest.raises(geo.AmbiguousTopologyError):
        box.face(near=(0, 0, 0))
    cases = [
        (geo.Cylinder(radius=1, height=2).face(surface_type="cylinder"), "curved"),
        (geo.Cylinder(radius=1, height=2).face(normal="z"), "not an exact rectangle"),
        (geo.Cylinder(radius=1, inner_radius=0.2, height=2).face(normal="z"), "holes"),
        (box.rotated("x", 30).face(near=(1, 1.5, 4), surface_type="plane"), "oblique"),
    ]
    for face, message in cases:
        with pytest.raises(ValueError, match=message):
            monitors.MonitorFieldFrequency.from_face(face, freqs=[1e9])


def test_port_rejects_interior_and_displaced_boundary_planes():
    outer = geo.Brick(size=(4, 4, 4), material="air")
    inner = geo.Translation((0, 0, -1)) @ _box()
    model = geo.GeometryModel().add((outer, inner))
    with pytest.raises(ValueError, match="interior"):
        ports.PortWaveguide.from_face(inner.face("end"), model=model, name="inside")
    for bc in ({"zmax": "CPML"}, {"zmax": "PMC"}, {"zmax": "ForceSymmetryPEC"}):
        model = geo.GeometryModel(boundary_conditions=bc).add(inner)
        with pytest.raises(ValueError, match="PEC domain face"):
            ports.PortWaveguide.from_face(inner.face("end"), model=model, name="displaced")


def test_port_face_and_explicit_port_produce_identical_mesh_and_modal_setup():
    box = _box(1e-3)
    selected = geo.GeometryModel(background=Material.pec()).add(box)
    selected.add_port(ports.PortWaveguide.from_face(box.face("end"), model=selected, name="feed"))
    explicit = geo.GeometryModel(background=Material.pec()).add(box)
    explicit.add_port(
        ports.PortWaveguide(name="feed", plane="zmax", corners=((0, 0, 4e-3), (2e-3, 3e-3, 4e-3)))
    )
    control = MeshControl(min_nodes_per_wavelength=8)
    a = Mesh.from_geometry(selected, control, f_max=100e9)
    b = Mesh.from_geometry(explicit, control, f_max=100e9)
    for axis in ("x", "y", "z"):
        np.testing.assert_array_equal(getattr(a.grid, axis), getattr(b.grid, axis))
    assert a.ports == b.ports
    selected_mode = (
        AnalysisScatteringTD(mesh=a, f_max=100e9, verbose=False).solve_ports()["feed"].modes[0]
    )
    explicit_mode = (
        AnalysisScatteringTD(mesh=b, f_max=100e9, verbose=False).solve_ports()["feed"].modes[0]
    )
    assert selected_mode.mode_type == explicit_mode.mode_type
    assert selected_mode.f_cutoff == explicit_mode.f_cutoff
    assert selected_mode.z_modal(80e9) == explicit_mode.z_modal(80e9)


def test_selected_and_explicit_observation_planes_record_identical_fields():
    box = _box(1e-3)
    face = box.face("end")
    selected = monitors.MonitorFieldFrequency.from_face(
        face, freqs=[1e9], fields=["Ex"], name="selected"
    )
    explicit = monitors.MonitorFieldFrequency(
        corners=((0, 0, 4e-3), (2e-3, 3e-3, 4e-3)), freqs=[1e9], fields=["Ex"], name="explicit"
    )
    grid = GridLines(
        x=np.linspace(0, 2e-3, 5),
        y=np.linspace(0, 3e-3, 6),
        z=np.linspace(0, 4e-3, 7),
    )
    mesh = Mesh.from_grid(grid)
    fields = FieldArrays.zeros(grid.Nx, grid.Ny, grid.Nz)
    fields.Ex[:] = np.arange(fields.Ex.size).reshape(fields.Ex.shape)
    for monitor in (selected, explicit):
        monitor.attach(mesh)
        monitor.record(fields, 0, 0.0, 1e-12)
    a = selected.spectrum_raw.cell_centred(squeeze=True)["Ex"]
    b = explicit.spectrum_raw.cell_centred(squeeze=True)["Ex"]
    np.testing.assert_array_equal(a, b)


def test_project_store_preserves_selected_face_and_resolved_port(tmp_path):
    box = _box(1e-3)
    model = geo.GeometryModel(background=Material.pec()).add(box)
    port = ports.PortWaveguide.from_face(box.face("end"), model=model, name="feed")
    model.add_port(port)
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=100e9)
    ProjectStore.create(tmp_path / "project", mesh, geometry=model)
    loaded = open_project(tmp_path / "project")
    assert loaded.mesh.ports == (port,)
    (owner,) = loaded.geometry
    restored_model = geo.GeometryModel(background=Material.pec()).add(owner)
    assert (
        ports.PortWaveguide.from_face(owner.face("end"), model=restored_model, name="feed") == port
    )
