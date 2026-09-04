"""Interactive 3D view of a geometry model and its mesh.

The view is built on PyVista (VTK).  It shows the CAD solids coloured
by material, optionally the FIT grid, and the declared features
(thin wires, ports, lumped elements, symmetry planes, domain box).
A single axis-aligned cutting plane, driven by controls in the widget
toolbar, opens the model and exposes the grid cells on the cut.

Where the view ends up depends on where it is called from:

* in a Jupyter notebook it is an interactive widget (rendered in the
  browser by default, see ``mode``),
* in a Sphinx-Gallery build it is a screenshot,
* in a plain script it opens an interactive window.

The cutting plane is deliberately axis-aligned and slider-driven —
the way cutting planes work in the EM suites users come from — rather
than a free plane grabbed in 3D: a FIT grid carries information only on
its own planes, and a 3D handle competes with the camera for the mouse.

Every dataset that reaches an actor is ``vtkPolyData``: the in-browser
renderer (vtk.js, through trame) serialises polydata, image data and
unstructured grids only — a rectilinear grid in the scene leaves the
widget blank.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from magnelio.mesh.mesher import Mesh

__all__ = ["show_geometry"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Display colours shared with the 2D cross-section plots.
_WIRE_COLOR = "#c8963c"
_PORT_COLOR = "#d62728"
_ELEMENT_COLOR = "#2ca02c"
_SYMMETRY_COLORS = {"PEC": "#4c72b0", "PMC": "#55a868"}
_DOMAIN_COLOR = "#606060"
_AIR_CELL_COLOR = (0.96, 0.96, 0.96)

# Jupyter backends accepted by ``mode``.  ``"client"`` renders in the
# browser (vtk.js): no OpenGL needed in the kernel, and immune to the
# kernel-thread trap described in ``_configure_pyvista``.  ``"server"``
# renders in the kernel and streams images; ``"trame"`` offers both
# with a toggle.  ``"static"`` embeds a screenshot; ``"none"`` builds
# the scene without showing it (scripts, tests).
_MODES = ("client", "server", "trame", "static", "none")

# Tessellation: fraction of the model's bounding-box diagonal used as
# the linear deflection, and the angular deflection [rad].  The export
# defaults (2e-3, 0.5 rad) leave visible facets on cylinders; the view
# is looked at, not measured, and can afford finer triangles.
_LINEAR_DEFLECTION = 5e-4
_ANGULAR_DEFLECTION = 0.15

# Object groups the toolbar can hide, in menu order.
_GROUPS = (
    ("solids", "Solids"),
    ("cut cells", "Grid on cut"),
    ("ports", "Ports"),
    ("elements", "Lumped elements"),
    ("wires", "Wires"),
    ("labels", "Labels"),
    ("symmetry", "Symmetry planes"),
    ("domain", "Domain box"),
)


def _in_notebook() -> bool:
    """True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython  # noqa: PLC0415
    except ImportError:  # pragma: no cover - IPython is a pyvista dependency
        return False
    ip = get_ipython()
    return ip is not None and "IPKernelApp" in getattr(ip, "config", {})


def _loop_is_running() -> bool:
    """True inside a running asyncio loop -- every cell of a Jupyter kernel."""
    import asyncio  # noqa: PLC0415

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


# Tasks started by _show_when_server_ready, held until they finish so
# the loop does not garbage-collect a pending one.
_PENDING_VIEWS: set = set()


def _launch_viewer_server():
    """Schedule the trame server's start on the running loop; return the server.

    Names the transport.  With the trame Jupyter server extension
    installed, the kernel inherits ``TRAME_BACKEND=jupyter``, and a start
    without an explicit backend takes it: a comm-based transport that
    binds no TCP port, so the iframe is built for ``localhost:0`` and
    stays blank.  PyVista's own path names the backend for the same
    reason; the widget talks over the websocket the viewer disabled the
    extension for (see :func:`_configure_pyvista`).  A second call finds
    the server pending or running and changes nothing.
    """
    import pyvista as pv  # noqa: PLC0415
    from pyvista.trame.jupyter import launch_server  # noqa: PLC0415

    backend = "jupyter" if pv.global_theme.trame.jupyter_extension_enabled else "aiohttp"
    return launch_server(wslink_backend=backend)


def _show_when_server_ready(pl, mode: str, jupyter_kwargs: dict) -> None:
    """Start the trame server on the running loop and show the view once it is up.

    PyVista's own first-call path (``elegantly_launch``) re-enters the
    kernel's event loop through ``nest_asyncio2`` and blocks until the
    server is ready.  Under ipykernel 7 every cell runs in its own
    ``contextvars`` context, and a nested loop that picks up the *next*
    queued cell -- exactly what "Run all" produces -- trips
    ``RuntimeError: cannot enter context ... is already entered`` and
    aborts the run.  Manual execution only worked because the queue was
    empty while the server came up.

    Here the server is started as a task on the loop that is already
    running (``launch_server`` does that) and an empty container widget
    takes the view's place in the cell at once.  The moment the server
    reports ready the view becomes the container's child.  That has to
    be a widget *state* change: once the cell has returned, the
    frontend no longer routes ``display_data`` to it (an
    ``ipywidgets.Output`` filled from a task stays blank), while a
    widget's state travels over the comm channel and renders whenever
    it arrives.  No loop is nested, so the queue is never read out of
    turn; ``nest_asyncio2`` is not needed.  Later calls find the server
    running and take the direct path (DD-251).

    The view therefore lands when the loop is next free.  Under *Run
    All* that is after the queued cells, and starting the server at
    import does not change it: measured in the browser, the loop gets
    no usable time between two queued cells, so a server scheduled at
    import is still pending when the next cell plots.
    """
    import asyncio  # noqa: PLC0415

    from IPython.display import display  # noqa: PLC0415
    from ipywidgets import HTML, VBox, Widget  # noqa: PLC0415

    box = VBox()
    display(box)
    server = _launch_viewer_server()

    async def fill() -> None:
        try:
            await server.ready
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Suppress rendering")
                view = pl.show(
                    jupyter_backend=mode, jupyter_kwargs=jupyter_kwargs, return_viewer=True
                )
            if not isinstance(view, Widget):
                # An IFrame (server-proxy setups) is plain HTML.
                view = HTML(view._repr_html_())
            box.children = (view,)
        except Exception as exc:  # pragma: no cover - reported into the cell
            box.children = (HTML(f"<pre>3D view could not start: {exc!r}</pre>"),)

    task = asyncio.get_running_loop().create_task(fill())
    _PENDING_VIEWS.add(task)
    task.add_done_callback(_PENDING_VIEWS.discard)


def _configure_pyvista() -> None:
    """One-time process settings the viewer relies on.

    * The Viskores (VTK-m) filter overrides in accelerated VTK builds
      try the device first and fall back after tens of seconds on the
      rectilinear grids used here; they are switched off.
    * trame's Jupyter transport goes through its own websocket, not
      through ``trame-jupyter-extension``: JupyterLab ≥ 4.5 executes
      comm messages in kernel subshells (ipykernel ≥ 7 runs those in a
      separate thread), and VTK rendering from a thread that never
      owned the OpenGL context yields black frames or aborts the
      kernel.  An explicit ``PYVISTA_TRAME_JUPYTER_MODE`` set by the
      user is respected.
    * trame's VTK serialiser calls deprecated VTK 9.6 accessors; the
      warnings carry nothing a user can act on.
    """
    import pyvista as pv  # noqa: PLC0415

    try:
        from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmFilterOverrides  # noqa: PLC0415

        vtkmFilterOverrides.SetEnabled(False)
    except ImportError:
        pass
    if "PYVISTA_TRAME_JUPYTER_MODE" not in os.environ:
        try:
            pv.global_theme.trame.jupyter_extension_enabled = False
        except AttributeError:  # pragma: no cover - older pyvista
            pass
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"trame_vtk(\..*)?")


# ---------------------------------------------------------------------------
# Scene data
# ---------------------------------------------------------------------------


@dataclass
class _Body:
    """One tessellated solid in display units."""

    name: str
    polydata: Any  # pv.PolyData
    color: tuple[float, float, float]
    opacity: float
    actor: Any = None


@dataclass
class _Overlay:
    """A feature drawn over the solids: tube, window, sheet or label.

    ``anchor`` labels are shown whole or not at all, by which side of
    the cut their anchor point lies on; everything else is clipped
    like the solids (open clip, no cap).
    """

    name: str
    group: str
    polydata: Any
    actor: Any = None
    anchor: tuple[float, float, float] | None = None
    clip: bool = True


@dataclass
class _CutState:
    axis: str | None = None  # None = no cut
    position: float = 0.0  # display units
    flip: bool = False


@dataclass
class _Scene:
    """Everything the widget controls need after the plotter is shown."""

    plotter: Any
    bodies: list[_Body]
    bounds: tuple[float, float, float, float, float, float]  # display units
    grid: Any = None  # the FIT grid dataset (display units), or None
    grid_actor: Any = None  # the cut sheet
    domain_actor: Any = None
    overlays: list[_Overlay] = field(default_factory=list)
    hidden_groups: set[str] = field(default_factory=set)
    cut: _CutState = field(default_factory=_CutState)
    initial_cut: _CutState = field(default_factory=_CutState)
    history: list[_CutState] = field(default_factory=list)
    unit: str = "mm"

    def groups_present(self) -> list[str]:
        present = {"solids"} if self.bodies else set()
        if self.grid_actor is not None:
            present.add("cut cells")
        if self.domain_actor is not None:
            present.add("domain")
        present.update(o.group for o in self.overlays)
        return [key for key, _ in _GROUPS if key in present]


def _shape_bodies(shapes, *, unit_scale: float, quality: float) -> list[_Body]:
    """Tessellate the volume shapes into display-unit polydata."""
    import pyvista as pv  # noqa: PLC0415
    from OCC.Core.Bnd import Bnd_Box  # noqa: PLC0415
    from OCC.Core.BRepBndLib import brepbndlib  # noqa: PLC0415

    from magnelio.io.paraview import _polydata, _tessellate_shape  # noqa: PLC0415
    from magnelio.post._colors import shape_color  # noqa: PLC0415

    geo_scale = 1.0
    if shapes and all(hasattr(s, "_analytic_bbox") for s in shapes):
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(shapes)

    bodies: list[_Body] = []
    for i, shape in enumerate(shapes):
        mat = getattr(shape, "material", None)
        if mat is not None and not getattr(mat, "visible", True):
            continue
        occ = shape._occ_shape(geo_scale)
        # An empty solid (e.g. a boolean that removed everything) has no
        # extent; tessellating it aborts inside OCC, so it is skipped
        # here, with a warning — silently missing bodies mislead.
        box = Bnd_Box()
        brepbndlib.Add(occ, box)
        if occ.IsNull() or box.IsVoid():
            warnings.warn(
                f"shape {getattr(shape, 'name', None) or i!r} has no volume and is not drawn",
                stacklevel=3,
            )
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        deflection = max(_LINEAR_DEFLECTION * diag / max(quality, 1e-3), 1e-12)
        tess = _tessellate_shape(occ, deflection, angular_deflection=_ANGULAR_DEFLECTION)
        if tess is None:
            continue
        points, tris = tess
        pd = pv.wrap(_polydata(points / geo_scale * unit_scale, tris))
        # OCC triangulates face by face with private nodes, so the
        # shell is open along every face edge as far as VTK is
        # concerned.  Merging the coincident nodes makes it watertight,
        # which the capped clip relies on; smooth shading still splits
        # sharp edges by feature angle, so nothing blurs.
        pd = pd.clean(tolerance=1e-7, absolute=False)
        rgba = shape_color(shape)
        is_air = rgba[3] < 1e-6
        color = (0.88, 0.91, 0.94) if is_air else tuple(float(c) for c in rgba[:3])
        opacity = 0.15 if is_air else float(rgba[3])
        bodies.append(
            _Body(
                name=str(getattr(shape, "name", None) or f"shape_{i}"),
                polydata=pd,
                color=color,  # type: ignore[arg-type]
                opacity=opacity,
            )
        )
    return bodies


def _grid_dataset(mesh: Mesh, *, unit_scale: float):
    """The FIT grid as a rectilinear dataset with per-cell material colours.

    Kept as a ``RectilinearGrid`` for slicing; nothing derived from it
    reaches an actor without being converted to polydata first.
    """
    import pyvista as pv  # noqa: PLC0415

    from magnelio.post._colors import material_color  # noqa: PLC0415

    g = mesh.grid
    rg = pv.RectilinearGrid(
        np.asarray(g.x, dtype=float) * unit_scale,
        np.asarray(g.y, dtype=float) * unit_scale,
        np.asarray(g.z, dtype=float) * unit_scale,
    )
    material_id = np.asarray(mesh.material_id)
    ids = np.unique(material_id)
    palette = np.full((int(ids.max()) + 1 if ids.size else 1, 3), 255, dtype=np.uint8)
    for mid in ids:
        mat = mesh.material_library.get(int(mid))
        rgba = material_color(mat) if mat is not None else (0.65, 0.65, 0.65, 1.0)
        rgb = _AIR_CELL_COLOR if rgba[3] < 1e-6 else rgba[:3]
        palette[int(mid)] = np.round(np.asarray(rgb, dtype=float) * 255).astype(np.uint8)
    flat = material_id.ravel(order="F")
    rg.cell_data["material_id"] = flat.astype(np.int32)
    rg.cell_data["color"] = palette[flat]
    rg.cell_data.active_scalars_name = "color"
    return rg


def _bounds_of(bodies: list[_Body], grid) -> tuple[float, ...]:
    if grid is not None:
        return tuple(float(b) for b in grid.bounds)
    if not bodies:
        return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    lo = np.min([b.polydata.bounds[0::2] for b in bodies], axis=0)
    hi = np.max([b.polydata.bounds[1::2] for b in bodies], axis=0)
    out = []
    for a, b in zip(lo, hi):
        out += [float(a), float(b)]
    return tuple(out)


# ---------------------------------------------------------------------------
# Cutting
# ---------------------------------------------------------------------------


def _clip_body(pd, cut: _CutState, *, closed: bool = True):
    """Clip a dataset with the cut plane.

    ``closed`` caps the opening (needs a watertight surface); otherwise
    the clip stays open — right for tubes, windows and sheets.  Returns
    ``None`` when nothing of the dataset remains.
    """
    if cut.axis is None:
        return pd
    axis = _AXIS_INDEX[cut.axis]
    lo, hi = pd.bounds[2 * axis], pd.bounds[2 * axis + 1]
    keep_positive = cut.flip
    if cut.position <= lo:
        return pd if keep_positive else None
    if cut.position >= hi:
        return None if keep_positive else pd
    # Both filters keep the side the normal points to (``clip`` with
    # ``invert=False``); the normal therefore points into the kept half.
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0 if keep_positive else -1.0
    origin = [0.0, 0.0, 0.0]
    origin[axis] = cut.position
    if closed:
        try:
            clipped = pd.clip_closed_surface(normal=normal, origin=origin)
            if clipped.n_cells:
                return clipped
        except Exception:
            pass
    clipped = pd.clip(normal=normal, origin=origin, invert=False)
    return clipped if clipped.n_cells else None


def _on_kept_side(point, cut: _CutState) -> bool:
    if cut.axis is None:
        return True
    value = float(point[_AXIS_INDEX[cut.axis]])
    return value >= cut.position if cut.flip else value <= cut.position


def _grid_slab(grid, cut: _CutState):
    """The grid cells on the cut plane, as a polydata sheet of cell faces.

    The sheet carries the material of the cell layer just *behind* the
    plane on the kept side — the cells the cut exposes.  It is offset by
    a hair into the removed half so that it does not fight the solids'
    cut caps for the same depth; being translucent, it leaves the caps
    visible underneath.
    """
    if grid is None or cut.axis is None:
        return None
    import pyvista as pv  # noqa: PLC0415

    axis = _AXIS_INDEX[cut.axis]
    nodes = [np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z)]
    n = nodes[axis]
    if n.size < 2 or not (n[0] <= cut.position <= n[-1]):
        return None
    # ``flip`` keeps the +axis half: the exposed layer then starts at
    # the plane; otherwise it ends there.
    if cut.flip:
        k = min(int(np.searchsorted(n, cut.position, side="left")), n.size - 2)
    else:
        k = max(int(np.searchsorted(n, cut.position, side="left")) - 1, 0)
    offset = 1e-3 * (n[k + 1] - n[k]) * (-1.0 if cut.flip else 1.0)
    u_axis, v_axis = (a for a in range(3) if a != axis)
    u, v = nodes[u_axis], nodes[v_axis]
    nu, nv = u.size - 1, v.size - 1
    # Points with u fastest, then v — the grid's own ordering with the
    # cut axis removed — so cell (iu, iv) is quad iu + nu*iv, the same
    # order as the layer's cell ids below.
    uu, vv = np.meshgrid(u, v, indexing="xy")  # shape (nv+1, nu+1)
    points = np.empty(((nu + 1) * (nv + 1), 3), dtype=float)
    points[:, u_axis] = uu.ravel()
    points[:, v_axis] = vv.ravel()
    points[:, axis] = cut.position + offset
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="xy")
    p0 = (iu + (nu + 1) * iv).ravel()
    faces = np.column_stack([np.full(p0.size, 4), p0, p0 + 1, p0 + nu + 2, p0 + nu + 1]).ravel()
    sheet = pv.PolyData(points, faces=faces)
    dims = tuple(len(nodes[i]) - 1 for i in range(3))
    cell_ids = np.arange(int(np.prod(dims))).reshape(dims, order="F")
    idx = [slice(None)] * 3
    idx[axis] = slice(k, k + 1)
    selected = cell_ids[tuple(idx)].ravel(order="F")
    for name in ("material_id", "color"):
        sheet.cell_data[name] = np.asarray(grid.cell_data[name])[selected]
    sheet.cell_data.active_scalars_name = "color"
    return sheet


def _set_input(actor, dataset) -> None:
    """Swap an actor's dataset (single-colour actors only)."""
    actor.mapper.SetInputData(dataset)


def _add_sheet(pl, sheet):
    """(Re)create the cut-sheet actor; replaces one of the same name."""
    actor = pl.add_mesh(
        sheet,
        scalars="color",
        rgb=True,
        show_edges=True,
        # Dark enough to read on the dielectric tints through the
        # sheet's translucency; light grey vanished on blue.
        edge_color="#606060",
        line_width=1,
        opacity=0.7,
        lighting=False,
        name="grid_cut",
        reset_camera=False,
        render=False,
    )
    # Name the colour array explicitly: the browser renderer receives
    # the mapper's array selection, not VTK's "active scalars" notion.
    mapper = actor.mapper
    mapper.SetScalarModeToUseCellFieldData()
    mapper.SelectColorArray("color")
    mapper.SetColorModeToDirectScalars()
    return actor


def _apply_cut(scene: _Scene) -> None:
    """Refresh every actor for the scene's cut state and hidden groups."""
    shown = {key for key, _ in _GROUPS} - scene.hidden_groups
    for body in scene.bodies:
        clipped = _clip_body(body.polydata, scene.cut, closed=True)
        visible = "solids" in shown and clipped is not None
        body.actor.SetVisibility(visible)
        if clipped is not None:
            _set_input(body.actor, clipped)
    for ov in scene.overlays:
        if ov.anchor is not None:
            visible = _on_kept_side(ov.anchor, scene.cut)
            dataset = ov.polydata
        elif ov.clip:
            dataset = _clip_body(ov.polydata, scene.cut, closed=False)
            visible = dataset is not None
        else:
            dataset, visible = ov.polydata, True
        ov.actor.SetVisibility(visible and ov.group in shown)
        if dataset is not None:
            _set_input(ov.actor, dataset)
    if scene.domain_actor is not None:
        scene.domain_actor.SetVisibility("domain" in shown)
    if scene.grid is not None:
        slab = _grid_slab(scene.grid, scene.cut)
        if slab is not None:
            # Rebuilt rather than swapped: the browser renderer keys a
            # mapper's dataset by the mapper, so a swapped-in sheet
            # arrived under the old identity and was dropped there.  A
            # fresh actor carries fresh identities for actor, mapper
            # and data.
            scene.grid_actor = _add_sheet(scene.plotter, slab)
            scene.grid_actor.SetVisibility("cut cells" in shown)
        elif scene.grid_actor is not None:
            scene.grid_actor.SetVisibility(False)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------


def _diag(bounds) -> float:
    return max(math.dist(bounds[0::2], bounds[1::2]), 1e-9)


def _add_overlay(pl, scene: _Scene, overlay: _Overlay, **kwargs) -> None:
    overlay.actor = pl.add_mesh(overlay.polydata, name=overlay.name, **kwargs)
    scene.overlays.append(overlay)


def _label_frame(pl) -> np.ndarray:
    """Rotation whose columns are the camera's right, up and towards-viewer axes.

    Labels are built in the text's own x/y plane and mapped through this
    frame, so they face the initial camera the right way round — a
    fixed plane normal reads mirrored from half of the viewpoints.
    """
    cam = pl.camera
    towards = np.asarray(cam.position, dtype=float) - np.asarray(cam.focal_point, dtype=float)
    towards /= max(np.linalg.norm(towards), 1e-12)
    up = np.asarray(cam.up, dtype=float)
    right = np.cross(up, towards)
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(towards, right)
    return np.column_stack([right, up, towards])


def _plane_frame(pl, axis: int) -> np.ndarray:
    """Like :func:`_label_frame`, but with the text lying in the plane
    normal to ``axis`` — for names written onto a port window.

    The plane normal is signed towards the camera so the text is not
    mirrored, and the text's up direction follows the camera's up.
    """
    camera = _label_frame(pl)
    towards = camera[:, 2]
    normal = np.zeros(3)
    normal[axis] = 1.0 if towards[axis] >= 0.0 else -1.0
    up = camera[:, 1] - np.dot(camera[:, 1], normal) * normal
    if np.linalg.norm(up) < 1e-6:  # camera up along the normal: use the right axis
        up = camera[:, 0] - np.dot(camera[:, 0], normal) * normal
    up /= max(np.linalg.norm(up), 1e-12)
    right = np.cross(up, normal)
    return np.column_stack([right, up, normal])


def _add_label(pl, scene: _Scene, text: str, *, center, height, color, frame=None) -> None:
    """A name as flat 3D text — polydata, so every renderer draws it.

    ``frame`` (columns: right, up, normal) places the text; default is
    a billboard facing the initial camera.
    """
    import pyvista as pv  # noqa: PLC0415

    try:
        mesh = pv.Text3D(text, depth=height * 0.02, height=height)
    except Exception:
        return
    if mesh.n_cells == 0:
        return
    matrix = np.eye(4)
    matrix[:3, :3] = _label_frame(pl) if frame is None else frame
    matrix[:3, 3] = np.asarray(center, dtype=float)
    mesh = mesh.transform(matrix, inplace=False)
    _add_overlay(
        pl,
        scene,
        _Overlay(f"label_{len(scene.overlays)}", "labels", mesh, anchor=tuple(center)),
        color=color,
        lighting=False,
    )


def _add_tube(pl, scene: _Scene, name, group, p0, p1, *, color, radius) -> None:
    import pyvista as pv  # noqa: PLC0415

    tube = pv.Line(p0, p1).tube(radius=radius, n_sides=12)
    _add_overlay(pl, scene, _Overlay(name, group, tube), color=color, smooth_shading=True)


def _add_wires(pl, scene: _Scene, wires, *, unit_scale, radius, geo_scale) -> None:
    import pyvista as pv  # noqa: PLC0415

    from magnelio.geo._occ_backend import sample_wire  # noqa: PLC0415
    from magnelio.post.plot_geometry import _wire_step  # noqa: PLC0415

    for i, wire in enumerate(wires):
        try:
            pts = np.asarray(sample_wire(wire._occ_shape(geo_scale), _wire_step(wire), geo_scale))
        except Exception:
            continue
        if len(pts) < 2:
            continue
        path = pv.lines_from_points(pts * unit_scale)
        r = max(float(getattr(wire, "radius", 0.0)) * unit_scale, radius)
        _add_overlay(
            pl,
            scene,
            _Overlay(f"wire_{i}", "wires", path.tube(radius=r, n_sides=12)),
            color=_WIRE_COLOR,
            smooth_shading=True,
        )


def _face_port_window(port, bounds, unit_scale):
    """Face axis, face position and the (u, v) extents of a port window."""
    from magnelio.mesh.mesher import _normalize_port_face  # noqa: PLC0415

    face = _normalize_port_face(port.plane)
    p_axis = face.normal_axis
    lo = list(bounds[0::2])
    hi = list(bounds[1::2])
    face_pos = hi[p_axis] if face.is_max else lo[p_axis]
    extent = {a: [lo[a], hi[a]] for a in range(3) if a != p_axis}
    window = getattr(port, "corners", None)
    if window is not None:
        p, q = window
        for axis in extent:
            a = float("-inf") if p[axis] is None else float(p[axis]) * unit_scale
            b = float("inf") if q[axis] is None else float(q[axis]) * unit_scale
            extent[axis] = [max(extent[axis][0], min(a, b)), min(extent[axis][1], max(a, b))]
    return p_axis, face_pos, extent


def _add_face_port(pl, scene: _Scene, port, *, bounds, unit_scale, index, name, labels) -> None:
    import pyvista as pv  # noqa: PLC0415

    try:
        p_axis, face_pos, extent = _face_port_window(port, bounds, unit_scale)
    except Exception:
        return
    (u, (u0, u1)), (v, (v0, v1)) = sorted(extent.items())
    if not (u0 < u1 and v0 < v1):
        return
    corners = []
    for cu, cv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
        pt = [0.0, 0.0, 0.0]
        pt[p_axis] = face_pos
        pt[u] = cu
        pt[v] = cv
        corners.append(pt)
    quad = pv.PolyData(np.asarray(corners), faces=[4, 0, 1, 2, 3])
    _add_overlay(
        pl,
        scene,
        _Overlay(f"port_{index}", "ports", quad),
        color=_PORT_COLOR,
        opacity=0.25,
        lighting=False,
    )
    _add_overlay(
        pl,
        scene,
        _Overlay(f"port_{index}_edge", "ports", quad.extract_feature_edges()),
        color=_PORT_COLOR,
        line_width=3,
    )
    if labels:
        center = [0.0, 0.0, 0.0]
        center[p_axis] = face_pos
        center[u] = 0.5 * (u0 + u1)
        center[v] = 0.5 * (v0 + v1)
        height = min(0.06 * _diag(bounds), 0.3 * min(u1 - u0, v1 - v0))
        _add_label(
            pl,
            scene,
            name,
            center=center,
            height=height,
            color=_PORT_COLOR,
            frame=_plane_frame(pl, p_axis),
        )


def _add_symmetry_planes(pl, scene: _Scene, boundary_conditions, *, bounds) -> None:
    """Translucent sheets on the domain faces declared as symmetry planes."""
    import pyvista as pv  # noqa: PLC0415

    symmetry = getattr(boundary_conditions, "symmetry", None)
    if not symmetry:
        return
    for face in symmetry:
        kind = str(getattr(boundary_conditions, face, ""))
        axis = _AXIS_INDEX.get(face[0])
        if axis is None:
            continue
        pos = bounds[2 * axis + 1] if face.endswith("max") else bounds[2 * axis]
        b = list(bounds)
        b[2 * axis] = b[2 * axis + 1] = pos
        _add_overlay(
            pl,
            scene,
            _Overlay(f"symmetry_{face}", "symmetry", pv.Box(b)),
            color=_SYMMETRY_COLORS.get(kind, "#888888"),
            opacity=0.2,
            lighting=False,
        )


def _add_overlays(
    pl,
    scene: _Scene,
    geometry,
    *,
    unit_scale,
    show_wires: bool,
    show_ports: bool,
    labels: bool,
) -> None:
    import pyvista as pv  # noqa: PLC0415

    from magnelio.geo.wire import ThinWire  # noqa: PLC0415

    bounds = scene.bounds
    diag = _diag(bounds)
    radius = diag * 2.5e-3
    label_height = 0.035 * diag
    shapes = list(geometry)
    wires = [s for s in shapes if isinstance(s, ThinWire)]
    if show_wires and wires:
        geo_scale = 1.0
        if all(hasattr(s, "_analytic_bbox") for s in shapes):
            from magnelio.geo._scaling import model_scale  # noqa: PLC0415

            geo_scale = model_scale(shapes)
        _add_wires(pl, scene, wires, unit_scale=unit_scale, radius=radius, geo_scale=geo_scale)

    def line_feature(index, group, name, start, end, color):
        p0 = np.asarray(start, dtype=float) * unit_scale
        p1 = np.asarray(end, dtype=float) * unit_scale
        actor_name = f"{'port' if group == 'ports' else 'element'}_{index}"
        _add_tube(pl, scene, actor_name, group, p0, p1, color=color, radius=radius)
        if labels:
            # Beside the tube, offset along the camera's right axis.
            center = 0.5 * (p0 + p1) + _label_frame(pl)[:, 0] * (3.0 * radius)
            _add_label(pl, scene, name, center=center, height=label_height, color=color)

    if show_ports:
        for i, port in enumerate(getattr(geometry, "ports", ())):
            name = str(getattr(port, "name", None) or f"port{i + 1}")
            if hasattr(port, "start") and hasattr(port, "end"):
                line_feature(i, "ports", name, port.start, port.end, _PORT_COLOR)
            elif hasattr(port, "plane"):
                _add_face_port(
                    pl,
                    scene,
                    port,
                    bounds=bounds,
                    unit_scale=unit_scale,
                    index=i,
                    name=name,
                    labels=labels,
                )
        for i, element in enumerate(getattr(geometry, "elements", ())):
            name = str(getattr(element, "name", None) or f"element{i + 1}")
            line_feature(i, "elements", name, element.start, element.end, _ELEMENT_COLOR)

    _add_symmetry_planes(pl, scene, getattr(geometry, "boundary_conditions", None), bounds=bounds)
    scene.domain_actor = pl.add_mesh(
        pv.Box(bounds).outline(), color=_DOMAIN_COLOR, line_width=1, name="domain"
    )


# ---------------------------------------------------------------------------
# Widget controls (trame)
# ---------------------------------------------------------------------------


def _attach_controls(scene: _Scene, server) -> Any:
    """Register state handlers and return the toolbar builder."""
    from trame.widgets import vuetify3 as vuetify  # noqa: PLC0415

    key = f"mio3d_{id(scene)}"
    k_axis, k_pos, k_flip = f"{key}_axis", f"{key}_pos", f"{key}_flip"
    k_min, k_max, k_step = f"{key}_min", f"{key}_max", f"{key}_step"
    k_reset, k_undo, k_show = f"{key}_reset", f"{key}_undo", f"{key}_show"
    state, ctrl = server.state, server.controller
    groups = scene.groups_present()
    titles = dict(_GROUPS)

    def slider_range(axis: str | None) -> tuple[float, float, float]:
        if axis is None:
            return 0.0, 1.0, 0.01
        a = _AXIS_INDEX[axis]
        lo, hi = scene.bounds[2 * a], scene.bounds[2 * a + 1]
        span = max(hi - lo, 1e-12)
        return lo, hi, span / 400.0

    lo, hi, step = slider_range(scene.cut.axis)
    state[k_axis] = scene.cut.axis or "off"
    state[k_pos] = scene.cut.position
    state[k_flip] = scene.cut.flip
    state[k_min], state[k_max], state[k_step] = lo, hi, step
    state[k_show] = [g for g in groups if g not in scene.hidden_groups]
    state[f"{key}_groups"] = [{"title": titles[g], "value": g} for g in groups]

    def refresh() -> None:
        _apply_cut(scene)
        from pyvista.trame.ui import get_viewer  # noqa: PLC0415

        # PyVista announces every toggle of ``suppress_rendering`` the
        # client-side view makes around a push; nothing to act on.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Suppress rendering")
            try:
                get_viewer(scene.plotter).update()
            except Exception:  # pragma: no cover - viewer not shown yet
                pass

    def push_state(new: _CutState, *, record: bool = True) -> None:
        if record and new != scene.cut:
            scene.history.append(_CutState(scene.cut.axis, scene.cut.position, scene.cut.flip))
            del scene.history[:-20]
        scene.cut = new

    @state.change(k_axis)
    def _on_axis(**kwargs):
        axis = kwargs[k_axis]
        axis = None if axis in (None, "off") else axis
        if axis == scene.cut.axis:
            return
        lo, hi, step = slider_range(axis)
        pos = scene.cut.position
        if axis is not None and not (lo <= pos <= hi):
            pos = 0.5 * (lo + hi)
        push_state(_CutState(axis, pos, scene.cut.flip))
        with state:
            state[k_min], state[k_max], state[k_step] = lo, hi, step
            state[k_pos] = pos
        refresh()

    @state.change(k_pos)
    def _on_pos(**kwargs):
        pos = float(kwargs[k_pos])
        if pos == scene.cut.position:
            return
        push_state(_CutState(scene.cut.axis, pos, scene.cut.flip))
        refresh()

    @state.change(k_flip)
    def _on_flip(**kwargs):
        flip = bool(kwargs[k_flip])
        if flip == scene.cut.flip:
            return
        push_state(_CutState(scene.cut.axis, scene.cut.position, flip))
        refresh()

    @state.change(k_show)
    def _on_show(**kwargs):
        shown = set(kwargs[k_show] or ())
        hidden = {g for g in groups if g not in shown}
        if hidden == scene.hidden_groups:
            return
        scene.hidden_groups = hidden
        refresh()

    def set_cut(new: _CutState) -> None:
        push_state(new, record=False)
        lo, hi, step = slider_range(new.axis)
        with state:
            state[k_axis] = new.axis or "off"
            state[k_min], state[k_max], state[k_step] = lo, hi, step
            state[k_pos] = new.position
            state[k_flip] = new.flip
        refresh()

    @ctrl.set(k_reset)
    def _reset():
        scene.history.append(_CutState(scene.cut.axis, scene.cut.position, scene.cut.flip))
        init = scene.initial_cut
        set_cut(_CutState(init.axis, init.position, init.flip))

    @ctrl.set(k_undo)
    def _undo():
        if scene.history:
            set_cut(scene.history.pop())

    def menu_items() -> None:
        vuetify.VSelect(
            v_model=(k_axis, state[k_axis]),
            items=("items", ["off", "x", "y", "z"]),
            label="Cut",
            density="compact",
            hide_details=True,
            variant="plain",
            style="width: 90px; margin-left: 8px;",
        )
        vuetify.VSlider(
            v_model=(k_pos, state[k_pos]),
            min=(k_min, lo),
            max=(k_max, hi),
            step=(k_step, step),
            thumb_label=True,
            hide_details=True,
            density="compact",
            style="width: 220px; margin-left: 8px;",
            disabled=(f"{k_axis} === 'off'",),
        )
        vuetify.VSwitch(
            v_model=(k_flip, state[k_flip]),
            label="Flip",
            density="compact",
            hide_details=True,
            style="margin-left: 8px;",
            disabled=(f"{k_axis} === 'off'",),
        )
        with vuetify.VBtn(icon=True, size="small", variant="text", click=ctrl[k_undo]):
            vuetify.VIcon("mdi-undo")
            vuetify.VTooltip("Undo cut change", activator="parent", location="bottom")
        with vuetify.VBtn(icon=True, size="small", variant="text", click=ctrl[k_reset]):
            vuetify.VIcon("mdi-backup-restore")
            vuetify.VTooltip("Reset cut", activator="parent", location="bottom")
        vuetify.VSelect(
            v_model=(k_show, state[k_show]),
            items=(f"{key}_groups", state[f"{key}_groups"]),
            label="Show",
            multiple=True,
            density="compact",
            hide_details=True,
            variant="plain",
            style="width: 150px; margin-left: 8px;",
        )

    return menu_items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_scene(
    geometry,
    *,
    mesh,
    cut,
    flip,
    show_ports,
    show_wires,
    show_grid,
    show_labels,
    size,
    render_edges,
    edge_color,
    quality,
    scale_mm,
    camera,
    off_screen,
) -> _Scene:
    import pyvista as pv  # noqa: PLC0415

    from magnelio.geo.wire import ThinWire  # noqa: PLC0415

    unit_scale = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"
    shapes = [s for s in geometry if not isinstance(s, ThinWire)]
    bodies = _shape_bodies(shapes, unit_scale=unit_scale, quality=quality)
    grid = _grid_dataset(mesh, unit_scale=unit_scale) if mesh is not None else None
    bounds = _bounds_of(bodies, grid)

    kwargs: dict[str, Any] = {"off_screen": off_screen}
    if size is not None:
        kwargs["window_size"] = tuple(int(v) for v in size)
    pl = pv.Plotter(**kwargs)
    pl.set_background("white")

    cut_state = _CutState()
    if cut is not None:
        axis, position = cut
        if axis not in _AXIS_INDEX:
            raise ValueError(f"cut normal must be 'x', 'y', or 'z'; got {axis!r}")
        cut_state = _CutState(axis, float(position) * unit_scale, bool(flip))

    for body in bodies:
        body.actor = pl.add_mesh(
            body.polydata,
            color=body.color,
            opacity=body.opacity,
            smooth_shading=True,
            show_edges=render_edges,
            edge_color=edge_color,
            specular=0.3,
            specular_power=15,
            name=body.name,
        )

    scene = _Scene(
        plotter=pl,
        bodies=bodies,
        bounds=bounds,  # type: ignore[arg-type]
        grid=grid,
        cut=cut_state,
        initial_cut=_CutState(cut_state.axis, cut_state.position, cut_state.flip),
        unit=unit,
    )

    if grid is not None:
        # The grid shows on the cutting plane only.  A wireframe of the
        # domain faces (a "cage" around the model) was tried and
        # rejected: its perspective foreshortening in front of the cut
        # confused more than it informed.
        if not show_grid:
            scene.hidden_groups.add("cut cells")
        # The sheet actor is created (and re-created) by ``_apply_cut``;
        # without a cut, a hidden placeholder keeps the group listed.
        seed = _grid_slab(grid, _CutState("z", bounds[4]))
        if seed is not None:
            scene.grid_actor = _add_sheet(pl, seed)
            scene.grid_actor.SetVisibility(False)

    # The camera is fixed before the overlays so that labels can face it.
    pl.enable_parallel_projection()
    if camera is not None:
        pl.camera_position = camera
    pl.reset_camera()

    _add_overlays(
        pl,
        scene,
        geometry,
        unit_scale=unit_scale,
        show_wires=show_wires,
        show_ports=show_ports,
        labels=show_labels,
    )

    _apply_cut(scene)
    pl.add_axes()
    pl.enable_anti_aliasing("ssaa") if off_screen else pl.enable_anti_aliasing("msaa")
    pl.reset_camera()
    return scene


def show_geometry(
    geometry,
    *,
    mesh: Mesh | None = None,
    cut: tuple[str, float] | None = None,
    flip: bool = False,
    show_ports: bool = True,
    show_wires: bool = True,
    show_grid: bool = True,
    show_labels: bool = True,
    mode: str | None = None,
    size: tuple[int, int] | None = None,
    render_edges: bool = False,
    edge_color: str = "#202020",
    quality: float = 1.0,
    scale_mm: bool = True,
    camera: Any = "iso",
):
    """Interactive 3D view of a :class:`~magnelio.geo.GeometryModel`.

    Solids are coloured by material (air and vacuum bodies are drawn as
    faint translucent shells), thin wires, ports, lumped elements and
    symmetry planes are overlaid, and the domain box is outlined.  With
    a ``mesh`` a cutting plane exposes the grid cells — each coloured by
    the material the mesher assigned — on the cut.

    The cutting plane is axis-aligned.  In the notebook widget it is
    driven from the toolbar (normal axis, position slider, flip side,
    undo, reset), which also hides or shows object groups; ``cut``
    sets the plane's initial state, and is the only way to place it
    for a screenshot.  The cut applies to the features as well: a port
    or element in the removed half disappears with it.

    Parameters
    ----------
    geometry : GeometryModel or iterable of shapes
        The geometry to display.  A :class:`~magnelio.geo.GeometryModel`
        contributes its ports, lumped elements and boundary declaration.
    mesh : Mesh, optional
        Show this mesh's grid with the geometry.
    cut : (str, float), optional
        Initial cutting plane as ``(normal, position)`` with the normal
        ``'x'``, ``'y'`` or ``'z'`` and the position in metres, e.g.
        ``("y", 0.0)``.  ``None`` (default) starts uncut.
    flip : bool, default False
        Which half the cut removes: by default the side the normal
        points to; ``True`` removes the other side.
    show_ports, show_wires : bool, default True
        Draw ports and lumped elements, and thin wires.
    show_grid : bool, default True
        With ``mesh``: draw the grid cells on the cutting plane.
    show_labels : bool, default True
        Write the names of ports and lumped elements next to them.
    mode : str, optional
        Where to render in a notebook: ``"client"`` (default) renders
        in the browser and needs no OpenGL in the kernel; ``"server"``
        renders in the kernel and streams images; ``"trame"`` offers
        both with a toggle; ``"static"`` embeds a screenshot;
        ``"none"`` builds the scene without showing it and returns the
        plotter.  Outside a notebook the value is ignored: a script
        opens an interactive window, a documentation build takes a
        screenshot.
    size : (int, int), optional
        Widget or window size in pixels.  Default: full cell width.
    render_edges : bool, default False
        Draw the tessellation edges on every solid.
    edge_color : str, default "#202020"
        Colour of those edges.
    quality : float, default 1.0
        Tessellation fineness; values above 1 give finer triangles.
    scale_mm : bool, default True
        Display in millimetres (``False``: metres).
    camera : str or sequence, default "iso"
        Initial camera: a PyVista preset (``"iso"``, ``"xy"``, ``"xz"``,
        ``"yz"``) or an explicit ``[position, focal_point, view_up]``.

    Returns
    -------
    pyvista.Plotter or None
        The plotter when ``mode="none"``; otherwise the view is displayed
        as a side effect and ``None`` is returned.

    Notes
    -----
    The widget needs the ``trame`` stack (``pip install magnelio[jupyter]``
    or the conda-forge packages ``trame``, ``trame-vtk``,
    ``trame-vuetify``).  Without it the view falls back to a static
    image with a warning.
    """
    import pyvista as pv  # noqa: PLC0415

    _configure_pyvista()
    if mode is not None and mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")

    notebook = _in_notebook()
    gallery = bool(getattr(pv, "BUILDING_GALLERY", False))
    off_screen = mode == "none" or gallery or bool(getattr(pv, "OFF_SCREEN", False))
    if notebook and mode is None:
        mode = "client"

    scene = _build_scene(
        geometry,
        mesh=mesh,
        cut=cut,
        flip=flip,
        show_ports=show_ports,
        show_wires=show_wires,
        show_grid=show_grid,
        show_labels=show_labels,
        size=size,
        render_edges=render_edges,
        edge_color=edge_color,
        quality=quality,
        scale_mm=scale_mm,
        camera=camera,
        off_screen=off_screen or (notebook and mode not in (None, "none")),
    )
    pl = scene.plotter

    if mode == "none":
        return pl

    if notebook and mode in ("client", "server", "trame"):
        server = None
        try:
            from trame.app import get_server  # noqa: PLC0415

            server = get_server(pv.global_theme.trame.jupyter_server_name, client_type="vue3")
            menu_items = _attach_controls(scene, server)
            jupyter_kwargs = {"add_menu_items": menu_items}
        except ImportError:
            jupyter_kwargs = {}
        if server is not None and not server.running and _loop_is_running():
            _show_when_server_ready(pl, mode, jupyter_kwargs)
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Suppress rendering")
            pl.show(jupyter_backend=mode, jupyter_kwargs=jupyter_kwargs)
        return None

    if notebook:  # static
        pl.show(jupyter_backend="static")
        return None

    # Script or documentation build: interactive window, or a screenshot
    # collected by the gallery scraper.
    pl.show()
    return None
