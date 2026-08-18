"""2D and 3D visualisation of OCC geometry.

Main entry points:
    plot_cross_section — render an axis-aligned slice through a GeometryModel
    show_geometry      — interactive 3D view in Jupyter via pythonocc
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from magnelio.geo import GeometryModel
    from magnelio.mesh.mesher import Mesh


# Patch pythreejs Euler trait to also accept lowercase order strings (e.g.
# 'xyz').  Three.js uses lowercase internally; the browser widget protocol
# sends lowercase back to Python, but pythreejs only lists uppercase in its
# Enum validator.  The Enum sub-trait is a shared descriptor on Object3D, so
# one patch fixes all subclasses for the lifetime of the Python process.
def _patch_euler_orders() -> None:
    try:
        from pythreejs.core.Object3D_autogen import Object3D as _O3D  # noqa: PLC0415

        _enum = _O3D.class_traits()["rotation"]._traits[3]  # pyright: ignore[reportAttributeAccessIssue]
        _missing = [v.lower() for v in _enum.values if v.lower() not in _enum.values]
        if _missing:
            _enum.values = list(_enum.values) + _missing
    except Exception:
        pass


_patch_euler_orders()

# (u, v) axis labels for each cutting-plane normal
_AXIS_LABELS: dict[str, tuple[str, str]] = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Colours for the non-solid features.  Deliberately outside the material
# palette: these are not materials, and a reader must not have to decide
# whether a red line is a lossy dielectric or a port.
_WIRE_COLOR = "#c8963c"  # thin-wire conductor
_PORT_COLOR = "#d62728"  # ports (excitation / measurement)
_ELEMENT_COLOR = "#2ca02c"  # passive lumped elements

# A thin wire is a sub-cell model: its radius is far below a cell, so
# drawing it to scale would make it invisible.  It is drawn as a line of
# fixed width instead, and a cut counts as "through the wire" when it
# passes within this multiple of the radius.
_WIRE_HIT_RADII = 1.0


def _segment_in_plane(p0, p1, n_axis: int, position: float, tol: float):
    """Where a 3D segment meets a cutting plane.

    Returns ``("line", p0, p1)`` when the segment lies *in* the plane
    (both endpoints within *tol*), ``("point", p)`` when it crosses,
    or ``None`` when it misses.  The distinction is the whole point: a
    wire running along the cut is a line on the picture, the same wire
    piercing the cut is a dot, and drawing either as the other is
    misleading rather than merely imprecise.
    """
    d0 = float(p0[n_axis]) - position
    d1 = float(p1[n_axis]) - position
    if abs(d0) <= tol and abs(d1) <= tol:
        return ("line", p0, p1)
    if d0 * d1 < 0.0:  # strict: a shared endpoint is handled by the branch above
        t = d0 / (d0 - d1)
        return ("point", [p0[k] + t * (p1[k] - p0[k]) for k in range(3)])
    return None


def _uv(point, u_axis: int, v_axis: int, scale: float, flip: bool):
    u, v = float(point[u_axis]) * scale, float(point[v_axis]) * scale
    return (v, u) if flip else (u, v)


def _wire_step(wire) -> float:
    """Sampling step for a wire's curve, in metres.

    Fine enough that a curved wire reads as a curve, coarse enough not
    to sample a straight run into thousands of points: a fraction of the
    curve's own extent, floored by the radius so a tight helix still
    resolves.
    """
    (lo, hi) = wire.bounding_box()
    diag = max((hi[k] - lo[k]) for k in range(3))
    return max(min(diag / 64.0, 1e-3), wire.radius)


def _edge_tol(start, end) -> float:
    """Numerical in-plane tolerance for a straight two-point feature."""
    span = max(abs(float(end[k]) - float(start[k])) for k in range(3))
    return 1e-9 * (1.0 + span)


def _draw_path(ax, points, *, n_axis, position, tol, u_axis, v_axis, scale, flip, color, label):
    """Draw one polyline's trace on the cut; return True if anything showed.

    Consecutive in-plane segments are merged into a single line so a
    sampled curve does not turn into hundreds of separate artists.
    """
    drawn = False
    run: list = []

    def flush():
        nonlocal drawn, run
        if len(run) >= 2:
            xs, ys = zip(*run)
            ax.plot(xs, ys, color=color, linewidth=2.0, solid_capstyle="round", zorder=3.0)
            drawn = True
        run = []

    for p0, p1 in zip(points[:-1], points[1:]):
        hit = _segment_in_plane(p0, p1, n_axis, position, tol)
        if hit is None:
            flush()
            continue
        if hit[0] == "line":
            a = _uv(hit[1], u_axis, v_axis, scale, flip)
            b = _uv(hit[2], u_axis, v_axis, scale, flip)
            if not run:
                run.append(a)
            run.append(b)
        else:
            flush()
            x, y = _uv(hit[1], u_axis, v_axis, scale, flip)
            # A conductor piercing the cut: ring rather than filled dot,
            # so a field plot underneath still reads through it.
            ax.plot(
                [x],
                [y],
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.6,
                linestyle="none",
                # Set explicitly: an unset colour would draw from the
                # axes' property cycle and shift every later plot.
                color=color,
                zorder=3.0,
            )
            drawn = True
    flush()

    if drawn and label:
        import matplotlib.patheffects as patheffects  # noqa: PLC0415

        # Anchor on the path's midpoint: it is inside the drawn extent
        # for a line and on the feature itself for a ring, where an
        # end-anchored label would float off into the background.
        mid = points[len(points) // 2]
        anchor = _uv(mid, u_axis, v_axis, scale, flip)
        ax.annotate(
            label,
            anchor,
            textcoords="offset points",
            xytext=(6, 4),
            color=color,
            fontsize=7,
            zorder=3.1,
            # Same under-stroke as the dashed outlines: these labels sit
            # on the field colour map in a field plot, where a thin
            # coloured glyph on a saturated patch is unreadable.
            path_effects=[
                patheffects.Stroke(linewidth=2.0, foreground="white", alpha=0.85),
                patheffects.Normal(),
            ],
        )
    return drawn


def _draw_face_port(ax, port, bbox, *, n_axis, position, u_axis, v_axis, scale, flip, color):
    """Draw a bbox-face port as the edge of the domain it occupies."""
    from magnelio.mesh.mesher import _normalize_port_face  # noqa: PLC0415

    try:
        face = _normalize_port_face(port.plane)
    except Exception:
        return False
    p_axis = face.normal_axis
    lo, hi = bbox
    face_pos = hi[p_axis] if face.is_max else lo[p_axis]

    if p_axis == n_axis:
        # The port plane is parallel to the cut; it only shows when the
        # cut lies on it, and then it covers the whole section.  Drawing
        # that as a band would swamp the picture, so it is left to the
        # label.
        return False

    # The port face meets the cutting plane along a line: fixed at the
    # face position on its own axis, spanning the *port window* on the
    # other — the whole face only when the port declares no sub-window.
    span_axis = ({0, 1, 2} - {n_axis, p_axis}).pop()
    span_lo, span_hi = lo[span_axis], hi[span_axis]
    window = getattr(port, "corners", None)
    if window is not None:
        # ``corners`` are two opposite 3D world-coordinate points; the
        # component along the face normal carries no information here,
        # and a tangential component may be None (up to the domain
        # boundary on that side).
        p, q = window
        extent = {}
        for axis in sorted({0, 1, 2} - {p_axis}):
            a = float("-inf") if p[axis] is None else float(p[axis])
            b = float("inf") if q[axis] is None else float(q[axis])
            extent[axis] = (min(a, b), max(a, b))
        # A cut that misses the window entirely sees plain wall there,
        # not a port — drawing the line anyway put every port of a face
        # on top of every other one, spanning the full domain edge.
        cut_lo, cut_hi = extent[n_axis]
        if not (cut_lo <= position <= cut_hi):
            return False
        win_lo, win_hi = extent[span_axis]
        span_lo, span_hi = max(span_lo, win_lo), min(span_hi, win_hi)
        if not span_lo < span_hi:
            return False
    ends = []
    for t in (span_lo, span_hi):
        pt = [0.0, 0.0, 0.0]
        pt[n_axis] = position
        pt[p_axis] = face_pos
        pt[span_axis] = t
        ends.append(_uv(pt, u_axis, v_axis, scale, flip))
    xs, ys = zip(*ends)
    ax.plot(xs, ys, color=color, linewidth=3.0, alpha=0.85, solid_capstyle="butt", zorder=3.0)
    ax.annotate(
        getattr(port, "name", "port"),
        (0.5 * (xs[0] + xs[1]), 0.5 * (ys[0] + ys[1])),
        textcoords="offset points",
        xytext=(4, 4),
        color=color,
        fontsize=7,
        zorder=3.1,
    )
    return True


def plot_cross_section(
    geometry: "GeometryModel",
    normal: str,
    position: float,
    *,
    mesh: "Mesh | None" = None,
    scale_mm: bool = True,
    flip: bool = False,
    ax: "matplotlib.axes.Axes | None" = None,
    title: str | None = None,
    deflection: float = 1e-4,
    outline_transparent: bool = True,
    show_wires: bool = True,
    show_ports: bool = True,
    slab: float = 0.0,
) -> tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Plot a 2D cross-section of 3D geometry at an axis-aligned plane.

    Intersects every shape in *geometry* with the specified plane and
    renders the resulting polygons as matplotlib patches, coloured by
    material.  Shapes with a fully transparent material (air/vacuum)
    are drawn as a dashed outline instead of being skipped — for a
    cavity carved into a conducting background, that outline *is* the
    wall.

    Features that carry no volume are drawn too, since a picture of a
    wire antenna showing only its air box is not a picture of the
    model: thin wires from their curve, discrete ports and lumped
    elements from their two endpoints, and face ports along the domain
    edge they occupy.  Each appears as a line where the cut runs along
    it and as a ring where the cut passes through it.

    Parameters
    ----------
    geometry : GeometryModel
        Geometry model containing the shapes to slice.
    normal : str
        Normal axis of the cutting plane: ``'x'``, ``'y'``, or ``'z'``.
    position : float
        Position along the normal axis in metres.
    mesh : Mesh or None, optional
        If given, overlay the mesh grid lines for the cutting plane.
    scale_mm : bool, optional
        If *True* (default), display axes in millimetres.
    flip : bool, optional
        If *True*, swap the horizontal and vertical axes.  Useful for
        structures that are tall and narrow in the default (u, v) layout
        (e.g. a long transmission line sliced at ``x = const``).
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw into.  A new figure is created when *None*.
    title : str or None, optional
        Plot title.  Defaults to ``"Cross-section at <axis> = <pos> <unit>"``.
    deflection : float, optional
        Chordal deflection for curve tessellation [m].  Passed through to
        :func:`~magnelio.geo._occ_backend.cross_section_polygons`.
    outline_transparent : bool, optional
        Draw transparent-material shapes as dashed outlines (default
        True).  Set False to skip them entirely (pre-existing
        behaviour).  Shapes whose material has ``visible=False`` are
        always skipped.
    show_wires : bool, optional
        Draw :class:`~magnelio.geo.ThinWire` conductors (default True).
        A wire is a sub-cell model, so it is drawn at a fixed line
        width rather than to its radius — which would be invisible.
        The cut counts as passing through a wire when it comes within
        one radius of it.
    show_ports : bool, optional
        Draw the model's declared ports and lumped elements (default
        True), labelled.  A port declared on a bbox face parallel to
        the cut is not drawn: it would cover the entire section.
    slab : float, optional
        Half-thickness of the layer the picture stands for [m], zero by
        default (a mathematical plane).  Volume-free features — thin
        wires, discrete ports, lumped elements — are drawn when they
        fall within this distance of the plane.  Field plots pass the
        half-height of the cell layer they display, so a wire on a grid
        node still appears in a picture whose field samples sit half a
        cell off it.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    import matplotlib.patheffects as patheffects  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.patches import Polygon as MplPolygon  # noqa: PLC0415

    from magnelio.geo._occ_backend import cross_section_polygons  # noqa: PLC0415
    from magnelio.post._colors import material_color  # noqa: PLC0415

    if normal not in _AXIS_LABELS:
        raise ValueError(f"normal must be 'x', 'y', or 'z'; got {normal!r}")

    scale = 1e3 if scale_mm else 1.0
    unit = "mm" if scale_mm else "m"
    u_label, v_label = _AXIS_LABELS[normal]
    if flip:
        u_label, v_label = v_label, u_label

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig: matplotlib.figure.Figure = ax.figure  # pyright: ignore[reportAssignmentType]

    from magnelio.geo.wire import ThinWire  # noqa: PLC0415

    all_shapes = list(geometry)
    geo_scale = 1.0
    if all_shapes and all(hasattr(s, "_analytic_bbox") for s in all_shapes):
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(all_shapes)

    # Thin wires carry no volume, so a plane section returns nothing for
    # them; they are drawn from their curve further down.
    wires = [s for s in all_shapes if isinstance(s, ThinWire)]
    shapes = [s for s in all_shapes if not isinstance(s, ThinWire)]

    for shape in shapes:
        mat = shape.material
        if not mat.visible:
            continue

        rgba = material_color(mat)
        transparent = rgba[3] < 1e-6
        if transparent and not outline_transparent:
            continue  # fully transparent (air) — skip

        try:
            occ_shape = shape._occ_shape(geo_scale)
        except Exception:
            continue

        polygons = cross_section_polygons(
            occ_shape,
            normal,
            position,
            deflection,
            scale=geo_scale,
            # Cutting exactly along a face is what a user asks for first
            # (the top of a substrate, the plane of a metal layer), and
            # it is the case a plain section gets wrong.
            exact_at_faces=True,
        )

        for poly_verts in polygons:
            if flip:
                poly_verts = poly_verts[:, ::-1]
            scaled = poly_verts * scale
            if transparent:
                # Boundary of an air/vacuum region (e.g. a cavity wall):
                # dashed outline with a white under-stroke so it reads on
                # any field colour map underneath.
                patch = MplPolygon(
                    scaled,
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.9,
                    linestyle=(0, (4, 3)),
                )
                patch.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=2.2, foreground="white", alpha=0.85),
                        patheffects.Normal(),
                    ]
                )
            else:
                patch = MplPolygon(
                    scaled,
                    closed=True,
                    facecolor=rgba[:3],
                    alpha=rgba[3],
                    edgecolor=(0.3, 0.3, 0.3, 0.5),
                    linewidth=0.6,
                )
            ax.add_patch(patch)

    # -- thin wires, ports and lumped elements ------------------------------
    n_axis = _AXIS_INDEX[normal]
    u_axis, v_axis = (_AXIS_INDEX[a] for a in _AXIS_LABELS[normal])
    common = dict(
        n_axis=n_axis,
        position=position,
        u_axis=u_axis,
        v_axis=v_axis,
        scale=scale,
        flip=flip,
    )

    if show_wires and wires:
        from magnelio.geo._occ_backend import sample_wire  # noqa: PLC0415

        for wire in wires:
            try:
                pts = sample_wire(wire._occ_shape(geo_scale), _wire_step(wire), geo_scale)
            except Exception:
                continue
            _draw_path(
                ax,
                pts,
                tol=max(_WIRE_HIT_RADII * wire.radius, slab),
                color=_WIRE_COLOR,
                label=wire.name,
                **common,
            )

    if show_ports:
        bbox = None
        for port in getattr(geometry, "ports", ()):
            if hasattr(port, "start") and hasattr(port, "end"):
                _draw_path(
                    ax,
                    [port.start, port.end],
                    # A discrete port bridges one edge; it is a line in
                    # the cut only when the cut contains it, so the
                    # tolerance is numerical — unless the caller states
                    # the layer thickness the picture stands for.
                    tol=max(_edge_tol(port.start, port.end), slab),
                    color=_PORT_COLOR,
                    label=getattr(port, "name", None),
                    **common,
                )
                continue
            if hasattr(port, "plane"):
                if bbox is None:
                    try:
                        bbox = geometry.bounding_box()
                    except Exception:
                        break
                _draw_face_port(ax, port, bbox, color=_PORT_COLOR, **common)

        for element in getattr(geometry, "elements", ()):
            _draw_path(
                ax,
                [element.start, element.end],
                tol=max(_edge_tol(element.start, element.end), slab),
                color=_ELEMENT_COLOR,
                label=getattr(element, "name", None),
                **common,
            )

    # -- mesh grid overlay --------------------------------------------------
    if mesh is not None:
        grid = mesh.grid
        u_nodes = getattr(grid, u_label) * scale
        v_nodes = getattr(grid, v_label) * scale
        ax.vlines(
            u_nodes,
            v_nodes[0],
            v_nodes[-1],
            colors="#888888",
            linewidths=0.3,
            alpha=0.4,
            zorder=1.5,
        )
        ax.hlines(
            v_nodes,
            u_nodes[0],
            u_nodes[-1],
            colors="#888888",
            linewidths=0.3,
            alpha=0.4,
            zorder=1.5,
        )

    ax.set_xlabel(f"{u_label} [{unit}]")
    ax.set_ylabel(f"{v_label} [{unit}]")
    ax.margins(0)
    ax.set_aspect("equal")
    ax.autoscale_view()

    if title is None:
        pos_display = position * scale
        title = f"Cross-section at {normal} = {pos_display:.2f} {unit}"
    ax.set_title(title, fontsize=9)

    return fig, ax


# ---------------------------------------------------------------------------
# 3D interactive view (Jupyter)
# ---------------------------------------------------------------------------


def _rgba_to_hex(r: float, g: float, b: float) -> str:
    """Convert (R, G, B) floats in [0, 1] to ``#rrggbb`` hex string."""
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def _display_renderer(renderer: Any) -> None:
    """Display a JupyterRenderer rebuilt for SI-unit (metre-scale) geometry.

    Replicates JupyterRenderer.Display() but with two corrections that the
    pythonocc default settings get wrong for metre-scale EM structures:

    1. CombinedCamera has separate clipping planes per mode — ``orthoNear`` /
       ``orthoFar`` for orthographic, ``near`` / ``far`` for perspective.  The
       pythonocc defaults (0.1, 2000) clip all geometry on a ~10 mm coax.
    2. CombinedCamera's ``width`` / ``height`` are the orthographic frustum
       in *world units*, not canvas pixels.  Passing ``self._size`` (pixel
       dimensions, e.g. 800×500) makes the visible viewport 320 m wide after
       zoom — the 10 mm coax becomes a sub-pixel speck on the canvas.

    Both planes and frustum dimensions are scaled with the bounding box, so
    the function is unit-independent.  The renderer canvas itself is still
    sized in pixels via the ``Renderer`` widget's own width/height.

    Must be called after renderer._shapes has been populated and wrapped in
    single-element lists (required by pythonocc ≥7.9.3's BoundingBox API).
    """
    import itertools  # noqa: PLC0415
    import math  # noqa: PLC0415

    from IPython.display import display  # noqa: PLC0415
    from ipywidgets import HBox, VBox  # noqa: PLC0415
    from OCC.Display.WebGl.jupyter_renderer import Axes, BoundingBox, Grid  # noqa: PLC0415
    from pythreejs import (  # noqa: PLC0415
        AmbientLight,
        CombinedCamera,
        DirectionalLight,
        OrbitControls,
        Picker,
        Scene,
    )
    from pythreejs import (
        Renderer as _WebGLRenderer,
    )

    _bb: Any = BoundingBox(list(renderer._shapes.values()))
    renderer._bb = _bb
    bb_max = _bb.max
    orbit_radius = 1.5 * _bb._max_dist_from_center()

    camera_target = _bb.center
    camera_pos = [c + s for c, s in zip(_bb.center, renderer._scale([1, 1, 1]))]

    # Frustum in world units, sized to give ~4×bb_max of visible scene after
    # the standard zoom factor.  Aspect ratio follows the canvas pixel ratio.
    canvas_aspect = renderer._size[0] / renderer._size[1]
    cam_view_height = 4.0 * bb_max * renderer._camera_initial_zoom
    cam_view_width = cam_view_height * canvas_aspect

    renderer._camera = CombinedCamera(
        position=camera_pos,
        width=cam_view_width,
        height=cam_view_height,
        near=bb_max * 1e-3,
        far=bb_max * 1e3,
        orthoNear=bb_max * 1e-3,
        orthoFar=bb_max * 1e3,
    )
    renderer._camera.up = (0.0, 0.0, 1.0)
    renderer._camera.mode = "orthographic"
    renderer._camera_target = camera_target
    renderer._camera.position = camera_pos

    positions = list(itertools.product(*[(-orbit_radius, orbit_radius)] * 3))
    key_lights = [DirectionalLight(color="white", position=pos, intensity=0.5) for pos in positions]
    ambient_light = AmbientLight(intensity=0.1)

    _occ_axes: Any = Axes(bb_center=_bb.center, length=bb_max * 1.1)
    renderer.axes = _occ_axes
    _hgrid: Any = Grid(
        bb_center=_bb.center,
        maximum=bb_max,
        colorCenterLine="#aaa",
        colorGrid="#ddd",
    )
    renderer.horizontal_grid = _hgrid
    _vgrid: Any = Grid(
        bb_center=_bb.center,
        maximum=bb_max,
        colorCenterLine="#aaa",
        colorGrid="#ddd",
    )
    renderer.vertical_grid = _vgrid

    environment = (
        _occ_axes.axes + key_lights + [ambient_light, _hgrid.grid, _vgrid.grid, renderer._camera]
    )
    scene = Scene(
        children=[
            renderer._displayed_pickable_objects,
            renderer._displayed_non_pickable_objects,
        ]
        + environment
    )

    renderer._controller = OrbitControls(controlling=renderer._camera, target=camera_target)
    renderer._camera.zoom = renderer._camera_initial_zoom
    renderer._update()

    renderer._picker = Picker(controlling=renderer._displayed_pickable_objects, event="dblclick")
    renderer._picker.observe(renderer.click)

    renderer._renderer = _WebGLRenderer(
        camera=renderer._camera,
        background=renderer._background,
        background_opacity=renderer._background_opacity,
        scene=scene,
        controls=[renderer._controller, renderer._picker],
        width=renderer._size[0],
        height=renderer._size[1],
        antialias=True,
    )

    _hgrid.set_position((0, 0, 0))
    _hgrid.set_rotation((math.pi / 2.0, 0, 0, "XYZ"))
    _vgrid.set_position((0, -bb_max, 0))
    renderer._savestate = (renderer._camera.rotation, renderer._controller.target)

    display(HBox([VBox([HBox(renderer._controls), renderer._renderer]), renderer.html]))


def _extract_solids(occ_shape: Any):
    """Decompose a TopoDS_Shape into individual solids.

    ``JupyterRenderer.DisplayShape`` cannot handle ``TopoDS_Compound``
    objects directly.  This helper extracts all ``TopoDS_Solid``
    sub-shapes via ``TopExp_Explorer``.  If the shape is already a
    solid (or contains none), it is returned as-is in a single-element
    list.
    """
    from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    if occ_shape.ShapeType() != TopAbs_COMPOUND:
        return [occ_shape]

    solids = []
    explorer = TopExp_Explorer(occ_shape, TopAbs_SOLID)  # pyright: ignore[reportArgumentType]
    while explorer.More():
        solids.append(topods.Solid(explorer.Current()))
        explorer.Next()
    return solids if solids else [occ_shape]


def show_geometry(
    geometry: Any,
    *,
    size: tuple[int, int] = (800, 500),
    render_edges: bool = True,
    edge_color: str = "#202020",
    quality: float = 1.0,
):
    """Interactive 3D view of a :class:`~magnelio.geo.GeometryModel`.

    Uses the pythonocc ``JupyterRenderer`` widget.  Each shape is
    coloured according to its material (same palette as
    :func:`plot_cross_section`).  Air/vacuum shapes are rendered
    semi-transparent so that internal structure is visible.

    Calling this function directly displays the widget as a side-effect
    via ``IPython.display``; the return value is ``None``.

    Parameters
    ----------
    geometry : GeometryModel or list of shapes
        The geometry to display.
    size : (int, int)
        Widget size in pixels ``(width, height)``.
    render_edges : bool
        Draw tessellation edges on every shape.
    edge_color : str
        Hex colour for edges.
    quality : float
        Tessellation quality (higher = finer mesh, slower).
    """
    from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer  # noqa: PLC0415

    from magnelio.post._colors import material_color  # noqa: PLC0415

    renderer: Any = JupyterRenderer(size=size)

    shapes = list(geometry)
    # DD-120: the 3D preview is unit-less — displaying the scaled-unit
    # shape is fine and keeps micron models buildable.
    geo_scale = 1.0
    if shapes and all(hasattr(s, "_analytic_bbox") for s in shapes):
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(shapes)

    for shape in shapes:
        occ_shp = shape._occ_shape(geo_scale)
        mat = shape.material
        rgba = material_color(mat)

        is_air = rgba[3] < 1e-6
        color_hex = _rgba_to_hex(*rgba[:3]) if not is_air else "#e0e8f0"
        opacity = rgba[3] if not is_air else 0.15
        transparency = opacity < 0.99

        for solid in _extract_solids(occ_shp):
            renderer.DisplayShape(
                solid,
                shape_color=color_hex,
                render_edges=render_edges,
                edge_color=edge_color,
                quality=quality,
                transparency=transparency,
                opacity=opacity,
            )

    # pythonocc ≥7.9.3: BoundingBox now calls _bbox(obj) for each element in
    # shapes.values(), expecting an iterable of OCC shapes.  Wrapping each
    # stored shape in a single-element list satisfies the new API while also
    # remaining harmless on older builds.
    for k in renderer._shapes:
        renderer._shapes[k] = [renderer._shapes[k]]

    # Remove shape-inspection controls (Compute dropdown, Hide/Show, Remove)
    # that require double-click selection and are irrelevant for EM viewing.
    renderer._controls = renderer._controls[:3]

    # Use _display_renderer() instead of renderer.Display() to set correct
    # camera near/far at construction time (see docstring there).
    _display_renderer(renderer)
    return None
