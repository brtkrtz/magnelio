# 3D viewer

`model.plot()` opens an interactive 3D view of a
{class}`~magnelio.GeometryModel`: the solids coloured by material, the
declared features, and — with a mesh — the FIT grid.  The same call
serves three situations:

- in a **Jupyter notebook** it is a widget, rendered in the browser;
- in a **script** it opens a window;
- in a **documentation build** it becomes a screenshot (this is how the
  3D figures in the tutorials are made).

```python
model.plot()                                  # geometry only
model.plot(mesh=mesh)                         # with the grid on the domain faces
model.plot(mesh=mesh, cut=("y", 0.0))         # opened along y = 0
```

## What is drawn

| Item | Appearance |
|---|---|
| Solids | Material colour of the 2D cross-sections (metals grey, dielectrics tinted by permittivity, air and vacuum as faint translucent shells); imported CAD colours are honoured. |
| Grid (`mesh=`) | Grid lines on the six domain faces.  On the cutting plane, the exposed cell layer as a sheet of cell faces, each coloured by the material the mesher assigned — the discretised model as the solver sees it. |
| Thin wires | Tubes in the wire colour. |
| Discrete ports, lumped elements | Tubes between their two end points (red for ports, green for elements), with their name beside them. |
| Waveguide ports | A translucent red window on the domain face they occupy — the declared sub-window, or the whole face — with the port name on it. |
| Symmetry planes | Tinted sheets on the domain faces declared `SymmetryPEC` / `SymmetryPMC` (blue / green). |
| Domain | Outline of the bounding box; the grid extent when a mesh is given. |

Lengths are shown in millimetres (`scale_mm=False` for metres).  The
projection is parallel, as in engineering drawings.

## The cutting plane

The cut is **axis-aligned**: a normal (`x`, `y` or `z`), a position
along that axis, and a side to remove.  In the widget it lives in the
toolbar:

- **Cut** — `off`, `x`, `y`, `z`;
- the **position slider** across the domain extent;
- **Flip** — remove the other half;
- **undo** (last change) and **reset** (initial state);
- **Show** — a menu of the object groups (solids, grid lines, cut
  cells, ports, lumped elements, wires, labels, symmetry planes, domain
  box); untick a group to hide it.

A single plane cuts every solid; the openings are capped, so a cut
metal body reads as solid metal, not as a hollow shell.  With a mesh,
the cells the cut exposes are laid over the cut faces as a translucent
sheet — the caps stay visible through it.
The features follow the cut: a wire is clipped with the solids, and a
port, element or label in the removed half disappears with it.

`cut=("y", 0.0)` sets the initial state of the plane (and `flip=True`
the side); it is the only way to place the plane for a screenshot, and
the way a tutorial fixes the picture it wants.

Why not a plane grabbed and turned freely in 3D?  A FIT grid carries
information only on its own planes — an oblique cut through cells shows
triangles that mean nothing — and a 3D handle competes with the camera
for the mouse.  Axis-aligned, slider-driven cutting planes are also
what users of commercial EM suites expect.

## Rendering modes

```{list-table}
:header-rows: 1

* - `mode`
  - Where the picture is rendered
  - When to use it
* - `"client"` (default)
  - In the browser (vtk.js).  The scene is sent once; orbit, pan and zoom
    cost nothing on the kernel side.  Needs no OpenGL in the kernel.
  - Everyday use.
* - `"server"`
  - In the kernel (VTK); images are streamed to the browser.
  - Scenes too large for the browser (many millions of triangles); needs
    OpenGL in the kernel.
* - `"trame"`
  - Both, with a toggle in the toolbar.
  - Comparing the two.
* - `"static"`
  - A screenshot embedded in the notebook.
  - Notebooks meant to be read without a kernel.
* - `"none"`
  - Not shown; the {class}`pyvista.Plotter` is returned.
  - Scripts that want `plotter.screenshot(...)`, tests.
```

The camera: left-drag orbits, right-drag (or shift + left-drag) pans,
the wheel zooms.  The toolbar's own buttons reset the camera to the
axis views.

## Requirements

The viewer is built on [PyVista](https://docs.pyvista.org), which is a
core dependency.  The notebook widget additionally needs the trame
stack:

```bash
pip install "magnelio[jupyter]"
# or, with conda-forge:
conda install trame trame-vtk trame-vuetify nest-asyncio2
```

Without it the view falls back to a static image and says so.  The
widget talks to the kernel over its own websocket on `localhost`; on a
remote JupyterHub set `PYVISTA_TRAME_JUPYTER_MODE` as described in the
PyVista documentation.

## Limitations

- Names are flat 3D text placed in the scene (the browser renderer has
  no screen-space labels); they read best from the default camera and
  scale with the model.
- The cutting plane is axis-aligned by design (see above).
- Field monitors are not yet shown in 3D; use the ParaView export
  ({doc}`sources-monitors`) or the 2D slice plots.
