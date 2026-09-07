# 3D viewer

`model.plot()` opens an interactive 3D view of a
{class}`~magnelio.GeometryModel`: the solids coloured by material, the
declared features, and — with a mesh — the FIT grid.  The same call
serves three situations:

- in a **Jupyter notebook** it is a widget, rendered in the browser;
- in a **script** it opens a window;
- in a **documentation build** it becomes a figure with two tabs: a
  screenshot, and the same scene as a rotatable view in the browser
  (this is how the 3D figures in the tutorials are made).  The browser
  tab has no toolbar — the cutting plane stays where the script put it.

```python
model.plot()                                  # geometry only
model.plot(mesh=mesh, cut=("y", 0.0))         # opened along y = 0, grid cells on the cut
```

## What is drawn

| Item | Appearance |
|---|---|
| Solids | Material colour of the 2D cross-sections (metals grey, dielectrics tinted by permittivity, air and vacuum as faint translucent shells); imported CAD colours are honoured. |
| Grid (`mesh=`) | On the cutting plane, the exposed cell layer as a sheet of cell faces, each coloured by the material the mesher assigned — the discretised model as the solver sees it.  The grid is shown nowhere else: a wireframe of the domain faces was tried and dropped, its foreshortened lines in front of the cut confused more than they informed. |
| Thin wires | Tubes in the wire colour. |
| Discrete ports, lumped elements | Tubes between their two end points (red for ports, green for elements), with their name beside them. |
| Waveguide ports | A translucent red window on the domain face they occupy — the declared sub-window, or the whole face — with the port name written in the window's plane. |
| Symmetry planes | Tinted sheets on the declared symmetry planes (`SymmetryPEC` blue, `SymmetryPMC` green) — where the solver cut the model, which for a fully modelled geometry is inside the picture, not on its edge. |
| Domain box | Outline of the computational domain — the grid extent when a mesh is given, including the absorbing buffer at open faces; the bounds of the solids and the field otherwise. |

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
- **Show** — a menu of the object groups (solids, grid on cut, ports,
  lumped elements, wires, labels, symmetry planes, domain box); untick
  a group to hide it.

Before the cut controls sit the camera buttons: reset, isometric view,
a view along x, y or z, the projection toggle (parallel, as the scene
opens, or perspective), a screenshot, a **pop-out** button that opens
the same view in a browser tab of its own (`size=` sets the height of
the widget in the notebook), and **help**, a dialog listing every
control and the mouse bindings.

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

## Fields on the cut

The same viewer shows a field: a monitor's recording, or a
{class}`~magnelio.fields.FieldState` such as an eigenmode.  The cell
layer the cut exposes is laid over the cut as a coloured sheet, and the
position slider walks that layer through the recorded volume.

```python
monitor.show()                                   # |E| with arrows, mid-plane of the region
monitor.show("Ez", normal="y", position=0.0)     # one signed component, diverging colours
monitor.show(geometry=model, mesh=mesh)          # with the solids; metal cut out of the sheet
pattern.show(f=10e9, phase=90.0)                 # a frequency monitor at a phase
eigen.show("H", geometry=model)                  # an eigenmode result: a slider over the modes
eigen.show(frame=2, glyph="cone")                # starting at mode 2, cones instead of arrows
```

| Item | Appearance |
|---|---|
| Field sheet | The exposed layer of cells, each coloured by the magnitude of `E` or `H` (dark to bright) or by one signed component (`Ex`, `Hz`, …; blue–white–red about zero).  The colour ceiling is the peak over every frame and layer of the recording, so a wave keeps its colour while the frame slider runs; `vmax=` fixes it. |
| Vectors | For `E` or `H`: arrows on an even lattice over the layer, centred on their sample points, all three components, coloured by their magnitude on the sheet's scale; the length grows with the magnitude from three tenths of the lattice spacing to one spacing, so a decaying field keeps readable arrows.  Arrows below 2 % of the ceiling are left out; `plot_type="color"` drops them, `arrow_color=` paints them one colour, `glyph="cone"` draws cones and `glyph_width=` sets the thickness. |
| Metal | With `mesh=`, cells buried in a perfect conductor are cut out of the sheet, so the solids' cut faces show through where no field is defined. |
| Symmetry | With `mesh=` (an eigenmode result brings its own), a field recorded behind the model's symmetry planes is continued across them with the parity of each component, so the picture is the whole model like every other field plot; `mirror=False` shows the modelled part.  A region that stops short of a plane is not mirrored across it. |
| Lines and points | A monitor of one cell along two or three axes shows its row of cells, or its one cell, with the arrows on it. |

The field controls sit in a **second toolbar row**: a **play button**
and a **frame slider** (time, frequency or mode, the value beside it
in a readout of fixed width), a **play button** and a **phase slider**
for complex data (a frequency monitor's pattern at `Re(F·e^{jφ})`; the
play turns the phase in steps of ten degrees), a **Field** selector
that switches between the recorded components, and the level and
density sliders of the volume representations below; *Field on cut*
and *Vectors on cut* join the *Show* menu.  Play runs the frames in a
loop at `fps=` (default 4) — each frame is one layer computed and sent
to the browser, so the rate is bounded by the size of the layer.  In a
script or a documentation build the initial frame is chosen with `t=`,
`f=` or `frame=`.

An eigenmode result shows the same way, `result.show()`: its modes are
the frames, labelled with index and eigenfrequency (degenerate pairs
share the frequency, so the index leads), the amplitudes in arbitrary
units; a complex Bloch mode is turned to the instant of its maximum
energy first, the phase slider turns it from there.

## Fields in the volume

The cut shows one layer; the volume behind it can carry the field
too, in the two forms every 3D field plot offers.

```python
monitor.show(volume="arrows")                    # arrows on a 3D lattice over the kept half
monitor.show(volume="isosurface", iso_level=0.4) # |E| at 40 % of the ceiling
monitor.show("Hx", volume="both", levels=[2.0])  # ±2 A/m surfaces, arrows too
```

| Item | Appearance |
|---|---|
| Field vectors | The same arrows as on the cut, on an even 3D lattice over the region (`density=` counts them along its longest axis), clipped to the kept half like the solids; they replace the vectors on the cut.  A slider sets the density. |
| Isosurfaces | Translucent surfaces where the magnitude equals a level — by default one at `iso_level` (half the colour ceiling), which a slider moves between 5 and 95 %; `levels=` fixes surfaces at values in V/m or A/m.  A signed component gets each level with both signs, blue and red.  The surfaces are clipped at the cut, so the cut shows their inside. |

Both are entries of the *Show* menu whenever the source is a volume
(isosurfaces need at least two cells along every axis, so a plane
monitor offers arrows only); `volume=` only chooses what is on at
first.  Without a cut (*Cut* set to *off*) the whole region is drawn.
A volume representation costs the whole region per frame instead of
one layer — a large monitor plays slower with it on.  The cell values
are interpolated to the grid nodes before the surfaces are contoured,
so a surface is as fine as the grid.  Volume rendering with an opacity
ramp is not offered: the browser renderer takes only uniform image
data, and a graded grid resampled onto one would lose its resolution.

One thing the picture is not: a plane.  Every value is the cell-centre
average of the staggered components in one layer of cells, the same
convention as the 2D slice plots.

A monitor read back from a project store (`project.monitors[...]`)
shows the same way; its frames are read from disk one at a time as the
slider moves, so a volume monitor of any size opens at once.

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

The mouse in the notebook widget: left drag orbits, middle drag (or
alt + left drag) pans, right drag, the wheel or ctrl + left drag zoom,
alt + shift + left drag rolls.  The rotatable views in these
documentation pages use vtk.js's own bindings instead: shift + left
drag pans there, alt + left drag zooms.  The toolbar's help button
lists the bindings of the widget.

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

The first `plot()` in a kernel starts that websocket server on the
kernel's own event loop and fills the cell's output the moment the
server is up — a fraction of a second after the cell returns when you
run cells by hand.  Under *Run All* the cells queued behind it hold the
loop, so the view appears once they have run.  Nothing is nested into
the running loop, so *Run All* works like running the cells one by one;
`nest-asyncio2` is what PyVista's own first-call path uses and is not
needed for Magnelio's viewer.

## Limitations

- Names are flat 3D text placed in the scene (the browser renderer has
  no screen-space labels): port names lie in the port plane, element
  names face the initial camera; both scale with the model.
- The cutting plane is axis-aligned by design (see above).
- Fields are drawn on the cut and, as arrows and isosurfaces, in the
  volume — not as a volume rendering; the
  {ref}`ParaView export <paraview-export>` covers that.
- The camera presets do not turn the flat labels: they face the
  camera the view opened with.
