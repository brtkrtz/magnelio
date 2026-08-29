# Changelog

All notable changes to Magnelio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).  While the
major version is 0, minor releases may change the public API.

## [Unreleased]

### Fixed

- The mesh build's thin-sheet pass sections a sheet's solid through
  the same in-house section engine the material passes use, paints
  its boundary band only where a segment can reach, and decides the
  substrate side of a sheet from the construction of the surrounding
  bodies instead of classifying points against a pocketed Boolean
  solid.  A 16 × 16 patch array with feed meshes in 7.8 s instead of
  9.6 s, sixteen Lange couplers in 9.8 s instead of 10.6 s; meshes
  are unchanged.
- The mesh build no longer fuses the metal of a PEC housing into the
  housing's own cut when constructing the effective PEC solid — the
  cut holds it already — and cuts the housing by an air box rather
  than by the air body carrying the imprint of every metal piece.
  Sixteen Lange couplers mesh in 10.6 s instead of 12.5 s, a 16 × 16
  patch array with feed in 9.6 s instead of 11.0 s; meshes are
  unchanged except in the last bits of the edge-fraction arrays of
  models whose kernel-fused vertices differed from the input
  coordinates by an ulp.
- The edge pass of the mesh build decides grid lines against planar
  faces with round outlines — the caps of posts and pins, plates with
  round holes — in-house, exactly on the circle, instead of through the
  geometry kernel's line–face intersector.  A row of 240 PEC posts
  meshes in 2.4 s instead of 3.0 s; meshes are unchanged.
- The mesh build's section engine answers every grid plane of an axis
  in one compiled pass instead of plane by plane through NumPy — the
  conformal area passes and the cell classifier of models with many
  small features were spending their time in per-plane overhead, not
  geometry.  A row of 240 PEC posts meshes in 3.5 s instead of 7.1 s,
  sixteen Lange couplers in 12.9 s instead of 17.2 s; meshes of planar
  models are unchanged, those with round features change only in the
  last bits.
- The edge pass of the mesh build intersects its grid lines with
  cylindrical faces — posts, pins, vias, holes — in closed form
  instead of through the geometry kernel's line–face intersector, and
  a point that a probe line merely touches on a curved conductor now
  counts as lying on it.  A row of 240 PEC posts meshes in 6.6 s
  instead of 8.5 s; meshes change only in the last bits.
- Planes through cylindrical faces — posts, pins, vias, holes, round
  wires, fillets — are sectioned by the mesher's own engine instead of
  the geometry kernel's Boolean: the crossing points of the rim circles
  are computed analytically, the trace on the cylinder is its exact
  conic or generatrix pair, tessellated at the kernel's own chord and
  angle rule.  A row of 240 PEC posts meshes in 8.4 s instead of 20 s;
  the sections agree with the kernel's to rounding, so meshes change
  only in the last bits.  Cones, spheres, tori and cylinders trimmed by
  other curves keep the kernel route.
- The conformal area passes of the mesh build no longer rebuild their
  per-shape section engine for every pass, group their faces by plane
  with one sort instead of a loop over every face, and test a plane
  against all shapes' bounding boxes at once (a 16 × 16 patch array
  17.4 s → 12.6 s, a row of 16 Lange couplers 20.1 s → 17.1 s;
  identical meshes).
- The union of coplanar rectangular conductors — the strips and pads
  of a planar layout — is computed on the grid of their vertex
  coordinates instead of by the geometry kernel's fuse (a 16 × 16
  patch array with its corporate feed: 2.6 s → 0.2 s for the union,
  the build 20.6 s → 17.4 s; an 8 × 8 array 5.0 s → 4.6 s).  Layouts
  with arcs or rotated edges take the kernel route as before.  The
  union's vertices are now the input coordinates exactly; where the
  kernel's rounding of a crossing had set a grid plane, the plane moves
  by that rounding (below 1e-18 m).
- Two mesh-build terms that grew faster than the layout on large planar
  designs are gone: the series-permittivity correction for grid edges
  crossing a floor-absorbed material plane runs over arrays instead of
  a Python loop per edge and segment, and the geometry-edge plane
  extraction no longer re-derives a face's parametric bounds for every
  edge it tests (a 16 × 16 patch array with its corporate feed: 27 s →
  2 s and 19 s → 0.7 s for the two steps, the build 65 s → 21 s; an 8 × 8 array 6.9 s → 5.0 s;
  identical meshes).
- The conformal edge pass of the mesh build runs in batch: the carrier
  lines of all grid edges are collected once and intersected in
  compiled passes, the per-edge bookkeeping and the classification of
  edges lying in conductor faces run over arrays, and a point on a
  conductor's corner line is recognised as boundary without the
  kernel's classifier.  A row of eight Lange couplers spends 0.3 s
  instead of 6 s there, an 8 × 8 patch array 0.2 s instead of 9 s
  (the full builds 11.6 s → 9.0 s and 10.8 s → 6.9 s; a row of 16 couplers 26.7 s → 20.6 s, a 16 × 16 array 92 s → 65 s); identical meshes.
- The effective PEC solid no longer cuts metal out of metal (a higher
  PEC shape is in the union anyway), and the overlap check of
  `GeometryModel.validate` skips the pairs a `Difference` proves
  disjoint — a body and the tools it was cut with.  A row of 16 Lange
  couplers in a PEC housing saves 6 s of kernel Booleans (32 s → 27 s);
  identical meshes, and a genuine overlap still raises.
- Mesh build time on large models drops by about a third: the planar
  section engine pairs a plane's crossings for all faces at once
  instead of face by face, and the conformal passes collect the
  boundary faces of the grid as arrays instead of walking every face
  in Python (a row of eight Lange couplers, 1.8 M cells: 20 s → 14 s;
  an 8 × 8 patch array: 15 s → 12 s; identical meshes).
- Mesh build of a body whose one planar face carries the outline of
  hundreds of features — an air body over a row of posts or vias, a
  ground layer under many pads — no longer pays for that face on every
  section across the row: the face is cut once into tiles and each
  plane is sectioned over the tile it crosses (a row of 240 posts:
  46 s → 24 s for the build, identical mesh).  Winding the contours
  of a section with hundreds of holes is vectorised (290 ms → 6 ms per
  section).
- A union of thousands of coplanar bodies — the copper of a large
  patch array with its feed, an imported layer of pads and traces —
  no longer spends its build in one kernel fuse of all of them: the
  bodies are fused pairwise up a spatial bisection tree in their
  plane (a 16 × 16 patch array with its corporate feed, 1 787 strips:
  13 s → 2.6 s for the union, identical point set).  The thin-sheet
  footprint tests each contour of the metal outline only on the edges
  inside its own bounding box (the same array: 3.8 s → 1.6 s for the
  sheet, identical masks).
- Unions of many coplanar bodies — the strips of a feed network, the
  pads of a layer, a row of posts — are fused in their common plane and
  raised once instead of going to the kernel's general fuser, which
  spends its time on exactly that overlap (an 8 × 8 patch array with
  its corporate feed: 16 s → 0.8 s for the union, and the fused copper
  carries 640 faces instead of 6 924 because the seams between the
  strips are gone).  The mesher's priority-resolved conductor solid is
  cut per body instead of against an accumulated union (a row of 16
  Lange couplers: 36 s → 2 s).  The fused point sets are unchanged.
- Meshing a model whose air body is cut by hundreds of parts — a row
  of couplers, a board with its copper subtracted — no longer spends
  most of its build outside the mesher passes: the overlap check
  intersects such a body with all its neighbours in one Boolean
  instead of one per neighbour, and the material probes of the
  singular-edge and thin-sheet detection keep one loaded classifier
  per shape and ask only the shapes whose bounding box holds the
  point (16 Lange couplers on one carrier: 106 s → 68 s to mesh; the
  meshes are unchanged).
- The conformal edge pass no longer slows down on a large planar face
  with a complex outline — the copper of a fused network, an imported
  board layer, a ground plane full of vias: such faces are classified
  through pieces of a few edges each, with bit-identical edge
  fractions (8 × 8 patch array with feed: 61 s → 34 s to mesh).
- Mesh build of a large planar copper network — a patch array with its
  feed, an imported board layer, any union of many traces — no longer
  spends minutes painting the thin-sheet footprint: the outline comes
  from one section of the metal at mid-thickness instead of one solid
  classification per grid edge.  A 4 × 4 patch array with corporate
  feed builds in 10 s instead of 475 s; the footprint itself is
  unchanged (holes stay open, edges on the outline count as metal).
- A thin metallisation reaching an absorbing wall is now continued into
  the absorber like any other conductor, so a microstrip can carry a
  waveguide-port window in a CPML face (the port used to be refused
  with an "inhomogeneous filling" message).
- On grids finer than about 15 µm a face lying in a conductor's end
  wall was blocked instead of free and the wall's area went unbooked:
  the side step that keeps a grid plane off a coincident model face was
  below the kernel's tolerance, where the section reports the face on
  both sides.  The step now clears the largest tolerance in the model
  by a factor of four.  The shifted planes are thereby answered by the
  exact planar engine instead of the kernel's Boolean, and the kernel's
  sections run over the faces a plane can reach instead of the whole
  body: the face pass of 16 Lange couplers no longer grows faster than
  the cell count.

### Added

- Far-field power balance: `FarFieldResult.surface_power` is the real
  power the recorded fields carry out of the far-field monitor's box,
  `power_balance` its ratio to `P_rad`, and `monitor.result(f)` warns
  when the two differ by more than 5 % — the sign of a box that sits
  too close to the radiator (the box lies at the absorbing faces, so
  the cure is clearance; half a wavelength in the direction of the
  main beam is enough for a patch).  Realized gain and gain are low by
  that factor when it fires; directivity is not affected.  Tutorial 19
  (Cassegrain) quotes its balance next to the radiated-over-accepted
  power, to show how the two numbers separate a close box from a
  feed guide's wall current.
- How-to *Patch array*: a 2 × 2 corporate-fed microstrip patch array
  designed from the element up — inset sweep and length trim of the
  patch, a feed network on the port solver's line impedances, in-phase
  feeding of the second row by a quarter-wave offset of the
  distribution line, a microstrip window port in the absorbing wall
  through a shielded launch, and the principal-plane cuts checked
  against element pattern times array factor.
- `mesh.planes` records where every grid plane came from — the material
  face, bounding box, geometry edge, thin sheet, wire, symmetry face or
  forced position that asked for it, per shape — together with the
  requested planes that were dropped or absorbed; `print(mesh.planes)`
  prints the report.  Meshes in a project store carry it.
- `plots.plot_mesh_section` / `mesh.plot_section`: a section of the mesh
  itself — grid lines styled by origin, absorber cells hatched, the
  model's exact outline on top, and the cells shaded by the exact
  conductor area share the sub-cell classifier measured
  (`fill="coverage"`, default), by the classified material
  (`fill="material"`) or by the permittivity the normal field component
  sees (`fill="conformal"`); `edges=True` adds the PEC-masked, partly
  free and borrowed edges.  `plot_cross_section(fill=False)` draws
  outlines only.
- The unit suite pins the grid planes of every example model
  (`tests/unit/data/gallery_planes/`), so a change of the meshing rules
  fails with a per-axis diff instead of a physics test downstream.
- Methods chapter *Mesh generation*: section *Inspecting the grid*;
  tutorial 02 prints the record and shows the mesh section.
- How-to guide *Lange coupler*: a 3-dB interdigitated coupler at 10 GHz
  on 254 µm alumina, dimensioned with the port solver alone — the
  interdigital synthesis gives the even- and odd-mode impedances one
  finger pair must have, sixteen slice meshes find the finger width and
  gap that deliver them on the grid the coupler is then solved on —
  with ribbon bonds as resolved metal and the coupling, quadrature
  phase, match and isolation read off the four-port run.
- Benchmark `benchmarks/bench_mesh_build.py`: mesh-build wall time per
  mesher pass on three production geometry classes (a row of Lange
  couplers, a patch array with corporate feed, a post row), with the
  section process pool off, in production mode or forced.

### Changed

- Tutorial 13 (dielectric filter) predicts the band its volume monitor
  records from the loaded resonator frequency and the design
  bandwidth instead of quoting fixed numbers; the corrected air bore
  of the ceramic ring (0.4.7) moves its passband to 3.02 GHz.
- The mesh build's edge pass — the free-length fraction of every grid
  edge against the conductors — decides the crossings of planar,
  axis-aligned faces itself (plane level, even-odd rule over the face
  outline with a tolerance band, outward normal), tiles large faces by
  clipping their outlines instead of cutting them with the kernel, and
  slices each edge from the crossings of its grid line instead of
  asking the kernel's intersector per edge: 8 × 8 patch array 34 →
  15 s, 16 Lange couplers 55 → 43 s, a 16 × 16 array 164 → 111 s
  (its edge pass 62 → 13 s), with the same fractions.  A grid edge lying exactly on a
  conductor's corner line that the kernel's classifier had missed now
  reads as conductor like its neighbours.

### Fixed

- `model.plot()` and the ParaView export no longer fail on bodies of a
  few tens of micrometres (a bond post, a via barrel): the tessellation
  deflection is floored at the kernel's precision instead of below it.
- Two thin metallisations at the same nominal height — a brick and a
  traced track on one substrate, say — no longer leave a pair of grid
  planes a float-rounding apart: sheet planes within the feature gap
  now share one plane (and snap onto a forced plane there), so the
  misleading "forced planes … closer than min_feature_gap" warning and
  the growth-factor warning that came with it are gone.
- A curved body touching a flat face — a cylinder inscribed in a box,
  a round hole reaching a wall — no longer places a grid plane through
  its own axis: the Boolean's split line along the touching line is
  not a geometry edge.  Such meshes are what they were in 0.4.4.

## [0.4.7] - 2026-08-26

### Added

- Tutorial 19 *Offset Cassegrain reflector*: paraboloid and
  hyperboloid built as parametric surfaces (the subreflector patch
  from a map that solves the quadric), turned into metal shells,
  fed by a pyramidal horn through a port window in an absorbing wall,
  with the far-field pattern of the tilted beam.  Rendered without
  execution in the gallery.
- How-to guide *Coupled-line directional coupler*: a −10 dB microstrip
  coupler dimensioned with the port solver alone — a port across both
  lines returns the even and odd modes with their impedances and
  effective permittivities — and built from two traced paths that
  turn onto the box walls; the four-port run is checked against the
  textbook coupled-line response.
- The port report lists the effective permittivity of every line mode
  (``report.modes[i].epsilon_eff``, printed alongside ``z_line``).
- Waveguide ports may sit in absorbing (CPML) faces: a window port in
  the cross-section of a conductor-enclosed guide that reaches the
  wall — the neck of a horn, a coax entering an open box.  The
  absorber is switched off behind the window, the far-field monitor
  leaves the feed guide out of its Huygens surface, and a whole-face
  port or a window in free space is refused with guidance.
- `geo.Surface.parametric(fn, u=, v=, samples=)` samples a parametric
  map (u, v) → (x, y, z) into a curved sheet — a paraboloid dish, a
  hyperboloid sub-reflector, a shaped surface given by a formula or a
  table — which `extruded()` or `thickened()` turn into a metal body.
  Moving, rotating, scaling or mirroring a sheet keeps it a sheet, so
  `Face(...).rotated(...).thickened(...)` works too.  New methods
  chapter *Geometry construction*.
- Waveguide ports on coupled lines — two or more signal conductors
  above a ground, such as an edge-coupled microstrip pair — now
  return the line's propagating modes (the even and odd pair of a
  symmetric pair, each with its own effective permittivity and
  impedance) instead of one mode per conductor.  Reading the
  even- and odd-mode impedances off the port report is the design
  step of a coupled-line coupler.
- Every 3D view in the tutorials and how-to guides now has an
  "Interactive Scene" tab next to the screenshot: the same scene,
  rotatable and zoomable in the browser.

### Fixed

- Hollow conductors — tubes, shells, plates with apertures — lost the
  conformal correction at their inner walls: the section contour of a
  bore came back with the same winding as the outer boundary and the
  dual faces at the bore wall were booked fully metallic.  Contours
  are now wound by nesting parity before the area kernels see them.
  The same defect filled the *air* bore of dielectric rings with their
  own material, so resonators with a hole change too — a ceramic ring
  (ε_r = 45, 4 mm / 2 mm bore) in a coarse test cavity moves from 2.33
  to 2.66 GHz; the solid puck sits at 2.23 GHz.
- A conductor touching an absorbing (CPML) face lost its PEC surface
  mask inside the absorber layer on conformal meshes (the classifier
  saw the solids end at the bounding box); the layer now carries the
  interior cross-section exactly.
- Far-field and frequency monitors of a run fed by a TE/TM waveguide
  port are now normalised to the incident power the run launched at
  each frequency rather than to the excitation waveform; gain and
  radiated power of horn-type feeds no longer carry the shape of the
  mode's wave impedance.  Lumped, TEM and quasi-TEM feeds are
  unchanged.

### Changed

- Free-form faces — parametric surfaces, lofts, imported B-spline
  patches — are sectioned on a triangulation whose section points are
  lifted back onto the exact surface and refined in-plane, instead of
  through the kernel Boolean: the Cassegrain tutorial meshes in under
  a minute instead of six, and the section polygons follow the exact
  surface more closely than the kernel's tessellation did.  Planar
  and analytic curved faces are handled as before.
- Tutorial 02, tutorial 06 and the stripline how-to show the model in
  the live 3D view (`model.plot()`) instead of a notebook screenshot;
  the magic tee gets a 3D view before its cross-sections.

## [0.4.6] - 2026-08-26

### Added

- The mesher's bulk cell size now follows the wavelength of the
  material *in the slab*, not of the densest material in the model:
  each interval between grid planes on an axis is meshed for the
  densest material whose bounding box reaches into it, so the air
  around a small ceramic or above a thin substrate is meshed at the
  air wavelength while the dielectric keeps its own.  A 10 × 10 × 2 mm
  ε_r = 4.3 block in an 80 mm air box drops from 1.43 M to 254 k cells
  at the same resolution inside the block.  Feature refinement,
  grading and the edge floor are unchanged; the new
  `MeshControl(wavelength_rule="global")` restores the previous rule.
  A dense background material now counts toward the wavelength as
  well (it was ignored before).
- `MeshControl(singularity_refinement=k)`: grading at the grid planes
  that hold a conductor edge — the edges of a strip, a patch, an
  iris, where the field is singular — starts at `h_fine / k` on both
  sides of the plane.  Edges are read from the CAD model (convex
  edges of metal bodies, and concave edges of vacuum bodies cut out
  of a metal background); cavity corners, fillet onsets and
  dielectric edges are not refined.  Off by default (`1`): the edge
  cell bounds the time step, so the factor trades bulk resolution
  for edge resolution rather than buying accuracy for free — see the
  meshing page for when it pays.

### Fixed

- An interval too short for the full ramp from the interface cell to
  the bulk cell no longer lets its integer cell count push the
  interface cell below the size the geometry asked for (up to 23 %
  at the default growth factor, which set the time step for the
  whole model).  The interface cell now stays at its requested size
  and the growth ratio relaxes instead; neighbouring cells still
  differ by at most `growth_factor`.
- An eigenmode solve with an explicit `sigma` that found fewer modes
  than requested because most of the Krylov vectors converged on the
  curl-curl null space now grows its request at the same shift
  (twice at most) instead of returning short; the factorisation is
  shared between the attempts.

### Changed

- The ports page and the `deembed` reference now say what a
  reference-plane shift on a quasi-TEM port can and cannot remove:
  the shift uses the port's quasi-static propagation constant, so
  the line's own frequency dispersion (about 22° over 16 mm of
  0.8 mm FR4 at 15 GHz) stays in the de-embedded phase — keep feeds
  short, or compare raw S-parameters between meshes.
- Meshes differ from 0.4.5: coarser far from a dielectric in models
  with more than one material (the slab rule above), and slightly
  coarser in short graded intervals (the fix above); cell counts and
  results quoted in the tutorials are updated accordingly.

## [0.4.5] - 2026-08-25

### Added

- The mesher now places grid planes on geometry edges: wherever an
  edge of a solid lies flat in an axis-normal plane — the onset of a
  chamfer or fillet, a loft section, the iris or equator circle of a
  revolved profile — the grid gets a plane, so the feature occupies a
  cell layer of its own and is seen by the cell's material average.
  Previously a chamfer smaller than half a cell had no effect at all
  and then switched on in one step.  Edge planes are floored by the
  new `MeshControl(max_edge_refinement=4.0)`: an edge whose cell would
  be finer than `h_max / 4` (or than `min_cell_size`) is dropped, and
  the mesher warns which feature is below the grid and which parameter
  resolves it.  `max_edge_refinement=0` restores the previous meshes.
- New how-to *Mesh convergence: a ladder around your simulation*: two
  code blocks to paste before and after your own mesh generation and
  analysis run the model on a ladder of mesh resolutions and report,
  per rung, the largest change of any complex S-parameter over the
  band, with a stop rule (below 0.02 on two consecutive rungs) and the
  same recipe for eigenfrequencies including an
  extrapolation to the infinitely fine mesh.  Demonstrated on a microstrip with a capacitive patch
  and on a pillbox cavity.

### Changed

- Meshes of models with chamfers, fillets, profile solids or other
  non-axis-aligned edges differ from 0.4.4 (more planes, usually a
  smaller time step near the feature); cell counts and results quoted
  in the tutorials are updated accordingly.

## [0.4.4] - 2026-08-25

### Added

- `model.plot()` is a new 3D viewer built on PyVista.  In a notebook
  it is a widget with an axis-aligned cutting plane driven from its
  toolbar (normal, position slider, flip, undo, reset) that opens every
  solid with capped cuts; with `mesh=` the cells the cut exposes are
  shown coloured by the material the mesher assigned.  Thin wires, ports (with their names),
  lumped elements, symmetry planes and the domain box are drawn and
  follow the cut; a toolbar menu hides or shows each object group;
  lengths are in millimetres, and the camera pans.  The same call opens a window in a
  script and renders a picture in a documentation build — the tutorials
  now show the 3D view.  Options: `cut`, `flip`, `show_ports`,
  `show_wires`, `show_grid`, `mode` (`"client"` in-browser rendering by
  default, `"server"`, `"trame"`, `"static"`, `"none"`), `scale_mm`,
  `camera`; the previous keywords keep their meaning.
- New documentation section **How-to guides**: task-oriented recipes
  meant to be downloaded and adapted.  First content: lumped-port
  termination guides — *Lumped ports: investigations* walks the
  measurement principle and the sensitivity sweeps (gap length,
  position, port impedance) for coax, microstrip and CPW, and one
  compact *Lumped port tuning* page per line type scores a candidate
  termination on your own cross-section, mesh resolution and
  frequency band (self-reflection, usable band, phase error).
- De-embedding: `result.deembed({"port1": d})` shifts port reference
  planes by a distance along the feed line and returns the S-matrix
  referenced there, without re-running.  The shift removes the exact
  propagation the grid applied — including the numerical-dispersion
  part an analytic `exp(-jβd)` would leave behind on coarse meshes —
  so de-embedding a uniform feed line cancels its phase down to the
  accuracy floor of the run itself.  Works on in-RAM and
  project-store results alike; the returned matrix answers the same
  `S`/`db`/`phase`/`plot_s` calls and Touchstone/scikit-rf exports.

### Fixed

- `model.plot()` no longer aborts the process on a shape without
  volume (an empty boolean result): the shape is skipped with a
  warning naming it.
- `GeometryModel.plot()`: the 3D view can now be panned (right-drag or
  shift + left-drag).  Panning had silently done nothing because the
  orthographic camera the viewer used only impersonated one and left
  the orbit controls with undefined frustum bounds.

### Changed

- `pyvista` is a new core dependency; the notebook 3D widget needs the
  new `magnelio[jupyter]` extra (`trame`, `trame-vtk`, `trame-vuetify`,
  `nest_asyncio2`).  pythreejs is no longer used.
- `model.plot()` returns `None` after displaying the view (it used to
  return the pythonocc renderer); `mode="none"` returns the
  `pyvista.Plotter` instead of displaying.
- The stripline pickup/kicker page moved from the tutorials (former
  tutorial 19) into the new How-to guides section; its content is
  unchanged, but the old `tutorials/plot_19_…` documentation URL no
  longer exists.

## [0.4.3] - 2026-08-24

### Fixed

- `geo.Loft(material="pec")` raised `AttributeError: 'str' object has
  no attribute 'visible'` later, in plotting or meshing, instead of
  resolving the built-in name like every other material argument —
  the `Loft` class constructor had been missed by the 0.4.2 name
  resolution.

## [0.4.2] - 2026-08-24

### Fixed

- `to_touchstone()` / `to_skrf()` no longer refuse to export when a
  port was solved for more modes than were excited.  The exported
  matrix is now the square sub-matrix over the excited channels, which
  is the network seen with the remaining channels matched — so
  exciting mode 0 on two 3-mode ports writes a `.s2p`, and exciting
  one port of a two-port writes its reflection as a `.s1p`.  An export
  that drops *propagating* higher modes at a port it does keep now
  warns, naming the port and the cut-off: such a file is not a
  complete model of the component.

### Added

- Built-in materials can be passed by name wherever a material is
  expected: `material="air"`, `background="pec"`, etc. resolve to
  the canonical `Material.air()` / `Material.vacuum()` /
  `Material.pec()` instances (case-insensitive).  Explicit `Material`
  objects work unchanged; an unknown name raises immediately, naming
  the recognised ones.
- The mesh now records the `f_max` it was generated for
  (`mesh.f_max`), and `AnalysisScatteringTD` defaults its band to it
  — `AnalysisScatteringTD(mesh=mesh)` no longer needs the `f_max`
  repeated.  Passing an explicit analysis `f_max` above the mesh's
  design frequency now warns: the grid undersamples the requested
  band.  Meshes built without a design frequency (`Mesh.from_grid`)
  keep requiring an explicit value.
- `to_touchstone()` checks the `.sNp` extension against the number of
  exported ports and fills a missing one in, so `to_touchstone("wr90")`
  writes `wr90.s2p`.  A mismatched extension is an error: Touchstone
  records the port count nowhere else, so the file would be unreadable.
- `channels=` on `to_touchstone()` / `to_skrf()` selects the exported
  sub-network explicitly, e.g. `channels=["port1", "port3"]` to cut a
  two-port out of a fully excited three-port.

## [0.4.1] - 2026-08-22

### Added

- Tutorial 19, *Striplines as pickups and kickers*: the beam
  instrumentation workflow on the public API — dimension a stripline
  pair with 2-D port solves, drive it as a kicker in its sum and
  difference modes through the symmetry plane between the strips, and
  obtain beam voltage, kicker constants, shunt impedances and (by
  Panofsky–Wenzel and reciprocity) the pickup transfer impedances and
  position sensitivity from one line monitor, checked against the
  ideal-stripline formulas.

## [0.4.0] - 2026-08-21

### Added

- Periodic structures in eigenmode analysis: a `"Periodic"` face pair
  on the mesh and `AnalysisEigenmode(phase_advance_deg=...)` solve the
  unit cell of an infinite periodic structure with a Bloch phase advance
  of 0…180 degrees per period — sweep it to trace a dispersion
  (Brillouin) diagram.  0 and 180 degrees stay real; in between the
  mode fields are complex and `result.plot` draws them as the real
  snapshot of maximum energy.  Previously a periodic face was solved as
  a magnetic wall without notice.
- `geo.Path.ellipse_to` / `geo.Curve.ellipse_arc`: elliptical arcs as
  profile segments — centre, the two semi-axes and the direction of the
  first, the sense fixed by `normal=` exactly as for circular arcs.
- CAD import: `magnelio.io.import_step` reads a STEP file — solids,
  their names, their display colours and the file's length unit, so a
  part drawn in millimetres arrives at its true size.  Materials are
  assigned on import by solid name, with wildcards and a literal-wins
  rule; a key matching no solid is an error, and an unmapped solid
  arrives as a construction body rather than silently as vacuum.
  Assemblies are flattened into a `Group` with every solid placed.
  `magnelio.io.import_brep` reads the geometry kernel's own format,
  which states no unit, so `unit=` is required.
- `magnelio.geo.ImportedSolid`: the shape type both importers and the
  project store return.  It is a full `Shape` — Boolean operators, the
  chainable verbs, `volume()` — and carries the name and display colour
  its source file gave it.
- Circuit board import: `magnelio.io.import_pcb` reads a fabrication
  export — Gerber copper and outline layers, Excellon drill files and
  the `.gbrjob` job file that records the stackup — and returns the
  board as solids.  Each copper and dielectric layer arrives at its real
  thickness and height, plated holes as copper barrels that join the
  layers their drill file declares (through, blind or buried), unplated
  holes and slots as cut-outs.  Materials are assigned by layer name as
  for CAD import; copper defaults to a perfect conductor and the
  substrate takes the permittivity the job file states.  Copper 35 µm
  thick does not have to be resolved by the grid — set
  `MeshControl(min_cell_size=...)` above the metal thickness and the
  mesher carries it below the cell.  A stated loss tangent is reported
  rather than modelled (it carries no reference frequency), and a
  dielectric with no stated permittivity arrives without a material
  instead of silently as vacuum.  Solder mask and silkscreen are
  ignored.  Anything the readers cannot turn into copper — step and
  repeat, negative images, thermal aperture macros — is refused with the
  file and line that asked for it, never dropped in silence.

### Fixed

- The eigenmode solver rejects CPML faces and unpaired `"Periodic"`
  faces instead of silently solving them as PMC.
- The automatic eigenmode shift escalates two steps beyond the
  empty-cavity estimate when a solve returns no mode, so shaped cells
  (irises, noses) whose fundamental sits well above the box estimate no
  longer come back empty.
- A cross-section drew solids with a hole as if they were solid: the
  bore of a tube, the gap of an annulus and the bore of a coaxial line
  were filled in the surrounding material's colour, hiding whatever sat
  inside them.  Which body won depended on the order shapes were added
  in, so the same model could look right or wrong.  Holes now stay open.
- Installing without pythonocc-core (the pip route) failed at the first
  mesh with a message about grid line arrays instead of naming the
  missing dependency: the mesher's per-shape guards, written to skip a
  shape the CAD kernel cannot handle, swallowed the import failure for
  every shape and left the grid empty.  The dependency error now
  reaches the caller, and says how to install it.

## [0.3.1] - 2026-08-19

### Fixed

- The released documentation carried a red banner announcing that it was
  *not* the released documentation, and the banner's "switch to stable
  version" button reloaded the page it was already on.  Whether a build
  warns is now decided by the channel it is published into, and the
  development channel names itself as such instead of borrowing the last
  release's version number.  Documentation only — the library is
  unchanged from 0.3.0.

## [0.3.0] - 2026-08-19

### Added

- Far-field computation: `monitors.MonitorFarField` records a closed
  Huygens surface during a run and returns antenna patterns —
  directivity, gain, realized gain, radiated power — at the requested
  frequencies.  Ground planes (PEC/PMC boundary faces) and symmetry
  planes are handled by image theory; the monitor streams to the
  project store and survives a resume.
- Far-field pattern plots: `plots.plot_pattern_cut` (polar cuts in
  the antenna convention) and `plots.plot_pattern_3d` (3D radiation
  surface), also available as `.plot_cut()` / `.plot_3d()` on the
  far-field result and monitor.
- `post.ntff_transform` and `post.FarFieldResult` for custom
  near-to-far-field pipelines.
- Lumped ports and elements on symmetry planes: declare the
  full-model device (endpoints, `Z0`, R/L/C values) and the solver
  derives the half model — a dipole fed across an electric symmetry
  plane now reports the full-model impedance instead of the monopole
  reading.  Invalid placements raise a clear error instead of
  silently snapping to the domain boundary.
- `Mesh.pml_cells`: public per-face absorber cell counts.
- Tutorial 15 (dipole as a symmetry half model, full-sphere far
  field); tutorial 08 now closes with the monopole's gain figures.

### Changed

- Plane-wave illumination no longer costs measurable runtime.  Every
  correction on the TF/SF box has the form
  `field += coefficient · waveform(t - delay)`, and only the time moves
  between steps, so the coefficients are folded once when the source is
  attached instead of being applied cell by cell at every step.  On a
  2.5 M-cell model with the box spanning the whole domain the march
  went from 246 to 32 ms per step on the CPU and from 2025 to 16 ms per
  step on the GPU; the box can now be sized for scattered-field
  accuracy rather than for speed.
- Geometry constructors and verbs now check their arguments where they
  are written instead of failing later in a plot, a mesh build or the
  CAD kernel.  A coordinate given as a single number, a point with two
  components, a negative radius, an axis that names no direction, a
  list passed where the shapes themselves belong, or a `chamfered()`
  without an edge selector each raise immediately, naming the argument
  and what it expects.  Coordinates accept NumPy arrays and are stored
  as plain float tuples.
- A thin wire lying entirely in the half-space removed by a symmetry
  declaration is skipped at mesh time (like a solid there) instead of
  failing to rasterise.

### Fixed

- Thin wires, discrete ports and lumped elements were missing from
  field plots: a field plane is sampled at cell centres, so those
  features sat half a cell off the drawn plane and fell outside its
  tolerance — a wire antenna's field picture showed an empty air box.
  Field and eigenmode plots now pass the thickness of the cell layer
  they display, and `plots.plot_cross_section` accepts it as `slab=`.

## [0.2.1] - 2026-08-17

### Fixed

- The near-field figure in the open-boundary tutorial showed a single
  flat block of colour instead of the radiation pattern.  Its colour
  scale was pinned to an absolute value picked before frequency-monitor
  data changed units in 0.2.0, so every point of the pattern sat above
  the top of the scale.  The scale now follows the pattern's own peak.
  Documentation only — the library is unchanged from 0.2.0

## [0.2.0] - 2026-08-16

### Added

- The method chapters cover symmetry planes: what declaring one does to
  the model, which fields each kind of plane preserves, how to pick
  between them, and how ports, power and losses are reported when the
  simulation runs on half the structure.  Tutorial 09 now uses a
  symmetry plane as standard practice.  The feature had been usable for
  several releases but appeared nowhere outside the API reference

### Changed

- The online documentation is published in two versions.  Opening the
  documentation now lands on the docs for the newest release; the
  development docs, built from the current state of the repository, are
  one entry away in the version menu in the navigation bar and mark
  themselves with a banner on every page.  Previously the site always
  showed the development state, so it could describe features that were
  not yet in any released version
- **Breaking:** `MonitorFieldFrequency.data` is now always in fields per
  1 W of incident CW power — E in V/m, H in A/m, per square root of a
  watt.  A run divides its own excitation spectrum out of the recorded
  transform, and so does a monitor read back from a project store, so
  the step no longer has to be remembered.  Where no excitation
  reference is available, `data` raises instead of quietly handing back
  the raw transform; `data_raw` returns that transform unchanged, in
  field units times seconds.  Scripts that read `data` without calling
  `renormalize` were off by the excitation spectrum — often by many
  orders of magnitude — and will now either be correct or say so.
  Calling `renormalize` yourself remains supported and cannot
  double-apply
- Frequency-monitor fields exported to ParaView carry the same units as
  the monitor in Python (fields per 1 W CW).  They were previously
  written as the raw transform, so a value read off the renderer and
  the same value read in a script disagreed
- Vector field plots (monitor slices and port-mode profiles) draw their
  arrows on an isotropic raster interpolated from the field instead of
  sampling the computational grid, so a locally refined mesh no longer
  shows up as clustered arrows.  `density` now counts arrows along the
  longer axis of the slice; the shorter one follows at equal spacing
- Port-mode plots of a port cut by a symmetry plane show the full
  cross-section — field and geometry overlay mirrored — instead of the
  simulated half, matching what the impedance and power figures already
  reported

### Fixed

- ParaView sessions show the full model across declared symmetry planes
  again.  Recent ParaView releases renamed the properties of their
  reflection filter, and the export fell back to the unmirrored data
  without saying so — leaving an unused mirror entry in the pipeline
  browser and a picture of half the model.  If a build exposes no
  usable reflection filter at all, the session now says so in the
  render view and the export raises a warning instead of quietly
  showing half
- Mirrored field data carries the correct sign.  Reflection alone
  continues a field the way it behaves across a magnetic wall, which is
  the opposite of what an electric wall requires, so on a model with
  planes of both kinds one mirrored half had every arrow reversed —
  visible as an even mode where the mode is odd.  Single components
  (`Ex`, `Hz_im`, …) were left uncorrected on every plane, and could
  disagree with the vector they belong to
- A solid could lose half its cross-section on a mesh plane without any
  warning at all, so parts of a conductor were meshed as if they were
  air.  Contour assembly followed the geometry kernel's own wire
  builder, which happily joins section curves into a branched shape
  that is no longer a single loop; everything past the branch was then
  silently skipped.  Cross-sections are now assembled directly from the
  section curves, and a junction is resolved by following the curve
  that continues smoothly.  This mostly shows up on bodies made by
  lofting, revolving or fusing, near the outer edge of a curved face —
  and it also fixes cutting planes that lie exactly in a flat face of a
  fused body, which previously came back short
- A curved solid could lose its entire cross-section on isolated mesh
  planes, warning about "open section chains" without saying which body
  or what it cost.  The plane in question grazes the solid's own curved
  face, and the mesher's recovery step — which re-takes such a plane
  slightly to one side — was too short to get clear of it on a fine
  mesh, so the material bookkeeping and the cell classification could
  end up disagreeing about where the conductor is.  The recovery step no
  longer shrinks with the mesh, and the two now always agree.  Affected
  regions lost their sub-cell resolution and fell back to a staircase
  approximation; conductor edges near the boundary of a lofted or
  revolved body regain it.  The warning, if it still appears, now names
  the body by its material, says how much was dropped and how far the
  recovery searched, and states that the bulk material distribution is
  unaffected — plus the remedy, which is to change the cell size near
  that plane
- Port-mode plots draw the outermost ring of arrows again.  It was
  blanked wherever the field component running *along* the window
  boundary vanished — which an electric wall forces it to do, on every
  port, everywhere along the frame — even though the component standing
  *perpendicular* to that wall is at its maximum there.  Boundary arrows
  now appear, standing perpendicular on the wall as they should.  Note
  that the arrow raster is independent of the mesh: to read the field in
  a thin gap, raise `density` rather than refining the mesh
- A waveguide port could quietly lose its exact, reflection-free
  termination and fall back to the first-order absorbing one when its
  cross-section sat on a mirrored copy of a curved solid — leaving a
  −30 dB-class reflection floor on that channel while a geometrically
  identical, unmirrored port on the same model had none.  Impedances and
  field results are unaffected; the reflection floor of the affected
  channel improves
- Material boundaries that run through a domain boundary face — a
  symmetry plane, a PEC or PMC wall, an absorbing face — are now
  resolved below the cell there, as they already were everywhere else.
  The outermost layer used to round every partially filled edge to
  fully dielectric or fully metal, which cost a half model up to a few
  parts per thousand against the full model it is meant to reproduce
  exactly.  A symmetric structure solved on half the mesh now returns
  the full model's answer to machine precision
- Waveguide ports on curved conductors no longer warn about a residual
  bias in their power scale — the bias is corrected instead.  Where a
  sub-cell conductor edge hands its share to a neighbour, the port's
  power bookkeeping now accounts for it; on a mixed conformal/staircase
  port pair the reciprocity residual drops from 0.013 dB to below
  0.00001 dB.  Absolute S-parameter magnitudes at ports cut by curved
  conductors change by a few parts in a thousand
- Port-mode plots now cover the whole port window.  They used to stop
  half a cell short on each side, which on a graded mesh could hide a
  tenth of the cross-section and put a seam through the middle of a
  mirrored full-model plot
- Port-mode plots draw the field itself instead of the solver's
  internal edge quantities.  On a graded mesh the two differ by the
  local cell size, which tilted the arrows and biased their length —
  most visibly at a curved conductor, where the profile read up to
  17 % low.  Both the E and the H picture are affected
- Arrows are no longer drawn inside conductors, and the interpolation
  no longer averages field values across a conductor boundary
- Saving a mirrored plot with `bbox_inches="tight"` produced a hugely
  oversized image that included geometry outside the plotted region

## [0.1.0] - 2026-08-14

First public release.

### Added

- FIT time-domain leapfrog solver on a structured non-uniform hexahedral
  grid, with conformal (sub-cell) material matrices and selectable
  float32/float64 precision
- NumPy (CPU) and CuPy (CUDA GPU) backends; `backend="auto"` picks the
  GPU when available, with CUDA-graph stepping
- Waveguide ports with exact discrete transparent boundaries:
  TEM / QTEM / TE / TM / hybrid modes, multi-mode, declared on the model
  before meshing; lumped (RLC-backed) ports
- Boundary conditions: PEC, PMC, CPML, periodic, and symmetry planes —
  declared once on the model, carried by the mesh
- Materials: isotropic and diagonal-anisotropic, pole-residue dispersion
  for ε(ω) and μ(ω) with built-in vector fitting, conductor losses
  (perturbative or SIBC wall model), surface roughness models
  (Hammerstad, Huray)
- Geometry: CSG primitives and Boolean operators (`a - b`, `a + b`,
  `a & b`), chainable transforms, profile-based construction (loft,
  sweep, revolve, shell), thin wires
- Field monitors (time and frequency domain, flux, wall loss),
  plane-wave source (TF/SF), 3D cavity eigenmode solver
- Project store on disk: streamed results, bit-exact resume,
  post-processing on stored data (HDF5); every run generates a
  ready-to-open ParaView session
- Touchstone (`.sNp`) export and `scikit-rf` adapter
- Sphinx documentation: tutorial series (14 executable tutorials),
  API reference, and method chapters with literature sources
