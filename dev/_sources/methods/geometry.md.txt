# Geometry construction

Magnelio models are built from constructive solid geometry on the Open
CASCADE kernel via `pythonocc-core` (DD-003, DD-016) — primitives,
Boolean operations, and a set of *verbs* that grow, move and modify
shapes.  The construction layer is engineering infrastructure on top
of a third-party kernel, not a numerical-methods contribution.  This chapter
covers the vocabulary of that construction: which objects are
*profiles* and which are *bodies*, how a curve or a surface becomes a
solid, and what the mesher makes of the result.  The API reference
lists every class and verb; the tutorials on profile geometry, CAD
import and the reflector antenna show them in use.

## Bodies, sheets and curves

Three kinds of object share the geometry namespace:

- **Bodies** — `Brick`, `Sphere`, `Cylinder`, `Cone`, `Torus`, `Loft`,
  imported solids, and everything a verb or a Boolean produces from
  them.  A body carries a material and is what a `GeometryModel`
  meshes.
- **Sheets** — zero-thickness regions: the planar `Face` (an
  axis-normal polygon), a `Curve.covered()` (any closed planar curve
  filled in), and the curved `Surface`.  A sheet without a material is
  a *construction profile*: it exists to be grown into a body by
  `extruded()` or `thickened()` — and, for the planar ones, `revolved()`
  or `swept()`, or as a section of a `Loft`.  A sheet with a material
  would be a *thin sheet*; its physics (an infinitely thin conductor or
  dielectric film) is not wired, so such a sheet cannot be meshed on its
  own — model it as a thin body instead.
- **Curves** — `Curve` (polyline, arc, spline, helix) and `Path`, which
  draws one segment by segment.  A closed planar curve becomes a sheet
  through `covered()`; any curve becomes a conductor track through
  `traced()` (widened in its plane, then given a metallisation
  thickness — the direct route from a routed centreline to the copper
  of a board); a `ThinWire` is a curve meshed as a sub-cell conductor.

Moving, turning, scaling and mirroring keep these kinds: a rotated
sheet is still a sheet and still a profile, a mirrored planar sheet is
still planar.  Booleans are defined on bodies.

A union of bodies that are prisms along one axis over the same
interval — the strips of a feed network, the pads of a layer, a row of
posts — is fused in their common plane and raised once, so the result
carries no seams between its operands; whatever else a union holds is
fused in space, and only where it meets something.  In the plane the
operands are fused pairwise up a spatial bisection tree, with the
seams removed at every node, so a network of thousands of coplanar
strips costs seconds rather than the minutes a single fuse of all of
them takes.  The point set is the same either way; the face count is
what the mesher sees.

## Lofts: between profiles, and between faces

Two constructors build a body that changes cross-section along its
length.  `Loft(*sections)` takes the profiles themselves — planar
sheets or closed curves, as many as the shape needs, in the order the
body passes through them — and is the way to draw a horn or a
multi-step matching section from sketches.  `a.lofted(near_a, b,
near_b)` takes one face of an existing body and one face of another,
and bridges them; the profiles are read off the two faces, so the
transition fits both parts exactly and follows them when a dimension
changes.

Both accept `blend="spline"` (one smooth surface through all profiles)
and `blend="ruled"` (straight surfaces between neighbours, a stack of
frusta).  With only two profiles the two are the same surface: a
straight run from one outline to the other, which meets each end at
whatever angle the straight connection makes — a crease at both joints
of a waveguide taper.

The face-to-face verb adds `blend="tangent"`, which leaves each face
along its outward normal, so the wall slope at both joints is zero and
the transition meets both parts without a crease.  It has two regimes,
chosen from the two normals:

- **Faces that look at each other** (antiparallel normals: the two ends
  of a taper, coaxial or laterally offset) get a loft whose
  cross-section eases out of one profile and into the other along a
  straight axis — the same family of intermediate sections the plain
  loft carries, redistributed under a law whose derivative vanishes at
  both ends.  The end tangency is exact by construction, not fitted, and
  the axial position stays linear in the surface parameter at the
  default `tension=1/3`.  A lateral offset between the two faces comes
  out as a smooth dog-leg with the sections still parallel to the faces.
- **Faces that point in different directions** (an electrode ending on
  a *z*-face, the pin it feeds beginning on a *y*-face) get a sweep of
  one profile into the other along a curved spine that leaves both faces
  along their normals, with the profiles held perpendicular to the path.

`tension` sets how far the blend holds its normal direction before
turning, as a fraction of the distance between the faces; a `(start,
end)` pair sets each end on its own.  Values well past `2/3` overshoot
into a bulge.  Two parallel faces that look *away* from each other are
refused: a transition leaving both along their normals would have to
pass through both bodies.

## From a map to a reflector: parametric surfaces

`Surface.parametric(fn, u=(u0, u1), v=(v0, v1), samples=(nu, nv))`
samples a map $(u, v) \mapsto (x, y, z)$ on a grid and passes a
degree-3 B-spline surface exactly through the samples (OpenCASCADE's
`GeomAPI_PointsToBSplineSurface`).  The map is any Python function of
two parameters — a paraboloid $z = (x^2 + y^2)/4F$, a hyperboloid, a
numerically shaped reflector given as a table — and the parameter
domain is the designer's choice: a reflector rim comes out as an exact
circle when the dish is parametrised in polar coordinates about the
aperture centre, with no trimming step.  A parameter row that collapses
onto a single point (the pole of such a parametrisation) is allowed;
the surface closes there.

The interpolant is exact at the samples and follows the map to within
the spacing-cubed between them: 32 × 32 samples place a 240 mm dish to
a few micrometres, 32 × 64 to 10 nm.  The sheet stores its samples,
not the map — a shape is a value, and, as for imported CAD, the
parametric history is not part of a model: a stored project returns
the extruded body, not the function that generated it.

Two verbs turn the sheet into metal:

- `extruded(vector=…)` sweeps the sheet along a fixed vector
  (a prism).  It is robust for any sheet and, for a perfect conductor,
  physically equivalent to a normal offset — the field never enters
  the metal, so only the reflecting surface matters.  This is the
  recommended route for reflectors.
- `thickened(thickness=…)` offsets a curved sheet along its own
  normal (`direction="forward"` or `"backward"`; `"symmetric"` is
  for planar sheets).  The kernel's offset can fold at very dense
  sample grids or where the thickness approaches the curvature radius;
  Magnelio checks the result (topology and volume against
  area × thickness) and refuses with a pointer to `extruded()` instead
  of returning a body of the wrong shape.

## What the mesher sees

The mesher places grid planes where the geometry has features — the
faces of bricks, the tangent planes of cylinders and spheres, the
edges that lie flat in an axis plane (see the chapter on conformal
meshing).  A free-form B-spline face contributes only the six planes of
its bounding box: the mesher has no analytic handle on it, so the
resolution *across* a reflector is whatever the wavelength rule and
`MeshControl(max_cell_size=…)` give.  Set the cell size explicitly for
such models.  Cross-sections through free-form faces are taken on a
triangulation of the body whose points are lifted back onto the exact
surface (see the conformal-meshing chapter), so a free-form body
meshes at about the cost of the same volume of primitives.

The thin-metallisation detection recognises a flat sheet whose
bounding box is thinner than a cell on one axis; a curved shell is
thick on every axis of its bounding box and is classified cell by cell
like any other body.  Give reflector shells a thickness of two cells or
more so that the conformal classifier resolves the metal on both faces
— for a perfect conductor the thickness has no electromagnetic effect.
