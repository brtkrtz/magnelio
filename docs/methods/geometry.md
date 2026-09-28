# Geometry construction

Magnelio models are built from constructive solid geometry on the Open
CASCADE kernel via `pythonocc-core` — primitives,
Boolean operations, and a set of *verbs* that grow, move and modify
shapes.  The construction layer is engineering infrastructure on top
of a third-party kernel, not a numerical-methods contribution.  This chapter
covers the vocabulary of that construction: which objects are
*profiles* and which are *bodies*, how a curve or a surface becomes a
solid, and what the mesher makes of the result.  The API reference
lists every class and verb; the tutorials on profile geometry, CAD
import and the reflector antenna show them in use.

## Shapes have a dimension

Every standalone geometry value is a `Shape`, with one explicit dimensional
category:

- **Solids** — `Solid` values such as `Brick`, `Sphere`, `Cylinder`, `Cone`, `Torus`, `Loft`,
  imported solids, and everything a verb or a Boolean produces from
  them.  A solid may carry a material and is what a `GeometryModel`
  meshes.
- **Sheets** — `Sheet` values are zero-thickness regions: the planar
  `Profile` factories bound a planar region with an outer wire and optional
  holes, while `Surface` is the curved-sheet category.  A sheet without a material is
  a *construction profile*: it exists to be grown into a body by
  `extruded()` or `thickened()` — and, for the planar ones, `revolved()`
  or `swept()`, or as a section of a `Loft`.  A sheet with a material
  would be a *thin sheet*; its physics (an infinitely thin conductor or
  dielectric film) is not wired, so such a sheet cannot be meshed on its
  own — model it as a thin body instead.
- **Curves** — `Curve` values (line, circle, ellipse, polyline, arc, spline, helix) are
  one-dimensional standalone shapes.  `Path` is a builder which
  draws one segment by segment.  A closed planar curve becomes a sheet
  through `Profile.from_wires(curve)`; any curve becomes a conductor track through
  `traced()` (widened in its plane, then given a metallisation
  thickness — the direct route from a routed centreline to the copper
  of a board); a `ThinWire` is a curve meshed as a sub-cell conductor.

Moving, turning, scaling and mirroring preserve the category, material and
name: a rotated curve remains a curve, a rotated profile remains a profile,
and a mirrored solid remains a solid.  `Group` is deliberately outside the
`Shape` hierarchy.  It is a material-preserving authoring collection whose
transforms apply member by member.  Booleans accept solids only; passing a
curve, sheet, profile or group raises a category-specific `TypeError` before
the CAD kernel is called.

## Exact curves and planar profiles

`Curve.line(start, end)` builds an exact straight segment. `Curve.circle`
and `Curve.ellipse` build closed analytic wires, retaining their exact
conic geometry through rotations and mirrors. They do not approximate the
boundary with a polygon. Circles take a world centre, radius and plane
normal. Ellipses take a world centre, two semi-axis lengths, a normal and
`major_axis`: the direction of the first semi-axis, projected into the
plane. Either semi-axis may be larger. `curve.length` reads the CAD length.
Arcs, elliptical arcs, splines, helices and `joined()` remain available;
`Path` builds mixed boundaries segment by segment.

A wire describes a boundary; a `Profile` describes the enclosed region.
Use `Profile.polygon(points)` for coplanar 3-D vertices (closed automatically),
`Profile.rectangle(center, size, normal=..., x_direction=...)` for an
oriented rectangle, `Profile.circle(center, radius, normal=...)` for a disc,
and `Profile.from_wires(outer, holes=...)` for any closed planar boundary.
`size` gives rectangle width and height. `x_direction` fixes its width direction
in the plane; height follows `normal x x_direction`. Without it, the least
parallel world axis supplies the projected width direction. There is no
ambient working coordinate system.

For example, an annular cross-section extrudes into a tube with its bore
already present:

```python
from magnelio import geo

outer = geo.Curve.circle((0, 0, 0), 2e-3)
inner = geo.Curve.circle((0, 0, 0), 1e-3)
ring = geo.Profile.from_wires(outer, holes=[inner], material="pec")
tube = ring.extruded((0, 0, 10e-3))
```

The factories validate closure, planarity, self-intersection and hole placement
before returning. Holes must lie strictly inside the outer wire, cannot touch
or intersect each other or the outer wire, and cannot nest. Winding is corrected
automatically. Invalid boundaries raise `ValueError`; inputs from the wrong
category raise `TypeError`. A profile can be placed in any plane by explicit
world points, normals and directions, or by the common affine methods.

`profile.area` measures the region with its holes removed. `profile.boundary()`
returns independent `Curve` values: the outer boundary first, then holes in
construction order. The curves have the profile's world placement and can be
transformed or used to construct another profile. This extraction is a
standalone value operation.

Extrusion, revolution and sweep carry all inner boundaries into the solid.
`Loft` carries holes by matching boundaries in the supplied order, so every
section must have the same number of holes and the same intended correspondence.
It cannot create, close, or merge holes partway through a transition. As for
any loft or sweep, valid sections alone cannot guarantee that the resulting
three-dimensional body avoids crossing itself. A material supplied to an
operation overrides the profile material; otherwise the material is inherited
(`Loft` uses its first section). Construction profiles need an explicit
material for extrusion, revolution and sweep.

(geometry-owned-topology)=
## Owned topology and named selections

A body face is a `FaceRef`, an edge is an `EdgeRef`, and a vertex is a
`VertexRef`. These are read-only views owned by one immutable `Solid`.
They carry their owner; they are not standalone shapes and cannot be moved,
added to a model, or combined with Boolean operators. Use `face.detached()`
to obtain a standalone `Profile` for a planar face or `Surface` for a curved
one, and `edge.as_curve()` for a standalone curve. Detachment keeps world
placement, face holes and the owner's material.

Select by physical meaning rather than kernel indices. `body.face(near=p)`
minimises distance to the complete trimmed face, including its boundaries;
it does not compare centroids. `normal=` compares oriented outward normals,
and `surface_type=` filters plane, cylinder, cone, sphere, torus, bspline or
other surfaces. Without `near`, a normal constraint selects planar faces;
with `near`, a curved face is tested at the closest point. Edges accept
`curve_type=` (line, circle, ellipse, hyperbola, parabola, bezier, bspline,
other); vertices accept `near=`. Filters apply before the distance comparison.

A singular selection needs at least one constraint. No match raises
`TopologySelectionError`; equally eligible candidates within the model's
converted CAD tolerance raise `AmbiguousTopologyError`. A point on a shared
edge can select two faces equally well: add a normal or surface-type
constraint. `body.faces(...)` and `body.edges(...)` deliberately return iterable
`FaceSetRef` and `EdgeSetRef` values. Without constraints they select all faces
or edges; with `near` they retain all nearest ties.

Register a selection before placement when it has an enduring physical role:

```python
from magnelio import geo

coax = geo.Cylinder(radius=2e-3, inner_radius=1e-3, height=5e-3, material="pec")
coax = coax.tag_face("port", near=(1.5e-3, 0, 5e-3), normal="z")
coax = coax.rotated("y", 22.5)
port = coax.face("port")
profile = port.detached()
extension = profile.extruded(tuple(1e-3 * n for n in port.normal))
```

Names are non-empty strings, unique per topology kind on their owner.
`tag_face`, `tag_edge`, `tag_vertex`, `tag_faces` and `tag_edges` return a new
owner; the input stays unchanged. Named lookup cannot be combined with
semantic constraints. Retrieve a singular tag with the singular selector
and a set tag with the plural selector.

Intrinsic measurements are read-only properties: `vertex.point`;
`edge.length`, `start`, `end`, `vertices`; and `face.centroid`, `area`,
`is_planar`, `edges`, `vertices`. Positions and lengths are in metres and areas
in square metres. Edge endpoints follow the edge's orientation on its owner;
reflection may reverse their traversal order. Boundary connectivity includes
inner wires and deduplicates
shared vertices and seam edges. A planar face has a constant outward `normal`.
Planarity is measured geometrically: a flat B-spline patch is planar even
when its analytic `surface_type` remains `bspline`. Detachment covers its exact
boundaries with a plane, giving a usable Profile.
For a curved face use `normal_at(point)` with a point on the trimmed face;
reading its `.normal` raises `ValueError`.

### How names follow construction

Translation, rotation, reflection and uniform scaling preserve names exactly,
including their new placement and outward orientation. An unnamed reference
continues to describe its original owner; transform the tagged owner and look
up the name there. Group placements keep each member's names independently.

Boolean and modification operations use the CAD kernel's construction history.
A singular name survives only with one provable successor. Deleted selections,
unprovable successors and singular names that split raise
`TopologyEvolutionError` during the construction call. The implementation never
falls back to the nearest face. When a split is intentional, register a set
before performing it:

```python
block = geo.Brick(size=(2e-3, 2e-3, 2e-3), material="pec")
block = block.tag_faces("cap", normal="z")
slot = geo.Brick(origin=(0.9e-3, -1e-3, -1e-3), size=(0.2e-3, 4e-3, 4e-3))
split = block - slot
assert len(split.faces("cap")) == 2
```

A set can split or merge, but deleting any selected member still raises.
Different operands carrying the same topology name must prove the same selection;
otherwise their Boolean combination reports a name conflict. In particular,
fusing tagged array copies can conflict; group them when each copy needs its
own selection names. Names should be registered after fusion when they describe
the fused body.

History availability depends on the operation and kernel. A shell can map an
outer face to both an outer and an inner face; use a deliberate set when that
is intended. A face-to-face loft built from boundary wires may provide no
provable face successor, so retaining that singular tag fails explicitly.
Detaching a face creates independent geometry and intentionally leaves the
owner's names behind. Existing profile construction verbs and loose-point
solid modification verbs retain their signatures.

Projects persist the semantic origin of each name and the relevant construction
path, with exact geometry snapshots at untagged origins. Read-back replays that
path and validates the registered and final cardinalities before exposing names.
It stores no subshape enumeration indices. Topology enumeration is cached per
owner and model scale; repeated named lookup resolves from the owner's cache.

## Placement and transform composition

The named methods are the normal spelling for one-off placement:

```python
feed = feed.rotated("z", 30.0).translated((12e-3, 0.0, 0.0))
route = route.mirrored("y")
```

Without additional options each call returns one new value of the same
dimensional category. For regular arrays, `translated()` and `rotated()`
accept `repeat`, `copy`, `unite`, and `group`. `repeat` counts transformed
copies, starting at one displacement or angle increment; `copy=True` adds
the untransformed original first. Thus seven copies plus the original make
an eight-element circular array:

```python
post = geo.Cylinder(radius=0.5e-3, height=5e-3, origin=(6e-3, 0, 0), material="pec")
posts = post.rotated("z", 45, repeat=7, copy=True, unite=True)
via = geo.Cylinder(radius=0.2e-3, height=1e-3, material="pec")
fence = via.translated((1e-3, 0, 0), repeat=7, copy=True, group=True)
pair = post.mirrored("x", copy=True, group=True)
```

Translation copy *i* is displaced by `i * vector`; rotation copy *i* turns
through `i * angle_deg` about the same axis and origin. Each placement acts
on the original geometry, preserving the source and its existing placement.
`repeat` must be a positive whole number. `mirrored()` accepts `copy`, `unite`
and `group`, but has no repetition count: two reflections in the same plane
would return to the original. Mirrored aggregation requires `copy=True`.
`scaled()` retains its single-placement signature.

| Options | Result |
| --- | --- |
| Default `repeat=1, copy=False` | One placed geometry value |
| Multiple copies or `copy=True`, without aggregation | A list in placement order, original first when included |
| `group=True` | A `Group`, preserving member materials and dimensional categories |
| `unite=True` | A `Union`, carrying one material inherited from the source solid |

`group` and `unite` are mutually exclusive. An explicitly requested Group or
Union is returned even for a single translated or rotated copy. Fusion is
valid only for `Solid`: curves, profiles and curved sheets may be repeated
and grouped, but cannot become Solid Boolean operands.

The same convenience options apply to a `Group` assembly. Each copy retains
its nested members and materials; `group=True` bundles these assemblies into
an enclosing Group. `unite=True` rejects a Group rather than collapsing its
potentially different materials. Lists, explicit `Group`/`Union` constructors
and Python comprehensions remain available for irregular placement patterns.

For a placement that is reused, build an immutable affine value.  `@` acts on
column-vector points, so the rightmost operation happens first:

```python
placement = geo.Translation((12e-3, 0.0, 0.0)) @ geo.Rotation("z", 30.0)
placed_curve = placement @ route
placed_profile = placement @ cross_section
placed_solid = placement @ housing
```

This is equivalent to rotating each value and then translating it.  `Mirror`
reflects across `point · normal == position`; `Scale` is uniform about its
centre. `Transform @ geometry` always produces one placement and has no
array or aggregation options. Geometry does not right-apply a transform, and `+ vector` is not a
translation: `+`, `-` and `&` remain solid Boolean operators.

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
