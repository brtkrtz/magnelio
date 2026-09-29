# Upgrading geometry construction

The development geometry API replaces standalone `Face` polygons and
`Curve.covered()` with `Profile` factories. This change will require a minor
release; it is not part of the published 0.8 releases. All coordinates in a
polygon are now three-dimensional world coordinates, and profiles can carry
holes directly.

| Previous construction | Current construction |
| --- | --- |
| `geo.Face(normal="z", points=uv, position=z)` | `geo.Profile.polygon([(u, v, z) for u, v in uv])` |
| `geo.Face(normal="y", points=uv, position=y)` | `geo.Profile.polygon([(u, y, v) for u, v in uv])` |
| `geo.Face(normal="x", points=uv, position=x)` | `geo.Profile.polygon([(x, u, v) for u, v in uv])` |
| `curve.covered(material=..., name=...)` | `geo.Profile.from_wires(curve, material=..., name=...)` |
| Several arcs approximating a full circle | `geo.Curve.circle(center, radius, normal=...)` |

Neither `Face` nor `covered()` remains an alias. Use `Profile.rectangle` when
the intended shape is a rectangle, and `Profile.circle` for a disc. Curve
construction through `polyline`, `arc`, `ellipse_arc`, `spline`, `helix`,
`joined`, and absolute `Path` steps remains available.

The `repeat`, `copy`, `unite` and `group` options of `translated()` and
`rotated()`, and the copy/aggregation options of `mirrored()`, remain available.
`repeat` counts transformed copies; `copy=True` includes the original first.
Fusion through `unite=True` accepts Solid geometry only; use `group=True`
for curves, profiles or material-preserving assemblies. An explicitly requested
Group or Union is returned even for one translated or rotated copy.

For an old rectangle in the plane `y = 2 mm`, with `(u, v)` meaning `(x, z)`,
the world-coordinate replacement is:

```python
from magnelio import geo

uv = [(0, 0), (4e-3, 0), (4e-3, 3e-3), (0, 3e-3)]
profile = geo.Profile.polygon([(u, 2e-3, v) for u, v in uv])
body = profile.extruded((0, 5e-3, 0), material="pec")
```

A previously covered closed outline becomes:

```python
from magnelio import geo

outline = (
    geo.Path((0, 0, 0))
    .line_to((4e-3, 0, 0))
    .line_to((4e-3, 3e-3, 0))
    .line_to((0, 3e-3, 0))
    .closed()
)
profile = geo.Profile.from_wires(outline, material="pec")
body = profile.extruded((0, 0, 5e-3))
```

Profile factories now check planarity and boundary validity immediately.
Handle invalid input at construction rather than waiting for the first mesh
or bounding-box query. `profile.boundary()` returns standalone curves, outer
first and then holes in the supplied order. `Loft` preserves those holes and
requires matching hole counts and correspondence in every section. Its material
now defaults to the first section's material, with an explicit `material=`
overriding inheritance.

The [geometry construction guide](methods/geometry.md) explains exact curves,
arbitrary orientation, holes and placement; Tutorial 14 constructs a hollow
elbow and compares its volume with profile area times spine length.

## Owned topology

`Solid.face(near=..., normal=..., surface_type=...)`, `edge(...)` and
`vertex(...)` now return owner-bound references. Keep enduring selections by
calling `tag_face`, `tag_edge` or `tag_vertex` before placement, then retrieve
the name from the placed owner. `faces`, `edges`, `tag_faces` and `tag_edges`
are deliberate set operations. Numeric kernel indices are not supported.

References have no independent transform methods. Detach a face with
`face.detached()` or an edge with `edge.as_curve()` when standalone geometry
is required. The existing solid modification signatures remain available.

Ambiguous semantic picks raise `AmbiguousTopologyError`; refine the selector
instead of relying on kernel order. Names follow affine placements exactly;
construction that loses or splits a singular name raises
`TopologyEvolutionError`. Name a set before an intentional split. Project
geometry now retains names by replaying their semantic origins and construction
histories; older projects without this metadata continue to load.

[Tutorial 21](tutorials/plot_21_topology_selection.rst) names a coax end face,
rotates its owner and sweeps the placed annulus directly for a continuation.

## Uniform operations and referenced modifications

Use the selected FaceRef itself for extrusion, revolution, sweep, thickness
and loft. Detachment is needed only for independent placement. Replace loose
point lofts with `start_face.lofted(end_face)`; `geo.Loft` can mix profiles,
planar sheets and face references. Holes survive all forms, including tangent
transitions. The start section supplies an omitted material.

| Previous form | Reference form |
| --- | --- |
| `body.extruded(vector, face_near=p)` | `body.face(near=p).extruded(vector)` |
| `a.lofted(p, b, q)` | `a.face(near=p).lofted(b.face(near=q))` |
| `body.filleted(face_near=p, radius=r)` | `body.filleted(faces=body.face(near=p), radius=r)` |
| `body.chamfered(near=p, distance=d)` | `body.chamfered(edges=body.edge(near=p), distance=d)` |
| `body.shelled(t, opening_face_near=p)` | `body.shelled(t, openings=body.face(near=p))` |

The point forms remain available and now reject ambiguous picks immediately.
Refs must belong to the exact receiver of an owner modification. After
transforming an owner, retrieve its refs again. A sweep transports the actual
profile by the shortest normal-to-tangent rotation, preserving its in-plane
roll; an already aligned section is kept as placed. This can change the roll
of asymmetric sections previously aligned through canonical world frames.
Thickening forward now follows the section's oriented normal rather than a
canonical positive world component.

Materialless profile operations now consistently produce construction solids
for Boolean use. Such results still require a material before model assembly.

```python
from magnelio import geo

body = geo.Brick(size=(4e-3, 3e-3, 2e-3), material="pec")
cap = body.face(normal="z")
extension = cap.extruded((0, 0, 3e-3))
housing = body.shelled(0.2e-3, openings=cap)
```

## Relative routes

Existing absolute `Path(start).line_to(...).arc_to(...).ellipse_to(...).spline_to(...)`
calls remain available. To continue from a planar owned face, replace manual
centroid/normal arithmetic with `Path.from_face(face, up=...)`. To route from
a point, use `Path.from_pose(point, tangent, up)`. Up must not be parallel to
tangent; it determines left/right and is carried without added tangent twist.

```python
from magnelio import geo

body = geo.Cylinder(axis="x", radius=2e-3, inner_radius=1e-3,
                    height=6e-3, material="pec").tag_face("port", normal="x")
body = body.rotated("z", 22.5)
face = body.face("port")
spine = (geo.Path.from_face(face, up="z")
         .turn_right(radius=8e-3, angle_deg=22.5)
         .straight_to_plane("x", 20e-3).curve())
extension = face.swept(spine)
```

Absolute segments now resolve their CAD endpoint tangent when appended.
Degenerate segments with no tangent therefore fail at that step. Relative
turns require a posed start; an absolute-only path acquires tangent but no
implicit up direction. `turn_to` rejects an opposite direction because it
does not determine a bend plane. The [routing guide](methods/geometry.md)
defines spatial transport and the distinction between routing pose and pipe
orientation. Tutorial 21 executes the oblique coax bend to the domain plane.
