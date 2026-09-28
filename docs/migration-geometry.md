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
rotates its owner and detaches the placed annulus for a continuation.
