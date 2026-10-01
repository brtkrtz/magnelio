# `magnelio.geo`

Standalone geometry is organised by dimension: `Curve`, `Sheet` (including
planar `Profile` and curved `Surface` values), and `Solid` all derive from
`Shape`.  The named affine methods are shared by every category; immutable
`Transform` values provide reusable composition with `@`.

Boolean `+`, `-` and `&` are restricted to `Solid` values.  `Group` is a
material-preserving authoring collection rather than a `Shape`; its affine
transforms distribute over its members.

`translated()` and `rotated()` optionally build regular arrays through
`repeat`, include the original with `copy=True`, and aggregate through
`group=True` or (for Solid only) `unite=True`. `mirrored()` provides the same
copy and aggregation options without `repeat`. Default calls and
`Transform @ geometry` retain one placement. The
[placement guide](../methods/geometry.md)
defines counts, return types and material preservation.

`Curve.line`, `Curve.circle` and `Curve.ellipse` construct exact wires.
`Profile.polygon`, `rectangle`, `circle` and `from_wires` construct bounded
planar regions, including holes. `Curve.length` and `Profile.area` are
read-only measurements; `Profile.boundary()` returns detached boundary curves.
See [geometry construction](../methods/geometry.md) for orientation, validation,
and hole correspondence during lofting.

`Curve.offset(distance, normal=...)` gives directed planar offsets with round
outer joins and uncapped open ends. `Profile.offset(distance)` dilates or
erodes the material region, retaining all connected results. A curved
`Sheet.offset(distance, tolerance=...)` moves along its oriented normal and
retains the bounded rim. All three return independent results in tuples.
The [offset guide](../methods/geometry.md) defines signs, hole behavior,
collapse and the default sampled geometry tolerance; Tutorial 27 gives the
construction recipe.

Owned topology is separate from standalone geometry: `TopologyRef`,
`VertexRef`, `EdgeRef`, `FaceRef`, `EdgeSetRef` and `FaceSetRef` hold one
`Solid` owner. The Solid selectors `face`, `edge`, `vertex`, `faces` and
`edges` accept semantic constraints or registered names; `tag_face`,
`tag_edge`, `tag_vertex`, `tag_faces` and `tag_edges` register immutable names.
`TopologySelectionError`, `AmbiguousTopologyError` and `TopologyEvolutionError`
distinguish missing, tied and lost selections. The
{ref}`owned topology guide <geometry-owned-topology>`
explains measurements, detachment, history failures and project persistence.

Profile operations also accept suitable FaceRef values directly. `face.extruded`,
`revolved`, `swept`, `thickened` and `lofted` return independent Solid geometry,
retaining holes and inheriting the owner's material. `Loft` accepts the same
planar sections, as well as closed Curve outlines. Solid modifications accept
owned selections through `edges=` / `faces=` for fillet and chamfer and
`openings=` for shell. See the [operation guide](../methods/geometry.md) for
eligibility, sweep roll, owner identity and materialless construction solids.

`Path.from_pose(point, tangent, up)` and `Path.from_face(face_ref, up=...)`
seed immutable relative routes. Read `current`, `tangent` and `up`; append
`forward`, exact `turn_left` / `turn_right`, a spatial `turn_to`, or
`straight_to_plane`. Absolute steps remain available and update the pose.
The {ref}`routing guide <geometry-routed-paths>` defines left/right,
rotation-minimizing transport, plane offsets and underdetermined bends.
Path's up direction guides routing; it does not select a sweep frame.

`swept(frame=...)` selects corrected Frenet, Frenet, fixed world orientation
or fixed binormal transport. Fixed binormal requires `binormal=`; other modes
reject that argument. See the [sweep orientation guide](../methods/geometry.md)
for oblique sections, volume rules and curvature degeneracies, and Tutorial 22
for parallel-aperture transitions.

`swept(twist_deg=..., draft_deg=..., tolerance=...)` adds uniform total roll
and a constant signed section-offset angle. Positive draft grows the exterior
and shrinks holes. Twist acts about the transported section normal and preserves
the selected frame's section planes. See the
[twist and draft guide](../methods/geometry.md) for arc-length laws, fitting
accuracy, boundary closure and periodic routes, and Tutorial 23 for worked use.

`partition(cutter)` or `partition(normal=..., position=...)` returns independent
connected Solid or Sheet regions, inheriting the source material. `section`
with the same cutter grammar returns intersection Curves; an explicit planar
`filled=True` section of a Solid returns Profiles with holes. Empty cuts,
tangencies, coincident faces and named-topology splits have explicit rules in
the [partition and section guide](../methods/geometry.md). Tutorial 24 shows
a hollow component and an oblique cutter.

`Solid.imprint(cutter)` returns a receiver with faces split at the cutter's
intersection curves while preserving its volume and material. `insert` takes
physical Solids with explicit integer priorities and optional material-less
void tools, trims losing volumes and returns a material-preserving `Group`.
The [imprint and insert guide](../methods/geometry.md) explains directed face
selection, equal-priority conflicts and named topology. Tutorial 25 constructs
a housing with a selectable imprinted contact and a dielectric insert.

```{eval-rst}
.. automodule:: magnelio.geo
   :members:
   :imported-members:
   :inherited-members:
```
