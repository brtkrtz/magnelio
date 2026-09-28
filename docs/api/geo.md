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

```{eval-rst}
.. automodule:: magnelio.geo
   :members:
   :imported-members:
```
