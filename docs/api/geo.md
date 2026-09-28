# `magnelio.geo`

Standalone geometry is organised by dimension: `Curve`, `Sheet` (including
planar `Profile` and curved `Surface` values), and `Solid` all derive from
`Shape`.  The named affine methods are shared by every category; immutable
`Transform` values provide reusable composition with `@`.

Boolean `+`, `-` and `&` are restricted to `Solid` values.  `Group` is a
material-preserving authoring collection rather than a `Shape`; its affine
transforms distribute over its members.

```{eval-rst}
.. automodule:: magnelio.geo
   :members:
   :imported-members:
```
