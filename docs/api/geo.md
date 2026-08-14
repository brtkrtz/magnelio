# `magnelio.geo`

Geometry is built from primitives (`Brick`, `Cylinder`, …) combined with
the Boolean operators `+`, `-` and `&`, then refined with chainable
verbs (`.translated()`, `.mirrored()`, `.filleted()`, …).

The operators and verbs are **not** listed on each primitive: they are
shared by every geometry object and documented once on
{class}`~magnelio.geo.Shape`, the base class all of them inherit from.
Start there when you are looking for what can be done *to* a shape;
the classes below describe what each shape *is*.

```{eval-rst}
.. automodule:: magnelio.geo
   :members:
   :imported-members:
```
