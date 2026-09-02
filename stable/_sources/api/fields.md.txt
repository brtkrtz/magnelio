# `magnelio.fields`

The field container every analysis hands out and every field source
takes in: the six Yee-staggered components together with the grid they
live on, in physical units.  It is the coupling channel between an
eigenmode result, a monitor snapshot and the initial or incident field
of a transient run.

`SurfaceRecording` is the second coupling object: the tangential
fields on a closed box over time, written by `MonitorFieldSurface` and
replayed by `SourceFieldSurface`, with `save`/`load` as the exchange
format between two runs.

```{eval-rst}
.. automodule:: magnelio.fields
   :members:
   :imported-members:
```
