# `magnelio.io`

An open `Project` is both the store reader and a scattering result: it
satisfies {class}`~magnelio.analysis.ScatteringResult`, so `S`, `db`,
`phase`, `plot_s` and the Touchstone / scikit-rf exports work on it
exactly as on the in-RAM result.  Its `runs` mapping hands out one
`Run` per run — state, step count, clock, energy trace, result and
monitors — and `checkpoint_state` reads a resume checkpoint back as a
`CheckpointState` mapping; see *Projects and runs* in the technical
description for the vocabulary.

```{eval-rst}
.. automodule:: magnelio.io
   :members:
   :imported-members:
   :inherited-members:
```
