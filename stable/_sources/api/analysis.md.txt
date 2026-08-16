# `magnelio.analysis`

`ScatteringTDResult` and the store-backed
{class}`~magnelio.io.project.Project` reader satisfy the same contract,
{class}`~magnelio.analysis.ScatteringResult` — a script works unchanged
against either. Four of its accessors (`phase`, `plot_s`,
`to_touchstone`, `to_skrf`) are shared verbatim between the two and are
listed on each class below through inheritance.

```{eval-rst}
.. automodule:: magnelio.analysis
   :members:
   :imported-members:
   :inherited-members:
```
