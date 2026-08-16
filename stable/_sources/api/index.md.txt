# API reference

The public API is organised along one axis — the domain:

- the **core** — the top-level `magnelio` namespace: the model
  container and run vocabulary (`GeometryModel`, `Material`,
  `Mesh`/`MeshControl`, `BoundaryConditions`), the problem classes
  (`Analysis*`) and the project-store entry points;
- the **domain namespaces** — `magnelio.geo`, `magnelio.ports`,
  `magnelio.monitors`, … — one per subject area.  Every public name
  has exactly one documented home.

Underscore modules are internal, with no stability guarantee.

```{toctree}
:maxdepth: 1

core
```

## Domain namespaces

```{toctree}
:maxdepth: 1

geo
materials
mesh
boundaries
ports
sources
monitors
circuit
signals
solver
analysis
post
plots
io
constants
```
