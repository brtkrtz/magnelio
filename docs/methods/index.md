# Numerical methods

The chapters below inventory every numerical method currently built
into Magnelio, in the order a simulation passes through them: spatial
discretisation, mesh generation and conformal geometry, boundary
conditions, port models, lumped circuit elements, dispersive
materials, conductor losses, sources and monitors, far-field
computation, and the eigenmode solver.  A final chapter covers implementation-level engineering
(backends, precision) that is not itself a research method.

Every method that originates in published research carries a citation;
in-house derivations are marked as such in the text.

```{toctree}
:maxdepth: 2

fit-discretization
meshing-conformal
boundaries
ports
lumped-elements
dispersive-materials
conductor-losses
sources-monitors
far-field
eigenmode-analysis
implementation
```
