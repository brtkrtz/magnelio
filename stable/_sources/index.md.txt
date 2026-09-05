# Magnelio

Magnelio is a Python library for full-wave 3D electromagnetic field
simulation. Its standard workflow is broadband S-parameter extraction
over waveguide ports on arbitrary 3D geometry; the workhorse solver is
a time-domain Finite Integration Technique (FIT-TD) engine.

```{image} _static/hero_dielectric_filter.png
:alt: Electric field vectors in a two-pole dielectric resonator filter, where two ceramic pucks stand in a metal housing, separated by a wall with a coupling window, with a probe pin at each end.
```

This documentation has five pillars:

1. **Tutorials.** Executable walkthroughs of the public API, ordered
   from the first simulation to advanced workflows. Every page is
   generated from a runnable script and can be downloaded as a Python
   script or a Jupyter notebook.

2. **How-to guides.** Task-oriented recipes: each page solves one
   concrete engineering problem end to end and is meant to be
   downloaded and adapted to your own device.

3. **API reference.** The public surface — the core namespace and
   the domain namespaces — generated from the docstrings.

4. **Technical description.** Background information, mostly as a developer reference.
   Every numerical method used in Magnelio is described together with
   the published scientific work it is based on. Each chapter also explains how the method is
   realised in the code base (module paths, discrete formulas,
   conventions), at the level of detail needed to audit the
   implementation or to adapt a new method into the same framework.

5. **Bibliography.** The works cited throughout the documentation.

Questions, ideas and feedback of any size go to
[GitHub Discussions](https://github.com/brtkrtz/magnelio/discussions); wrong results, crashes and refused valid
input to the [issue tracker](https://github.com/brtkrtz/magnelio/issues).

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: How-to guides

howto/index
```

```{toctree}
:maxdepth: 3
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Technical description

methods/index
```

```{toctree}
:maxdepth: 1
:caption: Upgrading

migration-0.6
migration-0.5
```

```{toctree}
:maxdepth: 1
:caption: Bibliography

bibliography
```
