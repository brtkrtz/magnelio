# Upgrading to 0.9: API naming

Magnelio 0.9 makes physical quantities and operation scope explicit.
Update Python calls together; removed spellings have no compatibility aliases.
It does not change numerical methods, phasor conventions or the interpretation
of supported stored projects. For the geometry construction changes in the
same release, see [Upgrading geometry construction](migration-geometry.md).

## Complete mapping

| Previous spelling or meaning | Current spelling and meaning |
|---|---|
| `Mode.gamma(omega)`, `z_wave(omega)`, `z_modal(omega)` in rad/s | `gamma(f=...)`, `z_wave(f=...)`, `z_modal(f=...)` in Hz; keyword-only on `Mode`. `ModeReport` also evaluates Hz. |
| `SurfaceCurrent.current_through()` | Removed without replacement: its area integral had units A·m, not A. Read `vector()`, `magnitude()`, `select()` and independent `power_loss()`. |
| Field monitor or `TDResult.renormalize(...)` | `normalize_to_excitation(...)`; state is `is_normalized_to_excitation`. Network `renormalize(z_ref)` retains its different meaning and returns a new result. |
| Rectangular `PortAnalytical(..., center=...)` and `PortSpecRectWG(..., center=...)` | `origin=...`, the minimum tangential corner. Coax ports retain `center=...`. Wrong-family anchors are rejected. |
| `ComponentRecord.at(..., normal=...)`, `FaceRecord.resample(..., normal=...)`, `ComponentRecord.normals` | `position=...` and `normal_positions`; these are coordinates in metres, not normal vectors. |
| `Project.monitor()` | `watch_panel()`: immediate widget with `stop()`. `watch()` still iterates or calls a callback; `follow()` still blocks while displaying progress. |
| Viewer `mode="client"` / `"server"` / ... | `render_mode=...`; `mode` selects a physical eigenmode through `field`, `plot` and `show`. |
| `GeometryModel.plot()`, `LoadedGeometry.plot()` | `show()` for interactive 3D. `plot_cross_section()` remains matplotlib; `interact()` remains a notebook slider plot. |
| `t_index=...`, `f_index=...` | `frame=...` for a recording index, `t=...` or `f=...` for a physical selection. Conflicting selectors are rejected. |
| `ModeReport.plot(field="E")` | `plot(component="E")`; supported transverse E/H profiles are unchanged. |
| `tag_face`, `tag_faces`, `tag_edge`, `tag_edges`, `tag_vertex`, `imprint` | `tagged_face`, `tagged_faces`, `tagged_edge`, `tagged_edges`, `tagged_vertex`, `imprinted`; each returns an immutable owner. Model `add*` still mutates; `Path.forward` and `section` retain their names. |
| `n_actual_steps`, `MemoryEstimate.runs` | Single-run `n_steps`; aggregate `max_run_steps` and `n_steps_by_run`; count `n_runs`. Collections remain `runs`. `RunSettings.n_steps` requires an identified run. |
| `Project.result(name=...)`, reader/resume `excited=...` | `run=...` selects an existing run; `name=...` names a new general run; scattering `excited=...` selects actual excitations. |
| `PortSpecRectWG.width_a`, `height_b` | `width`, `height` in global tangential axis order. |
| Missing curated geometry exports | `geo.Bend`, `geo.Wrap`, `geo.ImportedSheet` are included in the generated reference and `geo.__all__`. |
| Uncurated returned surface/loss types | `fields.SurfaceCurrent`, `post.WallLossQ`, preserving class identity. |
| `solver.EigenmodeResult` | `analysis.EigenmodeResult`, preserving the implementation and class identity. |
| Whole-axis `.f`, sampled-axis `FieldSpectrum.frequencies` | `.f_axis` for sampled spectral/result axes, including `Signal1D`; `frequencies` for requested recording samples and eigenfrequencies; `f` for a selected frequency. |
| Surface-impedance `f_lo`, `f_hi` | `f_min`, `f_max` in Hz; working, guard and evaluation bands retain their distinct meanings. |
| `centre`, `cell_centres`, `cell_centred`, `cell_centred_layer` | `center`, `cell_centers`, `cell_centered`, `cell_centered_layer`. |
| Degree-valued `phase=...` | `phase_deg=...`; `angle_deg`, `twist_deg`, `draft_deg`, `phase_advance_deg` and the S-parameter output selector `phase(deg=...)` remain. |
| `AnalysisTD.estimate()`, inherited scattering `estimate()` | `estimate_memory()`, returning a `MemoryEstimate` with `n_runs`; scenario assumptions do not configure a later run. |
| `Mesh.from_grid(regions=[(material, six_value_bbox)])` | `regions=[(material, ((x0, y0, z0), (x1, y1, z1)))]`; assignment still uses cell centers and later regions overwrite earlier ones. |
| Raw and CAD BREP pairs | Both retained: ordered raw metre-space `read_brep(path)` / `write_brep(shapes, path)` and geometry-object CAD `import_brep(path, units=...)` / `export_brep(path, geometry, units=...)`. Their argument order differs deliberately. |

## Physical frequency and anchors

Previously an angular frequency was passed to a `Mode`:

```python
mode.gamma(2 * np.pi * 10e9)  # previous API
```

Now the evaluation boundary takes Hz and converts once internally:

```python
mode.gamma(f=10e9)
mode.z_modal(f=10e9)
```

`omega_c` remains a stored angular cutoff in rad/s. Old positional and
`omega=` evaluations fail, preventing a silent factor of 2π error.

A rectangular declaration's former `center` was already its corner, so copy
its value unchanged:

```python
from magnelio import ports

port = ports.PortAnalytical(
    name="input", plane="zmin", family="rect_wg",
    width=22.86e-3, height=10.16e-3, origin=(3e-3, 4e-3, None),
)
```

The declaration uses world 3D coordinates. A `PortSpecRectWG` uses the two
coordinates along the global tangential axes, ordered by axis number.
Neither form denotes a midpoint. Coax declarations continue using `center`.

## Normalization, selection and counts

`monitor.normalize_to_excitation(signal)` changes the reference used to read
its field response. Calling it again replaces that reference. Raw DFT bins
and `spectrum_raw` remain unchanged. `result.renormalize(50)` instead returns
a network referenced to 50 Ω and leaves the original network untouched.

Use `recording.plot(frame=2)` or `recording.plot(t=1e-9)`; use
`spectrum.plot(f=10e9, phase_deg=90)` for a complex field picture. Eigenmode
results use `result.show(mode=2, render_mode="none")` to build a scene for
physical mode 2 without displaying it. Supported field choices are unchanged.

`project.result(run="run_2")`, `project.monitors_for(run="run_2")` and
`magnelio.resume(project, run="run_2")` select a stored run. An exact run
name takes precedence; scattering `(port, mode)` selectors remain supported.
A sole run may be implicit, but multiple runs require an unambiguous choice.

`n_steps_by_run` exposes each aggregate run's solver count. `max_run_steps`
is their maximum, never their sum. To inspect one stored run's settings, use
`project.result(run=...).settings.n_steps`. Combined settings do not silently
report a representative run's count. Signal extrapolation length is separate
from the number of steps the solver actually marched.

## Stored projects and CAD exchange

Supported files keep their existing schema and stable disk vocabulary.
Readers and reconstruction recipes map legacy `center`, `width_a`, `height_b`,
`phase`, `freqs` and surface-layer coordinate keys explicitly. Corner values,
degree units, angular cutoffs, excitation identities, raw bins and checkpoints
retain their meaning. File compatibility does not restore old Python calls.
Unsupported file versions still fail explicitly.

For CAD workflows use `import_brep` / `export_brep` with explicit units and
Magnelio geometry objects; see the [CAD exchange guide](methods/cad-import.md).
Use raw `read_brep` / `write_brep` only when transporting an ordered sequence
of OpenCascade shapes already expressed in metres. Raw I/O preserves sequence
order and carries no material or named-topology metadata.
