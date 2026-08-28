# Magnelio – Design Decisions Log

This document records architectural and implementation decisions with their rationale.
New entries are appended.  Closed entries may be compacted to their
decision/rationale/verdict essence, and superseded entries collapse to a
tombstone (struck-through title + a few lines of what/why/where-to) — but DD
numbers are never reused or deleted: they are the anchor system cited across
src/, tests/ and the reference documents
(gate: `validation/tools/check_dd_references.py`).

Citations of the form `investigations/<topic>/…` (measurement dossiers:
DERIVATION/MEASUREMENTS/FINDINGS records with their probe scripts) and
`userscripts/…` (developer worksheets) refer to the maintainers' internal
records, which are kept outside the public repository.  They remain in the
text as provenance anchors: they name the evidence a decision rests on,
even where the record itself is not shipped.

---

## DD-001 — CPML over UPML

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Use Convolutional PML (CPML) as the absorbing boundary condition, not Uniaxial PML
(UPML) or Berenger Split-Field PML.

**Options evaluated:**
| Option                  | Notes |
|-------------------------|-------|
| CPML                    | Selected |
| UPML (Uniaxial PML)     | Requires modified constitutive relations throughout the PML region |
| Berenger Split-Field PML| Increases number of field components, non-physical split |

**Rationale:** CPML integrates directly into the standard FIT update equations via auxiliary
convolution variables (ψ fields), without requiring globally modified Maxwell equations or
coordinate stretching. Absorption performance is equivalent to UPML. Widely validated in open
tools (openEMS, MEEP). Auxiliary field storage is localized to PML cells only.

---

## DD-002 — Structure-of-Arrays (SoA) field storage

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Store E and H field components as six separate arrays (Ex, Ey, Ez, Hx, Hy, Hz),
i.e., Structure-of-Arrays (SoA) layout.

**Options evaluated:**
| Option      | Notes |
|-------------|-------|
| SoA         | Selected — six separate arrays |
| AoS         | Array of shape (Nx, Ny, Nz, 6) — poor vectorization |
| Interleaved | Mixed layouts — complex index arithmetic |

**Rationale:** SoA allows NumPy and CuPy to operate on entire field components in a single
contiguous memory pass, maximizing cache utilization and vectorization. Staggered Yee shapes
(e.g., Ex: (Nx, Ny+1, Nz+1) vs Hz: (Nx, Ny, Nz+1)) make a single-array AoS impractical
without padding. SoA is the dominant convention in FIT/FDTD literature and open implementations.

---

## DD-003 — pythonocc-core as OCC binding

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Use `pythonocc-core` (Python bindings to OpenCASCADE Technology) for CSG geometry
operations and mesh-critical queries.

**Options evaluated:**
| Option          | Notes |
|-----------------|-------|
| pythonocc-core  | Selected — full OCCT API |
| cadquery        | Higher-level; wraps pythonocc-core but abstracts away low-level queries |
| build123d       | Similar to cadquery; even newer, smaller community |

**Rationale:** Bounding-box extraction and face/edge intersection queries required for the
grid-line generation algorithm need low-level OCCT API access (BRep_Builder, BRepBndLib,
IntCurvesFace_ShapeIntersector). Higher-level wrappers (cadquery, build123d) hide these APIs.
pythonocc-core is available on conda-forge and covers all required OCCT functionality.

---

## DD-004 — Python ≥ 3.11 requirement

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Require Python 3.11 or newer.

**Options evaluated:** Python 3.10, 3.11, 3.12

**Rationale:** Python 3.11 provides `tomllib` in the standard library (useful for config files),
meaningful performance improvements over 3.10 (10–60% faster in CPython benchmarks), and is
broadly supported by all key dependencies (numpy, scipy, pythonocc-core on conda-forge).
Python 3.12 was considered but narrows the user base with limited additional benefit at this stage.

---

## DD-005 — Explicit / Imperative API style

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Adopt an explicit, imperative API style (object mutation, step-by-step calls) rather
than a fluent builder or declarative DSL.

**Options evaluated:**
| Option               | Notes |
|----------------------|-------|
| Fluent / Builder     | Method chaining (`.set_bc().add_port().run()`) |
| Explicit/Imperative  | Selected — objects created and mutated explicitly |
| Declarative config   | YAML/TOML-driven — poor discoverability |

**Rationale:** Explicit imperative code is consistent with NumPy/SciPy conventions familiar to
the target audience. Each step is inspectable in a Jupyter notebook cell. Debuggers can set
breakpoints between steps. No magic or hidden state mutations.

---

## DD-006 — Custom backend abstraction (get_xp pattern)

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Implement a minimal custom backend abstraction (`get_xp()` returning `numpy` or
`cupy`) rather than using a third-party compatibility layer.

**Options evaluated:**
| Option              | Notes |
|---------------------|-------|
| Direct numpy import | Non-switchable; blocks GPU path |
| array-api-compat    | Full standard compliance; heavier dependency |
| Custom wrapper      | Selected — minimal, sufficient for current scope |

**Rationale:** With a NumPy-only backend, a thin `get_xp()` / `set_backend()` pattern
is sufficient and zero-dependency. All numerical modules import `xp = get_xp()` instead of `np`.
Migration to `array-api-compat` is straightforward when CuPy support is added.

**Superseded in part.**  The module-global `get_xp()`/`set_backend()` state
described above is no longer the production path: [[DD-090]] made the backend a
per-solver value resolved by `resolve_backend()` and passed down explicitly, and
only `boundaries/cpml.py` still consults the global as a default.  The
`array-api-compat` question was revisited in [[DD-180]] and answered the same
way — at ~87 `xp.*` call sites the compatibility layer still buys nothing.

---

## DD-007 — scipy.sparse.linalg.eigsh for eigenmode solver

**Date:** 2026-03-09
**Status:** Accepted

**Decision:** Use `scipy.sparse.linalg.eigsh` (ARPACK wrapper) for both 2D and 3D eigenmode
solvers.

**Options evaluated:**
| Option                   | Notes |
|--------------------------|-------|
| ARPACK via scipy (eigsh) | Selected |
| LOBPCG                   | Better for very large problems; less mature in scipy |
| Custom Lanczos           | Unnecessary complexity |
| FEAST                    | Contour-integral method; requires external library |

**Rationale:** `eigsh` is battle-tested, ships with SciPy (no extra dependency), and handles
2D problems (typically < 100k DOF) with excellent performance. For 3D eigenmodes at
moderate grid sizes, ARPACK is adequate. FEAST can be revisited later for large 3D problems.

---

## DD-008 — M_mu dual-cell convention (A_primal / l_dual)

**Date:** 2026-03-10
**Status:** Accepted

**Decision:** M_mu[f] = μ₀ · μr · A_primal[f] / l_dual[f], not the inverse.

For the Hx face (dual y-z face): `M_mu = μ₀ · μr · dy[j] · dz[k] / dx_avg`
where `dx_avg = (dx[i-1] + dx[i]) / 2` is the dual edge length.

**Options evaluated:**
| Option                   | Notes |
|--------------------------|-------|
| A_primal / l_dual        | Selected — correct unit [H] (Henry = V·s/A) |
| l_dual / A_primal        | Wrong unit [1/(H)] — initial erroneous implementation |

**Rationale:** The FIT mass matrix M_mu represents inductance per H-face. The correct formula
integrates the magnetic flux Φ = μ · H · A_primal over the primal face area and divides by the
dual edge length l_dual (the path length over which the EMF is defined). The inverted formula
l/A produces units of [1/H], causing eigenfrequencies to be off by a factor of Δ² (cell size
squared), shifting resonances from GHz to MHz. This bug was detected and fixed in Session 3 by
comparing a 10×8×6 mm cavity's computed TE101 mode against the analytical value (24.1 GHz).

---

## DD-009 — PEC boundary conditions in eigenmode solver via DOF submatrix extraction

**Date:** 2026-03-10
**Status:** Accepted

**Decision:** Apply PEC boundary conditions in the 3D eigenmode solver by identifying
tangential-E DOFs on PEC domain faces and extracting a reduced submatrix that excludes them,
rather than using a penalty method or projection.

**Options evaluated:**
| Option                   | Notes |
|--------------------------|-------|
| DOF submatrix extraction | Selected — exact, no tuning required |
| Penalty term             | Approximate; requires large penalty constant; can cause ill-conditioning |
| Projection               | Iterative; works but adds complexity; equivalent to extraction for simple masks |

**Rationale:** For an all-PEC cavity, the natural (Neumann) boundary condition of the
variational formulation is PMC (H_tangential = 0), not PEC. Solving the full unreduced system
therefore yields PMC-cavity modes, not the physically correct PEC modes. DOF elimination
(extracting the submatrix of free DOFs) is the exact discrete analogue of the strong PEC BC
E_tangential = 0, eliminates the null space entirely, and requires no penalty tuning. The free
DOF mask is computed from `build_pec_mask_faces()` in `mesh/indexing.py`.

---

## DD-010 — ARPACK shift-σ estimation BC-aware (0.75 · λ₁_est)

**Date:** 2026-03-10
**Status:** Accepted

**Decision:** Estimate the ARPACK shift parameter σ as `0.75 · λ₁_est` where λ₁_est is the
estimated lowest eigenvalue ω₁², computed from the two longest domain dimensions and the
boundary condition type on each axis.

```
k_n(axis) = π/L        (PEC–PEC: half-wave)
           = π/(2·L)   (PEC–PMC or PMC–PEC: quarter-wave)
           = 0         (PMC–PMC: no propagating mode → skip axis)

λ₁_est = (c₀²/ε_r,max) · (k²_L1 + k²_L2)   (two longest axes with k > 0)

σ = 0.75 · λ₁_est
```

**Options evaluated:**
| Option                  | Notes |
|-------------------------|-------|
| Fixed σ (e.g., 1e20)    | Fails for varying grid sizes and BCs |
| BC-naive heuristic      | Uses π/L for all axes regardless of BC; over-estimates σ for PMC BCs |
| BC-aware heuristic      | Selected — accounts for half-wave vs quarter-wave per axis |
| User-supplied σ         | Supported as override; falls back to heuristic if None |

**Rationale:** The shift σ must satisfy λ₁/2 < σ < λ₁ to place it between the lowest and
zero-frequency modes (null space), so that shift-invert ARPACK ("LM" mode) finds physical
cavity modes before any spurious null-space modes. PMC boundaries on an axis double the
effective wavelength (quarter-wave resonance) compared to PEC (half-wave), so a BC-naive
estimate using π/L for all axes over-estimates σ and may place it above λ₁, causing ARPACK to
converge to the wrong modes. The factor 0.75 provides a safe margin below λ₁ for any aspect
ratio. If fewer than 2 axes have k > 0 (e.g., all-PMC box), a `ValueError` is raised with
guidance to supply σ manually.

---

## DD-011 — ~~Discrete Port: soft-source injection~~ → Superseded by DD-030

Additive soft source with matched-load convention (V_inc = V_src/2);
replaced by DD-030's semi-implicit Thévenin loading, which owns the
port-edge update instead of injecting past it.

---

## DD-012 — ~~Port2D: Hz dual formulation~~ → Deprecated (Port2D removed)

Hz-dual EVP chosen to dodge the E-formulation gradient null space;
the whole Port2D solver was removed with the Phase-2 modal pipeline
(DD-040/DD-048), whose 2D operators are the 3D-restricted ones.

---

## DD-013 — PlaneWaveSource: TF/SF formulation with `attach(solver)` + split inject

**Date:** 2026-03-11
**Status:** Accepted

**Decision:** Implement the plane-wave TF/SF source as a two-hook injection scheme:
`inject_E(fields, t_E)` called after the E update, and `inject_H(fields, t_H)` called
after the H update.  A one-time `attach(solver)` call in `FITTimeDomainSolver.setup()`
caches the solver's `_beta_E`, `_beta_H`, grid, and snapped box indices.

**Correction formulae (example: +z propagation, x-polarisation):**

- **E-correction at z_min** (`k = iz0`): E-update for `Ex[i,j,iz0]` incorrectly used
  `Hy[i,j,iz0-1]` from the SF region. Correction: `Ex[i,j,iz0] += β_E · Hy_inc(r, t_H)`.
- **H-correction at z_min** (`k = iz0−1`): H-update for `Hy[i,j,iz0-1]` incorrectly
  used `Ex[i,j,iz0]` from the TF region. Correction: `Hy[i,j,iz0-1] -= β_H · Ex_inc(r, t_E)`.
- z_max, x_min/max, y_min/max faces follow the same pattern with signs determined by
  which C / C^T term references the mismatched region.

**Current restriction:** Axis-aligned propagation only (k ∈ {±x̂, ±ŷ, ẑ}).  Oblique
incidence raises `NotImplementedError`.

**Options evaluated:**
| Option | Notes |
|--------|-------|
| Single `inject()` call between E and H | Cannot correctly time-stamp both corrections |
| Split `inject_E` / `inject_H` hooks | Selected — matches leapfrog half-step timing exactly |
| Hard-wire 1-D auxiliary FDTD for inc. field | Needed for oblique incidence; deferred |

**Rationale:** The leapfrog scheme stores E at integer steps and H at half-steps.
An E-correction needs H_inc at t − dt/2; an H-correction needs E_inc at t.
Splitting into two hooks avoids any interpolation error and matches the existing
CPML `update_E` / `update_H` pattern already used in `FITTimeDomainSolver`.

---

## DD-014 — I/O: HDF5 + VTK binary mode; cell-centre truncation for visualisation

**Date:** 2026-03-11
**Status:** Accepted

**Decision:** A unified project directory format replaces the earlier separate
`save_hdf5` / `load_hdf5` / `export_vtk` / `export_xdmf` functions:

- **`save_project(path, mesh, ...)`** writes `data.h5` (HDF5 with all data),
  `fields.xdmf` (ParaView descriptor), and optionally `geometry.stl`.
- **`load_project(path)`** reads back the full state including live monitor objects.
- Monitor field data is stored in XDMF row-major order `(n, nz, ny, nx)` so that
  ParaView reads it directly; `load_project` transposes back automatically.
- Schema version stored in `/metadata/project_schema_version`.
- The standalone VTK export was removed — XDMF+HDF5 covers all ParaView use cases.

**S-parameter zero-division:** `compute_sparameters` previously triggered a
`RuntimeWarning` from NumPy's divide-by-zero even though the result was correctly masked
by `np.where`.  Fixed by pre-masking the denominator before division.

---

## DD-015 — Waveguide benchmark: PlaneWaveSource timing metric; CPML and port limitations

**Date:** 2026-03-11
**Status:** Accepted

**Context:** The `bench_waveguide_transmission.py` benchmark went through three failed
designs before a working metric was found.

**Approaches evaluated:**

| Approach | Problem |
|----------|---------|
| `DiscretePort` S11 energy ratio | Z0=50 Ω ≠ η₀=377 Ω free-space impedance → S11 > 0 dB by definition; not a propagation quality metric |
| `DiscretePort` in 3×3 mm PEC box | TE10 cutoff = c₀/(2a) = 50 GHz >> f_max=10 GHz; all modes evanescent; wave cannot propagate |
| CPML residual-energy absorption | All-PEC (no CPML) box: CPML achieves only ~−3 dB/bounce in the current 8-cell layer; energy leaks back before the test window; not a reliable metric |

**Decision:** Use `PlaneWaveSource` (TF/SF) injecting a Gaussian pulse into an all-PEC box,
with a `FieldProbe` measuring peak arrival time at z = L_z/2.

Acceptance criteria:
1. `peak_Ex > 1e-4 V/m` — wave reached the probe.
2. Timing error `|t_peak_measured − t_peak_expected| / t_peak_expected < 10 %` — numerical
   phase velocity ≈ c₀ within FIT dispersion (0.2 % achieved with 1 mm cells).

**Rationale:** This metric is insensitive to CPML quality, independent of port impedance
matching, and directly tests that the TF/SF injection and FIT leapfrog propagate a wave at
the correct phase velocity.  It serves as a combined integration test for `PlaneWaveSource`,
`FieldProbe`, and `FITTimeDomainSolver`.

---

## DD-016 — Geometry: `GeometryModel` container + `BRepClass3d_SolidClassifier` for material filling

**Date:** 2026-03-11
**Status:** Accepted

**Context:** `Mesh.from_geometry()` previously had a placeholder stub that filled all cells
with air regardless of the CSG geometry.

**Decisions:**

1. **`GeometryModel` class** — thin ordered list wrapper (`add(shape)`, iteration, `bounding_box()`).
   Accepted by `Mesh.from_geometry()` and `extract_critical_planes()` via the `__iter__` protocol,
   so plain Python lists of shapes also work.

2. **Material filling via `BRepClass3d_SolidClassifier`** — for each CSG shape, the cell
   centres are tested against the OCC solid classifier:
   ```python
   classifier.Load(occ_shape)
   classifier.Perform(gp_Pnt(x, y, z), tolerance)
   state = classifier.State()   # TopAbs_IN or TopAbs_ON → inside
   ```
   Shapes are applied in insertion order; later shapes overwrite earlier ones (last-wins
   semantics, same as `from_grid(regions=...)`).

3. **OCC tests gated by `pytest.importorskip`** — all 30 OCC-dependent tests are
   automatically skipped when `pythonocc-core` is not installed.  The 18 non-OCC tests
   (class creation, material inheritance, `GeometryModel` API) always run.

**Alternatives considered:**

| Approach | Problem |
|----------|---------|
| AABB material filling for `from_geometry()` | Lossy for non-box shapes (sphere, cylinder, boolean ops) |
| Ray-casting (custom) | Complex to implement correctly; OCC provides robust tested implementation |
| Penalty shift (project to nearest face) | Not a volume classifier; ambiguous for complex BReps |

**Limitation:** Point-in-shape loop is O(N_cells × N_shapes) with no spatial acceleration.
Acceptable for current mesh sizes.  A BVH or OCC `BVH_Tree` wrapper can be added later.

---

## DD-017 — Thin PEC conductor: edge-mask via `apply_thin_pec_sheet()`

**Date:** 2026-03-11 (revised 2026-03-11)
**Status:** Accepted

**Decision:** Model an electrically thin PEC metallization as an **edge mask** at a single
grid-node plane, with no PEC volume cell and no extra cell layer.  The function
`apply_thin_pec_sheet(mesh, axis, position, rect)` in `mesh/indexing.py` sets the
`pec_mask_edges` bits directly for all tangential E-edges at the specified plane.

For a microstrip trace of width *w* on a substrate of height *h* (sheet normal = y-axis):

```
apply_thin_pec_sheet(mesh, axis="y", position=h,
                     rect=(x_trace_start, 0.0, x_trace_end, L))
```

**FIT model (axis="y", sheet at y = y[j_h]):**

- Tangential directions: x and z.  Normal direction: y (Ey **not** affected).
- `Ex[i, j_h, k]` for `x_c[i] ∈ [x_min, x_max]` and `z[k] ∈ [z_min, z_max]` → PEC
  (flat index `i*(Ny+1)*(Nz+1) + j_h*(Nz+1) + k`)
- `Ez[i, j_h, k]` for `x[i] ∈ [x_min, x_max]` and `z_c[k] ∈ [z_min, z_max]` → PEC
  (flat index `i*(Ny+1)*Nz + j_h*Nz + k`)

Only **one** `forced_plane` at `y=h` is needed in `MeshControl` (to snap the grid node
to the substrate surface).  No PEC `Brick` in the geometry, no `y=h+t` plane.

**Options evaluated:**
| Option | Notes |
|--------|-------|
| Edge mask at one grid node (`apply_thin_pec_sheet`) | **Selected** — correct FIT model, no spurious cell layer |
| One-cell PEC brick + forced_planes at y=h and y=h+t | Previous approach (DD-017 v1) — adds an extra grid plane, violates thin-sheet spirit |
| Zero-thickness OCC face | OCC solids require non-zero volume; face-only BRep not classifiable by BRepClass3d |

**Rationale:** A PEC Brick with non-zero thickness creates two forced planes and adds a cell
layer, potentially degrading the mesh aspect ratio.  The edge-mask approach matches the FIT
theory for a perfectly conducting sheet: tangential E-field components at the sheet plane are
set to zero while the normal component is unaffected.  The mode solver reads
`pec_mask_edges` when building its free-DOF lists, so it automatically sees the
metallization.

---

## DD-018 — ~~Port2D S-parameter deembedding~~ → Deprecated (Port2D removed)

Incident-amplitude tracking via excite_signal died with Port2D;
power-wave a/b extraction lives in the modal recorder chain.

---

## DD-019 — Energy-based early stopping in `FITTimeDomainSolver`

**Date:** 2026-03-11
**Status:** Accepted

**Decision:** Add an `energy_stop_db: float | None` parameter to `FITTimeDomainSolver`.
When set, the solver monitors total electromagnetic energy every `energy_check_interval`
steps (default 100) and terminates once:

```
E_total(t) < E_peak × 10^(−energy_stop_db / 10)
```

Total energy is computed as:
```
E_total = 0.5 · (M_eps · e² + M_mu · h²)     [scalar dot-products]
```

using the precomputed `M_eps` and `M_mu` diagonals stored in `setup()`.

The solver also saves `_peak_energy` and `_actual_steps` for postprocessing diagnostics.

**Options evaluated:**
| Option | Notes |
|--------|-------|
| Fixed `total_time_steps` only | Simulation may terminate before energy decays (under-run) or waste time after (over-run) |
| Energy-based adaptive stopping | Selected — standard practice in production FIT/FDTD codes (e.g. openEMS) |
| Port-signal-based stopping | Requires FFT during run; more complex; misses stored energy in evanescent fields |

**Rationale:** For wideband S-parameter extraction, the simulation must run until all
transients have decayed to negligible amplitude. A fixed step count derived from
`10 / f_max` may under-run for long structures or over-run for short ones.
Energy monitoring at 10^(−25/10) ≈ −25 dB is the standard criterion in openEMS
(`--energy-limit -25`) and matches user expectations.  The check every 100 steps
adds negligible overhead (two dot products per check).

---

## DD-020 — ~~Port2D quasi-TEM Laplace solver~~ → Deprecated (Port2D removed)

Port2D's 2D electrostatic Laplace path; the idea returned
3D-consistently as ``solve_tem_laplace`` / ``solve_qtem_laplace``
on the FIT-restricted operators.

---

## DD-021 — ~~Mur scalar ABC on modal-port faces~~ → Superseded (modal operators own port faces)
Superseded by the modal port termination chain (DD-027 → DD-040 →
DD-054/DD-055).  What survives is the face-closure mechanism it
introduced: a bbox face hosting a modal port is exempted from the
global PEC closure and the port operator owns that boundary (the
`solver/fit_td.py` port-face masks cite this entry).  The entry text
itself was never authored — tombstone reconstructed in session 147.
---
## DD-022 — ~~Port2D waveform~~ → Deprecated (Port2D removed)

Gaussian (TEM) vs modulated-Gaussian (TE/TM) selection; survives as
the ``ExcitationSpec.waveform`` choice of the modal factory.

---

## DD-023 — ~~Port2D modal E+H ABC~~ → Superseded by DD-027

Full modal E+H absorbing condition at Port2D faces; folded into the
modal Mur chain (DD-027) and ultimately the modal port operator
(DD-040).

---

## DD-025 — Z_pi (power-current line impedance) for TEM modes

**Date:** 2026-03-19
**Status:** Accepted

**Decision:** Compute and store the power-current line impedance `Z_pi` in `ModeResult` for
TEM/quasi-TEM modes. TE/TM modes get `Z_pi = None`.

**Context:** The existing `Z_wave = √(μ₀/(ε₀·ε_eff))` is the wave impedance (E/H ratio), not
the characteristic line impedance (e.g. 50 Ω of a coaxial cable). `Z_pi` is needed for
S-parameter normalisation and circuit matching.

**Mathematics:** After Poynting normalisation (P = 1 W):
- `Z_pi = V²/P = V²`
- `V = |∫ E · dl|` along a path between conductors
- For TEM modes E is curl-free (Laplace), so V is path-independent
- All TEM impedance definitions coincide: `Z_pi = Z_pv = Z_vi = Z_char`

**Implementation:**
1. `ModeResult` gains `Z_pi: float | None = None` (backward-compatible default)
2. `FITModeSolver._compute_zpi_tem()` integrates E along a straight v- or u-directed path
   between the reference conductor (conductor 0) and the excited conductor
3. `AnalyticalCoaxialSolver` sets `Z_pi = Z_char` directly (exact analytical value)
4. `WaveguidePort` stores `_Z_pi_modes` alongside `_Z_modes`

**Poynting normalisation fix:** For TEM modes in inhomogeneous media (e.g. microstrip),
`_apply_poynting_norm` now accepts optional `mat_u`/`mat_v` arrays so each edge uses its
local wave impedance `η(x,y) = √(μ₀/(ε₀·ε_r(x,y)))` instead of the area-averaged `Z_w`.
Without this, Z_pi for a microstrip on RO4003 diverged from the Wheeler reference by ~25 %;
with local η the error drops below 5 % at N=30.

**Verification:** Parallel-plate Z_pi matches η₀·d/W within 5 %, stripline Z_pi < Z_wave,
microstrip Z_pi converges to Wheeler within 5 % at N=30, TE modes return None.

---

## DD-026a — H-field storage + API extension for WaveguidePort (Phase A)

**Date:** 2026-03-19
**Status:** Implemented

**Problem:** `WaveguidePort.interpolate_fields` only returned E-fields,
`plot_mode_field` could not plot H-fields, and `ModeResult` did not store H-fields.
For reference-grade waveguide ports, both field profiles must be available.

**Decision — Phase A:** Derive H_t from E_t via the plane-wave impedance relation
`H_t = (1/Z)(ẑ × E_t)`. This is exact for TEM and homogeneous-fill waveguide modes.

**Implementation:**

1. **ModeResult** extended with optional `h_t_u`, `h_t_v` fields (`None` default for
   backward compatibility).

2. **FIT eigenvalue solver** (TE/TM modes): After Poynting normalisation, computes
   `h_t_u = -e_t_v / Z_w`, `h_t_v = e_t_u / Z_w` using the scalar area-averaged Z_w.

3. **Laplace/TEM solver**: Uses per-DOF local impedance
   `η(i,j) = √(μ₀/(ε₀·ε_r(i,j)))` from `mat_u`/`mat_v` arrays. H_u DOFs sit at
   ev_free positions, H_v at eu_free positions.

4. **AnalyticalCoaxialSolver**: Sets `h_t_u = [0.0]`, `h_t_v = [C_norm/η]` and
   stores `_coax_C_h = C_norm/η` for `H_θ = C_h/r` interpolation.

5. **`interpolate_fields`**: New `field="E"|"H"` parameter. H uses swapped DOF lists
   (H_u on ev_free, H_v on eu_free). Inline fallback if `h_t_u is None`.

6. **`attach_to_mesh`**: Builds `h_profiles = [(Z_m*e_eu, Z_m*e_ev)]` for ABC overlap.
   `update_bc` uses pre-computed `h_profiles` instead of inline `Z_m * e_eu[loc]`.
   Functionally equivalent in Phase A, but decoupled for Phase B.

7. **Plotting** (`plot_waveguide_port.py`): `_fields_to_cell_centres`,
   `sample_mode_on_grid`, `plot_mode_field`, `_select_field`, and `plot_mode_summary`
   all accept `field="E"|"H"`. H-field staggering (H_u on Ev grid, H_v on Eu grid)
   is handled correctly in cell-centre interpolation.

**Phase B (DD-026b):** See below.

**Verification:** 49 unit tests pass including:
- `test_h_fields_populated`: h_t_u/h_t_v non-None after solve
- `test_h_field_impedance_relation`: h_t_u ≈ -e_t_v/Z, h_t_v ≈ e_t_u/Z (rtol=1e-12)
- `test_poynting_cross_product`: ∫(E_u·H_v - E_v·H_u) dA ≈ 1 W
- `test_tem_h_fields`: TEM H-fields populated and non-trivial
- `test_interpolate_fields_h`: field="H" returns correct values

---

### DD-026b — TM eigenvalue solver, β(f_ref), frequency-dependent Z_wave (Phase B)

**Problem:** Phase A (DD-026a) uses η = √(μ₀/(ε₀·ε_eff)) as wave impedance for all
modes. This is correct for TEM and at cutoff, but wrong for propagating TE/TM modes:
- TE: Z_TE(f) = ωμ/β = η / √(1-(f_c/f)²) — higher than η
- TM: Z_TM(f) = β/(ωε) = η · √(1-(f_c/f)²) — lower than η

Additionally, the Hz-based EVP only finds TE modes. TM modes (E_z-based) were missing.

**Solution (5 parts):**

1. **TM Eigenwert-Solver** (`_solve_tm_evp`): Scalar E_z eigenvalue problem
   -∇·((μ₀μ_r)⁻¹ ∇E_z) = ω_c² · ε₀ε_r · E_z, with Dirichlet E_z=0 on PEC nodes.
   The 1/MU0 factor in the stiffness matrix ensures eigenvalues are ω_c² directly
   (matching the TE EVP convention). Transversal E_t is reconstructed via
   E_t = -∇E_z at free edge DOFs.

2. **PEC node set extraction** (`_build_pec_node_set`): Factored out of
   `_count_conductors` for reuse by both the conductor counter and TM EVP.

3. **TE + TM mode merge**: `_solve_on_grid` now runs both the Hz-based TE EVP
   and the E_z-based TM EVP, then merges and sorts by f_cutoff, keeping the
   top n_modes_evp non-TEM modes.

4. **β and Z_wave(f_ref)**: When `f_ref` is provided:
   - β² = (ω_ref² - ω_c²) · μ₀ · ε₀ · ε_eff (evanescent if ≤ 0 → β=0)
   - TE: Z_wave = ω_ref · μ₀ / β
   - TM: Z_wave = β / (ω_ref · ε₀ · ε_eff)
   - TEM: β = ω_ref · √(μ₀ · ε₀ · ε_eff), Z_wave unchanged (η)
   When `f_ref=None` (default): Z_wave = η, β = None (Phase A behaviour).

5. **ModeResult.beta**: New optional field storing the propagation constant at f_ref.
   WaveguidePort gains `f_ref` parameter, passed through to solver.

**Backward compatibility:**
- `f_ref=None` (default) produces identical results to Phase A
- TM modes now appear with `mode_type="TM"`. Since TM11 cutoff > TE10 cutoff,
  existing tests with n_modes ≤ 3 in WR-90 are unaffected.

**Verification:** 61 unit tests pass (49 existing + 12 new):
- `test_tm11_cutoff`: TM11 in WR-90 matches analytical f_c = c/2·√(1/a²+1/b²) (<5%)
- `test_te_tm_degenerate_square`: TE11/TM11 degenerate in square waveguide
- `test_tm_poynting_norm`: TM mode Poynting integral ≈ 1 W
- `test_beta_te_with_fref`: β matches √((ω²-ω_c²)·μ₀ε₀) (<2%)
- `test_z_wave_te_with_fref`: Z_TE = ωμ/β at f_ref (<2%)
- `test_z_wave_tm_with_fref`: Z_TM = β/(ωε) at f_ref (<2%)
- `test_evanescent_beta_zero`: f_ref < f_c → β = 0
- `test_backward_compat`: f_ref=None → Z=η, β=None

---

## DD-027 — ~~Modal Mur-ABC for waveguide ports~~ → Superseded by DD-040

First-order Mur on the modal amplitude; superseded by the DD-040
modal port operator (and later the exact DTBC, DD-054/DD-055) —
modal Mur survives only as the analytical-path fallback branch.

---

## DD-028 — Feature-based mesh resolution (two-scale grading)

**Date:** 2026-03-23, refined 2026-04-25
**Status:** Accepted

**Decision:** `Mesh.from_geometry()` uses **two independent cell-size scales** per
axis:

1. **`h_max`** (bulk, wavelength-based):
   `h_max = λ_min / min_nodes_per_wavelength` — the global cap on cell size,
   applies far from material interfaces.
2. **`h_fine`** (interface, feature-based):
   `h_fine = min_gap / min_cells_per_feature` — the cell size at material
   interfaces, where `min_gap` is the smallest distance between adjacent critical
   planes on any axis that has at least 3 critical planes.

Crucially, **`h_fine` applies only locally** (cells touching material interfaces).
Between `h_fine` at interfaces and `h_max` in the bulk, cells grow geometrically
by `growth_factor`.

**Problem (initial, 2026-03-23):** Without feature-based resolution, geometries
with small features relative to the wavelength were severely under-resolved. Example:
rect-coax with 2 mm inner conductor at f_max=10 GHz (λ_min=20.7 mm in PTFE): only
2 cells across the inner conductor, causing S11=0 dB at DC (total reflection).

**Initial fix (2026-03-23):** Added the feature rule and combined it with the
wavelength rule via `h_target = min(h_wavelength, h_feature)`, applied globally.
Plus a "cell-size cap" guard that fell back to uniform spacing whenever any
graded cell exceeded `h_target`.

**Problem with the initial fix:** `min(h_wavelength, h_feature)` was applied
**globally** — a single 1.27 mm coax-inner gap forced 0.32 mm cells across the
entire 60 mm bulk waveguide, ~10× more cells than the wavelength criterion
required. The cell-size cap also disabled all grading in long intervals (max_step
inevitably > h_target after a few growth steps).  The boundary-grading unit tests
"passed" only by floating-point luck (cell_at_wall = 4.000000000000002e-04 vs
3.999999999999996e-04), hiding the broken behaviour.

**Refined fix (2026-04-25):** Split `h_target` into two parameters passed
separately to `_generate_axis_lines`:

* Per interval, cells start at `h_fine` next to material interfaces and grow by
  `growth_factor` toward `h_max`.
* Boundary intervals use `_grade_then_uniform`: ramp from `h_fine` at the
  interior interface, uniform fill with `h_max` toward the domain wall. The
  ramp's last cell is retargeted in a single pass so its size matches
  `h_uniform · g`, eliminating the back-step at the ramp/uniform boundary.
* Interior intervals use `_grade_symmetric_to_uniform`: symmetric ramps from
  both ends, uniform middle. Short intervals fall back to legacy
  `_graded_subdivision` for clean ratio-`g` cells.
* Tiny remainders (rest < 0.5·h_max) are absorbed into the last ramp cell
  rather than spawning a mini-cell that would create a backward jump.
* `from_geometry`: `h_max = h_wavelength` (never reduced); `h_fine =
  min(h_wavelength, h_feature)`.

**Parameter:** `MeshControl.min_cells_per_feature: int = 4` — minimum cells
across the smallest geometry gap. Set to 0 to disable feature-based refinement.

**Industry practice:** mainstream time-domain solvers default to 3–4 cells
per feature; FEM-based tools use adaptive refinement with similar geometric
constraints. openEMS recommends ≥3 cells per feature in its documentation.

**Bug fix (2026-03-23):** `extract_critical_planes()` in `occ_backend.py` was
not iterating over `GeometryModel` shapes — it treated the entire model as a
single shape and only extracted its overall bounding box. Fixed: `list(geometry)`
when iterable.

**Verification:**

* Rect-coax (a=2mm, b=10mm) with default `MeshControl()` at 10 GHz: Nx=20
  (uniform, all cells ≤ h_target). S11 < −50 dB, energy converges.
* WR-90 / coax transition (60 mm bulk, 1.27 mm inner-conductor gap) at
  f_max=12.4 GHz: before refinement 923 520 cells, mesh build 527 s, dx
  uniform 0.32 mm; after refinement ~33 000 cells, mesh build ~60 s, dx 0.32–
  ~2 mm with grading, max consecutive cell ratio 1.30 (= growth_factor) on
  boundary intervals.

## DD-029 — ~~Port grid independence~~ → Superseded by DD-048

Reframed as the reference-mode path of DD-048's two-path
architecture (the drive mode must be an eigenvector of the actual
FIT-TD operator; only the *reference* mode is mesh-independent).

---

## DD-030 — DiscretePort: semi-implicit Thévenin + update_bc integration

**Date:** 2026-04-02
**Status:** Accepted (supersedes DD-011)

**Decision:** Replace the soft-source `DiscretePort` with a semi-implicit Thévenin lumped
port that integrates with the `update_bc` / `SignalRecorder` pipeline used by
`TransientAnalysis`.

### Key changes from DD-011

| Aspect | DD-011 (old) | DD-030 (new) |
|--------|-------------|--------------|
| Definition | `position` + `direction` | `start` + `end` (two points) |
| Impedance model | None (soft source) | Semi-implicit Thévenin (single lumped element) |
| Solver integration | Legacy `inject_E` / `sample` | `update_bc` (same as WaveguidePort) |
| S-parameter pipeline | Own `s11()` method | `TransientAnalysis` + `SParameterResult` |
| Multi-edge | No | Yes (arbitrary number of edges along one axis) |

### Physics

The port acts as a **single lumped circuit element** across all its edges.  After the
standard E-update, the port reads the total voltage and applies a semi-implicit Thévenin
correction:

```
V_total = Σ E_k · dl_k                    (total voltage from standard update)
β_sum   = Σ beta_E_k                      (electromagnetic cell impedance [Ω])
I_port  = (V_src − V_total) / (Z0 + β_sum)  (semi-implicit lumped current)
ΔE_k    = beta_E_k · I_port / dl_k        (same I_port at every edge — series circuit)
```

The effective port impedance is `Z0 + β_sum`, which is always positive → unconditionally
stable.  For typical meshes `β_sum ≫ Z0`, so the port behaves as a nearly-transparent
source.  The `Z0` term provides physically correct absorption at the port.

Wave amplitudes (matched-load convention, unchanged from DD-011):
```
a = V_src / 2,   b = V_corrected − V_src / 2
V_corrected = V_total + I_port · β_sum
```

### Options evaluated

| Option | Notes |
|--------|-------|
| Distributed impedance (`M_sigma_port_k = L/(Z0·dl_k)`) per edge | Rejected — over-damps multi-edge ports; each edge independently absorbs, `V_total → 0` as `N → ∞` |
| Explicit post-correction (`I = (V_src − V)/Z0`) | Rejected — unstable when `β_sum > Z0` (typical case) |
| **Semi-implicit** (`I = (V_src − V)/(Z0 + β_sum)`) | Selected — stable, self-consistent, correct single- and multi-edge limit |

### Encapsulation

The solver passes its coefficient arrays (read-only) via `read_solver_coefficients()`.
The port stores `beta_E` values at its edges but does **not modify** any solver state.

**Applies to:** `port_discrete.py`, `fit_td.py` (5-line hook in `setup()`).

---

## DD-031 — ~~PML-backed waveguide ports with multi-mode S-parameters~~ → Superseded by DD-040

CPML-terminated port faces with dual E-overlap mode extraction;
superseded by the DD-040 modal operator after the x/y-face
instability (BUG entry below) and the cost of the PML slab.

---

## BUG — PML-backed waveguide ports unstable on x/y-axis faces

**Date:** 2026-04-05
**Status:** Fixed

**Root cause:** All four CPML convolutional corrections for `axis="y"` in
`cpml.py` had flipped signs (`+=` instead of `-=` and vice versa).  The curl
equations cycle as x→y→z→x; the y-axis code was written by analogy to z-axis
but applied the signs to the wrong tangential components (Ex instead of Ez and
vice versa), causing the PML to amplify instead of absorb.

**Fix:** In `update_E` y-axis: `Ex += …` → `Ex -= …`, `Ez -= …` → `Ez += …`.
In `update_H` y-axis: `Hx -= …` → `Hx += …`, `Hz += …` → `Hz -= …`.
The x-axis code was verified correct by derivation from Maxwell's curl equations.

---

## DD-032 — Three-tier solver kernel dispatch and GPU backend

**Date:** 2026-04-11
**Status:** Accepted (GPU path untested — CuPy not yet installed)

**Decision:** Replace the single sparse-matvec solver loop with a three-tier kernel dispatch
and make the entire solver pipeline backend-agnostic (NumPy / CuPy).

**Kernel tiers (fastest first):**

| Tier | Function | Backend | Temp buffers | How it works |
|------|----------|---------|-------------|--------------|
| 1 | `update_E_fused` / `update_H_fused` | Numba CPU | None | `@njit(parallel=True)` — fused curl + material multiply in one pass per component. Boundary E-edges handled via conditional accumulation. |
| 2 | `update_E_stencil` / `update_H_stencil` | CuPy GPU or NumPy | 6 curl buffers | Uses only `+=`, `-=`, `[:] =` slice ops — compatible with any array backend. GPU path: CuPy translates these to CUDA kernels. |

**Selection logic:** `setup()` checks `get_xp()`. GPU → always tier 2 (Numba cannot operate on
CuPy device arrays). CPU → tier 1 if Numba available, else tier 2.

**GPU data flow:**
- Fields allocated on GPU via `FieldState.zeros(xp=cupy)`.
- Material coefficients and PEC index arrays computed on CPU in `setup()`, then transferred
  once via `xp.asarray()`.
- PEC/CPML boundary conditions use slice assignments — CuPy-compatible.
- Energy monitoring uses `@` operator (returns Python float via implicit device→host scalar copy).
- CPML auxiliary arrays remain on CPU in this version (implicit transfers per step).
  Full GPU-native CPML is a follow-up.

**Performance results (CPU, xlarge 1.6M cells / 11201 steps):**

| Phase | Solve time | Speedup vs baseline |
|-------|-----------|---------------------|
| Baseline (sparse CSR matvec) | 2432s | 1× |
| + Flat FieldState + in-place ops | 1211s | 2× |
| + Matrix-free numpy stencils | 347s | 7× |
| + Numba fused kernels + PEC int-idx | 134s | 18× |

At 1.6M cells, the CPU solver is now memory-bandwidth-limited (~40 GB/s DDR4).
GPU (RTX 4070 Super, ~504 GB/s GDDR6X) is projected to yield ~12× additional speedup.

**Rationale:** FDTD is bandwidth-bound (arithmetic intensity ~2 FLOP/byte). Adding CPU threads
beyond what Numba's `parallel=True` already provides cannot exceed the memory bus. GPU has 10–25×
higher memory bandwidth than DDR4/DDR5, making it the natural next step. The CuPy drop-in approach
avoids a separate CUDA codebase while still enabling GPU acceleration for the entire solver loop.

**Later note — the tier numbers above were renumbered.**  When the hand-written
CUDA kernel arrived it took the top slot, so the module docstring of
`_operators/numba_kernels.py` today counts 1 = fused CUDA, 2 = fused Numba CPU,
3 = array stencil.  The table above predates that and numbers the surviving two
1 and 2.  The stencil tier is what both faster paths fall back to, and it is the
only reason the dispatch is portable at all; [[DD-180]] records what that
fallback does and does not buy a prospective third backend.

---

## DD-033 — Scalable 3D eigenmode solver: evaluated approaches

**Date:** 2026-04-11
**Status:** Accepted (iterative approaches evaluated and rejected)

**Problem:** The ARPACK `eigsh` shift-invert path builds the full sparse matrix
`A = C^T diag(M_mu_inv) C` and factors `(A - σB)` via SuperLU.  Both steps are O(n^1.5)
to O(n^2) for 3D problems and become the bottleneck above ~50k cells.

**Current solution:** ARPACK + SuperLU remains the production backend.  Works
reliably up to ~100³ grid cells.  SciPy's default COLAMD reordering provides
good factorisation performance without explicit configuration.

**CurlCurlOperator** (`operators/curl_curl_operator.py`): Matrix-free operator applying
`A @ x = C^T diag(M_mu_inv) C @ x` via stencil operations (`curl_e_stencil` → M_mu_inv
multiply → `curl_h_stencil`).  Operates in the PEC-free DOF subspace. Pre-allocates 9
workspace buffers.  Verified bit-exact against explicit sparse matrix in unit tests.
Retained for future GPU solvers.

**LOBPCG folded-spectrum backend** (`solver="lobpcg"`, experimental):
PEC cavities have a large gradient null space (~N³ modes with eigenvalue 0).  Naive
LOBPCG with `largest=False` finds these trivial modes first.  The folded-spectrum
transformation `S_std² y = μ y` with `S_std = B⁻½(A/σ−B)B⁻½` maps modes near σ to
the smallest μ = (λ/σ−1)² while the null space folds to μ ≈ 1.  The B⁻½ absorption
into the operator avoids B-orthogonalisation issues from small M_ε entries (~10⁻¹⁴).
True eigenvalues are recovered via the Rayleigh quotient on the original (A, B) pair.

Evaluated and rejected approaches for null-space elimination:
- Tree-cotree gauging: PEC restriction breaks de Rham exactness → spurious modes
- Gradient regularisation A + α·G·Gᵀ: ILU singular (GGᵀ is graph Laplacian),
  Jacobi amplifies null space by O(α/diag(A)) ≈ O(10¹⁴)
- Direct LOBPCG + AMG on free subspace: failed to converge (200 iterations)

Current limitation: unpreconditioned LOBPCG is ~75× slower than ARPACK+SuperLU at
2800 cells and achieves only ~2% accuracy (vs. 0.2% for ARPACK).  A Maxwell-specific
preconditioner (e.g. AMS from hypre, or subspace correction with the gradient operator)
would be needed to make LOBPCG competitive.  Not auto-dispatched; retained for
experimental use and as foundation for future GPU solvers.

**ILU-GMRES evaluated and rejected:**
- `spilu` fails with "Factor is exactly singular" for fill-reducing orderings (MMD,
  COLAMD) on the indefinite shifted matrix (A−σB)
- `NATURAL` ordering produces a valid ILU but GMRES doesn't converge (ILU quality
  too poor for the indefinite system — 9.6× fill yet wrong eigenvalues)

**pyamg AMG-CG evaluated and rejected:**
An AMG-preconditioned CG inner solve for the shifted system (A−σB) was implemented
and benchmarked using pyamg's smoothed-aggregation solver.  Results:

| Grid | n_free | AMG-CG [s] | SuperLU [s] | Ratio |
|------|--------|-----------|-------------|-------|
| 10×7×5 | 762 | 1.7 | 0.2 | 9× slower |
| 20×13×10 | 6,663 | 29 | 1.4 | 20× slower |
| 30×20×15 | 24,365 | 1,139 | 36 | 32× slower |

AMG gets **worse** with increasing mesh size: CG iteration counts grow instead of
staying constant, indicating that pyamg's scalar SA-AMG does not achieve mesh-independent
convergence for the vector-valued Maxwell curl-curl operator.  Tested configurations:
- Default Jacobi smoother (best: 162 CG iters at 6.6k DOFs)
- Gauss-Seidel smoother (worse: 246 iters)
- Near-null-space vectors (3 constant-field columns): diverged
- W-cycle: no improvement

Root cause: pyamg's smoothed-aggregation AMG is designed for scalar elliptic problems.
The curl-curl operator's vector-field structure requires Maxwell-specific AMG approaches
(e.g., Auxiliary-Space Maxwell Solver from hypre) which pyamg does not implement.

The AMG-CG path is retained as `solver="arpack-amg"` for experimental use but is not
auto-dispatched.  Inner tolerance: rtol=1e-8 (looser values produce wrong eigenvalues
because scipy's ARPACK wrapper does not support inexact shift-invert).

**CHOLMOD Cholesky evaluated (experimental):**
Approach: tree-cotree gauging eliminates the gradient null space, making the cotree system
positive definite → CHOLMOD Cholesky factorisation of (A_ct − σ·B_ct).

Implementation:
- `build_gradient_matrix(grid)` added to `operators/curl.py`: discrete gradient G (nodes→edges),
  verified C·G = 0 (de Rham exactness).
- Tree-cotree via BFS on PEC-component-collapsed super-node graph.  Union-Find identifies
  PEC-connected node groups.  Tree size = dim(ker(A_f)) exactly.
- Sigma estimation bug fixed: was using two LARGEST k² terms (giving σ > ω₁² for non-cubic
  cavities), corrected to two SMALLEST k² terms.

Results: tree-cotree gauging correctly eliminates the null space (A_cotree is PD), but the
gauging introduces **spurious low-frequency modes** (artifacts from constraining tree-edge
DOFs to zero).  These modes have eigenvalues well below the physical spectrum (e.g., 5e20
vs. ω₁²=3.2e21 for 30×20×15mm cavity).  Consequently (A_ct − σ·B_ct) remains indefinite
whenever σ exceeds the lowest spurious eigenvalue — which it typically does for auto-estimated
σ.  This is a known limitation of tree-cotree in eigenvalue (as opposed to source) problems.

Also evaluated and rejected:
- Gradient regularisation (A + α·G_f·G_fᵀ): PEC restriction breaks de Rham (C_f·G_f ≠ 0),
  so G_f·G_fᵀ perturbs physical modes by 10–100%.
- Diagonal beta shift (CHOLMOD beta parameter): gradient modes contaminate ARPACK spectrum
  at all β values (physical and shifted-gradient modes have similar ARPACK θ values).

The CHOLMOD path is retained as `solver="arpack-cholmod"` for cases where an explicit small
σ is provided, but is not recommended for general use.

**GPU shift-invert via CuPy evaluated and rejected (2026-07-18, session 111,
`benchmarks/profile_eigenmode_shift_invert.py`):**
Measured on a vacuum PEC cube (k=9, RTX 4070 SUPER, 16-core host), stage-split
via an instrumented `OPinv` passed to scipy `eigsh`:

| Grid | n_free | assembly | SuperLU factor | ARPACK (33 solves) | factor share |
|------|--------|----------|----------------|--------------------|--------------|
| 30³ | 75,690 | 0.15 s | 63.7 s (fill 215×, 2.5 GB) | 4.9 s | 93 % |
| 40³ | 182,520 | 0.35 s | 629 s (fill 365×, 10 GB) | 15.6 s | 98 % |

Two independent rejection grounds:
1. cupyx `splu`/`factorized` compute the LU **on the CPU by design** (their
   docstrings state the decomposition is not GPU-accelerated; only the
   triangular solves run on device via cuSPARSE `spsm`).  The factorisation is
   93–98 % of total time → nothing that dominates can move.
2. The triangular solves that *do* move are **~74× slower on the GPU**:
   125 ms/solve CPU vs 9,247 ms GPU on the identical N=30³ factors —
   device-resident timing 9,246 ms, so transfers are irrelevant; the 215×-fill
   factors' sequential dependency chains defeat `spsm` level scheduling.
   Solutions agree to 5e-14; ARPACK end-to-end 4.5 s CPU vs 305 s GPU-OPinv.

The iterative route (GPU matvecs through the xp-agnostic CurlCurlOperator) was
not re-measured: it fails on this DD's already-measured convergence wall (no
Maxwell preconditioner), which a GPU does not change.  cuSOLVER `csrlsvqr`
(cupyx `spsolve`) refactorises per RHS — rejected by construction.

**Future scaling options** (not yet implemented):
1. PARDISO/MUMPS: symmetric indefinite LDLᵀ factorisation — drop-in replacement for SuperLU
   with ~2× speedup.  Requires pypardiso or python-mumps (not currently installed).
2. hypre AMS via PETSc: Maxwell-specific AMG, mesh-independent convergence.
   Heavy dependency (MPI, PETSc, hypre).
3. ~~GPU shift-invert via CuPy + CurlCurlOperator~~ — evaluated and rejected
   2026-07-18, see above.  The genuine GPU direct-sparse candidate would be
   NVIDIA cuDSS (multifrontal factorisation on device); no binding in our
   stack, unevaluated.

---

## DD-034 — Conformal Material Matrices

**Date:** 2026-04-12
**Status:** Accepted (Phase 1–4 implemented)

**Decision:** Replace staircase-only material filling with a unified cross-section pipeline
that (a) assigns cell materials via 2D point-in-polygon on OCC cross-section polygons and
(b) computes area-weighted effective ε_r on dual faces at material boundaries.

**Options evaluated:**
| Option                          | Notes |
|---------------------------------|-------|
| Cross-section polygon clipping  | Selected — O((Nx+Ny+Nz)·N_shapes) OCC calls |
| Per-cell point-in-shape (OCC)   | Previous approach — O(Nx·Ny·Nz·N_shapes), very slow |
| Sub-cell sampling / Monte Carlo | Inaccurate for thin features, no polygon reuse |
| Analytic surface integration    | Over-complex for general OCC shapes |

**Rationale:** FIT material matrices are integrals over dual/primal faces:
`M_eps[e] = (1/dl) · ∫_Ã ε(r) dA`.  For cells at material boundaries, the staircase
approximation (binary assignment) converges as O(h).  Area-weighted averaging of ε_r
over the actual material polygons on the dual face recovers O(h²) convergence.

The cross-section pipeline serves dual purpose: (1) cell classification via point-in-polygon
on cross-section polygons replaces `point_in_shape()`, cutting OCC calls from ~5M to ~1.5k
for a 100³ mesh; (2) polygon clipping against dual-face rectangles yields conformal fractions.

**Key design choices:**
- **NaN-sentinel dense array:** `ConformalData.effective_eps_r` is NaN where staircase applies,
  enabling vectorized `np.where(isnan, staircase, conformal)` without dict lookups.
- **Boundary-only computation:** `detect_boundary_cells()` identifies the 1–5% of cells at
  material interfaces; only their edges get conformal treatment.
- **Domain-boundary edges skipped:** Edges at j=0, j=Ny (etc.) are excluded because the
  `_avg_d()` convention uses full cell width at boundaries (image-charge method in FIT).
  Conformal values would be inconsistent with this convention.
- **PEC exclusion:** PEC area on a dual face is excluded from the ε integral (D=0 in PEC).
  Uncovered area defaults to vacuum (ε_r=1).  PEC enforcement via E=0 mask is orthogonal.
- **Courant safety cap:** Automatically reduced to 0.90 when `mesh.conformal is not None`,
  vs. the normal 0.95, to account for locally modified material coefficients.
- **Sutherland-Hodgman + Shoelace:** Pure NumPy polygon clipping (no Shapely dependency).

- **Overlap handling:** When shapes overlap (e.g. PEC brick + air cylinder), entries are
  processed in reverse order (last-wins area budget).  Each shape claims its clipped area
  from the remaining budget; earlier shapes only get the leftover.  This prevents
  double-counting that inflated mu_r_eff and degraded eigenfrequency accuracy.
- **M_mu conformal (Phase 5):** Area-weighted mu_r on primal faces at grid-node cross-sections.
  PEC is included (mu_r = 1.0).  Separate `batch_cross_sections()` call for node planes.

**Implementation:**
- `geometry/polygon_clip.py` — `polygon_area()`, `clip_polygon_to_rect()`, `point_in_polygon()`
- `mesh/conformal.py` — `ConformalData`, `PECSurfaceData`, `detect_boundary_cells()`,
  `extract_pec_surface()`, `ThinSheetSpec`, `detect_thin_metallizations()`
- `geometry/filling.py` — `classify_cells_from_cross_sections()`, `compute_conformal_eps()`,
  `compute_conformal_mu()`, `_area_weighted_eps()`, `_area_weighted_mu()` (reverse-order overlap)
- `geometry/occ_backend.py` — `batch_cross_sections()` added
- `mesh/mesher.py` — `Mesh.from_geometry()` refactored to unified cross-section pipeline
- `operators/material_matrices.py` — `build_M_eps()` / `build_M_sigma()` / `build_M_mu()` apply conformal overlay

- **PEC mask correction for conformal (Phase 7.1):** The staircase PEC mask uses a union rule
  (any adjacent cell PEC → edge PEC).  This is correct for staircase but too aggressive for
  conformal: boundary edges with partial air coverage (eps_r > 0) must carry E ≠ 0 — the
  reduced M_eps handles the boundary physics.  After conformal computation, edges with
  non-NaN eps_r > 0 are un-masked from PEC.  Edges fully inside PEC (eps_r = 0 or NaN with
  staircase PEC) remain masked.

**Verification:**

Cylindrical PEC cavity TM010 eigenmode (a=15mm, PEC walls):
- Staircase h=5mm: 8.40% error, h=4mm: 8.22% error
- Conformal h=5mm: 8.07% error, h=4mm: 7.87% error
- Conformal consistently closer to analytical f = 7.65 GHz

Coaxial line Z₀ via energy method (a=5mm, b=15mm, Z₀_ana=65.87 Ω):
- h=5mm (13³): Staircase 3.18%, Conformal 1.28% (2.5× improvement)
- h=2mm (15³): Staircase 2.95%, Conformal 0.66% (4.5× improvement)
- Tests M_eps directly without eigensolver; PEC mask correction essential for effect

---

## DD-035 — Thin Metallization Auto-Detection

**Date:** 2026-04-12
**Status:** Accepted

**Decision:** Automatically detect PEC shapes thinner than one grid cell along any axis and
model them as thin PEC sheets (zero-thickness E=0 planes) instead of volumetric fills.

**Options evaluated:**
| Option                        | Notes |
|-------------------------------|-------|
| Bounding-box heuristic        | Selected — simple, fast, works for axis-aligned sheets |
| Mesh-based detection          | Would require meshing first, then re-meshing — circular |
| User annotation               | Burdens users; easy to forget for PCB traces |

**Rationale:** Thin metallizations (e.g., 35 μm copper on PCB) are thinner than any practical
grid cell.  Without special treatment they either vanish (no cell centre inside) or get
staircase-approximated with wrong thickness.  Auto-detection compares the bounding-box extent
along each axis against the local minimum cell size; if `extent < min_cell_size`, the shape
is reclassified as a `ThinSheetSpec` and applied via `apply_thin_pec_sheet()`.

**Key design choices:**
- Detection runs before the cross-section pipeline; detected thin shapes are removed from the
  volumetric fill list.
- Only axis-aligned thin sheets are supported (first implementation).
- The thin sheet position is the midpoint of the bounding box in the thin direction.
- Transverse extent is preserved from the original shape's bounding box.
- The conformal pipeline sees the finite-volume effect of the thin conductor on adjacent dual
  faces (reduced capacitance where PEC area is excluded from ε averaging).

**Implementation:**
- `mesh/conformal.py` — `ThinSheetSpec`, `detect_thin_metallizations()`
- `mesh/mesher.py` — Hook before cross-section fill + `apply_thin_pec_sheet()` after mesh build

---

## DD-036 — Dey-Mittra conformal PEC edge shortening

**Date:** 2026-04-12
**Status:** Accepted

**Problem:** Conformal material matrices (DD-034) compute area-weighted effective ε_r on dual faces
at PEC boundaries. When PEC covers most of the dual face, ε_r_eff becomes very small (e.g. 0.05),
increasing the local wave speed and requiring a ~3× CFL penalty (dt *= sqrt(0.1)). Edges below
the threshold (CONFORMAL_EPS_THRESHOLD = 0.1) are kept PEC-masked, losing accuracy.

**Decision:** Implement Dey-Mittra edge shortening for PEC boundary edges.

**Approach:** For each E-edge at a PEC boundary, compute the fraction f_L of the primal edge length
that lies outside PEC. The modified material matrix entry becomes:

    M_eps_DM[e] = EPS0 · eps_r_eff · A_dual / (f_L · L_primal)
               = M_eps_conformal[e] / f_L[e]

Since the existing conformal pipeline already encodes the area fraction f_A in eps_r_eff, the only
new geometric computation is f_L (edge-PEC intersection via line-polygon scanning on existing
cross-section polygons at grid-node planes).

**Enlarged-cell technique:** When f_L < η (default 0.4), the short edge remnant would violate CFL.
Instead, the edge is treated as PEC (E = 0) and its dual-face contribution (eps_r_eff · A_dual)
is transferred to the neighbouring edge along the same direction. This guarantees stability with
the **standard CFL time step** — no penalty needed (dt ratio ≈ 3.16× improvement).

**Key design choices:**
- f_L is computed via 3D line-solid intersection (`compute_edge_pec_fractions()` using
  `BRepIntCurveSurface_Inter` + segment-midpoint `BRepClass3d_SolidClassifier`).
  The effective PEC solid respects last-wins CSG ordering via `build_effective_pec_solid()`.
  Supersedes the earlier 2D node-plane scan-line approach (more robust, no column-swap hack
  for Ez, handles arbitrary 3D PEC geometries).
- DM and conformal dielectric averaging coexist: DM handles PEC boundaries (modifies denominator),
  conformal handles dielectric boundaries (modifies numerator). At PEC-dielectric-dielectric triple
  points, both apply.
- Enlarged-cell neighbours are selected along the edge direction (not perpendicular), preferring
  the neighbour with the highest f_L.
- Static energy tests (e.g. Z₀ via analytical E-field) disable DM because they use full edge
  lengths as edge voltages, which is incompatible with the shortened-edge assumption.
- DM is controlled via `MeshControl.dey_mittra_eta` (default 0.4, in line with
  established industry practice). Setting 0 disables DM.
- M_sigma also receives the DM f_L correction (Ampère: both D and J affected). Enlarged-cell
  borrowing for sigma deferred until conformal sigma computation is implemented.

**Implementation:**
- `geometry/occ_backend.py` — `build_effective_pec_solid()`, `compute_edge_pec_fractions()`
- `geometry/filling.py` — `classify_edges()` (unified classification), `EdgeClassification`,
  `_pec_boundary_masks()`, `_enlarged_cell()`
- `mesh/conformal.py` — `ConformalData`, `DeyMittraData` dataclasses
- `mesh/mesher.py` — `classify_edges()` call in `from_geometry()`
- `operators/material_matrices.py` — `_apply_dey_mittra()` for M_eps,
  `_apply_dey_mittra_sigma()` for M_sigma
- `solver/stability.py` — `courant_dt()` with `min_effective_eps`

---

## DD-037 — 3D Face-Solid Intersection for Conformal Material Matrices

**Date:** 2026-04-13
**Status:** Accepted
**Supersedes:** DD-034 conformal eps/mu computation method (area-fraction via 2D cross-sections)

**Decision:** Replace the 2D cross-section polygon clipping pipeline for conformal
eps_r and mu_r with 3D thin-box Boolean intersection using `BRepAlgoAPI_Common`
and `BRepGProp.VolumeProperties`.

**Problem:** The previous approach computed area fractions (f_A) via 2D polygon
clipping on tessellated cross-section boundaries, while Dey-Mittra length
fractions (f_L) used 3D line-solid intersection via `BRepIntCurveSurface_Inter`.
These different computational paths introduced numerical inconsistencies
(tessellation chordal deflection, polygon clipping rounding) that could violate
the stability condition f_A/f_L >= 1, requiring a CFL time-step penalty.

**Method:** For each boundary dual/primal face:

1. Construct a thin box (face rectangle extruded by ±δ along normal axis)
2. `BRepAlgoAPI_Common(thin_box, material_solid)` → intersection volume
3. `area = volume / (2·δ)` — exact material area on the face
4. Materials processed in reverse priority order (last wins), same as before

The half-thickness δ is set to `max(h_min × 10⁻³, 10⁻⁶ m)` where h_min is
the minimum cell size — well above OCC's geometric tolerance (~10⁻⁷) and well
below any cell dimension.

**Rationale:**
- Both f_A and f_L now use the same OCC Boolean kernel on the same 3D solids
  → geometric consistency by construction
- No tessellation step → no chordal deflection error
- No polygon clipping → no floating-point accumulation
- Cell classification (material_id via point-in-polygon on cross-sections) unchanged

**Trade-offs:**
- `BRepAlgoAPI_Common` is slower than polygon clipping (~1.5 ms vs. ~1 μs per face)
- Acceptable: only boundary faces are processed (1–5% of mesh), total ~seconds

**Verification:** 458 tests passed (identical to pre-change), including:
- Cylindrical cavity TM010 eigenfrequency
- Coaxial Z₀ via energy method
- Dey-Mittra stability, convergence, and CFL tests

**Resolves:** KB-005

**Implementation:**
- `geometry/occ_backend.py` — new `compute_face_material_areas()`
- `geometry/filling.py` — rewritten `compute_conformal_eps()` (returns eps + sigma),
  rewritten `compute_conformal_mu()`, updated `classify_edges()` signature
  (`shapes_with_material` instead of `cross_section_cache`)
- `mesh/mesher.py` — updated `from_geometry()` pipeline
- Removed: `_area_weighted_eps()`, `_area_weighted_mu()` (2D polygon clipping helpers)

---

## DD-038 — GeometryModel overhaul: background material, overlap detection, multi-tool Difference, transform repeat/copy

**Date:** 2026-04-13  
**Session:** 39

**Context:** GeometryModel had several known limitations (KB-001, KB-002,
KB-004) that together hindered clean CSG-based geometry workflows.  The
"last wins" overlap semantics diverged from standard CAD practice and
masked geometry errors.

**Decisions:**

1. **Configurable background material (KB-001):**
   `GeometryModel(background=Material.pec())` sets the material for cells
   not covered by any shape.  Default remains air.  `Mesh.from_geometry()`
   reads `geometry.background` automatically.  The conformal pipeline
   (`compute_face_material_areas`) uses background properties for uncovered
   face area instead of hardcoded air.

2. **Even-odd rule for cross-section classification (KB-002):**
   `classify_cells_from_cross_sections()` now counts how many contours of
   a shape contain each cell centre.  A point belongs to the shape iff the
   count is odd.  This correctly handles Difference shapes with holes.
   The polygon extraction (`cross_section_polygons`) already returned both
   outer and inner contours; only the classification logic was wrong.

3. **Overlap detection with strict error default (KB-004):**
   `GeometryModel(allow_overlaps=False)` (default).  At meshing time,
   `validate()` is called automatically.  It checks all shape pairs via
   bounding-box pre-filter + `BRepAlgoAPI_Common` volume check.  Raises
   `GeometryOverlapError` on nonzero volumetric intersection.  Legacy code
   can pass `allow_overlaps=True`.

4. **Multi-tool Difference:**
   `Difference(base, tool1, tool2, ...)` — variadic `*tools`, matching
   Union's existing `*shapes` pattern.  Multiple tools are fused first
   (`boolean_union`), then a single `BRepAlgoAPI_Cut` is performed.
   Backward compatible: `Difference(base, tool)` still works.

5. **Transform repeat/copy (the interface convention common in major EM suites):**
   `translate(shape, vec, repeat=3, copy=True)` returns
   `[original, shape@1×vec, shape@2×vec, shape@3×vec]`.  `unite=True`
   returns a Union.  Same for `rotate`.  `GeometryModel.add()` extended
   to accept lists.

**Resolves:** KB-001, KB-002, KB-004

**Verification:** 480 tests passed, 1 skipped (up from 458, +22 new tests).

**Implementation:**
- `geometry/__init__.py` — `GeometryModel` with `background`, `allow_overlaps`,
  `validate()`, `GeometryOverlapError`, `add(list)`
- `geometry/operations.py` — `Difference` with `*tools`
- `geometry/transforms.py` — `translate`/`rotate` with `repeat`, `copy`, `unite`
- `geometry/filling.py` — even-odd rule in `classify_cells_from_cross_sections`
- `geometry/occ_backend.py` — `check_pairwise_overlaps()`, background-aware
  `compute_face_material_areas()`
- `mesh/mesher.py` — background from GeometryModel, automatic overlap validation

---

## DD-039 — ~~Dey-Mittra deactivated for eigenmode solver~~ → Superseded by DD-051 Variante A

DD-039 documented an ``apply_dm=False`` switch in the eigenmode solver
to compensate for a systematic Rayleigh-quotient downward bias under
the original ConformalData + DeyMittraData overlay (one-sided ``M_ε``
reduction without ``M_μ`` correction).  DD-051 Variante A replaced
that pipeline with a *symmetric* Krietenstein sub-cell correction on
both ``M_ε`` and ``M_μ``; the rotated-cavity benchmark that DD-039
recorded as 6.95–17.59 % conformal+DM error now lands at 0.22–1.63 %
on the same resolutions (11–32× improvement; ``p_obs = 1.64`` vs
DD-039's 0.9).  ``build_M_eps`` no longer accepts an ``apply_dm``
keyword and the eigensolver consumes the unified mass matrices
directly.

See ``validation/rotated_cavity_convergence.py`` for the
empirical re-measurement.

---

## DD-040 — Modal waveguide port operator as solver plugin

**Date:** 2026-04-27 (session 49, Phase-1 closeout of the modal-port rewrite)
**Status:** Accepted
**Supersedes:** DD-027 (modal Mur-ABC, TEM only) and DD-031 (PML-backed, TE/TM)

**Decision:** Replace the dual ``WaveguidePort`` (Mur-ABC for TEM, PML-backed
dual-E for TE/TM, runtime-dispatched via ``needs_pml``) with a single
``PortOperatorModal`` that runs as a solver plugin at the port plane.  Each
mode is absorbed independently via a per-mode 1st-order absorbing condition
applied **after** the Ampère update, on a 2D slice of the E-vector.

**Rationale:** DD-027 + DD-031 had three structural problems whose interaction
made the system brittle:

1. **Two completely different absorbers, dispatched at runtime.**  TEM-only
   ports used a port-face Mur-ABC; ports with any TE/TM mode used a
   CPML-backed soft-source pipeline plus dual-E ``(V₁, V₂)``-decomposition.
   The dispatch was driven by ``needs_pml = any(mode.omega_c > 0)``, so
   adding a single TE10 mode to a TEM coax fundamentally changed the
   absorber, the mesh extent (``Mesh.with_port_pml``), the recorder
   channel-name suffixes (``_total``/``_excite`` vs ``_V1``/``_V2``), and
   the S-parameter post-processing — all unobservable from the call site.
2. **PML behind the port doubled the cost of every TE/TM run.**  16 cells
   of CPML on each port face plus the dual-E monitoring planes (20-cell
   offset + 10-cell separation) added at least 30 cells per port to the
   axial mesh size, just to absorb modes that the per-mode 1st-order
   absorbing condition handles in the port plane itself.
3. **Voltage S-parameters needed a √(Z_exc/Z_obs) post-correction**
   (``transient.py:295``) because the modal projection used B-orthonormal
   profiles instead of power-wave-normalised ones.  This worked but
   couldn't be inlined into the recorder; it lived in the assembly stage
   alongside the dual-E ``(V₁, V₂)``-to-``(a, b)`` decomposition.

The modal operator solves all three at once: one absorber for every mode
type (per-mode ``e_p −= dt · (1/Z_m) · ê_m,p · V_m^-``), no PML extensions,
power-wave S-parameters from a single-plane V/I projection (DD-041 + DD-042).

**Algorithm (per mode m, per time step):**

1. **Pre-Ampère hook** (optional, currently unused): ``before_e_step(n, h)``
   reads ``H_t`` at the port plane.  Phase-1 operator does nothing here —
   I-recording happens after the E-update so V_m and I_m are on the
   solver's standard staggered-Yee schedule (V at integer time, I at
   half-integer).

2. **Standard FIT E-update** runs unchanged, producing candidate
   ``e_p^{n+1}`` at port-plane primal edges.

3. **Post-Ampère hook**: ``after_e_step(n, e, h)``:

   a. **Project** ``V_m(t^{n+1}) = Σ_p ê_m,p · e_p^{n+1}`` and
      ``I_m(t^{n+1/2}) = Σ_q ĥ_m,q · h_q^{n+1/2}`` for each mode.
   b. **Compute** the outgoing wave amplitude
      ``V_m^- = (V_m − Z_m · I_m) / 2`` (with Z_m the modal wave impedance
      from ``Mode.z_modal`` at the user-supplied mode-calc frequency).
   c. **Apply** the modal-Mur-1 correction to ``e_p^{n+1}``:
      ``e_p^{n+1} ← e_p^{n+1} − dt · (1/Z_m) · ê_m,p · V_m^-`` summed
      over modes.
   d. **Inject** the soft source (excited port only): legacy TF/SF
      formula ``V_port = s(t) + scat_int_prev + r·(scat_int_now −
      scat_face_prev)`` with ``scat = total − incident`` and ``incident``
      from a linearly-interpolated source-history ring buffer at delay
      ``τ_m = dx_n / v_p,m``.  See ``ports/modal/operator.py``.

4. **No standalone Mur on the port face**: in Phase 1 the operator alone
   absorbs at the port; the global ABC infrastructure (``CPMLBoundary``,
   ``MurBoundary``) is not used at port faces.

**Cross-cutting design rules:**

- The ``ModeSolver`` is a Protocol; Phase 1 ships
  ``CoaxAnalyticalModeSolver`` and ``RectWGAnalyticalModeSolver`` only.
  Both produce ``Mode``s sorted by ascending ``omega_c``.  Phase 2 will
  add ``Numerical2DModeSolver`` for arbitrary cross-sections (rectangular
  coax, microstrip, etc.) — the operator's contract is unchanged.
- Mode profiles are B-orthonormalised in the M_ε inner product
  (``eᵀ M_ε e = 1``) at port-attach time.  ``ĥ`` is rescaled per mode
  (DD-042 V/I calibration) so that ``V_m / I_m = Z_modal`` holds exactly
  for the analytical mode at the port plane and approximately (modulo
  discretisation) for the FIT-evolved field.
- The operator owns the source-history ring buffer; the user supplies
  only the waveform closure via ``ExcitationSpec``.

**Public API:**

- ``CoaxPortSpec`` / ``RectWGPortSpec`` / ``ExcitationSpec`` — frozen
  dataclasses describing a port in the global tangential frame.
- ``build_modal_port(spec, mesh, m_eps, m_mu, *, dt, f_calc) →
  PortOperatorModal`` — the only public factory.  Dispatches on spec
  type, normalises the (u, v)-frame at the bbox face, calls
  ``discretize_modes`` for B-orthonormalisation, builds the operator.
- ``FITTimeDomainSolver(port_operators=[...])`` — solver loop calls
  ``op.after_e_step(n, e, h)`` between the E-update and the H-update.

**Where:**

- ``src/magnelio/ports/_modal/{operator,coax,rect,solver,mode,port_plane,
  discrete,recorder,factory}.py``
- ``src/magnelio/solver/fit_td.py`` — adds ``port_operators`` field;
  loop hooks ``op.after_e_step`` after the E-update.
- ``src/magnelio/postprocessing/modal_sparameters.py`` —
  ``compute_s_parameters`` (see DD-042).

**Verified (Phase-1 close):** mode-solver unit ladder green; WR-90
TE10 ``|S21| ≤ 0.23 dB`` / ``|S11| ≤ −19 dB`` across 8.2–12.4 GHz;
cross-mode margin −260 to −310 dB (FFT floor).  The −19 dB Mur-1st
absorber limit was later removed by the exact DTBC chain
(DD-054/DD-055); the PML-backed absorber idea died in DD-043
(rejected).

---

## DD-041 — Single-plane V/I recording at port face; tuple-keyed recorders

**Date:** 2026-04-27 (session 49)
**Status:** Accepted

**Decision:** All port signal recording goes through dedicated tuple-keyed
recorder classes; the legacy generic ``SignalRecorder`` (channel-name-keyed,
``"port0_mode0_total"`` strings) is removed.  Two recorders cover the active
port families:

- ``ModalSignalRecorder``: one ``(V_signal, I_signal)`` pair per
  ``(port_label, mode_idx)`` channel.  V is sampled at integer time,
  I at half-integer (Yee stagger).  V-projection uses M_ε-weighted ê_m
  against e^{n+1}; I-projection uses M_μ-weighted ĥ_m against h^{n+1/2}.
- ``DiscretePortRecorder``: one ``Signal1D`` per ``(port_label, kind)``
  channel where ``kind ∈ {"total", "excite"}``.  ``"excite"`` is recorded
  only when ``record(...)`` is called with a non-``None`` ``excite=`` argument
  (passive ports get a ``"total"`` channel only).

**Rationale:** Three concrete problems with the previous channel-name-keyed
approach:

1. **String parsing in postprocessing.**  ``compute_sparameters`` had to
   parse keys like ``"port1_mode0_V1"`` to recover ``(port_idx, mode_idx,
   "V1")``.  Tuple keys eliminate the parser and the coupled string-format
   contract that ``fit_td.py`` had to keep in sync with ``transient.py``
   and ``sparameters.py``.
2. **Doubly-stored buffers on the port objects.**  ``DiscretePort`` held a
   ``_signals[0]`` list and an ``_excite_signal`` list internally and
   simultaneously emitted ``recorder.record("port0_mode0_total", ...)`` to
   the external recorder.  Tests asserted against ``port.signal`` directly,
   ignoring the recorder.  The duplication invited drift and made the
   recorder's role unclear.
3. **API asymmetry across port families.**  ``PortOperatorModal`` already
   had a tuple-keyed contract through ``project_V`` / ``project_I``;
   ``DiscretePort`` used string keys.  Two recorder shapes were needed for
   the same conceptual job (record a port's time-domain signals).

**Resolution:**

- ``ModalSignalRecorder`` (``ports/modal/recorder.py``): existing tuple-keyed
  modal recorder, kept as-is.  Constructor takes ``port_operators``,
  validates unique labels, allocates per-``(label, mode)`` V/I lists at
  construction.  ``record(e, h)`` is called once per FIT step from the
  solver, between ``after_e_step`` and the H-update.
- ``DiscretePortRecorder`` (``ports/discrete_recorder.py``, new in step 9b):
  takes ``dt`` and a list of unique ``port_labels``.  ``record(port_label,
  *, total: float, excite: float | None = None)`` appends a sample;
  ``finalize() → dict[(label, kind), Signal1D]``.
- ``DiscretePort`` gained a ``label`` constructor argument (default
  ``f"port{port_id}"``); the legacy ``_signals[0]`` / ``_excite_signal``
  buffers and the ``port.signal`` / ``port.signals`` / ``port.excite_signal``
  properties were removed.  ``update_bc`` returns
  ``tuple[float, float] = (v_total, v_exc)`` directly (no per-mode
  amplitude array — DiscretePort is single-channel by definition).
- Solver loop in ``fit_td.py`` calls
  ``recorder.record(port.label, total=..., excite=v_exc if is_excited else None)``
  per port per step (``ModalSignalRecorder`` is hooked separately as
  ``modal_recorder``).
- ``SignalRecorder`` (``signals/recorder.py``) deleted in full.  The
  ``signals/__init__.py`` re-export and the ``TestSignalRecorder`` unit-test
  class are gone.

**Net architectural impact:** Every recorder in magnelio now uses tuple keys;
channel-name string formatting is gone from the codebase; ``port.signal`` is
not a thing anymore.

**Where:**

- ``src/magnelio/ports/_modal/recorder.py`` — ``ModalSignalRecorder``.
- ``src/magnelio/ports/discrete_recorder.py`` — ``DiscretePortRecorder``
  (new).
- ``src/magnelio/ports/port_discrete.py`` — ``DiscretePort.label``,
  ``update_bc`` return shape, removed signal accessors.
- ``src/magnelio/solver/fit_td.py`` — recorder hook in the run loop.
- ``src/magnelio/io/hdf5.py`` — ``save_project`` no longer writes a
  per-port ``signal`` dataset (signals live in the recorder).

**Verified:**

- ``tests/unit/test_modal_recorder.py`` — 13 unit tests for
  ``ModalSignalRecorder`` (construction, projection, finalize, repr).
- ``tests/unit/test_discrete_recorder.py`` — 13 unit tests for
  ``DiscretePortRecorder`` (construction validation, total-only / total
  + excite paths, mixed-port routing, ``finalize`` ``Signal1D``
  properties, ``__repr__``).
- ``tests/integration/test_discrete_port.py`` — both pre-existing
  smoke tests pass on the new recorder API.
- ``tests/integration/test_modal_*`` — all green; modal recorder
  unchanged in behaviour.

---

## DD-042 — Power-wave S-parameters with √W excitation normalisation; per-mode V/I calibration

**Date:** 2026-04-27 (session 49)
**Status:** Accepted
**Supersedes:** the ``S_power = S_voltage · √(Z_exc/Z_obs)`` post-correction
  in the deleted ``TransientAnalysis.run`` (DD-031, archived).

**Decision:** S-parameter post-processing for the modal pipeline uses the
power-wave decomposition

```
a_jn(ω) = (V_jn(ω) / √Re(Z_jn) + √Re(Z_jn) · I_jn(ω)) / 2
b_km(ω) = (V_km(ω) / √Re(Z_km) − √Re(Z_km) · I_km(ω)) / 2
S_(km, jn)(ω) = b_km(ω) / a_jn(ω)
```

evaluated on the recorded V and I time-series with an explicit Yee-stagger
phase correction ``I_FFT(ω) ← I_FFT(ω) · exp(+jω·dt/2)`` (V is sampled at
integer time, I at half-integer; the correction lifts I onto the same
time axis in the frequency domain).

This replaces both the dual-E ``(V₁, V₂)``-to-``(a, b)`` decomposition of
DD-031 and the legacy voltage-to-power-wave ``√(Z_exc/Z_obs)`` correction.
``compute_s_parameters`` returns a raw
``dict[(port_label, mode_idx), np.ndarray]`` keyed identically to the
recorder; one call per excited ``(port, mode)``.

**Rationale:** Three reasons to bake the √W normalisation into the
amplitude formula at recording time, rather than as a post-hoc voltage
correction:

1. **One formula, all mode types.**  TEM, propagating TE/TM, and the
   limit cases (open / short / matched load) all fall out of the same
   ``a/b`` definition.  No per-mode dispatch, no special case for
   ``Z_pi`` vs ``Z_wave``.
2. **Power-wave passivity ``Σ |S_km|² ≤ 1`` directly testable.**  The
   integration-test suite (``test_coax_pec_confined_wave_arrival``,
   ``test_rectwg_te10_*``) asserts ``\|S\|²-sum ≤ 1.1`` as the
   architectural sanity bound; voltage-S would fail this at frequencies
   where Z_exc ≠ Z_obs even on a perfectly lossless network.
3. **Cutoff-band frequencies kept (established solver convention).**  Below cutoff,
   ``Re(Z_modal) → 0`` and the formula becomes ill-conditioned.  Phase 1
   guards only ``\|a_excited\|`` against the relative noise floor (NaN if
   below threshold) and lets the user inspect cutoff-band S values
   diagnostically rather than masking them.

**V/I calibration (per-mode ĥ rescale):**

The mode-discretisation step (``discretize_modes``) B-orthonormalises ``ê``
in the M_ε inner product and applies the same scalar α to ``ĥ`` to keep
the analytical ``E/H = Z`` field-level ratio.  But the M-weighted FIT
projections ``V_m = Σ M_ε · ê · e``, ``I_m = Σ M_μ · ĥ · h`` use *different*
material matrices, and the asymmetric weighting breaks
``V_m / I_m = Z_modal`` for the analytical mode at the port plane.

**Resolution:** ``PortOperatorModal._calibrate_v_i`` (called once during
``__init__``) rescales ``ĥ`` per mode by
``γ = (V_test / I_test) / Re(Z_modal)`` evaluated on the analytical
discretised mode at the port plane.  After this, ``V_m / I_m = Z_modal``
holds exactly for the analytical mode and approximately (modulo
conformal-cell effects, ~few % at coarse mesh) for the FIT-evolved
field.  Evanescent modes (``\|Im(Z_modal)\| > \|Re(Z_modal)\|``) are passed
through unchanged — a real scalar rescale cannot match an imaginary
impedance, and below cutoff ``a/b`` is diagnostic anyway.

**Trick exploited by the calibration:** the FIT identity
``M_μ[h_v at u-edge] · L_dual_for_h_v = μ₀ · normal_dx · L_primal_u``
(and the v-edge analogue) holds for any bbox-aligned port face, so
``I_test`` can be computed without exposing the (currently-private)
dual-edge lengths.

**Where:**

- ``src/magnelio/postprocessing/modal_sparameters.py`` —
  ``compute_s_parameters``.
- ``src/magnelio/ports/_modal/operator.py`` —
  ``PortOperatorModal._calibrate_v_i``.

**Verified:**

- ``tests/unit/test_modal_sparameters.py`` — 14 unit tests:
  open/short/matched-load limit cases, multi-port routing,
  cutoff-band sanity, threshold-driven NaN behaviour, validation.
- ``tests/unit/test_modal_operator.py::TestVICalibration`` — 2 tests:
  synthetic FIT-discretised TEM gives ``V_m/I_m = Z_modal`` to machine
  precision; ``mode.z_modal`` unchanged by rescale.
- ``tests/integration/test_modal_source_decay.py::test_coax_pec_confined_wave_arrival``
  — end-to-end ``\|S\|²-sum ≤ 1.1`` and ``max \|S21\| > −2 dB`` after
  calibration (raw measurement before calibration was ``\|S\|² ≈ 3.49``
  at some frequencies on the same setup).

---

## DD-043 — Out-of-Phase-1: late-time silence absorber (PML-backed) — REJECTED

**Status:** **Rejected** (sessions 49–54, multi-pilot diagnostic).  Do
not revive; higher-order Mur, port-side PML, fractional-delay BCs and
similar approximations are excluded as crutches (developer guidance,
session 54).

**Question.**  The IC-driven WR-90 silence test stalls at a hard
−7.7 dB energy floor: standing-wave content has zero net Poynting
flux at the port plane, so the DD-040 per-mode Mur-1st absorber
cannot drain it by construction (closed-form ``|r| ≈ 0.29`` at the
test point).  Hypothesis under test: a thin CPML slab behind the port
plane, since material loss drains energy density independently of
flux.

**Falsified by measurement.**  (1) The floor is
mode-solver-independent — analytical and numerical 2D modes give
floors identical to the digit (−7.68 dB at 25 and 80 traversals), so
it is not a projection artefact.  (2) Four session-54 pilots
(wave-packet bandwidth, ``n_modes`` 1–8, phase-velocity calibration,
naive per-mode Klein-Gordon aux-line) leave the source-driven −30 dB
|S11| floor unchanged; the naive aux-line is 7 dB *worse* (closed-loop
coupling).  The floor originates deeper than the boundary
approximation — loss volume behind the plane cannot reach it.

**Consequence.**  True co-simulation is the only path to
reference-grade port performance: per-mode auxiliary 1D lines sharing
the port-plane node (Luo-Chen 2007), captured as DD-047 and
ultimately realised exactly by the DTBC chain (DD-054/DD-055).

---

## DD-044 — Numerical 2D Mode Solver (curl-curl + Laplace dispatch)

**Date:** 2026-04-27 (session 53; closes Phase-2a sub-block).
**Status:** Accepted, partially implemented.  Phase-2a delivered the
TE/TM curl-curl path; the TEM Laplace and QTEM dispatch land in
Phase 2b (the architecture-document §2.2 three-class deliverable).

**Decision:** Establish ``Numerical2DModeSolver`` (in
``src/magnelio/ports/_modal/numerical_2d.py``) as the second
implementation of the ``ModeSolver`` Protocol from DD-040, alongside
the analytical solvers.  The solver uses the FIT-consistent 2D
operators built by ``build_2d_curl_curl``
(``src/magnelio/ports/_modal/curl_curl_2d.py``) and produces
``Mode`` objects on the Phase-2 numerical path (``field_evaluator =
None``, four ``discrete_*_profile`` arrays filled).

**Eigenvalue formulation** (architecture §2.1).  Solve

    K · ê = ω_c² · M · ê

with ``K = C̃_2D · M_μ⁻¹ · C_2D`` (sparse SPD-with-null-space) and
``M = M̃_ε`` (sparse diagonal), via
``scipy.sparse.linalg.eigsh(K, k=n_total, M=M, sigma=σ)``.  σ is
chosen by a heuristic just below the lowest expected TE cut-off
(``σ ≈ 0.95² · (π · c_eff / L_max)²``) so the gradient null-space at
λ = 0 is suppressed but the lowest physical eigenmodes are inside
the shift-invert window.  A positive threshold filters surviving
null-space contamination; eigenvectors are then sorted by ascending
``λ = ω_c²``.

**Why generalised, not standard.**  ``A · ê = ω_c² · ê`` with
``A = M̃_ε⁻¹ · C̃_2D · M_μ⁻¹ · C_2D`` requires materialising
``M̃_ε⁻¹`` densely and loses the symmetric-positive structure that
ARPACK's shift-invert exploits.  The generalised form keeps both
operators sparse and symmetric.

**Sign convention** (architecture §2.4 / Reference §8.1).  After
each ``eigsh`` call, every eigenvector is sign-flipped so that its
largest-magnitude entry is positive.  Without this, the gauge is
machine-dependent and breaks inter-port phase consistency.  The
sign-flip is bilinear-invariant on the operator-level
``(E × H) · n̂`` invariant: H-profiles are derived from the modal
Ohm's law (``H_u = -E_v / Z``, ``H_v = +E_u / Z``), so a sign-flip
on E flips H consistently and Poynting orientation is preserved.

**Phase-2b extensions.**  The architecture-§2.2 three-class
dispatch (TEM Laplace, TE/TM curl-curl, QTEM single-frequency
curl-curl) is staged across Phase 2a (TE/TM only) and Phase 2b
(TEM and QTEM); see the architecture document §5 for the work order.

**Where:**
``src/magnelio/ports/_modal/numerical_2d.py`` and
``src/magnelio/ports/_modal/curl_curl_2d.py``.  Tests:
``tests/unit/test_modal_numerical_2d.py`` (21 unit tests),
``tests/unit/test_modal_curl_curl_2d.py`` (16 unit tests).

---

## DD-045 — Mode Dataclass Extension for Discrete Edge Profiles (Variant B)

**Date:** 2026-04-27 (session 53; closes Phase-2a sub-block).
**Status:** Accepted.

**Decision:** Extend the existing ``Mode`` dataclass
(``src/magnelio/ports/_modal/mode.py``) with four optional
``discrete_*_profile`` arrays carrying the per-edge eigenvector
data of a numerically-solved mode.  ``field_evaluator`` is made
optional in the same change.  The validity invariant — exactly
one of (``field_evaluator``, all four ``discrete_*_profile``)
populated — is enforced in ``__post_init__``.  ``discretize_modes``
dispatches on the populated path: analytical → existing
Gram-Schmidt; numerical → pass-through (no resampling, no
re-orthonormalisation, since the numerical solver already produces
M_ε-orthonormal vectors by construction).

**Why Variant B (extend ``Mode``) over Variant C (parallel
``DiscreteModeSolver`` Protocol).**  The factory, operator,
recorder, and S-parameter pipeline treat the resulting
``DiscreteMode`` identically — the only Phase-1-vs-Phase-2 fork is
*how* a ``Mode`` is filled, not *what* downstream code does with it.
A parallel Protocol would force every consumer to dispatch with no
clarity gain.

**Mixed-list policy.**  ``discretize_modes`` rejects lists that mix
analytical and numerical modes.  The two paths have incompatible
inner-product conventions — running Gram-Schmidt on a mixed list
would orthogonalise the analytical modes against the numerical ones
in list order, silently corrupting the numerical modes' native
orthonormality.

**Where:**
``src/magnelio/ports/_modal/mode.py`` (``Mode.__post_init__``
invariant), ``src/magnelio/ports/_modal/discrete.py``
(``discretize_modes`` dispatch + ``_discretize_numerical``
pass-through helper).  Tests:
``tests/unit/test_modal_mode.py::TestModeValidityInvariant``
(6 tests),
``tests/unit/test_modal_discretize.py::TestDiscretizeNumericalPassThrough``
(6 tests).

---

## DD-046 — Read-from-3D-Mesh Geometry for Numerical Port Specs (Phase-2a hybrid)

**Date:** 2026-04-27 (session 53; closes Phase-2a sub-block).
**Status:** Accepted with Phase-2a / Phase-2b split.  Phase-2a
implements an explicit ``lateral_pec_faces`` field on
``NumericalPortSpec``; Phase-2b will add the full read-from-mesh
path of architecture-document §2.6 for TEM/QTEM cross-sections.

**Decision (Phase-2a):** ``NumericalPortSpec``
(``src/magnelio/ports/_modal/factory.py``) describes a numerical-
mode-solver port via ``label`` / ``plane`` / ``n_modes`` /
``epsilon_r`` / ``mode_type`` / ``lateral_pec_faces:
tuple[BoxFace, ...]`` / ``excitation``.  ``build_modal_port``
dispatches the new spec type to the numerical path: builds
``K``, ``M``, ``primal_2d_indices`` via ``build_2d_curl_curl``,
constructs the lateral-PEC edge mask via the private helper
``_build_lateral_pec_edge_mask`` from ``lateral_pec_faces``, runs
``Numerical2DModeSolver``, and feeds the result through the
existing ``discretize_modes`` pass-through and
``PortOperatorModal`` build.

**Why a hybrid instead of architecture-§2.6 read-from-mesh.**
Architecture §2.6 specifies ``conductor_mask`` and
``epsilon_r_field`` as 3D-mesh-read inputs (cell arrays).  That
design assumes the conductor pattern is mesh material — correct
for TEM/QTEM cross-sections where the conductor *is* a mesh region
(rectangular coax, microstrip), but Phase-2a's primary use case
(WR-90 hollow waveguide) carries PEC walls via
``boundary_conditions={ymin: PECBoundary, ...}`` rather than
material-tagged cells.  The factory has no mesh-side conductor
mask available in that case.  Phase-2a therefore lands the
explicit ``lateral_pec_faces`` field; Phase-2b will add a
``conductor_groups`` argument that auto-detects from
``mesh.material_id`` + ``mesh.material_library[…].is_pec`` for
TEM/QTEM.

**PortOperatorModal calibration extension.**
``PortOperatorModal._calibrate_v_i`` is extended to handle the
numerical path: when ``mode.field_evaluator is None`` the four
``discrete_*_profile`` arrays are used directly as the per-edge
test fields (the only "test" available for a numerical mode).
DD-042's V/I calibration mechanism is unchanged otherwise — the
γ rescale logic still produces ``V_m / I_m = Z_modal`` to
discretisation accuracy.

**Where:**
``src/magnelio/ports/_modal/factory.py`` (``NumericalPortSpec`` +
``_build_lateral_pec_edge_mask`` + ``build_modal_port`` numerical
branch), ``src/magnelio/ports/_modal/operator.py``
(``_calibrate_v_i`` numerical-path fork).  Tests:
``tests/unit/test_modal_factory.py::TestNumericalPortSpecFactory``
(5 tests),
``tests/unit/test_modal_factory.py::TestLateralPecEdgeMaskHelper``
(2 tests).

---

## DD-047 — Phase 3: true co-simulation modal port operator (Luo-Chen) — realised by the DTBC chain

**Status:** Accepted session 54 as the firm long-term direction;
**goal reached** by the exact DTBC chain (DD-054/DD-055) — the
ghost-relation convolution is the exact closed form of the
semi-infinite per-mode auxiliary line Luo-Chen 2007 couples in, so
the co-simulation exterior is emulated without literal 1D aux-line
state.  Measured endpoint: rect-coax max in-band |S11| = −159.3 dB
(``tests/integration/test_rect_coax_sparams.py``).

**Problem (then).**  The DD-040/DD-042 modal Mur-1st port floors at
peak |S11| ≈ −19 dB near cutoff on hollow WR-90, mesh-independently —
inadequate for filters, sensitive matching and high-Q components.
Reference-grade solvers reach the −160 dB class on the same sanity
setup (Thoma 2019).

**Explicit non-paths (session-54 falsifications — do not revive).**
1. Phase-velocity calibration in the Mur formula: ~0.05 dB.
2. ``r = 0`` by construction (``v_p·dt = dx_n``): +1.7 dB — Mur
   reflection is not the dominant factor in-band.
3. Higher-order Mur / Higdon-N: cannot absorb evanescent content
   (Luo-Chen 2007); refuted again TD-measured in DD-069.
4. Naive modal aux-line as a post-update hook: −7 dB *worse*
   (closed-loop coupling unstable) — the coupling must be part of
   the update, which is what the DTBC formulation provides.

**Measurement-artefact findings absorbed elsewhere.**  The in-band
floor away from cutoff was largely the spatial half-cell stagger of
the I sampling plane (WP1 de-stagger, ~6 dB median) plus the
pointwise ``h = ±e/η`` H-voltage convention on graded transversal
meshes — resolved by the travelling-wave profiles + M-metric V/I
calibration (DD-052; graded parallel plate −23 → −64.1 dB).  Graded
transversal port meshes are first-class (regression-pinned); the
one-time "prefer uniform transversal spacing" guidance is withdrawn.
Re-measurement record: ``validation/wr90_te10_dd047_reeval.py``.

---

## DD-048 — Modal port pipeline: two-path architecture

**Date:** 2026-04-29 (session 58).
**Status:** Accepted.  Supersedes DD-029.  Refines DD-040 / DD-042 /
DD-044 / DD-046 by clarifying which mode object drives the FIT-TD
update versus user-visible reporting.

**Decision.**  Per modal port and per mode index, the pipeline
computes **two independent modes**:

- **Reference mode** (`mode_ref`) — mesh-independent.  Source: a
  closed-form analytical solver
  (``CoaxAnalyticalModeSolver``, ``RectWGAnalyticalModeSolver``)
  where one is available, otherwise ``solve_modes_refined``
  (adaptive 2D mesh refinement, Cleanup 3 of session 56).  Drives
  the user-visible reporting fields (``z_line_ref``, ``cutoff_ref``,
  ``beta_ref``).  **Does not** drive the FIT-TD coupling.

- **Operator-consistent mode** (`mode_num`) — solved on the 2D
  transversal slice of the 3D simulation mesh.  Source:
  ``solve_tem_laplace`` / ``solve_qtem_laplace`` (TEM/QTEM via 2D
  Laplace and dual-Laplace) or ``Numerical2DModeSolver`` (TE/TM via
  2D curl-curl).  Drives the ``PortOperatorModal`` excitation and
  loading and the S-parameter extraction.  **Mandatory** path: every
  spec must produce a ``mode_num`` even when an analytical reference
  is available.

The two modes converge as the 3D mesh is refined.  Their persistent
difference at the user's working mesh is *physical*, not a defect:
it is the discretisation gap between the continuous geometry and
its staircased FIT approximation.  Both numbers are correct in their
own context.

**Why two paths instead of one.**

- *Path (a) only* (analytical, projected to FIT slice).  The
  analytical mode is not an eigenmode of the staircased FIT operator;
  injecting it as the source mode produces a mode-orthogonal residue
  that the volume operator partially propagates as a mode-mismatched
  wave.  Observed in session 58 on
  ``examples/notebooks/straight_coax.ipynb`` (D_i = 0.41 mm,
  D_a = 5 mm, ε_r = 9, 27³ mesh): max |S11| ≈ +0.2 dB at 0.25 GHz,
  max |S|² ≈ 3.0 across [0.25, 10] GHz.  This was the historical
  ``CoaxPortSpec`` behaviour.

- *Path (b) only* (operator-consistent, no reference).  Z_line at
  the user's working mesh deviates from the analytical value by a
  few percent on staircased radial geometry (47.93 Ω measured vs
  50.0 Ω analytical for the notebook coax at 27³).  Without a
  reference, the user cannot tell whether 47.93 Ω is "correct".  It
  *is* correct for that mesh, but the user usually wants the
  continuous-geometry design target as well.

- *Path (a) + (b).*  Both numbers are reported; FIT-TD uses (b);
  Δ(Z_line) is informational, not a warning trigger.

**No Δ-warning.**  An automatic warning on
``|Z_ref - Z_num| / Z_ref > τ`` would suggest that one of the two
values is wrong.  Both are correct.  A user pursuing accuracy
refines the 3D mesh until Δ shrinks; ``solve_modes_refined``
provides per-level convergence data when that is needed.  Forcing a
default warning would clutter the reporting and train users to
ignore it.

**API impact.**

- ``PortOperatorModal`` gains a frozen ``port_report: PortOperatorReport``
  attribute populated by ``build_modal_port``.  Fields:

  ```
  z_line_num     : float           # always populated (Path b)
  z_line_ref     : float | None    # Path a, when available
  cutoff_num     : float | None    # TE/TM only
  cutoff_ref     : float | None    # TE/TM only
  refinement_log : ModeRefinementReport | None
                                   # numerical specs with
                                   # reference_refinement > 0
  ```

- ``CoaxPortSpec`` and ``RectWGPortSpec`` now build *both* paths.
  The analytical solver continues to fill the reference fields; the
  FIT slice produces the operator-consistent mode that drives the
  operator.  User-API surface is unchanged.

- ``RectWGPortSpec`` falls back to "all four lateral bbox faces are
  PEC" when the mesh PEC mask on the port slice is empty.  This
  covers test setups that build a bare grid with ``Mesh.from_grid``
  and do not run OCC geometry; production OCC runs always have a
  populated PEC mask.  ``CoaxPortSpec`` does **not** offer a
  fallback — its two-conductor topology has no sensible default.

- ``NumericalPortSpec`` and ``MultiConductorPortSpec`` keep their
  current API.  Their ``port_report`` carries ``z_line_num`` /
  ``cutoff_num`` (path b only) with the reference fields ``None``.
  Users who want a reference value invoke
  :func:`solve_modes_refined` separately — that function needs a
  ``GeometryModel`` and a ``MeshControl``, which the factory does not
  see — and overwrite ``op.port_report`` with the augmented record.
  Auto-invocation from inside the factory was rejected to keep the
  spec API geometry-free per DD-046.

- ``CoaxPortSpec`` / ``RectWGPortSpec`` validate at construction
  time that the auto-detected conductor centroids on the 3D mesh
  slice agree with ``spec.center`` to within one mesh cell.
  Otherwise ``ValueError`` — typical cause: mesh too coarse or
  geometry displaced from spec.

**Relation to DD-047.**  Bug-5 fix + DD-048 dropped the TEM |S11|
floor from ≥ −20 dB to ≤ −31 dB; the residual parallel-plate floor
was later identified as the spatial I-stagger measurement artefact
(WP1 de-stagger, session 68 → −71.9 dB; see DD-047/DD-052).

**Where.**

- ``src/magnelio/ports/_modal/factory.py`` — two-path dispatch per
  spec.
- ``src/magnelio/ports/_modal/operator.py`` — ``PortOperatorReport`` dataclass
  and operator field.
- ``src/magnelio/ports/_modal/__init__.py`` — public exports.

---

## DD-049 — Tangentialpunkt-Bug: background-PEC not represented in OCC effective PEC solid

**Date:** 2026-04-29 (session 58); **resolved** session 61; the
bbox-wall half **superseded by DD-103** (session 136).

**Problem.**  With ``GeometryModel(background=pec)`` and a dielectric
tangent to the bbox (coax with bbox = ``D_a × D_a × L``), the
background-PEC region was invisible to the OCC effective-PEC solid:
DM-active edges got ``conformal_eps = 0`` → ``min_effective_eps = 0``
→ dt clamped ×1000 → NaN at step 0, plus a fragmented bbox-wall PEC
mask that broke auto-conductor detection (Z_line +26 %, divergent
under refinement).  Historically hidden by a ``1.2 × D_a``
PEC-bbox-brick crutch in the coax notebook.

**Fix** (``mesher.py::Mesh.from_geometry``, two coordinated changes):
1. Synthesize an explicit bbox-sized PEC ``Brick`` at lowest priority
   before ``classify_edges`` / ``build_effective_pec_solid`` — the
   effective-PEC solid then includes the background region
   (``f_L = 0`` on embedded edges).
2. OR-in the bbox-face tangential E-edges via
   ``_or_in_bbox_pec_walls`` (shared with ``with_pec_boundaries``) so
   isolated tangency points cannot fragment the wall's connected
   component.  *This blanket per-face version silently overrode PMC
   and CPML faces — replaced by the closure-driven per-face rule of
   DD-103; read DD-103 before touching the wall mask.*

Clean coax geometry then gives ``Z_line_num ≈ 48 Ω`` (analytical
50 Ω, matches the retired crutch) and runs FIT-TD without NaN.

**Session-60 re-analysis.**  The suspected "broadening" to the
round-WG mantle was benign: those ``conformal_eps = 0`` edges are
mathematically correct (dual face fully embedded in PEC) and
correctly masked.  The round-WG TE11 accuracy gap traced to a
separate conformal-mass coupling issue (→ DD-051).

**Diagnostic scripts (session 58, kept):**
``validation/coax_transversal_bbox_pilots.py``,
``validation/coax_translation_invariance_pilot.py``,
``validation/coax_force_bbox_wall_pilot.py``.

---

## DD-050 — NumericalPortSpec consolidates on ``mesh.pec_mask_edges`` (DD-046 completion)

**Date:** 2026-04-29 (Session 60).

**Status:** Decided.

**Why this surfaced.**  When extending the TE/TM stress-test work
order (Session 59) from the rectangular hollow waveguide to the
round hollow waveguide with PEC bbox padding, the
``NumericalPortSpec`` factory branch was found to read PEC-wall
information *only* from a ``lateral_pec_faces: tuple[BoxFace, ...]``
field on the spec.  This is the pre-DD-046 path: the user enumerates
bbox faces declaratively rather than letting the 3D mesh be the
single source of truth.  For a round hollow waveguide the PEC wall
is a curved cylindrical contour — *not* describable as a tuple of
bbox faces — so ``NumericalPortSpec`` simply could not represent the
geometry.

DD-046 (Session 55, Cleanup 2) had already declared
``mesh.pec_mask_edges`` the canonical source of all PEC information;
DD-046 extended the ``Mesh`` API with ``Mesh.with_pec_boundaries``
to consolidate ``BoundaryConditions(<face>="PEC")`` into the same
mask.  ``RectWGPortSpec`` (Session 58, DD-048) and the two
``MultiConductor`` / ``Coax`` paths followed DD-046 fully.
``NumericalPortSpec`` did not — it was the last remaining island
of the old declarative-PEC API.

**Decision.**

The ``NumericalPortSpec.lateral_pec_faces`` field is **removed**.
Both factory branches that drive ``Numerical2DModeSolver``
(``RectWGPortSpec`` for hollow-rectangle TE/TM and
``NumericalPortSpec`` for arbitrary hollow-cross-section TE/TM) read
PEC information through a single internal helper
``_resolve_pec_edge_mask(plane, mesh)`` that:

1. slices ``mesh.pec_mask_edges`` onto the port plane's
   ``[e_u | e_v]`` 2D-edge basis (canonical DD-046 path);
2. falls back to "all four lateral bbox faces are PEC" when the
   sliced 2D mask is empty (typical for bare ``Mesh.from_grid``
   setups without OCC and without ``Mesh.with_pec_boundaries``).

The ``_build_lateral_pec_edge_mask`` helper is retained as an
implementation detail used only by the fallback path.  It is no
longer reachable from the public API except by direct import for
unit tests.

**TM-path consistency.**

The TM-path scalar Laplace operator
(:func:`build_2d_node_laplace`) gains an optional
``pec_edge_mask: np.ndarray | None`` parameter.  When provided, the
returned ``pec_node_mask`` is derived from the edge mask via
``node_pec[k] = any(edge_pec[e] for e ∈ incident_edges(k))`` — the
discrete form of the TM-mode E_z = 0 boundary condition on every
PEC wall (curved OCC contours, straight bbox walls, internal PEC
inclusions).  Without the parameter, the helper falls back to its
prior ``_hollow_pec_node_mask`` default (all four bbox-boundary
node rows), correct only for hollow-bbox setups.  Both factory TM
branches now thread the edge mask through.

**API impact.**

- ``NumericalPortSpec(lateral_pec_faces=...)`` no longer accepts
  the field; users who previously declared lateral PEC walls must
  migrate to ``BoundaryConditions(<face>="PEC")`` +
  ``Mesh.with_pec_boundaries(...)`` or to a real OCC body via
  ``Mesh.from_geometry(...)``.  Both paths populate
  ``mesh.pec_mask_edges`` automatically and the factory picks the
  result up.

- ``build_2d_node_laplace`` gains the optional ``pec_edge_mask``
  keyword.  Internal helper; no breaking change for direct callers
  who passed positional arguments only.

**Why eliminate rather than deprecate.**

``MAJOR = 0`` per the project's semver policy explicitly permits
API breakage.  A deprecation shim would have re-introduced the
parallel-PEC-source pathology DD-046 was meant to eliminate; that
is the exact opposite of what "consolidate" means in this context.

**Where:**

- ``src/magnelio/ports/_modal/factory.py`` — ``NumericalPortSpec``
  field removal, ``_resolve_pec_edge_mask`` helper, both factory
  branches simplified.
- ``src/magnelio/ports/_modal/curl_curl_2d.py`` —
  ``build_2d_node_laplace`` gains the ``pec_edge_mask`` parameter.
- ``tests/unit/test_modal_factory.py`` — ``TestNumericalPortSpecFactory``
  migrated to bare-mesh fallback path; new
  ``test_pec_mask_consolidated_via_with_pec_boundaries`` asserts
  the canonical DD-046 path matches the fallback.
  ``TestLateralPecEdgeMaskHelper`` rewritten as direct unit tests
  on the internal helper.

---

## DD-051 — Sub-cell classifier reformulation: unified (ε̄, A_free, L_free) per edge

**Date:** 2026-04-29 (session 62, Variante B + Variante A).
**Status:** **Implemented (Variante A complete).**
*Session-82/83 update (DD-053):* on H faces with a unique locally
translation-invariant ladder direction the Krietenstein
``A_face_free`` value is replaced at meshing time by the LC-consistent
pair value; Krietenstein remains the correction on genuinely 3D
contours.  *Session-91 update (DD-058):* the enlarged-cell H-face
donor follow-up is implemented, measured neutral to machine
precision, and dormant.

**Problem.**  Round-WG TE11 cut-off converged **non-monotonically**
under ``conformal=True`` (−8.87 → −4.34 → −4.45 → −3.02 % across
``n_t ∈ {17, 25, 33, 49}``; ``validation/round_wg_conformal_convergence.py``)
and stayed ~3× worse than its design target — the signature of a
structural defect, on exactly the geometry class (curved PEC) the
conformal path exists for.

**Root cause.**  The old pipeline composed two override stages on
``M_ε``: conformal ε averaged over the *free* dual-face area, then
applied against the *full* area; Dey-Mittra dividing by ``f_L``
afterwards.  Orthogonal on straight walls (disjoint edge classes),
double-counting on curved PEC where both stages hit the same edges.
Additionally — found by the Variante-A diagnostics — the *magnetic*
side carried no sub-cell correction at all: the primal-face-area
shrinkage on PEC-boundary H-faces was missing, over-estimating
boundary magnetic inertia (direction matches the measured 8.0 vs
8.79 GHz cut-off).  Two supporting diagnostics: OCC-pipeline
divergence empirically falsified (``Δf_L = Δε̄ = 0`` vs a 1000×-finer
tessellation, ``validation/round_wg_subcell_diagnostic.py``);
enlarged-cell threshold a contributor but not the cause
(``validation/round_wg_eta0_test.py``).

**Decision.**  One single-pass sub-cell classifier per E-edge
producing a category plus the triple ``(ε̄, A_free, L_free)`` from one
OCC pass against the shared ``pec_solid`` snapshot
(``EdgeMaterialData``; ``_apply_dey_mittra`` ceased to exist), with
the mirrored construction for H-faces (``FaceMaterialData``):

| Category | Condition | M_ε (E) / M_μ (H) |
|---|---|---|
| 0 interior bulk | homogeneous dual face | ``ε_r · A_dual / L_primal`` |
| 1 dielectric boundary | crosses dielectrics, no PEC | ``ε̄ · A_dual / L_primal`` |
| 2 curved-PEC sub-cell | crosses PEC | ``ε̄ · A_free / L_free`` |
| 3 interior PEC | fully inside PEC | masked (E only) |

A 1 % ``A_face_free`` floor keeps ``1/M_μ`` finite (falls back to
bulk staircase); ``courant_dt`` reads both ``min_effective_eps`` and
``min_effective_mu``.

**Measured (Variante A, round-WG TE11 cut-off, analytical 8.79 GHz):**

| n_t | rel.err staircase | rel.err conformal |
|---:|---:|---:|
| 17 | −2.86 % | **−0.45 %** |
| 25 | −1.69 % | **+0.10 %** |
| 33 | −1.16 % | **−0.22 %** |
| 49 | −0.91 % | **−0.21 %** |

Sub-percent on every resolution, 4–6× better than staircase.  The
~0.3 % oscillation around the analytical value (instead of monotone
``p_obs ≥ 1.8``) is the known cat-1/cat-2 classification-flip
signature of sub-cell methods on structured grids; the envelope is
the practical accuracy claim.  The rotated-cavity benchmark
(``validation/rotated_cavity_convergence.py``) re-measured
0.22–1.63 % vs 6.95–17.59 % under the old overlay (closes the DD-039
revision).  Eigensolver uses the unified pipeline directly — no
``apply_dm`` switch survives.

**Reference.**  Krietenstein, Schuhmann, Thoma, Weiland, LINAC'98 —
the per-edge sub-cell triple this DD reproduces (terminology note:
in-project we say "conformal material matrices").

---

## DD-052 — Exact discrete travelling-wave port profiles (pair-consistent launch/measurement chain)
**Status:** Decided ~sessions 80/81 (WP7.1/WP7.2); cited by code,
tests and validation scripts from the start, but the entry was never
authored — reconstructed from those citations in session 147.
**Decision.**  Port H profiles are the dual voltages of the *exact
discrete* travelling wave, not pointwise continuum samples: per
co-located (e, h) pair, ``h ∝ e_partner / M_μ`` with overall scale
``μ0·normal_dx / Z``
(``ports/_modal/tem_laplace.travelling_wave_h_profiles``).  On a
z-uniform feed section the modal amplitude then obeys the exact 1D
leapfrog Klein–Gordon chain whose modal Courant number is the
co-located pair product ``r = dt / sqrt(M_eps[e] · M_mu[h_partner])``
(pair identity ``M_eps · M_mu = εμ · dz · dz~`` on the flattened
port slab).
**Rationale.**  The pointwise convention ``H = ±E/Z`` is exact for
*fields* but wrong for FIT face *voltages* on non-uniform transversal
grids (WP7.1 spike, session 80); the dual-voltage form stays a
faithful travelling-wave probe on graded meshes and is the chain the
exact DTBC (DD-054/DD-055) terminates.
**Measured.**  Rect-coax broadband match −34.6 → −39.8 dB max |S11|
on the 0.8 mm mesh when these profiles replaced pointwise sampling
(later −159.3 dB with the exact DTBC;
``tests/integration/test_rect_coax_sparams.py``).
**References.**  ``validation/straight_coax_conformal_pair_diag.py``,
``validation/coax_pair_consistent_mu_spike.py``; DD-042 (V_m/I_m
calibration), DD-053 (conformal M_μ pairing, same pair identity).
---
## DD-053 — LC-consistent conformal M_μ coupling + tangential-surface masking

**Date:** 2026-07-09 (session 82 spike + session 83 production).
**Status:** **Implemented.**  STATUS construction site 0 closed;
follow-up of DD-051/DD-052.

**Why this surfaced.**  After DD-052, the straight_coax demo retained
a −32.6 dB |S11| floor that staircase meshes of the *same* resolution
beat by >12 dB (at 10 % worse z_line).  Session-81 diagnosis: on
conformal cross-sections the E-edge (ε, DD-051 cat 2) and Krietenstein
H-face (μ, DD-051 Variante A) sub-cell corrections are computed
independently, breaking the co-located pair identity

    M_ε[edge] · M_μ[face] = ε0μ0 · ε_pair · μ̄ · d · d̃

on which the DD-052 exact discrete TEM travelling wave rests
(profile-weighted pair spread 3.1 %).

**Spike findings (session 82,**
``validation/coax_pair_consistent_mu_spike.py``**).**  Two
independent defects, BOTH must be fixed:

1. Restoring the pair identity alone (μ override) drives the spread
   to exactly 0 and — because ``1/M_μ`` becomes ∝ ``M_ε`` per pair —
   turns the free Ez rows of ``C^T M_μ^{-1} C`` into the *same*
   weighted Laplacian the 2D TEM solve zeroes (residual 1.2 → 1e-13),
   yet the TD floor does not move (−32.5 → −32.8 dB).
2. The remaining wave-equation violation sits exclusively on
   *conductor-footprint nodes*: the 2D mode solve treats them as
   Dirichlet (surface charge), but the conformal classifier leaves
   their longitudinal edges unmasked (line-solid ``f_L = 1`` — they
   run parallel to the contour, the classifier cannot see the
   tangentiality), so the 3D update evolves a surface-tangential E
   that no transversal port profile can launch or record.  Masking
   them makes the lifted travelling wave exact on ALL free rows to
   machine precision.

Dead end recorded: replacing the φ-Laplace profile by a "discrete
harmonic" form has no discrete null space — after the μ pairing the
free rows already vanish for the Laplace profile; the conductor-node
rows are legitimately nonzero.  The profile was never the problem.

**Physical picture.**  The conformal ``M_ε`` on a curved-PEC edge is
the correct flux-tube capacitance ``C = ε·f_A·A_dual/(f_L·L)``
(f_A = free dual-face area fraction, f_L = free edge-length
fraction).  On a transmission line, ``L·C = εμ·d·d̃`` holds per
section *independently of the tube shape* — so the section inductance
is fixed by the capacitance.  The pair-consistent
``M_μ = μ0·μ̄·d·d̃·(L_primal/A_dual)·(f_L/f_A)`` is the LC partner of
the correct C; Krietenstein's ``A_face_free`` reduction is an
independent B-flux-exclusion rule — right where no ladder structure
exists (genuinely 3D contours), the wrong LC partner on a line.

**Decision (three mechanisms).**

1. **f_A in ``EdgeMaterialData``** — ``compute_conformal_eps`` returns
   the free-area fraction of the dual face from the same OCC
   tessellation as ``eps_avg``, so ``ε_pair = eps_avg / f_A`` is the
   material average over the free region (exact for inhomogeneous
   cross-sections too).
2. **Tangential-surface re-masking** (classifier step 6b) — an
   unmasked E edge whose two endpoint nodes are connected through the
   masked-edge graph runs tangentially along a PEC surface and stays
   masked (category 3).  The connected-COMPONENT test (not mere
   adjacency) spares edges bridging *different* conductors across a
   resolved gap.  One pass is a fixed point.  This is the DD-050
   consolidation line: 2D mode solvers and 3D update see the same
   conductor.  Side fix: enlarged-cell donors can no longer be
   masked edges (previously donated mass onto interior-PEC neighbours
   silently vanished).
3. **LC-consistent M_μ coupling** (mesher step 4b,
   ``couple_face_material_pairs``) — per H face, the two spanning
   axes are candidate ladder directions with the co-located E edges
   on the bounding planes as partners.  A ladder is valid when both
   partners are free and their targets agree (rtol 1e-6) — the local
   translation-invariance test; domain-boundary slabs inherit
   validity from the adjacent interior face (a single boundary
   partner cannot certify invariance — measured failure mode: bbox
   tangent points, DD-049 geometry).  One valid ladder, or two that
   agree → ``M_μ := ε0μ0·ε_pair·μ̄·d·d̃ / M_ε[partner]``, encoded as
   an equivalent ``A_face_free`` (so ``build_M_mu``, ``m_mu_flat``
   consumers and ``compute_min_effective_mu`` see it consistently);
   two valid ladders that disagree → genuinely 3D neighbourhood, no
   wave to preserve → Krietenstein stays.  Bulk pairs reproduce their
   bulk value exactly (strict no-op away from conformal contours).
   The ``build_M_mu`` 1 % ``A_face_free`` floor applies unchanged.

**Measured (conformal round coax, D_i 0.41 mm / D_a 5 mm / ε_r 9,**
``validation/straight_coax_conformal_pair_diag.py``**).**

    case              pair spread   z_line      max |S11|   median
    19×19×25 before      3.1 %      48.12 Ω     −32.5 dB    −35.8
    19×19×25 after       0 (exact)  48.12 Ω     −44.1 dB    −61.4
    staircase 19×19      0 (exact)  44.73 Ω     −45.3 dB    −60.6
    33×33×50 after       0 (exact)  49.60 Ω     −56.3 dB    −73.6

(Analytic z_line 49.97 Ω.)  Acceptance met: staircase-level port
floor at conformal impedance accuracy; refinement now buys floor
(−44 → −56 dB), which it previously did not (−32.5 → −41.2).

**DD-051 benchmark non-regression.**  Round-WG TE11 cut-off
(conformal): −0.29 / −0.14 / −0.15 / −0.17 % at n_t ∈ {17, 25, 33,
49} (DD-051 recorded −0.45 % at n_t = 17 — slightly improved, still
sub-percent everywhere; staircase branch untouched).  Rotated cavity
(production branch, h ∈ {4, 3, 2.5, 2, 1.5, 1.25} mm): 0.48 / 0.63 /
0.45 / 0.20 / 0.11 / 0.11 % with p_obs = 1.66 — better than the
DD-051 record (0.22–1.63 %, p_obs ≈ 1.64) at every resolution.

**Scope notes.**

* TE/TM ports keep their DD-047 Mur-limited floors; the coupling
  changes transversal H faces only where a unique invariant ladder
  exists, and hollow-WG cut-offs live on the (conflicting-ladder)
  Hz faces, which the rule leaves to Krietenstein by construction.
* QTEM/inhomogeneous sections get ``ε_pair`` per pair from
  ``eps_avg/f_A`` — the pair product then varies with the local ε,
  which is the physically correct ladder (the residual QTEM floor is
  the genuine E_z ≠ 0 modal limit, DD-052).
* The coupling runs at meshing time (``Mesh.from_geometry`` step 4b)
  and sees geometric PEC only; BC-PEC walls consolidated later
  (``with_pec_boundaries``) are planar and never conformal, so they
  need no coupling.

**Files.**  ``geometry/filling.py`` (f_A return),
``geometry/subcell.py`` (``EdgeMaterialData.f_A``, step 6b, donor
blocking), ``operators/material_matrices.py``
(``couple_face_material_pairs`` + staircase lookup helpers),
``mesh/mesher.py`` (step 4b call).  Tests:
``tests/unit/test_pair_consistent_subcell.py`` (f_A population,
re-mask fixed point, donor guard, per-pair identity, bulk no-op),
``tests/integration/test_conformal_coax_sparams.py`` (floor < −40 dB,
median < −55 dB, z_line 48.12 ± 1 %).

---
## DD-054 — Exact discrete transparent boundary condition (DTBC) for TEM port modes

**Date:** 2026-07-09 (session 84 method gate, session 85 production).
**Status:** **Implemented** — WP-R2 of ``REFLECTION_FREE_PLAN.md``.
*(Session 86: the WP-R3 status block of DD-055 extends the DTBC to
TE/TM; the "TEM only" limit below is retired.)*
Scope: TEM modes; TE/TM follow in WP-R3, inhomogeneous QTEM in WP-R4.
Supersedes the DD-047 Mur termination *for TEM modes only*; Mur stays
the fallback branch for every other mode.
**Method (gated analytically in WP-R1 before any solver code).**  On a
uniform feed line the modal amplitude is the exact 1D leapfrog chain
(DD-052/DD-053 separation); its exact discrete transparent boundary is
the ghost relation with symbol ``λ = A − √(A²−1)``,
``A(z) = 1 + (z − 2 + 1/z + q²)/(2r²)`` (TEM: ``q = 0``), kernel via
contour integration.  Reflection of an approximated symbol is exactly
``Γ = (λ̃−λ)/(λ⁻¹−λ̃)``.  See the plan's method note and
``validation/dtbc_kernel_spike.py``.
**Decisions.**
1. **Per-mode termination gate in ``PortOperatorModal``.**  A TEM mode
   is DTBC-terminated iff its co-located pair product certifies a
   uniform chain: ``r_pair = dt/√(M_ε[e]·M_μ[h_partner])`` per
   transversal pair (the DD-053 pair identity with
   ``dz = d̃z = normal_dx`` on the flattened port slab; no continuum
   velocity enters), modal-energy-weighted RMS spread ``< 1e-8``.
   Homogeneous sections (any ε_r) pass at roundoff; inhomogeneous QTEM
   lines fail at the material-contrast level and keep Mur — no silent
   misuse.  ``termination="mur"`` forces the legacy branch (A/B).
2. **Exactness by kernel auto-extension, not truncation.**  At step
   ``n`` the ghost convolution reaches ``n`` samples back, so a kernel
   longer than the run is the *exact* DTBC — no truncation error, no
   passivity question (the R1 truncation/passivity analysis applies
   only to a future compressed sum-of-exponentials form, which is
   deferred until the O(n) per-step convolution cost matters).
3. **Excitation through the ghost plane.**  The incident wave is
   prescribed at the ghost plane; the same kernel propagates it to the
   port plane (``u_inc = ℓ ⊛ s``) — the *exact discrete* incoming
   wave, replacing the Mur-era fractional-delay interpolation.  The
   transparent condition acts on the scattered part only.
4. **Discrete de-stagger in ``compute_s_parameters``.**  For DTBC
   channels the two-plane a/b solve uses the exact discrete half-cell
   factor ``λ^{1/2}(e^{jωdt})`` (``port_line_params``, threaded from
   ``PortOperatorModal.dtbc_line_params``) instead of the continuum
   ``e^{−γ·dz/2}``.  The continuum factor's grid-dispersion gap
   ``≈ (βdz/2)³(1−r²)/6`` capped *measured* floors near −70 dB on
   λ/20 meshes — that measurement gap, not Mur, set the old
   parallel-plate −71.9 dB record.  For the discrete TEM chain the
   V/I magnitude ratio is frequency-independent (equal to the static
   travelling-wave calibration), so the decomposition is exact for
   the discrete wave.
**Measured (session 85,**
``validation/dtbc_tem_port_floors.py``**, band
0.25–10 GHz).**
    geometry                        before        after (max / median)
    parallel plate uniform          −71.9 dB      −138.7 / −164.0 dB
    parallel plate graded (1.4)     −64.1 dB      −136.1 / −158.1 dB
    rect coax (PTFE, ε 2.1)         −39.8 dB      −159.3 / −159.4 dB
    conformal round coax (ε 9)      −44.1 dB      −131.0 / −131.3 dB
Conformal z_line unchanged at 48.12 Ω; |S21| flat within ±0.012 dB.
All four sit 30+ dB below the −100 dB acceptance line.  The conformal
coax passes the pair gate *because of* DD-053 (the coupling makes the
pair product exact per co-located pair); its former −44 dB residual is
hereby attributed entirely to the absorber + measurement chain, not to
conformal geometry.
**Alternatives considered.**  Truncated kernel with
passivity-enforced rational fit (R1 finding: raw truncation is weakly
active) — unnecessary while the kernel outlives the run; revisit for
very long runs.  CFS-PML / mode-matching / co-simulation — archived
2026-05-06 (``archive/reflection-free-ports``), not a design
reference (ground rule, sessions 63–65).
**Limits.**
* TEM only.  TE/TM (``q > 0`` Klein-Gordon kernel with the mode's
  *discrete* cut-off) is WP-R3; the DD-047 −19 dB near-cutoff Mur
  peak stands until then.
* The factory's three-equidistant-cells validation at the port is a
  prerequisite (uniform continuation of the boundary cell).
* Non-zero initial conditions start the chain with zero velocity
  (same convention as Mur's ``initialize_state``); exactness assumes
  a quiescent exterior at t = 0, which interior ICs satisfy.
* Per-step cost is one O(n) reduction per DTBC mode (two when
  excited): O(N²) per run — measured 2.2 s total at N = 6·10⁴,
  negligible against the 3D update; the compressed form is the
  escape hatch if profiling ever says otherwise.  Implementation
  note: the reduction is ``np.einsum``, *not* ``np.dot`` — BLAS ddot
  multithreads past ~16k elements and its per-call thread-team
  overhead (ms-class, all cores spinning) made long fixed-step runs
  ~1000× slower before the switch (session-85 finding).

**Files.**  ``ports/modal/dtbc.py`` (symbol, kernel, reflection
bound, ``destagger_theta``, ``DTBCTermination``),
``ports/modal/operator.py`` (gate, ghost excitation,
``dtbc_line_params``), ``postprocessing/modal_sparameters.py``
(``port_line_params``), ``analysis/scattering_td.py`` (threading).
Tests: ``tests/unit/test_dtbc.py`` (33), ``test_modal_operator.py``
(DTBC selection), ``test_modal_sparameters.py`` (discrete de-stagger
float-noise + continuum-leak anchor); integration bounds tightened to
−110…−130 dB on all three straight-line geometries.
---
## DD-055 — Klein-Gordon DTBC for TE/TM port modes + unified multi-mode port

**Date:** 2026-07-09 (session 86).
**Status:** **Implemented** — WP-R3 of ``REFLECTION_FREE_PLAN.md``.
Extends DD-054 from TEM to TE/TM; closes STATUS construction site 1
(unified multi-mode port) and retires the DD-047 −19 dB near-cut-off
Mur limitation structurally.  Inhomogeneous QTEM stays on Mur until
WP-R4.
**Pre-check gate (before any production code,**
``validation/kg_dtbc_precheck_spike.py``**).**  Ran the raw
3D leapfrog on modal initial conditions and projected per plane:
1. **TE separates exactly as built.**  The 2D TE eigenproblem
   (``build_2d_curl_curl``) is the index-sliced restriction of the
   3D operators, so the eigenvector is an exact transversal
   eigenvector and ``Mode.omega_c`` is the exact *discrete* cut-off:
   KG-chain residual ~6e-16 (WR-90 TE10) and ~7e-16 (conformal round
   TE11), ``q = omega_c·dt`` to 1e-15, pair spread ~1e-16 — DD-053
   already answers the transversal side; TE needs no further
   analogue, conformal sections included.
2. **TM did NOT separate with the lumped node-Laplace.**  The
   hand-built ``build_2d_node_laplace`` (geometric dual areas) was
   inconsistent with the 3D metric: ``q²`` off by 9e-10 (uniform
   staircase) to 1.7e-5 (conformal round TM01), profile leakage
   5.7e-5, end-to-end lock-in capped at −77 dB — the "DD-053
   analogue for TM" the plan's pre-check asked about, answered yes.
3. **Discrete wave impedance.**  The V/I ratio of the discrete
   travelling wave through the production projections is *not* the
   continuum ``z_wave(ω)`` (gap O((ωdt)², (βdz)²) — measured caps
   −33…−72 dB with continuum Z) but exactly
       Z_TE(ω) = z0·s/√(s²−(q/2)²),  Z_TM(ω) = z0·√(s²−(q/2)²)/s,
       s = sin(ω·dt/2),   z0 = r·nV·c_pair/(dt·nI)
   — the continuum relations under ``ω → (2/dt)·sin(ωdt/2)``,
   ``β → (2/dz)·sin(β̂dz/2)``, with the static constant from the
   stored profiles (``nV``/``nI`` the M_ε/M_μ norms, ``c_pair`` the
   per-pair dual-voltage ratio of the DD-052 travelling-wave H
   form).  Verified against the coupled-chain symbol to 1e-15 and by
   CW lock-in through a bit-faithful replica of the recorder
   sampling: |b/a| of a pure outgoing wave −152…−165 dB with the
   discrete Z vs −33…−72 dB with ``z_modal``.
**Decisions.**
1. **Exact TM eigenproblem ``build_2d_tm_curl_curl``** (replaces and
   removes ``build_2d_node_laplace``).  The TM cut-off oscillation is
   the pure (e_n, h_t) resonance of one longitudinal cell, so the
   exact discrete TM eigenproblem is
   ``C_s^T·M_μ⁻¹·C_s · ê_n = ω̂_c² · M_ε[e_n] · ê_n`` with ``C_s``
   the row/column slice of the 3D primal curl (rows = transversal H
   faces at the port-adjacent half-plane = ``plane.h_u/h_v_indices``,
   columns = the normal-E edges, 1:1 with plane nodes) — the exact
   TM counterpart of ``build_2d_curl_curl``.  Given the DD-053 pair
   identity, the remaining modal-closure conditions hold
   automatically and ``ê_t ∝ G_2d·ê_n`` (topological gradient) —
   the construction ``_solve_tm`` already used, now with the exact
   eigenpair.  Dirichlet = the 3D PEC state of the normal-E edges
   themselves (post-DD-053 tangential re-masking guarantees conductor
   footprints are included); hollow-window fallback for bare meshes.
   Post-fix: TM chain residuals 1e-15, q² to 4e-15, conformal TM01
   lock-in −77 → −132 dB.  Also repairs the former 9e-10 staircase
   inconsistency.
2. **KG-DTBC gate in ``PortOperatorModal``.**  The DD-054 pair gate
   generalises unchanged; the mass is ``q = omega_c·dt`` for
   *numerical-path* TE/TM modes (the discrete eigenvalue).
   Analytical-path modes carry the continuum cut-off — wrong ``q``
   at the (ω_c·dt)² level — and stay on Mur.  Excitation through the
   ghost plane is dispersion-exact by construction (the kernel *is*
   the propagator); nothing mode-specific was added.
3. **Discrete wave impedance in ``compute_s_parameters``.**
   ``dtbc_line_params`` values extend to ``(r, q, z0)``; channels
   with ``z0 ≠ None`` replace ``z_modal(ω)`` by
   ``dtbc_wave_impedance`` (closed form evaluated on the unit circle,
   offset-free; below cut-off the branch ``rad = −j√((q/2)²−s²)``
   continues the decaying root — Z_TE inductive, Z_TM capacitive,
   like the continuum).  z0 is covariant under the V/I calibration
   rescale, so the reflection zero is calibration-independent.
4. **Unified multi-mode port (closes STATUS site 1).**
   ``PortSpecNumerical.mode_type`` defaults to ``None`` = solve both
   TE and TM families, keep the ``n_modes`` lowest cut-offs, mixed
   types, in ONE operator — injection, recording and termination per
   mode in one place, so the former two-operator TE+TM
   source-injection collision cannot occur by construction.  The
   layer-a hollow resolution (``PortWaveguide``) uses it.
**Measured (session 86,**
``validation/kg_dtbc_wg_port_floors.py``**).**  CW lock-in
through the full production solver/operators/recorder (the certified
measurement; methodology finding below):

    geometry           |S11| at 1.01·f̂_c   …across the band
    WR-90 TE10             −150.4 dB        −153 … −166 dB
    WR-90 TM11             −137.3 dB        −154 … −165 dB
    round WG TE11 (conf.)      —            −124 … −132 dB  (1.05–1.5·f̂_c)
    round WG TM01 (conf.)      —            −124 / −129 dB  (1.05 / 1.2·f̂_c)

All 24+ dB below the −100 dB acceptance line; the WR-90 numbers sit
37–66 dB below it — including 1.01·f̂_c, where the DD-047 Mur peak
was −19 dB.  (The conformal round-WG near-edge points were skipped
for run-time only: the forced-plane mesh's small dt makes the CW
settling ~10⁶ steps; the measured 1.05–1.5 trend and the exactness
argument cover the edge.)

**Measurement-methodology finding (pulsed band-edge S-parameters).**
The broadband pulsed workflow is *finite-record truncation-limited*
near a cut-off: the band-edge resonance decays only algebraically
(v_g → 0; energy diffuses into the exact absorber ~ √t), so the
rectangular-window DFT of a run truncated at finite in-domain energy
leaks into the near-edge S-parameters — measured ~+10 dB per 10× run
length, independent of the absorber; the Tukey taper is no remedy
(it clips the incident pulse on long runs).  A property of pulsed
measurement on dispersive lines, not of the port — the R1 lesson
("CW lock-in, not pulse FFT ratios") carried end-to-end.  The
benchmark prints the pulsed overview alongside (WR-90 TE10 max
−33.6 dB / median −88.9 dB at default run length) to document the
practical pulsed floor.  Candidate future feature: late-time
autoregressive estimation for pulsed runs.
**Alternatives considered.**  Keeping the node-Laplace with a
correction factor — rejected (the exact restriction exists and is
simpler); continuum Z with a fitted frequency correction — forbidden
by the ground rule and unnecessary (closed form exact).
**Limits.**
* Homogeneous cross-sections (any scalar ε_r); inhomogeneous QTEM
  and hybrid modes are WP-R4.
* The eigsh eigenvector residual (~3e-8 l2) bounds modal leakage at
  the −150 dB class — irrelevant at the −100 dB criterion, listed
  for completeness.
* Analytical-path modes (layer-b use of the closed-form solvers)
  remain on Mur; the production factory paths all build
  numerical-path modes.
**Files.**  ``ports/modal/curl_curl_2d.py``
(``build_2d_tm_curl_curl``; ``build_2d_node_laplace`` removed),
``ports/modal/numerical_2d.py`` (TM path docs),
``ports/modal/factory.py`` (``_build_tm_operators``, unified
multi-mode branch, ``PortSpecNumerical.mode_type=None``),
``ports/modal/operator.py`` (``_chain_params``, ``_chain_z0``,
``dtbc_line_params → (r, q, z0)``), ``ports/modal/dtbc.py``
(``dtbc_wave_impedance``), ``postprocessing/modal_sparameters.py``
(discrete-Z consumption), ``ports/declarative.py`` (hollow
resolution).  Tests: KG-DTBC selection in ``test_modal_operator.py``,
unified-port merge in ``test_modal_factory.py``, TM builder parity in
``test_modal_numerical_2d.py``; integration in
``tests/integration/test_wg_kg_dtbc_sparams.py``.
---
## DD-056 — CW true-mode ports for inhomogeneous lines (per-frequency zeta pencil)

**Date:** 2026-07-09/10 (session 88).
**Status:** **Implemented** — WP-R4a of ``REFLECTION_FREE_PLAN.md``
(production form Option B, developer decision 2026-07-09).
Scope: QTEM / hybrid modes on transversally inhomogeneous straight
lines, measured **CW, one frequency per run** — the certified
intermediate step before the broadband band-subspace DTBC (WP-R4b).
Modal Mur-1st remains the fallback for pulsed broadband runs on
inhomogeneous lines and for analytical-path modes.
**Method (pre-check gate before any production code,**
``validation/qtem_cw_precheck_spike.py``**).**  The WP-R4
method-selection spike (``validation/qtem_dtbc_method_spike.py``)
established that on a z-uniform section the true discrete modes
at one frequency are the eigenpairs ``(ζ, φ)`` of the quadratic zeta
pencil over the period trace ``x_p = (e_t(p), e_z(p+1/2))``, and that
no *fixed* profile reaches −100 dB broadband.  A CW run only needs
exactness *at the drive frequency*; five legs make the existing
scalar production machinery carry the true mode there:
1. **Period blocks from the production matrices**
   (``ports/modal/zeta_pencil.py::build_period_blocks``): row slices
   of ``A = M_ε⁻¹CᵀM_μ⁻¹C`` at the port-adjacent periods (no global
   matrix product; DD-053 conformal metric and PEC masks enter
   as-is), with three certificates — ≥ 4 equidistant cells along the
   port normal, z-uniform PEC mask, and translation invariance of
   the blocks extracted at periods 1 vs 2 (rtol 1e-9).  The DD-054
   pair-gate analogue: it certifies that the pencil's uniform
   continuation IS the exterior the port emulates.
2. **On-circle eigensolve, sparse** (``solve_zeta_modes`` /
   ``find_propagating_modes``): ``σ̂ = 2 − 2cos(ω·dt)`` is real on
   the unit circle, eigenvalues come in conjugate pairs = the
   incident/reflected wave pair.  The linearised pencil has a
   singular ``B`` (infinite eigenvalues), so scipy's generalised
   path is bypassed: one sparse LU of ``A_lin − σB_lin`` + ARPACK on
   ``(A_lin − σB_lin)⁻¹B_lin``.  Shift targets spread along the
   unit-circle arc ``(0, 1.3·θ_DC]`` — every propagating mode has a
   smaller phase advance than the fundamental, and a single target
   near the fundamental misses higher modes hiding among the
   evanescent cluster near ``+1`` (measured failure).  Frequency
   continuation: the QTEM Laplace solve (DD-vintage dual-Laplace)
   bootstraps the tracking profile and the ``ε_eff``-based arc hint.
3. **Frequency-local exact chain fit** (``chain_fit``): the scalar
   KG chain matches ``ζ`` at ``ω`` exactly through ONE real
   equation, ``q² = 4[sin²(ω·dt/2) − r²sin²(θ/2)]`` (θ = −arg ζ) —
   pure trig arithmetic, no cancellation; ``r²`` is spent on
   matching the *derivative* ``dζ/dσ̂`` (group delay), closed-form
   by Hellmann-Feynman with the palindromic left eigenvector
   ``ψ = W·conj(φ)``.  By construction ``λ(e^{iωdt}; r, q) = ζ`` to
   machine precision (measured ≤ 2.5e-12, a-priori |Γ(ω)| ≤
   −209 dB), so the **unchanged** ``DTBCTermination`` terminates the
   true mode exactly at the drive frequency, and its ghost injection
   launches it.  ``q² ≥ 0`` ⇔ normal dispersion (ε_eff rising with
   f) — enforced, anomalous sections raise.  Passivity: (r, q) real
   → exact passive half-lattice symbol (R1).
4. **Real port profile + dual-basis projection.**  On the circle the
   pencil is real, and the z-reflection pairing fixes a gauge with
   ``φ_t`` real (measured residual ≤ 2.5e-13) — the stored profile
   is a standard real ``DiscreteMode``.  True modes of different
   branches are NOT M_ε-orthogonal on the e_t block, so multi-channel
   ports project with the Gram-inverse dual profiles
   (``PortOperatorModal(dual_e_profiles=…)``); reconstruction stays
   primal.  V/I calibration is skipped (``calibrate=False``) — the
   stored profiles must stay exactly as built.
5. **Exact V/I phasors, convention-proof** (``cw_wave_phasors``):
   the unit incident wave's recorder response is computed by
   synthesising its Bloch field on the two port-adjacent planes and
   applying the *actual* 3D curl, ``H = −dt(C E)/(M_μ(z^{1/2} −
   z^{−1/2}))``, then projecting with the stored profiles exactly as
   ``project_V``/``project_I`` do.  The reflected wave is the
   conjugate eigenpair with ``v_out = conj(v_in)`` but
   ``i_out = −conj(i_in)`` — the purely imaginary leapfrog
   denominator flips sign under conjugation; this is the discrete
   transmission-line reverse-current sign (session-88 finding: with
   ``+conj`` the S11 floor survives — the measurement is then a pure
   a-wave — but |S21| comes out +25 dB).  The CW a/b decomposition
   (``cw_lockin_phasors`` + ``cw_decompose``) solves the exact 2×2
   system per port — the R2 λ^{1/2} de-stagger and the R3 discrete
   wave impedance are *contained* in the phasors; no closed-form
   scalar impedance appears.
**Pre-check gate results (session 88, reduced vector chain).**
q_eff² > 0 at every band point on both spike geometries; production-
equivalent boundary floors −135.9 … −154.2 dB (single and dual
channel); sparse = dense to |Δζ| ≤ 1.2e-13; no exponential drift over
32k-step noise probes (PEC reference conserves to 1e-6; non-modal
content is trapped-neutral, not amplified).  Probe-methodology
lesson: stability probes must force within the image of the update
operator — a raw state kick excites the static gradient modes, whose
leapfrog Jordan block drifts linearly and mimics boundary activity.
**Measured (session 88,**
``validation/qtem_cw_dtbc_port_floors.py``**),** CW lock-in
|S11| through the full production chain (``build_cw_true_mode_port``
→ ``FITTimeDomainSolver`` → ``PortSignalRecorder`` → 2×2
decomposition), |S21| on the matched line 0.00 dB throughout:
    geometry          band / point                   |S11|
    layered plate     1.0 … 7.8 GHz (fundamental)    −244.6 … −196.5 dB
    layered plate     2nd mode, f̂_c = 8.4465 GHz:
                      1.01 / 1.05 / 1.2 · f̂_c        −176.3 / −194.6 / −200.6 dB
    dielectric block  2.1 … 6.2 GHz (fundamental)    −250.2 … −225.2 dB
    microstrip        1.0 … 7.8 GHz (fundamental)    −250.8 … −206.5 dB

All points 76–150 dB below the −100 dB acceptance line; ε̂_eff
tracks the dispersion (layered 1.6032 → 1.8584 against the 1.6035
DC anchor; microstrip 3.083 → 3.216).  The layered fundamental is
measured through the single-channel port (the second mode is
evanescent below 8.45 GHz); the second-mode points exercise the
two-channel dual-basis port for real.
**Cost-watch (developer caveat 2026-07-09) — RESOLVED, criterion
stands.**  Measured per-frequency mode-solve cost (blocks + sparse
eigensolve + fit + phasors, *per port* — two ports per S-point) vs
the 3D CW run on the same host:

    layered      41 ms/port    3D run   2.7 s/point    share ~3 % (2 ports)
    block        31 ms/port    3D run   1.9 s/point    share ~3 %
    microstrip  433 ms/port    3D run 196   s/point    share 0.9 %

On the production-sized case the share is < 1 %; the toy lines only
reach ~3 % because their whole 3D run lasts seconds.  Well inside
the "few percent" line — no renegotiation of the criterion.
Cross-section scaling (mode solve only, microstrip refinement):
N = 2 710 / 11 132 / 45 112 → 86 ms / 0.49 s / 3.6 s per point,
ARPACK-dominated (block extraction ≤ 0.16 s).  For the future
WP-R4b broadband path (one eigensolve per S-axis point in
post-processing) a 200-point sweep on the N = 45k cross-section
costs ~12 min — the point where warm-started continuation and the
R4b band-subspace form matter.
**Measurement methodology.**  Near a channel's cut-on the CW ramp
must narrow spectrally with the gap to the cut-off
(``σ ∝ 1/(ω·dt − 2asin(q/2))``), or its below-cut-off spectral
leakage never drains and floors the lock-in fit (the R3 band-edge
lesson carried over; measured: −97.6 dB with a fixed 6-period ramp
at 1.01·f̂_c, −174.4 dB with the gap-scaled ramp).
**Alternatives considered.**  Matrix-DTBC kernel in production —
exact broadband but the per-contour-point QZ at production sizes is
prohibitive and the raw kernel tail is full-rank (WP-R4 spike);
deferred to the band-subspace form (WP-R4b).  Complex ρ-offset
eigensolve (R1-style) — replaced by the exact on-circle solve; the
conjugate-pair structure is what makes the real gauge and the
reflected-wave phasors exact.
**Limits.**
* CW measurements only: the port is exact at ``f_cw``; off-frequency
  content is transient.  Broadband pulsed runs on inhomogeneous
  lines stay on Mur until WP-R4b.
* Requires ≥ 4 equidistant cells and a locally uniform straight feed
  at the port (certificates raise otherwise).
* Only propagating modes get channels; evanescent arrivals at the
  port plane are wiped (the R2 non-modal-remainder situation —
  keep ports away from discontinuities, unchanged guidance).
* Anomalous dispersion (``q_eff² < 0``) is not certified and raises.
* Multi-conductor lines track the fundamental of ``conductors[1]``;
  exciting a *different* fundamental-family member per frequency is
  future work.
**Files.**  ``ports/modal/zeta_pencil.py`` (new),
``ports/modal/factory.py`` (``build_cw_true_mode_port``),
``ports/modal/operator.py`` (``chain_overrides``,
``dual_e_profiles``, ``calibrate``), ``ports/modal/__init__.py``
(exports).  Tests: ``tests/unit/test_zeta_pencil.py`` (9),
``tests/integration/test_qtem_cw_dtbc_sparams.py``.  Benchmarks:
``validation/qtem_cw_precheck_spike.py`` (gate),
``validation/qtem_cw_dtbc_port_floors.py`` (acceptance +
cost-watch).
---
## DD-057 — Galerkin band-subspace DTBC: broadband ports for inhomogeneous lines

**Date:** 2026-07-10 (session 89 certificate gate, session 90
production).
**Status:** **Implemented** — WP-R4b / WP-R4b-impl of
``REFLECTION_FREE_PLAN.md``.
Scope: **pulsed broadband** port termination and S-parameters on
transversally inhomogeneous straight lines (QTEM / hybrid) — one
operator terminates the whole band, one pulsed run yields the whole
S-parameter axis.  Retires modal Mur as the pulsed fallback on these
lines (Mur remains only for analytical-path modes).  The CW
per-frequency path (DD-056) stays as the single-frequency instrument.
**Method (certificate gate before any solver code, session 89,**
``validation/qtem_band_dtbc_certificate_spike.py``**).**
The WP-R4 spike showed the raw matrix-DTBC kernel tail is full-rank,
but the tracked mode-family traces over the band span a rank-p
subspace (p ≈ 5–25).  The gate fixed the production form as the
**Galerkin-projected exterior**: with ``V`` a real W-orthonormal
family basis (W = M_ε over the period trace), the exterior half-line
is replaced by the projected lattice ``D̃_k = VᵀW·D_k·V`` (p×p).
The palindromic W-symmetry is inherited (``D̃₋₁ = D̃₊₁ᵀ``), so the
projected half-line is itself a lossless lattice: its exact
small-system DTBC is **passive by construction**, and the coupled
full-interior + projected-boundary-period system is block-symmetric
lossless.  The naive alternative — projecting the *kernel*
(``U·Uᵀ·W_t·Λ·V·Vᵀ·W``) — is weakly ACTIVE (noise probes grow 2.3–66×
over 4094 steps; the spike keeps it as the measured negative
control) and is not implemented.  Every contour/eigen operation runs
at size 2p instead of 2N — the broadband kernel is cheap at
production cross-sections.
**Decisions.**
1. **Boundary trace pairing** (``build_period_blocks(...,
   pairing="boundary")``): period ``p`` holds ``(e_t(p),
   e_z(p−1/2))``.  In this pairing the exterior-facing block
   ``D₋₁`` has ``e_t`` columns only, so interior periods touch the
   boundary period exclusively through the port-plane tangential
   trace, and the boundary period's ``e_z`` half-plane lies outside
   the mesh — it exists only inside the projected p-state ``xt``.
   The operator writes ``e_t(port) = V_t·xt`` each step (the modal-
   overwrite pattern; the unprojected remainder does not cap the
   floor — gate certificate ii) and touches nothing else.
2. **Two exact small kernels** (contour QZ at 2p, R1 recipe).  The
   ghost kernel is the DTBC of the *swapped* projected pencil
   (outgoing radiation decays toward −p); the excitation kernel is
   the DTBC of the *unswapped* pencil (the incident wave prescribed
   at the ghost period decays toward +p).  The scalar chain
   degenerates to one kernel — the matrix case does not.  Both
   auto-extend past the run length (DD-054 pattern: within a run the
   boundary is exact); the excitation kernel is built lazily, so
   passive ports pay half the contour cost.
3. **Frequency-tracked ghost source.**  A *fixed* source direction
   launches, away from its reference frequency, a wave whose profile
   deficit against the true mode excites an evanescent interface
   halo **at the measurement plane** — measured −40 dB |S11| class
   at the band edges (the single-profile refutation reappearing at
   the injection).  The ghost source is therefore synthesised
   spectrally, ``ŝ(f) = ŵ(f)·VᵀW·φ_f``, with the family direction
   cubically interpolated over the tracking grid.
   ``set_excitation_band`` provides the natural broadband drive: an
   erfc-product spectral window (flat measurement span, Gaussian-
   class roll-offs reaching the ``skirt`` level at the subspace band
   edges, hard zero outside).  A merely C¹ window decays like
   ``t⁻³`` and is still at 1e−4 of peak at the window end
   (measured); a source truncated while active kicks broadband grid
   modes up to Nyquist which the band boundary structurally does not
   absorb (measured: 1e−5 near-Nyquist ringing for thousands of
   steps).  A **compactness gate** (< 1e−6 of peak at the synthesis-
   window end) turns both failure modes into a loud contract:
   excitation spectra must fit inside the subspace band.
4. **Subspace + certificates.**  Families tracked per grid frequency
   by W-overlap continuation over the sparse arc-target eigensolve
   (DD-056 machinery); real W-orthonormal SVD basis of (Re φ, Im φ)
   columns, rank by relative singular-value threshold (default
   1e−8, the subspace-capture certificate).  The projected blocks
   must pass the palindromic-symmetry residual gate (1e−10, then
   enforced exactly).  ``band_apriori_reflection`` evaluates the
   exact frequency-domain modal reflection of the built boundary
   (the gate formula in production orientation: scattered ansatz
   over the into-domain branch set, N_t×N_t solve) — dense, for
   gate-sized cross-sections; evaluated off-grid it is flat at the
   ρ-offset evaluation floor (−130…−148 dB), i.e. the subspace
   captures the family *continuum*, not just the grid points.
5. **Pulsed postprocessing** (``compute_band_s_parameters``): DFT of
   the recorded fixed-channel V/I at arbitrary axis frequencies +
   per-frequency true-mode solve on the stored inward chain + exact
   cross-phasors of every (mode, recording channel) pair through
   ``cw_wave_phasors`` → joint 2·N_ch least-squares for ``a_j(f)``,
   ``b_j(f)``.  ONE 3D run serves the whole axis; the per-frequency
   mode-solve cost is the DD-056 cost-watch number (30 ms–3.6 s per
   point), independent of the 3D run.
**Measured (session 90,**
``validation/qtem_band_dtbc_port_floors.py``**, one pulsed
run per case, |S21| = 0.00 dB throughout).**
    layered fundamental   1.0–7.8 GHz (18 pts):  −159.6…−231.3 dB
    layered 2nd family    1.05–1.28·f̂_c:         −166.7…−189.8 dB
    dielectric block      2.1–6.2 GHz (12 pts):  −186.7…−202.8 dB
    microstrip            1.0–7.8 GHz (18 pts):  −171.1…−211.0 dB
Record end/peak 1e−11…1e−13 (complete ring-down inside the record).
A-priori ceilings on the family points: −114…−125 dB — the ρ-offset
*evaluation* floor at the cut-on grid points; the TD floors go
deeper, as in the gate spike.
**Alternatives considered.**  Kernel projection — refuted (weakly
active, gate certificate iii).  Sum-of-exponentials compression of
the p×p kernel — deferred until profiling demands (R2 precedent).
Fixed-direction injection — measured −40 dB class, replaced by the
tracked source.  Interior-plane trace measurement — unnecessary once
the injection halo is at subspace-capture level.
**Limits.**
* The measurement span must sit inside the subspace band with
  roll-off guard room; the compactness gate raises otherwise.  The
  synthesis window scales with 1/dt for the same physical roll-off
  (fine meshes need longer windows).
* Pulsed points close to a cut-on are finite-record limited (the
  WP-R3 methodology finding); higher families are measured pulsed
  from 1.05·f̂_c here, the 1.01·f̂_c anchor stays CW (DD-056:
  −176.3 dB).
* Kernel build: 4·n_kernel ordered-QZ solves of size 2p per kernel
  (~10 s at n_kernel = 16384, p = 17; ~1 min at 65536, p = 9); runs
  longer than the kernel trigger a rebuild at doubled length — set
  ``n_kernel_init`` ≥ the planned run length.  SOE compression is
  the escape hatch if this ever dominates.
* Out-of-band content is not certified: injection is band-limited by
  construction; deep sub-band content reflects near-totally
  (measured rattle before band-limiting was introduced).
* ``initialize_state`` assumes a quiescent port region (as DD-054).
**Files.**  ``ports/modal/band_dtbc.py`` (new: solvent/kernel,
family tracking, subspace, Galerkin exterior, boundary state
machine, port operator, a-priori evaluator),
``ports/modal/zeta_pencil.py`` (boundary pairing),
``ports/modal/factory.py`` (``build_band_dtbc_port``),
``postprocessing/modal_sparameters.py``
(``compute_band_s_parameters``), package exports.  Tests:
``tests/unit/test_band_dtbc.py`` (12),
``tests/integration/test_qtem_band_dtbc_sparams.py`` (3).
Benchmarks: ``validation/qtem_band_dtbc_certificate_spike.py``
(gate), ``validation/qtem_band_dtbc_port_floors.py``
(acceptance).

---

## DD-058 — H-face enlarged-cell donor: implemented, measured neutral, dormant

**Date:** 2026-07-10 (session 91, WP-R5 — the last open item of
``REFLECTION_FREE_PLAN.md``).
**Status:** **Implemented, dormant.**  The mechanism exists
(``assign_h_face_donors``), is unit-tested and callable, but is *not*
wired into ``Mesh.from_geometry``: the trigger benchmark built for it
measured the DD-051 trigger gate as **not met** and the mechanism
itself as neutral to machine precision.
**Why this existed.**  DD-051 Variante A left one asymmetry between
the ε and μ sub-cell corrections: cat-2 H-faces with
``A_face_free / A_face < 1 %`` fall back to the *bulk staircase*
``M_μ`` (to keep ``1/M_μ`` finite), over-estimating the local Faraday
inertia by up to 100×, while short E-edges get an enlarged-cell donor.
Round-WG measurements showed the fallback harmless there (~48 % floor
share, no mode energy on the floored faces); STATUS site 2 recorded
the trigger conditions under which a deep-PEC-inclusion geometry
(iris-loaded cavity, narrow aperture, DTL cell) would need the donor:
(a) floor share > 70 %, (b) convergence benchmark > 1–2 % off,
(c) mode energy at the PEC boundary — any one necessary, all three
sufficient.
**Trigger benchmark**
(``validation/iris_cavity_donor_trigger.py``): pillbox
cavity R = 9.7 mm split by a t = 2 mm PEC iris with an a = 3.1 mm
aperture — the TM010 0/π pair concentrates its field at the curved
aperture rim, which cuts transversal H-faces at grazing angles.
Three branches per resolution (n_t ∈ {17, 21, 25, 33}): staircase,
conformal (floor fallback), conformal + donor.  Reference =
Richardson extrapolation ``f_inf + C·h^p`` per branch (grid-search p,
linear LSQ), methodology anchored on the iris-free pillbox against
the analytic TM010 (the anchor shows the staircase extrapolation is
only good to ~1 %, the conformal finest-mesh values to a few tenths
of a percent — adequate for a 1–2 % criterion).
**Measured (session 91):**
* Trigger (a) **fires**: floor share 70.3–71.8 % of 4 500–19 420
  cat-2 faces (81.6–91.7 % inside the iris slab).
* Trigger (c) partially: 8.7–28.8 % of the *electric* mode energy
  within 2h of the rim, but only 1.3–7.4 % of the magnetic energy —
  falling with refinement.
* Trigger (b) **does not fire**: conformal finest-mesh error
  −0.17/−0.23 % (π/0 mode) vs the cross-branch reference;
  staircase +1.0/+1.1 %.  No accuracy deficit anywhere near the
  1–2 % line.
* **The donor is neutral to machine precision**: with 1 328–4 976
  donated faces the eigenfrequencies of the donor branch match the
  fallback branch to < 1e-15 relative (last-bit noise).
**Structural finding (why the fallback can never inject error
here).**  A face whose free area collapsed below 1 % lies ≥ 99 %
inside PEC — and its four circulation E-edges then sit inside (or
tangential to) the conductor, i.e. inside the PEC *mask*.  With
``C e = 0`` on the face's boundary, ``h`` never leaves 0 regardless
of the inertia assigned to it, and its EMF contribution ``C^T h``
stays 0.  The floored faces are *Faraday-dead by construction*; the
staircase fallback only serves to keep ``1/M_μ`` finite.  This is
the sharp version of the DD-051 "no mode energy there" heuristic,
and it holds independently of the geometry class — which is why even
the purpose-built deep-inclusion benchmark cannot trigger the donor.
**Mechanism (kept, dormant).**  ``assign_h_face_donors(mesh)``
(``operators/material_matrices.py``) mirrors the E-edge donor: per
floored cat-2 face, the residual inertia ``μ̄·A_face_free`` moves to
the neighbour face along the dual-edge axis (the shared flux tube;
receiver = larger free-area ratio, never a floored face, never a
staircase interior-PEC face), recorded in the new
``FaceMaterialData.enlarged_cell_donor / enlarged_cell_area`` fields
(default ``None``).  ``build_M_mu`` then freezes donated faces
(``M_μ = 0``) and adds ``μ0·borrowed/L_dual`` to the receiver.  The
update helpers translate ``M_μ = 0`` into the *exact* zero:
``β_H = 0`` (fit_td), ``1/M_μ = 0`` (eigenmode_3d, 2D mode solvers,
ζ-pencil), h-profile 0 (``tem_laplace`` travelling-wave pairs).  It
must run **after** ``couple_face_material_pairs``.  Wiring point and
condition are documented in ``Mesh.from_geometry`` (step 4b comment);
re-run the trigger benchmark adapted to the candidate geometry first.
**Hardening by-catches (production fixes in this session):**
1. ``couple_face_material_pairs`` gained a **finite gate**: on
   degenerate faces (μ̄ = 0) the encoded equivalent area was
   ``inf``, which ``build_M_mu`` turned into ``0·inf = NaN`` — a
   silently poisoned spectrum.  Non-finite values are never written
   into the mesh data any more.
2. The μ̄ = 0 faces came from **sliver cells**: cylinder tangent
   planes return from CSG solids with ~1e-16 float wiggle, and a
   tangent plane that lands within float distance of — but not
   bit-exactly on — a forced grid node produces a ~1e-18 m cell
   (the session-88 snap-plane lesson, third occurrence).  The
   benchmark guards this loudly; a mesher-side clustering of
   forced planes against geometry critical planes (within
   ``min_feature_gap``) is the structural fix — noted as an open
   mesher improvement, not done here.  *Resolved session 92
   (WP-M1, ``_merge_axis_planes``): forced planes are verbatim
   anchors, critical planes within ``min_feature_gap`` snap onto
   them; R = 10 mm / a = 3 mm meshes sliver-free at every n_t.*
3. The ``M_μ = 0`` guards double as protection against exactly such
   degenerate faces: they freeze cleanly instead of the historical
   ``dt/1.0`` placeholder coefficient.
**Files:** ``operators/material_matrices.py``
(``assign_h_face_donors``, ``_build_pec_cell_mask``, donor branch in
``build_M_mu``, finite gate in ``couple_face_material_pairs``),
``geometry/subcell.py`` (``FaceMaterialData`` donor fields),
``solver/fit_td.py`` / ``solver/eigenmode_3d.py`` /
``ports/modal/curl_curl_2d.py`` / ``ports/modal/zeta_pencil.py`` /
``ports/modal/tem_laplace.py`` (exact-zero guards),
``validation/iris_cavity_donor_trigger.py`` (sentinel
benchmark), ``validation/subcell_floor_histogram.py``
(pointer update), ``tests/unit/test_h_face_donor.py`` (8 tests).
---

## DD-059 — Thin-sheet pipeline reorder: detect-before-grid, one substrate-side plane + sub-cell filling

**Date:** 2026-07-10 (session 92, WP-M2 of ``MESHER_PLAN.md``;
developer decisions 2026-07-10).
**Status:** Implemented.
**Decision.**  A PEC shape whose bounding box is thinner than the hard
``min_cell_size`` floor along exactly one axis (and at least a floor
wide along the other two) is modelled as a **thin sheet**: it gets
**one** grid plane at its *substrate-side* face carrying the
``apply_thin_pec_sheet`` tangential-E mask, the far-side face is
dropped from the critical-plane set, and the metal volume **stays in
the DD-051 sub-cell classification** of the adjacent cells — the
thickness effect enters through the conformal material matrices
(normal-E ``L_free`` reduction, H-face ``A_face_free`` reduction)
instead of a resolved cell layer.
**Why the old path was dead.**  ``detect_thin_metallizations`` ran
*after* grid-line generation and classified "thin" against the local
cell size of the very grid that had already resolved the layer — both
metallization faces were critical planes, so the local cell was
exactly the layer thickness and ``extent < local_min_cell`` could
never hold above ``min_feature_gap``.  Measured on the session-91
microstrip reproducer (0.635 mm εr = 4.3 substrate, 1.8 mm × 35 µm
strip, floor 100 µm): grid nodes at 0.635 **and** 0.670 mm, a 35 µm
cell layer, ``courant_dt`` collapsed to 0.099 ps.
**Detection (rewritten, pre-grid).**  Threshold = the hard
``min_cell_size`` — thin-sheet handling is **opt-in via the floor**
(developer-accepted "simplest honest option": without a floor the
local cell size is unknowable before the grid exists).  The substrate
side is the face whose adjacent material has the higher permittivity,
probed at the transverse centre just outside each face
(``point_in_shape`` against the shape list in priority order; PEC
neighbours and a PEC background count as ε = 1; ties pick the
lower-coordinate face).  Sub-floor **wires** (thin along ≥ 2 axes)
and thin **dielectric** layers (solder-mask class) are *not*
detected — wires stay with the conformal machinery, dielectrics are
deferred.
**Three structural findings during implementation:**
1. **The far-side face re-enters through the negative imprint.**
   Dropping the far face from the thin shape's own critical-plane
   contribution is not enough: ``Difference(air, strip)`` contributes
   the cavity face at the same position.  The drop is therefore
   *global* (any plane within ``min_feature_gap`` of the far face,
   unless it is also within tolerance of the sheet plane) —
   implemented on top of the WP-M1 per-shape extraction
   (``extract_critical_planes_per_shape``).
2. **The DD-051 classifier is material_id-gated in every stage** and
   ``material_id`` cannot see a sub-cell-thin volume (no cell centre
   lies inside the metal).  Both candidate gates are therefore
   seeded explicitly from the metal boxes
   (``thin_sheet_boxes`` → ``compute_subcell_data`` /
   ``compute_subcell_data_mu``): cells overlapping a box become
   boundary cells for the conformal ε̄/μ̄ passes
   (``extra_boundary_cells``), and E-edges whose segment intersects a
   box (strict along the span, inclusive transversally) join the
   PEC-adjacency set for the line-solid ``f_L`` pass.  The metal
   volume itself enters through the effective PEC solid — the thin
   shape stays in the classifier shape list but is excluded from
   cell-centre filling (a centre inside the metal would resolve the
   sheet as a full staircase cell layer whenever ``d < 2t``).
3. **Cell-fill and classifier lists split.**
   ``shapes_with_material`` (cross-section cell filling) excludes
   detected sheets; ``classifier_shapes_all`` keeps them.
**Measured (microstrip reproducer).**  ONE grid plane at 0.635 mm, no
node inside the metal layer, no 35 µm cell (``dz`` above the sheet =
317.5 µm at ``min_cells_per_feature = 2``), ``courant_dt``
0.108 → 0.573 ps; the Ez edges above the strip are cat 2 with
``L_free = dz − t`` exactly.  **Impedance sanity**
(``validation/thin_sheet_impedance_sanity.py``, shielded
microstrip through the public modal-port factory, matched transverse
resolution ``max_cell_size = 100 µm``): thin-sheet vs resolved
reference ε_eff 3.5785 vs 3.5293 (**1.39 %**), Z₀ 39.76 vs 40.13 Ω
(**0.91 %**), with dt 0.174 vs 0.039 ps (4.5×).  At a shared *coarse*
bulk the deltas are dominated by ordinary transverse discretization
(6.5 % on Z₀ at 15 × 9 cells), not by the sheet model.  The 2 %
acceptance gate in the benchmark was **accepted by the developer
2026-07-10** (session 92) from these numbers.  Note: the thin-sheet
branch initially showed one 90.7 µm cell — the grading-refit floor
softness, closed by WP-M3 (re-measured there: d_min exactly
100.0 µm).
**Out of scope / deferred:** thin dielectric layers; conductor-loss
(R_s) sheet models; footprint-exact masking for non-brick sheets
(the rect mask is the shape's transverse bounding box — exact for
bricks; curved traces would over-mask and should gain a
cross-section-polygon footprint before detection admits them).
**Files:** ``mesh/conformal.py`` (``detect_thin_metallizations``
rewritten, ``ThinSheetSpec`` gains ``far_position``/``shape``,
``_probe_eps``), ``mesh/mesher.py`` (step-0 detection, per-shape
critical-plane filtering + global far-face drop, fill/classifier list
split, ``thin_sheet_boxes`` threading),
``geometry/occ_backend.py`` (``extract_critical_planes_per_shape``),
``geometry/subcell.py`` (``_thin_sheet_cell_seed``,
``_edges_intersecting_box``, ``thin_sheet_boxes`` parameters),
``geometry/filling.py`` (``extra_boundary_cells`` parameters),
``tests/unit/test_thin_sheet_detection.py`` (13 tests),
``validation/thin_sheet_impedance_sanity.py``.
---

## DD-060 — Hard min_cell_size: floor merge, floor-aware refits + longitudinal series-eps correction

**Date:** 2026-07-10 (session 92, WP-M3 of ``MESHER_PLAN.md``).
**Status:** Implemented.
**Decision.**  ``min_cell_size`` is a **hard floor**: no generated cell
may be smaller (the only exception: anchor pairs — user-forced planes
and thin-sheet planes — closer than the floor are respected verbatim
with a loud warning).  Session-91 finding: the floor was soft twice
over — (a) critical-plane intervals below it were meshed as one
smaller cell (the 35 µm cell at a 100 µm floor), and (b) the grading
refit produced sub-floor cells even inside wide intervals (measured
91.3/70.2 µm ramp cells; the clamp bound only the ``h_fine``/``h_max``
*targets*, then ``_grade_symmetric_to_uniform`` refit below it).
**(a) Floor merge** (``_floor_merge_planes``, a second merge stage
after the WP-M1 clustering): non-anchor planes within the floor of an
anchor drop (the anchor wins); among the rest a keep-first scan drops
every plane closer than the floor to the previously kept one; the
domain-end plane always survives.  **Survivor choice (open detail,
measured):** keep-first — the survivor is a *real* material face, so
grid nodes stay on geometry boundaries (consistent with the DD-059
substrate-side convention); with the longitudinal correction below,
keep-first vs midpoint is second-order (+0.31 % vs −0.11 % on the
measured example).
**(b) Floor-aware refits:** every node generator now receives
``min_cell`` — uniform fills cap the cell count so ``interval/n ≥
floor`` (``_n_uniform_floor``; the pre-fix ``rest/ceil(rest/h_max)``
undershoots whenever ``h_max < 2·floor``), sub-floor remainders are
absorbed into the adjacent ramp cell, and the one-sided/symmetric
ratio-g refits back off ``n`` while ``h0(n) < floor`` (h0 grows as n
shrinks — floor beats the ``h_fine`` refinement wish by design).
**Structural finding — the transverse-average assumption breaks.**
The plan's premise "dielectric planes → merged, conformal sub-cell
absorbs the offset" is **false as stated**: the DD-051 dual-face ε̄ is
a *transverse* area average at the edge midpoint and cannot represent
a **series stack along the edge**.  Before WP-M3 this could never
surface — material boundaries were always grid planes, so no primal
edge ever crossed one.  Measured (layered parallel plate, 60 µm ε=8
layer at a 100 µm floor, analytic series-capacitor reference
ε_eff = 1.38817): fine reference −0.000 %, floor merge without the
correction **+3.72 %** (keep-first) / **−2.56 %** (midpoint),
non-converging (locked to the floor scale).
**Fix — longitudinal harmonic eps** (``_apply_longitudinal_eps``):
``_floor_merge_planes`` records the absorbed material planes; edges
crossing one get the length-weighted **harmonic** mean over their
segments, ``ε̄ = L / Σ (L_seg / ε_seg)``, where each segment's ε is
the dual-face area average sectioned at the *segment* midpoint through
the same OCC backend (so combined transverse × longitudinal variation
is handled; σ analogously with the exact DC short-circuit ``σ_seg = 0
⇒ σ̄ = 0``).  PEC-adjacent edges are skipped (the line-solid f_L path
owns them).  Measured after the fix: keep-first **+0.31 %**, midpoint
**−0.11 %** (residual = the wall-edge columns the conformal pass has
always left to staircase).  The mu analogue (series stacks of
μ-contrast materials across a dual edge) is noted as deferred —
μ-contrast dielectrics are rare; the stress sentinel (WP-M5) will
flag it if it ever matters.
**Acceptance (property tests):** randomised axis-line generation
(300 parameter draws) and randomised brick-stack meshes (8 OCC
draws) — every cell ≥ floor, monotone nodes, exact endpoint
preservation; the session-91 refit case is pinned as a regression
test; negative control verified (both property tests fail on the
pre-WP-M3 code).  Thin-sheet impedance sanity re-measured with the
hard floor active: d_min exactly 100.0 µm, dt 0.183 ps, ΔZ₀ 1.09 % /
Δε_eff 1.49 % vs the resolved reference (2 % gate, accepted with the
session-92 sign-off).
**Files:** ``mesh/mesher.py`` (``_floor_merge_planes``,
``_n_uniform_floor``, ``_h0_symmetric``/``_h0_one_sided``,
``min_cell`` threading through ``_generate_axis_lines`` /
``_grade_then_uniform`` / ``_grade_symmetric_to_uniform`` /
``_n_one_sided``, ``_widths_to_nodes``, absorbed-plane threading),
``geometry/subcell.py`` (``_apply_longitudinal_eps``,
``absorbed_planes`` parameter), ``tests/unit/test_mesh.py``
(``TestFloorMergePlanes``, ``TestHardMinCellSize`` incl. the
harmonic-eps unit test).
---

## DD-061 — Mesher helper audit: per-axis h_fine, hard sliver gate, pinned generator contracts

**Date:** 2026-07-10 (session 92, WP-M4 of ``MESHER_PLAN.md``).
**Status:** Implemented.
**Per-axis ``h_fine``.**  The feature-based fine size was the *global*
minimum over all axes — one small gap on z refined every x/y interface
too.  Now per axis: ``h_fine[axis] = min_gap[axis] /
min_cells_per_feature`` (axes without interior material planes stay at
the wavelength size).  Measured on the microstrip reproducer
(floor 100 µm): 630 → 378 cells (−40 %), ``courant_dt``
0.573 → 0.783 ps (+37 %), x-interface cells 232 → 600 µm while the
z resolution is unchanged — the gap is a *directional* quantity and
``min_cells_per_feature`` now resolves each axis against its own
features (matching the behaviour of established meshers).
**Hard sliver gate.**  The DD-058 corruption class (a ~1e-18 m cell
whose degenerate faces silently poison M_μ) previously produced only
an aspect-ratio *warning*.  ``Mesh.from_geometry`` now **raises** when
a generated cell is below ``min_feature_gap`` unless both bounding
nodes are user anchors (forced / thin-sheet planes — those pairs are
kept verbatim by design and already warn).  After WP-M1…M3 the gate
is an invariant assertion: it cannot fire from geometry alone; if it
ever does, the clustering failed and the failure is loud instead of a
silently corrupted spectrum.  Aspect-ratio and growth-factor checks
stay warnings (legitimate meshes exceed them; the hard invariants are
the gate + the WP-M5 sentinel).
**Pinned generator contracts** (measured over randomised parameter
draws, 3000 offline + 500 in the suite): exact endpoints, strictly
positive monotone widths, hard floor respected, and neighbour-cell
ratio ≤ **1.5·g** — the 1.5 factor is the sub-``h_max/2`` remainder
absorbed into the last ramp cell; the pre-audit code held the same
bound but nothing pinned it.
**Closed audit items without code change:** ``_snap_planes`` midpoint
semantics — decided by WP-M1 (midpoint clustering for symmetric
material pairs, verbatim-anchor snap for forced/sheet planes);
``min_cells_per_feature`` × thin-layer explosion — gone by WP-M2 (the
sub-floor layer no longer exists as an interior feature gap; the
reproducer's z feature is the 0.635 mm substrate, giving 317.5 µm at
``min_cells_per_feature = 2``).
**New coverage:** PML extension (depth ≥ ``pml_thickness_cells``,
uniform extension cells at the boundary width, material continuation
into the PML slab, both min/max sides), degenerate axes (N = 1
cells), injected-sliver gate test (monkeypatched generator), forced
sub-gap pair pass-through.
**Files:** ``mesh/mesher.py`` (per-axis ``h_fine_axis``, hard gate,
docstring), ``tests/unit/test_mesh.py`` (``TestPerAxisHFine``,
``TestPMLExtension``, ``TestDegenerateAxis``,
``TestRampFixpointProperty``, ``TestSliverGate``).
---

## DD-062 — Permanent mesher stress sentinel (30 cases x 7 invariants)

**Date:** 2026-07-10 (session 92, WP-M5 of ``MESHER_PLAN.md`` — the
last work package; the plan is fully worked off).
**Status:** Implemented; runs as the permanent sentinel for every
future mesher change.
**What it is.**  ``validation/mesher_stress_sentinel.py`` —
30 randomised and pathological geometries through
``Mesh.from_geometry`` (``--fast`` for a CI-smoke subset), each
checked against machine-checkable invariants: **I1** monotone
positive axes; **I2** hard ``min_cell_size`` floor; **I3** no cell
below ``min_feature_gap`` (the DD-058 sliver class — the WP-M4 gate
raising counts as FAIL, not crash); **I4** neighbour growth ratio
≤ 1.5·g when feature refinement is active; **I5** M_ε/M_μ finite and
M_μ ≥ 0 (the DD-058 spectrum-poisoning class); **I6** production
``courant_dt`` within budget of the floor-implied bound; **I7**
thin-sheet planes on-grid, no node inside a detected metal layer.
Case families are the session-88/91/92 lessons as permanent
regressions: tangent cylinders on/off forced-grid multiples,
CSG-wiggled faces vs forced nodes (1e-16…1e-9), thin PEC layers at
0.1×…2× the floor, near-coincident dielectric faces (noise scale to
1.5× floor), rotated bricks, seeded random brick/cylinder soups with
random floors.  Measured: 30/30 clean in ~16 s.
**Two findings from the sentinel's first run:**
1. **Across-interval growth jumps with ``min_cells_per_feature = 0``**
   (measured 3.2–7.6× on the tangent-cylinder family): the two-scale
   design never promised smoothing between adjacent critical-plane
   intervals when feature refinement is off — every interval is
   meshed independently against ``h_max``.  With
   ``min_cells_per_feature ≥ 1`` every interval starts at the shared
   per-axis ``h_fine`` (DD-061), so the 1.5·g contract extends across
   boundaries; I4 is therefore gated on feature refinement, and the
   ``mcpf = 0`` jumps remain a user-accepted trade covered by the
   quality warning.  A global smoothing pass stays a possible future
   feature, not a defect.
2. **The OCC kernel cannot build solids at/below its precision**
   (``Precision::Confusion()`` = 1e-7 m): a 100 nm brick died deep in
   ``MakeBox`` with a cryptic ``Standard_DomainError``.  The geometry
   layer now rejects sub-precision dimensions with an informative
   ``ValueError`` (``occ_backend._check_dimensions`` on brick /
   sphere / cylinder builders) pointing at material/BC modelling for
   sub-100-nm features.
**Files:** ``validation/mesher_stress_sentinel.py``,
``geometry/occ_backend.py`` (``_check_dimensions``),
``tests/unit/test_geometry.py`` (``TestOccPrecisionGuard``, 3 tests).
---

## DD-063 — Layer-a band-pipeline auto-dispatch + exact time-domain power waves

**Date:** 2026-07-10 (session 93; developer request 2026-07-10:
"the high-level API must not use an outdated model on a microstrip").
**Status:** Implemented.
Scope: ``AnalysisScatteringTD`` (layer a) catches up with the
reflection-free port work (DD-054…DD-057) on two fronts: the port
pipeline for inhomogeneous lines and the time-domain power-wave
accessors.
**1. Port-pipeline dispatch (``port_model="auto"`` default).**
``run()`` probes every ``PortSpecMultiConductor`` by building its
modal operator and reading ``termination_kinds``: if all channels are
DTBC-certified (homogeneous cross-sections), the cheap modal pipeline
runs as before; if any channel would fall back to modal Mur-1st (the
inhomogeneous-QTEM case — microstrip, layered substrates, measured
−30 dB-class |S11|), the run switches to the DD-057 band-subspace
DTBC pipeline: ``build_band_dtbc_port`` per spec (built **once**,
reused across excitations via the new ``reset_state()`` — kernels
and subspace kept, boundary state and histories zeroed),
``set_excitation_band`` flat-spectrum drive over the measurement
span, fixed-length pulsed record, ``compute_band_s_parameters``
per-frequency true-mode decomposition.  ``port_model="modal"`` opts
back into the Mur fallback, ``"band"`` forces the band pipeline;
mixing an uncertified multi-conductor port with other spec types
raises (band decomposition needs band ports on every face — layer-b
territory).  Auto-derived parameters (``band_options`` overrides):
``f_band`` = measurement span padded 25 % both sides (floored at
``0.25·f_min``), ``n_grid`` = 25, ``n_syn`` from the erfc roll-off
compactness budget (13 Gaussian time constants, power of two,
min 8192) with automatic doubling when the DD-057 compactness gate
rejects, record = ``n_syn`` + ring-down (8 diagonal traversals at
``0.5·c``, min 4096), kernels pre-sized past the record.
``energy_stop_db`` / ``taper_signals`` do not apply to the band
record (fixed-length DFT contract; a ring-down-quality warning fires
at end/peak > 1e-4).  Measured (CI-scale layered line, pure
defaults): |S11| −204.8…−226.1 dB, |S21| 0.000 dB, ~1 min end to
end; with CI overrides −120…−227 dB in ~20 s.  The declarative
``PortWaveguide`` path feeds this dispatch naturally (inhomogeneous
faces resolve to ``epsilon_r=None`` multi-conductor specs).
**2. Exact time-domain power waves (``destagger=True`` default).**
``result.a()`` / ``result.b()`` previously used the co-located
``(V/√Z ∓ √Z·I)/2`` split with midpoint-aligned I; the spatial
half-cell stagger of the I plane leaks ``≈ β·dz/4`` of the incident
pulse into ``b`` — a smooth derivative-of-pulse ghost measured at
−37.8 dB of the a1 peak on the rect-coax notebook mesh while the
S-parameters (exactly de-staggered per frequency) sit at −159 dB.
The accessors now default to the frequency-domain path
(``destaggered_power_waves``): rfft → the *same* per-bin corrections
as ``compute_s_parameters`` (Yee half-step rotation, two-plane
de-stagger with the exact discrete ``λ^{1/2}`` on certified chains,
exact discrete wave impedance) → irfft.  The shared per-channel core
is factored into ``spectral_power_waves`` (consumed by both paths;
``guarded=True`` NaNs out singular bins — continuum cos-denominator
zeros beyond the trust band, wave-impedance cut-off collapse — which
then fall back to the frozen-Z co-located split; those bins carry no
pulse energy on band-limited excitations).  Measured on the
rect-coax regression fixture: time-domain max|b1|/max|a1| drops
1.29e-2 → 1.07e-8 (−37.8 → −159.4 dB, the port floor), a1→b2 energy
conserved to 1e-7.  ``destagger=False`` restores the historical
split.  ``ScatteringTDResult`` carries ``port_normal_dx`` /
``port_line_params`` / ``port_model_used`` for this.  On band
results the accessors raise with guidance: the band port's recorded
channels are fixed subspace projections whose a/b split is defined
per frequency (DD-057) — a scalar split has no calibrated Z
(measured: b1/a1 ≈ 1); a per-frequency-synthesised time-domain
split (phasor interpolation over the tracking grid) is a possible
future extension.
**Audit items closed in the same pass.**  Outdated ``energy_stop_db``
docstring (pre-DTBC "physical Mur-1 floor ~−20 dB"); ``f_calc``
semantics documented (only analytical-path Mur modes depend on it —
for QTEM the static ε_eff makes v_p frequency-flat, DTBC/band paths
are exact per frequency); commercial-solver names and the
conformal-averaging trademark removed from source/tests/benchmarks
(IP rule); the missing DD-059…DD-062 headings restored (the DD-055/
DD-056/DD-058 slip pattern — bodies were committed without their
``##`` lines).
**Near-DC cost gate (same-day developer field report).**  The first
real microstrip run through the new dispatch hung for minutes before
the TD progress bar, single-threaded, right after the dispatch
message — the *default* frequency axis (``f_min = 0`` → first point
at ``f_max/n_freq``) reaches toward DC, the pulsed band drive needs
spectral roll-off room *below* the first axis point (deep sub-band
content is structurally not absorbed, DD-057 limit), and that room
is bounded by ``f_axis[0]`` itself: the auto-sized pulse came out at
``O(1/f_axis[0])`` ≈ hundreds of thousands of steps, and the
single-threaded contour-QZ ghost-kernel build scales with it (hours,
silently).  Fix: ``_BAND_AUTO_N_SYN_MAX = 131072`` — the auto-sizing
raises a ValueError carrying the measured pulse length and the
*recommended axis start* (exact inversion of the sizing chain);
explicit ``band_options["n_syn"]`` bypasses the gate deliberately.
The verbose path announces the kernel-build phase *before* it starts
(it was the silent long phase).  Measuring close to DC with a pulsed
band port is inherently long — the gate turns physics into a loud
contract instead of a hang (the compactness-gate pattern).
**Alternatives considered.**  Band-dispatch on the *spec* marker
(``epsilon_r=None``) alone — rejected: the marker means "read ε from
the mesh", which may still be homogeneous; the termination
certificate probe (one cheap Laplace solve per port) decides on the
physics.  Making band the unconditional multi-conductor default —
rejected: homogeneous lines get −131…−159 dB from the modal DTBC at
a fraction of the build cost.  Time-domain destagger via
convolution filters — rejected: needs a dispersive fractional-delay
design per mode; the rfft path reuses the certified frequency-domain
corrections verbatim.
**Files.**  ``analysis/scattering_td.py`` (dispatch, ``_run_band``,
``_band_setup``, ``_set_band_excitation``, result fields, destagger
accessors), ``postprocessing/modal_sparameters.py``
(``spectral_power_waves``, ``destaggered_power_waves``),
``ports/modal/band_dtbc.py`` (``reset_state`` on boundary +
operator, ``band_source_spectrum``, ``channel_band``),
``postprocessing/__init__.py`` (export).  Tests:
``tests/integration/test_analysis_scattering_band.py`` (6),
``tests/integration/test_rect_coax_sparams.py``
(``test_destaggered_time_domain_power_waves``).
---

## DD-064 — QTEM default acceptance renegotiated: modal pipeline default, band DTBC opt-in

**Date:** 2026-07-10 (session 93, same-day follow-up to DD-063;
developer decision).
**Status:** Implemented.
**Developer decision.**  After the DD-063 field trial on a small
production microstrip the band pipeline is too slow for routine work
even with a sensible lower band edge (f_min = f_max/3: kernel build +
mandatory pulse/ring-down record on a 20×7×19-cell line still
minutes), and commercial-suite reference results for simple
microstrip lines sit at the −30 dB class themselves.  The original
−100 dB QTEM acceptance (2026-07-09) is therefore **renegotiated for
the default path**: "reflections at −30 dB, fast runtimes and
time-domain signal access are definitively worth more than −100 dB
with a long, band-limited run and no time-domain signals."  The
−100 dB machinery stays fully available as an explicit choice.
**Decision.**  ``AnalysisScatteringTD(port_model=...)`` defaults to
``"modal"``: exact DTBC on certified chains, modal Mur-1st on
inhomogeneous QTEM channels, with a one-time verbose notice naming
the Mur-fallback channels and pointing at ``port_model='band'``.
``"band"`` and ``"auto"`` keep their DD-063 semantics unchanged
(band-subspace DTBC, certificate-probe auto-dispatch, near-DC cost
gate).  Nothing else of DD-063 is rolled back — destaggered
time-domain power waves remain the default accessors and work on the
default path.
**Measured basis (this session's experiments; not shipped).**  A
cheap middle path was attempted: the DD-056 zeta-pencil true mode at
band centre as a *single-profile pulsed port* (fitted ``(r_eff,
q_eff)`` scalar DTBC termination, V/I calibration enabled so the
standard broadband decomposition applies).
* Extreme layered line (half ε_r = 4, ε_eff 1.6→1.9 over the band):
  Mur −13…−16 dB, mid-band single-profile −14…−20 dB — on strongly
  dispersive lines the **transverse profile drift dominates** and
  the longitudinal termination is secondary (the DD-057
  single-profile refutation reappearing at level a).
* Realistic shielded microstrip (ε_r = 4.3 substrate, 10×8×20
  cells, 2–10 GHz): Mur −25.8 dB worst (−26…−39 dB) at |S21| errors
  ≤ 0.01 dB; the mid-band port reaches −33…−48 dB in the band
  interior but **degrades to −19.1 dB at the lower band edge and
  pollutes |S21| by up to 1.2 dB**.  Structural cause: the
  frequency-local Klein-Gordon fit implies an *artificial cut-off*
  (``f̂_c = q_eff/(2π·dt)`` landed near the lower band edge) — the
  fitted chain treats low-band content as near-evanescent where the
  true QTEM mode propagates, and the constant-Z decomposition
  (``z0 = None``) misses the fitted chain's V/I there.
* Conclusion: **no robust cheap +10 dB exists today.**  Candidate
  future work (not started): a cut-off-free symbol fit (q ≡ 0,
  r matched to β(f_mid)), an edge-aware fit, or a 2–3-profile
  variant — each needs the full derive-then-measure treatment
  (feedback rule: no crutches).  The experimental ``calibrate``
  pass-through in ``build_cw_true_mode_port`` was reverted with the
  measurement recorded here.
**Files.**  ``analysis/scattering_td.py`` (default ``"modal"``,
docstrings, one-time Mur-fallback notice);
``tests/integration/test_analysis_scattering_band.py`` (default-is-
modal test incl. TD-wave availability; auto/near-DC tests pass
``port_model`` explicitly).
---

## DD-065 — Natural TD PMC wall + ``pmc_faces`` mesher placement

**Date:** 2026-07-11 (session 95, WP-U0 of ``PORT_MODES_PLAN.md``;
developer decision 2026-07-10: fix, two stages).
**Status:** Implemented (both stages).
**Problem (measured, session 94).**  Three PMC implementations placed
the magnetic wall at *two different* positions: the 2D port mode
solver and ``EigenmodeSolver3D`` use the natural boundary — wall Δ/2
*outside* the outermost primal grid line (parallel-plate TE(1,0)
cut-off = ``c/2(a+Δx)`` exactly) — while the TD ``PMCBoundary.apply``
zeroed tangential H on the first/last *cell-centre* layer, i.e. wall
~Δ/2 *inside*.  One full cell apart: on the PMC parallel plate the TD
transmission edge sat at ``c/2(a−Δx)`` ≈ 15.8 GHz vs. the port mode's
14.26 GHz, the S-matrix was non-passive between the two cut-offs
(|S21| up to +14.6 dB) and in-band |S11| never beat −17 dB.  TEM
modes carry no tangential H on PMC walls, which is why the TEM
notebooks never exposed this.
**Stage 1 — the TD update uses the natural boundary.**
``PMCBoundary`` performs no field surgery at all (pure face-coverage
marker): the E-update kernels already accumulate only in-domain H
faces (missing faces contribute zero) and ``_build_avg_d`` assigns
boundary E-edges a *full* dual cell — together exactly the mirror
closure ``H_tan(−Δ/2) = 0``, the same wall as both mode solvers.  The
natural BC is the free operator: symmetric, energy-conserving, no new
stencil.
**Stage 2 — the mesher places the wall ON the requested bbox face.**
``Mesh.from_geometry(..., pmc_faces=[...])`` pulls the outermost grid
line to one third of the original boundary cell: the boundary cell
shrinks to 2Δ/3 and its outside half-cell reaches exactly back to the
nominal face — the wall lands on the requested geometry for any Δ.
Local cell-ratio 1.5, below the quality-warning threshold; a face in
both ``pml_faces`` and ``pmc_faces`` raises; a user-forced plane on
the moved node warns.  Layer b (``Mesh.from_grid``) is untouched —
explicit grids keep user-placed lines, wall Δ/2 outside.
**Measured (fixture: 10×5×20 mm air brick, PMC x, PEC plates y,
TE(1,0) z-ports; ``validation/pmc_wall_te10_port_floor.py``):**
* CW lock-in |S11| −156.6 dB at 1.05 f̂_c, monotone to −165.2 dB at
  1.8 f̂_c (acceptance < −100 dB) — pre-fix in-band |S11| ≈ −17 dB.
* Pulsed passivity through the former double-cut-off window:
  max |S21| +0.055 dB / power sum 1.0589 at the 1.001 f̂_c edge bin,
  1.0045 at 1.05 f̂_c — bit-for-bit the truncation-limited measurement
  class of the accepted all-PEC WR-90 reference (+0.046 dB / 1.0594 /
  1.0042) run through the identical pipeline.  Pre-fix: +14.6 dB.
* TEM channel vs. the legacy zeroing: max |ΔS| = 1.7e-14 — the
  double-precision floor (the zeroing had erased ~1e-16 Laplace-solver
  noise in the wall Hy/Hz each step).  No physical change.
* Stage-2 gate: from-geometry TE(1,0) cut-off vs. ``c/2a`` rel. error
  −1.14e-3 (Δx = 0.5 mm) → −2.71e-4 (0.25 mm), ratio 4.2 = O(Δx²);
  without ``pmc_faces`` the O(Δx) half-cell bias is −4.9e-2.
**Files.**  ``boundaries/pmc.py`` (no-op marker + rationale);
``mesh/mesher.py`` (``pmc_faces`` step 2c);
``validation/pmc_wall_te10_port_floor.py`` (acceptance);
``examples/straight_waveguide_parallel_plate.py`` (``pmc_faces`` in
the layer-a target form); ``tests/integration/test_pmc_wall_te10.py``
(6: CW floor, wall-convention pin, pulsed passivity, wall-on-face,
second-order convergence, pml/pmc conflict);
``tests/unit/test_boundaries.py`` (PMC no-op contract).
---

## DD-066 — Unified multi-mode port: TEM ⊕ TE/TM factory merge + multi-TEM Gram-eigenbasis

**Date:** 2026-07-11 (session 95, WP-U1…U5 of ``PORT_MODES_PLAN.md``).
**Status:** Implemented; one open item (conformal-chain KG-DTBC
residual, see below).
**The merge (WP-U2).**  ``build_modal_port`` on a
``PortSpecMultiConductor`` with homogeneous scalar ``epsilon_r`` and
``n_modes > K−1`` no longer raises: the K−1 Laplace TEM channels are
joined by the lowest TE/TM curl-curl channels of the same
``PortPlane`` (both families were WP-U1-certified on
multiply-connected cross-sections; the measured discrete
cross-orthogonality through the production projections is
8e-16…2.2e-14).  Merged by ascending cut-off (TEM first, f_c = 0),
one operator, one recorder, per-channel termination dispatch
unchanged (TEM q = 0 / DD-054, TE/TM Klein-Gordon / DD-055),
family-explicit labels (``TEM_lap00``, ``TE_num00``, …).  The QTEM
path (``epsilon_r=None``) keeps the cap and raises with WP-U6
guidance — higher modes on inhomogeneous cross-sections are hybrid.
Layer a follows for free (WP-U3): ``resolve_declarative_port``
already produced the MC spec for any ``n_modes``.
**Multi-TEM channel basis (the load-bearing fix).**  The
per-conductor ``V_k = +1 V`` Laplace modes are individually
M-normalised but mutually NON-orthogonal — measured 32 % overlap on
the symmetric two-wire — while the entire downstream pipeline
(``discretize_modes`` numerical pass-through, operator projections,
TF/SF injection, per-channel DTBC) assumes an M_ε-orthonormal basis.
Measured consequence on the first-ever 2-signal fixture: the DTBC
feedback loop between overlapping channels blows up to 1e64, TEM
|S11| −17 dB (pre-existing bug, no merge involved).
``solve_tem_laplace`` now returns the **Gram-matrix eigenbasis of
the TEM subspace** (eigenvectors of the raw-field FIT-metric Gram =
capacitance-matrix eigenmodes; descending-eigenvalue order,
largest-|w| sign gauge): the exact odd/even pair on the symmetric
two-wire, with distinct line impedances (84.28 / 161.94 Ω), all TEM
channels of a homogeneous line being degenerate so any orthogonal
basis of the subspace is equally valid.  ``z_line`` of a mixed
channel refers to its unit-Euclidean conductor-voltage pattern.
Single-signal cross-sections keep the historical path bit-identically.
K > 2 QTEM retains the non-orthogonal per-conductor basis (per-mode
ε_eff forbids mixing) — a WP-U6 prerequisite, documented in code.
*(Revised by DD-196: the per-mode ε_eff is the eigenvalue of the
generalised capacitance pencil, not an obstacle to mixing.)*
**Degenerate-pair gauge (WP-U4).**
``_fix_degenerate_polarisation_gauge``: same-family numerical modes
with cut-offs within the degeneracy rtol are rotated to the
principal axes of the u-edge energy form (descending u-energy,
largest-|e_u| sign) — deterministic across meshes/runs, verified
exact unit ``project_V`` after rotation.  TEM channels excluded.
Total-power convention for degenerate arrivals documented in
``compute_s_parameters`` (Σ|S21|² over the degenerate subspace).
**Measured (acceptance, WP-U5).**  Examples 1–4 run through pure
layer-a defaults: parallel plate TEM −164.5 dB / TE(1,0) 14.974 GHz;
coax TEM −144.6 dB / TE11 pair pulsed −44.4 dB; two-wire odd+even
TEM −159.2 dB / 2 TE pulsed −43…−45 dB; WR-90 unchanged.  CW lock-in
floors (``merged_port_cw_floors.py``): two-wire TE#1
−149.5/−154.6/−157.2 dB (1.05/1.2/1.5 f̂_c), two-wire TEM odd
−159.7 dB, coax TEM −144.6 dB — acceptance < −100 dB passed with
~50 dB margin on grid-aligned chains.
**Open item — conformal-chain KG-DTBC residual (DD-055 scope,
predates the merge).**  The conformal coax TE11 CW floor is
−34.8/−42.1/−49.2 dB (1.05/1.2/1.5).  Isolated: the identical
channel on a staircase coax measures −157.9 dB, on the grid-aligned
rectangular coax −158.5 dB, and halving dx moves the conformal
number −42.1 → −57.5 dB — a *convergent* discretisation residual of
the conformal transversal operator (the 2D mode is not an exact
eigenvector of the volume-restricted chain; boundary-slab M_μ is not
port-flattened — suspect list), which the pair-product chain
certificate is structurally blind to (the DD-053 LC-consistent
coupling makes pair products uniform BY CONSTRUCTION on z-invariant
conformal contours).  The round-WG conformal fixture sat at
−124…−132 dB and passed WP-R3 — gentler curvature, finer relative
resolution.  Open developer question: accept as a documented
accuracy class (à la DD-064) or extend the certificate by the
measured 2D eigenresidual → honest Mur/band fallback.
Derive-then-measure required either way.
**Files.**  ``ports/modal/factory.py`` (MC-branch merge, QTEM
guidance raise, ``_fix_degenerate_polarisation_gauge``, PortOperatorReport
``cutoff_num`` for merged ports); ``ports/modal/tem_laplace.py``
(Gram-eigenbasis, helper returns ``e_raw``, QTEM note);
``ports/declarative.py`` (uniform mode-count semantics);
``postprocessing/modal_sparameters.py`` (degenerate-pair Notes);
``validation/merged_port_cw_floors.py`` (CW acceptance);
``examples/straight_waveguide_*.py`` (status + measured numbers);
``tests/integration/test_unified_multimode_port.py`` (6: composition
coax/two-wire, orthonormal TEM basis, QTEM raise, layer-a 4-mode
S-parameters incl. the 1e64 stability regression, two-wire TE CW
floor pin).
---

## DD-067 — Feed-chain slab-consistency certificate + port-plane μ-flatten

**Date:** 2026-07-11 (session 95; developer decision 2026-07-11 on
the DD-066 open item: "extend the certificate").
**Status:** Implemented; the DD-066 open item is CLOSED.
**Root cause (measured).**  The conformal boundary-slab **normal-face
M_μ** (Hz for a z-port) deviates 36 % from the first interior slab
on the RG-58 coax (halved cell neighbourhood on the bbox face —
the same mechanism the M_ε flatten has always corrected for
tangential E; interior slabs agree to 1e-15, transversal H faces to
8e-15).  The normal-face M_μ enters the TE transversal curl-curl
operator directly, so the 2D port mode was solved against a
*different* transversal operator than the volume propagates — the
DD-066 −34.8…−49.2 dB conformal coax TE11 CW floor, while the
transversal pair-product certificate (which forms no pair containing
Hz-M_μ, and whose pair products the DD-053 coupling makes uniform BY
CONSTRUCTION on z-invariant conformal contours) certified "uniform".
Staircase and grid-aligned chains have identical slabs — hence their
−158 dB floors.
**Fix — ``flatten_port_plane_mu``.**  The exact counterpart of
``flatten_port_plane_mass`` for the magnetic mass: the normal
H-faces ON the port plane get the first-interior-slab values; same
rationale (a z-invariant feed's wavefronts all see the interior
values, so mode solver and TD update at the port plane must too).
Applied in all three port factories (modal / CW true-mode / band)
and in ``FITTimeDomainSolver.setup`` alongside the M_ε and PEC-mask
flattens.  Transversal H faces are untouched (measured identical).
**Guard — the slab-consistency certificate stage.**
``_port_chain_slab_defect`` (factory): the maximum relative slab
deviation over EVERY mass entry feeding the port's 2D mode solve,
across the first feed cells — per E/H component, plane-sampled
components compare slabs (0,1) and (1,2), layer-sampled components
layers (0,1) and (1,2); entries zero in both compared slabs are
skipped.  A z-invariant feed measures ~1e-15.  Above
``_DTBC_SLAB_DEFECT_TOL = 1e-8`` the ``PortOperatorModal`` withholds
the exact DTBC on every channel (modal Mur-1st) and the factory
warns loudly with the measured defect.  This subsumes the "2D
eigenresidual" formulation of DD-066: with slab-consistent masses
and the flattened PEC mask, the 2D operator IS the volume-restricted
transversal operator, and the eigenresidual reduces to the eigsh
solver tolerance.
**Measured (gates).**
* Conformal coax TE11 CW floor: −34.8/−42.1/−49.2 dB →
  **−134.3/−139.5/−142.0 dB** at 1.05/1.2/1.5 f̂_c — the full
  certified-DTBC class; every channel of every merged port now
  passes the WP-U5 < −100 dB acceptance.
* Pulsed layer-a coax example: TE11 pair −44.4 → −78.2 dB max above
  1.2 f_c; TEM unchanged (−144.6 dB).  The TE cut-off moves
  34.344 → 34.195 GHz (the mode solve now consumes the interior
  Hz-M_μ — the discretisation the volume actually propagates); TM01
  is untouched (normal-face M_μ does not enter the TM node
  Laplacian).  WP-U1 leg A re-measured: convergence envelope
  unchanged in class (+1.6e-2 → −4.0e-4 over dx 0.24 → 0.03 mm).
* Guard: trips (Mur + warning) on a feed with a dielectric step in
  the second cell; silent on invariant feeds; conformal-coax defect
  0.36 without / <1e-12 with the μ-flatten
  (``tests/integration/test_chain_slab_certificate.py``).
* Staircase meshes: the flatten is a no-op (identical slabs) — all
  grid-aligned pins bit-unchanged.
**Files.**  ``operators/material_matrices.py``
(``flatten_port_plane_mu``); ``ports/modal/factory.py``
(``_port_chain_slab_defect``, μ-flatten in the three builders, loud
fallback warning); ``ports/modal/operator.py``
(``chain_slab_defect`` veto in ``_chain_params``,
``_DTBC_SLAB_DEFECT_TOL``); ``solver/fit_td.py`` (μ-flatten in
``setup``); benchmarks re-measured
(``merged_port_cw_floors.py``, ``curlcurl_conductor_certification.py``,
coax example); ``tests/integration/test_chain_slab_certificate.py``
(3 tests).
---

## DD-068 — Multi-mode QTEM ports via the ζ-pencil hybrid channels

**Date:** 2026-07-11 (session 95, the last work package of
``PORT_MODES_PLAN.md``).
**Status:** Implemented; the five straight-waveguide acceptance
scripts all run through pure layer-a defaults.
**What.**  ``PortSpecMultiConductor`` with ``epsilon_r=None``
(inhomogeneous cross-section) and ``n_modes > K−1`` no longer
raises: the K−1 Laplace QTEM line modes (unchanged — the DD-064
default path, bit-identical for ``n_modes ≤ K−1``) are extended by
the ``n`` lowest true hybrid eigenpairs of the DD-056 ζ-pencil of
the production matrices at ``f_calc``
(``_qtem_zeta_hybrid_modes``): ``find_propagating_modes`` arc
targeting, the Laplace family identified and dropped by W_t
overlap, channels ordered by ascending frequency-local cut-off.
No exact TEM/TE/TM split exists there — reusing the homogeneous
merge would be an unfounded crutch; the pencil eigenpairs are the
same objects the CW and band pipelines already trust.
**Channel form.**  DD-056 real gauge (tangential trace real to
~1e-13, enforced), M_ε-normalised; ``omega_c = q_eff/dt`` — the
frequency-local Klein-Gordon fit as *metadata* (report + the
dispersive Mur ``v_p``; per the DD-064 lesson no termination is
fitted from it); ``epsilon_r`` chosen so ``Mode.gamma(2π f_calc)``
equals the exact discrete ``β = θ/dz`` — Mur-1st then uses the
channel's true phase velocity at ``f_calc``; ``mode_type = TEM``
with the frequency-flat ``η₀/√ε_r`` V/I norm — a TE-form ``Z(ω)``
would diverge at the *estimated* ``f̂_c``, exactly the DD-064
artificial-cut-off failure mode.  Projections are dual-basis over
ALL channels (Gram inverse in the port-plane M_ε — the DD-056
machinery): the hybrids are not M_ε-orthogonal to each other or to
the Laplace modes, and primal projections would re-create the
DD-066 cross-talk instability class.  Termination per DD-064
defaults: the pair-product/slab certificates fail on inhomogeneous
fillings → every channel modal Mur-1st with the loud notice;
``port_model="band"`` stays the reflection-critical opt-in for the
tracked family.
**Measured (shielded-microstrip acceptance, 35-GHz band,
``straight_waveguide_microstrip.py`` +
``qtem_multimode_port_probe.py``).**
* QTEM fundamental: max |S11| −21.2 dB / median −29.1 dB — the
  documented Mur class of the default path on this band.
* Hybrid channels (f̂_c 18.2 / 27.2 GHz): max |S11| −9.5 / −7.8 dB
  above 1.2 f̂_c, median −18.1 / −10.5 dB; total transmitted power
  within ±1 dB.  Mur-1st with the port-wide ``f_calc = f_max`` phase
  velocity is increasingly detuned toward each channel's own
  cut-off — the honest Mur-class number, structurally the DD-047
  near-cut-off Mur behaviour.
* Hybrid profile drift vs. the f_calc profile (the WP-U6 honesty
  number; profiles are frequency-dependent): 4.7e-4 / 2.7e-4 at
  34 GHz → 2.1e-2 / 4.9e-3 at 30 GHz → 1.1e-1 / 1.3e-2 at 26 GHz.
* **Fundamental invariance**: ≤ 5.5e-2 |ΔS| between the 1-mode and
  3-mode ports below the first hybrid cut-off (the 3-mode dual
  projection separates evanescent hybrid tails the 1-mode primal
  projection folds into the QTEM channel — within the Mur
  measurement class).  ABOVE the hybrid cut-offs the 1-mode port is
  structurally under-modelled: the injected Laplace profile excites
  propagating hybrids that a 1-channel port neither absorbs nor
  separates — measured max |S11| −1.18 dB and |S21| overshoot
  +3.0 dB (n=1) vs. −21.2 dB / +0.6 dB (n=3).  A multi-mode port is
  what makes QTEM S-parameters meaningful on bands crossing hybrid
  cut-offs.
**Limits (loud, tested).**  Channels must propagate at ``f_calc``
(evanescent-at-f_calc raises with guidance: raise f_calc/f_max or
reduce n_modes).  Exactly degenerate hybrid pairs are not supported:
the pencil eigenvalue dedup collapses them to one representative
whose tangential profile is not real in the DD-056 gauge — the
real-gauge check refuses (coax-through-QTEM test); the target
cross-sections (microstrip & friends) are non-degenerate.
K > 2 QTEM line modes keep the non-orthogonal per-conductor basis
*between themselves* (per-mode ε_eff forbids the DD-066 Gram
mixing) — but the WP-U6 dual-basis projections now cover them too
whenever the port is multi-mode.  *(Revised by DD-196: K > 2 QTEM
line modes are the modal basis of the capacitance pencil, and the
dual-basis projections cover every multi-channel QTEM port.)*
**Files.**  ``ports/modal/factory.py`` (``_qtem_zeta_hybrid_modes``,
QTEM-branch extension, dual-basis projector construction);
``ports/declarative.py`` + spec docstrings (uniform layer-a
semantics); ``examples/straight_waveguide_microstrip.py`` (f_max
35 GHz, measured numbers);
``validation/qtem_multimode_port_probe.py``;
``tests/integration/test_unified_multimode_port.py`` (microstrip
3-mode composition incl. dual-projection unit response +
guidance/degeneracy raises).
---
## DD-069 — Per-channel Mur reference velocity for dispersive ports: investigated, DD-068 baseline retained
**Date:** 2026-07-11 (session 96; DD-068 documented follow-up
candidate — "Hybrid-Mur quality: per-channel v_p reference instead of
the port-wide f_calc").
**Status:** Investigated, both stages measured, **neither adopted** —
the DD-068 port-wide ``f_calc`` Mur reference is retained (developer
decision after the measurements below).  No code, test, or example
change ships from this entry; it is the record that closes the
follow-up candidate.  A branch (``dd069-dispersive-mur-reference``)
carries the full stage-1 implementation should it ever be revived.
**Problem.**  Every Mur-terminated channel takes its phase velocity
from ``Mode.gamma(ω_calc)`` at the single port-wide ``f_calc = f_max``
(``operator.py`` init loop).  A Mur-1st boundary annihilates exactly
one velocity, so on a dispersive channel (QTEM ζ-pencil hybrids,
DD-068; analytical/numerical TE/TM in the Mur fallback) the boundary
is exact at the top of the band and increasingly detuned toward the
channel's own cut-off, where ``v_p(ω) = ω/β(ω)`` grows — the honest
Mur class, but avoidably lop-sided (DD-068 measured hybrid max |S11|
−9.5/−7.8 dB above 1.2 f̂_c, medians −18.1/−10.5 dB on the
shielded microstrip).
**Stage 1 — band-minimax reference (measured, not adopted).**
A dispersive channel with a non-empty usable band
``[1.2·ω_c, ω_calc]`` would take ``v_ref = √(v_p(1.2·ω_c) ·
v_p(ω_calc))``, the geometric mean of the phase velocities at the two
band edges.  ``v_p`` is monotone-decreasing on that interval, so the
geometric mean equalises the residual mismatch
``|R| = |v_p − v_ref|/(v_p + v_ref)`` at both edges (the minimax point
for a one-velocity boundary).  ``β(ω_ref)`` comes from the same
Klein-Gordon parameters already carried on the Mode (``epsilon_r``,
``omega_c``) — no extra eigensolves.  TEM (``omega_c = 0``) keeps the
single-frequency evaluation *bit-identically* (no ``sqrt(v·v)``
round-trip); empty-band (``1.2·ω_c ≥ ω_calc``) and
evanescent-at-ω_calc channels keep the old reference / ``C0``.  The
same ``v_ref`` feeds the Mur coefficient ``r_m``, the TF/SF incident
delay ``τ_m``, and — through the unchanged init loop — every mode
including the DTBC-certified ones (inert there: ``update_e``
overwrites the naive Mur value for DTBC modes, so grid-aligned DTBC
pins stay bit-identical).
**Stage 2 — Higdon-2, measured and refuted.**  A single retuned
reference only *redistributes* one velocity's worth of error, so the
minimax buys the band edge at the band centre's expense (measured: H1
median −15.1 → −17.0 dB but H1 max −10.6 → −8.1 dB — a wash, not a
win).  The structural fix is the Higdon-2 product boundary: two
one-way factors at ``v_hi = v_p(ω_calc)`` and ``v_lo = v_p(1.2·ω_c)``,
implemented as a nested Mur recursion over three projection planes
(port + two interior) and three time levels, keeping the ``v_hi``
top-of-band exactness *and* nulling the band-edge velocity.
Analytically (single harmonic, real operator coefficients) Higdon-2
is 10–30 dB better than Mur-1 across the whole propagating band, with
a second reflection null exactly at 1.2 f_c; the boundary state
recursion is bounded over 4000 driven steps.  **But** in the coupled
TD run it regressed the QTEM fundamental catastrophically: max |S11|
−21.3 → **−2.6 dB** with 1.5 dB of transmitted-power loss, the damage
a sharp resonance at 17.0–18.1 GHz — just *below* the first hybrid's
cut-off (18.24 GHz).  Higdon-2 is a propagating-wave absorber; below
a channel's cut-off the mode is evanescent and the two-velocity
product of superluminal ``v_ref`` supports a near-cut-off boundary
resonance, which the shared dual-basis port reconstruction couples
back into the (propagating, plain-Mur) fundamental.  Mur-1st is
gently dissipative there — exactly why stage 1 left the fundamental
untouched (−21 dB).  The developer's "structurally never worse"
premise is disproved for the near-cut-off regime (cf. the DD-064
artificial-cut-off precedent — a measured, refuted upgrade is
recorded, not re-promoted).  Higdon damping (a δ term pulling the
poles off the real axis) could suppress the resonance but adds a
tuning parameter and degrades the deep null; not pursued given the
modest propagating-band gain.
**Decision.**  Stage 1 is a net wash (the H1 median gain trades
against the H1 max) and stage 2 regresses the operating mode, so
neither beats the DD-068 baseline enough to justify the added
machinery.  The **port-wide ``f_calc`` Mur reference is retained**;
this entry documents the closed follow-up.  The DD-068 sentence
"Mur-1st then uses the channel's true phase velocity at f_calc"
stands.
**Measured (shielded microstrip, identical mesh, 35-GHz band).**
* DD-068 baseline (retained): QTEM fundamental −21.2/−29.1 dB;
  hybrids max −9.5/−7.8 dB above 1.2 f̂_c, medians −18.1/−10.5 dB;
  leg-A below-16-GHz |ΔS| 5.2e-2/5.5e-2.
* Stage-1 band-minimax (not adopted): QTEM fundamental unchanged
  (−21.3/−29.0 dB); hybrids max −10.6/−8.4 dB, medians −15.1/−10.2 dB
  — the H1 median improves, the H1 max regresses; a wash.
* Stage-2 Higdon-2 (refuted): QTEM max −2.6 dB / power −1.52 dB;
  leg-A below-16-GHz |ΔS11| 1.16e-1 — the fundamental-invariance loss
  that ruled it out.
**Files.**  No production change (baseline retained): only this entry
and the STATUS session-96 log.  The full stage-1 implementation
(``_mur_reference_velocity`` in ``ports/modal/operator.py`` + the
dispersive-minimax / TEM-bit-identity / empty-band-guard tests) lives
on branch ``dd069-dispersive-mur-reference`` for future revival.

---

## DD-070 — On-disk project store: separation of simulation and post-processing

**Date:** 2026-07-12 (session 97; developer-initiated).
**Status:** Accepted; **fully implemented** (WP-S1…S9 + session-99
frequency/flux persistence + session-122 planned-run
pre-registration).

**Problem.**  The workflow was single-phase and in-RAM: ``run()``
marched to completion and returned everything in memory.  Failures at
production scale: a full-volume ``FieldTimeMonitor`` cannot hold its
snapshot list in RAM; and there was no way to watch a running
simulation converge or to continue a run that stopped too early
without restarting from zero.

**Decision.**  A **project directory** is the central artefact:
simulation streams into it on-the-fly, post-processing is a separate
read-only path over the same directory (live or after the fact),
finished/aborted runs resume bit-exactly.  Nine frozen decisions:

| # | Decision |
|---|----------|
| D1 | Project **directory**, not a monolith file (artefacts have write-once / streaming / overwrite lifecycles). |
| D2 | Streaming store = **HDF5 in SWMR mode** (live view; keeps ParaView/XDMF; no new dependency). |
| D3 | ``.run()`` returns a **lazy ``Project`` reader** — the same object ``open_project`` returns; live and batch post-processing are one code path. |
| D4 | **Bit-exact resume** — checkpoint the full solver state (E/H, CPML ψ, DTBC convolution history, band state machine, Mur prev-values); a seam transient would pollute the decaying tail being resolved. |
| D5 | The store is a **Level-B primitive**; Level A consumes it; RAM-only stays the Level-B default. |
| D6 | Store unit = **named run = one time-marching trajectory**; the S-matrix is **derived on read** from stored raw V/I — never stored — so partial fill and later fill-in are natural. |
| D7 | **Shared container for TD + eigen**: runs (streamable, TD) and results (one-shot); streaming/resume are TD-specific. |
| D8 | Geometry: **BREP** (exact round-trip) + **STL** (viz); STEP dropped *as a store format* — STEP import is DD-178. |
| D9 | Old ``save_project``/``load_project`` **deleted**; its writers reused inside the store. |

Rejected forks (developer, session 97): Zarr / hand-rolled append-log
(dependency / custom code); eager-in-RAM return; pragmatic E/H-only
resume (self-defeating against the <−100 dB port ethos).

**Layout.**  ``project.json`` (setup + frequency plan + run
index/status + run recipe), ``geometry.brep``/``.stl``, ``mesh.h5``
(write-once), ``runs/<name>/{results.h5 [SWMR-append], checkpoint.h5
[overwrite], fields.xdmf}``, ``eigenmodes.h5``,
``runs/<name>/fields_freq.h5`` (see below).

**Load-bearing implementation facts (WP-S1…S9, sessions 97–99):**

- **SWMR rules**: no object creation and no attribute edits after the
  mode switch → the recorded step count is the ``reference`` stream
  *length*, and mutable run ``state`` lives in ``project.json``.  The
  streaming sink (``FITTimeDomainSolver.sink``, flushed at every
  energy check) is the **single** write path.
- **Checkpointing** (``Checkpointable`` protocol): pure nested
  ``state_dict`` maps 1:1 onto the HDF5 tree (one recursive walker,
  cupy → host).  ``checkpoint.h5`` is overwritten via temp +
  ``os.replace`` (atomic; a resumer never sees a partial write).
  Same-operators rewind is **bit-identical** (``max|Δ| = 0``) on DTBC
  + CPML; an independent ARPACK TE/TM rebuild sits at ~1e-13, so
  resume reloads into the same operators where determinism matters.
- **Graceful abort is cooperative**: ``request_stop()`` checked at
  the top of each iteration → the break lands on a consistent
  leapfrog pair.  Level A traps ``SIGINT`` (checkpoint + ``aborted``
  + partial project); ``SIGUSR1`` = snapshot-and-continue (same
  consistency point, no stop).
- **Unbounded default**: ``total_time_steps=None`` marches on the
  energy criterion alone.  Subtlety: the energy-check cadence was
  tied to the step-count estimate, so unbounding one path shifted the
  stop step and broke in-RAM == streamed parity —
  ``energy_check_interval`` decouples cadence from cap; both paths
  resolve it identically.
- **``magnelio.resume(project, excited=…)``** rebuilds the run
  path-only from the stored recipe (``analysis/_recipe.py``);
  ``from_project`` reconstruction is idempotent on the stored
  (already consolidated) mesh.  Subtleties: the resume sink samples
  the reference at ``(step_offset + local_k)·dt`` (else the reference
  stream phase-shifts against the pre-resume tail); criterion knobs
  are treated as *one* setting (pass one → the other is disabled;
  pass neither → inherit the launch criterion); ``results.h5`` is
  reopened truncated to the checkpoint step (a hard kill can flush
  past it — the orphaned tail is dropped).  Acceptance: rebuilt +
  resumed run **bit-identical** to an uninterrupted one (TEM/DTBC
  line), including the derived S.
- **Monitors**: ``FieldTimeMonitor`` streams snapshots into
  ``results.h5`` (declared pre-SWMR, drained per flush — never
  accumulates in RAM) and is checkpointed → resumed monitors continue
  bit-exactly.  ``FluxTimeMonitor`` streams append-only likewise
  (session 99).  ``FieldFrequencyMonitor`` is different *in kind*
  (fixed-size running DFT sum, not append-safe): it gets its own
  ``fields_freq.h5``, written whole + atomically at each checkpoint —
  simultaneously the user's converging partial-DFT result *and* the
  resume source; ``n_completed`` ties it to ``checkpoint.h5`` (on
  mismatch after a crash between the two atomic writes: raise, never
  integrate from a wrong partial).  ``Project.monitors[name]``
  resolves all three kinds by name.
- **Planned-run pre-registration** (session 122): all planned
  excitations are registered ``pending`` up front, so finishing run
  *k* no longer flips ``status = "done"`` mid-analysis (live-watcher
  race, user-reported).  Watcher idiom: iterate ``project.runs``,
  skip ``state == "pending"``; polling ``status == "done"`` is
  race-free.  Fill-in never clobbers existing ``done``/``aborted``
  entries.

---

## DD-071 — Geometry authoring: Brick.from_corners + material-preserving Group

**Date:** 2026-07-14 (session 100; developer-initiated — a bundle of
geometry-authoring / lumped-element / thin-wire wishes sorted into a
roadmap, ``GEOMETRY_CIRCUIT_PLAN.md``).  This DD records the first two
work packages (Cluster 1, WP 1a + 1f); later cluster decisions land in
``DD-072…``.
**Status:** Accepted — implemented and merged behind the plan.
**Problem.**  Building a realistic multi-material assembly (the motivating
case: an SMA connector = PEC pin + PTFE dielectric + PEC shell) meant
placing each solid by hand, because the only multi-shape container was the
Boolean ``Union``, which *fuses* its operands into **one** solid carrying
**one** material (``operations.py``: ``Union.material = shapes[0].material``).
There was no way to translate/rotate a heterogeneous bundle as a unit while
each member keeps its own material.  Separately, an axis-aligned box could
only be given as ``origin`` + ``size``; a two-opposite-corners form (the
common CAD idiom) required the user to normalise min/max by hand.
**Decision.**
1. **``Brick.from_corners(p1, p2, *, material, name=None)``** — a
   classmethod that normalises min/max per axis and populates the ordinary
   ``origin``/``size`` fields, so the result is indistinguishable from a
   directly constructed ``Brick`` downstream.  No rounding: ``size = hi −
   lo`` is a plain FP subtraction (any ~1e-18 residue is far below OCC
   precision; a geometry constructor must not silently move coordinates).
2. **``Group(*shapes, name=None)``** — a new **compound-node category** in
   the geometry algebra: a *heterogeneous* container that **preserves each
   member's material and OCC solid**.  It is the material-preserving
   sibling of ``Union``.  Load-bearing consequences:
   - A Group has **no single** ``.material`` and **no** ``._occ_shape()`` —
     it is not a physical solid.  Its protocol is ``.members()`` (recursive
     leaf iterator, flattens nested Groups) + ``.bounding_box()`` (union of
     member boxes).  This is the one hard boundary: **CSG Boolean ops
     (``Union`` / ``Intersection`` / ``Difference``) reject a Group
     operand** (``_reject_group`` → ``TypeError``), because there is no
     single solid to cut/fuse.
   - **Transforms distribute over members.**  ``translate`` / ``rotate`` /
     ``scale`` applied to a Group return a new Group whose members are each
     transformed (recursively for nested Groups), so materials are
     preserved through placement.
   - The repeat helpers grow a **``group=True``** flag — the third branch of
     ``_apply_repeat`` beside ``unite=True`` — that aggregates repeated
     copies into a Group instead of fusing them into a ``Union`` or
     returning a bare list.  ``unite`` and ``group`` are mutually exclusive.
   - **Flattened at ``GeometryModel.add``**: a Group is expanded into its
     leaf members on insertion (recursively; lists may contain Groups), so
     the mesher, material filling and overlap layers **never see a Group**.
     This keeps the entire solver-facing pipeline (which keys materials by
     ``id(mat)`` and requires one ``._occ_shape()`` per shape) unchanged —
     the Group is purely an authoring-time convenience that vanishes before
     meshing.
**Rationale.**  A Group is deliberately *not* a Boolean op: fusing would
collapse the materials, which is exactly what the multi-material assembly
must avoid.  Keeping it a pure authoring container that flattens before the
mesh means no downstream code (mesher, filling, overlap check, sub-cell
machinery) needs to know Groups exist — the single flatten point in
``add`` is the whole integration surface.  The CSG-rejection is a hard
error rather than a silent "use the first member" so a heterogeneous bundle
can never leak a wrong material into a Boolean result.
**Consequences / scope.**  ``Group`` is exported from ``magnelio`` and
``magnelio.geometry``.  This is the foundation the roadmap builds on:
Cluster 1 continues with a standalone ``Face`` (optional material) + generalised
``extrude`` (1b/1d) and an abstract ``Curve`` + ``sweep``/``revolve`` (1c/1e);
Cluster 3 (discrete-port audit → lumped RLC → thin-wire) shares one
``Curve → grid-edge`` rasterizer.  Overlap policy 2a (same-material overlaps
allowed, compared by value) is a separate small change, not part of this DD.
---
## DD-072 — Standalone planar Face + generalised extrude

**Date:** 2026-07-14 (session 100; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 1,
WP 1b + 1d — the continuation of DD-071).
**Status:** Accepted — implemented and merged behind the plan.
**Problem.**  Solids could only be authored as CSG primitives + Boolean
ops; there was no way to give an arbitrary planar profile and sweep it into
a solid.  ``extrude`` existed but only for a *face of an existing solid*
selected by ``face_near`` — it could not take a free-standing profile.  The
motivating cases (spiral inductor, arbitrary microstrip pad, connector
cross-sections) start from a hand-drawn 2D polygon.
**Decision.**
1. **``Face(normal, points, offset=0.0, material=None, name=None)``** — a
   standalone planar polygon in an axis-normal plane.  ``points`` are
   in-plane ``(u, v)`` vertices; ``(u, v)`` map to the two axes orthogonal
   to *normal* using the **same convention as ``cross_section_polygons``**
   (normal ``x`` → u=y, v=z; ``y`` → u=x, v=z; ``z`` → u=x, v=y), so a Face
   and the cross-section of its extrusion share one coordinate frame.
   ``offset`` positions the plane along the normal axis.  Built by a new
   ``occ_backend.make_face`` (``BRepBuilderAPI_MakePolygon`` closed →
   ``BRepBuilderAPI_MakeFace(wire, planar=True)``).  Validated at
   construction: ``normal ∈ {x,y,z}`` and ``≥ 3`` points.
2. **Optional material** is the load-bearing design point.  A Face with
   ``material=None`` is a **construction profile** (input to
   ``extrude``/``sweep``, not a physical object).  A Face **with** a
   material is a **thin sheet** — but thin-sheet *physics* wiring (DD-035)
   is deferred, so:
   - a Face is not a ``_BaseShape`` (whose ``material`` is required); it is
     its own dataclass with ``material`` optional;
   - ``Mesh.from_geometry`` **rejects** any standalone Face up front with a
     clear ``NotImplementedError`` (before grid construction, which would
     otherwise die on the zero-thickness bounding box) — a zero-volume face
     must not silently mesh to nothing, and a ``material=None`` face must
     not reach the ``id(mat)`` material-keying.
3. **``extrude`` generalised** to ``extrude(shape, *, vector,
   face_near=None, material=None)``.  A **Face** input *is* the profile
   (``face_near`` unused; ``BRepPrimAPI_MakePrism`` runs directly on
   ``Face._occ_shape()`` — no ``find_nearest_face``); a **solid** input
   keeps the existing ``face_near`` selection.  The extruded solid needs a
   material: explicit ``material=`` wins, else the Face's own material, else
   (construction Face) a ``ValueError`` asks for one.  ``face_near`` became
   keyword-optional — backward compatible, since every existing caller
   passes ``face_near=``/``vector=`` by keyword.
**Rationale.**  Reusing the ``cross_section_polygons`` (u, v) convention
means no new coordinate mental model and guarantees consistency between a
profile and its swept solid's cross-sections.  Keeping ``material`` optional
lets the Face object carry the eventual thin-sheet material for free while
the physics is still deferred, and the up-front mesher rejection turns the
deferral into a loud, actionable error instead of a silent wrong result.
Making the Face-vs-solid dispatch live inside ``extrude`` (rather than a
separate ``extrude_face``) keeps one verb for "sweep a profile linearly",
matching how the later ``sweep``/``revolve`` (1c/1e) will also take a
profile.
**Consequences / scope.**  ``Face`` is exported from ``magnelio`` and
``magnelio.geometry``.  Thin-sheet meshing, ``sweep``/``revolve`` (Cluster 1
WP 1c/1e via the abstract ``Curve``), and the overlap-policy relaxation
(Cluster 2, 2a) remain future work.
---
## DD-073 — Curve (polyline / arc / spline / helix) + sweep/revolve; optimal bounding boxes

**Date:** 2026-07-14 (session 100; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 1,
WP 1c + 1e — continuation of DD-071 and DD-072).
**Status:** Accepted — implemented and merged behind the plan.
**Problem.**  There was no way to author a curved solid (a coil, a bent
trace, a solid of revolution).  The plan's motivating case — a spiral
inductor — is a rectangular profile (WP 1b ``Face``) swept along a helix.
That needs a *curve* object and *sweep*/*revolve* verbs.
**Decision.**
1. **``Curve``** — one abstract, OCC-backed 3D locus (a ``TopoDS_Wire``)
   with **no material** (1D is never a physical object on its own); a
   Curve exposes only ``_occ_shape()`` + ``bounding_box()``.  Consumers
   decide use: ``sweep`` (→ solid), later Cluster 3 (rasterise onto grid
   edges for voltage integration / thin-wire).  Four lazy classmethod
   constructors (in ``curves.py``): ``Curve.polyline`` (open wire),
   ``Curve.arc`` (3-point ``GC_MakeArcOfCircle``), ``Curve.spline``
   (``GeomAPI_PointsToBSpline``), ``Curve.helix``.  Cheap validation is
   eager (like ``Face``); OCC-dependent validation (collinear arc,
   degenerate points) stays lazy in the ``make_*`` helpers.
2. **Helix is exact, not sampled.**  ``make_helix`` builds the helix as a
   straight line in the ``(angle, height)`` parameter space of a
   ``Geom_CylindricalSurface`` (radius exact to machine precision), with
   ``breplib.BuildCurve3d`` forcing the 3D curve — **required**, or the
   edge carries only a pcurve on the surface and ``sweep``'s Frenet frame
   dies with ``Standard_NullObject``.  Parameter span
   ``last = turns·hypot(2π, pitch)``; handedness flips the 2D u-slope sign.
3. **``sweep(profile, spine)``** via ``BRepOffsetAPI_MakePipe``.  MakePipe
   uses the profile *at the position it already occupies*, so ``sweep``
   first **auto-positions** the profile: it reads the profile plane's
   normal + centroid (from the OCC face, so a transformed Face works too)
   and the spine's start point + unit tangent (``BRepAdaptor_CompCurve``,
   multi-edge safe), then applies a rigid ``gp_Trsf.SetDisplacement`` that
   moves the profile centroid onto the spine start with its normal along
   the start tangent (in-plane roll fixed by a deterministic perpendicular).
   The user need not pre-place the profile — the canonical call is
   ``sweep(Face(...), Curve.helix(...))``.  Verified: straight-spine volume
   is exactly area·length; helix/arc volume matches area·arclength to
   < 2 %; the tube is centred on the spine.
4. **``revolve(profile, axis, angle_deg, origin)``** via
   ``BRepPrimAPI_MakeRevol`` (no positioning ambiguity — profile used
   as-is; must not cross the axis).  Full-revolution volume matches Pappus
   to 1e-3.  ``axis`` accepts ``'x'/'y'/'z'`` or a vector.
5. **``bounding_box`` → ``brepbndlib.AddOptimal(shape, box, False, False)``**
   (a **correctness fix** these features force).  Plain ``Add`` bounds a
   freeform (B-spline) surface by the convex hull of its control poles,
   which over-sizes a swept / lofted / revolved solid by ~2× per axis
   (~8× mesh cells).  ``AddOptimal`` computes geometry-based bounds:
   **exact** for analytic primitives (box/cylinder/sphere/Boolean — and no
   slower there, so a safe drop-in; the old gap correction is dropped) and
   **tight** for freeform.  Measured: a 2 mm-radius helix coil now bounds
   to ±2.4 mm (was ±4–5 mm) and meshes on a ~10³ grid instead of a
   ~20³ one.  This also fixes the pre-existing ``loft`` bbox looseness.
**Rationale.**  A single ``Curve`` type (rather than PolylineCurve /
ArcCurve / … classes) matches the plan's "one locus, consumers decide use"
and keeps the future rasterizer (Cluster 3) pointed at one type.
Auto-positioning ``sweep`` matches how commercial suites present "sweep
along path" and removes the error-prone manual step of placing the profile
perpendicular to the path start.  The exact surface-helix (over a sampled
spline) honours the Genauigkeit-first design priority.  The bbox fix is not
optional polish: without it every curved solid inflates the grid, violating
the Effizienz priority and the "must scale to large geometries" goal.
**Consequences / scope.**  ``Curve`` exported from ``magnelio`` +
``magnelio.geometry``; ``sweep``/``revolve`` from ``magnelio.geometry`` (with
the other modifiers).  Cluster 1 (geometry authoring) is now complete
(WP 1a/1f/1b/1d/1c/1e).  Next: Cluster 2 (2a same-material overlaps,
compare by value) and Cluster 3 (discrete-port audit → RLC → thin-wire on
one shared ``Curve → grid-edge`` rasterizer).
---
## DD-074 — Same-material overlaps allowed (value-equal materials)

**Date:** 2026-07-14 (session 100; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 2,
WP 2a — the only Cluster-2 change; 2b "air is overwritable" was deleted
before implementation and must not be reintroduced).
**Status:** Accepted — implemented and merged behind the plan.
**Problem.**  ``GeometryModel.validate()`` reported *every* volumetric
overlap, forcing the user to partition with CSG (``Difference``/``Union``)
or set ``allow_overlaps=True`` globally.  But an overlap between two shapes
of the **same** material is physically unambiguous — the overlap region
gets that material whichever shape "wins" the cell classification — so
requiring the user to resolve it is friction with no benefit (and the whole
point of allowing overlaps is that users deliberately build them, e.g. an
L-shape from two overlapping bricks, or a ``Group`` of touching
same-material segments).
**Decision.**  Relax ``validate()`` so overlapping pairs whose materials
are **value-equal** are not reported; different-material overlaps still
raise ``GeometryOverlapError``.  Equality is
``Material.__eq__`` (the plain-dataclass value comparison over
name/ε/μ/σ/σ*/is_pec; color/alpha/visible are ``compare=False``) — **not**
``id()``: two distinct ``Material.pec()`` instances are the same material.
This is deliberately *not* identity, because the mesher keys its material
library by ``id(mat)`` and would otherwise treat two value-equal instances
as different.  The material *name* is part of the identity, so two
differently-named materials with identical physics are treated as different
(they are distinct library entries) and their overlap still raises.
Implementation: ``check_pairwise_overlaps`` gained an optional
``materials=`` argument; when supplied it skips value-equal pairs **before**
the expensive ``BRepAlgoAPI_Common`` Boolean (after the cheap AABB reject),
so a same-material overlap costs no intersection computation — this matters
for large assemblies of touching same-material primitives.  ``validate()``
passes ``[s.material for s in shapes]``.  The warning text and the
``allow_overlaps`` escape hatch are unchanged (the latter still bypasses the
whole check for the genuinely-different-material cases a user accepts).
**Rationale.**  Value equality over identity is required for the relaxation
to be useful at all (every ``Material.pec()`` call is a fresh instance).
Skipping before the Boolean keeps the check scalable.  Reporting
different-material overlaps preserves the safety the check exists for: those
*are* ambiguous (which ε fills the overlap?) and usually a modelling
mistake.
**Consequences / scope.**  Cluster 2 is complete (2a only).  No mesher
change beyond the existing ``validate()`` gate; the mesher still assigns one
library id per distinct material *object* (value-equal duplicates get
separate ids, as before — a pre-existing minor redundancy, out of scope
here).  Next: Cluster 3 (discrete-port audit → lumped RLC → thin-wire on one
shared ``Curve → grid-edge`` rasterizer).

---

## DD-075 — Discrete-port audit (3a): co-temporal power-wave fix + Thévenin-invariant gates; single-edge limit documented

**Date:** 2026-07-14 (session 101; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 3,
WP 3a — the foundation audit before the lumped-RLC generalisation 3b;
continuation of [[DD-042]]).
**Status:** Accepted — F3 fix + F1 doc + gates implemented; F4 documented as
a constraint on 3b.

**Problem.**  The plan's Cluster 3 builds a lumped-RLC companion model on top
of the existing ``PortOperatorLumped`` (a passive discrete port already *is*
a resistor ``Z0``).  But that operator was only smoke-tested (V/I ≠ 0, FFT
finite, energy bounded) and the one quantitative test
(``test_quarter_wave_stub_null``, S11-null at design frequency) was
``pytest.skip``.  "Audit before building on it": verify the matched-
termination reflection, the √W power-wave normalisation vs. DD-042, and the
``Σ e·dl`` convention.

**What the audit found.**

- **F2 — core physics is analytically sound.**  The semi-implicit Thévenin
  update solves ``i_port = (v_src − v_total)/(Z0 + Σβ)`` and injects it, so
  the post-update line integral is ``V = v_total + i_port·Σβ`` and hence
  ``v_src = V + Z0·I`` identically (I flows out of the + terminal into the
  line).  With the √W power-wave reference = ``Z0`` (the ``_LumpedModeStub``
  that ``AnalysisScatteringTD`` synthesises for ``compute_s_parameters``)
  this gives ``a₁ = v_src/(2√Z0)`` — **load-independent**, the defining
  property of a clean power-wave source — and ``S11 = (Z_in − Z0)/
  (Z_in + Z0)``.  The ``Σ e·dl`` sign/convention is correct.  So all three
  audit targets (√W, Σe·dl, Thévenin) pass by construction.
- **F3 — real bug: the Yee temporal half-step was applied to the lumped I.**
  ``compute_s_parameters`` multiplied *every* I channel by
  ``exp(+jω·dt/2)``.  That is correct for **modal** ports (V ∼ e at
  ``t^{n+1}``, I ∼ h at ``t^{n+1/2}``), but a lumped port's ``project_I``
  returns the ``t^{n+1}`` Thévenin current cached in ``update_e`` and
  *ignores h* (confirmed against the solver loop order), so lumped V and I
  are **co-temporal**.  The spurious ``ω·dt/2`` rotation caps a lumped
  port's achievable match at ``~π·f·dt/2`` (≈ −26 dB at band top on a λ/20
  mesh; measured: the co-temporal and temporally-corrected splits diverge by
  up to 0.9 dB at 14.75 GHz, dt = 2.24 ps).  The *spatial* de-stagger already
  had a lumped fall-back (``port_normal_dx`` absence → co-located split); the
  *temporal* one had no guard.
- **F4 — single-edge vs. distributed-mode mismatch (usage constraint, not a
  bug).**  A discrete port is a **single edge-chain**, a 2-terminal element
  on one grid edge; it does **not** span a distributed cross-section.  On a
  parallel-plate TEM line it therefore does not act as a clean termination:
  the mode occupies every transverse edge-column, so a single-edge load of
  value ``R`` terminates it as ``≈ (N_columns)·R`` (measured cleanly on an
  Nx=1 plate: effective ``2R`` — nulls at ``R = z_line/2``, ``−9.5/−4.4 dB``
  at ``z_line``/``2·z_line`` exactly per ``R_eff = 2R``), and on a wider mesh
  the localised load reflects the distributed wave resonantly
  (max\|S11\| → 0 dB for every ``Z0``, Nx = 1…8).  The clean modal-to-modal
  baseline on the *same* meshes is −168 dB, so this is the lumped port, not
  the line.  Consequence: a quantitative matched-termination / λ/4-stub gate
  is **not achievable** until a genuine single-edge line exists (thin-wire,
  Cluster 3c) — and **3b's RLC validation must use a single-edge context,
  never a distributed TEM line**.
- **F1 — stale doc:** the ``AnalysisScatteringTD`` docstring called lumped
  ports an uncovered "Limitation"; they are in fact fully wired into layer a
  (``_build_operator`` → ``build_lumped_port``, ``_modes_for_operator`` →
  ``_LumpedModeStub``).  **F5 — gotcha:** ``run()`` is unbounded by default
  (DD-070), and a weakly-absorbing lumped fixture never reaches
  ``energy_stop_db`` → pin ``total_time_steps`` in lumped tests.  **F6
  (benign):** the recorded lumped V/I sit at ``t^{n+1}`` while
  ``reference_signal`` is on the ``n·dt`` axis (a one-step offset); S-
  parameters extract ``a`` from V/I self-consistently, not from the
  reference, so this does not affect S — noted for direct V/I-vs-source
  comparisons only.

**Decision.**
1. **Fix F3.**  ``_LumpedModeStub`` grows an ``i_cotemporal`` property
   (``True``); **both** power-wave paths skip the ``exp(+jω·dt/2)`` factor on
   any channel whose mode reports ``getattr(mode, "i_cotemporal", False)`` —
   ``compute_s_parameters`` (the S-matrix) and ``destaggered_power_waves``
   (the ``result.a()``/``b()`` time-domain split, which carried the identical
   unguarded correction).  Real ``Mode`` objects lack the attribute → default
   ``False`` → the modal path is **bit-identical** (verified: the parallel-
   plate/modal S-parameter and a/b suites are unchanged).  A one-attribute
   signal keeps the two shared signatures stable.  The synthetic
   ``test_matched_forward_wave_has_vanishing_b`` (WP5.2) fed a lumped stub
   *staggered* I — encoding the old bug — and is corrected to co-temporal I,
   where the lumped matched wave gives ``b ≡ 0`` exactly (a tighter gate than
   the former O((ω·dt)²) bound).
2. **Fix F1** — the docstring now states lumped ports are covered and lists
   the genuine layer-a limitations (simultaneous multi-port drive, nonlinear
   materials, hand-tuned excitations).
3. **Replace the skipped λ/4 test** with two fixture-free physics gates
   (``tests/integration/test_discrete_port.py``):
   ``test_discrete_port_thevenin_invariant`` asserts ``V + Z0·I = s`` to
   machine precision (rel < 1e-12; validates √W + Σe·dl + sign, mesh-
   independent), and ``test_discrete_port_cotemporal_decomposition`` asserts
   ``result.S`` equals the co-temporal split and differs from the temporal
   one (the F3 regression guard, with teeth).
4. **Document F4** here and in the plan; the ``test_discrete_port`` module
   docstring records why the λ/4-null gate awaits 3c.

**Rationale.**  Because F4 makes a clean matched-termination null impossible
on any distributed fixture available today, the exact algebraic Thévenin
invariant ``V + Z0·I = s`` is the *stronger* gate — it pins the operator's
core physics at machine precision with zero wave-fixture confounders, where a
−40 dB null never could.  Fixing F3 in post-processing (not by re-centring
the operator's current) keeps the discrete port's own time-stepping untouched
and matches the physical fact that a lumped element ties V and I at one
instant (instantaneous Ohm's law), unlike a travelling wave's Yee-staggered
E/H.

**Consequences / scope.**  No change to the discrete-port operator, the
solver, or the modal path.  3b (lumped RLC companion model) inherits a
verified ``v_src``/``z_eff`` slot and a documented single-edge validation
requirement.  The degenerate rasterizer in ``ports/discrete/factory.py``
(argmin node-snap + single-axis edge run) is unchanged; it is subsumed later
by the shared ``Curve → grid-edge`` rasterizer whose first consumer is
``integrate_E`` (the F4 edge-multiplicity is exactly the normalisation that
rasterizer must get right).  ``DiscretePortSpec``/``build_lumped_port`` API
unchanged.

---

## DD-076 — Canonical curve rasteriser (Curve → ordered directed grid edges) + integrate_E

**Date:** 2026-07-14 (session 101; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 3,
the cross-cutting rasteriser + its first consumer ``integrate_E``;
continuation of [[DD-073]] (``Curve``) and [[DD-075]] (3a audit)).
**Status:** Accepted — implemented + validated.

**Problem.**  Every Cluster-3 grid consumer (voltage integration now; lumped
RLC and thin-wire later) must map an abstract ``Curve`` onto the primary-grid
E-edges it occupies.  ``build_lumped_port`` already contains a *degenerate*
rasteriser (argmin node-snap + a single-axis edge run,
``ports/discrete/factory.py``) that only handles an axis-aligned straight
segment.  The general problem — an oblique curve (a helix is oblique
everywhere), ordered and *signed* edges for ``Σ E·dl`` and current
continuity — needs one shared kernel.  **Guardrail: exactly one canonical
rasteriser**, or voltage integration and a thin-wire current could disagree
on which edges a curve occupies.

**Decision.**  A new package ``src/magnelio/circuit/`` (the home for Cluster-3
edge elements) with ``rasterize.py``:

1. **``rasterize_curve(curve, grid, *, samples_per_cell=4) → EdgePath``.**
   (a) *Sample* the wire at quasi-uniform arc length ≤ ``min_cell /
   samples_per_cell`` via a new ``occ_backend.sample_wire`` (a
   ``BRepAdaptor_CompCurve`` — which resolves multi-edge wire order and
   per-edge orientation — plus ``GCPnts_QuasiUniformAbscissa``), so no node is
   skipped.  (b) *Snap* each sample to its nearest primary node (per-axis
   independent on the rectilinear grid) and collapse consecutive repeats.
   (c) *Staircase*: walk the node chain in unit steps; a rare multi-axis jump
   between adjacent chain nodes is filled by a monotone x→y→z run.  Each unit
   step emits the primary E-edge at the **lower** node with ``sign = ±1`` and
   its length ``dl``.
2. **``EdgePath``** carries ``axes`` / ``ijk`` (lower node) / ``signs`` /
   ``dls`` / ``flat_indices`` (into the concatenated ``Ex|Ey|Ez`` layout — the
   ``FieldState`` / ``M_eps`` ordering), so both field-array and flat-vector
   consumers share one path.
3. **``integrate_E(field, curve, grid) → float``** — the first, read-only
   consumer: ``Σ sign · E · dl`` along the chain.  Exposed top-level from
   ``magnelio`` alongside ``rasterize_curve`` / ``EdgePath``.

**Rationale.**  A *signed* staircase makes the discrete line integral equal
the continuous one: the signed edge sum is a telescoping displacement, so for
a uniform (or any conservative) field the result depends only on the snapped
endpoints and an oblique curve integrates correctly despite the staircase
geometry.  Sampling the whole wire as one ``CompCurve`` sidesteps per-edge
orientation bookkeeping.  Snapping per-axis is exact on the rectilinear grid.
Placing the kernel in ``circuit/`` (not ``mesh/`` or ``ports/``) gives the
Cluster-3 elements one home and keeps ``ports/discrete`` free to adopt it
later without a mesh↔ports dependency.

**Validated** (``tests/unit/test_curve_rasterize.py``, 8 tests, all to
machine precision): a uniform field over a **diagonal polyline** integrates
to ``E·(B−A)``; over an **oblique helix** (108-edge / 108 mm staircase) to the
same net displacement (rel 1e-15) — the key path-independence property; a
non-uniform conservative field ``φ = x²`` (``E_x = −2x`` midpoint-sampled)
integrates to ``φ(A) − φ(B)`` **exactly on both uniform and graded grids**
(cell-midpoint of a linear ``E_x`` telescopes); flat-index and field-array
integration agree; reversing the curve negates the integral; an axis-aligned
segment yields the expected all-``z``/``+1`` edge run; a sub-cell curve and
``samples_per_cell < 2`` raise.

**Consequences / scope.**  ``integrate_E`` needs the field and the grid
(``FieldState`` carries no grid), so the signature is
``integrate_E(field, curve, grid)`` — a deliberate deviation from the plan's
2-arg sketch.  The degenerate rasteriser in ``ports/discrete/factory.py`` is
**not yet** replaced (a follow-up: ``build_lumped_port`` becomes
``rasterize_curve`` on a straight ``Curve``, and the resolved edges + ``dl``
match by construction).  F4 (DD-075) does **not** bite ``integrate_E`` — a
line integral is a single path, no transverse multiplicity — but the same
rasteriser is what a lumped element (3b) will use to place its companion
model, and there the single-edge-vs-distributed-mode normalisation must be
handled by the element, not the rasteriser.  Next: 3b lumped RLC on a
single-edge curve fixture.

---

## DD-077 — Trapezoidal RLC companion models (3b core); operator unification pending

**Date:** 2026-07-14 (session 101; ``GEOMETRY_CIRCUIT_PLAN.md`` Cluster 3,
WP 3b physics core; continuation of [[DD-075]] (the discrete-port audit) and
[[DD-076]] (the rasteriser)).
**Status:** Accepted — companion models implemented + validated; the
Port-protocol operator that consumes them is the next step (developer chose
the **unified** operator design, see "Consequences").

**Problem.**  3b generalises the discrete port's constant internal impedance
``Z0`` to a lumped **R/L/C** element.  A time-stepping solver needs the
element's constitutive relation as a per-step companion
``V^{n+1} = R_eq·I^{n+1} + V_hist`` (a constant equivalent resistance + a
history source), which drops into the discrete-port update
``i = (v_src − v_total)/(Σβ + Z0)`` by ``Z0 → R_eq`` and
``v_src → v_src − V_hist``.

**Decision (physics — developer sign-off).**  **Trapezoidal (bilinear)**
integration: 2nd-order like the FIT leapfrog and **energy-conserving for
L/C** — backward-Euler's numerical damping would corrupt the high-Q resonance
a lumped element usually models (its one cost, a known O(dt) response at a
hard-step discontinuity that decays over τ, is inert under the smooth erfc/
Gaussian excitations used in practice).  **Both series and parallel
topologies** (developer scope choice).  ``src/magnelio/circuit/companion.py``:

- ``SeriesRLC(R?, L?, C?)`` — shared current; ``R_eq = R + 2L/dt + dt/(2C)``;
  state ``{I, V_L, V_C}``; ``V_hist = [−(2L/dt)I^n − V_L^n] + [V_C^n +
  (dt/2C)I^n]``.
- ``ParallelRLC(R?, L?, C?)`` — shared voltage; Norton ``G_eq = 1/R + dt/(2L)
  + 2C/dt`` converted to Thévenin ``R_eq = 1/G_eq``, ``V_hist = −I_hist/G_eq``;
  state ``{V, I_L, I_C}``.

Absent elements (``None``) drop their term; ``SeriesRLC(R=Z0)`` *is* the
discrete port.  Each carries ``r_eq(dt)`` / ``v_hist(dt)`` / ``advance(i, v,
dt)`` / ``reset()`` + a ``state_dict`` for bit-exact resume.

**Validated** (``tests/unit/test_companion.py``, 7 tests): the ``r_eq`` /
``v_hist`` match the closed forms; a series-RL and series-RC step response
**converges** to ``i∞(1−e^{−t/τ})`` / ``i0·e^{−t/τ}`` (error falls ~O(dt) at
the step — the expected trapezoidal-at-a-discontinuity behaviour, asserted by
convergence not a magic tolerance); an underdamped series RLC rings at exactly
``ω_d = √(1/LC − (R/2L)²)`` (FFT peak ±2 %); a parallel RC converges to its
dual response.  The driver is a mini source-resistor circuit that mirrors the
solver solve (``Vs`` behind ``Rs`` ↔ ``v_src`` behind ``Σβ``), so it validates
the update structure too.

**Consequences / scope.**  The **operator unification is the next step**
(developer chose it over a parallel operator or a discrete-port extension): a
general ``LumpedElementOperator`` (EdgePath from the DD-076 rasteriser +
``CompanionElement`` + optional independent source) becomes the base, and
``PortOperatorLumped`` a thin ``SeriesRLC(R=Z0)`` special case.  Analysed as
**behaviour-neutral** (with ``r_eq=Z0, v_hist=0`` the update is byte-for-byte
the current discrete port, and an ascending axis-aligned segment rasterises to
all-``+1`` signs), but it touches the DD-070 checkpoint ``state_dict`` (now
also the companion state) and the recipe — so it is gated on the **full
resume/S-parameter suite staying bit-identical** and done as its own focused
pass, not rushed.  ``SeriesRLC`` / ``ParallelRLC`` are exported from
``magnelio.circuit`` (top-level export deferred until the operator makes them
usable in a run).

---

## DD-078 — Physical √W port-amplitude convention (κ pinning)

**Date:** 2026-07-15 (session 103; reframes [[DD-075]] F4 and the session-102
lumped-port investigation, `investigations/lumped_port/FINDINGS.md`).
**Status:** Accepted — implemented + gated (`tests/integration/test_port_units.py`).
**Problem.**  The analytical mode evaluators are Poynting-normalised to 1 W
(`ports/modal/coax.py`), but `discretize_modes` re-orthonormalises the runtime
profiles in the FIT M_ε inner product and *discards* the physical scale — the
recorded modal V/I live in a basis whose volts-per-unit κ is aperture- **and
mesh-dependent** (measured: κ = 7 916 / 10 620 / 3 958 / 2 799 across four
plate fixtures; pure Ny-refinement of the same geometry changes the implied
power by 4×).  Intra-port S-parameters cancel κ exactly, which is why nothing
was ever visible on same-cross-section fixtures.  But: (a) **heterogeneous
modal↔modal** S21 carried κ₁/κ₂ — an ε_r 1→4 TEM step measured |S21| = 1.886
(= 2×Fresnel) with |S11|²+|S21|² = 3.67 on a lossless line; (b) **mixed
lumped↔modal** S was meaningless — the session-102 "−78 dB point-feed mode
coupling" was exactly 1/κ, not physics (the field-level launch is the perfect
circuit-theory TEM wave; unitarity |S11|²+|S21|² ≈ 0.005 was the missed red
flag).  Developer directive: the modal machinery is the reference and must not
change; the intended convention is the commercial-style **mode amplitude =
√(1 W)**; lumped ports adapt.
**Decision.**  Per-mode physical pinning computed at build time inside
`PortOperatorModal._calibrate_v_i` (no dynamics touched):
- ``P₁`` = physical Poynting flux of the unit-coefficient travelling wave,
  ``P₁ = Σ_u ê_u·H_v·dA_u − Σ_v ê_v·H_u·dA_v`` with geometric area patches
  (primal edge length × dual node spacing, reconstructed from the plane's
  edge midpoints) and ``H`` the wave's physical field (analytic path: pre-γ ĥ
  is sampled A/m; numerical path: undo the dual-voltage convention,
  ``H = ĥ·M_μ/(μ₀·normal_dx)``).
- ``record_scale κ = √(|P₁|·Re Z_modal)`` — the **recorder** (only) multiplies
  projected V/I by κ; Mur/DTBC internals keep basis units bit-identically.
- ``source_scale = √(Re Z_modal)/κ`` — ``set_excitation`` scales the user
  waveform, which is thereby the incident power-wave amplitude **a(t) in √W**
  (developer-approved TD convention: default pulse peak 1 → 1 W peak
  instantaneous incident power).
- ``PortOperatorLumped.set_excitation`` scales by the Thévenin identity
  ``v_src = 2√Z0·a(t)`` (Z0 = 0 keeps the raw waveform); its recorded volts
  are already physical → no record scaling.
- Uncalibrated channels (evanescent at ω_calc; `calibrate=False` CW true-mode
  ports; band path) keep κ = 1 — behaviour unchanged.
**Validated.**  κ reproduces the field-probe measurement on all four fixtures
to 7e-4; the ε-step gate hits Fresnel (|S11| = 1/3, |S21| = √(8/9)) with
unitarity 1.000 ± 3e-4; the mixed gate moves lumped→modal |S21| from −78 dB to
−0.5 dB; identical-plane modal↔modal S is unchanged (κ cancels); full suite
green.
**Consequences / follow-ups.**  (i) Frequency monitors are now "fields per
1 W CW": the pre-existing ``FieldFrequencyMonitor.renormalize(result.
reference_signal)`` divides by the source spectrum, which since DD-078 is
a(f) in √W — verified to 1e-4 against the analytic √(z_line·1 W)/gap and
gated (``test_frequency_monitor_fields_per_1w_cw``); work item (ii) thereby
closed with zero new machinery.  (ii) The remaining −0.5 dB mixed-family deficit is *real*:
the single-column lumped port's V/I bookkeeping is not discrete-energy-
consistent by the transverse-coupling factor (1+f), f ≈ 1/Nx (sharpened
session-102 open item; converges away with transverse resolution; belongs to
the 3b lumped-element pass, NOT to a units fix — do not hard-code /(1+f)).
(iii) TE/TM κ inherits the ω_calc-fixed calibration approximation of the
existing V/I rescale; heterogeneous TE/TM port pairs should get their own
validation fixture when they become a use case.  (iv) Recorded signal
magnitudes changed (now physical): recipes/resume are internally consistent,
but results.h5 files written before DD-078 mix conventions with new runs.

---

## DD-079 — Unified LumpedElementOperator (3b part 2)

**Date:** 2026-07-15 (session 103; completes ``GEOMETRY_CIRCUIT_PLAN.md``
Cluster 3 WP 3b — part 1 was [[DD-077]]; developer chose the unification in
session 101).
**Status:** Accepted — implemented + gated (`tests/integration/test_lumped_element.py`).
**Decision.**  ``ports/discrete/operator.py`` now hosts the general
``LumpedElementOperator``: a trapezoidal companion element
(:class:`SeriesRLC` / :class:`ParallelRLC`, DD-077) in series with an
optional Thévenin source on a chain of grid edges with **per-edge field
components and orientation signs** (EdgePath-shaped, so the DD-076 canonical
rasteriser can drive it for future curve-based elements).  Per-step update
    i = (v_src − v_hist − v_total) / (r_eq + Σβ),
then ``element.advance(i, r_eq·i + v_hist, dt)``.  ``PortOperatorLumped``
is the thin special case (axis-aligned two-point chain, all-+1 signs,
``element = SeriesRLC(R=Z0)`` by default) with an unchanged constructor
surface; ``Z0`` stays the power-wave reference (``_modes_for_operator`` and
the DD-078 ``2√Z0`` source convention read it).  ``PortSpecLumped`` gains
``element=`` (deep-copied per built operator so spec instances stay
pristine); the DD-070 recipe serialises it (old recipes load unchanged);
``state_dict`` gains an ``element`` group (pre-unification checkpoints
without it still load).  ``SeriesRLC``/``ParallelRLC`` are now exported
top-level (the DD-077 deferral is over — the operator makes them usable).
**Behaviour-neutrality (the hard gate).**  For a pure resistance the
companion returns ``r_eq = Z0`` (same float) and ``v_hist = 0.0``; IEEE
subtraction of 0.0 and multiplication by ±1.0 are exact, so the update is
arithmetically identical.  Verified: a two-discrete-port S-run produces
**byte-identical V/I/S** before vs after the refactor (A/B npz compare),
and the full suite is unchanged.
**Validated (new gates).**
- Passive impedance: a passive ``element=`` port on a TEM plate presents
  exactly ``−Z_trap(ω)`` in ``DFT(V)/DFT(I)`` with the bilinear map
  ``jω̃ = j(2/dt)tan(ωdt/2)`` — rel < 1e-6 for series RLC, bare L, and
  parallel RC (a per-step KVL identity, immune to the (1+f) transverse
  factor because it reads the element's own V/I).
- Recipe + ``state_dict`` round-trips (incl. legacy-checkpoint tolerance).
- **Resume bit-exact:** an RLC-backed source (R+L+C history state) aborted
  at a checkpoint and resumed is byte-identical to one uninterrupted run.
**Follow-ups.**  (i) ``build_lumped_port``'s degenerate two-point
rasteriser still stands beside the canonical DD-076 one (needs OCC-free
handling before subsuming — a ``Curve``-based lumped element spec is the
natural vehicle).  (ii) The DD-078 sharpened item stands: the single-column
port's V/I is not discrete-energy-consistent by (1+f) — a 3c/thin-wire-era
question.  (iii) Next per plan: 3c thin-wire (own DD).

## DD-080 — Holland/Simpson thin-wire model (3c)

**Date:** 2026-07-16 (session 104; completes ``GEOMETRY_CIRCUIT_PLAN.md``
Cluster 3 WP 3c — the roadmap's largest item; developer approved the scope
(staircase paths, endpoint set, geometry-object API, lossless v1), the
**paired L/C correction** and the analytic-κ₀ policy in the planning pass).
**Status:** Accepted — implemented + gated (``tests/unit/test_thin_wire.py``,
``tests/integration/test_thin_wire_line.py``, ``test_thin_wire_antenna.py``).
**Problem.**  A conductor much thinner than the cell (bond wires, probes,
coil filaments) cannot be resolved as a solid.  A bare PEC-masked edge chain
behaves like a conductor of the *grid's* equivalent radius r₀ ≈ 0.2·Δ —
radius-blind, with the wrong per-length inductance.
**Decision.**  New geometry leaf category ``ThinWire(curve, radius, name=…)``
(``geometry/wire.py``): implicitly PEC (``.material`` → ``Material.pec()``),
``_occ_shape()`` = the curve's wire, bbox = the curve's (radius NOT included —
it is a sub-cell parameter, not a feature size).  The mesher splits wires off
before filling/classification (``mesher.py``), anchors grid planes on the
curve's OCC vertex coordinates + bbox extents (axis-aligned polyline segments
land exactly on grid lines; endpoint snap displacement > 0.3·cell warns), and
applies the model at mesh-build time through the ONE canonical rasteriser
(DD-076 ``rasterize_curve → EdgePath``) in ``mesh/thin_wire.py``:
- **PEC edge chain** (``mask_thin_wires``): the path's E-edges OR into
  ``pec_mask_edges``, BEFORE ``couple_face_material_pairs`` so no DD-053
  ladder is certified through a wire edge.
- **Paired Holland/Noda–Yokoyama correction** (``correct_thin_wire_materials``,
  after the coupling pass): per axis-aligned segment the 4 encircling H-faces
  scale M_μ by ``m_f = ln(Δ_f/a)/ln(Δ_f/r₀)``, ``r₀ = κ₀·Δ̄``,
  ``Δ̄ = (d_u⁻ d_u⁺ d_v⁻ d_v⁺)^{1/4}``, and the co-located radial E-edges
  (exactly the faces' DD-053 ladder partners) scale M_ε by ``1/m`` — the
  Holland & Simpson (1981) closure L'·C' = εμ in material-matrix form
  (Noda & Yokoyama 2002).  **The DD-053 pair identity M_ε·M_μ is preserved
  exactly** (machine-gated), so the exact discrete travelling wave survives
  on the wire; a μ-only correction would leave the wire wave at ~0.73c
  (a = 0.05Δ) and Z₀ ~12 % low — why the roadmap's literal "correct M_μ"
  was extended, with developer sign-off.
- **Encoding via existing channels only:** faces → cat-2
  ``A_face_free = m·(current M_μ)·L_dual/(μ₀·μ̄)`` (the
  ``couple_face_material_pairs`` convention; composes multiplicatively over
  cat-1 dielectric values), edges → cat-1 ``eps_avg = eps_eff/m`` with
  ``sigma_avg = NaN`` (σ stays staircase).  No new category, no solver change;
  ``build_M_mu``/``build_M_eps``/CFL/2D mode solvers/serialisation all see it
  natively.  Precedence: conformal-solid cat-2 faces/edges win (warned);
  PEC-masked radial edges (the monopole footpoint) skip silently.
- **Composition rule:** requests are collected globally over all wires, each
  face/edge takes the **minimum** m (never a product) — conservative toward
  the bare grid at staircase corners and between parallel wires < 2 cells
  apart (warned).
- **κ₀ policy (developer-approved):** ``KAPPA0 = e^(−γ)/2^{3/2} ≈ 0.19854``
  (square-lattice Green's function) as a named constant; gate T3 measures the
  grid's own bare r₀ (**measured ≈ 0.18–0.27 Δ** depending on extraction,
  inside the asserted [0.15, 0.30] window); recalibration only once, with
  sign-off and a derivation — no silent fitting.
- **Validity:** ``a ≥ 0.30·Δ_min`` raises (use a resolved cylinder);
  ``0.20–0.30·Δ_min`` warns (m < 1, dt shrinks ≤ ×0.86); the 1 %
  ``A_face_free`` floor is unreachable (min m ≈ 0.744, asserted).  CFL is
  handled by the existing machinery: the ε-side 1/m enters
  ``compute_min_effective_eps`` → dt × √(1/m) (×0.73 at a = 0.05Δ) —
  conservative (the physical wave speed is unchanged); a pair-aware CFL is a
  follow-up.
**Endpoints (v1, developer-selected).**  (a) PEC solid: the shared node joins
the masked-edge sets — current continuity is topological; the solid's
conformal faces take precedence in the last ring (O(1-cell) end error).
(b) Open end: nothing to do (current null emerges); Holland's known
end-capacitance bias (few %) documented.  (c) Lumped gap: two wires (or a
chain with a skipped edge) + ``PortSpecLumped(element=…)`` driving exactly
the gap edge — THE single-edge fixture DD-075 F4 deferred the quantitative
gates to.  (d) PMC wall: **image-theory CORRECTION to the plan** — the H
field of a perpendicular current is tangential to the wall, so PMC mirrors
it ANTI-directed: current null, i.e. a PMC wall is the ideal line OPEN (no
fringing), not a current-maximum mirror; the electric mirror (monopole,
current maximum) is the PEC wall.  Both are gated.
**Validated.**
- Unit (20): factor formula uniform+graded; ring stencil == the 4 faces with
  nonzero discrete-curl circulation (cross-checked against
  ``build_curl_matrix``) for all 3 axes; boundary clipping; **pair product
  preserved to 1e-15**; corner min-rule; precedence + silent masked-edge
  skip; cat-1 composition; floor unreachable; radius/anisotropy warnings;
  CFL minima; ``mesh.h5`` byte round-trip; OCC end-to-end
  (validate-with-contact, anchor planes, store ``kinds``/``radii``).
- Line gates (square PEC duct, closed form Z₀ = (η₀/2π)ln(1.0787·s/d)):
  **T1** Z₀ radius sweep a/Δ ∈ {0.05, 0.1, 0.2} — median Z_in within 5 %
  (measured 0.2/0.5/0.9 %), matched |S11| ≤ −15 dB (measured ≈ −22.5 dB),
  ln-tracking Z₀(0.05Δ)−Z₀(0.2Δ) = 83.1 Ω ± 15 % (measured 3 % off), and a
  both-axes-graded variant within 5 % (pins the Δ̄ rule).  **T2** λ/4 stub
  null (short → gap feed → open) at c/(4·L_eff) ± 4 %, L_eff = span − gap
  cell (exact two-stub series resonance).  **T5** wire ending ON a PMC wall:
  short(PEC)→open(PMC) resonator hits c/(4·L_eff) ± 2 % — the PMC end is an
  ideal open.  **T3** correction OFF: bit-identical across the radius sweep
  (radius-blind) and the bare plateau solves to r₀ ∈ [0.15, 0.30]·Δ.
  **T6** project-backed run with a wire resumed across a checkpoint seam is
  **bit-exact** (the wire lives in the stored consolidated mesh; resume
  never re-meshes).
- Antenna gates: **T4** center-gap-fed dipole (2h = 30Δ, Ω ≈ 14.4) — first
  Im Z_in = 0 at 0.44–0.50 of the half-wave frequency (measured 0.456,
  = 2h ≈ 0.457λ — the classical few-% thin-dipole shortening), R_in ∈
  [50, 90] Ω (measured 75 Ω).  **T5b** monopole on PEC: f_res within 2 % of
  the dipole, R_in ≈ R_dipole/2 ± 20 %.
- **(1+f) open item CLOSED by measurement** (``investigations/thin_wire/``):
  on the genuine single-edge line the mixed-power deficit
  |S21|²/(1−|S11|²) = 1.0000–1.0015 (vs 1/(1+f) ≈ 0.89 on the Nx=8 plate) —
  the DD-078/079 deficit is transverse multiplicity on distributed lines,
  not a port-bookkeeping defect; nothing to hard-code, standing instruction
  satisfied by doing nothing.
**Persistence.**  ``geometry.json`` gains schema-additive ``kinds``/``radii``;
wires are excluded from the STL (viz-only); the v1 reader exposes loaded
geometry without the wires + a warning (re-meshing a loaded project would
drop them; resume uses the stored mesh and is unaffected).
``GeometryModel.validate()`` excludes wires from the volumetric overlap check
(an endpoint inside a PEC solid is the documented monopole topology).
**Known limitations (v1, documented + warned).**
(i) Transverse **anisotropy** at the wire degrades the correction (~11 % low
Z₀ at 2.5:1 single-axis grading; measured) — warned above 1.5:1; grade both
transverse axes alike (the both-axes-graded gate passes at 5 %).  A
direction-resolved in-cell model (Edelvik/Ledfelt) is the follow-up.
(ii) Corner inductance is conservative (min rule), open ends carry Holland's
end-capacitance bias, and a wire within ~1 cell of a bbox/PML face loses part
of its ring (clipped).  (iii) Extreme grading where a face's own extent is
within 0.1 log units of r₀ keeps the bare value (warned).
**Non-goals (v1).**  Oblique in-cell segments (staircase only); wire-wire /
wire-sheet junctions; distributed Z'(ω)/skin loss (DD-077 companions are the
vehicle); insulated/coated wires; wires through modal port planes; ThinWire
reconstruction from a loaded BREP (metadata only); pair-aware CFL; subsuming
the discrete-factory two-point rasteriser (still a separate follow-up).

---

## DD-081 — Lossy metals, bulk sigma physics gate, sigma* (Cluster A)

**Date:** 2026-07-16 (session 105; ``MATERIAL_MODELS_PLAN.md`` Cluster A —
cluster order A→B→C→D and the power-loss sufficiency decision fixed by the
developer in the planning pass).
**Status:** Accepted — implemented + gated (``tests/unit/test_materials.py``,
``tests/unit/test_operators.py``, ``tests/integration/test_lossy_materials.py``).
**Problem.**  Three foundation defects blocked the material-models roadmap:
(i) ``Material.__post_init__`` forced ``sigma = inf`` on ``is_pec=True`` — a
pure marker (``build_M_sigma`` overrides PEC with ``pec_value=0.0``; PEC is
realised by edge masking) that made a finite-σ "lossy metal" unrepresentable;
(ii) bulk σ was wired end-to-end (conformal ``build_M_sigma``, semi-implicit
``α_E/β_E``) but had NO physics gate; (iii) ``sigma_m`` was a silent dead
dataclass field (``_alpha_H = ones``, no ``build_M_sigma_m``) — promised by
the dataclass and spec §3.1 since day one, ignored by the solver.
**Decision.**
- **Lossy metal (A1):** the σ=∞ override is dropped (grep-verified: no
  consumer read it).  ``Material.lossy_metal(name, sigma, mu=1)`` sets
  ``is_pec=True`` with finite σ retained — ``is_pec`` STAYS the single
  field-solve classification (all ~10 consumers unchanged, field solution
  bit-identical to PEC, gated), σ/μ are consumed only by loss models
  (Cluster B power-loss; R_s = √(ωμ₀μ_r/2σ)).  ``is_lossy_metal`` property
  (finite check excludes legacy stores carrying σ=inf on plain PEC);
  ``lossy_metal`` rejects σ ≤ 0, σ = inf.  Store round-trips by value —
  no schema change.
- **σ gates (A2):** parallel-plate TEM fixtures vs the EXACT
  ``γ = jω√(µε)·√((1−jσ/ωε)(1−jσ*/ωμ))``; two-length ratio
  ``S21(L2)/S21(L1)`` cancels the lossy-fill port mismatch (modal port is
  built lossless, |S11| ≈ −25 dB; residual = multiple-reflection ripple
  ~r²·e^(−2αL1) at the low band edge).  Measured: α 1.0 %, β 0.22 %
  (gates 2 %/0.5 %); lossy half-space Fresnel |r| 0.6 %, complex 0.9 %
  (gates 1.2 %/2 %) — the discrete staircase interface sits dz/2 in FRONT
  of the material plane (one-sided clamped E-edge sampling; de-embed with
  d − dz/2, shift-swept); transverse-only σ leaves the Ey mode lossless
  (|S21| < 0.006 dB).
- **σ* implemented (A3):** ``build_M_sigma_m`` — exact staircase mirror of
  ``build_M_mu``'s bulk part (same clamped one-sided face sampling, same
  geometric factor, ``pec_value=0.0``; conformal cat-1/2 overrides a
  recorded non-goal, matching M_sigma's staircase policy), unit-gated
  ``M_sigma_m == M_mu/μ₀`` when ``sigma_m ≡ mu``.  Solver:
  ``α_H = (M_μ−σ*Δt/2)/(M_μ+σ*Δt/2)``, ``β_H = Δt/(M_μ+σ*Δt/2)`` — the
  E-side form; WP-R5 donor faces (M_μ=0) stay frozen (σ* dropped there).
  With σ* = 0 the coefficients reduce BIT-EXACTLY to the lossless ones
  (x/x == 1.0; suite unchanged).  All three update kernels (CUDA fused,
  Numba fused, stencil) already took per-face ``α_H/β_H`` — no kernel
  change; CPML ``update_H(fields, beta_H)`` inherits the lossy β_H
  consistently.  No new state — checkpoints untouched.  Physics gate:
  σ*-filled line vs exact γ, α 1.2 %, β 0.22 % (gates 2.5 %/0.5 %).
**Non-goals (recorded).**  Conformal averaging of σ*/dispersive material
boundaries (staircase v1); thin conductive sheets (δ ≳ thickness) and
gyrotropic ferrites (project scope exclusions, 2026-07-16); surface-loss
consumption of lossy-metal σ arrives with Cluster B (DD-082).

---

## DD-082 — Perturbative power-loss wall losses (Cluster B)

**Date:** 2026-07-16 (session 105; ``MATERIAL_MODELS_PLAN.md`` Cluster B).
**Status:** Accepted — implemented + gated (``tests/unit/test_surfaces.py``,
``tests/integration/test_wall_loss_q.py``, ``test_wall_loss_monitor.py``).
**Problem.**  Wall losses are the dominant loss mechanism for cavities and
air-filled guides, but resolving the skin depth volumetrically (µm at GHz)
is infeasible.  The developer accepted the perturbative power-loss method
(``P = ½R_s∮|H_tan|²dA`` on the PEC-solve fields) as SUFFICIENT — the
self-consistent broadband SIBC stays deferred (Cluster D, optional).
**Decision.**
- **B1 — surface enumeration** (``mesh/surfaces.py``):
  ``enumerate_pec_surfaces(mesh, bc_pec_faces=…) → list[WallSurface]`` —
  per-tag (material id / BC face name) arrays of the wall-adjacent
  tangential-H SAMPLES with footprint-area weights (per face and
  tangential component the weights tile the full face area; corner
  samples accumulate both walls).  Covers material-PEC solids (staircase
  cell classification, faces between different PEC materials excluded,
  solid faces flush on the domain boundary skipped) AND PEC domain
  walls (cells already PEC excluded — no double count where a solid
  meets a wall).  ``inv_l_dual`` converts state → physical H using the
  SOLVER dual-length convention (``_build_avg_d``: boundary entry =
  FULL first/last cell) — the geometric half-cell misreads boundary
  samples ×2 (measured: PMC-wall Hx; a 1.182 flat error in the plate
  fraction, decomposed exactly as 1.3/1.1).
- **B2 — eigenmode Q** (``postprocessing/wall_loss.py``):
  ``wall_loss_Q(result, mode=0, sigma=…, mu=…) → WallLossQ`` — Q = ωW/P
  with W = ¼(eᵀM_ε e + hᵀM_μ h) (scale-invariant in the mode
  normalisation), per-tag P breakdown + partial ``Q_of(tag)`` (1/Q
  additivity gated to 1e-12).  Lossy-metal solids (DD-081) bring their
  own σ/μ_r; plain-PEC solids and BC walls take the caller's ``sigma``
  (missing → clear error).  ``AnalysisEigenmode`` now records
  ``boundary_conditions`` in ``solver_info`` (omitted faces default
  PEC) so the postprocessing can enumerate the domain walls.  Gates:
  copper TE₁₀₁ 20×10×30 mm vs the closed form (Pozar 6.46,
  cross-checked in-session against an independent explicit surface
  integration): +0.22 % at 1 mm cells, +0.05 % at 0.5 mm (O(h²));
  the SAME cavity as an air volume in a lossy-metal shell (material
  path, σ from the material) reproduces the BC-wall Q to 1e-4.
- **B3 — TD monitor** (``monitors/wall_loss.py``): ``WallLossMonitor(
  freqs, reference_plane=(axis,pos), sigma=…, bc_faces=…)`` — running
  DFT of ONLY the wall samples plus one reference cross-section
  (surface storage ~N², never a volume).  KEY DESIGN: results are the
  scale-free ratio ``dissipated_fraction = P_loss(f)/P_flow(f)`` with
  P_flow from the FIT identity ``P = ½Re Σ ê·ĥ*`` (no area weights in
  the grid-quantity basis; boundary H states rescaled ×½ to the
  physical half-cell patch).  Both quantities are quadratic in the
  states, so the run's global mesh-dependent state scale cancels —
  measured session 105: the volume states are FIT grid quantities
  ê = E·l, ĥ = H·l_dual times a GLOBAL constant C set by the port
  (measured C·dy_port = 1: 1 mm → 1000/m, 0.5 mm → 2000/m, graded →
  1051.75/m) — the M_ε-basis scale DD-078 pinned at the port recorders
  but which still lives in the volume states.  ``power_loss(P_in)``
  scales to Watts.  Gates: copper parallel plate vs α = R_s/(η₀b):
  max 0.14 % over 2–10 GHz; copper TE₁₀ waveguide vs the closed-form
  α_c over 1.4–2.3 f_c: max 1.3 % (bounded run — sub-cutoff pulse
  content makes the energy stop useless in a closed guide).
**Consequences / found issues.**  (i) The state-scale measurement shows
the EXISTING field-monitor family reports mesh-dependent absolute
values: ``FluxTimeMonitor`` multiplies state products by patch areas
(the ê·ĥ identity needs none), and Field monitors return raw states —
correct only where C·l = 1 (e.g. fully uniform cubic grids).  Recorded
as a follow-up (align the family on the ê/ĥ convention or pin the
volume-state scale — the DD-078 analogue for states); NOT changed in
this cluster to avoid silently altering existing results.
**Non-goals / follow-ups.**  Conformal face areas (staircase over-counts
oblique/curved walls ~4/π on cylinders — why the pillbox TM₀₁₀ gate is
deferred to the conformal follow-up; flat-wall fixtures gate exactly →
DONE in [[DD-087]]); surface roughness multipliers (Hammerstad/Huray) on
R_s → DONE in [[DD-088]]; WallLossMonitor project-store/recipe/checkpoint
integration (v1 is in-RAM on ``monitors=[…]`` runs) → DONE, see the
addendum below; modal-port faces are simply not passed as ``bc_faces``
(no automatic BC discovery in the monitor — the attach protocol only
sees the mesh).

### Addendum (2026-07-17, session 109) — WallLossMonitor store integration

Package 6 of ``FIELDS_AND_LOSSES_PLAN.md``; no new DD, this amends B3.

**Problem.**  The v1 monitor was RAM-only: a project-backed run dropped
its wall-loss result, and a resumed run silently lost the monitor
entirely (see the whitelist note below).

**Decision.**  A WallLossMonitor is the same KIND as a
FieldFrequencyMonitor — a fixed-size running DFT, not an append stream —
so it follows the session-99 Freq pattern (``a4af028``) exactly: its own
result file ``runs/<name>/wall_loss.h5``, written whole and atomically
(temp + ``os.replace``) at each checkpoint, tagged with the
``n_completed`` that ties it to ``checkpoint.h5``.  It is NOT in the
SWMR ``results.h5`` for the same reason the Freq monitor is not: an
in-place overwrite is not SWMR-consistent.  Written AFTER
``checkpoint.h5`` and with the same ``n_completed``, so a crash between
the two leaves ``wall_loss.h5`` older and the resume step-check rejects
the stale accumulator rather than integrating on from a wrong partial
DFT (gated by mutating the attribute).

**One structural difference from the Freq pattern, and it drives the
file layout.**  A frequency monitor's accumulators ARE its result; a
wall-loss monitor's result is a REDUCTION of them (P_loss/P_flow per
tag).  A reader gets only a run directory, so recomputing the reduction
would mean loading the mesh, re-enumerating the PEC surfaces and
re-resolving materials — a second place that produces, and could get
wrong, the same number.  So ``wall_loss.h5`` carries BOTH: the reduced
per-tag fractions (what ``_LoadedWallLossMonitor`` serves — making
"reader == in-RAM monitor" true by CONSTRUCTION, gated bit-exactly) and
the raw ``h_bins``/``ref_bins`` (what a resume reloads).  The reduction
costs ``n_freqs × (n_tags+1)`` floats next to accumulators orders of
magnitude larger.  Tags travel as their own JSON list because they are
heterogeneous (material ids are ints, BC walls are face-name strings)
and HDF5 group names are not.

``Project.monitors[name]`` resolves the new kind alongside time/flux/freq
— the user still knows only the name, not the kind or the file (DD-070).
The recipe carries the spec (reference plane, sigma/mu, bc_faces, and the
DD-088 roughness via the store's own ``_roughness_to_dict``, so there is
one format for the one concept).

**Found on the way (worth recording).**  ``_serialisable_monitors`` in
``analysis/_recipe.py`` is a WHITELIST, deliberately (a resume must not
reconstruct a ``_CallAtStep`` SIGINT control monitor — its abort would
re-fire).  The cost is that a new persisted monitor kind is SILENTLY
DROPPED from a resumed run until it is added there; both the resume gate
and the stale-file gate failed on exactly that before the entry was
added.  Its call site also still claimed "Only FieldTimeMonitors are
streamed + resumable" — stale since session 99; corrected.

**Gates** (``tests/integration/test_wall_loss_store.py``, 6): reader ==
in-RAM monitor bit-for-bit (fractions, ``f``, ``power_loss(P_in)``, tag
types preserved); resume across a checkpoint seam == uninterrupted run
(``assert_array_equal``); recipe round-trip incl. roughness; a rough
monitor's persisted fractions are the ROUGH ones (ratio == K(f) to
1e-12, i.e. the multiplier rides the recipe rather than silently
dropping to smooth); a run without the monitor writes no file and loads
unchanged (legacy projects); a mutated ``n_completed`` raises.

---

## DD-083 — Pole-residue DispersionModel (Cluster C1)

**Date:** 2026-07-16 (session 105) · **Status:** implemented
One general mechanism for frequency-dependent permittivity:
`eps(omega) = eps_inf + sum_p r_p/(j*omega - a_p)` with real poles and
complex-conjugate pairs stored once (`materials/dispersion.py`,
`DispersionModel`, exported top-level).  Debye (multi-term), Lorentz
(under- and overdamped branches), Drude and Djordjević–Sarkar are
*constructors* on this single form, not separate models — the solver
runs one recursion for all of them (DD-084).
Decisions:
- **Passivity check at construction is mandatory** (the plan's
  requirement): `Re(a_p) <= 0` for every pole, real poles need real
  residues, and `eps''(omega) >= 0` on 128 log-spaced samples across the
  declared validity band `f_band`.  `Re(a_p) = 0` is allowed only for
  the **real Drude DC pole** (with `r > 0`), whose trapezoidal update is
  exactly the semi-implicit conductor with `sigma = eps0*r` — undamped
  oscillatory poles (`Re = 0, Im != 0`) are rejected.  Every constructor
  defaults a sensible band; the generic constructor requires one.
- **Drude via partial fractions** carries that DC pole plus one
  relaxation pole (`a = -gamma`, `r = -omega_p^2/gamma`); the negative
  residue is fine — passivity is a property of `eps''`, not of residue
  signs.
- **Djordjević–Sarkar at 2 poles/decade**, `eps_inf`/`delta_eps` solved
  exactly from (`eps_r`, `tan_delta`) at `f_ref`.  At that density the
  comb ripple is negligible against the model's *inherent* causal
  tan-delta slope (~5 % per 1.5 decades from `f_ref` at 4.3/0.02) —
  densifying further does not flatten it; it IS the Kramers–Kronig
  behaviour the sigma_eff shortcut lacks.
- **`Material.dispersive(name, model, mu=, sigma=)`** sets
  `epsilon = (eps_inf,)*3`; `__post_init__` enforces the match (the mass
  matrix and the CFL limit read `epsilon`) and excludes `is_pec`.
  A static `sigma` may coexist (runs through the standard σ channel).
  `is_lossless` is False for any dispersive material; `dispersion`
  participates in Material equality (DD-074 overlap checks work).
- **Store schema-additive**: `_material_to_dict` gains a `dispersion`
  key (complex poles flattened to re/im quadruples for JSON); the one
  serialiser pair covers the mesh.h5 material table AND geometry.json.
  Old stores load unchanged; new stores with dispersive materials
  round-trip to equality.
Gates (`tests/unit/test_dispersion.py`, 20): every constructor vs its
closed form at machine precision; DS pins (`eps_r`, `tan_delta`) at
`f_ref` to 1e-12 and stays flat to 2 % / 6 % over ±0.5 / ±1.5 decades;
all rejection branches; Material integration; store round-trip.

---

## DD-084 — Trapezoidal ADE for dispersive materials (Cluster C2)

**Date:** 2026-07-16 (session 105) · **Status:** implemented
`solver/dispersion.py` (`DispersionOperator`) realises DD-083 models in
the FIT-TD leapfrog by the ADE method with the **DD-077 trapezoidal
convention** — per pole one polarisation-current state on the E-edges of
the dispersive region only.
Mechanics (derived and gated):
- Discrete Ampère on a dispersive edge:
  `M_eps_inf de/dt + M_sigma e + J|_{n+1/2} = C^T h`, pole recursion
  `J_p^{n+1} = k_p J_p^n + c_p g (e^{n+1}-e^n)` with
  `k_p = (1+a dt/2)/(1-a dt/2)`, `c_p = r_p/(1-a dt/2)`,
  `g = eps0*A_dual/l_primal` (the M_eps geometry factor).  Substituting
  the recursion splits the midpoint current into (a) a coefficient
  `W = g*sum_p w_p Re(c_p)` (`w_p` = 1 real / 2 pair) that enters BOTH
  sides of the semi-implicit E-update — folded into `alpha_E`/`beta_E`,
  so **all three update kernels stay untouched** — and (b) a history
  term `beta_E * sum_p w_p Re((1+k_p)/2 J_p^n)` subtracted from `e`
  right after the curl kernel.  `save_e` at the top of the iteration
  stashes the fully corrected `e^n`; on every edge no later stage
  rewrites (all bulk dispersive edges) the implicit solution is exact.
  A-stable for every passive pole at any dt (trapezoidal).
- **Edge subsets staircase-by-classification**: the identical clamped
  one-sided cell lookup as `build_M_sigma`'s bulk sampling; PEC-masked
  edges excluded at build time.  Conformal averaging of dispersive
  boundaries stays the recorded non-goal; the DD-081 interface
  convention applies (discrete interface dz/2 in front of the material
  plane — transmission needs no de-embedding, reflection does).
- **Memory**: index-subset storage only (never full-field); real poles
  (Debye/DS — the common substrate case) use float64 states, conjugate
  pairs complex128 stored once.  Elementwise xp ops, cupy-compatible
  (`bind` moves states to the backend).
- **CFL from eps_inf**: `Material.epsilon = eps_inf` drives the existing
  `courant_dt` chain unchanged — the high-frequency wave speed is the
  stability-relevant one; in-band `eps' > eps_inf` only slows the wave.
  Gated: production dt identical to a static `eps_inf` fill.
- **Checkpoint**: solver `state_dict` gains a schema-additive
  `dispersion` key (`{mat_id: {J0..}}`); `e_prev` is deliberately NOT
  state (rewritten by `save_e` before every use).  Old checkpoints load
  unchanged.
- **No-dispersion path bit-identical**: without dispersive materials the
  coefficient expressions are the exact master formulas (no `+0.0`
  detour) — gated by array equality.
- **Found by the exact-equivalence gate**: a factor-2 error in the
  W coefficient (midpoint ½ applied twice) that the line physics nearly
  averages out at small dt (beta was off only 0.2 %) — the Drude-DC ≡
  conductor gate (< 1e-10 after 200 steps) pinned it exactly.  This is
  why the reduction gates exist.
Measured physics (`tests/integration/test_dispersive_materials.py`, 7,
11×6 cross-section parallel-plate fixtures, 2–10 GHz, dz = 0.5 mm):
- Debye line (relaxation mid-band) vs exact `gamma(omega)`: alpha 1.8 %
  (4–10 GHz), beta 0.46 % full band.  The low band edge shows the
  DD-081 Fabry–Pérot residual amplified by the dispersive port mismatch
  (|S11| ≈ −16 dB vs −25 dB on the σ line: the modal port is built
  lossless at `eps_inf`, in-band `eps'` is higher) — oscillating sign,
  −9.9 % at 2 GHz, not an ADE error.
- Debye+σ half-space vs complex Fresnel with
  `eps_c = eps_debye(omega) − j sigma/(omega eps0)`: 0.68 % — the ADE
  and the σ channel share one denominator and are gated COMBINED; the
  Debye pole set moves r by 15 % beyond σ-only (the gate rejects a
  σ-only reference).
- Lorentz slab vs exact transfer matrix: resonance dip on frequency,
  complex S21 within 0.03.
- Drude slab: cutoff switch (0.37 → 0.96 across `f_p`) with |S21|
  matching the exact slab to **0.005 dB / 0.0012 complex** (planar
  grid-aligned interfaces are nearly exact).
- DS line: alpha vs the causal model 3 % full band, beta 0.5 %; the
  narrowband σ_eff shortcut is REJECTED by the same measurement (its
  1/f tan-delta misses alpha by > 30 % at the band edges, its constant
  `eps'` misses the causal phase slope by > 3× the DS beta error).
- Disk resume (WP-S8 chain): a Lorentz+σ fill resumed across a
  checkpoint seam is bit-identical to an uninterrupted run — dispersion
  survives the mesh.h5 material round-trip, complex pole currents
  survive checkpoint.h5.
Non-goals / follow-ups (recorded): magnetic dispersion `mu(omega)`
(structurally identical on H-faces; σ* covers narrowband magnetic
loss); conformal averaging of dispersive boundaries; C4 vector fitting
of tabulated `eps(f)` data onto the pole-residue form (the passivity
check is the acceptance filter); dispersive media extending into CPML
(the ψ recursion uses the W-folded `beta_E` consistently, but the pole
delta accounting misses the ψ corrections — keep an eps_inf buffer
before the PML, as every FDTD vendor recommends).
---

## DD-085 — Physical volume states (C = 1 at the source) + physical monitors

**Date:** 2026-07-16 (session 106; package 1 of `FIELDS_AND_LOSSES_PLAN.md`,
developer decision "pin at the source" from the session-105 planning pass).
**Status:** Accepted — implemented + gated (`tests/integration/test_physical_states.py`;
measurement record `investigations/state_scale/FINDINGS.md`).
**Problem (measured, [[DD-082]]).**  The volume states were FIT grid
quantities times a GLOBAL constant C set by the excitation (plate
fixtures: 1 mm → C = 1000/m, 0.5 mm → 2000/m, graded → 1455/m; C_e = C_h
exactly; C tracks the E-polarisation cell size).  Field monitors returned
raw states, `FluxTimeMonitor` multiplied states by patch areas — absolute
monitor values were mesh-dependent, correct only where C·l = 1 (uniform
cubic grids).  The session-106 measurement (state-scale probe) also
localised the per-family conventions:
- **Solver kernels** are the pure FIT incidence form → native states ARE
  grid quantities (`e = E·l_primal` [V], `h = H·l_dual` [A]).
- **Modal TF/SF** injected the M_ε-orthonormal basis profile → C = the
  basis scale × source_scale (the [[DD-078]] κ story one level deeper).
- **Discrete port** read `V = Σ e·dl` and injected `β·i/dl` — a
  self-consistent *field interpretation*, equal to grid quantities only
  on uniform grids (latent misscale on graded grids).
- **Plane wave** injected `β_E·H_inc [A/m]` / `β_H·E_inc [V/m]` — field
  interpretation again.
**Decision.**  Pin C = 1 at the source for every excitation family; the
volume states are thereafter physical FIT grid quantities, and monitors
convert states → fields locally with no port knowledge (the
attach-sees-only-the-mesh protocol survives):
- **Modal** (`_calibrate_v_i`): the state scale of the unit-coefficient
  wave is computed at build time,
  `C = sqrt(S_H / ⟨ĥ_post-γ, H·l_dual⟩_Mμ) / κ` with
  `S_H = Σ_u ê_u·H_v·dv_dual − Σ_v ê_v·H_u·du_dual` and H the physical
  A/m shape (analytic path: stored pre-γ profile; numerical path:
  `ĥ·M_μ/(μ₀·ndx·l_partner)`); Z cancels, profile scales cancel, the
  stored ĥ enters linearly as the projection-metric partner (derived
  from the calibration guarantee I = V/Z).  Then `source_scale /= C`,
  `record_scale *= C` — recorded V/I are analytically unchanged
  (measured vs master: max 2.9e-15 on V/I and S21; S11 shifts only on
  its DTBC floor).  Validated exact on five fixtures (plate uniform ×2 /
  graded / ε_r = 4, TE10); the measured C is kept as
  `PortOperatorModal.state_scale` for introspection.  Uncalibrated
  (evanescent, CW true-mode) channels keep C = 1 — unchanged.
- **Discrete port**: voltage form — `project_V = Σ ±e`, injection
  `e += β·i` (drops both `dl` factors).  Uniform grids: identical
  analytically; graded grids: the latent inconsistency is FIXED.
- **Plane wave**: injects grid quantities (`H_inc·l_dual`, `E_inc·l`,
  solver dual convention).  The absolute gate exposed a PRE-EXISTING
  bug: all 12 H-side TF/SF corrections had the sign of the kernel's
  `H = a·H − β·curl` outer minus wrong — measured TF amplitude 0.39
  of nominal with 0.19 SF leakage (master, uniform grid, in field
  units).  Fixed; now: monitor peak 0.99995 of amplitude, SF leakage
  −91 dB.
- **Monitors**: Field monitors divide each staggered sample by its own
  edge/dual length before cell-centre averaging → E [V/m], H [A/m]
  (uniform-grid outputs unchanged — C·l = 1 there); `FluxTimeMonitor`
  is the FIT identity `P = Σ e·h` in watts with boundary-h terms ×½
  (solver dual = full end cell vs physical half patch, the DD-082
  precedent); `WallLossMonitor` is a scale-free ratio — unchanged.
**Gates** (`test_physical_states.py`): grid independence — the same
plate problem on uniform and graded grids gives the same |Ey|, |Hx| per
1 W CW (vs analytic, 2 %) and the same flux energy (1 %; pre-fix the
graded flux was off by (C·l)² ≈ 0.5–3.3); flux through a matched line
= incident pulse energy (1 %; probe: 1e-4-class); per-edge pin identity
`ê·source_scale = (√Z/B)·l` to 1e-9 on the graded plate; discrete
project_V reads grid-quantity volts exactly on a graded grid; plane-wave
monitor peak = amplitude (5 %); the DD-078 1 W-CW frequency-monitor gate
stays green.
**Consequences.**  (i) 0.x MINOR break, developer-accepted: absolute
monitor values on non-uniform grids change (they were wrong); results.h5
written before DD-085 mixes conventions with new runs for *monitor*
data (V/I and S are unaffected).  (ii) Heterogeneous port pairs whose
modes have different C (different polarisation/shape on shared grids)
had a latent |S21| scale error C₁/C₂ in the old convention; all suite
fixtures pair equal-C ports (why S never showed it) — the pin removes
the error class.  (iii) Tests that build raw states must write grid
quantities (`E·l`), see updated `test_port_units` gate 1 /
`test_monitors` renormalize fixture.  (iv) The plane-wave TF/SF H-sign
fix changes every plane-wave result (amplitude was 0.39× nominal).
---

## DD-086 — In-repo vector fitting of tabulated eps(f) onto the pole-residue form (C4)

**Date:** 2026-07-16 (session 106; package 2 of `FIELDS_AND_LOSSES_PLAN.md`,
the C4 follow-up recorded in [[DD-084]]).
**Status:** Accepted — implemented + gated (`tests/unit/test_vector_fit.py`,
`tests/integration/test_from_table_line.py`).
**Decision.**  Measured permittivity tables enter the [[DD-083]]
pole-residue form through `DispersionModel.from_table(f, eps,
n_poles=None, f_band=None, tol=1e-3, max_poles=30)`:
- `eps` accepts complex `eps' − j·eps''` or an `(eps_prime, tan_delta)`
  pair; `f_band` defaults to the table span.
- **In-repo Gustavsen/Semlyen vector fitting**
  (`materials/vector_fit.py`, pure NumPy linear algebra — no new
  dependency): real-coefficient partial-fraction basis (conjugate pairs
  as two real columns), [Re; Im]-stacked least squares, pole relocation
  via the eigenvalues of `A − b·ĉᵀ`, per-iteration left-half-plane
  flipping (the standard VF stability rule).  Both canonical start sets
  are tried per order — log-spaced REAL poles (smooth relaxation data;
  without them the DS continuum stalled at 3e-3 with collapsing pole
  counts) and weakly damped complex pairs (resonant data) — best fit
  wins.
- **Automatic order** (`n_poles=None`): the order grows until the max
  relative table error beats `tol` AND the candidate passes the
  passivity filter; the cap raises an actionable error naming the best
  error and the `n_poles` escape hatch.  An explicit `n_poles` is
  accepted at whatever error it reaches (noisy measured tables) — only
  passivity is enforced.
- **Passivity is the acceptance filter** (the DD-083 mandate): the
  `DispersionModel` constructor rejects active/unstable fits with the
  offending frequency.  Gain data (`eps'' < 0` in the table itself) is
  rejected up front with its frequency.  Measured: 1 % noise overfit at
  an explicit n=6 lands in a clearly active pole set (`eps'' = −0.58`)
  — the filter is load-bearing, not theoretical.
**Validated.**  Debye table → 1 pole at 3e-16 (machine precision);
Lorentz → the exact conjugate pair (2.8e-6, off-grid < 1e-3); DS
continuum (5 decades) → 13 poles at 9.1e-4 on-grid / 4.2e-4 off-grid;
1 %-noise table with `tol = 0.04` (≈ noise max-norm) recovers the CLEAN
1-pole model to 8e-4.  Full chain: a two-relaxation table (knees 3 and
8 GHz) fitted and run on the DD-084 TEM line matches gamma computed by
complex interpolation FROM THE TABLE — alpha 1.9 % (≥ 4 GHz, the DD-084
Fabry–Pérot convention), beta 0.38 %; the dispersionless counterfeit
reference is rejected at 34 % beta error.
**Consequences / notes.**  (i) The noisy-data workflow is
auto-order-with-raised-tol (the max-norm criterion counts noise peaks;
`tol` above the noise max-norm lets the smooth low-order model win) —
documented in the docstring and the cap error message.  (ii) The fit
never produces a DC pole (`a = 0`), so from_table materials carry
conduction loss inside the pole set; a separate static sigma still
composes via `Material.dispersive(..., sigma=)`.  (iii) The passivity
check remains band-sampled (128 points) — a fit can in principle dip
negative between samples; unchanged from DD-083 (shared risk class).
---

## DD-087 — Conformal wall areas + conformal H sampling for wall loss

**Date:** 2026-07-16/17 (sessions 106–108; package 3 of
`FIELDS_AND_LOSSES_PLAN.md`).
**Status:** Accepted — implemented + gated
(`tests/integration/test_conformal_wall_area.py`; measurement record
`investigations/conformal_wall_area/FINDINGS.md` + probes).

**Problem ([[DD-082]]).**  Staircase face counting over-books a curved
PEC wall by exactly 4/π (measured 1.2732, resolution-independent —
the acceptance instrument), which put the perturbative wall-loss Q of
curved cavities ~21 % low; the pillbox TM010 gate was deferred to this
package.  Curved cavities are the accelerator norm.

**Decision — three mechanisms, all schema-additive, staircase path
bit-preserved.**

1. **Geometric PEC area channel.**  The DD-051/DD-053 face data is
   built for M_μ, not for area bookkeeping: `couple_face_material_pairs`
   overwrites `A_face_free` with the LC-consistent pair value, and
   `BRepAlgoAPI_Section` degenerates on planes lying on a shape
   bounding box (tangent planes, grid-snapped lids), reporting zero
   PEC on fully covered faces.  The mesher therefore carries a
   separate geometric channel: `pec_area_geom_out` re-evaluates
   degenerate planes at `plane ± deflection` and per face takes the
   MAX of both sides ("shift towards the PEC" — the wall lands in the
   adjacent air cell, where the live H samples are; the opposite
   choice parks lids inside PEC cells and measured 0.39× wall sums).
   → `FaceMaterialData.A_face_pec`, never NaN (staircase cell rule
   off the candidate set), written BEFORE the coupling pass.
2. **Divergence cell vector + signed-jump corner split.**  Per cut
   cell `w = Σ_faces A_face_pec·n_out` gives `‖w‖ = A_wall` exactly
   for ONE plane cut.  A corner cell holding a flat lid AND the curved
   mantle adds them as vectors (`|a+b| < |a|+|b|`, pillbox area 0.955,
   Q error growing on refinement).  The flat family is split off by
   the SIGNED JUMP `A_PEC(p+δ) − A_PEC(p−δ)` across the face's own
   plane: wall lying IN a plane is the set of face points with PEC on
   one side and non-PEC on the other, so |jump| is that wall's area
   and the sign names the owning (non-PEC) side; a face merely
   shadowed by a wall standing in front of it does not jump.
   `A_wall = Σ|w_flat| + ‖w − w_flat‖`.  REJECTED candidate: "exactly
   one of the two d-faces fully PEC" mistakes shadow for wall and
   over-books the through-cylinder (which has no flat wall at all) by
   24 %.  → `FaceMaterialData.A_face_pec_jump` from `pec_frac_jump`
   (the ±δ sections already computed for the max convention carry it
   for free).  Measured areas: cylinder side 0.9997/0.9999, pillbox
   total 0.9991/0.9996, lid channel alone 0.9987/0.9993 (1/0.5 mm).
3. **Conformal H sampling: uncut-only booking with a normal-direction
   walk (the DD-082 inheritance the Q gate exposed).**  With correct
   areas the pillbox Q still diverged (+3.8 % → +17.9 %); the
   measured root causes are that CUT-FACE STATES ARE NOT CLEAN GRID
   INTEGRALS in either convention: (a) fully-masked faces are
   Faraday-dead (`C e = 0` ⇒ `b ≡ 0`; their loss share grew
   16 % → 35 % on refinement — the divergence), and (b) partially-cut
   faces mis-read resolution-INDEPENDENTLY (~+18 % power at generic
   grid phase) because their rim edges live under the DD-051 sub-cell
   metric — a flux reading `b/(μ0·μ̄·A_free)` was measured and
   REJECTED (the discrete contour is the four full edges, so the
   effective area lies between A_free and A_full; the sub-cell-scale
   centred fixture hid this by symmetry: 10² = 6² + 8² puts lattice
   points exactly on the circle, and its +0.55 % turned out to be a
   phase fluke against −13/−18 % at shifted centres).  The estimator
   therefore books weight EXCLUSIVELY onto uncut, Faraday-live faces
   (`_face_alive_views`, ground truth `edge_material.pec_mask`),
   found by a ≤ 4-step walk that displaces the candidate cell along
   the inward wall normal read off the wall vector itself
   (air = −sign(w) per axis — geometry-blind; a walk along the
   component's own axis is structurally wrong: a z-invariant mantle
   cuts every z-face in the column, dropping the mantle's whole
   z-weight, measured 18-25 %).  Booked weight/area 0.997-1.000.
   Sampling H_tan a small step inside the volume along the surface
   normal is the standard perturbative-loss practice.

**Gates** (`test_conformal_wall_area.py`): cylinder side area < 0.5 %
vs 2πRL; pillbox total area < 0.2 % (corner cells); pillbox TM010 Q vs
`Q = x01·η0/(2·Rs·(1+R/H))` — measured **−11.1 % at 1 mm
(10 cells/radius) → −7.4 % at 0.5 mm**, IDENTICAL at generic grid
phase (−10.9/−6.9 %; the pre-repair estimator scattered +0.5 → −18 %),
gated as envelope + convergence + phase robustness.  The residual is
100 % inward sample-position bias: the J1 position-pullback
experiment recovers ALL fixtures to ±1.9 % — pure position, O(h).
DD-082's axis-aligned fixtures (TE101 +0.05 %, plate, TE10) never
enter the conformal branch and stay green on the untouched staircase
path.

**Consequences.**  (i) Absolute wall-loss numbers on conformal meshes
change (they were wrong: the area by 4/π-class factors, the sampling
by the cut-face reads); staircase meshes bit-identical.  (ii)
`WallSurface.area` is an explicit field — the historical `Σweight/2`
convention does not survive 3-component sampling.  (iii) Old stores
load `A_face_pec`/`A_face_pec_jump` as None and fall back to the
staircase path.  (iv) `wall_loss_Q(mu=)` is RELATIVE permeability
(passing μ0 costs √μ0 ≈ 1/892 in R_s — measured the hard way).
(v) Known O(h) residual: the inward sample-position bias (−11 % at
10 cells/radius, −7 % at 20, converging; the J1 position-pullback
ceiling test recovers ±1.9 %, so the residual is 100 % position).
**The two-point wall extrapolation was built and REJECTED by
measurement** (session 108, code on branch
`dd087-wall-extrapolation-rejected`): better or equal on every fixture
(0.5 mm mean −7.2 → −6.0 %) but the phase spread WIDENS 0.45 → 2.6
points, and it lands nowhere near the ±2-3 % target.  The reason is
fundamental to a walk-based stencil: one finite difference along the
step `s` yields `∇H·s` while the extrapolation needs `∇H·n̂`, and
`d_in` quantises the normal onto 26 directions — up to 45° off on a
curved wall, so the step runs partly TANGENTIAL, and a Cartesian
component (`Hx = −Hφ·y/r`) varies tangentially as strongly as
normally.  Measured, consistent: `|v2|/|v1| = 0.88` on extrapolated
mantle samples where J1 says the field RISES inward; a nonzero wall
offset makes Q monotonically WORSE (0 → 0.25 → 0.5: −7.40 → −7.66 →
−8.05 %) — a longer lever on the wrong gradient.  Nodal-line sign
flips were suspected and RULED OUT (3.0 % of weight).  The ceiling
test only succeeds because it knows the TRUE radius and the EXACT
field shape.  **Two well-posed successors, both recorded, neither
implemented (developer decision 2026-07-17, "state of the art"
discussion):**
  (a) *Exact wall plane per cut cell.*  The DD-087 geometry channels
  (`A_face_pec` on all 6 faces + the signed jump) determine the cut
  cell's wall PLANE, hence the true normal AND the true `d1` — then
  `H_wall = H1 − d1·(∇H·n̂)` with a full central-difference gradient
  (still linear, still DFT-compatible).  This is precisely what the
  commercial cartesian FIT codes hand their loss integrator and we
  currently withhold from ours; it is the wohlgestellt version of what
  failed here.
  (b) *SIBC* (the deferred material-plan Cluster D) is the strategic
  answer: it makes losses self-consistent in the TIME domain (lossy-S
  parameters, not just eigenmode Q) and removes post-hoc sampling
  entirely — the route the cartesian competition takes.  FEM codes
  dodge the problem by construction (curved higher-order elements
  evaluate H_tan ON the wall).
(vi) The pillbox
eigenmode fixture needs `n_modes ≥ 4` (ARPACK returns nothing at 1)
and probe resolutions must divide the lid heights (h = 0.35 mm puts
forced planes next to snapped geometry planes → sliver seams,
growth 67; f stays +0.2 % but the surface-sampled Q degrades to
+32 % — production meshes route through the DD-059 anchors).
---

## DD-088 — Conductor surface roughness as one factor K(f) on R_s

**Date:** 2026-07-17 (session 109; package 4 of
`FIELDS_AND_LOSSES_PLAN.md`).
**Status:** Accepted — implemented + gated
A rough conductor dissipates more than a smooth one of the same
footprint: the current follows a longer path over the profile, and once
the skin depth drops below the profile height the field starts to
resolve individual protrusions.  On the electrodeposited copper foils
of real boards this is not a correction but a factor of two in
conductor loss above ~10 GHz — the dominant error term of an otherwise
exact loss chain.
The perturbative chain (DD-082, conformal since DD-087) evaluates
    P_loss = 1/2 * R_s(f) * sum(w * |H_tan|^2)
per frequency, on a lossless PEC field solution.  Roughness enters that
expression through exactly one scalar.
Carry roughness as a frequency-dependent multiplier on the surface
resistance, `R_s,rough(f) = K(f) * R_s,smooth(f)`, K >= 1 — pure
postprocessing, no solver change, no field-solution change.  ONE
lossless solve therefore serves every roughness variant, which is also
how the gates are built (a single eigen run, two `wall_loss_Q` calls; a
single TD run, two `WallLossMonitor`s).
`materials/roughness.py` holds the models as frozen dataclasses behind
an abstract `SurfaceRoughness.factor(f, sigma, mu) -> K`; frozen means
they join `Material` equality and the store with no further machinery.
Both models are functions of the roughness scale relative to the skin
depth `delta = 1/sqrt(pi*f*mu*sigma)` only:
* `Hammerstad(rms_height)` — `K = 1 + (2/pi)*arctan(1.4*(Rq/delta)^2)`.
  The classical curve fit; one datasheet number.  Its ceiling K -> 2 is
  structural (the arctan saturates), which is exactly where it stops
  being right for strongly roughened foil.
* `Huray(radius, coverage, base_ratio=1)` — the physics-based snowball
  model, `K = base_ratio + (3/2)*coverage/(1 + delta/a +
  delta^2/(2a^2))` (Bracken DesignCon 2012 eq. 5; identical to Polar
  AP8195 eq. 1, whose leading `A_matte/A_flat` generalises Bracken's
  `1`).  `coverage = N*4*pi*a^2/A_flat` is the sphere surface per unit
  tile area.  The asymptotes are its content: `K -> base_ratio` when
  `delta >> a` (the field does not resolve the spheres) and
  `K -> base_ratio + 1.5*coverage` when `delta << a` — a ceiling that
  follows the profile instead of saturating at 2.
  `Huray.cannonball(rz)` closes the gap to what datasheets actually
  publish: the fixed 14-sphere stack (9+4+1) on a tile of side 6a with
  its height identified with Rz gives `a = Rz/16.73` (the published
  rounding of `4*sqrt(3)*(1+sqrt(2))`) and a CONSTANT
  `coverage = 56*pi/36` — Rz sets the radius alone.
Two channels, mirroring DD-082's sigma handling: lossy metals carry
their own (`Material.lossy_metal(..., roughness=)`, schema-additive
store field via the single `_material_to_dict` pair), and PEC walls
(plain-PEC solids, BC walls) take a caller override on `wall_loss_Q` /
`WallLossMonitor` alongside their `sigma`.  `Material.__post_init__`
rejects roughness on anything that is not a lossy metal: it multiplies
`R_s(sigma)`, so without a sigma there is nothing for it to act on —
caught at construction, not at loss-evaluation time.
1. `surface_resistance(f, sigma, mu, roughness=None)` is the single
   place K is applied; every consumer inherits it.  Smooth stays the
   default everywhere — no existing result moves (DD-082/DD-087 gates
   bit-unchanged).
2. In the monitor, K enters per DFT bin, so the reported fraction is
   frequency-SHAPED, not scaled by a constant (the plate gate spans
   K = 1.33..2.11 across 2-10 GHz).
3. `K(f)` is real-valued: it raises the loss but leaves the reactive
   part of the surface impedance alone, breaking the Hilbert relation
   between them — non-causal AS A TIME-DOMAIN IMPEDANCE BOUNDARY
   CONDITION (Bracken 2012, the paper's whole subject).  Harmless here
   because the chain evaluates power per frequency (an eigenmode's f0,
   a monitor's bins) and never forms a TD impedance.  RECORDED FOR THE
   SIBC FOLLOW-UP (material-plan Cluster D): a self-consistent surface
   impedance in the update needs the complex causal form, not this
   factor.
4. ACCURACY CAVEAT (DD-087 interaction): conformal wall losses carry a
   -7 % position bias at 20 cells/radius.  Roughness MULTIPLIES R_s, so
   in a smooth-vs-rough RATIO — the usual engineering question — the
   bias cancels exactly; in an absolute rough Q or alpha it does not.
   The bias is DD-087's, not DD-088's; axis-aligned walls (the staircase
   path) are unaffected either way.
5. Multiple Huray sphere classes are additive in the original model but
   are not offered as one object: the single-class Cannonball fit is
   the industry parameter set, and a multi-class fit needs SEM data
   that the datasheet channel cannot supply.  Sum per-class `factor`
   contributions manually if such data exists.
`tests/unit/test_roughness.py` (25): both closed forms at machine
precision; Hammerstad smooth limit (K == 1 for Rq = 0 exactly) and
saturation (K -> 2 from below, never past); Huray's two asymptotes;
monotonicity in f for both; Cannonball against the published equations
(r = Rz/16.73, A_flat = (6r)^2, N = 14 -> coverage = 56*pi/36) plus the
check that 16.73 IS `4*sqrt(3)*(1+sqrt(2))` and not a typo; parameter
validation; Material equality + store round-trip (smooth writes no key;
pre-DD-088 stores load unchanged); the `Material(roughness=)`-without-
sigma rejection.
`tests/integration/test_roughness_wall_loss.py` (4): TE101 Q divides by
K(f0) EXACTLY (1e-12, every tag, W and f untouched) and matches the
K-scaled closed form within the smooth gate's 1 %; the same cavity as a
rough-lossy-metal shell reaches the same closed form through the
MATERIAL channel; the copper plate line's rough fraction is the smooth
one times K(f) bin by bin (1e-12) and tracks alpha = K*R_s/(eta0*b)
within 0.3 %.
Deliberately NOT gated against a "published K value" for the Cannonball
worked example: the only numbers reachable for it came from an
unreliable secondary extraction (which also produced two demonstrably
wrong radii), and a gate on a mis-cited number would cement the
mis-citation as the target.  The gates hold the primary-source
EQUATIONS instead.
---

## DD-089 — mu(omega) dispersion — the H-side ADE by parameterising DD-084

**Date:** 2026-07-17 (session 109; package 5 of
`FIELDS_AND_LOSSES_PLAN.md`).
**Status:** Accepted — implemented + gated
DD-083/DD-084 gave every dispersive *permittivity* ONE mechanism: a
pole-residue `eps(omega)` realised by a trapezoidal ADE on the E-edges.
Permeability had no counterpart — magnetic materials were limited to a
static mu_r plus the sigma* channel, which cannot represent a
ferrite/absorber relaxation (mu' falling with frequency while mu''
peaks).
Mirror DD-084 exactly, and mirror it by PARAMETERISING the operator, not
by copying it.  Substituting `M_eps -> M_mu`, `M_sigma -> M_sigma_m`,
`g = eps0*A_dual/l_primal -> g_m = mu0*A_primal/l_dual` and
`(C^T h) -> -(C e)` leaves the DD-084 derivation character for
character: the curl term enters it only as an opaque right-hand side R
(the kernel produces `alpha f^n + beta R` whatever R's sign, and
`- beta*j_hist` completes the implicit solution either way).  So
`DispersionOperator` gained a `side` parameter ("E"/"H") and its hooks
became `save_field`/`update_field`; only `from_mesh` knows a real
difference (which material attribute, which owning-cell lookup, which
constant, which geometry factor, which exclusion).  A twin class would
have been a second place to get the coefficients wrong — and DD-084's
factor-2 bug lived in exactly those coefficients.
`DispersionModel` is reused VERBATIM: it is a relative-units
pole-residue form, so its `eps_inf` field carries mu_inf, its
constructors (debye/lorentz/drude/…) describe the magnetic analogues,
and its mandatory passivity check reads `mu'' >= 0`.  The field name is
the one wart of the reuse; the alternative (a parallel model class, or
renaming the field across DD-083/084/086) buys nothing but churn.
`Material.dispersion_mu` + `Material.dispersive_mu(name, model,
epsilon=, sigma=, sigma_m=)` (mu = (mu_inf,)*3, validated in
`__post_init__` like the eps side; mutually exclusive with is_pec).
Both channels may be dispersive at once — via the constructor directly,
which the both-dispersive gate exercises.  Store is schema-additive
through the single `_material_to_dict` pair, now sharing one
`_dispersion_to_dict`/`_from_dict` helper with the eps side.
Solver: `W_mu` folds into `alpha_H`/`beta_H` on BOTH sides alongside
sigma_m (one shared denominator), `save_field(h)` at the loop top,
`update_field(h)` right after the H-curl kernel.  The loop-top placement
is not cosmetic: CPML `update_H`, PMC `apply_H` and TF/SF `inject_H` all
run AFTER the ADE hook, so h^n must be the previous step's FINAL field —
the same reasoning the E side records.  WP-R5 donor faces (`M_mu == 0`,
exact `beta_H = 0`) are excluded from the subsets via the new `frozen`
argument AND carry `W_mu = 0`, so they stay frozen exactly as under
sigma_m.
1. CFL is the mu_inf chain with no code change: `Material.mu` IS mu_inf,
   so `compute_min_effective_mu` reads it automatically (gated).  The
   poles are A-stable at any dt — that is the trapezoidal rule's gift,
   inherited unchanged.
2. No-dispersion path bit-identical: the `if/else` mirrors the E side's
   "adding nothing, not 0.0" structure, so `alpha_H`/`beta_H` are
   array-EQUAL to the DD-081 expressions when no material is
   mu-dispersive (gated).
3. The two channels are independent: an eps-dispersive material builds
   no H-side operator and vice versa (gated both ways).
4. J_m states ride the checkpoint schema-additively under
   `"dispersion_mu"`; disk resume bit-exact.
5. NON-GOAL (unchanged from DD-083): conformal averaging of dispersive
   boundaries — the H subsets are staircase-by-classification, matching
   `build_M_sigma_m`'s policy.  Tensor mu (gyrotropic ferrites) stays
   out of scope: it is not a scalar pole-residue form and would need a
   different mechanism.
`tests/unit/test_solver_dispersion_mu.py` (11).  THE mandatory one
(plan-mandated after DD-084's lesson):
`test_drude_dc_pole_equals_sigma_m` — a mu-side DC pole (a = 0,
r = sigma_m/mu0) IS the semi-implicit magnetic conductor, matched over
200 steps to < 1e-10.  The reduction is DYNAMIC (with h^0 = 0 the
trapezoidal state J^n = W h^n holds by induction, so the pole current
accumulates into sigma_m*h), which is exactly why a wrong coefficient
cannot hide here the way it nearly did in the line physics.  Seeding E,
not H — the mu-ADE integrates dH/dt, so a nonzero h^0 would be a
deliberate initial-magnetisation offset.  Alongside it
`test_drude_dc_W_equals_M_sigma_m` pins the coefficient itself
(W_mu == build_M_sigma_m, face for face) so a failure localises in one
line.  Plus: subset construction (hand-checked single cell), g ==
M_mu/mu_inf, frozen-face exclusion, channel independence, the
array-equality no-dispersion path, mu_inf CFL, and the WP-S6 same-ops
rewind with complex (Lorentz) and real (Debye) mu-states plus an
eps-block in the same run.
`tests/integration/test_dispersive_mu.py` (5), all against the exact
`gamma = j*omega*sqrt(mu0*mu_c*eps0*eps_c)`, `mu_c = mu(omega) -
j*sigma_m/(omega*mu0)` — the DD-081 sigma_m form with mu_r promoted to
the model.  Measured, on par with the DD-084 eps mirror (1.8 %/0.46 %):
mu-Debye line alpha 2.27 % / beta 0.54 %; with sigma_m = 3500 alongside
1.92 % / 0.57 %; eps-Debye AND mu-Debye in one fill 0.54 % / 0.14 %.
Every counterfeit is REJECTED at 19-21 % beta (static mu_inf; sigma_m
without poles; either channel alone) — two orders above the gate, which
is what makes the fits meaningful.  Disk resume bit-exact across a
checkpoint seam with both ADEs and sigma_m crossing it.
MEASURED FIXTURE CONSTRAINT (recorded in the test module): the
two-length gamma extraction unwraps the ratio's phase over frequency and
therefore recovers beta only while `beta*(L2-L1) < pi` at the FIRST
frequency — `np.unwrap` fixes relative jumps, never the absolute branch
of f[0].  An eps-Debye AND mu-Debye fill at eps_inf = mu_inf = 2 reaches
n ~ 2.9 at 2 GHz (3.65 rad) and the gate then reads beta off by exactly
2*pi/dL: measured 1.728, which the arithmetic reproduces to three
digits.  The fixtures keep n <= ~1.7 at the band start (the zone the
DD-084 fixtures already live in) and `_assert_phase_branch` guards it.

---

## DD-090 — Public backend selection — GPU via backend="auto"

**Date:** 2026-07-18 (session 111, Workstream 3 of
`PERFORMANCE_PROFILING_PLAN.md`; developer decision: "auto" as default).
**Status:** Accepted — implemented + gated
(`tests/integration/test_gpu_backend.py`).
**Problem.**  The DD-032 three-tier kernel dispatch (Numba CPU / CUDA
fused / generic stencil) was architecturally wired but practically
unreachable: no public parameter selected the GPU, the only activation
path was the internal module-global `set_backend("cupy")` — called
nowhere in `src/`, `examples/` or `tests/`.  The GPU path had never run
end-to-end.
**Decision.**  `FITTimeDomainSolver` and `AnalysisScatteringTD` take
`backend="auto" | "numpy" | "cupy"` (default `"auto"`, the behaviour of
the large commercial suites): `resolve_backend()` probes once per
process for a usable CuPy + CUDA device (import, device count, forced
context creation) and falls back to NumPy with a one-time warning;
`"cupy"` raises with a clear message when the GPU is unavailable.  The
`MAGNELIO_BACKEND` environment variable overrides the `"auto"` probe
("numpy"/"cupy") — the deterministic anchor for test suites and batch
farms; `tests/conftest.py` pins the suite to NumPy so every
bit-exactness gate keeps meaning CPU rounding, and the dedicated GPU
tests request `backend="cupy"` explicitly.  The backend is per-solver
state (`xp` is resolved in `setup()` and forwarded to CPML
`initialize`/`set_pec_mask`); the module-global `get_xp()` remains only
for consumers not wired to a solver.  It is NOT persisted in the
project recipe — a resumed run re-resolves `"auto"` on the machine it
runs on.
**Port operators on the GPU.**  The modal V/I recursion (Mur/DTBC
histories, source ring buffers) stays host-side scalar work: the
port-edge subsets are gathered to the host once per call
(`_gather_host`, one small D2H per port per step) and the port-plane
modal reconstruction is built host-side and written back in one H2D
scatter per axis — bit-identical operation order on the CPU backend,
no GPU port of the recursion.  Discrete ports work unchanged
(element-wise `float()` reads are implicit syncs).
**Measured (RTX 4070 SUPER, session-111 CPU baselines).**  End-to-end
GPU S-parameters agree with the CPU **exactly** (max|ΔS| = 0 on the
coax gate; FIT updates are element-wise with fixed per-element op
order, the V/I recursion is host-side either way); the permanent gates
use 1e-12.  Scaling: the per-step GPU floor is launch/sync-bound
(~1.1–1.4 ms/step nearly independent of grid size), so the GPU loses
below ~300k cells (95k: 1.07 vs 0.65 ms/step CPU), breaks even around
~350k (379k baseline: 1.45 vs 1.47) and wins on dispersive loads
(379k: 1.63 vs 1.92, −15 %) — the margin grows with cell count toward
the multi-million-cell production regime.  `cupy-cuda12x` from pip
additionally needs `nvidia-cublas-cu12` (cuBLAS for the energy-check
matmul) — pinned in `environment.yml`.
**Rejected.**  A cell-count threshold inside `"auto"` (hardware-
dependent magic constant); porting the modal recursion to the GPU
(per-step D2H of a few hundred port values is not the bottleneck —
the launch/sync floor is, and that calls for kernel fusion or CUDA
graphs, recorded as a future option, not for moving scalar recursions).
---

## DD-091 — Broadband TD-SIBC conductor walls (opt-in wall_model="sibc")

**Date:** 2026-07-18/19 (sessions 112–116; the revived material-plan
Cluster D as its own initiative, `SIBC_PLAN.md`).
**Status:** Accepted — implemented + gated
(`tests/unit/test_surface_impedance.py`,
`test_sibc_surfaces.py`, `test_sibc_operator.py`,
`test_sibc_analysis.py`; `tests/integration/test_sibc_end_to_end.py`,
`test_sibc_validation.py`; derivation dossier
`investigations/sibc/DERIVATION.md`).
**Problem ([[DD-082]]/[[DD-087]]/[[DD-088]]).**  The perturbative loss
chain evaluates conductor loss on a lossless PEC field solution —
exact to first order only: no loaded-Q feedback, no lossy
S-parameters, the H-sample position bias stays outside the solution,
and the real roughness factor K(f) is non-causal as a TD boundary
condition.  The Leontovich condition `E_tan = Z_s(omega) (n x H)` puts
the loss into the update itself — the route the cartesian TD
competition takes.
**Decision.**  PEC edges STAY frozen; the SIBC is an additive damping
term in the Faraday update of wall-adjacent faces — the masked wall
edge's physical voltage restored from the face's own H state
(`T_f = Z_s G_f h_f`, `G_f = weight * inv_l_dual^2` — the
[[DD-082]]/[[DD-087]] wall booking reused VERBATIM, so no coefficient
ever lands on a cut or Faraday-dead face).  `Z_s(s)` is a
Foster/Stieltjes ladder `c0 + sum c_p s/(s + b_p)` fitted by NNLS on
log-spaced poles (guard ~10, normalised coordinates — raw physical
scales fail on conditioning alone), branch count from an acceptance
loop; every coefficient non-negative, so the ladder is *elementarily
passive* and an exact per-branch discrete dissipation identity gives
non-increasing energy at the UNCHANGED lossless CFL, independent of
fit accuracy.  Causal roughness: subtracted-KK completion of the
roughness EXCESS `(K-1) R_s` (<= 2e-4 vs the smooth closed form)
feeds the same real-part-targeted fit — the [[DD-088]] recorded
prerequisite.  The instantaneous `G_f R_inst` folds as a PLAIN
addition to the `M_sigma_m` diagonal; branch histories advance
trapezoidally on the midpoint `h_mid`, added `beta_H`-weighted after
the H kernel — `SIBCOperator` mirrors `DispersionOperator` (two-phase
hooks, block per tag, `bind`, Checkpointable), with a second
two-phase pass over blocks so bimetal-seam faces booked by two tags
stay jointly exact.  Opt-in at layer a:
`wall_model="perturbative"(default)|"sibc"` +
`wall_sigma/wall_mu/wall_roughness` on `AnalysisScatteringTD` (both
[[DD-082]] channels; port faces carry no wall — port planes stay
lossless; `AnalysisEigenmode` keeps the perturbative route);
recipe/checkpoint/resume through the [[DD-070]] machinery bit-exactly;
`WallLossMonitor` on an SIBC run reports the operator's OWN
accounting (`Re Z_fit` on the spec's faces and weights — no double
counting).
**Mandatory gate (twofold sigma -> inf => PEC).**  Gate A: a zero
fitted impedance takes the structural no-op path and is BIT-identical
to the master PEC run (coefficients and marched fields).  Gate B:
end-to-end at layer a, the deviation from the PEC run vanishes as
sigma^(-1/2) — measured slopes −0.4975 (max|S21 − S21_pec|) and
−0.4991 (mean −ln|S21|) over 5.8e3…5.8e7 S/m, copper endpoint 5.9e-4
against a 7e-8-clean lossless baseline; operator-level damping-rate
slope −0.5000 over four decades.
**Measured (WP-D6 record, a-priori targets from DERIVATION.md §7).**
Parallel-plate alpha ratio 0.981…0.997; WR-90 TE10 vs closed-form
alpha_c 0.944…0.976 (O(h) H-position class, largest toward cut-off);
conformal vacuum coax 0.723…0.815 at ~6 cells per inner radius →
0.768…0.860 at ~12.5 (the [[DD-087]] position mechanism amplified by
the line field's 1/r — curved conductors need resolution; the
DD-053-sized inner conductor at ~2 cells/radius under-reads by ~2x);
pillbox TM010 ring-down Q_sibc/Q_pert = 1.023 (combined-budget
agreement, SIBC closer to the closed form: −8.7 % vs −10.8 % at 10
cells/radius); rough/smooth attenuation ratio tracks K(f) per bin at
0.950…0.969 over K = 2.45→3.58 (the residual scales with |Z_s| — the
l_dual/2 offset does NOT fully cancel in the ratio on BC walls).
Recorded limitation: DTBC ports are exact for the LOSSLESS chain;
lossy walls raise the port floor to the O(Z_s/eta0) ≈ alpha/(2 beta)
mismatch class (plate −38…−50 dB, coax −26 dB at 3 GHz) — far below
physical reflections, but no longer the −130 dB class.
**Rejected.**  Free-sign vector fitting as the production route
(passivity would rest on a global numerical PR test; Foster-NNLS is
passive by construction at equal accuracy ~1e-3); fixed-pole
real-part LS (structurally dead: Lorentzian 1/omega^2 basis vs sqrt
growth); a fused Numba kernel for the branch recursion
(surface-scaling state counts sit far below the measured 65536-state
fused-ADE threshold); an eigenmode SIBC (nonlinear eigenproblem —
the eigen path keeps perturbative Q); a default flip to "sibc" (its
own decision now that the validation record exists).
---

## DD-092 — GPU step orchestration: device-staged recording, fused port transfers, CUDA graphs

**Date:** 2026-07-19 (sessions 117–118, `GPU_ORCHESTRATION_PLAN.md`
WP-G1…WP-G4; developer goal: fast parameter sweeps on mid-size grids,
~50k–350k cells).
**Status:** Accepted — implemented + gated
(`tests/unit/test_modal_recorder.py`,
`test_modal_port_fused_transfers.py`;
`tests/integration/test_gpu_backend.py` — nine GPU gates, all
verified live on the reference machine).
**Problem — and a refuted attribution.**  [[DD-090]] recorded a
grid-independent GPU per-step floor of ~1.1–1.4 ms/step and attributed
it to "launch/sync"; the GPU lost to the CPU below ~350k cells — the
sweep regime.  The session-117 prototype decomposed the floor at 95k
cells **additively and exactly** (0.123 kernel dispatch + 0.370 modal
ports + 0.321 recorder + 0.107 python rest = 0.921 ms/step full loop):
the launch-overhead hypothesis is **refuted** — kernel dispatch is 13 %
of the floor; the bulk is ~12–16 small blocking D2H/H2D round trips
per step (each a full pipeline drain, ~25 µs): four recorder
`_gather_host` per port per step, two port-plane gathers plus the modal
write-back scatters in `update_e`, each per-call re-uploading its NumPy
index array.
**Decision — three structural changes, all bit-identical by
construction** (they change *when and in how many pieces* samples cross
the bus, never the numbers; every dot/recursion stays host-side in the
master op order — max|Δ| = 0 gates throughout):
1. **WP-G1, recorder device staging.**  On the CuPy backend
   `PortSignalRecorder` gathers the raw port-plane samples into a
   per-port device ring buffer (`_DevicePortStage`, 8 MiB/port budget,
   16…4096 steps) — two fancy-index kernels per port per step, no
   sync — and drains one host block at `tail()` (the sink's flush
   hook), `finalize()`, or buffer-full; the port's unchanged host dots
   (`project_V_samples`/`project_I_samples`) then materialise V/I.
   NumPy path byte-for-byte unchanged; drain-on-`tail()` covers
   checkpoints (recorder stays outside the solver `state_dict`,
   [[DD-070]] WP-S8).
2. **WP-G2, fused port-plane transfers.**  `PortOperatorModal` builds
   concatenated index arrays at construction and caches device
   copies lazily: `project_V`/`project_V_interior`/`project_I` take
   ONE fused gather each, the `update_e` write-back ONE H2D scatter of
   the concatenated block; host math consumes the split halves
   unchanged.  Also removes the former per-call index re-upload.
3. **WP-G3, CUDA-graph capture (`solver/gpu_graphs.py`).**  The two
   contiguous device-only step segments — E phase (ADE/SIBC stashes →
   fused E kernel → ADE completion → BC E passes → optional PEC
   re-enforcement) and H phase (fused H kernel → μ-ADE/SIBC
   completion → CPML H → PMC) — are closures shared by the CPU path,
   GPU eager path and capture; after a double warm-up each phase is
   captured once and replayed as one graph launch per step.  Capture
   runs under a **private CuPy memory pool** so temp blocks recorded
   into the graph stay reserved for its lifetime (the
   silent-corruption hazard of naive capture: the main pool would
   hand those blocks to eager work between replays).  Capture failure
   warns once and stays eager; `MAGNELIO_GPU_GRAPHS=0` disables
   (deterministic anchor).  **Default ON** on the gate evidence:
   capture engages and the graph march is bit-identical
   (max|Δe,h| = 0) on a CPML fixture, a dispersive ADE case and an
   SIBC case.
**Supporting fix.**  Resume on the CuPy backend was previously
impossible: `load_state_dict` slice-assigned host checkpoint arrays
into device arrays (CuPy rejects that).  New
`magnelio._backend.array_api.copy_into(dst, src)` stages through
`cupy.asarray`; applied to solver e/h, CPML ψ, dispersion pole
currents, SIBC branch states.  GPU resume is now gated bit-exact
across the recorder drain seam.
**Measured (RTX 4070 SUPER, same-day A/B chain at the 95k-cell
"large" coax, 4000 steps).**  1.205 → 0.938 (WP-G1, −0.27) → 0.739
(WP-G2, −0.20) → **0.607 ms/step** (WP-G3, −0.13; −50 % total); V/I
and S bit-exact vs master at every stage.  Dispersive large 0.651 —
the GPU ADE surcharge shrinks to +0.04 because the ADE ops are inside
the captured graphs.  Break-even sweep (same-day pairs, ms/step
CPU / GPU): 10k cells 0.258/0.410 and 0.352/0.422 (dispersive);
38k 0.428/0.470 and 0.739/0.495; 95k 0.774/0.613 and 1.071/0.650;
379k 2.892/2.476 and 3.369/2.667.  **New break-even ≈ 50k cells
baseline, ≈ 25k dispersive** (was ~350k, [[DD-090]]); the small-grid
GPU floor is ~0.41 ms/step at 10k and no longer grid-independent.
Cross-day caveat: this day's CPU runs ~15 % above the session-111
record (0.65 → 0.77 ms/step at 95k, verified against a pre-WP-G1
worktree — no CPU regression from the refactor; warm Numba cache
required, a fresh worktree's first run carries JIT compile time);
the pairwise same-day comparisons are the evidence.
**Non-goals / rejected.**  Mega-kernel fusion (one fused step kernel)
— declined by the developer, the operator separation stays; GPU port
recursion ([[DD-090]]'s rejection stands — only the transfer pattern
changed); cross-step port batching (structurally impossible: the
corrected `e` feeds the next FIT update); the plan's projected
~0.25–0.3 ms/step floor was not fully reached (0.41 at 10k — the
remainder is the per-port feedback round trips plus the Python loop
rest; a future WP would need port-hook restructuring, recorded here,
not scheduled).
## DD-093 — Conformal averaging of dispersive / σ* boundaries

**Date:** 2026-07-28 (session 126, `CONFORMAL_DISPERSIVE_PLAN.md`
WP-C1…C5; planned session 118).
**Status:** Accepted — implemented + gated (22 new tests across
`test_material_fractions.py`, `test_conformal_dispersion.py`,
`test_conformal_sigma_m.py`, `test_pair_consistent_subcell.py`;
validation benchmark
`validation/rotated_dispersive_slab_convergence.py`).
**Problem.**  The four constitutive channels were booked
inconsistently at material boundaries: static ε (incl. `eps_inf`) and
σ conformal (`eps_avg`/`sigma_avg`), but the χ(ω) ADE subsets (both
sides) staircase-by-classification and σ* staircase by recorded
non-goal — a boundary edge saw a mixed instantaneous part with an
all-or-nothing dispersive part.
**Decision (developer, session 118).**  Arithmetic area-weighted
susceptibility mixing on all three channels: an entity joins the ADE
block of every dispersive material with its post-priority area share
as the weight on the entity's OWN mass-matrix geometry factor, so
``ε_eff(ω) = Σᵢ fᵢ·εᵢ(ω)`` (and the μ mirror) holds exactly — the
identical mixing rule as the static conformal averages; passivity is
structural.  Two-phase overlap support from the start.
**Implementation.**  WP-C1: per-material area fractions out of the
section pipeline (`face_shape_area_kernel` budget-cascade mirror;
`fraction_mids`/`material_fractions` on Edge/FaceMaterialData,
codec-native schema-additive; **NaN = not-processed** → staircase
lookup — a computed 0 is a genuine zero share; the DD-053 pair pass
promotes uniform-ladder faces to cat 2 WITHOUT an OCC statement,
which forced that convention).  WP-C2: E-side conformal membership
(cat-2 `L_primal/L_free` via `geom_conf`) + the two-phase
`update_field` (subtract every `β·j_hist`, then advance every pole
set — shared states become the joint implicit solve; fused kernel
split `_fused_subtract`/`_fused_advance`, disjoint bit-identical).
WP-C4: σ* as `prop="sigma_m"` through the face pipeline
(`sigma_m_avg`, `build_M_sigma_m` cat-1/2 mirroring `build_M_mu`'s
form incl. the 1 % floor).  WP-C3: H-side mirror, membership
restricted to cat-1 + SAFE cat-2 (floored/promoted faces keep
staircase on the BULK geometry factor).
**Key gates.**  Drude-DC ≡ conformal σ and μ-Drude-DC ≡ conformal σ*:
W equals the conformal conductivity diagonal edge/face-for-entity
(rtol 1e-12 — both channels consume the same fractions, a wrong
weight cannot hide) and 400-step marches match < 1e-10; shared-edge
joint recursion ≡ independent scalar reference (1e-13); GPU graph
capture bit-identical on a shared-edge mesh (verified live);
fraction-free meshes (from_grid, no dispersive materials)
bit-identical.
**Validation (WP-C5, rotated dispersive slab).**  DD-051 rotation
trick, cavity form (developer decision: waveguide ports are
axis-bound, so the plan's |S21| wording became the TE_z layered-cavity
resonance under PMC lids, exact for all three channels): complex
fundamental (f, γ) vs the exact transcendental root using the SAME
pole-residue models.  Result (rot 30°, h = 2…0.75 mm): the damping
error γ — the direct image of the dispersive loss — is conformal
< staircase throughout: ε −3.6e-3 vs −2.2e-2 (6×), μ +2.5e-2 vs
+5.3e-2 (2–2.6×), σ* ≈ ±1e-2 (extraction-floor limited) vs
+3.6e-2…+9.5e-2 (4–10×) at the finest/coarsest grids; |err_f|
converges ~O(h^1.1) in both configs (dominated by the shared static
conformal booking + the rotated PEC frame).  Regression: the planar
DD-084 Drude-slab gates stay green (suite).
**Found in flight.**  The WP-C5 μ-slab reference exposed a latent
DD-053 bug: `couple_face_material_pairs`' ladder target omitted the
μ̄ factor of its own documented LC identity, HALVING M_μ on every
μ_r = 2 uniform-ladder face (exactly invisible on the historical
μ_r = 1 fixtures).  Fixed (`tgt·mu_face`, bit-identical for μ_r = 1)
+ the sharp μ_r = 2 bulk no-op regression gate.
**Non-goals (unchanged).**  Harmonic/normal-direction mixing (not
pole-residue-realisable; the static scheme is arithmetic too);
per-edge vector refits; dispersive media extending into CPML (keep an
eps_inf buffer); tensor μ; `from_grid` meshes (no OCC sections — stay
staircase).

## DD-094 — Selectable time-loop precision (single default, double opt-in)

**Date:** 2026-07-21 (session 119, precision plan
`plans/quizzical-finding-kurzweil.md` WP0…WP4; developer goal: performance
parity with commercial FIT/FDTD tools, which default to single precision).
**Status:** Accepted — implemented + gated on CPU
(`tests/unit/test_precision.py`,
`tests/integration/test_precision_sparams.py`) AND live on the reference card
(RTX 4070 SUPER: `tests/integration/test_gpu_backend.py::TestGPUSinglePrecision`
+ 2 gates, all nine prior GPU gates re-verified; WP0/WP4 benchmarks below).
WP0 (kernel throughput micro-benchmark, ``benchmarks/precision_kernel_ab.py``)
caught+reversed a double-curl slowdown before it shipped.  WP1b (M-diag +
CPML dtypes) landed — bare-solver memory −50 %.  WP1c (ADE + SIBC aux-state
dtypes) landed too — the whole time-loop state is now single in single mode;
nothing remains deferred.
**Problem.**  Every grid state and update coefficient lived in `float64`
(`FieldState` `dtype=float`; the α/β/M coefficients built from float64 mesh
arrays; the CUDA kernel hard-wired to `double*`).  That is more precision
than the default path needs: the discretisation error (staircase/conformal,
finite h) dominates at **0.1–1 %** on the S-parameters, three-to-four orders
of magnitude above the single-precision field floor (~1e-7 ≈ −140 dB).  The
cost is real: on the target consumer card (RTX 4070 SUPER, Ada, FP64:FP32 =
1:64) the bandwidth-bound leapfrog pays ~2× in memory traffic plus a
crippled-FP64-ALU penalty; on CPU, double halves the AVX2 SIMD lane count.
And the update coefficients (α_E, β_E, α_H, β_H ≈ 12·N doubles) outweigh the
field arrays (e+h ≈ 6·N), so single halves *more* than "just the fields".
**Decision.**  A per-solver `precision="single"|"double"` knob (default
`None` → `MAGNELIO_PRECISION` else **single**), on `AnalysisScatteringTD` and
`FITTimeDomainSolver`, resolved by `resolve_precision` to a (real, complex)
dtype pair.  **Single is the production default**; double is the opt-in for
high-Q (Q ≳ 1e4–1e5, where float32 coefficient resolution limits Q) or
high-dynamic-range studies.  Precision (`dtype`) and backend (`xp`) are
**orthogonal axes** — any precision runs on any backend; `FieldState.zeros`
gained a `dtype=` beside its existing `xp=`.  An explicit `precision` value
wins over `MAGNELIO_PRECISION` (mirrors `backend="cupy"` bypassing
`MAGNELIO_BACKEND`) — no silent footgun for a run that explicitly asks for
double.
**Numerical policy — and a WP0-refuted accumulation choice.**  The per-step
kernel multipliers α/β cast down to the field dtype (the fused CUDA kernel
takes one `scalar_t` for fields *and* coefficients), but the `/denom`
coefficient arithmetic stays float64 (CFL/timestep resolution is never
single).  `_M_eps_diag` / `_M_mu_diag` are deliberately NOT cast, so the
energy reduction `(M·e)@e` promotes to double.  The curl accumulator was
initially planned `double` in both variants ("single storage, double
accumulation").  **WP0 refuted this on the GPU:** a `double curl` in the
float32 kernel makes single *slower than double* (0.63–0.68× at 97k–373k
cubic cells) — the few FP64 register ops dominate a bandwidth-bound kernel on
the 1:64 card.  A pure `scalar_t curl` restores the win (1.25× at 97k → 2.43×
at 373k).  So the float32 CUDA kernel and the float64 kernel both use a
`scalar_t curl` (float64 path byte-identical to before); the 4-term curl is a
sum of similar-magnitude neighbour differences, float32 accumulation stays at
the ~1e-7 field floor.  The CPU Numba kernels keep the float64 `curl = 0.0`
literal — double accumulation is *free* on a full-rate-FP64 CPU, so the
CPU/GPU single results differ by ~2e-6 (below the floor, and single≢double
bit-identity was never a goal).
**Double-precision islands stay double regardless of the knob**, because
they are not per-cell-per-step and cost negligibly: the DFT/Freq/wall-loss
accumulators (`complex128`, running sums — the naive-single-sum catastrophe
over 1e5–1e6 steps), the modal-port solve / eigenmode solver, and the whole
geometry/meshing pipeline (Boolean robustness).  The DFT already up-casts a
float32 sample into its complex128 bins (`_bins += phase*data`), so the
running sum was never at risk — WP3 makes that guarantee explicit.  This
mirrors how commercial tools accumulate the DFT in double under single
fields.
**WP2 — the one structural break.**  `operators/numba_kernels.py`
`_CUDA_SOURCE` is templated over a `typedef SCALAR_T scalar_t` prelude
(pointer args AND curl `scalar_t`, see the accumulation note above);
`_compile_cuda(dtype)` caches one RawModule per scalar dtype and the fused
entry points dispatch by `Ex.dtype`.  Because single is now the default,
single+GPU *requires* the float32 kernel to exist — a double-only kernel
would misread float32 memory.  The float64 path is byte-identical to before
(`scalar_t == double` there — verified: GPU-double vs CPU-double max|ΔS| = 0,
all nine existing GPU gates green).
**Evidence — CPU accuracy.**  Parallel-plate S-parameter A/B (11×6×41, 81
frequencies), single vs double: `|S21|` (insertion loss) agrees to **2.6e-7**
relative; the linear S-matrix to **2.1e-6**; the physical `|S21|`
discretisation deviation (9.76e-3 dB) is *identical* in both and four orders
of magnitude larger than the single-double gap.  The *only* visible effect:
the `|S11|` reflection floor rises from **−138.75 dB** (double) to
**−113.08 dB** (single) — exactly the float32 field floor (20·log₁₀(2e-6) ≈
−114 dB), still well below the −100 dB reflection-free acceptance line.  The
ultra-deep floor is a double-only feature; the high-dynamic-range user opts
in.  GPU single is equally faithful: coax GPU-single vs GPU-double max|ΔS11| =
5.2e-7, max|ΔS21| = 2.1e-6.
**Evidence — GPU performance (RTX 4070 SUPER), and the honest caveat.**  Raw
fused E+H kernel, float32-vs-float64 ms/step: 97k cubic 0.0295→0.0236
(**1.25×**), 373k cubic 0.0722→0.0297 (**2.43×**) — the bandwidth win grows
with grid size.  BUT the *whole-solver* speedup on the thin coax fixture is
marginal: `profile_solver` large (95k) 0.478→0.495 (single 3% *slower*),
xlarge (~380k) 0.909→0.886 (2.5% faster).  The reason is [[DD-092]]: the
fused kernel is only ~13–16 % of the coax step (ports/recorder/DFT dominate,
and those stay double by design), so a 2× kernel moves the total ~8 % at most,
and on the thin 14×14×N coax the kernel is in its small-grid regime.  Net: the
single default pays off most for **large 3D geometries** (the production
target — big meshes, few ports), least for thin port-heavy S-parameter
fixtures.
**Evidence — memory (WP1b landed).**  Bare solver (fields + coefficients +
M-diagonals + CPML ψ — the converted set, no port/recorder/DFT machinery),
216k cubic cells with CPML: double 63.9 MB → single **31.9 MB (−50.1 %)** —
the full halving for a field-dominated problem.  The earlier WP1-only coax
figure was −21 % because the port-heavy coax is dominated by the double-only
port/recorder/DFT arrays; WP1b converted the last full-grid double set
(M_eps/M_mu, each as large as the fields) and the CPML boundary state, so
field-dominated 3D now hits −50 %.
**Resume.**  The recipe (`_recipe.py`) persists the *resolved* concrete
precision ("single"/"double"), not the None sentinel, so a resumed run
reproduces the dtype the run used regardless of `MAGNELIO_PRECISION` at resume
time; a recipe predating this DD (missing key) means the old double-only
default.  A precision switch across a resume seam is out of scope.
**Suite pinning.**  `tests/conftest.py` pins `MAGNELIO_PRECISION=double` (the
existing accuracy/bit-exactness gates were written against double); single
tests pass `precision="single"` explicitly, which wins over the env pin.
**Non-goals / deferred.**  Half precision (fp16/bf16 — no headroom under the
field floor); driving DFT/port/eigenmode/geometry precision from the knob;
per-region or per-material mixed precision; a precision switch mid-run.
**WP1b — done (M-diagonals + CPML).**  `_M_eps_diag` / `_M_mu_diag` (a
full-grid array each, used ONLY by the energy monitor — the coefficients were
built from the float64 `M_eps`/`M_mu` locals) now carry the field dtype; the
energy reduction is `xp.sum(M·e·e, dtype=float64)`, forcing double
accumulation over the grid so the energy-decay stop criterion is unaffected.
CPML `initialize(dtype=)` allocates the ψ recursion state and the b/c/ck
coefficients at the field dtype, so the `β·ψ` correction is a same-dtype op
(no float64 penalty on a float32 GPU run); a `None` dtype keeps float64 for
standalone use.  Gated: GPU CPML march (`test_cpml_march_matches_cpu`) green,
1289 unit + resume/energy integration green (the dot→sum energy change is
float64-accumulated, no regression).
**WP1c — done (ADE + SIBC aux-states).**  The ADE pole current
(`solver/dispersion.py`) and the SIBC Foster-branch state (`solver/sibc.py`)
now follow the field dtype: `bind()` reads `beta.dtype` (already cast to the
real field dtype by the solver) and casts the geometry factor `g`, the field
stash `f_prev`/`h_prev` and the state array to it — real poles / branches →
float32, conjugate-pair poles → complex64.  The per-pole/branch scalar
coefficients (`k`/`c`/`q`/`r_inst`, a handful of numbers) stay double, so each
step is a single-store / double-op update; the fused ADE Numba kernel gains a
new dtype specialisation (a separate type signature, not the DD-090 parallel-
flag cache trap).  The real/complex-pole discriminator moved off
`p.J.dtype == complex128` (which the complex64 conversion would break) onto
`np.iscomplexobj(p.J)`.
*Why single is safe here (unlike the DFT).*  The DFT bin is a pure running
sum (`|phase| = 1`) → √N error growth → must stay complex128.  The ADE/SIBC
states are decaying IIR filters (`|k| < 1` for every passive pole / Foster
branch), so old contributions fade and the error stays at the ~1e-7 field
floor — the same structure that already put the CPML ψ state at the field
dtype in WP1b.  The one worst case is the Drude **DC pole** (`a = 0`, `k = 1`,
a pure integrator), but it telescopes: `J = r·g·(fₙ − f₀)` is field-bounded,
not a growing sum.
*Evidence (RTX 4070 SUPER, single vs double, rel. to max double field):*
Lorentz-filled cube (conjugate-pair pole, complex64) 400 steps CPU **1.2e-6**,
GPU 300 steps **9.7e-7**; Drude-DC cube (float32, the k=1 case) 4000 steps CPU
**2.8e-6** — stable, no √N blow-up; SIBC copper cavity 1500 steps CPU
**6.0e-6**, GPU 1200 steps **7.3e-6**; all finite, all at the float32 floor.
Debye-line two-port S21 single≡double < 1e-4 (`test_precision_sparams.py`).
Gated: `test_precision.py::TestDispersionAuxPrecision` (+3),
`test_sibc_operator.py::TestSIBCPrecision` (+3),
`test_precision_sparams.py::test_single_matches_double_on_dispersive_line`.
Resume stays consistent — the recipe pins precision, so a single run
checkpoints/loads complex64 on both sides of the seam.  Nothing deferred.

## DD-095 — Modal port power calibrated to the discrete Poynting reference (conformality patch)

**Date:** 2026-07-27 (sessions 123–124, `PORT_POWER_PLAN.md`
WP-P0…P4; trigger: developer-measured |S12| − |S21| ≈ 0.59 dB,
frequency-flat, between a conformal PTFE-coax TEM port and a rect-WG
TE10 port on the coax2rect fixture).
**Status:** Accepted — implemented + gated
(`tests/integration/test_port_power_reciprocity.py`; derivation dossier
`investigations/port_power/DERIVATION.md` with versioned evidence
scripts).  Full-size coax2rect re-run by the developer post-fix:
asymmetry < 0.02 dB.
**Problem.**  The modal V/I calibration ([[DD-078]]/[[DD-085]],
`_calibrate_v_i`) measured the constructed basis wave's power with a
flux surrogate `s_h` that combines a single global wave impedance with
*geometric* Voronoi patch areas.  At conformal port cross-sections the
true discrete travelling wave follows the reduced ([[DD-053]]
`eps_avg`/`f_A`) local admittance, so the surrogate over-counted the
flux at exactly the cut cells whose M_ε profile norm used reduced
areas: the round PTFE coax over-recorded power by s² = 1.0721
(over-injected by the same factor via `source_scale = √Z/record_scale`).
Same-type port pairs cancel s exactly and reflections are
scale-invariant, so the entire prior validation record (through-lines,
port floors, unitarity of same-type fixtures) was structurally blind;
only mixed-type pairs exposed it as S21 = s·T, S12 = T/s.
**Reference decision.**  Modal |a|², |b|² are defined against the
**discrete Poynting sum** through the port plane — the FIT identity
P = Σ e·h with half-cell weights at bbox-boundary nodes, exactly as
`FluxTimeMonitor` implements it.  Port power ≡ monitor power by
definition; the closed-form defect prediction
s² = ε₀·ε_r·d̃_n·Σ ê²·A_geo/l² (M_ε-orthonormality collapse) matched
the measurement within the gate floor *before* any code change
(staircase exact 1.00000; conformal 1.06443 predicted vs 1.0721
measured; mixed-fixture prediction 0.555 dB vs 0.5423 dB measured).
**Decision.**  `conformal_flux_patch_scale` (ports/modal/operator.py)
computes a per-edge conformality factor from the port's own flattened
M_ε and the first interior slab's edge classification —
χ = 1 (categories 0/1/3), χ = M_ε·l/(ε₀·eps_pair·A_geo) with
eps_pair = eps_avg/f_A (category 2) — and `_calibrate_v_i` multiplies
it onto the DD-078 patch areas, correcting `p_one` (κ, state_scale)
and `s_h` (record/source scale) in one place.  The correction is
conformality-only: dividing the slab M_ε by its own free-part
permittivity strips every dielectric contribution, so layered
staircase planes stay bit-identical.  Mode shapes, 2D eigenproblems,
z_line, DTBC kernels, reflections: untouched.  A Δχ estimator warns
when enlarged-cell mass parked on category-0/1 edges (the known χ
blind spot) would bias the scale by > 1e-3.
**Affectedness (measured matrix).**  Modal was the *only* violator.
Band ([[DD-057]]) acquitted: +0.0000 dB on the mixed fixture — its
recording functionals are bi-orthogonal to the per-frequency true
wave, making the unit-wave Wronskian port-independent (measured
0.9909352 at both ports to 7 digits despite 2.5 % different eigenvector
scales).  CW true-mode ([[DD-056]]) acquitted and power-unitary
(column sums 1.00000, lock-in residuals < 5e-7).  Band/CW cannot host
TE/TM hollow-waveguide ports (multiconductor-only) — n/a by
construction.
**Gates (post-fix).**  Conformal round coax |b|²/flux 1.0721 →
1.00725 (+0.031 dB, staircase floor class); plate TEM / WR-90 /
layered-QTEM staircase bit-identical; round→square mixed coax
reciprocity +0.5423 → −0.0001 dB; compact coax2rect +0.5625 →
+0.0663 dB (hard gate ≤ 0.1 dB; the residual is the constructed-basis
class bounded in the dossier §5d, not the blind spot — Δχ measured 0);
modal |S21| now equals the band/CW true transmission (−0.063 dB, was
−0.334); full suite 1530/15/0.  Historical mixed-pair transmissions
shift by design (~0.3 dB per direction); |T| = √(S12·S21) values are
unchanged.
**Non-goals / rejected.**  Candidate B (local admittance from the
partner-face M_μ) — refuted by measurement (s² = 0.954, wrong
direction; the plane M_μ is flattened and the DD-053 pair coupling
lives on other faces).  A per-edge ε_r generalization of the closed
form — rejected: layered staircase planes measure clean (1.0096), an
ε-mixing correction would *introduce* bias there.  H-B
(normal_dx vs graded dz) — cleared twice: graded port slabs are
rejected by the three-equidistant-cells validator, and on the
numerical path every μ₀·ndx/M_μ factor cancels through the γ step
(hq = 1/Re Z identically).  Machine-exact reciprocity — not claimed:
the corrected surrogate still evaluates the constructed basis pair;
the residual is the measured ≤ 0.7 % class, shrinking with mesh
refinement.
---
## DD-096 — Mur complement absorber + port-signal stop criterion

**Date:** 2026-07-28 (sessions 125–126, `MUR_STABILITY_PLAN.md`
WP-M0…M2; trigger: late-time energy regrowth on the shielded-microstrip
example — decay to ~−37 dB, then exponential regrowth to peak within
~400k steps, single AND double precision; workaround commit `5bbc09d`
had bounded the example).
**Status:** Accepted — implemented + gated
(`tests/integration/test_mur_complement_absorber.py`; derivation
dossier `investigations/mur_stability/DERIVATION.md`, measurement
record `MEASUREMENTS.md`, exact-spectrum tooling `exact_eigs.py` /
`candidate_eigs.py`).
**Problem (WP-M0/M1, measured and derived).**  The modal boundary
wipe (`update_e`: project → Mur-1 → overwrite the plane with the modal
reconstruction) pins every port-unrepresented transverse family to
zero at the plane.  Cut-off-trapped families thereby become Dirichlet
resonators coupled to the Mur channels through the oblique dual
projection (χ = ⟨w_c, ψ_t⟩ ≈ η = ⟨w_t, φ_c⟩ ≈ 0.23 on the DD-056
layered fixture); whether the closed loop damps or pumps is a phase
condition with no discrete energy statement.  The exact
boundary-closed companion matrix (one production step per unit
vector; sparse shift-invert) reproduces every measured rate,
frequency and sign flip: nz sweep 12/24/48/96 → −1.2e-6 / **+8.6e-5**
/ **+1.3e-5** / +2.0e-5 per step at the trapped 15.71-GHz resonance.
One Mur port + PEC far wall suffices (mirror symmetry); roundoff
seeds it — every such run eventually diverges.  QTEM channels have
χ = 0 by x-parity; DTBC-certified channels are unaffected.
**Decision (fix).**  Complement absorber on modal ports: the
port-unrepresented remainder at the interior companion plane
(exactly dual-orthogonal by construction) is advanced to the port
plane by a per-edge scalar Mur-1 and ADDED to the modal write —
unrepresented families see an absorbing plane instead of a Dirichlet
wall.  ``r_p = (c_p·dt − dx_n)/(c_p·dt + dx_n)`` with
``c_p = c0/√eps_eff,p`` from the exact per-edge ratio
``M_eps/M_eps_vacuum`` (geometric factors cancel; **no free
parameter**).  Residual-PEC plane edges (sub-face window frames) stay
pinned via a live mask.  Scoped to ports with ≥ 1 Mur channel
(developer decision): fully DTBC-certified ports keep the exact
pre-DD-096 path bit-identically.  Complement state (4 arrays/port)
joins ``state_dict`` (schema-additive; pre-DD-096 checkpoints restart
the absorber from rest).
**Decision (termination).**  New ``port_signal_stop_db`` solver
criterion (developer decision): stop when the cross-channel |V|
envelope (windowed max between energy checks, DD-078-scaled) decays
the given dB below its run peak.  Root cause: the only |λ| = 1 modes
surviving the absorber are TM-cut-off (k_z = 0) cavity resonances
with **zero tangential E everywhere** — invisible to any port-plane
tangential scheme (wipe, Mur, DTBC, absorber alike) and already
exactly neutral pre-fix.  They hold the stored-energy plateau
(fixture: −39 dB hybrid- / −14 dB QTEM-seeded), so ``energy_stop_db``
alone can never terminate shielded lossless runs; the port tails —
the S-parameter deliverable — collapse to the machine floor and are
the robust signal.  Unbounded runs accept either criterion.
**Gates (post-fix, exact + production).**  Exact spectra (production
code, complement state in the eigen-vector; linearity residual
~2e-16): no coupled mode above the unit circle at any nz, former
growth → decay (slowest coupled −1.1e-4 at nz 48); 1–40 GHz wide scan
clean.  TD fixture: hyb 400k steps — V tails at machine floor
(−335 dB vs −69.5 dB and growing pre-fix); qtem 2M — no
roundoff-seeded growth (was +1.30e-5).  Port-signal perturbation vs
pre-fix, normalized to the excited-channel peak: QTEM −300 dB
(untouched), hybrid −27 dB = removal of the spurious trapped ringing
(≪ DD-068 hybrid Mur error, median −10.5 dB).
**Non-goals / rejected.**  F-A (Mur-state dissipation
``V *= 1 − δ``) — refuted by exact measurement: d|λ|/dδ = 0.026 (the
trapped mode holds > 99 % of its energy in the field), a curative
δ ≈ 3.3e-3 would wreck the absorption floors.  F-B
(complement-preserving wipe without absorption) — leaves the families
neutral: the plateau still defeats the energy criterion.  Higdon-2
([[DD-069]] refuted; its "Mur-1 gently dissipative near cut-off"
intuition is measured-false for the full discrete loop).  Absorbing
the TM-cut-off tower — impossible from the port plane (zero
tangential trace); handled by the termination criterion instead.
---

## DD-097 — Wall-plane H reconstruction: derivation validated, estimator measured-REFUTED

**Status:** Decided 2026-07-28 (session 128).  **No production
change** — the DD-087 perturbative estimator and the DD-091 SIBC
stay as they are.  Full record:
`investigations/wall_plane/{DERIVATION,MEASUREMENTS}.md` + probes.

**What was derived and PASSED (reusable):**

- **WP-W0** re-calibrated the instrument: the session-108 centred
  pillbox numbers moved −11.11/−7.39 → −10.80/−7.99 % by
  ENVIRONMENT drift (a `43481c3` worktree A/B reproduces today's
  numbers bit-for-bit on the old code; areas and f0 unchanged; only
  the lattice-degenerate centred alignment responds).  Current
  baseline: phase spread 1.01 points at 20 cells/radius.
- **Gate 1 — the wall plane (n̂, p) of every cut cell is exactly
  reconstructible from EXISTING mesh channels at zero mesher cost.**
  n̂ = −w/‖w‖ (divergence identity); the offset by inverting the
  covered-area function A(q) of one cut, non-jump face (monotone
  piecewise-quadratic; sensitivity-weighted combination over faces).
  Synthetic single-plane cuts: 1e-14.  Pillbox mantle,
  geometry-blind: radius to O(h²) AT the secant-sagitta constant
  h²/(8R) (rms 11.5 → 3.3 µm at 1 → 0.5 mm), phase-robust.  Jump
  faces = flat families (their plane is the face plane — including
  tangent-plane apex cells, where that IS the tangent plane).  A
  V_pec/centroid mesher channel was evaluated and is NOT needed.
- **Gate 2 — the d1 + central-difference stencil is correct on
  smooth fields:** on the analytic J1 field at the actual sample
  positions/masks, +0.74…0.81 % at 20 cells/radius (phase spread
  0.07 points), +3.0…3.3 % at 10; dead-end fallback share ≤ 0.7 %.

**What FAILED (gate 3, the funding gate) and why:**

On MEASURED eigenfields the reconstruction lands −4.6…−7.0 % (band
was ±2–3 %) and WIDENS the 0.5 mm phase spread 0.70 → 2.46 points —
the session-108 failure mode reproduced with provably exact
geometry.  Isolated mechanism (A/B/C probe): the measured local FD
gradient deviates from the true gradient by **240–420 % RMS** — the
discrete near-wall field error is O(h) pointwise with cell-scale
phase-dependent structure, so ANY local finite difference over an
O(h) span carries an O(1) relative gradient error; the correction
inherits it.  Control C (exact analytic gradient on measured
samples) still misses the band at 10 cells/radius (−4.95 %): beyond
first-order position bias the near-wall samples carry field error
no linear geometric pullback can remove.  **The method class
"post-hoc sampling + local FD extrapolation" is refuted on measured
FIT fields — the session-108 rejection was not a geometry problem.**

**Recorded, unfunded successors:** non-local (patch-fit) gradient
estimation with a noise-averaging gate; the SIBC (DD-091) as the
accuracy route for curved-wall losses.  The WP-W4 SIBC scalar
position factor was NOT reached (its prerequisite gate failed).

*Session-128 developer decisions (same day, consultation pass):*

- **Plan RETIRED** with sign-off (git history keeps it).
- **Funded successor: the geometric curvature pullback (DD-098,
  `WALL_CURVATURE_PLAN.md`)** — a mode-free multiplicative booking
  factor from the PEC-wall identity ``∂H_tan/∂n = −κ·H_tan`` (the
  tangential curl component vanishes AT the wall because E_tan
  does), exact for 1/r line fields (the conformal-coax worst case).
  It consumes gate 1 of this record and NONE of the refuted
  machinery — no measured-field differentiation anywhere.
- **Deliberately DEFERRED, recorded as a note for a possible future
  initiative: a locally exact cut-cell update** (deformed Faraday
  contours around the PEC cut — the root-cause fix for near-wall
  field quality).  Rationale: the schemes shipped by the cartesian
  competition are proprietary, so this would be a clean-room
  re-derivation with real research risk (late-time stability is the
  historical minefield; CFL preservation would be a hard
  prerequisite) for a benefit concentrated on near-wall consumers.
  Noted explicitly because **wake impedances are a central intended
  use case** — if that initiative comes, near-wall field quality
  becomes load-bearing and this deferral is the first thing to
  revisit.  The patch-fit gradient route stays unfunded (its
  measured ceiling is the C column above).

---

## DD-098 — Geometric curvature pullback for curved-wall losses

**Status:** Decided 2026-07-28 (session 128).  Default-on for
conformal scenes (developer call, escape hatch
``curvature_correction=False`` on both enumeration functions).
Full record: `investigations/wall_curvature/{DERIVATION,
MEASUREMENTS}.md` + probes; spike branch
`spike/dd-098-sibc-curvature-factor` (K2 instrument, not merged).

**The identity (WP-K1 gate 1, analytic):** at a PEC wall
``E_tan ≡ 0`` and ``H_n ≡ 0`` kill both extra curl terms, leaving
exactly ``∂H_tan/∂n = −W·H_tan`` with ``W = ∇_tan n̂`` the shape
operator (n̂ into the air; convex-from-air positive).  No 3D
rest terms.  Integrating along the normal with the offset-surface
curvature gives the **linear pullback**

    c_b = max(1 + κ̃_b·d1_b, 0),   weights scaled by c_b²

exact at EVERY distance for 1/r line fields (coax class),
first-order exact generally, second-order J1 residue
−2.9·(d1/R)².  The divisive form 1/(1+κ̃d1) sketched in the plan
is only its first-order twin and misses 1/r at d1²/R² — rejected.
Scope limit (derived + measured): the per-entry factor keeps the
diagonal normal curvature along the component direction; mixed
components on non-axis-aligned curved walls are under-corrected
(the off-diagonal Weingarten term needs the other tangential
component).

**κ̃ from grid data (gate 2):** per wall FAMILY (flat = jump
plane; curved = DD-097 gate-1 plane with
**n̂ = −w_curved/‖w_curved‖** — the combined-w normal of a corner
cell is tilted by the full-area jump face, found + fixed here),
angle-gated (45°) symmetric LS fit of the neighbour-plane Gauss
map rotation, 3×3×3 stencil.  Rule (b) decided: flat families get
the same fit — coplanar-only neighbourhoods give bit-exact 0
(grid-aligned flat scenes are exact no-ops), apex tangent columns
recover 81–96 % of the mantle curvature.  Signs correct
everywhere (coax 0/9576 wrong); sliver cells without an
invertible face book unscaled (c = 1) and never feed a fit.

**Measured (gates 3, 4, K2):**

- Pillbox TM010 Q on measured eigenfields (per-family probe):
  −5.5…−7.9 % → **−1.9…+1.8 %** on all five fixtures, 0.5 mm
  phase spread 0.70 → 0.45 pts NARROWED — inside the ±2–3 % band
  the DD-097-refuted gradient class could not reach.  The
  multiplicative form is the decisive difference: it scales the
  measured sample instead of adding an O(h)-noisy derivative.
- Conformal coax α (SIBC end-to-end): 0.723…0.815 →
  **0.801…0.903** (0.16 mm), 0.768…0.860 → **0.828…0.926**
  (0.08 mm) — exactly the analytic position-bias share (×1.108
  measured vs 1.115 predicted).  The K2 target ≥ 0.95 was NOT
  met; the residual is measurably NOT the factor's class:
  booking coverage (0.960 inner / 0.854 outer, the z-invariant
  axial-walk deficit) and near-wall field error (~0.92 at
  6 cells/inner-radius, identical in raw and factored columns)
  cap ANY per-entry booking factor at ~0.92 @ 0.16 mm.
  Developer decision: **fund production anyway** — the factor
  removes its entire derived error class on both consumers; the
  coverage fallback is recorded as a separate candidate follow-up
  (DD-097 WP-W2 class), the field error's root-cause fix remains
  the deferred cut-cell update (DD-097 note).
- WR-90/staircase/BC walls: enumeration bitwise identical with
  the factor on/off.

**Implementation:** `mesh/curvature.py` (`CurvatureFactors`,
closed-form covered-area inversion — piecewise quadratic in the
offset with cancellation-free corner gaps), applied per booking
entry in `_conformal_solid_surfaces`; consumed by BOTH
`enumerate_pec_surfaces` (DD-087 perturbative weights, physical
monitors) and `enumerate_sibc_surfaces` (DD-091 ``G_f`` — a
non-negative scalar per branch, passivity identity untouched).
No mesher/store change; old stores work immediately.
``area_total`` stays the geometric area (the factor scales
weights, not areas).

**Addendum (2026-07-28, follow-up measurement,
`investigations/wall_curvature/probe_coverage_followup.py`):** the
"booking coverage" attribution above is corrected, and the
recorded fallback-walk candidate is REFUTED.  The production
per-cell walk (combined ``−sign(w)`` direction) drops nothing —
``Σweight/(3·area_total)`` = 0.997…1.0 on all pillbox fixtures and
0.99999 on the coax, dropped power 0.0000; the drop class existed
only in the investigation probes' per-family booking.  The coax
0.960/0.854 "coverage" is area REGISTRATION: the WP-D6 fixture's
outer conductor (r = 2.5 mm) is tangent to the domain bbox
(±2.5 mm), so four ~20° zones carry a sub-cell PEC sliver that
``A_face_pec`` never registers (booked/exact 0.068 per tangent-zone
bin).  A padded fixture (explicit PEC shell to 3.25 mm) restores
the outer wall area (0.839 → 1.026) and measures end-to-end TD α
**0.850…0.953** at 0.16 mm (stock 0.801…0.903) — the ~0.92
"booking-factor ceiling" above was an artifact of the tangent
fixture.  Open (developer decision): re-fixture
``test_conformal_coax_alpha`` on the padded shell; a mesher
warning/fix for the generic thin-shell/bbox-tangency registration
class (a curved conductor tangent to the bbox, or thinner than a
cell, silently books no wall loss there).

## DD-099 — Boundary walls join the wall-loss family

**Status:** Decided 2026-07-29 (sessions 129-131).  Shipped
default-on; full record: `BOUNDARY_WALL_PLAN.md`,
`investigations/boundary_wall/{DERIVATION,MEASUREMENTS}.md`.

**Problem.**  A conductor degenerating into the domain boundary lost
its wall: the bbox-tangent coax outer conductor silently booked
0.068 of the exact wall area in four ~20-degree zones (×1.056 alpha
under-read end-to-end, DD-098 addendum).  Root cause measured, NOT
the assumed w-cancellation: a candidate-gate registration VOID —
`detect_boundary_cells` is pure 6-neighbour material_id contrast, a
sliver that captures no cell centre is never sampled and does not
exist in any channel.

**Decisions (developer, sessions 129-131):**

1. **No silent domain padding** (PEC fill costs full bandwidth,
   +70 % on the coax fixture).
2. **Candidate-gate fix instead of a tangency detector**: non-PEC
   cells of the six boundary layers are seeded into the conformal
   sampler for the GEOMETRIC channels only (separate classifier
   call, fresh section cache; ``pec_frac_geom``/``jump`` feed only
   ``A_face_pec``/``A_face_pec_flat``, so material matrices are
   bit-identical by construction — verified exact).  PEC-classified
   cells are excluded (seeding them registers the domain's PEC hull
   as phantom walls).
3. **Port planes get continuation semantics** at enumeration
   (`_masked_face_pec_views`): a face hosting a port (or a non-PEC
   BC) takes the adjacent interior plane's coverage and zero jump —
   the structure continues, no wall books, no coverage step, the
   SIBC target gate keeps its exclusions.  The occ_backend
   degenerate max convention needed NO change: with a PEC background
   the world beyond a shape end plane IS a shorting lid, correct for
   portless ends; the port is analysis knowledge.  This also removed
   a latent pre-DD-099 error: the conformal cell path used to book
   ~1 % phantom conductor-cross-section wall area on port planes.
4. **Unregistered-wall warning** at mesh consolidation
   (`detect_unregistered_walls`, threshold 0.1 inside the measured
   empty window: suite floor 0.49 = ordinary cut cells, must-fire
   signal 0.0055 = unresolved 30 µm shell): one warning per scene
   for the interior w-cancellation class (a conductor shell thinner
   than one cell books ~nothing).
5. **The PEC boundary condition carries the wall material**
   (large-suite convention): ``PECBoundary(face, wall_sigma=...,
   wall_mu=..., wall_roughness=...)`` overrides the analysis-global
   fallback per face, through the shared `resolve_wall_conductors`
   rule (SIBC + perturbative monitor + eigen wall_loss_Q).  Parity
   gate: per-face declaration reproduces the global-fallback run to
   1e-12.

**Measured:** padded and tangent coax become EXACTLY equivalent
under the seed (booked area equal to 4 decimals) and read alpha
0.838…0.929 at 0.16 mm (window 0.83…0.97; tangent pre-fix
0.801…0.903, padded pre-fix 0.850…0.953 incl. the phantom
cross-sections).  Declared side BCs on top change nothing (the ok
gate drops BC pairs on covered faces — "registration wins" is the
de-facto seam rule; an uncovered flat BC wall books through the BC
leg as before).  Gate tests: ``test_tangent_coax_alpha``,
``test_bc_wall_material_parity``, ``test_unregistered_wall_warning``.

**Recorded conventions and limitations:**

- *Rim over-booking of the SIBC BC leg*: (N_t+1)/N_t per tangential
  family (full-voltage corner rule, SIBC DERIVATION §3) — an O(1/N)
  convention contained in the WR-90 record 0.944…0.976; kept.
- *Residual void class — inscribed tangency with an AIR background*:
  the seed recovers the wall only where a (PEC) background registers
  boundary-plane coverage; an air-background inscribed cavity keeps
  its tangency wedge unbooked.  The warning fires there correctly
  (verified true positive on the `test_conformal_convergence`
  cavity); kept as a documented limitation.
- *Curved interior shell missing ALL cell centres*: stays
  channel-invisible even after the fix; the corner-classify seed is
  the recorded complete-fix candidate (not scheduled).
- The warning names worst-cell indices/coordinates/ratio (no
  covering material id — recorded deviation from the dossier).

## DD-100 — Dead-tile skipping for the fused TD kernels

**Status:** Decided 2026-07-29 (session 132).  Shipped default-on
(GPU backend; `MAGNELIO_TILE_SKIP=0` kill switch); full record:
`investigations/pec_fill/` (census, tile-shape bench,
plan-vs-census gate, runtime A/B) — plan file retired.

**Problem.**  The fused curl kernels sweep dense full-grid arrays;
a PEC-frozen edge (`alpha_E = beta_E = 0`) costs the same memory
bandwidth as a live one.  Structured tensor-product grids make
this unavoidable at the mesh level: a single feed line forces the
bbox over large dead conductor blocks (`coax2rect`: 74 % of
elements dead; PEC-padding the coax fixture had cost +70 %
runtime, session 129).

**Decisions (developer, session 132):**

1. **Runtime only.**  Memory savings via block-structured storage
   stay off the roadmap (no current constraint; ~10x the effort).
   Bbox cropping was rejected on the flagship case: L-shaped live
   regions have a full-domain bbox — crop saves exactly nothing.
2. **Static live-tile launch lists at tile (2, 4, 32)** — chosen
   by measurement, not geometry intuition: cubic tiles capture
   best on curved dead regions but fail the dense gate (+11…+69 %
   float32: 32 B k-run segments, and 1024-thread blocks cap Ada
   occupancy at 1024/1536); k = 32 shapes at 256 threads are
   dense-free AND (2,4,32) beats the old flat block's capture on
   every fixture.
3. **Provable no-op skip rules on the solver-final coefficients**
   (port-plane flattening included — mesh-level masks are wrong):
   E edge `alpha == 0 and beta == 0`; H face `alpha == 1 and
   beta == 0` (WP-R5 donated) or curl-dead (all four bounding E
   edges frozen).  PMC faces suppress curl-dead skipping in their
   outermost layer; TF/SF field sources and BC types outside the
   `_pec_reenforce_after_bc` safe list self-disable the analysis
   (dense fallback).  Runtime-writer audit in the
   `solver/tile_skip.py` module docstring (ports write E only;
   CPML corrections vanish on curl-dead faces; thin-wire edges are
   plain PEC edges at runtime).
4. **Two kernel compile variants from one source** (`-DLISTED`).
   The planned single code path (dense == identity list) was
   measured-refuted: the dependent per-block list load costs +12 %
   in the L2-resident float32 regime typical of production sizes
   (~40 MB working set vs 48 MB L2), and a division-free decode
   does not recover it.  Dense keeps the direct 3-D grid at the
   new (32, 4, 2) block (+0.9 % / −0.2 % vs the old launch); the
   listed variant decodes packed ids `(bi<<20 | bj<<10 | bk)`
   (1024 tiles per axis, asserted host-side).
5. **One-time dead-element zeroing** before the march makes the
   skip invariant unconditional across checkpoint resumes;
   donated no-op faces are excluded — their frozen value must
   survive.

**Measured (RTX 4070 SUPER, graphs on, whole-step ms):** coax2rect
−33 % single / −57 % double; vacuum-tank fixture −10 % / −13 %
(24.9 % dead); fine tank −41 % / −49 % (43 % dead).  Skipping is
bit-identical by construction (elementwise kernels, skipping only
omits writes) and by gate: dense-vs-skip BIT equality marched with
CPML + graphs in both dtypes
(`tests/integration/test_tile_skip_solver.py`,
`test_tile_skip_kernels.py`); plan-vs-census equality to the digit
(68.03 % / 24.88 %); full suite 1607 passed.

**Limitations (recorded):** TF/SF plane-wave sources disable
skipping entirely (`inject_H` is beta_H-weighted and beta_H != 0
inside PEC; footprint marking is the upgrade path if scattering
runs ever need the speed); periodic BCs disable; CPU
Numba/stencil paths stay dense (possible follow-up); capture on
coarse curved geometries is resolution-limited (tank 24.9 % at
2 mm cells vs 65.7 % raw — the tile skin shrinks with refinement).

## DD-101 — Prefiltered line-solid queries for edge PEC fractions

**Status:** Decided 2026-07-30 (session 134).  Shipped default-on
(pure implementation change inside
`occ_backend.compute_edge_pec_fractions`; no API or semantics
change — bit-identical on the identity gate).

**Problem.**  `compute_edge_pec_fractions` dominated realistic mesh
builds (session-133 census: 118.7 s of 130.6 s, 90.9 %, at 96
primitives) and scaled superlinearly with the primitive count
(scaling profile at ~1000 primitives:
``benchmarks/profile_csg_scaling.py``):
every per-edge query — `BRepIntCurveSurface_Inter.Init(shape, …)`
for the crossings and `BRepClass3d_SolidClassifier.Perform` per
sub-segment midpoint — rescans *every* face of the fused PEC solid
(measured ~1 µs/face each), and the face count grows with the
primitive count.  Threads cannot help (pythonocc's SWIG layer
never releases the GIL, session-133 measurement).

**Decision — three cooperating mechanisms, one helper class**
(`_PrefilteredLineSolid`):

1. **Face-bbox prefilter.**  Faces and their `Bnd_Box` extents are
   collected once per call; a slab test over the (tolerance-padded)
   boxes yields each query line's candidate faces with their
   parameter intervals [w_in, w_out].  Intersections run only
   against candidates, through per-face
   `IntCurvesFace_Intersector` objects built lazily and cached —
   construction digests the face restriction once, which is the
   expensive part for faces with many wires (a plate pierced by
   hundreds of slots made every whole-shape `Init` touching it
   O(primitives), the last surviving superlinear term).  Verified:
   per-face intersectors return bit-identical W and state, and
   their `Transition` is already face-orientation-resolved (the
   whole-shape intersector reports it relative to the natural
   surface normal instead).
2. **Transition-derived states.**  Sub-segment inside/outside
   states come from the orientation-resolved crossing transitions
   (entering/leaving along the line) instead of per-midpoint solid
   classification.  Crossing parameters, dedup, boundary
   construction and the outside-length accumulation are unchanged
   line-for-line, so clean edges reproduce the old float results
   exactly.
3. **Carrier-line cache.**  Structured-grid edges are collinear in
   droves; the full crossing structure of each distinct carrier
   line (candidates, transversal crossings, clean flag) is computed
   once and shared by every edge on it.  Parity along a full line
   is anchored for free — a line enters the bounded solid from
   outside.  Fallback midpoints (edges whose window hits cannot
   anchor transitions: face-border hits, tangencies, inconsistent
   alternation) are classified by cached perpendicular probe
   lines under the same parity rule, with `TopAbs_ON` semantics
   preserved (a crossing within tolerance of the point = on the
   boundary = PEC side); the O(faces) full-solid classifier
   remains only as a last resort for points whose every probe
   line is untrusted (~0.02 % of edges on the slot fixture).

**Gate (bit-identity):** old implementation vs new on four solids
(48-tooth comb incl. deliberate in-plane/on-face edges, cylinder
bore incl. tangent edges, sphere+brick fuse, degenerate edges) —
`np.array_equal` exact on all 2 065 edges.

**Measured.**  Micro (comb, 1 158 faces): 3.80 → 0.088 ms/edge
(43x); per-edge cost now grows 2.1x for 15x faces (vectorised slab
residue) instead of 15x.  End-to-end slotline beam coupler
(`userscripts/beamcoupler_slotline.py` geometry, public API,
n slots / cells / `Mesh.from_geometry` wall):

| n | cells | before | after |
|---|---|---|---|
| 12 | 174 200 | 16.9 s | 6.4 s |
| 50 | 569 400 | 142 s (user-reported) | 33.1 s |
| 100 | 1 089 400 | ~295 s | 53.8 s |

Per-cell cost is flat-to-falling with size (37/58/49 µs) — the
superlinear term is gone.  f_L itself: 118.7 s → 23.2 s at the
doubled-size case; `compute_face_material_areas` (20.6 s at n=50)
is now the dominant mesh-build term (separate site, untouched).

**Limitations / follow-ups:** the whole loop is still one Python
thread — route (b) from session 133 (process pool over edge
chunks, prefill machinery reuse) remains available as an
independent multiplier; probe-line slab tests are O(faces) each
(NumPy-vectorised, ~ns/face — a bin/BVH structure would flatten
the residue if geometries grow another order of magnitude);
`compute_face_material_areas` deserves the same treatment next.

## DD-102 — Planar section engine for face material areas

**Status:** Decided 2026-07-30 (session 135).  Shipped default-on
(implementation change inside `occ_backend` /
`filling.compute_conformal_mu`; no API change — mesh results
bit-identical on all three mesh gates).

**Problem.**  After DD-101, `compute_face_material_areas` was the
dominant mesh-build term (20.6 s of 33 s at 50 slots): its
section-based pipeline calls `cross_section_polygons` once per
(plane, shape), and each call runs a `BRepAlgoAPI_Section` Boolean
against the whole shape.  Measured on the slotline coupler at n=50:
4 486 sections totalling 17.1 s sequential.  All three "use OCC
harder" routes were measured and rejected:

- candidate-face compound (only faces whose bbox meets the plane):
  6.8 → 4.9 ms per plane — the Boolean's fixed cost is ~0.35 ms per
  *candidate* face pair, and a single-face section still costs
  1.05 ms;
- one multi-plane Boolean per axis (compound of N bounded plane
  faces as tool): superlinear in N (1.3 ms/plane at N=10, 8.8
  ms/plane at N=1320) — worse than per-plane calls;
- the existing spawn pool: at this size it was *slower* than
  sequential (20.1 s vs 17.1 s — 3 pool startups, BRep broadcast,
  OCCT-TBB oversubscription).

In addition, the O(E²) `BRepBuilderAPI_MakeWire` wire assembly in
`cross_section_polygons` dominates contour-rich planes (an x-plane
through 50 slots: 58 ms assembly on 18 ms Boolean).

**Decision — four cooperating changes:**

1. **Exact planar fast path** (`_PlanarSectionEngine`, one instance
   per shape in `compute_face_material_areas` and
   `batch_cross_sections`).  Faces (planarity, outward normal from
   the parametric XDir × YDir cross product and the face
   orientation — NOT `Axis().Direction()`, which an indirect gp_Ax3
   flips), straight edges (exact endpoints), edge→face adjacency
   and bounding boxes are collected once per shape.  A plane whose
   bbox candidates are all planar faces and straight edges crossed
   strictly transversally (no vertex within the shape-tolerance-
   anchored on-plane band, no coplanar face) is sectioned exactly:
   one intersection point per crossed edge; per-face segments by
   parity along the face/plane intersection line, directed along
   n_plane × n_outward (outer contours and holes thereby
   counter-rotate consistently — the only orientation property the
   area kernels rely on, since they take abs() per shape); segments
   stitch into closed chains through shared edge indices — exact,
   no tolerance matching, because both adjacent faces reference the
   same intersection-point object of their common edge.
2. **Per-plane delegation for everything else.**  Curved candidate
   faces or edges, tangencies, vertex-on-plane, coplanar faces (all
   DD-087 degenerate planes are in this class by construction) and
   stitch anomalies return ``None`` and run through the unchanged
   `cross_section_polygons` Boolean — every boundary-case semantic,
   including the degenerate-plane behaviour the DD-051/DD-087 M_mu
   machinery depends on, is preserved verbatim.
3. **Pool triggers on delegated work only.**  Fast-path-answerable
   queries no longer reach the prefill; the pool fires on the raw
   query count (unchanged threshold) OR on the work-weighted sum
   ``Σ face_count`` of the queried shapes
   (`_SECTION_PARALLEL_MIN_FACE_WORK` = 150k ≈ the ~5 s pool
   startup at the measured 40–80 µs per face-query).
4. **The DD-099 geom-only call shares the section cache** (its
   SEPARATE call is what carries the DD-099 semantics; cache
   entries are keyed ``(axis, plane_pos, shape)`` and deterministic
   at fixed deflection, so a hit returns exactly what a fresh cache
   would recompute).  `compute_conformal_mu` now normalises
   ``section_cache=None`` to a call-shared dict like the eps path.

**Found on the way:** BOPAlgo orients disjoint section-contour
groups independently — on coupler x-planes the OCC path's net
signed area is off by twice a group area while the |area| multisets
match to 4e-13.  Production never noticed because the area kernels
take abs() per shape and a face rectangle essentially never spans
two independently-flipped groups.  The engine's net is consistent
by construction (verified against hand-computed cross sections);
the polygon gate therefore compares |area| multisets, not net sums.

**Gates.**  (a) Polygon gate: engine vs `cross_section_polygons`
on seven solids (coupler union/brick, cylinder, brick−cylinder,
sphere+brick fuse, plain brick, full coupler union), 360 planes
each across all axes: sorted-|area| multisets match ≤ 4.3e-13
relative; delegation is 100 % on the cylinder and exactly the
curved-candidate planes on mixed solids.  (b) Mesh gate:
`Mesh.from_geometry` old-vs-new on beamcoupler n=12, a coax
cylinder pair, and a mixed brick+cylinder-post model —
**bit-identical** in every float array of the mesh (not guaranteed
in general: polygon vertex rotation may differ on non-axis-aligned
geometry; the |area| gate is the general bound).  (c) Suite: unit
+ integration green.

**Measured** (slotline beam coupler, `Mesh.from_geometry` wall
time; f_L = DD-101 edge fractions, areas = this site):

| n | cells | total before | after | areas before | after |
|---|---|---|---|---|---|
| 12 | 174 200 | 6.4 s | 3.7 s | ~4.6 s | 0.58 s |
| 50 | 569 400 | 33.1 s | 14.0 s | 20.6 s | 2.38 s |
| 100 | 1 089 400 | 53.8 s | 33.6 s | ~21 s | 6.54 s |

Engine query cost ~0.2–0.5 ms/plane (27x under the OCC Boolean on
the coupler union).  The remaining delegated cost is the
coplanar-plane class (grid-snapped slot boundaries: O(n) planes ×
O(n) shape faces); the user-reported 569 400-cell case is now 14 s
against 142 s at the session-133 baseline.  With areas off the
critical path, `compute_edge_pec_fractions` dominates again
(22.8 s of 33.6 s at n=100) — route (b) is the next lever.

**Limitations / follow-ups** (revised 2026-08-26: DD-199 sections
free-form faces on a lifted triangulation, and KB-031 showed the
independent group orientation above was *not* harmless — a rectangle
inside a hole is covered by hole and outer contour alike; contours are
now wound by nesting parity)**:** curved faces always delegate — a
quadric extension (IntAna plane×cylinder sections are exact
circles/line pairs) would extend the fast path to coax-class
geometries if their meshing ever dominates; the O(E²) wire
assembly still stands in the delegated path; route (b) (process
pool over edge chunks for `compute_edge_pec_fractions`) remains
the next independent multiplier.

## DD-103 — Boundary conditions belong to the model, declared once

**Status:** Decided 2026-07-30 (session 136).  Shipped; hard API
break (MAJOR = 0).  Supersedes the background-driven bbox-wall rule
of DD-049 and folds in the BC-PEC consolidation of DD-050.

**Problem.**  The boundary closure was declared on the *analysis*
(`AnalysisScatteringTD(boundary_conditions=...)`), but three of its
four consequences happen at mesh-build time, before the analysis
exists.  They were therefore steered by separate, unchecked
parameters — or by nothing at all:

| consequence | was steered by |
|---|---|
| CPML grid extension | `Mesh.from_geometry(pml_faces=...)` |
| PMC grid-line pull-in (WP-U0) | `Mesh.from_geometry(pmc_faces=...)` |
| PEC wall mask | nothing per face — see below |
| runtime BC objects | `AnalysisScatteringTD(boundary_conditions=...)` |

Nothing tied the two declarations together.  The PML depth even
existed twice as an independent number (`MeshControl.
pml_thickness_cells` for the grid extension,
`AnalysisScatteringTD.cpml_thickness_cells` for the profile), so a
layer could grade over a span the grid did not have.

The wall mask was the damaging one.  DD-049 keyed it on the
*background material*: `background.is_pec` force-masked the
tangential E-edges of **all six** bbox faces, to keep a PEC chamber
wall one connected component for the auto-conductor detection
(a dielectric touching the bbox at isolated tangent points otherwise
un-masks the edges between those cells and fragments the wall).
That rule cannot see the declared closure, so it overrode it:

- a **PMC symmetry plane** in a PEC chamber became an electric wall.
  `PMCBoundary` is deliberately a no-op — the magnetic wall is the
  *natural* BC of the free curl operators — so nothing downstream
  could take the mask back off, and `fit_td` freezes masked edges
  with `alpha_E = beta_E = 0`.  The half-model then ran the opposite
  symmetry, silently.
- a **CPML face** became a mirror in front of the absorber (same
  mechanism; no shipped example hit it because every PML fixture
  uses an air background).
- on a **port touching such a plane**, the mode-path detection
  (`resolve_declarative_port` → `extract_conductor_groups_from_mesh`)
  saw the inner conductor fused to the wall frame: one PEC component
  instead of two, so the TEM/QTEM path was rejected and the port
  resolved as a hollow TE/TM guide.  Reported on a half-modelled
  rectangular coax (`userscripts/beamcoupler_slotline.py`), where
  both ports came back TE.

**Decision.**  The closure is a property of the modelled domain, not
of a run on it — a PMC face is a symmetry plane, a CPML face an
opening.  It is declared once, on the `GeometryModel` (or on
`Mesh.from_grid` for the OCC-free path), carried by the `Mesh`, and
read from there by the layer-a analyses:

```python
model = GeometryModel(background=pec, boundary_conditions={
    "xmin": "PMC", "xmax": "PEC", ...})
mesh = Mesh.from_geometry(model, control, f_max)
analysis = AnalysisScatteringTD(mesh=mesh, ports=[...], f_max=f_max)
```

Rules:

1. **The declaration decides the wall, per face.**  `PEC` masks that
   face's tangential edges; `PMC`/`CPML`/`Periodic` do not.  The
   background fills the *volume* outside every shape and no longer
   closes anything by itself.  DD-049's purpose survives: a declared
   PEC face is force-masked whole, so the wall stays one component
   under dielectric tangency.
2. **An undeclared face closes with PEC** — the conventional closure
   and the safe default.  Note the change of meaning: a *partial*
   BC dict used to leave the remaining faces to the free curl update
   (i.e. a magnetic wall); those faces are now electric.  Call sites
   that wanted the old behaviour say `"PMC"` explicitly.
3. **One type per face.**  The former `pml_faces`/`pmc_faces` overlap
   check is unrepresentable now and was dropped.
4. **The CPML depth lives on the declaration**
   (`BoundaryConditions.cpml_thickness_cells`), driving both the grid
   extension and the profile.  `MeshControl.pml_thickness_cells` and
   `AnalysisScatteringTD.cpml_thickness_cells` are gone.
5. **Layer C is untouched.**  `FITTimeDomainSolver` /
   `EigenmodeSolver3D` keep their `boundary_conditions=` argument —
   they must stay usable without a geometry.

`Mesh.with_boundary_conditions(bc)` *replaces* a closure on a built
mesh: PEC faces are masked, faces that are no longer PEC get the
edge values they had before any wall was forced on them (kept in
`Mesh._wall_backup` — the OR is lossy on its own).  It cannot grow a
CPML extension or move a PMC grid line, so declare those on the
model; conversely that is exactly why fixtures which must keep their
grid use it instead of re-declaring upstream.

**API changes (hard).**

- `GeometryModel(boundary_conditions=...)`, `Mesh.from_grid(
  boundary_conditions=...)`, `Mesh.boundary_conditions`,
  `Mesh.with_boundary_conditions()` — new.
- `Mesh.from_geometry(pml_faces=, pmc_faces=)`,
  `MeshControl.pml_thickness_cells`,
  `AnalysisScatteringTD(boundary_conditions=, cpml_thickness_cells=)`,
  `AnalysisEigenmode(boundary_conditions=)` — removed.  The two analyses
  expose `boundary_conditions` / `cpml_thickness_cells` as read-only
  properties onto the mesh.
- `BoundaryConditions.to_objects(grid)` — the
  `cpml_thickness_cells` argument moved into the dataclass.
- New readers in `boundaries.boundary_conditions`:
  `bc_type_entries` (any accepted form → `{face: type}`),
  `resolve_boundary_conditions`, `cpml_thickness_of`.  These replace
  four copies of an `isinstance` cascade in the analysis.
- The closure left the resume recipe (`RECIPE_SCHEMA_VERSION` 1.0 →
  2.0) for `mesh.h5`, where it round-trips as a type map plus the
  CPML depth.

**Verification.**  Full suite green: 1335 unit + 262 integration
(27 skipped).  `tests/unit/test_boundary_closure.py` pins the
defect: on the half-model rect coax the port resolves as
`PortSpecMultiConductor` with `xmin="PMC"` and as `PortSpecNumerical`
under the old all-PEC forcing; PMC and CPML faces stay unmasked
under a PEC background; `with_boundary_conditions` is path
independent and round-trips.  On the reported coupler both ports now
solve TEM (f_c = 0, Z = 117.7 Ω on the half model) instead of TE.

**Limitations / follow-ups.**

- A `PECBoundary` carrying its own wall material (DD-099) still does
  not survive a store round-trip — the closure persists as a type
  map.  Pre-existing (the recipe route had the same hole), not
  addressed here.
- `Mesh._wall_backup` is not serialised, so on a mesh reloaded from
  the store `with_boundary_conditions` can only re-declare the same
  closure (which is what the store does).
- `examples/pml_verification_coax_discrete.py` turned out to import
  `TransientAnalysis`, a class that no longer exists in `src/` — dead
  before this change, and deleted right after it (session 136).

## DD-104 — Monitor regions are corner boxes; open-ended recording schedules

**Status:** Decided 2026-07-30 (session 136).  Shipped; two hard API
breaks on the three monitor classes (MAJOR = 0).

**Problem.**  Both defects surfaced from one user script.

*Region.*  Monitors took `center=` and `size=`, with `size = 0`
encoding a point/plane and `size = inf` a full-domain extent.  A user
reaching for "the whole domain" wrote
`center=(-1e30,)*3, size=(2e30,)*3`, reading the pair as *start* and
*stop* — two corners.  The interval it actually describes is
`[-2e30, 0]`, whose upper bound lands on exactly 0: on a domain at
x, y ≥ 0 that resolves to the boundary-nearest cell in x and y, i.e.
a **line at the domain edge instead of the volume**, recorded without
complaint.  Nothing about the spelling is wrong — the mistake is
invisible because `(center, size)` and `(corner, corner)` are the same
shape of input with different meanings.  Every practical selection
(this box, this plane, this point) is naturally stated by its corners,
and the geometry API already does exactly that
(`Brick.from_corners`).

*Schedule.*  `FieldTimeMonitor` took `times=`, a materialised array of
instants.  A run whose length is set by a stop criterion has no known
end time, so "record every 0.5 ns until the run ends" was
inexpressible; the attempt, `np.arange(0.5e-9, 1e30, 0.5e-9)`, asks
for 2·10^39 elements and dies in `np.arange` (which is eager) before
allocating.

**Decision.**

1. **`corners=((x0, y0, z0), (x1, y1, z1))`** replaces `center`/`size`
   on `FieldTimeMonitor`, `FieldFrequencyMonitor` and
   `FluxTimeMonitor`.  Corner order is free (each axis is sorted).  A
   component may be `None` — "unbounded on this side", read as `∓inf`
   by position — or an explicit `±math.inf`.  Omitting `corners`
   records the whole domain.  An axis whose two values coincide is
   degenerate and selects one cell layer: that *is* the plane / line /
   point case, so the former `size = 0` convention disappears.
2. **`FluxTimeMonitor` reads its normal off the degenerate axis.**
   Exactly one axis may be degenerate, and its coordinate is the plane
   position — replacing "exactly one zero-extent dimension".  The
   tangential bounds remain unused: this monitor always integrates the
   full cross-section (unchanged behaviour, now stated in the
   docstring rather than hidden behind an `inf` that looked load-bearing).
3. **`FieldTimeMonitor(interval=, start=)`** records every *interval*
   seconds from *start* until the run ends; `times=` stays the
   explicit-instants form.  Exactly one of the two must be given.  The
   k-th target is `start + k·interval` computed absolutely, so the
   cadence cannot drift; the sample lands on the step nearest its
   target (±dt/2).  Progress stays a single counter, which is what
   `state_dict()` already persisted, so the resume path was unaffected.

**API changes (hard).**  `center=` / `size=` removed from all three
monitors; `resolve_region(center, size, grid)` →
`resolve_region(corners, grid)`, joined by `normalize_corners`.  The
store keeps a `(2, 3)` corner array instead of the `center`/`size`
attribute pair, the resume recipe `[[x0,y0,z0],[x1,y1,z1]]` with the
existing `±inf` string sentinels (a bare `Infinity` token is not
standard JSON), plus `interval`/`start`; recipes predating the
schedule carry `times` only and still load.

**Verification.**  Full suite green (1347 unit + 262 integration, 27
skipped), 13 new monitor tests: corner order, `None`/`inf`
equivalence, half-open axes, degenerate-axis validation, the interval
cadence over runs of different length, and the recipe round trip.

**Limitations / follow-ups.**  An open-ended interval monitor on a
long in-RAM run accumulates snapshots without bound — the streaming
path (`project=`) is the answer, and the docstring says so.  A
partial-aperture flux (integrating over a sub-rectangle rather than
the full cross-section) is still not available; the corner form now
makes it expressible, so implementing it is a contained follow-up.

## DD-105 — Mesh warnings report what costs something

**Status:** Decided 2026-07-30 (session 137).  Shipped; no API change.
One warning removed, one added, and the cell-count rule that caused
the reported loss given tolerance.

**Problem.**  `check_quality` warned above a global cell-size ratio
`max(dx ∪ dy ∪ dz) / min(dx ∪ dy ∪ dz) > 10`.  On the reported coupler
that fired at 76.3, pairing a z cell of 14.99 mm with a y cell of
0.197 mm — two cells 100 mm apart, on different axes, that never see
each other.  It is not a cell aspect ratio, and not a mesh property at
all: 14.99 mm *is* λ/10 at f_max (the wavelength criterion, met
exactly) and 0.197 mm is the feature size, so their ratio only
restates how far apart the problem's smallest geometry and its
wavelength are.  A 1 mm wall at 150 mm wavelength *has* ratio 76.  The
advice it gave ("adjust growth_factor or max_cell_size") points the
wrong way: a smaller `max_cell_size` adds cells and leaves dt
untouched, i.e. is strictly worse.

Behind the false alarm sat a real loss, unreported.  Cell counts per
interval are integers, and the cell-count rules took `h_fine` as a
hard bound — `_grade_symmetric_to_uniform` scanning for the smallest
count whose starting cell fits underneath it:

```python
if h0 <= h_fine * (1.0 + 1e-10):
```

On the coupler's 2 mm rail gap, `n = 6` yields `h0 = 0.2506 mm`, which
exceeds `h_fine = 0.25 mm` by **0.24 %** and was rejected; `n = 7`
then yielded 0.1965 mm.  Since the explicit time loop takes one global
step bounded by the smallest cell anywhere, that 21 % undershoot cost
steps across the whole model and bought resolution nowhere.  Not an
exotic case: the same 8·h_fine constellation appears in two unrelated
suite fixtures, because wall thicknesses and conductor thicknesses
tend to sit in small integer ratios.

**Decision.**

1. **The global aspect-ratio warning is removed.**  A dimensionless
   grid ratio with no reference to the wavelength cannot carry an
   accuracy or stability statement.
2. **Cell counts may overshoot `h_fine` by `_H_FINE_TOL` = 5 %.**
   `h_fine` is a convention (`min_gap / min_cells_per_feature`), not a
   physical constant, so a few percent above it is free — while
   refusing costs a whole extra cell.  Applied in
   `_grade_symmetric_to_uniform` and `_n_one_sided`, i.e. exactly
   where `h_fine` bounds a count.  **Never applied to `h_max`**: that
   is the user's wavelength criterion, set deliberately
   (`min_nodes_per_wavelength`), and 5 % of it is theirs to give, not
   the mesher's.
3. **A grading-undershoot warning covers the remainder**
   (`check_grading_undershoot`), reported only for the *globally*
   smallest cell — cells elsewhere do not bound dt, so reporting them
   gains nothing.  Silent on anchor pairs (forced planes, thin
   sheets), on intervals shorter than `h_fine`, on a cell already
   sitting on the user's `min_cell_size`, and on an axis whose fine
   size is wavelength-driven (`h_fine >= h_max`) rather than
   feature-driven.  Threshold 15 %: the gaps between consecutive cell
   counts mean an undershoot of up to ~13 % can be unavoidable (for a
   1 mm interval, `n = 3` overshoots by 21 % and `n = 4` undershoots
   by 13 % — there is nothing in between), so a tighter threshold
   would report what no setting can fix.  The message names
   `min_cell_size` with the value that removes it; taking that offer
   trades feature resolution for dt, which is the user's call.
4. **The growth-factor threshold stays at 2.0**, now with a measured
   basis rather than none — see below.

**Measurement (grid transition reflection).**  1D leapfrog on a
non-uniform grid — the same second-order scheme FIT uses per axis,
isolated from ports and modes.  The reflected wave is obtained by
differencing against a reference run whose grid stays fine everywhere,
so the source, the left boundary and its residual reflection cancel
exactly.  Calibration: a uniform grid (g = 1) returns −240 dB, i.e.
round-off.  Courant number 0.5, so the dispersion term carries
(1 − S²) = 0.75; for axis-parallel propagation in 3D the corresponding
value is ≳ 2/3, so the figures are representative and not flattered by
the 1D magic time step.

Single step h → g·h, Γ in dB by cells per wavelength N in the *coarse*
region:

| g | N=30 | N=20 | N=15 | N=10 |
|---|---|---|---|---|
| 1.1 | −66.4 | −59.2 | — | — |
| 1.3 | −59.0 | −51.8 | — | — |
| 2.0 | −53.7 | −46.6 | −41.4 | −34.1 |
| 4.0 | −51.7 | −44.6 | −39.6 | −32.2 |

A full ramp 1 mm → 8 mm, same endpoints, only the steepness differing:

| g | ramp cells | N=30 | N=20 | N=10 |
|---|---|---|---|---|
| 1.1 | 22 | −55.9 | −54.8 | −48.3 |
| 1.3 | 8 | −51.9 | −45.5 | −37.0 |
| 2.0 | 3 | −51.4 | −44.3 | −32.2 |
| 4.0 | 2 | −51.3 | −44.3 | −32.1 |

Both sets follow `Γ ≈ 2.5 · (h_coarse/λ)² · (1 − 1/g²)` to within
0.5 dB over the whole range.  The consequences:

- **Resolution dominates, the growth factor does not.**  Halving N
  costs 12 dB; g = 1.3 → 4 costs 7 dB, and g = 2 → 4 only 2 dB — the
  (1 − 1/g²) factor saturates.
- At the default `min_nodes_per_wavelength = 20`, one transition sits
  below −44 dB even at g = 4.  A threshold on g therefore separates
  nothing the wavelength criterion has not already settled.
- A gentle ramp is worth its cells only where the coarse region is
  marginally resolved: at N = 30, g = 1.1 buys 4.6 dB over g = 4 for
  22 cells instead of 2; at N = 10 the same trade buys 16 dB.

The threshold is therefore kept at 2.0 not as an accuracy limit but as
a **pathology gate**: above it a grading mesher has stopped grading,
which points at the mesher, not at the physics.  The warning text says
that instead of the former "may cause numerical dispersion", which the
measurement does not support at any g the mesher can produce.

**Verification.**  Full suite green (1624 passed, 27 skipped) with
every mesh in the repository changed by the tolerance and no numerical
reference moved — the 5 % are physically inert.  The six undershoot
warnings the suite produced before the tolerance are all gone; the
reported coupler is the one case where a remainder survives, which is
what the warning is now for.  15 tests in
`tests/unit/test_mesh_quality.py` cover the undershoot check
(threshold, anchor pairs, short intervals, user floor, wavelength-
driven axes, global-minimum selection, suggested value), pin that a
wide cell-size spread at a smooth gradient is *not* a warning, and pin
both sides of the tolerance — the six-cell result end to end, and the
warning still reaching the production path with the slack patched to
zero.  On the reported coupler the tolerance alone takes the mesh from
569 400 to 525 600 cells with h_min 0.1965 → 0.2112 mm; the warning
then offers `min_cell_size=0.00025`, worth 494 064 cells and h_min
0.2506 mm — together −13 % cells and +28 % on the time step.

**Limitations / follow-ups.**  The measurement covers reflection and
dispersion at a transition; it says nothing about the stability margin
of strongly non-uniform grids (magnelio derives dt from the actual
grid, so this is bounded anyway) or about material discretisation on
oblique interfaces in heavily graded regions.  `_H_FINE_TOL` is a
single global constant; a per-interval rule (accept the count whose
`h0` is *closest* to `h_fine` when the alternatives straddle it) would
capture the remaining cases but changes more meshes for less gain.

## DD-106 — Deterministic conventions on tangent section planes

**Status:** Decided 2026-07-30 (session 138).  Shipped; no API change
(one new optional `domain_bounds` parameter on
`compute_face_material_areas`).  Supersedes the DD-087 decision to
keep the matrix channel on raw exact-plane sections.

**Problem.**  `BRepAlgoAPI_Section` with a plane that *contains* a
flat face of the solid is ill-posed — the intersection is
two-dimensional, not a curve — and OCC returns coincident-face
boundary wires, tessellation fragments or nothing, depending on the
solid's topology.  This is not an edge case but the constructed
normal case: the mesher snaps grid lines onto exactly these planes
(`extract_critical_planes`), so every axis-parallel material boundary
sits on a section plane.  Two consequences, both measured on
`userscripts/beamcoupler_slotline.py` (session 137):

1. The M_μ matrix channel (`pec_frac` → `A_face_free`) took whatever
   the exact-plane section returned — DD-087 had deliberately left it
   raw for bit-identity.  On the coupler's chamber ceiling the PEC
   fraction ran 13.5 → 40.6 → 67.7 → 94.7 → 100 % along a
   translation-invariant wall (a tessellation-diagonal front), and
   mid-wall H columns flipped between 0.0 and 1.0.  The DD-067
   stage-2 port certificate correctly measured a feed-chain slab
   defect of 4.55e-01 and withheld the exact DTBC from a geometry
   that is perfectly invariant along the port normal.
2. Degeneracy detection tested only shape *bounding-box* tangency.
   The coupler ceiling lies in the interior of one
   `Union(chamber, slots, rail_line)` solid, so no bbox witnessed it:
   both channels took the garbage, and the flat-wall jump
   (`A_face_pec_jump`) stayed zero — the wall was also missing from
   the DD-099 wall-area bookkeeping on such geometries.

Reproduced minimally: an air chamber with one slotted wall, built as
a Union, returns a 0 → 0.5 → 1.0 `pec_frac` front stepping at a
position where the geometry has no feature.  Built from separate
bricks (bbox tangency fires), the same wall reads deterministically —
the answer depended on how the solid was *composed*.

**Decision.**

1. **Detection is per face, not per bbox.**  A plane is degenerate
   when it lies within `1e-12·(1+|p|)` of any axis-normal tangent
   position of any face of any shape, computed by
   `_face_critical_planes` — the same machinery whose output the
   mesher snaps grid lines to, so precisely the constructed
   coincidences are caught (planes exactly; cylinder/sphere mantle
   tangents analytically; cones/tori/free-form conservatively via the
   retained bbox extents).  Cached per shape in the shared section
   cache (`("__tangent_planes__", si)`).
2. **A degenerate plane is never sectioned exactly.**  Both shifted
   positions `p ± deflection` are sectioned instead and feed *every*
   channel.
3. **Matrix channel: min-convention.**
   `A_PEC := min(A_PEC(p+δ), A_PEC(p−δ))` — a face is blocked only
   where it is *embedded* in PEC on both sides; a wall merely
   tangential to the face leaves it free.  This is the staircase
   limit (a perfectly gridded PEC wall face keeps its DOF; the flux
   through it stays zero dynamically because its PEC edge circulation
   vanishes), it keeps conformal resolution for in-plane structure
   (a face partly wall, partly slot opening gets the transversal
   fraction), and it is translation-invariant along extruded feeds.
   Property averages (μ̄, σ*, DD-093 material fractions) take the
   arithmetic mean of the two sides — the staircase cell-pair mean in
   the gridded limit.
4. **Geometric channel: max-convention, unchanged semantics**
   (DD-087: wall area lands in the adjacent non-PEC cell, jump =
   signed side difference) — but now firing on interior tangent
   planes too, which closes the wall-bookkeeping gap on Union/
   Difference geometries.
5. **Domain-hull planes are one-sided for the matrix channel.**
   `compute_conformal_mu` passes the grid extents as
   `domain_bounds`; a degenerate plane on the hull uses the interior
   side only — averaging with the fictitious outside would report
   μ̄ = 1.5 for a μ_r = 2 feed at the port plane.  The geometric
   channel keeps both sides (a registered domain-end plane must still
   read as a shorting lid, DD-099).

**Bit-identity is given up** where DD-087 had preserved it: every
geometry with grid-snapped flat boundaries gets different (correct)
M_μ entries on tangential faces.  Measured consequences: the full
suite passes unchanged (1366 unit tests green, no numerical reference
moved); coupler port certificate 4.55e-01 → 2.6e-13 / 1.6e-12
(p1/p2), i.e. the invariant-feed class, exact DTBC restored on both
ports; TEM at 117.65 Ω unchanged; mesh time unchanged (15.0 s).

**Traps.**

- The flipping mid-wall columns (session 137, y = 50.5 mm) were on a
  plane the bbox test could never see; do not reintroduce bbox-only
  reasoning anywhere in degeneracy handling.
- The per-side property kernels write a value for every positive-area
  face (background fill), so "NaN = staircase" continues to mean
  *non-candidate* faces only; the min/mean combination must not
  invent values where both sides are unprocessed.
- `_face_critical_planes` filters tangent candidates against the
  trimmed face bbox — Boolean trim edges do not become degenerate
  planes.  Keep that filter; without it every Boolean seam would
  force the slow two-sided path.
- The DD-099 geom-only batch stays a separate classifier call; its
  faces share the section cache but never the per-call plane
  grouping.

## DD-107 — Three equidistant cells at every domain face

**Status:** Decided 2026-07-30 (session 138); shipped session 139
(session 138 ended in a machine crash mid-edit — the flag gating and
the `_absorb` helper were completed afterwards).  No API change.

**Problem.**  The modal-port operator (DD-040) is a difference
operator between the port plane and the next interior plane;
reference_waveguide_ports.md §2.4 requires the three cells adjacent
to the port plane to be equidistant, or the V/I projection silently
scales by orders of magnitude (the port factory validates this hard).
But ports are declared on the *analysis*, after meshing —
`Mesh.from_geometry` cannot know which faces carry one.  Whether the
boundary grading happened to satisfy §2.4 was decided by an accident
of interval arithmetic: on `beamcoupler_slotline.py` the ramp from
the slot region (h_fine 0.5 mm, g 1.3) saturated at `h_max` = 15 mm
(2 GHz) about 38 mm before the domain wall — uniform tail, port
valid.  At 1.4 GHz (`h_max` = 21.4 mm) saturation needs ~84 mm of the
100 mm beampipe, the last three cells were still on the ramp
(13.9/18.0/23.4 mm, ratio 1.69 = g²), and the port factory rejected a
previously working model because the *frequency* changed.

**Decision.**  The mesher guarantees a uniform tail of
`_BOUNDARY_BUFFER_CELLS` = 3 cells adjacent to **every** domain face
(commercial meshers do the equivalent with port-aware meshing; ports
here are attachable to any bbox face after the fact, so all six get
the buffer).  Two exceptions, both deliberate:

- the **single-cell degenerate axis** stays single-cell (2.5D-style
  thin domains, pinned by `TestDegenerateAxis`) — it can never host a
  waveguide port anyway;
- the hard **`min_cell_size` floor wins** (WP-M3): when three legal
  cells cannot fit, the legacy profile stands and the port validator
  reports the conflict if a port actually lands there.

Implementation is two-stage, because the buffer is a property of the
axis' outermost CELLS, not of its outermost interval — a
forced-planes grid can satisfy §2.4 across interval boundaries
(uniform raster), and rewriting its boundary interval would *destroy*
the equidistance:

1. The first pass generates every interval with the plain profiles
   (`boundary_buffer=False`, bit-identical to the pre-DD-107 grid).
2. The post-pass `_enforce_boundary_buffer` checks each assembled
   axis end and only where the buffer is violated regenerates the
   boundary interval with `boundary_buffer=True`.  Critical planes
   never move; if the regenerated interval still cannot host the
   buffer, the original grading is kept.

Inside `_grade_then_uniform(boundary_buffer=True)` (interior
intervals are untouched):

1. **Saturated path** (uniform run ≥ 3 cells): unchanged — this was
   the accidental-pass case.
2. **Short uniform run / absorbed remainder** (n_uniform < 3, incl.
   the old "absorb rest into the last ramp cell"): ramp cells donate
   their length to a 3-cell uniform tail; the donor count is chosen
   per case to minimise the seam ratio (a fixed count is g-class only
   for the default growth factor).
3. **Interval shorter than the ramp** (the legacy pure-geometric
   fallback — the coupler case): new `_tailed_widths` refit with two
   free parameters — fine-end size `h0` (held at the DD-105 `h_fine`
   tolerance whenever possible) and tail size `h_u` — coupled by the
   seam constraint ratio ≤ g.  Coupling the tail rigidly to the ramp
   end would coarsen the count granularity by the buffer factor and
   reintroduce the DD-105 fine-end undershoot (measured: −24 % on the
   DD-105 end-to-end fixture, warning reactivated; with the two-
   parameter refit: −11.7 %, same class as the legacy refit, silent).
4. Whole-axis-uniform intervals bump 2 cells to 3; 1 stays 1 (see
   above).

**Measured.**  Property sweep (16 632 parameter combinations,
h_fine/h_max/g/min_cell/L grid): buffer violations 5289 → 0, short
intervals 48 → 0; worst neighbour ratio per g: 1.49·g → 1.49·g
(g = 1.15, pre-existing saturated-path case), 1.48·g → **1.20·g** at
the default g = 1.3, 1.38·g → 1.25·g at g = 2.  Beamcoupler at
1.4 GHz: meshes and solves ports without workarounds (previously
`ValueError` from the §2.4 validator; re-verified post-crash:
557 175 cells, TEM 114.25 Ω both ports, chain-slab defect
2.3e-13 / 5.4e-13); at 2 GHz the DD-106 certificate level is
preserved.  Unit suite 1371 passed including the DD-105 undershoot
pins and the ramp fix-point property pin (≤ 1.5·g); 5 new gates in
`TestBoundaryBufferCells`.

**Measured costs (session 139, found by the integration suite the
crashed session never ran).**  Two fixture classes move:

- *Pinched boundary intervals* (a feature tangent plane within
  < 3 cells of the wall): the buffer re-splits the short interval
  and the re-split cell becomes the new global minimum — the
  conformal-dispersion fixture (forced 1 mm raster, cylinder tangent
  0.7 mm from the wall) went 0.3 → 0.233 mm and its fixed
  `DT = 1e-12` marched NaN.  The sanctioned remedy is the floor
  exception: `min_cell_size = 3e-4` restores the legacy grid
  bit-identically (fixture updated so).  **Warning gap (closed
  2026-08-12):** `check_grading_undershoot` skips wavelength-driven
  axes (`h_fine >= h_max`) because pre-DD-107 no undershoot could
  arise there; a buffer re-split cell in such an axis used to set dt
  silently.  Fixed as sketched: the mesher now hands the buffered
  axis ends (and whether they come from declared ports or the
  buffer-all-faces fallback) to the check, which exempts buffered
  boundary intervals from the skip — but only when the buffer forced
  more cells than a plain fill would have used (`n_plain <
  _BOUNDARY_BUFFER_CELLS`); ordinary integer rounding stays skipped
  as the user's accuracy choice.  The warning names the buffer and
  the applicable remedy (declare ports on the model / widen the
  interval).  Gates: `TestBufferUndershoot` (5 unit cases + one
  production-path fixture).
- *Bbox-tangent conductors*: the uniformised wall tail
  (4.25/3.25/2.5 → 4 × 2.5 mm on the DM coax fixture) puts a
  smaller cell in the tangent zone and cuts a thinner μ sliver —
  `μ_eff_min` 0.131 → 0.030, DM CFL ratio 0.285 → 0.138.  Not an
  ECT defect (the 1 % `A_face_free` floor bounds the ratio near
  0.08); the fixture gate is recalibrated 0.25 → 0.10.  Production
  meshes at 20/λ saturate their wall ramps and are untouched.

**Traps.**

- The 1.5·g fix-point pin is sensitive to every profile change at
  g far from the default; the adaptive donor count plus the
  refit-on-seam-violation fallback is what keeps large and small g
  inside it — a fixed donor count is not enough (measured 1.58·g at
  g = 2, 2.03·g at g = 1.09 with short ramps).
- `_tailed_widths` must never run on intervals where the ramp
  saturates (its geometric refit has no h_max cap); it is reachable
  only from the short-interval and donor-fallback paths, where the
  interval is bounded by ramp + 2·h_max.
- `boundary_buffer=False` must reproduce the legacy profile exactly:
  the post-pass compares the assembled axis against the buffer
  property, and an always-buffered first pass would chop a uniform
  forced-planes boundary interval (e.g. 1 mm raster) into three
  sub-cells — seam ratio 3 against the neighbouring raster cell,
  destroying the very equidistance the buffer exists for.
- The buffer changes meshes only where the boundary grading had not
  saturated — fixtures meshed at default `min_nodes_per_wavelength`
  = 20 mostly saturate and stay bit-identical; low-resolution runs
  (10/λ) are where cells move.


## DD-108 — Two-tier public namespace: high-level API + components

**Status:** Decided 2026-08-02 (session 143, pre-release API review
with the developer); shipped same session.  Hard API break (permitted:
MAJOR = 0, no external users yet).  Two-tier framing superseded by
DD-117 (thin core + domain namespaces); the one-home rule, the
underscore-internals marking and the audit tooling remain in force.

**Problem.**  The top-level namespace had grown to 63 flat names plus
two accidental leaks (`GridLines`, the deprecation-shim factory), 34
of them *also* exported from a subpackage — two documented homes per
name.  The "layer a/b/c" model (spec.md §8) never matched the code:
it suggested a horizontal cut through the whole library, but geometry
or materials have no a/b split — the distinction only ever applied to
*solving*.  Meanwhile the single most-needed object, `GeometryModel`,
was not importable from the top level at all.

**Decision.**  Two tiers plus internals, enforced by tooling:

- **High-level API** = the top-level `magnelio` namespace (30 names):
  the model vocabulary + the problem classes.  Placement rule: *a name
  a typical simulation script uses lives at the top level; a name only
  needed when assembling custom simulations from parts lives in
  exactly one component namespace; everything else is an underscore
  module.*
- **Components** = curated subpackage namespaces (`magnelio.ports`,
  `magnelio.solver`, `magnelio.post` — renamed from `postprocessing` —
  `magnelio.plot`, `magnelio.signals`, `magnelio.mesh`,
  `magnelio.boundaries`, `magnelio.materials`, `magnelio.circuit`,
  `magnelio.io`, `magnelio.sources`, `magnelio.constants`,
  `magnelio.analysis`, `magnelio.geometry`), each with a curated
  `__all__`; every public name has exactly one documented home.
- **Internals** = underscore packages/modules (`_operators`,
  `_fields`, `_backend`, `ports/_modal`, `ports/_lumped`, plus
  module-level `_`-prefixes wherever no name reaches a public
  `__all__`) — the former "layer c", made machine-readable.
- The terms "layer a/b/c" / "Level A/B" are retired everywhere
  (spec.md, STATUS.md, docstrings); the documentation vocabulary is
  **high-level API** and **components**.
- `validation/tools/check_api_surface.py` enforces the one-home rule
  and the no-underscore-in-`__all__` rule;
  `validation/tools/check_imports.py` AST-sweeps the script
  directories (no test coverage there) after every rename.

Physical constants moved to `magnelio.constants` (C0 exact, MU0 CODATA
2018, EPS0/ETA0 derived so the free-space relations hold exactly in
floating point) — previously 14 drifting definitions including a
mode-solver C0 of 299792457.66 m/s.  `BoxFace` moved from
`ports/modal/port_plane.py` to `mesh/faces.py`, removing the only
upward `_operators` → `ports` edge.

## DD-109 — Ports are declared on the model, before meshing

**Status:** Decided 2026-08-02 (session 143); shipped same session.
Refines DD-107 (whose all-face buffer remains as the fallback).

**Problem.**  Ports were declared on the analysis, *after* meshing —
the mesher could not know which faces carry one, so DD-107 buffered
all six domain faces, and port-driven mesh refinement (cross-section
resolution, adaptive meshing) had no hook at all.  The
boundary-closure work (DD-103) had already shown the right pattern:
declare a domain property once, early, and let the mesh carry it.

**Decision.**  The declarative ports (`PortWaveguide`,
`PortAnalytical`, `PortLumped`) are declared on the
`GeometryModel` via `add_port()` (unique labels; spec-level ports
rejected — the model must not depend on solver detail).
`Mesh.from_geometry` copies the declarations onto the new
`Mesh.ports` field and buffers exactly the declared faces
(bit-identical there to the all-face result); `Mesh.with_ports()` is
the late-attachment path for `from_grid` meshes.
`AnalysisScatteringTD.ports` defaults to `None` = resolve the
mesh-carried declarations; an explicit `ports=` overrides them
completely.  Mode physics resolution is unchanged (analysis-time,
against the finished mesh); `resume` recipes keep serialising
resolved specs.  The declarations round-trip through `mesh.h5`.

- A model with **no** declarations keeps the DD-107 all-face buffer,
  so "mesh first, ports at analysis time" workflows stay valid; the
  §2.4 port validator remains the backstop either way.
- The mesher still imports nothing from `ports/` — it reads only the
  declared plane (`BoxFace` lives in `mesh.faces`); interior ports
  (lumped) request no face buffer.
- `PortLumped` is new: the high-level spelling of the lumped Thévenin
  port (endpoints, Z0, optional RLC element), resolved to
  `PortSpecLumped`.

## DD-110 — One port naming scheme; "Lumped" is canonical

**Status:** Decided 2026-08-02 (session 143); shipped same session.

**Problem.**  Three naming schemes coexisted (`PortSpecCoax`,
`PortWaveguide`, `DiscretePortOperator` vs `ModalPortOperator`), four
report classes followed three patterns, and "discrete" vs "lumped"
named the same concept.

**Decision.**  The qualifier always follows "Port": declarative
`Port<X>` (`PortWaveguide`, `PortAnalytical`, `PortLumped`), specs
`PortSpec<X>`, operators `PortOperator<X>` (`PortOperatorModal`,
`PortOperatorLumped`, `PortOperatorBandDTBC`).  Canonical term is
**Lumped** (`ports/discrete/` → `ports/_lumped/`,
`PortSpecDiscrete` → `PortSpecLumped`, `build_discrete_port` →
`build_lumped_port`).  Reports: `PortReport` (user-facing, from
`solve_ports()`; formerly `PortModeReport`), `PortOperatorReport`
(the DD-048 operator diagnostic; formerly `PortReport` — renamed
first to vacate the name), `ModeReport` (one mode),
`ModeRefinementReport` (formerly `RefinedModeReport`).  Problem
classes share the `Analysis` prefix (`EigenAnalysis` →
`AnalysisEigenmode`) so autocompletion enumerates them.

## DD-111 — Store schema v1.0: one version, hard validation

**Status:** Decided 2026-08-02 (session 143); shipped same session.

**Problem.**  Four independent schema constants (project 2.0, results
2.0, checkpoint 1.0, recipe 2.0) were written but never read back for
branching; compatibility with older stores lived in silent
`.get()`-with-default sites.  Publishing that state would have made
the accumulated tolerances an invisible permanent contract.

**Decision.**  `io/_schema.py` holds the ONE `SCHEMA_VERSION = "1.0"`;
every artefact stamps it and every reader hard-validates it
(`ProjectSchemaError` with a "re-run the simulation" hint).  The
tolerance sites became required keys (recipe precision / monitors /
wall\_\*, checkpoint `peak_signal`); `Material.is_lossy_metal` dropped
the legacy `sigma = inf` special case.  Genuinely semantic absent-key
encodings stay (absent `dispersion` = non-dispersive).  Additions:
`Mesh.ports` round-trips through `mesh.h5` (DD-109); a free-form
`params=` dict on the analysis is stored and read back as
`Project.params` (the sweep/optimizer hook without a sweep
framework); the run index records `port_signal_stop_db` alongside
`energy_stop_db`.

## DD-112 — One scattering-result contract; Touchstone/scikit-rf export

**Status:** Decided 2026-08-02 (session 143); shipped same session.

**Problem.**  `run()` returned an in-RAM `ScatteringTDResult` or a
store-backed `Project` reader with similar-but-diverging surfaces
(dB yes / phase no; run settings not readable back; custom `f_axis`
recompute store-only), and there was no industry-format export at all.

**Decision.**  `magnelio/analysis/result_interface.py` pins the
contract: a `ScatteringResult` Protocol, a `RunSettings` dataclass
(f_max/f_min/n_freq/dt/n_actual_steps/energy_stop_db/…), and
`ScatteringResultMixin` providing `phase()`, `plot_s()` and the
export delegates.  Both implementations satisfy it;
`tests/integration/test_result_contract.py` runs identical assertions
over both (measured note: the in-RAM and streamed execution paths are
not bit-identical — max |dV| ≈ 7e-15 on the same setup — so
cross-checks allow eps-level run divergence).  Interop:
`SParameterResult.to_touchstone()` (.sNp; Touchstone ports =
channels, mapping in the comment header; hard error when any channel
was never excited — no silent padding) and `.to_skrf()`
(`skrf.Network`, optional `magnelio[interop]` extra, lazy import).
The completeness rule was superseded by DD-184: an export covers the
excited channels, with the rest matched.

## DD-113 — Geometry verbs: CSG operators + chainable methods

**Status:** Decided 2026-08-02 (session 143); shipped same session.

**Problem.**  CSG verbs were CamelCase classes (`Difference(a, b)`)
while transform/modifier verbs were snake_case free functions
(`translate(shape, v)`) — two conventions for the same kind of
operation, and the free functions forced extra imports in every
script.

**Decision.**  A `ShapeOps` mixin (geometry/_shape_ops.py), inherited
by every shape *including the private transform/modification
wrappers* (which is what makes chaining work): operators `a + b` /
`a - b` / `a & b` build `Union`/`Difference`/`Intersection` (the
classes remain as the explicit spelling and result types; Group
operands keep their descriptive rejection), and chainable methods
`.translated/.rotated/.scaled/.chamfered/.filleted/.extruded/
.revolved/.swept/.lofted` delegate to the implementations, which left
the public API.  Ergonomics decided alongside: every `axis=` accepts
a letter or any 3-vector (`geometry/_axes.normalize_axis`); a
negative primitive height extrudes along −axis; port bboxes are two
opposite corner points in the face's tangential frame (aligned with
the DD-104 monitor corner convention — beware: the old symmetric
spelling `((-r, r), (-r, r))` reads as two identical corners and now
raises "degenerate").

## DD-114 — Port-signal stop criterion on by default ("auto")

**Status:** Decided 2026-08-03 (session 144, developer sign-off);
shipped same session.

**Problem.**  The v0.1.0 example acceptance runs exposed a trap in
the DD-096 termination design: `port_signal_stop_db` was opt-in, and
`energy_stop_db` — the only default criterion — can *never* fire on a
shielded lossless structure, because the complement absorber leaves
the TM-cut-off cavity tower (zero tangential E, invisible from any
port plane) exactly neutral and its stored energy plateaus.  Four of
the five examples therefore ran unbounded (observed: 1.6 M steps and
climbing, with the exact-DTBC history convolution adding O(n) work
per step).  Any user modelling a closed lossless device with default
settings would hit the same silent infinite run.

**Decision.**  `AnalysisScatteringTD.run(port_signal_stop_db="auto")`
is the new default: it resolves to 60 dB (the DD-096 verification
value, also used by every example) when at least one modal port is
present, and to disabled on lumped-only runs — the criterion watches
the modal |V| envelope, and the solver rejects it without one.
`None` still disables it explicitly; an explicit float is forwarded
unchanged (including the lumped-only hard error, which now only an
explicit value can reach).  The band pipeline ignores the criterion
as before.

**Arming guard (measured-mandatory).**  Defaulting the raw DD-096
criterion broke 16 integration tests: the |V| envelope transiently
sits far below the incident peak in the quiet gap between the
excitation leaving the driven port and the (attenuated) response
reaching the far ports, so the criterion fired mid-transit
(`test_lossy_line_sigma_m_gamma`: α error 0.012 → 5.66).  The
criterion therefore only *arms* at the auto-sized step estimate
(``2·t0 + 25 diagonal transits``, the pre-DD-070 bounded-run size;
solver field ``port_signal_min_steps``, set by the analyses on both
the in-RAM and the streamed path).  Consequence: every run that
previously terminated on the energy criterion before the estimate
stops bit-identically — the signal criterion is a pure safety net
for the plateau runs that formerly never ended (it also protects
explicit `port_signal_stop_db` users from the same transit trap; the
resume/run-longer path stays energy-only, DD-096 follow-up note).
All 16 failures pass again with the guard; full suites green.

**Consequences.**  The examples, README quickstart and notebooks are
back to pure-default `run()` calls.  `RunSettings` and the project
store record the *resolved* float (or None), never the "auto"
string.  The resume/run-longer path keeps its explicit signature
(the DD-096 follow-up note stands).

## DD-115 — Ready-to-open ParaView sessions from the project store

**Status:** Decided 2026-08-04 (session 150); shipped same session.

**Problem.**  The store's ParaView surface was raw material, not a
result: `geometry.stl` collapsed all solids into one unnamed,
uncoloured triangle soup; `fields.xdmf` loaded but left the user to
hand-build every pipeline (cell→point conversion, slices, glyphs);
`FieldFrequencyMonitor` had *no* working ParaView path at all (the
`write_xdmf_xml` frequency branch was dead code that never matched
the actual `fields_freq.h5` layout — cell-centre axes, complex
`(nf, nx, ny, nz)` bins); and naive glyph scaling is unusable on FIT
results because edge singularities produce a few huge vectors that
dictate the arrow scale and colour range.

**Decision.**  A three-layer exporter in `io/paraview.py`, riding on
the already-declared (previously unused) `vtk` dependency:

1. **Geometry** — `export_vtm` tessellates each solid into its own
   named block of a `geometry.vtm` multiblock (OCC per-face
   triangulation, deflection 0.2 % of each solid's bbox diagonal),
   with a `MaterialIndex` cell array into a deterministic material
   table (colours from `post/_colors.material_color`, the 3D-viewer
   palette).  Replaces `geometry.stl` in the store (`export_stl`
   stays as API); `geometry.json` gains a schema-additive `names`
   list and `_LoadedShape` a `name`, so loaded projects keep block
   identity.
2. **Monitors** — field-time monitors stay on XDMF over `results.h5`
   (no data duplication; one descriptor per monitor under
   `runs/<run>/paraview/` for clean per-monitor pipelines).
   Frequency monitors become a `.vtr`-per-frequency series plus a
   `.pvd` collection (frequency as the ParaView time axis), with
   scalar `<comp>_re/_im`, vector `E_re/E_im` (glyphs at phase 0)
   and complex-magnitude `|E|` cell arrays; node axes are recovered
   exactly by re-resolving the stored corners against the stored
   grid.  The dead `write_xdmf_xml` was deleted.
3. **Session** — run close generates `paraview_open.py`
   (`paraview.simple`; open via `paraview --script=…`) and, when
   `pvpython` is on the PATH, bakes a double-clickable
   `paraview.pvsm`.  The pre-built pipeline per monitor:
   `CellDatatoPointData` → Calculator clipping the field vector to a
   cap (98th percentile of |v|, estimated from sampled steps at
   export time) → three slice planes through the monitor centre
   (default visible: normal to the shortest extent; planar monitors
   glyph directly) → arrow glyphs (uniform spatial distribution,
   scaled so a cap-length vector spans 1/25 of the monitor
   diagonal) → a geometry `Clip` whose plane is proxy-linked to the
   slice plane, so dragging one drags the other.  Colour ranges are
   pinned to `[0, cap]` — singularities can neither stretch arrows
   nor wash out the colormap.  Only the first monitor's default
   slice is shown; all other sources are created hidden and unread,
   so many-monitor projects stay RAM-cheap until toggled visible.
   `Project.export_paraview()` regenerates with different options;
   `MAGNELIO_PVSM_BAKE=0` suppresses the bake (test-suite pin).
   Everything is best-effort: a viz failure at run close warns,
   never invalidates the run.

**Found along the way (fixed).**  Since GPU became the production
default (DD-090), *every* field/flux monitor crashed on the cupy
backend: the DD-085 interpolation and the flux weights mixed NumPy
operands into device arrays (`TypeError` at the first recorded
step).  `_interp_to_cell_centres` now interpolates on the device and
transfers only region-sized results; `FluxTimeMonitor` moves its
weight planes to the device once.  `WallLossMonitor` has the same
defect and remains open — see known-bugs.md (KB-006).

**Glyph length law (measured correction, same session).**  The first
implementation used the clip cap as *both* the saturation point and
the length reference (`scale = l_ref / cap`), which made the glyphs
unusable on the first real fixture — measured on a slotline beam
coupler, whole-domain H monitor, 561 050 cells: 25 % of cells are
exactly zero and 54 % sit below 5 % of the p98 (PEC interior and quiet
volume), so the p98 lands next to the maximum and the *median* arrow
came out 1.1 mm on a 200 mm structure — invisible, while the maximum
ran to 82 mm.  The developer independently arrived at ~1e8 as a usable
factor; the fitted reference gives 8.6e7, confirming the diagnosis.
The two roles are now separate: the cap still saturates outliers, and
a length **exponent** is fitted per monitor so that the typical
field-carrying magnitude (p60 over cells above 1 % of the peak) is
drawn at 45 % of the full length, while the cap maps to exactly 1.
The Calculator emits a dimensionless 0…1 direction array, so the glyph
`ScaleFactor` is the longest arrow in metres — interpretable, and
inside the range ParaView's slider offers (the previous magnitude-
derived factor of 2.6e7 sat far outside it).  Result on that fixture:
median 1.1 → 4.4 mm, p75 7.6 → 13.8 mm, maximum 82 → 32 mm (bounded).
A distribution needing no compression fits exponent 1 (pure linear),
so the law degrades gracefully.  Verified in ParaView itself: both
direction arrays evaluate to magnitude range exactly 0…1.

**Every arrow sits on an even lattice; the computational grid is not
used for placement.**  ParaView's spatial glyph seeding snaps each seed
onto the nearest *mesh* point, so arrows expose the computational grid:
on the beam coupler the z spacing runs from 0.44 mm in the
geometry-refined slot region to 18.7 mm in the wavelength-sized
waveguide (a factor 43; 30 coarse cells cover 65 % of the length),
which reads as arrows crowding into isolated x-y planes with voids
between them.  Developer's verdict after seeing it: numerically honest,
practically unusable, and not what commercial tools show.  Every
monitor is therefore resampled (`ResampleToImage`) onto a lattice with
**one spacing for all axes** and glyphed with *All Points* — a fixed
sample count per axis would make the spacing directional on an
elongated region, reintroducing the very bias being removed.  A
correctly set up simulation has a grid fine enough that interpolating
is legitimate, and the user is after the field, not the discretisation
that produced it.  No mesh-true branch is kept: it was offered and
declined.
Two lattices per 3D monitor, because the two views need different
densities.  The **section** lattice is sized for arrows per unit area
(~2000 in the largest section, the one the default cut shows) and feeds
the three slice planes; the **volume** lattice is coarser (~8000 points
over the region) because glyphing every point of the section lattice in
3D would bury the field under its own arrows.  Planar monitors resample
to a single layer — degeneracy is decided by cell count, not by
thickness, so a one-cell-thick plane does not get several layers
through a thickness carrying no second sample.  Measured on the beam
coupler: section lattice 12x14x154 at 5.2-5.4 mm giving 2156 / 1848 /
168 arrows on the three cuts, volume lattice 9x10x109 at 7.4-7.6 mm
giving 1711 arrows after the threshold.
Resampling happens on the **raw** field, ahead of the direction
calculators, so the non-linear length map is applied to interpolated
field values rather than the interpolation being applied to compressed
ones.  Ahead of the volume glyphs sits a Threshold keeping only cells
above 2 % of the cap — 297 k of 561 k cells on that fixture, i.e.
exactly the PEC interior and quiet volume that would otherwise bury the
field in short arrows.  Thresholding needs a scalar (ParaView's
Threshold offers no vector-magnitude mode), hence a scalar
``<arr>_mag`` Calculator per field array.  Slice glyphs keep every
lattice point: on a single cut the empty regions read as information.
Volume sets are created hidden — a slice stays the cheaper first look.
**Frequency glyphs carry both phases.**  The first implementation
offered only the real part; where the field is mostly imaginary that
shows nearly nothing.  Real and imaginary part (phase 0 and −90°) each
get their own Calculator + glyph set sharing the cap and exponent from
the complex magnitude; the imaginary set is created hidden, one click
from visible.  Phase animation is not available — the `.pvd` time axis
is already spent on frequency.

**Consequences.**  `geometry.stl` is no longer written (alpha break;
store docs updated, round-trip test moved to `.vtm`).  `vtk` added
to `environment.yml` (was already in pyproject/recipe).  Gates:
`tests/unit/test_paraview_export.py` (material table, VTM blocks,
slice specs, script config round-trip),
`tests/integration/test_paraview_session.py` (artefacts on a real
streamed run, `.vtr` ↔ `fields_freq.h5` value/ordering parity,
regeneration, an actual `pvpython` state bake, and warn-not-raise on
export failure).

## DD-116 — Documentation portal: four pillars, tutorials from sphinx-gallery

**Status:** Decided 2026-08-07; structure skeleton shipped same day.
Tutorial content is deliberately deferred to its own work package.

**Problem.**  The Sphinx site was framed as "Scientific
Documentation" yet carried the API reference; the components
reference was a single unnavigable automodule stream (15 namespaces
on one page, no sidebar entries); and there was no user-facing
tutorial tier at all.

**Decision.**

1. **One portal** ("Magnelio Documentation"), four pillars:
   Tutorials / API reference / Numerical methods / Bibliography.
   The former scientific documentation lives on unchanged as the
   Numerical-methods pillar.  Separate user/scientific sites were
   rejected: duplicate infrastructure, shared bibliography, brittle
   cross-references.
2. **API reference: one page per component namespace**
   (`docs/api/<ns>.md`, 15 pages, workflow order geometry → … →
   constants).  Each namespace gets its own sidebar entry and the
   pydata theme's per-page member list — the one-page automodule
   stream is gone.
3. **Tutorials are generated from runnable scripts** by
   sphinx-gallery (the Scientific-Python standard:
   scipy/scikit-learn/matplotlib).  Source of truth =
   `examples/tutorials/*.py` (public API only, per the `examples/`
   policy); build products = HTML pages plus `.ipynb` downloads
   under `docs/tutorials/` (gitignored).  Scripts named `plot_*`
   execute at build time (gallery-cached; tutorials must be
   budgeted to run fast — coarse meshes are a feature, users copy
   fast examples), other names render without execution.
   `sphinx-gallery` added to the pyproject `[docs]` extra and to
   `environment.yml`.  The placeholder
   `examples/tutorials/plot_01_first_simulation.py` validates the
   pipeline end to end.

**Found along the way (open).**  The per-page autodoc sweep
surfaces ~11 pre-existing rST defects in public docstrings
(undefined `|V|`/`|S11|` substitution references, indentation
slips, ambiguous cross-references on the `io`/`plot` pages) —
tracked in STATUS.md, to be cleaned when those docstrings are next
touched.

## DD-117 — Thin core + domain namespaces (supersedes the DD-108 two-tier framing)

**Status:** Decided 2026-08-07 (API review session with the
developer); shipped same session.  Hard API break (permitted:
MAJOR = 0, no external users).

**Problem.**  Three developer-reported defects of the DD-108 surface:
(a) 30 flat top-level names scatter prefix-free across the alphabet —
geometry primitives and monitors do not cluster in autocompletion or
in the rendered API page; (b) the one-home rule emptied the domain
doc pages (`magnelio.geometry` documented only `ThinWire` +
`GeometryOverlapError`); (c) context-free top-level names are
unintelligible — `SeriesRLC`/`ParallelRLC` read as standalone
geometry-carrying components when they are port-attached companion
models (`PortLumped(..., element=...)`).  The "high-level vs
component" axis itself forced every placement to be re-litigated.

**Decision.**  One axis — the domain (SciPy-style):

1. **Core** (`magnelio`, 10 names, pinned as `EXPECTED_CORE` in
   `check_api_surface.py`): `GeometryModel`, `Material`,
   `Mesh`/`MeshControl`, `BoundaryConditions`,
   `AnalysisScatteringTD`/`AnalysisEigenmode`,
   `open_project`/`resume`, `__version__`.  No re-import shims for
   demoted names — a clean break.
2. **Domain namespaces** carry everything else, one documented home
   per name (rule and tooling from DD-108 unchanged): primitives/
   CSG/`Curve` → `magnelio.geometry`; declarative ports →
   `magnelio.ports`; `SeriesRLC`/`ParallelRLC` → `magnelio.circuit`;
   monitors → `magnelio.monitors`.  Class names are NOT shortened to
   their module (`ports.PortWaveguide`, not `ports.Waveguide`):
   self-describing on direct import, and the module split already
   provides the clustering.
3. **Monitors renamed to the DD-110 noun-first pattern** — the last
   naming exception falls: `MonitorFieldTime`,
   `MonitorFieldFrequency`, `MonitorFluxTime`, `MonitorWallLoss`.
   The store type tags and resume-recipe type strings follow the new
   names (schema vocabulary = class names; pre-release stores with
   monitors do not rehydrate — regenerate by re-running).
4. **Plumbing hidden generously** (dropped from curated `__all__`,
   imports kept — soft-private, checker-clean): port builders,
   `PortOperator*`, `Port` base, `PortPlane`, `PortSignalRecorder`,
   `LevelResult`, `solve_modes_refined`, `MonitorRegion`,
   `destaggered_power_waves`, `ScatteringResultMixin`.  Kept public
   (usage-verified): `PortSpec*` + conductor specs + `Mode`/
   `ModeType` + reports (the custom-setup tier), `GridLines`
   (parameter type of `Mesh.from_grid`), `BoxFace` (notebook API).
   No module underscore-renames — revisit only if soft-privacy is
   abused.

Blast radius (measured): 59 in-repo files + 26 private-workspace
scripts rewritten; migration was mechanical import-splitting.
Docs: `docs/api/highlevel.md` → `core.md`; domain pages fill
automatically from the curated `__all__`.  Gates:
`check_api_surface.py` (incl. the new core pin), full suite, ruff,
sphinx build, `check_imports.py` over the private script dirs.

## DD-118 — Magnetic-wall dual booking: PMC window edges own the full boundary cell

**Status:** Decided + shipped 2026-08-07 (found while grounding the
first tutorial: the parallel-plate `z_line` refused to be exact).

**Problem.**  The natural magnetic wall of the staggered grid lies
half the outer dual cell BEYOND the outermost grid line — DD-103
places it on the requested bbox face by pulling that line in by d/3;
a post-meshing declaration (or `from_grid`) leaves it half a cell
outside.  Three physical-bookkeeping quadratures still assumed
"wall ON the outermost line" at tangential PMC window edges:

1. the TEM/QTEM capacitance integrals
   (`tem_laplace._tangential_boundary_factor`, ×½ at bbox edges) —
   on the parallel plate the reported `z_line` was exactly
   `η0·b/(a − 2d/3)` instead of `η0·b/a`: an O(h) bias of `2/(3·Nx)`
   (+2.1 % at Nx = 32) although the discrete TEM mode itself is exact;
2. the DD-078 physical-power Poynting patches
   (`operator._patch_duals`, half-cell end duals) — a "1 W" injection
   physically carried `w_eff/w` watts;
3. `MonitorFluxTime`'s boundary-h ×½ weights — the flux through the
   two outer half-cell strips was dropped (energy ratio 0.909 on the
   Nx = 10 gate fixture).

(1) and (2) cancelled in the per-1W gates — both booked the same
fictional width — so the bias was invisible until `z_line` was held
against the analytic value.  The numerical TE/TM eigensolver needs no
change: its Neumann closure extends the half cell implicitly (TE10
cut-offs were already second-order correct on both declaration paths).

**Decision.**  One wall-position convention for every physical
quadrature: at a window end that coincides with a declared PMC bbox
face, the end dual extends to the wall — the FULL boundary cell
(factor/weight 1.0 instead of 0.5).  Shared predicate
`port_plane.magnetic_window_ends(plane, grid, boundary_conditions)`;
threaded as `boundary_conditions=` into `solve_tem_laplace` /
`solve_qtem_laplace` (all factory call sites pass
`mesh.boundary_conditions`), as `magnetic_patch_ends=` into
`PortOperatorModal` (factory-computed), and BC-aware weights in
`MonitorFluxTime.attach`.  The mode-normalisation metric (raw 3D
`M_ε`) is deliberately untouched: injected profiles, DTBC chains and
the certified port floors are bit-unchanged; only the physical
bookkeeping (`z_line`/`ε_eff`, the √W amplitude scales, flux watts)
moved.

**Measured.**  Parallel plate a×b, PMC sides, on-model declaration:
`z_line = η0·b/a` to 3e-12 relative on EVERY resolution incl. the
7×4×14 default mesh (was O(h)); post-meshing/`from_grid` path reports
`η0·b/(a + d)` — the impedance of the structure actually simulated
(wall half a cell outside; confirmed independently by the TE10
cut-off sitting at `c/2(a+d)`); flux-monitor energy ratio
0.909 → 1.000; |Ey|,|Hx| per 1 W match the analytic line values with
the effective width.  Regression:
`tests/unit/test_boundary_closure.py::TestMagneticWallCapacitance`
(both declaration paths, machine-precision gates); the three
integration gates that had encoded the requested-width fiction now
assert `w_eff = w + dx` (`test_solve_ports.py`,
`test_declarative_ports.py`, `test_physical_states.py`).

**Consequence for users.**  On lines-kept meshes the simulated line
is one outer cell wider than the requested geometry; `z_line` now
says so truthfully instead of echoing the request.  The mesher's
on-model path (the production default) simulates exactly the
requested faces and now reports exact TEM impedances on uniform
cross-sections.

## DD-119 — Namespace renames `geo`/`plots` + the standard example style

**Status:** Decided + shipped 2026-08-08 (developer style review of
the first two tutorials).

**Decision 1 — renames.**  `magnelio.geometry` → **`magnelio.geo`**
(used constantly when building models; the short name is the point,
following the scipy-style precedent of heavily-used namespaces) and
`magnelio.plot` → **`magnelio.plots`** (noun form, reads better in
the namespace roster next to `ports`/`constants`).  Clean break as in
DD-117: package directories renamed, no aliases, every dotted
reference migrated (src, tests, validation, benchmarks, examples,
docs pages, spec/STATUS, private workspace dirs — historical DD
entries untouched).

**Decision 2 — standard example style** (developer-authored in
`userscripts/plot_02_coax_line.ipynb`, adopted for `examples/`
including tutorials).  Rationale: the flat-import style forced the
reader to enumerate every primitive at the top of the script; the
namespace style defers that choice to the call site and gives a
constant three-line header:

```python
import magnelio as mio
from magnelio import geo, plots, ports   # domains as needed
from magnelio.constants import *         # curated 4-name __all__
```

Core names are used as ``mio.GeometryModel``, ``mio.Mesh`` …; domain
classes as ``geo.Cylinder``, ``ports.PortWaveguide``,
``plots.plot_cross_section``.  The constants star import is
well-defined (curated ``__all__``: C0/EPS0/MU0/ETA0) and covered by a
ruff per-file ignore (`examples/**`: F403/F405) — it is example
style, not library style; library code keeps explicit imports.

Gates: full suite (1410 + 315), `check_api_surface.py`,
`check_imports.py` over the private dirs (936 imports resolve),
notebook AST scan, ruff, sphinx build with regenerated `api/geo` and
`api/plots` pages.

## DD-120 — Scale-robust geometry pipeline: automatic OCC unit scaling + relative tolerances

**Status:** Decided + shipped 2026-08-08 (meter → optics scale
initiative; developer decisions: target = optics incl. sub-µm,
mechanism = automatic bbox-derived scaling, scope = full remediation).

**Problem.**  The public unit contract is SI meters, but the geometry
and mesh layers carried absolute meter-valued constants that only make
sense in the mm regime.  Fatal at optics scale (~µm coordinates):
`min_feature_gap = 1e-6` clustered every transverse critical plane of
a micron structure into one position (silent geometry annihilation),
and the OCC kernel's fixed `Precision::Confusion()` = 1e-7 model units
rejected sub-100-nm features outright (DD-062) while giving µm-scale
Booleans a ~10 % relative tolerance.  Silently degrading: the
tessellation deflection floor `1e-7` made the chordal error 10× the
cell below ~10 µm cells (conformal material matrices corrupted without
warning); `point_in_shape` (1e-7 m), `compute_edge_pec_fractions`
(1e-8 m) and the overlap tolerance (1e-18 **m³**, scaling as L³)
followed the same pattern.  OCC itself is unit-agnostic and its
Confusion constant is not configurable — the industry-standard fix is
coordinate scaling, exactly what commercial suites' "model unit"
provides for their CAD kernels (their field solvers are scale-free,
as is ours).

**Decision — WP-A: automatic internal unit scaling at the OCC
boundary.**  One **power-of-two** scale factor `s` per model, a pure
function of the shape set: `geo/_scaling.py::model_scale` computes it
OCC-free from conservative *analytic* bounding boxes (`_analytic_bbox()`
on every shape class; transform algebra for rotations, generous pads
for lofts/sweeps/splines — only the order of magnitude matters, and
the OCC-free computation breaks the bbox↔OCC circularity).  `s` is
threaded explicitly as a `scale=` keyword through `_occ_shape(scale)`
(per-instance cache now keyed by scale — no invalidation logic, a
changed scale is a different key) and every backend entry point.
Contract: **meters at every function boundary**, scaled units strictly
inside the backend; inputs ×s in bulk numpy, outputs ÷s / ÷s² / ÷s³;
dimensionless outputs pass through.  `compute_face_material_areas` /
`batch_cross_sections` convert at the *leaf* (`_PlanarSectionEngine` /
`cross_section_polygons` return meters), so the DD-101/102 accounting
machinery, its cache keys and the DD-106 degenerate handling are
structurally untouched; the spawn-pool workers receive blobs
serialised at `s` and tasks carrying `s`, keeping the meter-in/
meter-out contract identical to the sequential path.  Identity band:
`s = 1` for model diagonals within [1e-3, 1e4] m — every existing
meter/mm model runs the bit-identical legacy path; outside, `s`
brings the diagonal to O(128) scaled units (target 2^7), lifting the
100-nm wall (effective feature limit `1e-7 / s` m; DD-062 finding 2
superseded for auto-scaled models — `_check_dimensions` now checks in
scaled units and reports the effective meter limit).  Power-of-two
scaling is IEEE-754-lossless, so coordinates round-trip bit-exactly.
Rejected alternatives: a user-facing `unit=` parameter (API addition
the solver does not need); `s` owned by `GeometryModel` (bare-shape
calls and plain shape lists exist before any model); module-global
state (hidden coupling).  Known limitation (documented, accepted): a
model with diagonal ≥ 1 mm keeps `s = 1`, so sub-100-nm features
inside such a model remain rejected — resolving nm features across a
mm domain is computationally absurd anyway.

**Decision — WP-B: the library's own absolute tolerances became
relative.**  `MeshControl.min_feature_gap` default `None` →
`1e-5 ×` bbox diagonal (`mesher.resolve_feature_gap`; the DD-058 CSG
float wiggle is *relative*, so the tolerance must be; resolved value
exposed as `mesh._resolved_feature_gap` for the sentinel I3).
Tessellation deflection: pure `h_min · 1e-2` (conformal areas) and
`h_min · 0.1` (cell-centre classification) — the absolute 1e-4 cap
and 1e-7 floor are gone from the callers; the OCC robustness floor
(`Standard_ConstructionError` below 1e-7) now lives in *scaled units*
inside `cross_section_polygons`, unreachable for auto-scaled models.
`compute_edge_pec_fractions` tolerance `1e-4 · h_min` (from
`compute_subcell_data`), thin-sheet probe `point_in_shape` tolerance
`1e-3 · min_cell_size`, port-factory RegionConductor `eps_tol`
`1e-9 · extent` (the relative form the lateral-wall matching already
used), overlap tolerance `None` → per-pair `1e-12 · min(AABB volume)`,
STL deflection default `1e-3 ×` bbox diagonal, meter-domain bbox
slacks `1e-12 · (1+|pos|)`.  Backend-internal absolute epsilons
(polygon dedup 1e-12/1e-10, section-engine seeds) deliberately stay
numeric: they act in scaled units, the O(100) regime they were tuned
for.  Audited safe (no change): the 1e-30 exact-zero guards in
`_polygon_clip`/area budgets and `stability.py` sit 12+ decades below
worst-case nm-scale face areas (~1e-16 m²) — pinned by
`tests/unit/test_scaling.py::TestNanoscaleGuards`.

**Measured.**  mm-identity gate (WG90 + RG-58 coax, full mesh arrays
incl. edge/face material, NaN-aware): **bit-identical** after every
work package (internal record
`investigations/dd120_scale_robustness/`).
`validation/scale_invariance_certificate.py`: |S| over the normalized
frequency axis at geometric scales 1×/1e-3×/1e-6× (frequencies
inverse) — parallel plate TEM ≤ 5.8e-11, TE10 ≤ 1.9e-7, coax TEM ≤
1.7e-7; worst 1.9e-7, pinned bound 1e-6 (the 1e-7 level is the
float32 solver default of DD-094, not a geometry artefact; micron
coax runs at s = 2^23, nano coax at s = 2^33).
`validation/fiber_micron_regression.py`: step-index fiber (core
4.5 µm / cladding 62.5 µm / coating 125 µm) meshes at s = 2^18 with
every ±r feature plane intact on both transverse axes, resolved
feature gap 4.4 nm, all four materials present — pre-DD-120 this
geometry was silently annihilated.  A 20-nm brick builds through the
scaled path (unit-gated); `mesher_stress_sentinel.py --fast` 11/11;
pool-vs-sequential at s = 2^19 bit-equal
(`TestSectionPoolAtScale`); full suite 1763 passed.  CSG stress
benchmark (1002 primitives, `benchmarks/profile_csg_scaling.py`):
73.1 s vs. 73.9 s pre-change — the bulk-numpy scaling costs nothing
measurable.

**Consequence for users.**  Nothing at the API: everything stays SI
meters (now stated as a "Units" section in the user docs).  Micron-
and sub-micron-scale models (THz, integrated optics, fiber
cross-sections) mesh and solve without workarounds; `min_feature_gap`
only needs touching to *opt out* of the relative default.  Solver-side
follow-up (out of scope here): at optical frequencies the float32
default of DD-094 bounds |S| reproducibility near 1e-7 — consider
`precision="double"` for tighter needs.

## DD-121 — Slice plots for 3D field data + normal-component encoding in vector plots

**Status:** Decided + shipped 2026-08-09 (gap surfaced by tutorial 05,
which had to hand-roll Yee interpolation with internals; developer
decisions: ⊙/⊗ markers instead of a background colour layer, single
magnitude colour scale with sign in the marker shape, existing 2D
plots included in the fix, `interact()` in scope).

**Problem.**  Three related holes in field plotting.  (1) Both field
monitors record 3D volumes but `plot()` raised `NotImplementedError`
for `ndim == 3` — no way to look at a volume recording.  (2)
`EigenmodeResult` had no plot API at all; tutorial 05 accessed
`mode.Ex` / `mesh.grid` and re-derived the staggered-edge averaging
inline, violating the examples policy.  (3) `plot_field_vector`
silently dropped the out-of-plane component everywhere it is used: a
field crossing the plotted plane at right angles rendered as an
*empty* plot, and the colour bar claimed "Field magnitude" while
showing the in-plane projection only.  TEM-dominated test cases had
masked this; hybrid port modes and eigenmode slices expose it.

**Decision 1 — plane-view resolver.**  `monitors/base.py` gains
`PlaneView` + `resolve_plane_view(region, normal, position)`: for a 2D
region the plane is the region itself (a given `normal` is validated),
for a 3D region `normal=`/`position=` select the slice, snapped to the
nearest cell-centre plane — the same normal-plus-offset convention as
`plot_cross_section` and the geometry overlay.  Both monitors'
`plot()` **and** `interact()` route through it (slider = time/
frequency at a fixed plane); titles carry the plane (e.g. `y=0.667
mm`).  The duplicated `_free_axes`/`_make_overlay` pair collapsed into
it.

**Decision 2 — normal component in the shared vector renderer.**
`plot_field_vector` accepts an optional `w`.  Arrow direction and
relative length stay in-plane, arrow *colour* becomes the full 3D
magnitude on an explicit 0-anchored norm, and with `w` the
auto-scale also references the full-magnitude peak (in-plane peak
without `w`), so arrow length over colour reads as the out-of-plane
tilt everywhere.  The in-plane reference would amplify honest small
residues to full-length arrows — measured on the sphere quintet's
H slice, where the half-cell offset of the cell-centre slice plane
from the symmetry plane leaves a genuine ~0.5 %-energy in-plane
residue (checked against the raw staggered DOFs; it grows ∝ x² off
the plane, so it is physics, not interpolation) that used to bury
the pattern's node lines under visible arrows.  The monitors'
`interact()` fixed-scale precompute follows the same reference.  Grid points whose
vector tilts out of the plane by more than ~72° (`|w| >= 3x` the
in-plane part, at magnitude above `max(threshold, 0.02)` of the
peak) are drawn as filled circles on the same colour scale with a ⊙
(towards `+axis`) or ⊗ (towards `-axis`) glyph and a small legend —
deliberately axis-referenced, not "towards the viewer", which would
depend on axis handedness and `flip`.  The criterion is *local*
tilt, not a comparison against the global in-plane peak: quiver
auto-scales arrows to the in-plane maximum, so a slice pierced
almost at right angles everywhere (e.g. the H field of a TM
eigenmode on a meridional slice, in-plane residue ~6 %) would
otherwise still render as a full-length arrow picture — measured on
the sphere quintet before the criterion was fixed.  Rationale vs.
the background-colour alternative: markers keep geometry overlays
visible and keep one colour scale; the sign lives in the marker
shape (a 2D glyph has two visual channels for three vector
components plus sign — one channel must be the shape).  Without `w`
the behaviour is unchanged except the honest colour-bar label
"In-plane field magnitude".  Monitors pass `w` whenever the normal
component was recorded; port `ModeReport.plot()` passes no `w`
(`DiscreteMode` stores transverse profiles only — there is no
longitudinal profile to pass).

**Decision 3 — `EigenmodeResult.plot()`.**  Same signature family
(`mode=`, `component=`, `normal=`, `position=`, `plot_type=`,
`geometry=` overlay): interpolates only the requested slab via
`_interp_to_cell_centres` (the eigenmode `FieldState` holds FIT grid
quantities, `h = (1/ω)·M_μ⁻¹·C·e`, so the monitor converter applies
verbatim), labels amplitudes "arb. units" (normalisation
`eᵀ M_ε e = 1`).  Tutorial 05 now plots modes through this API; the
canonical ⊙/⊗ demonstration (developer-suggested) is quintet mode 3
on the slice where its E field lies fully in-plane: the E panel is
all arrows, the H panel — everywhere perpendicular to E — is all
markers with alternating ⊙/⊗ sectors.  Verified by an
energy-weighted normal-fraction scan over all modes and slice
normals (E/H normal fractions 0.00/0.99-style splits).  Tutorial
prose must stay sign-agnostic about which region is ⊙ vs. ⊗
(eigenvector sign is a gauge; it flipped between otherwise identical
runs), and every figure must be generated with the tutorial's exact
`n_modes` — the degenerate-cluster basis (orientation *and* mixing)
changes with `n_modes` (1 vs. 4 vs. 8 gave three different mode-0/3
orientations).

**Decision 6 — conductor-aware destaggering in port mode plots.**
Mode profiles carry exact `0.0` on every non-DOF edge; the plain
two-point average onto cell centres therefore halved the magnitude
and rotated the direction of every vector whose stencil touches a
conductor — measured on the tutorial coax (78 of 460 cells: median
angle error 10.3°, magnitude down to 0.39x of the 1/r value; the
"colour speckle at the inner conductor" the developer spotted
visually).  `_avg_nonzero` in `mode_report.py` now averages only the
live contributors: touched cells improve to median 5.4° / magnitude
>= 0.78x (the remainder is genuine conformal/staircase
discreteness); interior cells (median 0.24°, |E|·r flat to ±6 %) are
bit-unchanged.  Plot-side only — `_interp_to_cell_centres` (DD-085)
defines recorded *monitor data* and stays strictly two-point.

**Decision 4 — transparent materials as outlines in cross-sections.**
`plot_cross_section` used to *skip* fully transparent (air/vacuum)
shapes — correct for filled inspection plots of visible parts, but it
made the geometry overlay useless for the most common eigenmode
geometry, a cavity carved into a conducting background (the air
shape's boundary *is* the wall; tutorial 05 had to hand-draw the
circle).  Transparent shapes now render as a dashed black outline
with a white under-stroke (`patheffects`), readable on any field
colour map; `outline_transparent=False` restores the skip, and
`visible=False` still hides a shape unconditionally.  Tutorial 05
passes `geometry=model` instead of drawing the wall by hand.
Related trap recorded there: the overall sign of an eigenvector is
arbitrary, so tutorial prose must not pin which region carries ⊙
vs. ⊗ (it flipped between two otherwise identical runs).

**Decision 5 — geometry overlay on port mode plots.**
`ModeReport.plot()` accepts `geometry=` like the monitor and
eigenmode plots (tutorials 02/04 pass their model).  Two port-plane
subtleties, both pinned by
`test_solve_ports.py::test_geometry_overlay_wiring`: (1) the
cross-section is sliced **half a boundary cell inward**
(`coordinate + inward_sign · normal_dx/2`) — exactly on the bbox
face the OCC section is tangent to the solids' end faces
(ill-defined), and a port requires an extruded cross-section there
anyway; (2) three of the six faces order their local `(u, v)` axes
*descending* (`u x v` points inward: X_MAX, Y_MIN, Z_MAX), while the
cross-section renderer slices in ascending order —
`CrossSectionOverlay` gained a `swap_axes` flag that XORs with the
plot's `flip`.

**Gates.**  Unit suite 1471 passed (new: `test_eigenmode_plot.py`
with an exact uniform-field recovery check, plane-view resolution and
3D-slice-vs-data tests, marker/colour-norm tests, air-outline
cross-section tests); integration 317 passed incl. the port-overlay
wiring and `_avg_nonzero` tests (the 4 `test_tile_skip_solver`
bit-identity tests need the documented `CUPY_ACCELERATORS=""` when
the env binary is called directly); `check_imports.py` over the
private dirs (947 imports resolve); tutorials 02, 04 and 05 executed
end-to-end on the public API; ruff clean (plus `extend-exclude` for
the generated `docs/tutorials/` gallery output, which had drifted
into the lint scope).

## DD-122 — Port-signal stall watchdog + runtime cap for unbounded runs

**Status:** Decided + shipped 2026-08-10 (design round with the
developer: watchdog *and* cap, arming floor −40 dB, full stop-reason
bookkeeping; cap factor raised 10×→40× after quantifying the implied
Q-ceiling for resonant structures).  Closes KB-008.

**Problem.**  `port_signal_stop_db="auto"` (−60 dB) presumes the port
|V| envelope decays exponentially — then any threshold is reached in
finite time.  Band-edge (cut-off) ring-down decays *algebraically*
(Bessel-tail, vanishing group velocity), so its envelope plateaus; on
the WR-90 magic tee the E-arm drive plateaus near −56 dB and the
default unbounded run marched indefinitely (>40 000 extra steps, no
envelope movement — KB-008).  `energy_stop_db` is defeated by the same
content (the DD-096 motivation).  Truncating *at* the plateau is
harmless for S-parameters: the residual is at the plateau level and
`taper_signals` bounds its spectral leakage.

**Decision 1 — runtime cap `max_time_steps`.**  New knob on
`run()`/`resume()` (and the solver): unbounded runs get an absolute
step bound; hitting it stops with a `RuntimeWarning` and
`stop_reason="runtime_cap"` — the industry-standard backstop.
`"auto"` (default) = 40× the auto step estimate (≈10³ diagonal
transits).  A transit-based cap is implicitly a Q-ceiling,
Q ≲ 23·C·L_diag/λ: C = 40 accommodates loaded Q up to ~900·(size/λ)
before a 60-dB ring-down is cut short — covering realistic
narrow-band filters (C = 10 would already truncate compact Q_L ≈ 230).
`None` removes the cap (march forever — the pre-DD-122 contract);
an explicit `total_time_steps` wins (cap + watchdog off).  Each
launch/resume segment gets its own cap budget past its start step.

**Decision 2 — stall watchdog.**  `_SignalStallDetector`
(`solver/fit_td.py`): armed once the envelope is ≤ −40 dB below peak,
it least-squares-fits the envelope samples the criterion already
polls over a window spanning half the transit estimate of physical
time, and extrapolates the step at which the threshold would be
crossed.  Beyond the cap (or slope ≥ 0) → stop *now* with
`stop_reason="port_signal_stall"`, a `RuntimeWarning` naming the
achieved level, and the plateau accepted as the effective floor — by
construction the same outcome the cap would deliver, minus the wasted
marching.  The projection form avoids a hidden Q-limit that a fixed
"flatness" threshold would impose (a Q ≈ 10⁴ resonator decays only
~0.1 dB per window yet reaches −60 dB in finite time); window reset on
new peaks / recovery above the arming floor keeps regimes unmixed.
Active only on capped unbounded runs.

**Decision 3 — stop-reason bookkeeping.**  Every `run()` exit sets
`_stop_reason` ∈ {"steps", "energy", "port_signal",
"port_signal_stall", "runtime_cap", "aborted"} and
`_final_signal_db` (|V| level below peak at the stop; 0.0 legal when
the envelope never sampled below peak).  The store books both into
the run index (schema-additive) and the reader surfaces them via
`proj.runs[...]` and `RunSettings.stop_reason` /
`.final_port_signal_db`; the in-RAM path reports via the warning only
(deliberate scope cut).  The resume guard "done run already reached
energy_stop_db" now consults `stop_reason`: cap-/stall-truncated runs
are "done" *without* having reached their criterion, and a bare
`resume()` is their intended escape hatch (legacy projects without
the field keep the historical guard).

**Validation.**  `validation/wr90_tee_signal_stall_certificate.py`:
magic-tee E-arm with pure defaults stops at step 9101
(`port_signal_stall`, plateau −55.6 dB, exactly one warning, 7.9 s
GPU) instead of marching indefinitely; max |ΔS| = 4.3e-4 against the
`port_signal_stop_db=50` tutorial-06 reference across the design
band.  Unit: detector verdicts on synthetic plateau/exponential/
slow-exponential envelopes + solver wiring via a scripted fake port
(`tests/unit/test_stall_detector.py`).  Integration: cap truncation +
bare-resume completion, index/settings bookkeeping
(`tests/integration/test_project_scattering.py`).

---

## DD-123 — Declarative passive lumped elements (`circuit.LumpedElement`)

**Date:** 2026-08-10 (session: tutorial-10 design round; closes the
deferral opened in [[DD-077]] "top-level export deferred until the
operator makes them usable in a run").
**Status:** Accepted — implemented, certificate PASSED.

**Problem.**  Every discrete impedance was forcibly a *port*: the
general `LumpedElementOperator` (DD-077/078 unification) existed, but
its only consumer was `PortOperatorLumped` — no way to drop a passive
load (e.g. a Wilkinson isolation resistor) into a model without it
acquiring an S-matrix column, excitation capability and recording.

**Decision (developer sign-off: `circuit` namespace + pure sink).**
`circuit.LumpedElement(label, start, end, element=SeriesRLC/ParallelRLC)`
declared via `GeometryModel.add_element()` (deliberately *not*
`add_port` — it is a component, not a terminal).  Carried by the mesh
(`Mesh.elements`, `with_elements()` for the `from_grid` path) exactly
like DD-109 ports; `AnalysisScatteringTD(elements=...)` overrides.
Builder `build_lumped_element` (shared `_snap_edge_chain` with the
port builder) returns a plain `LumpedElementOperator` with `Z0 = 0`,
fresh-deep-copied companion per excitation.  The operator joins the
solver's port list (unified `update_e` hook, label-keyed checkpoint
state → bit-exact resume for free) but stays out of the recorder,
S-matrix, `excited=` validation and the port-signal criterion (only
modal operators expose `poll_signal_absmax`).  Ports and elements
share ONE label namespace (solver checkpoints key by label).  Scope
cut: pure sink — no dissipated-power readback (later, if wanted).

**En-route fix.**  The mesh round-trip serialiser used
`dataclasses.asdict` on RLC companions, which also emits the
``init=False`` transient-state fields (``_i``/``_vL``/…) that the
constructor rejects — every `PortLumped` carrying an `element=` was
un-reloadable from a store (latent since DD-109; no test covered it).
Now `_companion_to_dict` writes constructor fields (R/L/C) only.

**Traps.**  (i) `with_boundary_conditions`/`with_pec_boundaries`
reconstruct `Mesh(...)` field-by-field — a new carried field must be
added at BOTH sites or it silently drops (caught by unit test).
(ii) The element path has real distributed physics: deviation from
the closed shunt form grows ~(βd)² with the path's electrical length
(grid-independent; halving the path shrinks it ×4.7) and acts as a
~1 nH-class series inductance per few mm of path — keep element paths
electrically short, exactly like a physical SMD.

**Validation.**  `validation/lumped_element_shunt_certificate.py`
(PASSED): parallel-plate TEM line with exact DTBC ports, mid-line gap
shunt vs the closed form S11 = −Z0/(Z0+2Z), S21 = 2Z/(Z0+2Z) —
(1) quasi-static anchor < 1e-3 (measured 3.6e-4/9.1e-4 at βd ≈ 0.05);
(2) (βd)² envelope (ratio 4.71 on path halving); (3) series-RLC
resonance lands on f_res (2.90 vs 3.00 GHz with the path inductance
as a small documented perturbation).  Unit:
`tests/unit/test_lumped_element.py` (13: declaration, carriage,
builder, wiring).  Integration: store round-trip + cap-truncated
resume with an element (`tests/integration/test_project_scattering.py`).

---

## DD-124 — Footprint-exact thin-sheet rasterisation

**Date:** 2026-08-10 (Wilkinson tutorial groundwork; corrects a WP-M2
limitation in the [[DD-059]]-era thin-sheet mechanism).
**Status:** Accepted — implemented, gated.

**Problem.**  `ThinSheetSpec.rect` is the detected sheet's *bounding
box*, and the WP-M2 rasterisation painted that whole rectangle of
tangential E edges PEC.  Exact for a straight strip — every use until
now — but silently wrong for any non-rectangular thin metallization:
the Wilkinson racetrack (annulus + feed/stub/line bricks, one CSG
union 17 µm thin) turned into a solid metal plane spanning the box,
and the waveguide ports then failed with "hollow cross-section" (the
strip had merged with a full-plane short).  No warning at any stage.

**Decision.**  `rasterize_thin_sheet_footprint` (`mesh/_conformal.py`)
replaces the rect fill at the mesher's step 4b: the rect only
pre-filters candidates, and each candidate edge midpoint is classified
against the source shape's OCC solid (`point_in_shape` at the metal
mid-thickness, tolerance 0.45·t, DD-120 scale-aware).  Edges on the
lateral metal boundary count as metal (OCC `ON`), matching the
inclusive node selection of the rect path — bit-identical for straight
strips.  Fallback to the rect fill when the spec carries no shape or
OCC classification fails.  Cost: one point test per candidate edge
(~10 k for the Wilkinson plane, well under a second).

**Traps (measured en route).**  `min_cell_size` must stay ≥ ~3·t: at
32 µm < 2·t the cell above the sheet is nearly metal-filled, f_A → 0
faces produce NaN conformal matrices (32 µm run: NaN warnings + a 5 %
eps_eff jump; 42–51 µm clean).  Thin-sheet detection itself only runs
when `min_cell_size` is set — the sub-cell metallization story of the
Wilkinson tutorial depends on it.

**Validation.**  `tests/unit/test_thin_sheet_footprint.py`: an
L-shaped 17 µm sheet keeps its empty bbox corner PEC-free while the
legs rasterise; Wilkinson spike meshes with ports resolving at
49–50 Ω (previously unresolvable).  Straight-strip behaviour covered
by the existing microstrip fixtures (suite green).

---

## DD-125 — Meshes are immutable inputs: no write-back from solver or port builds

**Date:** 2026-08-10 (found via the DD-123 Wilkinson spike:
reciprocity broke on the second excitation).
**Status:** Accepted — implemented, gated.

**Problem.**  Both the FIT-TD solver setup and `build_modal_port` (all
three factory variants) applied the port-plane PEC flatten
([[DD-067]]-era: port plane := first interior slab, so the mode solver
and the update see the contour the volume wave sees) and then wrote
the flattened mask **back into the caller's mesh**
(`object.__setattr__(mesh, "pec_mask_edges", ...)`).  Every *later*
operator build on the same mesh — second excitation of one `run()`,
second `run()`, `solve_ports()` before a run — then computed its 2D
port modes against a plane already stripped of its wall contour:
`_pec_faces_from_mask` misclassified the cross-section boundary and
the mode solver produced a wrong profile, so the arriving field
projected to ~nothing.  Measured on the three-port Wilkinson: S21
−3.1 dB but S12 −26 dB (reciprocity broken), the port-1 *recording*
of the second run losing 23 dB while the raw physics (other channels)
was unchanged.  Every existing multi-port fixture has its ports on
opposite faces of one axis, where re-flattening is idempotent and
harmless — which is why 1800 tests never saw it.

**Decision.**  The flatten becomes strictly local: solver setup and
the port factories now derive a builder-local
`dataclasses.replace(mesh, pec_mask_edges=flattened)` view (the
flatten helper already returned a fresh array; only the write-back was
the defect).  The caller's mesh is never touched — codified invariant:
**meshes are immutable inputs to analyses, solvers and operator
builds**.  Side effect: within one build sequence, each port now
flattens against the pristine mask instead of inheriting earlier
ports' flattens — indistinguishable on opposite-face fixtures (suite
green), and the only correct choice for multi-face port sets.

**Validation.**
`test_analysis_scattering_td.py::test_run_leaves_the_mesh_untouched_and_is_repeatable`
(mask invariance + run-to-run identity); Wilkinson spike: S12 = S21 =
−3.11 dB, mask bit-identical before/after, repeat runs identical.

---

## DD-126 — Plane mirror as a geometry verb (`.mirrored()`)

**Date:** 2026-08-10
**Status:** Accepted — implemented, tested.

**Problem.**  The [[DD-113]] verb set covered translate / rotate /
scale but no reflection, so a structure symmetric about a plane had
to be either modelled twice or faked.  Both fakes are wrong in
general: `rotated('y', 180)` is a reflection *composed with* a flip of
the third axis, correct only for a body that is prismatic along that
axis and then only with a hand-computed offset; `scaled(-1.0)` is
`gp_Trsf::SetScale` with a negative factor — a point inversion through
the centre, i.e. all three axes negated, not a plane reflection.  Plane
symmetry is the common case in the target application (dividers,
couplers, filters, arrays), and a layer stack is almost never
symmetric about the plane normal, so the rotation fake breaks as soon
as the substrate and ground plane enter the model (found while
building a Wilkinson divider, internal record `userscripts/wilkinson.py`).

**Decision.**  `geo.transforms.mirror(shape, *, normal, position=0.0,
copy=False, unite=False, group=False)` plus the chainable
`ShapeOps.mirrored(...)`.  The plane is `p · n̂ == position`, so for an
axis letter *position* is simply the coordinate of the plane; *normal*
goes through `normalize_axis`, so a letter or any 3-vector works, per
the [[DD-113]] ergonomics rule.  Backend: `gp_Trsf::SetMirror(gp_Ax2)`
via `occ_mirror`, and `mirror_box` for the [[DD-120]] analytic box.

Two deliberate departures from the sibling transforms:

1. **No `repeat`.**  Mirroring twice across one plane is the identity,
   so a repeat count has no meaning.  `copy=True` (with `unite=True`)
   remains — it is the point of the verb: `half.mirrored(normal="x",
   copy=True, unite=True)` is the symmetric whole in one expression.
2. **`unite`/`group` without `copy` raise** instead of being silently
   ignored (the `_apply_repeat` behaviour the other verbs inherit).
   With no repeat count there is nothing to bundle, and honouring the
   flag silently would hand back a bare mirror image where the caller
   asked for the whole — a wrong geometry that meshes and solves.

**Reflections invert orientation** (determinant −1).
`BRepBuilderAPI_Transform` compensates by reversing the shape, which
is asserted rather than assumed: the test suite checks preserved
positive volume and `BRepCheck_Analyzer.IsValid()` on the image.

**Validation.**  `tests/unit/test_shape_ops.py::TestMirror` (15 cases):
reflection touches the normal axis only; mirror ≠ rotation on a chiral
body; volume/validity preserved; offset plane; involution; oblique
normal; analytic box matches the OCC box; copy/unite/group; Group
distribution keeps per-member materials; guard on `unite` without
`copy`.

---

## DD-127 — Material is optional: solids without one are construction bodies

**Date:** 2026-08-10
**Status:** Accepted — implemented, tested.

**Problem.**  `material` was a required field on every primitive, but
a large share of the solids in a CSG script are Boolean *tools* —
bodies that only shape other bodies and never reach the mesher.
`Difference`/`Union`/`Intersection` take their material from the
base (resp. first) operand and ignore the tools' entirely, so the
material on a cut cylinder was pure ceremony that had to be typed,
read and maintained.  The obvious alternative — defaulting to PEC —
is the worst option available in an EM solver: a forgotten `material=`
would silently insert a perfect conductor into the model, and PEC is
the one material whose accidental presence changes every result.

**Decision.**  `material` defaults to `None` on `_BaseShape`.  A solid
with a material is a physical object; a solid without one is a
**construction body**, the volumetric sibling of the material-less
`Face` construction profile that has existed since [[DD-035]].  The
omission is caught at the model boundary: `GeometryModel.add()`
rejects any shape whose `.material` is `None` (Groups are checked
member-by-member, since `add()` flattens them), naming the shape and
explaining the two ways out.  Because Boolean results inherit their
material, a rejected result also correctly points at the operand that
was the construction body.

**Rationale for the guard site.**  `add()` is the single door into a
model, so every downstream consumer (`mesher`, `_conformal`,
`plot_geometry`, `io.paraview`, `io.project`) keeps its unconditional
`shape.material` access and no None-check spreads through the code
base.  Failing there also fails *early* — at model assembly, not at
mesh time with a stack trace far from the offending line.

**Validation.**  `tests/unit/test_geometry.py::TestConstructionSolids`
(7 cases): cut tool needs no material; Boolean result inherits from
the base; model rejects a bare construction solid, a Boolean result
whose base was one, a Group member without one, and a transformed
one; the message names the shape.

---

## DD-128 — `Shape` is public: one documented home for the operators and verbs

**Date:** 2026-08-10
**Status:** Accepted — implemented, gated.

**Problem.**  The [[DD-113]] verbs were undiscoverable in the API
reference: `docs/api/geo.md` renders `automodule:: magnelio.geo`, and
nothing in that chain reached them.  Two independent gaps, found by the
developer while looking for `.translated()` in the built docs.

1. The verbs were methods of the `ShapeOps` mixin in the private
   `geo/_shape_ops.py`, which `geo/__init__.py` does not re-export, so
   autodoc never saw the class.  The primitives inherit them, but
   autodoc skips inherited members without `:inherited-members:`
   (verifiable in the built page: `id="magnelio.geo.Brick.from_corners"`
   exists, `…Brick.translated` does not).
2. The *substantive* documentation was not on the methods at all but on
   the free functions in `geo/transforms.py` / `geo/modifications.py`;
   the methods were one-liners of the form "Translated copy; see
   :func:`…transforms.translate`".  No doc page renders those two
   modules — every page is an `automodule` on a namespace — so the
   targets did not exist and the pointers went nowhere.

The root cause is a [[DD-113]] leftover: that decision moved the free
functions out of the public API in favour of the methods, but the
prose stayed behind at the now-private location.

**Options evaluated:**
| Option | Notes |
|--------|-------|
| Public `Shape` base class, full docstrings on the methods | Selected |
| `:inherited-members:` + render `transforms`/`modifications` | Republishes exactly the spelling DD-113 retired; the verb list repeats on all 12 shape classes and the prose lives at a second location |
| Full docstrings on the methods, mixin stays private | No API growth, but rendering still needs `:inherited-members:`, so all 10 verbs print in full on each of the 12 classes |

**Decision.**  `geo/_shape_ops.py` becomes `geo/shape.py` and `ShapeOps`
becomes **`Shape`**, exported from `magnelio.geo` and first in its
`__all__`.  The full descriptions move from the free functions onto the
methods — parameters, returns, raises, worked examples — and the free
functions keep a one-line "Implementation of :meth:`…Shape.<verb>`"
plus whatever is true only of them (Group distribution, the OCC
operation used).  The class docstring carries what applies to all of
them: shapes are immutable and every verb returns a new one; materials
follow the base operand; and the shared `copy`/`unite`/`group`
vocabulary is explained once.

`Shape` is honest as a public name for a second reason: it is the type
users already hold in every variable, and the one they need for an
`isinstance` check or a type annotation.  `Curve` and `ThinWire` stay
outside the hierarchy — they are 1D objects (a sweep spine, a sub-cell
wire), and neither the Boolean operators nor the verbs apply to them.

**Validation.**  `tests/unit/test_shape.py::TestDocumentedSurface`:
`Shape` is exported; every solid class and every Boolean result is a
`Shape`; each of the 10 verbs carries its own `Parameters` *and*
`Returns` section (this is the regression gate — a verb degraded back
to a pointer fails it); the three operators are documented.  Built
docs re-checked: `id="magnelio.geo.Shape.<verb>"` present for all ten,
zero new Sphinx warnings.

---

## DD-129 — The scattering-result contract documents itself

**Date:** 2026-08-10
**Status:** Accepted — implemented, gated.

**Problem.**  The object every user script holds after `run()` —
`ScatteringTDResult`, or a `Project` reader with a store — published
almost none of its contract in the API reference.  Found by sweeping
all 69 exported classes for members inherited from non-exported bases,
the follow-up to [[DD-128]].  Three independent causes, stacked:

1. **Inherited and invisible.**  `phase`, `plot_s`, `to_touchstone` and
   `to_skrf` come from `ScatteringResultMixin`, which
   `analysis/__init__.py` imports but leaves out of its `__all__`;
   autodoc skips inherited members by default, so the Touchstone and
   scikit-rf exports — the two calls that get results out of Magnelio
   and into anything else — appeared nowhere.
2. **Undocumented, therefore unrendered.**  `S`, `db`, `f_axis`,
   `channels` and `excitations` carried no docstring at all on
   `ScatteringTDResult` (and `db`, `f_axis`, `channels`, `excitations`
   on `Project`).  Autodoc renders only documented members, so the
   central accessor of the whole library, `result.S("port2", "port1")`,
   was absent from its own class page.
3. **A phantom member.**  The class docstring carried an invented
   numpydoc section, `Convenience\n-----------`, holding exactly the
   prose those methods were missing.  Napoleon does not know the
   heading, so it rendered as an attribute named "Convenience" and the
   text stayed stuck to the class instead of reaching the methods.

The `ScatteringResult` protocol had the same shape of hole: it declared
the contract but every member was a bare `...` stub, so the page showed
a class with a one-line summary and no members.

**Decision.**  The contract is documented where it is implemented, and
the reference is configured to show it.

- The `Convenience` section is dissolved: its content moves onto `S`,
  `db`, `a`/`b` as real docstrings, and what remains that is genuinely
  about the class as a whole becomes a proper `Notes` section.
- `S`, `db`, `f_axis`, `channels`, `excitations` get docstrings on both
  implementations; `settings` joins the `Attributes` block.
- The protocol's stubs get one-line docstrings and its class docstring
  lists the members, so it reads as the contract it claims to be.
- `docs/api/analysis.md` and `docs/api/io.md` gain
  `:inherited-members:`.  Unlike the geometry case in [[DD-128]] this
  is proportionate rather than wasteful: both classes inherit from
  exactly one mixin contributing exactly four methods, and those
  methods belong conceptually to the result object, which is where a
  reader looks for them.  No public name is added.

**Validation.**  `tests/unit/test_api_documentation.py` (39 cases):
every contract member is documented on both implementations (dataclass
fields via their `Attributes` entry), the protocol declares and
documents each, both implementations satisfy it, and no class docstring
invents a numpydoc section heading.  Both gates were shown to fail when
the defect is reintroduced — a removed docstring and a restored
`Convenience` heading are each caught.  Built docs: all twelve contract
members now render on both `ScatteringTDResult` and `Project`; warning
count unchanged at two, both pre-existing (the `GeometryModel`
definition-list warning and an ambiguous `label` cross-reference, the
latter verified present without the new option).

**En-route fix.**  The module docstring cited the cross-check as
`tests/unit/test_result_contract.py`; it lives in `tests/integration/`.

**Follow-up (same day): the gate generalised, one more defect found.**
Cause 3 turned out not to be a one-off.  The section-heading check and
a new phantom-*parameter* check now sweep every object reachable
through a namespace `__all__` (293 docstrings), and each found a case:

* `GeometryModel` wrote `Example::` where numpydoc wants an `Examples`
  section.  Napoleon ends a Parameters block only at a heading it
  knows, so both example paragraphs became parameter *names* with the
  example code as their description — the class advertised **five**
  parameters where it takes three, and this was the long-standing
  `Definition list ends without a blank line` warning in the docs build
  (recorded as open in this file's own status notes).  Fixed; the build
  is down to one warning, the remaining one pre-existing and unrelated
  (an ambiguous `label` cross-reference, present before this work).
* `Mesh.with_boundary_conditions` wrote `Note` for `Notes` — found by
  the generalised check on its first run.

Both gates were again shown to fail on the reintroduced defect.  The
lesson worth keeping: an invalid numpydoc heading fails *silently* in
the common case (`Convenience` produced no warning at all, just a
phantom attribute), so a linter cannot be traded for the build log
here.

**Follow-up, 2026-08-11 — the last warning is closed and the build is
clean.**  The remaining `label` cross-reference did not live in
`docs/api/io.md` at all; the warning names the *page*, not the source,
and carries no line number.  It came from the docstring of
`Project.export_paraview`, whose parameter type read
`str or (label, mode)`: napoleon turns a numpydoc type into a `:type:`
field, Sphinx resolves the field's contents as a cross-reference, and
two objects in the tree are called `label`.  The type now names only
resolvable types and the structure moved into the description, where
double backticks keep `(label, mode)` literal.

A scan of every public docstring found **19** such type fields —
`(fig, ax)`, `Signal1D`, `Nf` and friends.  None of them warned,
because outside nitpicky mode Sphinx reports only what is *ambiguous*,
never what is merely unresolvable; `label` happened to be the one name
that exists twice.  Latent rather than harmless: a second class gaining
a member `ax` would surface the same warning somewhere unrelated.  All
of them are now cleaned up — nine `(fig, ax)` return blocks became
named `fig`/`ax` entries with their real matplotlib types, five monitor
`corners` types became `tuple of tuple` with the corner layout moved
into the description, and the `shape (Nf,)` and
`dict[(str, int), (Signal1D, Signal1D)]` spellings lost the parts that
were never types.  `matplotlib` joined `intersphinx_mapping` so the new
return types link rather than merely read correctly.

`test_type_fields_name_only_resolvable_types` keeps it that way.  It
flags a parenthesised group in a type field whose words are not
resolvable type names — the check deliberately allows
`tuple of (str, int)`, which is ordinary numpydoc, because `str` and
`int` do resolve.  A first, stricter draft banned parentheses outright
and produced fourteen false positives on correct docstrings; the list
of resolvable names is therefore explicit in the test rather than
implied by a bracket heuristic.  Shown to fail on the reintroduced
`str or (label, mode)`.

One habit worth keeping: warning-freedom can only be established with
`sphinx -E`.  A cached rebuild does not repeat warnings for unchanged
files, so a quiet log from an incremental build proves nothing.

---

## DD-130 — `Brick.from_ranges`: one coordinate range per axis

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  Layered structures are described by the planes they lie
between, not by two opposite corners.  A substrate under a ground plane
at `z = 0` is "x from 0 to w, y from 0 to l, thickness h downwards";
spelling that as `from_corners((0, 0, -h), (w, l, 0))` interleaves the
three axes into two tuples, so a value belonging to z sits in the middle
of a corner and the thickness never appears literally — it has to be
carried into a coordinate by hand.  Both existing spellings force the
same rewrite, and it is where sign errors enter.

**Decision.**  A second classmethod, `Brick.from_ranges`, taking one
range per axis as keywords:

```python
Brick.from_ranges(x1=0, dx=w, y1=0, dy=length, z2=0, dz=h, material=fr4)
```

Each axis takes **exactly two** of its three keywords `a1`, `a2`, `da`.
That single rule covers all three useful spellings — two bounds, lower
bound plus extent, and upper bound plus extent (a box grown *downwards*
from a plane, which is what the substrate above needs) — instead of
privileging one bound as mandatory.  Rationale for the details:

- **Exactly two, never three.**  A redundant third value is rejected
  rather than checked for agreement.  Consistency checking would need a
  tolerance, and a tolerance turns a typo that happens to agree to
  within it into a silent acceptance.
- **Normalising, like `from_corners`.**  Bounds may come in either
  order and extents may be negative (matching `Cylinder.height`, where a
  negative height extrudes backwards).  `origin` always holds the
  minimum and `size` is non-negative, so both classmethods and the plain
  constructor produce indistinguishable fields downstream.
- **Keyword-only.**  Nine floats in fixed positions would be a trap;
  there is no defensible positional order for them.
- **Errors name the axis and its three keywords.**  The message says
  which axis is under- or over-determined and what it accepts, because
  the mistake is always local to one axis.

`from_corners` stays: it is the natural spelling for a cutting box
given by two corners, and rewriting it in terms of the other would be
churn.  The two are alternative front doors to the same fields.

**Consequence.**  No numerical behaviour changes — this is authoring
sugar over the existing `origin`/`size` fields, resolved before any OCC
shape exists.  Covered by `TestBrickFromRanges` in
`tests/unit/test_geometry.py` (14 cases: each spelling, mixed spellings
across axes, negative extents, swapped bounds, agreement with
`from_corners`, optional material, and the under-/over-determined
rejections).

---

## DD-131 — Profiles from curves: `joined`, `covered`, `Path`

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  `Curve` offered four constructors (polyline, arc, spline,
helix) and no way to combine them.  Every outline that mixes straight
runs with arcs — a chamfered pole piece, a rounded pad, a segment of a
ring — was therefore unbuildable: `Face` covers only axis-normal
polygons, and there was no route from a wire to a face at all
(`make_face` was hard-wired to `_FACE_UV` polygons).  The verbs that
consume profiles (`extruded`/`revolved`/`swept`) already existed and
already accepted a standalone `Face`, so the gap was exactly one link
wide.  Reported from modelling a 20° segment of a hollow cylinder.

**Decision.**  Three additions, one link each:

- **`Curve.joined(*curves)`** chains segments into one wire.  An
  *instance* method, not a classmethod: `a.joined(b, c)` reads as the
  chain it builds, whereas a classmethod called on an instance would
  silently drop the receiver.  Chains flatten (`_segments`), so
  `a.joined(b).joined(c)` is a three-segment wire, not a nest.
- **`Curve.covered()`** turns a closed planar curve into a planar
  sheet, via a new backend `make_wire_face` — the free-boundary sibling
  of `make_face`.
- **`geo.Path`** is a frozen-dataclass pen over the same machinery:
  `line_to`/`arc_to`/`spline_to` remember the current point, `curve()`
  and `closed()` delegate to `joined`.  Pure sugar, no backend of its
  own; immutable so a common prefix can branch into several outlines.

Supporting rationale:

- **`PlanarSheet` marker base.**  `Face` and the covered sheet are the
  same kind of thing to four dispatch sites (the extrude material
  guard, the revolve guard, the extrude profile-vs-solid branch, the
  mesher's standalone-sheet rejection).  Introducing a shared base and
  switching those sites from `isinstance(x, Face)` was cheaper than
  teaching each one about a second type, and it is what makes a covered
  sheet work in the existing verbs with no further change.
- **Seam tolerance is relative, not absolute** — 1e-6 of the chain's
  own bounding-box diagonal.  An absolute metre threshold cannot be
  right for both a micrometre profile and a kilometre one (DD-120).
  `BRepBuilderAPI_MakeWire` only fuses vertices within
  `Precision::Confusion()`, which is tighter, so seams inside the
  public tolerance but outside the kernel's are healed once through
  `ShapeFix_Wire`.  It is *not* run unconditionally: it may reorder and
  reverse edges, and exact input must stay untouched.
- **Closure is checked eagerly, planarity lazily.**  Endpoints are
  captured at construction (`_ends`), so a gap names the segment index
  and the distance — which the kernel cannot.  Planarity needs the
  actual curves and stays with `BRepBuilderAPI_MakeFace`.
- **`arc_to` takes `normal=`.**  The `center=` form has two solutions,
  and for diametrically opposite ends it has infinitely many (the plane
  itself is free).  That case is not exotic — it is the rounded end of
  a slot, the most common use of the form.  `normal=` names the axis
  the arc turns about and settles direction and plane at once, running
  counter-clockwise about it, the same handedness as `.rotated()`.
  Without it the short arc is drawn (`major=True` for the long one) and
  antipodal ends are rejected with a message pointing at `normal=`.
- **A helix has no `_ends`.**  Its endpoints depend on the `gp_Ax3`
  frame convention, which is deterministic but not worth pinning for
  this; a helix in a chain skips the eager check and relies on the
  kernel.  Helices are sweep spines, not profile segments.

**Deferred/rejected** (recorded so the gaps are known, not forgotten):
bend and cylindrical-bend operations; `Insert`/`Imprint` Booleans with
material precedence; STEP import and healing (taken up in DD-178);
analytical parametric
curves (in Python, generate points and spline them); elliptical
cylinder; sphere pole truncation; twist and taper on extrusions.
**Local/working coordinate systems are rejected outright**, not
deferred: they exist in GUI-driven tools because a mouse needs a
drawing plane, and in a Python API they would add a mode to every call
for nothing.

**Consequence.**  Covered by `TestCurveJoined`, `TestCurveCovered` and
`TestPath` in `tests/unit/test_geometry.py`, including the relative
tolerance at both millimetre and micrometre scale, and the mesher's
rejection of a standalone sheet.

---

## DD-132 — `Cylinder` gains a bore and an angular segment

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  A tube and a curved slab — a segmented electrode, a
septum, a shield sector — are everyday parts, and both needed Boolean
scaffolding: a tube two cylinders and a cut, a segment a wedge built by
hand.  The tube case is enough of a staple that the `_BaseShape`
docstring taught the Boolean spelling as an idiom.

**Decision.**  Two fields on `Cylinder`, both defaulted so every
existing call is untouched: `inner_radius=0.0` and
`angle_deg=None` (a scalar for a segment starting at zero, `(start,
end)` for one anywhere).  The plain solid cylinder keeps its original
backend path bit-for-bit; the general path builds the segment with the
four-argument `BRepPrimAPI_MakeCylinder` and cuts the bore with an
axially overshooting inner cylinder.

- **Angles turn like `.rotated()`.**  Right-handed about the axis, with
  zero on the first coordinate direction perpendicular to it (`+x` for
  `'z'`, `+y` for `'x'`).  Both the kernel frame and the analytic
  bounding box read that frame from one helper, `_axes.reference_dir`,
  so they cannot drift apart — the failure mode this guards against is
  a box that is tight but wrong.
- **A negative height moves the body, it does not reverse the sweep.**
  The frame is built from the declared axis and the base point shifted
  instead, so `height=-h` and `angle_deg=(0, 20)` mean what they read.
- **The segment box is tight, not merely containing.**  A sector's
  extremes lie on the arc ends, the apex or inner arc, and wherever the
  arc crosses a coordinate axis; `_scaling.sector_uv_points` enumerates
  exactly those.  A conservative full-radius box would inflate the
  DD-120 scale estimate for a thin sector by the ratio of the full disc
  to the sector.

**Mesher note.**  `_face_critical_planes` handles segments correctly by
construction: tangent candidates are filtered against the *trimmed*
face's bounding box, so a 20° segment contributes no far-side tangent
plane.  The two flat cut faces contribute a critical plane only when
they are axis-normal, i.e. at multiples of 90°; at other angles they
fall back to the shape's silhouette extents, which is the same
treatment every non-axis-aligned face gets.

**Consequence.**  Covered by `TestCylinderSegment` and
`TestSegmentCriticalPlanes` in `tests/unit/test_geometry.py`: all four
volume variants against their closed forms, the bit-identical default,
handedness against `.rotated()`, tightness of the box, containment on
every axis including a skew one, and the negative-height case.

---

## DD-133 — `shelled()` and `thickened()`: two verbs, not one

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  A housing, a waveguide run and a cavity are all "a solid
with its inside removed", which had to be modelled as a difference of
two solids whose inner one the user had to dimension by hand — for
anything but a box, that inner solid is not simply the outer one scaled.
The mirror-image gap: a drawn outline had no direct route to a
metallisation of a given thickness.

**Decision.**  Two verbs.  `shelled(thickness, opening_face_near=)`
hollows a solid, walls inward so the outer surface stays put, with the
named faces left out to become openings.  `thickened(thickness,
direction=)` grows a planar sheet into a slab along its own plane
normal.

- **Why not one dispatching verb.**  GUI tools spell this as a single
  command that switches on what is selected.  Here the two share
  nothing: different parameters (`opening_face_near` is meaningless for
  a sheet, `direction` for a solid), different input categories,
  different kernel machinery.  One verb would let a category mistake —
  shelling something you thought was a solid — produce a wrong body
  silently; two verbs turn it into a `TypeError` that names the
  sibling.
- **Thicken is a prism, not an offset.**  A sheet grown along its own
  plane normal is exactly an extrusion, so it reuses `make_extrude`;
  `BRepOffsetAPI_MakeThickSolid` would be a heavier route to the same
  slab with more ways to fail.  The normal's sign is canonicalised
  (largest component positive) so "forward" is reproducible, and
  `direction="backward"`/`"symmetric"` give the other placements.
- **Two kernel paths for shelling, because an empty face list is not a
  shell.**  `MakeThickSolidByJoin` with no faces to remove returns the
  *shrunk solid*, not a hollow one — silently the wrong shape, and the
  first version shipped it.  A sealed void is built instead as the
  original minus its inward offset (`MakeOffsetShape.PerformByJoin`).
- **`IsDone()` is not sufficient.**  An offset that cannot be built
  still reports success and hands back a null shape, and walls thicker
  than the body come back as the *untouched* solid.  Both are checked —
  the second by comparing volumes — and reported as a wall that does
  not fit rather than passing through.

Outward and centred shelling are deferred; inward is what keeps a
housing's outer dimensions, which is the case that matters here.

**Consequence.**  Covered by `TestShelled` and `TestThickened` in
`tests/unit/test_modifications.py`: wall volumes against their closed
forms for sealed, one-opening and two-opening boxes and for a cylinder,
opening deduplication, the three thicken directions by bounding box,
and both degenerate-wall reports.

---

## DD-134 — `Loft`: an n-ary constructor for free profiles

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  `make_loft` already wrapped `BRepOffsetAPI_ThruSections`,
which takes any number of sections, but the public `.lofted()` verb
reached it only through two *solids* and their face selection.  A horn
or a taper defined by its cross-sections — the natural way to define
one — had no route in, and neither did a three-section transition.

**Decision.**  `make_loft` takes a list of wires, and a new public
`Loft(*sections, blend=, material=, name=)` accepts planar sheets and
closed curves as sections.  `.lofted()` is unchanged.  (The mode
argument was spelled `ruled=True/False` until DD-144 turned it into the
three-valued `blend=`; `Loft` takes the two of those three modes that
free cross-sections can support.)

- **Why a CamelCase constructor, not an overload of the verb.**  The
  verb's grammar is receiver-anchored (`self` + `face_near` + `other` +
  `other_face_near`), which is right for its bridge-two-solids job.  An
  n-ary loft over free profiles has no privileged receiver and no face
  selection; forcing it through the verb would need a second argument
  form.  DD-113 already reserves CamelCase for the n-ary constructors
  (`Union`), and this is one.
- **`material` does not default to a section's.**  A cross-section is
  a curve or a sheet; neither carries a volume material worth
  inheriting, so the result is a construction body unless told
  otherwise, and `GeometryModel.add` remains the single guard (DD-127).
- **The ruled box is exact.**  Every point of a ruled loft is a convex
  combination of section points, so the union of the section boxes
  contains it; only the smooth case needs padding.

**Consequence.**  Covered by `TestLoftSections` in
`tests/unit/test_modifications.py`, including a frustum and a
two-frustum stack against their closed forms and agreement with
`.lofted()` on the same profiles.

---

## DD-135 — `Curve.traced()`: conductor tracks from a centreline

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  A routed track — the standard way a feed line or a
coupling stub is described — had to be assembled from individual bricks
and wedges, one per segment, with the corners worked out by hand.

**Decision.**  `Curve.traced(width, thickness, caps=, normal=)`
offsets the centreline within its plane and extrudes the outline.  A
verb on `Curve`, symmetric with `covered()`: one fills the inside of a
closed curve, the other fattens the curve itself.  (The `ThinWire`
precedent does not apply — that is a mesher category, DD-080, whereas a
track is an ordinary solid.)

- **The plane must be anchored, not inferred from the wire alone.**
  `BRepOffsetAPI_MakeOffset` on a bare wire fails for a *straight*
  centreline, because a straight line lies in infinitely many planes —
  and a straight feed line is the commonest case of all.  The spine is
  therefore offset against an explicit plane face (`Init(face)` +
  `AddWire`).  `normal=` supplies that plane when the curve does not
  determine one; the error distinguishes "straight, name the plane"
  from "not planar at all" by testing collinearity, since the two need
  opposite fixes.
- **Flat caps by trimming, not by wire assembly.**  Offsetting an open
  wire as a closed result gives round caps directly.  Building the
  square-ended outline instead by chaining two one-sided offsets and
  two cap edges is fragile; cutting the extruded solid with a
  half-space at each end, placed on the end tangent, is robust and
  reuses the Boolean path.  Flat is what a track meeting a port plane
  needs, so it had to work.
- **A closed centreline goes through its own face.**  With the plane
  anchor, both signs of the offset return the same inner contour for a
  closed spine; using the spine's own face as the offset input gives
  outer and inner as expected, and the track is the ring between them.
- **A self-intersecting offset is an error, not a repair.**  More than
  one contour means the widened sides ran into each other; the message
  names width, bend radius and clearance rather than guessing.

Outside corners of a polyline centreline come out rounded — a property
of offsetting, and closer to a fabricated track than a mitred corner.
Spline centrelines are best-effort: `MakeOffset` is most robust on line
and arc geometry.

**Consequence.**  Covered by `TestCurveTraced` in
`tests/unit/test_geometry.py`: straight flat and round, a right-angle
corner and a closed loop against their closed forms, both plane errors,
and the too-wide report.

---

## DD-136 — `Shape.volume()`: the built geometry, not the nominal one

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  There was no public way to ask a shape how much volume it
encloses.  Everything downstream of a Boolean or a modification is
exactly the case where the answer is not derivable from the parameters:
a difference reports what its operands were, not what is left; a
chamfered block reports its box.  The gap surfaced while writing
tutorial 14, where two constructions of the same electrode had to be
compared and only `bounding_box()` was available — which agrees for
many pairs of *different* solids and so proves nothing.  Internally the
quantity was computed in half a dozen places (`check_pairwise_overlaps`,
several test helpers), each re-writing the same `BRepGProp` call.

**Decision.**  `Shape.volume(scale=None)` returning cubic meters,
mirroring `bounding_box()` in shape and in contract: the DD-120 model
scale is chosen from the shape itself unless given, and the kernel's
`s³`-scaled result is divided back (lossless — `s` is a power of two).
The shared `occ_volume()` backend helper replaces the repeated
`GProp_GProps` boilerplate.

- **A method, not a property.**  It builds the OCC shape and runs a
  kernel integration; a property would hide that behind attribute
  syntax.  `bounding_box()` set the precedent for the same reason.
- **Absolute value.**  A reversed orientation would otherwise report a
  negative volume, which is a kernel detail, not an answer.
- **A planar sheet reports zero** rather than raising: a `Face` has no
  thickness, and zero is the true answer, not an error.
- **`Group` overrides it and sums its members.**  A Group is a bundle
  of separate solids with no fused shape of its own.  Summing double
  counts an overlap — which `GeometryModel` rejects anyway, so the
  simple sum is right wherever a Group is legal.

**Consequence.**  Covered by `TestShapeVolume` in
`tests/unit/test_geometry.py`, including a Boolean difference against
its closed form and the scale sweep from nanometre to kilometre models
(relative error 1.2e-16 at the extremes, where the automatic scale
factor has to cancel exactly out of the result).

---

## DD-137 — Cross-sections on a plane that lies in a face

**Date:** 2026-08-11
**Status:** Accepted — implemented, tested.

**Problem.**  Cutting exactly along a face — the top of a substrate,
the plane of a metal layer, `z = 0` on a strip standing on it — is the
first thing a user asks for, and it was quietly wrong.
`BRepAlgoAPI_Section` looks for intersection *curves*, but the
intersection of a solid with a plane lying in one of its faces is a
*face*; on a Boolean result the operator then returns only part of the
seam structure.  Measured: two 10 mm bricks fused end to end and cut at
their common bottom face returned 20 mm of strip instead of 40 mm.  It
looked like a broken `+` operator and was the plot.  Recorded as a trap
in the private notes since 2026-08-10; this closes it.

**Decision.**  `cross_section_polygons` gains `exact_at_faces=False`.
When set, a cheap scan detects whether an axis-normal planar face of
the solid lies in the cutting plane, and those planes are answered by a
face-face Boolean (`BRepAlgoAPI_Common` against the plane) whose
resulting faces yield the boundary wires directly.  The geometry plot
passes it; nothing else does.

- **Opt-in, not default.**  The mesher's cell classification calls this
  function per cell-centre plane, and by construction those never lie
  on a face (grid lines are anchored *on* material faces, cell centres
  fall between them).  Paying a face scan plus a heavier Boolean there
  would be pure overhead, and changing what a tangent plane returns
  could shift a cell's material assignment.  Measured overhead of the
  detection alone: 50 us against 2 ms for the section itself, so the
  cost is not the reason — the unchanged meshing path is.
- **The detection is exact coincidence**, within `Precision::Confusion()`.
  A plane merely *near* a face sections perfectly well; only the
  degenerate case is diverted.

**Correction found on the way.**  The documented return contract
("outer boundaries counter-clockwise, holes clockwise") was already
false: the `y` frame `(u, v) = (x, z)` is left-handed about its own
normal, so contours come out mirrored there, and a hole in a `z` cut
came back counter-clockwise too.  Nothing depends on it — both
consumers (`classify_cells_from_cross_sections` and the geometry plot)
use the even-odd rule, which is orientation-blind — so the docstring
now states that contract instead of a winding convention that was never
maintained.  Normalising the winding was rejected: it would require an
outer-versus-hole containment test per contour, on a hot path, for no
consumer.

**Consequence.**  Covered by `TestSectionAtFace` in
`tests/unit/test_geometry.py`: the plain section's half-strip pinned as
the motivating behaviour, both faces of the solid recovered, a hole
surviving a face cut, the plot spanning the whole strip, and — the
regression that matters — interior planes returning *bit-identical*
polygons with and without the option.

---

## DD-138 — Eigenmode auto-shift: escalation ladder instead of a single filled-cavity estimate

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.  Closes KB-011.

**Problem.**  `EigenmodeSolver3D` estimated its shift-invert target
once, dividing the empty-cavity eigenvalue estimate by the **global**
`eps_r_max` of the material library — correct for a filled cavity,
wrong for a sparsely filled one.  On the KB-011 fixture (ceramic ring,
eps_r = 45, ~1 % fill) the estimate lands a factor ~12 (in λ) below
the true fundamental, closer to the curl-curl null space than to any
physical mode; ARPACK converges on gradient vectors, the 1 MHz filter
discards them, and the solve hands back **0 of 6** requested modes —
silently.  Explicit shifts also under-delivered on that fixture
(shift 2 GHz: 3 of 6) with no indication.

**Decision.**  Two independent changes, matching the two defects:

1. **Auto-sigma is a ladder, not a point.**  Monotonicity gives
   rigorous brackets: raising ε anywhere only lowers eigenvalues, so
   the filled-cavity estimate (ε_r,max) bounds the fundamental from
   below and the empty-cavity estimate (ε_r = 1) from above.  The
   solve starts at the lower bound — escalating from below can never
   skip the fundamental — and, whenever fewer physical modes than
   requested come back, retries with the shift raised ×4 in λ (one
   octave in f) up to the upper bound, growing the ARPACK subspace
   (`+6` per attempt).  The **physical eigenpairs of all attempts are
   merged** (B-metric subspace dedup per near-degenerate cluster), so
   a raised shift cannot lose an already-found low mode and degenerate
   partners found by different attempts both survive.  An explicit
   user `sigma` stays a single attempt: the shift is a user decision.
   A filled or empty cavity has a single-entry ladder — behaviour and
   cost unchanged there.
2. **Under-delivery is loud.**  Whatever the path (auto or explicit,
   ARPACK or LOBPCG), returning fewer modes than requested emits a
   `RuntimeWarning` naming the found/requested counts, the discarded
   null-space count and the `sigma=(2*pi*f_estimate)**2` remedy.

Rejected alternative: a volume-weighted ε estimate (the other KB-011
candidate).  It is not a bound in either direction — on the DR fixture
it lands a factor ~8 (in λ) *above* the fundamental, where ARPACK can
return plausible-but-wrong "lowest" modes with no null-space symptom
to trigger a retry.  The ladder pays one extra factorisation instead
and keeps the never-skip guarantee.

**Measured.**  KB-011 fixture (30 926 cells): auto path 0 → 6 modes,
66 s (two attempts) against 34 s for the old empty result; spectrum
identical to the explicit-shift reference (2.6192 / 4.7522 ×2 /
6.7128 ×2 / 6.8043 GHz).  The merge also surfaced that 6.71 GHz is a
degenerate pair the old single-shift solve silently truncated (it
reported 6.7128 / 6.8043 as neighbours).  Coarse fixture (3 136
cells): 5 → 6 modes.  Rectangular-cavity and analysis suites
unchanged.

**Consequence.**  Gates in
`test_analysis_eigenmode.py::TestSparseHighContrastCavity`: the
escalation delivers 6/6 warning-free on the coarse puck fixture, and a
deliberately bad explicit shift under-delivers with the warning.  The
`AnalysisEigenmode`/solver docstrings document the escalation and the
warning; `spec.md` untouched (solver-internal heuristic).

---

## DD-139 — Eigenmodes reach ParaView on the same path as monitors

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.

**Problem.**  `Project.export_paraview` covered only monitors of driven
runs.  Rendering an eigenmode meant writing the `.vtr` by hand — the
tutorial-13 groundwork did exactly that (internal record
`investigations/dr_filter/MEASUREMENTS.md`, M11): pull the mode through
`_interp_to_cell_centres`, call `vtkXMLRectilinearGridWriter`, borrow
the driven run's `geometry.vtm`.  Roughly thirty lines of solver
internals per picture, and none of the session machinery (coloured
geometry, linked slice planes, calibrated glyphs) came with it.

**Decision.**  Eigenmodes get the frequency-monitor shape one directory
up: `paraview/eigenmodes/mode_*.vtr` + `paraview/eigenmodes.pvd` +
`paraview_open.py` + `paraview.pvsm`, written **in the project
directory** rather than under `runs/`, because that is where they
belong — an eigenmode analysis has no excitation and no run.  The
session pipeline is spec-driven and needed no change: the exporter
hands it one more monitor spec with `reader="pvd"`.

Three decisions inside that:

- **The ParaView axis carries the mode index, not the eigenfrequency.**
  Degenerate pairs share a frequency to the last digit and are the rule
  in any symmetric cavity — DD-138's cross-attempt merge made a
  previously truncated pair visible on the very fixture this was
  written for.  Two datasets at one timestep hide each other, so the
  frequency travels as field data instead: displayed, exact, and not
  load-bearing.
- **Fields are peak-normalised per mode, and the divisor is stored.**
  An eigenvector has no absolute amplitude; the solver's own scaling
  ran to 2.2e8 on the tutorial-13 cavity, which reads as V/m and is
  not.  Writing `E_peak_before_normalisation` / `H_peak_…` into the
  file keeps the normalisation reversible rather than lossy.
- **Glyphs on E only.**  E and H are normalised by unrelated peaks, so
  drawing both would render them at comparable lengths while they are
  physically incomparable.

**Measured.**  30 x 20 x 15 mm air cavity, 4 modes: session written in
one call, `|E| = 1` exactly per mode, cell dimensions equal to the
project grid.  Found and fixed en route: `_freq_vtr_grid` left the
coordinate arrays unnamed, and VTK then writes them as
`Array 0x<address>` — the **frequency-monitor export was not
byte-reproducible between processes** either.  Naming them fixes both.

**Consequence.**  `Project.export_paraview_eigenmodes()`, called
automatically by `ProjectStore.write_eigenmodes` under the same
best-effort guard as the run-close export (visualization must never
invalidate a stored result).  Gates:
`test_paraview_export.py::TestEigenmodeExport` — session generated on
write, mode-index axis, normalisation with recoverable divisors, grid
agreement, byte-identical regeneration, and the empty-project no-op.

---

## DD-140 — `MonitorFieldFrequency(interval=)`: sub-sampling a DFT is not a recording interval

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.

**Problem.**  A whole-volume frequency monitor accumulates a DFT over
every cell at every step — arithmetic comparable to the solver itself.
Measured on a 33 180-cell guide over 3 000 steps: the monitor costs
**+144 %** of the bare run's wall-clock time.  Tutorial 13 met the same
wall (5:32 → 9:54 with the monitor in), and the only remedy available
was to take the monitor out of the executed run and ship the picture as
a static asset.  `MonitorFieldTime` has had an `interval` since DD-104;
the frequency monitor had no way to thin its sampling at all.

**Decision.**  `interval` in seconds, matching `MonitorFieldTime`'s
vocabulary — but with guards that monitor does not need, because the
two things are not the same operation.  A time monitor's interval
decides how many snapshots are *kept*; a DFT interval is genuine
under-sampling of an oscillating integrand, and too few samples per
period do not coarsen the output, they corrupt it.  Hence:

- **Rejected below 4 samples per period** of the monitor's highest
  frequency, **warned below 10.**  Nyquist (2) is the theoretical floor
  and far too optimistic for a Riemann sum of an oscillation; the
  documented rule of thumb is 10 and above.
- **The stride rounds down**, never up: the interval is an upper bound
  on the sample spacing, and rounding up would sample coarser than
  asked — which would let an interval derived from a
  samples-per-period rule trip the very margin it was chosen to keep
  (measured: a request for exactly 10 per period landed on 9.8 and
  warned).
- **The integration weight is the realised spacing** (`stride * dt`),
  so bins keep their units and `renormalize` is untouched.  The
  leapfrog half-step for H stays on the *solver* `dt` — that is where H
  physically sits, not a property of the sampling.
- **The second condition cannot be checked and is documented instead:**
  the fields must carry nothing above the resulting Nyquist frequency,
  or that content folds onto the requested bins.  The monitor does not
  know the excitation bandwidth; sizing the interval from the
  analysis's `f_max` rather than the monitor's own frequencies is
  always safe.
- **Validated on the first recorded step**, the first moment a monitor
  sees `dt`, so a rejected interval stops the run immediately rather
  than after it has been paid for.  The stride is keyed on the absolute
  step index, so a resumed run samples the same instants.

Rejected alternative: a `samples_per_period` parameter, which would
carry the safety rule in its units.  It reads well but splits the
monitor vocabulary in two — and it still cannot see the excitation
bandwidth, so it would promise a safety it does not have.

**Measured** (33 180 cells, 3 000 steps, whole-volume E monitor, bins
at 10 and 12 GHz against an every-step reference):

| sampling | monitor overhead cut | max abs deviation / peak |
|---|---|---|
| 40 per period | 2 % | 2.9e-14 |
| 20 per period | 49 % | 3.5e-05 |
| 10 per period | 72 % | 1.0e-04 |

1e-4 is −80 dB on a field plot.  The 40-per-period row is the honest
one: on a mesh that fine the solver's own `dt` already oversamples, the
stride comes out 1, and there is nothing to win.

**What the interval can reclaim is exactly the solver's own
oversampling**, which is why the two tutorials that adopted it land
orders of magnitude apart.  Tutorial 13's step is set by a 0.635 mm
feed pin (`dt` = 0.125 ps against a 3.4 GHz band), so 20 samples per
period is a stride of **117** and the monitor goes from +79 % of the
page to inside the build-to-build scatter.  Tutorial 07's step is set
by the wavelength alone (`dt` = 2.98 ps against 12.4 GHz), so the DFT
is already sampled 27 times per period and the same rule gives stride
1 — nothing at all.  Dropping that page to 12 samples per period buys
a stride of 2, which is still 47 % of a +106 % overhead for a 1.6e-5
change in the field.  The rule of thumb for a caller: the finer the
feature driving `dt` relative to the band, the more an interval
returns.

**Consequence.**  Schema-additive `interval` attribute in
`fields_freq.h5` (absent means every step, so old files read
unchanged), carried by `result_dump` and rehydrated by the reader — a
stored DFT that cannot say how it was sampled cannot be resumed on the
same stride.  Gates: `test_monitors.py::TestFreqMonitorInterval`
(8 cases: default, value agreement against every-step, round-down,
rejection, warning, the 10-per-period rule staying silent, validation
against the *highest* frequency, absolute-step keying) and
`test_project_freq.py` (store round trip, and old files reading as
every-step).

---

## DD-141 — The section pool decides on a measured sample, not an estimate

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.

**Problem.**  The DD-102 prefill pool is admitted by two thresholds:
outstanding query count, and a work score that multiplies queries by
the queried shape's face count.  Face count is a proxy for cost, and
how good a proxy it is depends on the geometry class.  On a row of 60
small PEC posts it over-estimates by an order of magnitude — the fused
shape has many faces, but bbox prefiltering makes each section cheap —
so the pool was built for work that did not need it.  Measured on that
geometry: **12.1 s pooled against 6.9 s sequential**, i.e. the
"acceleration" cost 75 %.  The pool's own startup is the reason it
matters: eight fresh interpreters importing NumPy and OCC take ~5 s,
which is a floor no scheduling improvement can undercut.

**Decision.**  Keep both thresholds as *admission* tests, then measure
before committing.  `_sample_and_admit` computes
`_SECTION_SAMPLE_QUERIES` = 24 of the admitted queries in-process,
times them, and projects the remainder; the pool is built only when
that projection clears `_SECTION_POOL_STARTUP_S` (5 s, a property of
the machine) times a 1.5 margin.  Two details carry it:

- **The sample is drawn with a stride, not from the front.**  The
  schedule is deliberately cost-sorted, rarest axis first, so a head
  sample times the most expensive queries in the batch and would
  admit everything.
- **The sample is not overhead.**  It writes the same cache entries
  the pool would have written, so whichever way the decision goes,
  the work counts.

The split is deliberate: the geometry-dependent quantity is measured
per call, and only the machine-dependent one stays a constant.

**Measured.**  60-post fixture (29 592 cells), default settings:
**12.1 s → 5.3 s**, and `MAGNELIO_SECTION_WORKERS=1` now gives the
same 5.3 s — the pool correctly declines to build.  The stress case is
untouched where it pays: 501 lenses / 1002 primitives (149 583 cells)
runs **58.0 s sequential vs 33.3 s pooled**, and the pool is still
built.  A 151-lens middle case comes out neutral either way (8.9 vs
9.1 s), which is what a break-even test should do at break-even.

**Consequence.**  Gates in
`test_geometry.py::TestParallelSectionPrefill`: cheap batches stay
sequential with their sample cached, expensive batches are handed on,
sample and remainder partition the batch exactly, and the sample is
spread rather than taken from the head.  The pre-existing
bit-identity gate now also pins `_SECTION_POOL_STARTUP_S = 0`, without
which its deliberately cheap fixture would be declined by the new gate
and the test would silently compare sequential against itself.

---

## DD-142 — Deterministic ARPACK start in the 2D mode solver

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.  Closes KB-010.

**Problem.**  `test_coax_tem_vs_te_tm` failed once on 2026-08-10 and
again in the full run of 2026-08-12, roughly one run in twelve, with
no code in between to explain it.  `Numerical2DModeSolver` called
`eigsh(..., sigma=…)` without `v0`, so ARPACK started from a random
vector and the converged eigenvectors carried a run-dependent
residual.

The standing hypothesis was that the *basis* inside the degenerate TE
pair rotated run to run.  Measured over 30 rebuilds of the same port,
it does not: the basis angle came out 28.9° every single time.  What
moved was the residual — the cross-projection ratios the test asserts
on wandered over **3.1e-16 … 1.1e-13**, a factor of 70, against a
1e-12 gate.  Both ends are physically zero; the gate simply sat within
reach of the spread's upper tail.

**Decision.**  Pass a fixed generic start vector
(`np.random.default_rng(0).standard_normal(n)`) to both `eigsh` calls
in the module.  Generic rather than structured: a vector of ones can
sit orthogonal to a mode of interest and starve it.  The assertion
stays as it is — it was never wrong, it was measuring a quantity that
had no reason to be reproducible, and reproducibility is the fix.

**Measured.**  The same 30 rebuilds now return **identical** ratios
(5.61e-15 and 2.82e-14), a factor ~35 below the gate, and the test
passed 30 consecutive runs.

**Consequence.**  Mode profiles shift by their own convergence
residual (1e-14 class) against previously stored ones — below every
tolerance in the suite, and no longer a moving target.  KB-010 closed.

---

## DD-143 — Cross-sections draw what has no cross-section

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.

**Problem.**  `plot_cross_section` sliced the model's solids and
nothing else, because slicing is all it did.  Everything in a model
that carries no volume — `ThinWire` conductors, discrete ports, lumped
elements, face ports — returned an empty polygon list and silently
vanished.  Tutorial 08 showed the consequence plainly: a monopole
antenna rendered as an empty box of air, with neither the wire nor the
feed visible anywhere on the page.  The picture was not wrong about
the solids; it was simply not a picture of the model.

**Decision.**  Draw them from their *definitions* rather than from a
section, and let the cut's relationship to each feature choose the
mark:

- **along the cut → a line**, **through the cut → a hollow ring.**
  The two cases are geometrically different facts about the same
  wire, and drawing either as the other misleads.  The ring is hollow
  so a field plot underneath still reads through it.
- **A thin wire is drawn at fixed width, not to its radius.**  It is a
  sub-cell model by definition — to scale it would be invisible.  The
  radius is used for the only thing it can honestly decide: whether
  the cut passes *through* the wire (within one radius) or misses it.
- **Face ports become the domain edge they occupy.**  A port declared
  on a bbox face parallel to the cut is *not* drawn: it would cover
  the whole section and hide the geometry it is meant to annotate.
- **Feature colours sit outside the material palette** (wire, port and
  element each get their own).  A reader must never have to work out
  whether a coloured line is a lossy dielectric or a port.
- Everything is labelled, and `show_wires` / `show_ports` switch the
  two families off for a plain material section.

**Consequence.**  Tutorial 08's cut now shows the monopole and its
feed gap, and the page's plot moved *after* the port declaration —
before, the port did not exist yet when the figure was drawn, so no
amount of plotting could have shown it.  `GeometryModel` and
`Project` inherit the behaviour through their existing `**kwargs`
wrappers; a stored `LoadedGeometry` without ports degrades to the old
picture rather than failing (`getattr(..., "ports", ())`).  Gates:
`test_plot_geometry.py::TestVolumelessFeatures` — one artist per
feature, line-vs-ring by cut orientation, a miss drawing nothing, the
off switches, three distinct colours, and the parallel-face-port
abstention.

---

## DD-144 — `blend="tangent"`: a transition that leaves both faces squarely

**Date:** 2026-08-12
**Status:** Accepted — implemented, tested.

**Problem.**  `.lofted()` bridged two faces with `ThruSections`, which
has no concept of a tangent.  With only two sections its `ruled=False`
"smooth" mode is indistinguishable from `ruled=True` — measured
identical volumes to every digit on a stripline-to-coax transition —
because a spline through two profiles and a ruled surface through two
profiles are the same surface.  Both run straight from one profile to
the other, so wherever the two faces point in different directions the
solid meets them at a crease.  That is the normal case for a feedthrough:
an electrode ends on a face normal to *z*, the inner conductor it feeds
begins on a face normal to *y*.  The crease sits exactly where the
current crowds from the wide electrode onto the thin pin — a field
concentration where a real part is radiused, and a geometric
singularity the mesh then has to resolve.

Building the bend by hand was the only route, and a poor one: it means
inventing intermediate cross-sections that have no physical
definition, placing them by eye, and re-placing them whenever a
dimension changes.

**Decision.**  A third join mode on the verb, `blend="tangent"`, that
derives the bend from the two faces instead of from invented sections.
The end conditions are Hermite — leave each face along its own outward
normal — written as a cubic Bezier spine whose interior control points
sit at `centre + normal · tension · span`.  `MakePipeShell` sweeps one
face wire into the other along that spine, holding the profiles
perpendicular to it, so the solid meets both faces at a right angle by
construction rather than by fitting.

- **`blend=` replaces `ruled=`, rather than adding a flag beside it.**
  Three modes do not fit in a boolean, and `ruled=False, tangent=True`
  would be a state the caller can contradict.  Magnelio is unreleased,
  so the rename costs nothing and no deprecation path is owed.
- **The tension is public, and may differ per end.**  It is not an
  internal constant: at `1/3` the blend is a clean quarter bend, and by
  `0.7` the second control point has travelled behind the root of the
  originating solid and the profile bulges outward.  An asymmetric pair
  is what an asymmetric transition needs — stiff at the wide end, soft
  at the pin.
- **Corrected Frenet, not plain Frenet.**  A plain Frenet frame flips
  its normal at an inflection point, which a Bezier spine between two
  arbitrarily posed faces easily has.  On a planar spine, corrected
  Frenet and a fixed binormal agree to every digit; the corrected frame
  is the one that also holds up when the two faces are not coplanar,
  which is the general case Magnelio has to serve.
- **The normal's sign comes from the face's orientation in its solid,
  not from `face_plane_normal`.**  That helper deliberately forces a
  reproducible sign (largest component positive) so an offset direction
  is predictable; a blend needs the direction that points *out of the
  body*, which only `TopAbs_REVERSED` can tell it.  Hence the separate
  `face_outward_normal`.
- **`tension=` is rejected for the other two modes** rather than
  silently ignored, so a caller who sets it without switching the mode
  hears about it.

**Consequence.**  `Loft` (DD-134) takes the same `blend=` argument but
only its two section-based values; asking it for `"tangent"` names the
verb that can do it.  The analytic bbox padding scales with the tension
(`1.5 · max(tension)`), since the bow grows with it.

Gates: `TestTangentBlend` in `tests/unit/test_modifications.py`.  The
load-bearing one is `test_leaves_the_start_face_along_its_normal`,
which measures the sideways drift of the section at three depths and
asserts it grows as the *square* of depth — that exponent is the right
angle, since any residual tilt would add a first-order term and pull it
towards 1.  Measured 24.6 µm / 99.9 µm / 409 µm at 0.5 / 1 / 2 mm along
a 21.6 mm span: ratios 4.06 and 4.09.

---

## DD-145 — Structure diagrams: four views, because one would lie

**Date:** 2026-08-13
**Status:** Accepted — implemented.

**Problem.**  The package is ~43,000 lines across 20 subpackages and
149 classes with no overview of its own shape.  The obvious remedy —
an inheritance diagram — turns out to be the least informative one
available here: below `Shape` the hierarchy is 27 classes at a maximum
depth of **2**, so the picture is a star that says "everything is a
Shape".  What the code actually organises itself by is elsewhere: the
import graph between subpackages, and the composition tree that a
geometry expression forms at run time.

**Decision.**  One developer tool, `validation/tools/draw_structure.py`,
with four subcommands (`packages`, `inheritance`, `composition`,
`classes`).  It writes DOT directly and shells out to the system `dot`;
no third-party package is added, and nothing enters the published
documentation.  Each view also has a `--stats` text mode, which is what
makes the tool checkable rather than merely decorative.

- **`if TYPE_CHECKING:` is not a dependency.**  Counting those imports
  as load-time edges invented a `geo ↔ mesh` cycle that cannot occur:
  the block never executes.  With them excluded, the package has
  **exactly one** module-level cycle, `analysis ↔ io`.  The guard is
  recognised in `check_imports._walk_scoped`, so both tools agree.
- **Deferred imports are drawn separately, and off by default.**
  Magnelio deliberately defers 37 of its 87 inter-package imports into
  function bodies to break cycles.  Drawing them like load-time edges
  misrepresents the architecture; drawing them at all buried the graph,
  so `--deferred` opts in.
- **Layering runs on the condensed graph.**  Tarjan's SCC first, then
  longest-path on the resulting DAG.  A recursive depth over the raw
  graph is path-dependent wherever a cycle exists and put `mesh` in the
  bottom layer on one run and not on another.
- **Composition is drawn as arity, not as class-to-class edges.**  The
  child fields are all annotated `object`, so no static analysis can
  name the child's class.  What *is* derivable — and exactly — is which
  fields a class recurses into, by finding the `self.<field>._occ_shape()`
  calls it makes (plus a comprehension form for the n-ary containers).
  That yields leaf / unary / binary / n-ary, which is the concept the
  inheritance diagram fails to show.
- **Internal modules are imported too.**  A map that omits `_operators`
  and `_backend` hides the parts hardest to learn from the public API.

**Consequence.**  Verified against numbers measured independently of the
tool: 20 subpackages, 50 module-level edges, 37 deferred-only, one cycle;
27 `Shape` subclasses at depth 2, 15 internal; 19 classes recursing into
child shapes (13 unary, 3 binary, 3 n-ary).  Output goes to
`validation/results/structure/`, which is git-ignored — the images are
regenerable and would otherwise churn on every rename.

---

## DD-146 — Booleans never edit the shapes they are given

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.

**Problem.**  OCCT's Boolean kernel defaults to
`BOPAlgo_Options::NonDestructive = false`: an operation is allowed to
edit its *argument* shapes in place — raise edge and vertex tolerances,
insert p-curves, shift vertices — and it uses that freedom routinely.
Magnelio never set the option, in any of its four Boolean call sites.

Two of the library's own properties turn that default into a
correctness bug.  `cached_occ_shape` hands out **the same**
`TopoDS_Shape` on every call, so an edit is permanent rather than
scoped to one operation.  And an OCCT Boolean *result* shares the
sub-shapes it did not have to modify with its operands, so an edit
made to a derived solid reaches back into the body the user built.  A
mesh build takes thousands of `BRepAlgoAPI_Section` cuts through those
shapes, and the tolerance creep from each one accumulates.

Measured on a stripline-coupler assembly (internal record:
`investigations/boolean-operand-mutation`), one mesh build moved the
maximum edge tolerance of the vacuum body from 1.1e-4 m to 7.0e-3 m —
7 mm of fuzz on a model with 1 mm electrodes.  Intersecting that body
with the x>0, y>0 quarter space afterwards no longer returned the
quarter of a 90 mm pipe but a 4 mm coax stub: a second model built
from the same bodies meshed one cell across x and looked empty.
Nothing announced the degradation.  `bounding_box()` stayed exactly
right throughout — tolerance moves no geometry, only the fuzzy zone
the Boolean kernel resolves seams in — so the failure was visible only
in `BRep_Tool::Tolerance`.

**Decision.**  Every Boolean the library runs calls
`keep_operands_intact()` (i.e. `SetNonDestructive(True)`) before
`Build()`.  The named helper carries the rationale so the call cannot
be mistaken for a tuning knob and dropped.  It covers all four sites:
`_run_bop` (every CSG Union/Intersection/Difference),
`_face_region_wires`, the section in `cross_section_polygons`, and the
pair test in `check_pairwise_overlaps` — the last one rebuilt from the
two-shape constructor to the explicit
`SetArguments`/`SetTools`/`Build` sequence, since that constructor
builds before the option could take effect.

The alternative — deep-copying the cached solid before every Boolean —
was rejected: it pays a copy on every call to avoid an edit that
usually does not happen, where the kernel copies only the sub-shapes
it would actually have modified.

**Consequence.**  On the coupler case the tolerance now holds at
1.105e-4 m across a mesh build and the quarter-space cut stays the
quarter-space cut.  The meshes are identical with and without the
option (43 x 68 x 101 either way), and the build costs the same
(16.4 s vs. 17.1 s) — this buys correctness, not accuracy, and gives
up no speed.

One measured behaviour change, all of it inside the already-ill-posed
case: a *plain* section on a plane lying in a face returns less than
before (the two-brick seam of DD-137 went from half the strip to a
quarter).  The inflated tolerances had been smearing the degenerate
seam enough to catch more of it — accidentally, and no closer to the
right answer, which is the whole strip.  No production caller is
affected: `plot_cross_section` opts into `exact_at_faces` (DD-137,
still exact at 40e-6), and the mesher never takes the exact cut on a
degenerate plane — it detects tangency per face and re-sections a step
to either side.  `test_plain_section_loses_material_at_a_seam` was
asserting the old figure as if it were a contract; it now asserts what
it means (area comes back short), since neither figure is correct.

Gate:
`tests/integration/test_boolean_operands_intact.py` — edge tolerances
unmoved across a mesh build (fails without the option), and a second
model off the same bodies meshing to the same grid.

---

## DD-147 — An edge with no electric energy must not run the solver

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.

**Problem.**  `build_M_eps` gives a cat-2 edge whose `eps_avg` is 0 the
value `M_eps = 0`.  Such an edge lies wholly inside a conductor — the
sub-cell classifier left it cat-2 and *unmasked* instead of cat-3 — so
it stores no electric energy at all.  Two places read that 0 as if it
were a physical permittivity:

- `compute_min_effective_eps` returned **0**, and `courant_dt`'s
  `max(min_effective_eps, 1e-6)` guard turned that into
  `eps_factor = 1e-3`.  The guard only prevents a division by zero; it
  does not rescue the time step, it pins it three decades below the
  geometric Courant limit.
- the E-side update coefficients divided straight through `M_eps`, so
  `alpha_E` and `beta_E` came back NaN on those edges, and the NaN
  reached every field component on the first step.

Neither said anything.  Measured on a stripline-coupler quarter model
(internal record: `investigations/degenerate-conformal-edges`): 99 of
383 818 edges were degenerate, `dt` came out at 6.59e-17 s against a
geometric limit of 6.58e-13 s, and a 44 200-step run therefore covered
0.003 ns — the 1 GHz excitation had not started, the stored energy sat
at its initial value, and `a(t)`/`b(t)` were NaN.  It reads exactly
like a converging run that needs more steps.

**Decision.**  Treat `M_eps = 0` the way the H side has treated
`M_mu = 0` since DD-081: the edge is *frozen*, not solved.

- `compute_min_effective_eps` minimises over `eff_eps > 0` only.  A
  frozen edge cannot go unstable, so it has no business setting the
  stability limit.  This is the E-side counterpart of the 1 %
  `A_face_free` floor that `compute_min_effective_mu` already mirrors
  from `build_M_mu` — the comment there predicts this failure mode
  ("would shrink dt to the courant_dt internal lower bound and turn a
  20-second test into a multi-hour one"); only the E side lacked it.
- `alpha_E` / `beta_E` are built through `np.where(M_eps > 0, …)`,
  giving frozen edges `alpha = 1`, `beta = 0` — the same pair
  `alpha_H`/`beta_H` use.  Masking the edge would have produced the
  same behaviour; NaN never was an option.

The root cause — a classifier that leaves a fully-conducting edge cat-2
and unmasked — is deliberately *not* addressed here.  Changing the mask
moves the conductor contour the 2D mode solver and the port detection
read; the solver-side guard is the conservative half and is correct
independently of how the edge came to be.

**Consequence.**  On the coupler case `dt` returns to 2.60e-14 s (395x),
the divide warnings are gone, and the power waves carry signal instead
of NaN (`max|a| = 5.99e-07` where it had been NaN).  Meshes without
degenerate edges are unaffected — `eff_eps > 0` holds everywhere and
`np.where` reproduces the previous arithmetic exactly.  Gates:
`tests/unit/test_degenerate_edge_stability.py` — the minimum and the
time step both survive one planted degenerate edge, the coefficients
stay finite, and that edge comes out frozen rather than NaN.

---

## DD-148 — A sub-face port is drawn as its window, not as the whole wall

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.

**Problem.**  `_draw_face_port` (DD-143) drew every bbox-face port as a
line spanning the full domain edge, because it read the *domain* box
and never looked at `port.bbox`.  A coax port 3.5 mm across on a 45 mm
wall was drawn 45 mm wide, and two ports on one face — the normal
arrangement for a stripline coupler, one feed at each end — landed on
top of each other as the same line, with the second label hidden under
the first.  Worse, a cut nowhere near either window still drew both:
the picture asserted a port on a stretch of wall that is plain wall.

**Decision.**  The window is the port.  `_draw_face_port` reads
`port.bbox` in the documented global tangential-axis ordering, and

- **abstains when the cut misses the window** on the cutting plane's
  own axis — there is no port there to draw;
- **spans the window, clipped to the domain**, on the remaining axis.

`bbox=None` still means the whole face, so the full-face case is
unchanged.

**Consequence.**  Each feed of a two-port face is drawn on the cut that
actually passes through it, at its true width.  Gate:
`tests/unit/test_plot_geometry.py::TestVolumelessFeatures` — a window
spans 4 mm rather than the 40 mm domain, a cut between two windows
draws nothing, each window appears on its own cut, and a port without
a bbox still spans the whole edge.

---

## DD-149 — The free-area floor is a threshold, not an equality test

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.
**Supersedes** the cat-2 half of DD-147 (the DD-147 guards stay as a
backstop for the remaining paths to `M_eps = 0`).

**Problem.**  DD-147 guarded the case `eps_avg == 0` exactly.  The
conformal classifier does not always produce a clean zero.  On the same
coupler model at a 0.25 mm cell size (internal record:
`investigations/degenerate-conformal-edges`) it produced 10 edges at
exactly zero *and* three more at `eps_avg ≈ 3.2e-15` — the same
physical situation, a dual face whose free area has collapsed, but with
a rounding remainder instead of a zero.  An equality test walks past
those three; they alone held `min_effective_eps` at 7.2e-15 and `dt` at
5.95e-17 s against a geometric limit near 6e-13 s.  The user sees a run
that does not advance, and the workaround — nudge `MeshControl` until
the pathological edges happen not to appear — is not a workaround at
all, since which cell sizes are safe cannot be predicted.

`build_M_mu` has had the right shape of guard since the Krietenstein
reduction landed: a **threshold** on the free-area fraction,
`A_face_free > 1 % · A_face`, below which the sub-cell formula is not
applied.  The E side had no equivalent.  `EdgeMaterialData` already
carries `f_A`, the exact E-side analogue (free dual-face area / dual
area) — nothing needed measuring, only reading.

**Decision.**  Both mass matrices floor on the same constant,
`_FREE_AREA_FLOOR = 0.01`, mirrored by both effective-material helpers:

- `build_M_eps` applies the cat-2 sub-cell formula only where
  `f_A > 0.01`.  NaN `f_A` (an edge carrying no sub-cell data) compares
  False and is floored.
- `compute_min_effective_eps` mirrors the same condition, leaving
  floored edges at the 1.0 default so they stay out of the minimum.

**What replaces the formula differs per side, deliberately.**
`build_M_mu` hands a floored H-face its bulk-staircase value, which is
sound there because a floored H-face is Faraday-dead — its circulation
edges sit inside the PEC mask, so `C e = 0` and `h` stays 0 either way
(measured neutral to 1e-15 by the DD-058 donor-trigger benchmark).  An
E-edge carries no such guarantee: the bulk value would let a curl drive
an edge lying inside a conductor, where E = 0.  Floored E-edges are
therefore *frozen* (`M_eps = 0`), which routes them through the
`alpha_E = 1` / `beta_E = 0` branch DD-147 already built.

**Consequence.**  On the coupler quarter model the collapse is gone
across the mesh-control range that used to trigger it — `min_cell_size
= t/4` moves from `eps_min = 7.2e-15`, `dt = 5.95e-17 s` to
`eps_min = 0.118`, `dt = 2.04e-14 s` (343x); the `min_cells_per_feature
= 6` case moves 195x.  Configurations that were already healthy are
bit-identical (`t/3`: 0.0681 before and after; `min_cells_per_feature
= 8`: 0.0465 before and after), and the full suite is unchanged.

The threshold is not a judgement call on this geometry: counting the
affected edges (`investigations/degenerate-conformal-edges/floorcount.py`)
gives 0 floored of 4682 cat-2 edges at `t/3` and 0 of 7794 at
`min_cells_per_feature = 8` — those two meshes are bit-identical, not
merely close — against 13 of 4464 at `t/4` and 80 of 6934 at
`min_cells_per_feature = 6`.  Where it does fire, the largest floored
`f_A` is 5e-13 and the smallest surviving one 1.6e-02: **eleven decades
of gap**, so any threshold between 1e-12 and 1e-2 selects the same
edges.  A mesh whose thinnest sub-cell edge retains 1–2 % free area
keeps its reduction — the floor is not a general-purpose clamp on small
`ε_eff`.
Gate: `tests/unit/test_degenerate_edge_stability.py::TestNearlyDeadEdgeIsFlooredToo`
— a planted 3.2e-15 edge leaves the time step where it was, comes out
frozen, a 2 % edge keeps its reduction, and the three copies of the
constant are pinned to each other.

## DD-150 — The time step comes from the measured spectral radius, not the worst-case product

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.

**Problem.**  `courant_dt` scaled the geometric Courant limit by
`sqrt(eps_min · mu_min)` — the worst conformal edge times the worst
conformal face, as if both coincided with the smallest cell.  They
never do.  On the stripline-coupler quarter model (internal record:
`investigations/conformal-cfl/`) the heuristic held `dt` at 2–4 % of
the geometric limit while the exact stability limit sits at 67–75 %:
a factor **17.2x** (`min_cell_size = t/2`) to **34.2x**
(`t/8, min_cells_per_feature = 8`) of runtime thrown away.  The
`mu_min ≈ 0.0103` driving it is structural — any curved conformal wall
produces sliver H-faces just above the DD-149 1 % floor — so every
curved model paid the decade, and a 30 ns run needed 0.8–4.6 million
steps where ~45 000 suffice.  DD-147/DD-149 capped this collapse at
the floor; this entry removes the remaining decade by measuring
instead of estimating.

**Decision.**  `spectral_dt(mesh, accuracy, m_eps=, m_mu=)`
(`solver/stability.py`) computes the sharp leapfrog criterion

    dt_max = 2 / sqrt(lambda_max(M_eps^-1 C^T M_mu^-1 C))

on the live DOFs (PEC-masked and frozen `M_eps = 0` edges removed,
frozen H-faces via the exact `1/M_mu = 0` — the operator the solver
actually iterates).  `lambda_max` is measured by matrix-free Lanczos
(`eigsh`, k=1, LA, tol 1e-8) on the symmetrised operator
`D^-1/2 A D^-1/2`; the row-sum (Gershgorin) bound — computed with the
absolute-valued factors, strictly an upper bound on the spectral
radius — serves as fallback when Lanczos fails and as certified
ceiling (a Lanczos value above it is clamped).  Lanczos converges
from below; the developer accepted the standard 0.95 safety factor as
sufficient cover for the 1e-8-tolerance residual (2026-08-13).  The
measured `lambda_max` is cached on the mesh (`_spectral_lambda_max`),
so `solve_ports()` + `run()` pay one eigensolve; `accuracy` only
scales the safety factor.  `AnalysisScatteringTD` uses `spectral_dt`
at both dt sites; `courant_dt` stays for geometric estimates (port
refinement, step budgeting) and as the vacuum fallback when a mesh
has no live update operator.

**Sharpness, measured** (coupler t/2 mesh and the certificate
fixture): a bare matrix leapfrog with the production operators is
stable at `0.999 · dt_crit` (bounded noise, growth ~2) and blows up
before step 200 at `1.02 · dt_crit`.  The Gershgorin fallback
delivered 84 % of the exact step on the coupler, 88 % on the
certificate fixture.  Where `lambda_max` lives: the top eigenvector
concentrates on cat-2 sliver edges (`f_A` 0.02–0.5) — lifting them
further (enlarged cells, DD-058 machinery) could recover the residual
1.3–1.5x to the geometric limit and stays a possible follow-up, no
longer the fix.

**Scope of re-validation** (developer-agreed): the measured operator
is the lossless volume update; sigma/sigma* losses, pole-residue ADE,
SIBC, thin wire, lumped elements and the Mur/DTBC port updates also
depend on dt and had only ever run at the throttled step.  The first
integration run surfaced 19 failures in exactly these paths, all
diagnosed, none a physics regression:

1. **14 bit-exactness gates** (resume, streamed-vs-in-RAM, GPU
   staging, dispersive resume, thin-wire T3/T6): ARPACK's default
   random start left a run-to-run residual in `lambda_max`, so two
   builds of the same mesh got a dt differing in the last bits — the
   KB-010/DD-142 lesson replayed on the CFL path.  Fixed in
   `_measure_lambda_max` with the same deterministic generic start
   vector (`default_rng(0)`); dt is bit-identical across rebuilds.
2. **`test_interval_survives_the_store_round_trip`**: the DD-140
   fixture encoded its stride-2 interval as an absolute 8 ps against
   a dt of 3.45 ps; the spectral step is 4.12 ps (+19 % on a plain
   vacuum brick — the per-axis-minima geometric bound is conservative
   even without conformal cells) and the floored stride silently
   became 1.  The test now derives the interval from the measured
   step (`2.5 * spectral_dt`).
3. **Lumped-port co-temporal guard**: the teeth-check
   `|S_cotemporal - S_temporal| > 1e-3` sat on the measured value of
   one dt (1.50e-3 at 1.83 ps; 8.45e-4 at 2.01 ps — the separation
   moves with the fixture response, not just with omega*dt).  Floor
   lowered to 1e-4, still five decades above the 1e-9 equality gate
   it protects.
4. **SIBC end-to-end windows**: the 2 GHz band-edge point moved in
   both gates of the parallel-plate fixture on a 1 % dt change — the
   band edge carries the least excitation energy and the smallest
   alpha*L.  Alpha ratio 0.986 -> 0.9607 (converged; longer runs do
   not move it; interior stays in the 0.988–0.999 class, |S11| floors
   unchanged), window floor re-measured to 0.95; monitor/balance
   ceiling 1.13 -> 1.15 (band edge 1.139, interior at the rim-strip
   1.10–1.11 class).  The analytic anchor remains the
   dt-parameterised SIBC unit layer.

After the fix and the three fixture recalibrations the unit suite
(1746) and the full integration suite run green at the enlarged step.

**Known cost: this is the memory peak of a run** (measured
2026-08-21, coaxial line with two ports, RSS and `ru_maxrss` per
phase; internal record `investigations/fit-td-bandwidth/`).  At 2.58 Mcells the
process peaks at 4.39 GB, of which `spectral_dt` contributes 2.37 GB —
against 1.30 GB that stays resident and 0.072 GB/Mcell for the field
arrays the time loop actually iterates.  Meshing (0.57 GB/Mcell) and
the material matrices (60 MB, and they are the ones built in float64)
are not the driver, contrary to the obvious guess.  Two posts:

| post | per Mcell |
|---|---|
| ARPACK Krylov basis, default `ncv` = 20 vectors | ~0.49 GB |
| `C_abs = C.copy()` — full copy of the curl matrix | ~0.15 GB |

The peak is transient — it is gone before the first time step — so on
a machine with generous swap it costs page-outs, not a failed run,
which is why nothing was changed here.  Both posts are reducible, at
different prices.  Sharing `C`'s index arrays instead of copying the
structure, and dropping `C_abs` once the bound is in hand, is
bit-identical (`dt` and `lambda_max` unchanged to the last digit) but
measured only **3.6 %** off the peak — not worth touching production
code for.  Lowering `ncv` is the real lever, worth about half the
peak, and the one thing that cannot be done quietly: ARPACK then
converges on a slightly different value inside the same tolerance,
`dt` shifts, and the bit-identical rebuild this entry's fixed `v0`
exists to guarantee is gone — with it the project store's exact
resume.  Take it up only as a deliberate compatibility decision.

**Gates:** `tests/unit/test_spectral_dt.py` (stable at the measured
step, unstable 5 % above the limit, dominates the heuristic on a
conformal fixture, vacuum box stays at the geometric value, cache
skips the second eigensolve, Lanczos failure falls back safely);
`validation/spectral_dt_certificate.py` (waste factor, empirical
sharpness bracket, Gershgorin never below Lanczos).

**Files:** `solver/stability.py` (`spectral_dt`,
`_measure_lambda_max`), `analysis/scattering_td.py` (both dt sites),
`solver/__init__.py` (export), `docs/methods/fit-discretization.md`
(stability section rewritten around the algebraic criterion).

## DD-151 — Face planes outrank bounding-box extents in the plane clustering

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.  Closes KB-013.

**Problem (KB-013).**  OCCT Booleans on interpenetrating operands
return a bounding box inflated by `Precision::Confusion` (1e-7 model
units) beyond the true geometry — the coupler's coax stub reported
ymax = 75.000100 mm against the real end face at 75.000000 mm.  The
mesher collected both as untagged critical planes and `_snap_planes`
clustered them to the midpoint, 75.00005 mm: the domain boundary sat
50 nm past the material surface, the last cell carried a sliver fill
factor of `1 − 1.11e-5`, `_port_chain_slab_defect` measured exactly
that against the 1e-8 tolerance, and every port channel on the face
fell back to modal Mur-1st (−30 dB class instead of the −124…−166 dB
DTBC floors) — with a misleading warning blaming the (perfectly
translation-invariant) feed.

**Decision.**  Critical planes carry provenance from extraction to
clustering.  `extract_critical_planes_per_shape` returns
`(position, exact)` pairs — `exact = True` for planes read from an
analytic face surface (`_face_critical_planes`), `False` for shape
bounding-box extents.  `_snap_planes` collapses a cluster containing
at least one exact member to the midpoint of its *exact* members
only; bbox extents inside the cluster are absorbed without
influencing the position.  Clusters without any exact member
(silhouettes of tilted / free-form faces, which only the bbox covers)
keep the symmetric midpoint — as does a cluster of several exact
planes, so float-wiggle between two real faces behaves exactly as
before.  Thin-sheet positions and wire vertex coordinates count as
exact; wire bbox extents as approximate.  Bare floats normalise to
exact in `_merge_axis_planes`, keeping the historical unit-test
behaviour byte for byte.

**Blast radius, measured.**  The change can only move clusters that
mix face planes with bbox extents — precisely the Boolean-inflation
case.  All-exact and all-bbox clusters reproduce the historical
midpoint, and the full unit + integration suites pass unchanged.  On
the coupler quarter model the grid stays 27 x 29 x 93; the slab
defect the DTBC gate measures (`M_mu(Hx)`, ymax face) drops from
**1.11e-5 to 1.14e-11** — three decades below the 1e-8 tolerance —
and `solve_ports` builds both coax channels with the exact DTBC, no
slab warning, no Mur fallback (internal record:
`investigations/degenerate-conformal-edges/`, `planes.py` now prints
the provenance tags).

**Rejected alternatives** (recorded in KB-013): (b) trimming the
confusion-sized inflation off Boolean bounding boxes — risks cutting
real geometry that genuinely ends within 1e-7 of the box; (c) scaling
the DTBC gate with the measured defect — DD-066 shows the
defect-to-reflection relation is not linear, so the gate would need
its own measurement campaign; the geometry was correct and the grid
wrong, so the grid is what had to move.

**Gates:** `tests/unit/test_mesh.py::TestPlaneProvenance` — a mixed
cluster snaps onto the face plane bit-exactly, a bbox-only cluster
keeps its midpoint, faces-only midpoints ignore a bbox member, and
the end-to-end interpenetrating-union fixture (the coupler's failure
geometry, minimal) lands its grid boundary on the analytic face to
1e-12.

**Files:** `geo/_occ_backend.py` (`extract_critical_planes_per_shape`
provenance, aggregate strips tags), `mesh/mesher.py` (`critical_raw`
tagged, `_merge_axis_planes` normalisation, `_snap_planes`
exact-member collapse), `tests/unit/test_mesh.py`.

## DD-152 — Geometric queries never read triangulation

**Date:** 2026-08-13
**Status:** Accepted — implemented, tested.  Closes KB-012.

**Problem (KB-012).**  `GeometryModel.plot()` changed the mesh of a
model built afterwards (`N_y` 68 -> 75 on the coupler, reproducible),
while edge/face/vertex tolerances stayed bit-identical — the DD-146
class was ruled out by measurement, and the culprit remained "state
the renderer leaves behind somewhere other than the tolerances".

Found by measuring the mesher's *input*: the renderer
(`JupyterRenderer.DisplayShape` via `ShapeTesselator`) tessellates the
cached solids in place, and `BRepBndLib::Add` defaults to
`useTriangulation = True` — with a triangulation present, a face's
bounding box comes from the triangle nodes enlarged by the
tessellation deflection and differs from the analytic box by whole
tenths of a millimetre (measured 0.039 mm at 2 mm deflection on a
5 mm cylinder patch).  The plane-extraction trim filter
(`_face_critical_planes`: "keep tangent candidates inside the trimmed
face's bbox") then admits tangent positions of surface regions the
trimmed face does not cover — the probe showed two extra face planes
at y = 47.805 mm on the coupler after tessellating the bodies, and
Boolean results reuse (triangulated) input faces, so fresh
intersections built after a plot inherit the leak.  The mesh then
depends on whether the model was viewed first: a convergence study
run either side of a `plot()` call compares two discretisations.

**Decision.**  Every OCC bounding-box read that feeds meshing,
classification or construction passes `useTriangulation = False`:
the trim filter in `_face_critical_planes`, the face/edge bbox
indexes of the conformal classifier's OCC routing, the face boxes of
the ray-casting helper, and `_occ_bbox_diagonal`.
`shape.bounding_box` already used `AddOptimal(..., False, False)`.
For shapes without triangulation OCC uses the geometry either way, so
un-plotted paths are bit-identical by construction; the change only
removes the rendering-dependent branch.  The ParaView export's
deflection heuristic keeps the default — it feeds a display
tessellation, not geometry.

**Verified.**  `plottest.py` (internal record:
`investigations/boolean-operand-mutation`, plus the new `facebbox.py`
probe): N = 43 x 68 x 101 for all of PRE = none / plot3d / xsec /
both — the 68 -> 75 asymmetry is gone.  Critical planes of both the
cached and freshly built intersections are identical before and
after tessellation; the probe geometry meshes 37 x 62 x 104 with and
without a prior tessellation, bit-identically.

**Gate:**
`tests/unit/test_geometry.py::TestGeometryQueriesIgnoreTriangulation`
— a cylinder patch trimmed 10 µm short of its tangent keeps the
tangent candidate out of the plane set before and after an in-place
`BRepMesh_IncrementalMesh` at 2 mm deflection (the triangulated box
would admit it; the analytic box never does).

**Files:** `geo/_occ_backend.py` (five `Bnd_Box` call sites),
`tests/unit/test_geometry.py`.

## DD-153 — One vocabulary for boxes, planes, anchors and names across the public API

**Context.**  The pre-v0.1.0 API froze several historical dialects
side by side: box regions were spelled `corners=` (monitors),
`bbox=` (declarative ports, 2D tangential frame), `tf_sf_box=`
(plane-wave source, silently degrading swapped corners via
`searchsorted().clip()`), and `(p1, p2)` (`Brick.from_corners`);
axis-normal planes appeared as `normal`/`offset` (`Face`),
`normal`/`position` (`mirrored`, monitor plots),
`plane_normal`/`plane_position` (`plot_cross_section`),
`("z", pos)` tuples (`MonitorWallLoss.reference_plane`) and a
degenerate corners box (`MonitorFluxTime`); radii mixed
`inner_radius` with `r_bottom`/`r_major`; ports carried `label`
where shapes and monitors carried `name`; `Cylinder.origin` and
`Curve.helix(center=)` named the same anchor differently; and the
shape verbs disagreed on positional vs. keyword-only for identical
parameters (`translated(vector)` vs. `extruded(*, vector)`).
A final API review before freezing MINOR-stability caught the lot;
since nothing is released, all of it was changed at once, without
deprecation shims.

**Decision.**  Canonical vocabulary, applied everywhere:

- **Box regions** are `corners=` — two opposite corner points in
  world coordinates, any order, `None`/`±inf` components reaching
  the domain boundary (or the documented default) on that side.
  Applies to `MonitorFieldTime`/`MonitorFieldFrequency` (unchanged),
  `PlaneWaveSource` (was `tf_sf_box`, now normalises corner order)
  and `PortWaveguide` (was 2D `bbox`; corners are projected onto the
  port face — differing normal components raise as an axis mix-up).
  The range spelling `from_ranges(x1=, x2=, dx=, …)` exists on
  `Brick` (strict: exactly two per axis) and as classmethods on the
  two field monitors and the plane-wave source (lenient: open axes
  allowed); shared resolver in `geo/_ranges.py`.
  The spec layer keeps the tangential-2D window under the new name
  `window=` (`PortSpecNumerical`/`PortSpecMultiConductor`,
  `PortPlane.from_mesh`); `window_from_corners`/`point_on_face` in
  `ports/declarative.py` do the projection.
- **Axis-normal planes** are `normal=` + `position=` as a kwargs
  pair (`Face` — was `offset`; `plot_cross_section` — was
  `plane_normal`/`plane_position`; monitor `plot()`s unchanged), or
  a `(normal, position)` tuple where the plane is one value among
  several parameters (`MonitorFluxTime.plane` — was a degenerate
  corners box; `MonitorWallLoss.reference_plane` unchanged).
- **Radii** are written out: `Cone(bottom_radius=, top_radius=)`,
  `Torus(major_radius=, minor_radius=)`; `radius`/`inner_radius`/
  `outer_radius` unchanged.
- **Identity is `name`** everywhere: declarative ports, port specs,
  `LumpedElement`, `Mode`, result accessors (`port_names`).
  `label` survives only as matplotlib legend vocabulary and on
  `Signal1D` (a display label, not an identity).
- **Anchors:** `Curve.helix(origin=)` (was `center=`) matches
  `Cylinder.origin`; `Brick.origin` stays the min corner (industry
  convention), `center` stays the true centre (`Sphere`, `Torus`).
- **Verbs** take their one geometric core argument positionally,
  options keyword-only: `mirrored("x", position=…)`,
  `revolved("z", 180.0)`, `extruded((0, 0, h))`,
  `shelled(2e-3, …)`, `thickened(35e-6, …)`; keyword calls remain
  valid.  `chamfered(distance=)` (was `dist=`).
- **`PortAnalytical`**: `family=` (was `type=`, shadowed the
  builtin and collided with the project-store envelope key),
  `width=`/`height=` (were `width_a`/`height_b`), `center=` is a 3D
  world point projected onto the face.
- Google-style docstrings in public modules converted to NumPy
  style (`primitives`, `material`, `grid`, `plane_wave`,
  `boundaries/*`, `GeometryModel.add`).

Deliberately deferred: unifying `PlaneWaveSource`'s
`waveform`/`f_center`/`f_max` with the modal `ExcitationSpec` — that
is a semantic change (no `sine` family, no amplitude in
`ExcitationSpec`), not vocabulary, and needs its own design pass.

**Consequences.**  Old project stores (flux-monitor `corners` attrs,
port-spec `bbox`/`label` recipe keys) do not rehydrate; acceptable
pre-release, no migration shim.  All consumers (tests, examples,
docs 01–14, validation, benchmarks, private workspace scripts) were
swept in the same change.

**Files:** `geo/` (primitives, shape, curves, modifications,
`_ranges.py` new, `_occ_backend`), `monitors/` (flux, field_time,
field_frequency), `sources/plane_wave.py`, `ports/` (declarative,
base, recorder, `_modal/*`, `_lumped/*`), `circuit/element.py`,
`mesh/mesher.py`, `analysis/` (scattering_td, _recipe,
result_interface), `io/project.py`, `post/*`, `boundaries/*`,
`materials/material.py`.

---

## DD-154 — Symmetry planes are boundary declarations with a mesh-time domain clip

**Date:** 2026-08-13
**Status:** Accepted; stages A–F shipped (declaration + domain clip,
full-model port reports, mirrored plots/overlays, wall-loss
fractions, ParaView Reflect) and certified.  The initially deferred
excitation power semantics are resolved in [[DD-155]] (full-model
watts).

**Context.**  Exploiting mirror symmetry (the standard half/quarter/
eighth-model workflow of the large EM suites) previously required the
user to halve the geometry by hand — a loop of Boolean intersections
against a half-space box — and declare `{"xmin": "PMC"}`.  That
carried the DD-146 operand-mutation and KB-013/DD-151
tolerance-inflation risks straight onto the symmetry plane, halved
every reported port impedance and field plot without correction, and
scaled poorly for models with hundreds of primitives.

**Decision.**  A symmetry plane is a *boundary declaration*, not an
analysis option.  Physically it IS a PEC/PMC wall — the solver core
is untouched (the PMC wall is the natural BC of the free curl
operators, PEC is the existing edge mask) — plus the semantic "the
mirror image of the model exists beyond this wall", carried as
metadata that symmetry-aware readers interpret:

- **Declaration** extends the DD-103 closure vocabulary: type strings
  `"SymmetryPEC"` / `"SymmetryPMC"`, or a `Symmetry(kind,
  position=)` instance (public in `magnelio.boundaries`).
  *(Vocabulary reworked by DD-159 before v0.1.0: the class is gone;
  bare clip strings default to plane 0.0, tuples carry the position,
  `ForceSymmetry*` is the as-built form.  Semantics unchanged.)*  On
  normalisation the `BoundaryConditions` face field keeps the
  *physical* wall type — every existing consumer that dispatches on
  the type (`getattr(bc, face) == "PMC"` in the flux half-weights,
  the TEM Laplace path, the port plane; `bc_type_entries` everywhere
  else) keeps working unchanged — and the symmetry semantics move
  into the canonical `BoundaryConditions.symmetry` map, read through
  the new `symmetry_entries()`.  At most one symmetry face per axis
  (two parallel mirror planes describe an infinite image chain).
- **Domain clip** (mesher, `Mesh.from_geometry`): a declared
  `position=` clips the computational domain to the kept half-space
  before plane clustering — critical planes on the discarded side
  (including the clustering band around the plane, so the position
  survives verbatim) are dropped, the symmetry plane enters as an
  *exact* face plane (winning the KB-013 clustering), and forced
  planes beyond it drop with a warning.  The full geometry may be
  modelled; the mirror half is simply never meshed.  No Boolean is
  involved, so the plane is an exact grid coordinate rather than a
  CSG face with inflated OCC tolerance.  Wall placement downstream
  needs no special case: a SymmetryPMC face gets the step-2c PMC
  pull-in (the magnetic wall lands ON the declared plane), a
  SymmetryPEC face gets its edge mask.  Without a `position` the
  declaration is semantic only — the geometry already ends at the
  plane (half-model style).  Both modelling styles produce identical
  meshes (pinned bit-exact in the tests), because the conformal fill
  cannot distinguish "material ends at a shape face on the domain
  wall" from "material continues past the grid".
- `Mesh.with_boundary_conditions` cannot clip after the fact (the
  grid is taken as given): declaring a *new* symmetry position there
  raises; re-declaring the built closure and adding position-less
  symmetry semantics remain allowed.
- **Store:** the symmetry map rides in the `mesh.h5`
  `boundary_conditions` attribute (`symmetry` key, absent for
  symmetry-free meshes — old stores load unchanged).  The resume
  recipe stays physical: it rebuilds runtime walls; symmetry
  semantics round-trip with the mesh.

**Stage D (shipped).**  Port *reports* publish full-model
impedances.  The new geometric core `window_domain_faces()`
(`port_plane.py`) names the bbox face under each lateral window edge;
the factory joins it with `symmetry_entries()` and records the
cutting planes as `PortOperatorReport.symmetry_faces` at all three
operator-construction sites (modal, CW true-mode, band-DTBC).  The
report's numeric fields stay the raw half-window solver values (the
honest solver protocol — and the `Mode` objects keep their raw
normalisation, which drives injection/recording); the *publication*
layer applies `z_line_full_scale` (per cutting PMC plane
`z_full = z_half/2` — the halves sit in parallel; per PEC plane
`z_full = 2·z_half` — in series): `PortReport.z_line_num`,
`ModeReport.z_line`, the summary line (plus a "cut by symmetry
plane(s) … full-model values" note), and `z_line_delta_relative`
(the analytic reference solves the continuous *full* geometry, so
the delta compares the scaled value — before this stage a half port
showed a spurious ~+100 % delta against its reference).
S-parameters need no correction (excitation and a/b share the
half-model mode normalisation, which cancels), and mode
compatibility is enforced by construction — the 2D mode solver
inherits the wall on the port cross-section.  A plain `"PMC"` wall
is NOT a symmetry cut: only declared symmetry triggers the scale,
so every existing PMC-wall setup keeps its reading.  Lumped ports
carry no window solve and stay uncorrected (documented limitation).
Pinned on the parallel plate (exact η₀·d/W): raw half/full ratio
exactly 2, published values equal, both cut kinds
(`tests/unit/test_symmetry_declaration.py`).  Measured trap: on the
`from_grid` path (no PMC pull-in) the magnetic wall sits half a
boundary cell outside, so the raw ratio is (W_full+h)/(W_half+h),
not 2 — the exactness pin needs `from_geometry`.

**Stage E (shipped: mirrored plots + flux).**  Field plots show the
full model, mirrored on read.  `monitors/base.py` gains the
machinery: `MirrorSpec` (axis, wall, kind, at_low),
`resolve_mirrors(region, mesh)` — a plane counts only when the
region's cells reach the domain wall on that side, and the *wall*
coordinate is the physical mirror plane (PEC: the outermost grid
line; PMC: half the boundary cell outside it, where the natural
magnetic wall sits — after the mesher pull-in that is exactly the
declared plane, and on the `from_grid` path it is the physically
correct image plane too); `mirror_sign` implements the continuation
table (across PMC, E continues like a polar vector — normal odd,
tangential even — and H like a pseudovector; across PEC the roles
swap; magnitudes always even); `mirror_extend` /
`mirror_plane_arrays` do the axis-extend + sign-weighted flip.
Wired into `plot()`/`interact()` of `MonitorFieldTime` and
`MonitorFieldFrequency` (1D, 2D scalar and 2D vector branches;
`interact` delegates to `plot`), resolved in `attach()` and
round-tripped through the store as a schema-additive `symmetry`
attribute (`results.h5` monitor groups, `fields_freq.h5`), so the
`_Loaded*` readers mirror without mesh access.
**Fluxes and the excitation power semantics (measured).**
The first certificate run measured that a monitor-side flux ×2
double-counts under the half-window power-normalised excitation: at
equal injected power the raw half-model flux already matches the
full-model run (0.9659 vs 0.9670 W on the certificate case).  The
underlying question — should "1 W injected" *declare* full-model
watts — was deferred here and is resolved in [[DD-155]]: the source
now injects ×1/√2 per port-cutting plane and `MonitorFluxTime` books
×2 per plane cutting its cross-section, source-independently.

**`MonitorWallLoss` (stage E).**  A symmetry face never books as a
physical wall: the analysis masks declared symmetry faces alongside
port planes and non-PEC faces (`_non_wall_boundary_faces`, also the
SIBC wall enumeration), and a user-listed `bc_faces` entry on a
symmetry plane is dropped with a warning at attach.
`dissipated_fraction` (and everything derived from it, including the
stored reduction) carries full-model semantics: losses double per
symmetry plane and so does the reference power for planes cutting
the reference cross-section — those cancel; a plane *parallel* to
the reference cross-section contributes the remaining factor 2.
The fraction is a quotient of quadratic forms of the same fields,
so this is excitation-independent.

**Geometry overlays (stage E).**  `CrossSectionOverlay` carries the
in-plane mirrors; `render_geometry_overlay` draws the cross-section
once per mirror image, each image clipped to its half-space and
reflected via an artist transform (`Affine2D`) — no geometry is
rebuilt.  The display always shows what the solver saw: the
simulated half plus its mirror, for full and half-modelled geometry
alike (an asymmetric far half of a fully modelled geometry never
shows).

**ParaView (stage F).**  The generated `paraview.simple` pipeline
gains a `reflected()` stage: one Reflect (CopyInput,
FlipAllInputArrays where the ParaView version has it) per declared
plane on every monitor source, and Clip-then-Reflect on the geometry
reader.  Half-model data on disk, full model in the renderer.  The
symmetry planes travel in the session CONFIG (`_symmetry_config`,
reusing the monitor mirror resolution for the wall coordinate).
Known limitation: FlipAllInputArrays mirrors all vector arrays like
polar vectors — exact for E; an H pseudovector keeps its magnitude
but the mirrored-half arrow sign is inverted.

**Certificate:**
`validation/symmetry_full_vs_half_certificate.py` — shielded
microstrip with a dielectric block (non-trivial S11), FULL geometry
built twice, run without and with `{"xmin": "SymmetryPMC"}`:
44 892 → 24 768 cells, max |Δ|S11|| = 1.5e-3 and
|Δ|S21|| = 2.2e-4 over 3–9 GHz, published z_line 51.67 vs 51.23 Ω
(0.85 % — the two grids differ on the shared half-space), flux peak
Δ 0.12 %.  The remaining deltas are discretisation-level, as
expected for non-identical grids.

**Files:** `boundaries/boundary_conditions.py` (`Symmetry`,
`symmetry_entries`, normalisation), `boundaries/__init__.py`,
`mesh/mesher.py` (domain clip, `with_boundary_conditions` guard),
`io/project.py` (mesh round-trip), `analysis/_recipe.py`
(Symmetry-aware BC serialisation), `geo/__init__.py` (docstring),
`ports/_modal/port_plane.py` (`window_domain_faces`),
`ports/_modal/port_report.py` (`symmetry_faces`,
`z_line_full_scale`), `ports/_modal/factory.py`
(`_with_symmetry_faces`), `ports/_modal/mode_report.py`
(publication layer), `monitors/base.py` (mirror machinery),
`monitors/field_time.py`, `monitors/field_frequency.py`,
`monitors/wall_loss.py` (fraction factor + bc_faces guard),
`analysis/scattering_td.py` (symmetry faces masked from wall
booking), `post/plot_field.py` (mirrored overlays),
`io/paraview.py` (Reflect stage + `_symmetry_config`),
`validation/symmetry_full_vs_half_certificate.py`,
`tests/unit/test_symmetry_declaration.py`.

## DD-155 — Sources declare full-model watts under symmetry

**Date:** 2026-08-14
**Status:** Accepted, shipped, certified.

**Context.**  [[DD-154]] left one asymmetry: S-parameters, impedance
reports, loss fractions and plots were full-model, but absolute
power quantities followed the *source normalisation* — the modal
excitation is power-normalised on the meshed half window, so "1 W
injected" meant one watt into the half-space (full-model fields ×√2
too high at nominal 1 W), while a field-normalised source (plane
wave) already produced full-model fields but only half the
full-model flux.  No monitor-side factor can serve both source
families at once; the correction belongs to the source declaration.

**Decision.**  Declared source amplitudes are full-model
quantities, realised by two separate scales that each act exactly
once:

- **Injection** (`analysis/scattering_td.py`, `_excitation_scale`):
  a port cut by k symmetry planes injects its waveform ×1/√(2^k)
  (`PortOperatorReport.power_wave_full_scale` inverse) — half the
  full-model power enters the meshed half-space and the fields sit
  at full-model level.  Applied on the analysis layer at all
  injection sites: the modal/CW `set_excitation` wrapper and both
  band drives (`set_excitation_band` gained an `amplitude=` factor;
  an explicit band waveform wraps like the modal one).
  `reference_signal` keeps sampling the *unscaled* waveform — it is
  the full-model 1-√W reference the monitors renormalise against;
  scaling it too would cancel the correction.  Low-level operator
  use (`spec.excitation` at build time, direct CW drives) stays in
  the raw solver protocol.
- **Recording** (`ports/recorder.py`): the recorder composes
  ×√(2^k) per port onto the DD-078 `record_scale`, so every
  consumer of the recorded V/I — `a()`/`b()`, both S-parameter
  paths (`compute_s_parameters` and the band decomposition build
  b/a entirely from these signals), `result.signals`, the streaming
  sink and the store read layer — sees full-model wave amplitudes
  from one scale at one place.  A factor of exactly 1 keeps the
  scale entry `None`: non-symmetric runs are bit-identical.
- **Flux** (`monitors/flux.py`): `MonitorFluxTime` books ×2 per
  symmetry plane whose axis lies in its cross-section (a plane
  parallel to the surface leaves the aperture whole).  With the
  sources declaring full-model amplitudes this is
  source-independent: the port case composes ×1/2 (injection)
  ×2 (aperture) back to the full-model watt, the plane-wave case is
  the plain aperture factor.
- **Mixed port pairs.**  The per-port scale is not cosmetic: for a
  cut excited port and an uncut receiving port the half-window
  normalisations do *not* cancel, and the pre-DD-155 S21 was √2 off.
  With per-port ×√(2^k) the S-matrix entries are the true full-model
  values — under the only excitation the half model can realise for
  an uncut port, the symmetric (in-phase) drive of the port and its
  mirror twin.  Such ports warn at operator construction
  (`_with_symmetry_faces`): model the port on the plane, or drop the
  declaration.
- **Unchanged by construction:** S-parameters of symmetric port
  pairs (scale cancels in b/a), loss *fractions* (quotients of
  quadratic forms), stop criteria and ring-down gates (relative),
  DTBC/Mur terminations (linear).  Lumped ports remain the
  documented DD-154 limitation.

**Certificate** (`validation/symmetry_full_vs_half_certificate.py`,
extended): flux peak full 0.9670 vs half 0.9659 W (Δ 0.12 % — the
injection and aperture scales compose to the same physical watt the
pre-DD-155 pair measured), excited-port a(t) peak Δ 0.064 %
(full-model √W on both runs), 1-W-renormalised |E| probe beside the
strip Δ 2.6 % (discretisation-level point probe on two different
grids; the old semantics would read √2 ≈ 41 % high), S unchanged
(max |Δ|S11|| = 1.5e-3).  Unit pins in
`tests/unit/test_symmetry_declaration.py`
(`TestFullModelPowerSemantics`, flux aperture factors, mirror-twin
warning).

**Files:** `ports/_modal/port_report.py`
(`power_wave_full_scale`), `ports/recorder.py` (scale composition),
`analysis/scattering_td.py` (`_excitation_scale`, modal + band
injection), `ports/_modal/band_dtbc.py` (`amplitude=`),
`ports/_modal/factory.py` (mirror-twin warning),
`monitors/flux.py` (aperture factor).

## DD-156 — Conductor grouping: cell links fuse labels, never add nodes

**Date:** 2026-08-14
**Status:** Accepted — implemented, tested.

**Problem.**  The auto-derived conductor groups
(`ports/_modal/auto_conductors.py`) walked the PEC *edge* graph
alone; PEC-cell corner links were only a rescue for < 2 components.
On a curved conductor the staircase edge graph can leave a sub-cell
surface fragment disconnected from the conductor body (a u-edge above
the apex whose connecting v-edges fall below the classifier
threshold).  Grouped as its own "conductor", the fragment forms a
phantom TEM channel across a near-zero gap whose enormous C' sorts
FIRST in the capacitance-ordered channel basis and shadows the real
mode at small ``n_modes`` — measured on the stripline-coupler ZL
worksheet: ``z_line = 0.95 Ω`` reported instead of ~46 Ω, the phantom
being a 2-node component a hair above the electrode apex (internal
record: `investigations/section-open-chains/`).

**Decision.**  The PEC-cell corner links of the port-adjacent slab
are consulted unconditionally, but only to decide which edge-graph
components belong to the same conductor (label fusion).  The
conductor node sets stay those of the edge graph: adding cell-corner
nodes would widen a staircased conductor and shift its line
impedance (measured −10 % on the `test_modal_factory` coax fixture
with full node merging — the reason the naive "merge everything"
variant was rejected).  Distinct conductors cannot fuse: the
mesher's feature-gap floor keeps them at least one non-PEC cell
apart.  The under-resolved rescue (< 2 edge components) keeps the
full merged node sets — there the cell graph is the only material
source — and keeps its refine-the-mesh warning.

**Gates:** `tests/unit/test_modal_factory_auto_conductors.py::
TestSurfaceFragmentAbsorption` (a surgically isolated surface
fragment is a phantom on the edge graph alone and joins its
conductor under label fusion), plus the pre-existing extractor,
fallback and parallel-plate pins (unchanged numbers).

**Files:** `ports/_modal/auto_conductors.py`.

## DD-157 — Section contours are closed or the plane is re-taken

**Date:** 2026-08-14
**Status:** Accepted — implemented, tested.

**Problem.**  `cross_section_polygons` grouped the section edges into
wires and tessellated whatever came back; a wire that failed to close
was implicitly closed by the polygon consumers.  On a plane in the
near-tangent band of a curved face of a tolerance-inflated Boolean
solid (the DD-106 shifted re-evaluation samples exactly such planes:
``x = r ± deflection`` next to a bore tangent, with Boolean edge
tolerances of ~43 µm), ``BRepAlgoAPI_Section`` returns a mutilated
edge set; the implicit closure then books arbitrary coverage.
Measured on the stripline coupler: a single open 13-point chain
spanning both coax bores (69.6 mm start-to-end gap), booking a
bore-wall H face at 0.80 free instead of 0.19 — and y-layer
dependent, so the feed-chain mass slabs behind the coax port deviated
by 0.43 and every channel fell back to modal Mur-1st.  The
boolean-CUT quarter model sectioned cleanly, which is why the defect
appeared only when the DD-154 symmetry declaration replaced the
manual cut (internal record:
`investigations/section-open-chains/MEASUREMENTS.md`).

**Decision.**  Closedness is part of the section contract:

* A tessellated wire whose endpoints miss by more than
  ``max(8 · deflection, 5 % of the contour perimeter)`` is an OPEN
  chain — never implicitly closed.  (Small genuine seams — vertex
  tolerances, a dropped closing lid segment — stay far below 5 % and
  keep the historical implicit closure.)
* An open chain marks the *plane* as degenerate: the section is
  re-taken at deterministic nudges ``±4, ±8 tessellation lengths``.
  A nudge that comes back empty where the un-nudged plane saw
  material is rejected too — it stepped clear off a feature thinner
  than the nudge and would silently erase it.
* If open chains persist, they are dropped with a `UserWarning` and
  the closed subset of the un-nudged section is returned — a loud
  shortfall instead of silent fantasy coverage.
* The exact-in-face path (DD-137) keeps its position semantics: no
  nudge, open face-region chains are dropped loudly.

Explicitly NOT adopted: canonical even-odd re-winding of the
returned contours.  The winding of a section contour remains
meaningless by contract (consumers use the even-odd rule; the
signed-area consumer relies on the kernel pairing opposite windings
on degenerate tangency bands so their contributions cancel) — a
prototype that re-wound contours by nesting depth changed the
calibrated tangency-band bookkeeping and moved the coax
``z_line_num`` from < 5 % to −10.7 % off the closed form.

**Certificate:**
`validation/section_open_chain_guard_certificate.py` — the full
coupler union (the smallest body reproducing the mutilated section;
every reduced variant sections cleanly) has consistent signed vs
even-odd coverage on the offending plane and a y-invariant bore-wall
M_μ column.  Gate:
`tests/unit/test_geometry.py::TestSectionAtFace` (open chains on a
seam plane warn and are dropped).

**Files:** `geo/_occ_backend.py` (`cross_section_polygons`:
`_wires_at` / `_tessellate` split, closedness test, nudge retry),
`tests/unit/test_geometry.py`.

## DD-158 — Unregistered-wall warning only for scenes with lossy conductors

**Date:** 2026-08-14
**Status:** Accepted — implemented, tested.  Amends the DD-099 warning.

**Problem.**  The DD-099 unregistered-wall warning (a conductor shell
thinner than one cell cancels out of the loss registration) fired
material-blind at mesh time — also on all-PEC scenes, where there are
no losses to lose and every re-mesh of e.g. a stripline worksheet
printed a warning with no actionable content.  Developer call: the
noise outweighs the advance notice.

**Decision.**  The *registration* stays unconditional (it cannot be
reconstructed after meshing and is a shared by-product of the
conformal section pass).  The *warning* fires only when the scene can
actually dissipate on those walls:

* at mesh time, when a lossy wall conductor is declared —
  ``Material.lossy_metal`` in the library, or a ``PECBoundary``
  carrying its own ``wall_sigma`` (dict-form boundary conditions);
* otherwise at conductor-resolution time
  (``resolve_wall_conductors``), when a caller-supplied fallback
  ``sigma=`` / per-face override turns plain-PEC walls lossy after
  meshing — the single choke point shared by the perturbative chain
  (``wall_loss_Q`` / ``MonitorWallLoss``) and the SIBC setup, so no
  loss path can silently miss a dropped surface.  The default warning
  filter deduplicates repeats from that call site.

The message text is unchanged and now lives in one place
(`_surfaces.warn_unregistered_walls`).

**Gates:** `tests/unit/test_unregistered_wall_warning.py` — lossy
shell warns at mesh time, all-PEC shell stays silent (cells still
flagged by `detect_unregistered_walls`), fallback ``sigma=`` recovers
the warning at resolution time.

**Files:** `mesh/_surfaces.py` (`warn_unregistered_walls`,
`resolve_wall_conductors`), `mesh/mesher.py` (gated call).

## DD-159 — String/tuple symmetry vocabulary, `Symmetry` class removed

**Date:** 2026-08-14 (pre-v0.1.0-tag API break) — **Status:** shipped

**Context.**  Every name in the boundary vocabulary is a plain string
handed to `BoundaryConditions` — except the symmetry plane, which
required a dedicated class import (`from magnelio.boundaries import
Symmetry`).  Developer call: the class is a foreign body in an
otherwise string-typed facade.

**Decision.**  One declaration vocabulary, strings and tuples only
(the last vocabulary change before v0.1.0):

* `"SymmetryPEC"` / `"SymmetryPMC"` — symmetry plane **with domain
  clip at position 0.0** (symmetry planes conventionally sit on the
  global origin, so the common case needs no number);
* `("SymmetryPEC", position)` — clip at the given world coordinate;
* `"ForceSymmetryPEC"` / `"ForceSymmetryPMC"` — declaration only, no
  clip: the geometry is already built as the half model (this is the
  pre-DD-159 semantic of the bare `SymmetryP*` strings).

The `Symmetry` class is deleted, not deprecated (no public release
ever shipped it).  All parsing funnels through one private helper
(`_parse_symmetry_value`); the internal normal form stays the
`BoundaryConditions.symmetry` `{face: position_or_None}` map, so
every `symmetry_entries()` consumer (mesher clip, DD-155 power
scaling, monitors, mirroring, ParaView) and the project-store schema
are untouched.  The resume recipe now collapses *all* symmetry forms
to the physical wall type (previously bare `SymmetryP*` strings
leaked verbatim into the recipe — latent inconsistency fixed).

**Explicitly changed semantics:** a bare `"SymmetryPEC"`/
`"SymmetryPMC"` used to mean "no clip"; it now clips at 0.0.  The
as-built meaning moved to the `Force*` prefix.

**Gates:** `tests/unit/test_symmetry_declaration.py` (bare string
clips at 0.0, tuple position honoured, `Force*` does not clip,
malformed tuples rejected loudly);
`validation/symmetry_full_vs_half_certificate.py` and
`validation/section_open_chain_guard_certificate.py` run on the new
vocabulary.

**Files:** `boundaries/boundary_conditions.py`,
`boundaries/__init__.py`, `analysis/_recipe.py`, docstrings in
`geo/__init__.py` / `mesh/mesher.py`.


---

## DD-160 — Field plots resample onto a plot raster; port modes are drawn full-model

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  Two complaints about the port-mode pictures, plus one
measurement that reversed a third.

1. A port window cut by a declared symmetry plane was drawn as the
   solved half.  The full-model mirroring built for monitors
   (DD-154) was never wired into the port path, although the port
   report already knew it was cut (`PortOperatorReport.symmetry_faces`,
   used for the impedance scaling).
2. `plot_field_vector` picked arrows by striding the computational
   grid (`u[::sx, ::sy]`), with independent strides per axis.  On a
   graded mesh the arrow density therefore *drew the refinement*, not
   the field, and the raster was anisotropic under `aspect="equal"`.
3. Arrows near a conductor contour looked wrong, suspected to come
   from the conformal material averaging on cut cells.

**Measurement (internal record `investigations/port-mode-plots/`).**
Reconstruction error was separated from solution error by running the
plot path on an *analytical* coax TEM field sampled on the same port
plane edges.  On a half coax at 14 cells per diameter the destaggering
is accurate to 0.32° in direction and 0.3 % in magnitude on uncut
cells, and 1.6° / +2.7 % on cut cells — while the *solved* profile at
those same cells carries 5° of spurious tangential field on uncut
cells and reads 17 % low on cut ones, unchanged under refinement.
Weighting the destaggering average by the conformal free-area fraction
`f_A` changed the reconstruction by nothing measurable (the port-plane
weights are effectively binary: edges are either bulk or eliminated).

> **Correction (DD-161).**  The synthetic control fed the destaggering
> *field samples* while a solved profile carries *edge voltages*, so
> the two branches of this comparison were not the same quantity and
> the "reconstruction is fine" half of the conclusion does not hold.
> Most of the 5°/17 % was the missing length metric in the plot, not
> the mode solve.  The plot-raster, validity-mask and full-model
> decisions below are unaffected.

**Decision.**

* **Plot raster, not grid samples.**  `plot_field_vector` interpolates
  onto an isotropic raster spanning the slice; `density` counts arrows
  along the longer axis and the shorter one follows at equal spacing.
  Applies to every vector plot (monitor slices and port modes alike) —
  arrow positions are a property of the picture.
* **Validity mask.**  A new `valid=` argument marks cells the field
  does not live in.  They are dropped from the interpolation stencil
  instead of being read as zero, and raster points more than half
  invalid stay blank.  The port path derives it from the profiles
  themselves (all four bounding edges exactly `0.0` ⇒ buried in a
  conductor), which is the same convention `_avg_nonzero` already
  relies on.
* **Port modes are drawn full-model**, mirroring field and geometry
  overlay across every in-plane symmetry plane the window reaches.
  The mirror primitives moved to `post/_symmetry.py` so the monitor
  and port paths share one implementation of the continuation rules;
  `monitors/base.py` re-exports them for its existing call sites.
* **No cosmetic conformal correction.**  Neither `f_A` weighting nor a
  wall-normal projection of the arrows is applied: the measurement
  puts the residual error in the solved profile, not in the plot, and
  projecting the arrows onto the local conductor contour would hide a
  real discretisation error behind a tidy picture.  The mode-solution
  accuracy on cut cells is a separate matter (KB-018).

**Trap found on the way.**  Matplotlib short-circuits a *rectangular*
clip path into the artist's clip **box**, replacing the axes clipping.
The DD-154 mirror clip did exactly that, so a `bbox_inches="tight"`
save grew to the extent of the whole off-screen geometry (the picture
itself was always correct).  The clip is now a `Polygon` plus an
explicit `set_clip_box(ax.bbox)`.

**Gates:** `tests/unit/test_plot_field.py` (isotropic raster on a
graded grid, `density` semantics, interpolated — not sampled —
values, masked cells blank and not diluting their live neighbours);
`tests/integration/test_solve_ports.py`
(`test_symmetry_cut_port_plots_the_full_window`,
`test_symmetry_mirrors_reach_the_geometry_overlay`).

**Files:** `post/_symmetry.py` (new), `post/plot_field.py`,
`ports/_modal/mode_report.py`, `monitors/base.py`,
`analysis/scattering_td.py`.

## DD-161 — Mode profiles are grid quantities; the plot divides by the edge metric

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  KB-018 recorded a mode profile that was not the clean
radial field a coax TEM mode must be: several degrees of spurious
tangential content, `E_r·r` (constant for a coax) spread by 13 %, and a
17 % low reading on the cells the conductor contour cuts.  DD-160's
control experiment had placed the error in the solved profile rather
than in the picture.  That control was wrong: it fed the destaggering
an analytical field *sampled* at the edge midpoints, while a solved
profile holds something else.

**Finding.**  The 2D mode solvers return FIT **grid quantities**, not
field samples.  `solve_tem_laplace` builds `ê = -G₂ᴅ φ` with a
*topological* gradient, so the primal profile is the edge voltage
`E·l_primal`; `travelling_wave_h_profiles` builds the dual voltage
`ĥ = H·l_dual` with `l_dual = μ₀·normal_dx·l_partner/M_μ`.  The
operator's Poynting sum undoes both conventions explicitly; the plot
did not, and read the DoF vector as a field.  On a graded mesh the
difference is a per-edge factor: it tilts every cell-centre vector and
biases its magnitude, worst where the conductor contour forces a
locally different spacing.  Only the analytical mode families
(`Mode.field_evaluator` set) hold sampled V/m and A/m — and they never
drive a port, they serve as the DD-048 reference value.

Measured on the KB-018 coax port (r_i = 1.52 mm, r_a = 3.5 mm), the
invariants of the reconstructed cell-centre field:

| grid | quantity | spread of `E_r·r` | tangential angle, median / p90 |
|---|---|---|---|
| 67 annulus cells | DoF read as field | 0.188 | 5.89° / 12.05° |
| 67 annulus cells | **divided by the metric** | **0.077** | **2.76° / 9.41°** |
| 87 annulus cells | DoF read as field | 0.194 | 6.71° / 17.21° |
| 87 annulus cells | **divided by the metric** | **0.074** | **2.48° / 7.36°** |

The 17 % low reading on cut cells disappears (0.833 → 0.968 of the
reference), and the residual now *converges* under refinement, which
the raw reading did not.  The H picture, rebuilt through its own dual
lengths, lands on the identical figures — the exact `H = ẑ × E/Z`
relation of the travelling wave, which is a sharp check that both
metrics are right.

**Decision.**

* `ModeReport._field_profiles` converts the DoF vector to V/m or A/m
  before destaggering: primal profiles by the plane's primal edge
  lengths, dual profiles by `PortOperatorModal.h_dual_lengths` (new
  read-only property, the same expression the operator's power sum
  uses).  Faces frozen inside a conductor report length `0` and are
  skipped rather than divided by zero.
* Analytical families pass through untouched.
* `DiscreteMode`'s attribute docs name the convention per path; the
  old "in V/m" wording is what invited reading the array as a field.

Nothing outside the plot changes: the operator already converted where
it needed to, and the profiles keep driving injection and projection
in their FIT metric.

**Residual.**  After the fix the same coax still shows ~2.5° and ~7 %
spread at the contour, converging at the staircase rate.  That is the
ordinary discretisation gap DD-048 describes for the
operator-consistent path — the mode solve fixes its Dirichlet
potentials on whole nodes, so the conductor *contour* is staircased
even though the masses are not.  No further attribution is claimed;
the candidates KB-018 listed (enlarged-cell donation bias, rim-edge
sub-cell metric) were never separated from it.

A census taken while chasing this deserves recording, because it looks
alarming and is not: **no bbox face carries a single conformal E-edge**
(category 1 or 2), while the layer one cell behind each face carries
hundreds — 809/251/251/164/89/89 for xmin…zmax on a mesh with 28 471
in total.  The cause is the candidate mask in
``geo/_filling.py``, which is only ever written on the interior index
range of each transverse axis, so a boundary-face edge never gets a
conformal average and cannot become category 2 either (the f_L pass is
gated behind the same array).

It does not reach the port, because
:func:`flatten_port_plane_mass` already overwrites the port-plane slab
with the first interior slab — for both the mode solve and the FIT-TD
update — exactly because those boundary values were known to be wrong.
Measured: opening the mask changes ``build_M_eps`` on 28 port-plane
edges by up to 28 %, and the solved mode is bit-identical
(z_line 52.611565 Ω, profile fingerprint to nine decimals).  The
untreated case is a symmetry / PEC / PMC / CPML face with geometry
crossing it, which nothing flattens — filed as KB-019.

**Gates:** `tests/integration/test_solve_ports.py`
(`test_profiles_are_grid_quantities_not_field_samples` — a WR90 on a
transversally graded grid, where TE10's uniform `E_z` is a 3.5×
ramp in the raw DoF and flat to 1e-9 after the metric;
`test_analytical_families_keep_their_sampled_profiles`).

**Files:** `ports/_modal/mode_report.py`, `ports/_modal/operator.py`,
`ports/_modal/discrete.py`.

## DD-162 — Mode-profile plots reach the port window, not the last cell centre

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  Destaggering the edge profiles lands both transverse
components on cell centres, and the picture spanned exactly that: from
the first cell centre to the last.  The frame therefore stopped *half a
cell* short of the window on each of the four sides.  On a uniform mesh
that is invisible; on the graded mesh of a real port it is not.
Measured on the stripline-coupler worksheet's `zmin` port (internal
record `investigations/port-mode-plots/`):

| axis | window | drawn | missing |
|---|---|---|---|
| u (x) | 0.732 … 25.000 mm | 1.463 … 22.949 mm | 0.73 / 2.05 mm |
| v (y) | 0.000 … 29.000 mm | 2.983 … 28.526 mm | 2.98 / 0.47 mm |

The first v-cell is 5.97 mm wide, so the strip lost at that edge is a
tenth of the frame — which reads as a whole cell layer cropped away.
A mirrored full-model plot got the same gap as a seam through its
middle.

Nothing was missing from the *solution*: the window covers the face it
was asked for and every edge in it carries a DoF.

**Decision.**  `ModeReport.plot` extends the picture to the window
boundary before handing it to the renderer.  Each component is
staggered along one axis only, so it carries a genuine sample on the
two window boundary lines of its *other* axis; those go in unchanged.
The partner component has no sample out there and is carried out by its
nearest interior value, making a boundary arrow exact in one component
and first-order in the other.

Validity on the added lines is decided by the **genuine** component
alone.  A zero there means the edge is in or on a conductor, so
continuing the partner outward would invent an arrow inside the metal —
which is what a first attempt did, drawing field several millimetres
outside the beam pipe of the coupler worksheet.

**Trap this exposed.**  An electric symmetry plane *is* the outermost
grid line, so a picture that now reaches the window boundary has a
sample sitting exactly on the mirror.  `mirror_extend` concatenated its
reflection unconditionally, putting a duplicated coordinate — a
zero-width interval — into the vector the plot raster interpolates
against.  It now drops the self-image.  Cell-centre data (every monitor
slice, and every port plot before this change) never hit it.

**Gates:** `tests/integration/test_solve_ports.py`
(`test_plot_reaches_the_window_boundary` — graded WR90, axes must match
the guide cross-section exactly for E and H;
`test_electric_symmetry_does_not_duplicate_the_wall_line`;
`test_symmetry_cut_port_plots_the_full_window` tightened from "within
10 %" to the exact guide width).

**Files:** `ports/_modal/mode_report.py`, `post/_symmetry.py`.

---
## DD-163 — The conformality patch sees enlarged-cell donations

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  [[DD-095]] corrects the port-power patch at conformal cut
edges with χ = M_ε·l / (ε₀·eps_pair·A_geo), and leaves every
category-0/1 edge at χ = 1 so that dielectric staircase planes stay
untouched.  Its dossier (§5c, internal dossier
`investigations/port_power/DERIVATION.md`) recorded one edge class that
the invariant gets wrong: the **receiver of an enlarged-cell
donation**.  A short curved-PEC edge is masked and hands its dual-face
mass to a neighbour; when that neighbour is category 0/1 its M_ε
exceeds the staircase value, so the patch transports the masked edge's
share without booking it.  Session 124 could not close it — the round
coax it froze the spec on had only category-2 receivers — and shipped a
Δχ estimator that *warned* above 1e-3 instead.

That warning fires on ordinary work: the tutorial-02 polyethylene coax
reports 1.0e-2, the DD-095 fixture's own round port 3.5e-3, and four
tutorials carry it into the published documentation.  A user meeting it
on a plain coaxial port has no action to take.

**Decision.**  Fold the donation into χ.  The receiver's own staircase
mass is recovered by subtracting the donation back out of its M_ε, and
the patch is the ratio of the two:

    χ_d = (M_ε,d·l_d/ε₀) / (M_ε,d·l_d/ε₀ − Σ_s borrowed_s)

This is the dossier's own second recipe.  It needs no assumption about
the two edges' materials — the receiver's permittivity cancels — and it
is exactly 1 without a donation, so the conformality-only invariant
survives unchanged.  Category-2 receivers need nothing: their M_ε ratio
already carries the donation.  The Δχ arrays and the warning are gone.

**Measurements** (internal record
`investigations/port_power/donor_receiver_gate.py`):

| gate | before | after |
|---|---|---|
| mixed round→square reciprocity | 0.012923 dB | **0.000005 dB** |
| conformal round coax, port/flux | 1.004550 | 1.006045 |
| square coax (staircase), port/flux | 1.004734 | 1.004734 |
| plate TEM (staircase), port/flux | 1.002783 | 1.002783 |
| WR-90 TE10 (staircase), port/flux | 1.001000 | 1.001000 |

Reciprocity is the discriminating gate and it is now satisfied to
machine precision: |S12| = |S21| is an exact identity of the reciprocal
structure, all common-mode error cancels between the two ports of one
run, and the conformal port's power scale is thereby pinned to the
staircase port's.  The absolute port-versus-flux-monitor gate cannot
resolve the change — it scatters over 1.0010…1.0047 across fixtures
whose every edge is staircase and whose true value is therefore 1.
Reading the conformal coax's +0.15 % as a degradation would mean
trusting that gate well beyond its own 0.37 % spread; the dossier
put its floor at 0.5 %.

Staircase planes stay bit-identical, as DD-095 §5b(2) requires.

**Gates:** `tests/integration/test_port_flux_patch.py` — the trigger
fixture DD-095's WP-P2 asked for and never got (a coax whose port plane
carries category-0/1 receivers; 8 per component here, χ = 1.039…1.056),
plus the assertion that no other category-0/1 edge moves, plus the
user-visible half: `solve_ports` on a plain conformal coax is silent.
`tests/integration/test_port_power_reciprocity.py` tightened from
0.05 dB to 0.005 dB — a factor 2.6 below the 0.0129 dB defect it now
has to catch, a factor 1000 above the measurement.

**Files:** `ports/_modal/operator.py`.

## DD-164 — The conformal classifier reaches the domain boundary faces

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  The conformal candidate mask in `geo/_filling.py` was
written only on the interior index range of each transverse axis
(`bnd_ex[:, 1:Ny, 1:Nz] = ...` and its two siblings), so an E-edge lying
*in* a bbox face never received a conformal average, and the line-solid
`f_L` pass — gated behind the same array — could not reach category 2
there either.  Every partially filled edge on a domain face was rounded
to fully free or fully metal.  Port planes were unaffected
(`flatten_port_plane_mass` substitutes the first interior slab for the
port-plane slab precisely because the boundary values were known to be
wrong), but a symmetry / PEC / PMC / CPML face is not flattened, and its
outermost operator layer carried a staircased contour.  It was found and
measured in an earlier session and *not* shipped, for a defensible
reason: no certificate improved.  The pillbox quarter model gained 33–45
conformal edges on its symmetry faces and returned a bit-identical
eigenfrequency — TM010's `E_z` vanishes at the cylindrical wall, so that
fixture is blind by construction — and the only gate that moved was a
band-DTBC floor, in the wrong direction.

**Decision.**  Extend the mask to the boundary indices, with each
boundary edge's dual face clamped to `[wall, first dual line]` and
`eps_avg` / `f_A` left intensive: the consumer's `A_dual` is the full
boundary cell (the mirror convention of `_build_avg_d`), so an average
taken over the truncated half needs no factor, and the material
continues by mirror symmetry, by extrusion, or not at all.  The μ
pipeline already treats its domain-boundary faces this way; this removes
the asymmetry between the two.

An edge lying in a bbox face is additionally blocked as an
*enlarged-cell receiver*.  A face later closed with PEC has its
tangential edges masked by `Mesh.with_pec_boundaries`, i.e. after the
donors are picked, and the borrowed mass would vanish without a trace
(`test_pair_consistent_subcell::test_donors_are_never_masked` is the
existing gate for that failure mode).  Donating *out of* such an edge
stays allowed — `blocked` is consulted only for the receiver.

**The certificate.**  A half model closed with a magnetic symmetry wall
is the discrete restriction of its full model, so for a symmetric mode
the two eigenfrequencies agree to machine precision — an exact identity
with a known target, which is what the earlier attempt lacked.  Making
it exact takes one grid observation: the magnetic wall sits half a
boundary cell *outside* the outermost primal line, realised by moving
the clipped line at the plane to `h/3`, so a half grid is never the
restriction of a *uniform* full grid.  Forcing that same `h/3` line into
the full model's ladder makes the two grids agree on `x >= 0` line for
line, which the certificate asserts rather than assumes.

Four fixtures, one PEC box cavity carrying its TE101-like mode, whose
`E_y` is maximal *on* the symmetry plane:

| fixture | what lies in the plane | before | after |
|---|---|---|---|
| `empty` | nothing | 0 | 0 |
| `offset` | nothing (mirrored dielectric pair, two cells clear) | 2.2e-15 | 1.8e-15 |
| `brick` | a grid-aligned dielectric contour | **+2.0e-04** | 4.2e-15 |
| `cylinder` | a curved dielectric contour | **−2.3e-03** | 4.7e-15 |

The two unloaded fixtures are the floor and they stay put, which is what
separates a grid or wall-convention error from a classification error.
The loaded pair moves by ten to twelve orders of magnitude.  Note that
`brick` is affected at all: category 1 is an average over the *dual
face*, so a perfectly grid-aligned material interface running through
the plane needs it too — the defect was never confined to curved
boundaries.

**The one gate that moves the other way.**
`test_qtem_band_dtbc_sparams::test_s11_floor` reads −120.06 → −116.92 dB
against a −120 dB bound.  That bound cannot decide this: the quantity is
a kernel-fit residual, and its own sensitivity to the fit's resolution —
a knob that changes no physics — is far larger than the effect.  On the
unchanged code, raising `n_grid` from 9 to 11 and 13 moves the worst
point to −138.86 and −150.79 dB and individual frequency points by up to
52 dB.  The fixture was therefore defending 0.06 dB of margin on an
under-resolved fit.  Resolving it (`n_grid` 9 → 13, the bound untouched)
restores 29 dB of margin and brings the CI fixture in line with the
full-size benchmark's below-−155 dB; the classifier change costs a
consistent 1.7–3.1 dB at every resolution, one order of magnitude below
what the knob itself is worth.

**Gates:** `tests/integration/test_symmetry_boundary_face.py` (the
`cylinder` fixture plus the `offset` control, the grid-agreement
assertion that makes the comparison meaningful, and a guard that the cut
fixture still classifies its symmetry face — without it the certificate
would pass while testing nothing).  Full certificate with all four
fixtures: `validation/symmetry_boundary_face_certificate.py`.
Kernel-resolution spread: internal record
`investigations/port-mode-plots/dtbc_floor_spread.py`.

**Files:** `geo/_filling.py`, `geo/_subcell.py`.

## DD-165 — Between two agreeing ladders, conditioning decides

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  `couple_face_material_pairs` (DD-053) offers every H face
two candidate ladders, and defines the face mass through the pair
identity when they agree.  Agreement is a relative test at `rtol = 1e-6`
— but the DTBC uniform-chain gate that consumes the result admits
`1e-8`.  Anything landing in that band produces a mass that the pairing
certifies and the port rejects, and the rejection is quiet: only the
chain-slab branch warns.

The model that surfaced it is a stripline coupler whose second coaxial
stub is a *mirrored copy* of the first.  Mirroring inflates the tolerance
of the unioned solid; exactly one Ey edge next to the mirrored bore
came out `7.5e-7` off its partner, the weighted pair spread of port2's
only TEM channel read `1.73e-8` against the `1e-8` gate, and that
channel silently fell back to modal Mur-1st while its geometric twin on
the unmirrored stub certified at `7e-15`.

The root cause recorded when this was first measured — `eps_avg` and
`f_A` coming from inconsistent integrals of the same dual face, so
`eps_pair = eps_avg/f_A` misses 1 in vacuum — **is refuted**.  Measured
today on that same model: the identity holds on all 19 244 conformal
edges to `3.9e-15`, with no edge above `1e-12`, while the pairing error
is unchanged.  Both integrals run the same reverse-priority area budget,
so they cannot disagree; the inconsistency sits in the other factor of
`t = ε0μ0·ε_pair·d·d̃ / M_ε`.

**Decision.**  When both ladders are valid and agree, the one whose own
two partners disagree less supplies the target.  Three reasons, in
order of weight:

1. Agreement at `rtol` still admits a spread of up to `rtol`.  Inside
   that band the choice is not arbitrary — one candidate is measurably
   the better estimator of the same quantity.
2. The old rule was not invariant under axis permutation.  The ladder
   list is Hx → (z, y), Hy → (z, x), Hz → (y, x), so swapping y and z
   maps an Hx face onto an Hx face but flips its ladder order: the same
   model, differently oriented by the user, got a different material
   matrix.  A solver result must not depend on that.
3. The partner residual is a direct measure of the violation of the
   translation invariance the estimator assumes.

Unchanged: two valid ladders that *disagree* still yield no override —
a genuinely 3D neighbourhood keeps its Krietenstein value.

**Measured.**  Coupler, mirrored stub (`port2`) against its unmirrored
twin (`port1`) as the control:

    port              axis order              conditioning
    port1 (control)   7.15e-15  dtbc          7.14e-15  dtbc
    port2 (mirrored)  1.73e-08  mur           6.29e-14  dtbc
    port2 worst edge  3.75e-07                8.24e-13

`z_line` is bit-identical on both ports across the change, and the full
suite is unchanged at 2145 passed.  On benign geometry the rule is a
no-op: a prism whose cross-section is mirror-symmetric about the
diagonal `y = z`, meshed on a grid whose y lines equal its z lines,
returns bit-identical `M_mu` either way (that fixture was built as a
gate and rejected as one — it cannot produce diverging ladders, because
a prism is translation-invariant along its own axis).

**Not done, deliberately.**  Loosening the DTBC gate — it is the
quantity the port's exactness rests on.  Tightening `rtol` to the gate
instead: that would switch the pairing off wherever the jitter exceeds
`1e-8`, restoring the Krietenstein value that DD-053 exists to replace
on a line, and no measurement supports it.  The tolerance gap between
the pairing and the gate therefore remains: this decision makes the
choice inside the band optimal, it does not close the band.  If both
ladders are jittered, nothing here helps.

**Gates:**
`tests/unit/test_operators.py::TestCoupleFaceMaterialPairs::test_the_better_conditioned_ladder_supplies_the_target`
constructs the condition directly — one `M_eps` entry perturbed by less
than `rtol` leaves the ladder along its own axis inconsistent and the
transverse one exact — with
`test_uniform_box_needs_no_override` as the control that makes the
assertion meaningful.  Certificate on the full coupler:
`validation/pair_ladder_choice_certificate.py`; every reduced variant of
that model certifies either way, the same lesson as DD-157.

**Files:** `_operators/material_matrices.py`.

## DD-166 — A boundary line of the mode picture inherits its neighbour's validity

**Date:** 2026-08-15 — **Status:** shipped

**Context.**  DD-162 grew the mode-profile picture out to the port
window and decided the validity of the two added lines per axis from the
*genuine* component there — the one whose staggering puts a sample on
that line — reasoning that a zero means the edge is in or on a
conductor.  That component is the one tangential to the line, and the
frame of a 2D mode problem is an electric wall all the way round, so it
is identically zero on every port, always.  The outermost ring of arrows
was therefore dropped everywhere, including where the wall carried the
field's maximum.  Measured on a stripline port: tangential component
`0.0` on the lower window line, normal component `6.05e7` — three times
the tangential content of the first interior row.

**Decision.**  An added line is valid exactly where the interior line it
continues is.  What decides whether there is anything to draw is whether
the neighbourhood is metal, not how large one component happens to be:
*on* a conductor is not *in* one.  The resulting boundary arrow is exact
in its tangential component (identically zero on a wall, genuinely
sampled) and first-order in its normal one, so it stands perpendicular
on the wall — the boundary condition made visible rather than hidden.  A
window reaching into a conductor still stays blank, because there the
interior line is invalid too.

**Not a defect, and asked about in the same breath:** the arrows also
thin out in a narrow gap between conductors.  That is the arrow raster,
not the classifier — DD-160 made it isotropic and therefore independent
of the computational grid, so a refined region does not gain arrows.  On
a 50 mm wide window the default `density=20` spaces them 2.6 mm apart,
which steps over a 3 mm gap resolved by five grid planes; `density=40`
fills it.  The `density` docstring now says so.

**Gate:**
`tests/integration/test_solve_ports.py::TestModePlot::test_wall_line_keeps_its_perpendicular_arrows`
— raster points that resample to nothing never reach the quiver, so the
drawn extent reaching all four window edges *is* the assertion, plus a
check that an edge carries a real share of the peak so the test cannot
pass on a ring of zeros.

**Files:** `ports/_modal/mode_report.py`.

## DD-167 — The degeneracy escape gets a length of its own

**Date:** 2026-08-15
**Status:** Accepted — implemented, tested.  Amends DD-157.

**Problem.**  DD-157 re-takes a section that comes back with an open
chain at deterministic offsets `±4, ±8` steps away, and made the step
the tessellation `deflection`.  That ties an *escape distance* to a
*chordal-accuracy budget*.  The two are unrelated: the chord says how
faithfully a curve is drawn, the escape has to clear a near-tangency
band whose width comes from the geometry.  The mesher tessellates its
two passes an order apart on purpose — cell classification needs
point-in-polygon fidelity (`h/10`), the conformal-area sites integrate
over the polygon and take `h/100` — so the finer pass silently
inherited a ten times shorter reach.

Measured on the stripline coupler (internal record:
`investigations/section-open-chains/MEASUREMENTS.md`).  The mesher
anchors a grid line on the electrode's lateral extreme
(`x = 26 mm · sin 13.751° = 6.1804 mm`); the neighbouring cell-centre
plane then sits `h/2 = 0.168 mm` inside it, inside the 0.238 mm band
where the section plane grazes the electrode's own side face.  The
collision is structural, and **refining the mesh moves the cell centre
closer to the anchor, i.e. deeper into the band**.  With the escape at
`h/100` the ladder reached 20 µm and every retry stayed inside the
band; the classification pass, escaping 200 µm from the identical
plane, left it without trouble.  The control experiment is one call
with one parameter changed:

| plane | escape | conductor union |
|---|---|---|
| x = 6.0124 mm | chord (20 µm) | 0 contours, warning |
| x = 6.0124 mm | h/10 (200 µm) | 43.81 mm² |
| y = 25.303 mm | chord (20 µm) | 0 contours, warning |
| y = 25.303 mm | h/10 (200 µm) | 1744.02 mm² |

The consequence was worse than the shortfall itself: **the two passes
disagreed about where the material is.**  The cells were classified
conductor (that pass escaped) while the material matrices saw nothing
there (this one did not).  On the offending H-face plane the conformal
face count collapsed to 378 against 1165 and 654 either side, and
`A_face_pec` dipped 12.4 % below the mean of its neighbours on a
geometry whose cross-section varies smoothly.  Cell classification and
interior-PEC edges were untouched, so no metal disappeared — the
affected edges fell back from conformal to staircase.

**Decision.**  `cross_section_polygons` takes the escape step as its
own parameter, defaulting to `deflection` for callers with no grid to
relate it to.  All three mesher passes pass `h_min/10`, one constant
(`SECTION_NUDGE_FRACTION` in `geo/_filling.py`, beside the two chordal
fractions), so they cannot end up with different opinions again.  The
ladder's far end stays inside one cell (`8 · h/10 = 0.8 h`), which is
the upper bound on the step: the escape answers about a plane the
caller did not ask about, and that displacement has to stay small
against the cell it is booked into.

The warning also became actionable.  It named neither the body nor the
amount nor what happens next, so a user could do nothing with it; it
now identifies the solid by its material, reports how many chains were
dropped and how far the retry searched, and says that the bulk
classification survives while the sub-cell resolution does not.

**Residual, since closed by DD-168.**  Within the last ~70 µm before
the electrode's lateral extreme the section returned one of the two
rotationally symmetric slivers instead of both — silently, since the
surviving contour closed.  Read as the section operator failing at
grazing incidence, it was nothing of the kind: the kernel produced
every edge, and the wire builder here dropped half of them.  See
DD-168.

**Certificate:** `validation/section_nudge_reach_certificate.py` — the
section-level A/B above, the shared-escape invariant, and the full
coupler meshing with zero open-chain warnings and the H-face plane no
longer a hole between its neighbours (dip 12.4 % → −1.1 %).  Gates:
`tests/unit/test_geometry.py::TestMesherEscapeReach` (both passes
escape by the same distance, and the far end of the ladder stays
inside one cell) and
`tests/unit/test_geometry.py::TestSectionAtFace::test_the_escape_step_is_its_own_length`
(the two lengths are independent; the escape falls back on the chord
only when the caller declines to choose).  The seam fixture used to
carry the second gate — an escape tied to the chord over-stepped the
35 µm strip and the retry was rejected — but DD-168 made that plane
well-posed, so it can no longer demonstrate it.

**Files:** `geo/_occ_backend.py` (`cross_section_polygons`,
`compute_face_material_areas`, `batch_cross_sections`, the section
worker), `geo/_filling.py` (the three length fractions),
`geo/_subcell.py` (the wall-plane pass, the third caller sectioning
the same solids on the same grid), `mesh/mesher.py`,
`tests/unit/test_geometry.py`.  The geometry plot keeps the default —
it sections at a user-chosen deflection with no grid to relate a step
to, and its far coarser default already reaches ±0.8 mm.


## DD-168 — Section edges are chained, not wired

**Problem.**  DD-167 left a residual on record: over a ~70 µm band
before a lofted electrode's lateral extreme, the section returned one
of two rotationally symmetric slivers instead of both, silently,
because the surviving contour closed.  It was written down as the
section operator failing at grazing incidence — the situation DD-157
settles with "drop, never invent".  Measuring it says otherwise.  The
kernel produces every edge; they are lost afterwards, in this code.

Contours were assembled by handing edges to `BRepBuilderAPI_MakeWire`
one at a time and keeping whichever it accepted.  It accepts an edge
that reaches *any* free end of the wire so far, including a vertex
that already joins two — the result is a branched pseudo-wire, which
is not a wire at all.  `BRepTools_WireExplorer` then walks one arm of
the branch and stops.  The edges beyond the branch sit inside the
wire, never tessellated, never counted: no open chain, so no warning,
so nothing downstream can tell.

Measured on the coupler's electrode union, plane by plane:

| plane | raw section edges | reach tessellation | endpoint-graph valences |
|---|---|---|---|
| x = 6.090 mm | 12 | 12 | all 2 |
| x = 6.110 mm | 14 | **7** | two vertices of valence 3 |
| x = 6.130 mm | 12 | **8** | valence 1 and 2 only |

At 6.110 mm eight edges went into one wire and the explorer visited
one.  The near-tangency band is exactly where the extra short edges
that make such a junction appear.

**Decision.**  Chain the section edges here instead, on a graph of
their endpoints.

*Vertices* merge at the kernel's own tolerance
(`BRep_Tool::Tolerance`), not an invented one.  That tolerance is what
inflates in a tangency band — measured 5 µm away from it against
138 µm inside — so any fixed threshold either tears clean contours
apart or fuses distinct ones.  A first attempt clustered at 1 nm and
split contours that had been correct for years.

*Chaining* runs in two passes.  The first follows only vertices where
exactly two edges meet: no choice, no geometry, cannot go wrong.  The
second joins the resulting segments across branch vertices by tangent
continuity — of the ends meeting there, the pair that continues one
another is the pair whose inward tangents most nearly oppose, and
anything turning by 90° or more is treated as a different feature
touching and left for the open-chain guard.  Doing this in one greedy
pass instead is not enough: seeded at a loose end it starts on the
stub and swallows the contour behind it, which is what
`test_a_seam_stub_does_not_swallow_the_contour` pins.

*Direction* follows the edges' parameterisation, as the wire path did.
The section edges' own orientation flags do not serve: measured across
the grazing band they run opposite on two mirror-image contours of one
solid, and honouring them makes the pair's signed areas cancel.  This
matters because signed areas are summed per face rectangle downstream;
it is only the `abs()` at the end of that sum, and the fact that two
contours 11 mm apart never share a rectangle, that keeps a disagreeing
winding from erasing material.

**What it costs and buys.**  A/B over 514 section calls (six fixtures,
two chordal budgets, planes across each): 497 identical including the
sign of every contour area, 0 → 0 warnings, 797 → 802 contours.  The
17 that moved are the grazing planes, where a lost contour comes back.
Across the band the returned area now falls smoothly — 169.5, 156.5,
143.4, 130.4, 117.4, 104.4, 91.3, 78.3 mm² at 10 µm steps — where it
used to halve at 6.110 mm.  On the coupler mesh: same grid, no
warnings, conformal Hx faces 11737 → 11923, cells classified PEC
64904 → 64934, PEC area booked on Hx +3.8 mm².  Thirty cells of metal
that were being meshed as vacuum.

The wire builder also went away as a cost: it rebuilt and copied the
whole wire for every candidate edge it tried.

**Consequence for `exact_at_faces`.**  A plane lying in a face was
lossy on the plain path for the same reason — the seam edges branched
the wire.  With the edges chained, the two-brick seam fixture returns
its 40.00000 mm² whole and silently on the plain path, so the test
that pinned the shortfall now pins its absence.  The option itself is
unchanged and remains the supported way to ask about such a plane;
nothing here re-examines the cases that motivated it beyond that
fixture.

**Certificate:** `validation/section_chain_completeness_certificate.py`
— edge conservation on the grazing planes, continuity of the returned
area across the band (largest step 14.2 % where a lost contour is a
halving), and what the coupler mesh books.  Gates:
`tests/unit/test_geometry.py::TestSectionEdgeChaining` (every edge
lands in exactly one chain on a real solid; a seam stub does not
swallow the contour it stands on; a contour cut by two junctions is
put back together; and an open chain through a junction survives it —
its two halves are joined head to head, since which end of a segment
is its head only records where the first pass started on it) and
`tests/unit/test_geometry.py::TestSectionAtFace::test_the_seam_no_longer_eats_area`.

**Files:** `geo/_occ_backend.py` (`_chain_section_edges`, and
`cross_section_polygons` which now tessellates chains — the
exact-in-face path converts its genuine wires to chains and is
otherwise untouched), `tests/unit/test_geometry.py`,
`validation/section_chain_completeness_certificate.py`.  Measurements:
internal dossier `investigations/section-open-chains/`.

---

## DD-169 — The mirrored half carries the field, not a look-alike

**Problem.**  DD-155 made a declared symmetry plane visible: half-model
data on disk, full model in the renderer, one reflection filter per
plane.  Opened in ParaView 6 the session shows the simulated half only,
with a half-built `<monitor>_mirror_0` hanging in the pipeline browser
with nothing downstream of it.  Two separate defects, and the second
one is the reason the first one matters.

*The filter's property set was renamed, wholesale.*  Every name the
generator used is gone, and the free axis value it relied on went with
them:

| what the session set | ParaView 6 |
|---|---|
| `Plane = "X"` | `PlaneMode`, whose values are only `Interactive` and the six bounding-box faces |
| `Center = wall` | `ReflectionPlane`, a plane sub-proxy with `Origin`/`Normal` |
| `FlipAllInputArrays` | `ReflectAllInputArrays` |

The first assignment raises, a `try/except` around the whole loop
returns the *unreflected* input, and every consumer stays attached to
it.  Measured on a two-plane coupler run: `Et_points <- Et` and
`geometry_cut_Et_x <- geometry`, with `Et_mirror_0` present but fed to
nothing and `Et_mirror_1` never created at all.  The fallback was meant
as a courtesy — better unmirrored than nothing — and instead turned a
dead API into a silent misstatement about what is being displayed.

*The filter is not the physical continuation.*  It transforms every
3-component array as a polar vector, negating the component along the
mirror axis, and does not touch single components at all.  Against
`mirror_sign`, which is what the monitor plots continue their data
with, that is right for exactly two of the eight combinations:

| | E across PEC | E across PMC | H across PEC | H across PMC |
|---|---|---|---|---|
| vector array | global −1 | correct | correct | global −1 |
| single components | all three wrong | normal wrong | normal wrong | two tangential wrong |

A model with one plane of each type — the coupler has `xmin` magnetic
and `ymin` electric — therefore gets one mirrored half right and one
backwards, in a picture that looks symmetric either way and from which
an even mode reads as odd.

**Decision.**  Place the plane against whichever property set the build
exposes, and correct the sign in the pipeline rather than hope for it.

*Placement* tries `PlaneMode`/`ReflectionPlane` first and the flat
`Plane`/`Center` pair second, and treats the array-continuation flag as
mandatory rather than optional: the corrections below assume the filter
ran it.  Failing all of that is no longer a quiet return.  The
half-built filter is deleted so the pipeline does not advertise a
feature that is not there, a note appears in the render view, and the
script prints a marker that `bake_pvsm` turns into a `RuntimeWarning` —
the one channel that reaches a caller who never opens ParaView.

*Signs* come from `mirror_sign` itself, resolved at export time into a
per-plane list of `[array, factor]` pairs.  A vector's factor is
`-mirror_sign(field, axis, axis, kind)` — the wanted sign divided by
what the filter already did — and a single component's is the full
continuation factor.  Deriving the rule a second time next to the one
under test is precisely the mistake that cost a probe run here: the
first version of the certificate's probe re-derived it and got H
inverted, because `flips_normal` depends on the field and the copy did
not.

*Planes needing no correction* stay a single reflection with the input
copied.  The others reflect *without* the copy, so the mirrored branch
stands alone and its correction is a constant factor per array with no
coordinate test anywhere, and the halves are rejoined afterwards.

*Two flattening steps* are not decoration.  Reflecting a composite
dataset assigns the continued vector to different cells than the
untouched single components — measured, on the coupler: scalars and
vector components agreed to the last bit at the reader and disagreed by
2.2e3 V/m after a composite reflection, which would colour a glyph from
one place and aim it from another.  And joining two halves leaves
cell-to-point averaging blind to the seam, which left 2.0 % of the peak
as a tangential field sitting on an electric wall.  Flattening before
each reflection and once after the last join removes both; geometry has
neither problem and keeps its blocks, which carry the body names.

*The corrected copy keeps its input's element type.*  A `Calculator`
promotes its result to double, and a double array meeting its float
twin at the join drops out of the joined dataset entirely — no error,
no empty array, just a name that is no longer there.  On the coupler
this took `E`, `Ex` and `Ez` out and left `Ey`, which is to say it took
the field out and left something that still renders.

*The lattice keeps its dimensions.*  Its spacing is sized for a target
arrow count in the displayed picture, and the displayed picture is the
mirrored one, so keeping the count puts the target on the full model
instead of on each half.  The one exception is a monitor collapsed onto
a mirrored axis: mirroring turns its single cell layer into two, and a
lattice of one would sample the seam between them.

**What it costs and buys.**  Per plane, one flattening step plus a
reflection, and — only where a sign is wrong — one `Calculator` per
affected array and a join.  On the coupler's field monitor that is 4
corrections across two planes.  The slice planes still open where they
did; the plane of the default view is the centre of the *simulated*
region rather than of the full model, which is left alone deliberately,
since the centre of a mirrored axis is the symmetry plane itself and a
cut there shows a tangential field of zero.

The magnetic wall sits half a boundary cell outside the grid, so the
mirrored halves meet across a gap one cell wide.  Measured on the
coupler: gap 0.43 mm against a lattice spacing of 4.55 mm, and not one
of the 25 344 lattice points falls in it.

**Certificate:** `validation/paraview_symmetry_certificate.py` — on a
quarter cavity behind one plane of each type: every declared plane
becomes a reflection and the displayed branch hangs off the last of
them; single components still equal the vector's components (0.0);
and every component reproduces the sign `mirror_sign` prescribes across
every plane (worst 1.1e-15 of peak).  Gates:
`tests/unit/test_paraview_export.py::TestMirrorSigns` and
`::TestSymmetryReachesTheSession`, which pins that two planes of
opposite type put opposite corrections on the same field.

**Files:** `io/paraview.py` (`_mirror_signature`, `_mirror_factor`,
`_mirror_fixes`, `_prepare_mirroring`, `_symmetry_config` now carrying
the wall type, the `reflected`/`mirror_plane` pair in the generated
script, and `bake_pvsm`), `tests/unit/test_paraview_export.py`,
`validation/paraview_symmetry_certificate.py`.

---

## DD-170 — A monitor's `data` states a unit or refuses

**Problem.**  `MonitorFieldFrequency` accumulates
`Σ F(t_n)·exp(+jω t_n)·Δt` — the transient folded with the excitation
spectrum, carrying an extra factor of time.  Dividing that spectrum out
turns it into the field of a 1 W CW excitation (DD-078 pinned the
waveform as `a(t)` in √W), and `renormalize` has done exactly that
since before DD-078 named the units.  It was never called for the
caller: DD-078 closed its work item "with zero new machinery", so the
division stayed a user step, and the store reader inherited the
convention verbatim ("renormalising to 1 W stays a reader/user step").

Two consequences compounded.  First, `data` fell back to the raw bins
when no reference was set, so the *same* property returned two
physically different quantities — field·s or field per √W — with no
signal which.  Second, the pair `data` / `data_raw` asserts that one of
them is processed; a reader who sees both concludes `data` is the
prepared one.  In the store the two were literally the same object
(`data_raw = data`), so looking for the difference found none.

Neither `docs/` nor `examples/` mentioned `renormalize` — it appeared
only in tests and certificates.  The failure mode is silent by
construction: a field *pattern* is invariant against a constant complex
factor, so plots look right and only absolute numbers are wrong.
Measured on a stripline-kicker worksheet (internal record,
`userscripts/stripline_coupler.ipynb`): transverse shunt impedance off
by 5e18, the excitation's own spectral shape mistaken for the device's
frequency response.

**Decision.**  A run renormalises its own frequency monitors, and
`data` refuses to answer when it cannot.

- *Who divides.*  Every path that samples a run's reference waveform
  hands it to that run's frequency monitors: `_run_one_excitation`,
  `_run_band`, the streamed path, and resume (`_actual_steps` counts
  from step zero, so the sampled waveform spans the whole run, not the
  appended tail).  `attach` rebuilds the accumulators per run, so a
  multi-excitation sweep pairs each monitor state with its own
  excitation.  On the store side the reader divides by the run's
  persisted reference, sampled lazily and memoised, so listing a
  project's monitors still costs no results read.
- *No silent fallback.*  Without a reference, `data` and `component`
  raise and name both ways out (`data_raw`, `renormalize`); `data_raw`
  always returns the undivided bins.  `data` therefore has one unit:
  E in V/m, H in A/m, per √W.
- *Idempotent by construction.*  `renormalize` stores the divisor and
  never touches the accumulated bins, so the automatic call and an
  explicit one cannot stack.  This is what makes the change safe for
  existing scripts and certificates that call it themselves.
- *Not `None`.*  A `None` return travels to the caller's next
  subscript and fails there, explaining nothing; the exception carries
  the diagnosis at the point of the mistake.
- *Every output channel, not just Python.*  The ParaView export reads
  the stored bins directly rather than through `.data`, so it was
  untouched by the above and would have shipped a second channel whose
  numbers disagree with the first by nine decades.  It divides by the
  same spectrum now.  The division itself lives once, in
  `monitors/_dft.py` (`source_spectrum`, `divide_by_spectrum`), and the
  three call sites share it — a spectrum computed in a different
  convention than the accumulator's would cancel only approximately.

**Cost.**  A behaviour break at the public API: scripts that read
`data` without renormalising used to receive raw bins silently and now
raise.  That is the point of the change — those numbers were wrong by
the excitation spectrum — but it is a break, taken deliberately while
MAJOR is 0.

**Certificate.**  `tests/integration/test_port_units.py::
test_frequency_monitor_fields_per_1w_cw` now asserts the absolute TEM
value `√(z_line·1 W)/gap` **without** calling `renormalize`, so it
gates the automatic path against analytic physics rather than against
itself.  `tests/unit/test_monitors.py::TestDataStatesItsUnit` pins the
refusal, its wording, that raw bins stay reachable, and that repeated
renormalisation cannot stack.  Two channel-agreement gates sit in
`tests/integration/test_paraview_session.py`: the streamed run's
in-RAM monitor against the same monitor read back from the store
(1e-12), and the exported `.vtr` against the monitor bit for bit —
either would break the moment two sites sampled the reference waveform
over different spans.

**Files:** `monitors/_dft.py` (`source_spectrum`,
`divide_by_spectrum`), `monitors/field_frequency.py`
(`_require_source`, `data`, `data_raw`, `renormalize_all`),
`analysis/scattering_td.py` (`_sampled_excitation`,
`_renormalize_freq_monitors`, four call sites), `io/project.py`
(`_LoadedFreqMonitor`), `io/paraview.py` (`_run_source_spectrum`,
`_export_freq_monitor`), `tests/unit/test_monitors.py`,
`tests/integration/test_port_units.py`,
`tests/integration/test_paraview_session.py`,
`docs/methods/sources-monitors.md`,
`examples/tutorials/plot_06_field_monitors.py`.

---

## DD-171 — The published docs carry two channels, stable is the front door

**Problem.**  DD-116 put the documentation portal on GitHub Pages with
`actions/upload-pages-artifact`, which replaces the entire site on
every deployment.  There is therefore exactly one published build, and
since `docs.yml` triggers on pushes to main, that build documents
unreleased code.  A reader who installs the newest release and opens
the docs reads about API that is not in their install — and nothing on
the page says so.  The failure is silent in the direction that costs
the reader most: features described but absent, defaults quoted that
have since moved.

The obvious fix — build every version on every deployment — does not
survive the build cost.  sphinx-gallery executes all fourteen
tutorials, roughly an hour of CPU, so a build is something a channel
earns once and then keeps.

**Decision.**  The site holds two independent builds, and Pages serves
them from a branch rather than from a single artifact.

- *Channels.*  `/stable/` is built from a `v*` tag, `/dev/` from main;
  the site root is a redirect to `/stable/`, falling back to `/dev/`
  until a tag has published once.  Each channel is rebuilt only when
  its own source moves, so a main push never disturbs the release docs.
- *Storage.*  The `gh-pages` branch is the Pages source and a plain
  file store: a publish clones it shallow, replaces its own directory,
  and pushes.  A `.nojekyll` marker is mandatory there — Jekyll runs on
  a branch source and hides every path beginning with an underscore,
  which would drop `_static/` and serve the site unstyled.
- *Switcher.*  `pydata_sphinx_theme`'s version switcher reads
  `switcher.json` from the site **root**, not from the channel's own
  `_static/`.  A release build is frozen the day it ships; a switcher
  shipped inside it would forever offer the channels that existed at
  build time.  One shared file keeps every published page's menu
  current.  `conf.py` learns its own channel from
  `MAGNELIO_DOCS_CHANNEL` (unset — a local build — is `dev`), which
  feeds `version_match`.
- *Warning banner.*  Raised from the channel, not from the switcher.
  The theme decides it by comparing the build's own `release` against
  the `version` of the entry marked preferred, and only when **both**
  parse as release numbers; channel names never do, so the comparison —
  and with it the early return for "this is the preferred version" —
  never runs.  Left to the theme the banner therefore appeared on every
  page of *both* channels, and on `/stable/`, which the site root
  redirects to, its "switch to stable version" button linked to the page
  it was already on.  The dev build additionally carries a `+dev` local
  version segment, because main keeps the last release's number until
  the next bump and the theme reads the word out of that string to word
  the banner as a development version rather than as a release.

**Cost.**  The Pages source must be switched from "GitHub Actions" to
"Deploy from a branch: gh-pages / root" in the repository settings; the
OIDC `github-pages` environment is no longer used and the workflow
needs `contents: write` instead.  The branch accumulates a build's
worth of gallery images per publish, in a history that clones of the
source pull down by default — squash it to an orphan commit if it ever
grows uncomfortable.  The build passes `-d` for exactly this reason:
Sphinx puts its doctree cache inside the HTML output unless told
otherwise, which measured 96.3 MB of the first publish's 109.5 MB — a
browser never asks for it, and pickles change wholesale from build to
build, so it would have been the branch's dominant growth term.  Concurrency is serialised (`docs-publish`,
`cancel-in-progress: false`) because two publishes racing would push
conflicting trees, and cancelling one would throw away an hour.  The lock sits
on the publish job rather than on the workflow, and the job publishes an
uploaded artifact instead of its own working tree.  `cancel-in-progress:
false` protects the *running* member of a group, but a group holds only
one *queued* run and a third arrival cancels the one already waiting.
With the whole workflow in the group, that window spanned the full build:
a push landing while a tag built dropped the tag's publish — silently,
since every remaining run reported success and only `/stable/` stayed
behind (measured 2026-08-19, on the v0.3.1 publish).  Confined to the
publish job the window is the few seconds the branch push takes.

**Files:** `.github/workflows/docs.yml`, `docs/conf.py`
(`docs_channel`, `html_theme_options["switcher"]`,
`html_baseurl`), `docs/switcher.json`.

## DD-172 — A lumped element on a symmetry plane is half a device

**Problem.**  DD-154/155 wired full-model power semantics for modal
ports and excluded lumped ports as a documented limitation.  The gap
was worse than "uncorrected": `_snap_edge_chain` resolves endpoints by
nearest-node on the *clipped* grid, so a dipole feed declared across a
symmetry plane silently became a half-gap chain hanging on the wall
node — no warning, no scaling, and the reported input impedance was
the half-model value (the classic ~36 Ω monopole reading for a ~73 Ω
dipole).  Passive `LumpedElement` loads had the same failure.

**Decision.**  The user always declares the full-model device —
endpoints in full-model coordinates (tutorial-09 rule: the geometry
stays full, only the meshing changes), `Z0` and companion values as
the full element.  The builder relates the chain to every declared
symmetry plane and either books, clips, errors, or warns:

- *Electric crossing* (chain along the plane normal, plane strictly
  between the endpoints): allowed iff mirror-symmetric about the plane
  within half a boundary cell; the outside terminal is clipped onto
  the wall node.  This is the series cut — the meshed half carries
  **half** the device (`Z0/2`; `R/2, L/2, 2C` for both companion
  topologies).
- *Magnetic crossing*: error — a current normal to a magnetic plane
  mirrors anti-parallel (`test_t5_pmc_end_is_ideal_open`), no
  full-model element corresponds.
- *Magnetic containment* (chain lying in the plane): the parallel cut
  — the meshed half is one of two parallel branches and carries the
  **doubled** device (`2·Z0`; `2R, 2L, C/2`).  The declared on-plane
  coordinate snaps to the pulled-in outermost line (`plane + d/3`,
  DD-154 stage B); the half-boundary-cell tolerance covers exactly
  that offset.
- *Electric containment*: error — tangential edges inside an electric
  wall are shorted.
- *Endpoint in the discarded half* without being a crossing: error
  (was the silent clamp).
- *Terminal exactly on a clip-declared plane*: error with guidance —
  under full-model coordinates that declaration is a mirror-twin pair
  sharing a node, which is almost certainly a mis-declared crossing.
  Under `ForceSymmetry*` (as-built, halved geometry) the same shape
  *is* the crossing declaration and is booked without clipping.
- *Chain away from every plane*: the DD-155 mirror-twin warning,
  reworded for elements.

The same full-model rule extends to thin wires: a ``ThinWire`` whose
curve lies entirely in the discarded half-space is skipped at mesh
time (``mask_thin_wires`` books a ``None`` path) — like a solid
there, it is represented by its mirror image.  Before this, the
rasteriser died on the clamped single-node curve, which forced
exactly the hand-halved declarations the full-model rule exists to
avoid (found by the dipole tutorial).

The internal scaling is what makes the rest free: with the half-model
device inside the operator, the Thévenin split yields exactly the
modal convention `power_wave_full_scale = √2` per cutting plane for
*both* cut kinds (series: half the voltage; parallel: half the
current — either way `a → a/√2`).  A frozen `LumpedPortReport`
(`symmetry_faces` as `(face, kind, relation)` triples plus the scale
properties) rides the operator's `port_report` attribute, and the two
existing consumers pick it up unchanged: `_excitation_scale` injects
`1/√(2^k)` and `PortSignalRecorder` records `×√(2^k)` — zero new
plumbing, and a report-less operator keeps the recorder's `None`
fast path (non-symmetric runs stay bit-identical).  The
`_LumpedModeStub` carries the operator's internal (scaled) Z0, which
is precisely what makes the recorded power waves full-model: S11 and
`Z_in = Z0·(1+S11)/(1−S11)` with the user's full-model Z0 come out
right with no publication-layer correction.  The raw V/I signal
*ratio* stays half-model-shaped (same as modal ports, where
`z_line_full_scale` fixes it at publication) — a passive load's
measured `−V/I` under a parallel cut reads the doubled device, which
the methods chapter states explicitly.

**Measured** (`validation/lumped_symmetry_parity_certificate.py`):
gate A, all-PEC cavity, double precision, pinned dt — the half model
is the *exact discrete restriction* of the full model, V/I restriction
defect 4.5e-16 / 7.9e-16.  Gate B, CPML boundaries — full-vs-half S11
parity 2.15e-2, floored not by DD-172 but by the CPML min/max mirror
asymmetry the gate uncovered (KB-023); an unscaled feed sits at
O(0.3).  Gate C — a passive load in a magnetic plane presents the
doubled trapezoidal impedance to 1.0e-7.

**Files:** `src/magnelio/ports/_lumped/port_report.py` (new),
`ports/_lumped/factory.py` (`_resolve_symmetry`, `_scaled_element`),
`ports/_lumped/operator.py` (`port_report` field),
`tests/unit/test_lumped_symmetry.py`,
`tests/integration/test_lumped_symmetry_parity.py`,
`validation/lumped_symmetry_parity_certificate.py`.

## DD-173 — Far field from a Huygens box: monitor + transform, image theory included

**Problem.**  No far-field post-processing existed: tutorial 08 ends
on the statement that gain and patterns need a near-to-far-field
transform the library does not provide, and STATUS listed it as the
missing antenna feature.  The reference topology (a monopole on a PEC
ground) also rules out the naive implementation — a closed surface
cannot be drawn inside a domain whose floor is an electric wall.

**Decision.**  Two layers, split exactly like S-parameters:

- ``post.ntff_transform`` + ``post.FarFieldResult`` — the pure
  frequency-domain surface-equivalence transform (J = n̂×H,
  M = −n̂×E, radiation vectors, Taflove/Balanis) over arbitrary
  ``SurfacePatchSet``s, plus ``ImagePlane``s: every patch is mirrored
  across each such plane with the component signs of
  ``post._symmetry.mirror_sign`` — the plot-mirroring table *is* the
  image-current table, so there is no second sign table to maintain.
  A plane is either a real boundary (``physical_halfspace=True``, a
  ground plane: pattern masked beyond it) or a symmetry plane (full
  sphere physical).  Spherical convention: ISO physics, θ from +z,
  φ from +x, fixed to the global axes in v1.
- ``monitors.MonitorFarField(freqs, margin_cells=3)`` — an automatic
  closed Huygens box ``margin_cells`` inside the physical domain
  (absorber depth from the DD-172-era ``Mesh.pml_cells`` accessor,
  falling back to the declared CPML thickness on hand-built grids).
  Per face it runs a ``DFTAccumulator`` over the tangential E/H with
  the H bins stamped at ``t + dt/2`` (the MonitorWallLoss stagger
  treatment).  PEC/PMC domain faces and declared symmetry faces are
  omitted from the surface and booked as image planes; Periodic faces
  are rejected.  Renormalisation joins the DD-170 path
  (``renormalize_all`` dispatches it), and the analysis wires the
  run's accepted power ``1 − Σ|S|²`` into the monitor after each
  in-RAM run, which is what feeds ``gain`` and
  ``radiation_efficiency`` (``realized_gain`` and ``directivity``
  need no wiring).

Three deliberate subtleties, each pinned by a test:

- *Time convention.*  The accumulator convention
  ``Σ F·e^{+jωt}·dt`` produces phasors of the ``e^{−jωt}`` world;
  the textbook NTFF algebra is ``e^{+jωt}``.  The transform
  conjugates once at the entrance and once at the exit, so the
  public pattern is a phasor like every other frequency-domain
  quantity.  The analytic Hertzian dipole pins the sign of
  ``E_theta``.
- *Effective amplitudes.*  The per-1-W-CW normalisation of DD-170
  makes every renormalised phasor an effective (RMS) amplitude
  (``|V| = √(zP)``, port-units gate), so the radiation intensity is
  ``|E|²/η`` with **no** peak-phasor ½ — the first integration run
  showed exactly the factor 2 in the power closure before this was
  written down.
- *Node-plane sampling.*  Each box face lies on a grid-node plane;
  the tangential fields come from ``_interp_to_cell_centres`` on the
  two adjacent cell layers, linearly combined onto the plane.  The
  surface is therefore exactly closed (no half-cell fins at box
  edges) and second-order on graded grids.  Radiated power
  integrates the smooth unmasked sphere and scales by the physical
  solid-angle fraction (images make the pattern mirror-symmetric, so
  this is exact) instead of paying the half-cell bias of a hard mask
  edge.

Under symmetry no extra factor exists anywhere: the 1/√2-per-plane
injection (DD-155/DD-172) plus the 2^k image expansion already
reconstruct the full-model 1 W pattern — a flux-monitor-style ×2
would double-count.  The PMC pull-in leaves a ~d/3 gap between a
mirrored patch pair; second-order, absorbed by the parity gates.
Limitation: the accepted-power wiring covers the in-RAM path; a
store-streamed run serves ``realized_gain``/``directivity`` and gets
``gain`` when the reader wires accepted power (DD-070 follow-up).

**Measured** (`validation/farfield_dipole_certificate.py`,
`tests/integration/test_far_field_antenna.py`): thin-wire λ/2 dipole
D = 2.15 dBi ± 0.15 dB with lossless closure |P_rad − (1−|S11|²)| <
3 %; monopole on PEC ground: masked half space, horizon directivity
= 2× the dipole's within 5 %; SymmetryPEC half model with the
DD-172 lumped feed on the plane reproduces the full-model pattern
within 8 % and P_rad within 5 % (floored by KB-023, not by this
machinery — the analytic image-composition gates in
`tests/unit/test_ntff_transform.py` are machine-exact at 1e-10).

**Files:** `src/magnelio/post/far_field.py` (new),
`src/magnelio/monitors/far_field.py` (new), `monitors/__init__.py`,
`monitors/field_frequency.py` (`renormalize_all`),
`analysis/scattering_td.py` (`_wire_far_field_monitors`),
`tests/unit/test_ntff_transform.py`,
`tests/unit/test_monitor_far_field.py`,
`tests/integration/test_far_field_antenna.py`,
`validation/farfield_dipole_certificate.py`.

## DD-174 — Pattern plots: polar cuts and the 3D radiation surface

**Problem.**  The far field of DD-173 had no picture.  Nothing in the
plot layer draws polar or 3D axes — every existing function targets
rectangular field maps and line plots — and antenna work without a
polar cut and a pattern surface is not presentable.

**Decision.**  Two computation-free drawing functions in
`post/plot_pattern.py`, re-exported as `plots.plot_pattern_cut` and
`plots.plot_pattern_3d` (the two-tier rule: `FarFieldResult` computes
the quantity and delegates via `.plot_cut()` / `.plot_3d()`;
`MonitorFarField` and the store reader delegate through `result(f)`).
House conventions carried over: `(fig, ax)` return, lazy matplotlib
import, `db=` + `floor_db=` like `plot_s` (antenna default −40 dB, not
the S-parameter −200), keyword-only options, a caller-made Axes is
accepted — with a projection check, since a polar trace on a
rectangular Axes fails silently ugly rather than loudly.

Drawing choices worth recording: the polar cut uses the antenna
convention (zero angle up, clockwise), so a θ-cut has the zenith on
top; a φ-plane cut folds the back half through the opposite azimuth so
the trace closes over 0…2π.  The 3D surface maps radius to
``value_dB − floor_dB`` clipped at zero — the floor collapses to the
origin and nulls stay visible as indentations — with radius-shaded
face colors and hidden axes (the numbers live in the cuts, the surface
is for shape).

**Files:** `src/magnelio/post/plot_pattern.py` (new),
`plots/__init__.py`, `post/far_field.py` (`plot_cut`, `plot_3d`),
`monitors/far_field.py`, `io/project.py`
(`_LoadedFarFieldMonitor` delegation),
`tests/unit/test_plot_pattern.py`.

## DD-175 — A field picture stands for a cell layer, not a plane

**Problem.**  The antenna tutorial's field plot showed an empty air box:
the monopole wire and its feed port were missing from the very picture
they explain, although the same model drew both in its cross-section.
Measured on that model (grid 3.57 mm, monitor declared at `y = 0`): the
displayed plane resolves to the cell-centre coordinate `y = 1.786 mm`,
while wire and port are declared on the node plane `y = 0`.  The
geometry overlay draws volume-free features within a *numerical*
tolerance of the cut — 1e-9 relative for a two-point feature, one
radius for a wire — so at half a cell's distance both dropped out.
Volumes survive (a solid still intersects the shifted plane), which is
why the picture looked plausible rather than broken.

**Decision.**  The overlay states the thickness of the layer it stands
for.  `post.plot_geometry.plot_cross_section` gains `slab=0.0` [m], a
half-thickness that raises — never lowers — the in-plane tolerance of
every volume-free feature (`max(existing, slab)`); `slab=0` keeps the
plane-exact behaviour, so a hand-drawn cross-section is unchanged.
`CrossSectionOverlay` carries the value, and the plotting monitors fill
it from their own grid via `monitors.base.plane_slab_halfwidth`: half
the local cell in the normal direction, taken at the displayed cell
centre, hence correct on graded grids.  A missing grid yields 0.0.

**Why the layer, not the declared position.**  Snapping the overlay
back to the user's requested coordinate would fix this one case and
lie about the general one: a 3D monitor sliced anywhere shows the cells
it hits, and a wire 1 mm into that layer belongs in the picture just as
much as one on its boundary.  The layer thickness is the physically
honest statement of what a field pixel covers.

**Consequences.**  `_LoadedFreqMonitor` now takes the store's grid
(the time-monitor reader already did) and passes it to the hydrated
monitor, so a plot from disk carries the same features as the live
one.  Port mode plots keep plane-exact overlays: a port plane is a
declared surface, not a sampled layer.

**Files:** `src/magnelio/post/plot_geometry.py`,
`src/magnelio/post/plot_field.py`, `src/magnelio/monitors/base.py`
(`plane_slab_halfwidth`), `monitors/field_frequency.py`,
`monitors/field_time.py`, `solver/eigenmode_result.py`,
`io/project.py`, `tests/unit/test_plot_geometry.py`,
`tests/unit/test_plot_field.py`.

---

## DD-176 — Geometry arguments are checked where they are written

**Problem.**  Geometry objects are dataclasses, so their type
annotations document but never enforce.  `Sphere(center=R - d/2, ...)`
— a scalar where a point belongs, the natural slip when placing a
sphere on one axis — was accepted, stored, and surfaced four frames
later as `TypeError: 'float' object is not iterable` inside
`_scaling.pad_box`, raised by a `model.plot()` that has nothing to do
with the mistake.  The same shape of failure ran through the whole
subsystem: a two-component translation vector reached the CAD kernel, a
negative `Brick.size` came back as a complaint about OCC precision, and
`chamfered()` without a selector was only rejected once the solid was
finally built — the selector check lives in `resolve_edges`, which runs
at `_occ_shape()` time.

**Decision.**  Every geometry constructor and verb checks its arguments
at the call.  `geo/_validate.py` holds the shared checks — `point3`,
`vector3`, `point_list`, `positive`, `nonzero`, `nonnegative`, `count`,
`operand` — each taking the name of the field it guards, so the message
names the argument rather than the internal that tripped over it.
`_axes.normalize_axis` gained the same `what` parameter, which moves
axis validation into the constructors of `Cylinder`, `Cone` and `Torus`
instead of leaving it to the kernel call.

**The dividing line.**  A constructor checks what it can see without
the kernel: the shape of an argument, the sign and finiteness of a
number, the type of an operand.  Anything needing geometry —
self-intersection, a chamfer wider than its edge, a feature below the
model's OCC precision — stays where the kernel builds the shape.  The
two layers do not overlap: `_check_dimensions` still owns the
scale-relative precision floor, which no constructor can evaluate
because the model scale is not known until every shape is in.

**Type versus value.**  Following the convention already set by
`ThinWire.curve` and `Loft`: a wrong *type* raises `TypeError` (a
scalar where a point belongs, a list where an operand belongs), a
wrong *value* raises `ValueError` (two coordinates instead of three, a
negative radius).  Coordinates are normalised to `float` tuples on the
way in, so a NumPy array reaches the store and the kernel as a plain
tuple.

**Consequences.**  `Union` accepts a single operand (an existing
idiom — the union of one shape is that shape); `Difference` requires a
tool, and an empty operand list is rejected outright instead of dying
on `shapes[0]`.  `Brick.size` must now be positive in every direction,
which the kernel enforced all along through the precision floor, only
with a message about OCC precision rather than about the box.  One
message changed wording enough to move a test: `mirrored(normal="q")`
now names `mirrored(normal)` instead of "axis", the parameter the
caller actually wrote.

**Files:** `src/magnelio/geo/_validate.py` (new), `geo/_axes.py`,
`geo/primitives.py`, `geo/operations.py`, `geo/transforms.py`,
`geo/modifications.py`, `geo/curves.py`, `geo/path.py`, `geo/wire.py`,
`geo/__init__.py`, `tests/unit/test_geometry_arguments.py` (new),
`tests/unit/test_shape.py`.
## DD-177 — The TF/SF correction is a coefficient table, not a loop over boundary cells

**Problem.**  `PlaneWaveSource` corrected the six faces of its
total-field box element by element, in twelve pairs of nested Python
loops per half step.  The cost follows the box *surface*, so it grew
with the model rather than with the physics, and on a card it grew
worst: every element assignment was its own kernel launch.  Measured on
a 2.54 M-cell model (spherical lens, CPML on all faces, 40 steps),
against the same run without a source (milliseconds per time step):

| TF/SF box | boundary cells | CPU loops | CPU table | GPU loops | GPU table |
|---|---|---|---|---|---|
| none      |     — |  33.5 ms |     — |   23.6 ms |     — |
| a tenth   |   800 |  33.8 ms |  31.3 ms |   37.8 ms |  16.0 ms |
| half      | 18304 |  85.8 ms |  31.5 ms |  525.4 ms |  16.4 ms |
| full      | 75200 | 245.7 ms |  31.8 ms | 2024.8 ms |  16.1 ms |

The per-cell figures say the same thing more sharply: the loops cost
2.8 us per boundary cell on the CPU and 27 us on the GPU, which is why
a plane wave wrapped around the whole model — the ordinary case for a
scattering problem — ran an order of magnitude *slower* on the card
than on the processor, and why the excitation cost more than the field
solver it feeds.

**Decision.**  Every face correction has the same form,

    field[face] += beta * metric * A * f(t - k.r/c0)

and only `t` changes between steps.  `attach()` therefore folds
`beta * metric * A` into one coefficient array per face and stores the
retardation `k.r/c0` alongside it; the time loop is left with a single
array expression per face.  `_beta_views` reshapes the flat `_beta_E` /
`_beta_H` blocks onto their Yee component grids — views, so no copy and
no host transfer, and the coefficients stay on whichever device the
solver put them.  A face the incident field cannot excite yields no
record at all, which replaces the old per-component amplitude guards.

**Why the retardation stays separate.**  Folding it into the
coefficient would force a full-face waveform array every step.  Kept
apart, `k.r` is summed only over the axes with `k != 0`, so for
axis-aligned propagation it collapses on its own: a scalar on the two
faces normal to k, a 1-D array on the other four.  The waveform is
evaluated on a handful of values per step instead of on the face, and
the same expression stays correct if oblique incidence is added later —
it simply stops collapsing.

**Consequences.**  Injection is no longer measurable against the
source-free baseline on either backend, so TF/SF box size is once again
a question of scattered-field accuracy alone, not of runtime.  The
recommendation to run large-box plane waves on `backend="numpy"` is
withdrawn.  The rewrite is behaviour-preserving: field states agree
with the loop implementation to 3e-16 relative in double and 1.6e-7 in
single (one float32 ulp) across all six propagation axes, both
polarisations and three box placements, and a 60-step run agrees to
2e-16.  Two loose ends were closed on the way — the H-side corrections
at the upper box faces now carry the same "no scattered-field layer
beyond a flush face" guard the x-max and y-max cases already had, and
`excitation` accepts arrays, which is what lets a whole face be filled
in one call.

**Files:** `src/magnelio/sources/plane_wave.py`,
`tests/integration/test_plane_wave.py` (amplitude and leakage on all
six axes), `benchmarks/bench_plane_wave_tfsf.py` (the table above).

---

## DD-178 — CAD import: STEP is the format, names are the interface

**Problem.**  Every model had to be rebuilt with the CSG API.  For the
parts Magnelio is aimed at — connectors, housings, machined
accelerator components — a mechanical drawing already exists, and
redrawing it is both work and a source of discrepancies between what
is simulated and what is manufactured.  `read_brep` existed but only
as the project store's own reader: it returns bare `TopoDS_Shape`
objects, states no unit and knows no names.

**Decision.**  `magnelio.io.import_step` / `import_brep`, returning a
`geo.Group` of `geo.ImportedSolid`.  Five choices are worth recording.

*STEP over BREP.*  FreeCAD's `.brep` export is literally
`BRepTools::Write` — the same kernel, exact and lossless — which makes
it tempting.  It carries no length unit, no names and no colours,
though, so it cannot be read without out-of-band knowledge, and the
failure mode of guessing wrong is a model a thousand times too large
that nothing contradicts.  STEP states its unit, names its solids and
carries display colours, and every CAD system writes it.  BREP stays
as the secondary path with a mandatory `unit=`.

*Materials by name, assigned at import.*  No exchange format carries
the constitutive parameters a field solver needs; what CAD systems
call a material is a parts-list label.  Assignment is therefore keyed
on the solid names, which are what survives a re-export after the
drawing changes (positions, face counts and solid order do not).  The
resolution rules exist to make a stale mapping loud: literal beats
wildcard, two wildcards disagreeing over one solid is an error, and a
key matching nothing is an error naming the available solids.

*Unmapped solids are construction bodies, not vacuum.*  They arrive
with `material=None`, which DD-127 already defines as a body usable as
a Boolean operand but rejected by `GeometryModel.add`.  A half-mapped
assembly therefore cannot mesh with its unmapped parts silently
filled by the background.

*Flatten assemblies into a Group.*  The assembly tree is a container
structure, not geometry; `Group` (DD-071) already is the
material-preserving bundle the rest of the API consumes, distributes
transforms over its members and is flattened again at `add()`.
Reproducing the tree would add a second hierarchy with no consumer.
Component placements are accumulated down the tree and baked into the
leaf solids.

*Colour is a display channel, never a material fork.*  `Material.__eq__`
drives the material library and the overlap check, so cloning a
material per file colour would fragment both.  The colour rides on the
shape, and `post/_colors.material_color(mat, color)` takes the hue from
the file but the **opacity** from the material — opacity encodes what
a body is (metal opaque, dielectric translucent, vacuum invisible),
which is a modelling statement, not decoration.  An explicit
`Material(color=…)` still wins.

**Mechanics.**  `STEPCAFControl_Reader` with name and colour mode into
an XCAF document.  Units are normalised by setting the process-global
`xstep.cascade.unit` to `M` around the read and restoring it in a
`finally`; the stored `TopoDS_Shape` is therefore always meter-space,
which is exactly the contract the store's loaded shapes already had.
XCAF invents placeholder label names after the shape type (`SOLID`,
`COMPOUND`) for products that carried none — every unnamed solid would
get the same one, so those are treated as unnamed and replaced by
`solid_1…n`.  Colour is read from the placed instance first
(`GetInstanceColor`) and from the prototype label second
(`XCAFDoc_ColorTool.GetColor`, a static method), surface before
generic before curve.  Non-solid leaves (free faces, unstitched
shells) are warned about and skipped; a file with no solid at all is
an error.

`_LoadedShape`, the store's private BRep wrapper, **became**
`geo.ImportedSolid`: same lazy per-scale transform (DD-120), now a
`Shape` subclass, so an imported solid has the verbs and the Boolean
operators and the store's loaded geometry gained them with it.
Healing is `ShapeFix_Shape` per solid (`heal=True` by default for
STEP, off for BREP, which comes from this kernel);
`ShapeUpgrade_UnifySameDomain` is opt-in because it edits topology.
A solid still invalid after healing produces a warning naming it, not
an error — the mesher has its own guards.

**Consequences.**  Geometry that came from a file is indistinguishable
from geometry that was drawn: the integration gate meshes a STEP box
and the equivalent `Brick` and compares grid lines and material fill
element by element.  `geometry.json` gained a schema-additive
`"colors"` list, so a store round-trip keeps the file colours; a store
written before this reads back as all-`None`.  Parametric history is
*not* recoverable from any exchange format and is not attempted —
changing a dimension means changing it in the CAD system and
exporting again, which is what the name-keyed material mapping is
built to survive.

**Deferred:** sheet bodies and sewing shells into solids; IGES and
mesh formats.  Gerber/PCB import was the separate track named here and
is now [[DD-179]].

**Files:** `src/magnelio/io/cad.py`, `src/magnelio/geo/imported.py`,
`src/magnelio/io/project.py` (rehydration, `colors`),
`src/magnelio/post/_colors.py`, `src/magnelio/io/paraview.py`,
`tests/unit/test_import_cad.py`,
`tests/integration/test_import_step_pipeline.py`,
`docs/methods/cad-import.md`,
`examples/tutorials/plot_16_cad_import.py`.

## DD-179 — Board import: the fabrication set is the contract, and no Boolean is 3-D

**Date:** 2026-08-20 (phase 2 of the import track opened by [[DD-178]]).
**Status:** Accepted — implemented, gated.

**Problem.**  A printed circuit board could only be modelled by
redrawing it with primitives.  For the boards Magnelio is aimed at —
filters, antenna feeds, connector launches — the layout already exists,
and a hand-rebuilt copy is both work and a discrepancy between what is
simulated and what is manufactured.  Nothing in the geometry API
addresses the two properties that make a board different from a
machined part: it is a *stack* of layers whose thicknesses live outside
the drawing, and its metal is two decades thinner than any affordable
cell.

**Decision.**  `magnelio.io.import_pcb(path, materials, *, copper,
name)` reads a fabrication export — Gerber X2 copper and profile
layers, Excellon drill files, and the `.gbrjob` job file — and returns
a `geo.Group` of `geo.ImportedSolid`, one per stackup layer plus one
per plated barrel.  Six choices are worth recording.

*The fabrication set, not a project file.*  Gerber/Excellon is what
every layout tool writes and every board house reads, so the import is
not tied to one vendor's format or release cycle.  Parsing
`.kicad_pcb` would have bound Magnelio to one tool and to a format
with no stability contract.  The job file is **required**, not
optional: it is the only member of the set that carries layer
thicknesses and the dielectric, and a board without them has no shape.
A `stackup=` escape hatch for job-less sets is deferred — inventing a
stackup is exactly the silent error the requirement prevents.  The job
file's layer assignment is cross-checked against the `TF.FileFunction`
each Gerber carries: a disagreement means a hand-assembled set, and
building it anyway would stack the layers in the wrong order, which is
a plausible model of a different board.

*Own reader, written from the specification.*  `gerbonara` exists and
is permissively licensed (there is no rule against copyleft
dependencies — Magnelio is LGPL itself; the standing rules are only
never to vendor pythonocc and never to copy vector-fitting reference
code).  It is not in the environment, its conda-forge availability is
unclear, and the format's usable subset is a few hundred lines.  The
readers (`io/_gerber.py`, `io/_excellon.py`, `io/_gbrjob.py`) are
written against the Ucamco specification, hold no kernel dependency,
and are therefore testable against hand-written files without OCC.
What they cannot express they **refuse**, naming file and line —
step-and-repeat, negative images, the deprecated image transformations,
G74, rectangular-aperture draws, moiré and thermal macro primitives.  A
missing pad in a board is not something a caller can be expected to
notice, so silence was never an option.

*The stackup is taken literally, and no Boolean is three-dimensional.*
Each layer is assembled as a **2-D face set** at z = 0 — apertures,
tracks and regions merged, clipped to the profile, drill circles cut —
and extruded once.  Extruding every pad into a 35 µm slab and fusing
the slabs would hand OCC thousands of sliver-prone operands; in the
plane the operands are coplanar by construction, which is the case
Booleans handle best.  Layer heights come straight from the stackup, so
adjacent layers meet on coincident faces with no fitting.  The origin
is the **top face of the topmost dielectric** and the stack grows
downwards, so adding an outer layer does not move the board.

*A plated hole is a solid barrel that exactly fills its cut.*  The same
circle is removed from every layer the hole crosses and the barrel is
built to it, so barrel and pad are face-coincident with no overlap —
the model's overlap check confirms it, and the integration gate asserts
that the union of all solids has exactly the sum of their volumes.
Solid rather than a plated wall around a void: the wall is a closed
conductor and encloses no field either way, and the void would add two
surfaces for nothing.

*Thin copper is the mesher's existing thin-sheet path, not a new one.*
Copper arrives at its real thickness because [[DD-059]]/[[DD-124]]
already resolve a PEC layer below the cell — one grid plane on the
substrate side, thickness in the sub-cell fractions.  The condition is
`is_pec` and a `MeshControl.min_cell_size` above the metal thickness,
and it holds here for a reason worth stating: a copper layer is **one**
solid spanning the whole board, so what is compared against the floor
is its thickness and never an individual track width.  A consequence
accepted rather than fixed: between the copper of an inner layer, where
a real board has prepreg, the model has background.  Filling it would
mean a 35 µm dielectric slab beside the copper, which forces the grid
to resolve exactly what the thin-sheet path exists to avoid.

*Names are the interface, and dielectrics are numbered.*  Copper layers
keep their stackup names (`F.Cu`, `In1.Cu`); barrels are `via_n` in
coordinate order.  Dielectrics are `dielectric_n` **regardless** of the
name the job file gives them, because layout tools name every core and
prepreg after its material and two layers called `FR4` would make a
material mapping ambiguous.  A loss tangent is reported and never
modelled: it is one number at a frequency the job file does not record,
and a frequency-independent tan δ violates Kramers–Kronig — the warning
names `DispersionModel.djordjevic_sarkar` as the route once the caller
supplies the frequency.  A dielectric with no stated permittivity
arrives as a construction body ([[DD-127]]), never as vacuum.

**Mechanics.**  Construction runs at a private power-of-two scale
(`_scaling.fine_detail_scale`): [[DD-120]]'s identity band assumes
features within about three decades of the model size, and a board
misses that by two, landing its Booleans three decades from
`Precision::Confusion()`.  The result is scaled back to meters exactly.
Coplanar faces are merged per **bounding-box cluster** (sweep over x
with union-find), so the isolated pads that make up most of a layer
skip the Boolean entirely; polarity is folded in runs of equal
polarity, because a clear object removes what was drawn before it and
nothing after.  Hole orientation in a face is decided by
`ShapeFix_Face::FixOrientation()` from containment rather than assumed
from winding — a hole added with the wrong winding yields a face whose
area is the *sum* of its boundaries, and nothing about the shape looks
wrong.  A profile layer is a line, so the board area is recovered by
chaining its segments through an endpoint graph at the file's own
coordinate resolution; a node not met by exactly two segments is an
error, applying [[DD-168]]'s lesson that a wire builder walks one
branch of a fork in silence.

Measured (`benchmarks/bench_pcb_import.py`, results archived): the
kernel's prism builder is superlinear in the face count of a compound —
2.9 s for 3600 pad faces handed over in one compound against 0.11 s for
the same faces raised one at a time — so `extrude` raises each face
separately.  With that, a 3600-pad board imports in 0.7 s end to end,
and the layer merge is not the dominant stage anywhere: 25 ms for those
pads (which skip it, being their own clusters) and 120 ms for the
all-connected serpentine-plus-zone case.

**Consequences.**  A board is ordinary geometry: it meshes, stores and
runs like anything drawn with primitives, and the integration gate
carries an imported microstrip through the mesher and asserts that its
copper was recognised as a thin sheet.  Solder mask and silkscreen are
outside the model by decision, not by omission.

**Deferred:** step-and-repeat, thermal and moiré macro primitives,
rectangular-aperture draws, arc-routed slots; `stackup=` without a job
file; per-net chunking of the layer fuse via the `TO.N` attribute (read
and discarded today); a sheet-preserving public `Union` of coplanar
`PlanarSheet`s, which would let this drop its private face helpers.

**Files:** `src/magnelio/io/pcb.py`, `src/magnelio/io/_gerber.py`,
`src/magnelio/io/_excellon.py`, `src/magnelio/io/_gbrjob.py`,
`src/magnelio/io/_pcb_geom.py`,
`src/magnelio/geo/_occ_backend.py` (`make_face_with_holes`,
`boolean_difference_many`, `unify_same_domain`),
`src/magnelio/geo/_scaling.py` (`fine_detail_scale`),
`tests/unit/test_pcb_jobfile.py`, `tests/unit/test_pcb_gerber.py`,
`tests/unit/test_pcb_excellon.py`, `tests/unit/test_pcb_geometry.py`,
`tests/unit/test_import_pcb.py`,
`tests/integration/test_import_pcb_pipeline.py`,
`benchmarks/bench_pcb_import.py`, `docs/methods/pcb-import.md`,
`examples/tutorials/plot_17_pcb_import.py`.

## DD-180 — Backend portability: describe capabilities, not compare modules

**Date:** 2026-08-21 (assessment prompted by an outside question about
Apple Silicon and AMD support).
**Status:** Deferred — evaluated, not scheduled.  Nothing implemented;
this entry exists so the measurements are not re-derived.

**Problem.**  The backend axis knows exactly two names.
`resolve_backend` hard-codes the pair in three places
(`_backend/array_api.py:39,135,207`), and the solver's capability test
is `self._use_gpu = xp is not np` (`solver/fit_td.py:312`): "not NumPy"
means "CuPy on CUDA" throughout.  A third array module would take the
CUDA path immediately and fail at kernel compile.  Thirteen further
sites duck-type the same question — `hasattr(x, "get")` ten times,
`type(f.Ex).__module__ == "cupy"` three.  Every prospective backend
therefore pays the same preparatory bill before it can be evaluated at
all.

**What the code actually costs to port.**  Less than expected.  Of
63,800 lines under `src/magnelio/`, ~750–850 are backend-specific
(1.2 %), in 14 of 407 files.  The package makes ~87 `xp.*` calls,
almost all `asarray`, `zeros`, `empty`; the time loop is ufuncs, slices
and fancy indexing — no `einsum`, no `linalg`, and no sparse matrix at
all (the material matrices are diagonals as flat 1-D arrays).  The
hand-written CUDA is 204 lines (`_operators/numba_kernels.py:232–437`)
with no shared memory, atomics, warp intrinsics or barriers: one thread
per element, bandwidth-bound.  And the array-stencil fallback of [[DD-032]]
(`update_E_stencil` / `update_H_stencil`, `numba_kernels.py:137–205`) already
runs on any NumPy-like module.

**Two traps behind that comfort.**  Dead-tile skipping ([[DD-100]]) and
graph capture ([[DD-092]]) are both gated on `update_E_fused_cuda is
not None` (`fit_td.py:533`, `:932`), so a portable path loses both.
And a backend that serves only the array API lands on the stencil
fallback — six curl temporaries and several passes over memory — which on
the same machine is likely *slower* than the fused Numba kernel, not
faster.  The CPU comparison point is never NumPy.

**Rejected: Metal, in every form.**  On Apple Silicon the CPU and the
GPU share one memory with one connection, so a bandwidth-bound kernel
can only win the difference in *achievable* bandwidth:

| Chip | Peak | CPU reaches | GPU reaches | GPU/CPU |
|---|---|---|---|---|
| M1 (base) | ~67 GB/s | 59 | 60 | 1.0× |
| M4 (base) | 120 GB/s | 103 | 100 | 0.97× |
| M1 Max | 409 GB/s | 224 (P-cores), 243 with E-cores | ~330 | ~1.4× |

Base parts: STREAM figures from *Apple vs. Oranges: Evaluating the
Apple Silicon M-Series SoCs for HPC*, arXiv:2502.05317.  M1 Max:
AnandTech's bandwidth-scaling measurement — the CPU cluster saturates
at 224 GB/s of a 409 GB/s fabric, which is where the gap comes from,
not from GPU speed.  STREAM flatters the CPU (ideal prefetching) and a
3-D stencil with three neighbour strides suits a GPU's latency hiding
better, so the practical figure is nearer 2× than 1× — but the ceiling
is still the bandwidth.  Against that stand a third kernel dialect to
maintain and `precision="double"` lost outright, since Metal GPUs have
no native FP64, so the [[DD-094]] opt-in would be unavailable.  The
capacity advantage of unified memory — the genuinely attractive
property, one to two orders of magnitude more cells than a consumer
card — is already open to the NumPy/Numba path, which sees the same
memory.

The order of these arguments matters: maturity is the *second* reason,
not the first.  MacMetalPy is disqualified on its own (alpha, one
contributor, dormant since March 2026, cp312 wheels only, and a silent
float64→float32 downcast documented as intended behaviour), but
1.0×…1.4× does not improve when a library grows up.  MLX fails
differently: its lazy, non-in-place array model works against a
leapfrog built on `+=` over persistent buffers.  `metalcompute` and
`metalgpu` are kernel launchers with no array API at all.  Reopen only
on a structural change — a separate, substantially faster GPU memory
path, or FP64 in the hardware.

**Also rejected.**  PyTorch as a universal backend.  One backend would
cover CUDA, ROCm and Metal at once, but it costs a ~2 GB dependency and
foreign memory management; the developer chose to keep the dependency
tree lean.

**Direction sketched, not decided.**  Should this be taken up, the
shape that follows from the findings is a `BackendSpec` — name, array
module, `is_device`, `to_host`, `supports_float64`, optional fused
`kernels`, optional `graphs`, and a cross-backend `tolerance` — behind
a registry, so the two-element name sets become one list and the error
texts enumerate what is registered ([[DD-176]] style).
`precision="double"` would then fail at the front door on a backend
without FP64 rather than downcast silently.  A backend certificate
would be the admission condition, built on the two oracles that already
exist: the NumPy replica of the fused kernel with identical in-thread
operation order (`_reference_sweep` in
`tests/integration/test_tile_skip_kernels.py`) and the two-sided
activation gate in
`tests/integration/test_gpu_backend.py::TestGPUSinglePrecision`, which
asserts that a path both agreed with the reference *and* differed from
the fallback — without it a backend reports green while crawling on the
stencil fallback.

**Why nothing can be merged on trust.**  CI runs `tests/unit` only, so
all three NumPy↔CuPy cross-checks are local runs; of 2489 tests, 24 are
GPU-gated and none is backend-parametrised, and `resolve_backend` has
no unit test at all.  The sharp bounds those cross-checks use (1e-12
absolute, `max|Δ| == 0`) are statements about identical operation
order, not about physics — another compiler may contract FMAs.  Any new
backend needs its own branch, its own hardware and its own
measurements before it is documented as supported.  Cost of that
verification is not the obstacle: Apple arm64 CPU runners are free for
public repositories, and a 192 GB AMD part rents for 1.71–2.59 $/h, so
a first session is under 15 $ — the expense is the CuPy source build,
not the bill.

**Deferred candidates.**  CuPy on ROCm is the strongest: identical
array API, so the stencil fallback works at once, and the CUDA source
translates to hiprtc nearly verbatim.  It needs a source build
(`CUPY_INSTALL_USE_HIP=1`), is not on conda-forge, and its launch
geometry — `_BLK = (32, 4, 2)` with the matching `DEFAULT_TILE =
(2, 4, 32)`, both measured on one Ada card ([[DD-100]]) — would have to
be re-measured for a 64-wide wavefront.  Intel `dpnp` is the same shape
for a narrower audience.  `array-api-compat` stays unnecessary at 87
call sites, as [[DD-006]] already judged.

**Measured on hardware, not estimated.**  The table above rests on
STREAM; the kernel itself was then measured on an M1 Pro (10-core,
8P+2E, 16 GB, 200 GB/s fabric) under both macOS 15 and Asahi Linux,
against the Ryzen/Ada desktop.  The harness is geometry-free — a vacuum
cube with PEC walls driving `FITTimeDomainSolver` directly, so no port
operators or DFT accumulators enter the per-step cost — and derives the
step cost from a three-point least-squares fit of `t(N) = setup +
N·cost`, keeping it clear of the second-scale scatter in setup (internal
record `investigations/fit-td-bandwidth/fitbench.py`, results in `investigations/fit-td-bandwidth/`).
Fit residuals stayed below 0.7 %.

Thread scaling at 4 Mcells, achieved bandwidth [GB/s]:

| Threads | M1 Pro, macOS | M1 Pro, Asahi | 7800X3D |
|---|---|---|---|
| 1 | 23.0 | 22.6 | 42.5 |
| 4 | 64.6 | 58.6 | 79.5 |
| 8 | 71.7 | **82.5** | **81.3** |
| 10 / 16 | 68.1 | 51.5 | 71.9 |

Three findings.  *The kernel reaches about 41 % of the fabric*
(82.5 of 200 GB/s), far below STREAM's 88 %, which is the three-stride
prefetch penalty the Metal argument above already anticipated — so the
CPU-side figure in that table is optimistic and the headroom a Metal
backend could claim is nearer the 2× estimated there than the 1.0–1.4×
tabulated.  The rejection stands on its other grounds.  *Efficiency
cores subtract throughput*: `prange` splits the loop statically and
every kernel ends on a barrier, so two slow cores hold eight fast ones
at the barrier — costing 5 % under macOS, which migrates threads
between clusters, and 38 % under Asahi, which does not.  *At this mesh size and
equal thread count Asahi is 15 % faster than macOS*; the sweep's apparent
macOS lead (76.0 vs 51.9 GB/s at 8 Mcells) is entirely the ten-thread
default, not the platform.  That lead narrows on larger meshes — see the
re-sweep below.

Consequence for every CPU run: set the thread count to the number of
performance cores.  Not wired into the library — `NUMBA_NUM_THREADS` is
the user's to set, and the performance-core count is not portably
discoverable.

Re-sweeping all three machines at eight threads sizes that consequence
and settles the platform question.  Achieved bandwidth [GB/s], vendor
default → eight threads:

| Mcells | 7800X3D 16→8 | Asahi 10→8 | macOS 10→8 |
|---|---|---|---|
| 0.12 | 72.6 → 100.0 | 33.3 → 50.8 | 33.8 → 36.2 |
| 1.00 | 174.9 → 222.1 | 48.7 → 73.2 | 54.6 → 67.2 |
| 4.02 | 61.4 → 81.7 | 51.2 → 80.3 | 69.1 → 70.6 |
| 8.00 | 45.6 → 48.3 | 51.9 → 80.8 | 76.0 → 75.1 |
| 16.0 | — | 58.0 → **82.9** | 76.5 → 80.5 |

Three corrections follow.  *The x86 gain is 6–38 %, not the 8–10 % first
recorded* — that figure came from a single 8 Mcell measurement, the one
point where the effect is smallest, because a saturated memory system
leaves no thread layout anything to win; where the working set still fits
in cache, SMT siblings cost L1/L2 and issue slots and the gain is
largest.  *The M1 Pro ceiling is 82.9 GB/s, 41 % of its 200 GB/s
fabric* — the thread scan predicted 82.5 from a different direction, so
the wall is real and reproducible, and it is a prefetch wall, not a
parallelism one.  *At eight threads the two operating systems converge*:
82.9 vs 80.5 GB/s at 16 Mcells is a 3 % difference, not the platform gap
the default-thread sweep suggested.  Asahi keeps a real 14 % lead at
4 Mcells; beyond that both hit the same fabric limit.  Choosing between
macOS and Asahi is therefore not a performance decision once the thread
count is set.

Cross-check on measurement quality: the GPU column was re-measured
unintentionally in the same sweeps (thread count touches only the CPU
path) and reproduced to better than 1 % — 419.9/679.1/916.8/538.4/426.7/
351.3 against 420.7/674.8/919.4/532.5/426.5/351.1 GB/s.

**Adjacent, not part of this.**  If Apple Silicon performance matters,
the lever is the fused Numba CPU kernel, not a GPU backend.  Thread
scaling is now measured (above) and says the ceiling is not the core
count; what remains untested is `prange` parallelism when `Nx` is small
and cache blocking across the three neighbour strides — the latter now
the most promising item, since the 41 % fabric utilisation is a
prefetch problem, not a parallelism one.  That is backend-agnostic and
benefits every CPU.

**Files:** none — nothing was implemented.

## DD-181 — An elliptical arc is a profile segment, not a spline

**Date:** 2026-08-21.

**Problem.**  `geo.Path` could draw lines, circular arcs and splines.
The outline of an accelerator cell — the TESLA mid-cell, the standard
shape of a superconducting linac — is a circular arc at the equator
joined by a tangent line to an *elliptical* arc at the iris
{cite}`aune2000`.  Without an ellipse primitive the iris had to be
approximated by a spline through sampled points: the geometry kernel
then carries a B-spline where the design carries an analytic curve,
the sampling density becomes a hidden model parameter, and the
tangent-line construction (which needs the ellipse's normal direction)
has nothing exact to attach to.  Lens profiles and elliptical
waveguides hit the same wall.

**Decision.**  `Path.ellipse_to(end, *, center, semi_axes, major_axis,
normal)` and its constructor form `Curve.ellipse_arc(start, end, ...)`.
The vocabulary follows `arc_to(center=, normal=)`: the ellipse is named
by its centre, its two semi-axes and the direction of the first, and
the arc is the one running counter-clockwise about *normal* — reversing
*normal* gives the complementary arc, exactly as for circles.  Both
endpoints must lie on the ellipse and in its plane (checked with the
join tolerance, DD-176 style: at the call, naming the offending point).

**Kernel detail.**  OCC's `gp_Elips` insists that the first radius be
the major one.  The user's `semi_axes=(a, b)` is a statement about
*directions*, not about which is longer, so `make_ellipse_arc` swaps
the frame when `a < b`: `(u, v, a, b, t) -> (v, -u, b, a, t - pi/2)`
describes the same point set, and the parameter shift keeps the arc's
endpoints where they were.  The swap is invisible at the API; the test
draws the same arc both ways and checks the sampled points against the
implicit equation to 1e-9.

**Bounds.**  The curve's scale box is `center ± max(a, b)` — the
circumscribed circle, the same bound the circular arc uses.

**Files:** `src/magnelio/geo/path.py`, `geo/curves.py`
(`_ellipse_frame`, `Curve.ellipse_arc`), `geo/_occ_backend.py`
(`make_ellipse_arc`), `tests/unit/test_geometry.py`,
`examples/tutorials/plot_14_profile_geometry.py` (summary bullet).

## DD-182 — A periodic face pair is a Bloch condition, and the phase advance lives on the analysis

**Date:** 2026-08-21.

**Problem.**  `"Periodic"` was a valid face declaration that the
eigenmode solver silently solved as PMC: `_build_pec_dof_mask` knows
only "PEC or not", `_estimate_sigma` treated every non-PEC pair as a
magnetic wall, and nothing coupled the two faces — the mesher's own
docstring pointed at an operator-level mechanism that did not exist
(`mesher.py:1167`).  CPML at the eigensolver went the same silent
way.  Meanwhile the one thing a periodic eigenmode problem is *for* —
the dispersion diagram of an infinite periodic structure, phase
advance 0…π per cell — was unreachable: it needs the far face to be
the near face times `exp(-iφ)`, which is a complex Hermitian problem
for 0 < φ < π.

**Decision.**  The pairing is imposed by a congruence transformation.
`_build_floquet_projector` builds a sparse `P` of shape
`(n_E, n_kept)` that maps every kept edge (interior and near plane) to
the full set, with the far-plane tangential edges as images of their
near-plane partners times `exp(-iφ)`; the reduced problem is
`P^H A P`, `P^H B P`, solved by the unchanged shift-invert machinery.
A kept edge is PEC when any of its images is.  Two periodic axes
compose (a corner edge is the image of an image, phases multiply).
`P` is real (entries ±1) for φ ∈ {0, π}, so those cases stay on the
real symmetric path with every backend available; in between the
problem is complex Hermitian and only the SuperLU backend takes it —
CHOLMOD (SPD Cholesky), pyamg (`symmetry="symmetric"`) and the folded
LOBPCG path are real by construction and raise `NotImplementedError`
with that reason.  The phase advance is a parameter of the *analysis*
(`AnalysisEigenmode(phase_advance_deg=…)`, a number for one periodic
axis, `{axis: degrees}` for several), not of the boundary
declaration: the mesh is the unit cell and is built once, the sweep
over φ runs on it.  This is also the vocabulary of the large suites
(face pair + phase shift).

**The metric trap.**  The first implementation assumed the far-plane
edges carried *half* a dual cell, so that `P^H B P` would add the two
halves and assemble the periodic metric by itself.  The empty
periodic box refuted it: exact for φ = 0, 1–9 % off elsewhere, with
ghost modes below the TE₁₀ line.  The material matrices book a *full*
dual cell on every domain face (the mirror convention behind the
natural PMC wall), so the congruence was double-counting the face
terms of the identified plane while the cell terms entered once.  φ = 0
passed only because every reference mode there has k_z = 0, where the
mismatched terms vanish.  The fix is to strip the far plane of its
metric before the congruence (`M_eps` on far-plane edges and `1/M_mu`
on the far-plane faces of the component along the axis set to zero)
and let the near plane's full cell stand for the pair.  The H-field
reconstruction keeps the unstripped `1/M_mu`, so the returned fields
obey the Bloch condition on both planes.

**Verification.**  Empty PEC-walled box, Bloch in z, on a uniform
8×4×6 grid against the *discrete* dispersion relation
ω² = c²Σ[(2/h) sin(k h/2)]² with k_z = (φ + 2πp)/L: the four lowest
modes agree to 1e-8 for φ = 0°, 60°, 90°, 150°, 180° (the exact
discrete reference leaves no discretisation error to hide behind); the
returned `Ey`/`Hz` satisfy far = exp(-iφ)·near to 1e-12.  Pillbox with
iris (period 50 mm): the φ = 0 spectrum is the union of the
{PEC, PEC} and {PMC, PMC} half-cell spectra, φ = π the union of the
mixed pairs, to 1e-3 (exact where the half cell is a grid
restriction; 2e-4 where a PMC mid-plane pulls the grid in, DD-164).
Note the wall assignment for a TM₀₁₀-type chain: the **electric** wall
in the iris plane gives the 0-mode (E_z even across the plane), the
magnetic wall the π-mode.

**Consequences.**  `BoundaryConditions` rejects an unpaired
`"Periodic"` face; the eigensolver rejects CPML instead of solving it
as PMC.  Mode fields of a complex phase are complex arrays — the
`FieldState`, the HDF5 store and the ParaView path follow the dtype —
and `EigenmodeResult.plot` draws them as the real snapshot at the
instant of maximum energy on the slice (global phase of an eigenvector
is arbitrary; `_real_snapshot` rotates by half the argument of
Σ a²).  `solver_info` records `phase_advance_deg`.
`_estimate_sigma` takes the Bloch wavenumber φ/L as the axis's lowest
k (nothing for φ = 0, as for PMC–PMC).  The time-domain
`PeriodicBoundary` is untouched (zero phase advance only; a TD phase
advance needs complex fields and is a separate decision).  The
half-cell band-edge calculation remains the classical route to the
two ends of a passband; everything in between now has a solver.

**Shift ladder.**  The first gallery run of tutorial 18 returned no
mode at 60°: the auto shift comes from the box dimensions (0.85 GHz
for a 103 × 103 × 115 mm box), the shaped cell resonates at 1.28 GHz,
and with `eps_r = 1` the ladder had a single rung, so the null space —
nearer the shift than the band — took every Ritz value and nothing
escalated.  The ladder now carries two rungs of `_SIGMA_RETRY_FACTOR`
above the empty-cavity estimate; they are climbed only on
under-delivery, so nothing changes for a solve that succeeds on the
first attempt.

**Files:** `src/magnelio/solver/_eigenmode_3d.py`
(`_periodic_axes`, `_resolve_phase_advance`,
`_build_floquet_projector`, `solve`, `_estimate_sigma`,
`_merge_physical_modes` Hermitian), `analysis/eigenmode.py`
(`phase_advance_deg`), `boundaries/boundary_conditions.py` (pair
check), `solver/eigenmode_result.py` (`_real_snapshot`),
`tests/integration/test_floquet_eigenmode.py` (new),
`docs/methods/eigenmode-analysis.md`, `docs/methods/boundaries.md`.

## DD-183 — Beam-coupling figures are post-processing of a kicker run, not a solver feature

**Date:** 2026-08-22.

**Problem.**  Pickups and kickers are the bread-and-butter devices of
beam instrumentation, and their figures of merit (kicker constant,
shunt impedance, transfer impedance, position sensitivity) are not
S-parameters: a port model has no beam.  The developer's working
notebook for a stripline pair carried the chain as an ad-hoc function
with stale conventions (`repmat`, cumulative integrals, a transverse
gradient read at whatever cell the monitor landed on) and a geometry
whose free parameters were picked "somehow" (`alpha = w/dia` made the
24 mm strips 12 mm wide).

**Decision.**  The workflow is documented as tutorial 19
(`examples/tutorials/plot_19_stripline_pickup_kicker.py`) on the
public API only; no library code changes.  The chain is the one of
Goldberg and Lambertson's primer: drive the device as a kicker, beam
voltage `V = ∫E_z e^{-jk_B z}dz` from a line monitor, transverse kick
by Panofsky–Wenzel from the transverse gradient of `V`, pickup transfer
impedances by reciprocity.  Three conventions are fixed by it:

- **Sign of the transit phase.**  The frequency monitors accumulate
  `Σ f·exp(+jωt)dt`, so a particle moving toward +z carries
  `exp(-j k_B z)`.  Verified by directivity: the wrong sign leaves 4 %
  of the beam voltage at the design frequency (the cancellation at the
  second gap for a beam running with the wave).
- **Symmetry plane as mode selector and drive.**  The plane between
  the two strips as `"SymmetryPMC"` is the sum mode, as
  `"SymmetryPEC"` the difference mode; one excited port in the half
  model drives both downstream ports (1 W each, full model), so the
  total drive power is 2 W.  The port report's `z_line_num` under that
  plane is the pair impedance (DD-155: PMC → parallel, PEC → series);
  one strip's impedance is `2·z` resp. `z/2`.
- **Electrical length of a real stripline.**  The ideal-gap formulas
  take the feed-to-feed distance `l + 2 g_c`, not the strip length:
  the `E_z` peaks sit at the feeds.  The residual 10–20 % below the
  ideal peaks is the transit-time factor of the extended end fields
  and is the reason to simulate.

**Dimensioning by port solver.**  The strip height above the pit
floor is found from 2-D port solves on a 10 mm slice (five heights,
~1.5 s each) and interpolated to 50 Ω in the difference mode
(h = 7.7 mm at φ = 60°, b = 25 mm, 5 mm side gap).  The coax feed
(1.52/3.5 mm) needs `min_cells_per_feature=8` to land within 5 % of
50 Ω; the `min_cell_size` floor does not refine it.

**Verification.**  Shape of every curve against the ideal stripline
with `l_el = l + 2 g_c`: lobes, nulls at `k_B l_el = π`, `1/k_B²`
decline of `R_⊥T²`; at 0.5 GHz K_∥ 0.41 (ideal 0.46), K_⊥ 2.83 (3.26);
position sensitivity 7.2 %/mm (ideal `4 sin(φ/2)/(φ b)` = 7.6).
Run: 41 × 44 × 209 cells, two runs of 11 s on the GPU; the whole
script ~2:20 on the CPU.  Derivation and
measurements: internal record
`investigations/stripline_coupler/{DERIVATION,MEASUREMENTS}.md` with
the probes `probe_zline2.py`, `probe_3d.py`, `probe_coax.py`.

**Not done.**  No `docs/methods` chapter — the chain is post-processing
of public monitor data and lives in the tutorial; a chapter becomes
due once a library-side helper (beam voltage, Panofsky–Wenzel) exists.
Only β = 1 is exercised; `BETA` is a parameter of the script.

**Files:** `examples/tutorials/plot_19_stripline_pickup_kicker.py`,
`docs/references.bib` (`goldberglambertson1992`, `panofskywenzel1956`,
`wendt2020`).

## DD-184 — A Touchstone export is the sub-matrix over the excited channels

**Status:** Decided 2026-08-23 (session 194, developer sign-off);
shipped same session.

**Problem.**  Reported as issue #3.  `to_touchstone()` refused to
write anything unless every *observed* channel had also been excited
(DD-112: "hard error when any channel was never excited — no silent
padding").  But `channels` follows from `n_modes` and `excitations`
from `run(excited=...)`, two independent user statements, and the
`run()` default excites one channel — so the default run was never
exportable.  A user who solves a two-port for three modes each to
inspect the mode table, then excites mode 0 on both, was told to spend
four more time-domain runs on modes he does not want.  The error text
also mis-stated the case: nothing would have been padded, because the
excited channels already span a fully measured square sub-matrix.

**Decision.**  Rows and columns of an export are the *excited*
channels (`SParameterResult.export_channels`), so every exported
entry was measured.  The reduction is physically exact, not a
truncation: an unexcited channel is not left open, it carries its
reflection-free port boundary for the whole run, which is the
matched-termination condition the definition of S-parameters asks
for.  The export is therefore the network seen with the omitted
channels matched — a one-port reflection export of a two-port is a
valid `.s1p`, and the developer's acceptance case (compare a measured
input impedance against simulation) is exactly that.

Two guards replace the blanket refusal:

1. **A warning for dropped propagating modes at an exported port.**
   Omitting a whole port is a deliberate cut through the network and
   is silent.  Omitting higher modes at a port that *is* exported is
   the subtle case — the file looks like a complete N-port while the
   mode conversion at that port is missing, so cascading it loses the
   scattered power.  Only modes whose cut-off lies inside the
   exported band can carry that power; evanescent ones draw no
   warning, since solving for more modes than one excites is ordinary
   practice.  The cut-offs come from a `_channel_cutoffs()` hook on
   the result contract, overridden by both implementations from their
   port-mode records — `SParameterResult` stays a pure data class and
   knows no cut-offs.
2. **The `.sNp` extension must match the exported channel count.**
   Touchstone 1.x carries the port count *only* in the extension (the
   body has no field for it), so a `.s6p` holding two-port rows is
   unreadable rather than merely misnamed.  A mismatch raises; a path
   without an extension gets the matching one
   (`to_touchstone("wr90")` → `wr90.s2p`).  Deliberately *not* an
   auto-rename: writing to a different file than the caller named
   loses the file for the next script in the chain.  This guard is
   also what still catches the real mistake the old refusal aimed at
   — "I meant to excite both ports and forgot" now fails as
   "`.s2p` declares 2 ports, the export covers 1".

`channels=` selects the exported sub-network explicitly (bare port
name = mode 0); entries that were never excited raise.  `to_skrf()`
follows the same rules.  Supersedes the export half of DD-112.

**Files:** `src/magnelio/post/sparameter_result.py`,
`src/magnelio/analysis/result_interface.py`,
`src/magnelio/analysis/scattering_td.py`, `src/magnelio/io/project.py`,
`tests/unit/test_schema_interop.py`, `docs/methods/ports.md`.

## DD-185 — A material argument accepts the built-in names as strings

**Status:** Decided 2026-08-23 (issue #2 review, developer sign-off);
shipped 2026-08-24.

**Problem.**  Reported as issue #2 (part 1 of 4).  Every script opened
with `pec = mio.Material.pec()` / `air = mio.Material.air()` before
the first shape existed — ceremony for materials that carry no
parameters and have exactly one canonical instance each.  Meanwhile
the rest of the public API already spells its closed vocabularies as
strings (`boundary_conditions={"xmin": "PEC"}`, `axis="z"`,
`plane="zmin"`, `family="coax"`, `port_model="modal"`); materials
were the odd one out.

**Decision.**  Every public material argument (`material=` on
primitives, Booleans, the profile verbs and `ImportedSolid`;
`background=` on `GeometryModel` and `Mesh.from_grid`, including
`from_grid` region tuples) accepts, besides a `Material` instance,
the names of the parameter-free built-ins: `"air"`, `"vacuum"`,
`"pec"` (case-insensitive).  A central resolver
(`materials/material.py: resolve_material`, applied at the call site
in the `geo/_validate.py` manner) maps them to **canonical cached
instances** of the factories and raises on an unknown name, listing
the valid ones.  The instances are shared deliberately: the mesher's
material bookkeeping is identity-based (`id(mat)` keys), so a fresh
instance per shape would inflate the material library.  Resolution
happens at construction time — shapes, models and the project store
only ever hold `Material` instances, and nothing downstream learns
about strings.

**Rejected.**

- *`"copper"` and friends* — a named conductor implies a curated
  material database: a specific σ with a citable source, dispersion
  models for dielectrics.  That is a data-stewardship feature with
  provenance obligations, not syntax sugar; split off as its own
  issue.
- *Sticky material* (issue #2 part 2: last used material becomes the
  default for subsequent shapes) — order-dependent hidden state; a
  lost `material=` line during copy-paste or reordering silently
  changes the physics and still runs.  It would also make
  "material-less shape" ambiguous, destroying the construction-body
  invariant `GeometryModel.add()` enforces (DD-127).
- *Abbreviated kwarg aliases* (issue #2 part 4: `bg=`, `mat=`) —
  contradicts the one-vocabulary rule (DD-153, DD-117: no aliases);
  with string shortcuts in place the verbosity the alias targeted is
  gone anyway.

**Files:** `src/magnelio/materials/material.py`,
`src/magnelio/geo/__init__.py`, `src/magnelio/geo/primitives.py`,
`src/magnelio/geo/operations.py`, `src/magnelio/geo/modifications.py`,
`src/magnelio/geo/imported.py`, `src/magnelio/io/cad.py`,
`src/magnelio/mesh/mesher.py`,
`tests/unit/test_material_strings.py`,
`examples/tutorials/plot_03_coax_smatrix.py`.

## DD-186 — The mesh carries the f_max it was built for

**Status:** Decided 2026-08-23 (issue #2 review, developer sign-off);
shipped 2026-08-24.

**Problem.**  Reported as issue #2 (part 3 of 4).  The same `f_max`
was passed twice in consecutive lines — once to `Mesh.from_geometry`,
once to `AnalysisScatteringTD` — in practically every script.  The
issue proposed a session-global "last used `f_max`" buffer; rejected,
because it makes results depend on execution order (notebook cells
out of order, two models in one session, sweeps).  The real defect is
elsewhere: the mesh is *built for* a frequency (`λ_min =
c₀/(f_max·√n_max)` sizes `h_max`) but forgot it — `Mesh` had no
`f_max` attribute — so the analysis could not know the mesh's design
frequency even to check it.

**Decision.**  `f_max` travels on the mesh, the same way boundary
conditions (DD-103), declarative ports (DD-109) and lumped elements
(DD-123) already do — the mesh is what reaches the analysis.

1. `Mesh.from_geometry` records its `f_max` argument on the mesh
   (`Mesh.f_max`); the wall-rewrite copies (`with_pec_boundaries`,
   `with_boundary_conditions`) carry it along, and the project store
   serialises it as a mesh attribute.
2. `AnalysisScatteringTD(f_max=None)` (new default) resolves to
   `mesh.f_max`; an explicit argument overrides, and a run with
   neither raises, naming the `from_grid` origin of the gap.
3. New guard: an explicit analysis `f_max` above `mesh.f_max` warns —
   the mesh undersamples the requested band, a mismatch that
   previously passed silently (`check_quality` runs at mesh time and
   never sees the analysis band).  An analysis `f_max` *below* the
   mesh's is legitimate (finer mesh than needed) and silent.
4. `Mesh.from_grid` takes no `f_max`; the attribute stays `None`
   there and the analysis keeps requiring an explicit value on that
   path.  Pre-DD-186 stores rehydrate with `None` — the explicit
   behaviour they always had.

The eigenmode analysis takes no `f_max` and is untouched.

**Files:** `src/magnelio/mesh/mesher.py`,
`src/magnelio/analysis/scattering_td.py`, `src/magnelio/io/project.py`,
`tests/unit/test_mesh_design_frequency.py`,
`examples/tutorials/plot_03_coax_smatrix.py`.

## DD-187 — Post-hoc reference-plane shift (`result.deembed`) on the exact discrete chain dispersion

**Status:** Decided 2026-08-24 (developer approved the post-hoc form
over a CST-style per-port declaration); shipped 2026-08-24.

**Problem.**  Comparing simulated S-parameters against measurements,
against other tools, or against an on-grid device under test requires
moving the port reference planes off the domain boundary — classically
by multiplying with `exp(+γd)` per touched port.  Two open questions:
where exactly the DTBC's reference plane sits (half-cell/half-step
offsets would poison any shift), and which dispersion to shift with —
on the coarse meshes the discrete-port how-to guides target, grid
dispersion reaches degrees of phase, and a continuum `exp(-jβd)`
misattributes exactly that to the device under test.

**Measurement** (internal record
`investigations/port-deembedding/MEASUREMENTS.md`, reproduced by
`validation/deembed_uniform_line.py`): on a uniform matched line at
8 cells/λ, cancelling S21 with the discrete chain root `lambda(z)`
(DD-054/DD-055, `(r, q)` from `dtbc_line_params`, `dz` from
`port_normal_dx`) leaves −119.9 dB (TEM) resp. −67.4 dB (TE10) —
the run's own floor in both cases (S11 −123.2 / −76.7 dB) — with
**zero reference-plane offset**: the DTBC plane *is* the port plane.
The continuum factor leaves 9.7° (TEM) resp. 3.2° (TE10) of grid
dispersion, −15/−25 dB.

**Decision.**

1. **Post-hoc, not at declaration:** `result.deembed({"port": d})` on
   the scattering-result contract (RAM and store-backed alike)
   returns a new `SParameterResult` referenced at planes shifted `d`
   into the domain (negative = outward).  No re-run to iterate; a
   port-level declaration can later delegate to this.
2. **Discrete dispersion first:** channels with certified line
   parameters shift by `lambda^{-d/dz}`, evaluated exactly *on* the
   unit circle (`post/deembed.py:_chain_lambda_log`; the branch is
   decided by the real-valued `A(ω)` alone, so passband magnitudes
   stay exactly 1 — the `_EDGE_OFFSET` of the off-circle
   `lambda_symbol` would bias magnitudes by `O(1e-8 · d/dz)`).
   Channels without certified parameters fall back to the mode's
   continuum `γ(ω)`; lumped channels raise (no feed line).
3. **Shared accessors:** `phase`/`plot_s` moved from
   `ScatteringResultMixin` into `SDerivedAccessors`
   (`post/sparameter_result.py`), inherited by both the run results
   and `SParameterResult` — a de-embedded matrix answers the same
   calls as the result it came from.
4. Below cut-off the factor grows as `exp(+α̂d)`; those bins keep the
   diagnostic character the raw values have (no masking, matching the
   S-parameter convention).

The shift assumes the port cross-section continues over the shifted
length and uses the port-plane `dz` chain; on feed meshes graded along
the normal, the residual is the (second-order) dispersion difference
between the local and the port-plane spacing.

**Files:** `src/magnelio/post/deembed.py`,
`src/magnelio/post/sparameter_result.py`,
`src/magnelio/analysis/result_interface.py`,
`src/magnelio/analysis/scattering_td.py`, `src/magnelio/io/project.py`,
`tests/unit/test_deembed.py`,
`tests/integration/test_deembed_line.py`,
`tests/integration/test_result_contract.py`,
`validation/deembed_uniform_line.py`.

## DD-188 — A "How-to guides" gallery beside the tutorial curriculum

**Status:** Decided 2026-08-24 (developer chose the plan in the
planning discussion); shipped 2026-08-24.

**Problem.**  The tutorial gallery had become two things at once: a
numbered learning curriculum (01–12 core workflow, then feature
chapters) and, with tutorial 19 (striplines as pickups/kickers), a
home for task-oriented application recipes that teach no new library
capability.  The planned discrete-port characterisation scripts
(DD-189) would have stretched that further, and the "capstone" filter
sat at position 13 with six chapters after it.

**Decision.**

1. Second sphinx-gallery instance: sources in `examples/howto/`,
   rendered to `docs/howto/`, own toctree caption **"How-to guides"**.
   Same machinery as the tutorials — `plot_` pages execute at build
   time (a silent regression test) and every page gets the
   auto-generated `.py`/`.ipynb` downloads.
2. How-to pages are **unnumbered** (`plot_stripline_pickup_kicker.py`),
   alphabetical order; they are recipes to adapt, not a sequence to
   follow.
3. Tutorial 19 moved there (renamed, self-references "tutorial" →
   "guide"; its references *to* tutorials 06/09 stay).  The old
   `tutorials/plot_19_…` URL lapses — noted in the changelog.
4. The filter capstone stays at 13; its intro now anchors it as the
   close of the core workflow 01–12 and names the later chapters as
   feature chapters read on demand.

**Rejected:** renumbering the tutorials so the capstone comes last
(breaks `/stable/` URLs and the number references in tutorial prose
for cosmetics); a section named after the first content ("Discrete
Ports") instead of the generic "How-to guides" (the section will
grow: convergence practice, mesh practice, …); keeping the stripline
page a tutorial (it teaches no capability and the curriculum reads
tighter without it).

**Files:** `docs/conf.py`, `docs/index.md`, `.gitignore`,
`examples/howto/README.txt`,
`examples/howto/plot_stripline_pickup_kicker.py` (moved from
`examples/tutorials/plot_19_stripline_pickup_kicker.py`),
`examples/tutorials/plot_13_dielectric_filter.py`.

## DD-189 — Discrete-port characterisation as how-to guides, one per line type

**Status:** Decided 2026-08-24 (planning discussion with developer;
optimiser deliberately left out); coax guide shipped 2026-08-24,
microstrip and CPW planned.  **Amended 2026-08-24** after developer
review of the prototype: de-embedding is out of the guide entirely —
the target simulation has no de-embedding, the one gap *position* is
the compromise the user commits to — so the gap position is a
geometric knob with its own re-run sweep, and the phase ruler is a
reference run instead (see 2.).  **Amended again 2026-08-24**
(developer: the single page was overloaded): the material is split
into **four pages** — *Lumped ports: investigations* (long: the
measurement principle plus the sensitivity sweeps for all three line
types) and one compact *Lumped port tuning* download tool each for
coax, microstrip and CPW (given quantities → knobs → derived →
scoreboard, no sweeps).  File names sort the overview first and the
tools directly after it.  Shipped 2026-08-24, all three line types.
**Amended a third time 2026-08-24** (developer review of the CPW
pages, with a reference model in the private workspace): the
slot-port-plus-dummy-resistor termination is replaced by the
practitioner's scheme — the strip ends an **end gap** short of the
ground metallisation behind it and the lumped port bridges that gap
*longitudinally* on the pair's symmetry plane (``SymmetryPMC`` half
model, DD-172), exactly the coax picture and exactly the port the
target simulation excites through.  Open lid = **PMC** boundary, not
a metal cover; the closing ground plate falls out of one boolean.
Same knobs as the coax (end-gap width, end-gap position, impedance);
no resistor, no ``elements=``, no tight-box requirement.

**Problem.**  Discrete (lumped) ports are approximate line
terminations.  Vendor rules of thumb ("terminate a coax like this")
say nothing about the residual reflection and phase error on a
*given* grid, nor up to which frequency the termination is usable —
and the behaviour is grid-dependent, so no fixed rule can.

**Decision.**  Per line type, a how-to page (public API only,
DD-188 gallery) that **measures** the user's own termination instead
of prescribing one:

1. A waveguide port — reflection-free by construction — launches the
   exact grid mode down a short uniform line onto the lumped port
   under test.  `|S11|` is the termination's self-reflection; the
   page prints the band edge where it crosses −20 dB.
2. The phase ruler is a **reference run**: the same line with
   waveguide ports at both ends, transmission phase = the exact
   propagation of *this grid* over the reference length.  The
   termination's phase error is the difference of the lumped run's
   `arg S21` against that ruler — no closed-form dispersion needed,
   so microstrip/CPW need no extra machinery.  (The prototype read
   the phase error from a de-embedding sweep instead; dropped on
   developer review — the target simulation cannot de-embed a lumped
   port into position, the geometry itself must be optimised.)
3. Knobs on the page, each the compromise carried into the target
   model: gap length (re-run sweep → reflection level), **gap
   position relative to the reference plane** (re-run sweep → phase
   error; on the example grid with 0.5·r_i cells: 29.0° with the gap
   starting at the plane vs 5.7° starting 2–3 gap lengths before it;
   on a TEM line the best position holds across the band, on
   dispersive lines it becomes a band compromise), port impedance
   (grid line impedance from `solve_ports` vs catalogue value →
   low-frequency reflection floor).
4. **No built-in optimiser.**  2–3 parameters, seconds per run, and
   the sweep shows the sensitivity that a black box would hide; the
   page names `scipy.optimize` for readers who want the last
   fraction of a dB automated.
5. Measured values appear as properties of the example grid, never
   as transferable rules — the page's closing section says to
   re-measure whenever cross-section, resolution or band change.

The coax test grid pins `max_cell_size = min_cell_size` so the feed
is uniform and the reference run and the candidate runs see the same
line per unit length; the knob the user sets is the cross-section
cell size their production mesh will have (PCB cross-sections use
`min_nodes_per_wavelength` instead — multi-scale).

**Line-type findings** (internal record
`investigations/discrete-port-guides/MEASUREMENTS.md`):

- *Polarity normalisation*: the sign of a solved mode profile is a
  convention, so lumped-port vs waveguide-mode polarity is ±180°
  arbitrary; the phase error is referenced to the nearest multiple of
  180° at the low band edge.  (Surfaced on microstrip, where the raw
  error read ≈180°.)
- *Microstrip*: vertical port trace-end → ground; no gap-length knob;
  position optimum near −1.0·h_sub on the example grid (23.3° →
  2.9°), and the position compromise is frequency-dependent
  (dispersion) — the curves tilt.
- *CPW* (third amendment's scheme): symmetry-plane end-gap port on an
  open Rogers-4003 structure (PMC lid), measured optimum of the gap
  position at ≈ **+16·s** *beyond* the reference plane (21.4° →
  0.74°) — opposite sign to coax/microstrip, because the mode's
  return current detours through the ground plate behind the gap;
  gap width 1s/2s/4s → −26.6/−19.6/−14.6 dB.  Two findings from the
  discarded slot-loaded scheme stay valid as general knowledge (not
  in the guides): a lumped element declared after
  `Mesh.from_geometry` must be passed via `elements=` or it is
  silently absent (elements travel on the mesh, DD-123), and a
  shielded test fixture must stay single-mode over the band or its
  box modes masquerade as termination error.
- The empty-boolean crash the developer hit while building the CPW
  model (`add(a − b)` with b ⊇ a: `plot()` dies in an uncaught C++
  `std::invalid_argument`, the mesher in a cryptic `GridLines`
  error) is recorded as KB-026.

**Files:** `examples/howto/plot_lumped_port_investigations.py`,
`examples/howto/plot_lumped_port_tuning_coax.py`,
`examples/howto/plot_lumped_port_tuning_microstrip.py`,
`examples/howto/plot_lumped_port_tuning_cpw.py`,
`docs/methods/lumped-elements.md` (pointer).

## DD-190 — The 3D view moves from pythonocc's pythreejs renderer to PyVista, with an axis-aligned cutting plane

**Status:** Decided 2026-08-25 after a two-day spike with the developer
(planning 2026-08-24, browser tests 2026-08-25); implemented 2026-08-25.
**Amended 2026-08-25** after the developer's browser review of the
implementation: (a) the cut-cell sheet reached the actor as a
`vtkRectilinearGrid`, which trame's vtk.js serialiser does not know —
the widget stayed blank with a JS error whenever a mesh was shown; every
actor dataset is polydata now, and a test guards it.  (b) Features
follow the cut (wires clipped, ports/elements/labels hidden when their
anchor lies in the removed half) — they had stayed in view.  (c) Names
are flat 3D text (`Text3D` polydata) so they render in every mode; the
server-only screen labels are gone.  (d) A *Show* menu in the toolbar
hides or shows object groups (solids, grid lines, cut cells, ports,
elements, wires, labels, symmetry planes, domain box).
**Amended again 2026-08-25** (third review): the grid wireframe on the
six domain faces — a cage around the model — is dropped; the grid shows
on the cutting plane only (still a *Show* toggle).  Port names lie in
the port plane; element names keep facing the initial camera.

**Problem.**  `GeometryModel.plot()` was pythonocc's `JupyterRenderer`,
a thin wrapper over pythreejs (2.4.2, unmaintained since 2023).
Magnelio already replaced its scene assembly (`_display_renderer`) to
survive metre-scale geometry, and the camera could not pan at all —
pythreejs's `CombinedCamera` only impersonates an orthographic camera,
so `OrbitControls` computed the pan from undefined fields (fixed on
2026-08-24 by a real `OrthographicCamera`, `f6836a0`).  Nothing beyond
orbit/zoom was reachable: no cutting plane, no grid, no ports or wires
in 3D, and no way to render the view into the documentation (the
tutorials only *mention* `model.plot()`).

**Decision.**

1. **Backend: PyVista** (`pyvista` becomes a core dependency; VTK
   already was one for the ParaView export).  The scene is the same
   VTK data the export writes: solids as `PolyData` from the shared
   tessellator (`io/paraview.py::_tessellate_shape`, extended by an
   angular deflection), the FIT grid as a `RectilinearGrid` with the
   mesher's per-cell material as cell data.  Field monitors reach the
   viewer on the same path later (out of scope here).
2. **Cutting plane: axis-aligned, slider-driven** — normal `x/y/z`,
   position, flip side, undo, reset in the widget toolbar; `cut=`
   sets the initial state (and is the only handle in a screenshot).
   One plane cuts *every* solid (closed caps via
   `clip_closed_surface`) and exposes the grid cells on the cut as a
   sheet of cell faces coloured by assigned material, offset a hair
   into the removed half so it never fights the caps for depth.
   A free plane grabbed in 3D (PyVista's `add_mesh_clip_plane`) was
   tried and rejected on developer review: the handle competes with
   the camera for the mouse, has no reset, exists only under
   server-side rendering, and an oblique grid cut carries no
   information.  This is also how the cutting plane of the EM suites
   users come from behaves.
3. **Rendering mode: client-side by default** (`mode="client"`,
   vtk.js in the browser).  Verified in the developer's browser as the
   crisper picture, and it needs no OpenGL in the kernel.  Server
   rendering (`"server"`, `"trame"`) stays available for scenes too
   large for the browser.  Outside a notebook the same call opens a
   VTK window (scripts) or yields a screenshot (Sphinx-Gallery via
   PyVista's scraper — the tutorials now show the 3D view).
4. **Transport: trame's own websocket, never the Jupyter comm
   extension.**  JupyterLab ≥ 4.5 / Notebook 7 execute comm messages
   in kernel subshells (setting `commsOverSubshells`, default
   `perCommTarget`) and ipykernel ≥ 7 runs those in a separate
   thread; `trame-jupyter-extension` carries wslink over such a comm,
   so VTK rendered from a thread without the GL context — black
   frames, and a kernel abort whenever the uninitialised
   `GL_MAX_DRAW_BUFFERS` read came back negative
   (`vtkOpenGLFramebufferObject::ActivateDrawBuffers`,
   `bad_array_new_length`).  Proven by replaying the comm transport
   through `jupyter_client` with and without a `subshell_id` header
   (internal record `investigations/viewer3d/`).  The viewer sets
   `pv.global_theme.trame.jupyter_extension_enabled = False` unless
   the user pinned `PYVISTA_TRAME_JUPYTER_MODE`.  Upstream report
   pending.
5. **Overlays** as in `plot_cross_section`, same colours: thin wires
   and discrete ports / lumped elements as tubes, face ports as
   translucent windows on the domain face, names as flat 3D text
   (polydata — vtk.js has no label mapper), symmetry planes as tinted
   sheets, the domain box as an outline; display in millimetres
   (`scale_mm`).  Features follow the cut.
6. **Process settings** the viewer applies once: the Viskores
   (VTK-m) filter overrides are switched off (they try CUDA first and
   fall back after ~25 s per rectilinear slice), and trame_vtk's
   VTK 9.6 deprecation chatter is filtered.
7. **API.**  `model.plot(mesh=None, *, cut, flip, show_ports,
   show_wires, show_grid, mode, size, render_edges, edge_color,
   quality, scale_mm, camera)`; the four legacy keywords keep their
   meaning.  Returns the `pyvista.Plotter` for `mode="none"`, otherwise
   displays and returns `None`.  `plots.show_geometry` is the same
   function; `post.plot_3d` is its home.

**Consequences.**  pythreejs and the pythonocc renderer are gone from
the code path (pythonocc still depends on pythreejs; nothing to
uninstall).  `pyvista` joins the core dependencies (pure Python, VTK
was already there); the widget needs the `[jupyter]` extra
(`trame`, `trame-vtk`, `trame-vuetify`, `nest_asyncio2`).  The
conda-forge recipe must follow (feedstock PR).  CI and the docs build
render off-screen (`PYVISTA_OFF_SCREEN`).  KB-026 (empty boolean
aborts `plot()`) closes as a side effect: a shape without extent is
skipped with a warning instead of reaching the tessellator.

**Not done here (follow-ups).**  3D field views (`MonitorResult`,
eigenmodes) on the same datasets the ParaView export builds; a
`Mesh.plot()` without CAD (the discretised model alone).

**Update 2026-08-26 — interactive scenes in the docs.**  The gallery
now scrapes 3D views with PyVista's `DynamicScraper`: every
`model.plot()` becomes a `sphinx_design` tab set, "Static Scene"
(the PNG, also the thumbnail) and "Interactive Scene" (the scene
exported as `.vtksz`, rendered by trame-vtk's offline viewer — vtk.js
in an iframe, `_static/static_viewer.html`, 1 MB once per site).
Measured on tutorial 02: 10 kB for the geometry scene, 50 kB with the
grid sheet; build time unchanged.  The browser tab has no toolbar: the
cutting plane is frozen where the script set it, because the slider
and the *Show* menu are trame state that Python re-clips.  New docs
dependency `sphinx-design` (`environment.yml`, `[docs]` extra); the
directive imports `trame_vtk`, which the docs environment already
carried.  Sphinx trap: after adding the extension the doctree cache
must go, or the `tab-set` directive stays "unknown" from the previous
build.  Opt-out per script: `PYVISTA_GALLERY_FORCE_STATIC = True`.

**Measurements** (developer machine, RTX 4070 SUPER, 2026-08-24/25):
coax + 80k-cell grid scene 0.9 s to build; 27 M-cell rectilinear
grid: slice 0.5 s, re-render 11 ms; `glyph(tolerance=…)` 17 s (point
merging — never use it, subsample instead: 10 ms).

---

## DD-191 — Geometry-edge planes: a grid plane wherever a B-rep edge lies flat in an axis-normal plane, floored by `max_edge_refinement`

**Status:** Decided 2026-08-25 with the developer (planning discussion on
mesh generation: which of the rule-based, wavelength-local, edge-refinement
and adaptive strategies to pursue; this is the first); implemented
2026-08-25.

**Problem.**  The dielectric-resonator worksheet (internal record
`investigations/dr_filter/MEASUREMENTS.md`, M4/M4a) found the chamfer of
a ceramic puck to have *no* effect on its eigenfrequency — three
bit-identical values for 0 / 0.2 / 0.5 mm on a 1 mm grid, then a 16 %
jump at 0.8 mm.  M4a cleared the conformal material matrices: a radius
change of a twentieth of a cell moves f0 by a clean, linearly scaling
19 MHz.  The DD-051 entry `M_eps = eps_bar * A_dual / L_primal` averages
over the dual face *transverse* to the edge; a feature that thins the
puck *along* the z-edges inside the top and bottom cell layer has no
lever until it reaches the layer's midplane, and then switches on in one
step.  That is the construction, not a defect — but it means the mesher
must put a plane where the feature starts, and it never did:
`_face_critical_planes` reads planes, cylinders and spheres, a chamfer
is a cone, and the circle where the cone meets the cylinder is an edge.
No warning either: the feature vanished silently.  The developer knew
the artefact from no commercial suite; those meshers place fixpoints on
every CAD edge and vertex and prune them by a cell-ratio limit.

**Decision.**  A second, *edge* pass over the B-rep, and a soft plane
class with a reported floor.

1. **Which edges.**  An edge contributes the coordinate `a` on axis `k`
   iff the whole edge lies in the plane `x_k = a`: a straight edge on
   every axis where its end points agree (an axis-parallel edge yields
   its two transverse coordinates, an edge in a tilted plane the plane's
   axis, a skew edge nothing); a circle or ellipse on the axis its
   normal is parallel to, at the centre's coordinate (exact analytic
   position); any other curve on every axis along which its
   geometry-only bounding box has zero extent.  Only *sharp* edges count.  Skipped are seam edges
   (`BRep_Tool.IsClosed(edge, face)`: a cylinder's seam is a straight
   line through `(R, 0)` and would put a phantom plane through the
   axis; a sphere's seam meridian lies in an axis-normal plane through
   its centre), degenerated edges, and edges between two faces of the
   *same* analytic surface — a Boolean fuse leaves coplanar sub-faces
   unmerged, and the split line between them is not geometry (found on
   tutorial 06's magic tee: a plane at `y = 0` through the junction,
   in the middle of a flat wall).  Measured on the test bodies: chamfered annulus → `z = c, H − c` and nothing on x/y;
   filleted brick → the fillet onsets on all three axes; sphere,
   torus, cone, axis-aligned and tilted cylinders, brick → nothing new.
   Deliberately *not* the in-plane tangent extrema of the feature
   circles (`±(R − c)`): they are second-order refinements of a
   transverse boundary the conformal average already resolves; the
   plane normal to the edge's own plane is the one that ends the
   along-edge blindness.

2. **Soft class.**  Edge planes join after `_merge_axis_planes`
   (material + forced) in `_merge_feature_planes`: a candidate within
   the clustering tolerance of an existing plane *is* that plane
   (absorbed silently); one closer than the edge floor to *any*
   material/forced plane or to a previously kept edge plane
   (keep-first, ascending) is dropped and recorded; the rest join with
   `is_material = False`, `is_feature = True`.  They never move a
   material plane (KB-013 stays intact) and never drive
   `min_cells_per_feature`: an edge plane asks for **one** cell across
   each interval it bounds — enough for the cell's midplane to see the
   feature — entered into the shared per-axis `h_fine` so the
   neighbouring intervals ramp from that size (the generator grades
   from a common `h_fine`, DD-061 contract I4).  A thin sheet's thin
   axis is exempt for the sheet's own shape, and the sheet's far face
   — which re-enters through the *edges* of the imprint in the
   surrounding dielectric just as it does through its faces — is
   dropped globally before the merge (tutorial 10's 17 µm
   metallisation came back as a "dropped edge plane" warning until it
   was).

3. **Floor and warning.**  `edge_floor = max(h_max /
   max_edge_refinement, min_cell_size)`, new
   `MeshControl.max_edge_refinement = 4.0` (`0` disables the pass —
   the 0.4.4 meshes bit-exactly).  The ratio bounds the time-step cost
   of resolving small edges (one explicit step, bounded by the smallest
   cell anywhere).  Drops are reported by **one warning per mesh**
   with the count per axis and the *coarsest* dropped plane — the one
   nearest to being resolved, whose feature matters most — with the
   cell it would have created, the floor, the binding parameter and
   the `max_edge_refinement` that keeps it.  (The first version named
   the finest drop per axis: on tutorial 10's ring divider that was a
   43 µm sliver where a line's side wall meets the ring, 43 µm from
   the ring's own tangent plane — noise that trains the reader to
   ignore the message M4 lacked.)

4. **Domain-face buffer.**  A boundary interval bounded by an edge
   plane holds one cell by design.  The DD-107 buffer (three
   equidistant cells at a domain face) would triple it: at a *declared*
   port face it still may, floored by the edge floor (the port needs
   its cells); at the port-blind fallback faces it is skipped there
   (measured before the rule: a 0.5 mm chamfer layer at the housing
   wall became three 0.167 mm cells and a growth-ratio warning, for a
   port that does not exist).

**Measured** (`validation/edge_plane_chamfer_certificate.py`, the
worksheet's coarse grid, mnpw 12 at 3.5 GHz, h_max 1.064 mm, lowest
mode; 2026-08-25, re-based 2026-08-27 after DD-199's contour winding
gave the ring its air bore — the 2026-08-25 column values, 2.32784 GHz
plain and 2.70978 / 2.74736 GHz at 0.8 mm, were measured with the bore
booked as ceramic, KB-031):

| chamfer | ratio 0 (0.4.4) | ratio 4 (default) | ratio 8 |
|---|---|---|---|
| 0.0 mm | 2.65658 GHz, 4704 cells | same | same |
| 0.1 mm | 2.65658 (invisible) | 2.65658, **warned** | 2.65658, warned |
| 0.2 mm | 2.65658 (invisible) | 2.65658, **warned** | 2.66561, 11760 cells |
| 0.3 mm | 2.65658 (invisible) | 2.67973, 9408 | 2.67973 |
| 0.5 mm | 2.65658 (invisible) | 2.72840, 7056 | 2.72840 |
| 0.8 mm | 2.82089 (jump) | 2.86641, 5488 | 2.86641 |

The legacy column reproduces M4's plateau-and-jump pattern (M4's own
digits carry the KB-031 bore).  Resolved chamfers move f0
monotonically, and the steps follow the geometry: the ceramic a
chamfer removes grows with c², and Δf0 / Δ(c²) is 0.26 → 0.30 → 0.35
GHz/mm² along the default chain and 0.23 → 0.28 → 0.30 → 0.35 along
the ratio-8 chain (spread 1.38× and 1.57×), so the certificate checks
monotonicity and that spread (limit 3×; a chamfer switching on at the
cell midplane gives one near-zero step and one large one).  The first
measurement had read the chain as uneven — "the first tenths of a
millimetre matter most", the 0 → 0.2 mm step at ratio 8 was 0.143 GHz
against 0.009 GHz now — and pinned "no single step above half the
legacy jump"; that unevenness was the filled bore (the chamfer at the
bore edge cut into ceramic that should have been air), not physics,
and the old criterion would reject the corrected chain (the 0.5 →
0.8 mm step legitimately carries 84 % of the legacy jump).  Dropped
chamfers give the plain-puck grid bit-for-bit (unit test) and the
plain-puck f0 to solver noise.  DD-062 sentinel 30/30, unit suite
green.

**Side finding — tutorial 13's "9.9 % drift" was mostly this
artefact.**  The filter capstone compared its single resonator on two
grids (mnpw 10 and 16) and read a +9.9 % move of f0 as the general
non-convergence of a high-permittivity puck.  On the coarse grid
(dz = 1.2 mm) the 0.5 mm chamfer was invisible, on the finer one
(dz = 0.75 mm, half-cell 0.375 mm < 0.5 mm) it had switched on: the
drift was the chamfer appearing, not the resolution.  With the edge
pass both grids resolve the chamfer and — since the 0.5 mm feature
layer now sets `h_fine` on both — come out nearly identical (drift
−0.0 %; −0.1 % on the corrected annulus, 2026-08-27).  The tutorial is re-based on a grid pair that scales both
mesh knobs (see the how-to *Mesh convergence*), and its text no longer
quotes the ten percent.  Cost of the pass on the tutorials: 13
+50–80 % cells (dz 1.0 → 0.49 mm), 18 +10 % (iris/equator circles),
10 +20 % (line/ring junctions; the tutorial now sets
`max_edge_refinement=5` to keep the last junction plane instead of
printing the warning), 01–09, 11, 12, 15–17 unchanged (14 does not
mesh).

**Amendment 2026-08-26 (KB-028).**  The integration suite found the
"same analytic surface" skip too narrow: it tested *two* faces on one
surface.  A closed surface touching a flat face — the cylindrical
cavity of `test_conformal_convergence.py`, inscribed in its PEC block
— is split by the Boolean along the touching line *together with the
flat face*, so that straight edge carries four faces (two coplanar
wall halves, two coaxial cylinder quarters) and passed as geometry,
contributing its transverse coordinate: the plane through the
cylinder axis the seam rule exists to keep out.  The 5 mm cavity grid
went 7 × 7 × 5 → 8 × 8 × 5 with nodes on the tangency cusps, and the
Dey–Mittra TM010 error 3.7 % → 6.0 % (staircase 11.4 % → 4.1 %; the
same 8 × 8 grid pre-dates the change as the 4 mm fixture, with the
same numbers — DM on cusp-node grids is a separate, older observation).
The rule now groups the ancestor faces by analytic surface and skips
an edge at which *every* group has two or more members — every
surface continues across it.  Found on the way: the cylinder branch
of `_same_surface` called `gp_Ax1.Distance` (no such method); the
per-shape `except Exception` of the edge pass had hidden it, so a
shape whose first cylinder-split edge reached that branch silently
contributed *no* edge planes at all.  Now `gp_Lin(axis).Distance`.  A fillet onset keeps its plane (flat
face and fillet each sit on one side); a fuse line crossing an onset
splits the onset into two-face edges that are kept.  The cavity
grids are bit-identical to 0.4.4 again; the DD-194 singular-edge
pass shares the rule (tangency cusps on a domain face never counted
there anyway).  Gate:
`TestEdgeFeaturePlanes::test_tangency_cusps_of_an_inscribed_cylinder_contribute_nothing`.

**Rejected.**
- *Counting edge planes as material gaps* (`min_cells_per_feature`
  across the chamfer layer): a 0.2 mm chamfer would force 0.05 mm
  cells; one cell is what the dual-face argument needs.
- *Vertices as fixpoints* (the commercial rule): identical to the
  straight-edge rule for axis-parallel edges and contaminated by seam
  vertices otherwise.
- *A posteriori (energy-based) refinement* for this defect class: a
  feature below the grid has zero effect on the coarse solution, hence
  zero energy signature — no a-posteriori indicator can find it.  Only
  a rule that reads the CAD model can.  (Global ΔS convergence loops
  are a how-to recipe, not a mesher feature.)
- *Averaging ε over the dual volume along the edge* (subpixel
  smoothing) instead of a mesher rule: makes chamfers continuous without
  planes but rewrites DD-051 and every port/slab certificate.

**Files:** `src/magnelio/geo/_occ_backend.py`
(`_edge_feature_planes`, `extract_feature_planes_per_shape`),
`src/magnelio/mesh/mesher.py` (`MeshControl.max_edge_refinement`,
`_merge_feature_planes`, `_warn_dropped_edge_planes`, `end_floor` in
`_generate_axis_lines` / `_enforce_boundary_buffer`),
`validation/edge_plane_chamfer_certificate.py`,
`tests/unit/test_geometry.py` (`TestEdgeFeaturePlanes`),
`tests/unit/test_mesh.py` (`TestMergeFeaturePlanes`,
`TestEdgePlanesInTheMesh`, `TestEdgePlaneBoundaryBuffer`),
`docs/methods/meshing-conformal.md`, `spec.md` §6.1, `CHANGELOG.md`.

## DD-192 — The bulk cell size follows the wavelength of the slab, not of the model (`wavelength_rule="local"`)

**Status:** Decided 2026-08-25 with the developer (second item of the
mesh-generation plan after DD-191: local wavelength rule before
metal-edge refinement); implemented 2026-08-25.  Developer choices in
the planning discussion: default on (not opt-in), slab occupancy by
analytic bounding box (not by exact section), the wavelength from the
static `max(ε)` also for dispersive materials (as before).

**Problem.**  `Mesh.from_geometry` derived *one* bulk cell size
`h_max = c₀ / (f_max · n_max) / min_nodes_per_wavelength` from the
densest material anywhere in the model and applied it to every axis
interval.  A 10 × 10 × 2 mm ε_r = 4.3 block in an 80 mm air box
meshed the whole box at the ceramic wavelength — 2.07× finer per
axis than the air needs, 1.43 M cells where 254 k carry the same
resolution inside the block (measured, 20 nodes/λ at 10 GHz).  The
same penalty sits above every thin substrate and around every
electrically small dielectric.  A second, silent defect: the
background material never entered `n_max` — a model with a dense
background and only PEC solids meshed at the vacuum wavelength.

**Decision.**  The bulk size is per axis *interval*.  On a tensor
grid a grid line spans the whole domain, so the finest sensible
resolution is per slab: an interval `[p0, p1]` on one axis is the
slab of the domain between those two planes, and the densest
material whose analytic bounding box reaches into the slab by more
than the clustering tolerance sets its wavelength.  The background
counts in every slab.  A shape without an analytic box counts
everywhere (conservative).  `_local_bulk_sizes` builds
`{axis: [h_max per interval]}` once the grid planes are final;
`_generate_axis_lines` and `_enforce_boundary_buffer` take a scalar
or a per-interval list (the profiles already worked per interval).
The PML depth follows the boundary slab's bulk size.  Everything
else keeps the *global* `h_wavelength` as its reference: the DD-191
edge floor (bounds the time step, which follows the smallest cell
anywhere), the feature sentinel of `h_fine` and the DD-105
undershoot check.  `MeshControl(wavelength_rule="global")` restores
the old rule; `"local"` is the default — the rule is the general
case, the old one a special case of it.

**Consequences.**  Axes a dielectric spans entirely gain nothing
(the in-plane axes of a full-width substrate); the axis normal to it
gains in every slab outside the dielectric.  Interfaces keep their
`h_fine` ramps on both sides, so the jump between a dielectric slab
and an air slab is bounded by the growth factor.  The bounding box
is exact for bricks and conservative for spheres, cylinders, lofts —
the slab is never meshed coarser than the material in it, at the
price of some over-refinement in the box corners of a curved body.
Meshes of every model with mixed dielectrics change; the tutorial
numbers are updated from a re-run (internal record in the STATUS
entry).

**Rejected.**
- *Opt-in*: the rule is what a user of a per-material hex mesher
  expects; leaving it off would keep the penalty as the default.
- *Exact section occupancy* (OCC slab ∩ shape): tighter for curved
  bodies, but it brings the DD-167/DD-168 section machinery into the
  bulk-size decision and costs mesh time on every build; the
  bounding box is deterministic and cheap, and the conservative
  side is the safe side.
- *ε(f_max) for dispersive materials*: a Pole–Residue evaluation in
  the mesher plus a rule for Lorentz overshoot below `f_max`; the
  static `max(ε)` is conservative for relaxation models and was the
  rule before.
- *Per-cell (octree-like) local wavelength*: not expressible on a
  tensor grid.

**Files:** `src/magnelio/mesh/mesher.py` (`MeshControl.wavelength_rule`,
`_refractive_index`, `_local_bulk_sizes`, `_per_interval`, per-interval
`h_max` in `_generate_axis_lines` / `_enforce_boundary_buffer`, PML
depth), `tests/unit/test_mesh_local_wavelength.py`,
`docs/methods/meshing-conformal.md`, `CHANGELOG.md`.

## DD-193 — Short-interval grading keeps the fine-end cell at `h_fine` and relaxes the growth ratio

**Status:** Implemented 2026-08-25, on the DD-192 branch.

**Problem.**  An interval too short for the full ramp from `h_fine`
to `h_max` was refit with a fixed ratio `g` and an integer cell count
(`_n_one_sided` / the symmetric scan): the smallest count whose
fine-end cell lands at or below `h_fine · (1 + 5 %)`.  The fine-end
cell then falls out of the count — anywhere between `h_fine / g` and
`h_fine`, i.e. up to 23 % below what the interval asked for at the
default `g = 1.3`.  That is the DD-105 undershoot: it sets the time
step model-wide and buys no resolution.  DD-192 made the case common:
with the air slab above a thin trace meshed for the *air* wavelength,
the ramp no longer reaches `h_max` inside the interval, and the
mesh-convergence how-to warned on two of its rungs (19 % and 18 %
below the requested 66.7 µm / 50 µm).

**Decision.**  Keep the count, keep `h0 = h_fine`, and solve the
ratio `g' ∈ [1, g]` with which those cells fill the interval exactly
(`_ratio_for_exact_fill`, bisection on the monotone series sum;
one-sided and symmetric forms).  Applied only where the fixed-ratio
refit *would* undershoot (`h0 < h_fine`); a refit landing inside the
5 % overshoot band is kept as it was.  Every neighbour ratio stays
`≤ g` by construction.  The DD-107 buffered profile (`_tailed_widths`)
already kept `h0` at the tolerance and is untouched.  Measured on the
how-to microstrip, rungs 8…32: first cell above the strip = `h_fine`
exactly, maximum ratio 1.27–1.30, zero warnings.

**Consequences.**  Meshes change wherever a short interval used to
undershoot — homogeneous models included, so this is the change that
breaks bit-identity with 0.4.5 beyond DD-192's mixed-dielectric
scope (bit-identity is preferred when free, not a constraint).  The
cells in such intervals are slightly larger and the time step
slightly longer; nothing gets finer.

**Measured cost (2026-08-26, KB-028).**  The DD-053 conformal coax
certificate (19 × 19 × 25) is such a short interval: the three cells
spanning the 0.41 mm inner conductor were 0.121 / 0.168 / 0.121 mm
(ratio 1.39, fine end 12 % under `h_fine` = 0.137 mm) and are now
3 × 0.137 mm; nothing else in the grid moves.  The dual faces at the
conductor surface reach deeper into the metal, and the conformal line
impedance moves 48.12 → 48.94 Ω (analytic 49.97), the port floor
−131 → −135.6 dB (median −153.9).  The integration test is re-pinned
to 48.94 Ω; the DD-053 table keeps its historical grid.

**Files:** `src/magnelio/mesh/mesher.py` (`_ratio_for_exact_fill`,
short branches of `_grade_then_uniform` and
`_grade_symmetric_to_uniform`), `tests/unit/test_mesh_exact_fill.py`.

## DD-194 — Singularity refinement at conductor edges (`MeshControl.singularity_refinement`)

**Status:** Decided 2026-08-25 with the developer (third item of the
mesh-generation plan after DD-191 and DD-192); implemented
2026-08-25.  Developer choices in the planning discussion: metal
edges only (dielectric edges deferred), parameter name
`singularity_refinement` (the DD-191 ratio already owns the words
"edge refinement").  Default **1 (off)** — see the measurement.

**Problem.**  Where a conductor forms a wedge of interior angle
`α < 180°` the field and the surface current behave like
`r^(π/(2π−α) − 1)` — `r^(−1/3)` at the 90° edge of a strip, a patch or
an iris, `r^(−1/2)` at a knife edge.  A grid cannot represent the
singularity; the error of everything that integrates the edge field
— line impedance, effective permittivity, hence S-parameter phase and
resonant frequencies — converges roughly first order in the cell that
holds the edge, however fine the bulk is.  Measured on the how-to's
microstrip port mode (`validation/singularity_refinement_certificate.py`):
Z0 = 51.50 / 52.23 / 52.61 Ω at edge cells 50 / 25 / 12.5 µm, limit
≈ 52.95 Ω; three grids with the *same* 12.5 µm edge cell and cell
counts 1452 / 984 / 840 agree within 0.08 Ω.  The mesher graded every
plane from the axis-wide `h_fine`, blind to which planes hold an
edge.  Hex meshers of the commercial suites carry an edge-refinement
factor for exactly this.

**Decision.**

- *Which edges.*  `extract_singular_edge_planes` (geometry backend)
  walks the sharp edges of every shape (the DD-191 filter: no seams,
  no degenerated edges, no split lines inside one analytic surface)
  and classifies them with the kernel's own offset analysis
  (`BRepOffset_Analyse`, 5° tangency tolerance): a *convex* edge of a
  shape whose material `is_pec` (PEC and lossy metal) is singular; a
  *concave* edge of a non-metal shape is singular when the material in
  the open wedge — probed a short way along the bisector of the two
  outward normals, shape lookup with background fallback — is metal
  (a vacuum body in a PEC background: iris rims and ridges yes,
  cavity corners no).  Tangential edges (fillet onsets) and dielectric
  edges contribute nothing.  Each singular edge yields the axis-normal
  planes it lies flat in (`_edge_flat_planes`, the DD-191 rule).
- *Which planes.*  After all merges the final planes within the
  clustering tolerance of a singular position are flagged
  (`axis_is_singular`); the domain's own end planes never — a metal
  edge on a port face is the truncation, and the DD-107 buffer owns
  that interval.  Thin sheets: the far face is dropped as in DD-191,
  the sheet plane itself stays singular (knife edge).  Symmetry clip
  as for the other plane classes.
- *Profile.*  The fine size becomes per plane (`h_fine_planes`,
  `h_fine / k` on singular planes); `_generate_axis_lines` and the
  DD-107 regeneration grade each interval from its two ends' own
  sizes.  Boundary intervals use the interior plane's size in the
  existing one-sided profile; interior intervals with equal ends take
  the symmetric profile unchanged; unequal ends take
  `_grade_asymmetric_to_uniform`: both full ramps plus a uniform
  middle when they fit (a remainder below half a bulk cell is
  absorbed into the innermost ramp cells), otherwise the *tent* of
  `_two_ramp_fill` — the smaller fine size pinned exactly, one common
  ratio `r ≤ g` up to a peak and back down, the coarse-end cell free
  between the pinned size and 5 % above its own.  The coarse end may
  undershoot its `h_fine` because the time step is bound by the
  pinned cell already; a first attempt that pinned *both* ends and
  balanced the seam for ratio `g` produced seam jumps up to 1.7 after
  the DD-193 relaxation.  The hard floor `min_cell_size` caps the
  refinement (tutorial 06's `min_cell_size = 1.59 mm` leaves its
  grids at the floor).  `check_grading_undershoot` takes the per-plane
  sizes so the deliberate edge cells are not reported.
- *Factor 1 is bit-identical* to the DD-193 state (four models
  checked against a worktree of `d18bf7f`).

**Measurement (why the default is off).**  Two ladders on the
mesh-convergence how-to's structures, 2026-08-25.

*Port-mode Z0* (2D, no time step): the error is a function of the
edge cell alone (above); at equal edge cell the refined grids need
32 % / 42 % fewer cells (factor 2 / 4).

*Patch-microstrip S-parameters* (the how-to's ladder, reference
factor 3 at mnpw 48; `max|ΔS|` to it): factor 1 — 0.093 / 0.047 /
0.031 / 0.015 at mnpw 16 / 24 / 32 / 48; factor 2 — 0.047 / 0.021 /
0.0125 / 0.004; factor 3 — 0.032 / 0.015 / 0.007 / 0.  At a fixed
`MeshControl` the factor cuts the error 2.5–4×.  But the edge cell
sets the time step: with cost = cells / smallest cell, factor 1 at
mnpw 32 (41 850 cells, 25 µm, error 0.031), factor 3 at mnpw 16
(36 540 cells, 16.7 µm, 0.032) and factor 2 at mnpw 24 (48 300 cells,
16.7 µm, 0.021) lie on **one cost-versus-error curve** — the factor
redistributes resolution from the bulk to the edges, it does not
buy accuracy for free.  The how-to's stop rule (ΔS < 0.02 on two
rungs) fires at mnpw 32 for factor 1 with an actual error of 0.031
against the refined reference, and at mnpw 16 for factor 3 with 0.032:
the rule's blind spot is the same for both.  Tutorial re-run at
factor 2: 09 +68 % cells and half the time step, 10 +19 %, 12 +4–17 %,
13 +30–42 % (iris rims), 17 +27 %; 06/07 unchanged in count (floored);
the homogeneous models identical.  Tutorial 13's fine coupled pair at
factor 2 under-delivered eigenmodes at its pinned shift — DD-195.

So the factor is a *tool*, not a default: it pays where the impedance
or the effective permittivity is the quantity of interest (port
normalisation, line dispersion, resonators bounded by metal edges),
where the time step is bound by a `min_cell_size` floor or by another
axis anyway (then the edge refinement is free), and where memory
rather than time is the limit.  A default of 2 would make every
microstrip model about three times slower for an accuracy the user
can buy equally with `min_nodes_per_wavelength`.

**Consequences.**  Opt-in; no mesh changes at the default.  With a
factor set, the DD-105 check knows the edge cells; the DD-107 buffer
at a domain face may trim the innermost singular cell within the
legacy refit class.  Dielectric edges (exponent −0.1…−0.2 for ε_r
4…10) stay unrefined — a later opt-in if a case shows the need.  The
edge probe for concave edges costs one `point_in_shape` per concave
edge per shape; models without any metal skip the pass entirely.

**Rejected.**
- *Refining only the outside of the metal* (the field lives there):
  on a tensor grid the plane's cells on the metal side are the
  resolution above and below the strip next to its edge, where the
  field is singular too.
- *Static singularity correction of the material coefficients at
  the edge* (the published FDTD edge-correction schemes): rewrites
  the DD-051 conformal matrices and every port certificate; the
  mesher rule is what the hex meshers of the commercial suites do.
- *A default of 2* — see the measurement.
- *Both fine cells pinned in the short profile* — seam jumps after
  the ratio relaxation; the tent pins only the cell that bounds the
  time step.

**Files:** `src/magnelio/geo/_occ_backend.py` (`_sharp_edges`,
`_edge_flat_planes`, `extract_singular_edge_planes`),
`src/magnelio/mesh/mesher.py` (`MeshControl.singularity_refinement`,
`axis_is_singular`, `h_fine_planes`, `_fine_per_plane`,
`_grade_asymmetric_to_uniform`, `_two_ramp_fill`, `_full_ramp`),
`src/magnelio/mesh/_quality.py` (`h_fine_planes`),
`validation/singularity_refinement_certificate.py`,
`tests/unit/test_mesh_singularity.py`,
`docs/methods/meshing-conformal.md`, `spec.md` §2.6 / §6, `CHANGELOG.md`.

## DD-195 — The ARPACK request grows at one shift when null-space artefacts crowd it

**Status:** Implemented 2026-08-26, on the DD-194 branch.

**Problem.**  The 3D eigenmode solver asks ARPACK for
`n_modes + 4` vectors around the shift and keeps those above the 1 MHz
null-space floor.  The KB-011 ladder raises the shift and grows the
request on under-delivery — but only on the *auto* path; a
user-given `sigma` is a one-rung ladder by design (the shift is the
user's decision), so a request in which most vectors converge on the
curl-curl null space simply returned fewer modes, with the warning
that tells the user to pin a shift they had already pinned.  DD-194
exposed it: tutorial 13's coupled resonator pair at
`singularity_refinement=2` (104 × 60 × 12 cells, edge cells 83 µm at
the iris rims) returned 5 null-space artefacts among the 7 vectors
and 2 of 3 modes, where the unrefined grid (96 × 52 × 12) returned 3
of 3 with none — the tree-cotree gauge leaves a residual null space
that grows with the number of tiny conformal edges.

**Decision.**  Before moving the shift, grow the request at the
*same* shift: after an attempt that discarded `n_null` artefacts and
still lacks modes, ask for `k + n_null + 2` vectors, at most
`_NULL_GROW_RETRIES = 2` times per rung, keeping the union of physical
eigenpairs as the ladder does.  The SuperLU factorisation of
`A − σB` is built once per rung (`_arpack_op_inv`) and handed to
`eigsh` as `OPinv`, so the grows cost Lanczos iterations, not
factorisations (the physical eigenvalues agree with the
self-factorising call to 1e-8; the null residues differ, they are
noise at 1e-14 of the physical scale).  The under-delivery warning
stays for the case that even the grown requests fall short.

**Consequences.**  Solves that delivered in one request are
unchanged (same factorisation, same first request).  Under-delivering
solves at a pinned shift now cost one or two more Lanczos runs and
return the modes.  Tutorial 13 at factor 2 passes its own guard.

**Files:** `src/magnelio/solver/_eigenmode_3d.py`
(`_NULL_GROW_RETRIES`, `_arpack_op_inv`, grow loop in `solve`),
`tests/unit/test_eigen_null_grow.py`, `CHANGELOG.md`.

## DD-196 — Multi-conductor QTEM ports return the modal basis of the capacitance pencil

**Status:** Implemented 2026-08-26, on the `feat/qtem-modal-basis` branch.

**Problem.**  `solve_qtem_laplace` returned, for K > 2 conductor
groups, the per-conductor Laplace solutions (V_k = +1 V, the others
grounded) with ε_eff,k = C_kk/C_0,kk — the *conductor* basis of the
line, which DD-066 and DD-068 had left in place on the argument that
"per-mode ε_eff forbids the Gram mixing".  On coupled lines that
basis is not a set of modes: the patterns are not orthogonal in the
port mass, and on an inhomogeneous cross-section each of them is a
superposition of modes travelling at different speeds, so the
reported ε_eff and Z_line describe nothing that propagates.  Two
consequences.  A `PortWaveguide(n_modes=2)` on ground + two strips
(K = 3, `n_modes == K − 1`) did not raise `qtem_multimode`, so the
DD-068 dual-basis projections were *not* built and the port ran
primal projections over a non-orthogonal basis — the DD-066
instability class (measured there as a DTBC feedback blow-up to
1e64).  And the channel order of two identical strips hung on the
label tie-break of `auto_conductors` (descending node count, equal
by construction).  The coupled-line coupler how-to needs the
even-/odd-mode impedances from `solve_ports()`; that number did not
exist.

**Decision.**  Multiconductor transmission-line theory (Paul,
*Analysis of Multiconductor Transmission Lines*, 2nd ed., ch. 3 —
`paul2008`, VERIFY): the per-conductor fields yield the two Maxwell
capacitance matrices directly as energy forms,
`C_jk = ê_jᵀ M_ε,cap ê_k / normal_dx` (actual dielectric) and `C_0`
alike from the vacuum solve — the off-diagonals are negative by
construction, no charge bookkeeping needed.  `_qtem_modal_channels`
solves the generalised eigenproblem `C v = ε_eff C_0 v`
(`scipy.linalg.eigh(C, C_0)`), the quasi-static form of the modal
decomposition of `L C` with `L = μ_0 ε_0 C_0⁻¹`: eigenvectors are the
conductor-voltage patterns (the exact even/odd pair of a symmetric
pair), eigenvalues the modal ε_eff, fields the superpositions
`Σ v_k ê_k` of the per-conductor solutions, M_ε-normalised.  Modal
capacitances `C'_v = vᵀ C v`, `C'_0,v = vᵀ C_0 v` over the
unit-Euclidean pattern give `Z_0 = 1/(c √(C'_v C'_0,v))` — the same
gauge DD-066 chose for the TEM Gram eigenbasis, and for a symmetric
pair literally Z_0e / Z_0o.  Conventions follow DD-066: descending
ε_eff (a microstrip pair reports the even mode first), sign gauge
"leading near-maximal entry positive" with the 1e-9 tie tolerance.
The eigenvectors are C- and C_0-orthogonal, hence the modal fields
are exactly orthogonal in the capacitance-corrected mass and only at
tangential PMC window edges not exactly in M_ε — `build_modal_port`
therefore builds the DD-068 dual-basis projectors for *every*
multi-channel QTEM port (`epsilon_r is None and len(discrete) > 1`),
not only when hybrids are present.  K = 2 takes the historical block
bit-identically (no `eigh`, no renormalisation); the generalised
path reproduces it to 1e-12 (tested).  Labels stay `QTEM_lap0k`.

**Consequences.**  Single-conductor ports (every existing tutorial
and test) are unchanged.  K > 2 QTEM ports change their channels:
`report.modes[0]` is now the largest-ε_eff mode, not the first
signal conductor, and `z_line_num` in the operator report follows.
The CW (DD-056) and band (DD-057) bootstraps track `laplace_modes[0]`
— now a modal profile, which the pencil eigenpairs match more
closely than a per-conductor profile did; they still track one
family only (a Z_0e/Z_0o pair through a band port is one family plus
"the other families", as before).  The ζ-pencil hybrid extension
(DD-068) drops one pencil eigenpair per line mode by W_t overlap —
unchanged in count, better matched in profile.  Measured on the
edge-coupled FR4 pair of `test_unified_multimode_port.py` (w = 1.5 mm,
h = 0.8 mm, s = 0.5 mm): 1 < ε_odd < ε_even < 4.3, Z_e − Z_o > 15 % of
Z_e, unit self-response of both channels through `project_V` at
1e-9, coupling `(Z_e − Z_o)/(Z_e + Z_o)` rising as the gap closes.

**Files:** `src/magnelio/ports/_modal/tem_laplace.py`
(`_qtem_modal_channels`, `solve_qtem_laplace` branch, docstrings),
`src/magnelio/ports/_modal/factory.py` (dual-basis condition,
docstrings), `src/magnelio/ports/_modal/numerical_2d.py` (docstring),
`tests/unit/test_modal_qtem_laplace.py`
(`TestCoupledMicrostripModalBasis`),
`tests/integration/test_unified_multimode_port.py`
(`TestCoupledMicrostripModalPort`), `docs/methods/ports.md`,
`docs/references.bib` (`paul2008`), `CHANGELOG.md`.

## DD-197 — Curved sheets: `Surface.parametric` and sheet-preserving transforms

**Status:** Implemented 2026-08-26, on the `feat/parametric-surface` branch.

**Problem.**  Every sheet in the geometry namespace was planar — a
`Face` polygon or a covered curve — so the only curved metal a model
could hold came from primitives, lofts, sweeps and revolutions.  A
reflector antenna needs a *surface given by a formula*: an offset
paraboloid is a patch of a surface of revolution (buildable, awkwardly,
as `revolved()` plus a Boolean rim), a hyperboloid sub-reflector the
same, and a numerically shaped reflector is no surface of revolution at
all.  Two further gates stood in the way: `thickened()` and the
`_ExtrudedFaceShape` profile path were typed on `PlanarSheet` and
`face_plane_normal` refused non-planar faces; and the transform
wrappers (`_TranslatedShape` & co.) inherit from `Shape`, so even a
*planar* sheet lost its sheet-ness when moved — `Face(...).rotated(...)
.thickened(...)` raised `TypeError`.

**Decision.**  A marker hierarchy `Sheet(Shape)` → `PlanarSheet(Sheet)`
in `geo/_sheet.py`; `extruded()`, `thickened()`, `shelled()`'s refusal
and the mesher's standalone-sheet rejection gate on `Sheet`,
`revolved()`/`swept()`/`Loft` sections stay `PlanarSheet` (they need a
plane).  `geo.Surface` (`geo/surfaces.py`) is a `Sheet` holding a
sample grid; `Surface.parametric(fn, u=, v=, samples=(32, 32))`
evaluates the map on a `meshgrid` (array call first, scalar fallback),
and `make_bspline_surface` interpolates the grid with
`GeomAPI_PointsToBSplineSurface.Interpolate` (degree 3, exact at the
samples; a collapsed pole row is accepted).  The class stores points,
not the callable — a shape is a value, DD-178's rule on parametric
history holds, the store round-trips it as an `ImportedSolid` like
everything else.  `_analytic_bbox` = sample hull padded by a quarter
diagonal (the Loft's spline allowance).  Thickening branches on
`is_planar_face`: planar → the historical prism (bit-identical); curved
→ `make_thick_face`, a cascade `MakeThickSolidBySimple` →
`BRepOffset_MakeSimpleOffset`, each result healed by `ShapeFix_Shape`
if invalid and accepted only when valid *and* within 10 % of
area × thickness — the kernel's offset folds silently on dense grids
(48 × 96: twelve times the volume) and this is the only way to catch
it; `"symmetric"` is planar-only; the failure text points at
`extruded()`, which is `BRepPrimAPI_MakePrism` on any face and always
robust.  Transforms pick a sheet-preserving wrapper class
(`_wrapper(cls, inner)` builds `cls + marker` subclasses on demand), so
the marker survives `translated/rotated/scaled/mirrored`.

*Measured* (`investigations/parametric-surface/`, internal record;
offset paraboloid F = 180 mm, D = 240 mm, x_c = 150 mm in polar
parametrisation): interpolation 1–2 ms; off-sample deviation 3e-7 m at
16 × 32, 1.5e-6 m at 32 × 32 (the default), 1.5e-8 m at 32 × 64;
surface area equal to the closed-form integral to 1e-6; prism volume =
projected disc × length to 1e-4; `MakeThickSolidBySimple` valid at
32 × 64, invalid-but-healable at 32 × 32, folded at 48 × 96.

**Consequences.**  Additive API (`Surface` in `geo.__all__`); no
existing model changes.  `thickened()` on a curved sheet is best-effort
and loud; the reflector tutorial uses `extruded()`.  The mesher sees a
B-spline face only through its bounding box (no feature planes,
DD-106/DD-191 do not apply) and sections it through the kernel
(DD-102 delegation) — documented in the new methods chapter *Geometry
construction* (`docs/methods/geometry.md`), which also states the
"two cells thick" rule for curved shells (the thin-metallisation
detector, `_conformal.py`, needs a flat bounding box).  Not in scope:
Booleans on sheets (a polar parametrisation makes the rim exact
without trimming), thin-sheet physics, curved `revolved()`/`swept()`.

**Files:** `src/magnelio/geo/_sheet.py`, `src/magnelio/geo/surfaces.py`
(new), `src/magnelio/geo/_occ_backend.py` (`make_bspline_surface`,
`is_planar_face`, `_face_forward_sign`, `make_thick_face`),
`src/magnelio/geo/modifications.py`, `src/magnelio/geo/transforms.py`
(`_wrapper`), `src/magnelio/geo/shape.py` (docstrings),
`src/magnelio/geo/__init__.py`, `src/magnelio/mesh/mesher.py`,
`tests/unit/test_geo_surface.py` (new), `tests/unit/test_scaling.py`
(zoo), `tests/unit/test_geometry.py`, `docs/methods/geometry.md` (new),
`docs/methods/index.md`, `CHANGELOG.md`.

## DD-198 — Waveguide-port windows in absorbing faces

**Status:** Implemented 2026-08-26, on the `feat/port-window-cpml` branch.
Amended 2026-08-27 (KB-034): step 0 ran before the thin-wire and
thin-sheet passes, so a thin metallisation touching the absorbing face
— a microstrip feed with its window in a CPML wall — had no mask in
the extension and the window read as a hollow cross-section.  Step 4c
of the mesher now repeats the mask-only extension after the sheet pass
(`tests/unit/test_pml_extension.py`); found while feeding the patch
array of the antenna-array how-to through a shielded launch.

**Problem.**  A radiating structure fed through a guide — a horn's
neck, a coax entering an open box — puts its port on a wall that must
otherwise absorb.  Nothing refused a `PortWaveguide` on a CPML face,
and the result was silently wrong in three independent ways.  (1) The
mesher grows a CPML face outward by `n_pml` cells and continues the
cell materials (step 3b), so the port sits at the outer end of the
absorber; the CPML `update_E/update_H` kept running there (only
`apply_E/H` are skipped on port faces, DD-021), the guided wave was
absorbed between the domain and the port, and the DTBC/Mur assumption
of a lossless uniform continuation behind the port plane was violated
— the slab certificate (`_port_chain_slab_defect`) checks masses only
and was blind to it.  (2) On the conformal path the sub-cell
classifier works against the B-rep solids, which end at the nominal
bbox: inside the extension the guide's walls read as free space and
the Cat-2 un-mask dropped their PEC mask (KB-029; measured 156 of 284
Ey PEC edges left in slab 0 of a 20 × 10 mm tube on a 2 mm grid) — a
defect for *every* conductor touching an absorbing face, ports or
not.  (3) `MonitorFarField` cut the guide with its Huygens box and
integrated the guided wave as if it were an external source.  A
fourth defect surfaced while measuring: monitors fed by a TE/TM port
were normalised to the excitation waveform, not to the launched
incident power (KB-030; `P_rad/P_acc` = 0.77 with an exact image-plane
flange).

**Decision.**  Four bounded pieces, no change to the port plane's
position or to the port-face flatten.

*Step 0 — mesher.*  `mesh/_pml_extend.py::extend_subcell_data_into_pml`
mirrors step 3b for the sub-cell data (step 3d, before the wall masks):
the extension slabs take the first fully interior slab — node-sampled
quantities the plane one cell inside the interface (the interface
plane's own dual cell straddles the extension), cell-sampled ones the
first interior cell — for the PEC mask, `EdgeMaterialData` and
`FaceMaterialData`; enlarged-cell donor links are re-pointed within the
copied slab and dropped where they crossed slabs.  Meshes without a
conductor at an absorbing face are bit-identical (their slabs were
already equal).

*Step 1 — absorber.*  `CPMLBoundary.set_port_windows(windows)` builds
per-component stretching coefficients `_c_E1/_c_E2/_c_H1/_c_H2` and
`_ck_*` that equal the face's 1D profile outside the window footprints
and are zero inside them over the whole layer depth — literally
σ = 0, κ = 1 in those columns (`c = 0` keeps ψ ≡ 0, `ck = 0` removes
the κ term); the twelve update sites read the per-component arrays,
the default is the old broadcast (bit-identical without windows, no
extra multiply, CUDA-graph neutral).  Footprint rule per tangential
dimension: cell-sampled → `lo:hi`, node-sampled → `lo:hi+1` (the ring
included — it is conductor).  `FITTimeDomainSolver.setup` collects the
windows of the modal operators per face and wires them.

*Step 2 — validation.*  `validate_absorbing_face_window` (factory),
enforced where port and absorber meet — `FITTimeDomainSolver.setup`
for every modal operator on a CPML face, and `resolve_declarative_port`
for an early message on the `PortWaveguide` the user wrote: on a CPML
face a modal port needs a window, and every edge of the window ring
must be a PEC edge of the mesh on the port slab — the enclosure rule.
A port-only mode solve at the spec level is not gated: an open
cross-section with CPML side faces is a legitimate 2D problem (the
symmetry-declaration tests use it as a mesher device).  Physics: a port in an
absorbing wall is meaningful only as a shielded guide end, and the
lateral σ edge of the switch-off then falls on metal.  The 2D mode
solve needs no change: the Dirichlet ring comes from the material
mask (correct after step 0); `resolve_port_edge_pec` still says
"Neumann" for interior ring edges of a non-PEC face, which is inert.

*Step 3 — far field.*  The analysis hands the port windows on CPML
faces to `MonitorFarField` (`_wire_far_field_ports`, runtime wiring).
The face a feed crosses is sampled at the absorber interface (margin
0 — every cell of guide outside the box carries wall currents the
surface cannot see); patches inside a footprint and patches whose two
sampled cells are both conductor get zero area (`_BoxFace.keep`,
carried in the dump).  The remaining approximation — the outer-wall
currents inside the absorber — is the usual one for waveguide-fed
radiators.

*Step 4 — incident normalisation (KB-030).*  `compute_s_parameters(
return_incident=True)` hands back the separated incident wave `a(f)`
of the excited channel; `incident_amplitude_ratio` forms
`|a(f)|/|W(f)|` and the analysis (in-RAM run) and the project reader
(streamed/resumed run, alongside the S-matrix it derives) wire it into
far-field and frequency monitors as a second divisor.  Gated on the
excited channel's `mode_type ∈ {TE, TM}`: for lumped, TEM and QTEM
channels the ratio is unity by construction and nothing is wired, so
their monitors stay bit-identical (the store round-trip tests at
1e-10 pin that).  The recorder reads waves at full-model scale
(DD-155), so no symmetry factor enters.

**Measured** (`tests/integration/test_port_window_cpml.py`, PEC tube
20 × 10 mm bore, 3 mm walls, 2 mm grid, 8.6–11.8 GHz).  Through-tube
with the CPML-face port and a PEC-face port: |S21| > −0.5 dB, |S11| <
−40 dB in band, TE10 cut-off within 2 % of c/2a — without step 1 the
absorber eats S21, without step 0 the cut-off and S11 are off.
Open-ended tube in a CPML box: |S11| −13.5 dB, `P_rad/P_acc` 0.97 at
10 GHz (0.79 before step 4, 0.79 → 0.79 with the interface sampling
alone — the deficit was the normalisation); flanged variant (PEC face)
0.91, the image plane having a hole where the port window is; S11 of
the two variants within 3 dB.

**Consequences.**  Existing CPML runs without a conductor at the face
are unchanged; runs *with* one (a ground plane into the absorber)
change their absorber-slab masks — for the better.  TE/TM-fed monitor
results change by the impedance ratio; everything else is bit-identical.
`CPMLBoundary.apply` (the documented PEC backing) is still never
called by the solver — the outer plane of an absorbing face runs in the
free stencil; unchanged here, noted for the record.  `_resolve_bc`
builds the CPML with the declared thickness (8) while the mesher may
have appended more cells (`mesh.pml_cells`); the footprint follows the
CPML's own `_n_pml`, the extra cells are lossless continuation —
harmless.  Not in scope: curved guide necks (the window snaps to grid
nodes and the enclosure rule refuses when a wall falls between nodes —
the message says so), thin sheets and thin wires (not continued into
the absorber), the band pipeline's separate spectrum path.

**Files:** `src/magnelio/mesh/_pml_extend.py` (new),
`src/magnelio/mesh/mesher.py` (step 3d), `src/magnelio/boundaries/cpml.py`
(`set_port_windows`, per-component coefficients),
`src/magnelio/solver/fit_td.py` (wiring), `src/magnelio/ports/_modal/factory.py`
(`validate_absorbing_face_window`), `src/magnelio/ports/declarative.py`,
`src/magnelio/monitors/far_field.py` (`keep`, footprints, incident
ratio), `src/magnelio/monitors/field_frequency.py`,
`src/magnelio/post/modal_sparameters.py` (`return_incident`),
`src/magnelio/analysis/scattering_td.py` (`incident_amplitude_ratio`,
`_wire_far_field_ports`, `_wire_incident_amplitude`),
`src/magnelio/io/project.py` (dump keys, reader ratio),
`tests/unit/test_pml_extension.py`, `tests/unit/test_boundaries.py`,
`tests/unit/test_monitor_far_field.py`,
`tests/integration/test_port_window_cpml.py`, `docs/methods/ports.md`,
`docs/methods/boundaries.md`, `docs/methods/far-field.md`,
`docs/methods/sources-monitors.md`, `known-bugs.md` (KB-029, KB-030),
`CHANGELOG.md`.

## DD-199 — Free-form faces are sectioned on a lifted triangulation

**Status:** Implemented 2026-08-26, on the `feat/facet-section-engine`
branch.  Extends DD-102; closes KB-031 (found on the way).

**Problem.**  The tutorial-19 Cassegrain (two B-spline reflector
shells, a ruled-loft horn, 2.08 M cells) meshed in 372 s, 340 s of
which were 1 597 `BRepAlgoAPI_Section` Booleans at 0.24–0.52 s each:
every body of the model — the reflectors, the horn (a ruled `Loft`
yields B-spline faces even where they are geometrically planar) and
the air complement carrying their faces — delegated every plane to the
kernel, because the DD-102 engine answers planar faces only.  The
section pool never fired either: its admission weighs Σ face_count
(3–30 faces here) against a 150 k gate calibrated at 40–80 µs per
face, three orders of magnitude below the 0.2–0.5 s a free-form
section costs; the DD-141 measured-sample admission sits behind that
gate.  (`investigations/mesh-freeform-speed/` — internal dossier.)

**Decision.**  A shape with at least one free-form face (surface type
not plane/cylinder/cone/sphere/torus) is represented in the DD-102
engine by its *triangulation*: `BRepMesh_IncrementalMesh` at the
section deflection, once per engine, nodes welded (exact first, then
within the shape tolerance — seams and glued edges), degenerate
triangles dropped, every edge bordering exactly two triangles, and the
signed facet volume reproducing the kernel volume within 2 % (an
inconsistently oriented or open triangulation fails by O(1)).  The
triangles are planar facets with straight edges, so the engine's
index-based stitching applies unchanged.  Three things distinguish
the facet path from a plain triangle-soup section:

1. *Combinatorial orientation.*  A segment runs from the crossing on
   the corner-order edge that leaves the positive side ("+ → −") to
   the one that enters it; the shared edge of two neighbours is
   traversed in opposite corner orders, so every crossing is a start
   exactly once — also for the zero-length segments of a vertex on the
   plane (sign convention: on the plane counts as positive), where a
   geometric ordering would be arbitrary and was (planes through the
   pole and the seam delegated or failed).  No on-plane tolerance band
   is needed; a coplanar facet still delegates (DD-087 class).
2. *Lift onto the exact surface.*  The chord error δ of a
   triangulation is normal to the surface and reaches the section
   plane amplified by 1/sin(cut angle).  Each triangle keeps its
   corners' (u, v) on its B-Rep face; the crossing's parameters are
   interpolated along the edge, the surface point and normal are
   evaluated there (`Geom_Surface.D1`, ~2 µs), and the point is moved
   within the section plane along the in-plane part of the normal
   onto the tangent plane — one Newton step, second-order residual.
   Planar faces are exact as facets and skip the lift.  Measured on a
   paraboloid prism (δ = 1e-4): x-cuts −1e-2 → −1e-4 relative area.
   (The one-argument `BRep_Tool.Surface(face)` is the *located* copy;
   applying the face location a second time put the translated bottom
   face 200 δ off and silently disabled its lift.)
3. *In-plane refinement.*  Between two crossings the polygon follows
   a chord whose sagitta is set by the triangle size, not by δ — and
   a section has hundreds of them, adding up to −1.3e-3 on a convex
   cut.  The chord midpoint is lifted, its displacement *is* the
   sagitta, and the segment is split into ⌈√(sagitta/(δ/10))⌉ parts
   whose interior points are lifted likewise.  At a tenth of δ the
   facet path lands at ≤ 1.5e-4 of the kernel's untessellated slab
   reference on every cut of the paraboloid prism, where the kernel's
   own δ-tessellation is at up to 1.2e-3 (3–5× fewer vertices).

Analytic curved faces keep the kernel path: their sections are exact
conics tessellated at δ, and the pinned coax/Dey–Mittra results stay
bit-identical (unit suite unchanged).  `MAGNELIO_FACET_SECTIONS=0`
keeps free-form shapes on the kernel path (A/B runs).

**Measured.**  Cassegrain mesh 372 s → 58 s (0 kernel sections; the
remainder is classification, edge fractions and slab bookkeeping);
10 mm variant 184 s → 23 s.  Per plane: 0.5–2.9 ms facet (before
lift/refinement) against 14–402 ms kernel.  End-to-end, the tutorial
run on the facet mesh: peak directivity 19.24 dBi (19.52 on the
kernel mesh), beam 2.1° from the designed direction (same), |S11|
−16.7 dB at 10 GHz (−16), `P_rad/P_acc` 0.935 (0.93).  Gates:
`tests/unit/test_facet_section_engine.py` — |area| multisets within
δ × perimeter of the kernel path on a paraboloid prism, a Boolean
with free-form faces and planes through triangulation vertices;
≤ 5e-4 of the slab reference; a ruled loft exact to 1e-9; the
dish-shell mesh's z-dual-face fractions against the analytic annulus
(mean ≤ 5e-3).

**Found on the way — KB-031.**  The dish-shell mesh gate could not
be run against the kernel path: z-planes cut the shell in an annulus,
and the kernel's two contours came back with the *same* winding.
`compute_face_material_areas` sums signed areas per shape, so a dual
face inside the hole was covered twice and booked fully PEC — the
conformal correction at the inner wall of every hollow conductor was
silently lost (PEC tube in air, 1.5 mm grid: mean |f_A − exact| 0.12
over 1 188 z-dual faces, bore-wall faces 0.000 against 0.997).
DD-102 had recorded the kernel's independent group orientation and
judged it harmless ("a face rectangle essentially never spans two
independently-flipped groups") — a rectangle inside a hole spans
none and is covered by both.  Fix: `orient_nested_contours`
(`_polygon_clip.py`) winds every contour by nesting parity (bbox
containment plus a majority point-in-polygon vote) before the
kernels see it — outer positive, hole negative, island positive;
bbox-coincident pairs (the cancelling opposite windings of a
degenerate tangency band) are left alone.  Applied at the single
annotation site, so the kernel path, the planar engine and the facet
path are treated alike.  Tube afterwards: mean 4e-3, max < 5e-2
(`tests/unit/test_section_contour_orientation.py`).

**Consequences.**  Models with free-form bodies mesh in a fraction of
the time and their sub-cell fractions are more accurate than before;
models with hollow conductors (tubes, shells, apertures) change at
the inner walls — for the better; everything else is bit-identical
(unit suite green, integration subset unchanged apart from KB-028).
Not in scope: the analytic quadrics (an exact plane × quadric section
would be the DD-102 follow-up); `compute_edge_pec_fractions` on
B-spline faces (`IntCurvesFace`, second-order cost); the pool
admission for few-face free-form shapes (moot now that they never
delegate).

**Amendment 2026-08-27 (KB-031 fallout on dielectric rings).**  The
winding fix also corrects dielectric bodies with an *air* hole — the
priority rule had shielded only conductor-filled holes, so a ceramic
ring's bore had been booked as ceramic.  The KB-011 fixture
(`tests/integration/test_analysis_eigenmode.py::
TestSparseHighContrastCavity`, ε_r = 45 ring 4/2 mm, 28×28×4) moved
2.3279 → 2.6566 GHz on an unchanged grid (solid puck 2.2302 GHz: the
old value was a nearly filled bore), and the under-delivery test no
longer starved at its 100 MHz shift — the third DD-195 grow (k = 46)
happens to deliver all six modes there; from 50 MHz down every grow
returns artefacts only, so the test now pins 30 MHz.  The DD-191
chamfer certificate (same annulus) was re-based and its jump criterion
replaced by a c²-smoothness check (see the DD-191 table); tutorial 13
re-run — its passband moved 2.90 → 3.02 GHz with the loaded
resonator, so the monitor band is now derived (`F0_LOADED`, half
the design bandwidth either side; centre confirmed to 0.04 %)
instead of quoted, the phase drift across the band reads 62 / 97 /
129°.  Recorded in `known-bugs.md` (KB-031) and `CHANGELOG.md`
(0.4.7 entry, since the change shipped there).

**Files:** `src/magnelio/geo/_occ_backend.py` (`_PlanarSectionEngine`
facet path: `_build_facets`, `_section_facets`, `_lift_to_surfaces`,
`_refine_segments`, `_weld_nodes`, `_facet_edge_table`,
`_annotate_sections`), `src/magnelio/geo/_polygon_clip.py`
(`orient_nested_contours`), `tests/unit/test_facet_section_engine.py`,
`tests/unit/test_section_contour_orientation.py`,
`docs/methods/meshing-conformal.md`, `docs/methods/geometry.md`,
`examples/tutorials/19_cassegrain_reflector.py`, `known-bugs.md`
(KB-031), `CHANGELOG.md`.

## DD-200 — Grid planes carry their provenance: post-hoc attribution, example pins, mesh-section plot

**Status:** Decided 2026-08-27 with the developer (priorities: provenance
first, then test anchoring, plot as a separate function); implemented
2026-08-27.

**Problem.**  KB-028: the DD-191 edge pass put a phantom plane through
the axis of every cylinder inscribed in a box, and three releases
shipped with it.  The mesher classifies every plane it places —
critical face / bbox extent (`critical_raw`), edge (`feature_raw`),
singular mark, thin sheet, wire, symmetry, forced anchor — and forgets
it: the classification lives in locals of `from_geometry` and
`GridLines` is coordinates only.  `plot_cross_section(mesh=)` shows the
lines, not their reason; nothing pins the plane set of any model, so a
rule change surfaces, if at all, through a physics test downstream —
KB-028 through the Dey–Mittra ordering tests, three DDs later.

**Decision.**

1. *Record, not thread.*  Next to every raw plane list the mesher keeps
   `plane_sources[axis]: [(position, PlaneSource)]` with the
   contributing shape (index and label in model order, taken before the
   wire split) and attributes them **after** the merges by position
   (`attribute_planes`, `mesh/_planes.py`): nearest candidate within
   `2 × feature_gap` among the final planes, the dropped edge planes
   (`edge` sources only) and the floor-absorbed material planes
   (material-class sources only).  Every merge stage moves an entry by
   at most `feature_gap` and compares at the boundary, a longer snap
   chain can exceed 1×, and final planes are pairwise farther apart
   than the gap — so 2× is safe and unambiguous.  Whatever finds no
   candidate is `unplaced`: the invariant that catches a silent drop.
   The merge helpers (`_merge_axis_planes`, `_snap_planes`,
   `_merge_feature_planes`, `_floor_merge_planes`) are untouched.
2. *On the mesh.*  `Mesh.planes: GridPlanes | None` — frozen record with
   per-axis `PlaneRecord(position, sources, node, singular, domain_end,
   h_fine, moved_to)`, `h_bulk` per interval, `dropped` / `absorbed` /
   `unplaced`, `n_nodes`, `pml_cells`, `feature_gap`.  `position` is the
   merged geometry coordinate, `node` the index in the final axis
   (offset by the min-side absorber cells), a PMC pull-in is
   `moved_to`.  Carried by every rebuild (`with_boundary_conditions`,
   `with_pec_boundaries` — the DD-123 `elements` trap — and the
   `replace` paths); `from_grid` and older stores give `None`.  Store:
   JSON in a string *dataset* `mesh/planes_json` — HDF5 attributes cap
   at 64 KB and a CAD import lists hundreds of shapes.
3. *Report.*  `print(mesh.planes)` (`GridPlanes.summary`, the
   `PortReport.summary` style): header with gap, node counts and PML;
   per axis the `h_bulk` range and `h_fine`; one row per plane with
   position, sources grouped by kind (`face #0 Brick(air) #1
   Cylinder(pec); extent #0 #1; edge #0`), flags `domain end` /
   `singular` / `h_fine` where refined / `-> node at`; then dropped,
   absorbed, unplaced.
4. *Pins.*  `tests/unit/test_gallery_planes.py` runs every
   `examples/{tutorials,howto}/plot_*.py` to its first
   `Mesh.from_geometry` (the classmethod is wrapped to capture the mesh
   and stop; the 3D viewer, section plots and `plt.show` are stubbed)
   and compares `as_dict()` against
   `tests/unit/data/gallery_planes/<dir>__<stem>.json`: positions
   within `feature_gap / 2`, sources exactly, `n_nodes` exactly, `h_*`
   at 1e-3; unmatched planes within ten gaps are reported as *moved*,
   otherwise added / removed; leftovers as set differences.
   `MAGNELIO_UPDATE_PLANE_PINS=1` regenerates.  Excluded:
   `19_cassegrain_reflector.py` (370 s mesh; rule: first mesh > 10 s).
   Measured: 28 scripts in 38 s; every `unplaced` list empty, no plane
   without a source, no interior plane from a bounding box alone; the
   edge-only planes are the chamfer of tutorial 13, the iris and
   equator circles of tutorial 18 and the line/ring junctions of
   tutorial 10 — the DD-191 set.
5. *Plot.*  `plots.plot_mesh_section(mesh, normal, position, geometry=,
   fill=, flip=, legend=)` / `Mesh.plot_section`: `pcolormesh` of the
   `material_id` slab on the node coordinates (material palette, air
   transparent), absorber cells hatched, graded fill nodes as
   hairlines, plane lines styled by their highest-ranking kind
   (sheet > symmetry > forced > face > wire > edge > extent), singular
   planes with end markers, the model's outline via
   `plot_cross_section(fill=False)` (new keyword).  Sub-cell layers
   (PEC masks, f_A / f_L on edges) are deliberately later work.

**Rejected.**
- *Threading shape ids through the merge helpers*: every merge
  invariant (KB-013 exact-member midpoint, anchor snap, floor merge)
  would carry a parallel payload for a diagnostic; positional
  attribution reproduces the result without touching them.
- *A node → plane index map in `GridLines`*: the grid container stays
  coordinates-only (store, `from_grid`, solver read it); provenance is
  a property of the *build*, hence on the mesh.
- *String-equal pins*: float noise in kernel coordinates would repin
  every file; the tolerance compare fails exactly when a plane moves by
  a snap or more.
- *Pinning the node arrays*: any grading change would repin everything
  without saying what moved; `n_nodes` plus `h_bulk` / `h_fine` show a
  grading change with its reason.

**Side finding.**  An edge plane closer than `min_cell_size` to an
anchor is discarded by `_floor_merge_planes` without joining `absorbed`
(its `is_material` is False) — it would appear as `unplaced`.  No
example model triggers it; left as is, the record now makes it
visible.

**Amendment 2026-08-27 (sub-cell layers).**  The material fill alone
made tutorial 02 look like a staircase solver, because `material_id`
is the cell-centre classification the conformal matrices override.
Three fills now, all pure NumPy over data the mesh already carries:
`coverage` (default) — the geometric PEC area of the primal faces
normal to the cut in the nearest node plane (`FaceMaterialData.
A_face_pec`, DD-087; exact for classifier candidates, staircase
elsewhere) blended over the classified material colour; `material` —
the classification; `conformal` — `eps_avg` of the dual faces of the
normal edges on the dual tiling (cat 0 → staircase owner cell, cat 1/2
→ `eps_avg`, cat 2 under the free-area floor / cat 3 / masked → 0).
`edges=True` draws the in-plane edges of the nearest node plane:
`pec_mask_edges`, `0 < L_free/L_primal < 1`, `enlarged_cell_donor ≥ 0`.
Meshes without sub-cell data raise.  Decided with the developer after
the first probe: the dual-face `eps_avg` had been proposed as the
default, but on the coax it shows the pin as a plus shape one node
larger than the geometry — not a bug: the in-plane edges cutting the
pin have `f_L ≈ 0.27 < η = 0.4`, are masked and lent out, and the
DD-053 tangential rule (both endpoints in the same masked component)
then re-masks the normal edges around them although the conformal
pass had measured `ε̄ = 1.65, f_L = 1` there.  Correct for the field
normal to the cut, irrelevant for TEM, misleading as a default.  The
coverage fill shows the ring the H-side actually integrates (0.88 /
0.48 / 0.22 / 0.08 / 0.02 along the outer wall).  Tutorial 02 shows all
three side by side.  Probe layers at `Nz // 2` of a CPML-extended
grid sit in the absorber extension where the OCC sections find no
geometry — take the layer from the position, never from the count.

**Files:** `src/magnelio/mesh/_planes.py` (new),
`src/magnelio/mesh/mesher.py`, `src/magnelio/mesh/__init__.py`,
`src/magnelio/io/project.py`, `src/magnelio/post/plot_mesh.py` (new),
`src/magnelio/post/plot_geometry.py` (`fill=`),
`src/magnelio/plots/__init__.py`, `tests/unit/test_grid_planes.py`,
`tests/unit/test_plot_mesh.py`, `tests/unit/test_gallery_planes.py` +
`tests/unit/data/gallery_planes/`,
`examples/tutorials/plot_02_coax_line.py`,
`docs/methods/meshing-conformal.md`, `spec.md` §2.2, `CHANGELOG.md`.

## DD-201 — Mesh-build benchmark on production geometry classes; Lange-coupler how-to

**Status:** Decided 2026-08-27 with the developer, implemented
2026-08-27.  Developer choices: the test geometry is a Lange coupler
on 254 µm alumina at 10 GHz, delivered as a how-to *and* a benchmark
script with a size ladder; a tandem of two 8.34-dB four-finger Langes
was the first choice and was dropped on measurement (below); bond
wires were to be `ThinWire` and became resolved ribbon bonds because
of the DD-080 radius rule; the developer's premise throughout: the
test case has to be hard enough to measure the mesher on real-world
scale — "everything must scale".

**Problem.**  STATUS carried "mesh-build speed, item 1: a process
pool over edge chunks for `compute_edge_pec_fractions`" as the next
lever, on the strength of one private fixture (the slotline coupler
at n = 100, DD-102: 22.8 of 33.6 s in the edge pass).  Nothing in the
repository measured the mesher on a realistic geometry through the
public API, and the only stress case (`profile_csg_scaling.py`, 1002
spheres) exercises the CSG tree.  A decision of that kind has to be
made per geometry class, on numbers that are versioned and
reproducible.

**Deliverables.**

1. `examples/howto/plot_lange_coupler.py` — the 3-dB interdigitated
   coupler: Ou's synthesis (k fingers, coupling C → even/odd impedance
   of one adjacent pair; k = 4, 3 dB → 176.4 / 52.5 Ω, pair coupling
   −5.3 dB), a two-dimensional design step on the port solver (4
   widths × 4 gaps, sixteen slice meshes; the impedance ratio follows
   the gap, the geometric mean the width; two 1-D interpolations),
   ribbon bonds as three bricks, right-angle leads, four window ports,
   scoreboard with the balance band.  Measured (GPU, 0.22 M cells,
   120 k steps, 110 s): |S31| −2.72 dB, |S21| −3.32 dB, ∠S31 − ∠S21 =
   89.8° and flat over 6–14 GHz, |S11| −31.9 dB, |S41| −32.1 dB,
   |balance| ≤ 1 dB from 7.35 GHz to the band top.  Design on this
   grid: w = 12.6 µm, s = 25.4 µm (w/h 0.05 — the thin-film edge;
   635 µm alumina scales the transverse dimensions by 2.5), L =
   3.135 mm at ε_mean 5.717.
2. `benchmarks/bench_mesh_build.py` — mesh build only, wall time per
   pass by wrapping the pass entry points from outside
   (`compute_edge_pec_fractions`, `compute_face_material_areas` split
   by `prop`, classification sections and fill, plane extraction, PEC
   fuse, `cross_section_polygons` count, `_sample_and_admit` as "the
   pool really ran"); "other" = CSG evaluation + plane merge + grid +
   masks.  Pool arms off (`MAGNELIO_SECTION_WORKERS=0`), auto
   (production), forced (`_SECTION_PARALLEL_MIN_QUERIES=1`,
   `_SECTION_PARALLEL_MIN_FACE_WORK=0`, `_SECTION_POOL_STARTUP_S=0` — a
   policy change, reported as such); the fallback warning is an error
   in the pooled arms.  Families: `lange` (a row of n couplers, the
   how-to's layout, n = 1…16), `array` (n × n patch array at 10 GHz
   with an H-tree corporate feed and λ/4 transformers, one `Union` per
   copper layer), `posts` (the DD-141 row rebuilt: n posts of 0.5 mm
   radius at 2 mm pitch).  JSON is rewritten after every point
   (`benchmarks/results/bench_mesh_build.json`); the private record is
   `investigations/mesh-build-bench/MEASUREMENTS.md` (internal
   dossier).

**Measured** (CPU, fastest of one build, 2026-08-27):

| family | n | cells | faces | total off | auto | forced | edge pass | areas µ off → auto | classify | other |
|---|---|---|---|---|---|---|---|---|---|---|
| lange | 1 | 230 622 | 274 | 3.6 s | 3.6 | 9.4 | 0.8 | 0.9 → 0.9 | 0.1 | 0.5 |
| lange | 4 | 922 488 | 1 060 | 19.6 | 19.6 | 22.6 | 3.5 | 5.3 → 5.3 | 0.3 | 3.4 |
| lange | 16 | 3 689 952 | 4 204 | 171.9 | **140.8** | 140.7 | 15.8 | 51.8 → 20.9 | 1.3 | 38.2 |
| posts | 60 | 97 200 | 306 | 8.4 | 8.3 | 16.5 | 1.0 | 3.0 → 3.0 | 1.5 | 0.1 |
| posts | 240 | 385 200 | 1 206 | 70.9 | **44.7** | 44.9 | 4.3 | 26.9 → 11.5 | 15.6 | 0.6 |
| array | 2 | 84 150 | 383 | 23.1 | 23.2 | 23.1 | 1.1 | 0.4 | 0.1 | 20.8 |
| array | 4 | 222 497 | 2 147 | 475.1 | 478.7 | 476.9 | 4.3 | 1.4 | 0.2 | **466.1** |

**Decision on the edge pool: not the lever.**  On all three classes
the edge pass is 1–11 % of the build at the largest size (Lange row
15.8 of 141 s, posts 4.3 of 45 s, array 4.3 of 475 s), and it scales
linearly (DD-101's prefilter holds).  Item 1 is closed as measured;
three new items replace it, in value order: (1) the **N-ary Boolean
fuse of a planar copper network** is superlinear — 20.8 s for 30
strips, 466 s for 120 — and is 98 % of the array's build; fusing in
tiles (associative) is the known remedy (the DD-179 board-import
bbox-cluster fuse is the in-house precedent) — *corrected 2026-08-27
by DD-202: the residual was never timed; the fuse of those 107 strips
takes 0.74 s, the 466 s were the thin-sheet footprint rasteriser.  The
fuse is superlinear one tier up (443 strips 16 s) and stays the open
item with that number*; (2) the **classification sections** (`batch_cross_sections`,
15.6 of 45 s on the post row) have no pool at all — the prefill pool
could cover them; (3) the **face pass grows faster than the cell
count** on the Lange row (areas µ 15.5 → 51.8 s for 8 → 16 couplers:
every plane cuts every coupler), where the section pool already
halves it (20.9 s) — a per-plane face prefilter would take the
rest.  DD-141's sample admission is confirmed on production geometry:
auto equals forced where the pool pays (Lange 16, posts 240) and
declines correctly where it does not (forced costs the 5 s startup
per admitted call: Lange 1 3.6 → 9.4 s, posts 60 8.4 → 16.5 s).  The
array's 8 × 8 rung was not run (the fuse alone would take hours);
that is the finding, not a gap — *DD-202 ran it: 61 s.*

**Found on the way.**

- **KB-032** (fixed here): two thin sheets at one nominal height — the
  finger bricks and the Boolean-returned leads — differ by one ulp;
  sheet planes are verbatim anchors, so the merge warned "forced
  planes … closer than min_feature_gap (user positions win)" for
  planes nobody had forced and fed the singular grading a feature
  size of 1.7e-21 m; the grid deduplicated the sliver itself.
  `_unify_thin_sheet_positions` clusters sheet planes within the
  feature gap before they become anchors (a user-forced plane within
  reach wins, otherwise the lowest sheet).
- **KB-033** (fixed here): `model.plot()` raised on the ribbon bonds —
  the viewer's tessellation deflection (5e-4 of the body's diagonal)
  fell below OCC's confusion precision for a 66 µm body at metre
  scale; `_tessellate_shape` floors it at 1.1e-7 for the viewer and
  the ParaView export alike.
- **The odd mode of a tight gap converges only with surface cells
  below 10 µm**: Zo of a 35/25 µm pair 22.5 → 31.6 → 37.2 → 41.5 → 42.0 Ω
  for surface cells 64 / 32 / 16 / 8 / 1 µm (`singularity_refinement`
  1 / 2 / 4 / 8 / 64); a 240 µm line reads 45.7 Ω at the default and
  49.5 Ω (Hammerstad 49.5) from k = 4 on; the odd ε_eff keeps moving
  until the cells are below 4 µm (5.42 → 4.83).  The how-to designs
  and runs at k = 8 with a 6 µm floor; recorded in the meshing
  chapter.
- **A finger narrower than two cells is not a conductor on the port
  face** (one line mode, the port falls to the TE/TM path with an
  inhomogeneous-filling error).
- **The default run is unbounded and energy-gated at 70 dB**; a
  closed lossless housing holds the last few percent of energy in
  modes the ports barely see (−67.6 → −68.6 dB over 150 k steps), and
  the first attempt ran 1.8 M steps into a timeout.  The how-to sets
  `total_time_steps` and says why; the housing length keeps its first
  x-resonance above the band.
- **Two layout defects with a clean S-parameter signature**: two
  ribbon bonds of one end at the same x and height cross and short
  line A to line B (|S21| ≡ |S41| across the band; bonds staggered
  now), and two 240 µm leads running side by side at the fingers'
  spacing are a coupler of their own (right-angle leads now).
- The tandem of two 8.34-dB four-finger Langes (Ou: 130 / 75.6 Ω,
  mean 99 Ω) needs 10–15 µm fingers at 150 µm gaps on 254 µm alumina
  (25 µm fingers reach 81 Ω) — a GaAs-MMIC form; the single 3-dB
  Lange is the alumina part.  `ThinWire` bonds are excluded by
  DD-080's radius rule on this grid (r < 1.8 µm).

**Files:** `examples/howto/plot_lange_coupler.py`,
`benchmarks/bench_mesh_build.py`, `benchmarks/results/bench_mesh_build.json`,
`src/magnelio/mesh/mesher.py` (`_unify_thin_sheet_positions`),
`tests/unit/test_thin_sheet_detection.py`
(`TestThinSheetAnchorUnification`),
`tests/unit/data/gallery_planes/howto__plot_lange_coupler.json`,
`docs/methods/ports.md`, `docs/methods/meshing-conformal.md`,
`src/magnelio/io/paraview.py` (`_OCC_MIN_DEFLECTION`), `tests/unit/test_plot_3d.py` (`TestTinyBodies`), `spec.md` §10.3, `known-bugs.md` (KB-032, KB-033), `CHANGELOG.md`;
`investigations/mesh-build-bench/` (internal dossier: probes,
worksheet, GPU runs).

## DD-202 — Thin-sheet footprints from one section, not a solid classification per edge

**Status:** Decided and implemented 2026-08-27.  Planned as "the tiled
fuse" (DD-201's item 1); the plan-mode measurement turned it into this.

**Problem.**  DD-201 attributed the 4 × 4 patch array's residual — 466
of 475 s — to the N-ary Boolean fuse of its 107 copper strips.  The
residual was never timed.  A cProfile of the 2 × 2 build put 16.8 of
24.8 s into `BRepClass3d_SolidClassifier.Load` (9 160 calls) and 20.6 s
cumulative into `rasterize_thin_sheet_footprint`; the fuse of the 107
strips takes 0.74 s.  The footprint rasteriser (the WP-M2 fix that
replaced the bbox fill, so an L-shape or a ring keeps its corner or
bore open) classified every tangential edge midpoint inside the sheet's
bbox rect against the sheet's OCC solid — one fresh classifier per
point, whose `Load` is O(faces).  Candidates grow with the cells of the
rect and each costs the face count: 15 ms per point on the 1 336-face
copper network, superlinear on exactly the class Magnelio is for —
patch arrays with feeds, imported board layers (one part per layer, so
the rect is the whole board), any `Union` of traces.  The cell
classifier had left this path years ago (`classify_cells_from_cross_sections`:
one `cross_section_polygons` per plane, vectorised `points_in_polygon`,
even-odd rule); the sheet rasteriser was the one pass that never got
the same treatment.

**Decision.**  `rasterize_thin_sheet_footprint` takes **one section**
of the sheet solid at the probe plane (mid-thickness; in the face with
`exact_at_faces` when no far face is known), with the cell classifier's
chord budget (`CLASSIFY_DEFLECTION_FRACTION` × smallest transverse
cell) and an escape step capped at half the membership tolerance so
the DD-157 nudge ladder cannot leave the metal.  The candidate edge
midpoints of both tangential components are built as arrays, tested
by the even-odd rule over the contours (`points_in_polygon`) and OR-ed
with a boundary band of the old classifier tolerance
(`points_near_polygon`, new in `_polygon_clip.py`: vectorised
point-to-segment distance, segments outer, points chunked) — edges on
the outline stay metal, the inclusive semantics of the rect path and of
the classifier's `ON` state.  The mask is written with vectorised index
arithmetic.  The classifier path stays as the fallback for a section
that raises or comes back empty (`_rasterize_by_classifier`, its `Load`
hoisted to one per sheet); `point_in_shape` itself is untouched — its
other callers make one call per sheet or face.  The benchmark wraps the
rasteriser (`sheets`, top-level) and `boolean_union` (`fuse`, nested in
`pec_fuse` and the edge pass, hence not part of `other`) so a residual
can no longer pose as a pass.

**Measured** (CPU, `benchmarks/bench_mesh_build.py --family array --pool off`):

| n | cells | faces | strips | before | after | edge pass | areas µ | sheets | fuse | other |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 84 150 | 383 | 23 | 23.1 s | **2.7 s** | 1.1 | 0.4 | 0.0 | 0.1 | 0.2 |
| 4 | 222 497 | 2 147 | 107 | 475.1 s | **10.2 s** | 4.4 | 1.4 | 0.1 | 0.9 | 1.0 |
| 8 | 664 378 | 10 799 | 443 | — | **61.4 s** | 22.5 | 5.9 | 0.8 | 17.8 | 18.9 |

Equivalence: bit-identical to the classifier path on the L-shape
fixture; on a ring the two differ only along the tessellated rim
(≤ 3 % of the sheet edges, gated).  The full ladder rerun is in the
JSON (all rows carry the new columns): the Lange row's sheets cost
0.2 s at n = 16 (many small rects), the post row's 0.0 s — the
rasteriser was an array-class problem; the new `fuse` column shows the
Lange row's `Difference(air, *metal)` tool fuse at 23 of 167 s
(n = 16), the second place the fuse tier surfaces.  Lange 16 auto
136 s, posts 240 auto 45 s — DD-201's pool verdicts stand.

**Re-ranked open items** (STATUS): (1) the N-ary fuse *is* superlinear
one tier up — 0.74 → 16 s for 107 → 443 strips, 18 of the 8 × 8's 61 s
— tiles or the sheet-level fuse of DD-179's deferred list; (2) the
classification sections have no pool; (3) the per-plane face
prefilter.  The edge pass is 22.5 s of the 8 × 8 build (37 %) — DD-201's
"1–11 %" verdict held up to the sizes it measured; at 10 799 faces the
DD-101 residue ("a bin/BVH over face boxes") is back in view.

**Files.**  `src/magnelio/mesh/_conformal.py` (`rasterize_thin_sheet_footprint`,
`_rasterize_by_classifier`, `_sheet_*` helpers), `src/magnelio/geo/_polygon_clip.py`
(`points_near_polygon`), `tests/unit/test_thin_sheet_footprint.py`
(equivalence, ring, boundary), `tests/unit/test_polygon_clip.py`,
`benchmarks/bench_mesh_build.py` (+ `results/bench_mesh_build.json`),
`docs/methods/meshing-conformal.md`, `spec.md` §10.3, `CHANGELOG.md`;
DD-201 amended in place.  `investigations/mesh-build-bench/`
(internal dossier: `probe_sheet_footprint.py`, M7).

## DD-203 — Patch-array how-to: element-to-array design on the grid's line impedances, quarter-wave row offset, shielded launch port

**Status:** Decided and implemented 2026-08-27, branch
`feat/patch-array-howto`; the 2 × 2 topology was the developer's
choice over the 1 × 4 row this record's probe started from.
Amended 2026-08-27 (DD-204): the far-field deficit booked here as
KB-035 was the domain top 0.29 λ above the copper, not the window
port; `h_box` is now 0.7 λ and the page reads |S11| −14.4 dB at f0,
3.3 % band, D 13.0 dBi, G 12.9 dBi (+0.4 dB — the pattern amplitude
had been low), +5.6 dB over the element.

**Problem.**  The array family of the mesh-build benchmark (DD-201)
was a meshing fixture, not an antenna; with the sheet rasteriser fixed
(DD-202) a real patch-array how-to became affordable.  The probe
(`investigations/patch-array/`, internal record, 16 measurement
sections) found four things the page had to be built around.
(1) *No pin on a wide line.*  A lumped port terminating the 2.4 mm
50 Ω trunk reflects −17 dB on the how-to grid regardless of end
position or port impedance (the 0.7 mm 100 Ω line: −38 dB), and that
reflection interfered with the network's own to produce
parity-dependent matches (1 × 2 −13 dB, 1 × 4 −20 dB, 1 × 8 −14 dB)
that no transmission-line model reproduced.  (2) *Anti-phase rows.*
Two rows fed from facing edges by a mirror-symmetric network radiate
in anti-phase (a broadside null, measured −20 dBi); at 0.75 λ no
collision-free in-phase 2 × 2 network fits without a meander.
(3) *Grid impedances.*  The closed-form widths come out 11–13 % low on
a 0.25 mm floor (44.6 / 63.1 / 86.6 Ω for 50 / 70.7 / 100), converging
only with `singularity_refinement` and a 0.05 mm floor (48.8 Ω) — five
times the run time; the ratios hold to 1–3 %.  (4) *Thin sheets and
absorbing faces.*  A microstrip window port on a CPML wall was refused
as a hollow cross-section — KB-034, the sheet mask was never extended
into the absorber.

**Decision.**  The page (`examples/howto/plot_patch_array.py`)
designs a 2 × 2 on 0.787 mm ε_r 2.2, 10 GHz, lattice 0.75 λ (H) ×
0.85 λ (E):

- *Element*: cavity-model dimensions, one run at a nominal inset
  reports the resonance and trims the length (9.653 → 9.219 mm), then
  a three-point inset sweep on the trimmed patch picks the feed
  (0.25 L: −26.8 dB at 9.95 GHz, D 7.4 dBi).  Inset before trim, the
  original order, left the element 0.9 % low at −18 dB because the
  notch shortens the resonant path.
- *Network on the grid*: the three line impedances and ε_eff come from
  `solve_ports()` on slices with the production mesh control; the
  ports are referenced to the line modes, the text states the absolute
  error and its price.
- *Row phase without a meander*: the crossbar sits λ_g/4 below the
  array centre, so the arm to the upper row is λ_g/2 longer than the
  arm to the lower row — all arms straight.  The short arm needs
  (p_y − L)/2 − λ_g/4 of room, which is what sets p_y ≥ 0.8 λ on this
  substrate; 0.85 λ keeps the crossbar 2.6 mm (3.3 h) from the lower
  row's fed edge, and that distance decides the E-plane symmetry
  (0.8 λ: ±30° nulls at −13 / −0.2 dB; 0.85 λ: within 1.2 dB at ±20°).
  A δ trim does not centre the beam — the asymmetry is the lower row's
  environment, not line phase.  A higher-ε_r substrate (3.66, 1.524 mm)
  would allow 0.74 λ but its thick, wide trunk excites a mode in the
  launch and couples harder to the crossbar; rejected.
- *Feeds*: the element on a lumped port at its 100 Ω line (a pin on a
  narrow line is clean), the array through a shielded launch — two
  walls and a roof around the trunk for 6 mm at the `ymin` wall, the
  window port in that cross-section (DD-198 enclosure rule), which is
  what a connector body does.  Needs KB-034's fix.
- *Array trim*: the network loads the rows, the array resonates 1.6 %
  above the element; one trim of L by f_dip/f0 on the array, the same
  rule as for the element.
- *Check*: principal-plane cuts against element cut × array factor;
  the far-field power deficit of the window-port model (KB-035) is
  stated in the text and realized gain named as the robust number.

**Measured** (GPU, `howto_run3.log`, internal record): element as
above; array 0.53 M cells; first run −17.2 dB at 10.09 GHz, trimmed to
L = 9.30 mm: |S11| −16.7 dB at 10.00 GHz, −10 dB band 9.81–10.21 GHz
(4.0 %), D 12.96 dBi, G 12.45 dBi, 5.5 dB over the element (four
sources: 6.0; the rest is the E-plane shoulder of the 0.85 λ row
pitch).  With the inset chosen before the trim (0.30 L) the array
matched −25.6 dB but the element sat 0.9 % low — the network shifts
the inset optimum, which the page says rather than tunes.  Six
simulations ≈ 4 min GPU; expected CPU cost in the docs build
8–10 min.

**Files.**  `examples/howto/plot_patch_array.py`; `docs/methods/ports.md`
(shielded launch sentence); `known-bugs.md` KB-034 (resolved, DD-198
amendment), KB-035 (open); `CHANGELOG.md`; probe and protocol in
`investigations/patch-array/` (`probe_array.py`, `probe_pin.py`,
`tl_model.py`, `tl_quad.py`, `pattern_mult*.py`, `MEASUREMENTS.md`).

## DD-204 — Far-field power balance: the surface power as a closure check, and how close the Huygens box may sit

**Status:** Implemented 2026-08-27, on the `fix/far-field-power-balance`
branch.  Resolves KB-035.

**Problem.**  KB-035 recorded that the patch-array how-to's far field
collected only 0.84–0.89 of the incident power with the feed through a
window port, against 0.91–0.92 on a lumped pin, and blamed the window.
Both numbers were wrong for a lossless model — PEC copper, a substrate
without loss tangent — whose radiated power must equal $1 - |S_{11}|^2$
= 0.98.  The far-field monitor had no independent check: `P_rad`
could only be compared with the port's accepted power, which mixes the
port bookkeeping into the question.

**Finding** (probes in `investigations/patch-array/`, M18, internal
record).  The Poynting flux through the recording box, computed from
the very samples the transform uses, reproduces the accepted power to
a percent in every configuration tried (lumped and window feed, box
at 0.3 λ or 2 λ, λ/24 to λ/48 cells, graded and uniform grids).  The
transform of those samples fell short by 6.6 % with the domain top —
and with it the box's top face, which carries 87 % of the flux —
0.29 λ above the copper, by 6 % at the same height with the box's
sides moved out to 0.7 λ, by 2.8 % with the cells halved, and by
nothing (−0.4 … +2.5 %) once the top sat 0.7 λ or higher.  Angular
resolution, image-plane position, substrate extent beyond the copper
and grid grading changed nothing.  A wire monopole balances to 1 % a
dozen cells from the box at λ/25; an analytic vertical or horizontal
dipole over ground pushed through `record()` on uniform, sawtooth and
graded grids balances to 0.6 % at 0.33 λ — so neither the transform,
the image expansion nor the sampling is at fault: the FIT near field
of a printed resonator within half a wavelength is not the outgoing
free-space field the equivalence theorem assumes, and the shortfall
is a discretisation effect that fades with distance and resolution.
The pattern is scaled down as a whole: realized gain read 0.3–0.4 dB
low, directivity (normalised to `P_rad`) was right — the opposite of
KB-035's reading.  The window port's own contribution is the
documented outer-wall current in the absorber: `P_surf/P_acc` 0.965
on the launch, unchanged by box height, with the balance closing.

**Decision.**  (1) `post.far_field.surface_power(patches)` computes
$\mathrm{Re}\oint(\mathbf E\times\mathbf H^*)\cdot\hat n\,dS$ in the
library's effective-phasor units; `ntff_transform(surface_power=)`
carries it on `FarFieldResult.surface_power`, and `power_balance`
returns $P_\mathrm{rad}/P_\mathrm{surf}$.  (2) `MonitorFarField.result`
computes the flux for every call (cheap next to the transform) and
warns beyond a 5 % imbalance (`_CLOSURE_TOLERANCE`; good boxes scatter
±2.5 %, bad ones sit at −6 … −10 %), naming both powers and the cure —
clearance to the absorbing faces, half a wavelength or more.  No
geometric heuristic: the balance is measured, model-independent, and
also covers a box that is far enough but badly resolved.  (3) The
patch-array how-to sets `h_box = 0.7 λ` (21 mm; +40 % cells) and states
the balance; its element and array figures are re-measured (DD-203
amendment).  (4) `docs/methods/far-field.md` gets the section *Power
balance*.  Not done: moving the box inward from the absorber (the
box would still sit in the near zone of whatever touches the
boundary, and the flux shows the placement is not the issue); a
transform-side near-field correction (the samples, not the transform,
are off).

**Measured.**  Patch element on a pin, 10 GHz, accepted 0.983:
`P_surf` 0.977, `P_rad` 0.912 at h_box 12 mm (top 0.29 λ) → balance
0.934; 0.975/0.975 at 24 mm → 1.000; 0.978/0.960 at 40 mm → 0.982;
0.980/0.976 at 68 mm → 0.996; at 21 mm (the how-to) 0.978/1.003 →
1.025.  Same element on the window launch: 0.894 at 12 mm, 1.007 at
21 mm, `P_surf/P_acc` 0.965 throughout.  Wire dipole/monopole
(`validation/farfield_dipole_certificate.py` grid): 0.980/0.953 at
the certificate's 2-cell margin, 0.991–1.007 at 12 and 24 cells,
λ/65 and λ/25 alike.  Gates: `tests/unit/test_far_field_closure.py`
(analytic dipole box balances to 1 %; a 60° E/H rotation breaks it;
the monitor carries `surface_power` and warns; an empty box does not),
the DD-173 certificate unchanged (2.2 % at its close box).
Tutorial 19 (Cassegrain, horn neck through a window port, λ/2
clearance, 2026-08-27 rerun): `P_acc` 0.938, `P_surf` 0.872,
`P_rad` 0.877 → balance 1.005, no warning; its quoted 0.93 is
entirely `P_surf/P_acc` — the outer-wall current of the neck in the
absorber, as the page says — so the feed-guide share is 3.5 % for a
shielded microstrip launch and 7 % for a bare horn neck; the
tutorial's table now carries the balance line.

**Files:** `src/magnelio/post/far_field.py` (`surface_power`,
`FarFieldResult.surface_power`, `power_balance`),
`src/magnelio/monitors/far_field.py` (`result` closure warning),
`tests/unit/test_far_field_closure.py` (new),
`examples/howto/plot_patch_array.py` (`h_box`, balance paragraph),
`examples/tutorials/19_cassegrain_reflector.py` (balance line and reading),
`docs/methods/far-field.md` (*Power balance* section),
`known-bugs.md` KB-035 (resolved), `CHANGELOG.md`, `STATUS.md`;
probes `investigations/patch-array/probe_kb035*.py`, `kb035_*.py`
and M18 in `MEASUREMENTS.md` (internal record).

## DD-205 — Unions of prisms are fused in their plane; the effective PEC solid is cut per shape, not accumulated

**Status:** Decided and implemented 2026-08-28, branch `perf/planar-fuse`.
DD-202's re-ranked item 1.

**Problem.**  DD-202 measured the N-ary fuse of a planar copper network
as superlinear one tier up: 107 strips 0.74 s, 443 strips 16.2 s.  The
premise "superlinear in N" was wrong in an instructive way: the same
443 strips moved apart so that none touch fuse in 0.45 s.  The general
fuser's cost grows with the *interference* between operands, and
coplanar overlap is its worst case — every split of a cap face is
matched against every other cap piece in the plane.  The measured
alternatives: a 4 × 4 tiling of the strips 3.6 s (hierarchy helps,
does not cure); `BOPAlgo_GlueShift` 3.6 s and **wrong** (volume a
tenth — refuted, the glue modes assume no real intersections);
fusing the bottom caps in the plane and raising the result once,
0.74 s at the same volume to nine digits and 6 924 → 640 faces.  That
is what the board importer has done since DD-179 ("no Boolean is
3-D"); every other union in the library still went to the general
fuser.

A second finding on the way: the benchmark's `fuse` column on the Lange
ladder (23 s of 136 s at 16 couplers) was not the `Difference` tool
fuse (0.7 s) but `build_effective_pec_solid`, which subtracted the
*accumulated* union of all higher-priority shapes from each PEC shape
and grew that union pairwise on a growing compound — the O(N²) pattern
`boolean_union`'s own docstring records as 28 s vs 0.1 s, surviving in
the neighbouring function.  320 metal pieces: 35.9 s.

**Decision.**  `boolean_union` (`src/magnelio/geo/_prism_fuse.py`)
classifies each operand from its B-Rep as a prism along an axis when
its planar faces normal to that axis lie on exactly two levels and
every other face is ruled along it (planes containing the axis,
cylinders about it — bricks, cylinders, extruded profiles and imported
plates alike; spheres, cones, chamfers and steps are not).  It picks
the axis on which the most operands share an interval, fuses each
(axis, interval) group through its bottom caps — bounding-box clusters
in the plane, one planar fuse per cluster, `UnifySameDomain` for the
seams, one prism per fused face; operands whose caps touch no other
are kept untouched — and fuses what is left in space only inside
clusters of interfering 3-D bounding boxes.  An operand of a compound
that meets another through one of its solids has all of its caps
re-raised, so nothing of it is lost.  Empty operands (a Boolean that
removed everything) are dropped before clustering.  The result is the
general fuser's point set with fewer faces.  `Difference` tool fuses,
the mesher's PEC and edge-pass fuses and the importer's layer merge all
go through it; the importer's bounding-box sweep became the shared
`cluster_boxes`.

`build_effective_pec_solid` cuts each PEC shape once, N-ary, against
the higher-priority shapes whose bounding boxes reach it, and fuses the
contributions in one pass — same last-wins semantics, same volume and
face count as the pairwise loop on the Lange row.

**The edge pass had been living off the seams.**  The first ladder
rerun kept the 8 × 8 array at 59.6 s: fuse 17.7 → 0.8 s and the
residual 18.8 → 1.1 s were won, but the edge pass rose 22.1 → 49.2 s
with the face count 10 799 → 1 326.  Per-face profiling
(`probe_edge_pass_faces.py`): the two unified caps took 4.9 of 5.1 s of
intersector time on the 4 × 4 at the *same* number of calls as the
seamed pieces (132 k vs 142 k) — `IntCurvesFace_Intersector.Perform`
classifies the hit point at O(edges of the face), 16 µs on a 640-edge
cap against 3 µs on a 20-edge piece.  Two refuted fixes: tiling the
cap's bounding box into occupied sub-boxes (selectivity was never the
problem: 93 k → 83 k calls), and skipping planar faces for lines
parallel to them (the kernel returns no point for those anyway, but
they were few: 132 k → 125 k, and the NumPy test cost more per call
than it saved on the small arrays).  The fix that held:
`_PrefilteredLineSolid` cuts a large axis-aligned planar face
(≥ 24 edges) into *classification pieces* — a coplanar `Common` per
tile of a grid sized for about 12 edges per piece — and each piece is
a candidate row with its own box and intersector, oriented to the
face's effective normal; the solid itself is untouched.  A hit inside
a piece is the face's hit; a hit within tolerance of a tile border
reads `ON` and goes to the exact point classifier.
`tests/unit/test_line_solid_pieces.py` pins the edge fractions
bit-identical with and without pieces on a comb whose edges lie on
grid lines (grazing and clean crossings alike).  Piece targets of 6
and 24 edges were no better on the 8 × 8 (rows cost calls, edges cost
per call); 48 made the 4 × 4 slower.  This is the same weakness a
board layer or a ground plane pierced by vias has always paid — one
face spanning the model — and it is fixed for them too.

**Consequences.**  Bit-identity of the fused topology is given up
where operands are coplanar prisms: their seams are gone, so section
polygons have fewer collinear vertices.  The gallery-plane pins
(28 scripts) and the unit suite are unchanged; the how-to scoreboards
are re-measured below.  `tests/unit/test_import_cad.py` built its
seamed fixture with `boolean_union` and now uses the plain fuser.

**Measured** (`investigations/mesh-build-bench/probe_fuse_scaling.py`,
`probe_fuse_routes.py` — internal record; CPU, 16 cores):

| case | before | after | faces |
|---|---|---|---|
| Union, 4 × 4 array (107 strips) | 0.64 s | 0.10 s | 1 336 → 196 |
| Union, 8 × 8 array (443 strips) | 16.1 s | 0.79 s | 6 924 → 640 |
| `build_effective_pec_solid`, Lange 4 (80 PEC) | 2.9 s | 0.44 s | 448 → 424 |
| `build_effective_pec_solid`, Lange 16 (320 PEC) | 35.9 s | 1.9 s | 1 792 → 1 696 |
| same, with a background-PEC brick | 35.7 s | 4.7 s | — |

Full ladder after both changes (`benchmarks/bench_mesh_build.py
--family all --pool off auto forced --json`, CPU idle, `auto` arm;
before = DD-202's ladder):

| family, n | cells | faces before → after | total before → after | edge pass | fuse | other |
|---|---|---|---|---|---|---|
| Lange 16 | 3.69 M | 4 204 → 4 044 | 136.3 → **105.7 s** | 15.9 → 16.9 | 23.0 → 0.8 | 34.1 → 32.4 |
| Lange 8 | 1.84 M | 2 108 → 2 028 | 52.6 → 44.7 | 7.3 → 7.9 | 6.3 → 0.4 | 8.9 → 8.7 |
| array 8 × 8 | 664 k | 10 799 → 1 326 | 60.5 → **33.2 s** | 22.1 → 22.8 | 17.6 → 0.8 | 18.7 → 1.1 |
| array 4 × 4 | 222 k | 2 147 → 415 | 10.0 → 8.8 | 4.3 → 6.0 | 0.9 → 0.1 | 0.9 → 0.2 |
| array 2 × 2 | 84 k | 383 → 162 | 2.4 → 2.8 | 1.1 → 1.8 | 0.1 → 0.0 | 0.1 → 0.1 |
| posts 240 | 385 k | 1 206 | 44.9 → 45.1 | 4.3 → 4.4 | 0.5 → 0.0 | 0.6 → 0.6 |

The Lange residual (`other` 32 s) is what remains of the PEC solid and
the section passes; DD-201's pool verdicts stand.  The small arrays'
edge pass is slower by the looser boxes of irregular classification
pieces against the old rectangular strip pieces — within a second.
The how-to scoreboards (Lange coupler, patch array) reproduce to the
printed digit on the GPU; the unit suite (2 504) and the integration
subset (import pipeline, window ports, conformal convergence) are
green.

**Files:** `src/magnelio/geo/_prism_fuse.py` (new), `src/magnelio/geo/_occ_backend.py`
(`boolean_union`, `build_effective_pec_solid`), `src/magnelio/io/_pcb_geom.py`
(`_clusters` → `cluster_boxes`), `tests/unit/test_prism_fuse.py` (new),
`tests/unit/test_import_cad.py`, `src/magnelio/geo/_occ_backend.py`
(`_classification_pieces`, `_PrefilteredLineSolid` rows),
`tests/unit/test_line_solid_pieces.py` (new), `benchmarks/bench_mesh_build.py`
(results re-run), `docs/methods/geometry.md`.

## DD-206 — Point probes carry one loaded classifier per shape; the overlap check runs one Boolean per hub

**Status:** Decided and implemented 2026-08-28, branch `perf/mesh-residual`.
DD-205's re-ranked item 1.

**Problem.**  The Lange ladder's residual — `other` 32 s of 106 s at
16 couplers — was three functions outside the benchmark's wrapped
passes, and one of them, `extract_singular_edge_planes`, hid a fourth
inside the `singular` column (cProfile of the build,
`investigations/mesh-build-bench/probe_lange_residual.py` — internal
record):

* `GeometryModel.validate` → `check_pairwise_overlaps`, 22.4 s: the
  air body of the model is `Difference(air, *320 pieces)` with 2 118
  faces, and its bounding box meets every piece, so the check ran 320
  `BRepAlgoAPI_Common(air, piece)` at 75 ms each — the fuser pays the
  body's face count per call.  All 320 were empty.
* `extract_singular_edge_planes`, 15.6 s: the pocketed air body has
  2 016 concave edges, and the probe into each open wedge walked *all*
  322 shapes with `point_in_shape`, which loads a fresh
  `BRepClass3d_SolidClassifier` per call — 283 000 loads, 13.3 s, for
  probes that hit the second or third shape their box admits.
* `detect_thin_metallizations`, 9.6 s: 384 probes, each re-computing
  the 320 bounding boxes (123 000 `AddOptimal`, 3.3 s) and then loading
  the air body's classifier (14 ms — O(faces)) for the hit, 384 times.

**Decision.**  `PointClassifierSet` (`_occ_backend.py`) holds the
shapes of one probe pass with their bounding boxes and one
`BRepClass3d_SolidClassifier` per shape, loaded on first use and kept;
`first_containing(point, skip=, reverse=)` screens by box (padded by
the classification tolerance, so an `ON` point is never screened out)
and asks the survivors in list order or reversed.  `_material_at` in
the singular-edge pass and `_probe_eps` in the sheet detection walk
that set instead of the model; their semantics (first shape in model
order, last shape wins respectively, identity skip, exotic shapes
skipped) are unchanged and pinned by `tests/unit/test_point_classifier_set.py`
against the per-call loop.

`check_pairwise_overlaps` keeps its bounding-box and same-material
screens but no longer measures the candidate pairs one by one: the
shape with the most candidate partners (the hub) is intersected with
all of them in one `Common`, and since ``vol(A ∩ ∪B_k) ≥ vol(A ∩ B_k)``
for every *k*, a batch below the tightest pair tolerance clears every
pair it holds.  A batch with volume is bisected until the offending
pairs are isolated and measured on their own with the pair's exact
tolerance — the same numbers the pairwise loop reported, in the same
order (`tests/unit/test_geometry.py::TestOverlapBatching` compares
against a copy of the pairwise check and counts the Booleans).  Routes
measured on the Lange 16 model (`probe_overlap_routes.py`): pairwise
22.5 s; pairwise with `SetRunParallel` 22.9 s (no lever); one N-ary
`Common` 2.4 s; N-ary per 3-D bounding-box cluster (32 groups) 3.8 s;
the body cut to each cluster's box, then pairwise 6.3 s.

**Consequences.**  Meshes are unchanged: the gallery-plane pins
(28 scripts, both how-tos among them) and the unit suite pass; the
sheet detection now screens with the classification tolerance as pad
where it used a rounding pad before — the consistent choice, and no
pinned mesh moved.  The overlap check's error path (a real overlap
inside a large batch) costs the bisection, about 2 log₂ N Booleans
instead of N.  The benchmark reports `sheets_detect`, `overlaps` and
prints `singular` and `overlap` columns, so `other` is what is truly
unaccounted.

**Measured** (CPU, 16 cores, Lange 16, `auto` pool):
`singular` 15.6 → 1.0 s, `overlap` 22.4 → 3.1 s, sheet detection
9.6 → 0.3 s, `other` 32.4 → 1.2 s; total **105.7 → 67.7 s**.
Full ladder below.

Full ladder (`benchmarks/bench_mesh_build.py --family all --pool off
auto forced --json`, CPU idle, `auto` arm; before = DD-205's ladder):

| family, n | cells | total before → after | singular | overlap | sheets | other |
|---|---|---|---|---|---|---|
| Lange 16 | 3.69 M | 105.7 → **66.2 s** | 15.6 → 0.9 | 22.4 → 3.0 | 9.9 → 1.6 | 32.4 → 1.2 |
| Lange 8 | 1.84 M | 44.7 → **34.9 s** | 3.9 → 0.4 | 3.0 → 1.0 | 1.2 → 0.7 | 8.7 → 0.6 |
| Lange 4 | 922 k | 16.8 → 14.5 | 1.0 → 0.2 | 0.8 → 0.4 | 0.3 → 0.3 | 2.4 → 0.3 |
| Lange 2 | 461 k | 7.1 → 6.6 | 0.3 → 0.1 | 0.2 → 0.2 | 0.1 → 0.1 | 0.8 → 0.1 |
| array 8 × 8 | 664 k | 33.2 → 33.2 | 0.0 | 0.0 | 1.3 | 1.1 → 0.2 |
| array 4 × 4 | 222 k | 8.8 → 8.9 | 0.0 | 0.0 | 0.2 | 0.2 → 0.1 |
| posts 240 | 385 k | 45.1 → 45.9 | 0.0 | 0.4 | 0.0 | 0.6 → 0.2 |

The Lange rows scaled with the number of pockets in the air body
(concave edges × shapes, partners × faces) and now sit at the passes
the ladder was built to watch: at 16 couplers the face pass (20 s
pooled, 49 s not) and the edge pass (17 s) are 55 % of the build.
The `before` singular / overlap / sheets figures are the DD-206
profile's, the ladder had not printed them.  The unit suite (2 514)
and the integration suite (398, the four GPU tests re-run on the
device) are green.

**Files:** `src/magnelio/geo/_occ_backend.py` (`PointClassifierSet`,
`extract_singular_edge_planes`, `check_pairwise_overlaps`),
`src/magnelio/mesh/_conformal.py` (`_probe_eps`, `detect_thin_metallizations`),
`tests/unit/test_point_classifier_set.py` (new), `tests/unit/test_geometry.py`
(`TestOverlapBatching`), `benchmarks/bench_mesh_build.py` (columns, results re-run).

## DD-207 — The side step of a degenerate plane clears the kernel tolerance; kernel sections run over the faces a plane can reach

**Status:** Decided and implemented 2026-08-28, branch `perf/section-shift-slab`.
DD-206's re-ranked item 1; closes KB-036.

**Problem.**  The face pass of the Lange ladder grew faster than the
cell count (areas µ 14.7 → 48.6 s for 8 → 16 couplers, pool off), and
STATUS had filed it as "every plane cuts every coupler — a per-plane
face prefilter".  Instrumenting the sections instead of guessing
(`investigations/mesh-build-bench/probe_face_sections.py` — internal
record) showed what the 2 148 kernel sections of the 16-coupler build
were: **every one** a plane the planar engine had declined for
*coplanar face within tolerance*, i.e. the DD-106 side steps `p ± δ`
of the degenerate planes (finger ends, lead ends, bond edges — about
ten per coupler along x), taken with δ = deflection = 6e-8 m on this
6 µm grid, below the engine's tolerance screen (2 × max B-Rep
tolerance = 2–3e-7).  Of them 184 per 8 couplers hit the pocketed air
body (1 062 faces at 8, 2 118 at 16) at 59 ms each — the kernel's
section prepares every sub-shape before it looks for intersections, so
the cost per plane is the body's face count, and planes ∝ n × faces ∝ n
is the superlinear term.

The same measurement exposed a correctness defect.  A step below the
kernel's tolerance does not leave the face: the section Boolean
reports the face the plane was meant to escape on *both* sides — the
air body at `p ± 6e-8` returned one and the same 8-vertex polygon on
either side of a finger's end wall (the 4-vertex answer appears from
6e-7 on), and a finger brick 60 nm or 200 nm outside its end face still
reported its full section (correct from 4e-7).  In the face pass the
spurious pocket opening on the outside fell to the model's conducting
background, so every face lying in a finger's end wall read fully
blocked (min = max) with a wall jump of zero — DD-106's conventions
had nothing to choose from.  The threshold is the kernel's own and
topological rather than a distance: a 12.6 µm pocket on an air body's
bottom face answered correctly at 60 nm, the same pocket in the
interior did not, and a lone brick is protected by the bounding-box
screen.  Any model whose smallest cell is below about 15 µm at scale 1
was exposed (KB-036).

**Decision.**

1. **The side step is the larger of the section deflection and four
   times the largest B-Rep tolerance in the model** (at least the
   kernel's confusion, 1e-7 scaled): `_SECTION_SHIFT_TOLERANCES = 4`,
   the tolerance read once per shape by the new `_FaceSlabIndex`.
   Four is the margin over the measured failure (wrong at 1.3
   tolerances, right from 2.7) and, since the engine's screen is two
   tolerances, it also puts every shifted plane past the exact planar
   engine — the shifted sections of a planar model are now answered
   exactly, in microseconds, instead of by the kernel.  DD-106's
   conventions (min / mean for the matrix channel, max and jump for
   the geometric one, one-sided at the hull) are untouched; only the
   distance of the sampling planes changes, and only where the
   deflection was below four tolerances — grids finer than about 60 µm
   at scale 1.  There the step is at most a tenth of the smallest cell
   (the Lange: 6e-7 on 6 µm), still far below any cell; a sheet
   thinner than twice the step would be missed by both sides — sheets
   below a micron at scale 1 are beyond the kernel's tolerance anyway
   and are the thin-sheet pipeline's business (DD-202).

2. **Kernel sections run over the faces whose slab reaches the plane.**
   `_FaceSlabIndex(shape)` holds the faces, their geometry-only boxes
   and the shape's largest tolerance; `restrict(axis, pos)` returns a
   compound of the candidate faces (the shape itself when every face
   qualifies).  `cross_section_polygons(..., slab=)` sections that
   compound at every rung of its nudge ladder; the planar engine owns
   the index of its shape and hands it to every delegation
   (`compute_face_material_areas`, `batch_cross_sections`, the pool
   sampler, and the pool workers build their own at deserialisation).
   The kernel only intersects sub-shapes of *different* arguments and
   adjacent candidates share their boundary `TopoDS_Edge`, so the
   contours are the solid's — measured bit-identical on 194/194 Lange
   planes and 80/80 post planes.  The `exact_at_faces` path keeps the
   whole shape.

**Refuted on the way.**  A per-plane *face* prefilter as STATUS
imagined it would have left the engine declines in place; the slab
compound alone takes the Lange air body's x-planes from 53 to 15 ms
(29 candidate faces still carry 480 edges — the z = h face holds
every pocket outline), the y/z-planes only 1.3–1.7× — the decisive
lever was the step, the compound is what remains for curved bodies
(posts across the row 4.7 → 0.5 ms; along the row every face is a
candidate and nothing changes).

**Measured.**  Lange 8, pool off, before/after (`faces/probe8.log`,
`faces/probe8_after.log`): kernel sections in the face pass 1 092 →
**0** (the 96 remaining are the thin-sheet sections, one per sheet),
build 35.7 → 24.6 s.  Full ladder (`benchmarks/bench_mesh_build.py
--family all --pool off auto forced --json`, CPU idle; before =
DD-206's ladder):

| family, n | cells | total before → after (auto) | pool off | areas µ | areas ε | classify | kernel sections |
|---|---|---|---|---|---|---|---|
| Lange 16 | 3.69 M | 66.2 → **54.9 s** | 94.5 → 54.9 | 20.4 → 8.4 | 5.3 → 5.3 | 1.2 | 2 148 → 192 |
| Lange 8 | 1.84 M | 34.9 → **23.9 s** | 34.9 → 23.9 | 14.7 → 3.5 | 2.3 → 2.3 | 0.6 | 1 092 → 96 |
| Lange 4 | 922 k | 14.5 → 11.0 | 14.6 → 11.0 | 5.1 → 1.5 | 1.1 → 1.1 | 0.3 | 564 → 48 |
| Lange 2 | 461 k | 6.6 → 5.3 | 6.6 → 5.3 | 2.0 → 0.7 | 0.5 → 0.5 | 0.1 | 300 → 24 |
| Lange 1 | 231 k | 3.1 → 2.7 | 3.2 → 2.7 | 0.9 → 0.4 | 0.3 → 0.3 | 0.1 | 168 → 12 |
| posts 240 | 385 k | 45.9 → **38.1 s** | 71.1 → 48.4 | 12.0 → 11.6 | 10.6 → 10.3 | 15.6 → 8.5 | 1 968 (pooled) |
| posts 60 | 97 k | 8.4 → 7.2 | 8.4 → 7.1 | 3.0 → 2.5 | 2.3 → 1.9 | 1.5 → 1.1 | 1 604 |
| array 8 × 8 | 664 k | 33.2 → 34.1 | 33.4 → 33.9 | 1.6 | 1.2 | 0.2 | 1 |
| array 4 × 4 | 222 k | 8.9 → 9.1 | 8.8 → 9.0 | 0.6 | 0.5 | 0.1 | 1 |

The Lange rows no longer admit the pool at all (the remaining kernel
sections are the thin-sheet ones, one per sheet) and the face pass
grows with the cell count again (areas µ 3.5 → 8.4 s for 8 → 16
couplers, cells 1.84 → 3.69 M); at 16 couplers the edge pass (26.7 s)
is now half the build.  The post rows gain the slab compound on the
sections across the row (pool off: kernel sections 54.9 → 32.0 s,
classification 15.5 → 8.5 s); pooled, the prefill's 20 s (worker
start-up, serialisation, the along-the-row sections every face
qualifies for) bound the row.  The array rows are unchanged within
noise.  Gallery plane pins (28 scripts) unchanged; unit suite 2 520
passed, integration 402 passed (the GPU tests on the device).

**Files:** `src/magnelio/geo/_occ_backend.py` (`_FaceSlabIndex`,
`_PlanarSectionEngine.slab`, `cross_section_polygons(slab=)`,
`compute_face_material_areas` step, `batch_cross_sections`,
`_sample_and_admit`, `_section_worker_init`/`_section_worker`,
`_SECTION_SHIFT_TOLERANCES`), `tests/unit/test_section_slab_index.py`
(new), `docs/methods/meshing-conformal.md`, `known-bugs.md` (KB-036),
`benchmarks/results/bench_mesh_build.json` (re-run).
