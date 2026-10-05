# Magnelio — Technical Specification

> **Reference document.** Read when working on architecture, numerics, or API design.
> For current project status see `STATUS.md`.
> This is the internal design reference; the user-facing API reference
> is the Sphinx documentation (`docs/api`).
>
> Last updated: 2026-09-29

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Key Data Structures](#2-key-data-structures)
3. [FIT Numerics](#3-fit-numerics)
4. [CPML Design](#4-cpml-design)
5. [2D Eigenmode Port Solver](#5-2d-eigenmode-port-solver)
6. [Mesh Generation (Grid Line Algorithm)](#6-mesh-generation-grid-line-algorithm)
7. [Backend Abstraction](#7-backend-abstraction)
8. [Public Python API](#8-public-python-api)
9. [AnalysisScatteringTD.run() — convenience parameters](#9-analysisscatteringtdrun--convenience-parameters)
10. [Testing Strategy](#10-testing-strategy)
11. [Implementation Order](#11-implementation-order)
12. [Open Questions](#12-open-questions)
13. [Verification](#13-verification)
14. [Implementation Status](#14-implementation-status)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / API Layer                        │
│          Project, Simulation, Material, Port, BoundaryCondition │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Geometry         │  │ Mesh             │  │ Materials        │
│ Subsystem        │  │ Subsystem        │  │ Subsystem        │
│                  │  │                  │  │                  │
│ CSG tree         │  │ Grid line gen.   │  │ Material         │
│ OCC backend      │  │ Yee staggering   │  │ dataclass        │
│ BBox queries     │  │ Material filling │  │ Library          │
│ Intersect        │  │ Quality checks   │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                    │
          └────────────┬───────┘
                       ▼
          ┌──────────────────────┐      ┌──────────────────────┐
          │ Operator Subsystem   │      │ Boundary Subsystem   │
          │                      │      │                      │
          │ Sparse curl C / C^T  │      │ PEC / PMC masks      │
          │ Diag M_eps, M_mu     │      │ CPML aux fields      │
          │ M_sigma              │      │ Periodic ghost cells │
          └──────────┬───────────┘      └──────────┬───────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │    Solver Subsystem    │
                       │                        │
                       │ Leapfrog loop (FIT-TD) │
                       │ Stability / Courant    │
                       │ 2D / 3D eigenmode      │
                       └────────────┬───────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
   ┌────────────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │ Postprocessing     │  │ I/O Subsystem   │  │ Backend          │
   │                    │  │                 │  │ Abstraction      │
   │ S-params (FFT)     │  │ HDF5 project    │  │                  │
   │ Field probes       │  │ VTK export      │  │ get_xp() → xp    │
   │ Far field (future) │  │                 │  │ numpy / cupy     │
   └────────────────────┘  └─────────────────┘  └──────────────────┘
```

### Subsystem Responsibilities

| Subsystem       | Responsibility |
|-----------------|----------------|
| User/API Layer  | `Project`, `Simulation`, `Material`, `Port`, `BoundaryCondition` — user-facing orchestration |
| Geometry        | CSG tree construction, OCC backend queries, bounding-box extraction, face intersections |
| Mesh            | Grid line generation from geometry, Yee-cell staggering, material-ID filling, quality checks |
| Operators       | Sparse curl C/C^T (eigenmode solver); Numba-fused + array-stencil kernels (FIT-TD); diagonal M_eps, M_mu, M_sigma |
| Boundaries      | PEC/PMC edge masks, CPML auxiliary field arrays, periodic ghost-cell index maps |
| Solver          | Leapfrog update loop, Courant stability check, 2D/3D eigenmode solve |
| Postprocessing  | S-parameter FFT + mode decomposition, time-domain field probes |
| I/O             | HDF5 project persistence (full state), VTK field snapshots for ParaView |
| Backend         | `get_xp()` / `set_backend()` — swap NumPy ↔ CuPy without API changes |

---

## 2. Key Data Structures

### 2.1 GridLines

```python
@dataclass
class GridLines:
    x: np.ndarray  # shape (Nx+1,), sorted, units: meters
    y: np.ndarray  # shape (Ny+1,)
    z: np.ndarray  # shape (Nz+1,)

    # Derived (not stored, computed on access):
    # dx[i] = x[i+1] - x[i],  len(dx) = Nx
    # dy[j] = y[j+1] - y[j],  len(dy) = Ny
    # dz[k] = z[k+1] - z[k],  len(dz) = Nz
    # Nx = len(x) - 1, Ny = len(y) - 1, Nz = len(z) - 1
```

### 2.2 Mesh

```python
@dataclass
class Mesh:
    grid: GridLines
    material_id: np.ndarray          # shape (Nx, Ny, Nz), dtype int32
                                     # material_id[i,j,k] → index into material_library
    material_library: dict[int, Material]
    pec_mask_edges: np.ndarray       # bool, shape (3, Nx*Ny*Nz)
                                     # axis 0: Ex-edges, axis 1: Ey-edges, axis 2: Ez-edges
    boundary_conditions: BoundaryConditions | dict
                                     # closure of the six bbox faces (DD-103);
                                     # declared on the GeometryModel / from_grid,
                                     # PEC faces already folded into pec_mask_edges
    planes: GridPlanes | None        # provenance of every grid plane (DD-200):
                                     # per axis position, sources (rule + shape),
                                     # node index, dropped / absorbed / unplaced;
                                     # from_geometry only, None on from_grid
```

### 2.3 Material

```python
@dataclass
class Material:
    name: str
    epsilon: tuple[float, float, float] = (1.0, 1.0, 1.0)  # relative (εrx, εry, εrz)
    mu:      tuple[float, float, float] = (1.0, 1.0, 1.0)  # relative (μrx, μry, μrz)
    sigma:   tuple[float, float, float] = (0.0, 0.0, 0.0)  # electric conductivity [S/m]
    sigma_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # magnetic loss [Ω/m]
    is_pec:  bool = False
    dispersion: DispersionModel | None = None  # pole-residue ε(ω), DD-083

    @classmethod
    def air(cls) -> "Material": ...       # ε=μ=1, σ=0
    @classmethod
    def vacuum(cls) -> "Material": ...    # alias for air
    @classmethod
    def pec(cls) -> "Material": ...       # is_pec=True
    @classmethod
    def lossy_metal(cls, name, sigma, mu=1.0) -> "Material": ...
        # is_pec=True + finite σ: PEC in the field solve, σ/μ consumed
        # by surface-loss models (R_s = √(ωμ₀μr/2σ)).  DD-081.
    @classmethod
    def dispersive(cls, name, model, mu=1.0, sigma=0.0) -> "Material": ...
        # epsilon = (model.eps_inf,)*3 (mass matrix + CFL); the pole set
        # runs as a trapezoidal ADE in the solver.  DD-083/DD-084.
```

Dispersive permittivity is the pole-residue form
`ε(ω) = ε_∞ + Σ_p r_p/(jω − a_p)` (`DispersionModel`, exported
top-level) with `debye` / `lorentz` / `drude` / `djordjevic_sarkar`
constructors and a mandatory passivity check at construction; see
DD-083/DD-084.

### 2.4 FieldState (Structure of Arrays)

Yee-staggered field components on the primary grid (E) and dual grid (H):

```python
@dataclass
class FieldState:
    # E fields — on primary grid edges
    Ex: Array  # shape (Nx,   Ny+1, Nz+1)
    Ey: Array  # shape (Nx+1, Ny,   Nz+1)
    Ez: Array  # shape (Nx+1, Ny+1, Nz  )

    # H fields — on dual grid face centers
    Hx: Array  # shape (Nx+1, Ny,   Nz  )
    Hy: Array  # shape (Nx,   Ny+1, Nz  )
    Hz: Array  # shape (Nx,   Ny,   Nz+1)
```

Rationale for SoA: see `design-decisions.md` DD-002.

### 2.5 Solver Configuration

> **Note:** The `SimulationConfig` dataclass originally planned here was never implemented.
> Solver parameters (`total_time_steps`, `dt`, `energy_stop_db`, `excite_port_id`, …) are
> runtime arguments of `Simulation.run()` (Section 9) and `FITTimeDomainSolver`.

### 2.6 MeshControl

```python
@dataclass
class MeshControl:
    min_nodes_per_wavelength: int = 20      # dimensionless — N_wl
    wavelength_rule: str = "local"          # "local" (per slab) | "global"
    singularity_refinement: float = 1.0     # h_fine / k at conductor-edge planes (1 = off)
    min_cells_per_feature: int = 4          # dimensionless — cells per smallest gap
    growth_factor: float = 1.3             # max ratio h_{i+1}/h_i > 1
    max_cell_size: float | None = None     # absolute cap [meters]
    min_cell_size: float | None = None     # absolute floor [meters]
    forced_planes: dict[str, list[float]] = field(default_factory=dict)
                                           # e.g. {"z": [0.0, 1.6e-3]}
    conformal: bool = True                 # conformal/Dey-Mittra treatment
    dey_mittra_eta: float = 0.4            # dimensionless stability cutoff
    min_feature_gap: float | None = None   # plane-clustering tol [m];
                                           # None -> 1e-5 x bbox diagonal (DD-120)
```

The CPML layer thickness is *not* here — it belongs to the boundary
closure (`BoundaryConditions.cpml_thickness_cells`, DD-103), which
drives both the grid extension and the runtime profile.

---

## 3. FIT Numerics

### 3.1 Leapfrog Update Equations (SI units)

Half-step leapfrog scheme (Yee):

```
E^{n+1} = M_eps_inv ⊗ (α_E · E^n + β_E · C^T · H^{n+1/2})
H^{n+3/2} = M_mu_inv  ⊗ (α_H · H^{n+1/2} - β_H · C   · E^{n+1})
```

Where:
- `C`        — sparse curl matrix, shape `(3·Ne, 3·Ne)`, `Ne = Nx·Ny·Nz`
- `M_eps`    — diagonal mass matrix: `ε₀ · εr · A_face / dl` per E-edge
- `M_mu`     — diagonal mass matrix: `μ₀ · μr · dl / A_face` per H-face
- `α_E`, `β_E`, `α_H`, `β_H` — update coefficients incorporating `M_sigma` and `dt`

For lossless media (σ = 0):
```
α_E = 1,   β_E = dt
α_H = 1,   β_H = dt
```

For lossy media (electric loss σ ≠ 0), using implicit E-field update:
```
α_E = (1 - σ·dt/(2ε)) / (1 + σ·dt/(2ε))
β_E = dt / (ε · (1 + σ·dt/(2ε)))
```

### 3.2 Courant Stability Condition

```
dt ≤ 0.99 / (c₀ · √(1/dx_min² + 1/dy_min² + 1/dz_min²))
```

Safety factors by accuracy level:
| `accuracy` | Courant factor |
|------------|----------------|
| `"draft"`  | 0.90           |
| `"normal"` | 0.95           |
| `"high"`   | 0.99           |

### 3.3 Simulation Duration

When `f_max` is given and `total_time_steps` is not:
```
T_sim = 10 / f_max         # 10 periods at f_max
total_time_steps = ceil(T_sim / dt)
```

### 3.4 Discrete Curl Matrix C

The curl matrix C maps E-field edges to H-field faces. It is assembled as a sparse matrix in
CSR format. For a 3D grid of size Nx×Ny×Nz:

- E-edge count: Ne = Nx·(Ny+1)·(Nz+1) + (Nx+1)·Ny·(Nz+1) + (Nx+1)·(Ny+1)·Nz
- C has shape (3·Nf, 3·Ne) where Nf = (Nx+1)·Ny·Nz + Nx·(Ny+1)·Nz + Nx·Ny·(Nz+1)
- Each row has exactly 2 non-zero entries (+1 and -1, i.e., Whitney 1-forms)
- Discrete Stokes theorem: `C · e + C^T · h = 0` for exact discrete forms

---

## 4. CPML Design

Decision: CPML over UPML — see `design-decisions.md` DD-001.

### 4.1 CPML Auxiliary Variables

For each CPML layer and each field component direction, auxiliary convolution variables:

```
ψ_{Ex,y},  ψ_{Ex,z}   (for Ex updates in y and z directions)
ψ_{Ey,x},  ψ_{Ey,z}
ψ_{Ez,x},  ψ_{Ez,y}
ψ_{Hx,y},  ψ_{Hx,z}
ψ_{Hy,x},  ψ_{Hy,z}
ψ_{Hz,x},  ψ_{Hz,y}
```

Storage: only cells within the CPML region (default 8 cells per active face).

### 4.2 CFS-PML Update Equations

Full CFS stretching function: `s(ω) = κ + σ / (α + jωε₀)`.

```
ψ^{n+1} = b · ψ^n + c · ΔH
b = exp(-(σ/κ + α) · dt / ε₀)
c = σ / (κσ + κ²α) · (1 − b)     (= 0 where σ = 0)
ck = 1 − 1/κ

E_correction = β_E · (ck · ΔH + ψ)
```

### 4.3 CFS-PML Profiles

Polynomial grading (ρ ∈ [0, 1] = normalised depth):
```
σ(ρ) = σ_max · ρ^m                  (m = 3)
κ(ρ) = 1 + (κ_max − 1) · ρ^m       (κ_max = 7)
α(ρ) = α_max · (1 − ρ)             (α_max = 0.02)
σ_max = -(m+1) · c₀ · ε₀ · ln(R_target) / (2 · d_phys)
```

Default: `BoundaryConditions.cpml_thickness_cells = 8`; 16 is a
common choice for waveguide port PML.

**PEC re-enforcement:** After all CPML E-corrections, PEC/PMC boundary
conditions are re-applied to prevent PEC-wall violations inside PML regions.

---

## 5. 2D Eigenmode Port Solver

### 5.1 Problem Setup

Given a rectangular cross-section on a bounding face (e.g., xmin-plane), cut the 3D mesh
at that plane to obtain a 2D Yee grid. Assemble the 2D curl-curl eigenvalue problem:

```
(∇_t × μr⁻¹ · ∇_t ×) · E_t = ω² · ε₀ · μ₀ · εr · E_t
```

Discretized using 2D discrete curl operators (sparse matrices).

### 5.2 Solver Paths

Three solver paths exist, applied depending on the mode type:

| Path | Formulation | Modes found | Design Decision |
|------|-------------|-------------|-----------------|
| **Hz dual EVP** | `A_H = C2 · M_ε⁻¹ · C2ᵀ`, `B_H = diag(M_μ_hz)` | TE (Hz ≠ 0) | DD-012 |
| **E_z scalar EVP** | `-∇·(μ⁻¹∇E_z) = ω_c²·ε·E_z`, Dirichlet E_z=0 on PEC | TM (E_z ≠ 0) | DD-026b |
| **2D Laplace** | `∇·(ε∇φ) = 0`, φ=1 on trace, φ=0 on boundary | TEM (quasi-static) | DD-020 |

`_solve_on_grid` runs both TE and TM EVPs, merges and sorts by f_cutoff (DD-026b).

```python
from scipy.sparse.linalg import eigsh

eigenvalues, eigenvectors = eigsh(A, M=B, k=n_modes, which='SM')
omega_n = np.sqrt(eigenvalues.real)
```

See `design-decisions.md` DD-007 for rationale.

### 5.3 Propagation Constant and Frequency-Dependent Impedance (DD-026b)

When `f_ref` is provided to the solver:

```
β² = (ω_ref² − ω_c²) · μ₀ · ε₀ · ε_eff   (evanescent if ≤ 0 → β = 0)

TE:  Z_wave = ω_ref · μ₀ / β
TM:  Z_wave = β / (ω_ref · ε₀ · ε_eff)
TEM: Z_wave = η = √(μ₀ / (ε₀ · ε_eff)),  β = ω_ref · √(μ₀ · ε₀ · ε_eff)
```

Without `f_ref` (default): `Z_wave = η`, `β = None`.

### 5.4 Characteristic Impedance Z_pi (DD-025)

For TEM/quasi-TEM modes, the power-current line impedance `Z_pi` is computed via voltage
integration between conductors after Poynting normalisation (P = 1 W):
`Z_pi = V² / P = V²`. TE/TM modes get `Z_pi = None`.

### 5.5 H-Field Profiles (DD-026a)

H_t is derived from E_t via the plane-wave impedance relation `H_t = (1/Z)(ẑ × E_t)`.
For TEM modes in inhomogeneous media, per-DOF local impedance
`η(i,j) = √(μ₀/(ε₀·ε_r(i,j)))` is used. Both E and H profiles are stored in `ModeResult`.

### 5.6 Mode Classification

| Mode type | Criterion |
|-----------|-----------|
| TEM       | Ez ≈ 0, Hz ≈ 0 (requires multi-conductor cross-section) |
| TE        | Ez ≈ 0, Hz ≠ 0 |
| TM        | Hz ≈ 0, Ez ≠ 0 |
| Hybrid    | Both Ez ≠ 0, Hz ≠ 0 (EH or HE modes) |

Classification threshold: `‖E_z‖ / ‖E_t‖ < 1e-6` for TE, analogous for TM.

### 5.7 Port Types

**WaveguidePort** (`port_waveguide.py`, ~2100 lines) — **primary port type**. General
waveguide port supporting all 6 domain faces. Features:
- Per-mode termination (DD-054 TEM, DD-055 TE/TM): numerical-path modes on
  certified-uniform feed chains use the exact discrete transparent boundary
  condition (`ports/modal/dtbc.py`; Klein-Gordon mass `q = ω̂_c·dt` from the
  2D eigenvalue of the 3D-restricted transversal operator, `q = 0` for TEM)
  with ghost-plane source injection, the discrete `λ^{1/2}` de-stagger, and —
  for dispersive modes — the exact discrete wave impedance
  (`dtbc_wave_impedance`) in `compute_s_parameters` (straight-line floors at
  the float-noise class).  Remaining modes (analytical-path, inhomogeneous
  QTEM until WP-R4): modal Mur-ABC (DD-027, supersedes DD-023), first-order
  Mur absorber on E-overlap with TF/SF source injection.
- TM 2D eigenproblem (DD-055): `build_2d_tm_curl_curl` — the index-sliced
  restriction of the 3D operators onto the port slab's normal-E edges
  (exact discrete cut-off; replaces the former lumped node-Laplace).
- Supports TEM, TE, and TM modes with frequency-dependent impedance.
- Waveform: plain Gaussian for TEM, modulated Gaussian for TE/TM (DD-022).
  DD-281 sizes both by upper-edge amplitude attenuation (25 dB by default),
  relative to DC or the carrier-centre spectrum including mirrored lobes.
  Envelope `exp(-((t-peak_time)/tau)^2)`, peak at `4.5*tau`, nominal end
  `9*tau`; resolved width and timing persist with backward reconstruction
  of legacy Gaussian records. The width solve runs once on construction.
- Integrated into the high-level API as declarative ports (`PortWaveguide`,
  declared on the model — DD-109); optionally windowed to a sub-rectangle
  of the face via `corners=` (world-coordinate corner pair, DD-153).

### 5.8 Port Integration

- Port excitation: inject mode profile as a soft source at E-tangential edges
  (WaveguidePort also injects H-field for modal ABC)
- Port monitoring: overlap integral with mode profile at each time step
- S-parameter extraction: FFT of time-domain port signals; post-hoc
  reference-plane shift via `result.deembed` on the exact discrete
  chain dispersion (DD-187)
- Waveform: Gaussian for TEM, modulated Gaussian for TE/TM (DD-022)

---

## 6. Mesh Generation (Grid Line Algorithm)

### 6.1 Algorithm

```
Input:  GeometryModel, MeshControl, f_max
Output: GridLines(x, y, z)

1. Extract critical planes from OCC geometry:
   - Face pass: axis-normal planar faces, tangent positions of
     axis-aligned cylinders and spheres, shape bounding-box extents
     (material planes)
   - Edge pass (DD-191): every B-rep edge lying flat in an axis-normal
     plane — chamfer/fillet onsets, loft sections, iris circles —
     excluding seam and degenerated edges (soft "feature" planes)
   - Result: sets Px, Py, Pz of material planes plus Fx, Fy, Fz of
     feature planes

2. For each axis independently:
   a. Sort critical planes
   a'. Merge the feature planes (DD-191): a feature plane within the
      clustering tolerance of a material plane is that plane; one
      closer than max(h_max / max_edge_refinement, min_cell_size) to
      any material plane or to a previously kept feature plane is
      dropped and reported (one warning per axis); the rest join as
      non-material planes.  Feature planes never move a material plane.
   b. Determine the two cell-size scales (DD-028):
      - h_max  = λ_slab / N_wl per axis interval (DD-192): the interval
        [p_i, p_{i+1}] is a slab of the domain, and λ_slab =
        c₀ / f_max / n_slab with n_slab = sqrt(εr·μr) of the densest
        material (background included) whose analytic bounding box
        reaches into the slab; wavelength_rule="global" uses the
        densest material of the whole model for every interval.
        The global λ_min / N_wl stays the reference for the edge
        floor, the h_fine sentinel and the undershoot check.
        Intervals too short for the full ramp keep h0 = h_fine and
        relax the growth ratio to g' ≤ g so the count fills the
        interval exactly (DD-193); the integer count never pushes the
        fine-end cell below h_fine.
        The fine size is per plane (DD-194): planes holding a
        singular conductor edge (convex edge of a metal shape, or a
        concave edge of a non-metal shape with metal in the open
        wedge; domain end planes excluded) take
        h_fine / singularity_refinement.  An interior interval whose
        ends differ grades from each end at its own size — both ramps
        plus a uniform middle when they fit, else a tent with the
        smaller size pinned and one ratio r ≤ g up and down, the
        coarse end free between the pinned size and 1.05 h_fine.
      - h_fine = min_gap / min_cells_per_feature
        where min_gap is the smallest interior gap on any axis;
        a feature plane contributes its adjacent interval widths with
        divisor 1 (one cell across the feature layer)
   c. For each interval [p_i, p_{i+1}]:
      - Boundary intervals: ramp from h_fine at the interior interface,
        grow by factor g = MeshControl.growth_factor toward h_max, then
        uniform fill with h_max toward the domain wall
      - Interior intervals: symmetric ramps from both ends to h_max,
        uniform middle (or full graded subdivision when too short)
      - Single-interval domain: uniform with h_max
      - Apply max_cell_size cap and min_cell_size floor if set
      - Cell counts are integers, so the count chosen against h_fine may
        overshoot it by _H_FINE_TOL = 5 % (DD-105).  h_fine is a
        convention, but refusing to overshoot it adds a whole cell and
        shrinks the interval's smallest cell, which bounds dt globally.
        Never applied to h_max — that is the user's accuracy choice.
   d. Merge subdivisions from all intervals

3. Insert forced_planes exactly (split any cell that contains a forced plane)

4. Append PML cells at each face the closure declares CPML:
   - cpml_thickness_cells cells with graded sizing (fine near domain, coarse at PML boundary)

4b. Pull the outermost grid line inside the bbox on each PMC face
   (WP-U0 stage 2: puts the natural magnetic wall ON the face)

5. Quality check:
   - grading undershoot: the globally smallest cell (the one bounding dt)
     more than 15 % under the h_fine its interval asked for; warn with the
     min_cell_size that removes it (DD-105).  Skipped for anchor pairs,
     intervals shorter than h_fine, cells already on min_cell_size, and
     axes whose fine size is wavelength- rather than feature-driven
   - neighbour cell-size ratio > 2 on any axis; warn (a grading mesher
     that stopped grading — not an accuracy limit, see DD-105)
   - total cell count estimate; warn if > 10^7
```

### 6.2 MeshControl Parameters

| Parameter                   | Type              | Default | Unit       | Description |
|-----------------------------|-------------------|---------|------------|-------------|
| `min_nodes_per_wavelength`  | `int`             | 20      | —          | Minimum cells per wavelength of the slab's densest material |
| `wavelength_rule`           | `str`             | "local" | —          | "local": per-slab wavelength; "global": densest material everywhere |
| `singularity_refinement`    | `float`           | 1.0     | —          | Grading at conductor-edge planes starts at h_fine / k (1 = off) |
| `min_cells_per_feature`     | `int`             | 4       | —          | Cells across the smallest geometry gap (0 disables) |
| `growth_factor`             | `float`           | 1.3     | —          | Max ratio h_{i+1}/h_i |
| `max_cell_size`             | `float \| None`   | None    | meters     | Absolute upper bound on cell size |
| `min_cell_size`             | `float \| None`   | None    | meters     | Absolute lower bound on cell size |
| `forced_planes`             | `dict[str, list]` | {}      | meters     | Exact grid lines to insert |
| `conformal`                 | `bool`            | True    | —          | Conformal/Dey-Mittra material treatment |
| `dey_mittra_eta`            | `float`           | 0.4     | —          | Stability cutoff for Dey-Mittra cells |
| `min_feature_gap`           | `float \| None`   | None    | meters     | Critical-plane clustering tolerance; `None` resolves to 1e-5 × the model bbox diagonal (DD-120) |

### 6.3 Internal Geometry Scaling (DD-120)

The public API is SI meters everywhere; the OCC kernel, however, has a
fixed model-unit precision (`Precision::Confusion()` = 1e-7).  Every
mesh build therefore chooses one **power-of-two scale factor** `s`
(`geo/_scaling.py`: `model_scale`, computed OCC-free from analytic
primitive bounding boxes) and threads it explicitly through
`_occ_shape(scale)` and every backend entry point.  Contract:

- **meters at every function boundary** — inputs are multiplied by `s`
  in bulk, outputs divided by `s` (lengths), `s**2` (areas), `s**3`
  (volumes); dimensionless outputs (`f_L`, property averages) pass
  through;
- `s = 1` inside the identity band (model diagonal within 1e-3..1e4 m),
  so meter/mm-scale models run the bit-identical legacy path;
- outside the band `s` brings the diagonal to O(128) scaled units,
  which lifts the former 100-nm feature limit (the effective limit is
  `1e-7 / s` meters) and keeps micron-scale geometry far from the
  kernel tolerance;
- power-of-two scaling is lossless in IEEE-754, so coordinates
  round-trip bit-exactly.

Certificates: `validation/scale_invariance_certificate.py` (S-parameters
invariant over six decades of geometric scale),
`validation/fiber_micron_regression.py` (micron fiber meshes with all
feature planes intact).

---

## 7. Backend Abstraction

All numerical modules use `xp = get_xp()` instead of `import numpy as np` directly.
GPU acceleration is activated via `set_backend('cupy')` before solver creation.

```python
from magnelio._backend.array_api import set_backend
set_backend('cupy')          # all subsequent allocations go to GPU
# … create solver, run simulation …
set_backend('numpy')         # revert to CPU
```

### Solver kernel dispatch (three tiers)

| Priority | Path | Backend | Temp buffers | Description |
|----------|------|---------|-------------|-------------|
| 1 | `update_E_fused_cuda` / `update_H_fused_cuda` | CuPy GPU | None | CUDA RawModule kernels, one thread per grid point |
| 2 | `update_E_fused` / `update_H_fused` | Numba CPU | None | Single-pass JIT kernels, `parallel=True` |
| 3 | `update_E_stencil` / `update_H_stencil` | CuPy GPU or NumPy fallback | 6 curl buffers | Array slice ops (`+=`, `-=`, `[:]=`) |

The solver detects the active backend in `setup()` and picks the fastest available path.
Numba kernels cannot operate on CuPy arrays, so GPU always uses the stencil path.
All paths produce bitwise-identical results (verified on random fields).

### GPU-ready components

| Component | GPU-compatible | Notes |
|-----------|---------------|-------|
| FieldState (`field_arrays.py`) | ✅ | `_xp` attribute, flat arrays on device |
| Material coefficients | ✅ | Computed on CPU, transferred in `setup()` |
| PEC boundary | ✅ | Slice zeroing works on CuPy |
| PEC integer index (`e[pec_idx] = 0`) | ✅ | Index array transferred to GPU |
| Energy monitoring (`@` operator) | ✅ | Returns Python float via `float()` |
| CPML | ⚠️ Functional | Aux arrays on CPU, implicit transfers per step |
| WaveguidePort update_bc | ⚠️ Functional | Fancy indexing works on CuPy, not optimised |
| Monitors / Recorder | ⚠️ Functional | Read-only access, minimal transfer |

See `design-decisions.md` DD-006.

---

## 8. Public Python API

The public API is organised SciPy-style along one axis — the domain
(DD-117):

| Tier | Import path | Contents |
|------|-------------|----------|
| **Core** | `magnelio` (14 names) | The model container and run vocabulary (`GeometryModel`, `Material`, `Mesh`/`MeshControl`, `BoundaryConditions`, `Excitation`), the problem classes (`AnalysisTD`, `AnalysisScatteringTD`, `AnalysisEigenmode`), the store entry points (`open_project`, `resume`), verbosity controls (`set_verbosity`, `get_verbosity`), and `__version__`. |
| **Domain namespaces** | `magnelio.<domain>` | One namespace per subject area: `geo` (dimensional geometry, owned topology, primitives, CSG and transforms), `materials` (dispersion, roughness, impedance fits), `mesh` (`GridLines`, `BoxFace`), `boundaries` (BC classes), `ports` (declarative `Port*` and `PortSpec*` families, conductor specs, `Mode`/`ModeType`, reports), `sources`, `monitors` (`Monitor*` families), `circuit` (`SeriesRLC`/`ParallelRLC`, `EdgePath`, curve rasteriser), `signals`, `fields` (public field containers and recordings), `solver`, `analysis` (analyses' result contracts and memory estimates), `post` (post-processing), `plots`, `io`, `constants`. Reserved (DD-224, not yet shipped): `particles` (species, emission models), `optimize` (sweeps/optimisers). |
| Internals | underscore modules; names outside `__all__` | No stability guarantee (`magnelio._operators`, `magnelio.ports._modal`, …; plumbing such as port builders/operators, the V/I recorder and `MonitorRegion` is importable but not part of the documented surface). |

Placement rule: the core holds the model container, run vocabulary and
problem classes; every other public name lives in exactly one domain
namespace; plumbing is not exported.  Every public name has exactly
one documented home (`validation/tools/check_api_surface.py` enforces
this, including the pinned core surface).

`circuit.integrate_E(field, curve, grid)` integrates physical E samples
in V/m along the canonical directed edge path (DD-284). Real fields
retain the existing Python-float accumulation and return type; complex
frequency frames return a Python complex voltage, without conjugation.
The supplied grid belongs to the frame, including monitor subregions.
It is a path integral and need not be an endpoint potential difference.

DD-279 records the implemented naming revision: explicit physical units,
coordinates, normalization, selectors, operation scope and public type homes.
The version-neutral `docs/migration-api-naming.md` lists every breaking
mapping;
stored physical meaning is preserved through explicit recipe/reader mapping.

### 8.1 High-level example — `AnalysisScatteringTD`

S-parameters of a two-port rectangular waveguide, in the canonical
example style of the tutorial gallery (`examples/tutorials/`,
`import magnelio as mio` plus the `geo`/`ports` namespaces):

```python
import magnelio as mio
from magnelio import geo, ports

a, b, L = 22.86e-3, 10.16e-3, 40.0e-3
f_max, n_modes = 25.0e9, 5

model = mio.GeometryModel()
model.add(geo.Brick(origin=(0.0, 0.0, 0.0), size=(a, b, L),
                    material=mio.Material.from_isotropic(name="air", epsilon=1.0)))
model.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=n_modes))
model.add_port(ports.PortWaveguide(name="port2", plane="zmax", n_modes=n_modes))

mesh = mio.Mesh.from_geometry(
    model, mio.MeshControl(min_nodes_per_wavelength=15), f_max=f_max,
)

analysis = mio.AnalysisScatteringTD(mesh=mesh, f_max=f_max)
report = analysis.solve_ports(["port1"])["port1"]        # optional pre-check
result = analysis.run(excited=[("port1", m) for m in range(n_modes)])

s21 = result.S("port2", "port1", mode_out=0, mode_in=0)
result.plot_s()
result.to_touchstone("wr90")                    # -> .sNp over excited channels
```

Ports are declared on the model **before meshing** (DD-109) so the
mesher buffers exactly the faces that carry one; the mesh hands the
declarations to the analysis.  An explicit `ports=` on the analysis
overrides the mesh declarations (and may mix declarative ports with
`PortSpec*` objects from `magnelio.ports`).

DD-280 adds a port-name selection to `solve_ports`, a bare string for a
single mode-0 excitation in `run`, and shared magnitude/phase and time-wave
plotting on in-memory and stored scattering results. Viewer settings are
presentation-only: boundary and symmetry overlays start hidden, body
visibility is independent of cuts, and group paths are optional additive
project geometry metadata. Arrow sampling stays isotropic and independent
of mesh grading, with separate length, width, colour and threshold controls.

### 8.2 Domain example — custom port setup via specs

The `PortSpec*` family is the supported custom-assembly tier: a spec
carries the full port description and is handed to an analysis via
`ports=` (mixing with declarative ports is allowed):

```python
from magnelio import AnalysisScatteringTD
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecRectWG

spec = PortSpecRectWG(name="p1", plane=BoxFace.Z_MIN,
                      width=22.86e-3, height=10.16e-3, n_modes=1)
analysis = AnalysisScatteringTD(mesh=mesh, f_max=f_max, ports=[spec])
```

The builders, runtime operators and the V/I recorder behind the specs
are internal since DD-117 (importable, no stability guarantee).

### 8.3 Geometry ontology and construction contract

DD-275 fixes the geometry-foundation contract.  It is an accepted staged
contract: WP1 implements the dimensional hierarchy, affine values,
member-wise Group placement, and Solid-only CSG; WP2 implements exact curves
and validated planar Profile factories, including intrinsic holes. The
migration boundary is characterized in `tests/unit/test_geo_api_baseline.py`;
WP3 implements owned topology, semantic/named selection, history and project
replay. WP4 implements the uniform operations below; WP5 implements relative
routes and transported poses. A name listed here must not be
documented as shipped until its work package lands.

Standalone geometry is dimensional:

```text
Shape
|-- Curve
|-- Sheet
|   |-- Profile       planar bounded sheet, outer wire plus holes
|   `-- Surface       curved sheet
`-- Solid             closed 3-D body
```

The existing volume primitives (`Brick`, `Sphere`, `Cylinder`, `Cone`,
`Torus`, `ImportedSolid`) and Boolean results are `Solid` values. `ImportedSheet`
wraps an exact free CAD face without making it directly meshable. `Curve`,
`Sheet`, `Profile`, `Surface`, and `Solid` are public categories. The
axis-normal polygon `Face` and `Curve.covered()` have been removed in favour
of the fixed `Profile` factories.  `FaceRef` names owned topology.  `Group` is a transformable,
material-preserving aggregate but not a `Shape` or CSG operand.  `ThinWire`
is an EM mesh declaration around a `Curve`, not a standalone CAD category.

Solid primitive signatures remain as follows unless a later, explicit DD
amends one:

| Class | Constructor arguments |
| --- | --- |
| `Brick` | `origin, size, material=None, name=None`; also `from_corners` and `from_ranges` |
| `Sphere` | `center, radius, material=None, name=None` |
| `Cylinder` | `origin, radius, height, material=None, axis="z", inner_radius=0.0, angle_deg=None, name=None` |
| `Cone` | `origin, bottom_radius, top_radius, height, material=None, axis="z", name=None` |
| `Torus` | `center, major_radius, minor_radius, material=None, axis="z", name=None` |

Every axis or normal accepts `"x"`, `"y"`, `"z"`, or a non-zero
three-vector.  Geometry is always expressed in world coordinates; there is
no ambient local coordinate-system state.

Curve construction is `line`, `circle`, `ellipse`, `polyline`, `arc`,
`ellipse_arc`, `spline`, `helix`, and `joined`.  Planar construction is:

```python
Profile.polygon(points, *, material=None, name=None)
Profile.rectangle(center, size, *, normal="z", x_direction=None,
                  material=None, name=None)
Profile.circle(center, radius, *, normal="z", material=None, name=None)
Profile.from_wires(outer, holes=(), *, material=None, name=None)
```

Profile polygon points are 3-D and coplanar.  `from_wires` validates closed,
coplanar, nested, non-intersecting wires; holes are profile topology and do
not require solid Boolean scaffolding.  `Curve.covered()` is not part of the
new contract. `Curve.length` and `Profile.area` are read-only CAD measurements.
`Profile.boundary()` returns detached Curve values in world placement: the
outer wire followed by holes in input order, preserved by affine transforms.
All four profile-to-solid operations retain holes. Loft sections require equal
hole counts, match inner boundaries by input order, and inherit an omitted
material from the first Profile. Topology-changing holes are not supported.

### 8.4 Transform algebra and category-specific operations

Every standalone `Shape` provides exactly the common affine verbs
`translated`, `rotated`, `mirrored`, and `scaled`, plus `bounding_box()`.
By default the verbs return one value of the same dimensional category and
preserve material, name, and named topology. `translated` and `rotated` also
provide `repeat=1`, `copy=False`, `unite=False`, and `group=False`; `mirrored`
provides the last three options without repetition. `repeat` counts transformed
copies at successive displacement/angle increments, and `copy=True` includes
the original first. Without aggregation a multi-value result is a list;
`group=True` always returns a material-preserving Group, and `unite=True`
always returns a Union, restricted to Solid input. Group and union modes are
mutually exclusive, and mirrored aggregation requires `copy=True`. Repeated
Group assemblies retain their nested members and reject fusion. `scaled`
has no array options. Every copy is a direct affine placement of the source.

Reusable placement uses immutable values:

```python
Translation(vector)
Rotation(axis, angle_deg, origin=(0, 0, 0))
Mirror(normal, position=0)
Scale(factor, center=(0, 0, 0))
Transform.identity()
```

For column-vector points, `A @ B @ shape` applies `B` first and `A` second.
`A @ B` composes to `Transform`; a transform acts on a `Shape` or `Group`.
The reverse `shape @ transform` is undefined.  Scale is uniform.  Named
methods delegate to the same algebra.  `+` and `-` are never vector
placement.

CSG operators are `Solid`-only:

| Spelling | Meaning |
| --- | --- |
| `a + b` / `Union(*solids, ...)` | Boolean union |
| `a & b` / `Intersection(a, b, ...)` | Boolean intersection |
| `a - b` / `Difference(base, *tools, ...)` | Boolean difference |
| `Group(*geometry, name=None)` | Non-CSG aggregate; transforms distribute |

Profile-consuming operations accept standalone Profile, eligible Sheet or
FaceRef inputs and return independent Solid geometry. Extrusion and thickening
also accept curved sections; revolution, sweep and loft require planarity.
Geometrically flat spline sheets are re-covered from exact boundaries. All
operations preserve holes; Loft accepts mixed planar sections and closed Curve
conveniences. Direct FaceRef construction keeps owner registrations on that
original owner. Two-section tangent lofts construct corresponding hole tools
under the same full-section centroid/normal conditions. Geometry volume
measurement uses span-aware adaptive Gauss-Kronrod integration on a centred
copy scaled to a 128-unit diagonal with a power-of-two factor (DD-285).
Dividing by that factor cubed restores the built shape's units. Construction
and mesher topology remain unchanged; composed/placed results and project
read-back of rebuilt rational tangent surfaces use the same measurement.

Chamfer and fillet accept EdgeRef/EdgeSetRef inputs through `edges=` or
FaceRef/FaceSetRef boundaries through `faces=`. Shell accepts face refs through
`openings=`. Each also accepts sequences, deduplicated; refs must belong to the
exact receiver. Point conveniences remain, selecting semantic refs immediately
and rejecting ties. Conflicting modes, wrong kinds and empty selections fail at
the public call. Selected modifications build at the refs' resolved scale and
rescale the result/names exactly, preserving membership at other model scales.
Replay retains immutable reference origins; connected set members validate
unique exact BREP snapshot equality, never persistent topology indices or nearest
retargeting.

Sweep alignment transports the real section boundary by the shortest oriented
normal-to-start-tangent rotation, preserving in-plane roll for aligned sections.
Antiparallel normals use the plane X direction as a half-turn axis; subsequent
transport defaults to corrected Frenet. `swept(frame=...)` also accepts `frenet`,
`fixed` and `fixed_binormal`. Fixed world sections remain parallel to their
initial alignment; fixed binormal retains their angular relation to the supplied
world `binormal`, and may yield oblique sections on spatial paths. That direction
is required only for fixed binormal and cannot be parallel to the initial
tangent. Non-default frames use MakePipeShell with every outer/inner boundary
pinned to the same spine start vertex; hole solids are subtracted. The default
MakePipe route is retained. Planar thickening forward follows the
oriented sheet/face normal. Path routing poses are independent of this pipe
orientation. Nonzero `twist_deg` adds right-hand roll about the transported
section normal, uniformly over true arc length; `draft_deg` defines signed
planar offset `s*tan(angle)`, growing the exterior and shrinking holes for
positive angles. Polygon offsets use intersection joins. Roll preserves fixed
and oblique section planes; the draft angle is a spatial wall angle only on a
straight perpendicular route. Sections retain their boundary topology.
Law sweeps normalize CAD units, transport native compatible frames, integrate
and invert arc length with controlled accuracy, and fit isoparametric spline
sections separately over tangent-connected edge spans. Explicit correspondence
retains roll. Elliptic normal offsets preserve their native curve parameters
through a bounded spline representation. Quarter/middle/three-quarter stations and nine points per edge
in both distance directions govern adaptive refinement; `tolerance` is an
absolute sampled-fit target, default profile diagonal times 1e-6, not a global
surface-error certificate. Unattainable fits raise. Compatible closed sections
are sewn at the seam; nonperiodic laws and sharp route joints are rejected.
Boundary solids are fused over edge spans and holes subtracted. Zero laws keep
the previous pipe route. Extra sweep validity/self-intersection diagnostics
were explicitly excluded. The advanced-CAD programme remains open. An explicit
`material=` wins; otherwise a Profile supplies its material and a `FaceRef`
supplies its owner Solid's material.  Construction solids without material
remain valid Boolean tools but cannot enter `GeometryModel` directly.

WP6.4 adds `partition(cutter=None, *, normal=None, position=None)` and
`section(cutter=None, *, normal=None, position=None, filled=False)` on Solid
and Sheet. A cutter is a Solid or Sheet; a plane uses a unit world normal and
signed metre offset. Partition returns an independent connected Solid or
Sheet for each source region in one kernel evaluation, retaining source
material and dimension. No cut, tangency and coincident boundaries return an
independent unchanged region; pre-existing disconnected regions remain
separate. Result order is not a persistent identifier. Ordinary section
returns exact connected Curve wires, empty if there is no one-dimensional
intersection. Explicit `filled=True` requires a Solid and a world plane and
returns planar Profile regions with holes. Shared face regions raise rather
than silently taking their boundary; point tangency gives no curve. Named
source topology follows global Splitter history and then assigns successors
to owned output regions. Singular splits fail, deliberate sets survive,
unrelated new cut faces remain unnamed. Named result replay identifies the
chosen region by exact BREP matching of reconstructed partition results,
without numeric kernel indices. Gate: `tests/unit/test_geo_partition_foundation.py`.

WP6.5 adds `Solid.imprinted(cutter)` and
`geo.insert(*bodies, priorities=..., voids=...)`. Imprint is directed: only
the receiver's boundary faces split at intersection curves with a Solid or
Sheet; its volume and material remain unchanged, and the cutter remains
independent. OCC Section supplies edges, and SplitShape assigns them to the
receiver faces. A section with no boundary edges leaves an independent
receiver. Kernel validity and volume conservation are checked. Named receiver
topology follows SplitShape history; cutter names do not transfer.

Insert takes material-bearing Solid bodies and an explicit integer rank for
each. Larger ranks win; an actual equal-rank overlap raises. Pairwise volume
overlap detection uses every positive kernel volume and is strict about kernel
failure. For each losing body, one
N-ary Difference removes every overlapping higher-rank body and every
overlapping material-less void tool. A void never appears in the output.
Same-material overlap is trimmed, while mere contact remains separate.
Completely removed untagged bodies are omitted. The returned Group preserves
materials and can be flattened into a GeometryModel in any insertion order.
The internal cut-result subtype retains the Difference operand route for
meshing but evolves only the retained body's named topology. Imprinted and
inserted named results replay through construction recipes, with an exact BREP
check for the imprinted body. Gate: `tests/unit/test_geo_imprint_insert.py`.

WP6.6 adds `Curve.projected_onto(target, *, direction=None,
perspective_source=None, closest=False, clip=False, all_hits=False,
tolerance=None)`. The target is a bounded Sheet or FaceRef in world placement;
the output is a tuple of standalone Curves. Exactly one projection policy is
required. Parallel and perspective rays use the first forward hit by default,
or all forward branches with `all_hits=True`; a source crossing a target face
is split before rank selection. A partial ray image raises unless `clip=True`
requests its trimmed pieces. A whole miss under clipping gives an empty tuple.
A coincident ray has a zero-distance first image; its continuum of all hits
raises. Closest-point projection minimizes distance over the actual trimmed
face, including boundaries. The trace is adaptively fitted on the surface with
an absolute metre `tolerance` (default source-box diagonal times 1e-6, with a
scale-aware kernel floor). Exact boundary edges are reused when the closest
trace follows a trim; a post-fit Common checks that the result stays within
the face and seeds local refinement at missed trim crossings. Discontinuous
nearest assignments and surface parameter singularities raise; `clip` and
`all_hits` are invalid in closest mode. An independent projected Curve can
feed ordinary geometry construction, and the resulting body can be persisted.
Gate:
`tests/unit/test_geo_projection_foundation.py`.

WP6.7 adds independent offsets in three distinct meanings. A directed planar
`Curve.offset(distance, *, normal)` requires an explicit oriented plane and
returns a tuple of Curves. Positive is left of traversal, negative right;
outer corners are circular, inner corners intersect, and open ends have no
caps. A collapsed curve gives an empty tuple; a curvature cusp or invalid
contour raises. Model-scale normalization keeps the CAD offset stable for
small as well as large geometry. `Profile.offset(distance)` offsets its
material region independent of boundary winding: positive expands the
exterior and shrinks holes, negative erodes the exterior and grows holes.
Every disconnected surviving region is returned as an independent Profile
with inherited material. Complete collapse gives an empty tuple; an exact
zero-width pinch can raise for invalid topology. Round clearance joins here
are distinct from the sweep draft's intersection joins.

Curved `Sheet.offset(distance, *, tolerance=None)` moves the actual bounded
face along its oriented normal and returns an independent zero-thickness
Surface in a tuple. The signed distance selects side. The default absolute
metre tolerance is the larger of one millionth of the source extent and the
kernel-resolution distance at the model scale. The kernel result must be a
single valid face with no reported self-intersection; sampled derivatives
must preserve local orientation and sampled normal displacement must fit
the budget. This gate is not a global surface-error certificate. A curved
owned face is detached before offsetting. Offset geometry used in a stored
body replays from its exact BREP snapshot. Gate:
`tests/unit/test_geo_offsets_foundation.py`.

WP6.8 adds the reusable `geo.Bend(target, origin=..., along=...,
across=..., u=..., v=..., max_strain=..., tolerance=None)` value. Applying
`bend @ Solid`, `bend @ Sheet`, or `bend @ Group` uses one explicit world
source frame for all members. Source coordinates `(u, v, w)` map inside the
interval to `S(u, v) + w n(u, v)`, where the one-face target's bounded CAD
parameter chart corresponds linearly to the declared `u` and `v` intervals.
The source before the interval remains fixed; after it a single rigid end
frame is carried tangentially. Both target interval edges must match rigid
transverse lines within tolerance; the first must match the unchanged source
pose. A nondevelopable target may strain the neutral plane. The required
`max_strain` limits sampled principal in-plane stretches, while sampled
volume Jacobians must stay positive across the occupied normal extent.

The CAD builder imprints interval seams, copies the source topology, fits
smooth surfaces on deformed faces with an adaptive sampled error gate,
reassigns every shared edge's pcurves (including periodic seam pairs),
rebuilds 3-D edge curves, and checks CAD validity and self-interference.
Fully unchanged and rigidly carried faces retain exact transformed surfaces.
The default absolute fit budget is the larger of one millionth of the
source/target extent and kernel resolution at the model scale. This is a
sampled fit gate, not a global Hausdorff, strain or injectivity certificate.
Named source topology follows split/copy histories; singular splits fail.
Named project replay rebuilds the map and checks geometric equivalence by
bounding boxes and symmetric Boolean differences. Physical materials stay
with their independent members. Gate: `tests/unit/test_geo_bend_foundation.py`.

WP6.9 uses the existing planar face-to-face `blend="tangent"` as the G1
transition for adjoining walls whose tangent follows the selected face's
outward extrusion direction. Its Hermite end rows match wall normals at
both ends; no G2 curvature condition is claimed. The asymmetric two-end
normal gate is in `tests/unit/test_geo_wrap_foundation.py`. Other adjoining
wall tangent fields are not inferred from the selected caps.

`geo.Wrap` accepts the same one-face target chart, explicit source frame,
source `u`/`v` extents, required `max_strain`, and optional sampled CAD fit
budget as Bend. It maps the entire source by `F(u,v,w)=S(u,v)+w*n(u,v)` with
no rigid exterior segments or boundary pose conditions. The whole source
bounding box must fit the declared chart. A single chart cannot cross an
unselected periodic seam. Group mapping preserves individual materials;
openings and named topology follow the deformed BREP and project replay.
Sampled principal strain, occupied thickness Jacobian, CAD fit, topology
and self-interference checks have the same limits as WP6.8. The source may
change length and volume. Gate: `tests/unit/test_geo_wrap_foundation.py`.

#### Relative routing poses (WP5)

`Path.from_pose(point, tangent, up)` stores an immutable orthonormal pose;
`Path.from_face(face_ref, up=...)` requires a planar FaceRef and starts at its
area centroid along its outward normal. Up is projected perpendicular to
tangent; zero or parallel directions fail. Read-only `current`, `tangent`
and `up` expose world point and unit directions. Path direction parsing first
prescales finite vectors by their largest component, avoiding norm overflow or
underflow for dimensionless direction magnitudes. An absolute-only `Path(start)`
has no initial tangent or up; absolute segments acquire tangent but no implicit
roll reference. `forward`/plane intersection require tangent, all turns require up.

`forward(distance)` requires positive length. Left turns rotate about up,
right turns about -up; left is `up x tangent`. Radius is positive and angle
is strictly between 0 and 360 degrees. `turn_to(direction, radius=...)` builds
the shortest circular bend and rigidly transports up; aligned targets are
no-ops and opposite targets raise for an undetermined plane. Explicit left/right
half-turns determine a reversal. Exact Curve arcs carry the resulting route.

`straight_to_plane(normal, position)` uses `unit_normal dot point = position`,
with signed offset in meters. It continues along tangent; an already reached
plane is a no-op within eight floating-point ULPs of the projected coordinates.
Parallel rays and intersections behind the route fail. Absolute line, circular,
elliptical and spline steps retain their existing geometry grammar and update
the endpoint pose from actual CAD derivatives at the segment's model scale.
At a corner, up follows the shortest tangent rotation; an antiparallel corner
uses up as the half-turn axis. Along smooth segments, rotation-minimizing
transport integrates `du/dq = -t * (u dot dt/dq)` with adaptive DOP853
(`rtol=1e-10`, `atol=1e-12`, maximum step one-sixteenth of the parameter range),
then re-orthonormalizes the endpoint. Circular relative turns use their exact
rigid rotation. No torsion-derived Frenet frame or world-axis reset is used
for routing. Zero-tangent degeneracy fails rather than inventing a pose.

Final Curve values contain geometry, not a routing frame; pipe orientation
defaults to corrected Frenet and can be selected independently with `frame=`.
Project round trips retain
the resulting geometry and named faces; a new Path may start from those faces
after read-back, without a Path-specific store codec. Focused gate:
`tests/unit/test_geo_paths_foundation.py`.

### 8.5 Owned topology, selectors and persistence

Owned topology is a separate public hierarchy:

```text
TopologyRef
|-- VertexRef
|-- EdgeRef
|-- FaceRef
|-- EdgeSetRef
`-- FaceSetRef
```

A ref strongly owns its immutable `Solid` context, caches resolution at that
owner's model scale, and cannot transform, enter a model, or act as a CSG
operand.  `EdgeRef.as_curve()` and `FaceRef.detached()` explicitly produce
standalone geometry; the latter returns `Profile` for planar and `Surface`
for curved topology.

Selectors have named and semantic forms: `solid.face("port")` retrieves a
name, while `solid.face(near=..., normal=..., surface_type=...)` selects once.
Edges and vertices use `edge` / `vertex`; `faces(...)` and `edges(...)` are
the explicit plurals returning `FaceSetRef` and `EdgeSetRef`.  Persistent
registration is immutable through `tagged_face`, `tagged_faces`, `tagged_edge`,
`tagged_edges`, and `tagged_vertex`.  Public numeric OCC indices are forbidden.

Semantic constraints filter first; nearest distance is then measured to the
complete subshape.  Zero candidates raises `TopologySelectionError`.  A tie
within the owner-scale OCC tolerance raises `AmbiguousTopologyError`; kernel
enumeration order never decides.  Surface types use `plane`, `cylinder`,
`cone`, `sphere`, `torus`, `bspline`, and `other` (with analogous analytic
curve types).

Affine transforms preserve named refs exactly.  A topology-changing
operation maps them only through OCC `Modified` / `Generated` / `IsDeleted`
history.  A unique successor survives; deletion, an unprovable successor, or
a singular one-to-many split raises `TopologyEvolutionError` during the public
construction call. A deliberately named set may split or merge after duplicate
removal, but deletion of any selected member still raises. Conflicting names
from different operands raise unless they prove the same selection.  The project store persists semantic origin and
the construction-history path, never raw topology indices, and validates the
same cardinality when rebuilding. The additive, versioned
`geometry.json` recipe retains a DAG of named construction branches and scaled
BREP snapshots at untagged origins; operation inputs carry their material and
metadata. Snapshot geometry stays at its saved kernel scale to avoid degrading
nanometre topology on conversion to meter space. Legacy untagged projects keep
the final-BREP reader. No operation is reconstructed through arbitrary imports.

Named builds capture kernel histories in a scoped ContextVar, isolated from
nested source evaluation; per-scale caches retain resolved names and release
builders afterwards. Untagged geometry keeps the existing lazy/fast paths;
tagged Union/Difference use history-producing N-ary operations rather than
planar fusion or pre-fused tools whose source histories would be lost.
Topology inventories, owner scale and bounding boxes are cached per owner and
scale. A nearest query screens conservative bounds before exact distances.
Normal constraints on curved faces require `near` and use the closest face
point; without `near`, they filter planar faces only. Reflection normals use
surface derivatives, including indirect analytic plane frames. Geometric planarity uses
`GeomLib_IsPlanarSurface` at kernel tolerance, including flat spline patches;
planar non-plane faces detach by covering their exact wires with a plane.

The shipped history adapter covers affine, extrusion, revolution, sweep, loft,
Boolean, fillet/chamfer and shell kernel steps. Only Solid owners can register
names; WP4 implements direct FaceRef-consuming operations. A wire-based loft may
have no provable face successor, while shell offsets may split an original face
into outer and inner faces. Such evolution follows the failure rules, not
geometric rematching. `tests/unit/test_geo_topology_foundation.py` is the gate.

CAD exchange (WP6.12) operates on selected Solid and Sheet leaves of a Group.
STEP import walks the placed assembly tree and classifies free faces without
revisiting any solid's boundary faces. Open shell faces are independent sheets;
unsupported standalone curves/points are reported. `ImportedSheet` retains
exact trimmed BREP geometry and may supply a material to a thickened solid,
but remains unmeshable as a zero-thickness sheet. STEP transfer and healing
run in millimetres before complete bodies are scaled to meter coordinates;
direct transfer to meters can destroy nanometre solids at the kernel's
absolute sewing tolerance. STEP export records selected
geometry in an explicit unit with names and display colours. BREP export
requires the unit supplied to its reader later. Neither CAD format contains
material physics, named-selection replay or construction history. Project
storage remains the path for those. Both CAD exporters use temporary output
files and require `overwrite=True` to replace an existing target. Gate:
`tests/unit/test_import_cad.py` and the executed Tutorial 14.

Intrinsic reference measurements are read-only properties:
`VertexRef.point`; `EdgeRef.length`, `start`, `end`, `vertices`;
`FaceRef.centroid`, `area`, `is_planar`, `edges`, `vertices`, and the constant
planar `normal`.  Parameterized or value-producing queries are methods,
including `normal_at(point)`, `detached()`, `as_curve()`, `bounding_box()`,
and `volume()`.  Reading `.normal` on a curved face raises and names
`normal_at`.

Category mismatches raise `TypeError`; invalid or underdetermined geometry
raises `ValueError`; selection absence and ties use the public selection
exceptions; named-topology loss uses `TopologyEvolutionError`; a kernel
failure after valid input raises `RuntimeError` naming the public operation.

WP6.11 adapts a singular `FaceRef` to an existing waveguide port or
frequency-field monitor only when the trimmed face is an exact, hole-free,
axis-normal rectangle. The adapter compares the face area with its rectangle
rather than using its bounding box as an approximation. A port additionally
requires its owner to be in the model and its plane to be an actual PEC domain
face without symmetry or absorber displacement. `PortWaveguide.from_face`
returns the ordinary `plane`/`corners` declaration; `MonitorFieldFrequency.from_face`
returns the ordinary degenerate-box declaration. They capture current world
coordinates and use existing project serialization. Moving an owner requires
retrieving a new named reference and making a new declaration. Flux, domain
boundary and Huygens consumers retain their whole-cross-section or closed-box
semantics; no single-face adapter changes their numerical region.

### 8.6 BoundaryConditions

A string-typed thin facade.  Each face takes one of
``"PEC"`` | ``"PMC"`` | ``"CPML"`` | ``"Periodic"``, or a symmetry
declaration (DD-159) — ``"SymmetryPEC"`` / ``"SymmetryPMC"``
(domain clip at plane 0.0), a ``("SymmetryPEC", position)`` tuple
(clip at the given world coordinate), or ``"ForceSymmetryPEC"`` /
``"ForceSymmetryPMC"`` (no clip; the geometry is built as the half
model) — see below.  Declared on the
``GeometryModel`` (or ``Mesh.from_grid`` /
``mesh.with_boundary_conditions``) and carried by the ``Mesh`` —
DD-103.  One declaration drives all its consequences: CPML grid
extension, PMC grid-line pull-in, PEC wall mask, symmetry domain
clip, runtime BC objects.
An undeclared face closes with PEC.  The analyses read the closure off
the mesh (``analysis.boundary_conditions`` is read-only).

```python
@dataclass
class BoundaryConditions:
    xmin: str = "PEC"          # face values accept str or symmetry tuple;
    xmax: str = "PEC"          # normalised to the physical wall type
    ymin: str = "PEC"          # in __post_init__ (DD-154/DD-159)
    ymax: str = "PEC"
    zmin: str = "PEC"
    zmax: str = "PEC"
    cpml_thickness_cells: int = 8
    symmetry: dict = field(default_factory=dict)   # {face: position_or_None}
```

**Symmetry planes (DD-154/DD-155).**  A symmetry face is physically a
plain PEC/PMC wall plus the semantic "the mirror image of the model
exists beyond this plane".  On construction the face field is
normalised to the physical wall type (``"PEC"``/``"PMC"``), so every
consumer dispatching on the type keeps working; the semantics live in
the canonical ``symmetry`` map, read via
``boundaries.boundary_conditions.symmetry_entries()``.  At most one
symmetry face per axis.  A clip declaration (``"SymmetryPEC"``/
``"SymmetryPMC"``, plane 0.0, or the tuple form with an explicit
world coordinate) makes the mesher clip the computational domain to
the kept half-space before plane clustering (DD-154) — the full
geometry may be modelled, the mirror half is never meshed; a
``ForceSymmetry*`` declaration carries no position and the geometry
is taken to end at the plane.  Symmetry-aware readers restore full-model
semantics: port reports publish full-model impedances (per cutting
PMC plane ``z_full = z_half/2``, per PEC plane ``×2``), field plots /
overlays / ParaView exports mirror on read, and declared source
amplitudes are full-model watts (DD-155): the excitation injects
``×1/√2`` per port-cutting plane, the recorder composes ``×√2`` per
plane onto ``record_scale`` (one place for a/b, S and stored
signals), and ``MonitorFluxTime`` books ``×2`` per plane cutting its
cross-section — source-independently.

Advanced users who want per-face control over CPML thickness or custom
BC subclasses can pass a ``dict[str, BoundaryProtocol]`` directly to
``FITTimeDomainSolver(boundary_conditions=…)`` (the classes live in
``magnelio.boundaries``).

### 8.7 Ports — declarative objects, specs and operators

Declarative ports (top level; resolved by the analysis against the
finished mesh):

| Declarative | Purpose |
|-------------|---------|
| ``PortWaveguide`` | A face and a mode count; the analysis picks the TEM/QTEM Laplace or TE/TM curl-curl path from the cross-section.  Optional ``corners=`` window (DD-153): two opposite corner points in world coordinates, projected onto the port face, restricting the port to a sub-rectangle of the face; ``None`` components reach the domain boundary, default covers the whole face. |
| ``PortAnalytical`` | Closed-form reference modes (``family="coax"`` / ``"rect_wg"``): ``inner_radius=``/``outer_radius=`` for coax, ``width=``/``height=`` for rect-WG, ``center=`` a 3D world-coordinate anchor projected onto the face (coax axis resp. lower-left corner). |
| ``PortLumped`` | Lumped Thévenin port between two grid-aligned points, optionally RLC-backed. |

Specs and operators (``magnelio.ports``; what the declaratives resolve
into, usable directly for custom setups):

| Spec | Purpose | Operator |
|------|---------|----------|
| ``PortSpecLumped`` | Lumped Thévenin port between two grid-aligned points | ``PortOperatorLumped`` (n_modes = 1) |
| ``PortSpecCoax`` | Analytical TEM coax cross-section | ``PortOperatorModal`` |
| ``PortSpecRectWG`` | Analytical hollow rectangular waveguide TE/TM modes | ``PortOperatorModal`` |
| ``PortSpecNumerical`` | Numerical mode-solver on the FIT cross-section grid (single mode-type) | ``PortOperatorModal`` |
| ``PortSpecMultiConductor`` | Numerical TEM/QTEM Laplace solver with auto-detected conductors | ``PortOperatorModal`` |

At the spec layer ``plane`` is a ``BoxFace`` enum value (the string
spellings are declarative-layer convenience), and the sub-rectangle
window keeps the tangential-2D form under ``window=``
(``PortSpecNumerical`` / ``PortSpecMultiConductor``,
``PortPlane.from_mesh``) — ``window_from_corners`` in
``ports/declarative.py`` does the world-corner projection (DD-153).

All operators implement the :class:`Port` protocol: ``project_V``,
``project_I``, ``update_e``, ``set_excitation``, ``clear_excitation``,
``name``, ``n_modes``.  This is what allows the unified
``PortSignalRecorder`` and the single ``ports=[…]`` slot on the solver.

### 8.8 Problem-class roadmap (DD-224)

Problem classes are `Analysis<Problem><Formulation>`: the suffix names
the formulation the user reasons in (`TD`/`FD`, written only where a
problem exists in both), never the discretisation — the method follows
from the mesh's element type (`method=` overrides), the algebraic solver
from `solver=`.  Results are `<Problem><Formulation>Result`; the
suffix-free name is the contract protocol (`ScatteringResult`).

| Class | Status | Base | Result |
|-------|--------|------|--------|
| `AnalysisScatteringTD` | shipped | `AnalysisTD` | `ScatteringTDResult` |
| `AnalysisEigenmode` | shipped | `_AnalysisBase` | `EigenmodeResult` |
| `AnalysisTD` | shipped (Phase B) | `_AnalysisBase` | `TDResult` |
| `AnalysisWakefieldTD` | reserved | `AnalysisTD` | `WakefieldTDResult` |
| `AnalysisPIC` | reserved | `AnalysisTD` | `PICResult` |
| `AnalysisFD` | reserved | `_AnalysisBase` | `FDResult` |
| `AnalysisScatteringFD` | reserved | `AnalysisFD` | `ScatteringFDResult` |
| `AnalysisElectrostatic` / `AnalysisMagnetostatic` / `AnalysisCurrentStatic` | reserved | `_AnalysisBase` | `ElectrostaticResult` / `MagnetostaticResult` / `CurrentStaticResult` |
| `AnalysisParticleTracking` | reserved | `_AnalysisBase` | `ParticleTrackingResult` |

Excitation triad (DD-224, shipped in Phase A): `Source<Kind>` objects
are declared on the model before meshing (`model.add_source`, carried
as `Mesh.sources`, stored in `mesh.h5` by class-name tag),
`Waveform<Kind>` objects in `magnelio.signals` are pure time functions
with a bandwidth (ABC `Waveform`; the recipe serialises them by
class-name tag), and the core `Excitation(source, mode=, waveform=,
amplitude=, delay=, phase_deg=)` binds one to the other in
`run(excitations=[…])` — simultaneous in one run (`AnalysisTD`,
shipped in Phase B: `run(t_end=, name=)`, continuous-wave rules,
delay-aware run-length estimate, one excitation buffer per port mode,
`TDResult`, project-store runs by name with `Project.result(name)`;
store schema 2.0 with `mesh.h5:element`).  Sequential channel runs stay
`AnalysisScatteringTD.run(excited=…)`, with `waveform=` as the
override of the per-mode default.  The full reserved
vocabulary (sources, waveforms, monitors, namespaces, engines,
arguments) is the table in DD-224.

Stored scattering results use selective reads and calibrated channel
spectra (DD-283). Metadata access reads no V/I payloads; S(out,in)
evaluates the driven and observed channels of one excitation run.
Band ports retain all coupled projections of the involved cross-sections.
Completed-run spectral caches share denominators across queries and use
frequency-axis content and numerical options as keys. Custom axes are
cached as well. Metadata changes and refresh invalidate result caches;
running/aborted records are not cached as final results. Selected reads
retain the common flushed prefix across every stream in a run.
Direct Fourier evaluation shares frequency/sample blocks across records
and applies optional tapering within those blocks. The existing large-axis
modal rFFT/interpolation fallback remains; band evaluation stays direct.

---
## 9. AnalysisScatteringTD.run() — convenience parameters

```python
def run(
    self,
    f_axis: np.ndarray | None = None,    # override; default: constructor f-axis
    excited: list | None = None,         # bare port names or (name, mode) tuples
    accuracy: str = "normal",            # "draft" | "normal" | "high"
    energy_stop_db: float | None = 70.0, # early-stop threshold (DD-019)
    total_time_steps: int | None = None, # explicit hard cap (default: unbounded, DD-070)
    taper_signals: bool = False,         # opt-in Tukey window on V/I (alpha=0.05)
    checkpoint_interval: int | None = None,          # store path only (DD-070)
    port_signal_stop_db: float | str | None = "auto",  # |V|-envelope stop (DD-096/DD-114)
    max_time_steps: int | str | None = "auto",         # runtime backstop cap + stall watchdog
) -> ScatteringTDResult:
    """
    Run one independent FIT-TD simulation per excited (port, mode) pair
    and merge the resulting S-matrix columns.

    Default behaviour: the run is unbounded (DD-070) and ends on
    whichever stop criterion fires first — ``energy_stop_db = 70 dB``
    (stored EM energy decayed below peak) or the DD-096 port-signal
    criterion ``port_signal_stop_db`` (modal-port |V| envelope decayed
    below its run peak).  The latter defaults to ``"auto"`` (DD-114):
    60 dB when at least one modal port is present, disabled on
    lumped-only runs; it only arms once the auto-sized step estimate
    is reached, so it cannot fire mid-transit.  It is the criterion
    that actually terminates shielded lossless structures, whose
    stored energy plateaus on TM-cut-off cavity content no port can
    drain.
    Both decay criteria wait for every finite excitation's nominal end,
    including resolved delay and finite synthesised port-source buffers
    (DD-282). Attached plane-wave sources extend that guard by spatial
    retardation on their actual injection patches. Continuation uses
    absolute completed steps. Energy/voltage decay measures are heuristics,
    not bounds on the omitted Fourier tail or on complex S-parameter error.

    Parameter priority:
    - dt is computed from the Courant condition with the chosen accuracy
      safety factor and the mesh's effective ε / μ floors.
    - total_time_steps default ``None``: the run is unbounded; the
      ``ceil((t_pulse + 25·t_diag) / dt)`` estimate (with
      ``t_pulse = max(delay + waveform.t_end)``,
      ``t_diag = ‖bbox‖ / (0.5 c₀)``)
      only sets the stop-check cadence.  An explicit value restores a
      hard cap.
    - excited default: ``[(first_port_name, 0)]``.

    accuracy → Courant factor:
        "draft"  → 0.90
        "normal" → 0.95
        "high"   → 0.99

    energy_stop_db:
        Default 70 dB.  The solver checks total EM energy on a 100-step
        cadence and terminates once it has decayed by this many dB
        below peak, after the finite-excitation guard. The current port
        interval is polled before either decay decision; running peaks
        and the pending interval survive continuation checkpoints.
        ``PortSignalRecorder.finalize`` trims V/I buffers
        to the actual leapfrog count so the FFT does not see zero-
        padded tails.  On structures whose energy never falls 70 dB
        below peak (closed cavities, narrow filters) the port-signal
        criterion terminates the run instead (DD-096/DD-114).

    taper_signals:
        Default False.  When True, multiply every recorded V/I time
        series with a symmetric Tukey window of ``alpha = 0.05`` (the
        first and last 2.5 % of samples taper to zero) before the
        S-parameter DFT.  Suppresses rectangular-window sidelobes when
        the run is forced to terminate before the residual has fallen
        below the FFT noise floor — a niche option for runs with
        aggressive ``energy_stop_db`` cuts on TEM lines.  Off by
        default because for runs sized at the calibrated 70 dB depth
        the rectangular DFT is already the unbiased reference.
    """
```

---

## 10. Testing Strategy

### 10.1 Unit Tests (`tests/unit/`)

Each module is independently testable without running the full solver.

| Test file                | What is tested |
|--------------------------|----------------|
| `test_geometry.py`       | CSG Boolean ops: correct BRep volume, bounding-box accuracy |
| `test_mesh.py`           | Gradient condition h_{i+1}/h_i ≤ g, forced_planes exact, Yee cell count |
| `test_mesh_from_grid.py` | `Mesh.from_grid()` region-based material filling |
| `test_materials.py`      | Material matrix assembly for anisotropic case; PEC mask correctness |
| `test_operators.py`      | `C · e + C^T · h = 0` (discrete Stokes), sparsity pattern, symmetry |
| `test_boundaries.py`     | CPML σ profile monotonicity, aux-field dimensions vs PML thickness |
| ~~`test_ports.py`~~      | ~~Port2D~~ (removed) |
| `test_port_waveguide.py` | WaveguidePort: TE/TM/TEM modes, Z_pi, β(f_ref), H-field profiles, modal ABC |
| `test_solver.py`         | Single Leapfrog step energy conservation (lossless), Courant formula |
| `test_fit_td.py`         | FITTimeDomainSolver integration (energy stopping, port excitation) |
| `test_sources.py`        | SourcePlaneWave TF/SF injection |
| `test_postprocessing.py` | S-parameter FFT, field probes |
| `test_io.py`             | HDF5 roundtrip, VTK export |

### 10.2 Integration Tests (`tests/integration/`)

Small solver runs (≤ 20³ cells) verifiable in seconds.

| Test file                     | Scenario |
|-------------------------------|----------|
| `test_rectangular_cavity.py`  | 10×10×10 cell PEC cavity; 3 lowest eigenmodes, error < 5% |
| `test_waveguide.py`           | Rectangular WG section; S21 magnitude in passband > −1 dB |
| `test_discrete_port.py`       | λ/4 stub; S11 null at design frequency, error < 5% |
| `test_plane_wave.py`          | Plane-wave propagation: arrival time and peak amplitude |

### 10.3 Benchmarks (`benchmarks/`)

Full-scale validation against analytical solutions. Scripts output JSON report.

| Script                           | Case                        | Acceptance criterion |
|----------------------------------|-----------------------------|----------------------|
| `bench_rectangular_cavity.py`   | Rectangular cavity modes    | f_error < 2% |
| `bench_spherical_cavity.py`     | Spherical cavity modes      | f_error < 2% |
| `bench_waveguide_transmission.py`| Waveguide S-parameters     | S-param error < 2 dB |
| `bench_microstrip.py`           | Microstrip S-parameters     | S-param error < 2 dB |
| `bench_sphere_scattering.py`    | Plane-wave RCS from sphere  | RCS error < 2 dB |
| `bench_stripline.py`            | Stripline S-parameters      | S-param error < 2 dB |
| `bench_mesh_build.py`           | Mesh-build time per mesher pass on three production geometry classes (Lange-coupler row, patch array with corporate feed, post row), section-pool arms off / auto / forced; `sheets` (thin-sheet detection + footprints), `singular` (singular-edge planes), `overlap` (model overlap check) and `fuse` (every N-ary union, nested) columns | timing only; results in `benchmarks/results/bench_mesh_build.json` |

### 10.4 Notebooks (`examples/notebooks/`)

| Notebook | Content |
|----------|---------|
| `01_rectangular_cavity.ipynb` | Cavity eigenmodes (mode-sort, eigenfrequency comparison) |
| `02_waveguide_port.ipynb`     | Plane-wave propagation demo |
| ~~`03_microstrip.ipynb`~~     | ~~Microstrip S-parameters with Port2D~~ (removed) |
| `04_parallel_plate_waveguide.ipynb` | Parallel-plate TEM waveguide |
| `05_coaxial_rg58.ipynb`       | RG-58 coaxial cable simulation |
| `06_circular_waveguide.ipynb` | Circular waveguide modes |
| `07_microstrip_rogers4003.ipynb` | Microstrip on Rogers RO4003 substrate |
| ~~`08_rect_coax.ipynb`~~      | ~~Rectangular coaxial waveguide~~ (removed: depended on FIT eigenvalue 2D mode solver, deleted in step 9a as out of Phase-1 scope per `reference_architecture_waveguide_ports.md` §1) |

### 10.5 CI Strategy

- **Current:** Local Git, manual `pytest tests/` invocation
- **Future:** GitHub Actions — `pytest` + `ruff check` + `mypy src/`

---

## 11. Implementation Order

Recommended implementation sequence:

| Step | Module / Task |
|------|---------------|
| 1    | Repo structure + `pyproject.toml` — empty packages, `import magnelio` works |
| 2    | Backend abstraction — `get_xp()`, `set_backend()`, unit tests |
| 3    | `Material` dataclass + `GridLines` dataclass — pure Python, no deps |
| 4    | Operator matrices — sparse C, C^T, M_eps, M_mu (well-isolated, testable) |
| 5    | Mesh module — grid-line algorithm, OCC queries for critical planes |
| 6    | Geometry module — CSG primitives, Boolean ops, OCC backend |
| 7    | CPML + BCs — auxiliary fields, PEC/PMC masks, periodic ghost cells |
| 8    | FIT-TD solver — leapfrog loop (ports/sources stubbed out first) |
| 9    | Discrete port — simplest port type |
| 10   | 2D eigenmode port + S-parameter extraction — most complex part |
| 11   | Plane-wave TF/SF source |
| 12   | 3D eigenmode solver |
| 13   | I/O — HDF5 and VTK export |
| 14   | Run benchmarks, fix discrepancies |
| 15   | Example notebooks |
| 16   | Microstrip S-param: deembedding (DD-018), energy stop (DD-019), thin PEC (DD-017) |
| 17   | Waveform overhaul: Gaussian for TEM, modulated Gaussian for TE/TM (DD-022) |
| 18   | WaveguidePort: modal Mur-ABC (DD-027, supersedes DD-023), multi-mode solver, general 6-face port |
| 19   | Laplace TEM solver (DD-020), Z_pi power-current impedance (DD-025) |
| 20   | H-field storage + TM eigenwert-solver + β(f_ref) + Z_wave(f) (DD-026a, DD-026b) |
| 21   | WaveguidePort integration into `Simulation` high-level API |
| 22   | Fix 3D simulation with waveguide ports (currently broken) |

---

## 12. Open Questions

See `STATUS.md` for current open questions and project status.

---

## 13. Verification

Standing gates (run from the repo root, magnelio environment):

```bash
# All unit and integration tests pass
python -m pytest tests/

# Lint and formatting are clean (also enforced by pre-commit and CI)
ruff check . && ruff format --check .

# Public API surface: one name, one documented home
python validation/tools/check_api_surface.py

# Every DD-NNN citation resolves against design-decisions.md
python validation/tools/check_dd_references.py
```

The measured accuracy floors (port reflection, wall loss, curvature)
and the validation scripts that regenerate them are tracked per DD;
the current numbers are summarised in `STATUS.md`.

---

## 14. Implementation Status

Step completion, open questions, known bugs, and current benchmark
results are tracked in `STATUS.md`.  Raw benchmark data lives in
`benchmarks/results/`; the reasoning record is `design-decisions.md`.
