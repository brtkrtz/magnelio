# Development follow-ups

This file tracks unfinished engineering work: robustness and validation
campaigns, investigations, maintenance and deferred improvements. Entries are
work to evaluate or complete, not promises of features or accepted designs.
Check items off with a link to the resulting evidence or change; retain a
short reason when an item is cancelled or superseded.

- `STATUS.md` describes the current implementation and its limitations.
- `known-bugs.md` records investigated defects and their stable bug anchors.
- `design-decisions.md` records decisions once a strategy is accepted.
- This file records the work still needed before those outcomes exist.

## Mesher robustness and diagnosability

**Priority: high. Status: open; implementation is not started.**

The bounded-surface section route introduced several coupled numerical stages:
support intersection, face-domain trimming, shared-edge association, node
refinement and contour assembly. Recent valid-CAD rejection defects exposed
missing assumptions about periodic seams, embedded edges and repeated endpoint
tags. Conditioning retries can also mask a failure at an individual scale.
Fixes and passing regressions establish known cases, not production reliability
for arbitrary new CAD assemblies. Context: DD-287 and KB-043/053/054.

**Goal:** establish and report a defensible supported geometry envelope for
both section routes, and make failures reproducible at the failing stage
without debugging an entire large simulation. Cover general 3D assemblies,
arbitrary material layouts and hundreds of primitives; simple cavities and
lines are reference cases, not the intended scope.

### Work packages

- [ ] Define separate acceptance criteria for functional robustness (valid CAD
  meshes successfully) and numerical correctness (material regions and holes
  are preserved). A successful build is not an accuracy certificate; an
  explicit rejection of valid CAD is still a functional defect. Retain the
  default fast route's known tangency limitation until it is independently
  resolved. Keep applicability, performance and wall-loss work separate below.
- [ ] Establish explicit, individually testable contracts between intersection,
  trimming, endpoint association, node refinement and contour assembly.
  Identify which responsibilities actually need separation or refactoring;
  splitting files alone is not an improvement. Keep numerical residual,
  CAD-tolerance and contour-consistency criteria distinct.
- [ ] Build a reproducible corpus of valid geometry combinations: Boolean
  contacts and differences, fillets, drilled bodies, sweeps, lofts, imported
  trimmed faces, holes, periodic seams and degenerate boundaries. Vary cut
  axes/positions, grazing distance, units, physical scale, placements,
  operation order, supported CAD versions and serial/parallel execution.
  Add seeded generated cases alongside real failure reproducers; report the
  tested coverage and the classes not yet verified.
- [ ] Apply independent analytic or quadrature references where available,
  signed material-area checks and construction/decomposition invariants.
  Treat topology/closure and volume-consistency checks as complementary;
  agreement between two approximations that can share the same error is
  insufficient. Preserve existing acceptance bounds.
- [ ] Exercise every conditioning choice directly. Preserve failed attempts
  even when a retry succeeds, and test that reusing prepared geometry or a
  different operation order does not change the accepted region. Resolve
  underlying defects rather than widening tolerances, moving the requested
  plane or silently discarding endpoints or material islands.
- [ ] Define bounded, opt-in failure capture: original CAD or a replayable
  recipe, exact cut, scale/budget, route and stage, kernel version, workers,
  conditioning attempts and failed invariant. Support a standalone single-cut
  replay and a reduced reproducer. Keep confidential inputs in internal
  dossiers; no routine diagnostic output or plots in application scripts.
- [ ] Measure the hardened route on representative large assemblies, recording
  build time and peak memory as well as numerical outcomes. Avoid extrapolating
  from primitive counts or one small reference structure.

### Completion criteria

- [ ] A documented, reproducible corpus passes its functional and independent
  numerical criteria across the supported CAD/environment matrix.
- [ ] Representative real assemblies and previous failure cases pass without
  hidden reliance on one conditioning choice or silent loss of material.
- [ ] New failures can be isolated and replayed at a named stage with bounded
  diagnostic cost. Ambiguous or unresolved geometry is reported explicitly.
- [ ] Supported classes, unverified classes and remaining limits are documented;
  regression gates preserve them. Neither a test count nor this campaign is
  presented as a guarantee for every possible geometry. Any change to default
  routing or architectural policy receives its own decision after evaluation.

Evidence and reproducers: `investigations/release-0.9.0/CAD-FIX.md` and
`investigations/kb043-near-tangency/` (internal records).

## Port computation on the GPU

**Status: open investigation; backend placement is not decided.**

The maintainer observed CPU-side port work limiting the GPU run of the
51-cell / 102-port model. Quantify that bottleneck and investigate whether
port computation should move to the GPU. Distinguish one-time preparation
from work repeated at every time step; accelerating the field update alone
may leave port processing and CPU/GPU synchronisation dominant.

- [ ] Profile port mode/operator preparation separately from time-step boundary
  updates, excitation, modal projection and signal recording. Measure host/device
  transfers, synchronisation, CPU scheduling and post-processing separately.
  Preserve a reproducible input and environment; identify the dominant stages
  before selecting a GPU implementation.
- [ ] Compare CPU, GPU and hybrid placement, including batched operations and
  keeping port state/data resident on the device. Determine whether avoiding
  transfers or Python/per-port overhead is more useful than moving arithmetic.
- [ ] Evaluate representative port families and models, numbers of ports/modes,
  record lengths and precisions. Include the 51-cell / 102-port case and smaller
  workloads; cover general 3D structures rather than optimising one fixture.
- [ ] Check complex S-parameters, port signals, normalisation, boundary behaviour
  and energy/power budgets against the CPU reference at matching precision.
  Extend device-port coverage beyond the current dedicated band-DTBC test.
- [ ] Report end-to-end speed, preparation cost and RAM/VRAM peaks with controlled
  threads/load and repeated measurements. Document the crossover where each
  placement is worthwhile, implementation cost and remaining numerical limits.
  Record a design decision only after the evidence supports a strategy.

Context: DD-092 and the current GPU-port coverage in `STATUS.md`. Related
run records: `investigations/hesr-memory-budget/` and
`investigations/near-tangent-hesr-impact/` (internal dossiers). The observed
bottleneck is a profiling starting point, not a completed GPU-placement study.

## Documentation and maintenance

- [x] Reconcile `spec.md`'s current architecture/API/storage/timing descriptions
  and obsolete numerical sketches with code, methods and accepted decisions
  (2026-10-08). Correct curl dimensions/incidence and already-included inverse
  masses, spectral stepping, current ports and test/benchmark inventories.
  Historical planning is marked explicitly; no arithmetic or threshold changed.
- [x] Review/condense completed and refined DDs and resolved KBs (2026-10-08).
  Preserve headings, rationale, limits and evidence anchors; leave open defects
  and independent investigations intact. Detailed numerical records remain
  where their constraints are needed, with no arbitrary historical line limit.
- [x] Enforce structural project consistency in pre-commit and CI (2026-10-08):
  STATUS <=400 lines, three matching versions and dated changelog/tag coverage.
  CI also checks DD resolution. Tests/builds cannot certify semantic completeness
  of every feature's documentation; that remains part of feature acceptance.

Audit: `investigations/housekeeping-completion-2026-10-08/MEASUREMENTS.md`
(internal record). The separate private documentation backlog retains unfinished
runtime, meshing, ports, GPU and troubleshooting guides.

## Previously deferred work

* **Double-precision late energy growth:** understand the increase in both
  repeated-cell runs, already present before GPU resume; cause remains open.
* **Robust-section applicability:** find relevant geometries and indicators for
  targeted warnings/automatic selection. Actual wall losses remain uninvestigated
  and could be highly relevant; weight diagnostics are not loss-error measurements.
  Checklist: `investigations/near-tangent-hesr-impact/FOLLOW_UP.md` (internal dossier).

* **Pulsed band-edge S-parameters on dispersive lines** are record-
  truncation limited.  The candidate — late-time pole estimation — is
  built ([[DD-272]]) but untested on *this* case: a dispersive line's tail
  is not obviously a few resonances.
* **A third compute backend** — assessed, nothing built (DD-180);
  blocker is `xp is not np` as the capability test.  Metal rejected (no
  FP64), CuPy on ROCm is the candidate.
* **Temporal blocking of the CPU kernel** — the only route above the
  STREAM triad (DD-257); incompatible with the per-step hooks (ports,
  CPML, sources, recorder).  Not pursued.  A float32 curl accumulator
  on the E side would buy 3 % on Apple Silicon and is a numerics change.
* **Residual GPU small-grid floor** (~0.41 ms/step at 10k cells, port round
  trips — DD-092); **tensor (gyrotropic) μ** (DD-089's ADE is scalar per
  axis); **off-Yee field-monitor interpolation** (must preserve the DD-085
  units); **far-field accepted power on the streamed path** (``gain`` raises until it wires ``1 − Σ|S|²``, DD-070).
