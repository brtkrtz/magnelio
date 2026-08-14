# Mesh generation and conformal geometry

## Geometry kernel and material filling

Solid geometry is authored through a CSG layer (`geometry/`) backed by
the Open CASCADE kernel via `pythonocc-core` (DD-003, DD-016).
Material assignment on the grid uses exact boundary-representation
queries (solid classification, 3D face–solid intersection, planar
cross-sections) rather than voxel sampling.  This is engineering
infrastructure on top of a third-party kernel, not a numerical-methods
contribution.

## Graded Cartesian mesh

The mesh generator (`mesh/mesher.py`) produces a graded (non-uniform)
Cartesian tensor-product grid: geometry-derived fixpoints ("anchors",
plane clustering, DD-059…DD-062) plus feature-based two-scale
refinement (`h_fine` near features, `h_coarse` in bulk, geometric
grading between them, DD-028).  Graded Cartesian meshes and the
accuracy trade-offs of local grading are standard FDTD/FIT practice
{cite}`taflovehagness2005`; the specific fixpoint,
plane-clustering and thin-sheet heuristics are in-house engineering.

## Conformal sub-cell material matrices (partially filled cells)

Material boundaries that cut through grid cells are represented by
**area/length-weighted averaging in the mass matrices** instead of
staircasing: per primal edge the classifier stores an averaged
$\bar\varepsilon$, a free (non-PEC) length fraction and a free dual-face
area fraction; per dual face a corresponding $\bar\mu$ and free-area
data (unified per-edge/per-face sub-cell classification, DD-051).
This family of techniques — retaining the standard leapfrog update and
encoding sub-cell geometry purely in the material matrices — was
introduced for FIT by Krietenstein, Schuhmann, Thoma and Weiland
{cite}`krietenstein1998`.

For perfectly conducting boundaries the classifier additionally
shortens partially-PEC edges (free-length weighting), which is the
conformal-PEC idea of Dey and Mittra {cite}`deymittra1997`
(DD-036, since generalised into the unified classifier of DD-051).

Two refinements are in-house:

- **LC-consistent pair coupling** (DD-053, `couple_face_material_pairs`):
  on dual faces with a locally translation-invariant ladder direction,
  the averaged $\bar\mu$ is replaced by the value that makes the
  co-located product $M_\varepsilon M_\mu$ equal the exact
  transmission-line value $\varepsilon_0\mu_0\,\varepsilon\mu\,d\tilde d$,
  so a discrete travelling wave on a uniform line is exact
  (derivation in `design-decisions.md` DD-053).
- **Enlarged-cell donor** (DD-058, implemented but dormant — measured
  neutral): stabilising strongly cut cells by borrowing area from the
  uncut neighbour.  The published antecedent is the family of
  uniformly stable conformal schemes / enlarged-cell techniques, e.g.
  Zagorodnov, Schuhmann and Weiland {cite}`zagorodnov2003`.

## Thin conducting sheets

Zero-thickness or sub-cell metallisation is detected before gridding
(DD-035, DD-059) and represented as PEC edge masks on the primal grid
(`apply_thin_pec_sheet`, DD-017) — the standard thin-sheet treatment
in Cartesian time-domain solvers {cite}`taflovehagness2005`
(subcell thin-sheet models are ch. 10 there; the
detection pipeline itself is in-house).

## Thin-wire sub-cell model

`ThinWire(curve, radius)` embeds a conductor thinner than a cell as a
PEC edge chain with corrected surrounding material matrices
(`mesh/thin_wire.py`, DD-080).  The model is the classic thin-wire
sub-cell treatment of Holland and Simpson {cite}`hollandsimpson1981`,
realised in the paired $(m, 1/m)$ encoding of
Noda and Yokoyama {cite}`nodayokoyama2002`:
the four encircling
dual faces scale $M_\mu$ by
$m = \ln(\delta/a)/\ln(\delta/r_0)$ and the co-located radial edges
scale $M_\varepsilon$ by $1/m$, so the wire presents the physical
per-length inductance $L' = (\mu/2\pi)\ln(\delta/a)$ while the pair
product — and hence the wave speed and the CFL bound — is untouched.
The bare-grid equivalent radius $r_0 = \kappa_0\,\delta$ with
$\kappa_0 = e^{-\gamma}/2^{3/2} \approx 0.1985$ comes from the
square-lattice Green's function, as given in the thin-wire literature
{cite}`nodayokoyama2002`.

## Mesh quality safeguards

Hard minimum cell size with floor-aware refits and a longitudinal
series-$\varepsilon$ correction (DD-060), per-axis fine resolution
(DD-061) and a permanent 30-case stress sentinel (DD-062) are
in-house engineering.
