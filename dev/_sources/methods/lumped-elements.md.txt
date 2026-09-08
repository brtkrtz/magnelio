# Discrete ports and lumped circuit elements

## Semi-implicit Thévenin discrete port

The discrete (lumped) port drives a chain of grid edges with a
Thévenin source $V_s$ behind an internal impedance $Z_0$, coupled
into the E update **semi-implicitly**: the port current is solved
together with the local field update,

$$
i = \frac{v_{\text{src}} - v_{\text{hist}} - v_{\text{total}}}
         {r_{\text{eq}} + \Sigma\beta},
$$

which is unconditionally stable at the unchanged CFL limit
(`ports/discrete/operator.py`, DD-030/DD-075).  Embedding lumped
resistive sources and loads into the FDTD grid in this
field-circuit-consistent way is the established *lumped-element FDTD*
technique of Sui et al. {cite}`sui1992` and Piket-May,
Taflove and Baron {cite}`piketmay1994` (the
semi-implicit averaging of the local field term is the standard
stabilisation in that literature; the specific multi-edge chain
formulation follows the in-repo derivation).

A discrete port is not an exact line termination: its residual
reflection and phase offset depend on the grid at the gap, the gap
geometry and position, and the chosen $Z_0$.  The how-to guide
*Lumped ports: investigations* measures both against a waveguide port
(DD-189) instead of relying on rules of thumb, and the *Lumped port
tuning* pages package the measurement as a per-line-type pre-flight
tool.

## Trapezoidal RLC companion models

General series/parallel RLC two-terminal elements are reduced per time
step to a Thévenin companion $(R_{\text{eq}}, V_{\text{hist}})$ using
the trapezoidal rule (`circuit/companion.py`, DD-077):

$$
\text{inductor: } R_{\text{eq}} = 2L/\Delta t, \qquad
\text{capacitor: } R_{\text{eq}} = \Delta t/2C .
$$

Companion models with trapezoidal (bilinear) integration are the
classical workhorse of circuit simulators of the SPICE family; the
canonical references are Nagel's SPICE2 report {cite}`nagel1975`
and the circuit-simulation textbook treatment of Chua
and Lin {cite}`chualin1975`.  The trapezoidal rule was
chosen (over backward Euler) for its energy conservation on L/C —
matching the non-dissipative leapfrog interior — which is a standard
argument in both circuit and field simulation.

`LumpedElementOperator` (DD-079) unifies the discrete port and
general RLC elements under one operator; the classic resistive port
is the special case `SeriesRLC(R=Z0)` (bit-identical by
construction).

Excitation units follow the power-wave convention: a user waveform in
$\sqrt{\mathrm W}$ is realised as $v_{\text{src}} = 2\sqrt{Z_0}\,a(t)$
(DD-078), consistent with Kurokawa power waves
{cite}`kurokawa1965`.

## Lumped devices on symmetry planes

A lumped port or passive element is always declared as the
**full-model device** — endpoints in full-model coordinates, `Z0` and
R/L/C values of the whole element — even when a symmetry plane cuts
it.  The builder relates the edge chain to every declared plane and
derives the half model itself:

- A chain **crossing an electric symmetry plane** along the plane
  normal (a dipole feed on the mirror plane) must be mirror-symmetric
  about it; it is clipped to the meshed half, which carries half the
  device in series ($Z_0/2$, $R/2$, $L/2$, $2C$).  A chain crossing a
  *magnetic* plane is rejected: the mirrored current is anti-parallel,
  so no physical full-model element corresponds.
- A chain **lying in a magnetic symmetry plane** is one of two
  parallel branches; the meshed half carries the doubled device
  ($2Z_0$, $2R$, $2L$, $C/2$).  A chain lying in an electric wall is
  rejected — its edges would be shorted.
- With the as-built `ForceSymmetry…` spelling the geometry is
  declared halved, so a chain ending on the plane *is* the crossing
  declaration; with the clipping spellings a terminal exactly on the
  plane is rejected with guidance, since the full-model reading of
  that shape is a mirror-twin pair sharing a node.

With the internally scaled device, recorded power waves and the
excitation pick up the same $\sqrt2$-per-plane convention as modal
ports, so S-parameters — and the input impedance
$Z_0(1+S_{11})/(1-S_{11})$ computed with the declared full-model
$Z_0$ — come out as full-model quantities with no further correction.
One caveat mirrors the modal ports: the *ratio* of the raw recorded
terminal signals stays a half-model quantity, so a directly measured
$-V/I$ of a passive load in a magnetic plane reads the doubled
device.

## Edge-path rasterisation

Lumped elements and thin wires ride on a canonical curve rasteriser
that converts an arbitrary polyline/curve into an ordered, directed
staircase of grid edges with per-edge orientation signs
(`circuit/rasterize.py`, DD-076), plus the line integral
`integrate_E` along the path.  This is in-house infrastructure.

## Paths: any direction, at a measured price

A discrete port or lumped element is declared either by its two
terminals (`start` / `end`) or by a `path` — a sequence of points, or a
`Curve`.  Both forms go through the same canonical rasteriser as thin
wires and voltage probes, so the path is free to run obliquely, to
bend, or to follow a curve; the grid carries it as a staircase of
edges.  The chain must traverse each edge once: a two-terminal element
is a series chain, so a self-crossing or doubled-back path is rejected.

The port's polarity follows `start` → `end` (or the path's own
direction), and the recorded V and I change sign with it.

**What a staircase costs.**  The terminal relation is exact on any
path: the states are edge voltages, so KVL along the chain is KVL, and
the gap voltage is the plain signed sum whatever route the chain takes.
What an oblique path does change is the near field — the flux linked
within a few cells of the conductor, i.e. the element's parasitic
series inductance.  In-house measurement gives the excess as a local
quantity, set by the local staircase direction alone:

$$\Delta L' \approx 58\ \mathrm{nH/m} \cdot x^{0.61},
\qquad x = \frac{\text{staircase length}}{\text{chord length}} - 1$$

per unit **chord** length, where `x` runs from 0 for an axis-parallel
path to 0.41 for a 45° one.  For a strongly oblique path that is about
**32 pH per millimetre of element**: negligible for a short feed gap at
low frequency, worth knowing for a long slanted element or at the top
of a wide band.

Two properties of this excess are worth stating plainly, because both
are counter-intuitive:

* **It does not refine away.**  It falls only as `Δ^0.19` with the cell
  size — quadrupling the resolution buys about a fifth of it.  It is a
  floor for any usable mesh, not a discretisation error to be meshed
  out.
* **It is not proportional to the extra path length.**  A 45° staircase
  is 41 % longer than its chord but costs only a few percent of the
  inductance, because the zigzag excursions cancel pairwise beyond a
  cell or two.  Estimating the penalty from the length ratio
  overestimates it by an order of magnitude.

If an oblique path's parasitic inductance matters for your model, the
remedy available today is to align the element with the grid where you
can, and to keep obliquely-routed elements short.

**Conductor edges are not available.**  An edge held at zero by a
perfect conductor — inside a PEC body, or tangential to a PEC wall —
cannot carry the element's injection, so the element is shorted along
it.  The builder counts such edges and warns rather than silently
modelling something else; a delta-gap feed therefore needs a real gap
in the conductor, not a path laid on top of it.
