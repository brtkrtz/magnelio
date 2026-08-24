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
geometry, and the chosen $Z_0$.  The how-to guide *Characterising a
lumped port termination* measures both against a waveguide port
(DD-189) instead of relying on rules of thumb.

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
