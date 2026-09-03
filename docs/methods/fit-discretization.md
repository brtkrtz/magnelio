# Spatial discretisation and time integration

## Finite Integration Technique (FIT)

Magnelio discretises Maxwell's equations with the Finite Integration
Technique on a pair of staggered, axis-aligned Cartesian grids (primal
grid for electric grid voltages $\hat e = \int \mathbf E \cdot
\mathrm d\mathbf l$, dual grid for magnetic grid voltages $\hat h$).
FIT is the integral-form reformulation of Maxwell's equations
introduced by Weiland {cite}`weiland1977`;
the mature
formulation with the discrete topological operators (curl matrix
$\mathbf C$, exact $\mathbf S \mathbf C = 0$ identities) and diagonal
material mass matrices is summarised in {cite}`weiland1996`
and {cite}`clemensweiland2001`.

On a Cartesian dual-orthogonal grid the FIT time-domain update is
algebraically identical to the finite-difference time-domain (FDTD)
scheme of Yee {cite}`yee1966` — the source
code refers
to the staggered grid as the "Yee grid" throughout.  Magnelio keeps
the FIT viewpoint because the material matrices, sub-cell conformal
corrections and port-mode restrictions are all expressed as
modifications of the diagonal mass matrices
$\mathbf M_\varepsilon, \mathbf M_\mu, \mathbf M_\sigma$ rather than
of the stencil.

Implementation: `operators/curl.py` (sparse primal curl $\mathbf C$;
the dual curl is $\mathbf C^{\mathsf T}$), `operators/material_matrices.py`
(diagonal mass matrices $M_\varepsilon = \varepsilon_0\varepsilon_r
A_{\text{dual}}/l_{\text{primal}}$, $M_\mu = \mu_0\mu_r
A_{\text{primal}}/l_{\text{dual}}$, plus $M_\sigma$ and the magnetic
$M_{\sigma^*}$), `fields/field_arrays.py` (structure-of-arrays field
storage, DD-002).

## Leapfrog time integration

The time-domain solver (`solver/fit_td.py`) marches the fields with
the standard second-order leapfrog scheme, staggered by half a time
step:

$$
\hat e^{n+1} = \alpha_E\,\hat e^{n}
             + \beta_E\, \mathbf C^{\mathsf T} \hat h^{n+1/2},
\qquad
\hat h^{n+3/2} = \alpha_H\,\hat h^{n+1/2}
              - \beta_H\, \mathbf C\, \hat e^{n+1}.
$$

Ohmic loss ($\sigma$, and the magnetic $\sigma^*$ on the H side) is
folded into $\alpha,\beta$ with the **semi-implicit (time-averaged)
conductor update**, the standard exponential-free treatment of lossy
media in FDTD/FIT; the scheme and its properties are textbook material
{cite}`taflovehagness2005`.  The leapfrog scheme
itself goes back to Yee {cite}`yee1966`.

## Stability (CFL condition)

The explicit leapfrog scheme is conditionally stable under the
Courant–Friedrichs–Lewy criterion
{cite}`courant1928`, in the FDTD form

$$
\Delta t \le \frac{s}{c_{\max}\sqrt{\Delta x_{\min}^{-2}
 + \Delta y_{\min}^{-2} + \Delta z_{\min}^{-2}}},
\qquad s < 1,
$$

(`solver/stability.py`, safety factors 0.90/0.95/0.99).  The FDTD form
of the criterion is standard {cite}`taflovehagness2005`, but as a
worst-case product over per-axis minima it is loose on conformal
meshes, where partially filled sub-cells reduce the local mass-matrix
entries.  The production time step is therefore taken from the sharp
algebraic criterion instead: the leapfrog is stable iff

$$
\Delta t \le \frac{2}{\sqrt{\lambda_{\max}
 \left(M_\varepsilon^{-1} C^{\mathsf T} M_\mu^{-1} C\right)}},
$$

with the spectral radius measured at solver setup by a matrix-free
Lanczos iteration on the symmetrised operator (restricted to the
degrees of freedom the update actually advances) and cached on the
mesh.  If the iteration does not converge, a row-sum (Gershgorin)
upper bound on the same operator — strictly safe, typically within
20 % — takes its place.  Partially filled cells therefore cannot
destabilise the run, and they no longer throttle it either: the
worst-case product under-estimates the stable step by an order of
magnitude and more on curved conformal walls, while the measured
limit stays near the geometric value.

Sub-cell corrections that scale $M_\varepsilon$ and $M_\mu$ in
opposite directions (the thin-wire pair correction, the LC-consistent
conformal coupling) are constructed to leave the pair product — and
with it the local wave speed entering the CFL bound — unchanged; see
the [conformal geometry chapter](meshing-conformal.md).

## Selectable precision

The whole time-loop state (fields, update coefficients, CPML and
auxiliary ADE/SIBC states) can run in IEEE-754 single precision
(`precision="single"`, the production default) or double precision
(DD-094), while accumulating quantities — the energy reduction, the
DFT accumulators, the port arithmetic, geometry and mode solves —
always stay in double.  This mixed-precision layout is engineering
practice in production FDTD/FIT codes, not a research method.
Stability of the reduced-precision auxiliary recursions follows from
their contractive form ($|k| < 1$ for every decaying IIR branch); this
is analysed per operator in the repository (DD-094), not taken from
the literature.

What the choice costs and buys, and how to recognise a result limited
by the word length rather than by the mesh, is the subject of its own
chapter — see [numerical precision](precision.md).

## Simulation duration and energy stopping

Runs are terminated either after a fixed number of steps or by an
energy criterion: the total discrete field energy
$\tfrac12(\hat e^{\mathsf T} \mathbf M_\varepsilon \hat e + \hat
h^{\mathsf T} \mathbf M_\mu \hat h)$ must decay a configurable number
of dB below its peak (DD-019).  Energy-based stopping is common
engineering practice in time-domain S-parameter extraction; no
specific publication is claimed.
