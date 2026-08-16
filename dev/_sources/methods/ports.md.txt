# Waveguide ports and S-parameter extraction

The port machinery is Magnelio's largest methodological block.  It
combines standard 2D mode analysis with a family of **exact discrete
transparent boundary conditions (DTBC)** that were derived in-house
for the FIT leapfrog lattice; the published antecedents of each layer
are listed per section.

## 2D port-mode solvers

### Curl-curl eigenvalue problem (TE/TM and hybrid modes)

Port cross-section modes are computed from the generalised eigenvalue
problem $\mathbf K \hat e = \omega_c^2 \mathbf M \hat e$ with
$\mathbf K = \mathbf C_{2D}^{\mathsf T}\,\mathrm{diag}(M_\mu^{-1})\,
\mathbf C_{2D}$, where $\mathbf C_{2D}$ is the *row/column slice of
the 3D FIT curl matrix* at the port plane
(`ports/modal/curl_curl_2d.py`).  Discretising the 2D problem in the
same metric as the 3D problem (rather than with an independent 2D
solver) is the property that makes the discrete mode an exact
eigenvector of the 3D-restricted transversal operator; the FIT
curl-curl eigenformulation itself is standard
{cite}`weiland1996,clemensweiland2001`.  The TM
variant is the exact restriction of the same operator, an in-house
derivation (DD-055).

### TEM / quasi-TEM Laplace solver

Multi-conductor cross-sections use the electrostatic route
(`ports/modal/tem_laplace.py`): a 2D FIT Laplace problem
$\nabla\!\cdot(\varepsilon\nabla\varphi_k) = 0$ per signal conductor
with unit-potential boundary conditions, $\hat e_k = -\nabla\varphi_k$,
line capacitance from the discrete energy.  For inhomogeneous filling
the quasi-TEM effective parameters follow the classic two-capacitance
construction $\varepsilon_{\text{eff}} = C'/C'_0$,
$Z_0 = 1/(c\sqrt{C' C'_0})$ — standard quasi-static transmission-line
theory as found in the microwave-engineering literature
{cite}`pozar2012,collin1991` (the specific
formulation is textbook material; no single originating paper is
claimed).  The line-impedance definition used for TEM modes is the
power–current impedance $Z_{\pi}$ (DD-025); the coexistence of
$Z_{PI}$, $Z_{PV}$, $Z_{VI}$ definitions is classical waveguide
theory {cite}`marcuvitz1951,pozar2012`.  Analytical
reference modes for coaxial and rectangular-waveguide ports are
closed-form textbook solutions {cite}`pozar2012`.

### Mode classification and multi-mode merge

TE/TM and TEM/QTEM branches are merged into one unified multi-mode
port; degenerate multi-TEM subspaces are orthonormalised through a
Gram eigenbasis, and multi-channel projections use the dual basis
(Gram inverse) (DD-066).  Standard linear algebra.

## Exact discrete transparent boundary conditions (DTBC)

### Scalar DTBC on uniform feed lines (TEM: DD-054, TE/TM: DD-055)

On a uniform feed line, the longitudinal dynamics of one modal
amplitude in the FIT leapfrog is exactly a 1D leapfrog **discrete
Klein–Gordon chain** with modal Courant number $r$ (read off the
co-located pair product $M_\varepsilon M_\mu$) and mass
$q = \hat\omega_c \Delta t$ from the 2D eigenvalue ($q=0$ for TEM).
The exact transparent termination of the semi-infinite continuation
is obtained by $\mathcal Z$-transform: the outgoing characteristic
root $\lambda(z)$ yields the ghost relation
$\hat u_{K+1} = \lambda(z)\,\hat u_K$, realised in time domain as a
causal convolution over the boundary history whose kernel is computed
by contour integration and auto-extended past the run length, making
the boundary **exact (reflection-free to machine precision) within any
finite run** (`ports/modal/dtbc.py`).  Incident waves are prescribed
at the ghost plane through the same kernel.

This construction was derived in-house for the FIT/FDTD leapfrog
lattice (derivation in DD-054/DD-055; measured port floors −124 to
−250 dB).  The general concept of *exact discrete
transparent boundary conditions* — deriving the boundary kernel from
the discretised interior scheme rather than discretising a continuous
ABC — is established in the numerical-analysis literature, e.g. by Arnold,
Ehrhardt and Sofronov {cite}`arnold2003` and
Ehrhardt's work on discrete TBCs for Schrödinger-type equations
{cite}`ehrhardt1999`; discretely nonreflecting
boundary closures for finite-difference schemes of linear hyperbolic
systems are constructed by Rowley and Colonius
{cite}`rowleycolonius2000`.  None of these antecedents treats modal
FIT/FDTD waveguide ports; that application appears to be original to
Magnelio.

### CW true-mode ports for inhomogeneous lines (DD-056)

For inhomogeneous cross-sections measured at a single frequency, the
true discrete modes of the z-uniform feed section are computed as
eigenpairs of the **quadratic ζ-pencil**

$$
\left[\zeta^2 \mathbf D_{+1} + \zeta(\mathbf D_0 - \hat\sigma)
 + \mathbf D_{-1}\right]\varphi = 0
$$

built from the *production* system matrices at the port
(`ports/modal/zeta_pencil.py`), solved by sparse shift-invert ARPACK
on the linearised pencil.  Quadratic eigenvalue problems and their
linearisations are covered by the survey of Tisseur and Meerbergen
{cite}`tisseurmeerbergen2001`; computing waveguide
Bloch/propagation modes from a transfer/period formulation is
established practice in computational photonics and microwave theory
(no single originating publication is claimed).
The frequency-local closed-form $(r_\text{eff}, q_\text{eff})$ chain
fit (Hellmann–Feynman derivative matching so the scalar DTBC is exact
at the drive frequency) is an in-house derivation.

### Galerkin band-subspace DTBC for broadband runs (DD-057)

For pulsed broadband runs on inhomogeneous lines, the tracked
mode-family traces over the band span a low-rank W-orthonormal
subspace; the exterior half-line is Galerkin-projected onto it,
inheriting the palindromic symmetry that makes the projected lattice
lossless, and is closed by the exact small-system DTBC kernel
(`ports/modal/band_dtbc.py`).  Projection-based model order reduction
(Galerkin projection onto an SVD/POD subspace) is standard numerical
practice {cite}`benner2015` (survey reference; the
specific passive-by-construction boundary closure is in-house).

### Modal Mur fallback

Analytical-path modes (and, with explicit specs, inhomogeneous QTEM) are
terminated by a **first-order Mur absorbing boundary applied per mode
in modal-coefficient space**, with the per-mode phase velocity at the
mode-calculation frequency (`ports/modal/operator.py`).  The ABC is
Mur's {cite}`mur1981`.  The design of the
modal port
operator as a co-simulated 1D termination per mode follows the
modal-absorbing-port literature, specifically Luo and Chen
{cite}`luochen2007` (DD-047; higher-order Mur/Higdon
variants were evaluated and rejected, DD-069).

## Excitation and recording

Port excitation prescribes the incident modal amplitude at the ghost
plane (DTBC branch) or injects on the port plane (Mur branch); V/I are
recorded on a single co-located plane (DD-041).  Time signals are
smooth pulses (Gaussian and derived shapes, `signals/waveforms.py`) —
standard practice
{cite}`taflovehagness2005`.

## S-parameters

S-parameters are computed as **power waves** with $\sqrt{\mathrm W}$
normalisation, $a,b = (V \pm Z_0 I)/(2\sqrt{Z_0})$ per mode, from the
recorded modal V/I after Fourier transform
(`postprocessing/modal_sparameters.py`, DD-042/DD-078).  The
power-wave formalism is Kurokawa's {cite}`kurokawa1965`.
Two discrete-exactness refinements are
in-house (DD-063, DD-056): the a/b split de-staggers E and H
with the exact discrete factor $\lambda^{1/2}(z)$ and uses the exact
discrete wave impedance of the leapfrog chain (rather than the
continuum $Z_0$), which removes the $O(\beta\Delta z)$ staggering
leak; and CW measurements solve the exact 2×2 phasor system per port
(lock-in demodulation of incident and reflected discrete waves).

Excitation amplitudes are pinned to physical units at the source
(C = 1 convention, DD-085), so recorded V/I and monitor fields are in
SI units — an in-house calibration convention.
