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

The port report labels the impedance of an inhomogeneous cross-section
`(quasi-static)`: it is frequency-flat by construction, whereas the
impedance the discrete wave actually carries rises with frequency on a
dispersive line — by 3 to 7 % at 15 GHz on the tutorial microstrip —
and that difference, not the absorber, is where most of the
$-30$ dB-class $|S_{11}|$ floor of the default pipeline comes from.  On
a homogeneous line the label stays `(numerical)`.

**Coupled lines.**  With more than one signal conductor above the
ground — an edge-coupled microstrip pair, a stripline pickup, a
multi-wire bus — the per-conductor Laplace solutions are the
*conductor* basis of the line, not its *mode* basis: on an
inhomogeneous cross-section every voltage pattern travels at its own
speed, and only the eigen-patterns of the multiconductor telegrapher
equations propagate without exchanging energy.  The port therefore
returns those eigen-patterns.  From the per-conductor fields it forms
the two per-unit-length capacitance matrices — $\mathbf C$ with the
actual dielectric and $\mathbf C_0$ with the conductors in vacuum —
and solves $\mathbf C\,\mathbf v = \varepsilon_{\text{eff}}\,
\mathbf C_0\,\mathbf v$, the quasi-static form of the modal
decomposition of $\mathbf L\mathbf C$ with $\mathbf L =
\mu_0\varepsilon_0\mathbf C_0^{-1}$ {cite}`paul2008`.  Each
eigenvector is a conductor-voltage pattern (the even and odd modes of
a symmetric pair), its eigenvalue the modal $\varepsilon_{\text{eff}}$,
and its impedance $Z_0 = 1/(c\sqrt{C'_v C'_{0,v}})$ with the modal
capacitances $C'_v = \mathbf v^\top\mathbf C\,\mathbf v$ for the
unit-Euclidean pattern $\mathbf v$ — for a symmetric pair exactly
$Z_{0e}$ and $Z_{0o}$.  Dimensioning a coupled section from these two
modes alone — a pair of lines, or one pair of a Lange coupler's
fingers — is the subject of the how-to guides *Coupled-line
directional coupler* and *Lange coupler*.  Channels are ordered by descending
$\varepsilon_{\text{eff}}$ and are orthogonal in the port's
capacitance-corrected mass.  A single signal conductor is the $1
\times 1$ case of the same construction.  For a homogeneous filling
the pencil degenerates (all patterns share one speed), and the
channels are the capacitance-matrix eigenmodes of DD-066 instead.

### Mode classification and multi-mode merge

TE/TM and TEM/QTEM branches are merged into one unified multi-mode
port; degenerate multi-TEM subspaces are orthonormalised through a
Gram eigenbasis, and multi-channel projections use the dual basis
(Gram inverse) (DD-066).  Standard linear algebra.

### Ports in absorbing walls

A radiating structure is fed through a guide that reaches the domain
wall: the neck of a horn, a coax entering the box.  The wall is
absorbing, the port sits in the guide's cross-section on it (a window
port with ``corners``), and two things make that consistent (DD-198).
The absorber is switched off in the columns behind the window over its
whole depth, so the mode injected at the port travels a uniform,
lossless guide into the domain and the reflected wave meets the port's
own termination — the transparent-boundary assumptions above hold
because the feed *is* uniform there.  And the window must be enclosed
by conductor on the port slab: the lateral edge of the absorber
switch-off then lies on metal, where it cannot scatter.  A port that
covers the whole absorbing face, or a window whose ring lies in free
space, is refused with a pointer to these rules.  A printed line —
a microstrip feeding a patch array — satisfies them through a short
shielded launch: two walls and a roof around the trace where it meets
the wall, the window in the launch's cross-section, the way a
connector body encloses the line (how-to *patch array*).  The
far-field monitor knows about such feeds (far-field chapter).

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

### Which termination a channel gets, and how to see it

The exact termination is not a setting — it is a certificate.  A
channel earns it when the meshed feed section really is the uniform
discrete chain the derivation above assumes, which is tested in two
stages before the run: the **pair-product gate** measures how far the
co-located products $M_\varepsilon M_\mu$ spread across the feed
cross-section (a weighted RMS), and the **slab gate** compares every
mass entry on the port plane with its continuation into the first feed
cells.  A channel that fails either one keeps working — it falls back
to the modal Mur absorber — but its reflection floor rises from below
$-100$ dB to the order of $-30$ dB.

The threshold of the first gate is a *reflection budget* (DD-229).
The termination is exact for the weighted-mean chain, so a spread
$\delta$ across the cross-section leaves a residual mismatch, and that
mismatch was measured through the production solver on three fixtures
and two shapes of defect.  The worst case rises linearly, at about a
seventh of the spread; the gate uses the cruder bound
$|\Gamma| \le \delta$ and accepts up to $\delta = 2\cdot 10^{-6}$, so a
certified channel contributes at most $-114$ dB — well below the
acceptance line the port floors are held to.  Every channel therefore
publishes what its own cross-section costs it:

$$
\texttt{chain\_floor\_db} = 20\log_{10}\delta .
$$

The shape of the non-uniformity matters more than its size, which is
why the bound is deliberately crude.  A smooth transversal tilt is
antisymmetric against a symmetric mode, its first-order overlap
vanishes, and the reflection falls to second order; a *localised*
defect — one cell of the cross-section out of step, which is what
geometric tolerance produces — keeps first order and reflects orders
of magnitude more at the same measured spread.  A single scalar cannot
tell the two apart, so the gate assumes the worse one.

Failing a gate is two different events, and the reporting separates
them.  A cross-section that was *meant* to be uniform and missed the
budget is the surprising one, and it warns: the port, the channel, the
measured deviation, and where the mesh can explain it, the conformal
ladder behind the offending faces (DD-228).  A cross-section that is
genuinely inhomogeneous is not a defect; a quasi-TEM line deviates at
the material-contrast level and never qualified for the scalar chain
in the first place, so the answer there is a different port model, not
a mesh fix, and Magnelio does not editorialise about it every run.

That second case is worth stating plainly, because the default hides
how much is on the table.  On an inhomogeneous line the default
`port_model="modal"` terminates with the first-order absorber and
floors around $-26$ to $-39$ dB.  The same line on `port_model="auto"`
— which builds the certificates, sees the fallback and switches the
run to the band pipeline — floors at $-147$ to $-168$ dB on a shielded
microstrip.  So the default gives up 60 to 110 dB of reflection floor
for the cost of the band pipeline — the low end on a long run, the
high end on a short one; if the port matters, change it.

That spread is the second thing to know about the band floor: how deep
it lands depends on how long the run is.  The exact boundary carries
the whole recorded history rather than a few past steps, and its floor
drifts upward as that history grows — measured at roughly five to
seven decibels per doubling of the number of time steps.  The figures
above are what a run of the length the port floors are certified at
reaches; a much longer broadband sweep on the same model floors tens
of decibels above them.

Changing the port model changes more than the floor, and the
differences are worth knowing before the run rather than after.  The
band pipeline tracks the mode family over a band, but the
*measurement* axis is not confined to that band.  On a quasi-TEM line
the fundamental is the static mode of the cross-section, so its
excitation direction is continued below the tracked band and closed
with the static field solution; the drive then needs no roll-off room
at the bottom and the pulse duration follows the measurement span
instead of $1/f_\text{start}$.  A default axis — which starts at
$f_\text{max}/n_\text{freq}$ — runs.

Two limits remain.  The reflection floor of the boundary degrades
toward DC: on a coarse two-port fixture the same run measures $-91$ dB
at 34 MHz against $-117$ to $-140$ dB over the rest of the axis, so a
measurement whose lowest points carry the answer should still not
start lower than it needs to.  And a fundamental with a *cut-off* — a
hollow waveguide mode — has no static limit to be continued to, so
there the pulse duration still grows as $1/f_\text{start}$ and the
auto-sizing refuses an axis that would make the run disproportionate,
naming the axis start that fits.

The record has a fixed length, so `energy_stop_db`
and signal-decay stops do not apply to it.  The recorded channels are
subspace projections whose incident/outgoing split is defined per
frequency, so `result.a()` and `result.b()` are unavailable; the
S-parameters and the raw V/I are.  Everything else is unchanged —
including writing to a project store, from which the S-matrix is
re-derived on read exactly as on the default path, and continuing an
interrupted run with `resume()`.  The absorbing boundary here
remembers the *whole* record rather than a few past steps, so its
memory is checkpointed in full; a continued run is bit-identical to an
uninterrupted one of the same length.

None of this has to be looked up.  Whenever a run terminates a
channel with the Mur fallback, the analysis prints a short balance
sheet once: which channels fell back and what the cross-section
measured, the floor they trade away against the runtime and the power
waves they keep, and what `port_model="band"` costs — a few lines; the
detail is this section.

Either way the decision is published per channel, so it can be
inspected before a run is paid for:

```python
for name, report in analysis.solve_ports().items():
    for mode in report.modes:
        print(name, mode.name, mode.termination,
              mode.chain_spread, mode.chain_floor_db)
```

`termination` is `"dtbc"` or `"mur"` and `chain_spread` is the
cross-section measurement behind that decision.  `chain_floor_db` is
the reflection bound the spread implies, and it is published only
where it means something: it is `None` on a channel the first-order
absorber terminates, because there the floor is the absorber's and not
the cross-section's, and `None` again where the measurement does not
apply at all (a mode with a closed-form field evaluator is ineligible
by construction).  So on an inhomogeneous line, read `chain_spread` —
the number that explains *why* the channel fell back — and take the
floor from the absorber.  `print(report)` shows the same on one line
per mode.

The usual cause of a withheld certificate is a feed that is not
translation-invariant along the port normal — a taper, a bend or a
dielectric step too close behind the port plane.  The remedy is to
move the port back into the uniform part of the feed.  Where the feed
*is* uniform by construction and the gate still trips, the deviation
is geometric tolerance in the solid: a body built by mirroring or
unioning carries a looser tolerance than the shape it was built from,
and the conformal cross-section inherits it.

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

### Band ports: one decomposition per frequency (DD-235)

A band-subspace port does not record modal amplitudes whose
incident/outgoing split is fixed once for the run.  Its recording
channels are subspace projections, and the split is defined per
frequency: at every point of the measurement axis the true discrete
modes of the port cross-section are solved on the stored inward chain
(the ζ-pencil above), their exact V/I responses through the port's
*fixed* recording profiles are synthesised, and the recorded spectra
are decomposed by one joint least-squares system written over all
channels $c$ and all modes $j$ propagating at that frequency at once,

$$
V_c(f) = \sum_j a_j v^{\text{in}}_{cj} + b_j v^{\text{out}}_{cj},
\qquad
I_c(f) = \sum_j a_j i^{\text{in}}_{cj} + b_j i^{\text{out}}_{cj},
$$

with $S_{c} = b_c / a_\text{excited}$.  Solving the channels jointly
rather than one at a time is what makes the extraction exact: the
recording profiles are frozen when the port is built and are not the
modes of any single frequency, so every channel sees a mixture of
every mode present, and only the joint system unmixes them — provided
the basis it is written in contains them all.

Three things are worth keeping apart.  A **channel** is a recording
slot of the port, one per requested mode (`n_modes`), with a profile
fixed at build time.  A **propagating mode** is what the cross-section
actually carries at the frequency being evaluated; their number grows
along the axis, each new one starting at its **cut-on**.  Modes are
matched to channels per frequency by a weighted overlap test, and
nothing guarantees the two counts agree.

They can disagree in either direction, and the two cases behave
oppositely.  A channel without a mode is *visible*: it finds no
overlap, takes no assignment, its S stays NaN at that frequency, and
the channels that were matched are unaffected (measured at $10^{-15}$
relative, roundoff).  This is the ordinary evanescent case — declaring
more channels than the axis excites costs a NaN column and nothing
else.  A mode without a channel is *not* visible: nothing is skipped,
so nothing goes NaN, and the fit attributes the missing mode's
recorded content to the modes that remain.  The result is a biased
S-parameter that looks like a converged one.  Measured on a
two-family fixture whose port was truncated to `n_modes = 1` above its
second cut-on at 8.4465 GHz, $S_{11}$ came out 23 to 30 % wrong, with
the contamination at $-129$ dB when the unaccounted mode carried the
same amplitude as the fundamental and rising exactly 20 dB per decade
of that amplitude ratio.  Read that level as the mechanism and its
scaling law, not as a bound: its coefficient is the biorthogonality
defect between one port's recording profiles and its true modes
(about $6\cdot10^{-7}$ on that cross-section), and a port whose dual
basis is less clean sits correspondingly higher.

The rule for a band run follows: `n_modes` must cover every mode that
propagates *anywhere* on the measurement axis — which is not the same
as every mode of interest, and not the same as the tracked band —
or the axis must stay below the next cut-on of the cross-section.
Magnelio counts the modes it finds against the modes it assigns at
each axis point and warns once per port, naming how many frequencies
are affected and their span.  The warning does not repair the bias:
the port genuinely carries fewer channels than its cross-section
carries modes, and only a rebuilt port or a shorter axis fixes that.
What it does is keep a model error from reading as a result.

### Reference-plane shift (de-embedding, DD-187)

`result.deembed({"port1": d})` returns the S-matrix referenced at
planes shifted a distance $d$ from the port planes into the domain
(negative distances move outward): every S-parameter touching a
shifted port is multiplied by the inverse line propagation factor over
$d$ — reflections twice, transmissions once per shifted end.  This is
the classical reference-plane transformation
{cite}`pozar2012`; the in-house refinement is *which* propagation
factor is removed.

Wherever the run certified a channel's discrete line parameters, the
shift uses the **exact discrete dispersion of the feed chain** — the
same characteristic root $\lambda(z)$ the transparent boundary is
built from — evaluated on the unit circle, so passband magnitudes are
untouched exactly and the removed phase is exactly the phase the grid
applied.  De-embedding a uniform feed line therefore cancels it to
the accuracy floor of the run itself (measured: −120 dB TEM, −67 dB
TE10 at 8 cells/λ), whereas the textbook continuum $e^{-\gamma d}$
would leave the grid-dispersion gap behind — degrees of phase on
coarse meshes, silently attributed to the device under test.
Channels without certified line parameters fall back to the mode's
continuum $\gamma(\omega)$.

For a **quasi-TEM channel** — microstrip, CPW, any inhomogeneous
cross-section, terminated by modal Mur in the default pipeline — that
fallback is the quasi-static $\gamma$ of the 2D Laplace mode,
$\varepsilon_{\text{eff}} = C'/C'_0$ taken frequency-flat.  The
physical dispersion of the line (its $\varepsilon_{\text{eff}}$
rising with frequency) is therefore *not* removed and stays in the
de-embedded matrix: on a 16 mm microstrip on 0.8 mm
$\varepsilon_r = 4.3$ the residual S21 phase after de-embedding the
full length is about 1°, 8° and 22° at 5, 10 and 15 GHz, growing with
substrate thickness.  Keep quasi-TEM feed lines short when the
reference plane matters, or compare raw S-parameters.

The shift assumes the port cross-section continues over the shifted
length.  Below its cut-off a channel's factor grows as $e^{+\alpha
d}$; those bins keep the diagnostic character the raw values have.
Lumped ports carry no feed line and cannot be de-embedded.

### What a Touchstone export covers (DD-184)

`to_touchstone()` and `to_skrf()` export the square sub-matrix over
the **excited** channels: one Touchstone port per channel, so a
multi-mode port occupies one port per mode.  Channels that were
observed but not excited are dropped from both the rows and the
columns; nothing is padded.

Dropping them is sound because an unexcited channel is not an open
circuit.  Every port carries its own reflection-free boundary
throughout the run, so the omitted channels are *matched* — which is
exactly the termination condition the definition of S-parameters asks
for.  The export is the network seen with those channels terminated,
the same quantity a vector network analyser measures with its unused
ports on 50 Ω loads.  Exciting one port of a two-port and writing the
reflection as a `.s1p` is therefore a valid export, not a truncated
one.

What the reduction does *not* carry is mode conversion at a port that
is itself exported.  If `port1` is solved with three modes, only mode
0 is excited, and modes 1 and 2 propagate inside the exported band,
then power scattered from mode 0 into those modes leaves the matrix:
the file looks like a complete two-port while the component is not
one.  Such an export warns, naming the port and the cut-off above
which the omitted modes propagate.  Evanescent omitted modes draw no
warning — solving for more modes than one excites, so that the
evanescent content is represented at the port plane, is ordinary
practice.

Pass `channels=` to select the sub-network explicitly, e.g.
`channels=["port1", "port3"]` to cut a two-port out of a fully
excited three-port.

The `.sNp` extension must agree with the number of exported channels.
Touchstone 1.x records the port count nowhere else — the file body
has no field for it — so a `.s6p` holding two-port rows is not merely
misnamed but unreadable; a mismatch raises rather than writes.  A
path given without an extension gets the matching one, so
`to_touchstone("wr90")` writes `wr90.s2p`.
