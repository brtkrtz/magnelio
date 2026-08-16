# Dispersive and lossy materials

## Pole-residue material model

All frequency-dependent permittivities (and, mirrored, permeabilities)
are expressed in the **complex-conjugate pole-residue form**

$$
\varepsilon(\omega) = \varepsilon_\infty
 + \sum_p \frac{r_p}{j\omega - a_p},
$$

with real poles and conjugate pairs stored once
(`materials/dispersion.py`, DD-083).  Using this single general form —
with Debye, Lorentz, Drude and wideband-laminate models as
*constructors* on it rather than separate update schemes — follows
Han, Dutton and Fan {cite}`han2006`, who introduced
the complex-conjugate pole-residue formulation for dispersive FDTD.

The named constructors realise classical material models: Debye
relaxation, Lorentz resonance, Drude conduction (all textbook,
{cite}`taflovehagness2005`) and the wideband
causal laminate model of Djordjević, Biljić, Likar-Smiljanić and
Sarkar {cite}`djordjevic2001`.

**Passivity is enforced at construction** (left-half-plane poles plus
band-sampled $\varepsilon'' \ge 0$), acting as the mandatory
acceptance filter for fitted data — an in-house design rule motivated
by the well-known instability of non-passive dispersive FDTD models
(the Kramers–Kronig/passivity background is classical
{cite}`landaulifshitz_ecm`).

## Auxiliary differential equation (ADE) update

The solver realises the pole sum with the **auxiliary differential
equation method**: one polarisation-current state per pole on the
dispersive edges only, advanced with the **trapezoidal rule** and
folded semi-implicitly into the E update so the field kernels stay
untouched (`solver/dispersion.py`, DD-084).  The ADE technique is due
to Kashiwa and Fukai {cite}`kashiwafukai1990` and
Joseph, Hagness and Taflove {cite}`joseph1991`; the
textbook treatment is {cite}`taflovehagness2005`.
Discretising the auxiliary equations by the trapezoidal/bilinear
transform — A-stable for every passive pole, so the CFL limit stays
the $\varepsilon_\infty$ one — is the Möbius/bilinear-transform
approach of Pereda et al. {cite}`pereda2002`.

Two in-house exactness properties are used as structural
gates: the Drude DC pole ($a_p = 0$) reduces bit-exactly to
the standard semi-implicit conductor update with
$\sigma = \varepsilon_0 r_p$; and the magnetic mirror
($\mu(\omega)$, DD-089) is the *same operator* under the substitution
$M_\varepsilon \to M_\mu$, $C^{\mathsf T}\hat h \to -C\hat e$, gated
by the exact $\mu$-Drude-DC $\equiv \sigma^*$ reduction.

## Magnetic loss $\sigma^*$

Magnetic conductivity (the $\sigma^*$ term in the Faraday update) is
carried as a diagonal $M_{\sigma^*}$ with the same semi-implicit
time-averaged update as electric $\sigma$ (DD-081) — standard
FDTD/FIT practice {cite}`taflovehagness2005`.

## Vector fitting of measured data

Tabulated $\varepsilon(f)$ data are fitted onto the pole-residue form
with an in-repo implementation of **vector fitting**
(`materials/vector_fit.py`, DD-086), the pole-relocation iteration of
Gustavsen and Semlyen {cite}`gustavsensemlyen1999`,
including the standard unstable-pole flipping rule.  Passivity is
*not* enforced by the fit; the `DispersionModel` constructor is the
acceptance filter (see above).  The implementation was written from
the publication; no third-party vector-fitting code is included.
