# Upgrading from 0.7.x

Release 0.8 changes the sign of one exponent.  The running transform
behind every frequency monitor summed $F(t)\,e^{+j\omega t}$, which
made the complex fields it handed out the *conjugates* of the phasors
everything else in the library speaks — the S-parameters, the port
signals, the waveform spectra, the textbook far-field formulas.  It now
sums $F(t)\,e^{-j\omega t}$, so there is one convention throughout:
frequency-domain fields are $e^{+j\omega t}$ phasors, and the real
field at time $t$ is $\mathrm{Re}\left(F\,e^{+j\omega t}\right)$.

Every complex field the library hands out is therefore the complex
conjugate of what 0.7 returned.  Magnitudes, power, energy, gain,
directivity and S-parameters are unchanged — none of them can tell the
two signs apart.  What changes is anything that reads a *phase*.

Stored runs need no attention: a project written by an earlier release
is converted as it is read, and a partial run resumes exactly.

## Quick reference

| 0.7.x | 0.8 | If you leave it alone |
|---|---|---|
| `mon.spectrum.cell_centred(["Ez"])["Ez"]` | the conjugate of those values | `abs(...)` unchanged; `np.angle(...)` changes sign |
| `pattern.E_theta`, `pattern.E_phi` | the conjugate of those values | `directivity`, `realized_gain`, `P_rad` unchanged; a polarisation handedness computed by hand comes out the other way |
| a transit phase $\int E_z\,e^{-jk z}\,dz$ for a particle toward $+z$ | $\int E_z\,e^{+jk z}\,dz$ | the two directions swap: what was the forward wave reads as the backward one |
| the `<field>_im` arrays of `export_paraview()` | sign-flipped: the imaginary part is the field at $\omega t = -90°$ | the arrows of the imaginary part point the other way |
| `snapshot(phase=φ)`, `show(phase=φ)`, `plot(phase=φ)` | unchanged in meaning: `φ` is $\omega t$ | a picture at a phase other than 0° or 180° is the mirror in time of what 0.7.0 drew (fixed in this release) |
| a project written by 0.7.x | read it as it is | nothing; the conversion is automatic and exact |

If your post-processing only takes magnitudes — a pattern, a field
plot, an S-parameter, a loss figure — nothing in your scripts needs to
change.

## What to look for in your own code

Anything that multiplies a monitor's complex field by a phase factor of
its own.  The two cases that occur in practice:

**A transit phase.**  A beam voltage, a coupling integral along a path,
a pickup summed along a line — the factor that follows the particle or
the wave.  For motion toward $+z$ it is now $e^{+jk z}$.  The how-to
{doc}`howto/plot_stripline_pickup_kicker` carries the corrected form.

**A polarisation decomposition.**  Left- and right-hand circular
components of a far-field pattern, or an axial ratio computed from
$E_\theta$ and $E_\varphi$.  The textbook expressions apply directly
now; in 0.7 they had to be conjugated first.

A quick way to check an existing script: compute the quantity both
ways.  If the two results differ only in the sign of a phase or in
which of two directions comes out larger, the script had the 0.7
convention baked in.

## The project store

Result files carry a `phasor_convention` attribute naming the
convention their bins are in.  A file without it was written by an
earlier release, and its complex data is conjugated as it is read — a
finished transform as much as a partial sum a resume continues.  The
conversion is exact, so a resumed run is bit-identical to an
uninterrupted one, and the store's schema version is unchanged: 0.7
projects stay readable.

A file naming a convention this release does not know is refused with a
`ProjectSchemaError` rather than read as if the stamp were absent.

## Why

The library grew two conventions and kept them apart by hand.  The
S-parameter path, the port signals' transform, the waveform spectra and
the CW lock-in were all $e^{+j\omega t}$; only the field monitors' own
running sum ran the other way, and the near-to-far-field transform
bridged the gap by conjugating its inputs at the entrance and its
result at the exit.  It worked, but it left the phase of a field
monitor and the phase of a port voltage at the same frequency as
conjugates of each other, which is a trap for anyone who compares
them — and it made the standard textbook formulas wrong as written for
anyone post-processing a pattern.  One sign in the accumulator removes
the split, the conjugations in the far-field transform, and the trap.
