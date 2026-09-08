"""The Poynting vector as a derived view of a recorded field.

``S = E × H`` needs no material operators and no solver state — only
the two fields at one and the same point.  The staggering is what makes
it work: the six components live on six different Yee positions, so the
cross product is formed on the cell centres both fields interpolate to
(:func:`~magnelio.fields._interp._interp_to_cell_centres`), never on
the raw samples.

Two readings, decided by the dtype of the frame, following the reading
:meth:`~magnelio.fields.FieldState.energy` and
:meth:`~magnelio.fields.FieldState.flux` already established (DD-260):

* a **real** frame is an instant of a march and gives the instantaneous
  power density ``E(t) × H(t)`` [W/m²];
* a **complex** frame is an RMS phasor — what a monitor's spectrum
  holds, per 1 W CW — and gives the time-averaged density
  ``Re(E × H*)``, with no further factor of a half.  The full complex
  product ``E × H*`` is available on request; its imaginary part is the
  reactive power density of the stored near field.
"""

# Design: DD-270 (the Poynting vector as a derived view).

from __future__ import annotations

import numpy as np

_E = ("Ex", "Ey", "Ez")
_H = ("Hx", "Hy", "Hz")
_S = ("Sx", "Sy", "Sz")


def missing_fields(recorded) -> KeyError:
    """The error for a Poynting request that has no magnetic (or electric) field."""
    have = sorted(recorded)
    return KeyError(
        f"the Poynting vector needs all six field components at the same "
        f"point; recorded: {have or 'none'}.  Record both fields — "
        f"fields=['E', 'H'] on the monitor — and the vector follows."
    )


def check_available(recorded) -> None:
    """Raise unless every E and H component is among *recorded*."""
    have = set(recorded)
    if not set(_E + _H) <= have:
        raise missing_fields(recorded)


def cross(centred: dict, *, complex_product: bool = False) -> np.ndarray:
    """``E × H`` from cell-centred components, stacked on a trailing axis.

    Parameters
    ----------
    centred : dict
        Cell-centred physical components, the six names as keys; each
        array of the same shape (leading frame axes are carried
        through unchanged).
    complex_product : bool, default False
        For a complex (phasor) frame return ``E × H*`` itself instead
        of its real part.  Ignored for a real frame.

    Returns
    -------
    np.ndarray
        Shape ``(*array shape, 3)`` [W/m²].
    """
    check_available(centred.keys())
    e = [np.asarray(centred[c]) for c in _E]
    h = [np.asarray(centred[c]) for c in _H]
    is_complex = any(np.iscomplexobj(a) for a in e + h)
    if is_complex:
        h = [np.conj(a) for a in h]
    out = np.stack(
        [
            e[1] * h[2] - e[2] * h[1],
            e[2] * h[0] - e[0] * h[2],
            e[0] * h[1] - e[1] * h[0],
        ],
        axis=-1,
    )
    if is_complex and not complex_product:
        return np.real(out)
    return out


def as_components(centred: dict, *, complex_product: bool = False) -> dict[str, np.ndarray]:
    """The same product as :func:`cross`, keyed ``"Sx"``, ``"Sy"``, ``"Sz"``.

    The form the plotting and viewer paths speak, where a component is
    a name and a group is its three.
    """
    s = cross(centred, complex_product=complex_product)
    return {name: s[..., a] for a, name in enumerate(_S)}


# ── the name in the component vocabulary ─────────────────────────────────

COMPONENTS = _S
GROUP = "S"
SOURCES = _E + _H


def split(names) -> tuple[list[str], list[str]]:
    """Separate requested names into recorded fields and Poynting components.

    Raises for a name that is neither.
    """
    fields, poynting = [], []
    for name in names:
        if name in _S:
            poynting.append(name)
        elif name in SOURCES:
            fields.append(name)
        else:
            raise KeyError(f"component must be one of {SOURCES + _S}; got {name!r}")
    return fields, poynting


def add_to(centred: dict, names, *, complex_product: bool = False) -> dict:
    """Add the requested Poynting components to a dict of cell-centred fields.

    *centred* must hold all six field components; the Poynting entries
    are appended and the source entries left untouched, so a caller
    that asked for both gets both.
    """
    if not names:
        return centred
    derived = as_components(centred, complex_product=complex_product)
    return {**centred, **{n: derived[n] for n in names}}
