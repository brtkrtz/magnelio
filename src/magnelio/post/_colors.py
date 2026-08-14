"""Shared material colour utilities for plotting modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnelio.materials.material import Material


def material_color(mat: "Material") -> tuple[float, float, float, float]:
    """Return (R, G, B, A) for a *Material*, using auto-colours when needed.

    Colour logic:

    * Explicit ``mat.color`` → use it together with ``mat.alpha``.
    * PEC → grey ``(0.65, 0.65, 0.65, 1.0)``.
    * Air / vacuum (ε = μ = 1) → fully transparent.
    * Dielectric → tint based on average relative permittivity.
    """
    if mat.color is not None:
        return (*mat.color, mat.alpha)
    if mat.is_pec:
        return (0.65, 0.65, 0.65, 1.0)
    # Air / vacuum: fully transparent
    if mat.epsilon == (1.0, 1.0, 1.0) and mat.mu == (1.0, 1.0, 1.0):
        return (1.0, 1.0, 1.0, 0.0)
    # Dielectric: tint based on average epsilon_r
    eps_avg = sum(mat.epsilon) / 3.0
    t = min((eps_avg - 1.0) / 11.0, 1.0)
    r = 0.4 + 0.5 * t
    g = 0.7 - 0.2 * t
    b = 0.9 - 0.6 * t
    return (r, g, b, 0.6)
