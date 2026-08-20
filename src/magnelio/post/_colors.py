"""Shared material colour utilities for plotting modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnelio.materials.material import Material


def material_color(
    mat: "Material", color: tuple[float, float, float] | None = None
) -> tuple[float, float, float, float]:
    """Return (R, G, B, A) for a *Material*, using auto-colours when needed.

    Colour logic:

    * Explicit ``mat.color`` → use it together with ``mat.alpha``.
    * A *color* from an imported file → its hue, with the opacity the
      material would have got anyway (metals opaque, dielectrics
      translucent, vacuum invisible).  Opacity is a modelling statement,
      the hue is decoration, so only the hue is taken from the file.
    * PEC → grey ``(0.65, 0.65, 0.65, 1.0)``.
    * Air / vacuum (ε = μ = 1) → fully transparent.
    * Dielectric → tint based on average relative permittivity.
    """
    if mat is None:
        return (*(color or (0.65, 0.65, 0.65)), 1.0)
    if mat.color is not None:
        return (*mat.color, mat.alpha)
    if color is not None:
        return (*color, material_color(mat)[3])
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


def shape_color(shape) -> tuple[float, float, float, float]:
    """Return (R, G, B, A) for a shape, honouring an imported colour.

    Thin wrapper around :func:`material_color` that picks up the display
    colour a shape brought along from a CAD file
    (:class:`~magnelio.geo.ImportedSolid`); shapes built with the CSG
    API carry none and are coloured by their material alone.
    """
    return material_color(getattr(shape, "material", None), getattr(shape, "color", None))
