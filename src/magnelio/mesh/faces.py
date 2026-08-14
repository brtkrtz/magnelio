"""Bounding-box face identification for the structured hex mesh.

:class:`BoxFace` names one face of the simulation bounding box and
carries the face-local ``(u, v)`` axis convention used by port planes,
boundary handling and the material-matrix assembly.
"""

from __future__ import annotations

from enum import Enum


class BoxFace(Enum):
    """Identifies one face of the simulation bbox by its outward normal.

    The local ``(u, v)`` axes per face are chosen so ``u × v`` points
    into the simulation domain (matching the Poynting convention for
    forward modal waves).
    """

    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"

    @property
    def normal_axis(self) -> int:
        """Global axis index perpendicular to the face (0=x, 1=y, 2=z)."""
        return _AXIS_OF_FACE[self]

    @property
    def is_max(self) -> bool:
        """True for X_MAX / Y_MAX / Z_MAX faces."""
        return self.value.endswith("_max")

    @property
    def inward_sign(self) -> int:
        """+1 for MIN faces (inward = +axis); −1 for MAX faces."""
        return -1 if self.is_max else 1

    @property
    def u_axis(self) -> int:
        """Global axis index for the local-u tangential direction."""
        return _UV_AXES[self][0]

    @property
    def v_axis(self) -> int:
        """Global axis index for the local-v tangential direction."""
        return _UV_AXES[self][1]


_AXIS_OF_FACE: dict[BoxFace, int] = {
    BoxFace.X_MIN: 0,
    BoxFace.X_MAX: 0,
    BoxFace.Y_MIN: 1,
    BoxFace.Y_MAX: 1,
    BoxFace.Z_MIN: 2,
    BoxFace.Z_MAX: 2,
}

# Per-face (u_axis, v_axis) chosen so that u × v points inward.
_UV_AXES: dict[BoxFace, tuple[int, int]] = {
    BoxFace.X_MIN: (1, 2),  # y × z = +x
    BoxFace.X_MAX: (2, 1),  # z × y = −x
    BoxFace.Y_MIN: (2, 0),  # z × x = +y
    BoxFace.Y_MAX: (0, 2),  # x × z = −y
    BoxFace.Z_MIN: (0, 1),  # x × y = +z
    BoxFace.Z_MAX: (1, 0),  # y × x = −z
}

__all__ = ["BoxFace"]
