"""Symmetry reporting for lumped ports and elements.

A :class:`LumpedPortReport` records which declared symmetry planes cut
a lumped edge chain, mirroring the modal ``PortOperatorReport`` shape
(the recorder and the excitation scaling read both through the same
``port_report`` attribute).  The chain relation to each plane is one of

``"crossing"``
    The chain runs along the plane normal and the plane bisects it —
    the full-model element is cut in series (an electric plane only;
    a normal current mirrors anti-parallel across a magnetic plane, so
    a magnetic crossing has no full-model counterpart and is rejected
    by the builder).

``"containment"``
    The chain lies in the plane — the full-model element is cut in
    parallel (a magnetic plane only; inside an electric wall the chain
    edges would be shorted).

The raw solver quantities stay half-model; the scale properties below
restore full-model semantics at the established consumers
(``PortSignalRecorder`` reads ``power_wave_full_scale``, the analysis
reads its inverse for the injection).
"""

# Design: DD-172 (lumped ports/elements on symmetry planes).

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LumpedPortReport:
    """Symmetry planes cutting a lumped edge chain.

    Parameters
    ----------
    symmetry_faces : tuple
        ``((face, wall_kind, relation), ...)`` triples, e.g.
        ``(("zmin", "PEC", "crossing"),)``.
    """

    symmetry_faces: tuple = ()

    @property
    def z_internal_scale(self) -> float:
        """Full-model → half-model scale for the internal impedance.

        A series cut leaves half the device in the meshed half
        (``Z/2``); a parallel cut leaves one of two parallel branches
        (``2·Z``).  The builder applies this factor to the user's
        full-model ``Z0`` / companion element.
        """
        scale = 1.0
        for _, _, relation in self.symmetry_faces:
            scale *= 0.5 if relation == "crossing" else 2.0
        return scale

    @property
    def z_full_scale(self) -> float:
        """Half-model → full-model impedance scale (informative)."""
        return 1.0 / self.z_internal_scale

    @property
    def power_wave_full_scale(self) -> float:
        """Half-model → full-model scale for power-wave amplitudes.

        Each cutting plane halves the power the meshed half accounts
        for (series cut: half the voltage; parallel cut: half the
        current), so recorded amplitudes scale by √2 per plane and the
        excitation by 1/√2 — identical to the modal-port convention,
        which is what lets the shared recorder / injection plumbing
        apply both factors unchanged.
        """
        return 2.0 ** (0.5 * len(self.symmetry_faces))
