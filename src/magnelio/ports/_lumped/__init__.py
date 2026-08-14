"""Discrete (lumped) port — spec, builder, operator.

Public API:

* :class:`PortSpecLumped` — declarative spec (label, endpoints, Z₀).
* :func:`build_lumped_port` — builder (spec + mesh + M_eps + dt → operator).
* :func:`build_lumped_element` — builder for the passive
  :class:`magnelio.circuit.LumpedElement` (DD-123; no excitation, no
  recording).
* :class:`PortOperatorLumped` — runtime operator implementing the
  :class:`magnelio.ports.base.Port` protocol.

Mirrors the structure of :mod:`magnelio.ports._modal` (specs in
``factory``, operator in ``operator``).
"""

from magnelio.ports._lumped.factory import (
    PortSpecLumped,
    build_lumped_element,
    build_lumped_port,
)
from magnelio.ports._lumped.operator import (
    LumpedElementOperator,
    PortOperatorLumped,
)

__all__ = [
    "PortSpecLumped",
    "build_lumped_port",
    "build_lumped_element",
    "PortOperatorLumped",
    "LumpedElementOperator",
]
