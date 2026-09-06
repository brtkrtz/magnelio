"""Project-store schema version — one constant, hard validation.

Schema 1.0 was the first published store format; 2.0 (DD-224 Phase B)
names runs by their excitations instead of one excited port channel,
records the mesh's element type, keys the port checkpoints per excited
mode, and retired the pre-DD-224 spellings the 1.0 readers still
accepted; 3.0 (DD-259) records the field monitors as the grid
quantities on the Yee positions of their region — one dataset per
staggered component, the region's grid lines and dual widths beside
them — carries the same for the frequency monitors' bins, and writes
no XDMF descriptor any more (the ParaView export converts on its own).
A 2.x store's monitor data cannot be read by a 3.0 reader, and none is
converted.  Additions that an older reader can ignore ride on 3.0
without a bump, as DD-140, DD-154 and DD-198 did: the field monitors'
region operators (DD-260, an ``operators`` group whose ``valid`` flag
says whether it was filled).  Every artefact the store writes (``project.json``,
``results.h5``, ``checkpoint.h5``, the setup recipe) is stamped with
:data:`SCHEMA_VERSION` and every reader validates it via
:func:`validate_schema` — an unknown or missing version fails loudly
instead of silently degrading, because a store is a contract from the
moment the format is public.
"""

from __future__ import annotations

SCHEMA_VERSION = "3.0"


class ProjectSchemaError(ValueError):
    """A project artefact carries an incompatible schema version."""


def validate_schema(found, where: str) -> None:
    """Raise :class:`ProjectSchemaError` unless *found* is the current schema.

    Parameters
    ----------
    found : str or None
        The version string read from the artefact (``None`` when the
        artefact predates versioning).
    where : str
        Human-readable location for the error message (file or section).
    """
    if found != SCHEMA_VERSION:
        raise ProjectSchemaError(
            f"{where}: schema version {found!r} is not supported "
            f"(current: {SCHEMA_VERSION!r}). This store was written by "
            f"another magnelio release — re-run the simulation to "
            f"regenerate it."
        )


__all__ = ["SCHEMA_VERSION", "ProjectSchemaError", "validate_schema"]
