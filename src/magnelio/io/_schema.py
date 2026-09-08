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
says whether it was filled) and the phasor convention of the stored
DFT bins (DD-268, :data:`PHASOR_CONVENTION`).  Every artefact the store writes (``project.json``,
``results.h5``, ``checkpoint.h5``, the setup recipe) is stamped with
:data:`SCHEMA_VERSION` and every reader validates it via
:func:`validate_schema` — an unknown or missing version fails loudly
instead of silently degrading, because a store is a contract from the
moment the format is public.
"""

from __future__ import annotations

SCHEMA_VERSION = "3.0"

#: Phasor convention of the complex DFT bins in a result file.  Stamped
#: on ``fields_freq.h5``, ``wall_loss.h5`` and ``far_field.h5``; absent
#: on files written before the convention was settled, whose bins are
#: the complex conjugates of these (DD-268).
PHASOR_CONVENTION = "exp(+jwt)"


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


def stored_phasors_conjugated(attrs, where: str) -> bool:
    """Whether the complex bins of a result file must be conjugated on read.

    The running DFT summed ``e^{+jwt}`` up to and including v0.7.0, so
    its bins were the conjugates of every other phasor of the library;
    it sums ``e^{-jwt}`` now and the files say so.  A file without the
    stamp is one of the older ones, and conjugating its bins is exact —
    of a finished transform as much as of a partial sum a resume
    continues.

    Parameters
    ----------
    attrs : mapping
        The HDF5 file attributes to read the stamp from.
    where : str
        Human-readable location for the error message.

    Returns
    -------
    bool
        ``True`` when the file predates the stamp and its complex data
        is conjugated with respect to this release.

    Raises
    ------
    ProjectSchemaError
        The stamp is present but names a convention this release does
        not know.
    """
    found = attrs.get("phasor_convention")
    if found is None:
        return True
    found = found.decode() if isinstance(found, bytes) else str(found)
    if found != PHASOR_CONVENTION:
        raise ProjectSchemaError(
            f"{where}: phasor convention {found!r} is not supported "
            f"(current: {PHASOR_CONVENTION!r}). This store was written by "
            f"another magnelio release — re-run the simulation to "
            f"regenerate it."
        )
    return False


__all__ = [
    "PHASOR_CONVENTION",
    "SCHEMA_VERSION",
    "ProjectSchemaError",
    "stored_phasors_conjugated",
    "validate_schema",
]
