"""Excellon reader — the holes of a board, as plain records.

Drill data is the second half of a fabrication export, and it is the
half that decides what a board is in the third dimension: which holes
are plated (a barrel of copper joining layers) and which are just
absent material, and how far through the stack each one goes.

The format is old and loosely specified, and its traps are different
from Gerber's.  Two matter here.  Coordinates may be written without a
decimal point, and the zero-suppression keyword is named the opposite
way round from Gerber's: ``LZ`` means the *leading* zeros are present,
so it is the *trailing* ones that were dropped.  And slots come in two
spellings — a ``G85`` between two points, or a routed path with the
tool lowered by ``M15`` and raised by ``M16`` — with writers choosing
freely between them, so both have to be understood.

The plating and layer span of a file are carried in the X2 attributes
that ride along in ``; #@!`` comments; without them a hole cannot be
told from a via, which is why they are read here rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_UNITS = {"METRIC": 1e-3, "INCH": 0.0254}

# Digits before and after the point when coordinates come without one.
_DEFAULT_DIGITS = {"METRIC": (3, 3), "INCH": (2, 4)}


@dataclass(frozen=True)
class Hit:
    """A drilled hole."""

    at: tuple[float, float]
    diameter: float


@dataclass(frozen=True)
class Slot:
    """A routed slot — a hole swept from *start* to *end*."""

    start: tuple[float, float]
    end: tuple[float, float]
    diameter: float

    @property
    def at(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2.0, (self.start[1] + self.end[1]) / 2.0)


Hole = Hit | Slot


@dataclass(frozen=True)
class DrillFile:
    """One drill file, played back.

    Attributes
    ----------
    holes : tuple
        The holes the file drills, in file order.
    plated : bool or None
        Whether the file's holes are plated through, if the file says.
    span : tuple of int or None
        Copper layer numbers the holes run between, if the file says.
    unit : str
        ``"METRIC"`` or ``"INCH"``; coordinates are already in meters.
    attributes : dict
        File attributes read from the X2 comments.
    """

    holes: tuple[Hole, ...]
    plated: bool | None
    span: tuple[int, int] | None
    unit: str
    attributes: dict[str, tuple[str, ...]] = field(default_factory=dict)


_COORD_RE = re.compile(r"([XY])([+-]?[\d.]+)")
_TOOL_RE = re.compile(r"T(\d+)")
_DIAMETER_RE = re.compile(r"C([\d.]+)")
_FORMAT_RE = re.compile(r"0*\.?0*$")


class _Reader:
    def __init__(self, source: str) -> None:
        self.source = source
        self.line = 0

    def fail(self, message: str) -> ValueError:
        return ValueError(f"{self.source}, line {self.line}: {message}")


def _file_function(values: tuple[str, ...]) -> tuple[bool | None, tuple[int, int] | None]:
    """Plating and layer span from a ``.FileFunction`` attribute.

    The attribute reads ``(Non)Plated,<from>,<to>,<kind>``.  Anything
    less is treated as absent rather than as a default: a wrong guess
    turns a via into a hole or the other way round.
    """
    if not values or values[0] not in ("Plated", "NonPlated"):
        return None, None
    plated = values[0] == "Plated"
    if len(values) < 3:
        return plated, None
    try:
        start, end = int(values[1]), int(values[2])
    except ValueError:
        return plated, None
    return plated, (min(start, end), max(start, end))


class _Interpreter:
    def __init__(self, reader: _Reader) -> None:
        self._reader = reader
        self.unit: str | None = None
        self.scale: float | None = None
        self.trailing_present: bool | None = None
        self.digits: tuple[int, int] | None = None
        self.tools: dict[int, float] = {}
        self.current: int | None = None
        self.point: tuple[float, float] = (0.0, 0.0)
        self.in_header = False
        self.routing = False
        self.pen_down = False
        self.holes: list[Hole] = []
        self.attributes: dict[str, tuple[str, ...]] = {}
        self.done = False

    def fail(self, message: str) -> ValueError:
        return self._reader.fail(message)

    # -- numbers ------------------------------------------------------

    def _set_unit(self, name: str, parts: list[str]) -> None:
        self.unit = name
        self.scale = _UNITS[name]
        if self.digits is None:
            self.digits = _DEFAULT_DIGITS[name]
        for part in parts:
            folded = part.strip().upper()
            if folded == "LZ":
                # Leading zeros *present* — so the trailing ones were the
                # ones dropped, and the field is padded on the right.
                self.trailing_present = False
            elif folded == "TZ":
                self.trailing_present = True
            elif _FORMAT_RE.fullmatch(folded) and "0" in folded:
                integer, _, decimal = folded.partition(".")
                self.digits = (len(integer), len(decimal))

    def _coordinate(self, token: str) -> float:
        if self.scale is None:
            raise self.fail("a coordinate before the file stated its unit (METRIC or INCH).")
        if "." in token:
            return float(token) * self.scale
        sign = -1.0 if token.startswith("-") else 1.0
        digits = token.lstrip("+-")
        integer_digits, decimal_digits = self.digits or _DEFAULT_DIGITS[self.unit or "METRIC"]
        if self.trailing_present is False:
            digits = digits.ljust(integer_digits + decimal_digits, "0")
        return sign * int(digits) * 10.0**-decimal_digits * self.scale

    def _point(self, line: str) -> tuple[float, float] | None:
        """The point *line* names, or ``None`` if it names no coordinate."""
        found = _COORD_RE.findall(line)
        if not found:
            return None
        x, y = self.point
        for letter, token in found:
            if letter == "X":
                x = self._coordinate(token)
            else:
                y = self._coordinate(token)
        return (x, y)

    def _diameter(self) -> float:
        if self.current is None:
            raise self.fail("a hole before any tool was selected.")
        diameter = self.tools.get(self.current)
        if diameter is None:
            raise self.fail(f"tool T{self.current} is selected but never given a diameter.")
        if diameter <= 0.0:
            raise self.fail(f"tool T{self.current} has a diameter of {diameter}.")
        return diameter

    # -- lines --------------------------------------------------------

    def _attribute(self, text: str) -> None:
        """An X2 attribute riding in a ``; #@!`` comment."""
        body = text.split("#@!", 1)[1].strip()
        if not body.startswith(("TF", "TA", "TO", "TD")):
            return
        values = tuple(part.strip() for part in body[2:].split(","))
        if body.startswith("TF") and values:
            self.attributes[values[0]] = values[1:]

    def feed(self, raw: str) -> None:
        line = raw.strip()
        if not line:
            return
        if line.startswith(";"):
            if "#@!" in line:
                self._attribute(line)
            return

        head = line.upper()
        if head.startswith("M48"):
            self.in_header = True
            return
        if head in ("M95", "%"):
            self.in_header = False
            return
        if head.startswith(("M30", "M00")):
            self.done = True
            return

        for name in _UNITS:
            if head.startswith(name):
                self._set_unit(name, line[len(name) :].split(",")[1:])
                return
        if head.startswith("M71"):
            self._set_unit("METRIC", [])
            return
        if head.startswith("M72"):
            self._set_unit("INCH", [])
            return
        if head in ("LZ", "TZ"):
            self.trailing_present = head == "TZ"
            return
        if head.startswith("FMAT"):
            if head.strip() not in ("FMAT,2", "FMAT2", "FMAT"):
                raise self.fail(
                    f"drill format {line!r} is not supported; this reader "
                    f"understands format 2 (FMAT,2)."
                )
            return
        if head.startswith("ICI") and not head.endswith("OFF"):
            raise self.fail("incremental coordinates (ICI) are not supported.")
        if head.startswith("G91"):
            raise self.fail("incremental coordinates (G91) are not supported.")

        if self.in_header:
            self._header_line(line, head)
        else:
            self._body_line(line, head)

    def _header_line(self, line: str, head: str) -> None:
        match = _TOOL_RE.match(head)
        if match is None:
            return  # header settings that carry no geometry (VER, DETECT, …)
        number = int(match.group(1))
        diameter = _DIAMETER_RE.search(head)
        if diameter is None:
            return  # a tool line without a diameter states feed or speed
        if self.scale is None:
            raise self.fail("a tool diameter before the file stated its unit.")
        self.tools[number] = float(diameter.group(1)) * self.scale

    def _body_line(self, line: str, head: str) -> None:
        if head.startswith("G90") or head.startswith("G93"):
            return
        if head.startswith("G05"):
            self.routing = False
            self.pen_down = False
            return
        if head.startswith(("G02", "G03")):
            raise self.fail(
                "a circular routed slot (G02/G03) is not supported. Re-export "
                "the drill data with oval holes written as G85 slots."
            )
        if head.startswith("M15"):
            self.pen_down = True
            return
        if head.startswith("M16") or head.startswith("M17"):
            self.pen_down = False
            return

        # A tool line in the body selects; T0 unloads.
        match = _TOOL_RE.match(head)
        if match is not None and not _COORD_RE.search(head):
            number = int(match.group(1))
            # Headerless files state the diameter where they select the
            # tool; taking it here costs nothing and saves those files.
            diameter = _DIAMETER_RE.search(head)
            if diameter is not None and self.scale is not None:
                self.tools[number] = float(diameter.group(1)) * self.scale
            self.current = number or None
            return

        if "G85" in head:
            before, _, after = line.partition("G85")
            start = self._point(before)
            end = self._point(after) if after else None
            if start is None or end is None:
                raise self.fail(f"slot {line!r} does not give both of its ends.")
            self.holes.append(Slot(start=start, end=end, diameter=self._diameter()))
            self.point = end
            return

        routed = head.startswith("G00") or head.startswith("G01")
        if head.startswith("G00"):
            self.routing = True
        target = self._point(line)
        if target is None:
            if routed:
                return
            raise self.fail(f"cannot read {line!r}.")

        if self.pen_down:
            # Routed slot: the tool is down, so the move cuts material.
            self.holes.append(Slot(start=self.point, end=target, diameter=self._diameter()))
        elif not self.routing and not routed:
            self.holes.append(Hit(at=target, diameter=self._diameter()))
        self.point = target


def parse_excellon(text: str, *, source: str = "<drill>") -> DrillFile:
    """Read an Excellon drill file into its holes.

    Parameters
    ----------
    text : str
        Contents of the drill file.
    source : str
        File name, used in error messages.

    Returns
    -------
    DrillFile
        The holes, in meters, with the plating and layer span the file
        declares.

    Raises
    ------
    ValueError
        If the file is malformed or uses a construct with no equivalent
        in a 3D model.  The message names the line.
    """
    reader = _Reader(source)
    state = _Interpreter(reader)
    for number, raw in enumerate(text.splitlines(), start=1):
        reader.line = number
        state.feed(raw)
        if state.done:
            break

    if state.unit is None:
        raise reader.fail("the file states no unit (METRIC or INCH).")
    plated, span = _file_function(state.attributes.get(".FileFunction", ()))
    return DrillFile(
        holes=tuple(state.holes),
        plated=plated,
        span=span,
        unit=state.unit,
        attributes=dict(state.attributes),
    )
