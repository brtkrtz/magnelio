"""RS-274X (Gerber) reader — a copper layer as plain geometry records.

Gerber is a plotting language, not a shape format: a file is a stream
of commands that move an aperture over a plane and expose it.  What a
layer *is* only exists after that stream has been played back, so this
module is an interpreter with a graphics state — aperture, position,
interpolation mode, polarity — that emits one record per exposed
object.

The records are plain data (no geometry kernel is involved here), for
two reasons: the format's own traps — coordinate formats without a
decimal point, apertures defined by macro programs, contours that mix
lines and arcs — are all resolved before any face is built, and the
resulting reader can be tested against hand-written files without a
kernel present.

Written against the Gerber Layer Format Specification (Ucamco).  What
is not supported fails with the file and line that asked for it, rather
than being silently dropped: a missing pad in a board is not something
the caller can be expected to notice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Meters per file unit.  Gerber knows exactly these two.
_UNITS = {"MM": 1e-3, "IN": 0.0254}


# ─────────────────────────────────────────────────────────────────────
# apertures
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Circle:
    """Round aperture, optionally with a round hole."""

    diameter: float
    hole: float | None = None


@dataclass(frozen=True)
class Rect:
    """Rectangular aperture, optionally with a round hole."""

    width: float
    height: float
    hole: float | None = None


@dataclass(frozen=True)
class Obround:
    """Stadium-shaped aperture (a rectangle with semicircular ends)."""

    width: float
    height: float
    hole: float | None = None


@dataclass(frozen=True)
class RegularPolygon:
    """Regular polygon aperture, inscribed in *diameter*."""

    diameter: float
    vertices: int
    rotation: float = 0.0
    hole: float | None = None


@dataclass(frozen=True)
class MacroCircle:
    exposure: bool
    diameter: float
    center: tuple[float, float]
    rotation: float = 0.0


@dataclass(frozen=True)
class MacroVectorLine:
    exposure: bool
    width: float
    start: tuple[float, float]
    end: tuple[float, float]
    rotation: float = 0.0


@dataclass(frozen=True)
class MacroCenterLine:
    exposure: bool
    width: float
    height: float
    center: tuple[float, float]
    rotation: float = 0.0


@dataclass(frozen=True)
class MacroOutline:
    exposure: bool
    points: tuple[tuple[float, float], ...]
    rotation: float = 0.0


@dataclass(frozen=True)
class MacroPolygon:
    exposure: bool
    vertices: int
    center: tuple[float, float]
    diameter: float
    rotation: float = 0.0


@dataclass(frozen=True)
class MacroAperture:
    """An aperture built by a macro: primitives added and subtracted.

    The primitives are already evaluated — the macro's arithmetic and
    its ``$n`` parameters were resolved when the aperture was defined,
    so what remains is a list of shapes in meters, in the order they
    have to be applied (an ``exposure`` of ``False`` cuts).
    """

    name: str
    primitives: tuple[object, ...]


Aperture = Circle | Rect | Obround | RegularPolygon | MacroAperture


# ─────────────────────────────────────────────────────────────────────
# graphical objects
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Flash:
    """An aperture stamped down at one point."""

    at: tuple[float, float]
    aperture: Aperture


@dataclass(frozen=True)
class Stroke:
    """A straight track of *width*, with round ends."""

    start: tuple[float, float]
    end: tuple[float, float]
    width: float


@dataclass(frozen=True)
class ArcStroke:
    """A circular track of *width*, with round ends."""

    start: tuple[float, float]
    end: tuple[float, float]
    center: tuple[float, float]
    clockwise: bool
    width: float


@dataclass(frozen=True)
class LineSegment:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class ArcSegment:
    start: tuple[float, float]
    end: tuple[float, float]
    center: tuple[float, float]
    clockwise: bool


Segment = LineSegment | ArcSegment


@dataclass(frozen=True)
class Region:
    """A filled area, bounded by one or more closed contours."""

    contours: tuple[tuple[Segment, ...], ...]


GraphicalObject = Flash | Stroke | ArcStroke | Region


@dataclass(frozen=True)
class GerberLayer:
    """One Gerber file, played back.

    Attributes
    ----------
    objects : tuple
        ``(dark, object)`` pairs in the order the file draws them.
        ``dark`` is the polarity in force: a ``False`` object clears
        what earlier objects have drawn, so the order carries meaning
        and must not be sorted away.
    unit : str
        ``"MM"`` or ``"IN"`` — the unit the file was written in.  All
        coordinates in the objects are already in meters.
    attributes : dict
        File attributes (``%TF...%``), name to values, for instance
        ``".FileFunction": ("Copper", "L1", "Top")``.
    resolution : float
        Length [meters] of one step of the file's coordinate format.
        Two coordinates that were meant to be the same point are
        written with the same digits, so this is the exact distance
        below which two endpoints in this file are the same node —
        which is what chaining an outline out of loose segments needs,
        and what nothing but the file itself can supply.
    """

    objects: tuple[tuple[bool, GraphicalObject], ...]
    unit: str
    attributes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    resolution: float = 0.0

    @property
    def file_function(self) -> tuple[str, ...]:
        """The ``.FileFunction`` attribute, or ``()`` if the file has none."""
        return self.attributes.get(".FileFunction", ())


# ─────────────────────────────────────────────────────────────────────
# lexing
# ─────────────────────────────────────────────────────────────────────


class _Reader:
    """Command stream of a Gerber file, with line numbers for errors."""

    def __init__(self, text: str, source: str) -> None:
        self._text = text
        self._source = source
        self._at = 0
        self._line = 1

    def fail(self, line: int, message: str) -> "ValueError":
        return ValueError(f"{self._source}, line {line}: {message}")

    def _advance(self, to: int) -> None:
        self._line += self._text.count("\n", self._at, to)
        self._at = to

    def commands(self):
        """Yield ``(line, extended, payload)`` for every command."""
        text = self._text
        end = len(text)
        while self._at < end:
            char = text[self._at]
            if char.isspace():
                self._advance(self._at + 1)
                continue
            if char == "%":
                start_line = self._line
                close = text.find("%", self._at + 1)
                if close < 0:
                    raise self.fail(start_line, "extended command is not closed by '%'.")
                block = text[self._at + 1 : close]
                self._advance(close + 1)
                yield start_line, True, [part.strip() for part in block.split("*") if part.strip()]
                continue
            start_line = self._line
            close = text.find("*", self._at)
            if close < 0:
                raise self.fail(start_line, "command is not terminated by '*'.")
            word = text[self._at : close].strip()
            self._advance(close + 1)
            if word:
                yield start_line, False, word


# ─────────────────────────────────────────────────────────────────────
# aperture macro arithmetic
# ─────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\s*(\d+\.?\d*|\.\d+|\$\d+|[-+xX/()])")


def _evaluate(expression: str, variables: dict[int, float], fail) -> float:
    """Value of a macro expression, with ``$n`` taken from *variables*.

    Macros are small programs — ``$4=$1x0.75-$2`` — so their arithmetic
    is parsed rather than handed to the interpreter: the file is
    untrusted input, and ``x`` means multiplication here, which Python
    would not agree with anyway.
    """
    tokens: list[str] = []
    at = 0
    while at < len(expression):
        match = _TOKEN_RE.match(expression, at)
        if match is None:
            if expression[at:].isspace():
                break
            raise fail(f"cannot read {expression[at:]!r} in a macro expression.")
        tokens.append(match.group(1))
        at = match.end()
    if not tokens:
        raise fail("empty macro expression.")

    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        token = tokens[position]
        position += 1
        return token

    def factor() -> float:
        token = peek()
        if token is None:
            raise fail(f"macro expression {expression!r} ends early.")
        if token in "+-":
            take()
            value = factor()
            return -value if token == "-" else value
        if token == "(":
            take()
            value = expr()
            if peek() != ")":
                raise fail(f"macro expression {expression!r} misses a ')'.")
            take()
            return value
        token = take()
        if token.startswith("$"):
            index = int(token[1:])
            if index not in variables:
                raise fail(f"macro expression {expression!r} uses undefined ${index}.")
            return variables[index]
        try:
            return float(token)
        except ValueError:
            raise fail(f"macro expression {expression!r} is malformed near {token!r}.") from None

    def term() -> float:
        value = factor()
        while (token := peek()) in ("x", "X", "/"):
            take()
            other = factor()
            if token == "/":
                if other == 0.0:
                    raise fail(f"macro expression {expression!r} divides by zero.")
                value /= other
            else:
                value *= other
        return value

    def expr() -> float:
        value = term()
        while (token := peek()) in ("+", "-"):
            take()
            other = term()
            value = value + other if token == "+" else value - other
        return value

    result = expr()
    if position != len(tokens):
        raise fail(f"macro expression {expression!r} has trailing {tokens[position]!r}.")
    return result


_UNSUPPORTED_PRIMITIVES = {
    "2": "the deprecated vector line primitive (2) — re-export with primitive 20",
    "6": "the moiré primitive (6), which draws no copper of its own",
    "7": "the thermal primitive (7)",
    "22": "the deprecated lower-left line primitive (22) — re-export with primitive 21",
}


def _macro_primitives(name, body, arguments, scale, fail) -> tuple[object, ...]:
    """Evaluate a macro body into primitives, in meters."""
    variables = {index: value for index, value in enumerate(arguments, start=1)}
    primitives: list[object] = []

    for statement in body:
        if statement.startswith("0"):  # comment primitive
            continue
        if statement.startswith("$"):
            target, _, expression = statement.partition("=")
            if not expression:
                raise fail(f"macro {name!r} has a malformed assignment {statement!r}.")
            variables[int(target[1:])] = _evaluate(expression, variables, fail)
            continue

        parts = [part.strip() for part in statement.split(",")]
        kind = parts[0]
        if kind in _UNSUPPORTED_PRIMITIVES:
            raise fail(f"aperture macro {name!r} uses {_UNSUPPORTED_PRIMITIVES[kind]}.")
        values = [_evaluate(part, variables, fail) for part in parts[1:]]

        def need(count: int, what: str) -> list[float]:
            if len(values) < count:
                raise fail(f"macro {name!r}: {what} needs {count} parameters, got {len(values)}.")
            return values

        if kind == "1":
            need(4, "the circle primitive")
            primitives.append(
                MacroCircle(
                    exposure=values[0] != 0.0,
                    diameter=values[1] * scale,
                    center=(values[2] * scale, values[3] * scale),
                    rotation=values[4] if len(values) > 4 else 0.0,
                )
            )
        elif kind == "20":
            need(6, "the vector line primitive")
            primitives.append(
                MacroVectorLine(
                    exposure=values[0] != 0.0,
                    width=values[1] * scale,
                    start=(values[2] * scale, values[3] * scale),
                    end=(values[4] * scale, values[5] * scale),
                    rotation=values[6] if len(values) > 6 else 0.0,
                )
            )
        elif kind == "21":
            need(5, "the center line primitive")
            primitives.append(
                MacroCenterLine(
                    exposure=values[0] != 0.0,
                    width=values[1] * scale,
                    height=values[2] * scale,
                    center=(values[3] * scale, values[4] * scale),
                    rotation=values[5] if len(values) > 5 else 0.0,
                )
            )
        elif kind == "4":
            need(5, "the outline primitive")
            count = int(round(values[1]))
            # The vertex count excludes the repeated closing point, which
            # the format writes out; both are read, the loop closes itself.
            coordinates = values[2 : 2 + 2 * (count + 1)]
            if len(coordinates) < 2 * (count + 1):
                raise fail(
                    f"macro {name!r}: outline primitive announces {count} "
                    f"vertices but carries {len(coordinates) // 2} points."
                )
            points = tuple(
                (coordinates[i] * scale, coordinates[i + 1] * scale)
                for i in range(0, len(coordinates), 2)
            )
            rotation = values[2 + 2 * (count + 1)] if len(values) > 2 + 2 * (count + 1) else 0.0
            primitives.append(
                MacroOutline(exposure=values[0] != 0.0, points=points, rotation=rotation)
            )
        elif kind == "5":
            need(5, "the polygon primitive")
            primitives.append(
                MacroPolygon(
                    exposure=values[0] != 0.0,
                    vertices=int(round(values[1])),
                    center=(values[2] * scale, values[3] * scale),
                    diameter=values[4] * scale,
                    rotation=values[5] if len(values) > 5 else 0.0,
                )
            )
        else:
            raise fail(f"aperture macro {name!r} uses unknown primitive {kind!r}.")

    if not primitives:
        raise fail(f"aperture macro {name!r} defines no primitive.")
    return tuple(primitives)


# ─────────────────────────────────────────────────────────────────────
# the interpreter
# ─────────────────────────────────────────────────────────────────────

# The D code is unbounded: it is an operation below 10 and an aperture
# number above, and a dense board runs past three digits of apertures.
_WORD_RE = re.compile(r"G(\d{1,3})|([XYIJ])([+-]?\d+)|D(\d+)|M(\d{1,3})")

_DEPRECATED_TRANSFORMS = {
    "AS": "axis select",
    "MI": "mirror image",
    "OF": "offset",
    "SF": "scale factor",
    "IR": "image rotation",
}


class _Interpreter:
    def __init__(self, reader: _Reader) -> None:
        self._reader = reader
        self._line = 0
        self.integer_digits: int | None = None
        self.decimal_digits: int | None = None
        self.trailing_zeros = False
        self.scale: float | None = None
        self.unit: str | None = None
        self.apertures: dict[int, Aperture] = {}
        self.macros: dict[str, list[str]] = {}
        self.current: int | None = None
        self.point: tuple[float, float] | None = None
        self.mode = "linear"
        self.multi_quadrant = False
        self.dark = True
        self.objects: list[tuple[bool, GraphicalObject]] = []
        self.attributes: dict[str, tuple[str, ...]] = {}
        self.region: list[tuple[Segment, ...]] | None = None
        self._contour: list[Segment] = []
        self._contour_start: tuple[float, float] | None = None
        self.done = False

    # -- helpers ------------------------------------------------------

    def fail(self, message: str) -> ValueError:
        return self._reader.fail(self._line, message)

    def _aperture(self) -> Aperture:
        if self.current is None:
            raise self.fail("a draw or flash before any aperture was selected.")
        return self.apertures[self.current]

    def _stroke_width(self) -> float:
        aperture = self._aperture()
        if not isinstance(aperture, Circle):
            kind = type(aperture).__name__.lower()
            raise self.fail(
                f"aperture D{self.current} is a {kind}, and drawing a track "
                f"with anything but a round aperture is not supported "
                f"(the format deprecated it). Re-export the layer."
            )
        return aperture.diameter

    def _coordinate(self, token: str) -> float:
        assert self.decimal_digits is not None and self.integer_digits is not None
        sign = -1.0 if token.startswith("-") else 1.0
        digits = token.lstrip("+-")
        if self.trailing_zeros:
            digits = digits.ljust(self.integer_digits + self.decimal_digits, "0")
        if self.scale is None:
            raise self.fail("a coordinate before the unit was set by an MO command.")
        return sign * int(digits) * 10.0**-self.decimal_digits * self.scale

    # -- extended commands --------------------------------------------

    def extended(self, statements: list[str]) -> None:
        head = statements[0]
        code = head[:2]
        if code == "FS":
            self._format(head)
        elif code == "MO":
            self._unit(head)
        elif code == "AD":
            self._define_aperture(head)
        elif code == "AM":
            self.macros[head[2:]] = statements[1:]
        elif code == "LP":
            polarity = head[2:].strip()
            if polarity not in ("D", "C"):
                raise self.fail(f"unknown polarity {polarity!r}; expected LPD or LPC.")
            self.dark = polarity == "D"
        elif code in ("LM", "LR", "LS"):
            self._aperture_transform(code, head[2:].strip())
        elif code == "TF":
            values = tuple(part.strip() for part in head[2:].split(","))
            if values:
                self.attributes[values[0]] = values[1:]
        elif code in ("TA", "TO", "TD"):
            pass  # object attributes: metadata, no geometry
        elif code == "SR":
            if head[2:].strip() not in ("", "X1Y1I0J0", "X1Y1I0.0J0.0"):
                raise self.fail(
                    "step-and-repeat (SR) is not supported. Re-export the "
                    "layer with the repeats flattened into real copper."
                )
        elif code == "IP":
            if head[2:].strip() != "POS":
                raise self.fail(
                    "a negative image (IPNEG) is not supported. Re-export the "
                    "layer with positive polarity."
                )
        elif code in ("IN", "LN"):
            pass  # image / layer name: documentation only
        elif code in _DEPRECATED_TRANSFORMS:
            raise self.fail(
                f"the deprecated {_DEPRECATED_TRANSFORMS[code]} command ({code}) "
                f"is not supported. Re-export the layer."
            )
        else:
            raise self.fail(f"unknown extended command {head!r}.")

    def _format(self, head: str) -> None:
        match = re.fullmatch(r"FS([LT])?([AI])?X(\d)(\d)Y(\d)(\d)", head)
        if match is None:
            raise self.fail(f"cannot read the coordinate format {head!r}.")
        zeros, notation, xi, xd, yi, yd = match.groups()
        if notation == "I":
            raise self.fail("incremental coordinates (FS…I…) are not supported.")
        if (xi, xd) != (yi, yd):
            raise self.fail(f"X and Y coordinate formats differ in {head!r}.")
        self.trailing_zeros = zeros == "T"
        self.integer_digits = int(xi)
        self.decimal_digits = int(xd)

    def _unit(self, head: str) -> None:
        unit = head[2:].strip().upper()
        if unit not in _UNITS:
            raise self.fail(f"unknown unit {unit!r}; expected MM or IN.")
        self.unit = unit
        self.scale = _UNITS[unit]

    def _aperture_transform(self, code: str, value: str) -> None:
        neutral = {"LM": "N", "LR": "0", "LS": "1"}
        try:
            trivial = value == neutral[code] or float(value) == float(neutral[code])
        except ValueError:
            trivial = False
        if not trivial:
            raise self.fail(
                f"the aperture transformation {code}{value} is not supported. "
                f"Re-export the layer with the transformations resolved."
            )

    def _define_aperture(self, head: str) -> None:
        match = re.fullmatch(r"AD D?(\d+) ([^,]+) (?: , (.*) )?", head, re.VERBOSE)
        if match is None:
            raise self.fail(f"cannot read the aperture definition {head!r}.")
        number, template, arguments = match.groups()
        number = int(number)
        if number < 10:
            raise self.fail(f"aperture numbers below 10 are reserved; got D{number}.")
        if self.scale is None:
            raise self.fail("an aperture definition before the unit was set by an MO command.")
        values: list[float] = []
        for part in (arguments or "").split("X"):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part))
            except ValueError:
                raise self.fail(
                    f"aperture D{number} has a non-numeric parameter {part!r}."
                ) from None

        self.apertures[number] = self._build_aperture(number, template, values)

    def _build_aperture(self, number: int, template: str, values: list[float]) -> Aperture:
        scale = self.scale
        assert scale is not None

        def need(count: int, what: str) -> None:
            if len(values) < count:
                raise self.fail(
                    f"aperture D{number} ({what}) needs {count} parameters, got {len(values)}."
                )

        def hole(at: int) -> float | None:
            if len(values) <= at:
                return None
            if len(values) > at + 1:
                raise self.fail(
                    f"aperture D{number} has a rectangular hole, which is "
                    f"deprecated and not supported. Re-export the layer."
                )
            return values[at] * scale if values[at] > 0.0 else None

        if template == "C":
            need(1, "circle")
            return Circle(diameter=values[0] * scale, hole=hole(1))
        if template == "R":
            need(2, "rectangle")
            return Rect(width=values[0] * scale, height=values[1] * scale, hole=hole(2))
        if template == "O":
            need(2, "obround")
            return Obround(width=values[0] * scale, height=values[1] * scale, hole=hole(2))
        if template == "P":
            need(2, "polygon")
            vertices = int(round(values[1]))
            if vertices < 3:
                raise self.fail(f"aperture D{number} is a polygon with {vertices} vertices.")
            return RegularPolygon(
                diameter=values[0] * scale,
                vertices=vertices,
                rotation=values[2] if len(values) > 2 else 0.0,
                hole=hole(3),
            )
        body = self.macros.get(template)
        if body is None:
            raise self.fail(
                f"aperture D{number} uses the undefined template {template!r}. "
                f"Standard templates are C, R, O and P; anything else has to "
                f"be defined by an aperture macro earlier in the file."
            )
        return MacroAperture(
            name=template,
            primitives=_macro_primitives(template, body, values, scale, self.fail),
        )

    # -- words --------------------------------------------------------

    def word(self, word: str) -> None:
        if word.startswith("G04") or word.startswith("G4"):
            return
        codes: list[int] = []
        coordinates: dict[str, str] = {}
        operation: int | None = None
        position = 0
        for match in _WORD_RE.finditer(word):
            if match.start() != position:
                raise self.fail(f"cannot read {word[position : match.start()]!r} in {word!r}.")
            position = match.end()
            g_code, letter, digits, d_code, m_code = match.groups()
            if g_code is not None:
                codes.append(int(g_code))
            elif letter is not None:
                coordinates[letter] = digits
            elif d_code is not None:
                operation = int(d_code)
            elif m_code is not None:
                if int(m_code) in (0, 2):
                    self.done = True
        if position != len(word):
            raise self.fail(f"cannot read {word[position:]!r} in {word!r}.")

        for code in codes:
            self._graphics_code(code)
        if operation is None:
            return
        if operation >= 10:
            if operation not in self.apertures:
                raise self.fail(f"aperture D{operation} is selected but never defined.")
            self.current = operation
            return
        self._operation(operation, coordinates)

    def _graphics_code(self, code: int) -> None:
        if code == 1:
            self.mode = "linear"
        elif code == 2:
            self.mode = "cw"
        elif code == 3:
            self.mode = "ccw"
        elif code == 36:
            self._begin_region()
        elif code == 37:
            self._end_region()
        elif code == 74:
            raise self.fail(
                "single quadrant arc mode (G74) is deprecated and not "
                "supported. Re-export the layer in multi quadrant mode (G75)."
            )
        elif code == 75:
            self.multi_quadrant = True
        elif code == 70:
            self.unit, self.scale = "IN", _UNITS["IN"]
        elif code == 71:
            self.unit, self.scale = "MM", _UNITS["MM"]
        elif code == 90:
            pass  # absolute coordinates: the only mode supported anyway
        elif code == 91:
            raise self.fail("incremental coordinates (G91) are not supported.")
        elif code in (54, 55):
            pass  # deprecated aperture select / prepare flash prefixes
        else:
            raise self.fail(f"unknown command G{code:02d}.")

    def _point(self, coordinates: dict[str, str]) -> tuple[float, float]:
        previous = self.point or (0.0, 0.0)
        x = self._coordinate(coordinates["X"]) if "X" in coordinates else previous[0]
        y = self._coordinate(coordinates["Y"]) if "Y" in coordinates else previous[1]
        return (x, y)

    def _operation(self, operation: int, coordinates: dict[str, str]) -> None:
        target = self._point(coordinates)
        if operation == 2:  # move
            if self.region is not None:
                self._close_contour()
                self._contour_start = target
            self.point = target
            return
        if operation == 3:  # flash
            if self.region is not None:
                raise self.fail("a flash (D03) inside a region (G36).")
            self.objects.append((self.dark, Flash(at=target, aperture=self._aperture())))
            self.point = target
            return
        if operation != 1:
            raise self.fail(f"unknown operation D{operation:02d}.")

        start = self.point
        if start is None:
            raise self.fail("a draw (D01) before the current point was set by a move (D02).")
        if self.mode == "linear":
            segment: Segment = LineSegment(start=start, end=target)
        else:
            if not self.multi_quadrant:
                raise self.fail(
                    "an arc (G02/G03) without multi quadrant mode. Re-export "
                    "the layer with G75 in force."
                )
            if "I" not in coordinates and "J" not in coordinates:
                raise self.fail("an arc (G02/G03) without a centre offset (I/J).")
            offset_x = self._coordinate(coordinates.get("I", "0"))
            offset_y = self._coordinate(coordinates.get("J", "0"))
            center = (start[0] + offset_x, start[1] + offset_y)
            segment = ArcSegment(
                start=start, end=target, center=center, clockwise=self.mode == "cw"
            )

        if self.region is not None:
            if self._contour_start is None:
                self._contour_start = start
            self._contour.append(segment)
        elif isinstance(segment, LineSegment):
            self.objects.append((self.dark, Stroke(start, target, self._stroke_width())))
        else:
            self.objects.append(
                (
                    self.dark,
                    ArcStroke(
                        start=segment.start,
                        end=segment.end,
                        center=segment.center,
                        clockwise=segment.clockwise,
                        width=self._stroke_width(),
                    ),
                )
            )
        self.point = target

    # -- regions ------------------------------------------------------

    def _begin_region(self) -> None:
        if self.region is not None:
            raise self.fail("a region (G36) started while one was already open.")
        self.region = []
        self._contour = []
        self._contour_start = self.point

    def _close_contour(self) -> None:
        if not self._contour:
            return
        assert self.region is not None
        first = self._contour[0].start
        last = self._contour[-1].end
        if abs(first[0] - last[0]) > 0.0 or abs(first[1] - last[1]) > 0.0:
            # Writers differ on whether the closing segment is drawn; a
            # region is closed by definition, so the gap is closed here.
            self._contour.append(LineSegment(start=last, end=first))
        self.region.append(tuple(self._contour))
        self._contour = []

    def _end_region(self) -> None:
        if self.region is None:
            raise self.fail("a region end (G37) without a region start (G36).")
        self._close_contour()
        contours = tuple(self.region)
        self.region = None
        self._contour_start = None
        if contours:
            self.objects.append((self.dark, Region(contours=contours)))

    def resolution(self) -> float:
        """Length of one step of the coordinate format [meters]."""
        if self.decimal_digits is None or self.scale is None:
            return 0.0
        return 10.0**-self.decimal_digits * self.scale

    # -- driving ------------------------------------------------------

    def run(self) -> None:
        """Play the whole command stream, then check it ended cleanly."""
        for line, extended, payload in self._reader.commands():
            self._line = line
            if extended:
                self.extended(payload)
            else:
                self.word(payload)
            if self.done:
                break
        if self.unit is None:
            raise self.fail("the file sets no unit (MO command).")
        if self.region is not None:
            raise self.fail("a region (G36) is never closed by G37.")


def parse_gerber(text: str, *, source: str = "<gerber>") -> GerberLayer:
    """Play back a Gerber file into its graphical objects.

    Parameters
    ----------
    text : str
        Contents of the ``.gbr`` file.
    source : str
        File name, used in error messages.

    Returns
    -------
    GerberLayer
        The objects the file draws, in order, with all coordinates in
        meters.

    Raises
    ------
    ValueError
        If the file is malformed, or uses a construct that has no
        equivalent in a 3D model (step-and-repeat, negative images,
        deprecated transformations).  The message names the line.
    """
    reader = _Reader(text, source)
    state = _Interpreter(reader)
    state.run()
    return GerberLayer(
        objects=tuple(state.objects),
        unit=state.unit,
        attributes=dict(state.attributes),
        resolution=state.resolution(),
    )
