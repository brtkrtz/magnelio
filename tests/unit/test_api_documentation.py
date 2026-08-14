"""Guards that the public API documents itself in the API reference.

Autodoc renders a member only if it carries a docstring, and it skips
inherited members unless the page asks for them.  Both rules have
silently emptied parts of the reference before, so the contracts most
likely to be read are pinned here.
"""

import inspect

import pytest

from magnelio.analysis import ScatteringResult, ScatteringTDResult
from magnelio.io.project import Project

# The scattering-result contract: what a user script may call on the
# object run() hands back, whichever implementation it is.
CONTRACT = (
    "f_axis",
    "channels",
    "excitations",
    "settings",
    "S",
    "db",
    "phase",
    "a",
    "b",
    "plot_s",
    "to_touchstone",
    "to_skrf",
)

IMPLEMENTATIONS = (ScatteringTDResult, Project)


def _doc_of(cls, name):
    """Docstring of *name* on *cls*, seeing through property wrappers."""
    obj = inspect.getattr_static(cls, name, None)
    if obj is None:
        return None
    target = obj.fget if isinstance(obj, property) else obj
    return getattr(target, "__doc__", None)


@pytest.mark.parametrize("cls", IMPLEMENTATIONS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("member", CONTRACT)
def test_contract_member_is_documented(cls, member):
    # A dataclass field is documented in the class docstring's
    # Attributes block rather than on itself.
    fields = getattr(cls, "__dataclass_fields__", {})
    if member in fields:
        assert f"{member} :" in (cls.__doc__ or ""), (
            f"{cls.__name__}.{member} is a dataclass field but the class "
            f"docstring has no Attributes entry for it"
        )
        return
    assert _doc_of(cls, member), f"{cls.__name__}.{member} has no docstring — autodoc drops it"


@pytest.mark.parametrize("member", CONTRACT)
def test_protocol_declares_and_documents_the_member(member):
    assert hasattr(ScatteringResult, member) or member in (
        "plot_s",
        "to_touchstone",
        "to_skrf",
    ), f"{member} is in the contract but the ScatteringResult protocol omits it"
    if hasattr(ScatteringResult, member):
        assert _doc_of(ScatteringResult, member), f"ScatteringResult.{member} has no docstring"


@pytest.mark.parametrize("cls", IMPLEMENTATIONS, ids=lambda c: c.__name__)
def test_implementation_satisfies_the_protocol(cls):
    for member in CONTRACT:
        assert hasattr(cls, member), f"{cls.__name__} is missing {member}"


def test_docstrings_use_only_real_numpydoc_sections():
    """An invented section heading becomes a phantom member.

    ``Convenience\\n-----------`` in a class docstring rendered as an
    attribute named "Convenience" and took the surrounding prose out of
    the reference with it.
    """
    valid = {
        "Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Other Parameters",
        "Raises",
        "Warns",
        "Warnings",
        "See Also",
        "Notes",
        "References",
        "Examples",
        "Attributes",
        "Methods",
    }
    offenders = []
    for label, obj in _public_objects():
        lines = (obj.__doc__ or "").splitlines()
        for i, line in enumerate(lines[:-1]):
            underline = lines[i + 1].strip()
            heading = line.strip()
            if underline and set(underline) == {"-"} and heading and heading not in valid:
                offenders.append(f"{label}: {heading!r}")
    assert not offenders, f"invented numpydoc sections: {offenders}"


# ── phantom parameters across the whole public surface ───────────────────────


def _public_objects():
    """Every documented object reachable through a namespace ``__all__``."""
    import importlib

    namespaces = [
        "",
        "geo",
        "materials",
        "mesh",
        "boundaries",
        "ports",
        "sources",
        "monitors",
        "analysis",
        "circuit",
        "io",
        "post",
        "plots",
        "signals",
        "solver",
        "constants",
    ]
    seen = set()
    for ns in namespaces:
        name = f"magnelio.{ns}" if ns else "magnelio"
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - optional extras
            continue
        for attr in getattr(module, "__all__", []):
            obj = getattr(module, attr, None)
            if obj is None or not getattr(obj, "__doc__", None):
                continue
            if id(obj) not in seen:
                seen.add(id(obj))
                yield f"{name}.{attr}", obj
            if inspect.isclass(obj):
                for member_name, member in vars(obj).items():
                    if member_name.startswith("_"):
                        continue
                    target = member.fget if isinstance(member, property) else member
                    if callable(target) and getattr(target, "__doc__", None):
                        if id(target) not in seen:
                            seen.add(id(target))
                            yield f"{name}.{attr}.{member_name}", target


def _signature_names(obj):
    """Parameter names of *obj* (and of its ``__init__``), unadorned."""
    names = set()
    for candidate in (obj, getattr(obj, "__init__", None)):
        if candidate is None:
            continue
        try:
            names |= set(inspect.signature(candidate).parameters)
        except (ValueError, TypeError):
            pass
    return names


def test_no_docstring_invents_a_parameter():
    """Prose after the Parameters block is parsed as more parameters.

    Napoleon ends a numpydoc Parameters block only at the next section
    heading it knows.  A ``Example::`` line instead of an ``Examples``
    section therefore turned the sentences that followed into parameter
    *names*, with the example code as their description — the class
    advertised five parameters where it takes three.
    """
    import re

    from sphinx.ext.napoleon import Config, NumpyDocstring

    config = Config(napoleon_google_docstring=True, napoleon_numpy_docstring=True)
    offenders = []
    for label, obj in _public_objects():
        rendered = str(NumpyDocstring(inspect.cleandoc(obj.__doc__), config))
        declared = {m.group(1) for m in re.finditer(r"^:param ([^:]+):", rendered, re.M)}
        real = _signature_names(obj)
        for name in declared:
            # Napoleon escapes the stars of *args / **kwargs.
            bare = name.replace("\\", "").lstrip("*").split()[0]
            if bare not in real:
                offenders.append(f"{label}: {name[:60]!r}")
    assert not offenders, f"docstrings advertising parameters that do not exist: {offenders}"


def test_type_fields_name_only_resolvable_types():
    """A numpydoc type is resolved as a cross-reference, so it must be
    a type — not a shape, a tuple layout, or a pair of return names.

    ``excited : str or (label, mode)`` sent Sphinx looking for an object
    called ``label``; two exist in the tree, and the build warned about
    the ambiguity.  The names that resolve to nothing at all stay silent
    outside nitpicky mode, which makes them latent rather than harmless:
    they start warning as soon as a second class grows a member of that
    name.  Structure belongs in the description, inside double
    backticks, where nothing tries to resolve it.
    """
    import re

    from sphinx.ext.napoleon import Config, NumpyDocstring

    # Names Sphinx resolves without help.  A parenthesised group built
    # only from these — "tuple of (str, int)" — is ordinary numpydoc.
    resolvable = {
        "str", "int", "float", "complex", "bool", "bytes", "object", "type",
        "list", "tuple", "dict", "set", "frozenset", "callable", "None",
        "ndarray", "np", "array", "optional", "default", "or", "of",
    }  # fmt: skip

    config = Config(napoleon_google_docstring=True, napoleon_numpy_docstring=True)
    offenders = []
    for label, obj in _public_objects():
        rendered = str(NumpyDocstring(inspect.cleandoc(obj.__doc__), config))
        for m in re.finditer(r"^:(?:rtype|type [^:]+):\s*(.+)$", rendered, re.M):
            # Parenthesised groups are where free names hide: "(fig, ax)",
            # "shape (Nf,)", "((x0, y0, z0), (x1, y1, z1))".  Groups of
            # digits such as "(3,)" name no target at all.
            for group in re.findall(r"\(([^)]*)\)", m.group(1)):
                words = [w.strip(" `'\"") for w in re.split(r"[,\s]+", group) if w.strip()]
                free = [w for w in words if w[:1].isalpha() and w not in resolvable]
                if free:
                    offenders.append(f"{label}: :type: {m.group(1)[:50]!r} -> {free}")
    assert not offenders, f"type fields Sphinx will try to resolve as names: {offenders}"
