"""Typed accessors for the unified option block.

Every declaration in the surface carries options as a
``tuple[OptionEntry, ...]``. Each entry's value lives in the
`OptionValue` tagged union (flag / name / number / string /
list / call). The compiler frequently asks ``what is the value of
this option, decoded to T?'' for concrete T in {str, int, float,
bool, tuple[str, ...], OptionCall}; this module is the single
typed seam where that decoding happens.

The rules are deliberate:

* Missing key returns the supplied default (or None).
* Wrong shape (e.g. asking for an int when the entry is a string)
  is a `CompileError` at the declaration's source location.
* Identifier-valued options (``role=parameter``) decode through
  ``get_option_name``; string-valued options (``path="lex.tsv"``)
  decode through ``get_option_string``. The surface distinguishes
  the two, and the compiler should not silently coerce.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import overload

from quivers.dsl.ast_nodes import (
    OptionCall,
    OptionEntry,
    OptionFlag,
    OptionList,
    OptionName,
    OptionNumber,
    OptionString,
    OptionValue,
)
from quivers.dsl.compiler._prelude import CompileError

# Keys that belong to a continuous-space constructor's brace-delimited
# keyword block (``Real 1 {low=-1.0, high=1.0}``).  When one of these
# shows up in a declaration's ``[...]`` option block the user almost
# certainly meant to attach it to the codomain constructor, so the
# unknown-key error carries a dedicated hint.
_CONSTRUCTOR_OPTION_KEYS: frozenset[str] = frozenset({"low", "high"})


def check_option_keys(
    options: tuple[OptionEntry, ...],
    allowed: frozenset[str],
    *,
    owner: str,
    line: int = 0,
    col: int = 0,
) -> None:
    """Reject any option entry whose key is outside ``allowed``.

    Every declaration / step kind that reads options declares its
    closed key set next to the code that consumes it and calls this
    checker before decoding individual keys. The error points at the
    offending entry's own line/col and offers a did-you-mean
    suggestion over the allowed set.
    """
    for entry in options:
        if entry.key in allowed:
            continue
        ln, cl = _at(line, col, entry)
        parts = [f"{owner}: unknown option {entry.key!r}"]
        matches = get_close_matches(entry.key, sorted(allowed), n=1)
        if matches:
            parts.append(f"; did you mean {matches[0]!r}?")
        if entry.key in _CONSTRUCTOR_OPTION_KEYS:
            parts.append("; constructor options attach with braces: Real 1 {low=...}")
        parts.append(f" valid options: {sorted(allowed)}")
        raise CompileError("".join(parts), ln, cl)


def find_option(options: tuple[OptionEntry, ...], key: str) -> OptionEntry | None:
    """Linear scan for the first entry whose key matches.

    The unified option block is small and bounded; a list scan is
    both clearer and faster than dict construction at the call site.
    """
    for entry in options:
        if entry.key == key:
            return entry
    return None


def has_option(options: tuple[OptionEntry, ...], key: str) -> bool:
    """Whether ``key`` appears in ``options`` (any value, including flag)."""
    return find_option(options, key) is not None


def _at(line: int, col: int, entry: OptionEntry | None) -> tuple[int, int]:
    """Pick the most precise (line, col) we have for an option error."""
    if entry is not None:
        return entry.line, entry.col
    return line, col


def get_option_flag(options: tuple[OptionEntry, ...], key: str) -> bool:
    """Whether ``[key]`` appears as a bare flag (OptionFlag value).

    Distinct from `has_option`: ``key=true`` would pass
    ``has_option`` but not ``get_option_flag``.
    """
    entry = find_option(options, key)
    return entry is not None and isinstance(entry.value, OptionFlag)


def get_option_name(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: str | None = None,
) -> str | None:
    """Decode an identifier-valued option to its name.

    Matches ``key=identifier`` (and only that). ``role=parameter``,
    ``init=auto``, ``reduction=logsumexp`` all hit this path.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionName):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected identifier value, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def get_option_string(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: str | None = None,
) -> str | None:
    """Decode a string-literal-valued option."""
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionString):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected string literal, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def _render_option_value(value: OptionValue) -> str:
    """Render an option value back to its surface text."""
    if isinstance(value, OptionName | OptionString):
        return value.value
    if isinstance(value, OptionNumber):
        f = float(value.value)
        return str(int(f)) if f.is_integer() else str(f)
    if isinstance(value, OptionCall):
        args = ", ".join(_render_option_value(a) for a in value.args)
        return f"{value.func}({args})"
    raise CompileError(
        f"option value of kind {type(value).__name__!r} has no surface text",
        0,
        0,
    )


def get_option_int_list(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Decode a list-of-integers option (``hidden_dim=[64, 32]``).

    Accepts two surface shapes, mirroring `get_option_name_list`:

    * ``[hidden_dim=[64, 32]]`` -> OptionList of OptionNumbers.
    * ``[hidden_dim=64]`` -> single OptionNumber, lifted to ``(64,)``.

    One entry per layer, so a sequence says how many as well as how
    wide, which is what an MLP needs and a single number cannot say.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    v = entry.value

    def _as_int(value: OptionValue) -> int:
        if not isinstance(value, OptionNumber):
            ln, cl = _at(line, col, entry)
            raise CompileError(
                f"option {key!r}: expected an integer or a list of them, "
                f"got {type(value).__name__}",
                ln,
                cl,
            )
        number = float(value.value)
        if not number.is_integer():
            ln, cl = _at(line, col, entry)
            raise CompileError(
                f"option {key!r}: expected whole numbers, got {number}",
                ln,
                cl,
            )
        return int(number)

    if isinstance(v, OptionNumber):
        return (_as_int(v),)
    if isinstance(v, OptionList):
        return tuple(_as_int(item) for item in v.items)
    ln, cl = _at(line, col, entry)
    raise CompileError(
        f"option {key!r}: expected an integer or a list of them, such as "
        f"``{key}=64`` or ``{key}=[64, 32]``, got {type(v).__name__}",
        ln,
        cl,
    )


def get_option_call_text(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: str | None = None,
) -> str | None:
    """Decode an option written as a bare name or a call, to its
    surface text.

    Some options name a construction whose arguments are part of the
    choice rather than separate keys: ``param_source=mlp`` and
    ``param_source=mlp(64, 64)`` select the same architecture at
    different widths. The grammar admits both, so both decode here,
    to ``"mlp"`` and ``"mlp(64, 64)"``, and the consumer parses the
    arguments it defines.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionName | OptionString | OptionCall):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected a name or a call such as "
            f"``{key}=mlp`` or ``{key}=mlp(64, 64)``, got "
            f"{type(entry.value).__name__}",
            ln,
            cl,
        )
    return _render_option_value(entry.value)


@overload
def get_option_int(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = ...,
    col: int = ...,
    default: int,
) -> int: ...
@overload
def get_option_int(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = ...,
    col: int = ...,
    default: None = ...,
) -> int | None: ...
def get_option_int(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: int | None = None,
) -> int | None:
    """Decode a numeric option as a non-negative integer.

    The grammar's ``OptionNumber`` carries a Python float; this
    accessor enforces integrality. ``depth=8``, ``hidden_dim=64``,
    ``replicate=3`` all hit this path.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionNumber):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected numeric value, got {type(entry.value).__name__}",
            ln,
            cl,
        )
    value = entry.value.value
    if not value.is_integer():
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected integer, got {value!r}",
            ln,
            cl,
        )
    return int(value)


@overload
def get_option_float(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = ...,
    col: int = ...,
    default: float,
) -> float: ...
@overload
def get_option_float(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = ...,
    col: int = ...,
    default: None = ...,
) -> float | None: ...
def get_option_float(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: float | None = None,
) -> float | None:
    """Decode a numeric option as a Python float."""
    entry = find_option(options, key)
    if entry is None:
        return default
    if not isinstance(entry.value, OptionNumber):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected numeric value, got {type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value.value


def get_option_name_list(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Decode a list-of-identifiers option (``over=[a, b]``).

    Accepts two surface shapes for unary convenience:

    * ``[over=[a, b, c]]`` -> OptionList of OptionNames.
    * ``[over=a]`` -> single OptionName, lifted to ``(a,)``.
    """
    entry = find_option(options, key)
    if entry is None:
        return default
    v = entry.value
    if isinstance(v, OptionName):
        return (v.value,)
    if isinstance(v, OptionList):
        names: list[str] = []
        for item in v.items:
            if not isinstance(item, OptionName):
                ln, cl = _at(line, col, entry)
                raise CompileError(
                    f"option {key!r}: list items must be identifiers, "
                    f"got {type(item).__name__}",
                    ln,
                    cl,
                )
            names.append(item.value)
        return tuple(names)
    ln, cl = _at(line, col, entry)
    raise CompileError(
        f"option {key!r}: expected identifier or list-of-identifiers, "
        f"got {type(v).__name__}",
        ln,
        cl,
    )


def get_option_call(
    options: tuple[OptionEntry, ...],
    key: str,
    *,
    line: int = 0,
    col: int = 0,
) -> OptionCall | None:
    """Decode a callable-shaped option (``on=program(NAME)``)."""
    entry = find_option(options, key)
    if entry is None:
        return None
    if not isinstance(entry.value, OptionCall):
        ln, cl = _at(line, col, entry)
        raise CompileError(
            f"option {key!r}: expected call-shaped value (e.g. "
            f"``program(NAME)``), got {type(entry.value).__name__}",
            ln,
            cl,
        )
    return entry.value


def get_program_effects(
    options: tuple[OptionEntry, ...],
    *,
    line: int = 0,
    col: int = 0,
) -> frozenset[str] | None:
    """Decode the ``effects`` option of a program declaration.

    Two shapes are accepted, matching the surface forms ``[effects=Pure]``
    and ``[effects=[Sample, Score]]``. Absence returns ``None`` so the
    compiler can distinguish ``no annotation`` from ``empty set``.
    """
    if not has_option(options, "effects"):
        return None
    names = get_option_name_list(options, "effects", line=line, col=col)
    return frozenset(names)


def get_program_posterior_model(
    options: tuple[OptionEntry, ...],
    *,
    line: int = 0,
    col: int = 0,
) -> str | None:
    """Decode the ``posterior_of`` option of a program declaration.

    The surface form ``[posterior_of=<name>]`` marks a posterior block whose
    body is run against the named generative model; absence means
    the program is a vanilla generative program.
    """
    return get_option_name(options, "posterior_of", line=line, col=col)


def get_option_value(options: tuple[OptionEntry, ...], key: str) -> OptionValue | None:
    """Untyped option lookup; returns the raw `OptionValue`.

    For paths where the compiler dispatches on the value shape itself
    (e.g. ``loss.on`` accepts both ``[global]`` flag and ``on=...``
    call).
    """
    entry = find_option(options, key)
    return entry.value if entry is not None else None


__all__ = [
    "check_option_keys",
    "find_option",
    "get_option_call",
    "get_option_flag",
    "get_option_float",
    "get_option_int",
    "get_option_name",
    "get_option_name_list",
    "get_option_string",
    "get_option_value",
    "get_program_effects",
    "get_program_posterior_model",
    "has_option",
]
