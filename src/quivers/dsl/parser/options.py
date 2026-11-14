"""Walker for the unified ``[k=v, ...]`` option block.

Produces tuples of `OptionEntry` from the tree-sitter
``option_block`` vertex. The `OptionValue` tagged union covers
every surface shape:

* ``[role]`` -> OptionFlag
* ``role=parameter`` -> OptionName
* ``depth=8`` / ``scale=0.1`` -> OptionNumber
* ``path="lex.tsv"`` -> OptionString
* ``over=[a, b]`` -> OptionList
* ``via=product(a, b)`` -> OptionCall
"""

from __future__ import annotations

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
from quivers.dsl.parser._registry import ParseError, _Tree


def _walk_option_block(t: _Tree, vid: str) -> tuple[OptionEntry, ...]:
    """Walk an ``option_block`` vertex into a tuple of OptionEntry."""
    if t.kind(vid) != "option_block":
        raise ParseError(f"expected option_block, got {t.kind(vid)} at {vid}")
    entries: list[OptionEntry] = []
    for entry_vid in t.positional(vid):
        if t.kind(entry_vid) != "option_entry":
            continue
        entries.append(_walk_option_entry(t, entry_vid))
    return tuple(entries)


def _walk_option_entry(t: _Tree, vid: str) -> OptionEntry:
    key_vid = t.field(vid, "key")
    if key_vid is None:
        raise ParseError(f"option_entry missing key at {vid}")
    key = t.text(key_vid)
    line, col = t.line_col(vid)
    value_vid = t.field(vid, "value")
    if value_vid is None:
        return OptionEntry(key=key, value=OptionFlag(), line=line, col=col)
    return OptionEntry(
        key=key,
        value=_walk_option_value(t, value_vid),
        line=line,
        col=col,
    )


def _walk_option_value(t: _Tree, vid: str) -> OptionValue:
    """Map a tree-sitter option-value vertex to the OptionValue union."""
    k = t.kind(vid)
    if k == "identifier":
        return OptionName(value=t.text(vid))
    if k == "signed_number":
        # A ``signed_number`` wraps an integer or float literal with an
        # optional leading ``-``; the vertex text spans the sign, so a
        # single float() conversion carries it into the value.
        return OptionNumber(value=float(t.text(vid)))
    if k == "string":
        text = t.text(vid)
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return OptionString(value=text)
    if k == "option_list":
        items_vids = t.fields(vid, "item")
        return OptionList(
            items=tuple(_walk_option_value(t, iv) for iv in items_vids),
        )
    if k == "option_call":
        func_vid = t.field(vid, "func")
        if func_vid is None:
            raise ParseError(f"option_call missing func at {vid}")
        args_vids = t.fields(vid, "args")
        return OptionCall(
            func=t.text(func_vid),
            args=tuple(_walk_option_value(t, av) for av in args_vids),
        )
    raise ParseError(f"unexpected option value kind: {k}")


__all__ = []
