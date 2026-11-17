"""Low-level helpers shared by the per-statement walkers."""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
)
from quivers.dsl.parser._registry import ParseError, _Tree


def _required_text(
    t: _Tree, child_vid: str | None, parent_vid: str, field_name: str
) -> str:
    """Return the text of a required-by-grammar field, raising if missing.

    Tree-sitter guarantees the field exists on a successful parse, so
    a ``None`` here means the parse was corrupted (an ``ERROR`` node
    leaked through, or the grammar was edited without updating the
    walker).
    """
    if child_vid is None:
        raise ParseError(
            f"missing required {field_name!r} field at {parent_vid} (malformed parse)"
        )
    return t.text(child_vid)


def _required_field(t: _Tree, parent_vid: str, field_name: str) -> str:
    """Return a required-by-grammar field's vertex id, raising if missing."""
    child_vid = t.field(parent_vid, field_name)
    if child_vid is None:
        raise ParseError(
            f"missing required {field_name!r} field at {parent_vid} (malformed parse)"
        )
    return child_vid


def _field_text(t: _Tree, parent_vid: str, field_name: str) -> str:
    """Return the text of a required-by-grammar field."""
    return t.text(_required_field(t, parent_vid, field_name))


def _walk_draw_arg(t: _Tree, vid: str) -> DrawArg:
    """Walk a family-argument into a tagged
    [`DrawArg`][quivers.dsl.ast_nodes.DrawArg].

    Identifiers and bracket-index references walk to
    [`DrawArgName`][quivers.dsl.ast_nodes.DrawArgName] (the bracket
    form ``"theta[N]"`` is encoded as the variable's `text`). Numeric
    literals walk to
    [`DrawArgScalar`][quivers.dsl.ast_nodes.DrawArgScalar]. A
    `family_call_arg` (a nested `Family(...)` expression) walks to a
    [`DrawArgDist`][quivers.dsl.ast_nodes.DrawArgDist] carrying the
    family name and recursively-walked arguments; the compiler then
    recurses into the inner call to build a distribution-valued
    parameter for the outer family. A `list_arg` walks to a
    [`DrawArgList`][quivers.dsl.ast_nodes.DrawArgList] whose items
    are themselves draw args. A ``named_draw_arg`` becomes a
    [`DrawArgNamed`][quivers.dsl.ast_nodes.DrawArgNamed] entry that wraps
    the value and keeps its parameter name structurally.
    """
    k = t.kind(vid)
    if k == "named_draw_arg":
        pv = t.field(vid, "parameter")
        vv = t.field(vid, "value")
        if pv is None or vv is None:
            raise ParseError(f"named_draw_arg malformed at {vid}")
        line, col = t.line_col(vid)
        return DrawArgNamed(
            parameter=t.text(pv),
            value=_walk_draw_arg(t, vv),
            line=line,
            col=col,
        )
    if k == "identifier":
        return DrawArgName(text=t.text(vid))
    if k in ("signed_number", "integer", "float"):
        text = t.text(vid)
        value = float(text) if any(char in text for char in ".eE") else int(text)
        return DrawArgScalar(value=value)
    if k == "bracket_index_arg":
        nv = t.field(vid, "name")
        iv = t.field(vid, "index")
        if nv is None or iv is None:
            raise ParseError(f"bracket_index_arg malformed at {vid}")
        # The index field carries the bracket body verbatim. Downstream
        # consumers (plate-graph edges, transpilation) resolve each index
        # as a bare identifier naming an index tensor, so a compound or
        # non-identifier index is rejected here rather than silently
        # dropped.
        index_text = t.text(iv)
        tokens = [tok.strip() for tok in index_text.split(",")]
        if not all(tok.isidentifier() for tok in tokens):
            line, col = t.line_col(iv)
            raise ParseError(
                f"bracket index {index_text!r} at line {line}, col {col}: "
                f"each index must be a bare identifier naming an index tensor",
            )
        return DrawArgIndex(name=t.text(nv), indices=tuple(tokens))
    if k == "family_call_arg":
        fv = t.field(vid, "family")
        if fv is None:
            raise ParseError(f"family_call_arg malformed at {vid}")
        family = t.text(fv)
        args = tuple(_walk_draw_arg(t, av) for av in t.fields(vid, "args"))
        return DrawArgDist(family=family, args=args)
    if k == "list_arg":
        items = tuple(
            _walk_draw_arg(t, cv)
            for cv in t.positional(vid)
            if t.kind(cv) not in ("[", "]", ",")
        )
        return DrawArgList(items=items)
    raise ParseError(f"unexpected draw arg kind: {k}")


__all__ = ["_field_text", "_required_field", "_required_text", "_walk_draw_arg"]
