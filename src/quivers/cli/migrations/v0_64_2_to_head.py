"""Move static parameters and arguments into square-bracket telescopes.

Released programs put object, morphism, and scalar template parameters in one
parenthesized list. The target places object and morphism parameters in ``[]``
and keeps scalar values in ``()``. Calls to locally declared templates partition
their positional actuals by the same parameter order. Rule variables and schema
pattern parameters likewise change their delimiters from ``()`` to ``[]``.

Edits operate on the pinned historical CST's named fields and punctuation spans.
Unchanged source remains byte-identical; partitioned lists retain every item's
source bytes and comments, with same-line comments traveling with their item. Source
and assembled target both pass their pinned parse gates. Typed external value
headers and inferred extents are additive; no released ``data`` step exists.
``FinSet _`` previously parsed as an identifier cardinality, but an unbound
underscore was not well typed; its inference meaning thus affects no released
well-typed cardinality declaration.
Handler parameters are additive likewise: no released source writes ``with``
after a handler's answer type or after ``resume(...)``, nor parentheses after the
handler a ``handle`` installs.
Ground deduction rules and literal application arguments are additive
too, so existing deduction rules need no rewrite at this hop. The authored
proof interpretation, sparse relations, and fixed relation and string
graph score declarations likewise leave prior declarations unchanged.
Authored binary relation composition, explicit scoped interpretation
options, string factor graphs, and sparse relation transformations are
additive at this hop.

This is a directed migration, not a target-source formatter. Reapplying it to a
new static telescope fails the historical source gate rather than guessing at
its meaning. Source containing no relocated construct is byte-preserving.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from quivers.cli.migrations._common import (
    MigrationError,
    SchemaView,
    parse_validated_source,
)

_SOURCE_REV = "v0.64.2"
_TARGET_REV = "HEAD"
type _Edit = tuple[int, int, bytes]
_CALL_FIELDS = {
    "sample_step": "morphism",
    "observe_step": "morphism",
    "marginalize_step": "morphism",
    "morphism_call": "callee",
}


def _fail(view: SchemaView, vertex: str, message: str) -> MigrationError:
    """Locate a construct whose binding cannot be migrated unambiguously."""
    start, _ = view.span(vertex)
    prefix = view.source[:start]
    line = prefix.count(b"\n") + 1
    column = start - prefix.rfind(b"\n")
    return MigrationError(f"line {line}, column {column}: {message}")


def _list_delimiters(
    view: SchemaView,
    parent: str,
    *,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    """Locate this parent's list punctuation outside its named children."""
    tokens: list[tuple[int, str]] = []
    for start, text in view.interstitials(parent):
        position = start
        for character in text:
            if character in "()" and lower <= position < upper:
                tokens.append((position, character))
            position += len(character.encode("utf-8"))
    opens = [position for position, character in tokens if character == "("]
    closes = [position for position, character in tokens if character == ")"]
    if len(opens) != 1 or len(closes) != 1:
        raise _fail(
            view,
            parent,
            "cannot locate the old parameter list; split the declaration before migrating",
        )
    begin, end = opens[0], closes[0]
    return begin, end


def _partitioned_list(
    view: SchemaView,
    parent: str,
    items: list[str],
    static: tuple[bool, ...],
    *,
    lower: int,
    upper: int,
    call: bool,
    comment_positions: tuple[int, ...],
    comment_spans: tuple[tuple[int, int], ...],
) -> _Edit:
    """Partition one list, preserving item bytes and item-associated comments."""
    begin, end = _list_delimiters(view, parent, lower=lower, upper=upper)
    interior = view.source[begin + 1 : end]
    spans = [view.span(item) for item in items]
    starts = [start for start, _ in spans]
    leading: list[list[bytes]] = [[] for _ in items]
    trailing: list[list[bytes]] = [[] for _ in items]
    comment_start = bisect_right(comment_positions, begin)
    comment_stop = bisect_left(comment_positions, end)
    comments = comment_spans[comment_start:comment_stop]
    for start, stop in comments:
        previous = bisect_right(starts, start) - 1
        if previous >= 0 and stop <= spans[previous][1]:
            continue  # The item slice already contains this comment.
        comment = view.source[start:stop]
        if previous >= 0 and b"\n" not in view.source[spans[previous][1] : start]:
            trailing[previous].append(comment)
        else:
            leading[min(previous + 1, len(items) - 1)].append(comment)
    multiline = b"\n" in interior or bool(comments)
    newline = b"\r\n" if b"\r\n" in interior else b"\n"
    parent_start, _ = view.span(parent)
    line_start = view.source.rfind(b"\n", 0, parent_start) + 1
    indentation = view.source[line_start:parent_start]
    if indentation.strip(b" \t\r"):
        indentation = b""
    item_indent = indentation + b"    "
    groups: tuple[list[bytes], list[bytes]] = ([], [])
    for index, ((start, stop), is_static) in enumerate(zip(spans, static, strict=True)):
        value = view.source[start:stop]
        if multiline:
            before = b""
            for comment in leading[index]:
                before += comment + (
                    newline + item_indent
                    if comment.startswith(b"#") and not comment.startswith(b"#{")
                    else b" "
                )
            after = b""
            for comment in trailing[index]:
                after += b" " + comment
            value = item_indent + before + value + b"," + after
        groups[0 if is_static else 1].append(value)

    def render(chunks: list[bytes], opening: bytes, closing: bytes) -> bytes:
        if not chunks:
            return opening + closing
        if multiline:
            return (
                opening
                + newline
                + newline.join(chunks)
                + newline
                + indentation
                + closing
            )
        return opening + b", ".join(chunks) + closing

    static_values, runtime_values = groups
    replacement = render(static_values, b"[", b"]")
    if runtime_values or call:
        replacement += render(runtime_values, b"(", b")")
    return begin, end + 1, replacement


def _template_shapes(view: SchemaView) -> dict[str, tuple[bool, ...]]:
    """Recover positional static/value classifications from released headers."""
    templates: dict[str, tuple[bool, ...]] = {}
    for vertex in view.schema.vertices:
        if vertex.kind != "program_decl":
            continue
        parameters = view.fields(vertex.id, "params")
        shape: list[bool] = []
        for parameter in parameters:
            kind = view.field(parameter, "kind")
            shape.append(
                kind is not None and view.kind(kind) in {"object_kind", "morphism_kind"}
            )
        if not any(shape):
            continue
        name = view.field(vertex.id, "name")
        if name is None:
            raise _fail(
                view,
                vertex.id,
                "program lacks a name; name the template before migrating",
            )
        spelling = view.text(name)
        if spelling in templates:
            raise _fail(
                view,
                vertex.id,
                f"duplicate program {spelling!r}; give each template a distinct name before migrating",
            )
        templates[spelling] = tuple(shape)
    return templates


def migrate(source: bytes) -> bytes:
    """Partition released template headers/calls and relocate pattern binders.

    Parameters
    ----------
    source : bytes
        A complete file in released v0.64.2 syntax.

    Returns
    -------
    bytes
        Source in the current static/value telescope syntax.

    Raises
    ------
    MigrationError
        If source or target has parser recovery, or a known template call's
        actuals cannot be matched to its released positional parameters.
    """
    view = SchemaView(parse_validated_source(_SOURCE_REV, source), source)
    templates = _template_shapes(view)
    comment_spans = tuple(
        sorted(
            view.span(vertex.id)
            for vertex in view.schema.vertices
            if vertex.kind in {"line_comment", "block_comment", "doc_comment"}
        )
    )
    comment_positions = tuple(start for start, _ in comment_spans)
    edits: list[_Edit] = []
    for vertex in view.schema.vertices:
        kind = vertex.kind
        if kind == "program_decl":
            name = view.field(vertex.id, "name")
            if name is None or view.text(name) not in templates:
                continue
            parameters = view.fields(vertex.id, "params")
            domain = view.field(vertex.id, "domain")
            assert domain is not None
            edits.append(
                _partitioned_list(
                    view,
                    vertex.id,
                    parameters,
                    templates[view.text(name)],
                    lower=view.span(name)[1],
                    upper=view.span(domain)[0],
                    call=False,
                    comment_positions=comment_positions,
                    comment_spans=comment_spans,
                )
            )
        elif kind in {"schema_decl", "rule_decl"}:
            name = view.field(vertex.id, "name")
            assert name is not None
            _, upper = view.span(vertex.id)
            begin, end = _list_delimiters(
                view, vertex.id, lower=view.span(name)[1], upper=upper
            )
            edits.extend(((begin, begin + 1, b"["), (end, end + 1, b"]")))
        elif kind in _CALL_FIELDS:
            callee = view.field(vertex.id, _CALL_FIELDS[kind])
            if callee is None or view.text(callee) not in templates:
                continue
            shape = templates[view.text(callee)]
            arguments = view.fields(vertex.id, "args")
            if len(arguments) != len(shape):
                raise _fail(
                    view,
                    vertex.id,
                    f"template {view.text(callee)!r} declares {len(shape)} parameters but the call supplies {len(arguments)}; supply the complete positional argument list before migrating",
                )
            _, upper = view.span(vertex.id)
            options = view.field(vertex.id, "options")
            if options is not None:
                upper = view.span(options)[0]
            if kind == "marginalize_step":
                scope = view.fields(vertex.id, "scope")
                if scope:
                    upper = min(upper, view.span(scope[0])[0])
            edits.append(
                _partitioned_list(
                    view,
                    vertex.id,
                    arguments,
                    shape,
                    lower=view.span(callee)[1],
                    upper=upper,
                    call=True,
                    comment_positions=comment_positions,
                    comment_spans=comment_spans,
                )
            )
    result = bytearray(source)
    previous_start = len(source) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous_start:
            raise MigrationError(
                "overlapping parameter migrations; split nested template calls before migrating"
            )
        result[start:end] = replacement
        previous_start = start
    migrated = bytes(result)
    parse_validated_source(_TARGET_REV, migrated, role="migration target")
    return migrated


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(
    {
        "program_decl",
        "_program_param",  # Historical choice is replaced by the staged lists.
        "typed_program_param",
        "scalar_kind",
        "object_kind",
        "morphism_kind",
        "sample_step",
        "observe_step",
        "marginalize_step",
        "morphism_call",
        "schema_decl",
        "rule_decl",
    }
)

__all__ = []
