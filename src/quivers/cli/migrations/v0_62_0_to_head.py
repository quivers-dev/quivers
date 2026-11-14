"""Rename declaration options from the released QVR surface.

Morphism ``role=latent`` becomes ``role=parameter``, program ``over=``
becomes ``posterior_of=``, and composition ``level=bilinear_form`` becomes
``level=magmoid``. The grammar accepts these options as identifiers, so
the parser and schema are identical while their semantic vocabulary changes.

Edits use the named option fields of each declaration's own header. Other
options, comments, strings, and user identifiers retain their original bytes.
Source, changed declarations, and assembled output pass the pinned parse gates.
"""

from __future__ import annotations

from quivers.cli.migrations._common import (
    MigrationError,
    SchemaView,
    parse_validated_source,
)


_SOURCE_REV = "v0.62.0"
_TARGET_REV = "HEAD"
type _Edit = tuple[int, int, bytes]
_DECLARATION_KINDS = frozenset({"morphism_decl", "program_decl", "composition_decl"})


def _collect_option_edits(view: SchemaView, declaration: str) -> list[_Edit]:
    """Locate renamed identifiers in one declaration's header.

    Parameters
    ----------
    view
        The validated source schema and its bytes.
    declaration
        The vertex of a morphism, program, or composition declaration.

    Returns
    -------
    list[tuple[int, int, bytes]]
        Nonoverlapping replacements at absolute source byte offsets.

    Raises
    ------
    MigrationError
        If an option lacks its required key.
    """
    options = view.field(declaration, "options")
    if options is None:
        return []
    kind = view.kind(declaration)
    edits: list[_Edit] = []
    for entry in view.outgoing_vids(options):
        if view.kind(entry) != "option_entry":
            continue
        key = view.field(entry, "key")
        if key is None:
            start, _ = view.span(entry)
            prefix = view.source[:start]
            line = prefix.count(b"\n") + 1
            column = start - prefix.rfind(b"\n")
            raise MigrationError(
                f"line {line}, column {column}: option lacks a key; "
                "write a named declaration option"
            )
        name = view.text(key)
        value = view.field(entry, "value")
        if kind == "program_decl" and name == "over":
            start, end = view.span(key)
            edits.append((start, end, b"posterior_of"))
        elif value is not None and view.kind(value) == "identifier":
            spelling = view.text(value)
            replacement: bytes | None = None
            if kind == "morphism_decl" and name == "role" and spelling == "latent":
                replacement = b"parameter"
            elif (
                kind == "composition_decl"
                and name == "level"
                and spelling == "bilinear_form"
            ):
                replacement = b"magmoid"
            if replacement is not None:
                start, end = view.span(value)
                edits.append((start, end, replacement))
    return edits


def _apply_edits(source: bytes, edits: list[_Edit], *, offset: int = 0) -> bytes:
    """Replace validated identifier spans from right to left.

    Parameters
    ----------
    source
        The file or declaration bytes.
    edits
        Absolute byte spans and their replacements.
    offset
        The absolute start of the supplied source slice.

    Returns
    -------
    bytes
        The source with only the specified identifiers replaced.
    """
    result = bytearray(source)
    for start, end, replacement in sorted(edits, reverse=True):
        result[start - offset : end - offset] = replacement
    return bytes(result)


def migrate(source: bytes) -> bytes:
    """Rename the three declaration options without reformatting source.

    Parameters
    ----------
    source
        A complete file in the released QVR syntax.

    Returns
    -------
    bytes
        Source using the parameter, posterior, and magmoid option names.

    Raises
    ------
    MigrationError
        If input or output has parser recovery vertices.
    """
    view = SchemaView(parse_validated_source(_SOURCE_REV, source), source)
    edits: list[_Edit] = []
    for vertex in view.schema.vertices:
        if vertex.kind not in _DECLARATION_KINDS:
            continue
        replacements = _collect_option_edits(view, vertex.id)
        if not replacements:
            continue
        start, end = view.span(vertex.id)
        declaration = _apply_edits(source[start:end], replacements, offset=start)
        parse_validated_source(_TARGET_REV, declaration, role="migration declaration")
        edits.extend(replacements)
    result = _apply_edits(source, edits)
    parse_validated_source(_TARGET_REV, result, role="migration target")
    return result


SOURCE_RULE_COVERAGE: frozenset[str] = _DECLARATION_KINDS | frozenset({"option_entry"})


__all__ = []
