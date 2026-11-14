"""One-hop migrator from the released v0.19 surface to QVR v0.62.

The target accepts hanging-indented argument and list forms. Distribution-family
construction is keyword-only, so this hop names every positional family
argument according to the family catalog. Former splatted vector calls to
``Categorical`` and ``Dirichlet`` are gathered into their one vector parameter.
Other source remains byte-preserving.
"""

from __future__ import annotations

from quivers.cli.migrations._common import SchemaView, parse_validated_source
from quivers.dsl.family_schemas import family_parameter_names


_SOURCE_REV = "v0.19.0"
_TARGET_REV = "v0.62.0"


_FAMILY_CALL_KINDS: dict[str, str] = {
    "sample_step": "morphism",
    "observe_step": "morphism",
    "marginalize_step": "morphism",
    "morphism_init_family": "family",
    "morphism_call": "callee",
    "family_call_arg": "family",
}


_SPLATTED_VECTOR_FAMILIES: frozenset[str] = frozenset({"Categorical", "Dirichlet"})


def _keyword_argument_edits(view: SchemaView) -> list[tuple[int, str]]:
    """Name positional distribution-family arguments."""
    edits: list[tuple[int, str]] = []
    pending = list(reversed(view.top_level_decls()))
    while pending:
        vid = pending.pop()
        pending.extend(reversed(view.outgoing_vids(vid)))
        family_field = _FAMILY_CALL_KINDS.get(view.kind(vid))
        if family_field is None:
            continue
        family = view.field(vid, family_field)
        if family is None:
            continue
        family_name = view.text(family)
        parameters = family_parameter_names(family_name)
        if parameters is None:
            continue
        args = view.fields(vid, "args")
        if not args:
            continue
        positional = [arg for arg in args if view.kind(arg) != "named_draw_arg"]
        if not positional:
            continue
        named = {
            view.text(parameter)
            for arg in args
            if view.kind(arg) == "named_draw_arg"
            for parameter in (view.field(arg, "parameter"),)
            if parameter is not None
        }
        available = [parameter for parameter in parameters if parameter not in named]
        if (
            family_name in _SPLATTED_VECTOR_FAMILIES
            and len(parameters) == 1
            and len(positional) > 1
            and not named
        ):
            first_start, _ = view.span(positional[0])
            _, last_end = view.span(positional[-1])
            edits.extend(((first_start, f"{parameters[0]}=["), (last_end, "]")))
            continue
        for argument, parameter in zip(positional, available, strict=False):
            start, _ = view.span(argument)
            edits.append((start, f"{parameter}="))
    return edits


def migrate(source: bytes) -> bytes:
    """Name positional family arguments and validate the result."""
    schema = parse_validated_source(_SOURCE_REV, source)
    view = SchemaView(schema, source)
    migrated = bytearray(source)
    for position, insertion in sorted(_keyword_argument_edits(view), reverse=True):
        migrated[position:position] = insertion.encode("utf-8")
    result = bytes(migrated)
    parse_validated_source(_TARGET_REV, result, role="migration target")
    return result


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()


__all__ = []
