"""Validate the additive external-data syntax without rewriting source.

The grammar adds data_step as a new alternative of _program_step and leaves
all existing alternatives and fields intact. FinSet _ already parses as an
identifier size; its new meaning is opt-in and no released well-typed source
uses an unbound underscore as its cardinality. Existing source thus retains
its bytes and meaning. Both pinned parsers validate the byte-preserving edge.
"""

from quivers.cli.migrations._common import parse_validated_source


def migrate(source: bytes) -> bytes:
    """Validate source at both revisions and return the same bytes."""
    parse_validated_source("v0.64.2", source)
    parse_validated_source("HEAD", source, role="migration target")
    return source


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()

__all__ = []
