"""The vertices of a parsed runtime schema that a renderer grafts."""

from __future__ import annotations

from collections.abc import Iterable

import panproto


def subtree_vertices(schema: panproto.Schema, roots: Iterable[str]) -> set[str]:
    """Find every vertex reachable from some roots along outgoing edges.

    Parameters
    ----------
    schema : panproto.Schema
        The parsed schema.
    roots : Iterable[str]
        The vertices to start from.

    Returns
    -------
    set[str]
        The roots and everything reachable from them.
    """
    outgoing: dict[str, list[str]] = {}
    for edge in schema.edges:
        outgoing.setdefault(edge.src, []).append(edge.tgt)
    seen = set(roots)
    frontier = list(seen)
    while frontier:
        for target in outgoing.get(frontier.pop(), ()):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def vertex_kinds(schema: panproto.Schema) -> dict[str, str]:
    """Read the kind of every vertex of a schema.

    Parameters
    ----------
    schema : panproto.Schema
        The schema.

    Returns
    -------
    dict[str, str]
        Each vertex's kind, by its id.
    """
    return {vertex.id: vertex.kind for vertex in schema.vertices}


__all__ = ["subtree_vertices", "vertex_kinds"]
