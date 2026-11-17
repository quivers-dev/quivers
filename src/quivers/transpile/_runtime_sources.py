"""Compose and index standalone target runtimes from shared source fragments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import panproto


def python_runtime_source(path: Path) -> bytes:
    """Expand shared probability fragments before parsing a Python runtime.

    Parameters
    ----------
    path : Path
        Ordinary target runtime or checked distribution bridge.

    Returns
    -------
    bytes
        Standalone source with one canonical probability implementation.

    Raises
    ------
    ValueError
        If a runtime's shared-fragment marker is incomplete or repeated.
    """
    source = path.read_text()
    for target in ("pyro", "pymc", "numpyro", "edward2"):
        start = f"# qvr-shared-probability:{target}:start"
        end = f"# qvr-shared-probability:{target}:end"
        if start not in source and end not in source:
            continue
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError(f"invalid probability fragment markers in {path.name}")
        prefix, rest = source.split(start)
        _, suffix = rest.split(end)
        fragment = path.parent / f"runtime_probability_{target}.py"
        source = prefix + fragment.read_text() + suffix
    return source.encode()


def javascript_runtime_source(path: Path) -> bytes:
    """Prepend shared scalar probability definitions to a JavaScript runtime.

    Parameters
    ----------
    path : Path
        Ordinary WebPPL or checked QIEC runtime body.

    Returns
    -------
    bytes
        Complete standalone JavaScript source for parsing and grafting.
    """
    return (
        (path.parent / "runtime_probability.js").read_bytes()
        + b"\n"
        + path.read_bytes()
    )


def python_runtime_helpers(
    path: Path,
) -> tuple[panproto.Schema, dict[str, str], dict[str, set[str]]]:
    """Parse composed Python helpers and index their structural grafts once.

    Parameters
    ----------
    path : Path
        Target runtime whose shared fragments are expanded before parsing.

    Returns
    -------
    tuple[panproto.Schema, dict[str, str], dict[str, set[str]]]
        Parsed schema, named class/function roots, and reachable vertex sets.

    Raises
    ------
    RuntimeError
        If the runtime declares no named helper classes or functions.
    """
    from quivers.transpile._pipeline import parser_registry

    schema = parser_registry().parse_with_protocol(
        "python", python_runtime_source(path), str(path)
    )
    vertices = {vertex.id: vertex for vertex in schema.vertices}
    children: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for edge in schema.edges:
        children.setdefault(edge.src, []).append(edge.tgt)
        if edge.kind == "name" and edge.tgt in vertices:
            names[edge.src] = edge.tgt
    roots: dict[str, str] = {}
    for vertex in schema.vertices:
        if vertex.kind not in {"class_definition", "function_definition"}:
            continue
        name_id = names.get(vertex.id)
        if name_id is None:
            continue
        name = next(
            (
                constraint.value
                for constraint in schema.constraints_for(name_id)
                if constraint.sort == "literal-value"
            ),
            None,
        )
        if name is not None:
            roots[name] = vertex.id
    if not roots:
        raise RuntimeError(f"no named helper definitions found in {path}")
    subtrees: dict[str, set[str]] = {}
    for name, root in roots.items():
        seen = {root}
        frontier = [root]
        while frontier:
            for child in children.get(frontier.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        subtrees[name] = seen
    return schema, roots, subtrees
