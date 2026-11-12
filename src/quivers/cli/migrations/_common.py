"""Shared infrastructure for one-hop QVR grammar migrations.

A *hop* migrates source bytes from one tagged QVR grammar revision
to the immediately following one. Composition (``compose_migration``
in [`quivers.cli.migrations`][quivers.cli.migrations]) chains hops to span any pair of
listed revisions.

## Pipeline

Every hop follows the same three-stage panproto-native pipeline,
which is grammar-bound by construction at the third stage:

1. **Parse** the source bytes with the *source* revision's
   tree-sitter grammar via
   `panproto.ParseEmitLens`. The result is a panproto
   `panproto.Schema` whose vertex kinds and edge kinds are
   the source grammar's rule names and field names; per-vertex
   constraints carry ``literal-value`` (leaf text),
   ``chose-alt-fingerprint`` / ``chose-alt-child-kinds`` (choice
   discriminators), and ``field:NAME`` constants for tree-sitter
   ``field('NAME', '<literal>')`` choices.
2. **Build** a new schema for the *target* revision by walking the
   source schema and emitting an in-memory JSON tree of vertices,
   edges, and constraints shaped by the target grammar's rules.
   Helpers in this module construct the JSON tree directly so the
   per-hop modules can focus on the per-declaration translation
   without re-implementing the encoding.
3. **Emit** with the target revision's ``emit_pretty``, which walks
   the target ``grammar.json``'s production rules and dispatches
   on ``chose-alt-*`` constraints. Any schema that does not satisfy
   the target grammar's rules raises a panproto ``ParseError`` here;
   that is the grammar binding the user requires.

## SchemaTree

Within a per-hop migrator, the working representation is a Python
``SchemaTree`` --- a nested dict structure that builds up exactly
what ``panproto.Schema.from_json`` consumes. The helpers below
emit the panproto JSON envelope from a list of ``SchemaTree``
roots so per-hop code never touches the envelope directly.
"""

from __future__ import annotations

import sys

import json
from typing import Callable

import panproto

from quivers.cli.migrations._grammar import registry_for
from quivers.cli.migrations._vcs import blame_kind


_SOURCE_FILE_VID = "parse_emit_lens"


class MigrationError(Exception):
    """Raised when a requested migration cannot be composed, or when
    a source construct admits no rewrite under the target grammar."""


# ---------------------------------------------------------------------------
# Vertex tree builder
# ---------------------------------------------------------------------------


class Vertex:
    """One vertex in a target schema under construction.

    Attributes
    ----------
    vid : str
        The vertex id used in the panproto Schema. Per-hop migrators
        receive a `VidGen` from `migrate_source` to
        keep ids unique across the whole target schema.
    kind : str
        The target grammar's rule name for this vertex.
    constraints : list[tuple[str, str]]
        ``(sort, value)`` pairs. Always include
        ``chose-alt-fingerprint`` / ``chose-alt-child-kinds`` where
        the target grammar's emit walker dispatches on a choice;
        include ``literal-value`` on every leaf token; include
        ``field:NAME`` for any tree-sitter ``field('NAME',
        '<literal>')`` choice the target grammar declares.
    children : list[tuple[str, Vertex]]
        ``(edge_kind, child_vertex)`` pairs in document order. The
        ``edge_kind`` is ``"child_of"`` for an unnamed positional
        child or the field name for a named field.
    """

    __slots__ = ("vid", "kind", "constraints", "children")

    def __init__(
        self,
        vid: str,
        kind: str,
        constraints: list[tuple[str, str]] | None = None,
        children: list[tuple[str, Vertex]] | None = None,
    ) -> None:
        self.vid = vid
        self.kind = kind
        self.constraints = constraints if constraints is not None else []
        self.children = children if children is not None else []


class VidGen:
    """Allocator for vertex ids that stay unique across a target
    schema. The ``parse_emit_lens`` root keeps its canonical id;
    every other vertex gets ``parse_emit_lens::$N`` for a fresh ``N``."""

    def __init__(self) -> None:
        self._n = 0

    def fresh(self) -> str:
        vid = f"{_SOURCE_FILE_VID}::${self._n}"
        self._n += 1
        return vid


# ---------------------------------------------------------------------------
# JSON envelope assembly
# ---------------------------------------------------------------------------


def _walk_vertex(
    vertex: Vertex,
    vertices: dict,
    edges: list,
    constraints: dict,
    outgoing: dict,
    incoming: dict,
    between: dict,
) -> None:
    vertices[vertex.vid] = {
        "id": vertex.vid,
        "kind": vertex.kind,
        "nsid": None,
    }
    if vertex.constraints:
        constraints[vertex.vid] = [
            {"sort": sort, "value": value} for sort, value in vertex.constraints
        ]
    for edge_kind, child in vertex.children:
        edge_obj = {
            "src": vertex.vid,
            "tgt": child.vid,
            "kind": edge_kind,
            "name": None,
        }
        # ``edges`` is a list of ``[edge, edge.kind]`` pairs in panproto's
        # JSON shape (it serializes the HashMap<Edge, Name> as kv pairs).
        edges.append([edge_obj, edge_kind])
        outgoing.setdefault(vertex.vid, []).append(edge_obj)
        incoming.setdefault(child.vid, []).append(edge_obj)
        between.setdefault((vertex.vid, child.vid), []).append(edge_obj)
        _walk_vertex(
            child,
            vertices,
            edges,
            constraints,
            outgoing,
            incoming,
            between,
        )


def _build_schema_json(protocol_name: str, root: Vertex) -> str:
    """Serialize a target ``Vertex`` tree to the JSON envelope
    panproto's `Schema.from_json` consumes."""
    vertices: dict = {}
    edges: list = []
    constraints: dict = {}
    outgoing: dict = {}
    incoming: dict = {}
    between: dict = {}
    _walk_vertex(root, vertices, edges, constraints, outgoing, incoming, between)

    between_list = [
        [[src, tgt], edge_list] for (src, tgt), edge_list in between.items()
    ]

    envelope = {
        "protocol": protocol_name,
        "vertices": vertices,
        "edges": edges,
        "hyper_edges": {},
        "constraints": constraints,
        "required": {},
        "nsids": {},
        "entries": [],
        "variants": {},
        "orderings": [],
        "recursion_points": {},
        "spans": {},
        "usage_modes": [],
        "nominal": {},
        "coercions": [],
        "mergers": {},
        "defaults": {},
        "policies": {},
        "outgoing": outgoing,
        "incoming": incoming,
        "between": between_list,
    }
    return json.dumps(envelope)


# ---------------------------------------------------------------------------
# Source-schema walking helpers
# ---------------------------------------------------------------------------


class SchemaView:
    """Indexed read-only view of a parsed panproto Schema.

    Per-hop migrators receive a `SchemaView` and walk it to
    construct the target ``Vertex`` tree. The view caches a
    vertex-by-id map, an outgoing-edges map keyed by source vertex,
    and a constraint map so per-vertex lookups are constant-time.
    """

    __slots__ = ("schema", "source", "_vertices", "_outgoing", "_consts")

    def __init__(self, schema: panproto.Schema, source: bytes) -> None:
        self.schema = schema
        self.source = source
        self._vertices = {v.id: v for v in schema.vertices}
        self._outgoing: dict[str, list] = {}
        for e in schema.edges:
            self._outgoing.setdefault(e.src, []).append(e)
        self._consts: dict[str, dict[str, str]] = {}

    def kind(self, vid: str) -> str:
        return self._vertices[vid].kind

    def consts(self, vid: str) -> dict[str, str]:
        cached = self._consts.get(vid)
        if cached is None:
            cached = {c.sort: c.value for c in self.schema.constraints_for(vid)}
            self._consts[vid] = cached
        return cached

    def text(self, vid: str) -> str:
        c = self.consts(vid)
        lit = c.get("literal-value")
        if lit is not None:
            return lit
        sb = c.get("start-byte")
        eb = c.get("end-byte")
        if sb is None or eb is None:
            return ""
        return self.source[int(sb) : int(eb)].decode("utf-8")

    def field(self, parent_vid: str, name: str) -> str | None:
        for edge in self._outgoing.get(parent_vid, []):
            if edge.kind == name:
                return edge.tgt
        return None

    def fields(self, parent_vid: str, name: str) -> list[str]:
        out: list[str] = []
        for edge in self._outgoing.get(parent_vid, []):
            if edge.kind == name:
                out.append(edge.tgt)
        out.sort(key=lambda vid: int(self.consts(vid).get("start-byte", "0")))
        return out

    def positional(self, parent_vid: str) -> list[str]:
        kids = [
            e.tgt for e in self._outgoing.get(parent_vid, []) if e.kind == "child_of"
        ]
        kids.sort(key=lambda vid: int(self.consts(vid).get("start-byte", "0")))
        return kids

    def outgoing_vids(self, parent_vid: str) -> list[str]:
        """Document-order target vids of every outgoing edge of
        ``parent_vid``, regardless of edge kind. Use this to walk a
        whole declaration subtree when the migration logic keys on
        vertex kinds rather than field names."""
        kids = [e.tgt for e in self._outgoing.get(parent_vid, [])]
        kids.sort(key=lambda vid: int(self.consts(vid).get("start-byte", "0")))
        return kids

    def span(self, vid: str) -> tuple[int, int]:
        """Byte span ``(start, end)`` of ``vid`` in the source."""
        c = self.consts(vid)
        return int(c["start-byte"]), int(c["end-byte"])

    def interstitials(self, vid: str) -> list[tuple[int, str]]:
        """The anonymous-token / whitespace runs the parser recorded
        on ``vid``, as ``(absolute_start_byte, text)`` pairs in
        document order. These carry every literal keyword and
        punctuation token of the vertex's own production (named
        children are excluded), so migrations locate anonymous
        tokens (``'=>'``, ``'over'``, ``'let'``, ...) precisely."""
        c = self.consts(vid)
        out: list[tuple[int, str]] = []
        i = 0
        while True:
            text = c.get(f"interstitial-{i}")
            if text is None:
                break
            out.append((int(c[f"interstitial-{i}-start-byte"]), text))
            i += 1
        return out

    def top_level_decls(self) -> list[str]:
        """Document-order vertex ids of every top-level declaration
        under the synthetic ``source_file`` root."""
        return self.positional(_SOURCE_FILE_VID)

    def body_children(
        self,
        parent_vid: str,
        header_fields: frozenset[str],
    ) -> list[tuple[str, str]]:
        """Return ``(edge_kind, target_vid)`` pairs for every
        outgoing edge of ``parent_vid`` whose ``edge.kind`` is NOT
        in ``header_fields``, in document order by target start-byte.

        Use this to walk a declaration's body (steps, sub-blocks,
        and interleaved ``line_comment``/``doc_comment`` extras)
        without picking up header fields like ``name``, ``domain``,
        ``options``, etc.
        """
        items: list[tuple[str, str]] = []
        for edge in self._outgoing.get(parent_vid, []):
            if edge.kind in header_fields:
                continue
            items.append((edge.kind, edge.tgt))
        items.sort(
            key=lambda kv: int(self.consts(kv[1]).get("start-byte", "0")),
        )
        return items

    def list_children(
        self,
        parent_vid: str,
        item_edge_kind: str,
    ) -> list[tuple[str, str, str]]:
        """Walk the outgoing edges of a bracketed-list vertex in
        document order. Returns ``(role, kind, target_vid)`` triples
        where ``role`` is one of:

        * ``"item"``    -- a structural element of the list, reached
                           via an edge of kind ``item_edge_kind``.
        * ``"comment"`` -- a ``line_comment`` / ``doc_comment`` /
                           ``block_comment`` extra interspersed
                           inside the list.

        Edges of kinds other than ``item_edge_kind`` and the three
        comment kinds are skipped.

        Use this to drive a bracketed-list emitter that preserves
        interior comments: if any ``"comment"`` rows are present,
        emit multi-line; otherwise emit inline.
        """
        out: list[tuple[str, str, str]] = []
        for edge in self._outgoing.get(parent_vid, []):
            tgt_kind = self._vertices[edge.tgt].kind
            # Classify comment targets first; the parser may attach
            # ``line_comment``/``doc_comment``/``block_comment`` extras
            # via the same edge kind as items (typically ``child_of``).
            if tgt_kind in ("line_comment", "doc_comment", "block_comment"):
                out.append(("comment", tgt_kind, edge.tgt))
            elif edge.kind == item_edge_kind:
                out.append(("item", tgt_kind, edge.tgt))
        out.sort(
            key=lambda triple: int(self.consts(triple[2]).get("start-byte", "0")),
        )
        return out


# ---------------------------------------------------------------------------
# Bracketed-list emission: auto-detect multi-line when source has
# interior comments
# ---------------------------------------------------------------------------


def emit_bracketed_list(
    view: SchemaView,
    parent_vid: str,
    item_edge_kind: str,
    item_emitter: Callable[[SchemaView, str], str],
    *,
    open_char: str = "[",
    close_char: str = "]",
    indent: str = "",
    item_indent_step: str = "    ",
) -> str:
    """Emit ``parent_vid``'s items as a bracketed comma-separated
    list. Returns ``"OPEN i1, i2, i3 CLOSE"`` when no interior
    comments are present, or a multi-line form when they are:

    ::

        OPEN
            i1,
            # interior comment
            i2,
        CLOSE

    The grammar's `bracketedList` helper accepts both forms;
    this emitter picks based on whether the source vertex has
    ``line_comment`` / ``doc_comment`` / ``block_comment`` children
    interspersed with the items. For migration source revisions
    that don't allow interior-bracket comments, all output is
    inline (no behaviour change); for revisions that do, comments
    survive.

    ``item_emitter`` is a callable ``(view, item_vid) -> str``
    that renders one element's text (no surrounding comma or
    newline).
    """
    children = view.list_children(parent_vid, item_edge_kind)
    has_comments = any(role == "comment" for role, _kind, _vid in children)

    if not has_comments:
        items_text = ", ".join(
            item_emitter(view, vid) for role, _kind, vid in children if role == "item"
        )
        return f"{open_char}{items_text}{close_char}"

    # Multi-line form: each item on its own line with a trailing
    # comma; comments interleaved as their original source text.
    inner_indent = indent + item_indent_step
    lines: list[str] = [open_char]
    for role, _kind, vid in children:
        if role == "comment":
            lines.append(inner_indent + view.text(vid).rstrip())
        else:
            lines.append(inner_indent + item_emitter(view, vid) + ",")
    lines.append(indent + close_char)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Identity conversions for unchanged-between-revisions decls
# ---------------------------------------------------------------------------


def parse_template(target_rev: str, template_source: str) -> Vertex:
    """Parse a reference HEAD-or-target snippet, return the first
    top-level declaration as a `Vertex` tree ready to be
    composed into a target schema.

    Per-hop converters use this to obtain the target-grammar's
    correctly-shaped declaration form (vertex kinds, edges, choice
    discriminators, ``field:NAME`` constants), then mutate the
    domain-specific leaves (identifier names, integer literals, type
    expressions) to inject the source declaration's data.

    The returned `Vertex` tree has fresh ids allocated under
    a temporary `VidGen`; callers should reassign ids if they
    need to merge with another tree, which `migrate_source`
    handles automatically since it walks every returned ``Vertex``
    with its own ``VidGen``.
    """
    src_reg = registry_for(target_rev)
    src_lens = src_reg.lens("qvr")
    src_schema = src_lens.parse(template_source.encode("utf-8"))
    view = SchemaView(src_schema, template_source.encode("utf-8"))
    top = view.top_level_decls()
    if not top:
        raise ValueError(
            f"template_source did not parse as any top-level decl: {template_source!r}",
        )
    vids = VidGen()
    return clone_vertex(view, top[0], vids)


def reassign_ids(vertex: Vertex, vids: VidGen) -> Vertex:
    """Return a copy of ``vertex`` with all ids freshly allocated
    from ``vids``. Used after `parse_template` to merge a
    template into a target schema that uses a different id space."""
    new_vid = vids.fresh()
    new_vertex = Vertex(
        vid=new_vid,
        kind=vertex.kind,
        constraints=list(vertex.constraints),
    )
    for edge_kind, child in vertex.children:
        new_vertex.children.append((edge_kind, reassign_ids(child, vids)))
    return new_vertex


def set_constraint(vertex: Vertex, sort: str, value: str) -> None:
    """Replace (or insert) the ``sort`` constraint on ``vertex``."""
    for i, (s, _) in enumerate(vertex.constraints):
        if s == sort:
            vertex.constraints[i] = (sort, value)
            return
    vertex.constraints.append((sort, value))


def find_field(vertex: Vertex, field_name: str) -> Vertex | None:
    """Return the child of ``vertex`` reached by the named-field
    edge ``field_name`` (e.g. ``"name"``, ``"value"``), or
    ``None`` if no such field child exists."""
    for edge_kind, child in vertex.children:
        if edge_kind == field_name:
            return child
    return None


def clone_vertex(view: SchemaView, src_vid: str, vids: VidGen) -> Vertex:
    """Build a target ``Vertex`` that is a structural copy of
    ``src_vid`` in ``view``.

    Used by per-hop migrators for declarations whose surface is
    unchanged between adjacent revisions: walk the source vertex,
    keep every constraint and edge while allocating fresh target
    vertex ids.
    """
    new_vid = vids.fresh()
    vertex = Vertex(
        vid=new_vid,
        kind=view.kind(src_vid),
        constraints=list(view.consts(src_vid).items()),
    )
    edges_in_order = list(view._outgoing.get(src_vid, []))
    edges_in_order.sort(
        key=lambda e: int(view.consts(e.tgt).get("start-byte", "0")),
    )
    for edge in edges_in_order:
        child = clone_vertex(view, edge.tgt, vids)
        vertex.children.append((edge.kind, child))
    return vertex


# ---------------------------------------------------------------------------
# Per-hop entry point
# ---------------------------------------------------------------------------


DeclConverter = Callable[[SchemaView, str], str]


def _blame_unknown_kind(kind: str) -> str:
    """Look up the panproto VCS for when ``kind`` was introduced or
    last present. Returns a human-readable diagnostic, or the empty
    string if the lookup fails (e.g. VCS unavailable)."""
    try:
        report = blame_kind(kind)
    except Exception:
        return ""
    parts: list[str] = []
    if report.introduced_at_tag:
        parts.append(f"introduced at {report.introduced_at_tag}")
    if report.last_present_at_tag:
        parts.append(f"last present at {report.last_present_at_tag}")
    if not parts:
        return ""
    return "VCS blame: " + "; ".join(parts) + "."


def validate_decl(target_rev: str, text: str) -> None:
    """Parse ``text`` through ``target_rev``'s grammar and raise if
    any vertex parses as ``ERROR``. This is the per-decl
    grammar-binding gate: every per-hop converter must produce text
    that round-trips through the target lens as a single
    ``_statement`` without ERROR nodes."""
    reg = registry_for(target_rev)
    lens = reg.lens("qvr")
    schema = lens.parse(text.encode("utf-8"))
    validate_parsed_schema(
        schema,
        target_rev,
        role=f"converted decl text {text!r}",
    )


def validate_parsed_schema(
    schema: panproto.Schema,
    revision: str,
    *,
    role: str,
) -> None:
    """Reject explicit and zero-width tree-sitter recovery vertices.

    Tree-sitter does not represent every syntax error with an ``ERROR`` node.
    For a missing required token it may instead emit a zero-width vertex.  Both
    forms must fail closed or an identity migration can certify incomplete
    source as valid.
    """
    recovery: list[str] = []
    for vertex in schema.vertices:
        if vertex.kind == "ERROR":
            recovery.append(f"ERROR vertex {vertex.id}")
            continue
        if vertex.id == _SOURCE_FILE_VID:
            continue
        constraints = {
            constraint.sort: constraint.value
            for constraint in schema.constraints_for(vertex.id)
        }
        start = constraints.get("start-byte")
        if start is not None and start == constraints.get("end-byte"):
            recovery.append(f"missing {vertex.kind!r} token at vertex {vertex.id}")
    if recovery:
        raise MigrationError(
            f"{role} does not parse under {revision}: " + "; ".join(recovery[:5])
        )


def parse_validated_source(
    revision: str,
    source: bytes,
    *,
    role: str = "source",
) -> panproto.Schema:
    """Parse bytes with a pinned revision and reject recovery vertices."""
    try:
        schema = registry_for(revision).lens("qvr").parse(source)
    except Exception as error:
        raise MigrationError(
            f"{role} does not parse under {revision}: {error}"
        ) from error
    validate_parsed_schema(schema, revision, role=role)
    return schema


def migrate_source(
    source: bytes,
    source_rev: str,
    target_rev: str,
    decl_converters: dict[str, DeclConverter],
) -> bytes:
    """Run one hop end-to-end.

    Parses ``source`` with the source revision's tree-sitter
    grammar, walks the resulting Schema's top-level declarations,
    dispatches each to a per-kind converter that returns target-
    shaped text, and concatenates the result. Each converter's
    output is grammar-bound: it must parse cleanly through the
    target revision's lens or `validate_decl` raises.

    Decl kinds absent from ``decl_converters`` are passed through
    as their source byte slice (only safe when the kind exists with
    the same shape in the target grammar; per-hop modules pin
    converters for every kind that differs).

    Parameters
    ----------
    source : bytes
        Source bytes to migrate.
    source_rev : str
        Tree-sitter parser revision to parse with.
    target_rev : str
        Tree-sitter parser revision to validate each decl against.
    decl_converters : dict[str, DeclConverter]
        Mapping from source-revision top-level decl ``kind`` to a
        callable returning the target-revision source text for that
        decl. Returned text must include its trailing newline.

    Returns
    -------
    bytes
        Concatenated target-shape source bytes. The full output is
        validated by re-parsing through the target lens; any ERROR
        vertex in the assembled schema raises ``ValueError``.
    """
    src_schema = parse_validated_source(source_rev, source)
    view = SchemaView(src_schema, source)

    parts: list[str] = []
    for decl_vid in view.top_level_decls():
        kind = view.kind(decl_vid)
        # tree-sitter "extras" (``line_comment``, ``doc_comment``)
        # appear in the parse-tree Schema as positional children of
        # source_file. Pass them through as their raw source bytes
        # so explanatory header comments and inline notes survive
        # migration; the target grammar treats them as extras and
        # absorbs them silently between top-level statements.
        if kind in ("line_comment", "doc_comment"):
            text = view.text(decl_vid)
            if not text.endswith("\n"):
                text += "\n"
            parts.append(text)
            continue
        converter = decl_converters.get(kind)
        if converter is None:
            # Unknown decl kind for this hop. Look up the VCS for
            # when the rule was introduced or removed and surface
            # that information. The source bytes are then checked
            # against the target grammar like any converted
            # declaration, so a declaration the target does not admit
            # stops the migration with that diagnostic.
            blame_msg = _blame_unknown_kind(kind)
            if blame_msg:
                print(
                    f"qvr migrate [{source_rev} -> {target_rev}]: "
                    f"no converter for {kind!r}. {blame_msg}",
                    file=sys.stderr,
                )
            text = view.text(decl_vid)
            if not text.endswith("\n"):
                text += "\n"
        else:
            text = converter(view, decl_vid)
        validate_decl(target_rev, text)
        parts.append(text)

    result = "".join(parts)
    # Whole-file grammar binding: parse the concatenated output as a
    # complete source file through the target lens.
    tgt_lens = registry_for(target_rev).lens("qvr")
    final = tgt_lens.parse(result.encode("utf-8"))
    validate_parsed_schema(final, target_rev, role="assembled migration output")
    return result.encode("utf-8")
