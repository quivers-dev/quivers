"""Top-level parse entry points and whole-tree syntax validation."""

from __future__ import annotations

import re
from pathlib import Path

import panproto

from quivers.dsl.ast_nodes import Module, Statement
from quivers.dsl.parser._registry import ParseError, _Tree, _registry
from quivers.dsl.parser.statements import _walk_statement

_SNIPPET_MAX = 60

#: The words a let expression reads as something other than a name: the
#: unit value ``unit``, the Booleans ``true`` and ``false``, the negation
#: ``not``, and the comprehension keyword ``factor``. A binder spelled as
#: one of them could never be referred to, since every later occurrence
#: parses as the literal or the keyword, so a binder may not use them.
RESERVED_LET_WORDS: dict[str, str] = {
    "unit": "the unit value",
    "true": "the Boolean literal true",
    "false": "the Boolean literal false",
    "not": "Boolean negation",
    "factor": "the factor comprehension",
}

#: A reserved word in a lambda parameter's place: after a binding's
#: ``=``, an argument list's opening parenthesis or comma, or another
#: parameter's arrow. Such a lambda does not parse as one, so the source
#: text, not the tree, shows where it is bound.
_RESERVED_LAMBDA = re.compile(
    rb"(?:[(,=]|->)\s*\b(" + "|".join(RESERVED_LET_WORDS).encode() + rb")\s*->"
)

#: A line comment, blanked before the lambda scan reads the source.
_LINE_COMMENT = re.compile(rb"#[^\n]*")

#: Each binder position, as the CST node kind holding it and the field
#: naming the bound identifier.
_BINDER_FIELDS: dict[str, tuple[str, str]] = {
    "sample_step": ("vars", "sample site"),
    "observe_step": ("vars", "observed site"),
    "marginalize_step": ("var", "marginalized variable"),
    "let_step": ("name", "let binding"),
    "call_step": ("name", "call binding"),
    "score_step": ("name", "score"),
    "let_factor_binder": ("var", "factor binder"),
    "let_lambda": ("param", "lambda parameter"),
    "program_decl": ("params", "program parameter"),
    "typed_program_param": ("name", "program parameter"),
}

_PROGRAM_LINE = re.compile(rb"^[ \t]*program\b", re.MULTILINE)
_RETURN_WORD = re.compile(rb"\breturn\b")


def parse(source: str | bytes, file_path: str = "<source>") -> Module:
    """Parse `.qvr` source into a `Module`.

    Parameters
    ----------
    source : str | bytes
        The source text, or its UTF-8 bytes.
    file_path : str
        The path recorded on diagnostics.

    Returns
    -------
    Module
        The module's statements in source order.

    Raises
    ------
    ParseError
        If the source has a syntax error or a node the AST cannot
        represent.
    """
    if isinstance(source, str):
        source_bytes = source.encode("utf-8")
    else:
        source_bytes = source

    try:
        schema = _registry().parse_with_protocol("qvr", source_bytes, file_path)
    except panproto.PanprotoError as exc:
        raise ParseError(f"{file_path}: panproto failed to parse: {exc}") from exc
    tree = _Tree(schema, source_bytes)

    # A reserved binder name breaks every later reference to it, so it is
    # reported where it is bound, ahead of the syntax damage it causes.
    _reject_reserved_binders(tree, file_path)
    _reject_malformed(tree, file_path)

    root_id = next(
        (vid for vid in tree.vertices if tree.kind(vid) == "source_file"),
        None,
    )
    if root_id is None:
        raise ParseError(
            f"{file_path}: source failed to parse (the tree has no source_file "
            "root); check that every program body ends with a return step"
        )

    statements: list[Statement] = []
    for child in tree.positional(root_id):
        ckind = tree.kind(child)
        if ckind in ("line_comment", "block_comment"):
            # plain comments are extras and are dropped at parse time;
            # `#!` doc comments ride each declaration's `docs` field
            # and are attached by the per-declaration walkers.
            continue
        result = _walk_statement(tree, child)
        if isinstance(result, list):
            statements.extend(result)
        else:
            statements.append(result)
    return Module(statements=tuple(statements))


def parse_file(path: str | Path) -> Module:
    """Parse a `.qvr` file at `path`.

    Parameters
    ----------
    path : str | Path
        The file to read.

    Returns
    -------
    Module
        The file's statements in source order.

    Raises
    ------
    OSError
        If the file cannot be read.
    ParseError
        If the file does not parse.
    """
    p = Path(path)
    return parse(p.read_bytes(), str(p))


# ---------------------------------------------------------------------------
# whole-tree syntax validation
# ---------------------------------------------------------------------------


def _reject_malformed(tree: _Tree, file_path: str) -> None:
    """Raise `ParseError` if the tree contains any syntax damage.

    tree-sitter recovers from syntax errors by inserting ``ERROR``
    nodes (for unparseable spans) and zero-width MISSING tokens (for
    required named tokens). panproto surfaces the former as vertices
    of kind ``ERROR`` and the latter as zero-width vertices, wherever
    they sit in the tree. Both mean the source does not conform to the
    grammar, so the walk is refused outright rather than letting a
    corrupt fragment produce a silently wrong AST.

    A missing anonymous token (a dropped ``}`` or ``)``) does not
    surface at panproto's schema level: anonymous tokens never emit
    vertices, so a zero-width one leaves no trace. Those recoveries
    pass this check and parse as if the closer were present.

    When several damaged vertices exist, the innermost (smallest-span)
    one is reported, preferring leaves, so the error points at the
    most specific offending token.
    """
    findings: list[tuple[int, int, int, str, bool]] = []
    for vid in tree.vertices:
        kind = tree.kind(vid)
        consts = tree.consts(vid)
        sb_raw = consts.get("start-byte")
        eb_raw = consts.get("end-byte")
        if sb_raw is None or eb_raw is None:
            continue
        sb, eb = int(sb_raw), int(eb_raw)
        if kind == "ERROR":
            findings.append((eb - sb, sb, _leaf_rank(tree, vid), vid, False))
        elif sb == eb and kind != "source_file":
            findings.append((0, sb, _leaf_rank(tree, vid), vid, True))
    if not findings:
        return
    findings.sort(key=lambda f: (f[0], f[1], f[2]))
    _, _, _, vid, is_missing = findings[0]
    raise ParseError(_syntax_error_message(tree, vid, is_missing, file_path))


def _reject_reserved_binders(tree: _Tree, file_path: str) -> None:
    """Raise `ParseError` at a binder spelled as a reserved let word.

    The grammar admits any identifier in a binder position, while a let
    expression reads ``unit``, ``true``, ``false``, ``not``, and
    ``factor`` as a literal or a keyword. A binder with one of those
    names would be bound but never readable, and the mismatch would
    surface later as a type error about the literal far from its cause,
    so the binder itself is refused, at its own line and column. Every
    route through the compiler, checked lowering, and transpilation
    parses first, so all of them refuse it alike.

    Parameters
    ----------
    tree : _Tree
        The parsed tree.
    file_path : str
        The path recorded on the diagnostic.

    Raises
    ------
    ParseError
        At the first such binder in source order.
    """
    found: list[tuple[int, str, str]] = []
    for vid in tree.vertices:
        binder = _BINDER_FIELDS.get(tree.kind(vid))
        if binder is None:
            continue
        field_name, role = binder
        for target in tree.fields(vid, field_name):
            names = (
                tree.positional(target)
                if tree.kind(target) == "var_tuple"
                else [target]
            )
            for name_vid in names:
                if tree.kind(name_vid) != "identifier":
                    continue
                word = tree.text(name_vid)
                if word in RESERVED_LET_WORDS:
                    start = int(tree.consts(name_vid).get("start-byte", "0"))
                    found.append((start, word, role))
    # A lambda parameter spelled as a reserved word does not parse as a
    # lambda, so it is found in the source text rather than in the tree.
    code = _LINE_COMMENT.sub(lambda match: b" " * len(match.group()), tree.source)
    for match in _RESERVED_LAMBDA.finditer(code):
        found.append((match.start(1), match.group(1).decode(), "lambda parameter"))
    if not found:
        return
    start, word, role = min(found)
    line, col = tree.line_col_at(start)
    raise ParseError(
        f"{file_path}: reserved word at line {line}, col {col}: {word!r} names "
        f"{RESERVED_LET_WORDS[word]} in expressions, so it cannot name a {role}; "
        "choose another name"
    )


def _leaf_rank(tree: _Tree, vid: str) -> int:
    """0 for a leaf vertex, 1 otherwise (leaves report more precisely)."""
    return 0 if not tree.children.get(vid) else 1


def _syntax_error_message(
    tree: _Tree, vid: str, is_missing: bool, file_path: str
) -> str:
    consts = tree.consts(vid)
    start = int(consts["start-byte"])
    end = int(consts["end-byte"])
    src = tree.source
    # Skip the span's leading whitespace so the reported position and
    # snippet land on the first offending token, not on the newline or
    # indentation that tree-sitter folded into the ERROR span.
    pos = start
    while pos < end and src[pos] in b" \t\r\n":
        pos += 1
    line, col = tree.line_col_at(pos)
    if is_missing:
        detail = f"missing {tree.kind(vid)} in {_source_line(src, pos)!r}"
    else:
        span_lines = src[pos:end].decode("utf-8", errors="replace").splitlines()
        snippet = span_lines[0].strip() if span_lines else ""
        if len(snippet) > _SNIPPET_MAX:
            snippet = snippet[: _SNIPPET_MAX - 3] + "..."
        detail = repr(snippet)
    message = f"{file_path}: syntax error at line {line}, col {col}: {detail}"
    if _PROGRAM_LINE.search(src) and not _RETURN_WORD.search(src):
        message += "; every program body must end with a return step"
    return message


def _source_line(source: bytes, pos: int) -> str:
    """The full source line containing byte offset `pos`, stripped."""
    start = source.rfind(b"\n", 0, pos) + 1
    end = source.find(b"\n", pos)
    if end < 0:
        end = len(source)
    return source[start:end].decode("utf-8", errors="replace").strip()


__all__ = ["RESERVED_LET_WORDS", "parse", "parse_file"]
