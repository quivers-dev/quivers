"""Live syntax highlighting for QVR source, shared across all frontends.

The pipeline:

1. Re-parse the buffer via panproto's cached `ParseEmitLens`
   (sub-millisecond on edits after warm-up).
2. Walk the resulting schema's leaf vertices in source order.
3. Map each ``(vertex.kind, parent.kind, literal_text)`` triple to a
   semantic token type, via a single `STYLE_TABLE`.
4. Emit a list of `Span` records carrying ``(byte_start,
   byte_end, token, text)``.

Frontends consume `Span` lists in two flavours:

- Textual / Rich pull a ``rich.text.Text`` via `to_rich_text`.
- prompt_toolkit consumes ``(Pygments token, text)`` pairs via
  `to_pygments_pairs`.
- The LSP encodes spans into the SemanticTokens delta stream via
  `to_semantic_token_legend` and `to_semantic_token_data`.

The same `STYLE_TABLE` drives Rich styling and the LSP legend so
themes never drift across surfaces.
"""

from __future__ import annotations

from typing import Any

import didactic.api as dx
import tree_sitter
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
)
from pygments.token import Text as PygmentsText

from quivers.dsl.pygments_lexer import (
    _BUILTIN_FUNCTION_TOKENS,
    _BUILTIN_TYPE_TOKENS,
    _KEYWORD_TOKENS,
    _OPERATOR_TOKENS,
    _load_parser,
    _node_kind_to_pygments_token,
    _TokenType,
)

# ---------------------------------------------------------------------------
# semantic token vocabulary
# ---------------------------------------------------------------------------


SEMANTIC_TOKEN_TYPES: tuple[str, ...] = (
    "comment",
    "keyword",
    "operator",
    "punctuation",
    "string",
    "number",
    "variable",
    "type",
    "function",
    "namespace",
    "decorator",
    "error",
)
"""The semantic token types every QVR token reduces to. The order is
significant: a token's index here is its type in the LSP semantic-token
legend."""


SEMANTIC_TOKEN_MODIFIERS: tuple[str, ...] = (
    "declaration",
    "definition",
    "deprecated",
)
"""The semantic token modifiers the language server advertises in its
legend."""


STYLE_TABLE: dict[str, str] = {
    # Explicit truecolor hex codes (One-Dark inspired) instead of
    # palette-indexed names. Terminals with customised 16-colour
    # palettes were collapsing "magenta" / "blue" / "cyan" / "default"
    # into the same accent hue (most often red-orange), which made
    # every token look the same colour. Truecolor sidesteps the
    # palette entirely.
    "comment": "italic #5c6370",
    "keyword": "bold #c678dd",
    "operator": "#56b6c2",
    "punctuation": "#7f848e",
    "string": "#98c379",
    "number": "#d19a66",
    "variable": "#abb2bf",
    "type": "bold #61afef",
    "function": "bold #e5c07b",
    "namespace": "bold #e06c75",
    "decorator": "#c678dd",
    "error": "bold #e06c75 underline",
}
"""Rich style strings keyed by semantic token type, shared by the TUI,
the prompt_toolkit lexer, and the Jupyter kernel so that every front end
colours a token alike."""


_PUNCT = {"(", ")", "[", "]", "{", "}", ",", ":", "."}


class Span(dx.Model):
    """One styled run of source.

    Parameters
    ----------
    start : int
        The run's start byte offset in the source.
    end : int
        The run's end byte offset, exclusive.
    token : str
        Its semantic token type, one of `SEMANTIC_TOKEN_TYPES`.
    text : str
        The source text of the run.
    line : int
        Its 0-indexed start line.
    col : int
        Its 0-indexed start column, in UTF-16 code units.
    end_line : int
        Its 0-indexed end line.
    end_col : int
        Its 0-indexed end column, in UTF-16 code units.
    """

    start: int
    end: int
    token: str
    text: str
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------


def tokenize(
    source: str | bytes,
    *,
    env_kinds: dict[str, str] | None = None,
) -> list[Span]:
    """Return the styled span list for ``source``.

    Parameters
    ----------
    source : str or bytes
        The QVR source to highlight.
    env_kinds : dict[str, str] or None
        Semantic kinds of names bound in the environment, as described
        below; ``None`` highlights from the grammar alone.

    Returns
    -------
    list[Span]
        The source's spans in order, each tagged with a semantic token
        type.

    Drives directly on the tree-sitter parse so that anonymous tokens
    (keywords, punctuation) are preserved; panproto's Schema view
    strips those, so we go to tree-sitter for highlighting and keep
    panproto for parsing into the didactic AST.

    ``env_kinds`` enables semantic highlighting: a mapping from name
    to one of ``"type"``, ``"function"``, ``"namespace"`` (or any
    SEMANTIC_TOKEN_TYPES value). Identifiers the grammar leaves as
    ``"variable"`` are upgraded to their env-known kind, so a name
    looks the same across the input pane, ``:type``, ``:info``, and
    any other surface, regardless of whether the surrounding context
    parses as a valid declaration.

    Failure path: if the parser raises, the entire source is returned
    as a single ``error`` span. Callers can downgrade to a no-highlight
    render but the buffer still displays.
    """
    if isinstance(source, str):
        src_bytes = source.encode("utf-8")
    else:
        src_bytes = source
    if not src_bytes:
        return []
    try:
        parser, _, _ = _load_parser()
        tree = parser.parse(src_bytes)
    except Exception:
        return [
            Span(
                start=0,
                end=len(src_bytes),
                token="error",
                text=src_bytes.decode("utf-8", errors="replace"),
                line=0,
                col=0,
                end_line=src_bytes.count(b"\n"),
                end_col=0,
            )
        ]

    leaves: list[tuple[tree_sitter.Node, str | None, str | None]] = []

    def walk(
        node: tree_sitter.Node,
        parent_kind: str | None,
        field_name: str | None,
    ) -> None:
        """Collect leaf nodes with their enclosing grammar context.

        Parameters
        ----------
        node : tree_sitter.Node
            The current syntax node.
        parent_kind : str or None
            Its enclosing node's grammar kind.
        field_name : str or None
            Its field in the enclosing node.
        """
        if not node.children:
            leaves.append((node, parent_kind, field_name))
            return
        for index, child in enumerate(node.children):
            walk(child, node.type, node.field_name_for_child(index))

    walk(tree.root_node, None, None)

    spans: list[Span] = []
    cursor = 0
    for leaf, parent_kind, field_name in leaves:
        sb = leaf.start_byte
        eb = leaf.end_byte
        if sb > cursor:
            gap = src_bytes[cursor:sb].decode("utf-8", errors="replace")
            spans.append(
                _position(
                    Span(start=cursor, end=sb, token="variable", text=gap),
                    src_bytes,
                )
            )
        text = src_bytes[sb:eb].decode("utf-8", errors="replace")
        token = _classify(leaf.type, text, parent_kind, field_name)
        # Semantic upgrade: if the grammar produced a generic
        # "variable" classification but the env knows this name as a
        # type/function/namespace, paint it the env colour. This is
        # the seam that gives consistent highlighting across all REPL
        # output surfaces.
        if env_kinds and token == "variable":
            env_kind = env_kinds.get(text)
            if env_kind:
                token = env_kind
        spans.append(
            _position(Span(start=sb, end=eb, token=token, text=text), src_bytes)
        )
        cursor = eb
    if cursor < len(src_bytes):
        tail = src_bytes[cursor:].decode("utf-8", errors="replace")
        spans.append(
            _position(
                Span(
                    start=cursor,
                    end=len(src_bytes),
                    token="variable",
                    text=tail,
                ),
                src_bytes,
            )
        )
    return _merge_split_operators(spans, src_bytes)


def _merge_split_operators(spans: list[Span], source: bytes) -> list[Span]:
    """Rejoin a multi-character operator the recovering parser split.

    A fragment such as one program line parses with error recovery, and
    in a recovered state the lexer may read ``<-`` as ``<`` followed by
    ``-``. Two adjacent operator spans whose concatenation is itself a
    grammar operator are one token to the reader, so they are painted as
    one.

    Parameters
    ----------
    spans : list[Span]
        The classified spans, in source order.
    source : bytes
        The source the spans index into.

    Returns
    -------
    list[Span]
        The spans with split operators merged.
    """
    merged: list[Span] = []
    for span in spans:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.token == "operator"
            and span.token == "operator"
            and previous.end == span.start
            and previous.text + span.text in _OPERATOR_TOKENS
        ):
            merged[-1] = _position(
                Span(
                    start=previous.start,
                    end=span.end,
                    token="operator",
                    text=previous.text + span.text,
                ),
                source,
            )
            continue
        merged.append(span)
    return merged


#: The highlight class of each Pygments token family the lexer emits,
#: most specific first: the first family a token type descends from wins.
_TOKEN_CLASSES: tuple[tuple[_TokenType, str], ...] = (
    (Comment, "comment"),
    (Name.Decorator, "decorator"),
    (Number, "number"),
    (String.Symbol, "namespace"),
    (String, "string"),
    (Keyword, "keyword"),
    (Punctuation, "punctuation"),
    (Operator, "operator"),
    (Name.Class, "type"),
    (Name.Function, "function"),
    (Name.Builtin, "function"),
    (Name.Constant, "keyword"),
)


def _classify(
    kind: str,
    text: str,
    parent_kind: str | None,
    field_name: str | None = None,
) -> str:
    """The highlight class of one leaf, by the lexer's own classification.

    Parameters
    ----------
    kind : str
        The leaf's node kind.
    text : str
        The leaf's text.
    parent_kind : str | None
        The kind of the node the leaf sits in.
    field_name : str | None
        The field the leaf fills in its parent.

    Returns
    -------
    str
        One of `SEMANTIC_TOKEN_TYPES`: the Pygments token the lexer
        assigns, mapped to its highlight class, so every surface colours
        a construct the way the Pygments lexer does. Punctuation the
        lexer leaves as plain text is classed by the punctuation set.
    """
    if kind in _PUNCT:
        return "punctuation"
    if kind == "identifier" and parent_kind in {
        "pragma_entry",
        "pragma_outer",
        "pragma_inner",
    }:
        return "decorator"
    token = _node_kind_to_pygments_token(kind, text, parent_kind, field_name)
    if kind == "identifier" and token is Name.Variable:
        # When a tree-sitter parse error puts a known keyword in the
        # identifier bucket, the literal text still tells us what the
        # user wrote. Context wins first because QIEC permits
        # declarations such as ``index Nat`` whose name also has a
        # built-in spelling.
        if text in _KEYWORD_TOKENS:
            return "keyword"
        if text in _BUILTIN_FUNCTION_TOKENS:
            return "function"
        if text in _BUILTIN_TYPE_TOKENS:
            return "type"
    for family, name in _TOKEN_CLASSES:
        if token in family:
            return name
    return "variable"


def _position(span: Span, source: bytes) -> Span:
    """Attach line/col coordinates to a span."""
    sl, sc = _byte_to_line_col(source, span.start)
    el, ec = _byte_to_line_col(source, span.end)
    return Span(
        start=span.start,
        end=span.end,
        token=span.token,
        text=span.text,
        line=sl,
        col=sc,
        end_line=el,
        end_col=ec,
    )


def _byte_to_line_col(source: bytes, byte_offset: int) -> tuple[int, int]:
    if byte_offset > len(source):
        byte_offset = len(source)
    prefix = source[:byte_offset]
    line = prefix.count(b"\n")
    last_nl = prefix.rfind(b"\n")
    line_start = 0 if last_nl < 0 else last_nl + 1
    line_prefix = source[line_start:byte_offset].decode("utf-8")
    col = _utf16_length(line_prefix)
    return line, col


def _utf16_length(text: str) -> int:
    """Return the number of UTF-16 code units required by ``text``."""
    return len(text.encode("utf-16-le")) // 2


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------


def to_rich_text(
    source: str,
    *,
    env_kinds: dict[str, str] | None = None,
    link_action: str | None = None,
) -> Any:
    """Build a `rich.text.Text` from the highlighted source.

    Rich is imported lazily so that importing this module does
    not pull rich in for callers that need only raw spans.

    If ``link_action`` is set (e.g. ``"info"``), identifiers that the
    grammar or env classified as something meaningful (type / function
    / namespace / variable when the name is in env_kinds) are wrapped
    as Rich console links pointing to ``"<link_action>:<name>"``. The
    TUI's RichLog ``on_click`` translator then runs ``:link_action
    <name>`` when the user clicks.

    Parameters
    ----------
    source : str
        The QVR source to highlight.
    env_kinds : dict[str, str] or None
        Semantic kinds of names bound in the environment, as for
        `tokenize`.
    link_action : str or None
        The meta-command clickable identifiers run, or ``None`` for no
        links.

    Returns
    -------
    rich.text.Text
        The styled text.
    """
    from rich.style import Style
    from rich.text import Text

    out = Text()
    interesting = {"type", "function", "namespace"}
    for span in tokenize(source, env_kinds=env_kinds):
        style_str = STYLE_TABLE.get(span.token, "")
        if (
            link_action
            and span.token in interesting
            and span.text.replace("_", "").isalnum()
            and not span.text[0].isdigit()
        ):
            style = Style.parse(style_str) + Style(
                meta={"@click": f"{link_action}('{span.text}')"}
            )
            out.append(span.text, style=style)
        else:
            out.append(span.text, style=style_str)
    return out


def to_pygments_pairs(source: str) -> list[tuple[_TokenType, str]]:
    """Build ``(Pygments token, text)`` pairs suitable for prompt_toolkit.

    prompt_toolkit's PygmentsLexer wraps a Pygments Lexer that yields
    ``(index, token, text)``; here we deliver the simpler ``(token,
    text)`` form directly used by ``prompt_toolkit.formatted_text``.

    Parameters
    ----------
    source : str
        The QVR source to highlight.

    Returns
    -------
    list[tuple[pygments.token._TokenType, str]]
        One pair per span, in source order.
    """
    mapping = {
        "comment": Comment,
        "keyword": Keyword,
        "operator": Operator,
        "punctuation": Punctuation,
        "string": String,
        "number": Number,
        "variable": Name.Variable,
        "type": Name.Class,
        "function": Name.Builtin,
        "namespace": String.Symbol,
        "decorator": Name.Decorator,
        "error": PygmentsText,
    }
    out: list[tuple[_TokenType, str]] = []
    for span in tokenize(source):
        out.append((mapping.get(span.token, PygmentsText), span.text))
    return out


# ---------------------------------------------------------------------------
# LSP semantic-tokens helpers
# ---------------------------------------------------------------------------


def to_semantic_token_legend() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the semantic-token legend the language server advertises.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        `SEMANTIC_TOKEN_TYPES` and `SEMANTIC_TOKEN_MODIFIERS`.
    """
    return SEMANTIC_TOKEN_TYPES, SEMANTIC_TOKEN_MODIFIERS


def to_semantic_token_data(
    source: str, *, env_kinds: dict[str, str] | None = None
) -> list[int]:
    """Encode ``source`` as the LSP SemanticTokens 5-tuple stream.

    The stream is delta-encoded as ``[deltaLine, deltaStart, length,
    tokenType, tokenModifiers]`` per LSP 3.17. See
    https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_semanticTokens
    for the encoding rules.

    Parameters
    ----------
    source : str
        The QVR source to encode.
    env_kinds : dict[str, str] or None
        Semantic kinds of names bound in the environment, as for
        `tokenize`.

    Returns
    -------
    list[int]
        Five integers per token.
    """
    type_index = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}
    out: list[int] = []
    prev_line = 0
    prev_col = 0
    for span in tokenize(source, env_kinds=env_kinds):
        if span.token in ("variable",) and not span.text.strip():
            # Pure whitespace at the start of the buffer; skip.
            continue
        if span.line != prev_line:
            prev_col = 0
        delta_line = span.line - prev_line
        delta_col = span.col - prev_col
        # Tokens that cross a newline aren't supported by the protocol;
        # split conservatively at the next newline.
        text = span.text
        length = _utf16_length(text)
        if "\n" in text:
            first_segment = text.split("\n", 1)[0]
            length = _utf16_length(first_segment)
        out.extend(
            [
                delta_line,
                delta_col,
                length,
                type_index.get(span.token, type_index["variable"]),
                0,
            ]
        )
        prev_line = span.line
        prev_col = span.col
    return out


__all__ = [
    "STYLE_TABLE",
    "SEMANTIC_TOKEN_MODIFIERS",
    "SEMANTIC_TOKEN_TYPES",
    "Span",
    "to_pygments_pairs",
    "to_rich_text",
    "to_semantic_token_data",
    "to_semantic_token_legend",
    "tokenize",
]
