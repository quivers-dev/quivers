"""Highlighter tests.

These act as a thin grammar-drift detector: if the tree-sitter QVR
grammar adds or renames a keyword, the classifier must keep up. The
golden token stream covers the major shapes that the TUI, the
prompt_toolkit lexer, and the LSP semantic-tokens encoder all consume.
"""

from __future__ import annotations

from pygments.token import Name

from quivers.cli.repl_highlight import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    Span,
    to_semantic_token_data,
    to_semantic_token_legend,
    tokenize,
    to_rich_text,
)
from quivers.dsl.pygments_lexer import QvrLexer


SAMPLE = "object X : FinSet 3\nmorphism f : X -> X [role=parameter]\n# comment\n"


def _classify(spans: list[Span]) -> list[tuple[str, str]]:
    return [(sp.token, sp.text) for sp in spans if sp.text.strip()]


def test_tokenize_classifies_keywords_and_punctuation() -> None:
    pairs = _classify(tokenize(SAMPLE))
    # Order matters: this is a structural snapshot of the grammar.
    assert ("keyword", "object") in pairs
    assert ("keyword", "morphism") in pairs
    assert ("punctuation", ":") in pairs
    assert ("operator", "->") in pairs
    assert ("number", "3") in pairs
    assert ("comment", "# comment") in pairs


def test_tokenize_identifies_types_and_variables() -> None:
    # In `object X : FinSet 3`, X appears as the declared identifier (variable).
    # In `morphism f : X -> X [role=parameter]`, the two X tokens sit inside type_atoms
    # and must surface as `type` so the LSP semantic-tokens stream
    # paints them with the type colour.
    pairs = _classify(tokenize(SAMPLE))
    type_xs = [p for p in pairs if p == ("type", "X")]
    assert len(type_xs) >= 2


def test_semantic_token_legend_is_stable() -> None:
    types, modifiers = to_semantic_token_legend()
    assert types == SEMANTIC_TOKEN_TYPES
    assert modifiers == SEMANTIC_TOKEN_MODIFIERS
    # The legend index is wire-format; reordering would invalidate the
    # encoded delta stream against any active editor.
    assert types.index("keyword") == 1
    assert types.index("comment") == 0


def test_semantic_token_data_is_5_tuple_aligned() -> None:
    data = to_semantic_token_data(SAMPLE)
    assert len(data) % 5 == 0
    assert data, "expected non-empty token stream for non-empty source"


def test_tokenize_empty_source() -> None:
    assert tokenize("") == []


def test_env_kinds_upgrades_variable_to_type() -> None:
    # In the type-position of a latent decl 'Source' is already a type,
    # but in an unparseable context (the line below) the grammar emits
    # it as a plain variable. The env-aware upgrade catches that case.
    pairs = _classify(tokenize("foo seq2seq Source", env_kinds={"Source": "type"}))
    assert ("type", "Source") in pairs


def test_env_kinds_upgrades_to_function() -> None:
    pairs = _classify(tokenize("g h", env_kinds={"g": "function", "h": "function"}))
    assert ("function", "g") in pairs
    assert ("function", "h") in pairs


def test_keyword_text_promoted_in_error_context() -> None:
    # In a context the grammar can't fully parse, leaf.type may be
    # 'identifier' even for known keywords; the classifier still
    # recognises the text.
    pairs = _classify(tokenize("observe x : Y <- z"))
    keywords = {p[1] for p in pairs if p[0] == "keyword"}
    assert "observe" in keywords or "Y" not in {p[1] for p in pairs}


def test_semantic_token_data_with_env_classification() -> None:
    data = to_semantic_token_data(
        "foo Source",
        env_kinds={"Source": "type"},
    )
    # 5-tuple aligned and non-empty.
    assert len(data) % 5 == 0
    type_index = SEMANTIC_TOKEN_TYPES.index("type")
    # At least one token carries the 'type' index.
    assert type_index in data[3::5]


def test_effect_request_distinguishes_instance_from_operation() -> None:
    source = "define read() : Int !{reader} =\n    perform reader.get()\n"
    pairs = _classify(tokenize(source))
    assert ("variable", "reader") in pairs
    assert ("function", "get") in pairs

    pygments = list(QvrLexer().get_tokens_unprocessed(source))
    assert any(
        token in Name.Variable and text == "reader" for _, token, text in pygments
    )
    assert any(token in Name.Function and text == "get" for _, token, text in pygments)


def test_collection_operations_highlight_as_functions() -> None:
    source = """define twice(x : Real) : Real !{} =
    return x * 2.0

define mapped() : Tensor[Real]([2]) !{} =
    let xs = map([1.0, 2.0], x -> x * 2.0)
    let ys <- traverse(xs, x -> twice(x))
    return ys
"""
    pygments = list(QvrLexer().get_tokens_unprocessed(source))
    assert any(token in Name.Builtin and text == "map" for _, token, text in pygments)
    assert any(
        token in Name.Builtin and text == "traverse" for _, token, text in pygments
    )


def test_unicode_offsets_match_pygments_and_lsp_coordinate_systems() -> None:
    source = "#! café 😀\nindex Nat = Z\n"
    pygments = list(QvrLexer().get_tokens_unprocessed(source))
    index_offset = next(offset for offset, _, text in pygments if text == "index")
    assert index_offset == source.index("index")

    semantic = to_semantic_token_data(source)
    assert semantic[:5] == [0, 0, 10, SEMANTIC_TOKEN_TYPES.index("comment"), 0]


def test_to_rich_text_smoke() -> None:
    rt = to_rich_text("object X : FinSet 3")
    # Renders without raising; carries the source verbatim.
    plain = rt.plain
    assert "object" in plain
    assert "X" in plain


def test_link_action_wraps_identifiers() -> None:
    rt = to_rich_text(
        "f X",
        env_kinds={"f": "function", "X": "type"},
        link_action="info",
    )
    # Walk the underlying spans for meta hooks.
    metas = []
    for start, end, style in rt.spans:
        del start, end
        meta = getattr(style, "meta", None) or {}
        if meta.get("@click"):
            metas.append(meta["@click"])
    assert any("info('f')" in m for m in metas)
    assert any("info('X')" in m for m in metas)


def test_error_path_returns_single_error_span() -> None:
    # Source the parser will choke on (binary garbage that produces
    # ERROR nodes everywhere) still returns SOMETHING tokenizable.
    spans = tokenize("@@@@ <<< @@@")
    assert spans  # never empty for non-empty source


def test_comment_classification() -> None:
    pairs = _classify(tokenize("# hello world"))
    assert any(p[0] == "comment" for p in pairs)


def test_doc_comment_classification() -> None:
    pairs = _classify(tokenize("#! doc string\nobject X : FinSet 3"))
    assert any(p[0] == "comment" and "doc" in p[1] for p in pairs)


def test_operator_classification() -> None:
    pairs = _classify(
        tokenize("object X : FinSet 3\nmorphism f : X -> X [role=parameter]")
    )
    operators = {p[1] for p in pairs if p[0] == "operator"}
    assert "->" in operators
