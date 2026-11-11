"""LSP server tests.

Exercise the QVR language server in-process: build the
:class:`LanguageServer`, call the underlying handler functions
directly with crafted ``lsprotocol`` parameters, and assert on the
structured responses. This avoids the round-trip latency of a real
stdio subprocess and the asyncio-loop teardown issues that come with
multiple pygls clients in one pytest run.

Most LSP feature handlers live as nested functions inside
:func:`quivers.lsp.server.build_server`; we recover them through the
server's command/feature registry.
"""

from __future__ import annotations


from lsprotocol import types as lsp

from quivers.cli.repl_highlight import tokenize
from quivers.lsp import build_server
from quivers.lsp.document import DocumentState
from quivers.lsp.server import (
    _env_kinds_for,
    _format_document,
    _lsp_eof_position,
    _render_hover,
    _slice_source,
    _to_lsp_diag,
)


SAMPLE = """\
#! A small demo module.
object Alpha : FinSet 3
object Beta : FinSet 4
morphism f : Alpha -> Beta [role=latent]
"""


def _doc(source: str = SAMPLE) -> DocumentState:
    doc = DocumentState(uri="file:///tmp/sample.qvr")
    doc.update(source=source, version=1)
    return doc


def test_document_update_populates_module_and_env() -> None:
    doc = _doc()
    names: set[str] = set()
    for s in doc.module.statements:
        names.update(getattr(s, "names", ()))
    assert {"Alpha", "Beta", "f"} <= names
    assert "Alpha" in doc.env
    assert "f" in doc.env


def test_document_update_records_parse_error() -> None:
    doc = _doc(source="@@@@ bogus")
    assert doc.diagnostics
    assert doc.diagnostics[0].severity == "error"


def test_env_kinds_for_classifies_bindings() -> None:
    doc = _doc()
    kinds = _env_kinds_for(doc)
    assert kinds["Alpha"] == "type"
    assert kinds["Beta"] == "type"
    assert kinds["f"] == "function"


def test_env_kinds_empty_when_compiler_absent() -> None:
    doc = DocumentState(uri="file:///tmp/blank.qvr")
    assert _env_kinds_for(doc) == {}


def test_render_hover_stacks_qvr_and_ast() -> None:
    doc = _doc()
    decl = doc.find_decl("f")
    assert decl is not None
    hover = _render_hover(doc, "f")
    assert hover is not None
    # QVR section: bold header, fenced qvr block, verbatim source.
    assert "**QVR source**" in hover
    assert "```qvr" in hover
    assert "morphism f : Alpha -> Beta" in hover
    # Divider between the two panes.
    assert "\n---\n" in hover or hover.split("**AST")[0].rstrip().endswith("---")
    # AST section: bold header, collapsed fenced python block.
    assert "**AST (didactic)**" in hover
    assert "<details>" in hover
    assert "```python" in hover
    assert "MorphismDecl" in hover


def test_render_hover_unknown_name_returns_none() -> None:
    doc = _doc()
    assert _render_hover(doc, "nope") is None


def test_slice_source_returns_original_lines() -> None:
    doc = _doc()
    decl = doc.find_decl("Alpha")
    assert decl is not None
    sliced = _slice_source(doc, decl)
    assert sliced is not None
    assert "object Alpha : FinSet 3" in sliced


def test_to_lsp_diag_maps_severity_and_range() -> None:
    from quivers.cli.repl_session import Diagnostic

    diag = Diagnostic(message="oops", severity="error", line=4, col=2, code="compile")
    out = _to_lsp_diag(diag, _doc())
    assert out.range.start.line == 3  # 1-indexed -> 0-indexed
    assert out.range.start.character == 2
    assert out.severity == lsp.DiagnosticSeverity.Error
    assert out.source == "qvr-lsp"


def test_to_lsp_diag_warning_severity() -> None:
    from quivers.cli.repl_session import Diagnostic

    out = _to_lsp_diag(
        Diagnostic(message="m", severity="warning", line=1, col=0),
        _doc(),
    )
    assert out.severity == lsp.DiagnosticSeverity.Warning


def test_formatting_refuses_qiec_lowering_errors() -> None:
    doc = _doc(source=("effect E\n    op : Unit -> Unit\n    op : Unit -> Unit\n"))
    assert [diagnostic.code for diagnostic in doc.diagnostics] == ["qiec-handler"]
    assert _format_document(doc) is None


def test_formatting_range_ends_at_valid_trailing_newline_position() -> None:
    doc = _doc(source="index Nat=Z\n")
    edits = _format_document(doc)
    assert edits is not None and len(edits) == 1
    assert edits[0].range.end == lsp.Position(line=1, character=0)
    assert edits[0].new_text == "index Nat = Z\n"
    assert _lsp_eof_position("#! café 😀") == (0, 10)


def test_build_server_advertises_features() -> None:
    server = build_server()
    methods = set(server.protocol.fm.features.keys())
    expected = {
        "textDocument/didOpen",
        "textDocument/didChange",
        "textDocument/didSave",
        "textDocument/didClose",
        "textDocument/hover",
        "textDocument/definition",
        "textDocument/references",
        "textDocument/documentSymbol",
        "textDocument/completion",
        "textDocument/formatting",
        "textDocument/semanticTokens/full",
    }
    assert expected <= methods
    completion_options = server.protocol.fm.feature_options["textDocument/completion"]
    assert "." in completion_options.trigger_characters


def test_document_find_decl() -> None:
    doc = _doc()
    decl = doc.find_decl("f")
    assert decl is not None
    assert getattr(decl, "names", None) == ("f",)
    assert doc.find_decl("missing") is None


def test_document_name_at_position() -> None:
    doc = _doc()
    # Source layout (0-indexed lines):
    #   0: '#! A small demo module.'
    #   1: 'object Alpha : FinSet 3'
    #   2: 'object Beta : FinSet 4'
    #   3: 'morphism f : Alpha -> Beta [role=latent]'
    assert doc.name_at_position(1, 9) == "Alpha"
    assert doc.name_at_position(3, 9) == "f"
    assert doc.name_at_position(3, 14) == "Alpha"


def test_document_name_at_position_out_of_range() -> None:
    doc = _doc()
    assert doc.name_at_position(999, 0) is None


def test_pretty_ast_indents_one_field_per_line() -> None:
    from quivers.lsp.server import _pretty_ast

    doc = _doc()
    decl = doc.find_decl("f")
    assert decl is not None
    out = _pretty_ast(decl)
    # Multi-line shape (vs. single-line repr).
    assert "\n" in out
    # Every leading field starts on its own indented line; the
    # names tuple expands across lines.
    assert "    names=(" in out
    assert "        'f'," in out
    assert "    domain=" in out
    assert "    codomain=" in out
    # Empty `docs=()` and the AST discriminator field are stripped to
    # cut noise. (MorphismDecl has its own `morphism_kind` field,
    # which is legitimate -- we only strip the synthetic `kind=`
    # tagged-union discriminator.)
    assert "docs=" not in out
    assert "kind='morphism_decl'" not in out


def test_pretty_ast_handles_nested_tuple() -> None:
    from quivers.lsp.server import _pretty_ast

    src = (
        "object A : Real 2\n"
        "object B : Real 3\n"
        "morphism k : A * B -> B [role=kernel] ~ Normal()\n"
    )
    doc = DocumentState(uri="file:///nested.qvr")
    doc.update(source=src, version=1)
    decl = doc.find_decl("k")
    out = _pretty_ast(decl)
    # Components tuple expands across lines.
    assert "components=(" in out
    assert "TypeName(" in out


# ---------------------------------------------------------------------------
# Calls, handler clauses, rename, and retargeting
# ---------------------------------------------------------------------------


CALLS = """\
object Obs : FinSet 4

effect Echo
    ping : Int -> Int

instance echo : Echo
instance random : Random
instance score : Score

define twice(x : Int) : Int !{} =
    return x + x

handler doubling for Echo : Int -> Int [coverage=total, implementation=authored]
    ping(n : Int) resumes 1 =>
        let d <- twice(n)
        resume(d)
    return v =>
        return v

define identity[A : Type](value : A) : A !{} =
    return value

define noisy(x : Real) : Real !{random, score} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    perform score.add(weight(-0.25 * y * y))
    return y

define count(n : Int) : Int !{} =
    if n == 0 then
        return 0
    else
        let rest <- count(n - 1)
        let v <- identity[Int](rest)
        return v + 1

program prog : Obs -> Obs
    sample a <- Normal(loc=0.0, scale=1.0)
    let c <- noisy(a)
    observe y : Obs <- Normal(loc=c, scale=0.5)
    return c
export prog
"""


def _find(source: str, text: str, occurrence: int = 0) -> tuple[int, int]:
    """The 0-based line and column of one occurrence of ``text``.

    Parameters
    ----------
    source : str
        The document text.
    text : str
        The text to find.
    occurrence : int
        Which occurrence, counting from zero.

    Returns
    -------
    tuple[int, int]
        The position.
    """
    seen = 0
    for line, content in enumerate(source.splitlines()):
        start = 0
        while (col := content.find(text, start)) >= 0:
            if seen == occurrence:
                return line, col
            seen += 1
            start = col + 1
    raise AssertionError(f"{text!r} occurrence {occurrence} not in source")


def test_hover_at_a_call_renders_the_instantiated_signature_and_row() -> None:
    doc = _doc(source=CALLS)
    assert doc.diagnostics == []
    line, col = _find(CALLS, "identity[Int](rest)")
    hover = _render_hover(doc, "identity", line=line, col=col + 2)
    assert hover is not None
    assert "**Call**" in hover
    assert "identity[Int](value : Int) : Int !{}" in hover
    assert "**Inferred row**" in hover and "!{}" in hover
    line, col = _find(CALLS, "noisy(a)")
    hover = _render_hover(doc, "noisy", line=line, col=col + 1)
    assert hover is not None
    assert "noisy(x : Real) : Real !{random, score}" in hover
    assert "!{random : Random, score : Score}" in hover
    line, col = _find(CALLS, "count(n - 1)")
    hover = _render_hover(doc, "count", line=line, col=col + 1)
    assert hover is not None and "count(n : Int) : Int !{}" in hover


def test_hover_at_a_request_renders_the_instance_and_interface_operation() -> None:
    doc = _doc(source=CALLS)
    line, col = _find(CALLS, "random.sample")
    hover = _render_hover(doc, "sample", line=line, col=col + 8)
    assert hover is not None
    assert hover.startswith("**Operation**")
    assert "random.sample[a : Type] : Site[a] * Sampleable[a] -> a" in hover
    line, col = _find(CALLS, "score.add")
    hover = _render_hover(doc, "add", line=line, col=col + 7)
    assert hover is not None and "score.add : LogWeight -> Unit" in hover


def test_document_symbols_nest_calls_locals_and_handler_clauses() -> None:
    from quivers.lsp.server import _statement_symbols

    doc = _doc(source=CALLS)
    symbols = {
        symbol.name: symbol for symbol in _statement_symbols(doc, doc.module.statements)
    }
    handler = symbols["doubling"]
    assert [child.name for child in handler.children or []] == ["ping", "return"]
    ping = (handler.children or [])[0]
    assert [(child.name, child.detail) for child in ping.children or []] == [
        ("n", "parameter"),
        ("d", "local"),
        ("twice", "call"),
    ]
    count = symbols["count"]
    details = [(child.name, child.detail) for child in count.children or []]
    assert ("count", "recursive call") in details
    assert ("identity", "call") in details
    assert ("rest", "local") in details
    program = symbols["prog"]
    assert [(child.name, child.detail) for child in program.children or []] == [
        ("a", "sample"),
        ("c", "call noisy"),
        ("y", "observe"),
    ]


def test_rename_edits_every_reference_of_a_call_target() -> None:
    from quivers.lsp.server import _find_references, _renameable_declaration

    doc = _doc(source=CALLS)
    line, col = _find(CALLS, "define twice")
    position = lsp.Position(line=line, character=col + 8)
    assert _renameable_declaration(doc, "twice", position) is not None
    references = list(_find_references(doc, "twice", line=line, col=col + 8))
    assert [(r.range.start.line, r.range.start.character) for r in references] == [
        (line, col + 7),
        _find(CALLS, "twice(n)"),
    ]
    assert (
        _renameable_declaration(doc, "Random", lsp.Position(line=6, character=19))
        is None
    )
    server = build_server()
    methods = set(server.protocol.fm.features.keys())
    assert {"textDocument/rename", "textDocument/prepareRename"} <= methods


def test_retarget_recomputes_capabilities_without_reparsing() -> None:
    doc = _doc(source=CALLS)
    module = doc.module
    checked = doc.qiec_module
    doc.retarget("bugs")
    assert doc.module is module and doc.qiec_module is checked
    codes = {diagnostic.code for diagnostic in doc.diagnostics}
    assert codes and all(code.startswith("qiec:capability") for code in codes)
    doc.retarget(None)
    assert doc.diagnostics == []
    assert doc.registry() is doc.registry()


SCOPED_BINDERS = """\
effect Pair
    left : Int -> Int
    right : Int -> Int

instance pair : Pair

handler swap for Pair : Int -> Int [coverage=total, implementation=authored]
    left(n : Int) resumes 1 =>
        resume(n)
    right(n : Int) resumes 1 =>
        resume(n)

define count(n : Int) : Int !{} =
    if n == 0 then
        return 0
    else
        let rest <- count(n - 1)
        return rest + 1
"""


def test_references_and_definition_respect_clause_scopes() -> None:
    from quivers.lsp.server import _find_references

    doc = _doc(source=SCOPED_BINDERS)
    assert doc.diagnostics == []
    left_line, left_col = _find(SCOPED_BINDERS, "left(n : Int)")
    references = list(_find_references(doc, "n", line=left_line, col=left_col + 5))
    lines = sorted({r.range.start.line for r in references})
    assert lines == [left_line, left_line + 1], "a clause's binder stays in its clause"
    right_line, right_col = _find(SCOPED_BINDERS, "right(n : Int)")
    references = list(_find_references(doc, "n", line=right_line, col=right_col + 6))
    assert sorted({r.range.start.line for r in references}) == [
        right_line,
        right_line + 1,
    ]
    define_line, define_col = _find(SCOPED_BINDERS, "define count")
    references = list(_find_references(doc, "n", line=define_line, col=define_col + 13))
    assert all(r.range.start.line >= define_line for r in references)
    assert len(references) == 3
    call_line, call_col = _find(SCOPED_BINDERS, "count(n - 1)")
    declaration = doc.find_decl("count", line=call_line, col=call_col + 1)
    assert getattr(declaration, "line", 0) == define_line + 1


SCAN_OPTIONS = """\
object Val : Real 1
object Pair : Real 2

program cell(u_t, h_prev) : Val * Val -> Val
    let m = 0.8 * h_prev + u_t
    sample f <- Normal(loc=m, scale=0.5)
    return f

program back(u_t, h_prev) : Val * Val -> Val
    let n = 0.6 * h_prev - u_t
    sample b <- Normal(loc=n, scale=0.5)
    return b

program tag(p_t, s_prev) : Pair * Val -> Val
    let k = p_t[0] - p_t[1] + 0.3 * s_prev
    sample s <- Normal(loc=k, scale=0.5)
    observe r <- Normal(loc=s, scale=0.5)
    return s

define both = fan(scan(cell, outputs=all), scan(back, outputs=all, reverse=true))
define tagger = both >> scan(tag, outputs=all)

program prog : Val -> Val
    sample tags <- tagger
    return tags

export prog
"""


def test_scan_options_check_highlight_and_format() -> None:
    doc = _doc(source=SCAN_OPTIONS)
    assert doc.diagnostics == []
    pairs = {
        (span.token, span.text)
        for span in tokenize("define s = scan(c, outputs=all, reverse=true, init=h0)")
    }
    assert {
        ("keyword", "outputs"),
        ("keyword", "reverse"),
        ("keyword", "init"),
    } <= pairs
    # The values `all` and `last` are names, not reserved words.
    assert ("variable", "all") in pairs
    edits = _format_document(doc)
    assert edits is not None
    formatted = edits[0].new_text
    assert "scan(back, outputs=all, reverse=true)" in formatted
    assert "scan(tag, outputs=all)" in formatted


def test_a_malformed_scan_option_is_a_located_diagnostic() -> None:
    doc = _doc(source="define s = scan(c, outputs=every)\n")
    (diagnostic,) = doc.diagnostics
    assert diagnostic.severity == "error"
    assert "outputs are 'last'" in diagnostic.message
    assert (diagnostic.line, diagnostic.col) == (1, 27)
