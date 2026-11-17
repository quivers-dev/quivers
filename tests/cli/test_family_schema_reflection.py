"""One constructor catalogue drives REPL, editor, and notebook family help."""

from pathlib import Path

from quivers._family_schemas import family_parameterizations, render_parameterizations
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_session import ReplSession


def _signature():
    return "MatrixNormal " + render_parameterizations(
        family_parameterizations("MatrixNormal", ())
    )


def test_completion_and_repl_help_share_complete_constructor_alternatives():
    session = ReplSession()
    matches = [
        entry
        for entry in all_completions(session, "MatrixN")
        if entry.text == "MatrixNormal"
    ]
    assert len(matches) == 1
    assert matches[0].detail == _signature()
    reply = session.info("MatrixNormal")
    assert reply.ok
    assert reply.body == _signature()
    assert reply.body.count("row_scale_tril") == 2
    assert reply.body.count("col_scale_tril") == 2


def test_authored_bindings_shadow_builtin_schema_help():
    session = ReplSession()
    reply = session.dispatch("object MatrixNormal : FinSet 2")
    assert reply.ok
    assert "object MatrixNormal : FinSet 2" in session.info("MatrixNormal").body
    assert "row_scale_tril" not in session.info("MatrixNormal").body


def test_notebook_inspection_reuses_repl_constructor_help():
    from quivers.kernel.quivers_kernel import QuiversKernel

    kernel = QuiversKernel()
    reply = kernel.do_inspect("MatrixNormal", len("MatrixNormal"))
    assert reply["found"]
    assert reply["data"]["text/plain"] == _signature()
    assert "MatrixNormal" in kernel.do_complete("MatrixN", len("MatrixN"))["matches"]


def test_lsp_hover_and_completion_reuse_canonical_family_schemas(tmp_path: Path):
    from lsprotocol import types as lsp
    from quivers.lsp.server import build_server

    server = build_server()
    server.text_document_publish_diagnostics = lambda _params: None
    source = "object One : FinSet 1\n-- MatrixN\n"
    uri = (tmp_path / "factor-help.qvr").as_uri()
    server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN](
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri=uri, language_id="qvr", version=1, text=source
            )
        )
    )
    result = server.protocol.fm.features[lsp.TEXT_DOCUMENT_COMPLETION](
        lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=1, character=10),
        )
    )
    assert (
        next(item.detail for item in result.items if item.label == "MatrixNormal")
        == _signature()
    )
    hover_source = "object One : FinSet 1\n-- MatrixNormal\n"
    server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN](
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri=uri, language_id="qvr", version=2, text=hover_source
            )
        )
    )
    hover = server.protocol.fm.features[lsp.TEXT_DOCUMENT_HOVER](
        lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=1, character=8),
        )
    )
    assert _signature() in hover.contents.value
