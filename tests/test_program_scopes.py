"""Every ``program`` declaration is a scope, with or without typed parameters.

A program's sites are addressed by ``program::site`` paths through the
scope model; a program without typed parameters, which also compiles to
a morphism, is the same kind of scope as a template. Each consumer of
the scope model is exercised here: the resolver, the REPL commands, the
completer, the plate-graph builder, and the language server's hover,
go-to-definition, and completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lsprotocol import types as lsp

from quivers.analysis.plate_graph import build_plate_graph
from quivers.analysis.scope import (
    find_all_references,
    resolve_scoped_path,
    scope_children,
    top_level_bindings,
)
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_session import ReplSession
from quivers.lsp import build_server

PLAIN = """\
object Row : FinSet 3
object Val : Real 1

program plain : Row -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return y

export plain
"""

TEMPLATE = PLAIN.replace(
    "program plain : Row", "program plain(s : Real) : Row"
).replace("scale=1.0)\n    observe", "scale=s)\n    observe")

SOURCES = pytest.mark.parametrize(
    "source", [PLAIN, TEMPLATE], ids=["without-parameters", "template"]
)


def _session(tmp_path: Path, source: str) -> ReplSession:
    path = tmp_path / "module.qvr"
    path.write_text(source)
    session = ReplSession()
    session.load_file(str(path))
    return session


@SOURCES
def test_a_program_resolves_as_a_program_scope(tmp_path: Path, source: str) -> None:
    compiler = _session(tmp_path, source)._compiler
    top = resolve_scoped_path(compiler, "plain")
    assert top is not None and top.kind == "program"
    children = scope_children(top)
    assert children["mu"].kind == "sample-site"
    assert children["y"].kind == "observe-site"
    assert children["return"].kind == "return-site"
    site = resolve_scoped_path(compiler, "plain::mu")
    assert site is not None and site.kind == "sample-site"
    assert site.parent_kind == "program"


@SOURCES
def test_each_top_level_name_is_bound_once(tmp_path: Path, source: str) -> None:
    compiler = _session(tmp_path, source)._compiler
    bindings = top_level_bindings(compiler)
    names = [ref.name for ref in bindings]
    assert len(names) == len(set(names))
    assert {ref.name: ref.kind for ref in bindings}["plain"] == "program"


def test_a_program_without_parameters_is_also_a_morphism(tmp_path: Path) -> None:
    compiler = _session(tmp_path, PLAIN)._compiler
    assert "plain" in compiler.programs
    assert "plain" in compiler.morphisms


@SOURCES
def test_where_finds_a_programs_sites(tmp_path: Path, source: str) -> None:
    session = _session(tmp_path, source)
    paths = [ref.path for ref in find_all_references(session._compiler, "mu")]
    assert paths == ["plain::mu"]
    response = session.dispatch(":where mu")
    assert not response.diagnostics
    assert "plain::mu" in response.body


@SOURCES
@pytest.mark.parametrize("command", [":type", ":info", ":browse"])
def test_repl_commands_resolve_program_site_paths(
    tmp_path: Path, source: str, command: str
) -> None:
    response = _session(tmp_path, source).dispatch(f"{command} plain::mu")
    assert not response.diagnostics, response.diagnostics
    assert response.body


@SOURCES
def test_browse_lists_a_program_once_with_its_body(tmp_path: Path, source: str) -> None:
    output = _session(tmp_path, source).dispatch(":browse").body
    assert output.count("plain") >= 1
    heads = [line for line in output.splitlines() if line.lstrip().startswith("plain")]
    assert len(heads) == 1, heads
    assert "sample mu" in output


@SOURCES
def test_completion_reaches_a_programs_sites(tmp_path: Path, source: str) -> None:
    session = _session(tmp_path, source)
    scoped = {c.text for c in all_completions(session, "plain::")}
    assert {"plain::mu", "plain::y", "plain::return"} <= scoped
    bare = [c for c in all_completions(session, "pla") if c.text == "plain"]
    assert [c.detail for c in bare] == ["program"]
    assert "plain::mu" in {c.text for c in all_completions(session, "m")}


@SOURCES
def test_plate_graph_reads_any_program(tmp_path: Path, source: str) -> None:
    graph = build_plate_graph(_session(tmp_path, source)._compiler, "plain")
    assert graph is not None
    assert {"mu", "y"} <= {node.name for node in graph.nodes}


def _lsp_source(source: str) -> str:
    return source + "# See plain::mu for the shared shift.\n"


def _open(source: str):  # type: ignore[no-untyped-def]
    server = build_server()
    server.text_document_publish_diagnostics = lambda _params: None  # type: ignore[method-assign]
    features = server.protocol.fm.features
    uri = "file:///tmp/scopes.qvr"
    features[lsp.TEXT_DOCUMENT_DID_OPEN](
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri=uri, language_id="qvr", version=1, text=source
            )
        ),
    )
    return server, features, uri


def _comment_position(source: str, needle: str) -> lsp.Position:
    lines = source.splitlines()
    line = next(i for i, text in enumerate(lines) if text.startswith("# "))
    return lsp.Position(line=line, character=lines[line].index(needle))


@SOURCES
def test_lsp_hover_on_a_program_site_path(source: str) -> None:
    text = _lsp_source(source)
    server, features, uri = _open(text)
    hover = features[lsp.TEXT_DOCUMENT_HOVER](
        lsp.HoverParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=_comment_position(text, "mu for"),
        ),
    )
    assert hover is not None
    assert "sample mu <- Normal" in hover.contents.value
    assert "sample-site `plain::mu`" in hover.contents.value


@SOURCES
def test_lsp_definition_of_a_program_site_path(source: str) -> None:
    text = _lsp_source(source)
    server, features, uri = _open(text)
    locations = features[lsp.TEXT_DOCUMENT_DEFINITION](
        lsp.DefinitionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=_comment_position(text, "mu for"),
        ),
    )
    assert locations is not None and len(locations) == 1
    start = locations[0].range.start
    lines = text.splitlines()
    assert lines[start.line].strip().startswith("sample mu")
    assert lines[start.line][start.character : start.character + 2] == "mu"


@SOURCES
def test_lsp_completion_of_a_program_site_path(source: str) -> None:
    text = _lsp_source(source)
    server, features, uri = _open(text)
    position = _comment_position(text, "mu for")
    completions = features[lsp.TEXT_DOCUMENT_COMPLETION](
        lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=position,
        ),
    )
    assert "plain::mu" in {item.label for item in completions.items}
