"""User-facing tooling over the QVR QIEC surface."""

from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest
import json

from pygments.token import Name

from quivers.cli import check_file
from quivers.cli.run import main as run_main
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_highlight import tokenize
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse
from quivers.dsl.pygments_lexer import QvrLexer
from quivers.dsl.qiec_tooling import qiec_binding_map
from quivers.lsp.document import DocumentState
from quivers.lsp.server import (
    _completion_kind,
    _env_kinds_for,
    _render_hover,
    _statement_symbols,
)
from lsprotocol import types as lsp


SOURCE = """\
index Nat = Z | S(Nat)

effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance cell : State[Int]
"""


SCOPED_OPERATIONS = """\
effect Reader[T : Type]
    get : Unit -> T

effect Writer[T : Type]
    get : T -> Unit

instance reader : Reader[Int]
instance writer : Writer[Int]

define read() : Int !{reader} =
    let value <- perform reader.get()
    return value
"""


def test_check_accepts_a_qiec_module(tmp_path: Path) -> None:
    path = tmp_path / "state.qvr"
    path.write_text(SOURCE)
    assert check_file(path) == []


def test_check_exposes_stable_qiec_diagnostic_code(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.qvr"
    path.write_text("effect E\n    go : Unit -> Unit\n    go : Unit -> Unit\n")
    diagnostics = check_file(path)
    assert [(diagnostic.code, diagnostic.line) for diagnostic in diagnostics] == [
        ("qiec-handler", 1)
    ]


def test_host_only_builtin_is_visible_to_cli_and_lsp(tmp_path: Path) -> None:
    source = """object X : Real 1

program native_only : X -> X
    let y = dropout(x)
    return y

export native_only
"""
    path = tmp_path / "native-only.qvr"
    path.write_text(source)

    diagnostics = check_file(path)
    assert "qiec-builtin:host-only" in {diagnostic.code for diagnostic in diagnostics}

    document = DocumentState(uri=path.as_uri())
    document.update(source=source, version=1)
    assert "qiec-builtin:host-only" in {
        diagnostic.code for diagnostic in document.diagnostics
    }
    assert document.qiec_module is None


def test_program_elaboration_gap_is_visible_to_cli_and_lsp(tmp_path: Path) -> None:
    source = """object X : Real 1

program omitted : X -> X
    let y = external_helper(x)
    return y

export omitted
"""
    path = tmp_path / "program-gap.qvr"
    path.write_text(source)

    diagnostics = check_file(path)
    assert "qiec-program-gap" in {diagnostic.code for diagnostic in diagnostics}

    document = DocumentState(uri=path.as_uri())
    document.update(source=source, version=1)
    assert "qiec-program-gap" in {
        diagnostic.code for diagnostic in document.diagnostics
    }
    assert document.qiec_module is not None
    assert document.qiec_module.gap


def test_cli_and_repl_keep_parse_error_locations(tmp_path: Path) -> None:
    """One malformed source, one location, reported the same way twice.

    The operation here is missing its `:`, so the parse fails inside the
    effect body rather than at the header. What the test pins is that
    `qvr check` and a REPL load agree on where, since a diagnostic that
    moves between the two sends a user to the wrong line.
    """
    path = tmp_path / "parse-error.qvr"
    path.write_text("effect E\n    op Unit -> Unit\n")
    expected = ("parse", 2, 4)

    check_diagnostic = check_file(path)[0]
    assert (
        check_diagnostic.code,
        check_diagnostic.line,
        check_diagnostic.col,
    ) == expected

    load_diagnostic = ReplSession().load_file(path).diagnostics[0]
    assert (
        load_diagnostic.code,
        load_diagnostic.line,
        load_diagnostic.col,
    ) == expected


def test_repl_installs_qiec_bindings_and_reports_kinds() -> None:
    session = ReplSession()
    response = session.dispatch(SOURCE)
    assert response.ok, response.diagnostics
    assert session._qiec_module is not None  # noqa: SLF001
    assert {"Nat", "State", "cell", "get", "put"} <= set(session.env)
    assert session.env_kinds()["Nat"] == "type"
    assert session.env_kinds()["State"] == "type"
    assert session.kind_of("State").body.startswith("effect State")
    assert session.type_of("get").body == "get : Unit -> S"
    assert "operations: {get, put}" in session.effects("State").body


def test_completion_includes_qiec_members_and_qualified_operations() -> None:
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    state_items = {item.text: item for item in all_completions(session, "Sta")}
    assert state_items["State"].kind == "type"
    assert state_items["State.get"].kind == "function"
    assert {"cell.get", "cell.put"} == {
        item.text for item in all_completions(session, "cell.")
    }
    assert (
        next(
            item
            for item in all_completions(session, "cell.")
            if item.text == "cell.get"
        ).kind
        == "function"
    )
    assert _completion_kind(state_items["State"].kind) == lsp.CompletionItemKind.Class
    assert (
        _completion_kind(state_items["State.get"].kind)
        == lsp.CompletionItemKind.Function
    )


def test_completion_omits_ambiguous_bare_qiec_members() -> None:
    session = ReplSession()
    response = session.dispatch(
        "effect Reader\n    get : Unit -> Int\n\neffect Writer\n    get : Unit -> Int\n"
    )
    assert response.ok, response.diagnostics

    names = {item.text for item in all_completions(session, "")}
    assert {"Reader.get", "Writer.get"} <= names
    assert "get" not in names


def test_completion_includes_collection_operations() -> None:
    session = ReplSession()
    assert "map" in {item.text for item in all_completions(session, "ma")}
    assert "traverse" in {item.text for item in all_completions(session, "tra")}


def test_document_state_retains_checked_qiec_and_nested_declarations() -> None:
    document = DocumentState(uri="file:///tmp/state.qvr")
    document.update(source=SOURCE, version=1)
    assert document.diagnostics == []
    assert document.qiec_module is not None
    assert type(document.find_decl("put")).__name__ == "QiecOperationDecl"
    assert _env_kinds_for(document)["State"] == "type"
    hover = _render_hover(document, "put")
    assert hover is not None and "put : S -> Unit" in hover


def test_document_symbols_nest_qiec_signature_members() -> None:
    document = DocumentState(uri="file:///tmp/state.qvr")
    document.update(source=SOURCE, version=1)
    symbols = _statement_symbols(document, document.module.statements)
    effect = next(symbol for symbol in symbols if symbol.name == "State")
    assert effect.children is not None
    assert [child.name for child in effect.children] == ["get", "put"]


def test_repeated_operation_names_keep_qualified_identities() -> None:
    module = parse(SCOPED_OPERATIONS)
    bindings = qiec_binding_map(module)
    assert "get" not in bindings
    assert bindings["Reader.get"].declaration.line == 2
    assert bindings["Writer.get"].declaration.line == 5

    session = ReplSession()
    assert session.dispatch(SCOPED_OPERATIONS).ok
    ambiguous = session.type_of("get")
    assert not ambiguous.ok
    assert "Reader.get" in ambiguous.diagnostics[0].message
    assert "Writer.get" in ambiguous.diagnostics[0].message
    assert session.type_of("Reader.get").body == "Reader.get : Unit -> T"


def test_document_resolves_repeated_operations_from_source_context() -> None:
    document = DocumentState(uri="file:///tmp/scoped.qvr")
    document.update(source=SCOPED_OPERATIONS, version=1)

    reader_decl = document.find_decl("get", line=1, col=5)
    writer_decl = document.find_decl("get", line=4, col=5)
    request_decl = document.find_decl("get", line=10, col=34)
    assert getattr(reader_decl, "line", 0) == 2
    assert getattr(writer_decl, "line", 0) == 5
    assert request_decl == reader_decl
    assert document.find_decl("get") is None

    hover = _render_hover(document, "get", line=10, col=34)
    assert hover is not None
    assert "reader.get : Unit -> Int" in hover
    assert "Reader.get : Unit -> T" in hover


def test_qiec_owner_source_slice_includes_signature_members(tmp_path: Path) -> None:
    path = tmp_path / "scoped.qvr"
    path.write_text(SCOPED_OPERATIONS)
    session = ReplSession()
    assert session.load_file(path).ok
    info = session.info("Reader")
    assert "effect Reader" in info.body
    assert "get : Unit -> T" in info.body
    member_info = session.info("Reader.get")
    assert "get : Unit -> T" in member_info.body
    assert "effect Writer" not in member_info.body

    document = DocumentState(uri=path.as_uri())
    document.update(source=SCOPED_OPERATIONS, version=1)
    reader = document.find_decl("Reader")
    assert reader is not None
    from quivers.lsp.server import _slice_source

    assert "get : Unit -> T" in (_slice_source(document, reader) or "")
    reader_get = document.find_decl("Reader.get")
    assert reader_get is not None
    assert (_slice_source(document, reader_get) or "").strip() == "get : Unit -> T"


def test_qiec_semantic_highlighting_is_context_sensitive() -> None:
    pairs = {(span.token, span.text) for span in tokenize(SOURCE) if span.text.strip()}
    assert ("type", "Nat") in pairs
    assert ("type", "State") in pairs
    assert ("function", "get") in pairs
    assert ("type", "Unit") in pairs


def test_qiec_pygments_lexer_marks_types_and_operations() -> None:
    tokens = list(QvrLexer().get_tokens(SOURCE))
    assert any(token in Name.Class and text == "Nat" for token, text in tokens)
    assert any(token in Name.Function and text == "get" for token, text in tokens)


def test_tui_environment_tree_includes_qiec_bindings() -> None:
    from quivers.cli.repl_tui import _populate_scope_tree

    class Node:
        def __init__(self, label: str, *, data: str | None = None) -> None:
            self.label = label
            self.data = data
            self.children: list[Node] = []

        def add(
            self, label: str, *, data: str | None = None, expand: bool = False
        ) -> "Node":
            del expand
            child = Node(label, data=data)
            self.children.append(child)
            return child

        def add_leaf(self, label: str, *, data: str | None = None) -> "Node":
            return self.add(label, data=data)

    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    root = Node("environment")
    _populate_scope_tree(root, session, lambda _name: True, filter_text="")
    leaves = {
        child.data
        for category in root.children
        for child in category.children
        if child.data is not None
    }
    assert {"Nat", "State", "cell", "State.get", "State.put"} <= leaves


def test_qvr_run_and_repl_execute_named_polymorphic_computation(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "identity.qvr"
    path.write_text(
        "define identity[A : Type](value : A) : A !{} =\n    return value\n"
    )
    args = Namespace(
        file=str(path),
        computation="identity",
        arguments=["7"],
        static=["A=Int"],
        runtime=None,
        trace=False,
        json=True,
    )
    assert run_main(args) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 7

    session = ReplSession()
    assert session.load_file(path).ok
    response = session.dispatch(":run identity --static A=Int 7")
    assert response.ok
    assert json.loads(response.body)["value"] == 7
    assert session.last_run is not None
    assert session.runtime_label == "core"
    assert session.dispatch(":detach").ok
    assert session.runtime_label == "detached"
    detached = session.dispatch(":run identity --static A=Int 7")
    assert not detached.ok
    assert detached.diagnostics[0].code == "qiec-run-validator"
    assert session.dispatch(":runtime core").ok


def test_lsp_resolves_qiec_parameters_and_let_binders() -> None:
    source = (
        "effect Reader\n"
        "    get : Unit -> Int\n\n"
        "instance reader : Reader\n\n"
        "define keep(x : Int) : Int !{reader} =\n"
        "    let y <- perform reader.get()\n"
        "    return y\n"
    )
    document = DocumentState(uri="file:///tmp/locals.qvr")
    document.update(source=source, version=1)
    parameter = document.find_decl("x", line=6, col=20)
    local = document.find_decl("y", line=7, col=12)
    assert getattr(parameter, "name", None) == "x"
    assert getattr(parameter, "line", 0) == 6
    assert getattr(local, "name", None) == "y"
    assert getattr(local, "line", 0) == 7
    hover = _render_hover(document, "y", line=7, col=12)
    assert hover is not None and ": (inferred)" in hover


def test_lsp_reports_selected_target_qiec_capabilities() -> None:
    document = DocumentState(uri="file:///tmp/target.qvr", target="stan")
    document.update(source=SCOPED_OPERATIONS, version=1)
    capability = [
        diagnostic
        for diagnostic in document.diagnostics
        if diagnostic.code.startswith("qiec:capability:")
    ]
    assert capability
    assert {diagnostic.severity for diagnostic in capability} == {"error"}
    assert any(
        "Stan" in diagnostic.message or "stan" in diagnostic.message
        for diagnostic in capability
    )


def test_effectful_execution_crosses_cli_repl_and_tui_boundaries(
    tmp_path: Path, capsys
) -> None:
    source = (
        "effect Echo\n"
        "    ping : Int -> Int\n\n"
        "instance echo : Echo\n\n"
        "handler pass for Echo : Int -> Int [coverage=total, implementation=foreign]\n"
        "    ping resumes 1\n\n"
        "define effectful(x : Int) : Int !{} =\n"
        "    handle echo with pass in\n"
        "        let y <- perform echo.ping(x)\n"
        "        return y\n"
    )
    path = tmp_path / "effectful.qvr"
    path.write_text(source)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "core",
                        "options": {"handlers": {"pass": {"kind": "passthrough"}}},
                    }
                ]
            }
        )
    )
    args = Namespace(
        file=str(path),
        computation="effectful",
        arguments=["9"],
        static=[],
        runtime=str(runtime_path),
        trace=False,
        json=True,
    )
    assert run_main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["value"] == 9
    assert "operation.handled" in {event["event"] for event in payload["trace"]}

    session = ReplSession()
    assert session.load_file(path).ok
    assert session.dispatch(f":runtime {runtime_path}").ok
    assert session.dispatch(":run effectful 9").ok
    from quivers.cli.repl_tui import _runtime_status

    assert _runtime_status(session) == (
        "runtime:core entry:effectful(computation) last:effectful=9:Int"
    )


def test_qvr_run_and_repl_honor_a_fuel_budget(tmp_path: Path, capsys) -> None:
    """A diverging computation stops at its budget on both surfaces."""
    path = tmp_path / "spin.qvr"
    path.write_text("define spin(seed : Int) : Int !{} =\n    spin(seed)\n")
    args = Namespace(
        file=str(path),
        computation="spin",
        arguments=["1"],
        static=[],
        runtime=None,
        trace=False,
        json=True,
        fuel=200,
    )
    assert run_main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["code"] == "qiec-run-fuel"

    session = ReplSession()
    assert session.load_file(path).ok
    response = session.dispatch(":run spin 1 --fuel 200")
    assert not response.ok
    assert response.diagnostics[0].code == "qiec-run-fuel"
    rejected = session.dispatch(":run spin 1 --fuel 0")
    assert not rejected.ok
    assert rejected.diagnostics[0].code == "qiec-run-config"


# ---------------------------------------------------------------------------
# One check path: the same codes and positions on every surface.
# ---------------------------------------------------------------------------


_DIAGNOSTIC_SOURCES: dict[str, tuple[str, str, int, int]] = {
    # An unknown callee in a call graph.
    "call": (
        "define shift(x : Real) : Real !{} =\n"
        "    return x + 1.0\n\n"
        "define caller(x : Real) : Real !{} =\n"
        "    let y <- missing(x)\n"
        "    return y\n",
        "qiec-route",
        5,
        13,
    ),
    # A body performing an effect its declared row lacks.
    "row": (
        "instance random : Random\n\n"
        "define draws(x : Real) : Real !{} =\n"
        '    let y <- perform random.sample[Real](site("y"), Normal(x, 1.0))\n'
        "    return y\n",
        "qiec-row",
        3,
        0,
    ),
    # A handler clause naming an operation its interface lacks.
    "handler": (
        "effect Echo\n"
        "    ping : Int -> Int\n\n"
        "instance echo : Echo\n\n"
        "handler pass for Echo : Int -> Int [coverage=total, implementation=foreign]\n"
        "    pong resumes 1\n",
        "qiec-handler",
        7,
        4,
    ),
    # A recursive call at the wrong arity.
    "recursion": (
        "define spin(n : Int) : Int !{} =\n    spin(n, n)\n",
        "qiec-call-arity",
        2,
        4,
    ),
}


@pytest.mark.parametrize("case", sorted(_DIAGNOSTIC_SOURCES))
def test_check_repl_and_tui_report_one_diagnostic(case: str, tmp_path: Path) -> None:
    """`qvr check`, a REPL load, and the TUI's diagnostics panel report
    the same code at the same position for a call-graph, row,
    handler-body, and recursion error."""
    source, code, line, col = _DIAGNOSTIC_SOURCES[case]
    path = tmp_path / f"{case}.qvr"
    path.write_text(source)
    check = [(d.code, d.line, d.col) for d in check_file(path)]
    assert check[0] == (code, line, col), check
    session = ReplSession()
    response = session.load_file(path)
    loaded = [(d.code, d.line, d.col) for d in response.diagnostics]
    assert loaded[0] == (code, line, col), loaded
    assert loaded == check
    message = check_file(path)[0].message
    assert response.diagnostics[0].message == message
    assert "EffectRow(" not in message and "RowEntry(" not in message


def test_row_diagnostics_render_instances_by_name(tmp_path: Path) -> None:
    source, _, _, _ = _DIAGNOSTIC_SOURCES["row"]
    path = tmp_path / "row.qvr"
    path.write_text(source)
    message = check_file(path)[0].message
    assert message == (
        "computation body has effect row !{random : Random}, not declared row !{}"
    )


def test_check_and_repl_report_the_same_target_capability(tmp_path: Path) -> None:
    """`qvr check --target` and a REPL session with `target` set report a
    target's capability diagnostics alike, and the REPL's status line
    names the target."""
    from quivers.cli.repl_tui import _runtime_status

    path = tmp_path / "recursive.qvr"
    path.write_text(
        "define count(n : Int) : Int !{} =\n"
        "    if n == 0 then\n"
        "        return 0\n"
        "    else\n"
        "        let rest <- count(n - 1)\n"
        "        return rest + 1\n"
    )
    checked = [(d.code, d.line, d.col) for d in check_file(path, target="bugs")]
    session = ReplSession()
    assert session.set_option("target=bugs").ok
    response = session.load_file(path)
    loaded = [(d.code, d.line, d.col) for d in response.diagnostics]
    assert loaded == checked
    assert loaded, "BUGS admits no call, so the recursive computation is refused"
    assert all(code.startswith("qiec:capability") for code, _, _ in loaded)
    assert _runtime_status(session) == "runtime:core target:bugs"
    rejected = session.set_option("target=nowhere")
    assert not rejected.ok
    assert session.options.target == "bugs"
    assert session.set_option("target=").ok
    assert session.load_file(path).diagnostics == ()


def test_target_checks_include_program_plan_refusals(tmp_path: Path) -> None:
    """Target tooling must reject a checked construct the PPL plan cannot emit."""
    source = """\
object Item : FinSet 3
object Resp : FinSet 6
object Class : FinSet 4

program demo : Resp -> Resp
    sample probs : Class <- HalfNormal(scale=1.0)
    marginalize cls : Class <- Categorical(probs=probs) [over=Item, reduction=sum]
        observe r : Resp <- HalfNormal(scale=1.0) [via=idx]
    return probs

export demo
"""
    path = tmp_path / "sum-marginal.qvr"
    path.write_text(source)

    checked = check_file(path, target="pyro")
    assert {diagnostic.code for diagnostic in checked} == {"marginalize:reduction:sum"}

    document = DocumentState(uri=path.as_uri(), target="pyro")
    document.update(source=source, version=1)
    assert {diagnostic.code for diagnostic in document.diagnostics} == {
        "marginalize:reduction:sum"
    }

    session = ReplSession()
    assert session.set_option("target=pyro").ok
    response = session.load_file(path)
    assert {diagnostic.code for diagnostic in response.diagnostics} == {
        "marginalize:reduction:sum"
    }


def test_status_line_shows_program_entries_and_failures(tmp_path: Path) -> None:
    from quivers.cli.repl_tui import _runtime_status

    path = tmp_path / "prog.qvr"
    path.write_text(
        "object Obs : FinSet 4\n\n"
        "program prog : Obs -> Obs\n"
        "    sample a <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=a, scale=0.5)\n"
        "    return a\n"
        "export prog\n\n"
        "define spin(n : Int) : Int !{} =\n    spin(n)\n"
    )
    session = ReplSession()
    assert session.load_file(path).ok
    response = session.dispatch(":run prog [0.1,0.2,0.3,0.4] --site a=0.5 --seed 0")
    assert response.ok, response.diagnostics
    status = _runtime_status(session)
    assert status.startswith("runtime:core entry:prog(program) last:prog=0.5:Real")
    assert "log_joint=" in status
    failed = session.dispatch(":run spin 1 --fuel 50")
    assert not failed.ok
    assert failed.diagnostics[0].code == "qiec-run-fuel"
    assert _runtime_status(session) == "runtime:core failed:spin[qiec-run-fuel]"
    assert session.dispatch(":run spin --fuel 50").diagnostics[0].code == (
        "qiec-run-arity"
    )
    assert _runtime_status(session) == "runtime:core failed:spin[qiec-run-arity]"


def test_repl_introspects_prelude_effects_and_qualified_operations(
    tmp_path: Path,
) -> None:
    """`:kind`, `:type`, `:effects`, and completion reach the prelude's
    interfaces and the operations an instance addresses."""
    from quivers.cli.repl_complete import all_completions

    path = tmp_path / "calls.qvr"
    path.write_text(
        "instance random : Random\n"
        "instance score : Score\n\n"
        "define noisy(x : Real) : Real !{random, score} =\n"
        '    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))\n'
        "    perform score.add(weight(-0.25 * y * y))\n"
        "    return y\n\n"
        "define shift(x : Real) : Real !{} =\n"
        "    let y <- noisy(x)\n"
        "    return y + 1.0\n"
    )
    session = ReplSession()
    load = session.load_file(path)
    assert not load.ok
    assert load.diagnostics[0].code == "qiec-row"
    path.write_text(
        path.read_text().replace(
            "!{} =\n    let y <- noisy", "!{random, score} =\n    let y <- noisy"
        )
    )
    assert session.load_file(path).ok
    assert session.dispatch(":kind Random").body == "effect Random"
    assert session.dispatch(":type Random.sample").body == (
        "Random.sample[a : Type] : Site[a] * Sampleable[a] -> a"
    )
    assert session.dispatch(":type random.sample").body == (
        "random.sample[a : Type] : Site[a] * Sampleable[a] -> a"
    )
    assert session.dispatch(":type score.add").body == "score.add : LogWeight -> Unit"
    assert session.dispatch(":effects Random").body == (
        "effect Random (prelude)\n  operations: {sample}"
    )
    effects = session.dispatch(":effects shift").body
    assert "inferred  : !{random : Random, score : Score}" in effects
    assert {item.text for item in all_completions(session, "random.")} == {
        "random.sample"
    }
    missing = session.dispatch(":type random.draw")
    assert not missing.ok
    assert (
        session.dispatch(":type Random")
        .diagnostics[0]
        .message.endswith("use :kind Random")
    )


def test_instance_qualified_operations_substitute_the_instance_arguments() -> None:
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    assert session.dispatch(":type cell.get").body == "cell.get : Unit -> Int"
    assert session.dispatch(":type cell.put").body == "cell.put : Int -> Unit"
    assert session.dispatch(":type State.put").body == "State.put : S -> Unit"
    assert session.dispatch(":info cell").body.splitlines()[0] == (
        "instance cell : State[Int]"
    )
    assert "-- declared at <repl>:7:0" in session.dispatch(":info cell").body


@pytest.mark.parametrize(
    "module_name,entry,program",
    [
        ("quivers.cli", "main", "qvr"),
        ("quivers.cli.lsp", "_entry", "qvr-lsp"),
        ("quivers.kernel.install", "_entry", "qvr-kernel"),
    ],
)
def test_entry_points_report_the_installed_version(
    module_name, entry, program, monkeypatch, capsys
) -> None:
    import importlib
    import sys
    from quivers import __version__

    monkeypatch.setattr(sys, "argv", [program, "--version"])
    with pytest.raises(SystemExit) as exit_info:
        getattr(importlib.import_module(module_name), entry)()
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"{program} {__version__}"
