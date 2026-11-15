"""The same symbolic data contract remains usable in every frontend."""

from pathlib import Path

from quivers.cli.check import check_file
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_highlight import tokenize
from quivers.cli.repl_session import ReplSession
from quivers.dsl import module_to_source, parse
from quivers.lsp.document import DocumentState
from quivers.lsp.server import (
    _format_document,
    _qiec_local_completion_items,
    _render_hover,
)

SOURCE = """object Row : FinSet _
object Out : Real 1
program regression : Row -> Out
    data x : Tensor[Real]([Row])
    data y : Tensor[Real]([Row])
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


def test_symbolic_contract_checks_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "model.qvr"
    path.write_text(SOURCE)
    assert check_file(path) == []
    assert module_to_source(parse(module_to_source(parse(SOURCE)))) == module_to_source(
        parse(SOURCE)
    )


def test_repl_inspects_and_runs_before_tensor_specialization() -> None:
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    assert {"Row", "Out", "regression"} <= session.env.keys()
    assert "regression" in session.dispatch(":browse").body
    assert "Tensor[Real]([Row])" in session.dispatch(":type regression::x").body
    assert "data-input" in session.dispatch(":info regression::x").body
    assert "data x" in session.dispatch(":browse programs").body
    candidates = all_completions(session, "regression::")
    assert "regression::x" in {candidate.text for candidate in candidates}
    assert "data" in {candidate.text for candidate in all_completions(session, "dat")}
    result = session.dispatch(
        ":run regression --data x=[1.0,2.0] --data y=[0.5,1.0] --seed 7"
    )
    assert result.ok, result.diagnostics
    rejected = session.dispatch(":run regression --data x=[1.0] --data y=[0.5,1.0]")
    assert not rejected.ok


def test_lsp_resolves_declared_data_in_its_program_scope() -> None:
    doc = DocumentState(uri="file:///tmp/model.qvr")
    doc.update(source=SOURCE, version=1)
    assert not doc.diagnostics
    declaration = doc.find_decl("x", line=6, col=18)
    assert declaration is not None and declaration.line == 4
    assert doc.find_decl("x", line=9, col=0) is None
    hover = _render_hover(doc, "x", line=6, col=18)
    assert hover is not None and "data x : Tensor[Real]([Row])" in hover
    candidates = _qiec_local_completion_items(doc, "x", line=6, col=18)
    assert [item.label for item in candidates] == ["x"]
    assert candidates[0].detail == "data x : Tensor[Real]([Row])"
    edits = _format_document(doc)
    assert edits is not None and "data x" in edits[0].new_text
    tokens = tokenize(SOURCE)
    assert any(token.text == "data" and token.token == "keyword" for token in tokens)
    doc.update(
        source=SOURCE.replace("data y : Tensor[Real]", "data y : Tensor[Int]"),
        version=2,
    )
    assert doc.diagnostics


def test_jupyter_executes_and_inspects_declared_data() -> None:
    from quivers.kernel.quivers_kernel import QuiversKernel

    kernel = QuiversKernel()
    captured = []
    kernel.send_response = lambda socket, kind, content: captured.append(
        (kind, content)
    )
    kernel.iopub_socket = None
    assert kernel.do_execute(SOURCE, silent=True)["status"] == "ok"
    assert "regression::x" in kernel.do_complete("regression::", 12)["matches"]
    inspected = kernel.do_inspect("regression::x", 13)
    assert inspected["found"]
    assert "Tensor[Real]([Row])" in inspected["data"]["text/plain"]
    assert (
        kernel.do_execute(
            ":run regression --data x=[1.0,2.0] --data y=[0.5,1.0]", silent=True
        )["status"]
        == "ok"
    )
