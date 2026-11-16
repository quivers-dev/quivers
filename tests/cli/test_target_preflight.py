"""Shared target preflight preserves diagnostics without repeating analysis."""

from pathlib import Path
from unittest.mock import patch

import pytest

from quivers.cli.check import check_file
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse
from quivers.dsl.qiec_tooling import analyze_module, target_diagnostics
from quivers.lsp.document import DocumentState
from quivers.transpile.plan import checked_program_plan
from quivers.transpile.qiec_ir import analyze_qiec_capabilities


RECURSIVE = """\
define count(n : Int) : Int !{} =
    if n == 0 then
        return 0
    else
        let rest <- count(n - 1)
        return rest + 1
"""

PLAN_REFUSAL = """\
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

ADMITTED = """\
object Out : Real 1
program demo(x : Real) : Out -> Out
    sample a <- Normal(loc=x, scale=1.0)
    return a
export demo
"""


@pytest.mark.parametrize(
    ("source", "target", "has_capabilities", "has_diagnostics"),
    [
        (RECURSIVE, "bugs", True, True),
        (PLAN_REFUSAL, "pyro", False, True),
        (ADMITTED, "pyro", False, False),
    ],
)
def test_target_preflight_reuses_checked_evidence_and_skips_blocked_plan(
    source, target, has_capabilities, has_diagnostics
):
    syntax = parse(source)
    checked = analyze_module(syntax, module_name="target").qiec_module
    assert checked is not None
    with (
        patch(
            "quivers.transpile.qiec_ir.analyze_qiec_capabilities",
            wraps=analyze_qiec_capabilities,
        ) as capabilities,
        patch(
            "quivers.transpile.plan.checked_program_plan", wraps=checked_program_plan
        ) as plan,
        patch(
            "quivers.dsl.compiler.core.Compiler.__init__",
            side_effect=AssertionError("target preflight must reuse checked evidence"),
        ),
    ):
        diagnostics = target_diagnostics(checked, syntax, target)
    capabilities.assert_called_once_with(checked, target)
    assert bool(diagnostics) is has_diagnostics
    if has_capabilities:
        plan.assert_not_called()
        assert all(d.code.startswith("qiec:capability:") for d in diagnostics)
        assert all(d.line == 1 for d in diagnostics)
    else:
        plan.assert_called_once_with(checked, syntax, target)
        if has_diagnostics:
            assert [d.code for d in diagnostics] == ["marginalize:reduction:sum"]
            assert [(d.line, d.col) for d in diagnostics] == [(0, 0)]


@pytest.mark.parametrize(
    ("source", "target"), [(RECURSIVE, "bugs"), (PLAN_REFUSAL, "pyro")]
)
def test_target_frontends_preserve_identical_messages_locations_and_order(
    tmp_path: Path, source: str, target: str
):
    path = tmp_path / "target.qvr"
    path.write_text(source)
    with patch("quivers.cli.check.analyze_module", wraps=analyze_module) as analyze:
        checked = check_file(path, target=target)
    assert analyze.call_count == 1
    expected = [(d.code, d.message, d.line, d.col) for d in checked]
    assert expected

    session = ReplSession()
    assert session.set_option(f"target={target}").ok
    with patch(
        "quivers.cli.repl_session.analyze_module", wraps=analyze_module
    ) as analyze:
        response = session.load_file(path)
    assert analyze.call_count == 1
    assert [
        (d.code, d.message, d.line, d.col) for d in response.diagnostics
    ] == expected

    document = DocumentState(uri=path.as_uri(), target=target)
    with patch("quivers.lsp.document.analyze_module", wraps=analyze_module) as analyze:
        document.update(source=source, version=1)
    assert analyze.call_count == 1
    assert [
        (d.code, d.message, d.line, d.col) for d in document.diagnostics
    ] == expected

    syntax, checked_module = document.module, document.qiec_module
    with (
        patch("quivers.lsp.document.parse", side_effect=AssertionError("reparse")),
        patch(
            "quivers.lsp.document.analyze_module",
            side_effect=AssertionError("reanalyze"),
        ),
    ):
        document.retarget(None)
        assert document.diagnostics == []
        document.retarget(target)
        document.retarget(target)
    assert document.module is syntax and document.qiec_module is checked_module
    assert [
        (d.code, d.message, d.line, d.col) for d in document.diagnostics
    ] == expected
