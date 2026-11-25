"""Shared target preflight preserves diagnostics without repeating analysis."""

from pathlib import Path
from unittest.mock import patch

import pytest

from quivers.cli.check import check_file
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse
from quivers.dsl.qiec_tooling import analyze_module, target_diagnostics
from quivers.lsp.document import DocumentState
from quivers.qiec.discharge import discharge_module
from quivers.transpile.plan import Lower, _Planner, checked_program_plan
from quivers.transpile.qiec_ir import (
    IRQiecModule,
    analyze_qiec_capabilities,
    lower_qiec_ir,
    project_for_target,
)


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
            "quivers.transpile.plan._Planner.plan",
            autospec=True,
            side_effect=_Planner.plan,
        ) as plan,
        patch(
            "quivers.transpile.plan.project_for_target", wraps=project_for_target
        ) as project,
        patch(
            "quivers.transpile.qiec_ir.project_for_target", wraps=project_for_target
        ) as reproject,
        patch("quivers.transpile.qiec_ir.lower_qiec_ir", wraps=lower_qiec_ir) as lower,
        patch(
            "quivers.transpile.plan.discharge_module", wraps=discharge_module
        ) as discharge,
        patch(
            "quivers.transpile.qiec_ir.discharge_module", wraps=discharge_module
        ) as redischarge,
        patch(
            "quivers.dsl.compiler.core.Compiler.__init__",
            side_effect=AssertionError("target preflight must reuse checked evidence"),
        ),
    ):
        diagnostics = target_diagnostics(checked, syntax, target)
    assert capabilities.call_count == 1
    projection, selected_target = capabilities.call_args.args
    assert isinstance(projection, IRQiecModule) and selected_target == target
    discharge.assert_called_once_with(checked)
    redischarge.assert_not_called()
    project.assert_called_once()
    reproject.assert_not_called()
    assert project.call_args.args[0] is checked
    assert project.call_args.args[2] == target
    assert lower.call_count == 1
    assert bool(diagnostics) is has_diagnostics
    if has_capabilities:
        plan.assert_not_called()
        assert all(d.code.startswith("qiec:capability:") for d in diagnostics)
        assert all(d.line == 1 for d in diagnostics)
    else:
        assert plan.call_count == 1
        planner = plan.call_args.args[0]
        assert planner.preparation.source is syntax
        assert (
            planner.module is checked and planner.preparation.projection is projection
        )
        if has_diagnostics:
            assert [d.code for d in diagnostics] == ["marginalize:reduction:sum"]
            assert [(d.line, d.col) for d in diagnostics] == [(0, 0)]


def test_target_projection_is_owned_by_each_request_and_revised_source():
    source = parse(ADMITTED)
    checked = analyze_module(source, module_name="target").qiec_module
    changed_source = parse(ADMITTED.replace("scale=1.0", "scale=2.0"))
    changed_checked = analyze_module(changed_source, module_name="target").qiec_module
    assert checked is not None and changed_checked is not None
    with (
        patch(
            "quivers.transpile.plan.project_for_target", wraps=project_for_target
        ) as project,
        patch(
            "quivers.transpile.qiec_ir.analyze_qiec_capabilities",
            wraps=analyze_qiec_capabilities,
        ) as capabilities,
    ):
        assert target_diagnostics(checked, source, "pyro") == ()
        assert target_diagnostics(checked, source, "pyro") == ()
        assert target_diagnostics(changed_checked, changed_source, "pyro") == ()
    assert project.call_count == 3
    assert all(
        call.args[0] is expected
        for call, expected in zip(
            project.call_args_list, (checked, checked, changed_checked), strict=True
        )
    )
    projections = [call.args[0] for call in capabilities.call_args_list]
    assert len({id(projection) for projection in projections}) == 3


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize(
    "source", [ADMITTED, RECURSIVE], ids=["program", "computation"]
)
def test_lower_projects_its_checked_module_once(optimize, source):
    with patch(
        "quivers.transpile.plan.project_for_target", wraps=project_for_target
    ) as project:
        result = Lower().forward(parse(source), target="pyro", optimize=optimize)
    assert project.call_count == 1
    assert isinstance(result.module, IRQiecModule)


def test_public_checked_program_plan_keeps_the_same_owned_projection_path():
    source = parse(ADMITTED)
    checked = analyze_module(source, module_name="target").qiec_module
    assert checked is not None
    with patch(
        "quivers.transpile.plan.project_for_target", wraps=project_for_target
    ) as project:
        result = checked_program_plan(checked, source, "pyro")
    project.assert_called_once()
    assert project.call_args.args[0] is checked
    assert result is not None and result.name == "demo"


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
