"""Carrier shapes against authored control and independent reference runs."""

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    HandlerId,
    QiecModule,
    ResumptionGrade,
    classify_handler_carrier,
    dumps,
    loads,
    run_named,
)

_SOURCE = """
effect Step
    advance : Real -> Real

effect Audit
    read_value : Real -> Real

instance step : Step
instance another_step : Step
instance audit : Audit

handler relay for Audit : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    read_value(v : Real) resumes 1 =>
        resume(v + 1.0)

define pure_inner(v : Real) : Real !{} =
    return v + 1.0

define pure_outer(v : Real) : Real !{} =
    pure_inner(v)

define hidden_inner(v : Real) : Real !{} =
    handle audit with relay in
        perform audit.read_value(v)

define hidden_outer(v : Real) : Real !{} =
    hidden_inner(v)

define recursive_inner(v : Real) : Real !{} =
    if v > 0.0 then
        recursive_inner(v - 1.0)
    else
        return v

handler tail for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        let y <- pure_outer(v)
        resume(y + s) with (v)

handler optional for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes aff =>
        if v > 0.0 then
            resume(v + s)
        else
            return v

handler post for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        let y <- resume(v + s)
        return y + 1.0

handler branch_post for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        if v > 0.0 then
            resume(v + s)
        else
            let y <- resume(v + s)
            return y + 1.0

handler repeated for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes omega =>
        let first <- resume(v + s)
        let second <- resume(v - s)
        return first + second

handler hidden for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        let y <- hidden_outer(v)
        resume(y + s)

handler recursive for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        let y <- recursive_inner(v)
        resume(y + s)

handler changed_return for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return 2.0 * x
    advance(v : Real) resumes 1 =>
        let y <- pure_outer(v)
        resume(y + s) with (v)

handler constant_branch for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes aff =>
        if true then
            resume(v + s)
        else
            return v

handler foreign_step for Step : Real -> Real [coverage=total, implementation=foreign]
    advance resumes 1

handler declared_omega for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes omega =>
        resume(v + s)

handler aborts for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 0 =>
        return v - s

handler effectful_return for Step : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        hidden_outer(x)
    advance(v : Real) resumes 1 =>
        resume(v + s)

handler introduces_audit for Step : Real -> Real with (s : Real) [introduces=!{audit}, coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    advance(v : Real) resumes 1 =>
        let result <- perform audit.read_value(v)
        resume(result + s)

effect Either
    left : Real -> Real
    right : Real -> Real

handler partial for Either : Real -> Real [coverage=partial, forwards=unknown, implementation=authored]
    return x =>
        return x
    left(v : Real) resumes 1 =>
        resume(v)

handler mixed for Either : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    left(v : Real) resumes 1 =>
        resume(v)
    right(v : Real) resumes 0 =>
        return v

handler negate for Step : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return 0.0 - x
    advance(v : Real) resumes 1 =>
        resume(v)

handler overwrite for Step : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return 1.0
    advance(v : Real) resumes 1 =>
        resume(v)

define negate_outside() : Real !{} =
    handle step with negate in
        handle another_step with overwrite in
            let x <- perform step.advance(4.0)
            let y <- perform another_step.advance(5.0)
            return x + y

define overwrite_outside() : Real !{} =
    handle another_step with overwrite in
        handle step with negate in
            let x <- perform step.advance(4.0)
            let y <- perform another_step.advance(5.0)
            return x + y
"""

_CASES = {
    "tail": ((2, 10, 1),),
    "optional": ((2, 8, 1), (-2, -2, 0)),
    "post": ((2, 9, 1),),
    "branch_post": ((2, 8, 1), (-2, 1, 1)),
    "repeated": ((2, 8, 2),),
    "hidden": ((2, 10, 1),),
    "recursive": ((2, 4, 1),),
    "changed_return": ((2, 20, 1),),
    "constant_branch": ((2, 8, 1),),
    "aborts": ((2, 0, 0),),
    "declared_omega": ((2, 8, 1),),
    "effectful_return": ((2, 9, 1),),
}


def _compile(source: str = _SOURCE) -> QiecModule:
    wrappers = "".join(
        f"""
define test_{name}(v : Real) : Real !{{}} =
    handle step with {name}(2.0) in
        let y <- perform step.advance(v)
        return 2.0 * y
"""
        for name in _CASES
    )
    compiler = Compiler(parse(source + wrappers), module_name="handler-carriers")
    assert compiler.qiec_module is not None
    return compiler.qiec_module


@pytest.fixture(scope="module")
def module() -> QiecModule:
    return _compile()


def _handlers(module: QiecModule) -> dict[str, HandlerId]:
    return {handler.name: handler.id for handler in module.handlers}


@pytest.mark.parametrize(
    ("name", "carrier", "completion", "obstacle"),
    (
        ("tail", "tail-state", ("resume",), None),
        ("optional", "tail-or-abort", ("abort", "resume"), None),
        ("post", "continuation", (), "before its clause returns"),
        ("branch_post", "continuation", (), "before its clause returns"),
        ("repeated", "continuation", (), "before its clause returns"),
        ("hidden", "continuation", (), "Handle requires general control"),
        ("recursive", "continuation", (), "recursive call"),
        ("changed_return", "tail-state", ("resume",), None),
        ("constant_branch", "tail-state", ("resume",), None),
        ("foreign_step", "continuation", (), "opaque foreign implementation"),
        ("aborts", "tail-or-abort", ("abort",), None),
        ("declared_omega", "tail-state", ("resume",), None),
        ("effectful_return", "continuation", ("resume",), "return action"),
        ("introduces_audit", "continuation", (), "introduces residual effects"),
        ("partial", "continuation", (), "incomplete operation coverage"),
    ),
)
def test_classifies_actual_clause_and_prefix_shapes(
    module: QiecModule,
    name: str,
    carrier: str,
    completion: tuple[str, ...],
    obstacle: str | None,
) -> None:
    ids = _handlers(module)
    result = classify_handler_carrier(module, ids[name])
    assert result.handler == ids[name]
    assert result.kind == carrier
    assert result.clauses[0].completions == completion
    if obstacle is None:
        assert result.obstacle is None
    else:
        assert result.obstacle is not None
        assert obstacle in result.obstacle
    definition = next(item for item in module.handlers if item.id == ids[name])
    assert result.clauses[0].operation == definition.clauses[0].operation
    assert result.clauses[0].grade == definition.clauses[0].grade


def test_whole_handler_joins_all_nominal_operation_clauses(module: QiecModule) -> None:
    actual = classify_handler_carrier(module, _handlers(module)["mixed"])
    assert actual.kind == "tail-or-abort"
    assert tuple(clause.kind for clause in actual.clauses) == (
        "tail-state",
        "tail-or-abort",
    )
    assert actual.clauses[1].grade == ResumptionGrade.ZERO


def test_return_action_and_unrestricted_grade_stay_visible(module: QiecModule) -> None:
    ids = _handlers(module)
    identity = classify_handler_carrier(module, ids["tail"])
    transformed = classify_handler_carrier(module, ids["changed_return"])
    opaque = classify_handler_carrier(module, ids["foreign_step"])
    omega = classify_handler_carrier(module, ids["declared_omega"])
    assert identity.kind == transformed.kind == omega.kind == "tail-state"
    assert (
        identity.return_action,
        transformed.return_action,
        opaque.return_action,
    ) == ("identity", "authored", "opaque")
    assert omega.clauses[0].grade == ResumptionGrade.UNRESTRICTED
    assert transformed.fingerprint != identity.fingerprint


def test_tail_shapes_alone_do_not_establish_commutation(module: QiecModule) -> None:
    ids = _handlers(module)
    for name in ("negate", "overwrite"):
        report = classify_handler_carrier(module, ids[name])
        assert report.kind == "tail-state"
        assert report.return_action == "authored"

    first = run_named(module, "negate_outside").value
    second = run_named(module, "overwrite_outside").value
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)
    assert (float(first), float(second)) == (-1, 1)


def test_changed_nested_call_under_the_same_declared_handler_changes_its_shape(
    module: QiecModule,
) -> None:
    ids = _handlers(module)
    previous = classify_handler_carrier(module, ids["tail"])
    modified = _compile(_SOURCE.replace("    pure_inner(v)\n", "    hidden_inner(v)\n"))
    assert _handlers(modified)["tail"] == previous.handler
    current = classify_handler_carrier(modified, previous.handler)
    assert current.kind == "continuation"
    assert current.obstacle is not None
    assert "Handle" in current.obstacle
    assert current.fingerprint != previous.fingerprint


def test_binding_roundtrips_the_checked_source_and_refuses_unknown_ids(
    module: QiecModule,
) -> None:
    identity = _handlers(module)["tail"]
    decoded = loads(dumps(module))
    assert isinstance(decoded, QiecModule)
    assert classify_handler_carrier(module, identity) == classify_handler_carrier(
        decoded, identity
    )
    with pytest.raises(ValueError, match="unknown handler"):
        classify_handler_carrier(module, HandlerId.derive("another-module", "tail"))


@pytest.mark.parametrize(
    ("name", "value", "answer", "resumes"),
    tuple(
        (name, value, answer, resumes)
        for name, cases in _CASES.items()
        for value, answer, resumes in cases
    ),
)
def test_reference_trace_against_independent_scalar_formula(
    module: QiecModule, name: str, value: int, answer: int, resumes: int
) -> None:
    operation = next(
        str(item.operations[0].id) for item in module.effects if item.ref.name == "Step"
    )
    result = run_named(module, f"test_{name}", (value,))
    assert isinstance(result.value, torch.Tensor)
    assert float(result.value) == answer
    shots = tuple(
        item.detail["shot"]
        for item in result.trace
        if item.event == "resumption.invoked" and item.detail["operation"] == operation
    )
    assert shots == tuple(range(resumes))
