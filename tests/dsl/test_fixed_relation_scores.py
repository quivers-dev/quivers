"""End-to-end source-declared sparse relation scores."""

from __future__ import annotations

import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.dsl import Compiler, parse
from quivers.dsl.ast_nodes import QiecRelationScoreDecl
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.qiec import (
    LOG_WEIGHT,
    REAL,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationRuntimeProvider,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_runtime import _request


def _module(source: str) -> QiecModule:
    compiler = Compiler(
        parse(source), module_name="source_relation_scores", file_path="scores.qvr"
    )
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return module


def test_declarations_round_trip_remain_callable_and_keep_live_gradients() -> None:
    source = """define invoke() : Real !{} =
    let score <- paired_words()
    return score

#! A typed pair score.
relation_score paired_words : Real [operation=pair_weight]
relation_score likelihood : LogWeight [operation=conditional_log_prob]
"""
    ast = parse(source)
    printed = module_to_source(ast)
    declared = tuple(
        stmt
        for stmt in parse(printed).statements
        if isinstance(stmt, QiecRelationScoreDecl)
    )
    assert [(stmt.name, stmt.operation, stmt.docs) for stmt in declared] == [
        ("paired_words", "pair_weight", ("A typed pair score.",)),
        ("likelihood", "conditional_log_prob", ()),
    ]
    module = loads(dumps(_module(printed)))
    assert isinstance(module, QiecModule)
    pair, parameters, ids = _request("pair_weight", MARKOV)
    conditional, _, _ = _request("conditional_log_prob", MARKOV)
    providers = (
        CoreRuntimeProvider(),
        FixedRelationRuntimeProvider.for_score("paired_words", pair, parameters),
        FixedRelationRuntimeProvider.for_score("likelihood", conditional, parameters),
    )
    runtime = RuntimeConfiguration(selections=(), providers=providers)
    a, b, _ = (parameters[id_] for id_ in ids)
    scored = run_named(module, "invoke", runtime=runtime)
    assert scored.result_type == REAL
    assert isinstance(scored.value, torch.Tensor) and scored.value.item() == 0.25
    log_prob = run_named(module, "likelihood", runtime=runtime)
    assert log_prob.result_type == LOG_WEIGHT
    assert isinstance(log_prob.value, torch.Tensor)
    assert log_prob.value.exp().item() == pytest.approx(0.5)
    pair_grad = torch.autograd.grad(scored.value, (a, b))
    conditional_grad = torch.autograd.grad(log_prob.value, (a, b))
    assert [value.item() for value in pair_grad] == [2, 0]
    assert [value.item() for value in conditional_grad] == [4, -2]


def test_declared_operation_is_checked_even_when_the_host_result_type_agrees() -> None:
    module = _module("relation_score result : LogWeight [operation=pair_weight]\n")
    wrong, parameters, _ = _request("conditional_log_prob", LOG_PROB)
    provider = FixedRelationRuntimeProvider.for_score("result", wrong, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    with pytest.raises(ExecutionFailure, match="different declared operation"):
        run_named(module, "result", runtime=runtime)
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        projected = project_for_target(module, discharged, target)
        assert [(item.computation, item.reason) for item in projected.obstacles] == [
            (
                "result",
                "installs the foreign handler `__relation_score_result`, "
                "whose clause `apply` a runtime provider implements",
            ),
        ]


@pytest.mark.parametrize(
    "source, message",
    (
        ("relation_score bad : Bool [operation=pair_weight]\n", "Real or LogWeight"),
        (
            "relation_score bad : Real [operation=conditional_log_prob]\n",
            "requires a LogWeight",
        ),
        (
            "relation_score same : Real [operation=pair_weight]\n"
            "relation_score same : Real [operation=pair_weight]\n",
            "duplicate computation",
        ),
    ),
)
def test_invalid_score_declarations_refuse_during_checked_lowering(
    source, message
) -> None:
    with pytest.raises(QiecDiagnosticError, match=message):
        _module(source)
