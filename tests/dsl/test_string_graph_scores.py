"""Check authored string graph queries through native QVR and QIEC."""

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.dsl.ast_nodes import QiecStringGraphScoreDecl
from quivers.dsl import Compiler, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    QiecModule,
    RuntimeConfiguration,
    TreeStringGraphRuntimeProvider,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_string_graph_serialization import _TIED, _UNIT, _request


def _module(source: str) -> QiecModule:
    compiler = Compiler(
        parse(source), module_name="source_string_graph_scores", file_path="graph.qvr"
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


@pytest.mark.parametrize(
    ("operation", "name", "algebra", "word"),
    (
        ("tree_partition", "Real", MARKOV, None),
        ("tree_partition", "LogWeight", LOG_PROB, None),
        ("tree_log_prob", "LogWeight", MARKOV, (0,)),
        ("tree_log_prob", "LogWeight", LOG_PROB, (0,)),
    ),
)
def test_authored_graph_score_round_trips_and_uses_current_parameters(
    operation, name, algebra, word
):
    source = f"""define evaluate() : {name} !{{}} =
    let value <- graph_query()
    return value

#! A bounded tree query.
string_graph_score graph_query : {name} [operation={operation}]
"""
    printed = module_to_source(parse(source))
    declaration = next(
        item
        for item in parse(printed).statements
        if isinstance(item, QiecStringGraphScoreDecl)
    )
    assert declaration.name == "graph_query"
    assert declaration.operation == operation
    assert declaration.docs == ("A bounded tree query.",)

    p = nn.Parameter(
        torch.tensor(-1.0 if algebra is LOG_PROB else 0.4, dtype=torch.float64)
    )
    parameters = {_TIED: p, _UNIT: p.new_tensor(0.0 if algebra is LOG_PROB else 1.0)}
    request = _request(algebra, word)
    module = loads(dumps(_module(printed)))
    assert isinstance(module, QiecModule)
    provider = TreeStringGraphRuntimeProvider.for_score(
        "graph_query", request, parameters
    )
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    for weight in (0.4, 0.65):
        with torch.no_grad():
            raw = p.new_tensor(weight)
            p.copy_(raw.log() if algebra is LOG_PROB else raw)
        x = p.exp() if algebra is LOG_PROB else p
        partition = (2 * x.square() + 1) * (1 + x)
        if word is None:
            expected = partition.log() if algebra is LOG_PROB else partition
        else:
            expected = (2 * x.square() / (2 * x.square() + 1)).log()
        result = run_named(module, "evaluate", runtime=runtime)
        assert result.result_type == request.result_type
        assert isinstance(result.value, torch.Tensor)
        torch.testing.assert_close(result.value, expected)
        (gradient,) = torch.autograd.grad(result.value, p)
        (expected_gradient,) = torch.autograd.grad(expected, p)
        torch.testing.assert_close(gradient, expected_gradient)


def test_authored_handler_identity_checks_the_objective_and_target():
    module = _module(
        "string_graph_score query : LogWeight [operation=tree_partition]\n"
    )
    wrong = _request(LOG_PROB, (0,))
    provider = TreeStringGraphRuntimeProvider.for_score(
        "query", wrong, {_TIED: torch.tensor(-1.0), _UNIT: torch.tensor(0.0)}
    )
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    with pytest.raises(
        ExecutionFailure, match="incompatible tree string graph handler"
    ):
        run_named(module, "query", runtime=runtime)
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        projection = project_for_target(module, discharged, target)
        assert [(item.computation, item.reason) for item in projection.obstacles] == [
            (
                "query",
                "installs the foreign handler `__string_graph_score_query`, "
                "whose clause `apply` a runtime provider implements",
            )
        ]


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    (
        (
            "string_graph_score bad : Real [operation=tree_log_prob]\n",
            "tree word score requires a LogWeight",
        ),
        (
            "string_graph_score bad : Bool [operation=tree_partition]\n",
            "string graph score requires a Real or LogWeight",
        ),
        (
            "relation_score same : Real [operation=pair_weight]\n"
            "string_graph_score same : Real [operation=tree_partition]\n",
            "duplicate computation",
        ),
    ),
)
def test_invalid_graph_score_source_refuses_during_lowering(source, diagnostic):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        _module(source)
