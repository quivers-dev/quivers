"""Authored-to-runtime checks of explicitly typed binary sparse composition."""

import pytest
import torch
from torch import nn

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import qiec_bindings
from quivers.qiec import (
    LOG_WEIGHT,
    REAL,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedComposedRelationRequest,
    FixedRelationRuntimeProvider,
    ParameterId,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target

_MODULE = "authored_relation_composition"
_SOURCE = """object Letters : FinSet 2
object Other : FinSet 2

#! The two operands can be declared later.
relation pair_channel = compose(source_channel, target_channel)
relation_score pair : Real [operation=pair_weight, relation=pair_channel]

relation source_channel [algebra=markov, states=3]
    input Letters
    output Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(0, eps), weight=shared]
    arc 0 -> 1 [labels=(0, eps), weight=shared]
    arc 1 -> 2 [labels=(eps, 1), weight=one]
    final 2 [weight=one]

relation target_channel [algebra=markov, states=3]
    input Letters
    output Letters
    initial 0 [weight=one]
    arc 0 -> 1 [labels=(eps, 0), weight=shared]
    arc 0 -> 1 [labels=(eps, 0), weight=shared]
    arc 1 -> 2 [labels=(1, eps), weight=one]
    final 2 [weight=one]
"""


def _compile(source: str = _SOURCE) -> tuple[Compiler, QiecModule]:
    compiler = Compiler(parse(source), module_name=_MODULE, file_path="compose.qvr")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    return compiler, module


def _score(
    module: QiecModule,
    request: FixedComposedRelationRequest,
    parameters: dict[ParameterId, torch.Tensor],
) -> torch.Tensor:
    provider = FixedRelationRuntimeProvider.for_score("pair", request, parameters)
    result = run_named(
        module,
        "pair",
        runtime=RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), provider)
        ),
    )
    assert result.result_type == request.result_type
    assert isinstance(result.value, torch.Tensor)
    return result.value


def test_binary_composition_round_trips_in_source_and_executes_with_shared_weights():
    compiler, module = _compile()
    composed = compiler.composed_relation_specs["pair_channel"]
    assert tuple(compiler.relation_specs) == ("source_channel", "target_channel")
    assert composed.left == compiler.relation_specs["source_channel"]
    assert composed.right == compiler.relation_specs["target_channel"]
    assert composed.left != composed.right
    assert composed.left.weight_parameters == composed.right.weight_parameters
    assert any(
        binding.name == "pair_channel" and binding.kind == "relation"
        for binding in qiec_bindings(parse(_SOURCE))
    )
    emitted = module_to_source(parse(_SOURCE))
    assert "#! The two operands can be declared later." in emitted
    assert "relation pair_channel = compose(source_channel, target_channel)" in emitted
    emitted_compiler, emitted_module = _compile(emitted)
    assert emitted_compiler.composed_relation_specs == compiler.composed_relation_specs
    assert [handler.id for handler in emitted_module.handlers] == [
        handler.id for handler in module.handlers
    ]
    request = FixedComposedRelationRequest(
        relation=composed, words=((0,), (0,)), max_states=40, max_work=80
    )
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedComposedRelationRequest)
    assert decoded == request
    wire_module = loads(dumps(module))
    assert isinstance(wire_module, QiecModule)
    shared = nn.Parameter(torch.tensor(0.125, dtype=torch.float64))
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", "one"): shared.new_tensor(1.0),
        ParameterId.derive(_MODULE, "relation_weights", "shared"): shared,
    }
    for mode in ("lazy", "materialized"):
        total = _score(wire_module, decoded.with_(execution=mode), parameters)
        assert total.item() == 4 * 0.125**2
        (derivative,) = torch.autograd.grad(total, shared)
        assert derivative.item() == 8 * 0.125

    reversed_ = request.with_(
        relation=composed.with_(left=composed.right, right=composed.left)
    )
    assert reversed_.result_type == request.result_type == REAL
    with pytest.raises(ExecutionFailure, match="authored topology"):
        _score(wire_module, reversed_, parameters)
    dropped_arc = request.with_(
        relation=composed.with_(left=composed.left.with_(arcs=composed.left.arcs[1:]))
    )
    with pytest.raises(ExecutionFailure, match="authored topology"):
        _score(wire_module, dropped_arc, parameters)

    discharged = discharge_module(wire_module)
    for target in ("stan", "bugs", "jags"):
        obstacle = next(
            x
            for x in project_for_target(wire_module, discharged, target).obstacles
            if x.computation == "pair"
        )
        assert "__relation_score_pair" in obstacle.reason


def test_implicit_module_name_gives_the_same_weight_owners_as_lowering():
    compiler = Compiler(parse(_SOURCE), file_path="nested/derived.qvr")
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    relation = compiler.composed_relation_specs["pair_channel"]
    request = FixedComposedRelationRequest(
        relation=relation, words=((0,), (0,)), max_states=40, max_work=80
    )
    parameters = {
        ParameterId.derive(module.module, "relation_weights", "one"): torch.tensor(1.0),
        ParameterId.derive(module.module, "relation_weights", "shared"): torch.tensor(
            0.125
        ),
    }
    assert _score(module, request, parameters).item() == 4 * 0.125**2


def test_authored_logprob_composition_returns_log_weight_and_live_gradient():
    compiler, module = _compile(
        _SOURCE.replace("algebra=markov", "algebra=logprob").replace(
            "pair : Real", "pair : LogWeight"
        )
    )
    relation = compiler.composed_relation_specs["pair_channel"]
    request = FixedComposedRelationRequest(
        relation=relation, words=((0,), (0,)), max_states=40, max_work=80
    )
    assert request.result_type == LOG_WEIGHT
    shared = nn.Parameter(torch.tensor(-80.0, dtype=torch.float64))
    parameters = {
        ParameterId.derive(_MODULE, "relation_weights", "one"): shared.new_zeros(()),
        ParameterId.derive(_MODULE, "relation_weights", "shared"): shared,
    }
    total = _score(module, request, parameters)
    expected = 2 * shared.detach() + torch.log(shared.new_tensor(4.0))
    torch.testing.assert_close(total, expected)
    (derivative,) = torch.autograd.grad(total, shared)
    torch.testing.assert_close(derivative, shared.new_tensor(2.0))


@pytest.mark.parametrize(
    "old,new,diagnostic",
    (
        (
            "compose(source_channel, target_channel)",
            "compose(missing, target_channel)",
            "unknown authored relation 'missing'",
        ),
        (
            "compose(source_channel, target_channel)",
            "compose(pair_channel, target_channel)",
            "cyclic authored relation",
        ),
        (
            "relation source_channel [algebra=markov, states=3]\n    input Letters\n    output Letters",
            "relation source_channel [algebra=markov, states=3]\n    input Letters\n    output Other",
            "same nominal shared tape",
        ),
        (
            "relation target_channel [algebra=markov",
            "relation target_channel [algebra=logprob",
            "shared Markov or LogProb algebra",
        ),
        (
            "pair : Real [operation=pair_weight",
            "pair : Real [operation=path_entropy",
            "compositions currently support pair_weight",
        ),
        (
            "relation pair_channel = compose(source_channel, target_channel)",
            "relation source_channel = compose(source_channel, target_channel)",
            "duplicate authored relation",
        ),
        (
            "pair : Real",
            "pair : LogWeight",
            "pair weight requires Real",
        ),
    ),
)
def test_authored_binary_composition_refusals(old, new, diagnostic):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        _compile(_SOURCE.replace(old, new))
