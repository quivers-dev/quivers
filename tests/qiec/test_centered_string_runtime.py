"""Centered string scores through ordinary authored and serialized QVR."""

from hashlib import sha256
from math import inf, log

import didactic.api as dx
import pytest
import torch

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    CenteredStringLogProbRequest,
    CenteredStringLogProbRuntimeProvider,
    ExecutionFailure,
    FixedRelationRequest,
    QiecModule,
    RuntimeConfiguration,
    TreeStringGraphRequest,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.test_centered_string_log_scores import _oracle

_MODULE = "centered_log"
_SOURCE = """
object Letters : FinSet 2

relation channel [algebra=logprob, states=5]
    input Letters
    output Letters
    initial 0 [weight=unit]
    arc 0 -> 4 [labels=(0, 0), weight=shared]
    arc 0 -> 4 [labels=(0, 0), weight=shared]
    arc 0 -> 1 [labels=(0, eps), weight=shared]
    arc 1 -> 2 [labels=(eps, eps), weight=shared]
    arc 2 -> 3 [labels=(eps, eps), weight=opposite]
    arc 3 -> 4 [labels=(eps, 1), weight=bias]
    arc 0 -> 4 [labels=(0, 1), weight=optional]
    final 4 [weight=unit]

string_graph model
    variable input_word : Letters [observed=(0)]
    variable output_word : Letters
    factor channel(input_word, output_word)

relation_score conditional : LogWeight [operation=conditional_log_prob, relation=channel]
string_graph_score posterior : LogWeight [operation=tree_log_prob, graph=model]
"""
_LIMITS = {"max_states": 50, "max_work": 100}


def _compiler(source):
    compiler = Compiler(parse(source), module_name=_MODULE)
    compiler.compile()
    return compiler


def _fixture(mode):
    compiler = _compiler(_SOURCE)
    channel = compiler.relation_specs["channel"]
    fixed = FixedRelationRequest(
        relation=channel,
        operation="conditional_log_prob",
        words=((0,), (0,)),
        **_LIMITS,
    )
    graph = TreeStringGraphRequest(
        graph=compiler.string_graph_specs["model"],
        variable="output_word",
        word=(0,),
        **_LIMITS,
    )
    old_wires = tuple(
        sha256(dumps(item).encode()).hexdigest() for item in (fixed, graph)
    )
    assert old_wires == (
        "eb9e2a244439a3563b4609d52b1fcbc37d59eed6f212e338eec95094a414be0a",
        "6ae20d875e723ec0ddafb5c143a53225c4486edaa6fba08ca9d51dc42958348b",
    )
    request = CenteredStringLogProbRequest(
        observation=fixed if mode == "conditional" else graph,
        max_center_occurrences=100,
        max_dyadic_work=700,
    )
    weights = tuple(
        torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        for value in (0, -1e20, 1e20, 0, -inf)
    )
    assert channel.weight_names == ("unit", "shared", "opposite", "bias", "optional")
    parameters = dict(zip(channel.weight_parameters, weights, strict=True))
    return request, parameters, weights


def _native(request):
    report = "(LogWeight * Tensor[Int]([4]))"
    handler = request.handler_name
    source = (
        _SOURCE
        + f"""
instance centered : Compute[Unit, {report}]
handler {handler} for Compute[Unit, {report}] : {report} -> {report} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define likelihood : {report} !{{}} =
    handle centered with {handler} in
        let result <- perform centered.apply(unit)
        return result
"""
    )
    compiler = _compiler(module_to_source(parse(source)))
    module = loads(dumps(compiler.qiec_module))
    assert isinstance(module, QiecModule)
    old_names = {item.name: str(item.id) for item in module.handlers}
    assert old_names["__relation_score_conditional"] == (
        "qiec:handler:47a76f8767a6a902e33644d96d8321aa8900b72b1239fcf3f12bb674e5af5cac"
    )
    assert old_names["__string_graph_score_posterior"] == (
        "qiec:handler:b89b9ff53995576f0f1975ba334eecb82f60a1c1083edea8ce5601966f1fe819"
    )
    return module


def _run(module, provider):
    return run_named(
        module, "likelihood", runtime=RuntimeConfiguration(providers=(provider,))
    )


@pytest.mark.parametrize("mode", ("conditional", "tree"))
def test_native_centered_reports_agree_with_the_independent_dyadic_oracle(mode):
    source_request, parameters, weights = _fixture(mode)
    wire = dumps(source_request)
    request = loads(wire)
    assert request == source_request and dumps(request) == wire
    module = _native(request)
    provider = CenteredStringLogProbRuntimeProvider(request, parameters)
    for bias, optional in ((0, -inf), (log(2), -1e20)):
        with torch.no_grad():
            weights[3].fill_(bias)
            weights[4].fill_(optional)
        expected, expected_gradient = _oracle(torch.stack(weights), (0,))
        actual = _run(module, provider)
        assert actual.result_type == request.result_type
        value, work = actual.value
        assert value.item() == pytest.approx(expected, abs=1e-15)
        assert [g.item() for g in torch.autograd.grad(value, weights)] == pytest.approx(
            expected_gradient, abs=1e-15
        )
        assert work.tolist()[:3] == (
            [7, 13, 5] if mode == "conditional" else [14, 20, 7]
        )
        # The tree join gives the newly positive arc a continuation.
        expected_visits = {"conditional": (80, 82), "tree": (126, 132)}
        assert work.tolist()[3] == expected_visits[mode][optional != -inf]
    observation = request.observation
    changed = (
        observation.with_(words=((0,), (1,)))
        if isinstance(observation, FixedRelationRequest)
        else observation.with_(word=(1,))
    )
    if isinstance(observation, FixedRelationRequest):
        relation = observation.relation
        different_model = observation.with_(
            relation=relation.with_(arcs=relation.arcs[1:])
        )
    else:
        name, relation = observation.graph.relations[0]
        different_model = observation.with_(
            graph=observation.graph.with_(
                relations=((name, relation.with_(arcs=relation.arcs[1:])),)
            )
        )
    for mutation in (changed, different_model):
        substitute = request.with_(observation=mutation)
        assert substitute.result_type == request.result_type
        assert substitute.handler_name != request.handler_name
        with pytest.raises(ExecutionFailure, match="centered_string_log_prob"):
            _run(module, CenteredStringLogProbRuntimeProvider(substitute, parameters))

    bounded = request.with_(max_center_occurrences=1)
    assert bounded.handler_name == request.handler_name
    with pytest.raises(RelationLimitError, match="max_center_occurrences"):
        CenteredStringLogProbRuntimeProvider(bounded, parameters)
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        obstacles = project_for_target(module, discharged, target).obstacles
        assert any(
            item.computation == "likelihood" and request.handler_name in item.reason
            for item in obstacles
        )


def test_native_centered_request_refuses_other_objectives_and_nonbinary64_input():
    request, parameters, _ = _fixture("conditional")
    observed = request.observation
    for changed in (
        observed.with_(operation="pair_weight"),
        observed.with_(max_closure_size=1),
    ):
        with pytest.raises(dx.ValidationError, match="acyclic LogProb"):
            request.with_(observation=changed)
    for setting in ("max_center_occurrences", "max_dyadic_work"):
        with pytest.raises(dx.ValidationError, match=setting):
            request.with_(**{setting: 2**63})
    unsupported = {
        owner: value.detach().to(torch.float32) for owner, value in parameters.items()
    }
    provider = CenteredStringLogProbRuntimeProvider(request, unsupported)
    with pytest.raises(ExecutionFailure, match="binary64"):
        _run(_native(request), provider)


@pytest.mark.parametrize("mode", ("conditional", "tree"))
def test_native_centered_impossible_word_stays_log_zero(mode):
    request, parameters, _ = _fixture(mode)
    observation = request.observation
    impossible = (
        observation.with_(words=((0,), ()))
        if isinstance(observation, FixedRelationRequest)
        else observation.with_(word=())
    )
    request = request.with_(observation=impossible)
    value, work = _run(
        _native(request), CenteredStringLogProbRuntimeProvider(request, parameters)
    ).value
    assert value.item() == -inf and not value.requires_grad
    assert work.tolist()[3] > 0
