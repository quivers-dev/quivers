"""Checked fixed-word request and live provider tests."""

from fractions import Fraction
from hashlib import sha256
from math import log
from typing import cast

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV, Semiring
from quivers.core.relations import Arc, Boundary, RelationTopology
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    LOG_WEIGHT,
    REAL,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationCostRequest,
    FixedRelationRequest,
    FixedRelationRuntimeProvider,
    ParameterId,
    RationalRelationSpec,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.qiec.numbers import host_value
from quivers.transpile import project_for_target
from tests.test_relation_composition import _INPUT, _OUTPUT


def _module(result):
    text = f"""
handler channel for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply(argument : Unit) resumes 1

define evaluate(argument : Unit) : {result} !{{}} =
    with instance relation : Compute[Unit, {result}] in
        handle relation with channel in
            let value <- perform relation.apply(argument)
            return value

define signed() : Real !{{}} =
    return -2.0
"""
    compiler = Compiler(
        parse(text),
        module_name="relation_runtime_test",
        file_path="relation_runtime_test.qvr",
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _request(operation, algebra: Semiring = MARKOV):
    names = ("duplicate_a", "tied_a", "other_b", "unit")
    ids = tuple(
        ParameterId.derive("relation-runtime", name) for name in ("a", "b", "unit")
    )
    a_id, b_id, unit_id = ids
    topology = RelationTopology(
        (_INPUT, _OUTPUT),
        1,
        2,
        (Arc(0, 1, (0, 0), 0), Arc(0, 1, (0, 0), 1), Arc(0, 1, (0, 1), 2)),
        (Boundary(0, 3),),
        (Boundary(1, 3),),
        names,
    )
    spec = RationalRelationSpec.of(
        topology, algebra=algebra, weight_parameters=(a_id, a_id, b_id, unit_id)
    )
    values = torch.tensor([0.125, 0.25, 1.0], dtype=torch.float64)
    if algebra is LOG_PROB:
        values = values.log()
    a, b, one = values.unbind()
    bound = {a_id: nn.Parameter(a), b_id: nn.Parameter(b), unit_id: one}
    return (
        FixedRelationRequest(
            relation=spec,
            operation=operation,
            words=((0,), (0,)),
            max_states=16,
            max_work=16,
        ),
        bound,
        ids,
    )


@pytest.mark.parametrize(
    "operation, algebra, type_name, scalar_type",
    (
        ("pair_weight", MARKOV, "Real", REAL),
        ("pair_weight", LOG_PROB, "LogWeight", LOG_WEIGHT),
        ("conditional_log_prob", MARKOV, "LogWeight", LOG_WEIGHT),
        ("conditional_log_prob", LOG_PROB, "LogWeight", LOG_WEIGHT),
    ),
)
def test_serialized_fixed_relation_request_executes_in_checked_runtime(
    operation, algebra, type_name, scalar_type
):
    request, parameters, (a_id, b_id, _) = _request(operation, algebra)
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationRequest) and decoded == request
    assert decoded.result_type == scalar_type
    runtime = RuntimeConfiguration(
        selections=(),
        providers=(
            CoreRuntimeProvider(),
            FixedRelationRuntimeProvider(decoded, parameters, handler_name="channel"),
        ),
    )
    module = _module(type_name)
    result = run_named(module, "evaluate", (None,), runtime=runtime)
    assert result.result_type == scalar_type
    value = result.value
    assert isinstance(value, torch.Tensor)
    mass = value if scalar_type == REAL else value.exp()
    assert mass.item() == pytest.approx(
        Fraction(1, 4) if operation == "pair_weight" else Fraction(1, 2)
    )
    if operation == "pair_weight":
        expected = [2.0, 0.0] if algebra is MARKOV else [1.0, 0.0]
    else:
        expected = [4.0, -2.0] if algebra is MARKOV else [0.5, -0.5]
    derivative = torch.autograd.grad(value, (parameters[a_id], parameters[b_id]))
    assert [grad.item() for grad in derivative] == pytest.approx(expected)
    with torch.no_grad():
        new_value = torch.tensor(0.25, dtype=torch.float64)
        parameters[a_id].copy_(new_value if algebra is MARKOV else new_value.log())
    updated = cast(
        torch.Tensor, run_named(module, "evaluate", (None,), runtime=runtime).value
    )
    actual = updated.item() if scalar_type == REAL else updated.exp().item()
    assert actual == pytest.approx(0.5 if operation == "pair_weight" else 2 / 3)
    assert host_value(run_named(module, "signed", runtime=runtime).value) == -2.0


def test_fixed_relation_provider_rejects_mismatched_check_and_native_targets():
    request, parameters, _ = _request("pair_weight")
    provider = FixedRelationRuntimeProvider(request, parameters, handler_name="channel")
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    with pytest.raises(ExecutionFailure, match="incompatible fixed relation handler"):
        run_named(_module("LogWeight"), "evaluate", (None,), runtime=runtime)
    module = _module("Real")
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        projected = project_for_target(module, discharged, target)
        assert [(space.computation, space.reason) for space in projected.obstacles] == [
            (
                "evaluate",
                "installs the foreign handler `channel`, whose clause `apply` a runtime provider implements",
            )
        ]


def test_fixed_relation_request_checks_word_types_and_limits_before_binding():
    request, parameters, _ = _request("pair_weight")
    with pytest.raises(dx.ValidationError, match="invalid symbol"):
        request.with_(words=((2,), (0,)))
    with pytest.raises(dx.ValidationError, match="max_states"):
        request.with_(max_states=-1)
    depleted = request.with_(max_states=0)
    with pytest.raises(ValueError, match="max_states"):
        FixedRelationRuntimeProvider(depleted, parameters, handler_name="channel")


def _ambiguous_request(algebra):
    request, parameters, ids = _request("pair_weight", algebra)
    old = request.relation
    relation = old.with_(arcs=(*old.arcs[:2], (0, 1, (0, 0), 2)))
    return request.with_(relation=relation), parameters, ids


@pytest.mark.parametrize(
    "operation, algebra, original_sha",
    (
        (
            "pair_weight",
            MARKOV,
            "f539f1d825bce9d348daad5d06ae30ad2699fffed41ae6d0fdca7972743e231f",
        ),
        (
            "pair_weight",
            LOG_PROB,
            "58b4dff5098f8ed0650ea073953008d5be5b7b7b8bdf9939fd8f9be104f0968a",
        ),
        (
            "conditional_log_prob",
            MARKOV,
            "1e185248bcc86239b5f15feb83cd479eac3934b70e3c1f48672da1621068c46c",
        ),
        (
            "conditional_log_prob",
            LOG_PROB,
            "bc0a1302bd4fac3d9b71d639e8fad1204ce64e13f9e31cfbf80ed3906197dd46",
        ),
    ),
)
def test_pair_and_conditional_requests_keep_their_original_wire_bytes(
    operation, algebra, original_sha
):
    request, _, _ = _request(operation, algebra)
    text = dumps(request)
    assert sha256(text.encode("utf-8")).hexdigest() == original_sha
    assert loads(text) == request


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_checked_path_entropy_is_real_and_preserves_tied_gradients(algebra):
    request, parameters, (a_id, b_id, _) = _ambiguous_request(algebra)
    request = request.with_(operation="path_entropy")
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationRequest) and decoded == request
    assert decoded.result_type == REAL
    runtime = RuntimeConfiguration(
        selections=(),
        providers=(
            CoreRuntimeProvider(),
            FixedRelationRuntimeProvider(decoded, parameters, handler_name="channel"),
        ),
    )
    module = _module("Real")
    outcome = run_named(module, "evaluate", (None,), runtime=runtime)
    assert outcome.result_type == REAL
    assert isinstance(outcome.value, torch.Tensor)
    assert outcome.value.item() == pytest.approx(1.5 * log(2))
    actual = torch.autograd.grad(outcome.value, (parameters[a_id], parameters[b_id]))
    expected = (2 * log(2), -log(2)) if algebra is MARKOV else (log(2) / 4, -log(2) / 4)
    assert tuple(derivative.item() for derivative in actual) == pytest.approx(expected)
    with torch.no_grad():
        parameters[b_id].copy_(parameters[a_id])
    updated = run_named(module, "evaluate", (None,), runtime=runtime)
    assert isinstance(updated.value, torch.Tensor)
    assert updated.value.item() == pytest.approx(log(3))
    with pytest.raises(ExecutionFailure, match="incompatible fixed relation handler"):
        run_named(_module("LogWeight"), "evaluate", (None,), runtime=runtime)


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_signed_additive_path_cost_round_trip_and_original_gradients(algebra):
    base, parameters, (a_id, b_id, _) = _ambiguous_request(algebra)
    request = FixedRelationCostRequest(
        relation=base.relation,
        words=base.words,
        max_states=base.max_states,
        max_work=base.max_work,
        initial_costs=(-1.0,),
        arc_costs=(-7.0, 1.0, -2.0),
        final_costs=(0.5,),
    )
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationCostRequest) and decoded == request
    assert decoded.operation == "expected_path_cost"
    assert decoded.result_type == REAL
    runtime = RuntimeConfiguration(
        selections=(),
        providers=(
            CoreRuntimeProvider(),
            FixedRelationRuntimeProvider(decoded, parameters, handler_name="channel"),
        ),
    )
    module = _module("Real")
    result = run_named(module, "evaluate", (None,), runtime=runtime)
    assert result.result_type == REAL
    assert isinstance(result.value, torch.Tensor)
    assert result.value.item() == pytest.approx(-3.0)
    gradients = torch.autograd.grad(result.value, (parameters[a_id], parameters[b_id]))
    expected = [-2.0, 1.0] if algebra is MARKOV else [-0.25, 0.25]
    assert [grad.item() for grad in gradients] == pytest.approx(expected)
    with torch.no_grad():
        parameters[b_id].copy_(parameters[a_id])
    updated = run_named(module, "evaluate", (None,), runtime=runtime)
    assert isinstance(updated.value, torch.Tensor)
    assert updated.value.item() == pytest.approx(-8 / 3 - 0.5)


@pytest.mark.parametrize(
    "arc_costs, final_costs, diagnostic",
    (
        (None, None, "at least one cost group"),
        ((0.0, 1.0), None, "one finite value"),
        (None, (float("inf"),), "one finite value"),
    ),
)
def test_expected_cost_validates_each_original_occurrence_group(
    arc_costs, final_costs, diagnostic
):
    base, _, _ = _request("pair_weight")
    with pytest.raises(dx.ValidationError, match=diagnostic):
        FixedRelationCostRequest(
            relation=base.relation,
            words=base.words,
            max_states=base.max_states,
            max_work=base.max_work,
            arc_costs=arc_costs,
            final_costs=final_costs,
        )


def test_serialized_shared_cyclic_cost_reuses_checked_closure():
    base, parameters, (loop_id, _, _) = _request("pair_weight")
    relation = base.relation.with_(
        n_states=1,
        arcs=((0, 0, (None, None), 0), (0, 0, (None, None), 1)),
        final=((0, 3),),
    )
    request = FixedRelationCostRequest(
        relation=relation,
        words=((), ()),
        max_states=16,
        max_work=16,
        max_closure_size=2,
        arc_costs=(-2.0, -4.0),
    )
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedRelationCostRequest)
    provider = FixedRelationRuntimeProvider(decoded, parameters, handler_name="channel")
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    result = run_named(_module("Real"), "evaluate", (None,), runtime=runtime)
    assert isinstance(result.value, torch.Tensor)
    assert result.value.item() == pytest.approx(-1.0)
    (gradient,) = torch.autograd.grad(result.value, parameters[loop_id])
    assert gradient.item() == pytest.approx(-32 / 3)
    with pytest.raises(ValueError, match="max_closure_size"):
        FixedRelationRuntimeProvider(
            decoded.with_(max_closure_size=1), parameters, handler_name="channel"
        )
