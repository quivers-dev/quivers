"""Checked sparse unary operation scores and path multiplicity oracles."""

from fractions import Fraction
from math import log

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    CoreRuntimeProvider,
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    RuntimeConfiguration,
    UnaryRelationTransformSpec,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.qiec.test_relation_runtime import _request


def _execute(request, parameters):
    type_name = (
        "LogWeight" if request.relation.source.algebra == LOG_PROB.subject else "Real"
    )
    source = f"relation_score score : {type_name} [operation=pair_weight]\n"
    compiler = Compiler(parse(source), module_name="checked-unary-transform")
    compiler.compile()
    assert compiler.qiec_module is not None
    provider = FixedRelationRuntimeProvider.for_score("score", request, parameters)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )

    def evaluate() -> torch.Tensor:
        result = run_named(compiler.qiec_module, "score", runtime=runtime)
        assert isinstance(result.value, torch.Tensor)
        return result.value

    return evaluate


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
@pytest.mark.parametrize(
    "operation, positions, fixed_word, observed, a_count, b_count",
    (
        ("project", (0,), None, ((0,),), 2, 1),
        ("condition", (0,), (0,), ((0,),), 2, 0),
        ("invert", (), None, ((1,), (0,)), 0, 1),
        ("trim", (), None, ((0,), (0,)), 2, 0),
    ),
)
def test_checked_unary_transforms_keep_exact_mass_and_live_parameter_identity(
    algebra, operation, positions, fixed_word, observed, a_count, b_count
):
    base, parameters, (a_id, b_id, _) = _request("pair_weight", algebra)
    transform = UnaryRelationTransformSpec(
        source=base.relation, operation=operation, positions=positions, word=fixed_word
    )
    request = FixedTransformedRelationRequest(
        relation=transform, words=observed, max_states=32, max_work=48
    )
    decoded = loads(dumps(request))
    assert isinstance(decoded, FixedTransformedRelationRequest)
    assert decoded == request and dumps(decoded) == dumps(request)
    evaluate = _execute(decoded, parameters)
    actual = evaluate()
    a, b = Fraction(1, 8), Fraction(1, 4)
    expected = a_count * a + b_count * b
    value = actual if algebra is MARKOV else actual.exp()
    assert value.item() == pytest.approx(float(expected))
    gradients = torch.autograd.grad(value, (parameters[a_id], parameters[b_id]))
    expected_gradient = (
        (a_count, b_count) if algebra is MARKOV else (a_count * a, b_count * b)
    )
    assert [gradient.item() for gradient in gradients] == pytest.approx(
        [float(item) for item in expected_gradient]
    )
    with torch.no_grad():
        if algebra is MARKOV:
            parameters[a_id].mul_(2)
            parameters[b_id].mul_(2)
        else:
            parameters[a_id].add_(torch.log(actual.new_tensor(2)))
            parameters[b_id].add_(torch.log(actual.new_tensor(2)))
    updated = evaluate()
    assert (updated if algebra is MARKOV else updated.exp()).item() == pytest.approx(
        float(2 * expected)
    )


def test_checked_closure_reuses_the_supported_star_and_refuses_implicit_cycles():
    base, parameters, (a_id, _, _) = _request("pair_weight")
    zero_width = base.relation.with_(arcs=((0, 1, (None, None), 0),))
    spec = UnaryRelationTransformSpec(source=zero_width, operation="closure")
    request = FixedTransformedRelationRequest(
        relation=spec, words=((), ()), max_states=12, max_work=30, max_closure_size=4
    )
    value = _execute(request, parameters)()
    assert value.item() == pytest.approx(8 / 7)
    (gradient,) = torch.autograd.grad(value, parameters[a_id])
    assert gradient.item() == pytest.approx((8 / 7) ** 2)
    with pytest.raises(ProductiveEpsilonCycleError, match="productive"):
        request.with_(max_closure_size=None).plan(parameters)


def test_transform_types_and_work_bounds_are_explicit_in_the_wire_contract():
    base, parameters, _ = _request("pair_weight")
    with pytest.raises(dx.ValidationError, match="increasing distinct"):
        UnaryRelationTransformSpec(
            source=base.relation, operation="project", positions=(1, 0)
        )
    with pytest.raises(dx.ValidationError, match="invalid symbol"):
        UnaryRelationTransformSpec(
            source=base.relation, operation="condition", positions=(0,), word=(2,)
        )
    condition = UnaryRelationTransformSpec(
        source=base.relation, operation="condition", positions=(0,), word=(0,)
    )
    with pytest.raises(dx.ValidationError, match="one word per retained tape"):
        FixedTransformedRelationRequest(
            relation=condition, words=base.words, max_states=20, max_work=20
        )
    request = FixedTransformedRelationRequest(
        relation=condition, words=((0,),), max_states=3, max_work=0
    )
    with pytest.raises(
        RelationLimitError, match="conditioned relation exceeds max_work"
    ):
        request.plan(parameters)


def test_projected_log_score_keeps_underflowed_parallel_alignments_and_zero_support():
    base, parameters, (a_id, b_id, _) = _request("pair_weight", LOG_PROB)
    with torch.no_grad():
        parameters[a_id].fill_(-1000.0)
        parameters[b_id].fill_(-torch.inf)
    spec = UnaryRelationTransformSpec(
        source=base.relation, operation="project", positions=(0,)
    )
    request = FixedTransformedRelationRequest(
        relation=spec,
        words=((0,),),
        max_states=12,
        max_work=30,
    )
    value = _execute(request, parameters)()
    assert value.item() == pytest.approx(-1000 + log(2), abs=1e-12, rel=0)
    gradients = torch.autograd.grad(value, (parameters[a_id], parameters[b_id]))
    assert [value.item() for value in gradients] == pytest.approx([1, 0])


def test_transform_score_declaration_refuses_native_foreign_capability():
    compiler = Compiler(
        parse("relation_score weighted : Real [operation=pair_weight]"),
        module_name="transformed_score",
    )
    compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        obstacle = next(
            x
            for x in project_for_target(module, discharged, target).obstacles
            if x.computation == "weighted"
        )
        assert "__relation_score_weighted" in obstacle.reason
