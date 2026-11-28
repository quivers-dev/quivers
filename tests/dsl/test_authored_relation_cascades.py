"""Authored sparse cascades retain nominal dependencies and live parameters."""

from fractions import Fraction

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import DivergentClosureError
from quivers.core.relations import ProductiveEpsilonCycleError, RelationLimitError
from quivers.dsl import module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec import (
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    ParameterId,
    RelationCascadeSpec,
    RelationCascadeStep,
    RuntimeConfiguration,
    dumps,
    loads,
    run_named,
)
from quivers.qiec.relations import _RelationWeights
from tests.dsl.test_authored_relation_transforms import (
    _MODULE,
    _SOURCE,
    _compile,
    _parameters,
)

_CASCADE_DECLARATIONS = """relation cascade = compose(reverse, pair)
relation reverse = invert(duplicate)
relation duplicate = union(pair, pair)
relation pair = compose(base, link)
relation_score cascade_weight : Real [operation=pair_weight, relation=cascade]
"""
_CASCADE = _SOURCE + "\n" + _CASCADE_DECLARATIONS
_WORDS = ((0,), (0,))


def test_shared_authored_cascade_support_gradient_and_parameter_mutation(monkeypatch):
    compiler, module = _compile(_CASCADE)
    spec = compiler.transformed_relation_specs["cascade"]
    assert isinstance(spec, RelationCascadeSpec)
    assert tuple(step.name for step in spec.steps) == (
        "base",
        "link",
        "pair",
        "duplicate",
        "reverse",
        "cascade",
    )
    assert spec.steps[3].operands == (2, 2)
    assert spec.steps[-1].operands == (4, 2)
    assert compiler.composed_relation_specs["pair"].left == spec.steps[0].primitive
    assert (
        spec
        == _compile(module_to_source(parse(_CASCADE)))[0].transformed_relation_specs[
            "cascade"
        ]
    )
    assert (
        spec
        == _compile(_CASCADE_DECLARATIONS + "\n" + _SOURCE)[
            0
        ].transformed_relation_specs["cascade"]
    )

    request = FixedTransformedRelationRequest(
        relation=spec, words=_WORDS, max_work=240, max_states=120
    )
    request = loads(dumps(request))
    assert isinstance(request, FixedTransformedRelationRequest)
    assert request.relation == spec
    params = _parameters()
    original = _RelationWeights.forward
    source_reads = []

    def read(source):
        source_reads.append(id(source))
        return original(source)

    monkeypatch.setattr(_RelationWeights, "forward", read)
    provider = FixedRelationRuntimeProvider.for_score("cascade_weight", request, params)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    owner = params[ParameterId.derive(_MODULE, "relation_weights", "shared")]
    for p in (Fraction(1, 8), Fraction(1, 4)):
        with torch.no_grad():
            owner.fill_(float(p))
        source_reads.clear()
        result = run_named(module, "cascade_weight", runtime=runtime).value
        # Two distinct base paths pair with one link path. Union supplies
        # another path discriminator on the left of the final composition.
        # This gives eight compatible original path pairs of mass p**4 each.
        assert Fraction(result.item()) == 8 * p**4
        (gradient,) = torch.autograd.grad(result, owner, create_graph=True)
        (hessian,) = torch.autograd.grad(gradient, owner)
        assert Fraction(gradient.item()) == 32 * p**3
        assert Fraction(hessian.item()) == 96 * p**2
        assert len(source_reads) == len(set(source_reads)) == 2

    changed = spec.with_(
        steps=(*spec.steps[:3], spec.steps[3].with_(name="alias"), *spec.steps[4:])
    )
    request = request.with_(relation=changed)
    substitution = FixedRelationRuntimeProvider.for_score(
        "cascade_weight", request, params
    )
    with pytest.raises(ExecutionFailure, match="authored topology"):
        run_named(
            module,
            "cascade_weight",
            runtime=RuntimeConfiguration(
                selections=(), providers=(CoreRuntimeProvider(), substitution)
            ),
        )


def test_authored_cascade_rejects_source_and_wire_cycles_and_nominal_mismatch():
    with pytest.raises(QiecDiagnosticError, match="cyclic authored relation"):
        _compile(_CASCADE.replace("invert(duplicate)", "invert(cascade)"))
    with pytest.raises(QiecDiagnosticError, match="same nominal shared tape"):
        # Pair's output type differs from its own input type despite both
        # alphabets having cardinality two.
        _compile(_CASCADE.replace("compose(reverse, pair)", "compose(pair, pair)"))

    spec = _compile(_CASCADE)[0].transformed_relation_specs["cascade"]
    assert isinstance(spec, RelationCascadeSpec)
    with pytest.raises(dx.ValidationError, match="only earlier steps"):
        spec.with_(
            steps=(
                spec.steps[0],
                spec.steps[1].with_(primitive=None, operation="trim", operands=(2,)),
                *spec.steps[2:],
            )
        )
    with pytest.raises(dx.ValidationError, match="unused step"):
        spec.with_(
            steps=(
                *spec.steps,
                RelationCascadeStep(name="other", primitive=spec.steps[0].primitive),
            )
        )


def test_cascade_degree_and_storage_limits_are_enforced_before_eager_work(monkeypatch):
    primitive = _compile()[0].relation_specs["base"]
    chain = (
        RelationCascadeStep(name="base", primitive=primitive),
        *(
            RelationCascadeStep(name=f"n{index}", operation="invert", operands=(index,))
            for index in range(20)
        ),
    )
    spec = RelationCascadeSpec(steps=chain)
    # Empty bindings would fail if the step limit were checked after binding.
    with pytest.raises(RelationLimitError, match="step count exceeds max_work"):
        spec.bind({}, max_states=100, max_work=20)

    expand = RelationCascadeSpec(
        steps=(
            chain[0],
            RelationCascadeStep(name="sum", operation="union", operands=(0, 0)),
        )
    )

    def should_not_construct(*args, **kwargs):
        pytest.fail("oversized sparse union reached its allocator")

    monkeypatch.setattr(
        "quivers.qiec.relation_cascades._apply_binary", should_not_construct
    )
    with pytest.raises(RelationLimitError, match="topology exceeds max_states"):
        expand.bind(_parameters(), max_states=3, max_work=100)
    with pytest.raises(RelationLimitError, match="topology exceeds max_work records"):
        expand.bind(_parameters(), max_states=100, max_work=20)


def test_cascade_distinct_parameter_ids_cannot_alias_across_named_leaves():
    spec = _compile(_CASCADE)[0].transformed_relation_specs["cascade"]
    assert isinstance(spec, RelationCascadeSpec)
    right = spec.steps[1]
    assert right.primitive is not None
    shared = ParameterId.derive(_MODULE, "relation_weights", "shared")
    alias = ParameterId.derive(_MODULE, "relation_weights", "alias")
    slots = tuple(
        alias if owner == shared else owner
        for owner in right.primitive.weight_parameters
    )
    changed = spec.with_(
        steps=(
            spec.steps[0],
            right.with_(primitive=right.primitive.with_(weight_parameters=slots)),
            *spec.steps[2:],
        )
    )
    params = _parameters()
    with pytest.raises(ValueError, match="same parameter identity"):
        changed.bind(params | {alias: params[shared]}, max_states=80, max_work=180)


def test_cascade_state_cycles_follow_native_finite_closure_contract():
    source = """object Alphabet : FinSet 2
relation answer = trim(fixed)
relation fixed = condition(repeater, 0, ())
relation_score answer_weight : Real [operation=pair_weight, relation=answer]

relation repeater [algebra=markov, states=1]
    input Alphabet
    output Alphabet
    initial 0 [weight=one]
    arc 0 -> 0 [labels=(eps, eps), weight=shared]
    arc 0 -> 0 [labels=(eps, eps), weight=shared]
    final 0 [weight=one]
"""
    spec = _compile(source)[0].transformed_relation_specs["answer"]
    request = FixedTransformedRelationRequest(
        relation=spec, words=((),), max_states=80, max_work=180
    )
    params = _parameters()
    with pytest.raises(ProductiveEpsilonCycleError, match="productive"):
        request.plan(params)
    plan = request.with_(max_closure_size=8).plan(params)
    result = plan.weight()
    owner = params[ParameterId.derive(_MODULE, "relation_weights", "shared")]
    (gradient,) = torch.autograd.grad(result, owner)
    q = 1 - 2 * Fraction(1, 8)
    assert result.item() == pytest.approx(float(1 / q))
    assert gradient.item() == pytest.approx(float(2 / q**2))
    with torch.no_grad():
        owner.fill_(0.5)
    with pytest.raises(DivergentClosureError):
        plan.weight()
