"""Checked binary sparse operation semantics and operand ownership."""

from fractions import Fraction
from hashlib import sha256

import didactic.api as dx
import pytest
import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import RelationLimitError
from quivers.qiec import (
    BinaryRelationTransformSpec,
    FixedTransformedRelationRequest,
    ParameterId,
    UnaryRelationTransformSpec,
    dumps,
    loads,
)
from tests.qiec.test_relation_runtime import _request
from tests.qiec.test_relation_transforms import _execute


def _operands(operation, base):
    left = base.relation
    right = left
    options = {}
    if operation == "join":
        right = right.with_(tapes=(left.tapes[1], ("Target", 2)))
        options = {"shared_tapes": (1, 0), "keep_shared": True}
    elif operation == "intersect":
        left = left.with_(
            tapes=left.tapes[:1],
            arcs=tuple(
                (source, target, (labels[0],), slot)
                for source, target, labels, slot in left.arcs[:2]
            ),
        )
        right = left
    return left, right, options


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
@pytest.mark.parametrize(
    "operation, words, expected, derivative",
    (
        ("union", ((0,), (0,)), Fraction(1, 2), Fraction(4)),
        ("concatenate", ((0, 0), (0, 0)), Fraction(1, 16), Fraction(1)),
        ("intersect", ((0,),), Fraction(1, 16), Fraction(1)),
        ("join", ((0,), (0,), (0,)), Fraction(1, 16), Fraction(1)),
    ),
)
def test_checked_binary_recipes_keep_multiplicities_and_parameter_owners(
    algebra, operation, words, expected, derivative
):
    base, parameters, (a_id, b_id, _) = _request("pair_weight", algebra)
    left, right, options = _operands(operation, base)
    spec = BinaryRelationTransformSpec(
        left=left, right=right, operation=operation, **options
    )
    request = FixedTransformedRelationRequest(
        relation=spec, words=words, max_states=40, max_work=80
    )
    restored = loads(dumps(request))
    assert isinstance(restored, FixedTransformedRelationRequest) and restored == request
    observed = _execute(restored, parameters)()
    linear = observed if algebra is MARKOV else observed.exp()
    assert linear.item() == pytest.approx(float(expected))
    gradients = torch.autograd.grad(
        linear, (parameters[a_id], parameters[b_id]), allow_unused=True
    )
    if algebra is LOG_PROB:
        derivative *= Fraction(1, 8)
    assert gradients[0] is not None and gradients[0].item() == pytest.approx(
        float(derivative)
    )
    assert gradients[1] is None or gradients[1].item() == 0.0


def test_join_rejects_same_size_nominal_mismatch_distinct_id_alias_and_work_limit():
    base, parameters, (a_id, _, _) = _request("pair_weight")
    left, right, options = _operands("join", base)
    wrong = right.with_(tapes=(("Names", 2), right.tapes[1]))
    with pytest.raises(dx.ValidationError, match="same nominal shared tape"):
        BinaryRelationTransformSpec(left=left, right=wrong, operation="join", **options)
    with pytest.raises(dx.ValidationError, match="identical ordered tape types"):
        BinaryRelationTransformSpec(left=left, right=right, operation="union")
    spec = BinaryRelationTransformSpec(
        left=left, right=right, operation="join", **options
    )
    request = FixedTransformedRelationRequest(
        relation=spec, words=((0,), (0,), (0,)), max_states=20, max_work=0
    )
    with pytest.raises(RelationLimitError, match="max_work"):
        request.plan(parameters)

    alias_id = ParameterId.derive("binary transform", "alias")
    alias = right.with_(
        weight_parameters=tuple(
            alias_id if item == a_id else item for item in right.weight_parameters
        )
    )
    conflicting = spec.with_(right=alias)
    with pytest.raises(ValueError, match="same parameter identity"):
        conflicting.bind(
            parameters | {alias_id: parameters[a_id]}, max_states=30, max_work=60
        )


@pytest.mark.parametrize(
    "algebra, original_sha",
    (
        (MARKOV, "33e12b515cfdd00881e9c7f52c695cb30461014fc30c4814a939cdab85749d3b"),
        (LOG_PROB, "a3e8c270dc2a7149f29033c996c0aa7467cc1ad97038b41e4d0f3d4218123841"),
    ),
)
def test_unary_request_wire_bytes_are_preserved(algebra, original_sha):
    base, _, _ = _request("pair_weight", algebra)
    spec = UnaryRelationTransformSpec(
        source=base.relation, operation="project", positions=(0,)
    )
    request = FixedTransformedRelationRequest(
        relation=spec, words=((0,),), max_states=32, max_work=48
    )
    encoded = dumps(request)
    assert sha256(encoded.encode()).hexdigest() == original_sha
    assert loads(encoded) == request
