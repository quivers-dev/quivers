"""Check encoded dual numbers against ordinary and extended-range oracles."""

from itertools import permutations
from math import expm1, log

import pytest
import torch

from quivers.core.signed_log import SignedLogDualSemiring
from quivers.qiec.contracts import Acyclic, NoRetention, OccurrenceId, ParameterId
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import AxiomOccurrence, AxiomSlot
from quivers.stochastic.planning import plan
from tests.evidence import bodies, graphs


ALGEBRA = SignedLogDualSemiring()
LIKE = torch.zeros((), dtype=torch.float64)


def test_signed_dual_product_keeps_tangent_at_zero_primal():
    x = LIKE.new_tensor([0.0, 2.0, -5.0])
    y = LIKE.new_tensor([5.0, -3.0, 0.0])
    dx, dy = LIKE.new_tensor([-3.0, 7.0, 2.0]), LIKE.new_tensor([1.0, -4.0, 2.0])
    value = ALGEBRA.tensor_op(ALGEBRA.from_linear(x, dx), ALGEBRA.from_linear(y, dy))
    primal, tangent = ALGEBRA.to_linear(value)
    reference = torch.func.jvp(lambda a, b: a * b, (x, y), (dx, dy))
    torch.testing.assert_close(primal, reference[0])
    torch.testing.assert_close(tangent, reference[1])
    assert value[0].tolist()[0] == 0 and tangent[0].item() == pytest.approx(-15.0)


def test_equal_opposing_enormous_terms_cancel_before_small_terms():
    primal = LIKE.new_tensor([[1, 1000], [1, 0], [-1, 1000]])
    tangent = LIKE.new_tensor([0, -torch.inf])
    for order in permutations(range(3)):
        result = ALGEBRA.join(ALGEBRA.pack(primal[list(order)], tangent), dim=0)
        assert result.tolist() == [1, 0, 0, -torch.inf]
    with pytest.raises(FloatingPointError, match="summand rounds away"):
        ALGEBRA.join(ALGEBRA.pack(primal[:2], tangent), dim=0)


def test_near_opposed_tiny_terms_and_empty_index_axes():
    values = ALGEBRA.pack(
        LIKE.new_tensor([[1, -1000], [-1, -1000.125]]), LIKE.new_tensor([0, -torch.inf])
    )
    actual = ALGEBRA.join(values.reshape(1, 2, 4), dim=(0, 1))
    torch.testing.assert_close(actual[1], LIKE.new_tensor(-1000 + log(-expm1(-0.125))))
    assert actual[0].item() == 1
    with pytest.raises(FloatingPointError, match="represented in linear"):
        ALGEBRA.to_linear(actual)
    empty = ALGEBRA.join(LIKE.new_empty((3, 0, 4)), dim=1)
    torch.testing.assert_close(empty, ALGEBRA.zero_like((3,), LIKE))
    assert ALGEBRA.contains(actual).item()
    assert not ALGEBRA.contains(LIKE.new_tensor([1, 0, 0, 0])).item()
    with pytest.raises(TypeError, match="explicit signed log tangent"):
        ALGEBRA.from_linear(LIKE.clone().requires_grad_(), LIKE)
    with pytest.raises(ValueError, match="canonical log magnitude"):
        ALGEBRA.pack(LIKE.new_tensor([0, 0]), LIKE.new_tensor([1, 0]))


@pytest.mark.parametrize("executor", ["auto", "eliminate"])
def test_native_plan_keeps_a_zero_weight_tangent_through_large_intermediates(executor):
    slots = tuple(
        AxiomSlot(
            id=ConstructorId.derive("signed-log-plan", name),
            name=name,
            telescope=tuple(IndexBinder(name=axis, sort=NAT) for axis in ("x", "y")),
            judgment=bodies.judgment(name, "x", "y"),
        )
        for name in ("r1", "r2", "r3")
    )
    presentation = bodies.PRESENTATION.with_(rules=(bodies.STAR,), axioms=slots)
    occurrences = tuple(
        AxiomOccurrence(
            slot=slot.id,
            occurrence=OccurrenceId.derive("signed-log-plan", slot.name),
            binding=(("x", 0), ("y", 0)),
        )
        for slot in slots
    )
    zero = [0, -torch.inf]
    primal = LIKE.new_tensor([zero, [1, 1000], [1, -1000]])
    tangent = LIKE.new_tensor([[1, 0], zero, zero])
    weights: dict[OccurrenceId | ParameterId, torch.Tensor] = dict(
        zip(
            (axiom.occurrence for axiom in occurrences),
            ALGEBRA.pack(primal, tangent),
            strict=True,
        )
    )
    computation = plan(
        graphs.query().with_(presentation=presentation.id),
        presentation,
        ALGEBRA,
        Acyclic(),
        graphs.contract(),
        NoRetention(),
        executor=executor,
    )
    (goal,) = bodies.items("hub", [(0,)])
    result = computation.run(occurrences, weights, like=LIKE).value(goal)
    torch.testing.assert_close(result, LIKE.new_tensor([0, -torch.inf, 1, 0]))
