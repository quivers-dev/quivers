"""Category weights are checked before normalization can hide invalid signs."""

import math
import random

import pytest
import torch

from quivers.continuous._closed_probability import _ValidatedCategorical
from quivers.continuous.family_spec import FAMILY_REGISTRY
from quivers.qiec.distributions import ReferenceBackend
from quivers.qiec.reference_support import DistributionError
from quivers.transpile.family_meta import FAMILY_META


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, -2.0, -3.0],
        [-0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0],
        [math.nan, 1.0, 2.0],
        [math.inf, 1.0, 2.0],
        [],
    ],
)
@pytest.mark.parametrize("validate_args", [False, True])
def test_native_rejects_raw_invalid_weights_even_when_validation_is_disabled(
    weights, validate_args
):
    with pytest.raises(ValueError, match="Categorical probs"):
        FAMILY_META["Categorical"].distribution_class(
            probs=torch.tensor(weights), validate_args=validate_args
        )


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, -2.0, -3.0],
        [-0.1, 0.2, 0.3],
        [0.0, 0.0, 0.0],
        [math.nan, 1.0, 2.0],
        [math.inf, 1.0, 2.0],
        [],
    ],
)
def test_reference_rejects_invalid_weights_before_density_or_sampling(weights):
    backend = ReferenceBackend()
    with pytest.raises(DistributionError, match="probs"):
        backend.log_prob("Categorical", {"probs": weights}, 0)
    with pytest.raises(DistributionError, match="probs"):
        backend.sample("Categorical", {"probs": weights}, random.Random(1))


def test_one_bad_batch_row_cannot_be_normalized_into_a_valid_row():
    weights = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        FAMILY_REGISTRY["Categorical"].dist_class(probs=weights)
    weights = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive total"):
        _ValidatedCategorical(probs=weights)


@pytest.mark.parametrize("scale", [1.0, 1e308, 1e-300])
def test_finite_nonnegative_weights_normalize_without_overflow_or_underflow(scale):
    weights = torch.tensor([scale, scale, 0.0], dtype=torch.double)
    d = _ValidatedCategorical(probs=weights)
    torch.testing.assert_close(
        d.probs, torch.tensor([0.5, 0.5, 0.0], dtype=torch.double)
    )
    assert d.log_prob(torch.tensor(2)) == -torch.inf
    assert d.logits[2] == -torch.inf
    backend = ReferenceBackend()
    assert backend.log_prob(
        "Categorical", {"probs": weights.tolist()}, 0
    ) == pytest.approx(-math.log(2))
    assert backend.log_prob("Categorical", {"probs": weights.tolist()}, 2) == -math.inf
    assert d.expand((4,)).sample((3,)).shape == (3, 4)
    assert (d.sample((100,)) < 2).all()


def test_unnormalized_weights_keep_batched_reverse_forward_and_second_derivatives():
    weights = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]], dtype=torch.double, requires_grad=True
    )
    target = torch.tensor([1, 2])

    def score(value):
        return _ValidatedCategorical(probs=value).log_prob(target)

    def expected(value):
        return (
            value.gather(-1, target.unsqueeze(-1)).squeeze(-1).log()
            - value.sum(-1).log()
        )

    torch.testing.assert_close(score(weights), expected(weights))
    torch.testing.assert_close(
        torch.func.jacrev(score)(weights), torch.func.jacrev(expected)(weights)
    )
    torch.testing.assert_close(
        torch.func.jacfwd(score)(weights), torch.func.jacfwd(expected)(weights)
    )
    assert torch.autograd.gradcheck(score, (weights,))
    assert torch.autograd.gradgradcheck(score, (weights,))


def test_zero_weight_retains_finite_observed_cell_normalization_gradient():
    weights = torch.tensor([0.0, 2.0, 3.0], dtype=torch.double, requires_grad=True)
    score = _ValidatedCategorical(probs=weights).log_prob(torch.tensor(1))
    torch.testing.assert_close(
        torch.autograd.grad(score, weights)[0],
        torch.tensor([-0.2, 0.3, -0.2], dtype=torch.double),
    )


def test_checked_source_fixed_and_dynamic_probabilities_reject_all_negative_weights():
    from quivers.modeling import Model

    static = """object Row : FinSet 2
object Class : FinSet 3
program model(y:Tensor[Int]([Row])): Row -> Row
    observe y : Row <- Categorical(probs=[-1.0, -2.0, -3.0])
    return y
export model
"""
    with pytest.raises(ValueError):
        Model(static).bind({"y": torch.tensor([0, 1])}).trace({})
    dynamic = static.replace(
        "model(y:Tensor[Int]([Row]))",
        "model(p:Tensor[Real]([Class]), y:Tensor[Int]([Row]))",
    ).replace("probs=[-1.0, -2.0, -3.0]", "probs=p")
    with pytest.raises(ValueError, match="nonnegative"):
        Model(dynamic).bind(
            {"p": torch.tensor([-1.0, -2.0, -3.0]), "y": torch.tensor([0, 1])}
        ).trace({})


def test_tied_maximum_weights_keep_second_derivatives_and_logits_zero_support():
    weights = torch.tensor([1.0, 1.0, 0.5], dtype=torch.double, requires_grad=True)

    def score(value):
        return _ValidatedCategorical(probs=value).log_prob(torch.tensor(0))

    assert torch.autograd.gradcheck(score, (weights,))
    assert torch.autograd.gradgradcheck(score, (weights,))
    logits = torch.tensor(
        [0.0, -torch.inf, 0.0], dtype=torch.double, requires_grad=True
    )
    distribution = _ValidatedCategorical(logits=logits)
    assert distribution.log_prob(torch.tensor(1)) == -torch.inf
    assert not (distribution.expand((2,)).sample((100,)) == 1).any()
    torch.testing.assert_close(
        torch.autograd.grad(distribution.log_prob(torch.tensor(0)), logits)[0],
        torch.tensor([0.5, 0.0, -0.5], dtype=torch.double),
    )
