"""Marginal priors retain exact boundary and very small atom weights."""

import math
from collections.abc import Iterator

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.compiler.programs import _binary_log_prior, _categorical_log_prior


@pytest.fixture(autouse=True)
def seeded_prior_draws() -> Iterator[None]:
    with torch.random.fork_rng():
        torch.manual_seed(391)
        yield


def _source(family: str, grouped: bool) -> str:
    prior = (
        "Bernoulli(probs=q)" if family == "Bernoulli" else "Categorical(probs=weights)"
    )
    annotation = "" if family == "Bernoulli" else " : K"
    binding = (
        ""
        if family == "Bernoulli"
        else "    sample weights : K <- Normal(loc=0.0, scale=1.0)\n"
    )
    group = " [over=G]" if grouped else ""
    via = " [via=idx]" if grouped else ""
    return f"""
object Obs : FinSet 1
object K : FinSet 2
object G : FinSet 1
object Out : Real 1
program p : Obs -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
{binding}    marginalize z{annotation} <- {prior}{group}
        let probability = real(z)
        observe y : Obs <- Bernoulli(probs=probability){via}
    return a
export p
"""


@pytest.mark.parametrize(
    "family,grouped",
    [("Bernoulli", False), ("Categorical", False), ("Categorical", True)],
)
@pytest.mark.parametrize("probability", [0.0, 1e-50, 1.0])
def test_marginal_prior_preserves_atom_weight(
    family: str, grouped: bool, probability: float
) -> None:
    values = {
        "a": torch.tensor(0.0, dtype=torch.float64),
        "y": torch.tensor([1.0], dtype=torch.float64),
        "q": torch.tensor(probability, dtype=torch.float64),
        "weights": torch.tensor([1 - probability, probability], dtype=torch.float64),
        "idx": torch.tensor([0]),
    }
    actual = loads(_source(family, grouped)).morphism.log_joint(
        torch.zeros(1, 1), values
    )
    if family == "Categorical":
        actual = (
            actual
            + (0.5 * values["weights"].square() + 0.5 * math.log(2 * math.pi)).sum()
        )
    expected = values["q"].log() - 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(actual.reshape(()), expected, rtol=1e-6, atol=1e-5)
    if probability <= 1e-50:
        floored = math.log(1e-38) - 0.5 * math.log(2 * math.pi)
        assert not torch.isclose(actual, actual.new_tensor(floored)).any()


@pytest.mark.parametrize("grouped", [False, True])
def test_marginal_categorical_normalizes_weights(grouped: bool) -> None:
    values = {
        "a": torch.tensor(0.0, dtype=torch.float64),
        "y": torch.tensor([1.0], dtype=torch.float64),
        "weights": torch.tensor([2.0, 3.0], dtype=torch.float64),
        "idx": torch.tensor([0]),
    }
    actual = loads(_source("Categorical", grouped)).morphism.log_joint(
        torch.zeros(1, 1), values
    )
    actual = (
        actual + (0.5 * values["weights"].square() + 0.5 * math.log(2 * math.pi)).sum()
    )
    expected = math.log(3 / 5) - 0.5 * math.log(2 * math.pi)
    assert actual.item() == pytest.approx(expected, abs=1e-5)
    wrong = math.log(3) - 0.5 * math.log(2 * math.pi)
    assert abs(actual.item() - wrong) > 1


@pytest.mark.parametrize(
    "weights", [[-2.0, -3.0], [-1.0, 2.0], [0.0, 0.0], [math.nan, 1.0]]
)
def test_categorical_prior_refuses_invalid_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        _categorical_log_prior(torch.tensor(weights), "probs")


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
def test_binary_prior_refuses_invalid_probabilities(value: float) -> None:
    with pytest.raises(ValueError):
        _binary_log_prior(torch.tensor(value), "probs")


def test_categorical_logits_preserve_common_offset_invariance() -> None:
    logits = torch.full((2,), -1e100, dtype=torch.float64)
    torch.testing.assert_close(
        _categorical_log_prior(logits, "logits"),
        torch.full((2,), -math.log(2), dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("probability", [0.0, 1e-50, 0.3, 1.0])
def test_prior_mixture_gradients_are_finite_at_boundaries(probability: float) -> None:
    q = torch.tensor(probability, dtype=torch.float64, requires_grad=True)
    likelihood = torch.tensor([0.2, 0.8], dtype=torch.float64)
    score = torch.logsumexp(_binary_log_prior(q, "probs") + likelihood.log(), dim=-1)
    assert score.item() == pytest.approx(
        math.log((1 - probability) * 0.2 + probability * 0.8)
    )
    grad = torch.autograd.grad(score, q)[0]
    assert torch.isfinite(grad)
    if 0 < probability < 1:
        assert grad.item() == pytest.approx(
            0.6 / ((1 - probability) * 0.2 + probability * 0.8)
        )
