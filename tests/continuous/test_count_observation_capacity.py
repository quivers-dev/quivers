"""Original integer observations retain one exact native/reference boundary."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from quivers.continuous._closed_probability import (
    ClosedBernoulli,
    ClosedBinomial,
    ClosedNegativeBinomial,
    ClosedPoisson,
    ClosedGeometric,
    _ValidatedCategorical,
    _check_count_observation_source,
)
from quivers.qiec.distributions import ReferenceBackend
from quivers.transpile.family_meta import FAMILY_META


CASES = {
    "Poisson": {"rate": 3.0},
    "Binomial": {"total_count": 2**53, "probs": 0.4},
    "NegativeBinomial": {"total_count": 1.7, "probs": 0.4},
    "Geometric": {"probs": 0.4},
    "BetaBinomial": {
        "total_count": 2**53,
        "concentration1": 2.0,
        "concentration0": 3.0,
    },
    "ZeroInflatedPoisson": {"zero_prob": 0.2, "rate": 3.0},
    "HurdlePoisson": {"zero_prob": 0.2, "rate": 3.0},
    "ZeroInflatedBinomial": {"zero_prob": 0.2, "total_count": 2**53, "probs": 0.4},
    "ZeroInflatedNegativeBinomial": {
        "zero_prob": 0.2,
        "total_count": 1.7,
        "probs": 0.4,
    },
    "ZeroInflatedBetaBinomial": {
        "zero_prob": 0.2,
        "total_count": 2**53,
        "concentration1": 2.0,
        "concentration0": 3.0,
    },
    "HurdleNegativeBinomial": {"zero_prob": 0.2, "total_count": 1.7, "probs": 0.4},
    "DiscreteWeibull": {"probs": 0.4, "concentration": 1.3},
    "COMPoisson": {"rate": 3.0, "dispersion": 1.3},
    "Multinomial": {"total_count": 2**53, "probs": [0.4, 0.6]},
    "DirichletMultinomial": {"total_count": 2**53, "concentration": [2.0, 3.0]},
}


def _law(name):
    cls = {
        "Poisson": ClosedPoisson,
        "Binomial": ClosedBinomial,
        "NegativeBinomial": ClosedNegativeBinomial,
        "Geometric": ClosedGeometric,
    }.get(name, FAMILY_META[name].distribution_class)
    args = {
        key: torch.as_tensor(value, dtype=torch.double)
        for key, value in CASES[name].items()
    }
    return cls(**args, validate_args=False)


def _event(name, count):
    return [count, 0] if name in ("Multinomial", "DirichletMultinomial") else count


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("count", [2**53 + 1, 2**63 - 1, -(2**53 + 1), -(2**63) + 1])
def test_original_integer_capacity_is_explicit_in_both_backends(name, count):
    event = _event(name, count)
    # Native Int64 must be checked before a Float64 conversion can erase a bit.
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _law(name).log_prob(torch.tensor(event, dtype=torch.int64))
    with pytest.raises(ValueError, match=r"2\*\*53"):
        ReferenceBackend().log_prob(name, CASES[name], event)


@pytest.mark.parametrize("name", CASES)
def test_mixed_sequence_is_checked_before_default_dtype_inference(name):
    event = [0.5, 2**53 + 1]
    # Shape validity is secondary: never round an original sequence first.
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _law(name).log_prob(event)


@pytest.mark.parametrize(
    "value",
    [
        2**53,
        -(2**53),
        [0.5, 2**53],
        [[-(2**53)], [2**53]],
        torch.tensor([2**53, -(2**53)]),
        torch.tensor([2**53, -(2**53)], dtype=torch.double),
        np.array([2**53, -(2**53)], dtype=np.int64),
    ],
)
def test_shared_guard_preserves_exact_boundaries_without_conversion(value):
    _check_count_observation_source(value)


@pytest.mark.parametrize("name", CASES)
def test_exact_positive_capacity_is_supported_without_loosening_count_law(name):
    event = _event(name, 2**53)
    score = _law(name).log_prob(torch.tensor(event, dtype=torch.int64))
    reference = ReferenceBackend().log_prob(name, CASES[name], event)
    assert math.isfinite(float(score)) and float(score) <= 0
    assert math.isfinite(reference) and reference <= 0
    assert float(score) == pytest.approx(reference, rel=2e-13, abs=1e-8)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("value", [-1.0, 0.5, math.nan, math.inf, -math.inf])
def test_invalid_counts_are_impossible_without_integer_truncation(name, value):
    event = _event(name, value)
    score = _law(name).log_prob(torch.tensor(event, dtype=torch.double))
    assert torch.isneginf(score)
    assert ReferenceBackend().log_prob(name, CASES[name], event) == -math.inf


@pytest.mark.parametrize(
    "name,law",
    [
        ("Bernoulli", ClosedBernoulli(probs=torch.tensor(0.4), validate_args=False)),
        ("Bernoulli", ClosedBernoulli(logits=torch.tensor(0.4), validate_args=False)),
        (
            "Categorical",
            _ValidatedCategorical(probs=torch.tensor([0.4, 0.6]), validate_args=False),
        ),
    ],
)
@pytest.mark.parametrize("value", [0.5, 2**53 + 1, math.nan, math.inf, -math.inf])
def test_finite_discrete_support_is_checked_before_casting_indices(name, law, value):
    point = (
        torch.tensor(value)
        if isinstance(value, int)
        else torch.tensor(value, dtype=torch.double)
    )
    assert torch.isneginf(law.log_prob(point))
    args = {"probs": [0.4, 0.6]} if name == "Categorical" else {"probs": 0.4}
    assert ReferenceBackend().log_prob(name, args, value) == -math.inf


@pytest.mark.parametrize(
    "cls,kwargs,oracle",
    [
        (
            ClosedPoisson,
            {"rate": 33_554_433.0},
            lambda k: stats.poisson.logpmf(k, 33_554_433.0),
        ),
        (
            ClosedBinomial,
            {"total_count": 67_108_866, "probs": 0.5},
            lambda k: stats.binom.logpmf(k, 67_108_866, 0.5),
        ),
        (
            ClosedNegativeBinomial,
            {"total_count": 1.7, "probs": 0.4},
            lambda k: stats.nbinom.logpmf(k, 1.7, 0.6),
        ),
    ],
)
def test_large_exact_counts_still_match_independent_scores_and_head_gradients(
    cls, kwargs, oracle
):
    count = 33_554_433
    parameters = {
        key: torch.tensor(value, dtype=torch.double, requires_grad=True)
        for key, value in kwargs.items()
    }
    # Binomial trial count is structural, not a differentiable population head.
    if cls is ClosedBinomial:
        parameters["total_count"] = torch.tensor(
            kwargs["total_count"], dtype=torch.int64
        )
    score = cls(**parameters, validate_args=False).log_prob(torch.tensor(count))
    assert float(score.detach()) == pytest.approx(float(oracle(count)), abs=2e-6)
    differentiable = tuple(
        value for value in parameters.values() if value.requires_grad
    )
    gradient = torch.autograd.grad(score, differentiable)
    assert all(torch.isfinite(value).all() for value in gradient)
    if cls is ClosedPoisson:
        assert float(gradient[0]) == pytest.approx(0.0, abs=2e-15)
    elif cls is ClosedBinomial:
        assert float(gradient[0]) == pytest.approx(0.0, abs=2e-7)
    else:
        assert float(gradient[-1]) == pytest.approx(count / 0.4 - 1.7 / 0.6, rel=2e-13)


def test_invalid_count_masks_do_not_contaminate_learned_head_gradients():
    for cls, key, value in [
        (ClosedPoisson, "rate", 2.0),
        (ClosedNegativeBinomial, "probs", 0.4),
    ]:
        head = torch.tensor(value, dtype=torch.double, requires_grad=True)
        args = {key: head}
        if cls is ClosedNegativeBinomial:
            args["total_count"] = 1.7
        score = cls(**args, validate_args=False).log_prob(
            torch.tensor([0.5, math.nan, 2.0])
        )
        (gradient,) = torch.autograd.grad(
            torch.where(torch.isfinite(score), score, 0.0).sum(), head
        )
        expected = (
            2 / value - 1 if cls is ClosedPoisson else 2 / value - 1.7 / (1 - value)
        )
        assert float(gradient) == pytest.approx(expected, rel=2e-13)


def test_count_capacity_validation_composes_with_mapped_and_forward_ad():
    values = torch.tensor([2, 3, 4], dtype=torch.int64)
    rate = torch.tensor(2.0, dtype=torch.double)

    def score(r, value):
        return ClosedPoisson(r, validate_args=False).log_prob(value)

    torch.testing.assert_close(
        torch.vmap(score, in_dims=(None, 0))(rate, values), score(rate, values)
    )
    gradient = torch.vmap(torch.func.grad(score), in_dims=(None, 0))(rate, values)
    torch.testing.assert_close(gradient, values.double() / rate - 1)
    _, tangent = torch.func.jvp(
        lambda r: torch.vmap(score, in_dims=(None, 0))(r, values),
        (rate,),
        (torch.ones_like(rate),),
    )
    torch.testing.assert_close(tangent, gradient)
    bad = torch.tensor([2, 2**53 + 1], dtype=torch.int64)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        torch.vmap(score, in_dims=(None, 0))(rate, bad)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        torch.vmap(torch.func.grad(score), in_dims=(None, 0))(rate, bad)


@pytest.mark.parametrize(
    "source",
    [
        float(2**53 + 2),
        torch.tensor(2**53 + 2, dtype=torch.double),
        np.array([2**53 + 1], dtype=np.int64),
        [0.5, 2**53 + 1],
        [[2**53 + 1], [1]],
        2**1000,
    ],
)
def test_original_scalar_sequence_and_array_inputs_share_capacity_validation(source):
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _check_count_observation_source(source)


@pytest.mark.parametrize(
    "family,arguments",
    [
        ("Poisson", "rate=2.0"),
        ("NegativeBinomial", "total_count=1.7, probs=0.4"),
        ("ZeroInflatedPoisson", "zero_prob=0.2, rate=2.0"),
        ("COMPoisson", "rate=2.0, dispersion=1.3"),
    ],
)
def test_checked_source_preserves_original_count_until_capacity_validation(
    family, arguments
):
    from quivers.modeling import Model

    source = f"""object Out : Real 1
program count[n: Nat](y: Tensor[Int]([n])): n -> Out
    sample theta <- Normal(loc=0.0, scale=1.0)
    observe y: n <- {family}({arguments})
    return theta
export count
"""
    # Either bind-time validation or the scorer may reject, but neither may
    # round the original tensor into a different admitted integer observation.
    with pytest.raises(ValueError, match=r"2\*\*53"):
        Model(source).bind({"y": torch.tensor([1, 2**53 + 1])}).trace(
            {"theta": torch.tensor([0.0])}
        )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.uint16,
        torch.uint32,
        torch.float16,
    ],
)
def test_provably_small_storage_avoids_capacity_scan_and_host_synchronization(
    dtype, monkeypatch
):
    from quivers.continuous._closed_probability import _CountSourceCapacity

    def unexpected_scan(*args):
        raise AssertionError("bounded storage requires no physical count scan")

    monkeypatch.setattr(_CountSourceCapacity, "apply", unexpected_scan)
    _check_count_observation_source(torch.tensor([0, 1], dtype=dtype))


def test_float16_infinite_trials_remain_rejected_while_observations_are_off_support():
    from quivers.continuous._closed_probability import _check_trial_count_source

    point = torch.tensor(math.inf, dtype=torch.float16)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        _check_trial_count_source(point)
    _check_count_observation_source(point)
    assert torch.isneginf(ClosedPoisson(2.0, validate_args=False).log_prob(point))
