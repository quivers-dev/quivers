"""Independent CDF and derivative checks for native scalar count kernels."""

import pytest
import torch
from scipy import stats
from quivers.continuous._closed_probability import (
    ClosedBernoulli,
    ClosedBinomial,
    ClosedNegativeBinomial,
    ClosedPoisson,
    _ValidatedCategorical,
)


@pytest.mark.parametrize(
    "native,oracle",
    [
        (ClosedBernoulli(probs=torch.tensor(0.3)), stats.bernoulli(0.3)),
        (
            ClosedBinomial(10, probs=torch.tensor(0.3, dtype=torch.float64)),
            stats.binom(10, 0.3),
        ),
        (
            ClosedNegativeBinomial(2.0, probs=torch.tensor(0.3, dtype=torch.float64)),
            stats.nbinom(2.0, 0.7),
        ),
        (ClosedPoisson(torch.tensor(3.0, dtype=torch.float64)), stats.poisson(3.0)),
    ],
)
def test_count_cdf_and_log_tails_against_scipy(native, oracle):
    point = torch.tensor(
        [-torch.inf, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, torch.inf], dtype=torch.float64
    )
    torch.testing.assert_close(
        native.cdf(point),
        torch.as_tensor(oracle.cdf(point.numpy())),
        atol=1e-12,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        native.log_cdf(point),
        torch.as_tensor(oracle.logcdf(point.numpy())),
        atol=1e-12,
        rtol=1e-7,
    )
    torch.testing.assert_close(
        native.log_survival_function(point),
        torch.as_tensor(oracle.logsf(point.numpy())),
        atol=1e-12,
        rtol=1e-7,
    )


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_binomial_probability_cdf_endpoint_first_and_second_derivatives(p):
    probability = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    law = ClosedBinomial(3, probs=probability)
    for threshold in [0.0, 1.0, 2.0]:
        k = int(threshold)
        direct = sum(
            torch.tensor(float(__import__("math").comb(3, j)), dtype=torch.float64)
            * probability**j
            * (1 - probability) ** (3 - j)
            for j in range(k + 1)
        )
        score = law.cdf(torch.tensor(threshold, dtype=torch.float64))
        (first,) = torch.autograd.grad(
            score, probability, create_graph=True, retain_graph=True
        )
        (second,) = torch.autograd.grad(first, probability, retain_graph=True)
        (expected,) = torch.autograd.grad(
            direct, probability, create_graph=True, retain_graph=True
        )
        (expected_second,) = torch.autograd.grad(
            expected, probability, retain_graph=True
        )
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, expected_second)


def test_poisson_large_count_and_underflowed_log_tail():
    law = ClosedPoisson(torch.tensor([1e6, 1000.0], dtype=torch.float64))
    point = torch.tensor([1e6, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        law.log_cdf(point),
        torch.tensor([stats.poisson.logcdf(1e6, 1e6), -1000.0], dtype=torch.float64),
        atol=1e-11,
        rtol=1e-11,
    )


def test_poisson_tail_reverse_forward_and_second_derivatives():
    def score(rate):
        return ClosedPoisson(rate).log_cdf(torch.tensor(3.0, dtype=torch.float64))

    rate = torch.tensor(2.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(score, (rate,))
    assert torch.autograd.gradgradcheck(score, (rate,))
    torch.testing.assert_close(
        torch.func.jacfwd(score)(rate), torch.func.jacrev(score)(rate)
    )


def test_categorical_prefix_suffix_preserves_unequal_row_and_category_axes():
    probs = torch.tensor(
        [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=torch.float64, requires_grad=True
    )
    law = _ValidatedCategorical(probs=probs)
    point = torch.tensor([0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(
        law.cdf(point), torch.tensor([0.2, 0.7], dtype=torch.float64)
    )
    torch.testing.assert_close(
        law.survival_function(point), torch.tensor([0.8, 0.3], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda p: ClosedBernoulli(probs=p),
        lambda p: ClosedBinomial(3, probs=p),
        lambda p: ClosedNegativeBinomial(3, probs=p),
    ],
)
def test_shared_closed_probability_expansion_retains_identity_and_gradients(
    constructor,
):
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    law = constructor(p)
    expanded = law.expand((2, 3))
    assert type(expanded) is type(law)
    torch.testing.assert_close(
        expanded.cdf(torch.zeros(2, 3)), law.cdf(torch.tensor(0.0)).expand(2, 3)
    )
    (actual,) = torch.autograd.grad(expanded.cdf(torch.zeros(2, 3)).sum(), p)
    (expected,) = torch.autograd.grad(6 * law.cdf(torch.tensor(0.0)), p)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["beta", "zip", "zinb", "zib", "zibb", "hnb", "hp"])
def test_compound_count_cdf_matches_independent_finite_mass_sums(kind):
    from quivers.continuous._zero_inflated_count import (
        BetaBinomial,
        ZeroInflatedNegativeBinomial,
        ZeroInflatedBinomial,
        ZeroInflatedBetaBinomial,
        HurdleNegativeBinomial,
    )
    from quivers.continuous._zip_hurdle import ZeroInflatedPoisson, HurdlePoisson

    z = torch.tensor(0.3, dtype=torch.float64)
    if kind == "beta":
        law = BetaBinomial(8, 2.0, 3.0)
        base = stats.betabinom(8, 2, 3)
        zi = 0.0
    elif kind == "zip":
        law = ZeroInflatedPoisson(z, torch.tensor(2.0, dtype=torch.float64))
        base = stats.poisson(2)
        zi = 0.3
    elif kind == "zinb":
        law = ZeroInflatedNegativeBinomial(z, 2.0, 0.4)
        base = stats.nbinom(2, 0.6)
        zi = 0.3
    elif kind == "zib":
        law = ZeroInflatedBinomial(z, 8, 0.4)
        base = stats.binom(8, 0.4)
        zi = 0.3
    elif kind == "zibb":
        law = ZeroInflatedBetaBinomial(z, 8, 2.0, 3.0)
        base = stats.betabinom(8, 2, 3)
        zi = 0.3
    elif kind == "hnb":
        law = HurdleNegativeBinomial(z, 2.0, 0.4)
        base = stats.nbinom(2, 0.6)
        zi = 0.3
    else:
        law = HurdlePoisson(z, torch.tensor(2.0, dtype=torch.float64))
        base = stats.poisson(2)
        zi = 0.3
    point = torch.tensor(
        [-1.0, 0.0, 1.0, 2.0, 5.0, 10.0, torch.inf], dtype=torch.float64
    )
    expected = base.cdf(point.numpy())
    if kind in {"hnb", "hp"}:
        expected = (expected - base.pmf(0)) / (1 - base.pmf(0))
        expected[point.numpy() < 1] = 0
    expected = zi + (1 - zi) * expected
    expected[point.numpy() < 0] = 0
    torch.testing.assert_close(
        law.cdf(point), torch.as_tensor(expected), atol=1e-12, rtol=1e-10
    )
    torch.testing.assert_close(
        law.survival_function(point),
        1 - torch.as_tensor(expected),
        atol=1e-12,
        rtol=1e-10,
    )


@pytest.mark.parametrize("kind", ["poisson", "nb"])
def test_hurdle_degenerate_component_cdf_retains_first_second_derivatives(kind):
    from quivers.continuous._zero_inflated_count import HurdleNegativeBinomial
    from quivers.continuous._zip_hurdle import HurdlePoisson

    parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(0.2, dtype=torch.float64)
    law = (
        HurdlePoisson(z, parameter)
        if kind == "poisson"
        else HurdleNegativeBinomial(z, 2.0, parameter)
    )
    for threshold in [1.0, 2.0]:
        score = law.cdf(torch.tensor(threshold, dtype=torch.float64))
        (first,) = torch.autograd.grad(
            score, parameter, create_graph=True, retain_graph=True
        )
        (second,) = torch.autograd.grad(first, parameter, retain_graph=True)
        expected_first = (
            (-0.4 if threshold == 1 else 0.0)
            if kind == "poisson"
            else (-1.2 if threshold == 1 else 0.0)
        )
        expected_second = (
            (0.8 / 6 if threshold == 1 else -0.8 / 3)
            if kind == "poisson"
            else (0.4 if threshold == 1 else -3.2)
        )
        assert first.item() == pytest.approx(expected_first)
        assert second.item() == pytest.approx(expected_second)


def test_geometric_proper_domain_extreme_logits_and_closed_endpoint_ad():
    from quivers.continuous._closed_probability import ClosedGeometric

    for parameter in [0.0, -1.0]:
        with pytest.raises(ValueError, match="proper success"):
            ClosedGeometric(probs=parameter)
    with pytest.raises(ValueError, match="zero success"):
        ClosedGeometric(logits=-torch.inf)
    law = ClosedGeometric(logits=torch.tensor(-1000.0, dtype=torch.float64))
    assert law.log_cdf(torch.tensor(3.0)).item() == pytest.approx(
        -1000 + __import__("math").log(4)
    )
    assert law.log_prob(torch.tensor(0.0)).item() == -1000
    with pytest.raises(RuntimeError, match="capacity"):
        law.sample()
    p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    endpoint = ClosedGeometric(probs=p)
    for threshold, first, second in [(0.0, 1.0, 0.0), (1.0, 0.0, -2.0)]:
        cdf = endpoint.cdf(torch.tensor(threshold))
        (grad,) = torch.autograd.grad(cdf, p, create_graph=True, retain_graph=True)
        (hessian,) = torch.autograd.grad(grad, p, retain_graph=True)
        assert grad.item() == pytest.approx(first)
        assert hessian.item() == pytest.approx(second)


@pytest.mark.parametrize("kind", ["poisson", "nb"])
def test_tiny_hurdle_positive_survival_retains_leading_mass(kind):
    from quivers.continuous._zip_hurdle import HurdlePoisson
    from quivers.continuous._zero_inflated_count import HurdleNegativeBinomial

    p = torch.tensor(1e-300, dtype=torch.float64)
    z = torch.tensor(0.2, dtype=torch.float64)
    law = (
        HurdlePoisson(z, p) if kind == "poisson" else HurdleNegativeBinomial(z, 2.0, p)
    )
    # Conditional positive P(Y>1) is rate/2+O(rate²), or (r+1)p/2+O(p²).
    coefficient = 0.5 if kind == "poisson" else 1.5
    expected = __import__("math").log(0.8 * coefficient) + __import__("math").log(
        1e-300
    )
    assert law.log_survival_function(torch.tensor(1.0)).item() == pytest.approx(
        expected, abs=1e-12
    )


def test_geometric_certain_success_score_and_large_logit_cdf_gradients_are_finite():
    from quivers.continuous._closed_probability import ClosedGeometric

    p = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(
        ClosedGeometric(probs=p).log_prob(torch.tensor(0.0)), p
    )
    assert gradient.item() == 1.0
    logit = torch.tensor(1000.0, dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(
        ClosedGeometric(logits=logit).log_cdf(torch.tensor(3.0)), logit
    )
    assert gradient.item() == 0.0


@pytest.mark.parametrize(
    "class_name",
    [
        "Bernoulli",
        "Binomial",
        "NegativeBinomial",
        "Categorical",
        "Poisson",
        "Geometric",
    ],
)
def test_bare_torch_count_adapter_uses_shared_kernels(class_name):
    from quivers.continuous._count_tails import distribution_count_log_tails

    args = {"probs": torch.tensor(0.25, dtype=torch.float64)}
    oracles = {
        "Bernoulli": stats.bernoulli(0.25),
        "Binomial": stats.binom(8, 0.25),
        "NegativeBinomial": stats.nbinom(2, 0.75),
        "Poisson": stats.poisson(2),
        "Geometric": stats.geom(0.25, loc=-1),
    }
    if class_name in {"Binomial", "NegativeBinomial"}:
        args["total_count"] = torch.tensor(8 if class_name == "Binomial" else 2)
    if class_name == "Poisson":
        args = {"rate": torch.tensor(2.0, dtype=torch.float64)}
    if class_name == "Categorical":
        args = {"probs": torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)}
        oracle = stats.rv_discrete(values=([0, 1, 2], [0.25, 0.25, 0.5]))
    else:
        oracle = oracles[class_name]
    law = getattr(torch.distributions, class_name)(**args)
    points = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
    lc, ls = distribution_count_log_tails(law, points)
    torch.testing.assert_close(
        lc, torch.tensor(oracle.logcdf(points.numpy())), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        ls, torch.tensor(oracle.logsf(points.numpy())), atol=1e-12, rtol=1e-12
    )


def test_bare_geometric_cached_coordinate_ambiguity_is_explicit():
    from quivers.continuous._count_tails import distribution_count_log_tails

    law = torch.distributions.Geometric(
        logits=torch.tensor(-1000.0, dtype=torch.float64)
    )
    assert distribution_count_log_tails(law, torch.tensor(0.0))[0].item() == -1000.0
    _ = law.probs
    with pytest.raises(ValueError, match="provenance.*ClosedGeometric"):
        distribution_count_log_tails(law, torch.tensor(0.0))


def test_categorical_finite_logit_tails_survive_probability_underflow_and_caches():
    from quivers.continuous._count_tails import distribution_count_log_tails
    from quivers.qiec._count_tails import log_count_tails

    for cls in [_ValidatedCategorical, torch.distributions.Categorical]:
        logits = torch.tensor(
            [-1000.0, 0.0, -2000.0], dtype=torch.float64, requires_grad=True
        )
        law = cls(logits=logits)
        _ = law.probs
        lower, upper = distribution_count_log_tails(law, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(lower[0], logits[0])
        torch.testing.assert_close(upper[1], logits[2])
        torch.testing.assert_close(
            torch.autograd.grad(lower[0], logits)[0],
            torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
        )
    assert (
        log_count_tails("Categorical", {"logits": (-1000.0, 0.0, -2000.0)}, 0.0)[0]
        == -1000.0
    )


def test_poisson_tail_vectorized_ad_includes_underflow_rows_without_row_loops():
    from quivers.continuous._count_tails import poisson_log_tails

    rates = torch.tensor([0.3, 3.0, 1000.0], dtype=torch.float64)
    threshold = torch.tensor(3.0, dtype=torch.float64)

    def score(rate):
        return poisson_log_tails(rate, threshold)[0]

    actual = torch.vmap(score)(rates)
    expected = torch.stack([score(rate) for rate in rates])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.vmap(torch.func.grad(score))(rates),
        torch.stack([torch.func.grad(score)(rate) for rate in rates]),
    )
    torch.testing.assert_close(
        torch.func.jacfwd(lambda x: torch.vmap(score)(x))(rates),
        torch.func.jacrev(lambda x: torch.vmap(score)(x))(rates),
    )
