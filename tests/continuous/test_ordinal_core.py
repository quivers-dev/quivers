"""Independent probability, validation, broadcasting, and AD checks for ordinal kernels."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import expit, log_ndtr, ndtr

from quivers.continuous._ordinal import OrdinalDistribution, category_log_probs
from quivers.continuous._ordered import OrderedLogistic, OrderedProbit

KINDS = ("cumulative", "acat", "sratio", "cratio")
LINKS = ("logit", "probit")
ALL_LINKS = (*LINKS, "probit_approx", "cloglog", "cauchit", "softit")
CASES = [
    (kind, link)
    for kind in KINDS
    for link in ALL_LINKS
    if not (kind in {"sratio", "cratio"} and link == "softit")
]


def _oracle(eta, cuts, disc, kind, link, effects):
    z = disc * (cuts - eta - effects)

    def cdf(x):
        if link == "logit":
            return expit(x)
        if link == "probit":
            return ndtr(x)
        if link == "probit_approx":
            return expit(0.07056 * x**3 + 1.5976 * x)
        if link == "cloglog":
            return -np.expm1(-np.exp(x))
        if link == "cauchit":
            return 0.5 + np.arctan(x) / np.pi
        positive = np.logaddexp(0.0, x)
        return positive / (1 + positive)

    if kind == "cumulative":
        return np.diff(np.r_[0.0, cdf(z), 1.0])
    if kind == "acat":
        q = cdf(-z)
        raw = np.array(
            [np.prod(q[:j]) * np.prod(1 - q[j:]) for j in range(len(cuts) + 1)]
        )
        return raw / raw.sum()
    stop = cdf(z) if kind == "sratio" else 1 - cdf(-z)
    return np.array(
        [
            np.prod(1 - stop[:j]) * (stop[j] if j < len(cuts) else 1)
            for j in range(len(cuts) + 1)
        ]
    )


@pytest.mark.parametrize("kind,link", CASES)
def test_probabilities_match_independent_finite_products(kind, link):
    eta = torch.tensor([-1.2, 0.4, 2.1], dtype=torch.double)
    cuts = torch.tensor([-1.1, 0.2, 1.7], dtype=torch.double)
    effects = torch.tensor(
        [[0.1, -0.2, 0.2], [-0.3, 0.2, -0.1], [0.0, 0.1, -0.2]], dtype=torch.double
    )
    disc = torch.tensor([0.5, 1.0, 2.3], dtype=torch.double)
    dist = OrdinalDistribution(
        eta, cuts, kind=kind, link=link, discrimination=disc, category_effects=effects
    )
    expected = np.stack(
        [
            _oracle(e.item(), cuts.numpy(), d.item(), kind, link, f.numpy())
            for e, d, f in zip(eta, disc, effects, strict=True)
        ]
    )
    np.testing.assert_allclose(
        dist.category_probs().numpy(), expected, rtol=1e-11, atol=2e-15
    )
    torch.testing.assert_close(
        dist.category_probs().sum(-1), torch.ones(3, dtype=torch.double)
    )


@pytest.mark.parametrize("kind,link", CASES)
def test_every_parameter_reverse_forward_and_second_derivatives(kind, link):
    inputs = (
        torch.tensor([-0.4, 0.8], dtype=torch.double, requires_grad=True),
        torch.tensor([-1.0, 0.3, 1.4], dtype=torch.double, requires_grad=True),
        torch.tensor([0.7, 1.3], dtype=torch.double, requires_grad=True),
        torch.tensor(
            [[0.1, -0.1, 0.2], [-0.1, 0.1, -0.2]],
            dtype=torch.double,
            requires_grad=True,
        ),
    )

    def fn(eta, cuts, disc, effects):
        return category_log_probs(
            eta,
            cuts,
            kind=kind,
            link=link,
            discrimination=disc,
            category_effects=effects,
        )

    assert torch.autograd.gradcheck(fn, inputs, check_forward_ad=True)
    assert torch.autograd.gradgradcheck(fn, inputs)
    _, tangent = torch.func.jvp(
        fn, inputs, tuple(torch.ones_like(x) * 0.1 for x in inputs)
    )
    assert torch.isfinite(tangent).all()


@pytest.mark.parametrize("link", LINKS)
def test_cumulative_extreme_tail_mass_and_gradients(link):
    # Independent tail oracle mirrors SciPy log CDFs before subtracting.
    eta = torch.tensor([-40.0, 40.0], dtype=torch.double, requires_grad=True)
    cuts = torch.tensor([-0.3, 0.2], dtype=torch.double, requires_grad=True)
    actual = OrdinalDistribution(eta, cuts, link=link).category_log_probs()
    z = (cuts[None, :] - eta[:, None]).detach().numpy()
    if link == "probit":
        expected = []
        for a, b in z:
            hi, lo = (
                (log_ndtr(-a), log_ndtr(-b)) if a > 0 else (log_ndtr(b), log_ndtr(a))
            )
            expected.append(
                [log_ndtr(a), hi + np.log(-np.expm1(lo - hi)), log_ndtr(-b)]
            )
    else:
        expected = [
            [
                -np.logaddexp(0, -a),
                -np.logaddexp(0, -b) - np.logaddexp(0, a) + np.log(-np.expm1(a - b)),
                -np.logaddexp(0, b),
            ]
            for a, b in z
        ]
    np.testing.assert_allclose(actual.detach(), expected, rtol=2e-13, atol=2e-13)
    grads = torch.autograd.grad(actual.sum(), (eta, cuts), create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(
        torch.isfinite(g).all()
        for g in torch.autograd.grad(sum(g.sum() for g in grads), (eta, cuts))
    )


@pytest.mark.parametrize("kind", KINDS)
def test_batched_parameters_sampling_and_zero_size(kind):
    dist = OrdinalDistribution(
        torch.zeros(2, 1), torch.tensor([-1.0, 0.0, 1.0]).expand(3, 3), kind=kind
    )
    assert dist.category_log_probs().shape == (2, 3, 4)
    assert dist.log_prob(torch.tensor([0, 1, 2])).shape == (2, 3)
    assert dist.sample((5, 4)).shape == (5, 4, 2, 3)
    assert dist.sample((0,)).shape == (0, 2, 3)


@pytest.mark.parametrize("kind,link", CASES)
def test_binary_equals_bernoulli_and_discrimination_scale_invariance(kind, link):
    dist = OrdinalDistribution(
        torch.tensor(0.3, dtype=torch.double),
        torch.tensor([0.1], dtype=torch.double),
        kind=kind,
        link=link,
        discrimination=1.7,
    )
    expected = _oracle(0.3, np.array([0.1]), 1.7, kind, link, 0.0)
    np.testing.assert_allclose(dist.category_probs(), expected)
    scaled = OrdinalDistribution(
        torch.tensor(0.6, dtype=torch.double),
        torch.tensor([0.2], dtype=torch.double),
        kind=kind,
        link=link,
        discrimination=0.85,
    )
    torch.testing.assert_close(dist.category_probs(), scaled.category_probs())


@pytest.mark.parametrize("link", LINKS)
def test_symmetric_sequential_models_agree(link):
    kwargs = dict(
        predictor=torch.tensor([-0.3, 1.2]),
        cutpoints=torch.tensor([0.3, -1.0, 0.5]),
        link=link,
    )
    torch.testing.assert_close(
        OrdinalDistribution(**kwargs, kind="sratio").category_probs(),
        OrdinalDistribution(**kwargs, kind="cratio").category_probs(),
    )


@pytest.mark.parametrize("kind", ("acat", "sratio", "cratio"))
def test_unrestricted_thresholds_are_valid_for_noncumulative(kind):
    probs = OrdinalDistribution(
        torch.tensor(0.2), torch.tensor([1.0, -2.0, 0.4]), kind=kind
    ).category_probs()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


@pytest.mark.parametrize(
    "cutpoints", ([1.0, -1.0], [0.0, 0.0], [0.0, float("inf")], [0.0, float("nan")])
)
def test_cumulative_rejects_invalid_thresholds(cutpoints):
    with pytest.raises(ValueError, match="finite|increasing"):
        OrderedLogistic(torch.tensor(0.2), torch.tensor(cutpoints))


@pytest.mark.parametrize("disc", (0.0, -1.0, float("nan"), float("inf")))
def test_rejects_bad_discrimination(disc):
    with pytest.raises(ValueError, match="finite|positive"):
        OrdinalDistribution(
            torch.tensor(0.0), torch.tensor([-1.0, 1.0]), discrimination=disc
        )


def test_cumulative_checks_effect_adjusted_order_and_transition_shape():
    with pytest.raises(ValueError, match="increasing"):
        OrdinalDistribution(
            torch.tensor(0.0),
            torch.tensor([-1.0, 1.0]),
            category_effects=torch.tensor([-2.0, 2.0]),
        )
    with pytest.raises(ValueError, match="K-1"):
        OrdinalDistribution(
            torch.tensor(0.0), torch.tensor([-1.0, 1.0]), category_effects=torch.ones(3)
        )


@pytest.mark.parametrize(
    "value",
    (
        torch.tensor(-1),
        torch.tensor(3),
        torch.tensor(0.5),
        torch.tensor(float("inf")),
        torch.tensor(True),
    ),
)
def test_invalid_response_is_never_clamped(value):
    dist = OrderedProbit(torch.tensor(0.0), torch.tensor([-1.0, 1.0]))
    with pytest.raises(ValueError):
        dist.log_prob(value)
    unchecked = OrderedProbit(
        torch.tensor(0.0), torch.tensor([-1.0, 1.0]), validate_args=False
    )
    if value.dtype == torch.bool:
        with pytest.raises(ValueError):
            unchecked.log_prob(value)
    else:
        assert unchecked.log_prob(value).isneginf()


@pytest.mark.parametrize("hu", (0.0, 0.3, 1.0))
def test_hurdle_is_separate_lowest_category_with_safe_endpoints(hu):
    eta = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    hurdle = torch.tensor(hu, dtype=torch.double, requires_grad=True)
    cuts = torch.tensor([-1.0, 1.0], dtype=torch.double)
    base = OrdinalDistribution(eta, cuts).category_probs()
    dist = OrdinalDistribution(eta, cuts, hurdle_prob=hurdle)
    expected = torch.cat((hurdle[None], (1 - hurdle) * base))
    torch.testing.assert_close(dist.category_probs(), expected)
    assert dist.num_categories == 4
    assert torch.isfinite(
        torch.autograd.grad(dist.category_probs().sum(), (eta, hurdle))[0]
    )
    if hu == 1.0:
        assert (dist.sample((20,)) == 0).all()
    if hu == 0.0:
        assert (dist.sample((20,)) >= 1).all()
