"""Count mixtures match independent pmfs, derivatives, and discrete draw oracles."""

import math
import random

import numpy as np
import pytest
import scipy.stats as stats
import torch

from quivers.continuous import (
    HurdleNegativeBinomial,
    ZeroInflatedBetaBinomial,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
)
from quivers.continuous._zero_inflated_count import _zero_atom_log_mass
from quivers.continuous.families import (
    ConditionalBetaBinomial,
    ConditionalBinomial,
    ConditionalZeroInflatedBetaBinomial,
    ConditionalZeroInflatedBinomial,
)
from quivers.continuous.spaces import Euclidean
from quivers.qiec.distributions import ReferenceBackend


CASES = [
    (ZeroInflatedNegativeBinomial, (0.3, 1.7, 0.4), stats.nbinom(1.7, 0.6)),
    (ZeroInflatedBinomial, (0.3, 5, 0.4), stats.binom(5, 0.4)),
    (ZeroInflatedBetaBinomial, (0.3, 5, 1.7, 2.3), stats.betabinom(5, 1.7, 2.3)),
    (HurdleNegativeBinomial, (0.3, 1.7, 0.4), stats.nbinom(1.7, 0.6)),
]


def _oracle(cls, parameters, base, points):
    zero = parameters[0]
    positive = (1 - zero) * base.pmf(points)
    if cls is HurdleNegativeBinomial:
        positive = positive / (1 - base.pmf(0))
        return np.where(points == 0, zero, positive)
    return np.where(points == 0, zero + positive, positive)


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_full_normalized_mass_and_moments_match_scipy(cls, parameters, base):
    law = cls(*(torch.tensor(value, dtype=torch.float64) for value in parameters))
    points = np.arange(
        2000 if cls in (HurdleNegativeBinomial, ZeroInflatedNegativeBinomial) else 6
    )
    expected = _oracle(cls, parameters, base, points)
    actual = law.log_prob(torch.tensor(points)).exp().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=3e-12)
    assert actual.sum() == pytest.approx(1.0, abs=2e-12)
    assert law.mean.item() == pytest.approx(np.dot(expected, points), abs=2e-12)
    assert law.variance.item() == pytest.approx(
        np.dot(expected, points**2) - np.dot(expected, points) ** 2, abs=2e-12
    )


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_interior_first_and_second_derivatives_match_numerical_oracles(
    cls, parameters, base
):
    variable_positions = [
        0,
        *range(
            1 if cls in (ZeroInflatedNegativeBinomial, HurdleNegativeBinomial) else 2,
            len(parameters),
        ),
    ]
    variables = tuple(
        torch.tensor(parameters[i], dtype=torch.float64, requires_grad=True)
        for i in variable_positions
    )

    def law(values):
        arguments = [torch.tensor(value, dtype=torch.float64) for value in parameters]
        for i, value in zip(variable_positions, values, strict=True):
            arguments[i] = value
        return cls(*arguments)

    points = torch.tensor([0, 1, 3])
    assert torch.autograd.gradcheck(
        lambda *values: law(values).log_prob(points), variables
    )
    assert torch.autograd.gradgradcheck(
        lambda *values: law(values).log_prob(points), variables
    )
    assert torch.autograd.gradcheck(lambda *values: law(values).mean, variables)
    assert torch.autograd.gradcheck(lambda *values: law(values).variance, variables)


@pytest.mark.parametrize("cls, parameters, base", CASES[:3])
@pytest.mark.parametrize("zero", [0.0, 1.0])
def test_zero_atom_endpoint_derivatives_match_analytic_mixture_mass(
    cls, parameters, base, zero
):
    probability = torch.tensor(zero, dtype=torch.float64, requires_grad=True)
    law = cls(
        probability,
        *(torch.tensor(value, dtype=torch.float64) for value in parameters[1:]),
    )
    actual = law.log_prob(torch.tensor(0))
    mass = base.pmf(0)
    assert actual.item() == pytest.approx(math.log(zero + (1 - zero) * mass), abs=2e-12)
    gradient = torch.autograd.grad(actual, probability)[0]
    assert gradient.item() == pytest.approx(
        (1 - mass) / (zero + (1 - zero) * mass), rel=2e-12
    )


@pytest.mark.parametrize(
    "cls, rest",
    [
        (ZeroInflatedBinomial, (0, 0.0)),
        (ZeroInflatedBinomial, (0, 1.0)),
        (ZeroInflatedBetaBinomial, (0, 0.3, 0.7)),
        (ZeroInflatedNegativeBinomial, (0.0, 0.5)),
        (ZeroInflatedNegativeBinomial, (2.0, 0.0)),
    ],
)
@pytest.mark.parametrize("zero", [0.0, 0.4, 1.0])
def test_degenerate_count_component_has_exact_zero_mass_and_zero_inflation_derivative(
    cls, rest, zero
):
    probability = torch.tensor(zero, dtype=torch.float64, requires_grad=True)
    law = cls(
        probability, *(torch.tensor(value, dtype=torch.float64) for value in rest)
    )
    score = law.log_prob(torch.tensor(0))
    assert score.item() == 0.0
    assert torch.autograd.grad(score, probability)[0].item() == 0.0
    assert law.mean.item() == law.variance.item() == 0.0
    assert not law.sample((1000,)).any()


def test_extreme_base_zero_mass_remains_log_stable_and_inactive_gradients_do_not_nan():
    zero = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    law = ZeroInflatedNegativeBinomial(
        zero,
        torch.tensor(10000.0, dtype=torch.float64),
        torch.tensor(0.9, dtype=torch.float64),
    )
    zero_score = law.log_prob(torch.tensor(0))
    assert zero_score.item() == pytest.approx(10000 * math.log(0.1), abs=1e-10)
    assert torch.isposinf(torch.autograd.grad(zero_score, zero)[0])
    positive_score = law.log_prob(torch.tensor(1))
    assert torch.isfinite(positive_score)
    assert torch.autograd.grad(positive_score, zero)[0].item() == -1.0


def test_zero_atom_primitive_supports_broadcast_and_torch_func_jacrev():
    zero = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    log_mass = torch.tensor([-0.7, -1.2, -2.0], dtype=torch.float64)
    jacobian = torch.func.jacrev(lambda p: _zero_atom_log_mass(p, log_mass))(zero)
    mass = log_mass.exp()
    expected = torch.diag((1 - mass) / (zero + (1 - zero) * mass))
    torch.testing.assert_close(jacobian, expected)
    assert _zero_atom_log_mass(zero[:, None], log_mass).shape == (3, 3)


@pytest.mark.parametrize("shape", [0.2, 2.0, 30.0])
def test_hurdle_probability_zero_limit_has_correct_value_moments_and_derivatives(shape):
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    law = HurdleNegativeBinomial(
        torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(shape, dtype=torch.float64),
        p,
    )
    torch.testing.assert_close(
        law.log_prob(torch.tensor([0, 1])).exp(),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    assert torch.isneginf(law.log_prob(torch.tensor(2)))
    assert law.mean.item() == pytest.approx(0.8)
    assert law.variance.item() == pytest.approx(0.16)
    assert torch.autograd.grad(law.log_prob(torch.tensor(1)), p, retain_graph=True)[
        0
    ].item() == pytest.approx(-(shape + 1) / 2)
    assert torch.autograd.grad(law.mean, p)[0].item() == pytest.approx(
        0.8 * (shape + 1) / 2
    )
    draws = law.sample((1000,))
    assert ((draws == 0) | (draws == 1)).all()


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_draws_have_independent_zero_probability_mean_and_cdf(cls, parameters, base):
    torch.manual_seed(917)
    law = cls(*(torch.tensor(value, dtype=torch.float64) for value in parameters))
    count = 40000
    draws = law.sample((count,))
    assert draws.dtype == torch.long and not draws.requires_grad
    expected_mass = _oracle(cls, parameters, base, np.arange(2000))
    for boundary in (0, 1, 3, 6):
        probability = expected_mass[: boundary + 1].sum()
        standard_error = math.sqrt(probability * (1 - probability) / count)
        assert (
            abs((draws <= boundary).double().mean().item() - probability)
            <= 6 * standard_error + 1 / count
        )
    assert abs(draws.double().mean().item() - law.mean.item()) <= 6 * math.sqrt(
        law.variance.item() / count
    )


def test_tiny_shape_heavy_tail_hurdle_sampling_does_not_use_negative_binomial_rejection(
    monkeypatch,
):
    from quivers.continuous import _zero_inflated_count as implementation

    def forbidden(*args, **kwargs):
        raise AssertionError("tiny positive mass must not reject ordinary NB zeros")

    monkeypatch.setattr(implementation.ClosedNegativeBinomial, "sample", forbidden)
    torch.manual_seed(15)
    shape, probability = 1e-12, 0.9999
    law = HurdleNegativeBinomial(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(shape, dtype=torch.float64),
        torch.tensor(probability, dtype=torch.float64),
    )
    draws = law.sample((40000,))
    assert (draws > 0).all()
    # The r->0 positive NB limit is a logarithmic series, and the error
    # at this r is much smaller than the Monte Carlo standard error.
    for boundary in (1, 3, 10):
        expected = stats.logser.cdf(boundary, probability)
        assert abs(
            (draws <= boundary).double().mean().item() - expected
        ) < 6 * math.sqrt(expected * (1 - expected) / draws.numel())


@pytest.mark.parametrize("cls, parameters, base", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_expand_support_and_parameter_dtype_are_preserved(cls, parameters, base, dtype):
    tensors = [torch.tensor(value, dtype=dtype) for value in parameters]
    law = cls(*tensors).expand(torch.Size((2, 3)))
    assert law.batch_shape == (2, 3)
    assert law.sample((4,)).shape == (4, 2, 3)
    assert law.log_prob(torch.zeros(2, 3, dtype=torch.long)).dtype == torch.float64
    assert law.mean.dtype == law.variance.dtype == torch.float64
    assert not law.has_rsample
    with pytest.raises(NotImplementedError):
        law.rsample()
    for value in (-1.0, 0.5, float("inf"), float("nan")):
        assert torch.isneginf(
            cls(*tensors, validate_args=False).log_prob(torch.tensor(value))
        ).all()


@pytest.mark.parametrize(
    "cls, parameters",
    [
        (ZeroInflatedBinomial, (0.2, 1.5, 0.5)),
        (ZeroInflatedBinomial, (-0.1, 1.0, 0.5)),
        (ZeroInflatedBetaBinomial, (0.2, 1.0, 0.0, 1.0)),
        (ZeroInflatedNegativeBinomial, (0.2, 1.0, 1.0)),
        (HurdleNegativeBinomial, (0.2, 0.0, 0.5)),
    ],
)
def test_invalid_parameters_fail_explicitly(cls, parameters):
    with pytest.raises(ValueError):
        cls(*(torch.tensor(value) for value in parameters), validate_args=True)


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_reference_density_and_sampling_agree_with_independent_count_oracles(
    cls, parameters, base
):
    backend = ReferenceBackend()
    named = dict(zip(cls.arg_constraints, parameters, strict=True))
    for value in range(10):
        expected = _oracle(cls, parameters, base, np.asarray(value)).item()
        actual = backend.family_log_prob(cls.__name__, named, value)
        assert actual == pytest.approx(
            math.log(expected) if expected else -math.inf, abs=3e-12
        )
    rng = random.Random(712)
    samples = np.array(
        [backend.family_sample(cls.__name__, named, rng) for _ in range(12000)]
    )
    mean = cls(
        *(torch.tensor(value, dtype=torch.float64) for value in parameters)
    ).mean.item()
    assert abs(samples.mean() - mean) < 6 * math.sqrt(samples.var() / samples.size)
    for invalid in (-1, 0.5, float("nan")):
        assert backend.family_log_prob(cls.__name__, named, invalid) == -math.inf


@pytest.mark.parametrize(
    "cls",
    [
        ConditionalZeroInflatedBinomial,
        ConditionalZeroInflatedBetaBinomial,
        ConditionalBetaBinomial,
        ConditionalBinomial,
    ],
)
def test_fixed_trial_conditional_heads_support_zero_trials_and_integer_observations(
    cls,
):
    domain, codomain = Euclidean(name="X", dim=1), Euclidean(name="Y", dim=2)
    law = cls(domain, codomain, total_count=0)
    x = torch.zeros(3, 1)
    score = law.log_prob(x, torch.zeros(3, 2, dtype=torch.long))
    torch.testing.assert_close(score, torch.zeros_like(score))
    assert not law.sample(x).any()


def test_zero_atom_primitive_supports_forward_reverse_and_batched_sensitivities():
    zero = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)
    log_mass = torch.tensor([-0.7, -1.2, -2.0], dtype=torch.float64)

    def oracle(zero, base):
        return (zero + (1 - zero) * base.exp()).log()

    for transform in (torch.func.jacrev, torch.func.jacfwd):
        torch.testing.assert_close(
            transform(_zero_atom_log_mass, argnums=(0, 1))(zero, log_mass),
            transform(oracle, argnums=(0, 1))(zero, log_mass),
        )
    tangents = (
        torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64),
        torch.tensor([-0.3, 0.2, 0.1], dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.func.jvp(_zero_atom_log_mass, (zero, log_mass), tangents),
        torch.func.jvp(oracle, (zero, log_mass), tangents),
    )
    torch.testing.assert_close(
        torch.func.vmap(torch.func.grad(_zero_atom_log_mass, argnums=(0, 1)))(
            zero, log_mass
        ),
        torch.func.vmap(torch.func.grad(oracle, argnums=(0, 1)))(zero, log_mass),
    )
    # A base-only sensitivity must remain one when its mass underflows.
    extreme = (
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-1000.0, dtype=torch.float64),
    )
    _, tangent = torch.func.jvp(
        _zero_atom_log_mass,
        extreme,
        (torch.zeros_like(extreme[0]), torch.ones_like(extreme[1])),
    )
    assert tangent.item() == 1.0


@pytest.mark.parametrize(
    "shape, probability", [(1e-12, 0.9999), (1e-12, 1e-12), (1e-300, 1e-300)]
)
def test_reference_hurdle_tiny_parameters_remain_finite_and_normalized(
    shape, probability
):
    law = HurdleNegativeBinomial(
        torch.tensor(0.2, dtype=torch.float64),
        torch.tensor(shape, dtype=torch.float64),
        torch.tensor(probability, dtype=torch.float64),
    )
    backend = ReferenceBackend()
    arguments = {"zero_prob": 0.2, "total_count": shape, "probs": probability}
    for count in (0, 1, 2, 10):
        actual = backend.family_log_prob("HurdleNegativeBinomial", arguments, count)
        assert actual == pytest.approx(
            law.log_prob(torch.tensor(count)).item(), abs=3e-12
        )


@pytest.mark.parametrize("count", [1_000_001, 33_554_433, 2**31 - 1])
@pytest.mark.parametrize("cls", [ZeroInflatedBinomial, ZeroInflatedBetaBinomial])
def test_large_integer_trials_survive_default_float32_heads_with_double_scores(
    count, cls
):
    from scipy.special import digamma

    zero = torch.tensor(0.3, requires_grad=True)
    if cls is ZeroInflatedBinomial:
        p = torch.tensor(0.5, requires_grad=True)
        law = cls(zero, count, p)
        expected = stats.binom.logpmf(count // 2, count, p.item())
        variables, derivatives = (zero, p), (-1 / (1 - zero.item()), -2.0)
    else:
        a, b = (
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        )
        law = cls(zero, count, a, b)
        k = count // 2
        expected = stats.betabinom.logpmf(k, count, 2.0, 3.0)
        variables = (zero, a, b)
        derivatives = (
            -1 / (1 - zero.item()),
            digamma(2 + k) - digamma(5 + count) - digamma(2) + digamma(5),
            digamma(3 + count - k) - digamma(5 + count) - digamma(3) + digamma(5),
        )
    score = law.log_prob(torch.tensor(count // 2))
    assert law.total_count.item() == count
    assert score.dtype == torch.float64
    # Independent SciPy double-gamma arithmetic differs by several ulps
    # of its large intermediate terms at the signed32-bit trial limit.
    assert score.item() == pytest.approx(math.log1p(-zero.item()) + expected, abs=8e-6)
    for gradient, derivative in zip(
        torch.autograd.grad(score, variables), derivatives, strict=True
    ):
        assert gradient.dtype == torch.float32
        assert gradient.item() == pytest.approx(derivative, abs=2e-6)


@pytest.mark.parametrize("count", [1_000_001, 33_554_433, 2**31 - 1])
def test_closed_count_kernels_preserve_large_trials_and_poisson_center_mass(count):
    from quivers.continuous._closed_probability import (
        ClosedBinomial,
        ClosedNegativeBinomial,
        ClosedPoisson,
    )

    probability = torch.tensor(0.5, requires_grad=True)
    law = ClosedBinomial(count, probs=probability).expand((2,))
    assert law.total_count[0].item() == count
    score = law.log_prob(torch.tensor([count // 2, 0]))
    assert score.dtype == torch.float64
    assert score[0].item() == pytest.approx(
        stats.binom.logpmf(count // 2, count, 0.5), abs=8e-6
    )
    assert score[1].item() == pytest.approx(count * math.log(0.5), rel=1e-14)
    poisson = ClosedPoisson(count)
    center = poisson.log_prob(torch.tensor(count))
    stirling = (
        -0.5 * math.log(2 * math.pi * count) - 1 / (12 * count) + 1 / (360 * count**3)
    )
    assert center.item() == pytest.approx(stirling, abs=8e-6)
    assert (
        ClosedNegativeBinomial(1e20, probs=torch.tensor(0.0)).sample((100,)).eq(0).all()
    )
    assert (
        ClosedNegativeBinomial(2.0, probs=torch.tensor(0.0))
        .expand((2,))
        .mean.eq(0)
        .all()
    )


def test_zero_count_atom_is_exact_for_all_structural_probabilities():
    probabilities = torch.linspace(0, 1, 11, dtype=torch.float64, requires_grad=True)
    law = ZeroInflatedBinomial(probabilities, 0, torch.tensor(0.5))
    scores = law.log_prob(torch.zeros(11, dtype=torch.long))
    assert scores.eq(0).all()
    assert torch.autograd.grad(scores.sum(), probabilities)[0].eq(0).all()


@pytest.mark.parametrize("n", [2**53 + 1, 2**63 - 1])
@pytest.mark.parametrize("tensor_input", [False, True])
@pytest.mark.parametrize("family", ["binomial", "zi_binomial", "zi_beta_binomial"])
def test_unrepresentable_trial_inputs_rejected_before_float_cast(
    n, tensor_input, family
):
    from quivers.continuous._closed_probability import ClosedBinomial

    source = torch.tensor(n, dtype=torch.int64) if tensor_input else n
    p = torch.tensor(0.4)
    with pytest.raises(ValueError, match=r"2\*\*53"):
        if family == "binomial":
            ClosedBinomial(source, probs=p)
        elif family == "zi_binomial":
            ZeroInflatedBinomial(p, source, p)
        else:
            ZeroInflatedBetaBinomial(p, source, p, p)


@pytest.mark.parametrize(
    "n, binomial_logmass, beta_binomial_logmass, poisson_logmass",
    [
        (1_000_001, -7.1335478816253645, -13.468429427407295, -7.826694395519810),
        (33_554_433, -8.89013114689694545, -16.981596229604187, -9.583278307588677),
        (2**31 - 1, -10.96957265144029505, -21.140479246905151, -11.662719831689799),
        (2**53, -18.594191637483278, -36.389717219121502, -19.287338818043223),
    ],
)
def test_large_count_central_scores_match_70_digit_gamma_oracles(
    n, binomial_logmass, beta_binomial_logmass, poisson_logmass
):
    # Constants were independently evaluated with70decimal-digit log-gamma arithmetic.
    from quivers.continuous._closed_probability import ClosedBinomial, ClosedPoisson
    from quivers.continuous._zero_inflated_count import BetaBinomial

    p = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    law = ClosedBinomial(n, probs=p)
    k = torch.tensor(n // 2)
    score = law.log_prob(k)
    assert law.total_count.item() == n
    assert score.item() == pytest.approx(binomial_logmass, abs=6e-14)
    assert torch.autograd.grad(score, p)[0].item() == pytest.approx(
        0 if n % 2 == 0 else -2, abs=2e-13
    )
    logits = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    logit_score = ClosedBinomial(n, logits=logits).log_prob(k)
    assert logit_score.item() == pytest.approx(binomial_logmass, abs=6e-14)
    assert torch.autograd.grad(logit_score, logits)[0].item() == pytest.approx(
        0 if n % 2 == 0 else -0.5, abs=6e-14
    )
    bb = BetaBinomial(
        n,
        torch.tensor(1.7, dtype=torch.float64),
        torch.tensor(2.3, dtype=torch.float64),
    )
    assert bb.log_prob(k).item() == pytest.approx(beta_binomial_logmass, abs=6e-14)
    poisson = ClosedPoisson(torch.tensor(n, dtype=torch.float64))
    assert poisson.log_prob(torch.tensor(n)).item() == pytest.approx(
        poisson_logmass, abs=6e-14
    )


@pytest.mark.parametrize("n", [1_000_001, 2**31 - 1, 2**53])
def test_large_binomial_tails_remain_nonpositive_with_finite_head_gradients(n):
    from quivers.continuous._closed_probability import ClosedBinomial

    p = torch.tensor(
        [0.001, 0.25, 0.5, 0.75, 0.999], dtype=torch.float64, requires_grad=True
    )
    k = torch.tensor([0, 1, n // 2, n - 1, n])
    scores = ClosedBinomial(n, probs=p).log_prob(k)
    assert bool(torch.isfinite(scores).all())
    assert bool((scores <= 0).all())
    assert bool(torch.isfinite(torch.autograd.grad(scores.sum(), p)[0]).all())


def test_large_real_negative_binomial_shape_density_avoids_gamma_cancellation():
    from quivers.continuous._closed_probability import ClosedNegativeBinomial

    r = torch.tensor(1e16, dtype=torch.float64)
    p = torch.tensor(1e-12, dtype=torch.float64)
    count = torch.tensor(10000)
    # Independent80digit gamma oracle for the exact binary64 parameters.
    expected = -5.524117052526595
    score = ClosedNegativeBinomial(r, probs=p).log_prob(count)
    assert score.item() == pytest.approx(expected, abs=2e-12)
    logits = torch.logit(p)
    assert ClosedNegativeBinomial(r, logits=logits).log_prob(
        count
    ).item() == pytest.approx(expected, abs=2e-12)
    zero_prob = torch.tensor(0.3, dtype=torch.float64)
    for cls in (ZeroInflatedNegativeBinomial, HurdleNegativeBinomial):
        assert cls(zero_prob, r, p).log_prob(count).item() == pytest.approx(
            expected + math.log(0.7), abs=2e-12
        )


def test_double_count_scores_keep_float32_head_gradients_in_original_dtype():
    from quivers.continuous._closed_probability import (
        ClosedBinomial,
        ClosedNegativeBinomial,
        ClosedPoisson,
    )

    p = torch.tensor(0.3, requires_grad=True)
    mean = torch.tensor(2.3, requires_grad=True)
    score = (
        ClosedBinomial(5, probs=p).log_prob(torch.tensor(2))
        + ClosedNegativeBinomial(2.3, probs=p).log_prob(torch.tensor(2))
        + ClosedPoisson(mean).log_prob(torch.tensor(2))
    )
    assert score.dtype == torch.float64
    gradients = torch.autograd.grad(score, (p, mean))
    assert all(g.dtype == torch.float32 for g in gradients)
    expected_p = (
        2 / p.detach().double()
        - 3 / (1 - p.detach().double())
        + 2 / p.detach().double()
        - 2.3 / (1 - p.detach().double())
    )
    torch.testing.assert_close(gradients[0], expected_p.float())
    torch.testing.assert_close(gradients[1], (2 / mean.detach() - 1))


@pytest.mark.parametrize(
    "n, binomial_logmass, beta_binomial_logmass, poisson_logmass",
    [
        (1_000_001, -7.1335478816253645, -13.468429427407295, -7.826694395519810),
        (33_554_433, -8.89013114689694545, -16.981596229604187, -9.583278307588677),
        (2**31 - 1, -10.96957265144029505, -21.140479246905151, -11.662719831689799),
        (2**53, -18.594191637483278, -36.389717219121502, -19.287338818043223),
    ],
)
def test_scalar_reference_large_count_densities_match_high_precision_oracles(
    n, binomial_logmass, beta_binomial_logmass, poisson_logmass
):
    backend = ReferenceBackend()
    assert backend.family_log_prob(
        "Binomial", dict(total_count=n, probs=0.5), n // 2
    ) == pytest.approx(binomial_logmass, abs=8e-14)
    assert backend.family_log_prob("Poisson", dict(rate=n), n) == pytest.approx(
        poisson_logmass, abs=8e-14
    )
    assert backend.family_log_prob(
        "BetaBinomial",
        dict(total_count=n, concentration1=1.7, concentration0=2.3),
        n // 2,
    ) == pytest.approx(beta_binomial_logmass, abs=8e-14)
    for family, arguments, expected in [
        ("ZeroInflatedBinomial", dict(total_count=n, probs=0.5), binomial_logmass),
        (
            "ZeroInflatedBetaBinomial",
            dict(total_count=n, concentration1=1.7, concentration0=2.3),
            beta_binomial_logmass,
        ),
    ]:
        assert backend.family_log_prob(
            family, dict(zero_prob=0.3, **arguments), n // 2
        ) == pytest.approx(expected + math.log(0.7), abs=8e-14)


def test_scalar_reference_large_real_negative_binomial_shape_density():
    backend = ReferenceBackend()
    expected = -5.524117052526595
    arguments = dict(total_count=1e16, probs=1e-12)
    assert backend.family_log_prob(
        "NegativeBinomial", arguments, 10000
    ) == pytest.approx(expected, abs=2e-12)
    for family in ("ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial"):
        assert backend.family_log_prob(
            family, dict(zero_prob=0.3, **arguments), 10000
        ) == pytest.approx(expected + math.log(0.7), abs=2e-12)


@pytest.mark.parametrize(
    "family", ["ZeroInflatedNegativeBinomial", "HurdleNegativeBinomial"]
)
def test_reference_large_rate_sampling_uses_constant_work_gamma_poisson(
    monkeypatch, family
):
    import quivers.qiec._count_mixtures as reference

    def forbid_jumps(*args):
        raise AssertionError("large-rate NB sampling must not enumerate compound jumps")

    monkeypatch.setattr(reference, "_logseries", forbid_jumps)
    backend = ReferenceBackend()
    rng = random.Random(417)
    arguments = dict(zero_prob=0.0, total_count=1e16, probs=1e-12)
    draws = np.array(
        [backend.family_sample(family, arguments, rng) for _ in range(2000)]
    )
    assert draws.mean() == pytest.approx(10000.0, abs=10.0)
    assert np.all(draws > 0)


@pytest.mark.parametrize(
    "family",
    [
        "Poisson",
        "NegativeBinomial",
        "ZeroInflatedNegativeBinomial",
        "HurdleNegativeBinomial",
        "ZeroInflatedPoisson",
        "HurdlePoisson",
    ],
)
def test_native_count_sampling_refuses_unrepresentable_intensities(family):
    from quivers.continuous._closed_probability import (
        ClosedPoisson,
        ClosedNegativeBinomial,
    )
    from quivers.continuous import ZeroInflatedPoisson, HurdlePoisson

    classes = {
        "Poisson": lambda: ClosedPoisson(1e20),
        "NegativeBinomial": lambda: ClosedNegativeBinomial(
            1e20, probs=torch.tensor(0.5)
        ),
        "ZeroInflatedPoisson": lambda: ZeroInflatedPoisson(
            torch.tensor(0.0), torch.tensor(1e20)
        ),
        "HurdlePoisson": lambda: HurdlePoisson(torch.tensor(0.0), torch.tensor(1e20)),
        "ZeroInflatedNegativeBinomial": lambda: ZeroInflatedNegativeBinomial(
            0.0, 1e20, 0.5
        ),
        "HurdleNegativeBinomial": lambda: HurdleNegativeBinomial(0.0, 1e20, 0.5),
    }
    law = classes[family]()
    with pytest.raises(RuntimeError, match=r"2\*\*53"):
        law.sample((3,))
    assert bool(torch.isfinite(law.mean).all())
    assert bool(torch.isfinite(law.log_prob(torch.tensor(1))).all())


@pytest.mark.parametrize(
    "family",
    [
        "Poisson",
        "NegativeBinomial",
        "ZeroInflatedNegativeBinomial",
        "HurdleNegativeBinomial",
    ],
)
def test_native_sampler_checks_poisson_output_before_integer_cast(monkeypatch, family):
    from quivers.continuous._closed_probability import (
        ClosedPoisson,
        ClosedNegativeBinomial,
    )

    monkeypatch.setattr(
        torch, "poisson", lambda rate: torch.full_like(rate, float(2**63))
    )
    laws = {
        "Poisson": ClosedPoisson(2.0),
        "NegativeBinomial": ClosedNegativeBinomial(2.0, probs=torch.tensor(0.5)),
        "ZeroInflatedNegativeBinomial": ZeroInflatedNegativeBinomial(0.0, 2.0, 0.5),
        "HurdleNegativeBinomial": HurdleNegativeBinomial(0.0, 2.0, 0.5),
    }
    with pytest.raises(RuntimeError, match=r"2\*\*53"):
        laws[family].sample((3,))


@pytest.mark.parametrize(
    "family",
    [
        "ZeroInflatedNegativeBinomial",
        "HurdleNegativeBinomial",
        "ZeroInflatedPoisson",
        "HurdlePoisson",
    ],
)
def test_count_atom_sampling_skips_unreachable_huge_component(family):
    from quivers.continuous import ZeroInflatedPoisson, HurdlePoisson

    if family.endswith("NegativeBinomial"):
        cls = globals()[family]
        law = cls(
            torch.tensor([1.0, 0.0]),
            torch.tensor([1e20, 2.0]),
            torch.tensor([0.5, 0.5]),
        )
        atom = cls(1.0, 1e20, 0.5)
    else:
        cls = {
            "ZeroInflatedPoisson": ZeroInflatedPoisson,
            "HurdlePoisson": HurdlePoisson,
        }[family]
        law = cls(torch.tensor([1.0, 0.0]), torch.tensor([1e20, 2.0]))
        atom = cls(torch.tensor(1.0), torch.tensor(1e20))
    assert torch.equal(atom.sample((10,)), torch.zeros(10, dtype=torch.long))
    draws = law.sample((20,))
    assert draws.dtype == torch.long
    assert bool((draws[:, 0] == 0).all())
    assert bool((draws[:, 1] >= 0).all())


@pytest.mark.parametrize(
    "family, arguments",
    [
        ("Poisson", dict(rate=1e20)),
        ("NegativeBinomial", dict(total_count=1e20, probs=0.5)),
        (
            "ZeroInflatedNegativeBinomial",
            dict(zero_prob=0.0, total_count=1e20, probs=0.5),
        ),
        ("HurdleNegativeBinomial", dict(zero_prob=0.0, total_count=1e20, probs=0.5)),
        ("ZeroInflatedPoisson", dict(zero_prob=0.0, rate=1e20)),
        ("HurdlePoisson", dict(zero_prob=0.0, rate=1e20)),
    ],
)
def test_reference_sampling_refuses_unrepresentable_intensities(family, arguments):
    from quivers.qiec.reference_support import DistributionError

    with pytest.raises(DistributionError, match=r"2\*\*53"):
        ReferenceBackend().family_sample(family, arguments, random.Random(71))
    if "zero_prob" in arguments:
        assert (
            ReferenceBackend().family_sample(
                family, dict(arguments, zero_prob=1.0), random.Random(71)
            )
            == 0
        )


@pytest.mark.parametrize(
    "family", ["Poisson", "NegativeBinomial", "ZeroInflatedPoisson", "HurdlePoisson"]
)
def test_reference_large_mean_sampling_uses_correct_constant_work_law(family):
    arguments = (
        dict(total_count=1e16, probs=1e-12)
        if family == "NegativeBinomial"
        else dict(rate=10000.0)
    )
    if family in ("ZeroInflatedPoisson", "HurdlePoisson"):
        arguments["zero_prob"] = 0.0
    rng = random.Random(24)
    draws = [
        ReferenceBackend().family_sample(family, arguments, rng) for _ in range(2000)
    ]
    assert np.mean(draws) == pytest.approx(10000.0, abs=10.0)


def test_negative_binomial_extreme_logits_reach_explicit_capacity_refusal():
    from quivers.continuous._closed_probability import ClosedNegativeBinomial

    with pytest.raises(RuntimeError, match=r"2\*\*53"):
        ClosedNegativeBinomial(2.0, logits=torch.tensor(1000.0)).sample((3,))
    assert torch.equal(
        ClosedNegativeBinomial(0.0, logits=torch.tensor(1000.0)).sample((3,)),
        torch.zeros(3, dtype=torch.float64),
    )


def test_guarded_count_samplers_preserve_double_dtype_and_original_device():
    from quivers.continuous._closed_probability import (
        ClosedNegativeBinomial,
        ClosedPoisson,
    )

    for law in (
        ClosedPoisson(torch.tensor(2.0)),
        ClosedNegativeBinomial(2.0, probs=torch.tensor(0.4)),
        ClosedNegativeBinomial(2.0, logits=torch.tensor(0.1)),
    ):
        draws = law.sample((2, 3))
        assert draws.shape == (2, 3)
        assert draws.dtype == torch.float64
        assert draws.device == law.mean.device
        assert bool(((draws >= 0) & (draws == draws.floor())).all())
