"""Forward draws from the LKJ families follow the LKJ distribution.

Under LKJ(eta) in dimension d every off-diagonal correlation ``r``
satisfies ``(r + 1) / 2 ~ Beta(eta - 1 + d / 2, eta - 1 + d / 2)``. The
tests hold every forward entry point (the shared onion sampler, the
compiled program's draw, the fixed and variable-parameter factories,
the conditional family, ``LKJCorrelationFactor``, and a plate over the
family) to those marginals with a Kolmogorov-Smirnov statistic, and
check that the same statistic rejects the torch sampler, whose rows
after the first are too dispersed. The thresholds, sample sizes, and
seeds are fixed here, before any draw is made:

* one-sample KS: ``N = 20000`` draws, reject when the statistic exceeds
  the asymptotic critical value at ``alpha = 1e-6``, about ``0.019``;
* two-sample KS: reject above the asymptotic critical value at
  ``alpha = 1e-6`` for the two sample sizes;
* moment and entropy checks: reject beyond five standard errors.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest
import torch
import torch.distributions as D
from scipy import stats

from quivers.continuous._lkj import OnionLKJCholesky, lkj_cholesky_rsample
from quivers.continuous.families import ConditionalLKJCholesky, LKJCorrelationFactor
from quivers.continuous.inline import _dim_dependent_builder, make_fixed_lkj_cholesky
from quivers.continuous.plate import PlateDraw
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import CholeskyFactor, Euclidean
from quivers.dsl import loads
from quivers.qiec.distributions import ReferenceBackend

N = 20000
ALPHA = 1e-6
SEED = 1234
ONE_SAMPLE_CRITICAL = math.sqrt(math.log(2.0 / ALPHA) / (2.0 * N))
GRID = [(eta, d) for d in (2, 3, 5) for eta in (0.5, 1.0, 2.0, 10.0)]

type Sampler = Callable[[float, int], torch.Tensor]


def _two_sample_critical(n: int, m: int) -> float:
    """The asymptotic two-sample KS critical value at ``ALPHA``.

    Parameters
    ----------
    n : int
        The first sample's size.
    m : int
        The second sample's size.

    Returns
    -------
    float
        The statistic above which the two samples are judged different.
    """
    return math.sqrt(math.log(2.0 / ALPHA) / 2.0 * (n + m) / (n * m))


def _two_sample_statistic(left: torch.Tensor, right: torch.Tensor) -> float:
    """The two-sample Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    left : torch.Tensor
        The first sample, one-dimensional.
    right : torch.Tensor
        The second sample, one-dimensional.

    Returns
    -------
    float
        The largest gap between the two empirical distribution functions.
    """
    left = left.sort().values
    right = right.sort().values
    points = torch.cat([left, right])
    left_cdf = torch.searchsorted(left, points, right=True) / left.numel()
    right_cdf = torch.searchsorted(right, points, right=True) / right.numel()
    return float((left_cdf - right_cdf).abs().max())


def _correlations(factors: torch.Tensor) -> torch.Tensor:
    """The strictly lower correlations of a batch of Cholesky factors.

    Parameters
    ----------
    factors : torch.Tensor
        Factors of shape ``(n, d, d)``.

    Returns
    -------
    torch.Tensor
        Correlations of shape ``(n, d (d - 1) / 2)``.
    """
    d = factors.shape[-1]
    corr = factors.double() @ factors.double().transpose(-1, -2)
    rows, cols = torch.tril_indices(d, d, offset=-1)
    return corr[:, rows, cols]


def _worst_marginal_statistic(factors: torch.Tensor, eta: float) -> float:
    """The largest KS statistic of any correlation against its LKJ marginal.

    Parameters
    ----------
    factors : torch.Tensor
        Factors of shape ``(n, d, d)``.
    eta : float
        The LKJ concentration the factors should follow.

    Returns
    -------
    float
        The maximum over off-diagonal entries of the KS statistic of
        ``(r + 1) / 2`` against ``Beta(eta - 1 + d / 2, eta - 1 + d / 2)``.
    """
    d = factors.shape[-1]
    shape = eta - 1.0 + d / 2.0
    marginal = stats.beta(shape, shape)
    correlations = _correlations(factors)
    return max(
        float(stats.kstest(((column + 1.0) / 2.0).numpy(), marginal.cdf).statistic)
        for column in correlations.unbind(-1)
    )


def _assert_valid_factors(factors: torch.Tensor) -> None:
    """Assert that every factor is a correlation Cholesky factor.

    Parameters
    ----------
    factors : torch.Tensor
        Factors of shape ``(..., d, d)``.
    """
    assert torch.isfinite(factors).all()
    assert torch.equal(factors, factors.tril())
    assert (factors.diagonal(dim1=-2, dim2=-1) > 0).all()
    norms = factors.pow(2).sum(-1)
    tolerance = 1e-5 if factors.dtype == torch.float32 else 1e-12
    assert torch.allclose(norms, torch.ones_like(norms), atol=tolerance)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _shared_sampler(eta: float, d: int) -> torch.Tensor:
    torch.manual_seed(SEED)
    concentration = torch.tensor(eta, dtype=torch.float64)
    return OnionLKJCholesky(d, concentration).sample(torch.Size((N,)))


def _compiled_program(eta: float, d: int) -> torch.Tensor:
    source = (
        f"object Dim : FinSet {d}\n"
        f"object Factor : Real {d}\n\n"
        "program lkj_prior : Dim -> Factor\n"
        f"    sample chol : Dim <- LKJCholesky(concentration={eta})\n"
        "    return chol\n\n"
        "export lkj_prior\n"
    )
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    torch.manual_seed(SEED)
    draws = program.rsample(torch.zeros(N, 1))
    assert isinstance(draws, torch.Tensor)
    return draws.detach()


def _fixed_factory(eta: float, d: int) -> torch.Tensor:
    family = make_fixed_lkj_cholesky(eta, CholeskyFactor(name="L", dim=d))
    torch.manual_seed(SEED)
    draws = family.rsample(torch.zeros(N, 1)).detach()
    return draws.reshape(N, d, d)


def _variable_factory(eta: float, d: int) -> torch.Tensor:
    builder = _dim_dependent_builder("LKJCholesky", CholeskyFactor(name="L", dim=d))
    distribution = builder([torch.full((N,), eta, dtype=torch.float64)])
    torch.manual_seed(SEED)
    return distribution.rsample().detach()


def _correlation_factor(eta: float, d: int) -> torch.Tensor:
    prior = LKJCorrelationFactor(dim=d, eta=eta, domain=Euclidean(name="x", dim=1))
    torch.manual_seed(SEED)
    return prior.rsample(torch.zeros(N, 1)).detach().reshape(N, d, d)


def _plate_over_family(eta: float, d: int) -> torch.Tensor:
    family = make_fixed_lkj_cholesky(eta, CholeskyFactor(name="L", dim=d))
    plate = PlateDraw(index_size=N, family=family, domain=family.domain)
    torch.manual_seed(SEED)
    return plate.rsample(torch.zeros(1, 1)).detach().reshape(N, d, d)


def _conditional_family(eta: float, d: int) -> torch.Tensor:
    family = ConditionalLKJCholesky(
        Euclidean(name="x", dim=1), CholeskyFactor(name="L", dim=d)
    )
    # The concentration is softplus(raw) + 0.1; aim the raw value at eta.
    with torch.no_grad():
        for parameter in family.param_source.parameters():
            parameter.zero_()
        last_bias = [p for p in family.param_source.parameters() if p.ndim == 1][-1]
        last_bias.fill_(math.log(math.expm1(eta - 0.1)))
    torch.manual_seed(SEED)
    return family.rsample(torch.zeros(N, 1)).detach()


ENTRY_POINTS: dict[str, Sampler] = {
    "shared-sampler": _shared_sampler,
    "compiled-program": _compiled_program,
    "fixed-factory": _fixed_factory,
    "variable-factory": _variable_factory,
    "lkj-correlation-factor": _correlation_factor,
    "plate-draw": _plate_over_family,
    "conditional-family": _conditional_family,
}


@pytest.fixture
def torch_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every quivers LKJ draw through torch's own sampler."""

    def torch_sample(
        self: OnionLKJCholesky, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        return D.LKJCholesky.sample(self, sample_shape)

    monkeypatch.setattr(OnionLKJCholesky, "rsample", torch_sample)
    monkeypatch.setattr(OnionLKJCholesky, "sample", torch_sample)


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("eta", "d"), GRID)
def test_every_correlation_follows_its_lkj_beta_marginal(eta: float, d: int) -> None:
    factors = _shared_sampler(eta, d)
    _assert_valid_factors(factors)
    assert _worst_marginal_statistic(factors, eta) < ONE_SAMPLE_CRITICAL


@pytest.mark.parametrize(("eta", "d"), [(eta, d) for eta, d in GRID if d >= 3])
def test_the_marginal_check_rejects_torchs_sampler(
    eta: float, d: int, torch_sampler: None
) -> None:
    factors = _shared_sampler(eta, d)
    assert _worst_marginal_statistic(factors, eta) > 1.5 * ONE_SAMPLE_CRITICAL


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize(("eta", "d"), [(2.0, 4), (0.5, 3)])
def test_every_forward_entry_point_draws_lkj_marginals(
    entry: str, eta: float, d: int
) -> None:
    factors = ENTRY_POINTS[entry](eta, d)
    assert factors.shape == (N, d, d)
    _assert_valid_factors(factors)
    assert _worst_marginal_statistic(factors, eta) < ONE_SAMPLE_CRITICAL


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_every_forward_entry_point_draws_through_the_onion_sampler(
    entry: str, torch_sampler: None
) -> None:
    factors = ENTRY_POINTS[entry](2.0, 4)
    assert _worst_marginal_statistic(factors, 2.0) > 1.5 * ONE_SAMPLE_CRITICAL


def test_second_moments_match_the_lkj_value() -> None:
    eta, d = 2.0, 4
    correlations = _correlations(_shared_sampler(eta, d))
    squared = correlations.pow(2)
    expected = 1.0 / (2.0 * eta + d - 1.0)
    standard_error = squared.std(0) / math.sqrt(N)
    assert ((squared.mean(0) - expected).abs() < 5.0 * standard_error).all()


# ---------------------------------------------------------------------------
# Joint law at eta = 1, d = 3 against rejection sampling
# ---------------------------------------------------------------------------


def _uniform_correlations_by_rejection(count: int) -> torch.Tensor:
    """Draw uniform 3-by-3 correlation matrices by rejection.

    Parameters
    ----------
    count : int
        The number of matrices.

    Returns
    -------
    torch.Tensor
        Matrices of shape ``(count, 3, 3)``, uniform over the positive
        definite 3-by-3 correlation matrices, which is LKJ(1).
    """
    generator = torch.Generator().manual_seed(SEED + 1)
    accepted: list[torch.Tensor] = []
    total = 0
    while total < count:
        r = torch.rand(4 * count, 3, generator=generator, dtype=torch.float64)
        r = 2.0 * r - 1.0
        a, b, c = r.unbind(-1)
        determinant = 1.0 + 2.0 * a * b * c - a * a - b * b - c * c
        keep = r[determinant > 0]
        accepted.append(keep)
        total += keep.shape[0]
    entries = torch.cat(accepted)[:count]
    a, b, c = entries.unbind(-1)
    one = torch.ones_like(a)
    return torch.stack(
        [
            torch.stack([one, a, b], -1),
            torch.stack([a, one, c], -1),
            torch.stack([b, c, one], -1),
        ],
        -2,
    )


def test_uniform_correlations_match_rejection_sampling() -> None:
    reference = _uniform_correlations_by_rejection(N)
    factors = _shared_sampler(1.0, 3)
    drawn = factors.double() @ factors.double().transpose(-1, -2)
    critical = _two_sample_critical(N, N)
    rows, cols = torch.tril_indices(3, 3, offset=-1)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        statistic = _two_sample_statistic(drawn[:, i, j], reference[:, i, j])
        assert statistic < critical
    determinant = _two_sample_statistic(
        torch.linalg.det(drawn), torch.linalg.det(reference)
    )
    assert determinant < critical
    # The product of two correlations is not fixed by the marginals.
    product = _two_sample_statistic(
        drawn[:, 1, 0] * drawn[:, 2, 1], reference[:, 1, 0] * reference[:, 2, 1]
    )
    assert product < critical


# ---------------------------------------------------------------------------
# Log density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("eta", "d"), GRID)
def test_the_log_density_of_every_draw_is_finite(eta: float, d: int) -> None:
    concentration = torch.tensor(eta, dtype=torch.float64)
    factors = _shared_sampler(eta, d)
    assert torch.isfinite(D.LKJCholesky(d, concentration).log_prob(factors)).all()


@pytest.mark.parametrize("eta", [1.0, 2.0])
def test_mean_log_density_matches_an_importance_sampled_entropy(eta: float) -> None:
    """Compare ``E[log p]`` under the draws with an independent estimate.

    Rejection-sampled uniform correlation matrices reweighted by
    ``det(R) ** (eta - 1)`` are a self-normalized importance sample of
    LKJ(eta); torch's log density scores both sets of factors.
    """
    d = 3
    concentration = torch.tensor(eta, dtype=torch.float64)
    density = D.LKJCholesky(d, concentration)
    drawn = density.log_prob(_shared_sampler(eta, d))
    reference = _uniform_correlations_by_rejection(N)
    log_weights = (eta - 1.0) * torch.linalg.det(reference).log()
    weights = torch.softmax(log_weights, 0)
    scored = density.log_prob(torch.linalg.cholesky(reference))
    estimate = (weights * scored).sum()
    reference_variance = (weights.pow(2) * (scored - estimate).pow(2)).sum()
    drawn_variance = drawn.var() / N
    tolerance = 5.0 * math.sqrt(float(reference_variance + drawn_variance))
    assert abs(float(drawn.mean() - estimate)) < tolerance


# ---------------------------------------------------------------------------
# Shapes, dtypes, and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_draws_follow_the_sample_and_batch_shapes_and_dtype(
    dtype: torch.dtype,
) -> None:
    concentration = torch.tensor([[0.5, 1.0, 2.0], [3.0, 5.0, 10.0]], dtype=dtype)
    distribution = OnionLKJCholesky(4, concentration)
    torch.manual_seed(SEED)
    factors = distribution.rsample(torch.Size((5, 7)))
    assert factors.shape == (5, 7, 2, 3, 4, 4)
    assert factors.dtype == dtype
    assert factors.device == concentration.device
    _assert_valid_factors(factors)
    assert distribution.sample().shape == (2, 3, 4, 4)
    expanded = distribution.expand(torch.Size((6, 2, 3)))
    assert isinstance(expanded, OnionLKJCholesky)
    assert expanded.rsample().shape == (6, 2, 3, 4, 4)


def test_a_one_dimensional_factor_is_the_unit_matrix() -> None:
    factors = lkj_cholesky_rsample(1, torch.tensor(2.0), torch.Size((3,)))
    assert torch.equal(factors, torch.ones(3, 1, 1))


def test_sample_carries_no_gradient() -> None:
    concentration = torch.tensor(2.0, requires_grad=True)
    assert not OnionLKJCholesky(3, concentration).sample().requires_grad


@pytest.mark.parametrize(("eta", "d"), [(0.5, 3), (1.0, 3), (2.0, 5)])
def test_reparameterized_gradient_of_the_second_moment_matches_its_derivative(
    eta: float, d: int
) -> None:
    concentration = torch.tensor(eta, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(SEED)
    factors = OnionLKJCholesky(d, concentration).rsample(torch.Size((N,)))
    squared = _correlations(factors).pow(2).mean(0)
    expected = -2.0 / (2.0 * eta + d - 1.0) ** 2
    for entry in squared.unbind(0):
        (gradient,) = torch.autograd.grad(entry, concentration, retain_graph=True)
        assert float(gradient) == pytest.approx(expected, rel=0.05)


def test_the_conditional_family_passes_gradients_to_its_parameters() -> None:
    family = ConditionalLKJCholesky(
        Euclidean(name="x", dim=1), CholeskyFactor(name="L", dim=3)
    )
    factors = family.rsample(torch.zeros(64, 1))
    factors[..., 2, 0].pow(2).sum().backward()
    gradients = [p.grad for p in family.param_source.parameters()]
    assert any(g is not None and bool(g.abs().sum() > 0) for g in gradients)


# ---------------------------------------------------------------------------
# Agreement with the reference machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("eta", "d"), [(2.0, 4), (0.5, 3)])
def test_compiled_and_reference_draws_agree_in_distribution(eta: float, d: int) -> None:
    count = 5000
    backend = ReferenceBackend()
    rng = random.Random(SEED)
    reference = torch.tensor(
        [
            backend.sample("LKJCholesky", {"concentration": eta, "dimension": d}, rng)
            for _ in range(count)
        ],
        dtype=torch.float64,
    )
    _assert_valid_factors(reference)
    compiled = _compiled_program(eta, d)
    critical = _two_sample_critical(count, N)
    reference_correlations = _correlations(reference)
    compiled_correlations = _correlations(compiled)
    for left, right in zip(
        reference_correlations.unbind(-1), compiled_correlations.unbind(-1), strict=True
    ):
        assert _two_sample_statistic(left, right) < critical
