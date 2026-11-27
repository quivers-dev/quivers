"""Restriction excludes queries before evaluating partial base-density formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
import torch
from torch.distributions import Gamma, Pareto, Uniform

from quivers.continuous.families import ConditionalGamma, Truncated
from quivers.continuous.measure import Independent, Normalize, Restrict
from quivers.continuous.param_source import IdentitySource
from quivers.continuous.spaces import Euclidean


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("value", [-1.0, 2.5, -math.inf, math.inf, math.nan])
def test_excluded_gamma_query_has_zero_density_and_finite_inactive_gradients(
    normalized: bool, value: float
) -> None:
    shape = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    raw = Restrict(Gamma(shape, rate), 0.0, 2.0)
    model = Normalize(raw) if normalized else raw
    score = model.log_prob(torch.tensor(value, dtype=torch.float64))
    assert bool(torch.isneginf(score))
    gradients = torch.autograd.grad(score.exp(), (shape, rate), create_graph=True)
    for gradient in gradients:
        assert bool(torch.isfinite(gradient))
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("family", ["pareto", "uniform"])
def test_parameter_dependent_support_preserves_only_valid_rows_and_gradients(
    family: str, normalized: bool
) -> None:
    if family == "pareto":
        first = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([1.5, 0.8, 2.3], dtype=torch.float64, requires_grad=True)
        values = torch.tensor([0.5, 2.7, 4.5], dtype=torch.float64)
        active = torch.tensor([False, True, False])
        high = 4.0
        distribution = Pareto(first, second)
        reference = stats.pareto(
            b=second.detach().numpy(), scale=first.detach().numpy()
        )
        # Evaluate the elementary density only at the independently known valid row.
        expected_log = second[active].log() + second[active] * first[active].log()
        expected_log -= (second[active] + 1) * values[active].log()
        if normalized:
            expected_log -= torch.log1p(-(first[active] / high).pow(second[active]))
    else:
        first = torch.tensor([0.2, 1.0, 0.7], dtype=torch.float64, requires_grad=True)
        second = torch.tensor([1.4, 2.5, 1.8], dtype=torch.float64, requires_grad=True)
        values = torch.tensor([0.1, 1.4, 2.5], dtype=torch.float64)
        active = torch.tensor([False, True, False])
        high = 2.0
        distribution = Uniform(first, second)
        reference = stats.uniform(
            loc=first.detach().numpy(), scale=(second - first).detach().numpy()
        )
        width = (
            torch.minimum(second[active], second.new_tensor(high)) - first[active]
            if normalized
            else second[active] - first[active]
        )
        expected_log = -width.log()
    raw = Restrict(distribution, 0.0, high)
    model = Normalize(raw) if normalized else raw
    score = model.log_prob(values)
    assert score.shape == (3,)
    assert bool(torch.isneginf(score[~active]).all())
    expected_score = reference.logpdf(values.numpy())
    if normalized:
        expected_score -= np.log(reference.cdf(high) - reference.cdf(0.0))
    expected_score[~active.numpy()] = -np.inf
    np.testing.assert_allclose(
        score.detach().numpy(), expected_score, atol=1e-12, rtol=1e-12
    )
    actual = score.exp().sum()
    expected = expected_log.exp().sum()
    got = torch.autograd.grad(actual, (first, second), create_graph=True)
    want = torch.autograd.grad(expected, (first, second), create_graph=True)
    for gradient, oracle in zip(got, want, strict=True):
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, oracle, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            gradient[~active], torch.zeros(2, dtype=torch.float64)
        )


def _gamma_parameters(
    raw: torch.Tensor, dimensions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # These are the documented conditional-head transforms, independent of its builder.
    return (
        torch.logaddexp(raw[:, :dimensions], raw.new_tensor(0.0)) + 0.1,
        torch.logaddexp(raw[:, dimensions:], raw.new_tensor(0.0)) + 1e-7,
    )


def _gamma_density_oracle(
    raw: np.ndarray, values: np.ndarray, active: np.ndarray
) -> np.ndarray:
    dimensions = values.shape[-1]
    shape = np.logaddexp(raw[:, :dimensions], 0.0) + 0.1
    rate = np.logaddexp(raw[:, dimensions:], 0.0) + 1e-7
    density = stats.gamma.logpdf(
        values[active], a=shape[active], scale=1 / rate[active]
    )
    mass = stats.gamma.cdf(2.0, a=shape[active], scale=1 / rate[active])
    result = np.full(values.shape[0], -np.inf)
    result[active] = (density - np.log(mass)).sum(-1)
    return result


@pytest.mark.parametrize("dimensions", [1, 3])
@pytest.mark.parametrize("restriction_outside_event", [False, True])
def test_conditional_and_algebraic_restrictions_share_safe_independent_scoring(
    dimensions: int, restriction_outside_event: bool
) -> None:
    rows = 4
    raw = torch.linspace(-0.4, 0.7, rows * 2 * dimensions, dtype=torch.float64)
    raw = raw.reshape(rows, 2 * dimensions).requires_grad_()
    values = torch.linspace(0.4, 1.4, rows * dimensions, dtype=torch.float64).reshape(
        rows, dimensions
    )
    values[1, 0] = -1.0
    values[2, -1] = 2.5
    active = np.array([True, False, False, True])
    conditional = Truncated(
        ConditionalGamma(
            Euclidean(name="parameters", dim=2 * dimensions),
            Euclidean(name="response", dim=dimensions),
            param_source=IdentitySource(2 * dimensions),
        ),
        lower=0.0,
        upper=2.0,
    )
    shape, rate = _gamma_parameters(raw, dimensions)
    coordinates = Gamma(shape, rate)
    algebra = Normalize(
        Restrict(Independent(coordinates, 1), 0.0, 2.0)
        if restriction_outside_event
        else Independent(Restrict(coordinates, 0.0, 2.0), 1)
    )
    conditional_score = conditional.log_prob(raw, values)
    algebra_score = algebra.log_prob(values)
    assert conditional_score.shape == algebra_score.shape == (rows,)
    oracle = _gamma_density_oracle(raw.detach().numpy(), values.numpy(), active)
    for score in (conditional_score, algebra_score):
        np.testing.assert_allclose(
            score.detach().numpy(), oracle, atol=2e-12, rtol=2e-12
        )
    actual_grad = torch.autograd.grad(
        conditional_score.exp().sum(), raw, retain_graph=True
    )[0]
    algebra_grad = torch.autograd.grad(algebra_score.exp().sum(), raw)[0]
    assert bool(torch.isfinite(actual_grad).all())
    assert bool(torch.isfinite(algebra_grad).all())
    torch.testing.assert_close(
        actual_grad[~torch.tensor(active)],
        torch.zeros(2, 2 * dimensions, dtype=torch.float64),
    )
    torch.testing.assert_close(algebra_grad, actual_grad, atol=2e-11, rtol=2e-11)

    # Every head derivative is checked against SciPy, including the shape-dependent mass.
    values_np = values.numpy()
    raw_np = raw.detach().numpy()
    oracle_grad = np.zeros_like(raw_np)
    epsilon = 1e-5
    for row in range(rows):
        for column in range(2 * dimensions):
            plus, minus = raw_np.copy(), raw_np.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            oracle_grad[row, column] = (
                np.exp(_gamma_density_oracle(plus, values_np, active)).sum()
                - np.exp(_gamma_density_oracle(minus, values_np, active)).sum()
            ) / (2 * epsilon)
    np.testing.assert_allclose(actual_grad.numpy(), oracle_grad, atol=2e-9, rtol=2e-8)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("representation", ["python", "scalar_tensor", "batch_tensor"])
@pytest.mark.parametrize("count", [2**53 + 1, 2**63 - 1])
def test_restriction_preserves_original_integer_capacity_before_count_scoring(
    normalized: bool, representation: str, count: int
) -> None:
    from quivers.continuous._closed_probability import ClosedPoisson

    raw = Restrict(ClosedPoisson(torch.tensor(3.0, dtype=torch.float64)), 0.0)
    model = Normalize(raw) if normalized else raw
    value = (
        count
        if representation == "python"
        else torch.tensor(
            [count] if representation == "batch_tensor" else count, dtype=torch.int64
        )
    )
    with pytest.raises(ValueError, match=r"2\*\*53"):
        model.log_prob(value)


@pytest.mark.parametrize("normalized", [False, True])
def test_restriction_accepts_exact_count_capacity_boundary(normalized: bool) -> None:
    from quivers.continuous._closed_probability import ClosedPoisson

    n = 2**53
    rate = torch.tensor(n, dtype=torch.float64, requires_grad=True)
    raw = Restrict(ClosedPoisson(rate), 0.0)
    model = Normalize(raw) if normalized else raw
    score = model.log_prob(torch.tensor(n, dtype=torch.int64))
    # Stirling's remainder is between 1/(12n+1) and 1/(12n); here the
    # enclosure is narrower than Float64 rounding, independently of lgamma.
    expected = -0.5 * math.log(2 * math.pi * n) - 1 / (12 * n)
    assert score.item() == pytest.approx(expected, abs=4e-14)
    torch.testing.assert_close(
        torch.autograd.grad(score, rate)[0], torch.zeros_like(rate)
    )


def test_zero_reinterpreted_axes_preserves_coordinate_batch_and_inactive_gradients() -> (
    None
):
    from torch.distributions import Independent as TorchIndependent

    shape = torch.tensor([[0.7], [1.3], [2.1]], dtype=torch.float64, requires_grad=True)
    rate = torch.tensor([[0.8], [1.1], [1.6]], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([[0.5], [-1.0], [2.5]], dtype=torch.float64)
    base = TorchIndependent(Gamma(shape, rate), 0)
    model = Normalize(Restrict(base, 0.0, 2.0))
    score = model.log_prob(values)
    assert score.shape == (3, 1)
    assert bool(torch.isneginf(score[1:]).all())
    expected = stats.gamma.logpdf(0.5, a=0.7, scale=1 / 0.8) - math.log(
        stats.gamma.cdf(2.0, a=0.7, scale=1 / 0.8)
    )
    assert score[0, 0].item() == pytest.approx(expected, abs=2e-13)
    for gradient in torch.autograd.grad(score.exp().sum(), (shape, rate)):
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient[1:], torch.zeros(2, 1, dtype=torch.float64))


def test_coupled_simplex_support_retains_sample_batch_event_axes_and_derivatives() -> (
    None
):
    from torch.distributions import Dirichlet

    concentration = torch.tensor(
        [[0.7, 1.4, 2.1, 1.2], [1.1, 0.8, 1.7, 2.3], [1.3, 0.6, 1.9, 1.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [-0.1, 0.2, 0.3, 0.6], [0.2, 0.2, 0.2, 0.2]],
            [[0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.3, 0.4], [math.nan, 0.1, 0.3, 0.4]],
        ],
        dtype=torch.float64,
    )
    active = torch.tensor([[True, False, False], [True, True, False]])
    score = Restrict(Dirichlet(concentration), 0.0, 1.0).log_prob(values)
    assert score.shape == (2, 3)
    assert bool(torch.isneginf(score[~active]).all())
    expected = torch.full((2, 3), -torch.inf, dtype=torch.float64)
    parameters, points = [], []
    for sample, row in active.nonzero().tolist():
        expected[sample, row] = stats.dirichlet.logpdf(
            values[sample, row].numpy(), concentration[row].detach().numpy()
        )
        parameters.append(concentration[row])
        points.append(values[sample, row])
    torch.testing.assert_close(score, expected, atol=2e-13, rtol=2e-13)
    alpha, point = torch.stack(parameters), torch.stack(points)
    independent = (
        torch.lgamma(alpha.sum(-1))
        - torch.lgamma(alpha).sum(-1)
        + ((alpha - 1) * point.log()).sum(-1)
    )
    actual_gradient = torch.autograd.grad(
        score.exp().sum(), concentration, create_graph=True
    )[0]
    oracle_gradient = torch.autograd.grad(
        independent.exp().sum(), concentration, create_graph=True
    )[0]
    torch.testing.assert_close(actual_gradient, oracle_gradient, atol=2e-12, rtol=2e-12)
    assert bool(torch.isfinite(actual_gradient).all())
    torch.testing.assert_close(actual_gradient[2], torch.zeros(4, dtype=torch.float64))
    assert bool(
        torch.isfinite(
            torch.autograd.grad(actual_gradient.sum(), concentration)[0]
        ).all()
    )


def test_coupled_correlation_cholesky_support_uses_a_matrix_feasible_point() -> None:
    from torch.distributions import LKJCholesky

    eta = torch.tensor([0.7, 1.5, 2.3], dtype=torch.float64, requires_grad=True)
    values = torch.eye(2, dtype=torch.float64).expand(2, 3, 2, 2).clone()
    correlations = {(0, 0): 0.2, (1, 0): -0.4, (1, 1): 0.3}
    for (sample, row), rho in correlations.items():
        values[sample, row, 1, 0] = rho
        values[sample, row, 1, 1] = math.sqrt(1 - rho**2)
    values[0, 1, 0, 1] = 0.1
    values[0, 2] = 0.0
    values[1, 2, 0, 0] = math.nan
    score = Restrict(LKJCholesky(2, eta), -2.0, 2.0).log_prob(values)
    assert score.shape == (2, 3)
    expected = torch.full((2, 3), -torch.inf, dtype=torch.float64)
    oracle_terms = []
    for (sample, row), rho in correlations.items():
        term = (
            torch.lgamma(eta[row] + 0.5)
            - torch.lgamma(eta[row])
            - 0.5 * math.log(math.pi)
        )
        term += (eta[row] - 1) * math.log1p(-(rho**2))
        expected[sample, row] = term.detach()
        oracle_terms.append(term)
    torch.testing.assert_close(score, expected, atol=2e-13, rtol=2e-13)
    actual_gradient = torch.autograd.grad(score.exp().sum(), eta, create_graph=True)[0]
    oracle_gradient = torch.autograd.grad(
        torch.stack(oracle_terms).exp().sum(), eta, create_graph=True
    )[0]
    torch.testing.assert_close(actual_gradient, oracle_gradient, atol=2e-13, rtol=2e-13)
    assert bool(torch.isfinite(actual_gradient).all())
    assert actual_gradient[2] == 0
    assert bool(
        torch.isfinite(torch.autograd.grad(actual_gradient.sum(), eta)[0]).all()
    )


def test_coupled_positive_definite_support_handles_valid_and_invalid_matrix_queries() -> (
    None
):
    from torch.distributions import Wishart

    df = torch.tensor([3.2, 4.5, 5.7], dtype=torch.float64, requires_grad=True)
    raw = torch.tensor(
        [[0.1, 0.2, -0.1], [-0.2, 0.1, 0.3], [0.3, -0.1, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    zero = torch.zeros(3, dtype=torch.float64)
    factor = torch.stack(
        (raw[:, 0].exp(), zero, raw[:, 1], raw[:, 2].exp()), -1
    ).reshape(3, 2, 2)
    covariance = factor @ factor.transpose(-1, -2)
    values = torch.tensor(
        [
            [
                [[1.0, 0.1], [0.1, 0.8]],
                [[1.0, 2.0], [2.0, 1.0]],
                [[11.0, 0.0], [0.0, 11.0]],
            ],
            [
                [[1.0, 0.5], [0.2, 1.0]],
                [[2.0, 0.2], [0.2, 1.2]],
                [[math.nan, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=torch.float64,
    )
    active = [(0, 0), (1, 1)]
    score = Restrict(Wishart(df, scale_tril=factor), -10.0, 10.0).log_prob(values)
    assert score.shape == (2, 3)
    expected = torch.full((2, 3), -torch.inf, dtype=torch.float64)
    oracle_terms = []
    for sample, row in active:
        point = values[sample, row]
        expected[sample, row] = stats.wishart.logpdf(
            point.numpy(), df=df[row].item(), scale=covariance[row].detach().numpy()
        )
        multigamma = (
            0.5 * math.log(math.pi)
            + torch.lgamma(df[row] / 2)
            + torch.lgamma((df[row] - 1) / 2)
        )
        term = (
            (df[row] - 3) / 2 * torch.linalg.slogdet(point).logabsdet
            - 0.5 * torch.linalg.solve(covariance[row], point).diagonal().sum()
            - df[row] * math.log(2)
            - df[row] / 2 * torch.linalg.slogdet(covariance[row]).logabsdet
            - multigamma
        )
        oracle_terms.append(term)
    torch.testing.assert_close(score, expected, atol=2e-13, rtol=2e-13)
    got = torch.autograd.grad(score.exp().sum(), (df, raw), create_graph=True)
    want = torch.autograd.grad(
        torch.stack(oracle_terms).exp().sum(), (df, raw), create_graph=True
    )
    for gradient, oracle in zip(got, want, strict=True):
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, oracle, atol=2e-13, rtol=2e-13)
        assert bool((gradient[2] == 0).all())
    second = torch.autograd.grad(sum(gradient.sum() for gradient in got), (df, raw))
    assert all(bool(torch.isfinite(value).all()) for value in second)


@pytest.mark.parametrize("backend", ["native", "reference"])
def test_closed_singleton_mass_preserves_original_count_capacity(backend: str) -> None:
    count = 2**53 + 1
    if backend == "native":
        from quivers.continuous._closed_probability import ClosedPoisson
        from quivers.continuous._interval import log_interval_mass

        base = ClosedPoisson(torch.tensor(3.0, dtype=torch.float64))
    else:
        from quivers.qiec import HostLaw
        from quivers.qiec._interval import log_interval_mass

        base = HostLaw(family="Poisson", arguments={"rate": 3.0})
    with pytest.raises(ValueError, match=r"2\*\*53"):
        log_interval_mass(base, count, count)


def _conditional_count_family(family: str):
    from quivers.continuous.families import ConditionalBernoulli, ConditionalPoisson
    from quivers.core.objects import FinSet

    domain = Euclidean(name="parameters", dim=1)
    if family == "bernoulli":
        return ConditionalBernoulli(
            domain, FinSet(name="truth", cardinality=2), param_source=IdentitySource(1)
        )
    return ConditionalPoisson(
        domain, Euclidean(name="response", dim=1), param_source=IdentitySource(1)
    )


@pytest.mark.parametrize("family", ["bernoulli", "poisson"])
def test_truncated_count_sampling_retains_base_integer_dtype_and_geometry(
    family: str,
) -> None:
    base = _conditional_count_family(family)
    lower, upper = (-0.5, 0.5) if family == "bernoulli" else (1.0, 2.0)
    model = Truncated(base, lower, upper, max_rejection_iterations=128)
    inputs = torch.zeros(4, 1, dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(938)
        samples = model.rsample(inputs, torch.Size([2, 3]))
    expected_shape = (2, 3, 4) if family == "bernoulli" else (2, 3, 4, 1)
    assert samples.shape == expected_shape
    assert samples.dtype == torch.int64
    assert bool(((samples >= lower) & (samples <= upper)).all())
    if family == "bernoulli":
        assert bool((samples == 0).all())


@pytest.mark.parametrize("family", ["bernoulli", "poisson"])
@pytest.mark.parametrize("succeeds", [False, True])
def test_truncated_rejection_retains_accepted_cells_and_obeys_budget(
    family: str, succeeds: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _conditional_count_family(family)
    bernoulli = family == "bernoulli"
    lower, upper = (-0.5, 0.5) if bernoulli else (1.0, 2.0)
    shape = (2, 3) if bernoulli else (2, 3, 1)
    first = torch.full(shape, 1 if bernoulli else 0, dtype=torch.int64)
    first.reshape(-1)[0] = 0 if bernoulli else 1
    replacement = torch.full(shape, 0 if bernoulli else 2, dtype=torch.int64)
    calls = 0

    def draw(inputs: torch.Tensor, sample_shape: torch.Size) -> torch.Tensor:
        nonlocal calls
        assert inputs.shape == (3, 1)
        assert sample_shape == torch.Size([2])
        calls += 1
        return (replacement if succeeds and calls > 1 else first).clone()

    monkeypatch.setattr(base, "rsample", draw)
    budget = 1 if succeeds else 2
    model = Truncated(base, lower, upper, max_rejection_iterations=budget)
    inputs = torch.zeros(3, 1, dtype=torch.float64)
    if succeeds:
        samples = model.rsample(inputs, torch.Size([2]))
        expected = replacement.clone()
        expected.reshape(-1)[0] = first.reshape(-1)[0]
        torch.testing.assert_close(samples, expected)
        assert samples.dtype == torch.int64
    else:
        with pytest.raises(RuntimeError, match="rejection"):
            model.rsample(inputs, torch.Size([2]))
    assert calls == budget + 1


@pytest.mark.parametrize("location", [0.1, 1e-200])
def test_closed_singleton_mass_preserves_float64_atom_location(location: float) -> None:
    from quivers.continuous._interval import log_interval_mass
    from quivers.continuous.measure import PointMass

    base = PointMass(torch.tensor(location, dtype=torch.float64))
    score = log_interval_mass(base, location, location)
    assert score.dtype == torch.float64
    assert score.item() == 0.0
    adjacent = math.nextafter(location, math.inf)
    assert bool(torch.isneginf(log_interval_mass(base, adjacent, adjacent)))
