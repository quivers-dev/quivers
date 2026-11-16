"""Finite marginalization integrates recorded events exactly once."""

import numpy as np
import pytest
import torch
from scipy.special import gammaln, logsumexp

from quivers.dsl import loads


SOURCE = """object Row : FinSet 4
object Group : FinSet 2
object Class : FinSet 2
object Out : Real 1
program mixture : Row -> Out
    sample mu : Class <- Normal(loc=0.0, scale=1.0)
    sample pi : Group <- Dirichlet(concentration=1.0) [over=Class, iid_over=Group]
    marginalize z : Class <- Categorical(probs=pi) [over=Group]
        observe y : Row <- Normal(loc=mu[z], scale=1.0) [via=group, cens=status, cens_upper=upper, trunc_lower=-1.0, trunc_upper=2.0, weights=weight]
    return mu
export mixture
"""


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("sparse_weights", [False, True])
def test_native_marginal_events_match_independent_finite_sum_and_every_gradient(
    grouped, sparse_weights
):
    source = (
        SOURCE
        if grouped
        else SOURCE.replace(
            "sample pi : Group <- Dirichlet(concentration=1.0) [over=Class, iid_over=Group]",
            "sample pi <- Dirichlet(concentration=1.0) [over=Class]",
        )
        .replace(" [over=Group]\n", "\n")
        .replace("via=group, ", "")
    )
    mu = torch.tensor([-0.2, 0.7], dtype=torch.float64, requires_grad=True)
    pi = torch.tensor(
        [[0.3, 0.7], [0.6, 0.4]] if grouped else [0.3, 0.7],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch.tensor([0.1, 0.4, -0.3, 0.2], dtype=torch.float64)
    status = torch.tensor([0, 1, -1, 2])
    upper = torch.tensor(
        [float("nan"), float("nan"), float("nan"), 0.8], dtype=torch.float64
    )
    weight = torch.tensor(
        [1.0, 0.0, 2.0, 0.5] if sparse_weights else [1.0, 0.8, 2.0, 0.5],
        dtype=torch.float64,
    )
    groups = torch.tensor([0, 0, 1, 1])
    program = loads(source).morphism
    actual = program.log_joint(
        torch.zeros((1, 1), dtype=torch.float64),
        {
            "mu": mu,
            "pi": pi,
            "y": y,
            "status": status,
            "upper": upper,
            "weight": weight,
            "group": groups,
        },
    ).sum()
    # Independent Normal tail formulas and fixed truncation CDF ratios.
    columns = []
    for location in mu:
        normal = torch.distributions.Normal(location, 1.0)
        z = y - location
        mass = torch.special.ndtr(2 - location) - torch.special.ndtr(-1 - location)
        numerator = torch.stack(
            [
                normal.log_prob(y[0]).exp(),
                torch.special.ndtr(2 - location) - torch.special.ndtr(z[1]),
                torch.special.ndtr(z[2]) - torch.special.ndtr(-1 - location),
                torch.special.ndtr(upper[3] - location) - torch.special.ndtr(z[3]),
            ]
        )
        columns.append(weight * (numerator.log() - mass.log()))
    likelihood = torch.stack(columns, -1)
    prior = torch.distributions.Normal(0.0, 1.0).log_prob(mu).sum()
    prior += (
        torch.distributions.Dirichlet(torch.ones(2, dtype=torch.float64))
        .log_prob(pi)
        .sum()
    )
    log_pi = pi.log() - pi.sum(-1, keepdim=True).log()
    if grouped:
        expected = prior + sum(
            torch.logsumexp(log_pi[g] + likelihood[groups == g].sum(0), 0)
            for g in range(2)
        )
    else:
        expected = prior + torch.logsumexp(log_pi + likelihood.sum(0), 0)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for a, b in zip(
        torch.autograd.grad(actual, (mu, pi)),
        torch.autograd.grad(expected, (mu, pi)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("grouped", [False, True])
def test_marginalization_preserves_integer_counts_beyond_float32(grouped):
    """Class expansion preserves exact response integers before scoring."""
    source = """object Row : FinSet 2
object Group : FinSet 2
object Class : FinSet 2
object Out : Real 1
program mixture : Row -> Out
    sample pi : Group <- Dirichlet(concentration=1.0) [over=Class, iid_over=Group]
    marginalize z : Class <- Categorical(probs=pi) [over=Group]
        observe y : Row <- Poisson(rate=rate[z]) [via=group]
    return pi
export mixture
"""
    if not grouped:
        source = (
            source.replace(
                "sample pi : Group <- Dirichlet(concentration=1.0) [over=Class, iid_over=Group]",
                "sample pi <- Dirichlet(concentration=1.0) [over=Class]",
            )
            .replace(" [over=Group]\n", "\n")
            .replace(" [via=group]", "")
        )
    response = torch.tensor([2**24 + 1, 2**24 + 3], dtype=torch.int64)
    rate = torch.tensor([1000.0, 2000.0], dtype=torch.float64)
    pi = torch.tensor(
        [[0.3, 0.7], [0.6, 0.4]] if grouped else [0.3, 0.7],
        dtype=torch.float64,
    )
    actual = (
        loads(source)
        .morphism.log_joint(
            torch.zeros((1, 1), dtype=torch.float64),
            {"pi": pi, "rate": rate, "y": response, "group": torch.tensor([0, 1])},
        )
        .sum()
    )
    # SciPy supplies an independent double-precision integer-PMF oracle.
    y = response.numpy().astype("float64")[:, None]
    rates = rate.numpy()[None, :]
    scores = y * np.log(rates) - rates - gammaln(y + 1)
    expected = (
        logsumexp(np.log(pi.numpy()) + scores, axis=-1).sum()
        if grouped
        else logsumexp(np.log(pi.numpy()) + scores.sum(0))
    )
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.float64), atol=1e-7, rtol=0
    )
