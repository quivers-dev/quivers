"""Gradient checks for the variational objectives and estimators.

Each check compares the Monte Carlo mean of an estimator's gradient
with an exact reference: a closed form where the objective has one,
and otherwise Gauss-Hermite quadrature of the expected Monte Carlo
objective, differentiated by autograd.

The model is a two-dimensional ``z ~ Normal(theta * 1, I)`` with no
observations, so the evidence is 1 for every ``theta``, and ``theta``
is a model parameter whose gradient the estimators must also get
right. The guide is a mean-field Normal with every coordinate at
``N(mu, sigma^2)``. A non-plate latent is drawn
independently for every row of the program input, so one call with a
batch of ``B`` rows averages ``B`` independent replicates of the
objective; repeating the call gives the standard error.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch.distributions as D

from quivers.continuous import Draw, Euclidean, MonadicProgram
from quivers.continuous.inline import FixedDistribution
from quivers.core import Unit
from quivers.inference import ELBO, AutoNormalGuide, Objective, ScoreFunction
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentRegistry

_PLANE = Euclidean(name="R2", dim=2)


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


class _Problem:
    """Target ``N(theta * 1, I)`` on the plane and a mean-field Normal
    guide with both coordinates at ``N(mu, sigma^2)``."""

    def __init__(self, theta: float, mu: float, sigma: float) -> None:
        self.theta = torch.nn.Parameter(torch.tensor(theta))
        target_mean = self.theta

        def builder(batch: int, device: torch.device) -> D.Distribution:
            del device
            loc = target_mean.reshape(1, 1).expand(batch, 2)
            return D.Normal(loc, torch.ones(batch, 2))

        self.model = MonadicProgram(
            Unit,
            Euclidean(name="out", dim=1),
            steps=[Draw(names=("z",), morphism=FixedDistribution(_PLANE, builder))],
            return_vars=(),
        )
        self.guide = AutoNormalGuide(self.model, observed_names=set())
        with torch.no_grad():
            getattr(self.guide, "loc_z").fill_(mu)
            getattr(self.guide, "log_scale_z").fill_(math.log(sigma))


def _gradient_of_bound(
    problem: _Problem,
    guide: Guide,
    objective: Objective,
    *,
    sign: float,
    batch: int,
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mean and standard error of the gradient of ``sign * loss``.

    Returns the mean gradient over ``repeats`` calls of ``batch``
    replicates each, its standard error, and the mean loss times
    ``sign``.
    """
    x = torch.zeros(batch, 1)
    rows: list[torch.Tensor] = []
    values: list[float] = []
    parameters = [*list(guide.parameters()), problem.theta]
    for _ in range(repeats):
        for parameter in parameters:
            parameter.grad = None
        loss = objective(problem.model, guide, x, {})
        (sign * loss).backward()
        grads = []
        for parameter in parameters:
            assert parameter.grad is not None
            grads.append(parameter.grad.reshape(-1))
        rows.append(torch.cat(grads))
        values.append(sign * float(loss.detach()))
    stacked = torch.stack(rows)
    return (
        stacked.mean(0),
        stacked.std(0) / math.sqrt(repeats),
        sum(values) / len(values),
    )


def _closed_form_elbo(theta: float, mu: float, sigma: float) -> float:
    """``ELBO = 2 * (-((mu - theta)^2 + sigma^2)/2 + log sigma + 1/2)``."""
    return 2.0 * (-((mu - theta) ** 2 + sigma**2) / 2 + math.log(sigma) + 0.5)


def _closed_form_elbo_gradient(theta: float, mu: float, sigma: float) -> torch.Tensor:
    """ELBO gradient with respect to ``(mu_1, mu_2, log sigma_1,
    log sigma_2, theta)``."""
    d_mu = -(mu - theta)
    d_log_sigma = 1.0 - sigma**2
    return torch.tensor([d_mu, d_mu, d_log_sigma, d_log_sigma, 2.0 * (mu - theta)])


# ---------------------------------------------------------------------------
# Score function on the ELBO
# ---------------------------------------------------------------------------


def test_score_function_elbo_gradient_matches_closed_form() -> None:
    torch.manual_seed(0)
    problem = _Problem(theta=0.3, mu=1.0, sigma=0.8)
    mean, se, _ = _gradient_of_bound(
        problem,
        problem.guide,
        ELBO(estimator=ScoreFunction()),
        sign=-1.0,
        batch=2000,
        repeats=20,
    )
    exact = _closed_form_elbo_gradient(0.3, 1.0, 0.8)
    assert torch.all((mean - exact).abs() <= 4.0 * se), (mean, exact, se)
    # The location gradient is far from zero relative to its error.
    assert abs(float(exact[0])) > 10.0 * float(se[0])


class _NonReparameterizedNormalGuide(Guide):
    """A Normal guide whose draws carry no gradient (``sample``, not
    ``rsample``)."""

    def __init__(self, registry: LatentRegistry, mu: float, sigma: float) -> None:
        super().__init__()
        self._registry = registry
        self.loc = torch.nn.Parameter(torch.full((2,), mu))
        self.log_scale = torch.nn.Parameter(torch.full((2,), math.log(sigma)))

    def _dist(self) -> D.Normal:
        return D.Normal(self.loc, self.log_scale.exp())

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"z": self._dist().sample((x.shape[0],))}

    def log_prob(self, x: torch.Tensor, sites: dict[str, torch.Tensor]) -> torch.Tensor:
        del x
        return self._dist().log_prob(sites["z"]).sum(-1)

    @property
    def latent_names(self) -> list[str]:
        return ["z"]


def test_score_function_trains_a_guide_whose_draws_carry_no_gradient() -> None:
    torch.manual_seed(1)
    problem = _Problem(theta=-0.5, mu=0.4, sigma=1.3)
    guide = _NonReparameterizedNormalGuide(problem.guide.registry, 0.4, 1.3)
    mean, se, _ = _gradient_of_bound(
        problem,
        guide,
        ELBO(estimator=ScoreFunction()),
        sign=-1.0,
        batch=2000,
        repeats=20,
    )
    exact = _closed_form_elbo_gradient(-0.5, 0.4, 1.3)
    assert torch.all((mean - exact).abs() <= 4.0 * se), (mean, exact, se)
    assert abs(float(exact[0])) > 10.0 * float(se[0])


def test_score_function_reports_the_elbo_estimate() -> None:
    torch.manual_seed(2)
    problem = _Problem(theta=0.3, mu=1.0, sigma=0.8)
    _, _, value = _gradient_of_bound(
        problem,
        problem.guide,
        ELBO(estimator=ScoreFunction()),
        sign=-1.0,
        batch=4000,
        repeats=5,
    )
    # Per-draw sd of log p - log q is below 2 here; SE < 0.015.
    assert abs(value - _closed_form_elbo(0.3, 1.0, 0.8)) < 0.06
