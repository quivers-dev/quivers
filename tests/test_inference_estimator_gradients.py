"""Gradient checks for the variational objectives and estimators.

Each check compares the Monte Carlo mean of an estimator's gradient
with an exact reference: a closed form where the objective has one,
and otherwise Gauss-Hermite quadrature of the expected Monte Carlo
objective, differentiated by autograd.

The model is a two-dimensional ``z ~ Normal(theta * 1, I)`` with no
observations, so the evidence is 1 for every ``theta``, and ``theta``
is a model parameter whose gradient the estimators must also get
right. The guide is a mean-field Normal ``N(mu, diag(sigma^2))``. A
non-plate latent is drawn
independently for every row of the program input, so one call with a
batch of ``B`` rows averages ``B`` independent replicates of the
objective; repeating the call gives the standard error.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy.polynomial.hermite_e as hermite
import pytest
import torch
import torch.distributions as D

from quivers.continuous import Draw, Euclidean, MonadicProgram
from quivers.continuous.inline import FixedDistribution
from quivers.core import Unit
from quivers.dsl import loads
from quivers.inference import (
    ELBO,
    RWS,
    SVI,
    AutoNormalGuide,
    ChiVI,
    DoublyReparameterized,
    DReGsBound,
    GradientEstimator,
    IWAEBound,
    Objective,
    ParticleLogDensities,
    Reparameterized,
    RenyiBound,
    ScoreFunction,
    StickingTheLanding,
    VRIWAEBound,
    power_mean_bound,
)
from quivers.inference.guides.base import Guide
from quivers.inference.objectives import _draw_particles
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
    guide ``N(mu, diag(sigma^2))``; a scalar ``mu`` or ``sigma`` sets
    both coordinates."""

    def __init__(
        self,
        theta: float,
        mu: float | tuple[float, float],
        sigma: float | tuple[float, float],
    ) -> None:
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
        mus = (mu, mu) if isinstance(mu, float) else mu
        sigmas = (sigma, sigma) if isinstance(sigma, float) else sigma
        with torch.no_grad():
            getattr(self.guide, "loc_z").copy_(torch.tensor(mus))
            getattr(self.guide, "log_scale_z").copy_(torch.log(torch.tensor(sigmas)))


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
    ``sign``. Gradients are ordered as the guide's parameters followed
    by ``theta``.
    """
    x = torch.zeros(batch, 1)
    rows: list[torch.Tensor] = []
    values: list[float] = []
    parameters = [*list(guide.parameters()), problem.theta]
    for _ in range(repeats):
        for parameter in parameters:
            parameter.grad = None
        # The loss of ``batch`` independent runs is the sum of their
        # one-run losses; their mean is the one-run loss.
        loss = objective(problem.model, guide, x, {}) / batch
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


# ---------------------------------------------------------------------------
# Every (objective, estimator) pair against quadrature
# ---------------------------------------------------------------------------

_THETA = 0.1
_MU = (0.6, -0.2)
_SIGMA = (1.3, 1.15)


def _quadrature_expected_bound(
    power: float, num_particles: int, nodes: int = 24
) -> tuple[float, torch.Tensor]:
    """``E[B_a]`` for the table's problem and its exact gradient.

    Gauss-Hermite quadrature over the reparameterization noise of each
    particle (two dimensions per particle), differentiated by autograd
    with respect to ``(mu_1, mu_2, log sigma_1, log sigma_2, theta)``.
    """
    abscissae, weights = hermite.hermegauss(nodes)
    eps = torch.tensor(abscissae)
    node_weights = torch.tensor(weights) / math.sqrt(2.0 * math.pi)
    e1, e2 = torch.meshgrid(eps, eps, indexing="ij")
    grid = torch.stack([e1.reshape(-1), e2.reshape(-1)], dim=-1)
    grid_weights = (node_weights[:, None] * node_weights[None, :]).reshape(-1)

    mu = torch.tensor(_MU, requires_grad=True)
    log_sigma = torch.log(torch.tensor(_SIGMA)).requires_grad_(True)
    theta = torch.tensor(_THETA, requires_grad=True)
    z = mu + log_sigma.exp() * grid
    log_ratio = D.Normal(theta, 1.0).log_prob(z).sum(-1) - D.Normal(
        mu, log_sigma.exp()
    ).log_prob(z).sum(-1)
    if num_particles == 1:
        expected = (grid_weights * log_ratio).sum()
    else:
        assert num_particles == 2
        pair = torch.stack(
            [
                log_ratio[:, None].expand(-1, log_ratio.shape[0]),
                log_ratio[None, :].expand(log_ratio.shape[0], -1),
            ]
        )
        log_weights = torch.full((2, 1, 1), -math.log(2.0))
        bound = power_mean_bound(pair, log_weights, power)
        expected = (grid_weights[:, None] * grid_weights[None, :] * bound).sum()
    expected.backward()
    assert mu.grad is not None and log_sigma.grad is not None
    assert theta.grad is not None
    gradient = torch.cat([mu.grad, log_sigma.grad, theta.grad.reshape(1)])
    return float(expected.detach()), gradient


_REP = Reparameterized
_STL = StickingTheLanding
_DREG = DoublyReparameterized
_SCORE = ScoreFunction

# (id, objective factory, exponent a, particles, sign of the bound in the loss)
_VALID_PAIRS: list[tuple[str, Callable[[], Objective], float, int, float]] = []
for _label, _make, _power, _sign in [
    ("elbo", lambda e: ELBO(num_particles=2, estimator=e), 0.0, -1.0),
    ("iwae", lambda e: IWAEBound(num_particles=2, estimator=e), 1.0, -1.0),
    ("renyi-0.5", lambda e: RenyiBound(0.5, num_particles=2, estimator=e), 0.5, -1.0),
    ("renyi-neg1", lambda e: RenyiBound(-1.0, num_particles=2, estimator=e), 2.0, -1.0),
    ("vriwae-0.3", lambda e: VRIWAEBound(0.3, num_particles=2, estimator=e), 0.7, -1.0),
]:
    for _estimator_label, _estimator in [
        ("rep", _REP),
        ("stl", _STL),
        ("dreg", _DREG),
        ("score", _SCORE),
    ]:
        if _estimator is _STL and _power != 0.0:
            continue
        _VALID_PAIRS.append(
            (
                f"{_label}-{_estimator_label}",
                (lambda m, e: lambda: m(e()))(_make, _estimator),
                _power,
                2,
                _sign,
            )
        )
_VALID_PAIRS.append(("dregs-bound", lambda: DReGsBound(num_particles=2), 1.0, 2, -1.0))
_VALID_PAIRS.append(
    (
        "iwae-single-draw-stl",
        lambda: IWAEBound(num_particles=1, estimator=StickingTheLanding()),
        1.0,
        1,
        -1.0,
    )
)


@pytest.mark.parametrize(
    ("make", "power", "num_particles", "sign"),
    [pair[1:] for pair in _VALID_PAIRS],
    ids=[pair[0] for pair in _VALID_PAIRS],
)
def test_estimator_gradient_matches_quadrature(
    make: Callable[[], Objective], power: float, num_particles: int, sign: float
) -> None:
    """The Monte Carlo mean gradient of every valid pair equals the
    quadrature gradient of the expected bound within 4.5 standard
    errors, for the variational and the model parameters, and the
    reported value matches the expected bound."""
    torch.manual_seed(0)
    problem = _Problem(theta=_THETA, mu=_MU, sigma=_SIGMA)
    mean, se, value = _gradient_of_bound(
        problem, problem.guide, make(), sign=sign, batch=2000, repeats=16
    )
    exact_value, exact = _quadrature_expected_bound(power, num_particles)
    assert torch.all((mean - exact).abs() <= 4.5 * se + 1e-9), (mean, exact, se)
    # Reported value: 32000 replicates; per-replicate sd below 3.
    assert abs(value - exact_value) < 0.08, (value, exact_value)


@pytest.mark.parametrize(
    "make",
    [
        lambda: IWAEBound(num_particles=2, estimator=StickingTheLanding()),
        lambda: RenyiBound(0.5, num_particles=2, estimator=StickingTheLanding()),
        lambda: VRIWAEBound(0.3, num_particles=2, estimator=StickingTheLanding()),
    ],
    ids=["iwae", "renyi", "vriwae"],
)
def test_sticking_the_landing_is_rejected_for_multi_draw_bounds(
    make: Callable[[], Objective],
) -> None:
    with pytest.raises(ValueError, match="DoublyReparameterized"):
        make()


def test_sticking_the_landing_would_be_biased_for_iwae() -> None:
    """The rejection is not conservative: the dropped score term has a
    nonzero mean for IWAE, so the STL surrogate (applied directly,
    bypassing the objective's check) has a mean gradient many standard
    errors from the quadrature gradient."""
    torch.manual_seed(0)
    problem = _Problem(theta=_THETA, mu=_MU, sigma=_SIGMA)
    estimator = StickingTheLanding()
    x = torch.zeros(2000, 1)
    rows: list[torch.Tensor] = []
    for _ in range(16):
        particles = _draw_particles(
            problem.model, problem.guide, x, {}, 2, estimator, 1.0
        )
        parameters = [*list(problem.guide.parameters()), problem.theta]
        for parameter in parameters:
            parameter.grad = None
        estimator.surrogate(1.0, particles).mean().backward()
        grads: list[torch.Tensor] = []
        for parameter in parameters:
            assert parameter.grad is not None
            grads.append(parameter.grad.reshape(-1))
        rows.append(torch.cat(grads))
    stacked = torch.stack(rows)
    mean, se = stacked.mean(0), stacked.std(0) / 4.0
    _, exact = _quadrature_expected_bound(1.0, 2)
    assert torch.any((mean - exact).abs() > 8.0 * se), (mean, exact, se)


# ---------------------------------------------------------------------------
# Reweighted wake-sleep
# ---------------------------------------------------------------------------


def test_rws_wake_updates_hold_the_draws_fixed() -> None:
    """Replaying the draws: the guide gradient is
    -sum w * d log q(z_fixed) and the model gradient is
    -sum w * d log p(z_fixed), with w the self-normalised weights."""
    problem = _Problem(theta=_THETA, mu=_MU, sigma=_SIGMA)
    x = torch.zeros(3, 1)
    torch.manual_seed(4)
    loss = RWS(num_particles=5)(problem.model, problem.guide, x, {})
    loss.backward()
    loc = getattr(problem.guide, "loc_z")
    log_scale = getattr(problem.guide, "log_scale_z")
    assert loc.grad is not None and log_scale.grad is not None
    assert problem.theta.grad is not None

    torch.manual_seed(4)
    with torch.no_grad():
        draws = [problem.guide.rsample(x)["z"] for _ in range(5)]
        z = torch.stack(draws)  # (K, batch, 2)
        mu, sigma, theta = loc, log_scale.exp(), problem.theta
        log_p = D.Normal(theta, 1.0).log_prob(z).sum(-1)
        log_q = D.Normal(mu, sigma).log_prob(z).sum(-1)
        w = torch.softmax(log_p - log_q, dim=0)  # (K, batch)
        d_log_q_d_mu = (z - mu) / sigma**2
        d_log_q_d_log_sigma = ((z - mu) / sigma) ** 2 - 1.0
        d_log_p_d_theta = (z - theta).sum(-1)
        # The loss sums the three independent runs' terms.
        expected_mu = -(w[..., None] * d_log_q_d_mu).sum((0, 1))
        expected_log_sigma = -(w[..., None] * d_log_q_d_log_sigma).sum((0, 1))
        expected_theta = -(w * d_log_p_d_theta).sum()
    assert torch.allclose(loc.grad, expected_mu, atol=1e-10)
    assert torch.allclose(log_scale.grad, expected_log_sigma, atol=1e-10)
    assert torch.allclose(problem.theta.grad, expected_theta, atol=1e-10)


def test_rws_guide_gradient_targets_the_inclusive_kl() -> None:
    """With many particles the wake-phase guide gradient approaches
    grad KL(p || q): for p = N(theta, I) and q = N(mu, diag(sigma^2)),
    d/dmu = (mu - theta) / sigma^2 and
    d/dlog sigma = 1 - (1 + (mu - theta)^2) / sigma^2."""
    torch.manual_seed(5)
    problem = _Problem(theta=_THETA, mu=_MU, sigma=_SIGMA)
    runs = 200
    loss = RWS(num_particles=256)(
        problem.model, problem.guide, torch.zeros(runs, 1), {}
    )
    # The mean over the independent runs is the one-run loss.
    (loss / runs).backward()
    mu = torch.tensor(_MU)
    sigma = torch.tensor(_SIGMA)
    expected_mu = (mu - _THETA) / sigma**2
    expected_log_sigma = 1.0 - (1.0 + (mu - _THETA) ** 2) / sigma**2
    loc_grad = getattr(problem.guide, "loc_z").grad
    log_scale_grad = getattr(problem.guide, "log_scale_z").grad
    assert loc_grad is not None and log_scale_grad is not None
    # Self-normalised importance sampling over 256 draws with a wider
    # proposal: bias and noise of the batch mean are below 0.02.
    assert torch.allclose(loc_grad, expected_mu, atol=0.05), loc_grad
    assert torch.allclose(log_scale_grad, expected_log_sigma, atol=0.05), log_scale_grad


def test_rws_takes_no_estimator() -> None:
    with pytest.raises(TypeError):
        RWS(num_particles=4, estimator=Reparameterized())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ChiVI: the exponentiated CUBO of Dieng et al. (2017)
# ---------------------------------------------------------------------------

_CHI_MU = (0.4, -0.2)
_CHI_SIGMA = (1.1, 1.05)
_ORDER = 2.0


def _quadrature_chivi(nodes: int = 40) -> tuple[float, torch.Tensor, float]:
    """``CUBO_n``, ``grad L_n`` and ``E_q[w^{-n}]`` for the ChiVI problem.

    ``L_n = E_q[(p/q)^n] = exp(n CUBO_n)``; gradients are with respect to
    ``(mu_1, mu_2, log sigma_1, log sigma_2, theta)``.
    """
    abscissae, weights = hermite.hermegauss(nodes)
    eps = torch.tensor(abscissae)
    node_weights = torch.tensor(weights) / math.sqrt(2.0 * math.pi)
    e1, e2 = torch.meshgrid(eps, eps, indexing="ij")
    grid = torch.stack([e1.reshape(-1), e2.reshape(-1)], dim=-1)
    grid_weights = (node_weights[:, None] * node_weights[None, :]).reshape(-1)
    mu = torch.tensor(_CHI_MU, requires_grad=True)
    log_sigma = torch.log(torch.tensor(_CHI_SIGMA)).requires_grad_(True)
    theta = torch.tensor(_THETA, requires_grad=True)
    z = mu + log_sigma.exp() * grid
    log_ratio = D.Normal(theta, 1.0).log_prob(z).sum(-1) - D.Normal(
        mu, log_sigma.exp()
    ).log_prob(z).sum(-1)
    moment = (grid_weights * torch.exp(_ORDER * log_ratio)).sum()
    inverse_moment = float((grid_weights * torch.exp(-_ORDER * log_ratio)).sum())
    moment.backward()
    assert mu.grad is not None and log_sigma.grad is not None
    assert theta.grad is not None
    gradient = torch.cat([mu.grad, log_sigma.grad, theta.grad.reshape(1)])
    return float(torch.log(moment.detach())) / _ORDER, gradient, inverse_moment


class _LogOfMeanChiVI(Reparameterized):
    """The log-of-mean surrogate: the gradient of the reported CUBO
    estimate, which Dieng et al. show is biased."""

    def moment_surrogate(
        self, order: float, particles: ParticleLogDensities
    ) -> torch.Tensor:
        return power_mean_bound(
            particles.log_p - particles.log_q, particles.log_weights, order
        )


def _chivi_gradient_check(estimator: GradientEstimator) -> bool:
    """Whether the ChiVI surrogate's mean gradient matches the
    quadrature reference within 4.5 standard errors.

    With two particles the stabilising scale of particle ``k`` is the
    other particle's ``w^n``, independent of particle ``k``, so the
    expected surrogate gradient is ``E_q[w^{-n}] * grad L_n`` exactly.
    """
    torch.manual_seed(0)
    problem = _Problem(theta=_THETA, mu=_CHI_MU, sigma=_CHI_SIGMA)
    mean, se, _ = _gradient_of_bound(
        problem,
        problem.guide,
        ChiVI(_ORDER, num_particles=2, estimator=estimator),
        sign=1.0,
        batch=2000,
        repeats=16,
    )
    _, grad_moment, inverse_moment = _quadrature_chivi()
    expected = inverse_moment * grad_moment
    return bool(torch.all((mean - expected).abs() <= 4.5 * se))


@pytest.mark.parametrize(
    "estimator", [Reparameterized(), ScoreFunction()], ids=["rep", "score"]
)
def test_chivi_gradient_is_unbiased_for_the_exponentiated_cubo(
    estimator: GradientEstimator,
) -> None:
    """Guide and model gradients match ``E_q[w^{-n}] * grad L_n`` from
    quadrature within 4.5 standard errors."""
    assert _chivi_gradient_check(estimator)


def test_chivi_gradient_check_rejects_the_log_of_mean_surrogate() -> None:
    """Mutation check: the log-of-mean surrogate fails the same test."""
    assert not _chivi_gradient_check(_LogOfMeanChiVI())


def test_chivi_reports_a_cubo_estimate_biased_downward_at_finite_k() -> None:
    """The reported value is (1/n) log mean_k w_k^n. Its expectation lies
    below CUBO_n (Jensen) and rises toward it as K grows."""
    torch.manual_seed(1)
    problem = _Problem(theta=_THETA, mu=_CHI_MU, sigma=_CHI_SIGMA)
    cubo, _, _ = _quadrature_chivi()
    x = torch.zeros(2000, 1)
    means: list[float] = []
    for k in (2, 8, 64):
        objective = ChiVI(_ORDER, num_particles=k)
        with torch.no_grad():
            values = torch.tensor(
                [
                    # The mean over the 2000 independent runs.
                    float(objective(problem.model, problem.guide, x, {})) / 2000
                    for _ in range(4)
                ]
            )
        means.append(float(values.mean()))
    # 8000 replicates per K; the per-replicate sd is below 0.5.
    assert means[0] < means[1] < means[2] < cubo + 0.02, (means, cubo)
    assert cubo - means[0] > 0.02, (means, cubo)
    assert abs(means[2] - cubo) < 0.02, (means, cubo)


def test_chivi_rejects_invalid_configurations() -> None:
    with pytest.raises(ValueError, match="n must be > 1"):
        ChiVI(1.0)
    with pytest.raises(ValueError, match="num_particles must be >= 2"):
        ChiVI(2.0, num_particles=1)
    for estimator in (StickingTheLanding(), DoublyReparameterized()):
        with pytest.raises(ValueError, match="ScoreFunction"):
            ChiVI(2.0, estimator=estimator)


def test_chivi_sandwiches_the_evidence_of_a_conjugate_model() -> None:
    """``z ~ N(0, 1)``, ``y ~ N(z, 0.5^2)``, ``y = 0.8``: after fitting
    the guide with ChiVI, ELBO <= log p(y) <= CUBO_2 within Monte Carlo
    error, and the guide is near the exact posterior N(0.64, 0.2). The
    guide starts wider than the posterior, where the gradient estimate
    has finite variance."""
    model = loads(
        "object One : FinSet 1\n"
        "program p : One -> One\n"
        "    sample z <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : One <- Normal(loc=z, scale=0.5)\n"
        "    return z\n"
        "export p\n"
    ).morphism
    assert isinstance(model, MonadicProgram)
    observations = {"y": torch.tensor([0.8])}
    log_evidence = float(D.Normal(0.0, math.sqrt(1.25)).log_prob(torch.tensor(0.8)))
    torch.manual_seed(2)
    guide = AutoNormalGuide(model, observed_names={"y"})
    with torch.no_grad():
        getattr(guide, "log_scale_z").fill_(math.log(1.5))
    optim = torch.optim.Adam(guide.parameters(), lr=0.02)
    svi = SVI(model, guide, optim, ChiVI(2.0, num_particles=16))
    x = torch.zeros(1, 1)
    for _ in range(200):
        svi.step(x, observations)
    with torch.no_grad():
        assert abs(float(getattr(guide, "loc_z")) - 0.64) < 0.1
        assert abs(float(getattr(guide, "log_scale_z").exp()) - math.sqrt(0.2)) < 0.07

    with torch.no_grad():
        elbos = torch.tensor(
            [-float(ELBO()(model, guide, x, observations)) for _ in range(2000)]
        )
        cubos = torch.tensor(
            [
                float(ChiVI(2.0, num_particles=256)(model, guide, x, observations))
                for _ in range(8)
            ]
        )
    elbo_se = float(elbos.std()) / math.sqrt(len(elbos))
    cubo_se = float(cubos.std()) / math.sqrt(len(cubos))
    assert float(elbos.mean()) <= log_evidence + 4.0 * elbo_se
    assert float(cubos.mean()) >= log_evidence - 4.0 * cubo_se - 1e-3
