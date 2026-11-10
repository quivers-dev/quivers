"""Tests for the mixture variational guide.

`AutoMixtureGuide` scores points under the mixture density
``q(z) = sum_k pi_k q_k(z)``. Its ``rsample`` draws from that density
(a categorical component pick, then the component's draw), and its
``rsample_strata`` enumerates the components so the objectives
estimate the stratified ELBO ``sum_k pi_k E_{q_k}[log p - log q]`` and
the stratified importance-weighted bounds, with gradients to the
mixture logits through the finite sum.

The tests pin the mathematics against exact references: one-dimensional
quadrature of the ELBO and its gradient, and a mixture guide equal to a
known mixture target, where every bound equals ``log Z``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch.distributions as D

from quivers.continuous import Draw, Euclidean, MonadicProgram, make_fixed_normal
from quivers.continuous.inline import FixedDistribution
from quivers.core import Unit
from quivers.dsl import loads
from quivers.inference import (
    ELBO,
    RWS,
    SVI,
    AutoGuideList,
    AutoMixtureGuide,
    AutoNormalGuide,
    ChiVI,
    DoublyReparameterized,
    DReGsBound,
    IWAEBound,
    RenyiBound,
    Reparameterized,
    StickingTheLanding,
    VRIWAEBound,
)
from quivers.inference.guides.base import Guide

_REAL = Euclidean(name="R", dim=1)
_X = torch.zeros(1, 1)


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# Models and references
# ---------------------------------------------------------------------------


def _hierarchical_model() -> MonadicProgram:
    program = loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(probs=mu)\n"
        "    return mu\n"
        "export p\n"
    ).morphism
    assert isinstance(program, MonadicProgram)
    return program


def _make_obs() -> dict[str, torch.Tensor]:
    return {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }


def _standard_normal_model(names: tuple[str, ...] = ("z",)) -> MonadicProgram:
    """Independent ``Normal(0, 1)`` latents and no observations: log Z = 0."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(names=(name,), morphism=make_fixed_normal(0.0, 1.0, _REAL))
            for name in names
        ],
        return_vars=(),
    )


_TARGET_WEIGHTS = (0.3, 0.7)
_TARGET_LOCS = (-3.0, 3.0)
_TARGET_SCALES = (0.5, 0.5)


def _mixture_target_model() -> MonadicProgram:
    """``z ~ 0.3 N(-3, 0.5^2) + 0.7 N(3, 0.5^2)``, no observations: log Z = 0."""
    weights = torch.tensor(_TARGET_WEIGHTS)
    locs = torch.tensor(_TARGET_LOCS).reshape(1, 2, 1)
    scales = torch.tensor(_TARGET_SCALES).reshape(1, 2, 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        del device
        return D.MixtureSameFamily(
            D.Categorical(probs=weights.expand(batch, 2)),
            D.Independent(
                D.Normal(locs.expand(batch, 2, 1), scales.expand(batch, 2, 1)), 1
            ),
        )

    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[Draw(names=("z",), morphism=FixedDistribution(_REAL, builder))],
        return_vars=(),
    )


def _set_normal(component: Guide, loc: float, scale: float, site: str = "z") -> None:
    with torch.no_grad():
        getattr(component, f"loc_{site}").fill_(loc)
        getattr(component, f"log_scale_{site}").fill_(math.log(scale))


def _mixture(
    model: MonadicProgram,
    locs: tuple[float, ...],
    scales: tuple[float, ...],
    weights: tuple[float, ...] | None = None,
) -> AutoMixtureGuide:
    components: list[Guide] = []
    for loc, scale in zip(locs, scales):
        component = AutoNormalGuide(model, observed_names=set())
        _set_normal(component, loc, scale)
        components.append(component)
    guide = AutoMixtureGuide(components)
    if weights is not None:
        with torch.no_grad():
            guide.mixture_logits.copy_(torch.log(torch.tensor(weights)))
    return guide


def _quadrature_elbo(
    logits: torch.Tensor,
    locs: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    """``E_q[log N(z; 0, 1) - log q(z)]`` for a 1-D Normal mixture ``q``.

    Trapezoidal quadrature on a fine grid; differentiable in every
    argument, so ``autograd`` gives the exact gradient up to the
    quadrature error (far below the Monte Carlo tolerances used here).
    """
    grid = torch.linspace(-25.0, 25.0, 500_001)
    log_pi = torch.log_softmax(logits, dim=0)
    components = D.Normal(locs.unsqueeze(-1), log_scales.exp().unsqueeze(-1))
    log_q = torch.logsumexp(log_pi.unsqueeze(-1) + components.log_prob(grid), dim=0)
    log_p = D.Normal(0.0, 1.0).log_prob(grid)
    return torch.trapezoid(log_q.exp() * (log_p - log_q), grid)


def _separated_mixture_elbo() -> float:
    """Quadrature ELBO of 0.5 N(-5, 0.1^2) + 0.5 N(5, 0.1^2) against N(0, 1)."""
    return float(
        _quadrature_elbo(
            torch.zeros(2),
            torch.tensor([-5.0, 5.0]),
            torch.log(torch.tensor([0.1, 0.1])),
        )
    )


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_mixture_rsample_returns_constrained_shapes() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    samples = guide.rsample(torch.zeros(1, 1))
    ref = comp1.rsample(torch.zeros(1, 1))
    for name in ref:
        assert samples[name].shape == ref[name].shape


def test_mixture_log_prob_is_logsumexp_of_components() -> None:
    """log q(z) = logsumexp_k (log pi_k + log q_k(z))."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    samples = comp1.rsample(torch.zeros(1, 1))
    log_q_mix = guide.log_prob(torch.zeros(1, 1), samples)
    log_q_1 = comp1.log_prob(torch.zeros(1, 1), samples)
    log_q_2 = comp2.log_prob(torch.zeros(1, 1), samples)
    expected = math.log(0.5) + torch.logsumexp(torch.stack([log_q_1, log_q_2]), dim=0)
    assert torch.allclose(log_q_mix, expected, atol=1e-10)


def test_mixture_gradients_flow_to_logits_and_components() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    loss = ELBO()(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    assert guide.mixture_logits.grad is not None
    assert torch.isfinite(guide.mixture_logits.grad).all()
    for comp in guide.components:
        for param in comp.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_mixture_rejects_mismatched_component_registries() -> None:
    torch.manual_seed(0)
    model1 = _hierarchical_model()
    comp1 = AutoNormalGuide(model1, observed_names={"r"})
    comp2 = AutoNormalGuide(model1, observed_names={"r", "by_subj"})
    with pytest.raises(ValueError, match="different latent names"):
        AutoMixtureGuide([comp1, comp2])


def test_mixture_rejects_single_component() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    comp = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="at least 2 components"):
        AutoMixtureGuide([comp])


def test_mixture_runs_end_to_end_svi() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=1e-2
    )
    svi = SVI(model, guide, optim, ELBO())
    losses = [svi.step(torch.zeros(1, 1), _make_obs()) for _ in range(20)]
    for loss in losses:
        assert math.isfinite(loss)


def test_mixture_num_components_property() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    comps: list[Guide] = [
        AutoNormalGuide(model, observed_names={"r"}) for _ in range(3)
    ]
    assert AutoMixtureGuide(comps).num_components == 3


# ---------------------------------------------------------------------------
# Sampling follows the mixture density
# ---------------------------------------------------------------------------


def test_rsample_draws_follow_the_mixture_density() -> None:
    """Separated components at -5 and 5 with scale 0.1: no mass in |z| < 4."""
    torch.manual_seed(0)
    model = _standard_normal_model()
    guide = _mixture(model, (-5.0, 5.0), (0.1, 0.1))
    with torch.no_grad():
        draws = torch.cat([guide.rsample(_X)["z"].reshape(-1) for _ in range(4000)])
    assert float((draws.abs() < 4.0).double().mean()) == 0.0
    # Each component is picked with probability 1/2 (binomial SE 0.0079).
    assert abs(float((draws > 0).double().mean()) - 0.5) < 0.04


def test_rsample_picks_components_with_the_mixture_weights() -> None:
    torch.manual_seed(1)
    model = _standard_normal_model()
    guide = _mixture(model, (-5.0, 5.0), (0.1, 0.1), weights=(0.3, 0.7))
    with torch.no_grad():
        draws = torch.cat([guide.rsample(_X)["z"].reshape(-1) for _ in range(4000)])
    # Binomial SE for p = 0.7, n = 4000 is 0.0072.
    assert abs(float((draws > 0).double().mean()) - 0.7) < 0.03
    upper = draws[draws > 0]
    assert abs(float(upper.mean()) - 5.0) < 0.02
    assert abs(float(upper.std()) - 0.1) < 0.01


def test_rsample_strata_enumerate_components_with_log_mixture_weights() -> None:
    torch.manual_seed(0)
    model = _standard_normal_model()
    guide = _mixture(model, (-5.0, 5.0), (0.1, 0.1), weights=(0.3, 0.7))
    log_weights, draws = guide.rsample_strata(_X)
    assert len(draws) == 2
    assert torch.allclose(log_weights, torch.log(torch.tensor([0.3, 0.7])))
    assert float(draws[0]["z"].detach().reshape(-1)[0]) < 0.0
    assert float(draws[1]["z"].detach().reshape(-1)[0]) > 0.0
    log_weights.sum().backward()
    assert guide.mixture_logits.grad is not None


# ---------------------------------------------------------------------------
# Objectives estimate the mixture density's bounds
# ---------------------------------------------------------------------------


def test_elbo_of_separated_mixture_is_a_lower_bound() -> None:
    """Target N(0, 1) with log Z = 0; guide 0.5 N(-5, 0.1^2) + 0.5 N(5, 0.1^2).

    The estimate must be at most log Z = 0 and match the quadrature
    value of the mixture's ELBO within Monte Carlo error (per-stratum
    standard deviation below 1, so the SE at 1000 particles is < 0.03).
    """
    torch.manual_seed(0)
    model = _standard_normal_model()
    guide = _mixture(model, (-5.0, 5.0), (0.1, 0.1))
    with torch.no_grad():
        estimate = -float(ELBO(num_particles=1000)(model, guide, _X, {}))
    assert estimate <= 0.0
    assert abs(estimate - _separated_mixture_elbo()) < 0.15


@pytest.mark.parametrize(
    "objective",
    [
        IWAEBound(num_particles=64, estimator=Reparameterized()),
        RenyiBound(alpha=0.5, num_particles=64),
        VRIWAEBound(alpha=0.0, num_particles=64),
        VRIWAEBound(alpha=0.5, num_particles=64),
    ],
    ids=["iwae", "renyi", "vriwae-0", "vriwae-0.5"],
)
def test_importance_weighted_bounds_of_separated_mixture_stay_below_log_z(
    objective: IWAEBound | RenyiBound | VRIWAEBound,
) -> None:
    torch.manual_seed(0)
    model = _standard_normal_model()
    guide = _mixture(model, (-5.0, 5.0), (0.1, 0.1))
    with torch.no_grad():
        bound = -float(objective(model, guide, _X, {}))
    assert bound <= 0.0
    # These bounds dominate the ELBO in expectation.
    assert bound >= _separated_mixture_elbo() - 0.5


@pytest.mark.parametrize(
    "objective",
    [
        ELBO(num_particles=16),
        ELBO(num_particles=16, estimator=StickingTheLanding()),
        IWAEBound(num_particles=16, estimator=Reparameterized()),
        RenyiBound(alpha=0.5, num_particles=16),
        RenyiBound(alpha=-1.0, num_particles=16),
        VRIWAEBound(alpha=0.3, num_particles=16),
    ],
    ids=["elbo", "elbo-stl", "iwae", "renyi-0.5", "renyi-neg", "vriwae"],
)
def test_bounds_equal_log_z_when_the_guide_is_the_target(
    objective: ELBO | IWAEBound | RenyiBound | VRIWAEBound,
) -> None:
    """With q equal to the target mixture, log p - log q = 0 at every draw."""
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, _TARGET_LOCS, _TARGET_SCALES, weights=_TARGET_WEIGHTS)
    with torch.no_grad():
        value = -float(objective(model, guide, _X, {}))
    assert abs(value) < 1e-10


def test_chi_upper_bound_equals_log_z_when_the_guide_is_the_target() -> None:
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, _TARGET_LOCS, _TARGET_SCALES, weights=_TARGET_WEIGHTS)
    with torch.no_grad():
        cubo = float(ChiVI(n=2.0, num_particles=16)(model, guide, _X, {}))
    assert abs(cubo) < 1e-10


def test_sticking_the_landing_gradient_vanishes_at_the_target() -> None:
    """At q = p the STL integrand log p - log q_detached is identically zero."""
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, _TARGET_LOCS, _TARGET_SCALES, weights=_TARGET_WEIGHTS)
    loss = ELBO(num_particles=8, estimator=StickingTheLanding())(model, guide, _X, {})
    loss.backward()
    for name, parameter in guide.named_parameters():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().max()) < 1e-8, name


def _elbo_gradient_samples(
    model: MonadicProgram,
    guide: AutoMixtureGuide,
    objective: ELBO,
    repeats: int,
) -> torch.Tensor:
    """Stack ``repeats`` independent estimates of the ELBO's gradient.

    Returns shape ``(repeats, 6)``: the two logits, the two component
    locations, and the two component log-scales.
    """
    rows: list[torch.Tensor] = []
    for _ in range(repeats):
        guide.zero_grad()
        objective(model, guide, _X, {}).backward()
        logits_grad = guide.mixture_logits.grad
        assert logits_grad is not None
        row = [-logits_grad]
        for attribute in ("loc_z", "log_scale_z"):
            for component in guide.components:
                grad = getattr(component, attribute).grad
                assert grad is not None
                row.append(-grad.reshape(-1))
        rows.append(torch.cat(row))
    return torch.stack(rows, dim=0)


def test_reparameterized_gradient_is_zero_in_mean_at_the_target() -> None:
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, _TARGET_LOCS, _TARGET_SCALES, weights=_TARGET_WEIGHTS)
    samples = _elbo_gradient_samples(model, guide, ELBO(num_particles=25), repeats=40)
    mean = samples.mean(dim=0)
    standard_error = samples.std(dim=0) / math.sqrt(samples.shape[0])
    assert torch.all(mean.abs() <= 4.0 * standard_error + 1e-8), (mean, standard_error)


def test_elbo_gradient_matches_quadrature() -> None:
    """The ELBO gradient for logits and component parameters is unbiased.

    Target N(0, 1); guide with overlapping, unequal components so the
    logit gradient is far from zero. The exact gradient comes from
    differentiating the quadrature ELBO. The Monte Carlo mean must
    agree within 4 standard errors, and the logit gradient must be
    large compared with that tolerance so a missing term is detected.
    """
    torch.manual_seed(0)
    model = _standard_normal_model()
    locs = (0.2, 2.5)
    scales = (0.9, 0.5)
    guide = _mixture(model, locs, scales)
    with torch.no_grad():
        guide.mixture_logits.copy_(torch.tensor([0.3, -0.2]))

    logits = guide.mixture_logits.detach().clone().requires_grad_(True)
    loc_t = torch.tensor(locs, requires_grad=True)
    log_scale_t = torch.log(torch.tensor(scales)).requires_grad_(True)
    _quadrature_elbo(logits, loc_t, log_scale_t).backward()
    assert logits.grad is not None
    assert loc_t.grad is not None
    assert log_scale_t.grad is not None
    exact = torch.cat([logits.grad, loc_t.grad, log_scale_t.grad])

    samples = _elbo_gradient_samples(model, guide, ELBO(num_particles=25), repeats=40)
    mean = samples.mean(dim=0)
    standard_error = samples.std(dim=0) / math.sqrt(samples.shape[0])
    assert torch.all((mean - exact).abs() <= 4.0 * standard_error + 1e-3), (
        mean,
        exact,
        standard_error,
    )
    assert float(exact[:2].abs().min()) > 8.0 * float(standard_error[:2].max())


def test_elbo_logit_gradient_matches_finite_difference_of_the_estimate() -> None:
    """With common random numbers the estimate is a smooth function of
    the logits; its autograd derivative equals its central difference."""
    model = _standard_normal_model()
    guide = _mixture(model, (-1.0, 1.5), (0.5, 0.8))
    objective = ELBO(num_particles=4)
    torch.manual_seed(3)
    guide.zero_grad()
    objective(model, guide, _X, {}).backward()
    grad = guide.mixture_logits.grad
    assert grad is not None
    analytic = -grad.clone()
    step = 1e-5
    for k in range(2):
        values: list[float] = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                guide.mixture_logits[k] += sign * step
            torch.manual_seed(3)
            with torch.no_grad():
                values.append(-float(objective(model, guide, _X, {})))
            with torch.no_grad():
                guide.mixture_logits[k] -= sign * step
        numeric = (values[0] - values[1]) / (2 * step)
        assert abs(numeric - float(analytic[k])) < 1e-6


# ---------------------------------------------------------------------------
# Estimators and objectives over stratified particles
# ---------------------------------------------------------------------------


def test_doubly_reparameterized_rejects_stratified_particles() -> None:
    torch.manual_seed(0)
    model = _standard_normal_model()
    guide = _mixture(model, (-1.0, 1.0), (0.5, 0.5))
    with pytest.raises(RuntimeError, match="stratified"):
        DReGsBound(num_particles=4)(model, guide, _X, {})
    with pytest.raises(RuntimeError, match="stratified"):
        IWAEBound(num_particles=4, estimator=DoublyReparameterized())(
            model, guide, _X, {}
        )


def test_rws_weights_are_the_stratified_importance_weights() -> None:
    """At q = p the self-normalised weights reduce to the stratum weights.

    Both RWS phases then contribute ``-sum_n omega_n log q(z_n)``, with
    ``omega_n`` the stratum weight over the particle count.
    """
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, _TARGET_LOCS, _TARGET_SCALES, weights=_TARGET_WEIGHTS)
    num_particles = 3
    torch.manual_seed(5)
    with torch.no_grad():
        loss = float(RWS(num_particles=num_particles)(model, guide, _X, {}))
    torch.manual_seed(5)
    expected = 0.0
    with torch.no_grad():
        for _ in range(num_particles):
            log_weights, draws = guide.rsample_strata(_X)
            for log_weight, draw in zip(log_weights, draws):
                log_q = float(guide.log_prob(_X, draw).sum())
                expected += -2.0 * math.exp(float(log_weight)) / num_particles * log_q
    assert abs(loss - expected) < 1e-9


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_nested_mixture_enumerates_inner_components() -> None:
    torch.manual_seed(0)
    model = _standard_normal_model()
    inner_a = _mixture(model, (-2.0, -1.0), (0.4, 0.6), weights=(0.25, 0.75))
    inner_b = _mixture(model, (1.0, 2.5), (0.5, 0.3), weights=(0.6, 0.4))
    outer = AutoMixtureGuide([inner_a, inner_b])
    with torch.no_grad():
        outer.mixture_logits.copy_(torch.log(torch.tensor([0.4, 0.6])))
    log_weights, draws = outer.rsample_strata(_X)
    assert len(draws) == 4
    expected = torch.tensor([0.4 * 0.25, 0.4 * 0.75, 0.6 * 0.6, 0.6 * 0.4])
    assert torch.allclose(log_weights.exp(), expected)

    exact = float(
        _quadrature_elbo(
            torch.log(expected),
            torch.tensor([-2.0, -1.0, 1.0, 2.5]),
            torch.log(torch.tensor([0.4, 0.6, 0.5, 0.3])),
        )
    )
    with torch.no_grad():
        estimate = -float(ELBO(num_particles=400)(model, outer, _X, {}))
    # Per-stratum standard deviation is below 2; SE at 400 particles < 0.1.
    assert abs(estimate - exact) < 0.25


def test_mixture_inside_guide_list_is_enumerated() -> None:
    """A mixture part of an AutoGuideList contributes its strata.

    Model: independent N(0, 1) latents ``a`` and ``b``. Part ``b`` is
    the exact posterior, so the joint ELBO equals the mixture part's
    ELBO, known by quadrature.
    """
    torch.manual_seed(0)
    model = _standard_normal_model(("a", "b"))
    mixture_components: list[Guide] = []
    for loc, scale in ((-5.0, 0.1), (5.0, 0.1)):
        component = AutoNormalGuide(model, observed_names={"b"})
        _set_normal(component, loc, scale, site="a")
        mixture_components.append(component)
    exact_part = AutoNormalGuide(model, observed_names={"a"})
    _set_normal(exact_part, 0.0, 1.0, site="b")
    guide = AutoGuideList({"a": AutoMixtureGuide(mixture_components), "b": exact_part})

    log_weights, draws = guide.rsample_strata(_X)
    assert len(draws) == 2
    assert abs(float(torch.logsumexp(log_weights.detach(), dim=0))) < 1e-12
    assert all(set(draw) == {"a", "b"} for draw in draws)

    with torch.no_grad():
        a_draws = torch.cat([guide.rsample(_X)["a"].reshape(-1) for _ in range(2000)])
    assert float((a_draws.abs() < 4.0).double().mean()) == 0.0

    with torch.no_grad():
        estimate = -float(ELBO(num_particles=1000)(model, guide, _X, {}))
    assert estimate <= 0.0
    assert abs(estimate - _separated_mixture_elbo()) < 0.15


# ---------------------------------------------------------------------------
# Fitting a bimodal target
# ---------------------------------------------------------------------------


def test_fit_recovers_both_modes_and_the_mixture_weights() -> None:
    """Target 0.3 N(-3, 0.5^2) + 0.7 N(3, 0.5^2).

    Starting from overlapping components at -1 and 1 with equal
    weights, SVI on the stratified ELBO must recover each mode's
    location within 0.15, its scale within 0.1, and the weights
    within 0.05.
    """
    torch.manual_seed(0)
    model = _mixture_target_model()
    guide = _mixture(model, (-1.0, 1.0), (1.0, 1.0))
    optim = torch.optim.Adam(guide.parameters(), lr=0.05)
    svi = SVI(model, guide, optim, ELBO(num_particles=4))
    for _ in range(600):
        svi.step(_X, {})

    locs = torch.stack(
        [getattr(c, "loc_z").detach().reshape(()) for c in guide.components]
    )
    scales = torch.stack(
        [getattr(c, "log_scale_z").detach().exp().reshape(()) for c in guide.components]
    )
    order = torch.argsort(locs)
    weights = guide.mixture_weights().detach()[order]
    assert torch.allclose(locs[order], torch.tensor(_TARGET_LOCS), atol=0.15), locs
    assert torch.allclose(scales[order], torch.tensor(_TARGET_SCALES), atol=0.1), scales
    assert torch.allclose(weights, torch.tensor(_TARGET_WEIGHTS), atol=0.05), weights
