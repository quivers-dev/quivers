"""Tests for :class:`Predictive` driving from an :class:`MCMCResult`."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from quivers.continuous.family_spec import FAMILY_REGISTRY
from quivers.continuous.inline import _FAMILY_BUILDERS
from quivers.dsl import loads
from quivers.inference import (
    AutoNormalGuide,
    HMCKernel,
    MCMC,
    MCMCResult,
    NUTSKernel,
    Predictive,
    trace,
)


def _normal_normal_model():
    return loads(
        "object Obs : FinSet 12\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def test_predictive_with_mcmc_result_iterates_posterior_draws() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=50,
        num_samples=30,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})

    pred = Predictive(model, result)
    assert pred.num_samples == 2 * 30  # all draws by default

    draws = pred(torch.zeros(1, 1), observations={"y": y})
    assert "mu" in draws
    # First axis is the predictive-sample axis.
    assert draws["mu"].shape[0] == pred.num_samples


def test_predictive_caps_num_samples_at_mcmc_draw_count() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=5, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=10,
        num_samples=15,
        num_chains=1,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    pred = Predictive(model, result, num_samples=1000)
    assert pred.num_samples == 15  # capped at available


def test_predictive_rejects_invalid_posterior_type() -> None:
    import pytest

    model = _normal_normal_model()
    with pytest.raises(TypeError, match="must be Guide or MCMCResult"):
        Predictive(model, "not a posterior")  # type: ignore[arg-type]


def test_predictive_with_guide_still_works() -> None:
    """The Layer 3 behavior — Predictive driven by a Guide —
    must continue to work after the MCMC overload."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    guide = AutoNormalGuide(model, observed_names={"y"})
    pred = Predictive(model, guide, num_samples=5)
    draws = pred(torch.zeros(1, 1), observations={"y": y})
    assert "mu" in draws
    assert draws["mu"].shape[0] == 5


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_predictive_rejects_zero_num_samples() -> None:
    import pytest

    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        Predictive(model, guide, num_samples=0)


def test_predictive_negative_num_samples_rejected() -> None:
    import pytest

    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        Predictive(model, guide, num_samples=-5)


def test_predictive_with_guide_default_num_samples() -> None:
    """When num_samples is omitted, the guide path defaults to 100."""
    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    pred = Predictive(model, guide)
    assert pred.num_samples == 100


# ---------------------------------------------------------------------------
# One shape convention for guide and MCMC posteriors
# ---------------------------------------------------------------------------


def _plate_model():
    return loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    sample by_subj : Subj <- Normal(loc=0.0, scale=sigma)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    observe r : Resp <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


_SUBJ_IDX = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])


@pytest.mark.parametrize("batch", [1, 3])
def test_predictive_shapes_agree_for_guide_and_mcmc_posteriors(batch: int) -> None:
    model = _plate_model()
    observations = {"subj_idx": _SUBJ_IDX, "r": torch.linspace(-1.0, 1.0, 12)}
    result = MCMC(
        NUTSKernel(max_tree_depth=3), num_warmup=5, num_samples=4, num_chains=2, seed=0
    ).run(model, torch.zeros(1, 1), observations)
    guide = AutoNormalGuide(model, observed_names={"r"})
    x = torch.zeros(batch, 1)
    host = {"subj_idx": _SUBJ_IDX}

    from_mcmc = Predictive(model, result, num_samples=6)(x, observations=host)
    from_guide = Predictive(model, guide, num_samples=6)(x, observations=host)

    assert from_mcmc.keys() == from_guide.keys()
    for name in from_mcmc:
        assert from_mcmc[name].shape == from_guide[name].shape, name
    assert from_mcmc["sigma"].shape == (6, batch)
    # A plated site carries the run axis once there is more than one run.
    assert from_mcmc["by_subj"].shape == ((6, 4) if batch == 1 else (6, batch, 4))


def test_predictive_feeds_each_mcmc_draw_unchanged() -> None:
    """Every predictive latent is one recorded posterior draw, shared by
    every run of the input."""
    model = _plate_model()
    observations = {"subj_idx": _SUBJ_IDX, "r": torch.linspace(-1.0, 1.0, 12)}
    result = MCMC(
        NUTSKernel(max_tree_depth=3), num_warmup=5, num_samples=4, num_chains=2, seed=1
    ).run(model, torch.zeros(1, 1), observations)
    draws = Predictive(model, result)(
        torch.zeros(3, 1), observations={"subj_idx": _SUBJ_IDX}
    )
    recorded_sigma = result.samples["sigma"].reshape(-1)
    recorded_by_subj = result.samples["by_subj"].reshape(-1, 4)
    assert draws["sigma"].shape == (8, 3)
    for k in range(8):
        sigma_row = draws["sigma"][k]
        assert torch.equal(sigma_row, sigma_row[:1].expand(3))
        by_subj_rows = draws["by_subj"][k]
        assert torch.equal(by_subj_rows, by_subj_rows[:1].expand(3, 4))
        same_draw = (recorded_sigma == sigma_row[0]) & (
            recorded_by_subj == by_subj_rows[0]
        ).all(dim=-1)
        assert same_draw.any()


def test_predictive_rejects_an_mcmc_result_for_another_model() -> None:
    model = _plate_model()
    stranger = MCMCResult(
        samples={"tau": torch.zeros(1, 2)},
        log_densities=torch.zeros(1, 2),
        acceptance_rates=torch.ones(1),
        divergence_counts=torch.zeros(1, dtype=torch.int64),
        num_warmup=0,
        num_samples=2,
    )
    predictive = Predictive(model, stranger)
    with pytest.raises(ValueError, match="not sample sites of the model"):
        predictive(torch.zeros(1, 1), observations={"subj_idx": _SUBJ_IDX})


# ---------------------------------------------------------------------------
# Every discrete family samples and scores its own draws
# ---------------------------------------------------------------------------

# Arguments for an observe site of each discrete family, built from
# `a ~ Normal`, `q = sigmoid(a)`, `r = exp(a)`, a 3-simplex `pc`, and the
# observed cutpoint plate `cuts`; ``None`` marks a family with no literal form.
_DISCRETE_ARGUMENTS: dict[str, tuple[str, str | None, list[float]]] = {
    "Bernoulli": ("probs=q", "probs=0.3", [0, 1, 1, 0, 1, 1]),
    "Categorical": ("probs=pc", "probs=[0.2, 0.3, 0.5]", [0, 1, 2, 1, 0, 2]),
    "Poisson": ("rate=r", "rate=2.0", [0, 2, 1, 3, 0, 1]),
    "Geometric": ("probs=q", "probs=0.3", [0, 2, 1, 3, 0, 1]),
    "Binomial": ("total_count=5.0, probs=q", None, [0, 2, 5, 3, 1, 4]),
    "NegativeBinomial": (
        "total_count=3.0, probs=q",
        "total_count=3.0, probs=0.4",
        [0, 2, 1, 3, 0, 1],
    ),
    "BetaBinomial": (
        "total_count=5.0, concentration1=r, concentration0=1.0",
        None,
        [0, 2, 5, 3, 1, 4],
    ),
    "ZeroInflatedPoisson": ("zero_prob=q, rate=r", None, [0, 2, 0, 3, 0, 1]),
    "HurdlePoisson": ("zero_prob=q, rate=r", None, [0, 2, 0, 3, 0, 1]),
    "OrderedLogistic": ("predictor=a, cutpoints=cuts", None, [0, 1, 2, 1, 0, 2]),
    "OrderedProbit": (
        "eta=a, cutpoints=cuts",
        "eta=0.5, cutpoints=[-1.0, 1.0]",
        [0, 1, 2, 1, 0, 2],
    ),
}
_HOST = {"cuts": torch.tensor([-1.0, 1.0])}  # observed cutpoints


def test_the_discrete_family_table_covers_every_discrete_family() -> None:
    registered = {name for name, spec in FAMILY_REGISTRY.items() if spec.discrete}
    inline = {name for name, (_, _, discrete) in _FAMILY_BUILDERS.items() if discrete}
    assert set(_DISCRETE_ARGUMENTS) == registered | inline


def _discrete_program(family: str, arguments: str):
    return loads(
        "object Resp : FinSet 6\n"
        "object Cat : FinSet 3\n"
        "object Cut : FinSet 2\n"
        "program p : Resp -> Resp\n"
        "    sample a <- Normal(loc=0.0, scale=1.0)\n"
        "    sample cuts <- Normal(loc=0.0, scale=2.0) [over=Cut]\n"
        "    sample pc : Cat <- Dirichlet(concentration=1.0)\n"
        "    let q = sigmoid(a)\n"
        "    let r = exp(a)\n"
        f"    observe y : Resp <- {family}({arguments})\n"
        "    return a\n"
        "export p\n"
    ).morphism


def _cases() -> list[tuple[str, str]]:
    cases = []
    for family, (mixed, literal, _) in sorted(_DISCRETE_ARGUMENTS.items()):
        cases.append((family, mixed))
        if literal is not None:
            cases.append((family, literal))
    return cases


@pytest.mark.parametrize(("family", "arguments"), _cases())
def test_discrete_observe_sites_sample_and_score_their_draws(
    family: str, arguments: str
) -> None:
    model = _discrete_program(family, arguments)
    x = torch.zeros(1, 1)
    y = torch.tensor(_DISCRETE_ARGUMENTS[family][2], dtype=torch.get_default_dtype())
    support = FAMILY_REGISTRY[family].support if family in FAMILY_REGISTRY else None
    floating = family == "Bernoulli"

    def _check(draws: torch.Tensor) -> None:
        assert draws.dtype.is_floating_point == floating
        assert torch.isfinite(draws.double()).all()
        if support is not None:
            assert support.check(draws).all()

    model.rsample(x, observations=_HOST)
    traced = trace(model, x, observations=_HOST)
    _check(traced.sites["y"].value)
    assert torch.isfinite(traced.sites["y"].log_prob).all()

    guide = AutoNormalGuide(model, observed_names={"y", "cuts"})
    _check(Predictive(model, guide, num_samples=3)(x, observations=_HOST)["y"])

    fit = MCMC(
        NUTSKernel(max_tree_depth=3), num_warmup=5, num_samples=4, num_chains=1, seed=0
    ).run(model, x, {"y": y, **_HOST})
    _check(Predictive(model, fit, num_samples=3)(x, observations=_HOST)["y"])


# ---------------------------------------------------------------------------
# Ordinal families sample and score their closed-form category law
# ---------------------------------------------------------------------------


def _probit_cdf(z: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Normal(0.0, 1.0).cdf(z)


# The cumulative distribution function of each ordinal family's latent noise.
_ORDINAL_CDFS = {"OrderedLogistic": torch.sigmoid, "OrderedProbit": _probit_cdf}


def _category_probabilities(
    cdf: Callable[[torch.Tensor], torch.Tensor],
    predictor: torch.Tensor,
    cutpoints: torch.Tensor,
) -> torch.Tensor:
    """``P(Y = k) = F(c_k - eta) - F(c_{k-1} - eta)``, in float64."""
    offsets = cutpoints.double() - predictor.double().unsqueeze(-1)
    below = cdf(offsets)
    zeros = torch.zeros_like(below[..., :1])
    ones = torch.ones_like(below[..., :1])
    return torch.cat([below, ones], dim=-1) - torch.cat([zeros, below], dim=-1)


@pytest.mark.parametrize("family", sorted(_ORDINAL_CDFS))
def test_ordinal_observe_sites_follow_their_closed_form_law(family: str) -> None:
    torch.manual_seed(20261112)
    model = _discrete_program(family, _DISCRETE_ARGUMENTS[family][0])
    x = torch.zeros(1, 1)
    cuts = _HOST["cuts"]
    cdf = _ORDINAL_CDFS[family]
    wrong = _ORDINAL_CDFS[next(name for name in _ORDINAL_CDFS if name != family)]

    # The traced log density of the responses is the closed form's sum.
    y = torch.tensor(_DISCRETE_ARGUMENTS[family][2])
    traced = trace(model, x, observations={"y": y, **_HOST})
    eta = traced.sites["a"].value.reshape(())
    scored = float(traced.sites["y"].log_prob.sum())
    expected = float(_category_probabilities(cdf, eta, cuts)[y].log().sum())
    other = float(_category_probabilities(wrong, eta, cuts)[y].log().sum())
    assert scored == pytest.approx(expected, abs=1e-4)
    assert scored != pytest.approx(other, abs=1e-4)

    # Posterior-predictive draws follow the category law at each draw's
    # predictor: each category's count sits within four standard
    # deviations of its expected count, and the other family's law is
    # rejected by the same test.
    guide = AutoNormalGuide(model, observed_names={"y", "cuts"})
    draws = Predictive(model, guide, num_samples=500)(x, observations=_HOST)
    responses = draws["y"].reshape(500, -1)
    predictors = draws["a"].reshape(500, 1).expand_as(responses)

    def _within(law: Callable[[torch.Tensor], torch.Tensor]) -> bool:
        probs = _category_probabilities(law, predictors, cuts)
        counts = torch.stack([(responses == k).sum() for k in range(3)]).double()
        mean = probs.sum(dim=(0, 1))
        sd = (probs * (1 - probs)).sum(dim=(0, 1)).sqrt()
        return bool(((counts - mean).abs() <= 4 * sd).all())

    assert _within(cdf)
    assert not _within(wrong)
