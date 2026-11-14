"""Tests for the composable MCMC kernels.

Every stationarity check compares the chain's draws with a target known
in closed form: a Kolmogorov-Smirnov statistic for continuous sites and
a Pearson chi-square for discrete ones, each at level 0.001 with the
sample size replaced by the chain's effective sample size (capped at the
number of draws). Each check is paired with a deliberately wrong
variant (a Metropolis move without its Hastings correction, a Gibbs
update that ignores the likelihood) that must fail the same check.

Cover:

1. `MetropolisKernel` with asymmetric proposals (a drifting random
   walk, a defensive independence proposal) on a standard normal.
2. `GibbsKernel` by enumeration on a scalar Bernoulli site cycled and
   mixed with NUTS, on a plate of Categorical sites alone, and with a
   user-supplied conjugate update of a continuous site.
3. The composite plumbing: gradient caching across components,
   generator and adaptation forwarding, per-component acceptance, Gibbs
   sites in the samples, and the errors.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import pytest
import torch
from torch.distributions import Independent, MultivariateNormal, Normal

from quivers.continuous import (
    Draw,
    Euclidean,
    MonadicProgram,
    PlateDraw,
    make_fixed_normal,
)
from quivers.continuous.inline import make_fixed_bernoulli, make_fixed_categorical
from quivers.continuous.program_steps import Score, reading
from quivers.core import FinSet, Unit
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    GibbsUpdate,
    IndependenceProposal,
    KernelState,
    MCMCKernel,
    MetropolisKernel,
    MixtureKernel,
    NUTSKernel,
    PotentialFn,
    Proposal,
    RandomWalkProposal,
)
from quivers.diagnostics import ess as effective_sample_size
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


_KS_CRITICAL = 1.9495
"""Asymptotic one-sample Kolmogorov-Smirnov critical value at level 0.001,
``sqrt(-log(0.0005) / 2)``."""

_LEVEL = 1e-3


def _effective_size(draws: torch.Tensor) -> float:
    """ESS of ``(chains, n)`` draws, capped at the number of draws."""
    return min(
        float(effective_sample_size(draws.to(torch.float64), method="mean")),
        float(draws.numel()),
    )


def _ks_exceeds(draws: torch.Tensor, cdf: torch.Tensor) -> tuple[float, float]:
    """The KS statistic of ``(chains, n)`` draws against a CDF evaluated
    at their sorted values, and its ESS-based critical value."""
    n = draws.numel()
    empirical_hi = torch.arange(1, n + 1, dtype=torch.float64) / n
    empirical_lo = torch.arange(0, n, dtype=torch.float64) / n
    statistic = float(
        torch.maximum((empirical_hi - cdf).abs(), (cdf - empirical_lo).abs()).max()
    )
    return statistic, _KS_CRITICAL / math.sqrt(_effective_size(draws))


def _normal_cdf(values: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    return Normal(loc, scale).cdf(values.to(torch.float64))


def _chi_square_p(
    counts: torch.Tensor, probs: torch.Tensor, ess: float, n: int
) -> float:
    """p-value of a Pearson chi-square whose statistic is scaled by
    ``ess / n`` for autocorrelated draws."""
    expected = probs * n
    statistic = float(((counts - expected) ** 2 / expected).sum()) * ess / n
    dof = probs.numel() - 1
    return float(
        torch.special.gammaincc(
            torch.tensor(dof / 2.0, dtype=torch.float64),
            torch.tensor(statistic / 2.0, dtype=torch.float64),
        )
    )


def _normal_program() -> MonadicProgram:
    """``z ~ Normal(0, 1)``, nothing observed."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            )
        ],
        return_vars=(),
    )


# ---------------------------------------------------------------------------
# Metropolis-Hastings with asymmetric proposals
# ---------------------------------------------------------------------------


class _DriftingWalk(Proposal):
    """``z' = z + drift + N(0, 1)``: asymmetric for a non-zero drift."""

    def __init__(self, drift: float, corrected: bool = True) -> None:
        self._drift = drift
        self._corrected = corrected

    def propose(
        self, position: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = self._drift + torch.randn(position.shape, generator=generator)
        candidate = position + step
        increment = Normal(self._drift, 1.0)
        correction = (
            increment.log_prob(position - candidate).sum()
            - increment.log_prob(candidate - position).sum()
        )
        return candidate, correction if self._corrected else torch.zeros(())


class _UncorrectedIndependence(IndependenceProposal):
    """The independence proposal with its Hastings correction dropped."""

    def propose(
        self, position: torch.Tensor, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate, _ = super().propose(position, generator)
        return candidate, torch.zeros(())


def _independence(cls: type[IndependenceProposal]) -> IndependenceProposal:
    return cls(
        Independent(Normal(torch.tensor([1.5]), torch.tensor([1.0])), 1),
        defensive=Independent(Normal(torch.tensor([0.0]), torch.tensor([4.0])), 1),
        defensive_weight=0.2,
    )


def _metropolis_draws(proposal: Proposal) -> torch.Tensor:
    result = MCMC(
        MetropolisKernel(proposal),
        num_warmup=200,
        num_samples=3000,
        num_chains=2,
        seed=1,
    ).run(_normal_program(), torch.zeros(1, 1), {})
    return result.samples["z"]


@pytest.mark.parametrize(
    ("make", "wrong"),
    [
        (lambda: _DriftingWalk(0.8), lambda: _DriftingWalk(0.8, corrected=False)),
        (
            lambda: _independence(IndependenceProposal),
            lambda: _independence(_UncorrectedIndependence),
        ),
    ],
    ids=["drifting-walk", "defensive-independence"],
)
def test_metropolis_hastings_targets_the_posterior_and_needs_its_correction(
    make: Callable[[], Proposal], wrong: Callable[[], Proposal]
) -> None:
    draws = _metropolis_draws(make())
    values, _ = draws.reshape(-1).sort()
    ks, critical = _ks_exceeds(draws, _normal_cdf(values, 0.0, 1.0))
    assert ks < critical, (ks, critical)

    bad = _metropolis_draws(wrong())
    bad_values, _ = bad.reshape(-1).sort()
    bad_ks, bad_critical = _ks_exceeds(bad, _normal_cdf(bad_values, 0.0, 1.0))
    assert bad_ks > bad_critical, (bad_ks, bad_critical)


def test_independence_proposal_scores_the_defensive_mixture() -> None:
    g = MultivariateNormal(torch.zeros(2), torch.eye(2))
    d = MultivariateNormal(torch.ones(2), 9.0 * torch.eye(2))
    proposal = IndependenceProposal(g, defensive=d, defensive_weight=0.25)
    value = torch.tensor([0.3, -2.0])
    expected = torch.log(
        0.75 * g.log_prob(value).exp() + 0.25 * d.log_prob(value).exp()
    )
    assert torch.allclose(proposal.log_prob(value), expected)
    with pytest.raises(ValueError, match="defensive_weight must be in"):
        IndependenceProposal(g, defensive=d, defensive_weight=1.0)
    with pytest.raises(ValueError, match="event shape"):
        IndependenceProposal(Normal(torch.zeros(2), 1.0))


def test_random_walk_proposal_checks_its_scale() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        RandomWalkProposal(0.0)
    registry = LatentRegistry.from_model(_normal_program(), set())
    with pytest.raises(ValueError, match="flat vector has length 1"):
        RandomWalkProposal(torch.ones(3)).bind(registry)


# ---------------------------------------------------------------------------
# Gibbs blocks
# ---------------------------------------------------------------------------

_Y = 2.0


def _switch_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    """``mu | z ~ N(3 z, 1)`` against the ``N(0, 1)`` prior draw, and
    ``y = 2 ~ N(mu, 1)``."""
    z = env["z"].reshape(())
    mu = env["mu"].reshape(())
    shift = Normal(3.0 * z, 1.0).log_prob(mu) - Normal(0.0, 1.0).log_prob(mu)
    return (shift + Normal(mu, 1.0).log_prob(torch.tensor(_Y))).reshape(1)


def _switch_program() -> MonadicProgram:
    """``z ~ Bernoulli(0.3)``, ``mu | z ~ N(3 z, 1)``, ``y | mu ~ N(mu, 1)``."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_bernoulli(0.3, FinSet(name="B", cardinality=2)),
            ),
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=reading(_switch_likelihood, ("z", "mu"))),
        ],
        return_vars=(),
    )


def _switch_posterior() -> tuple[torch.Tensor, torch.Tensor]:
    """``p(z | y)`` and ``E[mu | z, y]``: ``y | z ~ N(3 z, 2)`` and
    ``mu | z, y ~ N((3 z + y) / 2, 1 / 2)``."""
    y = torch.tensor(_Y, dtype=torch.float64)
    log_w = torch.stack(
        [
            math.log(0.7) + Normal(0.0, math.sqrt(2.0)).log_prob(y),
            math.log(0.3) + Normal(3.0, math.sqrt(2.0)).log_prob(y),
        ]
    )
    return torch.softmax(log_w, 0), (3.0 * torch.tensor([0.0, 1.0]) + _Y) / 2.0


def _switch_checks(
    result_z: torch.Tensor, result_mu: torch.Tensor
) -> tuple[float, float, float]:
    """The chi-square p-value of ``z``, and the KS statistic of ``mu``
    with its critical value."""
    probs, means = _switch_posterior()
    z = result_z.to(torch.float64)
    counts = torch.stack([(z == 0).sum(), (z == 1).sum()]).to(torch.float64)
    p_value = _chi_square_p(counts, probs, _effective_size(z), z.numel())
    values, _ = result_mu.reshape(-1).to(torch.float64).sort()
    sd = math.sqrt(0.5)
    cdf = probs[0] * _normal_cdf(values, float(means[0]), sd) + probs[1] * _normal_cdf(
        values, float(means[1]), sd
    )
    ks, critical = _ks_exceeds(result_mu, cdf)
    return p_value, ks, critical


def _prior_switch(
    generator: torch.Generator | None,
    gibbs_values: dict[str, torch.Tensor],
    latent_values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A wrong Gibbs update: redraw ``z`` from its prior."""
    del latent_values
    draw = (torch.rand((), generator=generator) < 0.3).to(gibbs_values["z"].dtype)
    return {"z": draw.reshape(gibbs_values["z"].shape)}


@pytest.mark.parametrize("composite", ["cycle", "mixture"])
def test_gibbs_with_nuts_targets_the_mixed_posterior(composite: str) -> None:
    def build(update: GibbsUpdate | None) -> MCMCKernel:
        gibbs = GibbsKernel(["z"], update)
        nuts = NUTSKernel(max_tree_depth=4)
        if composite == "cycle":
            return CycleKernel({"gibbs": gibbs, "nuts": nuts})
        return MixtureKernel({"gibbs": gibbs, "nuts": nuts}, probs=[0.5, 0.5])

    def run(kernel: MCMCKernel) -> tuple[torch.Tensor, torch.Tensor]:
        result = MCMC(
            kernel, num_warmup=100, num_samples=800, num_chains=2, seed=2
        ).run(_switch_program(), torch.zeros(1, 1), {})
        assert result.samples["z"].shape == (2, 800)
        assert set(result.component_acceptance) == {"gibbs", "nuts"}
        return result.samples["z"], result.samples["mu"]

    p_value, ks, critical = _switch_checks(*run(build(None)))
    assert p_value > _LEVEL, p_value
    assert ks < critical, (ks, critical)

    bad_p, bad_ks, bad_critical = _switch_checks(*run(build(_prior_switch)))
    assert bad_p < _LEVEL or bad_ks > bad_critical, (bad_p, bad_ks, bad_critical)
    assert bad_p < _LEVEL, bad_p


_PRIOR = [0.2, 0.5, 0.3]
_OBS = torch.tensor([0.3, 1.8, 4.6])


def _label_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    """``y_i ~ N(2 c_i, 1)`` for the observed ``y``."""
    c = env["c"].to(torch.float32)
    return Normal(2.0 * c, 1.0).log_prob(_OBS).sum().reshape(1)


def _label_program() -> MonadicProgram:
    """Three Categorical labels over ``{0, 1, 2}``, each with its own
    observation; the labels' posteriors are independent."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("c",),
                morphism=PlateDraw(
                    3,
                    make_fixed_categorical(
                        _PRIOR,
                        FinSet(name="K", cardinality=3),
                        parameterization="probs",
                    ),
                ),
            ),
            Score(name="lik", score=reading(_label_likelihood, ("c",))),
        ],
        return_vars=(),
    )


def test_gibbs_alone_enumerates_a_plate_of_categorical_labels() -> None:
    result = MCMC(
        GibbsKernel(["c"]), num_warmup=20, num_samples=600, num_chains=2, seed=3
    ).run(_label_program(), torch.zeros(1, 1), {})
    labels = result.samples["c"]
    assert labels.shape == (2, 600, 3)
    assert labels.dtype == torch.int64
    log_prior = torch.tensor(_PRIOR, dtype=torch.float64).log()
    for i in range(3):
        log_post = log_prior + Normal(
            2.0 * torch.arange(3, dtype=torch.float64), 1.0
        ).log_prob(_OBS[i].to(torch.float64))
        probs = torch.softmax(log_post, 0)
        column = labels[:, :, i]
        counts = torch.stack([(column == k).sum() for k in range(3)]).to(torch.float64)
        p_value = _chi_square_p(counts, probs, _effective_size(column), column.numel())
        assert p_value > _LEVEL / 3, (i, p_value)


_DATA = torch.linspace(-0.5, 2.0, 12)


def _mean_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return Normal(env["mu"].reshape(()), 1.0).log_prob(_DATA).sum().reshape(1)


def _mean_program() -> MonadicProgram:
    """``mu ~ N(0, 1)``, ``y_i ~ N(mu, 1)``, with an independent nuisance
    ``u ~ N(0, 1)`` the Metropolis move samples."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Draw(
                names=("u",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=reading(_mean_likelihood, ("mu",))),
        ],
        return_vars=(),
    )


def _conjugate_mean(
    generator: torch.Generator | None,
    gibbs_values: dict[str, torch.Tensor],
    latent_values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The exact conditional ``mu | y ~ N(sum(y) / (n + 1), 1 / (n + 1))``."""
    del latent_values
    n = _DATA.numel()
    noise = torch.randn(gibbs_values["mu"].shape, generator=generator)
    return {"mu": float(_DATA.sum()) / (n + 1) + noise / math.sqrt(n + 1)}


def _prior_mean(
    generator: torch.Generator | None,
    gibbs_values: dict[str, torch.Tensor],
    latent_values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A wrong update: redraw ``mu`` from its prior."""
    del latent_values
    return {"mu": torch.randn(gibbs_values["mu"].shape, generator=generator)}


@pytest.mark.parametrize(
    ("update", "correct"), [(_conjugate_mean, True), (_prior_mean, False)]
)
def test_gibbs_update_function_samples_a_continuous_site(
    update: GibbsUpdate, correct: bool
) -> None:
    kernel = CycleKernel(
        [GibbsKernel(["mu"], update), MetropolisKernel(RandomWalkProposal(1.0))]
    )
    result = MCMC(kernel, num_warmup=20, num_samples=1500, num_chains=2, seed=4).run(
        _mean_program(), torch.zeros(1, 1), {}
    )
    mu = result.samples["mu"]
    n = _DATA.numel()
    values, _ = mu.reshape(-1).sort()
    ks, critical = _ks_exceeds(
        mu, _normal_cdf(values, float(_DATA.sum()) / (n + 1), 1.0 / math.sqrt(n + 1))
    )
    assert (ks < critical) == correct, (ks, critical)
    # The nuisance site is sampled by the random walk, from its prior.
    u_values, _ = result.samples["u"].reshape(-1).sort()
    u_ks, u_critical = _ks_exceeds(result.samples["u"], _normal_cdf(u_values, 0.0, 1.0))
    assert u_ks < u_critical


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_every_component_leaves_the_gradient_of_its_state() -> None:
    """After each composite transition the cached log-density and
    gradient are those of the potential conditioned on the state's Gibbs
    values, bit for bit."""
    model = _switch_program()
    recorded = trace(model, torch.zeros(1, 1), observations={})
    registry = LatentRegistry.from_trace(model, recorded, exclude={"z"})
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    kernel = CycleKernel(
        [
            MetropolisKernel(RandomWalkProposal(0.7)),
            GibbsKernel(["z"]),
            NUTSKernel(max_tree_depth=3),
        ]
    )
    kernel._set_generator(torch.Generator().manual_seed(5))
    state = kernel.init(registry, model, torch.zeros(1, 1), {}, torch.zeros(1))
    state = state.with_(gibbs_values={"z": recorded.sites["z"].value})
    for _ in range(30):
        state = kernel.step(state, potential.condition(state.gibbs_values))
        value, grad = potential.condition(state.gibbs_values).value_and_grad(
            state.position
        )
        assert torch.equal(state.log_density, value)
        assert torch.equal(state.grad_log_density, grad)
    assert set(state.stats) >= {"0.accepted", "1.accepted", "2.tree_depth", "accepted"}


class _RecordingKernel(MetropolisKernel):
    """A random-walk kernel that records its warmup length and steps."""

    def __init__(self) -> None:
        super().__init__(RandomWalkProposal(0.5))
        self.announced: list[int] = []
        self.adapting_steps = 0

    def start_adaptation(self, num_warmup: int) -> None:
        super().start_adaptation(num_warmup)
        self.announced.append(num_warmup)
        self.adapting_steps = 0

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        if self.is_adapting:
            self.adapting_steps += 1
        return super().step(state, potential)


def test_composites_forward_adaptation_and_generator() -> None:
    first, second = _RecordingKernel(), _RecordingKernel()
    mixture = MixtureKernel([first, second], probs=[0.3, 0.7])
    MCMC(mixture, num_warmup=50, num_samples=5, num_chains=1, seed=6).run(
        _normal_program(), torch.zeros(1, 1), {}
    )
    # Each component is told exactly how many warmup steps select it.
    assert first.announced[-1] == first.adapting_steps
    assert second.announced[-1] == second.adapting_steps
    assert first.adapting_steps + second.adapting_steps == 50
    assert not first.is_adapting and not second.is_adapting

    cycle_a, cycle_b = _RecordingKernel(), _RecordingKernel()
    MCMC(
        CycleKernel([cycle_a, cycle_b]),
        num_warmup=7,
        num_samples=2,
        num_chains=1,
        seed=0,
    ).run(_normal_program(), torch.zeros(1, 1), {})
    assert cycle_a.announced == [7] and cycle_b.announced == [7]

    def run() -> torch.Tensor:
        kernel = MixtureKernel(
            [NUTSKernel(max_tree_depth=3), MetropolisKernel(RandomWalkProposal(1.0))]
        )
        return (
            MCMC(kernel, num_warmup=10, num_samples=10, num_chains=2, seed=8)
            .run(_switch_program(), torch.zeros(1, 1), {})
            .samples["mu"]
        )

    with pytest.raises(ValueError, match="discrete"):
        run()

    def run_gibbs() -> dict[str, torch.Tensor]:
        kernel = MixtureKernel(
            [GibbsKernel(["z"]), MetropolisKernel(RandomWalkProposal(1.0))]
        )
        return (
            MCMC(kernel, num_warmup=10, num_samples=10, num_chains=2, seed=8)
            .run(_switch_program(), torch.zeros(1, 1), {})
            .samples
        )

    torch.manual_seed(321)
    before = torch.random.get_rng_state()
    a = run_gibbs()
    assert torch.equal(torch.random.get_rng_state(), before)
    b = run_gibbs()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_composites_refuse_malformed_components() -> None:
    nuts = NUTSKernel()
    with pytest.raises(ValueError, match="appears twice"):
        CycleKernel([nuts, nuts])
    with pytest.raises(ValueError, match="no dot"):
        CycleKernel({"a.b": NUTSKernel()})
    with pytest.raises(ValueError, match="at least one"):
        CycleKernel([])
    with pytest.raises(ValueError, match="sum to one"):
        MixtureKernel([NUTSKernel(), NUTSKernel()], probs=[0.5, 0.6])
    with pytest.raises(ValueError, match="2 kernels"):
        MixtureKernel([NUTSKernel(), NUTSKernel()], probs=[1.0])
    with pytest.raises(ValueError, match="name a site twice"):
        GibbsKernel(["z", "z"])


def test_gibbs_sites_are_checked_against_the_model() -> None:
    driver = MCMC(
        CycleKernel([GibbsKernel(["w"]), NUTSKernel()]),
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
    )
    with pytest.raises(ValueError, match=r"Gibbs sites \['w'\] are not latent"):
        driver.run(_switch_program(), torch.zeros(1, 1), {})
    # A continuous site cannot be enumerated without an update.
    driver = MCMC(
        CycleKernel([GibbsKernel(["z", "mu"])]),
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="pass update="):
        driver.run(_switch_program(), torch.zeros(1, 1), {})


def test_conditioning_replaces_and_refuses_registry_sites() -> None:
    model = _switch_program()
    recorded = trace(model, torch.zeros(1, 1), observations={})
    registry = LatentRegistry.from_trace(model, recorded, exclude={"z"})
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    zero = potential.condition({"z": torch.tensor([0.0])})
    one = zero.condition({"z": torch.tensor([1.0])})
    mu = torch.tensor([1.0])
    # mu = 1: z = 1 moves the conditional prior mean of mu from 0 to 3.
    expected = (
        math.log(0.3)
        + Normal(3.0, 1.0).log_prob(torch.tensor(1.0))
        + Normal(1.0, 1.0).log_prob(torch.tensor(_Y))
    )
    assert torch.allclose(one.log_density(mu), expected)
    assert one.conditioned_values.keys() == {"z"}
    with pytest.raises(ValueError, match="cannot also be conditioned on"):
        potential.condition({"mu": torch.tensor([[0.0]])})
