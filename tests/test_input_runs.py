"""An input of ``B`` rows is ``B`` independent runs of the whole program.

`MonadicProgram.log_joint` returns one joint per run, the joint of all
the runs is its sum, and every run reads its own slice of the values
given one per run and the whole of the values the runs share. A let or
score reads one run's values, so a factor, a collection, or an index
never reads the run axis. The posterior drivers that target one run's
latents (MCMC, the Laplace approximation, annealed importance sampling,
and the Bayesian lift of a plated program) take a one-row input; the
variational objectives sum their per-run terms, so a multi-row input
neither tempers nor sharpens the posterior. The latent registry is read
off a run, so a scanned program cell's per-position latents are
inferred.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Categorical, Dirichlet, HalfNormal, Normal

from quivers.dsl import loads
from quivers.inference import (
    ELBO,
    MCMC,
    SVI,
    AutoDAIS,
    AutoLaplaceApproximation,
    AutoNormalGuide,
    NUTSKernel,
    bayesian_lift_parameters,
    trace,
)
from quivers.inference.mcmc import PotentialFn
from quivers.inference.registry import LatentRegistry

ROWS = 8
Y = torch.linspace(-1.0, 2.0, ROWS)

CONJUGATE = f"""
object Obs : FinSet {ROWS}
program p : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Obs <- Normal(loc=mu, scale=1.0)
    return mu
export p
"""

LOCATION_SCALE = f"""
object Obs : FinSet {ROWS}
program p : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    observe y : Obs <- Normal(loc=mu, scale=sigma)
    return mu
export p
"""

FACTOR = """
object Step : FinSet 2
object Val : Real 1

program p : Val -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let k = factor t : Step in a * (real(t) + 1.0)
    score s = sum(k)
    return a

export p
"""

SHIFTED = """
object Row : FinSet 3
object Val : Real 1

program shifted : Row -> Val
    sample mu <- Normal(loc=0.0, scale=1.0)
    sample z : Row <- Normal(loc=0.0, scale=1.0)
    let m = mu + z
    observe y : Row <- Normal(loc=m, scale=se)
    return y

export shifted
"""

MIXTURE = """
object Resp : FinSet 5
object K : FinSet 2
object Val : Real 1
program p : Resp -> Val
    sample w <- Dirichlet(concentration=1.0) [over=K]
    sample m <- Normal(loc=0.0, scale=2.0) [over=K]
    marginalize c : Resp <- Categorical(probs=w) [reduction=logsumexp]
        let loc = m[c]
        observe y : Resp <- Normal(loc=loc, scale=1.0)
    return w
export p
"""

SCAN = """
object Driver : Real 2
object State : Real 3

morphism transition : Driver * State -> State ~ Normal

program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    return s

define run_all = scan(step)

program ssm : Driver -> State
    sample h <- run_all
    return h

export ssm
"""

CLASSES = (
    "object Class : FinSet 3\nobject Row : FinSet 4\nobject Val : Real 1\n"
    "program p : Row -> Val\n"
    "    sample probs <- Dirichlet(concentration=1.0) [over=Class]\n"
)
GATHERED = (
    CLASSES + "    sample mu <- Normal(loc=0.0, scale=2.0) [over=Class]\n"
    "    sample z : Row <- Categorical(probs=probs)\n"
    "    observe x : Row <- Normal(loc=mu[z], scale=1.0)\n    return z\nexport p\n"
)
DRAWN_CLASSES = (
    CLASSES + "    sample z : Row <- Categorical(probs=probs)\n    return z\nexport p\n"
)


def _model(source: str):
    return loads(source).morphism


def _runs(values: dict[str, torch.Tensor], runs: int) -> list[dict[str, torch.Tensor]]:
    """Each run's slice of values laid out one per run."""
    return [
        {name: value[b : b + 1] for name, value in values.items()} for b in range(runs)
    ]


# ---------------------------------------------------------------------------
# The joint is one number per run
# ---------------------------------------------------------------------------


def test_log_joint_is_one_hand_written_joint_per_run() -> None:
    model = _model(LOCATION_SCALE)
    mu, sigma = torch.tensor(0.3), torch.tensor(0.8)
    hand = (
        Normal(0.0, 1.0).log_prob(mu)
        + HalfNormal(1.0).log_prob(sigma)
        + Normal(mu, sigma).log_prob(Y).sum()
    )
    for runs in (1, ROWS):
        joint = model.log_joint(
            torch.zeros(runs, 1),
            {"mu": mu.reshape(1), "sigma": sigma.reshape(1), "y": Y},
        )
        assert joint.shape == (runs,)
        torch.testing.assert_close(joint, hand.expand(runs).to(joint.dtype))


def test_potential_is_the_hand_written_density_of_one_run() -> None:
    model = _model(LOCATION_SCALE)
    registry = LatentRegistry.from_model(model, {"y"})
    z = torch.tensor([0.3, -0.2])
    mu, sigma = z[0], z[1].exp()
    hand = (
        Normal(0.0, 1.0).log_prob(mu)
        + HalfNormal(1.0).log_prob(sigma)
        + Normal(mu, sigma).log_prob(Y).sum()
        + z[1]
    )
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {"y": Y})
    torch.testing.assert_close(potential.log_density(z), hand)


@pytest.mark.parametrize(
    ("source", "values", "shared"),
    [
        (CONJUGATE, {"mu": torch.tensor([0.2, -0.4, 1.1]), "y": Y}, ("y",)),
        (FACTOR, {"a": torch.tensor([0.3, -0.7, 1.5])}, ()),
        (
            SHIFTED,
            {
                "mu": torch.tensor([0.5, -0.1, 0.2]),
                "z": torch.randn(3, 3),
                "y": torch.randn(3, 3),
                "se": torch.tensor([0.5, 1.0, 1.5]),
            },
            ("se",),
        ),
        (
            MIXTURE,
            {
                "w": torch.tensor([[0.3, 0.7], [0.5, 0.5], [0.9, 0.1]]),
                "m": torch.tensor([[-0.5, 2.0], [0.0, 1.0], [1.0, -1.0]]),
                "y": torch.tensor([0.1, 2.3, -0.4, 1.9, 0.5]),
            },
            ("y",),
        ),
    ],
)
def test_the_joint_of_runs_is_the_sum_of_one_run_joints(
    source: str, values: dict[str, torch.Tensor], shared: tuple[str, ...]
) -> None:
    """Values given one per run (a leading axis of three) are each
    run's; the ones named in ``shared`` are every run's."""
    model = _model(source)
    runs = 3
    common = {name: values[name] for name in shared}
    per_run = {name: value for name, value in values.items() if name not in shared}
    together = model.log_joint(torch.zeros(runs, 1), values)
    assert together.shape == (runs,)
    alone = torch.cat(
        [
            model.log_joint(torch.zeros(1, 1), {**one, **common})
            for one in _runs(per_run, runs)
        ]
    )
    torch.testing.assert_close(together, alone)
    torch.testing.assert_close(together.sum(), alone.sum())


def test_a_scan_scores_each_run_on_its_own_sequence() -> None:
    torch.manual_seed(0)
    model = _model(SCAN)
    u = torch.randn(2, 5, 2)
    recorded = trace(model, u)
    values = {
        name: site.value
        for name, site in recorded.sites.items()
        if not site.is_deterministic
    }
    together = model.log_joint(u, values)
    alone = torch.cat(
        [
            model.log_joint(
                u[b : b + 1], {name: v[b : b + 1] for name, v in values.items()}
            )
            for b in range(2)
        ]
    )
    torch.testing.assert_close(together, alone)
    torch.testing.assert_close(recorded.log_joint, together)


def test_a_scanned_cell_scores_its_transition_density() -> None:
    """Moving one position's state moves the joint: the scan's latent
    trajectory is in the joint the drivers target."""
    torch.manual_seed(0)
    model = _model(SCAN)
    u = torch.randn(1, 5, 2)
    recorded = trace(model, u)
    values = {
        name: site.value
        for name, site in recorded.sites.items()
        if not site.is_deterministic
    }
    moved = {**values, "s@2": values["s@2"] + 100.0}
    assert float(model.log_joint(u, moved)) < float(model.log_joint(u, values)) - 100.0


# ---------------------------------------------------------------------------
# Lets and scores read one run's values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runs", [1, 3])
def test_a_factor_over_a_site_reduces_its_own_axis(runs: int) -> None:
    value = 0.3
    expected = -0.5 * value**2 - 0.5 * math.log(2 * math.pi) + value * (1.0 + 2.0)
    joint = _model(FACTOR).log_joint(
        torch.zeros(runs, 1), {"a": torch.full((runs,), value)}
    )
    torch.testing.assert_close(joint, torch.full((runs,), expected, dtype=joint.dtype))


def test_a_gathered_argument_indexes_within_its_run() -> None:
    probs = torch.tensor([0.2, 0.3, 0.5])
    mu = torch.tensor([-1.0, 0.5, 2.0])
    z = torch.tensor([0, 2, 1, 2])
    x = torch.tensor([0.1, 1.7, 1.2, 2.4])
    expected = (
        Dirichlet(torch.ones(3)).log_prob(probs)
        + Normal(0.0, 2.0).log_prob(mu).sum()
        + Categorical(probs=probs).log_prob(z).sum()
        + Normal(mu[z], 1.0).log_prob(x).sum()
    )
    model = _model(GATHERED)
    values = {"probs": probs, "mu": mu, "z": z, "x": x}
    for given in (values, {name: value[None] for name, value in values.items()}):
        joint = model.log_joint(torch.zeros(1, 1), given)
        torch.testing.assert_close(joint.sum(), expected.to(joint.dtype))


# ---------------------------------------------------------------------------
# Forward sampling keeps the run axis apart from plate rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runs", [1, 2, 4])
def test_a_plated_observation_draws_one_set_of_rows_per_run(runs: int) -> None:
    torch.manual_seed(1)
    model = _model(SHIFTED)
    se = torch.tensor([0.5, 1.0, 1.5])
    x = torch.zeros(runs, 1)
    draw = model.rsample(x, observations={"se": se})
    assert draw.numel() == runs * 3
    recorded = trace(model, x, observations={"se": se})
    assert recorded.sites["z"].value.numel() == runs * 3
    values = {
        name: site.value
        for name, site in recorded.sites.items()
        if not site.is_deterministic
    }
    joint = model.log_joint(x, {**values, "se": se})
    assert joint.shape == (runs,)
    assert torch.isfinite(joint).all()
    torch.testing.assert_close(joint, recorded.log_joint)


@pytest.mark.parametrize("runs", [2, 3])
def test_a_plated_draw_of_classes_from_a_per_run_site_samples_classes(
    runs: int,
) -> None:
    torch.manual_seed(2)
    model = _model(DRAWN_CLASSES)
    recorded = trace(model, torch.zeros(runs, 1))
    probs = recorded.sites["probs"].value
    z = recorded.sites["z"].value
    assert probs.shape == (runs, 3)
    assert torch.isfinite(probs).all()
    assert z.shape[0] == runs and z.numel() == runs * 4
    assert bool(((z >= 0) & (z < 3)).all())
    assert torch.isfinite(recorded.log_joint).all()


# ---------------------------------------------------------------------------
# One-run drivers, and posterior widths
# ---------------------------------------------------------------------------


def test_one_run_drivers_refuse_an_input_of_several_rows() -> None:
    model = _model(CONJUGATE)
    registry = LatentRegistry.from_model(model, {"y"})
    x = torch.zeros(ROWS, 1)
    with pytest.raises(ValueError, match="(?i)pass one input row"):
        PotentialFn(model, registry, x, {"y": Y})
    with pytest.raises(ValueError, match="(?i)pass one input row"):
        MCMC(NUTSKernel(), num_warmup=1, num_samples=1, num_chains=1, seed=0).run(
            model, x, {"y": Y}
        )
    base = AutoNormalGuide(model, observed_names={"y"})
    with pytest.raises(ValueError, match="(?i)pass one input row"):
        AutoDAIS(base, model, observations={"y": Y}, x=x, num_steps=1)
    laplace = AutoLaplaceApproximation(model, observed_names={"y"})
    with pytest.raises(ValueError, match="(?i)pass one input row"):
        laplace.fit_hessian(model, x, {"y": Y})
    with pytest.raises(ValueError, match="(?i)pass one input row"):
        bayesian_lift_parameters(model, x, {"y": Y}, additional_latents={"mu": (1,)})


def test_mcmc_recovers_the_conjugate_posterior_width() -> None:
    model = _model(CONJUGATE)
    exact_mean = float(Y.sum() / (ROWS + 1))
    exact_sd = (1.0 / (ROWS + 1)) ** 0.5
    result = MCMC(
        NUTSKernel(), num_warmup=300, num_samples=1000, num_chains=2, seed=0
    ).run(model, torch.zeros(1, 1), {"y": Y})
    draws = result.samples["mu"].reshape(-1)
    ess = float(result.ess["mu"])
    # Monte Carlo error of the mean is sd / sqrt(ESS); of the sd, about
    # sd / sqrt(2 ESS). Four standard errors bound both.
    assert abs(float(draws.mean()) - exact_mean) < 4 * exact_sd / math.sqrt(ess)
    assert abs(float(draws.std()) - exact_sd) < 4 * exact_sd / math.sqrt(2 * ess)


def test_svi_over_several_runs_targets_the_one_run_posterior() -> None:
    model = _model(CONJUGATE)
    exact_mean = float(Y.sum() / (ROWS + 1))
    exact_sd = (1.0 / (ROWS + 1)) ** 0.5
    torch.manual_seed(0)
    guide = AutoNormalGuide(model, observed_names={"y"})
    svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=0.02), ELBO())
    for _ in range(1500):
        svi.step(torch.zeros(4, 1), {"y": Y})
    sd = float(guide._scale("mu"))
    mean = float(guide._loc("mu"))
    assert abs(sd - exact_sd) < 0.06
    assert abs(mean - exact_mean) < 0.1


# ---------------------------------------------------------------------------
# A scanned cell's per-position latents are inferred
# ---------------------------------------------------------------------------


def test_the_registry_of_a_scan_holds_every_position() -> None:
    torch.manual_seed(0)
    model = _model(SCAN)
    u = torch.randn(1, 5, 2)
    registry = LatentRegistry.from_trace(model, trace(model, u))
    assert registry.names == tuple(f"s@{t}" for t in range(5))
    assert all(registry[name].constrained_shape == (3,) for name in registry.names)
    with pytest.raises(ValueError, match="from_trace"):
        LatentRegistry.from_model(model, set())


def test_svi_and_mcmc_infer_a_scanned_trajectory() -> None:
    torch.manual_seed(0)
    model = _model(SCAN)
    u = torch.randn(1, 5, 2)
    guide = AutoNormalGuide(model, observed_names=set(), x=u)
    assert guide.latent_names == [f"s@{t}" for t in range(5)]
    svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=1e-2), ELBO())
    losses = [svi.step(u, {}) for _ in range(5)]
    assert all(math.isfinite(loss) for loss in losses)
    result = MCMC(
        NUTSKernel(max_tree_depth=4),
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        seed=0,
    ).run(model, u, {})
    assert set(result.samples) == {f"s@{t}" for t in range(5)}
    assert all(draws.shape == (1, 10, 3) for draws in result.samples.values())
