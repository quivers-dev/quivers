"""A chain's density is evaluated when a transition or a draw reads it.

A Gibbs sweep that changes a value leaves the log-density and gradient
of the state unevaluated, so a run of Gibbs blocks costs one evaluation
however long it is. The draws, statistics, random streams, and
checkpoints are those of a Gibbs block that evaluates after every
sweep, which `_EagerGibbs` is.
"""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx
import pytest
import torch
from torch.distributions import Normal

from quivers.continuous import Draw, Euclidean, MonadicProgram, Score
from quivers.continuous.inline import make_fixed_bernoulli, make_fixed_normal
from quivers.core import FinSet, Unit
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    GibbsUpdate,
    KernelState,
    MCMCCallback,
    MCMCCheckpoint,
    MCMCKernel,
    MCMCPhase,
    MCMCResult,
    MetropolisKernel,
    NUTSKernel,
    PotentialFn,
    RandomWalkProposal,
)
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace

_X = torch.zeros(1, 1)


class _EagerGibbs(GibbsKernel):
    """A Gibbs block that evaluates the density after every sweep."""

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        out = super().step(state, potential)
        log_density, grad = potential.condition(out.gibbs_values).value_and_grad(
            out.position
        )
        return out.with_(log_density=log_density, grad_log_density=grad)


def _mixed_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    z, mu, u = (env[name].reshape(()) for name in ("z", "mu", "u"))
    shift = Normal(3.0 * z, 1.0).log_prob(mu) - Normal(0.0, 1.0).log_prob(mu)
    return (shift - 0.5 * (mu - u) ** 2).reshape(1)


def _mixed_program() -> MonadicProgram:
    """A Bernoulli ``z``, and ``mu`` and ``u`` coupled to it and each other."""
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
            Draw(
                names=("u",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="S", dim=1)),
            ),
            Score(name="lik", score=_mixed_score),
        ],
        return_vars=(),
    )


def _jump(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A Metropolis move on ``u`` from its prior, written as a Gibbs update."""
    u, mu = values["u"], latent["mu"].reshape(())
    candidate = torch.randn(u.shape, generator=generator, dtype=u.dtype)
    log_ratio = (
        -0.5 * (mu - candidate.reshape(())) ** 2 + 0.5 * (mu - u.reshape(())) ** 2
    )
    accept = torch.rand((), generator=generator) < torch.exp(log_ratio)
    return {"u": candidate if bool(accept) else u.clone()}


def _keep(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    del generator, latent
    return {name: value.clone() for name, value in values.items()}


def _shift(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """An update that always changes the block, for counting evaluations."""
    del generator, latent
    return {name: value + 0.01 for name, value in values.items()}


def _conjugate(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The exact conditional of ``u``: ``N(mu / 2, 1 / 2)``."""
    u = values["u"]
    noise = torch.randn(u.shape, generator=generator, dtype=u.dtype)
    return {"u": latent["mu"].reshape(u.shape).to(u.dtype) / 2.0 + noise * 0.5**0.5}


def _mixed_cycle(gibbs: type[GibbsKernel], order: str) -> CycleKernel:
    """Enumeration, a move that may keep the block, an exact draw that
    always changes it, a random walk, and NUTS, in one of three orders."""
    z: MCMCKernel = gibbs(["z"])
    jump: MCMCKernel = gibbs(["u"], update=_jump)
    exact: MCMCKernel = gibbs(["u"], update=_conjugate)
    walk: MCMCKernel = MetropolisKernel(RandomWalkProposal(0.7))
    nuts: MCMCKernel = NUTSKernel(max_tree_depth=4)
    orders = {
        "gibbs_first": ("z", "jump", "exact", "walk", "nuts"),
        "gibbs_last": ("walk", "nuts", "z", "jump", "exact"),
        "interleaved": ("z", "walk", "jump", "nuts", "exact"),
    }
    named = {"z": z, "jump": jump, "exact": exact, "walk": walk, "nuts": nuts}
    return CycleKernel({name: named[name] for name in orders[order]})


def _assert_same_run(first: MCMCResult, second: MCMCResult) -> None:
    """Every recorded quantity of two runs is equal bit for bit."""
    assert first.samples.keys() == second.samples.keys()
    for name in first.samples:
        assert torch.equal(first.samples[name], second.samples[name]), name
    assert torch.equal(first.log_densities, second.log_densities)
    assert first.sample_stats.keys() == second.sample_stats.keys()
    for name in first.sample_stats:
        assert torch.equal(first.sample_stats[name], second.sample_stats[name]), name
    assert torch.equal(first.acceptance_rates, second.acceptance_rates)
    assert torch.equal(first.divergence_counts, second.divergence_counts)
    assert first.component_acceptance.keys() == second.component_acceptance.keys()
    for name in first.component_acceptance:
        assert torch.equal(
            first.component_acceptance[name], second.component_acceptance[name]
        )


@pytest.mark.parametrize("order", ["gibbs_first", "gibbs_last", "interleaved"])
@pytest.mark.parametrize("thin", [1, 3])
def test_draws_and_statistics_are_those_of_evaluating_after_every_sweep(
    order: str, thin: int
) -> None:
    def run(gibbs: type[GibbsKernel]) -> MCMCResult:
        return MCMC(
            _mixed_cycle(gibbs, order), 15, 20, num_chains=2, seed=7, thin=thin
        ).run(_mixed_program(), _X, {})

    lazy, eager = run(GibbsKernel), run(_EagerGibbs)
    _assert_same_run(lazy, eager)
    assert bool(torch.isfinite(lazy.log_densities).all())
    # Three Gibbs blocks a cycle: evaluating after each costs more.
    assert bool((lazy.num_gradient_evaluations < eager.num_gradient_evaluations).all())
    assert bool((lazy.num_evaluations < eager.num_evaluations).all())


@pytest.mark.parametrize("order", ["gibbs_first", "gibbs_last"])
def test_an_interrupted_run_resumes_bitwise_with_its_evaluation_counts(
    order: str, tmp_path: Path
) -> None:
    def driver(
        checkpoint: MCMCCheckpoint | None = None, callback: MCMCCallback | None = None
    ) -> MCMC:
        return MCMC(
            _mixed_cycle(GibbsKernel, order),
            6,
            8,
            num_chains=2,
            seed=3,
            thin=2,
            checkpoint=checkpoint,
            callback=callback,
        )

    model = _mixed_program()
    expected = driver().run(model, _X, {})

    class Interrupted(Exception):
        pass

    def interrupt_in(stop_phase: MCMCPhase, stop_iteration: int) -> MCMCCallback:
        def interrupt(
            chain: int, iteration: int, state: KernelState, phase: MCMCPhase
        ) -> None:
            del state
            if chain == 1 and phase == stop_phase and iteration == stop_iteration:
                raise Interrupted

        return interrupt

    # Once in warmup and once after a thinned-out sampling transition:
    # with the Gibbs blocks last, the exact draw ends the cycle, so both
    # checkpoints store a state whose density is not evaluated.
    stops: tuple[tuple[MCMCPhase, int], ...] = (("warmup", 3), ("sampling", 5))
    for index, (phase, iteration) in enumerate(stops):
        directory = tmp_path / str(index)
        with pytest.raises(Interrupted):
            driver(
                MCMCCheckpoint(directory=str(directory), every=1),
                interrupt_in(phase, iteration),
            ).run(model, _X, {})
        stored = torch.load(directory / "mcmc_checkpoint.pt", weights_only=True)
        stored_density = stored["chains"][1]["state"]["log_density"]
        if order == "gibbs_last":
            assert stored_density is None
        resumed = driver().resume(directory, model, _X, {})
        _assert_same_run(expected, resumed)
        assert torch.equal(expected.num_evaluations, resumed.num_evaluations)
        assert torch.equal(
            expected.num_gradient_evaluations, resumed.num_gradient_evaluations
        )


def _coupled_program(calls: dict[str, int]) -> MonadicProgram:
    """``mu, u ~ N(0, 1)`` coupled by a score that counts its calls."""

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        calls["score"] += 1
        return -0.5 * (env["mu"].reshape(-1) - env["u"].reshape(-1)) ** 2

    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="mu_s", dim=1)),
            ),
            Draw(
                names=("u",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="u_s", dim=1)),
            ),
            Score(name="lik", score=score),
        ],
        return_vars=("mu", "u"),
    )


def _gibbs_then_nuts(count: int, update: GibbsUpdate) -> CycleKernel:
    components: dict[str, MCMCKernel] = {
        f"g{index}": GibbsKernel(["u"], update=update) for index in range(count)
    }
    components["nuts"] = NUTSKernel(max_tree_depth=1)
    return CycleKernel(components)


def test_sweeps_that_change_nothing_cost_no_evaluation() -> None:
    """The cost of a cycle of ``m`` unchanged Gibbs blocks and NUTS does
    not depend on ``m``: the score runs for the chain's first state and
    for NUTS's own leapfrog steps only."""
    draws = 50
    per_count: dict[int, tuple[int, MCMCResult]] = {}
    score_calls: dict[int, int] = {}
    for count in (1, 2, 5, 9):
        calls = {"score": 0}
        model = _coupled_program(calls)
        driver = MCMC(_gibbs_then_nuts(count, _keep), 0, draws, num_chains=1, seed=0)
        result = driver.run(model, _X, {})
        per_count[count] = (int(result.num_evaluations[0]), result)
        score_calls[count] = calls["score"]
        leapfrogs = int(result.sample_stats["nuts.num_steps"].sum())
        assert int(result.num_gradient_evaluations[0]) == leapfrogs + 1
        assert int(result.num_evaluations[0]) == leapfrogs + 1
    evaluations = {count: value for count, (value, _) in per_count.items()}
    assert len(set(evaluations.values())) == 1, evaluations
    # The score itself runs equally often whatever the number of blocks.
    assert len(set(score_calls.values())) == 1, score_calls
    assert score_calls[9] >= evaluations[9]
    # At most two leapfrog steps a transition at tree depth one.
    assert evaluations[9] <= 2 * draws + 1
    # The blocks draw nothing, so every cycle makes the same NUTS moves.
    for count in (2, 5, 9):
        assert torch.equal(
            per_count[1][1].samples["mu"], per_count[count][1].samples["mu"]
        )


@pytest.mark.parametrize("count", [1, 5, 9])
def test_a_run_of_changing_sweeps_costs_one_evaluation(count: int) -> None:
    """``m`` Gibbs blocks that each change the state, then NUTS: one
    gradient evaluation for the run of blocks and NUTS's leapfrog steps."""
    draws = 40
    result = MCMC(_gibbs_then_nuts(count, _shift), 0, draws, num_chains=1, seed=0).run(
        _coupled_program({"score": 0}), _X, {}
    )
    leapfrogs = int(result.sample_stats["nuts.num_steps"].sum())
    assert int(result.num_gradient_evaluations[0]) == leapfrogs + draws
    assert int(result.num_evaluations[0]) == leapfrogs + draws


def test_a_changed_sweep_leaves_the_density_for_its_reader() -> None:
    model = _mixed_program()
    recorded = trace(model, _X, observations={})
    registry = LatentRegistry.from_trace(model, recorded, exclude={"z", "u"})
    potential = PotentialFn(model, registry, _X, {})
    values = {name: recorded.sites[name].value for name in ("z", "u")}
    nuts = NUTSKernel(max_tree_depth=3)
    nuts._set_generator(torch.Generator().manual_seed(2))
    state = nuts.init(registry, model, _X, {}, torch.zeros(1)).with_(
        gibbs_values=values
    )
    state = nuts.step(state, potential.condition(values))
    assert state.has_density

    def step(update: GibbsUpdate, start: KernelState) -> KernelState:
        block = GibbsKernel(["u"], update=update)
        block._set_generator(torch.Generator().manual_seed(4))
        block.init(registry, model, _X, {}, start.position)
        return block.step(start, potential.condition(start.gibbs_values))

    kept = step(_keep, state)
    assert kept.has_density
    assert kept.log_density is state.log_density
    assert kept.grad_log_density is state.grad_log_density

    before = potential.num_evaluations
    moved = step(_shift, state)
    assert potential.num_evaluations == before
    assert not moved.has_density and moved.log_density is None
    with pytest.raises(RuntimeError, match="not evaluated"):
        moved.density()
    restored = KernelState.from_state_dict(moved.state_dict())
    assert restored.log_density is None and restored.grad_log_density is None
    assert torch.equal(restored.gibbs_values["u"], moved.gibbs_values["u"])

    conditioned = potential.condition(moved.gibbs_values)
    evaluated = moved.evaluated(conditioned)
    assert potential.num_gradient_evaluations > 0
    assert potential.num_evaluations == before + 1
    value, grad = conditioned.value_and_grad(moved.position)
    log_density, gradient = evaluated.density()
    assert torch.equal(log_density, value) and torch.equal(gradient, grad)
    assert evaluated.evaluated(conditioned) is evaluated
    with pytest.raises(dx.ValidationError, match="both are tensors or both are None"):
        moved.with_(log_density=value)


def test_a_sweep_onto_zero_density_raises_where_the_density_is_read() -> None:
    def barred(env: dict[str, torch.Tensor]) -> torch.Tensor:
        u = env["u"].reshape(-1)
        inside = -0.5 * (env["mu"].reshape(-1) - u) ** 2
        return torch.where(u > 5.0, torch.full_like(inside, -torch.inf), inside)

    def leave(
        generator: torch.Generator | None,
        values: dict[str, torch.Tensor],
        latent: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del generator, latent
        return {"u": torch.full_like(values["u"], 10.0)}

    model = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="mu_s", dim=1)),
            ),
            Draw(
                names=("u",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="u_s", dim=1)),
            ),
            Score(name="lik", score=barred),
        ],
        return_vars=("mu", "u"),
    )
    kernel = CycleKernel(
        {"u": GibbsKernel(["u"], update=leave), "nuts": NUTSKernel(max_tree_depth=2)}
    )
    with pytest.raises(RuntimeError, match=r"not finite at.*\['u'\]"):
        MCMC(kernel, 0, 2, num_chains=1, seed=0, initial_values={"u": 0.0}).run(
            model, _X, {}
        )
