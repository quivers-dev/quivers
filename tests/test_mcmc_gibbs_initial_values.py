"""Chains start at caller-given values of the sites a Gibbs block updates."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quivers.continuous import Draw, Euclidean, MonadicProgram, PlateDraw, Score
from quivers.continuous.inline import (
    make_fixed_bernoulli,
    make_fixed_categorical,
    make_fixed_normal,
)
from quivers.core import FinSet, Unit
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    KernelState,
    MCMCCallback,
    MCMCCheckpoint,
    MCMCPhase,
    NUTSKernel,
)


def _coupled_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return -0.5 * (env["mu"].reshape(-1) - env["u"].reshape(-1)) ** 2


def _coupled_program() -> MonadicProgram:
    """``mu, u ~ N(0, 1)`` coupled by a quadratic score."""
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
            Score(name="lik", score=_coupled_score),
        ],
        return_vars=("mu", "u"),
    )


def _discrete_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    shift = env["z"].reshape(()) + env["c"].to(torch.float32).sum()
    return (-0.5 * (env["mu"].reshape(()) - shift) ** 2).reshape(1)


def _discrete_program() -> MonadicProgram:
    """A Bernoulli ``z`` and a plate of three labels ``c`` that shift ``mu``."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_bernoulli(0.3, FinSet(name="B", cardinality=2)),
            ),
            Draw(
                names=("c",),
                morphism=PlateDraw(
                    3,
                    make_fixed_categorical(
                        [0.2, 0.3, 0.5],
                        FinSet(name="C", cardinality=3),
                        parameterization="probs",
                    ),
                ),
            ),
            Draw(
                names=("mu",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=_discrete_score),
        ],
        return_vars=(),
    )


def _keep(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A Gibbs update that leaves the block where it is."""
    del generator, latent
    return {name: value.clone() for name, value in values.items()}


def _cycle(*sites: str) -> CycleKernel:
    return CycleKernel(
        {"gibbs": GibbsKernel(list(sites), update=_keep), "nuts": NUTSKernel(3)}
    )


_X = torch.zeros(1, 1)


def test_a_chain_starts_at_the_given_value_of_a_gibbs_site() -> None:
    result = MCMC(
        _cycle("u"), 2, 2, num_chains=2, seed=0, initial_values={"u": 0.5}
    ).run(_coupled_program(), _X, {})
    # The update keeps the block, so every draw is the starting value.
    assert torch.equal(result.samples["u"], torch.full((2, 2), 0.5))
    assert result.samples["mu"].shape == (2, 2)


def test_each_chain_starts_at_its_own_given_values() -> None:
    result = MCMC(
        _cycle("u"),
        0,
        3,
        num_chains=2,
        seed=0,
        initial_values=[{"u": -1.25, "mu": 0.5}, {"u": torch.tensor(2.0)}],
    ).run(_coupled_program(), _X, {})
    assert torch.equal(result.samples["u"][0], torch.full((3,), -1.25))
    assert torch.equal(result.samples["u"][1], torch.full((3,), 2.0))


def test_the_update_receives_the_given_value_in_the_trace_layout() -> None:
    seen: list[dict[str, torch.Tensor]] = []

    def recording(
        generator: torch.Generator | None,
        values: dict[str, torch.Tensor],
        latent: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        seen.append({name: value.clone() for name, value in values.items()})
        return _keep(generator, values, latent)

    kernel = CycleKernel(
        {"gibbs": GibbsKernel(["z", "c"], update=recording), "nuts": NUTSKernel(3)}
    )
    reference = MCMC(kernel, 0, 1, num_chains=1, seed=3).run(
        _discrete_program(), _X, {}
    )
    drawn = dict(seen[0])
    seen.clear()
    result = MCMC(kernel, 0, 1, num_chains=1, seed=3, initial_values={"c": 2}).run(
        _discrete_program(), _X, {}
    )
    first = seen[0]
    # The given site holds the given value in the layout and dtype a
    # program draw has, and the other site is still a program draw.
    assert first["c"].shape == drawn["c"].shape
    assert first["c"].dtype == drawn["c"].dtype
    assert torch.equal(first["c"], torch.full_like(drawn["c"], 2))
    assert first["z"].shape == drawn["z"].shape
    assert bool(((first["z"] == 0) | (first["z"] == 1)).all())
    assert torch.equal(result.samples["c"], torch.full_like(reference.samples["c"], 2))


def test_every_given_gibbs_site_leaves_the_chain_stream_to_the_position() -> None:
    """With every Gibbs site given, no program run draws from the chain's
    stream, so the position is the first draw of that stream."""
    driver = MCMC(_cycle("u"), 0, 1, num_chains=1, seed=11, initial_values={"u": 0.5})
    (start,) = driver.chain_start_values(_coupled_program(), _X, {})
    expected = 0.1 * torch.randn(1, generator=MCMC._chain_generator(11, 0))
    assert torch.equal(start["mu"].reshape(1), expected)


def test_a_boundary_value_of_a_discrete_support_is_accepted() -> None:
    kernel = CycleKernel({"gibbs": GibbsKernel(["z", "c"]), "nuts": NUTSKernel(3)})
    seen: list[dict[str, torch.Tensor]] = []

    def record(
        chain: int, iteration: int, state: KernelState, phase: MCMCPhase
    ) -> None:
        del chain, iteration, phase
        seen.append(state.gibbs_values)

    labels = torch.tensor([0, 2, 1])
    result = MCMC(
        kernel,
        0,
        2,
        num_chains=1,
        seed=0,
        initial_values={"z": 0.0, "c": labels},
        callback=record,
    ).run(_discrete_program(), _X, {})
    assert result.samples["c"].shape == (1, 2, 3)
    assert result.samples["z"].shape[:2] == (1, 2)
    assert seen[0]["c"].shape == (3,)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"c": 3}, "outside its finite support"),
        ({"c": 0.5}, "outside its finite support"),
        ({"z": 2.0}, "outside its finite support"),
        ({"z": float("nan")}, "outside its finite support"),
        ({"z": True}, "real and non-Boolean"),
        ({"z": torch.zeros(2)}, "expected scalar or"),
        ({"c": torch.zeros(2)}, r"expected scalar or \(3,\)"),
        ({"w": 1.0}, "unknown latent sites"),
    ],
)
def test_a_given_gibbs_value_is_validated_before_sampling(
    values: dict[str, torch.Tensor | float], message: str
) -> None:
    kernel = CycleKernel({"gibbs": GibbsKernel(["z", "c"]), "nuts": NUTSKernel(3)})
    driver = MCMC(kernel, 0, 2, num_chains=1, seed=0, initial_values=values)
    with pytest.raises(ValueError, match=message):
        driver.run(_discrete_program(), _X, {})


def test_given_gibbs_values_take_part_in_checkpoint_compatibility(
    tmp_path: Path,
) -> None:
    def driver(
        value: float | None,
        checkpoint: MCMCCheckpoint | None = None,
        callback: MCMCCallback | None = None,
    ) -> MCMC:
        return MCMC(
            _cycle("u"),
            3,
            4,
            num_chains=2,
            seed=5,
            initial_values=None if value is None else {"u": value},
            checkpoint=checkpoint,
            callback=callback,
        )

    model = _coupled_program()
    expected = driver(0.5).run(model, _X, {})

    class Interrupted(Exception):
        pass

    def interrupt(
        chain: int, iteration: int, state: KernelState, phase: MCMCPhase
    ) -> None:
        del state
        if chain == 1 and phase == "sampling" and iteration == 1:
            raise Interrupted

    interrupted = driver(
        0.5, MCMCCheckpoint(directory=str(tmp_path), every=1), interrupt
    )
    with pytest.raises(Interrupted):
        interrupted.run(model, _X, {})
    for changed in (0.75, None):
        with pytest.raises(ValueError, match="different run settings"):
            driver(changed).resume(tmp_path, model, _X, {})
    resumed = driver(0.5).resume(tmp_path, model, _X, {})
    for name in expected.samples:
        assert torch.equal(expected.samples[name], resumed.samples[name])
    assert torch.equal(expected.log_densities, resumed.log_densities)
    for name in expected.sample_stats:
        assert torch.equal(expected.sample_stats[name], resumed.sample_stats[name])
    assert torch.equal(resumed.samples["u"], torch.full((2, 4), 0.5))
