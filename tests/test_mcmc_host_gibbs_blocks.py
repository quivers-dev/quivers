"""Gibbs blocks over host data the program reads but does not draw."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path

import pytest
import torch
from torch.distributions import Normal

from quivers.continuous import Draw, Euclidean, MonadicProgram, Score
from quivers.continuous.inline import make_fixed_normal
from quivers.core import Unit
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    GibbsUpdate,
    KernelState,
    MCMCCallback,
    MCMCCheckpoint,
    MCMCPhase,
    NUTSKernel,
)
from quivers.inference.smc import SMC

_X = torch.zeros(1, 1)
_PENALTY = 0.1


def _optional_host_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    total = -0.5 * (env["mu"].reshape(-1) - env["u"].reshape(-1)) ** 2
    return total - _PENALTY * env["z"].sum() if "z" in env else total


def _host_score(env: dict[str, torch.Tensor]) -> torch.Tensor:
    """A score that cannot run without the host data ``z``."""
    total = -0.5 * (env["mu"].reshape(-1) - env["u"].reshape(-1)) ** 2
    return total - _PENALTY * env["z"].sum()


def _program(
    score: Callable[[dict[str, torch.Tensor]], torch.Tensor] = _host_score,
) -> MonadicProgram:
    """``mu, u ~ N(0, 1)`` with a score that reads the host data ``z``."""
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


def _keep(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    del generator, latent
    return {name: value.clone() for name, value in values.items()}


def _redraw(
    generator: torch.Generator | None,
    values: dict[str, torch.Tensor],
    latent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The exact conditional of ``z``: independent Bernoulli memberships
    with odds ``exp(-_PENALTY)``."""
    del latent
    z = values["z"]
    probability = torch.sigmoid(torch.tensor(-_PENALTY))
    return {"z": (torch.rand(z.shape, generator=generator) < probability).to(z.dtype)}


def _cycle(update: GibbsUpdate = _redraw) -> CycleKernel:
    return CycleKernel(
        {
            "host": GibbsKernel(["z"], update=update, host=True),
            "nuts": NUTSKernel(max_tree_depth=3),
        }
    )


def test_a_host_block_runs_and_its_values_are_recorded_as_draws() -> None:
    start = torch.tensor([1, 0, 1], dtype=torch.int64)
    result = MCMC(
        _cycle(_keep), 2, 2, num_chains=1, seed=0, initial_values={"z": start}
    ).run(_program(_optional_host_score), _X, {})
    assert result.samples["z"].dtype == torch.int64
    assert torch.equal(result.samples["z"], start.expand(1, 2, 3))
    assert set(result.samples) == {"mu", "u", "z"}


def test_the_potential_is_conditioned_on_the_updated_host_data() -> None:
    result = MCMC(
        _cycle(),
        50,
        400,
        num_chains=2,
        seed=1,
        initial_values={"z": torch.zeros(5)},
    ).run(_program(), _X, {})
    z, mu, u = (result.samples[name].to(torch.float64) for name in ("z", "mu", "u"))
    assert z.shape == (2, 400, 5)
    # The recorded log-density is the program's joint at each draw's own
    # host data.
    standard = Normal(0.0, 1.0)
    expected = (
        standard.log_prob(mu)
        + standard.log_prob(u)
        - 0.5 * (mu - u) ** 2
        - _PENALTY * z.sum(-1)
    )
    assert torch.allclose(
        result.log_densities.to(torch.float64), expected, rtol=0.0, atol=2e-5
    )
    # Memberships follow their conditional, drawn independently per
    # transition, so their mean is within five standard errors of it.
    probability = 1.0 / (1.0 + math.exp(_PENALTY))
    error = math.sqrt(probability * (1.0 - probability) / z.numel())
    assert abs(float(z.mean()) - probability) < 5.0 * error
    assert z.unique().tolist() == [0.0, 1.0]


def test_each_chain_starts_at_its_own_host_data() -> None:
    result = MCMC(
        _cycle(_keep),
        0,
        2,
        num_chains=2,
        seed=0,
        initial_values=[{"z": torch.zeros(2)}, {"z": torch.ones(2)}],
    ).run(_program(), _X, {})
    assert torch.equal(result.samples["z"][0], torch.zeros(2, 2))
    assert torch.equal(result.samples["z"][1], torch.ones(2, 2))


def test_a_host_block_needs_a_starting_value_in_every_chain() -> None:
    with pytest.raises(ValueError, match=r"\['z'\] are host data.*starting value"):
        MCMC(_cycle(), 1, 1, num_chains=1, seed=0).run(_program(), _X, {})
    with pytest.raises(ValueError, match="give chain 1 a starting value"):
        MCMC(
            _cycle(),
            1,
            1,
            num_chains=2,
            seed=0,
            initial_values=[{"z": torch.zeros(2)}, {"mu": 0.5}],
        ).run(_program(), _X, {})
    with pytest.raises(ValueError, match="chain 1's starting value of the host"):
        MCMC(
            _cycle(),
            1,
            1,
            num_chains=2,
            seed=0,
            initial_values=[{"z": torch.zeros(2)}, {"z": torch.zeros(3)}],
        ).run(_program(), _X, {})


def test_a_host_declaration_is_checked_against_the_program_and_the_data() -> None:
    start = {"z": torch.zeros(2)}
    kernel = CycleKernel(
        {"u": GibbsKernel(["u"], update=_keep, host=True), "nuts": NUTSKernel(3)}
    )
    with pytest.raises(ValueError, match=r"\['u'\] are declared host.*binds them"):
        MCMC(kernel, 1, 1, num_chains=1, seed=0, initial_values={"u": 0.5}).run(
            _program(_optional_host_score), _X, {}
        )
    with pytest.raises(ValueError, match="not among the observations"):
        MCMC(_cycle(), 1, 1, num_chains=1, seed=0, initial_values=start).run(
            _program(), _X, {"z": torch.zeros(2)}
        )


def test_enumeration_and_undeclared_blocks_need_a_program_site() -> None:
    with pytest.raises(ValueError, match="no support to enumerate; pass update="):
        GibbsKernel(["z"], host=True)
    undeclared = CycleKernel(
        {"z": GibbsKernel(["z"], update=_keep), "nuts": NUTSKernel(3)}
    )
    with pytest.raises(ValueError, match=r"\['z'\] are not latent sample.*host=True"):
        MCMC(
            undeclared, 1, 1, num_chains=1, seed=0, initial_values={"z": torch.zeros(2)}
        ).run(_program(_optional_host_score), _X, {})
    with pytest.raises(ValueError, match="declare it the same way"):
        CycleKernel(
            [GibbsKernel(["z"], update=_keep, host=True), GibbsKernel(["z"], _keep)]
        )
    assert _cycle().host_sites == frozenset({"z"})
    assert _cycle().gibbs_sites == frozenset({"z"})
    assert NUTSKernel().host_sites == frozenset()
    assert GibbsKernel(["u"], update=_keep).host_sites == frozenset()
    # A block over program sites keeps the configuration it had, so its
    # checkpoints stay compatible; a host block records that it is one.
    assert "host" not in GibbsKernel(["u"], update=_keep).configuration()
    assert GibbsKernel(["z"], update=_keep, host=True).configuration()["host"] is True


def test_smc_refuses_host_data_blocks() -> None:
    with pytest.raises(ValueError, match="declares no prior for them"):
        SMC(_cycle(), num_particles=4, seed=0).run(_program(), _X, {})


def test_host_data_are_saved_in_checkpoints_and_resume_bitwise(
    tmp_path: Path,
) -> None:
    def driver(
        start: torch.Tensor,
        checkpoint: MCMCCheckpoint | None = None,
        callback: MCMCCallback | None = None,
    ) -> MCMC:
        return MCMC(
            _cycle(),
            3,
            5,
            num_chains=2,
            seed=9,
            initial_values={"z": start},
            checkpoint=checkpoint,
            callback=callback,
        )

    model = _program()
    start = torch.tensor([1.0, 0.0, 0.0, 1.0])
    expected = driver(start).run(model, _X, {})

    class Interrupted(Exception):
        pass

    def interrupt(
        chain: int, iteration: int, state: KernelState, phase: MCMCPhase
    ) -> None:
        del state
        if chain == 1 and phase == "sampling" and iteration == 2:
            raise Interrupted

    with pytest.raises(Interrupted):
        driver(start, MCMCCheckpoint(directory=str(tmp_path), every=1), interrupt).run(
            model, _X, {}
        )
    stored = torch.load(tmp_path / "mcmc_checkpoint.pt", weights_only=True)
    saved = stored["chains"][1]["state"]["gibbs_values"]["z"]
    assert saved.shape == (4,) and saved.dtype == start.dtype
    with pytest.raises(ValueError, match="different run settings"):
        driver(torch.zeros(4)).resume(tmp_path, model, _X, {})
    resumed = driver(start).resume(tmp_path, model, _X, {})
    for name in expected.samples:
        assert torch.equal(expected.samples[name], resumed.samples[name])
    assert torch.equal(expected.log_densities, resumed.log_densities)
    for name in expected.sample_stats:
        assert torch.equal(expected.sample_stats[name], resumed.sample_stats[name])
    assert not torch.equal(resumed.samples["z"][:, 0], resumed.samples["z"][:, -1])
