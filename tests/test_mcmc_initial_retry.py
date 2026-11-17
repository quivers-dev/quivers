"""Chains redraw an initial position whose log-density is not finite.

The program scores ``log(b - c)`` for a real latent ``b`` and a threshold
``c``, so its log-density is finite exactly where ``b > c``. The accepted
start is predicted independently by replaying the chain's uniform stream on
``(-2, 2)`` with NumPy-level arithmetic and taking the first draw above the
threshold.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from quivers.continuous import Draw, Euclidean, MonadicProgram, make_fixed_normal
from quivers.continuous.program_steps import Score, reading
from quivers.core import Unit
from quivers.inference.mcmc import MCMC, NUTSKernel
from quivers.inference.mcmc import driver as driver_module

RADIUS = 2.0
CHAINS = 6
SEED = 17


def _threshold_program(threshold: float) -> MonadicProgram:
    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        return (env["b"].reshape(1) - threshold).log()

    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("b",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=reading(score, ("b",))),
        ],
        return_vars=(),
    )


def _driver(strategy: str = "uniform", **options: object) -> MCMC:
    return MCMC(
        kernel=NUTSKernel(max_tree_depth=3),
        num_warmup=3,
        num_samples=3,
        num_chains=CHAINS,
        init_strategy=strategy,
        init_radius=RADIUS,
        seed=SEED,
        **options,
    )


def _replayed_starts(
    accept: Callable[[float], bool], limit: int
) -> tuple[list[float], list[int]]:
    """First accepted uniform draw of every chain, and its attempt number."""
    starts, attempts = [], []
    for chain in range(CHAINS):
        stream = MCMC._chain_generator(SEED, chain)
        for attempt in range(1, limit + 1):
            draw = float(RADIUS * (2.0 * torch.rand(1, generator=stream) - 1.0))
            if accept(draw):
                starts.append(draw)
                attempts.append(attempt)
                break
        else:
            raise AssertionError("the replay found no finite start")
    return starts, attempts


def test_a_non_finite_start_is_redrawn_from_the_chain_stream() -> None:
    threshold = 1.0
    expected, attempts = _replayed_starts(lambda draw: draw > threshold, 100)
    # A quarter of (-2, 2) is finite, so some chain must need a redraw for
    # the test to exercise the retry at all.
    assert max(attempts) > 1
    driver = _driver()
    program = _threshold_program(threshold)
    starts = driver.chain_start_values(program, torch.zeros(1, 1), {})
    observed = [float(values["b"].reshape(())) for values in starts]
    assert observed == pytest.approx(expected, abs=0.0)
    result = driver.run(program, torch.zeros(1, 1), {})
    draws = result.samples["b"]
    assert draws.shape[:2] == (CHAINS, 3)
    assert bool(torch.isfinite(draws).all()) and bool((draws > threshold).all())


def test_the_first_finite_start_is_kept_unchanged() -> None:
    # With the whole init region finite, every chain keeps its first draw.
    expected, attempts = _replayed_starts(lambda draw: draw > -3.0, 1)
    assert attempts == [1] * CHAINS
    starts = _driver().chain_start_values(
        _threshold_program(-3.0), torch.zeros(1, 1), {}
    )
    assert [float(v["b"].reshape(())) for v in starts] == pytest.approx(
        expected, abs=0.0
    )


def test_the_attempt_budget_is_reported_when_no_start_is_finite() -> None:
    # No point of (-2, 2) exceeds 5, so every attempt fails.
    with pytest.raises(ValueError, match="non-finite terms: lik") as caught:
        _driver().run(_threshold_program(5.0), torch.zeros(1, 1), {})
    notes = " ".join(caught.value.__notes__)
    assert f"attempt {driver_module.INITIAL_ATTEMPTS} of at most" in notes
    assert driver_module.INITIAL_ATTEMPTS == 100


def test_a_deterministic_start_is_not_redrawn() -> None:
    with pytest.raises(ValueError, match="non-finite terms: lik") as caught:
        _driver("zero").run(_threshold_program(1.0), torch.zeros(1, 1), {})
    assert "attempt 1 of at most" in " ".join(caught.value.__notes__)
    # Named values fixing every coordinate repeat the discarded position.
    with pytest.raises(ValueError, match="non-finite terms: lik") as caught:
        _driver(initial_values={"b": torch.tensor(0.5)}).run(
            _threshold_program(1.0), torch.zeros(1, 1), {}
        )
    assert "attempt 2 of at most" in " ".join(caught.value.__notes__)


def test_a_program_error_is_raised_at_the_first_attempt() -> None:
    calls = 0
    armed = False

    def score(env: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        if not armed:
            return env["b"].reshape(1) * 0.0
        calls += 1
        raise RuntimeError("broken score function")

    program = MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("b",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="lik", score=reading(score, ("b",))),
        ],
        return_vars=(),
    )
    driver = _driver()
    setup = driver._setup(program, torch.zeros(1, 1), {}, "test", SEED)
    armed = True
    with pytest.raises(RuntimeError, match="broken score function"):
        driver._accepted_start(setup, None, MCMC._chain_generator(SEED, 0), 0)
    assert calls == 1
