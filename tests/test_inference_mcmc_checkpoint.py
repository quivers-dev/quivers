"""Tests for MCMC checkpoints, resumption, budgets, and per-draw statistics.

Cover:

1. Resumption is bit for bit an uninterrupted run with the same seed,
   for every kernel type (HMC and NUTS in both adaptation modes,
   Metropolis, Gibbs alone, a cycle, and a mixture) and for
   interruptions mid-warmup, exactly at the warmup boundary,
   mid-sampling, and between chains.
2. Budgets stop a chain at a transition boundary, say why, and a
   budget-stopped run resumes to the uninterrupted result.
3. A write interrupted midway leaves the previous checkpoint intact
   and resumable.
4. A checkpoint from a different model, data, kernel configuration,
   run settings, seed, or format version is refused, and so is a file
   that would need code to be executed to load.
5. Every observation kind a run accepts enters its fingerprint by
   content: a run on a corpus of tokens checkpoints and resumes, a
   corpus differing in one token is refused, equal data (tensor, list,
   array, corpus, mapping) render equally, differing data render
   differently, and a datum of an unknown kind is refused, not dropped.
6. Per-draw statistics: shapes, exact evaluation counts against the
   integrator's arithmetic, the kinetic energy implied by ``energy``
   against its stationary Gamma law, and the ArviZ mapping.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
from typing import BinaryIO

import numpy
import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.diagnostics import to_datatree
from quivers.dsl import loads
from quivers.inference.mcmc import (
    CHECKPOINT_FILENAME,
    MCMC,
    CycleKernel,
    GibbsKernel,
    HMCKernel,
    KernelState,
    MCMCCheckpoint,
    MCMCCallback,
    MCMCKernel,
    MCMCPhase,
    MCMCResult,
    MetropolisKernel,
    MixtureKernel,
    NUTSKernel,
    RandomWalkProposal,
    StateValue,
)
from quivers.inference.mcmc.checkpoint import (
    ObservedDatum,
    _digest,
    observation_digest,
)
import quivers.inference.mcmc.driver as driver_module
from quivers.diagnostics import ess as effective_sample_size
from tests.test_inference_mcmc_compose import _label_program, _switch_program


def _location_scale_model(scale: float = 1.0) -> MonadicProgram:
    model = loads(
        "object Obs : FinSet 20\n"
        "program p : Obs -> Obs\n"
        f"    sample mu <- Normal(loc=0.0, scale={scale})\n"
        "    sample tau <- HalfNormal(scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=tau)\n"
        "    return mu\n"
        "export p\n"
    ).morphism
    assert isinstance(model, MonadicProgram)
    return model


_Y = {"y": torch.linspace(-1.0, 2.0, 20)}
_X = torch.zeros(1, 1)

_KERNELS: dict[str, tuple[Callable[[], MCMCKernel], Callable[[], MonadicProgram]]] = {
    "hmc-single": (
        lambda: HMCKernel(num_steps=5, mass_matrix="diagonal"),
        _location_scale_model,
    ),
    "hmc-windowed-dense": (
        lambda: HMCKernel(num_steps=5, mass_matrix="dense", adaptation="windowed"),
        _location_scale_model,
    ),
    "nuts-single": (lambda: NUTSKernel(max_tree_depth=5), _location_scale_model),
    "nuts-windowed": (
        lambda: NUTSKernel(max_tree_depth=5, adaptation="windowed"),
        _location_scale_model,
    ),
    "metropolis": (
        lambda: MetropolisKernel(RandomWalkProposal(0.3)),
        _location_scale_model,
    ),
    "gibbs": (lambda: GibbsKernel(["c"]), _label_program),
    "cycle": (
        lambda: CycleKernel(
            {
                "gibbs": GibbsKernel(["z"]),
                "nuts": NUTSKernel(max_tree_depth=4, adaptation="windowed"),
                "walk": MetropolisKernel(RandomWalkProposal(0.5)),
            }
        ),
        _switch_program,
    ),
    "mixture": (
        lambda: MixtureKernel(
            {
                "hmc": HMCKernel(num_steps=4, mass_matrix="diagonal"),
                "walk": MetropolisKernel(RandomWalkProposal(0.5)),
            },
            probs=[0.4, 0.6],
        ),
        _location_scale_model,
    ),
}

_WARMUP, _SAMPLES, _CHAINS, _EVERY = 24, 16, 2, 4


class _Crash(Exception):
    """A simulated interruption."""


def _crash_at(chain: int, iteration: int, phase: MCMCPhase) -> MCMCCallback:
    def callback(c: int, i: int, state: KernelState, p: MCMCPhase) -> None:
        del state
        if (c, i, p) == (chain, iteration, phase):
            raise _Crash

    return callback


def _data(name: str) -> dict[str, torch.Tensor]:
    return _Y if _KERNELS[name][1] is _location_scale_model else {}


def _mcmc(
    name: str,
    *,
    checkpoint: MCMCCheckpoint | None = None,
    callback: MCMCCallback | None = None,
    max_seconds: float | None = None,
    max_evaluations: int | None = None,
) -> MCMC:
    make_kernel, _ = _KERNELS[name]
    return MCMC(
        make_kernel(),
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=_CHAINS,
        seed=17,
        checkpoint=checkpoint,
        callback=callback,
        max_seconds=max_seconds,
        max_evaluations=max_evaluations,
    )


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality that treats NaN as equal to NaN."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        both_nan = torch.isnan(a) & torch.isnan(b)
        return bool(((a == b) | both_nan).all())
    return bool(torch.equal(a, b))


def _assert_identical(a: MCMCResult, b: MCMCResult) -> None:
    assert a.samples.keys() == b.samples.keys()
    for name in a.samples:
        assert _same(a.samples[name], b.samples[name]), name
        assert _same(a.r_hat[name], b.r_hat[name]), name
        assert _same(a.ess[name], b.ess[name]), name
    assert _same(a.log_densities, b.log_densities)
    assert _same(a.acceptance_rates, b.acceptance_rates)
    assert _same(a.divergence_counts, b.divergence_counts)
    assert a.sample_stats.keys() == b.sample_stats.keys()
    for key in a.sample_stats:
        assert _same(a.sample_stats[key], b.sample_stats[key]), key
    assert a.component_acceptance.keys() == b.component_acceptance.keys()
    for key in a.component_acceptance:
        assert _same(a.component_acceptance[key], b.component_acceptance[key]), key
    assert _same(a.num_evaluations, b.num_evaluations)
    assert _same(a.num_gradient_evaluations, b.num_gradient_evaluations)
    assert a.stopped_by == b.stopped_by
    assert _same(a.num_draws, b.num_draws)


_CRASHES: dict[str, tuple[int, int, MCMCPhase]] = {
    "mid-warmup": (0, 9, "warmup"),
    # The last checkpoint is the one after the last warmup transition,
    # before adaptation stops.
    "warmup-boundary": (1, 0, "sampling"),
    "mid-sampling": (1, 6, "sampling"),
    # Chain 0 finished; chain 1 had made one transition.
    "between-chains": (1, 0, "warmup"),
}


@pytest.mark.parametrize("crash", sorted(_CRASHES))
@pytest.mark.parametrize("name", sorted(_KERNELS))
def test_resumption_is_bitwise_an_uninterrupted_run(
    name: str, crash: str, tmp_path: Path
) -> None:
    model = _KERNELS[name][1]()
    reference = _mcmc(name).run(model, _X, _data(name))

    checkpoint = MCMCCheckpoint(directory=str(tmp_path), every=_EVERY)
    chain, iteration, phase = _CRASHES[crash]
    interrupted = _mcmc(
        name,
        checkpoint=checkpoint,
        callback=_crash_at(chain, iteration, phase),
    )
    with pytest.raises(_Crash):
        interrupted.run(model, _X, _data(name))
    assert (tmp_path / CHECKPOINT_FILENAME).is_file()

    resumed = _mcmc(name, checkpoint=checkpoint).resume(
        tmp_path, model, _X, _data(name)
    )
    _assert_identical(reference, resumed)
    assert resumed.stopped_by == ("draws",) * _CHAINS
    assert not resumed.truncated


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_evaluation_budget_stops_at_a_transition_boundary(tmp_path: Path) -> None:
    budget = 150
    steps = 5
    model = _location_scale_model()
    stopped = MCMC(
        HMCKernel(num_steps=steps, mass_matrix="diagonal"),
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=_CHAINS,
        seed=17,
        checkpoint=MCMCCheckpoint(directory=str(tmp_path), every=None),
        max_evaluations=budget,
    ).run(model, _X, _Y)
    assert stopped.stopped_by == ("evaluations", "evaluations")
    assert stopped.truncated
    # Each transition costs `steps` evaluations (plus one for the first
    # state), and the budget is checked between transitions.
    for count in stopped.num_evaluations.tolist():
        assert budget <= count < budget + steps
        assert (count - 1) % steps == 0
    transitions = (budget - 1 + steps - 1) // steps
    draws = transitions - _WARMUP
    assert stopped.num_draws.tolist() == [draws, draws]
    assert stopped.samples["mu"].shape == (_CHAINS, draws)
    assert stopped.sample_stats["energy"].shape == (_CHAINS, draws)

    resumed = MCMC(
        HMCKernel(num_steps=steps, mass_matrix="diagonal"),
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=_CHAINS,
        seed=17,
    ).resume(tmp_path, model, _X, _Y)
    reference = MCMC(
        HMCKernel(num_steps=steps, mass_matrix="diagonal"),
        num_warmup=_WARMUP,
        num_samples=_SAMPLES,
        num_chains=_CHAINS,
        seed=17,
    ).run(model, _X, _Y)
    _assert_identical(reference, resumed)


def test_time_budget_stops_before_any_draw_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A clock that advances one second per reading: every check of the
    # budget finds it spent.
    ticks = iter(range(10**6))
    monkeypatch.setattr(driver_module.time, "monotonic", lambda: float(next(ticks)))
    model = _location_scale_model()
    stopped = _mcmc(
        "nuts-windowed",
        checkpoint=MCMCCheckpoint(directory=str(tmp_path), every=None),
        max_seconds=0.5,
    ).run(model, _X, _Y)
    assert stopped.stopped_by == ("seconds", "seconds")
    assert stopped.num_draws.tolist() == [0, 0]
    assert stopped.samples["mu"].shape == (_CHAINS, 0)
    assert stopped.samples["tau"].shape == (_CHAINS, 0)
    assert stopped.num_samples == 0
    monkeypatch.undo()
    resumed = _mcmc("nuts-windowed").resume(tmp_path, model, _X, _Y)
    _assert_identical(_mcmc("nuts-windowed").run(model, _X, _Y), resumed)


def test_budgets_without_a_checkpoint_return_the_truncated_run() -> None:
    result = _mcmc("metropolis", max_evaluations=40).run(
        _location_scale_model(), _X, _Y
    )
    assert result.truncated
    assert all(reason == "evaluations" for reason in result.stopped_by)
    assert result.samples["mu"].shape[1] == int(result.num_draws.min())


# ---------------------------------------------------------------------------
# Interrupted writes
# ---------------------------------------------------------------------------


def test_a_write_interrupted_midway_leaves_the_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _location_scale_model()
    real_save = torch.save
    calls = [0]

    def failing_save(obj: dict[str, StateValue], handle: BinaryIO) -> None:
        calls[0] += 1
        if calls[0] == 3:
            # Half a file, then the process dies.
            handle.write(b"PK\x03\x04 truncated")
            raise OSError("simulated power loss")
        real_save(obj, handle)

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="simulated power loss"):
        _mcmc(
            "nuts-single", checkpoint=MCMCCheckpoint(directory=str(tmp_path), every=5)
        ).run(model, _X, _Y)
    monkeypatch.setattr(torch, "save", real_save)

    assert (tmp_path / (CHECKPOINT_FILENAME + ".tmp")).exists()
    resumed = _mcmc("nuts-single").resume(tmp_path, model, _X, _Y)
    _assert_identical(_mcmc("nuts-single").run(model, _X, _Y), resumed)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.fixture
def written(tmp_path: Path) -> Path:
    _mcmc(
        "nuts-single", checkpoint=MCMCCheckpoint(directory=str(tmp_path), every=5)
    ).run(_location_scale_model(), _X, _Y)
    return tmp_path


def test_resume_refuses_a_different_model_or_data(written: Path) -> None:
    driver = _mcmc("nuts-single")
    with pytest.raises(ValueError, match="different model or data"):
        driver.resume(written, _location_scale_model(scale=2.0), _X, _Y)
    with pytest.raises(ValueError, match="different model or data"):
        driver.resume(
            written, _location_scale_model(), _X, {"y": torch.linspace(0.0, 1.0, 20)}
        )


def test_resume_refuses_a_different_kernel_settings_or_seed(written: Path) -> None:
    model = _location_scale_model()
    with pytest.raises(ValueError, match="different kernel configuration"):
        MCMC(NUTSKernel(max_tree_depth=6), _WARMUP, _SAMPLES, _CHAINS, seed=17).resume(
            written, model, _X, _Y
        )
    with pytest.raises(ValueError, match="different kernel configuration"):
        MCMC(
            NUTSKernel(max_tree_depth=5, adaptation="windowed"),
            _WARMUP,
            _SAMPLES,
            _CHAINS,
            seed=17,
        ).resume(written, model, _X, _Y)
    with pytest.raises(ValueError, match="different run settings"):
        MCMC(
            NUTSKernel(max_tree_depth=5), _WARMUP, _SAMPLES + 1, _CHAINS, seed=17
        ).resume(written, model, _X, _Y)
    with pytest.raises(ValueError, match="seed 17, not 18"):
        MCMC(NUTSKernel(max_tree_depth=5), _WARMUP, _SAMPLES, _CHAINS, seed=18).resume(
            written, model, _X, _Y
        )


def test_resume_refuses_another_format_version(written: Path) -> None:
    path = written / CHECKPOINT_FILENAME
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint format 999"):
        _mcmc("nuts-single").resume(written, _location_scale_model(), _X, _Y)


class _Payload:
    """An object whose unpickling would run code."""

    def __reduce__(self) -> tuple[Callable[[str], None], tuple[str]]:
        return (print, ("code ran while loading",))


def test_a_file_that_needs_code_to_load_is_refused(tmp_path: Path) -> None:
    torch.save(
        {"format_version": 1, "header": _Payload()}, tmp_path / CHECKPOINT_FILENAME
    )
    with pytest.raises(ValueError, match="not a readable MCMC checkpoint"):
        _mcmc("nuts-single").resume(tmp_path, _location_scale_model(), _X, _Y)


def test_run_refuses_to_overwrite_and_resume_needs_a_checkpoint(
    written: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    with pytest.raises(FileExistsError, match="already holds a checkpoint"):
        _mcmc("nuts-single", checkpoint=MCMCCheckpoint(directory=str(written))).run(
            _location_scale_model(), _X, _Y
        )
    empty = tmp_path_factory.mktemp("empty")
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        _mcmc("nuts-single").resume(empty, _location_scale_model(), _X, _Y)
    with pytest.raises(Exception, match="every must be >= 1"):
        MCMCCheckpoint(directory=str(empty), every=0)


# ---------------------------------------------------------------------------
# Fingerprints of non-tensor observations
# ---------------------------------------------------------------------------

_PCFG = """
object Term : FinSet 16
object Weight : Real 1
object Sentence : FinSet 2

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4, normalize=lhs]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S)
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP)
    rule vp_v : span(I, J, V) |- span(I, J, VP)
    lexicon
        "the"    : Det = the
        "a"      : Det = a
        "cat"    : N   = cat
        "dog"    : N   = dog
        "sleeps" : V   = sleeps
        "runs"   : V   = runs

program bayesian_pcfg : Term -> Weight
    sample p_det <- Dirichlet(concentration=[1.0, 1.0])
    sample p_n <- Dirichlet(concentration=[1.0, 1.0])
    sample p_v <- Dirichlet(concentration=[1.0, 1.0])
    let chart = parse(PCFG, corpus, Sentence, ("Det", p_det), ("N", p_n), ("V", p_v))
    score evidence = chart.goal_weight()
    return evidence

export bayesian_pcfg
"""

_CORPUS = [["the", "cat", "runs"], ["a", "dog", "sleeps"]]


def _pcfg_mcmc(
    checkpoint: MCMCCheckpoint | None = None, callback: MCMCCallback | None = None
) -> MCMC:
    return MCMC(
        NUTSKernel(max_tree_depth=3),
        num_warmup=6,
        num_samples=6,
        num_chains=1,
        seed=5,
        checkpoint=checkpoint,
        callback=callback,
    )


def test_mcmc_runs_and_resumes_on_a_token_corpus(tmp_path: Path) -> None:
    program = loads(_PCFG).morphism
    assert isinstance(program, MonadicProgram)
    reference = _pcfg_mcmc().run(program, _X, {"corpus": _CORPUS})
    assert reference.samples["p_det"].shape[-1] == 2

    checkpoint = MCMCCheckpoint(directory=str(tmp_path), every=2)
    with pytest.raises(_Crash):
        _pcfg_mcmc(checkpoint, callback=_crash_at(0, 3, "sampling")).run(
            program, _X, {"corpus": _CORPUS}
        )
    # A corpus differing in one token is different data.
    swapped = [["the", "cat", "runs"], ["a", "cat", "sleeps"]]
    with pytest.raises(ValueError, match="different model or data"):
        _pcfg_mcmc(checkpoint).resume(tmp_path, program, _X, {"corpus": swapped})
    # The same corpus, rebuilt from tuples, is the same data.
    rebuilt = tuple(tuple(sentence) for sentence in _CORPUS)
    resumed = _pcfg_mcmc(checkpoint).resume(tmp_path, program, _X, {"corpus": rebuilt})
    _assert_identical(reference, resumed)


def _rendered(value: ObservedDatum) -> str:
    """The digest a run fingerprint takes of one datum."""
    return _digest(observation_digest("datum", value))


_EQUAL_DATA: dict[str, tuple[ObservedDatum, ObservedDatum]] = {
    "tensor": (torch.arange(6.0).reshape(2, 3), torch.arange(6.0).reshape(2, 3)),
    "tensor-and-list": (
        torch.tensor([[0.5, 1.5], [2.5, 3.5]]),
        [[0.5, 1.5], [2.5, 3.5]],
    ),
    "tensor-and-array": (
        torch.tensor([1, 2, 3]),
        numpy.array([1, 2, 3], dtype=numpy.int64),
    ),
    "number": (3, 3),
    "corpus": (
        [["the", "cat", "runs"], ["a", "dog", "sleeps"]],
        (("the", "cat", "runs"), ("a", "dog", "sleeps")),
    ),
    "ragged-corpus": ([["the", "cat"], ["runs"]], [["the", "cat"], ["runs"]]),
    "token": ("cat", "cat"),
    "mapping": (
        {"tokens": ["a", "b"], "weights": torch.ones(2)},
        {"weights": torch.ones(2), "tokens": ["a", "b"]},
    ),
}

_DIFFERING_DATA: dict[str, tuple[ObservedDatum, ObservedDatum]] = {
    "tensor-value": (torch.zeros(3), torch.tensor([0.0, 0.0, 1e-7])),
    "tensor-dtype": (torch.zeros(3), torch.zeros(3, dtype=torch.float64)),
    "tensor-shape": (torch.zeros(2, 3), torch.zeros(3, 2)),
    "number": (3, 4),
    "number-and-string": (1, "1"),
    "corpus-token": (
        [["the", "cat", "runs"], ["a", "dog", "sleeps"]],
        [["the", "cat", "runs"], ["a", "dog", "runs"]],
    ),
    "corpus-order": (
        [["the", "cat", "runs"], ["a", "dog", "sleeps"]],
        [["a", "dog", "sleeps"], ["the", "cat", "runs"]],
    ),
    "corpus-length": ([["the", "cat", "runs"]], [["the", "cat", "runs"]] * 2),
    "corpus-split": ([["the", "cat"], ["runs"]], [["the"], ["cat", "runs"]]),
    "token-and-sentence": ("cat", ["cat"]),
    "mapping-value": ({"tokens": ["a", "b"]}, {"tokens": ["a", "c"]}),
    "mapping-key": ({"tokens": ["a", "b"]}, {"words": ["a", "b"]}),
}


@pytest.mark.parametrize("case", sorted(_EQUAL_DATA))
def test_equal_observations_have_equal_fingerprints(case: str) -> None:
    first, second = _EQUAL_DATA[case]
    assert _rendered(first) == _rendered(second)


@pytest.mark.parametrize("case", sorted(_DIFFERING_DATA))
def test_differing_observations_have_differing_fingerprints(case: str) -> None:
    first, second = _DIFFERING_DATA[case]
    assert _rendered(first) != _rendered(second)


def test_an_observation_of_an_unknown_kind_is_refused_not_dropped() -> None:
    with pytest.raises(TypeError, match="'datum' holds a value of type set"):
        _rendered(["the", {"cat"}])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="key 1 of type int"):
        _rendered({1: "cat"})  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Per-draw statistics
# ---------------------------------------------------------------------------


def test_hmc_evaluation_counts_follow_the_integrator() -> None:
    steps = 6
    result = MCMC(
        HMCKernel(num_steps=steps), num_warmup=10, num_samples=15, num_chains=2, seed=1
    ).run(_location_scale_model(), _X, _Y)
    # One gradient for the first state, then `steps` per transition.
    expected = 1 + (10 + 15) * steps
    assert result.num_evaluations.tolist() == [expected, expected]
    assert result.num_gradient_evaluations.tolist() == [expected, expected]
    assert bool((result.sample_stats["num_steps"] == steps).all())


def test_metropolis_evaluation_counts_follow_its_acceptances() -> None:
    result = MCMC(
        MetropolisKernel(RandomWalkProposal(0.4)),
        num_warmup=0,
        num_samples=60,
        num_chains=2,
        seed=2,
    ).run(_location_scale_model(), _X, _Y)
    accepted = result.sample_stats["accepted"].sum(dim=1).to(torch.int64)
    # A value per proposal, a gradient for the first state and per
    # acceptance (each a forward pass too).
    assert result.num_gradient_evaluations.tolist() == (1 + accepted).tolist()
    assert result.num_evaluations.tolist() == (1 + 60 + accepted).tolist()


@pytest.mark.parametrize("kind", ["nuts", "hmc"])
def test_energy_implies_a_stationary_kinetic_energy(kind: str) -> None:
    """At stationarity the momentum paired with a draw is ``N(0, M)``,
    so the kinetic energy ``energy + log_density`` is ``Gamma(D/2, 1)``:
    mean ``D/2``, variance ``D/2``."""
    kernel: MCMCKernel = (
        NUTSKernel(max_tree_depth=6, mass_matrix="diagonal")
        if kind == "nuts"
        else HMCKernel(num_steps=8, mass_matrix="diagonal")
    )
    result = MCMC(kernel, num_warmup=200, num_samples=400, num_chains=2, seed=5).run(
        _location_scale_model(), _X, _Y
    )
    stats = result.sample_stats
    expected_keys = {"accept_prob", "accepted", "diverged", "step_size", "num_steps"}
    expected_keys |= {"energy"} | ({"tree_depth"} if kind == "nuts" else set())
    assert set(stats) == expected_keys
    for values in stats.values():
        assert values.shape == (2, 400)
        assert values.dtype == torch.float64
    kinetic = stats["energy"] + result.log_densities.to(torch.float64)
    assert bool((kinetic >= -1e-4).all())
    dim = 2
    ess = min(
        float(effective_sample_size(kinetic, method="mean")), float(kinetic.numel())
    )
    error = abs(float(kinetic.mean()) - dim / 2)
    assert error < 5.0 * math.sqrt(dim / 2 / ess), (error, ess)
    if kind == "nuts":
        depth = stats["tree_depth"]
        assert bool((depth == depth.round()).all())
        assert bool((stats["num_steps"] <= 2.0**depth - 1).all())


def test_callback_sees_every_transition_in_order() -> None:
    seen: list[tuple[int, int, str]] = []

    def callback(
        chain: int, iteration: int, state: KernelState, phase: MCMCPhase
    ) -> None:
        assert isinstance(state, KernelState)
        seen.append((chain, iteration, phase))

    MCMC(
        HMCKernel(num_steps=2),
        num_warmup=3,
        num_samples=2,
        num_chains=2,
        seed=0,
        callback=callback,
    ).run(_location_scale_model(), _X, _Y)
    expected = [
        (c, i, p)
        for c in range(2)
        for p, n in (("warmup", 3), ("sampling", 2))
        for i in range(n)
    ]
    assert seen == expected


def test_to_datatree_maps_sample_stats_to_arviz_names() -> None:
    result = MCMC(
        NUTSKernel(max_tree_depth=4),
        num_warmup=20,
        num_samples=10,
        num_chains=2,
        seed=0,
    ).run(_location_scale_model(), _X, _Y)
    stats = to_datatree(result)["sample_stats"]
    for name in (
        "lp",
        "diverging",
        "tree_depth",
        "n_steps",
        "energy",
        "step_size",
        "acceptance_rate",
    ):
        assert stats[name].shape == (2, 10), name
    assert stats["diverging"].dtype == bool
    assert stats["tree_depth"].dtype == "int64"
    assert stats["n_steps"].dtype == "int64"
