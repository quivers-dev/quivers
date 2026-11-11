"""Tests for the windowed (Stan-style) warmup adaptation.

Cover:

1. `AdaptationSchedule.windows` against an independent oracle: a
   line-by-line transcription of Stan's iteration-counter state machine
   (``stan::mcmc::windowed_adaptation``), for warmups long enough for
   the configured buffers, too short for them (the 15% / 75% / 10%
   rescaling), and shorter than 20 iterations (no metric estimation),
   plus the published 1000-iteration schedule.
2. The windowed kernels install the metric exactly at the end of each
   slow window, and the adapted metric matches the known covariance of
   an anisotropic Gaussian target within Monte Carlo error, with the
   tolerance set from the effective sample size of the window that
   produced it. The single-phase mode, started from the same far-off
   point, folds the transient into its estimate and fails the same
   check.
3. The single-phase mode reproduces its fixed-seed draws bit for bit.
4. Windowed NUTS recovers a conjugate posterior, and the step-size
   initialization draws only from the chain's generator and refuses an
   improper target.
"""

from __future__ import annotations

import math

import didactic.api as dx
import pytest
import torch
from torch.distributions import MultivariateNormal, Normal

from quivers.continuous import Draw, Euclidean, MonadicProgram, make_fixed_normal
from quivers.continuous.program_steps import Score, reading
from quivers.core import Unit
from quivers.dsl import loads
from quivers.inference.mcmc import (
    MCMC,
    AdaptationSchedule,
    AdaptationWindow,
    HMCKernel,
    NUTSKernel,
    PotentialFn,
)
from quivers.diagnostics import ess as effective_sample_size
from quivers.inference.registry import LatentRegistry


# ---------------------------------------------------------------------------
# The schedule against Stan's state machine
# ---------------------------------------------------------------------------


def _stan_windows(
    num_warmup: int, init_buffer: int, term_buffer: int, base_window: int
) -> tuple[set[int], list[int]]:
    """Run Stan's ``windowed_adaptation`` counter for one warmup.

    A transcription of ``set_window_params``, ``restart``,
    ``adaptation_window``, ``end_adaptation_window``, and
    ``compute_next_window`` from Stan's
    ``src/stan/mcmc/windowed_adaptation.hpp``, driven exactly as
    ``learn_variance`` drives it: once per warmup iteration, adding the
    iteration's sample when inside an adaptation window and updating the
    metric at a window's end.

    Returns
    -------
    tuple[set[int], list[int]]
        The iterations whose samples enter a metric estimate, and the
        iterations at which the metric is updated.
    """
    # Constructor state: every parameter zero.
    n_warm, init, term, base = 0, 0, 0, 0
    if num_warmup >= 20:
        if init_buffer + base_window + term_buffer > num_warmup:
            n_warm = num_warmup
            init = int(0.15 * num_warmup)
            term = int(0.1 * num_warmup)
            base = num_warmup - (init + term)
        else:
            n_warm, init, term, base = (
                num_warmup,
                init_buffer,
                term_buffer,
                base_window,
            )
    counter = 0
    window_size = base
    next_window = init + window_size - 1
    sampled: set[int] = set()
    updates: list[int] = []

    def adaptation_window() -> bool:
        return init <= counter < n_warm - term and counter != n_warm

    def end_adaptation_window() -> bool:
        return counter == next_window and counter != n_warm

    for _ in range(num_warmup):
        if adaptation_window():
            sampled.add(counter)
        if end_adaptation_window():
            # compute_next_window
            if next_window != n_warm - term - 1:
                window_size *= 2
                next_window = counter + window_size
                if next_window != n_warm - term - 1:
                    if next_window + 2 * window_size >= n_warm - term:
                        next_window = n_warm - term - 1
            updates.append(counter)
        counter += 1
    return sampled, updates


_WARMUPS = (0, 1, 7, 19, 20, 21, 33, 100, 149, 150, 151, 160, 175, 200, 260, 333)
_WARMUPS += (500, 999, 1000, 1001, 2000, 4321)
_SCHEDULES = (
    AdaptationSchedule(),
    AdaptationSchedule(init_buffer=10, term_buffer=10, base_window=5),
    AdaptationSchedule(init_buffer=0, term_buffer=0, base_window=1),
    AdaptationSchedule(init_buffer=30, term_buffer=7, base_window=13),
)


@pytest.mark.parametrize("schedule", _SCHEDULES)
@pytest.mark.parametrize("num_warmup", _WARMUPS)
def test_windows_follow_stans_state_machine(
    schedule: AdaptationSchedule, num_warmup: int
) -> None:
    windows = schedule.windows(num_warmup)
    sampled, updates = _stan_windows(
        num_warmup, schedule.init_buffer, schedule.term_buffer, schedule.base_window
    )
    ours_sampled = {
        i for w in windows if w.adapts_metric for i in range(w.start, w.stop)
    }
    ours_updates = [w.stop - 1 for w in windows if w.adapts_metric]
    assert ours_sampled == sampled
    assert ours_updates == updates
    # The windows tile the warmup without gaps or overlaps.
    cursor = 0
    for window in windows:
        assert window.start == cursor
        assert window.stop > window.start
        cursor = window.stop
    assert cursor == num_warmup


def test_windows_reproduce_the_published_default_schedule() -> None:
    # Stan's documented 1000-iteration warmup: a 75-iteration fast
    # buffer, slow windows of 25, 50, 100, 200, and 500, and a
    # 50-iteration terminal buffer.
    assert AdaptationSchedule().windows(1000) == (
        AdaptationWindow(start=0, stop=75, adapts_metric=False),
        AdaptationWindow(start=75, stop=100, adapts_metric=True),
        AdaptationWindow(start=100, stop=150, adapts_metric=True),
        AdaptationWindow(start=150, stop=250, adapts_metric=True),
        AdaptationWindow(start=250, stop=450, adapts_metric=True),
        AdaptationWindow(start=450, stop=950, adapts_metric=True),
        AdaptationWindow(start=950, stop=1000, adapts_metric=False),
    )


@pytest.mark.parametrize(
    ("num_warmup", "expected"),
    [
        # 75 + 25 + 50 > 100: buffers rescale to floor(15), floor(10).
        (100, ((0, 15, False), (15, 90, True), (90, 100, False))),
        # floor(0.15 * 33) = 4, floor(0.1 * 33) = 3.
        (33, ((0, 4, False), (4, 30, True), (30, 33, False))),
        # Below 20 iterations no metric is estimated.
        (19, ((0, 19, False),)),
        (20, ((0, 3, False), (3, 18, True), (18, 20, False))),
    ],
)
def test_short_warmups_follow_the_documented_rescaling(
    num_warmup: int, expected: tuple[tuple[int, int, bool], ...]
) -> None:
    windows = AdaptationSchedule().windows(num_warmup)
    assert tuple((w.start, w.stop, w.adapts_metric) for w in windows) == expected


def test_schedule_rejects_invalid_lengths() -> None:
    with pytest.raises(dx.ValidationError, match="base_window must be >= 1"):
        AdaptationSchedule(base_window=0)
    with pytest.raises(dx.ValidationError, match="must be >= 0"):
        AdaptationSchedule(init_buffer=-1)
    with pytest.raises(ValueError, match="num_warmup must be >= 0"):
        AdaptationSchedule().windows(-1)


def test_kernel_windows_reflect_mode_and_metric() -> None:
    assert NUTSKernel(adaptation="windowed").adaptation_windows(
        1000
    ) == AdaptationSchedule().windows(1000)
    custom = AdaptationSchedule(init_buffer=5, term_buffer=5, base_window=5)
    assert HMCKernel(
        mass_matrix="dense", adaptation="windowed", adaptation_schedule=custom
    ).adaptation_windows(200) == custom.windows(200)
    # An identity metric is never adapted: one fast window.
    assert HMCKernel(adaptation="windowed").adaptation_windows(300) == (
        AdaptationWindow(start=0, stop=300, adapts_metric=False),
    )
    # The single-phase mode estimates the metric over the whole warmup.
    assert NUTSKernel().adaptation_windows(300) == (
        AdaptationWindow(start=0, stop=300, adapts_metric=True),
    )
    assert NUTSKernel().adaptation_windows(0) == ()
    with pytest.raises(ValueError, match="adaptation must be"):
        NUTSKernel(adaptation="staged")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The adapted metric
# ---------------------------------------------------------------------------


def _gaussian_program(covariance: torch.Tensor) -> MonadicProgram:
    """A program whose latent ``z`` has the posterior ``N(0, covariance)``,
    a vector ``covariance`` standing for its diagonal.

    ``z`` is drawn from a standard normal prior, and a score adds
    ``log N(z; 0, covariance) - log N(z; 0, I)``.
    """
    dim = covariance.shape[0]
    matrix = torch.diag(covariance) if covariance.dim() == 1 else covariance
    target = MultivariateNormal(torch.zeros(dim), covariance_matrix=matrix)
    prior = Normal(0.0, 1.0)

    def correction(env: dict[str, torch.Tensor]) -> torch.Tensor:
        z = env["z"].reshape(-1, dim)
        return target.log_prob(z) - prior.log_prob(z).sum(-1)

    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=dim)),
            ),
            Score(name="target", score=reading(correction, ("z",))),
        ],
        return_vars=(),
    )


def _run_warmup(
    kernel: NUTSKernel | HMCKernel,
    covariance: torch.Tensor,
    num_warmup: int,
    start: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Drive a kernel through a warmup, recording the position and the
    metric after every transition."""
    model = _gaussian_program(covariance)
    registry = LatentRegistry.from_model(model, set())
    potential = PotentialFn(model, registry, torch.zeros(1, 1), {})
    kernel._set_generator(torch.Generator().manual_seed(seed))
    state = kernel.init(registry, model, torch.zeros(1, 1), {}, start)
    kernel.start_adaptation(num_warmup)
    positions = []
    metrics = []
    for _ in range(num_warmup):
        state = kernel.step(state, potential)
        positions.append(state.position)
        metrics.append(kernel.inverse_mass_matrix)
    kernel.stop_adaptation()
    return torch.stack(positions), metrics


def _ess_per_coordinate(draws: torch.Tensor) -> torch.Tensor:
    """Effective sample size of each column of a ``(n, D)`` draw matrix."""
    return torch.stack(
        [
            effective_sample_size(
                draws[:, d].to(torch.float64).unsqueeze(0), method="mean"
            )
            for d in range(draws.shape[1])
        ]
    )


def _regularised(covariance: torch.Tensor, n: int) -> torch.Tensor:
    """Stan's regularisation of an ``n``-draw estimate, applied to the
    true covariance: the estimate's expectation."""
    shrink = n / (n + 5.0)
    jitter = 1e-3 * 5.0 / (n + 5.0)
    if covariance.dim() == 1:
        return shrink * covariance + jitter
    return shrink * covariance + jitter * torch.eye(covariance.shape[0])


_SCALES = torch.tensor([0.2, 1.0, 5.0])
_DENSE = torch.tensor(
    [
        [0.25, 0.30, 0.0],
        [0.30, 1.00, -1.5],
        [0.0, -1.5, 4.0],
    ]
)


def _metric_error(
    estimate: torch.Tensor, covariance: torch.Tensor, draws: torch.Tensor, n: int
) -> tuple[float, float]:
    """The largest standardized error of a metric estimate, and the
    tolerance for it.

    The estimate is compared with the regularised true covariance in
    the target's whitened coordinates, where every entry of the
    difference has standard deviation at most ``sqrt(2 / ess)`` for a
    sample covariance of ``ess`` effective Gaussian draws. The
    tolerance is five such standard deviations, with ``ess`` the
    smallest effective sample size of the whitened draws that produced
    the estimate.
    """
    if covariance.dim() == 1:
        relative = estimate / _regularised(covariance, n) - 1.0
        whitened = draws / covariance.sqrt()
        error = float(relative.abs().max())
    else:
        chol = torch.linalg.cholesky(covariance)
        inverse = torch.linalg.inv(chol)
        delta = inverse @ (estimate - _regularised(covariance, n)) @ inverse.t()
        whitened = draws @ inverse.t()
        error = float(delta.abs().max())
    ess = float(_ess_per_coordinate(whitened).min())
    return error, 5.0 * math.sqrt(2.0 / ess)


@pytest.mark.parametrize("dense", [False, True])
def test_windowed_metric_matches_the_target_covariance(dense: bool) -> None:
    covariance = _DENSE if dense else _SCALES**2
    num_warmup = 1000
    kernel = NUTSKernel(
        step_size=0.1,
        max_tree_depth=8,
        mass_matrix="dense" if dense else "diagonal",
        adaptation="windowed",
    )
    windows = kernel.adaptation_windows(num_warmup)
    last = [w for w in windows if w.adapts_metric][-1]
    start = torch.full((3,), 25.0)
    positions, metrics = _run_warmup(kernel, covariance, num_warmup, start, seed=3)

    # The metric changes exactly at the end of each slow window.
    stops = {w.stop for w in windows if w.adapts_metric}
    for iteration in range(1, num_warmup):
        changed = not torch.equal(metrics[iteration], metrics[iteration - 1])
        assert changed == (iteration + 1 in stops), iteration

    estimate = kernel.inverse_mass_matrix
    draws = positions[last.start : last.stop]
    error, tolerance = _metric_error(
        estimate, covariance, draws, last.stop - last.start
    )
    assert error < tolerance, (error, tolerance)


@pytest.mark.parametrize("dense", [False, True])
def test_single_phase_metric_absorbs_the_transient(dense: bool) -> None:
    """The deliberately wrong variant: one estimate over the whole
    warmup, from the same far-off start, fails the same check."""
    covariance = _DENSE if dense else _SCALES**2
    num_warmup = 1000
    kernel = NUTSKernel(
        step_size=0.1,
        max_tree_depth=8,
        mass_matrix="dense" if dense else "diagonal",
        adaptation="single",
    )
    start = torch.full((3,), 25.0)
    positions, _ = _run_warmup(kernel, covariance, num_warmup, start, seed=3)
    error, tolerance = _metric_error(
        kernel.inverse_mass_matrix, covariance, positions, num_warmup
    )
    assert error > tolerance, (error, tolerance)


def test_windowed_step_size_is_retuned_under_the_final_metric() -> None:
    """After the last slow window, dual averaging restarts from a step
    size initialized under the installed metric, so the frozen step size
    belongs to the metric the chain samples with."""
    covariance = _SCALES**2
    kernel = NUTSKernel(
        step_size=0.1, max_tree_depth=8, mass_matrix="diagonal", adaptation="windowed"
    )
    _run_warmup(kernel, covariance, 400, torch.zeros(3), seed=5)
    assert kernel._dual_avg is not None
    windows = kernel.adaptation_windows(400)
    # The final dual-averaging run covers exactly the terminal window.
    assert kernel._dual_avg.step_count == windows[-1].stop - windows[-1].start
    # Under a metric matched to the target, the optimal step size is of
    # order one whitened unit, far above the 0.2 the narrowest
    # coordinate allows under the identity metric.
    assert 0.3 < kernel.step_size < 3.0


# ---------------------------------------------------------------------------
# Fixed-seed behaviour of the single-phase mode
# ---------------------------------------------------------------------------


def _location_scale_model() -> MonadicProgram:
    return loads(
        "object Obs : FinSet 20\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    sample tau <- HalfNormal(scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=tau)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


@pytest.mark.parametrize(
    ("kernel", "mu", "tau", "step_size"),
    [
        (
            HMCKernel(step_size=0.1, num_steps=5, mass_matrix="diagonal"),
            [
                0.5788787603378296,
                0.6797456741333008,
                0.7778991460800171,
                0.7101612091064453,
            ],
            [
                0.8588930368423462,
                0.8565139174461365,
                0.8598603010177612,
                0.8965176343917847,
            ],
            0.16133334028900623,
        ),
        (
            NUTSKernel(max_tree_depth=4, mass_matrix="dense"),
            [
                0.5872042775154114,
                0.5872042775154114,
                0.6138293743133545,
                0.7573347091674805,
            ],
            [
                1.039720058441162,
                1.039720058441162,
                0.867976188659668,
                0.7471525073051453,
            ],
            0.1469486534707658,
        ),
    ],
    ids=["hmc-diagonal", "nuts-dense"],
)
def test_single_phase_mode_reproduces_its_fixed_seed_draws(
    kernel: HMCKernel | NUTSKernel,
    mu: list[float],
    tau: list[float],
    step_size: float,
) -> None:
    """The default single-phase adaptation draws exactly what it drew
    before the windowed mode existed (values recorded from that code)."""
    assert kernel.adaptation == "single"
    result = MCMC(kernel, num_warmup=30, num_samples=4, num_chains=1, seed=11).run(
        _location_scale_model(),
        torch.zeros(1, 1),
        {"y": torch.linspace(-1.0, 2.0, 20)},
    )
    assert result.samples["mu"][0].tolist() == mu
    assert result.samples["tau"][0].tolist() == tau
    assert kernel.step_size == step_size


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _normal_normal_model() -> MonadicProgram:
    return loads(
        "object Obs : FinSet 20\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : Obs <- Normal(loc=mu, scale=1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


@pytest.mark.parametrize("kind", ["nuts", "hmc"])
def test_windowed_kernels_recover_a_conjugate_posterior(kind: str) -> None:
    y = torch.linspace(0.5, 2.5, 20)
    n = 20
    post_mean = float(y.sum()) / (n + 1)
    post_sd = math.sqrt(1.0 / (n + 1))
    kernel: NUTSKernel | HMCKernel = (
        NUTSKernel(adaptation="windowed")
        if kind == "nuts"
        else HMCKernel(num_steps=8, mass_matrix="diagonal", adaptation="windowed")
    )
    result = MCMC(kernel, num_warmup=300, num_samples=600, num_chains=2, seed=4).run(
        _normal_normal_model(), torch.zeros(1, 1), {"y": y}
    )
    draws = result.samples["mu"]
    ess = float(effective_sample_size(draws.to(torch.float64), method="mean"))
    # Five standard errors of the posterior mean and of the variance.
    assert abs(float(draws.mean()) - post_mean) < 5.0 * post_sd / math.sqrt(ess)
    assert abs(float(draws.var()) / post_sd**2 - 1.0) < 5.0 * math.sqrt(2.0 / ess)
    assert result.total_divergences == 0


def test_windowed_runs_are_reproducible_and_leave_global_rng_alone() -> None:
    model = _normal_normal_model()
    observations = {"y": torch.linspace(0.0, 1.0, 20)}

    def run() -> torch.Tensor:
        kernel = NUTSKernel(max_tree_depth=5, adaptation="windowed")
        return (
            MCMC(kernel, num_warmup=40, num_samples=10, num_chains=2, seed=9)
            .run(model, torch.zeros(1, 1), observations)
            .samples["mu"]
        )

    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    first = run()
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(first, run())


def _flat_program() -> MonadicProgram:
    """A program whose unconstrained log-density is constant."""

    def cancel(env: dict[str, torch.Tensor]) -> torch.Tensor:
        z = env["z"].reshape(-1)
        return 0.5 * z * z

    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_normal(0.0, 1.0, Euclidean(name="R", dim=1)),
            ),
            Score(name="flat", score=reading(cancel, ("z",))),
        ],
        return_vars=(),
    )


def test_step_size_initialization_refuses_an_improper_target() -> None:
    kernel = NUTSKernel(max_tree_depth=3, adaptation="windowed")
    driver = MCMC(kernel, num_warmup=50, num_samples=5, num_chains=1, seed=0)
    with pytest.raises(RuntimeError, match="posterior is likely improper"):
        driver.run(_flat_program(), torch.zeros(1, 1), {})
