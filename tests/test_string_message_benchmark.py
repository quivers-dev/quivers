"""Check the projection counts and beam support against independent observations."""

from dataclasses import replace
from math import exp

import pytest
import torch

from quivers.core.relations import (
    FiniteStringBeliefPlan,
    regular_ep,
    regular_projection,
)
from tools import benchmark_string_messages as benchmark


def test_ep_benchmark_counts_every_observed_projection_attempt_and_moment(monkeypatch):
    graph, bases, _ = benchmark._fixture()
    actual = []
    fits = []
    moments = []
    infer = benchmark.regular_string_ep
    project = regular_ep._projection
    evaluate = regular_projection._MomentPlan.__call__

    def observed_inference(*args, **kwargs):
        result = infer(*args, **kwargs)
        actual.append(result)
        return result

    def observed_fit(*args, **kwargs):
        fits.append(1)
        return project(*args, **kwargs)

    def observed_moment(*args, **kwargs):
        moments.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(benchmark, "regular_string_ep", observed_inference)
    monkeypatch.setattr(regular_ep, "_projection", observed_fit)
    monkeypatch.setattr(regular_projection._MomentPlan, "__call__", observed_moment)

    def check(report):
        assert report.fit_attempts == len(fits), (
            "reported fit count missed the actual schedule"
        )
        assert report.moment_evaluations == len(moments)

    observed = benchmark._projected(graph, bases, 0.1)
    check(observed)
    assert observed.sweeps > 0
    assert observed.fit_attempts > len(actual[0].updates)
    assert observed.moment_evaluations >= 2 * observed.fit_attempts

    def false_sweep(*_args, **_kwargs):
        saved = actual[0]
        return replace(saved, sweeps=saved.sweeps + 1)

    monkeypatch.setattr(benchmark, "regular_string_ep", false_sweep)
    with pytest.raises(AssertionError, match="actual schedule"):
        check(benchmark._projected(graph, bases, 0.1))


def _beam_oracle():
    # Enumerate the three retained incoming words for each direction.
    # The surviving joint support on each marginal is exactly 00, 01.
    x0 = exp(0.8) * (exp(1.3) + 2 * exp(0.7))
    x1 = exp(-0.2) * (exp(-0.1) + exp(1.8) + exp(0.9))
    y0 = exp(-0.3) * (exp(2.4) + 1 + exp(0.8))
    y1 = exp(0.7) * (exp(0.8) + exp(0.9) + exp(0.7))
    return torch.tensor(
        [
            [x0 / (x0 + x1), x1 / (x0 + x1), 0, 0],
            [y0 / (y0 + y1), y1 / (y0 + y1), 0, 0],
        ],
        dtype=torch.float64,
    )


def test_production_beam_counts_complete_passes_and_retains_the_oracle_support(
    monkeypatch,
):
    graph, _, exact = benchmark._fixture()
    passes = []
    proposals = FiniteStringBeliefPlan._pruned_proposals

    def observed_pass(*args, **kwargs):
        passes.append(1)
        return proposals(*args, **kwargs)

    monkeypatch.setattr(FiniteStringBeliefPlan, "_pruned_proposals", observed_pass)
    report = benchmark._beam(graph, 3)
    assert report.status == "converged (truncated)"
    assert report.beam_reduction_cells == (2 * 16 + 4 + 4) * len(passes)
    torch.testing.assert_close(report.probabilities, _beam_oracle(), rtol=0, atol=1e-14)
    torch.testing.assert_close(
        benchmark._beam(graph, 4).probabilities, exact, rtol=0, atol=1e-14
    )
    infer = FiniteStringBeliefPlan.run_pruned

    def widened(plan, **kwargs):
        return infer(plan, **(kwargs | {"beam": 4}))

    monkeypatch.setattr(FiniteStringBeliefPlan, "run_pruned", widened)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            benchmark._beam(graph, 3).probabilities, _beam_oracle()
        )
