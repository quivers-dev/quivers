"""Refit predictive units agree with independent exact Gaussian calculations."""

import math
from didactic.fields import ValidationError

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from quivers.diagnostics.cross_validation import (
    FoldPlan,
    FoldPrediction,
    cross_validate,
    fold_plan,
)


@pytest.mark.parametrize("size,folds", [(7, 3), (12, 4), (10000, 10000)])
def test_seeded_partition_is_balanced_reusable_and_preserves_caller_rng(size, folds):
    state = torch.get_rng_state().clone()
    first = fold_plan(size, folds=folds, seed=37)
    assert first == fold_plan(size, folds=folds, seed=37)
    torch.testing.assert_close(torch.get_rng_state(), state)
    counts = np.bincount(first.assignments)
    assert len(counts) == folds and counts.max() - counts.min() <= 1


def test_grouped_stratified_and_omission_partitions_preserve_original_rows():
    groups = ("a", "a", "b", "c", "c", "d", "e", "e")
    strata = (0, 0, 0, 1, 1, 1, 0, 0)
    plan = fold_plan(8, folds=3, groups=groups, strata=strata, seed=4)
    for group in set(groups):
        assert (
            len(
                {
                    plan.assignments[i]
                    for i, value in enumerate(groups)
                    if value == group
                }
            )
            == 1
        )
    for fold in range(3):
        split = plan.split(fold)
        assert not set(split.train) & set(split.test)
        assert sorted((*split.train, *split.test)) == list(range(8))
    loo_group = FoldPlan(assignments=tuple(range(8)), omission_groups=groups)
    split = loo_group.split(3)
    assert split.test == (3,) and split.omitted == (3, 4)
    assert split.train == (0, 1, 2, 5, 6, 7)
    with pytest.raises(ValueError, match="one stratum"):
        fold_plan(8, folds=2, groups=groups, strata=tuple(range(8)))


@pytest.mark.parametrize("assignments", [(0,), (0, 0), (0, 2), (True, 1), (-1, 0)])
def test_invalid_explicit_partitions_fail_before_refitting(assignments):
    with pytest.raises(ValidationError):
        FoldPlan(assignments=assignments)


def _normal_refit(data, calls):
    # Deterministic equal-probability quadrature approximates the analytic
    # conjugate posterior without relying on the implementation's sampler.
    quantiles = (torch.arange(16384, dtype=torch.float64) + 0.5) / 16384
    normal = torch.distributions.Normal(0.0, 1.0)
    standardized = normal.icdf(quantiles)

    def refit(split):
        calls.append(split)
        training = data[list(split.train)]
        variance = 1 / (1 + len(training))
        mean = variance * training.sum()
        theta = mean + math.sqrt(variance) * standardized
        scores = torch.distributions.Normal(theta[:, None], 1.0).log_prob(
            data[list(split.test)]
        )
        return FoldPrediction(
            log_likelihood={"y": scores.unsqueeze(0)},
            fit={"mean": mean, "variance": variance},
        )

    return refit


@pytest.mark.parametrize("joint", ["observation", "fold", "group"])
def test_normal_refits_match_closed_predictive_density_for_every_unit(joint):
    data = torch.tensor([-0.6, 0.2, 0.8, -0.3, 1.0, 0.4], dtype=torch.float64)
    plan = FoldPlan(assignments=(0, 0, 1, 1, 2, 2), seed=9)
    calls = []
    result = cross_validate(
        plan,
        _normal_refit(data, calls),
        joint=joint,
        groups=("a", "a", "b", "c", "d", "d") if joint == "group" else None,
        save_fits=True,
    )
    expected = []
    for fold, rows in result.units:
        train = data[list(plan.split(fold).train)]
        variance = 1 / (1 + len(train))
        mean = variance * train.sum()
        width = len(rows)
        law = torch.distributions.MultivariateNormal(
            mean.expand(width),
            torch.eye(width, dtype=torch.float64)
            + variance * torch.ones(width, width, dtype=torch.float64),
        )
        expected.append(law.log_prob(data[list(rows)]).item())
    np.testing.assert_allclose(
        result.log_predictive_density.numpy(), expected, atol=7e-5, rtol=0
    )
    assert result.complete and len(calls) == 3 and set(result.fits) == {0, 1, 2}
    assert result.elpd.item() == pytest.approx(sum(expected), abs=2e-4)
    assert result.se.item() == pytest.approx(
        np.std(expected, ddof=1) * math.sqrt(len(expected)), abs=2e-4
    )


def test_joint_responses_and_fold_scores_sum_within_draw_before_averaging():
    first = np.array([[[-1.0, -8.0], [-6.0, -2.0]]])
    second = np.array([[[-2.0, -5.0], [-1.0, -3.0]]])

    def refit(split):
        return FoldPrediction(
            log_likelihood={"a": torch.tensor(first), "b": torch.tensor(second)}
        )

    plan = FoldPlan(assignments=(0, 0, 1, 1))
    result = cross_validate(plan, refit, responses=("a", "b"), joint="fold")
    expected = logsumexp((first + second).sum(-1).reshape(-1)) - math.log(2)
    np.testing.assert_allclose(
        result.log_predictive_density.numpy(), [expected, expected]
    )
    wrong = (logsumexp(first.reshape(2, 2), axis=0) - math.log(2)).sum() + (
        logsumexp(second.reshape(2, 2), axis=0) - math.log(2)
    ).sum()
    assert abs(expected - wrong) > 0.2


def test_partial_folds_keep_ids_independent_streams_and_zero_density_units():
    plan = FoldPlan(assignments=(0, 1, 2, 0, 1, 2), seed=12)
    calls = []

    def refit(split):
        calls.append((split.fold, torch.rand(1).item()))
        return FoldPrediction(
            log_likelihood={"y": torch.full((2, 3, len(split.test)), -float("inf"))}
        )

    state = torch.get_rng_state().clone()
    result = cross_validate(plan, refit, fold_ids=(2, 0))
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert not result.complete and result.fold_ids == (2, 0)
    assert result.units == ((2, (2,)), (2, (5,)), (0, (0,)), (0, (3,)))
    assert result.elpd.item() == -float("inf") and torch.isnan(result.se)
    previous = dict(calls)
    calls.clear()
    cross_validate(plan, refit, fold_ids=(0, 2))
    assert dict(calls) == previous


def test_refitter_error_restores_rng_and_does_not_run_later_folds():
    state = torch.get_rng_state().clone()
    calls = []

    def refit(split):
        calls.append(split.fold)
        torch.rand(3)
        raise RuntimeError("fitter failed")

    with pytest.raises(RuntimeError, match="fitter failed"):
        cross_validate(FoldPlan(assignments=(0, 1)), refit)
    assert calls == [0]
    torch.testing.assert_close(torch.get_rng_state(), state)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 2, 2, 1),
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 2, dtype=torch.int64),
        torch.full((1, 2, 2), float("nan")),
        torch.full((1, 2, 2), float("inf")),
    ],
)
def test_invalid_score_contracts_are_rejected_without_silent_reshaping(bad):
    with pytest.raises((ValueError, TypeError)):
        cross_validate(
            FoldPlan(assignments=(0, 0, 1, 1)),
            lambda split: FoldPrediction(log_likelihood={"y": bad}),
        )


def test_response_selection_and_retained_fit_requirements_are_explicit():
    plan = FoldPlan(assignments=(0, 1))

    def refit(split):
        return FoldPrediction(
            log_likelihood={"a": torch.zeros(1, 2, 1), "b": torch.zeros(1, 2, 1)}
        )

    with pytest.raises(ValueError, match="select responses"):
        cross_validate(plan, refit)
    with pytest.raises(ValueError, match="actual fitted"):
        cross_validate(plan, refit, responses=("a",), save_fits=True)
    for kwargs in [
        {"fold_ids": ()},
        {"fold_ids": (0, 0)},
        {"fold_ids": (2,)},
        {"joint": "group"},
        {"responses": ()},
    ]:
        with pytest.raises(ValueError):
            cross_validate(plan, refit, **kwargs)


def test_result_units_and_scores_are_validated_when_constructed_directly():
    from quivers.diagnostics import CrossValidationResult

    valid = dict(
        plan=FoldPlan(assignments=(0, 1)),
        fold_ids=(0, 1),
        responses=("y",),
        joint="observation",
        units=((0, (0,)), (1, (1,))),
        log_predictive_density=torch.tensor([-2.0, -3.0], dtype=torch.float64),
    )
    result = CrossValidationResult(**valid)
    assert result.complete and result.elpd.item() == -5
    for changes in [
        {"units": ((0, (0,)), (0, (0,)))},
        {"units": ((0, (1,)), (1, (0,)))},
        {"fold_ids": (0,)},
        {"responses": ("y", "y")},
        {"log_predictive_density": torch.tensor([-2.0, float("inf")])},
        {"fits": {2: object()}},
    ]:
        with pytest.raises(ValidationError):
            CrossValidationResult(**(valid | changes))


def test_descriptive_score_se_uses_scaled_arithmetic_at_extreme_finite_values():
    values = torch.tensor([-1e300, 1e300], dtype=torch.float64)
    result = cross_validate(
        FoldPlan(assignments=(0, 1)),
        lambda split: FoldPrediction(
            log_likelihood={"y": values[split.fold].reshape(1, 1, 1)}
        ),
    )
    assert torch.isfinite(result.se) and result.se.item() == pytest.approx(2e300)
