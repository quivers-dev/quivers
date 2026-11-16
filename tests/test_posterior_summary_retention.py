"""Nested convergence records retain their opaque numerical tensors."""

from __future__ import annotations

import copy
import pickle
from unittest.mock import patch

import pytest
import torch

from quivers.diagnostics import PosteriorAnalysis, PosteriorSummary
from quivers.diagnostics.convergence import summary


def test_nested_convergence_retains_the_original_record_and_tensor_identities():
    values = torch.randn(2, 16, 2, generator=torch.Generator().manual_seed(8))
    expected = summary({"x": values})["x"]
    analysis = PosteriorAnalysis(samples={"x": values}, log_likelihood={}, kind="mcmc")
    with patch(
        "quivers.diagnostics.posterior.convergence_summary",
        return_value={"x": expected},
    ):
        result = analysis.summary()["x"]
    assert result.convergence is expected
    assert result.convergence.mean is expected.mean
    assert result.convergence.r_hat is expected.r_hat
    assert result.convergence.quantiles is expected.quantiles
    assert result.mean is expected.mean
    assert result.quantiles is expected.quantiles


@pytest.mark.parametrize(
    "operation", [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
def test_nested_convergence_roundtrips_without_losing_any_tensor_field(operation):
    values = torch.arange(64.0, dtype=torch.float64).reshape(2, 16, 2)
    result = PosteriorAnalysis(
        samples={"x": values}, log_likelihood={}, kind="mcmc"
    ).summary()["x"]
    restored = operation(result)
    assert isinstance(restored, PosteriorSummary)
    assert restored.convergence is not None
    for field in (
        "mean",
        "sd",
        "quantiles",
        "r_hat",
        "ess_bulk",
        "ess_tail",
        "mcse_mean",
        "mcse_sd",
        "mcse_quantiles",
    ):
        torch.testing.assert_close(
            getattr(restored.convergence, field),
            getattr(result.convergence, field),
            equal_nan=True,
        )
    assert restored.probs == result.probs


def test_variational_summary_preserves_absence_of_chain_diagnostics():
    result = PosteriorAnalysis(
        samples={"x": torch.tensor([[1.0, 2.0, 3.0]])},
        log_likelihood={},
        kind="variational",
    ).summary()["x"]
    assert result.convergence is None
    assert pickle.loads(pickle.dumps(result)).convergence is None
