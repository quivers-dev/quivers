"""Runtime references and independent diagnostic caches in MCMC results."""

import copy

import didactic.api as dx
import pytest
import torch

from quivers.diagnostics import rhat
from quivers.inference.mcmc import MCMCResult


def test_result_updates_recompute_diagnostics_and_preserve_tensor_references() -> None:
    """Updating draws creates a fresh cache without copying runtime tensors."""
    generator = torch.Generator().manual_seed(7)
    draws = torch.randn(4, 40, generator=generator)
    result = MCMCResult(
        samples={"mu": draws},
        log_densities=torch.zeros(4, 40),
        acceptance_rates=torch.ones(4),
        divergence_counts=torch.zeros(4, dtype=torch.int64),
        num_warmup=10,
        num_samples=40,
    )
    original = result.r_hat["mu"]
    shifted = draws.clone()
    shifted[0] += 3.0
    samples = {"mu": shifted}
    updated = result.with_(samples=samples)
    assert updated.samples is samples
    assert updated.log_densities is result.log_densities
    assert updated.r_hat is not result.r_hat
    torch.testing.assert_close(updated.r_hat["mu"], rhat(shifted), rtol=0, atol=0)
    assert not torch.allclose(updated.r_hat["mu"], original)
    assert copy.copy(result) == result
    assert hash(copy.copy(result)) == hash(result)
    assert updated != result
    with pytest.raises(dx.ValidationError, match="map names to tensors"):
        result.with_(samples={"mu": 3})
    with pytest.raises(dx.ValidationError, match="torch.Tensor"):
        result.with_(num_draws=(40, 40, 40, 40))
