"""Precision requirements for regression parameters and recording runs."""

from pathlib import Path
import math

import pytest
import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from tests.test_gallery_sweep import _extract_try_it_blocks


@pytest.mark.parametrize("family", ["beta", "negbin"])
@pytest.mark.parametrize("eta", [-30.0, 30.0])
def test_gallery_density_at_saturated_predictors(family: str, eta: float) -> None:
    """Compare the compiled density with an independent parameterization.

    Parameters
    ----------
    family : str
        Gallery family whose declared parameters are scored.
    eta : float
        Finite predictor at which float32 can lose a positive complement.
    """
    model = load(Path(f"docs/examples/source/{family}_regression.qvr")).morphism
    dtype = torch.float64 if family == "negbin" else torch.float32
    intercept = torch.full((3,), eta, dtype=dtype)
    slope = torch.zeros(3, dtype=dtype)
    scale = torch.full((3,), 2.0, dtype=dtype)
    rows = torch.arange(300) % 3
    values = torch.full((300,), 0.5 if family == "beta" else 2.0, dtype=dtype)
    sites = {
        "beta_0": intercept,
        "beta_1": slope,
        "phi" if family == "beta" else "dispersion": scale,
        "x": torch.zeros(300, dtype=dtype),
        "out_idx": rows,
        "y": values,
    }
    actual = model.log_joint(torch.zeros(1, 1, dtype=dtype), sites).sum()
    predictor = torch.full((300,), eta, dtype=torch.float64)
    if family == "beta":
        likelihood = D.Beta(2 * predictor.sigmoid(), 2 * (-predictor).sigmoid())
        likelihood_score = likelihood.log_prob(values.double()).sum()
        prior = D.HalfCauchy(2.5).log_prob(scale.double()).sum()
    else:
        # Score the represented probability, whose rounding is visible near
        # one, using the scalar PMF rather than a distribution constructor.
        probability = math.exp(eta) / (2.0 + math.exp(eta))
        likelihood_score = torch.tensor(
            300
            * (
                math.lgamma(4.0)
                - math.lgamma(2.0)
                - math.lgamma(3.0)
                + 2 * math.log1p(-probability)
                + 2 * math.log(probability)
            ),
            dtype=torch.float64,
        )
        prior = D.Gamma(2.0, 0.5).log_prob(scale.double()).sum()
    expected = (
        D.Normal(0.0, 5.0).log_prob(intercept.double()).sum()
        + D.Normal(0.0, 5.0).log_prob(slope.double()).sum()
        + prior
        + likelihood_score
    )
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-4)


def test_negbin_probability_requires_double_precision_at_large_predictors() -> None:
    """The float32 probability is invalid; float64 retains its complement."""
    eta32 = torch.tensor(30.0, dtype=torch.float32)
    p32 = eta32.exp() / (2.0 + eta32.exp())
    assert p32 == 1
    with pytest.raises(ValueError):
        D.NegativeBinomial(2.0, probs=p32, validate_args=True)
    eta64 = eta32.double()
    # Compute the complement directly instead of subtracting it from one.
    expected_complement = 2.0 / (2.0 + math.exp(30.0))
    p64 = eta64.exp() / (2.0 + eta64.exp())
    assert p64 < 1
    assert abs(float(1 - p64) - expected_complement) < torch.finfo(torch.float64).eps


def test_negbin_recording_run_retains_original_data_in_double_precision() -> None:
    """Float64 records the documented prior stream without changing the data."""
    blocks = _extract_try_it_blocks(
        Path("docs/examples/negbin-regression.md").read_text()
    )
    previous = torch.get_default_dtype()
    rng_state = torch.random.get_rng_state()
    namespace = {}
    try:
        torch.set_default_dtype(torch.float32)
        exec(compile(blocks[0], "negbin-regression.md::data", "exec"), namespace)
        mc = MCMC(NUTSKernel(), num_warmup=0, num_samples=1, num_chains=1, seed=2)
        with pytest.raises(ValueError, match="probs"):
            mc._setup(
                namespace["model"],
                namespace["x_in"],
                namespace["observations"],
                "test",
                2,
            )
        torch.set_default_dtype(torch.float64)
        observations = {
            name: value.double() if value.is_floating_point() else value
            for name, value in namespace["observations"].items()
        }
        for name, value in observations.items():
            assert torch.equal(
                value.to(namespace["observations"][name].dtype),
                namespace["observations"][name],
            )
        setup = mc._setup(
            namespace["model"], namespace["x_in"].double(), observations, "test", 2
        )
        setup.potential.check_initial(
            torch.zeros(setup.registry.total_unconstrained_dim)
        )
    finally:
        torch.set_default_dtype(previous)
        torch.random.set_rng_state(rng_state)
