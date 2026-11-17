"""Posterior and predictive model averaging, as brms posterior_average and pp_average.

Oracles: brms's largest-remainder allocation (``round_largest_remainder``
in brms ``R/misc.R``), the fits' own draws and predictions at the selected
draws, `model_weights` and `post_prob` for the weights, and the brms
summary of the pooled draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import PriorSpec, bf, fit, model_weights, set_rescor
from quivers.formulas._evidence import post_prob
from quivers.formulas._model_averaging import (
    averaging_weights,
    posterior_average,
    pp_average,
    round_largest_remainder,
)
from quivers.diagnostics.posterior_tables import posterior_summary
from quivers.inference.mcmc.driver import MCMCResult

Y = np.array([0.42, -0.31, 1.18, 0.77, -0.05, 0.93, 1.41, 0.12, 0.58, -0.62])
X = np.array([0.5, -1.1, 0.3, 1.7, -0.2, 0.9, 1.4, -0.6, 1.2, -0.8])
FRAME = pd.DataFrame({"y": Y, "x": X})
PRIORS = [PriorSpec(distribution="Normal(0.0, 1.0)", class_="Intercept")]


@pytest.fixture(scope="module")
def fits():
    full = fit("y ~ x", data=FRAME, method="svi", num_steps=30, seed=1)
    reduced = fit("y ~ 1", data=FRAME, method="svi", num_steps=30, seed=2)
    return {"full": full, "reduced": reduced}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([33.3, 33.3, 33.4], [33, 33, 34]),
        ([2.5, 2.5], [3, 2]),
        ([0.2 * 7, 0.8 * 7], [1, 6]),
        ([100.0, 0.0], [100, 0]),
    ],
)
def test_round_largest_remainder_matches_brms(values, expected):
    assert round_largest_remainder(values) == expected


def test_posterior_average_mixes_selected_draws(fits):
    result = posterior_average(fits, weights=[0.25, 0.75], ndraws=80, seed=3)
    assert result.weights == pytest.approx({"full": 0.25, "reduced": 0.75})
    assert result.ndraws == {"full": 20, "reduced": 60}
    # Only variables common to both models enter by default.
    assert set(result.draws) == set(fits["full"].variables()) & set(
        fits["reduced"].variables()
    )
    assert "b_x" not in result.draws
    for name, start, stop in (("full", 0, 20), ("reduced", 20, 80)):
        pooled = fits[name].variables()["b_Intercept"].reshape(-1).double()
        part = result.draws["b_Intercept"][start:stop]
        # Every averaged draw is a distinct draw of its model.
        matches = (part[:, None] == pooled[None, :]).sum(1)
        assert torch.all(matches >= 1)
        assert len(set(part.tolist())) == part.numel()
    frame = result.to_frame()
    assert list(frame.columns) == list(result.draws) and len(frame) == 80


def test_posterior_average_fills_missing_variables(fits):
    result = posterior_average(
        fits, variable="b_x", weights=[0.5, 0.5], ndraws=40, missing=0.0, seed=1
    )
    assert torch.all(result.draws["b_x"][20:] == 0.0)
    assert torch.all(result.draws["b_x"][:20] != 0.0)
    with pytest.raises(ValueError, match="not in all models"):
        posterior_average(fits, variable="b_x", weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="no value"):
        posterior_average(fits, variable="b_x", weights=[0.5, 0.5], missing={})
    with pytest.raises(ValueError, match="none of the models"):
        posterior_average(fits, variable="b_z", weights=[0.5, 0.5], missing=0.0)
    with pytest.raises(ValueError, match="only"):
        posterior_average(fits, weights=[1.0, 0.0], ndraws=1000)


def test_weights_follow_model_weights_and_validate_explicit_weights(fits):
    assert averaging_weights(fits, "waic", seed=0) == pytest.approx(
        model_weights(fits, weights="waic")
    )
    assert averaging_weights(fits, [1.0, 3.0], seed=0) == pytest.approx(
        {"full": 0.25, "reduced": 0.75}
    )
    with pytest.raises(ValueError, match="one value per model"):
        averaging_weights(fits, [1.0], seed=0)
    with pytest.raises(ValueError, match="nonnegative"):
        averaging_weights(fits, [1.0, -1.0], seed=0)
    with pytest.raises(ValueError, match="weights must be"):
        averaging_weights(fits, "marglik", seed=0)  # type: ignore[arg-type]


def test_bma_weights_are_posterior_model_probabilities():
    generator = torch.Generator().manual_seed(4)
    models = {}
    for name, design in (("full", "y ~ x"), ("reduced", "y ~ 1")):
        base = fit(
            design,
            data=FRAME,
            fixed_parameters={"sigma": 1.0},
            priors=PRIORS
            + (
                [PriorSpec(distribution="Normal(0.0, 1.0)", class_="b")]
                if design == "y ~ x"
                else []
            ),
            method="svi",
            num_steps=1,
            seed=1,
        )
        matrix = np.column_stack([np.ones(Y.size), X][: 2 if design == "y ~ x" else 1])
        precision = np.eye(matrix.shape[1]) + matrix.T @ matrix
        mean = np.linalg.solve(precision, matrix.T @ Y)
        chol = np.linalg.cholesky(np.linalg.inv(precision))
        z = torch.randn(
            2, 400, matrix.shape[1], generator=generator, dtype=torch.float64
        )
        draws = torch.from_numpy(mean) + z @ torch.from_numpy(chol).T
        names = ["intercept", "beta_x"][: matrix.shape[1]]
        posterior = MCMCResult(
            samples={n: draws[..., i].float() for i, n in enumerate(names)},
            log_densities=torch.zeros(2, 400),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2),
            num_warmup=0,
            num_samples=400,
        )
        models[name] = base.with_(posterior=posterior)
    weights = model_weights(models, weights="bma", seed=5)
    assert weights == pytest.approx(post_prob(models, seed=5))
    averaged = posterior_average(models, weights="bma", ndraws=100, seed=5)
    assert averaged.weights == pytest.approx(weights)
    assert sum(averaged.ndraws.values()) == 100


@pytest.mark.parametrize("method", ["posterior_epred", "posterior_linpred"])
def test_pp_average_pools_predictions_at_selected_draws(fits, method):
    new = pd.DataFrame({"x": [-1.0, 0.0, 2.0]})
    result = pp_average(
        fits,
        weights=[0.4, 0.6],
        method=method,
        ndraws=50,
        new_data=new,
        seed=7,
    )
    assert result.ndraws == {"full": 20, "reduced": 30}
    assert result.draws.shape == (50, 3)
    generator = torch.Generator().manual_seed(7)
    start = 0
    for name in ("full", "reduced"):
        count = result.ndraws[name]
        ids = sorted(torch.randperm(100, generator=generator)[:count].tolist())
        reference = getattr(fits[name], method)(new_data=new, seed=7)
        pooled = reference.reshape(-1, 3).double()[ids]
        torch.testing.assert_close(result.draws[start : start + count], pooled)
        start += count
    expected = posterior_summary(result.draws.unsqueeze(0))
    assert result.summary is not None
    torch.testing.assert_close(result.summary.values, expected.values)


def test_pp_average_predict_and_refusals(fits):
    result = pp_average(fits, weights=[0.5, 0.5], ndraws=20, summary=False, seed=2)
    assert result.summary is None and result.draws.shape == (20, Y.size)
    other = fit("y ~ 1", data=FRAME.assign(y=Y + 1), method="svi", num_steps=2)
    with pytest.raises(ValueError, match="different response data"):
        pp_average({"a": fits["full"], "b": other}, weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="ndraws instead"):
        pp_average(fits, weights=[0.5, 0.5], num_samples=3)
    with pytest.raises(ValueError, match="joint fits"):
        pp_average(fits, weights=[0.5, 0.5], resp="y")


def test_pp_average_of_joint_fits_selects_response():
    frame = FRAME.assign(w=Y[::-1])
    spec = bf("y ~ x") + bf("w ~ 1") + set_rescor(False)
    first = fit(spec, data=frame, method="svi", num_steps=5, seed=1)
    second = fit(
        bf("y ~ 1") + bf("w ~ 1") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=5,
        seed=2,
    )
    models = {"first": first, "second": second}
    result = pp_average(
        models, weights=[0.5, 0.5], method="posterior_epred", ndraws=10, resp="w"
    )
    assert result.draws.shape == (10, Y.size)
    with pytest.raises(ValueError, match="requires resp"):
        pp_average(models, weights=[0.5, 0.5], ndraws=10)


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_posterior_average_of_sampler_fits(method):
    sampled = {
        name: fit(
            design,
            data=FRAME,
            method=method,
            num_warmup=20,
            num_samples=20,
            num_chains=1,
            seed=3,
        )
        for name, design in (("a", "y ~ x"), ("b", "y ~ 1"))
    }
    result = posterior_average(sampled, weights=[0.5, 0.5], seed=1)
    assert sum(result.ndraws.values()) == 20
    assert result.draws["b_Intercept"].shape == (20,)
