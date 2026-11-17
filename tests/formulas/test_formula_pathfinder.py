"""brms ``algorithm = "pathfinder"`` through `fit` and the model interface."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import BayesianFit, PathfinderOptions, bf, fit
from quivers.formulas._multivariate import MultivariateFormula
from quivers.formulas._fit_controls import resolve_algorithm
from quivers.inference import PathfinderDraws
from quivers.modeling import Model, ModelFit, PathfinderMethod, load_fit, save_fit
from quivers.inference import Pathfinder

ROWS = np.arange(24)
FRAME = pd.DataFrame(
    {
        "x": np.sin(1.3 * ROWS),
        "z": np.cos(0.7 * ROWS),
    }
)
FRAME["y"] = 0.8 + 1.4 * FRAME.x + 0.3 * np.sin(5.1 * ROWS)
FRAME["y2"] = -0.4 + 0.6 * FRAME.z + 0.2 * np.cos(3.3 * ROWS)
PRIOR_SD = 2.0
PRIORS = {
    "intercept": f"Normal(0.0, {PRIOR_SD})",
    "beta_x": f"Normal(0.0, {PRIOR_SD})",
}
SMALL = {"draws": 200, "single_path_draws": 100, "num_elbo_draws": 10}


def _conjugate_posterior() -> tuple[np.ndarray, np.ndarray]:
    """Closed-form posterior of (intercept, beta_x) with sigma fixed at one."""
    design = np.column_stack([np.ones(len(FRAME)), FRAME.x.to_numpy()])
    precision = design.T @ design + np.eye(2) / PRIOR_SD**2
    covariance = np.linalg.inv(precision)
    return covariance @ design.T @ FRAME.y.to_numpy(), covariance


def test_algorithm_pathfinder_selects_the_pathfinder_method():
    assert resolve_algorithm("pathfinder", "nuts", None) == ("pathfinder", None)
    assert resolve_algorithm(None, "pathfinder", None) == ("pathfinder", None)


def test_conjugate_posterior_draws_match_the_closed_form():
    mean, covariance = _conjugate_posterior()
    fitted = fit(
        "y ~ x",
        data=FRAME,
        priors=PRIORS,
        fixed_parameters={"sigma": 1.0},
        algorithm="pathfinder",
        num_chains=4,
        pathfinder={"draws": 1000, "single_path_draws": 400, "num_elbo_draws": 10},
        seed=7,
    )
    posterior = fitted.posterior
    assert isinstance(posterior, PathfinderDraws)
    assert posterior.num_paths == 4 and posterior.num_samples == 1000
    draws = np.column_stack(
        [
            fitted.draws()["intercept"].reshape(-1).double().numpy(),
            fitted.draws()["beta_x"].reshape(-1).double().numpy(),
        ]
    )
    # Fixed in advance: resampled draws have an effective size of at least
    # 250 here; means within five standard errors, variances within 40%.
    sd = np.sqrt(np.diag(covariance))
    assert np.all(np.abs(draws.mean(0) - mean) < 5 * sd / math.sqrt(250))
    assert np.all(np.abs(draws.var(0) / np.diag(covariance) - 1) < 0.4)
    # The posterior is Gaussian, so every path's selected approximation is
    # close to it and the importance ratios are nearly constant.
    assert posterior.pareto_k < 0.7


def test_pathfinder_fit_predicts_refits_and_round_trips(tmp_path):
    fitted = fit(
        "y ~ x + z",
        data=FRAME,
        algorithm="pathfinder",
        num_chains=2,
        pathfinder=SMALL,
        seed=3,
    )
    new = FRAME.iloc[:5]
    assert fitted.posterior_linpred().shape == (1, 200, 24)
    assert fitted.posterior_epred(new_data=new).shape == (1, 200, 5)
    assert fitted.posterior_predict(new_data=new, seed=1).shape == (1, 200, 5)
    assert next(iter(fitted.log_likelihood().values())).shape == (1, 200, 24)
    beta = fitted.draws()["beta_x"][0]
    linpred = fitted.posterior_linpred(new_data=new)[0]
    expected = (
        fitted.draws()["intercept"][0][:, None]
        + beta[:, None] * torch.tensor(new.x.to_numpy(), dtype=beta.dtype)
        + fitted.draws()["beta_z"][0][:, None]
        * torch.tensor(new.z.to_numpy(), dtype=beta.dtype)
    )
    assert torch.allclose(linpred, expected, atol=1e-5)
    assert fitted.summary().algorithm == "pathfinder"
    refit = fitted.update(data=FRAME.iloc[:20])
    assert isinstance(refit.posterior, PathfinderDraws)
    assert refit.posterior.num_samples == 200
    path = tmp_path / "pathfinder.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    assert isinstance(restored.posterior, PathfinderDraws)
    assert isinstance(fitted.posterior, PathfinderDraws)
    for name, value in fitted.draws().items():
        assert torch.equal(restored.draws()[name], value)
    assert torch.equal(restored.posterior.log_approx, fitted.posterior.log_approx)
    assert torch.allclose(
        restored.posterior_epred(new_data=new), fitted.posterior_epred(new_data=new)
    )


def test_kfold_refits_each_fold_with_pathfinder():
    fitted = fit(
        "y ~ x",
        data=FRAME,
        algorithm="pathfinder",
        num_chains=2,
        pathfinder=SMALL,
        seed=3,
    )
    result = fitted.kfold(K=3, seed=1)
    assert math.isfinite(result.estimate.elpd)


def test_switching_methods_by_update_resets_the_other_settings():
    fitted = fit("y ~ x", data=FRAME, method="svi", num_steps=20, seed=1)
    switched = fitted.update(method="pathfinder", num_chains=1, pathfinder=SMALL)
    assert isinstance(switched.posterior, PathfinderDraws)
    assert switched.posterior.num_samples == 100
    back = switched.update(method="svi", num_steps=5, num_chains=None)
    assert not isinstance(back.posterior, PathfinderDraws)


def test_joint_formulas_fit_with_pathfinder():
    fitted = fit(
        cast(MultivariateFormula, bf("y ~ x") + bf("y2 ~ z")),
        data=FRAME,
        algorithm="pathfinder",
        num_chains=2,
        pathfinder=SMALL,
    )
    assert isinstance(fitted.posterior, PathfinderDraws)
    assert fitted.posterior_epred(new_data=FRAME.iloc[:3])["y"].shape == (1, 200, 3)


def test_model_interface_runs_pathfinder_and_round_trips(tmp_path):
    source = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""
    x = torch.tensor(FRAME.x.to_numpy(), dtype=torch.float32)
    y = 0.7 * x + torch.tensor(0.3 * np.sin(5.1 * ROWS), dtype=torch.float32)
    method = PathfinderMethod(
        pathfinder=Pathfinder(num_paths=2, num_draws=300, num_psis_draws=600)
    )
    fitted = Model(source).fit({"x": x, "y": y}, method=method)
    assert isinstance(fitted.posterior, PathfinderDraws)
    precision = float(x @ x) + 1.0
    mean, sd = float(x @ y) / precision, precision**-0.5
    draws = fitted.posterior.samples["b"].reshape(-1).double()
    assert abs(float(draws.mean()) - mean) < 5 * sd / math.sqrt(300)
    path = tmp_path / "model.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, ModelFit)
    assert isinstance(restored.posterior, PathfinderDraws)
    assert torch.equal(restored.posterior.samples["b"], fitted.posterior.samples["b"])


def test_pathfinder_arguments_are_checked():
    with pytest.raises(ValueError, match="unknown pathfinder arguments"):
        fit("y ~ x", data=FRAME, algorithm="pathfinder", pathfinder={"iter": 3})
    with pytest.raises(ValueError, match="num_samples"):
        fit("y ~ x", data=FRAME, algorithm="pathfinder", num_samples=10)
    with pytest.raises(ValueError, match="thin"):
        fit("y ~ x", data=FRAME, algorithm="pathfinder", thin=2)
    with pytest.raises(ValueError, match="apply to algorithm='pathfinder'"):
        fit("y ~ x", data=FRAME, method="svi", num_steps=2, pathfinder=SMALL)
    with pytest.raises(ValueError, match="MCMC strategy"):
        fit("y ~ x", data=FRAME, algorithm="pathfinder", init="jitter")
    assert PathfinderOptions(history_size=7).lbfgs().history_size == 7
