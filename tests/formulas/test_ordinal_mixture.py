"""Mixtures of ordinal families, brms ``mixture(cumulative, sratio, ...)``.

Oracles: category probabilities of the cumulative, stopping-ratio,
continuation-ratio and adjacent-category models written here in torch from
their definitions (values, first and second derivatives), and brms 2.20.4
``log_lik``, ``posterior_epred`` and ``pp_mixture`` at brms's own draws
(``oracles/brms_ordinal_mixture.R``), for separate thresholds
(``order = "none"``) and thresholds fixed across components
(``order = "mu"``).

Tolerances were fixed before comparing. The torch reference uses the same
float64 parameters over float32 design columns and agrees to 1e-9. brms
stores draws to six significant digits and its simplex ``theta`` is
renormalized here, so values computed from those draws agree to 1e-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import bf, families, fit, formula_to_qvr, mixture
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference._response import response_distribution
from quivers.inference.trace import trace
from quivers.modeling.artifact import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_ordinal_mixture.json").read_text()
)
TOL = 1e-9
BRMS_TOL = 1e-5
CATEGORIES = 4


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(ORACLE["data"])
    frame["y"] = pd.Categorical(frame["y"], categories=[1, 2, 3, 4], ordered=True)
    return frame


def _codes() -> torch.Tensor:
    return torch.as_tensor(np.asarray(ORACLE["data"]["y"]) - 1, dtype=torch.long)


def _column(name: str) -> torch.Tensor:
    return torch.tensor(_frame()[name].to_numpy(), dtype=torch.float32).double()


def _latent(fitted) -> tuple[str, ...]:
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names


def _pin(fitted, values):
    assert set(values) == set(_latent(fitted))
    samples = {
        name: torch.as_tensor(np.asarray(value), dtype=torch.float64).unsqueeze(0)
        for name, value in values.items()
    }
    count = next(iter(samples.values())).shape[1]
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, count),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _cdf(link: str, value: torch.Tensor) -> torch.Tensor:
    if link == "probit":
        return torch.distributions.Normal(0.0, 1.0).cdf(value)
    return torch.sigmoid(value)


def _probabilities(
    kind: str, thresholds: torch.Tensor, mu: torch.Tensor, link: str = "logit"
) -> torch.Tensor:
    """Category probabilities ``(row, category)`` from the model definitions.

    ``thresholds`` holds ``tau_1, ..., tau_{K-1}`` and ``mu`` the row
    predictors, in brms's convention ``tau_k - mu``.
    """
    delta = thresholds[None, :] - mu[:, None]
    ones = torch.ones_like(mu)[:, None]
    if kind == "cumulative":
        cumulative = torch.cat((_cdf(link, delta), ones), dim=-1)
        return torch.diff(cumulative, dim=-1, prepend=torch.zeros_like(ones))
    if kind == "sratio":
        # Stop at k with probability F(tau_k - mu) given k was reached.
        stop = _cdf(link, delta)
        reach = torch.cumprod(torch.cat((ones, 1 - stop), dim=-1), dim=-1)
        return reach * torch.cat((stop, ones), dim=-1)
    if kind == "cratio":
        # Continue past k with probability F(mu - tau_k).
        advance = _cdf(link, -delta)
        reach = torch.cumprod(torch.cat((ones, advance), dim=-1), dim=-1)
        return reach * torch.cat((1 - advance, ones), dim=-1)
    assert kind == "acat"
    scores = torch.cat((torch.zeros_like(ones), torch.cumsum(-delta, dim=-1)), dim=-1)
    return torch.softmax(scores, dim=-1)


def _centered(raw: torch.Tensor) -> torch.Tensor:
    cutpoints = torch.cat((raw.new_zeros(1), raw))
    return cutpoints - cutpoints.mean()


SITES = (
    "dpar_mu1_intercept",
    "dpar_mu1_beta_x",
    "dpar_mu1_cutpoint_log_spacing",
    "dpar_mu2_intercept",
    "dpar_mu2_beta_z",
    "dpar_mu2_cutpoint_offset",
    "theta",
)


def _unpack(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "dpar_mu1_intercept": theta[0],
        "dpar_mu1_beta_x": theta[1],
        "dpar_mu1_cutpoint_log_spacing": theta[2:4],
        "dpar_mu2_intercept": theta[4],
        "dpar_mu2_beta_z": theta[5],
        "dpar_mu2_cutpoint_offset": theta[6:8],
        "theta": torch.softmax(theta[8:10], dim=-1),
    }


def _component_probabilities(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-component category probabilities ``(component, row, category)``."""
    x, z = _column("x"), _column("z")
    first = _centered(torch.cumsum(values["dpar_mu1_cutpoint_log_spacing"].exp(), 0))
    second = _centered(values["dpar_mu2_cutpoint_offset"])
    return torch.stack(
        (
            _probabilities(
                "cumulative",
                first - values["dpar_mu1_intercept"],
                values["dpar_mu1_beta_x"] * x,
            ),
            _probabilities(
                "sratio",
                second - values["dpar_mu2_intercept"],
                values["dpar_mu2_beta_z"] * z,
            ),
        )
    )


def _oracle(theta: torch.Tensor) -> torch.Tensor:
    values = _unpack(theta)
    probabilities = _component_probabilities(values)
    mixed = (values["theta"][:, None, None] * probabilities).sum(0)
    return mixed.gather(-1, _codes()[:, None]).squeeze(-1).log()


def _model(fitted, theta: torch.Tensor) -> torch.Tensor:
    recorded = trace(
        fitted.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fitted.observations) | _unpack(theta),
    )
    return response_distribution(recorded, "y").log_prob(_codes()).reshape(-1)


@pytest.fixture(scope="module")
def separate():
    return fit(
        "y ~ x",
        data=_frame(),
        family=mixture("cumulative", "sratio"),
        distributional={"mu2": "~ z"},
        method="svi",
        num_steps=0,
    )


@pytest.fixture(scope="module")
def fixed():
    return fit(
        "y ~ x",
        data=_frame(),
        family=mixture("sratio", "cumulative", order="mu"),
        distributional={"mu2": "~ z"},
        method="svi",
        num_steps=0,
    )


POINT = torch.tensor(
    [0.3, -0.6, -0.2, 0.4, -0.5, 0.8, 0.7, 1.5, 0.2, -0.4], dtype=torch.float64
)


def test_density_and_derivatives_match_the_model_definitions(separate):
    assert set(_latent(separate)) == set(SITES)
    torch.testing.assert_close(
        _model(separate, POINT), _oracle(POINT), rtol=TOL, atol=TOL
    )
    point = POINT.clone().requires_grad_(True)
    (actual,) = torch.autograd.grad(_model(separate, point).sum(), point)
    (expected,) = torch.autograd.grad(_oracle(point).sum(), point)
    torch.testing.assert_close(actual, expected, rtol=TOL, atol=TOL)
    torch.testing.assert_close(
        torch.autograd.functional.hessian(
            lambda t: _model(separate, t).sum(), POINT.clone()
        ),
        torch.autograd.functional.hessian(lambda t: _oracle(t).sum(), POINT.clone()),
        rtol=1e-8,
        atol=1e-8,
    )
    # A wrong variant (both components cumulative) differs.
    values = _unpack(POINT)
    second = _centered(values["dpar_mu2_cutpoint_offset"])
    wrong = _probabilities(
        "cumulative",
        second - values["dpar_mu2_intercept"],
        values["dpar_mu2_beta_z"] * _column("z"),
    )
    assert not torch.allclose(wrong, _component_probabilities(values)[1], atol=1e-3)


def test_predictions_are_category_probabilities_and_memberships(separate):
    values = _unpack(POINT)
    pinned = _pin(separate, {k: v.unsqueeze(0) for k, v in values.items()})
    probabilities = _component_probabilities(values)
    weights = values["theta"]
    mixed = (weights[:, None, None] * probabilities).sum(0)
    epred = pinned.posterior_epred()[0, 0].double()
    assert epred.shape == (len(_frame()), CATEGORIES)
    torch.testing.assert_close(epred, mixed, rtol=TOL, atol=TOL)
    joint = weights[:, None] * probabilities.gather(
        -1, _codes()[None, :, None].expand(2, -1, 1)
    ).squeeze(-1)
    torch.testing.assert_close(
        pinned.pp_mixture()[0, 0].double(),
        (joint / joint.sum(0)).T,
        rtol=TOL,
        atol=TOL,
    )
    linpred = pinned.posterior_linpred(parameter="mu")[0, 0].double()
    torch.testing.assert_close(
        linpred[:, 1],
        values["dpar_mu2_intercept"] + values["dpar_mu2_beta_z"] * _column("z"),
        rtol=TOL,
        atol=TOL,
    )
    draws = pinned.posterior_predict(seed=5)
    assert set(draws.unique().tolist()) <= set(range(CATEGORIES))


def _thresholds(draws, name):
    return np.stack(
        [np.asarray(draws[f"{name}[{k}]"], dtype=np.float64) for k in (1, 2, 3)], -1
    )


def _weights(draws):
    theta = np.stack([np.asarray(draws["theta1"]), np.asarray(draws["theta2"])], -1)
    return theta / theta.sum(-1, keepdims=True)


def _numeric(values) -> torch.Tensor:
    array = np.array(values, dtype=object)
    return torch.tensor(
        np.where(array == "NaN", np.nan, array).astype(np.float64), dtype=torch.float64
    )


def _assert_brms(pinned, case, *, complete: bool = True):
    """Compare with brms where brms's R post-processing returns numbers.

    ``posterior_epred`` is compared everywhere. ``log_lik`` is also compared
    with the logarithm of brms's own ``posterior_epred`` at the observed
    category, which defines it on every row.
    """
    epred = _numeric(case["epred"])
    torch.testing.assert_close(
        pinned.posterior_epred()[0].double(), epred, rtol=BRMS_TOL, atol=BRMS_TOL
    )
    scores = pinned.log_likelihood()["y"][0].double()
    observed = _codes()[None, :, None].expand(epred.shape[0], -1, 1)
    torch.testing.assert_close(
        scores,
        epred.gather(-1, observed).squeeze(-1).log(),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )
    for actual, name in (
        (scores, "log_lik"),
        (pinned.pp_mixture()[0].double(), "pp_mixture"),
    ):
        expected = _numeric(case[name])
        finite = torch.isfinite(expected)
        assert bool(finite.all()) == complete
        assert bool(finite.any())
        torch.testing.assert_close(
            actual[finite], expected[finite], rtol=BRMS_TOL, atol=BRMS_TOL
        )


def test_separate_thresholds_match_brms_at_brms_draws(separate):
    case = ORACLE["separate"]
    assert case["order"] == "none" and separate.family.mixture.order == "none"
    stan = "\n".join(case["stan"])
    assert "ordered[nthres] Intercept_mu1" in stan
    assert "vector[nthres] Intercept_mu2" in stan
    draws = case["draws"]
    # brms thresholds tau; here tau_k = cutpoint_k - intercept, centered.
    first, second = (
        _thresholds(draws, "b_mu1_Intercept"),
        _thresholds(draws, "b_mu2_Intercept"),
    )
    pinned = _pin(
        separate,
        {
            "dpar_mu1_intercept": -first.mean(-1),
            "dpar_mu1_beta_x": np.asarray(draws["b_mu1_x"]),
            "dpar_mu1_cutpoint_log_spacing": np.log(np.diff(first, axis=-1)),
            "dpar_mu2_intercept": -second.mean(-1),
            "dpar_mu2_beta_z": np.asarray(draws["b_mu2_z"]),
            "dpar_mu2_cutpoint_offset": second[:, 1:] - second[:, :1],
            "theta": _weights(draws),
        },
    )
    _assert_brms(pinned, case)


def test_fixed_thresholds_match_brms_at_brms_draws(fixed):
    case = ORACLE["fixed"]
    assert case["order"] == "mu"
    stan = "\n".join(case["stan"])
    # One threshold vector, ordered because one component is cumulative.
    assert "ordered[nthres] fixed_Intercept" in stan
    assert "Intercept_mu1 = fixed_Intercept" in stan
    # The cumulative component samples the thresholds; the other reads them.
    assert set(_latent(fixed)) == {
        "dpar_mu1_beta_x",
        "dpar_mu2_intercept",
        "dpar_mu2_beta_z",
        "dpar_mu2_cutpoint_log_spacing",
        "theta",
    }
    draws = case["draws"]
    shared = _thresholds(draws, "b_mu1_Intercept")
    np.testing.assert_allclose(
        shared, _thresholds(draws, "b_mu2_Intercept"), rtol=0, atol=0
    )
    values = {
        "dpar_mu1_beta_x": np.asarray(draws["b_mu1_x"]),
        "dpar_mu2_intercept": -shared.mean(-1),
        "dpar_mu2_beta_z": np.asarray(draws["b_mu2_z"]),
        "dpar_mu2_cutpoint_log_spacing": np.log(np.diff(shared, axis=-1)),
        "theta": _weights(draws),
    }
    pinned = _pin(fixed, values)
    # brms's R log_lik and pp_mixture return NaN for the interior categories
    # of this model; its posterior_epred is complete.
    _assert_brms(pinned, case, complete=False)
    # The first component's reported intercept is the shared one.
    torch.testing.assert_close(
        pinned.draws()["dpar_mu1_intercept"].double(),
        pinned.draws()["dpar_mu2_intercept"].double(),
    )
    # A wrong variant (thresholds of the two components differing) fails.
    separate_thresholds = fit(
        "y ~ x",
        data=_frame(),
        family=mixture("sratio", "cumulative"),
        distributional={"mu2": "~ z"},
        method="svi",
        num_steps=0,
    )
    wrong = _pin(
        separate_thresholds,
        values
        | {
            "dpar_mu1_intercept": values["dpar_mu2_intercept"] + 0.5,
            "dpar_mu1_cutpoint_offset": shared[:, 1:] - shared[:, :1],
        },
    )
    assert (
        wrong.posterior_epred()[0].double() - _numeric(case["epred"])
    ).abs().max() > 1e-3


def test_adjacent_category_and_continuation_ratio_match_brms():
    case = ORACLE["ratio"]
    fitted = fit(
        "y ~ x",
        data=_frame(),
        family=mixture("acat", "cratio"),
        method="svi",
        num_steps=0,
    )
    draws = case["draws"]
    first, second = (
        _thresholds(draws, "b_mu1_Intercept"),
        _thresholds(draws, "b_mu2_Intercept"),
    )
    pinned = _pin(
        fitted,
        {
            "dpar_mu1_intercept": -first.mean(-1),
            "dpar_mu1_beta_x": np.asarray(draws["b_mu1_x"]),
            "dpar_mu1_cutpoint_offset": first[:, 1:] - first[:, :1],
            "dpar_mu2_intercept": -second.mean(-1),
            "dpar_mu2_beta_x": np.asarray(draws["b_mu2_x"]),
            "dpar_mu2_cutpoint_offset": second[:, 1:] - second[:, :1],
            "theta": _weights(draws),
        },
    )
    _assert_brms(pinned, case)


def test_probit_and_ratio_components_match_the_model_definitions():
    fitted = fit(
        "y ~ x",
        data=_frame(),
        family=mixture(families["cumulative"].with_links(mu="probit"), "cratio"),
        method="svi",
        num_steps=0,
    )
    spacing = torch.tensor([-0.3, 0.2], dtype=torch.float64)
    offset = torch.tensor([0.6, 1.1], dtype=torch.float64)
    weights = torch.tensor([0.35, 0.65], dtype=torch.float64)
    values = {
        "dpar_mu1_intercept": torch.tensor(0.2, dtype=torch.float64),
        "dpar_mu1_beta_x": torch.tensor(-0.7, dtype=torch.float64),
        "dpar_mu1_cutpoint_log_spacing": spacing,
        "dpar_mu2_intercept": torch.tensor(-0.4, dtype=torch.float64),
        "dpar_mu2_beta_x": torch.tensor(0.5, dtype=torch.float64),
        "dpar_mu2_cutpoint_offset": offset,
        "theta": weights,
    }
    x = _column("x")
    probabilities = weights[0] * _probabilities(
        "cumulative",
        _centered(torch.cumsum(spacing.exp(), 0)) - 0.2,
        -0.7 * x,
        link="probit",
    ) + weights[1] * _probabilities("cratio", _centered(offset) + 0.4, 0.5 * x)
    pinned = _pin(fitted, {k: v.unsqueeze(0) for k, v in values.items()})
    torch.testing.assert_close(
        pinned.posterior_epred()[0, 0].double(), probabilities, rtol=TOL, atol=TOL
    )
    torch.testing.assert_close(
        pinned.log_likelihood()["y"][0, 0].double(),
        probabilities.gather(-1, _codes()[:, None]).squeeze(-1).log(),
        rtol=TOL,
        atol=TOL,
    )


def test_family_rules_follow_brms():
    # brms: order = "none" by default for ordinal mixtures.
    assert mixture("cumulative", "cumulative").mixture.order == "none"
    assert mixture("cumulative", nmix=2, order=True).mixture.order == "mu"
    assert ORACLE["mixed_error"] == "Cannot mix ordinal and non-ordinal families."
    with pytest.raises(ValueError, match="cannot mix ordinal and non-ordinal"):
        mixture("cumulative", "poisson")
    assert "equidistant and fixed thresholds" in ORACLE["equidistant_error"]
    with pytest.raises(ValueError, match="require flexible thresholds"):
        mixture(
            families["cumulative"].with_thresholds("equidistant"),
            "cumulative",
            order="mu",
        )
    with pytest.raises(ValueError, match="no hurdle component"):
        mixture("hurdle_cumulative", "hurdle_cumulative", order="mu")
    frame = _frame()
    with pytest.raises(ValueError, match="keep or drop its intercept alike"):
        formula_to_qvr(
            "y ~ x",
            data=frame,
            family=mixture("cumulative", "cumulative", order="mu"),
            distributional={"mu2": "~ 0 + z"},
        )
    with pytest.raises(ValueError, match="thresholds_by is only valid"):
        formula_to_qvr(
            "y ~ x",
            data=frame.assign(g=["a", "b"] * 15),
            family=mixture("cumulative", "cumulative"),
            thresholds_by="g",
        )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_ordinal_mixtures_fit_predict_refit_and_restore(method, tmp_path):
    frame = _frame().assign(g=["a", "b", "c"] * 10)
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 15, "num_samples": 8, "num_chains": 1}
    )
    if method == "hmc":
        # Static HMC needs a step size that keeps the thresholds ordered.
        options["control"] = {"stepsize": 0.02}
    if method == "nuts":
        # The checks below concern shapes, finiteness and the artifact round
        # trip, not posterior accuracy, so each transition is capped at seven
        # leapfrog steps; an uncapped tree costs up to 1023 gradients.
        options["control"] = {"max_treedepth": 3}
    fitted = fit(
        "y ~ x + (1 | g)",
        data=frame,
        family=mixture("cumulative", "acat"),
        distributional={"mu2": "~ cs(z)", "disc1": "~ x"},
        method=method,
        seed=3,
        **options,
    )
    fresh = frame.iloc[:5].assign(x=lambda f: f.x + 0.1)
    epred = fitted.posterior_epred(new_data=fresh.drop(columns="y"), num_samples=3)
    assert epred.shape[-2:] == (5, CATEGORIES)
    torch.testing.assert_close(
        epred.sum(-1), torch.ones_like(epred.sum(-1)), rtol=1e-5, atol=1e-5
    )
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 5
    assert fitted.pp_mixture(new_data=fresh, num_samples=3).shape[-2:] == (5, 2)
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]).all()
    )
    path = tmp_path / "ordinal_mixture.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=3, seed=1),
        fitted.posterior_epred(new_data=fresh, num_samples=3, seed=1),
    )
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert "dpar_mu2_cutpoint_offset" in updated.draws()
        folds = fitted.kfold(K=2, seed=1)
        assert np.isfinite(folds.estimate.elpd)
        assert np.isfinite(fitted.loo().elpd)
        assert fitted.bayes_r2(num_samples=3).shape[-1] == 3


def test_ordinal_mixture_inside_a_joint_model():
    frame = _frame().assign(w=lambda f: 0.4 * f.x + f.z)
    spec = bf("y ~ x", family=mixture("cumulative", "acat")) + bf("w ~ x")
    joint = fit(spec, data=frame, method="svi", num_steps=5, seed=1)
    epred = joint.posterior_epred(resp="y", num_samples=2)["y"]
    assert epred.shape[-2:] == (len(frame), CATEGORIES)
    torch.testing.assert_close(
        epred.sum(-1), torch.ones_like(epred.sum(-1)), rtol=1e-5, atol=1e-5
    )
    scores = joint.log_likelihood(resp="y", num_samples=2)["y"]
    assert bool(torch.isfinite(scores).all())
