"""Finite mixtures of response families: brms ``mixture()``.

Oracles: hand-written torch mixture densities (values, first and second
derivatives), SciPy component laws, and brms 2.20.4 ``log_lik``,
``pp_mixture`` and ``posterior_epred`` at brms's own posterior draws
(``oracles/brms_mixture.R``). Float64 parameters over float32-stored data
give agreement to float64 rounding (relative 1e-10) for densities and their
derivatives; brms values are pinned to 15 digits and compared at 1e-8.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats
from scipy.special import logsumexp

from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.formulas import bf, families, fit, formula_to_qvr, mixture
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference._response import response_distribution
from quivers.inference.trace import trace
from quivers.modeling.artifact import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_mixture.json").read_text()
)
RTOL = 1e-10
ATOL = 1e-10


def _frame() -> pd.DataFrame:
    rows = ORACLE["data"]
    return pd.DataFrame(
        {
            "y": np.asarray([row["y"] for row in rows], dtype=np.float64),
            "x": np.asarray([row["x"] for row in rows], dtype=np.float64),
            "z": np.asarray([row["z"] for row in rows], dtype=np.float64),
            "count": np.asarray([row["count"] for row in rows], dtype=np.int64),
        }
    )


def _latent(fitted) -> tuple[str, ...]:
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names


def _pin(fitted, values: dict[str, np.ndarray]):
    """Replace the posterior by explicit draws on one chain."""
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


def _column(name: str) -> torch.Tensor:
    # The program reads float32 design columns; the oracle reads the same values.
    return torch.tensor(_frame()[name].to_numpy(), dtype=torch.float32).double()


HETERO_SITES = (
    "dpar_mu1_intercept",
    "dpar_mu1_beta_x",
    "dpar_sigma1_intercept",
    "dpar_sigma1_beta_x",
    "dpar_mu2_intercept",
    "dpar_mu2_beta_x",
    "sigma2",
    "nu2",
    "dpar_theta2_intercept",
    "dpar_theta2_beta_z",
)


def _hetero_oracle(theta: torch.Tensor) -> torch.Tensor:
    """Hand-written log likelihood of mixture(gaussian, student_t)."""
    a1, b1, s0, s1, a2, b2, sigma2, nu2, t0, t1 = theta.unbind()
    x, z, y = _column("x"), _column("z"), _column("y")
    mu1 = a1 + b1 * x
    sigma1 = torch.exp(s0 + s1 * x)
    mu2 = a2 + b2 * x
    logits = torch.stack((torch.zeros_like(z), t0 + t1 * z), dim=-1)
    log_weights = torch.log_softmax(logits, dim=-1)
    scores = torch.stack(
        (
            torch.distributions.Normal(mu1, sigma1).log_prob(y),
            torch.distributions.StudentT(nu2, mu2, sigma2).log_prob(y),
        ),
        dim=-1,
    )
    return torch.logsumexp(log_weights + scores, dim=-1)


@pytest.fixture(scope="module")
def hetero():
    return fit(
        "y ~ x",
        data=_frame(),
        family=mixture("gaussian", "student_t"),
        distributional={"sigma1": "~ x", "theta2": "~ z"},
        method="svi",
        num_steps=0,
    )


def _model_log_likelihood(fitted, names, theta: torch.Tensor) -> torch.Tensor:
    latents = dict(zip(names, theta.unbind(), strict=True))
    recorded = trace(
        fitted.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fitted.observations) | latents,
    )
    return response_distribution(recorded, "y").log_prob(_column("y")).reshape(-1)


POINT = torch.tensor(
    [0.2, 0.4, -0.3, 0.25, 4.6, -0.5, 1.3, 3.5, -0.7, 0.9], dtype=torch.float64
)


def test_heterogeneous_density_matches_hand_written_mixture(hetero):
    assert set(_latent(hetero)) == set(HETERO_SITES)
    actual = _model_log_likelihood(hetero, HETERO_SITES, POINT)
    torch.testing.assert_close(actual, _hetero_oracle(POINT), rtol=RTOL, atol=ATOL)


def test_heterogeneous_density_first_and_second_derivatives(hetero):
    def model(theta):
        return _model_log_likelihood(hetero, HETERO_SITES, theta).sum()

    def oracle(theta):
        return _hetero_oracle(theta).sum()

    point = POINT.clone().requires_grad_(True)
    (actual,) = torch.autograd.grad(model(point), point)
    (expected,) = torch.autograd.grad(oracle(point), point)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    hessian = torch.autograd.functional.hessian(model, POINT.clone())
    expected_hessian = torch.autograd.functional.hessian(oracle, POINT.clone())
    torch.testing.assert_close(hessian, expected_hessian, rtol=1e-9, atol=1e-9)


def test_a_wrong_component_variant_is_detected(hetero):
    actual = _model_log_likelihood(hetero, HETERO_SITES, POINT)
    a1, b1, s0, s1, a2, b2, sigma2, nu2, t0, t1 = POINT.unbind()
    # Swapping which component carries the predicted proportion changes the model.
    swapped = torch.stack((a1, b1, s0, s1, a2, b2, sigma2, nu2, -t0, -t1))
    assert not torch.allclose(actual, _hetero_oracle(swapped), rtol=1e-3, atol=1e-3)


def _count_draws():
    return {
        "dpar_mu1_intercept": np.array([0.6, 0.8]),
        "dpar_mu1_beta_x": np.array([0.1, -0.2]),
        "dpar_mu2_intercept": np.array([2.1, 2.3]),
        "dpar_mu2_beta_x": np.array([0.05, 0.0]),
        "disp2": np.array([3.0, 7.5]),
    }


def _count_mixture():
    fitted = fit(
        "count ~ x",
        data=_frame(),
        family=mixture("poisson", "negative_binomial"),
        fixed_parameters={"theta1": 1.0, "theta2": 3.0},
        method="svi",
        num_steps=0,
    )
    return _pin(fitted, _count_draws())


def _count_scipy():
    frame, draws = _frame(), _count_draws()
    x, y = frame.x.to_numpy(np.float32).astype(np.float64), frame["count"].to_numpy()
    weights = np.log([0.25, 0.75])
    log_lik, membership, means = [], [], []
    for d in range(2):
        mu1 = np.exp(draws["dpar_mu1_intercept"][d] + draws["dpar_mu1_beta_x"][d] * x)
        mu2 = np.exp(draws["dpar_mu2_intercept"][d] + draws["dpar_mu2_beta_x"][d] * x)
        shape = draws["disp2"][d]
        scores = np.stack(
            (
                stats.poisson.logpmf(y, mu1),
                stats.nbinom.logpmf(y, shape, shape / (shape + mu2)),
            ),
            axis=-1,
        )
        joint = weights + scores
        log_lik.append(logsumexp(joint, axis=-1))
        membership.append(np.exp(joint - logsumexp(joint, axis=-1, keepdims=True)))
        means.append(0.25 * mu1 + 0.75 * mu2)
    return np.array(log_lik), np.array(membership), np.array(means)


def test_count_mixture_matches_scipy_components_with_fixed_proportions():
    fitted = _count_mixture()
    assert "{0 -> 0.25, 1 -> 0.75}" in fitted.qvr_source
    log_lik, membership, means = _count_scipy()
    torch.testing.assert_close(
        fitted.log_likelihood()["count"].double(),
        torch.tensor(log_lik).unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        fitted.pp_mixture().double(),
        torch.tensor(membership).unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        fitted.posterior_epred().double(),
        torch.tensor(means).unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )


def _brms_draws(name: str) -> dict[str, np.ndarray]:
    rows = ORACLE[name]
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def test_brms_log_lik_pp_mixture_and_epred_at_brms_draws():
    frame = _frame()
    fitted = fit(
        "y ~ x",
        data=frame,
        family=mixture("gaussian", "student_t"),
        distributional={"theta2": "~ z"},
        method="svi",
        num_steps=0,
    )
    draws = _brms_draws("gs_draws")
    fitted = _pin(
        fitted,
        {
            "dpar_mu1_intercept": draws["b_mu1_Intercept"],
            "dpar_mu1_beta_x": draws["b_mu1_x"],
            "sigma1": draws["sigma1"],
            "dpar_mu2_intercept": draws["b_mu2_Intercept"],
            "dpar_mu2_beta_x": draws["b_mu2_x"],
            "sigma2": draws["sigma2"],
            "nu2": draws["nu2"],
            "dpar_theta2_intercept": draws["b_theta2_Intercept"],
            "dpar_theta2_beta_z": draws["b_theta2_z"],
        },
    )
    expected_log_lik = torch.tensor(ORACLE["gs_log_lik"], dtype=torch.float64)
    torch.testing.assert_close(
        fitted.log_likelihood()["y"][0].double(), expected_log_lik, rtol=1e-6, atol=1e-6
    )
    # brms pp_mixture(summary = FALSE) is (draw, observation, component).
    expected_membership = torch.tensor(ORACLE["gs_pp_mixture"], dtype=torch.float64)
    torch.testing.assert_close(
        fitted.pp_mixture()[0].double(), expected_membership, rtol=1e-6, atol=1e-6
    )
    expected_epred = torch.tensor(ORACLE["gs_epred"], dtype=torch.float64)
    torch.testing.assert_close(
        fitted.posterior_epred()[0].double(), expected_epred, rtol=1e-6, atol=1e-6
    )


def test_brms_ordered_count_mixture_at_brms_draws():
    frame = _frame()
    fitted = fit(
        "count ~ x",
        data=frame,
        family=mixture("poisson", "poisson"),
        method="svi",
        num_steps=0,
    )
    draws = _brms_draws("pp_draws")
    # quivers orders the uncentered intercepts; brms's draws may not satisfy
    # that order, so the ordering-free model with brms's coefficients is
    # compared through the location formulas of the unordered mixture.
    unordered = fit(
        "count ~ x",
        data=frame,
        family=mixture("poisson", "poisson", order="none"),
        method="svi",
        num_steps=0,
    )
    unordered = _pin(
        unordered,
        {
            "dpar_mu1_intercept": draws["b_mu1_Intercept"],
            "dpar_mu1_beta_x": draws["b_mu1_x"],
            "dpar_mu2_intercept": draws["b_mu2_Intercept"],
            "dpar_mu2_beta_x": draws["b_mu2_x"],
            "theta": np.stack((draws["theta1"], draws["theta2"]), axis=-1),
        },
    )
    torch.testing.assert_close(
        unordered.log_likelihood()["count"][0].double(),
        torch.tensor(ORACLE["pp_log_lik"], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        unordered.posterior_epred()[0].double(),
        torch.tensor(ORACLE["pp_epred"], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    assert "mixture_intercept_spacing" in _latent(fitted)


def test_brms_construction_pins_match_the_emitted_program():
    """brms Stan code facts the QVR program reproduces."""
    ordered = "\n".join(ORACLE["ordered_stan"])
    assert "ordered[2] ordered_Intercept" in ordered
    assert "simplex[2] theta" in ordered
    assert "dirichlet_lpdf(theta | con_theta)" in ordered
    predicted = "\n".join(ORACLE["predicted_stan"])
    assert "vector[N] theta1 = rep_vector(0.0, N)" in predicted
    assert "theta2[n] = theta2[n] - log_sum_exp_theta" in predicted
    assert ORACLE["fixed_stan"][0].strip().startswith("real<lower=0,upper=1> theta1")
    assert any("sigma2 = sigma1" in line for line in ORACLE["equal_stan"])
    assert ORACLE["order_default_same"] == "mu"
    assert ORACLE["order_default_hetero"] == "none"
    frame = _frame()
    same = mixture("gaussian", "gaussian")
    hetero = mixture("gaussian", "student_t")
    assert same.mixture.order == "mu" and hetero.mixture.order == "none"
    source = formula_to_qvr("y ~ x", data=frame, family=same)
    assert "sample theta <- Dirichlet(concentration=[1.0, 1.0])" in source
    assert (
        "let dpar_mu2_intercept = dpar_mu1_intercept + mixture_intercept_spacing[0]"
        in source
    )
    predicted_source = formula_to_qvr(
        "y ~ x", data=frame, family=same, distributional={"theta2": "~ z"}
    )
    assert "{0 -> mixture_row_zero + 0.0, 1 -> mixture_row_zero + dpar_theta2_eta}" in (
        predicted_source
    )
    assert "let mixture_weights = softmax(mixture_logits)" in predicted_source
    equal_source = formula_to_qvr(
        "y ~ x",
        data=frame,
        family=mixture("gaussian", "gaussian", equal={"sigma2": "sigma1"}),
    )
    assert "let sigma2 = sigma1" in equal_source
    assert "sample sigma2" not in equal_source


def _classes(targets) -> set[str]:
    # brms lists an auxiliary such as sigma1 as class sigma1 with no dpar.
    return {
        f"{target.class_}|{'' if target.dpar in (None, target.class_) else target.dpar}"
        for target in targets
    }


def test_prior_classes_follow_brms_names():
    """brms get_prior classes; ordering replaces the later intercept priors."""
    frame = _frame()

    def targets(**options):
        fitted = fit("y ~ x", data=frame, method="svi", num_steps=0, **options)
        return fitted.prediction_plan.prior_targets

    hetero = _classes(targets(family=mixture("gaussian", "student_t")))
    brms_hetero = {
        entry for entry in ORACLE["prior_classes_hetero"] if not entry.startswith("b|")
    } | {"b|mu1", "b|mu2"}
    assert brms_hetero == hetero
    predicted = _classes(
        targets(
            family=mixture("gaussian", "gaussian"), distributional={"theta2": "~ z"}
        )
    )
    brms_predicted = set(ORACLE["prior_classes_predicted"])
    # brms lists an unused theta1 class; QVR adds the ordered-intercept gap.
    assert brms_predicted - {"theta1|"} - {"Intercept|mu2"} == predicted - {
        "mixture_intercept_spacing|"
    }
    ordered = _classes(targets(family=mixture("gaussian", "gaussian")))
    assert "theta|" in ordered and "mixture_intercept_spacing|" in ordered
    assert "Intercept|mu1" in ordered and "Intercept|mu2" not in ordered


def _bimodal(seed: int = 0, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    low = rng.random(n) < 0.5
    return pd.DataFrame(
        {"y": np.where(low, rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n))}
    )


@pytest.mark.slow
def test_ordering_removes_the_label_switched_mode():
    frame = _bimodal()
    unordered = fit(
        "y ~ 1",
        data=frame,
        family=mixture("gaussian", "gaussian", order="none"),
        method="svi",
        num_steps=0,
    )
    point = {
        "dpar_mu1_intercept": torch.tensor(-2.0, dtype=torch.float64),
        "dpar_mu2_intercept": torch.tensor(2.0, dtype=torch.float64),
        "sigma1": torch.tensor(0.5, dtype=torch.float64),
        "sigma2": torch.tensor(0.6, dtype=torch.float64),
        "theta": torch.tensor([0.4, 0.6], dtype=torch.float64),
    }
    swapped = {
        "dpar_mu1_intercept": point["dpar_mu2_intercept"],
        "dpar_mu2_intercept": point["dpar_mu1_intercept"],
        "sigma1": point["sigma2"],
        "sigma2": point["sigma1"],
        "theta": point["theta"].flip(0),
    }

    def joint(fitted, latents):
        return trace(
            fitted.program,
            torch.zeros(1, 1, dtype=torch.long),
            dict(fitted.observations) | latents,
        ).log_joint.sum()

    # Without ordering, the relabelled point has the same posterior density.
    torch.testing.assert_close(joint(unordered, point), joint(unordered, swapped))
    # With ordering, the second intercept is the first plus a positive gap, so
    # the relabelled point is not in the parameter space; NUTS chains started
    # anywhere identify component 1 as the lower location.
    ordered = fit(
        "y ~ 1",
        data=frame,
        family=mixture("gaussian", "gaussian"),
        method="nuts",
        # Each gradient costs tens of milliseconds, so the budget is the
        # smallest that carries both chains from the wrong side to the mode:
        # 40 warmup transitions of at most 31 leapfrog steps. The posterior
        # deviation of an intercept is about 0.1, and the retained means sit
        # near -2 and 2 for every seed tried, so the thresholds at -1 and 1
        # are more than five Monte Carlo standard errors of a 20-draw mean
        # (at most 0.35 / sqrt(20)) away.
        num_warmup=40,
        num_samples=20,
        num_chains=2,
        seed=4,
        initial_values={"dpar_mu1_intercept": 1.5},
        control={"max_treedepth": 5},
    )
    draws = ordered.draws()
    assert bool((draws["mixture_intercept_spacing"] > 0).all())
    assert bool((draws["dpar_mu2_intercept"] > draws["dpar_mu1_intercept"]).all())
    per_chain = draws["dpar_mu1_intercept"].mean(1)
    assert bool((per_chain < -1.0).all())
    assert bool((draws["dpar_mu2_intercept"].mean(1) > 1.0).all())


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_inference_prediction_refit_and_artifact(method, tmp_path):
    frame = _frame()
    options = (
        {"num_steps": 60}
        if method == "svi"
        else {"num_warmup": 30, "num_samples": 20, "num_chains": 1}
    )
    if method == "nuts":
        # The checks below concern shapes, finiteness, row consistency and the
        # artifact round trip, not posterior accuracy, so each transition is
        # capped at seven leapfrog steps.
        options["control"] = {"max_treedepth": 3}
    fitted = fit(
        "y ~ x",
        data=frame,
        family=mixture("gaussian", "student_t"),
        distributional={"theta2": "~ z"},
        priors={"nu2": "Gamma(concentration=8.0, rate=1.0)"},
        method=method,
        seed=3,
        **options,
    )
    fresh = frame.iloc[:5].drop(columns=["y"])
    rows = frame.iloc[:5]
    assert fitted.posterior_epred(new_data=fresh, num_samples=4).shape[-1] == 5
    assert fitted.posterior_predict(new_data=fresh, num_samples=4).shape[-1] == 5
    assert fitted.posterior_linpred(new_data=fresh, num_samples=4).shape[-2:] == (5, 2)
    assert (
        fitted.posterior_linpred(parameter="mu2", new_data=fresh, num_samples=4).shape[
            -1
        ]
        == 5
    )
    log_lik = fitted.log_likelihood(new_data=rows, num_samples=4)["y"]
    assert log_lik.shape[-1] == 5 and bool(torch.isfinite(log_lik).all())
    membership = fitted.pp_mixture(new_data=rows, num_samples=4)
    torch.testing.assert_close(
        membership.sum(-1), torch.ones(membership.shape[:-1], dtype=membership.dtype)
    )
    # The new-row membership is the training-row membership of the same rows.
    torch.testing.assert_close(
        membership, fitted.pp_mixture(num_samples=4)[..., :5, :], rtol=1e-5, atol=1e-6
    )
    path = tmp_path / "mixture.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=4, seed=1)["y"],
        fitted.log_likelihood(num_samples=4, seed=1)["y"],
    )
    assert restored.family == fitted.family
    if method == "svi":
        updated = fitted.update(num_steps=5)
        assert updated.family.mixture == fitted.family.mixture
        callback = fitted.refit_callback(
            method_factory=lambda settings, seed: {
                **settings,
                "num_steps": 5,
                "seed": seed,
            },
            num_samples=4,
        )
        result = cross_validate(
            FoldPlan(assignments=tuple(i % 3 for i in range(len(frame))), seed=2),
            callback,
        )
        assert bool(torch.isfinite(result.log_predictive_density).all())


def test_truncated_and_censored_heterogeneous_mixture_matches_scipy():
    frame = _frame().assign(
        low=-1.5,
        high=7.0,
        status=lambda f: np.where(f.y > 5.5, 1, 0),
    )
    frame = frame.assign(yc=np.minimum(frame.y, 5.5))
    fitted = fit(
        "yc | cens(status) + trunc(lb = low, ub = high) ~ 1",
        data=frame,
        family=mixture("gaussian", "student_t"),
        method="svi",
        num_steps=0,
    )
    draws = {
        "dpar_mu1_intercept": np.array([0.1, -0.2]),
        "sigma1": np.array([1.1, 0.9]),
        "dpar_mu2_intercept": np.array([4.8, 5.1]),
        "sigma2": np.array([1.3, 0.8]),
        "nu2": np.array([4.0, 9.0]),
        "theta": np.array([[0.6, 0.4], [0.7, 0.3]]),
    }
    fitted = _pin(fitted, draws)
    assert "Normalize(base=Restrict(base=Normal(" in fitted.qvr_source
    y = frame.yc.to_numpy(np.float32).astype(np.float64)
    right = frame.status.to_numpy() == 1
    expected = []
    for d in range(2):
        laws = (
            stats.norm(draws["dpar_mu1_intercept"][d], draws["sigma1"][d]),
            stats.t(
                draws["nu2"][d], draws["dpar_mu2_intercept"][d], draws["sigma2"][d]
            ),
        )
        parts = []
        for law, weight in zip(laws, draws["theta"][d], strict=True):
            mass = law.cdf(7.0) - law.cdf(-1.5)
            event = np.where(right, law.cdf(7.0) - law.cdf(y), law.pdf(y))
            parts.append(weight * event / mass)
        expected.append(np.log(np.sum(parts, axis=0)))
    torch.testing.assert_close(
        fitted.log_likelihood()["yc"].double(),
        torch.tensor(np.array(expected)).unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    predicted = fitted.posterior_predict(seed=3)
    assert bool(((predicted >= -1.5) & (predicted <= 7.0)).all())


def test_mixture_response_in_a_joint_multivariate_model():
    frame = _frame()
    spec = bf("y ~ x", family=mixture("gaussian", "gaussian")) + bf(
        "count ~ x", family="poisson"
    )
    joint = fit(spec, data=frame, method="svi", num_steps=20, seed=2)
    fresh = frame.iloc[:4]
    assert (
        joint.posterior_epred(resp="y", new_data=fresh.drop(columns="y"))["y"].shape[-1]
        == 4
    )
    scores = joint.log_likelihood(resp="y", new_data=fresh)["y"]
    assert bool(torch.isfinite(scores).all())
    assert any(name.endswith("mixture_intercept_spacing") for name in joint.draws())


def test_equal_and_fixed_parameters_change_the_density_as_declared():
    frame = _frame()
    fitted = fit(
        "y ~ 1",
        data=frame,
        family=mixture(
            "gaussian", "gaussian", order="none", equal={"sigma2": "sigma1"}
        ),
        fixed_parameters={"theta1": 2.0, "theta2": 6.0},
        method="svi",
        num_steps=0,
    )
    assert set(_latent(fitted)) == {
        "dpar_mu1_intercept",
        "dpar_mu2_intercept",
        "sigma1",
    }
    latents = {
        "dpar_mu1_intercept": torch.tensor(0.1, dtype=torch.float64),
        "dpar_mu2_intercept": torch.tensor(4.9, dtype=torch.float64),
        "sigma1": torch.tensor(1.2, dtype=torch.float64),
    }
    y = _column("y")
    recorded = trace(
        fitted.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fitted.observations) | latents,
    )
    actual = response_distribution(recorded, "y").log_prob(y).reshape(-1)
    expected = torch.logsumexp(
        torch.stack(
            (
                math.log(0.25)
                + torch.distributions.Normal(
                    latents["dpar_mu1_intercept"], latents["sigma1"]
                ).log_prob(y),
                math.log(0.75)
                + torch.distributions.Normal(
                    latents["dpar_mu2_intercept"], latents["sigma1"]
                ).log_prob(y),
            )
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "family": mixture("gaussian", "gaussian", "gaussian"),
                "distributional": {"theta2": "~ z"},
            },
            "all but one",
        ),
        (
            {
                "family": mixture("gaussian", "gaussian"),
                "fixed_parameters": {"theta1": 1.0},
            },
            "no or all",
        ),
        (
            {
                "family": mixture("gaussian", "gaussian"),
                "fixed_parameters": {"theta1": -1.0, "theta2": 1.0},
            },
            "nonnegative",
        ),
        (
            {
                "family": mixture("gaussian", "gaussian"),
                "distributional": {"mu": "~ x"},
            },
            "mu1, mu2",
        ),
        (
            {
                "family": mixture("gaussian", "gaussian"),
                "distributional": {"mu1": "~ 0 + x"},
            },
            "intercepts to be present",
        ),
    ],
)
def test_rejected_specifications(options, message):
    frame = _frame().assign(g=["a", "b", "c"] * 4)
    with pytest.raises(ValueError, match=message):
        formula_to_qvr("y ~ x", data=frame, **options)


@pytest.mark.parametrize(
    "name",
    [
        "hurdle_gamma",
        "hurdle_lognormal",
        "zero_inflated_beta",
        "zero_one_inflated_beta",
        "xbeta",
        "zero_inflated_asym_laplace",
    ],
)
def test_components_with_a_point_mass_fit_and_predict(name):
    generator = np.random.default_rng(3)
    rows = 40
    family = families[name]
    interval = family.response_interval
    assert interval is not None
    if interval.upper is None and interval.lower is not None:
        response = generator.gamma(2.0, 1.0, size=rows)
    elif interval.upper is None:
        response = generator.normal(size=rows)
    else:
        response = np.clip(generator.beta(2.0, 5.0, size=rows), 0.01, 0.99)
    response[:8] = 0.0
    frame = pd.DataFrame({"y": response, "x": generator.normal(size=rows)})
    fitted = fit(
        "y ~ x",
        data=frame,
        family=mixture(name, name),
        method="svi",
        num_steps=20,
        seed=1,
    )
    scores = next(iter(fitted.log_likelihood().values()))
    assert torch.isfinite(scores).all()
    predicted = fitted.posterior_predict(num_samples=20, seed=2)
    assert torch.isfinite(predicted).all()
    # The point mass at zero survives mixing.
    assert (predicted == 0).any()
    assert torch.isfinite(fitted.posterior_epred()).all()


def test_rejected_families():
    with pytest.raises(ValueError, match="real and integer"):
        mixture("gaussian", "poisson")
    with pytest.raises(ValueError, match="not allowed in mixture"):
        mixture("categorical", "categorical")
    with pytest.raises(ValueError, match="cannot mix ordinal and non-ordinal"):
        mixture("cumulative", "poisson")
    with pytest.raises(ValueError, match="at least 2"):
        mixture("gaussian")
    with pytest.raises(ValueError, match="same class"):
        mixture("gaussian", "student_t", equal={"nu2": "sigma1"})
    with pytest.raises(ValueError, match="class 'mu'"):
        mixture("gaussian", "gaussian", equal={"mu2": "mu1"})
    with pytest.raises(ValueError, match="order"):
        mixture("gaussian", "gaussian", order="sigma")
    with pytest.raises(ValueError, match="nmix"):
        mixture("gaussian", "gaussian", nmix=(1, 2, 3))
    assert len(mixture("gaussian", "student_t", nmix=(2, 1)).mixture.components) == 3
    with pytest.raises(ValueError, match="mixture links"):
        mixture("gaussian", "gaussian").with_links(mu1="log")
    with pytest.raises(ValueError, match="poisson"):
        fit(
            "y ~ x",
            data=_frame(),
            family=mixture("poisson", "poisson"),
            method="svi",
            num_steps=0,
        )
    template = families["mixture"].mixture
    assert template is not None
    assert template.components == (families["gaussian"],)


def test_group_identifier_shared_across_components_is_one_block():
    """brms correlates (1 | p | g) across mixture components."""
    frame = _frame().assign(g=["a", "b", "c"] * 4)
    fitted = fit(
        "y ~ x + (1 | p | g)",
        data=frame,
        family=mixture("gaussian", "gaussian"),
        distributional={"sigma2": "~ 1 + (1 | p | g)"},
        method="svi",
        num_steps=5,
    )
    source = fitted.qvr_source
    assert source.count("sample cor_shared_p :") == 1
    layout = next(
        layout
        for layout in fitted.prediction_plan.group_layouts
        if layout.correlation_site == "cor_shared_p"
    )
    assert {member.prefix for member in layout.members} == {
        "dpar_mu1_",
        "dpar_mu2_",
        "dpar_sigma2_",
    }
    targets = {target.site: target for target in fitted.prediction_plan.prior_targets}
    assert targets["cor_shared_p"].class_ == "cor"
    assert targets["dpar_sigma2_sigma_g_Intercept"].dpar == "sigma2"
    fresh = pd.DataFrame({"x": [0.3], "g": ["new"]})
    assert fitted.posterior_epred(
        new_data=fresh, allow_new_levels=True, num_samples=2
    ).shape == (1, 2, 1)


def test_template_links_are_the_links_of_its_component():
    linked = families["mixture"].with_links(mu="log", sigma="softplus")
    component = families["gaussian"].with_links(mu="log", sigma="softplus")
    frame = _frame()
    source = formula_to_qvr("y ~ x", data=frame, family=linked)
    assert source == formula_to_qvr(
        "y ~ x", data=frame, family=mixture(component, component)
    )
    assert source != formula_to_qvr("y ~ x", data=frame, family="mixture")
    with pytest.raises(ValueError, match="admissible links"):
        families["mixture"].with_links(mu="softmax")
