"""Residual laws beside the family: readers of single and joint fits agree.

Each test states an exact identity: a reader of a joint fit against the
mathematics of the same law, a subsampled criterion against the full one at
the same draws, or a marginal against an independent oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student_t
import torch

from quivers.continuous._multivariate_student_t import MultivariateStudentT
from quivers.diagnostics.regression import bayesian_r2
from quivers.formulas import bf, fit, formula_to_qvr, lf, nlf, set_rescor
from quivers.formulas._joint_residual import coordinate, marginal

ROWS = 24
SVI = {"method": "svi", "num_steps": 2, "seed": 0}
DRAWS = {"num_samples": 40, "seed": 0}


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y1": generator.normal(size=ROWS).round(3),
            "y2": generator.normal(size=ROWS).round(3),
            "x": generator.normal(size=ROWS).round(3),
            "x2": generator.normal(size=ROWS).round(3),
            "t": list(range(12)) * 2,
            "g": ["a"] * 12 + ["b"] * 12,
            "n": generator.poisson(3, size=ROWS),
            "w0": [1] + [0] * (ROWS - 1),
        }
    )


def test_the_family_is_kept_beside_its_residual_law():
    fitted = fit("y1 ~ x + ar(time = t, gr = g)", data=_frame(), **SVI)
    assert fitted.family.name == "gaussian"
    assert fitted.summary(num_samples=3).family == "gaussian(identity)"
    assert fitted.residual is not None and fitted.residual.name == "ResidualNormal_y1"
    assert "ResidualNormal_y1(" in fitted.qvr_source


def test_joint_linear_predictor_includes_the_conditional_recursion():
    fitted = fit(
        bf("y1 ~ x + ar(time = t, gr = g)") + bf("y2 ~ x") + set_rescor(False),
        data=_frame(),
        **SVI,
    )
    linear = fitted.posterior_linpred(resp="y1", num_samples=3)["y1"]
    expected = fitted.posterior_epred(resp="y1", num_samples=3)["y1"]
    plain = fitted.posterior_linpred(resp="y1", num_samples=3, incl_autocor=False)
    # Identity link: brms's mu, recursion included, is the conditional mean.
    torch.testing.assert_close(linear, expected)
    assert float((linear - plain["y1"]).abs().max()) > 1e-6


def test_joint_incl_autocor_false_scores_independent_rows():
    frame = _frame()
    fitted = fit(
        bf("y1 ~ x + cosy(time = t, gr = g)") + bf("y2 ~ x") + set_rescor(False),
        data=frame,
        **SVI,
    )
    options = {"num_samples": 3, "seed": 0, "incl_autocor": False}
    scores = fitted.log_likelihood(resp="y1", **options)["y1"]
    means = fitted.posterior_epred(resp="y1", **options)["y1"]
    draws = fitted.draws(num_samples=3, seed=0)
    sigma = draws["response_y1__sigma"].reshape(1, 3, 1)
    expected = torch.distributions.Normal(means, sigma).log_prob(
        torch.as_tensor(frame["y1"].to_numpy())
    )
    torch.testing.assert_close(scores, expected)
    eta = draws["response_y1__intercept"].reshape(1, 3, 1) + draws[
        "response_y1__beta_x"
    ].reshape(1, 3, 1) * torch.as_tensor(frame["x"].to_numpy())
    torch.testing.assert_close(means, eta)


def test_joint_unstructured_correlation_predicts_new_rows():
    frame = _frame()
    fitted = fit(
        bf("y1 ~ x + unstr(time = t, gr = g)") + bf("y2 ~ x") + set_rescor(False),
        data=frame,
        **SVI,
    )
    rows = frame.iloc[:3]
    means = fitted.posterior_epred(new_data=rows, num_samples=2)["y1"]
    linear = fitted.posterior_linpred(new_data=rows, num_samples=2)["y1"]
    assert means.shape == (1, 2, 3)
    torch.testing.assert_close(means, linear)


def _single_cosy():
    return fit("y1 ~ x + cosy(time = t, gr = g)", data=_frame(), **SVI), {}


def _joint_cosy():
    return (
        fit(
            bf("y1 ~ x + cosy(time = t, gr = g)") + bf("y2 ~ x") + set_rescor(False),
            data=_frame(),
            **SVI,
        ),
        {"resp": "y1"},
    )


def _joint_level_covariance():
    levels = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    return (
        fit(
            bf("y1 ~ x + (1 | gr(g, cov = K))") + bf("y2 ~ x") + set_rescor(False),
            data=_frame(),
            data2={"K": levels},
            **SVI,
        ),
        {"resp": "y1"},
    )


def _joint_unused_matrix():
    weights = pd.DataFrame(np.ones((2, 2)) - np.eye(2))
    return (
        fit(
            bf("y1 ~ x") + bf("y2 ~ x") + set_rescor(False),
            data=_frame(),
            data2={"W": weights},
            **SVI,
        ),
        {},
    )


def _latent_residuals():
    return (
        fit("n ~ x + ar(time = t, gr = g)", family="poisson", data=_frame(), **SVI),
        {},
    )


def _host_predictor():
    return (
        fit(
            "y1 ~ x",
            data=_frame(),
            predictor=torch.nn.Linear(1, 1).double(),
            predictor_data=torch.linspace(-1.0, 1.0, ROWS, dtype=torch.float64)[
                :, None
            ],
            **SVI,
        ),
        {},
    )


@pytest.mark.parametrize(
    "build",
    [
        _single_cosy,
        _joint_cosy,
        _joint_level_covariance,
        _joint_unused_matrix,
        _latent_residuals,
        _host_predictor,
    ],
)
def test_subsampled_loo_equals_full_loo_at_the_sampled_units(build):
    fitted, selection = build()
    units = [1, 5, 9, 17]
    full = fitted.loo(**selection, **DRAWS)
    sampled = fitted.loo_subsample(observations=units, **selection, **DRAWS)
    torch.testing.assert_close(
        sampled.pointwise_elpd[units], full.pointwise_elpd[units]
    )


def test_joint_r_squared_reads_each_response_as_a_single_fit():
    frame = _frame()
    censored = fit(
        bf("y1 | cens(w0) ~ x") + bf("y2 ~ x") + set_rescor(False), data=frame, **SVI
    )
    with pytest.raises(ValueError, match="censoring bounds"):
        censored.bayes_r2(resp="y1", method="residual", num_samples=3)
    fitted = fit(bf("y1 ~ x") + bf("y2 ~ x") + set_rescor(False), data=frame, **SVI)
    automatic = fitted.bayes_r2(num_samples=3)
    model = fitted.bayes_r2(method="model", num_samples=3)
    residual = fitted.bayes_r2(method="residual", num_samples=3)
    for name in ("y1", "y2"):
        # A Gaussian response defaults to the model form, as a single fit does.
        torch.testing.assert_close(automatic[f"R2{name}"], model[f"R2{name}"])
        means = fitted.posterior_epred(resp=name, num_samples=3)[name]
        torch.testing.assert_close(
            residual[f"R2{name}"],
            bayesian_r2(means, observed=torch.as_tensor(frame[name].to_numpy())),
        )


def test_r_squared_under_residual_correlation_uses_the_coordinate_marginal():
    fitted = fit(bf("y1 ~ x") + bf("y2 ~ x") + set_rescor(True), data=_frame(), **SVI)
    values = fitted.bayes_r2(method="model", num_samples=3)
    means = fitted.posterior_epred(resp="y1", num_samples=3)["y1"]
    draws = fitted.draws(num_samples=3, seed=0)
    sigma = draws["response_y1__sigma"].reshape(1, 3, 1).expand_as(means)
    torch.testing.assert_close(
        values["R2y1"], bayesian_r2(means, residual_variance=sigma.square())
    )


def test_coordinates_of_elliptical_vectors_match_scipy():
    generator = torch.Generator().manual_seed(4)
    loc = torch.randn(3, dtype=torch.float64, generator=generator)
    root = torch.randn(3, 3, dtype=torch.float64, generator=generator)
    covariance = root @ root.T + torch.eye(3, dtype=torch.float64)
    factor = torch.linalg.cholesky(covariance)
    value = torch.randn(2, dtype=torch.float64, generator=generator)
    keep = (2, 0)
    selected = covariance[list(keep)][:, list(keep)].numpy()
    normal = torch.distributions.MultivariateNormal(loc, scale_tril=factor)
    np.testing.assert_allclose(
        float(marginal(normal, keep).log_prob(value)),
        multivariate_normal.logpdf(value.numpy(), loc[list(keep)].numpy(), selected),
        rtol=1e-12,
    )
    student = MultivariateStudentT(4.5, loc, scale_tril=factor)
    np.testing.assert_allclose(
        float(marginal(student, keep).log_prob(value)),
        multivariate_t.logpdf(value.numpy(), loc[list(keep)].numpy(), selected, df=4.5),
        rtol=1e-12,
    )
    point = float(value[0])
    scale = float(covariance[1, 1].sqrt())
    np.testing.assert_allclose(
        float(coordinate(normal, 1).log_prob(value[0])),
        norm.logpdf(point, float(loc[1]), scale),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(coordinate(student, 1).log_prob(value[0])),
        student_t.logpdf(point, 4.5, float(loc[1]), scale),
        rtol=1e-12,
    )


@pytest.mark.parametrize("joint", [False, True])
def test_nonlinear_scale_parameters_beside_a_linear_location(joint):
    frame = _frame()
    spec = bf("y1 ~ x") + nlf("sigma ~ exp(a * x2)") + lf("a ~ 1")
    if joint:
        spec = spec + bf("y2 ~ x") + set_rescor(False)
    fitted = fit(spec, data=frame, **SVI)
    prefix = "response_y1__" if joint else ""
    draws = fitted.draws(num_samples=3, seed=0)
    location = draws[prefix + "intercept"].reshape(1, 3, 1) + draws[
        prefix + "beta_x"
    ].reshape(1, 3, 1) * torch.as_tensor(frame["x"].to_numpy())
    scale = torch.exp(
        torch.exp(
            draws[prefix + "nlpar_a_intercept"].reshape(1, 3, 1)
            * torch.as_tensor(frame["x2"].to_numpy())
        )
    )
    expected = torch.distributions.Normal(location, scale).log_prob(
        torch.as_tensor(frame["y1"].to_numpy())
    )
    selection = {"resp": "y1"} if joint else {}
    scores = fitted.log_likelihood(num_samples=3, seed=0, **selection)
    torch.testing.assert_close(scores["y1"], expected)


def test_residual_correlation_of_a_count_response_is_a_plain_refusal():
    with pytest.raises(ValueError, match="homogeneous Gaussian or Student"):
        formula_to_qvr(
            bf("y1 ~ x") + bf("n ~ x", family="poisson") + set_rescor(True),
            data=_frame(),
        )


def _ar1_means(eta, y, phi, groups, order):
    """Conditional AR(1) means of brms, transcribed on time-sorted series."""
    means = eta.clone()
    for start in range(len(order)):
        row = order[start]
        if start and groups[order[start - 1]] == groups[row]:
            previous = order[start - 1]
            means[..., row] = eta[..., row] + phi * (y[previous] - eta[..., previous])
    return means


def test_conditional_arma_beside_residual_correlations_scores_each_rows_vector():
    frame = _frame()
    spec = bf("y1 ~ x + ar(time = t, gr = g)") + bf("y2 ~ x") + set_rescor(True)
    fitted = fit(spec, data=frame, **SVI)
    draws = fitted.draws(num_samples=3, seed=0)
    x = torch.as_tensor(frame["x"].to_numpy())
    y = torch.as_tensor(frame[["y1", "y2"]].to_numpy())
    order = sorted(range(ROWS), key=lambda row: (frame["g"][row], frame["t"][row]))

    def column(name):
        return draws[name].reshape(3, 1)

    eta1 = column("response_y1__intercept") + column("response_y1__beta_x") * x
    eta2 = column("response_y2__intercept") + column("response_y2__beta_x") * x
    means1 = _ar1_means(
        eta1, y[:, 0], column("response_y1__ar_1"), list(frame["g"]), order
    )
    scales = torch.stack(
        (column("response_y1__sigma"), column("response_y2__sigma")), -1
    )
    factor = draws["rescor"].reshape(3, 2, 2)
    tril = scales[..., :, None] * factor[:, None]
    law = torch.distributions.MultivariateNormal(
        torch.stack((means1, eta2), -1), scale_tril=tril
    )
    expected = law.log_prob(y).reshape(1, 3, ROWS)
    torch.testing.assert_close(
        fitted.log_likelihood(num_samples=3, seed=0)["y1__y2"], expected
    )
    # One response's score is its coordinate's law at the same means.
    torch.testing.assert_close(
        fitted.log_likelihood(resp="y1", num_samples=3, seed=0)["y1"],
        torch.distributions.Normal(means1, column("response_y1__sigma"))
        .log_prob(y[:, 0])
        .reshape(1, 3, ROWS),
    )
    # brms's mu of the response includes the recursion.
    torch.testing.assert_close(
        fitted.posterior_linpred(resp="y1", num_samples=3, seed=0)["y1"],
        means1.reshape(1, 3, ROWS),
    )
    torch.testing.assert_close(
        fitted.posterior_epred(num_samples=3, seed=0)["y1"], means1.reshape(1, 3, ROWS)
    )
