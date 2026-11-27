"""A family, a nonlinear formula and a mixture are kernels: their programs agree.

Oracles: the formula fits of the same models. At shared latent values the
program with formulas has the log density of the formula fit (the Gaussian
program against ``y ~ x + (1 | g)`` with ``sigma ~ z``; the decay program
against the nonlinear formula ``y ~ a * exp(0 - b * x)``), and the two-class
program has the pointwise log likelihood of ``mixture(gaussian, gaussian)``
row by row. Each comparison is paired with a model that differs in one
respect and must not agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.family_catalog import FamilyCatalog
from quivers.formulas import (
    AuxParam,
    NonlinearExpression,
    custom_family,
    fit,
    links,
    mixture,
)
from quivers.qiec.families import family as semantic_family

from tests.formulas._program_cases import (
    log_joint,
    DECAY,
    GAUSS,
    MIX2,
    frame,
    latent_names,
    pin,
    run,
)

TOLERANCE = 1e-10


def _svi(*arguments, **keywords):
    return fit(*arguments, data=frame(), method="svi", num_steps=1, seed=0, **keywords)


def _values(names, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(
            (3,) if name.startswith("z_") else (),
            generator=generator,
            dtype=torch.float64,
        )
        + 0.2
        for name in names
    }


def test_the_gaussian_program_is_the_gaussian_family() -> None:
    program = _svi(
        program=GAUSS,
        parameters={"mu": "~ x + (1 | g)", "sigma": "~ z"},
        response="y",
    )
    family = _svi("y ~ x + (1 | g)", distributional={"sigma": "~ z"})
    names = latent_names(program)
    assert set(names) == set(latent_names(family))
    values = _values(names, seed=1)
    joint = log_joint(run(program, values))
    assert joint == pytest.approx(log_joint(run(family, values)), rel=TOLERANCE)
    other = _svi("y ~ x + (1 | g)", distributional={"sigma": "~ x"})
    assert abs(joint - log_joint(run(other, values))) > 1e-3


def test_the_student_program_is_the_custom_student_family() -> None:
    source = GAUSS.replace(
        "observe y : Trial <- Normal(loc=mu, scale=sigma)",
        "observe y : Trial <- StudentT(df=5.0, loc=mu, scale=sigma)",
    )
    law = semantic_family("StudentT").with_(name="UserStudentT", targets={})
    catalog = FamilyCatalog().with_family(
        law,
        lambda parameters: torch.distributions.StudentT(
            parameters["df"], parameters["loc"], parameters["scale"]
        ),
    )
    family = custom_family(
        "UserStudentT",
        catalog=catalog,
        location="loc",
        auxiliary={
            "df": AuxParam(name="df", prior="HalfNormal(1.0)", link=links["log"]),
            "scale": AuxParam(name="sigma", prior="HalfNormal(1.0)", link=links["log"]),
        },
    )
    program = _svi(
        program=source, parameters={"mu": "~ x", "sigma": "~ z"}, response="y"
    )
    custom = _svi(
        "y ~ x",
        family=family,
        distributional={"sigma": "~ z"},
        fixed_parameters={"df": 5.0},
    )
    names = latent_names(program)
    assert set(names) == set(latent_names(custom))
    values = _values(names, seed=1)
    expected = log_joint(run(custom, values))
    assert log_joint(run(program, values)) == pytest.approx(expected, rel=TOLERANCE)
    other = _svi(
        program=source.replace("StudentT(df=5.0", "StudentT(df=2.0"),
        parameters={"mu": "~ x", "sigma": "~ z"},
        response="y",
    )
    assert abs(log_joint(run(other, values)) - expected) > 1e-3


def test_the_decay_program_is_the_nonlinear_formula() -> None:
    program = _svi(program=DECAY, parameters={"a": "~ 1", "b": "~ 1"}, response="y")
    formula = _svi(
        "y ~ a * exp(0 - b * x)",
        nonlinear={"a": "~ 1", "b": "~ 1"},
        priors={"sigma": "HalfNormal(1.0)"},
    )
    pairs = {
        "dpar_a_intercept": "nlpar_a_intercept",
        "dpar_b_intercept": "nlpar_b_intercept",
        "sigma": "sigma",
    }
    assert set(latent_names(program)) == set(pairs)
    assert set(latent_names(formula)) == set(pairs.values())
    values = {"dpar_a_intercept": 1.3, "dpar_b_intercept": 0.4, "sigma": 0.8}
    tensors = {
        name: torch.tensor(value, dtype=torch.float64) for name, value in values.items()
    }
    joint = log_joint(run(program, tensors))
    renamed = {pairs[name]: value for name, value in tensors.items()}
    assert joint == pytest.approx(log_joint(run(formula, renamed)), rel=TOLERANCE)
    growth = _svi(
        "y ~ a * exp(b * x)",
        nonlinear={"a": "~ 1", "b": "~ 1"},
        priors={"sigma": "HalfNormal(1.0)"},
    )
    assert abs(joint - log_joint(run(growth, renamed))) > 1e-3


def test_a_program_parameter_uses_the_shared_nonlinear_predictor() -> None:
    parameters = {
        "sigma": NonlinearExpression(expression="a + b * z"),
        "a": "~ 1",
        "b": "~ 1",
    }
    program = _svi(program=GAUSS, parameters={"mu": "~ x", **parameters}, response="y")
    family = _svi("y ~ x", distributional=parameters)
    names = latent_names(program)
    assert set(names) == set(latent_names(family))
    values = _values(names, seed=1)
    assert log_joint(run(program, values)) == pytest.approx(
        log_joint(run(family, values)), rel=TOLERANCE
    )
    other = _svi(
        program=GAUSS,
        parameters={
            "mu": "~ x",
            **parameters,
            "sigma": NonlinearExpression(expression="a - b * z"),
        },
        response="y",
    )
    assert abs(log_joint(run(program, values)) - log_joint(run(other, values))) > 1e-3


@pytest.mark.parametrize("nonlinear", [False, True])
def test_a_neural_predictor_adds_to_the_program_location(nonlinear: bool) -> None:
    data = torch.tensor(frame()["z"].to_numpy(), dtype=torch.float64)
    neural = {"predictor": torch.nn.Identity(), "predictor_data": data.square()}
    parameters = (
        {"mu": NonlinearExpression(expression="location"), "location": "~ 1"}
        if nonlinear
        else {"mu": "~ 1"}
    )
    program = _svi(program=GAUSS, parameters=parameters, response="y", **neural)
    ordinary = {"distributional": parameters} if nonlinear else {}
    family = _svi(
        "y ~ 0" if nonlinear else "y ~ 1",
        priors={"sigma": "HalfNormal(1.0)"},
        **neural,
        **ordinary,
    )
    names = latent_names(program)
    assert set(names) == set(latent_names(family))
    values = _values(names, seed=1)
    assert log_joint(run(program, values)) == pytest.approx(
        log_joint(run(family, values)), rel=TOLERANCE
    )
    other = _svi(program=GAUSS, parameters=parameters, response="y")
    assert abs(log_joint(run(program, values)) - log_joint(run(other, values))) > 1e-3


def test_a_program_vector_parameter_keeps_its_coordinate_axis() -> None:
    program = _svi(
        program=MIX2,
        parameters={
            "centers": NonlinearExpression(expression={0: "lo", 1: "hi"}),
            "lo": "~ 1",
            "hi": "~ 1",
        },
        response="y",
    )
    center = np.array([[-0.5, 1.7]])
    logits = np.array([[0.3, -0.4]])
    sigma = np.array([0.9])
    predicted = pin(
        program,
        {
            "nlpar_lo_intercept": center[:, 0],
            "nlpar_hi_intercept": center[:, 1],
            "logits": logits,
            "sigma": sigma,
        },
    ).log_likelihood()["y"]
    direct = _svi(program=MIX2, parameters={}, response="y")
    recorded = pin(
        direct, {"centers": center, "logits": logits, "sigma": sigma}
    ).log_likelihood()["y"]
    torch.testing.assert_close(
        predicted.double(), recorded.double(), rtol=TOLERANCE, atol=0
    )


@pytest.mark.parametrize("with_slope", [False, True])
def test_a_program_row_draw_can_take_a_regression(with_slope: bool) -> None:
    source = GAUSS.replace("sample mu <-", "sample mu : Trial <-")
    rows = frame().assign(y=np.arange(-6, 6) / 8, x=np.arange(12) / 8)
    fitted = fit(
        program=source,
        parameters={"mu": "~ x" if with_slope else "~ 1"},
        response="y",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    intercept, scale, slope = 0.375, 0.75, -0.5
    values = {"intercept": np.asarray([intercept]), "sigma": np.asarray([scale])}
    if with_slope:
        values["beta_x"] = np.asarray([slope])
    location = torch.full((len(rows),), intercept, dtype=torch.float64)
    if with_slope:
        location = location + slope * torch.tensor(rows["x"].to_numpy())
    actual = pin(fitted, values).log_likelihood()["y"].reshape(-1)
    y = torch.tensor(rows["y"].to_numpy())
    expected = torch.distributions.Normal(location, scale).log_prob(y)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    wrong = torch.distributions.Normal(location + 1.0, scale).log_prob(y)
    assert float((actual - wrong).abs().max()) > 1e-2


@pytest.mark.parametrize("per_row", [False, True])
def test_a_program_tensor_parameter_retains_its_nominal_coordinates(
    per_row: bool,
) -> None:
    shape = "[Trial, Dept, Kind]" if per_row else "[Dept, Kind]"
    location = "factor trial : Trial in grid[trial, 1, 0]" if per_row else "grid[1, 0]"
    source = f"""object Trial : FinSet 6
object Dept : FinSet 2
object Kind : FinSet 3

program tensor_mean(grid : Tensor[Real]({shape})) : Trial -> Trial
    let mu = {location}
    observe y : Trial <- Normal(loc=mu, scale=1.0)
    return y

export tensor_mean
"""
    rows = frame().assign(y=np.arange(-6, 6) / 8, x=np.arange(12) / 8)
    fitted = fit(
        program=source,
        parameters={"grid": "~ x" if per_row else "~ 1"},
        response="y",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    intercept = torch.tensor([0.6, -0.5, 0.9, -0.2, 1.75, -1.5], dtype=torch.float64)
    slope = torch.tensor([-0.4, 0.65, 1.1, 0.125, 0.2, -0.9], dtype=torch.float64)
    values = {"dpar_grid_intercept": intercept}
    if per_row:
        values["dpar_grid_beta_x"] = slope
    scores = (
        pin(fitted, {name: value.numpy()[None] for name, value in values.items()})
        .log_likelihood()["y"]
        .reshape(-1)
    )
    mean = intercept[3]
    if per_row:
        mean = mean + slope[3] * torch.tensor(rows["x"].to_numpy())
    response = torch.tensor(rows["y"].to_numpy())
    expected = torch.distributions.Normal(mean, 1.0).log_prob(response)
    torch.testing.assert_close(scores, expected, rtol=TOLERANCE, atol=0.0)
    wrong = torch.distributions.Normal(intercept[1], 1.0).log_prob(response)
    assert float((scores - wrong).abs().max()) > 1e-3


@pytest.mark.parametrize("law", ["poisson", "negative_binomial", "geometric", "logits"])
def test_a_program_count_observation_scales_with_exposure(law: str) -> None:
    constructor = {
        "poisson": "Poisson(rate=mu)",
        "negative_binomial": "NegativeBinomial(total_count=mu, probs=0.25)",
        "geometric": "Geometric(probs=mu)",
        "logits": "Geometric(logits=mu)",
    }[law]
    source = f"""object Trial : FinSet 6

program unit_count : Trial -> Trial
    sample mu <- Beta(concentration1=2.0, concentration0=3.0)
    observe y : Trial <- {constructor}
    return y

export unit_count
"""
    rows = frame().assign(y=np.arange(12) % 4, exposure=(np.arange(12) + 2) / 2)
    fitted = fit(
        program=source,
        parameters={},
        response="y | rate(exposure)",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    mu = torch.tensor(0.375, dtype=torch.float64)
    y = torch.tensor(rows["y"].to_numpy(), dtype=torch.float64)
    exposure = torch.tensor(rows["exposure"].to_numpy())
    distributions = torch.distributions
    if law == "poisson":
        accumulated = distributions.Poisson(mu * exposure)
        unit = distributions.Poisson(mu)
    elif law == "negative_binomial":
        probability = mu.new_tensor(0.25)
        accumulated = distributions.NegativeBinomial(mu * exposure, probs=probability)
        unit = distributions.NegativeBinomial(mu, probs=probability)
    elif law == "geometric":
        accumulated = distributions.NegativeBinomial(exposure, probs=1.0 - mu)
        unit = distributions.Geometric(probs=mu)
    else:
        accumulated = distributions.NegativeBinomial(exposure, logits=-mu)
        unit = distributions.Geometric(logits=mu)
    actual = pin(fitted, {"mu": mu.numpy()[None]}).log_likelihood()["y"].reshape(-1)
    expected = accumulated.log_prob(y)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    assert float((actual - unit.log_prob(y)).abs().max()) > 1e-2


@pytest.mark.parametrize("estimate_sigma", [False, True])
def test_the_program_observation_takes_known_errors(estimate_sigma: bool) -> None:
    rows = frame().assign(y=np.arange(-6, 6) / 8, error=(np.arange(12) + 2) / 8)
    sigma_option = ", sigma=True" if estimate_sigma else ""
    fitted = fit(
        program=GAUSS,
        parameters={"mu": "~ 1"},
        response=f"y | se(error{sigma_option})",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    values = {
        "intercept": torch.tensor(0.45, dtype=torch.float64),
        "sigma": torch.tensor(0.8, dtype=torch.float64),
    }
    y = torch.tensor(rows["y"].to_numpy())
    error = torch.tensor(rows["error"].to_numpy())
    combined = torch.hypot(error, values["sigma"])
    scale = combined if estimate_sigma else error
    scores = torch.distributions.Normal(values["intercept"], scale).log_prob(y)
    prior = torch.distributions.Normal(0.0, 5.0).log_prob(
        values["intercept"]
    ) + torch.distributions.HalfNormal(1.0).log_prob(values["sigma"])
    actual = log_joint(run(fitted, values))
    assert actual == pytest.approx(float(prior + scores.sum()), rel=1e-10)
    wrong_scale = error if estimate_sigma else combined
    wrong = torch.distributions.Normal(values["intercept"], wrong_scale).log_prob(y)
    assert abs(actual - float(prior + wrong.sum())) > 1e-2


def test_a_program_fixed_residual_covariance_scores_the_rows_together() -> None:
    rows = frame().assign(y=np.arange(-6, 6) / 8)
    covariance = 0.75 * np.eye(len(rows)) + 0.25
    correlated = fit(
        program=GAUSS.replace("sigma", "spread"),
        parameters={"mu": "~ 1 + fcor(V)"},
        response="y",
        data=rows,
        data2={"V": covariance},
        method="svi",
        num_steps=1,
        seed=0,
    )
    mean = torch.tensor(0.375, dtype=torch.float64)
    scale = torch.tensor(0.75, dtype=torch.float64)
    values = {"intercept": mean, "spread": scale}
    y = torch.tensor(rows["y"].to_numpy())
    prior = torch.distributions.Normal(0.0, 5.0).log_prob(
        mean
    ) + torch.distributions.HalfNormal(1.0).log_prob(scale)
    multivariate = torch.distributions.MultivariateNormal(
        mean.expand(len(rows)),
        covariance_matrix=scale.square() * torch.tensor(covariance),
    )
    actual = log_joint(run(correlated, values))
    assert actual == pytest.approx(float(prior + multivariate.log_prob(y)), rel=1e-9)
    independent = prior + torch.distributions.Normal(mean, scale).log_prob(y).sum()
    assert abs(actual - float(independent)) > 1e-2


def test_the_program_observation_composes_weights_and_recorded_events() -> None:
    rows = frame().assign(
        y=np.linspace(-1.0, 1.0, 12),
        w=np.linspace(0.25, 2.25, 12),
        status=np.tile([-1, 0, 1], 4),
    )
    fitted = fit(
        program=GAUSS,
        parameters={"mu": "~ 1"},
        response="y | cens(status) + trunc(lb = -2, ub = 2)",
        weights="w",
        data=rows,
        method="svi",
        num_steps=1,
        seed=0,
    )
    values = {
        "intercept": torch.tensor(0.45, dtype=torch.float64),
        "sigma": torch.tensor(0.8, dtype=torch.float64),
    }
    normal = torch.distributions.Normal(values["intercept"], values["sigma"])
    response = torch.tensor(rows["y"].to_numpy())
    status = torch.tensor(rows["status"].to_numpy())
    weight = torch.tensor(rows["w"].to_numpy())
    lower, upper = normal.cdf(torch.tensor([-2.0, 2.0], dtype=torch.float64))
    cumulative = normal.cdf(response)
    scores = (
        torch.where(
            status < 0,
            (cumulative - lower).log(),
            torch.where(
                status > 0, (upper - cumulative).log(), normal.log_prob(response)
            ),
        )
        - (upper - lower).log()
    )
    prior = torch.distributions.Normal(0.0, 5.0).log_prob(
        values["intercept"]
    ) + torch.distributions.HalfNormal(1.0).log_prob(values["sigma"])
    actual = log_joint(run(fitted, values))
    assert actual == pytest.approx(float(prior + (scores * weight).sum()), rel=1e-7)
    assert abs(actual - float(prior + scores.sum())) > 1e-3


def test_the_two_class_program_is_the_gaussian_mixture_row_by_row() -> None:
    program = _svi(program=MIX2, parameters={}, response="y")
    family = _svi("y ~ 1", family=mixture("gaussian", "gaussian", order="none"))
    logits = np.array([[0.3, -0.4]])
    centers = np.array([[-0.5, 1.7]])
    sigma = np.array([0.9])
    weights = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    scores = pin(
        program, {"logits": logits, "centers": centers, "sigma": sigma}
    ).log_likelihood()["y"]
    names = set(latent_names(family))
    assert names == {
        "dpar_mu1_intercept",
        "dpar_mu2_intercept",
        "sigma1",
        "sigma2",
        "theta",
    }
    mixed = pin(
        family,
        {
            "dpar_mu1_intercept": centers[:, 0],
            "dpar_mu2_intercept": centers[:, 1],
            "sigma1": sigma,
            "sigma2": sigma,
            "theta": weights,
        },
    ).log_likelihood()["y"]
    torch.testing.assert_close(scores.double(), mixed.double(), rtol=1e-10, atol=0.0)
    wider = pin(
        family,
        {
            "dpar_mu1_intercept": centers[:, 0],
            "dpar_mu2_intercept": centers[:, 1],
            "sigma1": sigma,
            "sigma2": 2.0 * sigma,
            "theta": weights,
        },
    ).log_likelihood()["y"]
    assert float((scores - wider).abs().max()) > 1e-3
