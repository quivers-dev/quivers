"""brms ``mm(..., cov = K)``: multi-membership terms with a known covariance.

Oracle: ``oracles/brms_membership_covariance.json`` (generator
``oracles/brms_membership_covariance.R``, brms 2.20.4) records brms's Stan
data (member codes, weights, covariates, and the Cholesky factor ``Lcov`` of
the covariance), the generated Stan statements, and ``posterior_linpred`` and
``log_lik`` on training and fresh rows at three draws.

brms's model is ``r = sd * (Lcov z)`` per coefficient, or
``r = scale_r_cor_cov(z, sd, L, Lcov)`` for correlated coefficients, with
standard-Normal ``z``, and each row adds ``sum_k W_k r[J_k] Z_k``. The tests
rebuild that density in torch from brms's own ``Lcov``, ``J``, ``W`` and ``Z``
and compare values and first and second derivatives with the emitted
program.

Tolerance, fixed before comparing: the program reads its design inputs
(covariates, membership weights, and the level factor) in single precision,
a relative error of 6e-8 per input, and a predictor or derivative here is a
sum of at most a few dozen products of two or three inputs of order one. An
absolute tolerance of 1e-6 on predictors and pointwise likelihoods and a
relative tolerance of 1e-5 on densities and derivatives cover that storage
error with a wide margin, and each test also checks that a wrong variant
fails by more than 1e-3.
"""

from __future__ import annotations

import json
import pathlib
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

import quivers.formulas._fit as api
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.dsl import Compiler
from quivers.formulas import (
    BayesianFit,
    FormulaToQVRModule,
    PriorSpec,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.formulas._grouping import resolve_group_covariances
from quivers.formulas.formula import _qvr_name
from quivers.inference.trace import trace
from quivers.modeling import load_fit, refit_callback, save_fit

ORACLE = json.loads(
    (
        pathlib.Path(__file__).parent / "oracles" / "brms_membership_covariance.json"
    ).read_text()
)
LEVELS = ORACLE["levels"]
ATOL = 1e-6
RTOL = 1e-5


def _frame(name: str = "data") -> pd.DataFrame:
    return pd.DataFrame(ORACLE[name])


def _data2() -> dict[str, pd.DataFrame]:
    return {
        name: pd.DataFrame(np.array(ORACLE[name]), index=LEVELS, columns=LEVELS)
        for name in ("K", "P")
    }


def _compiled(text: str, frame: pd.DataFrame):
    """Compile a formula and return its program, lens and bound inputs."""
    formula = formula_from_data(text, frame)
    lens = FormulaToQVRModule(
        families["gaussian"],
        group_covariances=resolve_group_covariances(text, _data2(), None),
    )
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    assert isinstance(program, MonadicProgram)
    observed = dict(lens.fixed_column_observations(formula))
    observed["y"] = torch.tensor(
        np.array(formula.response_values, dtype=np.float64), dtype=torch.float64
    )
    for group, codes in formula.group_indices.items():
        observed[f"{_qvr_name(group)}_idx"] = torch.tensor(codes, dtype=torch.long)
    return formula, lens, program, observed


def _effects(case: dict, draw: int) -> torch.Tensor:
    """brms's group-level effects ``r`` at one draw, levels by coefficients."""
    names = [name for name in case["variables"] if name.startswith("r_mmg1g2[")]
    coefficients = list(dict.fromkeys(name.split(",")[1][:-1] for name in names))
    return torch.tensor(
        [
            [case["draws"][f"r_mmg1g2[{level},{coef}]"][draw] for coef in coefficients]
            for level in LEVELS
        ],
        dtype=torch.float64,
    )


def _brms_mu(case: dict, draw: int, rows: slice) -> torch.Tensor:
    """brms's predictor statement evaluated from its Stan data at one draw."""
    data = case["data"]
    effects = _effects(case, draw)
    mu = case["draws"]["b_Intercept"][draw] + case["draws"]["b_x1"][
        draw
    ] * torch.tensor(_frame()["x1"].tolist(), dtype=torch.float64)
    blocks = sorted({key.split("_")[1] for key in data if key.startswith("J_")})
    column = 0
    for block in blocks:
        coefficients = sorted(
            {key.split("_")[2] for key in data if key.startswith(f"Z_{block}_")}
        )
        for coefficient in coefficients:
            for member in ("1", "2"):
                codes = torch.tensor(data[f"J_{block}_{member}"], dtype=torch.long) - 1
                mu = mu + (
                    torch.tensor(data[f"W_{block}_{member}"], dtype=torch.float64)
                    * effects[codes, column]
                    * torch.tensor(
                        data[f"Z_{block}_{coefficient}_{member}"], dtype=torch.float64
                    )
                )
            column += 1
    return mu[rows]


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["formula"])
def test_level_factor_is_brms_lcov_and_statements_are_pinned(case):
    _, lens, _, observed = _compiled(case["formula"], _frame())
    factor = observed["__formula_level_basis_mmg1g2_factor"].double()
    lcov = torch.tensor(case["Lcov"], dtype=torch.float64)
    # The factor is stored in single precision.
    torch.testing.assert_close(factor, lcov, rtol=0, atol=1e-6)
    correlated = "(1 + x1 |" in case["formula"]
    assert case["effects"] == (
        [
            "r_1 = scale_r_cor_cov(z_1, sd_1, L_1, Lcov_1);",
            "r_1_1 = r_1[, 1];",
            "r_1_2 = r_1[, 2];",
        ]
        if correlated
        else "r_1_1 = (sd_1[1] * (Lcov_1 * z_1[1]));"
    )
    assert case["group"] == "mmg1g2"
    sites = {target.site for target in lens.prior_targets}
    assert ("cor_mmg1g2" in sites) == correlated
    scales = {site for site in sites if site.startswith("sigma_mmg1g2_")}
    assert len(scales) == len(_listed(case["sd"]))


def _listed(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def _sites(case: dict, lens: FormulaToQVRModule, draw: int, factor: torch.Tensor):
    """Map one brms draw to the emitted latent sites."""
    draws = case["draws"]
    effects = _effects(case, draw)
    values: dict[str, torch.Tensor] = {
        "intercept": torch.tensor(draws["b_Intercept"][draw], dtype=torch.float64),
        "beta_x1": torch.tensor(draws["b_x1"][draw], dtype=torch.float64),
        "sigma": torch.tensor(draws["sigma"][draw], dtype=torch.float64),
    }
    scales = [draws[name][draw] for name in case["variables"] if name.startswith("sd_")]
    column = 0
    for layout in lens._group_layouts:
        width = len(layout.members)
        block = effects[:, column : column + width]
        scale = torch.tensor(scales[column : column + width], dtype=torch.float64)
        cholesky = torch.eye(width, dtype=torch.float64)
        if layout.correlation_site is not None:
            rho = draws["cor_mmg1g2__Intercept__x1"][draw]
            cholesky = torch.tensor(
                [[1.0, 0.0], [rho, (1 - rho * rho) ** 0.5]], dtype=torch.float64
            )
            values[layout.correlation_site] = cholesky
        column_factor = scale[:, None] * cholesky
        # r = F z (D L)^T, so z = F^{-1} r (D L)^{-T}.
        raw = torch.linalg.solve(factor, block) @ torch.linalg.inv(column_factor).T
        for member, value in zip(layout.members, scale, strict=True):
            assert member.scale_site is not None and member.raw_site is not None
            values[member.scale_site] = value
            values[member.raw_site] = raw
        column += width
    return values


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["formula"])
def test_predictor_and_pointwise_likelihood_equal_brms_at_its_draws(case):
    stacked = pd.concat([_frame(), _frame("fresh")], ignore_index=True)
    _, lens, program, observed = _compiled(case["formula"], stacked)
    factor = torch.tensor(case["Lcov"], dtype=torch.float64)
    y = observed["y"]
    for draw in range(3):
        values = _sites(case, lens, draw, factor)
        recorded = trace(program, torch.zeros(1, 1), observed | values)
        if draw == 0:
            # Effects mapped through the identity in place of Lcov differ.
            wrong = _sites(case, lens, draw, torch.eye(5, dtype=torch.float64))
            other = trace(program, torch.zeros(1, 1), observed | wrong)
            assert (
                float(
                    (other.sites["eta"].value - recorded.sites["eta"].value).abs().max()
                )
                > 1e-3
            )
        eta = recorded.sites["eta"].value.reshape(-1).double()
        expected = torch.tensor(
            case["linpred"][draw] + case["new_linpred"][draw], dtype=torch.float64
        )
        torch.testing.assert_close(eta, expected, rtol=0, atol=ATOL)
        # brms's own predictor statement, from its Stan data, on training rows.
        torch.testing.assert_close(
            eta[:12], _brms_mu(case, draw, slice(None)), rtol=0, atol=ATOL
        )
        pointwise = torch.distributions.Normal(eta, values["sigma"]).log_prob(y)
        torch.testing.assert_close(
            pointwise,
            torch.tensor(
                case["log_lik"][draw] + case["new_log_lik"][draw], dtype=torch.float64
            ),
            rtol=0,
            atol=ATOL,
        )


def _reference(case: dict, lens: FormulaToQVRModule, observed: dict, beta_scale: float):
    """brms's joint density in the emitted coordinates, written from its data."""
    data = case["data"]
    lcov = torch.tensor(case["Lcov"], dtype=torch.float64)
    x1 = torch.tensor(_frame()["x1"].tolist(), dtype=torch.float64)
    y = observed["y"]
    correlated = any(layout.correlation_site for layout in lens._group_layouts)
    blocks = sorted({key.split("_")[1] for key in data if key.startswith("J_")})

    def density(point: torch.Tensor) -> torch.Tensor:
        normal = torch.distributions.Normal
        intercept, beta, sigma = point[0], point[1], point[2]
        scales = point[3:5]
        raw = point[5:15].reshape(5, 2)
        total = (
            normal(0.0, 5.0).log_prob(intercept)
            + normal(0.0, beta_scale).log_prob(beta)
            + torch.distributions.HalfCauchy(2.0).log_prob(sigma)
            + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
            + normal(0.0, 1.0).log_prob(raw).sum()
        )
        if correlated:
            rho = torch.tanh(point[15])
            cholesky = torch.stack(
                (
                    torch.stack((torch.ones(()).double(), torch.zeros(()).double())),
                    torch.stack((rho, torch.sqrt(1 - rho * rho))),
                )
            )
            # LKJ(1) on the Cholesky factor of a 2 x 2 correlation matrix is
            # uniform in rho: its density in L[1, 1] cancels the Jacobian.
            total = total + torch.distributions.LKJCholesky(2, 1.0).log_prob(cholesky)
        else:
            cholesky = torch.eye(2, dtype=torch.float64)
        effects = lcov @ raw @ (scales[:, None] * cholesky).T
        mu = intercept + beta * x1
        column = 0
        for block in blocks:
            for coefficient in sorted(
                {key.split("_")[2] for key in data if key.startswith(f"Z_{block}_")}
            ):
                for member in ("1", "2"):
                    codes = torch.tensor(data[f"J_{block}_{member}"]) - 1
                    mu = mu + (
                        torch.tensor(data[f"W_{block}_{member}"], dtype=torch.float64)
                        * effects[codes, column]
                        * torch.tensor(
                            data[f"Z_{block}_{coefficient}_{member}"],
                            dtype=torch.float64,
                        )
                    )
                column += 1
        return total + normal(mu, sigma).log_prob(y).sum()

    return density, correlated


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in ORACLE["cases"]
        if case["data"]["M_1"] == 2 or "N_2" in case["data"]
    ],
    ids=lambda case: case["formula"],
)
def test_joint_density_and_derivatives_equal_the_brms_model(case):
    frame = _frame()
    _, lens, program, observed = _compiled(case["formula"], frame)
    text = get_prior(case["formula"], data=frame, data2=_data2())["beta_x1"]
    beta_scale = float(text.split("scale=")[1].rstrip(")"))
    density, correlated = _reference(case, lens, observed, beta_scale)
    # The level factor in double precision, so both sides use brms's Lcov.
    observed["__formula_level_basis_mmg1g2_factor"] = torch.tensor(
        case["Lcov"], dtype=torch.float64
    )
    layouts = lens._group_layouts

    def actual(point: torch.Tensor) -> torch.Tensor:
        values = {
            **observed,
            "intercept": point[0],
            "beta_x1": point[1],
            "sigma": point[2],
        }
        raw = point[5:15].reshape(5, 2)
        column = 0
        for layout in layouts:
            width = len(layout.members)
            for offset, member in enumerate(layout.members):
                assert member.scale_site is not None and member.raw_site is not None
                values[member.scale_site] = point[3 + column + offset]
                values[member.raw_site] = raw[:, column : column + width]
            column += width
            if layout.correlation_site is not None:
                rho = torch.tanh(point[15])
                values[layout.correlation_site] = torch.stack(
                    (
                        torch.stack(
                            (torch.ones(()).double(), torch.zeros(()).double())
                        ),
                        torch.stack((rho, torch.sqrt(1 - rho * rho))),
                    )
                )
        joint = trace(program, point.new_zeros(1, 1), values).log_joint
        assert joint is not None
        return joint.sum()

    generator = torch.Generator().manual_seed(6)
    point = 0.4 * torch.randn(16, generator=generator, dtype=torch.float64)
    point[2:5] = torch.tensor([0.8, 0.6, 1.1], dtype=torch.float64)
    point.requires_grad_()
    torch.testing.assert_close(actual(point), density(point), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(density, point),
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(density, point),
        rtol=RTOL,
        atol=ATOL,
    )
    # The identity in place of Lcov is a different density.
    observed["__formula_level_basis_mmg1g2_factor"] = torch.eye(5, dtype=torch.float64)
    assert abs(float((actual(point) - density(point)).detach())) > 1e-3
    assert correlated == ("(1 + x1 |" in case["formula"])


FORMULA = "y ~ x1 + (1 + x1 | mm(g1, g2, weights = cbind(w1, w2), cov = P))"


def _fit_with(method: Literal["svi", "hmc", "nuts"], **extra) -> BayesianFit:
    if method == "svi":
        return fit(
            FORMULA,
            data=_frame(),
            data2=_data2(),
            method="svi",
            num_steps=3,
            seed=2,
            **extra,
        )
    return fit(
        FORMULA,
        data=_frame(),
        data2=_data2(),
        method=method,
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        seed=2,
        **extra,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_inference_method_predicts_fresh_rows(
    method: Literal["svi", "hmc", "nuts"],
):
    fitted = _fit_with(method)
    fresh = _frame("fresh")
    for draws in (
        fitted.posterior_linpred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_epred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_predict(new_data=fresh, num_samples=2, seed=5),
        *fitted.log_likelihood(new_data=fresh, num_samples=2, seed=5).values(),
    ):
        assert draws.shape[-1] == 4 and torch.isfinite(draws).all()
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=_frame(), num_samples=2, seed=4),
        fitted.posterior_linpred(num_samples=2, seed=4),
    )
    names = list(fitted.variables(num_samples=2))
    assert {
        "sd_mmg1g2__Intercept",
        "sd_mmg1g2__x1",
        "cor_mmg1g2__Intercept__x1",
        "r_mmg1g2[e,x1]",
    } <= set(names)


def test_levels_of_the_covariance_unseen_in_training_follow_their_conditional_law():
    # Training rows without level e: K still covers it.
    frame = _frame()
    rows = [
        index for index in range(12) if "e" not in (frame.g1[index], frame.g2[index])
    ]
    kept = frame.iloc[rows].reset_index(drop=True)
    fitted = fit(
        "y ~ x1 + (1 | mm(g1, g2, cov = K))",
        data=kept,
        data2=_data2(),
        method="svi",
        num_steps=3,
        seed=1,
    )
    fresh = _frame("fresh")
    with pytest.raises(ValueError, match="allow_new_levels"):
        fitted.posterior_epred(new_data=fresh, num_samples=1)
    draws = fitted.posterior_epred(
        new_data=fresh,
        num_samples=3,
        seed=1,
        allow_new_levels=True,
        sample_new_levels="gaussian",
    )
    assert draws.shape[-1] == 4 and torch.isfinite(draws).all()
    # A level outside the covariance has no conditional law.
    with pytest.raises(ValueError, match="no entries for levels"):
        fitted.posterior_epred(
            new_data=fresh.assign(g1="z"),
            num_samples=1,
            allow_new_levels=True,
            sample_new_levels="gaussian",
        )


def test_priors_update_kfold_and_artifact_round_trip(tmp_path, monkeypatch):
    frame = _frame()
    selected = get_prior(
        FORMULA,
        data=frame,
        data2=_data2(),
        priors=PriorSpec(distribution="HalfNormal(0.5)", class_="sd", group="mmg1g2"),
    )
    assert selected["sigma_mmg1g2_Intercept"] == "HalfNormal(scale=0.5)"
    fitted = _fit_with("svi")
    refit = fitted.update(data=frame.iloc[:10].reset_index(drop=True))
    assert refit.formula.response_values.shape == (10,)
    assert refit.group_covariances.keys() == fitted.group_covariances.keys()
    recipe = fitted.refit_recipe
    assert recipe is not None
    callback = refit_callback(
        recipe,
        lambda data, rows: pd.DataFrame(data).iloc[list(rows)].copy(),
        lambda refitted, held, seed: refitted.log_likelihood(
            new_data=held,
            num_samples=2,
            seed=seed,
            allow_new_levels=True,
            sample_new_levels="gaussian",
        ),
        method_factory=lambda settings, seed: {**settings, "seed": seed},
    )
    result = cross_validate(
        FoldPlan(assignments=tuple(i % 2 for i in range(12)), seed=1), callback
    )
    assert result.complete and result.log_predictive_density.shape == (12,)
    fresh = _frame("fresh")
    expected = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=9)
    path = tmp_path / "membership_covariance.qfit"
    save_fit(fitted, path)

    def forbid(*args, **kwargs):
        raise AssertionError("restoring a fit must not run inference")

    monkeypatch.setattr(api, "_run_inference", forbid)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=2, seed=9), expected
    )


def test_invalid_membership_covariances_are_refused():
    frame = _frame()
    small = _data2()["K"].iloc[:4, :4]
    with pytest.raises(ValueError, match="no entries for levels"):
        fit(
            "y ~ x1 + (1 | mm(g1, g2, cov = K))",
            data=frame,
            data2={"K": small},
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="data2"):
        fit("y ~ x1 + (1 | mm(g1, g2, cov = K))", data=frame, method="svi", num_steps=1)
    ordered = frame.assign(
        o=pd.Categorical(
            np.array(["l", "m", "h"])[np.arange(12) % 3],
            categories=["l", "m", "h"],
            ordered=True,
        )
    )
    with pytest.raises(ValueError, match="not category-specific coefficients"):
        fit(
            "o ~ x1 + (cs(1) | mm(g1, g2))",
            data=ordered,
            family="acat",
            method="svi",
            num_steps=1,
        )
