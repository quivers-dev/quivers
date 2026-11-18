"""Group-block laws that combine level and coefficient factors.

Each test fixes a combination of the factors of a group block (level
covariance, Student-t mixing, component axis, row map, parameterization),
checks the emitted program's log density and gradient against a density
written directly in torch at a pinned point, checks that the coefficients
reconstructed from the retained sites equal the same closed form, and fits
the model with a few variational steps.

Tolerance, fixed before comparing: the program reads its design inputs in
single precision, a relative error of 6e-8 per input, and every density
here is a sum of at most a few dozen products of inputs of order one. A
relative and absolute tolerance of 2e-6 covers that storage error; each
density test also checks that dropping the factor under test moves the
density by more than 1e-3.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    GroupCovariance,
    families,
    fit,
    formula_from_data,
)
from quivers.formulas._fit import _prepare_formula
from quivers.formulas._group_law import (
    KnownCovariance,
    MembershipRows,
    group_coefficients,
    support_coefficients,
)
from quivers.formulas._group_prediction import GroupRemap, _remap_covariance_group
from quivers.inference.trace import trace

Parameterization = Literal["centered", "noncentered"]
TOLERANCE = 2e-6
LEVELS = ("a", "b", "c")
FULL = GroupCovariance(
    levels=(*LEVELS, "d"),
    matrix=(
        (1.0, 0.3, 0.1, 0.2),
        (0.3, 1.0, 0.2, 0.1),
        (0.1, 0.2, 1.0, 0.4),
        (0.2, 0.1, 0.4, 1.0),
    ),
)
DEFICIENT = GroupCovariance(
    levels=LEVELS,
    matrix=((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    rank=2,
)


def _frame(rows: int = 12) -> pd.DataFrame:
    generator = np.random.default_rng(31)
    groups = [LEVELS[index % 3] for index in range(rows)]
    return pd.DataFrame(
        {
            "y": generator.normal(size=rows),
            "x": generator.normal(size=rows),
            "g": groups,
            "h": [("u", "v", "w", "z")[index % 4] for index in range(rows)],
            "k3": [
                ("A", "B", "C")[(index * 5 + index // 3) % 3] for index in range(rows)
            ],
        }
    )


def _compiled(text: str, frame: pd.DataFrame, **options):
    prepared = _prepare_formula(text, data=frame, **options)
    lens, formula = prepared.lens, prepared.formula
    module, complement = lens.forward(formula)
    program = Compiler(module).compile().morphism
    assert isinstance(program, MonadicProgram)
    observed = dict(lens.fixed_column_observations(formula))
    observed[formula.response_name] = torch.as_tensor(formula.response_values.copy())
    for _, _, predictor in lens._predictor_items(formula):
        for group, codes in predictor.group_indices.items():
            observed[f"{group}_idx"] = torch.tensor(codes, dtype=torch.long)
    return lens, formula, program, observed, complement


def _log_joint(program: MonadicProgram, values: dict) -> torch.Tensor:
    joint = trace(program, torch.zeros(1, 1), values).log_joint
    assert joint is not None
    return joint.sum()


def _df_log_prior(df: torch.Tensor) -> torch.Tensor:
    """Gamma(2, 0.1) restricted to df > 1, normalized on that interval."""
    two, rate = df.new_tensor(2.0), df.new_tensor(0.1)
    return torch.distributions.Gamma(two, rate).log_prob(df) - torch.log1p(
        -torch.special.gammainc(two, rate)
    )


def _retained(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().reshape(1, 1, *value.shape)
        for name, value in values.items()
    }


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_known_covariance_of_a_factor_only_the_scale_predictor_uses(
    parameterization: Parameterization,
):
    """log sigma_n = a + r[g_n] with r ~ Normal(0, sd^2 V)."""
    frame = _frame()
    lens, formula, program, observed, complement = _compiled(
        "y ~ x",
        frame,
        distributional={"sigma": "~ 1 + (1 | g)"},
        group_covariances={"g": FULL},
        reparameterize=parameterization,
    )
    (law,) = complement.group_laws
    (member,) = law.members
    basis = law.level.basis
    assert basis is not None and member.raw_site is not None
    projection, coordinate, factor = basis.tensors()
    x = torch.as_tensor(frame["x"].to_numpy(dtype=float))
    y, rows = observed["y"], observed["g_idx"]
    latent = {
        "intercept": torch.tensor(0.1, dtype=torch.float64),
        "beta_x": torch.tensor(-0.2, dtype=torch.float64),
        "dpar_sigma_intercept": torch.tensor(-0.3, dtype=torch.float64),
        member.scale.site: torch.tensor(0.6, dtype=torch.float64),
        member.raw_site: torch.tensor([[0.4], [-0.7], [0.9]], dtype=torch.float64),
    }
    latent = {name: value.requires_grad_() for name, value in latent.items()}

    def oracle(known: bool = True) -> torch.Tensor:
        normal = torch.distributions.Normal
        sd, raw = latent[member.scale.site], latent[member.raw_site]
        identity = torch.eye(3, dtype=torch.float64)
        if parameterization == "centered":
            effects = ((projection if known else identity) @ raw)[:, 0]
            prior = torch.distributions.MultivariateNormal(
                torch.zeros(3, dtype=torch.float64),
                covariance_matrix=(coordinate if known else identity) * sd.square(),
            ).log_prob(raw[:, 0])
        else:
            effects = ((factor if known else identity) @ raw)[:, 0] * sd
            prior = normal(0.0, 1.0).log_prob(raw).sum()
        sigma = torch.exp(latent["dpar_sigma_intercept"] + effects[rows])
        return (
            normal(0.0, 5.0).log_prob(latent["intercept"])
            + normal(0.0, 5.0 / x.square().mean().sqrt()).log_prob(latent["beta_x"])
            + normal(0.0, 5.0).log_prob(latent["dpar_sigma_intercept"])
            + torch.distributions.HalfNormal(1.0).log_prob(sd)
            + prior
            + normal(latent["intercept"] + latent["beta_x"] * x, sigma)
            .log_prob(y)
            .sum()
        )

    actual = _log_joint(program, observed | latent)
    torch.testing.assert_close(actual, oracle(), rtol=TOLERANCE, atol=TOLERANCE)
    assert abs(float(oracle(known=False) - actual)) > 1e-3
    heads = tuple(latent.values())
    for got, expected in zip(
        torch.autograd.grad(actual, heads, retain_graph=True),
        torch.autograd.grad(oracle(), heads),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    # The implied covariance of the level coefficients is sd^2 V.
    sd = latent[member.scale.site].detach()
    implied = (factor * sd) @ (factor * sd).T
    torch.testing.assert_close(implied, sd.square() * FULL.tensor()[:3, :3])


def test_scale_predictor_known_covariance_fits_and_predicts_new_levels():
    frame = _frame(24)
    fitted = fit(
        "y ~ x",
        data=frame,
        distributional={"sigma": "~ 1 + (1 | g)"},
        group_covariances={"g": FULL},
        method="svi",
        num_steps=3,
        seed=2,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"])
    draws = fitted.posterior_predict(
        new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
    )
    assert draws.shape[-1] == 3 and torch.isfinite(draws).all()


@pytest.mark.parametrize("covariance", [FULL, DEFICIENT], ids=["full", "deficient"])
def test_student_coefficients_with_a_known_level_covariance(covariance):
    """B = diag(sqrt(df * udf)) . F Z . (diag(sd) L)^T for the support factor F."""
    frame = _frame()
    lens, formula, program, observed, complement = _compiled(
        "y ~ x + (1 + x | gr(g, dist = 'student'))",
        frame,
        group_covariances={"g": covariance},
    )
    (law,) = complement.group_laws
    basis = law.level.basis
    assert basis is not None and law.level.mixing is not None
    _, _, factor = basis.tensors()
    x = torch.as_tensor(frame["x"].to_numpy(dtype=float))
    y = observed["y"]
    rows = observed["g_idx"]
    generator = torch.Generator().manual_seed(3)
    latent = {
        "intercept": torch.tensor(0.1, dtype=torch.float64),
        "beta_x": torch.tensor(-0.2, dtype=torch.float64),
        "sigma": torch.tensor(0.8, dtype=torch.float64),
        "df_g": torch.tensor(3.5, dtype=torch.float64),
        "udf_g": torch.tensor([0.4, 0.7, 1.1], dtype=torch.float64),
        "sigma_g_Intercept": torch.tensor(0.7, dtype=torch.float64),
        "sigma_g_x": torch.tensor(0.5, dtype=torch.float64),
        f"cor_{law.identity}": torch.tensor(
            [[1.0, 0.0], [0.3, (1 - 0.3**2) ** 0.5]], dtype=torch.float64
        ),
        f"group_covariance_{law.identity}": torch.randn(
            basis.rank, 2, generator=generator, dtype=torch.float64
        ),
    }
    latent = {name: value.requires_grad_() for name, value in latent.items()}

    def coefficients(mixing: bool) -> torch.Tensor:
        scale = torch.stack([latent["sigma_g_Intercept"], latent["sigma_g_x"]])
        gaussian = (
            factor
            @ latent[f"group_covariance_{law.identity}"]
            @ (scale[:, None] * latent[f"cor_{law.identity}"]).T
        )
        if not mixing:
            return gaussian
        return torch.sqrt(latent["df_g"] * latent["udf_g"])[:, None] * gaussian

    def oracle(mixing: bool = True) -> torch.Tensor:
        normal = torch.distributions.Normal
        effects = coefficients(mixing)
        mu = (
            latent["intercept"]
            + latent["beta_x"] * x
            + effects[rows, 0]
            + effects[rows, 1] * x
        )
        scale = torch.stack([latent["sigma_g_Intercept"], latent["sigma_g_x"]])
        return (
            normal(0.0, 5.0).log_prob(latent["intercept"])
            + normal(0.0, 5.0 / x.square().mean().sqrt()).log_prob(latent["beta_x"])
            + torch.distributions.HalfCauchy(2.0).log_prob(latent["sigma"])
            + _df_log_prior(latent["df_g"])
            + torch.distributions.InverseGamma(
                latent["df_g"] / 2, latent["df_g"].new_tensor(0.5)
            )
            .log_prob(latent["udf_g"])
            .sum()
            + torch.distributions.HalfNormal(1.0).log_prob(scale).sum()
            + torch.distributions.LKJCholesky(
                2, torch.tensor(1.0, dtype=torch.float64)
            ).log_prob(latent[f"cor_{law.identity}"])
            + normal(0.0, 1.0)
            .log_prob(latent[f"group_covariance_{law.identity}"])
            .sum()
            + normal(mu, latent["sigma"]).log_prob(y).sum()
        )

    actual = _log_joint(program, observed | latent)
    torch.testing.assert_close(actual, oracle(), rtol=TOLERANCE, atol=TOLERANCE)
    assert abs(float(oracle(mixing=False) - actual)) > 1e-3
    heads = tuple(latent.values())
    for got, expected in zip(
        torch.autograd.grad(actual, heads, retain_graph=True),
        torch.autograd.grad(oracle(), heads),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    rebuilt = group_coefficients(
        _retained(latent), complement.group_laws, formula.group_levels
    )
    expected = coefficients(True).detach()
    torch.testing.assert_close(
        rebuilt[("", "g", "Intercept")].reshape(3), expected[:, 0]
    )
    torch.testing.assert_close(rebuilt[("", "g", "x")].reshape(3), expected[:, 1])


def test_student_known_covariance_fits_and_predicts_new_levels():
    frame = _frame(24)
    fitted = fit(
        "y ~ x + (1 + x | gr(g, dist = 'student'))",
        data=frame,
        group_covariances={"g": FULL},
        method="svi",
        num_steps=3,
        seed=2,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"])
    draws = fitted.posterior_predict(
        new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
    )
    assert draws.shape[-1] == 3 and torch.isfinite(draws).all()
    effects = fitted.ranef()
    assert effects is not None


def _simplex_frame(rows: int = 24) -> pd.DataFrame:
    generator = np.random.default_rng(41)
    parts = generator.dirichlet([2.0, 3.0, 4.0], size=rows)
    return _frame(rows).assign(p1=parts[:, 0], p2=parts[:, 1], p3=parts[:, 2])


@pytest.mark.parametrize(
    "bar,covariance",
    [("|", FULL), ("|", None), ("||", None)],
    ids=["correlated-known", "correlated", "independent"],
)
def test_student_coefficients_over_the_response_categories(
    bar: str, covariance: GroupCovariance | None
):
    """One coefficient per free category: B = diag(dfm) . L_V Z . (diag(sd) L)^T."""
    frame = _frame(24)
    parsed = formula_from_data(f"k3 ~ x + (1 {bar} gr(g, dist = 'student'))", frame)
    lens = FormulaToQVRModule(
        families["categorical"],
        group_covariances={"g": covariance} if covariance is not None else None,
    )
    module, complement = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    assert isinstance(program, MonadicProgram)
    observed = dict(lens.fixed_column_observations(parsed))
    observed["k3"] = torch.as_tensor(parsed.response_values.copy())
    observed["g_idx"] = torch.tensor(parsed.group_indices["g"], dtype=torch.long)
    (law,) = complement.group_laws
    (member,) = law.members
    basis = law.level.basis
    assert law.level.mixing is not None
    assert member.axis is not None and member.width == 2
    raw_site = member.raw_site
    assert raw_site is not None
    cholesky = torch.tensor(
        [[1.0, 0.0], [0.3, (1 - 0.3**2) ** 0.5]], dtype=torch.float64
    )
    sites = law.correlation.sites
    assert len(sites) == (0 if bar == "||" else 1)
    if basis is not None:
        _, _, factor = basis.tensors()
        shape: tuple[int, ...] = (basis.rank, 2)
    else:
        factor = torch.eye(3, dtype=torch.float64)
        shape = (6,)
    x = torch.as_tensor(frame["x"].to_numpy(dtype=float))
    y = observed["k3"].long()
    rows = observed["g_idx"]
    generator = torch.Generator().manual_seed(5)
    latent = {
        "intercept": torch.tensor([0.1, -0.3], dtype=torch.float64),
        "beta_x": torch.tensor([-0.2, 0.4], dtype=torch.float64),
        "df_g": torch.tensor(3.5, dtype=torch.float64),
        "udf_g": torch.tensor([0.4, 0.7, 1.1], dtype=torch.float64),
        member.scale.site: torch.tensor([0.7, 0.5], dtype=torch.float64),
        raw_site: torch.randn(shape, generator=generator, dtype=torch.float64),
    }
    if sites:
        latent[sites[0]] = cholesky
    latent = {name: value.requires_grad_() for name, value in latent.items()}

    def coefficients(mixing: bool) -> torch.Tensor:
        correlation = latent[sites[0]] if sites else torch.eye(2, dtype=torch.float64)
        # A flat raw site holds the levels in order, each with its components.
        gaussian = (
            factor
            @ latent[raw_site].reshape(-1, 2)
            @ (latent[member.scale.site][:, None] * correlation).T
        )
        if not mixing:
            return gaussian
        return torch.sqrt(latent["df_g"] * latent["udf_g"])[:, None] * gaussian

    def oracle(mixing: bool = True) -> torch.Tensor:
        normal = torch.distributions.Normal
        effects = coefficients(mixing)
        free = latent["intercept"] + latent["beta_x"] * x[:, None] + effects[rows]
        logits = torch.cat([torch.zeros(len(x), 1, dtype=torch.float64), free], dim=1)
        return (
            normal(0.0, 5.0).log_prob(latent["intercept"]).sum()
            + normal(0.0, 5.0 / x.square().mean().sqrt())
            .log_prob(latent["beta_x"])
            .sum()
            + _df_log_prior(latent["df_g"])
            + torch.distributions.InverseGamma(
                latent["df_g"] / 2, latent["df_g"].new_tensor(0.5)
            )
            .log_prob(latent["udf_g"])
            .sum()
            + torch.distributions.HalfNormal(1.0)
            .log_prob(latent[member.scale.site])
            .sum()
            + (
                torch.distributions.LKJCholesky(
                    2, torch.tensor(1.0, dtype=torch.float64)
                ).log_prob(latent[sites[0]])
                if sites
                else 0.0
            )
            + normal(0.0, 1.0).log_prob(latent[raw_site]).sum()
            + torch.distributions.Categorical(logits=logits).log_prob(y).sum()
        )

    actual = _log_joint(program, observed | latent)
    torch.testing.assert_close(actual, oracle(), rtol=TOLERANCE, atol=TOLERANCE)
    assert abs(float(oracle(mixing=False) - actual)) > 1e-3
    heads = tuple(latent.values())
    for got, expected in zip(
        torch.autograd.grad(actual, heads, retain_graph=True),
        torch.autograd.grad(oracle(), heads),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    rebuilt = group_coefficients(
        _retained(latent), complement.group_laws, parsed.group_levels
    )
    torch.testing.assert_close(
        rebuilt[("", "g", "Intercept")].reshape(3, 2), coefficients(True).detach()
    )


def test_student_category_known_covariance_fits_and_predicts_new_levels():
    frame = _simplex_frame()
    fitted = fit(
        "cbind(p1, p2, p3) ~ x + (1 | gr(g, dist = 'student'))",
        data=frame,
        family="dirichlet",
        group_covariances={"g": FULL},
        method="svi",
        num_steps=3,
        seed=2,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"])
    draws = fitted.posterior_predict(
        new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
    )
    assert torch.isfinite(torch.as_tensor(draws, dtype=torch.float64)).all()


OFF_MAIN = {
    "scale": ("y ~ x", {"distributional": {"sigma": "~ 1 + (1 | g)"}}),
    "nonlinear": (
        "y ~ a * exp(b * x)",
        {"nonlinear": {"a": "~ 1 + (1 | g)", "b": "~ 1"}},
    ),
    "category": (
        "k3 ~ x",
        {
            "family": "categorical",
            "distributional": {"muB": "~ x + (1 | g)", "muC": "~ x"},
        },
    ),
}


@pytest.mark.parametrize("kind", sorted(OFF_MAIN))
@pytest.mark.parametrize("covariance", [FULL, DEFICIENT], ids=["full", "deficient"])
def test_off_main_factor_coefficients_have_covariance_sd_squared_times_known(
    kind: str, covariance: GroupCovariance
):
    """Unit innovations give B B^T = sd^2 V on the fitted levels."""
    text, options = OFF_MAIN[kind]
    _, formula, _, _, complement = _compiled(
        text, _frame(), group_covariances={"g": covariance}, **options
    )
    (law,) = complement.group_laws
    (member,) = law.members
    basis = law.level.basis
    assert basis is not None and member.raw_site is not None
    sd = torch.tensor(0.6, dtype=torch.float64)
    columns = []
    for coordinate in range(basis.rank):
        unit = torch.zeros(basis.rank, 1, dtype=torch.float64)
        unit[coordinate] = 1.0
        rebuilt = group_coefficients(
            _retained({member.scale.site: sd, member.raw_site: unit}),
            complement.group_laws,
            {"g": LEVELS},
        )
        columns.append(rebuilt[(member.prefix, "g", "Intercept")].reshape(3))
    coefficients = torch.stack(columns, dim=1)
    torch.testing.assert_close(
        coefficients @ coefficients.T,
        sd.square() * covariance.tensor()[:3, :3],
        atol=1e-12,
        rtol=1e-12,
    )
    assert formula.response_name


@pytest.mark.parametrize("kind", ["nonlinear", "category"])
def test_off_main_factor_known_covariance_fits(kind: str):
    text, options = OFF_MAIN[kind]
    fitted = fit(
        text,
        data=_frame(24),
        group_covariances={"g": FULL},
        method="svi",
        num_steps=3,
        seed=2,
        **options,
    )
    draws = fitted.posterior_predict(num_samples=2, seed=1)
    assert torch.isfinite(torch.as_tensor(draws, dtype=torch.float64)).all()


def test_student_scale_coefficients_over_components_fit_and_predict_new_levels():
    frame = _simplex_frame()
    fitted = fit(
        "cbind(p1, p2, p3) ~ x",
        data=frame,
        family="logistic_normal",
        distributional={"sigma": "~ 1 + (1 | gr(g, dist = 'student'))"},
        method="svi",
        num_steps=3,
        seed=2,
    )
    assert fitted.prediction_plan is not None
    (law,) = fitted.prediction_plan.group_laws
    (member,) = law.members
    assert member.axis is not None and law.level.mixing is not None
    fresh = frame.iloc[:3].assign(g=["a", "new", "c"])
    draws = fitted.posterior_predict(
        new_data=fresh, num_samples=2, seed=1, allow_new_levels=True
    )
    assert torch.isfinite(torch.as_tensor(draws, dtype=torch.float64)).all()


@pytest.mark.parametrize("bar", ["|", "||"], ids=["correlated", "independent"])
def test_multi_membership_coefficients_over_the_response_categories(bar: str):
    """eta[n, c] = a_c + b_c x_n + sum_k B[J_k[n], c] C_k[n]."""
    frame = _frame(24)
    parsed = formula_from_data(f"k3 ~ x + (1 {bar} mm(g, h))", frame)
    lens = FormulaToQVRModule(families["categorical"])
    module, complement = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    assert isinstance(program, MonadicProgram)
    observed = dict(lens.fixed_column_observations(parsed))
    observed["k3"] = torch.as_tensor(parsed.response_values.copy())
    (law,) = complement.group_laws
    (member,) = law.members
    assert member.axis is not None and member.width == 2
    assert isinstance(member.rows, MembershipRows)
    levels = len(parsed.group_levels[law.group])
    raw_site = member.raw_site
    assert raw_site is not None
    sites = law.correlation.sites
    x = torch.as_tensor(frame["x"].to_numpy(dtype=float))
    y = observed["k3"].long()
    codes = [observed[name].long() for name in member.rows.indices]
    weights = [observed[name].to(torch.float64) for name in member.rows.covariates]
    assert len(codes) == 2
    generator = torch.Generator().manual_seed(7)
    latent = {
        "intercept": torch.tensor([0.1, -0.3], dtype=torch.float64),
        "beta_x": torch.tensor([-0.2, 0.4], dtype=torch.float64),
        member.scale.site: torch.tensor([0.7, 0.5], dtype=torch.float64),
        raw_site: torch.randn(levels * 2, generator=generator, dtype=torch.float64),
    }
    if sites:
        latent[sites[0]] = torch.tensor(
            [[1.0, 0.0], [0.3, (1 - 0.3**2) ** 0.5]], dtype=torch.float64
        )
    latent = {name: value.requires_grad_() for name, value in latent.items()}

    def oracle(members: int = 2) -> torch.Tensor:
        normal = torch.distributions.Normal
        correlation = latent[sites[0]] if sites else torch.eye(2, dtype=torch.float64)
        effects = (
            latent[raw_site].reshape(levels, 2)
            @ (latent[member.scale.site][:, None] * correlation).T
        )
        free = latent["intercept"] + latent["beta_x"] * x[:, None]
        for index, weight in list(zip(codes, weights, strict=True))[:members]:
            free = free + effects[index] * weight[:, None]
        logits = torch.cat([torch.zeros(len(x), 1, dtype=torch.float64), free], dim=1)
        return (
            normal(0.0, 5.0).log_prob(latent["intercept"]).sum()
            + normal(0.0, 5.0 / x.square().mean().sqrt())
            .log_prob(latent["beta_x"])
            .sum()
            + torch.distributions.HalfNormal(1.0)
            .log_prob(latent[member.scale.site])
            .sum()
            + (
                torch.distributions.LKJCholesky(
                    2, torch.tensor(1.0, dtype=torch.float64)
                ).log_prob(latent[sites[0]])
                if sites
                else 0.0
            )
            + normal(0.0, 1.0).log_prob(latent[raw_site]).sum()
            + torch.distributions.Categorical(logits=logits).log_prob(y).sum()
        )

    actual = _log_joint(program, observed | latent)
    torch.testing.assert_close(actual, oracle(), rtol=TOLERANCE, atol=TOLERANCE)
    assert abs(float(oracle(members=1) - actual)) > 1e-3
    heads = tuple(latent.values())
    for got, expected in zip(
        torch.autograd.grad(actual, heads, retain_graph=True),
        torch.autograd.grad(oracle(), heads),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "text,options",
    [
        (
            "cbind(p1, p2, p3) ~ x",
            {
                "family": "logistic_normal",
                "distributional": {"sigma": "~ 1 + (1 | mm(g, h))"},
            },
        ),
        ("o ~ x + (cs(x) | mm(g, h))", {"family": "acat"}),
        (
            "o ~ x + (cs(x) || mm(g, h))",
            {"family": "acat", "reparameterize": "centered"},
        ),
    ],
    ids=["scale-components", "category-specific", "category-specific-centered"],
)
def test_multi_membership_component_coefficients_fit_and_predict(
    text: str, options: dict
):
    frame = _simplex_frame().assign(o=[index % 4 for index in range(24)])
    fitted = fit(text, data=frame, method="svi", num_steps=3, seed=2, **options)
    assert fitted.prediction_plan is not None
    members = [
        member for law in fitted.prediction_plan.group_laws for member in law.members
    ]
    assert any(
        member.axis is not None and isinstance(member.rows, MembershipRows)
        for member in members
    )
    draws = fitted.posterior_predict(new_data=frame.iloc[:4], num_samples=2, seed=1)
    assert torch.isfinite(torch.as_tensor(draws, dtype=torch.float64)).all()


BY = {"a": "m", "b": "n", "c": "m", "d": "n"}
BY_CHOLESKY = (
    ((1.0, 0.0), (0.3, (1 - 0.3**2) ** 0.5)),
    ((1.0, 0.0), (-0.5, (1 - 0.5**2) ** 0.5)),
)


def _by_frame(rows: int = 12) -> pd.DataFrame:
    frame = _frame(rows)
    # by is constant within a level of g.
    return frame.assign(fb=[BY[level] for level in frame["g"]])


def _by_latent(laws, basis) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(13)
    latent = {
        "intercept": torch.tensor(0.1, dtype=torch.float64),
        "beta_x": torch.tensor(-0.2, dtype=torch.float64),
        "sigma": torch.tensor(0.8, dtype=torch.float64),
        "sigma_g_Intercept": torch.tensor([0.7, 1.2], dtype=torch.float64),
        "sigma_g_x": torch.tensor([0.5, 0.9], dtype=torch.float64),
    }
    for law in laws:
        for site, cholesky in zip(law.correlation.sites, BY_CHOLESKY, strict=False):
            latent[site] = torch.tensor(cholesky, dtype=torch.float64)
        latent[law.members[0].raw_site] = torch.randn(
            basis.rank, law.width, generator=generator, dtype=torch.float64
        )
    return latent


def _by_coefficients(laws, latent, factor, codes) -> torch.Tensor:
    """Row j is diag(sd[by_j]) L[by_j] (F Z)_j, written without the library."""
    columns = []
    for law in laws:
        gaussian = factor @ latent[law.members[0].raw_site]
        if law.correlation.sites:
            cholesky = torch.stack([latent[site] for site in law.correlation.sites])
            gaussian = (cholesky[codes] @ gaussian[:, :, None]).squeeze(-1)
        scale = torch.stack(
            [latent[member.scale.site] for member in law.members], dim=1
        )
        columns.append(scale[codes] * gaussian)
    return torch.cat(columns, dim=1)


@pytest.mark.parametrize("bar", ["|", "||"], ids=["correlated", "independent"])
@pytest.mark.parametrize("covariance", [FULL, DEFICIENT], ids=["full", "deficient"])
def test_by_partition_with_a_known_level_covariance(covariance, bar: str):
    """Cov(B_j, B_l) = V[j, l] A[by_j] A[by_l]^T with A[b] = diag(sd[b]) L[b]."""
    frame = _by_frame()
    lens, formula, program, observed, complement = _compiled(
        f"y ~ x + (1 + x {bar} gr(g, by = fb))",
        frame,
        group_covariances={"g": covariance},
    )
    laws = complement.group_laws
    assert len(laws) == (1 if bar == "|" else 2)
    by, basis = laws[0].level.by, laws[0].level.basis
    assert by is not None and basis is not None and by.levels == ("m", "n")
    assert by.codes == (0, 1, 0)
    codes = torch.tensor(by.codes)
    _, _, factor = basis.tensors()
    x = torch.as_tensor(frame["x"].to_numpy(dtype=float))
    y, rows = observed["y"], observed["g_idx"]
    latent = {
        name: value.requires_grad_() for name, value in _by_latent(laws, basis).items()
    }

    def oracle(partitioned: bool = True) -> torch.Tensor:
        normal = torch.distributions.Normal
        effects = _by_coefficients(
            laws, latent, factor, codes if partitioned else torch.zeros_like(codes)
        )
        mu = (
            latent["intercept"]
            + latent["beta_x"] * x
            + effects[rows, 0]
            + effects[rows, 1] * x
        )
        total = (
            normal(0.0, 5.0).log_prob(latent["intercept"])
            + normal(0.0, 5.0 / x.square().mean().sqrt()).log_prob(latent["beta_x"])
            + torch.distributions.HalfCauchy(2.0).log_prob(latent["sigma"])
            + torch.distributions.HalfNormal(1.0)
            .log_prob(latent["sigma_g_Intercept"])
            .sum()
            + torch.distributions.HalfNormal(1.0).log_prob(latent["sigma_g_x"]).sum()
            + normal(mu, latent["sigma"]).log_prob(y).sum()
        )
        for law in laws:
            total = (
                total + normal(0.0, 1.0).log_prob(latent[law.members[0].raw_site]).sum()
            )
            for site in law.correlation.sites:
                total = total + torch.distributions.LKJCholesky(
                    2, torch.tensor(1.0, dtype=torch.float64)
                ).log_prob(latent[site])
        return total

    actual = _log_joint(program, observed | latent)
    torch.testing.assert_close(actual, oracle(), rtol=TOLERANCE, atol=TOLERANCE)
    assert abs(float((oracle(partitioned=False) - actual).detach())) > 1e-3
    heads = tuple(latent.values())
    for got, expected in zip(
        torch.autograd.grad(actual, heads, retain_graph=True),
        torch.autograd.grad(oracle(), heads),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    # Closed form: the linear map from unit innovations to coefficients has
    # M M^T with blocks V[j, l] A[by_j] A[by_l]^T.
    detached = {name: value.detach() for name, value in latent.items()}
    images = []
    for law in laws:
        site = law.members[0].raw_site
        for index in range(basis.rank * law.width):
            unit = torch.zeros(basis.rank * law.width, dtype=torch.float64)
            unit[index] = 1.0
            draws = dict(detached)
            for other in laws:
                draws[other.members[0].raw_site] = torch.zeros(
                    basis.rank, other.width, dtype=torch.float64
                )
            draws[site] = unit.reshape(basis.rank, law.width)
            rebuilt = group_coefficients(_retained(draws), laws, {"g": LEVELS})
            images.append(
                torch.stack(
                    [
                        rebuilt[("", "g", "Intercept")].reshape(3),
                        rebuilt[("", "g", "x")].reshape(3),
                    ],
                    dim=1,
                ).reshape(-1)
            )
    linear = torch.stack(images, dim=1)
    known = covariance.tensor()[:3, :3]
    columns = []
    for level in range(2):
        scale = torch.stack(
            [detached["sigma_g_Intercept"][level], detached["sigma_g_x"][level]]
        )
        cholesky = (
            torch.tensor(BY_CHOLESKY[level], dtype=torch.float64)
            if bar == "|"
            else torch.eye(2, dtype=torch.float64)
        )
        columns.append(scale[:, None] * cholesky)
    expected = torch.zeros(6, 6, dtype=torch.float64)
    for first in range(3):
        for second in range(3):
            expected[2 * first : 2 * first + 2, 2 * second : 2 * second + 2] = (
                known[first, second]
                * columns[by.codes[first]]
                @ columns[by.codes[second]].T
            )
    torch.testing.assert_close(linear @ linear.T, expected, atol=1e-12, rtol=1e-12)
    assert formula.response_name == "y" and lens is not None


def test_new_level_of_a_by_partition_takes_the_column_factor_of_its_by_level():
    """With V[d, a] = V[a, a], level d repeats G_a under its own by level."""
    frame = _by_frame()
    _, _, _, _, complement = _compiled(
        "y ~ x + (1 + x | gr(g, by = fb))", frame, group_covariances={"g": FULL}
    )
    (law,) = complement.group_laws
    by, basis = law.level.by, law.level.basis
    assert by is not None and basis is not None
    repeated = GroupCovariance(
        levels=("a", "b", "c", "d"),
        matrix=(
            (1.0, 0.3, 0.1, 1.0),
            (0.3, 1.0, 0.2, 0.3),
            (0.1, 0.2, 1.0, 0.1),
            (1.0, 0.3, 0.1, 1.0),
        ),
        rank=3,
    )
    mean_map, conditional = repeated.conditional(LEVELS, ("d",))
    assert conditional.rank == 0
    compact_levels = ("a", "d", "c")
    target = repeated.basis(compact_levels)
    group = GroupRemap(
        group="g",
        fitted_size=3,
        fitted_indices=(0, -1, 2),
        row_indices=torch.tensor([0, 1, 2]),
        sites=law.layout.raw_sites,
        levels=compact_levels,
        covariance=repeated,
        covariance_laws=(law,),
        covariance_target=target,
        covariance_conditional=conditional,
        covariance_mean_map=tuple(tuple(row) for row in mean_map.tolist()),
    )
    latent = _by_latent((law,), basis)
    compact_codes = torch.tensor([0, 1, 0])
    site = law.members[0].raw_site
    assert site is not None
    mapped = _remap_covariance_group(
        group,
        latent,
        {site: (target.rank, 2)},
        "gaussian",
        {},
        {by.input: compact_codes},
    )
    compact = law.with_(
        level=law.level.with_(
            covariance=KnownCovariance(basis=target),
            by=by.with_(codes=tuple(compact_codes.tolist())),
        )
    )
    physical = support_coefficients(latent | mapped, compact, sample_ndims=0)
    gaussian = basis.tensors()[2] @ latent[site]
    cholesky = torch.tensor(BY_CHOLESKY, dtype=torch.float64)
    scale = torch.stack([latent["sigma_g_Intercept"], latent["sigma_g_x"]], dim=1)
    rows = gaussian[[0, 0, 2]]
    expected = scale[compact_codes] * (
        cholesky[compact_codes] @ rows[:, :, None]
    ).squeeze(-1)
    torch.testing.assert_close(physical, expected, atol=1e-10, rtol=1e-10)
    # The new level differs from the level it repeats only by its by level.
    assert (physical[1] - physical[0]).abs().max() > 1e-3


@pytest.mark.parametrize("bar", ["|", "||"], ids=["correlated", "independent"])
def test_by_known_covariance_fits_and_predicts_new_levels(bar: str):
    frame = _by_frame(24)
    fitted = fit(
        f"y ~ x + (1 + x {bar} gr(g, by = fb))",
        data=frame,
        group_covariances={"g": FULL},
        method="svi",
        num_steps=3,
        seed=2,
    )
    fresh = frame.iloc[:3].assign(g=["a", "d", "c"], fb=["m", "n", "m"])
    for policy in ("gaussian", "uncertainty"):
        draws = fitted.posterior_predict(
            new_data=fresh,
            num_samples=2,
            seed=1,
            allow_new_levels=True,
            sample_new_levels=policy,
        )
        assert draws.shape[-1] == 3 and torch.isfinite(draws).all()
    assert fitted.ranef() is not None
