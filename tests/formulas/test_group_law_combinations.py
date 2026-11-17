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
from quivers.formulas import GroupCovariance, fit
from quivers.formulas._fit import _prepare_formula
from quivers.formulas._group_law import group_coefficients
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
