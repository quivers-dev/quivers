"""A covariate may have any name a program uses for something else."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl.ast_nodes import (
    LetExprFactor,
    LetStep,
    ObjectDecl,
    ProgramDecl,
    SampleStep,
)
from quivers.dsl.let_expr_traversal import walk_let_expr
from quivers.dsl.program_templates import _bound_names
from quivers.formulas import BayesianFit, bf, fit, mixture
from quivers.formulas._term import ColumnName, resolve_names
from quivers.formulas.formula import _qvr_name
from quivers.inference.mcmc import PotentialFn
from quivers.inference.registry import LatentRegistry

ROWS = 24
FRAME = pd.DataFrame(
    {
        "y": np.random.default_rng(1).normal(size=ROWS).round(3),
        "o": [0, 1, 2, 3] * 6,
        "c": ["A", "B", "C"] * 8,
        "x": np.random.default_rng(3).normal(size=ROWS).round(3),
        "x2": np.random.default_rng(4).normal(size=ROWS).round(3),
        "w": np.linspace(0.5, 1.5, ROWS),
        "s": np.full(ROWS, 0.2),
        "z": [0, 1, 2, 3, 2, 1] * 4,
        "g": ["a", "b", "c"] * 8,
    }
)
EXTRA = np.random.default_rng(9).normal(size=ROWS).round(3)

#: Models whose programs together use every kind of name.
MODELS: dict[str, Callable[[str], BayesianFit]] = {}


def _model(name: str, formula: str, **options) -> None:
    def build(column: str) -> BayesianFit:
        data = FRAME.assign(**{column: EXTRA})
        fitted = fit(
            formula.replace("COLUMN", column),
            data=data,
            method="svi",
            num_steps=1,
            seed=0,
            **{
                key: value() if callable(value) else value
                for key, value in options.items()
            },
        )
        assert isinstance(fitted, BayesianFit)
        return fitted

    MODELS[name] = build


_model(
    "terms",
    "y | weights(w) ~ x + COLUMN + offset(w) + s(x2, k = 4) + gp(x, k = 3, c = 1.25)"
    " + mo(z) + (1 + x | g)",
)
_model("error", "y | se(s, sigma = TRUE) ~ x + COLUMN + (1 + x || g)")
_model("ordinal", "o ~ x + COLUMN + cs(x2) + (cs(x) | g)", family="acat")
_model("categorical", "c ~ x + COLUMN + s(x2, k = 4) + (1 | g)", family="categorical")
_model("dpar", "y ~ x + COLUMN", distributional=lambda: {"sigma": "~ x2 + (1 | g)"})
_model("mixture", "y ~ x + COLUMN", family=lambda: mixture("gaussian", "student_t"))
_model("brms", "y ~ x + COLUMN", default_priors="brms", decomp="QR")


def _names(fitted: BayesianFit) -> set[str]:
    assert fitted.prediction_plan is not None
    module = fitted.prediction_plan.module
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    names = set(_bound_names(program.draws))
    names |= {parameter.name for parameter in program.value_parameters}
    names |= set(fitted.observations)
    names |= {
        name
        for item in module.statements
        if isinstance(item, ObjectDecl)
        for name in item.names
    }
    names |= {
        binder.var
        for step in program.draws
        if isinstance(step, LetStep)
        for node in walk_let_expr(step.value)
        if isinstance(node, LetExprFactor)
        for binder in node.binders
    }
    return names


def _density(fitted: BayesianFit, rename: Callable[[str], str]) -> float:
    registry = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    )
    potential = PotentialFn(
        fitted.program, registry, torch.zeros(1, 1), dict(fitted.observations)
    )
    parts = registry.unflatten_unconstrained(
        torch.zeros(registry.total_unconstrained_dim)
    )
    values = {}
    for name, value in parts.items():
        seed = sum(ord(letter) * 31**k for k, letter in enumerate(rename(name)))
        generator = torch.Generator().manual_seed(seed % 2**31)
        values[name] = 0.4 * torch.randn(value.shape, generator=generator)
    flat = registry.flatten_unconstrained(values)
    return float(potential.log_density(flat))


@pytest.mark.parametrize("name", MODELS)
def test_a_covariate_may_take_every_name_of_the_program(name):
    """The density equals that of the model whose covariate has a free name."""
    build = MODELS[name]
    reference = build("zz")
    expected = _density(reference, lambda site: site)
    # A column with the response's name would replace the response, and the
    # authored variables of the formula are columns already.
    authored = {"y", "o", "c", "x", "x2", "w", "s", "z", "g", "zz"}
    authored |= {column.qvr_name for column in reference.formula.fixed_columns}
    reserved = sorted(
        item
        for item in _names(reference)
        if item not in authored
        and "zz" not in item
        and re.fullmatch(r"[A-Za-z]\w*", item)
    )
    assert len(reserved) > 5
    for item in reserved:
        fitted = build(item)
        assert item == _qvr_name(item)

        def restore(site: str, item: str = item) -> str:
            return re.sub(rf"beta_{re.escape(item)}(?=$|_qr$)", "beta_zz", site)

        assert _density(fitted, restore) == pytest.approx(expected, rel=1e-6), item


def test_a_renamed_covariate_predicts_new_rows():
    """Fresh rows bind the column under the name the program gives it."""
    for column in ("sigma", "eta", "intercept", "beta_x_per_row"):
        data = FRAME.assign(**{column: EXTRA})
        fitted = fit(f"y ~ x + {column}", data=data, method="svi", num_steps=1, seed=0)
        assert isinstance(fitted, BayesianFit)
        fresh = data.iloc[[3, 5]]
        torch.testing.assert_close(
            fitted.posterior_linpred(new_data=fresh, num_samples=2, seed=1),
            fitted.posterior_linpred(num_samples=2, seed=1)[..., [3, 5]],
        )


def test_private_bindings_yield_to_coefficient_sites():
    """A coefficient named like a row contribution keeps its public name."""
    data = FRAME.assign(x_per_row=EXTRA, g_x=EXTRA[::-1])
    fitted = fit(
        "y ~ x + x_per_row + g_x + (0 + x || g)",
        data=data,
        method="svi",
        num_steps=1,
        seed=0,
    )
    assert isinstance(fitted, BayesianFit)
    draws = {
        key: value[0, 0].double().numpy()
        for key, value in fitted.draws(num_samples=1, seed=1).items()
    }
    level = draws["sigma_g_x"] * draws["z_g_x"]
    x = FRAME["x"].to_numpy()
    expected = (
        draws["intercept"]
        + draws["beta_x"] * x
        + draws["beta_x_per_row"] * EXTRA
        + draws["beta_g_x"] * EXTRA[::-1]
        + level[np.arange(ROWS) % 3] * x
    )
    np.testing.assert_allclose(
        fitted.posterior_linpred(num_samples=1, seed=1)[0, 0].double().numpy(),
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_a_joint_response_names_its_own_columns():
    """Each response of a joint model renames its own covariates."""
    data = FRAME.assign(sigma=EXTRA, y2=EXTRA[::-1])
    fitted = fit(
        bf("y ~ sigma") + bf("y2 ~ x"), data=data, method="svi", num_steps=1, seed=0
    )
    value = fitted.posterior_linpred(num_samples=1, seed=1)["y"][0, 0]
    draws = fitted.draws(num_samples=1, seed=1)
    expected = draws["response_y__intercept"][0, 0] + draws["response_y__beta_sigma"][
        0, 0
    ] * torch.tensor(EXTRA)
    torch.testing.assert_close(value.double(), expected.double(), rtol=1e-5, atol=1e-5)


def test_names_avoid_each_other():
    """Columns avoid every name, and bindings only the public ones."""
    names = resolve_names(
        columns=[
            ColumnName(prefix=prefix, column=column)
            for prefix, column in (
                ("", "a"),
                ("", "eta"),
                ("", "data_eta"),
                ("", "k"),
                ("", "p_x"),
                ("p_", "x"),
            )
        ],
        bindings=[
            ("", "beta_x_per_row"),
            ("", "eta"),
            ("|g|x", "beta_g_x_per_row"),
            ("", "beta_g_x_per_row"),
        ],
        public=["sigma", "beta_x_per_row", "data_eta_"],
        taken=["k", "mu"],
    )
    assert names.inputs == {
        ":eta": "data_eta__",
        ":k": "data_k",
        "p_:x": "data_p_x",
    }
    assert names.bindings == {
        ":beta_x_per_row": "beta_x_per_row_binding",
        ":beta_g_x_per_row": "beta_g_x_per_row_binding",
    }
    assert names.input(ColumnName(prefix="", column="a")) == "a"


def test_columns_with_one_identifier_are_distinct_coefficients():
    """Labels that normalize to one identifier keep separate sites."""
    data = FRAME.assign(cs_x2=EXTRA)
    fitted = fit(
        "o ~ cs_x2 + cs(x2)",
        data=data,
        family="acat",
        method="svi",
        num_steps=1,
        seed=0,
    )
    assert isinstance(fitted, BayesianFit)
    draws = fitted.draws(num_samples=1, seed=1)
    assert draws["beta_cs_x2"].shape == (1, 1)
    assert draws["beta_cs_x2_2"].shape == (1, 1, 3)
    eta = fitted.posterior_linpred(num_samples=1, seed=1)[0, 0]
    expected = draws["intercept"][0, 0] + draws["beta_cs_x2"][0, 0] * torch.tensor(
        EXTRA
    )
    torch.testing.assert_close(eta.double(), expected.double(), rtol=1e-5, atol=1e-5)


def test_the_sites_of_a_program_keep_their_names():
    """Renaming a column changes no sampled site."""
    fitted = MODELS["terms"]("eta")
    assert fitted.prediction_plan is not None
    program = next(
        item
        for item in fitted.prediction_plan.module.statements
        if isinstance(item, ProgramDecl)
    )
    sites = {step.vars[0] for step in program.draws if isinstance(step, SampleStep)}
    assert "beta_eta" in sites
