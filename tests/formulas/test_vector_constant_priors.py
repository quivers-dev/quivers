"""Vector-valued constant priors: brms ``constant(v, broadcast = FALSE)``.

brms 2.23.0 fixes a parameter vector with ``<par> = <v>``
(``stan_constant_prior`` with ``broadcast = FALSE``), so entry ``k`` of ``v``
is the constant of the ``k``-th coefficient of ``b``, ``b_<dpar>``, or
``sd_<id>``. The oracle ``oracles/brms_vector_constants.json`` (generator
``oracles/brms_vector_constants.R``) records brms's coefficient order of each
of those vectors. Values here follow the order of the prior table, which is
the order of the formula's terms; R's model matrix, and thus brms, places
every interaction after the main effects, so the two orders differ for a
formula that writes a main effect after an interaction (pinned below). The
joint density with fixed coefficients is checked against
a torch reference written here, with first and second derivatives, at the
default float64 tolerance of ``torch.testing.assert_close``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
import torch

from quivers.formulas import (
    BayesianFit,
    ConstantPrior,
    PriorSet,
    PriorSpec,
    default_prior,
    fit,
    get_prior,
    validate_prior,
)
from quivers.inference.trace import trace
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_vector_constants.json").read_text()
)
FRAME = pd.DataFrame(ORACLE["data"])


def _vector(
    *values: float, class_: str, group: str | None = None, dpar: str | None = None
) -> PriorSpec:
    return PriorSpec(
        distribution=ConstantPrior(value=values, broadcast=False),
        class_=class_,
        group=group,
        dpar=dpar,
    )


def _listed(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def _svi(formula: str, priors: PriorSpec | PriorSet, **options) -> BayesianFit:
    return fit(
        formula, data=FRAME, priors=priors, method="svi", num_steps=3, seed=1, **options
    )


#: Formula whose term order differs from R's model-matrix order.
TERM_ORDER = "y ~ x * z + w"


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["formula"])
def test_values_follow_brms_coefficient_order(case):
    options = {"distributional": {"sigma": case["sigma"]}} if "sigma" in case else {}
    names = _listed(case["b"])
    if case["formula"] == TERM_ORDER:
        # brms: x, z, w, x:z. The prior table lists the terms as written.
        assert names == ["x", "z", "w", "x:z"]
        names = ["x", "z", "x:z", "w"]
        table = default_prior(case["formula"], data=FRAME).to_pandas()
        assert table.loc[table["class"] == "b", "coef"].tolist() == names
    values = [0.25 * (index + 1) for index in range(len(names))]
    fitted = _svi(case["formula"], _vector(*values, class_="b"), **options)
    variables = fitted.variables(num_samples=2)
    for name, value in zip(names, values, strict=True):
        assert torch.all(variables[f"b_{name}"] == value)
    for group, coefficients in case["sd"].items():
        scales = [0.3 + 0.2 * index for index in range(len(coefficients))]
        fitted = _svi(
            case["formula"], _vector(*scales, class_="sd", group=group), **options
        )
        variables = fitted.variables(num_samples=2)
        for coefficient, scale in zip(coefficients, scales, strict=True):
            assert torch.all(variables[f"sd_{group}__{coefficient}"] == scale)
    if "b_sigma" in case:
        names = _listed(case["b_sigma"])
        values = [0.1 * (index + 1) for index in range(len(names))]
        fitted = _svi(
            case["formula"], _vector(*values, class_="b", dpar="sigma"), **options
        )
        variables = fitted.variables(num_samples=2)
        for name, value in zip(names, values, strict=True):
            assert torch.all(variables[f"b_sigma_{name}"] == value)
        # The location coefficient is a different vector and stays sampled.
        assert variables["b_x"].unique().numel() > 1


def test_a_class_rule_fixes_each_vector_it_selects():
    """brms assigns the vector to every ``sd_<id>`` a group-free rule selects."""
    formula = "y ~ x + (1 + x | g) + (1 + z | h)"
    fitted = _svi(formula, _vector(0.5, 0.2, class_="sd"))
    variables = fitted.variables(num_samples=2)
    for name, value in (
        ("sd_g__Intercept", 0.5),
        ("sd_g__x", 0.2),
        ("sd_h__Intercept", 0.5),
        ("sd_h__z", 0.2),
    ):
        assert torch.all(variables[name] == value)
    # Vectors of different lengths cannot share one constant vector.
    with pytest.raises(ValueError, match="selects 1 scalar parameters"):
        _svi("y ~ x + (1 + x | g) + (1 | h)", _vector(0.5, 0.2, class_="sd"))


def test_prior_table_and_catalogue_report_each_constant():
    rule = _vector(0.5, -0.2, class_="b")
    table = validate_prior(rule, "y ~ x + z", data=FRAME)
    assert table.row("beta_x").prior == "constant(0.5)"
    assert table.row("beta_z").prior == "constant(-0.2)"
    # A fixed coefficient has no prior left to override.
    assert set(get_prior("y ~ x + z", data=FRAME, priors=rule)) == {
        "intercept",
        "sigma",
    }


def test_joint_density_and_derivatives_equal_a_torch_reference():
    """Fixed coefficients enter the likelihood and leave the prior."""
    fitted = _svi(
        "y ~ x + z + (1 | g)",
        _vector(0.5, -0.2, class_="b") + _vector(0.7, class_="sd", group="g"),
    )
    y = torch.tensor(FRAME["y"].tolist(), dtype=torch.float64)
    x = torch.tensor(FRAME["x"].tolist(), dtype=torch.float64)
    z = torch.tensor(FRAME["z"].tolist(), dtype=torch.float64)
    levels = sorted(set(FRAME["g"]))
    codes = torch.tensor([levels.index(value) for value in FRAME["g"]])
    program = fitted.program
    observed = dict(fitted.observations)

    def actual(point):
        values = {
            **observed,
            "intercept": point[0],
            "z_g_Intercept": point[1:5],
            "sigma": point[5],
        }
        joint = trace(program, point.new_zeros(1, 1), values).log_joint
        assert joint is not None
        return joint.sum()

    def reference(point):
        normal = torch.distributions.Normal
        mu = point[0] + 0.5 * x - 0.2 * z + 0.7 * point[1:5][codes]
        return (
            normal(0.0, 5.0).log_prob(point[0])
            + normal(0.0, 1.0).log_prob(point[1:5]).sum()
            + torch.distributions.HalfCauchy(2.0).log_prob(point[5])
            + normal(mu, point[5]).log_prob(y).sum()
        )

    point = torch.tensor(
        [0.3, 0.2, -0.4, 0.1, 0.6, 0.8], dtype=torch.float64, requires_grad=True
    )
    torch.testing.assert_close(actual(point), reference(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(reference, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(reference, point),
    )
    # Swapping the two constants is a different density.
    swapped = reference(point) - (
        torch.distributions.Normal(
            point[0] + 0.5 * x - 0.2 * z + 0.7 * point[1:5][codes], point[5]
        )
        .log_prob(y)
        .sum()
        - torch.distributions.Normal(
            point[0] - 0.2 * x + 0.5 * z + 0.7 * point[1:5][codes], point[5]
        )
        .log_prob(y)
        .sum()
    )
    assert abs(float((swapped - actual(point)).detach())) > 1e-3


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_method_fits_predicts_refits_and_restores(
    method: Literal["svi", "hmc", "nuts"], tmp_path: Path
):
    rule = _vector(0.5, -0.2, class_="b")
    if method == "svi":
        fitted = fit(
            "y ~ x + z", data=FRAME, priors=rule, method="svi", num_steps=5, seed=2
        )
    else:
        fitted = fit(
            "y ~ x + z",
            data=FRAME,
            priors=rule,
            method=method,
            num_warmup=10,
            num_samples=6,
            num_chains=1,
            seed=2,
        )
    draws = fitted.draws(num_samples=4, seed=3)
    assert torch.all(draws["beta_x"] == 0.5) and torch.all(draws["beta_z"] == -0.2)
    fresh = pd.DataFrame(
        {"x": [0.0, 1.0, -2.0], "z": [0.0, 0.5, 1.0], "y": [0.1, 0.2, 0.3]}
    )
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=4, seed=3)
    expected = draws["intercept"].double()[..., None] + torch.tensor(
        [0.0, 0.5 - 0.1, -1.0 - 0.2], dtype=torch.float64
    )
    torch.testing.assert_close(linpred.double(), expected)
    assert fitted.posterior_predict(new_data=fresh, num_samples=4).shape[-1] == 3
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=4)["y"]).all()
    )
    refit = fitted.update(data=FRAME.iloc[:18])
    assert torch.all(refit.draws(num_samples=2)["beta_z"] == -0.2)
    path = tmp_path / "constants.qvrfit"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=fresh, num_samples=4, seed=3), linpred
    )
    if method == "svi":
        assert math.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)


@pytest.mark.parametrize(
    ("formula", "rule", "message"),
    [
        (
            "y ~ x + z",
            _vector(0.5, 0.2, 0.1, class_="b"),
            r"3 values and broadcast=False selects 2 scalar parameters "
            r"\['beta_x', 'beta_z'\]",
        ),
        (
            "y ~ x + z + w",
            _vector(0.5, 0.2, class_="b"),
            "selects 3 scalar parameters",
        ),
        (
            "y ~ x",
            _vector(0.5, 0.2, class_="b"),
            "selects 1 scalar parameters",
        ),
        (
            "y ~ x + z",
            PriorSpec(distribution=ConstantPrior(value=(0.5, 0.2)), class_="b"),
            "takes one value, got 2",
        ),
        (
            "y ~ x + (1 + x | g)",
            _vector(0.5, 0.2, 0.3, class_="sd", group="g"),
            "selects 2 scalar parameters",
        ),
    ],
)
def test_wrong_counts_and_broadcast_vectors_are_refused(formula, rule, message):
    with pytest.raises(ValueError, match=message):
        validate_prior(rule, formula, data=FRAME)
