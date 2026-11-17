"""brms intercept centering, QR coordinates, and the ``Intercept`` term.

Oracle: ``tests/formulas/oracles/brms_population_coordinates.json`` from
``brms_population_coordinates.R`` (brms ``make_standata``/``make_stancode``
and the scaled thin QR factors brms builds in transformed data).
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import FormulaToQVRModule, families, fit, formula_from_data
from quivers.formulas._population_coordinates import thin_qr
from quivers.formulas._prior_tables import validate_prior
from quivers.formulas._priors import PriorSpec
from quivers.inference.trace import trace
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_population_coordinates.json").read_text()
)
FRAME = pd.DataFrame(ORACLE["data"])
FORMULA = "y ~ x + z + w"
RULES = (
    PriorSpec(distribution="Normal(0.5, 2.0)", class_="Intercept")
    + PriorSpec(distribution="Normal(0.2, 3.0)", class_="b")
    + PriorSpec(distribution="HalfNormal(1.5)", class_="sigma")
)
# brms center = FALSE: the intercept is class b, so no Intercept rule applies.
UNCENTERED_RULES = PriorSpec(distribution="Normal(0.2, 3.0)", class_="b") + PriorSpec(
    distribution="HalfNormal(1.5)", class_="sigma"
)


@contextmanager
def _double():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _program(**options):
    formula = formula_from_data(FORMULA, FRAME)
    rules = RULES if options.get("center", True) else UNCENTERED_RULES
    lens = FormulaToQVRModule(families["gaussian"], user_priors=rules, **options)
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observations = {
        name: value.double()
        for name, value in lens.fixed_column_observations(formula).items()
    } | {"y": torch.tensor(FRAME.y.to_numpy(), dtype=torch.double)}
    return lens, module, program, observations


def _design():
    return FRAME[["x", "z", "w"]].to_numpy(dtype=np.float64)


def test_scaled_thin_qr_matches_brms_transformed_data():
    for key, design in (
        ("centered", _design() - _design().mean(axis=0)),
        ("uncentered", np.column_stack([np.ones(len(FRAME)), _design()])),
    ):
        xq, xr = thin_qr(design)
        expected = ORACLE[key]
        np.testing.assert_allclose(xq, np.array(expected["XQ"]), atol=1e-12)
        np.testing.assert_allclose(xr, np.array(expected["XR"]), atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.inv(xr), np.array(expected["XR_inv"]), atol=1e-9
        )
        assert np.all(np.diag(xr) > 0)
        np.testing.assert_allclose(xq @ xr, design, atol=1e-12)
        np.testing.assert_allclose(
            xq.T @ xq, (len(FRAME) - 1) * np.eye(design.shape[1]), atol=1e-10
        )
    np.testing.assert_allclose(_design().mean(axis=0), ORACLE["means"], atol=1e-15)
    assert "XQ = qr_thin_Q(Xc) * sqrt(N - 1);" in ORACLE["stan_qr"]
    assert (
        "real b_Intercept = Intercept - dot_product(means_X, b);"
        in (ORACLE["stan_center"])
    )


def _hand_density(intercept, coordinates, sigma, *, centered, qr):
    """brms's log density in its sampled coordinates (its Stan model block).

    The design enters the likelihood as the single-precision columns quivers
    binds; centering means and QR factors come from the double-precision
    training columns, as emitted. Returns the density and the original-scale
    coefficients ``b`` (intercept first for an uncentered QR design).
    """
    design = torch.tensor(_design())
    observed = torch.tensor(_design(), dtype=torch.float32).double()
    means = design.mean(dim=0)
    y = torch.tensor(FRAME.y.to_numpy())
    if qr:
        source = (
            design - means
            if centered
            else torch.cat([torch.ones(len(FRAME), 1), design], dim=1)
        )
        _, xr = thin_qr(source.numpy())
        actual = torch.tensor(np.linalg.inv(xr)) @ coordinates
    else:
        actual = coordinates
    density = torch.distributions.Normal(0.2, 3.0).log_prob(coordinates).sum()
    density = density + torch.distributions.HalfNormal(1.5).log_prob(sigma)
    if centered:
        mean = intercept - means @ actual + observed @ actual
        density = density + torch.distributions.Normal(0.5, 2.0).log_prob(intercept)
    else:
        mean = actual[0] + observed @ actual[1:]
    return density + torch.distributions.Normal(mean, sigma).log_prob(y).sum(), actual


@pytest.mark.parametrize("centered, qr", [(True, False), (True, True), (False, True)])
def test_sampled_coordinates_score_brms_density_exactly(centered, qr):
    with _double():
        lens, module, program, observations = _program(
            center=centered, decomp="QR" if qr else "none"
        )
        intercept = torch.tensor(0.35, requires_grad=True)
        width = 3 if centered or not qr else 4
        coordinates = torch.tensor([0.4, -0.3, 0.25, 0.1][:width], requires_grad=True)
        sigma = torch.tensor(0.8, requires_grad=True)
        names = ["beta_x", "beta_z", "beta_w"]
        if qr and not centered:
            names = ["intercept"] + names
        suffix = "_qr" if qr else ""
        sites = {name + suffix: coordinates[i] for i, name in enumerate(names)}
        sites["sigma"] = sigma
        if centered:
            sites["intercept_centered"] = intercept
        recorded = trace(program, torch.zeros(1, 1), observations | sites)
        density, actual = _hand_density(
            intercept, coordinates, sigma, centered=centered, qr=qr
        )
        torch.testing.assert_close(
            recorded.log_joint.sum(), density, atol=1e-11, rtol=1e-12
        )
        parameters = (coordinates, sigma) + ((intercept,) if centered else ())
        actual_grad = torch.autograd.grad(
            recorded.log_joint.sum(), parameters, create_graph=True
        )
        expected_grad = torch.autograd.grad(density, parameters, create_graph=True)
        for a, e in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)
        hessian = torch.autograd.grad(actual_grad[0].sum(), coordinates)[0]
        expected_hessian = torch.autograd.grad(expected_grad[0].sum(), coordinates)[0]
        torch.testing.assert_close(hessian, expected_hessian, atol=1e-9, rtol=1e-9)
        # Reported coefficients are the stated change of variables.
        reported = {
            name: recorded.sites[name].value.reshape(())
            for name in ("intercept", "beta_x", "beta_z", "beta_w")
        }
        b = torch.stack([reported[n] for n in ("beta_x", "beta_z", "beta_w")])
        if qr and not centered:
            torch.testing.assert_close(reported["intercept"], actual[0])
            torch.testing.assert_close(b, actual[1:])
        else:
            torch.testing.assert_close(b, actual)
        if centered:
            means = torch.tensor(ORACLE["means"])
            torch.testing.assert_close(
                reported["intercept"], intercept - means @ b, atol=1e-14, rtol=0
            )


def test_centering_is_a_unit_jacobian_shear_of_the_uncentered_likelihood():
    with _double():
        _, _, centered, observations = _program(center=True)
        _, _, plain, _ = _program(center=False)
        means = torch.tensor(ORACLE["means"])
        b = torch.tensor([0.4, -0.3, 0.25])
        intercept_c = torch.tensor(0.35)
        sigma = torch.tensor(0.8)
        sites = {"beta_x": b[0], "beta_z": b[1], "beta_w": b[2], "sigma": sigma}
        left = trace(
            centered,
            torch.zeros(1, 1),
            observations | sites | {"intercept_centered": intercept_c},
        )
        right = trace(
            plain,
            torch.zeros(1, 1),
            observations | sites | {"intercept": intercept_c - means @ b},
        )
        torch.testing.assert_close(
            left.sites["y"].log_prob.sum(), right.sites["y"].log_prob.sum()
        )

        def shear(point):
            return torch.cat([point[:1] - means @ point[1:], point[1:]])

        jacobian = torch.autograd.functional.jacobian(
            shear, torch.cat([intercept_c[None], b])
        )
        assert torch.linalg.det(jacobian).item() == pytest.approx(1.0, abs=1e-14)


def test_prior_rules_select_sampled_coordinates():
    table = validate_prior(
        PriorSpec(distribution="Normal(1.0, 0.5)", class_="b", coef="x")
        + PriorSpec(distribution="Normal(0.0, 7.0)", class_="Intercept"),
        FORMULA,
        data=FRAME,
        center=True,
        decomp="QR",
    )
    assert table.row("beta_x_qr").prior == "Normal(loc=1.0, scale=0.5)"
    assert table.row("beta_x_qr").coef == "x"
    assert table.row("intercept_centered").class_ == "Intercept"
    uncentered = validate_prior(
        PriorSpec(distribution="Normal(0.0, 7.0)", class_="b"),
        FORMULA,
        data=FRAME,
        center=False,
    )
    # brms center = FALSE: the intercept is an ordinary class b coefficient.
    assert uncentered.row("intercept").class_ == "b"
    assert uncentered.row("intercept").prior == "Normal(loc=0.0, scale=7.0)"


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fits_predict_refit_and_restore_in_original_coordinates(method):
    budget = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 25, "num_samples": 6, "num_chains": 1}
    )
    fitted = fit(
        FORMULA,
        data=FRAME,
        priors=RULES,
        center=True,
        decomp="QR",
        method=method,
        seed=11,
        **budget,
    )
    draws = fitted.draws(num_samples=6, seed=4)
    means = torch.tensor(ORACLE["means"], dtype=draws["beta_x"].dtype)
    slopes = torch.stack([draws[n] for n in ("beta_x", "beta_z", "beta_w")], dim=-1)
    torch.testing.assert_close(
        draws["intercept"], draws["intercept_centered"] - slopes @ means
    )
    fresh = pd.DataFrame({"x": [0.0, 1.5], "z": [2.0, -1.0], "w": [0.3, 0.1]})
    expected = draws["intercept"][..., None] + slopes @ torch.tensor(
        fresh[["x", "z", "w"]].to_numpy().T, dtype=slopes.dtype
    )
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=6, seed=4)
    torch.testing.assert_close(linpred, expected, atol=1e-5, rtol=1e-5)
    effects = fitted.fixed_effects(num_samples=6, seed=4)
    torch.testing.assert_close(effects["Intercept"], draws["intercept"])
    assert fitted.log_likelihood(num_samples=6, seed=4)["y"].shape[-1] == len(FRAME)
    if method == "nuts":
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit"
            save_fit(fitted, path)
            restored = load_fit(path)
        assert restored.decomp == "QR" and restored.center is True
        torch.testing.assert_close(
            restored.draws()["intercept"], fitted.draws()["intercept"]
        )
        refit = fitted.update(num_warmup=10, num_samples=3, seed=2)
        assert {"beta_x_qr", "intercept_centered"} <= set(refit.posterior.samples)


def test_intercept_term_is_an_uncentered_class_b_coefficient():
    table = validate_prior(
        PriorSpec(distribution="Normal(0.0, 4.0)", class_="b"),
        "y ~ 0 + Intercept + x",
        data=FRAME,
        default_priors="brms",
    )
    row = table.row("beta_Intercept")
    assert (row.class_, row.coef, row.prior) == (
        "b",
        "Intercept",
        "Normal(loc=0.0, scale=4.0)",
    )
    assert all(r.class_ != "Intercept" for r in table.rows)
    fitted = fit(
        "y ~ 0 + Intercept + x",
        data=FRAME,
        priors=PriorSpec(distribution="Normal(0.0, 4.0)", class_="b"),
        method="svi",
        num_steps=20,
        seed=3,
    )
    draws = fitted.draws(num_samples=4, seed=1)
    fresh = pd.DataFrame({"x": [0.5, -2.0]})
    expected = draws["beta_Intercept"][..., None] + draws["beta_x"][
        ..., None
    ] * torch.tensor([0.5, -2.0], dtype=draws["beta_x"].dtype)
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=fresh, num_samples=4, seed=1),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )
    assert set(fitted.fixed_effects(num_samples=2)) == {"Intercept", "x"}


def test_unsupported_coordinates_are_refused():
    with pytest.raises(ValueError, match="not supported for the cumulative family"):
        FormulaToQVRModule(families["cumulative"], center=True)
    with pytest.raises(ValueError, match="non-intercept predictors"):
        fit("y ~ 1", data=FRAME, decomp="QR", method="svi", num_steps=1)
    collinear = FRAME.assign(v=2.0 * FRAME.x)
    with pytest.raises(ValueError, match="full column rank"):
        fit("y ~ x + v", data=collinear, decomp="QR", method="svi", num_steps=1)
    with pytest.raises(ValueError, match="more rows than predictors"):
        fit(
            "y ~ x + z + w", data=FRAME.iloc[:3], decomp="QR", method="svi", num_steps=1
        )
    with pytest.raises(ValueError, match="decomp must be"):
        FormulaToQVRModule(families["gaussian"], decomp="svd")  # type: ignore[arg-type]
    assert math.isfinite(float(np.linalg.cond(_design())))
