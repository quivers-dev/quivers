"""brms ``mi()`` terms as varying slopes of group-level terms.

brms multiplies the completed values of an ``mi()`` variable by the sum of
its population-level coefficient and the group's varying coefficient,
``(bsp + r[J[n]]) * Yl[n]``. The predictor statements and coefficient names
are pinned by ``oracles/brms_missing_slopes.R``. Densities are compared with
a hand-written joint density in Float64, with tolerance 1e-10 for values and
autograd derivatives; fitted programs run in Float32 and are compared with
relative 1e-5 and absolute 1e-6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import bf, fit, formula_to_qvr, set_rescor
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_missing_slopes.json").read_text()
)
SPEC = "y ~ mi(x) + (mi(x) | g)"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(ORACLE["data"]).astype({"x": float})


def _joint(formula: str = SPEC):
    return bf(formula) + bf("x | mi() ~ z") + set_rescor(False)


def _values() -> dict[str, torch.Tensor]:
    factor = torch.linalg.cholesky(torch.tensor([[1.0, 0.4], [0.4, 1.0]]))
    return {
        "response_y__cor_g": factor,
        "response_y__sigma_g_Intercept": torch.tensor(0.6),
        "response_y__z_g_Intercept": torch.tensor([0.5, -1.0, 0.3]),
        "response_y__sigma_g_x": torch.tensor(0.4),
        "response_y__z_g_x": torch.tensor([-0.7, 0.2, 1.1]),
        "response_y__intercept": torch.tensor(0.3),
        "response_y__beta_x": torch.tensor(0.8),
        "response_y__sigma": torch.tensor(0.7),
        "response_x__intercept": torch.tensor(0.1),
        "response_x__beta_z": torch.tensor(0.6),
        "response_x__sigma": torch.tensor(0.5),
        "mi_x__missing": torch.tensor([0.4, -0.3, 1.1]),
    }


def _effects(values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale brms's noncentered group effects: r = (diag(sd) L z)'."""
    innovations = torch.stack(
        [values["response_y__z_g_Intercept"], values["response_y__z_g_x"]]
    )
    transformed = values["response_y__cor_g"] @ innovations
    return (
        values["response_y__sigma_g_Intercept"] * transformed[0],
        values["response_y__sigma_g_x"] * transformed[1],
    )


def _likelihood(
    data: pd.DataFrame, values: dict[str, torch.Tensor], imputed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write brms's predictor and joint likelihood by hand."""
    observed = np.isfinite(data.x.to_numpy())
    completed = torch.tensor(np.nan_to_num(data.x.to_numpy()), dtype=imputed.dtype)
    completed = completed.masked_scatter(torch.tensor(~observed), imputed)
    group = torch.tensor(pd.Categorical(data.g).codes, dtype=torch.long)
    intercepts, slopes = _effects(values)
    z = torch.tensor(data.z.to_numpy(), dtype=imputed.dtype)
    y = torch.tensor(data.y.to_numpy(), dtype=imputed.dtype)
    mu_x = values["response_x__intercept"] + values["response_x__beta_z"] * z
    # brms: mu_y[n] += (bsp_y[1] + r_1_y_2[J[n]]) * Yl_x[n] + r_1_y_1[J[n]].
    mu_y = (
        values["response_y__intercept"]
        + (values["response_y__beta_x"] + slopes[group]) * completed
        + intercepts[group]
    )
    total = (
        torch.distributions.Normal(mu_x, values["response_x__sigma"])
        .log_prob(completed)
        .sum()
        + torch.distributions.Normal(mu_y, values["response_y__sigma"])
        .log_prob(y)
        .sum()
    )
    return mu_y, total


def test_brms_writes_the_varying_coefficient_onto_the_completed_values():
    assert ORACLE["slope"]["predictor"] == (
        "mu_y[n] += (bsp_y[1] + r_1_y_2[J_1_y[n]]) * Yl_x[n] "
        "+ r_1_y_1[J_1_y[n]] * Z_1_y_1[n];"
    )
    # Without the population-level term brms declares the slope and never
    # uses it in the predictor.
    assert ORACLE["orphan"]["predictor"] == "mu_y[n] += r_1_y_1[J_1_y[n]] * Z_1_y_1[n];"
    source = formula_to_qvr(_joint(), data=_frame())
    assert "response_y__beta_g_x[response_y__g_idx] * response_y__random_x" in source
    assert "let response_y__random_x = mi_y__random_x" in source


def test_joint_density_and_its_derivatives_match_the_hand_written_model():
    torch.set_default_dtype(torch.float64)
    try:
        data = _frame()
        fitted = fit(_joint(), data=data, method="svi", num_steps=1)
        values = {name: value.double() for name, value in _values().items()}

        def model(imputed: torch.Tensor) -> torch.Tensor:
            sites = dict(values) | {"mi_x__missing": imputed}
            return trace(
                fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
            ).log_joint.sum()

        def reference(imputed: torch.Tensor) -> torch.Tensor:
            return _likelihood(data, values, imputed)[1]

        point = values["mi_x__missing"]
        other = point + torch.tensor([0.3, -0.2, 0.5])
        # Priors do not involve the imputed values, so differences of the
        # log joint and its derivatives in them are the likelihood's.
        torch.testing.assert_close(
            model(other) - model(point),
            reference(other) - reference(point),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            torch.autograd.functional.jacobian(model, point),
            torch.autograd.functional.jacobian(reference, point),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(model, point),
            torch.autograd.functional.hessian(reference, point),
            rtol=1e-9,
            atol=1e-9,
        )
        # A slope that multiplied the placeholder column instead of the
        # completed values would not depend on the imputed values this way.
        fixed_only = dict(values) | {"response_y__sigma_g_x": torch.tensor(1e-12)}
        without = (
            _likelihood(data, fixed_only, other)[1]
            - _likelihood(data, fixed_only, point)[1]
        )
        assert abs(float(without - (reference(other) - reference(point)))) > 1e-3
    finally:
        torch.set_default_dtype(torch.float32)


def _pinned(fitted, values: dict[str, torch.Tensor], draws: int = 2):
    return fitted.with_(
        posterior=PriorDraws(
            samples={
                name: value.float().expand(1, draws, *value.shape).clone()
                for name, value in values.items()
            },
            num_samples=draws,
        ),
        response_fits={},
    )


def test_predictors_use_completed_values_on_training_rows_and_recorded_on_new():
    data = _frame()
    values = _values()
    fitted = _pinned(
        fit(_joint(), data=data, method="svi", num_steps=1, seed=1), values
    )
    expected, _ = _likelihood(data, values, values["mi_x__missing"])
    linpred = fitted.posterior_linpred(resp="y")["y"]
    torch.testing.assert_close(
        linpred[0, 0].float(), expected.float(), rtol=1e-5, atol=1e-6
    )
    fresh = data.dropna().iloc[[0, 3, 5, 7]].reset_index(drop=True)
    intercepts, slopes = _effects(values)
    group = torch.tensor(
        pd.Categorical(fresh.g, categories=["a", "b", "c"]).codes, dtype=torch.long
    )
    wanted = (
        values["response_y__intercept"]
        + (values["response_y__beta_x"] + slopes[group])
        * torch.tensor(fresh.x.to_numpy(), dtype=torch.float32)
        + intercepts[group]
    )
    torch.testing.assert_close(
        fitted.posterior_epred(resp="y", new_data=fresh)["y"][0, 0].float(),
        wanted.float(),
        rtol=1e-5,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="missing values of the mi\\(\\) column"):
        fitted.posterior_epred(resp="y", new_data=data.iloc[:3])


@pytest.mark.parametrize("name", ["slope", "slope_only", "interaction"])
def test_variable_names_follow_brms(name):
    formulas = {
        "slope": SPEC,
        "slope_only": "y ~ mi(x) + (0 + mi(x) | g)",
        "interaction": "y ~ mi(x) * w + (1 + mi(x):w | g)",
    }
    fitted = fit(
        _joint(formulas[name]), data=_frame(), method="svi", num_steps=1, seed=1
    )
    variables = set(fitted.variables(num_samples=1))
    prior = ORACLE[name]["prior"]
    for cls, coef, group, resp in zip(
        prior["class"], prior["coef"], prior["group"], prior["resp"], strict=True
    ):
        if cls == "sd":
            assert f"sd_{group}__{resp}_{coef}" in variables
            assert f"r_{group}__{resp}[a,{coef}]" in variables
        elif coef.startswith("mi"):
            assert f"bsp_{resp}_{coef}" in variables
        else:
            assert f"b_{resp}_{coef}" in variables
    if name == "slope":
        assert "cor_g__y_Intercept__y_mix" in variables


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_varying_mi_slopes_fit_predict_refit_and_restore(method, tmp_path):
    data = _frame()
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    fitted = fit(_joint(), data=data, seed=2, **settings)
    draws = fitted.draws(num_samples=3, seed=1)
    assert draws["mi_x__missing"].shape[-1] == 3
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    epred = fitted.posterior_epred(num_samples=3, seed=1)["y"]
    assert epred.shape == (1, 3, len(data)) and bool(torch.isfinite(epred).all())
    fresh = data.dropna().iloc[:4]
    predicted = fitted.posterior_predict(new_data=fresh, num_samples=3, seed=1)["y"]
    assert predicted.shape == (1, 3, 4)
    scores = fitted.log_likelihood(num_samples=3, seed=1)
    assert all(value.shape[-1] == len(data) for value in scores.values())
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert set(refitted.draws(num_samples=1)) == set(draws)
    path = tmp_path / "slopes.pt"
    save_fit(fitted, path)
    torch.testing.assert_close(
        load_fit(path).posterior_epred(num_samples=3, seed=1)["y"], epred
    )


def test_refusals_name_the_term():
    data = _frame()
    with pytest.raises(ValueError, match="needs the same term among the population"):
        formula_to_qvr(_joint("y ~ z + (mi(x) | g)"), data=data)
    with pytest.raises(ValueError, match="nonlinearly"):
        formula_to_qvr(_joint("y ~ mi(x) + (I(mi(x)**2) | g)"), data=data)
