"""Count-mixture predictions use native means and checked trial metadata."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    formula_to_qvr,
)
from quivers.formulas._fit import _binomial_trial_tensor
from quivers.inference.mcmc.driver import MCMCResult


NEW_COUNT_FAMILIES = (
    "zero_inflated_negbinomial",
    "zero_inflated_binomial",
    "zero_inflated_beta_binomial",
    "hurdle_negbinomial",
)
TRIAL_FAMILIES = (
    "binomial",
    "beta_binomial",
    "zero_inflated_binomial",
    "zero_inflated_beta_binomial",
)


def _recorded(result, values, count=3):
    return result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(2, count, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(2, count),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _fit_count(family, *, atom=0.25, trials="n", distributional=None):
    fixed = {"hu" if family == "hurdle_negbinomial" else "zi": atom}
    if family in {"zero_inflated_negbinomial", "hurdle_negbinomial"}:
        fixed["shape"] = 2.0
    elif family == "zero_inflated_beta_binomial":
        fixed["phi"] = 4.0
    if distributional:
        for parameter in distributional:
            fixed.pop(parameter, None)
    kwargs = (
        {"binomial_trials": trials}
        if families[family].trials_parameter is not None
        else {}
    )
    responses = (
        [0, 0, 0, 0]
        if atom == 1
        else [1, 1, 2, 1]
        if family == "hurdle_negbinomial" and atom == 0
        else [0, 1, 2, 0]
    )
    return fit(
        "y ~ x",
        data=pd.DataFrame(
            {
                "y": responses,
                "x": [-1.0, 0.0, 1.0, 2.0],
                "z": [0.0, 1.0, 2.0, 3.0],
                "n": [0, 2, 3, 4],
            }
        ),
        family=family,
        fixed_parameters=fixed,
        distributional=distributional,
        method="svi",
        num_steps=1,
        **kwargs,
    )


@pytest.mark.parametrize("family", NEW_COUNT_FAMILIES)
def test_new_count_mixture_predictions_match_independent_mean_and_link_oracles(family):
    result = _recorded(
        _fit_count(family),
        {"intercept": torch.tensor(0.3), "beta_x": torch.tensor(-0.2)},
    )
    fresh = pd.DataFrame({"x": [2.0, -1.0, 0.0], "n": [0, 7, 3]})
    eta = 0.3 - 0.2 * torch.tensor([2.0, -1.0, 0.0])
    if families[family].trials_parameter is not None:
        parameter = eta.sigmoid()
        expected = (
            0.75
            * torch.tensor([0.0, 7.0, 3.0], dtype=torch.float64)
            * parameter.double()
        )
    else:
        parameter = eta.exp()
        expected = 0.75 * parameter.double()
        if family == "hurdle_negbinomial":
            zero_probability = (2.0 / (2.0 + parameter.double())).square()
            expected = expected / (1 - zero_probability)
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh), eta.expand(2, 3, 3)
    )
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh, transform=True),
        parameter.expand(2, 3, 3),
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(2, 3, 3)
    )
    samples = result.posterior_predict(new_data=fresh, seed=47)
    assert samples.shape == (2, 3, 3)
    assert (
        (samples >= 0) & (samples == samples.round()) & torch.isfinite(samples)
    ).all()
    if families[family].trials_parameter is not None:
        assert (samples <= torch.tensor([0, 7, 3])).all()
        assert samples[..., 0].eq(0).all()


@pytest.mark.parametrize("family", NEW_COUNT_FAMILIES)
@pytest.mark.parametrize("atom", [0.0, 1.0])
def test_zero_inflation_and_hurdle_endpoints_preserve_exact_support(family, atom):
    result = _recorded(
        _fit_count(family, atom=atom),
        {"intercept": torch.tensor(0.0), "beta_x": torch.tensor(0.0)},
        count=10,
    )
    fresh = pd.DataFrame({"x": [0.0, 0.0], "n": [0, 4]})
    samples = result.posterior_predict(new_data=fresh, seed=61)
    means = result.posterior_epred(new_data=fresh)
    if atom == 1:
        assert samples.eq(0).all() and means.eq(0).all()
    elif family == "hurdle_negbinomial":
        assert samples.ge(1).all() and means.gt(1).all()
    elif families[family].trials_parameter is not None:
        assert samples[..., 0].eq(0).all() and means[..., 0].eq(0).all()


@pytest.mark.parametrize("family", NEW_COUNT_FAMILIES)
def test_distributional_atom_predictor_is_replayed_on_new_rows(family):
    atom = "hu" if family == "hurdle_negbinomial" else "zi"
    result = _recorded(
        _fit_count(family, distributional={atom: "~ z"}),
        {
            "intercept": torch.tensor(0.0),
            "beta_x": torch.tensor(0.0),
            f"dpar_{atom}_intercept": torch.tensor(0.2),
            f"dpar_{atom}_beta_z": torch.tensor(-0.3),
        },
    )
    fresh = pd.DataFrame({"x": [0.0, 1.0], "z": [2.0, -1.0], "n": [0, 5]})
    eta = 0.2 - 0.3 * torch.tensor([2.0, -1.0])
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh, parameter=atom), eta.expand(2, 3, 2)
    )
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh, parameter=atom, transform=True),
        eta.sigmoid().expand(2, 3, 2),
    )
    base = (
        torch.tensor([0.0, 2.5], dtype=torch.float64)
        if families[family].trials_parameter is not None
        else torch.ones(2, dtype=torch.float64)
    )
    if family == "hurdle_negbinomial":
        base = base / (1 - (2 / 3) ** 2)
    expected = (1 - eta.sigmoid().double()) * base
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(2, 3, 2)
    )


@pytest.mark.parametrize("family", TRIAL_FAMILIES)
def test_zero_trials_work_in_fit_source_emission_and_new_row_predictions(family):
    frame = pd.DataFrame({"y": [0, 0, 1], "n": [0, 0, 2], "x": [0.0, 1.0, 2.0]})
    source = formula_to_qvr("y ~ x", data=frame, family=family, binomial_trials="n")
    assert "total_count=n" in source
    result = fit(
        "y ~ x",
        data=frame,
        family=family,
        binomial_trials="n",
        method="svi",
        num_steps=1,
    )
    predictions = result.posterior_predict(
        new_data=pd.DataFrame({"x": [3.0], "n": [0]}), num_samples=2
    )
    assert predictions.eq(0).all()
    assert (
        result.posterior_epred(
            new_data=pd.DataFrame({"x": [3.0], "n": [0]}), num_samples=2
        )
        .eq(0)
        .all()
    )
    zero_frame = frame.assign(y=0)
    formula_to_qvr("y ~ x", data=zero_frame, family=family, binomial_trials=0)
    constant = fit(
        "y ~ x",
        data=zero_frame,
        family=family,
        binomial_trials=0,
        method="svi",
        num_steps=1,
    )
    assert (
        constant.posterior_predict(
            new_data=pd.DataFrame({"x": [1.0, 2.0]}), num_samples=1
        )
        .eq(0)
        .all()
    )


@pytest.mark.parametrize("family", TRIAL_FAMILIES)
@pytest.mark.parametrize("entry", ["fit", "formula_to_qvr"])
def test_zero_trial_rows_reject_positive_responses(family, entry):
    call = fit if entry == "fit" else formula_to_qvr
    kwargs = {"method": "svi", "num_steps": 1} if entry == "fit" else {}
    with pytest.raises(ValueError, match="between zero and binomial_trials"):
        call(
            "y ~ 1",
            data=pd.DataFrame({"y": [1], "n": [0]}),
            family=family,
            binomial_trials="n",
            **kwargs,
        )


@pytest.mark.parametrize(
    "count", [-1, True, 1.5, 2**53 + 1, 2**63 - 1, 2**63, 2**64, "missing"]
)
def test_scalar_trial_validation_rejects_invalid_or_out_of_range_values(count):
    with pytest.raises(ValueError, match="binomial_trials"):
        _binomial_trial_tensor(pd.DataFrame({"n": [1]}), count, n_obs=1)


@pytest.mark.parametrize(
    "column",
    [
        np.array([-1]),
        np.array([True]),
        np.array([1.5]),
        np.array([np.inf]),
        np.array([np.nan]),
        np.array([2**53 + 1], dtype=np.int64),
        np.array([2**63 - 1], dtype=np.int64),
        np.array([2**63], dtype=np.uint64),
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([float(2**63)]),
        np.array(["1"]),
        np.array([1 + 0j]),
    ],
)
def test_column_trial_validation_rejects_invalid_or_out_of_range_values(column):
    with pytest.raises(ValueError, match="binomial_trials"):
        _binomial_trial_tensor(pd.DataFrame({"n": column}), "n", n_obs=1)


def test_float64_trial_boundaries_and_zero_preserve_exact_integer_storage():
    values = [0, 2**24 + 1, 2**53]
    actual = _binomial_trial_tensor(
        pd.DataFrame({"n": pd.Series(values, dtype="int64")}), "n", n_obs=3
    )
    assert actual.dtype == torch.long and actual.tolist() == values
    assert (
        _binomial_trial_tensor(pd.DataFrame(index=range(1)), 2**53, n_obs=1).item()
        == 2**53
    )


@pytest.mark.parametrize("family", TRIAL_FAMILIES)
@pytest.mark.parametrize(
    "count", [-1, True, 1.5, np.inf, 2**53 + 1, 2**63 - 1, float(2**63)]
)
def test_invalid_new_trials_fail_before_latent_draws(monkeypatch, family, count):
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0, 1], "n": [0, 2]}),
        family=family,
        binomial_trials="n",
        method="svi",
        num_steps=1,
    )
    monkeypatch.setattr(
        type(result),
        "draws",
        lambda *a, **kw: pytest.fail("invalid trials reached posterior draws"),
    )
    with pytest.raises(ValueError, match="binomial_trials"):
        result.posterior_predict(new_data=pd.DataFrame({"n": [count]}))


@pytest.mark.parametrize("family", TRIAL_FAMILIES)
@pytest.mark.parametrize("count", [2**53 + 1, 2**63 - 1])
@pytest.mark.parametrize("entry", ["fit", "formula_to_qvr", "lens"])
def test_inexact_trial_counts_fail_before_source_or_fit(family, count, entry):
    with pytest.raises(ValueError, match="binomial_trials.*2\\*\\*53"):
        if entry == "lens":
            FormulaToQVRModule(families[family], binomial_trials=count)
        else:
            call = fit if entry == "fit" else formula_to_qvr
            kwargs = {"method": "svi", "num_steps": 1} if entry == "fit" else {}
            call(
                "y ~ 1",
                data=pd.DataFrame({"y": [0]}),
                family=family,
                binomial_trials=count,
                **kwargs,
            )


@pytest.mark.parametrize("family", TRIAL_FAMILIES)
def test_trial_metadata_survives_link_updates_and_formula_lens_roundtrip(family):
    original = families[family]
    changed = original.with_links(mu="probit").with_links()
    assert changed.trials_parameter == "total_count"
    frame = pd.DataFrame({"y": [0, 1, 0], "n": [0, 2, 3], "x": [1.0, 2.0, 3.0]})
    formula = formula_from_data("y ~ x", frame)
    lens = FormulaToQVRModule(changed, binomial_trials="n")
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert pickle.loads(pickle.dumps(formula)) == formula
    assert (
        original.trials_parameter == "total_count"
        and original.location_link.name == "logit"
    )
