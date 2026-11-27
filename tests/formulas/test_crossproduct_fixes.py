"""Entry points, prior rules, and accessors pinned beside the term matrix.

Each test states its oracle: a keyword fit of the same model for a
single-response ``bf``; the closed form of a correlation for a constant
Cholesky factor; the draws of a ``fixed_param`` run, which keep every chain's
initial values, for ``inits``; the documented definition of brms's
``bayes_R2`` computed by hand for a joint fit; and the zero-inflated and
hurdle Poisson probability mass functions for mixture predictions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

from quivers.continuous._batch_selection import select_batch
from quivers.continuous._zip_hurdle import HurdlePoisson, ZeroInflatedPoisson
from quivers.formulas import (
    R2D2,
    ConstantPrior,
    PriorSpec,
    RegularizedHorseshoe,
    bf,
    conditional_smooths,
    default_prior,
    families,
    fit,
    fit_multiple,
    formula_to_qvr,
    get_prior,
    mixture,
    set_rescor,
)
from quivers.formulas._variable_catalogue import (
    checked_variables,
    correlation_pairs,
    scalar_coordinates,
    smooth_label,
)

N = 24
SVI = {"method": "svi", "num_steps": 3, "seed": 0}


def _frame() -> pd.DataFrame:
    r = np.arange(N)
    rng = np.random.default_rng(0)
    d = pd.DataFrame(
        {
            "x": np.sin(r * 0.7),
            "z": np.cos(r * 1.3),
            "g": np.array(["a", "b", "c", "d"])[r % 4],
        }
    )
    d["y"] = 1.0 + 0.5 * d.x - 0.3 * d.z + 0.3 * rng.standard_normal(N)
    d["y2"] = 0.5 - 0.2 * d.x + 0.3 * rng.standard_normal(N)
    d["cnt"] = rng.poisson(3.0, N)
    d["unit"] = 1 / (1 + np.exp(-d.y + 1))
    d["p1"] = d.unit * 0.5
    d["p2"] = (1 - d.unit) * 0.5
    d["p3"] = 0.5
    d["ordy"] = pd.Categorical(
        np.array(["lo", "mid", "hi", "top"])[(r * 7) % 4],
        categories=["lo", "mid", "hi", "top"],
        ordered=True,
    )
    return d


DATA = _frame()


def test_single_response_bf_equals_the_keyword_form() -> None:
    authored = bf("y ~ x", "sigma ~ z")
    keyword = {"distributional": {"sigma": "sigma ~ z"}}
    assert get_prior(authored, data=DATA) == get_prior("y ~ x", data=DATA, **keyword)
    assert formula_to_qvr(authored, data=DATA) == formula_to_qvr(
        "y ~ x", data=DATA, **keyword
    )
    assert (
        default_prior(authored, data=DATA).rows
        == default_prior("y ~ x", data=DATA, **keyword).rows
    )
    fitted = fit(authored, data=DATA, **SVI)
    reference = fit("y ~ x", data=DATA, **keyword, **SVI)
    left = fitted.draws(num_samples=2, seed=1)
    right = reference.draws(num_samples=2, seed=1)
    assert left.keys() == right.keys()
    for site in left:
        torch.testing.assert_close(left[site], right[site])
    assert list(fitted.variables(num_samples=2)) == [
        "b_Intercept",
        "b_sigma_Intercept",
        "b_x",
        "b_sigma_z",
    ]
    # The retained recipe holds the string formula, so R's update works.
    assert "b_z" in fitted.update(formula=". ~ . + z").variables(num_samples=2)


def test_single_response_bf_carries_family_and_nonlinear_parameters() -> None:
    counts = fit(bf("cnt ~ x", family="poisson"), data=DATA, **SVI)
    assert counts.family.name == "poisson"
    curve = fit(
        bf("unit ~ b1 * exp(b2 * x)", "b1 ~ 1", "b2 ~ 1", nl=True), data=DATA, **SVI
    )
    assert {"b_b1_Intercept", "b_b2_Intercept"} <= set(curve.variables(num_samples=2))
    assert fit(bf("cnt ~ x"), data=DATA, family="poisson", **SVI).family.name == (
        "poisson"
    )


def test_single_response_bf_refuses_an_option_given_twice() -> None:
    with pytest.raises(ValueError, match=r"fit: family is given both in bf\(\)"):
        fit(bf("cnt ~ x", family="poisson"), data=DATA, family="gamma", **SVI)
    with pytest.raises(ValueError, match="distributional is given both"):
        get_prior(bf("y ~ x", "sigma ~ z"), data=DATA, distributional={"sigma": "~ x"})
    # The same value in both places is one statement of it.
    assert (
        fit(
            bf("cnt ~ x", family="poisson"), data=DATA, family="poisson", **SVI
        ).family.name
        == "poisson"
    )


def test_constant_cholesky_factor_fixes_the_group_correlation() -> None:
    rule = PriorSpec(
        distribution=ConstantPrior(value=(1.0, 0.0, 0.6, 0.8), broadcast=False),
        class_="cor",
    )
    fitted = fit("y ~ x + (1 + x | g)", data=DATA, priors=rule, **SVI)
    names = fitted.variables(num_samples=3)
    # (L L^T)[0, 1] = 1.0 * 0.6 for the factor [[1, 0], [0.6, 0.8]].
    torch.testing.assert_close(
        names["cor_g__Intercept__x"].double(),
        torch.full((1, 3), 0.6, dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    # The deviations follow the fixed factor: slopes are 0.6 z0 + 0.8 z1 scaled.
    draws = fitted.draws(num_samples=3)
    varying = fitted.random_effects(num_samples=3)["g"]
    z0, z1 = draws["z_g_Intercept"], draws["z_g_x"]
    torch.testing.assert_close(
        varying["x"], draws["sigma_g_x"].unsqueeze(-1) * (0.6 * z0 + 0.8 * z1)
    )
    assert torch.isfinite(fitted.posterior_predict(num_samples=2)).all()


def test_constant_on_a_correlation_refuses_values_that_are_not_a_factor() -> None:
    with pytest.raises(ValueError, match="fixes the 2 x 2 Cholesky factor"):
        fit(
            "y ~ x + (1 + x | g)",
            data=DATA,
            priors=PriorSpec(distribution=ConstantPrior(value=0.5), class_="cor"),
            **SVI,
        )
    with pytest.raises(ValueError, match="not the lower Cholesky factor"):
        fit(
            "y ~ x + (1 + x | g)",
            data=DATA,
            priors=PriorSpec(
                distribution=ConstantPrior(value=(1.0, 0.0, 0.6, 0.7), broadcast=False),
                class_="cor",
            ),
            **SVI,
        )
    with pytest.raises(ValueError, match="not the lower Cholesky factor"):
        fit(
            "y ~ x + (1 + x | g)",
            data=DATA,
            priors=PriorSpec(
                distribution=ConstantPrior(value=(0.8, 0.6, 0.0, 1.0), broadcast=False),
                class_="cor",
            ),
            **SVI,
        )


def test_brms_prior_class_names_select_their_sites() -> None:
    lkj = "LKJCholesky(3.0)"
    joint = bf("y ~ x") + bf("y2 ~ x") + set_rescor(True)
    for name in ("rescor", "Lrescor"):
        priors = get_prior(
            joint, data=DATA, priors=PriorSpec(distribution=lkj, class_=name)
        )
        assert priors["rescor"] == "LKJCholesky(concentration=3.0)"
    for name in ("lncor", "Llncor"):
        priors = get_prior(
            "cbind(p1, p2, p3) ~ x",
            data=DATA,
            family="logistic_normal",
            priors=PriorSpec(distribution=lkj, class_=name),
        )
        assert priors["rescor"] == "LKJCholesky(concentration=3.0)"
    spaced = get_prior(
        "ordy ~ x",
        data=DATA,
        family=families["cumulative"].with_thresholds("equidistant"),
        priors=PriorSpec(distribution="HalfNormal(2.0)", class_="delta"),
    )
    assert "HalfNormal(scale=2.0)" in spaced["cutpoint_spacing"]
    grouped = get_prior(
        "y ~ x + (1 + x | g)",
        data=DATA,
        priors=PriorSpec(distribution=lkj, class_="L"),
    )
    assert grouped["cor_g"] == "LKJCholesky(concentration=3.0)"
    # A brms name selects only its own kind of site.
    with pytest.raises(ValueError, match="unused prior selector"):
        get_prior(
            "y ~ x + (1 + x | g)",
            data=DATA,
            priors=PriorSpec(distribution=lkj, class_="rescor"),
        )
    with pytest.raises(ValueError, match="unused prior selector"):
        get_prior(joint, data=DATA, priors=PriorSpec(distribution=lkj, class_="L"))


def test_control_params_report_the_kernel_settings() -> None:
    budget = {"num_warmup": 3, "num_samples": 3, "num_chains": 1, "seed": 2}
    tuned = fit(
        "y ~ x",
        data=DATA,
        control={"adapt_delta": 0.9, "max_treedepth": 3, "metric": "dense_e"},
        **budget,
    )
    assert tuned.control_params() == {
        "adapt_delta": 0.9,
        "stepsize": 0.1,
        "metric": "dense_e",
        "max_treedepth": 3,
    }
    assert tuned.control_params(["max_treedepth"]) == {"max_treedepth": 3}
    stepped = fit(
        "y ~ x", data=DATA, control={"stepsize": 0.05, "max_treedepth": 2}, **budget
    )
    assert stepped.control_params(["stepsize", "adapt_delta"]) == {
        "stepsize": 0.05,
        "adapt_delta": 0.8,
    }
    with pytest.raises(ValueError, match="unknown control parameters"):
        tuned.control_params(["treedepth"])
    with pytest.raises(ValueError, match="fitted by MCMC"):
        fit("y ~ x", data=DATA, **SVI).control_params()


def test_inits_are_the_values_a_fixed_param_run_keeps() -> None:
    # A fixed_param run never moves, so its draws are the initial values.
    still = fit(
        "y ~ x + (1 | g)",
        data=DATA,
        algorithm="fixed_param",
        num_samples=2,
        num_chains=3,
        seed=4,
    )
    starts = still.inits()
    draws = still.draws()
    assert len(starts) == 3
    assert set(starts[0]) == set(draws)
    for chain, values in enumerate(starts):
        for site, value in values.items():
            torch.testing.assert_close(value, draws[site][chain, 0], rtol=0, atol=0)
    # Chains start from different points.
    assert not torch.equal(starts[0]["intercept"], starts[1]["intercept"])
    joint = fit(
        bf("y ~ x") + bf("y2 ~ x") + set_rescor(False),
        data=DATA,
        algorithm="fixed_param",
        num_samples=1,
        num_chains=2,
        seed=1,
    )
    joint_draws = joint.draws()
    for chain, values in enumerate(joint.inits()):
        for site, value in values.items():
            torch.testing.assert_close(
                value, joint_draws[site][chain, 0], rtol=0, atol=0
            )


def test_inits_return_given_and_zero_starts() -> None:
    budget = {
        "num_warmup": 2,
        "num_samples": 2,
        "num_chains": 2,
        "seed": 4,
        "control": {"max_treedepth": 2},
    }
    zero = fit("y ~ x", data=DATA, init=0, **budget).inits()
    for values in zero:
        # Zero on the unconstrained scale: the log-scale sigma is one.
        assert float(values["intercept"]) == 0.0
        assert float(values["beta_x"]) == 0.0
        assert float(values["sigma"]) == 1.0
    given = fit(
        "y ~ x",
        data=DATA,
        init=[{"beta_x": 0.5, "sigma": 2.0}, {"beta_x": -0.5, "sigma": 1.0}],
        **budget,
    ).inits()
    assert [round(float(values["beta_x"]), 6) for values in given] == [0.5, -0.5]
    assert [round(float(values["sigma"]), 6) for values in given] == [2.0, 1.0]
    with pytest.raises(ValueError, match="fitted by MCMC"):
        fit("y ~ x", data=DATA, **SVI).inits()


def test_joint_bayes_r2_matches_the_residual_definition() -> None:
    joint = fit(bf("y ~ x") + bf("y2 ~ x + z") + set_rescor(True), data=DATA, **SVI)
    ratios = joint.bayes_r2(method="residual", num_samples=4, seed=3)
    assert set(ratios) == {"R2y", "R2y2"}
    expected = joint.posterior_epred(num_samples=4, seed=3)
    for name in ("y", "y2"):
        mean = expected[name].double()
        observed = torch.as_tensor(DATA[name].to_numpy())
        explained = mean.var(dim=2)
        residual = (observed - mean).var(dim=2)
        torch.testing.assert_close(
            ratios[f"R2{name}"], explained / (explained + residual), rtol=1e-10, atol=0
        )
    assert set(joint.bayes_r2(resp="y", num_samples=2)) == {"R2y"}
    stored = joint.add_criterion("bayes_R2", num_samples=3)
    assert tuple(stored.criterion("bayes_R2").shape) == (3, 2)


@pytest.mark.parametrize(
    ("law", "zero", "positive"),
    [
        (
            ZeroInflatedPoisson,
            lambda pi, rate: pi + (1 - pi) * math.exp(-rate),
            lambda pi, rate, k: (1 - pi) * stats.poisson.pmf(k, rate),
        ),
        (
            HurdlePoisson,
            lambda pi, rate: pi,
            lambda pi, rate, k: (
                (1 - pi) * stats.poisson.pmf(k, rate) / (1 - math.exp(-rate))
            ),
        ),
    ],
    ids=["zero_inflated_poisson", "hurdle_poisson"],
)
def test_inflated_poisson_laws_select_batches_and_keep_their_mass(
    law, zero, positive
) -> None:
    pi = torch.tensor([0.2, 0.5, 0.7], dtype=torch.float64)
    rate = torch.tensor([1.5, 3.0, 0.4], dtype=torch.float64)
    selected = select_batch(law(pi, rate), torch.tensor([True, False, True]))
    assert type(selected) is law
    torch.testing.assert_close(selected.zero_prob, pi[[0, 2]])
    torch.testing.assert_close(selected.rate, rate[[0, 2]])
    for index, (p, r) in enumerate([(0.2, 1.5), (0.7, 0.4)]):
        assert float(selected.log_prob(torch.tensor(0.0))[index].exp()) == (
            pytest.approx(zero(p, r), rel=1e-12)
        )
        for k in (1, 2, 5):
            assert float(selected.log_prob(torch.tensor(float(k)))[index].exp()) == (
                pytest.approx(positive(p, r, k), rel=1e-12)
            )


@pytest.mark.parametrize("component", ["zero_inflated_poisson", "hurdle_poisson"])
def test_mixture_with_inflated_poisson_component_predicts_counts(
    component: str,
) -> None:
    fitted = fit("cnt ~ x", data=DATA, family=mixture(component, "poisson"), **SVI)
    draws = fitted.posterior_predict(num_samples=40, seed=5)
    assert tuple(draws.shape) == (1, 40, N)
    assert bool((draws >= 0).all()) and bool((draws == draws.round()).all())
    new = fitted.posterior_predict(new_data=DATA.iloc[:5], num_samples=3, seed=5)
    assert tuple(new.shape) == (1, 3, 5)
    # The predictive mean of the draws agrees with the analytic mixture mean
    # within Monte Carlo error: 40 draws by 24 rows of counts near three.
    mean = fitted.posterior_epred(num_samples=40, seed=5)
    assert abs(float(draws.double().mean() - mean.double().mean())) < 0.6


def test_exact_gp_draws_at_new_inputs_follow_the_seed_in_every_method() -> None:
    fitted = fit("y ~ gp(x)", data=DATA.iloc[:12], **SVI)
    new = pd.DataFrame({"x": [-2.0, 0.33, 1.7]})
    first = fitted.posterior_linpred(new_data=new, num_samples=4, seed=9)
    again = fitted.posterior_linpred(new_data=new, num_samples=4, seed=9)
    other = fitted.posterior_linpred(new_data=new, num_samples=4, seed=10)
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    assert not torch.equal(first, other)
    # A Gaussian identity-link mean is the linear predictor, so both methods
    # must read the same draw of the latent function at the new inputs.
    torch.testing.assert_close(
        fitted.posterior_epred(new_data=new, num_samples=4, seed=9).double(),
        first.double(),
    )
    # Away from the training inputs the conditional law is not degenerate.
    assert float(first[..., 0].std()) > 1e-3


def test_automatic_effects_accept_a_monte_carlo_budget_for_truncated_means() -> None:
    fitted = fit("y | trunc(lb = -10, ub = 10) ~ x", data=DATA, **SVI)
    # The truncated Gaussian mean is closed form; the budget is optional.
    for options in ({}, {"mc_samples": 8}):
        effects = fitted.conditional_effects(resolution=3, ndraws=2, **options)
        frame = effects.effects[0].frame
        assert np.isfinite(np.array(frame["estimate__"], dtype=float)).all()
    with pytest.raises(ValueError, match="mc_samples applies only to epred"):
        fitted.conditional_effects(
            resolution=3, ndraws=2, mc_samples=8, method="linpred"
        )


def test_horseshoe_coefficient_scales_match_the_regularized_form() -> None:
    prior = RegularizedHorseshoe()
    fitted = fit("y ~ x + z", data=DATA, priors=PriorSpec(distribution=prior), **SVI)
    names = fitted.variables(num_samples=5)
    # Piironen and Vehtari's regularized scale: tau * lambda_tilde with
    # lambda_tilde^2 = c^2 lambda^2 / (c^2 + tau^2 lambda^2), where the
    # global scale is in units of sigma under autoscaling.
    tau = names["hs_global"].double() * (
        names["sigma"].double() if prior.autoscale else 1.0
    )
    slab2 = prior.scale_slab**2 * names["hs_slab"].double()
    for index, coef in enumerate(("x", "z"), start=1):
        local = names[f"hs_local[{index}]"].double()
        expected = tau * (slab2 * local**2 / (slab2 + tau**2 * local**2)).sqrt()
        torch.testing.assert_close(
            names[f"sdb_{coef}"].double(), expected, rtol=1e-5, atol=0
        )


def test_r2d2_scales_decompose_the_explained_variance() -> None:
    prior = R2D2()
    fitted = fit("y ~ x + z", data=DATA, priors=PriorSpec(distribution=prior), **SVI)
    names = fitted.variables(num_samples=5)
    ratio = names["R2D2_R2"].double()
    tau2 = (
        ratio / (1 - ratio) * (names["sigma"].double() ** 2 if prior.autoscale else 1.0)
    )
    torch.testing.assert_close(names["R2D2_tau2"].double(), tau2, rtol=1e-5, atol=0)
    for index, coef in enumerate(("x", "z"), start=1):
        torch.testing.assert_close(
            names[f"sdb_{coef}"].double() ** 2,
            names[f"R2D2_phi[{index}]"].double() * tau2,
            rtol=1e-5,
            atol=0,
        )
    assert {"sdb_x", "sdb_z", "R2D2_tau2"} <= set(
        fitted.posterior_summary(num_samples=5).rows
    )


def test_multiple_fit_pools_chains_and_keeps_per_dataset_diagnostics() -> None:
    budget = {
        "method": "nuts",
        "num_warmup": 4,
        "num_samples": 6,
        "num_chains": 1,
        "seed": 3,
        "control": {"max_treedepth": 2},
    }
    imputed = [DATA, DATA.assign(y=DATA.y + 0.2), DATA.assign(y=DATA.y - 0.2)]
    multiple = fit_multiple(bf("y ~ x", "sigma ~ z"), data=imputed, **budget)
    assert len(multiple.fits) == 3
    assert multiple.pooled.nchains() == 3
    assert multiple.pooled.ndraws() == 18
    # The pooled draws are the per-dataset draws in dataset order.
    for index, single in enumerate(multiple.fits):
        torch.testing.assert_close(
            multiple.pooled.draws()["beta_x"][index], single.draws()["beta_x"][0]
        )
    rhats = multiple.rhats()
    assert len(rhats) == 3 and set(rhats[0]) == set(multiple.fits[0].draws())
    again = multiple.update(data=imputed[:2])
    assert len(again.fits) == 2 and again.pooled.nchains() == 2
    with pytest.raises(ValueError, match="'data' is required"):
        multiple.update()


def test_smooth_labels_follow_brms_frame_sm() -> None:
    assert smooth_label("s(x, k = 5)") == "sx"
    assert smooth_label("t2(x, z, sp = c(1, 2, 3))") == "t2xz"
    assert smooth_label('s(x, g, bs = "fs", m = 1)') == "sxg"
    assert smooth_label("s(x, by = z)") == "sxz"
    with pytest.raises(ValueError, match="is not a function call"):
        smooth_label("x + z")


def test_joint_conditional_smooths_equal_the_linear_predictor_without_intercept() -> (
    None
):
    joint = fit(bf("y ~ s(x)") + bf("y2 ~ x") + set_rescor(False), data=DATA, **SVI)
    rows = DATA.iloc[[1, 4, 9, 16]]
    draws = joint.draws(num_samples=4, seed=2)
    prefix = joint.emission.responses["y"].prefix
    # y's predictor is its intercept plus the smooth, so the smooth is the
    # linear predictor minus the intercept.
    linear = joint.posterior_linpred(resp="y", new_data=rows, num_samples=4, seed=2)
    contribution = conditional_smooths(
        joint._response_fit("y"), "s(x)", rows, draws=draws, site_prefix=prefix
    )
    torch.testing.assert_close(
        contribution.double(),
        (linear["y"] - draws[prefix + "intercept"][..., None]).double(),
        rtol=1e-4,
        atol=1e-5,
    )
    effects = joint.conditional_smooths(resolution=4, ndraws=3)
    assert [effect.name for effect in effects.effects] == ["mu_y: s(x)"]
    assert len(joint.conditional_smooths(resp="y", resolution=4).effects) == 1
    with pytest.raises(ValueError, match="no valid smooth terms found"):
        joint.conditional_smooths(resp="y2", resolution=4)
    with pytest.raises(ValueError, match="for a joint fit call its"):
        conditional_smooths(joint, "s(x)", rows)


def test_effects_exclude_latent_residuals_and_fix_vector_trials_at_one() -> None:
    frame = DATA.assign(time=np.arange(N) % 6, sub=np.arange(N) // 6)
    latent = fit(
        "cnt ~ x + ar(time = time, gr = sub)", data=frame, family="poisson", **SVI
    )
    effect = latent.conditional_effects(resolution=3, draw_ids=[0, 1, 2, 3]).effects[0]
    # brms's incl_autocor = FALSE: the effect is the mean of the population
    # part alone, which depends on x only, so any rows with the grid's x
    # reproduce the effect's draws.
    grid = frame.iloc[:3].assign(x=np.array(effect.frame["x"], dtype=float))
    mean = latent.posterior_epred(
        new_data=grid, incl_autocor=False, num_samples=4, seed=0
    )
    torch.testing.assert_close(
        effect.draws.reshape(-1, 3).double(),
        mean.reshape(-1, 3).double(),
        rtol=1e-6,
        atol=0,
    )
    counts = DATA.assign(c1=DATA.cnt, c2=DATA.cnt + 1, c3=2)
    multinomial = fit("cbind(c1, c2, c3) ~ x", data=counts, family="multinomial", **SVI)
    shares = (
        multinomial.conditional_effects(resolution=3, ndraws=4, categorical=True)
        .effects[0]
        .frame
    )
    # With one trial the expected counts are the category probabilities.
    totals = shares.groupby("x")["estimate__"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=0.05)


def test_criteria_score_the_complete_response_of_a_missing_data_model() -> None:
    frame = DATA.assign(xmis=DATA.x.where(np.arange(N) % 7 != 0))
    joint = fit(
        bf("y ~ mi(xmis) + z") + bf("xmis | mi() ~ z") + set_rescor(False),
        data=frame,
        **SVI,
    )
    whole = joint.loo(num_samples=30)
    assert whole.responses == ("y", "xmis")
    assert math.isfinite(float(whole.elpd))
    scored = joint.add_criterion(["loo", "waic"], resp="y", num_samples=30)
    assert scored.criterion("loo").responses == ("y",)
    assert scored.loo(num_samples=30).responses == whole.responses
    assert math.isfinite(float(scored.criterion("loo").elpd))
    assert math.isfinite(float(scored.criterion("waic").elpd))
    # The stored estimate is the estimate of the same response alone.
    assert float(scored.criterion("loo").elpd) == pytest.approx(
        float(joint.loo(resp="y", num_samples=30).elpd)
    )


def test_scalar_coordinates_index_from_one_in_row_major_order() -> None:
    value = torch.arange(2 * 3 * 2 * 3, dtype=torch.float64).reshape(2, 3, 2, 3)
    names = scalar_coordinates("v", value)
    assert list(names) == ["v[1,1]", "v[1,2]", "v[1,3]", "v[2,1]", "v[2,2]", "v[2,3]"]
    torch.testing.assert_close(names["v[2,1]"], value[..., 1, 0])
    assert list(scalar_coordinates("s", value[..., 0, 0])) == ["s"]


def test_correlation_pairs_follow_the_matrix() -> None:
    factor = torch.tensor(
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]], dtype=torch.float64
    ).expand(1, 2, 3, 3)
    matrix = factor @ factor.transpose(-1, -2)
    pairs = correlation_pairs("cor_g", ["a", "b", "c"], matrix)
    assert list(pairs) == ["cor_g__a__b", "cor_g__a__c", "cor_g__b__c"]
    assert float(pairs["cor_g__a__b"][0, 0]) == pytest.approx(0.6)
    assert float(pairs["cor_g__a__c"][0, 0]) == pytest.approx(0.0)
    assert float(pairs["cor_g__b__c"][0, 0]) == pytest.approx(0.48)
    with pytest.raises(ValueError, match="does not match the 2 coordinates"):
        correlation_pairs("cor_g", ["a", "b"], matrix)


def test_catalogue_refuses_a_variable_with_an_event_axis() -> None:
    with pytest.raises(ValueError, match=r"variable 'v' has shape \(1, 2, 3\)"):
        checked_variables({"v": torch.zeros(1, 2, 3)}, (1, 2))
    assert list(checked_variables({"v": torch.zeros(1, 2)}, (1, 2))) == ["v"]


def test_psis_refuses_too_few_draws_with_the_count() -> None:
    fitted = fit("y ~ x", data=DATA, **SVI)
    with pytest.raises(ValueError, match="24 pooled draws are fewer than 25"):
        fitted.loo(num_samples=24)
    with pytest.raises(ValueError, match="3 pooled draws are fewer than 25"):
        fitted.psis(num_samples=3)
    assert math.isfinite(float(fitted.loo(num_samples=25).elpd))
