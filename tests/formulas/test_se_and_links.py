"""Known standard errors follow brms's ``se(sigma = FALSE)`` default, and links
are restricted to those brms admits for each family parameter.

Oracles: hand-written Normal, Student-t, and mean-parameterized skew-normal
densities (SciPy), with the residual scale either absent (``se(e)``) or
combined in quadrature (``se(e, sigma=True)``); the brms link tables in
``R/family-lists.R`` and ``links_dpars`` in ``R/families.R`` transcribed in
`quivers.formulas._link_whitelist`. Tolerances: 1e-6 relative for float32
response densities.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
    links,
)
from quivers.formulas._link_whitelist import (
    AUXILIARY_CLASSES,
    AUXILIARY_LINKS,
    LOCATION_EXTENSIONS,
    LOCATION_LINKS,
    admissible_links,
)
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace

RTOL = 1e-6


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [-0.2, 0.3, 0.8, 0.1],
            "x": [-0.4, 0.2, 0.5, 0.8],
            "e": [0.2, 0.3, 0.4, 0.5],
        }
    )


def _scored(family: str, formula: str, fixed: dict[str, float]):
    frame = _frame()
    parsed = formula_from_data(formula, frame)
    lens = FormulaToQVRModule(families[family], fixed_parameters=fixed)
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    assert isinstance(program, MonadicProgram)
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
    recorded = trace(
        program,
        torch.zeros(1, 4),
        lens.fixed_column_observations(parsed)
        | {"y": y, "intercept": torch.tensor(0.2), "beta_x": torch.tensor(-0.3)},
    )
    sampleable = recorded.sites["y"].sampleable
    assert sampleable is not None
    scores = sampleable.raw_log_prob_pointwise(y)
    return frame, scores.detach().numpy(), recorded


def _skew_normal_logpdf(y, mean, scale, alpha):
    """brms's mean parameterization of the skew normal."""
    delta = alpha / math.sqrt(1 + alpha**2)
    center = math.sqrt(2 / math.pi) * delta
    omega = scale / np.sqrt(1 - center**2)
    return stats.skewnorm.logpdf(y, alpha, mean - omega * center, omega)


@pytest.mark.parametrize("family", ["gaussian", "student_t", "skew_normal"])
@pytest.mark.parametrize("keep_sigma", [False, True])
def test_known_errors_alone_scale_the_likelihood_unless_sigma_is_kept(
    family, keep_sigma
):
    fixed = {"nu": 4.0} if family == "student_t" else {}
    fixed |= {"alpha": 1.5} if family == "skew_normal" else {}
    if keep_sigma:
        fixed["sigma"] = 0.7
    formula = "y | se(e, sigma=True) ~ x" if keep_sigma else "y | se(e) ~ x"
    frame, actual, recorded = _scored(family, formula, fixed)
    mean = 0.2 - 0.3 * frame.x.to_numpy()
    scale = np.hypot(frame.e.to_numpy(), 0.7 if keep_sigma else 0.0)
    y = frame.y.to_numpy()
    if family == "gaussian":
        expected = stats.norm.logpdf(y, mean, scale)
    elif family == "student_t":
        expected = stats.t.logpdf(y, 4.0, mean, scale)
    else:
        expected = _skew_normal_logpdf(y, mean, scale, 1.5)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL)
    priors = get_prior(formula, data=frame, family=family)
    assert ("sigma" in priors) is keep_sigma or "sigma" in fixed
    if not keep_sigma:
        assert "sigma" not in recorded.sites


def test_default_known_errors_require_positive_errors_and_refuse_a_sigma_formula():
    frame = _frame().assign(e=[0.0, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="positive standard errors"):
        formula_from_data("y | se(e) ~ x", frame)
    with pytest.raises(ValueError, match=r"sigma=True"):
        fit(
            "y | se(e) ~ x",
            data=_frame(),
            distributional={"sigma": "~ x"},
            method="svi",
            num_steps=1,
        )


@pytest.mark.parametrize("family", ["gamma", "poisson", "lognormal", "beta"])
def test_se_is_refused_with_the_family_named(family):
    frame = _frame().assign(y=[0.2, 0.3, 0.8, 0.1])
    if family == "poisson":
        frame = frame.assign(y=[0, 1, 2, 1])
    with pytest.raises(ValueError, match=rf"se\(\).*'{family}'"):
        fit("y | se(e) ~ x", data=frame, family=family, method="svi", num_steps=1)


def test_prediction_with_known_errors_alone_has_the_error_spread():
    frame = _frame()
    fitted = fit("y | se(e) ~ x", data=frame, method="svi", num_steps=1, seed=3)
    draws = 4000
    pinned = fitted.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((1, draws), 0.2),
                "beta_x": torch.full((1, draws), -0.3),
            },
            log_densities=torch.zeros(1, draws),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    )
    fresh = pd.DataFrame({"x": [0.1, 0.6], "e": [0.05, 2.0]})
    predicted = pinned.posterior_predict(new_data=fresh, seed=11).reshape(draws, 2)
    mean = 0.2 - 0.3 * fresh.x.to_numpy()
    observed_sd = predicted.double().std(0).numpy()
    # Standard error of a sample standard deviation is sd / sqrt(2 (n - 1)).
    allowed = 5 * fresh.e.to_numpy() / math.sqrt(2 * (draws - 1))
    assert np.all(np.abs(observed_sd - fresh.e.to_numpy()) < allowed)
    observed_mean = predicted.double().mean(0).numpy()
    assert np.all(
        np.abs(observed_mean - mean) < 5 * fresh.e.to_numpy() / math.sqrt(draws)
    )


# Links -----------------------------------------------------------------------


def test_every_registered_family_has_a_link_decision_and_admits_its_defaults():
    assert set(families) <= set(LOCATION_LINKS)
    for name, family in families.items():
        location = (
            family.ordinal.link
            if family.ordinal is not None
            else family.location_link.name
        )
        assert location in admissible_links(name, "mu"), name
        for aux in family.aux_params:
            assert aux.link.name in admissible_links(name, aux.name), (name, aux.name)
    every = {link for table in LOCATION_LINKS.values() for link in table}
    every |= {link for table in AUXILIARY_LINKS.values() for link in table}
    assert every <= set(links)


def _admitted_cases():
    for name, family in families.items():
        if getattr(family, "mixture", None) is not None:
            # A finite mixture takes the links of its component families.
            continue
        for link in LOCATION_LINKS[name] + LOCATION_EXTENSIONS.get(name, ()):
            yield name, "mu", link
        for aux in family.aux_params:
            for link in AUXILIARY_LINKS[AUXILIARY_CLASSES.get(aux.name, aux.name)]:
                yield name, aux.name, link


@pytest.mark.parametrize(("name", "parameter", "link"), list(_admitted_cases()))
def test_every_admitted_link_is_accepted(name, parameter, link):
    family = families[name].with_links(**{parameter: link})
    if parameter == "mu" and family.ordinal is not None:
        assert family.ordinal.link == link
    elif parameter == "mu":
        expected = (
            "softmax" if link == "logit" and family.response is not None else link
        )
        assert family.location_link.name == expected
    else:
        assert {aux.name: aux.link.name for aux in family.aux_params}[parameter] == link


@pytest.mark.parametrize(
    ("name", "parameter", "link"),
    [
        ("lognormal", "mu", "log"),
        ("poisson", "mu", "logit"),
        ("gamma", "mu", "logit"),
        ("beta_binomial", "mu", "log"),
        ("gaussian", "sigma", "logit"),
        ("student_t", "nu", "log"),
        ("zero_inflated_poisson", "zi", "probit"),
        ("gen_extreme_value", "xi", "log"),
        ("von_mises", "mu", "logit"),
        ("cratio", "mu", "softit"),
        ("negative_binomial", "disp", "logit"),
    ],
)
def test_links_brms_rejects_are_refused_with_the_admissible_links(
    name, parameter, link
):
    with pytest.raises(ValueError, match="admissible links") as caught:
        families[name].with_links(**{parameter: link})
    message = str(caught.value)
    assert repr(link) in message and repr(name) in message


def test_stated_link_extensions_are_beyond_brms_and_real_valued():
    for name, extra in LOCATION_EXTENSIONS.items():
        assert not set(extra) & set(LOCATION_LINKS[name])
        assert families[name].location_link.name == "identity"
    with pytest.raises(ValueError, match="admissible links"):
        families["lognormal"].with_links(mu="sqrt")
