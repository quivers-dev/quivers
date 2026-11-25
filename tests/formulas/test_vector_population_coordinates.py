"""Centered and QR coordinates of vector and ordinal predictors.

The references are NumPy evaluations of the predictor from the sampled
coordinates, the column means, and NumPy's own QR factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import fit, get_prior
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry

ROWS = 24
RNG = np.random.default_rng(5)
FRAME = pd.DataFrame(
    {
        "c": ["A", "B", "C"] * 8,
        "o": [0, 1, 2, 3] * 6,
        "x": RNG.normal(size=ROWS) + 2.0,
        "w": RNG.normal(size=ROWS) - 1.0,
    }
)
DESIGN = FRAME[["x", "w"]].to_numpy()


def _pinned(fitted, seed=3, draws=2):
    names = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names
    generator = torch.Generator().manual_seed(seed)
    samples = {
        name: torch.randn(
            (1, draws, *fitted.draws(num_samples=1)[name].shape[2:]),
            generator=generator,
            dtype=torch.float64,
        )
        for name in names
    }
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, draws),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    ), {name: value[0].numpy() for name, value in samples.items()}


def _fit(formula, family, **options):
    return fit(formula, data=FRAME, family=family, method="svi", num_steps=0, **options)


def _qr(design):
    q, r = np.linalg.qr(design)
    signs = np.sign(np.diag(r))
    scale = np.sqrt(len(design) - 1.0)
    return q * signs * scale, signs[:, None] * r / scale


@pytest.mark.parametrize("decomp", ["none", "QR"])
def test_category_predictors_read_the_centered_design(decomp):
    """Each category has its own intercept on the same transformed design."""
    fitted, values = _pinned(
        _fit("c ~ x + w", "categorical", center=True, decomp=decomp)
    )
    centered = DESIGN - DESIGN.mean(axis=0)
    suffix = "" if decomp == "none" else "_qr"
    basis = centered if decomp == "none" else _qr(centered)[0]
    coefficients = np.stack(
        [values[f"beta_{name}{suffix}"] for name in ("x", "w")], axis=1
    )
    expected = values["intercept_centered"][:, None] + np.einsum(
        "djk,nj->dnk", coefficients, basis
    )
    for index, category in enumerate(("muB", "muC")):
        np.testing.assert_allclose(
            fitted.posterior_linpred(parameter=category)[0].numpy(),
            expected[..., index],
            rtol=1e-5,
            atol=1e-5,
        )
    slopes = coefficients
    if decomp == "QR":
        slopes = np.einsum("jl,dlk->djk", np.linalg.inv(_qr(centered)[1]), coefficients)
    named = fitted.variables()
    for index, category in enumerate("BC"):
        for column, name in enumerate(("x", "w")):
            np.testing.assert_allclose(
                named[f"b_mu{category}_{name}"][0].numpy(),
                slopes[:, column, index],
                rtol=1e-5,
                atol=1e-6,
            )
        np.testing.assert_allclose(
            named[f"b_mu{category}_Intercept"][0].numpy(),
            values["intercept_centered"][:, index]
            - slopes[:, :, index] @ DESIGN.mean(axis=0),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("family", ["cumulative", "acat"])
def test_ordinal_predictors_read_the_centered_design(family):
    """The thresholds keep their place and the location intercept moves."""
    fitted, values = _pinned(_fit("o ~ x + w", family, center=True))
    centered = DESIGN - DESIGN.mean(axis=0)
    expected = (
        values["intercept_centered"][:, None]
        + values["beta_x"][:, None] * centered[:, 0]
        + values["beta_w"][:, None] * centered[:, 1]
    )
    np.testing.assert_allclose(
        fitted.posterior_linpred()[0].numpy(), expected, rtol=1e-5, atol=1e-5
    )


def test_category_specific_columns_keep_their_coordinates():
    """Category-specific columns stay outside the centered design, as in brms."""
    priors = get_prior(
        "o ~ x + cs(w)", data=FRAME, family="acat", center=True, decomp="QR"
    )
    assert {"intercept_centered", "beta_x_qr"} <= set(priors)
    assert not any("w" in name and name.endswith("_qr") for name in priors)


@pytest.mark.parametrize(
    "options", [{}, {"distributional": {"muB": "~ x"}}], ids=["shared", "separate"]
)
def test_category_intercepts_have_the_brms_default(options):
    """brms gives each category's centered intercept student_t(3, 0, 2.5)."""
    priors = get_prior(
        "c ~ x", data=FRAME, family="categorical", default_priors="brms", **options
    )
    intercepts = [name for name in priors if name.endswith("intercept_centered")]
    assert intercepts
    assert {priors[name] for name in intercepts} == {
        "StudentT(df=3.0, loc=0.0, scale=2.5)"
    }
