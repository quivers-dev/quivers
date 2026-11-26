"""Prediction from a program fit on fresh rows and fresh group levels.

Oracles: the expected response of the grouped approximate-numeral program is
the class-weighted acceptance probability ``sum_r softmax(res_logits)_r
p_r(row)``, written out in torch at the pinned draws; predictive draws of
``accept`` agree with it within four binomial standard errors. A fresh level
of a grouping factor takes the group law, and a known level its fitted
effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from quivers.formulas import fit

from tests.formulas._program_cases import (
    AROUND_ROWS,
    GAUSS,
    HALFWIDTH,
    frame,
    latent_names,
    pin,
)


def _around(draws: int):
    fitted = fit(
        program=AROUND_ROWS,
        parameters={},
        response="accept",
        data=frame(),
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=1,
        seed=0,
    )
    return pin(
        fitted,
        {
            "res_logits": np.tile([[0.2, -0.1, 0.4]], (draws, 1)),
            "noise": np.full(draws, 0.1),
        },
    )


def _fresh() -> pd.DataFrame:
    return pd.DataFrame(
        {"dist": [0.5, 1.5, 2.5, 3.5], "trial_idx": np.arange(4), "x": 0.0}
    )


def _expected(dist: np.ndarray) -> torch.Tensor:
    weights = torch.softmax(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64), -1)
    inside = (
        torch.as_tensor(dist)[:, None] <= torch.as_tensor(HALFWIDTH)[None]
    ).double()
    p = 0.1 + (1.0 - 2.0 * 0.1) * inside
    return (p * weights).sum(-1)


def test_expected_response_on_fresh_rows_is_the_class_weighted_mean() -> None:
    fitted = _around(draws=2)
    expected = _expected(_fresh()["dist"].to_numpy())
    epred = fitted.posterior_epred(new_data=_fresh())
    assert epred.shape == (1, 2, 4)
    for draw in range(2):
        torch.testing.assert_close(
            epred[0, draw].double(), expected, rtol=1e-10, atol=1e-12
        )
    # On the training rows as well.
    training = fitted.posterior_epred()[0, 0].double()
    torch.testing.assert_close(
        training, _expected(frame()["dist"].to_numpy()), rtol=1e-10, atol=1e-12
    )


def test_predictive_frequencies_agree_with_the_expected_response() -> None:
    draws = 4000
    fitted = _around(draws=draws)
    predicted = fitted.posterior_predict(new_data=_fresh(), seed=3)
    assert predicted.shape == (1, draws, 4)
    frequency = predicted[0].double().mean(0)
    expected = _expected(_fresh()["dist"].to_numpy())
    bound = 4.0 * torch.sqrt(expected * (1.0 - expected) / draws)
    assert bool(((frequency - expected).abs() <= bound).all()), (frequency, expected)


def test_fresh_and_known_levels_of_a_program_parameter() -> None:
    fitted = fit(
        program=GAUSS,
        parameters={"mu": "~ 1 + (1 | g)"},
        response="y",
        data=frame(),
        method="svi",
        num_steps=1,
        seed=0,
    )
    names = latent_names(fitted)
    assert set(names) == {"intercept", "sigma_g_Intercept", "z_g_Intercept", "sigma"}
    z = np.array([[0.5, -1.0, 2.0]])
    pinned = pin(
        fitted,
        {
            "intercept": np.array([1.5]),
            "sigma_g_Intercept": np.array([0.3]),
            "z_g_Intercept": z,
            "sigma": np.array([0.7]),
        },
    )
    known = pd.DataFrame({"g": ["a", "b", "c"]})
    epred = pinned.posterior_epred(new_data=known)[0, 0].double()
    torch.testing.assert_close(
        epred, torch.as_tensor(1.5 + 0.3 * z[0]), rtol=1e-6, atol=1e-6
    )
    fresh = pd.DataFrame({"g": ["new", "new"]})
    predicted = pinned.posterior_epred(new_data=fresh, allow_new_levels=True)
    assert predicted.shape == (1, 1, 2)
    assert bool(torch.isfinite(predicted).all())
    # One fresh level is one draw of its effect, shared by its rows.
    assert float(predicted[0, 0, 0]) == float(predicted[0, 0, 1])
