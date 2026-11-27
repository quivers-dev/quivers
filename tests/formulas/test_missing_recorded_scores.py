"""Pointwise scores of joint ``mi()`` fits on rows with missing values.

On a training row where a response has no recorded value the program
completes it. The pointwise score of the row is the log density of its
recorded values under the joint law conditional on the completed values,
``log p(y_O | y_M)``; a row with no recorded value scores zero.

The oracle below writes the conditional law by the Schur complement,
``N(m_O + S_OM S_MM^-1 (y_M - m_M), S_OO - S_OM S_MM^-1 S_MO)``, from the
drawn parameters, in double precision, independently of the subtraction
``log p(v) - log p_M(v_M)`` that the reader evaluates. The conditional AR(1)
means are transcribed on the completed responses of the earlier rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import bf, fit, set_rescor

ROWS = 12
MISSING_Y1 = (3, 9)
MISSING_Y2 = (5, 9)
SVI = {"method": "svi", "num_steps": 2, "seed": 0}


def _frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": generator.normal(size=ROWS).round(3),
            "y1": generator.normal(size=ROWS).round(3),
            "y2": generator.normal(size=ROWS).round(3),
            "t": list(range(6)) * 2,
            "g": ["a"] * 6 + ["b"] * 6,
        }
    )
    frame.loc[list(MISSING_Y1), "y1"] = np.nan
    frame.loc[list(MISSING_Y2), "y2"] = np.nan
    return frame


def _spec(recursion: bool):
    term = " + ar(time = t, gr = g)" if recursion else ""
    return bf(f"y1 | mi() ~ x{term}") + bf("y2 | mi() ~ x") + set_rescor(True)


def _oracle(
    frame: pd.DataFrame, draws: dict[str, torch.Tensor], recursion: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the row scores, the y1 scores and the recorded-only y1 scores.

    The last are the scores a reader would give that ignored the
    completion: the marginal density of the recorded values alone.
    """
    x = torch.as_tensor(frame["x"].to_numpy())
    y = torch.as_tensor(frame[["y1", "y2"]].to_numpy()).clone()
    y[list(MISSING_Y1), 0] = draws["mi_y1__missing"].reshape(-1)
    y[list(MISSING_Y2), 1] = draws["mi_y2__missing"].reshape(-1)
    eta1 = draws["response_y1__intercept"] + draws["response_y1__beta_x"] * x
    eta2 = draws["response_y2__intercept"] + draws["response_y2__beta_x"] * x
    mean1 = eta1.clone()
    if recursion:
        phi = draws["response_y1__ar_1"].reshape(())
        for row in range(ROWS):
            if row % 6:
                mean1[row] = eta1[row] + phi * (y[row - 1, 0] - eta1[row - 1])
    means = torch.stack((mean1, eta2), -1)
    scales = torch.stack(
        (
            draws["response_y1__sigma"].reshape(()),
            draws["response_y2__sigma"].reshape(()),
        )
    )
    factor = scales[:, None] * draws["rescor"].reshape(2, 2)
    covariance = factor @ factor.T
    recorded = ~torch.as_tensor(np.isnan(frame[["y1", "y2"]].to_numpy()))
    scores = torch.zeros(ROWS, dtype=torch.float64)
    alone = torch.zeros(ROWS, dtype=torch.float64)
    for row in range(ROWS):
        seen = [k for k in range(2) if recorded[row, k]]
        hidden = [k for k in range(2) if not recorded[row, k]]
        if not seen:
            continue
        m, v = means[row], y[row]
        if hidden:
            gain = covariance[seen][:, hidden] @ torch.linalg.inv(
                covariance[hidden][:, hidden]
            )
            loc = m[seen] + gain @ (v[hidden] - m[hidden])
            cov = covariance[seen][:, seen] - gain @ covariance[hidden][:, seen]
        else:
            loc, cov = m[seen], covariance[seen][:, seen]
        scores[row] = torch.distributions.MultivariateNormal(
            loc, covariance_matrix=cov
        ).log_prob(v[seen])
        alone[row] = torch.distributions.MultivariateNormal(
            m[seen], covariance_matrix=covariance[seen][:, seen]
        ).log_prob(v[seen])
    # resp="y1": the marginal of y1, conditioned on nothing else.
    first = torch.distributions.Normal(means[:, 0], covariance[0, 0].sqrt()).log_prob(
        y[:, 0]
    )
    first = torch.where(recorded[:, 0], first, torch.zeros_like(first))
    return scores, first, alone


@pytest.mark.parametrize("recursion", [False, True])
def test_recorded_values_are_scored_given_the_completions(recursion):
    frame = _frame()
    fitted = fit(_spec(recursion), data=frame, **SVI)
    draws = {
        name: value.double()[0, 0]
        for name, value in fitted.draws(num_samples=1, seed=0).items()
    }
    scores, first, alone = _oracle(frame, draws, recursion)
    joint = fitted.log_likelihood(num_samples=1, seed=0)["y1__y2"][0, 0].double()
    assert bool(torch.isfinite(joint).all())
    torch.testing.assert_close(joint, scores, rtol=1e-5, atol=1e-5)
    # Row 9 has no recorded value and scores zero.
    assert float(joint[9]) == 0.0
    # Ignoring the completion gives other numbers at the partly missing rows,
    # by more than ten times the tolerance.
    partly = [3, 5]
    assert bool((alone[partly] - joint[partly]).abs().gt(1e-4).all())
    marginal = fitted.log_likelihood(resp="y1", num_samples=1, seed=0)["y1"][0, 0]
    torch.testing.assert_close(marginal.double(), first, rtol=1e-5, atol=1e-5)


def test_independent_responses_score_zero_at_missing_rows():
    frame = _frame()
    fitted = fit(
        bf("y1 | mi() ~ x") + bf("y2 | mi() ~ x") + set_rescor(False),
        data=frame,
        **SVI,
    )
    draws = {
        name: value.double()[0, 0]
        for name, value in fitted.draws(num_samples=1, seed=0).items()
    }
    scores = fitted.log_likelihood(resp="y1", num_samples=1, seed=0)["y1"][0, 0]
    x = torch.as_tensor(frame["x"].to_numpy())
    y1 = torch.as_tensor(np.nan_to_num(frame["y1"].to_numpy()))
    wanted = torch.distributions.Normal(
        draws["response_y1__intercept"] + draws["response_y1__beta_x"] * x,
        draws["response_y1__sigma"],
    ).log_prob(y1)
    wanted[list(MISSING_Y1)] = 0.0
    torch.testing.assert_close(scores.double(), wanted, rtol=1e-5, atol=1e-5)
