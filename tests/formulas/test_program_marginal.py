"""The two readings of a marginalized latent in a program with formulas.

Oracle: the log joint of the issue's approximate-numeral program written out
in torch, at float64. A ``marginalize`` without ``over=`` draws one reading
for every row, so the rows' likelihoods multiply before the reading is summed
out (-5.647981643676758 on the six rows below); the grouped form
``[over=Trial]`` sums out one reading per row (-5.0164475440979). The formulas
never choose between them: a parameter that varies over the rows cannot feed
the law of a reading the rows share, and the refusal says how to state the
per-row model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.effects.units import UnfactoredObservationError
from quivers.formulas import fit, formula_to_qvr

from tests.formulas._program_cases import (
    log_joint,
    AROUND,
    AROUND_ROWS,
    HALFWIDTH,
    pin,
    run,
)

RES_LOGITS = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
NOISE = torch.tensor(0.1, dtype=torch.float64)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dist": [0.5, 1.5, 2.5, 0.2, 1.2, 3.5],
            "accept": [True, False, True, True, False, False],
            "trial_idx": np.arange(6),
        }
    )


def _row_terms() -> torch.Tensor:
    """Log likelihood of each row under each reading, shape (rows, readings)."""
    rows = _rows()
    dist = torch.tensor(rows["dist"].to_numpy(), dtype=torch.float64)
    accept = torch.tensor(rows["accept"].to_numpy(), dtype=torch.float64)
    inside = (dist[:, None] <= torch.as_tensor(HALFWIDTH)[None]).double()
    p = NOISE + (1.0 - 2.0 * NOISE) * inside
    return torch.distributions.Bernoulli(probs=p).log_prob(accept[:, None])


def _prior() -> torch.Tensor:
    return torch.distributions.Normal(0.0, 1.0).log_prob(
        RES_LOGITS
    ).sum() + torch.distributions.Beta(1.0, 20.0).log_prob(NOISE)


def _weights() -> torch.Tensor:
    return torch.log_softmax(RES_LOGITS, -1)


def _fit(source: str, parameters: dict[str, str] | None = None):
    return fit(
        program=source,
        parameters=parameters or {},
        response="accept",
        data=_rows(),
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=1,
        seed=0,
    )


def test_the_two_readings_differ() -> None:
    shared = torch.logsumexp(_weights() + _row_terms().sum(0), -1) + _prior()
    per_row = torch.logsumexp(_weights() + _row_terms(), -1).sum() + _prior()
    assert float(shared) == pytest.approx(-5.647981643676758, abs=1e-6)
    assert float(per_row) == pytest.approx(-5.0164475440979, abs=1e-6)
    assert abs(float(shared - per_row)) > 0.5


def test_the_issue_program_shares_one_reading() -> None:
    fitted = _fit(AROUND)
    recorded = run(fitted, {"res_logits": RES_LOGITS, "noise": NOISE})
    shared = torch.logsumexp(_weights() + _row_terms().sum(0), -1) + _prior()
    assert log_joint(recorded) == pytest.approx(float(shared), rel=1e-10)


def test_the_grouped_program_reads_once_per_row() -> None:
    fitted = _fit(AROUND_ROWS)
    recorded = run(fitted, {"res_logits": RES_LOGITS, "noise": NOISE})
    per_row = torch.logsumexp(_weights() + _row_terms(), -1)
    expected = per_row.sum() + _prior()
    assert log_joint(recorded) == pytest.approx(float(expected), rel=1e-10)
    pinned = pin(
        fitted,
        {
            "res_logits": RES_LOGITS.numpy()[None],
            "noise": NOISE.numpy()[None],
        },
    )
    scores = pinned.log_likelihood()["accept"].reshape(-1).double()
    torch.testing.assert_close(scores, per_row, rtol=1e-10, atol=1e-12)
    # The rows' scores sum to the marginal score of the block.
    assert float(scores.sum()) == pytest.approx(
        float(log_joint(recorded) - float(_prior())), rel=1e-10
    )


def test_a_row_varying_weight_of_a_shared_reading_is_refused() -> None:
    with pytest.raises(ValueError, match=r"reads \['res_logits'\].*over=Trial"):
        formula_to_qvr(
            program=AROUND,
            parameters={"res_logits": "~ dist"},
            response="accept",
            data=_rows(),
            data2={"halfwidth": HALFWIDTH},
        )
    # A weight constant over the rows keeps the shared reading.
    source = formula_to_qvr(
        program=AROUND,
        parameters={"res_logits": "~ 1"},
        response="accept",
        data=_rows(),
        data2={"halfwidth": HALFWIDTH},
    )
    assert "[reduction=logsumexp]" in source
    # A row-varying response noise beside the shared reading is a model of
    # its own, whose rows do not factor.
    fitted = _fit(AROUND, {"noise": "~ dist"})
    with pytest.raises(UnfactoredObservationError, match="share one 'r'"):
        fitted.log_likelihood(num_samples=2)


def test_per_row_weights_need_the_grouped_block() -> None:
    fitted = _fit(AROUND_ROWS, {"res_logits": "~ dist", "noise": "~ dist"})
    scores = fitted.log_likelihood(num_samples=2)["accept"]
    assert scores.shape == (1, 2, 6)
    assert bool(torch.isfinite(scores).all())
    assert not math.isnan(float(scores.sum()))
