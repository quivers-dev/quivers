"""Model comparison of programs fit on one design.

Oracle: the pointwise log likelihood of each fit, from which PSIS-LOO is a
reweighting; ``kfold`` refits each program on held-out folds with the same
formulas. Two programs on one design differ only in the program, so their
comparison reads the same rows. A program whose rows share a marginalized
latent has no pointwise likelihood, and every criterion that needs one says
why.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quivers.effects.units import UnfactoredObservationError
from quivers.formulas import fit, loo_compare

from tests.formulas._program_cases import AROUND, DECAY, GAUSS, HALFWIDTH, frame

DESIGN = {"mu": "~ x + (1 | g)"}


def _fits():
    rows = frame(rows=16, seed=3)
    common = {"response": "y", "data": rows, "method": "svi", "num_steps": 30}
    gauss = fit(program=GAUSS, parameters=DESIGN, seed=1, **common)
    decay = fit(program=DECAY, parameters={"a": "~ 1 + (1 | g)"}, seed=1, **common)
    return gauss, decay


def test_loo_and_compare_of_two_programs() -> None:
    gauss, decay = _fits()
    estimates = {}
    for name, fitted in (("gauss", gauss), ("decay", decay)):
        scores = fitted.log_likelihood(num_samples=40)["y"]
        assert scores.shape == (1, 40, 16)
        estimate = fitted.loo(num_samples=40)
        # The effective number of parameters is the in-sample log predictive
        # density less the LOO estimate, both read from the same scores.
        lppd = torch.logsumexp(scores.reshape(40, 16).double(), 0) - np.log(40)
        assert estimate.p == pytest.approx(float(lppd.sum()) - estimate.elpd, rel=1e-6)
        estimates[name] = fitted.add_criterion("loo", num_samples=40)
    table = loo_compare(estimates)
    assert set(table.rows) == {"gauss", "decay"}


def test_kfold_of_a_program() -> None:
    gauss, _ = _fits()
    result = gauss.kfold(K=2, seed=0, num_samples=20)
    pointwise = torch.as_tensor(result.estimate.pointwise_elpd)
    assert pointwise.shape[-1] == 16
    assert bool(torch.isfinite(pointwise).all())


def test_a_shared_reading_has_no_pointwise_criterion() -> None:
    rows = frame(rows=12, seed=5)
    fitted = fit(
        program=AROUND,
        parameters={"noise": "~ 1"},
        response="accept",
        data=rows,
        data2={"halfwidth": HALFWIDTH},
        method="svi",
        num_steps=2,
        seed=0,
    )
    with pytest.raises(UnfactoredObservationError, match="share one 'r'"):
        fitted.loo(num_samples=40)
