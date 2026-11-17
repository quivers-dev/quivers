"""``posterior_linpred(parameter=)`` of an auxiliary parameter without a formula.

brms 2.23.0 ``posterior_epred.brmsprep`` returns, for a ``dpar`` that is not
predicted, ``matrix(dpars[[dpar]], nrow = ndraws, ncol = nobs)``: each draw of
the parameter repeated for every observation, on the parameter's own scale
for both ``scale = "linear"`` and ``scale = "response"``. The tests compare
with the retained draws exactly.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.diagnostics.posterior_tables import SummaryTable
from quivers.formulas import MultivariateFormula, bf, fit, mixture, set_rescor


def _frame() -> pd.DataFrame:
    rows = np.arange(24)
    frame = pd.DataFrame({"x": np.sin(rows * 0.7), "z": np.cos(rows * 1.3)})
    frame["y"] = 1.0 + 0.5 * frame["x"] + 0.2 * np.cos(rows * 2.1)
    frame["y2"] = 0.5 - 0.2 * frame["x"] + 0.1 * np.sin(rows * 1.9)
    frame["count"] = (rows * 7) % 5
    return frame


CASES = [
    ("y ~ x", "gaussian", "sigma"),
    ("y ~ x", "student_t", "nu"),
    ("y ~ x", "student_t", "sigma"),
    ("count ~ x", "negative_binomial", "disp"),
    ("count ~ x", "zero_inflated_poisson", "zi"),
    ("y ~ x", "skew_normal", "alpha"),
]


@pytest.mark.parametrize(("formula", "family", "parameter"), CASES)
def test_constant_parameter_draws_are_repeated_over_rows(formula, family, parameter):
    frame = _frame()
    fitted = fit(formula, data=frame, family=family, method="svi", num_steps=3, seed=1)
    draws = fitted.draws(num_samples=4, seed=2)[parameter]
    assert draws.shape == (1, 4)
    for transform in (False, True):
        training = fitted.posterior_linpred(
            parameter=parameter, transform=transform, num_samples=4, seed=2
        )
        assert training.shape == (1, 4, 24)
        assert torch.equal(training, draws[..., None].expand(1, 4, 24))
        fresh = fitted.posterior_linpred(
            parameter=parameter,
            transform=transform,
            new_data=frame.iloc[:5],
            num_samples=4,
            seed=2,
        )
        assert torch.equal(fresh, draws[..., None].expand(1, 4, 5))
    assert fitted.predictor_labels(parameter) == ()
    table = fitted.fitted(parameter=parameter, num_samples=4, seed=2)
    assert isinstance(table, SummaryTable)
    estimate = table.values[..., table.columns.index("Estimate")].reshape(-1)
    torch.testing.assert_close(
        estimate.double(), draws.double().mean().expand(24), rtol=1e-6, atol=1e-6
    )


def test_fixed_mixture_and_joint_parameters():
    frame = _frame()
    fixed = fit(
        "y ~ x",
        data=frame,
        fixed_parameters={"sigma": 0.7},
        method="svi",
        num_steps=2,
        seed=1,
    )
    values = fixed.posterior_linpred(parameter="sigma", num_samples=3)
    assert values.shape == (1, 3, 24)
    torch.testing.assert_close(values.double(), torch.full((1, 3, 24), 0.7).double())
    mixed = fit(
        "y ~ x",
        data=frame,
        family=mixture("gaussian", "gaussian"),
        method="svi",
        num_steps=2,
        seed=1,
    )
    scale = mixed.posterior_linpred(parameter="sigma2", num_samples=3, seed=4)
    assert torch.equal(
        scale, mixed.draws(num_samples=3, seed=4)["sigma2"][..., None].expand(1, 3, 24)
    )
    joint = fit(
        cast(MultivariateFormula, bf("y ~ x") + bf("y2 ~ x") + set_rescor(False)),
        data=frame,
        method="svi",
        num_steps=2,
        seed=1,
    )
    values = joint.posterior_linpred(
        resp="y2", parameter="sigma", num_samples=3, seed=4
    )
    assert set(values) == {"y2"} and values["y2"].shape == (1, 3, 24)
    assert bool((values["y2"] == values["y2"][..., :1]).all())
    assert bool((values["y2"] > 0).all())


def test_a_predicted_parameter_keeps_its_predictor_and_unknown_names_are_refused():
    frame = _frame()
    fitted = fit(
        "y ~ x",
        data=frame,
        distributional={"sigma": "~ z"},
        method="svi",
        num_steps=3,
        seed=1,
    )
    linear = fitted.posterior_linpred(parameter="sigma", num_samples=3, seed=2)
    natural = fitted.posterior_linpred(
        parameter="sigma", transform=True, num_samples=3, seed=2
    )
    torch.testing.assert_close(linear.exp(), natural)
    assert linear.unique(dim=-1).shape[-1] > 1
    plain = fit("y ~ x", data=frame, method="svi", num_steps=2, seed=1)
    with pytest.raises(ValueError, match="'nu' has no fitted distributional predictor"):
        plain.posterior_linpred(parameter="nu")
    with pytest.raises(ValueError, match="has no fitted distributional predictor"):
        plain.fitted(parameter="shape")
