"""Missing values in the variables of special terms remove the same rows.

Each case puts a missing value in a variable only a special term uses (a
smooth or Gaussian-process covariate, the ``time`` or ``gr`` column of an
autocorrelation term, a monotonic predictor, a multi-membership column or
weight) and checks that the fit equals, draw for draw, the fit on the frame
filtered by hand, as brms's ``na_omit`` removes those rows before anything
is built.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import fit


def _frame(rows: int = 30) -> pd.DataFrame:
    generator = np.random.default_rng(8)
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "x": (index * 0.618034) % 1,
            "v": (index * 0.414214) % 1,
            "k": index % 4,
            "t": index // 3,
            "s": np.array(["a", "b", "c"])[index % 3],
            "g1": np.array(["a", "b", "c", "d"])[index % 4],
            "g2": np.array(["b", "c", "d", "e"])[(index // 4) % 4],
            "w1": 1.0 + index % 3,
            "w2": 1.0 + index % 2,
        }
    )
    frame["y"] = np.sin(3 * frame["x"]) + generator.normal(scale=0.3, size=rows)
    return frame


_CASES = [
    ("smooth covariate", "x", "y ~ s(x, k = 5)"),
    ("gaussian process input", "v", "y ~ gp(v, k = 6)"),
    ("autocorrelation time", "t", "y ~ x + ar(time = t, gr = s)"),
    ("autocorrelation group", "s", "y ~ x + ar(time = t, gr = s)"),
    ("monotonic predictor", "k", "y ~ mo(k)"),
    ("membership column", "g2", "y ~ x + (1 | mm(g1, g2, weights = cbind(w1, w2)))"),
    ("membership weight", "w1", "y ~ x + (1 | mm(g1, g2, weights = cbind(w1, w2)))"),
]


@pytest.mark.parametrize(
    ("kind", "column", "formula"), _CASES, ids=[case[0] for case in _CASES]
)
def test_missing_special_term_variable_matches_the_filtered_fit(kind, column, formula):
    frame = _frame()
    frame[column] = frame[column].astype(object if column in {"s", "g2"} else float)
    frame.loc[[4, 17], column] = None
    kept = [i for i in range(30) if i not in (4, 17)]
    with pytest.warns(UserWarning, match="were excluded from the model"):
        fitted = fit(formula, data=frame, method="svi", num_steps=3, seed=6)
    clean = frame.iloc[kept].reset_index(drop=True).infer_objects()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        reference = fit(formula, data=clean, method="svi", num_steps=3, seed=6)
    assert fitted.included_rows == tuple(kept)
    torch.testing.assert_close(
        fitted.posterior_linpred(num_samples=3, seed=2),
        reference.posterior_linpred(num_samples=3, seed=2),
        rtol=0,
        atol=0,
    )
    expected = reference.draws(num_samples=2, seed=1)
    actual = fitted.draws(num_samples=2, seed=1)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
