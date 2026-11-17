"""mgcv's ``sp`` argument of ``s()`` and ``t2()``, accepted as brms accepts it.

brms estimates the smoothing standard deviations, so ``sp`` changes neither
its Stan program nor its data (``oracles/brms_smooth_sp.R``). The term is
thus the same model with a different label.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import fit, formula_to_qvr
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_smooth_sp.json").read_text()
)


def _frame() -> pd.DataFrame:
    rows = np.arange(30)
    x = (rows * 0.618034) % 1
    z = np.cos(rows)
    return pd.DataFrame({"x": x, "z": z, "y": np.sin(3 * x) + 0.1 * z})


def test_brms_and_mgcv_ignore_sp():
    assert all(
        ORACLE[key] is True
        for key in ("s_code", "s_data", "t2_code", "t2_data", "mgcv_design")
    )
    assert ORACLE["mgcv_random"] is True
    assert ORACLE["s_label"] == "s(x, sp = 0.5)"


@pytest.mark.parametrize(
    ("with_sp", "plain", "label", "bare"),
    [
        ("y ~ s(x, sp = 0.5)", "y ~ s(x)", "s_x_sp_0_5", "s_x"),
        (
            "y ~ t2(x, z, sp = c(1, 2, 3))",
            "y ~ t2(x, z)",
            "t2_x_z_sp_c_1_2_3",
            "t2_x_z",
        ),
    ],
)
def test_sp_changes_only_the_term_label(with_sp, plain, label, bare):
    frame = _frame()
    source = formula_to_qvr(with_sp, data=frame)
    assert label in source
    assert source.replace(label, bare) == formula_to_qvr(plain, data=frame)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_sp_terms_fit_and_predict_as_the_plain_term(method, tmp_path):
    frame = _frame()
    settings = (
        {"method": "svi", "num_steps": 5}
        if method == "svi"
        else {"method": method, "num_warmup": 3, "num_samples": 3, "num_chains": 1}
    )
    fitted = fit("y ~ s(x, sp = 0.5)", data=frame, seed=1, **settings)
    plain = fit("y ~ s(x)", data=frame, seed=1, **settings)
    fresh = frame.iloc[:6].assign(x=lambda rows: rows.x * 0.9)
    expected = fitted.posterior_epred(new_data=fresh, num_samples=3, seed=2)
    torch.testing.assert_close(
        expected, plain.posterior_epred(new_data=fresh, num_samples=3, seed=2)
    )
    assert fitted.log_likelihood(num_samples=2)["y"].shape == (1, 2, len(frame))
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert set(refitted.draws(num_samples=1)) == set(fitted.draws(num_samples=1))
    path = tmp_path / "sp.pt"
    save_fit(fitted, path)
    torch.testing.assert_close(
        load_fit(path).posterior_epred(new_data=fresh, num_samples=3, seed=2),
        expected,
    )


def test_sp_must_be_numeric():
    frame = _frame()
    for term in ('s(x, sp = "a")', "s(x, sp = TRUE)", 't2(x, z, sp = c(1, "b"))'):
        with pytest.raises(ValueError, match="sp must hold finite numbers"):
            formula_to_qvr(f"y ~ {term}", data=frame)
    with pytest.raises(ValueError, match="reads no xt entry"):
        formula_to_qvr("y ~ s(x, xt = 1)", data=frame)
