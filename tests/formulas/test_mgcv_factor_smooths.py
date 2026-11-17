"""Factor-smooth interactions ``s(x, g, bs = "fs")`` against mgcv and brms.

Oracle: ``oracles/mgcv_factor_smooths.R`` records brms's Stan data
(``smoothCon``, ``smooth2random``, ``s2rPred``; checked there against
``make_standata``), brms's prior rows and refusal of new factor levels, and
``posterior_linpred`` at brms's own draws.

mgcv takes the null-space columns of the base smooth from an
eigendecomposition in which they share a zero eigenvalue, so they are fixed
only up to an orthogonal rotation, and its penalty scaling reads the row sums
of those columns. The designs are thus compared by invariants: block sizes,
one linear map carrying this design onto mgcv's (signed and scaled range
columns, one rotation of the null columns repeated for every level), and the
same map on fresh rows. With a one-dimensional null space (``m = 1``) nothing
is free but column signs, and the columns agree directly.

Tolerances were fixed before comparing: 1e-8 for designs (the bases are
eigenvectors of matrices with condition number below 1e6) and 1e-5 for
values computed from brms's six-digit draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import conditional_smooths, fit, formula_from_data, get_prior
from quivers.inference.mcmc.driver import MCMCResult
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_factor_smooths.json").read_text()
)
TOL = 1e-8
BRMS_TOL = 1e-5
TRAIN = pd.DataFrame(ORACLE["train"])
FRESH = pd.DataFrame(ORACLE["fresh"])
SEEN = FRESH.iloc[:6].assign(y=[0.1, 0.4, 0.2, 0.9, 1.3, 0.7])


def _spec(entry) -> str:
    return entry["spec"].replace("'", '"')


def _blocks(entry, frame: pd.DataFrame | None = None) -> list[np.ndarray]:
    formula = formula_from_data(f"y ~ {_spec(entry)}", TRAIN)
    if frame is None:
        return [term.data for term in formula.design_terms]
    fitted = fit(f"y ~ {_spec(entry)}", data=TRAIN, method="svi", num_steps=0)
    plan = fitted.prediction_plan
    assert plan is not None
    term = next(
        value
        for name, value in plan.designs["mu"].common.terms.items()
        if name.startswith("s(")
    )
    values = np.asarray(term.eval_new_data(frame), dtype=np.float64)
    widths = np.cumsum([block.shape[1] for block in _blocks(entry)])[:-1]
    return np.split(values, widths, axis=1)


def _level_major(blocks: list[np.ndarray], levels: int) -> np.ndarray:
    """Stack the null-space blocks as ``(level, row, null column)``."""
    return np.stack(
        [np.column_stack(blocks)[:, level::levels] for level in range(levels)]
    )


@pytest.mark.parametrize("entry", ORACLE["entries"], ids=lambda entry: entry["spec"])
def test_design_matches_mgcv_up_to_its_free_rotation(entry):
    levels, rank, null = len(entry["levels"]), entry["rank"], entry["null_dim"]
    mine = _blocks(entry)
    theirs = [np.asarray(block) for block in entry["random"]]
    # smooth2random block structure: the range space, then each null column.
    assert [block.shape for block in mine] == [block.shape for block in theirs]
    assert [block.shape[1] for block in mine] == [rank * levels, *([levels] * null)]
    assert not [
        column
        for column in formula_from_data(f"y ~ {_spec(entry)}", TRAIN).fixed_columns
        if not column.is_intercept
    ]
    assert entry["pen_ind"] == ([1] * rank + list(range(2, null + 2))) * levels
    # Range columns: mgcv's up to a sign each and one common positive scale.
    ratio = (mine[0] * theirs[0]).sum(0) / (mine[0] ** 2).sum(0)
    scale = float(np.abs(ratio[0]))
    np.testing.assert_allclose(np.abs(ratio), scale, rtol=TOL, atol=0)
    np.testing.assert_allclose(mine[0] * ratio, theirs[0], rtol=0, atol=TOL)
    # The same sign pattern for every level.
    np.testing.assert_array_equal(
        np.sign(ratio).reshape(levels, rank),
        np.broadcast_to(np.sign(ratio[:rank]), (levels, rank)),
    )
    # Null columns: one rotation, the same for every level, at that scale.
    own, reference = _level_major(mine[1:], levels), _level_major(theirs[1:], levels)
    rotation = np.linalg.lstsq(own[0], reference[0], rcond=None)[0]
    np.testing.assert_allclose(
        rotation @ rotation.T, scale**2 * np.eye(null), rtol=0, atol=TOL
    )
    for level in range(levels):
        np.testing.assert_allclose(
            own[level] @ rotation, reference[level], rtol=0, atol=TOL
        )
    if null == 1:
        # Nothing is free: the design is mgcv's up to column signs.
        np.testing.assert_allclose(scale, 1.0, rtol=TOL, atol=0)
    # Fresh rows of fitted levels follow through the same map.
    fresh = _blocks(entry, SEEN)
    expected = [np.asarray(block)[:6] for block in entry["fresh_random"]]
    np.testing.assert_allclose(fresh[0] * ratio, expected[0], rtol=0, atol=TOL)
    own, reference = _level_major(fresh[1:], levels), _level_major(expected[1:], levels)
    for level in range(levels):
        np.testing.assert_allclose(
            own[level] @ rotation, reference[level], rtol=0, atol=TOL
        )


def test_null_columns_are_orthonormal_polynomials_and_stable():
    entry = ORACLE["entries"][0]
    blocks = _blocks(entry)
    rows = (TRAIN["g"] == entry["levels"][0]).to_numpy()
    constant, linear = blocks[1][rows, 0], blocks[2][rows, 0]
    # Constant first, then the linear polynomial orthogonal to it over all rows.
    np.testing.assert_allclose(constant, constant[0], rtol=0, atol=TOL)
    slope = np.polyfit(TRAIN["x"].to_numpy()[rows], linear, 1)
    np.testing.assert_allclose(
        np.polyval(slope, TRAIN["x"].to_numpy()[rows]), linear, rtol=0, atol=TOL
    )
    # mgcv's two null columns move by order one under this perturbation
    # (measured with oracles/mgcv_factor_smooths.R's data); these do not.
    moved = TRAIN.copy()
    moved.loc[6, "x"] *= 1 + 1e-13
    other = formula_from_data(f"y ~ {_spec(entry)}", moved).design_terms
    for block, term in zip(blocks, other, strict=True):
        np.testing.assert_allclose(term.data, block, rtol=0, atol=1e-9)


def test_new_factor_levels_are_refused_as_in_brms():
    assert ORACLE["fit"]["new_level_error"].startswith(
        "New factor levels are not allowed."
    )
    assert ORACLE["fit"]["new_level_allowed_error"].startswith(
        "New factor levels are not allowed."
    )
    fitted = fit(
        'y ~ s(x, g, bs = "fs", k = 5, m = 1)', data=TRAIN, method="svi", num_steps=0
    )
    for options in ({}, {"allow_new_levels": True}):
        with pytest.raises(ValueError, match="new factor levels are not allowed"):
            fitted.posterior_linpred(new_data=FRESH, num_samples=1, **options)


def _pin(fitted, values):
    samples = {
        name: torch.as_tensor(np.asarray(value), dtype=torch.float64).unsqueeze(0)
        for name, value in values.items()
    }
    count = next(iter(samples.values())).shape[1]
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, count),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def test_linear_predictor_matches_brms_at_brms_draws():
    case = ORACLE["fit"]
    formula = 'y ~ s(x, g, bs = "fs", k = 5, m = 1)'
    fitted = fit(formula, data=TRAIN, method="svi", num_steps=0)
    assert case["variables"][:4] == ["b_Intercept", "sds_sxg_1", "sds_sxg_2", "sigma"]
    blocks = fitted.formula.design_terms
    assert len(blocks) == 2
    assert {name for name in get_prior(formula, data=TRAIN) if "_scale" in name} == {
        block.qvr_name + "_scale" for block in blocks
    }
    draws = case["draws"]
    values = {
        "intercept": np.asarray(draws["b_Intercept"]),
        "sigma": np.asarray(draws["sigma"]),
    }
    for index, block in enumerate(blocks, start=1):
        theirs = np.asarray(case["design"][index - 1])
        # A one-dimensional null space leaves only column signs free.
        sign = np.sign((block.data * theirs).sum(0))
        np.testing.assert_allclose(block.data * sign, theirs, rtol=0, atol=TOL)
        coefficients = np.stack(
            [
                np.asarray(draws[f"s_sxg_{index}[{j}]"])
                for j in range(1, block.data.shape[1] + 1)
            ],
            -1,
        )
        sds = np.asarray(draws[f"sds_sxg_{index}"])
        values[block.qvr_name + "_scale"] = sds
        values[block.qvr_name + "_z"] = coefficients * sign / sds[:, None]
    pinned = _pin(fitted, values)
    torch.testing.assert_close(
        pinned.posterior_linpred()[0].double(),
        torch.tensor(case["linpred"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )
    torch.testing.assert_close(
        pinned.posterior_linpred(new_data=SEEN)[0].double(),
        torch.tensor(case["seen_linpred"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_factor_smooths_fit_predict_refit_and_restore(method, tmp_path):
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 15, "num_samples": 8, "num_chains": 1}
    )
    formula = 'y ~ s(x, g, bs = "fs", k = 5) + z'
    fitted = fit(formula, data=TRAIN, method=method, seed=2, **options)
    linpred = fitted.posterior_linpred(new_data=SEEN, num_samples=3)
    assert linpred.shape[-1] == 6 and bool(torch.isfinite(linpred).all())
    assert fitted.posterior_predict(new_data=SEEN, num_samples=3).shape[-1] == 6
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=SEEN, num_samples=3)["y"]).all()
    )
    # Training rows as new rows reproduce the training predictor.
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=TRAIN, num_samples=3, seed=1),
        fitted.posterior_linpred(num_samples=3, seed=1),
        rtol=1e-6,
        atol=1e-6,
    )
    term = next(iter(t.term for t in fitted.formula.design_terms))
    smooth = conditional_smooths(fitted, term, SEEN, num_samples=3, seed=1)
    blocks = [t for t in fitted.formula.design_terms if t.term == term]
    assert len(blocks) == 3
    draws = fitted.draws(num_samples=3, seed=1)
    design = np.column_stack(_blocks({"spec": 's(x, g, bs = "fs", k = 5)'}, SEEN))
    coefficients = torch.cat(
        [
            draws[b.qvr_name + "_scale"].double()[..., None]
            * draws[b.qvr_name + "_z"].double()
            for b in blocks
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        smooth.double(), coefficients @ torch.as_tensor(design).T, rtol=1e-9, atol=1e-9
    )
    path = tmp_path / "fs.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=SEEN, num_samples=3, seed=1),
        fitted.posterior_linpred(new_data=SEEN, num_samples=3, seed=1),
    )
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert set(updated.draws()) == set(fitted.draws())
        assert np.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)
        effects = fitted.conditional_smooths()
        assert effects.effects[0].effects == ("x", "g")


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ('y ~ s(x, bs = "fs")', "exactly one factor"),
        ('y ~ s(g, bs = "fs")', "exactly one factor"),
        ('y ~ s(x, g, h, bs = "fs")', "exactly one factor"),
        ('y ~ s(x, g, bs = "fs", fx = TRUE)', "brms fails on it"),
        ('y ~ s(x, g, bs = "fs", m = c(2, 0))', "one penalty order"),
        ('y ~ t2(x, z, bs = "fs")', "margins must use bases"),
    ],
)
def test_invalid_factor_smooths_are_rejected(formula, message):
    with pytest.raises(ValueError, match=message):
        formula_from_data(formula, TRAIN.assign(h=TRAIN["g"]))
