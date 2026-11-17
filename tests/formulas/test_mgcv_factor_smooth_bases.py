"""Factor-smooth interactions over other bases, with ``by``, against mgcv.

Oracle: ``oracles/mgcv_factor_smooth_bases.R`` records brms's Stan data for
``s(x, g, bs = "fs", xt = ...)`` over the ``cr``, ``ps``, ``cc``, ``bs``,
``gp`` and ``tp`` bases (``smoothCon``, ``smooth2random``, ``s2rPred``;
checked there against ``make_standata``), what mgcv and brms do with a
``by`` variable, and brms's failures for ``fx`` and the shrinkage bases.

The designs are compared by the invariants of
``test_mgcv_factor_smooths.py``: block sizes, signed range columns at one
common positive scale, one rotation of the null columns shared by the
levels, and the same map on fresh rows. Tolerances were fixed before
comparing: 1e-8 for designs and 1e-6 for the Gaussian process base, whose
eigenbasis mgcv truncates with a Lanczos iteration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import BayesianFit, conditional_smooths, fit, formula_from_data
from quivers.inference.mcmc.driver import MCMCResult
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_factor_smooth_bases.json").read_text()
)
TOL = 1e-8
GP_TOL = 1e-6
TRAIN = pd.DataFrame(ORACLE["train"])
FRESH = pd.DataFrame(ORACLE["fresh"]).assign(y=[0.1, 0.4, 0.2, 0.9, 1.3, 0.7])


def _spec(entry) -> str:
    return entry["spec"].replace("'", '"')


def _blocks(spec: str, frame: pd.DataFrame | None = None) -> list[np.ndarray]:
    formula = formula_from_data(f"y ~ {spec}", TRAIN)
    blocks = [term.data for term in formula.design_terms]
    if frame is None:
        return blocks
    fitted = fit(f"y ~ {spec}", data=TRAIN, method="svi", num_steps=0)
    plan = fitted.prediction_plan
    assert plan is not None
    term = next(
        value
        for name, value in plan.designs["mu"].common.terms.items()
        if name.startswith("s(")
    )
    values = np.asarray(term.eval_new_data(frame), dtype=np.float64)
    widths = np.cumsum([block.shape[1] for block in blocks])[:-1]
    return np.split(values, widths, axis=1)


def _level_major(blocks: list[np.ndarray], levels: int) -> np.ndarray:
    """Stack the null-space blocks as ``(level, row, null column)``."""
    return np.stack(
        [np.column_stack(blocks)[:, level::levels] for level in range(levels)]
    )


@pytest.mark.parametrize("entry", ORACLE["entries"], ids=lambda entry: entry["spec"])
def test_design_matches_mgcv_up_to_its_free_rotation(entry):
    tolerance = GP_TOL if "gp" in entry["spec"] else TOL
    levels, rank, null = len(entry["levels"]), entry["rank"], entry["null_dim"]
    mine = _blocks(_spec(entry))
    theirs = [np.asarray(block) for block in entry["random"]]
    assert [block.shape for block in mine] == [block.shape for block in theirs]
    assert [block.shape[1] for block in mine] == [rank * levels, *([levels] * null)]
    assert entry["pen_ind"] == ([1] * rank + list(range(2, null + 2))) * levels
    # Range columns: mgcv's up to a sign each and one common positive scale.
    ratio = (mine[0] * theirs[0]).sum(0) / (mine[0] ** 2).sum(0)
    scale = float(np.abs(ratio[0]))
    np.testing.assert_allclose(np.abs(ratio), scale, rtol=tolerance, atol=0)
    np.testing.assert_allclose(mine[0] * ratio, theirs[0], rtol=0, atol=tolerance)
    np.testing.assert_array_equal(
        np.sign(ratio).reshape(levels, rank),
        np.broadcast_to(np.sign(ratio[:rank]), (levels, rank)),
    )
    # Null columns: one rotation, the same for every level, at that scale.
    own, reference = _level_major(mine[1:], levels), _level_major(theirs[1:], levels)
    rotation = np.linalg.lstsq(own[0], reference[0], rcond=None)[0]
    np.testing.assert_allclose(
        rotation @ rotation.T, scale**2 * np.eye(null), rtol=0, atol=tolerance
    )
    for level in range(levels):
        np.testing.assert_allclose(
            own[level] @ rotation, reference[level], rtol=0, atol=tolerance
        )
    if null == 1:
        # Nothing is free: the design is mgcv's up to column signs.
        np.testing.assert_allclose(scale, 1.0, rtol=tolerance, atol=0)
    # Fresh rows follow through the same map.
    fresh = _blocks(_spec(entry), FRESH)
    expected = [np.asarray(block) for block in entry["fresh_random"]]
    np.testing.assert_allclose(fresh[0] * ratio, expected[0], rtol=0, atol=tolerance)
    own, reference = _level_major(fresh[1:], levels), _level_major(expected[1:], levels)
    for level in range(levels):
        np.testing.assert_allclose(
            own[level] @ rotation, reference[level], rtol=0, atol=tolerance
        )


def test_string_and_list_xt_name_the_same_base():
    listed = _blocks('s(x, g, bs = "fs", k = 5, xt = list(bs = "cr"))')
    named = _blocks('s(x, g, bs = "fs", k = 5, xt = "cr")')
    default = _blocks('s(x, g, bs = "fs", k = 5)')
    thin = _blocks('s(x, g, bs = "fs", k = 5, xt = list(bs = "tp"))')
    for a, b in zip(listed, named, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(default, thin, strict=True):
        np.testing.assert_array_equal(a, b)
    # A different base is a different design.
    assert np.max(np.abs(listed[0] - default[0])) > 1e-3


def test_null_columns_are_orthonormal_polynomials_of_every_base():
    for base in ("cr", "ps", "bs", "gp"):
        blocks = _blocks(f's(x, g, bs = "fs", k = 6, xt = list(bs = "{base}"))')
        rows = (TRAIN["g"] == "a").to_numpy()
        constant, linear = blocks[1][rows, 0], blocks[2][rows, 0]
        np.testing.assert_allclose(constant, constant[0], rtol=0, atol=GP_TOL)
        slope = np.polyfit(TRAIN["x"].to_numpy()[rows], linear, 1)
        np.testing.assert_allclose(
            np.polyval(slope, TRAIN["x"].to_numpy()[rows]), linear, rtol=0, atol=GP_TOL
        )
        assert constant[0] > 0 and slope[0] > 0


def test_by_variable_multiplies_the_design_as_mgcv_gam_does():
    recorded = ORACLE["by"]
    # mgcv's own design is masked or multiplied by the by variable, and
    # brms's Stan data are not.
    assert recorded["gam_factor_masks"] == 0 and recorded["gam_numeric_multiplies"] == 0
    assert recorded["brms_factor_ignores"] == 0
    assert recorded["brms_numeric_ignores"] == 0
    plain = _blocks('s(x, g, bs = "fs", k = 5)')
    factor = formula_from_data('y ~ s(x, g, bs = "fs", k = 5, by = h)', TRAIN)
    assert len(factor.design_terms) == 2 * len(plain)
    for position, level in enumerate(("u", "v")):
        mask = (TRAIN["h"] == level).to_numpy().astype(float)[:, None]
        for block, reference in zip(
            factor.design_terms[position * len(plain) : (position + 1) * len(plain)],
            plain,
            strict=True,
        ):
            np.testing.assert_allclose(block.data, mask * reference, rtol=0, atol=1e-12)
    numeric = _blocks('s(x, g, bs = "fs", k = 5, by = w)')
    for block, reference in zip(numeric, plain, strict=True):
        np.testing.assert_allclose(
            block, TRAIN["w"].to_numpy()[:, None] * reference, rtol=0, atol=1e-12
        )
    # The unmasked design brms builds is a different design.
    assert np.max(np.abs(factor.design_terms[0].data - plain[0])) > 1e-3
    # Fresh rows are masked by their own by level.
    fresh = _blocks('s(x, g, bs = "fs", k = 5, by = h)', FRESH)
    unmasked = _blocks('s(x, g, bs = "fs", k = 5)', FRESH)
    mask = (FRESH["h"] == "u").to_numpy().astype(float)[:, None]
    np.testing.assert_allclose(fresh[0], mask * unmasked[0], rtol=0, atol=1e-12)


def _fit(formula: str, method: str):
    options = (
        {"num_steps": 5}
        if method == "svi"
        else {"num_warmup": 6, "num_samples": 4, "num_chains": 1}
    )
    return fit(formula, data=TRAIN, method=method, seed=3, **options)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_factor_smooth_bases_fit_predict_refit_and_restore(method, tmp_path):
    formula = 'y ~ s(x, g, bs = "fs", k = 5, xt = list(bs = "cr"), by = h)'
    fitted = _fit(formula, method)
    if method != "svi":
        assert isinstance(fitted.posterior, MCMCResult)
    names = fitted.variables(num_samples=3)
    assert {"sds_sxghu_1", "sds_sxghu_2", "sds_sxghv_3", "s_sxghv_1[1]"} <= set(names)
    linpred = fitted.posterior_linpred(num_samples=3, seed=1)
    assert linpred.shape[-1] == len(TRAIN) and torch.isfinite(linpred).all()
    fresh = fitted.posterior_linpred(new_data=FRESH, num_samples=3, seed=1)
    assert fresh.shape[-1] == len(FRESH) and torch.isfinite(fresh).all()
    for method_name in ("posterior_epred", "posterior_predict"):
        value = getattr(fitted, method_name)(new_data=FRESH, num_samples=3, seed=1)
        assert value.shape[-1] == len(FRESH) and torch.isfinite(value).all()
    scores = fitted.log_likelihood(new_data=FRESH, num_samples=3, seed=1)["y"]
    assert torch.isfinite(scores).all()
    # Training rows as new rows reproduce the training predictor.
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=TRAIN, num_samples=3, seed=1),
        linpred,
        rtol=1e-6,
        atol=1e-6,
    )
    # The smooth's contribution is the linear predictor without the intercept.
    term = next(iter({block.term for block in fitted.formula.design_terms}))
    contribution = conditional_smooths(fitted, term, FRESH, num_samples=3, seed=1)
    intercept = fitted.draws(num_samples=3, seed=1)["intercept"]
    torch.testing.assert_close(
        contribution + intercept[..., None],
        fresh.to(contribution.dtype),
        atol=1e-5,
        rtol=0,
    )
    path = tmp_path / "fs.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, BayesianFit)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=FRESH, num_samples=3, seed=1), fresh
    )
    if method == "svi":
        assert isinstance(fitted, BayesianFit)
        updated = fitted.update(num_steps=3)
        assert set(updated.draws()) == set(fitted.draws())
        assert np.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)
        assert fitted.conditional_smooths().effects


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ('y ~ s(x, g, bs = "fs", fx = TRUE)', "brms fails on it"),
        ('y ~ s(x, g, bs = "fs", xt = list(bs = "cs"))', "penalty has full rank"),
        ('y ~ s(x, g, bs = "fs", xt = "ts")', "penalty has full rank"),
        ('y ~ s(x, g, bs = "fs", xt = list(bs = "re"))', "unsupported xt base"),
        ('y ~ s(x, g, bs = "fs", xt = list(bs = "fs"))', "unsupported xt base"),
        ('y ~ s(x, g, bs = "fs", xt = list(max.knots = 5))', "reads only xt"),
        ('y ~ s(x, g, bs = "fs", xt = list("cr"))', "takes named entries"),
        ('y ~ s(x, g, bs = "fs", xt = list(bs = 3))', "must be a basis name"),
        ('y ~ s(x, z, g, bs = "fs", xt = "cr")', "one-dimensional basis"),
        ('y ~ s(x, g, bs = "fs", m = c(2, 0))', "one penalty order"),
        ('y ~ s(x, g, bs = "fs", xt = "cr", m = 2)', "fixed penalty order"),
        ('y ~ s(x, bs = "cr", xt = list(bs = "cr"))', "reads no xt entry"),
    ],
)
def test_invalid_factor_smooth_bases_are_rejected(formula, message):
    with pytest.raises(ValueError, match=message):
        formula_from_data(formula, TRAIN)


def test_brms_fails_where_these_are_refused():
    assert "subscript out of bounds" in ORACLE["errors"]["fx"]
    assert ORACLE["errors"]["cs"] == "subscript out of bounds"
    assert ORACLE["errors"]["ts"] == "subscript out of bounds"
