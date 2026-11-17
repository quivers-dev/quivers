"""Markov random field smooths ``s(region, bs = "mrf")`` against mgcv and brms.

Oracle: ``oracles/mgcv_mrf_smooths.R`` records brms's Stan data
(``make_standata``, checked there against ``smoothCon`` and
``smooth2random``, with ``s2rPred`` on fresh rows) for penalties and
neighbour lists over three graphs, low-rank bases, ``by`` variables, ``fx``,
and a region without observations, together with brms's variable names,
``posterior_linpred`` at brms's own draws, and its error messages.

brms reads ``xt`` objects from R's global environment. Here they are
``data2`` entries named in ``xt``: a penalty as a data frame labeled by region
or an unlabeled array, and a neighbour list as a labeled matrix whose row
``i`` is nonzero at the neighbours listed for region ``i``.

mgcv's coordinates come from eigendecompositions, so repeated eigenvalues
leave rotations free. The designs are compared by what the model depends on:
the span of the unpenalized columns, and ``Z Z'`` for a penalized block
``Z``, which is the covariance of its contribution under independent
standard-Normal coefficients. Fresh rows are compared through the same
quantities across fresh and training rows. Tolerances were fixed before
comparing: 1e-8 for designs (six-by-six eigenproblems with condition number
below 1e4) and 1e-5 for values computed from brms's six-digit draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import (
    BayesianFit,
    bf,
    conditional_smooths,
    fit,
    get_prior,
    set_rescor,
)
from quivers.formulas._smooth import smooth_states
from quivers.inference.mcmc.driver import MCMCResult
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_mrf_smooths.json").read_text()
)
TOL = 1e-8
BRMS_TOL = 1e-5
REGIONS = ORACLE["regions"]
TRAIN = pd.DataFrame(ORACLE["train"])
FRESH = pd.DataFrame(ORACLE["fresh"]).assign(y=0.0)
PARTIAL = TRAIN[TRAIN["region"] != "r6"].assign(
    region=lambda frame: pd.Categorical(frame["region"], categories=REGIONS)
)


def _labeled(name: str, labels: list[str] | None = None) -> pd.DataFrame:
    labels = REGIONS if labels is None else labels
    return pd.DataFrame(
        np.asarray(ORACLE["structures"][name], dtype=float),
        index=labels,
        columns=labels,
    )


DATA2 = {
    "P_irregular": _labeled("P_irregular"),
    "P_ring": _labeled("P_ring"),
    "P_split": _labeled("P_split"),
    "P_dense": _labeled("P_dense"),
    "P_shuffled": _labeled("P_shuffled", ORACLE["structures"]["shuffled"]),
    "P_unnamed": np.asarray(ORACLE["structures"]["P_irregular"], dtype=float),
    "nb_irregular": _labeled("nb_irregular"),
    "nb_split": _labeled("nb_split"),
    "nb_self": _labeled("nb_self"),
}
PENALIZED = 'y ~ s(region, bs = "mrf", xt = list(penalty = P_irregular))'


def _spec(entry) -> str:
    return entry["spec"].replace("'", '"')


def _matrix(rows, count: int) -> np.ndarray:
    return np.asarray(rows, dtype=float).reshape(count, -1)


def _blocks(spec: str, train: pd.DataFrame, **options):
    """Return each block's training and fresh unpenalized and penalized columns."""
    fitted = fit(
        f"y ~ {spec}", data=train, data2=DATA2, method="svi", num_steps=0, **options
    )
    plan = fitted.prediction_plan
    assert plan is not None
    (label, state), *rest = smooth_states(plan.designs["mu"]).items()
    assert not rest
    term = plan.designs["mu"].common.terms[label]
    both = [
        np.asarray(term.eval_new_data(frame), dtype=np.float64).reshape(len(frame), -1)
        for frame in (train, FRESH)
    ]
    return [
        tuple(
            values[:, list(columns)]
            for columns in (layout.fixed, sum(layout.random, ()))
            for values in both
        )
        for layout in state.layout
    ]


def _same_span(mine, theirs, mine_fresh, theirs_fresh) -> None:
    assert mine.shape == theirs.shape
    if mine.shape[1] == 0:
        return
    mapping = np.linalg.lstsq(mine, theirs, rcond=None)[0]
    np.testing.assert_allclose(mine @ mapping, theirs, rtol=0, atol=TOL)
    assert np.linalg.matrix_rank(mapping) == mine.shape[1]
    np.testing.assert_allclose(mine_fresh @ mapping, theirs_fresh, rtol=0, atol=TOL)


def _same_covariance(mine, theirs, mine_fresh, theirs_fresh) -> None:
    assert mine.shape == theirs.shape
    np.testing.assert_allclose(mine @ mine.T, theirs @ theirs.T, rtol=0, atol=TOL)
    np.testing.assert_allclose(
        mine_fresh @ mine.T, theirs_fresh @ theirs.T, rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        mine_fresh @ mine_fresh.T, theirs_fresh @ theirs_fresh.T, rtol=0, atol=TOL
    )


def _check(entry, train: pd.DataFrame, **options) -> None:
    mine = _blocks(_spec(entry), train, **options)
    assert len(mine) == len(entry["blocks"])
    for (fixed, fixed_fresh, random, random_fresh), reference in zip(
        mine, entry["blocks"], strict=True
    ):
        _same_span(
            fixed,
            _matrix(reference["fixed"], len(train)),
            fixed_fresh,
            _matrix(reference["fresh_fixed"], len(FRESH)),
        )
        _same_covariance(
            random,
            _matrix(reference["random"], len(train)),
            random_fresh,
            _matrix(reference["fresh_random"], len(FRESH)),
        )


@pytest.mark.parametrize("entry", ORACLE["entries"], ids=lambda entry: entry["spec"])
def test_design_matches_brms(entry):
    _check(entry, TRAIN)


@pytest.mark.parametrize("entry", ORACLE["unobserved"], ids=lambda entry: entry["spec"])
def test_region_without_observations_is_kept_and_predicted(entry):
    # brms fails when the unused level is dropped (its default) and builds
    # the design below with drop_unused_levels = FALSE.
    assert ORACLE["errors"]["dropped_levels"] == "supplied penalty wrong dimension!"
    with pytest.raises(ValueError, match="do not match the regions of the data"):
        _blocks(_spec(entry), PARTIAL)
    _check(entry, PARTIAL, drop_unused_levels=False)


def test_equivalent_structures_give_one_design():
    reference = _blocks(_spec(ORACLE["entries"][0]), TRAIN)
    for index in (1, 2, 3):
        other = _blocks(_spec(ORACLE["entries"][index]), TRAIN)
        for a, b in zip(reference[0], other[0], strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    # A self-listed neighbour and another graph are different models.
    for index in (5, 8):
        other = _blocks(_spec(ORACLE["entries"][index]), TRAIN)
        assert (
            np.max(
                np.abs(
                    reference[0][2] @ reference[0][2].T - other[0][2] @ other[0][2].T
                )
            )
            > 1e-3
        )


def test_full_rank_prior_is_the_gaussian_markov_random_field():
    # Sum-to-zero region effects b with density proportional to
    # exp(-b' P b / (2 sds^2)): their covariance is the pseudo-inverse of P
    # up to mgcv's penalty rescaling, a positive constant.
    (_, _, random, _), *_ = _blocks(_spec(ORACLE["entries"][0]), TRAIN)
    rows = [int(np.flatnonzero(TRAIN["region"] == region)[0]) for region in REGIONS]
    covariance = random[rows] @ random[rows].T
    penalty = DATA2["P_irregular"].to_numpy()
    counts = TRAIN["region"].value_counts().reindex(REGIONS).to_numpy()
    # mgcv's constraint is zero weighted sum over rows; project accordingly.
    contrast = np.eye(6) - np.outer(np.ones(6), counts) / counts.sum()
    reference = contrast @ np.linalg.pinv(penalty) @ contrast.T
    ratio = covariance[0, 0] / reference[0, 0]
    assert ratio > 0
    np.testing.assert_allclose(covariance, ratio * reference, rtol=0, atol=TOL * ratio)


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


def test_names_and_linear_predictor_match_brms_at_brms_draws():
    case = ORACLE["fit"]
    fitted = fit(PENALIZED, data=TRAIN, data2=DATA2, method="svi", num_steps=0)
    assert set(fitted.variables(num_samples=1)) == set(case["variables"])
    (block,) = fitted.formula.design_terms
    assert set(get_prior(PENALIZED, data=TRAIN, data2=DATA2)) >= {
        block.qvr_name + "_scale"
    }
    theirs = np.asarray(case["design"])
    # The penalty's eigenvalues are distinct, so only column signs are free.
    sign = np.sign((block.data * theirs).sum(0))
    np.testing.assert_allclose(block.data * sign, theirs, rtol=0, atol=TOL)
    draws = case["draws"]
    sds = np.asarray(draws["sds_sregion_1"])
    coefficients = np.stack(
        [np.asarray(draws[f"s_sregion_1[{j}]"]) for j in range(1, 6)], -1
    )
    pinned = _pin(
        fitted,
        {
            "intercept": np.asarray(draws["b_Intercept"]),
            "sigma": np.asarray(draws["sigma"]),
            block.qvr_name + "_scale": sds,
            block.qvr_name + "_z": coefficients * sign / sds[:, None],
        },
    )
    torch.testing.assert_close(
        pinned.posterior_linpred()[0].double(),
        torch.tensor(case["linpred"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )
    torch.testing.assert_close(
        pinned.posterior_linpred(new_data=FRESH)[0].double(),
        torch.tensor(case["fresh_linpred"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )
    # brms refuses an unknown region, with or without allow_new_levels.
    assert case["new_level_error"].startswith("New factor levels are not allowed.")
    assert case["new_level_allowed_error"] == case["new_level_error"]
    unknown = FRESH.head(1).assign(region="r9")
    with pytest.raises(ValueError, match="new regions are not allowed"):
        pinned.posterior_linpred(new_data=unknown)
    with pytest.raises(ValueError, match="new regions are not allowed"):
        pinned.posterior_linpred(new_data=unknown, allow_new_levels=True)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_markov_random_field_fits_predict_refit_and_restore(method, tmp_path):
    options = (
        {"num_steps": 5}
        if method == "svi"
        else {"num_warmup": 6, "num_samples": 4, "num_chains": 1}
    )
    formula = 'y ~ w + s(region, bs = "mrf", xt = list(nb = nb_irregular), by = h)'
    fitted = fit(formula, data=TRAIN, data2=DATA2, method=method, seed=3, **options)
    if method != "svi":
        assert isinstance(fitted.posterior, MCMCResult)
    assert {"sds_sregionhu_1", "sds_sregionhv_1", "s_sregionhv_1[5]"} <= set(
        fitted.variables(num_samples=3)
    )
    linpred = fitted.posterior_linpred(num_samples=3, seed=1)
    assert linpred.shape[-1] == len(TRAIN) and torch.isfinite(linpred).all()
    fresh = fitted.posterior_linpred(new_data=FRESH, num_samples=3, seed=1)
    for name in ("posterior_epred", "posterior_predict"):
        value = getattr(fitted, name)(new_data=FRESH, num_samples=3, seed=1)
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
    # The smooth's contribution is the predictor without its other terms.
    term = next(iter({block.term for block in fitted.formula.design_terms}))
    contribution = conditional_smooths(fitted, term, FRESH, num_samples=3, seed=1)
    draws = fitted.draws(num_samples=3, seed=1)
    rest = draws["intercept"][..., None] + draws["beta_w"][..., None] * torch.as_tensor(
        FRESH["w"].to_numpy(), dtype=draws["beta_w"].dtype
    )
    torch.testing.assert_close(
        (contribution + rest).double(), fresh.double(), atol=1e-5, rtol=0
    )
    path = tmp_path / "mrf.pt"
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
        effects = fitted.conditional_smooths()
        assert effects.effects[0].effects == ("region", "h")


def test_markov_random_field_in_distributional_and_joint_models():
    single = fit(
        bf("y ~ w", 'sigma ~ s(region, bs = "mrf", xt = list(penalty = P_ring))'),
        data=TRAIN,
        data2=DATA2,
        method="svi",
        num_steps=3,
        seed=1,
    )
    assert "sds_sigma_sregion_1" in single.variables(num_samples=2)
    joint = fit(
        bf(PENALIZED)
        + bf("w ~ s(region, bs = 'mrf', xt = list(nb = nb_split))")
        + set_rescor(False),
        data=TRAIN,
        data2=DATA2,
        method="svi",
        num_steps=3,
        seed=1,
    )
    names = joint.variables(num_samples=2)
    assert {"sds_y_sregion_1", "sds_w_sregion_1", "bs_w_sregion_1"} <= set(names)
    predicted = joint.posterior_linpred(new_data=FRESH, num_samples=2)
    assert set(predicted) == {"y", "w"}
    assert all(torch.isfinite(value).all() for value in predicted.values())


@pytest.mark.parametrize(
    ("spec", "data2", "message"),
    [
        ('s(region, bs = "mrf")', DATA2, "requires xt = list"),
        ('s(region, bs = "mrf", xt = "P_irregular")', DATA2, "requires xt = list"),
        (
            's(region, bs = "mrf", xt = list(penalty = P_irregular, nb = nb_split))',
            DATA2,
            "requires xt = list",
        ),
        (
            's(region, bs = "mrf", xt = list(polys = P_irregular, nb = nb_split))',
            DATA2,
            "boundary polygons are not supported",
        ),
        ('s(region, bs = "mrf", xt = list(penalty = 3))', DATA2, "given in data2"),
        (
            's(region, bs = "mrf", k = 7, xt = list(penalty = P_irregular))',
            DATA2,
            "MRF basis dimension set too high",
        ),
        (
            's(region, bs = "mrf", k = 1, xt = list(penalty = P_irregular))',
            DATA2,
            "keeps no penalized coordinate",
        ),
        (
            's(region, bs = "mrf", k = 4, xt = list(penalty = P_ring))',
            DATA2,
            "cuts through a repeated eigenvalue",
        ),
        (
            's(region, bs = "mrf", m = 2, xt = list(penalty = P_irregular))',
            DATA2,
            "m is not accepted",
        ),
        ('s(w, bs = "mrf", xt = list(penalty = P_irregular))', DATA2, "one factor"),
        (
            's(region, h, bs = "mrf", xt = list(penalty = P_irregular))',
            DATA2,
            "one factor",
        ),
        (
            's(region, bs = "mrf", xt = list(penalty = P))',
            {"P": DATA2["P_irregular"].iloc[:5, :5]},
            "do not match the regions",
        ),
        (
            's(region, bs = "mrf", xt = list(penalty = P))',
            {"P": DATA2["P_unnamed"][:5, :5]},
            "need a 6 by 6 matrix",
        ),
        (
            's(region, bs = "mrf", xt = list(penalty = P))',
            {"P": DATA2["P_irregular"].set_axis(list("abcdef"), axis=0)},
            "same distinct region labels",
        ),
        (
            's(region, bs = "mrf", xt = list(penalty = P))',
            {"P": DATA2["P_unnamed"] + np.triu(np.ones((6, 6)), 1)},
            "finite and symmetric",
        ),
        (
            's(region, bs = "mrf", xt = list(penalty = P))',
            {"P": -DATA2["P_unnamed"]},
            "positive semidefinite",
        ),
        (
            's(region, bs = "mrf", xt = list(nb = A))',
            {"A": np.triu(np.ones((6, 6)), 1)},
            "listed symmetrically",
        ),
        ('t2(w, region, bs = c("cr", "mrf"))', DATA2, "margins must use bases"),
    ],
)
def test_invalid_markov_random_fields_are_rejected(spec, data2, message):
    with pytest.raises(ValueError, match=message):
        fit(f"y ~ {spec}", data=TRAIN, data2=data2, method="svi", num_steps=0)


def test_structure_must_be_a_data2_entry():
    # brms reads xt from R's global environment and not from data2.
    assert ORACLE["errors"]["data2"] == "object 'P_data2' not found"
    assert ORACLE["errors"]["too_high"] == "MRF basis dimension set too high"
    assert ORACLE["errors"]["no_structure"].startswith("penalty matrix, boundary")
    with pytest.raises(Exception, match="P_missing"):
        fit(
            'y ~ s(region, bs = "mrf", xt = list(penalty = P_missing))',
            data=TRAIN,
            data2=DATA2,
            method="svi",
            num_steps=0,
        )
