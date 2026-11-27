"""Automatic conditional effects and make_conditions against brms.

Oracle: ``oracles/brms_conditional_effects.R`` (brms 2.20.4 fits of the same
data). brms's estimates depend on its own draws, so the oracle pins what
does not: the default effects and their order, the list names, the
conditions chosen for other variables, the grids (resolution, integer
variables, trials, ``int_conditions`` with labels, surfaces and
``too_far``), the label columns, the point frames of ``select_points``, and
the spaghetti layout. Estimates are checked against the fit's own
predictions at the grid rows, summarized as brms's ``posterior_summary``.
Numeric tolerance 1e-9 relative, from the 15 significant digits of the export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.diagnostics import prediction_grid
from quivers.formulas import MultivariateFit, bf, conditional_smooths, fit, set_rescor
from quivers.modeling.artifact import load_fit, save_fit
from quivers.formulas._brms_effects import (
    BrmsConditionalEffects,
    exclude_too_far,
    make_conditions,
    rows2labels,
)

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_conditional_effects.json").read_text()
)
DATA = pd.DataFrame(ORACLE["data"])
DATA["o"] = DATA["s"].astype(int).astype(str)
SKIP = {"log(w)"}


def svi(formula, **options):
    return fit(formula, data=DATA, method="svi", num_steps=20, seed=1, **options)


@pytest.fixture(scope="module")
def gauss():
    return svi("y ~ x * z + f + log(w) + (1 | g)", distributional={"sigma": "~ x"})


def same(actual: pd.Series, expected: list, name: str) -> None:
    values = actual.tolist()
    assert len(values) == len(expected), (name, len(values), len(expected))
    for a, e in zip(values, expected, strict=True):
        if e is None:
            assert a is None or (isinstance(a, float) and np.isnan(a)), (name, a)
        elif isinstance(e, (int, float)) and not isinstance(e, bool):
            assert float(a) == pytest.approx(e, rel=1e-9, abs=1e-12), (name, a, e)
        else:
            assert str(a) == str(e), (name, a, e)


def check(result: BrmsConditionalEffects, key: str) -> None:
    case = ORACLE[key]
    names = case["names"] if isinstance(case["names"], list) else [case["names"]]
    assert list(result.names()) == names
    for effect, expected in zip(result.effects, case["effects"], strict=True):
        effects = expected["effects"]
        effects = effects if isinstance(effects, list) else [effects]
        assert list(effect.effects) == effects
        assert effect.surface == expected["surface"]
        assert effect.response == expected["response"]
        for column, values in expected["grid"].items():
            if column in SKIP:
                continue
            assert column in effect.frame.columns, (key, column)
            same(effect.frame[column], values, f"{key}:{column}")
        if expected["points"] is not None:
            assert effect.points is not None
            for column, values in expected["points"].items():
                same(effect.points[column], values, f"{key}:points:{column}")
        if expected["spaghetti_rows"] is not None:
            assert effect.spaghetti is not None
            assert len(effect.spaghetti) == expected["spaghetti_rows"]
            labels = sorted(str(v) for v in effect.spaghetti["sample__"].unique())
            assert labels == sorted(expected["spaghetti_samples"])


def test_make_conditions_matches_brms():
    for key, variables in (("z_f", ["z", "f"]), ("w", "w")):
        frame = make_conditions(DATA, variables)
        for column, values in ORACLE["make_conditions"][key].items():
            same(frame[column], values, column)
    assert rows2labels(pd.DataFrame({"a": [1.0, 2.004], "b": ["u", "v"]})) == [
        "a = 1 & b = u",
        "a = 2 & b = v",
    ]
    with pytest.raises(ValueError, match="not in the data"):
        make_conditions(DATA, "nope")


def test_default_effects_conditions_and_grids(gauss):
    check(gauss.conditional_effects(resolution=9), "gauss_default")


def test_effects_conditions_and_int_conditions(gauss):
    check(
        gauss.conditional_effects(
            ["z:x", "f"],
            conditions=make_conditions(DATA, "f"),
            int_conditions={"z": [-1, 1]},
            resolution=7,
        ),
        "gauss_conditions",
    )
    check(
        gauss.conditional_effects(
            "x:z", int_conditions={"z": {"low": -1, "high": 1}}, resolution=5
        ),
        "gauss_named_int",
    )


def test_surface_with_too_far(gauss):
    check(
        gauss.conditional_effects("x:z", surface=True, resolution=6, too_far=0.2),
        "gauss_surface",
    )


def test_points_spaghetti_and_dpar(gauss):
    check(
        gauss.conditional_effects(
            "x",
            conditions=make_conditions(DATA, ["z", "f"]),
            select_points=0.1,
            resolution=4,
        ),
        "gauss_points",
    )
    check(
        gauss.conditional_effects(["x", "x:f"], spaghetti=True, ndraws=3, resolution=4),
        "gauss_spaghetti",
    )
    check(gauss.conditional_effects("x", dpar="sigma", resolution=4), "gauss_sigma")


def test_trials_and_monotonic_integer_grids():
    model = svi("ys | trials(k) ~ x + mo(m)", family="binomial")
    check(model.conditional_effects(resolution=5), "binom_default")


def test_categorical_effects_per_category():
    model = svi("o ~ x + f", family="categorical")
    check(model.conditional_effects("x", categorical=True, resolution=3), "categorical")
    with pytest.raises(ValueError, match="categorical=True"):
        model.conditional_effects("x", resolution=3)
    with pytest.raises(ValueError, match="posterior_epred"):
        model.conditional_effects(
            "x", categorical=True, method="posterior_predict", resolution=3
        )


def test_smooth_terms_contribute_their_variables():
    model = svi("y ~ z + s(x, by = f, k = 4)")
    check(model.conditional_effects(resolution=4), "smooth_default")


def test_joint_fits_loop_effects_then_responses(monkeypatch: pytest.MonkeyPatch):
    model = svi(bf("y ~ x + f") + bf("y2 ~ z") + set_rescor(False))

    def rebuild(*_args):
        raise AssertionError("joint conditional effects must use the joint fit")

    monkeypatch.setattr(MultivariateFit, "_response_fit", rebuild)
    check(model.conditional_effects(resolution=3), "mv_default")
    check(model.conditional_effects(resp="y2", resolution=3), "mv_resp")


def test_joint_grids_keep_fitted_response_labels_and_draws(
    monkeypatch: pytest.MonkeyPatch,
):
    labels = ("3", "1", "4", "2")
    frame = DATA.assign(o=pd.Categorical(DATA["o"], categories=labels, ordered=True))
    model = fit(
        bf("o ~ x", family="categorical") + bf("y2 ~ x") + set_rescor(False),
        data=frame,
        method="svi",
        num_steps=2,
        seed=1,
    )
    assert isinstance(model, MultivariateFit)

    def rebuild(*_args):
        raise AssertionError(
            "labeling a joint prediction grid must use fitted metadata"
        )

    monkeypatch.setattr(MultivariateFit, "_response_fit", rebuild)
    grid = prediction_grid({"x": [-1.0, 0.0, 1.0]}, conditions={})
    draws = {"num_samples": 3, "seed": 2}
    for method, predict in (
        ("epred", model.posterior_epred),
        ("linpred", model.posterior_linpred),
    ):
        effects = model.conditional_effects(grid, method=method, **draws)
        expected = predict(new_data=grid.to_frame(), **draws)
        assert effects["o"].labels == labels
        assert effects["y2"].labels == ()
        for response, result in effects.items():
            assert torch.equal(result.draws, expected[response].unsqueeze(3))


def test_estimates_summarize_the_fits_own_predictions(gauss):
    result = gauss.conditional_effects("x", resolution=5, prob=0.8, robust=False)
    effect = result["x"]
    rows = effect.frame.drop(
        columns=[c for c in effect.frame.columns if c.endswith("__")]
    )
    draws = gauss.posterior_epred(
        new_data=rows, re_formula="~0", allow_new_levels=True
    ).double()
    pooled = draws.reshape(-1, draws.shape[-1])
    np.testing.assert_allclose(effect.frame["estimate__"], pooled.mean(0), rtol=1e-12)
    np.testing.assert_allclose(effect.frame["se__"], pooled.std(0), rtol=1e-12)
    np.testing.assert_allclose(
        effect.frame["lower__"], torch.quantile(pooled, 0.1, dim=0), rtol=1e-12
    )
    robust = gauss.conditional_effects("x", resolution=5)["x"]
    np.testing.assert_allclose(
        robust.frame["estimate__"], torch.quantile(pooled, 0.5, dim=0), rtol=1e-12
    )
    torch.testing.assert_close(effect.draws, draws)


def test_methods_and_refusals(gauss):
    linear = gauss.conditional_effects("x", method="posterior_linpred", resolution=3)
    predicted = gauss.conditional_effects("x", method="predict", resolution=3)
    assert len(linear["x"].frame) == len(predicted["x"].frame) == 3
    with pytest.raises(ValueError, match="invalid for this model"):
        gauss.conditional_effects("nope")
    with pytest.raises(ValueError, match="higher than 2"):
        gauss.conditional_effects("x:z:f")
    with pytest.raises(ValueError, match="spaghetti and surface"):
        gauss.conditional_effects("x:z", spaghetti=True, surface=True)
    with pytest.raises(ValueError, match="explicit grid"):
        gauss.conditional_effects(
            prediction_grid({"x": [0.0]}, conditions={"z": [0.0]}), resolution=3
        )
    with pytest.raises(ValueError, match="explicit prediction grids"):
        gauss.conditional_effects("x", num_samples=3)


def test_exclude_too_far_matches_mgcv_geometry():
    grid_x, grid_z = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    far = exclude_too_far(
        grid_x.ravel(), grid_z.ravel(), np.array([0.0]), np.array([0.0]), 0.5
    )
    # Only the grid point at the data point and its unit neighbours are kept.
    assert far.tolist() == [False, False, True, False, True, True, True, True, True]


@pytest.mark.parametrize("method", ["hmc", "nuts"])
def test_sampler_fits(method):
    model = fit(
        "y ~ x + f",
        data=DATA,
        method=method,
        num_warmup=15,
        num_samples=15,
        num_chains=1,
        seed=2,
    )
    result = model.conditional_effects(resolution=4)
    assert result.names() == ("x", "f")


@pytest.fixture(scope="module")
def smooth():
    return svi("y ~ z + s(x, by = f, k = 4)")


def test_conditional_smooths_grids_match_brms(smooth):
    """Grids follow brms 2.23.0, which differs from the 2.20.4 oracle in two places.

    brms 2.23.0 (``R/conditional_smooths.R``) moves the first ``by`` variable of
    a one-covariate smooth into the plotted effects, and with ``surface=FALSE``
    holds a numeric second covariate at its mean and mean plus and minus one
    standard deviation. brms 2.20.4 did neither, so the oracle pins the shared
    grid values, and the 2.23.0 layout is checked against its rule.
    """
    for key, options in (
        ("smooths", {}),
        ("smooths_flat", {"int_conditions": {"f": ["p", "r"]}}),
    ):
        result = smooth.conditional_smooths(resolution=4, **options)
        expected = ORACLE[key]
        assert list(result.names()) == [expected["names"]]
        effect, oracle = result.effects[0], expected["effects"][0]
        assert effect.effects == ("x", "f")
        for column in ("x", "f", "effect1__"):
            same(effect.frame[column], oracle["grid"][column], column)
        assert set(effect.frame["cond__"].astype(str)) == {"1"}
        assert effect.frame["effect2__"].tolist() == effect.frame["f"].tolist()
    surface = svi("y ~ s(x, z, k = 8)")
    check(surface.conditional_smooths(resolution=5, too_far=0.3), "smooths_surface")
    lines = surface.conditional_smooths(resolution=5, surface=False).effects[0]
    mean, sd = DATA["z"].mean(), DATA["z"].std()
    assert len(lines.frame) == 15 and not lines.surface
    np.testing.assert_allclose(
        sorted(set(lines.frame["z"])), [mean - sd, mean, mean + sd], rtol=1e-12
    )
    full = ORACLE["smooths_lines"]["effects"][0]["grid"]
    np.testing.assert_allclose(sorted(set(lines.frame["x"])), sorted(set(full["x"])))
    assert list(lines.frame["effect2__"].cat.categories) == [
        format(round(v, 2), ".15g") for v in (mean + sd, mean, mean - sd)
    ]


def test_conditional_smooths_summarize_the_smooth_contribution(smooth):
    from_function = smooth.conditional_smooths(resolution=4).effects[0]
    rows = from_function.frame[["x", "f"]].assign(z=0.0, y=0.0)
    draws = conditional_smooths(smooth, "s(x, by=f, k=4)", rows).double()
    torch.testing.assert_close(from_function.draws, draws)
    pooled = draws.reshape(-1, draws.shape[-1])
    np.testing.assert_allclose(
        from_function.frame["estimate__"], torch.quantile(pooled, 0.5, dim=0)
    )
    with pytest.raises(ValueError, match="no valid smooth"):
        smooth.conditional_smooths("s(w)")


def test_restored_artifact_reproduces_effects(gauss, tmp_path):
    save_fit(gauss, tmp_path / "fit")
    restored = load_fit(tmp_path / "fit")
    before = gauss.conditional_effects("x:z", resolution=4, seed=3)["x:z"].frame
    after = restored.conditional_effects("x:z", resolution=4, seed=3)["x:z"].frame
    pd.testing.assert_frame_equal(before, after)
