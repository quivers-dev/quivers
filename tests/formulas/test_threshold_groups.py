"""brms ``thres(n, gr)``: one unpooled threshold vector per group.

brms gives each level of ``gr`` its own threshold vector of its own length
(``ordered[nthres[g]] Intercept_g`` merged into one vector and indexed by
``Jthres``). The standata, Stan statements, prior rows, messages, variable
names, and the values of ``log_lik`` and ``posterior_epred`` at fixed draws
are pinned by ``oracles/brms_threshold_groups.R``.

Densities are compared in Float64 with category probabilities written in
this file from brms's Stan functions, one group at a time with the group's
own number of categories: tolerance 1e-10 for values and first derivatives
and 1e-8 for second derivatives. Fitted programs run in Float32 and are
compared with brms at relative 1e-4 and absolute 1e-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from quivers.diagnostics.cross_validation import FoldPlan
from quivers.formulas import families, fit, formula_to_qvr
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_threshold_groups.json").read_text()
)
RTOL, ATOL = 1e-4, 1e-5
COUNTS = (3, 2, 1)
LEVELS = ("a", "b", "c")
FAMILIES = {
    "cumulative": ("cumulative", "logit"),
    "cumulative_probit": ("cumulative", "probit"),
    "sratio": ("sratio", "logit"),
    "cratio": ("cratio", "cloglog"),
    "acat": ("acat", "logit"),
}


def _frame(source: str = "data") -> pd.DataFrame:
    """Read brms's rows with zero-based response codes."""
    frame = pd.DataFrame(ORACLE[source])
    return frame.assign(y=frame.y - 1)


def _fit(name: str, formula: str = "y | thres(gr = g) ~ x", **settings):
    kind, link = FAMILIES[name]
    settings = {"seed": 1} | (settings or {"method": "svi", "num_steps": 1})
    return fit(
        formula, data=_frame(), family=families[kind].with_links(mu=link), **settings
    )


def _inverse_link(link: str, value: torch.Tensor) -> torch.Tensor:
    if link == "logit":
        return torch.sigmoid(value)
    if link == "probit":
        return torch.distributions.Normal(0.0, 1.0).cdf(value)
    assert link == "cloglog"
    return 1 - torch.exp(-torch.exp(value))


def _category_probabilities(
    kind: str, link: str, eta: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    """Category probabilities of one row from brms's Stan functions.

    ``thresholds`` holds the row's own group vector, so the result has one
    more entry than the group has thresholds.
    """
    if kind == "cumulative":
        cumulative = torch.cat(
            [
                eta.new_zeros(1),
                _inverse_link(link, thresholds - eta),
                eta.new_ones(1),
            ]
        )
        return cumulative[1:] - cumulative[:-1]
    if kind == "sratio":
        stop = _inverse_link(link, thresholds - eta)
        survive = torch.cat([eta.new_ones(1), torch.cumprod(1 - stop, 0)])
        return torch.cat([stop, eta.new_ones(1)]) * survive
    if kind == "cratio":
        proceed = _inverse_link(link, eta - thresholds)
        reached = torch.cat([eta.new_ones(1), torch.cumprod(proceed, 0)])
        return torch.cat([1 - proceed, eta.new_ones(1)]) * reached
    assert kind == "acat"
    step = _inverse_link(link, eta - thresholds)
    scores = torch.cat(
        [eta.new_zeros(1), torch.cumsum(step.log() - (-step).log1p(), 0)]
    )
    return torch.softmax(scores, 0)


def _coordinates(kind: str, thresholds: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map brms's group vectors to the fitted program's coordinates.

    The first group's mean is minus the intercept, the other means are
    location contrasts, and the shapes are log spacings (cumulative) or
    offsets from the first threshold.
    """
    intercept = -thresholds[0].mean()
    locations = torch.stack([vector.mean() + intercept for vector in thresholds[1:]])
    shapes = [
        (vector[1:] - vector[:-1]).log()
        if kind == "cumulative"
        else vector[1:] - vector[0]
        for vector in thresholds
    ]
    return {
        "intercept": intercept,
        "threshold_location": locations,
        "cutpoint_log_spacing" if kind == "cumulative" else "cutpoint_offset": (
            torch.cat(shapes)
        ),
    }


def _reference(
    name: str,
    beta: torch.Tensor,
    thresholds: list[torch.Tensor],
    frame: pd.DataFrame | None = None,
) -> torch.Tensor:
    """Pointwise log-likelihood, each row scored with its group's vector."""
    kind, link = FAMILIES[name]
    frame = _frame() if frame is None else frame
    rows = []
    for y, x, group in zip(frame.y, frame.x, frame.g, strict=True):
        probabilities = _category_probabilities(
            kind, link, beta * x, thresholds[LEVELS.index(group)]
        )
        rows.append(probabilities[int(y)].log())
    return torch.stack(rows)


def _thresholds() -> list[torch.Tensor]:
    return [
        torch.tensor([-0.9, 0.2, 1.4], dtype=torch.float64),
        torch.tensor([-0.3, 0.8], dtype=torch.float64),
        torch.tensor([0.25], dtype=torch.float64),
    ]


def test_brms_keeps_one_threshold_vector_of_its_own_length_per_group():
    cases = ORACLE["cases"]
    assert cases["inferred"]["nthres"] == list(COUNTS)
    assert cases["column"]["nthres"] == list(COUNTS)
    assert cases["wide"]["nthres"] == [4, 2, 3]
    assert cases["scalar"]["nthres"] == [5, 5, 5]
    # Rows index the merged vector by the start and end of their group.
    assert cases["inferred"]["Jthres"][0] == [1, 3]
    assert cases["inferred"]["Jthres"][14] == [4, 5]
    assert cases["inferred"]["Jthres"][-1] == [6, 6]
    assert (
        "ordered[nthres[1]] Intercept_1;  // temporary thresholds for centered predictors"
        in (cases["inferred"]["lines"])
    )
    assert any(
        "ordered_logistic_merged_lpmf" in line for line in cases["inferred"]["lines"]
    )
    assert cases["category_specific"]["message"] == (
        "Cannot use category specific effects in models with multiple thresholds."
    )
    for spec, counts in (
        ("y | thres(gr = g) ~ x", COUNTS),
        ("y | thres(n, gr = g) ~ x", COUNTS),
        ("y | thres(wide, gr = g) ~ x", (4, 2, 3)),
        ("y | thres(5, gr = g) ~ x", (5, 5, 5)),
    ):
        fitted = _fit("cumulative", spec)
        groups = fitted.formula.modifiers.threshold_groups
        assert groups is not None
        assert groups.levels == LEVELS and groups.counts == counts
        assert fitted.posterior_epred(num_samples=1).shape[-1] == max(counts) + 1
    source = formula_to_qvr("y | thres(gr = g) ~ x", data=_frame(), family="cumulative")
    assert "sample cutpoint_log_spacing : FormulaThresholdShape <- Normal(" in source
    assert "sample threshold_location : FormulaThresholdLocation <- Normal(" in source
    assert "object FormulaThresholdShape : FinSet 3" in source
    assert "object FormulaThresholdLocation : FinSet 2" in source


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_density_and_derivatives_match_group_specific_category_laws(name):
    torch.set_default_dtype(torch.float64)
    try:
        kind, _ = FAMILIES[name]
        fitted = _fit(name)
        beta = torch.tensor(0.6, dtype=torch.float64)
        point = torch.cat([beta[None], *_thresholds()])

        def unpack(flat: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
            return flat[0], [flat[1:4], flat[4:6], flat[6:7]]

        def model(flat: torch.Tensor) -> torch.Tensor:
            slope, vectors = unpack(flat)
            sites = _coordinates(kind, vectors) | {"beta_x": slope}
            recorded = trace(
                fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
            )
            assert set(sites) <= set(recorded.sites)
            prior = sum(recorded.sites[site].log_prob.sum() for site in sites)
            return recorded.log_joint.sum() - prior

        def reference(flat: torch.Tensor) -> torch.Tensor:
            slope, vectors = unpack(flat)
            return _reference(name, slope, vectors).sum()

        torch.testing.assert_close(
            model(point), reference(point), rtol=1e-10, atol=1e-10
        )
        gradient = torch.autograd.functional.jacobian(model, point)
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(
            gradient,
            torch.autograd.functional.jacobian(reference, point),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(model, point),
            torch.autograd.functional.hessian(reference, point),
            rtol=1e-8,
            atol=1e-8,
        )
        # One threshold vector shared by all groups is a different model.
        pooled = [vector.clone() for vector in _thresholds()]
        pooled[1] = pooled[0][:2]
        pooled[2] = pooled[0][:1]
        shared = reference(torch.cat([beta[None], *pooled]))
        assert abs(float(shared - reference(point))) > 0.1
    finally:
        torch.set_default_dtype(torch.float32)


def _pinned(fitted, values: dict[str, torch.Tensor]):
    """Replace the posterior by fixed draws on ``(chain=1, draw)`` axes."""
    draws = next(iter(values.values())).shape[0]
    return fitted.with_(
        posterior=PriorDraws(
            samples={
                name: value.float()[None].clone() for name, value in values.items()
            },
            num_samples=draws,
        )
    )


def _brms_draws(name: str) -> dict[str, torch.Tensor]:
    """Map brms's draws to the fitted program's coordinates, draw by draw."""
    kind, _ = FAMILIES[name]
    draws = pd.DataFrame(ORACLE["values"][name]["draws"])
    rows = []
    for _, draw in draws.iterrows():
        vectors = [
            torch.tensor(
                [
                    draw[THRESHOLD_NAME.format(group=group, index=k + 1)]
                    for k in range(count)
                ],
                dtype=torch.float64,
            )
            for group, count in zip(LEVELS, COUNTS, strict=True)
        ]
        rows.append(
            _coordinates(kind, vectors)
            | {"beta_x": torch.tensor(draw["b_x"], dtype=torch.float64)}
        )
    return {site: torch.stack([row[site] for row in rows]) for site in rows[0]}


#: brms's name of threshold ``index`` of group ``group``.
THRESHOLD_NAME = "b_Intercept[{group},{index}]"


def test_brms_cannot_evaluate_two_of_the_models():
    # brms's meanfield fit of the cloglog continuation-ratio model fails, and
    # its log_lik of the stopping-ratio model with merged thresholds raises.
    # Their densities are checked against the category laws written above.
    assert "dropped evaluations" in ORACLE["values"]["cratio"]["error"]
    assert ORACLE["values"]["sratio"]["log_lik"] == {"error": "subscript out of bounds"}
    assert ORACLE["cases"]["too_few"]["message"] == (
        "Number of thresholds is smaller than required by the response."
    )


@pytest.mark.parametrize("name", ["acat", "cumulative", "cumulative_probit", "sratio"])
def test_scores_probabilities_and_names_match_brms_at_its_draws(name):
    oracle = ORACLE["values"][name]
    fitted = _pinned(_fit(name), _brms_draws(name))
    scored = name != "sratio"
    if scored:
        torch.testing.assert_close(
            fitted.log_likelihood()["y"][0].float(),
            torch.tensor(oracle["log_lik"], dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )
    probabilities = fitted.posterior_epred()[0].float()
    torch.testing.assert_close(
        probabilities,
        torch.tensor(oracle["epred"], dtype=torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    # A group's absent categories have probability exactly zero, as in brms.
    data = _frame()
    for label, count in zip(LEVELS, COUNTS, strict=True):
        rows = torch.tensor((data.g == label).to_numpy())
        assert bool((probabilities[:, rows, count + 1 :] == 0).all())
    fresh = _frame("fresh")
    if scored:
        torch.testing.assert_close(
            fitted.log_likelihood(new_data=fresh)["y"][0].float(),
            torch.tensor(oracle["new_log_lik"], dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )
    torch.testing.assert_close(
        fitted.posterior_epred(new_data=fresh)[0].float(),
        torch.tensor(oracle["new_epred"], dtype=torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    # brms also lists the fixed disc and its temporary Intercept_<g> vectors.
    expected = [
        variable
        for variable in oracle["variables"]
        if variable != "disc" and not variable.startswith("Intercept_")
    ]
    variables = fitted.variables()
    assert list(variables) == expected
    draws = pd.DataFrame(oracle["draws"])
    for variable in expected:
        torch.testing.assert_close(
            variables[variable][0].float(),
            torch.tensor(draws[variable].to_numpy(), dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )
    thresholds = fitted.posterior_thresholds()[0, :, [0, 14, 26]].float()
    assert bool(torch.isnan(thresholds[:, 1, 2:]).all())
    assert bool(torch.isnan(thresholds[:, 2, 1:]).all())


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("name", ["cumulative", "acat"])
def test_fit_predict_refit_and_restore(method, name, tmp_path):
    data = _frame()
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    if method == "hmc":
        # Static HMC needs a step size that keeps the spacings finite.
        settings["control"] = {"stepsize": 0.02}
    fitted = _fit(name, **settings, seed=2)
    draws = fitted.draws(num_samples=3, seed=1)
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    predicted = fitted.posterior_predict(num_samples=3, seed=1)
    assert predicted.shape == (1, 3, len(data))
    # No draw leaves its group's categories.
    limits = torch.tensor([COUNTS[LEVELS.index(group)] for group in data.g])
    assert bool((predicted <= limits).all())
    scores = fitted.log_likelihood(num_samples=3, seed=1)["y"]
    assert scores.shape == (1, 3, len(data)) and bool(torch.isfinite(scores).all())
    fresh = _frame("fresh")
    epred = fitted.posterior_epred(new_data=fresh, num_samples=3, seed=1)
    assert epred.shape == (1, 3, len(fresh), 4)
    torch.testing.assert_close(
        epred.sum(-1), torch.ones_like(epred.sum(-1)), rtol=1e-5, atol=1e-5
    )
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert set(refitted.draws(num_samples=1)) == set(draws)
    path = tmp_path / "thresholds.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=3, seed=1), epred
    )
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
    )
    split = FoldPlan(
        assignments=tuple(index % 3 for index in range(len(data))), seed=1
    ).split(0)
    prediction = callback(split)
    held_out = prediction.fit.log_likelihood(
        new_data=data.iloc[list(split.test)], num_samples=2, seed=split.seed
    )
    torch.testing.assert_close(prediction.log_likelihood["y"], held_out["y"])


def test_refusals_name_the_group_or_the_term():
    data = _frame()
    with pytest.raises(ValueError, match="Cannot use category specific effects"):
        fit(
            "y | thres(gr = g) ~ x + cs(x)",
            data=data,
            family="acat",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="holds the response 3"):
        fit(
            "y | thres(2, gr = g) ~ x",
            data=data,
            family="cumulative",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="one positive integer number of thresholds"):
        fit(
            "y | thres(k, gr = g) ~ x",
            data=data.assign(k=[3, 4] * 18),
            family="cumulative",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="requires an ordinal response family"):
        fit(
            "y | thres(gr = g) ~ x",
            data=data,
            family="poisson",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="cannot be combined with thresholds_by"):
        fit(
            "y | thres(gr = g) ~ x",
            data=data,
            family="cumulative",
            thresholds_by="g",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="flexible thresholds"):
        fit(
            "y | thres(gr = g) ~ x",
            data=data,
            family=families["cumulative"].with_thresholds("equidistant"),
            method="svi",
            num_steps=1,
        )
    fitted = _fit("cumulative")
    with pytest.raises(ValueError, match="no fitted thresholds for \\['d'\\]"):
        fitted.posterior_epred(new_data=data.head(2).assign(g="d"))
    # Group c has two categories, so a fresh response 2 is outside them.
    with pytest.raises(ValueError, match="holds the response 2"):
        fitted.log_likelihood(new_data=data.tail(2).assign(y=2))
