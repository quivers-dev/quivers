"""brms ``thres(n, gr)`` beyond one ordinal response.

Group-specific threshold vectors in a multivariate model, with the
``hurdle_cumulative`` family, and in mixtures of ordinal families. The
standata, Stan statements, messages, variable names, and the values of
``log_lik`` and ``posterior_epred`` at fixed draws are pinned by
``oracles/brms_threshold_group_models.R``. brms cannot evaluate ``log_lik``
or ``posterior_epred`` of a mixture with merged thresholds (pinned), so
mixtures are checked against the mixture law written in this file from
brms's Stan statements.

Densities are compared in Float64 with category probabilities written in
``test_threshold_groups`` from brms's Stan functions, one group at a time
with the group's own number of categories: tolerance 1e-10 for values and
first derivatives and 1e-8 for second derivatives. Fitted programs run in
Float32 and are compared with brms at relative 1e-4 and absolute 1e-5.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import test_threshold_groups as groups
from quivers.diagnostics.cross_validation import FoldPlan
from quivers.formulas import bf, fit, mixture
from quivers.inference import trace
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_threshold_group_models.json").read_text()
)
RTOL, ATOL = 1e-4, 1e-5
LEVELS, COUNTS = groups.LEVELS, groups.COUNTS
#: Slices of one component's six thresholds, group by group.
SLICES = (slice(0, 3), slice(3, 5), slice(5, 6))


def _frame(source: str = "data") -> pd.DataFrame:
    """Read brms's rows with zero-based ordinal codes.

    ``z`` keeps its codes: zero is the hurdle and ``k`` is category ``k``.
    """
    frame = pd.DataFrame(ORACLE[source])
    return frame.assign(y=frame.y - 1, r=frame.r - 1)


def _fit(name: str, data: pd.DataFrame | None = None, **settings):
    settings = {"seed": 1} | (settings or {"method": "svi", "num_steps": 1})
    data = _frame() if data is None else data
    if name == "joint":
        return fit(
            bf("y | thres(gr = g) ~ x", family="cumulative")
            + bf("r | thres(3) ~ x", family="cumulative")
            + bf("w ~ x"),
            data=data,
            **settings,
        )
    if name == "hurdle":
        formula, family = "z | thres(gr = g) ~ x", "hurdle_cumulative"
    elif name == "mixture":
        formula, family = "y | thres(gr = g) ~ x", mixture("cumulative", "acat")
    else:
        assert name == "shared"
        formula = "y | thres(gr = g) ~ x"
        family = mixture("acat", "cumulative", order="mu")
    return fit(formula, data=data, family=family, **settings)


def _vectors(flat: torch.Tensor) -> list[torch.Tensor]:
    return [flat[part] for part in SLICES]


def _probabilities(
    kind: str, eta: torch.Tensor, thresholds: list[torch.Tensor], group: str
) -> torch.Tensor:
    return groups._category_probabilities(
        kind, "logit", eta, thresholds[LEVELS.index(group)]
    )


def _hurdle_reference(
    values: dict[str, torch.Tensor], frame: pd.DataFrame
) -> torch.Tensor:
    """brms ``hurdle_cumulative``: code zero has probability ``hu``."""
    rows = []
    for z, x, group in zip(frame.z, frame.x, frame.g, strict=True):
        ordinal = _probabilities(
            "cumulative", values["b_x"] * x, _vectors(values["tau"]), group
        )
        rows.append(
            values["hu"].log()
            if z == 0
            else (1 - values["hu"]).log() + ordinal[int(z) - 1].log()
        )
    return torch.stack(rows)


def _mixture_reference(
    values: dict[str, torch.Tensor],
    frame: pd.DataFrame,
    kinds: tuple[str, str] = ("cumulative", "acat"),
) -> torch.Tensor:
    """brms's mixture: ``log_sum_exp`` of the weighted component masses.

    The proportions are normalized, as the program's mixture law does, so
    that derivatives are taken along the same coordinates.
    """
    weights = values["theta"] / values["theta"].sum()
    rows = []
    for y, x, group in zip(frame.y, frame.x, frame.g, strict=True):
        parts = [
            weights[index].log()
            + _probabilities(
                kind,
                values[f"b_mu{index + 1}_x"] * x,
                _vectors(values[f"tau{index + 1}"]),
                group,
            )[int(y)].log()
            for index, kind in enumerate(kinds)
        ]
        rows.append(torch.logsumexp(torch.stack(parts), 0))
    return torch.stack(rows)


def _joint_reference(
    values: dict[str, torch.Tensor], frame: pd.DataFrame
) -> dict[str, torch.Tensor]:
    """One grouped cumulative, one cumulative, and one Normal response."""
    grouped, single = [], []
    for y, r, x, group in zip(frame.y, frame.r, frame.x, frame.g, strict=True):
        grouped.append(
            _probabilities(
                "cumulative", values["b_y_x"] * x, _vectors(values["tau_y"]), group
            )[int(y)].log()
        )
        single.append(
            groups._category_probabilities(
                "cumulative", "logit", values["b_r_x"] * x, values["tau_r"]
            )[int(r)].log()
        )
    x = torch.tensor(frame.x.to_numpy(), dtype=torch.float64)
    # A joint program holds its real inputs in Float32, so the reference
    # scores the same rounded response values.
    w = torch.tensor(frame.w.to_numpy(), dtype=torch.float32).double()
    normal = torch.distributions.Normal(
        values["b_w_Intercept"] + values["b_w_x"] * x, values["sigma_w"]
    ).log_prob(w)
    return {"y": torch.stack(grouped), "r": torch.stack(single), "w": normal}


def _prefixed(prefix: str, sites: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {prefix + site: value for site, value in sites.items()}


def _sites(name: str, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map brms-scale parameters of one draw to the program's sites."""
    if name == "hurdle":
        return groups._coordinates("cumulative", _vectors(values["tau"])) | {
            "beta_x": values["b_x"],
            "hu": values["hu"],
        }
    if name == "mixture":
        return (
            _prefixed(
                "dpar_mu1_",
                groups._coordinates("cumulative", _vectors(values["tau1"]))
                | {"beta_x": values["b_mu1_x"]},
            )
            | _prefixed(
                "dpar_mu2_",
                groups._coordinates("acat", _vectors(values["tau2"]))
                | {"beta_x": values["b_mu2_x"]},
            )
            | {"theta": values["theta"]}
        )
    if name == "shared":
        # Component two (cumulative) holds the vectors both components read.
        return _prefixed(
            "dpar_mu2_",
            groups._coordinates("cumulative", _vectors(values["tau1"]))
            | {"beta_x": values["b_mu2_x"]},
        ) | {"dpar_mu1_beta_x": values["b_mu1_x"], "theta": values["theta"]}
    assert name == "joint"
    single = values["tau_r"]
    return (
        _prefixed(
            "response_y__",
            groups._coordinates("cumulative", _vectors(values["tau_y"]))
            | {"beta_x": values["b_y_x"]},
        )
        | _prefixed(
            "response_r__",
            {
                "intercept": -single.mean(),
                "cutpoint_log_spacing": (single[1:] - single[:-1]).log(),
                "beta_x": values["b_r_x"],
            },
        )
        | _prefixed(
            "response_w__",
            {
                "intercept": values["b_w_Intercept"],
                "beta_x": values["b_w_x"],
                "sigma": values["sigma_w"],
            },
        )
    )


#: Parameter layouts of each model's flat evaluation point.
LAYOUTS: dict[str, dict[str, slice | int]] = {
    "hurdle": {"b_x": 0, "hu": 1, "tau": slice(2, 8)},
    "mixture": {
        "b_mu1_x": 0,
        "b_mu2_x": 1,
        "theta": slice(2, 4),
        "tau1": slice(4, 10),
        "tau2": slice(10, 16),
    },
    "shared": {"b_mu1_x": 0, "b_mu2_x": 1, "theta": slice(2, 4), "tau1": slice(4, 10)},
    "joint": {
        "b_y_x": 0,
        "b_r_x": 1,
        "b_w_Intercept": 2,
        "b_w_x": 3,
        "sigma_w": 4,
        "tau_y": slice(5, 11),
        "tau_r": slice(11, 14),
    },
}
TAU = [-0.9, 0.2, 1.4, -0.3, 0.8, 0.25]
POINTS = {
    "hurdle": [0.6, 0.2, *TAU],
    "mixture": [0.6, -0.4, 0.35, 0.65, *TAU, -0.5, 0.1, 0.9, 0.4, -0.2, 0.3],
    "shared": [0.6, -0.4, 0.35, 0.65, *TAU],
    "joint": [0.6, -0.3, 0.1, 0.45, 0.8, *TAU, -1.0, 0.1, 1.2],
}


def _unpack(name: str, flat: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: flat[part] for key, part in LAYOUTS[name].items()}


def _total(name: str, values: dict[str, torch.Tensor]) -> torch.Tensor:
    frame = _frame()
    if name == "hurdle":
        return _hurdle_reference(values, frame).sum()
    if name == "mixture":
        return _mixture_reference(values, frame).sum()
    if name == "shared":
        # Both components read component one's vectors (brms fixed_Intercept).
        return _mixture_reference(
            values | {"tau2": values["tau1"]}, frame, ("acat", "cumulative")
        ).sum()
    return sum(part.sum() for part in _joint_reference(values, frame).values())


def test_brms_counts_thresholds_above_the_hurdle_and_per_component():
    cases = ORACLE["cases"]
    assert cases["hurdle"]["nthres"] == [3, 2, 1]
    assert cases["hurdle_column"]["nthres"] == [3, 2, 1]
    assert cases["hurdle_scalar"]["nthres"] == [3, 3, 3]
    # thres(3) of a hurdle response with codes 0, ..., 4 is three thresholds.
    assert cases["hurdle_plain"]["nthres"] == 3
    assert cases["hurdle_too_few"]["message"] == (
        "Number of thresholds is smaller than required by the response."
    )
    assert any(
        "hurdle_cumulative_ordered_logistic_merged_lpmf" in line
        for line in cases["hurdle"]["lines"]
    )
    mixed = cases["mixture"]["lines"]
    assert "ordered[nthres[2]] Intercept_mu1_2;" in " ".join(mixed)
    assert "vector[nthres[3]] Intercept_mu2_3;" in " ".join(mixed)
    assert cases["mixture"]["nthres"] == [3, 2, 1]
    assert cases["mixture_hurdle"]["status"] == "ok"

    def counts(formula: str, family) -> tuple[int, ...]:
        fitted = fit(
            formula, data=_frame(), family=family, method="svi", num_steps=1, seed=1
        )
        resolved = fitted.formula.modifiers.threshold_groups
        return resolved.counts

    assert counts("z | thres(gr = g) ~ x", "hurdle_cumulative") == (3, 2, 1)
    assert counts("z | thres(n, gr = g) ~ x", "hurdle_cumulative") == (3, 2, 1)
    assert counts("z | thres(3, gr = g) ~ x", "hurdle_cumulative") == (3, 3, 3)
    assert counts(
        "z | thres(gr = g) ~ x", mixture("hurdle_cumulative", "hurdle_cumulative")
    ) == (3, 2, 1)
    assert counts("y | thres(gr = g) ~ x", mixture("cumulative", "acat")) == (3, 2, 1)
    plain = fit(
        "z | thres(3) ~ x",
        data=_frame(),
        family="hurdle_cumulative",
        method="svi",
        num_steps=1,
        seed=1,
    )
    assert plain.posterior_epred(num_samples=1).shape[-1] == 5
    assert plain.posterior_thresholds(num_samples=1).shape[-1] == 3
    with pytest.raises(ValueError, match="response codes must be integers among"):
        fit(
            "z | thres(2) ~ x",
            data=_frame(),
            family="hurdle_cumulative",
            method="svi",
            num_steps=1,
        )


@pytest.mark.parametrize("name", sorted(POINTS))
def test_density_and_derivatives_match_group_specific_laws(name):
    torch.set_default_dtype(torch.float64)
    try:
        fitted = _fit(name)
        point = torch.tensor(POINTS[name], dtype=torch.float64)

        def model(flat: torch.Tensor) -> torch.Tensor:
            sites = _sites(name, _unpack(name, flat))
            recorded = trace(
                fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
            )
            assert set(sites) <= set(recorded.sites)
            prior = sum(recorded.sites[site].log_prob.sum() for site in sites)
            return recorded.log_joint.sum() - prior

        def reference(flat: torch.Tensor) -> torch.Tensor:
            return _total(name, _unpack(name, flat))

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
        pooled = point.clone()
        for part in LAYOUTS[name].values():
            if isinstance(part, slice) and part.stop - part.start == 6:
                first = pooled[part][:3]
                pooled[part] = torch.cat([first, first[:2], first[:1]])
        assert abs(float(reference(pooled) - reference(point))) > 0.1
    finally:
        torch.set_default_dtype(torch.float32)


def _draw_sites(
    name: str, read: Callable[[pd.Series], dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Map brms's draws to the program's coordinates, draw by draw."""
    draws = pd.DataFrame(ORACLE["values"][name]["draws"])
    rows = [_sites(name, read(draw)) for _, draw in draws.iterrows()]
    return {site: torch.stack([row[site] for row in rows]) for site in rows[0]}


def _number(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float64)


def _thresholds(draw: pd.Series, pattern: str) -> torch.Tensor:
    return torch.tensor(
        [
            draw[pattern.format(group=group, index=index + 1, position=position + 1)]
            for position, (group, count) in enumerate(zip(LEVELS, COUNTS, strict=True))
            for index in range(count)
        ],
        dtype=torch.float64,
    )


def _reported(variables: list[str]) -> list[str]:
    """Drop the fixed ``disc`` and brms's temporary ``Intercept_<g>`` vectors."""
    return [
        variable
        for variable in variables
        if not variable.startswith(("disc", "Intercept_"))
    ]


def _close(actual: torch.Tensor, expected: list) -> None:
    torch.testing.assert_close(
        actual.float(),
        torch.tensor(expected, dtype=torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )


def test_hurdle_scores_probabilities_and_names_match_brms_at_its_draws():
    oracle = ORACLE["values"]["hurdle"]
    fitted = groups._pinned(
        _fit("hurdle"),
        _draw_sites(
            "hurdle",
            lambda draw: {
                "b_x": _number(draw["b_x"]),
                "hu": _number(draw["hu"]),
                "tau": _thresholds(draw, "b_Intercept[{group},{index}]"),
            },
        ),
    )
    _close(fitted.log_likelihood()["z"][0], oracle["log_lik"])
    probabilities = fitted.posterior_epred()[0].float()
    _close(probabilities, oracle["epred"])
    # Column zero is the hurdle; a group's absent categories have mass zero.
    data = _frame()
    for label, count in zip(LEVELS, COUNTS, strict=True):
        rows = torch.tensor((data.g == label).to_numpy())
        assert bool((probabilities[:, rows, count + 2 :] == 0).all())
        assert bool((probabilities[:, rows, : count + 2] > 0).all())
    fresh = _frame("fresh")
    _close(fitted.log_likelihood(new_data=fresh)["z"][0], oracle["new_log_lik"])
    _close(fitted.posterior_epred(new_data=fresh)[0], oracle["new_epred"])
    variables = fitted.variables()
    expected = _reported(oracle["variables"])
    assert list(variables) == expected
    draws = pd.DataFrame(oracle["draws"])
    for variable in expected:
        _close(variables[variable][0], draws[variable].to_list())
    thresholds = fitted.posterior_thresholds()[0, :, [0, 14, 26]]
    assert bool(torch.isnan(thresholds[:, 1, 2:]).all())
    assert bool(torch.isnan(thresholds[:, 2, 1:]).all())


def test_joint_scores_probabilities_and_names_match_brms_at_its_draws(
    monkeypatch: pytest.MonkeyPatch,
):
    oracle = ORACLE["values"]["joint"]
    fitted = groups._pinned(
        _fit("joint"),
        _draw_sites(
            "joint",
            lambda draw: {
                "b_y_x": _number(draw["b_y_x"]),
                "b_r_x": _number(draw["b_r_x"]),
                "b_w_Intercept": _number(draw["b_w_Intercept"]),
                "b_w_x": _number(draw["b_w_x"]),
                "sigma_w": _number(draw["sigma_w"]),
                "tau_y": _thresholds(draw, "b_y_Intercept[{group},{index}]"),
                "tau_r": torch.tensor(
                    [draw[f"b_r_Intercept[{index}]"] for index in (1, 2, 3)],
                    dtype=torch.float64,
                ),
            },
        ),
    )

    def rebuild(*_args):
        raise AssertionError("naming joint thresholds must not compile a response fit")

    with monkeypatch.context() as naming:
        naming.setattr(type(fitted), "_response_fit", rebuild)
        variables = fitted.variables()
    expected = _reported(oracle["variables"])
    assert list(variables) == expected
    draws = pd.DataFrame(oracle["draws"])
    for variable in expected:
        _close(variables[variable][0], draws[variable].to_list())
    for resp in ("y", "r"):
        _close(fitted.log_likelihood(resp=resp)[resp][0], oracle[f"log_lik_{resp}"])
        _close(fitted.posterior_epred(resp=resp)[resp][0], oracle[f"epred_{resp}"])
    # summary() lists the same thresholds among the population-level rows.
    assert {"y_Intercept[a,1]", "y_Intercept[c,1]", "r_Intercept[3]"} <= set(
        fitted.summary().fixed.rows
    )


def test_mixture_scores_match_the_mixture_law_at_brms_draws():
    oracle = ORACLE["values"]["mixture"]
    # brms fits this model but cannot evaluate it afterwards, and it leaves
    # the group thresholds under unrenamed names.
    for key in ("log_lik", "epred", "new_log_lik", "new_epred"):
        assert oracle[key] == {"error": "subscript out of bounds"}
    assert "b_mu1_Intercept_2[1]" in oracle["variables"]

    def read(draw: pd.Series) -> dict[str, torch.Tensor]:
        return {
            "b_mu1_x": _number(draw["b_mu1_x"]),
            "b_mu2_x": _number(draw["b_mu2_x"]),
            "theta": torch.tensor(
                [draw["theta1"], draw["theta2"]], dtype=torch.float64
            ),
            "tau1": _thresholds(draw, "b_mu1_Intercept_{position}[{index}]"),
            "tau2": _thresholds(draw, "b_mu2_Intercept_{position}[{index}]"),
        }

    fitted = groups._pinned(_fit("mixture"), _draw_sites("mixture", read))
    draws = pd.DataFrame(oracle["draws"])
    for source in ("data", "fresh"):
        frame = _frame(source)
        expected = torch.stack(
            [_mixture_reference(read(draw), frame) for _, draw in draws.iterrows()]
        )
        new_data = None if source == "data" else frame
        torch.testing.assert_close(
            fitted.log_likelihood(new_data=new_data)["y"][0].float(),
            expected.float(),
            rtol=RTOL,
            atol=ATOL,
        )
        probabilities = fitted.posterior_epred(new_data=new_data)[0].float()
        observed = torch.tensor(frame.y.to_numpy())
        torch.testing.assert_close(
            probabilities.gather(-1, observed.expand(3, -1)[..., None])[..., 0].log(),
            expected.float(),
            rtol=RTOL,
            atol=ATOL,
        )
        for label, count in zip(LEVELS, COUNTS, strict=True):
            rows = torch.tensor((frame.g == label).to_numpy())
            assert bool((probabilities[:, rows, count + 1 :] == 0).all())
    names = set(fitted.variables())
    assert {"b_mu1_x", "theta1", "dpar_mu2_cutpoint_offset[3]"} <= names


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("name", ["hurdle", "joint", "mixture", "shared"])
def test_fit_predict_refit_and_restore(method, name, tmp_path):
    data = _frame()
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    if method != "svi":
        # Four warmup steps do not adapt the step size, and a long first
        # trajectory collapses Float32 spacings. A fixed small step keeps the
        # thresholds distinct.
        settings["control"] = {"stepsize": 0.02}
    fitted = _fit(name, **settings, seed=2)
    resp = "z" if name == "hurdle" else "y"
    extra = int(name == "hurdle")

    def one(value):
        return value[resp] if isinstance(value, dict) else value

    draws = fitted.draws(num_samples=3, seed=1)
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    if name == "shared":
        assert "dpar_mu1_threshold_location" not in draws
        assert "dpar_mu1_cutpoint_offset" not in draws
    predicted = one(fitted.posterior_predict(num_samples=3, seed=1))
    assert predicted.shape == (1, 3, len(data))
    # No draw leaves its group's categories.
    limits = torch.tensor([COUNTS[LEVELS.index(group)] + extra for group in data.g])
    assert bool((predicted <= limits).all())
    scores = fitted.log_likelihood(num_samples=3, seed=1)
    assert all(bool(torch.isfinite(value).all()) for value in scores.values())
    fresh = _frame("fresh")
    epred = one(fitted.posterior_epred(new_data=fresh, num_samples=3, seed=1))
    assert epred.shape == (1, 3, len(fresh), 4 + extra)
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
        one(restored.posterior_epred(new_data=fresh, num_samples=3, seed=1)), epred
    )
    assert sorted(restored.variables(num_samples=1)) == sorted(
        fitted.variables(num_samples=1)
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
    for key, value in held_out.items():
        torch.testing.assert_close(prediction.log_likelihood[key], value)


def test_refusals_name_the_family_or_the_response():
    data = _frame()

    def attempt(formula: str, family, **settings):
        return fit(
            formula, data=data, family=family, method="svi", num_steps=1, **settings
        )

    # brms reads one threshold count per group from one response coding.
    with pytest.raises(ValueError, match="all have the hurdle category or all lack"):
        attempt("z | thres(gr = g) ~ x", mixture("hurdle_cumulative", "cumulative"))
    with pytest.raises(ValueError, match="requires an ordinal response family"):
        attempt("y | thres(gr = g) ~ x", mixture("poisson", "poisson"))
    with pytest.raises(ValueError, match="Cannot use category specific effects"):
        attempt("z | thres(gr = g) ~ x + cs(x)", "hurdle_cumulative")
    # Group c declares one threshold above the hurdle: codes 0, 1, 2.
    with pytest.raises(ValueError, match=r"declares categories 0, ..., 2 but holds"):
        attempt("z | thres(n, gr = g) ~ x", "hurdle_cumulative").log_likelihood(
            new_data=data.iloc[[30]].assign(z=3)
        )
    with pytest.raises(ValueError, match="single response category"):
        fit(
            "z | thres(gr = g) ~ x",
            data=data.assign(z=data.z.clip(upper=1)),
            family="hurdle_cumulative",
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="thresholds_by is only valid for"):
        attempt(
            "y | thres(gr = g) ~ x", mixture("cumulative", "acat"), thresholds_by="g"
        )
