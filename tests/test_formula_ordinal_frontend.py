"""Formula ordinal likelihoods checked against finite-product probability oracles."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.formulas import OrdinalSpec, families, formula_to_qvr
from quivers.inference.trace import trace


@pytest.mark.parametrize("family", ("acat", "sratio", "cratio"))
@pytest.mark.parametrize("thresholds", ("flexible", "equidistant", "sum_to_zero"))
@pytest.mark.parametrize("rowwise", (True, False))
def test_proportional_ordinal_formula_matches_independent_products(
    family, thresholds, rowwise
):
    data = pd.DataFrame({"y": [0, 1, 2, 1], "x": [-1.0, 0.0, 0.7, 1.8]})
    descriptor = families[family].with_thresholds(thresholds)
    source = formula_to_qvr(
        "y ~ x" if rowwise else "y ~ 1",
        data=data,
        family=descriptor,
        priors={
            "intercept": "Normal(0.0, 1.2)",
            **({"beta_x": "Normal(0.0, 0.8)"} if rowwise else {}),
        },
    )
    assert module_to_source(parse(module_to_source(parse(source)))) == module_to_source(
        parse(source)
    )
    compiled = loads(source)
    sites = {"intercept": 0.3}
    if rowwise:
        sites["beta_x"] = -0.4
    if thresholds == "flexible":
        sites["cutpoint_offset"] = (1.4,)
        cuts = np.array([-0.7, 0.7])
    elif thresholds == "equidistant":
        sites["cutpoint_spacing"] = -1.4  # Noncumulative spacings are unrestricted.
        cuts = np.array([0.7, -0.7])
    else:
        sites["cutpoints_raw"] = (-0.9, 0.5)
        cuts = np.array([-0.7, 0.7])
    observations = {
        "x": torch.tensor(data.x.to_numpy(), dtype=torch.double),
        "y": torch.tensor(data.y.to_numpy()),
        **{
            name: torch.tensor(value, dtype=torch.double)
            for name, value in sites.items()
        },
    }
    recorded = trace(compiled.morphism, torch.zeros(1, 1), observations)
    eta = 0.3 - 0.4 * data.x.to_numpy() if rowwise else np.full(4, 0.3)
    expected = []
    for predictor, response in zip(eta, data.y, strict=True):
        if family == "acat":
            q = expit(predictor - cuts)
            masses = np.array([np.prod(q[:j]) * np.prod(1 - q[j:]) for j in range(3)])
            masses /= masses.sum()
        else:
            stop = expit(cuts - predictor)
            masses = np.array([stop[0], (1 - stop[0]) * stop[1], np.prod(1 - stop)])
        expected.append(math.log(masses[response]))
    assert recorded.sites["y"].log_prob.sum().item() == pytest.approx(
        sum(expected), rel=2e-6
    )
    # Compare the independent full joint on the checked reference machine too.
    joint = (
        sum(expected)
        - 0.5 * (0.3 / 1.2) ** 2
        - math.log(1.2)
        - 0.5 * math.log(2 * math.pi)
    )
    if rowwise:
        joint += -0.5 * (-0.4 / 0.8) ** 2 - math.log(0.8) - 0.5 * math.log(2 * math.pi)
    threshold_values = sites[
        next(name for name in sites if name.startswith("cutpoint"))
    ]
    for value in np.atleast_1d(threshold_values):
        joint += -0.5 * value**2 - 0.5 * math.log(2 * math.pi)
    reference = compiled.run(
        "model",
        data={"y": tuple(data.y), **({"x": tuple(data.x)} if rowwise else {})},
        sites=sites,
    )
    assert reference.log_joint == pytest.approx(joint, rel=1e-10)


def test_ordinal_metadata_and_link_changes_preserve_value_semantics():
    family = families["acat"]
    modified = family.with_links(mu="probit", disc="softplus").with_thresholds(
        "equidistant"
    )
    assert modified.ordinal == OrdinalSpec(
        kind="acat", link="probit", thresholds="equidistant"
    )
    assert family.ordinal == OrdinalSpec(kind="acat")
    assert modified.aux_params[0].fixed_default == 1.0
    assert modified == modified.with_()
    with pytest.raises(ValueError, match="ordinal"):
        families["gaussian"].with_thresholds("equidistant")


ORDINAL_CASES = [
    (kind, link)
    for kind in ("cumulative", "acat", "sratio", "cratio", "hurdle_cumulative")
    for link in ("logit", "probit", "probit_approx", "cloglog", "cauchit", "softit")
    if not (kind in {"sratio", "cratio"} and link == "softit")
]


@pytest.mark.parametrize("kind,link", ORDINAL_CASES)
def test_every_released_ordinal_family_link_has_native_and_reference_density(
    kind, link
):
    from scipy.special import ndtr

    hurdle = kind == "hurdle_cumulative"
    y = [0, 1, 2, 3] if hurdle else [0, 1, 2, 1]
    frame = pd.DataFrame({"y": y, "x": [-1.0, 0.0, 0.7, 1.8]})
    family = families[kind].with_links(mu=link)
    fixed = {"disc": 1.7, **({"hu": 0.2} if hurdle else {})}
    source = formula_to_qvr("y ~ x", data=frame, family=family, fixed_parameters=fixed)
    compiled = loads(source)
    sites = {"intercept": 0.3, "beta_x": -0.4}
    if kind in {"cumulative", "hurdle_cumulative"}:
        sites["cutpoint_log_spacing"] = (math.log(1.4),)
    else:
        sites["cutpoint_offset"] = (1.4,)
    obs = {
        "x": torch.tensor(frame.x.to_numpy(), dtype=torch.double),
        "y": torch.tensor(y),
        **{
            k: torch.tensor(v, dtype=torch.double, requires_grad=True)
            for k, v in sites.items()
        },
    }
    recorded = trace(compiled.morphism, torch.zeros(1, 1), obs)
    cuts = np.array([-0.7, 0.7])

    def cdf(z):
        if link == "logit":
            return expit(z)
        if link == "probit":
            return ndtr(z)
        if link == "probit_approx":
            return expit(0.07056 * z**3 + 1.5976 * z)
        if link == "cloglog":
            return -np.expm1(-np.exp(z))
        if link == "cauchit":
            return 0.5 + np.arctan(z) / np.pi
        sp = np.logaddexp(0.0, z)
        return sp / (1 + sp)

    expected = []
    for x, response in zip(frame.x, y, strict=True):
        z = 1.7 * (cuts - (0.3 - 0.4 * x))
        if kind in {"cumulative", "hurdle_cumulative"}:
            masses = np.diff(np.r_[0.0, cdf(z), 1.0])
        elif kind == "acat":
            q = cdf(-z)
            masses = np.array([np.prod(q[:j]) * np.prod(1 - q[j:]) for j in range(3)])
            masses /= masses.sum()
        else:
            stop = cdf(z) if kind == "sratio" else 1 - cdf(-z)
            masses = np.array([stop[0], (1 - stop[0]) * stop[1], np.prod(1 - stop)])
        if hurdle:
            masses = np.r_[0.2, 0.8 * masses]
        expected.append(math.log(masses[response]))
    assert recorded.sites["y"].log_prob.sum().item() == pytest.approx(
        sum(expected), rel=2e-6
    )
    reference = compiled.run(
        "model", data={"x": tuple(frame.x), "y": tuple(y)}, sites=sites
    )
    # Native and reference include identical explicit coefficient/threshold priors.
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=2e-6)

    # Independent central differences check every coefficient/threshold head,
    # rather than differentiating a copy of the production Torch kernel.
    names = tuple(sites)
    inputs = tuple(obs[name] for name in names)
    gradients = torch.autograd.grad(recorded.sites["y"].log_prob.sum(), inputs)

    def likelihood(point):
        a, b, t = point
        width = math.exp(t) if kind in {"cumulative", "hurdle_cumulative"} else t
        boundaries = np.array([-width / 2, width / 2])
        total = 0.0
        for x, response in zip(frame.x, y, strict=True):
            z = 1.7 * (boundaries - (a + b * x))
            if kind in {"cumulative", "hurdle_cumulative"}:
                masses = np.diff(np.r_[0.0, cdf(z), 1.0])
            elif kind == "acat":
                q = cdf(-z)
                masses = np.array(
                    [np.prod(q[:j]) * np.prod(1 - q[j:]) for j in range(3)]
                )
                masses /= masses.sum()
            else:
                stop = cdf(z) if kind == "sratio" else 1 - cdf(-z)
                masses = np.array([stop[0], (1 - stop[0]) * stop[1], np.prod(1 - stop)])
            if hurdle:
                masses = np.r_[0.2, 0.8 * masses]
            total += math.log(masses[response])
        return total

    point = np.array([float(value.detach().reshape(-1)[0]) for value in inputs])
    for index, gradient in enumerate(gradients):
        delta = np.zeros(3)
        delta[index] = 1e-5
        expected_gradient = (likelihood(point + delta) - likelihood(point - delta)) / (
            2e-5
        )
        assert gradient.item() == pytest.approx(expected_gradient, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("expression", ("cs(z)", "cs(1)", "cs(z):x"))
@pytest.mark.parametrize("family", ("acat", "sratio", "cratio"))
def test_category_specific_fixed_coefficients_lens_and_density(expression, family):
    from quivers.formulas import FormulaToQVRModule, formula_from_data

    frame = pd.DataFrame(
        {"y": [0, 1, 2, 1], "x": [-1.0, 0.0, 0.7, 1.8], "z": [0.2, -0.3, 0.5, 1.1]}
    )
    parsed = formula_from_data(f"y ~ {expression} + x", frame)
    lens = FormulaToQVRModule(families[family])
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    cs_columns = [c for c in parsed.fixed_columns if c.category_specific]
    assert len(cs_columns) == 1
    column = cs_columns[0]
    compiled = loads(module_to_source(module))
    slope = torch.tensor([0.2, -0.1], dtype=torch.double, requires_grad=True)
    intercept = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    ordinary = torch.tensor(-0.4, dtype=torch.double, requires_grad=True)
    threshold = torch.tensor(
        [math.log(1.4)] if family == "cumulative" else [1.4],
        dtype=torch.double,
        requires_grad=True,
    )
    obs = lens.fixed_column_observations(parsed)
    obs["y"] = torch.tensor(frame.y.to_numpy())
    obs.update(
        {
            "intercept": intercept,
            "beta_x": ordinary,
            f"beta_{column.qvr_name}": slope,
            "cutpoint_log_spacing"
            if family == "cumulative"
            else "cutpoint_offset": threshold,
        }
    )
    obs[column.qvr_name] = torch.tensor(column.data.copy(), dtype=torch.double)
    obs["x"] = torch.tensor(frame.x.to_numpy(), dtype=torch.double)
    recorded = trace(compiled.morphism, torch.zeros(1, 1), obs)
    eta = intercept + ordinary * obs["x"]
    gap = threshold.exp() if family == "cumulative" else threshold
    cuts = torch.cat((-gap / 2, gap / 2))
    z = cuts[None, :] - eta[:, None] - obs[column.qvr_name][:, None] * slope[None, :]
    if family == "acat":
        logits = torch.cat((torch.zeros(4, 1, dtype=torch.double), (-z).cumsum(-1)), -1)
        logp = logits.log_softmax(-1)
    elif family == "cumulative":
        cdf = torch.sigmoid(z)
        logp = torch.cat(
            (cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1
        ).log()
    else:
        stop = torch.sigmoid(z)
        logp = torch.stack(
            (stop[:, 0], (1 - stop[:, 0]) * stop[:, 1], (1 - stop).prod(-1)), -1
        ).log()
    expected = logp.gather(-1, torch.tensor(frame.y.to_numpy())[:, None]).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    params = (intercept, ordinary, slope, threshold)
    got = torch.autograd.grad(actual, params, retain_graph=True)
    want = torch.autograd.grad(expected, params)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("family", ("acat", "sratio", "cratio"))
def test_category_specific_group_intercept_compiles_without_row_count_capture(family):
    from quivers.formulas import FormulaToQVRModule, formula_from_data

    frame = pd.DataFrame(
        {"y": [0, 1, 2, 1], "x": [-1.0, 0.0, 0.7, 1.8], "g": ["a", "b", "a", "b"]}
    )
    parsed = formula_from_data("y ~ x + (0 + cs(1) | g)", frame)
    assert parsed.random_terms[0].column.category_specific
    lens = FormulaToQVRModule(families[family])
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    compiled = loads(module_to_source(module))
    observations = lens.fixed_column_observations(parsed)
    observations["y"] = torch.tensor(frame.y.to_numpy())
    observations["g_idx"] = torch.tensor(parsed.group_indices["g"])
    recorded = trace(compiled.morphism, torch.zeros(1, 1), observations)
    assert torch.isfinite(recorded.log_joint).all()


def test_cs_refusals_are_explicit_and_reserved_column_cannot_be_overridden():
    from quivers.formulas import formula_from_data

    data = pd.DataFrame({"y": [0, 1, 2], "x": [0.0, 1.0, 2.0], "g": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="ordinal"):
        formula_to_qvr("y ~ cs(x)", data=data, family="gaussian")
    with pytest.raises(ValueError, match="only valid for mu"):
        formula_to_qvr(
            "y ~ x", data=data, family="acat", distributional={"disc": "~cs(x)"}
        )
    centered = formula_to_qvr(
        "y ~ x + (0+cs(x)|g)", data=data, family="acat", reparameterize="centered"
    )
    assert "MultivariateNormal" in centered
    loads(centered)
    with pytest.raises(ValueError, match="reserved"):
        formula_from_data("y~cs(1)", data.assign(qvrCSInterceptOnes=2.0))


def _prediction_fit(formula, *, group=False):
    from quivers.formulas import fit
    from quivers.inference.mcmc.driver import MCMCResult

    frame = pd.DataFrame(
        {"y": [0, 1, 2, 1], "x": [-1.0, 0.0, 0.7, 1.8], "g": ["a", "b", "a", "b"]}
    )
    result = fit(formula, data=frame, family="acat", method="svi", num_steps=1, seed=5)
    values = {"intercept": torch.tensor(0.3), "cutpoint_offset": torch.tensor([1.4])}
    if "x" in formula:
        values["beta_x"] = torch.tensor(-0.4)
    if group:
        values.update(
            {
                "sigma_g_cs_1": torch.tensor([0.5, 0.8]),
                "z_g_cs_1": torch.tensor([0.1, -0.4, 0.6, 0.3]),
            }
        )
    else:
        values["beta_cs_1"] = torch.tensor([0.2, -0.1])
    result = result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(1, 2, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(1, 2),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=2,
        )
    )
    return result


def _acat_prediction(eta):
    cuts = torch.tensor([-0.7, 0.7])
    logits = torch.cat((torch.zeros(*eta.shape[:-1], 1), (eta - cuts).cumsum(-1)), -1)
    return logits.softmax(-1)


def test_cs_intercept_fresh_data_retains_transition_axis_and_does_not_mutate():
    result = _prediction_fit("y~1+cs(1)")
    frame = pd.DataFrame({"unused": [3.0, 4.0, 5.0]})
    before = frame.copy(deep=True)
    eta = torch.tensor([0.5, 0.2]).expand(3, 2)
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame), eta.expand(1, 2, 3, 2)
    )
    expected = _acat_prediction(eta).expand(1, 2, 3, 3)
    first = result.posterior_epred(new_data=frame)
    torch.testing.assert_close(first, expected)
    result.posterior_epred(new_data=pd.DataFrame({"unused": [0.0]}))
    torch.testing.assert_close(result.posterior_epred(new_data=frame), first)
    pd.testing.assert_frame_equal(frame, before)
    assert result.posterior_predict(new_data=frame, seed=9).shape == (1, 2, 3)
    assert result.fixed_effects()["cs(1)"].shape == (1, 2, 2)
    with pytest.raises(ValueError, match="thresholds"):
        result.posterior_linpred(new_data=frame, transform=True)
    with pytest.raises(ValueError, match="reserved"):
        result.posterior_epred(new_data=frame.assign(qvrCSInterceptOnes=4.0))


def test_cs_known_group_new_rows_and_coefficient_axes_match_independent_oracle():
    result = _prediction_fit("y~x+(0+cs(1)||g)", group=True)
    frame = pd.DataFrame({"x": [2.0, -1.0, 0.5], "g": ["b", "a", "b"]})
    effects = torch.tensor([[0.3, 0.24], [0.05, -0.32], [0.3, 0.24]])
    eta = (0.3 - 0.4 * torch.tensor(frame.x.to_numpy(), dtype=torch.float32))[
        :, None
    ] + effects
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame), eta.expand(1, 2, 3, 2)
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=frame), _acat_prediction(eta).expand(1, 2, 3, 3)
    )
    coefficients = result.random_effects()["g"]["cs(1)"]
    torch.testing.assert_close(
        coefficients, torch.tensor([[0.05, -0.32], [0.3, 0.24]]).expand(1, 2, 2, 2)
    )


def test_cs_new_group_joint_donor_and_excluded_effects():
    result = _prediction_fit("y~x+(0+cs(1)||g)", group=True)
    frame = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["fresh", "fresh", "a"]})
    predicted = result.posterior_linpred(
        new_data=frame, allow_new_levels=True, sample_new_levels="old_levels", seed=19
    )
    donor = predicted[..., 0, :] - 0.3
    allowed = torch.tensor([[0.05, -0.32], [0.3, 0.24]])
    assert torch.isclose(donor[..., None, :], allowed).all(-1).any(-1).all()
    torch.testing.assert_close(predicted[..., 0, :], predicted[..., 1, :])
    torch.testing.assert_close(
        predicted[..., 2, :], torch.tensor([0.35, -0.02]).expand(1, 2, 2)
    )
    expected = _acat_prediction(predicted)
    torch.testing.assert_close(
        result.posterior_epred(
            new_data=frame,
            allow_new_levels=True,
            sample_new_levels="old_levels",
            seed=19,
        ),
        expected,
    )
    excluded = result.posterior_linpred(
        new_data=frame.drop(columns="g"), re_formula="~0"
    )
    torch.testing.assert_close(excluded, torch.full((1, 2, 3, 2), 0.3))
    torch.testing.assert_close(
        result.posterior_epred(new_data=frame.drop(columns="g"), re_formula="~0"),
        _acat_prediction(excluded),
    )


@pytest.mark.parametrize("family", ("cumulative", "hurdle_cumulative"))
@pytest.mark.parametrize("expression", ("cs(z)", "cs(1)", "cs(z):x"))
def test_cumulative_category_specific_priors_have_checked_slack_constraints(
    family, expression
):
    data = pd.DataFrame(
        {"y": [0, 1, 2, 3], "x": [0.0, 1.0, 2.0, 3.0], "z": [0.1, 0.2, 0.3, 0.4]}
    )
    if expression == "cs(1)":
        with pytest.raises(ValueError, match="duplicates.*threshold profile"):
            formula_to_qvr(f"y ~ x + {expression}", data=data, family=family)
        return
    source = formula_to_qvr(f"y ~ x + {expression}", data=data, family=family)
    assert "ordinal_spacing_bound" in source
    assert "ordinal_spacing_bound + exp(cutpoint_log_spacing)" in source
    loads(source)
