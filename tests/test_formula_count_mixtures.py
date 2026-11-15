"""Count-mixture formulas agree with independent normalized response scores."""

import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference.trace import trace
from quivers.formulas._posterior import response_distribution


COUNT_FAMILIES = (
    "zero_inflated_negbinomial",
    "zero_inflated_binomial",
    "zero_inflated_beta_binomial",
    "hurdle_negbinomial",
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "y": [0, 1, 3, 0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.1, 0.5],
            "trials": [0, 2, 5, 4],
            "importance": [0.0, 0.5, 1.0, 2.0],
        }
    )


def _base_score(name, y, mean, auxiliary, trials):
    if "negbinomial" in name:
        shape = auxiliary["shape"]
        denominator = mean + shape
        return (
            torch.lgamma(y + shape)
            - torch.lgamma(shape)
            - torch.lgamma(y + 1)
            + shape * (shape.log() - denominator.log())
            + y * (mean.log() - denominator.log())
        )
    combination = (
        torch.lgamma(trials + 1) - torch.lgamma(y + 1) - torch.lgamma(trials - y + 1)
    )
    if "beta_binomial" in name:
        alpha = mean * auxiliary["phi"]
        beta = (1 - mean) * auxiliary["phi"]
        return (
            combination
            + torch.lgamma(y + alpha)
            + torch.lgamma(trials - y + beta)
            - torch.lgamma(trials + alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
            + torch.lgamma(alpha + beta)
        )
    return combination + y * mean.log() + (trials - y) * torch.log1p(-mean)


@pytest.mark.parametrize("name", COUNT_FAMILIES)
@pytest.mark.parametrize("weighted", [False, True])
def test_distributional_count_mixture_density_and_all_coefficient_gradients(
    frame, name, weighted
):
    family = families[name]
    formula = formula_from_data(
        "y ~ x", frame, weights="importance" if weighted else None
    )
    distributional = {
        parameter.name: formula_from_data("y ~ z", frame)
        for parameter in family.aux_params
    }
    lens = FormulaToQVRModule(
        family,
        distributional=distributional,
        binomial_trials="trials" if family.trials_parameter else 1,
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    loads(source)
    values = {
        "intercept": torch.tensor(-0.2, requires_grad=True),
        "beta_x": torch.tensor(0.15, requires_grad=True),
    }
    for index, parameter in enumerate(family.aux_params):
        values[f"dpar_{parameter.name}_intercept"] = torch.tensor(
            -0.4 + 0.2 * index, requires_grad=True
        )
        values[f"dpar_{parameter.name}_beta_z"] = torch.tensor(
            0.1 + 0.1 * index, requires_grad=True
        )
    observations = lens.fixed_column_observations(formula) | values
    observations["y"] = torch.tensor(frame.y.to_numpy())
    if family.trials_parameter:
        observations["trials"] = torch.tensor(frame.trials.to_numpy())
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1, dtype=torch.long),
        observations,
    )
    eta = (
        values["intercept"].double()
        + values["beta_x"].double() * observations["x"].double()
    )
    mean = eta.exp() if "negbinomial" in name else eta.sigmoid()
    auxiliary = {}
    for parameter in family.aux_params:
        linear = (
            values[f"dpar_{parameter.name}_intercept"].double()
            + values[f"dpar_{parameter.name}_beta_z"].double()
            * observations[f"dpar_{parameter.name}_z"].double()
        )
        auxiliary[parameter.name] = (
            linear.sigmoid() if parameter.name in {"zi", "hu"} else linear.exp()
        )
    y = observations["y"].double()
    trials = torch.tensor(frame.trials.to_numpy(), dtype=torch.float64)
    base = _base_score(name, y, mean, auxiliary, trials)
    zero = _base_score(name, torch.zeros_like(y), mean, auxiliary, trials)
    probability = auxiliary["hu" if name == "hurdle_negbinomial" else "zi"]
    if name == "hurdle_negbinomial":
        score = torch.where(
            y == 0,
            probability.log(),
            torch.log1p(-probability) + base - (-torch.expm1(zero)).log(),
        )
    else:
        score = torch.where(
            y == 0,
            torch.logaddexp(probability.log(), torch.log1p(-probability) + zero),
            torch.log1p(-probability) + base,
        )
    if weighted:
        score = score * observations["formula_weights"]
    expected = score.sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        assert torch.isfinite(actual_gradient)
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("name", COUNT_FAMILIES)
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_count_mixtures_fit_with_exact_auxiliary_prior_inspection(frame, name, method):
    family = families[name]
    trials = "trials" if family.trials_parameter else 1
    fixed = {
        parameter.name: 0.25 if parameter.name in {"zi", "hu"} else 2.0
        for parameter in family.aux_params
    }
    fitted = fit(
        "y ~ x",
        data=frame,
        family=name,
        binomial_trials=trials,
        fixed_parameters=fixed,
        method=method,
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=821,
    )
    mean = fitted.posterior_epred(num_samples=2, seed=4)
    assert mean.shape == (1, 2, 4)
    assert torch.isfinite(mean).all() and (mean >= 0).all()
    if family.trials_parameter:
        assert mean[..., 0].eq(0).all()
    assert set(get_prior("y ~ x", data=frame, family=name, binomial_trials=trials)) == {
        "intercept",
        "beta_x",
        *(parameter.name for parameter in family.aux_params),
    }


@pytest.mark.parametrize(
    "name",
    [
        "binomial",
        "beta_binomial",
        "zero_inflated_binomial",
        "zero_inflated_beta_binomial",
    ],
)
def test_trial_metadata_survives_parameter_link_selection(name):
    assert families[name].with_links(mu="probit").trials_parameter == "total_count"


@pytest.mark.parametrize(
    "name",
    [
        "binomial",
        "beta_binomial",
        "zero_inflated_binomial",
        "zero_inflated_beta_binomial",
    ],
)
@pytest.mark.parametrize("trials", [1000001, 33554433, 2147483647])
@pytest.mark.parametrize("runtime_trials", [False, True])
def test_large_trial_counts_preserve_integer_values_and_scipy_mass(
    name, trials, runtime_trials
):
    count = trials // 2
    frame = pd.DataFrame({"y": [count], "trials": [trials]})
    family = families[name]
    fixed = {
        parameter.name: 0.2 if parameter.name == "zi" else 3.2
        for parameter in family.aux_params
    }
    lens = FormulaToQVRModule(
        family,
        fixed_parameters=fixed,
        binomial_trials="trials" if runtime_trials else trials,
    )
    module, _ = lens.forward(formula_from_data("y ~ 1", frame))
    intercept = torch.tensor(0.0, requires_grad=True)
    observations = {"intercept": intercept, "y": torch.tensor([count])}
    if runtime_trials:
        observations["trials"] = torch.tensor([trials])
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1, dtype=torch.long),
        observations,
    )
    distribution = response_distribution(recorded, "y")
    assert distribution.total_count.item() == trials
    actual = recorded.sites["y"].log_prob.sum()
    eta = intercept.double()
    probability = eta.sigmoid()
    n = torch.tensor(float(trials), dtype=torch.float64)
    y = torch.tensor(float(count), dtype=torch.float64)
    if "beta_binomial" in name:
        precision = torch.tensor(fixed["phi"], dtype=torch.float64)
        expected = _base_score(name, y, probability, {"phi": precision}, n)
        scipy = stats.betabinom.logpmf(
            count, trials, fixed["phi"] / 2, fixed["phi"] / 2
        )
    else:
        expected = _base_score(name, y, probability, {}, n)
        scipy = stats.binom.logpmf(count, trials, 0.5)
    if name.startswith("zero_inflated_"):
        expected = expected + torch.tensor(0.8, dtype=torch.float64).log()
        scipy += float(torch.tensor(0.8, dtype=torch.float64).log())
    assert actual.dtype == torch.float64
    assert actual.item() <= 0
    assert actual.item() == pytest.approx(float(scipy), abs=3e-5)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, intercept, retain_graph=True)[0],
        torch.autograd.grad(expected, intercept)[0],
        atol=2e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("count", [2**53 + 1, 2**63 - 1])
@pytest.mark.parametrize("literal", [False, True])
@pytest.mark.parametrize(
    "family, suffix",
    [
        ("Binomial", "probs=0.5"),
        ("BetaBinomial", "concentration1=2.0, concentration0=3.0"),
        ("ZeroInflatedBinomial", "zero_prob=0.2, probs=0.5"),
        (
            "ZeroInflatedBetaBinomial",
            "zero_prob=0.2, concentration1=2.0, concentration0=3.0",
        ),
    ],
)
def test_source_count_packing_refuses_unrepresentable_integer_trials(
    count, literal, family, suffix
):
    """Reject original integer values before a float64 packing could round them."""
    trials = str(count) if literal else "trials"
    header = "" if literal else "(trials : Tensor[Int]([Row]))"
    source = f"""
object Row : FinSet 1
program counts{header} : Row -> Row
    observe y : Row <- {family}(total_count={trials}, {suffix})
    return y
export counts
"""
    with pytest.raises(ValueError, match=r"2\*\*53"):
        program = Compiler(parse(source)).compile().morphism
        observations = {"y": torch.tensor([0])}
        if not literal:
            observations["trials"] = torch.tensor([count])
        trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
