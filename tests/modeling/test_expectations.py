"""Source response expectations retain native laws and posterior dependence."""

import pytest
import torch

from quivers.diagnostics import prediction_grid
from quivers.inference import PriorDraws, posterior_expectations
from quivers.inference.predictive import Predictive
from quivers.modeling import Model, ModelFit, VariationalMethod
from tests.modeling.test_prediction import _fit, SOURCE


def test_source_expected_responses_preserve_chain_prefixes_and_native_means():
    fitted = _fit()
    actual = fitted.posterior_epred(num_samples=2)["y"]
    expected = fitted.samples["b"][:, :2, None] * fitted.model.data["x"]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 2, 3)
    new = fitted.posterior_epred({"x": torch.tensor([4.0, 5.0])}, num_samples=2)["y"]
    torch.testing.assert_close(
        new, fitted.samples["b"][:, :2, None] * torch.tensor([4.0, 5.0])
    )


def test_source_grid_keeps_covariance_and_empirical_reduction():
    fitted = _fit()
    grid = prediction_grid({"x": [0.0, 1.0, 2.0]}, conditions={})
    result = fitted.conditional_effects(grid)
    expected = (
        fitted.samples["b"][:, :, None, None]
        * torch.tensor([0.0, 1.0, 2.0])[None, None, :, None]
    )
    torch.testing.assert_close(result.draws, expected)
    assert result.kind == "mcmc"
    empirical = prediction_grid(
        {"x": [0.0, 1.0, 2.0]}, conditions={"x": [-5.0, 5.0]}, marginalize=True
    )
    torch.testing.assert_close(
        fitted.conditional_effects(empirical).draws, expected.squeeze(-1).double()
    )
    torch.testing.assert_close(
        result.analysis().covariance().covariance,
        torch.cov(expected.flatten(0, 1).flatten(1).double().T),
    )


def test_source_grid_binds_once_and_forwards_one_full_joint_request(monkeypatch):
    fitted = _fit()
    calls = []
    actual = type(fitted.model).rebind

    def counted(self, data, **kwargs):
        calls.append(data)
        return actual(self, data, **kwargs)

    monkeypatch.setattr(type(fitted.model), "rebind", counted)
    fitted.conditional_effects(prediction_grid({"x": [0.0, 1.0]}, conditions={}))
    assert len(calls) == 1 and len(calls[0]["x"]) == 2


def test_source_expected_values_use_no_response_replication_average(monkeypatch):
    fitted = _fit()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "native expectations must not average predictive responses"
        )

    monkeypatch.setattr(Predictive, "__call__", forbidden)
    torch.testing.assert_close(
        fitted.posterior_epred()["y"],
        fitted.samples["b"][:, :, None] * fitted.model.data["x"],
    )


def test_data_precision_and_variational_seed_are_retained():
    fitted = Model(SOURCE).fit(
        {
            "x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "y": torch.zeros(3, dtype=torch.float64),
        },
        method=VariationalMethod(num_steps=1, seed=8),
    )
    value = 0.1234567890123456
    grid = prediction_grid({"x": [value]}, conditions={})
    rng = torch.random.get_rng_state().clone()
    actual = fitted.conditional_effects(grid, num_samples=3, seed=12)
    draws = fitted.draws(num_samples=3, seed=12)
    torch.testing.assert_close(
        actual.draws,
        draws["b"][:, :, None, None].double() * value,
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual.kind == "variational" and actual.summary().convergence is None
    assert torch.equal(rng, torch.random.get_rng_state())


def test_same_extent_distinct_nominal_global_axis_is_not_replaced():
    source = (
        SOURCE.replace("object Out", "object Other:FinSet _\nobject Out")
        .replace(
            "y : Tensor[Real]([Row])",
            "y : Tensor[Real]([Row]), w:Tensor[Real]([Other])",
        )
        .replace("let mu = b * x", "let shift = sum(w)\n    let mu = b * x + shift")
    )
    bound = Model(source).bind(
        {"x": torch.ones(3), "y": torch.zeros(3), "w": torch.tensor([10.0, 20.0, 30.0])}
    )
    posterior = PriorDraws(
        samples={"b": torch.tensor([[0.0, 1.0, 2.0]])}, num_samples=3
    )
    fitted = ModelFit(model=bound, posterior=posterior)
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions={})
    actual = fitted.conditional_effects(grid)
    expected = (
        60
        + posterior.samples["b"][:, :, None, None]
        * torch.tensor([0.0, 1.0])[None, None, :, None]
    )
    torch.testing.assert_close(actual.draws, expected)
    assert bound.data["w"].shape == (3,)


def test_missing_row_condition_or_varying_scalar_fails_before_posterior_draws(
    monkeypatch,
):
    source = (
        SOURCE.replace("program regression(", "program regression(alpha:Real, ")
        .replace(
            "y : Tensor[Real]([Row])", "y : Tensor[Real]([Row]), z:Tensor[Real]([Row])"
        )
        .replace("b * x", "b * x + z + alpha")
    )
    bound = Model(source).bind(
        {"x": torch.ones(3), "y": torch.zeros(3), "z": torch.zeros(3), "alpha": 1.0}
    )
    fitted = ModelFit(
        model=bound,
        posterior=PriorDraws(samples={"b": torch.zeros(1, 2)}, num_samples=2),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid grid reached posterior")

    monkeypatch.setattr(type(fitted), "posterior_epred", forbidden)
    with pytest.raises(ValueError, match="explicit conditioning.*z"):
        fitted.conditional_effects(prediction_grid({"x": [0.0, 1.0]}, conditions={}))
    with pytest.raises(ValueError, match="scalar input.*cannot vary"):
        fitted.conditional_effects(
            prediction_grid({"alpha": [0.0, 1.0]}, conditions={"x": [0.0], "z": [0.0]})
        )


@pytest.mark.parametrize(
    "family,arguments",
    [
        ("Categorical", "logits=logits"),
        ("OrderedLogistic", "predictor=b,cutpoints=[0.0]"),
    ],
)
def test_nominal_and_ordinal_laws_return_probabilities(family, arguments):
    source = f"""object Row:FinSet _
object Out:Real 1
program p(y:Tensor[Int]([Row])):Row->Out
    sample b<-Normal(loc=0.0,scale=1.0)
    let logits = [0.0,b]
    observe y:Row<-{family}({arguments})
    return b
export p
"""
    bound = Model(source).bind({"y": torch.tensor([0, 1, 0])})
    b = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    fitted = ModelFit(
        model=bound, posterior=PriorDraws(samples={"b": b}, num_samples=4)
    )
    actual = fitted.posterior_epred()["y"]
    probability = torch.stack((torch.sigmoid(-b), torch.sigmoid(b)), -1)
    torch.testing.assert_close(actual, probability[:, :, None, :].expand(1, 4, 3, 2))


def test_vector_events_follow_checked_response_shape():
    source = """object Row:FinSet _
object Coordinate:FinSet 2
object Out:Real 1
program p(x:Tensor[Real]([Row,Coordinate]),y:Tensor[Real]([Row,Coordinate])):Row->Out
    sample b<-Normal(loc=0.0,scale=1.0)
    let mu=b*x
    observe y:Row<-MultivariateNormal(loc=mu,scale_tril=[[1.0,0.0],[0.0,1.0]]) [over=Coordinate]
    return b
export p
"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bound = Model(source).bind({"x": x, "y": torch.zeros_like(x)})
    b = torch.tensor([[0.0, 1.0, 2.0]])
    actual = ModelFit(
        model=bound, posterior=PriorDraws(samples={"b": b}, num_samples=3)
    ).posterior_epred()["y"]
    torch.testing.assert_close(actual, b[:, :, None, None] * x)


def test_upstream_observed_dependency_remains_conditioned_and_raw_scores_unchanged():
    source = SOURCE.replace(
        "    return b", "    observe z:Row <- Normal(loc=y,scale=1.0)\n    return b"
    ).replace(
        "y : Tensor[Real]([Row])", "y : Tensor[Real]([Row]),z:Tensor[Real]([Row])"
    )
    data = {
        "x": torch.tensor([1.0, 2.0, 3.0]),
        "y": torch.tensor([10.0, 20.0, 30.0]),
        "z": torch.zeros(3),
    }
    bound = Model(source).bind(data)
    b = torch.tensor([[0.5, 1.0]])
    fitted = ModelFit(
        model=bound, posterior=PriorDraws(samples={"b": b}, num_samples=2)
    )
    before = fitted.log_likelihood()
    actual = fitted.posterior_epred()
    torch.testing.assert_close(actual["y"], b[:, :, None] * data["x"])
    torch.testing.assert_close(actual["z"], data["y"].expand(1, 2, 3))
    after = fitted.log_likelihood()
    assert all(torch.equal(before[name], after[name]) for name in before)
    with pytest.raises(ValueError, match="select site"):
        fitted.conditional_effects(prediction_grid({"x": [0.0]}, conditions={}))


def test_generic_expectations_validate_budgets_and_sites():
    fitted = _fit()
    with pytest.raises(ValueError, match="one execution|one row|one run"):
        posterior_expectations(
            fitted.model.program, fitted.posterior, torch.zeros(2, 1), sites=("y",)
        )
    with pytest.raises(ValueError, match="distinct"):
        fitted.posterior_epred(sites=("y", "y"))
    with pytest.raises(ValueError, match="positive integer"):
        fitted.posterior_epred(mc_samples=0)
    with pytest.raises(TypeError, match="expose no native"):
        fitted.posterior_epred(sites=("missing",))


def test_scan_expectations_share_occurrence_order_and_exact_selector_precedence():
    from tests.dsl.test_scan_program_cells import STATE_SPACE, _linear, _EMISSION

    bound = Model(STATE_SPACE).bind(
        {"o": torch.zeros(3, 2)}, input=torch.zeros(1, 3, 2)
    )
    samples = {
        f"s@{position}": torch.full((1, 2, 1, 3), float(position))
        for position in range(3)
    }
    fitted = ModelFit(model=bound, posterior=PriorDraws(samples=samples, num_samples=2))
    weight, bias = _linear(bound.require_tensor_program(), _EMISSION)
    expected = torch.stack(
        [
            torch.nn.functional.linear(samples[f"s@{position}"][0, 0], weight, bias)[
                ..., :2
            ]
            for position in range(3)
        ]
    )
    all_sites = fitted.posterior_epred()["o"]
    torch.testing.assert_close(all_sites, expected.expand(1, 2, 3, 1, 2))
    selected = fitted.posterior_epred(sites=("o", "o@1"))
    torch.testing.assert_close(selected["o"], expected[[0, 2]].expand(1, 2, 2, 1, 2))
    torch.testing.assert_close(selected["o@1"], expected[1].expand(1, 2, 1, 2))
    before = fitted.log_likelihood()
    assert before["o"].shape[:3] == (1, 2, 3)
    after = fitted.log_likelihood()
    torch.testing.assert_close(before["o"], after["o"])


def test_lexically_computed_observation_uses_underlying_conditional_law():
    source = SOURCE.replace(", y : Tensor[Real]([Row])", "").replace(
        "    observe y", "    let y = 2.0 * x\n    observe y"
    )
    x = torch.tensor([1.0, 2.0, 3.0])
    bound = Model(source).bind({"x": x})
    b = torch.tensor([[0.5, 1.0]])
    fitted = ModelFit(
        model=bound, posterior=PriorDraws(samples={"b": b}, num_samples=2)
    )
    torch.testing.assert_close(fitted.posterior_epred()["y"], b[:, :, None] * x)
    result = fitted.conditional_effects(
        prediction_grid({"x": [4.0, 5.0]}, conditions={})
    )
    torch.testing.assert_close(
        result.draws,
        b[:, :, None, None] * torch.tensor([4.0, 5.0])[None, None, :, None],
    )


def test_recorded_event_expected_law_excludes_censoring_and_likelihood_weights():
    from tests.modeling.test_observation_events import SOURCE as EVENT_SOURCE, _data

    data = _data()
    bound = Model(EVENT_SOURCE).bind(data)
    b = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    fitted = ModelFit(
        model=bound, posterior=PriorDraws(samples={"b": b}, num_samples=2)
    )
    before = fitted.log_likelihood()
    torch.testing.assert_close(
        fitted.posterior_epred()["y"], b[:, :, None].expand(1, 2, 4)
    )
    assert torch.equal(before["y"], fitted.log_likelihood()["y"])


def test_explicit_finite_moment_budget_is_seeded_and_undefined_means_refused():
    source = SOURCE.replace("Normal(loc=mu, scale=1.0)", "Cauchy(loc=mu, scale=1.0)")
    fitted = _fit(source)
    with pytest.raises(ValueError, match="mean|moment|undefined"):
        fitted.posterior_epred()
    with pytest.raises(ValueError, match="mean|moment|finite"):
        fitted.posterior_epred(mc_samples=100)
    source = SOURCE.replace(
        "Normal(loc=mu, scale=1.0)",
        "Normal(loc=mu, scale=1.0) [trunc_lower=-0.5,trunc_upper=1.0]",
    )
    fitted = _fit(source)
    with pytest.raises(TypeError, match="mc_samples|analytic"):
        fitted.posterior_epred(num_samples=1)
    rng = torch.random.get_rng_state().clone()
    actual = fitted.posterior_epred(num_samples=1, mc_samples=6000, seed=17)["y"]
    mu = (fitted.samples["b"][:, :1, None] * fitted.model.data["x"]).double()
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    lower = -0.5 - mu
    upper = 1.0 - mu
    expected = mu + (
        torch.exp(normal.log_prob(lower)) - torch.exp(normal.log_prob(upper))
    ) / (normal.cdf(upper) - normal.cdf(lower))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.025)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(
        actual, fitted.posterior_epred(num_samples=1, mc_samples=6000, seed=17)["y"]
    )


def test_expected_and_predictive_responses_share_stray_posterior_diagnostics():
    fitted = _fit()
    bad = PriorDraws(samples={"not_a_site": torch.zeros(1, 2)}, num_samples=2)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="not_a_site.*not sample sites"):
        posterior_expectations(
            fitted.model.program,
            bad,
            fitted.model.input,
            fitted.model.tensor_data(),
            sites=("y",),
        )
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="not_a_site.*not sample sites"):
        Predictive(fitted.model.program, bad)(
            fitted.model.input, observations=fitted.model.tensor_data()
        )


def test_source_group_maps_keep_joint_group_draws_across_every_grid_point():
    source = """object Row : FinSet _
object Group : FinSet 2
object Out : Real 1
program p(x : Tensor[Real]([Row]), group_idx : Tensor[Int]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample intercept : Group <- Normal(loc=0.0, scale=1.0)
    let selected = intercept[group_idx]
    let mu = x + selected
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return intercept
export p
"""
    bound = Model(source).bind(
        {"x": torch.ones(3), "group_idx": torch.tensor([0, 1, 0]), "y": torch.zeros(3)}
    )
    intercept = torch.tensor([[[1.0, 4.0], [2.0, 6.0], [3.0, 8.0]]])
    fitted = ModelFit(
        model=bound,
        posterior=PriorDraws(samples={"intercept": intercept}, num_samples=3),
    )
    grid = prediction_grid({"x": [0.0, 10.0]}, conditions={"group_idx": [0, 1]})
    actual = fitted.conditional_effects(grid)
    expected = intercept[:, :, None, :] + torch.tensor([0.0, 10.0])[None, None, :, None]
    torch.testing.assert_close(actual.draws, expected)
    torch.testing.assert_close(
        actual.draws[:, :, 1] - actual.draws[:, :, 0], torch.full((1, 3, 2), 10.0)
    )
    with pytest.raises((ValueError, IndexError), match="bound|range|index"):
        fitted.conditional_effects(
            prediction_grid({"x": [0.0]}, conditions={"group_idx": [2]})
        )


def test_selected_source_response_accepts_explicit_upstream_observation_conditions():
    source = SOURCE.replace(
        "    return b", "    observe z : Row <- Normal(loc=y,scale=1.0)\n    return b"
    ).replace(
        "y : Tensor[Real]([Row])", "y : Tensor[Real]([Row]),z : Tensor[Real]([Row])"
    )
    bound = Model(source).bind(
        {"x": torch.ones(3), "y": torch.zeros(3), "z": torch.zeros(3)}
    )
    fitted = ModelFit(
        model=bound,
        posterior=PriorDraws(samples={"b": torch.tensor([[0.5, 1.0]])}, num_samples=2),
    )
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions={"y": [10.0, 20.0]})
    actual = fitted.conditional_effects(grid, site="z")
    torch.testing.assert_close(
        actual.draws, torch.tensor([10.0, 20.0]).expand(1, 2, 2, 2)
    )
    with pytest.raises(ValueError, match="not predictor inputs"):
        fitted.conditional_effects(
            prediction_grid({"z": [0.0, 1.0]}, conditions={"x": [0.0], "y": [1.0]}),
            site="z",
        )
