"""Power likelihoods preserve priors, raw densities, and predictive laws."""

import itertools
import math

import pytest
import torch

from quivers.effects import subsample
from quivers.inference import MCMCResult
from quivers.modeling import Model, ModelFit


SOURCE = """object Row : FinSet 3
object Out : Real 1
program regression(w : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=b, scale=1.0) [weights=w]
    return b
export regression
"""


def _normal(value, mean=0.0):
    return -0.5 * ((value - mean).square() + math.log(2 * math.pi))


@pytest.mark.parametrize("weights", [[0.0, 1.0, 2.0], [1.0] * 3, [0.0] * 3])
def test_weighted_native_density_and_latent_gradient_have_closed_form_oracles(weights):
    y = torch.tensor([-0.5, 0.2, 1.7])
    w = torch.tensor(weights)
    bound = Model(SOURCE).bind({"y": y, "w": w})
    b = torch.tensor([0.3], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint.sum()
    expected = _normal(b).sum() + (w * _normal(y, b)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
    )


def test_positive_weight_gradients_preserve_live_input_identity():
    y = torch.tensor([-0.5, 0.2, 1.7])
    w = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
    bound = Model(SOURCE).bind({"y": y, "w": w})
    assert bound.data["w"] is w
    actual = bound.trace({"b": torch.tensor([0.3])}).log_joint.sum()
    torch.testing.assert_close(torch.autograd.grad(actual, w)[0], _normal(y, 0.3))


@pytest.mark.parametrize("value", [0.0, 0.5, 2.0])
def test_declared_scalar_weight_broadcasts_across_observation_rows(value):
    source = SOURCE.replace("w : Tensor[Real]([Row])", "w : Real")
    y = torch.tensor([-0.5, 0.2, 1.7])
    bound = Model(source).bind({"y": y, "w": value})
    b = torch.tensor([0.3])
    torch.testing.assert_close(
        bound.trace({"b": b}).log_joint.sum(),
        _normal(b).sum() + value * _normal(y, b).sum(),
    )


def test_zero_weight_excludes_an_impossible_density_before_family_gradients():
    source = SOURCE.replace(
        "observe y : Row <- Normal(loc=b, scale=1.0)",
        "let rate = exp(b)\n    observe y : Row <- Exponential(rate=rate)",
    )
    bound = Model(source).bind(
        {"y": torch.tensor([-1.0, 2.0, 3.0]), "w": torch.tensor([0.0, 1.0, 2.0])}
    )
    b = torch.tensor([0.0], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint.sum()
    expected = _normal(b).sum() + 3 * b.sum() - 8 * b.exp().sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
    )


def test_zero_weight_prevents_upstream_overflow_in_location_and_scale_gradients():
    source = SOURCE.replace(
        "observe y : Row <- Normal(loc=b, scale=1.0)",
        "let sigma = exp(b)\n    observe y : Row <- Normal(loc=b, scale=sigma)",
    )
    y = torch.tensor([1e20, 0.2, 0.8])
    w = torch.tensor([0.0, 1.0, 2.0])
    bound = Model(source).bind({"y": y, "w": w})
    b = torch.tensor([0.0], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint.sum()
    kept = y[1:]
    raw = -0.5 * ((kept - b) / b.exp()).square() - b - 0.5 * math.log(2 * math.pi)
    expected = _normal(b).sum() + (w[1:] * raw).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
    )


def test_zero_weight_beta_endpoint_has_finite_concentration_gradient():
    source = SOURCE.replace(
        "observe y : Row <- Normal(loc=b, scale=1.0)",
        "let alpha = exp(b) + 2.0\n    observe y : Row <- Beta(concentration1=alpha, concentration0=2.0)",
    )
    y = torch.tensor([0.0, 0.2, 0.8])
    w = torch.tensor([0.0, 1.0, 2.0])
    bound = Model(source).bind({"y": y, "w": w})
    b = torch.tensor([0.0], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint.sum()
    alpha = b.exp() + 2.0
    raw = (
        (alpha + 2).lgamma()
        - alpha.lgamma()
        + (alpha - 1) * y[1:].log()
        + torch.log1p(-y[1:])
    )
    expected = _normal(b).sum() + (w[1:] * raw).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
    )


@pytest.mark.parametrize("already_kept", [False, True])
def test_subsample_average_is_the_full_weighted_objective_and_gradient(already_kept):
    bound = Model(SOURCE).bind(
        {"y": torch.tensor([-0.5, 0.2, 1.7]), "w": torch.tensor([0.0, 1.0, 2.0])}
    )
    b = torch.tensor([0.3], requires_grad=True)
    full = bound.trace({"b": b}).log_joint.sum()
    scores, gradients = [], []
    from quivers.inference.trace import trace

    for subset in itertools.combinations(range(3), 2):
        rows = torch.tensor(subset)
        data = bound.tensor_data()
        if already_kept:
            data = {**data, "y": data["y"].index_select(0, rows)}
        with subsample("Row", rows):
            score = trace(bound.program, bound.input, {**data, "b": b}).log_joint.sum()
        scores.append(score)
        gradients.append(torch.autograd.grad(score, b, retain_graph=True)[0])
    torch.testing.assert_close(torch.stack(scores).mean(), full)
    torch.testing.assert_close(
        torch.stack(gradients).mean(0), torch.autograd.grad(full, b)[0]
    )


def test_raw_pointwise_likelihood_and_predictive_draws_ignore_likelihood_weights():
    y = torch.tensor([-0.5, 0.2, 1.7])
    bound = Model(SOURCE).bind({"y": y, "w": torch.tensor([0.0, 1.0, 2.0])})
    posterior = MCMCResult(
        samples={"b": torch.tensor([[0.0, 0.5], [-0.5, 1.0]])},
        log_densities=torch.zeros(2, 2),
        acceptance_rates=torch.ones(2),
        divergence_counts=torch.zeros(2, dtype=torch.long),
        num_warmup=0,
        num_samples=2,
    )
    fit = ModelFit(model=bound, posterior=posterior)
    expected = _normal(y, fit.samples["b"][..., None])
    torch.testing.assert_close(fit.log_likelihood()["y"], expected)
    first = fit.predict({"w": torch.zeros(3)}, num_samples=4, seed=19)
    second = fit.predict({"w": torch.ones(3)}, num_samples=4, seed=19)
    torch.testing.assert_close(first["y"], second["y"], rtol=0, atol=0)


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_native_weight_values_fail_at_execution(bad):
    bound = Model(SOURCE).bind(
        {"y": torch.zeros(3), "w": torch.tensor([1.0, bad, 0.0])}
    )
    with pytest.raises((ValueError, RuntimeError), match="finite and nonnegative"):
        bound.trace({"b": torch.zeros(1)})


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("weights", [[0.0, 2.0, 0.5], [1.0, 2.0, 0.5], [0.0, 0.0, 0.0]])
def test_marginal_weights_precede_latent_sum_and_preserve_class_prior(grouped, weights):
    group = " [over=Group]" if grouped else ""
    index = " : Class" if grouped else ""
    via = "via=group_idx, " if grouped else ""
    source = f"""object Class : FinSet 2
object Group : FinSet 2
object Row : FinSet 3
object Out : Real 1
program main(mu : Tensor[Real]([Class]), pi : Tensor[Real]([Class])) : Row -> Out
    marginalize cls{index} <- Categorical(probs=pi){group}
        observe y : Row <- Normal(loc=mu[cls], scale=1.0) [{via}weights=w]
    return mu
export main
"""
    mu = torch.tensor([0.0, 3.0], requires_grad=True)
    pi = torch.tensor([0.25, 0.75], requires_grad=True)
    y = torch.tensor([1.0, 2.0, 4.0])
    w = torch.tensor(weights)
    data = {"mu": mu, "pi": pi, "y": y, "w": w}
    if grouped:
        data["group_idx"] = torch.tensor([0, 0, 1])
    bound = Model(source).bind(data)
    actual = bound.trace().log_joint.sum()
    row_scores = w[:, None] * _normal(y[:, None], mu)
    groups = [torch.tensor([0, 1]), torch.tensor([2])] if grouped else [torch.arange(3)]
    expected = sum(
        torch.logsumexp(
            pi.log() - pi.sum().log() + row_scores.index_select(0, rows).sum(0), 0
        )
        for rows in groups
    )
    torch.testing.assert_close(actual, expected)
    for derivative, oracle in zip(
        torch.autograd.grad(actual, (mu, pi)),
        torch.autograd.grad(expected, (mu, pi)),
        strict=True,
    ):
        torch.testing.assert_close(derivative, oracle)


def test_array_events_reduce_before_per_row_likelihood_weights():
    source = (
        SOURCE.replace("object Out", "object Copy : FinSet 2\nobject Out")
        .replace("y : Tensor[Real]([Row])", "y : Tensor[Real]([Row, Copy])")
        .replace("[weights=w]", "[over=Copy, weights=w]")
    )
    y = torch.tensor([[-1.0, 1.0], [0.0, 2.0], [-0.5, 0.5]])
    w = torch.tensor([0.0, 1.0, 2.0])
    bound = Model(source).bind({"y": y, "w": w})
    b = torch.tensor([0.3], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint.sum()
    expected = _normal(b).sum() + (w * _normal(y, b).sum(-1)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
    )


@pytest.mark.parametrize("actual", ["2.0", "importance"])
def test_native_template_weight_substitution_is_capture_avoiding(actual):
    source = f"""object Row : FinSet 3
object Out : Real 1
program sub[A : FinSet](y : Tensor[Real]([A]), weight : Real) : Row -> Out
    let w = weight + 1.0
    observe y : A <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y
program main : Row -> Out
    let w = 19.0
    sample z <- sub[Row](response, {actual})
    return z
export main
"""
    y = torch.tensor([-1.0, 0.5, 2.0])
    data = {"response": y}
    if actual == "importance":
        data["importance"] = torch.tensor(2.0)
    joint = Model(source).bind(data).trace().log_joint.sum()
    torch.testing.assert_close(joint, 3 * _normal(y).sum())


def test_native_scalar_unplated_weight_preserves_event_reduction():
    source = """object Unit : Real 1
object Response : Real 2
morphism emission : Unit -> Response ~ Normal
program main : Unit -> Response
    observe y <- emission [weights=0.0]
    return y
export main
"""
    from quivers.dsl import loads

    program = loads(source).morphism
    value = program.log_joint(torch.zeros(1, 1), {"y": torch.full((1, 2), 1e20)})
    torch.testing.assert_close(value, torch.zeros_like(value))
    gradients = torch.autograd.grad(value.sum(), tuple(program.parameters()))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_scan_observation_weights_are_captured_without_weighting_transitions(weight):
    from quivers.dsl import loads

    source = """object Input : Real 1
object State : Real 1
morphism transition : Input * State -> State ~ Normal
program cell(x_t, h_prev) : Input * State -> State
    sample state <- transition(x_t, h_prev)
    observe o <- Normal(loc=state, scale=1.0) [weights=importance]
    return state
define run_all = scan(cell)
program main(importance : Real) : Input -> State
    sample h <- run_all
    return h
export main
"""
    program = loads(source).morphism
    with torch.no_grad():
        for parameter in program.parameters():
            parameter.zero_()
    states = torch.tensor([[-0.5, 0.2, 1.0], [0.5, -0.2, -1.0]], requires_grad=True)
    y = torch.tensor([[[1.0], [2.0], [-0.5]], [[-1.0], [-2.0], [0.5]]])
    given = {f"state@{t}": states[:, t : t + 1] for t in range(3)}
    joint = program.log_joint(
        torch.zeros(2, 3, 1), {**given, "o": y, "importance": torch.tensor(weight)}
    )
    expected = (_normal(states) + weight * _normal(y.squeeze(-1), states)).sum(-1)
    torch.testing.assert_close(joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(joint.sum(), states)[0],
        torch.autograd.grad(expected.sum(), states)[0],
    )
