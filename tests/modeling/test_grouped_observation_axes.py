"""Grouped likelihood axes follow lexical provenance rather than equal extents."""

import pytest
import torch

from quivers.modeling import Model
from quivers.dsl import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program


def _reference_value(value):
    return (
        tuple(_reference_value(item) for item in value) if value.ndim else value.item()
    )


@pytest.mark.parametrize("dependent", ["location", "scale", "neither"])
@pytest.mark.parametrize("through_lets", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("weights", [None, [1.0, 2.0], [0.0, 1.0], [0.0, 0.0]])
def test_equal_row_and_class_extents_preserve_parameter_roles(
    dependent, through_lets, reverse, weights
):
    declarations = [
        "class_mu : Tensor[Real]([Class])",
        "class_sigma : Tensor[Real]([Class])",
        "row_mu : Tensor[Real]([Row])",
        "row_sigma : Tensor[Real]([Row])",
        "pi : Tensor[Real]([Class])",
    ]
    if reverse:
        declarations.reverse()
    loc = "class_mu[cls]" if dependent == "location" else "row_mu"
    scale = "class_sigma[cls]" if dependent == "scale" else "row_sigma"
    bindings = ""
    if through_lets:
        bindings = (
            f"        let selected_mu = {loc}\n"
            "        let shifted_mu = selected_mu + 0.25\n"
            f"        let selected_sigma = {scale}\n"
            "        let shifted_sigma = selected_sigma * 1.25\n"
        )
        loc, scale = "shifted_mu", "shifted_sigma"
    arguments = [f"loc={loc}", f"scale={scale}"]
    if reverse:
        arguments.reverse()
    options = "via=group_idx" + (", weights=w" if weights is not None else "")
    source = f"""object Class : FinSet 2
object Row : FinSet 2
object Group : FinSet 2
object Out : Real 1
program main({", ".join(declarations)}) : Row -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
{bindings}        observe y : Row <- Normal({", ".join(arguments)}) [{options}]
    return pi
export main
"""
    dtype = torch.float64
    class_mu = torch.tensor([0.0, 3.0], dtype=dtype, requires_grad=True)
    class_sigma = torch.tensor([0.4, 1.5], dtype=dtype, requires_grad=True)
    row_mu = torch.tensor([-0.2, 0.8], dtype=dtype, requires_grad=True)
    row_sigma = torch.tensor([0.2, 2.0], dtype=dtype, requires_grad=True)
    pi = torch.tensor([0.25, 0.75], dtype=dtype, requires_grad=True)
    y = torch.tensor([1.0, 2.0], dtype=dtype)
    data = {
        "class_mu": class_mu,
        "class_sigma": class_sigma,
        "row_mu": row_mu,
        "row_sigma": row_sigma,
        "pi": pi,
        "y": y,
        "group_idx": torch.tensor([0, 1]),
    }
    if weights is not None:
        data["w"] = torch.tensor(weights, dtype=dtype)
    actual = Model(source).bind(data).trace().log_joint.sum()
    mean = class_mu[None, :] if dependent == "location" else row_mu[:, None]
    sigma = class_sigma[None, :] if dependent == "scale" else row_sigma[:, None]
    if through_lets:
        mean, sigma = mean + 0.25, sigma * 1.25
    ll = torch.distributions.Normal(mean, sigma).log_prob(y[:, None]).expand(2, 2)
    if weights is not None:
        ll = ll * data["w"][:, None]
    expected = torch.logsumexp(pi.log() - pi.sum().log() + ll, -1).sum()
    torch.testing.assert_close(actual, expected)
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)
    active = (
        (class_mu, row_sigma, pi)
        if dependent == "location"
        else (row_mu, class_sigma, pi)
        if dependent == "scale"
        else (row_mu, row_sigma, pi)
    )
    for gradient, oracle in zip(
        torch.autograd.grad(actual, active),
        torch.autograd.grad(expected, active),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle)


@pytest.mark.parametrize("kept_parameters", [False, True])
@pytest.mark.parametrize("kept_responses", [False, True])
def test_zero_weights_preserve_independent_full_or_kept_parameter_response_layouts(
    kept_parameters, kept_responses
):
    from quivers.effects._observation_weights import WeightedObservation

    source = """object Row : FinSet 3
object Out : Real 1
program main : Row -> Out
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return y
export main
"""
    y = torch.tensor([-0.5, 0.2, 1.7])
    parameters = torch.tensor([[-0.5], [0.2], [0.3]], requires_grad=True)
    bound = Model(source).bind({"y": y, "mu": parameters.detach().squeeze(-1)})
    sampleable = bound.trace().sites["y"].sampleable
    assert sampleable is not None
    inner = sampleable.raw_morphism
    rows = torch.tensor([0, 2])
    x = parameters.index_select(0, rows) if kept_parameters else parameters
    response = y.index_select(0, rows) if kept_responses else y
    weighted = WeightedObservation(
        inner, sampleable.declared, torch.tensor([0.0, 1.0, 2.0]), x
    )
    actual = weighted.log_prob_rows(x, response, rows)
    expected = 2 * torch.distributions.Normal(parameters[2, 0], 1.0).log_prob(y[2])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, parameters)[0],
        torch.autograd.grad(expected, parameters)[0],
    )


@pytest.mark.parametrize("rows, classes", [(2, 2), (5, 2), (2, 3), (5, 3)])
@pytest.mark.parametrize("weight_mode", ["unweighted", "positive", "masked", "zero"])
def test_mixed_class_group_let_subgraph_matches_joint_and_derivative_oracles(
    rows, classes, weight_mode
):
    option = "via=group_idx" + ("" if weight_mode == "unweighted" else ", weights=w")
    source = f"""object Class : FinSet {classes}
object Row : FinSet {rows}
object Group : FinSet 3
object Out : Real 1
program main(mu : Tensor[Real]([Class]), log_sigma : Tensor[Real]([Class]), x : Tensor[Real]([Group]), z : Tensor[Real]([Group]), pi : Tensor[Real]([Class]), shift : Real) : Row -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
        let class_term = mu[cls] + shift
        let row_term = x * shift
        let center = class_term + row_term
        let adjusted = center + class_term
        let alias = adjusted
        let log_scale = log_sigma[cls] + z
        let scale = exp(log_scale)
        observe y : Row <- Normal(loc=alias, scale=scale) [{option}]
    return pi
export main
"""
    mu = torch.linspace(-0.5, 1.0, classes, requires_grad=True)
    log_sigma = torch.linspace(-0.5, 0.5, classes, requires_grad=True)
    x = torch.linspace(-0.3, 0.8, 3, requires_grad=True)
    z = torch.linspace(-0.2, 0.3, 3, requires_grad=True)
    shift = torch.tensor(0.25, requires_grad=True)
    pi = torch.arange(1.0, classes + 1, requires_grad=True)
    y = torch.linspace(0.1, 1.2, rows)
    group = (torch.arange(rows) * 2 + 1) % 3
    weights = torch.linspace(0.5, 1.5, rows)
    if weight_mode == "masked":
        weights[::2] = 0
    elif weight_mode == "zero":
        weights.zero_()
    data = {
        "mu": mu,
        "log_sigma": log_sigma,
        "x": x,
        "z": z,
        "pi": pi,
        "shift": shift,
        "y": y,
        "group_idx": group,
    }
    if weight_mode != "unweighted":
        data["w"] = weights
    actual = Model(source).bind(data).trace().log_joint.sum()
    mean = 2 * (mu[None, :] + shift) + x[group, None] * shift
    sigma = (log_sigma[None, :] + z[group, None]).exp()
    ll = torch.distributions.Normal(mean, sigma).log_prob(y[:, None])
    if weight_mode != "unweighted":
        ll = ll * weights[:, None]
    expected = sum(
        torch.logsumexp(pi.log() - pi.sum().log() + ll[group == g].sum(0), 0)
        for g in range(3)
    )
    torch.testing.assert_close(actual, expected)
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)
    variables = (mu, log_sigma, x, z, shift, pi)
    for gradient, oracle in zip(
        torch.autograd.grad(actual, variables),
        torch.autograd.grad(expected, variables),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle, rtol=1e-5, atol=1e-6)


def test_zero_weight_score_connects_parameter_handler_override_values():
    from quivers.continuous.families import ConditionalNormal
    from quivers.continuous.spaces import Euclidean
    from quivers.effects._observation_weights import WeightedObservation
    from quivers.effects.interpreter import _ParameterizedMorphism

    family = ConditionalNormal(Euclidean(name="X", dim=1), Euclidean(name="Y", dim=1))
    name, original = next(iter(family.named_parameters()))
    replacement = torch.ones_like(original, requires_grad=True)
    parameterized = _ParameterizedMorphism(family, {name: replacement})
    x = torch.zeros(2, 1)
    weighted = WeightedObservation(parameterized, family, 0.0, x)
    score = weighted.log_prob(x, torch.ones(2, 1))
    torch.testing.assert_close(
        torch.autograd.grad(score, replacement)[0], torch.zeros_like(replacement)
    )


def test_mixed_let_axes_stay_distinct_across_observation_plates_and_grouped_blocks():
    source = """object ClassA : FinSet 2
object ClassB : FinSet 3
object RowA : FinSet 3
object RowB : FinSet 5
object Group : FinSet 2
object Out : Real 1
program main(mu_a : Tensor[Real]([ClassA]), mu_b : Tensor[Real]([ClassB]), x_a : Tensor[Real]([Group]), x_b : Tensor[Real]([Group]), pi_a : Tensor[Real]([ClassA]), pi_b : Tensor[Real]([ClassB])) : RowA -> Out
    marginalize a : ClassA <- Categorical(probs=pi_a) [over=Group]
        let center_a = mu_a[a] + x_a
        observe y_a : RowA <- Normal(loc=center_a, scale=1.0) [via=idx_a, weights=w_a]
        let center_b = mu_a[a] + x_b
        observe y_b : RowB <- Normal(loc=center_b, scale=1.0) [via=idx_b, weights=w_b]
    marginalize b : ClassB <- Categorical(probs=pi_b) [over=Group]
        let center_c = mu_b[b] + x_b
        observe y_c : RowB <- Normal(loc=center_c, scale=1.0) [via=idx_b, weights=w_b]
    return pi_a
export main
"""
    data = {
        "mu_a": torch.tensor([-0.5, 0.7], requires_grad=True),
        "mu_b": torch.tensor([-0.2, 0.3, 1.1], requires_grad=True),
        "x_a": torch.tensor([-0.3, 0.8]),
        "x_b": torch.tensor([-0.1, 0.5]),
        "pi_a": torch.tensor([0.4, 0.6], requires_grad=True),
        "pi_b": torch.tensor([0.2, 0.3, 0.5], requires_grad=True),
        "y_a": torch.tensor([0.1, -0.2, 0.8]),
        "y_b": torch.linspace(0.2, 0.9, 5),
        "y_c": torch.linspace(-0.2, 0.5, 5),
        "idx_a": torch.tensor([0, 0, 1]),
        "idx_b": torch.tensor([1, 0, 1, 0, 1]),
        "w_a": torch.tensor([0.0, 1.0, 2.0]),
        "w_b": torch.tensor([1.0, 0.0, 0.5, 2.0, 1.0]),
    }
    actual = Model(source).bind(data).trace().log_joint.sum()

    def rows(response, covariate, mu, weight):
        return weight[:, None] * torch.distributions.Normal(
            covariate[:, None] + mu[None, :], 1.0
        ).log_prob(response[:, None])

    ll_a = rows(data["y_a"], data["x_a"][data["idx_a"]], data["mu_a"], data["w_a"])
    ll_b = rows(data["y_b"], data["x_b"][data["idx_b"]], data["mu_a"], data["w_b"])
    ll_c = rows(data["y_c"], data["x_b"][data["idx_b"]], data["mu_b"], data["w_b"])
    expected = sum(
        torch.logsumexp(
            data["pi_a"].log()
            + ll_a[data["idx_a"] == g].sum(0)
            + ll_b[data["idx_b"] == g].sum(0),
            0,
        )
        + torch.logsumexp(data["pi_b"].log() + ll_c[data["idx_b"] == g].sum(0), 0)
        for g in range(2)
    )
    # Normalize both live categorical parameterizations in the oracle.
    expected = expected - 2 * data["pi_a"].sum().log() - 2 * data["pi_b"].sum().log()
    torch.testing.assert_close(actual, expected)
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=2e-6)
    parameters = tuple(data[name] for name in ("mu_a", "mu_b", "pi_a", "pi_b"))
    for gradient, oracle in zip(
        torch.autograd.grad(actual, parameters),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle, rtol=1e-5, atol=1e-6)


def test_classwise_let_keeps_groups_batched_and_replays_only_mixed_subgraphs(
    monkeypatch,
):
    import quivers.dsl.compiler.programs as programs

    original = programs._classwise_let
    wrappers, calls = [], []

    def instrument(function, classes, sizes, input_classes):
        wrappers.append(classes)

        def counted(env):
            calls.append(
                tuple(max(env.values(), key=lambda value: value.numel()).shape)
            )
            return function(env)

        return original(counted, classes, sizes, input_classes)

    monkeypatch.setattr(programs, "_classwise_let", instrument)
    source = """object Class : FinSet 12
object Row : FinSet 1200
object Group : FinSet 1200
object Out : Real 1
program main(mu : Tensor[Real]([Class]), x : Tensor[Real]([Group]), pi : Tensor[Real]([Class])) : Row -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
        let class_only = mu[cls] + 0.25
        let row_only = x * 2.0
        let mixed = class_only + row_only
        let mixed_again = exp(mixed)
        observe y : Row <- Normal(loc=mixed_again, scale=1.0) [via=idx]
    return pi
export main
"""
    data = {
        "mu": torch.linspace(-0.5, 0.5, 12),
        "x": torch.zeros(1200),
        "pi": torch.ones(12),
        "y": torch.ones(1200),
        "idx": torch.arange(1200) % 2,
    }
    actual = Model(source).bind(data).trace().log_joint
    assert torch.isfinite(actual).all()
    assert wrappers == [("cls",), ("cls",)]
    assert calls == [(1200,)] * 24


@pytest.mark.parametrize("rows", [2, 5])
@pytest.mark.parametrize("declare_x", [False, True])
def test_group_data_pullback_with_inferred_static_nat_rows(rows, declare_x):
    source = f"""object Class : FinSet 2
object Group : FinSet 3
object Out : Real 1
program main[n : Nat](mu : Tensor[Real]([Class]), pi : Tensor[Real]([Class]){", x : Tensor[Real]([Group])" if declare_x else ""}) : n -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
        let center = mu[cls] + x
        observe y : n <- Normal(loc=center, scale=1.0) [via=group_idx]
    return pi
export main
"""
    mu = torch.tensor([-0.5, 0.7], requires_grad=True)
    pi = torch.tensor([0.4, 0.6], requires_grad=True)
    x = torch.tensor([-0.2, 0.3, 0.8], requires_grad=True)
    y = torch.linspace(0.1, 0.9, rows)
    index = torch.tensor([2, 0, 2, 1, 0])[:rows]
    data = {"mu": mu, "pi": pi, "x": x, "y": y, "group_idx": index}
    if not declare_x:
        # The free datum is inferred on the response axis; a group-valued
        # covariate needs the explicit declaration when those axes differ.
        with pytest.raises(QiecDiagnosticError, match="tensors of differing shapes"):
            Model(source).bind(data)
        return
    bound = Model(source).bind(data)
    actual = bound.trace().log_joint.sum()
    ll = torch.distributions.Normal(mu[None, :] + x[index, None], 1.0).log_prob(
        y[:, None]
    )
    expected = (
        sum(
            torch.logsumexp(pi.log() + ll[index == group].sum(0), 0)
            for group in range(3)
        )
        - 3 * pi.sum().log()
    )
    torch.testing.assert_close(actual, expected)
    for gradient, oracle in zip(
        torch.autograd.grad(actual, (mu, pi, x)),
        torch.autograd.grad(expected, (mu, pi, x)),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rows", [2, 5])
def test_scalar_covariate_with_static_nat_rows_keeps_class_only_vectorization(
    rows, monkeypatch
):
    import quivers.dsl.compiler.programs as programs

    def reject_class_replay(*args, **kwargs):
        raise AssertionError("a scalar covariate must not require per-class let replay")

    monkeypatch.setattr(programs, "_classwise_let", reject_class_replay)
    source = """object Class : FinSet 2
object Group : FinSet 3
object Out : Real 1
program main[n : Nat](mu : Tensor[Real]([Class]), pi : Tensor[Real]([Class]), x : Real) : n -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
        let center = mu[cls] + x
        observe y : n <- Normal(loc=center, scale=1.0) [via=group_idx]
    return pi
export main
"""
    mu = torch.tensor([-0.5, 0.7], requires_grad=True)
    pi = torch.tensor([0.4, 0.6], requires_grad=True)
    x = torch.tensor(0.3, requires_grad=True)
    y = torch.linspace(0.1, 0.9, rows)
    index = torch.tensor([2, 0, 2, 1, 0])[:rows]
    actual = (
        Model(source)
        .bind({"mu": mu, "pi": pi, "x": x, "y": y, "group_idx": index})
        .trace()
        .log_joint.sum()
    )
    ll = torch.distributions.Normal(mu[None, :] + x, 1.0).log_prob(y[:, None])
    expected = (
        sum(
            torch.logsumexp(pi.log() + ll[index == group].sum(0), 0)
            for group in range(3)
        )
        - 3 * pi.sum().log()
    )
    torch.testing.assert_close(actual, expected)
    for gradient, oracle in zip(
        torch.autograd.grad(actual, (mu, pi, x)),
        torch.autograd.grad(expected, (mu, pi, x)),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("class_sizes", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize("distinct_groups", [False, True])
@pytest.mark.parametrize("weight_mode", ["unweighted", "positive", "masked", "zero"])
def test_nested_joint_class_axes_preserve_density_and_all_parameter_gradients(
    class_sizes, distinct_groups, weight_mode
):
    depth = len(class_sizes)
    groups = [f"Group{i}" if distinct_groups else "Group" for i in range(depth)]
    declarations = [
        f"object Class{i} : FinSet {size}" for i, size in enumerate(class_sizes)
    ]
    declarations += [f"object {name} : FinSet 3" for name in dict.fromkeys(groups)]
    declarations += ["object Row : FinSet 7", "object Out : Real 1"]
    parameters = [
        item
        for i in range(depth)
        for item in (
            f"mu{i} : Tensor[Real]([Class{i}])",
            f"pi{i} : Tensor[Real]([Class{i}])",
        )
    ] + [f"x : Tensor[Real]([{groups[-1]}])", "shift : Real"]
    lines = [*declarations, f"program main({', '.join(parameters)}) : Row -> Out"]
    for i in range(depth):
        lines.append(
            f"{'    ' * (i + 1)}marginalize c{i} : Class{i} <- Categorical(probs=pi{i}) [over={groups[i]}]"
        )
    indent = "    " * (depth + 1)
    lines += [
        f"{indent}let joint_class = {' + '.join(f'mu{i}[c{i}]' for i in range(depth))}",
        f"{indent}let center = joint_class + x",
        f"{indent}let shifted = center + shift",
        f"{indent}let alias = shifted",
        f"{indent}observe y : Row <- Normal(loc=alias, scale=0.75) [via=idx{'' if weight_mode == 'unweighted' else ', weights=w'}]",
        "    return pi0",
        "export main",
    ]
    source = "\n".join(lines) + "\n"
    dtype = torch.float64
    means = [
        torch.linspace(-0.6, 0.8, size, dtype=dtype).requires_grad_()
        for size in class_sizes
    ]
    priors = [
        torch.linspace(0.2, 1.1, size, dtype=dtype).requires_grad_()
        for size in class_sizes
    ]
    x = torch.tensor([-0.2, 0.3, 0.8], dtype=dtype, requires_grad=True)
    shift = torch.tensor(0.125, dtype=dtype, requires_grad=True)
    y = torch.tensor([0.1, 0.3, -0.7, 0.7, 0.9, -0.1, 0.4], dtype=dtype)
    idx = torch.tensor([2, 0, 2, 1, 0, 2, 1])
    data = {
        **{f"mu{i}": value for i, value in enumerate(means)},
        **{f"pi{i}": value for i, value in enumerate(priors)},
        "x": x,
        "shift": shift,
        "y": y,
        "idx": idx,
    }
    weight = torch.ones(7, dtype=dtype)
    if weight_mode == "positive":
        weight = torch.tensor([0.5, 1.0, 2.0, 1.5, 0.25, 2.0, 1.0], dtype=dtype)
    elif weight_mode == "masked":
        weight = torch.tensor([0.0, 1.0, 0.0, 1.5, 0.25, 0.0, 1.0], dtype=dtype)
    elif weight_mode == "zero":
        weight = torch.zeros(7, dtype=dtype)
    if weight_mode != "unweighted":
        data["w"] = weight
    actual = Model(source).bind(data).trace().log_joint.sum()
    class_mean = sum(
        value.reshape(
            *(size if axis == i else 1 for axis, size in enumerate(class_sizes))
        )
        for i, value in enumerate(means)
    )
    mean = class_mean.unsqueeze(0) + x[idx].reshape(-1, *((1,) * depth)) + shift
    row_scores = torch.distributions.Normal(mean, 0.75).log_prob(
        y.reshape(-1, *((1,) * depth))
    ) * weight.reshape(-1, *((1,) * depth))
    grouped = row_scores.new_zeros(3, *class_sizes).index_add(0, idx, row_scores)
    for i, prior in enumerate(priors):
        grouped = grouped + (prior.log() - prior.sum().log()).reshape(
            1, *(size if axis == i else 1 for axis, size in enumerate(class_sizes))
        )
    expected = torch.logsumexp(grouped.flatten(1), -1).sum()
    torch.testing.assert_close(actual, expected)
    parameters = (*means, *priors, x, shift)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, parameters),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-12, rtol=2e-12
        )
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-10)


@pytest.mark.parametrize("group_layout", ["shared", "renamed", "product"])
@pytest.mark.parametrize("weighted", [False, True])
def test_nested_distinct_class_parameters_follow_checked_group_contract(
    group_layout, weighted
):
    outer_size = 2 if group_layout == "product" else 3
    inner = {"shared": "Outer", "renamed": "Other", "product": "[Outer, Other]"}[
        group_layout
    ]
    via = "[outer_idx, inner_idx]" if group_layout == "product" else "inner_idx"
    source = f"""object A : FinSet 2
object B : FinSet 4
object Outer : FinSet {outer_size}
object Other : FinSet 3
object Row : FinSet 7
object Out : Real 1
program main(mu : Tensor[Real]([A]), sigma : Tensor[Real]([B]), pa : Tensor[Real]([A]), pb : Tensor[Real]([B])) : Row -> Out
    marginalize a : A <- Categorical(probs=pa) [over=Outer]
        marginalize b : B <- Categorical(probs=pb) [over={inner}]
            observe y : Row <- Normal(loc=mu[a], scale=sigma[b]) [via={via}{", weights=w" if weighted else ""}]
    return pa
export main
"""
    if group_layout == "product":
        with pytest.raises(
            QiecDiagnosticError, match="leading dimensions follow the plate"
        ) as error:
            lower_qvr_to_qiec(parse(source))
        assert error.value.line > 0
        return
    dtype = torch.float64
    mu = torch.tensor([-0.5, 0.7], dtype=dtype, requires_grad=True)
    sigma = torch.tensor([0.3, 0.6, 1.0, 1.5], dtype=dtype, requires_grad=True)
    pa = torch.tensor([0.3, 0.9], dtype=dtype, requires_grad=True)
    pb = torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=dtype, requires_grad=True)
    y = torch.tensor([-0.7, 0.1, 0.3, 0.9, -0.5, 0.8, 0.2], dtype=dtype)
    outer_idx = torch.tensor([1, 0, 1, 0, 0, 1, 0])
    inner_idx = torch.tensor([2, 0, 2, 1, 0, 1, 0])
    weights = (
        torch.tensor([0.0, 1.5, 0.25, 0.0, 1.0, 2.0, 0.0], dtype=dtype)
        if weighted
        else torch.ones(7, dtype=dtype)
    )
    data = {
        "mu": mu,
        "sigma": sigma,
        "pa": pa,
        "pb": pb,
        "y": y,
        "outer_idx": outer_idx,
        "inner_idx": inner_idx,
    }
    if group_layout != "product":
        data.pop("outer_idx")
    if weighted:
        data["w"] = weights
    actual = Model(source).bind(data).trace().log_joint.sum()
    ll = (
        torch.distributions.Normal(mu[None, :, None], sigma[None, None, :]).log_prob(
            y[:, None, None]
        )
        * weights[:, None, None]
    )
    positions = outer_idx * 3 + inner_idx if group_layout == "product" else inner_idx
    groups = 6 if group_layout == "product" else 3
    grouped = ll.new_zeros(groups, 2, 4).index_add(0, positions, ll)
    inner_score = torch.logsumexp(grouped + pb.log() - pb.sum().log(), -1)
    if group_layout == "product":
        inner_score = inner_score.reshape(2, 3, 2).sum(1)
    expected = torch.logsumexp(inner_score + pa.log() - pa.sum().log(), -1).sum()
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, (mu, sigma, pa, pb)),
        torch.autograd.grad(expected, (mu, sigma, pa, pb)),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=2e-12, rtol=2e-12
        )
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-10)


@pytest.mark.parametrize("kept_axes", [(0,), (1,), (1, 0)])
def test_nested_group_projection_preserves_unequal_enclosing_class_axes(kept_axes):
    from quivers.dsl.compiler._marginal_blocks import _project_nested_aggregates

    values = torch.arange(48, dtype=torch.float64).reshape(6, 2, 4).requires_grad_()
    actual = _project_nested_aggregates(values, (2, 3), kept_axes, "nested")
    shaped = values.reshape(2, 3, 2, 4)
    if kept_axes == (0,):
        expected = shaped.sum(1)
    elif kept_axes == (1,):
        expected = shaped.sum(0)
    else:
        expected = shaped.permute(1, 0, 2, 3).reshape(6, 2, 4)
    torch.testing.assert_close(actual, expected)
    probe = torch.linspace(0.2, 1.1, actual.numel(), dtype=values.dtype).reshape(
        actual.shape
    )
    torch.testing.assert_close(
        torch.autograd.grad((actual * probe).sum(), values)[0],
        torch.autograd.grad((expected * probe).sum(), values)[0],
    )


@pytest.mark.parametrize("class_sizes", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize("parameterization", ["probs", "logits", "zero_probs"])
@pytest.mark.parametrize("weight_mode", ["unweighted", "positive", "masked", "zero"])
def test_nested_conditional_categorical_priors_keep_named_class_axes(
    class_sizes, parameterization, weight_mode
):
    depth = len(class_sizes)
    declarations = [
        f"object C{i} : FinSet {size}" for i, size in enumerate(class_sizes)
    ]
    declarations += [
        "object Group : FinSet 3",
        "object Row : FinSet 5",
        "object Out : Real 1",
    ]
    prior_parameters = ["p0 : Tensor[Real]([C0])"] + [
        f"p{i} : Tensor[Real]([C{i - 1}, C{i}])" for i in range(1, depth)
    ]
    lines = [
        *declarations,
        f"program main({', '.join(prior_parameters)}, mu : Tensor[Real]([C{depth - 1}])) : Row -> Out",
    ]
    key = "logits" if parameterization == "logits" else "probs"
    for i in range(depth):
        prior = f"p{i}" + (f"[c{i - 1}]" if i else "")
        lines.append(
            f"{'    ' * (i + 1)}marginalize c{i} : C{i} <- Categorical({key}={prior}) [over=Group]"
        )
    lines += [
        f"{'    ' * (depth + 1)}observe y : Row <- Normal(loc=mu[c{depth - 1}], scale=0.75) [via=idx{'' if weight_mode == 'unweighted' else ', weights=w'}]",
        "    return p0",
        "export main",
    ]
    source = "\n".join(lines) + "\n"
    dtype = torch.float64
    priors = []
    for i, size in enumerate(class_sizes):
        shape = (class_sizes[i - 1], size) if i else (size,)
        values = torch.linspace(
            0.1, 1.2, torch.Size(shape).numel(), dtype=dtype
        ).reshape(shape)
        if parameterization == "logits":
            values = values * 2 - 0.9
        elif parameterization == "zero_probs":
            values[..., 0] = 0.0
        priors.append(values.requires_grad_())
    mu = torch.linspace(-0.6, 0.9, class_sizes[-1], dtype=dtype).requires_grad_()
    idx = torch.tensor([2, 0, 2, 1, 0])
    y = torch.tensor([0.1, 0.3, -0.7, 0.7, 0.9], dtype=dtype)
    weight = {
        "unweighted": [1.0, 1.0, 1.0, 1.0, 1.0],
        "positive": [0.5, 1.0, 2.0, 1.5, 0.25],
        "masked": [0.0, 1.0, 0.0, 1.5, 0.25],
        "zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    }[weight_mode]
    weights = torch.tensor(weight, dtype=dtype)
    data = {
        **{f"p{i}": value for i, value in enumerate(priors)},
        "mu": mu,
        "idx": idx,
        "y": y,
    }
    if weight_mode != "unweighted":
        data["w"] = weights
    actual = Model(source).bind(data).trace().log_joint.sum()
    ll = torch.distributions.Normal(mu, 0.75).log_prob(y[:, None]) * weights[:, None]
    group_ll = ll.new_zeros(3, class_sizes[-1]).index_add(0, idx, ll)
    # Sum positive joint masses directly, so zero prior cells never require log(0).
    masses = [
        torch.softmax(prior, -1)
        if key == "logits"
        else prior / prior.sum(-1, keepdim=True)
        for prior in priors
    ]
    joint_prior = masses[0].reshape(class_sizes[0], *((1,) * (depth - 1)))
    for i in range(1, depth):
        shape = [1] * depth
        shape[i - 1 : i + 1] = class_sizes[i - 1 : i + 1]
        joint_prior = joint_prior * masses[i].reshape(shape)
    scores = group_ll.reshape(3, *((1,) * (depth - 1)), class_sizes[-1])
    expected = (scores.exp() * joint_prior).flatten(1).sum(-1).log().sum()
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (*priors, mu))
    expected_gradients = torch.autograd.grad(expected, (*priors, mu))
    for i, (gradient, oracle) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        assert torch.isfinite(gradient).all()
        if parameterization != "zero_probs" or i == depth:
            torch.testing.assert_close(gradient, oracle, atol=2e-12, rtol=2e-12)
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-10)


@pytest.mark.parametrize("through_let", [False, True])
@pytest.mark.parametrize("weight_mode", ["unweighted", "masked", "zero"])
def test_class_categorical_event_width_equal_to_group_extent_keeps_event_axis(
    through_let, weight_mode
):
    bindings = "        let emission = table[cls]\n" if through_let else ""
    argument = "emission" if through_let else "table[cls]"
    source = f"""object Class : FinSet 2
object Outcome : FinSet 3
object Group : FinSet 3
object Row : FinSet 5
object Out : Real 1
program main(table : Tensor[Real]([Class, Outcome]), pi : Tensor[Real]([Class])) : Row -> Out
    marginalize cls : Class <- Categorical(probs=pi) [over=Group]
{bindings}        observe y : Row <- Categorical(probs={argument}) [via=idx{"" if weight_mode == "unweighted" else ", weights=w"}]
    return pi
export main
"""
    dtype = torch.float64
    table = torch.tensor(
        [[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]], dtype=dtype, requires_grad=True
    )
    pi = torch.tensor([0.3, 0.9], dtype=dtype, requires_grad=True)
    idx = torch.tensor([2, 0, 2, 1, 0])
    y = torch.tensor([0, 2, 1, 2, 0])
    weights = torch.tensor(
        [0.0, 1.5, 0.25, 0.0, 1.0]
        if weight_mode == "masked"
        else [0.0] * 5
        if weight_mode == "zero"
        else [1.0] * 5,
        dtype=dtype,
    )
    data = {"table": table, "pi": pi, "idx": idx, "y": y}
    if weight_mode != "unweighted":
        data["w"] = weights
    actual = Model(source).bind(data).trace().log_joint.sum()
    ll = (table.log() - table.sum(-1, keepdim=True).log())[:, y].T * weights[:, None]
    group_ll = ll.new_zeros(3, 2).index_add(0, idx, ll)
    expected = torch.logsumexp(group_ll + pi.log() - pi.sum().log(), -1).sum()
    torch.testing.assert_close(actual, expected)
    for gradient, oracle in zip(
        torch.autograd.grad(actual, (table, pi)),
        torch.autograd.grad(expected, (table, pi)),
        strict=True,
    ):
        torch.testing.assert_close(gradient, oracle, atol=2e-12, rtol=2e-12)
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "main",
        data={name: _reference_value(value.detach()) for name, value in data.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-10)
