"""Nested sampleable values share ordinary latent-network sites and evaluation."""

import pytest
import torch

from quivers.dsl import parse
from quivers.dsl.compiler.core import Compiler
from quivers.inference import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _host(value):
    if value.ndim == 0:
        return value.item()
    return tuple(_host(part) for part in value)


@pytest.mark.parametrize("source_kind", ["linear", "mlp(3)"])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize(
    "operator",
    [
        "Restrict(base=net,low=-1.0,high=1.0)",
        "Normalize(base=Mixture(weights=[0.3,0.7],components=[Restrict(base=net,low=-1.0,high=1.0),Restrict(base=alias,low=-1.0,high=1.0)]))",
    ],
)
def test_nested_latent_network_joint_all_gradients_and_reference(
    source_kind, width, rows, operator
):
    source = f"""object Input:Real 2
object Output:Real {width}
morphism net:Input->Output [param_source={source_kind},weights=latent,prior_scale=0.5] ~ Normal
define alias = net
program model:Input->Output
    sample b <- {operator}
    return b
export model
"""
    compiler = Compiler(parse(source))
    program = compiler.compile()
    program.morphism.double()
    network = compiler._latent_networks["net"]
    weights = {
        network.latent_sites[path]: torch.linspace(
            -0.15, 0.2, rows * parameter.numel(), dtype=torch.double
        )
        .reshape(rows, *parameter.shape)
        .requires_grad_()
        for path, parameter in network.source.named_parameters()
    }
    x = torch.tensor([[0.3, -0.2], [0.1, 0.4]], dtype=torch.double)[
        :rows
    ].requires_grad_()
    y = torch.tensor([[0.15], [-0.2]], dtype=torch.double)[:rows].expand(-1, width)
    prior = sum(
        torch.distributions.Normal(x.new_tensor(0), x.new_tensor(0.5))
        .log_prob(weight)
        .flatten(1)
        .sum(1)
        for weight in weights.values()
    )
    inputs = x
    if source_kind != "linear":
        inputs = torch.tanh(
            torch.einsum("bji,bi->bj", weights["net_param_layer0_weight"], inputs)
            + weights["net_param_layer0_bias"]
        )
    raw = (
        torch.einsum("bji,bi->bj", weights["net_param_weight"], inputs)
        + weights["net_param_bias"]
    )
    normal = torch.distributions.Normal(raw[:, :width], raw[:, width:].exp())
    expected = prior + (
        normal.log_prob(y)
        - (normal.cdf(y.new_tensor(1)) - normal.cdf(y.new_tensor(-1))).log()
    ).sum(-1)
    recorded = trace(program.morphism, x, {**weights, "b": y})
    torch.testing.assert_close(recorded.log_joint, expected, atol=2e-12, rtol=2e-12)
    leaves = (x, *weights.values())
    for got, wanted in zip(
        torch.autograd.grad(recorded.log_joint.sum(), leaves, retain_graph=True),
        torch.autograd.grad(expected.sum(), leaves),
        strict=True,
    ):
        torch.testing.assert_close(got, wanted, atol=2e-11, rtol=2e-11)
    # Aliases and repeated components share each prior and raw evaluation;
    # there are no unused point-estimated copies of the latent heads.
    assert not list(program.morphism.parameters())
    assert [name for name in recorded.sites if name in weights] == list(weights)
    assert len([name for name in recorded.sites if name.startswith("net__raw_b")]) == 1
    assert [
        (parameter.name, parameter.role)
        for parameter in program.entry("model").parameters
    ] == [("input", "domain")]
    reference = program.run(
        "model",
        data={"input": _host(x[0].detach())},
        sites={
            "b": _host(y[0]),
            **{name: _host(value[0].detach()) for name, value in weights.items()},
        },
    )
    assert reference.log_joint == pytest.approx(expected[0].item(), abs=2e-11)


def test_nested_latent_raw_binding_does_not_shadow_an_authored_binding():
    source = """object Input:Real 2
object Output:Real 1
morphism net:Input->Output [weights=latent] ~ Normal
program model:Input->Output
    let net__raw_b = 0.2
    sample b <- Restrict(base=net,low=-1.0,high=1.0)
    return b
export model
"""
    compiled = Compiler(parse(source)).compile()
    recorded = trace(compiled.morphism, torch.zeros(1, 2))
    assert "net__raw_b_1" in recorded.sites
    torch.testing.assert_close(recorded.sites["net__raw_b"].value, torch.tensor([0.2]))


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_nested_latent_schema_and_inference_do_not_require_or_train_point_heads(method):
    from quivers.inference import HMCKernel, MCMC, NUTSKernel
    from quivers.modeling import MCMCMethod, Model, VariationalMethod

    source = """object Input:Real 2
object Output:Real 1
morphism net:Input->Output [weights=latent,prior_scale=0.3] ~ Normal
program model:Input->Output
    observe y <- Restrict(base=net,low=-1.0,high=1.0)
    return y
export model
"""
    model = Model(source)
    assert set(model.schema.describe()) == {"y"}
    bound = model.bind({"y": torch.tensor([0.2])}, input=torch.tensor([[0.3, -0.1]]))
    inference = (
        VariationalMethod(num_steps=2, seed=7)
        if method == "svi"
        else MCMCMethod(
            sampler=MCMC(
                HMCKernel() if method == "hmc" else NUTSKernel(),
                num_warmup=2,
                num_samples=2,
                num_chains=1,
                seed=7,
            )
        )
    )
    fit = bound.infer(inference)
    draws = fit.draws(num_samples=2)
    assert set(draws) == {"net_param_weight", "net_param_bias"}
    assert all(
        value.shape[:2] == (1, 2) and torch.isfinite(value).all()
        for value in draws.values()
    )
    with pytest.raises(ValueError, match="unknown|unexpected|not declared"):
        model.bind(
            {"y": torch.tensor([0.2]), "net_param_weight": torch.zeros(2, 2)},
            input=torch.tensor([[0.3, -0.1]]),
        )


def test_plain_latent_kernel_alias_uses_the_same_sampled_heads():
    source = """object Input:Real 2
object Output:Real 1
morphism net:Input->Output [weights=latent,prior_scale=0.5] ~ Normal
define alias = net
program model(x:Tensor[Real]([2])):Input->Output
    sample b <- alias(x)
    return b
export model
"""
    compiled = Compiler(parse(source)).compile()
    x = torch.tensor([[0.2, -0.1]])
    w, bias = torch.tensor([[0.1, 0.2], [0.3, -0.2]]), torch.tensor([0.15, -0.1])
    y = torch.tensor([[0.25]])
    actual = trace(
        compiled.morphism,
        x,
        {"net_param_weight": w, "net_param_bias": bias, "b": y, "x": x[0]},
    ).log_joint.sum()
    raw = x @ w.T + bias
    expected = (
        torch.distributions.Normal(raw[:, :1], raw[:, 1:].exp()).log_prob(y).sum()
        + torch.distributions.Normal(0.0, 0.5).log_prob(w).sum()
        + torch.distributions.Normal(0.0, 0.5).log_prob(bias).sum()
    )
    torch.testing.assert_close(actual, expected)


def test_numeric_lexical_shadow_does_not_create_unused_latent_priors():
    source = """object Input:Real 2
object Output:Real 1
morphism net:Input->Output [weights=latent] ~ Normal
program model(net:Real):Input->Output
    sample b <- Normal(loc=net,scale=1.0)
    return b
export model
"""
    compiled = Compiler(parse(source)).compile()
    recorded = trace(
        compiled.morphism,
        torch.zeros(1, 2),
        {"net": torch.tensor(0.2), "b": torch.tensor(0.1)},
    )
    assert set(recorded.sites) == {"b"}
    torch.testing.assert_close(
        recorded.log_joint.sum(),
        torch.distributions.Normal(0.2, 1.0).log_prob(torch.tensor(0.1)),
    )
