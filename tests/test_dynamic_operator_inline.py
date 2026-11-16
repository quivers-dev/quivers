"""Nested runtime constructors share ordinary inline packing and measure boundaries."""

import pytest
import torch

from quivers.dsl import loads
from quivers.inference.trace import trace


@pytest.fixture(autouse=True)
def double_literals():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _compiled(expression):
    return loads(f"""object Out:Real 1
program model(location:Real, scale:Real):Out->Out
    sample b <- {expression}
    return b
export model
""")


@pytest.mark.parametrize(
    "base", ["Normal(loc=location,scale=scale)", "HalfNormal(scale=scale)"]
)
def test_dynamic_restriction_has_normalized_density_all_gradients_and_exact_support(
    base,
):
    source = _compiled(f"Restrict(base={base},low=0.1,high=0.8)")
    point = torch.tensor([0.2, 0.6, 0.3], dtype=torch.double, requires_grad=True)

    def actual(theta):
        return trace(
            source.morphism,
            torch.zeros(1, 1),
            dict(zip(("location", "scale", "b"), theta.unbind(), strict=True)),
        ).log_joint.sum()

    def expected(theta):
        location, scale, value = theta.unbind()
        location = location if base.startswith("Normal") else location * 0
        normal = torch.distributions.Normal(location, scale)
        mass = normal.cdf(theta.new_tensor(0.8)) - normal.cdf(theta.new_tensor(0.1))
        return normal.log_prob(value) - mass.log()

    torch.testing.assert_close(actual(point), expected(point), atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(expected, point),
        atol=3e-10,
        rtol=3e-10,
    )
    assert torch.autograd.gradcheck(actual, (point,))
    reference = source.run(
        "model", data={"location": 0.2, "scale": 0.6, "out": (0.0,)}, sites={"b": 0.3}
    ).log_joint
    assert reference == pytest.approx(actual(point).item(), abs=2e-12)
    from quivers.inference.registry import LatentRegistry

    support = LatentRegistry.from_model(
        source.morphism, observed_names={"location", "scale", "out"}
    )["b"].support
    assert support.check(torch.tensor(0.4)) and not support.check(torch.tensor(-0.1))


def test_nested_dynamic_mixture_retains_component_submeasure_mass():
    source = _compiled(
        "Normalize(base=Mixture(weights=[0.3,0.7],components=[Restrict(base=Normal(loc=location,scale=scale),low=-0.5,high=0.8),Normal(loc=location,scale=2.0)]))"
    )
    location = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    scale = torch.tensor(0.7, dtype=torch.double, requires_grad=True)
    value = torch.tensor(0.4, dtype=torch.double)
    actual = trace(
        source.morphism,
        torch.zeros(1, 1),
        {"location": location, "scale": scale, "b": value},
    ).log_joint.sum()
    first = torch.distributions.Normal(location, scale)
    second = torch.distributions.Normal(location, torch.tensor(2.0, dtype=torch.double))
    mass = first.cdf(value.new_tensor(0.8)) - first.cdf(value.new_tensor(-0.5))
    expected = torch.log(
        0.3 * first.log_prob(value).exp() + 0.7 * second.log_prob(value).exp()
    ) - torch.log(0.3 * mass + 0.7)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for got, want in zip(
        torch.autograd.grad(actual, (location, scale)),
        torch.autograd.grad(expected, (location, scale)),
        strict=True,
    ):
        torch.testing.assert_close(got, want, atol=3e-10, rtol=3e-10)
    assert source.run(
        "model", data={"location": 0.2, "scale": 0.7, "out": (0.0,)}, sites={"b": 0.4}
    ).log_joint == pytest.approx(actual.item(), abs=2e-10)


def test_dynamic_independent_zero_preserves_declared_component_axis_and_reference():
    source = loads("""object D:FinSet 3
object Out:Real 3
program model(location:Tensor[Real]([D]),scale:Tensor[Real]([D,D])):Out->Out
    sample b <- Independent(base=MultivariateNormal(loc=location,scale_tril=scale),reinterpreted_batch_ndims=0)
    return b
export model
""")
    location = torch.tensor([0.1, 0.2, 0.3], dtype=torch.double, requires_grad=True)
    scale = torch.tensor(
        [[0.6, 0.0, 0.0], [0.1, 0.5, 0.0], [-0.2, 0.1, 0.7]],
        dtype=torch.double,
        requires_grad=True,
    )
    point = torch.tensor([-0.2, 0.7, 0.9], dtype=torch.double)
    actual = trace(
        source.morphism,
        torch.zeros(1, 1),
        {"location": location, "scale": scale, "b": point},
    ).log_joint.sum()
    expected = torch.distributions.MultivariateNormal(
        location, scale_tril=scale
    ).log_prob(point)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (location, scale)),
        torch.autograd.grad(expected, (location, scale)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)
    reference = source.run(
        "model",
        data={
            "location": (0.1, 0.2, 0.3),
            "scale": ((0.6, 0.0, 0.0), (0.1, 0.5, 0.0), (-0.2, 0.1, 0.7)),
            "out": (0.0, 0.0, 0.0),
        },
        sites={"b": (-0.2, 0.7, 0.9)},
    ).log_joint
    assert reference == pytest.approx(expected.item(), abs=2e-12)


def test_dynamic_independent_zero_preserves_scalar_density_and_gradient():
    source = _compiled(
        "Independent(base=Normal(loc=location,scale=scale),reinterpreted_batch_ndims=0)"
    )
    location = torch.tensor(0.2, requires_grad=True)
    scale = torch.tensor(0.6, requires_grad=True)
    actual = trace(
        source.morphism,
        torch.zeros(1, 1),
        {"location": location, "scale": scale, "b": torch.tensor(0.3)},
    ).log_joint.sum()
    expected = torch.distributions.Normal(location, scale).log_prob(torch.tensor(0.3))
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (location, scale)),
        torch.autograd.grad(expected, (location, scale)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)
    assert source.run(
        "model", data={"location": 0.2, "scale": 0.6, "out": (0.0,)}, sites={"b": 0.3}
    ).log_joint == pytest.approx(expected.item(), abs=2e-12)


@pytest.mark.parametrize("rows", [2, 5])
def test_indexed_dynamic_restriction_uses_declared_coefficient_axis(rows):
    source = loads(f"""object D:FinSet 2
object Row:FinSet {rows}
program model(scales:Tensor[Real]([D])):Row->Row
    sample b:D <- Restrict(base=Normal(loc=0.0,scale=scales),low=-0.5,high=0.8)
    let center = sum(b)
    observe y:Row <- Normal(loc=center,scale=1.0)
    return y
export model
""")
    scale = torch.tensor([0.4, 0.7], requires_grad=True)
    coefficient = torch.tensor([0.2, -0.3], requires_grad=True)
    response = torch.linspace(-0.5, 0.5, rows)
    recorded = trace(
        source.morphism,
        torch.zeros(1, 1),
        {"scales": scale, "b": coefficient, "y": response},
    )
    normal = torch.distributions.Normal(torch.zeros(2), scale)
    mass = normal.cdf(scale.new_tensor(0.8)) - normal.cdf(scale.new_tensor(-0.5))
    expected = (
        normal.log_prob(coefficient) - mass.log()
    ).sum() + torch.distributions.Normal(coefficient.sum(), 1.0).log_prob(
        response
    ).sum()
    torch.testing.assert_close(
        recorded.log_joint.sum(), expected, atol=2e-12, rtol=2e-12
    )
    for got, want in zip(
        torch.autograd.grad(recorded.log_joint.sum(), (scale, coefficient)),
        torch.autograd.grad(expected, (scale, coefficient)),
        strict=True,
    ):
        torch.testing.assert_close(got, want, atol=2e-11, rtol=2e-11)
    reference = source.run(
        "model",
        data={"scales": (0.4, 0.7), "y": tuple(response.tolist())},
        sites={"b": (0.2, -0.3)},
    ).log_joint
    assert reference == pytest.approx(expected.item(), abs=2e-12)


def test_symbolic_outer_extent_keeps_one_constructor_comprehension():
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.dsl.parser import parse
    from quivers.qiec.serialization import to_data

    source = """program model[n:Nat](scales:Tensor[Real]([n])) : n -> n
    sample b : n <- Restrict(base=Normal(loc=0.0,scale=scales),low=-0.5,high=0.8)
    return b
export model
"""
    checked = lower_qvr_to_qiec(parse(source))

    def nodes(value):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from nodes(item)
        elif isinstance(value, list):
            for item in value:
                yield from nodes(item)

    # One symbolic comprehension retains the extent; construction does not
    # generate a copy of the constructor for every possible coordinate.
    comprehensions = [
        node
        for node in nodes(to_data(checked))
        if node.get("$type") == "terms.Comprehension"
    ]
    assert len(comprehensions) == 1
    fields = {field["name"]: field["value"] for field in comprehensions[0]["fields"]}
    assert fields["extent"]["$type"] == "types.IndexVariable"
    extent_fields = {
        field["name"]: field["value"] for field in fields["extent"]["fields"]
    }
    assert extent_fields["name"] == "n"

    large = loads(source).model(n=4096)
    scale = torch.full((4096,), 0.7)
    coefficient = torch.full((4096,), 0.2)
    actual = trace(
        large.morphism, torch.zeros(1, 1), {"scales": scale, "b": coefficient}
    )
    base = torch.distributions.Normal(0.0, scale)
    expected = (
        base.log_prob(coefficient)
        - (base.cdf(torch.tensor(0.8)) - base.cdf(torch.tensor(-0.5))).log()
    ).sum()
    torch.testing.assert_close(actual.log_joint.sum(), expected)
