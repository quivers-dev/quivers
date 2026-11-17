"""Checked deterministic carriers survive planning and Stan block placement."""

from __future__ import annotations

import json
import subprocess

import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import CSInteger, transpile
from quivers.transpile.ir import (
    CSBoolean,
    CSNonnegativeInteger,
    CSReal,
    ConstraintSpec,
    DimDynamic,
    IRDataInput,
    IRDeterministic,
    IRSample,
    IRMarginalize,
    Plate,
    from_constraint,
)
from quivers.transpile.plan import Lower
from quivers.transpile._pipeline import EmitPretty
from quivers.transpile.renderers.stan import StanRenderer
from tests.transpile._tools import require_tool

SCALAR = """object Out:Real 1
program p(x:Real,y:Real):Out->Out
    let flag = not (x > 0.0)
    let mu = select(flag,3.0,-2.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
PLATED = """object Row:FinSet 3
object Out:Real 1
program p(x:Tensor[Real]([Row]),y:Tensor[Real]([Row])):Row->Out
    let flag = not (x > 0.0)
    let mu = select(flag,3.0,-2.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
INTEGER = """object Out:Real 1
program p(n:Int,y:Real):Out->Out
    let shifted = n - 2
    let flag = shifted < 0
    let mu = select(flag,real(shifted),-2.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
INTEGER_PLATE = """object Row:FinSet 3
object Out:Real 1
program p(n:Tensor[Int]([Row]),y:Tensor[Real]([Row])):Row->Out
    let shifted = n - 2
    let flag = shifted < 0
    let mu = select(flag,real(shifted),-2.0)
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
LATENT = """object Out:Real 1
program p(y:Real):Out->Out
    sample a <- Normal(loc=0.0,scale=1.0)
    let untouched = 2.0 * a
    let flag = a < 0.0
    let mu = select(flag,untouched,-2.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
CALL = """object Out:Real 1
define negative(n:Int):Bool !{} =
    if n < 0 then
        return true
    else
        return false
program p(n:Int,y:Real):Out->Out
    let flag <- negative(n)
    let mu = select(flag,3.0,-2.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""


@pytest.mark.parametrize(
    "source", [SCALAR, PLATED, INTEGER, INTEGER_PLATE, LATENT, CALL]
)
def test_actual_stanc_accepts_typed_intermediates_and_return_closures(source):
    emitted = transpile(parse(source), target="stan")
    done = subprocess.run(
        [require_tool("stanc"), "--info", "-"], input=emitted, capture_output=True
    )
    assert done.returncode == 0, done.stderr.decode() + "\n" + emitted.decode()
    assert b"int division" not in done.stderr
    info = json.loads(done.stdout)
    assert "mu_value" in info["generated quantities"]


@pytest.mark.parametrize(
    "target",
    [
        "bugs",
        "church",
        "edward2",
        "gen",
        "jags",
        "numpyro",
        "pymc",
        "pyro",
        "stan",
        "turing",
        "webppl",
    ],
)
def test_all_targets_retain_checked_integer_and_boolean_binding_constraints(target):
    plan = Lower().forward(parse(INTEGER), target=target)
    bindings = {
        node.name: node for node in plan.body if isinstance(node, IRDeterministic)
    }
    assert isinstance(bindings["shifted"].constraint, CSInteger)
    assert isinstance(bindings["flag"].constraint, CSBoolean)
    assert isinstance(bindings["mu"].constraint, CSReal)
    assert isinstance(
        next(value for value in plan.inputs if value.name == "n").constraint, CSInteger
    )
    assert transpile(parse(INTEGER), target=target)


def test_unbounded_integer_constraint_roundtrips_and_keeps_negative_values():
    spec = CSInteger()
    assert from_constraint(spec.to_constraint()) == spec
    restored = ConstraintSpec.model_validate(spec.model_dump())
    assert restored == spec
    assert bool(restored.to_constraint().check(torch.tensor([-9, -1, 0, 2])).all())


def test_runtime_nat_refinement_is_retained_without_restricting_arithmetic_result():
    plan = Lower().forward(parse(INTEGER.replace("n:Int", "n:Nat")), target="stan")
    assert isinstance(
        next(node for node in plan.inputs if node.name == "n").constraint,
        CSNonnegativeInteger,
    )
    assert isinstance(
        next(
            node
            for node in plan.body
            if isinstance(node, IRDeterministic) and node.name == "shifted"
        ).constraint,
        CSInteger,
    )


def test_stan_preserves_unrelated_transformed_parameter_shape_and_visibility():
    text = transpile(parse(LATENT), target="stan").decode()
    transformed = text.split("transformed parameters", 1)[1].split("model", 1)[0]
    assert "untouched" in transformed and "flag" not in transformed
    generated = text.split("generated quantities", 1)[1]
    assert "flag" in generated and "mu_value" in generated


def test_model_local_dimensions_are_refused_before_emitting_invalid_global_shapes():
    plan = Lower().forward(parse(SCALAR), target="stan")
    bad = IRDataInput(
        name="bad",
        constraint=CSReal(),
        plate=Plate(
            event_dims=(), batch_dims=(DimDynamic(size_name="flag", name="dynamic"),)
        ),
    )
    from quivers.transpile import UnsupportedConstruct

    with pytest.raises(UnsupportedConstruct, match="model-local binding `flag`"):
        StanRenderer().render(plan.with_(inputs=(*plan.inputs, bad)))


@pytest.mark.parametrize(
    "source,inputs,expected",
    [
        (SCALAR, {"x": torch.tensor(-1.0), "y": torch.tensor(0.4)}, torch.tensor(3.0)),
        (
            PLATED,
            {"x": torch.tensor([-1.0, 0.0, 2.0]), "y": torch.tensor([0.4, 0.2, -0.3])},
            torch.tensor([3.0, 3.0, -2.0]),
        ),
        (INTEGER, {"n": torch.tensor(-3), "y": torch.tensor(0.4)}, torch.tensor(-5.0)),
        (
            INTEGER_PLATE,
            {"n": torch.tensor([-3, 2, 5]), "y": torch.tensor([0.4, 0.2, -0.3])},
            torch.tensor([-5.0, -2.0, -2.0]),
        ),
    ],
)
def test_actual_pyro_preserves_values_and_complete_observation_density(
    source, inputs, expected
):
    import pyro

    namespace = {"torch": torch, "pyro": pyro}
    exec(transpile(parse(source), target="pyro"), namespace)
    trace = (
        pyro.poutine.trace(namespace["model"]).get_trace(torch.zeros(1, 1), **inputs)
        if source in (SCALAR, INTEGER)
        else pyro.poutine.trace(namespace["model"]).get_trace(**inputs)
    )
    torch.testing.assert_close(trace.nodes["_RETURN"]["value"], expected)
    torch.testing.assert_close(
        trace.log_prob_sum(),
        torch.distributions.Normal(expected, 1.0).log_prob(inputs["y"]).sum(),
    )


def test_model_local_dimension_cannot_escape_into_parameter_declaration():
    plan = Lower().forward(parse(SCALAR), target="stan")
    latent = next(
        node
        for node in Lower().forward(parse(LATENT), target="stan").body
        if isinstance(node, IRSample)
    ).with_(
        plate=Plate(
            event_dims=(), batch_dims=(DimDynamic(size_name="flag", name="dynamic"),)
        )
    )
    from quivers.transpile import UnsupportedConstruct

    with pytest.raises(UnsupportedConstruct, match="model-local binding `flag`"):
        StanRenderer().render(plan.with_(body=(*plan.body[:-1], latent, plan.body[-1])))


def test_continuous_marginal_scope_uses_same_checked_carrier_placement():
    plan = Lower().forward(parse(LATENT), target="stan")
    latent = next(node for node in plan.body if isinstance(node, IRSample))
    scope = tuple(node for node in plan.body if isinstance(node, IRDeterministic))
    marginal = IRMarginalize(
        latent=latent.name,
        family=latent.family,
        args=latent.args,
        arg_names=latent.arg_names,
        constraint=latent.constraint,
        plate=latent.plate,
        reduction="logsumexp",
        scope=scope,
    )
    body = tuple(node for node in plan.body if node is not latent and node not in scope)
    emitted = EmitPretty("stan").forward(
        StanRenderer().render(plan.with_(body=(marginal, *body)))
    )
    done = subprocess.run(
        [require_tool("stanc"), "--info", "-"], input=emitted, capture_output=True
    )
    assert done.returncode == 0, done.stderr.decode()
    assert "flag" in json.loads(done.stdout)["generated quantities"]


def test_model_local_dimensions_are_legal_locally_but_not_as_output_sizes():
    plan = Lower().forward(parse(SCALAR), target="stan")
    flag, mu = tuple(node for node in plan.body if isinstance(node, IRDeterministic))
    mu = mu.with_(
        plate=Plate(
            event_dims=(), batch_dims=(DimDynamic(size_name="flag", name="dynamic"),)
        )
    )
    emitted = EmitPretty("stan").forward(
        StanRenderer().render(plan.with_(body=(flag, mu)))
    )
    done = subprocess.run(
        [require_tool("stanc"), "--info", "-"], input=emitted, capture_output=True
    )
    assert done.returncode == 0, done.stderr.decode() + emitted.decode()
    from quivers.transpile import UnsupportedConstruct

    with pytest.raises(UnsupportedConstruct, match="model-local binding `flag`"):
        StanRenderer().render(plan.with_(body=(flag, mu, plan.body[-1])))
