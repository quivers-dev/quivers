"""Raw measure queries share native construction with density evaluation."""

import math

import pytest
import torch

from quivers.continuous._measure_queries import log_mass
from quivers.continuous.inline import _raw_distribution_value
from quivers.dsl import loads
from quivers.dsl.pure_builtins import BUILTIN_REGISTRY, PURE_BUILTINS


def _source(expression):
    return f"""object One : FinSet 1
instance score : Score
define scored(mu : Real) : Real !{{score}} =
    perform score.add(log_mass({expression}))
    return mu
program model : One -> One
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- scored(a)
    return m
export model
"""


@pytest.mark.parametrize(
    "measure,low,high,normalized",
    [
        ("Restrict(Normal(mu, 1.0), -1.0, 2.0)", -1.0, 2.0, False),
        ("Restrict(Restrict(Normal(mu, 1.0), -1.0, 2.0), -0.5, 1.0)", -0.5, 1.0, False),
        ("Normalize(Restrict(Normal(mu, 1.0), -1.0, 2.0))", -1.0, 2.0, True),
        ("Truncated(Normal(mu, 1.0), -1.0, 2.0)", -1.0, 2.0, True),
        ("Normal(mu, 1.0)", -math.inf, math.inf, True),
    ],
)
def test_native_and_reference_mass_match_independent_cdf_and_two_derivatives(
    measure, low, high, normalized
):
    source = _source(measure)
    compiled = loads(source).morphism
    mu = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    actual = compiled.log_joint(torch.zeros(1, 1), {"a": mu}).sum()
    prior = -0.5 * (mu.square() + math.log(2 * math.pi)).sum()
    mass = (
        mu.sum() * 0
        if normalized
        else torch.log(
            torch.special.ndtr(high - mu) - torch.special.ndtr(low - mu)
        ).sum()
    )
    expected = prior + mass
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    derivative = torch.autograd.grad(actual, mu, create_graph=True)[0]
    oracle_derivative = torch.autograd.grad(expected, mu, create_graph=True)[0]
    torch.testing.assert_close(derivative, oracle_derivative, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(derivative.sum(), mu)[0],
        torch.autograd.grad(oracle_derivative.sum(), mu)[0],
        atol=2e-12,
        rtol=2e-12,
    )
    reference = loads(source).run("model", sites={"a": 0.3}, seed=1).log_joint
    assert float(reference) == pytest.approx(float(expected.detach()), abs=2e-12)


def test_mass_reuses_checked_registry_without_claiming_a_real_primitive():
    spec = BUILTIN_REGISTRY["log_mass"]
    assert "log_mass" in PURE_BUILTINS
    assert spec.qiec_form == "measure-query" and spec.qiec_arities == {1}
    assert spec.eager is None and not spec.qiec_overloads
    density = BUILTIN_REGISTRY["log_prob"]
    assert "log_prob" in PURE_BUILTINS
    assert density.qiec_form == "measure-query" and density.qiec_arities == {2}
    assert density.shape_rule == "sampleable-density" and density.eager is None


def test_direct_program_density_query_reuses_the_checked_raw_constructor():
    source = """object One : FinSet 1
program model : One -> One
    sample a <- Normal(loc=0.0, scale=1.0)
    let base = Normal(a, 1.0)
    let raw = Restrict(base, -1.0, 2.0)
    let density = weight_value(log_prob(raw, 0.5))
    return a
export model
"""
    from quivers.inference.trace import trace

    mu = torch.tensor([0.3], dtype=torch.float64)
    recorded = trace(loads(source).morphism, torch.zeros(1, 1), {"a": mu})
    expected = torch.distributions.Normal(mu, 1.0).log_prob(
        torch.tensor(0.5, dtype=torch.float64)
    )
    torch.testing.assert_close(recorded.sites["density"].value, expected)


def test_structural_queries_share_completion_catalogues_without_real_overloads():
    from quivers.cli.repl_complete import all_completions
    from quivers.cli.repl_session import ReplSession
    from quivers.dsl.pygments_lexer import _BUILTIN_FUNCTION_TOKENS

    labels = {completion.text for completion in all_completions(ReplSession(), "log_")}
    assert {"log_mass", "log_prob"} <= labels
    assert {"log_mass", "log_prob"} <= _BUILTIN_FUNCTION_TOKENS


def test_authored_program_let_uses_the_same_checked_mass_lowering():
    source = """object One : FinSet 1
program model : One -> One
    sample a <- Normal(loc=0.0, scale=1.0)
    let total = log_mass(Restrict(Normal(a, 1.0), -1.0, 2.0))
    return a
export model
"""
    from quivers.inference.trace import trace

    mu = torch.tensor([0.3], dtype=torch.float64)
    recorded = trace(loads(source).morphism, torch.zeros(1, 1), {"a": mu})
    # Inspect the deterministic intermediate; it is not an implicit score.
    expected = torch.log(torch.special.ndtr(2 - mu) - torch.special.ndtr(-1 - mu))
    torch.testing.assert_close(recorded.sites["total"].value, expected)


def test_helper_distribution_locals_use_capture_avoiding_construction_substitution():
    source = _source("restricted").replace(
        "    perform score.add(log_mass(restricted))",
        "    let base = Normal(mu, 1.0)\n    let restricted = Restrict(base, -1.0, 2.0)\n    perform score.add(log_mass(restricted))",
    )
    mu = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    compiled = loads(source).morphism
    actual = compiled.log_joint(torch.zeros(1, 1), {"a": mu})
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(mu) + torch.log(
        torch.special.ndtr(2 - mu) - torch.special.ndtr(-1 - mu)
    )
    torch.testing.assert_close(actual, expected)
    assert all(
        "base" != getattr(step, "name", None)
        and "restricted" != getattr(step, "name", None)
        for step in compiled.steps
    )


@pytest.mark.parametrize("through_call", [False, True])
def test_program_distribution_locals_are_shared_checked_constructions(through_call):
    call = (
        "let total <- query(restricted)"
        if through_call
        else "let total = log_mass(restricted)"
    )
    source = f"""object One : FinSet 1
define query(d : Sampleable[Real]) : LogWeight !{{}} =
    return log_mass(d)
program model : One -> One
    sample a <- Normal(loc=0.0, scale=1.0)
    let base = Normal(a, 1.0)
    let restricted = Restrict(base, -1.0, 2.0)
    {call}
    return a
export model
"""
    from quivers.inference.trace import trace

    mu = torch.tensor([0.3], dtype=torch.float64)
    recorded = trace(loads(source).morphism, torch.zeros(1, 1), {"a": mu})
    expected = torch.log(torch.special.ndtr(2 - mu) - torch.special.ndtr(-1 - mu))
    torch.testing.assert_close(recorded.sites["total"].value, expected)


@pytest.mark.parametrize("closed,expected", [(True, 0.0), (False, -math.inf)])
def test_source_mass_preserves_boolean_endpoint_flags_and_atoms(closed, expected):
    text = "true" if closed else "false"
    source = _source(f"Restrict(PointMass(mu), mu, mu, {text}, true)")
    compiled = loads(source).morphism
    mu = torch.tensor([0.3], dtype=torch.float64)
    joint = compiled.log_joint(torch.zeros(1, 1), {"a": mu})
    prior = torch.distributions.Normal(0.0, 1.0).log_prob(mu)
    if math.isinf(expected):
        assert bool(torch.isneginf(joint).all())
    else:
        torch.testing.assert_close(joint, prior)
    reference = loads(source).run("model", sites={"a": 0.3}, seed=1).log_joint
    assert float(reference) == pytest.approx(float(prior) + expected)


@pytest.mark.parametrize("closed", [True, False])
def test_named_draw_endpoint_flags_are_reserved_boolean_literals(closed):
    flag = "true" if closed else "false"
    source = f"""object One : FinSet 1
program model : One -> One
    observe y <- Restrict(base=Poisson(rate=1.4), low=0.0, high=2.0, lower_closed={flag})
    return y
export model
"""
    compiled = loads(source).morphism
    y = torch.tensor([1])
    actual = compiled.log_joint(torch.zeros(1, 1), {"y": y})
    probabilities = [math.exp(-1.4) * 1.4**k / math.factorial(k) for k in range(3)]
    denominator = sum(probabilities if closed else probabilities[1:])
    expected = math.log(probabilities[1]) - math.log(denominator)
    assert float(actual) == pytest.approx(expected, abs=2e-6)
    reference = loads(source).run("model", data={"y": 1}, seed=1).log_joint
    assert float(reference) == pytest.approx(expected, abs=2e-12)


def test_native_raw_constructor_preserves_replications_and_intrinsic_events():
    normal = _raw_distribution_value(
        "Normal",
        {"loc": torch.tensor(0.0, dtype=torch.float64), "scale": 1.0},
        plate_shape=(2, 3),
    )
    high = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=torch.float64)
    restricted = _raw_distribution_value(
        "Restrict", {"base": normal, "low": -1.0, "high": high}, plate_shape=(2, 3)
    )
    oracle = torch.log(
        torch.special.ndtr(high)
        - torch.special.ndtr(torch.tensor(-1.0, dtype=torch.float64))
    )
    torch.testing.assert_close(log_mass(restricted, batch_axes=1), oracle.sum(-1))
    torch.testing.assert_close(log_mass(restricted), oracle.sum())
    concentration = torch.tensor(
        [0.4, 1.2, 2.0], dtype=torch.float64, requires_grad=True
    )
    simplex = _raw_distribution_value(
        "Dirichlet",
        {"concentration": concentration},
        event_shape=(3,),
        plate_shape=(2,),
    )
    assert simplex.batch_shape == (2,) and simplex.event_shape == (3,)
    mass = log_mass(simplex, batch_axes=1)
    assert mass.shape == (2,)
    torch.testing.assert_close(mass, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        torch.autograd.grad(mass.sum(), concentration)[0],
        torch.zeros_like(concentration),
    )
