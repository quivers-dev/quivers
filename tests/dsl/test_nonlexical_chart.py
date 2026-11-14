"""Nonlexical deductions saturate item weights without a proof-height cutoff."""

import math

import pytest
import torch

from quivers.qiec import dumps, loads
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.program_runtime import deduction_item, run_deduction
from tests.dsl.test_deduction_elaboration import _classic, _module

GROW = """\
object Term : FinSet 16
deduction Grow : Term -> Term [semiring=SEMIRING, start=Goal, depth=8]
    atoms A, F, P, Goal
    rule grow : P(X) |- P(F(X))
    rule finish : P(F(F(F(F(F(F(A))))))) |- Goal
"""


@pytest.mark.parametrize(
    "semiring,one",
    [("Boolean", True), ("Counting", 1), ("LogProb", 0.0), ("Viterbi", 0.0)],
)
def test_repeated_growth_reaches_a_delayed_goal(semiring, one):
    """Six uses of one rule exceed the old axiom-plus-rule height bound."""
    module = loads(dumps(_module(GROW, semiring)))
    axiom = deduction_item(module, "Grow", "P", deduction_item(module, "Grow", "A"))
    result = run_deduction(
        module, "Grow", axioms=[axiom], axiom_weights=[one], fuel=1_000_000
    )
    oracle = _classic(GROW, "Grow", semiring)(
        [(("P", ("atom", "A")), torch.tensor(one))]
    )
    assert result.weight == pytest.approx(one)
    assert result.weight == pytest.approx(oracle.goal_weight().item())


@pytest.mark.parametrize(
    "semiring,zero",
    [
        ("Boolean", False),
        ("Counting", 0),
        ("LogProb", -math.inf),
        ("Viterbi", -math.inf),
    ],
)
def test_empty_axioms_return_semiring_zero(semiring, zero):
    module = _module(GROW, semiring)
    assert run_deduction(module, "Grow", axioms=[], axiom_weights=[]).weight == zero


def test_constructor_depth_remains_a_side_condition():
    module = _module(GROW.replace("depth=8", "depth=4"), "Boolean")
    axiom = deduction_item(module, "Grow", "P", deduction_item(module, "Grow", "A"))
    assert (
        run_deduction(module, "Grow", axioms=[axiom], axiom_weights=[True]).weight
        is False
    )


def test_rule_chains_are_independent_of_constructor_depth():
    names = [f"A{k}" for k in range(13)]
    source = (
        "object Term : FinSet 16\n"
        + "deduction Chain : Term -> Term [semiring=Counting, start=A12, depth=1]\n"
    )
    source += "    atoms " + ", ".join(names) + "\n"
    source += "".join(
        f"    rule r{k} : {names[k]} |- {names[k + 1]}\n" for k in range(12)
    )
    module = _module(source)
    result = run_deduction(
        module,
        "Chain",
        axioms=[deduction_item(module, "Chain", "A0")],
        axiom_weights=[1],
    )
    assert result.weight == 1


def test_multi_premise_growth_and_duplicate_proofs():
    source = GROW.replace(
        "rule grow : P(X) |- P(F(X))",
        "rule grow : P(X), A |- P(F(X))\n    rule grow_again : P(X), A |- P(F(X))",
    )
    module = _module(source, "Counting")
    a = deduction_item(module, "Grow", "A")
    result = run_deduction(
        module,
        "Grow",
        axioms=[deduction_item(module, "Grow", "P", a), a],
        axiom_weights=[1, 1],
    )
    assert result.weight == 2**6
    oracle = _classic(source, "Grow", "Counting")(
        [(("P", ("atom", "A")), torch.tensor(1)), (("atom", "A"), torch.tensor(1))]
    )
    assert oracle.goal_weight().item() == result.weight


def test_nonconvergent_cycle_raises_when_the_execution_budget_is_spent():
    source = """object Term : FinSet 2
deduction Cycle : Term -> Term [semiring=Counting, start=A]
    atoms A
    rule loop : A |- A
"""
    module = _module(source)
    with pytest.raises(ExecutionFailure, match="did not finish within 2000 steps"):
        run_deduction(
            module,
            "Cycle",
            axioms=[deduction_item(module, "Cycle", "A")],
            axiom_weights=[1],
            fuel=2000,
        )


def test_learned_chain_values_and_finite_difference_gradient_match_agenda():
    source = """object Term : FinSet 4
deduction Weighted : Term -> Term [semiring=LogProb, start=C, depth=1]
    atoms A, B, C
    rule first : A |- B #[learnable]
    rule second : B |- C #[learnable]
    rule direct : A |- C #[learnable]
"""
    module = _module(source)
    axioms = [deduction_item(module, "Weighted", "A")]
    oracle = _classic(source, "Weighted")
    oracle_axioms = [(("atom", "A"), torch.tensor(0.0))]
    oracle(oracle_axioms)
    values = {"first": -0.2, "second": -0.3, "direct": -1.0}
    parameters = {}
    with torch.no_grad():
        for name, group in oracle.modules["_rule_module"].named_children():
            for parameter in group.values():
                parameter.fill_(values[name.removeprefix("rule_")])
                parameters[name.removeprefix("rule_")] = parameter
    expected = oracle(oracle_axioms).goal_weight()
    expected.backward()
    store = {f"Weighted.rule.{name}:": value for name, value in values.items()}

    def evaluate(overrides):
        return run_deduction(
            module,
            "Weighted",
            axioms=axioms,
            axiom_weights=[0.0],
            parameters={**store, **overrides},
        ).weight

    assert evaluate({}) == pytest.approx(expected.item(), abs=1e-6)
    h = 1e-4
    for name, value in values.items():
        key = f"Weighted.rule.{name}:"
        derivative = (evaluate({key: value + h}) - evaluate({key: value - h})) / (2 * h)
        assert derivative == pytest.approx(parameters[name].grad.item(), abs=1e-5)


def test_a_contractive_cycle_converges_in_value_and_derivative():
    """Repeated loop proofs sum geometrically rather than being truncated."""
    source = """object Term : FinSet 2
deduction Cycle : Term -> Term [semiring=LogProb, start=A, tolerance=0.0000000001]
    atoms A
    rule loop : A |- A #[learnable]
"""
    module = _module(source)
    axiom = deduction_item(module, "Cycle", "A")

    def evaluate(log_weight):
        return run_deduction(
            module,
            "Cycle",
            axioms=[axiom],
            axiom_weights=[0.0],
            parameters={"Cycle.rule.loop:": log_weight},
            fuel=100_000,
        ).weight

    log_weight = math.log(0.2)
    assert evaluate(log_weight) == pytest.approx(-math.log1p(-0.2), abs=1e-9)
    h = 1e-5
    derivative = (evaluate(log_weight + h) - evaluate(log_weight - h)) / (2 * h)
    assert derivative == pytest.approx(0.2 / 0.8, abs=1e-7)


def test_boolean_cycle_reaches_a_fixed_point():
    module = _module("""object Term : FinSet 2
deduction Cycle : Term -> Term [semiring=Boolean, start=A]
    atoms A
    rule loop : A |- A
""")
    assert (
        run_deduction(
            module,
            "Cycle",
            axioms=[deduction_item(module, "Cycle", "A")],
            axiom_weights=[True],
            fuel=10_000,
        ).weight
        is True
    )
