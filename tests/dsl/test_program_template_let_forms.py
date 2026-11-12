"""Instantiating a program template whose body uses every let-expression form.

A parametric program is instantiated by substituting its parameters and
alpha-renaming its locals through every step, let right-hand sides
included. The instantiated program must denote the same joint as the
program written out by hand at the same arguments, whichever form a let
takes: indexing by host data and by bound indices, nested indexing, both
``factor`` forms, calls, method calls, list and tuple collections,
``map`` and ``fold``, arithmetic, comparisons, and boolean connectives.
A program that calls a named computation, ``traverse`` among the calls,
runs only as its checked computation, so the runtime compiler refuses
to instantiate it at the call.
"""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads, parse
from quivers.dsl.ast_nodes import (
    CallStep,
    DrawArgName,
    DrawArgScalar,
    LetExprCall,
    LetExprFactor,
    LetExprLambda,
    LetExprNode,
    LetExprUnit,
    LetExprVar,
    LetStep,
    ProgramDecl,
    ProgramStep,
)
from quivers.dsl.compiler import Compiler
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.let_expr_traversal import map_let_expr, substitute_let_expr

HEAD = """
object Row : FinSet 4
object Group : FinSet 2
object K : FinSet 3
object Val : Real 1
"""

COMPUTATIONS = """
define bump(x : Real) : Real !{} =
    return x + 1.0
"""

# The let forms the scoring comparison runs, over the template's
# parameters `tau : Real` and `G : FinSet`: indexing by host data, by
# bound indices, and by a nested index; both factor forms; calls; and
# arithmetic with a lambda-free body.
SCORED_LETS = """
    sample effect : G <- Normal(loc=0.0, scale=tau)
    sample w : G <- Normal(loc=0.0, scale=1.0) [over=K]
    let row_effect = effect[group_idx]
    let doubled = factor g : G in effect[g] * tau
    let permuted = factor g : G in effect[int(perm[g])]
    let both = factor g : G, k : K in w[g, k] * tau
    let cases = factor g : Group in { 0 -> effect[0], 1 -> -effect[1] }
    let called = exp(effect)
    let arith = row_effect - doubled[0] / 2.0 + both[1, 2] + cases[1] + called[0] + permuted[0]
"""

# The scored forms without host data, for a draw from the template.
DRAWN_LETS = """
    sample effect : G <- Normal(loc=0.0, scale=tau)
    sample w : G <- Normal(loc=0.0, scale=1.0) [over=K]
    let doubled = factor g : G in effect[g] * tau
    let both = factor g : G, k : K in w[g, k] * tau
    let cases = factor g : Group in { 0 -> effect[0], 1 -> -effect[1] }
    let called = exp(effect)
    let arith = 0.5 - doubled[0] / 2.0 + both[1, 2] + cases[1] + called[0]
"""

# Every let-expression form the grammar has, over the same parameters. The instantiated right-hand sides
# are compared structurally with the hand-inlined program's.
EVERY_FORM_LETS = """
    sample effect : G <- Normal(loc=0.0, scale=tau)
    sample w : G <- Normal(loc=0.0, scale=1.0) [over=K]
    let row_effect = effect[group_idx]
    let doubled = factor g : G in effect[g] * tau
    let permuted = factor g : G in effect[int(perm[g])]
    let both = factor g : G, k : K in w[g, k] * tau
    let cases = factor g : Group in { 0 -> effect[0], 1 -> -effect[1] }
    let called = exp(effect)
    let method = effect.sum()
    let listed = [effect[0], tau, method]
    let paired = (effect[0], tau)
    let flags = (true, unit, "label")
    let mapped = map(cases, e -> e * tau)
    let folded = fold(mapped, 0.0, acc -> e -> acc + e)
    let compared = effect[0] > tau && not (effect[1] <= 0.0) || tau == 1.0
    let shadow = map(cases, tau -> tau * 2.0)
    let arith = -row_effect + 0.1 * folded - doubled[0] / 2.0 % 3.0 + both[1, 2] + cases[1] + called[0] + permuted[0] + shadow[1] - method * 0.25
"""

OBSERVE = """
    observe y : Row <- Normal(loc=arith, scale=1.0)
    return y
"""

TAU = 0.7


def _template(lets: str) -> str:
    return (
        HEAD
        + COMPUTATIONS
        + "\nprogram full(tau : Real, G : FinSet) : Row -> Val"
        + lets
        + OBSERVE
        + "\nexport full\n"
    )


def _inlined(lets: str) -> str:
    # The lambda parameter `tau` shadows the template's parameter, so
    # the hand-inlined program spells it with a fresh name.
    body = (
        lets.replace("tau -> tau", "t -> t")
        .replace("tau", repr(TAU))
        .replace(": G ", ": Group ")
        .replace(": G,", ": Group,")
    )
    return (
        HEAD
        + COMPUTATIONS
        + "\nprogram full : Row -> Val"
        + body
        + OBSERVE
        + "\nexport full\n"
    )


def _drawn(lets: str) -> str:
    return (
        HEAD
        + COMPUTATIONS
        + "\nprogram part(tau : Real, G : FinSet) : Row -> Val"
        + lets
        + "\n    return arith\n"
        + "\nprogram caller : Row -> Val\n"
        + f"    sample a <- part({TAU}, Group)\n"
        + OBSERVE.replace("loc=arith", "loc=a")
        + "\nexport caller\n"
    )


def _sites(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "effect": torch.randn(2, generator=generator),
        "w": torch.randn(2, 3, generator=generator),
        "y": torch.randn(4, generator=generator),
        "group_idx": torch.tensor([0, 1, 1, 0]),
        "perm": torch.tensor([1.0, 0.0]),
    }


def test_an_instantiated_template_scores_as_the_inlined_program() -> None:
    instantiated = loads(_template(SCORED_LETS)).full(tau=TAU, G="Group").morphism
    inlined = loads(_inlined(SCORED_LETS)).morphism
    x = torch.zeros(1, 1)
    for seed in range(4):
        sites = _sites(seed)
        expected = inlined.log_joint(x, sites)
        actual = instantiated.log_joint(x, sites)
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected)


def test_a_draw_from_a_template_scores_as_the_inlined_program() -> None:
    drawn = loads(_drawn(DRAWN_LETS)).morphism
    inlined = loads(_inlined(DRAWN_LETS)).morphism
    x = torch.zeros(1, 1)
    for seed in range(4):
        sites = _sites(seed)
        renamed = dict(sites)
        renamed["a$effect"] = renamed.pop("effect")
        renamed["a$w"] = renamed.pop("w")
        expected = inlined.log_joint(x, sites)
        actual = drawn.log_joint(x, renamed)
        torch.testing.assert_close(actual, expected)


def _strip_positions(expr: LetExprNode) -> LetExprNode:
    def strip(node: LetExprNode) -> LetExprNode:
        if isinstance(node, LetExprFactor):
            node = node.with_(
                binders=tuple(
                    binder.with_(line=0, col=0, index=binder.index.with_(line=0, col=0))
                    for binder in node.binders
                ),
                cases=tuple(case.with_(line=0, col=0) for case in node.cases),
            )
        return node.with_(line=0, col=0)

    lambdas = iter(range(1_000))

    def canonical_lambda(node: LetExprNode) -> LetExprNode:
        # Name each lambda parameter by its position, so that
        # alpha-equivalent expressions compare equal.
        if not isinstance(node, LetExprLambda):
            return node
        name = f"lambda_{next(lambdas)}"
        body = substitute_let_expr(node.body, {node.param: LetExprVar(name=name)})
        return node.with_(param=name, body=body)

    return map_let_expr(map_let_expr(expr, canonical_lambda), strip)


def _bound_values(steps: tuple[ProgramStep, ...]) -> dict[str, LetExprNode]:
    values: dict[str, LetExprNode] = {}
    for step in steps:
        if isinstance(step, LetStep):
            values[step.name] = _strip_positions(step.value)
        elif isinstance(step, CallStep):
            values[step.name] = _strip_positions(
                LetExprCall(func=step.call.callee, args=step.call.arguments)
            )
    return values


def test_instantiation_substitutes_through_every_let_expression_form() -> None:
    compiler = Compiler(parse(_template(EVERY_FORM_LETS)), module_name="t")
    compiler.compile()
    template = compiler._program_templates["full"]
    call_site = LetStep(name="a", value=LetExprUnit())
    steps = compiler._instantiate_template(
        template,
        "a",
        (DrawArgScalar(value=TAU), DrawArgName(text="Group")),
        call_site,
    )
    actual = _bound_values(steps)

    inlined = parse(_inlined(EVERY_FORM_LETS))
    program = next(d for d in inlined.statements if isinstance(d, ProgramDecl))
    expected_raw = _bound_values(program.draws)
    locals_ = {"effect", "w", "y", *expected_raw}
    to_caller = {name: LetExprVar(name=f"a${name}") for name in locals_}
    expected = {
        f"a${name}": _strip_positions(substitute_let_expr(value, to_caller))
        for name, value in expected_raw.items()
    }

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == value, name


def test_calling_a_numeric_parameter_is_a_located_compile_error() -> None:
    source = (
        HEAD
        + """
program bad(tau : Real) : Row -> Val
    sample effect : Group <- Normal(loc=0.0, scale=1.0)
    let called = tau(effect)
    observe y : Row <- Normal(loc=called, scale=1.0)
    return y

export bad
"""
    )
    with pytest.raises(CompileError, match="'tau' is a parameter that takes"):
        loads(source).bad(tau=TAU)


CALLING_TEMPLATE = (
    HEAD
    + COMPUTATIONS
    + """
program part(tau : Real) : Row -> Val
    sample effect : Group <- Normal(loc=0.0, scale=tau)
    let bumped <- traverse(effect, e -> bump(e))
    let first = bumped[0]
    observe y : Row <- Normal(loc=first, scale=1.0)
    return y
"""
)


def _part_oracle(effect: torch.Tensor, y: torch.Tensor, tau: float) -> float:
    """The joint of ``part``: the effects' prior and ``y`` around the
    first bumped effect."""
    prior = torch.distributions.Normal(0.0, tau).log_prob(effect).sum()
    first = effect[0] + 1.0
    return float(prior + torch.distributions.Normal(first, 1.0).log_prob(y).sum())


def test_instantiating_a_program_that_traverses_a_pure_computation() -> None:
    """The traverse maps ``bump`` over the effects as a comprehension."""
    source = CALLING_TEMPLATE + "\nexport part\n"
    program = loads(source).part(tau=TAU).morphism
    effect = torch.tensor([0.3, -0.8])
    y = torch.tensor([1.0, 1.5, 0.9, 1.2])
    joint = program.log_joint(torch.zeros(1, 1), {"effect": effect, "y": y})
    assert float(joint.sum()) == pytest.approx(_part_oracle(effect, y, TAU), rel=1e-5)
    # Reading the effect itself rather than its bump moves the observation's
    # mean by one, which the tolerance rejects.
    wrong = float(
        torch.distributions.Normal(0.0, TAU).log_prob(effect).sum()
        + torch.distributions.Normal(effect[0], 1.0).log_prob(y).sum()
    )
    assert float(joint.sum()) != pytest.approx(wrong, rel=1e-3)


def test_drawing_a_program_that_traverses_a_pure_computation() -> None:
    source = (
        CALLING_TEMPLATE.replace(
            "    observe y : Row <- Normal(loc=first, scale=1.0)\n    return y",
            "    return first",
        )
        + """
program caller : Row -> Val
    sample a <- part(0.7)
    observe y : Row <- Normal(loc=a, scale=1.0)
    return y

export caller
"""
    )
    program = loads(source).morphism
    effect = torch.tensor([0.3, -0.8])
    y = torch.tensor([1.0, 1.5, 0.9, 1.2])
    joint = program.log_joint(torch.zeros(1, 1), {"a$effect": effect, "y": y})
    assert float(joint.sum()) == pytest.approx(_part_oracle(effect, y, TAU), rel=1e-5)
