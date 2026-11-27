"""The same acyclic grammar partition through search, agenda, and contraction.

The only facts are three lexical rows. A unary inference moves from V to
S on the same span, and a binary inference increases span width. This
orders every ground inference and yields exactly the two ordered binary
trees on three positions. Each outer a uses one of two lexical rows, and
the inner b has only V. Every legal contraction assignment thus names one
of the eight proof occurrences. Its factors are the weight of each
occurrence in that proof. Distributivity gives the same sum when the
agenda aggregates an item before propagating it, or the contraction
eliminates an assignment coordinate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from functools import partial

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.wiring import EinsumWiring
from quivers.dsl.compiler.deductions import BASE_WEIGHT_KEY
from quivers.qiec.foreign import SemiringProduct, SemiringSum
from quivers.qiec.module import reached_handlers
from quivers.qiec.program_runtime import run_deduction
from quivers.stochastic.agenda import HashChart
from tests.evidence.checks._registry import add
from tests.evidence.compare import (
    Accuracy,
    compare_gradients,
    compare_tensor,
    differentiate,
)
from tests.evidence.oracles.partition import PROOFS, partition
from tests.evidence.problems import logarithm
from tests.evidence.sources import deduction, program

SENTENCE = ("a", "b", "a")
NAMES = ("a_s", "b_v", "a_v", "join", "lift")
# The fourth point has syntactic proofs but every one has zero weight.
POINTS = (
    (
        "unit",
        (Fraction(1),) * len(NAMES),
    ),
    (
        "positive",
        (
            Fraction(3, 5),
            Fraction(2, 5),
            Fraction(1, 4),
            Fraction(7, 10),
            Fraction(1, 3),
        ),
    ),
    (
        "updated",
        (
            Fraction(1, 7),
            Fraction(4, 5),
            Fraction(3, 5),
            Fraction(1, 2),
            Fraction(4, 7),
        ),
    ),
    (
        "zero branch",
        (Fraction(3, 10), Fraction(1, 2), Fraction(0), Fraction(4, 5), Fraction(3, 7)),
    ),
    (
        "zero mass",
        (Fraction(0), Fraction(2, 5), Fraction(0), Fraction(7, 10), Fraction(1, 3)),
    ),
)

# A proof contains at most eight factors. Charge 20 roundings per factor
# or binary alternative, including encoding, stable log-domain sums and
# their derivatives, and charge a complete traversal for each proof.
ROUNDINGS = 20 * len(PROOFS) * (8 + len(PROOFS))

type LogWeights = Mapping[str, torch.Tensor]
type Evaluation = tuple[torch.Tensor, LogWeights]
type Reader = Callable[[LogWeights], Evaluation]


def machine() -> Reader:
    """Reuse the module and its generated Search and Weight handlers."""
    module = program("weighted-lexical", SEMIRING="LogProb").qiec
    assert module is not None
    entry = next(
        computation
        for computation in module.computations
        if computation.name == "G__run"
    )
    installed = reached_handlers(module, entry.id)
    providers = {
        type(handler.foreign) for handler in module.handlers if handler.id in installed
    }
    assert {SemiringSum, SemiringProduct} <= providers

    def evaluate(weights: LogWeights) -> Evaluation:
        result = run_deduction(
            module,
            "G",
            tokens=SENTENCE,
            parameters={
                "G.lex.0": weights["a_s"],
                "G.lex.1": weights["b_v"],
                "G.lex.2": weights["a_v"],
                "G.rule.join:": weights["join"],
                "G.rule.lift:": weights["lift"],
            },
        )
        assert isinstance(result.weight, torch.Tensor)
        return result.weight, weights

    return evaluate


def agenda() -> Reader:
    """Reuse the grammar's FIFO agenda with trie joins and a HashChart."""
    system = deduction(program("weighted-lexical", SEMIRING="LogProb"), "G")
    assert system.chart_factory is HashChart
    for module in system.modules.values():
        module.double()
    parameters = dict(system.named_parameters())
    leaves = {
        "a_s": parameters["_axiom_module.lex_weight_0"],
        "b_v": parameters["_axiom_module.lex_weight_1"],
        "a_v": parameters["_axiom_module.lex_weight_2"],
        "join": parameters[f"_rule_module.rule_join.{BASE_WEIGHT_KEY}"],
        "lift": parameters[f"_rule_module.rule_lift.{BASE_WEIGHT_KEY}"],
    }

    def evaluate(weights: LogWeights) -> Evaluation:
        # Use the compiler's weight readers, without constructing a second
        # set of chart rules in the test. The backward pass connects their
        # product to these exact parameter coordinates.
        with torch.no_grad():
            for name, leaf in leaves.items():
                leaf.copy_(weights[name])
        return system(SENTENCE).goal_weight(), leaves

    return evaluate


def contraction() -> Reader:
    """Contract unaggregated lexical, unary, and binary rule factors."""
    wiring = EinsumWiring(LOG_PROB, "j,s,s,i,i,j,k,k->")

    def evaluate(weights: LogWeights) -> Evaluation:
        join = weights["join"].expand(2)
        unary = torch.stack((weights["lift"].new_zeros(()), weights["lift"]))
        a = torch.stack((weights["a_s"], weights["a_v"]))
        b = torch.stack((weights["b_v"].new_full((), -torch.inf), weights["b_v"]))
        # s chooses one of the two bracketings. i and k independently
        # choose the lexical category of the two a occurrences. j does
        # the same for b, for which the S row is structurally absent.
        # The two s factors represent the two binary nodes at either
        # bracketing. The i, j, k unary factors turn a V into an S.
        return wiring.apply(b, join, join, a, unary, unary, a, unary), weights

    return evaluate


def check_partition(make: Callable[[], Reader], executor: str) -> None:
    """Check each value and all five log-weight gradients against the oracle."""
    evaluate = make()
    for point, numbers in POINTS:
        weights = dict(zip(NAMES, numbers, strict=True))
        mass, gradient = partition(weights)
        tensors = {
            name: torch.tensor(
                logarithm(weight), dtype=torch.float64, requires_grad=True
            )
            for name, weight in weights.items()
        }
        answer, leaves = evaluate(tensors)
        label = f"{executor} {point}"
        expected = torch.tensor(logarithm(mass), dtype=torch.float64)
        compare_tensor(
            answer,
            expected,
            Accuracy(domain="log", operations=ROUNDINGS, magnitude=32.0),
            label,
        )
        if gradient is not None:
            found = differentiate(answer, list(leaves.values()), label)
            compare_gradients(
                dict(zip(leaves, found, strict=True)),
                gradient,
                Accuracy(domain="linear", operations=ROUNDINGS),
                label,
            )


for reader, row in (
    (machine, "machine-lexical"),
    (agenda, "agenda-fifo"),
    (contraction, "einsum-wiring"),
):
    add(f"partition/{row}", row, "Q1g", partial(check_partition, reader, row))
