"""Checks and deliberately wrong versions of directed spanning trees.

The checks compare the elimination with enumeration of every tree (O4),
with exact rational arithmetic (O5), and with derivatives of the
enumerated partition (O7), on asymmetric weights, exact zeros, impossible
graphs, empty batches, and extreme log weights; one more counts the
scalar work on graphs of growing size against ``n^3``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import torch

from quivers.core.algebras import LOG_PROB, MARKOV, MAX_PLUS
from quivers.core.spanning import tree_marginals, tree_partition
from quivers.qiec.contracts import (
    ALGEBRA_OPERATIONS,
    BestKnown,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    Phase,
    PhaseInapplicable,
    Unresolved,
)
from quivers.core.laws import Citation
from tests.evidence import scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.counters import measure
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import exact, trees

SPANNING = "quivers.core.spanning"

WEIGHTS = (
    (0, "3/2", "1/7", "2/5"),
    ("5/3", 0, "1/4", "7/2"),
    ("2/9", "6/5", 0, "1/3"),
    ("4/3", "1/8", "9/4", 0),
)
"""Asymmetric rational edge weights of four nodes; row is the parent."""

ROOT = 1
"""The root of the checks."""


def rational_weights(
    table: tuple[tuple[str | int, ...], ...],
) -> dict[tuple[int, int], Fraction]:
    """The weights of every edge other than a self edge.

    Parameters
    ----------
    table : tuple of tuple of str or int
        The weights by parent and child.

    Returns
    -------
    dict of tuple of int and int to Fraction
        The weight of each edge.
    """
    size = len(table)
    return {
        (x, y): Fraction(table[x][y])
        for x in range(size)
        for y in range(size)
        if x != y
    }


def tensor_of(weights: dict[tuple[int, int], Fraction], size: int) -> torch.Tensor:
    """The float64 matrix of rational weights.

    Parameters
    ----------
    weights : dict of tuple of int and int to Fraction
        The weights.
    size : int
        The number of nodes.

    Returns
    -------
    torch.Tensor
        The matrix, zero on the diagonal.
    """
    matrix = torch.zeros(size, size, dtype=torch.float64)
    for (x, y), value in weights.items():
        matrix[x, y] = float(value)
    return matrix


def _require(condition: bool, what: str) -> None:
    """Fail a check with an infinite deviation.

    Parameters
    ----------
    condition : bool
        What has to hold.
    what : str
        The failure.

    Raises
    ------
    Mismatch
        If the condition does not hold.
    """
    if not condition:
        raise Mismatch("value", math.inf, what)


def _run(body: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Run the library and turn a raised error into a mismatch.

    Parameters
    ----------
    body : Callable[[], torch.Tensor]
        The computation.

    Returns
    -------
    torch.Tensor
        Its result.

    Raises
    ------
    Mismatch
        If it raises.
    """
    try:
        return body()
    except (ZeroDivisionError, ValueError, RuntimeError) as error:
        raise Mismatch("refusal", math.inf, f"the library raised: {error}") from error


def check_partition() -> None:
    """Require the partition of asymmetric weights to be the enumerated one.

    Raises
    ------
    Mismatch
        If the linear or log partition differs from O4 and O5.
    """
    weights = rational_weights(WEIGHTS)
    enumerated = trees.partition(
        trees.spanning_trees(4, ROOT), weights, Fraction(0), Fraction(1)
    )
    _require(
        enumerated == exact.tree_partition(4, ROOT, weights),
        "the two oracles disagree",
    )
    matrix = tensor_of(weights, 4)
    linear = _run(lambda: tree_partition(matrix, ROOT, algebra=MARKOV))
    compare_number(
        float(linear), enumerated, Accuracy(domain="linear", operations=64), "linear"
    )
    logged = _run(lambda: tree_partition(matrix.log(), ROOT))
    compare_number(
        float(logged),
        math.log(enumerated),
        Accuracy(domain="log", operations=64),
        "log",
    )


def check_marginals() -> None:
    """Require the edge marginals to be the enumerated ones.

    Raises
    ------
    Mismatch
        If a marginal differs from the enumerated probability of its edge.
    """
    weights = rational_weights(WEIGHTS)
    expected = trees.edge_marginals(trees.spanning_trees(4, ROOT), weights)
    found = _run(lambda: tree_marginals(tensor_of(weights, 4).log(), ROOT))
    for (x, y), value in expected.items():
        compare_number(
            float(found[x, y]),
            value,
            Accuracy(domain="linear", operations=256),
            f"marginal of {x}->{y}",
        )


def check_zero_gradient() -> None:
    """Require the derivative at an exactly zero weight to be kept.

    With every edge into node 3 at zero the partition is zero, and its
    derivative in each of those weights is the partition of the trees
    that use the edge.

    Raises
    ------
    Mismatch
        If a derivative differs from the enumerated one.
    """
    weights = rational_weights(WEIGHTS)
    for parent in range(4):
        if parent != 3:
            weights[(parent, 3)] = Fraction(0)
    matrix = tensor_of(weights, 4).requires_grad_()
    partition = _run(lambda: tree_partition(matrix, ROOT, algebra=MARKOV))
    (gradient,) = torch.autograd.grad(partition, matrix)
    all_trees = trees.spanning_trees(4, ROOT)
    for parent in range(4):
        if parent == 3:
            continue
        using = [tree for tree in all_trees if (parent, 3) in tree]
        others = dict(weights)
        others[(parent, 3)] = Fraction(1)
        expected = trees.partition(using, others, Fraction(0), Fraction(1))
        compare_number(
            float(gradient[parent, 3]),
            expected,
            Accuracy(domain="linear", operations=256),
            f"derivative at the zero edge {parent}->3",
        )


def check_impossible() -> None:
    """Require a graph without a tree to give the zero with zero derivative.

    Raises
    ------
    Mismatch
        If the value is not the zero, a derivative is not zero, or an
        empty batch has another shape or dtype.
    """
    weights = torch.full((2, 3, 3), -math.inf, dtype=torch.float64)
    weights[1, 0, 1] = -0.5
    weights[1, 1, 2] = -0.25
    weights[0, 1, 2] = -1.0
    weights.requires_grad_()
    partition = _run(lambda: tree_partition(weights, 0))
    (gradient,) = torch.autograd.grad(partition.sum(), weights)
    _require(bool(torch.isneginf(partition[0])), f"no tree gave {partition[0]}")
    _require(float(partition[1]) == -0.75, f"one tree gave {partition[1]}")
    _require(bool(torch.isfinite(gradient).all()), "a derivative is not finite")
    _require(float(gradient[0].abs().sum()) == 0.0, "no tree has a derivative")
    empty = _run(lambda: tree_partition(torch.zeros(0, 4, 4, dtype=torch.float32), 0))
    _require(
        tuple(empty.shape) == (0,) and empty.dtype == torch.float32,
        f"an empty batch gave {tuple(empty.shape)} {empty.dtype}",
    )


def check_extreme_log_weights() -> None:
    """Require log weights to stay in log space.

    The graph of the module documentation: edges of log weight -60 and
    -61 from the root and a unit cycle; and one edge of log weight -1000.

    Raises
    ------
    Mismatch
        If either partition differs from its exact value.
    """
    weights = torch.full((3, 3), -math.inf, dtype=torch.float64)
    weights[0, 1], weights[0, 2] = -60.0, -61.0
    weights[1, 2], weights[2, 1] = 0.0, 0.0
    found = _run(lambda: tree_partition(weights, 0))
    exact_value = -60.0 + math.log1p(math.exp(-1.0) + math.exp(-61.0))
    compare_number(
        float(found), exact_value, Accuracy(domain="log", operations=16), "scaled"
    )
    single = torch.full((2, 2), -math.inf, dtype=torch.float64)
    single[0, 1] = -1000.0
    compare_number(
        float(_run(lambda: tree_partition(single, 0))),
        -1000.0,
        Accuracy(domain="log", operations=4),
        "one edge of log weight -1000",
    )


def check_best_tree() -> None:
    """Require the max-plus partition to be the best tree, and its derivative its edges.

    Raises
    ------
    Mismatch
        If the best weight or its edges differ from enumeration.
    """
    weights = rational_weights(WEIGHTS)
    logged = tensor_of(weights, 4).log()
    best = max(
        trees.spanning_trees(4, ROOT),
        key=lambda tree: sum(math.log(weights[edge]) for edge in tree),
    )
    value = _run(lambda: tree_partition(logged, ROOT, algebra=MAX_PLUS))
    compare_number(
        float(value),
        sum(math.log(weights[edge]) for edge in best),
        Accuracy(domain="log", operations=64),
        "best tree",
    )
    marked = _run(lambda: tree_marginals(logged, ROOT, algebra=MAX_PLUS))
    chosen = {(x, y) for x in range(4) for y in range(4) if float(marked[x, y]) == 1.0}
    _require(chosen == set(best), f"marked {chosen}, best {set(best)}")


def elimination_work(size: int) -> int:
    """The scalar work of one partition of a dense graph.

    Parameters
    ----------
    size : int
        The number of nodes.

    Returns
    -------
    int
        Scalar operations.
    """
    weights = torch.zeros(size, size, dtype=torch.float64)
    return measure(lambda: tree_partition(weights, 0, algebra=LOG_PROB)).scalar


WORK = scaling.Family(
    name="spanning trees by elimination",
    executor="spanning",
    question="Q19",
    varies="nodes",
    part="forward",
    sizes=(4, 8, 16, 32, 64),
    work=elimination_work,
    claim=lambda size: float(size**3),
    constant=16.0,
)
"""The family of dense graphs whose work is bounded by ``n^3``."""


def check_work() -> None:
    """Require the work to stay below ``c n^3`` and a control of ``n^4`` to fail.

    Raises
    ------
    Mismatch
        If the family breaks its bound, or the control does not.
    """
    scaling.check_family(WORK)


FOMIN = Citation(
    reference=(
        "Fomin, Grigoriev and Koshevoy 2016, Subtraction-free complexity, cluster "
        "transformations, and spanning trees"
    ),
    locator="abstract",
    doi="10.1007/s10208-014-9231-y",
)
"""The subtraction-free circuits for spanning trees."""

SPANNING_TREE_WORK = ComplexityClaim(
    query="partition and edge marginals of directed spanning trees",
    parameters=(
        ComplexityParameter(name="n", meaning="nodes"),
        ComplexityParameter(name="B", meaning="graphs in the batch"),
    ),
    model=ALGEBRA_OPERATIONS,
    bounds=(
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="time",
            expression="B n^3",
            status=Unresolved(
                obligation=(
                    "subtraction-free circuits of size n^3 exist and circuits with "
                    "sums and products only are exponential; with subtraction the "
                    "determinant costs n^omega but loses relative accuracy; whether "
                    "a subtraction-free circuit below n^3 exists is open"
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.EXECUTION,
            resource="space",
            expression="B n^2",
            status=BestKnown(sources=(FOMIN,)),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="time",
            expression="B n^3",
            status=BestKnown(
                sources=(
                    Citation(
                        reference="Baur and Strassen 1983, The complexity of partial derivatives",
                        locator="the cheap gradient theorem",
                        doi="10.1016/0304-3975(83)90110-X",
                    ),
                )
            ),
        ),
        ComplexityBound(
            phase=Phase.DIFFERENTIATION,
            resource="space",
            expression="B n^3",
            status=Unresolved(
                obligation="a recomputing backward pass in B n^2 space is not written"
            ),
        ),
    ),
    inapplicable=(
        PhaseInapplicable(
            phase=Phase.CONSTRUCTION,
            reason="nothing is built before the elimination; the weights are the input",
        ),
        PhaseInapplicable(
            phase=Phase.REFIT,
            reason="a refit is one more partition at the new weights, as execution",
        ),
    ),
    conditions=(
        "a zero-sum-free semifield with scalar elements; operations of unit cost",
    ),
)
"""The cost claim of elimination."""


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="spanning", question="Q19", run=run)
    for name, run in (
        ("spanning/partition", check_partition),
        ("spanning/marginals", check_marginals),
        ("spanning/zero-gradient", check_zero_gradient),
        ("spanning/impossible", check_impossible),
        ("spanning/extreme-log-weights", check_extreme_log_weights),
        ("spanning/best-tree", check_best_tree),
        ("spanning/work", check_work),
    )
}
"""Every check of directed spanning trees by name."""

TARGET = f"{SPANNING}:tree_partition"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="tree-orientation-swapped",
        defect_class="orientation",
        provenance="the incoming weight of a node is read from its row, as for a child-to-parent matrix",
        target=TARGET,
        also_in=("tests.evidence.spanning_controls",),
        rewrites=(
            Rewrite(
                old="pivots = algebra.join(current, -2)",
                new="pivots = algebra.join(current, -1)",
            ),
        ),
        rejected_by=("spanning/partition", "spanning/marginals"),
        min_effect=0.01,
    ),
    Mutation(
        name="tree-support-from-current-numbers",
        defect_class="support",
        provenance="edges whose weight is zero now are removed as if they were absent",
        target=TARGET,
        also_in=("tests.evidence.spanning_controls",),
        rewrites=(
            Rewrite(
                old="current = torch.where(allowed, weights, zero)",
                new="current = torch.where(allowed & ~algebra.is_zero(weights), weights, zero)",
            ),
        ),
        rejected_by=("spanning/zero-gradient",),
        min_effect=0.1,
    ),
    Mutation(
        name="tree-pivot-not-guarded",
        defect_class="empty reduction",
        provenance="a zero pivot is divided by when there is no tree",
        target=TARGET,
        also_in=("tests.evidence.spanning_controls",),
        rewrites=(
            Rewrite(
                old="algebra.is_zero(pivot), algebra.unit_like(batch, weights), pivot",
                new="algebra.is_zero(pivot), pivot, pivot",
            ),
        ),
        rejected_by=("spanning/impossible",),
        min_effect=1.0,
    ),
    Mutation(
        name="tree-log-weights-exponentiated",
        defect_class="log space",
        provenance="log weights are taken out of log space and back",
        target=TARGET,
        also_in=("tests.evidence.spanning_controls",),
        rewrites=(
            Rewrite(
                old="current = torch.where(allowed, weights, zero)",
                new="current = torch.where(allowed, weights.exp().log(), zero)",
            ),
        ),
        rejected_by=("spanning/extreme-log-weights",),
        min_effect=1.0,
    ),
)
"""Every wrong version of directed spanning trees and the checks that reject it."""
