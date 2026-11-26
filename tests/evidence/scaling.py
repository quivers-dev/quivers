"""Checks of work bounds on families of growing problems.

A family has at least five sizes over a factor of at least sixteen. Its
counter divided by the claimed bound has to stay below a stated constant
and must not grow beyond the second size by more than a tenth. Every
check also has to reject a control whose work is the same counter times
the size, the next higher order, so the check can fail.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.plate import marginalize_grouped
from quivers.continuous.programs import MonadicProgram
from quivers.core.algebras import LogProbAlgebra
from quivers.core.morphisms import observed
from quivers.core.objects import FinSet, ProductSet
from quivers.core.wiring import EinsumWiring
from quivers.dsl import loads
from quivers.stochastic._kleene import kleene_star
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.agenda import (
    AgendaResult,
    HashChart,
    InferenceRule,
    make_wildcard,
    run_agenda,
)
from quivers.stochastic.inside import InsideAlgorithm
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.semiring import LOG_PROB, VITERBI
from quivers.inference.mcmc import PotentialFn
from tests.evidence.checks import terms
from tests.evidence.checks.consumers import PROBLEMS, positions, registry_of
from tests.evidence.checks.span import double_precision
from tests.evidence.checks.deduction import geometric_run, weighted
from tests.evidence.compare import Mismatch
from tests.evidence.counters import (
    CallCounter,
    TensorCounter,
    forwarding,
    measure,
    memory,
)
from tests.evidence.sources import HMM


class Family(ValueModel):
    """A family of problems with a claimed bound.

    Parameters
    ----------
    name : str
        The family.
    executor : str
        The executor.
    question : str
        The problem class.
    varies : str
        What grows.
    part : str
        What is counted: ``construction``, ``validation``, ``forward``,
        ``backward``, ``saved``, ``provenance``, ``memory``, or ``refit``.
    sizes : tuple of int
        The sizes.
    work : Callable[[int], float]
        The counter at a size.
    claim : Callable[[int], float]
        The bound at a size, without its constant.
    constant : float
        The constant.
    slow : bool
        Whether the family runs only with the slow tests.
    """

    name: str
    executor: str
    question: str
    varies: str
    part: str
    sizes: tuple[int, ...]
    work: Callable[[int], float] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    claim: Callable[[int], float] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    constant: float
    slow: bool = False


def accept(
    sizes: Sequence[int],
    work: Sequence[float],
    claim: Sequence[float],
    constant: float,
    what: str,
) -> None:
    """Require counters to stay below a bound and their ratio not to grow.

    Parameters
    ----------
    sizes : Sequence of int
        The sizes.
    work : Sequence of float
        The counters.
    claim : Sequence of float
        The bounds without the constant.
    constant : float
        The constant.
    what : str
        A description for the report.


    Raises
    ------
    Mismatch
        If the sizes do not span enough, a ratio exceeds the constant, or
        a ratio beyond the second size grows by more than a tenth.
    """
    if len(sizes) < 5 or max(sizes) < 16 * min(sizes):
        raise Mismatch("support", math.inf, f"{what}: sizes {tuple(sizes)}")
    ratios = [counted / bound for counted, bound in zip(work, claim, strict=True)]
    if max(ratios) > constant:
        raise Mismatch(
            "work", max(ratios) / constant, f"{what}: ratios {ratios} above {constant}"
        )
    for before, after in itertools.pairwise(ratios[1:]):
        if after > 1.1 * before:
            raise Mismatch("work", after / before, f"{what}: ratios {ratios} grow")


def check_family(family: Family) -> None:
    """Check a family and its control of the next order.

    Parameters
    ----------
    family : Family
        The family.


    Raises
    ------
    Mismatch
        If the family breaks its bound, or the control does not.
    """
    work = [family.work(size) for size in family.sizes]
    claim = [family.claim(size) for size in family.sizes]
    accept(family.sizes, work, claim, family.constant, family.name)
    worse = [counted * size for counted, size in zip(work, family.sizes, strict=True)]
    try:
        accept(family.sizes, worse, claim, math.inf, family.name)
    except Mismatch:
        return
    raise Mismatch(
        "support", math.inf, f"{family.name}: the check accepts the next order"
    )


def _scalar(value: float) -> torch.Tensor:
    """A float64 scalar.

    Parameters
    ----------
    value : float
        The value.

    Returns
    -------
    torch.Tensor
        The scalar.
    """
    return torch.tensor(value, dtype=torch.float64)


def line(length: int) -> tuple[AgendaResult, dict[str, int]]:
    """Reachability along a line of edges.

    Parameters
    ----------
    length : int
        The number of edges.

    Returns
    -------
    tuple of (AgendaResult, dict of str to int)
        The run and its scalar sums and products.
    """
    semiring, counts = forwarding(LOG_PROB)
    i, j = make_wildcard("i"), make_wildcard("j")
    result = run_agenda(
        [
            (("reach", 0), _scalar(0.0)),
            *((("edge", k, k + 1), _scalar(-0.1)) for k in range(length)),
        ],
        [
            InferenceRule(
                name="step",
                premises=(("reach", i), ("edge", i, j)),
                conclusion=("reach", j),
            )
        ],
        semiring=semiring,
    )
    return result, counts


def skips(
    length: int, reach: int, leaf: torch.Tensor | None = None
) -> tuple[AgendaResult, dict[str, int]]:
    """Steps of bounded length along a line.

    Parameters
    ----------
    length : int
        The number of positions after the first.
    reach : int
        The longest step.
    leaf : torch.Tensor or None
        The weight every step shares.

    Returns
    -------
    tuple of (AgendaResult, dict of str to int)
        The run and its scalar sums and products.
    """
    semiring, counts = forwarding(LOG_PROB)
    weight = _scalar(-1.0) if leaf is None else leaf
    result = run_agenda(
        [(("n", 0), _scalar(0.0))],
        [
            InferenceRule(
                name=f"{k}+{d}",
                premises=(("n", k),),
                conclusion=("n", k + d),
                weight_fn=weighted(weight),
            )
            for k in range(length)
            for d in range(1, reach + 1)
            if k + d <= length
        ],
        semiring=semiring,
    )
    return result, counts


def skip_edges(length: int, reach: int) -> int:
    """The number of steps in a line of bounded steps.

    Parameters
    ----------
    length : int
        The number of positions after the first.
    reach : int
        The longest step.

    Returns
    -------
    int
        The number of edges.
    """
    return sum(min(reach, length - k) for k in range(length))


def _total(counts: dict[str, int]) -> int:
    """All scalar operations of a counting semiring.

    Parameters
    ----------
    counts : dict of str to int
        The counts.

    Returns
    -------
    int
        Their sum.
    """
    return sum(counts.values())


def _exact_claim(length: int, reach: int) -> int:
    """The semiring operations of one relaxation of every edge.

    Every edge multiplies once, its weight or its two premises, and every
    contribution to an item other than its first adds once.

    Parameters
    ----------
    length : int
        The number of positions after the first.
    reach : int
        The longest step.

    Returns
    -------
    int
        The number of operations.
    """
    edges = skip_edges(length, reach)
    return edges + edges - length


def skip_backward(length: int) -> int:
    """The scalar work of the gradient of a skip run.

    Parameters
    ----------
    length : int
        The length.

    Returns
    -------
    int
        Scalar operations of the backward pass alone.
    """
    leaf = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    result, _ = skips(length, 2, leaf)
    value = result.chart.get(("n", length))
    assert value is not None
    with TensorCounter() as counter:
        torch.autograd.grad(value, leaf)
    counts = counter.counts
    return (
        counts["sums"] + counts["products"] + counts["divisions"] + counts["elementary"]
    )


def skip_saved(length: int) -> int:
    """The elements a skip run saves for its gradient.

    Parameters
    ----------
    length : int
        The length.

    Returns
    -------
    int
        The elements.
    """
    leaf = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
    return measure(lambda: _last(skips(length, 2, leaf)[0], length)).saved


def _last(result: AgendaResult, length: int) -> torch.Tensor:
    """The weight of the last position.

    Parameters
    ----------
    result : AgendaResult
        The run.
    length : int
        The last position.

    Returns
    -------
    torch.Tensor
        The weight.
    """
    value = result.chart.get(("n", length))
    assert value is not None
    return value


def dense_matrix(size: int, *, requires_grad: bool = False) -> torch.Tensor:
    """A dense log matrix whose powers sum.

    Parameters
    ----------
    size : int
        The order.
    requires_grad : bool
        Whether it is differentiated.

    Returns
    -------
    torch.Tensor
        The matrix.
    """
    return torch.full(
        (size, size),
        -math.log(2.0 * size),
        dtype=torch.float64,
        requires_grad=requires_grad,
    )


def star_backward(size: int) -> int:
    """The scalar work of the gradient of a closure.

    Parameters
    ----------
    size : int
        The order.

    Returns
    -------
    int
        Scalar operations of the backward pass alone.
    """
    matrix = dense_matrix(size, requires_grad=True)
    value = kleene_star(matrix, LOG_PROB).sum()
    return measure(lambda: torch.autograd.grad(value, matrix)[0]).scalar


def best_path_closure(size: int) -> int:
    """The scalar work of a best-path closure on a line of states.

    Parameters
    ----------
    size : int
        The number of states.

    Returns
    -------
    int
        Sums, products and comparisons.
    """
    matrix = torch.full((size, size), -math.inf, dtype=torch.float64)
    for state in range(size - 1):
        matrix[state, state + 1] = -1.0
    return measure(lambda: kleene_star(matrix, VITERBI)).scalar


def star_near_one(turns: int) -> int:
    """The scalar work of a closure whose sum grows.

    Parameters
    ----------
    turns : int
        The value of the sum.

    Returns
    -------
    int
        Scalar operations.
    """
    matrix = torch.full((4, 4), -math.inf, dtype=torch.float64)
    matrix[0, 0] = math.log1p(-1.0 / turns)
    return measure(lambda: kleene_star(matrix, LOG_PROB)).scalar


def cycle_steps(turns: int) -> int:
    """The scalar sums of an agenda run around a cycle.

    Parameters
    ----------
    turns : int
        The value of the sum.

    Returns
    -------
    int
        Dispatched scalar operations.
    """
    return measure(lambda: geometric_run(1 - Fraction(1, turns), "fifo")).scalar


def tolerance_steps(digits: int) -> int:
    """The steps of an agenda run that stops on a residual.

    Parameters
    ----------
    digits : int
        The number of binary digits of the tolerance.

    Returns
    -------
    int
        The steps.
    """
    return run_agenda(
        [(("A",), _scalar(0.0))],
        [
            InferenceRule(
                name="turn",
                premises=(("A",),),
                conclusion=("A",),
                weight_fn=weighted(_scalar(math.log(0.5))),
            )
        ],
        semiring=LOG_PROB,
        chart=HashChart(tolerance=2.0**-digits),
    ).iterations


def matrix_product(size: int) -> int:
    """The scalar work of the product of two matrices.

    Parameters
    ----------
    size : int
        The order.

    Returns
    -------
    int
        Scalar operations.
    """
    wiring = EinsumWiring(LogProbAlgebra(), "ij,jk->ik")
    matrix = torch.zeros(size, size, dtype=torch.float64)
    return measure(lambda: wiring.apply(matrix, matrix)).scalar


def skewed_triangle(size: int) -> int:
    """The scalar work of a triangle over relations with one hub.

    The tensors are dense, so the work does not see the skew.

    Parameters
    ----------
    size : int
        The number of values.

    Returns
    -------
    int
        Scalar operations.
    """
    wiring = EinsumWiring(LogProbAlgebra(), "ij,jk,ki->")
    hub = torch.full((size, size), -math.inf, dtype=torch.float64)
    hub[0, :] = 0.0
    hub[:, 0] = 0.0
    return measure(lambda: wiring.apply(hub, hub, hub)).scalar


def contraction_construction(count: int) -> int:
    """The work of building a wiring of many inputs.

    Parameters
    ----------
    count : int
        The number of inputs.

    Returns
    -------
    int
        Letters of the wiring the constructor reads.
    """
    letters = [chr(0x100 + k) for k in range(count + 1)]
    spec = (
        ",".join(letters[k] + letters[k + 1] for k in range(count))
        + "->"
        + letters[0]
        + letters[count]
    )
    wiring = EinsumWiring(LogProbAlgebra(), spec)
    return sum(len(part) for part in wiring.input_specs)


def parser_of(categories: int) -> ChartParser:
    """A chart parser with one binary rule per category.

    Parameters
    ----------
    categories : int
        The number of categories.

    Returns
    -------
    ChartParser
        The parser.
    """
    return ChartParser(
        RuleSystem(
            binary_rules=tuple((a, a, (a + 1) % categories) for a in range(categories)),
            n_categories=categories,
        ),
        n_terminals=2,
        start_idx=0,
    )


def parser_work(categories: int, batch: int, length: int) -> int:
    """The scalar work of one chart parse.

    Parameters
    ----------
    categories : int
        The number of categories.
    batch : int
        The number of sentences.
    length : int
        Their length.

    Returns
    -------
    int
        Scalar operations.
    """
    parser = parser_of(categories)
    tokens = torch.zeros(batch, length, dtype=torch.long)
    return measure(lambda: parser(tokens)).scalar


def parser_refit(length: int) -> int:
    """The scalar work of a parse after a parameter update.

    Parameters
    ----------
    length : int
        The sentence length.

    Returns
    -------
    int
        Scalar operations of the second parse.
    """
    parser = parser_of(3)
    tokens = torch.zeros(1, length, dtype=torch.long)
    parser(tokens)
    with torch.no_grad():
        for parameter in parser.parameters():
            parameter.add_(0.5)
    return measure(lambda: parser(tokens)).scalar


def parser_backward(length: int) -> int:
    """The scalar work of the gradient of a parse.

    Parameters
    ----------
    length : int
        The sentence length.

    Returns
    -------
    int
        Scalar operations of the backward pass alone.
    """
    parser = parser_of(3)
    value = parser(torch.zeros(1, length, dtype=torch.long)).sum()
    parameters = list(parser.parameters())
    return measure(lambda: torch.autograd.grad(value, parameters)[0]).scalar


def inside_work(categories: int, length: int) -> int:
    """The scalar work of the inside algorithm.

    Parameters
    ----------
    categories : int
        The number of categories.
    length : int
        The sentence length.

    Returns
    -------
    int
        Scalar operations.
    """
    objects = FinSet(name="N", cardinality=categories)
    words = FinSet(name="T", cardinality=2)
    algorithm = InsideAlgorithm(
        observed(
            objects,
            ProductSet(components=(objects, objects)),
            torch.full((categories,) * 3, 0.5 / categories**2),
        ),
        observed(objects, words, torch.full((categories, 2), 0.25)),
    )
    return measure(lambda: algorithm(torch.zeros(1, length, dtype=torch.long))).scalar


def grouped_work(rows: int, classes: int) -> int:
    """The scalar work of a grouped mixture.

    Parameters
    ----------
    rows : int
        The number of rows.
    classes : int
        The number of classes.

    Returns
    -------
    int
        Scalar operations.
    """
    scores = torch.zeros(rows, classes, dtype=torch.float64)
    index = torch.arange(rows) % 2
    prior = torch.zeros(classes, dtype=torch.float64)
    return measure(lambda: marginalize_grouped(scores, index, prior, 2)).scalar


def chain_work(states: int, length: int) -> int:
    """The scalar work of a summed-out chain.

    Parameters
    ----------
    states : int
        The number of states.
    length : int
        The number of symbols.

    Returns
    -------
    int
        Scalar operations.
    """
    source = HMM.replace("State : FinSet 2", f"State : FinSet {states}")
    if states not in _CHAINS:
        model = loads(source).morphism
        assert isinstance(model, MonadicProgram)
        _CHAINS[states] = model
    model = _CHAINS[states]
    sites = {
        "initial_row": torch.full((states,), 1.0 / states),
        "transition_rows": torch.full((states, states), 1.0 / states),
        "emission_rows": torch.full((states, 3), 1.0 / 3),
    }
    tokens = torch.zeros(1, length, dtype=torch.long)
    return measure(lambda: model.log_joint(tokens, sites)).scalar


_CHAINS: dict[int, MonadicProgram] = {}


def fold_work(depth: int) -> int:
    """The scalar work of a fold.

    Parameters
    ----------
    depth : int
        The number of nodes.

    Returns
    -------
    int
        Scalar operations.
    """
    encode = terms.encoder(terms.parameters())
    term = terms.asymmetric(depth)
    return measure(lambda: encode(term)).scalar


def records_per_run(rules: int) -> int:
    """The records one run of a fixed rule set makes.

    Parameters
    ----------
    rules : int
        The number of rules.

    Returns
    -------
    int
        Records validated by the second of two runs: the result and its
        work report, whatever the number of rules.
    """
    system = [
        InferenceRule(name=f"r{k}", premises=(("n", k),), conclusion=("n", k + 1))
        for k in range(rules)
    ]

    def once() -> AgendaResult:
        """Run the rules.

        Returns
        -------
        AgendaResult
            The run.
        """
        return run_agenda([(("n", 0), _scalar(0.0))], system, semiring=LOG_PROB)

    once()
    with CallCounter({"made": dx.Model._validate_instance}) as calls:
        once()
    return calls.counts["made"]


def potential_calls(calls: int) -> int:
    """The records made by repeated calls of a prepared potential.

    Parameters
    ----------
    calls : int
        The number of calls.

    Returns
    -------
    int
        Records validated by the calls.
    """
    problem = PROBLEMS["hmm"]
    with double_precision():
        registry = registry_of(problem)
        x, data = problem.data()
        potential = PotentialFn(problem.model(), registry, x, data)
        position = positions(registry)[1]
        potential.value_and_grad(position)
        with CallCounter({"made": dx.Model._validate_instance}) as counter:
            for _ in range(calls):
                potential.value_and_grad(position)
    return counter.counts["made"]


def retained(length: int) -> int:
    """The rule applications a run keeps.

    Parameters
    ----------
    length : int
        The length of a line.

    Returns
    -------
    int
        The number of records.
    """
    return len(line(length)[0].applications)


def star_memory(size: int) -> int:
    """The largest number of bytes live during a closure.

    Parameters
    ----------
    size : int
        The order.

    Returns
    -------
    int
        The bytes.
    """
    matrix = dense_matrix(size)
    return memory(lambda: kleene_star(matrix, LOG_PROB)).peak_bytes


FAMILIES: tuple[Family, ...] = (
    Family(
        name="agenda along a line",
        executor="agenda-fifo",
        question="Q10",
        varies="edges",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=lambda n: _total(line(n)[1]),
        claim=lambda n: _exact_claim(n, 1),
        constant=1.0,
    ),
    Family(
        name="agenda with bounded steps",
        executor="agenda-fifo",
        question="Q10",
        varies="positions",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=lambda n: _total(skips(n, 2)[1]),
        claim=lambda n: _exact_claim(n, 2),
        constant=1.0,
    ),
    Family(
        name="agenda with growing out-degree",
        executor="agenda-fifo",
        question="Q10",
        varies="sparsity",
        part="forward",
        sizes=(1, 2, 4, 8, 16),
        work=lambda d: _total(skips(20, d)[1]),
        claim=lambda d: _exact_claim(20, d),
        constant=1.0,
    ),
    Family(
        name="agenda gradient",
        executor="agenda-fifo",
        question="Q1g",
        varies="positions",
        part="backward",
        sizes=(4, 8, 16, 32, 64),
        work=skip_backward,
        claim=lambda n: skip_edges(n, 2) + n + 1,
        constant=12.0,
    ),
    Family(
        name="agenda saved tensors",
        executor="agenda-fifo",
        question="Q1g",
        varies="positions",
        part="saved",
        sizes=(4, 8, 16, 32, 64),
        work=skip_saved,
        claim=lambda n: skip_edges(n, 2) + n + 1,
        constant=4.0,
    ),
    Family(
        name="agenda provenance",
        executor="agenda-fifo",
        question="Q5",
        varies="edges",
        part="provenance",
        sizes=(4, 8, 16, 32, 64),
        work=retained,
        claim=lambda n: n,
        constant=1.0,
    ),
    Family(
        name="agenda around a cycle",
        executor="agenda-fifo",
        question="Q8",
        varies="cycles",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=cycle_steps,
        # Powers of 1 - 1/n fall below one rounding after n ln(2^53) turns.
        claim=lambda n: n,
        constant=12.0 * 53 * math.log(2),
    ),
    Family(
        name="agenda to a tolerance",
        executor="agenda-fifo",
        question="Q8",
        varies="accuracy",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=tolerance_steps,
        claim=lambda digits: digits,
        constant=2.0,
    ),
    Family(
        name="closure",
        executor="kleene-star",
        question="Q7",
        varies="states",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=lambda n: measure(lambda: kleene_star(dense_matrix(n), LOG_PROB)).scalar,
        claim=lambda n: n**3,
        # Two sums of three factors, one sum, and lower-order pivots.
        constant=3.5,
    ),
    Family(
        name="closure gradient",
        executor="kleene-star",
        question="Q1g",
        varies="states",
        part="backward",
        sizes=(4, 8, 16, 32, 64),
        work=star_backward,
        claim=lambda n: n**3,
        constant=20.0,
    ),
    Family(
        name="closure saved tensors",
        executor="kleene-star",
        question="Q1g",
        varies="states",
        part="saved",
        sizes=(4, 8, 16, 32, 64),
        work=lambda n: (
            measure(
                lambda: kleene_star(dense_matrix(n, requires_grad=True), LOG_PROB)
            ).saved
        ),
        claim=lambda n: n**3,
        constant=8.0,
    ),
    Family(
        name="closure memory",
        executor="kleene-star",
        question="Q7",
        varies="states",
        part="memory",
        sizes=(8, 16, 32, 64, 128),
        work=star_memory,
        # Without a gradient a closure keeps a few matrices of eight bytes.
        claim=lambda n: 8 * n**2,
        constant=12.0,
    ),
    Family(
        name="best-path closure",
        executor="kleene-star",
        question="Q7",
        varies="states",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=best_path_closure,
        # Elimination makes, per state, n² products, n² selections between
        # two scores and n products of the loop's closure.
        claim=lambda n: n**3,
        constant=4.0,
    ),
    Family(
        name="closure near one",
        executor="kleene-star",
        question="Q8",
        varies="conditioning",
        part="forward",
        sizes=(2, 16, 128, 1024, 8192),
        work=star_near_one,
        claim=lambda turns: 1.0,
        constant=3.5 * 4**3,
    ),
    Family(
        name="matrix product",
        executor="einsum-wiring",
        question="Q6",
        varies="order",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=matrix_product,
        claim=lambda n: n**3,
        constant=8.0,
    ),
    Family(
        name="triangle with a hub",
        executor="einsum-wiring",
        question="Q6",
        varies="skew",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=skewed_triangle,
        claim=lambda n: n**3,
        constant=10.0,
    ),
    Family(
        name="wiring construction",
        executor="einsum-wiring",
        question="Q24",
        varies="inputs",
        part="construction",
        sizes=(2, 4, 8, 16, 32),
        work=contraction_construction,
        claim=lambda n: n,
        constant=2.0,
    ),
    Family(
        name="chart parser over length",
        executor="chart-parser",
        question="Q3",
        varies="length",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda n: parser_work(3, 1, n),
        claim=lambda n: 3 * n**3,
        constant=20.0,
    ),
    Family(
        name="chart parser over rules",
        executor="chart-parser",
        question="Q3",
        varies="grammar",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=lambda c: parser_work(c, 1, 4),
        claim=lambda c: c,
        constant=64.0,
    ),
    Family(
        name="chart parser over batch",
        executor="chart-parser",
        question="Q3",
        varies="batch",
        part="forward",
        sizes=(1, 2, 4, 8, 16),
        work=lambda b: parser_work(3, b, 4),
        claim=lambda b: b,
        constant=200.0,
    ),
    Family(
        name="chart parser after an update",
        executor="chart-parser",
        question="Q24",
        varies="length",
        part="refit",
        sizes=(2, 4, 8, 16, 32),
        work=parser_refit,
        claim=lambda n: 3 * n**3,
        constant=20.0,
    ),
    Family(
        name="chart parser gradient",
        executor="chart-parser",
        question="Q1g",
        varies="length",
        part="backward",
        sizes=(2, 4, 8, 16, 32),
        work=parser_backward,
        claim=lambda n: 3 * n**3,
        constant=60.0,
    ),
    Family(
        name="inside algorithm over length",
        executor="inside",
        question="Q3",
        varies="length",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda n: inside_work(2, n),
        claim=lambda n: 8 * n**3,
        constant=6.0,
    ),
    Family(
        name="inside algorithm over categories",
        executor="inside",
        question="Q3",
        varies="grammar",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda c: inside_work(c, 4),
        claim=lambda c: c**3,
        constant=48.0,
    ),
    Family(
        name="grouped mixture over rows",
        executor="marginalize-grouped",
        question="Q6",
        varies="rows",
        part="forward",
        sizes=(8, 16, 32, 64, 128),
        work=lambda n: grouped_work(n, 3),
        claim=lambda n: 3 * n,
        constant=4.0,
    ),
    Family(
        name="grouped mixture over classes",
        executor="marginalize-grouped",
        question="Q6",
        varies="classes",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda k: grouped_work(16, k),
        claim=lambda k: 16 * k,
        constant=4.0,
    ),
    Family(
        name="chain over length",
        executor="marginalize-chain",
        question="Q10",
        varies="length",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda n: chain_work(2, n),
        claim=lambda n: n,
        constant=400.0,
    ),
    Family(
        name="chain over states",
        executor="marginalize-chain",
        question="Q10",
        varies="width",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=lambda k: chain_work(k, 2),
        claim=lambda k: k**2,
        constant=200.0,
    ),
    Family(
        name="fold",
        executor="encoder",
        question="Q24",
        varies="nodes",
        part="forward",
        sizes=(2, 4, 8, 16, 32),
        work=fold_work,
        claim=lambda n: n,
        constant=4.0,
    ),
)
"""The families."""

REPEATED: tuple[Family, ...] = (
    Family(
        name="repeated runs of one rule set",
        executor="agenda-fifo",
        question="Q24",
        varies="rules",
        part="validation",
        sizes=(4, 8, 16, 32, 64),
        work=records_per_run,
        claim=lambda rules: 2.0,
        constant=1.0,
    ),
    Family(
        name="repeated calls of one potential",
        executor="potential",
        question="R3",
        varies="calls",
        part="validation",
        sizes=(2, 4, 8, 16, 32),
        work=potential_calls,
        claim=lambda calls: calls,
        constant=40.0,
    ),
)
"""Families of repeated calls whose records must not grow with the graph."""
