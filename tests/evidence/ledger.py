"""Known wrong results, each stated exactly.

An entry names facts that a run of the library shows today and what each
fact has to be. The facts are short strings, so every entry is an exact
statement. An entry holds in both directions: the observed facts must be
the recorded ones, and every recorded fact must differ from the required
one. A change of a defect, and its repair, both fail the entry, and the
change that repairs a defect deletes its entry and moves the cells of the
grid that name it.

The required facts are computed by the oracles and are compared with
their record as well, so an entry cannot state a wrong requirement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import cast

import didactic.api as dx
import torch
from quivers._values import OPAQUE_BY_IDENTITY, HeldValue, ValueModel
from quivers.core.algebras import (
    GODEL,
    LogProbAlgebra,
    ProductFuzzyAlgebra,
)
from quivers.core.morphisms import Morphism, observed
from quivers.core.objects import FinSet, ProductSet
from quivers.core.wiring import EinsumWiring
from quivers.dsl import loads
from quivers.dsl.ast_nodes import LetExprCall, LetExprVar, SchemaDecl, TypeName
from quivers.dsl.compiler.programs import LetValue, _ProgramsMixin
from quivers.effects.checked_program import CheckedProgram
from quivers.enriched.kan_extensions import Projection, left_kan
from quivers.inference import MCMC, AutoLaplaceApproximation, MetropolisKernel
from quivers.inference import RandomWalkProposal
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import (
    deduction_item,
    foreign_handler_configuration,
    run_deduction,
    run_program,
)
from quivers.monadic.instances import Identity
from quivers.stochastic._kleene import kleene_star
from quivers.stochastic._rule_system import RuleSystem
from quivers.stochastic.agenda import (
    AgendaResult,
    ChartView,
    DeductionSystem,
    FIFOAgenda,
    HashChart,
    InferenceRule,
    Item,
    LIFOAgenda,
    knuth_agenda,
    make_wildcard,
    match,
    run_agenda,
)
from quivers.stochastic.deduction import sample_corpus
from quivers.stochastic.effect_lifts import class_directed_lifts
from quivers.stochastic.parsers import ChartParser
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)
from quivers.structural.decoder import Decoder
from quivers.structural.signature import Signature
from tests.evidence.checks.consumers import GUIDES, PROBLEMS
from tests.evidence.checks.deduction import weighted
from tests.evidence.checks.machine import module_of
from tests.evidence.checks.span import (
    LEXICAL,
    chart_parser,
    double_precision,
    inside_algorithm,
)
from tests.evidence.checks.terms import asymmetric, encoder, node, leaf, parameters
from tests.evidence.compare import Mismatch
from tests.evidence.counters import CallCounter, TensorCounter
from tests.evidence.oracles import closed_forms, derivations, exact, gradients
from tests.evidence.oracles.derivations import Variable
from tests.evidence.problems import logarithm
from tests.evidence.sources import LEXICAL as LEXICAL_SOURCE
from tests.evidence.sources import deduction, program

type Facts = dict[str, str]
"""Named facts."""


class Entry(ValueModel):
    """One known defect.

    Parameters
    ----------
    name : str
        The entry's name, which cells of the grid refer to.
    where : str
        The code that is wrong.
    defect : str
        What is wrong.
    wrong : dict of str to str
        The facts shown today.
    right : dict of str to str
        The required facts, under the same names.
    observe : Callable[[], Facts]
        Reads the facts from the library.
    require : Callable[[], Facts]
        Computes the required facts.
    """

    name: str
    where: str
    defect: str
    wrong: dict[str, str]
    right: dict[str, str]
    observe: Callable[[], Facts] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    require: Callable[[], Facts] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def verify(entry: Entry) -> None:
    """Require an entry to hold in both directions.

    Parameters
    ----------
    entry : Entry
        The entry.


    Raises
    ------
    Mismatch
        If the recorded facts are not the observed ones, if a recorded
        requirement is not the computed one, or if a fact is recorded as
        wrong and is right.
    """
    if set(entry.wrong) != set(entry.right):
        raise Mismatch("support", math.inf, f"{entry.name}: facts do not pair up")
    for key, value in entry.wrong.items():
        if value == entry.right[key]:
            raise Mismatch("support", math.inf, f"{entry.name}: {key!r} is not wrong")
    required = entry.require()
    if required != entry.right:
        raise Mismatch(
            "value", math.inf, f"{entry.name}: required {required} for {entry.right}"
        )
    observed_facts = entry.observe()
    if observed_facts == entry.right:
        raise Mismatch(
            "value",
            math.inf,
            f"{entry.name}: the defect is repaired; delete the entry and move the "
            "cells that name it to pass",
        )
    if observed_facts != entry.wrong:
        raise Mismatch(
            "value",
            math.inf,
            f"{entry.name}: the defect changed: {observed_facts} for {entry.wrong}",
        )


def number(value: torch.Tensor | float) -> str:
    """A number with six significant digits.

    Parameters
    ----------
    value : torch.Tensor or float
        The number.

    Returns
    -------
    str
        Its text; ``"none"`` is never produced.
    """
    if isinstance(value, torch.Tensor):
        value = float(value.detach())
    return f"{value:.6g}"


def scalar(value: float, *, requires_grad: bool = False) -> torch.Tensor:
    """A float64 scalar.

    Parameters
    ----------
    value : float
        The value.
    requires_grad : bool
        Whether it is differentiated.

    Returns
    -------
    torch.Tensor
        The scalar.
    """
    return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)


def raised(body: Callable[[], torch.Tensor]) -> str:
    """The value of a computation, or the kind of its exception.

    Parameters
    ----------
    body : Callable[[], torch.Tensor]
        The computation.

    Returns
    -------
    str
        The numbers of the result, or the exception's class name.
    """
    try:
        value = body()
    except Exception as failure:
        return type(failure).__name__
    return " ".join(number(v) for v in value.reshape(-1).tolist()) or "empty"


def rule(
    name: str, premises: tuple[Item, ...], conclusion: Item, weight: float | None = None
) -> InferenceRule:
    """A rule of the agenda engine.

    Parameters
    ----------
    name : str
        The name.
    premises : tuple of Item
        The premises.
    conclusion : Item
        The conclusion.
    weight : float or None
        The rule's weight in the semiring's representation.

    Returns
    -------
    InferenceRule
        The rule.
    """
    return InferenceRule(
        name=name,
        premises=premises,
        conclusion=conclusion,
        weight_fn=None if weight is None else weighted(scalar(weight)),
    )


ENTRIES: dict[str, Entry] = {}
"""The entries by name."""


def record(
    name: str,
    where: str,
    defect: str,
    wrong: Mapping[str, str],
    right: Mapping[str, str],
    observe: Callable[[], Facts],
    require: Callable[[], Facts] | None = None,
) -> None:
    """Add an entry.

    Parameters
    ----------
    name : str
        The entry's name.
    where : str
        The code that is wrong.
    defect : str
        What is wrong.
    wrong : Mapping of str to str
        The facts shown today.
    right : Mapping of str to str
        The required facts.
    observe : Callable[[], Facts]
        Reads the facts.
    require : Callable[[], Facts] or None
        Computes the required facts; ``None`` when a requirement is a
        property of the interface and not a number.
    """
    fixed = dict(right)
    ENTRIES[name] = Entry(
        name=name,
        where=where,
        defect=defect,
        wrong=dict(wrong),
        right=fixed,
        observe=observe,
        require=require or (lambda: dict(fixed)),
    )


# -- the agenda engine ---------------------------------------------------


def _zero_update() -> Facts:
    """The derivative of ``3 (1 + 2 t)`` at ``t = 0`` on the agenda.

    Returns
    -------
    Facts
        The derivative, or ``"none"`` without a gradient path.
    """
    t = scalar(0.0, requires_grad=True)
    result = run_agenda(
        [(("X",), scalar(1.0)), (("A",), t)],
        [rule("add", (("A",),), ("X",), 2.0), rule("scale", (("X",),), ("Y",), 3.0)],
        semiring=COUNTING,
    )
    value = result.chart.get(("Y",))
    assert value is not None
    if not value.requires_grad:
        return {"derivative at zero": "none"}
    (found,) = torch.autograd.grad(value, t)
    return {"derivative at zero": number(found)}


def _zero_update_required() -> Facts:
    """The derivative from the trees of the same system.

    Returns
    -------
    Facts
        The derivative, which is constant in ``t``.
    """
    trees = derivations.derivations(
        [("x", ("X",)), ("t", ("A",))],
        [
            ("add", (("A",),), ("X",), "two"),
            ("scale", (("X",),), ("Y",), "three"),
        ],
        ("Y",),
    )
    fixed = {"two": Fraction(2), "three": Fraction(3), "x": Fraction(1)}
    at = [
        derivations.total(trees, {**fixed, "t": t}, Fraction(0), Fraction(1))
        for t in (Fraction(0), Fraction(1))
    ]
    assert float(at[1] - at[0]) == gradients.BOUNDARY_CASES[1].gradient[0]
    return {"derivative at zero": number(float(at[1] - at[0]))}


record(
    "zero-update-stops",
    "stochastic/agenda.py: HashChart.insert_or_aggregate, run_agenda",
    "an update that leaves a number unchanged stops propagating, so a "
    "parameter whose current weight is zero loses its gradient",
    {"derivative at zero": "none"},
    {"derivative at zero": "6"},
    _zero_update,
    _zero_update_required,
)


#: The turns of the cycles whose sums stop on a residual.
RESIDUAL_TURNS = ("9/10", "99/100", "999/1000")


def _residual() -> Facts:
    """The log weights of cycles that stop on a residual.

    Returns
    -------
    Facts
        The log weight for each turn.
    """
    facts = {}
    for turn in RESIDUAL_TURNS:
        result = run_agenda(
            [(("A",), scalar(0.0))],
            [rule("turn", (("A",),), ("A",), math.log(float(Fraction(turn))))],
            semiring=LOG_PROB,
            chart=HashChart(tolerance=1e-3),
        )
        value = result.chart.get(("A",))
        assert value is not None
        facts[f"agenda at {turn}"] = number(value)
    cycle = module_of("cycle", SEMIRING="LogProb", TOLERANCE=", tolerance=0.01")
    found = run_deduction(
        cycle,
        "Cycle",
        axioms=[deduction_item(cycle, "Cycle", "A")],
        axiom_weights=[0.0],
        parameters={"Cycle.rule.loop:": math.log(0.5)},
        fuel=300_000,
    ).weight
    assert isinstance(found, float)
    facts["machine chart at 1/2"] = number(found)
    chained = module_of("chained", SEMIRING="LogProb", TOLERANCE=", tolerance=0.1")
    found = run_deduction(
        chained,
        "M",
        tokens=["a"],
        parameters={"M.rule.there:": math.log(0.5), "M.rule.back:": 0.0},
        fuel=300_000,
    ).weight
    assert isinstance(found, float)
    facts["machine search at 1/2"] = number(found)
    return facts


def _residual_required() -> Facts:
    """The log weights of the same cycles in closed form.

    Returns
    -------
    Facts
        The log weight for each turn.
    """
    facts = {
        f"agenda at {turn}": number(logarithm(closed_forms.geometric(Fraction(turn))))
        for turn in RESIDUAL_TURNS
    }
    half = number(logarithm(closed_forms.geometric(Fraction(1, 2))))
    return {**facts, "machine chart at 1/2": half, "machine search at 1/2": half}


record(
    "residual-is-the-value",
    "stochastic/agenda.py: tolerance; dsl/deduction_elaboration.py: convergence; "
    "dsl/_deduction_chart.py: _Chart.equal",
    "a run that stops because two successive numbers are close returns the last "
    "number as the weight",
    {
        "agenda at 9/10": "2.29382",
        "agenda at 99/100": "4.51127",
        "agenda at 999/1000": "6.21581",
        "machine chart at 1/2": "0.685304",
        "machine search at 1/2": "0.628609",
    },
    {
        "agenda at 9/10": "2.30259",
        "agenda at 99/100": "4.60517",
        "agenda at 999/1000": "6.90776",
        "machine chart at 1/2": "0.693147",
        "machine search at 1/2": "0.693147",
    },
    _residual,
    _residual_required,
)


def _rounded_cycle(weights: tuple[float, ...]) -> torch.Tensor:
    """A max-plus matrix with one cycle.

    Parameters
    ----------
    weights : tuple of float
        The weights along the cycle.

    Returns
    -------
    torch.Tensor
        The matrix.
    """
    size = len(weights)
    matrix = torch.full((size, size), -math.inf, dtype=torch.float64)
    for state, weight in enumerate(weights):
        matrix[state, (state + 1) % size] = weight
    return matrix


#: A cycle whose weights, as exact numbers of float64, sum to zero, while
#: their sum from left to right is the spacing of float64 near two fifths.
ROUNDED = (0.2, 0.4, -0.2, -0.4)

record(
    "rounded-cycle-diverges",
    "stochastic/_kleene.py: _max_plus_star",
    "a cycle of exact weight zero is reported as divergent because the sum of "
    "its weights in one order rounds up",
    {"closure of a zero cycle": "DivergentClosureError"},
    {"closure of a zero cycle": "0"},
    lambda: {
        "closure of a zero cycle": raised(
            lambda: kleene_star(_rounded_cycle(ROUNDED), VITERBI)[0, 0]
        )
    },
    lambda: {
        "closure of a zero cycle": number(
            max(0.0, float(sum((Fraction(w) for w in ROUNDED), Fraction(0))))
        )
    },
)

record(
    "none-is-unbound",
    "stochastic/agenda.py: match",
    "a variable bound to None counts as unbound, so two premises with different "
    "values match",
    {"p(x, x) matches p(None, 3)": "True"},
    {"p(x, x) matches p(None, 3)": "False"},
    lambda: {
        "p(x, x) matches p(None, 3)": str(
            match(("p", make_wildcard("x"), make_wildcard("x")), ("p", None, 3))
            is not None
        )
    },
    lambda: {
        "p(x, x) matches p(None, 3)": str(
            derivations.bind(("p", Variable("x"), Variable("x")), ("p", None, 3), {})
            is not None
        )
    },
)


def reach(length: int) -> tuple[AgendaResult, int]:
    """Run reachability along a line and count pattern matches.

    Parameters
    ----------
    length : int
        The number of edges.

    Returns
    -------
    tuple of (AgendaResult, int)
        The run and the number of matches tried.
    """
    i, j = make_wildcard("i"), make_wildcard("j")
    axioms = [(("edge", k, k + 1), scalar(0.0)) for k in range(length)]
    with CallCounter({"match": match}) as calls:
        result = run_agenda(
            [*axioms, (("reach", 0), scalar(0.0))],
            [
                InferenceRule(
                    name="step",
                    premises=(("reach", i), ("edge", i, j)),
                    conclusion=("reach", j),
                )
            ],
            semiring=LOG_PROB,
        )
    return result, calls.counts["match"]


def hub(size: int) -> tuple[int, int]:
    """Run a triangle rule on three relations with one hub value.

    Parameters
    ----------
    size : int
        The number of neighbours of the hub.

    Returns
    -------
    tuple of (int, int)
        The number of rule applications and of matches tried.
    """
    x, y, z = (make_wildcard(name) for name in "xyz")
    axioms: list[tuple[Item, torch.Tensor]] = []
    for relation in "RST":
        axioms += [((relation, 0, k), scalar(1.0)) for k in range(size)]
        axioms += [((relation, k, 0), scalar(1.0)) for k in range(1, size)]
    with CallCounter({"match": match}) as calls:
        result = run_agenda(
            axioms,
            [
                InferenceRule(
                    name="triangle",
                    premises=(("R", x, y), ("S", y, z), ("T", z, x)),
                    conclusion=("triangle",),
                )
            ],
            semiring=COUNTING,
        )
    return len(result.applications), calls.counts["match"]


record(
    "rule-matching-scans",
    "stochastic/agenda.py: HashChart.lookup, _fire_remaining_premises",
    "every premise scans the whole chart and bindings are not pushed into the "
    "scan, so matching along a line is quadratic, and a triangle over a hub "
    "costs the product of the relations",
    {
        "matches at 20": "1363",
        "matches at 40": "5123",
        "matches at 80": "19843",
        "hub matches at 2": "172",
        "hub matches at 4": "1070",
        "hub matches at 8": "6970",
    },
    {
        "matches at 20": "20",
        "matches at 40": "40",
        "matches at 80": "80",
        "hub matches at 2": "4",
        "hub matches at 4": "10",
        "hub matches at 8": "22",
    },
    lambda: {
        **{f"matches at {n}": str(reach(n)[1]) for n in (20, 40, 80)},
        **{f"hub matches at {n}": str(hub(n)[1]) for n in (2, 4, 8)},
    },
    lambda: {
        **{f"matches at {n}": str(len(_reach_instances(n))) for n in (20, 40, 80)},
        **{f"hub matches at {n}": str(_hub_instances(n)) for n in (2, 4, 8)},
    },
)


def _reach_instances(length: int) -> list[derivations.Instance]:
    """The rule applications of reachability along a line.

    Parameters
    ----------
    length : int
        The number of edges.

    Returns
    -------
    list of Instance
        One per edge.
    """
    i, j = Variable("i"), Variable("j")
    return derivations.ground(
        [
            ("start", ("reach", 0)),
            *((f"e{k}", ("edge", k, k + 1)) for k in range(length)),
        ],
        [("step", (("reach", i), ("edge", i, j)), ("reach", j), None)],
    )


def _hub_instances(size: int) -> int:
    """The number of triangles over relations with one hub value.

    Parameters
    ----------
    size : int
        The number of neighbours of the hub.

    Returns
    -------
    int
        The number of triples whose three pairs are in the relation.
    """
    pairs = {(0, k) for k in range(size)} | {(k, 0) for k in range(size)}
    return sum(
        (a, b) in pairs and (b, c) in pairs and (c, a) in pairs
        for a in range(size)
        for b in range(size)
        for c in range(size)
    )


record(
    "provenance-always-kept",
    "stochastic/agenda.py: AgendaResult.applications",
    "a run that is asked for a weight only keeps every rule application",
    {"applications kept at 20": "20", "applications kept at 40": "40"},
    {"applications kept at 20": "0", "applications kept at 40": "0"},
    lambda: {
        f"applications kept at {n}": str(len(reach(n)[0].applications))
        for n in (20, 40)
    },
)


def _merged_occurrences() -> Facts:
    """The provenance of two equal axioms and of two rules with one name.

    Returns
    -------
    Facts
        The numbers of proofs that can be told apart.
    """
    twice = run_agenda(
        [(("A",), scalar(1.0)), (("A",), scalar(1.0))],
        [rule("r", (("A",),), ("X",))],
        semiring=COUNTING,
    )
    named = run_agenda(
        [(("A",), scalar(1.0))],
        [rule("r", (("A",),), ("X",)), rule("r", (("A",),), ("X",))],
        semiring=COUNTING,
    )
    return {
        "proofs from two equal axioms": str(len(twice.applications)),
        "proofs by two rules of one name": str(
            len(set(ChartView(named).derivations(("X",))))
        ),
    }


def _merged_occurrences_required() -> Facts:
    """The same numbers from enumerated derivations.

    Returns
    -------
    Facts
        The numbers of trees.
    """
    twice = derivations.derivations(
        [("one", ("A",)), ("two", ("A",))], [("r", (("A",),), ("X",), None)], ("X",)
    )
    named = derivations.derivations(
        [("one", ("A",))],
        [("r", (("A",),), ("X",), "first"), ("r", (("A",),), ("X",), "second")],
        ("X",),
    )
    return {
        "proofs from two equal axioms": str(len(set(twice))),
        "proofs by two rules of one name": str(len(set(named))),
    }


record(
    "occurrences-merge",
    "stochastic/agenda.py: run_agenda, ChartView.derivations",
    "equal axioms are one antecedent of one application, and applications of "
    "two rules with one name are equal",
    {"proofs from two equal axioms": "1", "proofs by two rules of one name": "1"},
    {"proofs from two equal axioms": "2", "proofs by two rules of one name": "2"},
    _merged_occurrences,
    _merged_occurrences_required,
)


def _steps(shape: str, size: int, discipline: str) -> int:
    """The number of items an agenda pops on an acyclic graph.

    Parameters
    ----------
    shape : str
        ``"skip"`` for steps of one and two, ``"diamond"`` for two routes
        per layer.
    size : int
        The number of layers.
    discipline : str
        The agenda.

    Returns
    -------
    int
        The number of pops.
    """
    if shape == "skip":
        rules = [
            rule(f"{i}+{d}", (("n", i),), ("n", i + d))
            for i in range(size)
            for d in (1, 2)
            if i + d <= size
        ]
    else:
        rules = [
            part
            for i in range(size)
            for part in (
                rule(f"straight{i}", (("n", i),), ("n", i + 1)),
                rule(f"out{i}", (("n", i),), ("m", i)),
                rule(f"back{i}", (("m", i),), ("n", i + 1)),
            )
        ]
    agenda = {"fifo": FIFOAgenda, "lifo": LIFOAgenda, "priority": knuth_agenda}
    return run_agenda(
        [(("n", 0), scalar(1.0))], rules, semiring=COUNTING, agenda=agenda[discipline]()
    ).iterations


def _items(shape: str, size: int) -> int:
    """The number of items of the same graph.

    Parameters
    ----------
    shape : str
        The graph.
    size : int
        The number of layers.

    Returns
    -------
    int
        The number of items, each of which has to be popped once.
    """
    return size + 1 if shape == "skip" else 2 * size + 1


#: Graphs, sizes, and agendas that pop an item more than once.
REPEATED = (
    *(("skip", n, d) for d in ("lifo", "priority") for n in (6, 8, 10, 12)),
    *(("diamond", n, d) for d in ("lifo", "priority") for n in (4, 6, 8)),
    *(("diamond", n, "fifo") for n in (4, 6, 8)),
)

record(
    "repeated-propagation",
    "stochastic/agenda.py: LIFOAgenda, PriorityQueueAgenda, FIFOAgenda",
    "an item is propagated once per contribution: the last-in and the priority "
    "agenda pop exponentially often on an acyclic graph, the first-in agenda "
    "quadratically often on layered routes",
    {
        "skip 6 lifo": "33",
        "skip 8 lifo": "88",
        "skip 10 lifo": "232",
        "skip 12 lifo": "609",
        "skip 6 priority": "33",
        "skip 8 priority": "88",
        "skip 10 priority": "232",
        "skip 12 priority": "609",
        "diamond 4 lifo": "46",
        "diamond 6 lifo": "190",
        "diamond 8 lifo": "766",
        "diamond 4 priority": "46",
        "diamond 6 priority": "190",
        "diamond 8 priority": "766",
        "diamond 4 fifo": "25",
        "diamond 6 fifo": "49",
        "diamond 8 fifo": "81",
    },
    {f"{shape} {n} {d}": str(_items(shape, n)) for shape, n, d in REPEATED},
    lambda: {f"{s} {n} {d}": str(_steps(s, n, d)) for s, n, d in REPEATED},
)


def _default_precision() -> Facts:
    """The precisions of results of float64 computations.

    Returns
    -------
    Facts
        The type of each result.
    """
    empty = ChartView(
        run_agenda(
            [(("A",), scalar(1.0))],
            [],
            semiring=LOG_PROB,
            goal=lambda item: item == ("Z",),
        )
    )
    parser = chart_parser(LOG_PROB, {})
    algorithm, _ = inside_algorithm({})
    problem = PROBLEMS["beta"]
    with double_precision():
        x, data = problem.data()
        result = MCMC(
            MetropolisKernel(RandomWalkProposal(0.5)),
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            seed=0,
        ).run(problem.model(), x, data)
        assert result.samples["theta"].dtype == torch.float64
    return {
        "goal weight of no goal": str(empty.goal_weight().dtype),
        "weight of an absent item": str(empty.try_weight(("Z",)).dtype),
        "chart parser score": str(parser(torch.tensor((0, 1))).dtype),
        "inside score": str(algorithm(torch.tensor((0, 1))).dtype),
        "recorded log density of a chain": str(result.log_densities.dtype),
    }


record(
    "default-precision",
    "stochastic/agenda.py: goal_weight, try_weight; stochastic/span.py: "
    "SpanChart.to_tensor; stochastic/inside.py: _fill_chart; "
    "inference/mcmc/driver.py: log_densities",
    "zeros and charts take the default type, and recorded log densities are "
    "single precision, so a double-precision run returns single precision",
    dict.fromkeys(
        (
            "goal weight of no goal",
            "weight of an absent item",
            "chart parser score",
            "inside score",
            "recorded log density of a chain",
        ),
        "torch.float32",
    ),
    dict.fromkeys(
        (
            "goal weight of no goal",
            "weight of an absent item",
            "chart parser score",
            "inside score",
            "recorded log density of a chain",
        ),
        "torch.float64",
    ),
    _default_precision,
)


def _nullary() -> Facts:
    """The weight of an item derived by a rule without premises.

    Returns
    -------
    Facts
        The weight, or ``"absent"``.
    """
    result = run_agenda(
        [(("A",), scalar(1.0))],
        [rule("unit", (), ("B",)), rule("use", (("A",), ("B",)), ("C",))],
        semiring=COUNTING,
    )
    value = result.chart.get(("C",))
    return {
        "weight over a rule without premises": "absent"
        if value is None
        else number(value)
    }


record(
    "nullary-rule-never-fires",
    "stochastic/agenda.py: run_agenda",
    "a rule without premises never fires, though its weight function returns the unit",
    {"weight over a rule without premises": "absent"},
    {"weight over a rule without premises": "1"},
    _nullary,
    lambda: {
        "weight over a rule without premises": number(
            len(
                derivations.derivations(
                    [("a", ("A",))],
                    [
                        ("unit", (), ("B",), None),
                        ("use", (("A",), ("B",)), ("C",), None),
                    ],
                    ("C",),
                )
            )
        )
    },
)


class _SizedChart(HashChart):
    """A chart with a length."""

    def __len__(self) -> int:
        """The number of items.

        Returns
        -------
        int
            The number.
        """
        return len(self._store)


def _chart_choice() -> Facts:
    """Which chart and which tolerance a run uses.

    Returns
    -------
    Facts
        Whether the given chart is used, and the tolerance that reaches it.
    """
    given = _SizedChart()
    result = run_agenda([(("A",), scalar(1.0))], [], semiring=COUNTING, chart=given)
    built: list[HashChart] = []

    def factory(tolerance: float = 0.0) -> HashChart:
        """Build a chart, failing on its first call.

        Parameters
        ----------
        tolerance : float
            The tolerance.

        Returns
        -------
        HashChart
            The chart.


        Raises
        ------
        TypeError
            On the first call, for a reason of its own.
        """
        if not built:
            built.append(HashChart())
            raise TypeError("an error inside the factory")
        return HashChart(tolerance)

    run = DeductionSystem(
        rules=(),
        semiring=COUNTING,
        axiom_injector=lambda value: [(("A",), scalar(1.0))],
        goal=lambda item: True,
        chart_factory=factory,
        tolerance=1e-3,
    ).run((1,))
    assert isinstance(run.chart, HashChart)
    return {
        "the given empty chart is used": str(result.chart is given),
        "tolerance after an error in the factory": number(run.chart._tolerance),
    }


record(
    "chart-argument-dropped",
    "stochastic/agenda.py: run_agenda, DeductionSystem._run",
    "an empty chart with a length is replaced, and a type error inside a chart "
    "factory drops the tolerance and is lost",
    {
        "the given empty chart is used": "False",
        "tolerance after an error in the factory": "0",
    },
    {
        "the given empty chart is used": "True",
        "tolerance after an error in the factory": "TypeError",
    },
    _chart_choice,
)

#: A rule whose weight is keyed by the categories it binds.
BINDING_WEIGHTS = """
object Term : FinSet 8
deduction G : Term -> Term [semiring=LogProb, start=S, depth=2]
    atoms S, N, Fwd, span, f, g, a
    rule apply : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    lexicon
        "f" : Fwd(S, N) = f
        "g" : Fwd(S, S) = g
        "a" : N = a
"""


def _parameters_while_firing() -> Facts:
    """The number of parameters before and after runs.

    Returns
    -------
    Facts
        The numbers.
    """
    system = deduction(loads(BINDING_WEIGHTS), "G")
    counts = [len(list(system.named_parameters()))]
    for sentence in (["f", "a"], ["g", "f", "a"]):
        system(sentence)
        counts.append(len(list(system.named_parameters())))
    return {
        "parameters before a run": str(counts[0]),
        "parameters after a short sentence": str(counts[1]),
    }


record(
    "parameters-created-by-runs",
    "dsl/compiler/deductions.py: _make_rule_weight_fn",
    "rule weights are created when a rule fires, so the set of parameters "
    "depends on the inputs seen",
    {"parameters before a run": "0", "parameters after a short sentence": "1"},
    {"parameters before a run": "2", "parameters after a short sentence": "2"},
    _parameters_while_firing,
    lambda: dict.fromkeys(
        ("parameters before a run", "parameters after a short sentence"),
        # The lexicon has the two functor categories Fwd(S, N) and Fwd(S, S).
        str(BINDING_WEIGHTS.count("Fwd(S,")),
    ),
)

record(
    "chart-view-promises",
    "stochastic/agenda.py: ChartView",
    "every result offers every observation, and one that the run cannot give "
    "fails when it is asked for",
    {"embedding on a run without an encoder": "RuntimeError", "offered": "True"},
    {"embedding on a run without an encoder": "AttributeError", "offered": "False"},
    lambda: {
        "embedding on a run without an encoder": raised(
            lambda: ChartView(
                run_agenda([(("A",), scalar(1.0))], [], semiring=COUNTING)
            ).embedding(("A",))
        ),
        "offered": str(hasattr(ChartView, "embedding")),
    },
)


def _named_goal(head: str) -> str:
    """The weight of a goal read off a chart whose items have a given head.

    Parameters
    ----------
    head : str
        The items' first slot.

    Returns
    -------
    str
        The weight.
    """
    system = DeductionSystem(
        rules=(),
        semiring=COUNTING,
        axiom_injector=lambda value: [
            ((head, 0, 1, "S"), scalar(1.0)),
            ((head, 0, 2, "S"), scalar(3.0)),
        ],
        goal=lambda item: item[-1] == "S",
    )
    return number(ChartView(system.run(("a", "b"))).goal_weight())


class _Forwarded(ChartSemiring):
    """A semiring that forwards every operation to the log semiring."""

    @property
    def name(self) -> str:
        """A name of its own.

        Returns
        -------
        str
            The name.
        """
        return "ForwardedLogProb"

    def tensor_op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward a product.

        Parameters
        ----------
        a : torch.Tensor
            The first factor.
        b : torch.Tensor
            The second factor.

        Returns
        -------
        torch.Tensor
            The product.
        """
        return LOG_PROB.tensor_op(a, b)

    def join(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Forward a reduction.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        return LOG_PROB.join(t, dim)

    def join_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward a binary sum.

        Parameters
        ----------
        a : torch.Tensor
            The first term.
        b : torch.Tensor
            The second term.

        Returns
        -------
        torch.Tensor
            The sum.
        """
        return LOG_PROB.join_pair(a, b)

    def meet(self, t: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
        """Forward a meet.

        Parameters
        ----------
        t : torch.Tensor
            The terms.
        dim : int or tuple of int
            The axes.

        Returns
        -------
        torch.Tensor
            The meet.
        """
        return LOG_PROB.meet(t, dim)

    def negate(self, t: torch.Tensor) -> torch.Tensor:
        """Forward a negation.

        Parameters
        ----------
        t : torch.Tensor
            The values.

        Returns
        -------
        torch.Tensor
            The negation.
        """
        return LOG_PROB.negate(t)

    def contains(self, t: torch.Tensor) -> torch.Tensor:
        """Forward membership.

        Parameters
        ----------
        t : torch.Tensor
            The values.

        Returns
        -------
        torch.Tensor
            Membership in the log carrier.
        """
        return LOG_PROB.contains(t)

    @property
    def zero(self) -> float:
        """The forwarded zero.

        Returns
        -------
        float
            The zero.
        """
        return LOG_PROB.zero

    @property
    def unit(self) -> float:
        """The forwarded unit.

        Returns
        -------
        float
            The unit.
        """
        return LOG_PROB.unit


def _renamed_handlers() -> str:
    """How many handlers of a deduction keep their meaning under renaming.

    Returns
    -------
    str
        The number of handlers of the module that are configured after
        the semiring is dropped from their names.
    """
    module = module_of("lexical", SEMIRING="Counting")
    renamed = module.with_(
        handlers=tuple(
            handler.with_(name=handler.name.replace("_counting", "_total"))
            for handler in module.handlers
        )
    )
    return str(len(foreign_handler_configuration(renamed)))


record(
    "names-decide-semantics",
    "stochastic/agenda.py: _full_extent_goal; stochastic/_kleene.py: kleene_star; "
    "qiec/program_runtime.py: _deduction_handler",
    "the first slot of an item restricts its goal, the class of a semiring "
    "decides whether it has a closure, and the name of a handler decides what "
    "it computes",
    {
        "goal weight of items headed span": "3",
        "closure in a forwarding semiring": "TypeError",
        "configured handlers after renaming": "0",
    },
    {
        "goal weight of items headed span": "4",
        "closure in a forwarding semiring": "0.458675",
        "configured handlers after renaming": "2",
    },
    lambda: {
        "goal weight of items headed span": _named_goal("span"),
        "closure in a forwarding semiring": raised(
            lambda: kleene_star(
                torch.tensor([[-1.0]], dtype=torch.float64), _Forwarded()
            )
        ),
        "configured handlers after renaming": _renamed_handlers(),
    },
    lambda: {
        "goal weight of items headed span": _named_goal("part"),
        "closure in a forwarding semiring": number(-math.log(1 - math.exp(-1))),
        "configured handlers after renaming": str(
            len(
                foreign_handler_configuration(module_of("lexical", SEMIRING="Counting"))
            )
        ),
    },
)

# -- closures, algebras, and contractions --------------------------------


def _zero_entry() -> Facts:
    """The derivative of a closure at a zero entry.

    Returns
    -------
    Facts
        The derivative of the entry from 0 to 1 in its own weight.
    """
    matrix = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    (found,) = torch.autograd.grad(kleene_star(matrix, COUNTING)[0, 1], matrix)
    return {"derivative at a zero entry": number(found[0, 1])}


def _zero_entry_required() -> Facts:
    """The same derivative from ``A* dA A*`` in exact arithmetic.

    Returns
    -------
    Facts
        The derivative.
    """
    star = exact.linear_star([[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]])
    value = star[0][0] * star[1][1]
    assert float(value) == gradients.BOUNDARY_CASES[0].gradient[1]
    return {"derivative at a zero entry": number(float(value))}


record(
    "zero-entry-gradient",
    "stochastic/_kleene.py: _linear_star",
    "the support is read from the current numbers, so an entry that is zero now "
    "has no gradient",
    {"derivative at a zero entry": "0"},
    {"derivative at a zero entry": "1"},
    _zero_entry,
    _zero_entry_required,
)

record(
    "max-plus-rounds",
    "stochastic/_kleene.py: _max_plus_star",
    "the best-path closure repeats a cubic product once per edge of the longest "
    "best path, so it is quartic",
    {
        "scalar work at 8": "8704",
        "scalar work at 16": "135168",
        "scalar work at 32": "2129920",
    },
    {
        "scalar work at 8": "1024",
        "scalar work at 16": "8192",
        "scalar work at 32": "65536",
    },
    lambda: {f"scalar work at {n}": str(_max_plus_work(n)) for n in (8, 16, 32)},
    # Elimination multiplies and compares once per pair and per state.
    lambda: {f"scalar work at {n}": str(2 * n**3) for n in (8, 16, 32)},
)


def _max_plus_work(size: int) -> int:
    """The scalar work of a best-path closure on a line.

    Parameters
    ----------
    size : int
        The number of states.

    Returns
    -------
    int
        Sums and products.
    """
    matrix = torch.full((size, size), -math.inf, dtype=torch.float64)
    for state in range(size - 1):
        matrix[state, state + 1] = -1.0
    with TensorCounter() as counter:
        kleene_star(matrix, VITERBI)
    return counter.counts["sums"] + counter.counts["products"]


def _empty() -> Facts:
    """The result of a best-path closure over no states.

    Returns
    -------
    Facts
        The result or the exception.
    """
    return {
        "best-path closure": raised(
            lambda: kleene_star(torch.zeros(0, 0, dtype=torch.float64), VITERBI)
        ),
    }


record(
    "empty-reduction-raises",
    "stochastic/_kleene.py: _max_plus_star",
    "a best-path closure over no states raises instead of returning the empty closure",
    {"best-path closure": "IndexError"},
    {"best-path closure": "empty"},
    _empty,
)


def _join_peak(length: int, size: int = 6) -> int:
    """The largest tensor of a contraction along a line of matrices.

    Parameters
    ----------
    length : int
        The number of matrices.
    size : int
        Their order.

    Returns
    -------
    int
        Elements of the largest tensor made.
    """
    letters = "abcdefgh"
    inputs = [letters[k : k + 2] for k in range(length)]
    wiring = EinsumWiring(LogProbAlgebra(), f"{','.join(inputs)}->a{letters[length]}")
    matrices = [torch.zeros(size, size, dtype=torch.float64)] * length
    with TensorCounter() as counter:
        wiring.apply(*matrices)
    return counter.largest


record(
    "dense-join",
    "core/wiring.py: EinsumWiring.apply",
    "a contraction builds the join of all its inputs before it reduces, so a "
    "line of matrices needs the order to the power of its length plus one",
    {f"largest tensor of {k} matrices": str(6 ** (k + 1)) for k in (3, 4, 5)},
    {f"largest tensor of {k} matrices": str(6**3) for k in (3, 4, 5)},
    lambda: {f"largest tensor of {k} matrices": str(_join_peak(k)) for k in (3, 4, 5)},
)

record(
    "certain-entry-clamped",
    "core/algebras.py: ProductFuzzyAlgebra.join",
    "a certain entry is clamped below one, so a disjunction with a true disjunct "
    "is not true",
    {"disjunction with a true disjunct": "0.9999999"},
    {"disjunction with a true disjunct": "1"},
    lambda: {
        "disjunction with a true disjunct": f"{
            float(
                ProductFuzzyAlgebra().join(
                    torch.tensor([1.0, 0.0], dtype=torch.float64), 0
                )
            ):.10g}"
    },
    lambda: {
        "disjunction with a true disjunct": f"{
            float(1 - (1 - Fraction(1)) * (1 - Fraction(0))):.10g}"
    },
)

record(
    "dense-is-the-contract",
    "core/morphisms.py: Morphism",
    "a dense tensor is part of the root interface of a morphism",
    {"the root interface requires a tensor": "True"},
    {"the root interface requires a tensor": "False"},
    lambda: {
        "the root interface requires a tensor": str(
            "tensor" in Morphism.__abstractmethods__
        )
    },
)


def _kan_gradient() -> Facts:
    """Whether a Kan extension of trainable values has a gradient path.

    Returns
    -------
    Facts
        Whether the result requires a gradient.
    """
    left = FinSet(name="A", cardinality=2)
    right = FinSet(name="B", cardinality=2)
    pair = ProductSet(components=(left, right))
    weights = torch.full((2, 2, 2), 0.5, dtype=torch.float64, requires_grad=True)
    morphism = observed(pair, right, weights, algebra=GODEL)
    result = left_kan(morphism, along=Projection(pair, keep_indices=(0,)))
    return {"extension of trainable values": str(result.tensor.requires_grad)}


record(
    "kan-extension-detaches",
    "enriched/kan_extensions.py: left_kan, right_kan",
    "a Kan extension drops the gradient path of the values it extends",
    {"extension of trainable values": "False"},
    {"extension of trainable values": "True"},
    _kan_gradient,
)

# -- span parsers --------------------------------------------------------


def _impossible_sentence() -> Facts:
    """The score of a sentence a grammar does not derive.

    Returns
    -------
    Facts
        The score.
    """
    categories = FinSet(name="N", cardinality=2)
    terminals = FinSet(name="T", cardinality=2)
    binary = torch.zeros(2, 2, 2, dtype=torch.float64)
    binary[0, 1, 1] = 1.0
    lexical = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    algorithm = type(inside_algorithm({})[0])(
        observed(categories, ProductSet(components=(categories, categories)), binary),
        observed(categories, terminals, lexical),
    )
    rules = torch.zeros(2, 2, 2, dtype=torch.float64)
    rules[0, 1, 1] = 1.0
    rules[0, 0, 1] = 0.5
    rules.requires_grad_()
    words = torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    sloped = type(algorithm)(
        observed(categories, ProductSet(components=(categories, categories)), rules),
        observed(categories, terminals, words),
    )
    (found,) = torch.autograd.grad(sloped(torch.tensor([0, 0, 0])), rules)
    return {
        "score of an impossible sentence": number(algorithm(torch.tensor([0, 1]))),
        "derivative in a rule of weight zero": number(found[1, 1, 1]),
    }


def _zero_rule_slope() -> str:
    """The derivative of a log score in a rule of weight zero, exactly.

    The rule derives the second category over two of three words, so the
    score is a polynomial of degree at most two in its weight and three
    values give its linear coefficient.

    Returns
    -------
    str
        The derivative at zero.
    """
    values = [
        exact.inside(
            {
                (0, 1, 1): Fraction(1),
                (0, 0, 1): Fraction(1, 2),
                (1, 1, 1): Fraction(level),
            },
            {(0, 0): Fraction(1, 2), (1, 0): Fraction(1, 2)},
            {},
            (0, 0, 0),
            2,
        )[0]
        for level in (0, 1, 2)
    ]
    slope = 2 * (values[1] - values[0]) - (values[2] - values[0]) / 2
    return number(float(slope / values[0]))


record(
    "zero-probability-clamped",
    "stochastic/inside.py: InsideAlgorithm._fill_chart",
    "zero probabilities are replaced by a small positive number, so a sentence "
    "without a derivation has a finite score and a rule of weight zero has no "
    "gradient",
    {
        "score of an impossible sentence": "-69.0776",
        "derivative in a rule of weight zero": "0",
    },
    {
        "score of an impossible sentence": "-inf",
        "derivative in a rule of weight zero": "3.33333",
    },
    _impossible_sentence,
    lambda: {
        "derivative in a rule of weight zero": _zero_rule_slope(),
        "score of an impossible sentence": number(
            logarithm(
                exact.inside(
                    {(0, 1, 1): Fraction(1)},
                    {(1, 0): Fraction(1)},
                    {},
                    (0, 1),
                    2,
                )[0]
            )
        ),
    },
)


def _linear_scores() -> Facts:
    """The scores of a sentence in the linear semirings.

    Returns
    -------
    Facts
        The score in each.
    """
    facts = {}
    with double_precision():
        for name, semiring in (("counting", COUNTING), ("Boolean", BOOLEAN)):
            parser = chart_parser(semiring, {})
            with torch.no_grad():
                # Every binary rule has the semiring's unit as its weight.
                parser.deductions[0].get_parameter("weights").fill_(1.0)
            facts[f"{name} score"] = number(parser(torch.tensor((0, 1))))
    return facts


def _linear_scores_required() -> Facts:
    """The number of parses and whether one exists.

    Returns
    -------
    Facts
        The count over every category pair a rule combines.
    """
    i, k, j = Variable("i"), Variable("k"), Variable("j")
    axioms = [
        (f"{p}:{a}", ("span", a, p, p + 1))
        for p, token in enumerate((0, 1))
        for a, t in LEXICAL
        if t == token
    ]
    trees = derivations.derivations(
        axioms,
        [
            (
                f"{a}{b}{c}",
                (("span", b, i, k), ("span", c, k, j)),
                ("span", a, i, j),
                None,
            )
            for a, b, c in ((0, 1, 2), (0, 0, 0), (1, 2, 1), (2, 1, 1), (1, 0, 2))
        ],
        ("span", 0, 0, 2),
    )
    return {"counting score": number(len(trees)), "Boolean score": number(bool(trees))}


record(
    "linear-semiring-reads-logarithms",
    "stochastic/span.py: LexicalAxiom",
    "the lexicon gives logarithms in every semiring, so counts and truth values "
    "are sums of products of logarithms",
    {"counting score": "3.08212", "Boolean score": "2.88272"},
    {"counting score": "2", "Boolean score": "1"},
    _linear_scores,
    _linear_scores_required,
)


def _category_cells(categories: int) -> int:
    """Elements allocated by a parser with one rule per category.

    Parameters
    ----------
    categories : int
        The number of categories.

    Returns
    -------
    int
        Elements allocated for a sentence of two words.
    """
    parser = ChartParser(
        RuleSystem(
            binary_rules=tuple((a, a, (a + 1) % categories) for a in range(categories)),
            n_categories=categories,
        ),
        n_terminals=1,
        start_idx=0,
    )
    with TensorCounter() as counter:
        parser(torch.zeros(1, 2, dtype=torch.long))
    return counter.counts["allocated"]


record(
    "row-rebuilt-per-category",
    "stochastic/span.py: _scatter_semiring",
    "a cell copies its whole row once per category, so a grammar with a fixed "
    "number of rules per category costs the square of its categories",
    {"four times the categories cost more than eight times": "True"},
    {"four times the categories cost more than eight times": "False"},
    lambda: {
        "four times the categories cost more than eight times": str(
            _category_cells(64) > 8 * _category_cells(16)
        )
    },
)

# -- the reference machine and the checked program -----------------------


def _models_made(module: QiecModule) -> int:
    """How many records one short run of a deduction validates.

    Parameters
    ----------
    module : QiecModule
        The module.

    Returns
    -------
    int
        The number.
    """
    run_deduction(module, "G", tokens=["b"])
    with CallCounter({"made": dx.Model._validate_instance}) as calls:
        run_deduction(module, "G", tokens=["b"])
    return calls.counts["made"]


def _unrelated(count: int) -> QiecModule:
    """A module with unrelated deductions beside the one that is run.

    Parameters
    ----------
    count : int
        The number of unrelated deductions.

    Returns
    -------
    QiecModule
        The module.
    """
    extra = "".join(
        f"""
deduction X{k} : Term -> Term [semiring=Counting, start=S, depth=1]
    atoms S, span, a
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S)
    lexicon
        "a" : S = a
"""
        for k in range(count)
    )
    module = loads(LEXICAL_SOURCE.replace("SEMIRING", "Counting") + extra).qiec
    assert module is not None
    return module


def _graph_per_call() -> Facts:
    """How the records of one run depend on unrelated declarations.

    Returns
    -------
    Facts
        Whether a run with unrelated deductions makes more records.
    """
    alone, beside = _models_made(_unrelated(0)), _models_made(_unrelated(4))
    return {
        "records of a run grow with unrelated declarations": str(beside > 2 * alone)
    }


record(
    "module-traversed-per-run",
    "qiec/program_runtime.py: run_deduction, run_named",
    "every run rebuilds records of the whole module, so its cost grows with "
    "declarations it does not use",
    {"records of a run grow with unrelated declarations": "True"},
    {"records of a run grow with unrelated declarations": "False"},
    _graph_per_call,
)


def _machine_values() -> Facts:
    """The type of a machine result under a tensor parameter.

    Returns
    -------
    Facts
        The type.
    """
    weight = scalar(-1.0)
    found = run_deduction(
        module_of("lexical", SEMIRING="LogProb"),
        "G",
        tokens=["a", "a"],
        parameters={"G.rule.join:": weight, "G.rule.lift:": 0.0},
    ).weight
    return {"result of a run with a tensor parameter": type(found).__name__}


record(
    "machine-values-are-numbers",
    "qiec/evaluator.py; qiec/builtins.py",
    "the reference machine turns tensors into numbers, so its results carry no "
    "gradient",
    {"result of a run with a tensor parameter": "float"},
    {"result of a run with a tensor parameter": "Tensor"},
    _machine_values,
)


def _checked() -> Facts:
    """Whether a checked program's joint has a gradient, and its runs.

    Returns
    -------
    Facts
        The gradient path and the number of machine runs for three rows.
    """
    model = program("recursive").morphism
    assert isinstance(model, CheckedProgram)
    data = {"t": torch.tensor([0.5], requires_grad=True), "y": torch.tensor([0.5])}
    joint = model.log_joint(torch.zeros(1, 1), data)
    with CallCounter({"runs": run_program}) as calls:
        model.log_joint(torch.zeros(3, 1), {**data, "t": torch.tensor([0.5])})
    return {
        "joint has a gradient": str(joint.requires_grad),
        "machine runs for three rows": str(calls.counts["runs"]),
    }


record(
    "checked-program-detached",
    "effects/checked_program.py: CheckedProgram.log_joint",
    "a checked program's log joint has no gradient and is computed by one "
    "machine run per row",
    {"joint has a gradient": "False", "machine runs for three rows": "3"},
    {"joint has a gradient": "True", "machine runs for three rows": "1"},
    _checked,
)


def _nested_marginalize() -> Facts:
    """The outcome of summing out a draw inside another one.

    Returns
    -------
    Facts
        The exception's class.
    """
    source = """
object K : FinSet 2
object Obs : FinSet 3
object V : Real 1
program nested : Obs -> V
    sample prior <- Dirichlet(concentration=1.0) [over=K]
    marginalize a : K <- Categorical(probs=prior) [reduction=logsumexp]
        marginalize b : K <- Categorical(probs=prior) [reduction=logsumexp]
            let rate = 0.125 + 0.25 * real(a) + 0.5 * real(b)
            observe y : Obs <- Bernoulli(probs=rate)
    return prior
export nested
"""
    try:
        loads(source)
    except Exception as failure:
        return {
            "outcome": type(failure).__name__,
            "has a message": str(bool(str(failure))),
        }
    return {"outcome": "compiled", "has a message": "True"}


record(
    "nested-marginalize-asserts",
    "dsl/program_elaboration.py: _elaborate_marginalize",
    "an ungrouped marginalization inside another one fails an internal "
    "assertion without a message",
    {"outcome": "AssertionError", "has a message": "False"},
    {"outcome": "compiled", "has a message": "True"},
    _nested_marginalize,
)

# -- deductions as programs ----------------------------------------------


def _composed() -> Facts:
    """What the composition of two deductions gives the second one.

    Returns
    -------
    Facts
        The inputs the second injector sees, and whether goal items of the
        first are axioms of the second without a map.
    """
    seen: list[str] = []

    def system(name: str, items: list[tuple[Item, torch.Tensor]]) -> DeductionSystem:
        """A deduction that reports its input.

        Parameters
        ----------
        name : str
            The deduction's name.
        items : list of (Item, torch.Tensor)
            Its axioms.

        Returns
        -------
        DeductionSystem
            The deduction.
        """

        def inject(value: HeldValue) -> list[tuple[Item, torch.Tensor]]:
            """Report the input and return the axioms.

            Parameters
            ----------
            value : HeldValue
                The input.

            Returns
            -------
            list of (Item, torch.Tensor)
                The axioms.
            """
            seen.append(f"{name}:{value}")
            return items

        return DeductionSystem(
            rules=(),
            semiring=COUNTING,
            axiom_injector=inject,
            goal=lambda item: item[0] == "G",
        )

    build = _ProgramsMixin._compile_let_expr(
        LetExprCall(func="compose", args=(LetExprVar(name="f"), LetExprVar(name="g")))
    )
    # The compiler hands deduction systems to `compose` through values its
    # annotation does not list.
    systems = cast(
        dict[str, LetValue],
        {
            "f": system("first", [(("G", "x"), scalar(1.0))]),
            "g": system("second", []),
        },
    )
    composed = build(systems)
    assert isinstance(composed, DeductionSystem)
    result = composed.run("input")
    return {
        "input of the second injector": seen[1].split(":")[1],
        "axioms of the second system": str([item for item, _ in result.goal_items]),
    }


record(
    "composition-has-no-map",
    "dsl/compiler/programs.py: compose",
    "a composition hands the goal items of the first system to the second "
    "unchanged and gives both injectors the same input",
    {
        "input of the second injector": "input",
        "axioms of the second system": "[('G', 'x')]",
    },
    {
        "input of the second injector": "refused without a map",
        "axioms of the second system": "refused without a map",
    },
    _composed,
)

#: A grammar of two words with every bracketing.
TWO_WORDS = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=1]
    atoms S, span, a, b
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S)
    lexicon
        "a" : S = a
        "b" : S = b
"""


def _corpus_runs() -> Facts:
    """The number of charts one draw of a string needs.

    Returns
    -------
    Facts
        The number for each length.
    """
    system = deduction(loads(TWO_WORDS), "G")
    facts = {}
    for length in (2, 3, 4):
        with CallCounter({"runs": DeductionSystem.run}) as calls:
            sample_corpus(system, length=length, n_samples=1, seed=0)
        facts[f"charts for one string of {length}"] = str(calls.counts["runs"])
    return facts


record(
    "corpus-enumerates-strings",
    "stochastic/deduction/sample.py: sample_corpus",
    "drawing one string runs a chart for every string of its length",
    {f"charts for one string of {n}": str(2**n) for n in (2, 3, 4)},
    {f"charts for one string of {n}": "1" for n in (2, 3, 4)},
    _corpus_runs,
)


def _fixed_lifts() -> Facts:
    """Whether the lifts of a rule depend on the rule.

    Returns
    -------
    Facts
        Whether two rules with different types have equal lifts.
    """
    base = SchemaDecl(
        name="r", parameters=(), domain=TypeName(name="A"), codomain=TypeName(name="B")
    )
    other = base.with_(domain=TypeName(name="C"), codomain=TypeName(name="D"))
    return {
        "lifts of two rules are equal": str(
            class_directed_lifts(base, Identity())
            == class_directed_lifts(other, Identity())
        )
    }


record(
    "lifts-ignore-the-rule",
    "stochastic/effect_lifts.py: class_directed_lifts",
    "a lift is a fixed shape that does not read the premises of the rule it lifts",
    {"lifts of two rules are equal": "True"},
    {"lifts of two rules are equal": "False"},
    _fixed_lifts,
)

# -- structured terms ----------------------------------------------------


def _signature_reads(extra: int) -> int:
    """How many declarations a fold of a small term visits.

    Parameters
    ----------
    extra : int
        The number of unused constructors.

    Returns
    -------
    int
        The number of entries of the dictionaries that are built.
    """
    encode = encoder(parameters(), extra)
    reads = {
        "constructors": Signature.constructors,
        "sorts": Signature.sorts,
        "binders": Signature.binders,
    }
    functions = {}
    for name, member in reads.items():
        assert isinstance(member, property)
        assert member.fget is not None
        functions[name] = member.fget
    with CallCounter(functions) as calls:
        encode(asymmetric(4))
    sizes = {"constructors": 2 + extra, "sorts": 1, "binders": 0}
    return sum(calls.counts[name] * sizes[name] for name in sizes)


record(
    "signature-rebuilt-per-node",
    "structural/signature.py: Signature.constructors, sorts, binders",
    "the declarations are put in a new dictionary at every read, so a fold costs "
    "the size of the term times the size of the signature",
    {"entries visited beside 64 unused constructors": "1196"},
    {"entries visited beside 64 unused constructors": "44"},
    lambda: {
        "entries visited beside 64 unused constructors": str(_signature_reads(64))
    },
    lambda: {"entries visited beside 64 unused constructors": str(_signature_reads(0))},
)


def _depth_columns() -> Facts:
    """The probability of a tree whose leaves carry a label at the depth bound.

    Returns
    -------
    Facts
        The share of one of two terminal operations at the bound.
    """
    base = _two_leaves(1)
    vector = torch.zeros(1, dtype=torch.float64)
    inner = node(leaf(), leaf())
    whole = float(base.log_prob(inner, vector))
    weights = [math.exp(v) for v in _HEAD]
    choice = math.log(weights[0] / math.fsum(weights))
    left_over = (whole - choice) / 2
    return {"odds of the first leaf at the bound": number(math.exp(left_over))}


#: The structure logits of node, leaf, and other leaf.
_HEAD = (0.0, 1.0, 3.0)


def _two_leaves(depth: int) -> Decoder:
    """A decoder whose signature lists a node before its two leaves.

    Parameters
    ----------
    depth : int
        The depth bound.

    Returns
    -------
    Decoder
        The decoder.
    """
    from_terms = _tree_signature()
    head = torch.tensor(_HEAD, dtype=torch.float64)
    return Decoder(
        name="trees",
        signature=from_terms,
        sort_dims={"T": 1},
        depth=depth,
        structure_fns={"T": lambda vector: head},
        primitive_fns={},
        factor_fns={"T": {2: lambda vector: (vector, vector)}},
        binder_select_fn=lambda vector, variables: torch.zeros(len(variables)),
        data_vocab={},
    )


def _tree_signature() -> Signature:
    """A signature that lists a node before two leaves.

    Returns
    -------
    Signature
        The signature.
    """
    plain = encoder(parameters()).signature
    first, second = plain.constructors_t
    return plain.with_(
        constructors_t=(second, first, first.with_(name="Other")),
    )


record(
    "depth-bound-shifts-choices",
    "structural/decoder.py: _structure_logits",
    "at the depth bound the remaining operations read the first logits, so an "
    "operation is scored by the logit of another one",
    {"odds of the first leaf at the bound": "0.268941"},
    {"odds of the first leaf at the bound": "0.119203"},
    _depth_columns,
    lambda: {
        "odds of the first leaf at the bound": number(
            math.exp(_HEAD[1]) / (math.exp(_HEAD[1]) + math.exp(_HEAD[2]))
        )
    },
)

# -- inference consumers -------------------------------------------------


def _first_row() -> Facts:
    """The densities a joint guide gives two different rows.

    Returns
    -------
    Facts
        Whether the two densities are equal.
    """
    facts = {}
    with double_precision(), torch.no_grad():
        for guide in ("mvn", "low-rank", "structured"):
            torch.manual_seed(0)
            built = GUIDES[guide](PROBLEMS["beta"])
            found = built.log_prob(
                torch.zeros(2, 1), {"theta": torch.tensor([0.5, 0.9])}
            )
            facts[f"{guide}: density of 0.9 equals that of 0.5"] = str(
                float(found[0]) == float(found[1])
            )
    return facts


def _first_row_required() -> Facts:
    """Whether the same guides give the rows equal densities one at a time.

    Returns
    -------
    Facts
        The comparison.
    """
    facts = {}
    with double_precision(), torch.no_grad():
        for guide in ("mvn", "low-rank", "structured"):
            torch.manual_seed(0)
            built = GUIDES[guide](PROBLEMS["beta"])
            one, two = (
                float(
                    built.log_prob(torch.zeros(1, 1), {"theta": torch.tensor([value])})
                )
                for value in (0.5, 0.9)
            )
            facts[f"{guide}: density of 0.9 equals that of 0.5"] = str(one == two)
    return facts


record(
    "guide-scores-first-row",
    "inference/guides: _MVNCommon.log_prob, AutoStructured.log_prob",
    "a joint guide scores only the first of several rows and repeats it",
    {
        f"{guide}: density of 0.9 equals that of 0.5": "True"
        for guide in ("mvn", "low-rank", "structured")
    },
    {
        f"{guide}: density of 0.9 equals that of 0.5": "False"
        for guide in ("mvn", "low-rank", "structured")
    },
    _first_row,
    _first_row_required,
)


def _laplace_variance() -> Facts:
    """The variance of a Laplace approximation.

    Returns
    -------
    Facts
        The variance.
    """
    problem = PROBLEMS["beta"]
    with double_precision():
        built = GUIDES["laplace"](problem)
        assert isinstance(built, AutoLaplaceApproximation)
        x, data = problem.data()
        with torch.no_grad():
            built.map_z.fill_(math.log(5 / 4))
        built.fit_hessian(problem.model(), x, data)
        scale = built._scale_tril
        assert isinstance(scale, torch.Tensor)
    return {"variance": number(float(scale[0, 0]) ** 2)}


record(
    "laplace-adds-jitter",
    "inference/guides/laplace.py: AutoLaplaceApproximation.fit_hessian",
    "a constant is added to the covariance of a well-conditioned curvature, so "
    "the approximation is not the Gaussian of the inverse Hessian it states",
    {"variance": "0.4501"},
    {"variance": "0.45"},
    _laplace_variance,
    # Five successes and four failures: the curvature is (a + b) s (1 - s).
    lambda: {"variance": number(float(1 / (9 * Fraction(5, 9) * Fraction(4, 9))))},
)
