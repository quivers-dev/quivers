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
    ProductFuzzyAlgebra,
)
from quivers.core.morphisms import Morphism, observed
from quivers.core.objects import FinSet, ProductSet
from quivers.dsl import loads
from quivers.dsl.ast_nodes import LetExprCall, LetExprVar, SchemaDecl, TypeName
from quivers.dsl.compiler.programs import LetValue, _ProgramsMixin
from quivers.effects.checked_program import CheckedProgram
from quivers.enriched.kan_extensions import Projection, left_kan
from quivers.inference import AutoLaplaceApproximation
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import (
    run_deduction,
    run_program,
)
from quivers.monadic.instances import Identity
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
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
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
)
from tests.evidence.checks.terms import encoder, node, leaf, parameters
from tests.evidence.compare import Mismatch
from tests.evidence.counters import CallCounter
from tests.evidence.oracles import closed_forms, derivations
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
    return facts


def _residual_required() -> Facts:
    """The log weights of the same cycles in closed form.

    Returns
    -------
    Facts
        The log weight for each turn.
    """
    return {
        f"agenda at {turn}": number(logarithm(closed_forms.geometric(Fraction(turn))))
        for turn in RESIDUAL_TURNS
    }


record(
    "residual-is-the-value",
    "stochastic/agenda.py: tolerance",
    "a run that stops because two successive numbers are close returns the last "
    "number as the weight",
    {
        "agenda at 9/10": "2.29382",
        "agenda at 99/100": "4.51127",
        "agenda at 999/1000": "6.21581",
    },
    {
        "agenda at 9/10": "2.30259",
        "agenda at 99/100": "4.60517",
        "agenda at 999/1000": "6.90776",
    },
    _residual,
    _residual_required,
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


# -- closures, algebras, and contractions --------------------------------


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
