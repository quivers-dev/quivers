"""Checks of the agenda engine against enumerated derivations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from functools import partial

import torch

from quivers.stochastic.agenda import (
    Agenda,
    AgendaResult,
    Bindings,
    ChartView,
    FIFOAgenda,
    InferenceRule,
    Item,
    LIFOAgenda,
    Pattern,
    Wildcard,
    knuth_agenda,
    make_wildcard,
    run_agenda,
)
from quivers.stochastic.semiring import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    VITERBI,
    ChartSemiring,
)
from tests.evidence.catalogue import OPERATIONS
from tests.evidence.checks._registry import add
from tests.evidence.compare import (
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_number,
    compare_tensor,
)
from tests.evidence.oracles import closed_forms, derivations
from tests.evidence.oracles.derivations import Term, Tree, Variable
from tests.evidence.problems import (
    DEDUCTIONS,
    DeductionProblem,
    accuracy_of,
    logarithm,
    trees_of,
)

AGENDAS: dict[str, Callable[[], Agenda]] = {
    "fifo": FIFOAgenda,
    "lifo": LIFOAgenda,
    "priority": knuth_agenda,
}
"""The agenda disciplines."""

SEMIRINGS: dict[str, ChartSemiring] = {
    "logprob": LOG_PROB,
    "viterbi": VITERBI,
    "counting": COUNTING,
    "boolean": BOOLEAN,
}
"""The chart semirings."""

#: Problems whose weights leave the range of float64 in the linear domain.
LOG_ONLY = frozenset({"tiny"})
#: Problems a derivation of which does not exist.
UNDERIVABLE = frozenset({"impossible"})


def pattern(term: Term) -> Pattern:
    """Read an oracle pattern as the engine's.

    Parameters
    ----------
    term : Term
        The pattern.

    Returns
    -------
    Pattern
        The engine's pattern.
    """
    if isinstance(term, Variable):
        return make_wildcard(str(term))
    assert isinstance(term, tuple)
    return tuple(
        pattern(part) if isinstance(part, (tuple, Variable)) else part for part in term
    )


def weighted(
    weight: torch.Tensor,
) -> Callable[[Bindings, tuple[torch.Tensor, ...], ChartSemiring], torch.Tensor]:
    """A rule weight times the product of the premise weights.

    Parameters
    ----------
    weight : torch.Tensor
        The rule's weight.

    Returns
    -------
    Callable
        The weight function.
    """

    def apply(
        bindings: Bindings,
        premises: tuple[torch.Tensor, ...],
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """Multiply the premises and the rule weight.

        Parameters
        ----------
        bindings : Bindings
            The bound variables, unused.
        premises : tuple of torch.Tensor
            The premise weights.
        semiring : ChartSemiring
            The semiring.

        Returns
        -------
        torch.Tensor
            The product.
        """
        del bindings
        product = weight
        for premise in premises:
            product = semiring.times(product, premise)
        return product

    return apply


def rules_of(
    problem: DeductionProblem, leaves: Mapping[str, torch.Tensor]
) -> tuple[InferenceRule, ...]:
    """The engine's rules for a problem.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    leaves : Mapping[str, torch.Tensor]
        The weight tensor of every occurrence and rule weight.

    Returns
    -------
    tuple of InferenceRule
        The rules in order.
    """
    built = []
    for name, premises, conclusion, weight in problem.rules:
        built.append(
            InferenceRule(
                name=name,
                premises=tuple(pattern(premise) for premise in premises),
                conclusion=pattern(conclusion),
                weight_fn=None if weight is None else weighted(leaves[weight]),
                side_condition=problem.sides.get(name),
            )
        )
    return tuple(built)


def encode(value: Fraction, semiring: str) -> float:
    """A rational weight in a semiring's representation.

    Parameters
    ----------
    value : Fraction
        The weight.
    semiring : str
        The semiring's name.

    Returns
    -------
    float
        Its representation.
    """
    if semiring in ("logprob", "viterbi"):
        return logarithm(value)
    if semiring == "boolean":
        return float(value != 0)
    return float(value)


def leaves_of(
    problem: DeductionProblem, semiring: str, *, requires_grad: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The weight tensors of a problem, by occurrence and by parameter.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    semiring : str
        The semiring's name.
    requires_grad : bool
        Whether the parameters are differentiated.

    Returns
    -------
    tuple of (dict, dict)
        One tensor per occurrence, computed from one tensor per
        parameter.
    """
    parameters = {
        name: torch.tensor(
            encode(Fraction(value), semiring),
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        for name, value in problem.values.items()
    }
    occurrences = {
        name: parameters[parameter].clone()
        for name, parameter in problem.weights.items()
    }
    return occurrences, parameters


def run(
    problem: DeductionProblem,
    semiring: ChartSemiring,
    discipline: str,
    leaves: Mapping[str, torch.Tensor],
    *,
    max_iterations: int = 100_000,
) -> AgendaResult:
    """Run the agenda engine on a problem.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    semiring : ChartSemiring
        The semiring.
    discipline : str
        The agenda's name.
    leaves : Mapping[str, torch.Tensor]
        The weight tensors.
    max_iterations : int
        The engine's budget.

    Returns
    -------
    AgendaResult
        The run.
    """
    goal = problem.goal

    def is_goal(item: Item) -> bool:
        """Whether an item is the goal.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        bool
            Whether it is.
        """
        return item == goal

    return run_agenda(
        [(_item(item), leaves[occurrence]) for occurrence, item in problem.axioms],
        rules_of(problem, leaves),
        semiring=semiring,
        agenda=AGENDAS[discipline](),
        goal=is_goal,
        max_iterations=max_iterations,
    )


def _item(term: Term) -> Item:
    """Read an oracle item as the engine's.

    Parameters
    ----------
    term : Term
        The item.

    Returns
    -------
    Item
        The item.
    """
    assert isinstance(term, tuple)
    return term


def reference(problem: DeductionProblem, semiring: str) -> float:
    """The weight of a problem's goal from its derivations.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    semiring : str
        The semiring's name.

    Returns
    -------
    float
        The weight in the semiring's representation.
    """
    trees = trees_of(problem.name)
    rational = problem.rational()
    if semiring == "viterbi":
        return logarithm(derivations.best(trees, rational))
    mass = derivations.total(trees, rational, Fraction(0), Fraction(1))
    if semiring == "logprob":
        return logarithm(mass)
    if semiring == "boolean":
        return float(mass != 0)
    return float(mass)


def domain_of(semiring: str) -> str:
    """The error metric of a semiring's representation.

    Parameters
    ----------
    semiring : str
        The semiring's name.

    Returns
    -------
    str
        ``"log"`` or ``"linear"``.
    """
    return "log" if semiring in ("logprob", "viterbi") else "linear"


def check_value(name: str, semiring: str, discipline: str) -> None:
    """Compare a goal weight with the sum over derivations.

    Parameters
    ----------
    name : str
        The problem.
    semiring : str
        The semiring's name.
    discipline : str
        The agenda's name.
    """
    problem = DEDUCTIONS[name]
    leaves, _ = leaves_of(problem, semiring)
    result = run(problem, SEMIRINGS[semiring], discipline, leaves)
    value = ChartView(result).goal_weight()
    wanted = reference(problem, semiring)
    compare_tensor(
        value,
        torch.tensor(wanted, dtype=torch.float64),
        accuracy_of(name, domain_of(semiring), OPERATIONS),
        f"{discipline} {semiring} {name}",
    )


def check_count(name: str, discipline: str) -> None:
    """Compare a derivation count, exactly, with the number of trees.

    Parameters
    ----------
    name : str
        The problem.
    discipline : str
        The agenda's name.

    Raises
    ------
    Mismatch
        If the count differs.
    """
    problem = DEDUCTIONS[name]
    leaves = {key: torch.tensor(1.0, dtype=torch.float64) for key in problem.weights}
    value = float(ChartView(run(problem, COUNTING, discipline, leaves)).goal_weight())
    wanted = len(trees_of(name))
    if value != wanted:
        raise Mismatch("value", abs(value - wanted), f"{value} trees for {wanted}")


def gradient(
    problem: DeductionProblem, semiring: str, discipline: str
) -> dict[str, torch.Tensor | None]:
    """The gradient of a goal weight by occurrence and by parameter.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    semiring : str
        The semiring's name.
    discipline : str
        The agenda's name.

    Returns
    -------
    dict of str to (torch.Tensor or None)
        The derivative in each occurrence and each parameter.
    """
    leaves, parameters = leaves_of(problem, semiring, requires_grad=True)
    result = run(problem, SEMIRINGS[semiring], discipline, leaves)
    value = ChartView(result).goal_weight()
    names = [*(f"occurrence {key}" for key in leaves), *parameters]
    if not value.requires_grad:
        return dict.fromkeys(names)
    found = torch.autograd.grad(
        value, [*leaves.values(), *parameters.values()], allow_unused=True
    )
    return dict(zip(names, found, strict=True))


def gradient_reference(problem: DeductionProblem, semiring: str) -> dict[str, float]:
    """The exact gradient by occurrence and by parameter.

    Parameters
    ----------
    problem : DeductionProblem
        The problem.
    semiring : str
        ``"logprob"``, for the derivative of a log weight in log weights,
        or ``"counting"``, for the derivative of a weight in weights.

    Returns
    -------
    dict of str to float
        The derivatives.
    """
    trees = trees_of(problem.name)
    rational = problem.rational()
    exact: dict[str, Fraction] = {}
    for key in problem.weights:
        if semiring == "logprob":
            exact[key] = derivations.logarithmic_gradient(trees, rational, key)
        else:
            exact[key] = _linear_gradient(trees, rational, key)
    by_parameter = dict.fromkeys(problem.values, Fraction(0))
    for key, parameter in problem.weights.items():
        by_parameter[parameter] += exact[key]
    return {
        **{f"occurrence {key}": float(value) for key, value in exact.items()},
        **{key: float(value) for key, value in by_parameter.items()},
    }


def _linear_gradient(
    trees: tuple[Tree, ...], rational: Mapping[str, Fraction], key: str
) -> Fraction:
    """The derivative of a sum of products in one of its factors.

    Parameters
    ----------
    trees : tuple of Tree
        The derivations.
    rational : Mapping[str, Fraction]
        The weights.
    key : str
        The factor.

    Returns
    -------
    Fraction
        The derivative, by the product rule on every tree.
    """
    answer = Fraction(0)
    for tree in trees:
        names = derivations.factors(tree)
        for position, name in enumerate(names):
            if name != key:
                continue
            product = Fraction(1)
            for other, factor in enumerate(names):
                if other != position:
                    product *= rational[factor]
            answer += product
    return answer


def check_gradient(name: str, semiring: str, discipline: str) -> None:
    """Compare a gradient with the exact one, by occurrence and parameter.

    Parameters
    ----------
    name : str
        The problem.
    semiring : str
        The semiring's name.
    discipline : str
        The agenda's name.
    """
    problem = DEDUCTIONS[name]
    compare_gradients(
        gradient(problem, semiring, discipline),
        gradient_reference(problem, semiring),
        accuracy_of(name, "linear", OPERATIONS, backward=True),
        f"{discipline} {semiring} {name}",
    )


def unfoldings(result: AgendaResult, goal: Item, axioms: list[Item]) -> int:
    """The number of trees the recorded rule applications unfold to.

    Parameters
    ----------
    result : AgendaResult
        A run on an acyclic problem.
    goal : Item
        The goal item.
    axioms : list of Item
        The axiom items, one per occurrence.

    Returns
    -------
    int
        The number of trees.
    """
    view = ChartView(result)
    known: dict[Item, int] = {}

    def count(item: Item) -> int:
        """The number of trees of one item.

        Parameters
        ----------
        item : Item
            The item.

        Returns
        -------
        int
            The number.
        """
        if item not in known:
            known[item] = axioms.count(item) + sum(
                math.prod(count(premise) for premise in premises)
                for _, premises in view.derivations(item)
            )
        return known[item]

    return count(goal)


def check_forest(name: str, discipline: str) -> None:
    """Compare the unfolding of the provenance with the number of trees.

    Parameters
    ----------
    name : str
        The problem.
    discipline : str
        The agenda's name.

    Raises
    ------
    Mismatch
        If the numbers differ.
    """
    problem = DEDUCTIONS[name]
    leaves, _ = leaves_of(problem, "logprob")
    result = run(problem, LOG_PROB, discipline, leaves)
    found = unfoldings(
        result, _item(problem.goal), [_item(item) for _, item in problem.axioms]
    )
    wanted = len(trees_of(name))
    if found != wanted:
        raise Mismatch("value", abs(found - wanted), f"{found} trees for {wanted}")
    instances = derivations.ground(problem.axioms, problem.rules, problem.sides)
    if len(result.applications) != len(instances):
        raise Mismatch(
            "value",
            abs(len(result.applications) - len(instances)),
            f"{len(result.applications)} applications for {len(instances)}",
        )


#: Proofs of two readings of one string: the reading, its weight names.
READINGS = (("low", "a"), ("low", "b"), ("high", "c"), ("high", "a"))
#: The weights of the proofs of the readings.
READING_WEIGHTS = {"a": "1/4", "b": "1/4", "c": "1/5"}


def check_outcomes(discipline: str) -> None:
    """Compare the mass of each outcome with the sum over its proofs.

    The outcome is part of the item, so every outcome has its own weight.
    Two proofs of one outcome have equal weight.

    Parameters
    ----------
    discipline : str
        The agenda's name.
    """
    axioms = [(f"w{i}", ("proof", i, reading)) for i, (reading, _) in enumerate(READINGS)]
    rules = [
        ("read", (("proof", Variable("i"), Variable("r")),), ("means", Variable("r")), None)
    ]
    weight_of = {
        f"w{i}": Fraction(READING_WEIGHTS[key]) for i, (_, key) in enumerate(READINGS)
    }
    result = run_agenda(
        [
            (item, torch.tensor(logarithm(weight_of[key]), dtype=torch.float64))
            for key, item in axioms
        ],
        [
            InferenceRule(
                name=name,
                premises=tuple(pattern(p) for p in premises),
                conclusion=pattern(conclusion),
            )
            for name, premises, conclusion, _ in rules
        ],
        semiring=LOG_PROB,
        agenda=AGENDAS[discipline](),
    )
    for reading in ("low", "high"):
        trees = derivations.derivations(axioms, rules, ("means", reading))
        wanted = logarithm(
            derivations.total(trees, weight_of, Fraction(0), Fraction(1))
        )
        found = result.chart.get(("means", reading))
        assert found is not None
        compare_number(
            float(found),
            wanted,
            Accuracy(domain="log", operations=200, magnitude=4.0),
            f"{discipline} outcome {reading}",
        )


def check_nonlinear_observer(discipline: str) -> None:
    """Compare the expected square of a proof value with enumeration.

    Two proofs have the values minus one and one. The square is taken of
    each proof's value, which the items keep apart, and not of an
    aggregate.

    Parameters
    ----------
    discipline : str
        The agenda's name.
    """
    values = {"minus": Fraction(-1), "plus": Fraction(1)}
    axioms = [(key, ("value", key)) for key in values]
    rules = [("keep", (("value", Variable("k")),), ("kept", Variable("k")), None)]
    wanted = sum(
        (
            derivations.nonlinear_fold(
                derivations.derivations(axioms, rules, ("kept", key)),
                dict.fromkeys(values, Fraction(1)),
                lambda name, children: children[0] if children else values[name],
                lambda value: value * value,
            )
            for key in values
        ),
        Fraction(0),
    )
    result = run_agenda(
        [
            (item, torch.tensor(float(values[key]), dtype=torch.float64))
            for key, item in axioms
        ],
        [
            InferenceRule(
                name=name,
                premises=tuple(pattern(p) for p in premises),
                conclusion=pattern(conclusion),
            )
            for name, premises, conclusion, _ in rules
        ],
        semiring=COUNTING,
        agenda=AGENDAS[discipline](),
    )
    kept = result.chart.lookup(("kept", Wildcard(name="k")))
    found = square_of_each([weight for _, weight in kept])
    compare_number(
        float(found),
        wanted,
        Accuracy(domain="linear", operations=8),
        f"{discipline} square of each proof",
    )


def square_of_each(weights: list[torch.Tensor]) -> torch.Tensor:
    """The sum of the squares of separate values.

    Parameters
    ----------
    weights : list of torch.Tensor
        The values.

    Returns
    -------
    torch.Tensor
        The sum of squares.
    """
    return torch.stack(weights).square().sum()


#: Cycle weights of the geometric cases, away from and near the boundary.
CYCLES = ("1/2", "9/10", "99/100", "999/1000")
#: The relative roundings of one turn around a cycle.
TURN = 24


def geometric_run(ratio: Fraction, discipline: str) -> torch.Tensor:
    """Run a one-item cycle until its number stops changing.

    Parameters
    ----------
    ratio : Fraction
        The weight of one turn.
    discipline : str
        The agenda's name.

    Returns
    -------
    torch.Tensor
        The item's log weight.
    """
    turn = torch.tensor(logarithm(ratio), dtype=torch.float64)
    result = run_agenda(
        [(("A",), torch.tensor(0.0, dtype=torch.float64))],
        [
            InferenceRule(
                name="turn",
                premises=(("A",),),
                conclusion=("A",),
                weight_fn=weighted(turn),
            )
        ],
        semiring=LOG_PROB,
        agenda=AGENDAS[discipline](),
    )
    weight = result.chart.get(("A",))
    assert weight is not None
    return weight


def check_cycle(ratio: str, discipline: str) -> None:
    """Compare a cycle's weight with the geometric closed form.

    The engine adds powers until the sum stops changing. Each power
    carries the roundings of its turns, and the omitted tail is at most
    one rounding of the sum divided by ``1 - c``. Both are covered by the
    condition number ``1 / (1 - c)`` times the roundings of the number of
    turns made.

    Parameters
    ----------
    ratio : str
        The weight of one turn.
    discipline : str
        The agenda's name.
    """
    exact = Fraction(ratio)
    turns = math.ceil(40 / (1 - float(exact)))
    compare_number(
        float(geometric_run(exact, discipline)),
        logarithm(closed_forms.geometric(exact)),
        Accuracy(
            domain="log",
            operations=TURN * turns,
            magnitude=40.0,
            condition=float(1 / (1 - exact)),
        ),
        f"{discipline} cycle {ratio}",
    )


def check_branching(branch: str, discipline: str) -> None:
    """Compare the least root of ``x = p x^2 + q`` with its closed form.

    Parameters
    ----------
    branch : str
        The weight ``p``.
    discipline : str
        The agenda's name.
    """
    p = Fraction(branch)
    result = run_agenda(
        [(("X",), torch.tensor(logarithm(1 - p), dtype=torch.float64))],
        [
            InferenceRule(
                name="split",
                premises=(("X",), ("X",)),
                conclusion=("X",),
                weight_fn=weighted(torch.tensor(logarithm(p), dtype=torch.float64)),
            )
        ],
        semiring=LOG_PROB,
        agenda=AGENDAS[discipline](),
    )
    weight = result.chart.get(("X",))
    assert weight is not None
    root = closed_forms.least_quadratic_root(p)
    slope = 2 * p * root
    compare_number(
        float(weight),
        logarithm(root),
        Accuracy(
            domain="log",
            operations=TURN * math.ceil(40 / (1 - float(slope))),
            magnitude=40.0,
            condition=float(1 / (1 - slope)),
        ),
        f"{discipline} branching {branch}",
    )


def check_budget(discipline: str) -> None:
    """Require a refusal when a cycle does not settle within its budget.

    Parameters
    ----------
    discipline : str
        The agenda's name.

    Raises
    ------
    Mismatch
        If the run returns a number.
    """
    try:
        result = run_agenda(
            [(("A",), torch.tensor(0.0, dtype=torch.float64))],
            [InferenceRule(name="turn", premises=(("A",),), conclusion=("A",))],
            semiring=LOG_PROB,
            agenda=AGENDAS[discipline](),
            max_iterations=50,
        )
    except RuntimeError as refusal:
        wanted = "agenda exceeded max_iterations=50; likely a non-terminating deduction"
        if str(refusal) != wanted:
            raise Mismatch("refusal", math.inf, str(refusal)) from refusal
        return
    raise Mismatch(
        "refusal", math.inf, f"a budget of 50 returned {result.chart.get(('A',))}"
    )


def check_update(discipline: str) -> None:
    """Compare two runs of one rule set across a parameter update.

    Parameters
    ----------
    discipline : str
        The agenda's name.
    """
    problem = DEDUCTIONS["diamond"]
    leaves, _ = leaves_of(problem, "logprob")
    rules = rules_of(problem, leaves)
    axioms = [(_item(item), leaves[key]) for key, item in problem.axioms]
    for shift in (0.0, -0.75):
        with torch.no_grad():
            for name in ("s0", "s1", "o2"):
                leaves[name].add_(shift)
        result = run_agenda(
            axioms, rules, semiring=LOG_PROB, agenda=AGENDAS[discipline]()
        )
        weights = {key: float(leaf.exp()) for key, leaf in leaves.items()}
        wanted = math.log(derivations.total(trees_of("diamond"), weights, 0.0, 1.0))
        found = result.chart.get(_item(problem.goal))
        assert found is not None
        compare_number(
            float(found),
            wanted,
            accuracy_of("diamond", "log", OPERATIONS),
            f"{discipline} after an update of {shift}",
        )


def questions(name: str) -> str:
    """The problem class of a deduction problem.

    Parameters
    ----------
    name : str
        The problem.

    Returns
    -------
    str
        Its column.
    """
    return {
        "edit_lattice": "Q13",
        "skip": "Q10",
        "reach": "Q10",
        "high_arity": "Q6",
    }.get(name, "Q1")


def _register() -> None:
    """Register every check of this module."""
    for discipline in AGENDAS:
        row = f"agenda-{discipline}"
        for name in DEDUCTIONS:
            for semiring in SEMIRINGS:
                if name in LOG_ONLY and domain_of(semiring) == "linear":
                    if semiring == "counting":
                        continue
                if name in UNDERIVABLE:
                    continue
                add(
                    f"deduction/value/{discipline}/{semiring}/{name}",
                    row,
                    questions(name),
                    partial(check_value, name, semiring, discipline),
                )
            if name in UNDERIVABLE:
                continue
            add(
                f"deduction/count/{discipline}/{name}",
                row,
                "Q10" if questions(name) == "Q10" else "Q1",
                partial(check_count, name, discipline),
            )
            add(
                f"deduction/forest/{discipline}/{name}",
                row,
                "Q5",
                partial(check_forest, name, discipline),
            )
            for semiring in ("logprob", "counting"):
                if name in LOG_ONLY and semiring == "counting":
                    continue
                add(
                    f"deduction/gradient/{discipline}/{semiring}/{name}",
                    row,
                    "Q1g" if questions(name) != "Q13" else "Q11",
                    partial(check_gradient, name, semiring, discipline),
                )
        add(f"deduction/outcomes/{discipline}", row, "Q23", partial(check_outcomes, discipline))
        add(
            f"deduction/nonlinear/{discipline}",
            row,
            "Q23",
            partial(check_nonlinear_observer, discipline),
        )
        for ratio in CYCLES:
            add(
                f"deduction/cycle/{discipline}/{ratio}",
                row,
                "Q8",
                partial(check_cycle, ratio, discipline),
            )
        for branch in ("1/5", "3/5", "4/5"):
            if discipline == "fifo":
                add(
                    f"deduction/branching/{discipline}/{branch}",
                    row,
                    "Q9",
                    partial(check_branching, branch, discipline),
                )
        add(f"deduction/budget/{discipline}", row, "Q8", partial(check_budget, discipline))
        add(f"deduction/update/{discipline}", row, "Q24", partial(check_update, discipline))


_register()
