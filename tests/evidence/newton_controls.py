"""Checks and deliberately wrong versions of monotone polynomial plans.

``x = p x^2 + q`` with ``p + q = 1`` has the least fixed point
``min(1, q / p)`` (O6), rational for the weights used here, and is
critical at ``p = 1/2``, where its Jacobian has spectral radius one. A
system of three items in two components, ``x = (2/5) x y + 2/5``,
``y = x`` and ``z = (1/3) z^2 + (9/8) x``, has the least fixed point
``(1/2, 1/2, 3/4)``. The checks compare values with these, decide each
interval exactly (O8: the upper end is a pre-fixed point of the exact
system, the lower end is below the exact solution), compare the implicit
derivative with its closed form (O7), and require refusals and settled
reports where the contract is not met.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import torch

from quivers.core.algebras import COUNTING, LOG_PROB, MARKOV, OrderPreservingSemiring
from quivers.qiec.contracts import (
    NOT_DIFFERENTIATED,
    SPECTRAL_RADIUS_BELOW_ONE,
    AbsoluteLogError,
    AccuracyContract,
    ContractNotMet,
    Certified,
    DerivativeSemantics,
    Exact,
    FailurePolicy,
    InterpretationId,
    MonotonePolynomial,
    NoRetention,
    Numerics,
    OccurrenceId,
    PackedForest,
    ParameterId,
    Phase,
    PresentationId,
    QuerySpec,
    Rational,
    ResourceBudget,
    Target,
    Unestablished,
    Weight,
    measured_total,
)
from quivers.qiec.identifiers import ConstructorId, FamilyId
from quivers.qiec.kinds import IndexBinder, UserIndexSort
from quivers.qiec.rules import (
    AxiomOccurrence,
    AxiomSlot,
    Item,
    Judgment,
    JudgmentDecl,
    Premise,
    Rule,
    RulePresentation,
)
from quivers.qiec.types import IndexLiteral
from quivers.stochastic.planning import (
    NEWTON_ALGORITHM,
    Evaluation,
    Plan,
    PlanRefusal,
    PlanRefused,
    plan,
)
from tests.evidence import scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.mutations import Mutation, Rewrite
from tests.evidence.oracles import closed_forms, enclosures

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar runs."""

type Inputs = dict[ParameterId | OccurrenceId, torch.Tensor]
"""Input weights."""

BOUND = 1e-9
"""The certified bound the contracts ask for, in absolute log error."""


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


def _contract(
    *,
    refuse: bool = True,
    budget: int | None = None,
    implicit: bool = False,
    exact: bool = False,
) -> AccuracyContract:
    """A contract for a least fixed point.

    Parameters
    ----------
    refuse : bool
        Whether a run that does not meet it is refused, else returned as
        unestablished.
    budget : int or None
        The most Newton steps.
    implicit : bool
        Whether the implicit derivative is asked for.
    exact : bool
        Whether an exact value is asked for.

    Returns
    -------
    AccuracyContract
        The contract.
    """
    return AccuracyContract(
        numerics=Numerics(dtype="float64", domain="log"),
        target=Target(quantity="least fixed point", metric=AbsoluteLogError()),
        guarantee=Exact() if exact else Certified(bound=Rational.of(BOUND)),
        budget=ResourceBudget(iterations=budget),
        on_failure=FailurePolicy.REFUSE
        if refuse
        else FailurePolicy.RETURN_UNESTABLISHED,
        gradient=DerivativeSemantics.implicit(SPECTRAL_RADIUS_BELOW_ONE)
        if implicit
        else NOT_DIFFERENTIATED,
    )


# -- one quadratic item ------------------------------------------------------

_NAME = "quadratic"
_X = FamilyId.derive(_NAME, "x")
_JUDGMENT = Judgment(family=_X, arguments=())
_BRANCH = ParameterId.derive(_NAME, "branch")
_STOP = OccurrenceId.derive(_NAME, "stop")
_SEED = AxiomSlot(
    id=ConstructorId.derive(_NAME, "seed"), name="seed", judgment=_JUDGMENT
)
QUADRATIC = RulePresentation(
    id=PresentationId.derive(_NAME),
    name=_NAME,
    judgments=(JudgmentDecl(id=_X, name="x", indices=()),),
    rules=(
        Rule(
            id=ConstructorId.derive(_NAME, "branch"),
            name="branch",
            premises=(Premise(judgment=_JUDGMENT), Premise(judgment=_JUDGMENT)),
            conclusion=_JUDGMENT,
            weight=_BRANCH,
        ),
    ),
    axioms=(_SEED,),
)
"""``x <- x, x`` with weight ``p`` and the input ``x`` with weight ``q``."""

GOAL = Item(family=_X, arguments=())
OCCURRENCES = [AxiomOccurrence(slot=_SEED.id, occurrence=_STOP, binding=())]


def _query(
    presentation: RulePresentation, wrt: Sequence[ParameterId | OccurrenceId] = ()
) -> QuerySpec:
    """The weight query of a presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    wrt : Sequence of ParameterId or OccurrenceId
        The inputs differentiated.

    Returns
    -------
    QuerySpec
        The query.
    """
    return QuerySpec(
        presentation=presentation.id,
        interpretation=InterpretationId.derive(presentation.name, "weight"),
        observable=Weight(),
        differentiate_with_respect_to=tuple(wrt),
    )


def _plan(
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    contract: AccuracyContract,
    wrt: Sequence[ParameterId | OccurrenceId] = (),
) -> Plan:
    """A monotone polynomial plan.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The algebra.
    contract : AccuracyContract
        The contract.
    wrt : Sequence of ParameterId or OccurrenceId
        The inputs differentiated.

    Returns
    -------
    Plan
        The plan.
    """
    return plan(
        _query(presentation, wrt),
        presentation,
        algebra,
        MonotonePolynomial(),
        contract,
        NoRetention(),
    )


def _weights(branch: Fraction, log: bool) -> Inputs:
    """The weights ``p`` and ``q = 1 - p``.

    Parameters
    ----------
    branch : Fraction
        ``p``.
    log : bool
        Whether log weights are given.

    Returns
    -------
    Inputs
        The weights.
    """
    stop = 1 - branch
    encode = (lambda value: math.log(value)) if log else float
    return {
        _BRANCH: torch.tensor(encode(branch), dtype=torch.float64),
        _STOP: torch.tensor(encode(stop), dtype=torch.float64),
    }


def _system(branch: Fraction) -> list[list[enclosures.Monomial]]:
    """``x = p x^2 + q`` for the enclosure oracle.

    Parameters
    ----------
    branch : Fraction
        ``p``.

    Returns
    -------
    list of list of Monomial
        The polynomial.
    """
    return [[(branch, (0, 0)), (1 - branch, ())]]


_BRANCHES = (Fraction(1, 3), Fraction(2, 5), Fraction(2, 3), Fraction(4, 5))
"""Branch weights away from the critical one half."""


def _evaluation(result: Evaluation | Unestablished, what: str) -> Evaluation:
    """A run that met its contract.

    Parameters
    ----------
    result : Evaluation or Unestablished
        The run.
    what : str
        A description.

    Returns
    -------
    Evaluation
        The run.

    Raises
    ------
    Mismatch
        If it is unestablished.
    """
    if not isinstance(result, Evaluation):
        raise Mismatch("value", math.inf, f"{what}: unestablished")
    return result


def check_closed_form() -> None:
    """Require the least root and an interval holding it, decided exactly (O6, O8).

    Raises
    ------
    Mismatch
        If a value differs, an interval misses the exact root, or its upper
        end is not a pre-fixed point of the exact system.
    """
    for algebra, log in ((MARKOV, False), (LOG_PROB, True)):
        for branch in _BRANCHES:
            evaluation = _evaluation(
                _plan(QUADRATIC, algebra, _contract()).run(
                    OCCURRENCES, _weights(branch, log), like=LIKE
                ),
                f"p = {branch}",
            )
            root = closed_forms.least_quadratic_root(branch)
            value = float(evaluation.value(GOAL))
            compare_number(
                value,
                math.log(root) if log else float(root),
                Accuracy(domain="log" if log else "linear", operations=64),
                f"least root at p = {branch}",
            )
            low, high = evaluation.lower[GOAL], evaluation.upper[GOAL]
            _require(enclosures.encloses(low, high, root), f"p = {branch}: misses")
            _require(
                enclosures.pre_fixed(_system(branch), [high]),
                f"p = {branch}: the upper end {high} is not a pre-fixed point",
            )
            _require(evaluation.newton[0].certified, f"p = {branch}: not certified")


# -- three items in two components -----------------------------------------

_SNAME = "decomposed"
_SORT = UserIndexSort(name="Var", constructors=("x", "y", "z"))
_V = FamilyId.derive(_SNAME, "v")


def _v(name: str) -> Judgment:
    """``v(name)``.

    Parameters
    ----------
    name : str
        The variable.

    Returns
    -------
    Judgment
        The pattern.
    """
    return Judgment(family=_V, arguments=(IndexLiteral(value=name, sort=_SORT),))


def _rule(name: str, premises: Sequence[str], conclusion: str) -> Rule:
    """A weighted rule.

    Parameters
    ----------
    name : str
        The rule.
    premises : Sequence of str
        The premises' variables.
    conclusion : str
        The conclusion's variable.

    Returns
    -------
    Rule
        The rule.
    """
    return Rule(
        id=ConstructorId.derive(_SNAME, name),
        name=name,
        premises=tuple(Premise(judgment=_v(p)) for p in premises),
        conclusion=_v(conclusion),
        weight=ParameterId.derive(_SNAME, name),
    )


_XSEED = AxiomSlot(
    id=ConstructorId.derive(_SNAME, "seed"), name="seed", judgment=_v("x")
)
DECOMPOSED = RulePresentation(
    id=PresentationId.derive(_SNAME),
    name=_SNAME,
    index_sorts=(_SORT,),
    judgments=(
        JudgmentDecl(
            id=_V,
            name="v",
            indices=(IndexBinder(name="n", sort=_SORT, refinable=True),),
        ),
    ),
    rules=(
        _rule("xy", ("x", "y"), "x"),
        _rule("yx", ("x",), "y"),
        _rule("zz", ("z", "z"), "z"),
        _rule("zx", ("x",), "z"),
    ),
    axioms=(_XSEED,),
)
"""``x = a x y + b``, ``y = x``, ``z = c z^2 + e x``."""

_WEIGHTS = {
    "xy": Fraction(2, 5),
    "yx": Fraction(1),
    "zz": Fraction(1, 3),
    "zx": Fraction(9, 8),
}
_B = Fraction(2, 5)
_DSEED = OccurrenceId.derive(_SNAME, "seed")
_SOLUTION = {"x": Fraction(1, 2), "y": Fraction(1, 2), "z": Fraction(3, 4)}


def _decomposed_system() -> list[list[enclosures.Monomial]]:
    """The three-item system for the enclosure oracle, in the order x, y, z.

    Returns
    -------
    list of list of Monomial
        The polynomials.
    """
    return [
        [(_WEIGHTS["xy"], (0, 1)), (_B, ())],
        [(_WEIGHTS["yx"], (0,))],
        [(_WEIGHTS["zz"], (2, 2)), (_WEIGHTS["zx"], (0,))],
    ]


def check_decomposed() -> None:
    """Require two components solved in order, each interval decided exactly (O8).

    Raises
    ------
    Mismatch
        If the components are not ``{x, y}`` then ``{z}``, an interval misses
        the exact solution, or the upper ends are not a pre-fixed point.
    """
    inputs: Inputs = {
        ParameterId.derive(_SNAME, name): torch.tensor(
            math.log(value), dtype=torch.float64
        )
        for name, value in _WEIGHTS.items()
    }
    inputs[_DSEED] = torch.tensor(math.log(_B), dtype=torch.float64)
    occurrence = AxiomOccurrence(slot=_XSEED.id, occurrence=_DSEED, binding=())
    evaluation = _evaluation(
        _plan(DECOMPOSED, LOG_PROB, _contract()).run([occurrence], inputs, like=LIKE),
        "three items",
    )
    order = [
        tuple(sorted(str(item.arguments[0]) for item in part.items))
        for part in evaluation.newton
    ]
    _require(order == [("x", "y"), ("z",)], f"components in order {order}")
    items = {name: Item(family=_V, arguments=(name,)) for name in "xyz"}
    upper = []
    for name, item in items.items():
        low, high = evaluation.lower[item], evaluation.upper[item]
        _require(
            enclosures.encloses(low, high, _SOLUTION[name]),
            f"{name}: [{low}, {high}] misses {_SOLUTION[name]}",
        )
        compare_number(
            float(evaluation.value(item)),
            math.log(_SOLUTION[name]),
            Accuracy(domain="log", operations=64),
            f"{name}",
        )
        upper.append(high)
    _require(
        enclosures.pre_fixed(_decomposed_system(), upper),
        f"the upper ends {upper} are not a pre-fixed point",
    )


def check_gradient() -> None:
    """Require the implicit derivative of the least root in closed form (O7).

    From ``x = p x^2 + q``: ``dx/dp = x^2 / (1 - 2 p x)`` and
    ``dx/dq = 1 / (1 - 2 p x)``, so in log weights
    ``d log x / d log p = p x / (1 - 2 p x)`` and
    ``d log x / d log q = q / (x (1 - 2 p x))``.

    Raises
    ------
    Mismatch
        If a derivative differs.
    """
    for branch in _BRANCHES:
        root = closed_forms.least_quadratic_root(branch)
        stop = 1 - branch
        slope = 1 - 2 * branch * root
        current = _plan(QUADRATIC, LOG_PROB, _contract(implicit=True), (_BRANCH, _STOP))
        found = current.gradient(OCCURRENCES, _weights(branch, True), GOAL, like=LIKE)
        _require(isinstance(found, dict), f"p = {branch}: unestablished derivative")
        assert isinstance(found, dict)
        for key, wanted in (
            (_BRANCH, branch * root / slope),
            (_STOP, stop / (root * slope)),
        ):
            compare_number(
                float(found[key]),
                float(wanted),
                Accuracy(domain="linear", operations=256, condition=float(1 / slope)),
                f"p = {branch}: derivative in {key}",
            )


def check_critical() -> None:
    """Require the critical system to be left uncertified, with its conditioning.

    At ``p = 1/2`` the only pre-fixed point is the root itself, so no
    upper end above an estimate from below certifies: the contract refuses
    or returns the value unestablished, and the component reports a
    spectral radius near one and a large condition number.

    Raises
    ------
    Mismatch
        If a certificate is issued, or the refusal or the report differs.
    """
    weights = _weights(Fraction(1, 2), True)
    try:
        _plan(QUADRATIC, LOG_PROB, _contract()).run(OCCURRENCES, weights, like=LIKE)
    except ContractNotMet as refused:
        _require("guarantee" in str(refused), f"refused for {refused}")
    else:
        raise Mismatch("refusal", math.inf, "a critical system was certified")
    returned = _plan(QUADRATIC, LOG_PROB, _contract(refuse=False)).settled(
        OCCURRENCES, weights, like=LIKE
    )
    _require(isinstance(returned, Unestablished), "the critical run was established")
    assert isinstance(returned, Unestablished)
    evaluation = returned.value
    assert isinstance(evaluation, Evaluation) and evaluation.accuracy is not None
    reason = getattr(evaluation.accuracy.attainment.guarantee, "reason", "")
    _require("condition number" in reason, f"the report says {reason!r}")
    part = evaluation.newton[0]
    _require(
        part.condition is not None and part.condition > 1e6,
        f"the condition number reported is {part.condition}",
    )
    _require(
        part.spectral is not None and abs(part.spectral - 1) < 1e-6,
        f"the spectral radius reported is {part.spectral}",
    )


def check_budget() -> None:
    """Require a budget that ends the iteration to be reported, never as a value.

    Raises
    ------
    Mismatch
        If two steps are reported complete, or their value is returned as
        meeting the contract.
    """
    weights = _weights(Fraction(2, 3), True)
    returned = _plan(QUADRATIC, LOG_PROB, _contract(refuse=False, budget=2)).settled(
        OCCURRENCES, weights, like=LIKE
    )
    _require(isinstance(returned, Unestablished), "two steps met the contract")
    assert isinstance(returned, Unestablished)
    evaluation = returned.value
    assert isinstance(evaluation, Evaluation) and evaluation.accuracy is not None
    stop = evaluation.accuracy.stop.tag
    _require(stop == "budget-exhausted", f"stopped by {stop}")
    _refused(
        lambda: _plan(QUADRATIC, LOG_PROB, _contract(refuse=False)).run(
            OCCURRENCES, weights, like=LIKE
        ),
        PlanRefusal.INADMISSIBLE,
        "a plain run of a contract that returns unestablished values",
    )
    try:
        _plan(QUADRATIC, LOG_PROB, _contract(budget=2)).run(
            OCCURRENCES, weights, like=LIKE
        )
    except ContractNotMet:
        return
    raise Mismatch("refusal", math.inf, "an exhausted budget was not refused")


def _refused(
    attempt: Callable[[], Evaluation | Unestablished | Plan],
    code: PlanRefusal,
    what: str,
) -> None:
    """Require an attempt to be refused for a reason.

    Parameters
    ----------
    attempt : Callable
        The attempt.
    code : PlanRefusal
        The reason.
    what : str
        A description.

    Raises
    ------
    Mismatch
        If it is admitted or refused for another reason.
    """
    try:
        attempt()
    except PlanRefused as refused:
        _require(refused.code is code, f"{what}: refused as {refused.code.value}")
        return
    except Exception as error:  # noqa: BLE001
        raise Mismatch(
            "refusal", math.inf, f"{what} raised {type(error).__name__}: {error}"
        ) from error
    raise Mismatch("refusal", math.inf, f"{what} was admitted")


def check_divergent() -> None:
    """Require a system whose least fixed point is infinite to be refused (O6).

    ``x = (3/5) x^2 + 3/5`` has no real root, and ``x = 2 x^2 + 1`` none
    either: both least fixed points are infinite, which a certified
    spectral radius at least one below them shows.

    Raises
    ------
    Mismatch
        If a number is returned.
    """
    for branch, stop in ((Fraction(3, 5), Fraction(3, 5)), (Fraction(2), Fraction(1))):
        weights: Inputs = {
            _BRANCH: torch.tensor(float(branch), dtype=torch.float64),
            _STOP: torch.tensor(float(stop), dtype=torch.float64),
        }
        _refused(
            lambda weights=weights: _plan(QUADRATIC, MARKOV, _contract()).run(
                OCCURRENCES, weights, like=LIKE
            ),
            PlanRefusal.DIVERGENT,
            f"x = {branch} x^2 + {stop}",
        )


def check_refusals() -> None:
    """Require refusals of what no plan here answers.

    An algebra without an embedding in the reals has no claim of it; an
    exact value of a monotone polynomial system has no finite method; a
    forest of its infinitely many derivations is not kept.

    Raises
    ------
    Mismatch
        If one is admitted or refused for another reason.
    """
    _refused(
        lambda: _plan(QUADRATIC, COUNTING, _contract()),
        PlanRefusal.INADMISSIBLE,
        "the counting algebra",
    )
    _refused(
        lambda: _plan(QUADRATIC, MARKOV, _contract(exact=True)),
        PlanRefusal.INADMISSIBLE,
        "an exact value",
    )
    forest = PackedForest()
    _refused(
        lambda: plan(
            _query(QUADRATIC).with_(retention=forest),
            QUADRATIC,
            MARKOV,
            MonotonePolynomial(),
            _contract(),
            forest,
        ),
        PlanRefusal.RECURSION_WITHOUT_EXECUTOR,
        "a forest",
    )
    _require(
        _plan(QUADRATIC, MARKOV, _contract()).record.algorithm == NEWTON_ALGORITHM,
        "another executor is selected",
    )


# -- work --------------------------------------------------------------------

_RNAME = "ring"
_RFAMILY = FamilyId.derive(_RNAME, "v")


def ring(size: int) -> tuple[RulePresentation, list[AxiomOccurrence], Inputs]:
    """``x_i = (1/4) x_(i+1) x_(i-1) + 1/2`` around a ring of ``size`` items.

    Parameters
    ----------
    size : int
        The items.

    Returns
    -------
    tuple
        The presentation, its inputs and weights.
    """
    sort = UserIndexSort(name="Ring", constructors=tuple(f"n{i}" for i in range(size)))

    def node(index: int) -> Judgment:
        """The item of a position.

        Parameters
        ----------
        index : int
            The position, modulo the size.

        Returns
        -------
        Judgment
            The pattern.
        """
        return Judgment(
            family=_RFAMILY,
            arguments=(IndexLiteral(value=f"n{index % size}", sort=sort),),
        )

    weight = ParameterId.derive(_RNAME, "pair")
    slots = tuple(
        AxiomSlot(
            id=ConstructorId.derive(_RNAME, f"s{i}"), name=f"s{i}", judgment=node(i)
        )
        for i in range(size)
    )
    presentation = RulePresentation(
        id=PresentationId.derive(_RNAME, str(size)),
        name=_RNAME,
        index_sorts=(sort,),
        judgments=(
            JudgmentDecl(
                id=_RFAMILY,
                name="v",
                indices=(IndexBinder(name="n", sort=sort, refinable=True),),
            ),
        ),
        rules=tuple(
            Rule(
                id=ConstructorId.derive(_RNAME, f"r{i}"),
                name=f"r{i}",
                premises=(
                    Premise(judgment=node(i + 1)),
                    Premise(judgment=node(i - 1)),
                ),
                conclusion=node(i),
                weight=weight,
            )
            for i in range(size)
        ),
        axioms=slots,
    )
    occurrences = [
        AxiomOccurrence(
            slot=slot.id, occurrence=OccurrenceId.derive(_RNAME, slot.name), binding=()
        )
        for slot in slots
    ]
    inputs: Inputs = {weight: torch.tensor(0.25, dtype=torch.float64)}
    inputs.update(
        {occ.occurrence: torch.tensor(0.5, dtype=torch.float64) for occ in occurrences}
    )
    return presentation, occurrences, inputs


def newton_step_work(size: int) -> float:
    """The measured time of one Newton step on a ring, averaged over the run.

    Parameters
    ----------
    size : int
        The items.

    Returns
    -------
    float
        The time counters over the steps taken.
    """
    presentation, occurrences, inputs = ring(size)
    current = _plan(presentation, MARKOV, _contract())
    evaluation = _evaluation(current.run(occurrences, inputs, like=LIKE), "a ring")
    report = evaluation.report()
    bound = next(
        bound
        for bound in current.record.complexity.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    return float(measured_total(bound, report)) / max(evaluation.iterations, 1)


def check_work() -> None:
    """Require one Newton step on a ring within ``S^3 + P^2 E`` operations.

    Raises
    ------
    Mismatch
        If the work breaks the bound, or the control of the next order
        does not.
    """
    scaling.check_family(
        scaling.Family(
            name="Newton steps on rings",
            executor="newton",
            question="Q9",
            varies="component size",
            part="forward",
            sizes=(2, 4, 8, 16, 32),
            work=newton_step_work,
            claim=lambda size: float(size**3 + 4 * 2 * size),
            constant=64.0,
        )
    )


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="newton", question="Q9", run=run)
    for name, run in (
        ("newton/closed-form", check_closed_form),
        ("newton/decomposed", check_decomposed),
        ("newton/gradient", check_gradient),
        ("newton/critical", check_critical),
        ("newton/budget", check_budget),
        ("newton/divergent", check_divergent),
        ("newton/refusals", check_refusals),
        ("newton/work", check_work),
    )
}
"""Every check of monotone polynomial plans by name."""

RECURSION = "quivers.stochastic.recursion"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="newton-upper-not-checked",
        defect_class="premature stop",
        provenance="the estimate is reported as its own upper bound, unchecked",
        target=f"{RECURSION}:_upper",
        rewrites=(
            Rewrite(
                old=(
                    "        candidate = [\n"
                    "            _up(here + width * own) for here, own in "
                    "zip(base, ones, strict=True)\n"
                    "        ]"
                ),
                new="        candidate = list(base)",
            ),
            Rewrite(
                old=(
                    "if all(image <= own for image, own in zip(images, candidate, "
                    "strict=True)):"
                ),
                new="if True:",
            ),
        ),
        rejected_by=("newton/closed-form",),
        min_effect=1.0,
    ),
    Mutation(
        name="newton-budget-reported-complete",
        defect_class="budget",
        provenance="a run the budget ended is reported as completed",
        target=f"{RECURSION}:solve_polynomial",
        also_in=("quivers.stochastic.planning",),
        rewrites=(Rewrite(old='stopped = "budget"', new='stopped = "completed"'),),
        rejected_by=("newton/budget",),
        min_effect=1.0,
    ),
    Mutation(
        name="newton-divergence-not-certified",
        defect_class="divergence",
        provenance="a spectral radius at least one below the solution is ignored",
        target=f"{RECURSION}:_diverges",
        rewrites=(
            Rewrite(
                old="if vector is not None and _expands(jacobian, vector, count):",
                new="if False:",
            ),
        ),
        rejected_by=("newton/divergent",),
        min_effect=1.0,
    ),
    Mutation(
        name="newton-implicit-without-inverse",
        defect_class="gradient path",
        provenance="the derivative of the fixed point is that of one evaluation",
        target=f"{RECURSION}:_implicit",
        rewrites=(
            Rewrite(
                old="values = fixed + inverse @ (image - image.detach())",
                new="values = fixed + (image - image.detach())",
            ),
        ),
        rejected_by=("newton/gradient",),
        min_effect=0.33,
    ),
)
"""Every wrong version of monotone polynomial plans and the checks that reject it."""
