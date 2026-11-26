"""Checks and deliberately wrong versions of linear cyclic plans.

A two-node cycle ``0 -> 1 -> 0`` has values in closed form: with start
weight ``s``, edge weights ``p`` and ``q`` and step weight ``w``,
``reach(0) = s / (1 - p q w^2)`` and ``reach(1) = reach(0) p w``. The
checks compare plans with these in the algebras that claim a least
closure, with the expected uses of each weight for gradients, with the
two-pass run on acyclic graphs, and with matrix products in closed form
on both sides of a product that does not commute. Algebras without a
closure are refused, as are divergent cycles, nonlinear components and
components multiplied on both sides of such a product.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from quivers.core.algebras import (
    BOOLEAN,
    COUNTING,
    LOG_PROB,
    MARKOV,
    MAX_PLUS,
    REAL,
    MatrixSemiring,
    OrderPreservingSemiring,
    Star,
)
from quivers.qiec.contracts import (
    Acyclic,
    LinearCyclic,
    NoRetention,
    OccurrenceId,
    ParameterId,
    Phase,
    PresentationId,
    measured_total,
)
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.rules import Judgment, Premise, Rule, RulePresentation
from quivers.qiec.types import IndexVariable
from quivers.stochastic.planning import Evaluation, Plan, PlanRefusal, PlanRefused, plan
from tests.evidence import graphs, scaling
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.mutations import Mutation, Rewrite

LIKE = torch.zeros((), dtype=torch.float64)
"""The dtype and device of scalar runs."""

type Inputs = dict[ParameterId | OccurrenceId, torch.Tensor]
"""Input weights."""

CYCLE = [(0, 1), (1, 0)]
"""The two-node cycle."""

START = OccurrenceId.derive(graphs.NAME, "start")
STEP = ParameterId.derive(graphs.NAME, "step")
S, P, Q, W = 0.9, 0.5, 0.6, 0.8
"""Start, edge and step weights, with ``p q w^2 < 1``."""


class StarMatrices(MatrixSemiring, Star):
    """Square matrices over a semiring with a closure, with the closure of block matrices.

    An ``n x n`` matrix of ``k x k`` matrices is an ``nk x nk`` matrix
    over the base, and block multiplication is its multiplication, so the
    two semirings are isomorphic and the base's least closure is one here.
    """

    @property
    def star_domain(self) -> str:
        """The block matrices whose flattened matrix the base closes.

        Returns
        -------
        str
            The domain.
        """
        base = self.base
        assert isinstance(base, Star)
        return f"block matrices whose flattened matrix lies in: {base.star_domain}"

    def star(self, matrix: torch.Tensor) -> torch.Tensor:
        """The closure, by the base's closure of the flattened matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``(n, n, k, k)``.

        Returns
        -------
        torch.Tensor
            The closure, of the same shape.
        """
        base = self.base
        assert isinstance(base, Star)
        n, k = matrix.shape[0], self.size
        flat = matrix.permute(0, 2, 1, 3).reshape(n * k, n * k)
        closed = base.star(flat)
        return closed.reshape(n, k, n, k).permute(0, 2, 1, 3)


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


def _plan(
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    recursion: LinearCyclic | Acyclic | None = None,
) -> Plan:
    """A plan of a reachability presentation.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The algebra.
    recursion : LinearCyclic or Acyclic or None
        The recursion; linear cyclic by default.

    Returns
    -------
    Plan
        The plan.
    """
    return plan(
        graphs.query().with_(presentation=presentation.id),
        presentation,
        algebra,
        LinearCyclic() if recursion is None else recursion,
        graphs.contract(),
        NoRetention(),
    )


def _inputs(transform: Callable[[float], float]) -> Inputs:
    """The cycle's weights under a transform.

    Parameters
    ----------
    transform : Callable[[float], float]
        Applied to each weight (``math.log`` for log weights).

    Returns
    -------
    Inputs
        The weights.
    """
    inputs: Inputs = {
        START: torch.tensor(transform(S), dtype=torch.float64),
        STEP: torch.tensor(transform(W), dtype=torch.float64),
    }
    inputs.update(
        {
            key: torch.tensor(transform(value), dtype=torch.float64)
            for key, value in zip(
                (graphs.edge_id(0), graphs.edge_id(1)), (P, Q), strict=True
            )
        }
    )
    return inputs


def _run(
    presentation: RulePresentation,
    algebra: OrderPreservingSemiring,
    inputs: Inputs,
    like: torch.Tensor = LIKE,
) -> Evaluation:
    """Run the cycle.

    Parameters
    ----------
    presentation : RulePresentation
        The presentation.
    algebra : OrderPreservingSemiring
        The algebra.
    inputs : Inputs
        The weights.
    like : torch.Tensor
        The dtype and device.

    Returns
    -------
    Evaluation
        The values.
    """
    return _plan(presentation, algebra).run(
        graphs.occurrences(CYCLE), inputs, like=like
    )


def check_closed_form() -> None:
    """Require the least solution in every algebra that claims a closure.

    Raises
    ------
    Mismatch
        If a value differs from its closed form.
    """
    loop = 1 - P * Q * W * W
    first, second = S / loop, S * P * W / loop
    accuracy = Accuracy(domain="linear", operations=64)
    real = _run(graphs.LEFT, MARKOV, _inputs(lambda x: x))
    compare_number(
        float(real.value(graphs.node(0))), first, accuracy, "markov reach(0)"
    )
    compare_number(
        float(real.value(graphs.node(1))), second, accuracy, "markov reach(1)"
    )
    log = _run(graphs.LEFT, LOG_PROB, _inputs(math.log))
    log_accuracy = Accuracy(domain="log", operations=64)
    compare_number(
        float(log.value(graphs.node(0))), math.log(first), log_accuracy, "log reach(0)"
    )
    compare_number(
        float(log.value(graphs.node(1))), math.log(second), log_accuracy, "log reach(1)"
    )
    best = _run(graphs.LEFT, MAX_PLUS, _inputs(math.log))
    compare_number(
        float(best.value(graphs.node(0))), math.log(S), log_accuracy, "best reach(0)"
    )
    compare_number(
        float(best.value(graphs.node(1))),
        math.log(S * P * W),
        log_accuracy,
        "best reach(1)",
    )
    boolean = _plan(graphs.LEFT, BOOLEAN).run(
        graphs.occurrences(CYCLE), {}, like=torch.zeros((), dtype=torch.bool)
    )
    _require(bool(boolean.value(graphs.node(1))), "reach(1) is not reachable")
    _require(
        any(part.side == "left" for part in log.components),
        "the weighted step was not solved on the left",
    )
    mixed = _run(graphs.MIXED, MARKOV, _inputs(lambda x: x))
    both = 1 - P * Q * (W + 1) ** 2
    compare_number(
        float(mixed.value(graphs.node(0))), S / both, accuracy, "both steps, reach(0)"
    )
    _require(
        any(part.side == "commuted" for part in mixed.components),
        "a commutative product was not folded to one side",
    )


def check_gradient() -> None:
    """Require the gradient of ``log reach(0)`` to be the expected uses of each weight.

    Each derivation of ``reach(0)`` uses ``p``, ``q`` and ``w`` twice per
    turn of the cycle, so the expected uses are ``r / (1 - r)`` for ``p``
    and ``q`` and twice that for ``w``, with ``r = p q w^2``; the start is
    used once.

    Raises
    ------
    Mismatch
        If a derivative differs.
    """
    current = _plan(graphs.LEFT, LOG_PROB)
    gradient = current.gradient(
        graphs.occurrences(CYCLE), _inputs(math.log), graphs.node(0), like=LIKE
    )
    ratio = P * Q * W * W
    turns = ratio / (1 - ratio)
    accuracy = Accuracy(domain="linear", operations=128)
    compare_number(float(gradient[START]), 1.0, accuracy, "d / d log s")
    compare_number(float(gradient[graphs.edge_id(0)]), turns, accuracy, "d / d log p")
    compare_number(float(gradient[graphs.edge_id(1)]), turns, accuracy, "d / d log q")
    compare_number(float(gradient[STEP]), 2 * turns, accuracy, "d / d log w")


def check_acyclic_agrees() -> None:
    """Require the cyclic executor to give the two-pass values on acyclic graphs.

    Raises
    ------
    Mismatch
        If a value differs.
    """
    edges = graphs.skips(6)
    inputs: Inputs = {
        **graphs.edge_weights(edges, [0.3 + 0.05 * i for i in range(len(edges))]),
        STEP: torch.tensor(0.9, dtype=torch.float64),
    }
    cyclic = _plan(graphs.LEFT, MARKOV).run(
        graphs.occurrences(edges), inputs, like=LIKE
    )
    acyclic = _plan(graphs.LEFT, MARKOV, Acyclic()).run(
        graphs.occurrences(edges), inputs, like=LIKE
    )
    for node in range(7):
        compare_number(
            float(cyclic.value(graphs.node(node))),
            float(acyclic.value(graphs.node(node))),
            Accuracy(domain="linear", operations=64),
            f"reach({node})",
        )


_A = torch.tensor([[0.2, 0.1], [0.0, 0.3]], dtype=torch.float64)
_B = torch.tensor([[0.1, 0.0], [0.4, 0.2]], dtype=torch.float64)
_R = torch.tensor([[0.9, 0.2], [0.1, 0.5]], dtype=torch.float64)
_S = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
"""Edge, step and start matrices whose products do not commute."""


def _matrix_inputs(*, weighted: bool) -> Inputs:
    """The cycle's matrices.

    Parameters
    ----------
    weighted : bool
        Whether the step rule has a matrix weight.

    Returns
    -------
    Inputs
        The weights.
    """
    inputs: Inputs = {START: _S, graphs.edge_id(0): _A, graphs.edge_id(1): _B}
    if weighted:
        inputs[STEP] = _R
    return inputs


def check_one_sided() -> None:
    """Require both one-sided forms over matrices that do not commute.

    Left: ``reach(j) = reach(i) E_ij R``, so ``reach(0) = S (I - A R B R)^{-1}``
    and ``reach(1) = reach(0) A R``. Right: ``reach(j) = E_ij reach(i)``, so
    ``reach(0) = (I - B A)^{-1} S`` and ``reach(1) = A reach(0)``.

    Raises
    ------
    Mismatch
        If a value differs from its closed form, or a side is not recorded.
    """
    algebra = StarMatrices(MARKOV, 2)
    like = torch.zeros((2, 2), dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    left = _run(graphs.LEFT, algebra, _matrix_inputs(weighted=True), like)
    first = _S @ torch.linalg.inv(eye - _A @ _R @ _B @ _R)
    expected = {0: first, 1: first @ _A @ _R}
    right = _run(graphs.RIGHT, algebra, _matrix_inputs(weighted=False), like)
    first_right = torch.linalg.inv(eye - _B @ _A) @ _S
    expected_right = {0: first_right, 1: _A @ first_right}
    for name, evaluation, values, side in (
        ("left", left, expected, "left"),
        ("right", right, expected_right, "right"),
    ):
        for node, value in values.items():
            deviation = float((evaluation.value(graphs.node(node)) - value).abs().max())
            if deviation > 1e-12:
                raise Mismatch("value", deviation, f"{name}: reach({node})")
        _require(
            any(part.side == side for part in evaluation.components),
            f"the {name} cycle was not solved on the {side}",
        )


_DOUBLE = Rule(
    id=ConstructorId.derive(graphs.NAME, "double"),
    name="double",
    telescope=(IndexBinder(name="i", sort=NAT), IndexBinder(name="j", sort=NAT)),
    premises=(
        Premise(
            judgment=Judgment(
                family=graphs.REACH, arguments=(IndexVariable(name="i", sort=NAT),)
            )
        ),
        Premise(
            judgment=Judgment(
                family=graphs.REACH, arguments=(IndexVariable(name="i", sort=NAT),)
            )
        ),
        Premise(
            judgment=Judgment(
                family=graphs.EDGE,
                arguments=(
                    IndexVariable(name="i", sort=NAT),
                    IndexVariable(name="j", sort=NAT),
                ),
            )
        ),
    ),
    conclusion=Judgment(
        family=graphs.REACH, arguments=(IndexVariable(name="j", sort=NAT),)
    ),
)
"""``reach(j) <- reach(i), reach(i), edge(i, j)``: quadratic on a cycle."""

NONLINEAR = graphs.PRESENTATION.with_(
    id=PresentationId.derive(graphs.NAME, "nonlinear"), rules=(_DOUBLE,)
)
"""A presentation whose cycle is not linear."""


def _refused(
    attempt: Callable[[], Evaluation | Plan], code: PlanRefusal, what: str
) -> str:
    """Require an attempt to be refused for a reason.

    Parameters
    ----------
    attempt : Callable
        The attempt.
    code : PlanRefusal
        The reason.
    what : str
        A description.

    Returns
    -------
    str
        The refusal's reason.

    Raises
    ------
    Mismatch
        If it is admitted or refused for another reason.
    """
    try:
        attempt()
    except PlanRefused as refused:
        _require(refused.code is code, f"{what}: refused as {refused.code.value}")
        return refused.reason
    raise Mismatch("refusal", math.inf, f"{what} was admitted")


def check_refusals() -> None:
    """Require refusals of what has no least solution here.

    An algebra without a closure is inadmissible; a counted cycle and a
    probability cycle of weight one or more diverge; a quadratic cycle is
    not linear; a cycle multiplied on both sides of matrices names both
    rules.

    Raises
    ------
    Mismatch
        If one is admitted or refused for another reason.
    """
    _refused(lambda: _plan(graphs.LEFT, REAL), PlanRefusal.INADMISSIBLE, "the reals")
    _refused(
        lambda: _plan(graphs.LEFT, COUNTING).run(
            graphs.occurrences(CYCLE), {}, like=LIKE
        ),
        PlanRefusal.DIVERGENT,
        "a counted cycle",
    )
    heavy = _inputs(lambda x: x)
    heavy[STEP] = torch.tensor(2.0, dtype=torch.float64)
    _refused(
        lambda: _run(graphs.LEFT, MARKOV, heavy),
        PlanRefusal.DIVERGENT,
        "a cycle of weight above one",
    )
    _refused(
        lambda: _run(NONLINEAR, MARKOV, _inputs(lambda x: x)),
        PlanRefusal.NONLINEAR_CYCLE,
        "a quadratic cycle",
    )
    reason = _refused(
        lambda: _run(
            graphs.MIXED,
            StarMatrices(MARKOV, 2),
            _matrix_inputs(weighted=True),
            torch.zeros((2, 2), dtype=torch.float64),
        ),
        PlanRefusal.MIXED_SIDES,
        "both sides of matrices",
    )
    _require(
        graphs.STEP.name in reason and graphs.STEP_RIGHT.name in reason,
        f"the refusal does not name both rules: {reason}",
    )


def _ring(size: int) -> list[tuple[int, int]]:
    """A ring ``0 -> 1 -> ... -> size - 1 -> 0``: one component of ``size`` items.

    Parameters
    ----------
    size : int
        The number of nodes.

    Returns
    -------
    list of tuple of int and int
        The edges.
    """
    return [(i, (i + 1) % size) for i in range(size)]


def _closure_work(size: int) -> float:
    """The measured time of solving a ring.

    Parameters
    ----------
    size : int
        The number of nodes.

    Returns
    -------
    float
        The time counters' total.
    """
    edges = _ring(size)
    inputs: Inputs = {**graphs.edge_weights(edges, [0.5] * len(edges))}
    current = _plan(graphs.LEFT, MARKOV)
    report = current.run(graphs.occurrences(edges), inputs, like=LIKE).report()
    time = next(
        bound
        for bound in current.record.complexity.bounds
        if bound.phase is Phase.EXECUTION and bound.resource == "time"
    )
    return float(measured_total(time, report))


def check_closure_work() -> None:
    """Require the measured closure work within the ``S^3`` claim on rings.

    A ring of ``n`` nodes is one component of ``n`` items, whose closure
    eliminates ``n`` states with ``n^2`` operations each; the control of
    the next order has to fail.

    Raises
    ------
    Mismatch
        If the work breaks the bound, or the control does not.
    """
    scaling.check_family(
        scaling.Family(
            name="cyclic closure on rings",
            executor="cyclic",
            question="Q8",
            varies="component size",
            part="forward",
            sizes=(4, 8, 16, 32, 64),
            work=_closure_work,
            claim=lambda size: float(size**3 + 3 * size),
            constant=4.0,
        )
    )


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="cyclic", question="Q8", run=run)
    for name, run in (
        ("cyclic/closed-form", check_closed_form),
        ("cyclic/gradient", check_gradient),
        ("cyclic/acyclic-agrees", check_acyclic_agrees),
        ("cyclic/one-sided", check_one_sided),
        ("cyclic/refusals", check_refusals),
        ("cyclic/closure-work", check_closure_work),
    )
}
"""Every check of linear cyclic plans by name."""

RECURSION = "quivers.stochastic.recursion"

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="cyclic-right-side-transposed",
        defect_class="order",
        provenance="a right-linear component is solved as b M* instead of M* b",
        target=f"{RECURSION}:solve",
        also_in=("quivers.stochastic.planning",),
        rewrites=(
            Rewrite(
                old="term = algebra.tensor_op(closure[i, j], constant[j])",
                new="term = algebra.tensor_op(constant[j], closure[i, j])",
            ),
        ),
        rejected_by=("cyclic/one-sided",),
        min_effect=1e-6,
    ),
    Mutation(
        name="cyclic-sides-folded-without-commuting",
        defect_class="order",
        provenance="a component multiplied on both sides is folded to one side",
        target=f"{RECURSION}:_side",
        rewrites=(Rewrite(old="if commutes:", new="if True:"),),
        rejected_by=("cyclic/refusals",),
        min_effect=1.0,
    ),
    Mutation(
        name="cyclic-nonlinear-not-checked",
        defect_class="recursion",
        provenance="a component with two premises inside is solved as if linear",
        target=f"{RECURSION}:solve",
        also_in=("quivers.stochastic.planning",),
        rewrites=(
            Rewrite(old="if len(places) > 1:", new="if False:"),
            Rewrite(old="(k,) = places", new="k = places[0]"),
        ),
        rejected_by=("cyclic/refusals",),
        min_effect=1.0,
    ),
    Mutation(
        name="cyclic-closure-truncated",
        defect_class="premature stop",
        provenance="the closure is replaced by the matrix itself, no paths of other lengths",
        target=f"{RECURSION}:solve",
        also_in=("quivers.stochastic.planning",),
        rewrites=(
            Rewrite(
                old="closure = _closure(algebra, filled, like, work)",
                new="closure = filled",
            ),
        ),
        rejected_by=("cyclic/closed-form",),
        min_effect=1e-6,
    ),
)
"""Every wrong version of linear cyclic plans and the checks that reject it."""
