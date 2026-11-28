"""Checked settlement, cyclic plans, and bounded forests have distinct values."""

from __future__ import annotations

from math import comb, log
from pathlib import Path

import pytest
import torch

from quivers.core.algebras import LOG_PROB, DivergentClosureError, UndecidedClosureError
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.contracts import (
    NOT_DIFFERENTIATED,
    SPECTRAL_RADIUS_BELOW_ONE,
    AbsoluteLogError,
    AccuracyContract,
    Bounded,
    Certified,
    ContractNotMet,
    ControlReached,
    DerivativeSemantics,
    Exact,
    FailurePolicy,
    Forest,
    Heuristic,
    InterpretationId,
    LinearCyclic,
    MonotonePolynomial,
    NoRetention,
    Numerics,
    OccurrenceId,
    PackedForest,
    QuerySpec,
    Rational,
    ResourceBudget,
    Target,
    Unestablished,
    Weight,
)
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_deduction
from quivers.qiec.rules import AxiomOccurrence, Item, RulePresentation
from quivers.qiec.serialization import dumps, loads
from quivers.stochastic.forest import edge_weight, forest_folds
from quivers.stochastic.planning import (
    NEWTON_ALGORITHM,
    SPAN_ALGORITHM,
    PlanRefusal,
    PlanRefused,
    plan,
)


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs/examples/qiec/checked-cyclic-deduction.qvr"
).read_text()


@pytest.fixture(scope="module")
def cyclic_source():
    compiled = load_source(SOURCE)
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec
    return compiled.deductions, module


def _logs(base: float, step: float) -> torch.Tensor:
    return torch.tensor([base, step], dtype=torch.float64).log().requires_grad_()


def _iterate(weights: torch.Tensor, arity: int, rounds: int) -> torch.Tensor:
    base, step = weights.exp().unbind()
    value = torch.zeros_like(base)
    for _ in range(rounds):
        value = base + step * value**arity
    return value.log()


def _parameters(name: str, weights: torch.Tensor) -> dict[str, torch.Tensor]:
    return {f"{name}.lex.0": weights[0], f"{name}.rule.step:": weights[1]}


def _plan_inputs(presentation: RulePresentation, weights: torch.Tensor):
    (rule,) = presentation.rules
    assert rule.weight is not None
    (slot,) = presentation.axioms
    occurrence = OccurrenceId.derive(presentation.id, "seed", 0)
    goal = Item(family=slot.judgment.family, arguments=(0, 1, "S"))
    (left, right, category) = slot.telescope
    axiom = AxiomOccurrence(
        slot=slot.id,
        occurrence=occurrence,
        binding=((left.name, 0), (right.name, 1), (category.name, "S")),
        values=("seed",),
    )
    wrt = (occurrence, rule.weight)
    query = QuerySpec(
        presentation=presentation.id,
        interpretation=InterpretationId.derive(presentation.id, "weight"),
        observable=Weight(),
        differentiate_with_respect_to=wrt,
    )
    return query, goal, (axiom,), dict(zip(wrt, weights.unbind(), strict=True))


def _accuracy(guarantee: Exact | Certified) -> AccuracyContract:
    return AccuracyContract(
        numerics=Numerics(dtype="float64"),
        target=Target(quantity="least inside mass", metric=AbsoluteLogError()),
        guarantee=guarantee,
        gradient=DerivativeSemantics.implicit(SPECTRAL_RADIUS_BELOW_ONE),
    )


@pytest.fixture(scope="module")
def parallel_span():
    source = SOURCE.replace(
        "    lexicon",
        "    rule second : span(I, J, S) |- span(I, J, S) #[learnable]\n    lexicon",
        1,
    )
    presentation = load_source(source).deductions["Linear"].presentation
    seed = torch.empty(2, dtype=torch.float64)
    query, goal, axioms, _ = _plan_inputs(
        presentation.with_(rules=presentation.rules[:1]), seed
    )
    parameters = tuple(
        weight for rule in presentation.rules if (weight := rule.weight) is not None
    )
    assert len(parameters) == 2
    slots = (axioms[0].occurrence, *parameters)
    selected = plan(
        query.with_(differentiate_with_respect_to=slots),
        presentation,
        LOG_PROB,
        LinearCyclic(),
        _accuracy(Exact()),
        NoRetention(),
    )
    assert selected.span is not None and selected.record.algorithm == SPAN_ALGORITHM
    return selected, goal, axioms, slots


def test_parallel_span_star_differentiates_each_original_occurrence(
    parallel_span,
) -> None:
    selected, goal, axioms, slots = parallel_span
    weights = (
        torch.tensor([0.3, 0.16, 0.24], dtype=torch.float64).log().requires_grad_()
    )
    run = selected.run(
        axioms, dict(zip(slots, weights.unbind(), strict=True)), like=weights
    )
    value = run.values[goal]
    a, first, second = weights.exp().unbind()
    # There are 2^h distinct unary-rule strings of length h. For each k
    # uses of the first rule, there are h choose k such strings.
    oracle = (
        a
        * sum(
            comb(h, k) * first**k * second ** (h - k)
            for h in range(30)
            for k in range(h + 1)
        )
    ).log()
    torch.testing.assert_close(value, oracle)
    assert not torch.isclose(value, (a / (1 - first)).log())
    gradient = torch.autograd.grad(value, weights)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(oracle, weights)[0])
    torch.testing.assert_close(
        gradient, torch.tensor([1, 0.16 / 0.6, 0.24 / 0.6], dtype=weights.dtype)
    )
    assert run.pivots == len(selected.span.categories) + 1
    assert run.allocated >= 2 * run.pivots**2


@pytest.mark.parametrize(
    ("loops", "cause"),
    [
        ((-log(2), -log(2)), UndecidedClosureError),
        ((log(0.55), log(0.5)), DivergentClosureError),
    ],
)
def test_parallel_span_star_certifies_original_coefficient_domain(
    parallel_span, loops, cause
):
    selected, _, axioms, slots = parallel_span
    weights = torch.tensor([log(0.3), *loops], dtype=torch.float64)
    with pytest.raises(PlanRefused) as refused:
        selected.run(
            axioms, dict(zip(slots, weights.unbind(), strict=True)), like=weights
        )
    assert isinstance(refused.value.__cause__, cause)


def test_cyclic_span_does_not_accept_an_unenforced_compute_budget(
    parallel_span,
) -> None:
    selected, _, _, _ = parallel_span
    accuracy = selected.record.accuracy.with_(
        budget=ResourceBudget(states=len(selected.span.categories))
    )
    with pytest.raises(PlanRefused, match="does not enforce a general resource budget"):
        plan(
            selected.record.query,
            selected.presentation,
            LOG_PROB,
            LinearCyclic(),
            accuracy,
            NoRetention(),
        )


@pytest.mark.parametrize(
    ("name", "arity", "rounds"), (("Linear", 1, 3), ("Branching", 2, 2))
)
def test_checked_settlement_refuses_and_exposes_only_a_finite_iterate(
    cyclic_source, name: str, arity: int, rounds: int
) -> None:
    _, module = cyclic_source
    weights = _logs(0.3, 0.4)
    parameters = _parameters(name, weights)
    with pytest.raises(ContractNotMet) as refused:
        run_deduction(module, name, tokens=("seed",), parameters=parameters)
    held = run_deduction(
        module,
        name,
        tokens=("seed",),
        parameters=parameters,
        on_failure=FailurePolicy.RETURN_UNESTABLISHED,
    ).weight
    assert isinstance(held, Unestablished)
    assert held.report == refused.value.report
    assert isinstance(held.report.attainment.guarantee, Heuristic)
    assert held.report.attainment.derivative == NOT_DIFFERENTIATED
    assert isinstance(held.report.stop, ControlReached)
    assert held.report.stop.control == "tolerance"
    expected = _iterate(weights, arity, rounds)
    difference = float((expected - _iterate(weights, arity, rounds - 1)).detach())
    assert float(held.report.stop.value.fraction) == pytest.approx(difference)
    assert difference < 0.15
    torch.testing.assert_close(held.value, expected)
    torch.testing.assert_close(
        torch.autograd.grad(held.value, weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )


def test_linear_span_closure_and_explicit_height_cap_mean_different_sums(
    cyclic_source,
) -> None:
    systems, module = cyclic_source
    presentation = systems["Linear"].presentation
    assert presentation.analysis.recursion() == LinearCyclic()
    weights = _logs(0.3, 0.4)
    query, goal, axioms, inputs = _plan_inputs(presentation, weights)
    selected = plan(
        query, presentation, LOG_PROB, LinearCyclic(), _accuracy(Exact()), NoRetention()
    )
    assert selected.record.algorithm == SPAN_ALGORITHM
    result = selected.run(axioms, inputs, like=weights[0])
    assert selected.span is not None and selected.span.unary_cyclic
    assert result.accuracy is None and result.pivots == len(selected.span.categories)
    base, step = weights.exp().unbind()
    expected = (base / (1.0 - step)).log()
    torch.testing.assert_close(result.values[goal], expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.values[goal], weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )

    retention = PackedForest()
    bounded = plan(
        query.with_(
            observable=Forest(), retention=retention, differentiate_with_respect_to=()
        ),
        presentation,
        LOG_PROB,
        Bounded(measure="height", limit=2),
        _accuracy(Exact()).with_(gradient=NOT_DIFFERENTIATED),
        retention,
    )
    assert bounded.height == 2
    forest = bounded.forest(axioms)
    assert not forest.acyclic and forest.count(goal, height=bounded.height) == 2
    value = forest_folds(
        forest,
        (goal,),
        LOG_PROB,
        lambda edge: edge_weight(forest, edge, LOG_PROB, inputs, like=weights[0]),
        like=weights[0],
        height=bounded.height,
    )[goal]
    finite = _iterate(weights, 1, 2)
    torch.testing.assert_close(value, finite)
    torch.testing.assert_close(
        torch.autograd.grad(value, weights)[0],
        torch.autograd.grad(finite, weights)[0],
    )
    assert not torch.isclose(value, result.values[goal])

    divergent = _logs(0.3, 1.0)
    _, _, _, infinite_inputs = _plan_inputs(presentation, divergent)
    with pytest.raises(PlanRefused) as invalid:
        selected.run(axioms, infinite_inputs, like=divergent[0])
    assert invalid.value.code is PlanRefusal.DIVERGENT
    assert isinstance(invalid.value.__cause__, DivergentClosureError)
    held = run_deduction(
        module,
        "Linear",
        tokens=("seed",),
        parameters=_parameters("Linear", divergent),
        on_failure=FailurePolicy.RETURN_UNESTABLISHED,
    ).weight
    assert isinstance(held, Unestablished)
    torch.testing.assert_close(held.value, _iterate(divergent, 1, 8))


def test_nonlinear_plan_encloses_the_least_root_and_certifies_its_derivative(
    cyclic_source,
) -> None:
    systems, module = cyclic_source
    presentation = systems["Branching"].presentation
    assert presentation.analysis.recursion() == MonotonePolynomial()
    weights = _logs(0.3, 0.4)
    query, goal, axioms, inputs = _plan_inputs(presentation, weights)
    contract = _accuracy(Certified(bound=Rational.of(1e-8)))
    selected = plan(
        query, presentation, LOG_PROB, MonotonePolynomial(), contract, NoRetention()
    )
    assert selected.record.algorithm == NEWTON_ALGORITHM
    solved = selected.run(axioms, inputs, like=weights[0])
    base, step = weights.exp().unbind()
    root = 2.0 * base / (1.0 + (1.0 - 4.0 * base * step).sqrt())
    assert solved.lower[goal] <= float(root.detach()) <= solved.upper[goal]
    assert solved.accuracy is not None
    assert isinstance(solved.accuracy.attainment.guarantee, Certified)
    assert solved.accuracy.attainment.derivative.implicit_conditions == (
        SPECTRAL_RADIUS_BELOW_ONE,
    )
    torch.testing.assert_close(solved.values[goal], root.log(), atol=1e-8, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(solved.values[goal], weights)[0],
        torch.autograd.grad(root.log(), weights)[0],
    )
    with pytest.raises(PlanRefused) as nonlinear:
        plan(
            query,
            presentation,
            LOG_PROB,
            LinearCyclic(),
            _accuracy(Exact()),
            NoRetention(),
        ).run(axioms, inputs, like=weights[0])
    assert nonlinear.value.code is PlanRefusal.NONLINEAR_CYCLE

    critical = _logs(0.5, 0.5)
    _, _, _, critical_inputs = _plan_inputs(presentation, critical)
    with pytest.raises(ContractNotMet) as refused:
        selected.run(axioms, critical_inputs, like=critical[0])
    derivative = refused.value.report.attainment.derivative
    assert not derivative.implicit_conditions
    assert any("spectral radius" in reason for reason in derivative.missing)
    held = run_deduction(
        module,
        "Branching",
        tokens=("seed",),
        parameters=_parameters("Branching", critical),
        on_failure=FailurePolicy.RETURN_UNESTABLISHED,
    ).weight
    assert isinstance(held, Unestablished) and isinstance(
        held.report.stop, ControlReached
    )
    torch.testing.assert_close(held.value, _iterate(critical, 2, 3))
    assert (
        float(held.report.stop.value.fraction) < 0.15 < abs(float(held.value.detach()))
    )


def test_checked_lexical_proof_queries_require_a_finite_span_order() -> None:
    source = (
        SOURCE.replace("S) #[learnable]", "S)")
        + """
object Result : Real 1
program invalid : Token -> Result
    let rows = parse_proofs(Linear, sentence, 2)
    let zero = 0.0
    return zero
"""
    )
    with pytest.raises(QiecDiagnosticError, match="cycl") as refused:
        load_source(source)
    assert refused.value.code == "qiec-deduction-outcomes"
