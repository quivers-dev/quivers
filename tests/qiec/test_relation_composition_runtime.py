"""Sparse checked composition requests preserve exact epsilon path counts."""

from math import log

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV, Semiring
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RelationTopology,
)
from quivers.qiec import (
    ComposedRelationSpec,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedComposedRelationRequest,
    FixedRelationRuntimeProvider,
    ParameterId,
    RationalRelationSpec,
    RuntimeConfiguration,
    dumps,
    loads,
    run_named,
)
from tests.dsl.test_authored_relation_topologies import _compile
from tests.qiec.test_relation_runtime import _module

_IN, _MID, _OUT = (
    StringSet(alphabet=FinSet(name=name, cardinality=2))
    for name in ("Input", "Middle", "Output")
)
_WEIGHT = ParameterId.derive("sparse-noisy-channel", "shared")
_ONE = ParameterId.derive("sparse-noisy-channel", "one")
_OTHER = ParameterId.derive("sparse-noisy-channel", "other")


def _operands(
    algebra: Semiring, *, cycle: bool = False
) -> tuple[RationalRelationSpec, RationalRelationSpec]:
    left_arcs = (
        Arc(0, 1, (0, None), 0),
        Arc(0, 1, (0, None), 0),
        Arc(1, 2, (None, 0), 1),
    )
    if cycle:
        left_arcs = (*left_arcs, Arc(0, 0, (None, None), 0))
    right_arcs = (
        Arc(0, 1, (None, 0), 0),
        Arc(0, 1, (None, 0), 0),
        Arc(1, 2, (0, None), 1),
    )

    def spec(tapes, arcs):
        return RationalRelationSpec.of(
            RelationTopology(
                tapes,
                1,
                3,
                arcs,
                (Boundary(0, 1),),
                (Boundary(2, 1),),
                ("shared", "one"),
            ),
            algebra,
            (_WEIGHT, _ONE),
        )

    return spec((_IN, _MID), left_arcs), spec((_MID, _OUT), right_arcs)


def _values(algebra: Semiring) -> dict[ParameterId, torch.Tensor]:
    logs = algebra is LOG_PROB
    return {
        _WEIGHT: nn.Parameter(
            torch.tensor(log(0.25) if logs else 0.25, dtype=torch.float64)
        ),
        _ONE: torch.tensor(0.0 if logs else 1.0, dtype=torch.float64),
    }


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
@pytest.mark.parametrize("execution", ["lazy", "materialized"])
def test_serialized_composition_preserves_shared_gradient_and_no_epsilon_duplication(
    algebra, execution
):
    left, right = _operands(algebra)
    spec = ComposedRelationSpec(left=left, right=right)
    request = FixedComposedRelationRequest(
        relation=spec,
        words=((0,), (0,)),
        max_states=32,
        max_work=64,
        execution=execution,
    )
    wire = loads(dumps(request))
    assert isinstance(wire, FixedComposedRelationRequest) and wire == request
    assert wire.relation.left.arcs[:2] == (wire.relation.left.arcs[0],) * 2
    chosen = _values(algebra)
    provider = FixedRelationRuntimeProvider(wire, chosen, handler_name="channel")
    type_name = "LogWeight" if algebra is LOG_PROB else "Real"
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    result = run_named(_module(type_name), "evaluate", (None,), runtime=runtime)
    assert isinstance(result.value, torch.Tensor)
    expected = 0.25
    score = result.value.exp() if algebra is LOG_PROB else result.value
    assert score.item() == pytest.approx(expected)
    (gradient,) = torch.autograd.grad(result.value, chosen[_WEIGHT])
    assert gradient.item() == pytest.approx(2)


def test_cyclic_lazy_composition_refuses_and_explicit_materialization_closes():
    left, right = _operands(MARKOV, cycle=True)
    request = FixedComposedRelationRequest(
        relation=ComposedRelationSpec(left=left, right=right),
        words=((0,), (0,)),
        max_states=128,
        max_work=256,
    )
    values = _values(MARKOV)
    with pytest.raises(ProductiveEpsilonCycleError):
        request.plan(values)
    with pytest.raises(dx.ValidationError, match="requires materialized execution"):
        request.with_(max_closure_size=16)
    closed = request.with_(execution="materialized", max_closure_size=16)
    assert closed.plan(values).weight().item() == pytest.approx(1 / 3)


def test_composition_checks_nominal_tape_and_cross_operand_weight_identity():
    left, right = _operands(MARKOV)
    with pytest.raises(dx.ValidationError, match="nominal shared tape"):
        ComposedRelationSpec(
            left=left, right=right.with_(tapes=(("Other", 2), ("Output", 2)))
        )
    values = _values(MARKOV)
    spec = ComposedRelationSpec(
        left=left, right=right.with_(weight_parameters=(_OTHER, _ONE))
    )
    assert isinstance(values[_WEIGHT], nn.Parameter)
    with pytest.raises(ValueError, match="same parameter identity"):
        spec.bind(values | {_OTHER: values[_WEIGHT]})


def test_source_pair_score_accepts_composition_when_its_topology_is_unbound():
    _, module = _compile()
    left, right = _operands(MARKOV)
    request = FixedComposedRelationRequest(
        relation=ComposedRelationSpec(left=left, right=right),
        words=((0,), (0,)),
        max_states=32,
        max_work=64,
    )
    values = _values(MARKOV)
    provider = FixedRelationRuntimeProvider.for_score("unbound", request, values)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), provider)
    )
    outcome = run_named(module, "unbound", runtime=runtime)
    assert isinstance(outcome.value, torch.Tensor)
    assert outcome.value.item() == pytest.approx(0.25)

    rejected = FixedRelationRuntimeProvider.for_score("pair", request, values)
    runtime = RuntimeConfiguration(
        selections=(), providers=(CoreRuntimeProvider(), rejected)
    )
    with pytest.raises(ExecutionFailure, match="authored topology"):
        run_named(module, "pair", runtime=runtime)
