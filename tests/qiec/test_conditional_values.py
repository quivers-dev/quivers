"""Pure control flow keeps inactive partial expressions unevaluated."""

import pytest

from quivers.qiec import (
    BOOL,
    INT,
    REAL,
    ConditionalValue,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    Return,
    dumps,
    loads,
    infer_value,
)
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.primitives import primitive
from quivers.qiec.terms import PrimitiveApplication
from quivers.qiec.evaluator import EvaluationError


ORIGIN = SourceOrigin(
    module="conditional", structural_path=("test",), role="test", source_protocol="test"
)


def _log(value):
    signature = primitive("log_probability")
    return PrimitiveApplication(
        primitive=signature.id,
        name=signature.name,
        arguments=(LiteralValue(value=value, type=REAL),),
        result_type=REAL,
        origin=ORIGIN,
    )


@pytest.mark.parametrize("selected", [False, True])
def test_only_selected_branch_is_evaluated_and_roundtrip_preserves_laziness(selected):
    valid, invalid = _log(0.5), _log(-0.5)
    term = ConditionalValue(
        condition=LiteralValue(value=selected, type=BOOL),
        then=valid if selected else invalid,
        otherwise=invalid if selected else valid,
        result_type=REAL,
    )
    assert infer_value(term, KernelRegistry()) == REAL
    restored = loads(dumps(Return(value=term)))
    assert restored == Return(value=term)
    assert Evaluator().evaluate(restored) == pytest.approx(-0.6931471805599453)


def test_selected_invalid_branch_keeps_its_validation_error():
    term = ConditionalValue(
        condition=LiteralValue(value=True, type=BOOL),
        then=_log(-0.5),
        otherwise=_log(0.5),
        result_type=REAL,
    )
    with pytest.raises(EvaluationError, match="log_probability"):
        Evaluator().evaluate(Return(value=term))


@pytest.mark.parametrize(
    "condition,otherwise,result",
    [
        (LiteralValue(value=1, type=INT), LiteralValue(value=1.0, type=REAL), REAL),
        (LiteralValue(value=True, type=BOOL), LiteralValue(value=1, type=INT), REAL),
        (LiteralValue(value=True, type=BOOL), LiteralValue(value=1.0, type=REAL), INT),
    ],
)
def test_condition_and_both_branch_types_are_checked(condition, otherwise, result):
    term = ConditionalValue(
        condition=condition,
        then=LiteralValue(value=0.0, type=REAL),
        otherwise=otherwise,
        result_type=result,
    )
    with pytest.raises(KernelError, match="conditional value"):
        infer_value(term, KernelRegistry())
